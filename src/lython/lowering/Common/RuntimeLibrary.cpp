#include "Common/RuntimeLibrary.h"

#include "Common/LoweringUtils.h"
#include "Common/Object.h"
#include "Common/RuntimeSupport.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <optional>
#include <string>
#include <system_error>
#include <vector>

namespace py::runtime_library {
namespace {

std::optional<std::string> sourceModuleDir() {
  if (auto env = llvm::sys::Process::GetEnv("LYTHON_RUNTIME_MLIR_DIR"))
    return *env;

#if defined(LYTHON_SOURCE_DIR)
  llvm::SmallString<256> dir(LYTHON_SOURCE_DIR);
  llvm::sys::path::append(dir, "src", "lython", "runtime", "objects");
  return dir.str().str();
#else
  return std::nullopt;
#endif
}

std::optional<std::string> bytecodeModuleDir() {
  if (auto env = llvm::sys::Process::GetEnv("LYTHON_RUNTIME_MLIR_BC_DIR"))
    return *env;

#if defined(LYTHON_RUNTIME_MLIR_BC_DIR)
  if (llvm::sys::fs::exists(LYTHON_RUNTIME_MLIR_BC_DIR))
    return std::string(LYTHON_RUNTIME_MLIR_BC_DIR);
#endif

  return std::nullopt;
}

bool isMlirSourceModule(llvm::StringRef path) {
  return llvm::sys::path::extension(path) == ".mlir";
}

bool isMlirBytecodeModule(llvm::StringRef path) {
  return llvm::sys::path::extension(path) == ".mlirbc";
}

bool isFallbackRuntimeModule(llvm::StringRef path) {
  llvm::StringRef ext = llvm::sys::path::extension(path);
  return ext == ".mlirbc" || ext == ".mlir";
}

using ModuleFilter = bool (*)(llvm::StringRef);

mlir::LogicalResult collectModuleFiles(llvm::StringRef dir, ModuleFilter filter,
                                       std::vector<std::string> &files) {
  if (!llvm::sys::fs::exists(dir)) {
    llvm::errs() << "error: runtime MLIR object directory does not exist: "
                 << dir << "\n";
    return mlir::failure();
  }

  std::error_code ec;
  for (llvm::sys::fs::directory_iterator it(dir, ec), end; it != end;
       it.increment(ec)) {
    if (ec)
      break;
    if (!llvm::sys::fs::is_regular_file(it->path()))
      continue;
    if (filter(it->path()))
      files.emplace_back(it->path());
  }
  if (ec) {
    llvm::errs() << "error: failed to enumerate runtime MLIR object modules in "
                 << dir << ": " << ec.message() << "\n";
    return mlir::failure();
  }

  std::sort(files.begin(), files.end());
  return mlir::success();
}

std::optional<std::string> bytecodePathForSource(llvm::StringRef sourcePath) {
  auto bytecodeDir = bytecodeModuleDir();
  if (!bytecodeDir)
    return std::nullopt;

  std::string bytecodeName = llvm::sys::path::stem(sourcePath).str();
  bytecodeName += ".mlirbc";
  llvm::SmallString<256> bytecodePath(*bytecodeDir);
  llvm::sys::path::append(bytecodePath, bytecodeName);
  if (!llvm::sys::fs::exists(bytecodePath))
    return std::nullopt;
  return bytecodePath.str().str();
}

mlir::LogicalResult
collectSourceBackedRuntimeModules(llvm::StringRef sourceDir,
                                  std::vector<std::string> &files) {
  std::vector<std::string> sources;
  if (mlir::failed(collectModuleFiles(sourceDir, isMlirSourceModule, sources)))
    return mlir::failure();

  for (const std::string &source : sources) {
    if (std::optional<std::string> bytecode = bytecodePathForSource(source)) {
      files.push_back(*bytecode);
      continue;
    }
    files.push_back(source);
  }
  return mlir::success();
}

mlir::LogicalResult collectRuntimeModules(std::vector<std::string> &files) {
  if (auto sourceDir = sourceModuleDir()) {
    if (llvm::sys::fs::exists(*sourceDir))
      return collectSourceBackedRuntimeModules(*sourceDir, files);
  }

  // Installed-tree fallback: without source manifests, trust the configured
  // runtime directory. In a build tree, source-backed collection above prevents
  // stale bytecode from reintroducing removed runtime symbols.
  if (auto bytecodeDir = bytecodeModuleDir()) {
    if (llvm::sys::fs::exists(*bytecodeDir))
      return collectModuleFiles(*bytecodeDir, isMlirBytecodeModule, files);
  }

  if (auto sourceDir = sourceModuleDir())
    if (llvm::sys::fs::exists(*sourceDir))
      return collectModuleFiles(*sourceDir, isFallbackRuntimeModule, files);

  llvm::errs() << "error: runtime MLIR object directory is not configured\n";
  return mlir::failure();
}

bool hasBody(mlir::Operation *op) {
  return op->getNumRegions() > 0 && !op->getRegion(0).empty();
}

void copyAttr(mlir::Operation *from, mlir::Operation *to,
              llvm::StringRef attrName) {
  if (!from || !to || to->hasAttr(attrName))
    return;
  if (mlir::Attribute attr = from->getAttr(attrName))
    to->setAttr(attrName, attr);
}

bool mergeableSymbolAttr(mlir::NamedAttribute attr) {
  llvm::StringRef name = attr.getName().getValue();
  return name != mlir::SymbolTable::getSymbolAttrName() &&
         name != "function_type" && name != "sym_visibility";
}

void mergeFunctionArgAttrs(mlir::Operation *existing,
                           mlir::Operation *incoming) {
  auto existingFunc =
      llvm::dyn_cast_or_null<mlir::FunctionOpInterface>(existing);
  auto incomingFunc =
      llvm::dyn_cast_or_null<mlir::FunctionOpInterface>(incoming);
  if (!existingFunc || !incomingFunc ||
      existingFunc.getNumArguments() != incomingFunc.getNumArguments())
    return;
  for (unsigned index = 0; index < incomingFunc.getNumArguments(); ++index) {
    for (mlir::NamedAttribute attr : incomingFunc.getArgAttrs(index))
      if (!existingFunc.getArgAttr(index, attr.getName()))
        existingFunc.setArgAttr(index, attr.getName(), attr.getValue());
  }
}

void mergeRuntimeContractAttrs(mlir::Operation *existing,
                               mlir::Operation *incoming) {
  if (!existing || !incoming)
    return;
  for (mlir::NamedAttribute attr : incoming->getAttrs()) {
    if (!mergeableSymbolAttr(attr) || existing->hasAttr(attr.getName()))
      continue;
    existing->setAttr(attr.getName(), attr.getValue());
  }
  mergeFunctionArgAttrs(existing, incoming);
}

void copyCallEffectAttrs(mlir::Operation *callee, mlir::Operation *call) {
  copyAttr(callee, call, OwnershipContractAttrs::kOwnedResults);
  copyAttr(callee, call, OwnershipContractAttrs::kBorrowedResults);
  if (call && (call->hasAttr(OwnershipContractAttrs::kAggregateRetain) ||
               call->hasAttr(OwnershipContractAttrs::kAggregateRelease)))
    return;
  copyAttr(callee, call, OwnershipContractAttrs::kRetainArgs);
  copyAttr(callee, call, OwnershipContractAttrs::kReleaseArgs);
  copyAttr(callee, call, OwnershipContractAttrs::kTransferArgs);
  copyAttr(callee, call, OwnershipContractAttrs::kObjectReleaseToZero);
}

void copyClassHelperAttrs(mlir::Operation *callee, mlir::Operation *call) {
  copyAttr(callee, call, ClassSafetyAttrs::kHelperKind);
  copyAttr(callee, call, ClassSafetyAttrs::kHelperClass);
  copyAttr(callee, call, ClassSafetyAttrs::kHelperFieldIndex);
  copyAttr(callee, call, ClassSafetyAttrs::kHelperFieldCount);
  copyAttr(callee, call, ClassSafetyAttrs::kHelperDirectRefcountFields);
  copyAttr(callee, call, ClassSafetyAttrs::kHelperContainerFields);
  copyAttr(callee, call, ClassSafetyAttrs::kHelperDirectRefcountFieldIndices);
  copyAttr(callee, call, ClassSafetyAttrs::kHelperContainerFieldIndices);
}

void copyPythonSemanticAttrs(mlir::Operation *callee, mlir::Operation *call) {
  copyAttr(callee, call, "nothrow");
  copyAttr(callee, call, "maythrow");
}

bool hasOwnedObjectHeaderResult(mlir::Operation *callee) {
  auto function = mlir::dyn_cast_or_null<mlir::FunctionOpInterface>(callee);
  if (!function)
    return false;
  for (int64_t index : lowering::attrs::i64Array(
           callee, OwnershipContractAttrs::kOwnedResults)) {
    if (index < 0 || static_cast<unsigned>(index) >= function.getNumResults())
      continue;
    if (object_abi::Header::isOwned(
            function.getResultTypes()[static_cast<unsigned>(index)]))
      return true;
  }
  return false;
}

void markCallHeaderProducer(mlir::Operation *call, mlir::Operation *callee) {
  if (!hasOwnedObjectHeaderResult(callee))
    return;
  mlir::Builder builder(call->getContext());
  call->setAttr(OwnershipContractAttrs::kObjectHeader, builder.getUnitAttr());
}

void materializeRuntimeContracts(mlir::ModuleOp module) {
  module.walk([&](mlir::func::CallOp call) {
    mlir::Operation *callee = module.lookupSymbol(call.getCallee());
    copyCallEffectAttrs(callee, call.getOperation());
    copyClassHelperAttrs(callee, call.getOperation());
    copyPythonSemanticAttrs(callee, call.getOperation());
    markCallHeaderProducer(call.getOperation(), callee);
  });
  module.walk([&](mlir::LLVM::CallOp call) {
    if (auto callee = call.getCallee()) {
      mlir::Operation *calleeOp = module.lookupSymbol(*callee);
      copyCallEffectAttrs(calleeOp, call.getOperation());
      copyClassHelperAttrs(calleeOp, call.getOperation());
      copyPythonSemanticAttrs(calleeOp, call.getOperation());
      markCallHeaderProducer(call.getOperation(), calleeOp);
    }
  });
  module.walk([&](mlir::LLVM::InvokeOp invoke) {
    if (auto callee = invoke.getCallee()) {
      mlir::Operation *calleeOp = module.lookupSymbol(*callee);
      copyCallEffectAttrs(calleeOp, invoke.getOperation());
      copyClassHelperAttrs(calleeOp, invoke.getOperation());
      copyPythonSemanticAttrs(calleeOp, invoke.getOperation());
      markCallHeaderProducer(invoke.getOperation(), calleeOp);
    }
  });
}

bool canReplaceDeclaration(mlir::Operation *existing,
                           mlir::Operation &incoming) {
  if (hasBody(existing) || !hasBody(&incoming))
    return false;
  if (auto existingFunc = llvm::dyn_cast<mlir::func::FuncOp>(existing)) {
    auto incomingFunc = llvm::dyn_cast<mlir::func::FuncOp>(&incoming);
    return static_cast<bool>(incomingFunc);
  }
  if (auto existingFunc = llvm::dyn_cast<mlir::LLVM::LLVMFuncOp>(existing)) {
    auto incomingFunc = llvm::dyn_cast<mlir::LLVM::LLVMFuncOp>(&incoming);
    if (incomingFunc)
      return true;
    return static_cast<bool>(llvm::dyn_cast<mlir::func::FuncOp>(&incoming));
  }
  return false;
}

void importRuntimeSymbol(mlir::ModuleOp target, mlir::Operation *op) {
  if (auto symbol = llvm::dyn_cast<mlir::SymbolOpInterface>(op)) {
    if (mlir::Operation *existing = target.lookupSymbol(symbol.getName())) {
      if (!canReplaceDeclaration(existing, *op)) {
        mergeRuntimeContractAttrs(existing, op);
        return;
      }
      existing->erase();
    }
  }

  mlir::OpBuilder builder(target.getContext());
  builder.setInsertionPointToEnd(target.getBody());
  builder.clone(*op);
}

mlir::LogicalResult importRuntimeModule(mlir::ModuleOp target,
                                        llvm::StringRef path) {
  auto source =
      mlir::parseSourceFile<mlir::ModuleOp>(path, target.getContext());
  if (!source) {
    target.emitError() << "failed to parse runtime MLIR object module: "
                       << path;
    return mlir::failure();
  }

  for (mlir::Operation &op : source->getBody()->getOperations())
    if (mlir::isa<mlir::SymbolOpInterface>(op))
      importRuntimeSymbol(target, &op);
  return mlir::success();
}

} // namespace

mlir::LogicalResult embedObjectModules(mlir::ModuleOp module) {
  std::vector<std::string> files;
  if (mlir::failed(collectRuntimeModules(files)))
    return mlir::failure();

  for (const std::string &file : files) {
    if (mlir::failed(importRuntimeModule(module, file)))
      return mlir::failure();
  }
  materializeRuntimeContracts(module);

  return mlir::success();
}

} // namespace py::runtime_library
