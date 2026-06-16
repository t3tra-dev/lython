#include "Common/RuntimeLibrary.h"

#include "Common/LoweringUtils.h"
#include "Common/Object.h"
#include "Common/RuntimeSupport.h"
#include "embedded.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/SourceMgr.h"

#include <optional>
#include <string>

namespace py::runtime_library {

bool prelowerGenerationMode() {
  static const bool enabled = [] {
    auto value = llvm::sys::Process::GetEnv("LYTHON_RUNTIME_PRELOWER");
    if (!value)
      return false;
    llvm::StringRef text(*value);
    return text == "1" || text.equals_insensitive("true") ||
           text.equals_insensitive("yes") || text.equals_insensitive("on");
  }();
  return enabled;
}

std::optional<std::string> prelinkedRuntimeIRPath() {
  if (prelowerGenerationMode())
    return std::nullopt;
  if (auto env = llvm::sys::Process::GetEnv("LYTHON_RUNTIME_PRELINKED_IR")) {
    if (env->empty())
      return std::nullopt; // explicit opt-out
    if (llvm::sys::fs::exists(*env))
      return *env;
    return std::nullopt;
  }
#if defined(LYTHON_RUNTIME_PRELINKED_IR)
  if (llvm::sys::fs::exists(LYTHON_RUNTIME_PRELINKED_IR))
    return std::string(LYTHON_RUNTIME_PRELINKED_IR);
#endif
  return std::nullopt;
}

namespace {

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
    mlir::Type resultType =
        function.getResultTypes()[static_cast<unsigned>(index)];
    if (object_abi::Header::isOwned(resultType) ||
        object_abi::exception_abi::Header::isOwned(resultType))
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

void importRuntimeSymbol(mlir::ModuleOp target, mlir::Operation *op,
                         bool declarationsOnly) {
  // With the pre-lowered runtime cache, function bodies stay out of the
  // per-compile module: only the signature and its materialized contracts
  // are needed by the lowering and the verifiers. Non-function symbols
  // (small-int cache globals etc.) are referenced exclusively by runtime
  // bodies and live in the cache.
  if (declarationsOnly && !mlir::isa<mlir::FunctionOpInterface>(op))
    return;

  if (auto symbol = llvm::dyn_cast<mlir::SymbolOpInterface>(op)) {
    if (mlir::Operation *existing = target.lookupSymbol(symbol.getName())) {
      if (declarationsOnly || !canReplaceDeclaration(existing, *op)) {
        mergeRuntimeContractAttrs(existing, op);
        return;
      }
      existing->erase();
    }
  }

  mlir::OpBuilder builder(target.getContext());
  builder.setInsertionPointToEnd(target.getBody());
  mlir::Operation *cloned =
      declarationsOnly ? builder.cloneWithoutRegions(*op) : builder.clone(*op);
  if (auto symbol = llvm::dyn_cast<mlir::SymbolOpInterface>(cloned)) {
    // The prelower generation run must keep the runtime bodies public so
    // SymbolDCE does not erase them from the otherwise empty module.
    if (!prelowerGenerationMode())
      symbol.setVisibility(mlir::SymbolTable::Visibility::Private);
  }
}

mlir::LogicalResult importRuntimeModule(mlir::ModuleOp target,
                                        const embedded::Module &entry,
                                        bool declarationsOnly) {
  // The bytecode was generated from source at build time and compiled into
  // this binary; it is the manifest, not a cache, so it cannot be stale.
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBuffer(
          llvm::StringRef(reinterpret_cast<const char *>(entry.data),
                          entry.size),
          entry.name, /*RequiresNullTerminator=*/false),
      llvm::SMLoc());
  auto source =
      mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, target.getContext());
  if (!source) {
    target.emitError() << "failed to parse embedded runtime MLIR module: "
                       << entry.name;
    return mlir::failure();
  }

  for (mlir::Operation &op : source->getBody()->getOperations())
    if (mlir::isa<mlir::SymbolOpInterface>(op))
      importRuntimeSymbol(target, &op, declarationsOnly);
  return mlir::success();
}

} // namespace

mlir::LogicalResult embedObjectModules(mlir::ModuleOp module) {
  const bool declarationsOnly = prelinkedRuntimeIRPath().has_value();
  for (std::size_t index = 0; index < embedded::moduleCount(); ++index) {
    const embedded::Module &entry = embedded::modules()[index];
    // The typing manifest is consumed by the frontend protocol oracle, not
    // by lowering.
    if (llvm::StringRef(entry.name) == "typing")
      continue;
    if (mlir::failed(importRuntimeModule(module, entry, declarationsOnly)))
      return mlir::failure();
  }
  materializeRuntimeContracts(module);

  return mlir::success();
}

} // namespace py::runtime_library
