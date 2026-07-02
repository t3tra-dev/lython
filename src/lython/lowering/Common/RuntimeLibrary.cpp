#include "Common/RuntimeLibrary.h"

#include "embedded.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/SourceMgr.h"

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
  auto validPrelinkedRuntime = [](llvm::StringRef path) {
    auto buffer = llvm::MemoryBuffer::getFile(path);
    if (!buffer)
      return false;
    llvm::StringRef contents = (*buffer)->getBuffer();
    return !contents.starts_with("; Runtime prelowering is disabled while "
                                 "PyDialect lowering is rebuilt.");
  };
  if (auto env = llvm::sys::Process::GetEnv("LYTHON_RUNTIME_PRELINKED_IR")) {
    if (env->empty())
      return std::nullopt;
    if (llvm::sys::fs::exists(*env) && validPrelinkedRuntime(*env))
      return *env;
    return std::nullopt;
  }
#if defined(LYTHON_RUNTIME_PRELINKED_IR)
  if (llvm::sys::fs::exists(LYTHON_RUNTIME_PRELINKED_IR) &&
      validPrelinkedRuntime(LYTHON_RUNTIME_PRELINKED_IR))
    return std::string(LYTHON_RUNTIME_PRELINKED_IR);
#endif
  return std::nullopt;
}

namespace {

constexpr llvm::StringLiteral kRuntimeContractsAttr{"ly.runtime.contracts"};

bool declarationsOnly() { return prelinkedRuntimeIRPath().has_value(); }

bool shouldImportSymbol(mlir::Operation &op) {
  if (!mlir::isa<mlir::SymbolOpInterface>(op))
    return false;
  if (!declarationsOnly())
    return true;
  return mlir::isa<mlir::FunctionOpInterface>(op);
}

bool isFunctionDeclaration(mlir::Operation &op) {
  if (auto function = mlir::dyn_cast<mlir::func::FuncOp>(op))
    return function.getBody().empty();
  if (!mlir::isa<mlir::FunctionOpInterface>(op))
    return false;
  return op.getNumRegions() == 0 || op.getRegion(0).empty();
}

void importSymbol(mlir::ModuleOp target, mlir::Operation &op) {
  auto symbol = mlir::cast<mlir::SymbolOpInterface>(op);
  if (mlir::Operation *existing = target.lookupSymbol(symbol.getName())) {
    if (isFunctionDeclaration(op))
      return;
    existing->erase();
  }

  mlir::OpBuilder builder(target.getContext());
  builder.setInsertionPointToEnd(target.getBody());
  mlir::Operation *cloned =
      declarationsOnly() ? builder.cloneWithoutRegions(op) : builder.clone(op);
  if (auto clonedSymbol = mlir::dyn_cast<mlir::SymbolOpInterface>(cloned)) {
    if (!prelowerGenerationMode())
      clonedSymbol.setVisibility(mlir::SymbolTable::Visibility::Private);
  }
}

mlir::LogicalResult mergeRuntimeContracts(mlir::ModuleOp target,
                                          mlir::ModuleOp source) {
  auto sourceContracts =
      source->getAttrOfType<mlir::ArrayAttr>(kRuntimeContractsAttr);
  if (!sourceContracts)
    return mlir::success();

  llvm::StringSet<> seen;
  llvm::SmallVector<mlir::Attribute, 16> merged;
  auto append = [&](mlir::ArrayAttr contracts,
                    mlir::Operation *diagnosticTarget) -> mlir::LogicalResult {
    if (!contracts)
      return mlir::success();
    for (mlir::Attribute attr : contracts) {
      auto contract = mlir::dyn_cast<mlir::StringAttr>(attr);
      if (!contract)
        return diagnosticTarget->emitError()
               << kRuntimeContractsAttr << " entries must be strings";
      if (seen.insert(contract.getValue()).second)
        merged.push_back(contract);
    }
    return mlir::success();
  };

  if (mlir::failed(
          append(target->getAttrOfType<mlir::ArrayAttr>(kRuntimeContractsAttr),
                 target)))
    return mlir::failure();
  if (mlir::failed(append(sourceContracts, source)))
    return mlir::failure();

  target->setAttr(kRuntimeContractsAttr,
                  mlir::ArrayAttr::get(target.getContext(), merged));
  return mlir::success();
}

mlir::LogicalResult importRuntimeModule(mlir::ModuleOp target,
                                        const embedded::Module &entry) {
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

  if (mlir::failed(mergeRuntimeContracts(target, *source)))
    return mlir::failure();

  for (mlir::Operation &op : source->getBody()->getOperations())
    if (shouldImportSymbol(op))
      importSymbol(target, op);
  return mlir::success();
}

} // namespace

mlir::LogicalResult embedObjectModules(mlir::ModuleOp module) {
  for (std::size_t index = 0; index < embedded::moduleCount(); ++index) {
    const embedded::Module &entry = embedded::modules()[index];
    if (entry.kind != embedded::ModuleKind::MLIRBytecode)
      continue;
    if (llvm::StringRef(entry.name) == "typing")
      continue;
    if (mlir::failed(importRuntimeModule(module, entry)))
      return mlir::failure();
  }
  return mlir::success();
}

} // namespace py::runtime_library
