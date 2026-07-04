#include "Common/RuntimeLibrary.h"

#include "embedded.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"

#include <memory>

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

bool isContractManifestModule(llvm::StringRef name) {
  return name == "typing" || name == "ctypes";
}

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
    if (isContractManifestModule(entry.name))
      continue;
    if (mlir::failed(importRuntimeModule(module, entry)))
      return mlir::failure();
  }
  return mlir::success();
}

mlir::LogicalResult linkPrelinkedRuntime(llvm::Module &llvmModule) {
  std::optional<std::string> path = prelinkedRuntimeIRPath();
  if (!path)
    return mlir::success();
  llvm::SMDiagnostic diagnostic;
  std::unique_ptr<llvm::Module> runtime =
      llvm::parseIRFile(*path, diagnostic, llvmModule.getContext());
  if (!runtime) {
    llvm::errs() << "error: failed to load pre-lowered runtime IR from '"
                 << *path << "': " << diagnostic.getMessage() << "\n";
    return mlir::failure();
  }
  runtime->setDataLayout(llvmModule.getDataLayout());
  runtime->setTargetTriple(llvmModule.getTargetTriple());
  if (llvm::Linker::linkModules(llvmModule, std::move(runtime),
                                llvm::Linker::Flags::LinkOnlyNeeded)) {
    llvm::errs() << "error: failed to link pre-lowered runtime IR\n";
    return mlir::failure();
  }
  return mlir::success();
}

namespace {

bool isPlatformNativeSupport(llvm::StringRef name) {
  return name == "support_darwin" || name == "support_linux" ||
         name == "support_windows";
}

bool shouldLinkEmbeddedLLVMRuntimeModule(llvm::StringRef name,
                                         const llvm::Triple &targetTriple) {
  if (name == "support_darwin")
    return targetTriple.isOSDarwin();
  if (name == "support_linux")
    return targetTriple.isOSLinux();
  if (name == "support_windows")
    return targetTriple.isOSWindows();
  return true;
}

void registerNativeRuntimeDialects(mlir::DialectRegistry &registry) {
  registry.insert<mlir::arith::ArithDialect, mlir::cf::ControlFlowDialect,
                  mlir::func::FuncDialect, mlir::LLVM::LLVMDialect,
                  mlir::memref::MemRefDialect, mlir::scf::SCFDialect>();
  mlir::registerConvertFuncToLLVMInterface(registry);
  mlir::registerConvertMemRefToLLVMInterface(registry);
  mlir::registerAllToLLVMIRTranslations(registry);
}

mlir::OwningOpRef<mlir::ModuleOp>
parseEmbeddedNativeRuntimeModule(const embedded::Module &entry,
                                 mlir::MLIRContext &context) {
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBuffer(
          llvm::StringRef(reinterpret_cast<const char *>(entry.data),
                          entry.size),
          entry.name, /*RequiresNullTerminator=*/false),
      llvm::SMLoc());
  return mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
}

mlir::LogicalResult lowerNativeRuntimeModule(mlir::ModuleOp module) {
  mlir::PassManager pm(module.getContext());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createConvertFuncToLLVMPass());
  pm.addPass(mlir::createConvertSCFToCFPass());
  pm.addPass(mlir::memref::createExpandStridedMetadataPass());
  pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(mlir::createArithToLLVMConversionPass());
  pm.addPass(mlir::createConvertControlFlowToLLVMPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  return pm.run(module);
}

} // namespace

mlir::LogicalResult linkEmbeddedNativeRuntime(llvm::Module &llvmModule) {
  llvm::Triple targetTriple(llvmModule.getTargetTriple());
  bool sawPlatformNativeSupport = false;
  bool linkedPlatformNativeSupport = false;
  mlir::DialectRegistry registry;
  registerNativeRuntimeDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  for (std::size_t index = 0; index < embedded::moduleCount(); ++index) {
    const embedded::Module &entry = embedded::modules()[index];
    if (entry.kind != embedded::ModuleKind::NativeMLIRBytecode)
      continue;
    llvm::StringRef name(entry.name);
    if (isPlatformNativeSupport(name))
      sawPlatformNativeSupport = true;
    if (!shouldLinkEmbeddedLLVMRuntimeModule(name, targetTriple))
      continue;
    if (isPlatformNativeSupport(name))
      linkedPlatformNativeSupport = true;

    mlir::OwningOpRef<mlir::ModuleOp> nativeModule =
        parseEmbeddedNativeRuntimeModule(entry, context);
    if (!nativeModule) {
      llvm::errs() << "error: failed to parse embedded native runtime MLIR "
                      "bytecode module '"
                   << entry.name << "'\n";
      return mlir::failure();
    }
    if (mlir::failed(lowerNativeRuntimeModule(*nativeModule))) {
      llvm::errs() << "error: failed to lower embedded native runtime MLIR "
                      "module '"
                   << entry.name << "'\n";
      return mlir::failure();
    }
    std::unique_ptr<llvm::Module> runtime =
        mlir::translateModuleToLLVMIR(*nativeModule, llvmModule.getContext());
    if (!runtime) {
      llvm::errs() << "error: failed to translate embedded native runtime MLIR "
                      "module '"
                   << entry.name << "' to LLVM IR\n";
      return mlir::failure();
    }
    runtime->setDataLayout(llvmModule.getDataLayout());
    runtime->setTargetTriple(llvmModule.getTargetTriple());
    if (llvm::Linker::linkModules(llvmModule, std::move(runtime))) {
      llvm::errs() << "error: failed to link embedded native runtime module '"
                   << entry.name << "'\n";
      return mlir::failure();
    }
  }

  if (sawPlatformNativeSupport && !linkedPlatformNativeSupport) {
    llvm::errs() << "error: no embedded native runtime support module matches "
                 << targetTriple.str() << "\n";
    return mlir::failure();
  }
  return mlir::success();
}

} // namespace py::runtime_library
