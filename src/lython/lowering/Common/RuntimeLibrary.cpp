#include "Common/RuntimeLibrary.h"

#include "Common/RuntimeSupportBuilder.h"
#include "embedded.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
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
#include "llvm/Linker/Linker.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"

#include "PyDialectTypes.h"
#define GET_OP_CLASSES
#include "PyOps.h.inc"

#include <memory>

namespace py::runtime_library {

namespace embedded {
namespace {
const Module *g_extraModules = nullptr;
std::size_t g_extraModuleCount = 0;
} // namespace

void registerExtraModules(const Module *extra, std::size_t count) {
  g_extraModules = extra;
  g_extraModuleCount = count;
}
const Module *extraModules() { return g_extraModules; }
std::size_t extraModuleCount() { return g_extraModuleCount; }
} // namespace embedded

namespace {

constexpr llvm::StringLiteral kContractsAttr{"ly.runtime.contracts"};

bool shouldImportSymbol(mlir::Operation &op) {
  return mlir::isa<mlir::SymbolOpInterface>(op);
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
  mlir::Operation *cloned = builder.clone(op);
  if (auto clonedSymbol = mlir::dyn_cast<mlir::SymbolOpInterface>(cloned)) {
    clonedSymbol.setVisibility(mlir::SymbolTable::Visibility::Private);
  }
}

mlir::LogicalResult mergeContracts(mlir::ModuleOp target,
                                   mlir::ModuleOp source) {
  auto sourceContracts = source->getAttrOfType<mlir::ArrayAttr>(kContractsAttr);
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
               << kContractsAttr << " entries must be strings";
      if (seen.insert(contract.getValue()).second)
        merged.push_back(contract);
    }
    return mlir::success();
  };

  if (mlir::failed(append(
          target->getAttrOfType<mlir::ArrayAttr>(kContractsAttr), target)))
    return mlir::failure();
  if (mlir::failed(append(sourceContracts, source)))
    return mlir::failure();

  target->setAttr(kContractsAttr,
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

  if (mlir::failed(mergeContracts(target, *source)))
    return mlir::failure();

  for (mlir::Operation &op : source->getBody()->getOperations()) {
    // py.class contracts are typing metadata and transform libraries belong
    // to the interpreter, not the user module -- modules/<name>.mlir
    // co-locates contracts, strategies, and implementation in one file, so
    // the import must filter rather than take the module wholesale.
    if (mlir::isa<py::ClassOp>(op))
      continue;
    if (op.hasAttr("transform.with_named_sequence") ||
        op.getDialect()->getNamespace() == "transform")
      continue;
    if (shouldImportSymbol(op))
      importSymbol(target, op);
  }
  return mlir::success();
}

} // namespace

mlir::LogicalResult embedObjectModules(mlir::ModuleOp module) {
  for (std::size_t index = 0; index < embedded::moduleCount(); ++index) {
    const embedded::Module &entry = embedded::modules()[index];
    if (entry.kind != embedded::ModuleKind::MLIRBytecode)
      continue;
    if (mlir::failed(importRuntimeModule(module, entry)))
      return mlir::failure();
  }
  return mlir::success();
}

// Layer 4 of the transformation stack:
// applies the lowering strategies carried by the embedded module manifests:
// each modules/<name>.mlir may nest a strategy-library module marked
// `transform.with_named_sequence` whose `__lython_strategy_*` named sequences
// are interpreted against the user module (the sequence's single argument
// handle binds the user module root).
mlir::LogicalResult applyEmbeddedLoweringStrategies(mlir::ModuleOp module) {
  for (std::size_t index = 0; index < embedded::moduleCount(); ++index) {
    const embedded::Module &entry = embedded::modules()[index];
    if (entry.kind != embedded::ModuleKind::MLIRBytecode)
      continue;
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(
        llvm::MemoryBuffer::getMemBuffer(
            llvm::StringRef(reinterpret_cast<const char *>(entry.data),
                            entry.size),
            entry.name, /*RequiresNullTerminator=*/false),
        llvm::SMLoc());
    auto source =
        mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, module.getContext());
    if (!source)
      continue; // importRuntimeModule already diagnosed parse failures
    for (mlir::Operation &op : source->getBody()->getOperations()) {
      auto strategyModule = mlir::dyn_cast<mlir::ModuleOp>(op);
      if (!strategyModule ||
          !strategyModule->hasAttr("transform.with_named_sequence"))
        continue;
      for (mlir::Operation &inner : strategyModule.getBody()->getOperations()) {
        auto sequence = mlir::dyn_cast<mlir::transform::NamedSequenceOp>(inner);
        if (!sequence ||
            !sequence.getSymName().starts_with("__lython_strategy_"))
          continue;
        mlir::transform::TransformOptions options;
        if (mlir::failed(mlir::transform::applyTransformNamedSequence(
                module, sequence, strategyModule, options)))
          return module.emitError()
                 << "lowering strategy '" << sequence.getSymName()
                 << "' from manifest '" << entry.name << "' failed";
      }
    }
  }
  return mlir::success();
}

namespace {

// Platform-specific runtime modules carry an `_<os>` name suffix (the
// pre-lowered runtime-internal lib modules are compiled once per triple);
// only the module matching the final target triple links.
bool isPlatformNativeSupport(llvm::StringRef name) {
  return name.ends_with("_darwin") || name.ends_with("_linux") ||
         name.ends_with("_windows");
}

bool shouldLinkEmbeddedLLVMRuntimeModule(llvm::StringRef name,
                                         const llvm::Triple &targetTriple) {
  if (name.ends_with("_darwin"))
    return targetTriple.isOSDarwin();
  if (name.ends_with("_linux"))
    return targetTriple.isOSLinux();
  if (name.ends_with("_windows"))
    return targetTriple.isOSWindows();
  return true;
}

void registerNativeRuntimeDialects(mlir::DialectRegistry &registry) {
  registry.insert<mlir::arith::ArithDialect, mlir::cf::ControlFlowDialect,
                  mlir::func::FuncDialect, mlir::LLVM::LLVMDialect,
                  mlir::math::MathDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect, mlir::ub::UBDialect,
                  mlir::transform::TransformDialect>();
  mlir::registerConvertFuncToLLVMInterface(registry);
  mlir::registerConvertMemRefToLLVMInterface(registry);
  mlir::registerConvertMathToLLVMInterface(registry);
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
  pm.addPass(mlir::createSCFToControlFlowPass());
  pm.addPass(mlir::memref::createExpandStridedMetadataPass());
  pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(mlir::createConvertMathToLLVMPass());
  pm.addPass(mlir::createArithToLLVMConversionPass());
  pm.addPass(mlir::createConvertControlFlowToLLVMPass());
  pm.addPass(mlir::createUBToLLVMConversionPass());
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

  // Lowers a native-runtime MLIR module to LLVM and links it into llvmModule.
  auto lowerTranslateAndLink =
      [&](mlir::ModuleOp nativeModule,
          llvm::StringRef label) -> mlir::LogicalResult {
    if (mlir::failed(lowerNativeRuntimeModule(nativeModule))) {
      llvm::errs() << "error: failed to lower native runtime module '" << label
                   << "'\n";
      return mlir::failure();
    }
    std::unique_ptr<llvm::Module> runtime =
        mlir::translateModuleToLLVMIR(nativeModule, llvmModule.getContext());
    if (!runtime) {
      llvm::errs() << "error: failed to translate native runtime module '"
                   << label << "' to LLVM IR\n";
      return mlir::failure();
    }
    runtime->setDataLayout(llvmModule.getDataLayout());
    runtime->setTargetTriple(llvmModule.getTargetTriple());
    if (llvm::Linker::linkModules(llvmModule, std::move(runtime))) {
      llvm::errs() << "error: failed to link native runtime module '" << label
                   << "'\n";
      return mlir::failure();
    }
    return mlir::success();
  };

  auto linkEntry = [&](const embedded::Module &entry) -> mlir::LogicalResult {
    if (entry.kind != embedded::ModuleKind::NativeMLIRBytecode)
      return mlir::success();
    llvm::StringRef name(entry.name);
    if (isPlatformNativeSupport(name))
      sawPlatformNativeSupport = true;
    if (!shouldLinkEmbeddedLLVMRuntimeModule(name, targetTriple))
      return mlir::success();
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
    return lowerTranslateAndLink(*nativeModule, entry.name);
  };

  // Runtime support the compiler builds directly (the former hand-written
  // native/support.mlir, fully migrated).
  {
    mlir::OwningOpRef<mlir::ModuleOp> builtModule =
        buildNativeRuntimeSupportModule(context);
    if (!builtModule) {
      llvm::errs() << "error: failed to build native runtime support module\n";
      return mlir::failure();
    }
    if (mlir::failed(lowerTranslateAndLink(*builtModule, "runtime-support")))
      return mlir::failure();
  }

  for (std::size_t index = 0; index < embedded::moduleCount(); ++index)
    if (mlir::failed(linkEntry(embedded::modules()[index])))
      return mlir::failure();
  // Pre-lowered runtime/lib modules registered by the host binary (lyc).
  for (std::size_t index = 0; index < embedded::extraModuleCount(); ++index)
    if (mlir::failed(linkEntry(embedded::extraModules()[index])))
      return mlir::failure();

  if (sawPlatformNativeSupport && !linkedPlatformNativeSupport) {
    llvm::errs() << "error: no embedded native runtime support module matches "
                 << targetTriple.str() << "\n";
    return mlir::failure();
  }
  return mlir::success();
}

} // namespace py::runtime_library
