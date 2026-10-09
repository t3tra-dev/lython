#include "Common/RuntimeLibrary.h"
#include "Common/TypeObjects.h"

#include "Common/Instrumentation.h"

#include "Common/LibcPrototypes.h"

#include "Common/RuntimeSupport.h"
#include "Common/UnwindABI.h"
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
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"

#include "PyDialectTypes.h"
#define GET_OP_CLASSES
#include "PyOps.h.inc"

#include <cstdlib>
#include <memory>

namespace py::runtime_library {

namespace embedded {
namespace {
const Module *g_extraModules = nullptr;
std::size_t g_extraModuleCount = 0;
} // namespace

const PrecompiledNativeRuntime *g_precompiledNativeRuntime = nullptr;

void registerPrecompiledNativeRuntime(const PrecompiledNativeRuntime *runtime) {
  g_precompiledNativeRuntime = runtime;
}
const PrecompiledNativeRuntime *precompiledNativeRuntime() {
  return g_precompiledNativeRuntime;
}

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

// A symbol another manifest defines: a function without a body, or a global
// without a value. ⛔ Not only functions: the object manifests share their
// constant globals (objects/object.mlir's repr punctuation) by declaring
// them, and a declaration imported after the definition replaced it.
bool isSymbolDeclaration(mlir::Operation &op) {
  if (auto function = mlir::dyn_cast<mlir::func::FuncOp>(op))
    return function.getBody().empty();
  if (auto global = mlir::dyn_cast<mlir::memref::GlobalOp>(op))
    return global.isExternal();
  if (!mlir::isa<mlir::FunctionOpInterface>(op))
    return false;
  return op.getNumRegions() == 0 || op.getRegion(0).empty();
}

void importSymbol(mlir::ModuleOp target, mlir::Operation &op) {
  auto symbol = mlir::cast<mlir::SymbolOpInterface>(op);
  if (mlir::Operation *existing = target.lookupSymbol(symbol.getName())) {
    if (isSymbolDeclaration(op))
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

  // A manifest that names a module attribute is the program's only when the
  // program has it: `_js` reaches functions only a JavaScript host provides,
  // and its deallocator is kept by the release dispatch whether or not a
  // proxy is ever made.
  if (auto onlyWith =
          (*source)->getAttrOfType<mlir::StringAttr>("ly.runtime.only_with"))
    if (!target->hasAttr(onlyWith.getValue()))
      return mlir::success();

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
    llvm::StringRef bytes(reinterpret_cast<const char *>(entry.data),
                          entry.size);
    // ⛔ Not every manifest parsed again: one in forty carries a strategy
    // library, and a sequence is matched by its `__lython_strategy_` name,
    // which bytecode keeps verbatim in its string section -- a manifest
    // without that prefix has nothing to apply.
    if (!bytes.contains("__lython_strategy_"))
      continue;
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(
        llvm::MemoryBuffer::getMemBuffer(bytes, entry.name,
                                         /*RequiresNullTerminator=*/false),
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
// pre-lowered runtime-internal lib modules are compiled once per triple), with
// `32` appended for an ILP32 build: their ctypes structs are laid out at the
// pointer width they were lowered for. Only the module matching the final
// target triple links.
constexpr llvm::StringLiteral kPlatformSuffixes[] = {
    "_darwin", "_linux", "_linux32", "_windows", "_wasi32"};

bool isPlatformNativeSupport(llvm::StringRef name) {
  return llvm::any_of(kPlatformSuffixes, [&](llvm::StringRef suffix) {
    return name.ends_with(suffix);
  });
}

std::string platformSuffixFor(const llvm::Triple &triple) {
  std::string suffix;
  if (triple.isOSDarwin())
    suffix = "_darwin";
  else if (triple.isOSLinux())
    suffix = "_linux";
  else if (triple.isOSWindows())
    suffix = "_windows";
  else if (triple.isOSWASI())
    suffix = "_wasi";
  else
    return "";
  if (triple.isArch32Bit())
    suffix += "32";
  return suffix;
}

bool shouldLinkEmbeddedLLVMRuntimeModule(llvm::StringRef name,
                                         const llvm::Triple &targetTriple) {
  if (!isPlatformNativeSupport(name))
    return true;
  std::string suffix = platformSuffixFor(targetTriple);
  return !suffix.empty() && name.ends_with(suffix);
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
  // Aligned, as the program's own lowering is (LoweringPipeline.cpp,
  // SymbolTableConvertToLLVM): the runtime's objects go through the same
  // allocator and must not be padded either.
  {
    mlir::FinalizeMemRefToLLVMConversionPassOptions memrefOptions;
    memrefOptions.useAlignedAlloc = true;
    pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass(memrefOptions));
  }
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

// The ctypes symbols the runtime declares, carried from the build of the
// runtime module to the program's link.
constexpr llvm::StringLiteral kCtypesSymbolsMetadata{
    "lython.native_runtime.ctypes_symbols"};

mlir::LogicalResult buildNativeRuntimeModule(llvm::Module &llvmModule) {
  llvm::Triple targetTriple(llvmModule.getTargetTriple());
  bool sawPlatformNativeSupport = false;
  bool linkedPlatformNativeSupport = false;
  mlir::DialectRegistry registry;
  registerNativeRuntimeDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  // The ctypes symbols the embedded modules declare. Read before lowering,
  // where the declarations still say so (`ly.native.symbol`), and marked on
  // the linked module once every declaration of each has merged.
  llvm::SmallVector<std::string, 8> ctypesSymbols;

  // Lowers a native-runtime MLIR module to LLVM and links it into llvmModule.
  auto lowerTranslateAndLink =
      [&](mlir::ModuleOp nativeModule,
          llvm::StringRef label) -> mlir::LogicalResult {
    py::collectCtypesForeignSymbols(nativeModule, ctypesSymbols);
    // The type objects this module names, defined in it as every module
    // defines them (TypeObjects.h).
    mlir::FailureOr<py::type_objects::StaticClassWords> staticClassWords =
        py::type_objects::collectStaticClassWords(nativeModule);
    if (mlir::failed(staticClassWords) ||
        mlir::failed(py::type_objects::resolveManifestClassWords(nativeModule)))
      return mlir::failure();
    if (mlir::failed(lowerNativeRuntimeModule(nativeModule))) {
      llvm::errs() << "error: failed to lower native runtime module '" << label
                   << "'\n";
      return mlir::failure();
    }
    unsigned pointerBits = llvmModule.getDataLayout().getPointerSizeInBits();
    if (mlir::failed(py::type_objects::patchStaticClassWords(
            nativeModule, *staticClassWords, pointerBits)) ||
        mlir::failed(
            py::type_objects::defineDeclared(nativeModule, pointerBits))) {
      llvm::errs() << "error: failed to define the type objects of native "
                      "runtime module '"
                   << label << "'\n";
      return mlir::failure();
    }
    // ⛔ The layout goes on BEFORE translation, not after. Translating folds
    // a constant struct GEP into byte offsets with whatever layout the module
    // has then -- LLVM's default, 8-byte pointers -- and a `setDataLayout`
    // afterwards changes the label, not the offsets: on armv7 the runtime
    // wrote `g_current_parts` as 120 bytes of LP64 descriptors that every
    // reader read as 96.
    nativeModule->setAttr(mlir::LLVM::LLVMDialect::getDataLayoutAttrName(),
                          mlir::StringAttr::get(nativeModule.getContext(),
                                                llvmModule.getDataLayoutStr()));
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
        buildNativeRuntimeSupportModule(context, targetTriple);
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
  llvm::NamedMDNode *carried =
      llvmModule.getOrInsertNamedMetadata(kCtypesSymbolsMetadata);
  for (const std::string &symbol : ctypesSymbols)
    carried->addOperand(llvm::MDNode::get(
        llvmModule.getContext(),
        llvm::MDString::get(llvmModule.getContext(), symbol)));
  return mlir::success();
}

mlir::LogicalResult linkEmbeddedNativeRuntime(llvm::Module &llvmModule) {
  llvm::Triple targetTriple(llvmModule.getTargetTriple());
  std::unique_ptr<llvm::Module> runtime;
  // The build's, when it was made for exactly this target and layout.
  //
  // LYTHON_ABLATE_PRECOMPILED_RUNTIME=1 lowers the runtime here instead.
  // ⛔ Not left out: both arms link the same module built the same way, so
  // their IR must be identical byte for byte, and that is only evidence when
  // one binary produces both -- a separate build re-proves the build.
  static const bool ablated = [] {
    const char *value = std::getenv("LYTHON_ABLATE_PRECOMPILED_RUNTIME");
    return value && *value && llvm::StringRef(value) != "0";
  }();
  if (const embedded::PrecompiledNativeRuntime *precompiled =
          embedded::precompiledNativeRuntime();
      !ablated && precompiled &&
      llvmModule.getTargetTriple().str() == precompiled->triple &&
      llvmModule.getDataLayoutStr() == precompiled->dataLayout) {
    PerfScope perf("link-runtime.precompiled");
    llvm::Expected<std::unique_ptr<llvm::Module>> parsed =
        llvm::parseBitcodeFile(
            llvm::MemoryBufferRef(
                llvm::StringRef(
                    reinterpret_cast<const char *>(precompiled->data),
                    precompiled->size),
                "lython-native-runtime"),
            llvmModule.getContext());
    if (!parsed) {
      llvm::errs() << "error: the precompiled native runtime does not parse: "
                   << llvm::toString(parsed.takeError()) << "\n";
      return mlir::failure();
    }
    runtime = std::move(*parsed);
  } else {
    PerfScope perf("link-runtime.lowered");
    runtime = std::make_unique<llvm::Module>("lython-native-runtime",
                                             llvmModule.getContext());
    runtime->setTargetTriple(llvmModule.getTargetTriple());
    runtime->setDataLayout(llvmModule.getDataLayout());
    if (mlir::failed(buildNativeRuntimeModule(*runtime)))
      return mlir::failure();
  }

  llvm::SmallVector<std::string, 8> ctypesSymbols;
  if (llvm::NamedMDNode *carried =
          runtime->getNamedMetadata(kCtypesSymbolsMetadata)) {
    for (llvm::MDNode *entry : carried->operands())
      ctypesSymbols.push_back(
          llvm::cast<llvm::MDString>(entry->getOperand(0))->getString().str());
    runtime->eraseNamedMetadata(carried);
  }
  if (llvm::Linker::linkModules(llvmModule, std::move(runtime))) {
    llvm::errs() << "error: failed to link the native runtime\n";
    return mlir::failure();
  }

  py::branchLocalRaisesToTheirHandler(llvmModule);
  py::flushAssertionMessages(llvmModule);
  for (const std::string &symbol : ctypesSymbols)
    if (llvm::Function *function = llvmModule.getFunction(symbol))
      function->addFnAttr(kCtypesForeignSymbolAttr);
  if (py::runtime_library::framePointersEnableCompactUnwind(targetTriple))
    py::forceFramePointers(llvmModule);
  return mlir::success();
}

} // namespace py::runtime_library
