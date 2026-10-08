// Layer 3 of the transformation stack:
// the global lowering stages and their order. Per-op lowerings live in
// RuntimeBundleLowerer (layers 1-2, Passes/Runtime/); target-selected
// schedules ship as transform-dialect strategies in the module manifests
// (layer 4, applied at phase 8b).
#include "Common/LoweringPipeline.h"

#include "Common/Instrumentation.h"
#include "Common/RuntimeLibrary.h"
#include "Common/RuntimeSupport.h"
#include "Common/TypeObjects.h"
#include "Passes/Runtime/Arch/Arm/AppleAMX.h"
#include "Passes/Runtime/Arch/Arm/ArmSME.h"
#include "Passes/Runtime/Cleanup/Transforms.h"
#include "Passes/Runtime/Ctypes/CallbackThunks.h"
#include "Passes/Runtime/Js/Host.h"
#include "Passes/Runtime/Common/ContractRewrite.h"
#include "Passes/Runtime/Primitive/TensorParallel.h"
#include "runtime/Verification.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <utility>

using namespace mlir;

namespace py {
namespace {

std::string trimEnvToken(llvm::StringRef token) {
  token = token.trim();
  return token.str();
}

void dumpMLIRForPass(const IRDumpConfig &config, llvm::StringRef passName,
                     ModuleOp module) {
  if (!config.shouldDump(passName))
    return;
  llvm::errs() << "\n=== [LYTHON_IR_DUMP:" << passName << " MLIR] ===\n";
  module->print(llvm::errs());
  llvm::errs() << "\n";
}

template <typename Populate>
LogicalResult runLoweringPhase(llvm::StringRef name, ModuleOp module,
                               bool enableVerifier, Populate populate) {
  std::string phase = (llvm::Twine("lowering.") + name).str();
  PerfScope perf(phase);
  // A pattern inside a greedy rewrite can emit an error diagnostic without
  // failing its pass (the driver only skips the pattern), which would let a
  // known-broken op flow on and mis-execute silently. Count error
  // diagnostics and refuse a phase that emitted any while reporting success.
  unsigned emittedErrors = 0;
  ScopedDiagnosticHandler diagnosticGuard(
      module.getContext(), [&](Diagnostic &diag) {
        if (diag.getSeverity() == DiagnosticSeverity::Error)
          ++emittedErrors;
        return failure();
      });
  PassManager pm(module.getContext());
  pm.enableVerifier(enableVerifier);
  populate(pm);
  if (failed(pm.run(module))) {
    if (llvm::sys::Process::GetEnv("LYTHON_DUMP_ON_FAILURE")) {
      llvm::errs() << "\n=== [FAILED PHASE " << name << "] ===\n";
      mlir::OpPrintingFlags flags;
      flags.assumeVerified();
      // Each op's source location, to find the one a diagnostic names.
      if (llvm::sys::Process::GetEnv("LYTHON_DUMP_LOCS"))
        flags.enableDebugInfo(/*enable=*/true, /*prettyForm=*/true);
      module->print(llvm::errs(), flags);
    }
    return failure();
  }
  if (emittedErrors != 0) {
    module.emitError() << "lowering phase '" << name << "' emitted "
                       << emittedErrors
                       << " error(s) but reported success; refusing to "
                          "continue with potentially mis-lowered IR";
    return failure();
  }
  return success();
}

// The EH marker calls carry their try id as an OPERAND, and that is exactly
// what block merging promotes to a block argument: two arms of an `if` inside
// one `try` are identical apart from their operands, so they merge and the
// merged marker's id becomes a phi of the two arms' anchor ids. The final EH
// phase can only wire a call to a handler through a constant id, so it drops
// the call site, the raising call keeps only its traceback cleanup edge, and
// the exception escapes a `try` that CPython catches.
//
// Restating the id as an ATTRIBUTE makes the two markers non-identical
// OPERATIONS, and MLIR compares attribute dictionaries before it merges: the
// arms stay separate exactly when merging would lose the id, and still merge
// when the ids agree. Nothing reads this attribute.
//
// ⛔ NOT "use an EH-safe copy of the merging pass" (tried, for
// `memref::createExpandStridedMetadataPass` -- which does merge here): six
// upstream passes in the final phase run the greedy driver with its DEFAULT
// AGGRESSIVE region simplification, and replacing each one leaves the next
// release free to add a seventh. ⛔ NOT "resolve the phi in the EH phase"
// (also tried): the incoming ids DISAGREE by construction, because each arm
// has its own traceback anchor for its own line number.
void discriminateEHMarkers(ModuleOp module) {
  mlir::Builder builder(module.getContext());
  module.walk([&](mlir::func::CallOp call) {
    llvm::StringRef callee = call.getCallee();
    if (callee != "LyEH_TryCallSiteMarker" && callee != "LyEH_TryCatchMarker" &&
        callee != "LyEH_TryCatchAnchor")
      return;
    if (call.getNumOperands() != 1)
      return;
    mlir::IntegerAttr id;
    if (!mlir::matchPattern(call.getOperand(0), mlir::m_Constant(&id)))
      return;
    call->setAttr("ly.eh.marker_id",
                  builder.getI64IntegerAttr(id.getInt()));
  });
}

// Canonicalization for phases where the lowered EH skeleton exists (runtime
// lowering onward). Not the plain createCanonicalizerPass(): its default
// AGGRESSIVE region simplification merges structurally identical blocks by
// promoting differing operands to block arguments, and a merged pair of
// exception-handler entries carries its `LyEH_TryCatchMarker` id as a block
// argument. The final EH phase can only wire invokes to handlers with
// CONSTANT marker ids, and it erases the anchor edges that fed the merged
// block's arguments, so the merge leaves unwirable, phi-broken handlers.
//
// ⛔ AND NOT UPSTREAM'S `cf` PATTERNS AS A WHOLE: MLIR 23 added a fold that
// replaces a block argument every predecessor feeds the same value with that
// value (`simplifyUniformBlockArgs`, in `cf.br`'s canonicalize and in
// `SimplifyUniformBlockArguments` for `cf.cond_br` / `cf.switch`). These
// phases run after refcount insertion, which gave such an argument a token of
// its own and retained it on each edge; folded, the retains name the outer
// value -- still right at run time, and a use after release to the ownership
// verifier (`out = {}; for ...: out.setdefault(...)`). `cf.br` keeps the two
// simplifications it had before, written here, since its canonicalize has no
// name to disable it by.
namespace {

// A successor holding nothing but an unconditional branch elsewhere: its
// destination and the operands it would receive. Upstream's collapseBranch.
LogicalResult collapsePassThrough(Block *&successor, ValueRange &operands,
                                  SmallVectorImpl<Value> &storage) {
  if (std::next(successor->begin()) != successor->end())
    return failure();
  auto branch = dyn_cast<cf::BranchOp>(successor->getTerminator());
  if (!branch)
    return failure();
  for (BlockArgument argument : successor->getArguments())
    for (Operation *user : argument.getUsers())
      if (user != branch)
        return failure();
  Block *destination = branch.getDest();
  if (destination == successor)
    return failure();
  auto next = dyn_cast<cf::BranchOp>(destination->getTerminator());
  llvm::DenseSet<Block *> visited{successor, destination};
  while (next) {
    Block *nextDestination = next.getDest();
    if (!visited.insert(nextDestination).second)
      return failure();
    next = dyn_cast<cf::BranchOp>(nextDestination->getTerminator());
  }
  OperandRange forwarded = branch.getOperands();
  if (successor->args_empty()) {
    successor = destination;
    operands = forwarded;
    return success();
  }
  for (Value operand : forwarded) {
    auto argument = dyn_cast<BlockArgument>(operand);
    storage.push_back(argument && argument.getOwner() == successor
                          ? operands[argument.getArgNumber()]
                          : operand);
  }
  successor = destination;
  operands = storage;
  return success();
}

// `cf.br` merged into a single-predecessor destination, or short-circuited
// past a pass-through block.
LogicalResult canonicalizeBranch(cf::BranchOp op, PatternRewriter &rewriter) {
  Block *destination = op.getDest();
  Block *parent = op->getBlock();
  if (destination != parent &&
      llvm::hasSingleElement(destination->getPredecessors()) &&
      llvm::none_of(op.getOperands(), [&](Value operand) {
        auto argument = dyn_cast<BlockArgument>(operand);
        return argument && argument.getOwner() == destination;
      })) {
    SmallVector<Value> operands(op.getOperands());
    rewriter.eraseOp(op);
    rewriter.mergeBlocks(destination, parent, operands);
    return success();
  }
  ValueRange operands = op.getOperands();
  SmallVector<Value, 4> storage;
  if (destination == parent ||
      failed(collapsePassThrough(destination, operands, storage)))
    return failure();
  rewriter.replaceOpWithNewOp<cf::BranchOp>(op, destination, operands);
  return success();
}

struct EHSafeCanonicalizer
    : public PassWrapper<EHSafeCanonicalizer, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EHSafeCanonicalizer)

  StringRef getArgument() const final { return "lython-eh-safe-canonicalize"; }

  LogicalResult initialize(MLIRContext *context) override {
    RewritePatternSet set(context);
    for (Dialect *dialect : context->getLoadedDialects())
      dialect->getCanonicalizationPatterns(set);
    for (RegisteredOperationName op : context->getRegisteredOperations())
      if (op.getStringRef() != cf::BranchOp::getOperationName())
        op.getCanonicalizationPatterns(set, context);
    set.add(canonicalizeBranch);
    patterns = FrozenRewritePatternSet(
        std::move(set),
        /*disabledPatternLabels=*/
        {"(anonymous namespace)::SimplifyUniformBlockArguments"});
    return success();
  }

  void runOnOperation() override {
    GreedyRewriteConfig config;
    config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Normal);
    (void)applyPatternsGreedily(getOperation(), patterns, config);
  }

  FrozenRewritePatternSet patterns;
};

} // namespace

std::unique_ptr<Pass> createEHSafeCanonicalizerPass() {
  return std::make_unique<EHSafeCanonicalizer>();
}

namespace {

// MLIR's convert-to-llvm in its static mode -- the patterns every loaded
// dialect's ConvertToLLVMPatternInterface contributes, built once and
// applied in one partial conversion -- except that the func dialect's are
// handed a symbol table.
//
// ⛔ Not upstream's pass: its func patterns get none, so each call resolved
// its callee by scanning the module, which after the runtime import holds
// thousands of symbols -- calls x symbols, 1.3 s of a 128-case batch.
// ⛔ And not convert-func-to-llvm ahead of it, which does take a table: a
// second conversion orders the argument materializations differently and
// gives them the function's location instead of their first use's.
class SymbolTableConvertToLLVM
    : public PassWrapper<SymbolTableConvertToLLVM, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SymbolTableConvertToLLVM)

  StringRef getArgument() const final { return "lython-convert-to-llvm"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
    registerConvertToLLVMDependentDialectLoading(registry);
  }

  LogicalResult initialize(MLIRContext *context) override {
    symbolTables = std::make_shared<SymbolTableCollection>();
    target = std::make_shared<ConversionTarget>(*context);
    // `memref.alloc` lowers to `aligned_alloc(alignment, size)`, which the
    // object allocator answers with an already-aligned block.
    // ⛔ Not the default `malloc` lowering, which pads every request by the
    // alignment and hands on an aligned pointer inside the block: 16 bytes on
    // every object that asked for `{alignment = 16}`, and every block this
    // allocator returns is 16-aligned already (docs/object-abi.md 6.2).
    LowerToLLVMOptions options(context);
    options.allocLowering = LowerToLLVMOptions::AllocLowering::AlignedAlloc;
    typeConverter = std::make_shared<LLVMTypeConverter>(context, options);
    target->addLegalDialect<LLVM::LLVMDialect>();
    RewritePatternSet collected(context);
    for (Dialect *dialect : context->getLoadedDialects()) {
      // A dialect that promised the interface and was never given it has
      // none, which is what upstream's release build reads it as.
      // ⛔ Not left to dyn_cast: a build with assertions treats asking about
      // an unfulfilled promise as a fatal error.
      if (dialect->hasPromisedInterface(
              dialect->getTypeID(),
              ConvertToLLVMPatternInterface::getInterfaceID()))
        continue;
      auto *iface = dyn_cast<ConvertToLLVMPatternInterface>(dialect);
      if (!iface)
        continue;
      if (isa<func::FuncDialect>(dialect)) {
        populateFuncToLLVMConversionPatterns(*typeConverter, collected,
                                             symbolTables.get());
        continue;
      }
      iface->populateConvertToLLVMConversionPatterns(*target, *typeConverter,
                                                     collected);
    }
    patterns = std::make_shared<FrozenRewritePatternSet>(std::move(collected));
    return success();
  }

  void runOnOperation() override {
    // Built for this module: the collection fills itself on first use.
    symbolTables->invalidateSymbolTable(getOperation());
    ConversionConfig config;
    config.allowPatternRollback = true;
    if (failed(applyPartialConversion(getOperation(), *target, *patterns,
                                      config)))
      signalPassFailure();
  }

private:
  std::shared_ptr<SymbolTableCollection> symbolTables;
  std::shared_ptr<ConversionTarget> target;
  std::shared_ptr<LLVMTypeConverter> typeConverter;
  std::shared_ptr<FrozenRewritePatternSet> patterns;
};

} // namespace

LogicalResult requireNoAsyncDialectOps(ModuleOp module) {
  LogicalResult result = success();
  module.walk([&](Operation *op) {
    if (op->getName().getDialectNamespace() != "async")
      return WalkResult::advance();
    op->emitError()
        << "unlowered async dialect operation remains after Lython async "
           "runtime lowering; MLIR bundled async runtime is not part of the "
           "runtime model";
    result = failure();
    return WalkResult::interrupt();
  });
  return result;
}

} // namespace

IRDumpConfig IRDumpConfig::fromEnv() {
  IRDumpConfig config;
  auto value = llvm::sys::Process::GetEnv("LYTHON_IR_DUMP");
  if (!value || value->empty())
    return config;
  llvm::SmallVector<llvm::StringRef, 16> tokens;
  llvm::StringRef(*value).split(tokens, ",", /*MaxSplit=*/-1,
                                /*KeepEmpty=*/false);
  for (llvm::StringRef token : tokens) {
    std::string name = trimEnvToken(token);
    if (name.empty())
      continue;
    if (name == "all" || name == "*") {
      config.all = true;
      continue;
    }
    config.passes.insert(std::move(name));
  }
  return config;
}

bool IRDumpConfig::shouldDump(llvm::StringRef passName) const {
  return all || passes.count(passName.str()) != 0;
}

LogicalResult runLoweringPipeline(ModuleOp module,
                                  TensorLoweringTarget tensorTarget,
                                  const IRDumpConfig &irDump,
                                  LoweringPipelineOptions options) {
  dumpMLIRForPass(irDump, "frontend", module);
  auto runPhase = [&](llvm::StringRef name, auto populate) {
    return runLoweringPhase(name, module, options.enableVerifiers, populate);
  };
  auto runVerifierPhase = [&](llvm::StringRef name, auto populate) {
    if (!options.enableVerifiers)
      return success();
    return runPhase(name, populate);
  };

  // Phase 1: native target and ABI facts.
  if (failed(runVerifierPhase("native-verification", [&](PassManager &pm) {
        pm.addPass(createNativeVerificationPass());
      })))
    return failure();
  dumpMLIRForPass(irDump, "native-verification", module);

  // Phase 2: publish high-level callable/runtime metadata before rewrites.
  if (failed(runPhase("publication-preparation", [&](PassManager &pm) {
        pm.addPass(createPublicationPreparationPass());
        pm.addPass(createObservableReleasePass());
      })))
    return failure();
  dumpMLIRForPass(irDump, "publication-preparation", module);

  // Phase 3: high-level Py semantic optimizations before runtime import.
  if (failed(runPhase("py-optimization", [&](PassManager &pm) {
        pm.addPass(createPyOptimizationPass());
      })))
    return failure();
  dumpMLIRForPass(irDump, "py-optimization", module);

  // Phase 4: semantic evidence verification before lowering consumes Py ops.
  if (failed(runVerifierPhase(
          "type-evidence-verifier", [&](PassManager &pm) {
            pm.addPass(createTypeEvidenceVerifierPass());
          })))
    return failure();
  dumpMLIRForPass(irDump, "type-evidence-verifier", module);

  // Phase 5: quantitative ownership verification over high-level Py IR.
  if (failed(runVerifierPhase("ownership-verifier", [&](PassManager &pm) {
        pm.addPass(createOwnershipVerifierPass());
      })))
    return failure();
  dumpMLIRForPass(irDump, "ownership-verifier", module);

  // Phase 6: generic cleanup while Python-level structure is still visible.
  if (failed(runPhase("canonicalize", [&](PassManager &pm) {
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::createCSEPass());
      })))
    return failure();
  dumpMLIRForPass(irDump, "canonicalize", module);

  // Phase 7: numeric kernel lowering before Python object lowering.
  if (failed(runPhase("linalg-lowering", [&](PassManager &pm) {
        pm.addPass(createLinalgLoweringPass(tensorTarget));
      })))
    return failure();
  dumpMLIRForPass(irDump, "linalg-lowering", module);

  // Phase 8: import runtime object definitions written in MLIR.
  {
    PerfScope perf("lowering.runtime-objects");
    if (failed(runtime_library::embedObjectModules(module)))
      return failure();
    // Before anything folds: a manifest's class word is a placeholder
    // constant until here.
    if (failed(type_objects::resolveManifestClassWords(module)))
      return failure();
  }
  dumpMLIRForPass(irDump, "runtime-objects", module);

  // Phase 8b: interpret manifest-declared lowering strategies (transform
  // dialect) against the user module.
  {
    PerfScope perf("lowering.lowering-strategies");
    if (failed(runtime_library::applyEmbeddedLoweringStrategies(module)))
      return failure();
  }
  dumpMLIRForPass(irDump, "lowering-strategies", module);

  if (options.auditRuntimeManifest && options.enableVerifiers) {
    if (failed(runVerifierPhase(
            "runtime-manifest-completeness", [&](PassManager &pm) {
              pm.addPass(createRuntimeManifestCompletenessVerifierPass());
            })))
      return failure();
    dumpMLIRForPass(irDump, "runtime-manifest-completeness", module);
  }

  // Phase 8c: every JavaScript value is one runtime contract from here on.
  // After the verifiers that read the program's types, which are the stub's.
  {
    PerfScope perf("lowering.js-host-erasure");
    lowering::js::eraseJsHostContracts(module);
  }
  dumpMLIRForPass(irDump, "js-host-erasure", module);

  // Phase 8d: a coroutine is a generator from here on -- the same frame, the
  // same drivers. Its own type kept `for` and `await` apart while the program
  // was checked; nothing below distinguishes them.
  {
    PerfScope perf("lowering.coroutine-erasure");
    lowering::rewriteContracts(module, [](py::ContractType contract)
                                           -> mlir::Type {
      if (contract.getContractName() != "types.CoroutineType")
        return contract;
      return py::ContractType::get(contract.getContext(),
                                   "types.GeneratorType",
                                   contract.getArguments());
    });
  }
  dumpMLIRForPass(irDump, "coroutine-erasure", module);

  // Phase 9: lower Py dialect values into runtime bundles and calls.
  if (failed(runPhase("runtime-lowering", [&](PassManager &pm) {
        pm.addPass(createRuntimeLoweringPass());
      })))
    return failure();
  dumpMLIRForPass(irDump, "runtime-lowering", module);

  if (failed(runVerifierPhase("runtime-native-verifier", [&](PassManager &pm) {
        pm.addPass(createNativeVerificationPass());
        // Between the pass that mints frame-ownership tokens and the one that
        // consumes them: two tokens on one value are two retains against one
        // release. `proof/`'s `WFES.backed` as a phase gate.
        pm.addPass(createOwnedTokenUniquenessVerifierPass());
      })))
    return failure();
  dumpMLIRForPass(irDump, "runtime-native-verifier", module);

  // Phase 10: insert and simplify ownership operations once calls are concrete.
  // First, the arms whose owned values the unwind cleanup could not see
  // otherwise are written out as blocks (Runtime/Passes/RegionExits.cpp).
  if (failed(runPhase("region-exit-flattening", [&](PassManager &pm) {
        pm.addPass(createRegionExitFlatteningPass());
      })))
    return failure();
  dumpMLIRForPass(irDump, "region-exit-flattening", module);

  if (failed(runPhase("refcount-insertion", [&](PassManager &pm) {
        pm.addPass(createRefCountInsertionPass());
      }))) {
    if (::getenv("LYTHON_DUMP_ON_FAILURE"))
      module.dump();
    return failure();
  }
  dumpMLIRForPass(irDump, "refcount-insertion", module);

  if (failed(runPhase("refcount-elision", [&](PassManager &pm) {
        pm.addPass(createRefCountPairElisionPass());
      })))
    return failure();
  dumpMLIRForPass(irDump, "refcount-elision", module);

  // ⭐ THE BODIES NOBODY CAN REACH ARE EMPTIED, NOT REMOVED. The merged
  // manifest brings ~1500 functions into a module whose program uses a few
  // hundred; every verifier and every conversion after this point walked all
  // of them. Symbol DCE cannot run here -- the verifiers read manifest
  // declarations as contract witnesses (that is why it runs after phase 13,
  // and moving it up made `LyLong_FromI64` look like it may unwind) -- so the
  // symbol and its attributes stay and only the body goes.
  //
  // ⛔ Placed AFTER refcount insertion, not before: that pass is what mints
  // the calls to deallocators, and a deallocator is unreachable until it does.
  // Nothing after this point introduces a call to a manifest body; the unwind
  // insertion below calls LLVM externals the support builder declares.
  {
    PerfScope perf("lowering.unreachable-body-strip");
    lowering::runtime::cleanup::stripUnreachableManifestBodies(module);
  }

  if (failed(runVerifierPhase(
          "pre-cleanup-llvm-call-verifier", [&](PassManager &pm) {
            pm.addPass(createLLVMCallOwnershipVerifierPass());
          })))
    return failure();
  dumpMLIRForPass(irDump, "pre-cleanup-llvm-call-verifier", module);

  // Phase 11: fold what the lowering left foldable before symbol cleanup.
  if (failed(runPhase("post-lowering-canonicalize", [&](PassManager &pm) {
        pm.addPass(createEHSafeCanonicalizerPass());
        pm.addPass(mlir::createCSEPass());
      })))
    return failure();
  dumpMLIRForPass(irDump, "post-lowering-canonicalize", module);
  if (failed(requireNoAsyncDialectOps(module)))
    return failure();

  // Phase 12: remove artifacts from runtime embedding and lowering. Symbol
  // DCE runs AFTER the ownership verifiers (phase 13): manifest contract
  // witnesses (e.g. ly.runtime.shape declarations) must outlive verification.
  if (failed(runPhase("post-lowering-cleanup", [&](PassManager &pm) {
        pm.addPass(createEHSafeCanonicalizerPass());
        pm.addPass(mlir::createCSEPass());
      })))
    return failure();
  {
    PerfScope perf("lowering.pointer-roundtrip-cleanup");
    while (lowering::runtime::cleanup::pointerRoundTrips(module))
      ;
  }
  dumpMLIRForPass(irDump, "post-lowering-cleanup", module);

  // Phase 12b: canonicalization above folds statically-decided region ops
  // and hoists their calls to the top level, where the phase-10 unwind
  // model could not see them; re-run only the unwind-cleanup step so the
  // phase-13 verifier checks obligations the pipeline can still discharge.
  if (failed(runPhase("post-cleanup-unwind-insertion", [&](PassManager &pm) {
        pm.addPass(createPostCleanupUnwindInsertionPass());
      })))
    return failure();
  dumpMLIRForPass(irDump, "post-cleanup-unwind-insertion", module);

  // Phase 13: validate ownership and no-GIL contracts before final lowering.
  if (failed(runVerifierPhase("llvm-call-verifier", [&](PassManager &pm) {
        pm.addPass(createLLVMCallOwnershipVerifierPass());
      })))
    return failure();
  dumpMLIRForPass(irDump, "llvm-call-verifier", module);

  if (failed(runVerifierPhase("thread-safety-verifier", [&](PassManager &pm) {
        pm.addPass(createLLVMThreadSafeVerifierPass());
      })))
    return failure();
  dumpMLIRForPass(irDump, "thread-safety-verifier", module);

  // Contract witnesses are no longer needed once verification is done.
  if (failed(runPhase("post-verifier-symbol-dce", [&](PassManager &pm) {
        pm.addPass(mlir::createSymbolDCEPass());
      })))
    return failure();
  dumpMLIRForPass(irDump, "post-verifier-symbol-dce", module);

  {
    PerfScope perf("lowering.discriminate-eh-markers");
    discriminateEHMarkers(module);
  }

  // Phase 14: final lowering to LLVM dialect.
  {
    // A static object's class word is an address its dense initializer cannot
    // spell; recorded here, written once the global is an LLVM global.
    FailureOr<type_objects::StaticClassWords> staticClassWords =
        type_objects::collectStaticClassWords(module);
    if (failed(staticClassWords))
      return failure();
    LoweredSafetyContracts finalSafetyContracts;
    if (options.enableVerifiers) {
      PerfScope perf("lowering.collect-final-safety-contracts");
      collectLoweredSafetyContracts(module, finalSafetyContracts);
    }
    if (failed(runPhase("convert-to-llvm", [&](PassManager &pm) {
          mlir::ConvertVectorToLLVMPassOptions vectorOptions;
          vectorOptions.reassociateFPReductions = true;
          vectorOptions.x86 = tensorTarget.usesX86();
          mlir::VectorTransferToSCFOptions transferOptions;
          transferOptions.setTargetRank(1);
          pm.addPass(mlir::createLowerAffinePass());
          pm.addPass(mlir::memref::createExpandStridedMetadataPass());
          pm.addPass(mlir::createLowerAffinePass());
          if (tensorTarget.usesArmSME())
            lowering::arch::arm::addSMEPreControlFlowLLVMPrepPipeline(pm);
          pm.addNestedPass<mlir::func::FuncOp>(
              mlir::vector::createLowerVectorMultiReductionPass(
                  mlir::vector::VectorMultiReductionLowering::InnerReduction));
          pm.addPass(mlir::createConvertVectorToSCFPass(transferOptions));
          pm.addPass(mlir::createLowerAffinePass());
          pm.addPass(createEHSafeCanonicalizerPass());
          pm.addPass(mlir::createConvertVectorToLLVMPass(vectorOptions));
          pm.addPass(mlir::createSCFToControlFlowPass());
          if (tensorTarget.usesArmSME())
            lowering::arch::arm::addSMEPostControlFlowLLVMPrepPipeline(pm);
          pm.addPass(mlir::createArithToLLVMConversionPass());
          pm.addPass(mlir::createUBToLLVMConversionPass());
          pm.addPass(mlir::createConvertControlFlowToLLVMPass());
          pm.addPass(std::make_unique<SymbolTableConvertToLLVM>());
          pm.addPass(mlir::createReconcileUnrealizedCastsPass());
          pm.addNestedPass<mlir::func::FuncOp>(
              mlir::createReconcileUnrealizedCastsPass());
          pm.addPass(createEHSafeCanonicalizerPass());
        })))
      return failure();
    if (options.enableVerifiers) {
      PerfScope perf("lowering.preserve-final-safety-contracts");
      if (failed(preserveLoweredSafetyContracts(module, finalSafetyContracts)))
        return failure();
    }
    {
      PerfScope perf("lowering.final-llvm-cleanup");
      optimizer::pipeline::finalLLVMCleanup(module);
    }
    {
      PerfScope perf("lowering.type-objects");
      unsigned pointerBits = type_objects::pointerBitsOf(module);
      if (failed(type_objects::patchStaticClassWords(module, *staticClassWords,
                                                     pointerBits)))
        return failure();
      if (failed(type_objects::defineSlotHooks(module)))
        return failure();
      if (failed(type_objects::defineDeclared(module, pointerBits)))
        return failure();
    }
  }
  dumpMLIRForPass(irDump, "convert-to-llvm", module);

  // Phase 13c: materialize ctypes callback thunks -- function addresses only
  // exist at the LLVM layer (see Ctypes/CallbackThunks.h).
  {
    PerfScope perf("lowering.callback-thunks");
    if (failed(lowering::ctypes::materializeCallbackThunks(module)))
      return failure();
    if (failed(lowering::ctypes::materializeSymbolAddresses(module)))
      return failure();
  }
  dumpMLIRForPass(irDump, "callback-thunks", module);

  // Phase 13c2: define the AMX run-time engine probe -- libc calls and
  // reserved instruction words only exist at the LLVM layer.
  {
    PerfScope perf("lowering.matrix-backend-probe");
    if (failed(lowering::arch::apple::materializeMatrixBackendProbe(module)))
      return failure();
  }
  dumpMLIRForPass(irDump, "matrix-backend-probe", module);

  // Phase 13d: materialize parallel kernel dispatch -- context struct layouts
  // and function addresses only exist at the LLVM layer (see
  // Primitive/TensorParallel.h).
  {
    PerfScope perf("lowering.parallel-dispatch");
    if (failed(lowering::materializeParallelDispatch(module)))
      return failure();
  }
  dumpMLIRForPass(irDump, "parallel-dispatch", module);

  // Phase 14: re-check contracts after final conversion rewrites.
  if (failed(runVerifierPhase("final-native-verifier", [&](PassManager &pm) {
        pm.addPass(createNativeVerificationPass());
      })))
    return failure();
  dumpMLIRForPass(irDump, "final-native-verifier", module);

  if (failed(runVerifierPhase("final-ownership-verifier", [&](PassManager &pm) {
        pm.addPass(createLLVMCallOwnershipVerifierPass());
      })))
    return failure();
  dumpMLIRForPass(irDump, "final-ownership-verifier", module);

  if (failed(runVerifierPhase("final-thread-safety-verifier",
                              [&](PassManager &pm) {
                                pm.addPass(createLLVMThreadSafeVerifierPass());
                              })))
    return failure();
  dumpMLIRForPass(irDump, "final-thread-safety-verifier", module);

  return success();
}

} // namespace py
