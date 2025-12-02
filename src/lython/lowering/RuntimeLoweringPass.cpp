// This file implements the main RuntimeLoweringPass which orchestrates the
// complete lowering pipeline from Py dialect to LLVM dialect. It coordinates
// the various conversion phases:
//   1. Function conversion (py.func -> func.func)
//   2. Function object conversion (py.func_object -> references)
//   3. Call conversion (py.call_vector -> runtime calls or direct calls)
//   4. Value conversion (py.* ops -> LLVM ops via runtime calls)
//
// Individual lowering patterns are implemented in separate files:
//   - PyFuncLowering.cpp: Function and calling convention patterns
//   - PyCallLowering.cpp: Call operation patterns
//   - PyValueLowering.cpp: Value creation patterns
//   - PyTupleLowering.cpp: Tuple operation patterns
//   - PyDictLowering.cpp: Dictionary operation patterns
//   - PyRefCountLowering.cpp: Reference counting patterns
//
// Optimizations are implemented in PyOptimizationPass.cpp.

#include "RuntimeSupport.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"

#include "PyDialect.h.inc"

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

using namespace mlir;

namespace py {
namespace {

// Utility functions

/// Replaces UnrealizedConversionCastOp involving py.* types with
/// CastIdentityOp. Returns true if any replacements were made.
static bool replaceUnrealizedCastsWithIdentity(Operation *container) {
  SmallVector<UnrealizedConversionCastOp> pending;

  container->walk([&](UnrealizedConversionCastOp cast) {
    if (cast->getNumOperands() != 1 || cast->getNumResults() != 1)
      return;
    Type inputType = cast->getOperand(0).getType();
    Type resultType = cast->getResultTypes().front();
    bool involvesPy = isPyType(inputType) || isPyType(resultType);
    if (!involvesPy)
      return;
    pending.push_back(cast);
  });

  for (auto cast : pending) {
    OpBuilder builder(cast);
    auto identity = builder.create<CastIdentityOp>(
        cast.getLoc(), cast->getResultTypes().front(), cast->getOperand(0));
    cast->getResult(0).replaceAllUsesWith(identity.getResult());
    cast->erase();
  }

  return !pending.empty();
}

// Type cast lowering patterns

struct UnrealizedCastLowering
    : public OpConversionPattern<UnrealizedConversionCastOp> {
  UnrealizedCastLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<UnrealizedConversionCastOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(UnrealizedConversionCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "expected single operand");

    Type inputType = op.getOperandTypes().front();
    Type resultType = op.getResultTypes().front();
    if (!isPyType(inputType) && !isPyType(resultType))
      return rewriter.notifyMatchFailure(
          op, "unrelated to py.* types, keep default handling");

    auto identity = rewriter.create<CastIdentityOp>(
        op.getLoc(), resultType, adaptor.getOperands().front());
    rewriter.replaceOp(op, identity.getResult());
    return success();
  }
};

/// py.upcast forwards the operand since all py.* types share the same
/// runtime representation (PyObject*).
struct UpcastLowering : public OpConversionPattern<UpcastOp> {
  UpcastLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<UpcastOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(UpcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (adaptor.getOperands().empty())
      return failure();
    rewriter.replaceOp(op, adaptor.getOperands().front());
    return success();
  }
};

struct CastIdentityLowering : public OpConversionPattern<CastIdentityOp> {
  CastIdentityLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<CastIdentityOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(CastIdentityOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (adaptor.getInput())
      rewriter.replaceOp(op, adaptor.getInput());
    else
      rewriter.eraseOp(op);
    return success();
  }
};

// RuntimeLoweringPass: Main pipeline orchestration

struct RuntimeLoweringPass
    : public PassWrapper<RuntimeLoweringPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RuntimeLoweringPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, func::FuncDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();
    PyLLVMTypeConverter typeConverter(ctx);
    bool dumpLowering = static_cast<bool>(
        llvm::sys::Process::GetEnv("LYTHON_DUMP_LOWERING_IR"));

    auto materializationFilter = [&](Diagnostic &diag) -> LogicalResult {
      std::string message;
      llvm::raw_string_ostream os(message);
      diag.print(os);
      os.flush();
      if (message.find("unresolved materialization") != std::string::npos)
        return success();
      return failure();
    };

    // Phase 1: Function conversion (py.func/py.return -> func.func/func.return)

    auto runFuncConversion = [&]() -> LogicalResult {
      while (true) {
        RewritePatternSet patterns(ctx);
        populatePyFuncLoweringPatterns(typeConverter, patterns);
        patterns.add<UnrealizedCastLowering>(typeConverter, ctx);

        ConversionTarget target(*ctx);
        target.addLegalDialect<py::PyDialect>();
        target.addLegalOp<ModuleOp>();
        target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
        target.addIllegalOp<FuncOp, ReturnOp>();

        ScopedDiagnosticHandler diagHandler(ctx, materializationFilter);
        auto result =
            applyPartialConversion(module, target, std::move(patterns));
        if (succeeded(result))
          return success();
        if (!replaceUnrealizedCastsWithIdentity(module))
          return failure();
      }
    };

    if (failed(runFuncConversion())) {
      signalPassFailure();
      return;
    }
    replaceUnrealizedCastsWithIdentity(module);

    if (dumpLowering) {
      llvm::errs() << "[After func conversion]\n";
      module.dump();
    }

    // Phase 2: Function object conversion (py.func_object -> references)

    auto runFuncObjectConversion = [&]() -> LogicalResult {
      while (true) {
        RewritePatternSet patterns(ctx);
        populatePyFuncLoweringPatterns(typeConverter, patterns);
        patterns.add<UnrealizedCastLowering>(typeConverter, ctx);

        ConversionTarget target(*ctx);
        target.addLegalDialect<py::PyDialect>();
        target.addLegalOp<ModuleOp>();
        target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
        target.addIllegalOp<FuncObjectOp>();

        ScopedDiagnosticHandler diagHandler(ctx, materializationFilter);
        auto result =
            applyPartialConversion(module, target, std::move(patterns));
        if (succeeded(result))
          return success();
        if (!replaceUnrealizedCastsWithIdentity(module))
          return failure();
      }
    };

    if (failed(runFuncObjectConversion())) {
      signalPassFailure();
      return;
    }
    replaceUnrealizedCastsWithIdentity(module);

    if (dumpLowering) {
      llvm::errs() << "[After func object conversion]\n";
      module.dump();
    }

    // Phase 3: Call conversion (py.call_vector/py.call -> calls)

    auto runCallConversion = [&]() -> LogicalResult {
      while (true) {
        RewritePatternSet patterns(ctx);
        populatePyCallLoweringPatterns(typeConverter, patterns);
        patterns.add<UnrealizedCastLowering>(typeConverter, ctx);

        ConversionTarget target(*ctx);
        target.addLegalDialect<py::PyDialect>();
        target.addLegalOp<ModuleOp>();
        target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
        target.addIllegalOp<CallVectorOp, CallOp>();

        ScopedDiagnosticHandler diagHandler(ctx, materializationFilter);
        auto result =
            applyPartialConversion(module, target, std::move(patterns));
        if (succeeded(result))
          return success();
        if (!replaceUnrealizedCastsWithIdentity(module))
          return failure();
      }
    };

    if (failed(runCallConversion())) {
      signalPassFailure();
      return;
    }
    replaceUnrealizedCastsWithIdentity(module);

    // Apply pre-lowering optimizations
    runPreLoweringOptimizations(module);

    if (dumpLowering) {
      llvm::errs() << "[After call conversion]\n";
      module.dump();
    }

    // Phase 4: Value conversion (py.* ops -> LLVM ops)

    auto runValueConversion = [&]() -> LogicalResult {
      while (true) {
        RewritePatternSet patterns(ctx);
        populatePyValueLoweringPatterns(typeConverter, patterns);
        populatePyTupleLoweringPatterns(typeConverter, patterns);
        populatePyDictLoweringPatterns(typeConverter, patterns);
        populatePyRefCountLoweringPatterns(typeConverter, patterns);
        patterns
            .add<UpcastLowering, CastIdentityLowering, UnrealizedCastLowering>(
                typeConverter, ctx);

        ConversionTarget target(*ctx);
        target.addLegalDialect<LLVM::LLVMDialect, func::FuncDialect>();
        target.addLegalOp<ModuleOp>();
        target.addIllegalOp<
            StrConstantOp, IntConstantOp, FloatConstantOp, TupleEmptyOp,
            TupleCreateOp, DictEmptyOp, DictInsertOp, NoneOp, FuncObjectOp,
            NumAddOp, NumSubOp, NumLeOp, CastToPrimOp, CastIdentityOp, UpcastOp,
            IncRefOp, DecRefOp, ClassNewOp, AttrGetOp, AttrSetOp, ClassOp>();

        ScopedDiagnosticHandler diagHandler(ctx, materializationFilter);
        auto result =
            applyPartialConversion(module, target, std::move(patterns));

        if (dumpLowering) {
          llvm::errs() << "[After final conversion attempt]\n";
          module.dump();
        }

        if (succeeded(result))
          return success();

        if (dumpLowering)
          llvm::errs() << "[Final conversion failure]\n";

        if (!replaceUnrealizedCastsWithIdentity(module))
          return failure();
      }
    };

    if (failed(runValueConversion())) {
      signalPassFailure();
      return;
    }

    // Apply post-lowering optimizations
    runPostLoweringOptimizations(module);

    if (dumpLowering) {
      llvm::errs() << "[After optimizations]\n";
      module.dump();
    }

    // Phase 5: Cleanup remaining casts

    replaceUnrealizedCastsWithIdentity(module);
    RewritePatternSet cleanupPatterns(ctx);
    cleanupPatterns.add<CastIdentityLowering>(typeConverter, ctx);
    GreedyRewriteConfig config;
    if (failed(applyPatternsGreedily(module, std::move(cleanupPatterns),
                                     config))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createRuntimeLoweringPass() {
  return std::make_unique<RuntimeLoweringPass>();
}

} // namespace py
