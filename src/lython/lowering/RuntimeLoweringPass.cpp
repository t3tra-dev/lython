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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
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
    registry.insert<LLVM::LLVMDialect, func::FuncDialect,
                    cf::ControlFlowDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();
    ctx->loadDialect<bufferization::BufferizationDialect,
                     memref::MemRefDialect>();
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
        target.addIllegalOp<CallVectorOp, CallOp, InvokeOp>();

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
        target.addLegalDialect<LLVM::LLVMDialect, func::FuncDialect,
                               arith::ArithDialect, tensor::TensorDialect,
                               linalg::LinalgDialect, memref::MemRefDialect,
                               bufferization::BufferizationDialect>();
        target.addLegalOp<ModuleOp>();
        target.addIllegalOp<
            StrConstantOp, IntConstantOp, FloatConstantOp, TupleEmptyOp,
            TupleCreateOp, DictEmptyOp, DictInsertOp, NoneOp, FuncObjectOp,
            NumAddOp, NumSubOp, NumLtOp, NumLeOp, NumGtOp, NumGeOp, NumEqOp,
            NumNeOp, CastToPrimOp, CastFromPrimOp, CastIdentityOp, UpcastOp,
            IncRefOp, DecRefOp, ClassNewOp, AttrGetOp, AttrSetOp, ClassOp,
            ExceptionNullOp, TracebackNullOp, LocationCurrentOp,
            ExceptionNewOp, RaiseOp, RaiseCurrentOp, TryOp, TryYieldOp,
            ExceptYieldOp, FinallyYieldOp, ExceptMatchOp>();

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

    // Normalize invoke unwind block arguments to LLVM pointer types.
    {
      auto pyObject = LLVM::LLVMPointerType::get(ctx);
      module.walk([&](LLVM::InvokeOp invoke) {
        Block *unwind = invoke.getUnwindDest();
        if (!unwind)
          return;
        for (BlockArgument arg : llvm::make_early_inc_range(
                 unwind->getArguments())) {
          if (!isPyType(arg.getType()))
            continue;
          arg.setType(pyObject);
          for (auto &use :
               llvm::make_early_inc_range(arg.getUses())) {
            auto *owner = use.getOwner();
            auto cast = dyn_cast<UnrealizedConversionCastOp>(owner);
            if (!cast)
              continue;
            if (cast->getNumOperands() != 1 || cast->getNumResults() != 1)
              continue;
            if (cast.getResult(0).getType() != pyObject)
              continue;
            cast.getResult(0).replaceAllUsesWith(arg);
            cast.erase();
          }
        }
      });
    }

    // Some passes may drop llvm.personality; restore it for landingpads.
    auto ensurePersonalityForLandingpads = [&]() {
      auto personality =
          FlatSymbolRefAttr::get(ctx, "__gxx_personality_v0");
      module.walk([&](func::FuncOp func) {
        if (func->hasAttr("llvm.personality"))
          return;
        bool hasLandingpad = false;
        func.walk([&](LLVM::LandingpadOp) { hasLandingpad = true; });
        if (hasLandingpad)
          func->setAttr("llvm.personality", personality);
      });
    };
    ensurePersonalityForLandingpads();

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

    // Phase 6: Convert func.func to llvm.func before final EH materialization.
    {
      if (dumpLowering) {
        llvm::errs() << "[Before func-to-llvm conversion]\n";
        module.dump();
      }
      RewritePatternSet patterns(ctx);
      populateFuncToLLVMConversionPatterns(typeConverter, patterns);
      ConversionTarget target(*ctx);
      target.addLegalDialect<LLVM::LLVMDialect>();
      target.addIllegalDialect<func::FuncDialect>();
      target.addLegalOp<ModuleOp>();
      target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
      if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
        signalPassFailure();
        return;
      }
      if (dumpLowering) {
        llvm::errs() << "[After func-to-llvm conversion]\n";
        module.dump();
      }
    }

    // Finalize unwind blocks with landingpad in LLVM world.
    {
      auto pyObject = LLVM::LLVMPointerType::get(ctx);
      auto getOrCreatePersonality = [&]() {
        StringRef name = "__gxx_personality_v0";
        if (auto fn = module.lookupSymbol<LLVM::LLVMFuncOp>(name))
          return fn;
        OpBuilder builder(module.getBody(), module.getBody()->begin());
        auto fnType =
            LLVM::LLVMFunctionType::get(builder.getI32Type(), {}, true);
        return builder.create<LLVM::LLVMFuncOp>(module.getLoc(), name, fnType);
      };
      auto getOrCreateRuntimeFunc = [&](StringRef name, Type resultType,
                                        ArrayRef<Type> argTypes) {
        if (auto fn = module.lookupSymbol<LLVM::LLVMFuncOp>(name))
          return fn;
        OpBuilder builder(module.getBody(), module.getBody()->begin());
        auto fnType =
            LLVM::LLVMFunctionType::get(resultType, argTypes, false);
        return builder.create<LLVM::LLVMFuncOp>(module.getLoc(), name, fnType);
      };
      auto emitRuntimeCall = [&](OpBuilder &builder, Location loc,
                                 StringRef name, Type resultType,
                                 ValueRange operands) {
        SmallVector<Type> argTypes;
        argTypes.reserve(operands.size());
        for (Value operand : operands)
          argTypes.push_back(operand.getType());
        Type actualResult =
            resultType ? resultType : LLVM::LLVMVoidType::get(ctx);
        auto callee = getOrCreateRuntimeFunc(name, actualResult, argTypes);
        auto symbol = SymbolRefAttr::get(ctx, callee.getName());
        SmallVector<Type> results;
        if (!isa<LLVM::LLVMVoidType>(actualResult))
          results.push_back(actualResult);
        return builder.create<LLVM::CallOp>(loc, results, symbol, operands);
      };

      auto personality = getOrCreatePersonality();
      auto personalityRef = FlatSymbolRefAttr::get(ctx, personality.getName());

      module.walk([&](LLVM::InvokeOp invoke) {
        auto func = invoke->getParentOfType<LLVM::LLVMFuncOp>();
        if (func && !func->hasAttr("personality"))
          func->setAttr("personality", personalityRef);

        Block *unwind = invoke.getUnwindDest();
        if (!unwind)
          return;
        unwind->clear();
        OpBuilder builder(ctx);
        builder.setInsertionPointToStart(unwind);
        auto lpType = LLVM::LLVMStructType::getLiteral(
            ctx, ArrayRef<Type>{pyObject, builder.getI32Type()});
        auto lp = builder.create<LLVM::LandingpadOp>(
            invoke.getLoc(), lpType, builder.getUnitAttr(), ValueRange{});
        Value raw = builder.create<LLVM::ExtractValueOp>(
            invoke.getLoc(), pyObject, lp.getRes(),
            builder.getDenseI64ArrayAttr({0}));
        emitRuntimeCall(builder, invoke.getLoc(), RuntimeSymbols::kEHCapture,
                        pyObject, ValueRange{raw});
        builder.create<LLVM::ResumeOp>(invoke.getLoc(), lp.getRes());
      });
      if (dumpLowering) {
        llvm::errs() << "[After EH finalize]\n";
        module.dump();
      }
    }

    // Insert a top-level exception handler wrapper for `main`.
    {
      auto mainFunc = module.lookupSymbol<LLVM::LLVMFuncOp>("main");
      if (mainFunc) {
        auto fnType = mainFunc.getFunctionType();
        if (fnType.getNumParams() == 0 &&
            !isa<LLVM::LLVMVoidType>(fnType.getReturnType())) {
          StringRef implName = "__lython_main";
          if (!module.lookupSymbol<LLVM::LLVMFuncOp>(implName)) {
            mainFunc.setName(implName);

            auto getOrCreateRuntimeFunc = [&](StringRef name, Type resultType,
                                              ArrayRef<Type> argTypes) {
              if (auto fn = module.lookupSymbol<LLVM::LLVMFuncOp>(name))
                return fn;
              OpBuilder builder(module.getBody(), module.getBody()->begin());
              auto fnType =
                  LLVM::LLVMFunctionType::get(resultType, argTypes, false);
              return builder.create<LLVM::LLVMFuncOp>(module.getLoc(), name,
                                                      fnType);
            };
            auto emitRuntimeCall = [&](OpBuilder &builder, Location loc,
                                       StringRef name, Type resultType,
                                       ValueRange operands) {
              SmallVector<Type> argTypes;
              argTypes.reserve(operands.size());
              for (Value operand : operands)
                argTypes.push_back(operand.getType());
              Type actualResult =
                  resultType ? resultType : LLVM::LLVMVoidType::get(ctx);
              auto callee =
                  getOrCreateRuntimeFunc(name, actualResult, argTypes);
              auto symbol = SymbolRefAttr::get(ctx, callee.getName());
              SmallVector<Type> results;
              if (!isa<LLVM::LLVMVoidType>(actualResult))
                results.push_back(actualResult);
              return builder.create<LLVM::CallOp>(loc, results, symbol,
                                                  operands);
            };

            auto getOrCreatePersonality = [&]() {
              StringRef name = "__gxx_personality_v0";
              if (auto fn = module.lookupSymbol<LLVM::LLVMFuncOp>(name))
                return fn;
              OpBuilder builder(module.getBody(), module.getBody()->begin());
              auto fnType =
                  LLVM::LLVMFunctionType::get(builder.getI32Type(), {}, true);
              return builder.create<LLVM::LLVMFuncOp>(module.getLoc(), name,
                                                      fnType);
            };

            OpBuilder builder(ctx);
            builder.setInsertionPointToEnd(module.getBody());
            auto wrapper =
                builder.create<LLVM::LLVMFuncOp>(module.getLoc(), "main",
                                                 fnType);

            auto personality = getOrCreatePersonality();
            auto personalityRef =
                FlatSymbolRefAttr::get(ctx, personality.getName());
            wrapper->setAttr("personality", personalityRef);

            Block *entry = wrapper.addEntryBlock(builder);
            Block *normal = builder.createBlock(&wrapper.getBody());
            Block *unwind = builder.createBlock(&wrapper.getBody());

            builder.setInsertionPointToStart(entry);
            auto ptrType = LLVM::LLVMPointerType::get(ctx);
            Value catchAll = builder.create<LLVM::ZeroOp>(module.getLoc(),
                                                          ptrType);
            auto invoke = builder.create<LLVM::InvokeOp>(
                module.getLoc(), fnType.getReturnType(),
                FlatSymbolRefAttr::get(ctx, implName), ValueRange{}, normal,
                ValueRange{}, unwind, ValueRange{});

            builder.setInsertionPointToStart(normal);
            builder.create<LLVM::ReturnOp>(module.getLoc(),
                                           invoke.getResult());

            builder.setInsertionPointToStart(unwind);
            auto i32Type = builder.getI32Type();
            auto lpType = LLVM::LLVMStructType::getLiteral(
                ctx, ArrayRef<Type>{ptrType, i32Type});
            auto lp = builder.create<LLVM::LandingpadOp>(
                module.getLoc(), lpType, builder.getUnitAttr(),
                ValueRange{catchAll});
            Value raw = builder.create<LLVM::ExtractValueOp>(
                module.getLoc(), ptrType, lp.getRes(),
                builder.getDenseI64ArrayAttr({0}));
            auto captured = emitRuntimeCall(
                builder, module.getLoc(), RuntimeSymbols::kEHCapture, ptrType,
                ValueRange{raw});
            auto reported = emitRuntimeCall(
                builder, module.getLoc(), RuntimeSymbols::kEHReportUnhandled,
                i32Type, ValueRange{captured.getResult()});
            builder.create<LLVM::ReturnOp>(module.getLoc(),
                                           reported.getResult());
          }
        }
      }
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createRuntimeLoweringPass() {
  return std::make_unique<RuntimeLoweringPass>();
}

} // namespace py
