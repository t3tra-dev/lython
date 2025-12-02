#include "RuntimeSupport.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <string>

#include "PyDialect.h.inc"

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

using namespace mlir;

namespace py {
namespace {

static LogicalResult translateFunctionSignature(
    FuncSignatureType sig, const PyLLVMTypeConverter &typeConverter,
    SmallVectorImpl<Type> &pyInputs, SmallVectorImpl<Type> &convertedInputs,
    SmallVectorImpl<Type> &convertedResults, Operation *emitOnError) {
  pyInputs.clear();
  convertedInputs.clear();
  convertedResults.clear();

  auto appendConverted = [&](Type ty,
                             SmallVectorImpl<Type> &storage) -> LogicalResult {
    Type converted = typeConverter.convertType(ty);
    if (!converted) {
      emitOnError->emitError("failed to convert type ") << ty;
      return failure();
    }
    storage.push_back(converted);
    return success();
  };

  auto positional = sig.getPositionalTypes();
  pyInputs.append(positional.begin(), positional.end());
  auto kwonly = sig.getKwOnlyTypes();
  pyInputs.append(kwonly.begin(), kwonly.end());
  if (sig.hasVararg())
    pyInputs.push_back(sig.getVarargType());
  if (sig.hasKwarg())
    pyInputs.push_back(sig.getKwargType());

  for (Type ty : pyInputs)
    if (failed(appendConverted(ty, convertedInputs)))
      return failure();

  for (Type result : sig.getResultTypes())
    if (failed(appendConverted(result, convertedResults)))
      return failure();

  return success();
}

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
    auto materializationFilter = [&](Diagnostic &diag) -> LogicalResult {
      std::string message;
      llvm::raw_string_ostream os(message);
      diag.print(os);
      os.flush();
      if (message.find("unresolved materialization") != std::string::npos)
        return success();
      return failure();
    };

    // py.func / py.return を func.func / func.return に下げる
    struct FuncOpLowering : public OpConversionPattern<FuncOp> {
      FuncOpLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
          : OpConversionPattern<FuncOp>(converter, ctx) {}

      LogicalResult
      matchAndRewrite(FuncOp op, OpAdaptor adaptor,
                      ConversionPatternRewriter &rewriter) const override {
        auto nameAttr = op->getAttrOfType<StringAttr>("sym_name");
        if (!nameAttr)
          return rewriter.notifyMatchFailure(op, "missing sym_name");
        if (nameAttr.getValue() == "__builtin_print") {
          rewriter.eraseOp(op);
          return success();
        }
        auto sigAttr = op->getAttrOfType<TypeAttr>("function_type");
        if (!sigAttr)
          return rewriter.notifyMatchFailure(op, "missing function_type attr");
        auto sig = dyn_cast<FuncSignatureType>(sigAttr.getValue());
        if (!sig)
          return rewriter.notifyMatchFailure(op, "expected FuncSignatureType");

        auto *tc = static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
        SmallVector<Type, 8> pyInputTypes;
        SmallVector<Type, 8> llvmInputTypes;
        SmallVector<Type, 4> llvmResultTypes;
        if (failed(translateFunctionSignature(
                sig, *tc, pyInputTypes, llvmInputTypes, llvmResultTypes, op)))
          return failure();

        auto funcType =
            FunctionType::get(getContext(), llvmInputTypes, llvmResultTypes);
        auto newFunc = rewriter.create<func::FuncOp>(
            op.getLoc(), nameAttr.getValue(), funcType);
        newFunc->setAttr("llvm.emit_c_interface", rewriter.getUnitAttr());

        if (op.getBody().empty())
          op.getBody().emplaceBlock();

        rewriter.inlineRegionBefore(op.getBody(), newFunc.getBody(),
                                    newFunc.getBody().end());
        auto &entry = newFunc.getBody().front();
        TypeConverter::SignatureConversion conversion(pyInputTypes.size());
        for (auto [idx, ty] : llvm::enumerate(llvmInputTypes)) {
          SmallVector<Type, 1> packed{ty};
          conversion.addInputs(idx, packed);
        }
        auto *convertedEntry =
            rewriter.applySignatureConversion(&entry, conversion);
        if (!convertedEntry)
          return failure();

        rewriter.eraseOp(op);
        return success();
      }
    };

    struct ReturnLowering : public OpConversionPattern<ReturnOp> {
      ReturnLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
          : OpConversionPattern<ReturnOp>(converter, ctx) {}

      LogicalResult
      matchAndRewrite(ReturnOp op, OpAdaptor adaptor,
                      ConversionPatternRewriter &rewriter) const override {
        rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getOperands());
        return success();
      }
    };

    auto runFuncConversion = [&]() -> LogicalResult {
      while (true) {
        RewritePatternSet funcPatterns(ctx);
        funcPatterns
            .add<FuncOpLowering, ReturnLowering, UnrealizedCastLowering>(
                typeConverter, ctx);
        ConversionTarget funcTarget(*ctx);
        funcTarget.addLegalDialect<py::PyDialect>();
        funcTarget.addLegalOp<ModuleOp>();
        funcTarget.markUnknownOpDynamicallyLegal(
            [](Operation *) { return true; });
        funcTarget.addIllegalOp<FuncOp, ReturnOp>();
        ScopedDiagnosticHandler diagHandler(ctx, materializationFilter);
        auto result =
            applyPartialConversion(module, funcTarget, std::move(funcPatterns));
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
    if (llvm::sys::Process::GetEnv("LYTHON_DUMP_LOWERING_IR")) {
      llvm::errs() << "[After func conversion]\n";
      module.dump();
    }

    struct FuncObjectLowering : public OpConversionPattern<FuncObjectOp> {
      FuncObjectLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
          : OpConversionPattern<FuncObjectOp>(converter, ctx) {}

      LogicalResult
      matchAndRewrite(FuncObjectOp op, OpAdaptor adaptor,
                      ConversionPatternRewriter &rewriter) const override {
        static llvm::StringMap<llvm::StringLiteral> builtinTable = {
            {"__builtin_print", RuntimeSymbols::kGetBuiltinPrint},
            {"print", RuntimeSymbols::kGetBuiltinPrint},
        };

        ModuleOp module = op->getParentOfType<ModuleOp>();
        if (!module)
          return failure();

        StringRef symbol = op.getTargetAttr().getValue();
        if (auto it = builtinTable.find(symbol); it != builtinTable.end()) {
          auto *converter =
              static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
          RuntimeAPI runtime(module, rewriter, *converter);
          Type resultType = converter->convertType(op.getResult().getType());
          auto call =
              runtime.call(op.getLoc(), it->second, resultType, ValueRange{});
          rewriter.replaceOp(op, call.getResults());
          return success();
        }

        auto func = module.lookupSymbol<func::FuncOp>(symbol);
        if (!func)
          return rewriter.notifyMatchFailure(
              op, "unknown function reference '" + symbol + "'");

        auto constOp = rewriter.create<func::ConstantOp>(
            op.getLoc(), func.getFunctionType(),
            SymbolRefAttr::get(rewriter.getContext(), symbol));
        auto identity = rewriter.create<CastIdentityOp>(
            op.getLoc(), op.getResult().getType(), constOp.getResult());
        rewriter.replaceOp(op, identity.getResult());
        return success();
      }
    };

    auto runFuncObjectConversion = [&]() -> LogicalResult {
      while (true) {
        RewritePatternSet funcObjectPatterns(ctx);
        funcObjectPatterns.add<FuncObjectLowering, UnrealizedCastLowering>(
            typeConverter, ctx);
        ConversionTarget funcObjectTarget(*ctx);
        funcObjectTarget.addLegalDialect<py::PyDialect>();
        funcObjectTarget.addLegalOp<ModuleOp>();
        funcObjectTarget.markUnknownOpDynamicallyLegal(
            [](Operation *) { return true; });
        funcObjectTarget.addIllegalOp<FuncObjectOp>();
        ScopedDiagnosticHandler diagHandler(ctx, materializationFilter);
        auto result = applyPartialConversion(module, funcObjectTarget,
                                             std::move(funcObjectPatterns));
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
    if (llvm::sys::Process::GetEnv("LYTHON_DUMP_LOWERING_IR")) {
      llvm::errs() << "[After func object conversion]\n";
      module.dump();
    }

    auto runCallConversion = [&]() -> LogicalResult {
      while (true) {
        RewritePatternSet callPatterns(ctx);
        populatePyCallLoweringPatterns(typeConverter, callPatterns);
        callPatterns.add<UnrealizedCastLowering>(typeConverter, ctx);
        ConversionTarget callTarget(*ctx);
        callTarget.addLegalDialect<py::PyDialect>();
        callTarget.addLegalOp<ModuleOp>();
        callTarget.markUnknownOpDynamicallyLegal(
            [](Operation *) { return true; });
        callTarget.addIllegalOp<CallVectorOp, CallOp>();
        ScopedDiagnosticHandler diagHandler(ctx, materializationFilter);
        auto result =
            applyPartialConversion(module, callTarget, std::move(callPatterns));
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

    // Clean up dead tuple operations whose only users are DecRefOps
    auto cleanupDeadTuples = [&]() {
      SmallVector<Operation *> toErase;
      module.walk([&](Operation *tupleOp) {
        if (!isa<TupleCreateOp, TupleEmptyOp>(tupleOp))
          return;
        Value result = tupleOp->getResult(0);
        SmallVector<Operation *> decrefsToErase;
        bool canErase = true;
        for (Operation *user : result.getUsers()) {
          if (auto decref = dyn_cast<DecRefOp>(user)) {
            decrefsToErase.push_back(decref);
          } else {
            canErase = false;
            break;
          }
        }
        if (canErase && !decrefsToErase.empty()) {
          for (Operation *decref : decrefsToErase)
            toErase.push_back(decref);
          toErase.push_back(tupleOp);
        }
      });
      for (Operation *op : toErase)
        op->erase();
      return !toErase.empty();
    };
    cleanupDeadTuples();

    // Optimization 1: Hoist integer constants to entry block (CSE)
    auto hoistIntConstants = [&]() {
      module.walk([&](func::FuncOp func) {
        if (func.isExternal())
          return;
        Block &entryBlock = func.getBody().front();
        llvm::StringMap<IntConstantOp> constantMap;

        // First pass: collect all IntConstantOp
        SmallVector<IntConstantOp> allConstants;
        func.walk([&](IntConstantOp op) { allConstants.push_back(op); });

        // Second pass: CSE and hoist (using string value as key)
        for (IntConstantOp op : allConstants) {
          llvm::StringRef value = op.getValue();
          auto it = constantMap.find(value);
          if (it != constantMap.end()) {
            // Replace with existing constant
            op.getResult().replaceAllUsesWith(it->second.getResult());
            op->erase();
          } else {
            // Move to entry block if not already there
            if (op->getBlock() != &entryBlock) {
              op->moveBefore(&entryBlock, entryBlock.begin());
            }
            constantMap[value] = op;
          }
        }
      });
    };
    hoistIntConstants();

    // Optimization 2: Remove decref for small integer constants (-5 to 256)
    auto removeSmallIntDecrefs = [&]() {
      SmallVector<DecRefOp> toErase;
      module.walk([&](DecRefOp decref) {
        Value obj = decref.getObject();
        if (auto intConst = obj.getDefiningOp<IntConstantOp>()) {
          llvm::StringRef valueStr = intConst.getValue();
          // Parse the string to check if it's in small int range
          char *end;
          long long value = std::strtoll(valueStr.data(), &end, 10);
          // Only apply optimization if parsing succeeded and value is in range
          if (end == valueStr.data() + valueStr.size() && value >= -5 &&
              value <= 256) {
            toErase.push_back(decref);
          }
        }
      });
      for (auto op : toErase)
        op->erase();
    };
    removeSmallIntDecrefs();

    if (llvm::sys::Process::GetEnv("LYTHON_DUMP_LOWERING_IR")) {
      llvm::errs() << "[After call conversion]\n";
      module.dump();
    }

    // py.upcast は実体が同一表現（PyObject*）なので単純に転送
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

    auto runValueConversion = [&]() -> LogicalResult {
      bool dumpLowering = static_cast<bool>(
          llvm::sys::Process::GetEnv("LYTHON_DUMP_LOWERING_IR"));
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
        auto conversionResult =
            applyPartialConversion(module, target, std::move(patterns));
        if (dumpLowering) {
          llvm::errs() << "[After final conversion attempt]\n";
          module.dump();
        }
        if (succeeded(conversionResult))
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

    // Optimization: CSE for LyUnicode_FromUTF8 calls within each function
    // This reduces redundant string key creation for attribute access
    // After CSE, we add decref at function end for the cached strings
    auto cseStringCreation = [&]() {
      module.walk([&](func::FuncOp func) {
        if (func.isExternal())
          return;

        // Map from (string_global_symbol, length) -> first call result
        llvm::DenseMap<std::pair<StringRef, int64_t>, LLVM::CallOp> stringCache;
        SmallVector<LLVM::CallOp> toErase;
        SmallVector<LLVM::CallOp> cachedStrings;

        func.walk([&](LLVM::CallOp callOp) {
          auto callee = callOp.getCallee();
          if (!callee || *callee != "LyUnicode_FromUTF8")
            return;

          // Get the string global and length arguments
          if (callOp.getNumOperands() != 2)
            return;

          // First operand should be a GEP pointing to a global string
          auto gepOp = callOp.getOperand(0).getDefiningOp<LLVM::GEPOp>();
          if (!gepOp)
            return;
          auto addrOp = gepOp.getBase().getDefiningOp<LLVM::AddressOfOp>();
          if (!addrOp)
            return;
          StringRef globalName = addrOp.getGlobalName();

          // Second operand should be a constant length
          auto lenConst =
              callOp.getOperand(1).getDefiningOp<LLVM::ConstantOp>();
          if (!lenConst)
            return;
          auto lenAttr = llvm::dyn_cast<IntegerAttr>(lenConst.getValue());
          if (!lenAttr)
            return;
          int64_t len = lenAttr.getInt();

          auto key = std::make_pair(globalName, len);
          auto it = stringCache.find(key);
          if (it != stringCache.end()) {
            // Replace with cached result
            callOp.getResult().replaceAllUsesWith(it->second.getResult());
            toErase.push_back(callOp);
          } else {
            stringCache[key] = callOp;
            cachedStrings.push_back(callOp);
          }
        });

        for (auto op : toErase)
          op->erase();

        // Add decref for cached strings before each return
        if (!cachedStrings.empty()) {
          func.walk([&](func::ReturnOp returnOp) {
            OpBuilder builder(returnOp);
            auto decrefFunc =
                module.lookupSymbol<LLVM::LLVMFuncOp>("Ly_DecRef");
            if (decrefFunc) {
              for (auto cachedCall : cachedStrings) {
                builder.create<LLVM::CallOp>(
                    returnOp.getLoc(), decrefFunc,
                    ValueRange{cachedCall.getResult()});
              }
            }
          });
        }
      });
    };
    cseStringCreation();

    if (llvm::sys::Process::GetEnv("LYTHON_DUMP_LOWERING_IR")) {
      llvm::errs() << "[After string CSE]\n";
      module.dump();
    }

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
