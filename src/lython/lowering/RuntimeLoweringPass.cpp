#include "RuntimeSupport.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

using namespace mlir;

namespace py {
namespace {

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

    RewritePatternSet patterns(ctx);
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
        auto sigAttr = op->getAttr("function_type");
        if (!sigAttr)
          return rewriter.notifyMatchFailure(op, "missing function_type attr");
        auto sig =
            dyn_cast<FuncSignatureType>(cast<TypeAttr>(sigAttr).getValue());
        if (!sig)
          return rewriter.notifyMatchFailure(
              op, "function_type is not FuncSignatureType");

        // 現状は引数無しのみ対応（__main__など）。引数ありは今後拡張。
        if (!sig.getPositionalTypes().empty() ||
            !sig.getKwOnlyTypes().empty() || sig.hasVararg() || sig.hasKwarg())
          return rewriter.notifyMatchFailure(
              op, "only zero-arg functions supported for now");

        SmallVector<Type> resultTypes;
        auto *tc = static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
        for (Type rt : sig.getResultTypes()) {
          Type conv = tc->convertType(rt);
          if (!conv)
            return rewriter.notifyMatchFailure(op,
                                               "failed to convert result type");
          resultTypes.push_back(conv);
        }

        auto funcType =
            FunctionType::get(getContext(), TypeRange{}, resultTypes);
        auto newFunc = rewriter.create<func::FuncOp>(
            op.getLoc(), nameAttr.getValue(), funcType);
        newFunc->setAttr("llvm.emit_c_interface", rewriter.getUnitAttr());

        // 本体をそのまま移す（引数なし前提）
        rewriter.inlineRegionBefore(op.getBody(), newFunc.getBody(),
                                    newFunc.getBody().end());

        rewriter.replaceOp(op, std::nullopt);
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

    populatePyValueLoweringPatterns(typeConverter, patterns);
    populatePyTupleLoweringPatterns(typeConverter, patterns);
    populatePyDictLoweringPatterns(typeConverter, patterns);
    populatePyCallLoweringPatterns(typeConverter, patterns);
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

    patterns.add<FuncOpLowering, ReturnLowering, UpcastLowering>(typeConverter,
                                                                 ctx);

    ConversionTarget target(*ctx);
    target.addLegalDialect<LLVM::LLVMDialect, func::FuncDialect>();
    target.addLegalOp<ModuleOp>();
    target.addIllegalOp<StrConstantOp, TupleEmptyOp, TupleCreateOp, DictEmptyOp,
                        DictInsertOp, NoneOp, FuncObjectOp, CallVectorOp,
                        CallOp, UpcastOp, FuncOp, ReturnOp>();

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createRuntimeLoweringPass() {
  return std::make_unique<RuntimeLoweringPass>();
}

} // namespace py
