#include "RuntimeSupport.h"

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

using namespace mlir;

namespace py {
namespace {

class BasePyOpPattern : public RewritePattern {
public:
  BasePyOpPattern(StringRef rootName, MLIRContext *ctx,
                  PyLLVMTypeConverter &converter)
      : RewritePattern(rootName, 1, ctx), typeConverter(converter) {}

protected:
  PyLLVMTypeConverter &typeConverter;
};

struct FuncObjectLowering : public OpConversionPattern<FuncObjectOp> {
  FuncObjectLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<FuncObjectOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(FuncObjectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();
    auto *converter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *converter);

    static llvm::StringMap<llvm::StringLiteral> builtinTable = {
        {"__builtin_print", RuntimeSymbols::kGetBuiltinPrint},
        {"print", RuntimeSymbols::kGetBuiltinPrint},
    };

    auto symbol = op.getTargetAttr().getValue();
    auto it = builtinTable.find(symbol);
    if (it == builtinTable.end())
      return rewriter.notifyMatchFailure(
          op, "unsupported builtin function reference '" + symbol + "'");

    Type resultType = converter->convertType(op.getResult().getType());
    auto call = runtime.call(op.getLoc(), it->second, resultType, ValueRange{});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};

struct CallVectorLowering : public OpConversionPattern<CallVectorOp> {
  CallVectorLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<CallVectorOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(CallVectorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    if (op.getNumResults() > 1)
      return rewriter.notifyMatchFailure(
          op, "runtime lowering supports at most one result");

    auto *converter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *converter);
    Type pyObject = runtime.getPyObjectPtrType();

    SmallVector<Value> operands{adaptor.getCallable(), adaptor.getPosargs(),
                                adaptor.getKwnames(), adaptor.getKwvalues()};
    auto call = runtime.call(op.getLoc(), RuntimeSymbols::kCallVectorcall,
                             pyObject, operands);

    if (op.getNumResults() == 0) {
      rewriter.eraseOp(op);
      return success();
    }

    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};

struct CallOpLowering : public OpConversionPattern<CallOp> {
  CallOpLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<CallOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    if (op.getNumResults() > 1)
      return rewriter.notifyMatchFailure(
          op, "runtime lowering supports at most one result");

    auto *converter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *converter);
    Type pyObject = runtime.getPyObjectPtrType();

    auto call =
        runtime.call(op.getLoc(), RuntimeSymbols::kCall, pyObject,
                     ValueRange{adaptor.getCallable(), adaptor.getPosargs(),
                                adaptor.getKwargs()});
    if (op.getNumResults() == 0) {
      rewriter.eraseOp(op);
      return success();
    }

    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};

} // namespace

void populatePyCallLoweringPatterns(PyLLVMTypeConverter &typeConverter,
                                    RewritePatternSet &patterns) {
  auto *ctx = patterns.getContext();
  patterns.add<FuncObjectLowering, CallVectorLowering, CallOpLowering>(
      typeConverter, ctx);
}

} // namespace py
