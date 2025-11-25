#include "RuntimeSupport.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/STLExtras.h"

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

using namespace mlir;

namespace py {
namespace {

struct CallVectorLowering : public OpConversionPattern<CallVectorOp> {
  CallVectorLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<CallVectorOp>(converter, ctx) {}

  LogicalResult lowerViaRuntime(CallVectorOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
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

  LogicalResult
  matchAndRewrite(CallVectorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    Value callable = op.getCallable();
    if (auto identity = callable.getDefiningOp<CastIdentityOp>())
      callable = identity.getInput();
    Operation *producer = callable.getDefiningOp();
    auto constOp = dyn_cast_or_null<func::ConstantOp>(producer);
    if (!constOp)
      return lowerViaRuntime(op, adaptor, rewriter);

    SymbolRefAttr symbolRef = constOp.getValueAttr();
    StringRef symbolName = symbolRef.getLeafReference().empty()
                               ? symbolRef.getRootReference().getValue()
                               : symbolRef.getLeafReference().getValue();
    auto func = module.lookupSymbol<func::FuncOp>(symbolName);
    if (!func)
      return lowerViaRuntime(op, adaptor, rewriter);

    if (!isa<TupleEmptyOp>(op.getKwnames().getDefiningOp()) ||
        !isa<TupleEmptyOp>(op.getKwvalues().getDefiningOp()))
      return lowerViaRuntime(op, adaptor, rewriter);

    auto funcType = func.getFunctionType();
    SmallVector<Value> callOperands;
    auto posargsOp = op.getPosargs().getDefiningOp();
    if (auto tupleCreate = dyn_cast_or_null<TupleCreateOp>(posargsOp)) {
      if (tupleCreate.getElements().size() != funcType.getNumInputs())
        return lowerViaRuntime(op, adaptor, rewriter);
      for (auto [idx, element] : llvm::enumerate(tupleCreate.getElements())) {
        Type expected = funcType.getInput(idx);
        Value operand = element;
        if (operand.getType() != expected)
          operand =
              rewriter.create<CastIdentityOp>(op.getLoc(), expected, operand);
        callOperands.push_back(operand);
      }
    } else if (isa<TupleEmptyOp>(posargsOp)) {
      if (funcType.getNumInputs() != 0)
        return lowerViaRuntime(op, adaptor, rewriter);
    } else {
      return lowerViaRuntime(op, adaptor, rewriter);
    }

    // Create direct call, bypassing the tuple-based vectorcall
    auto call = rewriter.create<func::CallOp>(op.getLoc(), func, callOperands);

    if (call.getNumResults() == 0) {
      rewriter.eraseOp(op);
    } else {
      rewriter.replaceOp(op, call.getResults());
    }

    // Dead tuple ops and their DecRefOps are cleaned up by
    // cleanupDeadTuples() in RuntimeLoweringPass after call conversion

    if (constOp->use_empty())
      rewriter.eraseOp(constOp);
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
  patterns.add<CallVectorLowering, CallOpLowering>(typeConverter, ctx);
}

} // namespace py
