#include "Common/LoweringUtils.h"
#include "Common/RuntimeSupport.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/ADT/STLExtras.h"

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

using namespace mlir;

namespace py {
namespace {

static LogicalResult emitStaticClassRefCountCall(
    Operation *op, Value object, ClassType classType, ModuleOp module,
    ConversionPatternRewriter &rewriter, StringRef suffix) {
  auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
  auto helper = getOrInsertLLVMFunc(
      op->getLoc(), module, rewriter, getClassHelperName(classType, suffix),
      LLVM::LLVMVoidType::get(rewriter.getContext()), {ptrType});
  auto helperRef = SymbolRefAttr::get(module.getContext(), helper.getName());
  rewriter.create<LLVM::CallOp>(op->getLoc(), TypeRange{}, helperRef,
                                ValueRange{object});
  return success();
}

struct IncRefLowering : public OpConversionPattern<IncRefOp> {
  IncRefLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<IncRefOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(IncRefOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.setInsertionPoint(op);
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    if (isCompilerOwnedMemRefListType(op.getObject().getType()) ||
        isCompilerOwnedMemRefDictType(op.getObject().getType()) ||
        isCompilerOwnedMemRefTupleType(op.getObject().getType())) {
      rewriter.eraseOp(op);
      return success();
    }
    if (auto classType = dyn_cast<ClassType>(op.getObject().getType())) {
      (void)typeConverter;
      if (failed(emitStaticClassRefCountCall(op, adaptor.getObject(), classType,
                                             module, rewriter, "incref")))
        return failure();
      rewriter.eraseOp(op);
      return success();
    }
    RuntimeAPI runtime(module, rewriter, *typeConverter);

    runtime.call(op.getLoc(), RuntimeSymbols::kIncRef, /*resultType=*/nullptr,
                 adaptor.getObject());
    rewriter.eraseOp(op);
    return success();
  }
};

struct DecRefLowering : public OpConversionPattern<DecRefOp> {
  DecRefLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<DecRefOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(DecRefOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.setInsertionPoint(op);
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    if (isCompilerOwnedMemRefListType(op.getObject().getType()) ||
        isCompilerOwnedMemRefDictType(op.getObject().getType()) ||
        isCompilerOwnedMemRefTupleType(op.getObject().getType())) {
      rewriter.eraseOp(op);
      return success();
    }
    if (auto classType = dyn_cast<ClassType>(op.getObject().getType())) {
      (void)typeConverter;
      StringRef suffix = op->hasAttr("lython.final_local_class_decref")
                             ? "destroy_local"
                             : "decref";
      if (failed(emitStaticClassRefCountCall(op, adaptor.getObject(), classType,
                                             module, rewriter, suffix)))
        return failure();
      rewriter.eraseOp(op);
      return success();
    }
    RuntimeAPI runtime(module, rewriter, *typeConverter);

    runtime.call(op.getLoc(), RuntimeSymbols::kDecRef, /*resultType=*/nullptr,
                 adaptor.getObject());
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populatePyRefCountLoweringPatterns(PyLLVMTypeConverter &typeConverter,
                                        RewritePatternSet &patterns) {
  auto *ctx = patterns.getContext();
  patterns.add<IncRefLowering, DecRefLowering>(typeConverter, ctx);
}

} // namespace py
