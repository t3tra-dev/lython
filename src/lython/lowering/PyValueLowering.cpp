#include "RuntimeSupport.h"

#include <cerrno>
#include <cstdlib>

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

using namespace mlir;

namespace py {
namespace {

struct StrConstantLowering : public OpConversionPattern<StrConstantOp> {
  StrConstantLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<StrConstantOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(StrConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *typeConverter);

    Type resultType = typeConverter->convertType(op.getResult().getType());
    if (!resultType)
      return failure();

    Value dataPtr = runtime.getStringLiteral(op.getLoc(), op.getValueAttr());
    Value length = runtime.getI64Constant(op.getLoc(),
                                          op.getValueAttr().getValue().size());

    auto call = runtime.call(op.getLoc(), RuntimeSymbols::kStrFromUtf8,
                             resultType, ValueRange{dataPtr, length});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};

struct NoneLowering : public OpConversionPattern<NoneOp> {
  NoneLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<NoneOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(NoneOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *typeConverter);
    Type resultType = typeConverter->convertType(op.getResult().getType());
    auto call = runtime.call(op.getLoc(), RuntimeSymbols::kGetNone, resultType,
                             ValueRange{});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};

struct IntConstantLowering : public OpConversionPattern<IntConstantOp> {
  IntConstantLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<IntConstantOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(IntConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *typeConverter);
    Type resultType = typeConverter->convertType(op.getResult().getType());
    if (!resultType)
      return failure();

    // Integer value is stored as decimal string to support arbitrary precision
    StringRef valueStr = op.getValue();

    // Hybrid approach: try to parse as int64_t for fast path
    // Max int64_t is 19 digits, so only try for shorter strings
    if (valueStr.size() <= 19) {
      char *end;
      errno = 0;
      long long parsed = std::strtoll(valueStr.data(), &end, 10);
      // Check if parsing succeeded: entire string consumed, no overflow
      if (end == valueStr.data() + valueStr.size() && errno != ERANGE) {
        // Use fast LyLong_FromI64 path
        Value value =
            runtime.getI64Constant(op.getLoc(), static_cast<int64_t>(parsed));
        auto call = runtime.call(op.getLoc(), RuntimeSymbols::kLongFromI64,
                                 resultType, ValueRange{value});
        rewriter.replaceOp(op, call.getResults());
        return success();
      }
    }

    // Fall back to string-based creation for arbitrary precision
    Value dataPtr = runtime.getStringLiteral(
        op.getLoc(), StringAttr::get(rewriter.getContext(), valueStr));
    Value length = runtime.getI64Constant(
        op.getLoc(), static_cast<int64_t>(valueStr.size()));
    auto call = runtime.call(op.getLoc(), RuntimeSymbols::kLongFromString,
                             resultType, ValueRange{dataPtr, length});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};

struct FloatConstantLowering : public OpConversionPattern<FloatConstantOp> {
  FloatConstantLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<FloatConstantOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(FloatConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *typeConverter);
    Type resultType = typeConverter->convertType(op.getResult().getType());
    if (!resultType)
      return failure();
    Value value =
        runtime.getF64Constant(op.getLoc(), op.getValue().convertToDouble());
    auto call = runtime.call(op.getLoc(), RuntimeSymbols::kFloatFromDouble,
                             resultType, ValueRange{value});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};

// Lowering numeric operations to runtime calls
template <typename OpT>
struct NumericBinaryLowering : public OpConversionPattern<OpT> {
  NumericBinaryLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx,
                        llvm::StringLiteral symbol)
      : OpConversionPattern<OpT>(converter, ctx), symbol(symbol) {}

  LogicalResult
  matchAndRewrite(OpT op, typename OpConversionPattern<OpT>::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->template getParentOfType<ModuleOp>();
    if (!module)
      return failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(this->getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *typeConverter);
    Type resultType = typeConverter->convertType(op.getResult().getType());
    if (!resultType)
      return failure();
    auto call = runtime.call(op.getLoc(), symbol, resultType,
                             ValueRange{adaptor.getOperands()});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  llvm::StringLiteral symbol;
};

// Type-specialized lowering for NumAddOp
struct NumAddLowering : public OpConversionPattern<NumAddOp> {
  NumAddLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<NumAddOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(NumAddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *typeConverter);
    Type resultType = typeConverter->convertType(op.getResult().getType());
    if (!resultType)
      return failure();

    // Use type-specialized function for int operands
    llvm::StringRef symbol =
        isPyIntType(op.getLhs().getType()) && isPyIntType(op.getRhs().getType())
            ? RuntimeSymbols::kLongAdd
            : RuntimeSymbols::kNumberAdd;
    auto call =
        runtime.call(op.getLoc(), symbol, resultType, adaptor.getOperands());
    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};

// Type-specialized lowering for NumSubOp
struct NumSubLowering : public OpConversionPattern<NumSubOp> {
  NumSubLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<NumSubOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(NumSubOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *typeConverter);
    Type resultType = typeConverter->convertType(op.getResult().getType());
    if (!resultType)
      return failure();

    // Use type-specialized function for int operands
    llvm::StringRef symbol =
        isPyIntType(op.getLhs().getType()) && isPyIntType(op.getRhs().getType())
            ? RuntimeSymbols::kLongSub
            : RuntimeSymbols::kNumberSub;
    auto call =
        runtime.call(op.getLoc(), symbol, resultType, adaptor.getOperands());
    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};

// Type-specialized lowering for NumLeOp
struct NumLeLowering : public OpConversionPattern<NumLeOp> {
  NumLeLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<NumLeOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(NumLeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *typeConverter);
    Type resultType = typeConverter->convertType(op.getResult().getType());
    if (!resultType)
      return failure();

    // For int operands, use LyLong_Compare + LyBool_FromBool
    if (isPyIntType(op.getLhs().getType()) &&
        isPyIntType(op.getRhs().getType())) {
      // LyLong_Compare returns int: -1 (less), 0 (equal), 1 (greater)
      auto cmpCall = runtime.call(op.getLoc(), RuntimeSymbols::kLongCompare,
                                  rewriter.getI32Type(), adaptor.getOperands());
      // For <=: compare result <= 0
      Value zero = rewriter.create<LLVM::ConstantOp>(
          op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
      Value leZero = rewriter.create<LLVM::ICmpOp>(
          op.getLoc(), LLVM::ICmpPredicate::sle, cmpCall.getResult(), zero);
      // Convert i1 to LyBool
      auto boolCall = runtime.call(op.getLoc(), RuntimeSymbols::kBoolFromBool,
                                   resultType, ValueRange{leZero});
      rewriter.replaceOp(op, boolCall.getResults());
    } else {
      auto call = runtime.call(op.getLoc(), RuntimeSymbols::kNumberLe,
                               resultType, adaptor.getOperands());
      rewriter.replaceOp(op, call.getResults());
    }
    return success();
  }
};

struct CastToPrimLowering : public OpConversionPattern<CastToPrimOp> {
  CastToPrimLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<CastToPrimOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(CastToPrimOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getResult().getType() != rewriter.getI1Type())
      return rewriter.notifyMatchFailure(op, "only bool casts supported");
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *typeConverter);
    auto call = runtime.call(op.getLoc(), RuntimeSymbols::kBoolAsBool,
                             rewriter.getI1Type(), adaptor.getInput());
    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};

struct CastFromPrimLowering : public OpConversionPattern<CastFromPrimOp> {
  CastFromPrimLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<CastFromPrimOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(CastFromPrimOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *typeConverter);
    Type resultType = typeConverter->convertType(op.getResult().getType());
    if (!resultType)
      return failure();

    Value input = adaptor.getInput();
    Type inputType = input.getType();

    // Handle integer types -> !py.int
    if (auto intType = llvm::dyn_cast<IntegerType>(inputType)) {
      unsigned width = intType.getWidth();
      Value i64Val;

      if (width < 64) {
        // Sign-extend to i64
        auto i64Type = rewriter.getI64Type();
        i64Val = rewriter.create<LLVM::SExtOp>(op.getLoc(), i64Type, input);
      } else if (width == 64) {
        i64Val = input;
      } else {
        return rewriter.notifyMatchFailure(
            op, "integer width > 64 not supported for from_prim");
      }

      auto call = runtime.call(op.getLoc(), RuntimeSymbols::kLongFromI64,
                               resultType, ValueRange{i64Val});
      rewriter.replaceOp(op, call.getResults());
      return success();
    }

    // Handle float types -> !py.float
    if (llvm::isa<Float32Type>(inputType)) {
      // Extend f32 to f64
      auto f64Type = rewriter.getF64Type();
      Value f64Val =
          rewriter.create<LLVM::FPExtOp>(op.getLoc(), f64Type, input);
      auto call = runtime.call(op.getLoc(), RuntimeSymbols::kFloatFromDouble,
                               resultType, ValueRange{f64Val});
      rewriter.replaceOp(op, call.getResults());
      return success();
    }

    if (llvm::isa<Float64Type>(inputType)) {
      auto call = runtime.call(op.getLoc(), RuntimeSymbols::kFloatFromDouble,
                               resultType, ValueRange{input});
      rewriter.replaceOp(op, call.getResults());
      return success();
    }

    return rewriter.notifyMatchFailure(op,
                                       "unsupported input type for from_prim");
  }
};

// For now, class instances are represented as dictionaries.
// ClassNewOp creates a new empty dictionary as instance storage.
struct ClassNewLowering : public OpConversionPattern<ClassNewOp> {
  ClassNewLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<ClassNewOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(ClassNewOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *typeConverter);
    Type resultType = typeConverter->convertType(op.getResult().getType());
    if (!resultType)
      return failure();
    auto call = runtime.call(op.getLoc(), RuntimeSymbols::kDictNew, resultType,
                             ValueRange{});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};

// AttrGetOp lowers to dict lookup
struct AttrGetLowering : public OpConversionPattern<AttrGetOp> {
  AttrGetLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<AttrGetOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(AttrGetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *typeConverter);
    Type resultType = typeConverter->convertType(op.getResult().getType());
    if (!resultType)
      return failure();

    // Create string key for attribute name
    StringRef attrName = op.getNameAttr().getValue();
    Value namePtr = runtime.getStringLiteral(
        op.getLoc(), StringAttr::get(rewriter.getContext(), attrName));
    Value nameLen = runtime.getI64Constant(
        op.getLoc(), static_cast<int64_t>(attrName.size()));
    auto keyCall = runtime.call(op.getLoc(), RuntimeSymbols::kStrFromUtf8,
                                typeConverter->getPyObjectPtrType(),
                                ValueRange{namePtr, nameLen});

    // Lookup in dict
    auto getCall =
        runtime.call(op.getLoc(), RuntimeSymbols::kDictGetItem, resultType,
                     ValueRange{adaptor.getObject(), keyCall.getResult()});
    rewriter.replaceOp(op, getCall.getResults());
    return success();
  }
};

// AttrSetOp lowers to dict insert
struct AttrSetLowering : public OpConversionPattern<AttrSetOp> {
  AttrSetLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<AttrSetOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(AttrSetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *typeConverter);

    // Create string key for attribute name
    StringRef attrName = op.getNameAttr().getValue();
    Value namePtr = runtime.getStringLiteral(
        op.getLoc(), StringAttr::get(rewriter.getContext(), attrName));
    Value nameLen = runtime.getI64Constant(
        op.getLoc(), static_cast<int64_t>(attrName.size()));
    auto keyCall = runtime.call(op.getLoc(), RuntimeSymbols::kStrFromUtf8,
                                typeConverter->getPyObjectPtrType(),
                                ValueRange{namePtr, nameLen});

    // Insert into dict
    runtime.call(op.getLoc(), RuntimeSymbols::kDictInsert,
                 typeConverter->getPyObjectPtrType(),
                 ValueRange{adaptor.getObject(), keyCall.getResult(),
                            adaptor.getValue()});
    rewriter.eraseOp(op);
    return success();
  }
};

// ClassOp is just a container - erase it and keep its methods at module scope
struct ClassOpLowering : public OpConversionPattern<ClassOp> {
  ClassOpLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<ClassOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(ClassOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Just erase the class op - methods are already at module scope
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populatePyValueLoweringPatterns(PyLLVMTypeConverter &typeConverter,
                                     RewritePatternSet &patterns) {
  auto *ctx = patterns.getContext();
  patterns
      .add<StrConstantLowering, NoneLowering, IntConstantLowering,
           FloatConstantLowering, NumAddLowering, NumSubLowering, NumLeLowering,
           CastToPrimLowering, CastFromPrimLowering, ClassNewLowering,
           AttrGetLowering, AttrSetLowering, ClassOpLowering>(typeConverter,
                                                              ctx);
}

} // namespace py
