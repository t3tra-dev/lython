#include "Common/RuntimeSupport.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"

#include <cerrno>
#include <cstdlib>
#include <string>

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

namespace py {
namespace {

static bool formatPrimitiveAttr(mlir::Attribute attr, std::string &out) {
  if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(attr)) {
    out = llvm::toString(intAttr.getValue(), 10, true);
    return true;
  }
  if (auto floatAttr = llvm::dyn_cast<mlir::FloatAttr>(attr)) {
    llvm::SmallString<16> buffer;
    floatAttr.getValue().toString(buffer);
    out = buffer.str().str();
    return true;
  }
  return false;
}

static bool collectTensorLiteralElements(mlir::Value input,
                                         llvm::SmallVector<std::string> &flat) {
  if (auto fromElems = llvm::dyn_cast_or_null<mlir::tensor::FromElementsOp>(
          input.getDefiningOp())) {
    for (mlir::Value element : fromElems.getElements()) {
      auto constOp = element.getDefiningOp<mlir::arith::ConstantOp>();
      if (!constOp)
        return false;
      std::string valueStr;
      if (!formatPrimitiveAttr(constOp.getValue(), valueStr))
        return false;
      flat.push_back(valueStr);
    }
    return true;
  }

  if (auto constOp = input.getDefiningOp<mlir::arith::ConstantOp>()) {
    if (auto dense =
            llvm::dyn_cast<mlir::DenseElementsAttr>(constOp.getValue())) {
      for (auto val : dense.getValues<mlir::Attribute>()) {
        std::string valueStr;
        if (!formatPrimitiveAttr(val, valueStr))
          return false;
        flat.push_back(valueStr);
      }
      return true;
    }
  }

  return false;
}

static bool collectFloatLiteral(mlir::Attribute attr, double &out) {
  if (auto floatAttr = llvm::dyn_cast<mlir::FloatAttr>(attr)) {
    out = floatAttr.getValue().convertToDouble();
    return true;
  }
  if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(attr)) {
    out = intAttr.getValue().signedRoundToDouble();
    return true;
  }
  return false;
}

static bool collectScalarConstant(mlir::Value value, double &out) {
  auto constOp = value.getDefiningOp<mlir::arith::ConstantOp>();
  return constOp && collectFloatLiteral(constOp.getValue(), out);
}

static int64_t getStaticElementCount(mlir::RankedTensorType type) {
  if (!type || !type.hasStaticShape())
    return -1;
  int64_t count = 1;
  for (int64_t dim : type.getShape())
    count *= dim;
  return count;
}

static bool collectTensorLiteralNumbers(mlir::Value input,
                                        llvm::SmallVector<double> &flat);

static bool collectElementwiseTensorLiteral(
    mlir::Value lhs, mlir::Value rhs, llvm::SmallVector<double> &flat,
    llvm::function_ref<double(double, double)> apply) {
  llvm::SmallVector<double> lhsFlat;
  llvm::SmallVector<double> rhsFlat;
  if (!collectTensorLiteralNumbers(lhs, lhsFlat) ||
      !collectTensorLiteralNumbers(rhs, rhsFlat) ||
      lhsFlat.size() != rhsFlat.size())
    return false;

  flat.reserve(flat.size() + lhsFlat.size());
  for (auto [lhsValue, rhsValue] : llvm::zip_equal(lhsFlat, rhsFlat))
    flat.push_back(apply(lhsValue, rhsValue));
  return true;
}

static bool isGeneratedMatmulBody(mlir::linalg::GenericOp op) {
  if (op.getRegion().empty())
    return false;
  mlir::Block &body = op.getRegion().front();
  if (body.getNumArguments() != 3)
    return false;
  auto *terminator = body.getTerminator();
  if (!terminator || terminator->getNumOperands() != 1)
    return false;
  auto add = terminator->getOperand(0).getDefiningOp<mlir::arith::AddFOp>();
  if (!add)
    return false;

  auto isMulOfInputs = [&](mlir::Value value) {
    auto mul = value.getDefiningOp<mlir::arith::MulFOp>();
    if (!mul)
      return false;
    return ((mul.getLhs() == body.getArgument(0) &&
             mul.getRhs() == body.getArgument(1)) ||
            (mul.getLhs() == body.getArgument(1) &&
             mul.getRhs() == body.getArgument(0)));
  };

  return (isMulOfInputs(add.getLhs()) && add.getRhs() == body.getArgument(2)) ||
         (isMulOfInputs(add.getRhs()) && add.getLhs() == body.getArgument(2));
}

static bool collectGeneratedMatmulLiteral(mlir::linalg::GenericOp op,
                                          llvm::SmallVector<double> &flat) {
  if (!isGeneratedMatmulBody(op))
    return false;
  if (op.getInputs().size() != 2 || op.getOutputs().size() != 1 ||
      op->getNumResults() != 1)
    return false;

  auto lhsType =
      llvm::dyn_cast<mlir::RankedTensorType>(op.getInputs()[0].getType());
  auto rhsType =
      llvm::dyn_cast<mlir::RankedTensorType>(op.getInputs()[1].getType());
  auto outType =
      llvm::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!lhsType || !rhsType || !outType || lhsType.getRank() != 2 ||
      rhsType.getRank() != 2 || outType.getRank() != 2)
    return false;

  int64_t m = lhsType.getDimSize(0);
  int64_t k = lhsType.getDimSize(1);
  int64_t rhsK = rhsType.getDimSize(0);
  int64_t n = rhsType.getDimSize(1);
  if (k != rhsK || outType.getDimSize(0) != m || outType.getDimSize(1) != n)
    return false;

  llvm::SmallVector<double> lhsFlat;
  llvm::SmallVector<double> rhsFlat;
  llvm::SmallVector<double> initFlat;
  if (!collectTensorLiteralNumbers(op.getInputs()[0], lhsFlat) ||
      !collectTensorLiteralNumbers(op.getInputs()[1], rhsFlat) ||
      !collectTensorLiteralNumbers(op.getOutputs()[0], initFlat))
    return false;

  flat.reserve(flat.size() + static_cast<size_t>(m * n));
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      double acc = initFlat[static_cast<size_t>(i * n + j)];
      for (int64_t kk = 0; kk < k; ++kk) {
        double lhsValue = lhsFlat[static_cast<size_t>(i * k + kk)];
        double rhsValue = rhsFlat[static_cast<size_t>(kk * n + j)];
        acc += lhsValue * rhsValue;
      }
      flat.push_back(acc);
    }
  }
  return true;
}

static bool collectTensorLiteralNumbers(mlir::Value input,
                                        llvm::SmallVector<double> &flat) {
  auto tensorType = llvm::dyn_cast<mlir::RankedTensorType>(input.getType());
  if (!tensorType || !tensorType.hasStaticShape())
    return false;

  mlir::Operation *def = input.getDefiningOp();
  if (!def)
    return false;

  if (auto fromElems = llvm::dyn_cast<mlir::tensor::FromElementsOp>(def)) {
    for (mlir::Value element : fromElems.getElements()) {
      double value = 0.0;
      if (!collectScalarConstant(element, value))
        return false;
      flat.push_back(value);
    }
    return static_cast<int64_t>(flat.size()) ==
           getStaticElementCount(tensorType);
  }

  if (auto constOp = llvm::dyn_cast<mlir::arith::ConstantOp>(def)) {
    auto dense = llvm::dyn_cast<mlir::DenseElementsAttr>(constOp.getValue());
    if (!dense)
      return false;
    for (mlir::Attribute valueAttr : dense.getValues<mlir::Attribute>()) {
      double value = 0.0;
      if (!collectFloatLiteral(valueAttr, value))
        return false;
      flat.push_back(value);
    }
    return static_cast<int64_t>(flat.size()) ==
           getStaticElementCount(tensorType);
  }

  if (auto add = llvm::dyn_cast<mlir::arith::AddFOp>(def))
    return collectElementwiseTensorLiteral(
        add.getLhs(), add.getRhs(), flat,
        [](double lhs, double rhs) { return lhs + rhs; });
  if (auto sub = llvm::dyn_cast<mlir::arith::SubFOp>(def))
    return collectElementwiseTensorLiteral(
        sub.getLhs(), sub.getRhs(), flat,
        [](double lhs, double rhs) { return lhs - rhs; });
  if (auto mul = llvm::dyn_cast<mlir::arith::MulFOp>(def))
    return collectElementwiseTensorLiteral(
        mul.getLhs(), mul.getRhs(), flat,
        [](double lhs, double rhs) { return lhs * rhs; });

  if (auto fill = llvm::dyn_cast<mlir::linalg::FillOp>(def)) {
    if (fill.getInputs().size() != 1 || fill->getNumResults() != 1)
      return false;
    double value = 0.0;
    if (!collectScalarConstant(fill.getInputs().front(), value))
      return false;
    int64_t count = getStaticElementCount(tensorType);
    if (count < 0)
      return false;
    flat.append(static_cast<size_t>(count), value);
    return true;
  }

  if (auto generic = llvm::dyn_cast<mlir::linalg::GenericOp>(def))
    return collectGeneratedMatmulLiteral(generic, flat);

  return false;
}

static llvm::SmallVector<std::string>
formatTensorLiteralNumbers(llvm::ArrayRef<double> values) {
  llvm::SmallVector<std::string> result;
  result.reserve(values.size());
  for (double value : values) {
    llvm::SmallString<24> buffer;
    llvm::APFloat(value).toString(buffer);
    result.push_back(buffer.str().str());
  }
  return result;
}

static bool isErasableTensorProducer(mlir::Operation *op) {
  return llvm::isa<mlir::tensor::FromElementsOp, mlir::tensor::EmptyOp,
                   mlir::arith::ConstantOp, mlir::arith::AddFOp,
                   mlir::arith::SubFOp, mlir::arith::MulFOp,
                   mlir::linalg::FillOp, mlir::linalg::GenericOp>(op);
}

static void eraseDeadTensorProducerTree(mlir::Value value,
                                        mlir::PatternRewriter &rewriter) {
  mlir::Operation *def = value.getDefiningOp();
  if (!def || !value.use_empty() || !isErasableTensorProducer(def))
    return;

  llvm::SmallVector<mlir::Value> operands(def->getOperands());
  rewriter.eraseOp(def);
  for (mlir::Value operand : operands)
    eraseDeadTensorProducerTree(operand, rewriter);
}

static mlir::func::FuncOp getOrInsertTensorReprFunc(mlir::ModuleOp module,
                                                    llvm::StringRef name,
                                                    mlir::Type memrefType,
                                                    mlir::Type resultType) {
  auto markOwnedResult = [](mlir::func::FuncOp fn) {
    mlir::Builder builder(fn.getContext());
    fn->setAttr(OwnershipContractAttrs::kOwnedResults,
                builder.getArrayAttr({builder.getI64IntegerAttr(0)}));
  };
  if (auto fn = module.lookupSymbol<mlir::func::FuncOp>(name)) {
    markOwnedResult(fn);
    return fn;
  }

  mlir::OpBuilder builder(module.getContext());
  builder.setInsertionPointToStart(module.getBody());
  auto funcType =
      mlir::FunctionType::get(module.getContext(), {memrefType}, {resultType});
  auto func =
      builder.create<mlir::func::FuncOp>(module.getLoc(), name, funcType);
  func->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
  func.setVisibility(mlir::SymbolTable::Visibility::Private);
  markOwnedResult(func);
  return func;
}

static std::string formatTensorLiteral(llvm::ArrayRef<int64_t> shape,
                                       llvm::ArrayRef<std::string> flat,
                                       size_t &index) {
  if (shape.empty()) {
    if (index >= flat.size())
      return "<invalid>";
    return flat[index++];
  }

  std::string result = "[";
  for (int64_t i = 0; i < shape.front(); ++i) {
    if (i > 0)
      result.append(", ");
    result.append(formatTensorLiteral(shape.drop_front(), flat, index));
  }
  result.push_back(']');
  return result;
}

struct StrConstantLowering : public mlir::OpConversionPattern<StrConstantOp> {
  StrConstantLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<StrConstantOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(StrConstantOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return mlir::failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *typeConverter);

    mlir::Type resultType =
        typeConverter->convertType(op.getResult().getType());
    if (!resultType)
      return mlir::failure();

    mlir::Value dataPtr =
        runtime.getStringLiteral(op.getLoc(), op.getValueAttr());
    mlir::Value length = runtime.getI64Constant(
        op.getLoc(), op.getValueAttr().getValue().size());

    auto call = runtime.call(op.getLoc(), RuntimeSymbols::kStrInternStaticUtf8,
                             resultType, mlir::ValueRange{dataPtr, length});
    rewriter.replaceOp(op, call.getResults());
    return mlir::success();
  }
};

struct NoneLowering : public mlir::OpConversionPattern<NoneOp> {
  NoneLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<NoneOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(NoneOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return mlir::failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *typeConverter);
    mlir::Type resultType =
        typeConverter->convertType(op.getResult().getType());
    auto call = runtime.call(op.getLoc(), RuntimeSymbols::kGetNone, resultType,
                             mlir::ValueRange{});
    rewriter.replaceOp(op, call.getResults());
    return mlir::success();
  }
};

struct IntConstantLowering : public mlir::OpConversionPattern<IntConstantOp> {
  IntConstantLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<IntConstantOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(IntConstantOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return mlir::failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *typeConverter);
    mlir::Type resultType =
        typeConverter->convertType(op.getResult().getType());
    if (!resultType)
      return mlir::failure();

    // Integer value is stored as decimal string to support arbitrary precision
    llvm::StringRef valueStr = op.getValue();

    // Hybrid approach: try to parse as int64_t for fast path
    // Max int64_t is 19 digits, so only try for shorter strings
    if (valueStr.size() <= 19) {
      char *end;
      errno = 0;
      long long parsed = std::strtoll(valueStr.data(), &end, 10);
      // Check if parsing succeeded: entire string consumed, no overflow
      if (end == valueStr.data() + valueStr.size() && errno != ERANGE) {
        // Use fast LyLong_FromI64 path
        mlir::Value value =
            runtime.getI64Constant(op.getLoc(), static_cast<int64_t>(parsed));
        auto call = runtime.call(op.getLoc(), RuntimeSymbols::kLongFromI64,
                                 resultType, mlir::ValueRange{value});
        rewriter.replaceOp(op, call.getResults());
        return mlir::success();
      }
    }

    // Fall back to string-based creation for arbitrary precision
    mlir::Value dataPtr = runtime.getStringLiteral(
        op.getLoc(), mlir::StringAttr::get(rewriter.getContext(), valueStr));
    mlir::Value length = runtime.getI64Constant(
        op.getLoc(), static_cast<int64_t>(valueStr.size()));
    auto call = runtime.call(op.getLoc(), RuntimeSymbols::kLongFromString,
                             resultType, mlir::ValueRange{dataPtr, length});
    rewriter.replaceOp(op, call.getResults());
    return mlir::success();
  }
};

struct FloatConstantLowering
    : public mlir::OpConversionPattern<FloatConstantOp> {
  FloatConstantLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<FloatConstantOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(FloatConstantOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return mlir::failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *typeConverter);
    mlir::Type resultType =
        typeConverter->convertType(op.getResult().getType());
    if (!resultType)
      return mlir::failure();
    mlir::Value value =
        runtime.getF64Constant(op.getLoc(), op.getValue().convertToDouble());
    auto call = runtime.call(op.getLoc(), RuntimeSymbols::kFloatFromDouble,
                             resultType, mlir::ValueRange{value});
    rewriter.replaceOp(op, call.getResults());
    return mlir::success();
  }
};

// Lowering numeric operations to runtime calls
template <typename OpT>
struct NumericBinaryLowering : public mlir::OpConversionPattern<OpT> {
  NumericBinaryLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx,
                        llvm::StringLiteral symbol)
      : mlir::OpConversionPattern<OpT>(converter, ctx), symbol(symbol) {}

  mlir::LogicalResult
  matchAndRewrite(OpT op,
                  typename mlir::OpConversionPattern<OpT>::OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::ModuleOp module = op->template getParentOfType<mlir::ModuleOp>();
    if (!module)
      return mlir::failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(this->getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *typeConverter);
    mlir::Type resultType =
        typeConverter->convertType(op.getResult().getType());
    if (!resultType)
      return mlir::failure();
    auto call = runtime.call(op.getLoc(), symbol, resultType,
                             mlir::ValueRange{adaptor.getOperands()});
    rewriter.replaceOp(op, call.getResults());
    return mlir::success();
  }

private:
  llvm::StringLiteral symbol;
};

// mlir::Type-specialized lowering for AddOp
struct AddLowering : public mlir::OpConversionPattern<AddOp> {
  AddLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<AddOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(AddOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return mlir::failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *typeConverter);
    mlir::Type resultType =
        typeConverter->convertType(op.getResult().getType());
    if (!resultType)
      return mlir::failure();

    // Use type-specialized function for int operands
    llvm::StringRef symbol;
    if (isPyIntType(op.getLhs().getType()) &&
        isPyIntType(op.getRhs().getType()))
      symbol = RuntimeSymbols::kLongAdd;
    else if (isPyStrType(op.getLhs().getType()) &&
             isPyStrType(op.getRhs().getType()))
      symbol = RuntimeSymbols::kUnicodeConcat;
    else
      symbol = RuntimeSymbols::kNumberAdd;
    auto call =
        runtime.call(op.getLoc(), symbol, resultType, adaptor.getOperands());
    rewriter.replaceOp(op, call.getResults());
    return mlir::success();
  }
};

// mlir::Type-specialized lowering for SubOp
struct SubLowering : public mlir::OpConversionPattern<SubOp> {
  SubLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<SubOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(SubOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return mlir::failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *typeConverter);
    mlir::Type resultType =
        typeConverter->convertType(op.getResult().getType());
    if (!resultType)
      return mlir::failure();

    // Use type-specialized function for int operands
    llvm::StringRef symbol =
        isPyIntType(op.getLhs().getType()) && isPyIntType(op.getRhs().getType())
            ? RuntimeSymbols::kLongSub
            : RuntimeSymbols::kNumberSub;
    auto call =
        runtime.call(op.getLoc(), symbol, resultType, adaptor.getOperands());
    rewriter.replaceOp(op, call.getResults());
    return mlir::success();
  }
};

// mlir::Type-specialized lowering for LeOp
struct LeLowering : public mlir::OpConversionPattern<LeOp> {
  LeLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<LeOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(LeOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return mlir::failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *typeConverter);
    mlir::Type resultType =
        typeConverter->convertType(op.getResult().getType());
    if (!resultType)
      return mlir::failure();

    // For int operands, use LyLong_Compare + LyBool_FromBool
    if (isPyIntType(op.getLhs().getType()) &&
        isPyIntType(op.getRhs().getType())) {
      // LyLong_Compare returns int: -1 (less), 0 (equal), 1 (greater)
      auto cmpCall = runtime.call(op.getLoc(), RuntimeSymbols::kLongCompare,
                                  rewriter.getI32Type(), adaptor.getOperands());
      // For <=: compare result <= 0
      mlir::Value zero = rewriter.create<mlir::LLVM::ConstantOp>(
          op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
      mlir::Value leZero = rewriter.create<mlir::LLVM::ICmpOp>(
          op.getLoc(), mlir::LLVM::ICmpPredicate::sle, cmpCall.getResult(),
          zero);
      // Convert i1 to LyBool
      auto boolCall = runtime.call(op.getLoc(), RuntimeSymbols::kBoolFromBool,
                                   resultType, mlir::ValueRange{leZero});
      rewriter.replaceOp(op, boolCall.getResults());
    } else {
      auto call = runtime.call(op.getLoc(), RuntimeSymbols::kNumberLe,
                               resultType, adaptor.getOperands());
      rewriter.replaceOp(op, call.getResults());
    }
    return mlir::success();
  }
};

// mlir::Type-specialized lowering for LtOp
struct LtLowering : public mlir::OpConversionPattern<LtOp> {
  LtLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<LtOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(LtOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return mlir::failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *typeConverter);
    mlir::Type resultType =
        typeConverter->convertType(op.getResult().getType());
    if (!resultType)
      return mlir::failure();

    // For int operands, use LyLong_Compare + LyBool_FromBool
    if (isPyIntType(op.getLhs().getType()) &&
        isPyIntType(op.getRhs().getType())) {
      // LyLong_Compare returns int: -1 (less), 0 (equal), 1 (greater)
      auto cmpCall = runtime.call(op.getLoc(), RuntimeSymbols::kLongCompare,
                                  rewriter.getI32Type(), adaptor.getOperands());
      // For <: compare result < 0
      mlir::Value zero = rewriter.create<mlir::LLVM::ConstantOp>(
          op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
      mlir::Value ltZero = rewriter.create<mlir::LLVM::ICmpOp>(
          op.getLoc(), mlir::LLVM::ICmpPredicate::slt, cmpCall.getResult(),
          zero);
      // Convert i1 to LyBool
      auto boolCall = runtime.call(op.getLoc(), RuntimeSymbols::kBoolFromBool,
                                   resultType, mlir::ValueRange{ltZero});
      rewriter.replaceOp(op, boolCall.getResults());
    } else {
      auto call = runtime.call(op.getLoc(), RuntimeSymbols::kNumberLt,
                               resultType, adaptor.getOperands());
      rewriter.replaceOp(op, call.getResults());
    }
    return mlir::success();
  }
};

// mlir::Type-specialized lowering for GtOp
struct GtLowering : public mlir::OpConversionPattern<GtOp> {
  GtLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<GtOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(GtOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return mlir::failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *typeConverter);
    mlir::Type resultType =
        typeConverter->convertType(op.getResult().getType());
    if (!resultType)
      return mlir::failure();

    // For int operands, use LyLong_Compare + LyBool_FromBool
    if (isPyIntType(op.getLhs().getType()) &&
        isPyIntType(op.getRhs().getType())) {
      // LyLong_Compare returns int: -1 (less), 0 (equal), 1 (greater)
      auto cmpCall = runtime.call(op.getLoc(), RuntimeSymbols::kLongCompare,
                                  rewriter.getI32Type(), adaptor.getOperands());
      // For >: compare result > 0
      mlir::Value zero = rewriter.create<mlir::LLVM::ConstantOp>(
          op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
      mlir::Value gtZero = rewriter.create<mlir::LLVM::ICmpOp>(
          op.getLoc(), mlir::LLVM::ICmpPredicate::sgt, cmpCall.getResult(),
          zero);
      // Convert i1 to LyBool
      auto boolCall = runtime.call(op.getLoc(), RuntimeSymbols::kBoolFromBool,
                                   resultType, mlir::ValueRange{gtZero});
      rewriter.replaceOp(op, boolCall.getResults());
    } else {
      auto call = runtime.call(op.getLoc(), RuntimeSymbols::kNumberGt,
                               resultType, adaptor.getOperands());
      rewriter.replaceOp(op, call.getResults());
    }
    return mlir::success();
  }
};

// mlir::Type-specialized lowering for GeOp
struct GeLowering : public mlir::OpConversionPattern<GeOp> {
  GeLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<GeOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(GeOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return mlir::failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *typeConverter);
    mlir::Type resultType =
        typeConverter->convertType(op.getResult().getType());
    if (!resultType)
      return mlir::failure();

    // For int operands, use LyLong_Compare + LyBool_FromBool
    if (isPyIntType(op.getLhs().getType()) &&
        isPyIntType(op.getRhs().getType())) {
      // LyLong_Compare returns int: -1 (less), 0 (equal), 1 (greater)
      auto cmpCall = runtime.call(op.getLoc(), RuntimeSymbols::kLongCompare,
                                  rewriter.getI32Type(), adaptor.getOperands());
      // For >=: compare result >= 0
      mlir::Value zero = rewriter.create<mlir::LLVM::ConstantOp>(
          op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
      mlir::Value geZero = rewriter.create<mlir::LLVM::ICmpOp>(
          op.getLoc(), mlir::LLVM::ICmpPredicate::sge, cmpCall.getResult(),
          zero);
      // Convert i1 to LyBool
      auto boolCall = runtime.call(op.getLoc(), RuntimeSymbols::kBoolFromBool,
                                   resultType, mlir::ValueRange{geZero});
      rewriter.replaceOp(op, boolCall.getResults());
    } else {
      auto call = runtime.call(op.getLoc(), RuntimeSymbols::kNumberGe,
                               resultType, adaptor.getOperands());
      rewriter.replaceOp(op, call.getResults());
    }
    return mlir::success();
  }
};

// mlir::Type-specialized lowering for EqOp
struct EqLowering : public mlir::OpConversionPattern<EqOp> {
  EqLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<EqOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(EqOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return mlir::failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *typeConverter);
    mlir::Type resultType =
        typeConverter->convertType(op.getResult().getType());
    if (!resultType)
      return mlir::failure();

    // For int operands, use LyLong_Compare + LyBool_FromBool
    if (isPyIntType(op.getLhs().getType()) &&
        isPyIntType(op.getRhs().getType())) {
      // LyLong_Compare returns int: -1 (less), 0 (equal), 1 (greater)
      auto cmpCall = runtime.call(op.getLoc(), RuntimeSymbols::kLongCompare,
                                  rewriter.getI32Type(), adaptor.getOperands());
      // For ==: compare result == 0
      mlir::Value zero = rewriter.create<mlir::LLVM::ConstantOp>(
          op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
      mlir::Value eqZero = rewriter.create<mlir::LLVM::ICmpOp>(
          op.getLoc(), mlir::LLVM::ICmpPredicate::eq, cmpCall.getResult(),
          zero);
      // Convert i1 to LyBool
      auto boolCall = runtime.call(op.getLoc(), RuntimeSymbols::kBoolFromBool,
                                   resultType, mlir::ValueRange{eqZero});
      rewriter.replaceOp(op, boolCall.getResults());
    } else {
      auto call = runtime.call(op.getLoc(), RuntimeSymbols::kNumberEq,
                               resultType, adaptor.getOperands());
      rewriter.replaceOp(op, call.getResults());
    }
    return mlir::success();
  }
};

// mlir::Type-specialized lowering for NeOp
struct NeLowering : public mlir::OpConversionPattern<NeOp> {
  NeLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<NeOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(NeOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return mlir::failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *typeConverter);
    mlir::Type resultType =
        typeConverter->convertType(op.getResult().getType());
    if (!resultType)
      return mlir::failure();

    // For int operands, use LyLong_Compare + LyBool_FromBool
    if (isPyIntType(op.getLhs().getType()) &&
        isPyIntType(op.getRhs().getType())) {
      // LyLong_Compare returns int: -1 (less), 0 (equal), 1 (greater)
      auto cmpCall = runtime.call(op.getLoc(), RuntimeSymbols::kLongCompare,
                                  rewriter.getI32Type(), adaptor.getOperands());
      // For !=: compare result != 0
      mlir::Value zero = rewriter.create<mlir::LLVM::ConstantOp>(
          op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
      mlir::Value neZero = rewriter.create<mlir::LLVM::ICmpOp>(
          op.getLoc(), mlir::LLVM::ICmpPredicate::ne, cmpCall.getResult(),
          zero);
      // Convert i1 to LyBool
      auto boolCall = runtime.call(op.getLoc(), RuntimeSymbols::kBoolFromBool,
                                   resultType, mlir::ValueRange{neZero});
      rewriter.replaceOp(op, boolCall.getResults());
    } else {
      auto call = runtime.call(op.getLoc(), RuntimeSymbols::kNumberNe,
                               resultType, adaptor.getOperands());
      rewriter.replaceOp(op, call.getResults());
    }
    return mlir::success();
  }
};
struct CastToPrimLowering : public mlir::OpConversionPattern<CastToPrimOp> {
  CastToPrimLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<CastToPrimOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(CastToPrimOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return mlir::failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *typeConverter);
    mlir::Type resultType = op.getResult().getType();
    mlir::Type inputType = op.getInput().getType();

    if (inputType == BoolType::get(rewriter.getContext()) &&
        resultType == rewriter.getI1Type()) {
      auto call = runtime.call(op.getLoc(), RuntimeSymbols::kBoolAsBool,
                               rewriter.getI1Type(), adaptor.getInput());
      rewriter.replaceOp(op, call.getResults());
      return mlir::success();
    }

    if (inputType == IntType::get(rewriter.getContext())) {
      mlir::Value asI64 = runtime
                              .call(op.getLoc(), RuntimeSymbols::kLongAsI64,
                                    rewriter.getI64Type(), adaptor.getInput())
                              .getResult();
      if (resultType == rewriter.getI64Type()) {
        rewriter.replaceOp(op, asI64);
        return mlir::success();
      }
      if (resultType == rewriter.getI32Type()) {
        rewriter.replaceOpWithNewOp<mlir::LLVM::TruncOp>(op, resultType, asI64);
        return mlir::success();
      }
    }

    if (inputType == FloatType::get(rewriter.getContext()) &&
        resultType == rewriter.getF64Type()) {
      auto call = runtime.call(op.getLoc(), RuntimeSymbols::kFloatAsDouble,
                               rewriter.getF64Type(), adaptor.getInput());
      rewriter.replaceOp(op, call.getResults());
      return mlir::success();
    }

    return rewriter.notifyMatchFailure(op,
                                       "unsupported cast.to_prim conversion");
  }
};

struct CastFromPrimLowering : public mlir::OpConversionPattern<CastFromPrimOp> {
  CastFromPrimLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<CastFromPrimOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(CastFromPrimOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return mlir::failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *typeConverter);
    mlir::Type resultType =
        typeConverter->convertType(op.getResult().getType());
    if (!resultType)
      return mlir::failure();

    mlir::Value input = adaptor.getInput();
    mlir::Type inputType = input.getType();
    mlir::Type pyResultType = op.getResult().getType();

    if (inputType == rewriter.getI1Type() &&
        pyResultType == BoolType::get(rewriter.getContext())) {
      auto call = runtime.call(op.getLoc(), RuntimeSymbols::kBoolFromBool,
                               resultType, mlir::ValueRange{input});
      rewriter.replaceOp(op, call.getResults());
      return mlir::success();
    }

    // Handle integer types -> !py.int
    if (auto intType = llvm::dyn_cast<mlir::IntegerType>(inputType)) {
      unsigned width = intType.getWidth();
      mlir::Value i64Val;

      if (width < 64) {
        // Sign-extend to i64
        auto i64Type = rewriter.getI64Type();
        i64Val =
            rewriter.create<mlir::LLVM::SExtOp>(op.getLoc(), i64Type, input);
      } else if (width == 64) {
        i64Val = input;
      } else {
        return rewriter.notifyMatchFailure(
            op, "integer width > 64 not supported for from_prim");
      }

      auto call = runtime.call(op.getLoc(), RuntimeSymbols::kLongFromI64,
                               resultType, mlir::ValueRange{i64Val});
      rewriter.replaceOp(op, call.getResults());
      return mlir::success();
    }

    // Handle float types -> !py.float
    if (llvm::isa<mlir::Float32Type>(inputType)) {
      // Extend f32 to f64
      auto f64Type = rewriter.getF64Type();
      mlir::Value f64Val =
          rewriter.create<mlir::LLVM::FPExtOp>(op.getLoc(), f64Type, input);
      auto call = runtime.call(op.getLoc(), RuntimeSymbols::kFloatFromDouble,
                               resultType, mlir::ValueRange{f64Val});
      rewriter.replaceOp(op, call.getResults());
      return mlir::success();
    }

    if (llvm::isa<mlir::Float64Type>(inputType)) {
      auto call = runtime.call(op.getLoc(), RuntimeSymbols::kFloatFromDouble,
                               resultType, mlir::ValueRange{input});
      rewriter.replaceOp(op, call.getResults());
      return mlir::success();
    }

    if (auto tensorType = llvm::dyn_cast<mlir::RankedTensorType>(inputType)) {
      if (!tensorType.hasStaticShape())
        return rewriter.notifyMatchFailure(
            op, "from_prim for tensor requires static shape");

      llvm::SmallVector<std::string> flat;
      if (!collectTensorLiteralElements(input, flat)) {
        llvm::SmallVector<double> numericFlat;
        if (collectTensorLiteralNumbers(input, numericFlat)) {
          flat = formatTensorLiteralNumbers(numericFlat);
        }
      }

      if (flat.empty()) {
        auto elemType = tensorType.getElementType();
        auto floatType = llvm::dyn_cast<mlir::FloatType>(elemType);
        if (!floatType)
          return rewriter.notifyMatchFailure(
              op, "from_prim for tensor requires float element type");

        unsigned width = floatType.getWidth();
        llvm::StringRef funcName;
        if (width == 16) {
          funcName = "LyTensorF16_Repr";
        } else if (width == 32) {
          funcName = "LyTensorF32_Repr";
        } else if (width == 64) {
          funcName = "LyTensorF64_Repr";
        } else if (width == 128) {
          funcName = "LyTensorF128_Repr";
        } else {
          return rewriter.notifyMatchFailure(
              op, "unsupported float width for tensor repr");
        }

        auto memrefType =
            mlir::MemRefType::get(tensorType.getShape(), elemType);
        mlir::Value memref = rewriter.create<mlir::bufferization::ToMemrefOp>(
            op.getLoc(), memrefType, input);
        auto unrankedType = mlir::UnrankedMemRefType::get(elemType, 0);
        mlir::Value unranked = rewriter.create<mlir::memref::CastOp>(
            op.getLoc(), unrankedType, memref);

        auto func = getOrInsertTensorReprFunc(module, funcName, unrankedType,
                                              resultType);
        auto call = rewriter.create<mlir::func::CallOp>(
            op.getLoc(), func, mlir::ValueRange{unranked});
        rewriter.replaceOp(op, call.getResults());
        return mlir::success();
      }

      size_t index = 0;
      std::string text =
          formatTensorLiteral(tensorType.getShape(), flat, index);

      mlir::Value dataPtr = runtime.getStringLiteral(
          op.getLoc(), mlir::StringAttr::get(rewriter.getContext(), text));
      mlir::Value length = runtime.getI64Constant(
          op.getLoc(), static_cast<int64_t>(text.size()));
      auto call = runtime.call(op.getLoc(), RuntimeSymbols::kStrFromUtf8,
                               resultType, mlir::ValueRange{dataPtr, length});
      rewriter.replaceOp(op, call.getResults());
      eraseDeadTensorProducerTree(input, rewriter);
      return mlir::success();
    }

    return rewriter.notifyMatchFailure(op,
                                       "unsupported input type for from_prim");
  }
};

} // namespace

namespace lowering::value::number::Patterns {
void populate(PyLLVMTypeConverter &typeConverter,
              mlir::RewritePatternSet &patterns) {
  auto *ctx = patterns.getContext();
  patterns.add<StrConstantLowering, NoneLowering, IntConstantLowering,
               FloatConstantLowering, AddLowering, SubLowering, LtLowering,
               LeLowering, GtLowering, GeLowering, EqLowering, NeLowering,
               CastToPrimLowering, CastFromPrimLowering>(typeConverter, ctx);
}
} // namespace lowering::value::number::Patterns

} // namespace py
