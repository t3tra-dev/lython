#include "Common/RuntimeSupport.h"

#include "Common/Object.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"

#include <algorithm>
#include <cstdint>
#include <optional>
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

static bool isLongPartsTypes(llvm::ArrayRef<mlir::Type> types) {
  return types.size() == 3 && object_abi::long_abi::Parts::isHeader(types[0]) &&
         object_abi::long_abi::Parts::isMeta(types[1]) &&
         object_abi::long_abi::Parts::isDigits(types[2]);
}

static bool isUnicodePartsTypes(llvm::ArrayRef<mlir::Type> types) {
  return types.size() == 2 && object_abi::str_abi::Parts::isHeader(types[0]) &&
         object_abi::str_abi::Parts::isBytes(types[1]);
}

static llvm::SmallVector<mlir::ValueRange, 1>
asReplacement(mlir::ValueRange results) {
  return llvm::SmallVector<mlir::ValueRange, 1>{mlir::ValueRange(results)};
}

static llvm::SmallVector<mlir::ValueRange, 1>
asReplacement(mlir::Operation::result_range results) {
  return asReplacement(mlir::ValueRange(results));
}

static mlir::Value stripSingleInputCasts(mlir::Value value) {
  while (auto cast = value.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
    if (cast->getNumOperands() != 1)
      break;
    value = cast.getOperand(0);
  }
  return value;
}

static std::optional<int64_t> intLiteralI64(mlir::Value value) {
  auto literal = stripSingleInputCasts(value).getDefiningOp<IntConstantOp>();
  if (!literal)
    return std::nullopt;

  llvm::SmallString<64> cleaned;
  for (char ch : literal.getValue()) {
    if (ch != '_')
      cleaned.push_back(ch);
  }

  int64_t parsed = 0;
  if (llvm::StringRef(cleaned).getAsInteger(10, parsed))
    return std::nullopt;
  return parsed;
}

static mlir::LogicalResult
replaceWithPartsCall(mlir::Operation *op, RuntimeAPI::Call call,
                     mlir::ConversionPatternRewriter &rewriter) {
  if (!call || call.getOperation()->getNumResults() == 0)
    return mlir::failure();
  rewriter.replaceOpWithMultiple(op, asReplacement(call.getResults()));
  return mlir::success();
}

struct LongLiteralDigits {
  int64_t sign = 0;
  llvm::SmallVector<int64_t, 4> digits;
};

static std::optional<LongLiteralDigits>
parseLongLiteralDigits(llvm::StringRef text) {
  LongLiteralDigits result;
  bool negative = false;
  if (text.consume_front("-"))
    negative = true;
  else
    text.consume_front("+");

  llvm::SmallVector<uint8_t, 32> decimal;
  decimal.reserve(text.size());
  for (char ch : text) {
    if (ch == '_')
      continue;
    if (ch < '0' || ch > '9')
      return std::nullopt;
    decimal.push_back(static_cast<uint8_t>(ch - '0'));
  }
  while (!decimal.empty() && decimal.front() == 0)
    decimal.erase(decimal.begin());
  if (decimal.empty())
    return result;

  result.sign = negative ? -1 : 1;
  constexpr uint64_t base = 1ULL << object_abi::long_abi::kDigitBits;
  while (!decimal.empty()) {
    llvm::SmallVector<uint8_t, 32> quotient;
    uint64_t remainder = 0;
    bool seenNonZero = false;
    for (uint8_t digit : decimal) {
      uint64_t accum = remainder * 10 + digit;
      uint8_t q = static_cast<uint8_t>(accum / base);
      remainder = accum % base;
      if (q != 0 || seenNonZero) {
        quotient.push_back(q);
        seenNonZero = true;
      }
    }
    result.digits.push_back(static_cast<int64_t>(remainder));
    decimal = std::move(quotient);
  }
  return result;
}

static std::string longLiteralGlobalName(const LongLiteralDigits &literal) {
  llvm::hash_code hash = llvm::hash_combine(literal.sign);
  hash = llvm::hash_combine(hash, literal.digits.size());
  for (int64_t digit : literal.digits)
    hash = llvm::hash_combine(hash, digit);
  return ("__ly_long_const_" +
          llvm::utohexstr(static_cast<uint64_t>(static_cast<size_t>(hash))));
}

static llvm::SmallVector<mlir::Value, 3>
emitLongPartsLiteral(mlir::Location loc, const LongLiteralDigits &literal,
                     mlir::ModuleOp module, mlir::PatternRewriter &rewriter) {
  std::string base = longLiteralGlobalName(literal);
  auto *ctx = rewriter.getContext();
  auto i64Type = rewriter.getI64Type();
  auto i32Type = rewriter.getI32Type();
  mlir::MemRefType headerType = object_abi::Header::owned(ctx);
  mlir::MemRefType metaType = object_abi::long_abi::Meta::storage(ctx);
  mlir::MemRefType dynamicDigitsType =
      object_abi::long_abi::Digits::storage(ctx);
  int64_t digitSlots =
      std::max<int64_t>(1, static_cast<int64_t>(literal.digits.size()));
  mlir::MemRefType digitsType =
      mlir::MemRefType::get({digitSlots}, rewriter.getI32Type());

  // All literal parts are true constants: the refcount kernel checks the
  // immortal sentinel with an acquire load before its RMW, so immortal
  // headers are never written and may live in read-only sections.
  auto ensureGlobal = [&](llvm::StringRef suffix, mlir::MemRefType type,
                          mlir::DenseElementsAttr initial, bool constant) {
    std::string symbol = base + suffix.str();
    if (!module.lookupSymbol<mlir::memref::GlobalOp>(symbol)) {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      rewriter.create<mlir::memref::GlobalOp>(
          loc, symbol, rewriter.getStringAttr("private"), type, initial,
          constant, mlir::IntegerAttr());
    }
    return symbol;
  };

  llvm::SmallVector<int64_t, 2> headerValues{object_abi::kImmortalRefcount,
                                             object_abi::long_abi::kLayoutId};
  auto headerAttr = mlir::DenseIntElementsAttr::get(
      mlir::RankedTensorType::get({object_abi::kHeaderSlots}, i64Type),
      headerValues);
  std::string headerSymbol =
      ensureGlobal("_header", headerType, headerAttr, /*constant=*/true);

  auto i64Attr = [](int64_t value) {
    return llvm::APInt(64, static_cast<uint64_t>(value), /*isSigned=*/true);
  };
  llvm::SmallVector<llvm::APInt, 2> metaValues(
      static_cast<size_t>(object_abi::long_abi::kMetaSlots), i64Attr(0));
  metaValues[object_abi::long_abi::kSignSlot] = i64Attr(literal.sign);
  metaValues[object_abi::long_abi::kDigitCountSlot] =
      i64Attr(static_cast<int64_t>(literal.digits.size()));
  auto metaAttr = mlir::DenseIntElementsAttr::get(
      mlir::RankedTensorType::get({object_abi::long_abi::kMetaSlots}, i64Type),
      metaValues);
  std::string metaSymbol =
      ensureGlobal("_meta", metaType, metaAttr, /*constant=*/true);

  auto i32Attr = [](int64_t value) {
    return llvm::APInt(32, static_cast<uint64_t>(value), /*isSigned=*/true);
  };
  llvm::SmallVector<llvm::APInt, 3> digitValues(static_cast<size_t>(digitSlots),
                                                i32Attr(0));
  for (auto [index, digit] : llvm::enumerate(literal.digits)) {
    digitValues[object_abi::long_abi::kDigitsBaseSlot + index] =
        i32Attr(digit & object_abi::long_abi::kDigitMask);
  }
  auto digitsAttr = mlir::DenseIntElementsAttr::get(
      mlir::RankedTensorType::get({digitSlots}, i32Type), digitValues);
  std::string digitsSymbol =
      ensureGlobal("_digits", digitsType, digitsAttr, /*constant=*/true);

  mlir::Value header =
      rewriter.create<mlir::memref::GetGlobalOp>(loc, headerType, headerSymbol);
  mlir::Value meta =
      rewriter.create<mlir::memref::GetGlobalOp>(loc, metaType, metaSymbol);
  mlir::Value digits =
      rewriter.create<mlir::memref::GetGlobalOp>(loc, digitsType, digitsSymbol);
  if (digits.getType() != dynamicDigitsType) {
    digits =
        rewriter.create<mlir::memref::CastOp>(loc, dynamicDigitsType, digits);
  }
  for (mlir::Value value : {header, meta, digits})
    if (mlir::Operation *def = value.getDefiningOp())
      def->setAttr(OwnershipContractAttrs::kImmortalObject,
                   rewriter.getUnitAttr());
  return {header, meta, digits};
}

static mlir::Value boxBool(mlir::Location loc, mlir::Value value,
                           mlir::Type resultType, RuntimeAPI &runtime) {
  auto *ctx = resultType.getContext();
  if (resultType == mlir::IntegerType::get(ctx, 1)) {
    if (value.getType() == resultType)
      return value;
    if (auto intType = mlir::dyn_cast<mlir::IntegerType>(value.getType())) {
      mlir::OpBuilder &builder = runtime.getBuilder();
      mlir::Value zero =
          builder.create<mlir::arith::ConstantIntOp>(loc, 0, intType);
      return builder.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::ne, value, zero);
    }
    return {};
  }
  if (!object_abi::Type::isStorageLike(resultType))
    return {};
  return {};
}

static mlir::Value boxFloat(mlir::Location loc, mlir::Value value,
                            mlir::Type resultType, RuntimeAPI &runtime) {
  auto *ctx = resultType.getContext();
  if (resultType == mlir::Float64Type::get(ctx)) {
    mlir::OpBuilder &builder = runtime.getBuilder();
    if (value.getType() == resultType)
      return value;
    if (value.getType().isInteger(64))
      return builder.create<mlir::arith::BitcastOp>(loc, resultType, value);
    if (mlir::isa<mlir::Float32Type>(value.getType()))
      return builder.create<mlir::arith::ExtFOp>(loc, resultType, value);
    return {};
  }
  if (!object_abi::Type::isStorageLike(resultType))
    return {};
  return {};
}

static mlir::Value unboxBool(mlir::Location loc, mlir::Value value,
                             mlir::PatternRewriter &rewriter,
                             RuntimeAPI &runtime) {
  (void)runtime;
  if (value.getType() == rewriter.getI1Type())
    return value;
  if (auto intType = mlir::dyn_cast<mlir::IntegerType>(value.getType())) {
    mlir::Value zero =
        rewriter.create<mlir::arith::ConstantIntOp>(loc, 0, intType);
    return rewriter.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::ne, value, zero);
  }
  return {};
}

static mlir::Value unboxFloat(mlir::Location loc, mlir::Value value,
                              mlir::PatternRewriter &rewriter,
                              RuntimeAPI &runtime) {
  (void)runtime;
  if (value.getType() == rewriter.getF64Type())
    return value;
  if (value.getType().isInteger(64))
    return rewriter.create<mlir::arith::BitcastOp>(loc, rewriter.getF64Type(),
                                                   value);
  return {};
}

static mlir::Value lowerFloatBinary(mlir::Location loc, mlir::Value lhs,
                                    mlir::Value rhs, mlir::Type resultType,
                                    bool subtract,
                                    mlir::PatternRewriter &rewriter,
                                    RuntimeAPI &runtime) {
  mlir::Value lhsValue = unboxFloat(loc, lhs, rewriter, runtime);
  mlir::Value rhsValue = unboxFloat(loc, rhs, rewriter, runtime);
  if (!lhsValue || !rhsValue)
    return {};
  mlir::Value result =
      subtract ? rewriter.create<mlir::arith::SubFOp>(loc, lhsValue, rhsValue)
                     .getResult()
               : rewriter.create<mlir::arith::AddFOp>(loc, lhsValue, rhsValue)
                     .getResult();
  return boxFloat(loc, result, resultType, runtime);
}

static mlir::Value
lowerFloatCompare(mlir::Location loc, mlir::Value lhs, mlir::Value rhs,
                  mlir::arith::CmpFPredicate predicate, mlir::Type resultType,
                  mlir::PatternRewriter &rewriter, RuntimeAPI &runtime) {
  mlir::Value lhsValue = unboxFloat(loc, lhs, rewriter, runtime);
  mlir::Value rhsValue = unboxFloat(loc, rhs, rewriter, runtime);
  if (!lhsValue || !rhsValue)
    return {};
  mlir::Value bit =
      rewriter.create<mlir::arith::CmpFOp>(loc, predicate, lhsValue, rhsValue);
  return boxBool(loc, bit, resultType, runtime);
}

static mlir::Value
lowerBoolCompare(mlir::Location loc, mlir::Value lhs, mlir::Value rhs,
                 mlir::LLVM::ICmpPredicate predicate, mlir::Type resultType,
                 mlir::PatternRewriter &rewriter, RuntimeAPI &runtime) {
  mlir::Value lhsValue = unboxBool(loc, lhs, rewriter, runtime);
  mlir::Value rhsValue = unboxBool(loc, rhs, rewriter, runtime);
  if (!lhsValue || !rhsValue)
    return {};
  auto arithPredicate = predicate == mlir::LLVM::ICmpPredicate::eq
                            ? mlir::arith::CmpIPredicate::eq
                            : mlir::arith::CmpIPredicate::ne;
  mlir::Value bit = rewriter.create<mlir::arith::CmpIOp>(loc, arithPredicate,
                                                         lhsValue, rhsValue);
  return boxBool(loc, bit, resultType, runtime);
}

static mlir::Value lowerUnicodeCompare(mlir::Location loc, mlir::ValueRange lhs,
                                       mlir::ValueRange rhs,
                                       mlir::LLVM::ICmpPredicate predicate,
                                       mlir::Type resultType,
                                       mlir::PatternRewriter &rewriter,
                                       RuntimeAPI &runtime) {
  if (lhs.size() != 2 || rhs.size() != 2)
    return {};
  llvm::SmallVector<mlir::Value, 4> operands;
  operands.append(lhs.begin(), lhs.end());
  operands.append(rhs.begin(), rhs.end());
  mlir::Value bit = runtime
                        .call(loc, RuntimeSymbols::kUnicodeEqBool,
                              rewriter.getI1Type(), operands)
                        .getResult();
  if (predicate == mlir::LLVM::ICmpPredicate::ne)
    bit = rewriter.create<mlir::arith::XOrIOp>(
        loc, bit,
        rewriter.create<mlir::arith::ConstantIntOp>(loc, true,
                                                    rewriter.getI1Type()));
  return boxBool(loc, bit, resultType, runtime);
}

static mlir::Value longAsI64(mlir::Location loc, mlir::ValueRange value,
                             mlir::PatternRewriter &rewriter,
                             RuntimeAPI &runtime) {
  if (value.size() == 3) {
    return runtime
        .call(loc, RuntimeSymbols::kLongAsI64, rewriter.getI64Type(), value)
        .getResult();
  }
  if (value.size() == 1 && value.front().getType() == rewriter.getI64Type())
    return value.front();
  return {};
}

static mlir::LogicalResult
replaceWithLongFromI64(mlir::Operation *op, mlir::Value value,
                       llvm::ArrayRef<mlir::Type> results, RuntimeAPI &runtime,
                       mlir::ConversionPatternRewriter &rewriter) {
  if (isLongPartsTypes(results)) {
    auto call = runtime.call(op->getLoc(), RuntimeSymbols::kLongFromI64,
                             mlir::TypeRange(results), mlir::ValueRange{value});
    return replaceWithPartsCall(op, call, rewriter);
  }
  return mlir::failure();
}

static mlir::LogicalResult replaceWithLongBinaryCall(
    mlir::Operation *op, mlir::ValueRange lhsRange, mlir::ValueRange rhsRange,
    llvm::ArrayRef<mlir::Type> resultTypes, llvm::StringRef callee,
    RuntimeAPI &runtime, mlir::ConversionPatternRewriter &rewriter) {
  if (!isLongPartsTypes(resultTypes) || lhsRange.size() != 3 ||
      rhsRange.size() != 3)
    return mlir::failure();
  llvm::SmallVector<mlir::Value, 6> operands;
  operands.append(lhsRange.begin(), lhsRange.end());
  operands.append(rhsRange.begin(), rhsRange.end());
  auto call = runtime.call(op->getLoc(), callee, mlir::TypeRange(resultTypes),
                           operands);
  return replaceWithPartsCall(op, call, rewriter);
}

enum class IntBinaryKind {
  FloorDiv,
  Mod,
  LShift,
  RShift,
  BitAnd,
  BitOr,
  BitXor,
};

static mlir::Value emitSignedFloorDiv(mlir::Location loc, mlir::Value lhs,
                                      mlir::Value rhs,
                                      mlir::PatternRewriter &rewriter) {
  mlir::Value truncQuotient =
      rewriter.create<mlir::arith::DivSIOp>(loc, lhs, rhs);
  mlir::Value truncRemainder =
      rewriter.create<mlir::arith::RemSIOp>(loc, lhs, rhs);
  mlir::Value zero =
      rewriter.create<mlir::arith::ConstantIntOp>(loc, 0, lhs.getType());
  mlir::Value hasRemainder = rewriter.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::ne, truncRemainder, zero);
  mlir::Value lhsNegative = rewriter.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::slt, lhs, zero);
  mlir::Value rhsNegative = rewriter.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::slt, rhs, zero);
  mlir::Value signsDiffer =
      rewriter.create<mlir::arith::XOrIOp>(loc, lhsNegative, rhsNegative);
  mlir::Value adjust =
      rewriter.create<mlir::arith::AndIOp>(loc, hasRemainder, signsDiffer);
  mlir::Value one =
      rewriter.create<mlir::arith::ConstantIntOp>(loc, 1, lhs.getType());
  mlir::Value adjusted =
      rewriter.create<mlir::arith::SubIOp>(loc, truncQuotient, one);
  return rewriter.create<mlir::arith::SelectOp>(loc, adjust, adjusted,
                                                truncQuotient);
}

static mlir::LogicalResult replaceWithIntBinary(
    mlir::Operation *op, mlir::ValueRange lhsRange, mlir::ValueRange rhsRange,
    llvm::ArrayRef<mlir::Type> resultTypes, IntBinaryKind kind,
    RuntimeAPI &runtime, mlir::ConversionPatternRewriter &rewriter) {
  mlir::Location loc = op->getLoc();
  mlir::Value lhs = longAsI64(loc, lhsRange, rewriter, runtime);
  mlir::Value rhs = longAsI64(loc, rhsRange, rewriter, runtime);
  if (!lhs || !rhs)
    return mlir::failure();

  mlir::Value result;
  switch (kind) {
  case IntBinaryKind::FloorDiv:
    result = emitSignedFloorDiv(loc, lhs, rhs, rewriter);
    break;
  case IntBinaryKind::Mod: {
    mlir::Value quotient = emitSignedFloorDiv(loc, lhs, rhs, rewriter);
    mlir::Value product =
        rewriter.create<mlir::arith::MulIOp>(loc, quotient, rhs);
    result = rewriter.create<mlir::arith::SubIOp>(loc, lhs, product);
    break;
  }
  case IntBinaryKind::LShift:
    result = rewriter.create<mlir::arith::ShLIOp>(loc, lhs, rhs);
    break;
  case IntBinaryKind::RShift:
    result = rewriter.create<mlir::arith::ShRSIOp>(loc, lhs, rhs);
    break;
  case IntBinaryKind::BitAnd:
    result = rewriter.create<mlir::arith::AndIOp>(loc, lhs, rhs);
    break;
  case IntBinaryKind::BitOr:
    result = rewriter.create<mlir::arith::OrIOp>(loc, lhs, rhs);
    break;
  case IntBinaryKind::BitXor:
    result = rewriter.create<mlir::arith::XOrIOp>(loc, lhs, rhs);
    break;
  }
  return replaceWithLongFromI64(op, result, resultTypes, runtime, rewriter);
}

static mlir::Value lowerIntCompare(mlir::Location loc, mlir::ValueRange lhs,
                                   mlir::ValueRange rhs,
                                   mlir::arith::CmpIPredicate arithPredicate,
                                   mlir::Type resultType,
                                   mlir::PatternRewriter &rewriter,
                                   RuntimeAPI &runtime) {
  if (lhs.size() != 3 || rhs.size() != 3)
    return {};
  llvm::SmallVector<mlir::Value, 6> operands;
  operands.append(lhs.begin(), lhs.end());
  operands.append(rhs.begin(), rhs.end());
  mlir::Value cmp = runtime
                        .call(loc, RuntimeSymbols::kLongCompare,
                              rewriter.getI64Type(), operands)
                        .getResult();
  mlir::Value zero = rewriter.create<mlir::arith::ConstantIntOp>(loc, 0, 64);
  mlir::Value bit =
      rewriter.create<mlir::arith::CmpIOp>(loc, arithPredicate, cmp, zero);
  return boxBool(loc, bit, resultType, runtime);
}

static mlir::Value
lowerIntLiteralCompare(mlir::Location loc, mlir::Value lhs, mlir::Value rhs,
                       mlir::arith::CmpIPredicate arithPredicate,
                       mlir::Type resultType, mlir::PatternRewriter &rewriter,
                       RuntimeAPI &runtime) {
  std::optional<int64_t> lhsValue = intLiteralI64(lhs);
  std::optional<int64_t> rhsValue = intLiteralI64(rhs);
  if (!lhsValue || !rhsValue)
    return {};

  mlir::Value lhsConst =
      rewriter.create<mlir::arith::ConstantIntOp>(loc, *lhsValue, 64);
  mlir::Value rhsConst =
      rewriter.create<mlir::arith::ConstantIntOp>(loc, *rhsValue, 64);
  mlir::Value bit = rewriter.create<mlir::arith::CmpIOp>(loc, arithPredicate,
                                                         lhsConst, rhsConst);
  return boxBool(loc, bit, resultType, runtime);
}

static llvm::SmallVector<mlir::Value, 3>
unicodeLiteralOwnedParts(mlir::Location loc, llvm::StringRef text,
                         mlir::OpBuilder &builder, RuntimeAPI &runtime) {
  auto attr = builder.getStringAttr(text);
  mlir::Value bytes = runtime.getByteLiteral(loc, attr);
  mlir::Value start = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
  mlir::Value length =
      runtime.getI64Constant(loc, static_cast<int64_t>(text.size()));
  llvm::SmallVector<mlir::Type, 3> resultTypes;
  object_abi::str_abi::Parts::storageTypes(builder.getContext(), resultTypes);
  auto call = runtime.call(loc, RuntimeSymbols::kUnicodeFromBytes,
                           mlir::TypeRange(resultTypes),
                           mlir::ValueRange{bytes, start, length});
  return llvm::SmallVector<mlir::Value, 3>(call.getResults());
}

static void yieldUnicodeLiteral(mlir::Location loc, llvm::StringRef text,
                                mlir::OpBuilder &builder, RuntimeAPI &runtime) {
  llvm::SmallVector<mlir::Value, 3> values =
      unicodeLiteralOwnedParts(loc, text, builder, runtime);
  builder.create<mlir::scf::YieldOp>(loc, values);
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

    llvm::SmallVector<mlir::Type, 3> resultTypes;
    if (mlir::failed(typeConverter->convertType(op.getResult().getType(),
                                                resultTypes)) ||
        resultTypes.empty())
      return mlir::failure();

    if (isUnicodePartsTypes(resultTypes)) {
      mlir::Value bytes =
          runtime.getByteLiteral(op.getLoc(), op.getValueAttr());
      mlir::Value start =
          rewriter.create<mlir::arith::ConstantIndexOp>(op.getLoc(), 0);
      mlir::Value length =
          runtime.getI64Constant(op.getLoc(), op.getValue().size());
      auto call = runtime.call(op.getLoc(), RuntimeSymbols::kUnicodeFromBytes,
                               mlir::TypeRange(resultTypes),
                               mlir::ValueRange{bytes, start, length});
      return replaceWithPartsCall(op, call, rewriter);
    }

    return rewriter.notifyMatchFailure(
        op, "str constants require unicode header/payload parts");
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
    mlir::Value none = runtime.getNoneValue(op.getLoc());
    if (none.getType() != resultType)
      none = rewriter
                 .create<mlir::UnrealizedConversionCastOp>(
                     op.getLoc(), mlir::TypeRange{resultType},
                     mlir::ValueRange{none})
                 .getResult(0);
    rewriter.replaceOp(op, none);
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
    llvm::SmallVector<mlir::Type, 3> resultTypes;
    if (mlir::failed(typeConverter->convertType(op.getResult().getType(),
                                                resultTypes)) ||
        resultTypes.empty())
      return mlir::failure();

    // Integer value is stored as decimal string to support arbitrary precision
    llvm::StringRef valueStr = op.getValue();

    if (isLongPartsTypes(resultTypes)) {
      std::optional<LongLiteralDigits> literal =
          parseLongLiteralDigits(valueStr);
      if (!literal)
        return rewriter.notifyMatchFailure(op,
                                           "unsupported integer literal text");
      llvm::SmallVector<mlir::Value, 3> values =
          emitLongPartsLiteral(op.getLoc(), *literal, module, rewriter);
      rewriter.replaceOpWithMultiple(
          op, llvm::ArrayRef<mlir::ValueRange>{mlir::ValueRange(values)});
      return mlir::success();
    }

    return rewriter.notifyMatchFailure(
        op, "py.int lowering requires long header/payload parts");
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
    if (resultType == rewriter.getF64Type()) {
      rewriter.replaceOp(op, value);
      return mlir::success();
    }
    if (!object_abi::Type::isStorageLike(resultType))
      return rewriter.notifyMatchFailure(
          op, "py.float lowering requires memref object storage");
    mlir::Value boxed = boxFloat(op.getLoc(), value, resultType, runtime);
    if (!boxed)
      return rewriter.notifyMatchFailure(
          op, "py.float lowering requires native float storage");
    rewriter.replaceOp(op, boxed);
    return mlir::success();
  }
};

// mlir::Type-specialized lowering for AddOp
struct AddLowering : public mlir::OpConversionPattern<AddOp> {
  AddLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<AddOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(AddOp op, OneToNOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return mlir::failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *typeConverter);
    llvm::SmallVector<mlir::Type, 3> resultTypes;
    if (mlir::failed(typeConverter->convertType(op.getResult().getType(),
                                                resultTypes)) ||
        resultTypes.empty())
      return mlir::failure();

    if (isPyIntType(op.getLhs().getType()) &&
        isPyIntType(op.getRhs().getType())) {
      if (mlir::succeeded(replaceWithLongBinaryCall(
              op, adaptor.getLhs(), adaptor.getRhs(), resultTypes,
              RuntimeSymbols::kLongAdd, runtime, rewriter)))
        return mlir::success();
      return rewriter.notifyMatchFailure(op, "int add requires LyLong parts");
    }

    if (isPyFloatType(op.getLhs().getType()) &&
        isPyFloatType(op.getRhs().getType())) {
      mlir::Value result =
          lowerFloatBinary(op.getLoc(), adaptor.getLhs().front(),
                           adaptor.getRhs().front(), resultTypes.front(),
                           /*subtract=*/false, rewriter, runtime);
      if (!result)
        return rewriter.notifyMatchFailure(
            op, "float add requires memref object storage");
      rewriter.replaceOp(op, result);
      return mlir::success();
    }

    if (isPyStrType(op.getLhs().getType()) &&
        isPyStrType(op.getRhs().getType())) {
      if (isUnicodePartsTypes(resultTypes)) {
        if (adaptor.getLhs().size() != 2 || adaptor.getRhs().size() != 2)
          return mlir::failure();
        llvm::SmallVector<mlir::Value, 4> operands;
        operands.append(adaptor.getLhs().begin(), adaptor.getLhs().end());
        operands.append(adaptor.getRhs().begin(), adaptor.getRhs().end());
        auto call = runtime.call(op.getLoc(), RuntimeSymbols::kUnicodeConcat,
                                 mlir::TypeRange(resultTypes), operands);
        return replaceWithPartsCall(op, call, rewriter);
      }
      return rewriter.notifyMatchFailure(
          op, "str add requires unicode header/payload parts");
    }

    return rewriter.notifyMatchFailure(op, "unsupported statically typed add");
  }
};

struct StrConcat3Lowering : public mlir::OpConversionPattern<StrConcat3Op> {
  StrConcat3Lowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<StrConcat3Op>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(StrConcat3Op op, OneToNOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return mlir::failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *typeConverter);
    llvm::SmallVector<mlir::Type, 3> resultTypes;
    if (mlir::failed(typeConverter->convertType(op.getResult().getType(),
                                                resultTypes)) ||
        resultTypes.empty())
      return mlir::failure();
    if (isUnicodePartsTypes(resultTypes)) {
      if (adaptor.getLhs().size() != 2 || adaptor.getMiddle().size() != 2 ||
          adaptor.getRhs().size() != 2)
        return mlir::failure();
      llvm::SmallVector<mlir::Value, 6> operands;
      operands.append(adaptor.getLhs().begin(), adaptor.getLhs().end());
      operands.append(adaptor.getMiddle().begin(), adaptor.getMiddle().end());
      operands.append(adaptor.getRhs().begin(), adaptor.getRhs().end());
      auto call = runtime.call(op.getLoc(), RuntimeSymbols::kUnicodeConcat3,
                               mlir::TypeRange(resultTypes), operands);
      return replaceWithPartsCall(op, call, rewriter);
    }
    return rewriter.notifyMatchFailure(
        op, "str.concat3 requires unicode header/payload parts");
  }
};

// mlir::Type-specialized lowering for SubOp
struct SubLowering : public mlir::OpConversionPattern<SubOp> {
  SubLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<SubOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(SubOp op, OneToNOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return mlir::failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *typeConverter);
    llvm::SmallVector<mlir::Type, 3> resultTypes;
    if (mlir::failed(typeConverter->convertType(op.getResult().getType(),
                                                resultTypes)) ||
        resultTypes.empty())
      return mlir::failure();

    if (isPyIntType(op.getLhs().getType()) &&
        isPyIntType(op.getRhs().getType())) {
      if (mlir::succeeded(replaceWithLongBinaryCall(
              op, adaptor.getLhs(), adaptor.getRhs(), resultTypes,
              RuntimeSymbols::kLongSub, runtime, rewriter)))
        return mlir::success();
      return rewriter.notifyMatchFailure(op, "int sub requires LyLong parts");
    }

    if (isPyFloatType(op.getLhs().getType()) &&
        isPyFloatType(op.getRhs().getType())) {
      mlir::Value result =
          lowerFloatBinary(op.getLoc(), adaptor.getLhs().front(),
                           adaptor.getRhs().front(), resultTypes.front(),
                           /*subtract=*/true, rewriter, runtime);
      if (!result)
        return rewriter.notifyMatchFailure(
            op, "float sub requires memref object storage");
      rewriter.replaceOp(op, result);
      return mlir::success();
    }

    return rewriter.notifyMatchFailure(op, "unsupported statically typed sub");
  }
};

// mlir::Type-specialized lowering for MulOp
struct MulLowering : public mlir::OpConversionPattern<MulOp> {
  MulLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<MulOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(MulOp op, OneToNOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return mlir::failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *typeConverter);
    llvm::SmallVector<mlir::Type, 3> resultTypes;
    if (mlir::failed(typeConverter->convertType(op.getResult().getType(),
                                                resultTypes)) ||
        resultTypes.empty())
      return mlir::failure();

    if (isPyIntType(op.getLhs().getType()) &&
        isPyIntType(op.getRhs().getType())) {
      if (mlir::succeeded(replaceWithLongBinaryCall(
              op, adaptor.getLhs(), adaptor.getRhs(), resultTypes,
              RuntimeSymbols::kLongMul, runtime, rewriter)))
        return mlir::success();
      return rewriter.notifyMatchFailure(op, "int mul requires LyLong parts");
    }

    if (isPyFloatType(op.getLhs().getType()) &&
        isPyFloatType(op.getRhs().getType())) {
      mlir::Value lhs =
          unboxFloat(op.getLoc(), adaptor.getLhs().front(), rewriter, runtime);
      mlir::Value rhs =
          unboxFloat(op.getLoc(), adaptor.getRhs().front(), rewriter, runtime);
      if (!lhs || !rhs)
        return rewriter.notifyMatchFailure(
            op, "float mul requires memref object storage");
      mlir::Value product =
          rewriter.create<mlir::arith::MulFOp>(op.getLoc(), lhs, rhs);
      mlir::Value result =
          boxFloat(op.getLoc(), product, resultTypes.front(), runtime);
      if (!result)
        return rewriter.notifyMatchFailure(
            op, "float mul requires memref object storage");
      rewriter.replaceOp(op, result);
      return mlir::success();
    }

    return rewriter.notifyMatchFailure(op, "unsupported statically typed mul");
  }
};

struct DivLowering : public mlir::OpConversionPattern<DivOp> {
  DivLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<DivOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(DivOp op, OneToNOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return mlir::failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *typeConverter);
    llvm::SmallVector<mlir::Type, 3> resultTypes;
    if (mlir::failed(typeConverter->convertType(op.getResult().getType(),
                                                resultTypes)) ||
        resultTypes.empty())
      return mlir::failure();

    if (isPyFloatType(op.getLhs().getType()) &&
        isPyFloatType(op.getRhs().getType())) {
      mlir::Value lhs =
          unboxFloat(op.getLoc(), adaptor.getLhs().front(), rewriter, runtime);
      mlir::Value rhs =
          unboxFloat(op.getLoc(), adaptor.getRhs().front(), rewriter, runtime);
      if (!lhs || !rhs)
        return rewriter.notifyMatchFailure(
            op, "float div requires memref object storage");
      mlir::Value quotient =
          rewriter.create<mlir::arith::DivFOp>(op.getLoc(), lhs, rhs);
      mlir::Value result =
          boxFloat(op.getLoc(), quotient, resultTypes.front(), runtime);
      if (!result)
        return rewriter.notifyMatchFailure(
            op, "float div requires memref object storage");
      rewriter.replaceOp(op, result);
      return mlir::success();
    }

    return rewriter.notifyMatchFailure(op, "unsupported statically typed div");
  }
};

template <typename OpTy, IntBinaryKind Kind>
struct IntBinaryLowering : public mlir::OpConversionPattern<OpTy> {
  using Base = mlir::OpConversionPattern<OpTy>;
  using OneToNOpAdaptor = typename Base::OneToNOpAdaptor;

  IntBinaryLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : Base(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(OpTy op, OneToNOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::ModuleOp module = op->template getParentOfType<mlir::ModuleOp>();
    if (!module)
      return mlir::failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(this->getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *typeConverter);
    llvm::SmallVector<mlir::Type, 3> resultTypes;
    if (mlir::failed(typeConverter->convertType(op.getResult().getType(),
                                                resultTypes)) ||
        resultTypes.empty())
      return mlir::failure();
    if (!isPyIntType(op.getLhs().getType()) ||
        !isPyIntType(op.getRhs().getType()))
      return rewriter.notifyMatchFailure(
          op, "unsupported statically typed integer binary op");
    if (mlir::succeeded(replaceWithIntBinary(op, adaptor.getLhs(),
                                             adaptor.getRhs(), resultTypes,
                                             Kind, runtime, rewriter)))
      return mlir::success();
    return rewriter.notifyMatchFailure(op,
                                       "int op requires memref object storage");
  }
};

// mlir::Type-specialized lowering for LeOp
struct LeLowering : public mlir::OpConversionPattern<LeOp> {
  LeLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx,
             mlir::PatternBenefit benefit = 1)
      : mlir::OpConversionPattern<LeOp>(converter, ctx, benefit) {}

  mlir::LogicalResult
  matchAndRewrite(LeOp op, OneToNOpAdaptor adaptor,
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

    if (isPyIntType(op.getLhs().getType()) &&
        isPyIntType(op.getRhs().getType())) {
      if (mlir::Value result = lowerIntLiteralCompare(
              op.getLoc(), op.getLhs(), op.getRhs(),
              mlir::arith::CmpIPredicate::sle, resultType, rewriter, runtime)) {
        rewriter.replaceOp(op, result);
        return mlir::success();
      }
      mlir::Value result = lowerIntCompare(
          op.getLoc(), adaptor.getLhs(), adaptor.getRhs(),
          mlir::arith::CmpIPredicate::sle, resultType, rewriter, runtime);
      if (!result)
        return rewriter.notifyMatchFailure(
            op, "int compare requires memref object storage");
      rewriter.replaceOp(op, result);
      return mlir::success();
    }

    if (isPyFloatType(op.getLhs().getType()) &&
        isPyFloatType(op.getRhs().getType())) {
      mlir::Value result = lowerFloatCompare(
          op.getLoc(), adaptor.getLhs().front(), adaptor.getRhs().front(),
          mlir::arith::CmpFPredicate::OLE, resultType, rewriter, runtime);
      if (!result)
        return rewriter.notifyMatchFailure(
            op, "float compare requires memref object storage");
      rewriter.replaceOp(op, result);
      return mlir::success();
    }

    return rewriter.notifyMatchFailure(op, "unsupported statically typed le");
  }
};

// mlir::Type-specialized lowering for LtOp
struct LtLowering : public mlir::OpConversionPattern<LtOp> {
  LtLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx,
             mlir::PatternBenefit benefit = 1)
      : mlir::OpConversionPattern<LtOp>(converter, ctx, benefit) {}

  mlir::LogicalResult
  matchAndRewrite(LtOp op, OneToNOpAdaptor adaptor,
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

    if (isPyIntType(op.getLhs().getType()) &&
        isPyIntType(op.getRhs().getType())) {
      if (mlir::Value result = lowerIntLiteralCompare(
              op.getLoc(), op.getLhs(), op.getRhs(),
              mlir::arith::CmpIPredicate::slt, resultType, rewriter, runtime)) {
        rewriter.replaceOp(op, result);
        return mlir::success();
      }
      mlir::Value result = lowerIntCompare(
          op.getLoc(), adaptor.getLhs(), adaptor.getRhs(),
          mlir::arith::CmpIPredicate::slt, resultType, rewriter, runtime);
      if (!result)
        return rewriter.notifyMatchFailure(
            op, "int compare requires memref object storage");
      rewriter.replaceOp(op, result);
      return mlir::success();
    }

    if (isPyFloatType(op.getLhs().getType()) &&
        isPyFloatType(op.getRhs().getType())) {
      mlir::Value result = lowerFloatCompare(
          op.getLoc(), adaptor.getLhs().front(), adaptor.getRhs().front(),
          mlir::arith::CmpFPredicate::OLT, resultType, rewriter, runtime);
      if (!result)
        return rewriter.notifyMatchFailure(
            op, "float compare requires memref object storage");
      rewriter.replaceOp(op, result);
      return mlir::success();
    }

    return rewriter.notifyMatchFailure(op, "unsupported statically typed lt");
  }
};

// mlir::Type-specialized lowering for GtOp
struct GtLowering : public mlir::OpConversionPattern<GtOp> {
  GtLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx,
             mlir::PatternBenefit benefit = 1)
      : mlir::OpConversionPattern<GtOp>(converter, ctx, benefit) {}

  mlir::LogicalResult
  matchAndRewrite(GtOp op, OneToNOpAdaptor adaptor,
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

    if (isPyIntType(op.getLhs().getType()) &&
        isPyIntType(op.getRhs().getType())) {
      if (mlir::Value result = lowerIntLiteralCompare(
              op.getLoc(), op.getLhs(), op.getRhs(),
              mlir::arith::CmpIPredicate::sgt, resultType, rewriter, runtime)) {
        rewriter.replaceOp(op, result);
        return mlir::success();
      }
      mlir::Value result = lowerIntCompare(
          op.getLoc(), adaptor.getLhs(), adaptor.getRhs(),
          mlir::arith::CmpIPredicate::sgt, resultType, rewriter, runtime);
      if (!result)
        return rewriter.notifyMatchFailure(
            op, "int compare requires memref object storage");
      rewriter.replaceOp(op, result);
      return mlir::success();
    }

    if (isPyFloatType(op.getLhs().getType()) &&
        isPyFloatType(op.getRhs().getType())) {
      mlir::Value result = lowerFloatCompare(
          op.getLoc(), adaptor.getLhs().front(), adaptor.getRhs().front(),
          mlir::arith::CmpFPredicate::OGT, resultType, rewriter, runtime);
      if (!result)
        return rewriter.notifyMatchFailure(
            op, "float compare requires memref object storage");
      rewriter.replaceOp(op, result);
      return mlir::success();
    }

    return rewriter.notifyMatchFailure(op, "unsupported statically typed gt");
  }
};

// mlir::Type-specialized lowering for GeOp
struct GeLowering : public mlir::OpConversionPattern<GeOp> {
  GeLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx,
             mlir::PatternBenefit benefit = 1)
      : mlir::OpConversionPattern<GeOp>(converter, ctx, benefit) {}

  mlir::LogicalResult
  matchAndRewrite(GeOp op, OneToNOpAdaptor adaptor,
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

    if (isPyIntType(op.getLhs().getType()) &&
        isPyIntType(op.getRhs().getType())) {
      if (mlir::Value result = lowerIntLiteralCompare(
              op.getLoc(), op.getLhs(), op.getRhs(),
              mlir::arith::CmpIPredicate::sge, resultType, rewriter, runtime)) {
        rewriter.replaceOp(op, result);
        return mlir::success();
      }
      mlir::Value result = lowerIntCompare(
          op.getLoc(), adaptor.getLhs(), adaptor.getRhs(),
          mlir::arith::CmpIPredicate::sge, resultType, rewriter, runtime);
      if (!result)
        return rewriter.notifyMatchFailure(
            op, "int compare requires memref object storage");
      rewriter.replaceOp(op, result);
      return mlir::success();
    }

    if (isPyFloatType(op.getLhs().getType()) &&
        isPyFloatType(op.getRhs().getType())) {
      mlir::Value result = lowerFloatCompare(
          op.getLoc(), adaptor.getLhs().front(), adaptor.getRhs().front(),
          mlir::arith::CmpFPredicate::OGE, resultType, rewriter, runtime);
      if (!result)
        return rewriter.notifyMatchFailure(
            op, "float compare requires memref object storage");
      rewriter.replaceOp(op, result);
      return mlir::success();
    }

    return rewriter.notifyMatchFailure(op, "unsupported statically typed ge");
  }
};

// mlir::Type-specialized lowering for EqOp
struct EqLowering : public mlir::OpConversionPattern<EqOp> {
  EqLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx,
             mlir::PatternBenefit benefit = 1)
      : mlir::OpConversionPattern<EqOp>(converter, ctx, benefit) {}

  mlir::LogicalResult
  matchAndRewrite(EqOp op, OneToNOpAdaptor adaptor,
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

    if (isPyIntType(op.getLhs().getType()) &&
        isPyIntType(op.getRhs().getType())) {
      if (mlir::Value result = lowerIntLiteralCompare(
              op.getLoc(), op.getLhs(), op.getRhs(),
              mlir::arith::CmpIPredicate::eq, resultType, rewriter, runtime)) {
        rewriter.replaceOp(op, result);
        return mlir::success();
      }
      mlir::Value result = lowerIntCompare(
          op.getLoc(), adaptor.getLhs(), adaptor.getRhs(),
          mlir::arith::CmpIPredicate::eq, resultType, rewriter, runtime);
      if (!result)
        return rewriter.notifyMatchFailure(
            op, "int compare requires memref object storage");
      rewriter.replaceOp(op, result);
      return mlir::success();
    }

    if (isPyFloatType(op.getLhs().getType()) &&
        isPyFloatType(op.getRhs().getType())) {
      mlir::Value result = lowerFloatCompare(
          op.getLoc(), adaptor.getLhs().front(), adaptor.getRhs().front(),
          mlir::arith::CmpFPredicate::OEQ, resultType, rewriter, runtime);
      if (!result)
        return rewriter.notifyMatchFailure(
            op, "float compare requires memref object storage");
      rewriter.replaceOp(op, result);
      return mlir::success();
    }

    if (isPyBoolType(op.getLhs().getType()) &&
        isPyBoolType(op.getRhs().getType())) {
      mlir::Value result = lowerBoolCompare(
          op.getLoc(), adaptor.getLhs().front(), adaptor.getRhs().front(),
          mlir::LLVM::ICmpPredicate::eq, resultType, rewriter, runtime);
      if (!result)
        return rewriter.notifyMatchFailure(
            op, "bool compare requires memref object storage");
      rewriter.replaceOp(op, result);
      return mlir::success();
    }

    if (isPyStrType(op.getLhs().getType()) &&
        isPyStrType(op.getRhs().getType())) {
      mlir::Value result = lowerUnicodeCompare(
          op.getLoc(), adaptor.getLhs(), adaptor.getRhs(),
          mlir::LLVM::ICmpPredicate::eq, resultType, rewriter, runtime);
      if (!result)
        return rewriter.notifyMatchFailure(
            op, "str compare requires unicode header/payload parts");
      rewriter.replaceOp(op, result);
      return mlir::success();
    }

    return rewriter.notifyMatchFailure(op, "unsupported statically typed eq");
  }
};

// mlir::Type-specialized lowering for NeOp
struct NeLowering : public mlir::OpConversionPattern<NeOp> {
  NeLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx,
             mlir::PatternBenefit benefit = 1)
      : mlir::OpConversionPattern<NeOp>(converter, ctx, benefit) {}

  mlir::LogicalResult
  matchAndRewrite(NeOp op, OneToNOpAdaptor adaptor,
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

    if (isPyIntType(op.getLhs().getType()) &&
        isPyIntType(op.getRhs().getType())) {
      if (mlir::Value result = lowerIntLiteralCompare(
              op.getLoc(), op.getLhs(), op.getRhs(),
              mlir::arith::CmpIPredicate::ne, resultType, rewriter, runtime)) {
        rewriter.replaceOp(op, result);
        return mlir::success();
      }
      mlir::Value result = lowerIntCompare(
          op.getLoc(), adaptor.getLhs(), adaptor.getRhs(),
          mlir::arith::CmpIPredicate::ne, resultType, rewriter, runtime);
      if (!result)
        return rewriter.notifyMatchFailure(
            op, "int compare requires memref object storage");
      rewriter.replaceOp(op, result);
      return mlir::success();
    }

    if (isPyFloatType(op.getLhs().getType()) &&
        isPyFloatType(op.getRhs().getType())) {
      mlir::Value result = lowerFloatCompare(
          op.getLoc(), adaptor.getLhs().front(), adaptor.getRhs().front(),
          mlir::arith::CmpFPredicate::UNE, resultType, rewriter, runtime);
      if (!result)
        return rewriter.notifyMatchFailure(
            op, "float compare requires memref object storage");
      rewriter.replaceOp(op, result);
      return mlir::success();
    }

    if (isPyBoolType(op.getLhs().getType()) &&
        isPyBoolType(op.getRhs().getType())) {
      mlir::Value result = lowerBoolCompare(
          op.getLoc(), adaptor.getLhs().front(), adaptor.getRhs().front(),
          mlir::LLVM::ICmpPredicate::ne, resultType, rewriter, runtime);
      if (!result)
        return rewriter.notifyMatchFailure(
            op, "bool compare requires memref object storage");
      rewriter.replaceOp(op, result);
      return mlir::success();
    }

    if (isPyStrType(op.getLhs().getType()) &&
        isPyStrType(op.getRhs().getType())) {
      mlir::Value result = lowerUnicodeCompare(
          op.getLoc(), adaptor.getLhs(), adaptor.getRhs(),
          mlir::LLVM::ICmpPredicate::ne, resultType, rewriter, runtime);
      if (!result)
        return rewriter.notifyMatchFailure(
            op, "str compare requires unicode header/payload parts");
      rewriter.replaceOp(op, result);
      return mlir::success();
    }

    return rewriter.notifyMatchFailure(op, "unsupported statically typed ne");
  }
};

struct ReprLowering : public mlir::OpConversionPattern<ReprOp> {
  ReprLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx,
               mlir::PatternBenefit benefit = 1)
      : mlir::OpConversionPattern<ReprOp>(converter, ctx, benefit) {}

  mlir::LogicalResult
  matchAndRewrite(ReprOp op, OneToNOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return mlir::failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *typeConverter);
    llvm::SmallVector<mlir::Type, 3> resultTypes;
    if (mlir::failed(typeConverter->convertType(op.getResult().getType(),
                                                resultTypes)) ||
        resultTypes.empty())
      return mlir::failure();
    bool partsResult = isUnicodePartsTypes(resultTypes);

    mlir::Type inputType = op.getInput().getType();
    mlir::ValueRange input = adaptor.getInput();
    if (!partsResult)
      return rewriter.notifyMatchFailure(
          op, "repr requires unicode header/payload parts result");
    if (auto intType = mlir::dyn_cast<mlir::IntegerType>(inputType)) {
      if (input.size() != 1)
        return mlir::failure();
      if (intType.getWidth() == 1) {
        auto select = rewriter.create<mlir::scf::IfOp>(
            op.getLoc(), mlir::TypeRange(resultTypes), input.front(),
            /*withElseRegion=*/true);
        {
          mlir::OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointToStart(select.thenBlock());
          yieldUnicodeLiteral(op.getLoc(), "True", rewriter, runtime);
        }
        {
          mlir::OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointToStart(select.elseBlock());
          yieldUnicodeLiteral(op.getLoc(), "False", rewriter, runtime);
        }
        rewriter.replaceOpWithMultiple(op, asReplacement(select.getResults()));
        return mlir::success();
      }
      if (intType.getWidth() > 64)
        return rewriter.notifyMatchFailure(
            op, "primitive integer repr supports widths up to 64");
      mlir::Value asI64 = input.front();
      if (intType.getWidth() < 64)
        asI64 = rewriter.create<mlir::arith::ExtSIOp>(
            op.getLoc(), rewriter.getI64Type(), asI64);
      auto call =
          runtime.call(op.getLoc(), RuntimeSymbols::kUnicodeFromI64,
                       mlir::TypeRange(resultTypes), mlir::ValueRange{asI64});
      return replaceWithPartsCall(op, call, rewriter);
    }
    if (mlir::isa<BoolType>(inputType)) {
      if (input.size() != 1)
        return mlir::failure();
      mlir::Value bit =
          unboxBool(op.getLoc(), input.front(), rewriter, runtime);
      auto select = rewriter.create<mlir::scf::IfOp>(
          op.getLoc(), mlir::TypeRange(resultTypes), bit,
          /*withElseRegion=*/true);
      {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(select.thenBlock());
        yieldUnicodeLiteral(op.getLoc(), "True", rewriter, runtime);
      }
      {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(select.elseBlock());
        yieldUnicodeLiteral(op.getLoc(), "False", rewriter, runtime);
      }
      rewriter.replaceOpWithMultiple(op, asReplacement(select.getResults()));
      return mlir::success();
    }

    if (mlir::isa<NoneType>(inputType)) {
      llvm::SmallVector<mlir::Value, 3> values =
          unicodeLiteralOwnedParts(op.getLoc(), "None", rewriter, runtime);
      rewriter.replaceOpWithMultiple(op, asReplacement(values));
      return mlir::success();
    }

    if (mlir::isa<IntType>(inputType)) {
      if (auto literal = stripSingleInputCasts(op.getInput())
                             .getDefiningOp<IntConstantOp>()) {
        llvm::SmallVector<mlir::Value, 3> values = unicodeLiteralOwnedParts(
            op.getLoc(), literal.getValue(), rewriter, runtime);
        rewriter.replaceOpWithMultiple(op, asReplacement(values));
        return mlir::success();
      }
      if (partsResult && input.size() == 3) {
        auto call = runtime.call(op.getLoc(), RuntimeSymbols::kLongRepr,
                                 mlir::TypeRange(resultTypes), input);
        return replaceWithPartsCall(op, call, rewriter);
      }
      return rewriter.notifyMatchFailure(
          op, "int repr requires long header/meta/digits parts");
    }

    if (mlir::isa<StrType>(inputType)) {
      if (partsResult && input.size() == 2) {
        auto repr = runtime.call(op.getLoc(), RuntimeSymbols::kUnicodeCopy,
                                 mlir::TypeRange(resultTypes), input);
        return replaceWithPartsCall(op, repr, rewriter);
      }
      return rewriter.notifyMatchFailure(
          op, "str repr requires unicode header/payload parts");
    }

    if (mlir::isa<ExceptionType>(inputType)) {
      if (input.size() != 3)
        return mlir::failure();
      auto repr = runtime.call(op.getLoc(), RuntimeSymbols::kUnicodeCopy,
                               mlir::TypeRange(resultTypes),
                               mlir::ValueRange{input[1], input[2]});
      return replaceWithPartsCall(op, repr, rewriter);
    }

    return rewriter.notifyMatchFailure(op, "unsupported statically typed repr");
  }
};
struct CastToPrimLowering : public mlir::OpConversionPattern<CastToPrimOp> {
  CastToPrimLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<CastToPrimOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(CastToPrimOp op, OneToNOpAdaptor adaptor,
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
      if (adaptor.getInput().size() != 1)
        return mlir::failure();
      mlir::Value value =
          unboxBool(op.getLoc(), adaptor.getInput().front(), rewriter, runtime);
      if (!value)
        return rewriter.notifyMatchFailure(
            op, "bool cast.to_prim requires memref object storage");
      rewriter.replaceOp(op, value);
      return mlir::success();
    }

    if (inputType == IntType::get(rewriter.getContext())) {
      mlir::Value asI64 =
          longAsI64(op.getLoc(), adaptor.getInput(), rewriter, runtime);
      if (!asI64)
        return rewriter.notifyMatchFailure(
            op, "int cast.to_prim requires memref object storage");
      if (resultType == rewriter.getI64Type()) {
        rewriter.replaceOp(op, asI64);
        return mlir::success();
      }
      if (resultType == rewriter.getI32Type()) {
        rewriter.replaceOpWithNewOp<mlir::arith::TruncIOp>(op, resultType,
                                                           asI64);
        return mlir::success();
      }
    }

    if (inputType == FloatType::get(rewriter.getContext()) &&
        resultType == rewriter.getF64Type()) {
      if (adaptor.getInput().size() != 1)
        return mlir::failure();
      mlir::Value value = unboxFloat(op.getLoc(), adaptor.getInput().front(),
                                     rewriter, runtime);
      if (!value)
        return rewriter.notifyMatchFailure(
            op, "float cast.to_prim requires memref object storage");
      rewriter.replaceOp(op, value);
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
    llvm::SmallVector<mlir::Type, 3> resultTypes;
    if (mlir::failed(typeConverter->convertType(op.getResult().getType(),
                                                resultTypes)) ||
        resultTypes.empty())
      return mlir::failure();
    mlir::Type resultType =
        resultTypes.size() == 1 ? resultTypes.front() : mlir::Type();

    mlir::Value input = adaptor.getInput();
    mlir::Type inputType = input.getType();
    mlir::Type pyResultType = op.getResult().getType();

    if (inputType == rewriter.getI1Type() &&
        pyResultType == BoolType::get(rewriter.getContext())) {
      mlir::Value boxed = boxBool(op.getLoc(), input, resultType, runtime);
      if (!boxed)
        return rewriter.notifyMatchFailure(
            op, "bool cast.from_prim requires memref object storage");
      rewriter.replaceOp(op, boxed);
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
            rewriter.create<mlir::arith::ExtSIOp>(op.getLoc(), i64Type, input);
      } else if (width == 64) {
        i64Val = input;
      } else {
        return rewriter.notifyMatchFailure(
            op, "integer width > 64 not supported for from_prim");
      }

      return replaceWithLongFromI64(op, i64Val, resultTypes, runtime, rewriter);
    }

    // Handle float types -> !py.float
    if (llvm::isa<mlir::Float32Type>(inputType)) {
      // Extend f32 to f64
      auto f64Type = rewriter.getF64Type();
      mlir::Value f64Val =
          rewriter.create<mlir::arith::ExtFOp>(op.getLoc(), f64Type, input);
      mlir::Value boxed = boxFloat(op.getLoc(), f64Val, resultType, runtime);
      if (!boxed)
        return rewriter.notifyMatchFailure(
            op, "float cast.from_prim requires memref object storage");
      rewriter.replaceOp(op, boxed);
      return mlir::success();
    }

    if (llvm::isa<mlir::Float64Type>(inputType)) {
      mlir::Value boxed = boxFloat(op.getLoc(), input, resultType, runtime);
      if (!boxed)
        return rewriter.notifyMatchFailure(
            op, "float cast.from_prim requires memref object storage");
      rewriter.replaceOp(op, boxed);
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
        if (!llvm::isa<mlir::FloatType, mlir::IntegerType>(elemType))
          return rewriter.notifyMatchFailure(
              op, "from_prim for tensor requires numeric element type");
        return rewriter.notifyMatchFailure(
            op, "dynamic tensor repr requires unicode parts implementation");
      }

      size_t index = 0;
      std::string text =
          formatTensorLiteral(tensorType.getShape(), flat, index);

      if (isUnicodePartsTypes(resultTypes)) {
        llvm::SmallVector<mlir::Value, 3> values =
            unicodeLiteralOwnedParts(op.getLoc(), text, rewriter, runtime);
        rewriter.replaceOpWithMultiple(op, asReplacement(values));
        eraseDeadTensorProducerTree(input, rewriter);
        return mlir::success();
      }
      return rewriter.notifyMatchFailure(
          op, "tensor repr requires unicode header/payload parts result");
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
  patterns.add<LeLowering, LtLowering, GtLowering, GeLowering, EqLowering,
               NeLowering, ReprLowering>(typeConverter, ctx,
                                         mlir::PatternBenefit(2));
  patterns.add<StrConstantLowering, NoneLowering, IntConstantLowering,
               FloatConstantLowering, AddLowering, StrConcat3Lowering,
               SubLowering, MulLowering, DivLowering,
               IntBinaryLowering<FloorDivOp, IntBinaryKind::FloorDiv>,
               IntBinaryLowering<ModOp, IntBinaryKind::Mod>,
               IntBinaryLowering<LShiftOp, IntBinaryKind::LShift>,
               IntBinaryLowering<RShiftOp, IntBinaryKind::RShift>,
               IntBinaryLowering<BitAndOp, IntBinaryKind::BitAnd>,
               IntBinaryLowering<BitOrOp, IntBinaryKind::BitOr>,
               IntBinaryLowering<BitXorOp, IntBinaryKind::BitXor>,
               CastToPrimLowering, CastFromPrimLowering>(typeConverter, ctx);
}
} // namespace lowering::value::number::Patterns

} // namespace py
