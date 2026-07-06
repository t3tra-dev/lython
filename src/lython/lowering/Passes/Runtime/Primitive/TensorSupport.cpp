#include "TensorSupport.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "llvm/ADT/SmallVector.h"

#include <limits>

namespace py::lowering {

bool isPrimitiveElementType(mlir::Type type) {
  return mlir::isa<mlir::FloatType, mlir::IntegerType>(type);
}

std::optional<int64_t> primitiveElementBitWidth(mlir::Type type) {
  if (auto floatType = mlir::dyn_cast<mlir::FloatType>(type))
    return floatType.getWidth();
  if (auto intType = mlir::dyn_cast<mlir::IntegerType>(type))
    return intType.getWidth();
  return std::nullopt;
}

std::optional<mlir::Value> createPrimitiveZeroValue(mlir::OpBuilder &builder,
                                                    mlir::Location loc,
                                                    mlir::Type type) {
  if (auto floatType = mlir::dyn_cast<mlir::FloatType>(type)) {
    return mlir::arith::ConstantOp::create(
        builder, loc, floatType, builder.getFloatAttr(floatType, 0.0));
  }
  if (auto intType = mlir::dyn_cast<mlir::IntegerType>(type)) {
    return mlir::arith::ConstantOp::create(builder, loc, intType,
                                           builder.getIntegerAttr(intType, 0));
  }
  return std::nullopt;
}

bool memrefHasContiguousInnerDimension(mlir::MemRefType type) {
  if (!type.hasStaticShape() || type.getRank() == 0)
    return false;

  llvm::SmallVector<int64_t, 4> strides;
  int64_t offset = 0;
  if (mlir::failed(type.getStridesAndOffset(strides, offset)))
    return false;
  return strides.back() == 1;
}

std::uint64_t staticElementCount(mlir::MemRefType type) {
  if (!type.hasStaticShape())
    return 0;

  constexpr std::uint64_t max = std::numeric_limits<std::uint64_t>::max();
  std::uint64_t count = 1;
  for (int64_t dim : type.getShape()) {
    if (dim <= 0)
      return 0;
    std::uint64_t value = static_cast<std::uint64_t>(dim);
    if (count > max / value)
      return max;
    count *= value;
  }
  return count;
}

mlir::Value sourceMemrefRoot(mlir::Value value) {
  while (true) {
    if (auto subview = value.getDefiningOp<mlir::memref::SubViewOp>()) {
      value = subview.getSource();
      continue;
    }
    if (auto cast = value.getDefiningOp<mlir::memref::ReinterpretCastOp>()) {
      value = cast.getSource();
      continue;
    }
    break;
  }
  return value;
}

bool sameMemrefRoot(mlir::Value lhs, mlir::Value rhs) {
  return sourceMemrefRoot(lhs) == sourceMemrefRoot(rhs);
}

std::optional<mlir::Value>
createSubviewOffsetForDimension(mlir::OpBuilder &builder, mlir::Location loc,
                                mlir::Value value, unsigned dimension) {
  if (auto subview = value.getDefiningOp<mlir::memref::SubViewOp>()) {
    if (dimension >= subview.getType().getRank() ||
        dimension >= subview.getMixedOffsets().size())
      return std::nullopt;
    std::optional<mlir::Value> baseOffset = createSubviewOffsetForDimension(
        builder, loc, subview.getSource(), dimension);
    if (!baseOffset)
      return std::nullopt;
    mlir::Value localOffset = mlir::getValueOrCreateConstantIndexOp(
        builder, loc, subview.getMixedOffsets()[dimension]);
    return mlir::arith::AddIOp::create(builder, loc, *baseOffset, localOffset)
        .getResult();
  }
  if (auto cast = value.getDefiningOp<mlir::memref::CastOp>())
    return createSubviewOffsetForDimension(builder, loc, cast.getSource(),
                                           dimension);

  auto type = mlir::dyn_cast<mlir::MemRefType>(value.getType());
  if (!type || dimension >= type.getRank())
    return std::nullopt;
  return mlir::arith::ConstantIndexOp::create(builder, loc, 0).getResult();
}

std::optional<int64_t> selectDivisibleVectorLanes(mlir::Type elementType,
                                                  int64_t tripCount,
                                                  int64_t vectorBits,
                                                  int64_t minLanes) {
  std::optional<int64_t> bitWidth = primitiveElementBitWidth(elementType);
  if (!bitWidth || *bitWidth <= 0 || tripCount < minLanes || vectorBits <= 0 ||
      minLanes < 1)
    return std::nullopt;

  int64_t lanes = vectorBits / *bitWidth;
  if (lanes < minLanes)
    return std::nullopt;
  if (lanes > tripCount)
    lanes = tripCount;
  while (lanes > minLanes && tripCount % lanes != 0)
    --lanes;
  if (tripCount % lanes != 0)
    return std::nullopt;
  return lanes;
}

} // namespace py::lowering
