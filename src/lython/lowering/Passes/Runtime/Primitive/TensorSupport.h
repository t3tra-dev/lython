#pragma once

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"

#include <cstdint>
#include <optional>

namespace mlir {
class Location;
class OpBuilder;
} // namespace mlir

namespace py::lowering {

inline constexpr std::uint64_t kPrimitiveTensorPackedMinSourceElements =
    256ull * 256ull;

bool isPrimitiveElementType(mlir::Type type);

mlir::Value createIndexConstant(mlir::OpBuilder &builder, mlir::Location loc,
                                std::int64_t value);
bool isBlockArgumentDefinedInside(mlir::Value value, mlir::Operation *scope);

std::optional<int64_t> primitiveElementBitWidth(mlir::Type type);

std::optional<mlir::Value> createPrimitiveZeroValue(mlir::OpBuilder &builder,
                                                    mlir::Location loc,
                                                    mlir::Type type);

bool memrefHasContiguousInnerDimension(mlir::MemRefType type);

std::uint64_t staticElementCount(mlir::MemRefType type);

mlir::Value sourceMemrefRoot(mlir::Value value);

bool sameMemrefRoot(mlir::Value lhs, mlir::Value rhs);

std::optional<mlir::Value>
createSubviewOffsetForDimension(mlir::OpBuilder &builder, mlir::Location loc,
                                mlir::Value value, unsigned dimension);

std::optional<int64_t> selectDivisibleVectorLanes(mlir::Type elementType,
                                                  int64_t tripCount,
                                                  int64_t vectorBits,
                                                  int64_t minLanes);

} // namespace py::lowering
