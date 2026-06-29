#pragma once

#include "Ast.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>
#include <utility>

namespace lython::emitter {

class AlgorithmM;

enum class PrimitiveTypeKind { Int, Float, Vector, Matrix, Tensor };

struct PrimitiveTypeSpec {
  PrimitiveTypeKind kind;
  mlir::Type type;
  mlir::Type elementType;
  llvm::SmallVector<std::int64_t, 4> shape;

  bool isScalar() const { return shape.empty(); }
  bool isShaped() const { return !shape.empty(); }
};

bool isLyrtPrimitiveIntName(llvm::StringRef name);
bool isLyrtPrimitiveTypeName(llvm::StringRef name);
std::optional<std::int64_t> integerLiteralValue(const parser::Node *node);

std::optional<PrimitiveTypeSpec>
primitiveTypeSpecFromSubscript(const parser::Node *node,
                               const AlgorithmM &types);
std::optional<unsigned> primitiveIntWidthFromSubscript(const parser::Node *node,
                                                       const AlgorithmM &types);
std::optional<mlir::IntegerType>
primitiveIntTypeFromSubscript(const parser::Node *node,
                              const AlgorithmM &types);
std::optional<std::pair<mlir::IntegerType, std::int64_t>>
primitiveIntegerConstantConstructor(const parser::Node *node,
                                    const AlgorithmM &types);
mlir::Type primitivePythonResultType(mlir::Type primitiveType,
                                     const AlgorithmM &types);

} // namespace lython::emitter
