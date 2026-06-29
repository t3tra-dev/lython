#include "PrimitiveTypes.h"

#include "AstAccess.h"
#include "TypeSystem.h"

#include "mlir/IR/MLIRContext.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

namespace lython::emitter {
namespace {

std::string canonicalPrimitiveName(const parser::Node *node,
                                   const AlgorithmM &types) {
  if (!node)
    return {};
  std::string qualified = ast::qualifiedName(node);
  llvm::StringRef name = qualified.empty()
                             ? llvm::StringRef(ast::nameSpelling(*node))
                             : llvm::StringRef(qualified);
  if (name.empty())
    return {};
  if (std::optional<std::string> resolved = types.lookupCanonicalBinding(name))
    return *resolved;
  return name.str();
}

bool primitiveNameIs(llvm::StringRef name, llvm::StringRef bareName) {
  std::string prim = (llvm::Twine("prim.") + bareName).str();
  std::string qualified = (llvm::Twine("lyrt.prim.") + bareName).str();
  return name == bareName || name == prim || name == qualified;
}

std::optional<PrimitiveTypeKind> primitiveKind(llvm::StringRef name) {
  if (primitiveNameIs(name, "Int"))
    return PrimitiveTypeKind::Int;
  if (primitiveNameIs(name, "Float"))
    return PrimitiveTypeKind::Float;
  if (primitiveNameIs(name, "Vector"))
    return PrimitiveTypeKind::Vector;
  if (primitiveNameIs(name, "Matrix"))
    return PrimitiveTypeKind::Matrix;
  if (primitiveNameIs(name, "Tensor"))
    return PrimitiveTypeKind::Tensor;
  return std::nullopt;
}

std::optional<unsigned> positiveWidth(const parser::Node *node) {
  std::optional<std::int64_t> value = integerLiteralValue(node);
  if (!value || *value <= 0)
    return std::nullopt;
  return static_cast<unsigned>(*value);
}

std::optional<mlir::FloatType> floatTypeForWidth(mlir::MLIRContext *context,
                                                 unsigned width) {
  switch (width) {
  case 16:
    return mlir::Float16Type::get(context);
  case 32:
    return mlir::Float32Type::get(context);
  case 64:
    return mlir::Float64Type::get(context);
  default:
    return std::nullopt;
  }
}

std::optional<PrimitiveTypeSpec> scalarPrimitiveSpecFromSubscript(
    const parser::Node *node, const AlgorithmM &types, PrimitiveTypeKind kind) {
  const parser::Node *slice = ast::node(*node, "slice");
  std::optional<unsigned> width = positiveWidth(slice);
  if (!width)
    return std::nullopt;
  mlir::MLIRContext *context = &types.getContext();
  if (kind == PrimitiveTypeKind::Int) {
    mlir::Type type = mlir::IntegerType::get(context, *width);
    return PrimitiveTypeSpec{kind, type, type, {}};
  }
  if (kind == PrimitiveTypeKind::Float) {
    std::optional<mlir::FloatType> type = floatTypeForWidth(context, *width);
    if (!type)
      return std::nullopt;
    return PrimitiveTypeSpec{kind, *type, *type, {}};
  }
  return std::nullopt;
}

std::optional<PrimitiveTypeSpec> shapedPrimitiveSpecFromSubscript(
    const parser::Node *node, const AlgorithmM &types, PrimitiveTypeKind kind) {
  const parser::Node *slice = ast::node(*node, "slice");
  if (!slice || slice->kind != "Tuple")
    return std::nullopt;
  const auto *elts = ast::nodeList(*slice, "elts");
  if (!elts || elts->empty())
    return std::nullopt;

  std::size_t expectedSize = 0;
  switch (kind) {
  case PrimitiveTypeKind::Vector:
    expectedSize = 2;
    break;
  case PrimitiveTypeKind::Matrix:
    expectedSize = 3;
    break;
  case PrimitiveTypeKind::Tensor:
    expectedSize = elts->size();
    if (expectedSize < 2)
      return std::nullopt;
    break;
  case PrimitiveTypeKind::Int:
  case PrimitiveTypeKind::Float:
    return std::nullopt;
  }
  if (elts->size() != expectedSize)
    return std::nullopt;

  std::optional<PrimitiveTypeSpec> element =
      primitiveTypeSpecFromSubscript(elts->front().get(), types);
  if (!element || !element->isScalar())
    return std::nullopt;

  llvm::SmallVector<std::int64_t, 4> shape;
  shape.reserve(elts->size() - 1);
  for (const parser::NodePtr &dimNode : llvm::drop_begin(*elts)) {
    std::optional<std::int64_t> dim = integerLiteralValue(dimNode.get());
    if (!dim || *dim <= 0)
      return std::nullopt;
    shape.push_back(*dim);
  }

  mlir::Type type = mlir::RankedTensorType::get(shape, element->elementType);
  return PrimitiveTypeSpec{kind, type, element->elementType, shape};
}

mlir::Type nestedListType(mlir::Type elementType,
                          llvm::ArrayRef<std::int64_t> shape,
                          const AlgorithmM &types) {
  mlir::Type result = elementType;
  for (std::int64_t ignored : llvm::reverse(shape)) {
    (void)ignored;
    result = types.listOf(result);
  }
  return result;
}

} // namespace

bool isLyrtPrimitiveIntName(llvm::StringRef name) {
  return primitiveNameIs(name, "Int");
}

bool isLyrtPrimitiveTypeName(llvm::StringRef name) {
  return static_cast<bool>(primitiveKind(name));
}

std::optional<std::int64_t> integerLiteralValue(const parser::Node *node) {
  if (!node)
    return std::nullopt;
  if (node->kind == "Constant") {
    if (auto value = ast::integer(*node, "value"))
      return *value;
    return std::nullopt;
  }
  if (node->kind == "UnaryOp" &&
      ast::isOperator(ast::node(*node, "op"), "USub")) {
    std::optional<std::int64_t> value =
        integerLiteralValue(ast::node(*node, "operand"));
    if (value)
      return -*value;
  }
  return std::nullopt;
}

std::optional<PrimitiveTypeSpec>
primitiveTypeSpecFromSubscript(const parser::Node *node,
                               const AlgorithmM &types) {
  if (!node || node->kind != "Subscript")
    return std::nullopt;
  std::string canonical =
      canonicalPrimitiveName(ast::node(*node, "value"), types);
  std::optional<PrimitiveTypeKind> kind = primitiveKind(canonical);
  if (!kind)
    return std::nullopt;
  if (*kind == PrimitiveTypeKind::Int || *kind == PrimitiveTypeKind::Float)
    return scalarPrimitiveSpecFromSubscript(node, types, *kind);
  return shapedPrimitiveSpecFromSubscript(node, types, *kind);
}

std::optional<unsigned>
primitiveIntWidthFromSubscript(const parser::Node *node,
                               const AlgorithmM &types) {
  std::optional<PrimitiveTypeSpec> spec =
      primitiveTypeSpecFromSubscript(node, types);
  if (!spec || spec->kind != PrimitiveTypeKind::Int)
    return std::nullopt;
  return mlir::cast<mlir::IntegerType>(spec->type).getWidth();
}

std::optional<mlir::IntegerType>
primitiveIntTypeFromSubscript(const parser::Node *node,
                              const AlgorithmM &types) {
  std::optional<PrimitiveTypeSpec> spec =
      primitiveTypeSpecFromSubscript(node, types);
  if (!spec || spec->kind != PrimitiveTypeKind::Int)
    return std::nullopt;
  return mlir::cast<mlir::IntegerType>(spec->type);
}

std::optional<std::pair<mlir::IntegerType, std::int64_t>>
primitiveIntegerConstantConstructor(const parser::Node *node,
                                    const AlgorithmM &types) {
  if (!node || node->kind != "Call")
    return std::nullopt;
  std::optional<mlir::IntegerType> type =
      primitiveIntTypeFromSubscript(ast::node(*node, "func"), types);
  if (!type)
    return std::nullopt;
  const auto *args = ast::nodeList(*node, "args");
  const auto *keywords = ast::nodeList(*node, "keywords");
  if (!args || args->size() != 1 || (keywords && !keywords->empty()))
    return std::nullopt;
  std::optional<std::int64_t> value = integerLiteralValue(args->front().get());
  if (!value)
    return std::nullopt;
  return std::make_pair(*type, *value);
}

mlir::Type primitivePythonResultType(mlir::Type primitiveType,
                                     const AlgorithmM &types) {
  if (mlir::isa<mlir::IntegerType>(primitiveType))
    return types.intType();
  if (mlir::isa<mlir::FloatType>(primitiveType))
    return types.floatType();
  if (auto tensor =
          mlir::dyn_cast_if_present<mlir::RankedTensorType>(primitiveType)) {
    mlir::Type element =
        primitivePythonResultType(tensor.getElementType(), types);
    return nestedListType(element, tensor.getShape(), types);
  }
  return types.object();
}

} // namespace lython::emitter
