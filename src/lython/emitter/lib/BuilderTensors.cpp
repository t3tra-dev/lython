#include "BuilderImpl.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "llvm/ADT/STLExtras.h"

namespace lython::emitter {

namespace {

std::optional<double> numericConstant(const parser::Node &node) {
  if (node.kind != "Constant")
    return std::nullopt;
  const parser::FieldValue *value = valueField(node, "value");
  if (!value)
    return std::nullopt;
  if (const auto *number = std::get_if<double>(value))
    return *number;
  if (const auto *integer = std::get_if<std::int64_t>(value))
    return static_cast<double>(*integer);
  return std::nullopt;
}

std::optional<llvm::SmallVector<int64_t>>
shapeOfLiteralNode(const parser::Node &node) {
  if (numericConstant(node))
    return llvm::SmallVector<int64_t>{};
  if (node.kind != "List" && node.kind != "Tuple")
    return std::nullopt;
  const std::vector<parser::NodePtr> *elements = nodeListField(node, "elts");
  if (!elements)
    return std::nullopt;
  llvm::SmallVector<int64_t> shape{static_cast<int64_t>(elements->size())};
  if (elements->empty())
    return shape;
  if (!elements->front())
    return std::nullopt;
  std::optional<llvm::SmallVector<int64_t>> childShape =
      shapeOfLiteralNode(*elements->front());
  if (!childShape)
    return std::nullopt;
  for (const parser::NodePtr &child : llvm::drop_begin(*elements)) {
    if (!child)
      return std::nullopt;
    std::optional<llvm::SmallVector<int64_t>> currentShape =
        shapeOfLiteralNode(*child);
    if (!currentShape || *currentShape != *childShape)
      return std::nullopt;
  }
  shape.append(childShape->begin(), childShape->end());
  return shape;
}

bool flattenLiteralNode(const parser::Node &node,
                        llvm::SmallVectorImpl<double> &out) {
  if (std::optional<double> number = numericConstant(node)) {
    out.push_back(*number);
    return true;
  }
  if (node.kind != "List" && node.kind != "Tuple")
    return false;
  const std::vector<parser::NodePtr> *elements = nodeListField(node, "elts");
  if (!elements)
    return false;
  for (const parser::NodePtr &child : *elements) {
    if (!child || !flattenLiteralNode(*child, out))
      return false;
  }
  return true;
}

} // namespace

Value Builder::Impl::emitTensorConstructor(const parser::Node &expr) {
  const parser::NodePtr *func = nodeField(expr, "func");
  const std::vector<parser::NodePtr> *args = nodeListField(expr, "args");
  const std::vector<parser::NodePtr> *keywords =
      nodeListField(expr, "keywords");
  if (!func || !*func || !isTensorConstructorCallee(**func))
    return Value{{}, noneType()};
  if (!args || args->size() != 1 || !args->front() ||
      (keywords && !keywords->empty())) {
    error(expr, "primitive tensor constructor expects one literal argument");
    return Value{{}, noneType()};
  }

  std::optional<mlir::Type> parsedType = typeFromAnnotation(*func);
  auto tensorType = parsedType
                        ? mlir::dyn_cast<mlir::RankedTensorType>(*parsedType)
                        : mlir::RankedTensorType();
  if (!tensorType) {
    error(**func, "primitive tensor constructor has unsupported type");
    return Value{{}, noneType()};
  }

  std::optional<llvm::SmallVector<int64_t>> actualShape =
      shapeOfLiteralNode(*args->front());
  if (!actualShape) {
    error(*args->front(), "primitive tensor constructor requires literals");
    return Value{{}, tensorType};
  }
  if (llvm::ArrayRef(*actualShape) != tensorType.getShape()) {
    error(*args->front(),
          "tensor literal shape does not match constructor type");
    return Value{{}, tensorType};
  }

  llvm::SmallVector<double> flat;
  if (!flattenLiteralNode(*args->front(), flat)) {
    error(*args->front(), "primitive tensor constructor requires literals");
    return Value{{}, tensorType};
  }
  llvm::SmallVector<mlir::Value> elements;
  elements.reserve(flat.size());
  mlir::Type elementType = tensorType.getElementType();
  for (double number : flat) {
    if (auto floatType = mlir::dyn_cast<mlir::FloatType>(elementType)) {
      mlir::Value value = builder.create<mlir::arith::ConstantOp>(
          loc(expr), elementType, builder.getFloatAttr(floatType, number));
      elements.push_back(value);
      continue;
    }
    if (auto intType = mlir::dyn_cast<mlir::IntegerType>(elementType)) {
      mlir::Value value = builder.create<mlir::arith::ConstantIntOp>(
          loc(expr), static_cast<int64_t>(number), intType.getWidth());
      elements.push_back(value);
      continue;
    }
    error(expr, "unsupported tensor element type " + typeString(elementType));
    return Value{{}, tensorType};
  }

  mlir::Value tensor = builder.create<mlir::tensor::FromElementsOp>(
      loc(expr), tensorType, elements);
  return Value{tensor, tensorType};
}

Value Builder::Impl::emitTensorMatmul(const parser::Node &expr,
                                      const Value &lhs, const Value &rhs) {
  auto lhsType = mlir::dyn_cast<mlir::RankedTensorType>(lhs.type);
  auto rhsType = mlir::dyn_cast<mlir::RankedTensorType>(rhs.type);
  if (!lhsType || !rhsType || lhsType.getRank() != 2 ||
      rhsType.getRank() != 2) {
    error(expr, "matrix multiplication requires rank-2 tensors");
    return Value{{}, noneType()};
  }
  if (lhsType.getElementType() != rhsType.getElementType()) {
    error(expr, "matrix multiplication element types must match");
    return Value{{}, noneType()};
  }
  if (lhsType.getDimSize(1) != rhsType.getDimSize(0)) {
    error(expr, "matrix multiplication dimensions are incompatible");
    return Value{{}, noneType()};
  }

  mlir::Type elementType = lhsType.getElementType();
  llvm::SmallVector<int64_t> resultShape{lhsType.getDimSize(0),
                                         rhsType.getDimSize(1)};
  auto resultType = mlir::RankedTensorType::get(resultShape, elementType);
  mlir::Value zero;
  if (auto floatType = mlir::dyn_cast<mlir::FloatType>(elementType)) {
    zero = builder.create<mlir::arith::ConstantOp>(
        loc(expr), elementType, builder.getFloatAttr(floatType, 0.0));
  } else {
    error(expr, "matrix multiplication currently supports float tensors");
    return Value{{}, resultType};
  }

  mlir::Value empty = builder.create<mlir::tensor::EmptyOp>(
      loc(expr), resultType, mlir::ValueRange{});
  auto fill = builder.create<mlir::linalg::FillOp>(
      loc(expr), mlir::TypeRange{resultType}, mlir::ValueRange{zero},
      mlir::ValueRange{empty}, llvm::ArrayRef<mlir::NamedAttribute>{});

  mlir::AffineExpr d0 = builder.getAffineDimExpr(0);
  mlir::AffineExpr d1 = builder.getAffineDimExpr(1);
  mlir::AffineExpr d2 = builder.getAffineDimExpr(2);
  mlir::ArrayAttr maps = builder.getAffineMapArrayAttr(
      {mlir::AffineMap::get(3, 0, {d0, d2}, &context),
       mlir::AffineMap::get(3, 0, {d2, d1}, &context),
       mlir::AffineMap::get(3, 0, {d0, d1}, &context)});
  mlir::ArrayAttr iterators = builder.getArrayAttr(
      {mlir::linalg::IteratorTypeAttr::get(&context,
                                           mlir::utils::IteratorType::parallel),
       mlir::linalg::IteratorTypeAttr::get(&context,
                                           mlir::utils::IteratorType::parallel),
       mlir::linalg::IteratorTypeAttr::get(
           &context, mlir::utils::IteratorType::reduction)});

  auto generic = builder.create<mlir::linalg::GenericOp>(
      loc(expr), mlir::TypeRange{resultType},
      mlir::ValueRange{lhs.value, rhs.value},
      mlir::ValueRange{fill.getResult(0)}, maps, iterators, mlir::StringAttr{},
      mlir::StringAttr{});

  mlir::Block *block = new mlir::Block();
  generic.getRegion().push_back(block);
  block->addArgument(elementType, loc(expr));
  block->addArgument(elementType, loc(expr));
  block->addArgument(elementType, loc(expr));
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(block);
  mlir::Value product = builder.create<mlir::arith::MulFOp>(
      loc(expr), block->getArgument(0), block->getArgument(1));
  mlir::Value sum = builder.create<mlir::arith::AddFOp>(loc(expr), product,
                                                        block->getArgument(2));
  builder.create<mlir::linalg::YieldOp>(loc(expr), sum);

  return Value{generic.getResult(0), resultType};
}

} // namespace lython::emitter
