#include "EmitterCore.h"
#include "EmitterPyOps.h"
#include "EmitterSupport.h"
#include "PrimitiveTypes.h"

#include "AstAccess.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/ADT/STLExtras.h"

#include <functional>
#include <optional>
#include <string>

namespace lython::emitter {
namespace {

std::optional<double> numericLiteralValue(const parser::Node *node) {
  if (!node)
    return std::nullopt;
  if (node->kind == "Constant") {
    if (auto value = ast::floating(*node, "value"))
      return *value;
    if (auto value = ast::integer(*node, "value"))
      return static_cast<double>(*value);
    return std::nullopt;
  }
  if (node->kind == "UnaryOp" &&
      ast::isOperator(ast::node(*node, "op"), "USub")) {
    std::optional<double> value =
        numericLiteralValue(ast::node(*node, "operand"));
    if (value)
      return -*value;
  }
  return std::nullopt;
}

std::optional<mlir::Attribute> primitiveElementAttr(mlir::OpBuilder &builder,
                                                    mlir::Type elementType,
                                                    const parser::Node *node) {
  if (auto integer = mlir::dyn_cast<mlir::IntegerType>(elementType)) {
    std::optional<std::int64_t> value = integerLiteralValue(node);
    if (!value)
      return std::nullopt;
    return builder.getIntegerAttr(integer, *value);
  }
  if (auto floating = mlir::dyn_cast<mlir::FloatType>(elementType)) {
    std::optional<double> value = numericLiteralValue(node);
    if (!value)
      return std::nullopt;
    return builder.getFloatAttr(floating, *value);
  }
  return std::nullopt;
}

bool collectTensorLiteralAttrs(mlir::OpBuilder &builder,
                               mlir::RankedTensorType tensorType,
                               const parser::Node *node, unsigned depth,
                               llvm::SmallVectorImpl<mlir::Attribute> &attrs) {
  if (depth == static_cast<unsigned>(tensorType.getRank())) {
    std::optional<mlir::Attribute> attr =
        primitiveElementAttr(builder, tensorType.getElementType(), node);
    if (!attr)
      return false;
    attrs.push_back(*attr);
    return true;
  }

  if (!node || (node->kind != "List" && node->kind != "Tuple"))
    return false;
  const auto *elts = ast::nodeList(*node, "elts");
  if (!elts ||
      elts->size() != static_cast<std::size_t>(tensorType.getDimSize(depth)))
    return false;
  for (const parser::NodePtr &elt : *elts)
    if (!collectTensorLiteralAttrs(builder, tensorType, elt.get(), depth + 1,
                                   attrs))
      return false;
  return true;
}

mlir::Value constantIndex(mlir::OpBuilder &builder, mlir::Location loc,
                          std::int64_t value) {
  return builder.create<mlir::arith::ConstantIndexOp>(loc, value).getResult();
}

mlir::Value tensorExtract(mlir::OpBuilder &builder, mlir::Location loc,
                          mlir::Value tensor,
                          llvm::ArrayRef<std::int64_t> indices) {
  llvm::SmallVector<mlir::Value, 4> indexValues;
  indexValues.reserve(indices.size());
  for (std::int64_t index : indices)
    indexValues.push_back(constantIndex(builder, loc, index));
  return builder.create<mlir::tensor::ExtractOp>(loc, tensor, indexValues)
      .getResult();
}

mlir::Value coerceFloatValue(mlir::OpBuilder &builder, mlir::Location loc,
                             mlir::Value value, mlir::FloatType targetType) {
  auto sourceType = mlir::dyn_cast<mlir::FloatType>(value.getType());
  if (!sourceType || sourceType == targetType)
    return value;
  unsigned sourceWidth = sourceType.getWidth();
  unsigned targetWidth = targetType.getWidth();
  if (sourceWidth < targetWidth)
    return builder.create<mlir::arith::ExtFOp>(loc, targetType, value)
        .getResult();
  if (sourceWidth > targetWidth)
    return builder.create<mlir::arith::TruncFOp>(loc, targetType, value)
        .getResult();
  return value;
}

std::optional<mlir::Value>
createScalarBinary(mlir::OpBuilder &builder, mlir::Location loc,
                   const parser::Node *op, mlir::Value lhs, mlir::Value rhs) {
  mlir::Type type = lhs.getType();
  if (auto integer = mlir::dyn_cast<mlir::IntegerType>(type)) {
    (void)integer;
    if (rhs.getType() != type)
      return std::nullopt;
    if (ast::isOperator(op, "Sub"))
      return builder.create<mlir::arith::SubIOp>(loc, lhs, rhs).getResult();
    if (ast::isOperator(op, "Mult"))
      return builder.create<mlir::arith::MulIOp>(loc, lhs, rhs).getResult();
    if (ast::isOperator(op, "Add"))
      return builder.create<mlir::arith::AddIOp>(loc, lhs, rhs).getResult();
    return std::nullopt;
  }
  if (auto floating = mlir::dyn_cast<mlir::FloatType>(type)) {
    rhs = coerceFloatValue(builder, loc, rhs, floating);
    if (rhs.getType() != type)
      return std::nullopt;
    if (ast::isOperator(op, "Sub"))
      return builder.create<mlir::arith::SubFOp>(loc, lhs, rhs).getResult();
    if (ast::isOperator(op, "Mult"))
      return builder.create<mlir::arith::MulFOp>(loc, lhs, rhs).getResult();
    if (ast::isOperator(op, "Add"))
      return builder.create<mlir::arith::AddFOp>(loc, lhs, rhs).getResult();
    return std::nullopt;
  }
  return std::nullopt;
}

void enumerateTensorIndices(
    llvm::ArrayRef<std::int64_t> shape,
    llvm::function_ref<void(llvm::ArrayRef<std::int64_t>)> callback,
    llvm::SmallVectorImpl<std::int64_t> &current, unsigned depth = 0) {
  if (depth == shape.size()) {
    callback(current);
    return;
  }
  for (std::int64_t index = 0; index < shape[depth]; ++index) {
    current.push_back(index);
    enumerateTensorIndices(shape, callback, current, depth + 1);
    current.pop_back();
  }
}

std::optional<Value>
emitElementwiseTensorBinary(mlir::OpBuilder &builder, mlir::Location loc,
                            const parser::Node &expr, Value lhs, Value rhs,
                            const parser::Node *op,
                            mlir::RankedTensorType tensorType) {
  if (rhs.value.getType() != tensorType)
    return std::nullopt;

  llvm::SmallVector<mlir::Value, 8> elements;
  llvm::SmallVector<std::int64_t, 4> current;
  enumerateTensorIndices(
      tensorType.getShape(),
      [&](llvm::ArrayRef<std::int64_t> indices) {
        mlir::Value lhsElement =
            tensorExtract(builder, loc, lhs.value, indices);
        mlir::Value rhsElement =
            tensorExtract(builder, loc, rhs.value, indices);
        std::optional<mlir::Value> result =
            createScalarBinary(builder, loc, op, lhsElement, rhsElement);
        if (result)
          elements.push_back(*result);
      },
      current);
  if (elements.size() != static_cast<std::size_t>(tensorType.getNumElements()))
    return std::nullopt;

  auto result =
      builder.create<mlir::tensor::FromElementsOp>(loc, tensorType, elements);
  return Value{result.getResult(), tensorType};
}

std::optional<Value> emitMatrixMatmul(mlir::OpBuilder &builder,
                                      mlir::Location loc,
                                      const parser::Node &expr, Value lhs,
                                      Value rhs, mlir::RankedTensorType lhsType,
                                      mlir::RankedTensorType rhsType) {
  (void)expr;
  if (lhsType.getRank() != 2 || rhsType.getRank() != 2 ||
      lhsType.getElementType() != rhsType.getElementType() ||
      lhsType.getDimSize(1) != rhsType.getDimSize(0))
    return std::nullopt;

  llvm::SmallVector<std::int64_t, 2> resultShape{lhsType.getDimSize(0),
                                                 rhsType.getDimSize(1)};
  auto resultType =
      mlir::RankedTensorType::get(resultShape, lhsType.getElementType());

  mlir::Type elementType = lhsType.getElementType();
  mlir::Value zero;
  if (auto integer = mlir::dyn_cast<mlir::IntegerType>(elementType)) {
    zero = builder
               .create<mlir::arith::ConstantOp>(
                   loc, integer, builder.getIntegerAttr(integer, 0))
               .getResult();
  } else if (auto floating = mlir::dyn_cast<mlir::FloatType>(elementType)) {
    zero = builder
               .create<mlir::arith::ConstantOp>(
                   loc, floating, builder.getFloatAttr(floating, 0.0))
               .getResult();
  } else {
    return std::nullopt;
  }

  // Keep matrix multiplication as a structured contraction.  Expanding it
  // here would generate O(M*N*K) scalar IR and would also hide the alias-free
  // contraction from linalg/affine/vector lowering.
  auto init =
      builder.create<mlir::tensor::EmptyOp>(loc, resultType, mlir::ValueRange{})
          .getResult();
  auto filled = builder.create<mlir::linalg::FillOp>(
      loc, mlir::TypeRange{resultType}, mlir::ValueRange{zero},
      mlir::ValueRange{init});
  auto matmul = builder.create<mlir::linalg::MatmulOp>(
      loc, mlir::TypeRange{resultType}, mlir::ValueRange{lhs.value, rhs.value},
      mlir::ValueRange{filled.getResult(0)});
  return Value{matmul.getResult(0), resultType};
}

} // namespace

std::optional<Value>
ModuleEmitter::emitPrimitiveConstructorCall(const parser::Node &expr,
                                            const parser::Node *calleeNode) {
  std::optional<PrimitiveTypeSpec> primitive =
      primitiveTypeSpecFromSubscript(calleeNode, types);
  if (!primitive)
    return std::nullopt;

  const auto *args = ast::nodeList(expr, "args");
  const auto *keywords = ast::nodeList(expr, "keywords");
  if (!args || args->size() != 1 || (keywords && !keywords->empty())) {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, expr.range.start,
        "lyrt.prim constructor expects exactly one positional argument"});
    return emitNone(expr);
  }

  if (auto primitiveInt = mlir::dyn_cast<mlir::IntegerType>(primitive->type)) {
    if (std::optional<std::int64_t> literal =
            integerLiteralValue(args->front().get()))
      return emitPrimitiveConstant(expr,
                                   PrimitiveConstant{primitiveInt, *literal});
    return coercePrimitiveInteger(emitExpr(args->front().get()), primitiveInt,
                                  expr);
  }

  if (auto primitiveFloat = mlir::dyn_cast<mlir::FloatType>(primitive->type)) {
    if (std::optional<double> literal =
            numericLiteralValue(args->front().get())) {
      auto attr = builder.getFloatAttr(primitiveFloat, *literal);
      auto op = builder.create<mlir::arith::ConstantOp>(loc(expr),
                                                        primitiveFloat, attr);
      return Value{op.getResult(), primitiveFloat};
    }
    Value value = emitExpr(args->front().get());
    if (auto sourceFloat =
            mlir::dyn_cast<mlir::FloatType>(value.value.getType())) {
      (void)sourceFloat;
      mlir::Value coerced =
          coerceFloatValue(builder, loc(expr), value.value, primitiveFloat);
      return Value{coerced, primitiveFloat};
    }
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, expr.range.start,
        "lyrt.prim.Float constructor currently requires a numeric literal or "
        "primitive float value"});
    return emitNone(expr);
  }

  auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(primitive->type);
  if (!tensorType)
    return std::nullopt;

  llvm::SmallVector<mlir::Attribute, 16> attrs;
  if (!collectTensorLiteralAttrs(builder, tensorType, args->front().get(),
                                 /*depth=*/0, attrs)) {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, expr.range.start,
        "lyrt.prim tensor constructor requires a nested numeric literal whose "
        "shape matches the type parameters"});
    return emitNone(expr);
  }
  auto dense = mlir::DenseElementsAttr::get(tensorType, attrs);
  auto op =
      builder.create<mlir::arith::ConstantOp>(loc(expr), tensorType, dense);
  return Value{op.getResult(), tensorType};
}

std::optional<Value>
ModuleEmitter::emitPrimitiveRuntimeCall(const parser::Node &expr,
                                        const parser::Node *calleeNode) {
  if (!calleeNode)
    return std::nullopt;

  std::string qualified = ast::qualifiedName(calleeNode);
  llvm::StringRef name = qualified.empty()
                             ? llvm::StringRef(ast::nameSpelling(*calleeNode))
                             : llvm::StringRef(qualified);
  std::optional<std::string> canonical = types.lookupCanonicalBinding(name);
  if (!canonical || *canonical != "lyrt.from_prim")
    return std::nullopt;

  const auto *args = ast::nodeList(expr, "args");
  const auto *keywords = ast::nodeList(expr, "keywords");
  if (!args || args->size() != 1 || (keywords && !keywords->empty())) {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, expr.range.start,
        "lyrt.from_prim expects exactly one positional argument"});
    return emitNone(expr);
  }

  Value input = emitExpr(args->front().get());
  if (mlir::isa<mlir::IntegerType>(input.value.getType())) {
    auto op = builder.create<py::CastFromPrimOp>(loc(expr), types.intType(),
                                                 input.value);
    return Value{op.getResult(), types.intType()};
  }
  if (auto primitiveFloat =
          mlir::dyn_cast<mlir::FloatType>(input.value.getType())) {
    mlir::Value raw = input.value;
    if (!primitiveFloat.isF64())
      raw =
          builder
              .create<mlir::arith::ExtFOp>(loc(expr), builder.getF64Type(), raw)
              .getResult();
    auto op =
        builder.create<py::CastFromPrimOp>(loc(expr), types.floatType(), raw);
    return Value{op.getResult(), types.floatType()};
  }
  if (auto tensorType =
          mlir::dyn_cast<mlir::RankedTensorType>(input.value.getType())) {
    llvm::SmallVector<std::int64_t, 4> indices;
    std::function<Value(unsigned)> emitNested = [&](unsigned depth) -> Value {
      if (depth == static_cast<unsigned>(tensorType.getRank())) {
        mlir::Value element =
            tensorExtract(builder, loc(expr), input.value, indices);
        mlir::Type elementType = tensorType.getElementType();
        mlir::Type resultType = primitivePythonResultType(elementType, types);
        if (mlir::isa<mlir::IntegerType>(elementType)) {
          auto cast = builder.create<py::CastFromPrimOp>(loc(expr), resultType,
                                                         element);
          return Value{cast.getResult(), resultType};
        }
        if (auto floatType = mlir::dyn_cast<mlir::FloatType>(elementType)) {
          if (!floatType.isF64())
            element = builder
                          .create<mlir::arith::ExtFOp>(
                              loc(expr), builder.getF64Type(), element)
                          .getResult();
          auto cast = builder.create<py::CastFromPrimOp>(loc(expr), resultType,
                                                         element);
          return Value{cast.getResult(), resultType};
        }
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, expr.range.start,
            "lyrt.from_prim tensor element type is not supported"});
        return emitNone(expr);
      }

      llvm::SmallVector<Value, 8> children;
      for (std::int64_t index = 0; index < tensorType.getDimSize(depth);
           ++index) {
        indices.push_back(index);
        children.push_back(emitNested(depth + 1));
        indices.pop_back();
      }
      llvm::SmallVector<mlir::Value, 8> operands;
      operands.reserve(children.size());
      for (Value child : children)
        operands.push_back(child.value);
      mlir::Type elementType =
          children.empty() ? types.object() : children.front().type;
      mlir::Type resultType = types.listOf(elementType);
      auto pack = builder.create<py::PackOp>(loc(expr), resultType, operands);
      return Value{pack.getResult(), resultType};
    };
    return emitNested(/*depth=*/0);
  }

  {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, expr.range.start,
        "lyrt.from_prim expects a primitive scalar or shaped primitive value"});
    return input;
  }
}

std::optional<Value>
ModuleEmitter::emitDirectPrimitiveFunctionCall(const parser::Node &expr,
                                               const parser::Node *calleeNode) {
  if (!calleeNode || calleeNode->kind != "Name")
    return std::nullopt;

  llvm::StringRef name = ast::nameSpelling(*calleeNode);
  auto symbolType = types.lookupSymbol(name);
  auto callable = mlir::dyn_cast_if_present<py::CallableType>(
      symbolType ? *symbolType : mlir::Type());
  if (!isPrimitiveOnlyCallable(callable))
    return std::nullopt;

  auto target = module.lookupSymbol<mlir::func::FuncOp>(name);
  if (!target)
    return std::nullopt;

  const auto *args = ast::nodeList(expr, "args");
  const auto *keywords = ast::nodeList(expr, "keywords");
  if (!args || args->size() != callable.getPositionalTypes().size() ||
      (keywords && !keywords->empty()))
    return std::nullopt;
  if (llvm::any_of(*args, [](const parser::NodePtr &arg) {
        return arg && arg->kind == "Starred";
      }))
    return std::nullopt;

  CallOperands callOperands = emitCallOperands(expr);
  llvm::SmallVector<mlir::Value, 8> operands;
  operands.reserve(callOperands.positional.size());
  for (auto [index, argument] : llvm::enumerate(callOperands.positional)) {
    Value value = argument;
    auto expected =
        mlir::dyn_cast<mlir::IntegerType>(callable.getPositionalTypes()[index]);
    if (expected)
      value = coercePrimitiveInteger(value, expected, expr);
    operands.push_back(value.value);
  }

  auto call = builder.create<mlir::func::CallOp>(
      loc(expr), target.getSymName(), callable.getResultTypes(), operands);
  return Value{call.getResult(0), callable.getResultTypes().front()};
}

std::optional<Value>
ModuleEmitter::emitPrimitiveBinary(const parser::Node &expr, Value lhs,
                                   Value rhs, const parser::Node *op) {
  mlir::Location location = loc(expr);
  if (auto lhsTensor =
          mlir::dyn_cast<mlir::RankedTensorType>(lhs.value.getType())) {
    auto rhsTensor =
        mlir::dyn_cast<mlir::RankedTensorType>(rhs.value.getType());
    if (!rhsTensor)
      return std::nullopt;
    if (ast::isOperator(op, "MatMult"))
      return emitMatrixMatmul(builder, location, expr, lhs, rhs, lhsTensor,
                              rhsTensor);
    if (lhsTensor == rhsTensor)
      return emitElementwiseTensorBinary(builder, location, expr, lhs, rhs, op,
                                         lhsTensor);
    return std::nullopt;
  }

  if (auto lhsFloat = mlir::dyn_cast<mlir::FloatType>(lhs.value.getType())) {
    auto rhsFloat = mlir::dyn_cast<mlir::FloatType>(rhs.value.getType());
    if (!rhsFloat)
      return std::nullopt;
    (void)rhsFloat;
    rhs.value = coerceFloatValue(builder, location, rhs.value, lhsFloat);
    if (std::optional<mlir::Value> result =
            createScalarBinary(builder, location, op, lhs.value, rhs.value))
      return Value{*result, lhsFloat};
    return std::nullopt;
  }

  auto lhsInt = mlir::dyn_cast<mlir::IntegerType>(lhs.value.getType());
  if (!lhsInt || !mlir::isa<mlir::IntegerType>(rhs.value.getType()))
    return std::nullopt;

  rhs = coercePrimitiveInteger(rhs, lhsInt, expr);
  if (ast::isOperator(op, "Sub")) {
    auto result =
        builder.create<mlir::arith::SubIOp>(loc(expr), lhs.value, rhs.value);
    return Value{result.getResult(), lhsInt};
  }
  if (ast::isOperator(op, "Mult")) {
    auto result =
        builder.create<mlir::arith::MulIOp>(loc(expr), lhs.value, rhs.value);
    return Value{result.getResult(), lhsInt};
  }
  if (ast::isOperator(op, "Add")) {
    auto result =
        builder.create<mlir::arith::AddIOp>(loc(expr), lhs.value, rhs.value);
    return Value{result.getResult(), lhsInt};
  }
  return std::nullopt;
}

std::optional<Value>
ModuleEmitter::emitPrimitiveCompare(const parser::Node &expr, Value lhs,
                                    Value rhs, const parser::Node *op) {
  auto lhsInt = mlir::dyn_cast<mlir::IntegerType>(lhs.value.getType());
  if (!lhsInt || !mlir::isa<mlir::IntegerType>(rhs.value.getType()))
    return std::nullopt;

  rhs = coercePrimitiveInteger(rhs, lhsInt, expr);
  mlir::arith::CmpIPredicate predicate = mlir::arith::CmpIPredicate::eq;
  if (ast::isOperator(op, "NotEq"))
    predicate = mlir::arith::CmpIPredicate::ne;
  else if (ast::isOperator(op, "Lt"))
    predicate = mlir::arith::CmpIPredicate::slt;
  else if (ast::isOperator(op, "LtE"))
    predicate = mlir::arith::CmpIPredicate::sle;
  else if (ast::isOperator(op, "Gt"))
    predicate = mlir::arith::CmpIPredicate::sgt;
  else if (ast::isOperator(op, "GtE"))
    predicate = mlir::arith::CmpIPredicate::sge;

  auto result = builder.create<mlir::arith::CmpIOp>(loc(expr), predicate,
                                                    lhs.value, rhs.value);
  return Value{result.getResult(), builder.getI1Type()};
}

Value ModuleEmitter::emitPrimitiveConstant(const parser::Node &anchor,
                                           const PrimitiveConstant &constant) {
  auto integerType = mlir::dyn_cast<mlir::IntegerType>(constant.type);
  if (!integerType) {
    diagnostics.push_back(
        parser::Diagnostic{parser::Severity::Error, anchor.range.start,
                           "primitive constant has a non-integer type"});
    return emitNone(anchor);
  }
  auto op = builder.create<mlir::arith::ConstantIntOp>(
      loc(anchor), constant.integerValue, integerType.getWidth());
  return {op.getResult(), integerType};
}

Value ModuleEmitter::coercePrimitiveInteger(Value value,
                                            mlir::IntegerType targetType,
                                            const parser::Node &anchor) {
  if (!targetType || value.value.getType() == targetType)
    return {value.value, targetType};

  auto sourceType = mlir::dyn_cast<mlir::IntegerType>(value.value.getType());
  if (!sourceType) {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, anchor.range.start,
        "primitive integer conversion requires an integer value"});
    return value;
  }

  if (sourceType.getWidth() < targetType.getWidth()) {
    auto op = builder.create<mlir::arith::ExtSIOp>(loc(anchor), targetType,
                                                   value.value);
    return {op.getResult(), targetType};
  }
  if (sourceType.getWidth() > targetType.getWidth()) {
    auto op = builder.create<mlir::arith::TruncIOp>(loc(anchor), targetType,
                                                    value.value);
    return {op.getResult(), targetType};
  }
  return {value.value, targetType};
}

} // namespace lython::emitter
