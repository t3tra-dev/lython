#include "EmitterCore.h"
#include "EmitterPyOps.h"
#include "EmitterSupport.h"

#include "AstAccess.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/STLExtras.h"

#include <optional>
#include <string>

namespace lython::emitter {

std::optional<Value>
ModuleEmitter::emitPrimitiveConstructorCall(const parser::Node &expr,
                                            const parser::Node *calleeNode) {
  std::optional<mlir::IntegerType> primitiveInt =
      primitiveIntTypeFromSubscript(calleeNode, types);
  if (!primitiveInt)
    return std::nullopt;

  const auto *args = ast::nodeList(expr, "args");
  const auto *keywords = ast::nodeList(expr, "keywords");
  if (!args || args->size() != 1 || (keywords && !keywords->empty())) {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, expr.range.start,
        "lyrt.prim.Int constructor expects exactly one positional argument"});
    return emitNone(expr);
  }

  if (std::optional<std::int64_t> literal =
          integerLiteralValue(args->front().get()))
    return emitPrimitiveConstant(expr,
                                 PrimitiveConstant{*primitiveInt, *literal});
  return coercePrimitiveInteger(emitExpr(args->front().get()), *primitiveInt,
                                expr);
}

std::optional<Value>
ModuleEmitter::emitPrimitiveRuntimeCall(const parser::Node &expr,
                                        const parser::Node *calleeNode) {
  if (!calleeNode || calleeNode->kind != "Name")
    return std::nullopt;

  llvm::StringRef name = ast::nameSpelling(*calleeNode);
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
  if (!mlir::isa<mlir::IntegerType>(input.value.getType())) {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, expr.range.start,
        "lyrt.from_prim currently expects an integer primitive"});
    return input;
  }
  auto op = builder.create<py::CastFromPrimOp>(loc(expr), types.intType(),
                                               input.value);
  return Value{op.getResult(), types.intType()};
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
