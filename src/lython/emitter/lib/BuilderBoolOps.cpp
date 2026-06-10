#include "BuilderImpl.h"

#include "mlir/Dialect/SCF/IR/SCF.h"

namespace lython::emitter {
namespace {

enum class BoolKind { And, Or };

std::optional<BoolKind> boolKind(llvm::StringRef op) {
  if (op == "and")
    return BoolKind::And;
  if (op == "or")
    return BoolKind::Or;
  return std::nullopt;
}

} // namespace

Value Builder::Impl::emitBoolOp(const parser::Node &expr) {
  std::optional<std::string> op = symbolField(expr, "op");
  const std::vector<parser::NodePtr> *values = nodeListField(expr, "values");
  if (!op || !values || values->empty()) {
    error(expr, "BoolOp.op or BoolOp.values is missing");
    return Value{{}, boolType()};
  }
  std::optional<BoolKind> kind = boolKind(*op);
  if (!kind) {
    error(expr, "unsupported boolean operator '" + *op + "'");
    return Value{{}, boolType()};
  }

  mlir::Type resultType = inferExpressionType(expr).value_or(boolType());
  if (resultType != i1Type() && resultType != boolType()) {
    error(expr, "boolean operator must infer i1 or !py.bool, got " +
                    typeString(resultType));
    return Value{{}, boolType()};
  }

  const parser::NodePtr &firstNode = values->front();
  if (!firstNode) {
    error(expr, "BoolOp contains an empty operand");
    return Value{{}, boolType()};
  }
  Value first = emitCondition(*firstNode);
  if (!first.value)
    return Value{{}, resultType};
  mlir::Value current = first.value;

  for (const parser::NodePtr &operandNode :
       llvm::ArrayRef<parser::NodePtr>(*values).drop_front()) {
    if (!operandNode) {
      error(expr, "BoolOp contains an empty operand");
      return Value{{}, resultType};
    }

    auto ifOp = builder.create<mlir::scf::IfOp>(
        loc(*operandNode), mlir::TypeRange{i1Type()}, current,
        /*withElseRegion=*/true);
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(ifOp.thenBlock());
      if (*kind == BoolKind::And) {
        Value rhs = emitCondition(*operandNode);
        if (!rhs.value)
          return Value{{}, resultType};
        builder.create<mlir::scf::YieldOp>(loc(*operandNode), rhs.value);
      } else {
        builder.create<mlir::scf::YieldOp>(loc(*operandNode), current);
      }
    }
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(ifOp.elseBlock());
      if (*kind == BoolKind::And) {
        builder.create<mlir::scf::YieldOp>(loc(*operandNode), current);
      } else {
        Value rhs = emitCondition(*operandNode);
        if (!rhs.value)
          return Value{{}, resultType};
        builder.create<mlir::scf::YieldOp>(loc(*operandNode), rhs.value);
      }
    }
    builder.setInsertionPointAfter(ifOp);
    current = ifOp.getResult(0);
  }

  if (resultType == i1Type())
    return Value{current, i1Type()};
  mlir::Value result =
      builder.create<py::CastFromPrimOp>(loc(expr), boolType(), current);
  return Value{result, boolType()};
}

} // namespace lython::emitter
