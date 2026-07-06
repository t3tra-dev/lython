#include "EmitterCore.h"
#include "EmitterOps.h" // IWYU pragma: keep
#include "EmitterPyOps.h"
#include "EmitterSupport.h"

#include "AstAccess.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/STLExtras.h"

#include <string>

namespace lython::emitter {

void ModuleEmitter::emitStatements(
    const std::vector<parser::NodePtr> *statements, bool skipDeclarations) {
  if (!statements)
    return;
  for (const parser::NodePtr &statement : *statements) {
    if (insertionBlockTerminated(builder))
      break;
    if (statement && (!skipDeclarations || !isTopLevelDecl(*statement)))
      emitStatement(*statement);
  }
}

void ModuleEmitter::emitStatement(const parser::Node &statement) {
  if (statement.kind == "Expr") {
    emitExpr(ast::node(statement, "value"));
  } else if (statement.kind == "Import") {
    bindImportStatement(statement, /*diagnoseUnsupported=*/true);
  } else if (statement.kind == "ImportFrom") {
    bindImportStatement(statement, /*diagnoseUnsupported=*/true);
  } else if (statement.kind == "Assign") {
    const parser::Node *rhs = ast::node(statement, "value");
    Value value{{}, {}};
    bool emittedWithContext = false;
    if (rhs && rhs->kind == "Lambda") {
      if (const auto *targets = ast::nodeList(statement, "targets")) {
        if (targets->size() == 1 && targets->front() &&
            targets->front()->kind == "Name") {
          llvm::StringRef name = ast::nameSpelling(*targets->front());
          if (auto expectedType = types.lookupSymbol(name)) {
            if (auto expectedCallable =
                    mlir::dyn_cast_if_present<py::CallableType>(
                        *expectedType)) {
              value = emitLambda(*rhs, expectedCallable);
              emittedWithContext = true;
            }
          }
        }
      }
    }
    if (!emittedWithContext)
      value = emitExpr(rhs);
    if (const auto *targets = ast::nodeList(statement, "targets"))
      for (const parser::NodePtr &target : *targets)
        emitAssignTarget(*target, value);
  } else if (statement.kind == "AnnAssign") {
    mlir::Type annotated =
        types.annotationType(ast::node(statement, "annotation"));
    if (const parser::Node *rhs = ast::node(statement, "value")) {
      Value raw =
          rhs->kind == "Lambda"
              ? emitLambda(*rhs, mlir::dyn_cast_if_present<py::CallableType>(
                                     annotated))
              : emitExpr(rhs);
      Value value = coerceValue(raw, annotated, statement);
      emitAssignTarget(*ast::node(statement, "target"), value);
      return;
    }
    const parser::Node *target = ast::node(statement, "target");
    if (target && target->kind == "Name")
      types.bindSymbol(ast::nameSpelling(*target), annotated);
  } else if (statement.kind == "AugAssign") {
    Value lhs = emitExpr(ast::node(statement, "target"));
    Value rhs = emitExpr(ast::node(statement, "value"));
    Value value = emitBinarySpecial<py::AddOp>(
        statement, "__add__", lhs, rhs,
        types.widenLiteral(types.join({lhs.type, rhs.type})));
    emitAssignTarget(*ast::node(statement, "target"), value);
  } else if (statement.kind == "If") {
    emitIf(statement);
  } else if (statement.kind == "For") {
    emitFor(statement);
  } else if (statement.kind == "AsyncFor") {
    emitAsyncFor(statement);
  } else if (statement.kind == "With") {
    emitWith(statement, false);
  } else if (statement.kind == "AsyncWith") {
    emitWith(statement, true);
  } else if (statement.kind == "Raise") {
    if (const parser::Node *exception = ast::node(statement, "exc")) {
      Value value = emitExpr(exception);
      py::RaiseOp::create(builder, loc(statement), value.value);
    } else {
      py::RaiseCurrentOp::create(builder, loc(statement));
    }
  } else if (statement.kind == "FunctionDef" ||
             statement.kind == "AsyncFunctionDef") {
    Value function = emitNestedFunctionDecl(statement);
    if (auto name = ast::string(statement, "name")) {
      values[*name] = function;
      types.bindSymbol(*name, function.type);
    }
  } else if (statement.kind == "Return") {
    const parser::Node *returnValue = ast::node(statement, "value");
    Value value = returnValue && returnValue->kind == "Lambda"
                      ? emitLambda(*returnValue,
                                   mlir::dyn_cast_if_present<py::CallableType>(
                                       currentReturnType))
                      : emitExpr(returnValue);
    if (!inlineReturnContexts.empty()) {
      InlineReturnContext &ctx = inlineReturnContexts.back();
      Value result = ctx.resultType
                         ? coerceValue(value, ctx.resultType, statement)
                         : value;
      emitActiveCleanups(statement);
      mlir::cf::BranchOp::create(builder, loc(statement), ctx.target,
                                 result.value);
      return;
    }
    if (currentReturnType) {
      Value result = coerceValue(value, currentReturnType, statement);
      emitActiveCleanups(statement);
      mlir::func::ReturnOp::create(builder, loc(statement), result.value);
    }
  } else if (statement.kind == "Pass") {
    return;
  } else if (statement.kind == "Try") {
    emitTry(statement);
  } else if (statement.kind == "TryStar") {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, statement.range.start,
        "except* requires exception-group-aware py.try lowering"});
  } else {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, statement.range.start,
        "unsupported statement kind '" + statement.kind + "'"});
  }
}

void ModuleEmitter::emitAssignTarget(const parser::Node &target, Value value) {
  if (target.kind == "Name") {
    llvm::StringRef name = ast::nameSpelling(target);
    values[name] = value;
    types.bindSymbol(name, value.type);
    return;
  }
  if (target.kind == "Attribute") {
    Value object = emitExpr(ast::node(target, "value"));
    if (auto attr = ast::string(target, "attr")) {
      auto op = py::AttrSetOp::create(builder, loc(target), object.value, *attr,
                                      value.value);
      if (lookupClassField(object.type, *attr))
        op->setAttr("ly.attr.kind", builder.getStringAttr("field"));
      if (auto contract =
              mlir::dyn_cast_if_present<py::ContractType>(object.type))
        op->setAttr("ly.attr.owner",
                    builder.getStringAttr(contract.getContractName()));
    }
    return;
  }
  if (target.kind == "Subscript") {
    Value container = emitExpr(ast::node(target, "value"));
    Value index = emitExpr(ast::node(target, "slice"));
    CallInferenceResult inference = types.inferMethodCallWithEvidence(
        container.type, "__setitem__", {index.type, value.type});
    py::SetItemOp::create(builder, loc(target),
                          mlir::FlatSymbolRefAttr::get(&context, "__setitem__"),
                          callProtocolFor(inference), container.value,
                          index.value, value.value);
    return;
  }
  if (target.kind == "Tuple" || target.kind == "List") {
    if (const auto *elts = ast::nodeList(target, "elts")) {
      for (auto [index, elt] : llvm::enumerate(*elts)) {
        Value indexValue{py::IntConstantOp::create(
                             builder, loc(*elt),
                             types.literal(std::to_string(index)),
                             builder.getStringAttr(std::to_string(index)))
                             .getResult(),
                         types.literal(std::to_string(index))};
        CallInferenceResult inference = types.inferMethodCallWithEvidence(
            value.type, "__getitem__", {indexValue.type});
        auto getItem = py::GetItemOp::create(
            builder, loc(*elt), types.object(),
            mlir::FlatSymbolRefAttr::get(&context, "__getitem__"),
            callProtocolFor(inference), value.value, indexValue.value);
        Value item{getItem.getResult(), types.object()};
        emitAssignTarget(*elt, item);
      }
    }
  }
}

} // namespace lython::emitter
