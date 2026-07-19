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
    else if (statement && skipDeclarations)
      // A skipped module-level def still EXECUTES here in CPython terms:
      // its non-constant defaults evaluate at this spot, once, into their
      // module-lifetime cells (R6).
      emitPendingDefaultCells(*statement);
  }
}

void ModuleEmitter::emitPendingDefaultCells(const parser::Node &statement) {
  auto pending = pendingDefaultCells.find(&statement);
  if (pending == pendingDefaultCells.end())
    return;
  for (const PendingDefaultCell &cell : pending->second) {
    Value value = emitExprExpected(cell.expr.get(), cell.declaredType);
    Value coerced = coerceValue(value, cell.declaredType, statement);
    py::GlobalSetOp::create(builder, loc(statement),
                            builder.getStringAttr(cell.cellName),
                            coerced.value);
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
      Value raw = emitExprExpected(rhs, annotated);
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
  } else if (statement.kind == "While") {
    emitWhile(statement);
  } else if (statement.kind == "AsyncFor") {
    emitAsyncFor(statement);
  } else if (statement.kind == "With") {
    emitWith(statement, false);
  } else if (statement.kind == "AsyncWith") {
    emitWith(statement, true);
  } else if (statement.kind == "Raise") {
    if (const parser::Node *exception = ast::node(statement, "exc")) {
      Value value = emitExpr(exception);
      mlir::Value cause;
      bool fromNone = false;
      if (const parser::Node *causeNode = ast::node(statement, "cause")) {
        Value causeValue = emitExpr(causeNode);
        auto literal =
            mlir::dyn_cast_if_present<py::LiteralType>(causeValue.type);
        if (literal && literal.getSpelling() == "None") {
          // `raise X from None` suppresses implicit __context__ display;
          // there is no cause object to carry.
          fromNone = true;
        } else if (literal || !causeValue.value) {
          diagnostics.push_back(parser::Diagnostic{
              parser::Severity::Error, causeNode->range.start,
              "raise ... from cause must be an exception instance or None"});
          return;
        } else {
          cause = causeValue.value;
        }
      }
      py::RaiseOp::create(builder, loc(statement), value.value, cause,
                          fromNone);
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
    Value value = returnValue
                      ? emitExprExpected(returnValue, currentReturnType)
                      : emitExpr(returnValue);
    if (!inlineReturnContexts.empty()) {
      InlineReturnContext &ctx = inlineReturnContexts.back();
      emitActiveCleanups(statement);
      if (ctx.carryResult) {
        Value result = ctx.resultType
                           ? coerceValue(value, ctx.resultType, statement)
                           : value;
        mlir::cf::BranchOp::create(builder, loc(statement), ctx.target,
                                   result.value);
      } else {
        mlir::cf::BranchOp::create(builder, loc(statement), ctx.target);
      }
      return;
    }
    if (currentReturnType) {
      Value result = coerceValue(value, currentReturnType, statement);
      emitActiveCleanups(statement);
      mlir::func::ReturnOp::create(builder, loc(statement), result.value);
    }
  } else if (statement.kind == "Break") {
    if (loopControlContexts.empty() ||
        !loopControlContexts.back().breakTarget) {
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, statement.range.start,
          "break outside a supported loop is not implemented yet"});
      return;
    }
    emitActiveCleanups(statement);
    const LoopControlContext &loop = loopControlContexts.back();
    mlir::cf::BranchOp::create(
        builder, loc(statement), loop.breakTarget,
        loopCarriedBranchOperands(statement, loop, loop.breakTarget));
  } else if (statement.kind == "Continue") {
    if (loopControlContexts.empty() ||
        !loopControlContexts.back().continueTarget) {
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, statement.range.start,
          "continue outside a supported loop is not implemented yet"});
      return;
    }
    emitActiveCleanups(statement);
    const LoopControlContext &loop = loopControlContexts.back();
    mlir::cf::BranchOp::create(
        builder, loc(statement), loop.continueTarget,
        loopCarriedBranchOperands(statement, loop, loop.continueTarget));
  } else if (statement.kind == "Global") {
    // `global NAME, ...`: writes to these names in the current function target
    // the module global. Only module globals we track (int-annotated) are
    // storable; others are accepted silently (no local storage change).
    if (const auto *names = ast::stringList(statement, "names"))
      for (const std::string &name : *names)
        currentGlobalDecls.insert(name);
    return;
  } else if (statement.kind == "Delete") {
    emitDelete(statement);
  } else if (statement.kind == "Nonlocal") {
    // R6 wants a refcounted shared box; the closure machinery currently
    // captures by value only, so an honest rejection beats a silent copy.
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, statement.range.start,
        "nonlocal is not implemented yet (closures capture by value; the R6 "
        "shared-box cell representation is pending)"});
  } else if (statement.kind == "Pass") {
    return;
  } else if (statement.kind == "Match") {
    emitMatch(statement);
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

// `del` (R6): only subscript deletion is representable — variables live in
// static SSA scopes (released at scope exit), and instance attributes are
// fixed storage slots, so both are rejected with an explanation instead of a
// generic unsupported-statement error.
void ModuleEmitter::emitDelete(const parser::Node &statement) {
  const auto *targets = ast::nodeList(statement, "targets");
  if (!targets)
    return;
  for (const parser::NodePtr &target : *targets) {
    if (!target)
      continue;
    if (target->kind == "Subscript") {
      if (const parser::Node *sliceNode = ast::node(*target, "slice");
          sliceNode && sliceNode->kind == "Slice") {
        diagnostics.push_back(
            parser::Diagnostic{parser::Severity::Error, target->range.start,
                               "slice deletion is not supported yet"});
        continue;
      }
      Value container = emitExpr(ast::node(*target, "value"));
      Value index = emitExpr(ast::node(*target, "slice"));
      if (std::optional<MethodBinding> method =
              lookupClassMethod(container.type, "__delitem__")) {
        emitInlineOperatorCall(*target, container, *method, {index});
        continue;
      }
      CallInferenceResult inference = types.inferMethodCallWithEvidence(
          container.type, "__delitem__", {index.type});
      if (!requireStaticEvidence(*target, inference))
        continue;
      py::DelItemOp::create(
          builder, loc(*target),
          mlir::FlatSymbolRefAttr::get(&context, "__delitem__"),
          callProtocolFor(inference), container.value, index.value);
      continue;
    }
    if (target->kind == "Name") {
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, target->range.start,
          "`del " + std::string(ast::nameSpelling(*target)) +
              "` is rejected (Lython deviation from CPython): locals are "
              "released when their scope ends, so deleting a variable is "
              "unnecessary"});
      continue;
    }
    if (target->kind == "Attribute") {
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, target->range.start,
          "`del` on an attribute is rejected (Lython deviation from CPython): "
          "instance attributes are fixed storage slots in the static object "
          "layout"});
      continue;
    }
    diagnostics.push_back(
        parser::Diagnostic{parser::Severity::Error, target->range.start,
                           "unsupported del target '" + target->kind + "'"});
  }
}

void ModuleEmitter::emitAssignTarget(const parser::Node &target, Value value) {
  if (target.kind == "Name") {
    llvm::StringRef name = ast::nameSpelling(target);
    if (isModuleGlobalWrite(name)) {
      mlir::Type type = moduleGlobals.lookup(name);
      Value coerced = coerceValue(value, type, target);
      py::GlobalSetOp::create(builder, loc(target),
                              builder.getStringAttr(name), coerced.value);
      return;
    }
    value = pinLoopCarriedTensor(name, value, target);
    values[name] = value;
    types.bindSymbol(name, value.type);
    return;
  }
  if (target.kind == "Attribute") {
    const parser::Node *objectNode = ast::node(target, "value");
    Value object = emitExpr(objectNode);
    if (auto attr = ast::string(target, "attr")) {
      auto op = py::AttrSetOp::create(builder, loc(target), object.value, *attr,
                                      value.value);
      if (lookupClassField(object.type, *attr))
        op->setAttr("ly.attr.kind", builder.getStringAttr("field"));
      if (auto contract =
              mlir::dyn_cast_if_present<py::ContractType>(object.type))
        op->setAttr("ly.attr.owner",
                    builder.getStringAttr(contract.getContractName()));
      // Manifest-declared field assignments may refine the receiver's
      // contract parameters (ly.typing.field_param_bindings -- e.g. ctypes'
      // `fn.restype = c_int` binds CFuncPtr's T so `__call__` types as int):
      // rebind the local to the refined type. The attr.set op above stays --
      // lowering reads the same assignment as evidence.
      if (objectNode && objectNode->kind == "Name") {
        if (std::optional<mlir::Type> refined =
                types.fieldAssignmentRefinement(object.type, *attr,
                                                value.type)) {
          llvm::StringRef name = ast::nameSpelling(*objectNode);
          auto bound = values.find(name);
          if (bound != values.end() && bound->second.value == object.value) {
            bound->second.type = *refined;
            types.bindSymbol(name, *refined);
          }
        }
      }
    }
    return;
  }
  if (target.kind == "Subscript") {
    const parser::Node *containerNode = ast::node(target, "value");
    Value container = emitExpr(containerNode);
    if (container.value &&
        mlir::isa<mlir::RankedTensorType>(container.value.getType())) {
      // Shaped primitives are values, so the element write produces a new
      // one. Only a named local can observe it; anything else would drop the
      // result on the floor.
      if (!containerNode || containerNode->kind != "Name") {
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, target.range.start,
            "shaped primitive element assignment requires a named local "
            "target"});
        return;
      }
      llvm::StringRef containerName = ast::nameSpelling(*containerNode);
      auto bound = values.find(containerName);
      if (bound == values.end() || bound->second.value != container.value) {
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, target.range.start,
            "shaped primitive element assignment requires a named local "
            "target"});
        return;
      }
      if (std::optional<Value> updated = emitPrimitiveTensorSetItem(
              target, container, ast::node(target, "slice"), value)) {
        if (updated->value)
          values[containerName] =
              pinLoopCarriedTensor(containerName, *updated, target);
        return;
      }
      return;
    }
    if (const parser::Node *sliceNode = ast::node(target, "slice");
        sliceNode && sliceNode->kind == "Slice") {
      diagnostics.push_back(
          parser::Diagnostic{parser::Severity::Error, target.range.start,
                             "slice assignment is not supported yet"});
      return;
    }
    Value index = emitExpr(ast::node(target, "slice"));
    if (std::optional<MethodBinding> method =
            lookupClassMethod(container.type, "__setitem__")) {
      emitInlineOperatorCall(target, container, *method, {index, value});
      return;
    }
    CallInferenceResult inference = types.inferMethodCallWithEvidence(
        container.type, "__setitem__", {index.type, value.type});
    if (!requireStaticEvidence(target, inference))
      return;
    // Manifest-declared structural mutators may reallocate the container's
    // storage: the op carries an extra container-typed result that rebinds
    // the local (same channel as mutating bound-method calls).
    if (containerNode && containerNode->kind == "Name" &&
        types.isStructuralMutatorMethod(container.type, "__setitem__")) {
      llvm::StringRef containerName = ast::nameSpelling(*containerNode);
      auto bound = values.find(containerName);
      if (bound != values.end() && bound->second.value == container.value) {
        auto op = py::SetItemOp::create(
            builder, loc(target),
            mlir::TypeRange{container.value.getType()},
            mlir::FlatSymbolRefAttr::get(&context, "__setitem__"),
            callProtocolFor(inference), container.value, index.value,
            value.value);
        op->setAttr("ly.structural_mutation", builder.getUnitAttr());
        values[containerName] = Value{op.getResult(0), container.type};
        return;
      }
    }
    py::SetItemOp::create(builder, loc(target), mlir::TypeRange{},
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
        if (!requireStaticEvidence(*elt, inference))
          return;
        mlir::Type itemType = inference.resultType;
        auto getItem = py::GetItemOp::create(
            builder, loc(*elt), itemType,
            mlir::FlatSymbolRefAttr::get(&context, "__getitem__"),
            callProtocolFor(inference), value.value, indexValue.value);
        Value item{getItem.getResult(), itemType};
        emitAssignTarget(*elt, item);
      }
    }
  }
}

} // namespace lython::emitter
