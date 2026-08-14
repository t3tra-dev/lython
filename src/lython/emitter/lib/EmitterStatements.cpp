#include "AstSynth.h"
#include "EmitterCore.h"

#include "llvm/ADT/ScopeExit.h"
#include "EmitterOps.h" // IWYU pragma: keep
#include "EmitterPyOps.h"
#include "EmitterSupport.h"

#include "AstAccess.h"
#include "Contracts.h"

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
    else if (statement && skipDeclarations) {
      // The class contract was declared up front, but its attribute
      // initializers evaluate here -- at the class statement's position in
      // module flow, like CPython's class-body execution.
      if (statement->kind == "ClassDef")
        emitClassAttrInitializers(*statement);
      // A skipped module-level def still EXECUTES here in CPython terms:
      // its non-constant defaults evaluate at this spot, once, into their
      // module-lifetime cells (R6). Not ClassDef-exclusive: method defaults
      // registered under a class statement flow through the same cells.
      emitPendingDefaultCells(*statement);
    }
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
    // ⭐ A store into a declared field knows what type it wants, so the value
    // is emitted against it.
    //
    //     class Acc:
    //         xs: list[int]
    //         def __init__(self) -> None:
    //             self.xs = []
    //
    //     attribute value 'list[object]' is not assignable to field
    //     'list[int]'
    //
    // An empty literal has nothing to infer an element type from and comes out
    // as `list[object]`. Writing the annotation inline (`self.xs: list[int] =
    // []`) already works, because AnnAssign passes its annotation down as the
    // expected type -- this is the same expectation, read from the field the
    // target names instead of from an annotation beside it. Same for `{}` into
    // a declared `dict[K, V]`.
    if (!emittedWithContext && rhs) {
      if (const auto *targets = ast::nodeList(statement, "targets"))
        if (targets->size() == 1 && targets->front() &&
            targets->front()->kind == "Attribute")
          if (const parser::Node *object =
                  ast::node(*targets->front(), "value"))
            if (auto attr = ast::string(*targets->front(), "attr")) {
              mlir::Type objectType = types.inferExpr(object);
              if (std::optional<mlir::Type> fieldType =
                      lookupClassField(objectType, *attr)) {
                value = emitExprExpected(rhs, *fieldType);
                emittedWithContext = true;
              }
            }
    }
    // ⭐ An EMPTY container literal into a name that already has a declared
    // type takes that type, the same way the field store above does. It has
    // nothing of its own to infer an element type from, so it came out as
    // `list[object]`, and in the canonical Optional idiom
    //
    //     def f(xs: list[int] | None = None) -> int:
    //         if xs is None:
    //             xs = []
    //
    // the branch join was `list[int] | list[object] | None`, which nothing
    // accepts.
    //
    // ⛔ Empty only: a literal WITH elements is what the rebinding says it
    // is. `xs = [1, 2]` where xs was `list[str]` must become `list[int]`, not
    // be pushed at the old element type.
    if (!emittedWithContext && rhs) {
      bool emptyLiteral =
          (rhs->kind == "List" || rhs->kind == "Tuple" || rhs->kind == "Set") &&
          [&] {
            const auto *elts = ast::nodeList(*rhs, "elts");
            return !elts || elts->empty();
          }();
      if (!emptyLiteral && rhs->kind == "Dict") {
        const auto *keys = ast::nodeList(*rhs, "keys");
        emptyLiteral = !keys || keys->empty();
      }
      if (emptyLiteral)
        if (const auto *targets = ast::nodeList(statement, "targets"))
          if (targets->size() == 1 && targets->front() &&
              targets->front()->kind == "Name")
          {
            llvm::StringRef target = ast::nameSpelling(*targets->front());
            // The pre-narrowing type wins: inside `if xs is None:` the flow
            // type of xs IS None, which is the right answer for a read and no
            // constraint at all on a write.
            mlir::Type declared = narrowedFromTypes.lookup(target);
            if (!declared)
              if (auto flow = types.lookupSymbol(target))
                declared = *flow;
            if (declared) {
              value = emitExprExpected(rhs, declared);
              emittedWithContext = true;
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
    // Desugar to the equivalent BinOp over shared subtrees so the operator
    // dispatch (and the primitive path) is exactly the `x = x <op> v` one.
    // The previous inline emission hardcoded __add__ for every operator.
    auto sharedSubtree = [&](llvm::StringRef name) -> parser::NodePtr {
      const parser::Field *field = parser::findField(statement, name);
      if (!field || !std::holds_alternative<parser::NodePtr>(field->value))
        return nullptr;
      return std::get<parser::NodePtr>(field->value);
    };
    parser::NodePtr target = sharedSubtree("target");
    parser::NodePtr op = sharedSubtree("op");
    parser::NodePtr rhs = sharedSubtree("value");
    // ⭐ The target subtree is used TWICE below -- once as the left operand of
    // the rewritten binary op and once as the store target -- so anything in
    // it with a side effect ran twice, and the store landed wherever the
    // SECOND evaluation pointed:
    //
    //     a: list[int] = [0, 0, 0]
    //     a[f()] += 1        # f() called twice; stored [0, 0, 1]
    //                        # CPython calls it once and stores [0, 1, 0]
    //
    // The receiver, the index and a slice's bounds are each emitted once here
    // and referenced from both places through a `LyValueRef`. A Name target
    // needs none of this -- re-reading a name has no effect -- which is why
    // the defect only ever showed through a subscript or an attribute.
    std::size_t valueRefStart = pendingValueRefs.size();
    auto shareSubexpression = [&](parser::NodePtr &slotNode) {
      if (!slotNode)
        return;
      Value evaluated = emitExpr(slotNode.get());
      parser::NodePtr ref = parser::makeNode("LyValueRef", statement.range);
      parser::addField(*ref, "slot",
                       static_cast<std::int64_t>(pendingValueRefs.size()));
      pendingValueRefs.push_back(evaluated);
      slotNode = std::move(ref);
    };
    auto shareField = [&](parser::Node &node, llvm::StringRef name) {
      parser::Field *field = parser::findField(node, name);
      if (!field || !std::holds_alternative<parser::NodePtr>(field->value))
        return;
      parser::NodePtr child = std::get<parser::NodePtr>(field->value);
      shareSubexpression(child);
      field->value = std::move(child);
    };
    if (target && (target->kind == "Subscript" || target->kind == "Attribute")) {
      parser::NodePtr shared = parser::makeNode(target->kind, target->range);
      for (const parser::Field &field : target->fields)
        parser::addField(*shared, field.name, field.value);
      const parser::Node *targetSlice =
          shared->kind == "Subscript" ? ast::node(*shared, "slice") : nullptr;
      bool sliceTarget = targetSlice && targetSlice->kind == "Slice";
      // ⛔ A SLICE target keeps its receiver as written. The slice-assignment
      // path requires a named local -- it rebinds the local to the resized
      // list -- and handing it a `LyValueRef` refuses the program outright.
      // A slice receiver is re-read, which is a second evaluation this does
      // not remove; the BOUNDS, which are where an index expression usually
      // sits, are shared.
      if (!sliceTarget)
        shareField(*shared, "value");
      if (shared->kind == "Subscript") {
        const parser::Node *slice = ast::node(*shared, "slice");
        if (slice && slice->kind == "Slice") {
          parser::NodePtr sliceCopy =
              parser::makeNode("Slice", slice->range);
          for (const parser::Field &field : slice->fields)
            parser::addField(*sliceCopy, field.name, field.value);
          shareField(*sliceCopy, "lower");
          shareField(*sliceCopy, "upper");
          shareField(*sliceCopy, "step");
          parser::Field *sliceField = parser::findField(*shared, "slice");
          if (sliceField)
            sliceField->value = std::move(sliceCopy);
        } else {
          shareField(*shared, "slice");
        }
      }
      target = std::move(shared);
    }
    auto releaseValueRefs = llvm::make_scope_exit(
        [&] { pendingValueRefs.resize(valueRefStart); });
    if (!target || !op || !rhs) {
      diagnostics.push_back(parser::Diagnostic{parser::Severity::Error,
                                               statement.range.start,
                                               "malformed augmented assignment"});
      return;
    }
    // ⭐ An augmented assignment whose operator has an in-place dunder must
    // MUTATE the object, not rebind the name: every other alias observes it.
    //
    //     a: list[int] = [1, 2]
    //     b: list[int] = a
    //     b += [3]
    //     print(a)      # printed [1, 2]; CPython prints [1, 2, 3]
    //
    // Desugaring to `b = b + [3]` builds a fresh list, so the mutation was
    // invisible through `a` -- and through the caller when the target was a
    // parameter. CPython's `list.__iadd__` is `extend`, and `dict.__ior__` is
    // `update`; the dict case was already rewritten this way, so this is the
    // same rule stated once for both rather than a second special case.
    struct InPlaceRewrite {
      llvm::StringRef opKind;
      llvm::StringRef contract;
      llvm::StringRef method;
    };
    static constexpr InPlaceRewrite kInPlaceRewrites[] = {
        {"BitOr", "builtins.dict", "update"},
        {"Add", "builtins.list", "extend"},
        // set's four, which were missing and silently rebound a fresh set:
        // `a |= {9}` left every alias of `a` holding the old one.
        {"BitOr", "builtins.set", "update"},
        {"Sub", "builtins.set", "difference_update"},
        {"BitAnd", "builtins.set", "intersection_update"},
        {"BitXor", "builtins.set", "symmetric_difference_update"},
    };
    llvm::StringRef inPlaceMethod;
    for (const InPlaceRewrite &rewrite : kInPlaceRewrites)
      if (op->kind == rewrite.opKind &&
          exprHasContract(target.get(), rewrite.contract))
        inPlaceMethod = rewrite.method;
    if (!inPlaceMethod.empty()) {
      parser::NodePtr updateAttr = synth::attribute(target, std::string(inPlaceMethod), statement.range);
      parser::NodePtr updateCall = synth::call(std::move(updateAttr), std::vector<parser::NodePtr>{rhs}, statement.range);
      parser::NodePtr updateStatement = synth::exprStmt(std::move(updateCall), statement.range);
      emitStatement(*updateStatement);
      return;
    }
    // ⭐ A user class that defines the IN-PLACE dunder gets it. CPython tries
    // `__iadd__` before `__add__` and REBINDS the result, so a class defining
    // both silently ran the wrong one:
    //
    //     class M:
    //         def __add__(self, o): return M(1)
    //         def __iadd__(self, o): return M(2)
    //     x = M(0); x += M(0)
    //     print(x.v)      # printed 1; CPython prints 2
    //
    // The table above is the same rule for the manifest containers, whose
    // in-place dunder is spelled as a named method; this is the source-class
    // half, and both end in "call it, then bind the target".
    static constexpr struct {
        llvm::StringRef opKind;
        llvm::StringRef method;
    } kInPlaceDunders[] = {
        {"Add", "__iadd__"},       {"Sub", "__isub__"},
        {"Mult", "__imul__"},      {"Div", "__itruediv__"},
        {"FloorDiv", "__ifloordiv__"}, {"Mod", "__imod__"},
        {"Pow", "__ipow__"},       {"LShift", "__ilshift__"},
        {"RShift", "__irshift__"}, {"BitAnd", "__iand__"},
        {"BitOr", "__ior__"},      {"BitXor", "__ixor__"},
        {"MatMult", "__imatmul__"},
    };
    for (const auto &entry : kInPlaceDunders) {
      if (op->kind != entry.opKind)
        continue;
      mlir::Type targetType = types.inferExpr(target.get());
      std::optional<MethodBinding> inPlace =
          lookupClassMethod(targetType, entry.method);
      if (!inPlace || !inPlace->method)
        break;
      parser::NodePtr attribute = synth::attribute(target, std::string(entry.method), statement.range);
      parser::NodePtr call = synth::call(std::move(attribute), std::vector<parser::NodePtr>{rhs}, statement.range);
      Value updated = emitExpr(call.get());
      emitAssignTarget(*target, updated);
      return;
    }
    parser::NodePtr binop = parser::makeNode("BinOp", statement.range);
    parser::addField(*binop, "left", target);
    parser::addField(*binop, "op", op);
    parser::addField(*binop, "right", rhs);
    Value value = emitBinary(*binop);
    emitAssignTarget(*target, value);
  } else if (statement.kind == "If") {
    emitIf(statement);
  } else if (statement.kind == "For") {
    emitFor(statement);
  } else if (statement.kind == "While") {
    emitWhile(statement);
  } else if (statement.kind == "AsyncFor") {
    emitAsyncFor(statement);
  } else if (statement.kind == "LyWithEnter") {
    // Synthesized by emitWith: evaluate one manager and call its __enter__,
    // inside the try that guards the managers entered before it.
    const parser::Field *slot = parser::findField(statement, "slot");
    if (slot && std::holds_alternative<std::int64_t>(slot->value)) {
      auto index = static_cast<std::size_t>(std::get<std::int64_t>(slot->value));
      if (index < pendingWithItems.size() && pendingWithItems[index].item)
        emitWithEnter(*pendingWithItems[index].item,
                      pendingWithItems[index].async);
    }
  } else if (statement.kind == "LyWithCleanup") {
    // Synthesized by emitWith: the `finally` body of the implicit try that
    // wraps a `with` block. It carries an index into `activeWithCleanups`
    // rather than an expression, because the manager is an already-emitted
    // SSA value and there is no spelling for one in the AST.
    const parser::Field *slot = parser::findField(statement, "slot");
    if (slot && std::holds_alternative<std::int64_t>(slot->value)) {
      auto index = static_cast<std::size_t>(std::get<std::int64_t>(slot->value));
      if (index < activeWithCleanups.size())
        emitWithCleanup(statement, activeWithCleanups[index]);
    }
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
    } else if (exceptHandlerDepth == 0) {
      // ⭐ A bare `raise` outside every handler has nothing to re-raise.
      //
      // CPython raises `RuntimeError: No active exception to reraise`; this
      // compiler emitted `py.raise.current`, whose runtime arm for an empty
      // slot is a trap -- `raise` on its own aborted with no output. Deciding
      // it HERE and not in the lowering is forced: `py.try`'s regions are
      // flattened before the lowering sees the op, so by then the parent chain
      // is just `func.func`, and a handler re-raise looks identical to this.
      //
      // Whether a handler is running is a question about where the statement
      // sits, which is exactly what this walk knows. A handler that already
      // completed does not count, matching CPython: `try/except/pass` then
      // `raise` raises there too.
      parser::NodePtr message = synth::strConstant(std::string("No active exception to reraise"), statement.range);
      parser::NodePtr callee = synth::name(std::string("RuntimeError"), statement.range);
      parser::NodePtr call = synth::call(std::move(callee), std::vector<parser::NodePtr>{std::move(message)}, statement.range);
      parser::NodePtr synthesized = synth::raiseStmt(std::move(call), statement.range);
      emitStatement(*synthesized);
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
    // The enclosing function already promoted these locals to shared cells
    // (nonlocalBoxedNames) and this function captured the cell instances;
    // reads/writes dispatch on the cell binding, so the declaration itself
    // only validates that each target resolved to a cell.
    if (const auto *names = ast::stringList(statement, "names"))
      for (const std::string &name : *names) {
        auto found = values.find(name);
        if (found == values.end() ||
            !isCellContract(found->second.type)) {
          diagnostics.push_back(parser::Diagnostic{
              parser::Severity::Error, statement.range.start,
              "no binding for nonlocal '" + name +
                  "' found (the target must be assigned in an enclosing "
                  "function before this definition)"});
        }
      }
    return;
  } else if (statement.kind == "Pass") {
    return;
  } else if (statement.kind == "Match") {
    emitMatch(statement);
  } else if (statement.kind == "Try") {
    emitTry(statement);
  } else if (statement.kind == "TryStar") {
    emitTryStar(statement);
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
        emitSliceMutation(*target, ast::node(*target, "value"), *sliceNode,
                          "__delslice__", std::nullopt);
        continue;
      }
      Value container = emitExpr(ast::node(*target, "value"));
      Value index = emitExpr(ast::node(*target, "slice"));
      if (tryEmitClassDunder(*target, container, "__delitem__", {index}))
        continue;
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

// Slice assignment / deletion on a list local: both are structural
// mutations (the splice may reallocate the items storage), so they lower
// through the same rebinding bound-method call shape as list.append —
// `__setslice__`/`__delslice__` carry the (start, stop, step, mask) pack the
// slice READ path already uses, plus the replacement list for assignment.
void ModuleEmitter::emitSliceMutation(const parser::Node &target,
                                      const parser::Node *containerNode,
                                      const parser::Node &sliceNode,
                                      llvm::StringRef methodName,
                                      std::optional<Value> payload) {
  bool isAssignment = payload.has_value();
  auto unsupported = [&](llvm::StringRef reason) {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, target.range.start,
        (isAssignment ? llvm::Twine("slice assignment ")
                      : llvm::Twine("slice deletion "))
                .concat(reason)
                .str()});
  };
  if (!containerNode || containerNode->kind != "Name") {
    unsupported("requires a named local list target (field containers are "
                "not supported yet)");
    return;
  }
  Value container = emitExpr(containerNode);
  llvm::StringRef containerName = ast::nameSpelling(*containerNode);
  auto bound = values.find(containerName);
  if (bound == values.end() || bound->second.value != container.value) {
    unsupported("requires a rebindable local list target");
    return;
  }

  const parser::Node *lower = ast::node(sliceNode, "lower");
  const parser::Node *upper = ast::node(sliceNode, "upper");
  const parser::Node *step = ast::node(sliceNode, "step");
  auto intConstant = [&](long long value) -> Value {
    std::string text = std::to_string(value);
    mlir::Type type = types.literal(text);
    auto op = py::IntConstantOp::create(builder, loc(target), type,
                                        builder.getStringAttr(text));
    return {op.getResult(), type};
  };
  Value startValue = lower ? emitExpr(lower) : intConstant(0);
  Value stopValue = upper ? emitExpr(upper) : intConstant(0);
  Value stepValue = step ? emitExpr(step) : intConstant(1);
  long long maskBits = (lower ? 1 : 0) | (upper ? 2 : 0);
  Value maskValue = intConstant(maskBits);

  llvm::SmallVector<mlir::Type, 5> argumentTypes{
      startValue.type, stopValue.type, stepValue.type, maskValue.type};
  llvm::SmallVector<Value, 5> arguments{startValue, stopValue, stepValue,
                                        maskValue};
  if (payload) {
    argumentTypes.push_back(payload->type);
    arguments.push_back(*payload);
  }
  CallInferenceResult inference = types.inferMethodCallWithEvidence(
      container.type, methodName, argumentTypes);
  if (!inference) {
    unsupported(isAssignment
                    ? "is only supported for a list target with a list value"
                    : "is only supported for list targets");
    return;
  }
  if (!requireStaticEvidence(target, inference))
    return;
  if (!types.isStructuralMutatorMethod(container.type, methodName)) {
    unsupported("target's manifest does not declare the slice mutator");
    return;
  }
  Value posPack = emitPack(arguments);
  Value namePack = emitPack({});
  Value valuePack = emitPack({});
  auto op = py::CallOp::create(
      builder, loc(target),
      mlir::TypeRange{inference.resultType, container.value.getType()},
      callProtocolFor(inference), container.value, posPack.value,
      namePack.value, valuePack.value);
  op->setAttr("ly.bound_method", builder.getStringAttr(methodName));
  op->setAttr("ly.structural_mutation", builder.getUnitAttr());
  values[containerName] = Value{op.getResult(1), container.type};
}

void ModuleEmitter::emitAssignTarget(const parser::Node &target, Value value) {
  if (target.kind == "Name") {
    llvm::StringRef name = ast::nameSpelling(target);
    // ⭐ `global X` where X is not a global this walk can WRITE. Only
    // storage-backed (annotated int) module globals get a cell, so a
    // container global fell through to the local binding below and the write
    // was a silent no-op:
    //
    //     X: list[int] = [1]
    //     def f() -> None:
    //         global X
    //         X = [2]
    //     f(); print(X)      # printed [1]; CPython prints [2]
    //
    // Refused rather than made to work: the cell a container global would
    // need does not exist, and the declaration is an explicit statement that
    // this assignment is not a local one -- so binding a local is the one
    // answer it cannot have.
    if (!atModuleScope && currentGlobalDecls.count(name) &&
        !moduleGlobals.count(name)) {
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, target.range.start,
          "'global " + name.str() +
              "' names a module global this compiler does not give storage "
              "to, so the assignment cannot reach it; an annotated module "
              "global of a scalar, bytes, ctypes-pointer or user-class type "
              "is writable from a function, a container is not"});
      return;
    }
    if (isModuleGlobalWrite(name)) {
      mlir::Type type = moduleGlobals.lookup(name);
      // ⭐ A module global's cell has ONE runtime representation, fixed by the
      // declaration, and `coerceValue` no longer retypes between the numeric
      // contracts because that retyping was a lie. So the mismatch has to be
      // reported here, at the write.
      //
      // ⛔ Why NOT convert the value to the cell's type: `x: float = 3` would
      // then print 3.0 where CPython prints 3 -- the annotation does not
      // convert there either (tests/probe/wb_argument_boundary_numeric_tower.py
      // measures the same rejection at a parameter boundary).
      //
      // ⛔ And why NOT let the store through, which is what the retyping used
      // to do: the read still comes back at the cell's declared type, so the
      // int's lanes were reinterpreted as a double and `x: float = 3`
      // printed 5e-324. It reported "assignment value group has 3 values,
      // expected 1" before that, which named the count and not the cause.
      if (mlir::Type widened = types.widenLiteral(value.type);
          widened != type && isNumericPrimitiveContract(widened) &&
          isNumericPrimitiveContract(type)) {
        auto spell = [&](mlir::Type numeric) -> llvm::StringRef {
          if (numeric == types.boolType())
            return "bool";
          return numeric == types.intType() ? "int" : "float";
        };
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, target.range.start,
            "module global '" + name.str() + "' holds " +
                spell(type).str() + " and this assignment gives it " +
                spell(widened).str() +
                "; a module global has one runtime representation and these "
                "two do not share one, so write the value in the declared "
                "type"});
        return;
      }
      Value coerced = coerceValue(value, type, target);
      auto op = py::GlobalSetOp::create(builder, loc(target),
                                        builder.getStringAttr(name),
                                        coerced.value);
      markBoxedModuleGlobal(op);
      return;
    }
    // A name bound to a nonlocal cell (owner scope or a `nonlocal`-declaring
    // closure) writes THROUGH the cell: the binding itself never changes.
    auto bound = values.find(name);
    if (bound != values.end() && isCellContract(bound->second.type)) {
      emitCellStore(target, bound->second, value);
      return;
    }
    if (bound == values.end() && currentBoxedLocals.contains(name)) {
      // First binding of a boxed local: its storage is a fresh shared cell.
      Value cell = emitCellAlloc(target, value);
      values[name] = cell;
      types.bindSymbol(name, cellContentType(cell.type));
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
      // Property writes inline the setter; a getter without a setter is the
      // CPython AttributeError, surfaced statically.
      if (std::optional<MethodBinding> setter = lookupClassMethod(
              object.type, (llvm::Twine(*attr) + ".setter").str())) {
        if (setter->kind == "property_setter") {
          emitInlineMethodBody(target, object, /*bindDescriptorReceiver=*/true,
                               *setter, {value}, {});
          return;
        }
      }
      if (std::optional<MethodBinding> getter =
              lookupClassMethod(object.type, *attr);
          getter && getter->kind == "property") {
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, target.range.start,
            "property '" + std::string(*attr) + "' has no setter"});
        return;
      }
      // Mutable class attribute writes target the defining class's cell.
      if (auto typeObject =
              mlir::dyn_cast_if_present<py::TypeType>(object.type)) {
        if (auto contract = mlir::dyn_cast_if_present<py::ContractType>(
                typeObject.getInstanceType())) {
          llvm::StringRef className = contract.getContractName();
          if (std::optional<std::pair<llvm::StringRef, mlir::Type>> slot =
                  resolveClassAttrSlot(className, *attr)) {
            if (slot->first != className) {
              // CPython would create a NEW attribute on the subclass,
              // shadowing the base's; static storage has no per-subclass
              // presence, so writing through the subclass name is loud.
              diagnostics.push_back(parser::Diagnostic{
                  parser::Severity::Error, target.range.start,
                  "assigning class attribute '" + std::string(*attr) +
                      "' through subclass '" +
                      py::contracts::displayClassNameForContract(className) +
                      "' would shadow '" +
                      py::contracts::displayClassNameForContract(
                          slot->first) +
                      "'; assign through the defining class"});
              return;
            }
            std::string cellName =
                (llvm::Twine(slot->first) + "." + *attr).str();
            Value coerced = coerceValue(value, slot->second, target);
            py::GlobalSetOp::create(builder, loc(target),
                                    builder.getStringAttr(cellName),
                                    coerced.value);
            return;
          }
        }
      } else if (!lookupClassField(object.type, *attr)) {
        if (auto contract =
                mlir::dyn_cast_if_present<py::ContractType>(object.type)) {
          if (resolveClassAttrSlot(contract.getContractName(), *attr)) {
            // CPython creates an instance attribute shadowing the class
            // attribute; the static layout only has fields declared in
            // __init__, so this write cannot be represented.
            diagnostics.push_back(parser::Diagnostic{
                parser::Severity::Error, target.range.start,
                "assigning '" + std::string(*attr) +
                    "' through an instance would shadow the class "
                    "attribute; initialize it in __init__ to make it an "
                    "instance field"});
            return;
          }
        }
      }
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
      emitSliceMutation(target, containerNode, *sliceNode, "__setslice__",
                        value);
      return;
    }
    Value index = emitExpr(ast::node(target, "slice"));
    if (tryEmitClassDunder(target, container, "__setitem__", {index, value}))
      return;
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
      // ⭐ An unpacking target is answered element by element, so a target
      // this walk cannot answer has to stop it. Both of these used to be
      // SILENT:
      //
      //     a, *rest = [1, 2, 3, 4]     # rest kept its previous value
      //     a, b = (1, 2, 3)            # printed 1 2; CPython raises
      //
      // A starred target has no index to read -- it takes however many are
      // left -- so the loop below simply skipped it and whatever `rest` had
      // been bound to before stayed. Where the name did not exist yet the
      // program was refused, which is why this only ever showed up when the
      // target was pre-declared.
      for (const parser::NodePtr &elt : *elts)
        if (elt && elt->kind == "Starred") {
          diagnostics.push_back(parser::Diagnostic{
              parser::Severity::Error, elt->range.start,
              "starred assignment target is not supported: it takes a "
              "statically unknown number of elements"});
          return;
        }
      // The count IS checked, against the object: a check on the type alone
      // catches only heterogeneous tuples (a tuple whose members share a type
      // collapses to `tuple[T]`, and a list carries its length in the object),
      // so it fired wrongly on `a, b = (1, 2)` and took out two working
      // programs before it caught anything.
      emitUnpackArityCheck(target, value, elts->size());
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

// The runtime half of `a, b = xs`: CPython's UNPACK_SEQUENCE raises when the
// source length is not the target arity, and this walk unpacks by index, so
// a longer source silently dropped its tail and a shorter one read past the
// end. The arity is not in the type for the common case -- a tuple whose
// members share a type collapses to `tuple[T]` (TypeSystem.cpp, `uniform`)
// and a list carries its length in the object -- so the check has to be a
// length comparison against the object.
//
// ⛔ Why NOT a static refusal when the arity IS in the type (a heterogeneous
// tuple): `try: a, b = t; except ValueError:` is legal Python, and the
// runtime raise it expects is what this emits. The static case only skips
// the comparison it can already answer.
//
// CPython's two messages differ in one detail that is decided statically:
// the tuple/list fast paths in ceval report what they got, the generic
// iterator path does not know it ("too many values to unpack (expected 2)"
// for a str source, "(expected 2, got 3)" for a list).
void ModuleEmitter::emitUnpackArityCheck(const parser::Node &target,
                                         Value source, std::size_t expected) {
  mlir::Type sourceType = types.widenLiteral(source.type);
  if (auto contract = mlir::dyn_cast_if_present<py::ContractType>(sourceType))
    if (contract.getContractName() == "builtins.tuple" &&
        contract.getArguments().size() > 1 &&
        contract.getArguments().size() == expected)
      return;
  if (!types.inferMethodCallWithEvidence(sourceType, "__len__", {}))
    return;

  bool reportsCount = false;
  if (auto contract = mlir::dyn_cast_if_present<py::ContractType>(sourceType))
    reportsCount = contract.getContractName() == "builtins.tuple" ||
                   contract.getContractName() == "builtins.list";

  parser::SourceRange range = target.range;
  unsigned serial = ++listCompCounter;
  std::string sourceName = "__lyunpack" + std::to_string(serial) + "_s";
  std::string lengthName = "__lyunpack" + std::to_string(serial) + "_n";
  std::string expectedText = std::to_string(expected);

  auto message = [&](const char *prefix) {
    parser::NodePtr text = synth::strConstant(std::string(prefix) +
                                      " values to unpack (expected " +
                                      expectedText + ", got ",
                                  range);
    parser::NodePtr count = synth::call(synth::name("str", range),
                             {synth::name(lengthName, range)}, range);
    return synth::binOp(synth::binOp(std::move(text), "Add", std::move(count), range),
                     "Add", synth::strConstant(")", range), range);
  };
  auto raiseNode = [&](parser::NodePtr text) {
    parser::NodePtr node = synth::raiseStmt(synth::call(synth::name("ValueError", range),
                              {std::move(text)}, range), range);
    return node;
  };

  parser::NodePtr tooMany =
      reportsCount ? message("too many")
                   : synth::strConstant("too many values to unpack (expected " +
                                        expectedText + ")",
                                    range);
  parser::NodePtr overLong = synth::ifStmt(
      synth::compare(synth::name(lengthName, range), "Gt",
                  synth::intConstant(static_cast<std::int64_t>(expected), range),
                  range),
      {raiseNode(std::move(tooMany))}, {}, range);
  parser::NodePtr tooFew = synth::ifStmt(
      synth::compare(synth::name(lengthName, range), "Lt",
                  synth::intConstant(static_cast<std::int64_t>(expected), range),
                  range),
      {raiseNode(message("not enough"))}, {}, range);

  std::vector<parser::NodePtr> guarded;
  guarded.push_back(std::move(overLong));
  guarded.push_back(std::move(tooFew));
  parser::NodePtr check = synth::ifStmt(
      synth::compare(synth::name(lengthName, range), "NotEq",
                  synth::intConstant(static_cast<std::int64_t>(expected), range),
                  range),
      std::move(guarded), {}, range);

  runWithScratchNames({sourceName, lengthName}, [&] {
    values[sourceName] = source;
    emitStatement(*synth::assign(synth::name(lengthName, range),
                              synth::lenCall(synth::name(sourceName, range), range),
                              range));
    emitStatement(*check);
  });
}

} // namespace lython::emitter
