#include "EmitterCore.h"
#include "EmitterPyOps.h"
#include "EmitterSupport.h"

#include "AstAccess.h"

#include "llvm/ADT/STLExtras.h"

namespace lython::emitter {

void ModuleEmitter::emitWithEnter(const parser::Node &item, bool async) {
  {
      Value contextValue = emitExpr(ast::node(item, "context_expr"));
      Value entered;
      if (async) {
        AsyncContextMethodInferenceResult enterInference =
            types.inferAsyncContextEnterWithEvidence(contextValue.type);
        if (!requireStaticEvidence(item, enterInference))
          return;
        auto enter = py::AEnterOp::create(
            builder, loc(item), enterInference.awaitableType, "__aenter__",
            callProtocolFor(enterInference.method), contextValue.value,
            mlir::UnitAttr());
        entered =
            emitAwaitValue(item,
                           Value{enter.getResult(),
                                 enterInference.awaitableType},
                           enterInference.awaitResult);
      } else if (std::optional<MethodBinding> method =
                     (refuseUnresolvableDispatch(item, contextValue,
                                                 "__enter__")
                          ? std::nullopt
                          : lookupClassMethod(contextValue.type,
                                              "__enter__"))) {
        // ⭐ A context manager written in Python. `py.enter` is answered from
        // the runtime manifest, so a user class reached the lowering and was
        // refused there -- "runtime manifest has no Ctx.__enter__ method" --
        // for the plainest `with` in the language. Every other dunder on a
        // user class is inlined at this layer (`__len__`, `__getitem__`,
        // `__eq__`); `__enter__` and `__exit__` were the two that were not,
        // and the manifest op is right only for a manager the manifest knows.
        entered = emitInlineOperatorCall(item, contextValue, *method, {});
      } else {
        CallInferenceResult enterInference =
            types.inferMethodCallWithEvidence(contextValue.type, "__enter__",
                                              {});
        if (!requireStaticEvidence(item, enterInference))
          return;
        mlir::Type enterType = enterInference.resultType;
        auto enter =
            py::EnterOp::create(builder, loc(item), enterType, "__enter__",
                                callProtocolFor(enterInference),
                                contextValue.value, mlir::UnitAttr());
        entered = Value{enter.getResult(), enterType};
      }
      if (const parser::Node *optional = ast::node(item, "optional_vars"))
        emitAssignTarget(*optional, entered);
      activeWithCleanups.push_back(WithCleanup{contextValue, async});
  }
}

// ⭐ The block runs inside an implicit try/finally, which is what `with` MEANS
// -- CPython's compiler builds the same thing, and one try PER ITEM:
//
//     enter A
//     try:
//         enter B
//         try:
//             BODY
//         finally:
//             exit B
//     finally:
//         exit A
//
// Emitting the cleanup only where the body falls through left the exception
// path with no __exit__ at all, and entering every manager before opening any
// try left `with A(), B()` skipping A's __exit__ when B's __enter__ raised.
// Routing through `emitTry` rather than adding an unwind edge here is what
// makes return, break, continue and the raise all one path: that machinery
// already answers every one of them.
//
// ⛔ Why NOT desugar into `mgr.__exit__(None, None, None)` as an AST call: the
// manager is an already-emitted SSA value, and a manifest manager (a file)
// does not answer __exit__ through attribute access at all. The synthesized
// statements carry an INDEX instead -- into the item list for the enter, into
// the cleanup stack for the exit -- which is the one thing an AST node can
// hold about a value.
//
// ⚠️ What this still does not do: `__exit__` always receives
// `(None, None, None)` and its result is discarded, so it cannot tell the
// exception path from the normal one and a truthy return does not suppress.
void ModuleEmitter::emitWith(const parser::Node &statement, bool async) {
  std::size_t cleanupStart = activeWithCleanups.size();
  std::size_t itemStart = pendingWithItems.size();
  const auto *items = ast::nodeList(statement, "items");
  if (items)
    for (const parser::NodePtr &item : *items)
      pendingWithItems.push_back(PendingWithItem{item.get(), async});

  std::vector<parser::NodePtr> nested(
      ast::nodeList(statement, "body") ? *ast::nodeList(statement, "body")
                                       : std::vector<parser::NodePtr>{});
  std::size_t count = items ? items->size() : 0;
  for (std::size_t offset = count; offset > 0; --offset) {
    std::size_t index = offset - 1;
    parser::NodePtr cleanup =
        parser::makeNode("LyWithCleanup", statement.range);
    parser::addField(*cleanup, "slot",
                     static_cast<std::int64_t>(cleanupStart + index));
    parser::NodePtr guarded = parser::makeNode("Try", statement.range);
    parser::addField(*guarded, "body", std::move(nested));
    parser::addField(*guarded, "handlers", std::vector<parser::NodePtr>{});
    parser::addField(*guarded, "orelse", std::vector<parser::NodePtr>{});
    parser::addField(*guarded, "finalbody",
                     std::vector<parser::NodePtr>{std::move(cleanup)});
    parser::NodePtr enter = parser::makeNode("LyWithEnter", statement.range);
    parser::addField(*enter, "slot",
                     static_cast<std::int64_t>(itemStart + index));
    nested = std::vector<parser::NodePtr>{std::move(enter),
                                          std::move(guarded)};
  }
  emitStatements(&nested);
  activeWithCleanups.resize(cleanupStart);
  pendingWithItems.resize(itemStart);
}

void ModuleEmitter::emitWithCleanup(const parser::Node &anchor,
                                    const WithCleanup &cleanup) {
  auto noneOp = py::NoneOp::create(builder, loc(anchor), types.none());
  Value none{noneOp.getResult(), types.none()};
  if (cleanup.async) {
    AsyncContextMethodInferenceResult exitInference =
        types.inferAsyncContextExitWithEvidence(
            cleanup.manager.type, {none.type, none.type, none.type});
    if (!requireStaticEvidence(anchor, exitInference))
      return;
    auto exit = py::AExitOp::create(
        builder, loc(anchor), exitInference.awaitableType, "__aexit__",
        callProtocolFor(exitInference.method), cleanup.manager.value,
        none.value, none.value, none.value, mlir::UnitAttr());
    (void)emitAwaitValue(anchor,
                         Value{exit.getResult(), exitInference.awaitableType},
                         exitInference.awaitResult);
    return;
  }

  // The `__enter__` instance of the same rule; see emitWith.
  if (std::optional<MethodBinding> method =
          (refuseUnresolvableDispatch(anchor, cleanup.manager, "__exit__")
               ? std::nullopt
               : lookupClassMethod(cleanup.manager.type, "__exit__"))) {
    (void)emitInlineOperatorCall(anchor, cleanup.manager, *method,
                                 {none, none, none});
    return;
  }

  CallInferenceResult exitInference = types.inferMethodCallWithEvidence(
      cleanup.manager.type, "__exit__", {none.type, none.type, none.type});
  if (!requireStaticEvidence(anchor, exitInference))
    return;
  py::ExitOp::create(builder, loc(anchor), types.boolType(), "__exit__",
                     callProtocolFor(exitInference), cleanup.manager.value,
                     none.value, none.value, none.value, mlir::UnitAttr());
}

} // namespace lython::emitter
