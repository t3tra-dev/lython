#include "EmitterCore.h"
#include "EmitterPyOps.h"
#include "EmitterSupport.h"

#include "AstAccess.h"

#include "llvm/ADT/STLExtras.h"

namespace lython::emitter {

void ModuleEmitter::emitWith(const parser::Node &statement, bool async) {
  std::size_t cleanupStart = activeWithCleanups.size();
  if (const auto *items = ast::nodeList(statement, "items")) {
    for (const parser::NodePtr &item : *items) {
      Value contextValue = emitExpr(ast::node(*item, "context_expr"));
      Value entered;
      if (async) {
        AsyncContextMethodInferenceResult enterInference =
            types.inferAsyncContextEnterWithEvidence(contextValue.type);
        if (!requireStaticEvidence(*item, enterInference))
          return;
        auto enter = py::AEnterOp::create(
            builder, loc(*item), enterInference.awaitableType, "__aenter__",
            callProtocolFor(enterInference.method), contextValue.value,
            mlir::UnitAttr());
        entered =
            emitAwaitValue(*item,
                           Value{enter.getResult(),
                                 enterInference.awaitableType},
                           enterInference.awaitResult);
      } else if (std::optional<MethodBinding> method =
                     lookupClassMethod(contextValue.type, "__enter__")) {
        // ⭐ A context manager written in Python. `py.enter` is answered from
        // the runtime manifest, so a user class reached the lowering and was
        // refused there -- "runtime manifest has no Ctx.__enter__ method" --
        // for the plainest `with` in the language. Every other dunder on a
        // user class is inlined at this layer (`__len__`, `__getitem__`,
        // `__eq__`); `__enter__` and `__exit__` were the two that were not,
        // and the manifest op is right only for a manager the manifest knows.
        entered = emitInlineOperatorCall(*item, contextValue, *method, {});
      } else {
        CallInferenceResult enterInference =
            types.inferMethodCallWithEvidence(contextValue.type, "__enter__",
                                              {});
        if (!requireStaticEvidence(*item, enterInference))
          return;
        mlir::Type enterType = enterInference.resultType;
        auto enter =
            py::EnterOp::create(builder, loc(*item), enterType, "__enter__",
                                callProtocolFor(enterInference),
                                contextValue.value, mlir::UnitAttr());
        entered = Value{enter.getResult(), enterType};
      }
      if (const parser::Node *optional = ast::node(*item, "optional_vars"))
        emitAssignTarget(*optional, entered);
      activeWithCleanups.push_back(WithCleanup{contextValue, async});
    }
  }
  emitStatements(ast::nodeList(statement, "body"));

  if (!insertionBlockTerminated(builder)) {
    for (std::size_t index = activeWithCleanups.size(); index > cleanupStart;
         --index)
      emitWithCleanup(statement, activeWithCleanups[index - 1]);
  }
  activeWithCleanups.resize(cleanupStart);
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
          lookupClassMethod(cleanup.manager.type, "__exit__")) {
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

// ⭐ Only the cleanups this scope opened. A `return` inside an INLINED method
// body leaves that body, not the enclosing function, so the `with` blocks
// around the call site are still live and must not be torn down:
//
//     class Ctx:
//         def __exit__(self, a, b, c) -> bool:
//             return False
//     with Ctx():
//         ...
//
// The first thing `__exit__`'s inlined body does is return, which ran the
// cleanup that was inlining it -- "recursive class method call is not
// supported (__exit__ -> __exit__)". Any inlined method called inside a
// `with` block had the same shape; `__exit__` is only the one that closes the
// cycle into itself.
std::size_t ModuleEmitter::currentWithCleanupWatermark() const {
  return inlineReturnContexts.empty()
             ? 0
             : inlineReturnContexts.back().withCleanupWatermark;
}

void ModuleEmitter::emitActiveCleanups(const parser::Node &anchor) {
  std::size_t watermark = currentWithCleanupWatermark();
  for (std::size_t index = activeWithCleanups.size(); index > watermark; --index)
    emitWithCleanup(anchor, activeWithCleanups[index - 1]);
}

} // namespace lython::emitter
