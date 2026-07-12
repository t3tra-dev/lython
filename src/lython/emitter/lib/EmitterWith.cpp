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

  CallInferenceResult exitInference = types.inferMethodCallWithEvidence(
      cleanup.manager.type, "__exit__", {none.type, none.type, none.type});
  if (!requireStaticEvidence(anchor, exitInference))
    return;
  py::ExitOp::create(builder, loc(anchor), types.boolType(), "__exit__",
                     callProtocolFor(exitInference), cleanup.manager.value,
                     none.value, none.value, none.value, mlir::UnitAttr());
}

void ModuleEmitter::emitActiveCleanups(const parser::Node &anchor) {
  for (const WithCleanup &cleanup : llvm::reverse(activeWithCleanups))
    emitWithCleanup(anchor, cleanup);
}

} // namespace lython::emitter
