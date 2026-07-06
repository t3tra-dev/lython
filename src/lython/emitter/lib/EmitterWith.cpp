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
      CallInferenceResult enterInference = types.inferMethodCallWithEvidence(
          contextValue.type, async ? "__aenter__" : "__enter__", {});
      mlir::Type enterType =
          enterInference ? enterInference.resultType : types.object();
      Value entered;
      if (async) {
        auto enter =
            py::AEnterOp::create(builder, loc(*item), enterType, "__aenter__",
                                 callProtocolFor(enterInference),
                                 contextValue.value, mlir::UnitAttr());
        entered = emitAwaitValue(*item, Value{enter.getResult(), enterType});
      } else {
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
    CallInferenceResult exitInference = types.inferMethodCallWithEvidence(
        cleanup.manager.type, "__aexit__", {none.type, none.type, none.type});
    mlir::Type exitType =
        exitInference ? exitInference.resultType : types.object();
    auto exit = py::AExitOp::create(builder, loc(anchor), exitType, "__aexit__",
                                    callProtocolFor(exitInference),
                                    cleanup.manager.value, none.value,
                                    none.value, none.value, mlir::UnitAttr());
    (void)emitAwaitValue(anchor, Value{exit.getResult(), exitType});
    return;
  }

  CallInferenceResult exitInference = types.inferMethodCallWithEvidence(
      cleanup.manager.type, "__exit__", {none.type, none.type, none.type});
  py::ExitOp::create(builder, loc(anchor), types.boolType(), "__exit__",
                     callProtocolFor(exitInference), cleanup.manager.value,
                     none.value, none.value, none.value, mlir::UnitAttr());
}

void ModuleEmitter::emitActiveCleanups(const parser::Node &anchor) {
  for (const WithCleanup &cleanup : llvm::reverse(activeWithCleanups))
    emitWithCleanup(anchor, cleanup);
}

} // namespace lython::emitter
