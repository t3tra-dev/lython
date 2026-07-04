#include "Common/RuntimeSupport.h"

#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/STLExtras.h"

#include <memory>

namespace py::runtime_lowering {
namespace {

class AsyncThunkLoweringPass
    : public mlir::PassWrapper<AsyncThunkLoweringPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AsyncThunkLoweringPass)

  llvm::StringRef getArgument() const final {
    return "lython-async-thunk-lowering";
  }
  llvm::StringRef getDescription() const final {
    return "lower Lython-owned async dialect thunks to the runtime ABI";
  }

  void runOnOperation() final {
    llvm::SmallVector<mlir::async::ExecuteOp, 8> executes;
    getOperation().walk([&](mlir::async::ExecuteOp execute) {
      if (execute->hasAttr("ly.async.python_await"))
        executes.push_back(execute);
    });

    for (mlir::async::ExecuteOp execute : executes) {
      if (mlir::failed(lowerPythonAwaitThunk(execute))) {
        signalPassFailure();
        return;
      }
    }
  }

private:
  static bool isCoroutineFrameMarker(mlir::Operation *op) {
    return mlir::isa<mlir::async::CoroIdOp, mlir::async::CoroBeginOp,
                     mlir::async::CoroEndOp, mlir::async::CoroFreeOp,
                     mlir::async::CoroSaveOp>(op);
  }

  static bool canEraseCoroutineFrameMarker(mlir::Operation *op) {
    return llvm::all_of(op->getResults(),
                        [](mlir::Value result) { return result.use_empty(); });
  }

  static mlir::LogicalResult eraseCoroutineFrameMarkers(
      mlir::async::ExecuteOp execute,
      llvm::SmallVectorImpl<mlir::Operation *> &markers) {
    bool changed = true;
    while (changed) {
      changed = false;
      for (auto marker = markers.begin(); marker != markers.end();) {
        if (!canEraseCoroutineFrameMarker(*marker)) {
          ++marker;
          continue;
        }
        (*marker)->erase();
        marker = markers.erase(marker);
        changed = true;
      }
    }

    if (!markers.empty())
      return execute.emitError()
             << "Lython Python-await coroutine frame marker still has uses "
                "after thunk lowering";
    return mlir::success();
  }

  mlir::LogicalResult lowerPythonAwaitThunk(mlir::async::ExecuteOp execute) {
    if (!execute.getToken().use_empty())
      return execute.emitError()
             << "Lython Python-await async thunk token must be unused";

    mlir::Block &body = execute.getBodyRegion().front();
    auto yield = mlir::dyn_cast<mlir::async::YieldOp>(body.getTerminator());
    if (!yield)
      return execute.emitError()
             << "Lython Python-await async thunk must end with async.yield";
    if (yield.getOperands().size() != execute.getBodyResults().size())
      return execute.emitError()
             << "Lython Python-await async thunk yield/result arity mismatch";

    llvm::SmallVector<mlir::async::AwaitOp, 4> awaits;
    awaits.reserve(execute.getBodyResults().size());
    mlir::Operation *firstAwait = nullptr;
    for (mlir::Value result : execute.getBodyResults()) {
      if (!result.hasOneUse())
        return execute.emitError()
               << "Lython Python-await async thunk result must have one use";
      auto await =
          mlir::dyn_cast<mlir::async::AwaitOp>(*result.getUsers().begin());
      if (!await)
        return execute.emitError()
               << "Lython Python-await async thunk result must be consumed by "
                  "async.await";
      if (await->getBlock() != execute->getBlock())
        return await.emitError()
               << "Lython Python-await async thunk await must stay in the "
                  "producer block";
      if (!firstAwait || await->isBeforeInBlock(firstAwait))
        firstAwait = await;
      awaits.push_back(await);
    }

    mlir::Operation *anchor = firstAwait ? firstAwait : execute.getOperation();
    llvm::SmallVector<mlir::Operation *, 4> coroutineFrameMarkers;
    while (!body.empty() && &body.front() != yield) {
      mlir::Operation *moved = &body.front();
      moved->moveBefore(anchor);
      if (isCoroutineFrameMarker(moved))
        coroutineFrameMarkers.push_back(moved);
    }

    if (mlir::failed(
            eraseCoroutineFrameMarkers(execute, coroutineFrameMarkers)))
      return mlir::failure();

    for (auto [index, await] : llvm::enumerate(awaits)) {
      if (await.getNumResults() != 1)
        return await.emitError()
               << "Lython Python-await async thunk expects value awaits";
      await.getResult().replaceAllUsesWith(yield.getOperand(index));
      await.erase();
    }

    execute.erase();
    return mlir::success();
  }
};

} // namespace
} // namespace py::runtime_lowering

namespace py {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createAsyncThunkLoweringPass() {
  return std::make_unique<runtime_lowering::AsyncThunkLoweringPass>();
}

} // namespace py
