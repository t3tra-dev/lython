#include "cpp/PyVerifier/Common.h"

#include "mlir/Dialect/Async/IR/Async.h"
#include "llvm/ADT/STLExtras.h"

namespace py {

static mlir::Type getAwaitableResultType(mlir::Type type) {
  if (auto coroType = mlir::dyn_cast<CoroutineType>(type))
    return coroType.getResultType();
  if (auto taskType = mlir::dyn_cast<TaskType>(type))
    return taskType.getResultType();
  if (auto futureType = mlir::dyn_cast<FutureType>(type))
    return futureType.getResultType();
  if (auto valueType = mlir::dyn_cast<mlir::async::ValueType>(type))
    return valueType.getValueType();
  return {};
}

mlir::LogicalResult CoroCreateOp::verify() {
  if (!mlir::isa<CoroutineType>(getResult().getType()))
    return emitOpError("result must be of type !py.coro");
  return mlir::success();
}

mlir::LogicalResult CoroStartOp::verify() {
  auto coroType = mlir::dyn_cast<CoroutineType>(getCoroutine().getType());
  if (!coroType)
    return emitOpError("operand must be of type !py.coro");

  auto resultType =
      mlir::dyn_cast<mlir::async::ValueType>(getResult().getType());
  if (!resultType)
    return emitOpError("result must be of type !async.value");
  if (resultType.getValueType() != coroType.getResultType())
    return emitOpError("async value payload must match coroutine result type");
  return mlir::success();
}

mlir::LogicalResult AwaitOp::verify() {
  mlir::Type payloadType = getAwaitableResultType(getAwaitable().getType());
  if (!payloadType)
    return emitOpError(
        "operand must be !py.coro, !py.task, !py.future, or !async.value");
  if (getResult().getType() != payloadType)
    return emitOpError("result type must match awaitable payload type");
  return mlir::success();
}

mlir::LogicalResult TaskCreateOp::verify() {
  auto coroType = mlir::dyn_cast<CoroutineType>(getCoroutine().getType());
  if (!coroType)
    return emitOpError("operand must be of type !py.coro");

  auto taskType = mlir::dyn_cast<TaskType>(getResult().getType());
  if (!taskType)
    return emitOpError("result must be of type !py.task");
  if (taskType.getResultType() != coroType.getResultType())
    return emitOpError("task payload must match coroutine result type");
  return mlir::success();
}

mlir::LogicalResult TaskCancelOp::verify() {
  if (!mlir::isa<TaskType>(getTask().getType()))
    return emitOpError("operand must be of type !py.task");
  if (!mlir::isa<BoolType>(getAccepted().getType()))
    return emitOpError("result must be of type !py.bool");
  return mlir::success();
}

mlir::LogicalResult AsyncSleepOp::verify() {
  if (!mlir::isa<FloatType, mlir::FloatType>(getSeconds().getType()))
    return emitOpError("seconds operand must be !py.float or an MLIR float");

  auto futureType = mlir::dyn_cast<FutureType>(getResult().getType());
  if (!futureType)
    return emitOpError("result must be of type !py.future");
  if (!mlir::isa<NoneType>(futureType.getResultType()))
    return emitOpError("async.sleep result must be !py.future<!py.none>");
  return mlir::success();
}

mlir::LogicalResult AsyncGatherOp::verify() {
  auto tupleType = mlir::dyn_cast<TupleType>(getResult().getType());
  if (!tupleType)
    return emitOpError("result must be of type !py.tuple");

  auto elementTypes = tupleType.getElementTypes();
  if (elementTypes.size() != getAwaitables().size())
    return emitOpError("result tuple arity must match awaitable count");

  for (auto [index, awaitable] : llvm::enumerate(getAwaitables())) {
    mlir::Type payloadType = getAwaitableResultType(awaitable.getType());
    if (!payloadType)
      return emitOpError()
             << "operand #" << index
             << " must be !py.coro, !py.task, !py.future, or !async.value";
    if (payloadType != elementTypes[index])
      return emitOpError() << "result tuple element #" << index
                           << " must match awaitable payload type";
  }
  return mlir::success();
}

} // namespace py
