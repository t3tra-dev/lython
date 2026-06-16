#include "cpp/PyVerifier/Common.h"

namespace py {

mlir::LogicalResult AwaitOp::verify() {
  mlir::Type payloadType = awaitablePayloadType(getAwaitable().getType());
  if (!payloadType)
    return emitOpError(
        "operand must satisfy Awaitable[T], Coroutine[Any, Any, T], "
        "Future[T], Task[T], or be !async.value<T>");
  if (getResult().getType() != payloadType)
    return emitOpError("result type must match awaitable payload type");
  return mlir::success();
}

mlir::LogicalResult AsyncNextOp::verify() {
  mlir::Type payloadType = awaitablePayloadType(getAwaitable().getType());
  if (!payloadType)
    return emitOpError(
        "operand must satisfy Awaitable[T], Coroutine[Any, Any, T], "
        "Future[T], Task[T], or be !async.value<T>");
  if (getElement().getType() != payloadType)
    return emitOpError("element result type must match awaitable payload type");
  return mlir::success();
}

} // namespace py
