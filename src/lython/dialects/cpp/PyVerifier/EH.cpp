#include "cpp/PyVerifier/Common.h"

namespace py {

mlir::LogicalResult TryOp::verify() {
  bool hasExcept = !getExceptRegion().empty();
  bool hasFinally = !getFinallyRegion().empty();
  if (!hasExcept && !hasFinally)
    return emitOpError("py.try must have except or finally region");

  if (getTryRegion().empty())
    return emitOpError("try region must not be empty");

  auto requireYieldTypesMatch =
      [&](mlir::Region &region, llvm::StringRef kind) -> mlir::LogicalResult {
    if (region.empty())
      return mlir::success();
    auto *term = region.front().getTerminator();
    if (!term)
      return emitOpError(kind) << " region must have a terminator";
    if (kind == "try" && !mlir::isa<TryYieldOp>(term))
      return mlir::success();
    if (kind == "except" && !mlir::isa<ExceptYieldOp>(term))
      return mlir::success();
    auto expected = getResultTypes();
    auto yielded = term->getOperandTypes();
    if (expected.size() != yielded.size())
      return emitOpError(kind)
             << " region yield count must match py.try result count";
    for (auto [exp, got] : llvm::zip(expected, yielded))
      if (exp != got)
        return emitOpError(kind)
               << " region yield types must match py.try result types";
    return mlir::success();
  };

  if (hasExcept) {
    auto &region = getExceptRegion();
    if (region.empty())
      return emitOpError("except region must not be empty");
    mlir::Block &entry = region.front();
    if (entry.getNumArguments() != 1)
      return emitOpError("except region must take a single !py.exception");
    if (!isPyExceptionType(entry.getArgument(0).getType()))
      return emitOpError("except region argument must be !py.exception");
  }

  if (mlir::failed(requireYieldTypesMatch(getTryRegion(), "try")))
    return mlir::failure();
  if (hasExcept &&
      mlir::failed(requireYieldTypesMatch(getExceptRegion(), "except")))
    return mlir::failure();
  if (hasFinally) {
    auto &region = getFinallyRegion();
    if (region.empty())
      return emitOpError("finally region must not be empty");
    auto *term = region.front().getTerminator();
    if (!term)
      return emitOpError("finally region must have a terminator");
    if (!mlir::isa<FinallyYieldOp>(term))
      return emitOpError("finally region must terminate with py.finally.yield");
  }

  return mlir::success();
}

mlir::LogicalResult RaiseOp::verify() {
  if (!isPyExceptionType(getException().getType()))
    return emitOpError("operand must be !py.exception");
  return mlir::success();
}

mlir::LogicalResult RaiseCurrentOp::verify() {
  auto *parent = getOperation()->getParentOp();
  auto tryOp = mlir::dyn_cast_or_null<TryOp>(parent);
  if (tryOp) {
    if (getOperation()->getParentRegion() != &tryOp.getExceptRegion())
      return emitOpError("must be inside py.try except region");
    return mlir::success();
  }

  mlir::Block *block = getOperation()->getBlock();
  if (!block)
    return emitOpError("must be nested inside py.try except region");
  mlir::Operation *container = block->getParentOp();
  bool isInvokeUnwind = false;
  container->walk([&](InvokeOp invoke) {
    if (invoke.getUnwindDest() == block)
      isInvokeUnwind = true;
  });
  if (isInvokeUnwind)
    return mlir::success();

  return emitOpError("must be nested inside py.try except region");
}

mlir::LogicalResult FinallyYieldOp::verify() {
  auto *parent = getOperation()->getParentOp();
  auto tryOp = mlir::dyn_cast_or_null<TryOp>(parent);
  if (!tryOp)
    return emitOpError("must be nested inside py.try finally region");
  if (getOperation()->getParentRegion() != &tryOp.getFinallyRegion())
    return emitOpError("must be inside py.try finally region");
  auto expected = tryOp.getResultTypes();
  auto yielded = getOperation()->getOperandTypes();
  if (expected.size() != yielded.size())
    return emitOpError("operand count must match py.try result count");
  for (auto [exp, got] : llvm::zip(expected, yielded))
    if (exp != got)
      return emitOpError("operand types must match py.try result types");
  return mlir::success();
}

mlir::LogicalResult ExceptionNewOp::verify() {
  if (!isPyClassType(getType().getType()))
    return emitOpError("type must be !py.class");
  if (!isPyStrType(getMessage().getType()))
    return emitOpError("message must be !py.str");
  if (!isPyTupleType(getArgs().getType()))
    return emitOpError("args must be !py.tuple");
  if (!isPyExceptionType(getCause().getType()))
    return emitOpError("cause must be !py.exception");
  if (!isPyExceptionType(getContext().getType()))
    return emitOpError("context must be !py.exception");
  if (!isPyTracebackType(getTraceback().getType()))
    return emitOpError("traceback must be !py.traceback");
  if (!isPyLocationType(getLocation().getType()))
    return emitOpError("location must be !py.location");
  if (!isPyDictType(getExtras().getType()))
    return emitOpError("extras must be !py.dict");
  if (!isPyExceptionType(getResult().getType()))
    return emitOpError("result must be !py.exception");
  return mlir::success();
}

mlir::LogicalResult ExceptMatchOp::verify() {
  auto *parent = getOperation()->getParentOp();
  auto tryOp = mlir::dyn_cast_or_null<TryOp>(parent);
  if (!tryOp)
    return emitOpError("must be nested inside py.try except region");
  if (getOperation()->getParentRegion() != &tryOp.getExceptRegion())
    return emitOpError("must be inside py.try except region");
  return mlir::success();
}

} // namespace py
