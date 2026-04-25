#include "cpp/PyVerifier/Common.h"

using namespace mlir;

namespace py {

LogicalResult TryOp::verify() {
  bool hasExcept = !getExceptRegion().empty();
  bool hasFinally = !getFinallyRegion().empty();
  if (!hasExcept && !hasFinally)
    return emitOpError("py.try must have except or finally region");

  if (getTryRegion().empty())
    return emitOpError("try region must not be empty");

  auto requireYieldTypesMatch = [&](Region &region,
                                    StringRef kind) -> LogicalResult {
    if (region.empty())
      return success();
    auto *term = region.front().getTerminator();
    if (!term)
      return emitOpError(kind) << " region must have a terminator";
    if (kind == "try" && !isa<TryYieldOp>(term))
      return emitOpError("try region must terminate with py.try.yield");
    if (kind == "except" && !isa<ExceptYieldOp>(term))
      return emitOpError("except region must terminate with py.except.yield");
    auto expected = getResultTypes();
    auto yielded = term->getOperandTypes();
    if (expected.size() != yielded.size())
      return emitOpError(kind)
             << " region yield count must match py.try result count";
    for (auto [exp, got] : llvm::zip(expected, yielded))
      if (exp != got)
        return emitOpError(kind)
               << " region yield types must match py.try result types";
    return success();
  };

  if (hasExcept) {
    auto &region = getExceptRegion();
    if (region.empty())
      return emitOpError("except region must not be empty");
    Block &entry = region.front();
    if (entry.getNumArguments() != 1)
      return emitOpError("except region must take a single !py.exception");
    if (!isPyExceptionType(entry.getArgument(0).getType()))
      return emitOpError("except region argument must be !py.exception");
  }

  if (failed(requireYieldTypesMatch(getTryRegion(), "try")))
    return failure();
  if (hasExcept && failed(requireYieldTypesMatch(getExceptRegion(), "except")))
    return failure();
  if (hasFinally) {
    auto &region = getFinallyRegion();
    if (region.empty())
      return emitOpError("finally region must not be empty");
    auto *term = region.front().getTerminator();
    if (!term)
      return emitOpError("finally region must have a terminator");
    if (!isa<FinallyYieldOp>(term))
      return emitOpError("finally region must terminate with py.finally.yield");
  }

  return success();
}

LogicalResult RaiseOp::verify() {
  if (!isPyExceptionType(getException().getType()))
    return emitOpError("operand must be !py.exception");
  return success();
}

LogicalResult RaiseCurrentOp::verify() {
  auto *parent = getOperation()->getParentOp();
  auto tryOp = dyn_cast_or_null<TryOp>(parent);
  if (tryOp) {
    if (getOperation()->getParentRegion() != &tryOp.getExceptRegion())
      return emitOpError("must be inside py.try except region");
    return success();
  }

  Block *block = getOperation()->getBlock();
  if (!block)
    return emitOpError("must be nested inside py.try except region");
  Operation *container = block->getParentOp();
  bool isInvokeUnwind = false;
  container->walk([&](InvokeOp invoke) {
    if (invoke.getUnwindDest() == block)
      isInvokeUnwind = true;
  });
  if (isInvokeUnwind)
    return success();

  return emitOpError("must be nested inside py.try except region");
}

LogicalResult FinallyYieldOp::verify() {
  if (getOperation()->getNumOperands() != 0)
    return emitOpError("must not have operands");
  auto *parent = getOperation()->getParentOp();
  auto tryOp = dyn_cast_or_null<TryOp>(parent);
  if (!tryOp)
    return emitOpError("must be nested inside py.try finally region");
  if (getOperation()->getParentRegion() != &tryOp.getFinallyRegion())
    return emitOpError("must be inside py.try finally region");
  return success();
}

LogicalResult ExceptionNewOp::verify() {
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
  return success();
}

LogicalResult ExceptMatchOp::verify() {
  auto *parent = getOperation()->getParentOp();
  auto tryOp = dyn_cast_or_null<TryOp>(parent);
  if (!tryOp)
    return emitOpError("must be nested inside py.try except region");
  if (getOperation()->getParentRegion() != &tryOp.getExceptRegion())
    return emitOpError("must be inside py.try except region");
  return success();
}

} // namespace py
