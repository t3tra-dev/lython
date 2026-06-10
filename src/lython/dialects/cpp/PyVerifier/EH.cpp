#include "cpp/PyVerifier/Common.h"

#include "cpp/PyTypeObject.h"

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
    if (getOperation()->getParentRegion() != &tryOp.getExceptRegion()) {
      if (getOperation()->getParentRegion() == &tryOp.getTryRegion()) {
        mlir::Block *block = getOperation()->getBlock();
        for (mlir::Block *pred : block->getPredecessors()) {
          auto invoke = mlir::dyn_cast_or_null<InvokeOp>(pred->getTerminator());
          if (invoke && invoke.getUnwindDest() == block)
            return mlir::success();
        }
        return emitOpError("must be inside py.try except region");
      } else {
        return emitOpError("must be inside py.try except region");
      }
    }
    return mlir::success();
  }

  mlir::Block *block = getOperation()->getBlock();
  if (!block)
    return emitOpError("must be nested inside py.try except region");
  mlir::Operation *container = block->getParentOp();
  bool isInvokeUnwind = false;
  auto detectInvokeUnwind = [&](mlir::Operation *scope) {
    if (!scope)
      return;
    scope->walk([&](InvokeOp invoke) {
      if (invoke.getUnwindDest() == block)
        isInvokeUnwind = true;
    });
  };
  detectInvokeUnwind(container);
  if (!isInvokeUnwind)
    if (mlir::Region *region = block->getParent())
      if (mlir::Operation *parent = region->getParentOp())
        detectInvokeUnwind(parent);
  if (!isInvokeUnwind)
    if (mlir::Operation *parent =
            container ? container->getParentOp() : nullptr)
      detectInvokeUnwind(parent);
  if (!isInvokeUnwind) {
    for (mlir::Block *pred : block->getPredecessors()) {
      auto invoke = mlir::dyn_cast_or_null<InvokeOp>(pred->getTerminator());
      if (invoke && invoke.getUnwindDest() == block) {
        isInvokeUnwind = true;
        break;
      }
    }
  }
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
  if (!isPyExceptionType(getResult().getType()))
    return emitOpError("result must be !py.exception");
  mlir::FailureOr<bool> hasExceptionRoot = type_object::isSubclassOf(
      getOperation(), type_object::kException, type_object::kBaseException);
  if (mlir::failed(hasExceptionRoot))
    return mlir::failure();
  if (!*hasExceptionRoot)
    return emitOpError("requires Exception <: BaseException");
  if (getArgs().size() > 1)
    return emitOpError("takes at most one message argument");
  for (mlir::Value arg : getArgs())
    if (!isPyStrType(arg.getType()))
      return emitOpError("message argument must be !py.str");
  if (auto classAttr =
          (*this)->getAttrOfType<mlir::StringAttr>("py.exception.class")) {
    mlir::FailureOr<bool> isExceptionClass = type_object::isSubclassOf(
        getOperation(), classAttr.getValue(), type_object::kBaseException);
    if (mlir::failed(isExceptionClass))
      return mlir::failure();
    if (!*isExceptionClass)
      return emitOpError("'py.exception.class' must name a BaseException "
                         "subclass, got '")
             << classAttr.getValue() << "'";
  }
  return mlir::success();
}

mlir::LogicalResult ExceptMatchOp::verify() {
  auto *parent = getOperation()->getParentOp();
  auto tryOp = mlir::dyn_cast_or_null<TryOp>(parent);
  if (!tryOp)
    return emitOpError("must be nested inside py.try except region");
  if (getOperation()->getParentRegion() != &tryOp.getExceptRegion())
    return emitOpError("must be inside py.try except region");
  auto handlerType = mlir::dyn_cast<ClassType>(getHandlerAttr().getValue());
  if (!handlerType)
    return emitOpError("handler must be a !py.class type attribute");
  mlir::FailureOr<bool> isExceptionClass = type_object::isSubclassOf(
      getOperation(), handlerType.getClassName(), type_object::kBaseException);
  if (mlir::failed(isExceptionClass))
    return mlir::failure();
  if (!*isExceptionClass)
    return emitOpError("handler class '")
           << handlerType.getClassName()
           << "' is not a static subclass of BaseException";
  return mlir::success();
}

} // namespace py
