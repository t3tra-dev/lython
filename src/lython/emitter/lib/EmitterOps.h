#pragma once

#include "EmitterCore.h"

#include "mlir/IR/BuiltinAttributes.h"

namespace lython::emitter {

template <typename Op>
Value ModuleEmitter::emitBinarySpecial(const parser::Node &anchor,
                                       llvm::StringRef method, Value lhs,
                                       Value rhs, mlir::Type resultType) {
  // Source-class operator methods (including MRO-inherited and dataclass-
  // synthesized ones) inline like any other source method call -- through the
  // same gate `x.m()` goes through, since `a == b` on a base-typed `a` is the
  // same unresolvable dispatch written differently.
  if (dispatchIsUnresolvable(lhs, method, /*receiverNode=*/nullptr,
                             /*throughSuper=*/false)) {
    if (std::optional<Value> dispatched =
            tryEmitVirtualDispatchWithValues(anchor, lhs, method, {rhs}))
      return *dispatched;
    if (refuseUnresolvableDispatch(anchor, lhs, method))
      return emitNone(anchor);
  }
  if (std::optional<MethodBinding> binding =
          lookupClassMethod(lhs.type, method);
      binding && binding->method)
    return emitInlineOperatorCall(anchor, lhs, *binding, {rhs});
  CallInferenceResult inference =
      types.inferMethodCallWithEvidence(lhs.type, method, {rhs.type});
  // The left operand has no answer: CPython asks the right one for the
  // reflected operator before giving up (tryEmitReflectedBinary).
  if (!inference)
    if (std::optional<Value> reflected =
            tryEmitReflectedBinary(anchor, method, lhs, rhs))
      return *reflected;
  if (!requireStaticEvidence(anchor, inference))
    return emitNone(anchor);
  if (inference)
    resultType = inference.resultType;
  auto op = Op::create(builder, loc(anchor), resultType,
                       mlir::FlatSymbolRefAttr::get(&context, method), method,
                       callProtocolFor(inference), lhs.value, rhs.value);
  return {op.getResult(), resultType};
}

template <typename Op>
Value ModuleEmitter::emitUnarySpecial(const parser::Node &anchor,
                                      llvm::StringRef method, Value input,
                                      mlir::Type resultType) {
  CallInferenceResult inference =
      types.inferMethodCallWithEvidence(input.type, method, {});
  if (!requireStaticEvidence(anchor, inference))
    return emitNone(anchor);
  if (inference)
    resultType = inference.resultType;
  auto op = Op::create(builder, loc(anchor), resultType,
                       mlir::FlatSymbolRefAttr::get(&context, method), method,
                       callProtocolFor(inference), input.value);
  return {op.getResult(), resultType};
}

} // namespace lython::emitter
