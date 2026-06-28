#pragma once

#include "EmitterCore.h"

#include "mlir/IR/BuiltinAttributes.h"

namespace lython::emitter {

template <typename Op>
Value ModuleEmitter::emitBinarySpecial(const parser::Node &anchor,
                                       llvm::StringRef method, Value lhs,
                                       Value rhs, mlir::Type resultType) {
  CallInferenceResult inference =
      types.inferMethodCallWithEvidence(lhs.type, method, {rhs.type});
  if (inference)
    resultType = inference.resultType;
  auto op = builder.create<Op>(
      loc(anchor), resultType, mlir::FlatSymbolRefAttr::get(&context, method),
      method, callProtocolFor(inference), lhs.value, rhs.value);
  return {op.getResult(), resultType};
}

template <typename Op>
Value ModuleEmitter::emitUnarySpecial(const parser::Node &anchor,
                                      llvm::StringRef method, Value input,
                                      mlir::Type resultType) {
  CallInferenceResult inference =
      types.inferMethodCallWithEvidence(input.type, method, {});
  if (inference)
    resultType = inference.resultType;
  auto op = builder.create<Op>(loc(anchor), resultType,
                               mlir::FlatSymbolRefAttr::get(&context, method),
                               method, callProtocolFor(inference), input.value);
  return {op.getResult(), resultType};
}

} // namespace lython::emitter
