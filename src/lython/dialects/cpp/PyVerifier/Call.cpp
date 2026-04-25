#include "cpp/PyVerifier/Common.h"

using namespace mlir;

namespace py {

LogicalResult CallOp::verify() {
  Type callableType = getCallable().getType();

  if (!isCallableType(callableType))
    return emitOpError("callable operand must be !py.func");

  FailureOr<ThrowEffect> effectOr =
      resolveCallableThrowEffect(getOperation(), getCallable());
  if (failed(effectOr))
    return failure();
  ThrowEffect effect = *effectOr;
  if (effect == ThrowEffect::MayThrow)
    return emitOpError("maythrow callee must be invoked with py.invoke");

  FuncSignatureType signature;
  if (FuncType funcTy = dyn_cast<FuncType>(callableType)) {
    signature = funcTy.getSignature();
  } else {
    return emitOpError("unexpected callable type");
  }

  TupleType tupleTy = dyn_cast<TupleType>(getPosargs().getType());
  if (!tupleTy)
    return emitOpError("posargs operand must be a !py.tuple value");
  mlir::ArrayRef<mlir::Type> tupleElems = tupleTy.getElementTypes();
  bool homogeneous = tupleElems.size() == 1;

  mlir::ArrayRef<mlir::Type> positionalTypes = signature.getPositionalTypes();
  unsigned minPositionalCount =
      getMinimumPositionalCountForCallable(signature, getCallable());
  if (!signature.hasVararg()) {
    if (tupleElems.size() < minPositionalCount ||
        tupleElems.size() > positionalTypes.size())
      return emitOpError(
          "posargs length mismatch with callee positional parameters");
    for (auto [elemType, expected] : llvm::zip(tupleElems, positionalTypes))
      if (!isSubtypeOf(elemType, expected))
        return emitOpError(
            "posargs element type does not match positional parameter type");
  } else {
    TupleType calleeVarargTy = dyn_cast<TupleType>(signature.getVarargType());
    if (!calleeVarargTy || calleeVarargTy.getElementTypes().size() != 1)
      return emitOpError(
          "vararg parameter must be !py.tuple<T> with single element type");
    Type calleeElemType = calleeVarargTy.getElementTypes().front();

    if (homogeneous && !tupleElems.empty() && positionalTypes.empty()) {
      if (!isSubtypeOf(tupleElems.front(), calleeElemType))
        return emitOpError(
            "posargs element type is incompatible with vararg element type");
    } else {
      if (tupleElems.size() < positionalTypes.size())
        return emitOpError(
            "posargs length shorter than positional parameter count");
      for (size_t i = 0; i < positionalTypes.size(); ++i)
        if (!isSubtypeOf(tupleElems[i], positionalTypes[i]))
          return emitOpError("posargs element type does not match positional "
                             "parameter type at index ")
                 << i;
      for (size_t i = positionalTypes.size(); i < tupleElems.size(); ++i)
        if (!isSubtypeOf(tupleElems[i], calleeElemType))
          return emitOpError("vararg element type incompatible at index ") << i;
    }
  }

  Type kwargsType = getKwargs().getType();
  bool kwargsIsNone = isa<NoneType>(kwargsType);
  DictType dictTy = dyn_cast<DictType>(kwargsType);
  if (!dictTy && !kwargsIsNone)
    return emitOpError("kwargs operand must be !py.dict or !py.none");

  mlir::ArrayRef<mlir::Type> kwonlyTypes = signature.getKwOnlyTypes();
  bool kwargsRequired = signature.hasKwarg() || !kwonlyTypes.empty();
  if (kwargsRequired) {
    if (kwargsIsNone)
      return emitOpError(
          "callee requires keyword arguments but received !py.none");
    if (!isSubtypeOf(dictTy.getKeyType(), StrType::get(getContext())))
      return emitOpError("kwargs mapping must use !py.str keys");
    if (!isSubtypeOf(dictTy.getValueType(), ObjectType::get(getContext())))
      return emitOpError("kwargs mapping must use !py.object values");
  } else if (!kwargsIsNone) {
    return emitOpError(
        "callee does not accept keyword arguments; use !py.none");
  }

  mlir::ArrayRef<mlir::Type> resultTypes = signature.getResultTypes();
  if (getNumResults() != resultTypes.size())
    return emitOpError("result count mismatch with callee signature");
  for (auto [result, expected] : llvm::zip(getResultTypes(), resultTypes))
    if (result != expected)
      return emitOpError("result types must match callee return types");

  return success();
}

LogicalResult CallVectorOp::verify() {
  ArrayAttr expectedArgNamesAttr;
  ArrayAttr expectedKwNamesAttr;
  FailureOr<FuncSignatureType> signatureOr = resolveCallableSignature(
      getOperation(), getCallable(), expectedArgNamesAttr, expectedKwNamesAttr);
  if (failed(signatureOr))
    return failure();
  FuncSignatureType signature = *signatureOr;

  FailureOr<ThrowEffect> effectOr =
      resolveCallableThrowEffect(getOperation(), getCallable());
  if (failed(effectOr))
    return failure();
  ThrowEffect effect = *effectOr;
  if (effect == ThrowEffect::MayThrow)
    return emitOpError("maythrow callee must be invoked with py.invoke");

  if (failed(verifyVectorCallOperands(
          getOperation(), signature, getCallable(), getPosargs(), getKwnames(),
          getKwvalues(), expectedArgNamesAttr, expectedKwNamesAttr)))
    return failure();

  auto resultTypes = signature.getResultTypes();
  if (getNumResults() != resultTypes.size())
    return emitOpError("result count mismatch with callee signature");
  for (auto [result, expected] : llvm::zip(getResultTypes(), resultTypes)) {
    if (result != expected)
      return emitOpError("result types must match callee return types");
  }

  return success();
}

LogicalResult InvokeOp::verify() {
  ArrayAttr expectedArgNamesAttr;
  ArrayAttr expectedKwNamesAttr;
  FailureOr<FuncSignatureType> signatureOr = resolveCallableSignature(
      getOperation(), getCallable(), expectedArgNamesAttr, expectedKwNamesAttr);
  if (failed(signatureOr))
    return failure();
  FuncSignatureType signature = *signatureOr;

  FailureOr<ThrowEffect> effectOr =
      resolveCallableThrowEffect(getOperation(), getCallable());
  if (failed(effectOr))
    return failure();
  ThrowEffect effect = *effectOr;
  if (effect == ThrowEffect::NoThrow)
    return emitOpError("nothrow callee must be invoked with py.call");

  if (failed(verifyVectorCallOperands(
          getOperation(), signature, getCallable(), getPosargs(), getKwnames(),
          getKwvalues(), expectedArgNamesAttr, expectedKwNamesAttr)))
    return failure();

  auto resultTypes = signature.getResultTypes();
  bool allNone =
      llvm::all_of(resultTypes, [](Type type) { return isPyNoneType(type); });
  if (getNormalDestOperands().empty() && allNone) {
    // Allow dropping !py.none results for statement invokes.
  } else {
    if (getNormalDestOperands().size() != resultTypes.size())
      return emitOpError("normal destination operand count mismatch");
    for (auto [operand, expected] :
         llvm::zip(getNormalDestOperands(), resultTypes))
      if (operand.getType() != expected)
        return emitOpError(
            "normal destination operand types must match callee return types");
  }

  Block *normalDest = getNormalDest();
  if (normalDest->getNumArguments() != getNormalDestOperands().size())
    return emitOpError("normal destination block argument count mismatch");
  for (auto [operand, arg] :
       llvm::zip(getNormalDestOperands(), normalDest->getArguments()))
    if (operand.getType() != arg.getType())
      return emitOpError("normal destination block argument types must match");

  if (getUnwindDestOperands().size() != 1)
    return emitOpError("unwind destination must take a single !py.exception");
  if (!isPyExceptionType(getUnwindDestOperands()[0].getType()))
    return emitOpError("unwind destination must take !py.exception");
  Block *unwindDest = getUnwindDest();
  if (unwindDest->getNumArguments() != getUnwindDestOperands().size())
    return emitOpError("unwind destination block argument count mismatch");
  for (auto [operand, arg] :
       llvm::zip(getUnwindDestOperands(), unwindDest->getArguments()))
    if (operand.getType() != arg.getType())
      return emitOpError("unwind destination block argument types must match");
  return success();
}

LogicalResult NativeCallOp::verify() {
  PrimFuncType primFuncTy = dyn_cast<PrimFuncType>(getCallee().getType());
  if (!primFuncTy)
    return emitOpError("callee must be of type !py.prim.func");

  FunctionType fnType = primFuncTy.getSignature();

  if (fnType.getNumInputs() != getArgs().size())
    return emitOpError("operand count mismatch with callee signature");
  for (auto [operand, expected] : llvm::zip(getArgs(), fnType.getInputs())) {
    if (operand.getType() != expected)
      return emitOpError("operand types must match callee argument types");
    if (isPyType(expected))
      return emitOpError("native callee arguments must not be !py.* types");
  }

  if (fnType.getNumResults() != getNumResults())
    return emitOpError("result count mismatch with callee signature");
  for (auto [result, expected] :
       llvm::zip(getResultTypes(), fnType.getResults())) {
    if (result != expected)
      return emitOpError("result types must match callee return types");
    if (isPyType(expected))
      return emitOpError("native callee results must not be !py.* types");
  }

  return success();
}

} // namespace py
