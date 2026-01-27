#include "cpp/PyVerifier/Common.h"

using namespace mlir;

namespace py {

ClassOp lookupClassSymbol(Operation *from, ClassType classType) {
  StringAttr nameAttr =
      StringAttr::get(from->getContext(), classType.getClassName());
  for (Operation *symbolTableOp = from; symbolTableOp;
       symbolTableOp = symbolTableOp->getParentOp()) {
    if (!symbolTableOp->hasTrait<OpTrait::SymbolTable>())
      continue;
    if (Operation *symbol =
            SymbolTable::lookupSymbolIn(symbolTableOp, nameAttr))
      if (ClassOp classOp = dyn_cast<ClassOp>(symbol))
        return classOp;
  }
  return nullptr;
}

FuncOp lookupMethodByName(ClassOp classOp, StringRef methodName) {
  for (FuncOp method : classOp.getBody().front().getOps<FuncOp>()) {
    if (mlir::StringAttr nameAttr = method.getSymNameAttr())
      if (nameAttr.getValue() == methodName)
        return method;
  }
  return nullptr;
}

static ThrowEffect getEffectFromFuncOp(FuncOp funcOp) {
  if (funcOp->getAttr("nothrow"))
    return ThrowEffect::NoThrow;
  if (funcOp->getAttr("maythrow"))
    return ThrowEffect::MayThrow;
  return ThrowEffect::MayThrow;
}

FailureOr<FuncSignatureType>
resolveCallableSignature(Operation *op, Value callable,
                         ArrayAttr &expectedKwNamesAttr) {
  Type callableType = callable.getType();
  if (auto funcTy = dyn_cast<FuncType>(callableType)) {
    if (auto funcObject = callable.getDefiningOp<FuncObjectOp>()) {
      Operation *symbol = SymbolTable::lookupNearestSymbolFrom(
          op, funcObject.getTargetAttr());
      if (auto funcOp = dyn_cast_or_null<FuncOp>(symbol))
        expectedKwNamesAttr = funcOp.getKwonlyNamesAttr();
    } else if (auto makeFunc = callable.getDefiningOp<MakeFunctionOp>()) {
      Operation *symbol = SymbolTable::lookupNearestSymbolFrom(
          op, makeFunc.getTargetAttr());
      if (auto funcOp = dyn_cast_or_null<FuncOp>(symbol))
        expectedKwNamesAttr = funcOp.getKwonlyNamesAttr();
    }
    return funcTy.getSignature();
  }

  if (auto classTy = dyn_cast<ClassType>(callableType)) {
    ClassOp classOp = lookupClassSymbol(op, classTy);
    if (!classOp) {
      op->emitOpError("unable to resolve class '")
          << classTy.getClassName() << "'";
      return failure();
    }

    FuncOp callMethod = lookupMethodByName(classOp, "__call__");
    if (!callMethod) {
      op->emitOpError("class '")
          << classTy.getClassName() << "' does not define a '__call__' method";
      return failure();
    }

    auto fnTypeAttr = callMethod.getFunctionTypeAttr();
    if (!fnTypeAttr) {
      callMethod.emitOpError("requires 'function_type' attribute");
      return failure();
    }
    auto methodSig = dyn_cast<FuncSignatureType>(fnTypeAttr.getValue());
    if (!methodSig) {
      callMethod.emitOpError("'function_type' must be a FuncSignatureType");
      return failure();
    }

    auto positionalTypes = methodSig.getPositionalTypes();
    if (positionalTypes.empty()) {
      callMethod.emitOpError(
          "__call__ must declare a positional 'self' parameter");
      return failure();
    }

    Type selfType = positionalTypes.front();
    if (selfType != classTy && !isPyObjectType(selfType)) {
      callMethod.emitOpError(
          "first positional parameter must be of type !py.class<")
          << classOp.getSymName() << "> or !py.object";
      return failure();
    }

    llvm::SmallVector<Type> callPositional(positionalTypes.begin() + 1,
                                           positionalTypes.end());
    llvm::SmallVector<Type> callKwOnly(methodSig.getKwOnlyTypes().begin(),
                                       methodSig.getKwOnlyTypes().end());
    Type callVararg =
        methodSig.hasVararg() ? methodSig.getVarargType() : Type();
    Type callKwarg = methodSig.hasKwarg() ? methodSig.getKwargType() : Type();

    expectedKwNamesAttr = callMethod.getKwonlyNamesAttr();
    return FuncSignatureType::get(methodSig.getContext(), callPositional,
                                  callKwOnly, callVararg, callKwarg,
                                  methodSig.getResultTypes());
  }

  op->emitOpError("unexpected callable type");
  return failure();
}

FailureOr<ThrowEffect> resolveCallableThrowEffect(Operation *op,
                                                  Value callable) {
  Type callableType = callable.getType();
  if (auto funcTy = dyn_cast<FuncType>(callableType)) {
    if (auto funcObject = callable.getDefiningOp<FuncObjectOp>()) {
      Operation *symbol = SymbolTable::lookupNearestSymbolFrom(
          op, funcObject.getTargetAttr());
      if (auto funcOp = dyn_cast_or_null<FuncOp>(symbol))
        return getEffectFromFuncOp(funcOp);
    } else if (auto makeFunc = callable.getDefiningOp<MakeFunctionOp>()) {
      Operation *symbol = SymbolTable::lookupNearestSymbolFrom(
          op, makeFunc.getTargetAttr());
      if (auto funcOp = dyn_cast_or_null<FuncOp>(symbol))
        return getEffectFromFuncOp(funcOp);
    }
    return ThrowEffect::MayThrow;
  }

  if (auto classTy = dyn_cast<ClassType>(callableType)) {
    ClassOp classOp = lookupClassSymbol(op, classTy);
    if (!classOp) {
      op->emitOpError("unable to resolve class '")
          << classTy.getClassName() << "'";
      return failure();
    }

    FuncOp callMethod = lookupMethodByName(classOp, "__call__");
    if (!callMethod) {
      op->emitOpError("class '")
          << classTy.getClassName() << "' does not define a '__call__' method";
      return failure();
    }

    return getEffectFromFuncOp(callMethod);
  }

  op->emitOpError("unexpected callable type");
  return failure();
}

FailureOr<TupleType> requireTuple(Operation *op, Value operand,
                                  StringRef what) {
  if (auto tupleTy = dyn_cast<TupleType>(operand.getType()))
    return tupleTy;
  op->emitOpError() << what << " operand must be a !py.tuple value";
  return failure();
}

LogicalResult verifyVectorCallOperands(Operation *op, FuncSignatureType signature,
                                       Value posargs, Value kwnames,
                                       Value kwvalues,
                                       ArrayAttr expectedKwNamesAttr) {
  auto posTupleOr = requireTuple(op, posargs, "posargs");
  if (failed(posTupleOr))
    return failure();
  TupleType posTuple = *posTupleOr;
  auto posElems = posTuple.getElementTypes();
  bool homogeneous = posElems.size() == 1;

  auto positionalTypes = signature.getPositionalTypes();
  if (!signature.hasVararg()) {
    if (posElems.size() != positionalTypes.size())
      return op->emitOpError(
          "posargs length mismatch with callee positional parameters");
    for (auto [elemType, expected] : llvm::zip(posElems, positionalTypes))
      if (!isSubtypeOf(elemType, expected))
        return op->emitOpError(
            "posargs element type does not match positional parameter type");
  } else {
    auto calleeVarargTy = dyn_cast<TupleType>(signature.getVarargType());
    if (!calleeVarargTy || calleeVarargTy.getElementTypes().size() != 1)
      return op->emitOpError(
          "vararg parameter must be !py.tuple<T> with single element type");
    Type calleeElemType = calleeVarargTy.getElementTypes().front();

    if (homogeneous && !posElems.empty() && positionalTypes.empty()) {
      if (!isSubtypeOf(posElems.front(), calleeElemType))
        return op->emitOpError(
            "posargs element type is incompatible with vararg element type");
    } else {
      if (posElems.size() < positionalTypes.size())
        return op->emitOpError(
            "posargs length shorter than positional parameter count");
      for (size_t i = 0; i < positionalTypes.size(); ++i)
        if (!isSubtypeOf(posElems[i], positionalTypes[i]))
          return op->emitOpError(
                     "posargs element type does not match positional "
                     "parameter type at index ")
                 << i;
      for (size_t i = positionalTypes.size(); i < posElems.size(); ++i)
        if (!isSubtypeOf(posElems[i], calleeElemType))
          return op->emitOpError("vararg element type incompatible at index ")
                 << i;
    }
  }

  auto nameTupleOr = requireTuple(op, kwnames, "kwnames");
  if (failed(nameTupleOr))
    return failure();
  auto valueTupleOr = requireTuple(op, kwvalues, "kwvalues");
  if (failed(valueTupleOr))
    return failure();

  TupleType nameTuple = *nameTupleOr;
  TupleType valueTuple = *valueTupleOr;
  auto nameElems = nameTuple.getElementTypes();
  for (Type elem : nameElems)
    if (!isPyStrType(elem))
      return op->emitOpError("kwnames tuple elements must be !py.str");

  auto valueElems = valueTuple.getElementTypes();
  for (Type elem : valueElems)
    if (!isPyObjectType(elem))
      return op->emitOpError("kwvalues tuple elements must be !py.object");

  if (nameElems.size() != valueElems.size())
    return op->emitOpError(
        "kwnames and kwvalues must describe the same number of entries");

  auto kwonlyTypes = signature.getKwOnlyTypes();
  bool kwargsRequired = signature.hasKwarg() || !kwonlyTypes.empty();
  bool namesEmpty = nameElems.empty();
  if (kwargsRequired && namesEmpty)
    return op->emitOpError(
        "callee expects keyword arguments but none were provided");
  if (!kwargsRequired && !namesEmpty)
    return op->emitOpError("callee does not accept keyword arguments");

  auto collectTupleValues = [&](Value tupleValue,
                                SmallVectorImpl<Value> &values) -> bool {
    if (auto create = tupleValue.getDefiningOp<TupleCreateOp>()) {
      values.append(create.getElements().begin(), create.getElements().end());
      return true;
    }
    if (tupleValue.getDefiningOp<TupleEmptyOp>())
      return true;
    return false;
  };

  SmallVector<Value> nameValues;
  bool hasLiteralNames = collectTupleValues(kwnames, nameValues);
  SmallVector<StringRef> providedKwNames;
  if (hasLiteralNames) {
    for (Value nameVal : nameValues) {
      if (auto strConst = nameVal.getDefiningOp<StrConstantOp>()) {
        providedKwNames.push_back(strConst.getValueAttr().getValue());
      } else {
        hasLiteralNames = false;
        break;
      }
    }
  }

  if (!kwargsRequired && !nameValues.empty())
    return op->emitOpError("callee does not accept keyword arguments");
  if (kwargsRequired && nameValues.empty() && nameElems.empty())
    return op->emitOpError(
        "callee expects keyword arguments but none were provided");

  if (expectedKwNamesAttr && hasLiteralNames) {
    llvm::SmallVector<StringRef> expectedNames;
    for (Attribute attr : expectedKwNamesAttr) {
      if (auto strAttr = dyn_cast<StringAttr>(attr))
        expectedNames.push_back(strAttr.getValue());
    }
    llvm::SmallSet<StringRef, 8> providedSet;
    for (StringRef providedName : providedKwNames)
      if (!providedSet.insert(providedName).second)
        return op->emitOpError("duplicate keyword '") << providedName << "'";
    for (StringRef expected : expectedNames)
      if (!providedSet.contains(expected))
        return op->emitOpError("missing keyword argument '") << expected
                                                             << "'";
    if (!signature.hasKwarg()) {
      if (providedKwNames.size() != expectedNames.size())
        return op->emitOpError("unexpected keyword argument provided");
    }
  }

  return success();
}

} // namespace py
