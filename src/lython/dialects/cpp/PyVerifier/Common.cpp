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

FailureOr<Type> lookupClassFieldType(Operation *from, ClassType classType,
                                     StringRef fieldName) {
  ClassOp classOp = lookupClassSymbol(from, classType);
  if (!classOp) {
    from->emitOpError("unable to resolve class '")
        << classType.getClassName() << "'";
    return failure();
  }

  ArrayAttr fieldNames = classOp.getFieldNamesAttr();
  ArrayAttr fieldTypes = classOp.getFieldTypesAttr();
  if (!fieldNames || !fieldTypes) {
    from->emitOpError("class '") << classType.getClassName()
                                 << "' does not define a static field schema";
    return failure();
  }

  for (auto [nameAttr, typeAttr] : llvm::zip(fieldNames, fieldTypes)) {
    auto stringAttr = dyn_cast<StringAttr>(nameAttr);
    auto mlirTypeAttr = dyn_cast<TypeAttr>(typeAttr);
    if (!stringAttr || !mlirTypeAttr) {
      from->emitOpError("class '")
          << classType.getClassName() << "' has malformed field schema";
      return failure();
    }
    if (stringAttr.getValue() == fieldName)
      return mlirTypeAttr.getValue();
  }

  from->emitOpError("class '")
      << classType.getClassName() << "' has no field '" << fieldName << "'";
  return failure();
}

static ThrowEffect getEffectFromFuncOp(FuncOp funcOp) {
  if (funcOp->getAttr("nothrow"))
    return ThrowEffect::NoThrow;
  if (funcOp->getAttr("maythrow"))
    return ThrowEffect::MayThrow;
  return ThrowEffect::MayThrow;
}

static Value stripIdentityCasts(Value value);

static FuncOp lookupReturnedCallableFuncFromValue(Operation *op,
                                                  Value callable) {
  if (auto identity = callable.getDefiningOp<CastIdentityOp>()) {
    if (auto symbolAttr = identity->getAttrOfType<SymbolRefAttr>(
            "lython.returned_callable_symbol")) {
      Operation *symbol = SymbolTable::lookupNearestSymbolFrom(op, symbolAttr);
      return dyn_cast_or_null<FuncOp>(symbol);
    }
    return lookupReturnedCallableFuncFromValue(op, identity.getInput());
  }
  if (auto callVector = callable.getDefiningOp<CallVectorOp>()) {
    if (auto symbolAttr = callVector->getAttrOfType<SymbolRefAttr>(
            "lython.returned_callable_symbol")) {
      Operation *symbol = SymbolTable::lookupNearestSymbolFrom(op, symbolAttr);
      return dyn_cast_or_null<FuncOp>(symbol);
    }
  }
  if (auto call = callable.getDefiningOp<CallOp>()) {
    if (auto symbolAttr = call->getAttrOfType<SymbolRefAttr>(
            "lython.returned_callable_symbol")) {
      Operation *symbol = SymbolTable::lookupNearestSymbolFrom(op, symbolAttr);
      return dyn_cast_or_null<FuncOp>(symbol);
    }
  }
  return nullptr;
}

static IntegerAttr getReturnedCallableDefaultsCountAttr(Value callable) {
  if (auto identity = callable.getDefiningOp<CastIdentityOp>()) {
    if (auto attr = identity->getAttrOfType<IntegerAttr>(
            "lython.returned_callable_defaults_count"))
      return attr;
    return getReturnedCallableDefaultsCountAttr(identity.getInput());
  }
  if (auto callVector = callable.getDefiningOp<CallVectorOp>())
    return callVector->getAttrOfType<IntegerAttr>(
        "lython.returned_callable_defaults_count");
  if (auto call = callable.getDefiningOp<CallOp>())
    return call->getAttrOfType<IntegerAttr>(
        "lython.returned_callable_defaults_count");
  return {};
}

static ArrayAttr getReturnedCallableKwdefaultNamesAttr(Value callable) {
  if (auto identity = callable.getDefiningOp<CastIdentityOp>()) {
    if (auto attr = identity->getAttrOfType<ArrayAttr>(
            "lython.returned_callable_kwdefault_names"))
      return attr;
    return getReturnedCallableKwdefaultNamesAttr(identity.getInput());
  }
  if (auto callVector = callable.getDefiningOp<CallVectorOp>())
    return callVector->getAttrOfType<ArrayAttr>(
        "lython.returned_callable_kwdefault_names");
  if (auto call = callable.getDefiningOp<CallOp>())
    return call->getAttrOfType<ArrayAttr>(
        "lython.returned_callable_kwdefault_names");
  return {};
}

FailureOr<FuncSignatureType>
resolveCallableSignature(Operation *op, Value callable,
                         ArrayAttr &expectedArgNamesAttr,
                         ArrayAttr &expectedKwNamesAttr) {
  Type callableType = callable.getType();
  if (auto funcTy = dyn_cast<FuncType>(callableType)) {
    Value strippedCallable = stripIdentityCasts(callable);
    if (auto funcObject = strippedCallable.getDefiningOp<FuncObjectOp>()) {
      Operation *symbol =
          SymbolTable::lookupNearestSymbolFrom(op, funcObject.getTargetAttr());
      if (auto funcOp = dyn_cast_or_null<FuncOp>(symbol)) {
        expectedArgNamesAttr = funcOp.getArgNamesAttr();
        expectedKwNamesAttr = funcOp.getKwonlyNamesAttr();
      }
    } else if (auto makeFunc =
                   strippedCallable.getDefiningOp<MakeFunctionOp>()) {
      Operation *symbol =
          SymbolTable::lookupNearestSymbolFrom(op, makeFunc.getTargetAttr());
      if (auto funcOp = dyn_cast_or_null<FuncOp>(symbol)) {
        expectedArgNamesAttr = funcOp.getArgNamesAttr();
        expectedKwNamesAttr = funcOp.getKwonlyNamesAttr();
      }
    } else if (auto funcOp =
                   lookupReturnedCallableFuncFromValue(op, callable)) {
      expectedArgNamesAttr = funcOp.getArgNamesAttr();
      expectedKwNamesAttr = funcOp.getKwonlyNamesAttr();
    }
    return funcTy.getSignature();
  }

  op->emitOpError("unexpected callable type");
  return failure();
}

FailureOr<ThrowEffect> resolveCallableThrowEffect(Operation *op,
                                                  Value callable) {
  Type callableType = callable.getType();
  if (auto funcTy = dyn_cast<FuncType>(callableType)) {
    Value strippedCallable = stripIdentityCasts(callable);
    if (auto funcObject = strippedCallable.getDefiningOp<FuncObjectOp>()) {
      Operation *symbol =
          SymbolTable::lookupNearestSymbolFrom(op, funcObject.getTargetAttr());
      if (auto funcOp = dyn_cast_or_null<FuncOp>(symbol))
        return getEffectFromFuncOp(funcOp);
    } else if (auto makeFunc =
                   strippedCallable.getDefiningOp<MakeFunctionOp>()) {
      Operation *symbol =
          SymbolTable::lookupNearestSymbolFrom(op, makeFunc.getTargetAttr());
      if (auto funcOp = dyn_cast_or_null<FuncOp>(symbol))
        return getEffectFromFuncOp(funcOp);
    } else if (auto funcOp =
                   lookupReturnedCallableFuncFromValue(op, callable)) {
      return getEffectFromFuncOp(funcOp);
    }
    return ThrowEffect::MayThrow;
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

unsigned getMinimumPositionalCountForCallable(FuncSignatureType signature,
                                              Value callable) {
  unsigned positionalCount = signature.getPositionalTypes().size();
  if (auto defaultsAttr = getReturnedCallableDefaultsCountAttr(callable)) {
    unsigned defaultsCount =
        static_cast<unsigned>(defaultsAttr.getValue().getZExtValue());
    if (defaultsCount > positionalCount)
      return 0;
    return positionalCount - defaultsCount;
  }
  auto makeFunc = stripIdentityCasts(callable).getDefiningOp<MakeFunctionOp>();
  if (!makeFunc)
    return positionalCount;
  Value defaults = makeFunc.getDefaults();
  if (!defaults)
    return positionalCount;
  auto defaultsTy = dyn_cast<TupleType>(defaults.getType());
  if (!defaultsTy)
    return positionalCount;
  unsigned defaultsCount = defaultsTy.getElementTypes().size();
  if (defaultsCount > positionalCount)
    return 0;
  return positionalCount - defaultsCount;
}

static Value stripIdentityCasts(Value value) {
  while (auto identity = value.getDefiningOp<CastIdentityOp>())
    value = identity.getInput();
  return value;
}

static bool collectStaticDictStringKeys(Value dictValue,
                                        SmallVectorImpl<StringRef> &keys) {
  SmallVector<StringRef> reversed;
  Value current = stripIdentityCasts(dictValue);
  while (auto insert = current.getDefiningOp<DictInsertOp>()) {
    Value key = stripIdentityCasts(insert.getKey());
    auto strConst = key.getDefiningOp<StrConstantOp>();
    if (!strConst)
      return false;
    reversed.push_back(strConst.getValueAttr().getValue());
    current = stripIdentityCasts(insert.getDict());
  }

  if (!current.getDefiningOp<DictEmptyOp>())
    return false;

  keys.assign(reversed.rbegin(), reversed.rend());
  return true;
}

static void collectDefaultedKeywordNames(Value callable,
                                         llvm::StringSet<> &names) {
  if (auto attr = getReturnedCallableKwdefaultNamesAttr(callable)) {
    for (Attribute entry : attr) {
      if (auto strAttr = dyn_cast<StringAttr>(entry))
        names.insert(strAttr.getValue());
    }
  }
  auto makeFunc = stripIdentityCasts(callable).getDefiningOp<MakeFunctionOp>();
  if (!makeFunc || !makeFunc.getKwdefaults())
    return;

  SmallVector<StringRef> keyStorage;
  if (!collectStaticDictStringKeys(makeFunc.getKwdefaults(), keyStorage))
    return;
  for (StringRef key : keyStorage)
    names.insert(key);
}

static bool callableHasKwdefaults(Value callable) {
  if (auto attr = getReturnedCallableKwdefaultNamesAttr(callable))
    return !attr.empty();
  auto makeFunc = stripIdentityCasts(callable).getDefiningOp<MakeFunctionOp>();
  return makeFunc && static_cast<bool>(makeFunc.getKwdefaults());
}

LogicalResult verifyVectorCallOperands(Operation *op,
                                       FuncSignatureType signature,
                                       Value callable, Value posargs,
                                       Value kwnames, Value kwvalues,
                                       ArrayAttr expectedArgNamesAttr,
                                       ArrayAttr expectedKwNamesAttr) {
  auto posTupleOr = requireTuple(op, posargs, "posargs");
  if (failed(posTupleOr))
    return failure();
  TupleType posTuple = *posTupleOr;
  auto posElems = posTuple.getElementTypes();
  bool homogeneous = posElems.size() == 1;

  auto positionalTypes = signature.getPositionalTypes();
  unsigned minPositionalCount =
      getMinimumPositionalCountForCallable(signature, callable);
  if (!signature.hasVararg()) {
    if (posElems.size() > positionalTypes.size())
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
  bool keywordsAccepted =
      signature.hasKwarg() || !kwonlyTypes.empty() ||
      (expectedArgNamesAttr && !expectedArgNamesAttr.empty());
  bool namesEmpty = nameElems.empty();
  if (!keywordsAccepted && !namesEmpty)
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

  if (!keywordsAccepted && !nameValues.empty())
    return op->emitOpError("callee does not accept keyword arguments");
  llvm::StringSet<> defaultedKeywordNames;
  collectDefaultedKeywordNames(callable, defaultedKeywordNames);
  bool hasCallableKwdefaults = callableHasKwdefaults(callable);

  if (hasLiteralNames) {
    llvm::SmallVector<StringRef> positionalNames;
    llvm::StringMap<unsigned> positionalNameToIndex;
    if (expectedArgNamesAttr) {
      positionalNames.reserve(expectedArgNamesAttr.size());
      for (auto [index, attr] : llvm::enumerate(expectedArgNamesAttr)) {
        auto strAttr = dyn_cast<StringAttr>(attr);
        if (!strAttr)
          continue;
        positionalNames.push_back(strAttr.getValue());
        positionalNameToIndex[strAttr.getValue()] = index;
      }
    }

    llvm::SmallVector<StringRef> kwonlyNames;
    llvm::StringMap<unsigned> kwonlyNameToIndex;
    if (expectedKwNamesAttr) {
      kwonlyNames.reserve(expectedKwNamesAttr.size());
      for (auto [index, attr] : llvm::enumerate(expectedKwNamesAttr)) {
        auto strAttr = dyn_cast<StringAttr>(attr);
        if (!strAttr)
          continue;
        kwonlyNames.push_back(strAttr.getValue());
        kwonlyNameToIndex[strAttr.getValue()] = index;
      }
    }

    llvm::SmallSet<StringRef, 8> providedSet;
    for (StringRef providedName : providedKwNames)
      if (!providedSet.insert(providedName).second)
        return op->emitOpError("duplicate keyword '") << providedName << "'";

    if (!signature.hasKwarg()) {
      for (StringRef providedName : providedKwNames)
        if (!positionalNameToIndex.count(providedName) &&
            !kwonlyNameToIndex.count(providedName))
          return op->emitOpError("unexpected keyword argument '")
                 << providedName << "'";
    }

    for (auto [index, name] : llvm::enumerate(positionalNames)) {
      if (index < posElems.size())
        continue;
      if (providedSet.contains(name))
        continue;
      if (index < minPositionalCount)
        return op->emitOpError("missing required argument '") << name << "'";
    }

    for (StringRef expected : kwonlyNames) {
      if (defaultedKeywordNames.contains(expected))
        continue;
      if (providedSet.contains(expected))
        continue;
      if (hasCallableKwdefaults && namesEmpty)
        continue;
      return op->emitOpError("missing keyword argument '") << expected << "'";
    }
  } else if (!kwonlyTypes.empty() && namesEmpty && !hasCallableKwdefaults &&
             defaultedKeywordNames.size() < kwonlyTypes.size()) {
    return op->emitOpError(
        "callee expects keyword arguments but none were provided");
  } else if (!signature.hasVararg() && posElems.size() < minPositionalCount) {
    return op->emitOpError(
        "posargs length mismatch with callee positional parameters");
  }

  return success();
}

} // namespace py
