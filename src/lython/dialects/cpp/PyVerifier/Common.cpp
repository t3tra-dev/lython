#include "cpp/PyVerifier/Common.h"

namespace py {

ClassOp lookupClassSymbol(mlir::Operation *from, ClassType classType) {
  mlir::StringAttr nameAttr =
      mlir::StringAttr::get(from->getContext(), classType.getClassName());
  for (mlir::Operation *symbolTableOp = from; symbolTableOp;
       symbolTableOp = symbolTableOp->getParentOp()) {
    if (!symbolTableOp->hasTrait<mlir::OpTrait::SymbolTable>())
      continue;
    if (mlir::Operation *symbol =
            mlir::SymbolTable::lookupSymbolIn(symbolTableOp, nameAttr))
      if (ClassOp classOp = mlir::dyn_cast<ClassOp>(symbol))
        return classOp;
  }
  return nullptr;
}

FuncOp lookupMethodByName(ClassOp classOp, llvm::StringRef methodName) {
  for (FuncOp method : classOp.getBody().front().getOps<FuncOp>()) {
    if (mlir::StringAttr nameAttr = method.getSymNameAttr())
      if (nameAttr.getValue() == methodName)
        return method;
  }
  return nullptr;
}

mlir::FailureOr<mlir::Type> lookupClassFieldType(mlir::Operation *from,
                                                 ClassType classType,
                                                 llvm::StringRef fieldName) {
  ClassOp classOp = lookupClassSymbol(from, classType);
  if (!classOp) {
    from->emitOpError("unable to resolve class '")
        << classType.getClassName() << "'";
    return mlir::failure();
  }

  mlir::ArrayAttr fieldNames = classOp.getFieldNamesAttr();
  mlir::ArrayAttr fieldTypes = classOp.getFieldTypesAttr();
  if (!fieldNames || !fieldTypes) {
    from->emitOpError("class '") << classType.getClassName()
                                 << "' does not define a static field schema";
    return mlir::failure();
  }

  for (auto [nameAttr, typeAttr] : llvm::zip(fieldNames, fieldTypes)) {
    auto stringAttr = mlir::dyn_cast<mlir::StringAttr>(nameAttr);
    auto mlirTypeAttr = mlir::dyn_cast<mlir::TypeAttr>(typeAttr);
    if (!stringAttr || !mlirTypeAttr) {
      from->emitOpError("class '")
          << classType.getClassName() << "' has malformed field schema";
      return mlir::failure();
    }
    if (stringAttr.getValue() == fieldName)
      return mlirTypeAttr.getValue();
  }

  from->emitOpError("class '")
      << classType.getClassName() << "' has no field '" << fieldName << "'";
  return mlir::failure();
}

static ThrowEffect getEffectFromFuncOp(FuncOp funcOp) {
  if (funcOp->getAttr("nothrow"))
    return ThrowEffect::NoThrow;
  if (funcOp->getAttr("maythrow"))
    return ThrowEffect::MayThrow;
  return ThrowEffect::MayThrow;
}

static mlir::Value stripBridgeCasts(mlir::Value value);

static FuncOp lookupReturnedCallableFuncFromValue(mlir::Operation *op,
                                                  mlir::Value callable) {
  if (auto cast = callable.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
    if (auto symbolAttr = cast->getAttrOfType<mlir::SymbolRefAttr>(
            "ly.returned_callable_symbol")) {
      mlir::Operation *symbol =
          mlir::SymbolTable::lookupNearestSymbolFrom(op, symbolAttr);
      return mlir::dyn_cast_or_null<FuncOp>(symbol);
    }
    if (cast->getNumOperands() == 1)
      return lookupReturnedCallableFuncFromValue(op, cast.getOperand(0));
  }
  if (auto callVector = callable.getDefiningOp<CallVectorOp>()) {
    if (auto symbolAttr = callVector->getAttrOfType<mlir::SymbolRefAttr>(
            "ly.returned_callable_symbol")) {
      mlir::Operation *symbol =
          mlir::SymbolTable::lookupNearestSymbolFrom(op, symbolAttr);
      return mlir::dyn_cast_or_null<FuncOp>(symbol);
    }
  }
  if (auto call = callable.getDefiningOp<CallOp>()) {
    if (auto symbolAttr = call->getAttrOfType<mlir::SymbolRefAttr>(
            "ly.returned_callable_symbol")) {
      mlir::Operation *symbol =
          mlir::SymbolTable::lookupNearestSymbolFrom(op, symbolAttr);
      return mlir::dyn_cast_or_null<FuncOp>(symbol);
    }
  }
  return nullptr;
}

static mlir::IntegerAttr
getReturnedCallableDefaultsCountAttr(mlir::Value callable) {
  if (auto cast = callable.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
    if (auto attr = cast->getAttrOfType<mlir::IntegerAttr>(
            "ly.returned_callable_defaults_count"))
      return attr;
    if (cast->getNumOperands() == 1)
      return getReturnedCallableDefaultsCountAttr(cast.getOperand(0));
  }
  if (auto callVector = callable.getDefiningOp<CallVectorOp>())
    return callVector->getAttrOfType<mlir::IntegerAttr>(
        "ly.returned_callable_defaults_count");
  if (auto call = callable.getDefiningOp<CallOp>())
    return call->getAttrOfType<mlir::IntegerAttr>(
        "ly.returned_callable_defaults_count");
  return {};
}

static mlir::ArrayAttr
getReturnedCallableKwdefaultNamesAttr(mlir::Value callable) {
  if (auto cast = callable.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
    if (auto attr = cast->getAttrOfType<mlir::ArrayAttr>(
            "ly.returned_callable_kwdefault_names"))
      return attr;
    if (cast->getNumOperands() == 1)
      return getReturnedCallableKwdefaultNamesAttr(cast.getOperand(0));
  }
  if (auto callVector = callable.getDefiningOp<CallVectorOp>())
    return callVector->getAttrOfType<mlir::ArrayAttr>(
        "ly.returned_callable_kwdefault_names");
  if (auto call = callable.getDefiningOp<CallOp>())
    return call->getAttrOfType<mlir::ArrayAttr>(
        "ly.returned_callable_kwdefault_names");
  return {};
}

mlir::FailureOr<FuncSignatureType>
resolveCallableSignature(mlir::Operation *op, mlir::Value callable,
                         mlir::ArrayAttr &expectedArgNamesAttr,
                         mlir::ArrayAttr &expectedKwNamesAttr) {
  mlir::Type callableType = callable.getType();
  if (auto funcTy = mlir::dyn_cast<FuncType>(callableType)) {
    mlir::Value strippedCallable = stripBridgeCasts(callable);
    if (auto funcObject = strippedCallable.getDefiningOp<FuncObjectOp>()) {
      mlir::Operation *symbol = mlir::SymbolTable::lookupNearestSymbolFrom(
          op, funcObject.getTargetAttr());
      if (auto funcOp = mlir::dyn_cast_or_null<FuncOp>(symbol)) {
        expectedArgNamesAttr = funcOp.getArgNamesAttr();
        expectedKwNamesAttr = funcOp.getKwonlyNamesAttr();
      }
    } else if (auto makeFunc =
                   strippedCallable.getDefiningOp<MakeFunctionOp>()) {
      mlir::Operation *symbol = mlir::SymbolTable::lookupNearestSymbolFrom(
          op, makeFunc.getTargetAttr());
      if (auto funcOp = mlir::dyn_cast_or_null<FuncOp>(symbol)) {
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
  return mlir::failure();
}

mlir::FailureOr<ThrowEffect> resolveCallableThrowEffect(mlir::Operation *op,
                                                        mlir::Value callable) {
  mlir::Type callableType = callable.getType();
  if (auto funcTy = mlir::dyn_cast<FuncType>(callableType)) {
    mlir::Value strippedCallable = stripBridgeCasts(callable);
    if (auto funcObject = strippedCallable.getDefiningOp<FuncObjectOp>()) {
      mlir::Operation *symbol = mlir::SymbolTable::lookupNearestSymbolFrom(
          op, funcObject.getTargetAttr());
      if (auto funcOp = mlir::dyn_cast_or_null<FuncOp>(symbol))
        return getEffectFromFuncOp(funcOp);
    } else if (auto makeFunc =
                   strippedCallable.getDefiningOp<MakeFunctionOp>()) {
      mlir::Operation *symbol = mlir::SymbolTable::lookupNearestSymbolFrom(
          op, makeFunc.getTargetAttr());
      if (auto funcOp = mlir::dyn_cast_or_null<FuncOp>(symbol))
        return getEffectFromFuncOp(funcOp);
    } else if (auto funcOp =
                   lookupReturnedCallableFuncFromValue(op, callable)) {
      return getEffectFromFuncOp(funcOp);
    }
    return ThrowEffect::MayThrow;
  }

  op->emitOpError("unexpected callable type");
  return mlir::failure();
}

mlir::FailureOr<TupleType>
requireTuple(mlir::Operation *op, mlir::Value operand, llvm::StringRef what) {
  if (auto tupleTy = mlir::dyn_cast<TupleType>(operand.getType()))
    return tupleTy;
  op->emitOpError() << what << " operand must be a !py.tuple value";
  return mlir::failure();
}

unsigned getMinimumPositionalCountForCallable(FuncSignatureType signature,
                                              mlir::Value callable) {
  unsigned positionalCount = signature.getPositionalTypes().size();
  if (auto defaultsAttr = getReturnedCallableDefaultsCountAttr(callable)) {
    unsigned defaultsCount =
        static_cast<unsigned>(defaultsAttr.getValue().getZExtValue());
    if (defaultsCount > positionalCount)
      return 0;
    return positionalCount - defaultsCount;
  }
  auto makeFunc = stripBridgeCasts(callable).getDefiningOp<MakeFunctionOp>();
  if (!makeFunc)
    return positionalCount;
  mlir::Value defaults = makeFunc.getDefaults();
  if (!defaults)
    return positionalCount;
  auto defaultsTy = mlir::dyn_cast<TupleType>(defaults.getType());
  if (!defaultsTy)
    return positionalCount;
  unsigned defaultsCount = defaultsTy.getElementTypes().size();
  if (defaultsCount > positionalCount)
    return 0;
  return positionalCount - defaultsCount;
}

static mlir::Value stripBridgeCasts(mlir::Value value) {
  while (auto cast = value.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
    if (cast->getNumOperands() != 1)
      break;
    value = cast.getOperand(0);
  }
  return value;
}

static bool collectStaticDictStringKeyInserts(
    mlir::Value dictValue, mlir::Operation *beforeOp,
    llvm::SmallVectorImpl<DictInsertOp> &inserts) {
  mlir::Value root = stripBridgeCasts(dictValue);
  if (!root.getDefiningOp<DictEmptyOp>())
    return false;

  mlir::Block *block = beforeOp ? beforeOp->getBlock() : nullptr;
  for (mlir::Operation *user : root.getUsers()) {
    auto insert = mlir::dyn_cast<DictInsertOp>(user);
    if (!insert)
      continue;
    if (block && insert->getBlock() != block)
      return false;
    if (beforeOp && !insert->isBeforeInBlock(beforeOp))
      continue;
    inserts.push_back(insert);
  }

  llvm::sort(inserts, [](DictInsertOp lhs, DictInsertOp rhs) {
    return lhs->isBeforeInBlock(rhs);
  });
  return true;
}

static bool
collectStaticDictStringKeys(mlir::Value dictValue, mlir::Operation *beforeOp,
                            llvm::SmallVectorImpl<llvm::StringRef> &keys) {
  llvm::SmallVector<DictInsertOp, 8> inserts;
  if (!collectStaticDictStringKeyInserts(dictValue, beforeOp, inserts))
    return false;

  for (DictInsertOp insert : inserts) {
    mlir::Value key = stripBridgeCasts(insert.getKey());
    auto strConst = key.getDefiningOp<StrConstantOp>();
    if (!strConst)
      return false;
    keys.push_back(strConst.getValueAttr().getValue());
  }
  return true;
}

static void collectDefaultedKeywordNames(mlir::Value callable,
                                         llvm::StringSet<> &names) {
  if (auto attr = getReturnedCallableKwdefaultNamesAttr(callable)) {
    for (mlir::Attribute entry : attr) {
      if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(entry))
        names.insert(strAttr.getValue());
    }
  }
  auto makeFunc = stripBridgeCasts(callable).getDefiningOp<MakeFunctionOp>();
  if (!makeFunc || !makeFunc.getKwdefaults())
    return;

  llvm::SmallVector<llvm::StringRef> keyStorage;
  if (!collectStaticDictStringKeys(makeFunc.getKwdefaults(),
                                   makeFunc.getOperation(), keyStorage))
    return;
  for (llvm::StringRef key : keyStorage)
    names.insert(key);
}

static bool callableHasKwdefaults(mlir::Value callable) {
  if (auto attr = getReturnedCallableKwdefaultNamesAttr(callable))
    return !attr.empty();
  auto makeFunc = stripBridgeCasts(callable).getDefiningOp<MakeFunctionOp>();
  return makeFunc && static_cast<bool>(makeFunc.getKwdefaults());
}

mlir::LogicalResult verifyVectorCallOperands(
    mlir::Operation *op, FuncSignatureType signature, mlir::Value callable,
    mlir::Value posargs, mlir::Value kwnames, mlir::Value kwvalues,
    mlir::ArrayAttr expectedArgNamesAttr, mlir::ArrayAttr expectedKwNamesAttr) {
  auto posTupleOr = requireTuple(op, posargs, "posargs");
  if (mlir::failed(posTupleOr))
    return mlir::failure();
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
    auto calleeVarargTy = mlir::dyn_cast<TupleType>(signature.getVarargType());
    if (!calleeVarargTy || calleeVarargTy.getElementTypes().size() != 1)
      return op->emitOpError(
          "vararg parameter must be !py.tuple<T> with single element type");
    mlir::Type calleeElemType = calleeVarargTy.getElementTypes().front();

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
  if (mlir::failed(nameTupleOr))
    return mlir::failure();
  auto valueTupleOr = requireTuple(op, kwvalues, "kwvalues");
  if (mlir::failed(valueTupleOr))
    return mlir::failure();

  TupleType nameTuple = *nameTupleOr;
  TupleType valueTuple = *valueTupleOr;
  auto nameElems = nameTuple.getElementTypes();
  for (mlir::Type elem : nameElems)
    if (!isPyStrType(elem))
      return op->emitOpError("kwnames tuple elements must be !py.str");

  auto valueElems = valueTuple.getElementTypes();
  for (mlir::Type elem : valueElems)
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

  auto collectTupleValues =
      [&](mlir::Value tupleValue,
          llvm::SmallVectorImpl<mlir::Value> &values) -> bool {
    if (auto create = tupleValue.getDefiningOp<TupleCreateOp>()) {
      values.append(create.getElements().begin(), create.getElements().end());
      return true;
    }
    if (tupleValue.getDefiningOp<TupleEmptyOp>())
      return true;
    return false;
  };

  llvm::SmallVector<mlir::Value> nameValues;
  bool hasLiteralNames = collectTupleValues(kwnames, nameValues);
  llvm::SmallVector<llvm::StringRef> providedKwNames;
  if (hasLiteralNames) {
    for (mlir::Value nameVal : nameValues) {
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
    llvm::SmallVector<llvm::StringRef> positionalNames;
    llvm::StringMap<unsigned> positionalNameToIndex;
    if (expectedArgNamesAttr) {
      positionalNames.reserve(expectedArgNamesAttr.size());
      for (auto [index, attr] : llvm::enumerate(expectedArgNamesAttr)) {
        auto strAttr = mlir::dyn_cast<mlir::StringAttr>(attr);
        if (!strAttr)
          continue;
        positionalNames.push_back(strAttr.getValue());
        positionalNameToIndex[strAttr.getValue()] = index;
      }
    }

    llvm::SmallVector<llvm::StringRef> kwonlyNames;
    llvm::StringMap<unsigned> kwonlyNameToIndex;
    if (expectedKwNamesAttr) {
      kwonlyNames.reserve(expectedKwNamesAttr.size());
      for (auto [index, attr] : llvm::enumerate(expectedKwNamesAttr)) {
        auto strAttr = mlir::dyn_cast<mlir::StringAttr>(attr);
        if (!strAttr)
          continue;
        kwonlyNames.push_back(strAttr.getValue());
        kwonlyNameToIndex[strAttr.getValue()] = index;
      }
    }

    llvm::SmallSet<llvm::StringRef, 8> providedSet;
    for (llvm::StringRef providedName : providedKwNames)
      if (!providedSet.insert(providedName).second)
        return op->emitOpError("duplicate keyword '") << providedName << "'";

    if (!signature.hasKwarg()) {
      for (llvm::StringRef providedName : providedKwNames)
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

    for (llvm::StringRef expected : kwonlyNames) {
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

  return mlir::success();
}

} // namespace py
