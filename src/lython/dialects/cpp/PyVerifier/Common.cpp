#include "cpp/PyVerifier/Common.h"

#include "cpp/PyTypeObject.h"

#include "mlir/Dialect/Async/IR/Async.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"

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

CallableFuncOp lookupMethodByName(ClassOp classOp, llvm::StringRef methodName) {
  auto mro =
      type_object::mroNames(classOp, classOp.getSymNameAttr().getValue());
  if (mlir::failed(mro))
    return nullptr;
  for (llvm::StringRef className : *mro) {
    ClassOp owner = type_object::lookup(classOp, className);
    if (!owner || owner.getBody().empty())
      continue;
    for (CallableFuncOp method :
         owner.getBody().front().getOps<CallableFuncOp>()) {
      if (mlir::StringAttr nameAttr = method.getSymNameAttr())
        if (nameAttr.getValue() == methodName)
          return method;
    }
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

  auto mro = type_object::mroNames(from, classType.getClassName());
  if (mlir::failed(mro))
    return mlir::failure();
  for (llvm::StringRef ownerName : *mro) {
    ClassOp owner = type_object::lookup(from, ownerName);
    if (!owner)
      continue;
    mlir::ArrayAttr fieldNames = owner.getFieldNamesAttr();
    mlir::ArrayAttr fieldTypes = owner.getFieldTypesAttr();
    if (!fieldNames && !fieldTypes)
      continue;
    if (!fieldNames || !fieldTypes || fieldNames.size() != fieldTypes.size()) {
      from->emitOpError("class '")
          << ownerName << "' has malformed field schema";
      return mlir::failure();
    }

    for (auto [nameAttr, typeAttr] : llvm::zip(fieldNames, fieldTypes)) {
      auto stringAttr = mlir::dyn_cast<mlir::StringAttr>(nameAttr);
      auto mlirTypeAttr = mlir::dyn_cast<mlir::TypeAttr>(typeAttr);
      if (!stringAttr || !mlirTypeAttr) {
        from->emitOpError("class '")
            << ownerName << "' has malformed field schema";
        return mlir::failure();
      }
      if (stringAttr.getValue() == fieldName)
        return mlirTypeAttr.getValue();
    }
  }

  from->emitOpError("class '")
      << classType.getClassName() << "' has no field '" << fieldName << "'";
  return mlir::failure();
}

static ThrowEffect getEffectFromFuncOp(CallableFuncOp funcOp) {
  if (funcOp->getAttr("nothrow"))
    return ThrowEffect::NoThrow;
  if (funcOp->getAttr("maythrow"))
    return ThrowEffect::MayThrow;
  return ThrowEffect::MayThrow;
}

static ThrowEffect getEffectFromSymbol(mlir::Operation *symbol) {
  if (!symbol)
    return ThrowEffect::MayThrow;
  if (symbol->getAttr("nothrow"))
    return ThrowEffect::NoThrow;
  if (symbol->getAttr("maythrow"))
    return ThrowEffect::MayThrow;
  return ThrowEffect::MayThrow;
}

static mlir::Value stripBridgeCasts(mlir::Value value);

static mlir::LogicalResult
collectClosureTypes(mlir::Operation *op, mlir::ArrayAttr closureTypesAttr,
                    llvm::SmallVectorImpl<mlir::Type> &closureTypes) {
  if (!closureTypesAttr)
    return mlir::success();
  closureTypes.reserve(closureTypes.size() + closureTypesAttr.size());
  for (mlir::Attribute attr : closureTypesAttr) {
    auto typeAttr = mlir::dyn_cast<mlir::TypeAttr>(attr);
    if (!typeAttr)
      return op->emitOpError(
          "closure_types must contain only TypeAttr elements");
    closureTypes.push_back(typeAttr.getValue());
  }
  return mlir::success();
}

static mlir::FailureOr<CallableType>
callableTypeForAsyncFunc(mlir::Operation *op, mlir::async::FuncOp asyncFunc) {
  mlir::FunctionType asyncType = asyncFunc.getFunctionType();
  if (asyncType.getNumInputs() == 0 || asyncType.getNumResults() != 1) {
    op->emitOpError("async target must carry an exception cell and one async "
                    "result");
    return mlir::failure();
  }
  auto asyncResult =
      mlir::dyn_cast<mlir::async::ValueType>(asyncType.getResult(0));
  if (!asyncResult) {
    op->emitOpError("async target result must be !async.value");
    return mlir::failure();
  }
  mlir::MLIRContext *ctx = op->getContext();
  mlir::Type coroutineResult = ProtocolType::get(
      ctx, "Coroutine",
      {ObjectType::get(ctx), ObjectType::get(ctx), asyncResult.getValueType()});
  llvm::SmallVector<mlir::Type> publicInputs(
      asyncType.getInputs().drop_back().begin(),
      asyncType.getInputs().drop_back().end());
  return CallableType::get(ctx, publicInputs, {}, {}, {}, {coroutineResult});
}

static mlir::FailureOr<CallableType>
callableTypeForTargetSymbol(mlir::Operation *op, mlir::Operation *symbol) {
  if (auto pyFunc = mlir::dyn_cast_or_null<CallableFuncOp>(symbol))
    return mlir::cast<CallableType>(pyFunc.getFunctionTypeAttr().getValue());
  if (auto asyncFunc = mlir::dyn_cast_or_null<mlir::async::FuncOp>(symbol))
    return callableTypeForAsyncFunc(op, asyncFunc);
  if (auto func = mlir::dyn_cast_or_null<mlir::func::FuncOp>(symbol)) {
    if (auto typeAttr = func->getAttrOfType<mlir::TypeAttr>("function_type"))
      if (auto callable = mlir::dyn_cast<CallableType>(typeAttr.getValue()))
        return callable;
    return CallableType{};
  }
  return CallableType{};
}

static mlir::FlatSymbolRefAttr
getReturnedCallableSymbolAttr(mlir::Value callable) {
  if (auto cast = callable.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
    if (auto attr = cast->getAttrOfType<mlir::FlatSymbolRefAttr>(
            "ly.returned_callable_symbol"))
      return attr;
    if (cast->getNumOperands() == 1)
      return getReturnedCallableSymbolAttr(cast.getOperand(0));
  }
  if (auto call = callable.getDefiningOp<CallOp>())
    return call->getAttrOfType<mlir::FlatSymbolRefAttr>(
        "ly.returned_callable_symbol");
  return {};
}

static CallableFuncOp
lookupReturnedCallableFuncFromValue(mlir::Operation *op, mlir::Value callable) {
  if (auto symbolAttr = getReturnedCallableSymbolAttr(callable))
    return mlir::dyn_cast_or_null<CallableFuncOp>(
        mlir::SymbolTable::lookupNearestSymbolFrom(op, symbolAttr));
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
  if (auto call = callable.getDefiningOp<CallOp>())
    return call->getAttrOfType<mlir::ArrayAttr>(
        "ly.returned_callable_kwdefault_names");
  return {};
}

mlir::FailureOr<TupleType>
requireTuple(mlir::Operation *op, mlir::Value operand, llvm::StringRef what) {
  if (auto tupleTy = mlir::dyn_cast<TupleType>(operand.getType()))
    return tupleTy;
  op->emitOpError() << what << " operand must be a !py.tuple value";
  return mlir::failure();
}

unsigned
getMinimumPositionalCountForCallable(const CallableEvidence &evidence) {
  CallableType signature = evidence.signature;
  unsigned positionalCount = signature.getPositionalTypes().size();
  if (auto defaultsAttr = evidence.returnedCallableDefaultsCount) {
    unsigned defaultsCount =
        static_cast<unsigned>(defaultsAttr.getValue().getZExtValue());
    if (defaultsCount > positionalCount)
      return 0;
    return positionalCount - defaultsCount;
  }
  if (!evidence.defaults)
    return positionalCount;
  auto defaultsTy = mlir::dyn_cast<TupleType>(evidence.defaults.getType());
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

static void collectDefaultedKeywordNames(CallableEvidence &evidence,
                                         llvm::StringSet<> &names) {
  if (auto attr = evidence.returnedCallableKwdefaultNames) {
    for (mlir::Attribute entry : attr) {
      if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(entry))
        names.insert(strAttr.getValue());
    }
  }
  if (!evidence.kwdefaults)
    return;

  llvm::SmallVector<llvm::StringRef> keyStorage;
  if (!collectStaticDictStringKeys(evidence.kwdefaults,
                                   evidence.materializer.getOperation(),
                                   keyStorage))
    return;
  for (llvm::StringRef key : keyStorage)
    names.insert(key);
}

static bool callableHasKwdefaults(const CallableEvidence &evidence) {
  if (auto attr = evidence.returnedCallableKwdefaultNames)
    return !attr.empty();
  return static_cast<bool>(evidence.kwdefaults);
}

static llvm::SmallVector<mlir::Type, 4>
getCallableObjectClosureTypes(mlir::Operation *op, mlir::Value callable) {
  auto funcObject =
      stripBridgeCasts(callable).getDefiningOp<CallableObjectOp>();
  if (!funcObject)
    return {};
  mlir::Operation *symbol = mlir::SymbolTable::lookupNearestSymbolFrom(
      op, funcObject.getTargetAttr());
  auto funcOp = mlir::dyn_cast_or_null<CallableFuncOp>(symbol);
  if (!funcOp)
    return {};
  mlir::ArrayAttr closureTypesAttr = funcOp.getClosureTypesAttr();
  if (!closureTypesAttr)
    return {};
  llvm::SmallVector<mlir::Type, 4> closureTypes;
  if (mlir::failed(collectClosureTypes(op, closureTypesAttr, closureTypes)))
    return {};
  return closureTypes;
}

static CallableFuncOp lookupCallableFuncFromValue(mlir::Operation *op,
                                                  mlir::Value callable) {
  mlir::Value strippedCallable = stripBridgeCasts(callable);
  if (auto funcObject = strippedCallable.getDefiningOp<CallableObjectOp>()) {
    mlir::Operation *symbol = mlir::SymbolTable::lookupNearestSymbolFrom(
        op, funcObject.getTargetAttr());
    return mlir::dyn_cast_or_null<CallableFuncOp>(symbol);
  }
  if (auto makeFunc = strippedCallable.getDefiningOp<MakeFunctionOp>()) {
    mlir::Operation *symbol = mlir::SymbolTable::lookupNearestSymbolFrom(
        op, makeFunc.getTargetAttr());
    return mlir::dyn_cast_or_null<CallableFuncOp>(symbol);
  }
  return lookupReturnedCallableFuncFromValue(op, callable);
}

mlir::FailureOr<CallableEvidence>
resolveCallableEvidence(mlir::Operation *op, mlir::Value callable) {
  CallableType signature = getCallableContract(callable.getType());
  if (!signature) {
    op->emitOpError("unexpected callable type");
    return mlir::failure();
  }

  CallableEvidence evidence;
  evidence.signature = signature;
  if (auto makeFunc =
          stripBridgeCasts(callable).getDefiningOp<MakeFunctionOp>()) {
    evidence.materializer = makeFunc;
    evidence.defaults = makeFunc.getDefaults();
    evidence.kwdefaults = makeFunc.getKwdefaults();
    evidence.closure = makeFunc.getClosure();
  }
  evidence.target = lookupCallableFuncFromValue(op, callable);
  if (evidence.target) {
    evidence.effect = getEffectFromFuncOp(evidence.target);
    evidence.argNames = evidence.target.getArgNamesAttr();
    evidence.kwonlyNames = evidence.target.getKwonlyNamesAttr();
  }
  evidence.closureTypes = getCallableObjectClosureTypes(op, callable);
  evidence.returnedCallable = lookupReturnedCallableFuncFromValue(op, callable);
  evidence.returnedCallableSymbol = getReturnedCallableSymbolAttr(callable);
  evidence.returnedCallableDefaultsCount =
      getReturnedCallableDefaultsCountAttr(callable);
  evidence.returnedCallableKwdefaultNames =
      getReturnedCallableKwdefaultNamesAttr(callable);
  evidence.minPositionalCount = getMinimumPositionalCountForCallable(evidence);
  collectDefaultedKeywordNames(evidence, evidence.defaultedKeywordNames);
  evidence.hasKwdefaults = callableHasKwdefaults(evidence);
  return evidence;
}

mlir::FailureOr<CallableEvidence>
resolveMakeFunctionEvidence(MakeFunctionOp op) {
  CallableType resultSig = getCallableContract(op.getResult().getType());
  if (!resultSig) {
    op.emitOpError("result must be a Callable contract");
    return mlir::failure();
  }

  mlir::Operation *symbol = mlir::SymbolTable::lookupNearestSymbolFrom(
      op.getOperation(), op.getTargetAttr());
  CallableFuncOp pyFunc = mlir::dyn_cast_or_null<CallableFuncOp>(symbol);
  if (!pyFunc) {
    op.emitOpError("target must reference a py.callable.func symbol");
    return mlir::failure();
  }

  CallableType expectedSig =
      mlir::cast<CallableType>(pyFunc.getFunctionTypeAttr().getValue());
  if (expectedSig != resultSig) {
    op.emitOpError(
        "result type signature must match referenced py.callable.func");
    return mlir::failure();
  }

  CallableEvidence evidence;
  evidence.signature = resultSig;
  evidence.target = pyFunc;
  evidence.effect = getEffectFromFuncOp(pyFunc);
  evidence.argNames = pyFunc.getArgNamesAttr();
  evidence.kwonlyNames = pyFunc.getKwonlyNamesAttr();
  evidence.materializer = op;
  evidence.defaults = op.getDefaults();
  evidence.kwdefaults = op.getKwdefaults();
  evidence.closure = op.getClosure();
  if (mlir::failed(collectClosureTypes(op.getOperation(),
                                       pyFunc.getClosureTypesAttr(),
                                       evidence.closureTypes)))
    return mlir::failure();
  evidence.minPositionalCount = getMinimumPositionalCountForCallable(evidence);
  collectDefaultedKeywordNames(evidence, evidence.defaultedKeywordNames);
  evidence.hasKwdefaults = callableHasKwdefaults(evidence);
  return evidence;
}

mlir::LogicalResult verifyMakeFunctionEvidence(MakeFunctionOp op) {
  mlir::FailureOr<CallableEvidence> evidenceOr =
      resolveMakeFunctionEvidence(op);
  if (mlir::failed(evidenceOr))
    return mlir::failure();
  CallableEvidence &evidence = *evidenceOr;
  CallableType signature = evidence.signature;

  if (mlir::Value defaults = evidence.defaults) {
    auto tupleTy = mlir::dyn_cast<TupleType>(defaults.getType());
    if (!tupleTy)
      return op.emitOpError("defaults must be a !py.tuple value");
    auto positionalTypes = signature.getPositionalTypes();
    auto defaultTypes = tupleTy.getElementTypes();
    if (defaultTypes.size() > positionalTypes.size())
      return op.emitOpError(
          "defaults length must not exceed positional parameter count");
    unsigned start = positionalTypes.size() - defaultTypes.size();
    for (auto [idx, elemType] : llvm::enumerate(defaultTypes)) {
      mlir::Type expected = positionalTypes[start + idx];
      if (!isSubtypeOf(elemType, expected, op.getOperation()))
        return op.emitOpError("default value type does not match positional "
                              "parameter type");
    }
  }

  if (mlir::Value kwdefaults = evidence.kwdefaults) {
    auto dictTy = mlir::dyn_cast<DictType>(kwdefaults.getType());
    if (!dictTy)
      return op.emitOpError("kwdefaults must be a !py.dict value");
    if (!isSubtypeOf(dictTy.getKeyType(), StrType::get(op.getContext()),
                     op.getOperation()))
      return op.emitOpError("kwdefaults keys must be compatible with !py.str");

    auto kwonlyTypes = signature.getKwOnlyTypes();
    if (!evidence.kwonlyNames ||
        evidence.kwonlyNames.size() != kwonlyTypes.size())
      return op.emitOpError("target keyword-only parameters must define "
                            "kwonly_names for kwdefaults verification");

    llvm::StringMap<mlir::Type> expectedByName;
    for (auto [nameAttr, type] : llvm::zip(evidence.kwonlyNames, kwonlyTypes)) {
      auto strAttr = mlir::dyn_cast<mlir::StringAttr>(nameAttr);
      if (!strAttr)
        return op.emitOpError(
            "kwonly_names must contain only StringAttr elements");
      expectedByName[strAttr.getValue()] = type;
    }

    llvm::SmallVector<DictInsertOp> inserts;
    if (collectStaticDictStringKeyInserts(kwdefaults, op.getOperation(),
                                          inserts) &&
        !inserts.empty()) {
      llvm::StringSet<> seen;
      auto stripStoredValueType = [](mlir::Value value) -> mlir::Type {
        while (true) {
          if (auto cast =
                  value.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
            if (cast->getNumOperands() != 1)
              return value.getType();
            value = cast.getOperand(0);
            continue;
          }
          return value.getType();
        }
      };
      for (DictInsertOp insert : inserts) {
        mlir::Value keyValue = stripBridgeCasts(insert.getKey());
        auto key = keyValue.getDefiningOp<StrConstantOp>();
        if (!key)
          return op.emitOpError("statically provided kwdefaults keys must be "
                                "py.str.constant values");
        llvm::StringRef keyName = key.getValueAttr().getValue();
        auto it = expectedByName.find(keyName);
        if (it == expectedByName.end())
          return op.emitOpError("kwdefaults key '")
                 << keyName << "' is not a keyword-only parameter";
        if (!seen.insert(keyName).second)
          return op.emitOpError("kwdefaults key '")
                 << keyName << "' appears more than once";
        if (!isSubtypeOf(stripStoredValueType(insert.getValue()), it->second,
                         op.getOperation()))
          return op.emitOpError("kwdefaults value type for key '")
                 << keyName
                 << "' does not match the target keyword-only parameter type";
      }
    }
  }

  if (mlir::Value closure = evidence.closure) {
    auto tupleTy = mlir::dyn_cast<TupleType>(closure.getType());
    if (!tupleTy)
      return op.emitOpError("closure must be a !py.tuple value");
    auto elems = tupleTy.getElementTypes();
    if (elems.size() != evidence.closureTypes.size())
      return op.emitOpError(
          "closure tuple length must match target closure_types");
    for (auto [elemType, expected] : llvm::zip(elems, evidence.closureTypes))
      if (!isSubtypeOf(elemType, expected, op.getOperation()))
        return op.emitOpError(
            "closure element type does not match target closure_types");
  } else if (!evidence.closureTypes.empty()) {
    return op.emitOpError(
        "target requires closure operand matching closure_types");
  }

  if (mlir::Value annotations = op.getAnnotations()) {
    auto dictTy = mlir::dyn_cast<DictType>(annotations.getType());
    if (!dictTy)
      return op.emitOpError("annotations must be a !py.dict value");
    if (!isSubtypeOf(dictTy.getKeyType(), StrType::get(op.getContext()),
                     op.getOperation()))
      return op.emitOpError("annotations keys must be compatible with !py.str");
  }

  if (mlir::Value module = op.getModule())
    if (!isSubtypeOf(module.getType(), StrType::get(op.getContext()),
                     op.getOperation()))
      return op.emitOpError("module metadata must be compatible with !py.str");

  return mlir::success();
}

mlir::FailureOr<SpecialMethodEvidence> resolveSpecialMethodEvidence(
    mlir::Operation *op, mlir::FlatSymbolRefAttr target, mlir::Type calleeType,
    mlir::StringRef methodName) {
  if (!target) {
    op->emitOpError(methodName) << " requires target";
    return mlir::failure();
  }
  CallableType signature = getCallableContract(calleeType);
  if (!signature) {
    op->emitOpError(methodName) << " callee_type must be Callable";
    return mlir::failure();
  }

  mlir::Operation *symbol =
      mlir::SymbolTable::lookupNearestSymbolFrom(op, target);
  if (!symbol) {
    op->emitOpError(methodName)
        << " target must reference an existing callable symbol";
    return mlir::failure();
  }

  if (!mlir::isa<CallableFuncOp, mlir::func::FuncOp, mlir::async::FuncOp>(
          symbol)) {
    op->emitOpError(methodName)
        << " target must reference py.callable.func, func.func, or async.func";
    return mlir::failure();
  }

  mlir::FailureOr<CallableType> targetSignatureOr =
      callableTypeForTargetSymbol(op, symbol);
  if (mlir::failed(targetSignatureOr))
    return mlir::failure();
  if (*targetSignatureOr && *targetSignatureOr != signature) {
    op->emitOpError(methodName)
        << " callee_type does not match target callable contract";
    return mlir::failure();
  }

  SpecialMethodEvidence evidence;
  evidence.signature = signature;
  evidence.target = symbol;
  evidence.effect = getEffectFromSymbol(symbol);
  if (auto pyFunc = mlir::dyn_cast<CallableFuncOp>(symbol)) {
    evidence.argNames = pyFunc.getArgNamesAttr();
    evidence.kwonlyNames = pyFunc.getKwonlyNamesAttr();
    if (mlir::failed(collectClosureTypes(op, pyFunc.getClosureTypesAttr(),
                                         evidence.closureTypes)))
      return mlir::failure();
  } else {
    evidence.argNames = symbol->getAttrOfType<mlir::ArrayAttr>("arg_names");
    evidence.kwonlyNames =
        symbol->getAttrOfType<mlir::ArrayAttr>("kwonly_names");
    if (mlir::failed(collectClosureTypes(
            op, symbol->getAttrOfType<mlir::ArrayAttr>("closure_types"),
            evidence.closureTypes)))
      return mlir::failure();
  }
  return evidence;
}

mlir::LogicalResult verifySpecialMethodOperands(
    mlir::Operation *op, const SpecialMethodEvidence &evidence,
    mlir::ValueRange operands, std::optional<mlir::Type> resultType,
    mlir::StringRef methodName) {
  llvm::SmallVector<mlir::Type, 4> operandTypes;
  operandTypes.reserve(operands.size());
  for (mlir::Value operand : operands)
    operandTypes.push_back(operand.getType());
  return verifySpecialMethodOperandTypes(op, evidence, operandTypes, resultType,
                                         methodName);
}

mlir::LogicalResult verifySpecialMethodOperandTypes(
    mlir::Operation *op, const SpecialMethodEvidence &evidence,
    mlir::TypeRange operandTypes, std::optional<mlir::Type> resultType,
    mlir::StringRef methodName) {
  CallableType signature = evidence.signature;
  if (signature.hasVararg() || signature.hasKwarg() ||
      !signature.getKwOnlyTypes().empty())
    return op->emitOpError(methodName)
           << " callee must have a fixed positional-only lowering signature";
  if (!evidence.closureTypes.empty())
    return op->emitOpError(methodName)
           << " special-method op cannot represent target closure_types";
  if (signature.getPositionalTypes().size() != operandTypes.size())
    return op->emitOpError(methodName)
           << " callee expects " << operandTypes.size()
           << " positional operands including self, got "
           << signature.getPositionalTypes().size();
  for (auto [actual, expected] :
       llvm::zip(operandTypes, signature.getPositionalTypes())) {
    if (!isSubtypeOf(actual, expected, op))
      return op->emitOpError(methodName)
             << " operand type " << actual
             << " does not match callee parameter type " << expected;
  }
  if (signature.getResultTypes().size() != 1)
    return op->emitOpError(methodName)
           << " callee must return exactly one result";
  if (resultType && signature.getResultTypes().front() != *resultType)
    return op->emitOpError(methodName)
           << " result type " << *resultType << " does not match callee result "
           << signature.getResultTypes().front();
  return mlir::success();
}

mlir::LogicalResult verifyCallOperands(mlir::Operation *op,
                                       const CallableEvidence &evidence,
                                       mlir::Value posargs, mlir::Value kwnames,
                                       mlir::Value kwvalues) {
  auto posTupleOr = requireTuple(op, posargs, "posargs");
  if (mlir::failed(posTupleOr))
    return mlir::failure();
  TupleType posTuple = *posTupleOr;
  auto posElems = posTuple.getElementTypes();
  bool homogeneous = posElems.size() == 1;

  CallableType signature = evidence.signature;
  auto positionalTypes = signature.getPositionalTypes();
  auto kwonlyTypes = signature.getKwOnlyTypes();
  llvm::ArrayRef<mlir::Type> closureTypes = evidence.closureTypes;
  unsigned minPositionalCount = evidence.minPositionalCount;
  unsigned normalizedKwonlyCount = 0;
  if (!signature.hasVararg()) {
    unsigned visiblePosargLimit = positionalTypes.size() + kwonlyTypes.size();
    unsigned normalizedPosargLimit = visiblePosargLimit + closureTypes.size();
    if (posElems.size() > normalizedPosargLimit)
      return op->emitOpError(
          "posargs length mismatch with callee positional parameters");
    for (size_t index = 0;
         index < posElems.size() && index < positionalTypes.size(); ++index)
      if (!isSubtypeOf(posElems[index], positionalTypes[index], op))
        return op->emitOpError(
            "posargs element type does not match positional parameter type");
    unsigned visiblePosargs = std::min<unsigned>(
        static_cast<unsigned>(posElems.size()), visiblePosargLimit);
    if (visiblePosargs > positionalTypes.size()) {
      normalizedKwonlyCount = visiblePosargs - positionalTypes.size();
      for (size_t index = 0; index < normalizedKwonlyCount; ++index) {
        if (!isSubtypeOf(posElems[positionalTypes.size() + index],
                         kwonlyTypes[index], op))
          return op->emitOpError("normalized keyword-only argument type "
                                 "does not match parameter type");
      }
    }
    if (posElems.size() > visiblePosargLimit) {
      unsigned closureCount =
          static_cast<unsigned>(posElems.size() - visiblePosargLimit);
      for (unsigned index = 0; index < closureCount; ++index) {
        if (!isSubtypeOf(posElems[visiblePosargLimit + index],
                         closureTypes[index], op))
          return op->emitOpError("closure argument type does not match "
                                 "target closure_types");
      }
    }
  } else {
    auto calleeVarargTy = mlir::dyn_cast<TupleType>(signature.getVarargType());
    if (!calleeVarargTy)
      return op->emitOpError("vararg parameter must be !py.tuple");
    auto varargTypes = calleeVarargTy.getElementTypes();
    bool homogeneousVararg = varargTypes.size() == 1;
    llvm::ArrayRef<mlir::Type> visiblePosElems = posElems;
    if (!closureTypes.empty()) {
      if (posElems.size() < closureTypes.size())
        return op->emitOpError(
            "posargs length shorter than target closure_types");
      unsigned closureStart =
          static_cast<unsigned>(posElems.size() - closureTypes.size());
      for (auto [index, expected] : llvm::enumerate(closureTypes)) {
        if (!isSubtypeOf(posElems[closureStart + index], expected, op))
          return op->emitOpError("closure argument type does not match "
                                 "target closure_types");
      }
      visiblePosElems = posElems.take_front(closureStart);
    }

    if (homogeneousVararg && homogeneous && !visiblePosElems.empty() &&
        positionalTypes.empty()) {
      if (!isSubtypeOf(visiblePosElems.front(), varargTypes.front(), op))
        return op->emitOpError(
            "posargs element type is incompatible with vararg element type");
    } else {
      if (visiblePosElems.size() < positionalTypes.size())
        return op->emitOpError(
            "posargs length shorter than positional parameter count");
      for (size_t i = 0; i < positionalTypes.size(); ++i)
        if (!isSubtypeOf(visiblePosElems[i], positionalTypes[i], op))
          return op->emitOpError(
                     "posargs element type does not match positional "
                     "parameter type at index ")
                 << i;
      std::size_t suppliedVarargs =
          visiblePosElems.size() - positionalTypes.size();
      if (!homogeneousVararg && suppliedVarargs != varargTypes.size())
        return op->emitOpError(
            "posargs length mismatch with exact vararg tuple");
      for (size_t i = positionalTypes.size(); i < visiblePosElems.size(); ++i) {
        mlir::Type expected = homogeneousVararg
                                  ? varargTypes.front()
                                  : varargTypes[i - positionalTypes.size()];
        if (!isSubtypeOf(visiblePosElems[i], expected, op))
          return op->emitOpError("vararg element type incompatible at index ")
                 << i;
      }
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
  if (nameElems.size() != valueElems.size())
    return op->emitOpError(
        "kwnames and kwvalues must describe the same number of entries");

  bool keywordsAccepted = signature.hasKwarg() || !kwonlyTypes.empty() ||
                          (evidence.argNames && !evidence.argNames.empty()) ||
                          !signature.getPositionalNames().empty();
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
  const llvm::StringSet<> &defaultedKeywordNames =
      evidence.defaultedKeywordNames;
  bool hasCallableKwdefaults = evidence.hasKwdefaults;

  if (hasLiteralNames) {
    llvm::SmallVector<llvm::StringRef> positionalNames;
    llvm::StringMap<unsigned> positionalNameToIndex;
    if (evidence.argNames && !evidence.argNames.empty()) {
      positionalNames.reserve(evidence.argNames.size());
      for (auto [index, attr] : llvm::enumerate(evidence.argNames)) {
        auto strAttr = mlir::dyn_cast<mlir::StringAttr>(attr);
        if (!strAttr)
          continue;
        positionalNames.push_back(strAttr.getValue());
        if (index >= signature.getPositionalOnlyCount() &&
            !strAttr.getValue().empty())
          positionalNameToIndex[strAttr.getValue()] = index;
      }
    } else {
      positionalNames.reserve(signature.getPositionalNames().size());
      for (auto [index, attr] :
           llvm::enumerate(signature.getPositionalNames())) {
        positionalNames.push_back(attr.getValue());
        if (index >= signature.getPositionalOnlyCount() &&
            !attr.getValue().empty())
          positionalNameToIndex[attr.getValue()] = index;
      }
    }

    llvm::SmallVector<llvm::StringRef> kwonlyNames;
    llvm::StringMap<unsigned> kwonlyNameToIndex;
    if (evidence.kwonlyNames && !evidence.kwonlyNames.empty()) {
      kwonlyNames.reserve(evidence.kwonlyNames.size());
      for (auto [index, attr] : llvm::enumerate(evidence.kwonlyNames)) {
        auto strAttr = mlir::dyn_cast<mlir::StringAttr>(attr);
        if (!strAttr)
          continue;
        kwonlyNames.push_back(strAttr.getValue());
        kwonlyNameToIndex[strAttr.getValue()] = index;
      }
    } else {
      kwonlyNames.reserve(signature.getKwOnlyNames().size());
      for (auto [index, attr] : llvm::enumerate(signature.getKwOnlyNames())) {
        kwonlyNames.push_back(attr.getValue());
        if (!attr.getValue().empty())
          kwonlyNameToIndex[attr.getValue()] = index;
      }
    }

    llvm::SmallSet<llvm::StringRef, 8> providedSet;
    for (llvm::StringRef providedName : providedKwNames)
      if (!providedSet.insert(providedName).second)
        return op->emitOpError("duplicate keyword '") << providedName << "'";

    mlir::Type kwargValueType;
    if (signature.hasKwarg()) {
      auto kwargType = mlir::dyn_cast<DictType>(signature.getKwargType());
      if (!kwargType)
        return op->emitOpError("kwarg parameter must be a !py.dict value");
      kwargValueType = kwargType.getValueType();
    }

    if (!signature.hasKwarg()) {
      for (llvm::StringRef providedName : providedKwNames)
        if (!positionalNameToIndex.count(providedName) &&
            !kwonlyNameToIndex.count(providedName))
          return op->emitOpError("unexpected keyword argument '")
                 << providedName << "'";
    }

    for (auto [index, providedName] : llvm::enumerate(providedKwNames)) {
      mlir::Type expected;
      if (auto positionalIt = positionalNameToIndex.find(providedName);
          positionalIt != positionalNameToIndex.end()) {
        if (positionalIt->second < posElems.size())
          return op->emitOpError("argument '")
                 << providedName
                 << "' was provided both positionally and by keyword";
        expected = positionalTypes[positionalIt->second];
      } else if (auto kwonlyIt = kwonlyNameToIndex.find(providedName);
                 kwonlyIt != kwonlyNameToIndex.end()) {
        expected = kwonlyTypes[kwonlyIt->second];
      } else if (kwargValueType) {
        expected = kwargValueType;
      } else {
        return op->emitOpError("unexpected keyword argument '")
               << providedName << "'";
      }
      if (!isSubtypeOf(valueElems[index], expected, op))
        return op->emitOpError("keyword argument '")
               << providedName << "' has type " << valueElems[index]
               << " but expected " << expected;
    }

    for (auto [index, name] : llvm::enumerate(positionalNames)) {
      if (index < posElems.size())
        continue;
      if (providedSet.contains(name))
        continue;
      if (index < minPositionalCount)
        return op->emitOpError("missing required argument '") << name << "'";
    }

    for (auto [index, expected] : llvm::enumerate(kwonlyNames)) {
      if (index < normalizedKwonlyCount)
        continue;
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
  } else if (!valueElems.empty()) {
    return op->emitOpError(
        "keyword argument names must be statically literal for typed kwvalues");
  } else if (!signature.hasVararg() && posElems.size() < minPositionalCount) {
    return op->emitOpError(
        "posargs length mismatch with callee positional parameters");
  }

  return mlir::success();
}

static mlir::FailureOr<CallableType>
requireCallableContract(mlir::Operation *op, mlir::Type calleeType,
                        llvm::StringRef methodName) {
  CallableType signature = getCallableContract(calleeType);
  if (!signature) {
    op->emitOpError(methodName) << " callee_type must be Callable";
    return mlir::failure();
  }
  return signature;
}

static mlir::LogicalResult
verifyFixedMethodContractShape(mlir::Operation *op, CallableType signature,
                               unsigned parameterCount,
                               llvm::StringRef methodName) {
  if (signature.getPositionalTypes().size() != parameterCount)
    return op->emitOpError(methodName)
           << " contract must take exactly " << parameterCount
           << " positional parameters";
  if (!signature.getKwOnlyTypes().empty() || signature.hasVararg() ||
      signature.hasKwarg())
    return op->emitOpError(methodName)
           << " contract must not take keyword, vararg, or kwarg parameters";
  return mlir::success();
}

static bool methodOperandAssignable(mlir::Type actual, mlir::Type expected,
                                    mlir::Operation *op) {
  if (isSubtypeOf(actual, expected, op))
    return true;
  if (mlir::isa<ObjectType>(expected))
    return static_cast<bool>(actual);
  if (mlir::isa<IntType>(expected) &&
      (mlir::isa<mlir::IntegerType>(actual) || actual.isIndex()))
    return true;
  if (mlir::isa<BoolType>(expected)) {
    auto integer = mlir::dyn_cast<mlir::IntegerType>(actual);
    return integer && integer.getWidth() == 1;
  }
  if (mlir::isa<FloatType>(expected) && mlir::isa<mlir::FloatType>(actual))
    return true;
  return false;
}

mlir::LogicalResult
verifyUnaryMethodContract(mlir::Operation *op, mlir::Type calleeType,
                          mlir::Type receiverType, mlir::Type argumentType,
                          mlir::Type resultType, mlir::StringRef methodName) {
  mlir::FailureOr<CallableType> signatureOr =
      requireCallableContract(op, calleeType, methodName);
  if (mlir::failed(signatureOr))
    return mlir::failure();
  CallableType signature = *signatureOr;
  if (mlir::failed(
          verifyFixedMethodContractShape(op, signature, 2, methodName)))
    return mlir::failure();
  llvm::ArrayRef<mlir::Type> positional = signature.getPositionalTypes();
  if (!methodOperandAssignable(receiverType, positional[0], op))
    return op->emitOpError(methodName)
           << " receiver type " << receiverType
           << " does not satisfy contract receiver type " << positional[0];
  if (!methodOperandAssignable(argumentType, positional[1], op))
    return op->emitOpError(methodName)
           << " argument type " << argumentType
           << " does not satisfy contract argument type " << positional[1];
  if (signature.getResultTypes().size() != 1)
    return op->emitOpError(methodName) << " contract must return one value";
  mlir::Type expected = signature.getResultTypes().front();
  if (resultType != expected)
    return op->emitOpError(methodName)
           << " result type " << resultType
           << " does not match contract result type " << expected;
  return mlir::success();
}

mlir::LogicalResult verifyContainsMethodContract(mlir::Operation *op,
                                                 mlir::Type calleeType,
                                                 mlir::Type receiverType,
                                                 mlir::Type itemType) {
  mlir::FailureOr<CallableType> signatureOr =
      requireCallableContract(op, calleeType, "__contains__");
  if (mlir::failed(signatureOr))
    return mlir::failure();
  CallableType signature = *signatureOr;
  if (mlir::failed(
          verifyFixedMethodContractShape(op, signature, 2, "__contains__")))
    return mlir::failure();
  llvm::ArrayRef<mlir::Type> positional = signature.getPositionalTypes();
  if (!methodOperandAssignable(receiverType, positional[0], op))
    return op->emitOpError("__contains__")
           << " receiver type " << receiverType
           << " does not satisfy contract receiver type " << positional[0];
  if (!methodOperandAssignable(itemType, positional[1], op))
    return op->emitOpError("__contains__")
           << " item type " << itemType
           << " does not satisfy contract item type " << positional[1];
  if (signature.getResultTypes().size() != 1 ||
      !mlir::isa<BoolType>(signature.getResultTypes().front()))
    return op->emitOpError("__contains__ contract must return !py.bool");
  return mlir::success();
}

} // namespace py
