#include "PyDialectTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

using namespace mlir;

namespace py {

namespace {

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

} // namespace

LogicalResult ClassOp::verify() {
  Region &body = getBody();
  if (body.empty())
    return emitOpError("must contain a body region");
  if (!body.hasOneBlock())
    return emitOpError("body must consist of a single block");

  Block &block = body.front();
  llvm::StringSet<> methodNames;
  ClassType classType =
      ClassType::get(getContext(), getSymNameAttr().getValue());

  for (Operation &op : block) {
    FuncOp method = dyn_cast<FuncOp>(&op);
    if (!method)
      return emitOpError("body may only contain py.func operations");

    mlir::StringAttr nameAttr = method.getSymNameAttr();
    if (!nameAttr)
      return method.emitOpError("requires 'sym_name' attribute");
    if (!methodNames.insert(nameAttr.getValue()).second)
      return emitOpError("duplicate method name '")
             << nameAttr.getValue() << "'";

    mlir::TypeAttr fnTypeAttr = method.getFunctionTypeAttr();
    if (!fnTypeAttr)
      return method.emitOpError("requires 'function_type' attribute");
    auto signature = dyn_cast<FuncSignatureType>(fnTypeAttr.getValue());
    if (!signature)
      return method.emitOpError("'function_type' must be a FuncSignatureType");

    auto positionalTypes = signature.getPositionalTypes();
    if (positionalTypes.empty())
      return method.emitOpError(
          "must declare a positional 'self' parameter as the first argument");

    Type selfType = positionalTypes.front();
    if (selfType != classType && !isPyObjectType(selfType))
      return method.emitOpError(
                 "first positional parameter must be of type !py.class<")
             << getSymNameAttr().getValue() << "> or !py.object";
  }

  return success();
}

LogicalResult FuncOp::verify() {
  mlir::TypeAttr fnTypeAttr = getFunctionTypeAttr();
  if (!fnTypeAttr)
    return emitOpError("requires 'function_type' attribute");
  FuncSignatureType signature =
      dyn_cast<FuncSignatureType>(fnTypeAttr.getValue());
  if (!signature)
    return emitOpError("'function_type' must be a FuncSignatureType");

  bool attrHasVararg = static_cast<bool>(getHasVarargAttr());
  bool attrHasKwarg = static_cast<bool>(getHasKwargsAttr());

  if (attrHasVararg != signature.hasVararg())
    return emitOpError("has_vararg attribute must match signature");
  if (attrHasKwarg != signature.hasKwarg())
    return emitOpError("has_kwargs attribute must match signature");

  if (signature.hasVararg() && !isa<TupleType>(signature.getVarargType()))
    return emitOpError("vararg parameter must be of type !py.tuple");
  if (signature.hasVararg()) {
    TupleType tupleTy = cast<TupleType>(signature.getVarargType());
    mlir::ArrayRef<mlir::Type> elems = tupleTy.getElementTypes();
    if (elems.size() != 1)
      return emitOpError(
          "*args parameter must be !py.tuple<T> with single element type");
    if (!isPyType(elems.front()))
      return emitOpError("*args element type must be a !py.* type");
  }

  if (signature.hasKwarg()) {
    DictType dictTy = dyn_cast<DictType>(signature.getKwargType());
    if (!dictTy)
      return emitOpError("kwarg parameter must be of type !py.dict");
    if (!isPyStrType(dictTy.getKeyType()) ||
        !isPyObjectType(dictTy.getValueType()))
      return emitOpError("kwarg mapping must be !py.dict<!py.str, !py.object>");
  }

  mlir::ArrayRef<mlir::Type> positionalTypes = signature.getPositionalTypes();
  mlir::ArrayRef<mlir::Type> kwonlyTypes = signature.getKwOnlyTypes();

  if (ArrayAttr argNames = getArgNamesAttr()) {
    if (argNames.size() != positionalTypes.size())
      return emitOpError(
          "arg_names length must match positional parameter count");
    for (Attribute attr : argNames)
      if (!isa<StringAttr>(attr))
        return emitOpError("arg_names must contain only StringAttr elements");
  }

  if (ArrayAttr kwonlyNames = getKwonlyNamesAttr()) {
    if (kwonlyNames.size() != kwonlyTypes.size())
      return emitOpError(
          "kwonly_names length must match keyword-only parameters");
    for (Attribute attr : kwonlyNames)
      if (!isa<StringAttr>(attr))
        return emitOpError(
            "kwonly_names must contain only StringAttr elements");
  } else if (!kwonlyTypes.empty()) {
    return emitOpError(
        "keyword-only parameters require kwonly_names attribute");
  }

  Region &body = getBody();
  if (body.empty())
    return emitOpError("must have a body region");

  Block &entry = body.front();
  unsigned expectedArgs = positionalTypes.size() + kwonlyTypes.size();
  if (signature.hasVararg())
    ++expectedArgs;
  if (signature.hasKwarg())
    ++expectedArgs;

  if (entry.getNumArguments() != expectedArgs)
    return emitOpError(
        "entry block argument count must match signature inputs");

  unsigned index = 0;
  for (Type expected : positionalTypes) {
    if (entry.getArgument(index).getType() != expected)
      return emitOpError("entry block argument ")
             << index << " type does not match positional parameter";
    ++index;
  }
  for (Type expected : kwonlyTypes) {
    if (entry.getArgument(index).getType() != expected)
      return emitOpError("entry block argument ")
             << index << " type does not match keyword-only parameter";
    ++index;
  }
  if (signature.hasVararg()) {
    if (entry.getArgument(index).getType() != signature.getVarargType())
      return emitOpError("*args slot must use the type declared in signature");
    ++index;
  }
  if (signature.hasKwarg()) {
    if (entry.getArgument(index).getType() != signature.getKwargType())
      return emitOpError(
          "**kwargs slot must use the type declared in signature");
  }

  bool foundReturn = false;
  auto expectedResults = signature.getResultTypes();
  for (Block &block : body) {
    auto ret = dyn_cast<ReturnOp>(block.getTerminator());
    if (!ret)
      continue;
    foundReturn = true;
    if (ret.getNumOperands() != expectedResults.size())
      return ret.emitOpError("result count mismatch with function signature");
    for (auto [value, expected] : llvm::zip(ret.getOperands(), expectedResults))
      if (value.getType() != expected)
        return ret.emitOpError(
            "return operand type does not match function result type");
  }

  if (!foundReturn)
    return emitOpError("body must contain at least one py.return");

  return success();
}

LogicalResult CallOp::verify() {
  Type callableType = getCallable().getType();

  if (!isCallableType(callableType))
    return emitOpError("callable operand must be !py.func or !py.class");

  FuncSignatureType signature;
  if (FuncType funcTy = dyn_cast<FuncType>(callableType)) {
    signature = funcTy.getSignature();
  } else if (ClassType classTy = dyn_cast<ClassType>(callableType)) {
    ClassOp classOp = lookupClassSymbol(getOperation(), classTy);
    if (!classOp)
      return emitOpError("unable to resolve class '")
             << classTy.getClassName() << "'";
    FuncOp callMethod = lookupMethodByName(classOp, "__call__");
    if (!callMethod) {
      emitOpError("class '")
          << classTy.getClassName() << "' does not define a '__call__' method";
      return failure();
    }
    mlir::TypeAttr fnTypeAttr = callMethod.getFunctionTypeAttr();
    if (!fnTypeAttr) {
      callMethod.emitOpError("requires 'function_type' attribute");
      return failure();
    }
    FuncSignatureType methodSig =
        dyn_cast<FuncSignatureType>(fnTypeAttr.getValue());
    if (!methodSig) {
      callMethod.emitOpError("'function_type' must be a FuncSignatureType");
      return failure();
    }

    mlir::ArrayRef<mlir::Type> positionalTypes = methodSig.getPositionalTypes();
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

    signature = FuncSignatureType::get(
        methodSig.getContext(),
        ArrayRef<Type>(positionalTypes.begin() + 1, positionalTypes.end()),
        methodSig.getKwOnlyTypes(),
        methodSig.hasVararg() ? methodSig.getVarargType() : Type(),
        methodSig.hasKwarg() ? methodSig.getKwargType() : Type(),
        methodSig.getResultTypes());
  } else {
    return emitOpError("unexpected callable type");
  }

  TupleType tupleTy = dyn_cast<TupleType>(getPosargs().getType());
  if (!tupleTy)
    return emitOpError("posargs operand must be a !py.tuple value");
  mlir::ArrayRef<mlir::Type> tupleElems = tupleTy.getElementTypes();
  bool homogeneous = tupleElems.size() == 1;

  mlir::ArrayRef<mlir::Type> positionalTypes = signature.getPositionalTypes();
  if (!signature.hasVararg()) {
    if (tupleElems.size() != positionalTypes.size())
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

LogicalResult MakeFunctionOp::verify() {
  FuncType funcType = dyn_cast<FuncType>(getResult().getType());
  if (!funcType)
    return emitOpError("result must be of type !py.func");

  Operation *symbol =
      SymbolTable::lookupNearestSymbolFrom(getOperation(), getTargetAttr());
  FuncOp pyFunc = dyn_cast_or_null<FuncOp>(symbol);
  if (!pyFunc)
    return emitOpError("target must reference a py.func symbol");

  FuncSignatureType expectedSig =
      cast<FuncSignatureType>(pyFunc.getFunctionTypeAttr().getValue());
  if (expectedSig != funcType.getSignature())
    return emitOpError("result type signature must match referenced py.func");

  return success();
}

LogicalResult MakeNativeOp::verify() {
  PrimFuncType primFuncTy = dyn_cast<PrimFuncType>(getResult().getType());
  if (!primFuncTy)
    return emitOpError("result must be of type !py.prim.func");

  Operation *symbol =
      SymbolTable::lookupNearestSymbolFrom(getOperation(), getTargetAttr());
  mlir::func::FuncOp funcOp = dyn_cast_or_null<mlir::func::FuncOp>(symbol);
  if (!funcOp)
    return emitOpError("target must reference a func.func symbol");

  FunctionType signature = funcOp.getFunctionType();
  if (primFuncTy.getSignature() != signature)
    return emitOpError("result type signature must match referenced func.func");

  for (Type type : signature.getInputs())
    if (isPyType(type))
      return emitOpError("func.func arguments referenced by py.make_native "
                         "must not use !py.* types");
  for (Type type : signature.getResults())
    if (isPyType(type))
      return emitOpError("func.func results referenced by py.make_native must "
                         "not use !py.* types");

  return success();
}

LogicalResult CastFromPrimOp::verify() {
  Type inputType = getInput().getType();
  Type resultType = getResult().getType();

  if (isPyType(inputType))
    return emitOpError("input must be a primitive type, not !py.* type");
  if (!isPyType(resultType))
    return emitOpError("result must be a !py.* type");

  auto checkConversion = [&](Type prim, Type pyType) -> bool {
    return inputType == prim && resultType == pyType;
  };

  mlir::MLIRContext *ctx = getContext();
  if (checkConversion(::mlir::IntegerType::get(ctx, 32), IntType::get(ctx)))
    return success();
  if (checkConversion(::mlir::IntegerType::get(ctx, 64), IntType::get(ctx)))
    return success();
  if (checkConversion(::mlir::Float64Type::get(ctx), FloatType::get(ctx)))
    return success();
  if (checkConversion(::mlir::IntegerType::get(ctx, 1), BoolType::get(ctx)))
    return success();

  return emitOpError("unsupported type conversion from ")
         << inputType << " to " << resultType;
}

LogicalResult CastToPrimOp::verify() {
  Type inputType = getInput().getType();
  Type resultType = getResult().getType();

  if (!isPyType(inputType))
    return emitOpError("input must be a !py.* type");
  if (isPyType(resultType))
    return emitOpError("result must be a primitive type, not !py.* type");

  StringAttr modeAttr = getModeAttr();
  if (!modeAttr)
    return emitOpError("requires 'mode' attribute");
  StringRef mode = modeAttr.getValue();
  if (mode != "exact" && mode != "truncate" && mode != "saturate")
    return emitOpError("mode must be 'exact', 'truncate', or 'saturate'");

  auto checkConversion = [&](Type pyType, Type prim) -> bool {
    return inputType == pyType && resultType == prim;
  };

  mlir::MLIRContext *ctx = getContext();
  if (checkConversion(IntType::get(ctx), ::mlir::IntegerType::get(ctx, 32)))
    return success();
  if (checkConversion(IntType::get(ctx), ::mlir::IntegerType::get(ctx, 64)))
    return success();
  if (checkConversion(FloatType::get(ctx), ::mlir::Float64Type::get(ctx)))
    return success();
  if (checkConversion(BoolType::get(ctx), ::mlir::IntegerType::get(ctx, 1)))
    return success();

  return emitOpError("unsupported type conversion from ")
         << inputType << " to " << resultType;
}

LogicalResult CastIdentityOp::verify() {
  Type inputType = getInput().getType();
  Type resultType = getResult().getType();

  bool inputIsPy = isPyType(inputType);
  bool resultIsPy = isPyType(resultType);
  if (!inputIsPy && !resultIsPy)
    return emitOpError("at least one of input or result must be a !py.* type");

  if (inputType == resultType)
    return success();

  if (inputIsPy && resultIsPy)
    return emitOpError(
        "when both sides are !py.* types the element types must match");

  return success();
}

LogicalResult UpcastOp::verify() {
  Type inputType = getInput().getType();
  Type resultType = getResult().getType();

  if (!isPyType(inputType))
    return emitOpError("input must be a !py.* type");
  if (!isPyObjectType(resultType))
    return emitOpError("result must be of type !py.object");

  return success();
}

LogicalResult StrConstantOp::verify() {
  if (!getValueAttr())
    return emitOpError("requires 'value' attribute");
  if (!isPyStrType(getResult().getType()))
    return emitOpError("result must be of type !py.str");
  return success();
}

LogicalResult TupleEmptyOp::verify() {
  auto tupleTy = dyn_cast<TupleType>(getResult().getType());
  if (!tupleTy)
    return emitOpError("result must be a !py.tuple type");
  if (!tupleTy.getElementTypes().empty())
    return emitOpError(
        "result type must encode no element types for tuple.empty");
  return success();
}

LogicalResult TupleCreateOp::verify() {
  auto resultTy = dyn_cast<TupleType>(getResult().getType());
  if (!resultTy)
    return emitOpError("result must be a !py.tuple type");

  auto operands = getElements();
  auto elementTypes = resultTy.getElementTypes();

  if (elementTypes.empty()) {
    if (!operands.empty())
      return emitOpError("cannot populate an empty tuple with elements");
    return success();
  }

  if (elementTypes.size() == 1) {
    Type target = elementTypes.front();
    for (Value operand : operands)
      if (!isSubtypeOf(operand.getType(), target))
        return emitOpError("element type ")
               << operand.getType()
               << " is not compatible with tuple element type " << target;
    return success();
  }

  if (operands.size() != elementTypes.size())
    return emitOpError("number of operands must match tuple arity");

  for (auto [value, target] : llvm::zip(operands, elementTypes))
    if (!isSubtypeOf(value.getType(), target))
      return emitOpError("element type ")
             << value.getType() << " is not compatible with tuple element type "
             << target;

  return success();
}

LogicalResult DictInsertOp::verify() {
  auto dictTy = dyn_cast<DictType>(getDict().getType());
  if (!dictTy)
    return emitOpError("dict operand must be !py.dict");

  if (getResult().getType() != getDict().getType())
    return emitOpError("result type must match dictionary operand type");

  if (!isSubtypeOf(getKey().getType(), dictTy.getKeyType()))
    return emitOpError("key type ")
           << getKey().getType()
           << " is not compatible with dictionary key type "
           << dictTy.getKeyType();

  if (!isSubtypeOf(getValue().getType(), dictTy.getValueType()))
    return emitOpError("value type ")
           << getValue().getType()
           << " is not compatible with dictionary value type "
           << dictTy.getValueType();

  return success();
}

LogicalResult FuncObjectOp::verify() {
  auto funcType = dyn_cast<FuncType>(getResult().getType());
  if (!funcType)
    return emitOpError("result must be of type !py.func");

  Operation *symbol =
      SymbolTable::lookupNearestSymbolFrom(getOperation(), getTargetAttr());
  auto pyFunc = dyn_cast_or_null<FuncOp>(symbol);
  if (!pyFunc)
    return emitOpError("target must reference a py.func symbol");

  auto expectedSig =
      cast<FuncSignatureType>(pyFunc.getFunctionTypeAttr().getValue());
  if (funcType.getSignature() != expectedSig)
    return emitOpError("result type signature must match referenced py.func");

  return success();
}

LogicalResult CallVectorOp::verify() {
  Type callableType = getCallable().getType();

  ArrayAttr expectedKwNamesAttr;
  FuncSignatureType signature;
  if (auto funcTy = dyn_cast<FuncType>(callableType)) {
    signature = funcTy.getSignature();
    if (auto funcObject = getCallable().getDefiningOp<FuncObjectOp>()) {
      Operation *symbol = SymbolTable::lookupNearestSymbolFrom(
          getOperation(), funcObject.getTargetAttr());
      if (auto funcOp = dyn_cast_or_null<FuncOp>(symbol))
        expectedKwNamesAttr = funcOp.getKwonlyNamesAttr();
    } else if (auto makeFunc = getCallable().getDefiningOp<MakeFunctionOp>()) {
      Operation *symbol = SymbolTable::lookupNearestSymbolFrom(
          getOperation(), makeFunc.getTargetAttr());
      if (auto funcOp = dyn_cast_or_null<FuncOp>(symbol))
        expectedKwNamesAttr = funcOp.getKwonlyNamesAttr();
    }
  } else if (auto classTy = dyn_cast<ClassType>(callableType)) {
    ClassOp classOp = lookupClassSymbol(getOperation(), classTy);
    if (!classOp)
      return emitOpError("unable to resolve class '")
             << classTy.getClassName() << "'";

    FuncOp callMethod = lookupMethodByName(classOp, "__call__");
    if (!callMethod) {
      emitOpError("class '")
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

    signature = FuncSignatureType::get(methodSig.getContext(), callPositional,
                                       callKwOnly, callVararg, callKwarg,
                                       methodSig.getResultTypes());
    expectedKwNamesAttr = callMethod.getKwonlyNamesAttr();
  } else {
    return emitOpError("unexpected callable type");
  }

  auto requireTuple = [&](Value operand,
                          StringRef what) -> mlir::FailureOr<TupleType> {
    if (auto tupleTy = dyn_cast<TupleType>(operand.getType()))
      return tupleTy;
    emitOpError() << what << " operand must be a !py.tuple value";
    return mlir::failure();
  };

  mlir::FailureOr<TupleType> posTupleOr = requireTuple(getPosargs(), "posargs");
  if (failed(posTupleOr))
    return failure();
  TupleType posTuple = *posTupleOr;
  auto posElems = posTuple.getElementTypes();
  bool homogeneous = posElems.size() == 1;

  auto positionalTypes = signature.getPositionalTypes();
  if (!signature.hasVararg()) {
    if (posElems.size() != positionalTypes.size())
      return emitOpError(
          "posargs length mismatch with callee positional parameters");
    for (auto [elemType, expected] : llvm::zip(posElems, positionalTypes))
      if (!isSubtypeOf(elemType, expected))
        return emitOpError(
            "posargs element type does not match positional parameter type");
  } else {
    auto calleeVarargTy = dyn_cast<TupleType>(signature.getVarargType());
    if (!calleeVarargTy || calleeVarargTy.getElementTypes().size() != 1)
      return emitOpError(
          "vararg parameter must be !py.tuple<T> with single element type");
    Type calleeElemType = calleeVarargTy.getElementTypes().front();

    if (homogeneous && !posElems.empty() && positionalTypes.empty()) {
      if (!isSubtypeOf(posElems.front(), calleeElemType))
        return emitOpError(
            "posargs element type is incompatible with vararg element type");
    } else {
      if (posElems.size() < positionalTypes.size())
        return emitOpError(
            "posargs length shorter than positional parameter count");
      for (size_t i = 0; i < positionalTypes.size(); ++i)
        if (!isSubtypeOf(posElems[i], positionalTypes[i]))
          return emitOpError("posargs element type does not match positional "
                             "parameter type at index ")
                 << i;
      for (size_t i = positionalTypes.size(); i < posElems.size(); ++i)
        if (!isSubtypeOf(posElems[i], calleeElemType))
          return emitOpError("vararg element type incompatible at index ") << i;
    }
  }

  mlir::FailureOr<TupleType> nameTupleOr =
      requireTuple(getKwnames(), "kwnames");
  if (failed(nameTupleOr))
    return failure();
  TupleType nameTuple = *nameTupleOr;

  mlir::FailureOr<TupleType> valueTupleOr =
      requireTuple(getKwvalues(), "kwvalues");
  if (failed(valueTupleOr))
    return failure();
  TupleType valueTuple = *valueTupleOr;

  auto nameElems = nameTuple.getElementTypes();
  for (Type elem : nameElems)
    if (!isPyStrType(elem))
      return emitOpError("kwnames tuple elements must be !py.str");

  auto valueElems = valueTuple.getElementTypes();
  for (Type elem : valueElems)
    if (!isPyObjectType(elem))
      return emitOpError("kwvalues tuple elements must be !py.object");

  if (nameElems.size() != valueElems.size())
    return emitOpError(
        "kwnames and kwvalues must describe the same number of entries");

  auto kwonlyTypes = signature.getKwOnlyTypes();
  bool kwargsRequired = signature.hasKwarg() || !kwonlyTypes.empty();
  bool namesEmpty = nameElems.empty();
  if (kwargsRequired && namesEmpty)
    return emitOpError(
        "callee expects keyword arguments but none were provided");
  if (!kwargsRequired && !namesEmpty)
    return emitOpError("callee does not accept keyword arguments");

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
  bool hasLiteralNames = collectTupleValues(getKwnames(), nameValues);
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
    return emitOpError("callee does not accept keyword arguments");
  if (kwargsRequired && nameValues.empty() && nameElems.empty())
    return emitOpError(
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
        return emitOpError("duplicate keyword '") << providedName << "'";
    for (StringRef expected : expectedNames)
      if (!providedSet.contains(expected))
        return emitOpError("missing keyword argument '") << expected << "'";
    if (!signature.hasKwarg()) {
      if (providedKwNames.size() != expectedNames.size())
        return emitOpError("unexpected keyword argument provided");
    }
  }

  auto resultTypes = signature.getResultTypes();
  if (getNumResults() != resultTypes.size())
    return emitOpError("result count mismatch with callee signature");
  for (auto [result, expected] : llvm::zip(getResultTypes(), resultTypes)) {
    if (result != expected)
      return emitOpError("result types must match callee return types");
  }

  return success();
}

LogicalResult NumAddOp::verify() {
  Type lhsType = getLhs().getType();
  Type rhsType = getRhs().getType();
  Type resultType = getResult().getType();

  if (lhsType != rhsType)
    return emitOpError("operand types must match");
  if (lhsType != resultType)
    return emitOpError("result type must match operand types");

  if (!isPyIntType(lhsType) && !isPyFloatType(lhsType))
    return emitOpError("operands must be !py.int or !py.float");

  return success();
}

LogicalResult FloatConstantOp::verify() { return success(); }

LogicalResult NumLeOp::verify() {
  Type lhsType = getLhs().getType();
  Type rhsType = getRhs().getType();
  if (lhsType != rhsType)
    return emitOpError("operand types must match");
  if (!isPyIntType(lhsType) && !isPyFloatType(lhsType))
    return emitOpError("operands must be !py.int or !py.float");
  if (!isPyBoolType(getResult().getType()))
    return emitOpError("result must be !py.bool");
  return success();
}

} // namespace py
