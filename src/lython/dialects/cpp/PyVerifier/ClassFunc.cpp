#include "cpp/PyVerifier/Common.h"

using namespace mlir;

namespace py {

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

  bool hasMaythrow = static_cast<bool>(getMaythrowAttr());
  bool hasNothrow = static_cast<bool>(getNothrowAttr());
  if (hasMaythrow && hasNothrow)
    return emitOpError("maythrow and nothrow attributes are mutually exclusive");

  if (signature.hasVararg() && !isa<TupleType>(signature.getVarargType()))
    return emitOpError("vararg parameter must be of type !py.tuple");
  if (signature.hasVararg())
{
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
    Operation *term = block.getTerminator();
    if (!term)
      continue;
    if (auto ret = dyn_cast<ReturnOp>(term)) {
      foundReturn = true;
      if (ret.getNumOperands() != expectedResults.size())
        return ret.emitOpError("result count mismatch with function signature");
      for (auto [value, expected] : llvm::zip(ret.getOperands(), expectedResults))
        if (value.getType() != expected)
          return ret.emitOpError(
              "return operand type does not match function result type");
      continue;
    }
    if (isa<RaiseOp, RaiseCurrentOp>(term)) {
      foundReturn = true;
      continue;
    }
  }

  if (!foundReturn)
    return emitOpError("body must contain at least one py.return");

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

} // namespace py
