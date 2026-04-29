#include "cpp/PyVerifier/Common.h"

using namespace mlir;

namespace py {

static bool isSupportedStaticFieldType(Type type) {
  if (isa<IntType, FloatType, BoolType, StrType, NoneType, IntegerType,
          mlir::FloatType>(type))
    return true;
  auto isSupportedContainerElement = [](Type elementType) {
    return isa<ClassType, IntType, FloatType, BoolType, StrType, NoneType,
               ObjectType>(elementType);
  };
  if (auto listType = dyn_cast<ListType>(type)) {
    return isSupportedContainerElement(listType.getElementType());
  }
  if (auto dictType = dyn_cast<DictType>(type)) {
    return isSupportedContainerElement(dictType.getKeyType()) &&
           isSupportedContainerElement(dictType.getValueType());
  }
  if (auto tupleType = dyn_cast<TupleType>(type)) {
    return llvm::all_of(tupleType.getElementTypes(),
                        isSupportedContainerElement);
  }
  return false;
}

static LogicalResult verifyClassFieldSchema(ClassOp op) {
  ArrayAttr fieldNames = op.getFieldNamesAttr();
  ArrayAttr fieldTypes = op.getFieldTypesAttr();

  if (!fieldNames && !fieldTypes)
    return success();
  if (!fieldNames || !fieldTypes)
    return op.emitOpError(
        "field_names and field_types must be provided together");
  if (fieldNames.size() != fieldTypes.size())
    return op.emitOpError("field_names and field_types must have the same "
                          "number of elements");

  llvm::StringSet<> seenNames;
  for (Attribute attr : fieldNames) {
    auto nameAttr = dyn_cast<StringAttr>(attr);
    if (!nameAttr)
      return op.emitOpError("field_names must contain only StringAttr values");
    if (!seenNames.insert(nameAttr.getValue()).second)
      return op.emitOpError("duplicate field name '")
             << nameAttr.getValue() << "'";
  }

  for (Attribute attr : fieldTypes) {
    auto typeAttr = dyn_cast<TypeAttr>(attr);
    if (!typeAttr)
      return op.emitOpError("field_types must contain only TypeAttr values");
    if (!isSupportedStaticFieldType(typeAttr.getValue()))
      return op.emitOpError("unsupported static field type ")
             << typeAttr.getValue()
             << "; supported field types are !py.int, !py.float, !py.bool, "
                "!py.str, !py.none, integers, floats, typed lists, and typed "
                "dicts/tuples";
  }

  return success();
}

static bool isAllowedClassBodyMetadataOp(Operation *op) {
  StringRef name = op->getName().getStringRef();
  return name == "py.str.constant" || name == "py.int.constant" ||
         name == "py.float.constant" || name == "py.none" ||
         name == "py.tuple.empty" || name == "py.tuple.create" ||
         name == "py.dict.empty" || name == "py.dict.insert" ||
         name == "py.upcast" || name == "py.func.object" ||
         name == "py.make_function" || name == "py.publish";
}

LogicalResult ClassOp::verify() {
  Region &body = getBody();
  if (body.empty())
    return emitOpError("must contain a body region");
  if (!body.hasOneBlock())
    return emitOpError("body must consist of a single block");
  if (failed(verifyClassFieldSchema(*this)))
    return failure();

  Block &block = body.front();
  llvm::StringSet<> methodNames;
  ClassType classType =
      ClassType::get(getContext(), getSymNameAttr().getValue());

  for (Operation &op : block) {
    if (isAllowedClassBodyMetadataOp(&op))
      continue;
    FuncOp method = dyn_cast<FuncOp>(&op);
    if (!method)
      return emitOpError(
          "body may only contain py.func operations or metadata ops");

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
    if (selfType != classType)
      return method.emitOpError(
                 "first positional parameter must be of type !py.class<")
             << getSymNameAttr().getValue() << ">";
  }

  return success();
}

LogicalResult ClassNewOp::verify() {
  auto resultType = dyn_cast<ClassType>(getResult().getType());
  if (!resultType)
    return emitOpError("result must be of type !py.class");

  StringRef symbolName = getClassNameAttr().getValue();
  if (resultType.getClassName() != symbolName)
    return emitOpError("result type must match referenced class symbol '")
           << symbolName << "'";

  Operation *symbol =
      SymbolTable::lookupNearestSymbolFrom(getOperation(), getClassNameAttr());
  if (!dyn_cast_or_null<ClassOp>(symbol))
    return emitOpError("class_name must reference a py.class symbol");

  return success();
}

LogicalResult ClassPromoteOp::verify() {
  if (getInput().getType() != getResult().getType())
    return emitOpError("input and result types must match");
  if (!isa<ClassType>(getInput().getType()))
    return emitOpError("operand must be of type !py.class");
  return success();
}

LogicalResult PublishOp::verify() {
  if (getInput().getType() != getResult().getType())
    return emitOpError("input and result types must match");
  if (!isPyType(getInput().getType()))
    return emitOpError("operand must be a !py.* type");
  return success();
}

LogicalResult AttrGetOp::verify() {
  auto classType = dyn_cast<ClassType>(getObject().getType());
  if (!classType)
    return emitOpError("object operand must be of type !py.class");

  FailureOr<Type> expectedType =
      lookupClassFieldType(getOperation(), classType, getNameAttr().getValue());
  if (failed(expectedType))
    return failure();
  if (getResult().getType() != *expectedType)
    return emitOpError("result type must match declared field type");

  return success();
}

LogicalResult ListNewOp::verify() {
  auto listType = dyn_cast<ListType>(getResult().getType());
  if (!listType)
    return emitOpError("result must be of type !py.list");
  Type elementType = listType.getElementType();
  if (!isa<ClassType, IntType, FloatType, BoolType, StrType, NoneType,
           ObjectType>(elementType))
    return emitOpError("unsupported list element type ") << elementType;
  return success();
}

LogicalResult ListAppendOp::verify() {
  auto listType = dyn_cast<ListType>(getList().getType());
  if (!listType)
    return emitOpError("list operand must be of type !py.list");
  if (getValue().getType() != listType.getElementType())
    return emitOpError("value type must match list element type");
  return success();
}

LogicalResult ListRemoveOp::verify() {
  auto listType = dyn_cast<ListType>(getList().getType());
  if (!listType)
    return emitOpError("list operand must be of type !py.list");
  if (getValue().getType() != listType.getElementType())
    return emitOpError("value type must match list element type");
  return success();
}

LogicalResult ListGetOp::verify() {
  auto listType = dyn_cast<ListType>(getList().getType());
  if (!listType)
    return emitOpError("list operand must be of type !py.list");
  if (!isa<IntType, IntegerType>(getIndex().getType()))
    return emitOpError("index operand must be !py.int or an integer");
  if (getResult().getType() != listType.getElementType())
    return emitOpError("result type must match list element type");
  return success();
}

LogicalResult DictGetOp::verify() {
  auto dictType = dyn_cast<DictType>(getDict().getType());
  if (!dictType)
    return emitOpError("dict operand must be of type !py.dict");
  if (getKey().getType() != dictType.getKeyType())
    return emitOpError("key type must match dict key type");
  if (getResult().getType() != dictType.getValueType())
    return emitOpError("result type must match dict value type");
  return success();
}

LogicalResult AttrSetOp::verify() {
  auto classType = dyn_cast<ClassType>(getObject().getType());
  if (!classType)
    return emitOpError("object operand must be of type !py.class");

  FailureOr<Type> expectedType =
      lookupClassFieldType(getOperation(), classType, getNameAttr().getValue());
  if (failed(expectedType))
    return failure();
  if (getValue().getType() != *expectedType)
    return emitOpError("value type must match declared field type");

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
    return emitOpError(
        "maythrow and nothrow attributes are mutually exclusive");

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

  SmallVector<Type, 4> closureTypes;
  if (ArrayAttr closureTypesAttr = getClosureTypesAttr()) {
    closureTypes.reserve(closureTypesAttr.size());
    for (Attribute attr : closureTypesAttr) {
      auto typeAttr = dyn_cast<TypeAttr>(attr);
      if (!typeAttr)
        return emitOpError("closure_types must contain only TypeAttr elements");
      closureTypes.push_back(typeAttr.getValue());
    }
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
  expectedArgs += closureTypes.size();

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
    ++index;
  }
  for (auto [closureIndex, expected] : llvm::enumerate(closureTypes)) {
    if (entry.getArgument(index).getType() != expected)
      return emitOpError("entry block closure argument ")
             << closureIndex << " type does not match closure_types";
    ++index;
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
      for (auto [value, expected] :
           llvm::zip(ret.getOperands(), expectedResults))
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

  if (Value defaults = getDefaults()) {
    auto tupleTy = dyn_cast<TupleType>(defaults.getType());
    if (!tupleTy)
      return emitOpError("defaults must be a !py.tuple value");
    auto positionalTypes = expectedSig.getPositionalTypes();
    auto defaultTypes = tupleTy.getElementTypes();
    if (defaultTypes.size() > positionalTypes.size())
      return emitOpError(
          "defaults length must not exceed positional parameter count");
    unsigned start = positionalTypes.size() - defaultTypes.size();
    for (auto [idx, elemType] : llvm::enumerate(defaultTypes)) {
      Type expected = positionalTypes[start + idx];
      if (!isSubtypeOf(elemType, expected))
        return emitOpError("default value type does not match positional "
                           "parameter type");
    }
  }

  if (Value kwdefaults = getKwdefaults()) {
    auto dictTy = dyn_cast<DictType>(kwdefaults.getType());
    if (!dictTy)
      return emitOpError("kwdefaults must be a !py.dict value");
    if (!isSubtypeOf(dictTy.getKeyType(), StrType::get(getContext())))
      return emitOpError("kwdefaults keys must be compatible with !py.str");
    if (!isSubtypeOf(dictTy.getValueType(), ObjectType::get(getContext())))
      return emitOpError(
          "kwdefaults values must be compatible with !py.object");

    auto kwonlyTypes = expectedSig.getKwOnlyTypes();
    auto kwonlyNames = pyFunc.getKwonlyNamesAttr();
    if (!kwonlyNames || kwonlyNames.size() != kwonlyTypes.size())
      return emitOpError("target keyword-only parameters must define "
                         "kwonly_names for kwdefaults verification");

    llvm::StringMap<Type> expectedByName;
    for (auto [nameAttr, type] : llvm::zip(kwonlyNames, kwonlyTypes)) {
      auto strAttr = dyn_cast<StringAttr>(nameAttr);
      if (!strAttr)
        return emitOpError(
            "kwonly_names must contain only StringAttr elements");
      expectedByName[strAttr.getValue()] = type;
    }

    SmallVector<DictInsertOp> inserts;
    Value current = kwdefaults;
    while (auto insert = current.getDefiningOp<DictInsertOp>()) {
      inserts.push_back(insert);
      current = insert.getDict();
    }
    if (!inserts.empty()) {
      if (!current.getDefiningOp<DictEmptyOp>())
        return emitOpError("statically provided kwdefaults must be built from "
                           "py.dict.empty/py.dict.insert");
      llvm::StringSet<> seen;
      auto stripStoredValueType = [](Value value) -> Type {
        while (true) {
          if (auto identity = value.getDefiningOp<CastIdentityOp>()) {
            value = identity.getInput();
            continue;
          }
          if (auto upcast = value.getDefiningOp<UpcastOp>()) {
            value = upcast.getInput();
            continue;
          }
          return value.getType();
        }
      };
      for (DictInsertOp insert : llvm::reverse(inserts)) {
        auto key = insert.getKey().getDefiningOp<StrConstantOp>();
        if (!key)
          return emitOpError("statically provided kwdefaults keys must be "
                             "py.str.constant values");
        StringRef keyName = key.getValueAttr().getValue();
        auto it = expectedByName.find(keyName);
        if (it == expectedByName.end())
          return emitOpError("kwdefaults key '")
                 << keyName << "' is not a keyword-only parameter";
        if (!seen.insert(keyName).second)
          return emitOpError("kwdefaults key '")
                 << keyName << "' appears more than once";
        if (!isSubtypeOf(stripStoredValueType(insert.getValue()), it->second))
          return emitOpError("kwdefaults value type for key '")
                 << keyName
                 << "' does not match the target keyword-only parameter type";
      }
    }
  }

  SmallVector<Type, 4> closureTypes;
  if (ArrayAttr closureTypesAttr = pyFunc.getClosureTypesAttr()) {
    closureTypes.reserve(closureTypesAttr.size());
    for (Attribute attr : closureTypesAttr) {
      auto typeAttr = dyn_cast<TypeAttr>(attr);
      if (!typeAttr)
        return emitOpError(
            "target closure_types must contain only TypeAttr elements");
      closureTypes.push_back(typeAttr.getValue());
    }
  }

  if (Value closure = getClosure()) {
    auto tupleTy = dyn_cast<TupleType>(closure.getType());
    if (!tupleTy)
      return emitOpError("closure must be a !py.tuple value");
    auto elems = tupleTy.getElementTypes();
    if (elems.size() != closureTypes.size())
      return emitOpError(
          "closure tuple length must match target closure_types");
    for (auto [elemType, expected] : llvm::zip(elems, closureTypes))
      if (!isSubtypeOf(elemType, expected))
        return emitOpError(
            "closure element type does not match target closure_types");
  } else if (!closureTypes.empty()) {
    return emitOpError(
        "target requires closure operand matching closure_types");
  }

  if (Value annotations = getAnnotations()) {
    auto dictTy = dyn_cast<DictType>(annotations.getType());
    if (!dictTy)
      return emitOpError("annotations must be a !py.dict value");
    if (!isSubtypeOf(dictTy.getKeyType(), StrType::get(getContext())))
      return emitOpError("annotations keys must be compatible with !py.str");
  }

  if (Value module = getModule())
    if (!isSubtypeOf(module.getType(), StrType::get(getContext())))
      return emitOpError("module metadata must be compatible with !py.str");

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
