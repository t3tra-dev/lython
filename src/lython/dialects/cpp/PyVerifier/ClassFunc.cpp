#include "cpp/PyVerifier/Common.h"

#include "cpp/PyTypeObject.h"

#include "llvm/ADT/StringSet.h"

namespace py {

static bool isSupportedStaticFieldType(mlir::Type type) {
  if (mlir::isa<ClassType, IntType, FloatType, BoolType, StrType, NoneType,
                mlir::IntegerType, mlir::FloatType>(type))
    return true;
  auto isSupportedContainerElement = [](mlir::Type elementType) {
    return mlir::isa<ClassType, IntType, FloatType, BoolType, StrType, NoneType,
                     mlir::IntegerType, mlir::FloatType>(elementType);
  };
  if (auto listType = mlir::dyn_cast<ListType>(type)) {
    return isSupportedContainerElement(listType.getElementType());
  }
  if (auto dictType = mlir::dyn_cast<DictType>(type)) {
    return isSupportedContainerElement(dictType.getKeyType()) &&
           isSupportedContainerElement(dictType.getValueType());
  }
  if (auto tupleType = mlir::dyn_cast<TupleType>(type)) {
    return llvm::all_of(tupleType.getElementTypes(),
                        isSupportedContainerElement);
  }
  return false;
}

static mlir::LogicalResult verifyClassFieldSchema(ClassOp op) {
  mlir::ArrayAttr fieldNames = op.getFieldNamesAttr();
  mlir::ArrayAttr fieldTypes = op.getFieldTypesAttr();

  if (!fieldNames && !fieldTypes)
    return mlir::success();
  if (!fieldNames || !fieldTypes)
    return op.emitOpError(
        "field_names and field_types must be provided together");
  if (fieldNames.size() != fieldTypes.size())
    return op.emitOpError("field_names and field_types must have the same "
                          "number of elements");

  llvm::StringSet<> seenNames;
  for (mlir::Attribute attr : fieldNames) {
    auto nameAttr = mlir::dyn_cast<mlir::StringAttr>(attr);
    if (!nameAttr)
      return op.emitOpError("field_names must contain only StringAttr values");
    if (!seenNames.insert(nameAttr.getValue()).second)
      return op.emitOpError("duplicate field name '")
             << nameAttr.getValue() << "'";
  }

  for (mlir::Attribute attr : fieldTypes) {
    auto typeAttr = mlir::dyn_cast<mlir::TypeAttr>(attr);
    if (!typeAttr)
      return op.emitOpError("field_types must contain only TypeAttr values");
    if (!isSupportedStaticFieldType(typeAttr.getValue()))
      return op.emitOpError("unsupported static field type ")
             << typeAttr.getValue()
             << "; supported field types are !py.class, !py.int, !py.float, "
                "!py.bool, !py.str, !py.none, integers, floats, typed lists, "
                "and typed dicts/tuples";
  }

  return mlir::success();
}

static bool isAllowedClassBodyMetadataOp(mlir::Operation *op) {
  llvm::StringRef name = op->getName().getStringRef();
  return name == "py.str.constant" || name == "py.int.constant" ||
         name == "py.float.constant" || name == "py.none" ||
         name == "py.tuple.empty" || name == "py.tuple.create" ||
         name == "py.dict.empty" || name == "py.dict.insert" ||
         name == "py.func.object" || name == "py.make_function" ||
         name == "py.publish";
}

mlir::LogicalResult ClassOp::verify() {
  mlir::Region &body = getBody();
  if (body.empty())
    return emitOpError("must contain a body region");
  if (!body.hasOneBlock())
    return emitOpError("body must consist of a single block");
  if (mlir::failed(verifyClassFieldSchema(*this)))
    return mlir::failure();
  if (mlir::failed(type_object::verifyBases(*this)))
    return mlir::failure();

  mlir::Block &block = body.front();
  llvm::StringSet<> methodNames;
  ClassType classType =
      ClassType::get(getContext(), getSymNameAttr().getValue());
  for (mlir::Operation &op : block) {
    if (isAllowedClassBodyMetadataOp(&op))
      continue;
    FuncOp method = mlir::dyn_cast<FuncOp>(&op);
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
    auto signature = mlir::dyn_cast<FuncSignatureType>(fnTypeAttr.getValue());
    if (!signature)
      return method.emitOpError("'function_type' must be a FuncSignatureType");

    auto positionalTypes = signature.getPositionalTypes();
    if (positionalTypes.empty())
      return method.emitOpError(
          "must declare a positional 'self' parameter as the first argument");

    mlir::Type selfType = positionalTypes.front();
    if (selfType != classType)
      return method.emitOpError(
                 "first positional parameter must be of type !py.class<")
             << getSymNameAttr().getValue() << ">";
  }

  return mlir::success();
}

mlir::LogicalResult ClassNewOp::verify() {
  auto resultType = mlir::dyn_cast<ClassType>(getResult().getType());
  if (!resultType)
    return emitOpError("result must be of type !py.class");

  llvm::StringRef symbolName = getClassNameAttr().getValue();
  if (resultType.getClassName() != symbolName)
    return emitOpError("result type must match referenced class symbol '")
           << symbolName << "'";

  mlir::Operation *symbol = mlir::SymbolTable::lookupNearestSymbolFrom(
      getOperation(), getClassNameAttr());
  if (!mlir::dyn_cast_or_null<ClassOp>(symbol))
    return emitOpError("class_name must reference a py.class symbol");

  return mlir::success();
}

mlir::LogicalResult ClassPromoteOp::verify() {
  if (getInput().getType() != getResult().getType())
    return emitOpError("input and result types must match");
  if (!mlir::isa<ClassType>(getInput().getType()))
    return emitOpError("operand must be of type !py.class");
  return mlir::success();
}

mlir::LogicalResult PublishOp::verify() {
  if (getInput().getType() != getResult().getType())
    return emitOpError("input and result types must match");
  if (!isPyType(getInput().getType()))
    return emitOpError("operand must be a !py.* type");
  return mlir::success();
}

mlir::LogicalResult AttrGetOp::verify() {
  auto classType = mlir::dyn_cast<ClassType>(getObject().getType());
  if (!classType)
    return emitOpError("object operand must be of type !py.class");

  mlir::FailureOr<mlir::Type> expectedType =
      lookupClassFieldType(getOperation(), classType, getNameAttr().getValue());
  if (mlir::failed(expectedType))
    return mlir::failure();
  if (getResult().getType() != *expectedType)
    return emitOpError("result type must match declared field type");

  return mlir::success();
}

mlir::LogicalResult AttrGetLocalOp::verify() {
  auto classType = mlir::dyn_cast<ClassType>(getObject().getType());
  if (!classType)
    return emitOpError("object operand must be of type !py.class");

  mlir::FailureOr<mlir::Type> expectedType =
      lookupClassFieldType(getOperation(), classType, getNameAttr().getValue());
  if (mlir::failed(expectedType))
    return mlir::failure();
  if (getResult().getType() != *expectedType)
    return emitOpError("result type must match declared field type");

  return mlir::success();
}

mlir::LogicalResult ListNewOp::verify() {
  auto listType = mlir::dyn_cast<ListType>(getResult().getType());
  if (!listType)
    return emitOpError("result must be of type !py.list");
  mlir::Type elementType = listType.getElementType();
  if (!mlir::isa<ClassType, IntType, FloatType, BoolType, StrType, NoneType,
                 mlir::IntegerType, mlir::FloatType>(elementType))
    return emitOpError("unsupported list element type ") << elementType;
  return mlir::success();
}

mlir::LogicalResult ListAppendOp::verify() {
  auto listType = mlir::dyn_cast<ListType>(getList().getType());
  if (!listType)
    return emitOpError("list operand must be of type !py.list");
  if (getValue().getType() != listType.getElementType())
    return emitOpError("value type must match list element type");
  return mlir::success();
}

mlir::LogicalResult ListRemoveOp::verify() {
  auto listType = mlir::dyn_cast<ListType>(getList().getType());
  if (!listType)
    return emitOpError("list operand must be of type !py.list");
  if (getValue().getType() != listType.getElementType())
    return emitOpError("value type must match list element type");
  return mlir::success();
}

mlir::LogicalResult ListGetOp::verify() {
  auto listType = mlir::dyn_cast<ListType>(getList().getType());
  if (!listType)
    return emitOpError("list operand must be of type !py.list");
  if (!mlir::isa<IntType, mlir::IntegerType>(getIndex().getType()))
    return emitOpError("index operand must be !py.int or an integer");
  if (getResult().getType() != listType.getElementType())
    return emitOpError("result type must match list element type");
  return mlir::success();
}

mlir::LogicalResult DictGetOp::verify() {
  auto dictType = mlir::dyn_cast<DictType>(getDict().getType());
  if (!dictType)
    return emitOpError("dict operand must be of type !py.dict");
  if (getKey().getType() != dictType.getKeyType())
    return emitOpError("key type must match dict key type");
  if (getResult().getType() != dictType.getValueType())
    return emitOpError("result type must match dict value type");
  return mlir::success();
}

mlir::LogicalResult AttrSetOp::verify() {
  auto classType = mlir::dyn_cast<ClassType>(getObject().getType());
  if (!classType)
    return emitOpError("object operand must be of type !py.class");

  mlir::FailureOr<mlir::Type> expectedType =
      lookupClassFieldType(getOperation(), classType, getNameAttr().getValue());
  if (mlir::failed(expectedType))
    return mlir::failure();
  if (getValue().getType() != *expectedType)
    return emitOpError("value type must match declared field type");

  return mlir::success();
}

mlir::LogicalResult AttrSetLocalOp::verify() {
  auto classType = mlir::dyn_cast<ClassType>(getObject().getType());
  if (!classType)
    return emitOpError("object operand must be of type !py.class");

  mlir::FailureOr<mlir::Type> expectedType =
      lookupClassFieldType(getOperation(), classType, getNameAttr().getValue());
  if (mlir::failed(expectedType))
    return mlir::failure();
  if (getValue().getType() != *expectedType)
    return emitOpError("value type must match declared field type");

  return mlir::success();
}

mlir::LogicalResult FuncOp::verify() {
  mlir::TypeAttr fnTypeAttr = getFunctionTypeAttr();
  if (!fnTypeAttr)
    return emitOpError("requires 'function_type' attribute");
  FuncSignatureType signature =
      mlir::dyn_cast<FuncSignatureType>(fnTypeAttr.getValue());
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

  if (signature.hasVararg() && !mlir::isa<TupleType>(signature.getVarargType()))
    return emitOpError("vararg parameter must be of type !py.tuple");
  if (signature.hasVararg()) {
    TupleType tupleTy = mlir::cast<TupleType>(signature.getVarargType());
    mlir::ArrayRef<mlir::Type> elems = tupleTy.getElementTypes();
    if (elems.size() != 1)
      return emitOpError(
          "*args parameter must be !py.tuple<T> with single element type");
    if (!isPyType(elems.front()))
      return emitOpError("*args element type must be a !py.* type");
  }

  if (signature.hasKwarg()) {
    DictType dictTy = mlir::dyn_cast<DictType>(signature.getKwargType());
    if (!dictTy)
      return emitOpError("kwarg parameter must be of type !py.dict");
    if (!isPyStrType(dictTy.getKeyType()) || !isPyType(dictTy.getValueType()))
      return emitOpError("kwarg mapping must be !py.dict<!py.str, !py.*>");
  }

  mlir::ArrayRef<mlir::Type> positionalTypes = signature.getPositionalTypes();
  mlir::ArrayRef<mlir::Type> kwonlyTypes = signature.getKwOnlyTypes();

  if (mlir::ArrayAttr argNames = getArgNamesAttr()) {
    if (argNames.size() != positionalTypes.size())
      return emitOpError(
          "arg_names length must match positional parameter count");
    for (mlir::Attribute attr : argNames)
      if (!mlir::isa<mlir::StringAttr>(attr))
        return emitOpError("arg_names must contain only StringAttr elements");
  }

  if (mlir::ArrayAttr kwonlyNames = getKwonlyNamesAttr()) {
    if (kwonlyNames.size() != kwonlyTypes.size())
      return emitOpError(
          "kwonly_names length must match keyword-only parameters");
    for (mlir::Attribute attr : kwonlyNames)
      if (!mlir::isa<mlir::StringAttr>(attr))
        return emitOpError(
            "kwonly_names must contain only StringAttr elements");
  } else if (!kwonlyTypes.empty()) {
    return emitOpError(
        "keyword-only parameters require kwonly_names attribute");
  }

  llvm::SmallVector<mlir::Type, 4> closureTypes;
  if (mlir::ArrayAttr closureTypesAttr = getClosureTypesAttr()) {
    closureTypes.reserve(closureTypesAttr.size());
    for (mlir::Attribute attr : closureTypesAttr) {
      auto typeAttr = mlir::dyn_cast<mlir::TypeAttr>(attr);
      if (!typeAttr)
        return emitOpError("closure_types must contain only TypeAttr elements");
      closureTypes.push_back(typeAttr.getValue());
    }
  }

  mlir::Region &body = getBody();
  if (body.empty())
    return emitOpError("must have a body region");

  mlir::Block &entry = body.front();
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
  for (mlir::Type expected : positionalTypes) {
    if (entry.getArgument(index).getType() != expected)
      return emitOpError("entry block argument ")
             << index << " type does not match positional parameter";
    ++index;
  }
  for (mlir::Type expected : kwonlyTypes) {
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
  for (mlir::Block &block : body) {
    mlir::Operation *term = block.getTerminator();
    if (!term)
      continue;
    if (auto ret = mlir::dyn_cast<ReturnOp>(term)) {
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
    if (mlir::isa<RaiseOp, RaiseCurrentOp>(term)) {
      foundReturn = true;
      continue;
    }
  }

  if (!foundReturn)
    return emitOpError("body must contain at least one py.return");

  return mlir::success();
}

mlir::LogicalResult FuncObjectOp::verify() {
  auto funcType = mlir::dyn_cast<FuncType>(getResult().getType());
  if (!funcType)
    return emitOpError("result must be of type !py.func");

  mlir::Operation *symbol = mlir::SymbolTable::lookupNearestSymbolFrom(
      getOperation(), getTargetAttr());
  auto pyFunc = mlir::dyn_cast_or_null<FuncOp>(symbol);
  if (!pyFunc)
    return emitOpError("target must reference a py.func symbol");

  auto expectedSig =
      mlir::cast<FuncSignatureType>(pyFunc.getFunctionTypeAttr().getValue());
  if (funcType.getSignature() != expectedSig)
    return emitOpError("result type signature must match referenced py.func");

  return mlir::success();
}

mlir::LogicalResult MakeFunctionOp::verify() {
  FuncType funcType = mlir::dyn_cast<FuncType>(getResult().getType());
  if (!funcType)
    return emitOpError("result must be of type !py.func");

  mlir::Operation *symbol = mlir::SymbolTable::lookupNearestSymbolFrom(
      getOperation(), getTargetAttr());
  FuncOp pyFunc = mlir::dyn_cast_or_null<FuncOp>(symbol);
  if (!pyFunc)
    return emitOpError("target must reference a py.func symbol");

  FuncSignatureType expectedSig =
      mlir::cast<FuncSignatureType>(pyFunc.getFunctionTypeAttr().getValue());
  if (expectedSig != funcType.getSignature())
    return emitOpError("result type signature must match referenced py.func");

  if (mlir::Value defaults = getDefaults()) {
    auto tupleTy = mlir::dyn_cast<TupleType>(defaults.getType());
    if (!tupleTy)
      return emitOpError("defaults must be a !py.tuple value");
    auto positionalTypes = expectedSig.getPositionalTypes();
    auto defaultTypes = tupleTy.getElementTypes();
    if (defaultTypes.size() > positionalTypes.size())
      return emitOpError(
          "defaults length must not exceed positional parameter count");
    unsigned start = positionalTypes.size() - defaultTypes.size();
    for (auto [idx, elemType] : llvm::enumerate(defaultTypes)) {
      mlir::Type expected = positionalTypes[start + idx];
      if (!isSubtypeOf(elemType, expected))
        return emitOpError("default value type does not match positional "
                           "parameter type");
    }
  }

  if (mlir::Value kwdefaults = getKwdefaults()) {
    auto dictTy = mlir::dyn_cast<DictType>(kwdefaults.getType());
    if (!dictTy)
      return emitOpError("kwdefaults must be a !py.dict value");
    if (!isSubtypeOf(dictTy.getKeyType(), StrType::get(getContext())))
      return emitOpError("kwdefaults keys must be compatible with !py.str");
    if (!isPyType(dictTy.getValueType()))
      return emitOpError("kwdefaults values must be !py.* typed values");

    auto kwonlyTypes = expectedSig.getKwOnlyTypes();
    auto kwonlyNames = pyFunc.getKwonlyNamesAttr();
    if (!kwonlyNames || kwonlyNames.size() != kwonlyTypes.size())
      return emitOpError("target keyword-only parameters must define "
                         "kwonly_names for kwdefaults verification");

    llvm::StringMap<mlir::Type> expectedByName;
    for (auto [nameAttr, type] : llvm::zip(kwonlyNames, kwonlyTypes)) {
      auto strAttr = mlir::dyn_cast<mlir::StringAttr>(nameAttr);
      if (!strAttr)
        return emitOpError(
            "kwonly_names must contain only StringAttr elements");
      expectedByName[strAttr.getValue()] = type;
    }

    llvm::SmallVector<DictInsertOp> inserts;
    mlir::Value current = kwdefaults;
    while (auto cast =
               current.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
      if (cast->getNumOperands() != 1)
        break;
      current = cast.getOperand(0);
    }
    if (auto empty = current.getDefiningOp<DictEmptyOp>()) {
      (void)empty;
      for (mlir::Operation *user : current.getUsers()) {
        auto insert = mlir::dyn_cast<DictInsertOp>(user);
        if (!insert)
          continue;
        if (insert->getBlock() != getOperation()->getBlock())
          return emitOpError("statically provided kwdefaults inserts must be "
                             "in the same block as py.make_function");
        if (!insert->isBeforeInBlock(getOperation()))
          continue;
        inserts.push_back(insert);
      }
      llvm::sort(inserts, [](DictInsertOp lhs, DictInsertOp rhs) {
        return lhs->isBeforeInBlock(rhs);
      });
    }
    if (!inserts.empty()) {
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
        auto key = insert.getKey().getDefiningOp<StrConstantOp>();
        if (!key)
          return emitOpError("statically provided kwdefaults keys must be "
                             "py.str.constant values");
        llvm::StringRef keyName = key.getValueAttr().getValue();
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

  llvm::SmallVector<mlir::Type, 4> closureTypes;
  if (mlir::ArrayAttr closureTypesAttr = pyFunc.getClosureTypesAttr()) {
    closureTypes.reserve(closureTypesAttr.size());
    for (mlir::Attribute attr : closureTypesAttr) {
      auto typeAttr = mlir::dyn_cast<mlir::TypeAttr>(attr);
      if (!typeAttr)
        return emitOpError(
            "target closure_types must contain only TypeAttr elements");
      closureTypes.push_back(typeAttr.getValue());
    }
  }

  if (mlir::Value closure = getClosure()) {
    auto tupleTy = mlir::dyn_cast<TupleType>(closure.getType());
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

  if (mlir::Value annotations = getAnnotations()) {
    auto dictTy = mlir::dyn_cast<DictType>(annotations.getType());
    if (!dictTy)
      return emitOpError("annotations must be a !py.dict value");
    if (!isSubtypeOf(dictTy.getKeyType(), StrType::get(getContext())))
      return emitOpError("annotations keys must be compatible with !py.str");
  }

  if (mlir::Value module = getModule())
    if (!isSubtypeOf(module.getType(), StrType::get(getContext())))
      return emitOpError("module metadata must be compatible with !py.str");

  return mlir::success();
}

mlir::LogicalResult MakeNativeOp::verify() {
  PrimFuncType primFuncTy = mlir::dyn_cast<PrimFuncType>(getResult().getType());
  if (!primFuncTy)
    return emitOpError("result must be of type !py.prim.func");

  mlir::Operation *symbol = mlir::SymbolTable::lookupNearestSymbolFrom(
      getOperation(), getTargetAttr());
  mlir::func::FuncOp funcOp =
      mlir::dyn_cast_or_null<mlir::func::FuncOp>(symbol);
  if (!funcOp)
    return emitOpError("target must reference a func.func symbol");

  mlir::FunctionType signature = funcOp.getFunctionType();
  if (primFuncTy.getSignature() != signature)
    return emitOpError("result type signature must match referenced func.func");

  for (mlir::Type type : signature.getInputs())
    if (isPyType(type))
      return emitOpError("func.func arguments referenced by py.make_native "
                         "must not use !py.* types");
  for (mlir::Type type : signature.getResults())
    if (isPyType(type))
      return emitOpError("func.func results referenced by py.make_native must "
                         "not use !py.* types");

  return mlir::success();
}

} // namespace py
