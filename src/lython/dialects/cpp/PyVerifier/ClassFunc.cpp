#include "cpp/PyVerifier/Common.h"

#include "cpp/PyTypeObject.h"

#include "mlir/Dialect/Async/IR/Async.h"
#include "llvm/ADT/StringSet.h"

namespace py {

static bool isSupportedStaticFieldType(mlir::Type type) {
  if (mlir::isa<ClassType, TypeType, ProtocolType, ObjectType, IntType,
                FloatType, BoolType, StrType, NoneType, ExceptionType,
                TracebackType, mlir::IntegerType, mlir::FloatType>(type))
    return true;
  auto isSupportedContainerElement = [](mlir::Type elementType) {
    return isPyType(elementType) ||
           mlir::isa<mlir::IntegerType, mlir::FloatType>(elementType);
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
             << "; supported field types are !py.* contract/value types, "
                "integers, floats, typed lists, and typed dicts/tuples";
  }

  return mlir::success();
}

static mlir::LogicalResult verifyClassMethodSchema(ClassOp op) {
  mlir::ArrayAttr methodNames = op.getMethodNamesAttr();
  mlir::ArrayAttr methodTypes = op.getMethodTypesAttr();
  mlir::ArrayAttr methodKinds = op.getMethodKindsAttr();

  if (!methodNames && !methodTypes && !methodKinds)
    return mlir::success();
  if (!methodNames || !methodTypes || !methodKinds)
    return op.emitOpError(
        "method_names, method_types, and method_kinds must be provided "
        "together");
  if (methodNames.size() != methodTypes.size() ||
      methodNames.size() != methodKinds.size())
    return op.emitOpError("method_names, method_types, and method_kinds must "
                          "have the same number of elements");

  llvm::StringSet<> seenNames;
  for (mlir::Attribute attr : methodNames) {
    auto nameAttr = mlir::dyn_cast<mlir::StringAttr>(attr);
    if (!nameAttr)
      return op.emitOpError("method_names must contain only StringAttr values");
    if (!seenNames.insert(nameAttr.getValue()).second)
      return op.emitOpError("duplicate method name '")
             << nameAttr.getValue() << "'";
  }

  for (mlir::Attribute attr : methodTypes) {
    auto typeAttr = mlir::dyn_cast<mlir::TypeAttr>(attr);
    if (!typeAttr)
      return op.emitOpError("method_types must contain only TypeAttr values");
    if (!mlir::isa<CallableType>(typeAttr.getValue()))
      return op.emitOpError(
          "method_types must contain only CallableType values");
  }

  for (mlir::Attribute attr : methodKinds) {
    auto kindAttr = mlir::dyn_cast<mlir::StringAttr>(attr);
    if (!kindAttr)
      return op.emitOpError("method_kinds must contain only StringAttr values");
    llvm::StringRef kind = kindAttr.getValue();
    if (kind != "instance" && kind != "static" && kind != "class")
      return op.emitOpError("unsupported method kind '") << kind << "'";
  }

  return mlir::success();
}

static std::optional<std::pair<CallableType, llvm::StringRef>>
lookupMethodSchema(ClassOp op, llvm::StringRef methodName) {
  mlir::ArrayAttr methodNames = op.getMethodNamesAttr();
  mlir::ArrayAttr methodTypes = op.getMethodTypesAttr();
  mlir::ArrayAttr methodKinds = op.getMethodKindsAttr();
  if (!methodNames || !methodTypes || !methodKinds)
    return std::nullopt;
  for (auto [index, attr] : llvm::enumerate(methodNames)) {
    auto nameAttr = mlir::dyn_cast<mlir::StringAttr>(attr);
    if (!nameAttr || nameAttr.getValue() != methodName)
      continue;
    auto typeAttr = mlir::dyn_cast<mlir::TypeAttr>(methodTypes[index]);
    auto kindAttr = mlir::dyn_cast<mlir::StringAttr>(methodKinds[index]);
    auto signature = typeAttr
                         ? mlir::dyn_cast<CallableType>(typeAttr.getValue())
                         : CallableType();
    if (!signature || !kindAttr)
      return std::nullopt;
    return std::make_pair(signature, kindAttr.getValue());
  }
  return std::nullopt;
}

static llvm::StringRef localMethodName(llvm::StringRef symbolName,
                                       llvm::StringRef className) {
  if (symbolName.consume_front(className) && symbolName.consume_front("."))
    return symbolName;
  return symbolName;
}

static bool isAllowedClassBodyMetadataOp(mlir::Operation *op) {
  llvm::StringRef name = op->getName().getStringRef();
  return name == "py.str.constant" || name == "py.int.constant" ||
         name == "py.float.constant" || name == "py.none" ||
         name == "py.tuple.empty" || name == "py.tuple.create" ||
         name == "py.dict.empty" || name == "py.dict.insert" ||
         name == "py.callable.object" || name == "py.make_function" ||
         name == "py.publish";
}

mlir::LogicalResult ClassOp::verify() {
  mlir::Region &body = getBody();
  // Declaration-only classes (the typing manifest's abstract tower) carry an
  // empty body region; base and schema checks still apply.
  if (body.empty()) {
    if (mlir::failed(verifyClassFieldSchema(*this)))
      return mlir::failure();
    if (mlir::failed(verifyClassMethodSchema(*this)))
      return mlir::failure();
    return type_object::verifyBases(*this);
  }
  if (!body.hasOneBlock())
    return emitOpError("body must consist of a single block");
  if (mlir::failed(verifyClassFieldSchema(*this)))
    return mlir::failure();
  if (mlir::failed(verifyClassMethodSchema(*this)))
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
    CallableFuncOp method = mlir::dyn_cast<CallableFuncOp>(&op);
    if (!method)
      return emitOpError(
          "body may only contain py.callable.func operations or metadata ops");

    mlir::StringAttr nameAttr = method.getSymNameAttr();
    if (!nameAttr)
      return method.emitOpError("requires 'sym_name' attribute");
    if (!methodNames.insert(nameAttr.getValue()).second)
      return emitOpError("duplicate method name '")
             << nameAttr.getValue() << "'";

    mlir::TypeAttr fnTypeAttr = method.getFunctionTypeAttr();
    if (!fnTypeAttr)
      return method.emitOpError("requires 'function_type' attribute");
    auto signature = mlir::dyn_cast<CallableType>(fnTypeAttr.getValue());
    if (!signature)
      return method.emitOpError("'function_type' must be a CallableType");

    llvm::StringRef simpleName =
        localMethodName(nameAttr.getValue(), getSymNameAttr().getValue());
    std::optional<std::pair<CallableType, llvm::StringRef>> schema =
        lookupMethodSchema(*this, simpleName);
    llvm::StringRef kind = schema ? schema->second : "instance";
    if (!schema) {
      if (method->hasAttr("ly.typing.classmethod"))
        kind = "class";
      else if (method->hasAttr("ly.typing.staticmethod"))
        kind = "static";
    }
    if (schema && schema->first != signature)
      return method.emitOpError("function_type does not match method schema "
                                "for '")
             << simpleName << "'";

    if (kind == "static")
      continue;

    auto positionalTypes = signature.getPositionalTypes();
    if (positionalTypes.empty())
      return method.emitOpError(
          kind == "class"
              ? "must declare a positional 'cls' parameter as the first "
                "argument"
              : "must declare a positional 'self' parameter as the first "
                "argument");

    mlir::Type selfType = positionalTypes.front();
    if (kind == "class") {
      mlir::Type classObjectType = TypeType::get(getContext(), classType);
      if (selfType != classObjectType)
        return method.emitOpError(
                   "first positional parameter must be of type !py.type<!py."
                   "class<")
               << getSymNameAttr().getValue() << ">>";
      continue;
    }

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

mlir::LogicalResult ClassObjectOp::verify() {
  auto resultType = mlir::dyn_cast<TypeType>(getResult().getType());
  if (!resultType)
    return emitOpError("result must be of type !py.type");

  llvm::StringRef symbolName = getClassNameAttr().getValue();
  mlir::Operation *symbol = mlir::SymbolTable::lookupNearestSymbolFrom(
      getOperation(), getClassNameAttr());
  if (!mlir::dyn_cast_or_null<ClassOp>(symbol))
    return emitOpError("class_name must reference a py.class symbol");

  mlir::Type instanceType = resultType.getInstanceType();
  if (mlir::isa<ExceptionType>(instanceType)) {
    if (symbolName == type_object::kBaseException ||
        symbolName == type_object::kException)
      return mlir::success();
    mlir::FailureOr<bool> subclass = type_object::isSubclassOf(
        getOperation(), symbolName, type_object::kBaseException);
    if (mlir::failed(subclass))
      return mlir::failure();
    if (*subclass)
      return mlir::success();
  }

  auto classType = mlir::dyn_cast<ClassType>(instanceType);
  if (!classType || classType.getClassName() != symbolName)
    return emitOpError("result instance type must match referenced class "
                       "symbol '")
           << symbolName << "'";

  return mlir::success();
}

mlir::LogicalResult ClassPromoteOp::verify() {
  if (getInput().getType() != getResult().getType())
    return emitOpError("input and result types must match");
  if (!mlir::isa<ClassType>(getInput().getType()))
    return emitOpError("operand must be of type !py.class");
  return mlir::success();
}

mlir::LogicalResult ClassUpcastOp::verify() {
  auto inputType = mlir::dyn_cast<ClassType>(getInput().getType());
  auto resultType = mlir::dyn_cast<ClassType>(getResult().getType());
  if (!inputType || !resultType)
    return emitOpError("input and result must be !py.class types");
  if (!isSubtypeOf(inputType, resultType, getOperation()))
    return emitOpError("input class ")
           << inputType << " is not a subclass of result class " << resultType;
  return mlir::success();
}

mlir::LogicalResult ClassRefineOp::verify() {
  auto inputType = mlir::dyn_cast<ClassType>(getInput().getType());
  auto resultType = mlir::dyn_cast<ClassType>(getResult().getType());
  if (!inputType || !resultType)
    return emitOpError("input and result must be !py.class types");
  if (!isSubtypeOf(resultType, inputType, getOperation()))
    return emitOpError("result class ")
           << resultType << " is not a subclass of input class " << inputType;
  return mlir::success();
}

mlir::LogicalResult ProtocolViewOp::verify() {
  mlir::Type inputType = getInput().getType();
  if (!isPyType(inputType))
    return emitOpError("input must be a !py value type");
  if (!mlir::isa<ProtocolType>(getResult().getType()))
    return emitOpError("result must be a !py.protocol type");

  // The current runtime ABI can forward object-family values whose first
  // lowered part is the generic object header. Typed tuple/list/dict values use
  // specialized container headers and need a separate object view before they
  // can escape as Protocol.
  if (mlir::isa<ClassType, ObjectType, ProtocolType, TypeType, TracebackType,
                IntType, StrType>(inputType))
    return mlir::success();
  return emitOpError("input type ")
         << inputType << " does not have a direct object-header protocol view";
}

mlir::LogicalResult ClassTestOp::verify() {
  auto inputType = mlir::dyn_cast<ClassType>(getInput().getType());
  auto targetType = mlir::dyn_cast<ClassType>(getTarget());
  if (!inputType || !targetType)
    return emitOpError("input and target must be !py.class types");
  if (isSubtypeOf(inputType, targetType, getOperation()) ||
      isSubtypeOf(targetType, inputType, getOperation()))
    return mlir::success();
  return emitOpError("target class ")
         << targetType << " is not comparable with input class " << inputType;
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

mlir::LogicalResult ContainsOp::verify() {
  if (!getResult().getType().isInteger(1))
    return emitOpError("contains result must be i1");
  return verifyContainsMethodContract(getOperation(), getCalleeType(),
                                      getContainer().getType(),
                                      getItem().getType());
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

mlir::LogicalResult CallableFuncOp::verify() {
  mlir::TypeAttr fnTypeAttr = getFunctionTypeAttr();
  if (!fnTypeAttr)
    return emitOpError("requires 'function_type' attribute");
  CallableType signature = mlir::dyn_cast<CallableType>(fnTypeAttr.getValue());
  if (!signature)
    return emitOpError("'function_type' must be a CallableType");

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
    for (mlir::Type elem : tupleTy.getElementTypes())
      if (!isPyType(elem))
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
  // A bodyless py.callable.func is a declaration: the typing manifest records
  // method signatures this way.
  if (body.empty())
    return mlir::success();

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
           llvm::zip(ret.getOperands(), expectedResults)) {
        if (value.getType() == expected)
          continue;
        if (getCallableContract(value.getType()) &&
            getCallableContract(expected) &&
            isSubtypeOf(value.getType(), expected, getOperation()))
          continue;
        if (value.getType() != expected)
          return ret.emitOpError(
              "return operand type does not match function result type");
      }
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

mlir::LogicalResult CallableObjectOp::verify() {
  CallableType resultSig = getCallableContract(getResult().getType());
  if (!resultSig)
    return emitOpError("result must be a Callable contract");

  mlir::Operation *symbol = mlir::SymbolTable::lookupNearestSymbolFrom(
      getOperation(), getTargetAttr());
  CallableType expectedSig;
  if (auto pyFunc = mlir::dyn_cast_or_null<CallableFuncOp>(symbol)) {
    expectedSig =
        mlir::cast<CallableType>(pyFunc.getFunctionTypeAttr().getValue());
  } else if (auto asyncFunc =
                 mlir::dyn_cast_or_null<mlir::async::FuncOp>(symbol)) {
    mlir::FunctionType asyncType = asyncFunc.getFunctionType();
    if (asyncType.getNumInputs() == 0 || asyncType.getNumResults() != 1)
      return emitOpError("async target must carry an exception cell and one "
                         "async result");
    auto asyncResult =
        mlir::dyn_cast<mlir::async::ValueType>(asyncType.getResult(0));
    if (!asyncResult)
      return emitOpError("async target result must be !async.value");
    mlir::TypeRange inputs = asyncType.getInputs();
    mlir::MLIRContext *ctx = getContext();
    mlir::Type coroutineResult =
        ProtocolType::get(ctx, "Coroutine",
                          {ObjectType::get(ctx), ObjectType::get(ctx),
                           asyncResult.getValueType()});
    llvm::SmallVector<mlir::Type> publicInputs(inputs.drop_back().begin(),
                                               inputs.drop_back().end());
    expectedSig =
        CallableType::get(ctx, publicInputs, {}, {}, {}, {coroutineResult});
  } else {
    return emitOpError(
        "target must reference a py.callable.func or async.func symbol");
  }
  if (resultSig != expectedSig)
    return emitOpError("result type signature must match referenced function");

  return mlir::success();
}

mlir::LogicalResult MakeFunctionOp::verify() {
  return verifyMakeFunctionEvidence(*this);
}

} // namespace py
