#include "BuilderImpl.h"

#include "lython/parser/Parser.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/STLExtras.h"

#include <functional>
#include <memory>
#include <set>
#include <utility>

namespace lython::emitter {
namespace {

bool locationAfter(const parser::SourceLocation &lhs,
                   const parser::SourceLocation &rhs) {
  if (lhs.line != rhs.line)
    return lhs.line > rhs.line;
  if (lhs.column != rhs.column)
    return lhs.column > rhs.column;
  return lhs.offset > rhs.offset;
}

void collectArgumentNames(const parser::Node &arguments,
                          std::set<std::string> &names) {
  auto collectList = [&](llvm::StringRef fieldName) {
    const std::vector<parser::NodePtr> *args =
        nodeListField(arguments, fieldName);
    if (!args)
      return;
    for (const parser::NodePtr &arg : *args)
      if (arg)
        if (const std::string *name = stringField(*arg, "arg"))
          names.insert(*name);
  };
  collectList("posonlyargs");
  collectList("args");
  collectList("kwonlyargs");
  for (llvm::StringRef fieldName : {"vararg", "kwarg"}) {
    const parser::NodePtr *arg = nodeField(arguments, fieldName);
    if (arg && *arg)
      if (const std::string *name = stringField(**arg, "arg"))
        names.insert(*name);
  }
}

class NestedCaptureCollector {
public:
  void collect(const parser::Node &function) {
    localNames.clear();
    loadedNames.clear();
    seenLoaded.clear();
    globalNames.clear();
    nonlocalNames.clear();
    unsupportedControl.clear();

    const parser::NodePtr *args = nodeField(function, "args");
    if (args && *args)
      collectArgumentNames(**args, localNames);
    const std::vector<parser::NodePtr> *body = nodeListField(function, "body");
    if (!body)
      return;
    for (const parser::NodePtr &stmt : *body) {
      if (!stmt)
        continue;
      collectAssignedNames(*stmt, localNames);
      collectLocalDefinitionName(*stmt);
    }
    for (const parser::NodePtr &stmt : *body)
      if (stmt)
        visit(*stmt);
    for (const std::string &name : nonlocalNames) {
      if (seenLoaded.count(name))
        continue;
      loadedNames.push_back(name);
      seenLoaded.insert(name);
    }
  }

  std::set<std::string> localNames;
  std::set<std::string> globalNames;
  std::set<std::string> nonlocalNames;
  std::vector<std::string> loadedNames;
  std::set<std::string> seenLoaded;
  std::string unsupportedControl;

private:
  void collectLocalDefinitionName(const parser::Node &stmt) {
    if (stmt.kind != "FunctionDef" && stmt.kind != "AsyncFunctionDef" &&
        stmt.kind != "ClassDef")
      return;
    if (const std::string *name = stringField(stmt, "name"))
      localNames.insert(*name);
  }

  void visitFieldValue(const parser::FieldValue &value) {
    if (const auto *node = std::get_if<parser::NodePtr>(&value)) {
      if (*node)
        visit(**node);
      return;
    }
    if (const auto *nodes = std::get_if<std::vector<parser::NodePtr>>(&value))
      for (const parser::NodePtr &node : *nodes)
        if (node)
          visit(*node);
  }

  void visit(const parser::Node &node) {
    if (node.kind == "FunctionDef" || node.kind == "AsyncFunctionDef" ||
        node.kind == "ClassDef")
      return;
    if (node.kind == "Global") {
      std::optional<std::vector<std::string>> names =
          symbolListField(node, "names");
      if (names)
        globalNames.insert(names->begin(), names->end());
      return;
    }
    if (node.kind == "Nonlocal") {
      std::optional<std::vector<std::string>> names =
          symbolListField(node, "names");
      if (names)
        nonlocalNames.insert(names->begin(), names->end());
      return;
    }
    if (node.kind == "Name") {
      const std::string *name = stringField(node, "id");
      std::optional<std::string> context = symbolField(node, "ctx");
      if (name && context && *context == "Load" && !seenLoaded.count(*name)) {
        loadedNames.push_back(*name);
        seenLoaded.insert(*name);
      }
      return;
    }
    for (const parser::Field &field : node.fields)
      visitFieldValue(field.value);
  }
};

} // namespace

std::optional<mlir::Type>
Builder::Impl::parameterType(const parser::Node &arg, mlir::Type fallbackType) {
  const parser::NodePtr *annotation = nodeField(arg, "annotation");
  std::optional<mlir::Type> argType =
      annotation ? typeFromAnnotation(*annotation) : std::nullopt;
  if (!argType) {
    const std::string *typeComment = stringField(arg, "type_comment");
    if (typeComment && !typeComment->empty()) {
      parser::NodePtr annotationText = parser::makeNode("Constant", arg.range);
      parser::addField(*annotationText, "value", *typeComment);
      argType = typeFromAnnotation(annotationText);
    }
  }
  if (!argType && fallbackType)
    argType = fallbackType;
  return argType;
}

bool Builder::Impl::appendAnnotatedParameter(const parser::Node &arg,
                                             FunctionInfo &info,
                                             llvm::StringRef role,
                                             mlir::Type fallbackType) {
  const std::string *argName = stringField(arg, "arg");
  std::optional<mlir::Type> argType = parameterType(arg, fallbackType);
  if (!argName || !argType) {
    error(arg, role.str() + " must have simple static annotations");
    return false;
  }
  info.argNames.push_back(*argName);
  info.argTypes.push_back(*argType);
  return true;
}

std::optional<FunctionTypeComment>
Builder::Impl::parseFunctionTypeComment(const parser::Node &function) {
  const std::string *comment = stringField(function, "type_comment");
  if (!comment || comment->empty())
    return std::nullopt;

  parser::ParseOptions options;
  options.mode = parser::ParseMode::FunctionType;
  parser::ParseResult parsed =
      parser::parse(*comment, "<function type comment>", options);
  if (!parsed.ok() || !parsed.tree) {
    std::string message = "function type_comment is invalid";
    if (!parsed.diagnostics.empty())
      message += ": " + parsed.diagnostics.front().message;
    error(function, std::move(message));
    return std::nullopt;
  }
  if (parsed.tree->kind != "FunctionType") {
    error(function, "function type_comment did not parse to FunctionType");
    return std::nullopt;
  }

  const std::vector<parser::NodePtr> *argNodes =
      nodeListField(*parsed.tree, "argtypes");
  const parser::NodePtr *returns = nodeField(*parsed.tree, "returns");
  if (!argNodes || !returns || !*returns) {
    error(function, "function type_comment is missing argtypes or returns");
    return std::nullopt;
  }

  FunctionTypeComment result;
  for (const parser::NodePtr &arg : *argNodes) {
    std::optional<mlir::Type> argType = typeFromAnnotation(arg);
    if (!argType) {
      error(function, "function type_comment contains unsupported argument "
                      "annotation");
      return std::nullopt;
    }
    result.argTypes.push_back(*argType);
  }

  std::optional<mlir::Type> resultType = typeFromAnnotation(*returns);
  if (!resultType) {
    error(function,
          "function type_comment contains unsupported return annotation");
    return std::nullopt;
  }
  result.resultType = *resultType;
  return result;
}

bool Builder::Impl::collectCallableDefaults(const parser::Node &callable,
                                            const parser::Node &arguments,
                                            FunctionInfo &info) {
  bool ok = true;
  const parser::NodePtr *vararg = nodeField(arguments, "vararg");
  if (vararg && *vararg) {
    const std::string *argName = stringField(**vararg, "arg");
    std::optional<mlir::Type> elementType = parameterType(**vararg);
    if (!argName || !elementType) {
      error(**vararg, "vararg parameter must have a simple static annotation");
      ok = false;
    } else if (!py::isPyType(*elementType)) {
      error(**vararg,
            "vararg parameter element type must be a !py.* type, got " +
                typeString(*elementType));
      ok = false;
    } else {
      info.varargName = *argName;
      info.varargType = py::TupleType::get(&context, {*elementType});
    }
  }

  const parser::NodePtr *kwarg = nodeField(arguments, "kwarg");
  if (kwarg && *kwarg) {
    const std::string *argName = stringField(**kwarg, "arg");
    std::optional<mlir::Type> valueType = parameterType(**kwarg);
    if (!argName || !valueType) {
      error(**kwarg, "kwarg parameter must have a simple static annotation");
      ok = false;
    } else if (!py::isPyType(*valueType)) {
      error(**kwarg, "kwarg parameter value type must be a !py.* type, got " +
                         typeString(*valueType));
      ok = false;
    } else {
      info.kwargName = *argName;
      info.kwargType = dictType(strType(), *valueType);
    }
  }

  const std::vector<parser::NodePtr> *defaults =
      nodeListField(arguments, "defaults");
  if (defaults) {
    if (defaults->size() > info.positionalCount) {
      error(callable, "too many default argument values");
      ok = false;
    } else {
      info.defaultValues.assign(defaults->begin(), defaults->end());
    }
  }

  const std::vector<parser::NodePtr> *kwonlyargs =
      nodeListField(arguments, "kwonlyargs");
  const std::vector<parser::NodePtr> *kwDefaults =
      nodeListField(arguments, "kw_defaults");
  if (kwonlyargs || kwDefaults) {
    std::size_t kwonlyCount = kwonlyargs ? kwonlyargs->size() : 0;
    std::size_t defaultCount = kwDefaults ? kwDefaults->size() : 0;
    if (kwonlyCount != defaultCount) {
      error(callable, "keyword-only defaults length must match keyword-only "
                      "parameter count");
      ok = false;
    }
    for (std::size_t index = 0; index < kwonlyCount; ++index) {
      const parser::NodePtr &arg = (*kwonlyargs)[index];
      if (!arg) {
        ok = false;
        continue;
      }
      const std::string *argName = stringField(*arg, "arg");
      std::optional<mlir::Type> argType = parameterType(*arg);
      if (!argName || !argType) {
        error(*arg,
              "keyword-only arguments must have simple static annotations");
        ok = false;
        continue;
      }
      info.argNames.push_back(*argName);
      info.argTypes.push_back(*argType);
      info.kwonlyNames.push_back(*argName);
      info.kwonlyDefaultValues.push_back(
          kwDefaults && index < kwDefaults->size() ? (*kwDefaults)[index]
                                                   : parser::NodePtr{});
    }
  }
  return ok;
}

std::optional<mlir::Type> Builder::Impl::closureStorageType(mlir::Type type) {
  if (py::isPyType(type))
    return type;
  if (mlir::isa<mlir::ShapedType>(type))
    return std::nullopt;
  if (mlir::isa<mlir::FloatType>(type))
    return floatType();
  if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(type))
    return intTy.getWidth() == 1 ? mlir::Type(boolType())
                                 : mlir::Type(intType());
  return std::nullopt;
}

Value Builder::Impl::materializeClosureStorage(const parser::Node &anchor,
                                               const ClosureCapture &capture) {
  if (!capture.value.value) {
    error(anchor,
          "closure capture '" + capture.name + "' has no materialized value");
    return Value{{}, capture.storageType};
  }
  if (capture.value.type == capture.storageType)
    return capture.value;
  mlir::Value storage = builder.create<py::CastFromPrimOp>(
      loc(anchor), capture.storageType, capture.value.value);
  return Value{storage, capture.storageType};
}

Value Builder::Impl::restoreClosureValue(const parser::Node &anchor,
                                         mlir::Value storage,
                                         mlir::Type storageType,
                                         mlir::Type valueType) {
  if (storageType == valueType)
    return Value{storage, valueType};
  if (!py::isPyType(storageType)) {
    error(anchor, "closure storage type " + typeString(storageType) +
                      " cannot restore captured value type " +
                      typeString(valueType));
    return Value{{}, valueType};
  }
  mlir::Value value = builder.create<py::CastToPrimOp>(loc(anchor), valueType,
                                                       storage, "exact");
  return Value{value, valueType};
}

std::vector<ClosureCapture>
Builder::Impl::collectNestedFunctionCaptures(const parser::Node &function) {
  NestedCaptureCollector collector;
  collector.collect(function);
  if (!collector.unsupportedControl.empty()) {
    error(function, collector.unsupportedControl +
                        " statement is not supported in nested functions");
    return {};
  }

  std::vector<ClosureCapture> captures;
  for (const std::string &name : collector.loadedNames) {
    if (collector.globalNames.count(name))
      continue;
    if (collector.localNames.count(name) &&
        !collector.nonlocalNames.count(name))
      continue;
    auto symbol = symbols.find(name);
    if (symbol == symbols.end())
      continue;
    std::optional<mlir::Type> storageType =
        closureStorageType(symbol->second.type);
    if (!storageType) {
      error(function, "capturing value '" + name + "' of type " +
                          typeString(symbol->second.type) +
                          " in a nested closure is not supported yet");
      continue;
    }
    captures.push_back(ClosureCapture{name, symbol->second, *storageType});
  }
  return captures;
}

bool Builder::Impl::laterRebindsCapturedName(
    const parser::Node &function, llvm::ArrayRef<ClosureCapture> captures) {
  if (captures.empty() || functionStack.empty())
    return false;
  const parser::Node *enclosing = functionStack.back();
  const std::vector<parser::NodePtr> *body = nodeListField(*enclosing, "body");
  if (!body)
    return false;

  std::set<std::string> capturedNames;
  for (const ClosureCapture &capture : captures)
    capturedNames.insert(capture.name);

  for (const parser::NodePtr &stmt : *body) {
    if (!stmt || !locationAfter(stmt->range.start, function.range.start))
      continue;
    std::set<std::string> assigned;
    collectAssignedNames(*stmt, assigned);
    for (const std::string &name : assigned) {
      if (!capturedNames.count(name))
        continue;
      error(*stmt, "rebinding captured variable '" + name +
                       "' after nested function definition is not supported");
      return true;
    }
  }
  return false;
}

bool Builder::Impl::hasCallableMetadata(const FunctionInfo &info) const {
  if (info.varargType)
    return true;
  if (info.kwargType)
    return true;
  if (info.defaultValues.size() > 0)
    return true;
  if (!info.closureCaptures.empty())
    return true;
  return llvm::any_of(info.kwonlyDefaultValues,
                      [](const parser::NodePtr &node) { return !!node; });
}

bool Builder::Impl::hasCallableFormal(const FunctionInfo &info) const {
  return llvm::any_of(info.argTypes, [](mlir::Type type) {
    return mlir::isa<py::FuncType>(type);
  });
}

bool Builder::Impl::hasDefaultForFormal(const FunctionInfo &info,
                                        std::size_t formalIndex) const {
  if (formalIndex < info.positionalCount) {
    const std::size_t defaultCount = info.defaultValues.size();
    const std::size_t defaultStart = info.positionalCount >= defaultCount
                                         ? info.positionalCount - defaultCount
                                         : info.positionalCount;
    return formalIndex >= defaultStart && formalIndex < info.positionalCount &&
           info.defaultValues[formalIndex - defaultStart];
  }

  const std::size_t kwonlyIndex = formalIndex - info.positionalCount;
  return kwonlyIndex < info.kwonlyDefaultValues.size() &&
         info.kwonlyDefaultValues[kwonlyIndex];
}

Value Builder::Impl::emitFunctionDefaults(const parser::Node &anchor,
                                          const FunctionInfo &info) {
  if (info.defaultValues.empty())
    return Value{{}, {}};

  const std::size_t defaultCount = info.defaultValues.size();
  const std::size_t defaultStart = info.positionalCount >= defaultCount
                                       ? info.positionalCount - defaultCount
                                       : info.positionalCount;
  std::vector<Value> defaults;
  defaults.reserve(defaultCount);
  for (std::size_t index = 0; index < defaultCount; ++index) {
    const parser::NodePtr &defaultNode = info.defaultValues[index];
    if (!defaultNode) {
      error(anchor, "positional default metadata contains an empty value");
      return Value{{}, {}};
    }
    if (expressionMayThrow(*defaultNode)) {
      error(*defaultNode, "default argument for function '" + info.name +
                              "' must be nothrow for metadata materialization");
      return Value{{}, {}};
    }
    mlir::Type expected = info.argTypes[defaultStart + index];
    Value value = emitExpressionWithExpectedType(*defaultNode, expected);
    if (!value.value)
      return Value{{}, {}};
    if (!typeAssignable(expected, value.type)) {
      error(*defaultNode, "default argument for function '" + info.name +
                              "' must be " + typeString(expected) + ", got " +
                              typeString(value.type));
      return Value{{}, {}};
    }
    defaults.push_back(value);
  }
  return emitTuple(defaults);
}

Value Builder::Impl::emitFunctionKwdefaults(const parser::Node &anchor,
                                            const FunctionInfo &info) {
  if (info.kwonlyDefaultValues.empty())
    return Value{{}, {}};

  mlir::Type valueType;
  for (std::size_t index = 0; index < info.kwonlyDefaultValues.size();
       ++index) {
    if (!info.kwonlyDefaultValues[index])
      continue;
    mlir::Type expected = info.argTypes[info.positionalCount + index];
    if (!valueType) {
      valueType = expected;
      continue;
    }
    if (valueType != expected) {
      error(anchor, "keyword-only defaults for function '" + info.name +
                        "' currently require a single static value type");
      return Value{{}, {}};
    }
  }
  if (!valueType)
    return Value{{}, {}};

  mlir::Type resultType = dictType(strType(), valueType);
  mlir::Value dict = builder.create<py::DictEmptyOp>(loc(anchor), resultType);
  for (std::size_t index = 0; index < info.kwonlyDefaultValues.size();
       ++index) {
    const parser::NodePtr &defaultNode = info.kwonlyDefaultValues[index];
    if (!defaultNode)
      continue;
    if (expressionMayThrow(*defaultNode)) {
      error(*defaultNode, "keyword-only default for function '" + info.name +
                              "' must be nothrow for metadata materialization");
      return Value{{}, {}};
    }
    mlir::Type expected = info.argTypes[info.positionalCount + index];
    Value value = emitExpressionWithExpectedType(*defaultNode, expected);
    if (!value.value)
      return Value{{}, {}};
    if (!typeAssignable(expected, value.type)) {
      error(*defaultNode, "keyword-only default for function '" + info.name +
                              "' must be " + typeString(expected) + ", got " +
                              typeString(value.type));
      return Value{{}, {}};
    }
    mlir::Value key = builder.create<py::StrConstantOp>(
        loc(*defaultNode), strType(), info.kwonlyNames[index]);
    builder.create<py::DictInsertOp>(loc(*defaultNode), dict, key, value.value);
  }
  return Value{dict, resultType};
}

Value Builder::Impl::emitFunctionClosure(const parser::Node &anchor,
                                         const FunctionInfo &info) {
  if (info.closureCaptures.empty())
    return Value{{}, {}};
  std::vector<Value> closureValues;
  closureValues.reserve(info.closureCaptures.size());
  for (const ClosureCapture &capture : info.closureCaptures) {
    Value storage = materializeClosureStorage(anchor, capture);
    if (!storage.value)
      return Value{{}, {}};
    closureValues.push_back(storage);
  }
  return emitTuple(closureValues);
}

Value Builder::Impl::emitFunctionObject(const parser::Node &anchor,
                                        const FunctionInfo &info) {
  bool needsMakeFunction =
      !info.defaultValues.empty() || !info.closureCaptures.empty() ||
      llvm::any_of(info.kwonlyDefaultValues,
                   [](const parser::NodePtr &node) { return !!node; });
  if (info.isNative || info.isAsync || !needsMakeFunction) {
    mlir::Value materialized = builder.create<py::FuncObjectOp>(
        loc(anchor), info.functionType, info.symbolName);
    return Value{materialized, info.functionType};
  }

  Value defaults = emitFunctionDefaults(anchor, info);
  if (!defaults.value && !info.defaultValues.empty())
    return Value{{}, info.functionType};
  Value kwdefaults = emitFunctionKwdefaults(anchor, info);
  if (!kwdefaults.value &&
      llvm::any_of(info.kwonlyDefaultValues,
                   [](const parser::NodePtr &node) { return !!node; }))
    return Value{{}, info.functionType};
  Value closure = emitFunctionClosure(anchor, info);
  if (!closure.value && !info.closureCaptures.empty())
    return Value{{}, info.functionType};

  mlir::Value materialized = builder.create<py::MakeFunctionOp>(
      loc(anchor), info.functionType, info.symbolName, defaults.value,
      kwdefaults.value, closure.value, mlir::Value{}, mlir::Value{});
  return Value{materialized, info.functionType};
}

void Builder::Impl::emitFunctionBinding(const parser::Node &function) {
  if (!inModuleMain)
    return;
  const std::string *name = stringField(function, "name");
  if (!name)
    return;
  auto found = functions.find(*name);
  if (found == functions.end())
    return;
  if (found->second.isNative || found->second.isAsync)
    return;
  if (hasCallableFormal(found->second))
    return;
  Value functionValue = emitFunctionObject(function, found->second);
  if (!functionValue.value)
    return;
  symbols[*name] = functionValue;
  callableAliases[*name] = found->second;
}

std::optional<FunctionInfo>
Builder::Impl::parseFunctionInfo(const parser::Node &function) {
  const std::string *name = stringField(function, "name");
  const parser::NodePtr *argsNode = nodeField(function, "args");
  if (!name || !argsNode || !*argsNode) {
    error(function, "FunctionDef.name or FunctionDef.args is missing");
    return std::nullopt;
  }
  if (hasTypeParams(function)) {
    error(function, "generic function type parameters are parsed from CPython "
                    "3.14 syntax but static specialization is not implemented "
                    "in the C++ emitter yet");
    return std::nullopt;
  }

  FunctionInfo info;
  info.name = *name;
  info.isAsync = function.kind == "AsyncFunctionDef";
  info.symbolName = *name == "main" ? (info.isAsync ? "__lython_async_main"
                                                    : "__lython_user_main")
                                    : *name;
  const std::vector<parser::NodePtr> *decorators =
      nodeListField(function, "decorator_list");
  bool unsupportedDecorator = false;
  if (decorators) {
    for (const parser::NodePtr &decorator : *decorators) {
      if (!decorator)
        continue;
      if (info.isAsync) {
        error(*decorator, "decorators on async functions are not supported");
        unsupportedDecorator = true;
        continue;
      }
      if (isNativeDecorator(*decorator)) {
        info.isNative = true;
        continue;
      }
      error(*decorator, "function decorators other than @native(gc=\"none\") "
                        "are parsed but not implemented in the C++ emitter "
                        "yet");
      unsupportedDecorator = true;
    }
  }
  if (unsupportedDecorator)
    return std::nullopt;

  std::optional<FunctionTypeComment> typeComment =
      parseFunctionTypeComment(function);
  if (stringField(function, "type_comment") && !typeComment)
    return std::nullopt;
  std::size_t typeCommentArgIndex = 0;
  auto nextCommentArgType = [&]() -> mlir::Type {
    if (!typeComment || typeCommentArgIndex >= typeComment->argTypes.size())
      return {};
    return typeComment->argTypes[typeCommentArgIndex++];
  };

  const std::vector<parser::NodePtr> *args = nodeListField(**argsNode, "args");
  const std::vector<parser::NodePtr> *posonlyargs =
      nodeListField(**argsNode, "posonlyargs");
  if (!args) {
    error(**argsNode, "arguments.args is missing");
    return std::nullopt;
  }
  std::size_t positionalParameterCount =
      (posonlyargs ? posonlyargs->size() : 0) + args->size();
  if (typeComment && typeComment->argTypes.size() != positionalParameterCount) {
    error(function, "function type_comment argument count does not match "
                    "function parameters");
    return std::nullopt;
  }

  if (posonlyargs) {
    for (const parser::NodePtr &arg : *posonlyargs) {
      if (!arg)
        continue;
      if (!appendAnnotatedParameter(*arg, info, "positional-only argument",
                                    nextCommentArgType()))
        return std::nullopt;
    }
  }
  info.positionalOnlyCount = info.argNames.size();
  for (const parser::NodePtr &arg : *args) {
    if (!arg)
      continue;
    if (!appendAnnotatedParameter(*arg, info, "function argument",
                                  nextCommentArgType()))
      return std::nullopt;
  }
  info.positionalCount = info.argNames.size();
  if (!collectCallableDefaults(function, **argsNode, info))
    return std::nullopt;
  if ((info.varargType || info.kwargType) && (info.isNative || info.isAsync)) {
    error(function,
          "*args/**kwargs are not supported on native or async functions yet");
    return std::nullopt;
  }

  const parser::NodePtr *returns = nodeField(function, "returns");
  std::optional<mlir::Type> resultType;
  if (returns && *returns)
    resultType = typeFromAnnotation(*returns);
  else if (typeComment)
    resultType = typeComment->resultType;
  else
    resultType = noneType();
  if (!resultType) {
    error(function, "function return annotation is unsupported");
    return std::nullopt;
  }
  info.resultType = *resultType;
  llvm::ArrayRef<mlir::Type> allArgTypes(info.argTypes);
  info.signatureType = functionSignatureType(
      allArgTypes.take_front(info.positionalCount), info.resultType,
      info.varargType, allArgTypes.drop_front(info.positionalCount),
      info.kwargType);
  info.functionType = py::FuncType::get(&context, info.signatureType);
  if (info.isAsync)
    info.asyncFunctionType = asyncFunctionType(info.argTypes, info.resultType);
  llvm::SmallVector<mlir::Type> nativeResults;
  if (info.resultType != noneType())
    nativeResults.push_back(info.resultType);
  info.nativeFunctionType =
      mlir::FunctionType::get(&context, info.argTypes, nativeResults);
  return info;
}

std::optional<FunctionInfo>
Builder::Impl::parseLambdaInfo(const parser::Node &lambda,
                               py::FuncType expectedType) {
  const parser::NodePtr *argsNode = nodeField(lambda, "args");
  const parser::NodePtr *bodyNode = nodeField(lambda, "body");
  if (!argsNode || !*argsNode || !bodyNode || !*bodyNode) {
    error(lambda, "Lambda.args or Lambda.body is missing");
    return std::nullopt;
  }

  py::FuncSignatureType signature = expectedType.getSignature();
  if (signature.hasVararg() || signature.hasKwarg() ||
      !signature.getKwOnlyTypes().empty()) {
    error(lambda, "lambda expected Callable type must be a fixed positional "
                  "signature");
    return std::nullopt;
  }
  llvm::ArrayRef<mlir::Type> positionalTypes = signature.getPositionalTypes();
  llvm::ArrayRef<mlir::Type> resultTypes = signature.getResultTypes();
  if (resultTypes.size() != 1) {
    error(lambda, "lambda expected Callable type must have exactly one "
                  "result type");
    return std::nullopt;
  }

  auto rejectOptionalArg = [&](llvm::StringRef fieldName,
                               llvm::StringRef message) -> bool {
    const parser::NodePtr *node = nodeField(**argsNode, fieldName);
    if (node && *node) {
      error(**node, message.str());
      return true;
    }
    return false;
  };
  if (rejectOptionalArg("vararg", "lambda *args are not supported by the "
                                  "C++ emitter yet") ||
      rejectOptionalArg("kwarg", "lambda **kwargs are not supported by the "
                                 "C++ emitter yet"))
    return std::nullopt;

  const std::vector<parser::NodePtr> *posonlyargs =
      nodeListField(**argsNode, "posonlyargs");
  const std::vector<parser::NodePtr> *args = nodeListField(**argsNode, "args");
  const std::vector<parser::NodePtr> *kwonlyargs =
      nodeListField(**argsNode, "kwonlyargs");
  if (!args) {
    error(**argsNode, "lambda arguments.args is missing");
    return std::nullopt;
  }
  if (kwonlyargs && !kwonlyargs->empty()) {
    error(**argsNode, "lambda keyword-only parameters require named Callable "
                      "metadata and are not supported yet");
    return std::nullopt;
  }

  const std::size_t posonlyCount = posonlyargs ? posonlyargs->size() : 0;
  const std::size_t positionalCount = posonlyCount + args->size();
  if (positionalCount != positionalTypes.size()) {
    error(lambda, "lambda expected Callable argument count is " +
                      std::to_string(positionalTypes.size()) + ", got " +
                      std::to_string(positionalCount));
    return std::nullopt;
  }

  FunctionInfo info;
  info.name = "<lambda>";
  info.symbolName =
      "__lython_lambda_" + std::to_string(++nestedFunctionCounter);
  info.positionalOnlyCount = posonlyCount;
  info.positionalCount = positionalCount;
  info.resultType = resultTypes.front();
  info.signatureType = signature;
  info.functionType = expectedType;
  info.nativeFunctionType = mlir::FunctionType::get(
      &context, positionalTypes,
      info.resultType == noneType() ? mlir::TypeRange{}
                                    : mlir::TypeRange{info.resultType});
  info.mayThrow = expressionMayThrow(**bodyNode);

  std::set<std::string> seenNames;
  auto appendLambdaArg = [&](const parser::NodePtr &arg,
                             std::size_t index) -> bool {
    if (!arg)
      return false;
    const std::string *name = stringField(*arg, "arg");
    if (!name) {
      error(*arg, "lambda parameter name is missing");
      return false;
    }
    if (!seenNames.insert(*name).second) {
      error(*arg, "duplicate lambda parameter '" + *name + "'");
      return false;
    }
    info.argNames.push_back(*name);
    info.argTypes.push_back(positionalTypes[index]);
    return true;
  };

  std::size_t index = 0;
  if (posonlyargs)
    for (const parser::NodePtr &arg : *posonlyargs)
      if (!appendLambdaArg(arg, index++))
        return std::nullopt;
  for (const parser::NodePtr &arg : *args)
    if (!appendLambdaArg(arg, index++))
      return std::nullopt;

  const std::vector<parser::NodePtr> *defaults =
      nodeListField(**argsNode, "defaults");
  if (defaults) {
    if (defaults->size() > info.positionalCount) {
      error(lambda, "too many lambda default argument values");
      return std::nullopt;
    }
    info.defaultValues.assign(defaults->begin(), defaults->end());
  }
  return info;
}

void Builder::Impl::scanFunctions(const parser::Node &moduleNode) {
  const std::vector<parser::NodePtr> *body = nodeListField(moduleNode, "body");
  if (!body)
    return;
  for (const parser::NodePtr &stmt : *body) {
    if (!stmt ||
        (stmt->kind != "FunctionDef" && stmt->kind != "AsyncFunctionDef"))
      continue;
    std::optional<FunctionInfo> info = parseFunctionInfo(*stmt);
    if (!info)
      continue;
    if (functions.count(info->name)) {
      error(*stmt, "duplicate function definition '" + info->name + "'");
      continue;
    }
    callableFunctionsBySymbol[info->symbolName] = *info;
    functions.emplace(info->name, std::move(*info));
  }
  scanReturnedCallableSummaries(moduleNode);
  scanMethodReturnedCallableSummaries();
  propagateMayThrow(moduleNode);
}

void Builder::Impl::scanReturnedCallableSummaries(
    const parser::Node &moduleNode) {
  const std::vector<parser::NodePtr> *body = nodeListField(moduleNode, "body");
  if (!body)
    return;
  for (const parser::NodePtr &stmt : *body) {
    if (!stmt ||
        (stmt->kind != "FunctionDef" && stmt->kind != "AsyncFunctionDef"))
      continue;
    const std::string *name = stringField(*stmt, "name");
    if (!name)
      continue;
    auto found = functions.find(*name);
    if (found == functions.end())
      continue;
    if (!mlir::isa<py::FuncType>(found->second.resultType))
      continue;
    found->second.returnedCallable =
        directReturnedNestedCallable(found->second, *stmt);
    if (found->second.returnedCallable) {
      found->second.returnedCallableSymbolName =
          found->second.returnedCallable->info.symbolName;
      callableFunctionsBySymbol[found->second.returnedCallable->info
                                    .symbolName] =
          found->second.returnedCallable->info;
      callableFunctionsBySymbol[found->second.symbolName] = found->second;
      continue;
    }
    found->second.returnedCallableArgIndex =
        directReturnedCallableArgIndex(found->second, *stmt);
    if (found->second.returnedCallableArgIndex) {
      callableFunctionsBySymbol[found->second.symbolName] = found->second;
      continue;
    }
    if (std::optional<FunctionInfo> nested =
            directReturnedNestedCallableMetadata(found->second, *stmt)) {
      found->second.returnedCallableSymbolName = nested->symbolName;
      callableFunctionsBySymbol[nested->symbolName] = *nested;
      callableFunctionsBySymbol[found->second.symbolName] = found->second;
      continue;
    }
    found->second.returnedCallableSymbolName =
        directReturnedCallableSymbol(*stmt);
    callableFunctionsBySymbol[found->second.symbolName] = found->second;
  }
}

void Builder::Impl::scanMethodReturnedCallableSummaries() {
  for (auto &[className, classInfo] : classes) {
    (void)className;
    for (const auto &[methodName, methodNode] : classInfo.methodNodes) {
      if (!methodNode)
        continue;
      auto methodSymbol = classInfo.methods.find(methodName);
      if (methodSymbol == classInfo.methods.end())
        continue;
      auto function = functions.find(methodSymbol->second);
      if (function == functions.end())
        continue;
      FunctionInfo &info = function->second;
      if (!mlir::isa<py::FuncType>(info.resultType)) {
        callableFunctionsBySymbol[info.symbolName] = info;
        continue;
      }

      info.returnedCallable.reset();
      info.returnedCallableSymbolName.reset();
      info.returnedCallableArgIndex.reset();

      info.returnedCallable = directReturnedNestedCallable(info, *methodNode);
      if (info.returnedCallable) {
        info.returnedCallableSymbolName =
            info.returnedCallable->info.symbolName;
        callableFunctionsBySymbol[info.returnedCallable->info.symbolName] =
            info.returnedCallable->info;
        callableFunctionsBySymbol[info.symbolName] = info;
        continue;
      }

      info.returnedCallableArgIndex =
          directReturnedCallableArgIndex(info, *methodNode);
      if (info.returnedCallableArgIndex) {
        callableFunctionsBySymbol[info.symbolName] = info;
        continue;
      }
      if (std::optional<FunctionInfo> nested =
              directReturnedNestedCallableMetadata(info, *methodNode)) {
        info.returnedCallableSymbolName = nested->symbolName;
        callableFunctionsBySymbol[nested->symbolName] = *nested;
      } else {
        info.returnedCallableSymbolName =
            directReturnedCallableSymbol(*methodNode);
      }
      callableFunctionsBySymbol[info.symbolName] = info;
    }
  }
}

std::shared_ptr<ReturnedCallableSummary>
Builder::Impl::directReturnedNestedCallable(const FunctionInfo &outer,
                                            const parser::Node &function) {
  const std::vector<parser::NodePtr> *body = nodeListField(function, "body");
  if (!body)
    return nullptr;

  std::optional<std::string> returnedName;
  bool sawReturn = false;
  for (const parser::NodePtr &stmt : *body) {
    if (!stmt)
      continue;
    if (stmt->kind == "FunctionDef" || stmt->kind == "AsyncFunctionDef")
      continue;
    if (stmt->kind != "Return")
      return nullptr;

    const parser::NodePtr *valueNode = nodeField(*stmt, "value");
    if (!valueNode || !*valueNode || (*valueNode)->kind != "Name")
      return nullptr;
    const std::string *name = stringField(**valueNode, "id");
    if (!name)
      return nullptr;
    if (!sawReturn) {
      returnedName = *name;
      sawReturn = true;
      continue;
    }
    if (returnedName != *name)
      return nullptr;
  }
  if (!returnedName)
    return nullptr;

  const parser::Node *nested = nullptr;
  for (const parser::NodePtr &stmt : *body) {
    if (!stmt || stmt->kind != "FunctionDef")
      continue;
    const std::string *name = stringField(*stmt, "name");
    if (name && *name == *returnedName) {
      nested = stmt.get();
      break;
    }
  }
  if (!nested)
    return nullptr;

  std::optional<FunctionInfo> nestedInfo = parseFunctionInfo(*nested);
  if (!nestedInfo || nestedInfo->isNative || nestedInfo->isAsync)
    return nullptr;
  nestedInfo->symbolName = outer.symbolName + "__" + *returnedName;
  nestedInfo->requiresCallableValue = true;

  NestedCaptureCollector collector;
  collector.collect(*nested);
  if (!collector.unsupportedControl.empty())
    return nullptr;

  std::vector<std::optional<std::size_t>> captureArgIndices;
  for (const std::string &name : collector.loadedNames) {
    if (collector.globalNames.count(name))
      continue;
    if (collector.localNames.count(name) &&
        !collector.nonlocalNames.count(name))
      continue;
    auto argIt = std::find(outer.argNames.begin(), outer.argNames.end(), name);
    if (argIt == outer.argNames.end())
      return nullptr;
    std::size_t argIndex =
        static_cast<std::size_t>(std::distance(outer.argNames.begin(), argIt));
    if (argIndex >= outer.argTypes.size())
      return nullptr;
    if (mlir::isa<py::ClassType>(outer.argTypes[argIndex]))
      return nullptr;
    std::optional<mlir::Type> storageType =
        closureStorageType(outer.argTypes[argIndex]);
    if (!storageType)
      return nullptr;
    nestedInfo->closureCaptures.push_back(ClosureCapture{
        name, Value{{}, outer.argTypes[argIndex]}, *storageType});
    captureArgIndices.push_back(argIndex);
  }

  const std::vector<parser::NodePtr> *nestedBody =
      nodeListField(*nested, "body");
  if (nestedBody && statementListMayThrow(*nestedBody))
    nestedInfo->mayThrow = true;

  auto summary = std::make_shared<ReturnedCallableSummary>();
  summary->info = std::move(*nestedInfo);
  summary->closureCaptureArgIndices = std::move(captureArgIndices);
  return summary;
}

std::optional<FunctionInfo> Builder::Impl::directReturnedNestedCallableMetadata(
    const FunctionInfo &outer, const parser::Node &function) {
  const std::vector<parser::NodePtr> *body = nodeListField(function, "body");
  if (!body)
    return std::nullopt;

  std::optional<std::string> returnedName;
  bool sawReturn = false;
  for (const parser::NodePtr &stmt : *body) {
    if (!stmt)
      continue;
    if (stmt->kind == "FunctionDef" || stmt->kind == "AsyncFunctionDef" ||
        stmt->kind == "AnnAssign" || stmt->kind == "Assign" ||
        stmt->kind == "Expr" || stmt->kind == "Pass")
      continue;
    if (stmt->kind != "Return")
      return std::nullopt;

    const parser::NodePtr *valueNode = nodeField(*stmt, "value");
    if (!valueNode || !*valueNode || (*valueNode)->kind != "Name")
      return std::nullopt;
    const std::string *name = stringField(**valueNode, "id");
    if (!name)
      return std::nullopt;
    if (!sawReturn) {
      returnedName = *name;
      sawReturn = true;
      continue;
    }
    if (returnedName != *name)
      return std::nullopt;
  }
  if (!returnedName)
    return std::nullopt;

  const parser::Node *nested = nullptr;
  for (const parser::NodePtr &stmt : *body) {
    if (!stmt || stmt->kind != "FunctionDef")
      continue;
    const std::string *name = stringField(*stmt, "name");
    if (name && *name == *returnedName) {
      nested = stmt.get();
      break;
    }
  }
  if (!nested)
    return std::nullopt;

  std::optional<FunctionInfo> nestedInfo = parseFunctionInfo(*nested);
  if (!nestedInfo || nestedInfo->isNative || nestedInfo->isAsync)
    return std::nullopt;
  nestedInfo->symbolName = outer.symbolName + "__" + *returnedName;
  nestedInfo->requiresCallableValue = true;
  nestedInfo->closureCaptures.clear();

  const std::vector<parser::NodePtr> *nestedBody =
      nodeListField(*nested, "body");
  if (nestedBody && statementListMayThrow(*nestedBody))
    nestedInfo->mayThrow = true;
  return nestedInfo;
}

std::optional<std::string> Builder::Impl::directReturnedCallableSymbol(
    const parser::Node &function) const {
  const std::vector<parser::NodePtr> *body = nodeListField(function, "body");
  if (!body)
    return std::nullopt;

  std::optional<std::string> returnedSymbol;
  bool sawReturn = false;
  for (const parser::NodePtr &stmt : *body) {
    if (!stmt)
      continue;
    if (stmt->kind == "FunctionDef" || stmt->kind == "AsyncFunctionDef")
      continue;
    if (stmt->kind != "Return")
      continue;

    const parser::NodePtr *valueNode = nodeField(*stmt, "value");
    if (!valueNode || !*valueNode)
      return std::nullopt;
    std::optional<FunctionInfo> info = resolveCallableInfo(**valueNode);
    if (!info)
      return std::nullopt;
    if (!sawReturn) {
      returnedSymbol = info->symbolName;
      sawReturn = true;
      continue;
    }
    if (returnedSymbol != info->symbolName)
      return std::nullopt;
  }
  return sawReturn ? returnedSymbol : std::nullopt;
}

std::optional<std::size_t> Builder::Impl::directReturnedCallableArgIndex(
    const FunctionInfo &info, const parser::Node &function) const {
  const std::vector<parser::NodePtr> *body = nodeListField(function, "body");
  if (!body)
    return std::nullopt;

  std::optional<std::size_t> returnedIndex;
  bool sawReturn = false;
  for (const parser::NodePtr &stmt : *body) {
    if (!stmt)
      continue;
    if (stmt->kind != "Return")
      return std::nullopt;

    const parser::NodePtr *valueNode = nodeField(*stmt, "value");
    if (!valueNode || !*valueNode || (*valueNode)->kind != "Name")
      return std::nullopt;
    const std::string *name = stringField(**valueNode, "id");
    if (!name)
      return std::nullopt;

    auto argIt = std::find(info.argNames.begin(), info.argNames.end(), *name);
    if (argIt == info.argNames.end())
      return std::nullopt;
    std::size_t index =
        static_cast<std::size_t>(std::distance(info.argNames.begin(), argIt));
    if (index >= info.argTypes.size() ||
        !mlir::isa<py::FuncType>(info.argTypes[index]))
      return std::nullopt;

    if (!sawReturn) {
      returnedIndex = index;
      sawReturn = true;
      continue;
    }
    if (returnedIndex != index)
      return std::nullopt;
  }
  return sawReturn ? returnedIndex : std::nullopt;
}

bool Builder::Impl::expressionMayThrow(const parser::Node &expr) const {
  if (expr.kind == "Constant" || expr.kind == "Name")
    return false;
  if (expr.kind == "BoolOp") {
    const std::vector<parser::NodePtr> *values = nodeListField(expr, "values");
    return values && llvm::any_of(*values, [&](const parser::NodePtr &node) {
             return node && expressionMayThrow(*node);
           });
  }
  if (expr.kind == "IfExp") {
    const parser::NodePtr *test = nodeField(expr, "test");
    const parser::NodePtr *body = nodeField(expr, "body");
    const parser::NodePtr *orelse = nodeField(expr, "orelse");
    return (test && *test && expressionMayThrow(**test)) ||
           (body && *body && expressionMayThrow(**body)) ||
           (orelse && *orelse && expressionMayThrow(**orelse));
  }
  if (expr.kind == "BinOp") {
    const parser::NodePtr *lhs = nodeField(expr, "left");
    const parser::NodePtr *rhs = nodeField(expr, "right");
    return (lhs && *lhs && expressionMayThrow(**lhs)) ||
           (rhs && *rhs && expressionMayThrow(**rhs));
  }
  if (expr.kind == "UnaryOp") {
    const parser::NodePtr *operand = nodeField(expr, "operand");
    return operand && *operand && expressionMayThrow(**operand);
  }
  if (expr.kind == "Await")
    return true;
  if (expr.kind == "Compare") {
    const parser::NodePtr *lhs = nodeField(expr, "left");
    if (lhs && *lhs && expressionMayThrow(**lhs))
      return true;
    const std::vector<parser::NodePtr> *comparators =
        nodeListField(expr, "comparators");
    if (!comparators)
      return false;
    return llvm::any_of(*comparators, [&](const parser::NodePtr &node) {
      return node && expressionMayThrow(*node);
    });
  }
  if (expr.kind == "Call") {
    const parser::NodePtr *func = nodeField(expr, "func");
    const std::vector<parser::NodePtr> *args = nodeListField(expr, "args");
    const std::vector<parser::NodePtr> *keywords =
        nodeListField(expr, "keywords");
    bool argsMayThrow =
        args && llvm::any_of(*args, [&](const parser::NodePtr &node) {
          return node && expressionMayThrow(*node);
        });
    if (argsMayThrow)
      return true;
    bool keywordsMayThrow =
        keywords && llvm::any_of(*keywords, [&](const parser::NodePtr &node) {
          if (!node)
            return false;
          const parser::NodePtr *value = nodeField(*node, "value");
          return value && *value && expressionMayThrow(**value);
        });
    if (keywordsMayThrow)
      return true;
    if (func && *func && expressionMayThrow(**func))
      return true;
    if (func && *func && (*func)->kind == "Name") {
      const std::string *name = stringField(**func, "id");
      if (name) {
        auto found = functions.find(*name);
        if (found != functions.end() && found->second.mayThrow)
          return true;
        auto classFound = classes.find(*name);
        if (classFound != classes.end()) {
          auto initFound = classFound->second.methods.find("__init__");
          if (initFound == classFound->second.methods.end())
            return false;
          auto initFunction = functions.find(initFound->second);
          return initFunction != functions.end() &&
                 initFunction->second.mayThrow;
        }
      }
    }
    return false;
  }
  if (expr.kind == "Attribute") {
    const parser::NodePtr *value = nodeField(expr, "value");
    return value && *value && expressionMayThrow(**value);
  }
  if (expr.kind == "Subscript") {
    const parser::NodePtr *value = nodeField(expr, "value");
    const parser::NodePtr *slice = nodeField(expr, "slice");
    return (value && *value && expressionMayThrow(**value)) ||
           (slice && *slice && expressionMayThrow(**slice));
  }
  if (expr.kind == "List" || expr.kind == "Tuple") {
    const std::vector<parser::NodePtr> *elements = nodeListField(expr, "elts");
    return elements &&
           llvm::any_of(*elements, [&](const parser::NodePtr &node) {
             return node && expressionMayThrow(*node);
           });
  }
  if (expr.kind == "Dict") {
    const std::vector<parser::NodePtr> *keys = nodeListField(expr, "keys");
    const std::vector<parser::NodePtr> *values = nodeListField(expr, "values");
    return (keys && llvm::any_of(*keys,
                                 [&](const parser::NodePtr &node) {
                                   return node && expressionMayThrow(*node);
                                 })) ||
           (values && llvm::any_of(*values, [&](const parser::NodePtr &node) {
              return node && expressionMayThrow(*node);
            }));
  }
  return false;
}

bool Builder::Impl::statementMayThrow(const parser::Node &stmt) const {
  if (stmt.kind == "Raise" || stmt.kind == "Assert")
    return true;
  if (stmt.kind == "Assign") {
    const parser::NodePtr *value = nodeField(stmt, "value");
    return value && *value && expressionMayThrow(**value);
  }
  if (stmt.kind == "AnnAssign") {
    const parser::NodePtr *value = nodeField(stmt, "value");
    return value && *value && expressionMayThrow(**value);
  }
  if (stmt.kind == "AugAssign") {
    const parser::NodePtr *target = nodeField(stmt, "target");
    const parser::NodePtr *value = nodeField(stmt, "value");
    return (target && *target && expressionMayThrow(**target)) ||
           (value && *value && expressionMayThrow(**value));
  }
  if (stmt.kind == "Expr") {
    const parser::NodePtr *value = nodeField(stmt, "value");
    return value && *value && expressionMayThrow(**value);
  }
  if (stmt.kind == "Return") {
    const parser::NodePtr *value = nodeField(stmt, "value");
    return value && *value && expressionMayThrow(**value);
  }
  if (stmt.kind == "If") {
    const parser::NodePtr *test = nodeField(stmt, "test");
    const std::vector<parser::NodePtr> *body = nodeListField(stmt, "body");
    const std::vector<parser::NodePtr> *orelse = nodeListField(stmt, "orelse");
    return (test && *test && expressionMayThrow(**test)) ||
           (body && statementListMayThrow(*body)) ||
           (orelse && statementListMayThrow(*orelse));
  }
  if (stmt.kind == "While") {
    const parser::NodePtr *test = nodeField(stmt, "test");
    const std::vector<parser::NodePtr> *body = nodeListField(stmt, "body");
    const std::vector<parser::NodePtr> *orelse = nodeListField(stmt, "orelse");
    return (test && *test && expressionMayThrow(**test)) ||
           (body && statementListMayThrow(*body)) ||
           (orelse && statementListMayThrow(*orelse));
  }
  if (stmt.kind == "Try") {
    const std::vector<parser::NodePtr> *body = nodeListField(stmt, "body");
    const std::vector<parser::NodePtr> *handlers =
        nodeListField(stmt, "handlers");
    const std::vector<parser::NodePtr> *orelse = nodeListField(stmt, "orelse");
    const std::vector<parser::NodePtr> *finalbody =
        nodeListField(stmt, "finalbody");
    bool hasHandlers = handlers && !handlers->empty();
    bool handlersMayThrow =
        handlers && llvm::any_of(*handlers, [&](const parser::NodePtr &node) {
          const std::vector<parser::NodePtr> *body =
              node ? nodeListField(*node, "body") : nullptr;
          return body && statementListMayThrow(*body);
        });
    return (!hasHandlers && body && statementListMayThrow(*body)) ||
           handlersMayThrow || (orelse && statementListMayThrow(*orelse)) ||
           (finalbody && statementListMayThrow(*finalbody));
  }
  if (stmt.kind == "With") {
    const std::vector<parser::NodePtr> *items = nodeListField(stmt, "items");
    const std::vector<parser::NodePtr> *body = nodeListField(stmt, "body");
    bool itemMayThrow =
        items && llvm::any_of(*items, [&](const parser::NodePtr &item) {
          if (!item)
            return false;
          const parser::NodePtr *contextExpr = nodeField(*item, "context_expr");
          return contextExpr && *contextExpr &&
                 expressionMayThrow(**contextExpr);
        });
    return itemMayThrow || (body && statementListMayThrow(*body));
  }
  return false;
}

bool Builder::Impl::statementListMayThrow(
    const std::vector<parser::NodePtr> &statements) const {
  return llvm::any_of(statements, [&](const parser::NodePtr &stmt) {
    return stmt && statementMayThrow(*stmt);
  });
}

void Builder::Impl::propagateMayThrow(const parser::Node &moduleNode) {
  const std::vector<parser::NodePtr> *body = nodeListField(moduleNode, "body");
  if (!body)
    return;

  bool changed = true;
  while (changed) {
    changed = false;
    for (const parser::NodePtr &stmt : *body) {
      if (!stmt ||
          (stmt->kind != "FunctionDef" && stmt->kind != "AsyncFunctionDef"))
        continue;
      const std::string *name = stringField(*stmt, "name");
      const std::vector<parser::NodePtr> *functionBody =
          nodeListField(*stmt, "body");
      if (!name || !functionBody)
        continue;
      auto found = functions.find(*name);
      if (found == functions.end() || found->second.mayThrow)
        continue;
      if (statementListMayThrow(*functionBody)) {
        found->second.mayThrow = true;
        changed = true;
      }
    }
    for (const parser::NodePtr &stmt : *body) {
      if (!stmt || stmt->kind != "ClassDef")
        continue;
      const std::string *className = stringField(*stmt, "name");
      const std::vector<parser::NodePtr> *classBody =
          nodeListField(*stmt, "body");
      if (!className || !classBody)
        continue;
      for (const parser::NodePtr &member : *classBody) {
        if (!member || member->kind != "FunctionDef")
          continue;
        const std::string *methodName = stringField(*member, "name");
        const std::vector<parser::NodePtr> *methodBody =
            nodeListField(*member, "body");
        if (!methodName || !methodBody)
          continue;
        auto found = functions.find(*className + "." + *methodName);
        if (found == functions.end() || found->second.mayThrow)
          continue;
        if (statementListMayThrow(*methodBody)) {
          found->second.mayThrow = true;
          changed = true;
        }
      }
    }
  }
}

void Builder::Impl::emitMain(const parser::Node &moduleNode) {
  py::FuncSignatureType signature = functionSignatureType({}, i32Type());
  const std::vector<parser::NodePtr> *body = nodeListField(moduleNode, "body");
  bool mainMayThrow = body && statementListMayThrow(*body);
  py::FuncOp main = createFunc("main", signature, {}, /*hasVararg=*/false,
                               /*hasKwarg=*/false, mainMayThrow);
  addEntryBlock(main, {});

  mlir::Type savedReturnType = currentReturnType;
  bool savedTerminated = blockTerminated;
  bool savedInModuleMain = inModuleMain;
  currentReturnType = i32Type();
  blockTerminated = false;
  inModuleMain = true;

  if (!body) {
    error(moduleNode, "Module.body is missing");
  } else {
    for (const parser::NodePtr &stmt : *body) {
      if (stmt && !blockTerminated)
        emitStatement(*stmt);
    }
  }

  if (!blockTerminated) {
    mlir::Value zero = builder.create<mlir::arith::ConstantIntOp>(loc(), 0, 32);
    builder.create<py::ReturnOp>(loc(), mlir::ValueRange{zero});
  }
  currentReturnType = savedReturnType;
  blockTerminated = savedTerminated;
  inModuleMain = savedInModuleMain;
  builder.setInsertionPointToEnd(module->getBody());
}

void Builder::Impl::emitUserFunctions(const parser::Node &moduleNode) {
  const std::vector<parser::NodePtr> *body = nodeListField(moduleNode, "body");
  if (!body)
    return;
  for (const parser::NodePtr &stmt : *body) {
    if (!stmt)
      continue;
    if (stmt->kind == "FunctionDef")
      emitFunctionDef(*stmt);
    else if (stmt->kind == "AsyncFunctionDef")
      emitAsyncFunctionDef(*stmt);
  }

  auto emitPendingSpecializations = [&]() -> bool {
    bool emittedAny = false;
    for (const parser::NodePtr &stmt : *body) {
      if (!stmt || stmt->kind != "FunctionDef")
        continue;
      const std::string *name = stringField(*stmt, "name");
      if (!name)
        continue;
      auto found = functions.find(*name);
      if (found == functions.end() || !hasCallableFormal(found->second))
        continue;

      std::vector<std::string> pendingKeys;
      for (const auto &entry : functionSpecializations) {
        const FunctionSpecialization &specialization = entry.second;
        if (specialization.info.name != *name)
          continue;
        if (emittedFunctionSpecializations.count(entry.first))
          continue;
        pendingKeys.push_back(entry.first);
      }

      for (const std::string &key : pendingKeys) {
        auto specialization = functionSpecializations.find(key);
        if (specialization == functionSpecializations.end())
          continue;
        emittedFunctionSpecializations.insert(key);
        emitFunctionBody(*stmt, specialization->second.info,
                         specialization->second.callableAliases);
        emittedAny = true;
      }
    }
    return emittedAny;
  };

  while (emitPendingSpecializations()) {
  }
}

void Builder::Impl::emitFunctionDef(const parser::Node &function) {
  const std::string *name = stringField(function, "name");
  if (!name)
    return;
  auto found = functions.find(*name);
  if (found == functions.end())
    return;
  if (hasCallableFormal(found->second))
    return;
  emitFunctionBody(function, found->second);
}

void Builder::Impl::emitFunctionBody(const parser::Node &function,
                                     const FunctionInfo &info) {
  emitFunctionBody(function, info, {});
}

void Builder::Impl::emitFunctionBody(
    const parser::Node &function, const FunctionInfo &info,
    const std::map<std::string, FunctionInfo> &initialCallableAliases) {
  const std::vector<parser::NodePtr> *body = nodeListField(function, "body");
  if (!body) {
    error(function, "FunctionDef.body is missing");
    return;
  }

  mlir::OpBuilder::InsertionGuard insertionGuard(builder);
  builder.setInsertionPointToEnd(module->getBody());

  std::map<std::string, Value> savedSymbols = std::move(symbols);
  std::map<std::string, FunctionInfo> savedLocalFunctions =
      std::move(localFunctions);
  std::map<std::string, FunctionInfo> savedCallableAliases =
      std::move(callableAliases);
  std::set<std::string> savedActiveGlobalNames = std::move(activeGlobalNames);
  std::set<std::string> savedActiveNonlocalNames =
      std::move(activeNonlocalNames);
  mlir::Type savedReturnType = currentReturnType;
  bool savedTerminated = blockTerminated;
  bool savedInNativeFunction = inNativeFunction;
  unsigned savedExceptionContextDepth = exceptionContextDepth;
  functionStack.push_back(&function);
  functionSymbolStack.push_back(info.symbolName);
  symbols.clear();
  localFunctions.clear();
  callableAliases.clear();
  activeGlobalNames.clear();
  activeNonlocalNames.clear();
  callableAliases = initialCallableAliases;
  currentReturnType = info.resultType;
  blockTerminated = false;
  exceptionContextDepth = 0;

  if (info.isNative) {
    mlir::func::FuncOp func = builder.create<mlir::func::FuncOp>(
        loc(function), info.symbolName, info.nativeFunctionType);
    func->setAttr("native", builder.getUnitAttr());
    addEntryBlock(func);
    inNativeFunction = true;
    for (auto indexed : llvm::enumerate(info.argNames)) {
      mlir::Value arg = func.getBody().front().getArgument(indexed.index());
      symbols.emplace(indexed.value(),
                      Value{arg, info.argTypes[indexed.index()]});
    }
  } else {
    llvm::ArrayRef<std::string> allArgNames(info.argNames);
    llvm::SmallVector<mlir::Type, 4> closureTypes;
    closureTypes.reserve(info.closureCaptures.size());
    for (const ClosureCapture &capture : info.closureCaptures)
      closureTypes.push_back(capture.storageType);
    llvm::SmallVector<mlir::Type, 8> entryTypes(info.argTypes.begin(),
                                                info.argTypes.end());
    if (info.varargType)
      entryTypes.push_back(info.varargType);
    if (info.kwargType)
      entryTypes.push_back(info.kwargType);
    entryTypes.append(closureTypes.begin(), closureTypes.end());
    py::FuncOp func = createFunc(
        info.symbolName, info.signatureType,
        stringArrayAttr(allArgNames.take_front(info.positionalCount)),
        /*hasVararg=*/static_cast<bool>(info.varargType),
        /*hasKwarg=*/static_cast<bool>(info.kwargType), info.mayThrow,
        stringArrayAttr(info.kwonlyNames),
        closureTypes.empty() ? mlir::ArrayAttr{} : typeArrayAttr(closureTypes));
    addEntryBlock(func, entryTypes);
    inNativeFunction = false;
    for (auto indexed : llvm::enumerate(info.argNames)) {
      mlir::Value arg = func.getBody().front().getArgument(indexed.index());
      symbols.emplace(indexed.value(),
                      Value{arg, info.argTypes[indexed.index()]});
    }
    if (info.varargName && info.varargType) {
      mlir::Value arg =
          func.getBody().front().getArgument(info.argTypes.size());
      symbols.emplace(*info.varargName, Value{arg, info.varargType});
    }
    if (info.kwargName && info.kwargType) {
      mlir::Value arg = func.getBody().front().getArgument(
          info.argTypes.size() + (info.varargType ? 1 : 0));
      symbols.emplace(*info.kwargName, Value{arg, info.kwargType});
    }
    const unsigned closureOffset =
        static_cast<unsigned>(info.argTypes.size() + (info.varargType ? 1 : 0) +
                              (info.kwargType ? 1 : 0));
    for (auto indexed : llvm::enumerate(info.closureCaptures)) {
      const ClosureCapture &capture = indexed.value();
      mlir::Value storage =
          func.getBody().front().getArgument(closureOffset + indexed.index());
      Value restored = restoreClosureValue(
          function, storage, capture.storageType, capture.value.type);
      if (!restored.value)
        continue;
      symbols.emplace(capture.name, restored);
      auto capturedFunction = savedLocalFunctions.find(capture.name);
      if (capturedFunction != savedLocalFunctions.end())
        localFunctions.emplace(capture.name, capturedFunction->second);
      auto capturedAlias = savedCallableAliases.find(capture.name);
      if (capturedAlias != savedCallableAliases.end())
        callableAliases.emplace(capture.name, capturedAlias->second);
    }
  }

  for (const parser::NodePtr &stmt : *body) {
    if (stmt && !blockTerminated)
      emitStatement(*stmt);
  }
  if (!blockTerminated) {
    if (info.resultType == noneType()) {
      if (info.isNative) {
        builder.create<mlir::func::ReturnOp>(loc());
      } else {
        mlir::Value none = builder.create<py::NoneOp>(loc(), noneType());
        builder.create<py::ReturnOp>(loc(), mlir::ValueRange{none});
      }
    } else {
      error(function, "function may exit without returning " +
                          typeString(info.resultType));
    }
  }
  symbols = std::move(savedSymbols);
  localFunctions = std::move(savedLocalFunctions);
  callableAliases = std::move(savedCallableAliases);
  activeGlobalNames = std::move(savedActiveGlobalNames);
  activeNonlocalNames = std::move(savedActiveNonlocalNames);
  currentReturnType = savedReturnType;
  blockTerminated = savedTerminated;
  inNativeFunction = savedInNativeFunction;
  exceptionContextDepth = savedExceptionContextDepth;
  functionSymbolStack.pop_back();
  functionStack.pop_back();
}

void Builder::Impl::emitNestedFunctionDef(const parser::Node &function) {
  const std::string *name = stringField(function, "name");
  if (!name)
    return;
  if (function.kind != "FunctionDef") {
    error(function, "only nested FunctionDef is supported here");
    return;
  }

  std::optional<FunctionInfo> parsed = parseFunctionInfo(function);
  if (!parsed)
    return;
  if (parsed->isNative || parsed->isAsync) {
    error(function, "nested native/async functions are not supported yet");
    return;
  }
  parsed->closureCaptures = collectNestedFunctionCaptures(function);
  if (laterRebindsCapturedName(function, parsed->closureCaptures))
    return;

  const std::vector<parser::NodePtr> *body = nodeListField(function, "body");
  if (body && statementListMayThrow(*body))
    parsed->mayThrow = true;
  parsed->symbolName = functionSymbolStack.empty()
                           ? "__lython_nested_" +
                                 std::to_string(++nestedFunctionCounter) + "_" +
                                 *name
                           : functionSymbolStack.back() + "__" + *name;

  emitFunctionBody(function, *parsed);
  Value functionValue = emitFunctionObject(function, *parsed);
  if (!functionValue.value)
    return;
  symbols[*name] = functionValue;
  localFunctions[*name] = *parsed;
  callableAliases[*name] = std::move(*parsed);
}

Value Builder::Impl::emitLambda(const parser::Node &expr,
                                mlir::Type expectedType) {
  auto funcType = mlir::dyn_cast<py::FuncType>(expectedType);
  if (!funcType) {
    error(expr, "lambda expression requires an expected Callable[...] type, "
                "got " +
                    typeString(expectedType));
    return Value{{}, expectedType};
  }

  std::optional<FunctionInfo> parsed = parseLambdaInfo(expr, funcType);
  if (!parsed)
    return Value{{}, expectedType};

  const parser::NodePtr *argsNode = nodeField(expr, "args");
  const parser::NodePtr *bodyNode = nodeField(expr, "body");
  if (!argsNode || !*argsNode || !bodyNode || !*bodyNode) {
    error(expr, "Lambda.args or Lambda.body is missing");
    return Value{{}, expectedType};
  }

  parser::NodePtr returnNode = parser::makeNode("Return", (*bodyNode)->range);
  parser::addField(*returnNode, "value", *bodyNode);

  parser::NodePtr function = parser::makeNode("FunctionDef", expr.range);
  parser::addField(*function, "name", parsed->symbolName);
  parser::addField(*function, "args", *argsNode);
  parser::addField(*function, "body", std::vector<parser::NodePtr>{returnNode});
  parser::addField(*function, "decorator_list", std::vector<parser::NodePtr>{});
  parser::addField(*function, "returns", parser::NodePtr{});
  parser::addField(*function, "type_comment", parser::FieldValue{});
  parser::addField(*function, "type_params", std::vector<parser::NodePtr>{});

  parsed->closureCaptures = collectNestedFunctionCaptures(*function);
  if (functionStack.empty() && !parsed->closureCaptures.empty()) {
    error(expr, "module-scope lambda captures are not supported because "
                "Python globals use late binding semantics");
    return Value{{}, expectedType};
  }
  if (laterRebindsCapturedName(*function, parsed->closureCaptures))
    return Value{{}, expectedType};

  mlir::Operation *enclosingTopLevel = nullptr;
  if (mlir::Block *block = builder.getBlock()) {
    mlir::Operation *parent = block->getParentOp();
    while (parent && parent->getParentOp() != module.get())
      parent = parent->getParentOp();
    enclosingTopLevel = parent;
  }

  emitFunctionBody(*function, *parsed);
  if (enclosingTopLevel) {
    if (auto lambdaFunc = module->lookupSymbol<py::FuncOp>(parsed->symbolName))
      lambdaFunc->moveBefore(enclosingTopLevel);
  }

  Value functionValue = emitFunctionObject(expr, *parsed);
  if (!functionValue.value)
    return Value{{}, expectedType};
  localFunctions[parsed->symbolName] = *parsed;
  return functionValue;
}

} // namespace lython::emitter
