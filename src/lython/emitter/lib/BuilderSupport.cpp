#include "BuilderImpl.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include <utility>

#include "PyDialect.h.inc"

namespace lython::emitter {

Builder::Impl::Impl(mlir::MLIRContext &context, std::string moduleName)
    : moduleName(std::move(moduleName)), context(context), builder(&context) {
  context.loadDialect<py::PyDialect, mlir::arith::ArithDialect,
                      mlir::async::AsyncDialect, mlir::cf::ControlFlowDialect,
                      mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                      mlir::scf::SCFDialect, mlir::tensor::TensorDialect>();
}

EmitResult Builder::Impl::emit(const parser::Node &moduleNode) {
  if (moduleNode.kind != "Module") {
    error(moduleNode, "C++ emitter expected a Module root");
    return finish();
  }

  scanStaticImports(moduleNode);
  scanTypeVariables(moduleNode);
  scanClasses(moduleNode);
  scanFunctions(moduleNode);
  module = mlir::ModuleOp::create(loc(), moduleName);
  builder.setInsertionPointToEnd(module->getBody());
  emitPrelude();
  emitMain(moduleNode);
  emitClassDefs(moduleNode);
  emitUserFunctions(moduleNode);
  emitPendingGenericClassDefs();

  EmitResult result;
  result.module = std::move(module);
  result.diagnostics = std::move(diagnostics);
  return result;
}

mlir::Location Builder::Impl::loc() { return builder.getUnknownLoc(); }
mlir::Location Builder::Impl::loc(const parser::Node &node) {
  int line = node.range.start.line > 0 ? node.range.start.line : 1;
  int column = node.range.start.column >= 0 ? node.range.start.column : 0;
  return mlir::FileLineColLoc::get(&context, moduleName,
                                   static_cast<unsigned>(line),
                                   static_cast<unsigned>(column));
}

Builder::Impl::NameBindingSnapshot
Builder::Impl::snapshotNameBinding(llvm::StringRef name) const {
  NameBindingSnapshot snapshot;
  std::string key = name.str();
  if (auto found = symbols.find(key); found != symbols.end())
    snapshot.symbol = found->second;
  if (auto found = callableAliases.find(key); found != callableAliases.end())
    snapshot.callableAlias = found->second;
  if (auto found = primitiveConstants.find(key);
      found != primitiveConstants.end())
    snapshot.primitiveConstant = found->second;
  return snapshot;
}

void Builder::Impl::restoreNameBinding(llvm::StringRef name,
                                       NameBindingSnapshot snapshot) {
  std::string key = name.str();
  if (snapshot.symbol)
    symbols[key] = *snapshot.symbol;
  else
    symbols.erase(key);

  if (snapshot.callableAlias)
    callableAliases[key] = *snapshot.callableAlias;
  else
    callableAliases.erase(key);

  if (snapshot.primitiveConstant)
    primitiveConstants[key] = *snapshot.primitiveConstant;
  else
    primitiveConstants.erase(key);
}

void Builder::Impl::bindTemporaryName(llvm::StringRef name, Value value) {
  std::string key = name.str();
  symbols[key] = value;
  callableAliases.erase(key);
  primitiveConstants.erase(key);
}

EmitResult Builder::Impl::finish() {
  EmitResult result;
  result.diagnostics = std::move(diagnostics);
  return result;
}

void Builder::Impl::error(const parser::Node &node, std::string message) {
  diagnostics.push_back(parser::Diagnostic{
      parser::Severity::Error, node.range.start, std::move(message)});
}

mlir::ArrayAttr
Builder::Impl::stringArrayAttr(llvm::ArrayRef<std::string> values) {
  llvm::SmallVector<mlir::Attribute> attrs;
  attrs.reserve(values.size());
  for (const std::string &value : values)
    attrs.push_back(builder.getStringAttr(value));
  return builder.getArrayAttr(attrs);
}

mlir::ArrayAttr
Builder::Impl::typeArrayAttr(llvm::ArrayRef<mlir::Type> values) {
  llvm::SmallVector<mlir::Attribute> attrs;
  attrs.reserve(values.size());
  for (mlir::Type value : values)
    attrs.push_back(mlir::TypeAttr::get(value));
  return builder.getArrayAttr(attrs);
}

bool Builder::Impl::isBuiltinExceptionClass(llvm::StringRef name) const {
  return name == "BaseException" || name == "Exception" ||
         name == "RuntimeError" || name == "TypeError" ||
         name == "ValueError" || name == "KeyError" || name == "IndexError" ||
         name == "AssertionError" || name == "StopIteration" ||
         name == "StopAsyncIteration";
}

bool Builder::Impl::isNativeDecorator(const parser::Node &node) {
  const parser::Node *call = &node;
  if (node.kind == "Name") {
    std::optional<std::string> name = lyrtBuiltinName(node);
    return name && *name == "native";
  }
  if (node.kind != "Call")
    return false;

  const parser::NodePtr *func = nodeField(node, "func");
  if (!func || !*func)
    return false;
  std::optional<std::string> name = lyrtBuiltinName(**func);
  if (!name || *name != "native")
    return false;

  const std::vector<parser::NodePtr> *args = nodeListField(*call, "args");
  if (args && !args->empty()) {
    error(node, "@native positional arguments are not supported");
    return true;
  }

  const std::vector<parser::NodePtr> *keywords =
      nodeListField(*call, "keywords");
  if (!keywords)
    return true;
  for (const parser::NodePtr &keyword : *keywords) {
    if (!keyword)
      continue;
    const std::string *arg = stringField(*keyword, "arg");
    const parser::NodePtr *value = nodeField(*keyword, "value");
    if (!arg || *arg != "gc" || !value || !*value ||
        (*value)->kind != "Constant") {
      error(*keyword, "@native currently accepts only gc=\"none\"");
      continue;
    }
    const parser::FieldValue *fieldValue = valueField(**value, "value");
    const auto *text =
        fieldValue ? std::get_if<std::string>(fieldValue) : nullptr;
    if (!text || *text != "none")
      error(**value, "@native currently accepts only gc=\"none\"");
  }
  return true;
}

py::CallableType Builder::Impl::functionSignatureType(
    llvm::ArrayRef<mlir::Type> argTypes, mlir::Type resultType,
    mlir::Type varargType, llvm::ArrayRef<mlir::Type> kwonlyTypes,
    mlir::Type kwargType, llvm::ArrayRef<std::string> positionalNames,
    llvm::ArrayRef<char> positionalDefaults,
    llvm::ArrayRef<std::string> kwonlyNames,
    llvm::ArrayRef<char> kwonlyDefaults, llvm::StringRef varargName,
    llvm::StringRef kwargName, std::size_t positionalOnlyCount) {
  llvm::SmallVector<mlir::Type> results{resultType};
  llvm::SmallVector<mlir::StringAttr> positionalNameAttrs;
  positionalNameAttrs.reserve(positionalNames.size());
  for (const std::string &name : positionalNames)
    positionalNameAttrs.push_back(builder.getStringAttr(name));
  llvm::SmallVector<mlir::StringAttr> kwonlyNameAttrs;
  kwonlyNameAttrs.reserve(kwonlyNames.size());
  for (const std::string &name : kwonlyNames)
    kwonlyNameAttrs.push_back(builder.getStringAttr(name));
  llvm::SmallVector<mlir::BoolAttr> positionalDefaultAttrs;
  positionalDefaultAttrs.reserve(positionalDefaults.size());
  for (char hasDefault : positionalDefaults)
    positionalDefaultAttrs.push_back(builder.getBoolAttr(hasDefault != 0));
  llvm::SmallVector<mlir::BoolAttr> kwonlyDefaultAttrs;
  kwonlyDefaultAttrs.reserve(kwonlyDefaults.size());
  for (char hasDefault : kwonlyDefaults)
    kwonlyDefaultAttrs.push_back(builder.getBoolAttr(hasDefault != 0));
  mlir::StringAttr varargNameAttr = varargName.empty()
                                        ? mlir::StringAttr{}
                                        : builder.getStringAttr(varargName);
  mlir::StringAttr kwargNameAttr =
      kwargName.empty() ? mlir::StringAttr{} : builder.getStringAttr(kwargName);
  return py::CallableType::get(
      &context, argTypes, kwonlyTypes, varargType, kwargType, results,
      positionalNameAttrs, kwonlyNameAttrs, positionalDefaultAttrs,
      kwonlyDefaultAttrs, varargNameAttr, kwargNameAttr,
      static_cast<unsigned>(positionalOnlyCount));
}

py::CallableType
Builder::Impl::callableParameterPackFromSignature(py::CallableType signature) {
  return py::CallableType::get(
      &context, signature.getPositionalTypes(), signature.getKwOnlyTypes(),
      signature.getVarargType(), signature.getKwargType(), {},
      signature.getPositionalNames(), signature.getKwOnlyNames(),
      signature.getPositionalDefaults(), signature.getKwOnlyDefaults(),
      signature.getVarargName(), signature.getKwargName(),
      signature.getPositionalOnlyCount());
}

py::CallableType Builder::Impl::callableParameterPackFromTuple(
    llvm::ArrayRef<mlir::Type> positionalTypes) {
  return py::CallableType::get(&context, positionalTypes, {}, {}, {}, {});
}

py::CallableType
Builder::Impl::callableSignatureWithResult(py::CallableType pack,
                                           mlir::Type resultType) {
  llvm::SmallVector<mlir::Type> results{resultType};
  return py::CallableType::get(
      &context, pack.getPositionalTypes(), pack.getKwOnlyTypes(),
      pack.getVarargType(), pack.getKwargType(), results,
      pack.getPositionalNames(), pack.getKwOnlyNames(),
      pack.getPositionalDefaults(), pack.getKwOnlyDefaults(),
      pack.getVarargName(), pack.getKwargName(), pack.getPositionalOnlyCount());
}

py::CallableType
Builder::Impl::prependCallableParameterPack(llvm::ArrayRef<mlir::Type> prefix,
                                            py::CallableType suffix) {
  if (prefix.empty())
    return suffix;

  llvm::SmallVector<mlir::Type> positional;
  positional.reserve(prefix.size() + suffix.getPositionalTypes().size());
  positional.append(prefix.begin(), prefix.end());
  positional.append(suffix.getPositionalTypes().begin(),
                    suffix.getPositionalTypes().end());

  llvm::SmallVector<mlir::StringAttr> positionalNames;
  if (!suffix.getPositionalNames().empty()) {
    positionalNames.reserve(positional.size());
    for (std::size_t index = 0; index < prefix.size(); ++index)
      positionalNames.push_back(builder.getStringAttr(""));
    positionalNames.append(suffix.getPositionalNames().begin(),
                           suffix.getPositionalNames().end());
  }

  llvm::SmallVector<mlir::BoolAttr> positionalDefaults;
  if (!suffix.getPositionalDefaults().empty()) {
    positionalDefaults.reserve(positional.size());
    for (std::size_t index = 0; index < prefix.size(); ++index)
      positionalDefaults.push_back(builder.getBoolAttr(false));
    positionalDefaults.append(suffix.getPositionalDefaults().begin(),
                              suffix.getPositionalDefaults().end());
  }

  return py::CallableType::get(
      &context, positional, suffix.getKwOnlyTypes(), suffix.getVarargType(),
      suffix.getKwargType(), suffix.getResultTypes(), positionalNames,
      suffix.getKwOnlyNames(), positionalDefaults, suffix.getKwOnlyDefaults(),
      suffix.getVarargName(), suffix.getKwargName(),
      static_cast<unsigned>(prefix.size() + suffix.getPositionalOnlyCount()));
}

std::optional<py::CallableType>
Builder::Impl::dropCallableParameterPrefix(py::CallableType pack,
                                           std::size_t prefixCount) {
  if (pack.getPositionalTypes().size() < prefixCount)
    return std::nullopt;

  llvm::SmallVector<mlir::Type> positional;
  positional.append(pack.getPositionalTypes().begin() + prefixCount,
                    pack.getPositionalTypes().end());

  llvm::SmallVector<mlir::StringAttr> positionalNames;
  if (!pack.getPositionalNames().empty()) {
    if (pack.getPositionalNames().size() < prefixCount)
      return std::nullopt;
    positionalNames.append(pack.getPositionalNames().begin() + prefixCount,
                           pack.getPositionalNames().end());
  }

  llvm::SmallVector<mlir::BoolAttr> positionalDefaults;
  if (!pack.getPositionalDefaults().empty()) {
    if (pack.getPositionalDefaults().size() < prefixCount)
      return std::nullopt;
    positionalDefaults.append(pack.getPositionalDefaults().begin() +
                                  prefixCount,
                              pack.getPositionalDefaults().end());
  }

  return py::CallableType::get(
      &context, positional, pack.getKwOnlyTypes(), pack.getVarargType(),
      pack.getKwargType(), pack.getResultTypes(), positionalNames,
      pack.getKwOnlyNames(), positionalDefaults, pack.getKwOnlyDefaults(),
      pack.getVarargName(), pack.getKwargName(),
      pack.getPositionalOnlyCount() > prefixCount
          ? static_cast<unsigned>(pack.getPositionalOnlyCount() - prefixCount)
          : 0);
}

void Builder::Impl::refreshFunctionTypes(FunctionInfo &info) {
  llvm::ArrayRef<mlir::Type> allArgTypes(info.argTypes);
  std::vector<std::string> positionalNames;
  positionalNames.reserve(info.positionalCount);
  for (std::size_t index = 0;
       index < info.positionalCount && index < info.argNames.size(); ++index)
    positionalNames.push_back(info.argNames[index]);

  llvm::SmallVector<char> positionalDefaults(info.positionalCount, 0);
  const std::size_t defaultStart =
      info.positionalCount >= info.defaultValues.size()
          ? info.positionalCount - info.defaultValues.size()
          : info.positionalCount;
  for (std::size_t index = defaultStart; index < info.positionalCount; ++index)
    positionalDefaults[index] = 1;

  llvm::SmallVector<char> kwonlyDefaults(info.kwonlyNames.size(), 0);
  for (std::size_t index = 0; index < info.kwonlyNames.size(); ++index)
    if (index < info.kwonlyDefaultValues.size() &&
        info.kwonlyDefaultValues[index])
      kwonlyDefaults[index] = 1;

  mlir::Type publicResultType =
      info.isAsync ? coroutineType(info.resultType) : info.resultType;
  info.signatureType = functionSignatureType(
      allArgTypes.take_front(info.positionalCount), publicResultType,
      info.varargType, allArgTypes.drop_front(info.positionalCount),
      info.kwargType, positionalNames, positionalDefaults, info.kwonlyNames,
      kwonlyDefaults,
      info.varargName ? llvm::StringRef(*info.varargName) : llvm::StringRef{},
      info.kwargName ? llvm::StringRef(*info.kwargName) : llvm::StringRef{},
      info.positionalOnlyCount);
  info.functionType = info.signatureType;

  if (info.isAsync)
    info.asyncFunctionType = asyncFunctionType(info.argTypes, info.resultType);

  llvm::SmallVector<mlir::Type> nativeResults;
  if (info.resultType != noneType())
    nativeResults.push_back(info.resultType);
  info.nativeFunctionType =
      mlir::FunctionType::get(&context, info.argTypes, nativeResults);
}

mlir::FunctionType
Builder::Impl::asyncFunctionType(llvm::ArrayRef<mlir::Type> argTypes,
                                 mlir::Type resultType) {
  llvm::SmallVector<mlir::Type> inputs(argTypes.begin(), argTypes.end());
  inputs.push_back(exceptionCellType());
  mlir::Type asyncResult = mlir::async::ValueType::get(resultType);
  return mlir::FunctionType::get(&context, inputs, {asyncResult});
}

bool Builder::Impl::lowerableAwaitableType(mlir::Type type) const {
  if (mlir::isa_and_nonnull<mlir::async::ValueType>(type))
    return true;
  if (!type)
    return false;
  return py::isCoroutineProtocolType(type);
}

mlir::Type
Builder::Impl::lowerableAwaitableValueType(const Value &value) const {
  if (lowerableAwaitableType(value.type))
    return value.type;
  if (value.protocolConcreteType &&
      lowerableAwaitableType(*value.protocolConcreteType))
    return *value.protocolConcreteType;
  return {};
}

mlir::Type Builder::Impl::methodAwaitableType(const FunctionInfo &method) {
  return method.isAsync ? coroutineType(method.resultType) : method.resultType;
}

Value Builder::Impl::awaitConcreteValue(const parser::Node &anchor,
                                        const Value &awaitable,
                                        llvm::StringRef contextLabel) {
  mlir::Type payloadType = awaitablePayloadType(awaitable.type);
  mlir::Type concreteType = lowerableAwaitableValueType(awaitable);
  if (!awaitable.value)
    return Value{{}, payloadType ? payloadType : noneType()};
  if (!payloadType) {
    error(anchor, contextLabel.str() + " must resolve to an awaitable, got " +
                      typeString(awaitable.type));
    return Value{{}, noneType()};
  }
  if (!lowerableAwaitableType(concreteType)) {
    error(anchor, contextLabel.str() + " resolves statically to " +
                      typeString(payloadType) +
                      ", but lowering currently requires a native Coroutine "
                      "protocol descriptor or async.value");
    return Value{{}, payloadType};
  }
  mlir::Value result =
      builder.create<py::AwaitOp>(loc(anchor), payloadType, awaitable.value);
  return Value{result, payloadType};
}

py::ClassOp Builder::Impl::createClass(llvm::StringRef name,
                                       mlir::ArrayAttr baseNames) {
  auto op = builder.create<py::ClassOp>(
      loc(), name, baseNames, mlir::ArrayAttr{}, mlir::ArrayAttr{},
      mlir::ArrayAttr{}, mlir::ArrayAttr{}, mlir::ArrayAttr{});
  op.getBody().emplaceBlock();
  return op;
}

py::CallableFuncOp Builder::Impl::createFunc(
    llvm::StringRef name, py::CallableType signature, mlir::ArrayAttr argNames,
    bool hasVararg, bool hasKwarg, bool mayThrow, mlir::ArrayAttr kwonlyNames,
    mlir::ArrayAttr closureTypes) {
  return builder.create<py::CallableFuncOp>(
      loc(), name, signature,
      hasVararg ? builder.getUnitAttr() : mlir::UnitAttr{},
      hasKwarg ? builder.getUnitAttr() : mlir::UnitAttr{}, argNames,
      kwonlyNames, closureTypes, mlir::UnitAttr{},
      mayThrow ? builder.getUnitAttr() : mlir::UnitAttr{},
      mayThrow ? mlir::UnitAttr{} : builder.getUnitAttr());
}

void Builder::Impl::addEntryBlock(py::CallableFuncOp func,
                                  llvm::ArrayRef<mlir::Type> argTypes) {
  mlir::Block *block = new mlir::Block();
  func.getBody().push_back(block);
  for (mlir::Type type : argTypes)
    block->addArgument(type, loc());
  builder.setInsertionPointToStart(block);
}

void Builder::Impl::addEntryBlock(mlir::func::FuncOp func) {
  mlir::Block *block = func.addEntryBlock();
  builder.setInsertionPointToStart(block);
}

void Builder::Impl::addAsyncEntryBlock(mlir::Operation *func,
                                       llvm::ArrayRef<mlir::Type> argTypes) {
  mlir::Region &body = func->getRegion(0);
  mlir::Block *block = new mlir::Block();
  body.push_back(block);
  for (mlir::Type type : argTypes)
    block->addArgument(type, loc());
  block->addArgument(exceptionCellType(), loc());
  builder.setInsertionPointToStart(block);
}

void Builder::Impl::emitPrelude() {
  createClass("BaseException");
  createClass("Exception", stringArrayAttr({"BaseException"}));
  for (llvm::StringRef name :
       {"RuntimeError", "TypeError", "ValueError", "KeyError", "IndexError",
        "AssertionError", "StopIteration", "StopAsyncIteration"})
    createClass(name, stringArrayAttr({"Exception"}));

  py::TupleType varargType = py::TupleType::get(&context, {strType()});
  py::CallableType signature =
      functionSignatureType({}, noneType(), varargType);
  for (llvm::StringRef name : {"__builtin_print", "__builtin_print_raw"}) {
    builder.setInsertionPointToEnd(module->getBody());
    py::CallableFuncOp builtin =
        createFunc(name, signature, {}, /*hasVararg=*/true,
                   /*hasKwarg=*/false);
    addEntryBlock(builtin, {varargType});
    mlir::Value none = builder.create<py::NoneOp>(loc(), noneType());
    builder.create<py::ReturnOp>(loc(), mlir::ValueRange{none});
  }

  builder.setInsertionPointToEnd(module->getBody());
}

} // namespace lython::emitter
