#include "BuilderImpl.h"

#include "mlir/Dialect/Async/IR/Async.h"
#include "llvm/ADT/STLExtras.h"

#include <utility>

namespace lython::emitter {

void Builder::Impl::emitAsyncFunctionDef(const parser::Node &function) {
  const std::string *name = stringField(function, "name");
  if (!name)
    return;
  auto found = functions.find(*name);
  if (found == functions.end())
    return;
  if (hasCallableFormal(found->second) || hasProtocolFormal(found->second) ||
      hasClassFormal(found->second) || !found->second.typeParameters.empty())
    return;
  emitAsyncFunctionDef(function, found->second, {});
}

void Builder::Impl::emitAsyncFunctionDef(
    const parser::Node &function, const FunctionInfo &info,
    const std::map<std::string, FunctionInfo> &initialCallableAliases) {
  const std::vector<parser::NodePtr> *body = nodeListField(function, "body");
  if (!body) {
    error(function, "AsyncFunctionDef.body is missing");
    return;
  }
  if (info.isNative) {
    error(function, "async @native functions are not supported");
    return;
  }

  const std::vector<parser::NodePtr> *decorators =
      nodeListField(function, "decorator_list");
  if (decorators && !decorators->empty()) {
    error(function, "decorators on async functions are not supported");
    return;
  }

  std::map<std::string, Value> savedSymbols = std::move(symbols);
  std::map<std::string, FunctionInfo> savedCallableAliases =
      std::move(callableAliases);
  mlir::Type savedReturnType = currentReturnType;
  bool savedTerminated = blockTerminated;
  bool savedInNativeFunction = inNativeFunction;
  bool savedInAsyncFunction = inAsyncFunction;
  unsigned savedExceptionContextDepth = exceptionContextDepth;
  symbols.clear();
  callableAliases.clear();
  callableAliases = initialCallableAliases;
  currentReturnType = info.resultType;
  blockTerminated = false;
  inNativeFunction = false;
  inAsyncFunction = true;
  exceptionContextDepth = 0;

  auto func = builder.create<mlir::async::FuncOp>(
      loc(function), info.symbolName, info.asyncFunctionType);
  func->setAttr(info.mayThrow ? "maythrow" : "nothrow", builder.getUnitAttr());
  addAsyncEntryBlock(func.getOperation(), info.argTypes);
  for (auto indexed : llvm::enumerate(info.argNames)) {
    mlir::Value arg = func.getBody().front().getArgument(indexed.index());
    symbols.emplace(indexed.value(),
                    Value{arg, info.argTypes[indexed.index()]});
  }

  for (const parser::NodePtr &stmt : *body) {
    if (stmt && !blockTerminated)
      emitStatement(*stmt);
  }
  if (!blockTerminated) {
    if (info.resultType == noneType()) {
      mlir::Value none = builder.create<py::NoneOp>(loc(function), noneType());
      builder.create<mlir::async::ReturnOp>(loc(function),
                                            mlir::ValueRange{none});
    } else {
      error(function, "async function may exit without returning " +
                          typeString(info.resultType));
    }
  }

  symbols = std::move(savedSymbols);
  callableAliases = std::move(savedCallableAliases);
  currentReturnType = savedReturnType;
  blockTerminated = savedTerminated;
  inNativeFunction = savedInNativeFunction;
  inAsyncFunction = savedInAsyncFunction;
  exceptionContextDepth = savedExceptionContextDepth;
  builder.setInsertionPointToEnd(module->getBody());
}

std::optional<std::string>
Builder::Impl::asyncioBuiltinName(const parser::Node &func) {
  if (func.kind == "Name") {
    const std::string *name = stringField(func, "id");
    if (!name)
      return std::nullopt;
    auto found = staticModuleSymbols.find(*name);
    if (found != staticModuleSymbols.end() && found->second.first == "asyncio")
      return found->second.second;
    return std::nullopt;
  }
  if (func.kind != "Attribute")
    return std::nullopt;
  const parser::NodePtr *value = nodeField(func, "value");
  const std::string *attr = stringField(func, "attr");
  if (!value || !*value || (*value)->kind != "Name" || !attr)
    return std::nullopt;
  const std::string *moduleName = stringField(**value, "id");
  if (!moduleName)
    return std::nullopt;
  auto found = staticModules.find(*moduleName);
  if (found == staticModules.end() || found->second != "asyncio")
    return std::nullopt;
  return *attr;
}

Value Builder::Impl::emitAwait(const parser::Node &expr) {
  if (!inAsyncFunction) {
    error(expr, "await is valid only inside async functions");
    return Value{{}, noneType()};
  }
  const parser::NodePtr *valueNode = nodeField(expr, "value");
  if (!valueNode || !*valueNode) {
    error(expr, "Await.value is missing");
    return Value{{}, noneType()};
  }

  if ((*valueNode)->kind == "Call") {
    const parser::NodePtr *func = nodeField(**valueNode, "func");
    const std::vector<parser::NodePtr> *args =
        nodeListField(**valueNode, "args");
    if (func && *func && args) {
      std::optional<std::string> name = asyncioBuiltinName(**func);
      if (name && *name == "gather")
        return emitAsyncioGather(**valueNode, *args);
    }
  }

  Value awaitable = emitExpression(**valueNode);
  if (!awaitable.value)
    return Value{{}, noneType()};
  return awaitConcreteValue(expr, awaitable, "await operand");
}

Value Builder::Impl::emitAsyncioCall(const parser::Node &expr,
                                     llvm::StringRef name,
                                     const std::vector<parser::NodePtr> &args) {
  const std::vector<parser::NodePtr> *keywords =
      nodeListField(expr, "keywords");
  if (keywords && !keywords->empty()) {
    error(expr,
          "asyncio." + name.str() + " keyword arguments are not supported");
    return Value{{}, noneType()};
  }

  if (name == "run") {
    if (inAsyncFunction) {
      error(expr, "asyncio.run cannot be called from an async function");
      return Value{{}, noneType()};
    }
    if (args.size() != 1 || !args.front()) {
      error(expr, "asyncio.run expects exactly one awaitable");
      return Value{{}, noneType()};
    }
    if (args.front()->kind == "Call") {
      const parser::NodePtr *func = nodeField(*args.front(), "func");
      const std::vector<parser::NodePtr> *gatherArgs =
          nodeListField(*args.front(), "args");
      if (func && *func && gatherArgs) {
        std::optional<std::string> gatherName = asyncioBuiltinName(**func);
        if (gatherName && *gatherName == "gather")
          return emitAsyncioGather(*args.front(), *gatherArgs);
      }
    }
    Value awaitable = emitExpression(*args.front());
    mlir::Type payloadType = awaitablePayloadType(awaitable.type);
    if (!awaitable.value || !payloadType) {
      error(*args.front(), "asyncio.run expects a statically typed awaitable");
      return Value{{}, noneType()};
    }
    if (!lowerableAwaitableValueType(awaitable)) {
      error(*args.front(),
            "asyncio.run awaitable " + typeString(awaitable.type) +
                " resolves statically to " + typeString(payloadType) +
                ", but lowering currently requires a native "
                "Coroutine protocol descriptor or async.value");
      return Value{{}, payloadType};
    }
    mlir::Value result =
        builder.create<py::AwaitOp>(loc(expr), payloadType, awaitable.value);
    return Value{result, payloadType};
  }

  if (name == "create_task" || name == "ensure_future") {
    if (args.size() != 1 || !args.front()) {
      error(expr, "asyncio." + name.str() + " expects exactly one coroutine");
      return Value{{}, noneType()};
    }
    Value coroutine = emitExpression(*args.front());
    mlir::Type payloadType = awaitablePayloadType(coroutine.type);
    if (!coroutine.value || !payloadType ||
        !lowerableAwaitableValueType(coroutine)) {
      error(*args.front(),
            "asyncio." + name.str() + " expects a statically typed coroutine");
      return Value{{}, noneType()};
    }
    mlir::Type resultType = taskType(payloadType);
    error(expr, "asyncio." + name.str() +
                    " now resolves through the _asyncio.Task contract; "
                    "runtime Task object lowering is not implemented yet");
    return Value{{}, resultType};
  }

  if (name == "sleep") {
    if (args.size() != 1 || !args.front()) {
      error(expr, "asyncio.sleep expects exactly one duration");
      return Value{{}, noneType()};
    }
    mlir::Type resultType = futureType(noneType());
    error(expr, "asyncio.sleep now resolves through the _asyncio.Future "
                "contract; event-loop Future lowering is not implemented yet");
    return Value{{}, resultType};
  }

  if (name == "gather") {
    error(expr, "asyncio.gather must be immediately awaited or passed to "
                "asyncio.run");
    return Value{{}, noneType()};
  }

  error(expr, "asyncio." + name.str() + " is not supported");
  return Value{{}, noneType()};
}

Value Builder::Impl::emitAsyncioGather(
    const parser::Node &expr, const std::vector<parser::NodePtr> &args) {
  const std::vector<parser::NodePtr> *keywords =
      nodeListField(expr, "keywords");
  if (keywords && !keywords->empty()) {
    error(expr, "asyncio.gather keyword arguments are not supported");
    return Value{{}, noneType()};
  }

  llvm::SmallVector<mlir::Type> payloadTypes;
  for (const parser::NodePtr &arg : args) {
    if (!arg)
      continue;
    Value awaitable = emitExpression(*arg);
    mlir::Type payloadType = awaitablePayloadType(awaitable.type);
    if (!awaitable.value || !payloadType) {
      error(*arg, "asyncio.gather expects statically typed awaitables");
      return Value{{}, noneType()};
    }
    if (!lowerableAwaitableValueType(awaitable)) {
      error(*arg, "asyncio.gather awaitable " + typeString(awaitable.type) +
                      " resolves statically to " + typeString(payloadType) +
                      ", but lowering currently requires native Coroutine "
                      "protocol descriptors or async.value operands");
      return Value{{}, noneType()};
    }
    payloadTypes.push_back(payloadType);
  }

  if (payloadTypes.empty())
    return emitEmptyTuple();

  mlir::Type tupleType = py::TupleType::get(&context, payloadTypes);
  error(expr, "asyncio.gather now resolves through the asyncio contract; "
              "general gather runtime lowering is not implemented yet");
  return Value{{}, tupleType};
}

} // namespace lython::emitter
