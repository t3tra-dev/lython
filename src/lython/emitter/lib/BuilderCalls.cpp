#include "BuilderImpl.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>

namespace lython::emitter {
namespace {

bool isStaticDefaultTypeSupported(mlir::Type type) {
  return mlir::isa<mlir::IntegerType, mlir::FloatType, mlir::RankedTensorType>(
      type);
}

std::optional<double> staticNumericValue(const parser::Node &node) {
  if (node.kind == "Constant") {
    const parser::FieldValue *value = valueField(node, "value");
    if (const auto *number = value ? std::get_if<double>(value) : nullptr)
      return *number;
    if (const auto *integer =
            value ? std::get_if<std::int64_t>(value) : nullptr)
      return static_cast<double>(*integer);
    return std::nullopt;
  }
  if (node.kind == "UnaryOp") {
    std::optional<std::string> op = symbolField(node, "op");
    const parser::NodePtr *operand = nodeField(node, "operand");
    if (!op || !operand || !*operand)
      return std::nullopt;
    std::optional<double> value = staticNumericValue(**operand);
    if (!value)
      return std::nullopt;
    if (*op == "+")
      return *value;
    if (*op == "-")
      return -*value;
  }
  return std::nullopt;
}

std::optional<std::string> staticStringConstant(const parser::Node &node) {
  if (node.kind != "Constant")
    return std::nullopt;
  const parser::FieldValue *value = valueField(node, "value");
  if (const auto *text = value ? std::get_if<std::string>(value) : nullptr)
    return *text;
  return std::nullopt;
}

std::optional<std::string>
staticIntConstructorDecimal(const parser::Node &node) {
  if (node.kind == "Constant") {
    const parser::FieldValue *value = valueField(node, "value");
    if (!value)
      return std::nullopt;
    if (const auto *integer = std::get_if<std::int64_t>(value))
      return std::to_string(*integer);
    if (const auto *integer = std::get_if<parser::BigInteger>(value))
      return integer->decimal;
    if (const auto *boolean = std::get_if<bool>(value))
      return *boolean ? "1" : "0";
    if (const auto *number = std::get_if<double>(value)) {
      if (!std::isfinite(*number))
        return std::nullopt;
      double truncated = std::trunc(*number);
      if (truncated <
              static_cast<double>(std::numeric_limits<std::int64_t>::min()) ||
          truncated >
              static_cast<double>(std::numeric_limits<std::int64_t>::max()))
        return std::nullopt;
      return std::to_string(static_cast<std::int64_t>(truncated));
    }
    return std::nullopt;
  }
  if (node.kind == "UnaryOp") {
    std::optional<std::string> op = symbolField(node, "op");
    const parser::NodePtr *operand = nodeField(node, "operand");
    if (!op || !operand || !*operand)
      return std::nullopt;
    std::optional<std::string> value = staticIntConstructorDecimal(**operand);
    if (!value)
      return std::nullopt;
    if (*op == "+")
      return *value;
    if (*op == "-") {
      if (!value->empty() && value->front() == '-')
        return value->substr(1);
      if (*value == "0")
        return *value;
      return "-" + *value;
    }
  }
  return std::nullopt;
}

std::optional<double> staticFloatConstructorValue(const parser::Node &node) {
  if (node.kind == "Constant") {
    const parser::FieldValue *value = valueField(node, "value");
    if (!value)
      return std::nullopt;
    if (const auto *number = std::get_if<double>(value))
      return *number;
    if (const auto *integer = std::get_if<std::int64_t>(value))
      return static_cast<double>(*integer);
    if (const auto *boolean = std::get_if<bool>(value))
      return *boolean ? 1.0 : 0.0;
    return std::nullopt;
  }
  if (node.kind == "UnaryOp") {
    std::optional<std::string> op = symbolField(node, "op");
    const parser::NodePtr *operand = nodeField(node, "operand");
    if (!op || !operand || !*operand)
      return std::nullopt;
    std::optional<double> value = staticFloatConstructorValue(**operand);
    if (!value)
      return std::nullopt;
    if (*op == "+")
      return *value;
    if (*op == "-")
      return -*value;
  }
  return std::nullopt;
}

std::size_t utf8CodepointCount(llvm::StringRef text) {
  std::size_t count = 0;
  for (std::size_t index = 0; index < text.size(); ++count) {
    unsigned char byte = static_cast<unsigned char>(text[index]);
    std::size_t width = 1;
    if ((byte & 0x80) == 0)
      width = 1;
    else if ((byte & 0xE0) == 0xC0)
      width = 2;
    else if ((byte & 0xF0) == 0xE0)
      width = 3;
    else if ((byte & 0xF8) == 0xF0)
      width = 4;
    index += std::min(width, text.size() - index);
  }
  return count;
}

bool isStaticNoneConstant(const parser::Node &node) {
  if (node.kind != "Constant")
    return false;
  const parser::FieldValue *value = valueField(node, "value");
  return value && std::holds_alternative<std::monostate>(*value);
}

std::optional<bool> staticBoolConstant(const parser::Node &node) {
  if (node.kind != "Constant")
    return std::nullopt;
  const parser::FieldValue *value = valueField(node, "value");
  if (const auto *boolean = value ? std::get_if<bool>(value) : nullptr)
    return *boolean;
  return std::nullopt;
}

mlir::Value stripCallableCasts(mlir::Value value) {
  while (auto cast = value.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
    if (cast->getNumOperands() != 1)
      break;
    value = cast.getOperand(0);
  }
  return value;
}

bool isMaterializedCallable(mlir::Value value) {
  mlir::Value stripped = stripCallableCasts(value);
  return stripped.getDefiningOp<py::FuncObjectOp>() ||
         stripped.getDefiningOp<py::MakeFunctionOp>();
}

std::optional<std::string> callableTargetSymbol(mlir::Value callable) {
  mlir::Value stripped = stripCallableCasts(callable);
  if (auto funcObject = stripped.getDefiningOp<py::FuncObjectOp>())
    return funcObject.getTargetAttr().getValue().str();
  if (auto makeFunction = stripped.getDefiningOp<py::MakeFunctionOp>())
    return makeFunction.getTargetAttr().getValue().str();
  if (mlir::Operation *op = stripped.getDefiningOp()) {
    if (auto returned = op->getAttrOfType<mlir::FlatSymbolRefAttr>(
            "ly.returned_callable_symbol"))
      return returned.getValue().str();
  }
  return std::nullopt;
}

void attachReturnedCallableSymbol(mlir::Operation *op,
                                  const FunctionInfo &info) {
  if (!op || !info.returnedCallableSymbolName)
    return;
  op->setAttr("ly.returned_callable_symbol",
              mlir::FlatSymbolRefAttr::get(op->getContext(),
                                           *info.returnedCallableSymbolName));
}

bool isNameRef(const parser::Node &node, llvm::StringRef name) {
  if (node.kind != "Name")
    return false;
  const std::string *id = stringField(node, "id");
  return id && *id == name;
}

std::optional<mlir::Value> invokeNormalSeed(mlir::OpBuilder &builder,
                                            mlir::Location loc,
                                            mlir::Type type) {
  if (auto intType = mlir::dyn_cast<mlir::IntegerType>(type))
    return builder.create<mlir::arith::ConstantIntOp>(loc, 0,
                                                      intType.getWidth());
  if (auto floatType = mlir::dyn_cast<mlir::FloatType>(type))
    return builder.create<mlir::arith::ConstantOp>(
        loc, type, builder.getFloatAttr(floatType, 0.0));
  return std::nullopt;
}

std::string pythonStringRepr(llvm::StringRef text) {
  std::string result = "'";
  for (char ch : text) {
    switch (ch) {
    case '\\':
      result += "\\\\";
      break;
    case '\'':
      result += "\\'";
      break;
    case '\n':
      result += "\\n";
      break;
    case '\r':
      result += "\\r";
      break;
    case '\t':
      result += "\\t";
      break;
    default:
      result.push_back(ch);
      break;
    }
  }
  result += "'";
  return result;
}

} // namespace

std::optional<std::size_t>
Builder::Impl::staticRangeLength(const StaticRangeSpec &spec) const {
  if (spec.step > 0) {
    if (spec.start >= spec.stop)
      return 0;
    __int128 diff = static_cast<__int128>(spec.stop) -
                    static_cast<__int128>(spec.start) - 1;
    __int128 count = diff / static_cast<__int128>(spec.step) + 1;
    if (count > static_cast<__int128>(std::numeric_limits<std::size_t>::max()))
      return std::nullopt;
    return static_cast<std::size_t>(count);
  }

  if (spec.start <= spec.stop)
    return 0;
  __int128 diff =
      static_cast<__int128>(spec.start) - static_cast<__int128>(spec.stop) - 1;
  __int128 step = -static_cast<__int128>(spec.step);
  __int128 count = diff / step + 1;
  if (count > static_cast<__int128>(std::numeric_limits<std::size_t>::max()))
    return std::nullopt;
  return static_cast<std::size_t>(count);
}

std::optional<StaticRangeSpec>
Builder::Impl::staticRangeSpec(const parser::Node &rangeCall,
                               llvm::StringRef context,
                               mlir::Type preferredElementType) {
  if (rangeCall.kind != "Call") {
    error(rangeCall, context.str() + " requires a range(...) argument");
    return std::nullopt;
  }
  const parser::NodePtr *func = nodeField(rangeCall, "func");
  const std::vector<parser::NodePtr> *args = nodeListField(rangeCall, "args");
  const std::vector<parser::NodePtr> *keywords =
      nodeListField(rangeCall, "keywords");
  if (!func || !*func || !isNameRef(**func, "range") || !args ||
      args->empty() || args->size() > 3 || (keywords && !keywords->empty())) {
    error(rangeCall, context.str() +
                         " supports only range(...) with 1 to 3 positional "
                         "arguments");
    return std::nullopt;
  }

  struct StaticBound {
    std::int64_t value = 0;
    mlir::Type explicitType;
  };

  auto readBound = [&](const parser::Node &node) -> std::optional<StaticBound> {
    if (std::optional<PrimitiveConstant> constant =
            primitiveIntConstructorConstant(node))
      return StaticBound{constant->integerValue, constant->type};
    if (std::optional<std::int64_t> value = staticIndexValue(node))
      return StaticBound{*value, {}};

    Value emitted = emitExpression(node);
    if (!emitted.value)
      return std::nullopt;
    if (emitted.type == intType()) {
      if (std::optional<std::int64_t> value = staticPyIntValue(emitted.value))
        return StaticBound{*value, {}};
    }
    error(node,
          context.str() + " range bounds must be statically known integers");
    return std::nullopt;
  };

  llvm::SmallVector<StaticBound, 3> bounds;
  bounds.reserve(args->size());
  for (const parser::NodePtr &arg : *args) {
    if (!arg) {
      error(rangeCall, context.str() + " range argument is missing");
      return std::nullopt;
    }
    std::optional<StaticBound> bound = readBound(*arg);
    if (!bound)
      return std::nullopt;
    bounds.push_back(*bound);
  }

  mlir::Type elementType = preferredElementType;
  for (const StaticBound &bound : bounds) {
    if (!bound.explicitType)
      continue;
    if (!elementType)
      elementType = bound.explicitType;
    else if (elementType != bound.explicitType) {
      error(rangeCall, context.str() +
                           " range primitive integer bounds must have a "
                           "single static type");
      return std::nullopt;
    }
  }
  if (!elementType)
    elementType = builder.getI64Type();
  auto intTy = mlir::dyn_cast<mlir::IntegerType>(elementType);
  if (!intTy) {
    error(rangeCall, context.str() +
                         " range elements must lower to a primitive integer "
                         "type");
    return std::nullopt;
  }

  std::int64_t start = 0;
  std::int64_t stop = bounds.front().value;
  std::int64_t step = 1;
  if (bounds.size() >= 2) {
    start = bounds[0].value;
    stop = bounds[1].value;
  }
  if (bounds.size() == 3)
    step = bounds[2].value;
  if (step == 0) {
    error(*(*args)[2], context.str() + " range step cannot be zero");
    return std::nullopt;
  }

  return StaticRangeSpec{start, stop, step, elementType};
}

std::optional<StaticRangeElements>
Builder::Impl::emitStaticRangeElements(const parser::Node &rangeCall,
                                       llvm::StringRef context,
                                       mlir::Type preferredElementType) {
  std::optional<StaticRangeSpec> spec =
      staticRangeSpec(rangeCall, context, preferredElementType);
  if (!spec)
    return std::nullopt;

  auto intTy = mlir::cast<mlir::IntegerType>(spec->elementType);

  std::vector<Value> elements;
  constexpr std::size_t maxStaticRangeElements = 1'000'000;
  auto append = [&](std::int64_t value) {
    mlir::Value constant = builder.create<mlir::arith::ConstantIntOp>(
        loc(rangeCall), value, intTy.getWidth());
    elements.push_back(Value{constant, spec->elementType});
  };

  for (std::int64_t current = spec->start;
       spec->step > 0 ? current < spec->stop : current > spec->stop;) {
    if (elements.size() >= maxStaticRangeElements) {
      error(rangeCall,
            context.str() + " range is too large to materialize statically");
      return std::nullopt;
    }
    append(current);
    if (spec->step > 0 &&
        current > std::numeric_limits<std::int64_t>::max() - spec->step)
      break;
    if (spec->step < 0 &&
        current < std::numeric_limits<std::int64_t>::min() - spec->step)
      break;
    current += spec->step;
  }
  return StaticRangeElements{std::move(elements), spec->elementType};
}

std::optional<std::size_t>
Builder::Impl::staticLength(const parser::Node &anchor, const Value &value,
                            llvm::StringRef context) {
  if (auto tupleType = mlir::dyn_cast<py::TupleType>(value.type)) {
    llvm::ArrayRef<mlir::Type> elementTypes = tupleType.getElementTypes();
    if (elementTypes.empty())
      return 0;
    if (elementTypes.size() > 1)
      return elementTypes.size();
    if (auto create = value.value.getDefiningOp<py::TupleCreateOp>())
      return create.getElements().size();
    error(anchor,
          context.str() + " requires a statically finite tuple argument");
    return std::nullopt;
  }

  if (listElementType(value.type)) {
    auto found = finiteListElements.find(value.value);
    if (found != finiteListElements.end())
      return found->second.size();
    error(anchor, context.str() +
                      " requires a list argument with statically known "
                      "elements");
    return std::nullopt;
  }

  if (value.type == strType()) {
    if (auto literal = value.value.getDefiningOp<py::StrConstantOp>())
      return utf8CodepointCount(literal.getValue());
    error(anchor, context.str() +
                      " requires a statically known string literal argument");
    return std::nullopt;
  }

  error(anchor, context.str() +
                    " supports only statically finite tuple/list/str "
                    "arguments in the C++ emitter");
  return std::nullopt;
}

Value Builder::Impl::emitCall(const parser::Node &expr) {
  const parser::NodePtr *func = nodeField(expr, "func");
  const std::vector<parser::NodePtr> *args = nodeListField(expr, "args");
  if (!func || !*func || !args) {
    error(expr, "Call.func or Call.args is missing");
    return Value{{}, noneType()};
  }

  if (std::optional<PrimitiveConstant> constant =
          primitiveIntConstructorConstant(expr))
    return emitPrimitiveIntConstructor(expr, *constant);
  if (std::optional<mlir::Type> targetType = typeFromAnnotation(*func);
      targetType && mlir::isa<mlir::IntegerType, mlir::FloatType>(*targetType))
    return emitPrimitiveScalarConstructor(expr, *targetType, *args);
  if (isTensorConstructorCallee(**func))
    return emitTensorConstructor(expr);

  if (std::optional<std::string> lyrtName = lyrtBuiltinName(**func)) {
    if (*lyrtName == "from_prim")
      return emitFromPrimCall(expr, *args);
    if (*lyrtName == "to_prim")
      return emitToPrimCall(expr, *args);
  }

  if ((*func)->kind == "Name") {
    if (std::optional<std::string> asyncioName = asyncioBuiltinName(**func))
      return emitAsyncioCall(expr, *asyncioName, *args);
    const std::string *name = stringField(**func, "id");
    if (name && *name == "print")
      return emitPrintCall(expr, *args);
    if (name && *name == "bool")
      return emitBoolConstructorCall(expr, *args);
    if (name && *name == "int")
      return emitIntConstructorCall(expr, *args);
    if (name && *name == "float")
      return emitFloatConstructorCall(expr, *args);
    if (name && *name == "str")
      return emitStrConstructorCall(expr, *args);
    if (name && *name == "repr")
      return emitReprCall(expr, *args);
    if (name && *name == "len")
      return emitLenCall(expr, *args);
    if (name && *name == "list")
      return emitListConstructorCall(expr, *args);
    if (name && *name == "tuple")
      return emitTupleConstructorCall(expr, *args);
    if (name && isBuiltinExceptionClass(*name))
      return emitExceptionCall(expr, *name, *args);
    if (name && classes.count(*name))
      return emitClassConstructorCall(expr, *name, *args);
    if (name) {
      auto alias = callableAliases.find(*name);
      auto bound = symbols.find(*name);
      if (alias != callableAliases.end() &&
          (bound == symbols.end() ||
           mlir::isa<mlir::BlockArgument>(bound->second.value) ||
           !alias->second.requiresCallableValue))
        return emitFunctionCall(expr, alias->second, *args);
    }
    if (name && !symbols.count(*name)) {
      auto local = localFunctions.find(*name);
      if (local != localFunctions.end())
        return emitFunctionCall(expr, local->second, *args);
    }
    if (name && !symbols.count(*name) && functions.count(*name))
      return emitFunctionCall(expr, functions.at(*name), *args);
  }
  if ((*func)->kind == "Attribute") {
    if (std::optional<std::string> asyncioName = asyncioBuiltinName(**func))
      return emitAsyncioCall(expr, *asyncioName, *args);
    return emitMethodCall(expr, **func, *args);
  }

  Value callable = emitExpression(**func);
  if (callable.value) {
    auto callableType = mlir::dyn_cast<py::FuncType>(callable.type);
    if (callableType) {
      if (std::optional<FunctionInfo> info =
              resolveCallableInfo(callable.value)) {
        if (info->isNative || info->isAsync) {
          error(expr, "Callable value calls for native or async functions are "
                      "not supported yet");
          return Value{{}, info->resultType};
        }
        if (callable.type != info->functionType) {
          error(expr, "Callable value type does not match resolved function "
                      "metadata for '" +
                          info->name + "'");
          return Value{{}, info->resultType};
        }

        std::optional<CallArgumentTuples> tuples =
            emitExplicitCallArgumentTuples(expr, *info, *args);
        if (!tuples)
          return Value{{}, info->resultType};

        mlir::Value callee = callable.value;
        bool returnedCallableValue = false;
        if (mlir::Operation *definingOp =
                stripCallableCasts(callee).getDefiningOp()) {
          returnedCallableValue =
              definingOp->hasAttr("ly.returned_callable_symbol");
        }
        if ((!isMaterializedCallable(callee) && !returnedCallableValue) ||
            (hasCallableMetadata(*info) &&
             !stripCallableCasts(callee).getDefiningOp<py::MakeFunctionOp>() &&
             !returnedCallableValue)) {
          Value materialized = emitFunctionObject(expr, *info);
          if (!materialized.value)
            return Value{{}, info->resultType};
          callee = materialized.value;
        }

        if (info->mayThrow)
          return emitMayThrowFunctionCall(expr, *info, callee, tuples->posargs,
                                          tuples->kwnames, tuples->kwvalues);
        auto call = builder.create<py::CallVectorOp>(
            loc(expr), mlir::TypeRange{info->resultType}, callee,
            tuples->posargs.value, tuples->kwnames.value,
            tuples->kwvalues.value, mlir::UnitAttr{});
        attachReturnedCallableSymbol(call.getOperation(), *info);
        return Value{call.getResults().front(), info->resultType};
      }

      error(expr, "Callable value calls require recoverable static function "
                  "metadata");
      return Value{{},
                   callableType.getSignature().getResultTypes().empty()
                       ? noneType()
                       : callableType.getSignature().getResultTypes().front()};
    }
  }

  error(expr, "C++ emitter only supports direct print(...), from_prim(...), "
              "to_prim(...), builtin exception constructors, primitive "
              "scalar constructors, class constructors, static methods, and "
              "static function calls for now");
  return Value{{}, noneType()};
}

std::optional<FunctionInfo> Builder::Impl::specializeFunctionCall(
    const parser::Node &expr, const FunctionInfo &info,
    const std::vector<parser::NodePtr> &args) {
  if (info.isSpecialization || info.isNative || info.isAsync ||
      info.varargType || !hasCallableFormal(info))
    return std::nullopt;

  std::optional<std::vector<const parser::Node *>> expandedArgs =
      expandStaticCallArgs(expr, args, info.positionalCount, info.name);
  if (!expandedArgs)
    return std::nullopt;
  std::optional<std::vector<StaticKeywordArg>> expandedKeywords =
      expandStaticCallKeywords(expr, info.name);
  if (!expandedKeywords)
    return std::nullopt;

  std::vector<const parser::Node *> actuals(info.argTypes.size(), nullptr);
  bool ok = true;
  for (std::size_t index = 0; index < expandedArgs->size(); ++index) {
    if (index >= actuals.size()) {
      ok = false;
      continue;
    }
    actuals[index] = (*expandedArgs)[index];
  }

  std::vector<std::string> seenKeywords;
  for (const StaticKeywordArg &keyword : *expandedKeywords) {
    if (!keyword.anchor || !keyword.value) {
      ok = false;
      continue;
    }
    if (std::find(seenKeywords.begin(), seenKeywords.end(), keyword.name) !=
        seenKeywords.end()) {
      error(*keyword.anchor, "function '" + info.name +
                                 "' got duplicate keyword '" + keyword.name +
                                 "'");
      ok = false;
      continue;
    }
    seenKeywords.push_back(keyword.name);

    auto nameIt =
        std::find(info.argNames.begin(), info.argNames.end(), keyword.name);
    if (nameIt == info.argNames.end()) {
      error(*keyword.anchor, "function '" + info.name + "' has no parameter '" +
                                 keyword.name + "'");
      ok = false;
      continue;
    }
    std::size_t formalIndex =
        static_cast<std::size_t>(std::distance(info.argNames.begin(), nameIt));
    if (formalIndex < info.positionalOnlyCount) {
      error(*keyword.anchor, "function '" + info.name +
                                 "' got positional-only argument '" +
                                 keyword.name + "' passed as keyword");
      ok = false;
      continue;
    }
    if (formalIndex < expandedArgs->size() || actuals[formalIndex]) {
      error(*keyword.anchor, "function '" + info.name +
                                 "' got multiple values for argument '" +
                                 keyword.name + "'");
      ok = false;
      continue;
    }
    actuals[formalIndex] = keyword.value;
  }
  if (!ok)
    return std::nullopt;

  std::map<std::string, FunctionInfo> aliases;
  std::string key = info.symbolName;
  for (std::size_t formalIndex = 0; formalIndex < info.argTypes.size();
       ++formalIndex) {
    if (!mlir::isa<py::FuncType>(info.argTypes[formalIndex]))
      continue;

    const parser::Node *actual = actuals[formalIndex];
    std::optional<FunctionInfo> actualInfo;
    if (actual)
      actualInfo = resolveCallableInfo(*actual);
    if (!actualInfo && hasDefaultForFormal(info, formalIndex)) {
      const parser::NodePtr *defaultNode = nullptr;
      if (formalIndex < info.positionalCount) {
        const std::size_t defaultCount = info.defaultValues.size();
        const std::size_t defaultStart =
            info.positionalCount >= defaultCount
                ? info.positionalCount - defaultCount
                : info.positionalCount;
        if (formalIndex >= defaultStart && formalIndex < info.positionalCount)
          defaultNode = &info.defaultValues[formalIndex - defaultStart];
      } else {
        const std::size_t kwonlyIndex = formalIndex - info.positionalCount;
        if (kwonlyIndex < info.kwonlyDefaultValues.size())
          defaultNode = &info.kwonlyDefaultValues[kwonlyIndex];
      }
      if (defaultNode && *defaultNode)
        actualInfo = resolveCallableInfo(**defaultNode);
    }
    if (!actualInfo) {
      error(expr, "function '" + info.name + "' callable argument '" +
                      info.argNames[formalIndex] +
                      "' requires a statically known callable");
      return std::nullopt;
    }
    if (actualInfo->functionType != info.argTypes[formalIndex]) {
      const parser::Node &anchor = actual ? *actual : expr;
      error(anchor, "callable argument '" + info.argNames[formalIndex] +
                        "' for function '" + info.name + "' must be " +
                        typeString(info.argTypes[formalIndex]) + ", got " +
                        typeString(actualInfo->functionType));
      return std::nullopt;
    }
    aliases.emplace(info.argNames[formalIndex], *actualInfo);
    key += "|" + std::to_string(formalIndex) + ":" + actualInfo->symbolName;
  }

  auto existing = functionSpecializations.find(key);
  if (existing != functionSpecializations.end())
    return existing->second.info;

  FunctionInfo specialized = info;
  specialized.isSpecialization = true;
  specialized.symbolName = "__lython_spec_" +
                           std::to_string(++functionSpecializationCounter) +
                           "_" + info.symbolName;
  FunctionSpecialization stored{specialized, std::move(aliases)};
  functionSpecializations.emplace(key, std::move(stored));
  return specialized;
}

std::optional<FunctionInfo>
Builder::Impl::resolveCallableInfo(const parser::Node &expr) const {
  if (expr.kind == "Call") {
    const parser::NodePtr *func = nodeField(expr, "func");
    if (!func || !*func)
      return std::nullopt;
    std::optional<FunctionInfo> callee = resolveCallableInfo(**func);
    if (!callee)
      return std::nullopt;
    if (callee->returnedCallableArgIndex)
      return resolveCallableArgInfo(expr, *callee,
                                    *callee->returnedCallableArgIndex);
    if (!callee->returnedCallableSymbolName)
      return std::nullopt;
    return findCallableInfoBySymbol(*callee->returnedCallableSymbolName);
  }

  if (expr.kind != "Name")
    return std::nullopt;
  const std::string *name = stringField(expr, "id");
  if (!name)
    return std::nullopt;

  auto alias = callableAliases.find(*name);
  if (alias != callableAliases.end())
    return alias->second;

  auto local = localFunctions.find(*name);
  if (local != localFunctions.end())
    return local->second;

  auto function = functions.find(*name);
  if (function != functions.end())
    return function->second;

  return std::nullopt;
}

std::optional<FunctionInfo>
Builder::Impl::resolveCallableArgInfo(const parser::Node &call,
                                      const FunctionInfo &info,
                                      std::size_t formalIndex) const {
  const std::vector<parser::NodePtr> *args = nodeListField(call, "args");
  if (!args)
    return std::nullopt;
  if (formalIndex < args->size() && (*args)[formalIndex])
    return resolveCallableInfo(*(*args)[formalIndex]);

  const std::vector<parser::NodePtr> *keywords =
      nodeListField(call, "keywords");
  if (keywords && formalIndex < info.argNames.size()) {
    const std::string &formalName = info.argNames[formalIndex];
    for (const parser::NodePtr &keyword : *keywords) {
      if (!keyword)
        continue;
      const std::string *argName = stringField(*keyword, "arg");
      const parser::NodePtr *value = nodeField(*keyword, "value");
      if (argName && *argName == formalName && value && *value)
        return resolveCallableInfo(**value);
    }
  }

  if (!hasDefaultForFormal(info, formalIndex))
    return std::nullopt;

  const parser::NodePtr *defaultNode = nullptr;
  if (formalIndex < info.positionalCount) {
    const std::size_t defaultCount = info.defaultValues.size();
    const std::size_t defaultStart = info.positionalCount >= defaultCount
                                         ? info.positionalCount - defaultCount
                                         : info.positionalCount;
    if (formalIndex >= defaultStart && formalIndex < info.positionalCount)
      defaultNode = &info.defaultValues[formalIndex - defaultStart];
  } else {
    const std::size_t kwonlyIndex = formalIndex - info.positionalCount;
    if (kwonlyIndex < info.kwonlyDefaultValues.size())
      defaultNode = &info.kwonlyDefaultValues[kwonlyIndex];
  }
  if (!defaultNode || !*defaultNode)
    return std::nullopt;
  return resolveCallableInfo(**defaultNode);
}

std::optional<FunctionInfo>
Builder::Impl::findCallableInfoBySymbol(llvm::StringRef symbolName) const {
  for (const auto &entry : localFunctions) {
    const FunctionInfo &info = entry.second;
    if (info.symbolName == symbolName)
      return info;
  }
  for (const auto &entry : functions) {
    const FunctionInfo &info = entry.second;
    if (info.symbolName == symbolName)
      return info;
  }
  auto callable = callableFunctionsBySymbol.find(symbolName.str());
  if (callable != callableFunctionsBySymbol.end())
    return callable->second;
  return std::nullopt;
}

std::optional<FunctionInfo>
Builder::Impl::resolveCallableInfo(mlir::Value callable) const {
  std::optional<std::string> symbolName = callableTargetSymbol(callable);
  if (!symbolName)
    return std::nullopt;
  return findCallableInfoBySymbol(*symbolName);
}

Value Builder::Impl::emitPrimitiveIntConstructor(const parser::Node &expr,
                                                 PrimitiveConstant constant) {
  mlir::Value result = builder.create<mlir::arith::ConstantIntOp>(
      loc(expr), constant.integerValue,
      mlir::cast<mlir::IntegerType>(constant.type).getWidth());
  return Value{result, constant.type};
}

Value Builder::Impl::emitPrimitiveScalarConstructor(
    const parser::Node &expr, mlir::Type targetType,
    const std::vector<parser::NodePtr> &args) {
  const std::vector<parser::NodePtr> *keywords =
      nodeListField(expr, "keywords");
  if (args.size() != 1 || !args.front() || (keywords && !keywords->empty())) {
    error(expr, "primitive scalar constructor expects one positional argument");
    return Value{{}, targetType};
  }

  if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(targetType)) {
    if (std::optional<std::int64_t> integer = staticIndexValue(*args.front())) {
      mlir::Value result = builder.create<mlir::arith::ConstantIntOp>(
          loc(expr), *integer, intTy.getWidth());
      return Value{result, targetType};
    }
    Value value = emitExpression(*args.front());
    if (!value.value)
      return Value{{}, targetType};
    auto sourceIntTy = mlir::dyn_cast<mlir::IntegerType>(value.type);
    if (!sourceIntTy) {
      error(*args.front(), "integer primitive constructor requires a primitive "
                           "integer argument");
      return Value{{}, targetType};
    }
    if (sourceIntTy == intTy)
      return Value{value.value, targetType};
    if (sourceIntTy.getWidth() < intTy.getWidth()) {
      mlir::Value result = builder.create<mlir::arith::ExtSIOp>(
          loc(expr), targetType, value.value);
      return Value{result, targetType};
    }
    mlir::Value result = builder.create<mlir::arith::TruncIOp>(
        loc(expr), targetType, value.value);
    return Value{result, targetType};
  }

  if (auto floatTy = mlir::dyn_cast<mlir::FloatType>(targetType)) {
    if (std::optional<double> number = staticNumericValue(*args.front())) {
      mlir::Value result = builder.create<mlir::arith::ConstantOp>(
          loc(expr), targetType, builder.getFloatAttr(floatTy, *number));
      return Value{result, targetType};
    }
    Value value = emitExpression(*args.front());
    if (!value.value)
      return Value{{}, targetType};
    auto sourceFloatTy = mlir::dyn_cast<mlir::FloatType>(value.type);
    if (!sourceFloatTy) {
      error(*args.front(),
            "float primitive constructor requires a primitive float argument");
      return Value{{}, targetType};
    }
    if (sourceFloatTy == floatTy)
      return Value{value.value, targetType};
    if (sourceFloatTy.getWidth() < floatTy.getWidth()) {
      mlir::Value result = builder.create<mlir::arith::ExtFOp>(
          loc(expr), targetType, value.value);
      return Value{result, targetType};
    }
    mlir::Value result = builder.create<mlir::arith::TruncFOp>(
        loc(expr), targetType, value.value);
    return Value{result, targetType};
  }

  error(expr, "unsupported primitive scalar constructor result type " +
                  typeString(targetType));
  return Value{{}, targetType};
}

std::optional<std::vector<const parser::Node *>>
Builder::Impl::expandStaticCallArgs(const parser::Node &expr,
                                    const std::vector<parser::NodePtr> &args,
                                    std::optional<std::size_t> positionalLimit,
                                    llvm::StringRef calleeName) {
  std::vector<const parser::Node *> expanded;
  bool ok = true;

  auto targetName = [&]() -> std::string {
    if (calleeName.empty())
      return "call";
    return "function '" + calleeName.str() + "'";
  };

  for (const parser::NodePtr &arg : args) {
    if (!arg) {
      ok = false;
      continue;
    }
    if (arg->kind != "Starred") {
      expanded.push_back(arg.get());
      continue;
    }

    const parser::NodePtr *value = nodeField(*arg, "value");
    if (!value || !*value) {
      error(*arg, "starred call argument is missing a value");
      ok = false;
      continue;
    }

    const parser::Node &container = **value;
    if (container.kind != "Tuple" && container.kind != "List") {
      error(container, "static *args lowering for " + targetName() +
                           " requires a tuple or list literal");
      ok = false;
      continue;
    }
    const std::vector<parser::NodePtr> *elements =
        nodeListField(container, "elts");
    if (!elements) {
      error(container, "static *args literal has no element list");
      ok = false;
      continue;
    }

    for (const parser::NodePtr &element : *elements) {
      if (!element) {
        ok = false;
        continue;
      }
      if (element->kind == "Starred") {
        error(*element, "nested starred elements inside static *args are not "
                        "implemented yet");
        ok = false;
        continue;
      }
      expanded.push_back(element.get());
    }
  }

  if (!ok)
    return std::nullopt;
  if (positionalLimit && expanded.size() > *positionalLimit) {
    error(expr, targetName() + " expects " + std::to_string(*positionalLimit) +
                    " positional arguments, got " +
                    std::to_string(expanded.size()));
    return std::nullopt;
  }
  return expanded;
}

std::optional<std::vector<StaticKeywordArg>>
Builder::Impl::expandStaticCallKeywords(const parser::Node &expr,
                                        llvm::StringRef calleeName) {
  const std::vector<parser::NodePtr> *keywords =
      nodeListField(expr, "keywords");
  std::vector<StaticKeywordArg> expanded;
  if (!keywords)
    return expanded;

  auto targetName = [&]() -> std::string {
    if (calleeName.empty())
      return "call";
    return "function '" + calleeName.str() + "'";
  };

  bool ok = true;
  for (const parser::NodePtr &keyword : *keywords) {
    if (!keyword) {
      ok = false;
      continue;
    }

    const parser::NodePtr *valueNode = nodeField(*keyword, "value");
    if (!valueNode || !*valueNode) {
      error(*keyword, "keyword argument is missing a value");
      ok = false;
      continue;
    }

    if (const std::string *arg = stringField(*keyword, "arg")) {
      expanded.push_back(
          StaticKeywordArg{keyword.get(), *arg, valueNode->get()});
      continue;
    }

    const parser::Node &dict = **valueNode;
    if (dict.kind != "Dict") {
      error(dict, "static **kwargs lowering for " + targetName() +
                      " requires a dict literal");
      ok = false;
      continue;
    }
    const std::vector<parser::NodePtr> *keys = nodeListField(dict, "keys");
    const std::vector<parser::NodePtr> *values = nodeListField(dict, "values");
    if (!keys || !values || keys->size() != values->size()) {
      error(dict, "static **kwargs dict literal has invalid key/value lists");
      ok = false;
      continue;
    }

    for (std::size_t i = 0; i < keys->size(); ++i) {
      const parser::NodePtr &key = (*keys)[i];
      const parser::NodePtr &value = (*values)[i];
      if (!key || !value) {
        error(dict, "nested **kwargs inside static **kwargs literal are not "
                    "implemented yet");
        ok = false;
        continue;
      }
      std::optional<std::string> name = staticStringConstant(*key);
      if (!name) {
        error(*key, "static **kwargs keys for " + targetName() +
                        " must be string constants");
        ok = false;
        continue;
      }
      expanded.push_back(StaticKeywordArg{keyword.get(), *name, value.get()});
    }
  }

  if (!ok)
    return std::nullopt;
  return expanded;
}

std::optional<std::vector<Value>> Builder::Impl::emitStaticArguments(
    const parser::Node &expr, const FunctionInfo &info,
    const std::vector<parser::NodePtr> &args, std::size_t firstFormal) {
  if (info.varargType) {
    error(expr, "internal error: vararg calls must use vectorcall lowering");
    return std::nullopt;
  }
  if (firstFormal > info.argTypes.size()) {
    error(expr, "invalid static call signature for '" + info.name + "'");
    return std::nullopt;
  }

  const std::size_t formalCount = info.argTypes.size() - firstFormal;
  const std::size_t positionalLimit = info.positionalCount > firstFormal
                                          ? info.positionalCount - firstFormal
                                          : 0;
  std::optional<std::vector<const parser::Node *>> expandedArgs =
      expandStaticCallArgs(expr, args, positionalLimit, info.name);
  if (!expandedArgs)
    return std::nullopt;
  std::optional<std::vector<StaticKeywordArg>> expandedKeywords =
      expandStaticCallKeywords(expr, info.name);
  if (!expandedKeywords)
    return std::nullopt;

  std::vector<Value> actuals(formalCount);
  std::vector<bool> filled(formalCount, false);
  bool ok = true;

  auto storeActual = [&](const parser::Node &node, std::size_t formalIndex,
                         Value value) {
    std::size_t localIndex = formalIndex - firstFormal;
    mlir::Type expected = info.argTypes[formalIndex];
    if (!typeAssignable(expected, value.type)) {
      error(node, "argument '" + info.argNames[formalIndex] +
                      "' for function '" + info.name + "' must be " +
                      typeString(expected) + ", got " + typeString(value.type));
      ok = false;
      return;
    }
    actuals[localIndex] = value;
    filled[localIndex] = true;
  };

  for (std::size_t i = 0; i < expandedArgs->size(); ++i) {
    if (!(*expandedArgs)[i]) {
      ok = false;
      continue;
    }
    const std::size_t formalIndex = firstFormal + i;
    Value value = emitExpressionWithExpectedType(*(*expandedArgs)[i],
                                                 info.argTypes[formalIndex]);
    if (!value.value) {
      ok = false;
      continue;
    }
    storeActual(*(*expandedArgs)[i], formalIndex, std::move(value));
  }

  if (!expandedKeywords->empty()) {
    std::vector<std::string> seen;
    for (const StaticKeywordArg &keyword : *expandedKeywords) {
      if (!keyword.anchor || !keyword.value) {
        ok = false;
        continue;
      }
      if (std::find(seen.begin(), seen.end(), keyword.name) != seen.end()) {
        error(*keyword.anchor, "function '" + info.name +
                                   "' got duplicate keyword '" + keyword.name +
                                   "'");
        ok = false;
        continue;
      }
      seen.push_back(keyword.name);

      auto nameBegin = info.argNames.begin() + firstFormal;
      auto nameIt = std::find(nameBegin, info.argNames.end(), keyword.name);
      if (nameIt == info.argNames.end()) {
        error(*keyword.anchor, "function '" + info.name +
                                   "' has no parameter '" + keyword.name + "'");
        ok = false;
        continue;
      }
      std::size_t formalIndex = static_cast<std::size_t>(
          std::distance(info.argNames.begin(), nameIt));
      if (formalIndex < info.positionalOnlyCount) {
        error(*keyword.anchor, "function '" + info.name +
                                   "' got positional-only argument '" +
                                   keyword.name + "' passed as keyword");
        filled[formalIndex - firstFormal] = true;
        ok = false;
        continue;
      }
      std::size_t localIndex = formalIndex - firstFormal;
      if (localIndex < expandedArgs->size() || filled[localIndex]) {
        error(*keyword.anchor, "function '" + info.name +
                                   "' got multiple values for argument '" +
                                   keyword.name + "'");
        ok = false;
        continue;
      }

      Value value = emitExpressionWithExpectedType(*keyword.value,
                                                   info.argTypes[formalIndex]);
      if (!value.value) {
        ok = false;
        continue;
      }
      storeActual(*keyword.value, formalIndex, std::move(value));
    }
  }
  if (!ok)
    return std::nullopt;

  for (std::size_t i = 0; i < formalCount; ++i) {
    if (filled[i])
      continue;
    std::size_t formalIndex = firstFormal + i;
    const parser::NodePtr *defaultNode = nullptr;
    if (formalIndex < info.positionalCount) {
      const std::size_t defaultCount = info.defaultValues.size();
      const std::size_t defaultStart = info.positionalCount >= defaultCount
                                           ? info.positionalCount - defaultCount
                                           : info.positionalCount;
      if (formalIndex >= defaultStart && formalIndex < info.positionalCount)
        defaultNode = &info.defaultValues[formalIndex - defaultStart];
    } else {
      const std::size_t kwonlyIndex = formalIndex - info.positionalCount;
      if (kwonlyIndex < info.kwonlyDefaultValues.size())
        defaultNode = &info.kwonlyDefaultValues[kwonlyIndex];
    }
    if (defaultNode && *defaultNode) {
      mlir::Type expected = info.argTypes[formalIndex];
      if (!isStaticDefaultTypeSupported(expected)) {
        error(**defaultNode, "default argument for function '" + info.name +
                                 "' has ownership-carrying type " +
                                 typeString(expected) +
                                 ", which requires callable metadata lowering");
        ok = false;
        continue;
      }
      if (expressionMayThrow(**defaultNode)) {
        error(**defaultNode,
              "default argument for function '" + info.name +
                  "' must be nothrow for static materialization");
        ok = false;
        continue;
      }
      Value value = emitExpressionWithExpectedType(**defaultNode, expected);
      if (!value.value) {
        ok = false;
        continue;
      }
      storeActual(**defaultNode, formalIndex, std::move(value));
      continue;
    }
    error(expr, "function '" + info.name + "' missing required argument '" +
                    info.argNames[formalIndex] + "'");
    ok = false;
  }
  if (!ok)
    return std::nullopt;
  return actuals;
}

std::optional<CallArgumentTuples> Builder::Impl::emitExplicitCallArgumentTuples(
    const parser::Node &expr, const FunctionInfo &info,
    const std::vector<parser::NodePtr> &args, std::size_t firstFormal,
    llvm::ArrayRef<Value> leadingArgs) {
  if (firstFormal > info.argTypes.size()) {
    error(expr, "invalid explicit call signature for '" + info.name + "'");
    return std::nullopt;
  }
  if (leadingArgs.size() > firstFormal) {
    error(expr, "too many leading arguments for explicit call to '" +
                    info.name + "'");
    return std::nullopt;
  }
  const std::size_t formalCount = info.argTypes.size();
  const std::size_t positionalLimit = info.positionalCount > firstFormal
                                          ? info.positionalCount - firstFormal
                                          : 0;
  std::optional<std::vector<const parser::Node *>> expandedArgs =
      expandStaticCallArgs(expr, args,
                           info.varargType
                               ? std::nullopt
                               : std::optional<std::size_t>(positionalLimit),
                           info.name);
  if (!expandedArgs)
    return std::nullopt;
  std::optional<std::vector<StaticKeywordArg>> expandedKeywords =
      expandStaticCallKeywords(expr, info.name);
  if (!expandedKeywords)
    return std::nullopt;

  std::vector<Value> posargs;
  std::vector<Value> kwnames;
  std::vector<Value> kwvalues;
  std::vector<bool> filled(formalCount, false);
  bool ok = true;
  mlir::Type kwargValueType;
  if (info.kwargType) {
    auto dictType = mlir::dyn_cast<py::DictType>(info.kwargType);
    if (!dictType || dictType.getKeyType() != strType()) {
      error(expr, "invalid kwarg signature for '" + info.name + "'");
      return std::nullopt;
    }
    kwargValueType = dictType.getValueType();
  }

  for (auto indexed : llvm::enumerate(leadingArgs)) {
    std::size_t formalIndex = indexed.index();
    const Value &value = indexed.value();
    if (!value.value) {
      ok = false;
      continue;
    }
    mlir::Type expected = info.argTypes[formalIndex];
    if (!typeAssignable(expected, value.type)) {
      error(expr, "leading argument '" + info.argNames[formalIndex] +
                      "' for function '" + info.name + "' must be " +
                      typeString(expected) + ", got " + typeString(value.type));
      ok = false;
      continue;
    }
    posargs.push_back(value);
    filled[formalIndex] = true;
  }

  auto emitActual = [&](const parser::Node &node,
                        std::size_t formalIndex) -> Value {
    mlir::Type expected = info.argTypes[formalIndex];
    Value value = emitExpressionWithExpectedType(node, expected);
    if (!value.value) {
      ok = false;
      return Value{{}, expected};
    }
    if (!typeAssignable(expected, value.type)) {
      error(node, "argument '" + info.argNames[formalIndex] +
                      "' for function '" + info.name + "' must be " +
                      typeString(expected) + ", got " + typeString(value.type));
      ok = false;
      return Value{{}, expected};
    }
    return value;
  };
  mlir::Type varargElementType;
  if (info.varargType) {
    auto tupleType = mlir::dyn_cast<py::TupleType>(info.varargType);
    if (!tupleType || tupleType.getElementTypes().size() != 1) {
      error(expr, "invalid vararg signature for '" + info.name + "'");
      return std::nullopt;
    }
    varargElementType = tupleType.getElementTypes().front();
  }

  auto emitVarargActual = [&](const parser::Node &node) -> Value {
    Value value = emitExpressionWithExpectedType(node, varargElementType);
    if (!value.value) {
      ok = false;
      return Value{{}, varargElementType};
    }
    if (!typeAssignable(varargElementType, value.type)) {
      error(node, "vararg element for function '" + info.name + "' must be " +
                      typeString(varargElementType) + ", got " +
                      typeString(value.type));
      ok = false;
      return Value{{}, varargElementType};
    }
    return value;
  };

  for (std::size_t index = 0; index < expandedArgs->size(); ++index) {
    if (!(*expandedArgs)[index]) {
      ok = false;
      continue;
    }
    std::size_t formalIndex = firstFormal + index;
    Value value = formalIndex < info.positionalCount
                      ? emitActual(*(*expandedArgs)[index], formalIndex)
                      : emitVarargActual(*(*expandedArgs)[index]);
    if (!value.value)
      continue;
    posargs.push_back(value);
    if (formalIndex < formalCount)
      filled[formalIndex] = true;
  }

  if (!expandedKeywords->empty()) {
    std::vector<std::string> seen;
    for (const StaticKeywordArg &keyword : *expandedKeywords) {
      if (!keyword.anchor || !keyword.value) {
        ok = false;
        continue;
      }
      if (std::find(seen.begin(), seen.end(), keyword.name) != seen.end()) {
        error(*keyword.anchor, "function '" + info.name +
                                   "' got duplicate keyword '" + keyword.name +
                                   "'");
        ok = false;
        continue;
      }
      seen.push_back(keyword.name);

      auto nameBegin = info.argNames.begin() + firstFormal;
      auto nameIt = std::find(nameBegin, info.argNames.end(), keyword.name);
      if (nameIt == info.argNames.end()) {
        if (!kwargValueType) {
          error(*keyword.anchor, "function '" + info.name +
                                     "' has no parameter '" + keyword.name +
                                     "'");
          ok = false;
          continue;
        }
        Value value =
            emitExpressionWithExpectedType(*keyword.value, kwargValueType);
        if (!value.value) {
          ok = false;
          continue;
        }
        if (!typeAssignable(kwargValueType, value.type)) {
          error(*keyword.value, "keyword argument '" + keyword.name +
                                    "' for function '" + info.name +
                                    "' must be " + typeString(kwargValueType) +
                                    ", got " + typeString(value.type));
          ok = false;
          continue;
        }
        mlir::Value key = builder.create<py::StrConstantOp>(
            loc(*keyword.anchor), strType(), keyword.name);
        kwnames.push_back(Value{key, strType()});
        kwvalues.push_back(value);
        continue;
      }
      std::size_t formalIndex = static_cast<std::size_t>(
          std::distance(info.argNames.begin(), nameIt));
      if (formalIndex < info.positionalOnlyCount) {
        error(*keyword.anchor, "function '" + info.name +
                                   "' got positional-only argument '" +
                                   keyword.name + "' passed as keyword");
        ok = false;
        continue;
      }
      std::size_t localIndex = formalIndex - firstFormal;
      std::size_t fixedPositionalSupplied =
          info.positionalCount > firstFormal
              ? std::min(expandedArgs->size(),
                         info.positionalCount - firstFormal)
              : 0;
      bool suppliedAsFixedPositional = localIndex < fixedPositionalSupplied;
      if (suppliedAsFixedPositional || filled[formalIndex]) {
        error(*keyword.anchor, "function '" + info.name +
                                   "' got multiple values for argument '" +
                                   keyword.name + "'");
        ok = false;
        continue;
      }

      Value value = emitActual(*keyword.value, formalIndex);
      if (!value.value)
        continue;
      mlir::Value key = builder.create<py::StrConstantOp>(
          loc(*keyword.anchor), strType(), keyword.name);
      kwnames.push_back(Value{key, strType()});
      kwvalues.push_back(value);
      filled[formalIndex] = true;
    }
  }

  for (std::size_t formalIndex = firstFormal; formalIndex < formalCount;
       ++formalIndex) {
    if (filled[formalIndex] || hasDefaultForFormal(info, formalIndex))
      continue;
    error(expr, "function '" + info.name + "' missing required argument '" +
                    info.argNames[formalIndex] + "'");
    ok = false;
  }
  if (!ok)
    return std::nullopt;

  auto emitStaticTuple = [&](const std::vector<Value> &elements) -> Value {
    if (elements.empty())
      return emitEmptyTuple();
    llvm::SmallVector<mlir::Type> types;
    llvm::SmallVector<mlir::Value> values;
    types.reserve(elements.size());
    values.reserve(elements.size());
    for (const Value &element : elements) {
      types.push_back(element.type);
      values.push_back(element.value);
    }
    py::TupleType tupleType = py::TupleType::get(&context, types);
    mlir::Value tuple =
        builder.create<py::TupleCreateOp>(loc(expr), tupleType, values);
    return Value{tuple, tupleType};
  };

  return CallArgumentTuples{emitStaticTuple(posargs), emitStaticTuple(kwnames),
                            emitStaticTuple(kwvalues)};
}

Value Builder::Impl::emitFromPrimCall(
    const parser::Node &expr, const std::vector<parser::NodePtr> &args) {
  const std::vector<parser::NodePtr> *keywords =
      nodeListField(expr, "keywords");
  if (keywords && !keywords->empty()) {
    error(expr, "from_prim keyword arguments are not supported");
    return Value{{}, intType()};
  }
  if (args.size() != 1 || !args.front()) {
    error(expr, "from_prim expects exactly one argument");
    return Value{{}, intType()};
  }
  Value input = emitExpression(*args.front());
  if (!input.value)
    return Value{{}, intType()};
  if (mlir::isa<mlir::RankedTensorType>(input.type)) {
    mlir::Value boxed =
        builder.create<py::CastFromPrimOp>(loc(expr), strType(), input.value);
    return Value{boxed, strType()};
  }
  if (mlir::isa<mlir::FloatType>(input.type)) {
    mlir::Value boxed =
        builder.create<py::CastFromPrimOp>(loc(expr), floatType(), input.value);
    return Value{boxed, floatType()};
  }
  if (!mlir::isa<mlir::IntegerType>(input.type)) {
    error(
        *args.front(),
        "from_prim currently supports only integer, float, and tensor values");
    return Value{{}, intType()};
  }
  mlir::Value boxed =
      builder.create<py::CastFromPrimOp>(loc(expr), intType(), input.value);
  return Value{boxed, intType()};
}

Value Builder::Impl::emitToPrimCall(const parser::Node &expr,
                                    const std::vector<parser::NodePtr> &args) {
  const std::vector<parser::NodePtr> *keywords =
      nodeListField(expr, "keywords");
  if (keywords && !keywords->empty()) {
    error(expr, "to_prim keyword arguments are not supported");
    return Value{{}, noneType()};
  }
  if (args.size() != 2 || !args[0] || !args[1]) {
    error(expr, "to_prim expects exactly two arguments");
    return Value{{}, noneType()};
  }

  Value input = emitExpression(*args[0]);
  if (!input.value)
    return Value{{}, noneType()};
  if (!py::isPyType(input.type)) {
    error(*args[0], "to_prim expects a Python object value");
    return Value{{}, noneType()};
  }

  std::optional<mlir::Type> primitiveType = typeFromAnnotation(args[1]);
  if (!primitiveType ||
      !mlir::isa<mlir::IntegerType, mlir::FloatType>(*primitiveType)) {
    error(*args[1],
          "to_prim second argument must be a primitive scalar annotation");
    return Value{{}, noneType()};
  }
  if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(*primitiveType)) {
    if (intTy.getWidth() != 32 && intTy.getWidth() != 64) {
      error(*args[1], "to_prim integer target currently supports Int[32] or "
                      "Int[64]");
      return Value{{}, *primitiveType};
    }
  }
  if (auto floatTy = mlir::dyn_cast<mlir::FloatType>(*primitiveType)) {
    if (floatTy.getWidth() != 64) {
      error(*args[1], "to_prim float target currently supports Float[64]");
      return Value{{}, *primitiveType};
    }
  }

  mlir::Value primitive = builder.create<py::CastToPrimOp>(
      loc(expr), *primitiveType, input.value, "exact");
  return Value{primitive, *primitiveType};
}

Value Builder::Impl::emitExceptionCall(
    const parser::Node &expr, llvm::StringRef className,
    const std::vector<parser::NodePtr> &args) {
  const std::vector<parser::NodePtr> *keywords =
      nodeListField(expr, "keywords");
  if (keywords && !keywords->empty()) {
    error(expr, className.str() + " keyword arguments are not supported yet");
    return Value{{}, exceptionType()};
  }
  if (args.size() > 1) {
    error(expr, className.str() + "(...) supports zero or one argument");
    return Value{{}, exceptionType()};
  }

  llvm::SmallVector<mlir::Value> messageArgs;
  if (!args.empty() && args.front()) {
    Value message = emitExpression(*args.front());
    if (message.type != strType()) {
      error(*args.front(), className.str() + "(...) argument must be !py.str");
      return Value{{}, exceptionType()};
    }
    messageArgs.push_back(message.value);
  }
  auto exception = builder.create<py::ExceptionNewOp>(
      loc(expr), exceptionType(), messageArgs);
  exception->setAttr("py.exception.class", builder.getStringAttr(className));
  return Value{exception.getResult(), exceptionType()};
}

Value Builder::Impl::emitFunctionCall(
    const parser::Node &expr, const FunctionInfo &info,
    const std::vector<parser::NodePtr> &args) {
  if (!info.isSpecialization && !info.varargType && hasCallableFormal(info)) {
    std::optional<FunctionInfo> specialized =
        specializeFunctionCall(expr, info, args);
    if (!specialized)
      return Value{{}, info.resultType};
    return emitFunctionCall(expr, *specialized, args);
  }

  if (!info.isNative && !info.isAsync && hasCallableMetadata(info)) {
    std::optional<CallArgumentTuples> tuples =
        emitExplicitCallArgumentTuples(expr, info, args);
    if (!tuples)
      return Value{{}, info.resultType};

    mlir::Value callee;
    auto bound = symbols.find(info.name);
    if (bound != symbols.end() && bound->second.type == info.functionType)
      callee = bound->second.value;
    else {
      Value materialized = emitFunctionObject(expr, info);
      if (!materialized.value)
        return Value{{}, info.resultType};
      callee = materialized.value;
    }

    if (info.mayThrow)
      return emitMayThrowFunctionCall(expr, info, callee, tuples->posargs,
                                      tuples->kwnames, tuples->kwvalues);
    auto call = builder.create<py::CallVectorOp>(
        loc(expr), mlir::TypeRange{info.resultType}, callee,
        tuples->posargs.value, tuples->kwnames.value, tuples->kwvalues.value,
        mlir::UnitAttr{});
    attachReturnedCallableSymbol(call.getOperation(), info);
    return Value{call.getResults().front(), info.resultType};
  }

  std::optional<std::vector<Value>> argValues =
      emitStaticArguments(expr, info, args);
  if (!argValues)
    return Value{{}, info.resultType};

  if (!info.mayThrow && info.returnedCallableArgIndex) {
    std::size_t index = *info.returnedCallableArgIndex;
    if (index >= argValues->size()) {
      error(expr, "returned callable argument summary for function '" +
                      info.name + "' has invalid argument index");
      return Value{{}, info.resultType};
    }
    Value returned = (*argValues)[index];
    if (!mlir::isa<py::FuncType>(returned.type)) {
      error(expr, "returned callable argument for function '" + info.name +
                      "' must be !py.func, got " + typeString(returned.type));
      return Value{{}, info.resultType};
    }
    return returned;
  }

  if (!info.mayThrow && info.returnedCallable) {
    FunctionInfo returned = info.returnedCallable->info;
    if (returned.closureCaptures.size() !=
        info.returnedCallable->closureCaptureArgIndices.size()) {
      error(expr, "returned callable summary for function '" + info.name +
                      "' has inconsistent closure metadata");
      return Value{{}, info.resultType};
    }
    for (auto indexed : llvm::enumerate(returned.closureCaptures)) {
      std::optional<std::size_t> argIndex =
          info.returnedCallable->closureCaptureArgIndices[indexed.index()];
      if (!argIndex || *argIndex >= argValues->size()) {
        error(expr, "returned callable summary for function '" + info.name +
                        "' cannot map closure capture '" +
                        indexed.value().name + "'");
        return Value{{}, info.resultType};
      }
      indexed.value().value = (*argValues)[*argIndex];
    }
    Value callable = emitFunctionObject(expr, returned);
    if (!callable.value)
      return Value{{}, info.resultType};
    return callable;
  }

  llvm::SmallVector<mlir::Value> operands;
  for (const Value &argValue : *argValues)
    operands.push_back(argValue.value);

  if (info.isAsync) {
    mlir::Type resultType = coroutineType(info.resultType);
    mlir::Value coroutine = builder.create<py::CoroCreateOp>(
        loc(expr), resultType, info.symbolName, operands);
    return Value{coroutine, resultType};
  }

  if (info.isNative) {
    auto call = builder.create<mlir::func::CallOp>(
        loc(expr), info.symbolName,
        info.resultType == noneType() ? mlir::TypeRange{}
                                      : mlir::TypeRange{info.resultType},
        operands);
    if (info.resultType == noneType()) {
      mlir::Value none = builder.create<py::NoneOp>(loc(expr), noneType());
      return Value{none, noneType()};
    }
    return Value{call.getResults().front(), info.resultType};
  }

  mlir::Value callee = builder.create<py::FuncObjectOp>(
      loc(), info.functionType, info.symbolName);

  CallArgumentTuples tuples = emitCallArgumentTuples(info, *argValues);
  if (info.mayThrow)
    return emitMayThrowFunctionCall(expr, info, callee, tuples.posargs,
                                    tuples.kwnames, tuples.kwvalues);
  auto call = builder.create<py::CallVectorOp>(
      loc(), mlir::TypeRange{info.resultType}, callee, tuples.posargs.value,
      tuples.kwnames.value, tuples.kwvalues.value, mlir::UnitAttr{});
  attachReturnedCallableSymbol(call.getOperation(), info);
  return Value{call.getResults().front(), info.resultType};
}

CallArgumentTuples
Builder::Impl::emitCallArgumentTuples(const FunctionInfo &info,
                                      const std::vector<Value> &actuals) {
  (void)info;
  // Static calls are fully resolved by emitStaticArguments. Keep keyword-only
  // values in formal order so lowerings can call the target function directly.
  return CallArgumentTuples{emitTuple(actuals), emitEmptyTuple(),
                            emitEmptyTuple()};
}

Value Builder::Impl::emitMayThrowFunctionCall(const parser::Node &expr,
                                              const FunctionInfo &info,
                                              mlir::Value callee, Value posargs,
                                              Value kwnames, Value kwvalues) {
  return emitMayThrowCallableCall(expr, callee, info.resultType, posargs,
                                  kwnames, kwvalues);
}

Value Builder::Impl::emitMayThrowCallableCall(const parser::Node &expr,
                                              mlir::Value callee,
                                              mlir::Type resultType,
                                              Value posargs, Value kwnames,
                                              Value kwvalues) {
  std::optional<mlir::Value> normalSeed;
  if (resultType != noneType()) {
    normalSeed = invokeNormalSeed(builder, loc(expr), resultType);
    if (!normalSeed) {
      error(expr, "C++ emitter does not support maythrow calls returning " +
                      typeString(resultType) + " yet");
      return Value{{}, resultType};
    }
  }

  mlir::Region *region = builder.getBlock()->getParent();
  mlir::Block *normalBlock = new mlir::Block();
  mlir::Block *unwindBlock = new mlir::Block();
  mlir::Value exceptionNull =
      builder.create<py::ExceptionNullOp>(loc(expr), exceptionType());
  if (normalSeed)
    normalBlock->addArgument(resultType, loc(expr));
  unwindBlock->addArgument(exceptionType(), loc(expr));
  region->push_back(normalBlock);
  region->push_back(unwindBlock);

  llvm::SmallVector<mlir::Value> normalOperands;
  if (normalSeed)
    normalOperands.push_back(*normalSeed);
  builder.create<py::InvokeOp>(loc(expr), callee, posargs.value, kwnames.value,
                               kwvalues.value, mlir::ValueRange{normalOperands},
                               mlir::ValueRange{exceptionNull}, normalBlock,
                               unwindBlock);

  builder.setInsertionPointToStart(unwindBlock);
  builder.create<py::RaiseCurrentOp>(loc(expr));

  builder.setInsertionPointToStart(normalBlock);
  blockTerminated = false;
  if (normalSeed)
    return Value{normalBlock->getArgument(0), resultType};
  mlir::Value none = builder.create<py::NoneOp>(loc(expr), noneType());
  return Value{none, noneType()};
}

Value Builder::Impl::emitListConstructorCall(
    const parser::Node &expr, const std::vector<parser::NodePtr> &args) {
  const std::vector<parser::NodePtr> *keywords =
      nodeListField(expr, "keywords");
  if (keywords && !keywords->empty()) {
    error(expr, "list() does not accept keyword arguments");
    return Value{{}, noneType()};
  }
  if (args.empty()) {
    error(expr, "list() requires an expected list[...] type when called "
                "without arguments");
    return Value{{}, noneType()};
  }
  if (args.size() != 1 || !args.front()) {
    error(expr, "list() expects at most one argument");
    return Value{{}, noneType()};
  }
  if (args.front()->kind == "GeneratorExp")
    return emitListComprehension(*args.front());
  if (args.front()->kind == "Call") {
    const parser::NodePtr *rangeFunc = nodeField(*args.front(), "func");
    if (rangeFunc && *rangeFunc && isNameRef(**rangeFunc, "range")) {
      std::optional<StaticRangeElements> range =
          emitStaticRangeElements(*args.front(), "list()");
      if (!range)
        return Value{{}, noneType()};
      return emitListFromValues(expr, range->values, range->elementType);
    }
  }

  Value source = emitExpression(*args.front());
  if (!source.value)
    return Value{{}, noneType()};
  std::optional<std::vector<Value>> elements =
      finiteSequenceElements(*args.front(), source, "list()");
  if (!elements)
    return Value{{}, noneType()};
  if (elements->empty()) {
    mlir::Type elementType = listElementType(source.type);
    if (!elementType) {
      error(*args.front(),
            "list() empty tuple conversion requires an expected list[...] "
            "type");
      return Value{{}, noneType()};
    }
    return emitListFromValues(expr, {}, elementType);
  }
  return emitListFromValues(expr, *elements);
}

Value Builder::Impl::emitTupleConstructorCall(
    const parser::Node &expr, const std::vector<parser::NodePtr> &args) {
  const std::vector<parser::NodePtr> *keywords =
      nodeListField(expr, "keywords");
  if (keywords && !keywords->empty()) {
    error(expr, "tuple() does not accept keyword arguments");
    return Value{{}, noneType()};
  }
  if (args.empty())
    return emitEmptyTuple();
  if (args.size() != 1 || !args.front()) {
    error(expr, "tuple() expects at most one argument");
    return Value{{}, noneType()};
  }
  if (args.front()->kind == "GeneratorExp") {
    std::optional<std::vector<Value>> elements =
        emitStaticGeneratorElements(*args.front(), "tuple()");
    if (!elements)
      return Value{{}, noneType()};
    return emitTuple(*elements);
  }
  if (args.front()->kind == "Call") {
    const parser::NodePtr *rangeFunc = nodeField(*args.front(), "func");
    if (rangeFunc && *rangeFunc && isNameRef(**rangeFunc, "range")) {
      std::optional<StaticRangeElements> range =
          emitStaticRangeElements(*args.front(), "tuple()");
      if (!range)
        return Value{{}, noneType()};
      return emitTuple(range->values);
    }
  }

  Value source = emitExpression(*args.front());
  if (!source.value)
    return Value{{}, noneType()};
  std::optional<std::vector<Value>> elements =
      finiteSequenceElements(*args.front(), source, "tuple()");
  if (!elements)
    return Value{{}, noneType()};
  return emitTuple(*elements);
}

Value Builder::Impl::emitBoolConstructorCall(
    const parser::Node &expr, const std::vector<parser::NodePtr> &args) {
  const std::vector<parser::NodePtr> *keywords =
      nodeListField(expr, "keywords");
  if (keywords && !keywords->empty()) {
    error(expr, "bool() does not accept keyword arguments");
    return Value{{}, boolType()};
  }
  if (args.empty()) {
    mlir::Value bit =
        builder.create<mlir::arith::ConstantIntOp>(loc(expr), false, 1);
    mlir::Value result =
        builder.create<py::CastFromPrimOp>(loc(expr), boolType(), bit);
    return Value{result, boolType()};
  }
  if (args.size() != 1 || !args.front()) {
    error(expr, "bool() expects at most one argument");
    return Value{{}, boolType()};
  }

  Value condition = emitCondition(*args.front());
  if (!condition.value)
    return Value{{}, boolType()};
  mlir::Value result = builder.create<py::CastFromPrimOp>(loc(expr), boolType(),
                                                          condition.value);
  return Value{result, boolType()};
}

Value Builder::Impl::emitIntConstructorCall(
    const parser::Node &expr, const std::vector<parser::NodePtr> &args) {
  const std::vector<parser::NodePtr> *keywords =
      nodeListField(expr, "keywords");
  if (keywords && !keywords->empty()) {
    error(expr, "int() keyword/base arguments are not supported by the C++ "
                "emitter yet");
    return Value{{}, intType()};
  }
  if (args.empty()) {
    mlir::Value result =
        builder.create<py::IntConstantOp>(loc(expr), intType(), "0");
    return Value{result, intType()};
  }
  if (args.size() != 1 || !args.front()) {
    error(expr, "int() expects at most one positional argument in the C++ "
                "emitter");
    return Value{{}, intType()};
  }

  if (std::optional<std::string> literal =
          staticIntConstructorDecimal(*args.front())) {
    mlir::Value result =
        builder.create<py::IntConstantOp>(loc(expr), intType(), *literal);
    return Value{result, intType()};
  }

  Value value = emitExpression(*args.front());
  if (!value.value)
    return Value{{}, intType()};
  if (value.type == intType())
    return value;

  error(*args.front(), "int() currently supports only static numeric/bool "
                       "literals or !py.int values");
  return Value{{}, intType()};
}

Value Builder::Impl::emitFloatConstructorCall(
    const parser::Node &expr, const std::vector<parser::NodePtr> &args) {
  const std::vector<parser::NodePtr> *keywords =
      nodeListField(expr, "keywords");
  if (keywords && !keywords->empty()) {
    error(expr, "float() keyword arguments are not supported by the C++ "
                "emitter");
    return Value{{}, floatType()};
  }
  if (args.empty()) {
    mlir::Value result = builder.create<py::FloatConstantOp>(
        loc(expr), floatType(), builder.getF64FloatAttr(0.0));
    return Value{result, floatType()};
  }
  if (args.size() != 1 || !args.front()) {
    error(expr, "float() expects at most one positional argument");
    return Value{{}, floatType()};
  }

  if (std::optional<double> literal =
          staticFloatConstructorValue(*args.front())) {
    mlir::Value result = builder.create<py::FloatConstantOp>(
        loc(expr), floatType(), builder.getF64FloatAttr(*literal));
    return Value{result, floatType()};
  }

  Value value = emitExpression(*args.front());
  if (!value.value)
    return Value{{}, floatType()};
  if (value.type == floatType())
    return value;
  if (value.type == intType()) {
    if (std::optional<std::int64_t> integer = staticPyIntValue(value.value)) {
      mlir::Value result = builder.create<py::FloatConstantOp>(
          loc(expr), floatType(),
          builder.getF64FloatAttr(static_cast<double>(*integer)));
      return Value{result, floatType()};
    }
  }

  error(*args.front(), "float() currently supports only static numeric/bool "
                       "literals, !py.float values, or static !py.int values");
  return Value{{}, floatType()};
}

Value Builder::Impl::emitStrConstructorCall(
    const parser::Node &expr, const std::vector<parser::NodePtr> &args) {
  const std::vector<parser::NodePtr> *keywords =
      nodeListField(expr, "keywords");
  if (keywords && !keywords->empty()) {
    error(expr, "str() keyword arguments are not supported by the C++ emitter");
    return Value{{}, strType()};
  }
  if (args.empty()) {
    mlir::Value result =
        builder.create<py::StrConstantOp>(loc(expr), strType(), "");
    return Value{result, strType()};
  }
  if (args.size() != 1 || !args.front()) {
    error(expr, "str() expects at most one positional argument in the C++ "
                "emitter");
    return Value{{}, strType()};
  }

  Value value = emitExpression(*args.front());
  if (!value.value)
    return Value{{}, strType()};
  if (value.type == strType())
    return value;
  return emitRepr(value);
}

Value Builder::Impl::emitReprCall(const parser::Node &expr,
                                  const std::vector<parser::NodePtr> &args) {
  const std::vector<parser::NodePtr> *keywords =
      nodeListField(expr, "keywords");
  if (keywords && !keywords->empty()) {
    error(expr, "repr() does not accept keyword arguments");
    return Value{{}, strType()};
  }
  if (args.size() != 1 || !args.front()) {
    error(expr, "repr() expects exactly one argument");
    return Value{{}, strType()};
  }

  Value value = emitExpression(*args.front());
  if (!value.value)
    return Value{{}, strType()};
  if (value.type == strType()) {
    auto literal = value.value.getDefiningOp<py::StrConstantOp>();
    if (!literal) {
      error(*args.front(),
            "repr(str) requires a statically known string literal in the C++ "
            "emitter");
      return Value{{}, strType()};
    }
    mlir::Value result = builder.create<py::StrConstantOp>(
        loc(expr), strType(), pythonStringRepr(literal.getValue()));
    return Value{result, strType()};
  }
  return emitRepr(value);
}

Value Builder::Impl::emitLenCall(const parser::Node &expr,
                                 const std::vector<parser::NodePtr> &args) {
  const std::vector<parser::NodePtr> *keywords =
      nodeListField(expr, "keywords");
  if (args.size() != 1 || !args.front()) {
    error(expr, "len() expects exactly one argument");
    return Value{{}, intType()};
  }
  if (keywords && !keywords->empty()) {
    error(expr, "len() does not accept keyword arguments");
    return Value{{}, intType()};
  }

  if (args.front()->kind == "Call") {
    const parser::NodePtr *rangeFunc = nodeField(*args.front(), "func");
    if (rangeFunc && *rangeFunc && isNameRef(**rangeFunc, "range")) {
      std::optional<StaticRangeSpec> spec =
          staticRangeSpec(*args.front(), "len()");
      if (!spec)
        return Value{{}, intType()};
      std::optional<std::size_t> length = staticRangeLength(*spec);
      if (!length) {
        error(*args.front(), "len() range length is too large to represent");
        return Value{{}, intType()};
      }
      mlir::Value result = builder.create<py::IntConstantOp>(
          loc(expr), intType(), std::to_string(*length));
      return Value{result, intType()};
    }
  }

  Value input = emitExpression(*args.front());
  if (!input.value)
    return Value{{}, intType()};

  std::optional<std::size_t> length =
      staticLength(*args.front(), input, "len()");
  if (!length)
    return Value{{}, intType()};

  mlir::Value result = builder.create<py::IntConstantOp>(
      loc(expr), intType(), std::to_string(*length));
  return Value{result, intType()};
}

Value Builder::Impl::emitPrintCall(const parser::Node &expr,
                                   const std::vector<parser::NodePtr> &args) {
  std::string separator = " ";
  std::optional<std::string> end;
  std::optional<std::vector<StaticKeywordArg>> expandedKeywords =
      expandStaticCallKeywords(expr, "print");
  if (!expandedKeywords)
    return Value{{}, noneType()};

  std::vector<std::string> seenKeywords;
  for (const StaticKeywordArg &keyword : *expandedKeywords) {
    if (!keyword.anchor || !keyword.value)
      return Value{{}, noneType()};
    if (std::find(seenKeywords.begin(), seenKeywords.end(), keyword.name) !=
        seenKeywords.end()) {
      error(*keyword.anchor,
            "print got duplicate keyword '" + keyword.name + "'");
      return Value{{}, noneType()};
    }
    seenKeywords.push_back(keyword.name);

    if (keyword.name != "sep" && keyword.name != "end" &&
        keyword.name != "file" && keyword.name != "flush") {
      error(*keyword.anchor, "print keyword '" + keyword.name +
                                 "' is not supported by the C++ emitter yet");
      return Value{{}, noneType()};
    }

    if (keyword.name == "file") {
      if (!isStaticNoneConstant(*keyword.value)) {
        error(*keyword.value,
              "print file must be None for the current stdout-only host I/O "
              "boundary");
        return Value{{}, noneType()};
      }
      continue;
    }
    if (keyword.name == "flush") {
      if (isStaticNoneConstant(*keyword.value))
        continue;
      std::optional<bool> flush = staticBoolConstant(*keyword.value);
      if (!flush) {
        error(*keyword.value,
              "print flush must be a statically known bool or None");
        return Value{{}, noneType()};
      }
      if (*flush) {
        error(*keyword.value,
              "print flush=True requires a host I/O flush boundary and is not "
              "implemented in the C++ emitter yet");
        return Value{{}, noneType()};
      }
      continue;
    }

    if (isStaticNoneConstant(*keyword.value)) {
      if (keyword.name == "sep")
        separator = " ";
      else
        end.reset();
      continue;
    }
    std::optional<std::string> value = staticStringConstant(*keyword.value);
    if (!value) {
      error(*keyword.value, "print " + keyword.name +
                                " must be a statically known string constant "
                                "or None");
      return Value{{}, noneType()};
    }
    if (keyword.name == "sep")
      separator = *value;
    else
      end = *value;
  }

  py::TupleType varargType = py::TupleType::get(&context, {strType()});
  py::FuncSignatureType signature =
      functionSignatureType({}, noneType(), varargType);
  py::FuncType printType = py::FuncType::get(&context, signature);
  mlir::Value callee = builder.create<py::FuncObjectOp>(
      loc(), printType, end ? "__builtin_print_raw" : "__builtin_print");

  std::optional<std::vector<const parser::Node *>> expandedArgs =
      expandStaticCallArgs(expr, args, std::nullopt, "print");
  if (!expandedArgs)
    return Value{{}, noneType()};

  std::vector<Value> printArgs;
  for (const parser::Node *arg : *expandedArgs) {
    if (!arg)
      continue;
    Value value = emitExpression(*arg);
    if (!value.value)
      return Value{{}, noneType()};
    if (value.type != strType())
      value = emitRepr(value);
    if (!value.value)
      return Value{{}, noneType()};
    printArgs.push_back(std::move(value));
  }
  if (printArgs.empty()) {
    mlir::Value empty =
        builder.create<py::StrConstantOp>(loc(expr), strType(), "");
    printArgs.push_back(Value{empty, strType()});
  }
  if (printArgs.size() > 1) {
    Value joined = printArgs.front();
    for (std::size_t i = 1; i < printArgs.size(); ++i) {
      mlir::Value lhs = joined.value;
      if (!separator.empty()) {
        mlir::Value sep =
            builder.create<py::StrConstantOp>(loc(expr), strType(), separator);
        lhs = builder.create<py::AddOp>(loc(expr), strType(), lhs, sep);
      }
      mlir::Value concatenated = builder.create<py::AddOp>(
          loc(expr), strType(), lhs, printArgs[i].value);
      joined = Value{concatenated, strType()};
    }
    printArgs = {joined};
  }
  if (end) {
    mlir::Value suffix =
        builder.create<py::StrConstantOp>(loc(expr), strType(), *end);
    mlir::Value withEnd = builder.create<py::AddOp>(
        loc(expr), strType(), printArgs.front().value, suffix);
    printArgs = {Value{withEnd, strType()}};
  }

  Value posargs = emitTuple(printArgs);
  Value kwnames = emitEmptyTuple();
  Value kwvalues = emitEmptyTuple();
  auto call = builder.create<py::CallVectorOp>(
      loc(), mlir::TypeRange{noneType()}, callee, posargs.value, kwnames.value,
      kwvalues.value, mlir::UnitAttr{});
  return Value{call.getResults().front(), noneType()};
}

Value Builder::Impl::emitRepr(const Value &input) {
  if (!input.value)
    return Value{{}, strType()};
  mlir::Value result =
      builder.create<py::ReprOp>(loc(), strType(), input.value);
  return Value{result, strType()};
}

Value Builder::Impl::emitTuple(const std::vector<Value> &elements) {
  if (elements.empty())
    return emitEmptyTuple();
  llvm::SmallVector<mlir::Type> types;
  llvm::SmallVector<mlir::Value> values;
  types.reserve(elements.size());
  values.reserve(elements.size());
  for (const Value &element : elements) {
    types.push_back(element.type);
    values.push_back(element.value);
  }
  py::TupleType tupleType = py::TupleType::get(&context, types);
  mlir::Value tuple =
      builder.create<py::TupleCreateOp>(loc(), tupleType, values);
  return Value{tuple, tupleType};
}

Value Builder::Impl::emitEmptyTuple() {
  py::TupleType tupleType = py::TupleType::get(&context, {});
  mlir::Value tuple = builder.create<py::TupleEmptyOp>(loc(), tupleType);
  return Value{tuple, tupleType};
}

} // namespace lython::emitter
