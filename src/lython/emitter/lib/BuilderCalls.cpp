#include "BuilderImpl.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
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
  return stripped.getDefiningOp<py::CallableObjectOp>() ||
         stripped.getDefiningOp<py::MakeFunctionOp>();
}

std::optional<std::string> callableTargetSymbol(mlir::Value callable) {
  mlir::Value stripped = stripCallableCasts(callable);
  if (auto funcObject = stripped.getDefiningOp<py::CallableObjectOp>())
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
  if (mlir::isa<py::IntType>(type))
    return builder.create<py::IntConstantOp>(
        loc, py::IntType::get(builder.getContext()), "0");
  if (mlir::isa<py::FloatType>(type))
    return builder.create<py::FloatConstantOp>(
        loc, py::FloatType::get(builder.getContext()),
        builder.getF64FloatAttr(0.0));
  if (mlir::isa<py::StrType>(type))
    return builder.create<py::StrConstantOp>(
        loc, py::StrType::get(builder.getContext()), "");
  if (mlir::isa<py::NoneType>(type))
    return builder.create<py::NoneOp>(loc,
                                      py::NoneType::get(builder.getContext()));
  if (mlir::isa<py::BoolType>(type)) {
    mlir::Type intType = py::IntType::get(builder.getContext());
    mlir::Value zero = builder.create<py::IntConstantOp>(loc, intType, "0");
    mlir::Value one = builder.create<py::IntConstantOp>(loc, intType, "1");
    return builder.create<py::EqOp>(loc, type, zero, one);
  }
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

Value Builder::Impl::emitCall(const parser::Node &expr) {
  const parser::NodePtr *func = nodeField(expr, "func");
  const std::vector<parser::NodePtr> *args = nodeListField(expr, "args");
  if (!func || !*func || !args) {
    error(expr, "Call.func or Call.args is missing");
    return Value{{}, noneType()};
  }

  if ((*func)->kind == "Subscript") {
    const parser::NodePtr *value = nodeField(**func, "value");
    const parser::NodePtr *slice = nodeField(**func, "slice");
    if (value && *value && (*value)->kind == "Name" && slice && *slice) {
      const std::string *name = stringField(**value, "id");
      std::optional<FunctionInfo> info;
      if (name && !symbols.count(*name)) {
        auto local = localFunctions.find(*name);
        if (local != localFunctions.end())
          info = local->second;
      }
      if (name && !symbols.count(*name) && !info) {
        auto found = functions.find(*name);
        if (found != functions.end())
          info = found->second;
      }
      if (info && !info->typeParameters.empty()) {
        llvm::SmallVector<parser::NodePtr> typeArgs;
        if ((*slice)->kind == "Tuple") {
          const std::vector<parser::NodePtr> *items =
              nodeListField(**slice, "elts");
          if (items)
            typeArgs.append(items->begin(), items->end());
        } else {
          typeArgs.push_back(*slice);
        }
        if (typeArgs.size() != info->typeParameters.size()) {
          error(**func, "generic function '" + info->name + "' expects " +
                            std::to_string(info->typeParameters.size()) +
                            " type arguments, got " +
                            std::to_string(typeArgs.size()));
          return Value{{}, info->resultType};
        }
        std::map<std::string, mlir::Type> explicitTypeArguments;
        bool ok = true;
        for (auto indexed : llvm::enumerate(typeArgs)) {
          const TypeAliasParameter &parameter =
              info->typeParameters[indexed.index()];
          if (parameter.kind != TypeAliasParameterKind::TypeVar) {
            error(*indexed.value(),
                  "generic function type parameter '" + parameter.name +
                      "' cannot be explicitly specialized yet");
            ok = false;
            continue;
          }
          std::optional<mlir::Type> type = typeFromAnnotation(indexed.value());
          if (!type) {
            error(*indexed.value(),
                  "generic function '" + info->name +
                      "' type argument must be a static type annotation");
            ok = false;
            continue;
          }
          explicitTypeArguments[parameter.name] = *type;
        }
        if (!ok)
          return Value{{}, info->resultType};
        std::optional<FunctionInfo> specialized = specializeGenericFunctionCall(
            expr, *info, *args, &explicitTypeArguments);
        if (!specialized)
          return Value{{}, info->resultType};
        return emitFunctionCall(expr, *specialized, *args);
      }
    }
  }

  if (std::optional<PrimitiveConstant> constant =
          primitiveIntConstructorConstant(expr))
    return emitPrimitiveIntConstructor(expr, *constant);
  if (std::optional<mlir::Type> targetType = typeFromAnnotation(*func);
      targetType && mlir::isa<mlir::IntegerType, mlir::FloatType>(*targetType))
    return emitPrimitiveScalarConstructor(expr, *targetType, *args);
  if (std::optional<mlir::Type> targetType = typeFromAnnotation(*func);
      targetType) {
    if (std::optional<std::string> className = classNameFromType(*targetType))
      return emitClassConstructorCall(expr, *className, *args);
  }
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
    if (name && *name == "isinstance")
      return emitIsinstanceCall(expr);
    if (name && *name == "iter")
      return emitIterCall(expr, *args);
    if (name && *name == "next")
      return emitNextCall(expr, *args);
    if (name && *name == "len")
      return emitLenCall(expr, *args);
    if (name && *name == "list")
      return emitListConstructorCall(expr, *args);
    if (name && *name == "tuple")
      return emitTupleConstructorCall(expr, *args);
    if (name && isBuiltinExceptionClass(*name))
      return emitExceptionCall(expr, *name, *args);
    if (name && classes.count(*name) && !classes.at(*name).isGenericTemplate)
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
    if (py::isCallableType(callable.type)) {
      std::optional<FunctionInfo> info =
          callable.callableInfo
              ? std::optional<FunctionInfo>(*callable.callableInfo)
              : resolveCallableInfo(callable.value);
      if (info) {
        if (info->isNative || info->isAsync) {
          error(expr, "Callable value calls for native or async functions are "
                      "not supported yet");
          return Value{{}, info->resultType};
        }
        if (!typeAssignable(info->functionType, callable.type)) {
          error(expr, "Callable value type does not match resolved function "
                      "metadata for '" +
                          info->name + "'");
          return Value{{}, info->resultType};
        }

        mlir::Value callee = callable.value;
        if (std::optional<FunctionInfo> specialized =
                specializeParamSpecForwardingFunctionCall(expr, *info, *args)) {
          info = *specialized;
          mlir::Value stripped = stripCallableCasts(callable.value);
          if (auto makeFunction =
                  stripped.getDefiningOp<py::MakeFunctionOp>()) {
            callee = builder.create<py::MakeFunctionOp>(
                loc(expr), info->functionType, info->symbolName,
                makeFunction.getDefaults(), makeFunction.getKwdefaults(),
                makeFunction.getClosure(), makeFunction.getAnnotations(),
                makeFunction.getModule());
          } else if (auto funcObject =
                         stripped.getDefiningOp<py::CallableObjectOp>()) {
            (void)funcObject;
            callee = builder.create<py::CallableObjectOp>(
                loc(expr), info->functionType, info->symbolName);
          } else {
            Value materialized = emitFunctionObject(expr, *info);
            if (!materialized.value)
              return Value{{}, info->resultType};
            callee = materialized.value;
          }
        }

        std::optional<CallArgumentTuples> tuples =
            emitExplicitCallArgumentTuples(expr, *info, *args);
        if (!tuples)
          return Value{{}, info->resultType};

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
        auto call = builder.create<py::CallOp>(
            loc(expr), mlir::TypeRange{info->resultType}, callee,
            tuples->posargs.value, tuples->kwnames.value,
            tuples->kwvalues.value);
        attachReturnedCallableSymbol(call.getOperation(), *info);
        return applyReturnedClassSummary(
            Value{call.getResults().front(), info->resultType}, *info);
      }

      py::CallableType signature = py::getCallableContract(callable.type);
      if (!signature) {
        error(expr, "Callable value has no recoverable Callable contract");
        return Value{{}, noneType()};
      }
      llvm::ArrayRef<mlir::Type> resultTypes = signature.getResultTypes();
      if (resultTypes.size() > 1) {
        error(expr, "Callable value calls with multiple return values are not "
                    "supported yet");
        return Value{{}, noneType()};
      }

      FunctionInfo contractInfo;
      contractInfo.name = "<callable>";
      contractInfo.symbolName = "<callable>";
      contractInfo.functionType = callable.type;
      contractInfo.signatureType = signature;
      contractInfo.resultType =
          resultTypes.empty() ? noneType() : resultTypes.front();
      contractInfo.positionalOnlyCount = signature.getPositionalOnlyCount();
      llvm::ArrayRef<mlir::Type> positionalTypes =
          signature.getPositionalTypes();
      llvm::ArrayRef<mlir::StringAttr> positionalNames =
          signature.getPositionalNames();
      contractInfo.positionalCount = positionalTypes.size();
      for (auto indexed : llvm::enumerate(positionalTypes)) {
        contractInfo.argTypes.push_back(indexed.value());
        if (indexed.index() < positionalNames.size())
          contractInfo.argNames.push_back(
              positionalNames[indexed.index()].getValue().str());
        else
          contractInfo.argNames.push_back("arg" +
                                          std::to_string(indexed.index()));
      }
      llvm::ArrayRef<mlir::StringAttr> kwonlyNames = signature.getKwOnlyNames();
      for (auto indexed : llvm::enumerate(signature.getKwOnlyTypes())) {
        contractInfo.argTypes.push_back(indexed.value());
        std::string name = indexed.index() < kwonlyNames.size()
                               ? kwonlyNames[indexed.index()].getValue().str()
                               : "kw" + std::to_string(indexed.index());
        contractInfo.argNames.push_back(name);
        contractInfo.kwonlyNames.push_back(std::move(name));
      }
      if (signature.hasVararg())
        contractInfo.varargType = signature.getVarargType();
      if (signature.hasKwarg())
        contractInfo.kwargType = signature.getKwargType();

      std::optional<CallArgumentTuples> tuples =
          emitExplicitCallArgumentTuples(expr, contractInfo, *args);
      if (!tuples)
        return Value{{}, contractInfo.resultType};

      auto callableKnownNoThrow = [&]() -> bool {
        mlir::Value stripped = stripCallableCasts(callable.value);
        mlir::SymbolRefAttr target;
        if (auto funcObject = stripped.getDefiningOp<py::CallableObjectOp>())
          target = funcObject.getTargetAttr();
        else if (auto makeFunction =
                     stripped.getDefiningOp<py::MakeFunctionOp>())
          target = makeFunction.getTargetAttr();
        else if (mlir::Operation *definingOp = stripped.getDefiningOp())
          target = definingOp->getAttrOfType<mlir::SymbolRefAttr>(
              "ly.returned_callable_symbol");
        if (!target)
          return false;
        auto func = module->lookupSymbol<py::CallableFuncOp>(target);
        return func && static_cast<bool>(func->getAttr("nothrow"));
      };

      if (!callableKnownNoThrow())
        return emitMayThrowCallableCall(
            expr, callable.value, contractInfo.resultType, tuples->posargs,
            tuples->kwnames, tuples->kwvalues);

      mlir::TypeRange callResultTypes = resultTypes.empty()
                                            ? mlir::TypeRange{}
                                            : mlir::TypeRange(resultTypes);
      auto call = builder.create<py::CallOp>(
          loc(expr), callResultTypes, callable.value, tuples->posargs.value,
          tuples->kwnames.value, tuples->kwvalues.value);
      if (resultTypes.empty()) {
        mlir::Value none = builder.create<py::NoneOp>(loc(expr), noneType());
        return Value{none, noneType()};
      }
      return Value{call.getResults().front(), resultTypes.front()};
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
    if (!py::isCallableType(info.argTypes[formalIndex]))
      continue;

    const parser::Node *actual = actuals[formalIndex];
    std::optional<FunctionInfo> actualInfo;
    if (actual)
      actualInfo = resolveCallableInfo(*actual);
    if (!actualInfo && actual && actual->kind == "Lambda") {
      auto cached = lambdaCallableInfos.find(actual);
      if (cached != lambdaCallableInfos.end()) {
        actualInfo = cached->second;
      } else if (std::optional<FunctionInfo> parsed =
                     parseLambdaInfo(*actual, info.argTypes[formalIndex])) {
        lambdaCallableInfos.emplace(actual, *parsed);
        actualInfo = *parsed;
      }
    }
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
    if (!typeAssignable(info.argTypes[formalIndex], actualInfo->functionType)) {
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

std::optional<FunctionInfo> Builder::Impl::specializeProtocolFunctionCall(
    const parser::Node &expr, const FunctionInfo &info,
    const std::vector<parser::NodePtr> &args) {
  if (info.isNative || info.varargType || !hasProtocolFormal(info) ||
      !info.definition)
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

  auto defaultNodeForFormal =
      [&](std::size_t formalIndex) -> const parser::Node * {
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
      return nullptr;
    return defaultNode->get();
  };

  FunctionInfo specialized = info;
  specialized.isSpecialization = true;
  specialized.argTypes.assign(info.argTypes.begin(), info.argTypes.end());
  std::string key = info.symbolName;
  bool changed = false;
  std::function<std::optional<mlir::Type>(const parser::Node &,
                                          py::ProtocolType)>
      concreteTypeForProtocolActual;
  concreteTypeForProtocolActual =
      [&](const parser::Node &actual,
          py::ProtocolType protocol) -> std::optional<mlir::Type> {
    auto concreteKnownCallResultType =
        [&](const parser::Node &call) -> std::optional<mlir::Type> {
      if (call.kind != "Call")
        return std::nullopt;
      const parser::NodePtr *func = nodeField(call, "func");
      const std::vector<parser::NodePtr> *callArgs =
          nodeListField(call, "args");
      if (!func || !*func || (*func)->kind != "Name")
        return std::nullopt;
      const std::string *name = stringField(**func, "id");
      if (!name)
        return std::nullopt;
      std::optional<FunctionInfo> callee;
      auto localFunction = localFunctions.find(*name);
      if (localFunction != localFunctions.end())
        callee = localFunction->second;
      else {
        auto function = functions.find(*name);
        if (function != functions.end())
          callee = function->second;
      }
      if (!callee)
        return std::nullopt;
      if (callee->isAsync)
        return coroutineType(callee->resultType);

      auto resultProtocol =
          mlir::dyn_cast<py::ProtocolType>(callee->resultType);
      if (!resultProtocol || !callee->returnedValueArgIndex ||
          *callee->returnedValueArgIndex >= callee->argTypes.size() ||
          !callArgs)
        return std::nullopt;

      std::optional<std::vector<const parser::Node *>> expandedArgs =
          expandStaticCallArgs(call, *callArgs, callee->positionalCount,
                               callee->name);
      if (!expandedArgs)
        return std::nullopt;
      std::optional<std::vector<StaticKeywordArg>> expandedKeywords =
          expandStaticCallKeywords(call, callee->name);
      if (!expandedKeywords)
        return std::nullopt;

      std::vector<const parser::Node *> callActuals(callee->argTypes.size(),
                                                    nullptr);
      for (std::size_t index = 0;
           index < expandedArgs->size() && index < callActuals.size(); ++index)
        callActuals[index] = (*expandedArgs)[index];
      for (const StaticKeywordArg &keyword : *expandedKeywords) {
        auto nameIt = std::find(callee->argNames.begin(),
                                callee->argNames.end(), keyword.name);
        if (nameIt == callee->argNames.end())
          return std::nullopt;
        std::size_t formalIndex = static_cast<std::size_t>(
            std::distance(callee->argNames.begin(), nameIt));
        if (formalIndex >= callActuals.size() || callActuals[formalIndex])
          return std::nullopt;
        callActuals[formalIndex] = keyword.value;
      }

      const parser::Node *returnedActual =
          callActuals[*callee->returnedValueArgIndex];
      if (!returnedActual)
        return std::nullopt;
      if (std::optional<mlir::Type> concrete =
              concreteTypeForProtocolActual(*returnedActual, resultProtocol))
        return concrete;
      std::optional<mlir::Type> inferred = inferExpressionType(*returnedActual);
      if (inferred && !mlir::isa<py::ProtocolType>(*inferred) &&
          typeAssignable(resultProtocol, *inferred))
        return inferred;
      return std::nullopt;
    };

    if (actual.kind == "Name") {
      const std::string *name = stringField(actual, "id");
      auto found = name ? symbols.find(*name) : symbols.end();
      if (found != symbols.end()) {
        if (found->second.protocolConcreteType)
          return found->second.protocolConcreteType;
        if (found->second.type &&
            !mlir::isa<py::ProtocolType>(found->second.type))
          return found->second.type;
      }
    }
    if (std::optional<mlir::Type> concrete =
            concreteKnownCallResultType(actual))
      return concrete;
    if (std::optional<mlir::Type> inferred = inferExpressionType(actual))
      return inferred;

    auto isNoArgCallNamed = [](const parser::Node &node, llvm::StringRef name) {
      if (node.kind != "Call")
        return false;
      const parser::NodePtr *func = nodeField(node, "func");
      const std::vector<parser::NodePtr> *args = nodeListField(node, "args");
      const std::vector<parser::NodePtr> *keywords =
          nodeListField(node, "keywords");
      if (!func || !*func || (*func)->kind != "Name" || !args ||
          !args->empty() || (keywords && !keywords->empty()))
        return false;
      const std::string *id = stringField(**func, "id");
      return id && *id == name;
    };

    if (actual.kind == "List" || isNoArgCallNamed(actual, "list")) {
      if (std::optional<mlir::Type> element = protocolIterableElement(protocol))
        return listType(*element);
    }

    if (actual.kind == "Dict" || isNoArgCallNamed(actual, "dict")) {
      std::optional<std::pair<mlir::Type, mlir::Type>> dictTypes =
          protocolMappingKeyValueTypes(protocol);
      if (dictTypes)
        return dictType(dictTypes->first, dictTypes->second);
    }

    return std::nullopt;
  };
  for (std::size_t formalIndex = 0; formalIndex < info.argTypes.size();
       ++formalIndex) {
    auto protocol =
        mlir::dyn_cast<py::ProtocolType>(info.argTypes[formalIndex]);
    if (!protocol)
      continue;

    const parser::Node *actual = actuals[formalIndex];
    const bool usesDefault = !actual;
    if (!actual)
      actual = defaultNodeForFormal(formalIndex);
    if (!actual) {
      error(expr, "protocol argument '" + info.argNames[formalIndex] +
                      "' for function '" + info.name +
                      "' requires an explicit statically typed actual or a "
                      "statically typed default");
      return std::nullopt;
    }
    std::optional<mlir::Type> actualType =
        concreteTypeForProtocolActual(*actual, protocol);
    if (!actualType) {
      error(*actual, std::string("cannot infer concrete type for ") +
                         (usesDefault ? "defaulted " : "") +
                         "protocol argument '" + info.argNames[formalIndex] +
                         "' in function '" + info.name + "'");
      return std::nullopt;
    }
    if (mlir::isa<py::ProtocolType>(*actualType)) {
      error(*actual, "protocol argument '" + info.argNames[formalIndex] +
                         "' for function '" + info.name +
                         "' requires a concrete " +
                         (usesDefault ? std::string("default type")
                                      : std::string("actual type")) +
                         ", got " + typeString(*actualType));
      return std::nullopt;
    }
    if (!typeAssignable(protocol, *actualType)) {
      error(*actual, "argument '" + info.argNames[formalIndex] +
                         "' for function '" + info.name +
                         "' does not satisfy protocol " + typeString(protocol) +
                         ", got " + typeString(*actualType));
      return std::nullopt;
    }
    specialized.argTypes[formalIndex] = *actualType;
    key += "|P:" + std::to_string(formalIndex) + "=" +
           specializationTypeKey(*actualType);
    changed = true;
  }

  if (!changed)
    return std::nullopt;

  if (specialized.returnedValueArgIndex &&
      *specialized.returnedValueArgIndex < specialized.argTypes.size() &&
      mlir::isa<py::ProtocolType>(specialized.resultType)) {
    mlir::Type returnedArgType =
        specialized.argTypes[*specialized.returnedValueArgIndex];
    if (!mlir::isa<py::ProtocolType>(returnedArgType) &&
        typeAssignable(specialized.resultType, returnedArgType))
      specialized.resultType = returnedArgType;
  }
  if (mlir::isa<py::ProtocolType>(specialized.resultType)) {
    std::map<std::string, mlir::Type> knownNames;
    for (std::size_t index = 0; index < specialized.argTypes.size() &&
                                index < specialized.argNames.size();
         ++index)
      knownNames.emplace(specialized.argNames[index],
                         specialized.argTypes[index]);
    if (std::optional<mlir::Type> concreteResult = directReturnedConcreteType(
            specialized.resultType, *specialized.definition, knownNames))
      specialized.resultType = *concreteResult;
  }

  auto existing = functionSpecializations.find(key);
  if (existing != functionSpecializations.end())
    return existing->second.info;

  specialized.symbolName = "__lython_spec_" +
                           std::to_string(++functionSpecializationCounter) +
                           "_" + info.symbolName;
  refreshFunctionTypes(specialized);
  FunctionSpecialization stored{specialized, {}};
  functionSpecializations.emplace(key, std::move(stored));
  return specialized;
}

std::optional<FunctionInfo> Builder::Impl::specializeClassFunctionCall(
    const parser::Node &expr, const FunctionInfo &info,
    const std::vector<parser::NodePtr> &args) {
  if (info.isSpecialization || info.isNative || info.varargType ||
      !hasClassFormal(info) || !info.definition)
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
    if (!(*expandedArgs)[index]) {
      ok = false;
      continue;
    }
    actuals[index] = (*expandedArgs)[index];
  }
  for (const StaticKeywordArg &keyword : *expandedKeywords) {
    if (!keyword.anchor || !keyword.value) {
      ok = false;
      continue;
    }
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

  auto defaultNodeForFormal =
      [&](std::size_t formalIndex) -> const parser::Node * {
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
      return nullptr;
    return defaultNode->get();
  };

  FunctionInfo specialized = info;
  specialized.isSpecialization = true;
  specialized.argTypes.assign(info.argTypes.begin(), info.argTypes.end());
  std::string key = info.symbolName;
  bool changed = false;

  auto inferClassActualType =
      [&](const parser::Node &actual) -> std::optional<mlir::Type> {
    if (actual.kind != "Call")
      return inferExpressionType(actual);
    const parser::NodePtr *func = nodeField(actual, "func");
    const std::vector<parser::NodePtr> *callArgs =
        nodeListField(actual, "args");
    const std::vector<parser::NodePtr> *keywords =
        nodeListField(actual, "keywords");
    if (!func || !*func || (*func)->kind != "Name" || !callArgs ||
        (keywords && !keywords->empty()))
      return inferExpressionType(actual);
    const std::string *className = stringField(**func, "id");
    auto classFound = className ? classes.find(*className) : classes.end();
    if (!className || classFound == classes.end())
      return inferExpressionType(actual);
    auto initMethod = classFound->second.methods.find("__init__");
    auto initInfo = initMethod == classFound->second.methods.end()
                        ? functions.end()
                        : functions.find(initMethod->second);
    if (initInfo == functions.end())
      return classType(*className);

    llvm::SmallVector<mlir::Type> actualTypes;
    actualTypes.reserve(callArgs->size());
    for (const parser::NodePtr &arg : *callArgs) {
      if (!arg)
        return inferExpressionType(actual);
      std::optional<mlir::Type> argType = inferExpressionType(*arg);
      if (!argType)
        return inferExpressionType(actual);
      actualTypes.push_back(*argType);
    }
    if (std::optional<std::string> specializedClass =
            specializeClassForConstructorFieldTypes(
                actual, *className, initInfo->second, actualTypes))
      return classType(*specializedClass);
    return classType(*className);
  };

  for (std::size_t formalIndex = 0; formalIndex < info.argTypes.size();
       ++formalIndex) {
    auto formalClass =
        mlir::dyn_cast<py::ClassType>(info.argTypes[formalIndex]);
    if (!formalClass)
      continue;

    const parser::Node *actual = actuals[formalIndex];
    if (!actual)
      actual = defaultNodeForFormal(formalIndex);
    if (!actual)
      continue;

    std::optional<mlir::Type> actualType = inferClassActualType(*actual);
    auto actualClass = actualType ? mlir::dyn_cast<py::ClassType>(*actualType)
                                  : py::ClassType();
    if (!actualClass)
      continue;
    if (!typeAssignable(formalClass, actualClass))
      continue;

    specialized.argTypes[formalIndex] = actualClass;
    key += "|C:" + std::to_string(formalIndex) + "=" +
           specializationTypeKey(actualClass);
    changed = true;
  }

  if (!changed)
    return std::nullopt;

  if (specialized.returnedValueArgIndex &&
      *specialized.returnedValueArgIndex < specialized.argTypes.size()) {
    mlir::Type returnedArgType =
        specialized.argTypes[*specialized.returnedValueArgIndex];
    if (typeAssignable(specialized.resultType, returnedArgType))
      specialized.resultType = returnedArgType;
  }

  auto existing = functionSpecializations.find(key);
  if (existing != functionSpecializations.end())
    return existing->second.info;

  specialized.symbolName = "__lython_spec_" +
                           std::to_string(++functionSpecializationCounter) +
                           "_" + info.symbolName;
  refreshFunctionTypes(specialized);
  FunctionSpecialization stored{specialized, {}};
  functionSpecializations.emplace(key, std::move(stored));
  return specialized;
}

bool Builder::Impl::bindTypeVariablesFromAnnotation(
    const parser::NodePtr &annotation, mlir::Type actual,
    const std::map<std::string, TypeAliasParameterKind> &parameterKinds,
    std::map<std::string, mlir::Type> &substitutions) {
  if (!annotation || !actual)
    return true;

  auto bindName = [&](llvm::StringRef name, mlir::Type type) {
    if (!parameterKinds.count(name.str()))
      return true;
    auto existing = substitutions.find(name.str());
    if (existing == substitutions.end()) {
      substitutions[name.str()] = type;
      return true;
    }
    return existing->second == type;
  };

  auto bindParamSpecName = [&](const parser::NodePtr &pack,
                               py::CallableType signature) {
    if (!pack || pack->kind != "Name")
      return false;
    const std::string *name = stringField(*pack, "id");
    if (!name)
      return false;
    auto found = parameterKinds.find(*name);
    if (found == parameterKinds.end() ||
        found->second != TypeAliasParameterKind::ParamSpec)
      return false;
    return bindName(*name, signature);
  };

  auto isEllipsisConstant = [](const parser::NodePtr &expr) {
    if (!expr || expr->kind != "Constant")
      return false;
    const parser::FieldValue *value = valueField(*expr, "value");
    return value && std::holds_alternative<parser::Ellipsis>(*value);
  };

  auto typingAnnotationName =
      [&](const parser::Node &expr) -> std::optional<std::string> {
    if (expr.kind == "Name") {
      const std::string *id = stringField(expr, "id");
      if (!id)
        return std::nullopt;
      auto alias = staticAnnotationAliases.find(*id);
      return alias == staticAnnotationAliases.end()
                 ? std::optional<std::string>(*id)
                 : std::optional<std::string>(alias->second);
    }
    if (expr.kind != "Attribute")
      return std::nullopt;
    const parser::NodePtr *value = nodeField(expr, "value");
    const std::string *attr = stringField(expr, "attr");
    if (!value || !*value || (*value)->kind != "Name" || !attr)
      return std::nullopt;
    const std::string *module = stringField(**value, "id");
    if (!module)
      return std::nullopt;
    auto found = staticModules.find(*module);
    if (found == staticModules.end() ||
        (found->second != "typing" && found->second != "collections.abc"))
      return std::nullopt;
    return *attr;
  };

  auto annotationArguments = [](const parser::NodePtr &slice)
      -> std::optional<llvm::SmallVector<parser::NodePtr>> {
    if (!slice)
      return std::nullopt;
    llvm::SmallVector<parser::NodePtr> args;
    if (slice->kind == "Tuple") {
      const std::vector<parser::NodePtr> *items = nodeListField(*slice, "elts");
      if (!items)
        return std::nullopt;
      args.append(items->begin(), items->end());
    } else {
      args.push_back(slice);
    }
    return args;
  };

  if (annotation->kind == "Name") {
    const std::string *name = stringField(*annotation, "id");
    return !name || bindName(*name, actual);
  }

  if (annotation->kind == "BinOp") {
    std::optional<std::string> op = symbolField(*annotation, "op");
    if (!op || *op != "|")
      return true;
    const parser::NodePtr *lhs = nodeField(*annotation, "left");
    const parser::NodePtr *rhs = nodeField(*annotation, "right");
    auto unionType = mlir::dyn_cast<py::UnionType>(actual);
    if (!unionType || !lhs || !*lhs || !rhs || !*rhs)
      return true;
    return bindTypeVariablesFromAnnotation(*lhs, actual, parameterKinds,
                                           substitutions) &&
           bindTypeVariablesFromAnnotation(*rhs, actual, parameterKinds,
                                           substitutions);
  }

  if (annotation->kind != "Subscript")
    return true;
  const parser::NodePtr *value = nodeField(*annotation, "value");
  const parser::NodePtr *slice = nodeField(*annotation, "slice");
  if (!value || !*value || !slice || !*slice)
    return true;

  auto bindSingle = [&](mlir::Type type) {
    return bindTypeVariablesFromAnnotation(*slice, type, parameterKinds,
                                           substitutions);
  };

  if (isTypingName(**value, "Optional")) {
    if (auto unionType = mlir::dyn_cast<py::UnionType>(actual)) {
      for (mlir::Type member : unionType.getMemberTypes()) {
        if (!mlir::isa<py::NoneType>(member))
          return bindSingle(member);
      }
    }
    return true;
  }
  if (isTypingName(**value, "list") || isTypingName(**value, "List")) {
    if (mlir::Type element = listElementType(actual))
      return bindSingle(element);
    return true;
  }
  if (isTypingName(**value, "dict") || isTypingName(**value, "Dict")) {
    auto dict = mlir::dyn_cast<py::DictType>(actual);
    if (!dict || (*slice)->kind != "Tuple")
      return true;
    const std::vector<parser::NodePtr> *items = nodeListField(**slice, "elts");
    if (!items || items->size() != 2)
      return true;
    return bindTypeVariablesFromAnnotation(items->front(), dict.getKeyType(),
                                           parameterKinds, substitutions) &&
           bindTypeVariablesFromAnnotation((*items)[1], dict.getValueType(),
                                           parameterKinds, substitutions);
  }
  if (isTypingName(**value, "tuple") || isTypingName(**value, "Tuple")) {
    auto tuple = mlir::dyn_cast<py::TupleType>(actual);
    if (!tuple)
      return true;
    llvm::ArrayRef<mlir::Type> elements = tuple.getElementTypes();
    if ((*slice)->kind != "Tuple") {
      if (elements.size() == 1)
        return bindSingle(elements.front());
      return true;
    }
    const std::vector<parser::NodePtr> *items = nodeListField(**slice, "elts");
    if (!items || items->size() != elements.size())
      return true;
    for (auto indexed : llvm::enumerate(*items)) {
      if (!bindTypeVariablesFromAnnotation(indexed.value(),
                                           elements[indexed.index()],
                                           parameterKinds, substitutions))
        return false;
    }
    return true;
  }

  if (isTypingName(**value, "Callable")) {
    py::CallableType signature = py::getCallableContract(actual);
    if (!signature || (*slice)->kind != "Tuple")
      return true;
    const std::vector<parser::NodePtr> *items = nodeListField(**slice, "elts");
    if (!items || items->size() != 2 || !items->front() || !(*items)[1])
      return true;

    py::CallableType parameterPack =
        callableParameterPackFromSignature(signature);

    std::function<bool(const parser::NodePtr &, py::CallableType)>
        bindCallablePack = [&](const parser::NodePtr &pack,
                               py::CallableType actualPack) -> bool {
      if (!pack)
        return actualPack.getPositionalTypes().empty() &&
               actualPack.getKwOnlyTypes().empty() && !actualPack.hasVararg() &&
               !actualPack.hasKwarg();
      if (bindParamSpecName(pack, actualPack))
        return true;
      if (pack->kind == "Starred") {
        const parser::NodePtr *value = nodeField(*pack, "value");
        return value && *value && bindCallablePack(*value, actualPack);
      }
      if (pack->kind == "List" || pack->kind == "Tuple") {
        if (actualPack.hasVararg() || actualPack.hasKwarg() ||
            !actualPack.getKwOnlyTypes().empty())
          return false;
        const std::vector<parser::NodePtr> *packItems =
            nodeListField(*pack, "elts");
        if (!packItems ||
            packItems->size() != actualPack.getPositionalTypes().size())
          return false;
        for (auto indexed : llvm::enumerate(*packItems)) {
          if (!bindTypeVariablesFromAnnotation(
                  indexed.value(),
                  actualPack.getPositionalTypes()[indexed.index()],
                  parameterKinds, substitutions))
            return false;
        }
        return true;
      }
      if (pack->kind == "Subscript") {
        const parser::NodePtr *packValue = nodeField(*pack, "value");
        const parser::NodePtr *packSlice = nodeField(*pack, "slice");
        if (packValue && *packValue && packSlice && *packSlice &&
            isTypingName(**packValue, "Concatenate")) {
          llvm::SmallVector<parser::NodePtr> concatItems;
          if ((*packSlice)->kind == "Tuple") {
            const std::vector<parser::NodePtr> *elts =
                nodeListField(**packSlice, "elts");
            if (!elts)
              return false;
            concatItems.append(elts->begin(), elts->end());
          } else {
            concatItems.push_back(*packSlice);
          }
          if (concatItems.size() < 2)
            return false;
          std::size_t prefixCount = concatItems.size() - 1;
          if (actualPack.getPositionalTypes().size() < prefixCount)
            return false;
          for (std::size_t index = 0; index < prefixCount; ++index) {
            if (!bindTypeVariablesFromAnnotation(
                    concatItems[index], actualPack.getPositionalTypes()[index],
                    parameterKinds, substitutions))
              return false;
          }
          std::optional<py::CallableType> suffixPack =
              dropCallableParameterPrefix(actualPack, prefixCount);
          return suffixPack &&
                 bindCallablePack(concatItems.back(), *suffixPack);
        }
      }
      if (!actualPack.hasVararg() && !actualPack.hasKwarg() &&
          actualPack.getKwOnlyTypes().empty() &&
          actualPack.getPositionalTypes().size() == 1)
        return bindTypeVariablesFromAnnotation(
            pack, actualPack.getPositionalTypes().front(), parameterKinds,
            substitutions);
      return false;
    };

    if (!isEllipsisConstant(items->front()) &&
        !bindCallablePack(items->front(), parameterPack))
      return false;

    mlir::ArrayRef<mlir::Type> results = signature.getResultTypes();
    if (results.size() != 1)
      return false;
    return bindTypeVariablesFromAnnotation((*items)[1], results.front(),
                                           parameterKinds, substitutions);
  }

  if (isTypingName(**value, "type") || isTypingName(**value, "Type")) {
    auto meta = mlir::dyn_cast<py::TypeType>(actual);
    if (!meta)
      return true;
    return bindSingle(meta.getInstanceType());
  }

  if (std::optional<std::string> protocolName = typingAnnotationName(**value)) {
    const protocols::Table &table = protocols::Table::get(context);
    const protocols::ProtocolInfo *info = table.lookup(*protocolName);
    if (info && info->isProtocol && *protocolName != "Protocol") {
      std::optional<llvm::SmallVector<parser::NodePtr>> formalArgs =
          annotationArguments(*slice);
      if (!formalArgs || formalArgs->size() > info->params.size())
        return true;
      std::optional<std::vector<mlir::Type>> actualArgs =
          table.protocolArgumentsFor(actual, *protocolName);
      if (!actualArgs)
        return true;
      if (formalArgs->size() > actualArgs->size())
        return true;
      for (auto indexed : llvm::enumerate(*formalArgs)) {
        if (!bindTypeVariablesFromAnnotation(indexed.value(),
                                             (*actualArgs)[indexed.index()],
                                             parameterKinds, substitutions))
          return false;
      }
      return true;
    }
  }

  if ((*value)->kind == "Name") {
    const std::string *className = stringField(**value, "id");
    std::optional<std::string> actualClass = classNameFromType(actual);
    auto actualInfo = actualClass ? classes.find(*actualClass) : classes.end();
    if (className && actualInfo != classes.end() &&
        actualInfo->second.templateName == *className) {
      auto templ = classes.find(*className);
      if (templ == classes.end())
        return true;
      llvm::SmallVector<parser::NodePtr> args;
      if ((*slice)->kind == "Tuple") {
        const std::vector<parser::NodePtr> *items =
            nodeListField(**slice, "elts");
        if (items)
          args.append(items->begin(), items->end());
      } else {
        args.push_back(*slice);
      }
      if (args.size() != templ->second.typeParameters.size())
        return true;
      for (auto indexed : llvm::enumerate(args)) {
        const std::string &paramName =
            templ->second.typeParameters[indexed.index()].name;
        auto concrete = actualInfo->second.typeSubstitutions.find(paramName);
        if (concrete == actualInfo->second.typeSubstitutions.end())
          continue;
        if (!bindTypeVariablesFromAnnotation(indexed.value(), concrete->second,
                                             parameterKinds, substitutions))
          return false;
      }
    }
  }
  return true;
}

std::optional<FunctionInfo> Builder::Impl::specializeGenericFunctionCall(
    const parser::Node &expr, const FunctionInfo &info,
    const std::vector<parser::NodePtr> &args,
    const std::map<std::string, mlir::Type> *explicitTypeArguments) {
  if (info.typeParameters.empty() || info.isSpecialization || !info.definition)
    return std::nullopt;

  std::optional<std::vector<const parser::Node *>> expandedArgs =
      expandStaticCallArgs(
          expr, args,
          info.varargName ? std::nullopt
                          : std::optional<std::size_t>(info.positionalCount),
          info.name);
  if (!expandedArgs)
    return std::nullopt;
  std::optional<std::vector<StaticKeywordArg>> expandedKeywords =
      expandStaticCallKeywords(expr, info.name);
  if (!expandedKeywords)
    return std::nullopt;

  std::vector<const parser::Node *> actuals(info.argNames.size(), nullptr);
  std::size_t forwardedVarargCount = 0;
  if (info.varargName && expandedArgs->size() > info.positionalCount)
    forwardedVarargCount = expandedArgs->size() - info.positionalCount;
  std::vector<std::string> forwardedKwargNames;
  bool ok = true;
  for (std::size_t index = 0; index < expandedArgs->size(); ++index) {
    if (index >= actuals.size()) {
      if (!info.varargName)
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
      if (!info.kwargName) {
        error(*keyword.anchor, "function '" + info.name +
                                   "' has no parameter '" + keyword.name + "'");
        ok = false;
      }
      forwardedKwargNames.push_back(keyword.name);
      continue;
    }
    std::size_t formalIndex =
        static_cast<std::size_t>(std::distance(info.argNames.begin(), nameIt));
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

  const parser::NodePtr *argsNode = nodeField(*info.definition, "args");
  if (!argsNode || !*argsNode)
    return std::nullopt;
  std::vector<const parser::Node *> formals;
  if (const std::vector<parser::NodePtr> *posonlyargs =
          nodeListField(**argsNode, "posonlyargs")) {
    for (const parser::NodePtr &arg : *posonlyargs)
      if (arg)
        formals.push_back(arg.get());
  }
  if (const std::vector<parser::NodePtr> *regularArgs =
          nodeListField(**argsNode, "args")) {
    for (const parser::NodePtr &arg : *regularArgs)
      if (arg)
        formals.push_back(arg.get());
  }
  if (formals.size() < info.positionalCount)
    return std::nullopt;

  std::map<std::string, TypeAliasParameterKind> parameterKinds;
  for (const TypeAliasParameter &parameter : info.typeParameters)
    parameterKinds[parameter.name] = parameter.kind;

  std::map<std::string, mlir::Type> substitutions;
  if (explicitTypeArguments)
    substitutions = *explicitTypeArguments;
  for (std::size_t index = 0; index < info.positionalCount; ++index) {
    const parser::Node *actual =
        index < actuals.size() ? actuals[index] : nullptr;
    if (!actual)
      continue;
    std::optional<FunctionInfo> callableInfo = resolveCallableInfo(*actual);
    std::optional<mlir::Type> actualType =
        callableInfo ? std::optional<mlir::Type>(callableInfo->functionType)
                     : inferExpressionType(*actual);
    if (!actualType) {
      error(*actual, "cannot infer type argument for generic function '" +
                         info.name + "'");
      return std::nullopt;
    }
    const parser::NodePtr *annotation =
        nodeField(*formals[index], "annotation");
    if (!bindTypeVariablesFromAnnotation(
            annotation ? *annotation : parser::NodePtr{}, *actualType,
            parameterKinds, substitutions)) {
      error(*actual, "conflicting type arguments for generic function '" +
                         info.name + "'");
      return std::nullopt;
    }
  }

  for (const TypeAliasParameter &parameter : info.typeParameters) {
    if (substitutions.count(parameter.name))
      continue;
    if (parameter.defaultValue) {
      std::optional<mlir::Type> defaultType =
          typeFromAnnotation(parameter.defaultValue, substitutions);
      if (defaultType) {
        substitutions[parameter.name] = *defaultType;
        continue;
      }
    }
    error(expr, "could not infer type parameter '" + parameter.name +
                    "' for generic function '" + info.name + "'");
    return std::nullopt;
  }
  for (const TypeAliasParameter &parameter : info.typeParameters) {
    mlir::Type argument = substitutions[parameter.name];
    if (parameter.bound && !typeAssignable(parameter.bound, argument)) {
      error(expr, "type argument " + typeString(argument) +
                      " does not satisfy bound " + typeString(parameter.bound) +
                      " for '" + parameter.name + "'");
      return std::nullopt;
    }
  }

  std::string key = info.symbolName;
  for (const TypeAliasParameter &parameter : info.typeParameters)
    key += "|T:" + parameter.name + "=" +
           specializationTypeKey(substitutions[parameter.name]);
  if (info.varargName)
    key += "|VA:" + std::to_string(forwardedVarargCount);
  if (info.kwargName) {
    for (llvm::StringRef name : forwardedKwargNames)
      key += "|KW:" + name.str();
  }
  std::map<std::string, FunctionInfo> aliases;
  for (std::size_t formalIndex = 0; formalIndex < info.positionalCount;
       ++formalIndex) {
    if (formalIndex >= formals.size() || formalIndex >= info.argNames.size())
      continue;
    const parser::NodePtr *annotation =
        nodeField(*formals[formalIndex], "annotation");
    std::optional<mlir::Type> expectedType = typeFromAnnotation(
        annotation ? *annotation : parser::NodePtr{}, substitutions);
    if (!expectedType || !py::isCallableType(*expectedType))
      continue;

    const parser::Node *actual =
        formalIndex < actuals.size() ? actuals[formalIndex] : nullptr;
    std::optional<FunctionInfo> actualInfo;
    if (actual)
      actualInfo = resolveCallableInfo(*actual);
    if (!actualInfo && hasDefaultForFormal(info, formalIndex)) {
      const parser::NodePtr *defaultNode = nullptr;
      const std::size_t defaultCount = info.defaultValues.size();
      const std::size_t defaultStart = info.positionalCount >= defaultCount
                                           ? info.positionalCount - defaultCount
                                           : info.positionalCount;
      if (formalIndex >= defaultStart && formalIndex < info.positionalCount)
        defaultNode = &info.defaultValues[formalIndex - defaultStart];
      if (defaultNode && *defaultNode)
        actualInfo = resolveCallableInfo(**defaultNode);
    }
    if (!actualInfo) {
      error(expr, "generic function '" + info.name + "' callable argument '" +
                      info.argNames[formalIndex] +
                      "' requires a statically known callable");
      return std::nullopt;
    }
    if (!typeAssignable(*expectedType, actualInfo->functionType)) {
      const parser::Node &anchor = actual ? *actual : expr;
      error(anchor, "callable argument '" + info.argNames[formalIndex] +
                        "' for generic function '" + info.name + "' must be " +
                        typeString(*expectedType) + ", got " +
                        typeString(actualInfo->functionType));
      return std::nullopt;
    }
    aliases.emplace(info.argNames[formalIndex], *actualInfo);
    key += "|C:" + std::to_string(formalIndex) + ":" + actualInfo->symbolName;
  }
  auto existing = functionSpecializations.find(key);
  if (existing != functionSpecializations.end())
    return existing->second.info;

  std::optional<FunctionInfo> parsed =
      parseFunctionInfo(*info.definition, &substitutions);
  if (!parsed)
    return std::nullopt;

  auto restrictArgsPack =
      [&](py::CallableType pack,
          std::size_t count) -> std::optional<py::CallableType> {
    llvm::SmallVector<mlir::Type> positional;
    llvm::SmallVector<mlir::StringAttr> positionalNames;
    llvm::SmallVector<mlir::BoolAttr> positionalDefaults;
    positional.reserve(count);
    auto types = pack.getPositionalTypes();
    auto names = pack.getPositionalNames();
    auto defaults = pack.getPositionalDefaults();
    mlir::Type repeatedType;
    if (pack.hasVararg()) {
      auto tuple = mlir::dyn_cast<py::TupleType>(pack.getVarargType());
      if (tuple && tuple.getElementTypes().size() == 1)
        repeatedType = tuple.getElementTypes().front();
    }
    for (std::size_t index = 0; index < count; ++index) {
      if (index < types.size()) {
        positional.push_back(types[index]);
        if (index < names.size())
          positionalNames.push_back(names[index]);
        if (index < defaults.size())
          positionalDefaults.push_back(defaults[index]);
        continue;
      }
      if (!repeatedType) {
        error(expr, "ParamSpec *args forwarding for function '" + info.name +
                        "' cannot materialize argument " +
                        std::to_string(index));
        return std::nullopt;
      }
      positional.push_back(repeatedType);
    }
    return py::CallableType::get(&context, positional, {}, {}, {}, {},
                                 positionalNames, {}, positionalDefaults, {},
                                 {}, {},
                                 static_cast<unsigned>(std::min<std::size_t>(
                                     pack.getPositionalOnlyCount(), count)));
  };

  auto restrictKwargsPack = [&](py::CallableType pack,
                                llvm::ArrayRef<std::string> namesToForward)
      -> std::optional<py::CallableType> {
    llvm::SmallVector<mlir::Type> positional;
    llvm::SmallVector<mlir::Type> kwonly;
    llvm::SmallVector<mlir::StringAttr> positionalNames;
    llvm::SmallVector<mlir::StringAttr> kwonlyNames;
    llvm::SmallVector<mlir::BoolAttr> positionalDefaults;
    llvm::SmallVector<mlir::BoolAttr> kwonlyDefaults;

    auto fullPosTypes = pack.getPositionalTypes();
    auto fullPosNames = pack.getPositionalNames();
    auto fullPosDefaults = pack.getPositionalDefaults();
    unsigned fullPosOnlyCount = pack.getPositionalOnlyCount();
    auto fullKwTypes = pack.getKwOnlyTypes();
    auto fullKwNames = pack.getKwOnlyNames();
    auto fullKwDefaults = pack.getKwOnlyDefaults();
    auto appendKnownPositional = [&](std::size_t index) {
      positional.push_back(fullPosTypes[index]);
      positionalNames.push_back(fullPosNames[index]);
      if (index < fullPosDefaults.size())
        positionalDefaults.push_back(fullPosDefaults[index]);
    };
    auto appendKnownKwOnly = [&](std::size_t index) {
      kwonly.push_back(fullKwTypes[index]);
      kwonlyNames.push_back(fullKwNames[index]);
      if (index < fullKwDefaults.size())
        kwonlyDefaults.push_back(fullKwDefaults[index]);
    };

    mlir::Type openKwargValueType;
    if (pack.hasKwarg()) {
      auto dict = mlir::dyn_cast<py::DictType>(pack.getKwargType());
      if (dict && dict.getKeyType() == strType())
        openKwargValueType = dict.getValueType();
    }

    for (llvm::StringRef name : namesToForward) {
      bool matched = false;
      if (!fullPosTypes.empty()) {
        if (fullPosNames.size() != fullPosTypes.size()) {
          error(expr, "ParamSpec **kwargs forwarding for function '" +
                          info.name +
                          "' requires names for positional parameters");
          return std::nullopt;
        }
        for (std::size_t index = fullPosOnlyCount; index < fullPosNames.size();
             ++index) {
          if (fullPosNames[index].getValue() != name)
            continue;
          appendKnownPositional(index);
          matched = true;
          break;
        }
      }
      if (matched)
        continue;
      if (fullKwNames.size() != fullKwTypes.size()) {
        error(expr, "ParamSpec **kwargs forwarding for function '" + info.name +
                        "' requires names for keyword-only parameters");
        return std::nullopt;
      }
      for (std::size_t index = 0; index < fullKwNames.size(); ++index) {
        if (fullKwNames[index].getValue() != name)
          continue;
        appendKnownKwOnly(index);
        matched = true;
        break;
      }
      if (matched)
        continue;
      if (!openKwargValueType) {
        error(expr, "ParamSpec **kwargs forwarding for function '" + info.name +
                        "' has no parameter '" + name.str() + "'");
        return std::nullopt;
      }
      kwonly.push_back(openKwargValueType);
      kwonlyNames.push_back(builder.getStringAttr(name));
    }

    return py::CallableType::get(&context, positional, kwonly, {}, {}, {},
                                 positionalNames, kwonlyNames,
                                 positionalDefaults, kwonlyDefaults, {}, {}, 0);
  };

  bool narrowedForwardingPacks = false;
  if (parsed->varargParameterPack) {
    std::optional<py::CallableType> narrowed =
        restrictArgsPack(parsed->varargParameterPack, forwardedVarargCount);
    if (!narrowed)
      return std::nullopt;
    parsed->varargParameterPack = *narrowed;
    parsed->varargType =
        py::TupleType::get(&context, narrowed->getPositionalTypes());
    narrowedForwardingPacks = true;
  }
  if (parsed->kwargParameterPack) {
    std::optional<py::CallableType> narrowed =
        restrictKwargsPack(parsed->kwargParameterPack, forwardedKwargNames);
    if (!narrowed)
      return std::nullopt;
    parsed->kwargParameterPack = *narrowed;
    llvm::SmallVector<mlir::Type> keywordValueTypes;
    keywordValueTypes.append(narrowed->getPositionalTypes().begin(),
                             narrowed->getPositionalTypes().end());
    keywordValueTypes.append(narrowed->getKwOnlyTypes().begin(),
                             narrowed->getKwOnlyTypes().end());
    mlir::Type fallbackValueType = py::ObjectType::get(&context);
    if (auto originalKwargs = mlir::dyn_cast<py::DictType>(parsed->kwargType))
      fallbackValueType = originalKwargs.getValueType();
    mlir::Type valueType =
        keywordValueTypes.empty()
            ? fallbackValueType
            : py::UnionType::getNormalized(&context, keywordValueTypes);
    if (!valueType)
      valueType = py::ObjectType::get(&context);
    parsed->kwargType = py::DictType::get(&context, strType(), valueType);
    narrowedForwardingPacks = true;
  }
  if (narrowedForwardingPacks)
    refreshFunctionTypes(*parsed);

  parsed->isSpecialization = true;
  parsed->typeParameters.clear();
  parsed->typeSubstitutions = substitutions;
  parsed->symbolName = "__lython_spec_" +
                       std::to_string(++functionSpecializationCounter) + "_" +
                       info.symbolName;
  for (std::size_t formalIndex = 0; formalIndex < parsed->argTypes.size() &&
                                    formalIndex < parsed->argNames.size();
       ++formalIndex) {
    auto alias = aliases.find(parsed->argNames[formalIndex]);
    if (alias == aliases.end())
      continue;
    if (py::isCallableType(parsed->argTypes[formalIndex]) &&
        typeAssignable(parsed->argTypes[formalIndex],
                       alias->second.functionType))
      parsed->argTypes[formalIndex] = alias->second.functionType;
  }
  if (std::optional<std::size_t> returnedCallableArgIndex =
          directReturnedCallableArgIndex(*parsed, *info.definition)) {
    if (*returnedCallableArgIndex < parsed->argTypes.size() &&
        py::isCallableType(parsed->resultType) &&
        py::isCallableType(parsed->argTypes[*returnedCallableArgIndex]) &&
        typeAssignable(parsed->resultType,
                       parsed->argTypes[*returnedCallableArgIndex]))
      parsed->resultType = parsed->argTypes[*returnedCallableArgIndex];
  }
  refreshFunctionTypes(*parsed);
  if (!parsed->isNative && !parsed->isAsync &&
      py::isCallableType(parsed->resultType)) {
    parsed->returnedCallable =
        directReturnedNestedCallable(*parsed, *info.definition);
    if (parsed->returnedCallable) {
      parsed->returnedCallableSymbolName =
          parsed->returnedCallable->info.symbolName;
      callableFunctionsBySymbol[parsed->returnedCallable->info.symbolName] =
          parsed->returnedCallable->info;
    } else {
      parsed->returnedCallableArgIndex =
          directReturnedCallableArgIndex(*parsed, *info.definition);
      if (!parsed->returnedCallableArgIndex) {
        if (std::optional<FunctionInfo> nested =
                directReturnedNestedCallableMetadata(*parsed,
                                                     *info.definition)) {
          parsed->returnedCallableSymbolName = nested->symbolName;
          callableFunctionsBySymbol[nested->symbolName] = *nested;
        } else {
          parsed->returnedCallableSymbolName =
              directReturnedCallableSymbol(*info.definition);
        }
      }
    }
    callableFunctionsBySymbol[parsed->symbolName] = *parsed;
  }
  FunctionSpecialization stored{*parsed, std::move(aliases)};
  functionSpecializations.emplace(key, std::move(stored));
  return *parsed;
}

std::optional<FunctionInfo>
Builder::Impl::specializeParamSpecForwardingFunctionCall(
    const parser::Node &expr, const FunctionInfo &info,
    const std::vector<parser::NodePtr> &args) {
  if (info.isSpecialization || !info.definition ||
      (!info.varargParameterPack && !info.kwargParameterPack))
    return std::nullopt;

  std::optional<std::vector<const parser::Node *>> expandedArgs =
      expandStaticCallArgs(
          expr, args,
          info.varargName ? std::nullopt
                          : std::optional<std::size_t>(info.positionalCount),
          info.name);
  if (!expandedArgs)
    return std::nullopt;
  std::optional<std::vector<StaticKeywordArg>> expandedKeywords =
      expandStaticCallKeywords(expr, info.name);
  if (!expandedKeywords)
    return std::nullopt;

  std::size_t forwardedVarargCount = 0;
  if (info.varargName && expandedArgs->size() > info.positionalCount)
    forwardedVarargCount = expandedArgs->size() - info.positionalCount;

  std::vector<std::string> forwardedKwargNames;
  bool ok = true;
  std::vector<const parser::Node *> actuals(info.argNames.size(), nullptr);
  for (std::size_t index = 0; index < expandedArgs->size(); ++index) {
    if (index < actuals.size())
      actuals[index] = (*expandedArgs)[index];
    else if (!info.varargName)
      ok = false;
  }

  std::vector<std::string> seenKeywords;
  for (const StaticKeywordArg &keyword : *expandedKeywords) {
    if (!keyword.anchor || (!keyword.value && !keyword.generatedValue)) {
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
      if (!info.kwargName) {
        error(*keyword.anchor, "function '" + info.name +
                                   "' has no parameter '" + keyword.name + "'");
        ok = false;
      } else {
        forwardedKwargNames.push_back(keyword.name);
      }
      continue;
    }
    std::size_t formalIndex =
        static_cast<std::size_t>(std::distance(info.argNames.begin(), nameIt));
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

  std::string key = info.symbolName;
  if (info.varargName)
    key += "|VA:" + std::to_string(forwardedVarargCount);
  if (info.kwargName)
    for (llvm::StringRef name : forwardedKwargNames)
      key += "|KW:" + name.str();

  auto existing = functionSpecializations.find(key);
  if (existing != functionSpecializations.end())
    return existing->second.info;

  auto restrictArgsPack =
      [&](py::CallableType pack,
          std::size_t count) -> std::optional<py::CallableType> {
    llvm::SmallVector<mlir::Type> positional;
    llvm::SmallVector<mlir::StringAttr> positionalNames;
    llvm::SmallVector<mlir::BoolAttr> positionalDefaults;
    auto types = pack.getPositionalTypes();
    auto names = pack.getPositionalNames();
    auto defaults = pack.getPositionalDefaults();
    mlir::Type repeatedType;
    if (pack.hasVararg()) {
      auto tuple = mlir::dyn_cast<py::TupleType>(pack.getVarargType());
      if (tuple && tuple.getElementTypes().size() == 1)
        repeatedType = tuple.getElementTypes().front();
    }
    for (std::size_t index = 0; index < count; ++index) {
      if (index < types.size()) {
        positional.push_back(types[index]);
        if (index < names.size())
          positionalNames.push_back(names[index]);
        if (index < defaults.size())
          positionalDefaults.push_back(defaults[index]);
      } else if (repeatedType) {
        positional.push_back(repeatedType);
      } else {
        error(expr, "ParamSpec *args forwarding for function '" + info.name +
                        "' cannot materialize argument " +
                        std::to_string(index));
        return std::nullopt;
      }
    }
    return py::CallableType::get(&context, positional, {}, {}, {}, {},
                                 positionalNames, {}, positionalDefaults, {},
                                 {}, {},
                                 static_cast<unsigned>(std::min<std::size_t>(
                                     pack.getPositionalOnlyCount(), count)));
  };

  auto restrictKwargsPack = [&](py::CallableType pack,
                                llvm::ArrayRef<std::string> namesToForward)
      -> std::optional<py::CallableType> {
    llvm::SmallVector<mlir::Type> positional;
    llvm::SmallVector<mlir::Type> kwonly;
    llvm::SmallVector<mlir::StringAttr> positionalNames;
    llvm::SmallVector<mlir::StringAttr> kwonlyNames;
    llvm::SmallVector<mlir::BoolAttr> positionalDefaults;
    llvm::SmallVector<mlir::BoolAttr> kwonlyDefaults;
    auto fullPosTypes = pack.getPositionalTypes();
    auto fullPosNames = pack.getPositionalNames();
    auto fullPosDefaults = pack.getPositionalDefaults();
    unsigned fullPosOnlyCount = pack.getPositionalOnlyCount();
    auto fullKwTypes = pack.getKwOnlyTypes();
    auto fullKwNames = pack.getKwOnlyNames();
    auto fullKwDefaults = pack.getKwOnlyDefaults();

    mlir::Type openKwargValueType;
    if (pack.hasKwarg()) {
      auto dict = mlir::dyn_cast<py::DictType>(pack.getKwargType());
      if (dict && dict.getKeyType() == strType())
        openKwargValueType = dict.getValueType();
    }

    for (llvm::StringRef name : namesToForward) {
      bool matched = false;
      if (!fullPosTypes.empty()) {
        if (fullPosNames.size() != fullPosTypes.size()) {
          error(expr, "ParamSpec **kwargs forwarding for function '" +
                          info.name +
                          "' requires names for positional parameters");
          return std::nullopt;
        }
        for (std::size_t index = fullPosOnlyCount; index < fullPosNames.size();
             ++index) {
          if (fullPosNames[index].getValue() != name)
            continue;
          positional.push_back(fullPosTypes[index]);
          positionalNames.push_back(fullPosNames[index]);
          if (index < fullPosDefaults.size())
            positionalDefaults.push_back(fullPosDefaults[index]);
          matched = true;
          break;
        }
      }
      if (matched)
        continue;
      if (fullKwNames.size() != fullKwTypes.size()) {
        error(expr, "ParamSpec **kwargs forwarding for function '" + info.name +
                        "' requires names for keyword-only parameters");
        return std::nullopt;
      }
      for (std::size_t index = 0; index < fullKwNames.size(); ++index) {
        if (fullKwNames[index].getValue() != name)
          continue;
        kwonly.push_back(fullKwTypes[index]);
        kwonlyNames.push_back(fullKwNames[index]);
        if (index < fullKwDefaults.size())
          kwonlyDefaults.push_back(fullKwDefaults[index]);
        matched = true;
        break;
      }
      if (matched)
        continue;
      if (!openKwargValueType) {
        error(expr, "ParamSpec **kwargs forwarding for function '" + info.name +
                        "' has no parameter '" + name.str() + "'");
        return std::nullopt;
      }
      kwonly.push_back(openKwargValueType);
      kwonlyNames.push_back(builder.getStringAttr(name));
    }

    return py::CallableType::get(&context, positional, kwonly, {}, {}, {},
                                 positionalNames, kwonlyNames,
                                 positionalDefaults, kwonlyDefaults, {}, {}, 0);
  };

  const std::map<std::string, mlir::Type> *typeVariables =
      info.typeSubstitutions.empty() ? nullptr : &info.typeSubstitutions;
  std::optional<FunctionInfo> parsed =
      parseFunctionInfo(*info.definition, typeVariables);
  if (!parsed)
    return std::nullopt;
  parsed->closureCaptures = info.closureCaptures;
  parsed->requiresCallableValue = info.requiresCallableValue;
  parsed->mayThrow = info.mayThrow;

  bool narrowed = false;
  if (parsed->varargParameterPack) {
    std::optional<py::CallableType> pack =
        restrictArgsPack(parsed->varargParameterPack, forwardedVarargCount);
    if (!pack)
      return std::nullopt;
    parsed->varargParameterPack = *pack;
    parsed->varargType =
        py::TupleType::get(&context, pack->getPositionalTypes());
    narrowed = true;
  }
  if (parsed->kwargParameterPack) {
    std::optional<py::CallableType> pack =
        restrictKwargsPack(parsed->kwargParameterPack, forwardedKwargNames);
    if (!pack)
      return std::nullopt;
    parsed->kwargParameterPack = *pack;
    llvm::SmallVector<mlir::Type> keywordValueTypes;
    keywordValueTypes.append(pack->getPositionalTypes().begin(),
                             pack->getPositionalTypes().end());
    keywordValueTypes.append(pack->getKwOnlyTypes().begin(),
                             pack->getKwOnlyTypes().end());
    mlir::Type fallbackValueType = py::ObjectType::get(&context);
    if (auto originalKwargs = mlir::dyn_cast<py::DictType>(parsed->kwargType))
      fallbackValueType = originalKwargs.getValueType();
    mlir::Type valueType =
        keywordValueTypes.empty()
            ? fallbackValueType
            : py::UnionType::getNormalized(&context, keywordValueTypes);
    if (!valueType)
      valueType = py::ObjectType::get(&context);
    parsed->kwargType = py::DictType::get(&context, strType(), valueType);
    narrowed = true;
  }
  if (narrowed)
    refreshFunctionTypes(*parsed);
  parsed->isSpecialization = true;
  parsed->typeParameters.clear();
  parsed->typeSubstitutions = info.typeSubstitutions;
  parsed->symbolName = "__lython_spec_" +
                       std::to_string(++functionSpecializationCounter) + "_" +
                       info.symbolName;
  refreshFunctionTypes(*parsed);

  FunctionSpecialization stored{*parsed, {}};
  functionSpecializations.emplace(key, std::move(stored));
  callableFunctionsBySymbol[parsed->symbolName] = *parsed;
  emittedFunctionSpecializations.insert(key);
  emitFunctionBody(*info.definition, *parsed);
  return *parsed;
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
    if (dict.kind == "Name") {
      const std::string *name = stringField(dict, "id");
      auto found = name ? symbols.find(*name) : symbols.end();
      if (found != symbols.end() && found->second.paramSpecKwargs) {
        py::CallableType pack = found->second.paramSpecKwargs;
        auto dictType = mlir::dyn_cast<py::DictType>(found->second.type);
        if (!dictType || dictType.getKeyType() != strType()) {
          error(dict, "ParamSpec **kwargs forwarding for " + targetName() +
                          " requires a dict[str, T] value");
          ok = false;
          continue;
        }
        if (pack.hasKwarg()) {
          error(dict, "ParamSpec **kwargs forwarding for " + targetName() +
                          " cannot statically enumerate open **kwargs yet");
          ok = false;
          continue;
        }

        auto appendGenerated = [&](llvm::StringRef keywordName) {
          mlir::Value key = builder.create<py::StrConstantOp>(
              loc(dict), strType(), keywordName);
          py::CallableType contract = unaryMethodContract(
              found->second.type, key.getType(), dictType.getValueType());
          mlir::Value value =
              builder.create<py::GetItemOp>(loc(dict), dictType.getValueType(),
                                            contract, found->second.value, key);
          expanded.push_back(
              StaticKeywordArg{keyword.get(), keywordName.str(), nullptr,
                               Value{value, dictType.getValueType()}});
        };

        llvm::ArrayRef<mlir::Type> positionalTypes = pack.getPositionalTypes();
        llvm::ArrayRef<mlir::StringAttr> positionalNames =
            pack.getPositionalNames();
        if (!positionalTypes.empty() &&
            positionalNames.size() != positionalTypes.size()) {
          error(dict, "ParamSpec **kwargs forwarding for " + targetName() +
                          " requires names for positional parameters");
          ok = false;
          continue;
        }
        for (mlir::StringAttr keywordName : positionalNames)
          appendGenerated(keywordName.getValue());

        llvm::ArrayRef<mlir::Type> kwonlyTypes = pack.getKwOnlyTypes();
        llvm::ArrayRef<mlir::StringAttr> kwonlyNames = pack.getKwOnlyNames();
        if (kwonlyNames.size() != kwonlyTypes.size()) {
          error(dict, "ParamSpec **kwargs forwarding for " + targetName() +
                          " requires names for keyword-only parameters");
          ok = false;
          continue;
        }
        for (mlir::StringAttr keywordName : kwonlyNames)
          appendGenerated(keywordName.getValue());
        continue;
      }
    }

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
    error(expr,
          "internal error: vararg calls must use canonical call lowering");
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
    value = coerceToExpectedType(node, std::move(value), expected);
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
      if (!keyword.anchor || (!keyword.value && !keyword.generatedValue)) {
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

      Value value;
      const parser::Node *valueAnchor =
          keyword.value ? keyword.value : keyword.anchor;
      if (keyword.generatedValue) {
        value = *keyword.generatedValue;
      } else {
        value = emitExpressionWithExpectedType(*keyword.value,
                                               info.argTypes[formalIndex]);
        if (!value.value) {
          ok = false;
          continue;
        }
      }
      storeActual(*valueAnchor, formalIndex, std::move(value));
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
  const std::vector<parser::NodePtr> *keywords =
      nodeListField(expr, "keywords");
  std::vector<std::string> paramSpecStarredNames;

  auto emitParamSpecStarredPosargs =
      [&]() -> std::optional<std::vector<Value>> {
    if (!leadingArgs.empty())
      return std::nullopt;
    bool sawParamSpecStarred = false;
    paramSpecStarredNames.clear();
    std::vector<Value> values;
    for (const parser::NodePtr &arg : args) {
      if (!arg)
        return std::vector<Value>{};
      if (arg->kind != "Starred") {
        Value value = emitExpression(*arg);
        if (!value.value)
          return std::vector<Value>{};
        values.push_back(value);
        continue;
      }

      const parser::NodePtr *starredValue = nodeField(*arg, "value");
      if (!starredValue || !*starredValue)
        return std::vector<Value>{};
      Value posargsValue = emitExpression(**starredValue);
      if (!posargsValue.value)
        return std::vector<Value>{};
      if (!posargsValue.paramSpecArgs) {
        if (sawParamSpecStarred)
          error(**starredValue, "mixed static *args forwarding for function '" +
                                    info.name +
                                    "' supports only ParamSpec starred values");
        return std::nullopt;
      }
      sawParamSpecStarred = true;
      if (posargsValue.paramSpecArgs.hasVararg()) {
        error(**starredValue, "ParamSpec *args forwarding for function '" +
                                  info.name +
                                  "' must have a finite positional pack");
        return std::vector<Value>{};
      }
      llvm::ArrayRef<mlir::Type> exactTypes =
          posargsValue.paramSpecArgs.getPositionalTypes();
      for (mlir::StringAttr name :
           posargsValue.paramSpecArgs.getPositionalNames())
        paramSpecStarredNames.push_back(name.getValue().str());
      for (auto [index, type] : llvm::enumerate(exactTypes)) {
        mlir::Value indexValue = builder.create<mlir::arith::ConstantIndexOp>(
            loc(**starredValue), static_cast<int64_t>(index));
        py::CallableType contract =
            unaryMethodContract(posargsValue.type, indexValue.getType(), type);
        mlir::Value value =
            builder.create<py::GetItemOp>(loc(**starredValue), type, contract,
                                          posargsValue.value, indexValue);
        values.push_back(Value{value, type});
      }
    }
    if (!sawParamSpecStarred)
      return std::nullopt;
    return values;
  };

  if (args.size() == 1 && args.front() && args.front()->kind == "Starred" &&
      leadingArgs.empty() && (!keywords || keywords->empty())) {
    const parser::NodePtr *starredValue = nodeField(*args.front(), "value");
    if (starredValue && *starredValue) {
      Value posargsValue = emitExpression(**starredValue);
      if (!posargsValue.value)
        return std::nullopt;
      if (!mlir::isa<py::TupleType>(posargsValue.type)) {
        error(**starredValue, "static *args forwarding for function '" +
                                  info.name + "' requires a tuple value");
        return std::nullopt;
      }
      if (posargsValue.paramSpecArgs &&
          !posargsValue.paramSpecArgs.hasVararg()) {
        llvm::ArrayRef<mlir::Type> exactTypes =
            posargsValue.paramSpecArgs.getPositionalTypes();
        if (exactTypes.empty())
          return CallArgumentTuples{emitEmptyTuple(), emitEmptyTuple(),
                                    emitEmptyTuple()};
        llvm::SmallVector<mlir::Value> values;
        values.reserve(exactTypes.size());
        for (auto [index, type] : llvm::enumerate(exactTypes)) {
          mlir::Value indexValue = builder.create<mlir::arith::ConstantIndexOp>(
              loc(**starredValue), static_cast<int64_t>(index));
          py::CallableType contract = unaryMethodContract(
              posargsValue.type, indexValue.getType(), type);
          values.push_back(
              builder.create<py::GetItemOp>(loc(**starredValue), type, contract,
                                            posargsValue.value, indexValue));
        }
        py::TupleType tupleType = py::TupleType::get(&context, exactTypes);
        mlir::Value tuple = builder.create<py::TupleCreateOp>(
            loc(**starredValue), tupleType, values);
        return CallArgumentTuples{Value{tuple, tupleType}, emitEmptyTuple(),
                                  emitEmptyTuple()};
      }
      return CallArgumentTuples{posargsValue, emitEmptyTuple(),
                                emitEmptyTuple()};
    }
  }

  std::optional<std::vector<Value>> generatedPosargs =
      emitParamSpecStarredPosargs();
  std::optional<std::vector<const parser::Node *>> expandedArgs;
  if (!generatedPosargs) {
    expandedArgs = expandStaticCallArgs(
        expr, args,
        info.varargType ? std::nullopt
                        : std::optional<std::size_t>(positionalLimit),
        info.name);
    if (!expandedArgs)
      return std::nullopt;
  }
  std::optional<std::vector<StaticKeywordArg>> expandedKeywords =
      expandStaticCallKeywords(expr, info.name);
  if (!expandedKeywords)
    return std::nullopt;
  if (generatedPosargs && !paramSpecStarredNames.empty()) {
    expandedKeywords->erase(
        std::remove_if(expandedKeywords->begin(), expandedKeywords->end(),
                       [&](const StaticKeywordArg &keyword) {
                         return keyword.generatedValue &&
                                std::find(paramSpecStarredNames.begin(),
                                          paramSpecStarredNames.end(),
                                          keyword.name) !=
                                    paramSpecStarredNames.end();
                       }),
        expandedKeywords->end());
  }

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
    Value value = indexed.value();
    if (!value.value) {
      ok = false;
      continue;
    }
    mlir::Type expected = info.argTypes[formalIndex];
    value = coerceToExpectedType(expr, std::move(value), expected);
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
    value = coerceToExpectedType(node, std::move(value), expected);
    if (!typeAssignable(expected, value.type)) {
      error(node, "argument '" + info.argNames[formalIndex] +
                      "' for function '" + info.name + "' must be " +
                      typeString(expected) + ", got " + typeString(value.type));
      ok = false;
      return Value{{}, expected};
    }
    return value;
  };
  llvm::ArrayRef<mlir::Type> varargTypes;
  bool homogeneousVararg = false;
  if (info.varargType) {
    auto tupleType = mlir::dyn_cast<py::TupleType>(info.varargType);
    if (!tupleType) {
      error(expr, "invalid vararg signature for '" + info.name + "'");
      return std::nullopt;
    }
    varargTypes = tupleType.getElementTypes();
    homogeneousVararg = varargTypes.size() == 1;
  }

  auto emitVarargActual = [&](const parser::Node &node,
                              std::size_t varargIndex) -> Value {
    mlir::Type expected;
    if (homogeneousVararg)
      expected = varargTypes.front();
    else if (varargIndex < varargTypes.size())
      expected = varargTypes[varargIndex];
    else {
      error(node, "too many vararg elements for function '" + info.name + "'");
      ok = false;
      return Value{{}, py::ObjectType::get(&context)};
    }

    Value value = emitExpressionWithExpectedType(node, expected);
    if (!value.value) {
      ok = false;
      return Value{{}, expected};
    }
    if (!typeAssignable(expected, value.type)) {
      error(node, "vararg element for function '" + info.name + "' must be " +
                      typeString(expected) + ", got " + typeString(value.type));
      ok = false;
      return Value{{}, expected};
    }
    return value;
  };

  const std::size_t positionalActualCount =
      generatedPosargs ? generatedPosargs->size() : expandedArgs->size();
  for (std::size_t index = 0; index < positionalActualCount; ++index) {
    std::size_t formalIndex = firstFormal + index;
    Value value;
    if (generatedPosargs) {
      value = (*generatedPosargs)[index];
      mlir::Type expected =
          formalIndex < info.positionalCount
              ? info.argTypes[formalIndex]
              : (homogeneousVararg ? varargTypes.front() : mlir::Type{});
      if (!expected && formalIndex >= info.positionalCount &&
          formalIndex - info.positionalCount < varargTypes.size())
        expected = varargTypes[formalIndex - info.positionalCount];
      if (!expected) {
        error(expr,
              "too many vararg elements for function '" + info.name + "'");
        ok = false;
        continue;
      }
      value = coerceToExpectedType(expr, std::move(value), expected);
      if (!typeAssignable(expected, value.type)) {
        error(expr, "argument for function '" + info.name + "' must be " +
                        typeString(expected) + ", got " +
                        typeString(value.type));
        ok = false;
        continue;
      }
    } else {
      if (!(*expandedArgs)[index]) {
        ok = false;
        continue;
      }
      value = formalIndex < info.positionalCount
                  ? emitActual(*(*expandedArgs)[index], formalIndex)
                  : emitVarargActual(*(*expandedArgs)[index],
                                     formalIndex - info.positionalCount);
    }
    if (!value.value)
      continue;
    posargs.push_back(value);
    if (formalIndex < formalCount)
      filled[formalIndex] = true;
  }

  if (!expandedKeywords->empty()) {
    std::vector<std::string> seen;
    for (const StaticKeywordArg &keyword : *expandedKeywords) {
      if (!keyword.anchor || (!keyword.value && !keyword.generatedValue)) {
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
        Value value;
        const parser::Node *valueAnchor =
            keyword.value ? keyword.value : keyword.anchor;
        if (keyword.generatedValue) {
          value = *keyword.generatedValue;
        } else {
          value =
              emitExpressionWithExpectedType(*keyword.value, kwargValueType);
          if (!value.value) {
            ok = false;
            continue;
          }
        }
        if (!typeAssignable(kwargValueType, value.type)) {
          error(*valueAnchor, "keyword argument '" + keyword.name +
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
              ? std::min(positionalActualCount,
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

      Value value;
      const parser::Node *valueAnchor =
          keyword.value ? keyword.value : keyword.anchor;
      if (keyword.generatedValue) {
        mlir::Type expected = info.argTypes[formalIndex];
        value = coerceToExpectedType(*valueAnchor, *keyword.generatedValue,
                                     expected);
        if (!typeAssignable(expected, value.type)) {
          error(*valueAnchor, "argument '" + info.argNames[formalIndex] +
                                  "' for function '" + info.name +
                                  "' must be " + typeString(expected) +
                                  ", got " + typeString(value.type));
          ok = false;
          continue;
        }
      } else {
        value = emitActual(*keyword.value, formalIndex);
      }
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
  if (info.varargType && !homogeneousVararg) {
    std::size_t suppliedVarargs = positionalActualCount > positionalLimit
                                      ? positionalActualCount - positionalLimit
                                      : 0;
    if (suppliedVarargs != varargTypes.size()) {
      error(expr, "function '" + info.name + "' expects " +
                      std::to_string(varargTypes.size()) +
                      " vararg elements, got " +
                      std::to_string(suppliedVarargs));
      ok = false;
    }
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
  if (!info.isSpecialization && !info.typeParameters.empty()) {
    std::optional<FunctionInfo> specialized =
        specializeGenericFunctionCall(expr, info, args);
    if (!specialized)
      return Value{{}, info.resultType};
    return emitFunctionCall(expr, *specialized, args);
  }

  if (!info.isSpecialization && !info.varargType && hasCallableFormal(info)) {
    std::optional<FunctionInfo> specialized =
        specializeFunctionCall(expr, info, args);
    if (!specialized)
      return Value{{}, info.resultType};
    return emitFunctionCall(expr, *specialized, args);
  }

  if (hasProtocolFormal(info)) {
    std::optional<FunctionInfo> specialized =
        specializeProtocolFunctionCall(expr, info, args);
    if (!specialized)
      return Value{{}, info.resultType};
    return emitFunctionCall(expr, *specialized, args);
  }

  if (!info.isSpecialization && hasClassFormal(info)) {
    std::optional<FunctionInfo> specialized =
        specializeClassFunctionCall(expr, info, args);
    if (specialized)
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
    auto call = builder.create<py::CallOp>(
        loc(expr), mlir::TypeRange{info.resultType}, callee,
        tuples->posargs.value, tuples->kwnames.value, tuples->kwvalues.value);
    attachReturnedCallableSymbol(call.getOperation(), info);
    return applyReturnedClassSummary(
        Value{call.getResults().front(), info.resultType}, info);
  }

  bool hasStarredArg = llvm::any_of(args, [](const parser::NodePtr &arg) {
    return arg && arg->kind == "Starred";
  });
  if (!info.isNative && !info.isAsync && hasStarredArg) {
    std::optional<CallArgumentTuples> tuples =
        emitExplicitCallArgumentTuples(expr, info, args);
    if (!tuples)
      return Value{{}, info.resultType};
    mlir::Value callee = builder.create<py::CallableObjectOp>(
        loc(), info.functionType, info.symbolName);
    if (info.mayThrow)
      return emitMayThrowFunctionCall(expr, info, callee, tuples->posargs,
                                      tuples->kwnames, tuples->kwvalues);
    auto call = builder.create<py::CallOp>(
        loc(), mlir::TypeRange{info.resultType}, callee, tuples->posargs.value,
        tuples->kwnames.value, tuples->kwvalues.value);
    attachReturnedCallableSymbol(call.getOperation(), info);
    return applyReturnedClassSummary(
        Value{call.getResults().front(), info.resultType}, info);
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
    if (!py::isCallableType(returned.type)) {
      error(expr, "returned callable argument for function '" + info.name +
                      "' must be Callable, got " + typeString(returned.type));
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
    callable.callableInfo = std::make_shared<FunctionInfo>(returned);
    return callable;
  }

  if (!info.mayThrow && info.returnedClassArgIndex) {
    std::size_t index = *info.returnedClassArgIndex;
    if (index >= argValues->size()) {
      error(expr, "returned class argument summary for function '" + info.name +
                      "' has invalid argument index");
      return Value{{}, info.resultType};
    }
    Value returned =
        coerceToExpectedType(expr, (*argValues)[index], info.resultType);
    if (!returned.value)
      return Value{{}, info.resultType};
    return returned;
  }

  if (!info.mayThrow && info.returnedValueArgIndex) {
    std::size_t index = *info.returnedValueArgIndex;
    if (index >= argValues->size()) {
      error(expr, "returned value argument summary for function '" + info.name +
                      "' has invalid argument index");
      return Value{{}, info.resultType};
    }
    Value returned =
        coerceToExpectedType(expr, (*argValues)[index], info.resultType);
    if (!returned.value)
      return Value{{}, info.resultType};
    if (!typeAssignable(info.resultType, returned.type)) {
      error(expr, "returned value argument for function '" + info.name +
                      "' must be " + typeString(info.resultType) + ", got " +
                      typeString(returned.type));
      return Value{{}, info.resultType};
    }
    return returned;
  }

  llvm::SmallVector<mlir::Value> operands;
  for (const Value &argValue : *argValues)
    operands.push_back(argValue.value);

  if (info.isAsync) {
    mlir::Type resultType = coroutineType(info.resultType);
    mlir::Value callee = builder.create<py::CallableObjectOp>(
        loc(expr), info.functionType, info.symbolName);
    CallArgumentTuples tuples = emitCallArgumentTuples(info, *argValues);
    auto call = builder.create<py::CallOp>(
        loc(expr), mlir::TypeRange{resultType}, callee, tuples.posargs.value,
        tuples.kwnames.value, tuples.kwvalues.value);
    return Value{call.getResults().front(), resultType};
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
    return applyReturnedClassSummary(
        Value{call.getResults().front(), info.resultType}, info);
  }

  mlir::Value callee = builder.create<py::CallableObjectOp>(
      loc(), info.functionType, info.symbolName);

  CallArgumentTuples tuples = emitCallArgumentTuples(info, *argValues);
  if (info.mayThrow)
    return emitMayThrowFunctionCall(expr, info, callee, tuples.posargs,
                                    tuples.kwnames, tuples.kwvalues);
  auto call = builder.create<py::CallOp>(
      loc(), mlir::TypeRange{info.resultType}, callee, tuples.posargs.value,
      tuples.kwnames.value, tuples.kwvalues.value);
  attachReturnedCallableSymbol(call.getOperation(), info);
  return applyReturnedClassSummary(
      Value{call.getResults().front(), info.resultType}, info);
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
  return applyReturnedClassSummary(
      emitMayThrowCallableCall(expr, callee, info.resultType, posargs, kwnames,
                               kwvalues),
      info);
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

std::optional<Value> Builder::Impl::emitProtocolUnaryConversionCall(
    const parser::Node &anchor, Value receiver, llvm::StringRef protocolName,
    llvm::StringRef methodName, mlir::Type resultType) {
  if (std::optional<Value> concrete = concreteProtocolValue(receiver))
    receiver = *concrete;
  if (!mlir::isa<py::ClassType>(receiver.type))
    return std::nullopt;

  std::optional<py::ProtocolType> protocol = protocolType(protocolName, {});
  if (!protocol) {
    error(anchor, "unknown protocol '" + protocolName.str() + "'");
    return Value{{}, resultType};
  }
  if (!typeAssignable(*protocol, receiver.type)) {
    auto classType = mlir::cast<py::ClassType>(receiver.type);
    auto classFound = classes.find(classType.getClassName().str());
    if (classFound != classes.end() &&
        classFound->second.methods.count(methodName.str())) {
      error(anchor, "class '" + classType.getClassName().str() +
                        "' defines method '" + methodName.str() +
                        "' but does not satisfy protocol '" +
                        protocolName.str() + "'");
      return Value{{}, resultType};
    }
    return std::nullopt;
  }

  std::optional<FunctionInfo> method =
      resolveClassMethod(anchor, receiver, methodName);
  if (!method)
    return Value{{}, resultType};
  if (!typeAssignable(resultType, method->resultType)) {
    error(anchor, "method '" + methodName.str() + "' selected via protocol '" +
                      protocolName.str() + "' returns " +
                      typeString(method->resultType) + ", expected " +
                      typeString(resultType));
    return Value{{}, resultType};
  }

  Value result = emitResolvedMethodCall(anchor, receiver, *method, {});
  if (!result.value)
    return Value{{}, resultType};
  if (result.type == resultType)
    return result;
  result = coerceToExpectedType(anchor, std::move(result), resultType);
  if (!result.value || !typeAssignable(resultType, result.type))
    return Value{{}, resultType};
  return result;
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
  if (std::optional<Value> converted = emitProtocolUnaryConversionCall(
          *args.front(), value, "SupportsInt", "__int__", intType()))
    return *converted;

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
  if (std::optional<Value> converted = emitProtocolUnaryConversionCall(
          *args.front(), value, "SupportsFloat", "__float__", floatType()))
    return *converted;

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

Value Builder::Impl::emitIsinstanceCall(const parser::Node &expr) {
  auto isinstanceMatch = matchIsinstanceCall(expr);
  if (!isinstanceMatch) {
    error(expr, "isinstance requires a value and one statically resolvable "
                "type argument");
    return Value{{}, boolType()};
  }
  Value value = emitExpression(*isinstanceMatch->first);
  if (!value.value)
    return Value{{}, boolType()};
  mlir::Type memberType = isinstanceMatch->second;
  mlir::Value bit;
  auto falseBit = [&]() -> mlir::Value {
    return builder.create<mlir::arith::ConstantIntOp>(loc(expr), 0, 1);
  };
  auto trueBit = [&]() -> mlir::Value {
    return builder.create<mlir::arith::ConstantIntOp>(loc(expr), 1, 1);
  };
  auto exactTruth = [&](const Value &input,
                        mlir::Type target) -> std::optional<bool> {
    if (mlir::isa<py::NoneType>(target)) {
      if (mlir::isa<py::NoneType>(input.type))
        return true;
      if (input.exactClass || mlir::isa<py::ClassType>(input.type))
        return false;
      return std::nullopt;
    }
    auto targetClass = mlir::dyn_cast<py::ClassType>(target);
    if (!targetClass)
      return std::nullopt;
    if (input.exactClass)
      return classSubtypeOf(*input.exactClass, targetClass.getClassName());
    if (input.provenClass) {
      if (classSubtypeOf(*input.provenClass, targetClass.getClassName()))
        return true;
      if (!classSubtypeOf(targetClass.getClassName(), *input.provenClass))
        return false;
    }
    return std::nullopt;
  };
  auto classTest = [&](Value input, mlir::Type target) -> mlir::Value {
    auto inputClass = mlir::dyn_cast<py::ClassType>(input.type);
    auto targetClass = mlir::dyn_cast<py::ClassType>(target);
    if (!input.value || !inputClass || !targetClass)
      return falseBit();
    if (std::optional<bool> truth = exactTruth(input, target))
      return *truth ? trueBit() : falseBit();
    if (typeSubtypeOf(input.type, target))
      return trueBit();
    if (!classSubtypeOf(targetClass.getClassName(), inputClass.getClassName()))
      return falseBit();
    return builder.create<py::ClassTestOp>(loc(expr), i1Type(), input.value,
                                           mlir::TypeAttr::get(target));
  };
  if (std::optional<bool> truth = exactTruth(value, memberType)) {
    bit = *truth ? trueBit() : falseBit();
  } else if (auto unionType = mlir::dyn_cast<py::UnionType>(value.type)) {
    std::vector<UnionMemberMatch> matches =
        unionMembersMatchingType(unionType, memberType,
                                 /*requireLayoutCompatibleDowncast=*/false);
    if (matches.empty()) {
      bit = falseBit();
    } else if (matches.size() == unionType.getMemberTypes().size() &&
               llvm::all_of(matches, [](const UnionMemberMatch &match) {
                 return match.sourceMember == match.narrowedType;
               })) {
      bit = trueBit();
    } else {
      for (const UnionMemberMatch &match : matches) {
        mlir::Value test;
        mlir::Value live = builder.create<py::UnionTestOp>(
            loc(expr), i1Type(), value.value,
            mlir::TypeAttr::get(match.sourceMember));
        if (match.sourceMember == match.narrowedType) {
          test = live;
        } else {
          mlir::Value unwrapped = builder.create<py::UnionUnwrapOp>(
              loc(expr), match.sourceMember, value.value);
          Value unwrappedValue{unwrapped, match.sourceMember, value.exactClass,
                               value.provenClass};
          auto ifOp = builder.create<mlir::scf::IfOp>(
              loc(expr), mlir::TypeRange{i1Type()}, live,
              /*withElseRegion=*/true);
          {
            mlir::OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointToStart(ifOp.thenBlock());
            mlir::Value dynamicTest =
                classTest(unwrappedValue, match.narrowedType);
            builder.create<mlir::scf::YieldOp>(loc(expr), dynamicTest);
          }
          {
            mlir::OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointToStart(ifOp.elseBlock());
            builder.create<mlir::scf::YieldOp>(loc(expr), falseBit());
          }
          test = ifOp.getResult(0);
        }
        bit = bit ? builder.create<mlir::arith::OrIOp>(loc(expr), bit, test)
                  : test;
      }
    }
  } else {
    bit = classTest(value, memberType);
  }
  mlir::Value result =
      builder.create<py::CastFromPrimOp>(loc(expr), boolType(), bit);
  return Value{result, boolType()};
}

std::optional<SyncIteratorResolution>
Builder::Impl::resolveSyncIterator(const parser::Node &anchor, Value input,
                                   bool requireConcreteIterator) {
  if (std::optional<Value> concrete = concreteProtocolValue(input))
    input = *concrete;

  auto classType = mlir::dyn_cast<py::ClassType>(input.type);
  if (!classType) {
    std::optional<mlir::Type> elementType = protocolIterableElement(input.type);
    if (!elementType) {
      error(anchor, "iter() receiver " + typeString(input.type) +
                        " does not satisfy Iterable[T]");
      return std::nullopt;
    }
    error(anchor,
          "iter() on " + typeString(input.type) +
              " resolves through typing.mlir, but lowering for non-class "
              "iterator producers is not implemented yet");
    return std::nullopt;
  }

  std::optional<FunctionInfo> iterMethod =
      resolveClassMethod(anchor, input, "__iter__");
  if (!iterMethod)
    return std::nullopt;
  if (iterMethod->isAsync) {
    error(anchor, "iter() requires synchronous __iter__");
    return std::nullopt;
  }
  if (!iterMethod->methodKind.empty() && iterMethod->methodKind != "instance") {
    error(anchor, "iter() requires instance method __iter__");
    return std::nullopt;
  }

  mlir::Type iteratorType = iterMethod->resultType;
  mlir::Type elementType;
  std::optional<FunctionInfo> nextMethod;
  bool iterReturnsReceiver = false;
  if (auto iteratorProtocol = mlir::dyn_cast<py::ProtocolType>(iteratorType)) {
    if (iteratorProtocol.getProtocolName() == "Iterator" &&
        iteratorProtocol.getArguments().size() == 1)
      elementType = iteratorProtocol.getArguments().front();
    if (requireConcreteIterator && iterMethod->returnedValueArgIndex &&
        *iterMethod->returnedValueArgIndex == 0 &&
        typeAssignable(iteratorProtocol, input.type)) {
      iteratorType = input.type;
      iterReturnsReceiver = true;
      Value iteratorReceiver{{}, iteratorType};
      nextMethod = resolveClassMethod(anchor, iteratorReceiver, "__next__");
      if (!nextMethod)
        return std::nullopt;
      if (nextMethod->isAsync) {
        error(anchor, "iter() requires synchronous __next__");
        return std::nullopt;
      }
      if (!nextMethod->methodKind.empty() &&
          nextMethod->methodKind != "instance") {
        error(anchor, "iter() requires instance method __next__");
        return std::nullopt;
      }
      elementType = nextMethod->resultType;
    }
    if (requireConcreteIterator) {
      if (!nextMethod) {
        error(anchor,
              "for iteration requires __iter__ to return a concrete iterator "
              "class, got " +
                  typeString(iteratorProtocol));
        return std::nullopt;
      }
    }
  } else if (auto iteratorClass = mlir::dyn_cast<py::ClassType>(iteratorType)) {
    Value iteratorReceiver{{}, iteratorClass};
    nextMethod = resolveClassMethod(anchor, iteratorReceiver, "__next__");
    if (!nextMethod)
      return std::nullopt;
    if (nextMethod->isAsync) {
      error(anchor, "iter() requires synchronous __next__");
      return std::nullopt;
    }
    if (!nextMethod->methodKind.empty() &&
        nextMethod->methodKind != "instance") {
      error(anchor, "iter() requires instance method __next__");
      return std::nullopt;
    }
    elementType = nextMethod->resultType;
  }
  if (!elementType) {
    error(anchor, "class __iter__ on " + typeString(input.type) +
                      " must return Iterator[T] or a class with __next__");
    return std::nullopt;
  }

  std::optional<py::ProtocolType> iterableProtocol =
      protocolType("Iterable", {elementType});
  if (!iterableProtocol) {
    error(anchor, "failed to instantiate Iterable protocol");
    return std::nullopt;
  }
  if (!typeAssignable(*iterableProtocol, input.type)) {
    error(anchor, "receiver " + typeString(input.type) +
                      " does not satisfy Iterable[" + typeString(elementType) +
                      "]");
    return std::nullopt;
  }

  std::optional<py::ProtocolType> iteratorProtocol =
      protocolType("Iterator", {elementType});
  if (!iteratorProtocol) {
    error(anchor, "failed to instantiate Iterator protocol");
    return std::nullopt;
  }
  if (!typeAssignable(*iteratorProtocol, iteratorType)) {
    error(anchor, "__iter__ result " + typeString(iteratorType) +
                      " does not satisfy Iterator[" + typeString(elementType) +
                      "]");
    return std::nullopt;
  }

  return SyncIteratorResolution{
      input,        *iterMethod, nextMethod.value_or(FunctionInfo{}),
      iteratorType, elementType, iterReturnsReceiver};
}

Value Builder::Impl::emitIterCall(const parser::Node &expr,
                                  const std::vector<parser::NodePtr> &args) {
  const std::vector<parser::NodePtr> *keywords =
      nodeListField(expr, "keywords");
  if (args.size() != 1 || !args.front()) {
    error(expr, "iter() expects exactly one argument");
    return Value{{}, noneType()};
  }
  if (keywords && !keywords->empty()) {
    error(expr, "iter() does not accept keyword arguments");
    return Value{{}, noneType()};
  }

  Value input = emitExpression(*args.front());
  if (!input.value)
    return Value{{}, noneType()};
  std::optional<SyncIteratorResolution> resolution =
      resolveSyncIterator(*args.front(), input,
                          /*requireConcreteIterator=*/false);
  if (!resolution)
    return Value{{}, noneType()};

  std::optional<std::vector<Value>> prepared =
      prepareResolvedMethodCallArguments(*args.front(), resolution->iterable,
                                         resolution->iterMethod, {});
  if (!prepared)
    return Value{{}, resolution->iteratorType};
  bool returnedSelf =
      resolution->iterMethod.returnedValueArgIndex &&
      *resolution->iterMethod.returnedValueArgIndex == 0 &&
      typeAssignable(resolution->iteratorType, resolution->iterable.type);
  mlir::Type resultType =
      returnedSelf ? resolution->iterable.type : resolution->iteratorType;
  mlir::UnitAttr returnedSelfAttr =
      returnedSelf ? builder.getUnitAttr() : mlir::UnitAttr{};
  auto iterOp = builder.create<py::IterOp>(
      loc(expr), resultType, resolution->iterMethod.symbolName,
      resolution->iterMethod.functionType, (*prepared)[0].value,
      returnedSelfAttr);
  if (returnedSelf)
    return Value{iterOp.getResult(), resultType,
                 resolution->iterable.exactClass,
                 resolution->iterable.provenClass};
  return Value{iterOp.getResult(), resultType};
}

Value Builder::Impl::emitNextValue(const parser::Node &anchor, Value iterator,
                                   llvm::StringRef contextLabel) {
  if (!iterator.value)
    return Value{{}, noneType()};
  if (std::optional<Value> concrete = concreteProtocolValue(iterator))
    iterator = *concrete;
  if (!mlir::isa<py::ClassType>(iterator.type)) {
    std::optional<mlir::Type> elementType =
        protocols::Table::get(context).resolveMethodResultOn(iterator.type,
                                                             "__next__", {});
    if (elementType) {
      error(anchor, contextLabel.str() + " on " + typeString(iterator.type) +
                        " resolves statically to " + typeString(*elementType) +
                        ", but lowering for non-class iterators is not "
                        "implemented yet");
      return Value{{}, *elementType};
    }
    error(anchor, contextLabel.str() + " receiver " +
                      typeString(iterator.type) + " has no __next__ contract");
    return Value{{}, noneType()};
  }

  std::optional<FunctionInfo> nextMethod =
      resolveClassMethod(anchor, iterator, "__next__");
  if (!nextMethod)
    return Value{{}, noneType()};
  if (nextMethod->isAsync) {
    error(anchor, contextLabel.str() + " requires synchronous __next__");
    return Value{{}, nextMethod->resultType};
  }
  if (!nextMethod->methodKind.empty() && nextMethod->methodKind != "instance") {
    error(anchor, contextLabel.str() + " requires instance method __next__");
    return Value{{}, nextMethod->resultType};
  }

  std::optional<std::vector<Value>> prepared =
      prepareResolvedMethodCallArguments(anchor, iterator, *nextMethod, {});
  if (!prepared)
    return Value{{}, nextMethod->resultType};

  mlir::Region *region = builder.getBlock()->getParent();
  mlir::Block *raiseBlock = new mlir::Block();
  mlir::Block *afterBlock = new mlir::Block();
  region->push_back(raiseBlock);
  region->push_back(afterBlock);

  auto nextOp = builder.create<py::NextOp>(
      loc(anchor),
      mlir::TypeRange{nextMethod->resultType, i1Type(), iterator.type},
      nextMethod->symbolName, nextMethod->functionType, (*prepared)[0].value);
  builder.create<mlir::cf::CondBranchOp>(loc(anchor), nextOp.getValid(),
                                         afterBlock, mlir::ValueRange{},
                                         raiseBlock, mlir::ValueRange{});
  blockTerminated = true;

  builder.setInsertionPointToStart(raiseBlock);
  auto exception = builder.create<py::ExceptionNewOp>(
      loc(anchor), exceptionType(), mlir::ValueRange{});
  exception->setAttr("py.exception.class",
                     builder.getStringAttr("StopIteration"));
  builder.create<py::RaiseOp>(loc(anchor), exception.getResult());

  builder.setInsertionPointToStart(afterBlock);
  blockTerminated = false;
  return Value{nextOp.getElement(), nextMethod->resultType};
}

Value Builder::Impl::emitNextCall(const parser::Node &expr,
                                  const std::vector<parser::NodePtr> &args) {
  const std::vector<parser::NodePtr> *keywords =
      nodeListField(expr, "keywords");
  if (args.size() != 1 || !args.front()) {
    error(expr, "next() expects exactly one iterator argument");
    return Value{{}, noneType()};
  }
  if (keywords && !keywords->empty()) {
    error(expr, "next() does not accept keyword arguments");
    return Value{{}, noneType()};
  }
  Value iterator = emitExpression(*args.front());
  if (!iterator.value)
    return Value{{}, noneType()};
  return emitNextValue(expr, iterator, "next()");
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
  if (std::optional<Value> concrete = concreteProtocolValue(input))
    input = *concrete;

  if (auto classType = mlir::dyn_cast<py::ClassType>(input.type)) {
    std::optional<py::ProtocolType> sized = protocolType("Sized", {});
    if (!sized) {
      error(*args.front(), "unknown Sized protocol");
      return Value{{}, intType()};
    }
    if (!classConformsToProtocol(classType, *sized)) {
      error(*args.front(), "class receiver " + typeString(input.type) +
                               " does not satisfy Sized.__len__ -> int");
      return Value{{}, intType()};
    }
    std::optional<FunctionInfo> method =
        resolveClassMethod(*args.front(), input, "__len__");
    if (!method)
      return Value{{}, intType()};
    if (method->resultType != intType()) {
      error(*args.front(), "class __len__ on " + typeString(input.type) +
                               " must return !py.int, got " +
                               typeString(method->resultType));
      return Value{{}, intType()};
    }
    return emitResolvedMethodCall(expr, input, *method, {});
  }

  std::optional<protocols::ProtocolMethod> lenContract =
      protocols::Table::get(context).resolveMethodContractOn(input.type,
                                                             "__len__", {});
  if (!lenContract) {
    error(*args.front(), "len() receiver " + typeString(input.type) +
                             " has no __len__ contract in typing.mlir");
    return Value{{}, intType()};
  }

  llvm::ArrayRef<mlir::Type> results = lenContract->signature.getResultTypes();
  if (results.size() != 1 || results.front() != intType()) {
    error(*args.front(), "len() receiver " + typeString(input.type) +
                             " has invalid __len__ contract: expected one "
                             "!py.int result");
    return Value{{}, intType()};
  }

  if (mlir::isa<py::ProtocolType>(input.type)) {
    error(expr, "protocol len() on " + typeString(input.type) +
                    " resolves statically to !py.int, but lowering for "
                    "protocol-typed receivers is not implemented yet");
    return Value{{}, intType()};
  }

  mlir::Value result = builder.create<py::LenOp>(
      loc(expr), intType(), lenContract->signature, input.value);
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
  py::CallableType signature =
      functionSignatureType({}, noneType(), varargType);
  mlir::Type printType = signature;
  mlir::Value callee = builder.create<py::CallableObjectOp>(
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
  auto call =
      builder.create<py::CallOp>(loc(), mlir::TypeRange{noneType()}, callee,
                                 posargs.value, kwnames.value, kwvalues.value);
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
