#include "BuilderImpl.h"

#include "Parser.h"

#include "mlir/Dialect/Async/IR/Async.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <charconv>
#include <cstring>
#include <functional>
#include <set>
#include <system_error>

namespace lython::emitter {
namespace {

std::int64_t clampSliceIndex(std::int64_t value, std::int64_t length,
                             std::int64_t lower, std::int64_t upper) {
  if (value < 0)
    value += length;
  if (value < lower)
    return lower;
  if (value > upper)
    return upper;
  return value;
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

void collectProtocolMethodNames(const protocols::Table &table,
                                llvm::StringRef protocolName,
                                std::set<std::string> &out,
                                std::set<std::string> &visited,
                                unsigned depth = 0) {
  if (depth > 16)
    return;
  std::string key = protocolName.str();
  if (!visited.insert(key).second)
    return;
  const protocols::ProtocolInfo *info = table.lookup(protocolName);
  if (!info)
    return;
  for (const auto &[methodName, ignored] : info->methods) {
    (void)ignored;
    out.insert(methodName);
  }
  for (const protocols::ProtocolBase &base : info->bases)
    collectProtocolMethodNames(table, base.name, out, visited, depth + 1);
}

bool typeInferenceDependsOnEnvironmentUncached(const parser::Node &node) {
  if (node.kind == "Name" || node.kind == "Attribute" || node.kind == "Call" ||
      node.kind == "Subscript" || node.kind == "Await" ||
      node.kind == "Lambda" || node.kind == "ListComp" ||
      node.kind == "SetComp" || node.kind == "DictComp" ||
      node.kind == "GeneratorExp" || node.kind == "Yield" ||
      node.kind == "YieldFrom")
    return true;

  for (const parser::Field &field : node.fields) {
    if (const auto *child = std::get_if<parser::NodePtr>(&field.value)) {
      if (*child && typeInferenceDependsOnEnvironmentUncached(**child))
        return true;
      continue;
    }
    if (const auto *children =
            std::get_if<std::vector<parser::NodePtr>>(&field.value)) {
      for (const parser::NodePtr &child : *children)
        if (child && typeInferenceDependsOnEnvironmentUncached(*child))
          return true;
    }
  }
  return false;
}

bool isLenCall(const parser::Node &node) {
  if (node.kind != "Call")
    return false;
  const parser::NodePtr *func = nodeField(node, "func");
  const std::vector<parser::NodePtr> *args = nodeListField(node, "args");
  const std::vector<parser::NodePtr> *keywords =
      nodeListField(node, "keywords");
  if (!func || !*func || (*func)->kind != "Name" || !args ||
      args->size() != 1 || !args->front() || (keywords && !keywords->empty()))
    return false;
  const std::string *name = stringField(**func, "id");
  return name && *name == "len";
}

bool isRangeCall(const parser::Node &node) {
  if (node.kind != "Call")
    return false;
  const parser::NodePtr *func = nodeField(node, "func");
  if (!func || !*func || (*func)->kind != "Name")
    return false;
  const std::string *name = stringField(**func, "id");
  return name && *name == "range";
}

bool isEllipsisConstant(const parser::NodePtr &expr) {
  if (!expr || expr->kind != "Constant")
    return false;
  const parser::FieldValue *value = valueField(*expr, "value");
  return value && std::holds_alternative<parser::Ellipsis>(*value);
}

bool isCallableEllipsisContract(py::CallableType signature) {
  if (!signature.getPositionalTypes().empty() ||
      !signature.getKwOnlyTypes().empty() || !signature.hasVararg() ||
      !signature.hasKwarg() || signature.hasParameterMetadata())
    return false;
  auto varargTuple = mlir::dyn_cast<py::TupleType>(signature.getVarargType());
  if (!varargTuple || varargTuple.getElementTypes().size() != 1 ||
      !mlir::isa<py::ObjectType>(varargTuple.getElementTypes().front()))
    return false;
  auto kwargsDict = mlir::dyn_cast<py::DictType>(signature.getKwargType());
  return kwargsDict && mlir::isa<py::StrType>(kwargsDict.getKeyType()) &&
         mlir::isa<py::ObjectType>(kwargsDict.getValueType());
}

mlir::Type callableVarargElementType(mlir::Type varargType) {
  auto tuple = mlir::dyn_cast_if_present<py::TupleType>(varargType);
  if (!tuple || tuple.getElementTypes().size() != 1)
    return {};
  return tuple.getElementTypes().front();
}

struct LambdaOracle final : typing::Oracle {
  std::function<std::optional<typing::Term>(const typing::Term &,
                                            llvm::StringRef)>
      attributeFn;
  std::function<std::optional<typing::Term>(const typing::Term &,
                                            llvm::ArrayRef<typing::Term>)>
      callFn;
  std::function<std::optional<typing::Term>(const typing::Term &,
                                            const typing::Term &)>
      subscriptFn;
  std::function<std::optional<typing::Term>(const typing::Term &)> awaitFn;

  std::optional<typing::Term> attribute(const typing::Term &owner,
                                        llvm::StringRef name) const override {
    if (attributeFn)
      return attributeFn(owner, name);
    return typing::Oracle::attribute(owner, name);
  }

  std::optional<typing::Term>
  call(const typing::Term &callee,
       llvm::ArrayRef<typing::Term> args) const override {
    if (callFn)
      return callFn(callee, args);
    return typing::Oracle::call(callee, args);
  }

  std::optional<typing::Term>
  subscript(const typing::Term &container,
            const typing::Term &index) const override {
    if (subscriptFn)
      return subscriptFn(container, index);
    return typing::Oracle::subscript(container, index);
  }

  std::optional<typing::Term>
  awaitable(const typing::Term &awaitable) const override {
    if (awaitFn)
      return awaitFn(awaitable);
    return typing::Oracle::awaitable(awaitable);
  }
};

} // namespace

std::optional<typing::Term>
Builder::Impl::typeTermFromType(mlir::Type type) const {
  if (!type)
    return std::nullopt;
  if (mlir::isa<py::IntType>(type))
    return typing::Term::con("int");
  if (mlir::isa<py::FloatType>(type))
    return typing::Term::con("float");
  if (mlir::isa<py::BoolType>(type))
    return typing::Term::con("bool");
  if (mlir::isa<py::StrType>(type))
    return typing::Term::con("str");
  if (mlir::isa<py::NoneType>(type))
    return typing::Term::con("None");
  if (mlir::isa<py::ObjectType>(type))
    return typing::Term::con("object");
  if (mlir::isa<py::ExceptionType>(type))
    return typing::Term::con("Exception");
  if (mlir::isa<py::TracebackType>(type))
    return typing::Term::con("TracebackType");
  if (auto typeTy = mlir::dyn_cast<py::TypeType>(type)) {
    std::optional<typing::Term> instance =
        typeTermFromType(typeTy.getInstanceType());
    if (!instance)
      return std::nullopt;
    return typing::Term::con("type", {*instance});
  }
  if (auto classTy = mlir::dyn_cast<py::ClassType>(type))
    return typing::Term::con("class:" + classTy.getClassName().str());
  if (auto protocolTy = mlir::dyn_cast<py::ProtocolType>(type)) {
    std::vector<typing::Term> args;
    args.reserve(protocolTy.getArguments().size());
    for (mlir::Type argType : protocolTy.getArguments()) {
      std::optional<typing::Term> arg = typeTermFromType(argType);
      if (!arg)
        return std::nullopt;
      args.push_back(*arg);
    }
    return typing::Term::con("protocol:" + protocolTy.getProtocolName().str(),
                             args);
  }
  if (auto listTy = mlir::dyn_cast<py::ListType>(type)) {
    std::optional<typing::Term> element =
        typeTermFromType(listTy.getElementType());
    if (!element)
      return std::nullopt;
    return typing::Term::con("list", {*element});
  }
  if (auto dictTy = mlir::dyn_cast<py::DictType>(type)) {
    std::optional<typing::Term> key = typeTermFromType(dictTy.getKeyType());
    std::optional<typing::Term> value = typeTermFromType(dictTy.getValueType());
    if (!key || !value)
      return std::nullopt;
    return typing::Term::con("dict", {*key, *value});
  }
  if (auto tupleTy = mlir::dyn_cast<py::TupleType>(type)) {
    std::vector<typing::Term> elements;
    elements.reserve(tupleTy.getElementTypes().size());
    for (mlir::Type elementType : tupleTy.getElementTypes()) {
      std::optional<typing::Term> element = typeTermFromType(elementType);
      if (!element)
        return std::nullopt;
      elements.push_back(*element);
    }
    return typing::Term::con("tuple", elements);
  }
  if (auto unionTy = mlir::dyn_cast<py::UnionType>(type)) {
    std::vector<typing::Term> members;
    members.reserve(unionTy.getMemberTypes().size());
    for (mlir::Type memberType : unionTy.getMemberTypes()) {
      std::optional<typing::Term> member = typeTermFromType(memberType);
      if (!member)
        return std::nullopt;
      members.push_back(*member);
    }
    return typing::Term::con("union", members);
  }
  if (auto asyncValue = mlir::dyn_cast<mlir::async::ValueType>(type)) {
    std::optional<typing::Term> value =
        typeTermFromType(asyncValue.getValueType());
    if (!value)
      return std::nullopt;
    return typing::Term::con("async.value", {*value});
  }
  if (py::CallableType signature = py::getCallableContract(type)) {
    std::vector<typing::Term> args;
    std::vector<typing::Term> kwonly;
    std::vector<typing::Term> vararg;
    std::vector<typing::Term> kwarg;
    std::vector<typing::Term> results;
    args.reserve(signature.getPositionalTypes().size());
    kwonly.reserve(signature.getKwOnlyTypes().size());
    results.reserve(signature.getResultTypes().size());
    for (mlir::Type argType : signature.getPositionalTypes()) {
      std::optional<typing::Term> arg = typeTermFromType(argType);
      if (!arg)
        return std::nullopt;
      args.push_back(*arg);
    }
    for (mlir::Type argType : signature.getKwOnlyTypes()) {
      std::optional<typing::Term> arg = typeTermFromType(argType);
      if (!arg)
        return std::nullopt;
      kwonly.push_back(*arg);
    }
    if (signature.hasVararg()) {
      std::optional<typing::Term> arg =
          typeTermFromType(signature.getVarargType());
      if (!arg)
        return std::nullopt;
      vararg.push_back(*arg);
    }
    if (signature.hasKwarg()) {
      std::optional<typing::Term> arg =
          typeTermFromType(signature.getKwargType());
      if (!arg)
        return std::nullopt;
      kwarg.push_back(*arg);
    }
    for (mlir::Type resultType : signature.getResultTypes()) {
      std::optional<typing::Term> result = typeTermFromType(resultType);
      if (!result)
        return std::nullopt;
      results.push_back(*result);
    }
    std::optional<unsigned> positionalOnlyCount;
    if (signature.hasParameterMetadata())
      positionalOnlyCount = signature.getPositionalOnlyCount();
    return typing::Term::func(args, results, kwonly, vararg, kwarg,
                              positionalOnlyCount);
  }
  if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(type))
    return typing::Term::con("Int[" + std::to_string(intTy.getWidth()) + "]");
  if (auto floatTy = mlir::dyn_cast<mlir::FloatType>(type))
    return typing::Term::con("Float[" + std::to_string(floatTy.getWidth()) +
                             "]");
  if (auto tensorTy = mlir::dyn_cast<mlir::RankedTensorType>(type)) {
    std::vector<typing::Term> args;
    std::optional<typing::Term> element =
        typeTermFromType(tensorTy.getElementType());
    if (!element)
      return std::nullopt;
    args.push_back(*element);
    for (std::int64_t dim : tensorTy.getShape())
      args.push_back(typing::Term::con(std::to_string(dim)));
    return typing::Term::con("Tensor", args);
  }
  return typing::Term::con(typeString(type));
}

std::optional<mlir::Type>
Builder::Impl::typeFromTypeTerm(const typing::Term &term) {
  if (term.kind == typing::Term::Kind::Var)
    return std::nullopt;
  if (term.kind == typing::Term::Kind::Func) {
    llvm::SmallVector<mlir::Type> args;
    llvm::SmallVector<mlir::Type> kwonly;
    llvm::SmallVector<mlir::Type> results;
    mlir::Type vararg;
    mlir::Type kwarg;
    if (term.vararg.size() > 1 || term.kwarg.size() > 1)
      return std::nullopt;
    args.reserve(term.args.size());
    kwonly.reserve(term.kwonly.size());
    results.reserve(term.results.size());
    for (const typing::Term &argTerm : term.args) {
      std::optional<mlir::Type> argType = typeFromTypeTerm(argTerm);
      if (!argType)
        return std::nullopt;
      args.push_back(*argType);
    }
    for (const typing::Term &argTerm : term.kwonly) {
      std::optional<mlir::Type> argType = typeFromTypeTerm(argTerm);
      if (!argType)
        return std::nullopt;
      kwonly.push_back(*argType);
    }
    if (!term.vararg.empty()) {
      std::optional<mlir::Type> argType = typeFromTypeTerm(term.vararg.front());
      if (!argType)
        return std::nullopt;
      vararg = *argType;
    }
    if (!term.kwarg.empty()) {
      std::optional<mlir::Type> argType = typeFromTypeTerm(term.kwarg.front());
      if (!argType)
        return std::nullopt;
      kwarg = *argType;
    }
    for (const typing::Term &resultTerm : term.results) {
      std::optional<mlir::Type> resultType = typeFromTypeTerm(resultTerm);
      if (!resultType)
        return std::nullopt;
      results.push_back(*resultType);
    }
    return py::CallableType::get(&context, args, kwonly, vararg, kwarg, results,
                                 {}, {}, {}, {}, {}, {},
                                 term.positionalOnlyCount.value_or(0));
  }

  if (term.name == "int")
    return intType();
  if (term.name == "float")
    return floatType();
  if (term.name == "bool")
    return boolType();
  if (term.name == "str")
    return strType();
  if (term.name == "None")
    return noneType();
  if (term.name == "object")
    return py::ObjectType::get(&context);
  if (term.name == "Exception")
    return exceptionType();
  if (term.name == "TracebackType")
    return py::TracebackType::get(&context);
  if (term.name == "type" && term.args.size() == 1) {
    std::optional<mlir::Type> instance = typeFromTypeTerm(term.args.front());
    if (!instance)
      return std::nullopt;
    return py::TypeType::get(&context, *instance);
  }
  if (llvm::StringRef(term.name).starts_with("class:"))
    return classType(llvm::StringRef(term.name).drop_front(strlen("class:")));
  if (llvm::StringRef(term.name).starts_with("protocol:")) {
    llvm::StringRef name =
        llvm::StringRef(term.name).drop_front(strlen("protocol:"));
    llvm::SmallVector<mlir::Type> args;
    args.reserve(term.args.size());
    for (const typing::Term &arg : term.args) {
      std::optional<mlir::Type> argType = typeFromTypeTerm(arg);
      if (!argType)
        return std::nullopt;
      args.push_back(*argType);
    }
    std::optional<py::ProtocolType> protocol = protocolType(name, args);
    if (!protocol)
      return std::nullopt;
    return *protocol;
  }
  if (term.name == "list" && term.args.size() == 1) {
    std::optional<mlir::Type> element = typeFromTypeTerm(term.args.front());
    if (!element)
      return std::nullopt;
    return listType(*element);
  }
  if (term.name == "dict" && term.args.size() == 2) {
    std::optional<mlir::Type> key = typeFromTypeTerm(term.args[0]);
    std::optional<mlir::Type> value = typeFromTypeTerm(term.args[1]);
    if (!key || !value)
      return std::nullopt;
    return dictType(*key, *value);
  }
  if (term.name == "tuple") {
    llvm::SmallVector<mlir::Type> elements;
    elements.reserve(term.args.size());
    for (const typing::Term &arg : term.args) {
      std::optional<mlir::Type> element = typeFromTypeTerm(arg);
      if (!element)
        return std::nullopt;
      elements.push_back(*element);
    }
    return py::TupleType::get(&context, elements);
  }
  if (term.name == "union" && term.args.size() >= 2) {
    llvm::SmallVector<mlir::Type> members;
    members.reserve(term.args.size());
    for (const typing::Term &arg : term.args) {
      std::optional<mlir::Type> member = typeFromTypeTerm(arg);
      if (!member)
        return std::nullopt;
      members.push_back(*member);
    }
    mlir::Type normalized = py::UnionType::getNormalized(&context, members);
    if (!normalized)
      return std::nullopt;
    return normalized;
  }
  if (term.name == "async.value" && term.args.size() == 1) {
    std::optional<mlir::Type> result = typeFromTypeTerm(term.args.front());
    if (!result)
      return std::nullopt;
    return mlir::async::ValueType::get(*result);
  }
  if (llvm::StringRef(term.name).starts_with("Int[") &&
      llvm::StringRef(term.name).ends_with("]")) {
    unsigned width = 0;
    llvm::StringRef text(term.name);
    if (!text.drop_front(4).drop_back().getAsInteger(10, width) && width > 0)
      return builder.getIntegerType(width);
  }
  if (llvm::StringRef(term.name).starts_with("Float[") &&
      llvm::StringRef(term.name).ends_with("]")) {
    unsigned width = 0;
    llvm::StringRef text(term.name);
    if (!text.drop_front(6).drop_back().getAsInteger(10, width)) {
      if (width == 16)
        return builder.getF16Type();
      if (width == 32)
        return builder.getF32Type();
      if (width == 64)
        return builder.getF64Type();
    }
  }
  if (term.name == "Tensor" && term.args.size() >= 2) {
    std::optional<mlir::Type> element = typeFromTypeTerm(term.args.front());
    if (!element)
      return std::nullopt;
    llvm::SmallVector<std::int64_t> shape;
    for (const typing::Term &dim :
         llvm::ArrayRef<typing::Term>(term.args).drop_front()) {
      std::int64_t value = 0;
      if (dim.kind != typing::Term::Kind::Con ||
          llvm::StringRef(dim.name).getAsInteger(10, value))
        return std::nullopt;
      shape.push_back(value);
    }
    return mlir::RankedTensorType::get(shape, *element);
  }
  return std::nullopt;
}

std::string typeString(mlir::Type type) {
  std::string storage;
  llvm::raw_string_ostream os(storage);
  type.print(os);
  return storage;
}

std::optional<py::ProtocolType>
Builder::Impl::protocolType(llvm::StringRef protocolName,
                            llvm::ArrayRef<mlir::Type> suppliedArguments) {
  std::optional<std::vector<mlir::Type>> completed =
      protocols::Table::get(context).completeProtocolArguments(
          protocolName, suppliedArguments);
  if (!completed)
    return std::nullopt;
  return py::ProtocolType::get(&context, protocolName, *completed);
}

py::IntType Builder::Impl::intType() { return py::IntType::get(&context); }
py::FloatType Builder::Impl::floatType() {
  return py::FloatType::get(&context);
}
py::BoolType Builder::Impl::boolType() { return py::BoolType::get(&context); }
py::StrType Builder::Impl::strType() { return py::StrType::get(&context); }
py::NoneType Builder::Impl::noneType() { return py::NoneType::get(&context); }
py::ExceptionType Builder::Impl::exceptionType() {
  return py::ExceptionType::get(&context);
}
py::ExceptionCellType Builder::Impl::exceptionCellType() {
  return py::ExceptionCellType::get(&context);
}
py::ClassType Builder::Impl::classType(llvm::StringRef className) {
  return py::ClassType::get(&context, className);
}
mlir::Type Builder::Impl::coroutineType(mlir::Type resultType) {
  if (std::optional<py::ProtocolType> protocol =
          protocolType("Coroutine", {resultType}))
    return *protocol;
  return py::ProtocolType::get(&context, "Coroutine",
                               {py::ObjectType::get(&context),
                                py::ObjectType::get(&context), resultType});
}
mlir::Type Builder::Impl::taskType(mlir::Type resultType) {
  if (std::optional<py::ProtocolType> protocol =
          protocolType("Task", {resultType}))
    return *protocol;
  return py::ProtocolType::get(&context, "Task", {resultType});
}
mlir::Type Builder::Impl::futureType(mlir::Type resultType) {
  if (std::optional<py::ProtocolType> protocol =
          protocolType("Future", {resultType}))
    return *protocol;
  return py::ProtocolType::get(&context, "Future", {resultType});
}
py::DictType Builder::Impl::dictType(mlir::Type keyType, mlir::Type valueType) {
  return py::DictType::get(&context, keyType, valueType);
}
py::ListType Builder::Impl::listType(mlir::Type elementType) {
  return py::ListType::get(&context, elementType);
}
mlir::Type Builder::Impl::i32Type() { return builder.getI32Type(); }
mlir::Type Builder::Impl::i1Type() { return builder.getI1Type(); }

bool Builder::Impl::typeInferenceDependsOnEnvironment(
    const parser::Node &node) {
  auto cached = typeInferenceDependencyCache.find(&node);
  if (cached != typeInferenceDependencyCache.end())
    return cached->second;
  bool depends = typeInferenceDependsOnEnvironmentUncached(node);
  typeInferenceDependencyCache.try_emplace(&node, depends);
  return depends;
}

std::optional<mlir::Type>
Builder::Impl::inferExpressionTypeAlgorithmM(const parser::Node &expr) {
  LambdaOracle oracle;
  typing::Oracle defaults;

  oracle.attributeFn =
      [&](const typing::Term &owner,
          llvm::StringRef name) -> std::optional<typing::Term> {
    if (owner.kind != typing::Term::Kind::Con)
      return std::nullopt;
    llvm::StringRef ownerName(owner.name);
    if (!ownerName.starts_with("class:"))
      return std::nullopt;
    auto cls = classes.find(ownerName.drop_front(strlen("class:")).str());
    if (cls == classes.end())
      return std::nullopt;
    auto field = cls->second.fields.find(name.str());
    if (field == cls->second.fields.end())
      return std::nullopt;
    return typeTermFromType(field->second);
  };

  oracle.callFn =
      [&](const typing::Term &callee,
          llvm::ArrayRef<typing::Term> args) -> std::optional<typing::Term> {
    if (callee.kind == typing::Term::Kind::Con) {
      if (callee.name == "builtin:bool" && args.size() <= 1)
        return typing::Term::con("bool");
      if (callee.name == "builtin:int" && args.size() <= 1)
        return typing::Term::con("int");
      if (callee.name == "builtin:float" && args.size() <= 1)
        return typing::Term::con("float");
      if ((callee.name == "builtin:str" && args.size() <= 1) ||
          (callee.name == "builtin:repr" && args.size() == 1))
        return typing::Term::con("str");
      if (callee.name == "builtin:len" && args.size() == 1)
        return typing::Term::con("int");
      if (llvm::StringRef(callee.name).starts_with("class-constructor:"))
        return typing::Term::con("class:" +
                                 llvm::StringRef(callee.name)
                                     .drop_front(strlen("class-constructor:"))
                                     .str());
    }
    return defaults.call(callee, args);
  };

  oracle.subscriptFn =
      [&](const typing::Term &container,
          const typing::Term &index) -> std::optional<typing::Term> {
    return defaults.subscript(container, index);
  };
  oracle.awaitFn =
      [&](const typing::Term &awaitable) -> std::optional<typing::Term> {
    return defaults.awaitable(awaitable);
  };

  typing::AlgorithmM inference(oracle);
  std::map<std::string, typing::Term> localTerms;

  auto termFromKnownType = [&](mlir::Type type) -> std::optional<typing::Term> {
    return typeTermFromType(type);
  };

  auto functionResultTerm =
      [&](const FunctionInfo &info) -> std::optional<typing::Term> {
    mlir::Type resultType =
        info.isAsync ? static_cast<mlir::Type>(coroutineType(info.resultType))
                     : info.resultType;
    return termFromKnownType(resultType);
  };

  std::function<std::optional<typing::Term>(const parser::Node &)> infer =
      [&](const parser::Node &node) -> std::optional<typing::Term> {
    if (node.kind == "Name") {
      const std::string *name = stringField(node, "id");
      if (!name)
        return std::nullopt;
      auto local = localTerms.find(*name);
      if (local != localTerms.end())
        return local->second;
      auto symbol = symbols.find(*name);
      if (symbol != symbols.end())
        return termFromKnownType(symbol->second.type);
      auto constant = primitiveConstants.find(*name);
      if (constant != primitiveConstants.end())
        return termFromKnownType(constant->second.type);
      auto localFunction = localFunctions.find(*name);
      if (localFunction != localFunctions.end())
        return termFromKnownType(localFunction->second.functionType);
      auto function = functions.find(*name);
      if (function != functions.end())
        return termFromKnownType(function->second.functionType);
      if (*name == "bool" || *name == "int" || *name == "float" ||
          *name == "str" || *name == "repr" || *name == "len")
        return typing::Term::con("builtin:" + *name);
      if (isBuiltinExceptionClass(*name))
        return typing::Term::con("type", {typing::Term::con("Exception")});
      if (classes.count(*name))
        return typing::Term::con("type", {typing::Term::con("class:" + *name)});
      return std::nullopt;
    }

    if (node.kind == "Constant") {
      const parser::FieldValue *value = valueField(node, "value");
      if (!value)
        return std::nullopt;
      if (std::holds_alternative<std::string>(*value))
        return typing::Term::con("str");
      if (std::holds_alternative<std::int64_t>(*value) ||
          std::holds_alternative<parser::BigInteger>(*value))
        return typing::Term::con("int");
      if (std::holds_alternative<double>(*value))
        return typing::Term::con("float");
      if (std::holds_alternative<bool>(*value))
        return typing::Term::con("bool");
      if (std::holds_alternative<std::monostate>(*value))
        return typing::Term::con("None");
      return std::nullopt;
    }

    if (node.kind == "JoinedStr" || node.kind == "FormattedValue" ||
        node.kind == "TemplateStr" || node.kind == "Interpolation")
      return typing::Term::con("str");

    if (node.kind == "NamedExpr") {
      const parser::NodePtr *value = nodeField(node, "value");
      if (!value || !*value)
        return std::nullopt;
      return infer(**value);
    }

    if (node.kind == "Attribute") {
      const parser::NodePtr *value = nodeField(node, "value");
      const std::string *name = stringField(node, "attr");
      if (!value || !*value || !name)
        return std::nullopt;
      std::optional<typing::Term> owner = infer(**value);
      if (!owner)
        return std::nullopt;
      typing::Term result = inference.fresh("attr");
      inference.requireAttribute(*owner, *name, result,
                                 {"attribute resolution"});
      return result;
    }

    if (node.kind == "Call") {
      if (std::optional<PrimitiveConstant> constant =
              primitiveScalarConstructorConstant(node))
        return termFromKnownType(constant->type);

      const parser::NodePtr *func = nodeField(node, "func");
      const std::vector<parser::NodePtr> *args = nodeListField(node, "args");
      if (!func || !*func || !args)
        return std::nullopt;

      if (std::optional<mlir::Type> targetType = typeFromAnnotation(*func);
          targetType &&
          mlir::isa<mlir::IntegerType, mlir::FloatType>(*targetType))
        return termFromKnownType(*targetType);

      if ((*func)->kind == "Name") {
        const std::string *name = stringField(**func, "id");
        if (!name)
          return std::nullopt;
        if (*name == "list" && args->size() == 1 && args->front()) {
          std::optional<typing::Term> source = infer(*args->front());
          if (!source || source->kind != typing::Term::Kind::Con)
            return std::nullopt;
          if (source->name == "list" && source->args.size() == 1)
            return *source;
          if (source->name == "tuple" && !source->args.empty()) {
            const typing::Term &element = source->args.front();
            for (const typing::Term &current : source->args)
              if (current != element)
                return std::nullopt;
            return typing::Term::con("list", {element});
          }
        }
        if (*name == "tuple") {
          if (args->empty())
            return typing::Term::con("tuple");
          if (args->size() == 1 && args->front()) {
            std::optional<typing::Term> source = infer(*args->front());
            if (!source || source->kind != typing::Term::Kind::Con)
              return std::nullopt;
            if (source->name == "tuple")
              return *source;
            if (source->name == "list" && source->args.size() == 1)
              return typing::Term::con("tuple", {source->args.front()});
          }
        }
        if (isBuiltinExceptionClass(*name))
          return typing::Term::con("Exception");
        auto cls = classes.find(*name);
        if (cls != classes.end()) {
          std::string className = *name;
          const std::vector<parser::NodePtr> *keywords =
              nodeListField(node, "keywords");
          auto initMethod = cls->second.methods.find("__init__");
          auto initInfo = initMethod == cls->second.methods.end()
                              ? functions.end()
                              : functions.find(initMethod->second);
          if ((!keywords || keywords->empty()) && initInfo != functions.end()) {
            llvm::SmallVector<mlir::Type> actualTypes;
            bool haveActualTypes = true;
            actualTypes.reserve(args->size());
            for (const parser::NodePtr &arg : *args) {
              if (!arg) {
                haveActualTypes = false;
                break;
              }
              std::optional<typing::Term> argTerm = infer(*arg);
              std::optional<mlir::Type> argType =
                  argTerm ? typeFromTypeTerm(*argTerm) : std::nullopt;
              if (!argType) {
                haveActualTypes = false;
                break;
              }
              actualTypes.push_back(*argType);
            }
            if (haveActualTypes) {
              if (std::optional<std::string> specialized =
                      specializeClassForConstructorFieldTypes(
                          node, *name, initInfo->second, actualTypes))
                className = *specialized;
            }
          }
          return typing::Term::con("class:" + className);
        }
        auto localFunction = localFunctions.find(*name);
        if (localFunction != localFunctions.end())
          return functionResultTerm(localFunction->second);
        auto function = functions.find(*name);
        if (function != functions.end())
          return functionResultTerm(function->second);
      }

      if ((*func)->kind == "Attribute") {
        const parser::NodePtr *value = nodeField(**func, "value");
        const std::string *methodName = stringField(**func, "attr");
        if (!value || !*value || !methodName)
          return std::nullopt;
        std::optional<typing::Term> owner = infer(**value);
        if (owner && owner->kind == typing::Term::Kind::Con) {
          llvm::StringRef ownerName(owner->name);
          if (ownerName.starts_with("class:")) {
            auto cls =
                classes.find(ownerName.drop_front(strlen("class:")).str());
            if (cls != classes.end()) {
              auto method = cls->second.methods.find(*methodName);
              if (method != cls->second.methods.end()) {
                auto info = callableFunctionsBySymbol.find(method->second);
                if (info != callableFunctionsBySymbol.end())
                  return functionResultTerm(info->second);
              }
            }
          }
        }
      }

      std::optional<typing::Term> callee = infer(**func);
      if (!callee)
        return std::nullopt;
      std::vector<typing::Term> argTerms;
      argTerms.reserve(args->size());
      for (const parser::NodePtr &arg : *args) {
        if (!arg)
          return std::nullopt;
        std::optional<typing::Term> argTerm = infer(*arg);
        if (!argTerm)
          return std::nullopt;
        argTerms.push_back(*argTerm);
      }
      typing::Term result = inference.fresh("call");
      inference.requireCall(*callee, argTerms, result, {"call resolution"});
      return result;
    }

    if (node.kind == "Subscript") {
      const parser::NodePtr *value = nodeField(node, "value");
      const parser::NodePtr *slice = nodeField(node, "slice");
      if (!value || !*value || !slice || !*slice)
        return std::nullopt;
      std::optional<typing::Term> container = infer(**value);
      if (!container)
        return std::nullopt;
      if (container->kind == typing::Term::Kind::Con &&
          container->name == "tuple") {
        std::optional<std::int64_t> index = staticIndexValue(**slice);
        if (!index || *index < 0)
          return std::nullopt;
        if (container->args.size() == 1)
          return container->args.front();
        if (*index >= static_cast<std::int64_t>(container->args.size()))
          return std::nullopt;
        return container->args[static_cast<std::size_t>(*index)];
      }
      std::optional<typing::Term> index = infer(**slice);
      if (!index)
        return std::nullopt;
      typing::Term result = inference.fresh("subscript");
      inference.requireSubscript(*container, *index, result,
                                 {"subscript resolution"});
      return result;
    }

    if (node.kind == "List") {
      const std::vector<parser::NodePtr> *elements =
          nodeListField(node, "elts");
      if (!elements || elements->empty())
        return std::nullopt;
      typing::Term element = inference.fresh("list.elem");
      for (const parser::NodePtr &item : *elements) {
        if (!item)
          return std::nullopt;
        std::optional<typing::Term> itemType = infer(*item);
        if (!itemType)
          return std::nullopt;
        inference.requireEqual(element, *itemType,
                               {"list element type consistency"});
      }
      return typing::Term::con("list", {element});
    }

    if (node.kind == "Tuple") {
      const std::vector<parser::NodePtr> *elements =
          nodeListField(node, "elts");
      if (!elements)
        return std::nullopt;
      std::vector<typing::Term> elementTerms;
      elementTerms.reserve(elements->size());
      for (const parser::NodePtr &item : *elements) {
        if (!item)
          return std::nullopt;
        std::optional<typing::Term> itemType = infer(*item);
        if (!itemType)
          return std::nullopt;
        elementTerms.push_back(*itemType);
      }
      return typing::Term::con("tuple", elementTerms);
    }

    if (node.kind == "Dict") {
      const std::vector<parser::NodePtr> *keys = nodeListField(node, "keys");
      const std::vector<parser::NodePtr> *values =
          nodeListField(node, "values");
      if (!keys || !values || keys->empty() || keys->size() != values->size())
        return std::nullopt;
      typing::Term key = inference.fresh("dict.key");
      typing::Term value = inference.fresh("dict.value");
      for (std::size_t i = 0; i < keys->size(); ++i) {
        if (!(*keys)[i] || !(*values)[i])
          return std::nullopt;
        std::optional<typing::Term> keyType = infer(*(*keys)[i]);
        std::optional<typing::Term> valueType = infer(*(*values)[i]);
        if (!keyType || !valueType)
          return std::nullopt;
        inference.requireEqual(key, *keyType, {"dict key type consistency"});
        inference.requireEqual(value, *valueType,
                               {"dict value type consistency"});
      }
      return typing::Term::con("dict", {key, value});
    }

    if (node.kind == "ListComp" || node.kind == "DictComp") {
      const std::vector<parser::NodePtr> *generators =
          nodeListField(node, "generators");
      if (!generators || generators->size() != 1 || !generators->front())
        return std::nullopt;
      const parser::Node &generator = *generators->front();
      const parser::NodePtr *target = nodeField(generator, "target");
      const parser::NodePtr *iter = nodeField(generator, "iter");
      const parser::FieldValue *isAsyncValue =
          valueField(generator, "is_async");
      const auto *isAsync =
          isAsyncValue ? std::get_if<std::int64_t>(isAsyncValue) : nullptr;
      if (!target || !*target || (*target)->kind != "Name" || !iter || !*iter ||
          (isAsync && *isAsync != 0))
        return std::nullopt;
      const std::string *targetName = stringField(**target, "id");
      std::optional<mlir::Type> targetType = inferRangeTargetType(**iter);
      std::optional<typing::Term> targetTerm =
          targetType ? termFromKnownType(*targetType) : std::nullopt;
      if (!targetName || !targetTerm)
        return std::nullopt;

      std::optional<typing::Term> saved;
      auto found = localTerms.find(*targetName);
      if (found != localTerms.end())
        saved = found->second;
      localTerms[*targetName] = *targetTerm;

      std::optional<typing::Term> result;
      if (node.kind == "ListComp") {
        const parser::NodePtr *element = nodeField(node, "elt");
        if (element && *element) {
          std::optional<typing::Term> elementType = infer(**element);
          if (elementType)
            result = typing::Term::con("list", {*elementType});
        }
      } else {
        const parser::NodePtr *keyNode = nodeField(node, "key");
        const parser::NodePtr *valueNode = nodeField(node, "value");
        if (keyNode && *keyNode && valueNode && *valueNode) {
          std::optional<typing::Term> keyType = infer(**keyNode);
          std::optional<typing::Term> valueType = infer(**valueNode);
          if (keyType && valueType)
            result = typing::Term::con("dict", {*keyType, *valueType});
        }
      }

      if (saved)
        localTerms[*targetName] = *saved;
      else
        localTerms.erase(*targetName);
      return result;
    }

    if (node.kind == "BinOp") {
      const parser::NodePtr *lhs = nodeField(node, "left");
      const parser::NodePtr *rhs = nodeField(node, "right");
      std::optional<std::string> op = symbolField(node, "op");
      if (!lhs || !*lhs || !rhs || !*rhs || !op)
        return std::nullopt;
      if (*op == "@" || *op == "*")
        return std::nullopt;
      std::optional<typing::Term> lhsType = infer(**lhs);
      std::optional<typing::Term> rhsType = infer(**rhs);
      if (!lhsType || !rhsType)
        return std::nullopt;
      inference.requireEqual(*lhsType, *rhsType,
                             {"binary operand type consistency"});
      return *lhsType;
    }

    if (node.kind == "UnaryOp") {
      std::optional<std::string> op = symbolField(node, "op");
      const parser::NodePtr *operand = nodeField(node, "operand");
      if (!op || !operand || !*operand)
        return std::nullopt;
      std::optional<typing::Term> operandType = infer(**operand);
      if (!operandType)
        return std::nullopt;
      if (*op == "not")
        return typing::Term::con("bool");
      if (*op == "+" || *op == "-" || *op == "~")
        return *operandType;
      return std::nullopt;
    }

    if (node.kind == "IfExp") {
      const parser::NodePtr *body = nodeField(node, "body");
      const parser::NodePtr *orelse = nodeField(node, "orelse");
      if (!body || !*body || !orelse || !*orelse)
        return std::nullopt;
      std::optional<typing::Term> bodyType = infer(**body);
      std::optional<typing::Term> elseType = infer(**orelse);
      if (!bodyType || !elseType)
        return std::nullopt;
      inference.requireEqual(*bodyType, *elseType,
                             {"conditional expression type consistency"});
      return *bodyType;
    }

    if (node.kind == "Await") {
      const parser::NodePtr *value = nodeField(node, "value");
      if (!value || !*value)
        return std::nullopt;
      std::optional<typing::Term> awaitable = infer(**value);
      if (!awaitable)
        return std::nullopt;
      typing::Term result = inference.fresh("await");
      inference.requireAwait(*awaitable, result, {"await resolution"});
      return result;
    }

    return std::nullopt;
  };

  try {
    std::optional<typing::Term> term = infer(expr);
    if (!term)
      return std::nullopt;
    typing::Term resolved = inference.resolve(*term);
    return typeFromTypeTerm(resolved);
  } catch (const typing::Error &) {
    return std::nullopt;
  }
}

std::optional<mlir::Type>
Builder::Impl::inferExpressionType(const parser::Node &expr) {
  auto cached = stableTypeInferenceCache.find(&expr);
  if (cached != stableTypeInferenceCache.end())
    return cached->second;

  if (typeInferenceDependsOnEnvironment(expr))
    return inferExpressionTypeUncached(expr);

  std::optional<mlir::Type> inferred = inferExpressionTypeUncached(expr);
  if (inferred)
    stableTypeInferenceCache.try_emplace(&expr, *inferred);
  return inferred;
}

std::optional<mlir::Type>
Builder::Impl::inferExpressionTypeUncached(const parser::Node &expr) {
  if (std::optional<mlir::Type> inferred = inferExpressionTypeAlgorithmM(expr))
    return inferred;

  if (expr.kind == "Name") {
    const std::string *name = stringField(expr, "id");
    if (!name)
      return std::nullopt;
    auto symbol = symbols.find(*name);
    if (symbol != symbols.end())
      return symbol->second.type;
    auto constant = primitiveConstants.find(*name);
    if (constant != primitiveConstants.end())
      return constant->second.type;
    auto localFunction = localFunctions.find(*name);
    if (localFunction != localFunctions.end())
      return localFunction->second.functionType;
    auto function = functions.find(*name);
    if (function != functions.end())
      return function->second.functionType;
    if (isBuiltinExceptionClass(*name))
      return py::TypeType::get(&context, exceptionType());
    if (classes.count(*name))
      return py::TypeType::get(&context, classType(*name));
    return std::nullopt;
  }

  if (expr.kind == "Constant") {
    const parser::FieldValue *value = valueField(expr, "value");
    if (!value)
      return std::nullopt;
    if (std::holds_alternative<std::string>(*value))
      return strType();
    if (std::holds_alternative<std::int64_t>(*value) ||
        std::holds_alternative<parser::BigInteger>(*value))
      return intType();
    if (std::holds_alternative<double>(*value))
      return floatType();
    if (std::holds_alternative<bool>(*value))
      return boolType();
    if (std::holds_alternative<std::monostate>(*value))
      return noneType();
    return std::nullopt;
  }

  if (expr.kind == "JoinedStr" || expr.kind == "FormattedValue" ||
      expr.kind == "TemplateStr" || expr.kind == "Interpolation")
    return strType();

  if (expr.kind == "NamedExpr") {
    const parser::NodePtr *value = nodeField(expr, "value");
    if (!value || !*value)
      return std::nullopt;
    return inferExpressionType(**value);
  }

  if (expr.kind == "Call") {
    if (std::optional<PrimitiveConstant> constant =
            primitiveScalarConstructorConstant(expr))
      return constant->type;
    const parser::NodePtr *func = nodeField(expr, "func");
    const std::vector<parser::NodePtr> *args = nodeListField(expr, "args");
    const std::vector<parser::NodePtr> *keywords =
        nodeListField(expr, "keywords");
    if (!func || !*func)
      return std::nullopt;
    auto inferArgumentTypes =
        [&]() -> std::optional<llvm::SmallVector<mlir::Type>> {
      if (!args)
        return std::nullopt;
      llvm::SmallVector<mlir::Type> argumentTypes;
      argumentTypes.reserve(args->size());
      for (const parser::NodePtr &arg : *args) {
        if (!arg)
          return std::nullopt;
        std::optional<mlir::Type> argType = inferExpressionType(*arg);
        if (!argType)
          return std::nullopt;
        argumentTypes.push_back(*argType);
      }
      return argumentTypes;
    };
    if (std::optional<mlir::Type> targetType = typeFromAnnotation(*func);
        targetType &&
        mlir::isa<mlir::IntegerType, mlir::FloatType>(*targetType))
      return *targetType;
    if (std::optional<mlir::Type> targetType = typeFromAnnotation(*func);
        targetType && mlir::isa<py::ClassType>(*targetType))
      return *targetType;
    if (isTensorConstructorCallee(**func))
      return typeFromAnnotation(*func);
    if (std::optional<std::string> lyrtName = lyrtBuiltinName(**func);
        lyrtName && *lyrtName == "from_prim" && args && args->size() == 1 &&
        args->front()) {
      std::optional<mlir::Type> inputType = inferExpressionType(*args->front());
      if (inputType && mlir::isa<mlir::RankedTensorType>(*inputType))
        return strType();
      if (inputType && mlir::isa<mlir::IntegerType>(*inputType))
        return intType();
      if (inputType && mlir::isa<mlir::FloatType>(*inputType))
        return floatType();
    }
    if (std::optional<std::string> lyrtName = lyrtBuiltinName(**func);
        lyrtName && *lyrtName == "to_prim" && args && args->size() == 2 &&
        (*args)[1]) {
      std::optional<mlir::Type> primitiveType = typeFromAnnotation((*args)[1]);
      if (primitiveType && !mlir::isa<mlir::RankedTensorType>(*primitiveType))
        return *primitiveType;
    }
    if ((*func)->kind == "Name") {
      const std::string *name = stringField(**func, "id");
      if (!name)
        return std::nullopt;
      if (*name == "bool" && args && args->size() <= 1)
        return boolType();
      if (*name == "int" && args && args->size() <= 1)
        return intType();
      if (*name == "float" && args && args->size() <= 1)
        return floatType();
      if ((*name == "str" && args && args->size() <= 1) ||
          (*name == "repr" && args && args->size() == 1))
        return strType();
      if (*name == "len" && args && args->size() == 1)
        return intType();
      if (*name == "list" && args && args->size() == 1 && args->front() &&
          (!keywords || keywords->empty())) {
        if (isRangeCall(*args->front())) {
          if (std::optional<mlir::Type> targetType =
                  inferRangeTargetType(*args->front()))
            return listType(*targetType);
          return std::nullopt;
        }
        std::optional<mlir::Type> sourceType =
            inferExpressionType(*args->front());
        if (!sourceType)
          return std::nullopt;
        if (mlir::Type elementType = listElementType(*sourceType))
          return listType(elementType);
        if (auto tupleType = mlir::dyn_cast<py::TupleType>(*sourceType)) {
          llvm::ArrayRef<mlir::Type> elementTypes = tupleType.getElementTypes();
          if (elementTypes.empty())
            return std::nullopt;
          mlir::Type elementType = elementTypes.front();
          for (mlir::Type current : elementTypes)
            if (current != elementType)
              return std::nullopt;
          return listType(elementType);
        }
      }
      if (*name == "tuple" && args && (!keywords || keywords->empty())) {
        if (args->empty())
          return py::TupleType::get(&context, {});
        if (args->size() == 1 && args->front()) {
          if (isRangeCall(*args->front())) {
            if (std::optional<mlir::Type> targetType =
                    inferRangeTargetType(*args->front()))
              return py::TupleType::get(&context, {*targetType});
            return std::nullopt;
          }
          std::optional<mlir::Type> sourceType =
              inferExpressionType(*args->front());
          if (!sourceType)
            return std::nullopt;
          if (auto tupleType = mlir::dyn_cast<py::TupleType>(*sourceType))
            return tupleType;
          if (mlir::Type elementType = listElementType(*sourceType))
            return py::TupleType::get(&context, {elementType});
        }
      }
      if (isBuiltinExceptionClass(*name))
        return exceptionType();
      auto cls = classes.find(*name);
      if (cls != classes.end() && !cls->second.isGenericTemplate)
        return classType(*name);
      auto localFunction = localFunctions.find(*name);
      if (localFunction != localFunctions.end())
        return localFunction->second.resultType;
      auto function = functions.find(*name);
      if (function != functions.end()) {
        if (function->second.isAsync)
          return coroutineType(function->second.resultType);
        return function->second.resultType;
      }
    }
    if ((*func)->kind == "Attribute" && (!keywords || keywords->empty())) {
      const parser::NodePtr *receiver = nodeField(**func, "value");
      const std::string *methodName = stringField(**func, "attr");
      if (!receiver || !*receiver || !methodName)
        return std::nullopt;
      std::optional<mlir::Type> receiverType = inferExpressionType(**receiver);
      std::optional<llvm::SmallVector<mlir::Type>> argumentTypes =
          inferArgumentTypes();
      if (!receiverType || !argumentTypes)
        return std::nullopt;
      std::optional<mlir::Type> resultType =
          protocols::Table::get(context).resolveMethodResultOn(
              *receiverType, *methodName, *argumentTypes);
      if (resultType)
        return *resultType;

      std::optional<std::string> className = classNameFromType(*receiverType);
      if (!className)
        return std::nullopt;
      auto classFound = classes.find(*className);
      if (classFound == classes.end())
        return std::nullopt;
      auto methodFound = classFound->second.methods.find(*methodName);
      if (methodFound == classFound->second.methods.end())
        return std::nullopt;
      auto functionFound = functions.find(methodFound->second);
      if (functionFound == functions.end())
        return std::nullopt;
      return functionFound->second.resultType;
    }
    return std::nullopt;
  }

  if (expr.kind == "Attribute") {
    const parser::NodePtr *value = nodeField(expr, "value");
    const std::string *name = stringField(expr, "attr");
    if (!value || !*value || !name)
      return std::nullopt;
    std::optional<mlir::Type> objectType = inferExpressionType(**value);
    std::optional<std::string> className =
        objectType ? classNameFromType(*objectType) : std::nullopt;
    if (!className)
      return std::nullopt;
    auto cls = classes.find(*className);
    if (cls == classes.end())
      return std::nullopt;
    auto field = cls->second.fields.find(*name);
    if (field == cls->second.fields.end())
      return std::nullopt;
    return field->second;
  }

  if (expr.kind == "Subscript") {
    const parser::NodePtr *value = nodeField(expr, "value");
    if (!value || !*value)
      return std::nullopt;
    std::optional<mlir::Type> containerType = inferExpressionType(**value);
    if (!containerType)
      return std::nullopt;
    const parser::NodePtr *slice = nodeField(expr, "slice");
    if (!slice || !*slice)
      return std::nullopt;
    if ((*slice)->kind == "Slice" && !mlir::isa<py::TupleType>(*containerType))
      return std::nullopt;
    if (mlir::Type elementType = listElementType(*containerType))
      return elementType;
    if (auto tupleType = mlir::dyn_cast<py::TupleType>(*containerType)) {
      auto elementTypes = tupleType.getElementTypes();
      if ((*slice)->kind == "Slice") {
        if (elementTypes.size() == 1)
          return std::nullopt;
        const parser::NodePtr *lowerNode = nodeField(**slice, "lower");
        const parser::NodePtr *upperNode = nodeField(**slice, "upper");
        const parser::NodePtr *stepNode = nodeField(**slice, "step");
        std::optional<std::int64_t> lower = lowerNode && *lowerNode
                                                ? staticIndexValue(**lowerNode)
                                                : std::optional<std::int64_t>{};
        std::optional<std::int64_t> upper = upperNode && *upperNode
                                                ? staticIndexValue(**upperNode)
                                                : std::optional<std::int64_t>{};
        std::optional<std::int64_t> step = stepNode && *stepNode
                                               ? staticIndexValue(**stepNode)
                                               : std::optional<std::int64_t>{1};
        if ((lowerNode && *lowerNode && !lower) ||
            (upperNode && *upperNode && !upper) ||
            (stepNode && *stepNode && !step) || *step == 0)
          return std::nullopt;

        const std::int64_t length =
            static_cast<std::int64_t>(elementTypes.size());
        std::int64_t start = 0;
        std::int64_t stop = length;
        if (*step < 0) {
          start = length - 1;
          stop = -1;
        }
        if (lower)
          start = *step > 0 ? clampSliceIndex(*lower, length, 0, length)
                            : clampSliceIndex(*lower, length, -1, length - 1);
        if (upper)
          stop = *step > 0 ? clampSliceIndex(*upper, length, 0, length)
                           : clampSliceIndex(*upper, length, -1, length - 1);

        llvm::SmallVector<mlir::Type> resultTypes;
        for (std::int64_t i = start; *step > 0 ? i < stop : i > stop;
             i += *step)
          resultTypes.push_back(elementTypes[static_cast<std::size_t>(i)]);
        return py::TupleType::get(&context, resultTypes);
      }
      std::optional<std::int64_t> index = staticIndexValue(**slice);
      if (!index || *index < 0)
        return std::nullopt;
      if (elementTypes.empty())
        return std::nullopt;
      if (elementTypes.size() == 1)
        return elementTypes.front();
      if (*index >= static_cast<std::int64_t>(elementTypes.size()))
        return std::nullopt;
      return elementTypes[*index];
    }
    std::optional<mlir::Type> indexType = inferExpressionType(**slice);
    if (!indexType)
      return std::nullopt;
    std::optional<mlir::Type> resultType =
        protocols::Table::get(context).resolveMethodResultOn(
            *containerType, "__getitem__", {*indexType});
    if (resultType)
      return *resultType;
    std::optional<std::string> className = classNameFromType(*containerType);
    if (!className)
      return std::nullopt;
    auto classFound = classes.find(*className);
    if (classFound == classes.end())
      return std::nullopt;
    auto methodFound = classFound->second.methods.find("__getitem__");
    if (methodFound == classFound->second.methods.end())
      return std::nullopt;
    auto functionFound = functions.find(methodFound->second);
    if (functionFound == functions.end())
      return std::nullopt;
    const FunctionInfo &method = functionFound->second;
    if (method.argTypes.size() != 2 ||
        !typeAssignable(method.argTypes[1], *indexType))
      return std::nullopt;
    return method.resultType;
  }

  if (expr.kind == "List") {
    const std::vector<parser::NodePtr> *elements = nodeListField(expr, "elts");
    if (!elements || elements->empty() || !elements->front())
      return std::nullopt;
    std::optional<mlir::Type> elementType =
        inferExpressionType(*elements->front());
    if (!elementType)
      return std::nullopt;
    for (const parser::NodePtr &element :
         llvm::ArrayRef<parser::NodePtr>(*elements).drop_front()) {
      if (!element)
        return std::nullopt;
      std::optional<mlir::Type> currentType = inferExpressionType(*element);
      if (!currentType || *currentType != *elementType)
        return std::nullopt;
    }
    return listType(*elementType);
  }

  if (expr.kind == "ListComp") {
    const parser::NodePtr *element = nodeField(expr, "elt");
    const std::vector<parser::NodePtr> *generators =
        nodeListField(expr, "generators");
    if (!element || !*element || !generators || generators->size() != 1 ||
        !generators->front())
      return std::nullopt;
    const parser::Node &generator = *generators->front();
    const parser::NodePtr *target = nodeField(generator, "target");
    const parser::NodePtr *iter = nodeField(generator, "iter");
    const std::vector<parser::NodePtr> *ifs = nodeListField(generator, "ifs");
    const parser::FieldValue *isAsyncValue = valueField(generator, "is_async");
    const auto *isAsync =
        isAsyncValue ? std::get_if<std::int64_t>(isAsyncValue) : nullptr;
    if (!target || !*target || (*target)->kind != "Name" || !iter || !*iter ||
        (ifs && !ifs->empty()) || (isAsync && *isAsync != 0))
      return std::nullopt;
    const std::string *targetName = stringField(**target, "id");
    std::optional<mlir::Type> targetType = inferRangeTargetType(**iter);
    if (!targetName || !targetType)
      return std::nullopt;

    NameBindingSnapshot saved = snapshotNameBinding(*targetName);
    bindTemporaryName(*targetName, Value{{}, *targetType});
    std::optional<mlir::Type> elementType = inferExpressionType(**element);
    restoreNameBinding(*targetName, std::move(saved));
    if (!elementType)
      return std::nullopt;
    return listType(*elementType);
  }

  if (expr.kind == "DictComp") {
    const parser::NodePtr *key = nodeField(expr, "key");
    const parser::NodePtr *value = nodeField(expr, "value");
    const std::vector<parser::NodePtr> *generators =
        nodeListField(expr, "generators");
    if (!key || !*key || !value || !*value || !generators ||
        generators->size() != 1 || !generators->front())
      return std::nullopt;
    const parser::Node &generator = *generators->front();
    const parser::NodePtr *target = nodeField(generator, "target");
    const parser::NodePtr *iter = nodeField(generator, "iter");
    const parser::FieldValue *isAsyncValue = valueField(generator, "is_async");
    const auto *isAsync =
        isAsyncValue ? std::get_if<std::int64_t>(isAsyncValue) : nullptr;
    if (!target || !*target || (*target)->kind != "Name" || !iter || !*iter ||
        (isAsync && *isAsync != 0))
      return std::nullopt;
    const std::string *targetName = stringField(**target, "id");
    std::optional<mlir::Type> targetType = inferRangeTargetType(**iter);
    if (!targetName || !targetType)
      return std::nullopt;

    NameBindingSnapshot saved = snapshotNameBinding(*targetName);
    bindTemporaryName(*targetName, Value{{}, *targetType});
    std::optional<mlir::Type> keyType = inferExpressionType(**key);
    std::optional<mlir::Type> valueType = inferExpressionType(**value);
    restoreNameBinding(*targetName, std::move(saved));
    if (!keyType || !valueType || !dictStorageSupported(*keyType, *valueType))
      return std::nullopt;
    return dictType(*keyType, *valueType);
  }

  if (expr.kind == "Tuple") {
    const std::vector<parser::NodePtr> *elements = nodeListField(expr, "elts");
    if (!elements)
      return std::nullopt;
    llvm::SmallVector<mlir::Type> elementTypes;
    elementTypes.reserve(elements->size());
    for (const parser::NodePtr &element : *elements) {
      if (!element)
        return std::nullopt;
      std::optional<mlir::Type> elementType = inferExpressionType(*element);
      if (!elementType)
        return std::nullopt;
      elementTypes.push_back(*elementType);
    }
    return py::TupleType::get(&context, elementTypes);
  }

  if (expr.kind == "Dict") {
    const std::vector<parser::NodePtr> *keys = nodeListField(expr, "keys");
    const std::vector<parser::NodePtr> *values = nodeListField(expr, "values");
    if (!keys || !values || keys->empty() || keys->size() != values->size() ||
        !keys->front() || !values->front())
      return std::nullopt;
    std::optional<mlir::Type> keyType = inferExpressionType(*keys->front());
    std::optional<mlir::Type> valueType = inferExpressionType(*values->front());
    if (!keyType || !valueType)
      return std::nullopt;
    for (std::size_t i = 1; i < keys->size(); ++i) {
      if (!(*keys)[i] || !(*values)[i])
        return std::nullopt;
      std::optional<mlir::Type> currentKey = inferExpressionType(*(*keys)[i]);
      std::optional<mlir::Type> currentValue =
          inferExpressionType(*(*values)[i]);
      if (!currentKey || !currentValue || *currentKey != *keyType ||
          *currentValue != *valueType)
        return std::nullopt;
    }
    return dictType(*keyType, *valueType);
  }

  if (expr.kind == "BinOp") {
    const parser::NodePtr *lhs = nodeField(expr, "left");
    const parser::NodePtr *rhs = nodeField(expr, "right");
    std::optional<std::string> op = symbolField(expr, "op");
    if (!lhs || !*lhs || !rhs || !*rhs || !op)
      return std::nullopt;
    std::optional<mlir::Type> lhsType = inferExpressionType(**lhs);
    std::optional<mlir::Type> rhsType = inferExpressionType(**rhs);
    if (!lhsType || !rhsType)
      return std::nullopt;
    if (*op == "@") {
      auto lhsTensor = mlir::dyn_cast<mlir::RankedTensorType>(*lhsType);
      auto rhsTensor = mlir::dyn_cast<mlir::RankedTensorType>(*rhsType);
      if (!lhsTensor || !rhsTensor || lhsTensor.getRank() != 2 ||
          rhsTensor.getRank() != 2 ||
          lhsTensor.getElementType() != rhsTensor.getElementType() ||
          lhsTensor.getDimSize(1) != rhsTensor.getDimSize(0))
        return std::nullopt;
      return mlir::RankedTensorType::get(
          {lhsTensor.getDimSize(0), rhsTensor.getDimSize(1)},
          lhsTensor.getElementType());
    }
    if (*op == "*") {
      auto repeatTuple = [&](py::TupleType tupleType,
                             std::int64_t count) -> mlir::Type {
        llvm::ArrayRef<mlir::Type> elementTypes = tupleType.getElementTypes();
        if (count <= 0 || elementTypes.empty())
          return py::TupleType::get(&context, {});
        llvm::SmallVector<mlir::Type> repeated;
        repeated.reserve(elementTypes.size() * static_cast<std::size_t>(count));
        for (std::int64_t iteration = 0; iteration < count; ++iteration)
          repeated.append(elementTypes.begin(), elementTypes.end());
        return py::TupleType::get(&context, repeated);
      };
      if (auto lhsTuple = mlir::dyn_cast<py::TupleType>(*lhsType)) {
        if (std::optional<std::int64_t> count = staticIndexValue(**rhs))
          return repeatTuple(lhsTuple, *count);
      }
      if (auto rhsTuple = mlir::dyn_cast<py::TupleType>(*rhsType)) {
        if (std::optional<std::int64_t> count = staticIndexValue(**lhs))
          return repeatTuple(rhsTuple, *count);
      }
      if (listElementType(*lhsType) && staticIndexValue(**rhs))
        return *lhsType;
      if (listElementType(*rhsType) && staticIndexValue(**lhs))
        return *rhsType;
    }
    if (*lhsType == *rhsType)
      return *lhsType;
    if (mlir::isa<mlir::IntegerType>(*lhsType) && staticIndexValue(**rhs))
      return *lhsType;
    if (mlir::isa<mlir::IntegerType>(*rhsType) && staticIndexValue(**lhs))
      return *rhsType;
    return std::nullopt;
  }

  if (expr.kind == "Compare") {
    const parser::NodePtr *lhs = nodeField(expr, "left");
    std::optional<std::vector<std::string>> ops = symbolListField(expr, "ops");
    const std::vector<parser::NodePtr> *comparators =
        nodeListField(expr, "comparators");
    if (!lhs || !*lhs || !ops || !comparators || comparators->empty() ||
        ops->size() != comparators->size())
      return std::nullopt;
    if (ops->size() == 1 && ((*ops)[0] == "is" || (*ops)[0] == "is not")) {
      if (!(*comparators)[0])
        return std::nullopt;
      std::optional<int> lhsKey = singletonKey(**lhs);
      std::optional<int> rhsKey = singletonKey(*(*comparators)[0]);
      if (lhsKey && rhsKey)
        return boolType();
      if (!lhsKey && !rhsKey)
        return std::nullopt;
      const parser::Node &valueNode = lhsKey ? *(*comparators)[0] : **lhs;
      std::optional<mlir::Type> valueType = inferExpressionType(valueNode);
      if (!valueType)
        return std::nullopt;
      if (*valueType == boolType() || *valueType == noneType() ||
          valueNode.kind == "Name" || valueNode.kind == "Constant")
        return boolType();
      return std::nullopt;
    }
    if (ops->size() == 1 && ((*ops)[0] == "in" || (*ops)[0] == "not in") &&
        (*comparators)[0] && (*comparators)[0]->kind == "Tuple") {
      std::optional<mlir::Type> lhsType = inferExpressionType(**lhs);
      if (!lhsType || !*lhsType ||
          (!mlir::isa<mlir::IntegerType, mlir::FloatType>(*lhsType) &&
           *lhsType != intType() && *lhsType != floatType()))
        return std::nullopt;
      const std::vector<parser::NodePtr> *elements =
          nodeListField(*(*comparators)[0], "elts");
      if (!elements)
        return std::nullopt;
      for (const parser::NodePtr &element : *elements) {
        if (!element)
          return std::nullopt;
        std::optional<mlir::Type> elementType = inferExpressionType(*element);
        if (!elementType || !*elementType || *elementType != *lhsType)
          return std::nullopt;
      }
      return i1Type();
    }
    if (ops->size() == 1 && ((*ops)[0] == "in" || (*ops)[0] == "not in") &&
        (*comparators)[0]) {
      std::optional<mlir::Type> itemType = inferExpressionType(**lhs);
      std::optional<mlir::Type> containerType =
          inferExpressionType(*(*comparators)[0]);
      if (!itemType || !containerType)
        return std::nullopt;
      std::optional<mlir::Type> resultType =
          protocols::Table::get(context).resolveMethodResultOn(
              *containerType, "__contains__", {*itemType});
      if (resultType && *resultType == boolType())
        return i1Type();
      return std::nullopt;
    }
    std::optional<mlir::Type> lhsType = inferExpressionType(**lhs);
    if (!lhsType || !*lhsType)
      return std::nullopt;
    bool primitiveChain =
        mlir::isa<mlir::IntegerType, mlir::FloatType>(*lhsType);
    mlir::Type previousType = *lhsType;
    for (const parser::NodePtr &comparator : *comparators) {
      if (!comparator)
        return std::nullopt;
      std::optional<mlir::Type> rhsType = inferExpressionType(*comparator);
      if (!rhsType || !*rhsType)
        return std::nullopt;
      if (*rhsType != previousType) {
        if (mlir::isa<mlir::IntegerType>(previousType) &&
            staticIndexValue(*comparator))
          rhsType = previousType;
        else
          return std::nullopt;
      }
      if (!mlir::isa<mlir::IntegerType, mlir::FloatType>(*rhsType))
        primitiveChain = false;
      previousType = *rhsType;
    }
    if (primitiveChain)
      return i1Type();
    if (comparators->size() == 1)
      return boolType();
    return std::nullopt;
  }

  if (expr.kind == "UnaryOp") {
    std::optional<std::string> op = symbolField(expr, "op");
    const parser::NodePtr *operand = nodeField(expr, "operand");
    if (!op || !operand || !*operand)
      return std::nullopt;
    std::optional<mlir::Type> operandType = inferExpressionType(**operand);
    if (!operandType)
      return std::nullopt;
    if (*op == "not") {
      if (*operandType == boolType())
        return boolType();
      if (*operandType == intType() || *operandType == floatType() ||
          *operandType == strType())
        return boolType();
      if (auto tupleType = mlir::dyn_cast<py::TupleType>(*operandType)) {
        if (tupleType.getElementTypes().size() != 1 ||
            (*operand)->kind == "Tuple")
          return boolType();
      }
      if (mlir::isa<mlir::IntegerType, mlir::FloatType>(*operandType))
        return i1Type();
      return std::nullopt;
    }
    if (*op == "+" || *op == "-")
      return *operandType;
    if (*op == "~" && (mlir::isa<mlir::IntegerType>(*operandType) ||
                       *operandType == intType()))
      return *operandType;
    return std::nullopt;
  }

  if (expr.kind == "BoolOp") {
    const std::vector<parser::NodePtr> *values = nodeListField(expr, "values");
    if (!values || values->empty())
      return std::nullopt;
    bool allI1 = true;
    for (const parser::NodePtr &value : *values) {
      if (!value)
        return std::nullopt;
      std::optional<mlir::Type> valueType = inferExpressionType(*value);
      if (!valueType)
        return std::nullopt;
      if (*valueType != i1Type())
        allI1 = false;
    }
    return allI1 ? i1Type() : boolType();
  }

  if (expr.kind == "IfExp") {
    const parser::NodePtr *body = nodeField(expr, "body");
    const parser::NodePtr *orelse = nodeField(expr, "orelse");
    if (!body || !*body || !orelse || !*orelse)
      return std::nullopt;
    std::optional<mlir::Type> bodyType = inferExpressionType(**body);
    std::optional<mlir::Type> elseType = inferExpressionType(**orelse);
    if (!bodyType || !elseType || *bodyType != *elseType)
      return std::nullopt;
    return *bodyType;
  }

  if (expr.kind == "Await") {
    const parser::NodePtr *value = nodeField(expr, "value");
    if (!value || !*value)
      return std::nullopt;
    std::optional<mlir::Type> awaitableType = inferExpressionType(**value);
    if (!awaitableType)
      return std::nullopt;
    if (mlir::Type payloadType = awaitablePayloadType(*awaitableType))
      return payloadType;
  }

  return std::nullopt;
}

std::optional<mlir::Type>
Builder::Impl::inferRangeTargetType(const parser::Node &iter) {
  if (iter.kind != "Call")
    return std::nullopt;
  const parser::NodePtr *func = nodeField(iter, "func");
  const std::vector<parser::NodePtr> *args = nodeListField(iter, "args");
  const std::vector<parser::NodePtr> *keywords =
      nodeListField(iter, "keywords");
  if (!func || !*func || (*func)->kind != "Name" || !args || args->empty() ||
      args->size() > 3 || (keywords && !keywords->empty()))
    return std::nullopt;
  const std::string *callee = stringField(**func, "id");
  if (!callee || *callee != "range")
    return std::nullopt;

  std::optional<mlir::Type> rangeType;
  bool sawStaticIntegerLiteral = false;
  for (const parser::NodePtr &arg : *args) {
    if (!arg)
      return std::nullopt;
    std::optional<mlir::Type> argType;
    if (std::optional<PrimitiveConstant> constant =
            primitiveIntConstructorConstant(*arg)) {
      argType = constant->type;
    } else if (staticIndexValue(*arg) || isLenCall(*arg)) {
      sawStaticIntegerLiteral = true;
    } else {
      argType = inferExpressionType(*arg);
      if (!argType || !mlir::isa<mlir::IntegerType>(*argType))
        return std::nullopt;
    }
    if (!argType)
      continue;
    if (!mlir::isa<mlir::IntegerType>(*argType))
      return std::nullopt;
    if (!rangeType)
      rangeType = *argType;
    else if (*rangeType != *argType)
      return std::nullopt;
  }
  if (!rangeType && sawStaticIntegerLiteral)
    return builder.getI64Type();
  return rangeType;
}

std::optional<mlir::Type>
Builder::Impl::typeFromAnnotation(const parser::NodePtr &node) {
  if (!node)
    return std::nullopt;
  auto cached = annotationTypeCache.find(node.get());
  if (cached != annotationTypeCache.end())
    return cached->second;

  const std::map<std::string, mlir::Type> typeVariables;
  std::set<std::string> activeAliases;
  std::optional<mlir::Type> resolved =
      typeFromAnnotation(node, typeVariables, activeAliases);
  if (resolved)
    annotationTypeCache.try_emplace(node.get(), *resolved);
  return resolved;
}

std::optional<mlir::Type> Builder::Impl::typeFromAnnotation(
    const parser::NodePtr &node,
    const std::map<std::string, mlir::Type> &typeVariables) {
  std::set<std::string> activeAliases;
  return typeFromAnnotation(node, typeVariables, activeAliases);
}

std::optional<py::CallableType>
Builder::Impl::callableParameterPackFromAnnotation(
    const parser::NodePtr &node,
    const std::map<std::string, mlir::Type> &typeVariables,
    std::set<std::string> &activeAliases) {
  return callableParameterPackFromAnnotation(node, typeVariables, activeAliases,
                                             AnnotationUse::Value);
}

std::optional<py::CallableType>
Builder::Impl::callableParameterPackFromAnnotation(
    const parser::NodePtr &node,
    const std::map<std::string, mlir::Type> &typeVariables,
    std::set<std::string> &activeAliases, AnnotationUse annotationUse) {
  if (!node)
    return std::nullopt;

  const parser::NodePtr *packExpr = &node;
  if (node->kind == "Starred") {
    const parser::NodePtr *value = nodeField(*node, "value");
    if (!value || !*value)
      return std::nullopt;
    packExpr = value;
  }

  if ((*packExpr)->kind == "List" || (*packExpr)->kind == "Tuple") {
    const std::vector<parser::NodePtr> *items =
        nodeListField(**packExpr, "elts");
    if (!items)
      return std::nullopt;
    llvm::SmallVector<mlir::Type> elementTypes;
    for (const parser::NodePtr &item : *items) {
      std::optional<mlir::Type> itemType =
          typeFromAnnotation(item, typeVariables, activeAliases, annotationUse);
      if (!itemType)
        return std::nullopt;
      elementTypes.push_back(*itemType);
    }
    return callableParameterPackFromTuple(elementTypes);
  }
  if (isEllipsisConstant(*packExpr)) {
    mlir::Type object = py::ObjectType::get(&context);
    mlir::Type vararg = py::TupleType::get(&context, {object});
    mlir::Type kwargs = dictType(strType(), object);
    return py::CallableType::get(&context, {}, {}, vararg, kwargs, {});
  }

  if ((*packExpr)->kind == "Subscript") {
    const parser::NodePtr *value = nodeField(**packExpr, "value");
    const parser::NodePtr *slice = nodeField(**packExpr, "slice");
    if (!value || !*value || !slice || !*slice)
      return std::nullopt;
    if (isTypingName(**value, "Concatenate")) {
      llvm::SmallVector<parser::NodePtr> items;
      if ((*slice)->kind == "Tuple") {
        const std::vector<parser::NodePtr> *elts =
            nodeListField(**slice, "elts");
        if (!elts)
          return std::nullopt;
        items.append(elts->begin(), elts->end());
      } else {
        items.push_back(*slice);
      }
      if (items.size() < 2 || !items.back())
        return std::nullopt;
      llvm::SmallVector<mlir::Type> prefixTypes;
      for (const parser::NodePtr &item : llvm::ArrayRef(items).drop_back()) {
        std::optional<mlir::Type> itemType = typeFromAnnotation(
            item, typeVariables, activeAliases, annotationUse);
        if (!itemType)
          return std::nullopt;
        prefixTypes.push_back(*itemType);
      }
      std::optional<py::CallableType> suffix =
          callableParameterPackFromAnnotation(items.back(), typeVariables,
                                              activeAliases, annotationUse);
      if (!suffix)
        return std::nullopt;
      return prependCallableParameterPack(prefixTypes, *suffix);
    }
  }

  std::optional<mlir::Type> packType = typeFromAnnotation(
      *packExpr, typeVariables, activeAliases, annotationUse);
  if (!packType)
    return std::nullopt;
  if (auto signature = mlir::dyn_cast<py::CallableType>(*packType))
    return signature;
  if (auto tuple = mlir::dyn_cast<py::TupleType>(*packType))
    return callableParameterPackFromTuple(tuple.getElementTypes());
  return std::nullopt;
}

std::optional<mlir::Type> Builder::Impl::typeFromAnnotation(
    const parser::NodePtr &node,
    const std::map<std::string, mlir::Type> &typeVariables,
    std::set<std::string> &activeAliases) {
  return typeFromAnnotation(node, typeVariables, activeAliases,
                            AnnotationUse::Value);
}

std::optional<mlir::Type> Builder::Impl::typeFromAnnotation(
    const parser::NodePtr &node,
    const std::map<std::string, mlir::Type> &typeVariables,
    std::set<std::string> &activeAliases, AnnotationUse annotationUse) {
  if (!node)
    return std::nullopt;
  auto knownAnnotationName =
      [&](const parser::Node &expr) -> std::optional<std::string> {
    if (expr.kind == "Name") {
      const std::string *name = stringField(expr, "id");
      if (!name)
        return std::nullopt;
      auto alias = staticAnnotationAliases.find(*name);
      if (alias != staticAnnotationAliases.end())
        return alias->second;
      return *name;
    }
    if (expr.kind == "Attribute") {
      const parser::NodePtr *value = nodeField(expr, "value");
      const std::string *attr = stringField(expr, "attr");
      if (!value || !*value || !attr || (*value)->kind != "Name")
        return std::nullopt;
      const std::string *moduleAlias = stringField(**value, "id");
      if (!moduleAlias)
        return std::nullopt;
      auto module = staticModules.find(*moduleAlias);
      if (module == staticModules.end())
        return std::nullopt;
      if (module->second == "typing" || module->second == "collections.abc" ||
          module->second == "asyncio" || module->second == "types")
        return *attr;
      return std::nullopt;
    }
    return std::nullopt;
  };
  auto isEllipsisConstant = [](const parser::NodePtr &expr) {
    if (!expr || expr->kind != "Constant")
      return false;
    const parser::FieldValue *value = valueField(*expr, "value");
    return value && std::holds_alternative<parser::Ellipsis>(*value);
  };
  auto typePackFromAnnotation =
      [&](const parser::NodePtr &expr,
          const std::map<std::string, mlir::Type> &variables,
          std::set<std::string> &aliases) -> std::optional<py::TupleType> {
    if (!expr)
      return std::nullopt;
    const parser::NodePtr *packExpr = &expr;
    if (expr->kind == "Starred") {
      const parser::NodePtr *value = nodeField(*expr, "value");
      if (!value || !*value)
        return std::nullopt;
      packExpr = value;
    }
    std::optional<mlir::Type> packType =
        typeFromAnnotation(*packExpr, variables, aliases, annotationUse);
    if (!packType)
      return std::nullopt;
    return mlir::dyn_cast<py::TupleType>(*packType);
  };
  auto annotationArguments = [&](const parser::NodePtr &slice)
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
  auto protocolAnnotationType =
      [&](llvm::StringRef protocolName, const parser::NodePtr &slice,
          const std::map<std::string, mlir::Type> &variables,
          std::set<std::string> &aliases) -> std::optional<mlir::Type> {
    const protocols::Table &table = protocols::Table::get(context);
    const protocols::ProtocolInfo *info = table.lookup(protocolName);
    if (!info || !info->isProtocol || protocolName == "Protocol")
      return std::nullopt;
    llvm::SmallVector<parser::NodePtr> parsedArgsStorage;
    if (slice) {
      std::optional<llvm::SmallVector<parser::NodePtr>> parsedArgs =
          annotationArguments(slice);
      if (!parsedArgs)
        return std::nullopt;
      parsedArgsStorage = std::move(*parsedArgs);
    }
    llvm::SmallVector<mlir::Type> args;
    for (const parser::NodePtr &arg : parsedArgsStorage) {
      std::optional<mlir::Type> argType = typeFromAnnotation(
          arg, variables, aliases, AnnotationUse::TypeContract);
      if (!argType)
        return std::nullopt;
      args.push_back(*argType);
    }
    std::optional<std::vector<mlir::Type>> completed =
        table.completeProtocolArguments(protocolName, args);
    if (!completed)
      return std::nullopt;
    return py::ProtocolType::get(&context, protocolName, *completed);
  };
  if (node->kind == "Name") {
    const std::string *name = stringField(*node, "id");
    if (!name)
      return std::nullopt;
    std::string annotationName = *name;
    auto annotationAlias = staticAnnotationAliases.find(*name);
    if (annotationAlias != staticAnnotationAliases.end())
      annotationName = annotationAlias->second;
    auto typeVariable = typeVariables.find(*name);
    if (typeVariable != typeVariables.end())
      return typeVariable->second;
    if (annotationName == "Any" || annotationName == "object")
      return py::ObjectType::get(&context);
    if (annotationName == "NoneType")
      return noneType();
    if (annotationName == "TracebackType")
      return py::TracebackType::get(&context);
    if (annotationName == "type" || annotationName == "Type")
      return py::TypeType::get(&context, py::ObjectType::get(&context));
    if (*name == "int")
      return intType();
    if (*name == "str")
      return strType();
    if (*name == "float")
      return floatType();
    if (*name == "bool")
      return boolType();
    if (*name == "None")
      return noneType();
    if (std::optional<mlir::Type> protocol = protocolAnnotationType(
            annotationName, parser::NodePtr{}, typeVariables, activeAliases))
      return protocol;
    if (isBuiltinExceptionClass(*name))
      return exceptionType();
    auto alias = typeAliases.find(*name);
    if (alias != typeAliases.end())
      return alias->second;
    auto genericAlias = genericTypeAliases.find(*name);
    if (genericAlias != genericTypeAliases.end()) {
      if (!activeAliases.insert(*name).second)
        return std::nullopt;
      std::map<std::string, mlir::Type> substituted(typeVariables);
      for (const TypeAliasParameter &parameter :
           genericAlias->second.parameters) {
        if (!parameter.defaultValue) {
          activeAliases.erase(*name);
          return std::nullopt;
        }
        if (parameter.kind == TypeAliasParameterKind::TypeVarTuple ||
            parameter.kind == TypeAliasParameterKind::ParamSpec) {
          if (parameter.kind == TypeAliasParameterKind::ParamSpec) {
            std::optional<py::CallableType> pack =
                callableParameterPackFromAnnotation(parameter.defaultValue,
                                                    substituted, activeAliases,
                                                    annotationUse);
            if (!pack) {
              activeAliases.erase(*name);
              return std::nullopt;
            }
            substituted[parameter.name] = *pack;
          } else {
            std::optional<py::TupleType> pack = typePackFromAnnotation(
                parameter.defaultValue, substituted, activeAliases);
            if (!pack) {
              activeAliases.erase(*name);
              return std::nullopt;
            }
            substituted[parameter.name] = *pack;
          }
          continue;
        }
        std::optional<mlir::Type> defaultType = typeFromAnnotation(
            parameter.defaultValue, substituted, activeAliases, annotationUse);
        if (!defaultType || (parameter.bound &&
                             !typeAssignable(parameter.bound, *defaultType))) {
          activeAliases.erase(*name);
          return std::nullopt;
        }
        substituted[parameter.name] = *defaultType;
      }
      std::optional<mlir::Type> result =
          typeFromAnnotation(genericAlias->second.value, substituted,
                             activeAliases, annotationUse);
      activeAliases.erase(*name);
      return result;
    }
    auto classFound = classes.find(*name);
    if (classFound != classes.end() && !classFound->second.isGenericTemplate)
      return classType(*name);
    return std::nullopt;
  }
  if (node->kind == "Attribute") {
    std::optional<std::string> annotationName = knownAnnotationName(*node);
    if (!annotationName)
      return std::nullopt;
    if (*annotationName == "Any" || *annotationName == "object")
      return py::ObjectType::get(&context);
    if (*annotationName == "NoneType")
      return noneType();
    if (*annotationName == "TracebackType")
      return py::TracebackType::get(&context);
    if (*annotationName == "type" || *annotationName == "Type")
      return py::TypeType::get(&context, py::ObjectType::get(&context));
    if (std::optional<mlir::Type> protocol = protocolAnnotationType(
            *annotationName, parser::NodePtr{}, typeVariables, activeAliases))
      return protocol;
    return std::nullopt;
  }
  // Union values lower with an explicit active-member tag, so primitive-backed
  // members can participate without relying on nullable object headers.
  auto supportedUnionAnnotation = [annotationUse](mlir::Type type) {
    auto unionType = mlir::dyn_cast<py::UnionType>(type);
    if (!unionType)
      return true;
    if (annotationUse == AnnotationUse::TypeContract)
      return true;
    for (mlir::Type member : unionType.getMemberTypes()) {
      if (mlir::isa<py::NoneType>(member))
        continue;
      if (!mlir::isa<py::IntType, py::BoolType, py::FloatType, py::StrType,
                     py::ExceptionType, py::TracebackType, py::ClassType,
                     py::TypeType>(member))
        return false;
    }
    return true;
  };
  if (node->kind == "BinOp") {
    std::optional<std::string> op = symbolField(*node, "op");
    const parser::NodePtr *lhs = nodeField(*node, "left");
    const parser::NodePtr *rhs = nodeField(*node, "right");
    if (!op || *op != "|" || !lhs || !*lhs || !rhs || !*rhs)
      return std::nullopt;
    std::optional<mlir::Type> lhsType =
        typeFromAnnotation(*lhs, typeVariables, activeAliases, annotationUse);
    std::optional<mlir::Type> rhsType =
        typeFromAnnotation(*rhs, typeVariables, activeAliases, annotationUse);
    if (!lhsType || !rhsType)
      return std::nullopt;
    mlir::Type normalized =
        py::UnionType::getNormalized(&context, {*lhsType, *rhsType});
    if (!normalized || !supportedUnionAnnotation(normalized))
      return std::nullopt;
    return normalized;
  }
  if (node->kind == "Subscript") {
    std::optional<unsigned> width = intWidthFromSubscript(*node);
    if (width)
      return builder.getIntegerType(*width);
    const parser::NodePtr *value = nodeField(*node, "value");
    const parser::NodePtr *slice = nodeField(*node, "slice");
    if (!value || !*value || !slice || !*slice)
      return std::nullopt;
    if ((*value)->kind == "Name") {
      const std::string *name = stringField(**value, "id");
      auto alias =
          name ? genericTypeAliases.find(*name) : genericTypeAliases.end();
      if (alias != genericTypeAliases.end()) {
        llvm::SmallVector<parser::NodePtr> args;
        if ((*slice)->kind == "Tuple") {
          const std::vector<parser::NodePtr> *items =
              nodeListField(**slice, "elts");
          if (!items)
            return std::nullopt;
          args.append(items->begin(), items->end());
        } else {
          args.push_back(*slice);
        }
        const TypeAliasInfo &info = alias->second;
        if (!activeAliases.insert(*name).second)
          return std::nullopt;
        std::map<std::string, mlir::Type> substituted(typeVariables);

        std::optional<std::size_t> packIndex;
        for (auto indexed : llvm::enumerate(info.parameters)) {
          if (indexed.value().kind == TypeAliasParameterKind::TypeVarTuple ||
              indexed.value().kind == TypeAliasParameterKind::ParamSpec) {
            packIndex = indexed.index();
            break;
          }
        }

        if (!packIndex) {
          if (args.size() > info.parameters.size()) {
            activeAliases.erase(*name);
            return std::nullopt;
          }
          for (std::size_t index = 0; index < info.parameters.size(); ++index) {
            const TypeAliasParameter &parameter = info.parameters[index];
            const bool usesDefault = index >= args.size();
            const parser::NodePtr *argument =
                usesDefault ? &parameter.defaultValue : &args[index];
            if (!argument || !*argument) {
              activeAliases.erase(*name);
              return std::nullopt;
            }
            std::optional<mlir::Type> argumentType = typeFromAnnotation(
                *argument, usesDefault ? substituted : typeVariables,
                activeAliases, annotationUse);
            if (!argumentType) {
              activeAliases.erase(*name);
              return std::nullopt;
            }
            if (parameter.bound &&
                !typeAssignable(parameter.bound, *argumentType)) {
              activeAliases.erase(*name);
              return std::nullopt;
            }
            substituted[parameter.name] = *argumentType;
          }
        } else {
          const std::size_t fixedCount = *packIndex;
          if (args.size() < fixedCount) {
            for (std::size_t index = args.size(); index < fixedCount; ++index) {
              if (!info.parameters[index].defaultValue) {
                activeAliases.erase(*name);
                return std::nullopt;
              }
            }
          }
          for (std::size_t index = 0; index < fixedCount; ++index) {
            const TypeAliasParameter &parameter = info.parameters[index];
            const bool usesDefault = index >= args.size();
            const parser::NodePtr *argument =
                usesDefault ? &parameter.defaultValue : &args[index];
            if (!argument || !*argument) {
              activeAliases.erase(*name);
              return std::nullopt;
            }
            std::optional<mlir::Type> argumentType = typeFromAnnotation(
                *argument, usesDefault ? substituted : typeVariables,
                activeAliases, annotationUse);
            if (!argumentType ||
                (parameter.bound &&
                 !typeAssignable(parameter.bound, *argumentType))) {
              activeAliases.erase(*name);
              return std::nullopt;
            }
            substituted[parameter.name] = *argumentType;
          }

          const TypeAliasParameter &packParameter = info.parameters[*packIndex];
          llvm::SmallVector<mlir::Type> packTypes;
          std::optional<py::CallableType> paramSpecPack;
          if (args.size() > fixedCount) {
            if (packParameter.kind == TypeAliasParameterKind::ParamSpec &&
                args.size() == fixedCount + 1) {
              paramSpecPack = callableParameterPackFromAnnotation(
                  args[fixedCount], typeVariables, activeAliases,
                  annotationUse);
              if (!paramSpecPack) {
                activeAliases.erase(*name);
                return std::nullopt;
              }
            } else {
              for (std::size_t index = fixedCount; index < args.size();
                   ++index) {
                const parser::NodePtr &argument = args[index];
                if (!argument) {
                  activeAliases.erase(*name);
                  return std::nullopt;
                }
                if (argument->kind == "Starred") {
                  std::optional<py::TupleType> expanded =
                      typePackFromAnnotation(argument, typeVariables,
                                             activeAliases);
                  if (!expanded) {
                    activeAliases.erase(*name);
                    return std::nullopt;
                  }
                  packTypes.append(expanded->getElementTypes().begin(),
                                   expanded->getElementTypes().end());
                  continue;
                }
                std::optional<mlir::Type> argumentType = typeFromAnnotation(
                    argument, typeVariables, activeAliases, annotationUse);
                if (!argumentType) {
                  activeAliases.erase(*name);
                  return std::nullopt;
                }
                packTypes.push_back(*argumentType);
              }
            }
          } else if (packParameter.defaultValue) {
            if (packParameter.kind == TypeAliasParameterKind::ParamSpec) {
              paramSpecPack = callableParameterPackFromAnnotation(
                  packParameter.defaultValue, substituted, activeAliases,
                  annotationUse);
              if (!paramSpecPack) {
                activeAliases.erase(*name);
                return std::nullopt;
              }
            } else {
              std::optional<py::TupleType> defaultPack = typePackFromAnnotation(
                  packParameter.defaultValue, substituted, activeAliases);
              if (!defaultPack) {
                activeAliases.erase(*name);
                return std::nullopt;
              }
              packTypes.append(defaultPack->getElementTypes().begin(),
                               defaultPack->getElementTypes().end());
            }
          }
          if (packParameter.kind == TypeAliasParameterKind::ParamSpec)
            substituted[packParameter.name] =
                paramSpecPack
                    ? mlir::Type(*paramSpecPack)
                    : mlir::Type(callableParameterPackFromTuple(packTypes));
          else
            substituted[packParameter.name] =
                py::TupleType::get(&context, packTypes);
        }
        std::optional<mlir::Type> result = typeFromAnnotation(
            info.value, substituted, activeAliases, annotationUse);
        activeAliases.erase(*name);
        return result;
      }
      auto classTemplate = name ? classes.find(*name) : classes.end();
      if (classTemplate != classes.end() &&
          classTemplate->second.isGenericTemplate) {
        llvm::SmallVector<parser::NodePtr> args;
        if ((*slice)->kind == "Tuple") {
          const std::vector<parser::NodePtr> *items =
              nodeListField(**slice, "elts");
          if (!items)
            return std::nullopt;
          args.append(items->begin(), items->end());
        } else {
          args.push_back(*slice);
        }
        std::optional<std::string> specialized =
            instantiateGenericClassFromAnnotation(*node, *name, args,
                                                  typeVariables, activeAliases);
        if (!specialized)
          return std::nullopt;
        return classType(*specialized);
      }
    }
    std::optional<std::string> annotationBase = knownAnnotationName(**value);
    if (annotationBase) {
      if (*annotationBase == "Optional") {
        std::optional<mlir::Type> payload = typeFromAnnotation(
            *slice, typeVariables, activeAliases, annotationUse);
        if (!payload)
          return std::nullopt;
        mlir::Type normalized =
            py::UnionType::getNormalized(&context, {*payload, noneType()});
        if (!normalized || !supportedUnionAnnotation(normalized))
          return std::nullopt;
        return normalized;
      }
      if (*annotationBase == "Union") {
        llvm::SmallVector<parser::NodePtr> items;
        if ((*slice)->kind == "Tuple") {
          const std::vector<parser::NodePtr> *elements =
              nodeListField(**slice, "elts");
          if (!elements)
            return std::nullopt;
          items.append(elements->begin(), elements->end());
        } else {
          items.push_back(*slice);
        }
        llvm::SmallVector<mlir::Type> members;
        for (const parser::NodePtr &item : items) {
          std::optional<mlir::Type> member = typeFromAnnotation(
              item, typeVariables, activeAliases, annotationUse);
          if (!member)
            return std::nullopt;
          members.push_back(*member);
        }
        mlir::Type normalized = py::UnionType::getNormalized(&context, members);
        if (!normalized || !supportedUnionAnnotation(normalized))
          return std::nullopt;
        return normalized;
      }
      if (*annotationBase == "Iterator") {
        std::optional<mlir::Type> element = typeFromAnnotation(
            *slice, typeVariables, activeAliases, annotationUse);
        if (!element)
          return std::nullopt;
        std::optional<py::ProtocolType> iterator =
            protocolType("Iterator", {*element});
        if (!iterator)
          return std::nullopt;
        return *iterator;
      }
      if (*annotationBase == "type" || *annotationBase == "Type") {
        std::optional<mlir::Type> instance = typeFromAnnotation(
            *slice, typeVariables, activeAliases, annotationUse);
        if (!instance)
          return std::nullopt;
        return py::TypeType::get(&context, *instance);
      }
      if (*annotationBase == "Task" || *annotationBase == "Future") {
        std::optional<mlir::Type> resultType = typeFromAnnotation(
            *slice, typeVariables, activeAliases, annotationUse);
        if (!resultType)
          return std::nullopt;
        if (*annotationBase == "Task")
          return taskType(*resultType);
        return futureType(*resultType);
      }
      if (std::optional<mlir::Type> protocol = protocolAnnotationType(
              *annotationBase, *slice, typeVariables, activeAliases))
        return protocol;
      if (*annotationBase == "Callable") {
        if ((*slice)->kind != "Tuple")
          return std::nullopt;
        const std::vector<parser::NodePtr> *items =
            nodeListField(**slice, "elts");
        if (!items || items->size() != 2 || !items->front() || !(*items)[1])
          return std::nullopt;
        const parser::NodePtr &argsNode = items->front();
        std::optional<py::CallableType> pack =
            callableParameterPackFromAnnotation(argsNode, typeVariables,
                                                activeAliases, annotationUse);
        if (!pack)
          return std::nullopt;
        std::optional<mlir::Type> resultType = typeFromAnnotation(
            (*items)[1], typeVariables, activeAliases, annotationUse);
        if (!resultType)
          return std::nullopt;
        return callableSignatureWithResult(*pack, *resultType);
      }
      if (*annotationBase == "dict" || *annotationBase == "Dict") {
        if ((*slice)->kind != "Tuple")
          return std::nullopt;
        const std::vector<parser::NodePtr> *items =
            nodeListField(**slice, "elts");
        if (!items || items->size() != 2 || !items->front() || !(*items)[1])
          return std::nullopt;
        std::optional<mlir::Type> keyType = typeFromAnnotation(
            items->front(), typeVariables, activeAliases, annotationUse);
        std::optional<mlir::Type> valueType = typeFromAnnotation(
            (*items)[1], typeVariables, activeAliases, annotationUse);
        if (!keyType || !valueType)
          return std::nullopt;
        return dictType(*keyType, *valueType);
      }
      if (*annotationBase == "list" || *annotationBase == "List") {
        std::optional<mlir::Type> elementType = typeFromAnnotation(
            *slice, typeVariables, activeAliases, annotationUse);
        if (!elementType)
          return std::nullopt;
        return listType(*elementType);
      }
      if (*annotationBase == "tuple" || *annotationBase == "Tuple") {
        if ((*slice)->kind != "Tuple") {
          if ((*slice)->kind == "Starred") {
            std::optional<py::TupleType> pack =
                typePackFromAnnotation(*slice, typeVariables, activeAliases);
            if (!pack)
              return std::nullopt;
            return py::TupleType::get(&context, pack->getElementTypes());
          }
          std::optional<mlir::Type> elementType = typeFromAnnotation(
              *slice, typeVariables, activeAliases, annotationUse);
          if (!elementType)
            return std::nullopt;
          return py::TupleType::get(&context, {*elementType});
        }
        const std::vector<parser::NodePtr> *items =
            nodeListField(**slice, "elts");
        if (!items)
          return std::nullopt;
        if (items->size() == 2 && isEllipsisConstant((*items)[1])) {
          std::optional<mlir::Type> elementType = typeFromAnnotation(
              items->front(), typeVariables, activeAliases, annotationUse);
          if (!elementType)
            return std::nullopt;
          return py::TupleType::get(&context, {*elementType});
        }
        llvm::SmallVector<mlir::Type> elementTypes;
        for (const parser::NodePtr &item : *items) {
          if (item && item->kind == "Starred") {
            std::optional<py::TupleType> pack =
                typePackFromAnnotation(item, typeVariables, activeAliases);
            if (!pack)
              return std::nullopt;
            elementTypes.append(pack->getElementTypes().begin(),
                                pack->getElementTypes().end());
            continue;
          }
          std::optional<mlir::Type> elementType = typeFromAnnotation(
              item, typeVariables, activeAliases, annotationUse);
          if (!elementType)
            return std::nullopt;
          elementTypes.push_back(*elementType);
        }
        return py::TupleType::get(&context, elementTypes);
      }
    }
    if (std::optional<std::string> name = primitiveTypeName(**value)) {
      if (*name == "Float") {
        if ((*slice)->kind != "Constant")
          return std::nullopt;
        const parser::FieldValue *bitsValue = valueField(**slice, "value");
        const auto *bits =
            bitsValue ? std::get_if<std::int64_t>(bitsValue) : nullptr;
        if (!bits)
          return std::nullopt;
        if (*bits == 16)
          return builder.getF16Type();
        if (*bits == 32)
          return builder.getF32Type();
        if (*bits == 64)
          return builder.getF64Type();
        return std::nullopt;
      }
      if (*name == "Matrix" || *name == "Tensor") {
        if ((*slice)->kind != "Tuple")
          return std::nullopt;
        const std::vector<parser::NodePtr> *items =
            nodeListField(**slice, "elts");
        if (!items || items->size() < 2 || !items->front())
          return std::nullopt;
        std::optional<mlir::Type> elementType = typeFromAnnotation(
            items->front(), typeVariables, activeAliases, annotationUse);
        if (!elementType)
          return std::nullopt;
        llvm::SmallVector<int64_t> shape;
        for (std::size_t i = 1; i < items->size(); ++i) {
          if (!(*items)[i] || (*items)[i]->kind != "Constant")
            return std::nullopt;
          const parser::FieldValue *dimValue =
              valueField(*(*items)[i], "value");
          const auto *dim =
              dimValue ? std::get_if<std::int64_t>(dimValue) : nullptr;
          if (!dim)
            return std::nullopt;
          shape.push_back(*dim);
        }
        if (*name == "Matrix" && shape.size() != 2)
          return std::nullopt;
        return mlir::RankedTensorType::get(shape, *elementType);
      }
      return std::nullopt;
    }
    return std::nullopt;
  }
  if (node->kind == "Constant") {
    const parser::FieldValue *value = valueField(*node, "value");
    if (const auto *annotation =
            value ? std::get_if<std::string>(value) : nullptr) {
      if (annotationUse == AnnotationUse::Value && typeVariables.empty()) {
        auto cached = stringAnnotationTypeCache.find(*annotation);
        if (cached != stringAnnotationTypeCache.end())
          return cached->second;
      }

      parser::ParseOptions options;
      options.mode = parser::ParseMode::Expression;
      parser::ParseResult parsed =
          parser::parse(*annotation, "<annotation>", options);
      if (!parsed.ok() || !parsed.tree)
        return std::nullopt;
      const parser::NodePtr *body = nodeField(*parsed.tree, "body");
      if (!body || !*body || (*body)->kind == "Constant")
        return std::nullopt;
      std::optional<mlir::Type> resolved = typeFromAnnotation(
          *body, typeVariables, activeAliases, annotationUse);
      if (resolved && annotationUse == AnnotationUse::Value &&
          typeVariables.empty())
        stringAnnotationTypeCache.try_emplace(*annotation, *resolved);
      return resolved;
    }
    if (value && std::holds_alternative<std::monostate>(*value))
      return noneType();
  }
  return std::nullopt;
}

bool Builder::Impl::isTypeAliasMarker(const parser::NodePtr &node) const {
  if (!node)
    return false;
  if (node->kind == "Name") {
    const std::string *name = stringField(*node, "id");
    if (!name)
      return false;
    if (*name == "TypeAlias")
      return true;
    auto alias = staticAnnotationAliases.find(*name);
    return alias != staticAnnotationAliases.end() &&
           alias->second == "TypeAlias";
  }
  if (node->kind != "Attribute")
    return false;
  const parser::NodePtr *value = nodeField(*node, "value");
  const std::string *attr = stringField(*node, "attr");
  if (!value || !*value || (*value)->kind != "Name" || !attr ||
      *attr != "TypeAlias")
    return false;
  const std::string *moduleAlias = stringField(**value, "id");
  if (!moduleAlias)
    return false;
  auto module = staticModules.find(*moduleAlias);
  return module != staticModules.end() && module->second == "typing";
}

bool Builder::Impl::classSubtypeOf(llvm::StringRef derived,
                                   llvm::StringRef base) const {
  if (derived == base)
    return true;

  auto found = classes.find(derived.str());
  if (found == classes.end())
    return false;

  const ClassInfo &info = found->second;
  if (!info.mro.empty())
    return llvm::is_contained(info.mro, base.str());

  std::set<std::string> seen;
  std::function<bool(llvm::StringRef)> walk = [&](llvm::StringRef current) {
    auto currentInfo = classes.find(current.str());
    if (currentInfo == classes.end())
      return false;
    if (!seen.insert(current.str()).second)
      return false;
    for (const std::string &baseName : currentInfo->second.baseNames) {
      if (baseName == base || walk(baseName))
        return true;
    }
    return false;
  };
  return walk(derived);
}

bool Builder::Impl::classLayoutCompatible(llvm::StringRef derived,
                                          llvm::StringRef base) const {
  if (!classSubtypeOf(derived, base))
    return false;
  auto derivedIt = classes.find(derived.str());
  auto baseIt = classes.find(base.str());
  if (derivedIt == classes.end() || baseIt == classes.end())
    return false;
  return derivedIt->second.fields == baseIt->second.fields;
}

bool Builder::Impl::classConformsToProtocol(py::ClassType subtype,
                                            py::ProtocolType protocol) {
  auto classFound = classes.find(subtype.getClassName().str());
  if (classFound == classes.end())
    return false;
  if (!classFound->second.inheritanceResolved) {
    if (!classFound->second.definition)
      return false;
    std::set<std::string> resolving;
    if (!resolveClassInheritance(classFound->second.name,
                                 *classFound->second.definition, resolving))
      return false;
    classFound = classes.find(subtype.getClassName().str());
    if (classFound == classes.end() || !classFound->second.inheritanceResolved)
      return false;
  }

  const protocols::Table &table = protocols::Table::get(context);
  const protocols::ProtocolInfo *protocolInfo =
      table.lookup(protocol.getProtocolName());
  if (!protocolInfo || !protocolInfo->isProtocol)
    return false;

  std::string conformanceKey =
      subtype.getClassName().str() + "<:" + typeString(protocol);
  if (activeProtocolConformance.count(conformanceKey))
    return true;
  activeProtocolConformance.insert(conformanceKey);
  struct ConformanceGuard {
    std::set<std::string> &active;
    std::string key;
    ~ConformanceGuard() { active.erase(key); }
  } guard{activeProtocolConformance, conformanceKey};

  std::set<std::string> methodNames;
  std::set<std::string> visitedProtocols;
  collectProtocolMethodNames(table, protocol.getProtocolName(), methodNames,
                             visitedProtocols);

  auto varargElementType = [](mlir::Type varargType) -> mlir::Type {
    auto tuple = mlir::dyn_cast_if_present<py::TupleType>(varargType);
    if (!tuple || tuple.getElementTypes().size() != 1)
      return {};
    return tuple.getElementTypes().front();
  };

  auto methodSatisfiesContract =
      [&](const FunctionInfo &method,
          const protocols::ProtocolMethod &contract) -> bool {
    if (!method.methodKind.empty() && method.methodKind != "instance")
      return false;
    if (method.argTypes.empty() || !contract.signature)
      return false;

    llvm::ArrayRef<mlir::Type> expectedPositional =
        contract.signature.getPositionalTypes();
    if (expectedPositional.empty())
      return false;
    llvm::ArrayRef<mlir::Type> expectedUserPositional =
        expectedPositional.drop_front();

    llvm::ArrayRef<mlir::Type> actualArgs(method.argTypes);
    if (method.positionalCount == 0 ||
        method.positionalCount > actualArgs.size())
      return false;
    llvm::ArrayRef<mlir::Type> actualUserPositional =
        actualArgs.take_front(method.positionalCount).drop_front();

    std::size_t defaultCount =
        std::min(method.defaultValues.size(), actualUserPositional.size());
    std::size_t requiredActual = actualUserPositional.size() - defaultCount;
    if (requiredActual > expectedUserPositional.size())
      return false;

    mlir::Type actualVarargElement = varargElementType(method.varargType);
    if (expectedUserPositional.size() > actualUserPositional.size() &&
        !actualVarargElement)
      return false;

    for (auto [index, expected] : llvm::enumerate(expectedUserPositional)) {
      mlir::Type actual = index < actualUserPositional.size()
                              ? actualUserPositional[index]
                              : actualVarargElement;
      if (!actual || !typeAssignable(actual, expected))
        return false;
    }

    if (contract.signature.hasVararg()) {
      mlir::Type expectedVarargElement =
          varargElementType(contract.signature.getVarargType());
      if (!actualVarargElement || !expectedVarargElement ||
          !typeAssignable(actualVarargElement, expectedVarargElement))
        return false;
    }

    llvm::ArrayRef<mlir::Type> expectedKwonly =
        contract.signature.getKwOnlyTypes();
    llvm::ArrayRef<mlir::Type> actualKwonly =
        actualArgs.drop_front(method.positionalCount);
    if (!expectedKwonly.empty() || !actualKwonly.empty()) {
      std::size_t requiredActualKwonly = 0;
      for (const parser::NodePtr &defaultValue : method.kwonlyDefaultValues)
        if (!defaultValue)
          ++requiredActualKwonly;
      if (requiredActualKwonly > expectedKwonly.size())
        return false;
      if (expectedKwonly.size() > actualKwonly.size() && !method.kwargType)
        return false;
      for (auto [index, expected] : llvm::enumerate(expectedKwonly)) {
        if (index >= actualKwonly.size())
          break;
        if (!typeAssignable(actualKwonly[index], expected))
          return false;
      }
    }

    if (contract.signature.hasKwarg() && !method.kwargType)
      return false;

    llvm::ArrayRef<mlir::Type> expectedResults =
        contract.signature.getResultTypes();
    if (expectedResults.size() != 1)
      return false;
    mlir::Type actualResult =
        method.isAsync ? coroutineType(method.resultType) : method.resultType;
    return typeAssignable(expectedResults.front(), actualResult);
  };

  for (const std::string &methodName : methodNames) {
    auto methodSymbol = classFound->second.methods.find(methodName);
    if (methodSymbol == classFound->second.methods.end())
      return false;
    auto methodFound = functions.find(methodSymbol->second);
    if (methodFound == functions.end())
      return false;

    std::vector<protocols::ProtocolMethod> contracts =
        table.methodContractsOn(protocol, methodName);
    if (contracts.empty())
      return false;
    for (const protocols::ProtocolMethod &contract : contracts)
      if (!methodSatisfiesContract(methodFound->second, contract))
        return false;
  }
  return true;
}

bool Builder::Impl::classConformsToCallable(py::ClassType subtype,
                                            py::CallableType callable) {
  auto classFound = classes.find(subtype.getClassName().str());
  if (classFound == classes.end())
    return false;
  if (!classFound->second.inheritanceResolved) {
    if (!classFound->second.definition)
      return false;
    std::set<std::string> resolving;
    if (!resolveClassInheritance(classFound->second.name,
                                 *classFound->second.definition, resolving))
      return false;
    classFound = classes.find(subtype.getClassName().str());
    if (classFound == classes.end() || !classFound->second.inheritanceResolved)
      return false;
  }

  auto methodSymbol = classFound->second.methods.find("__call__");
  if (methodSymbol == classFound->second.methods.end())
    return false;
  auto methodFound = functions.find(methodSymbol->second);
  if (methodFound == functions.end())
    return false;

  auto varargElementType = [](mlir::Type varargType) -> mlir::Type {
    auto tuple = mlir::dyn_cast_if_present<py::TupleType>(varargType);
    if (!tuple || tuple.getElementTypes().size() != 1)
      return {};
    return tuple.getElementTypes().front();
  };

  const FunctionInfo &method = methodFound->second;
  if (!method.methodKind.empty() && method.methodKind != "instance")
    return false;
  if (method.argTypes.empty() || !callable)
    return false;

  llvm::ArrayRef<mlir::Type> expectedUserPositional =
      callable.getPositionalTypes();
  llvm::ArrayRef<mlir::Type> actualArgs(method.argTypes);
  if (method.positionalCount == 0 || method.positionalCount > actualArgs.size())
    return false;
  llvm::ArrayRef<mlir::Type> actualUserPositional =
      actualArgs.take_front(method.positionalCount).drop_front();

  std::size_t defaultCount =
      std::min(method.defaultValues.size(), actualUserPositional.size());
  std::size_t requiredActual = actualUserPositional.size() - defaultCount;
  if (requiredActual > expectedUserPositional.size())
    return false;

  mlir::Type actualVarargElement = varargElementType(method.varargType);
  if (expectedUserPositional.size() > actualUserPositional.size() &&
      !actualVarargElement)
    return false;

  for (auto [index, expected] : llvm::enumerate(expectedUserPositional)) {
    mlir::Type actual = index < actualUserPositional.size()
                            ? actualUserPositional[index]
                            : actualVarargElement;
    if (!actual || !typeAssignable(actual, expected))
      return false;
  }

  if (callable.hasVararg()) {
    mlir::Type expectedVarargElement =
        varargElementType(callable.getVarargType());
    if (!actualVarargElement || !expectedVarargElement ||
        !typeAssignable(actualVarargElement, expectedVarargElement))
      return false;
  }

  llvm::ArrayRef<mlir::Type> expectedKwonly = callable.getKwOnlyTypes();
  llvm::ArrayRef<mlir::Type> actualKwonly =
      actualArgs.drop_front(method.positionalCount);
  if (!expectedKwonly.empty() || !actualKwonly.empty()) {
    std::size_t requiredActualKwonly = 0;
    for (const parser::NodePtr &defaultValue : method.kwonlyDefaultValues)
      if (!defaultValue)
        ++requiredActualKwonly;
    if (requiredActualKwonly > expectedKwonly.size())
      return false;
    if (expectedKwonly.size() > actualKwonly.size() && !method.kwargType)
      return false;
    for (auto [index, expected] : llvm::enumerate(expectedKwonly)) {
      if (index >= actualKwonly.size())
        break;
      if (!typeAssignable(actualKwonly[index], expected))
        return false;
    }
  }

  if (callable.hasKwarg() && !method.kwargType)
    return false;

  llvm::ArrayRef<mlir::Type> expectedResults = callable.getResultTypes();
  if (expectedResults.size() != 1)
    return false;
  mlir::Type actualResult =
      method.isAsync ? coroutineType(method.resultType) : method.resultType;
  return typeAssignable(expectedResults.front(), actualResult);
}

bool Builder::Impl::typeSubtypeOf(mlir::Type subtype, mlir::Type supertype) {
  if (subtype == supertype)
    return true;
  if (!subtype || !supertype)
    return false;

  if (mlir::isa<py::ObjectType>(supertype))
    return py::isPyType(subtype);

  auto subtypeOfUnionMember = [&](mlir::Type candidate,
                                  py::UnionType unionType) {
    return llvm::any_of(unionType.getMemberTypes(), [&](mlir::Type member) {
      return typeSubtypeOf(candidate, member);
    });
  };

  if (auto superUnion = mlir::dyn_cast<py::UnionType>(supertype)) {
    if (auto subUnion = mlir::dyn_cast<py::UnionType>(subtype)) {
      return llvm::all_of(subUnion.getMemberTypes(), [&](mlir::Type member) {
        return subtypeOfUnionMember(member, superUnion);
      });
    }
    return subtypeOfUnionMember(subtype, superUnion);
  }

  if (auto subUnion = mlir::dyn_cast<py::UnionType>(subtype)) {
    return llvm::all_of(subUnion.getMemberTypes(), [&](mlir::Type member) {
      return typeSubtypeOf(member, supertype);
    });
  }

  if (py::CallableType superCallable = py::getCallableContract(supertype)) {
    if (!py::getCallableContract(subtype)) {
      if (auto subClass = mlir::dyn_cast<py::ClassType>(subtype))
        return classConformsToCallable(subClass, superCallable);
      return false;
    }
  }

  if (auto superProtocol = mlir::dyn_cast<py::ProtocolType>(supertype)) {
    if (protocols::Table::get(context).conformsTo(subtype, superProtocol))
      return true;
    if (auto subClass = mlir::dyn_cast<py::ClassType>(subtype))
      return classConformsToProtocol(subClass, superProtocol);
    return false;
  }

  auto subClass = mlir::dyn_cast<py::ClassType>(subtype);
  auto superClass = mlir::dyn_cast<py::ClassType>(supertype);
  if (subClass && superClass)
    return classSubtypeOf(subClass.getClassName(), superClass.getClassName());

  auto subTuple = mlir::dyn_cast<py::TupleType>(subtype);
  auto superTuple = mlir::dyn_cast<py::TupleType>(supertype);
  if (subTuple && superTuple) {
    auto subElems = subTuple.getElementTypes();
    auto superElems = superTuple.getElementTypes();
    if (subElems.size() != superElems.size())
      return false;
    for (auto [sub, sup] : llvm::zip(subElems, superElems))
      if (!typeSubtypeOf(sub, sup))
        return false;
    return true;
  }

  auto subList = mlir::dyn_cast<py::ListType>(subtype);
  auto superList = mlir::dyn_cast<py::ListType>(supertype);
  if (subList && superList)
    return subList.getElementType() == superList.getElementType();

  auto subDict = mlir::dyn_cast<py::DictType>(subtype);
  auto superDict = mlir::dyn_cast<py::DictType>(supertype);
  if (subDict && superDict)
    return subDict.getKeyType() == superDict.getKeyType() &&
           subDict.getValueType() == superDict.getValueType();

  py::CallableType subSig = py::getCallableContract(subtype);
  py::CallableType superSig = py::getCallableContract(supertype);
  if (subSig && superSig) {
    if (isCallableEllipsisContract(superSig)) {
      if (subSig.getResultTypes().size() != superSig.getResultTypes().size())
        return false;
      for (auto [subResult, superResult] :
           llvm::zip(subSig.getResultTypes(), superSig.getResultTypes()))
        if (!typeSubtypeOf(subResult, superResult))
          return false;
      return true;
    }
    if (isCallableEllipsisContract(subSig))
      return false;
    if ((!subSig.hasVararg() && superSig.hasVararg()) ||
        (superSig.hasKwarg() && !subSig.hasKwarg()) ||
        subSig.getPositionalOnlyCount() > superSig.getPositionalOnlyCount())
      return false;
    if (subSig.getKwOnlyTypes().size() != superSig.getKwOnlyTypes().size() ||
        subSig.getResultTypes().size() != superSig.getResultTypes().size())
      return false;
    llvm::ArrayRef<mlir::Type> subPositional = subSig.getPositionalTypes();
    llvm::ArrayRef<mlir::Type> superPositional = superSig.getPositionalTypes();
    if (!subSig.hasVararg() && subPositional.size() != superPositional.size())
      return false;
    if (subSig.hasVararg() && !superSig.hasVararg() &&
        subPositional.size() > superPositional.size())
      return false;
    mlir::Type subVarargElement =
        subSig.hasVararg() ? callableVarargElementType(subSig.getVarargType())
                           : mlir::Type();
    if (subSig.hasVararg() && !subVarargElement)
      return false;
    for (auto [index, superArg] : llvm::enumerate(superPositional)) {
      mlir::Type subArg = index < subPositional.size() ? subPositional[index]
                                                       : subVarargElement;
      if (!subArg)
        return false;
      if (index < subPositional.size() && subArg != superArg)
        return false;
      if (index >= subPositional.size() && !typeSubtypeOf(superArg, subArg))
        return false;
    }
    for (auto [subArg, superArg] :
         llvm::zip(subSig.getKwOnlyTypes(), superSig.getKwOnlyTypes()))
      if (!typeSubtypeOf(superArg, subArg))
        return false;
    if (superSig.hasVararg()) {
      mlir::Type superVarargElement =
          callableVarargElementType(superSig.getVarargType());
      if (!superVarargElement ||
          !typeSubtypeOf(superVarargElement, subVarargElement))
        return false;
    }
    if (superSig.hasKwarg() &&
        !typeSubtypeOf(superSig.getKwargType(), subSig.getKwargType()))
      return false;
    for (auto [subResult, superResult] :
         llvm::zip(subSig.getResultTypes(), superSig.getResultTypes()))
      if (!typeSubtypeOf(subResult, superResult))
        return false;
    return true;
  }

  return py::isSubtypeOf(subtype, supertype);
}

std::optional<std::string>
Builder::Impl::mostSpecificKnownClass(const Value &value) const {
  if (value.exactClass)
    return value.exactClass;
  if (value.provenClass)
    return value.provenClass;
  if (value.type) {
    auto classType = mlir::dyn_cast<py::ClassType>(value.type);
    if (classType)
      return classType.getClassName().str();
  }
  return std::nullopt;
}

std::optional<std::string>
Builder::Impl::classFactForView(const Value &value) const {
  if (value.exactClass)
    return value.exactClass;
  if (value.provenClass)
    return value.provenClass;
  return std::nullopt;
}

Value Builder::Impl::markExactClass(Value value,
                                    llvm::StringRef className) const {
  value.exactClass = className.str();
  value.provenClass = className.str();
  return value;
}

Value Builder::Impl::markProvenClass(Value value,
                                     llvm::StringRef className) const {
  if (value.exactClass)
    return value;
  if (value.provenClass && classSubtypeOf(*value.provenClass, className))
    return value;
  value.provenClass = className.str();
  return value;
}

std::optional<Value>
Builder::Impl::concreteProtocolValue(const Value &value) const {
  if (!value.value || !mlir::isa<py::ProtocolType>(value.type) ||
      !value.protocolConcreteType)
    return std::nullopt;
  mlir::Type concrete = *value.protocolConcreteType;
  if (!concrete || mlir::isa<py::ProtocolType>(concrete) ||
      concrete == value.type)
    return std::nullopt;
  if (value.value.getType() != concrete)
    return std::nullopt;
  return Value{value.value, concrete, value.exactClass, value.provenClass};
}

Value Builder::Impl::applyReturnedClassSummary(Value value,
                                               const FunctionInfo &info) const {
  if (!value.value || !info.returnedExactClass)
    return value;
  return markExactClass(std::move(value), *info.returnedExactClass);
}

Value Builder::Impl::viewClassAs(const parser::Node &anchor, Value value,
                                 llvm::StringRef targetClassName) {
  py::ClassType inputClass =
      value.type ? mlir::dyn_cast<py::ClassType>(value.type) : py::ClassType();
  if (!value.value || !inputClass) {
    error(anchor, "class view refinement requires a class value");
    return Value{{}, classType(targetClassName)};
  }

  py::ClassType targetType = classType(targetClassName);
  if (inputClass.getClassName() == targetClassName) {
    if (value.exactClass &&
        !classSubtypeOf(*value.exactClass, targetClassName)) {
      value.exactClass.reset();
    }
    return markProvenClass(std::move(value), targetClassName);
  }

  if (typeSubtypeOf(value.type, targetType)) {
    mlir::Value cast =
        builder.create<py::ClassUpcastOp>(loc(anchor), targetType, value.value);
    Value result{cast, targetType, value.exactClass, value.provenClass};
    std::optional<std::string> fact = mostSpecificKnownClass(value);
    if (fact)
      result.provenClass = *fact;
    return result;
  }

  if (!classSubtypeOf(targetClassName, inputClass.getClassName())) {
    error(anchor, "class view refinement from " + typeString(value.type) +
                      " to " + typeString(targetType) +
                      " is not a nominal downcast");
    return Value{{}, targetType};
  }

  std::optional<std::string> fact = classFactForView(value);
  if (fact && !classSubtypeOf(*fact, targetClassName)) {
    error(anchor, "known runtime class '" + *fact +
                      "' is not compatible with refined view " +
                      typeString(targetType));
    return Value{{}, targetType};
  }

  mlir::Value refined =
      builder.create<py::ClassRefineOp>(loc(anchor), targetType, value.value);
  Value result{refined, targetType, value.exactClass, value.provenClass};
  result = markProvenClass(std::move(result), targetClassName);
  return result;
}

std::vector<Builder::Impl::UnionMemberMatch>
Builder::Impl::unionMembersMatchingType(py::UnionType unionType,
                                        mlir::Type tested,
                                        bool requireLayoutCompatibleDowncast) {
  std::vector<UnionMemberMatch> matches;
  if (!unionType || !tested)
    return matches;
  for (mlir::Type member : unionType.getMemberTypes()) {
    if (mlir::isa<py::NoneType>(tested)) {
      if (mlir::isa<py::NoneType>(member))
        matches.push_back(UnionMemberMatch{member, member});
      continue;
    }
    if (typeSubtypeOf(member, tested)) {
      matches.push_back(UnionMemberMatch{member, member});
      continue;
    }
    auto memberClass = mlir::dyn_cast<py::ClassType>(member);
    auto testedClass = mlir::dyn_cast<py::ClassType>(tested);
    if (memberClass && testedClass &&
        classSubtypeOf(testedClass.getClassName(),
                       memberClass.getClassName()) &&
        (!requireLayoutCompatibleDowncast ||
         classLayoutCompatible(testedClass.getClassName(),
                               memberClass.getClassName())))
      matches.push_back(UnionMemberMatch{member, tested});
  }
  return matches;
}

bool Builder::Impl::typeAssignable(mlir::Type expected, mlir::Type actual) {
  auto &byActual = typeAssignableCache[expected];
  auto cached = byActual.find(actual);
  if (cached != byActual.end())
    return cached->second;

  auto remember = [&](bool result) {
    byActual.try_emplace(actual, result);
    return result;
  };

  if (expected == actual)
    return remember(true);

  if (!expected || !actual)
    return remember(false);

  std::string assignabilityKey =
      typeString(expected) + "<-" + typeString(actual);
  if (activeTypeAssignable.count(assignabilityKey))
    return true;
  activeTypeAssignable.insert(assignabilityKey);
  struct TypeAssignableGuard {
    std::set<std::string> &active;
    std::string key;
    ~TypeAssignableGuard() { active.erase(key); }
  } guard{activeTypeAssignable, assignabilityKey};

  if (auto expectedProtocol = mlir::dyn_cast<py::ProtocolType>(expected)) {
    if (protocols::Table::get(context).conformsTo(actual, expectedProtocol))
      return remember(true);
    if (auto actualClass = mlir::dyn_cast<py::ClassType>(actual)) {
      if (classConformsToProtocol(actualClass, expectedProtocol))
        return remember(true);
      return false;
    }
  }

  // Subtyping is assignability. The IR-level representation adaptation
  // remains a separate concern handled by coerceToExpectedType where needed.
  if (typeSubtypeOf(actual, expected))
    return remember(true);

  std::optional<typing::Term> expectedTerm = typeTermFromType(expected);
  std::optional<typing::Term> actualTerm = typeTermFromType(actual);
  if (!expectedTerm || !actualTerm)
    return remember(false);

  typing::Oracle oracle;
  typing::AlgorithmM inference(oracle);
  try {
    inference.unify(*expectedTerm, *actualTerm, {{"assignment"}});
    return remember(true);
  } catch (const typing::Error &) {
  }

  if (expectedTerm->kind != typing::Term::Kind::Con ||
      actualTerm->kind != typing::Term::Kind::Con ||
      expectedTerm->name != "tuple" || actualTerm->name != "tuple" ||
      expectedTerm->args.size() != 1 || actualTerm->args.empty())
    return remember(false);

  typing::AlgorithmM tupleInference(oracle);
  try {
    for (const typing::Term &element : actualTerm->args)
      tupleInference.unify(expectedTerm->args.front(), element,
                           {{"homogeneous tuple assignment"}});
    return remember(true);
  } catch (const typing::Error &) {
    return remember(false);
  }
}

std::optional<protocols::ProtocolMethod>
Builder::Impl::resolveProtocolMethodContract(
    const parser::Node &anchor, mlir::Type receiverType,
    llvm::StringRef methodName, llvm::ArrayRef<mlir::Type> argumentTypes,
    llvm::StringRef contextLabel) {
  std::optional<protocols::ProtocolMethod> contract =
      protocols::Table::get(context).resolveMethodContractOn(
          receiverType, methodName, argumentTypes);
  if (!contract) {
    std::string message = contextLabel.str() + " has no " + methodName.str() +
                          " contract on " + typeString(receiverType) +
                          " for argument types";
    if (argumentTypes.empty()) {
      message += " []";
    } else {
      message += " [";
      llvm::raw_string_ostream os(message);
      for (auto indexed : llvm::enumerate(argumentTypes)) {
        if (indexed.index() != 0)
          os << ", ";
        indexed.value().print(os);
      }
      os << "]";
    }
    error(anchor, message);
    return std::nullopt;
  }
  return contract;
}

std::optional<mlir::Type> Builder::Impl::resolveProtocolMethodResult(
    const parser::Node &anchor, mlir::Type receiverType,
    llvm::StringRef methodName, llvm::ArrayRef<mlir::Type> argumentTypes,
    llvm::StringRef contextLabel) {
  std::optional<protocols::ProtocolMethod> contract =
      resolveProtocolMethodContract(anchor, receiverType, methodName,
                                    argumentTypes, contextLabel);
  if (!contract)
    return std::nullopt;

  llvm::ArrayRef<mlir::Type> results = contract->signature.getResultTypes();
  if (results.size() != 1) {
    error(anchor, contextLabel.str() + " contract " + methodName.str() +
                      " on " + typeString(receiverType) +
                      " must return one "
                      "value, got " +
                      std::to_string(results.size()));
    return std::nullopt;
  }
  return results.front();
}

py::CallableType Builder::Impl::unaryMethodContract(mlir::Type receiverType,
                                                    mlir::Type argumentType,
                                                    mlir::Type resultType) {
  return py::CallableType::get(&context, {receiverType, argumentType}, {}, {},
                               {}, {resultType});
}

py::CallableType Builder::Impl::containsMethodContract(mlir::Type receiverType,
                                                       mlir::Type itemType) {
  return unaryMethodContract(receiverType, itemType, boolType());
}

// Protocol oracle: the conformance closure of the abstract container table
// (rfc/iterator-protocol.md). Iterable[T]/Iterator[T]/Container[T]/Sized/
// Reversible[T]/Collection[T]/Sequence[T] form the abstraction tower; these
// queries answer what the expansion of a concrete type's bases yields.
//   list[T]        |- Sequence[T]            -> Iterable[T], Sized
//   tuple[T, ...]  |- Sequence[T]   (homogeneous)
//   str            |- Sequence[str]
//   dict[K, V]     |- Collection[K]          -> Iterable[K], Sized
//   range          |- Sequence[int] (final)
//   Iterator[E]    |- Iterable[E]
std::optional<mlir::Type>
Builder::Impl::protocolIterableElement(mlir::Type type) {
  const protocols::Table &table = protocols::Table::get(context);
  std::optional<mlir::Type> result =
      table.resolveMethodResultOn(type, "__iter__", {});
  if (!result)
    return std::nullopt;
  auto iterator = mlir::dyn_cast<py::ProtocolType>(*result);
  if (!iterator || iterator.getProtocolName() != "Iterator" ||
      iterator.getArguments().size() != 1)
    return std::nullopt;
  return iterator.getArguments().front();
}

std::optional<mlir::Type>
Builder::Impl::protocolAsyncIterableElement(mlir::Type type) {
  const protocols::Table &table = protocols::Table::get(context);
  std::optional<mlir::Type> iteratorType =
      table.resolveMethodResultOn(type, "__aiter__", {});
  if (!iteratorType)
    return std::nullopt;

  std::optional<mlir::Type> awaitableType =
      table.resolveMethodResultOn(*iteratorType, "__anext__", {});
  if (!awaitableType)
    return std::nullopt;

  mlir::Type payloadType = awaitablePayloadType(*awaitableType);
  if (!payloadType)
    return std::nullopt;
  return payloadType;
}

Value Builder::Impl::coerceToExpectedType(const parser::Node &anchor,
                                          Value value, mlir::Type expected) {
  if (!value.value || !expected || value.type == expected)
    return value;

  if (mlir::isa<py::BoolType>(expected)) {
    auto integer = mlir::dyn_cast<mlir::IntegerType>(value.type);
    if (integer && integer.getWidth() == 1) {
      mlir::Value boxed = builder.create<py::CastFromPrimOp>(
          loc(anchor), expected, value.value);
      return Value{boxed, expected};
    }
  }
  if (mlir::isa<py::IntType>(expected) &&
      mlir::isa<mlir::IntegerType>(value.type)) {
    mlir::Value boxed =
        builder.create<py::CastFromPrimOp>(loc(anchor), expected, value.value);
    return Value{boxed, expected};
  }
  if (mlir::isa<py::FloatType>(expected) &&
      mlir::isa<mlir::FloatType>(value.type)) {
    mlir::Value boxed =
        builder.create<py::CastFromPrimOp>(loc(anchor), expected, value.value);
    return Value{boxed, expected};
  }

  if (py::isCallableType(expected)) {
    if (py::isCallableType(value.type) && typeAssignable(expected, value.type))
      return value;
    if (py::isCallableType(value.type) && value.callableInfo &&
        (value.callableInfo->varargParameterPack ||
         value.callableInfo->kwargParameterPack)) {
      py::CallableType pack = value.callableInfo->varargParameterPack
                                  ? value.callableInfo->varargParameterPack
                                  : value.callableInfo->kwargParameterPack;
      mlir::Type externalType =
          callableSignatureWithResult(pack, value.callableInfo->resultType);
      if (typeAssignable(expected, externalType)) {
        mlir::Value cast = builder
                               .create<mlir::UnrealizedConversionCastOp>(
                                   loc(anchor), expected, value.value)
                               .getResult(0);
        Value result{cast, expected};
        result.callableInfo = value.callableInfo;
        return result;
      }
    }
  }

  if (mlir::isa<py::ProtocolType>(value.type) && value.protocolConcreteType &&
      !mlir::isa<py::ProtocolType>(expected) &&
      typeAssignable(expected, *value.protocolConcreteType)) {
    if (std::optional<Value> concrete = concreteProtocolValue(value))
      return *concrete;
  }

  if (auto expectedProtocol = mlir::dyn_cast<py::ProtocolType>(expected)) {
    mlir::Type concrete =
        value.protocolConcreteType ? *value.protocolConcreteType : value.type;
    if (concrete && !mlir::isa<py::ProtocolType>(concrete) &&
        typeAssignable(expectedProtocol, concrete))
      return Value{value.value, expectedProtocol, value.exactClass,
                   value.provenClass, concrete};
  }

  auto upcastClass = [&](Value input, mlir::Type target) -> Value {
    auto inputClass = mlir::dyn_cast<py::ClassType>(input.type);
    auto targetClass = mlir::dyn_cast<py::ClassType>(target);
    if (!input.value || !inputClass || !targetClass ||
        !typeSubtypeOf(input.type, target) || input.type == target)
      return input;
    mlir::Value cast =
        builder.create<py::ClassUpcastOp>(loc(anchor), target, input.value);
    Value result{cast, target, input.exactClass, input.provenClass};
    std::optional<std::string> fact = mostSpecificKnownClass(input);
    if (fact)
      result.provenClass = *fact;
    return result;
  };

  if (auto expectedClass = mlir::dyn_cast<py::ClassType>(expected)) {
    if (mlir::isa<py::UnionType>(value.type) &&
        (typeSubtypeOf(value.type, expected) ||
         (classFactForView(value) &&
          classSubtypeOf(*classFactForView(value),
                         expectedClass.getClassName())))) {
      mlir::Value unwrapped =
          builder.create<py::UnionUnwrapOp>(loc(anchor), expected, value.value);
      Value result{unwrapped, expected, value.exactClass, value.provenClass};
      if (result.exactClass &&
          !classSubtypeOf(*result.exactClass, expectedClass.getClassName()))
        result.exactClass.reset();
      result = markProvenClass(std::move(result), expectedClass.getClassName());
      return result;
    }
    if (auto valueClass = mlir::dyn_cast<py::ClassType>(value.type)) {
      if (!typeSubtypeOf(value.type, expected) &&
          classSubtypeOf(expectedClass.getClassName(),
                         valueClass.getClassName())) {
        std::optional<std::string> fact = classFactForView(value);
        if (fact && classSubtypeOf(*fact, expectedClass.getClassName()))
          return viewClassAs(anchor, std::move(value),
                             expectedClass.getClassName());
      }
    }
    return upcastClass(value, expected);
  }

  auto unionType = mlir::dyn_cast<py::UnionType>(expected);
  if (!unionType)
    return value;
  if (auto valueUnion = mlir::dyn_cast<py::UnionType>(value.type)) {
    // Subset widening: union<...S> flows into union<...T> when S ⊆ T.
    if (!py::isSubtypeOf(valueUnion, unionType))
      return value;
  } else if (!unionType.hasMember(value.type)) {
    mlir::Type matchingMember;
    for (mlir::Type member : unionType.getMemberTypes()) {
      if (typeSubtypeOf(value.type, member)) {
        matchingMember = member;
        break;
      }
    }
    if (!matchingMember)
      return value;
    value = upcastClass(value, matchingMember);
    if (value.type != matchingMember)
      return value;
  }
  mlir::Value wrapped =
      builder.create<py::UnionWrapOp>(loc(anchor), expected, value.value);
  return Value{wrapped, expected, value.exactClass, value.provenClass};
}

mlir::Type Builder::Impl::typeFromClassAnnotation(llvm::StringRef className) {
  auto found = classes.find(className.str());
  if (found == classes.end())
    return {};
  return classType(className);
}

mlir::Type Builder::Impl::listElementType(mlir::Type type) {
  if (auto list = mlir::dyn_cast<py::ListType>(type))
    return list.getElementType();
  return {};
}

std::optional<std::pair<mlir::Type, mlir::Type>>
Builder::Impl::dictKeyValueTypes(mlir::Type type) {
  if (auto dict = mlir::dyn_cast<py::DictType>(type))
    return std::make_pair(dict.getKeyType(), dict.getValueType());
  return std::nullopt;
}

mlir::Type
Builder::Impl::listLiteralElementTypeForExpected(mlir::Type expectedType) {
  if (mlir::Type element = listElementType(expectedType))
    return element;
  std::optional<mlir::Type> protocolElement =
      protocolIterableElement(expectedType);
  return protocolElement ? *protocolElement : mlir::Type();
}

std::optional<std::pair<mlir::Type, mlir::Type>>
Builder::Impl::protocolMappingKeyValueTypes(mlir::Type type) {
  auto protocol = mlir::dyn_cast<py::ProtocolType>(type);
  if (!protocol || protocol.getArguments().size() < 2)
    return std::nullopt;
  llvm::StringRef name = protocol.getProtocolName();
  if (name != "Mapping" && name != "MutableMapping")
    return std::nullopt;
  return std::make_pair(protocol.getArguments()[0], protocol.getArguments()[1]);
}

std::optional<std::pair<mlir::Type, mlir::Type>>
Builder::Impl::dictLiteralKeyValueTypesForExpected(mlir::Type expectedType) {
  if (std::optional<std::pair<mlir::Type, mlir::Type>> dictTypes =
          dictKeyValueTypes(expectedType))
    return dictTypes;
  return protocolMappingKeyValueTypes(expectedType);
}

bool Builder::Impl::dictStorageSupported(mlir::Type keyType,
                                         mlir::Type valueType) {
  auto isSlotType = [](mlir::Type type) {
    return mlir::isa<mlir::IntegerType, mlir::FloatType, py::IntType,
                     py::BoolType, py::FloatType, py::NoneType, py::StrType>(
        type);
  };
  return isSlotType(keyType) && isSlotType(valueType);
}

std::optional<std::string> Builder::Impl::classNameFromType(mlir::Type type) {
  if (auto classTy = mlir::dyn_cast<py::ClassType>(type))
    return classTy.getClassName().str();
  return std::nullopt;
}

bool Builder::Impl::isTensorConstructorCallee(const parser::Node &node) const {
  if (node.kind != "Subscript")
    return false;
  const parser::NodePtr *value = nodeField(node, "value");
  if (!value || !*value)
    return false;
  std::optional<std::string> name = primitiveTypeName(**value);
  return name && (*name == "Matrix" || *name == "Tensor");
}

std::optional<std::int64_t>
Builder::Impl::staticIndexValue(const parser::Node &node) const {
  if (node.kind == "Constant") {
    const parser::FieldValue *value = valueField(node, "value");
    const auto *integer = value ? std::get_if<std::int64_t>(value) : nullptr;
    if (integer)
      return *integer;
    return std::nullopt;
  }

  if (node.kind == "UnaryOp") {
    std::optional<std::string> op = symbolField(node, "op");
    const parser::NodePtr *operand = nodeField(node, "operand");
    if (!op || !operand || !*operand)
      return std::nullopt;
    std::optional<std::int64_t> value = staticIndexValue(**operand);
    if (!value)
      return std::nullopt;
    if (*op == "+")
      return *value;
    if (*op == "-")
      return -*value;
  }

  return std::nullopt;
}

std::optional<std::int64_t>
Builder::Impl::staticPyIntValue(mlir::Value value) const {
  auto constant =
      value ? value.getDefiningOp<py::IntConstantOp>() : py::IntConstantOp();
  if (!constant)
    return std::nullopt;

  llvm::StringRef text = constant.getValue();
  std::int64_t result = 0;
  const char *begin = text.data();
  const char *end = text.data() + text.size();
  auto parsed = std::from_chars(begin, end, result);
  if (parsed.ec != std::errc{} || parsed.ptr != end)
    return std::nullopt;
  return result;
}

std::optional<unsigned>
Builder::Impl::intWidthFromSubscript(const parser::Node &node) {
  if (node.kind != "Subscript")
    return std::nullopt;
  const parser::NodePtr *value = nodeField(node, "value");
  const parser::NodePtr *slice = nodeField(node, "slice");
  if (!value || !*value || !slice || !*slice || (*slice)->kind != "Constant")
    return std::nullopt;
  std::optional<std::string> name = primitiveTypeName(**value);
  const parser::FieldValue *widthValue = valueField(**slice, "value");
  const auto *width =
      widthValue ? std::get_if<std::int64_t>(widthValue) : nullptr;
  if (!name || *name != "Int" || !width || *width <= 0 || *width > 64)
    return std::nullopt;
  return static_cast<unsigned>(*width);
}

std::optional<PrimitiveConstant>
Builder::Impl::primitiveIntConstructorConstant(const parser::Node &node) {
  if (node.kind != "Call")
    return std::nullopt;
  const parser::NodePtr *func = nodeField(node, "func");
  const std::vector<parser::NodePtr> *args = nodeListField(node, "args");
  const std::vector<parser::NodePtr> *keywords =
      nodeListField(node, "keywords");
  if (!func || !*func || !args || args->size() != 1 || !args->front() ||
      (keywords && !keywords->empty()))
    return std::nullopt;

  std::optional<unsigned> width = intWidthFromSubscript(**func);
  std::optional<std::int64_t> integer = staticIndexValue(*args->front());
  if (!width || !integer)
    return std::nullopt;
  return PrimitiveConstant{
      mlir::IntegerType::get(&context, static_cast<unsigned>(*width)), *integer,
      0.0};
}

std::optional<PrimitiveConstant>
Builder::Impl::primitiveScalarConstructorConstant(const parser::Node &node) {
  if (std::optional<PrimitiveConstant> integer =
          primitiveIntConstructorConstant(node))
    return integer;

  if (node.kind != "Call")
    return std::nullopt;
  const parser::NodePtr *func = nodeField(node, "func");
  const std::vector<parser::NodePtr> *args = nodeListField(node, "args");
  const std::vector<parser::NodePtr> *keywords =
      nodeListField(node, "keywords");
  if (!func || !*func || !args || args->size() != 1 || !args->front() ||
      (keywords && !keywords->empty()))
    return std::nullopt;

  std::optional<mlir::Type> type = typeFromAnnotation(*func);
  if (!type)
    return std::nullopt;
  if (auto floatTy = mlir::dyn_cast<mlir::FloatType>(*type)) {
    std::optional<double> number = staticNumericValue(*args->front());
    if (!number)
      return std::nullopt;
    return PrimitiveConstant{floatTy, 0, *number};
  }
  return std::nullopt;
}

mlir::Type Builder::Impl::awaitablePayloadType(mlir::Type type) {
  return protocols::Table::get(context).awaitablePayloadType(type);
}

} // namespace lython::emitter
