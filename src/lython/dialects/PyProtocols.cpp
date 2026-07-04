#include "PyProtocols.h"

#include "CallableArgumentMatcher.h"
#include "CandidateSelection.h"
#include "PyCallableShape.h"

#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <utility>

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

namespace py::protocols {

bool sameMethodContract(const ProtocolMethod &lhs, const ProtocolMethod &rhs) {
  return lhs.signature == rhs.signature && lhs.mayThrow == rhs.mayThrow &&
         lhs.noThrow == rhs.noThrow;
}

std::optional<std::int64_t> normalizeFiniteTupleIndex(std::int64_t index,
                                                      std::size_t size) {
  if (size == 0)
    return std::nullopt;
  if (index < 0)
    index += static_cast<std::int64_t>(size);
  if (index < 0 || index >= static_cast<std::int64_t>(size))
    return std::nullopt;
  return index;
}

namespace {

std::vector<std::string> stringArray(mlir::Operation *op,
                                     llvm::StringRef name) {
  std::vector<std::string> values;
  if (auto array = op->getAttrOfType<mlir::ArrayAttr>(name))
    for (mlir::Attribute element : array)
      if (auto text = mlir::dyn_cast<mlir::StringAttr>(element))
        values.push_back(text.getValue().str());
  return values;
}

std::vector<unsigned> unsignedArray(mlir::Operation *op, llvm::StringRef name) {
  std::vector<unsigned> values;
  if (auto array = op->getAttrOfType<mlir::ArrayAttr>(name))
    for (mlir::Attribute element : array) {
      auto integer = mlir::dyn_cast<mlir::IntegerAttr>(element);
      if (!integer || integer.getValue().isNegative())
        continue;
      values.push_back(integer.getValue().getZExtValue());
    }
  return values;
}

std::vector<mlir::Type> typeArray(mlir::Operation *op, llvm::StringRef name) {
  std::vector<mlir::Type> values;
  if (auto array = op->getAttrOfType<mlir::ArrayAttr>(name))
    for (mlir::Attribute element : array)
      if (auto typeAttr = mlir::dyn_cast<mlir::TypeAttr>(element))
        values.push_back(typeAttr.getValue());
  return values;
}

std::vector<mlir::Type> trailingTypeDefaults(mlir::Operation *op,
                                             llvm::StringRef name,
                                             std::size_t parameterCount) {
  std::vector<mlir::Type> values(parameterCount);
  auto array = op->getAttrOfType<mlir::ArrayAttr>(name);
  if (!array)
    return values;
  std::vector<mlir::Type> trailing;
  for (mlir::Attribute element : array)
    if (auto typeAttr = mlir::dyn_cast<mlir::TypeAttr>(element))
      trailing.push_back(typeAttr.getValue());
  std::size_t copyCount = std::min(parameterCount, trailing.size());
  for (std::size_t index = 0; index < copyCount; ++index)
    values[parameterCount - copyCount + index] =
        trailing[trailing.size() - copyCount + index];
  return values;
}

std::optional<std::vector<mlir::Type>>
completeDirectArguments(const ProtocolInfo &info,
                        llvm::ArrayRef<mlir::Type> supplied) {
  if (supplied.size() > info.params.size())
    return std::nullopt;
  std::vector<mlir::Type> args(supplied.begin(), supplied.end());
  for (std::size_t index = args.size(); index < info.params.size(); ++index) {
    if (index >= info.paramDefaults.size() || !info.paramDefaults[index])
      return std::nullopt;
    args.push_back(info.paramDefaults[index]);
  }
  return args;
}

std::optional<std::vector<mlir::Type>>
completeShortArguments(const ProtocolInfo &info, const ProtocolShortForm &form,
                       llvm::ArrayRef<mlir::Type> supplied) {
  if (form.positions.size() != supplied.size())
    return std::nullopt;
  std::vector<mlir::Type> args(info.params.size());
  std::vector<bool> occupied(info.params.size());
  for (auto [index, position] : llvm::enumerate(form.positions)) {
    if (position >= info.params.size() || occupied[position])
      return std::nullopt;
    args[position] = supplied[index];
    occupied[position] = true;
  }

  std::size_t defaultIndex = 0;
  for (std::size_t index = 0; index < info.params.size(); ++index) {
    if (occupied[index])
      continue;
    if (defaultIndex < form.defaults.size()) {
      args[index] = form.defaults[defaultIndex++];
      continue;
    }
    if (index < info.paramDefaults.size() && info.paramDefaults[index]) {
      args[index] = info.paramDefaults[index];
      continue;
    }
    return std::nullopt;
  }
  return args;
}

bool hasMarker(mlir::Operation *op, llvm::StringRef name) {
  return op && op->hasAttr(name);
}

// Type variable occurrences are class types whose name starts with '$'.
std::optional<llvm::StringRef> typeVariableName(mlir::Type type) {
  if (auto typeVar = mlir::dyn_cast<py::TypeVarType>(type))
    return typeVar.getName();
  if (auto paramSpec = mlir::dyn_cast<py::ParamSpecType>(type))
    return paramSpec.getName();
  if (auto typeVarTuple = mlir::dyn_cast<py::TypeVarTupleType>(type))
    return typeVarTuple.getName();
  if (auto contract = mlir::dyn_cast<py::ContractType>(type)) {
    llvm::StringRef name = contract.getContractName();
    if (name.starts_with("$"))
      return name.drop_front();
  }
  auto classType = mlir::dyn_cast<py::ClassType>(type);
  if (!classType || !classType.getClassName().starts_with("$"))
    return std::nullopt;
  return classType.getClassName().drop_front();
}

bool hasBase(const ProtocolInfo &info, llvm::StringRef baseName) {
  return llvm::any_of(info.bases, [&](const ProtocolBase &base) {
    return base.name == baseName;
  });
}

py::CallableType
substituteSignature(py::CallableType signature,
                    const std::map<std::string, mlir::Type> &binding);

mlir::Type substitute(mlir::Type type,
                      const std::map<std::string, mlir::Type> &binding) {
  if (!type)
    return type;
  if (mlir::isa<py::SelfType>(type)) {
    auto found = binding.find("Self");
    return found == binding.end() ? type : found->second;
  }
  if (std::optional<llvm::StringRef> variable = typeVariableName(type)) {
    auto found = binding.find(variable->str());
    return found == binding.end() ? type : found->second;
  }
  if (auto classType = mlir::dyn_cast<py::ClassType>(type)) {
    auto found = binding.find(classType.getClassName().str());
    return found == binding.end() ? type : found->second;
  }
  if (auto contract = mlir::dyn_cast<py::ContractType>(type)) {
    llvm::SmallVector<mlir::Type> args;
    args.reserve(contract.getArguments().size());
    for (mlir::Type arg : contract.getArguments()) {
      mlir::Type substituted = substitute(arg, binding);
      if (!substituted)
        return {};
      args.push_back(substituted);
    }
    return py::ContractType::get(type.getContext(), contract.getContractName(),
                                 args);
  }
  if (auto unpack = mlir::dyn_cast<py::UnpackType>(type)) {
    mlir::Type packed = substitute(unpack.getPackedType(), binding);
    return packed ? py::UnpackType::get(type.getContext(), packed)
                  : mlir::Type();
  }
  if (auto tuple = mlir::dyn_cast<py::TupleType>(type)) {
    llvm::SmallVector<mlir::Type> elements;
    for (mlir::Type element : tuple.getElementTypes()) {
      mlir::Type substituted = substitute(element, binding);
      if (!substituted)
        return {};
      elements.push_back(substituted);
    }
    return py::TupleType::get(type.getContext(), elements);
  }
  if (auto list = mlir::dyn_cast<py::ListType>(type)) {
    mlir::Type element = substitute(list.getElementType(), binding);
    return element ? py::ListType::get(type.getContext(), element)
                   : mlir::Type();
  }
  if (auto dict = mlir::dyn_cast<py::DictType>(type)) {
    mlir::Type key = substitute(dict.getKeyType(), binding);
    mlir::Type value = substitute(dict.getValueType(), binding);
    return key && value ? py::DictType::get(type.getContext(), key, value)
                        : mlir::Type();
  }
  if (auto typeType = mlir::dyn_cast<py::TypeType>(type)) {
    mlir::Type instance = substitute(typeType.getInstanceType(), binding);
    return instance ? py::TypeType::get(type.getContext(), instance)
                    : mlir::Type();
  }
  if (auto protocol = mlir::dyn_cast<py::ProtocolType>(type)) {
    llvm::SmallVector<mlir::Type> args;
    for (mlir::Type arg : protocol.getArguments()) {
      mlir::Type substituted = substitute(arg, binding);
      if (!substituted)
        return {};
      args.push_back(substituted);
    }
    return py::ProtocolType::get(type.getContext(), protocol.getProtocolName(),
                                 args);
  }
  if (auto unionType = mlir::dyn_cast<py::UnionType>(type)) {
    llvm::SmallVector<mlir::Type> members;
    for (mlir::Type member : unionType.getMemberTypes()) {
      mlir::Type substituted = substitute(member, binding);
      if (!substituted)
        return {};
      members.push_back(substituted);
    }
    return py::UnionType::getNormalized(type.getContext(), members);
  }
  if (auto overload = mlir::dyn_cast<py::OverloadType>(type)) {
    llvm::SmallVector<mlir::Type> candidates;
    for (mlir::Type candidate : overload.getCandidateTypes()) {
      mlir::Type substituted = substitute(candidate, binding);
      if (!substituted)
        return {};
      if (auto nested = mlir::dyn_cast<py::OverloadType>(substituted)) {
        candidates.append(nested.getCandidateTypes().begin(),
                          nested.getCandidateTypes().end());
        continue;
      }
      if (!mlir::isa<py::CallableType>(substituted))
        return {};
      if (!llvm::is_contained(candidates, substituted))
        candidates.push_back(substituted);
    }
    if (candidates.empty())
      return {};
    if (candidates.size() == 1)
      return candidates.front();
    return py::OverloadType::get(type.getContext(), candidates);
  }
  if (auto signature = mlir::dyn_cast<py::CallableType>(type))
    return substituteSignature(signature, binding);
  if (auto asyncValue = mlir::dyn_cast<mlir::async::ValueType>(type)) {
    mlir::Type value = substitute(asyncValue.getValueType(), binding);
    return value ? mlir::async::ValueType::get(value) : mlir::Type();
  }
  return type;
}

std::optional<ProtocolMethod> callableCallContract(py::CallableType callable) {
  if (!callable)
    return std::nullopt;

  mlir::MLIRContext *context = callable.getContext();
  llvm::SmallVector<mlir::Type> positional;
  positional.push_back(py::ClassType::get(context, "Callable"));
  positional.append(callable.getPositionalTypes().begin(),
                    callable.getPositionalTypes().end());
  llvm::SmallVector<mlir::StringAttr> positionalNames;
  if (!callable.getPositionalNames().empty()) {
    positionalNames.push_back(mlir::StringAttr::get(context, "__callable__"));
    positionalNames.append(callable.getPositionalNames().begin(),
                           callable.getPositionalNames().end());
  }
  llvm::SmallVector<mlir::BoolAttr> positionalDefaults;
  if (!callable.getPositionalDefaults().empty()) {
    positionalDefaults.push_back(mlir::BoolAttr::get(context, false));
    positionalDefaults.append(callable.getPositionalDefaults().begin(),
                              callable.getPositionalDefaults().end());
  }
  mlir::Type vararg =
      callable.hasVararg() ? callable.getVarargType() : mlir::Type();
  mlir::Type kwarg =
      callable.hasKwarg() ? callable.getKwargType() : mlir::Type();
  ProtocolMethod contract;
  contract.signature = py::CallableType::get(
      context, positional, callable.getKwOnlyTypes(), vararg, kwarg,
      callable.getResultTypes(), positionalNames, callable.getKwOnlyNames(),
      positionalDefaults, callable.getKwOnlyDefaults(),
      callable.getVarargName(), callable.getKwargName(),
      callable.getPositionalOnlyCount() + 1);
  contract.mayThrow = true;
  return contract;
}

std::optional<std::map<std::string, mlir::Type>>
bindProtocolParams(const ProtocolInfo &info, llvm::ArrayRef<mlir::Type> args) {
  if (args.size() > info.params.size())
    return std::nullopt;
  std::map<std::string, mlir::Type> binding;
  for (auto [index, param] : llvm::enumerate(info.params)) {
    mlir::Type arg = index < args.size() ? args[index] : mlir::Type();
    if (!arg && index < info.paramDefaults.size())
      arg = info.paramDefaults[index];
    if (!arg)
      return std::nullopt;
    binding.emplace(param, arg);
  }
  return binding;
}

std::optional<std::vector<mlir::Type>>
completeProtocolInstantiationArguments(const ProtocolInfo &info,
                                       llvm::ArrayRef<mlir::Type> supplied) {
  if (std::optional<std::vector<mlir::Type>> direct =
          completeDirectArguments(info, supplied))
    return direct;
  for (const ProtocolShortForm &shortForm : info.shortForms)
    if (std::optional<std::vector<mlir::Type>> expanded =
            completeShortArguments(info, shortForm, supplied))
      return expanded;
  return std::nullopt;
}

std::optional<std::map<std::string, mlir::Type>>
bindProtocolInstantiation(const ProtocolInfo &info,
                          llvm::ArrayRef<mlir::Type> supplied) {
  std::optional<std::vector<mlir::Type>> args =
      completeProtocolInstantiationArguments(info, supplied);
  if (!args)
    return std::nullopt;
  return bindProtocolParams(info, *args);
}

std::optional<std::map<std::string, mlir::Type>>
bindBaseInstantiation(const std::map<std::string, ProtocolInfo> &classes,
                      const ProtocolBase &baseInfo,
                      const std::map<std::string, mlir::Type> &binding) {
  auto base = classes.find(baseInfo.name);
  if (base == classes.end())
    return std::nullopt;

  llvm::SmallVector<mlir::Type> args;
  args.reserve(baseInfo.arguments.size());
  for (mlir::Type arg : baseInfo.arguments) {
    mlir::Type substituted = substitute(arg, binding);
    if (!substituted)
      return std::nullopt;
    args.push_back(substituted);
  }

  std::optional<std::map<std::string, mlir::Type>> baseBinding =
      bindProtocolInstantiation(base->second, args);
  if (!baseBinding)
    return std::nullopt;
  auto self = binding.find("Self");
  if (self != binding.end())
    (*baseBinding)["Self"] = self->second;
  return baseBinding;
}

py::CallableType
substituteSignature(py::CallableType signature,
                    const std::map<std::string, mlir::Type> &binding) {
  llvm::SmallVector<mlir::Type> positional;
  for (mlir::Type type : signature.getPositionalTypes()) {
    mlir::Type substituted = substitute(type, binding);
    if (!substituted)
      return {};
    positional.push_back(substituted);
  }
  llvm::SmallVector<mlir::Type> results;
  for (mlir::Type type : signature.getResultTypes()) {
    mlir::Type substituted = substitute(type, binding);
    if (!substituted)
      return {};
    results.push_back(substituted);
  }
  llvm::SmallVector<mlir::Type> kwonly;
  for (mlir::Type type : signature.getKwOnlyTypes()) {
    mlir::Type substituted = substitute(type, binding);
    if (!substituted)
      return {};
    kwonly.push_back(substituted);
  }
  mlir::Type vararg;
  if (signature.hasVararg()) {
    vararg = substitute(signature.getVarargType(), binding);
    if (!vararg)
      return {};
  }
  mlir::Type kwarg;
  if (signature.hasKwarg()) {
    kwarg = substitute(signature.getKwargType(), binding);
    if (!kwarg)
      return {};
  }
  return py::CallableType::get(
      signature.getContext(), positional, kwonly, vararg, kwarg, results,
      signature.getPositionalNames(), signature.getKwOnlyNames(),
      signature.getPositionalDefaults(), signature.getKwOnlyDefaults(),
      signature.getVarargName(), signature.getKwargName(),
      signature.getPositionalOnlyCount());
}

py::CallableType bindReceiverCallableImpl(py::CallableType signature) {
  if (!signature || signature.getPositionalTypes().empty())
    return {};

  mlir::MLIRContext *ctx = signature.getContext();
  llvm::ArrayRef<mlir::Type> positionalTail =
      signature.getPositionalTypes().drop_front();
  llvm::SmallVector<mlir::Type> positional(positionalTail.begin(),
                                           positionalTail.end());

  llvm::SmallVector<mlir::StringAttr> positionalNames;
  if (!signature.getPositionalNames().empty()) {
    if (signature.getPositionalNames().size() !=
        signature.getPositionalTypes().size())
      return {};
    auto names = signature.getPositionalNames().drop_front();
    positionalNames.append(names.begin(), names.end());
  }

  llvm::SmallVector<mlir::BoolAttr> positionalDefaults;
  if (!signature.getPositionalDefaults().empty()) {
    if (signature.getPositionalDefaults().size() !=
        signature.getPositionalTypes().size())
      return {};
    auto defaults = signature.getPositionalDefaults().drop_front();
    positionalDefaults.append(defaults.begin(), defaults.end());
  }

  unsigned positionalOnlyCount = signature.getPositionalOnlyCount();
  if (positionalOnlyCount != 0)
    --positionalOnlyCount;

  return py::CallableType::get(
      ctx, positional, signature.getKwOnlyTypes(),
      signature.hasVararg() ? signature.getVarargType() : mlir::Type(),
      signature.hasKwarg() ? signature.getKwargType() : mlir::Type(),
      signature.getResultTypes(), positionalNames, signature.getKwOnlyNames(),
      positionalDefaults, signature.getKwOnlyDefaults(),
      signature.getVarargName(), signature.getKwargName(), positionalOnlyCount);
}

// Binds a concrete dialect type to its manifest class and parameter map.
std::optional<std::pair<std::string, std::map<std::string, mlir::Type>>>
bindConcrete(mlir::Type type) {
  if (mlir::isa<py::ObjectType>(type))
    return {{"object", {}}};
  if (mlir::isa<py::BoolType>(type))
    return {{"bool", {}}};
  if (mlir::isa<py::IntType>(type))
    return {{"int", {}}};
  if (mlir::isa<py::FloatType>(type))
    return {{"float", {}}};
  if (auto list = mlir::dyn_cast<py::ListType>(type))
    return {{"list", {{"T", list.getElementType()}}}};
  if (mlir::isa<py::StrType>(type))
    return {{"str", {}}};
  if (auto tuple = mlir::dyn_cast<py::TupleType>(type)) {
    llvm::ArrayRef<mlir::Type> elements = tuple.getElementTypes();
    if (elements.empty())
      return {{"tuple", {{"T", py::ObjectType::get(type.getContext())}}}};
    mlir::Type elementType = elements.front();
    for (mlir::Type element : elements)
      if (element != elementType) {
        elementType = py::UnionType::getNormalized(type.getContext(), elements);
        break;
      }
    return {{"tuple", {{"T", elementType}}}};
  }
  if (auto dict = mlir::dyn_cast<py::DictType>(type))
    return {{"dict", {{"K", dict.getKeyType()}, {"V", dict.getValueType()}}}};
  if (mlir::Type element = py::primitiveIteratorStateElementType(type))
    return {{"Iterator", {{"T", element}}}};
  if (auto asyncValue = mlir::dyn_cast<mlir::async::ValueType>(type))
    return {{"Awaitable", {{"T", asyncValue.getValueType()}}}};
  return std::nullopt;
}

bool isIntegerLiteralSpelling(llvm::StringRef spelling) {
  if (spelling.empty())
    return false;
  if (spelling.front() == '-')
    spelling = spelling.drop_front();
  return !spelling.empty() &&
         llvm::all_of(spelling, [](char ch) { return ch >= '0' && ch <= '9'; });
}

std::optional<std::string> manifestClassNameForLiteral(py::LiteralType literal) {
  llvm::StringRef spelling = literal.getSpelling();
  if (spelling == "True" || spelling == "False")
    return std::string("bool");
  if (spelling == "None")
    return std::string("NoneType");
  if (spelling.starts_with("\"") && spelling.ends_with("\""))
    return std::string("str");
  if (isIntegerLiteralSpelling(spelling))
    return std::string("int");
  return std::nullopt;
}

std::string manifestClassNameForContract(llvm::StringRef name) {
  for (llvm::StringRef prefix :
       {"builtins.", "typing.", "types.", "contextlib.", "_asyncio.",
        "asyncio.", "contextvars.", "ctypes.", "_ctypes.", "_typeshed."}) {
    if (name.consume_front(prefix))
      return name.str();
  }
  return name.str();
}

std::optional<std::pair<std::string, std::map<std::string, mlir::Type>>>
bindReceiver(mlir::Type type,
             const std::map<std::string, ProtocolInfo> &classes) {
  if (auto concrete = bindConcrete(type))
    return concrete;

  if (auto literal = mlir::dyn_cast<py::LiteralType>(type)) {
    std::optional<std::string> className = manifestClassNameForLiteral(literal);
    if (className && classes.find(*className) != classes.end())
      return {{std::move(*className), {}}};
  }

  if (auto contract = mlir::dyn_cast<py::ContractType>(type)) {
    std::string className =
        manifestClassNameForContract(contract.getContractName());
    auto found = classes.find(className);
    if (found != classes.end()) {
      std::optional<std::map<std::string, mlir::Type>> binding =
          bindProtocolInstantiation(found->second, contract.getArguments());
      if (binding)
        return {{std::move(className), *binding}};
    }
  }

  if (auto typeObject = mlir::dyn_cast<py::TypeType>(type)) {
    auto found = classes.find("type");
    if (found != classes.end()) {
      std::optional<std::map<std::string, mlir::Type>> binding =
          bindProtocolInstantiation(found->second, {});
      if (binding) {
        (*binding)["T"] = typeObject.getInstanceType();
        return {{"type", *binding}};
      }
    }
  }

  if (auto classType = mlir::dyn_cast<py::ClassType>(type)) {
    auto found = classes.find(classType.getClassName().str());
    if (found != classes.end()) {
      std::optional<std::map<std::string, mlir::Type>> binding =
          bindProtocolInstantiation(found->second, {});
      if (binding)
        return {{classType.getClassName().str(), *binding}};
    }
  }

  auto protocol = mlir::dyn_cast<py::ProtocolType>(type);
  if (!protocol)
    return std::nullopt;
  auto found = classes.find(protocol.getProtocolName().str());
  if (found == classes.end() || !found->second.isProtocol)
    return std::nullopt;
  std::optional<std::map<std::string, mlir::Type>> binding =
      bindProtocolInstantiation(found->second, protocol.getArguments());
  if (!binding)
    return std::nullopt;
  return {{protocol.getProtocolName().str(), *binding}};
}

std::map<std::string, mlir::Type>
withSelfBinding(const std::map<std::string, mlir::Type> &binding,
                mlir::Type receiverType) {
  std::map<std::string, mlir::Type> result = binding;
  result.emplace("Self", receiverType);
  return result;
}

std::optional<std::vector<mlir::Type>>
protocolArgumentsForImpl(const std::map<std::string, ProtocolInfo> &classes,
                         mlir::Type receiverType,
                         llvm::StringRef protocolName) {
  auto target = classes.find(protocolName.str());
  if (target == classes.end() || !target->second.isProtocol)
    return std::nullopt;
  auto binding = bindReceiver(receiverType, classes);
  if (!binding)
    return std::nullopt;

  std::function<std::optional<std::vector<mlir::Type>>(
      llvm::StringRef, const std::map<std::string, mlir::Type> &, unsigned)>
      walk = [&](llvm::StringRef className,
                 const std::map<std::string, mlir::Type> &currentBinding,
                 unsigned depth) -> std::optional<std::vector<mlir::Type>> {
    if (depth > 16)
      return std::nullopt;
    auto found = classes.find(className.str());
    if (found == classes.end())
      return std::nullopt;
    const ProtocolInfo &info = found->second;

    if (className == protocolName) {
      std::vector<mlir::Type> args;
      args.reserve(info.params.size());
      for (const std::string &param : info.params) {
        auto arg = currentBinding.find(param);
        if (arg == currentBinding.end())
          return std::nullopt;
        args.push_back(arg->second);
      }
      return args;
    }

    for (const ProtocolBase &baseInfo : info.bases) {
      std::optional<std::map<std::string, mlir::Type>> baseBinding =
          bindBaseInstantiation(classes, baseInfo, currentBinding);
      if (!baseBinding)
        continue;
      if (std::optional<std::vector<mlir::Type>> result =
              walk(baseInfo.name, *baseBinding, depth + 1))
        return result;
    }
    return std::nullopt;
  };

  return walk(binding->first, binding->second, 0);
}

Variance parseVariance(llvm::StringRef variance) {
  if (variance == "invariant")
    return Variance::Invariant;
  if (variance == "contravariant")
    return Variance::Contravariant;
  return Variance::Covariant;
}

bool sameProtocolEvidence(const std::optional<ProtocolEvidence> &lhs,
                          const std::optional<ProtocolEvidence> &rhs) {
  if (lhs.has_value() != rhs.has_value())
    return false;
  if (!lhs)
    return true;
  return lhs->manifestClass == rhs->manifestClass &&
         lhs->binding == rhs->binding && lhs->info == rhs->info;
}

bool sameContractResolution(const ContractResolution &lhs,
                            const ContractResolution &rhs) {
  return sameMethodContract(lhs.method, rhs.method) &&
         lhs.methodName == rhs.methodName &&
         lhs.typeBindings == rhs.typeBindings &&
         sameProtocolEvidence(lhs.receiverEvidence, rhs.receiverEvidence);
}

} // namespace

py::CallableType bindReceiverCallable(py::CallableType signature) {
  return bindReceiverCallableImpl(signature);
}

namespace {

std::mutex &tableMutex() {
  static std::mutex mutex;
  return mutex;
}

std::map<mlir::MLIRContext *, std::unique_ptr<Table>> &tableSlots() {
  static std::map<mlir::MLIRContext *, std::unique_ptr<Table>> tables;
  return tables;
}

} // namespace

const Table &Table::get(mlir::MLIRContext &context) {
  std::lock_guard<std::mutex> lock(tableMutex());
  std::unique_ptr<Table> &slot = tableSlots()[&context];
  if (slot)
    return *slot;
  slot = std::make_unique<Table>();

  llvm::ArrayRef<ManifestSource> manifests = manifestSources();
  if (manifests.empty()) {
    llvm::errs() << "warning: typing manifest sources are missing; protocol "
                    "conformance queries are empty\n";
    return *slot;
  }

  for (const ManifestSource &entry : manifests) {
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(
        llvm::MemoryBuffer::getMemBuffer(
            llvm::StringRef(entry.data, entry.size), entry.name,
            /*RequiresNullTerminator=*/false),
        llvm::SMLoc());
    mlir::OwningOpRef<mlir::ModuleOp> manifest =
        mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
    if (!manifest) {
      llvm::errs() << "warning: failed to parse typing manifest source '"
                   << entry.name << "'\n";
      continue;
    }

    for (mlir::Operation &op : manifest->getBody()->getOperations()) {
      auto classOp = mlir::dyn_cast<py::ClassOp>(&op);
      if (!classOp)
        continue;
      ProtocolInfo classInfo;
      classInfo.params = stringArray(classOp, "ly.typing.params");
      classInfo.paramVariance =
          stringArray(classOp, "ly.typing.param_variance");
      classInfo.paramVariance.resize(classInfo.params.size(), "covariant");
      classInfo.paramDefaults = trailingTypeDefaults(
          classOp, "ly.typing.param_defaults", classInfo.params.size());
      ProtocolShortForm shortForm{
          unsignedArray(classOp, "ly.typing.short_arg_positions"),
          typeArray(classOp, "ly.typing.short_arg_defaults")};
      if (!shortForm.positions.empty())
        classInfo.shortForms.push_back(std::move(shortForm));
      classInfo.isProtocol = hasMarker(classOp, "ly.typing.protocol");
      classInfo.isAbstract = hasMarker(classOp, "ly.typing.abstract");
      classInfo.isFinal = hasMarker(classOp, "ly.typing.final");
      std::vector<std::string> baseNames;
      if (auto bases = classOp.getBaseNamesAttr())
        for (mlir::Attribute base : bases)
          if (auto text = mlir::dyn_cast<mlir::StringAttr>(base))
            baseNames.push_back(text.getValue().str());
      std::vector<std::vector<mlir::Type>> baseArgs;
      if (auto args =
              classOp->getAttrOfType<mlir::ArrayAttr>("ly.typing.base_args")) {
        for (mlir::Attribute group : args) {
          std::vector<mlir::Type> types;
          if (auto inner = mlir::dyn_cast<mlir::ArrayAttr>(group))
            for (mlir::Attribute element : inner)
              if (auto typeAttr = mlir::dyn_cast<mlir::TypeAttr>(element))
                types.push_back(typeAttr.getValue());
          baseArgs.push_back(std::move(types));
        }
      }
      baseArgs.resize(baseNames.size());
      for (auto [baseName, args] : llvm::zip(baseNames, baseArgs))
        classInfo.bases.push_back(ProtocolBase{baseName, std::move(args)});
      if (auto fieldNames = classOp.getFieldNamesAttr()) {
        mlir::ArrayAttr fieldTypes = classOp.getFieldContractTypesAttr();
        if (!fieldTypes)
          fieldTypes = classOp.getFieldTypesAttr();
        if (fieldTypes && fieldNames.size() == fieldTypes.size()) {
          for (auto [nameAttr, typeAttr] : llvm::zip(fieldNames, fieldTypes)) {
            auto name = mlir::dyn_cast<mlir::StringAttr>(nameAttr);
            auto type = mlir::dyn_cast<mlir::TypeAttr>(typeAttr);
            if (name && type)
              classInfo.fields[name.getValue().str()] = type.getValue();
          }
        }
      }
      if (auto methodNames = classOp.getMethodNamesAttr()) {
        mlir::ArrayAttr methodContracts = classOp.getMethodContractsAttr();
        if (methodContracts && methodNames.size() == methodContracts.size()) {
          for (auto [nameAttr, typeAttr] :
               llvm::zip(methodNames, methodContracts)) {
            auto name = mlir::dyn_cast<mlir::StringAttr>(nameAttr);
            auto type = mlir::dyn_cast<mlir::TypeAttr>(typeAttr);
            if (!name || !type)
              continue;

            auto pushSignature = [&](py::CallableType signature) {
              if (!signature)
                return;
              ProtocolMethod methodInfo;
              methodInfo.signature = signature;
              methodInfo.mayThrow = true;
              classInfo.methods[name.getValue().str()].push_back(methodInfo);
            };

            if (auto signature =
                    mlir::dyn_cast<py::CallableType>(type.getValue())) {
              pushSignature(signature);
              continue;
            }
            if (auto overload =
                    mlir::dyn_cast<py::OverloadType>(type.getValue())) {
              for (mlir::Type candidate : overload.getCandidateTypes())
                pushSignature(mlir::dyn_cast<py::CallableType>(candidate));
            }
          }
        }
      }
      slot->classes.emplace(classOp.getSymName().str(), std::move(classInfo));
    }
  }
  const ProtocolInfo *root = slot->lookup("Protocol");
  if (!root || !root->isProtocol) {
    llvm::errs() << "warning: typing manifest does not declare a Protocol "
                    "root marked ly.typing.protocol\n";
  }
  for (const auto &[name, info] : slot->classes) {
    if (!info.isProtocol || name == "Protocol")
      continue;
    if (!hasBase(info, "Protocol"))
      llvm::errs() << "warning: typing protocol '" << name
                   << "' does not directly inherit Protocol\n";
  }
  return *slot;
}

Table &Table::getMutable(mlir::MLIRContext &context) {
  (void)get(context);
  std::lock_guard<std::mutex> lock(tableMutex());
  return *tableSlots()[&context];
}

void Table::registerClass(llvm::StringRef name, ProtocolInfo info) {
  classes[name.str()] = std::move(info);
}

bool Table::collectMethodContractsIn(
    llvm::StringRef className, const std::map<std::string, mlir::Type> &binding,
    llvm::StringRef methodName, unsigned depth,
    std::vector<ProtocolMethod> &out) const {
  if (depth > 16)
    return false;
  auto found = classes.find(className.str());
  if (found == classes.end())
    return false;
  const ProtocolInfo &entry = found->second;

  auto method = entry.methods.find(methodName.str());
  if (method != entry.methods.end()) {
    std::map<std::string, mlir::Type> methodBinding = binding;
    auto self = binding.find("Self");
    if (self != binding.end())
      methodBinding.emplace(className.str(), self->second);
    for (ProtocolMethod overload : method->second) {
      overload.signature =
          substituteSignature(overload.signature, methodBinding);
      if (overload.signature)
        out.push_back(overload);
    }
    return true;
  }

  for (const ProtocolBase &baseInfo : entry.bases) {
    std::optional<std::map<std::string, mlir::Type>> baseBinding =
        bindBaseInstantiation(classes, baseInfo, binding);
    if (!baseBinding)
      continue;
    if (collectMethodContractsIn(baseInfo.name, *baseBinding, methodName,
                                 depth + 1, out))
      return true;
  }
  return false;
}

std::optional<FieldResolution> Table::collectFieldResolutionIn(
    llvm::StringRef className, const std::map<std::string, mlir::Type> &binding,
    llvm::StringRef fieldName, unsigned depth) const {
  if (depth > 16)
    return std::nullopt;
  auto found = classes.find(className.str());
  if (found == classes.end())
    return std::nullopt;
  const ProtocolInfo &entry = found->second;

  auto field = entry.fields.find(fieldName.str());
  if (field != entry.fields.end()) {
    mlir::Type substituted = substitute(field->second, binding);
    if (substituted)
      return FieldResolution{substituted, fieldName.str(), binding,
                             std::nullopt};
  }

  for (const ProtocolBase &baseInfo : entry.bases) {
    std::optional<std::map<std::string, mlir::Type>> baseBinding =
        bindBaseInstantiation(classes, baseInfo, binding);
    if (!baseBinding)
      continue;
    if (std::optional<FieldResolution> inherited = collectFieldResolutionIn(
            baseInfo.name, *baseBinding, fieldName, depth + 1))
      return inherited;
  }
  return std::nullopt;
}

std::vector<ProtocolMethod>
Table::collectReceiverMethodContracts(mlir::Type receiverType,
                                      llvm::StringRef methodName) const {
  std::vector<ProtocolMethod> result;
  if (methodName == "__call__") {
    auto callable = mlir::dyn_cast<py::CallableType>(receiverType);
    if (std::optional<ProtocolMethod> contract = callableCallContract(callable))
      result.push_back(*contract);
    if (!result.empty())
      return result;
  }

  auto binding = bindReceiver(receiverType, classes);
  if (!binding)
    return result;
  std::map<std::string, mlir::Type> selfBinding =
      withSelfBinding(binding->second, receiverType);
  collectMethodContractsIn(binding->first, selfBinding, methodName, 0, result);
  return result;
}

std::vector<ContractResolution>
Table::methodContractCandidatesWithEvidence(mlir::Type receiverType,
                                            llvm::StringRef methodName) const {
  std::vector<ContractResolution> result;
  std::optional<ProtocolEvidence> receiverEvidence = evidenceFor(receiverType);
  std::map<std::string, mlir::Type> receiverBinding =
      receiverEvidence
          ? withSelfBinding(receiverEvidence->binding, receiverType)
          : std::map<std::string, mlir::Type>{};

  for (ProtocolMethod method :
       collectReceiverMethodContracts(receiverType, methodName)) {
    result.push_back(ContractResolution{std::move(method), methodName.str(),
                                        receiverBinding, receiverEvidence,
                                        /*score=*/0});
  }
  return result;
}

std::optional<FieldResolution>
Table::resolveFieldContractWithEvidence(mlir::Type receiverType,
                                        llvm::StringRef fieldName) const {
  std::optional<ProtocolEvidence> receiverEvidence = evidenceFor(receiverType);
  if (!receiverEvidence)
    return std::nullopt;
  std::map<std::string, mlir::Type> selfBinding =
      withSelfBinding(receiverEvidence->binding, receiverType);
  std::optional<FieldResolution> resolution = collectFieldResolutionIn(
      receiverEvidence->manifestClass, selfBinding, fieldName, 0);
  if (!resolution)
    return std::nullopt;
  resolution->receiverEvidence = std::move(receiverEvidence);
  return resolution;
}

std::optional<std::vector<mlir::Type>>
Table::protocolArgumentsFor(mlir::Type receiverType,
                            llvm::StringRef protocolName) const {
  return protocolArgumentsForImpl(classes, receiverType, protocolName);
}

std::optional<std::vector<mlir::Type>>
Table::completeProtocolArguments(llvm::StringRef protocolName,
                                 llvm::ArrayRef<mlir::Type> supplied) const {
  const ProtocolInfo *info = lookup(protocolName);
  if (!info || !info->isProtocol)
    return std::nullopt;
  return completeProtocolInstantiationArguments(*info, supplied);
}

Variance Table::parameterVariance(llvm::StringRef protocolName,
                                  unsigned index) const {
  const ProtocolInfo *info = lookup(protocolName);
  if (!info || index >= info->paramVariance.size())
    return Variance::Covariant;
  return parseVariance(info->paramVariance[index]);
}

bool Table::isProtocolSubtypeOf(
    py::ProtocolType subtype, py::ProtocolType supertype,
    llvm::function_ref<bool(mlir::Type, mlir::Type, Variance)> argumentMatches)
    const {
  const ProtocolInfo *targetInfo = lookup(supertype.getProtocolName());
  if (!targetInfo || !targetInfo->isProtocol)
    return false;

  std::optional<std::vector<mlir::Type>> targetArgs =
      completeProtocolInstantiationArguments(*targetInfo,
                                             supertype.getArguments());
  if (!targetArgs)
    return false;

  auto binding = bindReceiver(subtype, classes);
  if (!binding)
    return false;

  auto walk = [&](auto &&self, llvm::StringRef className,
                  const std::map<std::string, mlir::Type> &currentBinding,
                  unsigned depth) -> bool {
    if (depth > 16)
      return false;
    auto found = classes.find(className.str());
    if (found == classes.end())
      return false;
    const ProtocolInfo &info = found->second;

    if (className == supertype.getProtocolName()) {
      std::vector<mlir::Type> sourceArgs;
      sourceArgs.reserve(info.params.size());
      for (const std::string &param : info.params) {
        auto arg = currentBinding.find(param);
        if (arg == currentBinding.end())
          return false;
        sourceArgs.push_back(arg->second);
      }
      if (sourceArgs.size() != targetArgs->size())
        return false;
      for (auto indexed : llvm::enumerate(llvm::zip(sourceArgs, *targetArgs))) {
        auto [sourceArg, targetArg] = indexed.value();
        if (!argumentMatches(sourceArg, targetArg,
                             parameterVariance(supertype.getProtocolName(),
                                               indexed.index())))
          return false;
      }
      return true;
    }

    for (const ProtocolBase &baseInfo : info.bases) {
      std::optional<std::map<std::string, mlir::Type>> baseBinding =
          bindBaseInstantiation(classes, baseInfo, currentBinding);
      if (!baseBinding)
        continue;
      if (self(self, baseInfo.name, *baseBinding, depth + 1))
        return true;
    }
    return false;
  };

  return walk(walk, binding->first, binding->second, 0);
}

std::optional<ProtocolEvidence>
Table::evidenceFor(mlir::Type receiverType) const {
  auto binding = bindReceiver(receiverType, classes);
  if (!binding)
    return std::nullopt;
  const ProtocolInfo *info = lookup(binding->first);
  if (!info)
    return std::nullopt;
  return ProtocolEvidence{binding->first, std::move(binding->second), info};
}

bool Table::isManifestSubclassOf(mlir::Type receiverType,
                                 llvm::StringRef baseClassName) const {
  std::optional<ProtocolEvidence> evidence = evidenceFor(receiverType);
  if (!evidence)
    return false;
  std::string target = manifestClassNameForContract(baseClassName);

  auto walk = [&](auto &&self, llvm::StringRef className,
                  unsigned depth) -> bool {
    if (depth > 32)
      return false;
    if (className == target)
      return true;
    auto found = classes.find(className.str());
    if (found == classes.end())
      return false;
    for (const ProtocolBase &base : found->second.bases)
      if (self(self, base.name, depth + 1))
        return true;
    return false;
  };

  return walk(walk, evidence->manifestClass, 0);
}

static mlir::Type awaitIteratorPayloadType(mlir::Type iteratorType) {
  auto generator = mlir::dyn_cast_if_present<py::ProtocolType>(iteratorType);
  if (!generator || generator.getProtocolName() != "Generator" ||
      generator.getArguments().size() != 3)
    return {};
  return generator.getArguments()[2];
}

std::optional<AwaitableResolution>
Table::resolveAwaitableWithEvidence(mlir::Type type) const {
  if (!type)
    return std::nullopt;

  auto selection = lython::selection::bestCandidate<ContractResolution>(
      [](const ContractResolution &candidate) { return candidate.score; },
      [](const ContractResolution &lhs, const ContractResolution &rhs) {
        return sameContractResolution(lhs, rhs);
      });
  for (ContractResolution candidate :
       methodContractCandidatesWithEvidence(type, "__await__")) {
    std::optional<py::CallableSignatureShape> shape =
        py::callableSignatureShape(candidate.method.signature,
                                   /*firstParameter=*/1);
    if (!shape)
      continue;
    lython::callable::InvocationSpecificityScore observer;
    bool matched = py::matchCallableInvocationWithObserver(
        *shape, llvm::ArrayRef<mlir::Type>{}, llvm::ArrayRef<KeywordArgument>{},
        [](mlir::Type, mlir::Type) { return true; },
        [](const KeywordArgument &keyword) -> llvm::StringRef {
          return keyword.name;
        },
        [](const KeywordArgument &keyword) -> mlir::Type {
          return keyword.type;
        },
        observer);
    if (!matched)
      continue;
    candidate.score = observer.score;
    selection.consider(std::move(candidate));
  }
  if (std::optional<ContractResolution> resolution =
          std::move(selection).finish()) {
    llvm::ArrayRef<mlir::Type> results =
        resolution->method.signature.getResultTypes();
    if (results.size() == 1)
      if (mlir::Type payload = awaitIteratorPayloadType(results.front()))
        return AwaitableResolution{payload, std::move(resolution)};
  }

  if (auto asyncValue = mlir::dyn_cast<mlir::async::ValueType>(type))
    return AwaitableResolution{asyncValue.getValueType(), std::nullopt};
  if (mlir::Type payload = awaitableDescriptorPayloadType(type))
    return AwaitableResolution{payload, std::nullopt};

  return std::nullopt;
}

mlir::Type Table::awaitablePayloadType(mlir::Type type) const {
  std::optional<AwaitableResolution> resolution =
      resolveAwaitableWithEvidence(type);
  if (resolution)
    return resolution->payloadType;
  return {};
}

const ProtocolInfo *Table::lookup(llvm::StringRef name) const {
  auto found = classes.find(name.str());
  return found == classes.end() ? nullptr : &found->second;
}

bool Table::isProtocol(llvm::StringRef name) const {
  const ProtocolInfo *info = lookup(name);
  return info && info->isProtocol;
}

} // namespace py::protocols
