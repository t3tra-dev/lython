#include "PyProtocols.h"

#include "CallableArgumentMatcher.h"
#include "CandidateSelection.h"
#include "PyCallableShape.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <set>
#include <tuple>
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

std::optional<std::pair<std::string, std::string>>
splitQualifiedExport(llvm::StringRef qualifiedName) {
  auto split = qualifiedName.rsplit('.');
  if (split.first.empty() || split.second.empty())
    return std::nullopt;
  return {{split.first.str(), split.second.str()}};
}

std::optional<std::tuple<std::string, std::string, std::string>>
parseClassExportSpec(llvm::StringRef spec) {
  auto assignment = spec.split('=');
  if (assignment.first.empty() || assignment.second.empty())
    return std::nullopt;
  std::optional<std::pair<std::string, std::string>> qualified =
      splitQualifiedExport(assignment.first);
  if (!qualified)
    return std::nullopt;
  return {{qualified->first, qualified->second, assignment.second.str()}};
}

std::optional<std::string> stringAttr(mlir::Operation *op,
                                      llvm::StringRef name) {
  if (auto attr = op->getAttrOfType<mlir::StringAttr>(name))
    return attr.getValue().str();
  return std::nullopt;
}

std::string manifestClassContractName(py::ClassOp classOp,
                                      llvm::StringRef moduleName) {
  if (std::optional<std::string> contract =
          stringAttr(classOp, "ly.typing.contract"))
    return *contract;
  if (std::optional<std::string> contract =
          stringAttr(classOp, "ly.runtime.contract"))
    return *contract;
  if (std::optional<std::string> contract =
          stringAttr(classOp, "ly.typeshed.contract"))
    return *contract;
  if (!moduleName.empty())
    return (llvm::Twine(moduleName) + "." + classOp.getSymName()).str();
  return classOp.getSymName().str();
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

// Type variable occurrences are manifest contracts whose name starts with '$'.
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
  return std::nullopt;
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
  return type;
}

std::optional<ProtocolMethod> callableCallContract(py::CallableType callable) {
  if (!callable)
    return std::nullopt;

  mlir::MLIRContext *context = callable.getContext();
  llvm::SmallVector<mlir::Type> positional;
  positional.push_back(py::ContractType::get(context, "builtins.function"));
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
  if (auto contract = mlir::dyn_cast_if_present<py::ContractType>(type)) {
    llvm::StringRef name = contract.getContractName();
    llvm::ArrayRef<mlir::Type> arguments = contract.getArguments();
    if (name == "builtins.object")
      return {{"object", {}}};
    if (name == "builtins.bool")
      return {{"bool", {}}};
    if (name == "builtins.int")
      return {{"int", {}}};
    if (name == "builtins.float")
      return {{"float", {}}};
    if (name == "builtins.str")
      return {{"str", {}}};
    if (name == "types.NoneType")
      return {{"NoneType", {}}};
    if (name == "builtins.list" && arguments.size() == 1)
      return {{"list", {{"T", arguments.front()}}}};
    if (name == "builtins.dict" && arguments.size() == 2)
      return {{"dict", {{"K", arguments[0]}, {"V", arguments[1]}}}};
    if (name != "builtins.tuple")
      return std::nullopt;
    llvm::ArrayRef<mlir::Type> elements = arguments;
    if (elements.empty())
      return {{"tuple", {{"T", py::pyObjectContractType(type.getContext())}}}};
    mlir::Type elementType = elements.front();
    for (mlir::Type element : elements)
      if (element != elementType) {
        elementType = py::UnionType::getNormalized(type.getContext(), elements);
        break;
      }
    return {{"tuple", {{"T", elementType}}}};
  }
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

    llvm::StringRef moduleName;
    if (auto moduleAttr = manifest->getOperation()
                              ->getAttrOfType<mlir::StringAttr>(
                                  "ly.typing.module"))
      moduleName = moduleAttr.getValue();

    bool hasExplicitClassExports = false;
    if (auto exports = manifest->getOperation()
                           ->getAttrOfType<mlir::ArrayAttr>(
                               "ly.typing.class_exports")) {
      hasExplicitClassExports = true;
      for (mlir::Attribute element : exports) {
        auto spec = mlir::dyn_cast<mlir::StringAttr>(element);
        if (!spec)
          continue;
        std::optional<std::tuple<std::string, std::string, std::string>>
            parsed = parseClassExportSpec(spec.getValue());
        if (!parsed) {
          llvm::errs() << "warning: ignored malformed class export '"
                       << spec.getValue() << "' in typing manifest '"
                       << entry.name << "'\n";
          continue;
        }
        auto &[module, exportedName, contract] = *parsed;
        slot->classExportsByModule[module][exportedName] = contract;
      }
    }

    if (auto exports = manifest->getOperation()
                           ->getAttrOfType<mlir::ArrayAttr>(
                               "ly.typing.callable_exports")) {
      for (mlir::Attribute element : exports) {
        auto spec = mlir::dyn_cast<mlir::StringAttr>(element);
        if (!spec)
          continue;
        std::optional<std::pair<std::string, std::string>> parsed =
            splitQualifiedExport(spec.getValue());
        if (!parsed) {
          llvm::errs() << "warning: ignored malformed callable export '"
                       << spec.getValue() << "' in typing manifest '"
                       << entry.name << "'\n";
          continue;
        }
        slot->callableExportsByModule[parsed->first].push_back(parsed->second);
      }
    }

    // Manifest-declared Callable contracts for free (module-level / builtin)
    // functions: parallel `ly.typing.function_names` (fully-qualified string
    // names) and `ly.typing.function_contracts` (Callable contract types). This
    // is the manifest source for function signatures that would otherwise be
    // synthesized by a C++ contract table.
    if (auto functionNames =
            manifest->getOperation()->getAttrOfType<mlir::ArrayAttr>(
                "ly.typing.function_names")) {
      auto functionContracts =
          manifest->getOperation()->getAttrOfType<mlir::ArrayAttr>(
              "ly.typing.function_contracts");
      if (!functionContracts ||
          functionContracts.size() != functionNames.size()) {
        llvm::errs() << "warning: ly.typing.function_names and "
                        "ly.typing.function_contracts must be parallel arrays "
                        "in typing manifest '"
                     << entry.name << "'\n";
      } else {
        for (auto [nameAttr, contractAttr] :
             llvm::zip(functionNames, functionContracts)) {
          auto name = mlir::dyn_cast<mlir::StringAttr>(nameAttr);
          auto contract = mlir::dyn_cast<mlir::TypeAttr>(contractAttr);
          if (!name || !contract) {
            llvm::errs() << "warning: ignored malformed free-function contract "
                            "in typing manifest '"
                         << entry.name << "'\n";
            continue;
          }
          slot->freeFunctionContracts[name.getValue().str()] =
              contract.getValue();
        }
      }
    }

    // Manifest float constants: parallel `ly.typing.float_constant_names`
    // (fully-qualified) and `ly.typing.float_constant_values` arrays.
    if (auto constantNames =
            manifest->getOperation()->getAttrOfType<mlir::ArrayAttr>(
                "ly.typing.float_constant_names")) {
      auto constantValues =
          manifest->getOperation()->getAttrOfType<mlir::ArrayAttr>(
              "ly.typing.float_constant_values");
      if (!constantValues || constantValues.size() != constantNames.size()) {
        llvm::errs() << "warning: ly.typing.float_constant_names and "
                        "ly.typing.float_constant_values must be parallel "
                        "arrays in typing manifest '"
                     << entry.name << "'\n";
      } else {
        for (auto [nameAttr, valueAttr] :
             llvm::zip(constantNames, constantValues)) {
          auto name = mlir::dyn_cast<mlir::StringAttr>(nameAttr);
          auto value = mlir::dyn_cast<mlir::FloatAttr>(valueAttr);
          if (!name || !value) {
            llvm::errs() << "warning: ignored malformed float constant in "
                            "typing manifest '"
                         << entry.name << "'\n";
            continue;
          }
          std::string qualified = name.getValue().str();
          slot->floatConstants[qualified] = value.getValueAsDouble();
          auto split = name.getValue().rsplit('.');
          slot->floatConstantsByModule[split.first.str()].push_back(
              split.second.str());
        }
      }
    }

    for (mlir::Operation &op : manifest->getBody()->getOperations()) {
      auto classOp = mlir::dyn_cast<py::ClassOp>(&op);
      if (!classOp)
        continue;
      std::string symName = classOp.getSymName().str();
      std::string contractName = manifestClassContractName(classOp, moduleName);
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
      for (const std::string &mutator :
           stringArray(classOp, "ly.typing.structural_mutators"))
        classInfo.structuralMutators.insert(mutator);
      classInfo.matchArgs = stringArray(classOp, "ly.typing.match_args");
      if (auto fieldsSpec = classOp->getAttrOfType<mlir::StringAttr>(
              "ly.typing.fields_spec")) {
        auto [attrName, viaBase] = fieldsSpec.getValue().split(':');
        classInfo.fieldsSpecAttrName = attrName.str();
        classInfo.fieldsSpecViaBase = viaBase.str();
      }
      for (const std::string &spec :
           stringArray(classOp, "ly.typing.field_param_bindings")) {
        llvm::StringRef rest(spec);
        auto [field, tail] = rest.split(':');
        auto [param, viaBase] = tail.split(':');
        if (field.empty() || param.empty() || viaBase.empty()) {
          llvm::errs() << "warning: ignored malformed field param binding '"
                       << spec << "' in class '" << symName << "'\n";
          continue;
        }
        classInfo.fieldParamBindings.push_back(
            FieldParamBinding{field.str(), param.str(), viaBase.str()});
      }
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
      std::set<std::string> aliases;
      aliases.insert(symName);
      aliases.insert(contractName);
      if (std::optional<std::string> contract =
              stringAttr(classOp, "ly.typing.contract"))
        aliases.insert(*contract);
      if (std::optional<std::string> contract =
              stringAttr(classOp, "ly.runtime.contract"))
        aliases.insert(*contract);
      if (std::optional<std::string> contract =
              stringAttr(classOp, "ly.typeshed.contract"))
        aliases.insert(*contract);
      if (!moduleName.empty())
        aliases.insert((llvm::Twine(moduleName) + "." + symName).str());

      for (const std::string &alias : aliases)
        slot->classes[alias] = classInfo;

      if (!moduleName.empty() && !hasExplicitClassExports)
        slot->classExportsByModule[moduleName.str()][symName] = contractName;
    }
  }

  for (const auto &[moduleName, exports] : slot->classExportsByModule) {
    (void)moduleName;
    for (const auto &[exportedName, contract] : exports) {
      (void)exportedName;
      if (slot->classes.find(contract) != slot->classes.end())
        continue;
      llvm::StringRef contractRef(contract);
      llvm::StringRef shortName = contractRef.rsplit('.').second;
      auto found = slot->classes.find(shortName.str());
      if (found != slot->classes.end())
        slot->classes[contract] = found->second;
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

std::optional<std::string>
Table::moduleClassExport(llvm::StringRef moduleName,
                         llvm::StringRef exportedName) const {
  auto module = classExportsByModule.find(moduleName.str());
  if (module == classExportsByModule.end())
    return std::nullopt;
  auto found = module->second.find(exportedName.str());
  if (found == module->second.end())
    return std::nullopt;
  return found->second;
}

std::optional<std::string>
Table::qualifiedClassExport(llvm::StringRef qualifiedName) const {
  std::optional<std::pair<std::string, std::string>> split =
      splitQualifiedExport(qualifiedName);
  if (!split)
    return std::nullopt;
  return moduleClassExport(split->first, split->second);
}

std::optional<std::string>
Table::bareClassExport(llvm::StringRef exportedName) const {
  for (const auto &[moduleName, exports] : classExportsByModule) {
    (void)moduleName;
    auto found = exports.find(exportedName.str());
    if (found != exports.end())
      return found->second;
  }
  return std::nullopt;
}

std::vector<std::pair<std::string, std::string>>
Table::moduleClassExports(llvm::StringRef moduleName) const {
  std::vector<std::pair<std::string, std::string>> result;
  auto module = classExportsByModule.find(moduleName.str());
  if (module == classExportsByModule.end())
    return result;
  result.reserve(module->second.size());
  for (const auto &[exportedName, contract] : module->second)
    result.push_back({exportedName, contract});
  return result;
}

bool Table::isModuleCallableExport(llvm::StringRef moduleName,
                                   llvm::StringRef exportedName) const {
  auto module = callableExportsByModule.find(moduleName.str());
  if (module == callableExportsByModule.end())
    return false;
  return llvm::is_contained(module->second, exportedName.str());
}

std::vector<std::string>
Table::moduleCallableExports(llvm::StringRef moduleName) const {
  auto module = callableExportsByModule.find(moduleName.str());
  if (module == callableExportsByModule.end())
    return {};
  return module->second;
}

std::optional<mlir::Type>
Table::freeFunctionContract(llvm::StringRef qualifiedName) const {
  auto found = freeFunctionContracts.find(qualifiedName.str());
  if (found == freeFunctionContracts.end())
    return std::nullopt;
  return found->second;
}

std::optional<double>
Table::moduleFloatConstant(llvm::StringRef qualifiedName) const {
  auto found = floatConstants.find(qualifiedName.str());
  if (found == floatConstants.end())
    return std::nullopt;
  return found->second;
}

std::vector<std::string>
Table::moduleFloatConstantExports(llvm::StringRef moduleName) const {
  auto found = floatConstantsByModule.find(moduleName.str());
  if (found == floatConstantsByModule.end())
    return {};
  return found->second;
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
  if (binding) {
    std::map<std::string, mlir::Type> selfBinding =
        withSelfBinding(binding->second, receiverType);
    collectMethodContractsIn(binding->first, selfBinding, methodName, 0, result);
  }

  // Classmethod resolution on a type object: `type[C].method(...)` also sees
  // classmethods declared on C's own class hierarchy (their first parameter
  // is `type[...]`, so the call-application shape binds it to the type-object
  // receiver; instance methods there fail that bind and are filtered out).
  if (auto typeObject = mlir::dyn_cast<py::TypeType>(receiverType)) {
    auto instanceBinding =
        bindReceiver(typeObject.getInstanceType(), classes);
    if (instanceBinding) {
      std::map<std::string, mlir::Type> selfBinding = withSelfBinding(
          instanceBinding->second, typeObject.getInstanceType());
      collectMethodContractsIn(instanceBinding->first, selfBinding, methodName,
                               0, result);
    }
  }
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

std::optional<mlir::Type>
Table::refineContractByFieldAssignment(mlir::Type receiverType,
                                       llvm::StringRef fieldName,
                                       mlir::Type valueType) const {
  auto receiver = mlir::dyn_cast_if_present<ContractType>(receiverType);
  if (!receiver)
    return std::nullopt;
  std::optional<ProtocolEvidence> evidence = evidenceFor(receiverType);
  if (!evidence || !evidence->info)
    return std::nullopt;
  const ProtocolInfo &info = *evidence->info;
  const FieldParamBinding *binding = nullptr;
  for (const FieldParamBinding &candidate : info.fieldParamBindings)
    if (candidate.field == fieldName) {
      binding = &candidate;
      break;
    }
  if (!binding)
    return std::nullopt;
  auto paramPosition =
      llvm::find(info.params, binding->param) - info.params.begin();
  if (paramPosition >= static_cast<std::ptrdiff_t>(info.params.size()))
    return std::nullopt;

  // The bound argument: literal None stays None; `type[C]` contributes the
  // single argument of C's `via_base[X]` manifest base.
  mlir::Type bound;
  if (auto literal = mlir::dyn_cast_if_present<LiteralType>(valueType)) {
    if (literal.getSpelling() == "None")
      bound = literal;
  } else if (auto typeObject =
                 mlir::dyn_cast_if_present<TypeType>(valueType)) {
    bound = conversionTypeViaBase(typeObject.getInstanceType(),
                                  binding->viaBase)
                .value_or(mlir::Type());
  }
  if (!bound)
    return std::nullopt;

  llvm::SmallVector<mlir::Type, 4> arguments(receiver.getArguments().begin(),
                                             receiver.getArguments().end());
  arguments.resize(info.params.size());
  for (auto [index, argument] : llvm::enumerate(arguments))
    if (!argument && index < info.paramDefaults.size())
      argument = info.paramDefaults[index];
  arguments[paramPosition] = bound;
  return ContractType::get(receiverType.getContext(),
                           receiver.getContractName(), arguments);
}

std::optional<mlir::Type>
Table::conversionTypeViaBase(mlir::Type instanceType,
                             llvm::StringRef viaBase) const {
  std::optional<ProtocolEvidence> evidence = evidenceFor(instanceType);
  if (!evidence || !evidence->info)
    return std::nullopt;
  for (const ProtocolBase &base : evidence->info->bases)
    if (base.name == viaBase && base.arguments.size() == 1)
      return base.arguments.front();
  return std::nullopt;
}

std::optional<std::pair<std::string, std::string>>
Table::aggregateFieldsSpec(llvm::StringRef className) const {
  std::string resolved = className.str();
  if (std::optional<std::string> qualified = qualifiedClassExport(className))
    resolved = *qualified;
  else if (std::optional<std::string> bare = bareClassExport(className))
    resolved = *bare;

  llvm::SmallVector<std::string, 8> worklist{resolved};
  for (unsigned depth = 0; depth < 16 && !worklist.empty(); ++depth) {
    llvm::SmallVector<std::string, 8> next;
    for (const std::string &name : worklist) {
      auto found = classes.find(name);
      if (found == classes.end())
        continue;
      const ProtocolInfo &info = found->second;
      if (!info.fieldsSpecAttrName.empty())
        return std::make_pair(info.fieldsSpecAttrName, info.fieldsSpecViaBase);
      for (const ProtocolBase &base : info.bases)
        next.push_back(base.name);
    }
    worklist = std::move(next);
  }
  return std::nullopt;
}

std::optional<std::vector<mlir::Type>>
Table::protocolArgumentsFor(mlir::Type receiverType,
                            llvm::StringRef protocolName) const {
  if (std::optional<std::vector<mlir::Type>> nominal =
          protocolArgumentsForImpl(classes, receiverType, protocolName))
    return nominal;

  auto oneConsistentPayload =
      [](llvm::ArrayRef<mlir::Type> candidates)
      -> std::optional<std::vector<mlir::Type>> {
    mlir::Type selected;
    for (mlir::Type candidate : candidates) {
      if (!candidate)
        continue;
      if (!selected) {
        selected = candidate;
        continue;
      }
      if (selected != candidate)
        return std::nullopt;
    }
    if (!selected)
      return std::nullopt;
    return std::vector<mlir::Type>{selected};
  };

  if (protocolName == "AsyncIterator") {
    llvm::SmallVector<mlir::Type, 2> payloads;
    for (ContractResolution candidate :
         methodContractCandidatesWithEvidence(receiverType, "__anext__")) {
      llvm::ArrayRef<mlir::Type> results =
          candidate.method.signature.getResultTypes();
      if (results.size() != 1)
        continue;
      if (mlir::Type payload = awaitablePayloadType(results.front()))
        payloads.push_back(payload);
    }
    return oneConsistentPayload(payloads);
  }

  if (protocolName == "AsyncIterable") {
    llvm::SmallVector<mlir::Type, 2> payloads;
    for (ContractResolution candidate :
         methodContractCandidatesWithEvidence(receiverType, "__aiter__")) {
      llvm::ArrayRef<mlir::Type> results =
          candidate.method.signature.getResultTypes();
      if (results.size() != 1)
        continue;
      std::optional<std::vector<mlir::Type>> iteratorArgs =
          protocolArgumentsFor(results.front(), "AsyncIterator");
      if (iteratorArgs && iteratorArgs->size() == 1)
        payloads.push_back(iteratorArgs->front());
    }
    return oneConsistentPayload(payloads);
  }

  return std::nullopt;
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

bool Table::isStructuralMutator(mlir::Type receiverType,
                                llvm::StringRef methodName) const {
  std::optional<ProtocolEvidence> evidence = evidenceFor(receiverType);
  if (!evidence || !evidence->info)
    return false;
  return evidence->info->structuralMutators.count(methodName.str()) > 0;
}

std::optional<std::vector<std::string>>
Table::matchArgsFor(mlir::Type receiverType) const {
  std::optional<ProtocolEvidence> evidence = evidenceFor(receiverType);
  if (!evidence || !evidence->info || evidence->info->matchArgs.empty())
    return std::nullopt;
  return evidence->info->matchArgs;
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

static mlir::Type awaitIteratorPayloadType(const Table &table,
                                           mlir::Type iteratorType) {
  auto generator = mlir::dyn_cast_if_present<py::ProtocolType>(iteratorType);
  if (!generator || generator.getProtocolName() != "Generator" ||
      generator.getArguments().size() != 3)
    if (std::optional<std::vector<mlir::Type>> arguments =
            table.protocolArgumentsFor(iteratorType, "Generator"))
      if (arguments->size() == 3)
        return (*arguments)[2];
  if (generator && generator.getProtocolName() == "Generator" &&
      generator.getArguments().size() == 3)
    return generator.getArguments()[2];
  return {};
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
      if (mlir::Type payload = awaitIteratorPayloadType(*this, results.front()))
        return AwaitableResolution{payload, std::move(resolution)};
  }

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
