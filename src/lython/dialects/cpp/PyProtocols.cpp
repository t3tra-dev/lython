#include "PyProtocols.h"

#include "embedded.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <mutex>

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

namespace py::protocols {
namespace {

// The typing manifest (runtime/typing.mlir) is parsed, verified, and
// bytecode-compiled into this binary at build time alongside the runtime
// object modules.
const py::runtime_library::embedded::Module *embeddedTypingModule() {
  namespace embedded = py::runtime_library::embedded;
  for (std::size_t index = 0; index < embedded::moduleCount(); ++index)
    if (llvm::StringRef(embedded::modules()[index].name) == "typing")
      return &embedded::modules()[index];
  return nullptr;
}

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

std::string methodName(mlir::Operation *op, llvm::StringRef fallback) {
  if (auto attr = op->getAttrOfType<mlir::StringAttr>("ly.typing.method_name"))
    return attr.getValue().str();
  return fallback.str();
}

// Type variable occurrences are class types whose name starts with '$'.
std::optional<llvm::StringRef> typeVariableName(mlir::Type type) {
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
  if (auto signature = mlir::dyn_cast<py::CallableType>(type))
    return substituteSignature(signature, binding);
  return type;
}

bool contractAssignable(mlir::Type expected, mlir::Type actual,
                        const std::map<std::string, ProtocolInfo> *classes);

bool protocolArgumentAssignable(
    mlir::Type expected, mlir::Type actual, llvm::StringRef variance,
    const std::map<std::string, ProtocolInfo> *classes) {
  if (variance == "contravariant")
    return contractAssignable(actual, expected, classes);
  if (variance == "invariant")
    return expected == actual;
  return contractAssignable(expected, actual, classes);
}

bool callableContractAssignable(py::CallableType expected, mlir::Type actual) {
  py::CallableType actualCallable = py::getCallableContract(actual);
  return actualCallable && py::isSubtypeOf(actualCallable, expected);
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

bool contractAssignable(mlir::Type expected, mlir::Type actual,
                        const std::map<std::string, ProtocolInfo> *classes) {
  if (expected == actual)
    return true;
  if (mlir::isa<py::ObjectType>(expected))
    return static_cast<bool>(actual);
  if (mlir::isa<py::IntType>(expected) && mlir::isa<mlir::IntegerType>(actual))
    return true;
  if (mlir::isa<py::BoolType>(expected)) {
    auto integer = mlir::dyn_cast<mlir::IntegerType>(actual);
    if (integer && integer.getWidth() == 1)
      return true;
  }
  if (mlir::isa<py::FloatType>(expected) && mlir::isa<mlir::FloatType>(actual))
    return true;
  if (auto unionType = mlir::dyn_cast<py::UnionType>(expected))
    return llvm::any_of(unionType.getMemberTypes(), [&](mlir::Type member) {
      return contractAssignable(member, actual, classes);
    });
  if (auto expectedCallable = mlir::dyn_cast<py::CallableType>(expected))
    return callableContractAssignable(expectedCallable, actual);
  auto expectedProtocol = mlir::dyn_cast<py::ProtocolType>(expected);
  auto actualProtocol = mlir::dyn_cast<py::ProtocolType>(actual);
  if (expectedProtocol && actualProtocol) {
    if (expectedProtocol.getProtocolName() != actualProtocol.getProtocolName())
      return false;
    if (expectedProtocol.getArguments().size() !=
        actualProtocol.getArguments().size())
      return false;
    const ProtocolInfo *info = nullptr;
    if (classes) {
      auto found = classes->find(expectedProtocol.getProtocolName().str());
      if (found != classes->end())
        info = &found->second;
    }
    for (auto [index, args] : llvm::enumerate(llvm::zip(
             expectedProtocol.getArguments(), actualProtocol.getArguments()))) {
      auto [expectedArg, actualArg] = args;
      llvm::StringRef variance = "covariant";
      if (info && index < info->paramVariance.size())
        variance = info->paramVariance[index];
      if (!protocolArgumentAssignable(expectedArg, actualArg, variance,
                                      classes))
        return false;
    }
    return true;
  }
  auto expectedMeta = mlir::dyn_cast<py::TypeType>(expected);
  auto actualMeta = mlir::dyn_cast<py::TypeType>(actual);
  if (expectedMeta && actualMeta)
    return contractAssignable(expectedMeta.getInstanceType(),
                              actualMeta.getInstanceType(), classes);
  return false;
}

bool bindMethodTypeVariables(mlir::Type expected, mlir::Type actual,
                             std::map<std::string, mlir::Type> &binding) {
  if (!expected || !actual)
    return false;
  if (mlir::isa<py::SelfType>(expected)) {
    auto found = binding.find("Self");
    if (found != binding.end())
      return found->second == actual;
    binding.emplace("Self", actual);
    return true;
  }
  if (std::optional<llvm::StringRef> variable = typeVariableName(expected)) {
    std::string name = variable->str();
    auto found = binding.find(name);
    if (found != binding.end())
      return found->second == actual;
    binding.emplace(std::move(name), actual);
    return true;
  }
  if (auto expectedTuple = mlir::dyn_cast<py::TupleType>(expected)) {
    auto actualTuple = mlir::dyn_cast<py::TupleType>(actual);
    if (!actualTuple || expectedTuple.getElementTypes().size() !=
                            actualTuple.getElementTypes().size())
      return false;
    bool bound = false;
    for (auto [expectedElement, actualElement] : llvm::zip(
             expectedTuple.getElementTypes(), actualTuple.getElementTypes())) {
      if (bindMethodTypeVariables(expectedElement, actualElement, binding))
        bound = true;
    }
    return bound;
  }
  if (auto expectedList = mlir::dyn_cast<py::ListType>(expected)) {
    auto actualList = mlir::dyn_cast<py::ListType>(actual);
    return actualList &&
           bindMethodTypeVariables(expectedList.getElementType(),
                                   actualList.getElementType(), binding);
  }
  if (auto expectedDict = mlir::dyn_cast<py::DictType>(expected)) {
    auto actualDict = mlir::dyn_cast<py::DictType>(actual);
    if (!actualDict)
      return false;
    bool bound = bindMethodTypeVariables(expectedDict.getKeyType(),
                                         actualDict.getKeyType(), binding);
    return bindMethodTypeVariables(expectedDict.getValueType(),
                                   actualDict.getValueType(), binding) ||
           bound;
  }
  if (auto expectedType = mlir::dyn_cast<py::TypeType>(expected)) {
    auto actualType = mlir::dyn_cast<py::TypeType>(actual);
    return actualType &&
           bindMethodTypeVariables(expectedType.getInstanceType(),
                                   actualType.getInstanceType(), binding);
  }
  if (auto expectedSignature = mlir::dyn_cast<py::CallableType>(expected)) {
    auto actualSignature = mlir::dyn_cast<py::CallableType>(actual);
    if (!actualSignature ||
        expectedSignature.getPositionalTypes().size() !=
            actualSignature.getPositionalTypes().size() ||
        expectedSignature.getKwOnlyTypes().size() !=
            actualSignature.getKwOnlyTypes().size() ||
        expectedSignature.getResultTypes().size() !=
            actualSignature.getResultTypes().size() ||
        expectedSignature.hasVararg() != actualSignature.hasVararg() ||
        expectedSignature.hasKwarg() != actualSignature.hasKwarg())
      return false;
    bool bound = false;
    for (auto [expectedType, actualType] :
         llvm::zip(expectedSignature.getPositionalTypes(),
                   actualSignature.getPositionalTypes()))
      if (bindMethodTypeVariables(expectedType, actualType, binding))
        bound = true;
    for (auto [expectedType, actualType] :
         llvm::zip(expectedSignature.getKwOnlyTypes(),
                   actualSignature.getKwOnlyTypes()))
      if (bindMethodTypeVariables(expectedType, actualType, binding))
        bound = true;
    for (auto [expectedType, actualType] :
         llvm::zip(expectedSignature.getResultTypes(),
                   actualSignature.getResultTypes()))
      if (bindMethodTypeVariables(expectedType, actualType, binding))
        bound = true;
    if (expectedSignature.hasVararg() &&
        bindMethodTypeVariables(expectedSignature.getVarargType(),
                                actualSignature.getVarargType(), binding))
      bound = true;
    if (expectedSignature.hasKwarg() &&
        bindMethodTypeVariables(expectedSignature.getKwargType(),
                                actualSignature.getKwargType(), binding))
      bound = true;
    return bound;
  }
  if (py::CallableType expectedFunc = py::getCallableContract(expected)) {
    py::CallableType actualFunc = py::getCallableContract(actual);
    return actualFunc &&
           bindMethodTypeVariables(expectedFunc, actualFunc, binding);
  }
  if (auto expectedProtocol = mlir::dyn_cast<py::ProtocolType>(expected)) {
    auto actualProtocol = mlir::dyn_cast<py::ProtocolType>(actual);
    if (!actualProtocol ||
        expectedProtocol.getProtocolName() !=
            actualProtocol.getProtocolName() ||
        expectedProtocol.getArguments().size() !=
            actualProtocol.getArguments().size())
      return false;
    bool bound = false;
    for (auto [expectedArg, actualArg] : llvm::zip(
             expectedProtocol.getArguments(), actualProtocol.getArguments())) {
      if (bindMethodTypeVariables(expectedArg, actualArg, binding))
        bound = true;
    }
    return bound;
  }
  return false;
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

bool protocolInstantiationMatches(
    const std::map<std::string, ProtocolInfo> &classes,
    llvm::StringRef actualName,
    const std::map<std::string, mlir::Type> &actualBinding,
    py::ProtocolType expected) {
  if (actualName != expected.getProtocolName())
    return false;
  auto found = classes.find(actualName.str());
  if (found == classes.end())
    return false;
  const ProtocolInfo &info = found->second;
  std::optional<std::map<std::string, mlir::Type>> expectedBinding =
      bindProtocolParams(info, expected.getArguments());
  if (!expectedBinding)
    return false;
  for (auto [index, param] : llvm::enumerate(info.params)) {
    auto actual = actualBinding.find(param);
    auto expectedArg = expectedBinding->find(param);
    if (actual == actualBinding.end() || expectedArg == expectedBinding->end())
      return false;
    llvm::StringRef variance = "covariant";
    if (index < info.paramVariance.size())
      variance = info.paramVariance[index];
    if (!protocolArgumentAssignable(expectedArg->second, actual->second,
                                    variance, &classes))
      return false;
  }
  return true;
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

// Binds a concrete dialect type to its manifest class and parameter map.
std::optional<std::pair<std::string, std::map<std::string, mlir::Type>>>
bindConcrete(mlir::Type type) {
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
    for (mlir::Type element : elements)
      if (element != elements.front())
        return std::nullopt;
    return {{"tuple", {{"T", elements.front()}}}};
  }
  if (auto dict = mlir::dyn_cast<py::DictType>(type))
    return {{"dict", {{"K", dict.getKeyType()}, {"V", dict.getValueType()}}}};
  return std::nullopt;
}

std::optional<std::pair<std::string, std::map<std::string, mlir::Type>>>
bindReceiver(mlir::Type type,
             const std::map<std::string, ProtocolInfo> &classes) {
  if (auto concrete = bindConcrete(type))
    return concrete;

  auto protocol = mlir::dyn_cast<py::ProtocolType>(type);
  if (!protocol)
    return std::nullopt;
  auto found = classes.find(protocol.getProtocolName().str());
  if (found == classes.end() || !found->second.isProtocol)
    return std::nullopt;
  std::optional<std::map<std::string, mlir::Type>> binding =
      bindProtocolParams(found->second, protocol.getArguments());
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

} // namespace

const Table &Table::get(mlir::MLIRContext &context) {
  static std::mutex mutex;
  static std::map<mlir::MLIRContext *, std::unique_ptr<Table>> tables;
  std::lock_guard<std::mutex> lock(mutex);
  std::unique_ptr<Table> &slot = tables[&context];
  if (slot)
    return *slot;
  slot = std::make_unique<Table>();

  const py::runtime_library::embedded::Module *typing = embeddedTypingModule();
  if (!typing) {
    llvm::errs() << "warning: embedded typing manifest is missing; protocol "
                    "conformance queries are empty\n";
    return *slot;
  }
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBuffer(
          llvm::StringRef(reinterpret_cast<const char *>(typing->data),
                          typing->size),
          typing->name, /*RequiresNullTerminator=*/false),
      llvm::SMLoc());
  mlir::OwningOpRef<mlir::ModuleOp> manifest =
      mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!manifest) {
    llvm::errs() << "warning: failed to parse the embedded typing manifest\n";
    return *slot;
  }

  for (mlir::Operation &op : manifest->getBody()->getOperations()) {
    auto classOp = mlir::dyn_cast<py::ClassOp>(&op);
    if (!classOp)
      continue;
    ProtocolInfo entry;
    entry.params = stringArray(classOp, "ly.typing.params");
    entry.paramVariance = stringArray(classOp, "ly.typing.param_variance");
    entry.paramVariance.resize(entry.params.size(), "covariant");
    entry.paramDefaults = trailingTypeDefaults(
        classOp, "ly.typing.param_defaults", entry.params.size());
    ProtocolShortForm shortForm{
        unsignedArray(classOp, "ly.typing.short_arg_positions"),
        typeArray(classOp, "ly.typing.short_arg_defaults")};
    if (!shortForm.positions.empty())
      entry.shortForms.push_back(std::move(shortForm));
    entry.isProtocol = hasMarker(classOp, "ly.typing.protocol");
    entry.isAbstract = hasMarker(classOp, "ly.typing.abstract");
    entry.isFinal = hasMarker(classOp, "ly.typing.final");
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
      entry.bases.push_back(ProtocolBase{baseName, std::move(args)});
    if (!classOp.getBody().empty()) {
      for (mlir::Operation &member : classOp.getBody().front()) {
        auto method = mlir::dyn_cast<py::CallableFuncOp>(&member);
        if (!method)
          continue;
        auto signature = mlir::dyn_cast<py::CallableType>(
            method.getFunctionTypeAttr().getValue());
        if (signature) {
          ProtocolMethod methodInfo;
          methodInfo.signature = signature;
          methodInfo.mayThrow = hasMarker(method, "maythrow");
          methodInfo.noThrow = hasMarker(method, "nothrow");
          entry.methods[methodName(method, method.getSymName())].push_back(
              methodInfo);
        }
      }
    }
    slot->classes.emplace(classOp.getSymName().str(), std::move(entry));
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

void Table::collectMethodContractsIn(
    llvm::StringRef className, const std::map<std::string, mlir::Type> &binding,
    llvm::StringRef methodName, unsigned depth,
    std::vector<ProtocolMethod> &out) const {
  if (depth > 16)
    return;
  auto found = classes.find(className.str());
  if (found == classes.end())
    return;
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
  }

  for (const ProtocolBase &baseInfo : entry.bases) {
    auto base = classes.find(baseInfo.name);
    if (base == classes.end())
      continue;
    llvm::SmallVector<mlir::Type> args;
    for (mlir::Type arg : baseInfo.arguments) {
      mlir::Type substituted = substitute(arg, binding);
      if (!substituted)
        return;
      args.push_back(substituted);
    }
    std::optional<std::map<std::string, mlir::Type>> baseBinding =
        bindProtocolParams(base->second, args);
    if (!baseBinding)
      return;
    auto self = binding.find("Self");
    if (self != binding.end())
      (*baseBinding)["Self"] = self->second;
    collectMethodContractsIn(baseInfo.name, *baseBinding, methodName, depth + 1,
                             out);
  }
}

std::optional<py::CallableType>
Table::methodOn(mlir::Type receiverType, llvm::StringRef methodName) const {
  if (methodName == "__call__") {
    auto callable = mlir::dyn_cast<py::CallableType>(receiverType);
    if (std::optional<ProtocolMethod> contract = callableCallContract(callable))
      return contract->signature;
  }

  auto binding = bindReceiver(receiverType, classes);
  if (!binding)
    return std::nullopt;
  std::map<std::string, mlir::Type> selfBinding =
      withSelfBinding(binding->second, receiverType);
  std::vector<ProtocolMethod> contracts;
  collectMethodContractsIn(binding->first, selfBinding, methodName, 0,
                           contracts);
  if (contracts.empty())
    return std::nullopt;
  return contracts.front().signature;
}

std::vector<py::CallableType>
Table::methodOverloadsOn(mlir::Type receiverType,
                         llvm::StringRef methodName) const {
  std::vector<py::CallableType> result;
  if (methodName == "__call__") {
    auto callable = mlir::dyn_cast<py::CallableType>(receiverType);
    if (std::optional<ProtocolMethod> contract = callableCallContract(callable))
      result.push_back(contract->signature);
    if (!result.empty())
      return result;
  }

  auto binding = bindReceiver(receiverType, classes);
  if (!binding)
    return result;
  std::map<std::string, mlir::Type> selfBinding =
      withSelfBinding(binding->second, receiverType);
  std::vector<ProtocolMethod> contracts;
  collectMethodContractsIn(binding->first, selfBinding, methodName, 0,
                           contracts);
  for (const ProtocolMethod &contract : contracts)
    result.push_back(contract.signature);
  return result;
}

std::vector<ProtocolMethod>
Table::methodContractsOn(mlir::Type receiverType,
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

std::optional<ProtocolMethod>
Table::resolveMethodContractOn(mlir::Type receiverType,
                               llvm::StringRef methodName,
                               llvm::ArrayRef<mlir::Type> argumentTypes) const {
  for (ProtocolMethod contract : methodContractsOn(receiverType, methodName)) {
    py::CallableType signature = contract.signature;
    llvm::ArrayRef<mlir::Type> positional = signature.getPositionalTypes();
    if (positional.empty())
      continue;
    llvm::ArrayRef<mlir::Type> explicitParams = positional.drop_front();
    if (!signature.hasVararg() && explicitParams.size() != argumentTypes.size())
      continue;
    if (signature.hasVararg() && argumentTypes.size() < explicitParams.size())
      continue;
    bool matches = true;
    std::map<std::string, mlir::Type> methodBinding;
    for (auto [expected, actual] : llvm::zip(
             explicitParams, argumentTypes.take_front(explicitParams.size()))) {
      auto expectedProtocol = mlir::dyn_cast<py::ProtocolType>(expected);
      if (!bindMethodTypeVariables(expected, actual, methodBinding) &&
          !contractAssignable(expected, actual, &classes) &&
          !(expectedProtocol && conformsTo(actual, expectedProtocol))) {
        matches = false;
        break;
      }
    }
    if (matches && signature.hasVararg() &&
        argumentTypes.size() > explicitParams.size()) {
      auto varargTuple =
          mlir::dyn_cast<py::TupleType>(signature.getVarargType());
      if (!varargTuple || varargTuple.getElementTypes().size() != 1) {
        matches = false;
      } else {
        mlir::Type repeatedExpected = varargTuple.getElementTypes().front();
        for (mlir::Type actual :
             argumentTypes.drop_front(explicitParams.size())) {
          auto expectedProtocol =
              mlir::dyn_cast<py::ProtocolType>(repeatedExpected);
          if (!bindMethodTypeVariables(repeatedExpected, actual,
                                       methodBinding) &&
              !contractAssignable(repeatedExpected, actual, &classes) &&
              !(expectedProtocol && conformsTo(actual, expectedProtocol))) {
            matches = false;
            break;
          }
        }
      }
    }
    if (matches) {
      if (!methodBinding.empty())
        contract.signature = substituteSignature(signature, methodBinding);
      return contract.signature ? std::optional<ProtocolMethod>(contract)
                                : std::nullopt;
    }
  }
  return std::nullopt;
}

std::optional<py::CallableType>
Table::resolveMethodOn(mlir::Type receiverType, llvm::StringRef methodName,
                       llvm::ArrayRef<mlir::Type> argumentTypes) const {
  std::optional<ProtocolMethod> contract =
      resolveMethodContractOn(receiverType, methodName, argumentTypes);
  return contract ? std::optional<py::CallableType>(contract->signature)
                  : std::nullopt;
}

std::optional<mlir::Type>
Table::resolveMethodResultOn(mlir::Type receiverType,
                             llvm::StringRef methodName,
                             llvm::ArrayRef<mlir::Type> argumentTypes) const {
  std::optional<ProtocolMethod> contract =
      resolveMethodContractOn(receiverType, methodName, argumentTypes);
  if (!contract || contract->signature.getResultTypes().size() != 1)
    return std::nullopt;
  return contract->signature.getResultTypes().front();
}

std::optional<std::vector<mlir::Type>>
Table::protocolArgumentsFor(mlir::Type receiverType,
                            llvm::StringRef protocolName) const {
  const ProtocolInfo *target = lookup(protocolName);
  if (!target || !target->isProtocol)
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
      auto base = classes.find(baseInfo.name);
      if (base == classes.end())
        continue;
      llvm::SmallVector<mlir::Type> args;
      for (mlir::Type arg : baseInfo.arguments) {
        mlir::Type substituted = substitute(arg, currentBinding);
        if (!substituted)
          return std::nullopt;
        args.push_back(substituted);
      }
      std::optional<std::map<std::string, mlir::Type>> baseBinding =
          bindProtocolParams(base->second, args);
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

std::optional<std::vector<mlir::Type>>
Table::completeProtocolArguments(llvm::StringRef protocolName,
                                 llvm::ArrayRef<mlir::Type> supplied) const {
  const ProtocolInfo *info = lookup(protocolName);
  if (!info || !info->isProtocol)
    return std::nullopt;
  if (std::optional<std::vector<mlir::Type>> direct =
          completeDirectArguments(*info, supplied))
    return direct;
  for (const ProtocolShortForm &shortForm : info->shortForms)
    if (std::optional<std::vector<mlir::Type>> expanded =
            completeShortArguments(*info, shortForm, supplied))
      return expanded;
  return std::nullopt;
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

mlir::Type Table::awaitablePayloadType(mlir::Type type) const {
  if (mlir::Type payload = py::awaitablePayloadType(type))
    return payload;
  auto protocol = mlir::dyn_cast<py::ProtocolType>(type);
  if (!protocol)
    return {};
  std::optional<mlir::Type> awaitResult =
      resolveMethodResultOn(protocol, "__await__", {});
  if (!awaitResult)
    return {};
  auto generator = mlir::dyn_cast<py::ProtocolType>(*awaitResult);
  if (!generator || generator.getProtocolName() != "Generator" ||
      generator.getArguments().size() != 3)
    return {};
  return generator.getArguments()[2];
}

bool Table::conformsTo(mlir::Type receiverType,
                       py::ProtocolType protocol) const {
  auto binding = bindReceiver(receiverType, classes);
  if (!binding)
    return false;

  std::function<bool(llvm::StringRef, const std::map<std::string, mlir::Type> &,
                     unsigned)>
      walk = [&](llvm::StringRef className,
                 const std::map<std::string, mlir::Type> &currentBinding,
                 unsigned depth) -> bool {
    if (depth > 16)
      return false;
    if (protocolInstantiationMatches(classes, className, currentBinding,
                                     protocol))
      return true;
    auto found = classes.find(className.str());
    if (found == classes.end())
      return false;
    for (const ProtocolBase &baseInfo : found->second.bases) {
      auto base = classes.find(baseInfo.name);
      if (base == classes.end())
        continue;
      llvm::SmallVector<mlir::Type> args;
      for (mlir::Type arg : baseInfo.arguments) {
        mlir::Type substituted = substitute(arg, currentBinding);
        if (!substituted)
          return false;
        args.push_back(substituted);
      }
      std::optional<std::map<std::string, mlir::Type>> baseBinding =
          bindProtocolParams(base->second, args);
      if (baseBinding && walk(baseInfo.name, *baseBinding, depth + 1))
        return true;
    }
    return false;
  };

  return walk(binding->first, binding->second, 0);
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
