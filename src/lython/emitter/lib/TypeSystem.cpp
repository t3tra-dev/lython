#include "TypeSystem.h"

#include "AstAccess.h"
#include "CandidateSelection.h"
#include "cpp/PyCallableShape.h"
#include "cpp/PyProtocols.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <map>
#include <string>

namespace lython::emitter {
namespace {

bool isNoneConstant(const parser::Node *node) {
  return node && node->kind == "Constant" && ast::isNoneField(*node, "value");
}

std::string literalSpelling(const parser::Node &constant) {
  if (ast::isNoneField(constant, "value"))
    return "None";
  if (auto value = ast::boolean(constant, "value"))
    return *value ? "True" : "False";
  if (auto value = ast::integer(constant, "value"))
    return std::to_string(*value);
  if (auto value = ast::string(constant, "value"))
    return "\"" + std::string(*value) + "\"";
  if (const auto *fieldValue = ast::field(constant, "value")) {
    if (const auto *big = std::get_if<parser::BigInteger>(fieldValue))
      return big->decimal;
  }
  return "object";
}

llvm::SmallVector<parser::NodePtr, 8>
concatArgs(const parser::Node &arguments, unsigned &positionalOnlyCount) {
  llvm::SmallVector<parser::NodePtr, 8> result;
  if (const auto *posOnly = ast::nodeList(arguments, "posonlyargs")) {
    positionalOnlyCount = static_cast<unsigned>(posOnly->size());
    result.append(posOnly->begin(), posOnly->end());
  }
  if (const auto *args = ast::nodeList(arguments, "args"))
    result.append(args->begin(), args->end());
  return result;
}

bool hasDefault(std::size_t index, std::size_t total, std::size_t defaults) {
  return defaults != 0 && index + defaults >= total;
}

mlir::Type inferReturnExpr(const AlgorithmM &types, const parser::Node *node,
                           const llvm::StringMap<mlir::Type> &localCallables) {
  if (node && node->kind == "Name") {
    auto found = localCallables.find(ast::nameSpelling(*node));
    if (found != localCallables.end())
      return found->second;
  }
  return types.inferExpr(node);
}

void collectReturnTypes(const AlgorithmM &types, const parser::Node *node,
                        const llvm::StringMap<mlir::Type> &localCallables,
                        llvm::SmallVectorImpl<mlir::Type> &results) {
  if (!node)
    return;
  if (node->kind == "FunctionDef" || node->kind == "AsyncFunctionDef" ||
      node->kind == "Lambda" || node->kind == "ClassDef")
    return;
  if (node->kind == "Return") {
    results.push_back(
        inferReturnExpr(types, ast::node(*node, "value"), localCallables));
    return;
  }
  for (const parser::Field &field : node->fields) {
    if (const auto *child = std::get_if<parser::NodePtr>(&field.value)) {
      if (*child)
        collectReturnTypes(types, child->get(), localCallables, results);
    } else if (const auto *children =
                   std::get_if<std::vector<parser::NodePtr>>(&field.value)) {
      for (const parser::NodePtr &child : *children)
        collectReturnTypes(types, child.get(), localCallables, results);
    }
  }
}

mlir::Type inferredFunctionResult(const AlgorithmM &types,
                                  const parser::Node &function) {
  llvm::StringMap<mlir::Type> localCallables;
  if (const auto *body = ast::nodeList(function, "body")) {
    for (const parser::NodePtr &statement : *body) {
      if (!statement || (statement->kind != "FunctionDef" &&
                         statement->kind != "AsyncFunctionDef"))
        continue;
      if (auto name = ast::string(*statement, "name"))
        localCallables[*name] = types.functionSignature(*statement).callable;
    }
  }

  llvm::SmallVector<mlir::Type, 4> results;
  if (const auto *body = ast::nodeList(function, "body"))
    for (const parser::NodePtr &statement : *body)
      collectReturnTypes(types, statement.get(), localCallables, results);
  return results.empty() ? types.none() : types.join(results);
}

bool isObjectTop(const AlgorithmM &types, mlir::Type type) {
  if (!type)
    return false;
  if (type == types.object())
    return true;
  if (auto contract = mlir::dyn_cast_if_present<py::ContractType>(type))
    return contract.getContractName() == "typing.Any";
  return false;
}

std::string manifestNameForContract(llvm::StringRef name) {
  for (llvm::StringRef prefix :
       {"builtins.", "typing.", "types.", "contextlib.", "_asyncio.",
        "asyncio.", "contextvars."}) {
    if (name.consume_front(prefix))
      return name.str();
  }
  return name.str();
}

std::optional<std::string> staticParameterName(mlir::Type type) {
  if (mlir::isa<py::SelfType>(type))
    return std::string("Self");
  if (auto typeVar = mlir::dyn_cast_if_present<py::TypeVarType>(type))
    return typeVar.getName().str();
  if (auto paramSpec = mlir::dyn_cast_if_present<py::ParamSpecType>(type))
    return paramSpec.getName().str();
  if (auto contract = mlir::dyn_cast_if_present<py::ContractType>(type)) {
    llvm::StringRef name = contract.getContractName();
    if (name.starts_with("$"))
      return name.drop_front().str();
  }
  if (auto classType = mlir::dyn_cast_if_present<py::ClassType>(type)) {
    llvm::StringRef name = classType.getClassName();
    if (name.starts_with("$"))
      return name.drop_front().str();
  }
  return std::nullopt;
}

bool structuralProtocolAccepts(const AlgorithmM &types,
                               py::ContractType expected, mlir::Type actual) {
  const py::protocols::Table &table =
      py::protocols::Table::get(types.getContext());
  std::string protocolName =
      manifestNameForContract(expected.getContractName());
  const py::protocols::ProtocolInfo *info = table.lookup(protocolName);
  if (!info || !info->isProtocol)
    return false;

  if (std::optional<std::vector<mlir::Type>> args =
          table.protocolArgumentsFor(actual, protocolName)) {
    llvm::ArrayRef<mlir::Type> expectedArgs = expected.getArguments();
    if (expectedArgs.empty() || expectedArgs.size() == args->size())
      return true;
  }

  return llvm::all_of(info->methods, [&](const auto &method) {
    return !table.methodContractCandidatesWithEvidence(actual, method.first)
                .empty();
  });
}

bool typeAccepts(const AlgorithmM &types, mlir::Type expected,
                 mlir::Type actual) {
  expected = types.widenLiteral(expected);
  actual = types.widenLiteral(actual);
  if (!expected || !actual)
    return true;
  if (expected == actual)
    return true;
  if (isObjectTop(types, expected) || isObjectTop(types, actual))
    return true;
  if (auto expectedContract =
          mlir::dyn_cast_if_present<py::ContractType>(expected))
    if (structuralProtocolAccepts(types, expectedContract, actual))
      return true;
  return py::isAssignableTo(actual, expected);
}

using TypeBindingMap = std::map<std::string, mlir::Type>;

struct CallSolution {
  mlir::Type result;
  TypeBindingMap bindings;
  py::CallableType callableContract;
  std::string methodName;
  std::optional<std::string> receiverManifestClass;
  int score = 0;
};

mlir::Type substituteType(const AlgorithmM &types, mlir::Type type,
                          const TypeBindingMap &bindings,
                          bool eraseUnbound = false);

bool bindExpectedType(const AlgorithmM &types, mlir::Type expected,
                      mlir::Type actual, TypeBindingMap &bindings);

std::optional<std::string> paramSpecName(mlir::Type type) {
  if (auto paramSpec = mlir::dyn_cast_if_present<py::ParamSpecType>(type))
    return paramSpec.getName().str();
  return std::nullopt;
}

bool sameKeywords(mlir::ArrayRef<CallKeywordType> lhs,
                  mlir::ArrayRef<CallKeywordType> rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (auto [left, right] : llvm::zip(lhs, rhs))
    if (left.name != right.name || left.type != right.type)
      return false;
  return true;
}

void collectExplicitKeywordFormalNames(
    py::CallableType callable, std::size_t firstParameter,
    std::optional<std::size_t> paramSpecIndex, llvm::StringSet<> &names) {
  llvm::ArrayRef<mlir::StringAttr> positionalNames =
      callable.getPositionalNames();
  for (auto [index, name] : llvm::enumerate(positionalNames)) {
    if (!name || index < firstParameter ||
        (paramSpecIndex && index == *paramSpecIndex) ||
        index >= callable.getPositionalTypes().size() ||
        index < callable.getPositionalOnlyCount())
      continue;
    names.insert(name.getValue());
  }

  for (mlir::StringAttr name : callable.getKwOnlyNames())
    if (name)
      names.insert(name.getValue());
}

llvm::SmallVector<CallKeywordType, 4>
captureParamSpecKeywords(py::CallableType callable, std::size_t firstParameter,
                         std::optional<std::size_t> paramSpecIndex,
                         mlir::ArrayRef<CallKeywordType> keywords) {
  llvm::StringSet<> explicitNames;
  collectExplicitKeywordFormalNames(callable, firstParameter, paramSpecIndex,
                                    explicitNames);

  llvm::SmallVector<CallKeywordType, 4> captured;
  captured.reserve(keywords.size());
  for (const CallKeywordType &keyword : keywords) {
    if (!keyword.name.empty() && explicitNames.contains(keyword.name))
      continue;
    captured.push_back(keyword);
  }
  return captured;
}

py::CallableType makeParamSpecPack(mlir::MLIRContext *context,
                                   mlir::ArrayRef<mlir::Type> positional,
                                   mlir::ArrayRef<CallKeywordType> keywords) {
  llvm::SmallVector<mlir::Type, 4> kwonly;
  llvm::SmallVector<mlir::StringAttr, 4> kwonlyNames;
  kwonly.reserve(keywords.size());
  kwonlyNames.reserve(keywords.size());
  for (const CallKeywordType &keyword : keywords) {
    kwonly.push_back(keyword.type);
    kwonlyNames.push_back(mlir::StringAttr::get(context, keyword.name));
  }
  return py::CallableType::get(context, positional, kwonly, {}, {}, {}, {},
                               kwonlyNames, {}, {});
}

std::optional<llvm::SmallVector<CallKeywordType, 4>>
paramSpecPackKeywords(py::CallableType pack) {
  llvm::SmallVector<CallKeywordType, 4> keywords;
  llvm::ArrayRef<mlir::StringAttr> names = pack.getKwOnlyNames();
  for (auto [index, kwType] : llvm::enumerate(pack.getKwOnlyTypes())) {
    if (index >= names.size())
      return std::nullopt;
    keywords.push_back(CallKeywordType{names[index].getValue().str(), kwType});
  }
  return keywords;
}

bool bindParamSpecPack(const AlgorithmM &types, llvm::StringRef name,
                       mlir::ArrayRef<mlir::Type> positional,
                       mlir::ArrayRef<CallKeywordType> keywords,
                       TypeBindingMap &bindings, bool merge = false) {
  py::CallableType pack =
      makeParamSpecPack(&types.getContext(), positional, keywords);
  auto found = bindings.find(name.str());
  if (found == bindings.end()) {
    bindings[name.str()] = pack;
    return true;
  }
  auto existing = mlir::dyn_cast_if_present<py::CallableType>(found->second);
  if (!existing)
    return false;
  std::optional<llvm::SmallVector<CallKeywordType, 4>> existingKeywords =
      paramSpecPackKeywords(existing);
  if (!existingKeywords)
    return false;
  if (!merge) {
    return existing.getPositionalTypes() == positional &&
           sameKeywords(*existingKeywords, keywords);
  }

  llvm::ArrayRef<mlir::Type> existingPositionals =
      existing.getPositionalTypes();
  llvm::SmallVector<mlir::Type, 4> mergedPositionals;
  if (!existingPositionals.empty() && !positional.empty() &&
      existingPositionals != positional)
    return false;
  llvm::ArrayRef<mlir::Type> selectedPositionals =
      !positional.empty() ? positional : existingPositionals;
  mergedPositionals.append(selectedPositionals.begin(),
                           selectedPositionals.end());

  llvm::SmallVector<CallKeywordType, 4> mergedKeywords;
  if (!existingKeywords->empty() && !keywords.empty() &&
      !sameKeywords(*existingKeywords, keywords))
    return false;
  llvm::ArrayRef<CallKeywordType> selectedKeywords =
      !keywords.empty() ? keywords : *existingKeywords;
  mergedKeywords.append(selectedKeywords.begin(), selectedKeywords.end());

  found->second =
      makeParamSpecPack(&types.getContext(), mergedPositionals, mergedKeywords);
  return true;
}

bool bindTypeParameter(const AlgorithmM &types, llvm::StringRef name,
                       mlir::Type actual, TypeBindingMap &bindings) {
  actual = types.widenLiteral(actual);
  auto found = bindings.find(name.str());
  if (found == bindings.end()) {
    bindings[name.str()] = actual ? actual : types.object();
    return true;
  }
  if (typeAccepts(types, found->second, actual))
    return true;
  if (typeAccepts(types, actual, found->second)) {
    found->second = actual;
    return true;
  }
  return false;
}

bool bindTypeList(const AlgorithmM &types, mlir::ArrayRef<mlir::Type> expected,
                  mlir::ArrayRef<mlir::Type> actual, TypeBindingMap &bindings) {
  if (expected.size() != actual.size())
    return false;
  for (auto [expectedType, actualType] : llvm::zip(expected, actual))
    if (!bindExpectedType(types, expectedType, actualType, bindings))
      return false;
  return true;
}

bool bindUnionMember(const AlgorithmM &types, py::UnionType expected,
                     mlir::Type actual, TypeBindingMap &bindings) {
  for (mlir::Type member : expected.getMemberTypes()) {
    TypeBindingMap candidate = bindings;
    if (bindExpectedType(types, member, actual, candidate)) {
      bindings = std::move(candidate);
      return true;
    }
  }
  return false;
}

bool bindProtocolView(const AlgorithmM &types, py::ProtocolType expected,
                      mlir::Type actual, TypeBindingMap &bindings) {
  const py::protocols::Table &table =
      py::protocols::Table::get(types.getContext());
  llvm::ArrayRef<mlir::Type> expectedArgs = expected.getArguments();
  if (auto actualProtocol = mlir::dyn_cast_if_present<py::ProtocolType>(actual))
    if (actualProtocol.getProtocolName() == expected.getProtocolName())
      return expectedArgs.empty() ||
             bindTypeList(types, expectedArgs, actualProtocol.getArguments(),
                          bindings);

  std::optional<std::vector<mlir::Type>> actualArgs =
      table.protocolArgumentsFor(actual, expected.getProtocolName());
  if (!actualArgs)
    return typeAccepts(types, expected, actual);
  if (expectedArgs.empty())
    return true;
  return bindTypeList(types, expectedArgs, *actualArgs, bindings);
}

bool bindContractView(const AlgorithmM &types, py::ContractType expected,
                      mlir::Type actual, TypeBindingMap &bindings) {
  if (auto actualContract = mlir::dyn_cast_if_present<py::ContractType>(actual))
    if (actualContract.getContractName() == expected.getContractName())
      return expected.getArguments().empty() ||
             bindTypeList(types, expected.getArguments(),
                          actualContract.getArguments(), bindings);

  if (structuralProtocolAccepts(types, expected, actual))
    return true;
  return typeAccepts(types, expected, actual);
}

bool bindCallableView(const AlgorithmM &types, py::CallableType expected,
                      py::CallableType actual, TypeBindingMap &bindings) {
  if (expected.getPositionalTypes().size() !=
          actual.getPositionalTypes().size() ||
      expected.getKwOnlyTypes().size() != actual.getKwOnlyTypes().size() ||
      expected.getResultTypes().size() != actual.getResultTypes().size())
    return typeAccepts(types, expected, actual);

  for (auto [expectedArg, actualArg] :
       llvm::zip(expected.getPositionalTypes(), actual.getPositionalTypes()))
    if (!bindExpectedType(types, expectedArg, actualArg, bindings))
      return false;
  for (auto [expectedArg, actualArg] :
       llvm::zip(expected.getKwOnlyTypes(), actual.getKwOnlyTypes()))
    if (!bindExpectedType(types, expectedArg, actualArg, bindings))
      return false;
  for (auto [expectedResult, actualResult] :
       llvm::zip(expected.getResultTypes(), actual.getResultTypes()))
    if (!bindExpectedType(types, expectedResult, actualResult, bindings))
      return false;
  return true;
}

bool bindExpectedType(const AlgorithmM &types, mlir::Type expected,
                      mlir::Type actual, TypeBindingMap &bindings) {
  expected = substituteType(types, expected, bindings);
  actual = types.widenLiteral(actual);

  if (!expected || !actual)
    return true;
  if (std::optional<std::string> name = staticParameterName(expected))
    return bindTypeParameter(types, *name, actual, bindings);
  if (expected == actual)
    return true;
  if (isObjectTop(types, expected) || isObjectTop(types, actual))
    return true;

  if (auto unionType = mlir::dyn_cast_if_present<py::UnionType>(expected))
    return bindUnionMember(types, unionType, actual, bindings);
  if (auto expectedType = mlir::dyn_cast_if_present<py::TypeType>(expected)) {
    if (auto actualType = mlir::dyn_cast_if_present<py::TypeType>(actual))
      return bindExpectedType(types, expectedType.getInstanceType(),
                              actualType.getInstanceType(), bindings);
  }
  if (auto expectedProtocol =
          mlir::dyn_cast_if_present<py::ProtocolType>(expected))
    return bindProtocolView(types, expectedProtocol, actual, bindings);
  if (auto expectedContract =
          mlir::dyn_cast_if_present<py::ContractType>(expected))
    return bindContractView(types, expectedContract, actual, bindings);
  if (auto expectedCallable =
          mlir::dyn_cast_if_present<py::CallableType>(expected))
    if (auto actualCallable =
            mlir::dyn_cast_if_present<py::CallableType>(actual))
      return bindCallableView(types, expectedCallable, actualCallable,
                              bindings);

  return typeAccepts(types, expected, actual);
}

py::CallableType substituteCallable(const AlgorithmM &types,
                                    py::CallableType callable,
                                    const TypeBindingMap &bindings,
                                    bool eraseUnbound) {
  if (!callable)
    return {};

  auto boundParamSpecPack = [&](llvm::StringRef name) -> py::CallableType {
    auto found = bindings.find(name.str());
    if (found == bindings.end())
      return {};
    return mlir::dyn_cast_if_present<py::CallableType>(found->second);
  };

  llvm::SmallVector<mlir::Type, 8> positional;
  llvm::SmallVector<mlir::StringAttr, 8> positionalNames;
  llvm::SmallVector<mlir::BoolAttr, 8> positionalDefaults;
  bool hasPositionalNames = !callable.getPositionalNames().empty();
  bool hasPositionalDefaults = !callable.getPositionalDefaults().empty();
  for (auto [index, arg] : llvm::enumerate(callable.getPositionalTypes())) {
    if (std::optional<std::string> name = paramSpecName(arg)) {
      py::CallableType pack = boundParamSpecPack(*name);
      if (pack) {
        for (mlir::Type packArg : pack.getPositionalTypes()) {
          positional.push_back(
              substituteType(types, packArg, bindings, eraseUnbound));
          if (hasPositionalNames)
            positionalNames.push_back(mlir::StringAttr());
          if (hasPositionalDefaults)
            positionalDefaults.push_back(
                mlir::BoolAttr::get(callable.getContext(), false));
        }
        continue;
      }
    }
    positional.push_back(substituteType(types, arg, bindings, eraseUnbound));
    if (hasPositionalNames) {
      llvm::ArrayRef<mlir::StringAttr> names = callable.getPositionalNames();
      positionalNames.push_back(index < names.size() ? names[index]
                                                     : mlir::StringAttr());
    }
    if (hasPositionalDefaults) {
      llvm::ArrayRef<mlir::BoolAttr> defaults =
          callable.getPositionalDefaults();
      positionalDefaults.push_back(
          index < defaults.size()
              ? defaults[index]
              : mlir::BoolAttr::get(callable.getContext(), false));
    }
  }

  mlir::Type vararg;
  if (callable.hasVararg()) {
    mlir::Type varargType = callable.getVarargType();
    std::optional<std::string> name = paramSpecName(varargType);
    py::CallableType pack =
        name ? boundParamSpecPack(*name) : py::CallableType();
    if (pack) {
      for (mlir::Type packArg : pack.getPositionalTypes()) {
        positional.push_back(
            substituteType(types, packArg, bindings, eraseUnbound));
        if (hasPositionalNames)
          positionalNames.push_back(mlir::StringAttr());
        if (hasPositionalDefaults)
          positionalDefaults.push_back(
              mlir::BoolAttr::get(callable.getContext(), false));
      }
    } else {
      vararg = substituteType(types, varargType, bindings, eraseUnbound);
    }
  }

  llvm::SmallVector<mlir::Type, 4> kwonly;
  llvm::SmallVector<mlir::StringAttr, 4> kwonlyNames;
  llvm::SmallVector<mlir::BoolAttr, 4> kwonlyDefaults;
  for (mlir::Type arg : callable.getKwOnlyTypes())
    kwonly.push_back(substituteType(types, arg, bindings, eraseUnbound));
  kwonlyNames.append(callable.getKwOnlyNames().begin(),
                     callable.getKwOnlyNames().end());
  kwonlyDefaults.append(callable.getKwOnlyDefaults().begin(),
                        callable.getKwOnlyDefaults().end());

  llvm::StringSet<> expandedKeywordParamSpecs;
  auto appendParamSpecKeywords = [&](llvm::StringRef name) {
    if (!expandedKeywordParamSpecs.insert(name).second)
      return;
    py::CallableType pack = boundParamSpecPack(name);
    if (!pack)
      return;
    if (!pack.getKwOnlyTypes().empty() && kwonlyNames.empty() &&
        !kwonly.empty()) {
      for (std::size_t index = 0, end = kwonly.size(); index < end; ++index)
        kwonlyNames.push_back(mlir::StringAttr());
    }
    if (!pack.getKwOnlyTypes().empty() && kwonlyDefaults.empty() &&
        !kwonly.empty()) {
      for (std::size_t index = 0, end = kwonly.size(); index < end; ++index)
        kwonlyDefaults.push_back(
            mlir::BoolAttr::get(callable.getContext(), false));
    }
    for (auto [index, kwType] : llvm::enumerate(pack.getKwOnlyTypes())) {
      kwonly.push_back(substituteType(types, kwType, bindings, eraseUnbound));
      llvm::ArrayRef<mlir::StringAttr> names = pack.getKwOnlyNames();
      kwonlyNames.push_back(index < names.size() ? names[index]
                                                 : mlir::StringAttr());
      kwonlyDefaults.push_back(
          mlir::BoolAttr::get(callable.getContext(), false));
    }
  };

  for (mlir::Type arg : callable.getPositionalTypes())
    if (std::optional<std::string> name = paramSpecName(arg))
      appendParamSpecKeywords(*name);
  if (callable.hasVararg())
    if (std::optional<std::string> name =
            paramSpecName(callable.getVarargType()))
      appendParamSpecKeywords(*name);
  if (callable.hasKwarg())
    if (std::optional<std::string> name =
            paramSpecName(callable.getKwargType()))
      appendParamSpecKeywords(*name);

  llvm::SmallVector<mlir::Type, 1> results;
  for (mlir::Type result : callable.getResultTypes())
    results.push_back(substituteType(types, result, bindings, eraseUnbound));

  mlir::Type kwarg;
  if (callable.hasKwarg()) {
    mlir::Type kwargType = callable.getKwargType();
    std::optional<std::string> name = paramSpecName(kwargType);
    if (!name || !boundParamSpecPack(*name))
      kwarg = substituteType(types, kwargType, bindings, eraseUnbound);
  }

  return py::CallableType::get(
      callable.getContext(), positional, kwonly, vararg, kwarg, results,
      positionalNames, kwonlyNames, positionalDefaults, kwonlyDefaults,
      vararg ? callable.getVarargName() : mlir::StringAttr(),
      kwarg ? callable.getKwargName() : mlir::StringAttr(),
      callable.getPositionalOnlyCount());
}

mlir::Type substituteType(const AlgorithmM &types, mlir::Type type,
                          const TypeBindingMap &bindings, bool eraseUnbound) {
  if (!type)
    return type;
  if (std::optional<std::string> name = staticParameterName(type)) {
    auto found = bindings.find(*name);
    if (found != bindings.end())
      return found->second;
    return eraseUnbound ? types.object() : type;
  }
  if (auto contract = mlir::dyn_cast_if_present<py::ContractType>(type)) {
    llvm::SmallVector<mlir::Type, 4> args;
    for (mlir::Type arg : contract.getArguments())
      args.push_back(substituteType(types, arg, bindings, eraseUnbound));
    return py::ContractType::get(type.getContext(), contract.getContractName(),
                                 args);
  }
  if (auto protocol = mlir::dyn_cast_if_present<py::ProtocolType>(type)) {
    llvm::SmallVector<mlir::Type, 4> args;
    for (mlir::Type arg : protocol.getArguments())
      args.push_back(substituteType(types, arg, bindings, eraseUnbound));
    return py::ProtocolType::get(type.getContext(), protocol.getProtocolName(),
                                 args);
  }
  if (auto typeType = mlir::dyn_cast_if_present<py::TypeType>(type)) {
    mlir::Type instance = substituteType(types, typeType.getInstanceType(),
                                         bindings, eraseUnbound);
    return instance ? py::TypeType::get(type.getContext(), instance)
                    : mlir::Type();
  }
  if (auto unionType = mlir::dyn_cast_if_present<py::UnionType>(type)) {
    llvm::SmallVector<mlir::Type, 4> members;
    for (mlir::Type member : unionType.getMemberTypes())
      members.push_back(substituteType(types, member, bindings, eraseUnbound));
    return py::UnionType::getNormalized(type.getContext(), members);
  }
  if (auto callable = mlir::dyn_cast_if_present<py::CallableType>(type))
    return substituteCallable(types, callable, bindings, eraseUnbound);
  if (auto overload = mlir::dyn_cast_if_present<py::OverloadType>(type)) {
    llvm::SmallVector<mlir::Type, 4> candidates;
    for (mlir::Type candidate : overload.getCandidateTypes())
      candidates.push_back(
          substituteType(types, candidate, bindings, eraseUnbound));
    return py::OverloadType::get(type.getContext(), candidates);
  }
  if (auto tuple = mlir::dyn_cast_if_present<py::TupleType>(type)) {
    llvm::SmallVector<mlir::Type, 4> elements;
    for (mlir::Type element : tuple.getElementTypes())
      elements.push_back(
          substituteType(types, element, bindings, eraseUnbound));
    return py::TupleType::get(type.getContext(), elements);
  }
  if (auto list = mlir::dyn_cast_if_present<py::ListType>(type))
    return py::ListType::get(
        type.getContext(),
        substituteType(types, list.getElementType(), bindings, eraseUnbound));
  if (auto dict = mlir::dyn_cast_if_present<py::DictType>(type))
    return py::DictType::get(
        type.getContext(),
        substituteType(types, dict.getKeyType(), bindings, eraseUnbound),
        substituteType(types, dict.getValueType(), bindings, eraseUnbound));
  return type;
}

mlir::Type callableResultType(const AlgorithmM &types,
                              py::CallableType callable,
                              const TypeBindingMap &bindings) {
  llvm::ArrayRef<mlir::Type> results = callable.getResultTypes();
  if (results.size() == 1)
    return substituteType(types, results.front(), bindings,
                          /*eraseUnbound=*/true);
  if (!results.empty()) {
    llvm::SmallVector<mlir::Type, 4> substituted;
    for (mlir::Type result : results)
      substituted.push_back(
          substituteType(types, result, bindings, /*eraseUnbound=*/true));
    return types.tupleOf(types.join(substituted));
  }
  return mlir::Type();
}

std::optional<py::CallableType>
expandParamSpecForInvocation(const AlgorithmM &types, py::CallableType callable,
                             mlir::ArrayRef<mlir::Type> positional,
                             mlir::ArrayRef<CallKeywordType> keywords,
                             TypeBindingMap &bindings,
                             std::size_t firstParameter) {
  llvm::ArrayRef<mlir::Type> formals = callable.getPositionalTypes();
  std::optional<std::size_t> paramSpecIndex;
  for (std::size_t index = firstParameter, end = formals.size(); index < end;
       ++index) {
    if (!paramSpecName(formals[index]))
      continue;
    if (paramSpecIndex)
      return std::nullopt;
    paramSpecIndex = index;
  }

  if (paramSpecIndex) {
    std::size_t visibleBefore = *paramSpecIndex - firstParameter;
    std::size_t trailing = formals.size() - *paramSpecIndex - 1;
    if (positional.size() < visibleBefore + trailing)
      return std::nullopt;

    std::size_t capturedCount = positional.size() - visibleBefore - trailing;
    llvm::ArrayRef<mlir::Type> captured =
        positional.slice(visibleBefore, capturedCount);
    std::optional<std::string> name = paramSpecName(formals[*paramSpecIndex]);
    llvm::SmallVector<CallKeywordType, 4> capturedKeywords =
        captureParamSpecKeywords(callable, firstParameter, paramSpecIndex,
                                 keywords);
    if (!name ||
        !bindParamSpecPack(types, *name, captured, capturedKeywords, bindings))
      return std::nullopt;
  } else if (callable.hasVararg()) {
    if (std::optional<std::string> name =
            paramSpecName(callable.getVarargType())) {
      std::size_t fixedVisible = formals.size() - firstParameter;
      llvm::ArrayRef<mlir::Type> captured;
      if (positional.size() > fixedVisible)
        captured = positional.drop_front(fixedVisible);
      if (!bindParamSpecPack(types, *name, captured, {}, bindings,
                             /*merge=*/true))
        return std::nullopt;
    }
  }

  if (callable.hasKwarg()) {
    if (std::optional<std::string> name =
            paramSpecName(callable.getKwargType())) {
      llvm::SmallVector<CallKeywordType, 4> capturedKeywords =
          captureParamSpecKeywords(callable, firstParameter, paramSpecIndex,
                                   keywords);
      if (!bindParamSpecPack(types, *name, {}, capturedKeywords, bindings,
                             /*merge=*/true))
        return std::nullopt;
    }
  }

  return substituteCallable(types, callable, bindings,
                            /*eraseUnbound=*/false);
}

int unboundStaticParameterCount(mlir::Type type) {
  if (!type)
    return 0;
  if (staticParameterName(type))
    return 1;
  if (auto contract = mlir::dyn_cast_if_present<py::ContractType>(type)) {
    int count = 0;
    for (mlir::Type arg : contract.getArguments())
      count += unboundStaticParameterCount(arg);
    return count;
  }
  if (auto protocol = mlir::dyn_cast_if_present<py::ProtocolType>(type)) {
    int count = 0;
    for (mlir::Type arg : protocol.getArguments())
      count += unboundStaticParameterCount(arg);
    return count;
  }
  if (auto unionType = mlir::dyn_cast_if_present<py::UnionType>(type)) {
    int count = 0;
    for (mlir::Type member : unionType.getMemberTypes())
      count += unboundStaticParameterCount(member);
    return count;
  }
  if (auto typeType = mlir::dyn_cast_if_present<py::TypeType>(type))
    return unboundStaticParameterCount(typeType.getInstanceType());
  if (auto callable = mlir::dyn_cast_if_present<py::CallableType>(type)) {
    int count = 0;
    for (mlir::Type arg : callable.getPositionalTypes())
      count += unboundStaticParameterCount(arg);
    for (mlir::Type arg : callable.getKwOnlyTypes())
      count += unboundStaticParameterCount(arg);
    for (mlir::Type result : callable.getResultTypes())
      count += unboundStaticParameterCount(result);
    if (callable.hasVararg())
      count += unboundStaticParameterCount(callable.getVarargType());
    if (callable.hasKwarg())
      count += unboundStaticParameterCount(callable.getKwargType());
    return count;
  }
  return 0;
}

int matchSpecificity(const AlgorithmM &types, mlir::Type expected,
                     mlir::Type actual) {
  expected = types.widenLiteral(expected);
  actual = types.widenLiteral(actual);
  if (!expected || !actual)
    return 0;
  if (expected == actual)
    return 12;
  if (isObjectTop(types, expected))
    return 0;
  if (auto unionType = mlir::dyn_cast_if_present<py::UnionType>(expected)) {
    int best = 0;
    for (mlir::Type member : unionType.getMemberTypes())
      best = std::max(best, matchSpecificity(types, member, actual));
    return best > 0 ? best - 1 : 0;
  }
  if (typeAccepts(types, expected, actual))
    return 4;
  return 0;
}

int callableSpecificity(const AlgorithmM &types, py::CallableType callable,
                        std::size_t firstParameter) {
  if (!callable)
    return 0;
  int score = 0;
  llvm::ArrayRef<mlir::Type> positional = callable.getPositionalTypes();
  for (std::size_t index = firstParameter, end = positional.size(); index < end;
       ++index) {
    if (!isObjectTop(types, positional[index]))
      score += 2;
    score -= 4 * unboundStaticParameterCount(positional[index]);
  }
  for (mlir::Type arg : callable.getKwOnlyTypes()) {
    if (!isObjectTop(types, arg))
      score += 2;
    score -= 4 * unboundStaticParameterCount(arg);
  }
  for (mlir::Type result : callable.getResultTypes()) {
    if (!isObjectTop(types, result))
      score += 3;
    score -= 8 * unboundStaticParameterCount(result);
  }
  if (!callable.hasVararg())
    score += 2;
  if (!callable.hasKwarg())
    score += 2;
  return score;
}

std::optional<CallSolution>
tryCallableApplication(const AlgorithmM &types, py::CallableType callable,
                       mlir::ArrayRef<mlir::Type> positional,
                       mlir::ArrayRef<CallKeywordType> keywords,
                       TypeBindingMap bindings = {},
                       std::size_t firstParameter = 0) {
  std::optional<py::CallableType> expanded = expandParamSpecForInvocation(
      types, callable, positional, keywords, bindings, firstParameter);
  if (!expanded)
    return std::nullopt;

  py::CallableApplicationShapeOptions opts;
  opts.firstParameter = firstParameter;
  int specificity = callableSpecificity(types, *expanded, firstParameter);
  auto resolved = py::resolveCallableApplicationShape(
      *expanded, positional, keywords, opts,
      [&](mlir::Type expected, mlir::Type actual) {
        if (!bindExpectedType(types, expected, actual, bindings))
          return false;
        specificity += matchSpecificity(
            types,
            substituteType(types, expected, bindings, /*eraseUnbound=*/true),
            actual);
        return true;
      },
      [](const CallKeywordType &keyword) -> llvm::StringRef {
        return keyword.name;
      },
      [](const CallKeywordType &keyword) { return keyword.type; });
  if (!resolved)
    return std::nullopt;
  py::CallableType evidence =
      substituteCallable(types, *expanded, bindings, /*eraseUnbound=*/true);
  mlir::Type result = callableResultType(types, *expanded, bindings);
  return CallSolution{result ? result : types.object(),
                      std::move(bindings),
                      evidence,
                      {},
                      std::nullopt,
                      resolved->score + specificity};
}

bool sameCallSolution(const CallSolution &lhs, const CallSolution &rhs) {
  return lhs.result == rhs.result &&
         lhs.callableContract == rhs.callableContract &&
         lhs.methodName == rhs.methodName &&
         lhs.receiverManifestClass == rhs.receiverManifestClass;
}

std::optional<CallSolution> selectCallableApplication(
    const AlgorithmM &types, llvm::ArrayRef<py::CallableType> candidates,
    mlir::ArrayRef<mlir::Type> positional,
    mlir::ArrayRef<CallKeywordType> keywords, TypeBindingMap bindings = {},
    std::size_t firstParameter = 0) {
  auto selection = lython::selection::bestCandidate<CallSolution>(
      [](const CallSolution &solution) { return solution.score; },
      sameCallSolution);
  for (py::CallableType candidate : candidates) {
    if (std::optional<CallSolution> solution = tryCallableApplication(
            types, candidate, positional, keywords, bindings, firstParameter))
      selection.consider(std::move(*solution));
  }
  return std::move(selection).finish();
}

std::optional<CallSolution>
tryManifestMethod(const AlgorithmM &types, mlir::Type receiverType,
                  llvm::StringRef methodName,
                  mlir::ArrayRef<mlir::Type> positional,
                  mlir::ArrayRef<CallKeywordType> keywords = {}) {
  const py::protocols::Table &table =
      py::protocols::Table::get(types.getContext());
  auto selection = lython::selection::bestCandidate<CallSolution>(
      [](const CallSolution &solution) { return solution.score; },
      sameCallSolution);

  for (py::protocols::ContractResolution candidate :
       table.methodContractCandidatesWithEvidence(receiverType, methodName)) {
    py::CallableType signature = candidate.method.signature;
    TypeBindingMap bindings = std::move(candidate.typeBindings);
    if (signature.getPositionalTypes().empty())
      continue;
    if (!bindExpectedType(types, signature.getPositionalTypes().front(),
                          receiverType, bindings))
      continue;
    std::optional<CallSolution> solution = tryCallableApplication(
        types, signature, positional, keywords, std::move(bindings),
        /*firstParameter=*/1);
    if (!solution)
      continue;
    solution->score += candidate.score;
    solution->methodName = methodName.str();
    if (candidate.receiverEvidence)
      solution->receiverManifestClass =
          candidate.receiverEvidence->manifestClass;
    selection.consider(std::move(*solution));
  }
  return std::move(selection).finish();
}

} // namespace

AlgorithmM::AlgorithmM(mlir::MLIRContext &context) : context(context) {}

AlgorithmM::Scope::Scope(Scope &&other) noexcept : owner(other.owner) {
  other.owner = nullptr;
}

AlgorithmM::Scope &AlgorithmM::Scope::operator=(Scope &&other) noexcept {
  if (this == &other)
    return *this;
  reset();
  owner = other.owner;
  other.owner = nullptr;
  return *this;
}

AlgorithmM::Scope::~Scope() { reset(); }

void AlgorithmM::Scope::reset() {
  if (!owner)
    return;
  owner->popScope();
  owner = nullptr;
}

void AlgorithmM::seedBuiltins() {
  bindSymbol("None", none());
  bindSymbol("True", literal("True"));
  bindSymbol("False", literal("False"));
  bindSymbol("print", contract("builtins.function"));
  bindSymbol("len", py::CallableType::get(&context, {object()}, {}, {}, {},
                                          {intType()}));
  bindClass("int", intType());
  bindClass("float", floatType());
  bindClass("BaseException", contract("builtins.BaseException"));
  bindClass("Exception", contract("builtins.Exception"));
  bindClass("RuntimeError", contract("builtins.RuntimeError"));
  bindClass("TypeError", contract("builtins.TypeError"));
  bindClass("ValueError", contract("builtins.ValueError"));
  bindClass("KeyError", contract("builtins.KeyError"));
  bindClass("IndexError", contract("builtins.IndexError"));
  bindClass("AssertionError", contract("builtins.AssertionError"));
  bindClass("StopIteration", contract("builtins.StopIteration"));
  bindClass("StopAsyncIteration", contract("builtins.StopAsyncIteration"));
  bindClass("nullcontext", contract("contextlib.nullcontext"));
  bindSymbol("range", typeObject(contract("builtins.range")));
}

mlir::Type AlgorithmM::object() const { return contract("builtins.object"); }
mlir::Type AlgorithmM::none() const { return literal("None"); }
mlir::Type AlgorithmM::boolType() const { return contract("builtins.bool"); }
mlir::Type AlgorithmM::intType() const { return contract("builtins.int"); }
mlir::Type AlgorithmM::strType() const { return contract("builtins.str"); }
mlir::Type AlgorithmM::floatType() const { return contract("builtins.float"); }

mlir::Type AlgorithmM::contract(llvm::StringRef name,
                                mlir::ArrayRef<mlir::Type> arguments) const {
  return py::ContractType::get(&context, name, arguments);
}

mlir::Type AlgorithmM::protocol(llvm::StringRef name,
                                mlir::ArrayRef<mlir::Type> arguments) const {
  return py::ProtocolType::get(&context, name, arguments);
}

mlir::Type AlgorithmM::literal(llvm::StringRef spelling) const {
  return py::LiteralType::get(&context, spelling);
}

mlir::Type AlgorithmM::typeObject(mlir::Type instanceType) const {
  return py::TypeType::get(&context, instanceType);
}

mlir::Type AlgorithmM::tupleOf(mlir::Type elementType) const {
  llvm::SmallVector<mlir::Type, 1> args;
  if (elementType)
    args.push_back(elementType);
  return contract("builtins.tuple", args);
}

mlir::Type AlgorithmM::listOf(mlir::Type elementType) const {
  llvm::SmallVector<mlir::Type, 1> args;
  if (elementType)
    args.push_back(elementType);
  return contract("builtins.list", args);
}

mlir::Type AlgorithmM::dictOf(mlir::Type keyType, mlir::Type valueType) const {
  llvm::SmallVector<mlir::Type, 2> args;
  if (keyType)
    args.push_back(keyType);
  if (valueType)
    args.push_back(valueType);
  return contract("builtins.dict", args);
}

mlir::Type AlgorithmM::iteratorOf(mlir::Type elementType) const {
  llvm::SmallVector<mlir::Type, 1> args;
  if (elementType)
    args.push_back(elementType);
  return protocol("Iterator", args);
}

AlgorithmM::Scope AlgorithmM::pushScope() const {
  scopes.emplace_back();
  return Scope(*this);
}

void AlgorithmM::popScope() const {
  if (!scopes.empty())
    scopes.pop_back();
}

void AlgorithmM::bindLocalSymbol(llvm::StringRef name, mlir::Type type) const {
  if (scopes.empty())
    return;
  scopes.back()[name] = type ? type : object();
}

void AlgorithmM::bindSymbol(llvm::StringRef name, mlir::Type type) {
  if (!scopes.empty()) {
    scopes.back()[name] = type ? type : object();
    return;
  }
  symbols[name] = type ? type : object();
}

std::optional<mlir::Type> AlgorithmM::lookupSymbol(llvm::StringRef name) const {
  for (auto it = scopes.rbegin(), e = scopes.rend(); it != e; ++it) {
    auto found = it->find(name);
    if (found != it->end())
      return found->second;
  }
  auto found = symbols.find(name);
  if (found == symbols.end())
    return std::nullopt;
  return found->second;
}

void AlgorithmM::bindClass(llvm::StringRef name, mlir::Type instanceType) {
  classes[name] = instanceType ? instanceType : contract(name);
  symbols[name] = typeObject(classes[name]);
}

std::optional<mlir::Type> AlgorithmM::lookupClass(llvm::StringRef name) const {
  auto found = classes.find(name);
  if (found == classes.end())
    return std::nullopt;
  return found->second;
}

mlir::Type AlgorithmM::annotationType(const parser::Node *node) const {
  if (!node)
    return object();
  if (node->kind == "Name") {
    llvm::StringRef name = ast::nameSpelling(*node);
    if (name == "int")
      return intType();
    if (name == "str")
      return strType();
    if (name == "bool")
      return boolType();
    if (name == "float")
      return floatType();
    if (name == "object" || name == "Any")
      return object();
    if (name == "None")
      return none();
    if (name == "Self")
      return py::SelfType::get(&context);
    if (auto knownClass = lookupClass(name))
      return *knownClass;
    return contract((llvm::Twine("builtins.") + name).str());
  }
  if (node->kind == "Constant")
    return isNoneConstant(node) ? none() : literal(literalSpelling(*node));
  if (node->kind == "Attribute") {
    llvm::StringRef attr = ast::nameSpelling(*node);
    if (attr == "Any")
      return object();
    if (attr == "Self")
      return py::SelfType::get(&context);
    return contract(attr);
  }
  if (node->kind == "BinOp" && ast::isOperator(ast::node(*node, "op"), "BitOr"))
    return py::UnionType::getNormalized(
        &context, {annotationType(ast::node(*node, "left")),
                   annotationType(ast::node(*node, "right"))});
  if (node->kind == "Subscript") {
    const parser::Node *base = ast::node(*node, "value");
    const parser::Node *slice = ast::node(*node, "slice");
    llvm::StringRef baseName = base ? ast::nameSpelling(*base) : "";
    if (baseName == "Optional")
      return py::UnionType::getNormalized(&context,
                                          {annotationType(slice), none()});
    if (baseName == "Union") {
      llvm::SmallVector<mlir::Type, 4> members;
      if (slice && slice->kind == "Tuple") {
        if (const auto *elts = ast::nodeList(*slice, "elts"))
          for (const parser::NodePtr &elt : *elts)
            members.push_back(annotationType(elt.get()));
      } else {
        members.push_back(annotationType(slice));
      }
      return py::UnionType::getNormalized(&context, members);
    }
    if (baseName == "list" || baseName == "List")
      return listOf(annotationType(slice));
    if (baseName == "tuple" || baseName == "Tuple") {
      if (slice && slice->kind == "Tuple") {
        if (const auto *elts = ast::nodeList(*slice, "elts"))
          return tupleOf(elts->empty() ? object()
                                       : annotationType(elts->front().get()));
      }
      return tupleOf(annotationType(slice));
    }
    if (baseName == "dict" || baseName == "Dict" || baseName == "Mapping") {
      mlir::Type key = object();
      mlir::Type value = object();
      if (slice && slice->kind == "Tuple") {
        if (const auto *elts = ast::nodeList(*slice, "elts")) {
          if (!elts->empty())
            key = annotationType(elts->front().get());
          if (elts->size() > 1)
            value = annotationType((*elts)[1].get());
        }
      }
      return dictOf(key, value);
    }
    if (baseName == "type" || baseName == "Type")
      return typeObject(annotationType(slice));
    if (baseName == "Callable") {
      llvm::SmallVector<mlir::Type, 4> positional;
      mlir::Type vararg;
      mlir::Type kwarg;
      mlir::Type result = object();
      if (slice && slice->kind == "Tuple") {
        if (const auto *elts = ast::nodeList(*slice, "elts")) {
          if (!elts->empty()) {
            const parser::Node *params = elts->front().get();
            if (params && params->kind == "List") {
              if (const auto *paramElts = ast::nodeList(*params, "elts"))
                for (const parser::NodePtr &param : *paramElts)
                  positional.push_back(annotationType(param.get()));
            } else if (params && params->kind == "Tuple") {
              if (const auto *paramElts = ast::nodeList(*params, "elts"))
                for (const parser::NodePtr &param : *paramElts)
                  positional.push_back(annotationType(param.get()));
            } else {
              vararg = tupleOf(object());
              kwarg = dictOf(strType(), object());
            }
          }
          if (elts->size() > 1)
            result = annotationType((*elts)[1].get());
        }
      }
      return py::CallableType::get(&context, positional, {}, vararg, kwarg,
                                   {result});
    }
    if (baseName == "Literal")
      return literal(slice && slice->kind == "Constant"
                         ? literalSpelling(*slice)
                         : "object");
  }
  return object();
}

mlir::Type AlgorithmM::inferExpr(const parser::Node *node) const {
  if (!node)
    return object();
  if (node->kind == "Constant") {
    if (ast::isNoneField(*node, "value"))
      return none();
    if (auto value = ast::boolean(*node, "value"))
      return literal(*value ? "True" : "False");
    if (auto value = ast::integer(*node, "value"))
      return literal(std::to_string(*value));
    if (ast::floating(*node, "value"))
      return floatType();
    if (auto value = ast::string(*node, "value"))
      return literal("\"" + std::string(*value) + "\"");
    if (const auto *fieldValue = ast::field(*node, "value"))
      if (const auto *big = std::get_if<parser::BigInteger>(fieldValue))
        return literal(big->decimal);
    return object();
  }
  if (node->kind == "Name") {
    llvm::StringRef name = ast::nameSpelling(*node);
    if (auto found = lookupSymbol(name))
      return *found;
    return object();
  }
  if (node->kind == "Attribute")
    if (mlir::Type objectType = inferExpr(ast::node(*node, "value"))) {
      const py::protocols::Table &table = py::protocols::Table::get(context);
      if (auto attr = ast::string(*node, "attr")) {
        if (std::optional<py::protocols::FieldResolution> field =
                table.resolveFieldContractWithEvidence(widenLiteral(objectType),
                                                       *attr))
          return field->contractType;
        if (std::optional<CallSolution> method =
                tryManifestMethod(*this, widenLiteral(objectType), *attr, {}))
          return method->result;
      }
      return object();
    }
  if (node->kind == "List")
    return listOf(object());
  if (node->kind == "Tuple")
    return tupleOf(object());
  if (node->kind == "Dict")
    return dictOf(object(), object());
  if (node->kind == "Subscript") {
    mlir::Type container = widenLiteral(inferExpr(ast::node(*node, "value")));
    mlir::Type index = inferExpr(ast::node(*node, "slice"));
    if (std::optional<CallSolution> result =
            tryManifestMethod(*this, container, "__getitem__", {index}))
      return result->result;
    return object();
  }
  if (node->kind == "Compare")
    return boolType();
  if (node->kind == "BoolOp")
    return boolType();
  if (node->kind == "UnaryOp") {
    const parser::Node *op = ast::node(*node, "op");
    mlir::Type operand = inferExpr(ast::node(*node, "operand"));
    if (ast::isOperator(op, "Not"))
      return boolType();
    if (ast::isOperator(op, "USub")) {
      const parser::Node *operandNode = ast::node(*node, "operand");
      if (operandNode && operandNode->kind == "Constant") {
        if (auto value = ast::integer(*operandNode, "value"))
          return literal("-" + std::to_string(*value));
        if (const auto *fieldValue = ast::field(*operandNode, "value"))
          if (const auto *big = std::get_if<parser::BigInteger>(fieldValue))
            return literal("-" + big->decimal);
      }
      if (std::optional<CallSolution> result =
              tryManifestMethod(*this, widenLiteral(operand), "__neg__", {}))
        return result->result;
    }
    if (ast::isOperator(op, "UAdd"))
      if (std::optional<CallSolution> result =
              tryManifestMethod(*this, widenLiteral(operand), "__pos__", {}))
        return result->result;
    if (ast::isOperator(op, "Invert"))
      if (std::optional<CallSolution> result =
              tryManifestMethod(*this, widenLiteral(operand), "__invert__", {}))
        return result->result;
    return widenLiteral(operand);
  }
  if (node->kind == "BinOp") {
    const parser::Node *op = ast::node(*node, "op");
    mlir::Type left = widenLiteral(inferExpr(ast::node(*node, "left")));
    mlir::Type right = widenLiteral(inferExpr(ast::node(*node, "right")));
    llvm::StringRef method = "__add__";
    if (ast::isOperator(op, "Sub"))
      method = "__sub__";
    else if (ast::isOperator(op, "Mult"))
      method = "__mul__";
    else if (ast::isOperator(op, "Div"))
      method = "__truediv__";
    else if (ast::isOperator(op, "FloorDiv"))
      method = "__floordiv__";
    else if (ast::isOperator(op, "Mod"))
      method = "__mod__";
    else if (ast::isOperator(op, "LShift"))
      method = "__lshift__";
    else if (ast::isOperator(op, "RShift"))
      method = "__rshift__";
    else if (ast::isOperator(op, "BitAnd"))
      method = "__and__";
    else if (ast::isOperator(op, "BitOr"))
      method = "__or__";
    else if (ast::isOperator(op, "BitXor"))
      method = "__xor__";
    if (std::optional<CallSolution> result =
            tryManifestMethod(*this, left, method, {right}))
      return result->result;
    if (left == strType() && right == strType())
      return strType();
    if (ast::isOperator(op, "Div") &&
        (left == intType() || left == floatType()) &&
        (right == intType() || right == floatType()))
      return floatType();
    if (left == floatType() || right == floatType())
      return floatType();
    if (left == intType() && right == intType())
      return intType();
    return join({left, right});
  }
  if (node->kind == "Call") {
    const parser::Node *callee = ast::node(*node, "func");
    llvm::SmallVector<mlir::Type, 8> positional;
    if (const auto *args = ast::nodeList(*node, "args"))
      for (const parser::NodePtr &arg : *args)
        positional.push_back(inferExpr(arg.get()));

    llvm::SmallVector<CallKeywordType, 4> keywords;
    if (const auto *keywordNodes = ast::nodeList(*node, "keywords")) {
      for (const parser::NodePtr &keyword : *keywordNodes) {
        auto name = ast::string(*keyword, "arg");
        if (!name)
          continue;
        keywords.push_back(CallKeywordType{
            std::string(*name), inferExpr(ast::node(*keyword, "value"))});
      }
    }

    if (callee && callee->kind == "Name") {
      llvm::StringRef name = ast::nameSpelling(*callee);
      if (name == "len") {
        if (!positional.empty())
          if (std::optional<CallSolution> result = tryManifestMethod(
                  *this, widenLiteral(positional.front()), "__len__", {}))
            return result->result;
        return intType();
      }
      if (name == "round") {
        const auto *args = ast::nodeList(*node, "args");
        if (args && !args->empty()) {
          mlir::Type input = widenLiteral(inferExpr(args->front().get()));
          llvm::SmallVector<mlir::Type, 1> extra;
          if (args->size() > 1)
            extra.push_back(inferExpr((*args)[1].get()));
          if (std::optional<CallSolution> result =
                  tryManifestMethod(*this, input, "__round__", extra))
            return result->result;
          if (input == intType())
            return intType();
        }
        return object();
      }
      if (name == "range")
        return contract("builtins.range");
      if (auto cls = lookupClass(name))
        return *cls;
      if (auto symbol = lookupSymbol(name))
        return inferCall(*symbol, positional, keywords);
    }
    if (callee)
      return inferCall(inferExpr(callee), positional, keywords);
    return object();
  }
  if (node->kind == "Lambda") {
    return functionSignature(*node).callable;
  }
  return object();
}

mlir::Type
AlgorithmM::inferCall(mlir::Type calleeType,
                      mlir::ArrayRef<mlir::Type> positional,
                      mlir::ArrayRef<CallKeywordType> keywords) const {
  CallInferenceResult inference =
      inferCallWithEvidence(calleeType, positional, keywords);
  return inference ? inference.resultType : object();
}

CallInferenceResult AlgorithmM::inferMethodCallWithEvidence(
    mlir::Type receiverType, llvm::StringRef methodName,
    mlir::ArrayRef<mlir::Type> positional,
    mlir::ArrayRef<CallKeywordType> keywords) const {
  if (std::optional<CallSolution> result = tryManifestMethod(
          *this, receiverType, methodName, positional, keywords)) {
    return CallInferenceResult{
        result->result,
        CallInferenceEvidence{result->callableContract, result->methodName,
                              result->receiverManifestClass},
        true};
  }
  return CallInferenceResult{
      object(), CallInferenceEvidence{protocol("Callable"), methodName.str(),
                                      std::nullopt}};
}

CallInferenceResult AlgorithmM::inferCallWithEvidence(
    mlir::Type calleeType, mlir::ArrayRef<mlir::Type> positional,
    mlir::ArrayRef<CallKeywordType> keywords) const {
  if (!calleeType)
    return CallInferenceResult{
        object(),
        CallInferenceEvidence{protocol("Callable"), "__call__", std::nullopt}};
  if (auto typeType = mlir::dyn_cast_if_present<py::TypeType>(calleeType))
    return CallInferenceResult{
        typeType.getInstanceType(),
        CallInferenceEvidence{protocol("Callable"), "__call__", std::nullopt},
        true};
  if (auto callable = mlir::dyn_cast_if_present<py::CallableType>(calleeType)) {
    if (std::optional<CallSolution> result =
            tryCallableApplication(*this, callable, positional, keywords))
      return CallInferenceResult{result->result,
                                 CallInferenceEvidence{result->callableContract,
                                                       "__call__",
                                                       std::nullopt},
                                 true};
    return CallInferenceResult{
        object(), CallInferenceEvidence{callable, "__call__", std::nullopt}};
  }
  if (auto overload = mlir::dyn_cast_if_present<py::OverloadType>(calleeType)) {
    llvm::SmallVector<py::CallableType, 4> callables;
    for (mlir::Type candidate : overload.getCandidateTypes()) {
      if (auto callable =
              mlir::dyn_cast_if_present<py::CallableType>(candidate)) {
        callables.push_back(callable);
        continue;
      }
      if (auto typeType = mlir::dyn_cast_if_present<py::TypeType>(candidate))
        if (positional.empty() && keywords.empty())
          return CallInferenceResult{typeType.getInstanceType(),
                                     CallInferenceEvidence{protocol("Callable"),
                                                           "__call__",
                                                           std::nullopt},
                                     true};
    }
    if (std::optional<CallSolution> selected =
            selectCallableApplication(*this, callables, positional, keywords))
      return CallInferenceResult{
          selected->result,
          CallInferenceEvidence{selected->callableContract, "__call__",
                                std::nullopt},
          true};
  }
  if (std::optional<CallSolution> result = tryManifestMethod(
          *this, calleeType, "__call__", positional, keywords))
    return CallInferenceResult{
        result->result,
        CallInferenceEvidence{result->callableContract, result->methodName,
                              result->receiverManifestClass},
        true};
  return CallInferenceResult{
      object(),
      CallInferenceEvidence{protocol("Callable"), "__call__", std::nullopt}};
}

mlir::Type AlgorithmM::join(mlir::ArrayRef<mlir::Type> types) const {
  llvm::SmallVector<mlir::Type, 4> present;
  llvm::SmallVector<mlir::Type, 8> worklist(types.begin(), types.end());
  while (!worklist.empty()) {
    mlir::Type type = worklist.pop_back_val();
    if (!type)
      continue;
    if (isObjectTop(*this, type)) {
      present.clear();
      present.push_back(object());
      continue;
    }
    if (isObjectTop(*this, present.empty() ? mlir::Type{} : present.front()))
      continue;
    if (auto unionType = mlir::dyn_cast_if_present<py::UnionType>(type)) {
      for (mlir::Type member : unionType.getMemberTypes())
        worklist.push_back(member);
      continue;
    }
    if (!llvm::is_contained(present, type))
      present.push_back(type);
  }
  if (present.empty())
    return object();
  if (present.size() == 1)
    return present.front();
  return py::UnionType::getNormalized(&context, present);
}

mlir::Type AlgorithmM::widenLiteral(mlir::Type type) const {
  auto literalType = mlir::dyn_cast_or_null<py::LiteralType>(type);
  if (!literalType)
    return type ? type : object();
  llvm::StringRef spelling = literalType.getSpelling();
  if (spelling == "True" || spelling == "False")
    return boolType();
  if (spelling == "None")
    return none();
  if (!spelling.empty() && spelling.front() == '"')
    return strType();
  return intType();
}

FunctionSignature
AlgorithmM::functionSignature(const parser::Node &function,
                              std::optional<llvm::StringRef> selfName,
                              py::CallableType expectedCallable) const {
  FunctionSignature sig;
  const parser::Node *arguments = ast::node(function, "args");
  if (arguments) {
    unsigned positionalOnlyCount = 0;
    llvm::SmallVector<parser::NodePtr, 8> positional =
        concatArgs(*arguments, positionalOnlyCount);
    sig.positionalOnlyCount = positionalOnlyCount;
    std::size_t defaults = ast::nodeList(*arguments, "defaults")
                               ? ast::nodeList(*arguments, "defaults")->size()
                               : 0;
    llvm::ArrayRef<mlir::Type> expectedPositional =
        expectedCallable ? expectedCallable.getPositionalTypes()
                         : llvm::ArrayRef<mlir::Type>();
    llvm::ArrayRef<mlir::Type> expectedKwOnly =
        expectedCallable ? expectedCallable.getKwOnlyTypes()
                         : llvm::ArrayRef<mlir::Type>();
    for (auto [index, arg] : llvm::enumerate(positional)) {
      std::string name(ast::nameSpelling(*arg));
      mlir::Type type = annotationType(ast::node(*arg, "annotation"));
      if (selfName && index == 0 && name == *selfName)
        type = py::SelfType::get(&context);
      if (function.kind == "Lambda" && index < expectedPositional.size())
        type = expectedPositional[index];
      sig.positionalNames.push_back(std::move(name));
      sig.positionalTypes.push_back(type);
      sig.positionalDefaults.push_back(
          hasDefault(index, positional.size(), defaults));
    }

    if (const auto *kwonly = ast::nodeList(*arguments, "kwonlyargs")) {
      std::size_t index = 0;
      for (const parser::NodePtr &arg : *kwonly) {
        sig.kwOnlyNames.push_back(std::string(ast::nameSpelling(*arg)));
        mlir::Type type = annotationType(ast::node(*arg, "annotation"));
        if (function.kind == "Lambda" && index < expectedKwOnly.size())
          type = expectedKwOnly[index];
        sig.kwOnlyTypes.push_back(type);
        bool hasKwDefault = false;
        if (const auto *kwDefaults = ast::nodeList(*arguments, "kw_defaults"))
          hasKwDefault = index < kwDefaults->size() && (*kwDefaults)[index];
        sig.kwOnlyDefaults.push_back(hasKwDefault);
        ++index;
      }
    }
    if (const parser::Node *vararg = ast::node(*arguments, "vararg")) {
      sig.varargName = std::string(ast::nameSpelling(*vararg));
      sig.varargType = annotationType(ast::node(*vararg, "annotation"));
    }
    if (const parser::Node *kwarg = ast::node(*arguments, "kwarg")) {
      sig.kwargName = std::string(ast::nameSpelling(*kwarg));
      sig.kwargType = annotationType(ast::node(*kwarg, "annotation"));
    }
  }

  auto scope = pushScope();
  for (auto [index, name] : llvm::enumerate(sig.positionalNames))
    bindLocalSymbol(name, sig.positionalTypes[index]);
  for (auto [index, name] : llvm::enumerate(sig.kwOnlyNames))
    bindLocalSymbol(name, sig.kwOnlyTypes[index]);
  if (sig.varargName)
    bindLocalSymbol(*sig.varargName, sig.varargType);
  if (sig.kwargName)
    bindLocalSymbol(*sig.kwargName, sig.kwargType);

  if (function.kind == "Lambda") {
    sig.resultType = inferExpr(ast::node(function, "body"));
  } else if (const parser::Node *returns = ast::node(function, "returns")) {
    sig.resultType = annotationType(returns);
  } else {
    sig.resultType = ast::nameSpelling(function) == "__init__"
                         ? none()
                         : inferredFunctionResult(*this, function);
  }
  refreshCallable(sig);
  return sig;
}

void AlgorithmM::refreshCallable(FunctionSignature &sig) const {
  llvm::SmallVector<mlir::StringAttr, 8> posNames;
  llvm::SmallVector<mlir::StringAttr, 4> kwNames;
  llvm::SmallVector<mlir::BoolAttr, 8> posDefaults;
  llvm::SmallVector<mlir::BoolAttr, 4> kwDefaults;
  for (const std::string &name : sig.positionalNames)
    posNames.push_back(mlir::StringAttr::get(&context, name));
  for (const std::string &name : sig.kwOnlyNames)
    kwNames.push_back(mlir::StringAttr::get(&context, name));
  for (bool value : sig.positionalDefaults)
    posDefaults.push_back(mlir::BoolAttr::get(&context, value));
  for (bool value : sig.kwOnlyDefaults)
    kwDefaults.push_back(mlir::BoolAttr::get(&context, value));

  llvm::SmallVector<mlir::Type, 1> results{sig.resultType};
  sig.callable = py::CallableType::get(
      &context, sig.positionalTypes, sig.kwOnlyTypes, sig.varargType,
      sig.kwargType, results, posNames, kwNames, posDefaults, kwDefaults,
      sig.varargName ? mlir::StringAttr::get(&context, *sig.varargName)
                     : mlir::StringAttr(),
      sig.kwargName ? mlir::StringAttr::get(&context, *sig.kwargName)
                    : mlir::StringAttr(),
      sig.positionalOnlyCount);
}

} // namespace lython::emitter
