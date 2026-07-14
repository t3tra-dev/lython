#include "TypeSystemSolver.h"

#include "CandidateSelection.h"
#include "PyCallableShape.h"
#include "PyProtocols.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"

#include <algorithm>
#include <utility>

namespace lython::emitter {

bool bindExpectedType(const TypeSystem &types, mlir::Type expected,
                      mlir::Type actual, TypeBindingMap &bindings);

std::optional<py::CallableType>
expandParamSpecForInvocation(const TypeSystem &types, py::CallableType callable,
                             mlir::ArrayRef<mlir::Type> positional,
                             mlir::ArrayRef<CallKeywordType> keywords,
                             TypeBindingMap &bindings,
                             std::size_t firstParameter = 0);

std::optional<std::string> staticParameterName(mlir::Type type) {
  if (mlir::isa<py::SelfType>(type))
    return std::string("Self");
  if (auto typeVar = mlir::dyn_cast_if_present<py::TypeVarType>(type))
    return typeVar.getName().str();
  if (auto paramSpec = mlir::dyn_cast_if_present<py::ParamSpecType>(type))
    return paramSpec.getName().str();
  if (auto typeVarTuple = mlir::dyn_cast_if_present<py::TypeVarTupleType>(type))
    return typeVarTuple.getName().str();
  if (auto contract = mlir::dyn_cast_if_present<py::ContractType>(type)) {
    llvm::StringRef name = contract.getContractName();
    if (name.starts_with("$"))
      return name.drop_front().str();
  }
  return std::nullopt;
}

bool structuralProtocolAccepts(const TypeSystem &types,
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

bool typeAccepts(const TypeSystem &types, mlir::Type expected,
                 mlir::Type actual) {
  // Zonk only; never unify here. typeAccepts is called speculatively (both
  // directions in bindTypeParameter, per-member in union trials), so a
  // binding side effect would corrupt the store on the rejected branch. An
  // unbound variable that survives zonk therefore falls through to the
  // ground checks and is rejected -- the sound default.
  expected = types.inference().zonk(expected);
  actual = types.inference().zonk(actual);
  expected = types.widenLiteral(expected);
  actual = types.widenLiteral(actual);
  if (!expected || !actual)
    return true;
  if (expected == actual)
    return true;
  if (isObjectTop(types, expected) || isObjectTop(types, actual))
    return true;
  if (auto expectedContract =
          mlir::dyn_cast_if_present<py::ContractType>(expected)) {
    const py::protocols::Table &table =
        py::protocols::Table::get(types.getContext());
    if (table.isManifestSubclassOf(actual, expectedContract.getContractName()))
      return true;
    if (structuralProtocolAccepts(types, expectedContract, actual))
      return true;
  }
  return py::isAssignableTo(actual, expected);
}

std::optional<std::string> paramSpecName(mlir::Type type) {
  if (auto paramSpec = mlir::dyn_cast_if_present<py::ParamSpecType>(type))
    return paramSpec.getName().str();
  return std::nullopt;
}

std::optional<std::string> typeVarTupleName(mlir::Type type) {
  if (auto typeVarTuple = mlir::dyn_cast_if_present<py::TypeVarTupleType>(type))
    return typeVarTuple.getName().str();
  return std::nullopt;
}

std::optional<std::string> unpackedTypeVarTupleName(mlir::Type type) {
  auto unpack = mlir::dyn_cast_if_present<py::UnpackType>(type);
  if (!unpack)
    return std::nullopt;
  return typeVarTupleName(unpack.getPackedType());
}

llvm::ArrayRef<mlir::Type> typeVarTupleElements(mlir::Type type) {
  if (auto pack = mlir::dyn_cast_if_present<py::CallableType>(type))
    return pack.getPositionalTypes();
  if (auto tuple = mlir::dyn_cast_if_present<py::ContractType>(type))
    if (tuple.getContractName() == "builtins.tuple")
      return tuple.getArguments();
  return {};
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

bool bindParamSpecPack(const TypeSystem &types, llvm::StringRef name,
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

bool bindTypeVarTuplePack(const TypeSystem &types, llvm::StringRef name,
                          mlir::ArrayRef<mlir::Type> positional,
                          TypeBindingMap &bindings) {
  if (positional.size() == 1)
    if (std::optional<std::string> forwarded =
            unpackedTypeVarTupleName(positional.front()))
      if (*forwarded == name)
        return true;
  mlir::Type pack =
      py::CallableType::get(&types.getContext(), positional, {}, {}, {}, {});
  auto found = bindings.find(name.str());
  if (found == bindings.end()) {
    bindings[name.str()] = pack;
    return true;
  }
  return found->second == pack;
}

bool bindTypeParameter(const TypeSystem &types, llvm::StringRef name,
                       mlir::Type actual, TypeBindingMap &bindings) {
  actual = types.widenLiteral(actual);
  if (!actual)
    return false;
  auto found = bindings.find(name.str());
  if (found == bindings.end()) {
    bindings[name.str()] = actual;
    return true;
  }
  if (std::optional<std::string> existing = staticParameterName(found->second))
    if (*existing == name) {
      found->second = actual;
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

bool bindTypeList(const TypeSystem &types, mlir::ArrayRef<mlir::Type> expected,
                  mlir::ArrayRef<mlir::Type> actual, TypeBindingMap &bindings) {
  if (expected.size() != actual.size())
    return false;
  for (auto [expectedType, actualType] : llvm::zip(expected, actual))
    if (!bindExpectedType(types, expectedType, actualType, bindings))
      return false;
  return true;
}

bool bindUnionMember(const TypeSystem &types, py::UnionType expected,
                     mlir::Type actual, TypeBindingMap &bindings) {
  for (mlir::Type member : expected.getMemberTypes()) {
    TypeBindingMap candidate = bindings;
    InferenceContext::Speculation attempt(types.inference());
    if (bindExpectedType(types, member, actual, candidate)) {
      attempt.commit();
      bindings = std::move(candidate);
      return true;
    }
  }
  return false;
}

bool bindProtocolView(const TypeSystem &types, py::ProtocolType expected,
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

bool bindContractView(const TypeSystem &types, py::ContractType expected,
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

bool bindCallableView(const TypeSystem &types, py::CallableType expected,
                      py::CallableType actual, TypeBindingMap &bindings) {
  llvm::SmallVector<CallKeywordType, 4> actualKeywords;
  llvm::ArrayRef<mlir::StringAttr> actualKwNames = actual.getKwOnlyNames();
  for (auto [index, type] : llvm::enumerate(actual.getKwOnlyTypes())) {
    std::string name;
    if (index < actualKwNames.size() && actualKwNames[index])
      name = actualKwNames[index].getValue().str();
    actualKeywords.push_back(CallKeywordType{std::move(name), type});
  }

  TypeBindingMap candidateBindings = bindings;
  std::optional<py::CallableType> expandedExpected =
      expandParamSpecForInvocation(types, expected, actual.getPositionalTypes(),
                                   actualKeywords, candidateBindings);
  if (!expandedExpected)
    return typeAccepts(types, expected, actual);
  expected = *expandedExpected;

  if (expected.getPositionalTypes().size() !=
          actual.getPositionalTypes().size() ||
      expected.getKwOnlyTypes().size() != actual.getKwOnlyTypes().size() ||
      expected.getResultTypes().size() != actual.getResultTypes().size())
    return typeAccepts(types, expected, actual);

  for (auto [expectedArg, actualArg] :
       llvm::zip(expected.getPositionalTypes(), actual.getPositionalTypes()))
    if (!bindExpectedType(types, expectedArg, actualArg, candidateBindings))
      return false;
  for (auto [expectedArg, actualArg] :
       llvm::zip(expected.getKwOnlyTypes(), actual.getKwOnlyTypes()))
    if (!bindExpectedType(types, expectedArg, actualArg, candidateBindings))
      return false;
  for (auto [expectedResult, actualResult] :
       llvm::zip(expected.getResultTypes(), actual.getResultTypes()))
    if (!bindExpectedType(types, expectedResult, actualResult,
                          candidateBindings))
      return false;
  bindings = std::move(candidateBindings);
  return true;
}

bool bindExpectedType(const TypeSystem &types, mlir::Type expected,
                      mlir::Type actual, TypeBindingMap &bindings) {
  expected = substituteType(types, expected, bindings);
  expected = types.inference().zonk(expected);
  actual = types.inference().zonk(actual);
  actual = types.widenLiteral(actual);

  if (!expected || !actual)
    return true;
  if (std::optional<std::string> name = staticParameterName(expected))
    return bindTypeParameter(types, *name, actual, bindings);
  if (expected == actual)
    return true;
  if (isObjectTop(types, expected) || isObjectTop(types, actual))
    return true;
  // A variable heading either side after zonk is an engine-owned unknown;
  // binding it is unify's job (occurs check included), not the name-keyed
  // manifest map's. Checked after the object-top acceptors on purpose: a
  // match against the top type carries no information, so the variable must
  // stay free rather than get polluted with object().
  if (py::isPyInferVarType(expected) || py::isPyInferVarType(actual))
    return static_cast<bool>(types.inference().unify(expected, actual));

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

py::CallableType substituteCallable(const TypeSystem &types,
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
  auto boundTypeVarTuplePack =
      [&](llvm::StringRef name) -> std::optional<llvm::ArrayRef<mlir::Type>> {
    auto found = bindings.find(name.str());
    if (found == bindings.end())
      return std::nullopt;
    return typeVarTupleElements(found->second);
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
    if (std::optional<std::string> name = unpackedTypeVarTupleName(arg)) {
      std::optional<llvm::ArrayRef<mlir::Type>> pack =
          boundTypeVarTuplePack(*name);
      if (pack) {
        for (mlir::Type packArg : *pack) {
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
    } else if (std::optional<std::string> tupleName =
                   unpackedTypeVarTupleName(varargType)) {
      std::optional<llvm::ArrayRef<mlir::Type>> tuplePack =
          boundTypeVarTuplePack(*tupleName);
      if (tuplePack) {
        for (mlir::Type packArg : *tuplePack) {
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

mlir::Type substituteType(const TypeSystem &types, mlir::Type type,
                          const TypeBindingMap &bindings, bool eraseUnbound) {
  return py::mapPyTypeStructure(
      type, [&](mlir::Type node) -> std::optional<mlir::Type> {
        if (std::optional<std::string> name = staticParameterName(node)) {
          auto found = bindings.find(*name);
          if (found != bindings.end())
            return found->second;
          // An unbound type parameter with eraseUnbound set becomes an
          // `object` top (e.g. an unspecialized generic call result or
          // storage crossing). This is not an invalid-operation fallback:
          // the erased top carries no protocol contract, so any later
          // observation of the value requires fresh evidence it cannot
          // obtain and is rejected at the static boundary rather than
          // silently dispatched.
          return eraseUnbound ? types.object() : node;
        }
        if (auto callable = mlir::dyn_cast<py::CallableType>(node))
          return mlir::Type(
              substituteCallable(types, callable, bindings, eraseUnbound));
        return std::nullopt;
      });
}

mlir::Type callableResultType(const TypeSystem &types,
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
expandParamSpecForInvocation(const TypeSystem &types, py::CallableType callable,
                             mlir::ArrayRef<mlir::Type> positional,
                             mlir::ArrayRef<CallKeywordType> keywords,
                             TypeBindingMap &bindings,
                             std::size_t firstParameter) {
  llvm::ArrayRef<mlir::Type> formals = callable.getPositionalTypes();
  std::optional<std::size_t> paramSpecIndex;
  std::optional<std::size_t> typeVarTupleIndex;
  for (std::size_t index = firstParameter, end = formals.size(); index < end;
       ++index) {
    bool isParamSpec = static_cast<bool>(paramSpecName(formals[index]));
    bool isTypeVarTuple =
        static_cast<bool>(unpackedTypeVarTupleName(formals[index]));
    if (!isParamSpec && !isTypeVarTuple)
      continue;
    if (isParamSpec && (paramSpecIndex || typeVarTupleIndex))
      return std::nullopt;
    if (isTypeVarTuple && (paramSpecIndex || typeVarTupleIndex))
      return std::nullopt;
    if (isParamSpec)
      paramSpecIndex = index;
    else
      typeVarTupleIndex = index;
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
  } else if (typeVarTupleIndex) {
    std::size_t visibleBefore = *typeVarTupleIndex - firstParameter;
    std::size_t trailing = formals.size() - *typeVarTupleIndex - 1;
    if (positional.size() < visibleBefore + trailing)
      return std::nullopt;

    std::size_t capturedCount = positional.size() - visibleBefore - trailing;
    llvm::ArrayRef<mlir::Type> captured =
        positional.slice(visibleBefore, capturedCount);
    std::optional<std::string> name =
        unpackedTypeVarTupleName(formals[*typeVarTupleIndex]);
    if (!name || !bindTypeVarTuplePack(types, *name, captured, bindings))
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
    } else if (std::optional<std::string> tupleName =
                   unpackedTypeVarTupleName(callable.getVarargType())) {
      std::size_t fixedVisible = formals.size() - firstParameter;
      llvm::ArrayRef<mlir::Type> captured;
      if (positional.size() > fixedVisible)
        captured = positional.drop_front(fixedVisible);
      if (!bindTypeVarTuplePack(types, *tupleName, captured, bindings))
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
  if (auto unpack = mlir::dyn_cast_if_present<py::UnpackType>(type))
    return unboundStaticParameterCount(unpack.getPackedType());
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

int matchSpecificity(const TypeSystem &types, mlir::Type expected,
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

int callableSpecificity(const TypeSystem &types, py::CallableType callable,
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
tryCallableApplication(const TypeSystem &types, py::CallableType callable,
                       mlir::ArrayRef<mlir::Type> positional,
                       mlir::ArrayRef<CallKeywordType> keywords,
                       TypeBindingMap bindings, std::size_t firstParameter) {
  // The name-keyed bindings map is call-local, but inference-variable
  // bindings made through bindExpectedType land in the shared union-find
  // store; a failed application must not leave them behind.
  InferenceContext::Speculation speculation(types.inference());
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
  py::CallableType resolvedEvidence =
      substituteCallable(types, *expanded, bindings, /*eraseUnbound=*/false);
  if (!resolvedEvidence || unboundStaticParameterCount(resolvedEvidence) != 0)
    return std::nullopt;
  mlir::Type result = callableResultType(types, *expanded, bindings);
  if (!result || unboundStaticParameterCount(result) != 0)
    return std::nullopt;
  speculation.commit();
  return CallSolution{result,
                      std::move(bindings),
                      resolvedEvidence,
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
    const TypeSystem &types, llvm::ArrayRef<py::CallableType> candidates,
    mlir::ArrayRef<mlir::Type> positional,
    mlir::ArrayRef<CallKeywordType> keywords, TypeBindingMap bindings,
    std::size_t firstParameter) {
  struct RankedCandidate {
    CallSolution solution;
    py::CallableType candidate;
  };
  auto selection = lython::selection::bestCandidate<RankedCandidate>(
      [](const RankedCandidate &ranked) { return ranked.solution.score; },
      [](const RankedCandidate &lhs, const RankedCandidate &rhs) {
        return sameCallSolution(lhs.solution, rhs.solution);
      });
  // Every candidate ranks against the same store state and only the winner
  // re-applies for real afterwards: a losing candidate's inference-variable
  // bindings must not skew (or outlive) the other attempts.
  for (py::CallableType candidate : candidates) {
    InferenceContext::Speculation attempt(types.inference());
    if (std::optional<CallSolution> solution = tryCallableApplication(
            types, candidate, positional, keywords, bindings, firstParameter))
      selection.consider(RankedCandidate{std::move(*solution), candidate});
  }
  std::optional<RankedCandidate> best = std::move(selection).finish();
  if (!best)
    return std::nullopt;
  return tryCallableApplication(types, best->candidate, positional, keywords,
                                std::move(bindings), firstParameter);
}

std::optional<CallSolution>
tryManifestMethod(const TypeSystem &types, mlir::Type receiverType,
                  llvm::StringRef methodName,
                  mlir::ArrayRef<mlir::Type> positional,
                  mlir::ArrayRef<CallKeywordType> keywords) {
  // Positional tuple typing: a tuple contract carrying one argument PER
  // POSITION (heterogeneous annotations/literals, dict.items()'s
  // tuple[$K,$V]) types a literal-index __getitem__ as that position's
  // member. The homogeneous view (joined members) instantiates the manifest
  // contract; only the RESULT narrows to the indexed member.
  if (methodName == "__getitem__" && positional.size() == 1 &&
      keywords.empty()) {
    auto tuple = mlir::dyn_cast_if_present<py::ContractType>(
        types.widenLiteral(receiverType));
    if (tuple && tuple.getContractName() == "builtins.tuple" &&
        tuple.getArguments().size() > 1) {
      if (std::optional<std::int64_t> index =
              literalIntegerFromType(positional.front())) {
        llvm::ArrayRef<mlir::Type> members = tuple.getArguments();
        std::int64_t position =
            *index < 0 ? *index + static_cast<std::int64_t>(members.size())
                       : *index;
        if (position >= 0 &&
            position < static_cast<std::int64_t>(members.size())) {
          mlir::Type joinedView = types.tupleOf(types.join(members));
          if (std::optional<CallSolution> solution = tryManifestMethod(
                  types, joinedView, methodName, positional, keywords)) {
            solution->result = members[position];
            return solution;
          }
        }
      }
    }
  }

  const py::protocols::Table &table =
      py::protocols::Table::get(types.getContext());
  struct RankedResolution {
    CallSolution solution;
    py::protocols::ContractResolution candidate;
  };
  auto selection = lython::selection::bestCandidate<RankedResolution>(
      [](const RankedResolution &ranked) { return ranked.solution.score; },
      [](const RankedResolution &lhs, const RankedResolution &rhs) {
        return sameCallSolution(lhs.solution, rhs.solution);
      });

  auto apply = [&](const py::protocols::ContractResolution &candidate)
      -> std::optional<CallSolution> {
    py::CallableType signature = candidate.method.signature;
    if (signature.getPositionalTypes().empty())
      return std::nullopt;
    TypeBindingMap bindings = candidate.typeBindings;
    if (!bindExpectedType(types, signature.getPositionalTypes().front(),
                          receiverType, bindings))
      return std::nullopt;
    std::optional<CallSolution> solution = tryCallableApplication(
        types, signature, positional, keywords, std::move(bindings),
        /*firstParameter=*/1);
    if (!solution)
      return std::nullopt;
    solution->score += candidate.score;
    solution->methodName = methodName.str();
    if (candidate.receiverEvidence)
      solution->receiverManifestClass =
          candidate.receiverEvidence->manifestClass;
    return solution;
  };

  // Same explore-then-reapply shape as selectCallableApplication: candidate
  // ranking must not leave inference-variable bindings behind.
  for (py::protocols::ContractResolution candidate :
       table.methodContractCandidatesWithEvidence(receiverType, methodName)) {
    InferenceContext::Speculation attempt(types.inference());
    if (std::optional<CallSolution> solution = apply(candidate))
      selection.consider(
          RankedResolution{std::move(*solution), std::move(candidate)});
  }
  std::optional<RankedResolution> best = std::move(selection).finish();
  if (!best)
    return std::nullopt;
  return apply(best->candidate);
}

} // namespace lython::emitter
