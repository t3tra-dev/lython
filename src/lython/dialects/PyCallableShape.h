#ifndef LYTHON_PY_CALLABLE_SHAPE_H
#define LYTHON_PY_CALLABLE_SHAPE_H

#include "CallableArgumentMatcher.h"
#include "PyDialectTypes.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace py {

struct CallableSignatureShape {
  llvm::SmallVector<mlir::Type, 8> positional;
  llvm::SmallVector<mlir::Type, 4> kwonly;
  std::optional<mlir::Type> vararg;
  std::optional<mlir::Type> kwarg;
  std::vector<std::string> positionalNames;
  std::vector<char> positionalDefaults;
  std::vector<std::string> kwonlyNames;
  std::vector<char> kwonlyDefaults;
  unsigned positionalOnlyCount = 0;

  lython::callable::Signature<mlir::Type> signature() const {
    return {positional,
            kwonly,
            vararg,
            kwarg,
            llvm::ArrayRef<std::string>(positionalNames),
            llvm::ArrayRef<char>(positionalDefaults),
            llvm::ArrayRef<std::string>(kwonlyNames),
            llvm::ArrayRef<char>(kwonlyDefaults),
            positionalOnlyCount};
  }
};

struct CallableApplicationShapeOptions {
  std::size_t firstParameter = 0;
  std::optional<std::vector<std::string>> positionalNames;
  std::optional<std::vector<char>> positionalDefaults;
  std::optional<std::vector<std::string>> kwonlyNames;
  std::optional<std::vector<char>> kwonlyDefaults;
};

struct CallableApplicationShapeResolution {
  CallableSignatureShape shape;
  int score = 0;
};

inline std::optional<CallableSignatureShape>
callableSignatureShape(CallableType callable, std::size_t firstParameter = 0) {
  if (!callable)
    return std::nullopt;

  llvm::ArrayRef<mlir::Type> positional = callable.getPositionalTypes();
  if (firstParameter > positional.size())
    return std::nullopt;

  CallableSignatureShape shape;
  shape.positional.append(positional.begin() +
                              static_cast<std::ptrdiff_t>(firstParameter),
                          positional.end());
  shape.kwonly.append(callable.getKwOnlyTypes().begin(),
                      callable.getKwOnlyTypes().end());

  if (callable.hasVararg())
    shape.vararg = callable.getVarargType();
  if (callable.hasKwarg())
    shape.kwarg = callable.getKwargType();

  llvm::ArrayRef<mlir::StringAttr> positionalNames =
      callable.getPositionalNames();
  shape.positionalNames.resize(shape.positional.size());
  for (std::size_t index = 0; index < shape.positional.size(); ++index) {
    std::size_t fullIndex = index + firstParameter;
    if (fullIndex < positionalNames.size() && positionalNames[fullIndex])
      shape.positionalNames[index] =
          positionalNames[fullIndex].getValue().str();
  }

  llvm::ArrayRef<mlir::BoolAttr> positionalDefaults =
      callable.getPositionalDefaults();
  shape.positionalDefaults.resize(shape.positional.size(), 0);
  for (std::size_t index = 0; index < shape.positional.size(); ++index) {
    std::size_t fullIndex = index + firstParameter;
    if (fullIndex < positionalDefaults.size() &&
        positionalDefaults[fullIndex].getValue())
      shape.positionalDefaults[index] = 1;
  }

  shape.kwonlyNames.resize(shape.kwonly.size());
  llvm::ArrayRef<mlir::StringAttr> kwonlyNames = callable.getKwOnlyNames();
  for (std::size_t index = 0; index < shape.kwonly.size(); ++index)
    if (index < kwonlyNames.size() && kwonlyNames[index])
      shape.kwonlyNames[index] = kwonlyNames[index].getValue().str();

  shape.kwonlyDefaults.resize(shape.kwonly.size(), 0);
  llvm::ArrayRef<mlir::BoolAttr> kwonlyDefaults = callable.getKwOnlyDefaults();
  for (std::size_t index = 0; index < shape.kwonly.size(); ++index)
    if (index < kwonlyDefaults.size() && kwonlyDefaults[index].getValue())
      shape.kwonlyDefaults[index] = 1;

  shape.positionalOnlyCount = callable.getPositionalOnlyCount();
  if (shape.positionalOnlyCount > firstParameter)
    shape.positionalOnlyCount -= static_cast<unsigned>(firstParameter);
  else
    shape.positionalOnlyCount = 0;

  return shape;
}

inline void applyCallableApplicationShapeOptions(
    CallableSignatureShape &shape,
    const CallableApplicationShapeOptions &opts) {
  if (opts.positionalNames)
    shape.positionalNames = *opts.positionalNames;
  if (opts.positionalDefaults)
    shape.positionalDefaults = *opts.positionalDefaults;
  if (opts.kwonlyNames)
    shape.kwonlyNames = *opts.kwonlyNames;
  if (opts.kwonlyDefaults)
    shape.kwonlyDefaults = *opts.kwonlyDefaults;
}

inline lython::callable::VarargShape<mlir::Type>
callableVarargShape(mlir::Type varargType) {
  if (auto unpack = mlir::dyn_cast_if_present<UnpackType>(varargType)) {
    mlir::Type packed = unpack.getPackedType();
    if (mlir::isa<TypeVarTupleType>(packed))
      return lython::callable::VarargShape<mlir::Type>::repeatedOf(
          pyObjectContractType(varargType.getContext()));
    return callableVarargShape(packed);
  }
  if (auto pack = mlir::dyn_cast_if_present<CallableType>(varargType)) {
    if (!pack.getPositionalTypes().empty() && !pack.hasVararg())
      return lython::callable::VarargShape<mlir::Type>::exactOf(
          pack.getPositionalTypes());
    if (pack.hasVararg())
      return callableVarargShape(pack.getVarargType());
    return lython::callable::VarargShape<mlir::Type>::exactOf({});
  }
  if (auto contract = mlir::dyn_cast_if_present<ContractType>(varargType)) {
    if (contract.getContractName() == "builtins.tuple") {
      llvm::ArrayRef<mlir::Type> arguments = contract.getArguments();
      if (arguments.size() == 1)
        return lython::callable::VarargShape<mlir::Type>::repeatedOf(
            arguments.front());
      return lython::callable::VarargShape<mlir::Type>::exactOf(arguments);
    }
  }
  return lython::callable::VarargShape<mlir::Type>::invalid();
}

inline std::optional<mlir::Type> callableKwargValueType(mlir::Type kwargType) {
  if (auto pack = mlir::dyn_cast_if_present<CallableType>(kwargType)) {
    llvm::SmallVector<mlir::Type, 8> valueTypes;
    if (!pack.getPositionalNames().empty())
      valueTypes.append(pack.getPositionalTypes().begin(),
                        pack.getPositionalTypes().end());
    valueTypes.append(pack.getKwOnlyTypes().begin(),
                      pack.getKwOnlyTypes().end());
    if (pack.hasKwarg()) {
      std::optional<mlir::Type> fallback =
          callableKwargValueType(pack.getKwargType());
      if (!fallback)
        return std::nullopt;
      valueTypes.push_back(*fallback);
    }
    if (valueTypes.empty())
      return pyObjectContractType(kwargType.getContext());
    return UnionType::getNormalized(kwargType.getContext(), valueTypes);
  }
  if (auto contract = mlir::dyn_cast_if_present<ContractType>(kwargType)) {
    if (contract.getContractName() == "builtins.dict") {
      llvm::ArrayRef<mlir::Type> arguments = contract.getArguments();
      if (arguments.size() < 2)
        return pyObjectContractType(kwargType.getContext());
      auto key = mlir::dyn_cast_if_present<ContractType>(arguments.front());
      if (!key || key.getContractName() != "builtins.str")
        return std::nullopt;
      return arguments[1];
    }
  }
  return kwargType;
}

inline std::optional<mlir::Type>
callableUnpackedKeywordValueType(mlir::Type mappingType) {
  if (auto contract = mlir::dyn_cast_if_present<ContractType>(mappingType)) {
    if (contract.getContractName() == "builtins.dict") {
      llvm::ArrayRef<mlir::Type> arguments = contract.getArguments();
      if (arguments.size() >= 2) {
        auto key = mlir::dyn_cast_if_present<ContractType>(arguments.front());
        if (key && key.getContractName() == "builtins.str")
          return arguments[1];
      }
    }
  }
  auto protocol = mlir::dyn_cast_if_present<ProtocolType>(mappingType);
  if (protocol &&
      (protocol.getProtocolName() == "Mapping" ||
       protocol.getProtocolName() == "MutableMapping") &&
      protocol.getArguments().size() >= 2 &&
      isPyStrType(protocol.getArguments().front()))
    return protocol.getArguments()[1];
  return std::nullopt;
}

template <typename Keyword, typename MatchExpected, typename KeywordName,
          typename KeywordValue, typename MatchObserver>
bool matchCallableInvocationWithObserver(
    const CallableSignatureShape &shape, llvm::ArrayRef<mlir::Type> positional,
    llvm::ArrayRef<Keyword> keywords, MatchExpected matchExpected,
    KeywordName keywordName, KeywordValue keywordValue, MatchObserver &observer,
    bool requireSatisfiedDefaults = true) {
  return lython::callable::matchInvocationWithObserver(
      shape.signature(), positional, keywords, std::move(matchExpected),
      [](mlir::Type expected) { return callableVarargShape(expected); },
      [](mlir::Type expected) -> std::optional<mlir::Type> {
        return callableKwargValueType(expected);
      },
      [](mlir::Type actual) -> std::optional<mlir::Type> {
        return callableUnpackedKeywordValueType(actual);
      },
      std::move(keywordName), std::move(keywordValue), observer,
      requireSatisfiedDefaults);
}

template <typename Keyword, typename MatchExpected, typename KeywordName,
          typename KeywordValue>
std::optional<CallableApplicationShapeResolution>
resolveCallableApplicationShape(const CallableSignatureShape &signatureShape,
                                llvm::ArrayRef<mlir::Type> positional,
                                llvm::ArrayRef<Keyword> keywords,
                                MatchExpected matchExpected,
                                KeywordName keywordName,
                                KeywordValue keywordValue,
                                bool requireSatisfiedDefaults = true) {
  lython::callable::InvocationSpecificityScore observer;
  bool matched = matchCallableInvocationWithObserver(
      signatureShape, positional, keywords, std::move(matchExpected),
      std::move(keywordName), std::move(keywordValue), observer,
      requireSatisfiedDefaults);
  if (!matched)
    return std::nullopt;
  return CallableApplicationShapeResolution{signatureShape, observer.score};
}

template <typename Keyword, typename MatchExpected, typename KeywordName,
          typename KeywordValue>
std::optional<CallableApplicationShapeResolution>
resolveCallableApplicationShape(CallableType callable,
                                llvm::ArrayRef<mlir::Type> positional,
                                llvm::ArrayRef<Keyword> keywords,
                                const CallableApplicationShapeOptions &opts,
                                MatchExpected matchExpected,
                                KeywordName keywordName,
                                KeywordValue keywordValue,
                                bool requireSatisfiedDefaults = true) {
  std::optional<CallableSignatureShape> shape =
      callableSignatureShape(callable, opts.firstParameter);
  if (!shape)
    return std::nullopt;
  applyCallableApplicationShapeOptions(*shape, opts);
  return resolveCallableApplicationShape(
      *shape, positional, keywords, std::move(matchExpected),
      std::move(keywordName), std::move(keywordValue),
      requireSatisfiedDefaults);
}

struct CallableKeyword {
  std::string name;
  mlir::Type type;
};

using CallableInvocation =
    lython::callable::Invocation<mlir::Type, CallableKeyword>;

inline bool appendCallableAcceptanceSamples(
    CallableType callable, llvm::SmallVectorImpl<CallableInvocation> &samples) {
  std::optional<CallableSignatureShape> shape =
      callableSignatureShape(callable);
  if (!shape)
    return false;
  return lython::callable::appendAcceptanceSamples<mlir::Type, CallableKeyword>(
      shape->signature(), samples, [](llvm::StringRef name, mlir::Type value) {
        return CallableKeyword{name.str(), value};
      });
}

} // namespace py

#endif // LYTHON_PY_CALLABLE_SHAPE_H
