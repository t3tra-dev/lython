#pragma once

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace lython::callable {

inline bool hasDefault(llvm::ArrayRef<char> defaults, std::size_t index) {
  return index < defaults.size() && defaults[index] != 0;
}

template <typename T>
bool metadataCompatible(llvm::ArrayRef<T> required,
                        llvm::ArrayRef<T> candidate) {
  return required.empty() || candidate.empty() || required == candidate;
}

inline bool optionalMetadataCompatible(llvm::StringRef required,
                                       llvm::StringRef candidate) {
  return required.empty() || candidate.empty() || required == candidate;
}

template <typename T>
bool optionalMetadataCompatible(const T &required, const T &candidate) {
  return !static_cast<bool>(required) || !static_cast<bool>(candidate) ||
         required == candidate;
}

template <typename T, typename IsDefault>
bool defaultMetadataCompatible(llvm::ArrayRef<T> required,
                               llvm::ArrayRef<T> candidate,
                               IsDefault isDefault) {
  if (required.empty() || candidate.empty())
    return true;
  if (required.size() != candidate.size())
    return false;
  for (auto [requiredDefault, candidateDefault] :
       llvm::zip(required, candidate))
    if (isDefault(requiredDefault) && !isDefault(candidateDefault))
      return false;
  return true;
}

inline bool defaultMetadataCompatible(llvm::ArrayRef<char> required,
                                      llvm::ArrayRef<char> candidate) {
  return defaultMetadataCompatible(required, candidate,
                                   [](char value) { return value != 0; });
}

template <typename NameT, typename DefaultT, typename OptionalT,
          typename IsDefault>
bool parameterMetadataCompatible(
    unsigned requiredPositionalOnlyCount, unsigned candidatePositionalOnlyCount,
    llvm::ArrayRef<NameT> requiredPositionalNames,
    llvm::ArrayRef<NameT> candidatePositionalNames,
    llvm::ArrayRef<DefaultT> requiredPositionalDefaults,
    llvm::ArrayRef<DefaultT> candidatePositionalDefaults,
    llvm::ArrayRef<NameT> requiredKwonlyNames,
    llvm::ArrayRef<NameT> candidateKwonlyNames,
    llvm::ArrayRef<DefaultT> requiredKwonlyDefaults,
    llvm::ArrayRef<DefaultT> candidateKwonlyDefaults,
    const OptionalT &requiredVarargName, const OptionalT &candidateVarargName,
    const OptionalT &requiredKwargName, const OptionalT &candidateKwargName,
    IsDefault isDefault, bool comparePositionalOnlyCount = true) {
  if (comparePositionalOnlyCount &&
      candidatePositionalOnlyCount > requiredPositionalOnlyCount)
    return false;
  if (!metadataCompatible(requiredPositionalNames, candidatePositionalNames))
    return false;
  if (!defaultMetadataCompatible(requiredPositionalDefaults,
                                 candidatePositionalDefaults, isDefault))
    return false;
  if (!metadataCompatible(requiredKwonlyNames, candidateKwonlyNames))
    return false;
  if (!defaultMetadataCompatible(requiredKwonlyDefaults,
                                 candidateKwonlyDefaults, isDefault))
    return false;
  if (!optionalMetadataCompatible(requiredVarargName, candidateVarargName))
    return false;
  if (!optionalMetadataCompatible(requiredKwargName, candidateKwargName))
    return false;
  return true;
}

template <typename T> struct Signature {
  llvm::ArrayRef<T> positional;
  llvm::ArrayRef<T> kwonly;
  std::optional<T> vararg;
  std::optional<T> kwarg;
  llvm::ArrayRef<std::string> positionalNames;
  llvm::ArrayRef<char> positionalDefaults;
  llvm::ArrayRef<std::string> kwonlyNames;
  llvm::ArrayRef<char> kwonlyDefaults;
  unsigned positionalOnlyCount = 0;

  bool valid() const {
    if (positionalOnlyCount > positional.size())
      return false;
    if (!positionalNames.empty() && positionalNames.size() != positional.size())
      return false;
    if (!positionalDefaults.empty() &&
        positionalDefaults.size() != positional.size())
      return false;
    if (!kwonlyNames.empty() && kwonlyNames.size() != kwonly.size())
      return false;
    if (!kwonlyDefaults.empty() && kwonlyDefaults.size() != kwonly.size())
      return false;
    return true;
  }
};

template <typename T>
bool signatureMetadataCompatible(Signature<T> required, Signature<T> candidate,
                                 bool comparePositionalOnlyCount = true) {
  if (!required.valid() || !candidate.valid())
    return false;
  return parameterMetadataCompatible(
      required.positionalOnlyCount, candidate.positionalOnlyCount,
      required.positionalNames, candidate.positionalNames,
      required.positionalDefaults, candidate.positionalDefaults,
      required.kwonlyNames, candidate.kwonlyNames, required.kwonlyDefaults,
      candidate.kwonlyDefaults, llvm::StringRef{}, llvm::StringRef{},
      llvm::StringRef{}, llvm::StringRef{},
      [](char value) { return value != 0; }, comparePositionalOnlyCount);
}

template <typename T, typename Keyword> struct Invocation {
  llvm::SmallVector<T, 8> positional;
  llvm::SmallVector<Keyword, 8> keywords;
};

struct NoopMatchObserver {
  void onDirectPositional(std::size_t) {}
  void onVariadicPositionalParameter() {}
  void onVariadicPositional(std::size_t) {}
  void onNamedPositional(llvm::StringRef) {}
  void onKeywordOnly(llvm::StringRef) {}
  void onKeywordVariadicParameter() {}
  void onKeywordVariadic(llvm::StringRef) {}
  void onUnpackedKeywordVariadic() {}
  void onDefaultedPositional(std::size_t) { onDefaultedParameter(); }
  void onDefaultedKeywordOnly(std::size_t) { onDefaultedParameter(); }
  void onDefaultedParameter() {}
};

template <typename OnDefaulted>
struct DefaultedOnlyMatchObserver : NoopMatchObserver {
  OnDefaulted onDefaulted;

  explicit DefaultedOnlyMatchObserver(OnDefaulted onDefaulted)
      : onDefaulted(std::move(onDefaulted)) {}

  void onDefaultedParameter() { onDefaulted(); }
  void onDefaultedPositional(std::size_t) { onDefaulted(); }
  void onDefaultedKeywordOnly(std::size_t) { onDefaulted(); }
};

struct InvocationSpecificityScore : NoopMatchObserver {
  int score = 0;

  void onDirectPositional(std::size_t) { score += 4; }
  void onNamedPositional(llvm::StringRef) { score += 3; }
  void onKeywordOnly(llvm::StringRef) { score += 4; }
  void onVariadicPositionalParameter() { score -= 2; }
  void onVariadicPositional(std::size_t) { score -= 5; }
  void onKeywordVariadicParameter() { score -= 2; }
  void onKeywordVariadic(llvm::StringRef) { score -= 5; }
  void onUnpackedKeywordVariadic() { score -= 8; }
  void onDefaultedParameter() { score -= 3; }
  void onDefaultedPositional(std::size_t) { onDefaultedParameter(); }
  void onDefaultedKeywordOnly(std::size_t) { onDefaultedParameter(); }
};

template <typename KindT, typename TypeT, typename ValueT> struct Slot {
  using Kind = KindT;
  using Type = TypeT;
  using Value = ValueT;

  Kind kind;
  unsigned index = 0;
  Type type;
  Value value;
  std::string keywordName;
  Value keywordNameValue;
};

template <typename SlotT>
SlotT makeFormalSlot(typename SlotT::Kind kind, unsigned index,
                     typename SlotT::Type type) {
  SlotT slot;
  slot.kind = kind;
  slot.index = index;
  slot.type = type;
  return slot;
}

template <typename SlotT>
llvm::SmallVector<SlotT, 8>
makeFormalSlots(llvm::ArrayRef<typename SlotT::Type> types,
                typename SlotT::Kind kind) {
  llvm::SmallVector<SlotT, 8> slots;
  slots.reserve(types.size());
  for (auto [index, type] : llvm::enumerate(types))
    slots.push_back(
        makeFormalSlot<SlotT>(kind, static_cast<unsigned>(index), type));
  return slots;
}

template <typename SlotT>
SlotT makeActualSlot(typename SlotT::Kind kind, typename SlotT::Value value,
                     typename SlotT::Type type) {
  SlotT slot;
  slot.kind = kind;
  slot.type = type;
  slot.value = value;
  return slot;
}

template <typename SlotT> struct Keyword {
  std::string name;
  typename SlotT::Value nameValue;
  SlotT value;
};

template <typename T> struct VarargShape {
  enum class Kind { Invalid, Repeated, Exact };

  Kind kind = Kind::Invalid;
  std::optional<T> repeated;
  llvm::SmallVector<T, 8> exact;

  static VarargShape invalid() { return {}; }

  static VarargShape repeatedOf(T element) {
    VarargShape shape;
    shape.kind = Kind::Repeated;
    shape.repeated = std::move(element);
    return shape;
  }

  static VarargShape exactOf(llvm::ArrayRef<T> elements) {
    VarargShape shape;
    shape.kind = Kind::Exact;
    shape.exact.append(elements.begin(), elements.end());
    return shape;
  }

  bool valid() const { return kind != Kind::Invalid; }
};

template <typename T, typename MatchT, typename MatchElement, typename Merge>
MatchT matchVarargContainment(VarargShape<T> actualShape,
                              VarargShape<T> expectedShape, MatchT compatible,
                              MatchT mismatch, MatchElement matchElement,
                              Merge merge) {
  using Shape = VarargShape<T>;
  if (!actualShape.valid() || !expectedShape.valid())
    return mismatch;

  if (expectedShape.kind == Shape::Kind::Repeated) {
    if (actualShape.kind != Shape::Kind::Repeated)
      return mismatch;
    return matchElement(*expectedShape.repeated, *actualShape.repeated);
  }

  MatchT result = compatible;
  if (actualShape.kind == Shape::Kind::Repeated) {
    for (const T &expectedElement : expectedShape.exact) {
      result =
          merge(result, matchElement(expectedElement, *actualShape.repeated));
      if (result == mismatch)
        return result;
    }
    return result;
  }

  if (actualShape.exact.size() != expectedShape.exact.size())
    return mismatch;
  for (auto [actualElement, expectedElement] :
       llvm::zip(actualShape.exact, expectedShape.exact)) {
    result = merge(result, matchElement(expectedElement, actualElement));
    if (result == mismatch)
      return result;
  }
  return result;
}

template <typename T>
llvm::StringRef nameAt(llvm::ArrayRef<T> names, std::size_t index) {
  if (index >= names.size())
    return {};
  return names[index];
}

template <typename T, typename Keyword, typename MakeKeyword>
bool appendAcceptanceSamples(
    Signature<T> signature,
    llvm::SmallVectorImpl<Invocation<T, Keyword>> &samples,
    MakeKeyword makeKeyword) {
  if (!signature.valid())
    return false;

  auto appendSample = [&](llvm::ArrayRef<T> positional,
                          llvm::ArrayRef<Keyword> keywords) {
    Invocation<T, Keyword> invocation;
    invocation.positional.append(positional.begin(), positional.end());
    invocation.keywords.append(keywords.begin(), keywords.end());
    samples.push_back(std::move(invocation));
  };

  llvm::SmallVector<Keyword, 8> allKwonly;
  llvm::SmallVector<Keyword, 8> requiredKwonly;
  for (auto [index, expectedArg] : llvm::enumerate(signature.kwonly)) {
    llvm::StringRef name = nameAt(signature.kwonlyNames, index);
    if (name.empty())
      return false;
    Keyword keyword = makeKeyword(name, expectedArg);
    allKwonly.push_back(keyword);
    if (!hasDefault(signature.kwonlyDefaults, index))
      requiredKwonly.push_back(keyword);
  }

  appendSample(signature.positional, allKwonly);

  std::size_t requiredPositional = signature.positional.size();
  while (requiredPositional > 0 &&
         hasDefault(signature.positionalDefaults, requiredPositional - 1))
    --requiredPositional;
  appendSample(signature.positional.take_front(requiredPositional),
               requiredKwonly);

  bool hasKeywordCallablePositional = false;
  llvm::SmallVector<T, 8> positionalPrefix;
  llvm::SmallVector<T, 8> requiredPositionalPrefix;
  llvm::SmallVector<Keyword, 8> positionalAsKeywords;
  llvm::SmallVector<Keyword, 8> requiredPositionalAsKeywords;
  for (auto [index, expectedArg] : llvm::enumerate(signature.positional)) {
    if (index < signature.positionalOnlyCount) {
      positionalPrefix.push_back(expectedArg);
      if (index < requiredPositional)
        requiredPositionalPrefix.push_back(expectedArg);
      continue;
    }

    llvm::StringRef name = nameAt(signature.positionalNames, index);
    if (name.empty())
      continue;
    hasKeywordCallablePositional = true;
    Keyword keyword = makeKeyword(name, expectedArg);
    positionalAsKeywords.push_back(keyword);
    if (index < requiredPositional)
      requiredPositionalAsKeywords.push_back(keyword);
  }

  if (hasKeywordCallablePositional || !signature.kwonly.empty()) {
    llvm::SmallVector<Keyword, 16> fullKeywords;
    fullKeywords.append(positionalAsKeywords.begin(),
                        positionalAsKeywords.end());
    fullKeywords.append(allKwonly.begin(), allKwonly.end());
    appendSample(positionalPrefix, fullKeywords);

    llvm::SmallVector<Keyword, 16> minimalKeywords;
    minimalKeywords.append(requiredPositionalAsKeywords.begin(),
                           requiredPositionalAsKeywords.end());
    minimalKeywords.append(requiredKwonly.begin(), requiredKwonly.end());
    appendSample(requiredPositionalPrefix, minimalKeywords);
  }

  return true;
}

template <typename T, typename Keyword, typename MatchExpected,
          typename VarargShapeFn, typename KwargValue,
          typename UnpackedKeywordValue, typename KeywordName,
          typename KeywordValue, typename MatchObserver>
bool matchInvocationWithObserver(
    Signature<T> signature, llvm::ArrayRef<T> positionalArgs,
    llvm::ArrayRef<Keyword> keywords, MatchExpected matchExpected,
    VarargShapeFn varargShape, KwargValue kwargValue,
    UnpackedKeywordValue unpackedKeywordValue, KeywordName keywordName,
    KeywordValue keywordValue, MatchObserver &observer,
    bool requireSatisfiedDefaults = true) {
  if (!signature.valid())
    return false;

  std::vector<char> positionalProvided(signature.positional.size(), 0);
  std::vector<char> kwonlyProvided(signature.kwonly.size(), 0);

  const std::size_t directCount =
      std::min(positionalArgs.size(), signature.positional.size());
  for (std::size_t index = 0; index < directCount; ++index) {
    positionalProvided[index] = 1;
    if (!matchExpected(signature.positional[index], positionalArgs[index]))
      return false;
    observer.onDirectPositional(index);
  }

  std::size_t suppliedVarargs = 0;
  if (positionalArgs.size() > signature.positional.size())
    suppliedVarargs = positionalArgs.size() - signature.positional.size();
  if (signature.vararg) {
    VarargShape<T> shape = varargShape(*signature.vararg);
    if (!shape.valid())
      return false;
    if (shape.kind == VarargShape<T>::Kind::Repeated)
      observer.onVariadicPositionalParameter();
    if (shape.kind == VarargShape<T>::Kind::Exact &&
        suppliedVarargs != shape.exact.size())
      return false;
    if (suppliedVarargs) {
      llvm::ArrayRef<T> extra =
          positionalArgs.drop_front(signature.positional.size());
      for (auto [index, actual] : llvm::enumerate(extra)) {
        const T *expected = nullptr;
        if (shape.kind == VarargShape<T>::Kind::Repeated) {
          expected = &*shape.repeated;
        } else if (index < shape.exact.size()) {
          expected = &shape.exact[index];
        }
        if (!expected || !matchExpected(*expected, actual))
          return false;
        if (shape.kind == VarargShape<T>::Kind::Repeated)
          observer.onVariadicPositional(signature.positional.size() + index);
      }
    }
  } else if (suppliedVarargs) {
    return false;
  }

  llvm::StringSet<> seenKeywords;
  llvm::StringMap<unsigned> positionalNameToIndex;
  for (auto [index, name] : llvm::enumerate(signature.positionalNames)) {
    if (index < signature.positional.size() &&
        index >= signature.positionalOnlyCount && !name.empty())
      positionalNameToIndex[name] = static_cast<unsigned>(index);
  }

  llvm::StringMap<unsigned> kwonlyNameToIndex;
  for (auto [index, name] : llvm::enumerate(signature.kwonlyNames)) {
    if (index < signature.kwonly.size() && !name.empty())
      kwonlyNameToIndex[name] = static_cast<unsigned>(index);
  }

  if (signature.kwarg)
    observer.onKeywordVariadicParameter();

  for (const Keyword &keyword : keywords) {
    llvm::StringRef name = keywordName(keyword);
    auto actual = keywordValue(keyword);
    if (name.empty()) {
      if (!signature.kwarg)
        return false;
      std::optional<T> expected = kwargValue(*signature.kwarg);
      std::optional<T> unpacked = unpackedKeywordValue(actual);
      if (!expected || !unpacked || !matchExpected(*expected, *unpacked))
        return false;
      observer.onUnpackedKeywordVariadic();
      continue;
    }
    if (!seenKeywords.insert(name).second)
      return false;

    auto positional = positionalNameToIndex.find(name);
    if (positional != positionalNameToIndex.end()) {
      unsigned index = positional->second;
      if (index >= signature.positional.size() || positionalProvided[index])
        return false;
      positionalProvided[index] = 1;
      if (!matchExpected(signature.positional[index], actual))
        return false;
      observer.onNamedPositional(name);
      continue;
    }

    auto kwonly = kwonlyNameToIndex.find(name);
    if (kwonly != kwonlyNameToIndex.end()) {
      unsigned index = kwonly->second;
      if (index >= signature.kwonly.size() || kwonlyProvided[index])
        return false;
      kwonlyProvided[index] = 1;
      if (!matchExpected(signature.kwonly[index], actual))
        return false;
      observer.onKeywordOnly(name);
      continue;
    }

    if (!signature.kwarg)
      return false;
    std::optional<T> value = kwargValue(*signature.kwarg);
    if (!value || !matchExpected(*value, actual))
      return false;
    observer.onKeywordVariadic(name);
  }

  if (!requireSatisfiedDefaults)
    return true;

  for (std::size_t index = 0; index < signature.positional.size(); ++index) {
    if (positionalProvided[index])
      continue;
    if (!hasDefault(signature.positionalDefaults, index))
      return false;
    observer.onDefaultedPositional(index);
  }

  for (std::size_t index = 0; index < signature.kwonly.size(); ++index) {
    if (kwonlyProvided[index])
      continue;
    if (!hasDefault(signature.kwonlyDefaults, index))
      return false;
    observer.onDefaultedKeywordOnly(index);
  }

  return true;
}

template <typename T, typename Keyword, typename MatchExpected,
          typename VarargShapeFn, typename KwargValue,
          typename UnpackedKeywordValue, typename KeywordName,
          typename KeywordValue, typename OnDefaulted>
bool matchInvocation(Signature<T> signature, llvm::ArrayRef<T> positionalArgs,
                     llvm::ArrayRef<Keyword> keywords,
                     MatchExpected matchExpected, VarargShapeFn varargShape,
                     KwargValue kwargValue,
                     UnpackedKeywordValue unpackedKeywordValue,
                     KeywordName keywordName, KeywordValue keywordValue,
                     OnDefaulted onDefaulted,
                     bool requireSatisfiedDefaults = true) {
  DefaultedOnlyMatchObserver<OnDefaulted> observer(std::move(onDefaulted));
  return matchInvocationWithObserver(
      signature, positionalArgs, keywords, std::move(matchExpected),
      std::move(varargShape), std::move(kwargValue),
      std::move(unpackedKeywordValue), std::move(keywordName),
      std::move(keywordValue), observer, requireSatisfiedDefaults);
}

} // namespace lython::callable
