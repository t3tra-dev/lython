#pragma once

#include "CandidateSelection.h"
#include "StructuralShape.h"

#include "llvm/ADT/ArrayRef.h"

#include <algorithm>
#include <set>
#include <utility>
#include <vector>

namespace lython::matching {

template <typename State, typename Match, typename MakeContextFn, typename Fn,
          typename ValidateFn>
Match transaction(State &state, Match mismatch, MakeContextFn makeContext,
                  Fn &&fn, ValidateFn validate) {
  return lython::structural::transaction(
      state, mismatch, std::move(makeContext), std::forward<Fn>(fn),
      std::move(validate));
}

template <typename State, typename Match, typename MakeContextFn, typename Fn>
Match transaction(State &state, Match mismatch, MakeContextFn makeContext,
                  Fn &&fn) {
  return lython::matching::transaction(state, mismatch, std::move(makeContext),
                                       std::forward<Fn>(fn),
                                       [](const State &) { return true; });
}

template <typename State, typename Range, typename Match,
          typename MakeContextFn, typename MatchCandidateFn,
          typename ScoreCandidateFn, typename EquivalentStateFn,
          typename ValidateStateFn>
Match selectBestTransactional(State &state, const Range &candidates,
                              Match mismatch, MakeContextFn makeContext,
                              MatchCandidateFn matchCandidate,
                              ScoreCandidateFn scoreCandidate,
                              EquivalentStateFn equivalentState,
                              ValidateStateFn validateState) {
  auto selected = lython::selection::selectBestTransactionalState(
      state, candidates, mismatch, std::move(makeContext),
      std::move(matchCandidate), std::move(scoreCandidate),
      std::move(equivalentState), std::move(validateState));
  if (!selected)
    return mismatch;
  state = std::move(selected->state);
  return selected->match;
}

template <typename State, typename Range, typename Match,
          typename MakeContextFn, typename MatchCandidateFn,
          typename ScoreCandidateFn, typename EquivalentStateFn>
Match selectBestTransactional(State &state, const Range &candidates,
                              Match mismatch, MakeContextFn makeContext,
                              MatchCandidateFn matchCandidate,
                              ScoreCandidateFn scoreCandidate,
                              EquivalentStateFn equivalentState) {
  return lython::matching::selectBestTransactional(
      state, candidates, mismatch, std::move(makeContext),
      std::move(matchCandidate), std::move(scoreCandidate),
      std::move(equivalentState), [](const State &) { return true; });
}

template <typename State, typename Match, typename MakeContextFn,
          typename Operand, typename MatchOperandFn, typename MergeFn,
          typename ValidateFn>
Match matchOperandsInTransaction(State &state, Match compatible, Match mismatch,
                                 MakeContextFn makeContext,
                                 llvm::ArrayRef<Operand> operands,
                                 MatchOperandFn matchOperand, MergeFn merge,
                                 ValidateFn validate) {
  return lython::matching::transaction(
      state, mismatch, std::move(makeContext),
      [&](auto &context) {
        return lython::structural::matchOperands<Match>(
            operands, compatible, mismatch,
            [&](const Operand &operand) {
              return matchOperand(context, operand);
            },
            merge);
      },
      std::move(validate));
}

template <typename State, typename Match, typename MakeContextFn,
          typename Operand, typename MatchOperandFn, typename MergeFn>
Match matchOperandsInTransaction(State &state, Match compatible, Match mismatch,
                                 MakeContextFn makeContext,
                                 llvm::ArrayRef<Operand> operands,
                                 MatchOperandFn matchOperand, MergeFn merge) {
  return lython::matching::matchOperandsInTransaction(
      state, compatible, mismatch, std::move(makeContext), operands,
      std::move(matchOperand), std::move(merge),
      [](const State &) { return true; });
}

template <typename Map, typename StateRange, typename GetMapFn,
          typename NormalizeValueFn, typename MakeUnionFn>
Map mergeChangedUnionMap(const Map &base, const StateRange &states,
                         GetMapFn getMap, NormalizeValueFn normalizeValue,
                         MakeUnionFn makeUnion) {
  using Key = typename Map::key_type;
  using Value = typename Map::mapped_type;

  std::set<Key> changedKeys;
  for (const auto &state : states) {
    const Map &bindings = getMap(state);
    for (const auto &[key, value] : bindings) {
      auto baseValue = base.find(key);
      if (baseValue == base.end() || baseValue->second != value)
        changedKeys.insert(key);
    }
  }

  Map merged = base;
  for (const Key &key : changedKeys) {
    std::vector<Value> alternatives;
    for (const auto &state : states) {
      const Map &bindings = getMap(state);
      auto found = bindings.find(key);
      if (found == bindings.end())
        continue;
      auto baseValue = base.find(key);
      if (baseValue != base.end() && baseValue->second == found->second)
        continue;
      Value normalized = normalizeValue(state, key, found->second);
      if (std::find(alternatives.begin(), alternatives.end(), normalized) ==
          alternatives.end())
        alternatives.push_back(std::move(normalized));
    }
    if (!alternatives.empty())
      merged[key] = makeUnion(llvm::ArrayRef<Value>(alternatives));
  }
  return merged;
}

} // namespace lython::matching
