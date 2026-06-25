#pragma once

#include <limits>
#include <optional>
#include <type_traits>
#include <utility>

namespace lython::selection {

template <typename State, typename Match> struct TransactionalState {
  State state;
  Match match;
};

template <typename T, typename ScoreFn, typename EquivalentFn>
class BestCandidate {
public:
  BestCandidate(ScoreFn score, EquivalentFn equivalent)
      : score(std::move(score)), equivalent(std::move(equivalent)) {}

  void consider(T candidate) {
    int candidateScore = score(candidate);
    if (!best || candidateScore > bestScore) {
      best = std::move(candidate);
      bestScore = candidateScore;
      ambiguous = false;
      return;
    }
    if (candidateScore == bestScore && !equivalent(*best, candidate))
      ambiguous = true;
  }

  std::optional<T> finish() && {
    if (ambiguous)
      return std::nullopt;
    return std::move(best);
  }

private:
  std::optional<T> best;
  int bestScore = std::numeric_limits<int>::min();
  bool ambiguous = false;
  ScoreFn score;
  EquivalentFn equivalent;
};

template <typename T, typename ScoreFn, typename EquivalentFn>
BestCandidate<T, ScoreFn, EquivalentFn> bestCandidate(ScoreFn score,
                                                      EquivalentFn equivalent) {
  return BestCandidate<T, ScoreFn, EquivalentFn>(std::move(score),
                                                 std::move(equivalent));
}

template <typename State, typename Range, typename TryCandidateFn,
          typename ScoreCandidateFn, typename EquivalentStateFn>
std::optional<State> selectBestState(const Range &candidates,
                                     TryCandidateFn tryCandidate,
                                     ScoreCandidateFn scoreCandidate,
                                     EquivalentStateFn equivalentState) {
  struct ScoredState {
    State state;
    int score = 0;
  };

  auto selection = bestCandidate<ScoredState>(
      [](const ScoredState &candidate) { return candidate.score; },
      [&](const ScoredState &lhs, const ScoredState &rhs) {
        return equivalentState(lhs.state, rhs.state);
      });

  for (const auto &candidate : candidates) {
    std::optional<State> state = tryCandidate(candidate);
    if (!state)
      continue;
    selection.consider({std::move(*state), scoreCandidate(candidate)});
  }

  std::optional<ScoredState> selected = std::move(selection).finish();
  if (!selected)
    return std::nullopt;
  return std::move(selected->state);
}

template <typename State, typename Range, typename Match,
          typename MakeContextFn, typename MatchCandidateFn,
          typename ScoreCandidateFn, typename EquivalentStateFn,
          typename ValidateStateFn>
std::optional<TransactionalState<State, Match>> selectBestTransactionalState(
    const State &baseState, const Range &candidates, Match mismatch,
    MakeContextFn makeContext, MatchCandidateFn matchCandidate,
    ScoreCandidateFn scoreCandidate, EquivalentStateFn equivalentState,
    ValidateStateFn validateState) {
  using Result = TransactionalState<State, Match>;
  return selectBestState<Result>(
      candidates,
      [&](const auto &candidate) -> std::optional<Result> {
        State candidateState = baseState;
        auto context = makeContext(candidateState);
        Match match = matchCandidate(context, candidate);
        if (match == mismatch || !validateState(candidateState))
          return std::nullopt;
        return Result{std::move(candidateState), match};
      },
      scoreCandidate,
      [&](const Result &lhs, const Result &rhs) {
        return equivalentState(lhs.state, rhs.state);
      });
}

template <typename State, typename Range, typename Match,
          typename MakeContextFn, typename MatchCandidateFn,
          typename ScoreCandidateFn, typename EquivalentStateFn>
std::optional<TransactionalState<State, Match>> selectBestTransactionalState(
    const State &baseState, const Range &candidates, Match mismatch,
    MakeContextFn makeContext, MatchCandidateFn matchCandidate,
    ScoreCandidateFn scoreCandidate, EquivalentStateFn equivalentState) {
  return selectBestTransactionalState(
      baseState, candidates, mismatch, std::move(makeContext),
      std::move(matchCandidate), std::move(scoreCandidate),
      std::move(equivalentState), [](const State &) { return true; });
}

template <typename Range, typename CaptureStateFn, typename RestoreStateFn,
          typename TryCandidateFn, typename ScoreCandidateFn,
          typename EquivalentStateFn>
std::optional<std::decay_t<std::invoke_result_t<CaptureStateFn>>>
selectBestRestoredState(const Range &candidates, CaptureStateFn captureState,
                        RestoreStateFn restoreState,
                        TryCandidateFn tryCandidate,
                        ScoreCandidateFn scoreCandidate,
                        EquivalentStateFn equivalentState) {
  using State = std::decay_t<std::invoke_result_t<CaptureStateFn>>;
  State baseState = captureState();
  return selectBestState<State>(
      candidates,
      [&](const auto &candidate) -> std::optional<State> {
        restoreState(baseState);
        if (!tryCandidate(candidate)) {
          restoreState(baseState);
          return std::nullopt;
        }
        State selectedState = captureState();
        restoreState(baseState);
        return selectedState;
      },
      std::move(scoreCandidate), std::move(equivalentState));
}

} // namespace lython::selection
