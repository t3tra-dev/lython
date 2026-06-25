#pragma once

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <utility>

namespace lython::structural {

enum class MatchResult { Mismatch, Compatible, Bound };

inline MatchResult mergeMatch(MatchResult lhs, MatchResult rhs) {
  if (lhs == MatchResult::Mismatch || rhs == MatchResult::Mismatch)
    return MatchResult::Mismatch;
  if (lhs == MatchResult::Bound || rhs == MatchResult::Bound)
    return MatchResult::Bound;
  return MatchResult::Compatible;
}

enum class ShapeState { Absent, Mismatch, Matched };

template <typename Value, typename Variance = llvm::StringRef> struct Operand {
  using ValueType = Value;
  using VarianceType = Variance;

  Value expected;
  Value actual;
  Variance variance{"covariant"};
  bool useVariance = false;
};

template <typename Operand, unsigned InlineOperandCount = 4> struct Shape {
  using State = ShapeState;
  using OperandType = Operand;

  State state = State::Absent;
  int specificity = 0;
  bool includeOperandSpecificity = true;
  llvm::SmallVector<Operand, InlineOperandCount> operands;

  static Shape absent() { return {}; }

  static Shape mismatch(int specificity = 0,
                        bool includeOperandSpecificity = true) {
    Shape shape;
    shape.state = State::Mismatch;
    shape.specificity = specificity;
    shape.includeOperandSpecificity = includeOperandSpecificity;
    return shape;
  }

  static Shape matched(int specificity = 0,
                       bool includeOperandSpecificity = true) {
    Shape shape;
    shape.state = State::Matched;
    shape.specificity = specificity;
    shape.includeOperandSpecificity = includeOperandSpecificity;
    return shape;
  }
};

template <typename ShapeT, typename ExpectedArgs, typename ActualArgs,
          typename MakeOperandFn>
ShapeT decomposeSameHead(bool headMatches, const ExpectedArgs &expectedArgs,
                         const ActualArgs &actualArgs, int specificity,
                         bool matchedIncludeOperandSpecificity,
                         bool mismatchIncludeOperandSpecificity,
                         MakeOperandFn makeOperand) {
  if (!headMatches || expectedArgs.size() != actualArgs.size())
    return ShapeT::mismatch(specificity, mismatchIncludeOperandSpecificity);

  ShapeT shape = ShapeT::matched(specificity, matchedIncludeOperandSpecificity);
  for (auto indexed : llvm::enumerate(llvm::zip(expectedArgs, actualArgs))) {
    auto [expected, actual] = indexed.value();
    shape.operands.push_back(makeOperand(indexed.index(), expected, actual));
  }
  return shape;
}

template <typename ShapeT>
ShapeT unary(bool headMatches, typename ShapeT::OperandType::ValueType expected,
             typename ShapeT::OperandType::ValueType actual, int specificity,
             bool includeOperandSpecificity = true) {
  if (!headMatches)
    return ShapeT::mismatch(specificity, includeOperandSpecificity);
  ShapeT shape = ShapeT::matched(specificity, includeOperandSpecificity);
  shape.operands.push_back({expected, actual});
  return shape;
}

template <typename ShapeT>
ShapeT binary(bool headMatches,
              typename ShapeT::OperandType::ValueType firstExpected,
              typename ShapeT::OperandType::ValueType firstActual,
              typename ShapeT::OperandType::ValueType secondExpected,
              typename ShapeT::OperandType::ValueType secondActual,
              int specificity, bool includeOperandSpecificity = true) {
  if (!headMatches)
    return ShapeT::mismatch(specificity, includeOperandSpecificity);
  ShapeT shape = ShapeT::matched(specificity, includeOperandSpecificity);
  shape.operands.push_back({firstExpected, firstActual});
  shape.operands.push_back({secondExpected, secondActual});
  return shape;
}

template <typename Match, typename Operand, typename MatchOperandFn,
          typename MergeFn>
Match matchOperands(llvm::ArrayRef<Operand> operands, Match compatible,
                    Match mismatch, MatchOperandFn matchOperand,
                    MergeFn merge) {
  Match result = compatible;
  for (const Operand &operand : operands) {
    result = merge(result, matchOperand(operand));
    if (result == mismatch)
      return result;
  }
  return result;
}

template <typename State, typename Match, typename MakeContextFn, typename Fn,
          typename ValidateFn>
Match transaction(State &state, Match mismatch, MakeContextFn makeContext,
                  Fn fn, ValidateFn validate) {
  State candidate = state;
  auto context = makeContext(candidate);
  Match match = fn(context);
  if (match == mismatch || !validate(candidate))
    return mismatch;
  state = std::move(candidate);
  return match;
}

template <typename State, typename Match, typename MakeContextFn, typename Fn>
Match transaction(State &state, Match mismatch, MakeContextFn makeContext,
                  Fn fn) {
  return transaction(state, mismatch, std::move(makeContext), std::move(fn),
                     [](const State &) { return true; });
}

template <typename ShapeT, typename ScoreOperandFn>
int specificity(const ShapeT &shape, ScoreOperandFn scoreOperand) {
  int score = shape.specificity;
  if (shape.state == ShapeT::State::Matched &&
      shape.includeOperandSpecificity) {
    for (const auto &operand : shape.operands)
      score += scoreOperand(operand);
  }
  return score;
}

} // namespace lython::structural
