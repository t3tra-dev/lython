#include "EmitterCore.h"
#include "EmitterPyOps.h"
#include "EmitterSupport.h"

#include "AstAccess.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"

namespace lython::emitter {

void ModuleEmitter::emitIf(const parser::Node &statement) {
  const parser::Node *test = ast::node(statement, "test");
  std::optional<BranchTypeNarrowing> narrowing =
      test ? optionalBranchTypeNarrowing(*test, types, module) : std::nullopt;
  auto applyNarrowing = [&](const BranchTypeNarrowing &fact,
                            bool conditionIsTrue) {
    mlir::Type narrowed = conditionIsTrue ? fact.trueType : fact.falseType;
    mlir::Type sourceType =
        conditionIsTrue ? fact.trueSourceType : fact.falseSourceType;
    if (!narrowed)
      return;
    auto found = values.find(fact.name);
    if (found != values.end()) {
      if (mlir::isa<py::UnionType>(found->second.value.getType()) &&
          found->second.value.getType() != narrowed) {
        auto unionType =
            mlir::cast<py::UnionType>(found->second.value.getType());
        if (unionType.hasMember(narrowed)) {
          auto unwrap = py::UnionUnwrapOp::create(
              builder, loc(statement), narrowed, found->second.value);
          found->second.value = unwrap.getResult();
        } else if (sourceType && unionType.hasMember(sourceType)) {
          auto unwrap = py::UnionUnwrapOp::create(
              builder, loc(statement), sourceType, found->second.value);
          found->second.value = unwrap.getResult();
          if (sourceType != narrowed &&
              mlir::isa<py::ContractType>(sourceType) &&
              mlir::isa<py::ContractType>(narrowed) &&
              py::isAssignableTo(narrowed, sourceType, module)) {
            auto refine = py::ClassRefineOp::create(
                builder, loc(statement), narrowed, found->second.value);
            found->second.value = refine.getResult();
          }
        }
      } else if (found->second.value.getType() != narrowed &&
                 mlir::isa<py::ContractType>(found->second.value.getType()) &&
                 mlir::isa<py::ContractType>(narrowed) &&
                 py::isAssignableTo(narrowed, found->second.value.getType(),
                                    module)) {
        auto refine = py::ClassRefineOp::create(builder, loc(statement),
                                                narrowed, found->second.value);
        found->second.value = refine.getResult();
      }
      if (found->second.value.getType() == narrowed)
        found->second.type = narrowed;
    }
    if (found == values.end() || found->second.value.getType() == narrowed)
      types.bindSymbol(fact.name, narrowed);
  };

  std::optional<bool> staticTruth =
      test ? optionalStaticBranchTruth(*test, types, module) : std::nullopt;
  if (staticTruth) {
    // A statically taken branch is inline code, not a scope (CPython `if`
    // does not scope): its bindings — assignments, imports, narrowing —
    // persist after the fold, so the platform-switch idiom can bind names
    // (`if os.name == "posix": from posix import *`).
    if (narrowing)
      applyNarrowing(*narrowing, *staticTruth);
    const auto *selected = *staticTruth ? ast::nodeList(statement, "body")
                                        : ast::nodeList(statement, "orelse");
    if (selected && !selected->empty())
      emitStatements(selected);
    return;
  }

  mlir::Value condition = emitBoolValue(emitExpr(test), statement);
  const auto *orelse = ast::nodeList(statement, "orelse");
  bool hasElse = orelse && !orelse->empty();

  // Merge candidates: names freshly assigned (not pre-existing) in BOTH
  // branches. Threading them as continuation block arguments lets their value
  // escape the per-branch scopes (otherwise `if c: y=1 else: y=2` leaves `y`
  // unresolved after the if). Pre-existing reassigned locals thread separately
  // below (mutation rebinds or replacement merges).
  llvm::SmallVector<std::string, 4> mergeCandidates;
  if (hasElse) {
    llvm::StringSet<> assignedThen, assignedElse;
    collectAssignedNames(ast::nodeList(statement, "body"), assignedThen);
    collectAssignedNames(orelse, assignedElse);
    for (const auto &entry : assignedThen)
      if (assignedElse.contains(entry.getKey()) &&
          values.find(entry.getKey()) == values.end())
        mergeCandidates.push_back(entry.getKey().str());
    llvm::sort(mergeCandidates);
  }

  // Pre-existing locals rebound by structural-mutation calls in a branch
  // (e.g. `if c: xs.append(v)`): the (possibly reallocated) representation
  // must escape the branch, so thread them through continuation arguments.
  // Edges that do not mutate forward the outer value unchanged — the token is
  // forwarded on identity edges and consumed by the call on mutation edges,
  // so no replacement release is emitted anywhere. Branch values that turn
  // out NOT to derive from the outer value via mutation fall back to the old
  // scoped behavior.
  llvm::SmallVector<std::string, 2> mutationCandidates;
  llvm::SmallVector<Value, 2> mutationOuterValues;
  {
    llvm::StringSet<> assignedAnywhere;
    collectAssignedNames(ast::nodeList(statement, "body"), assignedAnywhere);
    if (hasElse)
      collectAssignedNames(orelse, assignedAnywhere);
    llvm::SmallVector<std::string, 2> names;
    for (const auto &entry : assignedAnywhere)
      if (values.find(entry.getKey()) != values.end())
        names.push_back(entry.getKey().str());
    llvm::sort(names);
    for (const std::string &name : names) {
      mutationCandidates.push_back(name);
      mutationOuterValues.push_back(values.find(name)->second);
    }
  }

  mlir::Block *entry = builder.getInsertionBlock();
  mlir::Region *region = entry->getParent();
  mlir::Block *continuation = entry->splitBlock(builder.getInsertionPoint());
  mlir::Block *thenBlock =
      builder.createBlock(region, continuation->getIterator());
  mlir::Block *elseBlock =
      hasElse ? builder.createBlock(region, continuation->getIterator())
              : continuation;

  // Emit each branch body; if it reaches the join, record its EXIT block (the
  // current block after the body, which may differ from the branch entry when
  // the body has its own control flow) and capture the merge values in scope.
  mlir::Block *thenExit = nullptr, *elseExit = nullptr;
  llvm::SmallVector<Value, 4> thenValues, elseValues;
  llvm::SmallVector<Value, 2> thenMutationValues, elseMutationValues;
  builder.setInsertionPointToStart(thenBlock);
  {
    ScopedEmitterScope scope(values, types);
    if (narrowing)
      applyNarrowing(*narrowing, /*conditionIsTrue=*/true);
    emitStatements(ast::nodeList(statement, "body"));
    if (!insertionBlockTerminated(builder)) {
      thenExit = builder.getInsertionBlock();
      for (const std::string &name : mergeCandidates) {
        auto found = values.find(name);
        thenValues.push_back(found != values.end() ? found->second : Value{});
      }
      for (const std::string &name : mutationCandidates) {
        auto found = values.find(name);
        thenMutationValues.push_back(found != values.end() ? found->second
                                                           : Value{});
      }
    }
  }
  bool thenTerminates = thenExit == nullptr;

  bool elseTerminates = false;
  if (hasElse) {
    builder.setInsertionPointToStart(elseBlock);
    {
      ScopedEmitterScope scope(values, types);
      if (narrowing)
        applyNarrowing(*narrowing, /*conditionIsTrue=*/false);
      emitStatements(orelse);
      if (!insertionBlockTerminated(builder)) {
        elseExit = builder.getInsertionBlock();
        for (const std::string &name : mergeCandidates) {
          auto found = values.find(name);
          elseValues.push_back(found != values.end() ? found->second : Value{});
        }
        for (const std::string &name : mutationCandidates) {
          auto found = values.find(name);
          elseMutationValues.push_back(found != values.end() ? found->second
                                                             : Value{});
        }
      }
    }
    elseTerminates = elseExit == nullptr;
  }

  // Determine merged locals: candidates that produced a value on both reaching
  // branches. The merged type joins the widened branch types.
  llvm::SmallVector<unsigned, 4> mergedCandidateIndices;
  llvm::SmallVector<mlir::Type, 4> mergedTypes;
  if (hasElse && thenExit && elseExit) {
    for (auto [index, name] : llvm::enumerate(mergeCandidates)) {
      (void)name;
      if (!thenValues[index].value || !elseValues[index].value)
        continue;
      mlir::Type merged = types.join({types.widenLiteral(thenValues[index].type),
                                      types.widenLiteral(elseValues[index].type)});
      mergedCandidateIndices.push_back(static_cast<unsigned>(index));
      mergedTypes.push_back(merged);
      continuation->addArgument(merged, loc(statement));
    }
  }

  // Thread structural-mutation rebinds: on every reaching edge the value must
  // be the outer value itself or a mutation chain over it; otherwise the name
  // keeps the old scoped behavior.
  llvm::SmallVector<unsigned, 2> threadedMutationIndices;
  for (auto [index, name] : llvm::enumerate(mutationCandidates)) {
    // Capturing the structured binding `index` directly is a C++20 extension;
    // bind a plain local for the lambda.
    const std::size_t idx = index;
    const Value &outer = mutationOuterValues[index];
    auto edgeAcceptable = [&](llvm::ArrayRef<Value> branchExitValues,
                              mlir::Block *exitBlock) {
      if (!exitBlock)
        return true;
      if (idx >= branchExitValues.size() || !branchExitValues[idx].value)
        return false;
      mlir::Value incoming = branchExitValues[idx].value;
      return incoming == outer.value ||
             (branchExitValues[idx].type == outer.type &&
              derivesViaStructuralMutation(incoming, outer.value));
    };
    bool mutatedSomewhere =
        (thenExit && index < thenMutationValues.size() &&
         thenMutationValues[index].value &&
         thenMutationValues[index].value != outer.value) ||
        (hasElse && elseExit && index < elseMutationValues.size() &&
         elseMutationValues[index].value &&
         elseMutationValues[index].value != outer.value);
    if (!mutatedSomewhere)
      continue;
    if (!edgeAcceptable(thenMutationValues, thenExit) ||
        (hasElse && !edgeAcceptable(elseMutationValues, elseExit)))
      continue;
    threadedMutationIndices.push_back(static_cast<unsigned>(index));
    continuation->addArgument(outer.value.getType(), loc(statement));
  }

  // Pre-existing locals REASSIGNED (not mutation-derived) on a reaching branch
  // thread as replacement merges: every reaching edge forwards its (coerced)
  // branch value — the outer value on non-assigning edges — and the existing
  // mixed-edge machinery balances the tokens (owned new values transfer,
  // identity forwards get borrow retains; the loop back-edge decref-on-replace
  // releases the replaced token).
  llvm::SmallVector<unsigned, 2> replacementIndices;
  llvm::SmallVector<mlir::Type, 2> replacementTypes;
  for (auto [index, name] : llvm::enumerate(mutationCandidates)) {
    (void)name;
    if (llvm::is_contained(threadedMutationIndices,
                           static_cast<unsigned>(index)))
      continue;
    const Value &outer = mutationOuterValues[index];
    bool reassigned =
        (thenExit && index < thenMutationValues.size() &&
         thenMutationValues[index].value &&
         thenMutationValues[index].value != outer.value) ||
        (hasElse && elseExit && index < elseMutationValues.size() &&
         elseMutationValues[index].value &&
         elseMutationValues[index].value != outer.value);
    if (!reassigned)
      continue;
    llvm::SmallVector<mlir::Type, 3> parts;
    if (!hasElse)
      parts.push_back(types.widenLiteral(outer.type));
    bool valuesPresent = true;
    if (thenExit) {
      if (index < thenMutationValues.size() && thenMutationValues[index].value)
        parts.push_back(types.widenLiteral(thenMutationValues[index].type));
      else
        valuesPresent = false;
    }
    if (hasElse && elseExit) {
      if (index < elseMutationValues.size() && elseMutationValues[index].value)
        parts.push_back(types.widenLiteral(elseMutationValues[index].type));
      else
        valuesPresent = false;
    }
    if (!valuesPresent || parts.empty())
      continue;
    mlir::Type merged = types.join(parts);
    if (!merged)
      continue;
    replacementIndices.push_back(static_cast<unsigned>(index));
    replacementTypes.push_back(merged);
    continuation->addArgument(merged, loc(statement));
  }

  auto branchToContinuation = [&](mlir::Block *exitBlock,
                                  llvm::ArrayRef<Value> branchValues,
                                  llvm::ArrayRef<Value> branchMutationValues) {
    builder.setInsertionPointToEnd(exitBlock);
    llvm::SmallVector<mlir::Value, 4> operands;
    for (auto [slot, candidateIndex] : llvm::enumerate(mergedCandidateIndices))
      operands.push_back(
          coerceValue(branchValues[candidateIndex], mergedTypes[slot], statement)
              .value);
    for (unsigned candidateIndex : threadedMutationIndices)
      operands.push_back(candidateIndex < branchMutationValues.size() &&
                                 branchMutationValues[candidateIndex].value
                             ? branchMutationValues[candidateIndex].value
                             : mutationOuterValues[candidateIndex].value);
    for (auto [slot, candidateIndex] : llvm::enumerate(replacementIndices)) {
      const Value &incoming =
          candidateIndex < branchMutationValues.size() &&
                  branchMutationValues[candidateIndex].value
              ? branchMutationValues[candidateIndex]
              : mutationOuterValues[candidateIndex];
      operands.push_back(
          coerceValue(incoming, replacementTypes[slot], statement).value);
    }
    mlir::cf::BranchOp::create(builder, loc(statement), continuation, operands);
  };

  // The entry terminator is created only now so the fall-through edge into
  // the continuation can forward the outer values of threaded mutation
  // locals.
  {
    builder.setInsertionPointToEnd(entry);
    llvm::SmallVector<mlir::Value, 2> falseOperands;
    if (!hasElse) {
      for (auto [slot, candidateIndex] :
           llvm::enumerate(mergedCandidateIndices)) {
        (void)slot;
        (void)candidateIndex;
      }
      for (unsigned candidateIndex : threadedMutationIndices)
        falseOperands.push_back(mutationOuterValues[candidateIndex].value);
      for (auto [slot, candidateIndex] : llvm::enumerate(replacementIndices))
        falseOperands.push_back(
            coerceValue(mutationOuterValues[candidateIndex],
                        replacementTypes[slot], statement)
                .value);
    }
    mlir::cf::CondBranchOp::create(builder, loc(statement), condition,
                                   thenBlock, mlir::ValueRange{}, elseBlock,
                                   falseOperands);
  }

  if (thenExit)
    branchToContinuation(thenExit, thenValues, thenMutationValues);
  if (hasElse && elseExit)
    branchToContinuation(elseExit, elseValues, elseMutationValues);

  setInsertionBeforeTerminator(builder, *continuation);
  for (auto [slot, candidateIndex] : llvm::enumerate(mergedCandidateIndices)) {
    const std::string &name = mergeCandidates[candidateIndex];
    values[name] =
        Value{continuation->getArgument(slot), mergedTypes[slot]};
    types.bindSymbol(name, mergedTypes[slot]);
  }
  for (auto [offset, candidateIndex] :
       llvm::enumerate(threadedMutationIndices)) {
    const std::string &name = mutationCandidates[candidateIndex];
    values[name] = Value{
        continuation->getArgument(mergedCandidateIndices.size() + offset),
        mutationOuterValues[candidateIndex].type};
  }
  for (auto [slot, candidateIndex] : llvm::enumerate(replacementIndices)) {
    const std::string &name = mutationCandidates[candidateIndex];
    unsigned argIndex = static_cast<unsigned>(mergedCandidateIndices.size() +
                                              threadedMutationIndices.size() +
                                              slot);
    values[name] =
        Value{continuation->getArgument(argIndex), replacementTypes[slot]};
    types.bindSymbol(name, replacementTypes[slot]);
  }
  if (narrowing && thenTerminates && !elseTerminates)
    applyNarrowing(*narrowing, /*conditionIsTrue=*/false);
  else if (narrowing && hasElse && elseTerminates && !thenTerminates)
    applyNarrowing(*narrowing, /*conditionIsTrue=*/true);
}

} // namespace lython::emitter
