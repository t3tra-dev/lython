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
namespace {

enum class FinallyCompletion {
  Fallthrough,
  Return,
  Break,
  Continue,
};

void collectAssignedNameTargets(const parser::Node *node,
                                llvm::StringSet<> &names) {
  if (!node)
    return;
  if (node->kind == "Name") {
    names.insert(ast::nameSpelling(*node));
    return;
  }
  if (node->kind == "Tuple" || node->kind == "List") {
    if (const auto *elts = ast::nodeList(*node, "elts"))
      for (const parser::NodePtr &elt : *elts)
        collectAssignedNameTargets(elt.get(), names);
  }
}

// The carried local was replaced through a structural-mutation call chain
// (`ly.structural_mutation`), which already CONSUMED (transferred) the
// previous representation into the call; releasing the previous value again
// would double-consume the ownership token.
bool derivesViaStructuralMutationImpl(
    mlir::Value current, mlir::Value previous,
    llvm::SmallPtrSetImpl<void *> &visited, unsigned depth) {
  if (depth > 32)
    return false;
  while (current && current != previous) {
    // Any structural-mutation op (py.call append, py.setitem, ...) exposes
    // the rebound receiver as its LAST result and the receiver as operand 0.
    if (mlir::Operation *definition = current.getDefiningOp()) {
      if (!definition->hasAttr("ly.structural_mutation") ||
          definition->getNumResults() < 1 ||
          current != definition->getResult(definition->getNumResults() - 1) ||
          definition->getNumOperands() < 1)
        return false;
      current = definition->getOperand(0);
      continue;
    }
    // A merge or loop-header block argument derives via structural mutation
    // when EVERY incoming edge forwards the previous value itself (identity
    // path: the token is forwarded, not consumed), a mutation chain over it,
    // or — coinductively — a chain rooted at an already-visited argument
    // (loop back-edges: assume the invariant holds and check the remaining
    // edges). Skipping the replacement release is sound on all path kinds.
    auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(current);
    if (!blockArg)
      return false;
    if (!visited.insert(current.getAsOpaquePointer()).second)
      return true;
    mlir::Block *block = blockArg.getOwner();
    if (block->hasNoPredecessors())
      return false;
    for (mlir::Block *pred : block->getPredecessors()) {
      mlir::Operation *terminator = pred->getTerminator();
      auto branch = mlir::dyn_cast<mlir::BranchOpInterface>(terminator);
      if (!branch)
        return false;
      for (auto [index, successor] :
           llvm::enumerate(terminator->getSuccessors())) {
        if (successor != block)
          continue;
        mlir::SuccessorOperands operands = branch.getSuccessorOperands(
            static_cast<unsigned>(index));
        mlir::Value incoming = operands[blockArg.getArgNumber()];
        if (!incoming)
          return false;
        if (incoming != previous &&
            !derivesViaStructuralMutationImpl(incoming, previous, visited,
                                              depth + 1))
          return false;
      }
    }
    return true;
  }
  return current == previous;
}

bool derivesViaStructuralMutation(mlir::Value current, mlir::Value previous) {
  llvm::SmallPtrSet<void *, 8> visited;
  return derivesViaStructuralMutationImpl(current, previous, visited,
                                          /*depth=*/0);
}

void collectAssignedNames(const parser::Node *node, llvm::StringSet<> &names) {
  if (!node)
    return;
  if (node->kind == "FunctionDef" || node->kind == "AsyncFunctionDef" ||
      node->kind == "ClassDef" || node->kind == "Lambda")
    return;
  if (node->kind == "Call") {
    // Structural-mutation candidates (`x.append(...)`) rebind `x` at the
    // emitter when the manifest declares the method a structural mutator.
    // This syntactic pre-pass only over-approximates which locals may be
    // reassigned; threading a local that ends up not reassigned is a benign
    // identity forward.
    if (const parser::Node *func = ast::node(*node, "func")) {
      if (func->kind == "Attribute") {
        if (auto attr = ast::string(*func, "attr");
            attr && (*attr == "append" || *attr == "add")) {
          if (const parser::Node *value = ast::node(*func, "value"))
            if (value->kind == "Name")
              names.insert(ast::nameSpelling(*value));
        }
      }
    }
  }
  if (node->kind == "Assign") {
    if (const auto *targets = ast::nodeList(*node, "targets"))
      for (const parser::NodePtr &target : *targets) {
        collectAssignedNameTargets(target.get(), names);
        // `x[k] = v` may structurally mutate (and rebind) `x`; threading a
        // local that ends up not reassigned is a benign identity forward.
        if (target && target->kind == "Subscript")
          if (const parser::Node *container = ast::node(*target, "value"))
            if (container->kind == "Name")
              names.insert(ast::nameSpelling(*container));
      }
  } else if (node->kind == "AnnAssign" || node->kind == "AugAssign" ||
             node->kind == "NamedExpr") {
    collectAssignedNameTargets(ast::node(*node, "target"), names);
  } else if (node->kind == "For" || node->kind == "AsyncFor") {
    collectAssignedNameTargets(ast::node(*node, "target"), names);
  } else if (node->kind == "With" || node->kind == "AsyncWith") {
    if (const auto *items = ast::nodeList(*node, "items"))
      for (const parser::NodePtr &item : *items)
        collectAssignedNameTargets(ast::node(*item, "optional_vars"), names);
  }

  for (const parser::Field &field : node->fields) {
    if (const auto *child = std::get_if<parser::NodePtr>(&field.value)) {
      if (*child)
        collectAssignedNames(child->get(), names);
    } else if (const auto *children =
                   std::get_if<std::vector<parser::NodePtr>>(&field.value)) {
      for (const parser::NodePtr &child : *children)
        if (child)
          collectAssignedNames(child.get(), names);
    }
  }
}

void collectAssignedNames(const std::vector<parser::NodePtr> *statements,
                          llvm::StringSet<> &names) {
  if (!statements)
    return;
  for (const parser::NodePtr &statement : *statements)
    collectAssignedNames(statement.get(), names);
}

Value stripLocalProtocolView(Value value) {
  if (!value.value)
    return value;
  auto view = value.value.getDefiningOp<py::ProtocolViewOp>();
  if (!view)
    return value;
  return Value{view.getInput(), view.getInput().getType()};
}

bool isIntegerLiteralSpelling(llvm::StringRef spelling) {
  if (spelling.empty())
    return false;
  if (spelling.front() == '-')
    spelling = spelling.drop_front();
  return !spelling.empty() &&
         llvm::all_of(spelling, [](char c) { return c >= '0' && c <= '9'; });
}

bool isSupportedFinallyReturnCarrierType(mlir::Type type) {
  if (!type)
    return false;
  // NOTE: `builtins.int` (and bare integer literals) are intentionally excluded.
  // An int-returning function may be lowered with the unboxed i64 primitive
  // return ABI, whose return lowering (RuntimeBundleLowerer::lowerFunctionReturns
  // primitive-i64 path) does not yet handle a value carried out of a finally
  // block and crashes on it. Excluding int here turns that into the stable
  // "return inside try is not implemented yet" diagnostic instead of a compiler
  // crash. Re-enable once the i64 return path handles finally-carried returns.
  if (auto literal = mlir::dyn_cast<py::LiteralType>(type)) {
    llvm::StringRef spelling = literal.getSpelling();
    return spelling == "None" || spelling == "True" || spelling == "False" ||
           (spelling.size() >= 2 && spelling.front() == '"' &&
            spelling.back() == '"');
  }
  if (auto contract = mlir::dyn_cast<py::ContractType>(type)) {
    llvm::StringRef name = contract.getContractName();
    return name == "types.NoneType" || name == "builtins.bool" ||
           name == "builtins.float" || name == "builtins.str" ||
           name == "builtins.object";
  }
  return false;
}

template <typename YieldOp, typename BuildValues>
unsigned terminateOpenRegionBlocks(mlir::OpBuilder &builder, mlir::Location loc,
                                   mlir::Region &region,
                                   BuildValues buildValues) {
  llvm::SmallVector<mlir::Block *, 8> openBlocks;
  for (mlir::Block &block : region)
    if (!blockHasTerminator(block))
      openBlocks.push_back(&block);
  for (mlir::Block *block : openBlocks) {
    builder.setInsertionPointToEnd(block);
    llvm::SmallVector<mlir::Value, 4> values;
    buildValues(values);
    YieldOp::create(builder, loc, values);
  }
  return static_cast<unsigned>(openBlocks.size());
}

template <typename YieldOp>
unsigned terminateOpenRegionBlocks(mlir::OpBuilder &builder, mlir::Location loc,
                                   mlir::Region &region) {
  return terminateOpenRegionBlocks<YieldOp>(
      builder, loc, region, [](llvm::SmallVectorImpl<mlir::Value> &) {});
}

} // namespace

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

void ModuleEmitter::emitMatch(const parser::Node &statement) {
  const parser::Node *subjectNode = ast::node(statement, "subject");
  const auto *cases = ast::nodeList(statement, "cases");
  if (!subjectNode || !cases || cases->empty()) {
    diagnostics.push_back(parser::Diagnostic{parser::Severity::Error,
                                             statement.range.start,
                                             "empty match is not supported"});
    return;
  }
  Value subject = emitExpr(subjectNode);

  mlir::Block *entry = builder.getInsertionBlock();
  mlir::Region *region = entry->getParent();
  mlir::Block *continuation = entry->splitBlock(builder.getInsertionPoint());
  mlir::Block *check = builder.createBlock(region, continuation->getIterator());
  builder.setInsertionPointToEnd(entry);
  mlir::cf::BranchOp::create(builder, loc(statement), check);

  // Equality test `subject == <constant node>` yielding an i1 condition.
  auto equalsConstant = [&](const parser::Node &anchor,
                            const parser::Node *valueNode) -> mlir::Value {
    Value patternValue = emitExpr(valueNode);
    Value compared = emitBinarySpecial<py::EqOp>(anchor, "__eq__", subject,
                                                 patternValue, types.boolType());
    return emitBoolValue(compared, anchor);
  };

  bool matchedAll = false;
  // Flow-sensitive subject narrowing across the case chain: after a failed
  // union-member class test, the remaining cases see the union minus that
  // member, so the final member's class pattern becomes irrefutable and the
  // chain provably terminates (no fall-through path).
  mlir::Type matchSubjectType = subject.type;
  for (const parser::NodePtr &caseNodePtr : *cases) {
    if (!caseNodePtr)
      continue;
    const parser::Node &caseNode = *caseNodePtr;
    const parser::Node *pattern = ast::node(caseNode, "pattern");
    const parser::Node *guard = ast::node(caseNode, "guard");
    const auto *body = ast::nodeList(caseNode, "body");
    if (!pattern) {
      diagnostics.push_back(parser::Diagnostic{parser::Severity::Error,
                                               statement.range.start,
                                               "match case has no pattern"});
      return;
    }

    ScopedEmitterScope scope(values, types);
    builder.setInsertionPointToStart(check);

    // A nullopt condition means the pattern is irrefutable; unsupported
    // pattern kinds are rejected below with a diagnostic instead of silently
    // falling through.
    std::optional<mlir::Value> condition;
    bool unsupported = false;
    bool staticallyFalse = false;
    if (pattern->kind == "MatchAs" && !ast::node(*pattern, "pattern")) {
      if (std::optional<std::string_view> name =
              ast::string(*pattern, "name")) {
        values[std::string(*name)] = subject;
        types.bindSymbol(*name, subject.type);
      }
    } else if (pattern->kind == "MatchValue") {
      condition = equalsConstant(*pattern, ast::node(*pattern, "value"));
    } else if (pattern->kind == "MatchSingleton" &&
               ast::isNoneField(*pattern, "value")) {
      // `case None:` — identity test against the None singleton.
      if (auto unionType =
              mlir::dyn_cast_if_present<py::UnionType>(subject.type)) {
        if (unionType.hasMember(types.none()))
          condition = py::UnionTestOp::create(
                          builder, loc(statement), builder.getI1Type(),
                          subject.value, mlir::TypeAttr::get(types.none()))
                          .getResult();
        else
          unsupported = true;
      } else if (subject.type == types.none()) {
        // Subject is always None: irrefutable (condition stays nullopt).
      } else {
        unsupported = true;
      }
    } else if (pattern->kind == "MatchSingleton") {
      // `case True:` / `case False:` — use the subject's truthiness (its
      // runtime `__eq__` is not available). Only sound for a bool subject,
      // where the truth value distinguishes the two singletons; for other
      // subjects `case True` means `== 1`, which truthiness does not capture.
      std::optional<bool> flag = ast::boolean(*pattern, "value");
      if (flag && subject.type == types.boolType()) {
        mlir::Value truth = emitBoolValue(subject, *pattern);
        if (*flag) {
          condition = truth;
        } else {
          mlir::Value one =
              mlir::arith::ConstantIntOp::create(builder, loc(statement), 1, 1);
          condition =
              mlir::arith::XOrIOp::create(builder, loc(statement), truth, one)
                  .getResult();
        }
      } else {
        unsupported = true;
      }
    } else if (pattern->kind == "MatchOr") {
      const auto *alts = ast::nodeList(*pattern, "patterns");
      if (!alts || alts->empty()) {
        unsupported = true;
      } else {
        for (const parser::NodePtr &alt : *alts) {
          if (!alt || alt->kind != "MatchValue") {
            unsupported = true;
            break;
          }
          mlir::Value altCond = equalsConstant(*alt, ast::node(*alt, "value"));
          condition = condition ? mlir::arith::OrIOp::create(
                                      builder, loc(statement), *condition,
                                      altCond)
                                      .getResult()
                                : altCond;
        }
      }
    } else if (pattern->kind == "MatchSequence") {
      // Sequence destructuring over a tuple/list subject. A sequence pattern
      // is a runtime length test (`len(subject) == N`) guarding per-element
      // extraction; element getitems are emitted only behind the length gate
      // so a shorter subject never reaches an out-of-range access.
      const auto *subPatterns = ast::nodeList(*pattern, "patterns");
      auto contract =
          mlir::dyn_cast_if_present<py::ContractType>(subject.type);
      bool sequenceSubject =
          contract && (contract.getContractName() == "builtins.tuple" ||
                       contract.getContractName() == "builtins.list");
      bool shapeSupported = sequenceSubject && subPatterns && !guard;
      constexpr unsigned kNoStar = ~0u;
      unsigned starIndex = kNoStar;
      if (shapeSupported)
        for (auto [subIndex, subPattern] : llvm::enumerate(*subPatterns)) {
          bool captureLike = subPattern && subPattern->kind == "MatchAs" &&
                             !ast::node(*subPattern, "pattern");
          bool literalLike = subPattern && subPattern->kind == "MatchValue";
          bool starLike = subPattern && subPattern->kind == "MatchStar";
          if (starLike) {
            // One star, and only in the trailing position for now.
            if (starIndex != kNoStar || subIndex + 1 != subPatterns->size()) {
              shapeSupported = false;
              break;
            }
            starIndex = static_cast<unsigned>(subIndex);
            continue;
          }
          if (!captureLike && !literalLike) {
            shapeSupported = false;
            break;
          }
        }
      if (!shapeSupported) {
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, statement.range.start,
            "match sequence pattern requires a tuple/list subject with "
            "capture or literal elements (one trailing *rest allowed; guards "
            "not supported here yet)"});
        return;
      }
      unsigned prefixCount = starIndex == kNoStar
                                 ? static_cast<unsigned>(subPatterns->size())
                                 : starIndex;

      CallInferenceResult lenInference =
          types.inferMethodCallWithEvidence(subject.type, "__len__", {});
      if (!lenInference) {
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, statement.range.start,
            "match sequence pattern subject has no __len__ evidence"});
        return;
      }
      Value length{py::LenOp::create(
                       builder, loc(statement), lenInference.resultType,
                       mlir::FlatSymbolRefAttr::get(&context, "__len__"),
                       callProtocolFor(lenInference), subject.value)
                       .getResult(),
                   lenInference.resultType};
      std::string arityText = std::to_string(prefixCount);
      mlir::Type arityType = types.literal(arityText);
      Value arity{py::IntConstantOp::create(builder, loc(statement), arityType,
                                            builder.getStringAttr(arityText))
                      .getResult(),
                  arityType};
      Value lengthCompared =
          starIndex == kNoStar
              ? emitBinarySpecial<py::EqOp>(*pattern, "__eq__", length, arity,
                                            types.boolType())
              : emitBinarySpecial<py::GeOp>(*pattern, "__ge__", length, arity,
                                            types.boolType());
      mlir::Value lengthMatches = emitBoolValue(lengthCompared, *pattern);

      mlir::Block *elementBlock =
          builder.createBlock(region, continuation->getIterator());
      mlir::Block *nextCheck =
          builder.createBlock(region, continuation->getIterator());
      builder.setInsertionPointToEnd(check);
      mlir::cf::CondBranchOp::create(builder, loc(statement), lengthMatches,
                                     elementBlock, mlir::ValueRange{},
                                     nextCheck, mlir::ValueRange{});

      builder.setInsertionPointToStart(elementBlock);
      auto sequenceElement = [&](unsigned index) -> std::optional<Value> {
        std::string text = std::to_string(index);
        mlir::Type literalType = types.literal(text);
        Value indexValue{
            py::IntConstantOp::create(builder, loc(statement), literalType,
                                      builder.getStringAttr(text))
                .getResult(),
            literalType};
        CallInferenceResult inference = types.inferMethodCallWithEvidence(
            subject.type, "__getitem__", {indexValue.type});
        if (!inference)
          return std::nullopt;
        auto op = py::GetItemOp::create(
            builder, loc(statement), inference.resultType,
            mlir::FlatSymbolRefAttr::get(&context, "__getitem__"),
            callProtocolFor(inference), subject.value, indexValue.value);
        return Value{op.getResult(), inference.resultType};
      };
      std::optional<mlir::Value> elementCondition;
      bool elementsSupported = true;
      for (auto [index, subPattern] : llvm::enumerate(*subPatterns)) {
        if (subPattern->kind == "MatchStar")
          continue; // handled below
        if (subPattern->kind == "MatchAs") {
          std::optional<std::string_view> name =
              ast::string(*subPattern, "name");
          if (!name)
            continue; // wildcard element
          std::optional<Value> element =
              sequenceElement(static_cast<unsigned>(index));
          if (!element) {
            elementsSupported = false;
            break;
          }
          values[std::string(*name)] = *element;
          types.bindSymbol(*name, element->type);
          continue;
        }
        std::optional<Value> element =
            sequenceElement(static_cast<unsigned>(index));
        if (!element) {
          elementsSupported = false;
          break;
        }
        Value patternValue = emitExpr(ast::node(*subPattern, "value"));
        Value compared = emitBinarySpecial<py::EqOp>(
            *subPattern, "__eq__", *element, patternValue, types.boolType());
        mlir::Value elementCond = emitBoolValue(compared, *subPattern);
        elementCondition =
            elementCondition
                ? mlir::arith::AndIOp::create(builder, loc(statement),
                                              *elementCondition, elementCond)
                      .getResult()
                : elementCond;
      }
      if (!elementsSupported) {
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, statement.range.start,
            "match sequence pattern element has no __getitem__ evidence"});
        return;
      }
      if (starIndex != kNoStar) {
        // `*rest` materializes the remaining elements as a fresh list via a
        // synthetic build loop; `*_` needs no materialization (the >= length
        // gate is the whole check).
        std::optional<std::string_view> starName =
            ast::string(*(*subPatterns)[starIndex], "name");
        if (starName) {
          CallInferenceResult getInference = types.inferMethodCallWithEvidence(
              subject.type, "__getitem__", {types.contract("builtins.int")});
          if (!getInference) {
            diagnostics.push_back(parser::Diagnostic{
                parser::Severity::Error, statement.range.start,
                "match sequence *rest requires runtime-index __getitem__ "
                "evidence"});
            return;
          }
          mlir::Type elementType = types.widenLiteral(getInference.resultType);
          mlir::Type restType = py::ContractType::get(
              builder.getContext(), "builtins.list", {elementType});
          std::string subjLocal =
              "__matchseq" + std::to_string(++listCompCounter);
          std::string restLocal =
              "__matchrest" + std::to_string(listCompCounter);
          std::string idxLocal = "__matchidx" + std::to_string(listCompCounter);
          values[subjLocal] = subject;
          types.bindSymbol(subjLocal, subject.type);
          auto packRest = py::PackOp::create(builder, loc(statement), restType,
                                             mlir::ValueRange{});
          values[restLocal] = Value{packRest.getResult(), restType};
          types.bindSymbol(restLocal, restType);
          auto nameNode = [&](const std::string &id) {
            parser::NodePtr node = parser::makeNode("Name", statement.range);
            parser::addField(*node, "id", id);
            return node;
          };
          // for __idx in range(<prefix>, len(__subj)):
          //   __rest.append(__subj[__idx])
          parser::NodePtr prefixNode =
              parser::makeNode("Constant", statement.range);
          parser::addField(*prefixNode, "value",
                           static_cast<std::int64_t>(prefixCount));
          parser::NodePtr lenCall = parser::makeNode("Call", statement.range);
          parser::addField(*lenCall, "func", nameNode("len"));
          parser::addField(*lenCall, "args",
                           std::vector<parser::NodePtr>{nameNode(subjLocal)});
          parser::addField(*lenCall, "keywords",
                           std::vector<parser::NodePtr>{});
          parser::NodePtr rangeCall = parser::makeNode("Call", statement.range);
          parser::addField(*rangeCall, "func", nameNode("range"));
          parser::addField(*rangeCall, "args",
                           std::vector<parser::NodePtr>{prefixNode, lenCall});
          parser::addField(*rangeCall, "keywords",
                           std::vector<parser::NodePtr>{});
          parser::NodePtr subscript =
              parser::makeNode("Subscript", statement.range);
          parser::addField(*subscript, "value", nameNode(subjLocal));
          parser::addField(*subscript, "slice", nameNode(idxLocal));
          parser::NodePtr appendAttr =
              parser::makeNode("Attribute", statement.range);
          parser::addField(*appendAttr, "value", nameNode(restLocal));
          parser::addField(*appendAttr, "attr", std::string("append"));
          parser::NodePtr appendCall =
              parser::makeNode("Call", statement.range);
          parser::addField(*appendCall, "func", appendAttr);
          parser::addField(*appendCall, "args",
                           std::vector<parser::NodePtr>{subscript});
          parser::addField(*appendCall, "keywords",
                           std::vector<parser::NodePtr>{});
          parser::NodePtr appendStmt =
              parser::makeNode("Expr", statement.range);
          parser::addField(*appendStmt, "value", appendCall);
          parser::NodePtr buildLoop =
              parser::makeNode("For", statement.range);
          parser::addField(*buildLoop, "target", nameNode(idxLocal));
          parser::addField(*buildLoop, "iter", rangeCall);
          parser::addField(*buildLoop, "body",
                           std::vector<parser::NodePtr>{appendStmt});
          parser::addField(*buildLoop, "orelse",
                           std::vector<parser::NodePtr>{});
          emitFor(*buildLoop);
          auto builtRest = values.find(restLocal);
          if (builtRest != values.end() && builtRest->second.value) {
            values[std::string(*starName)] = builtRest->second;
            types.bindSymbol(*starName, builtRest->second.type);
          }
          values.erase(restLocal);
          values.erase(subjLocal);
          values.erase(idxLocal);
        }
      }
      if (elementCondition) {
        mlir::Block *conditionBlock = builder.getInsertionBlock();
        mlir::Block *bodyBlock =
            builder.createBlock(region, continuation->getIterator());
        builder.setInsertionPointToEnd(conditionBlock);
        mlir::cf::CondBranchOp::create(builder, loc(statement),
                                       *elementCondition, bodyBlock,
                                       mlir::ValueRange{}, nextCheck,
                                       mlir::ValueRange{});
        builder.setInsertionPointToStart(bodyBlock);
      }
      emitStatements(body);
      if (!insertionBlockTerminated(builder))
        mlir::cf::BranchOp::create(builder, loc(statement), continuation);
      check = nextCheck;
      continue;
    } else if (pattern->kind == "MatchClass") {
      // Class pattern over a statically resolved class: reuses the isinstance
      // evidence analysis (union member test / subclass test), then binds
      // attribute captures and evaluates literal sub-pattern equalities from
      // the narrowed value inside the gated block. Positional sub-patterns
      // resolve their attribute names through the class's __match_args__.
      const parser::Node *clsNode = ast::node(*pattern, "cls");
      const auto *positionalSubs = ast::nodeList(*pattern, "patterns");
      const auto *kwdAttrs = ast::stringList(*pattern, "kwd_attrs");
      const auto *kwdPatterns = ast::nodeList(*pattern, "kwd_patterns");
      auto supportedSubPattern = [](const parser::NodePtr &sub) {
        if (!sub)
          return false;
        if (sub->kind == "MatchAs")
          return ast::node(*sub, "pattern") == nullptr;
        return sub->kind == "MatchValue";
      };
      bool shapeSupported =
          !guard && ((kwdAttrs == nullptr) == (kwdPatterns == nullptr));
      std::size_t keywordCount = kwdAttrs ? kwdAttrs->size() : 0;
      if (shapeSupported && kwdPatterns) {
        if (kwdPatterns->size() != keywordCount)
          shapeSupported = false;
        else
          for (const parser::NodePtr &sub : *kwdPatterns)
            if (!supportedSubPattern(sub)) {
              shapeSupported = false;
              break;
            }
      }
      if (shapeSupported && positionalSubs)
        for (const parser::NodePtr &sub : *positionalSubs)
          if (!supportedSubPattern(sub)) {
            shapeSupported = false;
            break;
          }
      std::optional<mlir::Type> target =
          shapeSupported ? isinstanceTargetType(clsNode, types) : std::nullopt;
      if (!shapeSupported || !target) {
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, statement.range.start,
            "match class pattern requires a statically resolved class with "
            "capture or literal sub-patterns (no guards)"});
        return;
      }
      // (attribute name, sub-pattern) pairs: positional names resolve through
      // the class's __match_args__ tuple, keyword names are explicit.
      llvm::SmallVector<std::pair<std::string, const parser::Node *>, 4>
          attrPatterns;
      if (positionalSubs && !positionalSubs->empty()) {
        std::optional<std::vector<std::string>> matchArgs =
            types.classMatchArgs(*target);
        if (!matchArgs || positionalSubs->size() > matchArgs->size()) {
          diagnostics.push_back(parser::Diagnostic{
              parser::Severity::Error, statement.range.start,
              "match class positional patterns require a __match_args__ "
              "string-literal tuple with at least as many names"});
          return;
        }
        for (auto [index, sub] : llvm::enumerate(*positionalSubs))
          attrPatterns.push_back({(*matchArgs)[index], sub.get()});
      }
      for (std::size_t index = 0; index < keywordCount; ++index)
        attrPatterns.push_back(
            {std::string((*kwdAttrs)[index]), (*kwdPatterns)[index].get()});
      IsInstanceAnalysis analysis =
          analyzeIsInstance(matchSubjectType, *target, types, module);
      if (analysis.kind == IsInstanceAnalysis::Kind::Unsupported ||
          analysis.kind == IsInstanceAnalysis::Kind::UnionClassTest ||
          (analysis.kind == IsInstanceAnalysis::Kind::UnionTest &&
           !analysis.trueType)) {
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, statement.range.start,
            analysis.failureReason.empty()
                ? "match class pattern has unsupported isinstance evidence"
                : analysis.failureReason});
        return;
      }
      if (analysis.kind == IsInstanceAnalysis::Kind::AlwaysFalse)
        continue; // statically impossible case

      mlir::Block *matchBlock = nullptr;
      mlir::Block *nextCheck = nullptr;
      if (analysis.kind != IsInstanceAnalysis::Kind::AlwaysTrue) {
        mlir::Value bit;
        if (analysis.kind == IsInstanceAnalysis::Kind::UnionTest) {
          bit = py::UnionTestOp::create(
                    builder, loc(statement), builder.getI1Type(), subject.value,
                    mlir::TypeAttr::get(analysis.trueType))
                    .getResult();
        } else { // ClassTest
          bit = py::ClassTestOp::create(
                    builder, loc(statement), builder.getI1Type(), subject.value,
                    mlir::TypeAttr::get(analysis.targetType))
                    .getResult();
        }
        matchBlock = builder.createBlock(region, continuation->getIterator());
        nextCheck = builder.createBlock(region, continuation->getIterator());
        builder.setInsertionPointToEnd(check);
        mlir::cf::CondBranchOp::create(builder, loc(statement), bit, matchBlock,
                                       mlir::ValueRange{}, nextCheck,
                                       mlir::ValueRange{});
        builder.setInsertionPointToStart(matchBlock);
      }

      Value narrowed = subject;
      if (analysis.kind == IsInstanceAnalysis::Kind::UnionTest) {
        auto unwrap = py::UnionUnwrapOp::create(builder, loc(statement),
                                                analysis.trueType,
                                                subject.value);
        narrowed = Value{unwrap.getResult(), analysis.trueType};
        if (analysis.trueType != analysis.targetType &&
            mlir::isa<py::ContractType>(analysis.trueType) &&
            mlir::isa<py::ContractType>(analysis.targetType) &&
            py::isAssignableTo(analysis.targetType, analysis.trueType,
                               module)) {
          auto refine = py::ClassRefineOp::create(
              builder, loc(statement), analysis.targetType, narrowed.value);
          narrowed = Value{refine.getResult(), analysis.targetType};
        }
      } else if (analysis.kind == IsInstanceAnalysis::Kind::ClassTest) {
        auto refine = py::ClassRefineOp::create(
            builder, loc(statement), analysis.targetType, subject.value);
        narrowed = Value{refine.getResult(), analysis.targetType};
      } else if (analysis.kind == IsInstanceAnalysis::Kind::AlwaysTrue &&
                 mlir::isa<py::UnionType>(subject.value.getType()) &&
                 mlir::isa<py::ContractType>(matchSubjectType)) {
        // The chain narrowed the subject to a single union member; the SSA
        // value is still union-shaped, so extract the member payload.
        auto unwrap = py::UnionUnwrapOp::create(builder, loc(statement),
                                                matchSubjectType,
                                                subject.value);
        narrowed = Value{unwrap.getResult(), matchSubjectType};
      }

      bool capturesSupported = true;
      std::optional<mlir::Value> valueCondition;
      for (auto &[attrName, sub] : attrPatterns) {
        std::optional<mlir::Type> field =
            lookupClassField(narrowed.type, attrName);
        if (!field) {
          capturesSupported = false;
          break;
        }
        bool isCapture = sub->kind == "MatchAs";
        std::optional<std::string_view> captureName =
            isCapture ? ast::string(*sub, "name") : std::nullopt;
        if (isCapture && !captureName)
          continue; // wildcard positional: field existence is the only check
        auto attrGet = py::AttrGetOp::create(
            builder, loc(statement), *field, narrowed.value, attrName);
        attrGet->setAttr("ly.attr.kind", builder.getStringAttr("field"));
        if (auto contract =
                mlir::dyn_cast_if_present<py::ContractType>(narrowed.type))
          attrGet->setAttr("ly.attr.owner",
                           builder.getStringAttr(contract.getContractName()));
        if (isCapture) {
          values[std::string(*captureName)] = Value{attrGet.getResult(), *field};
          types.bindSymbol(*captureName, *field);
          continue;
        }
        // MatchValue: gate the case body on attribute equality.
        Value element{attrGet.getResult(), *field};
        Value patternValue = emitExpr(ast::node(*sub, "value"));
        Value compared = emitBinarySpecial<py::EqOp>(
            *sub, "__eq__", element, patternValue, types.boolType());
        mlir::Value condition = emitBoolValue(compared, *sub);
        valueCondition =
            valueCondition
                ? mlir::arith::AndIOp::create(builder, loc(statement),
                                              *valueCondition, condition)
                      .getResult()
                : condition;
      }
      if (!capturesSupported) {
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, statement.range.start,
            "match class pattern sub-pattern must name a declared field"});
        return;
      }

      bool refutableByValue = valueCondition.has_value();
      if (valueCondition) {
        mlir::Block *conditionBlock = builder.getInsertionBlock();
        if (!nextCheck)
          nextCheck = builder.createBlock(region, continuation->getIterator());
        mlir::Block *bodyBlock =
            builder.createBlock(region, continuation->getIterator());
        builder.setInsertionPointToEnd(conditionBlock);
        mlir::cf::CondBranchOp::create(builder, loc(statement),
                                       *valueCondition, bodyBlock,
                                       mlir::ValueRange{}, nextCheck,
                                       mlir::ValueRange{});
        builder.setInsertionPointToStart(bodyBlock);
      }
      emitStatements(body);
      if (!insertionBlockTerminated(builder))
        mlir::cf::BranchOp::create(builder, loc(statement), continuation);
      if (!nextCheck) {
        // Irrefutable class pattern: terminates the chain.
        matchedAll = true;
        break;
      }
      // On the fall-through edge the tested member is excluded — but only
      // when falling through can only mean the class test failed (a value
      // inequality also falls through without excluding the member).
      if (analysis.kind == IsInstanceAnalysis::Kind::UnionTest &&
          analysis.falseType && !refutableByValue)
        matchSubjectType = analysis.falseType;
      check = nextCheck;
      continue;
    } else if (pattern->kind == "MatchMapping") {
      // Mapping pattern over a dict subject: a `key in subject` test per
      // pattern key guards the value extraction, so a missing key is a
      // non-match (never a KeyError).
      const auto *keys = ast::nodeList(*pattern, "keys");
      const auto *valuePatterns = ast::nodeList(*pattern, "patterns");
      auto contract =
          mlir::dyn_cast_if_present<py::ContractType>(subject.type);
      bool shapeSupported = contract &&
                            contract.getContractName() == "builtins.dict" &&
                            keys && valuePatterns &&
                            keys->size() == valuePatterns->size() && !guard &&
                            !ast::node(*pattern, "rest");
      if (shapeSupported)
        for (const parser::NodePtr &sub : *valuePatterns) {
          bool captureLike = sub && sub->kind == "MatchAs" &&
                             !ast::node(*sub, "pattern");
          bool literalLike = sub && sub->kind == "MatchValue";
          if (!captureLike && !literalLike) {
            shapeSupported = false;
            break;
          }
        }
      if (!shapeSupported) {
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, statement.range.start,
            "match mapping pattern requires a dict subject with capture or "
            "literal values (no rest/guard)"});
        return;
      }

      // Stage 1: presence conditions in the check block.
      llvm::SmallVector<Value, 4> keyValues;
      std::optional<mlir::Value> present;
      for (const parser::NodePtr &keyNode : *keys) {
        Value key = emitExpr(keyNode.get());
        keyValues.push_back(key);
        CallInferenceResult inference = types.inferMethodCallWithEvidence(
            subject.type, "__contains__", {key.type});
        if (!inference) {
          diagnostics.push_back(parser::Diagnostic{
              parser::Severity::Error, statement.range.start,
              "match mapping subject has no __contains__ evidence"});
          return;
        }
        auto contains = py::ContainsOp::create(
            builder, loc(statement), builder.getI1Type(),
            mlir::FlatSymbolRefAttr::get(&context, "__contains__"),
            callProtocolFor(inference), subject.value, key.value);
        present = present ? mlir::arith::AndIOp::create(
                                builder, loc(statement), *present,
                                contains.getResult())
                                .getResult()
                          : contains.getResult();
      }

      mlir::Block *valueBlock =
          builder.createBlock(region, continuation->getIterator());
      mlir::Block *nextCheck =
          builder.createBlock(region, continuation->getIterator());
      builder.setInsertionPointToEnd(check);
      if (present) {
        mlir::cf::CondBranchOp::create(builder, loc(statement), *present,
                                       valueBlock, mlir::ValueRange{},
                                       nextCheck, mlir::ValueRange{});
      } else {
        mlir::cf::BranchOp::create(builder, loc(statement), valueBlock);
      }

      // Stage 2: gated value extraction, capture binds, literal compares.
      builder.setInsertionPointToStart(valueBlock);
      std::optional<mlir::Value> valueCondition;
      for (auto [index, sub] : llvm::enumerate(*valuePatterns)) {
        Value key = keyValues[index];
        CallInferenceResult inference = types.inferMethodCallWithEvidence(
            subject.type, "__getitem__", {key.type});
        if (!inference) {
          diagnostics.push_back(parser::Diagnostic{
              parser::Severity::Error, statement.range.start,
              "match mapping subject has no __getitem__ evidence"});
          return;
        }
        auto item = py::GetItemOp::create(
            builder, loc(statement), inference.resultType,
            mlir::FlatSymbolRefAttr::get(&context, "__getitem__"),
            callProtocolFor(inference), subject.value, key.value);
        Value element{item.getResult(), inference.resultType};
        if (sub->kind == "MatchAs") {
          if (std::optional<std::string_view> name = ast::string(*sub, "name")) {
            values[std::string(*name)] = element;
            types.bindSymbol(*name, element.type);
          }
          continue;
        }
        Value patternValue = emitExpr(ast::node(*sub, "value"));
        Value compared = emitBinarySpecial<py::EqOp>(
            *sub, "__eq__", element, patternValue, types.boolType());
        mlir::Value bit = emitBoolValue(compared, *sub);
        valueCondition = valueCondition
                             ? mlir::arith::AndIOp::create(
                                   builder, loc(statement), *valueCondition, bit)
                                   .getResult()
                             : bit;
      }
      if (valueCondition) {
        mlir::Block *conditionBlock = builder.getInsertionBlock();
        mlir::Block *bodyBlock =
            builder.createBlock(region, continuation->getIterator());
        builder.setInsertionPointToEnd(conditionBlock);
        mlir::cf::CondBranchOp::create(builder, loc(statement),
                                       *valueCondition, bodyBlock,
                                       mlir::ValueRange{}, nextCheck,
                                       mlir::ValueRange{});
        builder.setInsertionPointToStart(bodyBlock);
      }
      emitStatements(body);
      if (!insertionBlockTerminated(builder))
        mlir::cf::BranchOp::create(builder, loc(statement), continuation);
      check = nextCheck;
      continue;
    } else {
      unsupported = true;
    }
    if (staticallyFalse)
      continue;
    if (unsupported) {
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, statement.range.start,
          "match pattern '" + pattern->kind + "' is not implemented yet"});
      return;
    }

    // A guard makes even an irrefutable pattern refutable.
    if (guard) {
      mlir::Value guardCond = emitBoolValue(emitExpr(guard), *guard);
      condition = condition ? mlir::arith::AndIOp::create(
                                  builder, loc(statement), *condition, guardCond)
                                  .getResult()
                            : guardCond;
    }

    if (!condition) {
      // Irrefutable: emit the body and terminate the chain.
      emitStatements(body);
      if (!insertionBlockTerminated(builder))
        mlir::cf::BranchOp::create(builder, loc(statement), continuation);
      matchedAll = true;
      break;
    }

    mlir::Block *bodyBlock =
        builder.createBlock(region, continuation->getIterator());
    mlir::Block *nextCheck =
        builder.createBlock(region, continuation->getIterator());
    builder.setInsertionPointToEnd(check);
    mlir::cf::CondBranchOp::create(builder, loc(statement), *condition,
                                   bodyBlock, mlir::ValueRange{}, nextCheck,
                                   mlir::ValueRange{});
    builder.setInsertionPointToStart(bodyBlock);
    emitStatements(body);
    if (!insertionBlockTerminated(builder))
      mlir::cf::BranchOp::create(builder, loc(statement), continuation);
    check = nextCheck;
  }

  // No irrefutable case matched: fall through to the continuation.
  if (!matchedAll) {
    builder.setInsertionPointToEnd(check);
    if (!insertionBlockTerminated(builder))
      mlir::cf::BranchOp::create(builder, loc(statement), continuation);
  }
  builder.setInsertionPointToStart(continuation);
}

llvm::SmallVector<mlir::Value, 4>
ModuleEmitter::loopCarriedBranchOperands(const parser::Node &anchor,
                                         const LoopControlContext &loop,
                                         mlir::Block *target) {
  llvm::SmallVector<mlir::Value, 4> operands;
  operands.reserve(loop.carriedLocals.size());
  for (auto [index, name] : llvm::enumerate(loop.carriedLocals)) {
    mlir::Type expected = target->getArgument(index).getType();
    auto found = values.find(name);
    if (found == values.end() || !found->second.value) {
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, anchor.range.start,
          "loop lost carried local '" + name + "'"});
      continue;
    }
    mlir::Value value = coerceValue(found->second, expected, anchor).value;
    // Release the current-iteration header value when the body replaced it, so
    // the loop-carried ownership token stays balanced on this break/continue
    // edge (mirrors the fall-through back-edge decref-on-replace).
    if (loop.headerBlock && index < loop.headerBlock->getNumArguments()) {
      mlir::Value previous = loop.headerBlock->getArgument(index);
      if (value != previous &&
          !derivesViaStructuralMutation(value, previous))
        py::DecRefOp::create(builder, loc(anchor), previous);
    }
    operands.push_back(value);
  }
  return operands;
}

// A generator expression consumed directly by a for loop fuses into nested
// loops — no generator machinery involved: `for v in (E for x in IT if C):
// BODY` becomes `for x in IT: if C: v = E; BODY`. Genexpr targets are scope
// isolated (bindings restored) like comprehension targets.
void ModuleEmitter::emitGeneratorExpFor(const parser::Node &statement,
                                        const parser::Node &genexpr) {
  auto reject = [&](llvm::StringRef reason) {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, statement.range.start, std::string(reason)});
  };
  const parser::Field *eltField = parser::findField(genexpr, "elt");
  if (!eltField || !std::holds_alternative<parser::NodePtr>(eltField->value))
    return reject("malformed generator expression");
  parser::NodePtr elt = std::get<parser::NodePtr>(eltField->value);
  const auto *generators = ast::nodeList(genexpr, "generators");
  if (!elt || !generators || generators->empty())
    return reject("malformed generator expression");

  struct GenEntry {
    parser::NodePtr target;
    parser::NodePtr iter;
    std::string targetName;
    llvm::SmallVector<parser::NodePtr, 2> filters;
  };
  llvm::SmallVector<GenEntry, 2> chain;
  for (const parser::NodePtr &generator : *generators) {
    if (!generator)
      return reject("malformed generator expression");
    if (ast::integer(*generator, "is_async").value_or(0))
      return reject("async generator expressions are not supported");
    const parser::Field *targetField = parser::findField(*generator, "target");
    const parser::Field *iterField = parser::findField(*generator, "iter");
    if (!targetField ||
        !std::holds_alternative<parser::NodePtr>(targetField->value) ||
        !iterField ||
        !std::holds_alternative<parser::NodePtr>(iterField->value))
      return reject("malformed generator expression");
    GenEntry entry;
    entry.target = std::get<parser::NodePtr>(targetField->value);
    entry.iter = std::get<parser::NodePtr>(iterField->value);
    if (!entry.target || entry.target->kind != "Name" || !entry.iter)
      return reject("generator expression target must be a simple name");
    entry.targetName = std::string(ast::nameSpelling(*entry.target));
    if (const auto *ifs = ast::nodeList(*generator, "ifs"))
      entry.filters.append(ifs->begin(), ifs->end());
    chain.push_back(std::move(entry));
  }

  const parser::Field *forTargetField = parser::findField(statement, "target");
  const parser::Field *forBodyField = parser::findField(statement, "body");
  if (!forTargetField ||
      !std::holds_alternative<parser::NodePtr>(forTargetField->value) ||
      !forBodyField ||
      !std::holds_alternative<std::vector<parser::NodePtr>>(
          forBodyField->value))
    return reject("malformed for statement over a generator expression");
  parser::NodePtr forTarget =
      std::get<parser::NodePtr>(forTargetField->value);
  const auto &forBody =
      std::get<std::vector<parser::NodePtr>>(forBodyField->value);
  if (!forTarget)
    return reject("malformed for statement over a generator expression");

  // Innermost statements: bind the loop variable to the element expression,
  // then run the original body.
  parser::NodePtr assign = parser::makeNode("Assign", statement.range);
  parser::addField(*assign, "targets", std::vector<parser::NodePtr>{forTarget});
  parser::addField(*assign, "value", elt);
  std::vector<parser::NodePtr> current{assign};
  current.insert(current.end(), forBody.begin(), forBody.end());

  for (const GenEntry &entry : llvm::reverse(chain)) {
    for (const parser::NodePtr &filter : llvm::reverse(entry.filters)) {
      parser::NodePtr guard = parser::makeNode("If", statement.range);
      parser::addField(*guard, "test", filter);
      parser::addField(*guard, "body", current);
      parser::addField(*guard, "orelse", std::vector<parser::NodePtr>{});
      current = {guard};
    }
    parser::NodePtr loop = parser::makeNode("For", statement.range);
    parser::addField(*loop, "target", entry.target);
    parser::addField(*loop, "iter", entry.iter);
    parser::addField(*loop, "body", current);
    parser::addField(*loop, "orelse", std::vector<parser::NodePtr>{});
    current = {loop};
  }

  llvm::SmallVector<std::pair<std::string, std::optional<Value>>, 2>
      priorTargets;
  for (const GenEntry &entry : chain) {
    std::optional<Value> prior;
    if (auto found = values.find(entry.targetName); found != values.end())
      prior = found->second;
    priorTargets.push_back({entry.targetName, prior});
  }
  emitFor(*current.front());
  for (auto &[name, prior] : priorTargets) {
    if (prior)
      values[name] = *prior;
    else
      values.erase(name);
  }
}

void ModuleEmitter::emitFor(const parser::Node &statement) {
  if (const auto *orelse = ast::nodeList(statement, "orelse")) {
    if (!orelse->empty()) {
      diagnostics.push_back(
          parser::Diagnostic{parser::Severity::Error, statement.range.start,
                             "for/else is not implemented yet"});
      return;
    }
  }
  if (const parser::Node *iterNode = ast::node(statement, "iter");
      iterNode && iterNode->kind == "GeneratorExp") {
    emitGeneratorExpFor(statement, *iterNode);
    return;
  }
  // A loop over an empty container literal statically runs zero iterations:
  // emit nothing (the body never executes; the target stays unbound, matching
  // CPython's observable behavior). This also covers the reducer desugars
  // over empty literals (any([]) / all([]) / max([]) -> ValueError guard).
  if (const parser::Node *iterNode = ast::node(statement, "iter")) {
    bool emptyLiteral = false;
    if (iterNode->kind == "List" || iterNode->kind == "Tuple") {
      const auto *elts = ast::nodeList(*iterNode, "elts");
      emptyLiteral = !elts || elts->empty();
    } else if (iterNode->kind == "Dict") {
      const auto *keys = ast::nodeList(*iterNode, "keys");
      emptyLiteral = !keys || keys->empty();
    }
    if (emptyLiteral)
      return;
  }
  // Loop-carried locals: pre-existing locals reassigned in the body (an
  // accumulator carried across iterations). Thread them as loop-header /
  // after-block arguments so the mutated value flows across the back-edge, and
  // release the replaced header value on the back-edge to keep ownership
  // balanced. Combining carried locals with break/continue is not yet
  // supported (break/continue would need to forward the carried values too).
  struct CarriedLocal {
    std::string name;
    mlir::Type type;
  };
  llvm::SmallVector<CarriedLocal, 4> carried;
  llvm::SmallVector<mlir::Value, 4> carriedInitial;
  {
    llvm::StringSet<> assignedInBody;
    collectAssignedNames(ast::nodeList(statement, "body"), assignedInBody);
    llvm::StringSet<> targetNames;
    collectAssignedNameTargets(ast::node(statement, "target"), targetNames);
    llvm::SmallVector<std::string, 4> names;
    for (const auto &assigned : assignedInBody)
      if (!targetNames.contains(assigned.getKey()) &&
          values.find(assigned.getKey()) != values.end())
        names.push_back(assigned.getKey().str());
    llvm::sort(names);
    for (const std::string &name : names) {
      Value initial = values.find(name)->second;
      mlir::Type carriedType = types.widenLiteral(initial.type);
      carried.push_back(CarriedLocal{name, carriedType});
      carriedInitial.push_back(coerceValue(initial, carriedType, statement).value);
    }
  }

  Value iterable = emitExpr(ast::node(statement, "iter"));
  CallInferenceResult iterInference =
      types.inferMethodCallWithEvidence(iterable.type, "__iter__", {});
  if (!requireStaticEvidence(statement, iterInference))
    return;
  mlir::Type iteratorType = iterInference.resultType;
  CallInferenceResult nextInference =
      types.inferMethodCallWithEvidence(iteratorType, "__next__", {});
  if (!requireStaticEvidence(statement, nextInference))
    return;
  mlir::Type elem = nextInference.resultType;
  mlir::UnitAttr returnedSelf =
      iteratorType == iterable.type ? builder.getUnitAttr() : mlir::UnitAttr();
  auto iterator = py::IterOp::create(builder, loc(statement), iteratorType,
                                     "__iter__", callProtocolFor(iterInference),
                                     iterable.value, returnedSelf);

  mlir::Block *entry = builder.getInsertionBlock();
  mlir::Region *region = entry->getParent();
  mlir::Block *afterBlock = entry->splitBlock(builder.getInsertionPoint());
  mlir::Block *checkBlock =
      builder.createBlock(region, afterBlock->getIterator());
  mlir::Block *bodyBlock =
      builder.createBlock(region, afterBlock->getIterator());
  for (const CarriedLocal &local : carried) {
    checkBlock->addArgument(local.type, loc(statement));
    afterBlock->addArgument(local.type, loc(statement));
  }

  builder.setInsertionPointToEnd(entry);
  mlir::cf::BranchOp::create(builder, loc(statement), checkBlock, carriedInitial);

  builder.setInsertionPointToStart(checkBlock);
  for (auto [index, local] : llvm::enumerate(carried)) {
    values[local.name] = Value{checkBlock->getArgument(index), local.type};
    types.bindSymbol(local.name, local.type);
  }
  llvm::SmallVector<mlir::Value, 4> checkArgs;
  for (unsigned index = 0; index < carried.size(); ++index)
    checkArgs.push_back(checkBlock->getArgument(index));
  auto next = py::NextOp::create(
      builder, loc(statement), elem, builder.getI1Type(), iteratorType,
      "__next__", callProtocolFor(nextInference), iterator.getResult());
  mlir::cf::CondBranchOp::create(builder, loc(statement), next.getValid(),
                                 bodyBlock, mlir::ValueRange{}, afterBlock,
                                 checkArgs);

  builder.setInsertionPointToStart(bodyBlock);
  {
    ScopedEmitterScope scope(values, types);
    for (auto [index, local] : llvm::enumerate(carried)) {
      values[local.name] = Value{checkBlock->getArgument(index), local.type};
      types.bindSymbol(local.name, local.type);
    }
    LoopControlContext loop{afterBlock, checkBlock};
    for (const CarriedLocal &local : carried)
      loop.carriedLocals.push_back(local.name);
    loop.headerBlock = checkBlock;
    loopControlContexts.push_back(loop);
    emitAssignTarget(*ast::node(statement, "target"),
                     Value{next.getElement(), elem});
    emitStatements(ast::nodeList(statement, "body"));
    loopControlContexts.pop_back();
    if (!insertionBlockTerminated(builder)) {
      llvm::SmallVector<mlir::Value, 4> nextCarried;
      for (auto [index, local] : llvm::enumerate(carried)) {
        auto found = values.find(local.name);
        mlir::Value value =
            coerceValue(found->second, local.type, statement).value;
        // Release the previous header value when the body replaced it so the
        // loop-carried ownership token stays balanced across the back-edge.
        mlir::Value previous = checkBlock->getArgument(index);
        if (value != previous &&
            !derivesViaStructuralMutation(value, previous))
          py::DecRefOp::create(builder, loc(statement), previous);
        nextCarried.push_back(value);
      }
      mlir::cf::BranchOp::create(builder, loc(statement), checkBlock,
                                 nextCarried);
    }
  }

  builder.setInsertionPointToStart(afterBlock);
  for (auto [index, local] : llvm::enumerate(carried)) {
    values[local.name] = Value{afterBlock->getArgument(index), local.type};
    types.bindSymbol(local.name, local.type);
  }
}

void ModuleEmitter::emitWhile(const parser::Node &statement) {
  if (const auto *orelse = ast::nodeList(statement, "orelse")) {
    if (!orelse->empty()) {
      diagnostics.push_back(
          parser::Diagnostic{parser::Severity::Error, statement.range.start,
                             "while/else is not implemented yet"});
      return;
    }
  }
  const parser::Node *test = ast::node(statement, "test");
  if (!test) {
    diagnostics.push_back(parser::Diagnostic{parser::Severity::Error,
                                             statement.range.start,
                                             "while requires a test expression"});
    return;
  }

  // Loop-carried locals: pre-existing locals assigned in the body (counters,
  // accumulators). Threaded as loop-header / after-block arguments, with the
  // replaced header value released on the back-edge. Combining with
  // break/continue is not yet supported.
  struct CarriedLocal {
    std::string name;
    mlir::Type type;
  };
  llvm::SmallVector<CarriedLocal, 4> carried;
  llvm::SmallVector<mlir::Value, 4> carriedInitial;
  {
    llvm::StringSet<> assignedInBody;
    collectAssignedNames(ast::nodeList(statement, "body"), assignedInBody);
    llvm::SmallVector<std::string, 4> names;
    for (const auto &assigned : assignedInBody)
      if (values.find(assigned.getKey()) != values.end())
        names.push_back(assigned.getKey().str());
    llvm::sort(names);
    for (const std::string &name : names) {
      Value initial = values.find(name)->second;
      mlir::Type carriedType = types.widenLiteral(initial.type);
      carried.push_back(CarriedLocal{name, carriedType});
      carriedInitial.push_back(
          coerceValue(initial, carriedType, statement).value);
    }
  }

  mlir::Block *entry = builder.getInsertionBlock();
  mlir::Region *region = entry->getParent();
  mlir::Block *afterBlock = entry->splitBlock(builder.getInsertionPoint());
  mlir::Block *headerBlock =
      builder.createBlock(region, afterBlock->getIterator());
  mlir::Block *bodyBlock =
      builder.createBlock(region, afterBlock->getIterator());
  for (const CarriedLocal &local : carried) {
    headerBlock->addArgument(local.type, loc(statement));
    afterBlock->addArgument(local.type, loc(statement));
  }

  auto bindCarried = [&](mlir::Block *block) {
    for (auto [index, local] : llvm::enumerate(carried)) {
      values[local.name] = Value{block->getArgument(index), local.type};
      types.bindSymbol(local.name, local.type);
    }
  };

  builder.setInsertionPointToEnd(entry);
  mlir::cf::BranchOp::create(builder, loc(statement), headerBlock,
                             carriedInitial);

  // Header: bind carried locals, evaluate the condition, and on false forward
  // the current header values to the after-block.
  builder.setInsertionPointToStart(headerBlock);
  bindCarried(headerBlock);
  llvm::SmallVector<mlir::Value, 4> headerArgs;
  for (unsigned index = 0; index < carried.size(); ++index)
    headerArgs.push_back(headerBlock->getArgument(index));
  mlir::Value condition = emitBoolValue(emitExpr(test), statement);
  mlir::cf::CondBranchOp::create(builder, loc(statement), condition, bodyBlock,
                                 mlir::ValueRange{}, afterBlock, headerArgs);

  builder.setInsertionPointToStart(bodyBlock);
  {
    ScopedEmitterScope scope(values, types);
    bindCarried(headerBlock);
    LoopControlContext loop{afterBlock, headerBlock};
    for (const CarriedLocal &local : carried)
      loop.carriedLocals.push_back(local.name);
    loop.headerBlock = headerBlock;
    loopControlContexts.push_back(loop);
    emitStatements(ast::nodeList(statement, "body"));
    loopControlContexts.pop_back();
    if (!insertionBlockTerminated(builder)) {
      llvm::SmallVector<mlir::Value, 4> nextCarried;
      for (auto [index, local] : llvm::enumerate(carried)) {
        auto found = values.find(local.name);
        mlir::Value value =
            coerceValue(found->second, local.type, statement).value;
        mlir::Value previous = headerBlock->getArgument(index);
        if (value != previous &&
            !derivesViaStructuralMutation(value, previous))
          py::DecRefOp::create(builder, loc(statement), previous);
        nextCarried.push_back(value);
      }
      mlir::cf::BranchOp::create(builder, loc(statement), headerBlock,
                                 nextCarried);
    }
  }

  builder.setInsertionPointToStart(afterBlock);
  bindCarried(afterBlock);
}

void ModuleEmitter::emitAsyncFor(const parser::Node &statement) {
  if (const auto *orelse = ast::nodeList(statement, "orelse")) {
    if (!orelse->empty()) {
      diagnostics.push_back(
          parser::Diagnostic{parser::Severity::Error, statement.range.start,
                             "async for/else is not implemented yet"});
      return;
    }
  }
  if (containsReturnStatement(ast::nodeList(statement, "body"))) {
    diagnostics.push_back(
        parser::Diagnostic{parser::Severity::Error, statement.range.start,
                           "return inside async for is not implemented yet"});
    return;
  }

  Value iterable = emitExpr(ast::node(statement, "iter"));
  Value concreteIterable = stripLocalProtocolView(iterable);
  Value methodIterable = concreteIterable.value ? concreteIterable : iterable;
  mlir::Type iteratorType;
  Value iteratorValue;
  Value sourceIteratorReceiver;
  std::optional<AsyncIterationInferenceResult> staticIteration;
  if (std::optional<MethodBinding> sourceAiter =
          lookupClassMethod(methodIterable.type, "__aiter__")) {
    if (sourceAiter->async) {
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, statement.range.start,
          "async __aiter__ methods are not supported; __aiter__ must return an "
          "AsyncIterator directly"});
      return;
    } else {
      iteratorValue =
          emitInlineMethodCall(statement, methodIterable, *sourceAiter);
      iteratorType = iteratorValue.type;
      if (lookupClassMethod(methodIterable.type, "__anext__"))
        sourceIteratorReceiver = methodIterable;
    }
  } else {
    AsyncIterationInferenceResult iterInference =
        types.inferAsyncIterationWithEvidence(iterable.type);
    if (!requireStaticEvidence(statement, iterInference))
      return;
    iteratorType = iterInference.iteratorType;
    mlir::UnitAttr returnedSelf = iteratorType == iterable.type
                                      ? builder.getUnitAttr()
                                      : mlir::UnitAttr();
    auto iterator = py::AIterOp::create(
        builder, loc(statement), iteratorType, "__aiter__",
        callProtocolFor(iterInference.aiter), iterable.value, returnedSelf);
    iteratorValue = Value{iterator.getResult(), iteratorType};
    staticIteration = iterInference;
  }

  Value sourceAnextReceiver =
      sourceIteratorReceiver.value ? sourceIteratorReceiver : iteratorValue;
  std::optional<MethodBinding> sourceAnextMethod;
  if (sourceAnextReceiver.value)
    if (std::optional<MethodBinding> method =
            lookupClassMethod(sourceAnextReceiver.type, "__anext__")) {
      sourceAnextMethod = *method;
    }

  mlir::Block *entryBlock = builder.getInsertionBlock();
  mlir::Region *region = entryBlock ? entryBlock->getParent() : nullptr;
  if (!region) {
    diagnostics.push_back(
        parser::Diagnostic{parser::Severity::Error, statement.range.start,
                           "async for requires an active insertion region"});
    return;
  }

  struct CarriedLocal {
    std::string name;
    mlir::Type type;
    mlir::Value initial;
  };
  llvm::StringSet<> assignedInBody;
  collectAssignedNames(ast::nodeList(statement, "body"), assignedInBody);
  llvm::SmallVector<CarriedLocal, 4> carriedLocals;
  llvm::SmallVector<mlir::Value, 4> carriedInitialValues;
  for (const auto &entry : values) {
    if (!assignedInBody.contains(entry.getKey()))
      continue;
    Value initial = entry.getValue();
    if (!initial.value)
      continue;
    mlir::Type carriedType = types.widenLiteral(initial.type);
    Value coerced = coerceValue(initial, carriedType, statement);
    carriedLocals.push_back(
        CarriedLocal{entry.getKey().str(), carriedType, coerced.value});
    carriedInitialValues.push_back(coerced.value);
  }

  mlir::Block *afterBlock = entryBlock->splitBlock(builder.getInsertionPoint());
  mlir::Block *loopBlock =
      builder.createBlock(region, afterBlock->getIterator());
  for (CarriedLocal &local : carriedLocals) {
    loopBlock->addArgument(local.type, loc(statement));
    afterBlock->addArgument(local.type, loc(statement));
  }
  builder.setInsertionPointToEnd(entryBlock);
  mlir::cf::BranchOp::create(builder, loc(statement), loopBlock,
                             carriedInitialValues);
  builder.setInsertionPointToStart(loopBlock);
  for (auto [index, local] : llvm::enumerate(carriedLocals)) {
    Value loopValue{loopBlock->getArgument(index), local.type};
    values[local.name] = loopValue;
    types.bindSymbol(local.name, local.type);
  }

  mlir::OperationState tryState(loc(statement), py::TryOp::getOperationName());
  tryState.addTypes(builder.getI1Type());
  for (const CarriedLocal &local : carriedLocals)
    tryState.addTypes(local.type);
  tryState.addRegion();
  tryState.addRegion();
  tryState.addRegion();
  auto tryOp = mlir::cast<py::TryOp>(builder.create(tryState));

  mlir::Block *tryBlock = new mlir::Block;
  tryOp.getTryRegion().push_back(tryBlock);
  builder.setInsertionPointToStart(tryBlock);
  mlir::Type awaitableType;
  Value awaitable;
  if (sourceAnextMethod) {
    if (sourceAnextMethod->async) {
      if (sourceAnextMethod->symbolName.empty()) {
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, statement.range.start,
            "async __anext__ method has no lowered callable symbol"});
        awaitable = emitNone(statement);
      } else {
        Value sourceAnextCallable = emitMethodObject(
            statement, sourceAnextReceiver, *sourceAnextMethod);
        awaitable = emitCallableDispatch(
            statement, sourceAnextCallable,
            emitCallOperands(statement, {}, /*includeAstArguments=*/false));
      }
    } else {
      awaitable = emitInlineMethodCall(statement, sourceAnextReceiver,
                                       *sourceAnextMethod);
    }
    awaitableType = awaitable.type;
  } else if (staticIteration) {
    awaitableType = staticIteration->nextAwaitableType;
    auto next = py::ANextOp::create(
        builder, loc(statement), awaitableType, "__anext__",
        callProtocolFor(staticIteration->anext), iteratorValue.value);
    awaitable = Value{next.getAwaitable(), awaitableType};
  } else {
    CallInferenceResult nextInference =
        types.inferMethodCallWithEvidence(iteratorType, "__anext__", {});
    if (!requireStaticEvidence(statement, nextInference))
      return;
    if (nextInference)
      awaitableType = nextInference.resultType;
    auto next = py::ANextOp::create(builder, loc(statement), awaitableType,
                                    "__anext__", callProtocolFor(nextInference),
                                    iteratorValue.value);
    awaitable = Value{next.getAwaitable(), awaitableType};
  }
  Value item = staticIteration ? emitAwaitValue(statement, awaitable,
                                                staticIteration->awaitNext)
                               : emitAwaitValue(statement, awaitable);
  {
    ScopedEmitterScope scope(values, types);
    emitAssignTarget(*ast::node(statement, "target"), item);
    emitStatements(ast::nodeList(statement, "body"));
    if (!blockHasTerminator(*tryBlock)) {
      llvm::SmallVector<mlir::Value, 4> yieldValues;
      yieldValues.push_back(
          mlir::arith::ConstantIntOp::create(builder, loc(statement), 1, 1));
      for (auto [index, local] : llvm::enumerate(carriedLocals)) {
        auto found = values.find(local.name);
        if (found == values.end()) {
          diagnostics.push_back(parser::Diagnostic{
              parser::Severity::Error, statement.range.start,
              "async for lost loop-carried local '" + local.name + "'"});
          continue;
        }
        Value carried = found->second;
        mlir::Value previous = loopBlock->getArgument(index);
        if (carried.value && carried.value != previous &&
            !derivesViaStructuralMutation(carried.value, previous))
          py::DecRefOp::create(builder, loc(statement), previous);
        yieldValues.push_back(
            coerceValue(carried, local.type, statement).value);
      }
      py::TryYieldOp::create(builder, loc(statement), yieldValues);
    }
  }

  mlir::Block *checkBlock = new mlir::Block;
  mlir::Block *stopBlock = new mlir::Block;
  mlir::Block *rethrowBlock = new mlir::Block;
  tryOp.getExceptRegion().push_back(checkBlock);
  tryOp.getExceptRegion().push_back(stopBlock);
  tryOp.getExceptRegion().push_back(rethrowBlock);

  builder.setInsertionPointToStart(checkBlock);
  mlir::Type stopAsyncIteration =
      types.typeObject(types.contract("builtins.StopAsyncIteration"));
  mlir::OperationState matchState(loc(statement),
                                  py::ExceptCurrentMatchOp::getOperationName());
  matchState.addTypes(builder.getI1Type());
  matchState.addAttribute("handler", mlir::TypeAttr::get(stopAsyncIteration));
  auto match = mlir::cast<py::ExceptCurrentMatchOp>(builder.create(matchState));
  mlir::cf::CondBranchOp::create(builder, loc(statement), match.getResult(),
                                 stopBlock, mlir::ValueRange{}, rethrowBlock,
                                 mlir::ValueRange{});

  builder.setInsertionPointToStart(stopBlock);
  llvm::SmallVector<mlir::Value, 4> stopValues;
  stopValues.push_back(
      mlir::arith::ConstantIntOp::create(builder, loc(statement), 0, 1));
  for (auto [index, local] : llvm::enumerate(carriedLocals))
    stopValues.push_back(loopBlock->getArgument(index));
  py::ExceptYieldOp::create(builder, loc(statement), stopValues);

  builder.setInsertionPointToStart(rethrowBlock);
  py::RaiseCurrentOp::create(builder, loc(statement));

  builder.setInsertionPointAfter(tryOp);
  mlir::Value keepGoing = tryOp.getResult(0);
  llvm::SmallVector<mlir::Value, 4> nextCarriedValues;
  nextCarriedValues.reserve(carriedLocals.size());
  for (auto [index, local] : llvm::enumerate(carriedLocals)) {
    mlir::Value result = tryOp.getResult(static_cast<unsigned>(index) + 1);
    nextCarriedValues.push_back(result);
    values[local.name] = Value{result, local.type};
    types.bindSymbol(local.name, local.type);
  }
  mlir::cf::CondBranchOp::create(builder, loc(statement), keepGoing, loopBlock,
                                 nextCarriedValues, afterBlock,
                                 nextCarriedValues);

  builder.setInsertionPointToStart(afterBlock);
  for (auto [index, local] : llvm::enumerate(carriedLocals)) {
    values[local.name] = Value{afterBlock->getArgument(index), local.type};
    types.bindSymbol(local.name, local.type);
  }
}

void ModuleEmitter::emitTry(const parser::Node &statement) {
  const auto *handlers = ast::nodeList(statement, "handlers");
  const auto *finalbody = ast::nodeList(statement, "finalbody");
  bool hasFinally = finalbody && !finalbody->empty();
  bool tryBodyHasReturn =
      containsReturnStatement(ast::nodeList(statement, "body"));
  bool tryBodyHasLoopControl =
      containsBreakOrContinueStatement(ast::nodeList(statement, "body"));
  bool finalbodyHasReturn = hasFinally && containsReturnStatement(finalbody);
  bool finalbodyHasLoopControl =
      hasFinally && containsBreakOrContinueStatement(finalbody);
  bool handlerBodyHasReturn = false;
  bool handlerBodyHasLoopControl = false;
  if (const auto *handlersForReturn = ast::nodeList(statement, "handlers")) {
    for (const parser::NodePtr &handler : *handlersForReturn) {
      handlerBodyHasReturn =
          handlerBodyHasReturn ||
          (handler && containsReturnStatement(ast::nodeList(*handler, "body")));
      handlerBodyHasLoopControl =
          handlerBodyHasLoopControl ||
          (handler &&
           containsBreakOrContinueStatement(ast::nodeList(*handler, "body")));
    }
  }
  bool protectedBodyHasReturn = tryBodyHasReturn || handlerBodyHasReturn;
  // The completion machinery (flag results + carried return payload on the
  // py.try op) works both with a finally region and with plain try/except:
  // without a finally the flags simply dispatch right after the op.
  const auto *orelseForEligibility = ast::nodeList(statement, "orelse");
  const auto *handlersForEligibility = ast::nodeList(statement, "handlers");
  bool completionEligible =
      (hasFinally ||
       (handlersForEligibility && !handlersForEligibility->empty())) &&
      (!orelseForEligibility || orelseForEligibility->empty());
  bool supportsNoneReturnThroughFinally =
      completionEligible && currentReturnType == types.none() &&
      (protectedBodyHasReturn || finalbodyHasReturn);
  bool supportsValueReturnThroughFinally =
      completionEligible && currentReturnType != types.none() &&
      protectedBodyHasReturn &&
      isSupportedFinallyReturnCarrierType(currentReturnType);
  bool supportsReturnThroughFinally =
      supportsNoneReturnThroughFinally || supportsValueReturnThroughFinally;
  bool supportsLoopControlThroughFinally =
      completionEligible && !loopControlContexts.empty() &&
      (tryBodyHasLoopControl || handlerBodyHasLoopControl ||
       finalbodyHasLoopControl);
  bool usesFinallyCompletion =
      supportsReturnThroughFinally || supportsLoopControlThroughFinally;
  if ((!handlers || handlers->empty()) && !hasFinally) {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, statement.range.start,
        "try without except or finally is not implemented yet"});
    return;
  }
  const auto *orelse = ast::nodeList(statement, "orelse");
  bool hasElse = orelse && !orelse->empty();
  // Locals assigned in the try body and visible in the else block: the else
  // runs only on normal completion, so try-body bindings are guaranteed
  // there. They travel as extra py.try results (yielded by the try region;
  // the except region yields inert defaults nobody reads). Restricted to the
  // scalar carrier contracts the yield machinery supports.
  struct ElseCarriedLocal {
    std::string name;
    mlir::Value value;
    mlir::Type type;
  };
  llvm::SmallVector<ElseCarriedLocal, 4> elseCarriedLocals;
  // Post-try visibility (plain try/except): locals bound at the END of the
  // try body AND at the end of every falling-through handler become extra
  // py.try results -- the try region yields its end-of-body values, each
  // handler yields its own end-of-handler values, and the statement's
  // continuation binds the merged lanes. Same scalar carrier restriction as
  // the else lanes.
  bool postTryEligible = false;
  llvm::SmallVector<std::string, 8> postCandidateNames;
  llvm::StringMap<Value> postTryEndBindings;
  mlir::Block *postTryFallThrough = nullptr;
  struct HandlerExit {
    mlir::Block *block = nullptr;
    llvm::StringMap<Value> bindings;
  };
  llvm::SmallVector<HandlerExit, 4> postHandlerExits;
  struct PostCarriedLocal {
    std::string name;
    mlir::Type type;
  };
  llvm::SmallVector<PostCarriedLocal, 4> postCarriedLocals;
  if (hasElse && hasFinally) {
    diagnostics.push_back(
        parser::Diagnostic{parser::Severity::Error, statement.range.start,
                           "try/else/finally is not implemented yet"});
    return;
  }
  if (tryBodyHasReturn && !supportsReturnThroughFinally) {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, statement.range.start,
        hasFinally ? "return value type through try/finally is "
                     "not implemented yet"
                   : "return inside try is not implemented yet"});
    return;
  }
  if (const auto *handlersForReturn = ast::nodeList(statement, "handlers")) {
    for (const parser::NodePtr &handler : *handlersForReturn) {
      if (handler && containsReturnStatement(ast::nodeList(*handler, "body")) &&
          !supportsReturnThroughFinally) {
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, handler->range.start,
            hasFinally
                ? "return value type through except/finally is not "
                  "implemented yet"
                : "return inside except handler is not implemented yet"});
        return;
      }
    }
  }
  if (finalbodyHasReturn && currentReturnType != types.none()) {
    diagnostics.push_back(
        parser::Diagnostic{parser::Severity::Error, statement.range.start,
                           "value-carrying return inside finally is not "
                           "implemented yet"});
    return;
  }
  if (finalbodyHasLoopControl && loopControlContexts.empty()) {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, statement.range.start,
        "break/continue inside finally requires an enclosing supported loop"});
    return;
  }
  if (finalbodyHasLoopControl && supportsValueReturnThroughFinally) {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, statement.range.start,
        "break/continue inside finally overriding a value-carrying return is "
        "not implemented yet"});
    return;
  }
  if ((tryBodyHasLoopControl || handlerBodyHasLoopControl) &&
      !supportsLoopControlThroughFinally) {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, statement.range.start,
        hasFinally ? "break/continue through try/finally requires an enclosing "
                     "supported loop"
                   : "break/continue inside try is not implemented yet"});
    return;
  }

  postTryEligible = !hasElse && !hasFinally && !usesFinallyCompletion &&
                    handlers && !handlers->empty();
  if (postTryEligible) {
    llvm::StringSet<> assignedNames;
    collectAssignedNames(ast::nodeList(statement, "body"), assignedNames);
    for (const parser::NodePtr &handler : *handlers)
      if (handler)
        collectAssignedNames(ast::nodeList(*handler, "body"), assignedNames);
    for (const auto &entry : assignedNames)
      postCandidateNames.push_back(entry.getKey().str());
    llvm::sort(postCandidateNames);
    if (postCandidateNames.empty())
      postTryEligible = false;
  }

  mlir::OperationState state(loc(statement), py::TryOp::getOperationName());
  if (hasElse)
    state.addTypes(builder.getI1Type());
  else if (usesFinallyCompletion) {
    state.addTypes(builder.getI1Type());
    state.addTypes(builder.getI1Type());
    state.addTypes(builder.getI1Type());
    if (supportsValueReturnThroughFinally)
      state.addTypes(currentReturnType);
  }
  state.addRegion();
  state.addRegion();
  state.addRegion();
  mlir::Operation *rawTry = builder.create(state);
  auto tryOp = mlir::cast<py::TryOp>(rawTry);

  auto appendBoolYield = [&](llvm::SmallVectorImpl<mlir::Value> &yieldValues,
                             bool value) {
    yieldValues.push_back(mlir::arith::ConstantIntOp::create(
        builder, loc(statement), value ? 1 : 0, 1));
  };
  auto emitDefaultReturnValue = [&](mlir::Type target) -> Value {
    if (auto literal = mlir::dyn_cast<py::LiteralType>(target)) {
      llvm::StringRef spelling = literal.getSpelling();
      if (spelling == "None") {
        auto op = py::NoneOp::create(builder, loc(statement), target);
        return {op.getResult(), target};
      }
      if (spelling == "True" || spelling == "False") {
        auto op =
            py::BoolConstantOp::create(builder, loc(statement), target,
                                       builder.getBoolAttr(spelling == "True"));
        return {op.getResult(), target};
      }
      if (isIntegerLiteralSpelling(spelling)) {
        auto op = py::IntConstantOp::create(builder, loc(statement), target,
                                            builder.getStringAttr(spelling));
        return {op.getResult(), target};
      }
      if (spelling.size() >= 2 && spelling.front() == '"' &&
          spelling.back() == '"') {
        auto op = py::StrConstantOp::create(
            builder, loc(statement), target,
            builder.getStringAttr(spelling.drop_front().drop_back()));
        return {op.getResult(), target};
      }
    }
    if (auto contract = mlir::dyn_cast<py::ContractType>(target)) {
      llvm::StringRef name = contract.getContractName();
      if (name == "types.NoneType" || name == "builtins.object") {
        Value value = emitNone(statement);
        return coerceValue(value, target, statement);
      }
      if (name == "builtins.bool") {
        mlir::Type literalType = types.literal("False");
        Value value{py::BoolConstantOp::create(builder, loc(statement),
                                               literalType,
                                               builder.getBoolAttr(false))
                        .getResult(),
                    literalType};
        return coerceValue(value, target, statement);
      }
      if (name == "builtins.int") {
        mlir::Type literalType = types.literal("0");
        Value value{py::IntConstantOp::create(builder, loc(statement),
                                              literalType,
                                              builder.getStringAttr("0"))
                        .getResult(),
                    literalType};
        return coerceValue(value, target, statement);
      }
      if (name == "builtins.float") {
        auto op = py::FloatConstantOp::create(builder, loc(statement), target,
                                              builder.getF64FloatAttr(0.0));
        return {op.getResult(), target};
      }
      if (name == "builtins.str") {
        mlir::Type literalType = types.literal("\"\"");
        Value value{py::StrConstantOp::create(builder, loc(statement),
                                              literalType,
                                              builder.getStringAttr(""))
                        .getResult(),
                    literalType};
        return coerceValue(value, target, statement);
      }
    }
    return emitNone(statement);
  };
  auto appendFallthroughReturnPayload =
      [&](llvm::SmallVectorImpl<mlir::Value> &yieldValues) {
        appendBoolYield(yieldValues, false);
        appendBoolYield(yieldValues, false);
        appendBoolYield(yieldValues, false);
        if (supportsValueReturnThroughFinally)
          yieldValues.push_back(emitDefaultReturnValue(currentReturnType).value);
      };
  auto appendCompletionYield =
      [&](llvm::SmallVectorImpl<mlir::Value> &yieldValues,
          FinallyCompletion completion) {
        appendBoolYield(yieldValues, completion == FinallyCompletion::Return);
        appendBoolYield(yieldValues, completion == FinallyCompletion::Break);
        appendBoolYield(yieldValues, completion == FinallyCompletion::Continue);
        if (supportsValueReturnThroughFinally) {
          if (completion == FinallyCompletion::Return)
            return;
          yieldValues.push_back(emitDefaultReturnValue(currentReturnType).value);
        }
      };

  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    auto *tryBlock = new mlir::Block;
    tryOp.getTryRegion().push_back(tryBlock);
    mlir::Block *tryReturnBlock = nullptr;
    mlir::Block *tryBreakBlock = nullptr;
    mlir::Block *tryContinueBlock = nullptr;
    if (supportsReturnThroughFinally && tryBodyHasReturn) {
      tryReturnBlock = new mlir::Block;
      if (supportsValueReturnThroughFinally)
        tryReturnBlock->addArgument(currentReturnType, loc(statement));
      tryOp.getTryRegion().push_back(tryReturnBlock);
    }
    if (supportsLoopControlThroughFinally) {
      tryBreakBlock = new mlir::Block;
      tryContinueBlock = new mlir::Block;
      tryOp.getTryRegion().push_back(tryBreakBlock);
      tryOp.getTryRegion().push_back(tryContinueBlock);
    }
    builder.setInsertionPointToStart(tryBlock);
    {
      ScopedEmitterScope scope(values, types);
      if (tryReturnBlock)
        inlineReturnContexts.push_back(
            InlineReturnContext{tryReturnBlock, currentReturnType,
                                supportsValueReturnThroughFinally});
      if (supportsLoopControlThroughFinally)
        loopControlContexts.push_back(
            LoopControlContext{tryBreakBlock, tryContinueBlock});
      emitStatements(ast::nodeList(statement, "body"));
      if (supportsLoopControlThroughFinally)
        loopControlContexts.pop_back();
      if (tryReturnBlock)
        inlineReturnContexts.pop_back();
      if (postTryEligible) {
        mlir::Block *fallThrough = builder.getInsertionBlock();
        unsigned openCount = 0;
        for (mlir::Block &block : tryOp.getTryRegion())
          if (!blockHasTerminator(block))
            ++openCount;
        if (fallThrough && !blockHasTerminator(*fallThrough) &&
            openCount == 1) {
          postTryFallThrough = fallThrough;
          for (const std::string &name : postCandidateNames) {
            auto found = values.find(name);
            if (found != values.end() && found->second.value)
              postTryEndBindings[name] = found->second;
          }
        } else if (openCount != 0) {
          postTryEligible = false; // multi-exit try body: lanes would not
                                   // dominate every yield
        }
      }
      if (hasElse) {
        mlir::Block *fallThrough = builder.getInsertionBlock();
        unsigned openCount = 0;
        for (mlir::Block &block : tryOp.getTryRegion())
          if (!blockHasTerminator(block))
            ++openCount;
        // The carried values must dominate the fall-through yield; bail out
        // of carrying when the region shape leaves more than that one block
        // open (each open block receives the same yield operands).
        if (fallThrough && !blockHasTerminator(*fallThrough) &&
            openCount == 1) {
          llvm::StringSet<> assignedInTry;
          collectAssignedNames(ast::nodeList(statement, "body"),
                               assignedInTry);
          llvm::SmallVector<llvm::StringRef, 8> orderedNames;
          for (const auto &entry : assignedInTry)
            orderedNames.push_back(entry.getKey());
          llvm::sort(orderedNames);
          for (llvm::StringRef name : orderedNames) {
            auto found = values.find(std::string(name));
            if (found == values.end() || !found->second.value)
              continue;
            mlir::Region *definedIn = found->second.value.getParentRegion();
            if (!definedIn || !tryOp.getTryRegion().isAncestor(definedIn))
              continue;
            mlir::Type carried = types.widenLiteral(found->second.type);
            auto contract =
                mlir::dyn_cast_if_present<py::ContractType>(carried);
            if (!contract)
              continue;
            llvm::StringRef contractName = contract.getContractName();
            if (contractName != "builtins.int" &&
                contractName != "builtins.str" &&
                contractName != "builtins.bool" &&
                contractName != "builtins.float")
              continue;
            Value coerced = coerceValue(found->second, carried, statement);
            elseCarriedLocals.push_back(
                ElseCarriedLocal{std::string(name), coerced.value, carried});
          }
        }
      }
    }
    if (tryReturnBlock) {
      builder.setInsertionPointToStart(tryReturnBlock);
      llvm::SmallVector<mlir::Value, 2> yieldValues;
      appendCompletionYield(yieldValues, FinallyCompletion::Return);
      if (supportsValueReturnThroughFinally)
        yieldValues.push_back(tryReturnBlock->getArgument(0));
      py::TryYieldOp::create(builder, loc(statement), yieldValues);
    }
    if (tryBreakBlock) {
      builder.setInsertionPointToStart(tryBreakBlock);
      llvm::SmallVector<mlir::Value, 4> yieldValues;
      appendCompletionYield(yieldValues, FinallyCompletion::Break);
      py::TryYieldOp::create(builder, loc(statement), yieldValues);
    }
    if (tryContinueBlock) {
      builder.setInsertionPointToStart(tryContinueBlock);
      llvm::SmallVector<mlir::Value, 4> yieldValues;
      appendCompletionYield(yieldValues, FinallyCompletion::Continue);
      py::TryYieldOp::create(builder, loc(statement), yieldValues);
    }
    if (!postTryEligible) {
      bool tryCanFallThrough =
          terminateOpenRegionBlocks<py::TryYieldOp>(
              builder, loc(statement), tryOp.getTryRegion(),
              [&](llvm::SmallVectorImpl<mlir::Value> &yieldValues) {
                if (hasElse) {
                  appendBoolYield(yieldValues, true);
                  for (const ElseCarriedLocal &local : elseCarriedLocals)
                    yieldValues.push_back(local.value);
                } else if (usesFinallyCompletion)
                  appendFallthroughReturnPayload(yieldValues);
              }) > 0;
      tryOp->setAttr("ly.try.source_can_fallthrough",
                     builder.getBoolAttr(tryCanFallThrough));
    }
    // postTryEligible: the try region terminates AFTER the handlers are
    // emitted, once the post-try lanes are known.
  }

  bool exceptCanFallThrough = false;
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    llvm::SmallVector<mlir::Block *, 8> checkBlocks;
    llvm::SmallVector<mlir::Block *, 8> bodyBlocks;
    if (handlers) {
      checkBlocks.reserve(handlers->size());
      bodyBlocks.reserve(handlers->size());
      for (std::size_t index = 0; index < handlers->size(); ++index) {
        checkBlocks.push_back(new mlir::Block);
        bodyBlocks.push_back(new mlir::Block);
        tryOp.getExceptRegion().push_back(checkBlocks.back());
        tryOp.getExceptRegion().push_back(bodyBlocks.back());
      }
    }
    mlir::Block *rethrowBlock = nullptr;
    if (handlers && !handlers->empty()) {
      rethrowBlock = new mlir::Block;
      tryOp.getExceptRegion().push_back(rethrowBlock);
    }
    mlir::Block *exceptReturnBlock = nullptr;
    mlir::Block *exceptBreakBlock = nullptr;
    mlir::Block *exceptContinueBlock = nullptr;
    if (supportsReturnThroughFinally && handlerBodyHasReturn && handlers &&
        !handlers->empty()) {
      exceptReturnBlock = new mlir::Block;
      if (supportsValueReturnThroughFinally)
        exceptReturnBlock->addArgument(currentReturnType, loc(statement));
      tryOp.getExceptRegion().push_back(exceptReturnBlock);
    }
    if (supportsLoopControlThroughFinally && handlers && !handlers->empty()) {
      exceptBreakBlock = new mlir::Block;
      exceptContinueBlock = new mlir::Block;
      tryOp.getExceptRegion().push_back(exceptBreakBlock);
      tryOp.getExceptRegion().push_back(exceptContinueBlock);
    }

    if (handlers) {
      for (auto [index, handlerPtr] : llvm::enumerate(*handlers)) {
        const parser::Node &handler = *handlerPtr;
        std::optional<std::string_view> handlerName =
            ast::string(handler, "name");

        const parser::Node *typeNode = ast::node(handler, "type");
        if (!typeNode && index + 1 != handlers->size()) {
          diagnostics.push_back(
              parser::Diagnostic{parser::Severity::Error, handler.range.start,
                                 "bare except must be the last handler"});
          continue;
        }

        llvm::SmallVector<mlir::Type, 4> handlerTypes;
        llvm::SmallVector<mlir::Location, 4> handlerTypeLocs;
        if (!typeNode) {
          handlerTypes.push_back(
              types.typeObject(types.contract("builtins.BaseException")));
          handlerTypeLocs.push_back(loc(handler));
        } else {
          llvm::SmallVector<const parser::Node *, 4> candidateTypes;
          if (typeNode->kind == "Tuple") {
            if (const auto *elts = ast::nodeList(*typeNode, "elts"))
              for (const parser::NodePtr &elt : *elts)
                if (elt)
                  candidateTypes.push_back(elt.get());
          } else {
            candidateTypes.push_back(typeNode);
          }

          for (const parser::Node *candidate : candidateTypes) {
            mlir::Type candidateType = types.inferExpr(candidate);
            if (!mlir::isa_and_nonnull<py::TypeType>(candidateType)) {
              diagnostics.push_back(parser::Diagnostic{
                  parser::Severity::Error,
                  candidate ? candidate->range.start : handler.range.start,
                  "except handler must resolve to a Python type object"});
              handlerTypes.clear();
              handlerTypeLocs.clear();
              break;
            }
            handlerTypes.push_back(candidateType);
            handlerTypeLocs.push_back(loc(*candidate));
          }
        }
        if (handlerTypes.empty())
          continue;
        if (handlerName && handlerTypes.size() != 1) {
          diagnostics.push_back(parser::Diagnostic{
              parser::Severity::Error, handler.range.start,
              "except-as binding for tuple handlers is not implemented yet"});
          continue;
        }

        mlir::Block *miss = index + 1 == handlers->size()
                                ? rethrowBlock
                                : checkBlocks[index + 1];
        mlir::Block *currentCheck = checkBlocks[index];
        for (auto [matchIndex, handlerType] : llvm::enumerate(handlerTypes)) {
          builder.setInsertionPointToStart(currentCheck);
          mlir::Location matchLoc = handlerTypeLocs[matchIndex];
          mlir::OperationState matchState(
              matchLoc, py::ExceptCurrentMatchOp::getOperationName());
          matchState.addTypes(builder.getI1Type());
          matchState.addAttribute("handler", mlir::TypeAttr::get(handlerType));
          auto match =
              mlir::cast<py::ExceptCurrentMatchOp>(builder.create(matchState));
          mlir::Block *nextMiss = miss;
          if (matchIndex + 1 != handlerTypes.size()) {
            nextMiss = new mlir::Block;
            tryOp.getExceptRegion().push_back(nextMiss);
          }
          mlir::cf::CondBranchOp::create(builder, matchLoc, match.getResult(),
                                         bodyBlocks[index], mlir::ValueRange{},
                                         nextMiss, mlir::ValueRange{});
          currentCheck = nextMiss;
        }

        builder.setInsertionPointToStart(bodyBlocks[index]);
        {
          ScopedEmitterScope scope(values, types);
          if (handlerName) {
            auto handlerType = mlir::cast<py::TypeType>(handlerTypes.front());
            mlir::Type exceptionType = handlerType.getInstanceType();
            auto current = py::ExceptCurrentValueOp::create(
                               builder, loc(handler), exceptionType,
                               mlir::TypeAttr::get(handlerTypes.front()))
                               .getResult();
            std::string name(*handlerName);
            values[name] = Value{current, exceptionType};
            types.bindSymbol(name, exceptionType);
          }
          if (exceptReturnBlock)
            inlineReturnContexts.push_back(
                InlineReturnContext{exceptReturnBlock, currentReturnType,
                                    supportsValueReturnThroughFinally});
          if (supportsLoopControlThroughFinally)
            loopControlContexts.push_back(
                LoopControlContext{exceptBreakBlock, exceptContinueBlock});
          emitStatements(ast::nodeList(handler, "body"));
          if (supportsLoopControlThroughFinally)
            loopControlContexts.pop_back();
          if (exceptReturnBlock)
            inlineReturnContexts.pop_back();
          if (postTryEligible) {
            mlir::Block *exit = builder.getInsertionBlock();
            if (exit && !blockHasTerminator(*exit)) {
              HandlerExit record;
              record.block = exit;
              for (const std::string &name : postCandidateNames) {
                auto found = values.find(name);
                if (found != values.end() && found->second.value)
                  record.bindings[name] = found->second;
              }
              postHandlerExits.push_back(std::move(record));
            }
          }
        }
      }
    }

    if (rethrowBlock) {
      builder.setInsertionPointToStart(rethrowBlock);
      py::RaiseCurrentOp::create(builder, loc(statement));
      if (exceptReturnBlock) {
        builder.setInsertionPointToStart(exceptReturnBlock);
        llvm::SmallVector<mlir::Value, 2> yieldValues;
        appendCompletionYield(yieldValues, FinallyCompletion::Return);
        if (supportsValueReturnThroughFinally)
          yieldValues.push_back(exceptReturnBlock->getArgument(0));
        py::ExceptYieldOp::create(builder, loc(statement), yieldValues);
      }
      if (exceptBreakBlock) {
        builder.setInsertionPointToStart(exceptBreakBlock);
        llvm::SmallVector<mlir::Value, 4> yieldValues;
        appendCompletionYield(yieldValues, FinallyCompletion::Break);
        py::ExceptYieldOp::create(builder, loc(statement), yieldValues);
      }
      if (exceptContinueBlock) {
        builder.setInsertionPointToStart(exceptContinueBlock);
        llvm::SmallVector<mlir::Value, 4> yieldValues;
        appendCompletionYield(yieldValues, FinallyCompletion::Continue);
        py::ExceptYieldOp::create(builder, loc(statement), yieldValues);
      }
      if (postTryEligible) {
        // Every open except-region block must be a recorded handler exit so
        // its yield can carry that handler's bindings; anything else means a
        // shape the lanes cannot dominate -> fall back to laneless yields.
        for (mlir::Block &block : tryOp.getExceptRegion())
          if (!blockHasTerminator(block) &&
              llvm::none_of(postHandlerExits, [&](const HandlerExit &exit) {
                return exit.block == &block;
              })) {
            postTryEligible = false;
            break;
          }
      }
      if (postTryEligible) {
        // Lanes: bound at try end AND at every falling-through handler end,
        // all lanes carrier-typed; the lane type is the widened join.
        auto carrierType = [&](mlir::Type type) -> mlir::Type {
          mlir::Type widened = types.widenLiteral(type);
          auto contract = mlir::dyn_cast_if_present<py::ContractType>(widened);
          if (!contract)
            return {};
          llvm::StringRef name = contract.getContractName();
          if (name == "builtins.int" || name == "builtins.str" ||
              name == "builtins.bool" || name == "builtins.float")
            return widened;
          return {};
        };
        for (const std::string &name : postCandidateNames) {
          auto tryBound = postTryEndBindings.find(name);
          if (tryBound == postTryEndBindings.end())
            continue;
          llvm::SmallVector<mlir::Type, 4> parts;
          mlir::Type tryPart = carrierType(tryBound->second.type);
          if (!tryPart)
            continue;
          parts.push_back(tryPart);
          bool everywhere = true;
          for (const HandlerExit &exit : postHandlerExits) {
            auto found = exit.bindings.find(name);
            mlir::Type part = found != exit.bindings.end()
                                  ? carrierType(found->second.type)
                                  : mlir::Type();
            if (!part) {
              everywhere = false;
              break;
            }
            parts.push_back(part);
          }
          if (!everywhere)
            continue;
          mlir::Type merged = types.join(parts);
          if (!merged || !carrierType(merged))
            continue;
          postCarriedLocals.push_back(PostCarriedLocal{name, merged});
        }
      }
      if (postTryEligible) {
        // Per-handler yields carry that handler's own bindings.
        for (const HandlerExit &exit : postHandlerExits) {
          builder.setInsertionPointToEnd(exit.block);
          llvm::SmallVector<mlir::Value, 4> yieldValues;
          for (const PostCarriedLocal &local : postCarriedLocals) {
            Value bound = exit.bindings.lookup(local.name);
            yieldValues.push_back(
                coerceValue(bound, local.type, statement).value);
          }
          py::ExceptYieldOp::create(builder, loc(statement), yieldValues);
        }
        exceptCanFallThrough = !postHandlerExits.empty();
      } else {
        exceptCanFallThrough =
            terminateOpenRegionBlocks<py::ExceptYieldOp>(
                builder, loc(statement), tryOp.getExceptRegion(),
                [&](llvm::SmallVectorImpl<mlir::Value> &yieldValues) {
                  if (hasElse) {
                    appendBoolYield(yieldValues, false);
                    // Inert defaults: the else block (the only reader of the
                    // carried lanes) is unreachable on this path.
                    for (const ElseCarriedLocal &local : elseCarriedLocals)
                      yieldValues.push_back(
                          emitDefaultReturnValue(local.type).value);
                  } else if (usesFinallyCompletion)
                    appendFallthroughReturnPayload(yieldValues);
                }) > 0;
      }
    }
  }

  if (!hasElse && !usesFinallyCompletion && !hasFinally) {
    // Deferred try-region termination for the plain path: yield the post-try
    // lanes (or nothing when none survived).
    mlir::OpBuilder::InsertionGuard guard(builder);
    bool tryCanFallThrough = false;
    if (postTryFallThrough && !blockHasTerminator(*postTryFallThrough)) {
      builder.setInsertionPointToEnd(postTryFallThrough);
      llvm::SmallVector<mlir::Value, 4> yieldValues;
      for (const PostCarriedLocal &local : postCarriedLocals) {
        Value bound = postTryEndBindings.lookup(local.name);
        yieldValues.push_back(coerceValue(bound, local.type, statement).value);
      }
      py::TryYieldOp::create(builder, loc(statement), yieldValues);
      tryCanFallThrough = true;
    }
    tryCanFallThrough =
        terminateOpenRegionBlocks<py::TryYieldOp>(builder, loc(statement),
                                                  tryOp.getTryRegion()) > 0 ||
        tryCanFallThrough;
    if (!tryOp->hasAttr("ly.try.source_can_fallthrough"))
      tryOp->setAttr("ly.try.source_can_fallthrough",
                     builder.getBoolAttr(tryCanFallThrough));
  }

  if (hasFinally) {
    mlir::OpBuilder::InsertionGuard guard(builder);
    auto *finallyBlock = new mlir::Block;
    tryOp.getFinallyRegion().push_back(finallyBlock);
    mlir::Block *finallyReturnBlock = nullptr;
    mlir::Block *finallyBreakBlock = nullptr;
    mlir::Block *finallyContinueBlock = nullptr;
    if (supportsReturnThroughFinally && finalbodyHasReturn) {
      finallyReturnBlock = new mlir::Block;
      if (supportsValueReturnThroughFinally)
        finallyReturnBlock->addArgument(currentReturnType, loc(statement));
      tryOp.getFinallyRegion().push_back(finallyReturnBlock);
    }
    if (supportsLoopControlThroughFinally && finalbodyHasLoopControl) {
      finallyBreakBlock = new mlir::Block;
      finallyContinueBlock = new mlir::Block;
      tryOp.getFinallyRegion().push_back(finallyBreakBlock);
      tryOp.getFinallyRegion().push_back(finallyContinueBlock);
    }
    builder.setInsertionPointToStart(finallyBlock);
    {
      ScopedEmitterScope scope(values, types);
      if (finallyReturnBlock)
        inlineReturnContexts.push_back(
            InlineReturnContext{finallyReturnBlock, currentReturnType,
                                supportsValueReturnThroughFinally});
      if (finallyBreakBlock)
        loopControlContexts.push_back(
            LoopControlContext{finallyBreakBlock, finallyContinueBlock});
      emitStatements(finalbody);
      if (finallyBreakBlock)
        loopControlContexts.pop_back();
      if (finallyReturnBlock)
        inlineReturnContexts.pop_back();
    }
    if (finallyReturnBlock) {
      builder.setInsertionPointToStart(finallyReturnBlock);
      llvm::SmallVector<mlir::Value, 4> yieldValues;
      appendCompletionYield(yieldValues, FinallyCompletion::Return);
      if (supportsValueReturnThroughFinally)
        yieldValues.push_back(finallyReturnBlock->getArgument(0));
      py::FinallyYieldOp::create(builder, loc(statement), yieldValues);
    }
    if (finallyBreakBlock) {
      builder.setInsertionPointToStart(finallyBreakBlock);
      llvm::SmallVector<mlir::Value, 4> yieldValues;
      appendCompletionYield(yieldValues, FinallyCompletion::Break);
      py::FinallyYieldOp::create(builder, loc(statement), yieldValues);
    }
    if (finallyContinueBlock) {
      builder.setInsertionPointToStart(finallyContinueBlock);
      llvm::SmallVector<mlir::Value, 4> yieldValues;
      appendCompletionYield(yieldValues, FinallyCompletion::Continue);
      py::FinallyYieldOp::create(builder, loc(statement), yieldValues);
    }
    terminateOpenRegionBlocks<py::FinallyYieldOp>(builder, loc(statement),
                                                  tryOp.getFinallyRegion());
  }

  if (!postCarriedLocals.empty()) {
    // Recreate py.try with the post-try lane results (the lanes were
    // discovered while emitting the regions, after the op existed).
    mlir::OperationState widenedState(loc(statement),
                                      py::TryOp::getOperationName());
    for (const PostCarriedLocal &local : postCarriedLocals)
      widenedState.addTypes(local.type);
    widenedState.addRegion();
    widenedState.addRegion();
    widenedState.addRegion();
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(rawTry);
    mlir::Operation *widened = builder.create(widenedState);
    for (unsigned index = 0; index < 3; ++index)
      widened->getRegion(index).takeBody(rawTry->getRegion(index));
    widened->setAttrs(rawTry->getAttrDictionary());
    rawTry->erase();
    rawTry = widened;
    tryOp = mlir::cast<py::TryOp>(widened);
  }

  if (hasElse && !elseCarriedLocals.empty()) {
    // The carried locals were discovered while emitting the try body, after
    // the op was already created: recreate py.try with the extra result
    // lanes and move the regions over (the completion flag stays result 0).
    mlir::OperationState widenedState(loc(statement),
                                      py::TryOp::getOperationName());
    widenedState.addTypes(builder.getI1Type());
    for (const ElseCarriedLocal &local : elseCarriedLocals)
      widenedState.addTypes(local.type);
    widenedState.addRegion();
    widenedState.addRegion();
    widenedState.addRegion();
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(rawTry);
    mlir::Operation *widened = builder.create(widenedState);
    for (unsigned index = 0; index < 3; ++index)
      widened->getRegion(index).takeBody(rawTry->getRegion(index));
    widened->setAttrs(rawTry->getAttrDictionary());
    rawTry->erase();
    rawTry = widened;
    tryOp = mlir::cast<py::TryOp>(widened);
  }

  builder.setInsertionPointAfter(tryOp);
  for (auto [index, local] : llvm::enumerate(postCarriedLocals)) {
    values[local.name] =
        Value{tryOp.getResult(static_cast<unsigned>(index)), local.type};
    types.bindSymbol(local.name, local.type);
  }
  if (usesFinallyCompletion) {
    constexpr unsigned returnFlagIndex = 0;
    constexpr unsigned breakFlagIndex = 1;
    constexpr unsigned continueFlagIndex = 2;
    constexpr unsigned returnPayloadIndex = 3;
    auto emitReturnCompletion = [&]() {
      Value returned =
          supportsValueReturnThroughFinally
              ? Value{tryOp.getResult(returnPayloadIndex), currentReturnType}
              : emitNone(statement);
      if (!inlineReturnContexts.empty()) {
        InlineReturnContext &ctx = inlineReturnContexts.back();
        if (ctx.carryResult) {
          Value result = ctx.resultType
                             ? coerceValue(returned, ctx.resultType, statement)
                             : returned;
          mlir::cf::BranchOp::create(builder, loc(statement), ctx.target,
                                     result.value);
        } else {
          mlir::cf::BranchOp::create(builder, loc(statement), ctx.target);
        }
      } else {
        Value result = coerceValue(returned, currentReturnType, statement);
        mlir::func::ReturnOp::create(builder, loc(statement), result.value);
      }
    };
    auto discardInactiveReturnPayload = [&]() {
      if (supportsValueReturnThroughFinally &&
          mlir::isa<py::ContractType>(currentReturnType))
        py::DecRefOp::create(builder, loc(statement),
                             tryOp.getResult(returnPayloadIndex));
    };
    bool canFallThrough = false;
    if (auto attr = tryOp->getAttrOfType<mlir::BoolAttr>(
            "ly.try.source_can_fallthrough"))
      canFallThrough = attr.getValue();
    canFallThrough = canFallThrough || exceptCanFallThrough;
    mlir::Value returnFlag = tryOp.getResult(returnFlagIndex);
    mlir::Value breakFlag = tryOp.getResult(breakFlagIndex);
    mlir::Value continueFlag = tryOp.getResult(continueFlagIndex);
    if (!canFallThrough && supportsReturnThroughFinally &&
        !supportsLoopControlThroughFinally) {
      emitReturnCompletion();
      return;
    }

    mlir::Block *tryBlock = tryOp->getBlock();
    mlir::Block *afterCompletionCheck =
        tryBlock->splitBlock(builder.getInsertionPoint());
    mlir::Block *afterReturnCheck = tryBlock;
    builder.setInsertionPointToEnd(tryBlock);
    if (supportsReturnThroughFinally) {
      mlir::Block *returnBlock = new mlir::Block;
      afterReturnCheck = new mlir::Block;
      tryBlock->getParent()->getBlocks().insert(
          afterCompletionCheck->getIterator(), returnBlock);
      tryBlock->getParent()->getBlocks().insert(
          afterCompletionCheck->getIterator(), afterReturnCheck);
      mlir::cf::CondBranchOp::create(builder, loc(statement), returnFlag,
                                     returnBlock, mlir::ValueRange{},
                                     afterReturnCheck, mlir::ValueRange{});

      builder.setInsertionPointToStart(returnBlock);
      emitReturnCompletion();
      builder.setInsertionPointToStart(afterReturnCheck);
    }
    if (supportsLoopControlThroughFinally) {
      mlir::Block *breakBlock = new mlir::Block;
      mlir::Block *afterBreakCheck = new mlir::Block;
      mlir::Block *continueBlock = new mlir::Block;
      mlir::Block *afterContinueCheck = new mlir::Block;
      tryBlock->getParent()->getBlocks().insert(
          afterCompletionCheck->getIterator(), breakBlock);
      tryBlock->getParent()->getBlocks().insert(
          afterCompletionCheck->getIterator(), afterBreakCheck);
      tryBlock->getParent()->getBlocks().insert(
          afterCompletionCheck->getIterator(), continueBlock);
      tryBlock->getParent()->getBlocks().insert(
          afterCompletionCheck->getIterator(), afterContinueCheck);

      mlir::cf::CondBranchOp::create(builder, loc(statement), breakFlag,
                                     breakBlock, mlir::ValueRange{},
                                     afterBreakCheck, mlir::ValueRange{});

      builder.setInsertionPointToStart(breakBlock);
      discardInactiveReturnPayload();
      mlir::cf::BranchOp::create(builder, loc(statement),
                                 loopControlContexts.back().breakTarget);

      builder.setInsertionPointToStart(afterBreakCheck);
      mlir::cf::CondBranchOp::create(builder, loc(statement), continueFlag,
                                     continueBlock, mlir::ValueRange{},
                                     afterContinueCheck, mlir::ValueRange{});

      builder.setInsertionPointToStart(continueBlock);
      discardInactiveReturnPayload();
      mlir::cf::BranchOp::create(builder, loc(statement),
                                 loopControlContexts.back().continueTarget);

      builder.setInsertionPointToStart(afterContinueCheck);
    }
    discardInactiveReturnPayload();
    mlir::cf::BranchOp::create(builder, loc(statement), afterCompletionCheck);
    builder.setInsertionPointToStart(afterCompletionCheck);
  } else if (hasElse) {
    mlir::Block *tryBlock = tryOp->getBlock();
    mlir::Block *afterElseBlock =
        tryBlock->splitBlock(builder.getInsertionPoint());
    mlir::Block *elseBlock = new mlir::Block;
    tryBlock->getParent()->getBlocks().insert(afterElseBlock->getIterator(),
                                              elseBlock);
    builder.setInsertionPointToEnd(tryBlock);
    mlir::Value completedNormally = tryOp.getResult(0);
    mlir::cf::CondBranchOp::create(builder, loc(statement), completedNormally,
                                   elseBlock, mlir::ValueRange{},
                                   afterElseBlock, mlir::ValueRange{});

    builder.setInsertionPointToStart(elseBlock);
    {
      ScopedEmitterScope scope(values, types);
      for (auto [index, local] : llvm::enumerate(elseCarriedLocals)) {
        values[local.name] =
            Value{tryOp.getResult(1 + static_cast<unsigned>(index)),
                  local.type};
        types.bindSymbol(local.name, local.type);
      }
      emitStatements(orelse);
    }
    if (!blockHasTerminator(*elseBlock))
      mlir::cf::BranchOp::create(builder, loc(statement), afterElseBlock);
    builder.setInsertionPointToStart(afterElseBlock);
  }
}

} // namespace lython::emitter
