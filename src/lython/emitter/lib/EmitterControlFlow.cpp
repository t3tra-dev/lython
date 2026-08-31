#include "EmitterCore.h"
#include "EmitterPyOps.h"
#include "EmitterSupport.h"

#include "AstAccess.h"
#include "ClosureAnalysis.h"

#include "llvm/ADT/ScopeExit.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"

#include <cstddef>

namespace lython::emitter {

// The one place a proved fact becomes a narrower SSA value. `if`, the
// conditional expression and the `while` body all reach it: each of them
// used to be its own copy, and the loop had none at all.
void ModuleEmitter::applyBranchNarrowing(const parser::Node &anchor,
                                         const BranchTypeNarrowing &fact,
                                         bool conditionIsTrue) {
    if (std::optional<mlir::Type> before = types.lookupSymbol(fact.name))
      narrowedFromTypes[fact.name] = *before;
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
              builder, loc(anchor), narrowed, found->second.value);
          found->second.value = unwrap.getResult();
        } else if (auto narrowedUnion =
                       mlir::dyn_cast<py::UnionType>(narrowed);
                   narrowedUnion &&
                   llvm::all_of(narrowedUnion.getMemberTypes(),
                                [&](mlir::Type member) {
                                  return unionType.hasMember(member);
                                })) {
          // ⭐ ELIMINATION LEAVES A SUB-UNION, and that is not a MEMBER. Two
          // members narrow to one and land in the arm above; three narrow to
          // two, `hasMember` says no, and the narrowing silently did nothing:
          //
          //     def outer(value: "int | str | None") -> int:
          //         if value is not None:
          //             return take(value)   # still int | str | None
          //
          // was refused for passing the un-narrowed type. `py.union.unwrap`
          // has taken a union RESULT since it was written -- it remaps the tag
          // member by member -- so the op this needs already existed.
          auto unwrap = py::UnionUnwrapOp::create(
              builder, loc(anchor), narrowedUnion, found->second.value);
          found->second.value = unwrap.getResult();
        } else if (sourceType && unionType.hasMember(sourceType)) {
          auto unwrap = py::UnionUnwrapOp::create(
              builder, loc(anchor), sourceType, found->second.value);
          found->second.value = unwrap.getResult();
          if (sourceType != narrowed &&
              mlir::isa<py::ContractType>(sourceType) &&
              mlir::isa<py::ContractType>(narrowed) &&
              (py::isAssignableTo(narrowed, sourceType, module) ||
               declaredSubclassOfType(narrowed, sourceType, types))) {
            auto refine = py::ClassRefineOp::create(
                builder, loc(anchor), narrowed, found->second.value);
            found->second.value = refine.getResult();
          }
        }
        // ⛔ `declaredSubclassOfType` beside the subtype walk for the same
        // reason the isinstance analysis needs it: the walk reads class ops,
        // and a class declared FURTHER DOWN the module has none yet, so the
        // refine was skipped and the branch went on using the base type -- the
        // narrowing silently did nothing inside a guard that had just proved
        // the class.
      } else if (found->second.value.getType() != narrowed &&
                 mlir::isa<py::ContractType>(found->second.value.getType()) &&
                 mlir::isa<py::ContractType>(narrowed) &&
                 (py::isAssignableTo(narrowed, found->second.value.getType(),
                                     module) ||
                  declaredSubclassOfType(narrowed,
                                         found->second.value.getType(),
                                         types))) {
        auto refine = py::ClassRefineOp::create(builder, loc(anchor),
                                                narrowed, found->second.value);
        found->second.value = refine.getResult();
      }
      if (found->second.value.getType() == narrowed)
        found->second.type = narrowed;
      // ⭐ A UNION THAT SHRINKS TO A SMALLER UNION IS STILL A PROOF, and the
      // walk above can only spend it when the result is one MEMBER: a union
      // value is a tag plus lanes, so re-tagging it as a narrower union would
      // be a branch, not a view. So the VALUE keeps its shape and the NAME
      // takes the smaller type, which is what the next test reads:
      //
      //     def f(v: int | str | None) -> str:
      //         if v is None: return "none"
      //         if isinstance(v, int): return "i" + str(v)
      //         return "s" + v        # v is str
      //
      // The `is None` guard left `v` a three-member union, so the isinstance
      // after it had TWO members remaining on its false arm and narrowed to
      // neither -- and the tail was refused. With the name at `int | str` the
      // false arm has one member left, which the unwrap above can spend.
      //
      // ⛔ The value is NOT retyped, so every read still produces what it
      // physically is; only the static description a later proof reasons from
      // gets sharper.
      if (auto valueUnion = mlir::dyn_cast<py::UnionType>(
              found->second.value.getType()))
        if (auto narrowedUnion = mlir::dyn_cast<py::UnionType>(narrowed))
          if (narrowedUnion.getMemberTypes().size() <
              valueUnion.getMemberTypes().size()) {
            bool subset = true;
            for (mlir::Type member : narrowedUnion.getMemberTypes())
              subset = subset && valueUnion.hasMember(member);
            if (subset)
              types.bindSymbol(fact.name, narrowed);
          }
    }
    if (found == values.end() || found->second.value.getType() == narrowed)
      types.bindSymbol(fact.name, narrowed);
}

void ModuleEmitter::emitIf(const parser::Node &statement) {
  const parser::Node *test = ast::node(statement, "test");
  std::optional<BranchTypeNarrowing> narrowing =
      test ? optionalBranchTypeNarrowing(*test, types, module) : std::nullopt;
  llvm::StringMap<mlir::Type> savedNarrowedFrom = narrowedFromTypes;
  auto restoreNarrowedFrom = llvm::make_scope_exit(
      [&] { narrowedFromTypes = std::move(savedNarrowedFrom); });

  std::optional<bool> staticTruth =
      test ? optionalStaticBranchTruth(*test, types, module) : std::nullopt;
  if (staticTruth) {
    // A statically taken branch is inline code, not a scope (CPython `if`
    // does not scope): its bindings — assignments, imports, narrowing —
    // persist after the fold, so the platform-switch idiom can bind names
    // (`if os.name == "posix": from posix import *`).
    if (narrowing)
      applyBranchNarrowing(statement, *narrowing, *staticTruth);
    const auto *selected = *staticTruth ? ast::nodeList(statement, "body")
                                        : ast::nodeList(statement, "orelse");
    if (selected && !selected->empty())
      emitStatements(selected);
    return;
  }

  mlir::Value condition = emitBoolValue(emitExpr(test), statement);
  const auto *orelse = ast::nodeList(statement, "orelse");
  bool hasElse = orelse && !orelse->empty();
  // Before the branches, so a name only one of them binds has somewhere to
  // live afterwards. Names BOTH arms bind are merged as block arguments
  // below and are already in scope by the time this runs for them.
  bindConditionallyAssignedLocals(
      statement, {ast::nodeList(statement, "body"), orelse});

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
      applyBranchNarrowing(statement, *narrowing, /*conditionIsTrue=*/true);
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
        applyBranchNarrowing(statement, *narrowing, /*conditionIsTrue=*/false);
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
  // ⭐ The fall-through edge of an else-less `if` carries the NEGATIVE
  // narrowing, the same fact an else block would have been given. Without it
  // the edge contributed the unnarrowed outer type, so the canonical Optional
  // idiom kept its None forever:
  //
  //     def f(n: int | None = None) -> int:
  //         if n is None:
  //             n = 0
  //         return n + 1     # union<int, None> does not provide '__add__'
  //
  // The then edge already narrowed (it assigned), and the join of the two is
  // int. `if n is None: return 0` worked only because that edge never
  // reaches the join.
  llvm::SmallVector<mlir::Type, 2> replacementFallThroughTypes;
  for (auto [index, name] : llvm::enumerate(mutationCandidates)) {
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
    // Null unless the fall-through edge is NARROWED: it doubles as the flag
    // that the edge has to unwrap, and an unconditional compare against the
    // outer type would fire for a literal that merely widens.
    mlir::Type fallThroughType;
    if (!hasElse) {
      mlir::Type edgeType = types.widenLiteral(outer.type);
      if (narrowing && narrowing->name == name && narrowing->falseType)
        if (auto unionType = mlir::dyn_cast<py::UnionType>(outer.type))
          if (unionType.hasMember(narrowing->falseType)) {
            fallThroughType = narrowing->falseType;
            edgeType = fallThroughType;
          }
      parts.push_back(edgeType);
    }
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
    replacementFallThroughTypes.push_back(fallThroughType);
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
      for (auto [slot, candidateIndex] : llvm::enumerate(replacementIndices)) {
        Value incoming = mutationOuterValues[candidateIndex];
        if (mlir::Type narrowed = replacementFallThroughTypes[slot];
            narrowed && narrowed != incoming.type) {
          auto unwrap = py::UnionUnwrapOp::create(builder, loc(statement),
                                                  narrowed, incoming.value);
          incoming = Value{unwrap.getResult(), narrowed};
        }
        falseOperands.push_back(
            coerceValue(incoming, replacementTypes[slot], statement).value);
      }
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
    applyBranchNarrowing(statement, *narrowing, /*conditionIsTrue=*/false);
  else if (narrowing && hasElse && elseTerminates && !thenTerminates)
    applyBranchNarrowing(statement, *narrowing, /*conditionIsTrue=*/true);
}

namespace {

// The statements that bind `name` here, not counting nested scopes. Only the
// forms whose type the source states are collected: an annotated assignment
// says it outright and a plain one says it through its right-hand side. A
// binding this does not recognise (a `for` target, a `with ... as`, an
// augmented assignment) leaves the name alone -- it keeps the unresolved-name
// diagnostic rather than getting a slot whose type would be a guess.
// A binding site for one name: the expression it takes its value from, and --
// for a tuple unpack -- which element of that expression is the name's.
struct NameBindingSite {
  const parser::Node *value;
  int element = -1;
};

// ⭐ WHICH ELEMENT OF THE UNPACK IS THIS NAME. `key, _, value = s.partition(c)`
// inside a loop bound NEITHER name for the scope around it: an unpack target
// went to the `opaque` arm below, which is the answer for a target this walk
// cannot read a type out of -- a subscript, an attribute, a star. A flat tuple
// of names is not that: the value's type has the element in it. Every such
// binding was refused as "unresolved name" after the region, for the spelling
// that reads a split line.
int unpackElementFor(const parser::Node *target, llvm::StringRef name) {
  if (!target || (target->kind != "Tuple" && target->kind != "List"))
    return -1;
  const auto *elements = ast::nodeList(*target, "elts");
  if (!elements)
    return -1;
  int found = -1;
  for (std::size_t index = 0; index < elements->size(); ++index) {
    const parser::Node *element = (*elements)[index].get();
    // A star target absorbs a variable number of elements, so no fixed index
    // describes anything after it -- and the star's own name is a list whose
    // length is not known here either.
    if (!element || element->kind != "Name")
      return -1;
    if (llvm::StringRef(ast::nameSpelling(*element)) == name)
      found = static_cast<int>(index);
  }
  return found;
}

void collectNameBindingExpressions(
    const parser::Node *node, llvm::StringRef name,
    llvm::SmallVectorImpl<NameBindingSite> &values,
    llvm::SmallVectorImpl<const parser::Node *> &annotations, bool &opaque) {
  if (!node)
    return;
  if (node->kind == "FunctionDef" || node->kind == "AsyncFunctionDef" ||
      node->kind == "ClassDef" || node->kind == "Lambda")
    return;
  auto targetsName = [&](const parser::Node *target) {
    return target && target->kind == "Name" &&
           llvm::StringRef(ast::nameSpelling(*target)) == name;
  };
  if (node->kind == "Assign") {
    if (const auto *targets = ast::nodeList(*node, "targets"))
      for (const parser::NodePtr &target : *targets) {
        if (targetsName(target.get())) {
          const parser::Node *value = ast::node(*node, "value");
          if (!value)
            opaque = true;
          // ⛔ A BINDING THAT READS THE NAME SAYS NOTHING ABOUT ITS TYPE.
          // `result = result + "!"` in a try's `else` is a REBINDING; the type
          // comes from the bindings that do not depend on it, and inferring
          // this one (with the name still unbound) answered `object` and took
          // the whole join down with it.
          else if (!containsNameLoad(value, name))
            values.push_back({value, -1});
        } else if (int element = unpackElementFor(target.get(), name);
                   element >= 0) {
          const parser::Node *value = ast::node(*node, "value");
          if (!value || containsNameLoad(value, name))
            opaque = true;
          else
            values.push_back({value, element});
        } else {
          llvm::StringSet<> written;
          collectAssignedNameTargets(target.get(), written);
          if (written.contains(name))
            opaque = true;
        }
      }
  } else if (node->kind == "AnnAssign") {
    if (targetsName(ast::node(*node, "target"))) {
      if (const parser::Node *annotation = ast::node(*node, "annotation"))
        annotations.push_back(annotation);
      else
        opaque = true;
    }
  } else {
    llvm::StringSet<> written;
    if (node->kind == "AugAssign" || node->kind == "NamedExpr" ||
        node->kind == "For" || node->kind == "AsyncFor" ||
        node->kind == "With" || node->kind == "AsyncWith")
      collectAssignedNames(node, written);
    if (written.contains(name))
      opaque = true;
  }

  for (const parser::Field &field : node->fields) {
    if (const auto *child = std::get_if<parser::NodePtr>(&field.value)) {
      if (*child)
        collectNameBindingExpressions(child->get(), name, values, annotations,
                                      opaque);
    } else if (const auto *children =
                   std::get_if<std::vector<parser::NodePtr>>(&field.value)) {
      for (const parser::NodePtr &child : *children)
        if (child)
          collectNameBindingExpressions(child.get(), name, values, annotations,
                                        opaque);
    }
  }
}

// Is `name` a loop or `with` TARGET anywhere in this suite? Such a name is
// rebound per statement and its type is whatever the current iterable yields,
// which a slot -- one storage with one type -- cannot represent.
bool containsBindingTarget(const parser::Node *node, llvm::StringRef name) {
  if (!node)
    return false;
  llvm::StringSet<> targets;
  if (node->kind == "For" || node->kind == "AsyncFor")
    collectAssignedNameTargets(ast::node(*node, "target"), targets);
  else if (node->kind == "With" || node->kind == "AsyncWith") {
    if (const auto *items = ast::nodeList(*node, "items"))
      for (const parser::NodePtr &item : *items)
        collectAssignedNameTargets(ast::node(*item, "optional_vars"), targets);
  } else if (node->kind == "comprehension")
    collectAssignedNameTargets(ast::node(*node, "target"), targets);
  if (targets.contains(name))
    return true;

  for (const parser::Field &field : node->fields) {
    if (const auto *child = std::get_if<parser::NodePtr>(&field.value)) {
      if (*child && containsBindingTarget(child->get(), name))
        return true;
    } else if (const auto *children =
                   std::get_if<std::vector<parser::NodePtr>>(&field.value)) {
      for (const parser::NodePtr &child : *children)
        if (child && containsBindingTarget(child.get(), name))
          return true;
    }
  }
  return false;
}


} // namespace

// ⭐ IS THIS NAME READ AFTER THE STATEMENT BEING EMITTED, anywhere the type it
// carries could be observed? Only answerable at the outermost suite of a scope,
// where "the rest of this suite" IS the rest of the scope; deeper in, the
// enclosing suites are on the C++ stack and this walk cannot see them, so it
// says YES and the caller keeps whatever it does when the answer is unknown.
//
// Distinct from `nameIsReadAfterCurrentStatement` below, which answers a
// different question -- it refuses outright for any name the suite BINDS,
// because a slot is the wrong shape for a loop target. Here the binding is the
// point.
bool ModuleEmitter::nameMayBeReadAfterCurrentStatement(
    llvm::StringRef name) const {
  // ⛔ NEVER FOR A NAME THE EMITTER ITSELF INTRODUCED. Every synthesized loop
  // spells its scratch with a `__` stem (`__lyfuse3_v`, `__maxbi1`), and the
  // reads that go with them are synthesized too -- they are emitted directly
  // and never appear in the AST this walk reads, so the walk would conclude
  // that nothing reads them and stop carrying the very value the fold returns.
  // Seven goldens said so: the max/min index fold, the genexp fusions and the
  // lambda callee all lost their result.
  if (suiteStack.empty() || name.starts_with("__"))
    return true;
  for (unsigned level = static_cast<unsigned>(suiteStack.size());
       level > suiteStackFloor; --level) {
    const auto &[suite, index] = suiteStack[level - 1];
    if (!suite)
      return true;
    for (std::size_t position = index; position < suite->size(); ++position)
      if (containsNameLoad((*suite)[position].get(), name))
        return true;
  }
  return false;
}

bool ModuleEmitter::nameIsReadAfterCurrentStatement(llvm::StringRef name) const {
  if (!currentSuite)
    return false;
  // ⛔ A LOOP TARGET NEVER GETS A SLOT, whatever reads it later. Its type is
  // the current iterable's element type and the same spelling is reused across
  // loops over different things -- `for i, x in enumerate(["a"])` then
  // `for i, x in enumerate([5])` -- so one storage with one type is the wrong
  // shape for it. The lazy-iterator fusions make this reachable even where the
  // source has no assignment: they rewrite the target into a body assignment,
  // which is what put a `str` slot under an `int` and reported
  // "builtins.int does not provide manifest method '__add__'".
  for (const parser::NodePtr &statement : *currentSuite)
    if (containsBindingTarget(statement.get(), name))
      return false;
  for (std::size_t index = currentSuiteIndex; index < currentSuite->size();
       ++index)
    if (containsNameLoad((*currentSuite)[index].get(), name))
      return true;
  return false;
}

// ⭐ CPYTHON LEAVES THE TARGET BOUND, and refusing the read after the loop
// made the idiom that reports the last index reachable only by declaring the
// name first. `nameIsReadAfterCurrentStatement` cannot answer this: it says NO
// for every name the suite binds, which is what keeps a REUSED target spelling
// (`for i, x in enumerate(["a"])` then `for i, x in enumerate([5])`) from being
// forced through one slot of one type. That rule is about the spelling being
// bound MORE THAN ONCE, so this asks exactly that instead.
//
// ⛔ ONE BINDING SITE, not one type. Two loops whose targets happen to agree
// still keep the refusal here (`for i in range(3): for i in range(2): pass` /
// `print(i)`) -- checking that would mean inferring an element type per site,
// and the sites that disagree are the reason the restriction exists at all.
//
// ⛔ AND ONE SUITE. The read has to be in the suite the loop is in, so an
// INNER loop's target read after the OUTER loop (`for row in rows: for cell in
// row: ...` / `print(cell)`) keeps the refusal: `currentSuite` is the inner
// body by then and the enclosing suites are on the C++ stack.
bool ModuleEmitter::loopTargetOutlivesLoop(
    llvm::StringRef name, const parser::Node &statement) const {
  if (!currentSuite)
    return false;
  for (const parser::NodePtr &other : *currentSuite)
    if (other.get() != &statement && containsBindingTarget(other.get(), name))
      return false;
  // Rebound inside the loop, so the element type is not what the slot holds.
  for (const char *field : {"body", "orelse"})
    if (const auto *body = ast::nodeList(statement, field))
      for (const parser::NodePtr &inner : *body)
        if (containsBindingTarget(inner.get(), name))
          return false;
  // The `else` clause runs on the exit edge and reads the target like anything
  // after the loop does -- `for i in r: pass` / `else: print(i)`.
  if (const auto *orelse = ast::nodeList(statement, "orelse"))
    for (const parser::NodePtr &inner : *orelse)
      if (containsNameLoad(inner.get(), name))
        return true;
  for (std::size_t index = currentSuiteIndex; index < currentSuite->size();
       ++index)
    if (containsNameLoad((*currentSuite)[index].get(), name))
      return true;
  return false;
}

mlir::Type ModuleEmitter::inferConditionalLocalType(
    llvm::ArrayRef<const std::vector<parser::NodePtr> *> bodies,
    llvm::StringRef name) {
  llvm::SmallVector<NameBindingSite, 4> valueNodes;
  llvm::SmallVector<const parser::Node *, 2> annotationNodes;
  bool opaque = false;
  for (const std::vector<parser::NodePtr> *body : bodies) {
    if (!body)
      continue;
    for (const parser::NodePtr &statement : *body)
      collectNameBindingExpressions(statement.get(), name, valueNodes,
                                    annotationNodes, opaque);
  }
  if (opaque)
    return {};
  // An annotation is the author's answer and outranks any inference.
  if (!annotationNodes.empty()) {
    mlir::Type annotated = types.annotationType(annotationNodes.front());
    for (const parser::Node *other : llvm::drop_begin(annotationNodes))
      if (types.annotationType(other) != annotated)
        return {};
    return annotated;
  }
  if (valueNodes.empty())
    return {};
  llvm::SmallVector<mlir::Type, 4> inferred;
  for (const NameBindingSite &site : valueNodes) {
    mlir::Type type = types.widenLiteral(types.inferExpr(site.value));
    if (!type)
      return {};
    if (site.element >= 0) {
      auto source = mlir::dyn_cast<py::ContractType>(type);
      if (!source)
        return {};
      llvm::StringRef sourceName = source.getContractName();
      if (sourceName == "builtins.tuple") {
        if (static_cast<std::size_t>(site.element) >=
            source.getArguments().size())
          return {};
        type = source.getArguments()[site.element];
      } else if (sourceName == "builtins.list" &&
                 source.getArguments().size() == 1) {
        // `k, v = line.split("=")` unpacks a homogeneous sequence: how many
        // elements it has is a RUNTIME question (and the wrong count is a
        // ValueError there, not here), but every one of them has the same
        // type, so the index does not change the answer.
        type = source.getArguments().front();
      } else {
        return {};
      }
      type = types.widenLiteral(type);
      if (!type)
        return {};
    }
    inferred.push_back(type);
  }
  mlir::Type joined = types.join(inferred);
  // ⛔ ONLY A PLAIN CONTRACT GETS A SLOT. The slot is a synthesized class's
  // box-fronted field, so whatever goes in it has to be storable there: a
  // union keeps every member's lanes and is refused at the box, and a `type[X]`
  // has no object handle at all -- `WPROTO = ctypes.CFUNCTYPE(...)` inside a
  // nested `if` (runtime/lib/stackguard_support.py) failed to lower as
  // "collection payload element ... has no physical object handle". A join
  // that reached the erased top is excluded for the opposite reason: the slot
  // would accept every write and refuse every read.
  if (!mlir::isa_and_nonnull<py::ContractType>(joined) ||
      py::isPyObjectType(joined))
    return {};
  return joined;
}

// ⭐ A NAME BOUND INSIDE A REGION IS STILL A LOCAL OF THE SCOPE AROUND IT.
// CPython decides that syntactically -- an assignment anywhere in a function
// body makes the name local to the whole body -- and this compiler decided it
// by DOMINANCE, so `for v in xs: last = v` followed by `print(last)` was
// "unresolved name 'last'" for a program CPython runs. The binding is given a
// slot before the region, which is where the enclosing scope can see it, and
// the slot records whether it was written so the read can raise.
void ModuleEmitter::bindConditionallyAssignedLocals(
    const parser::Node &anchor,
    llvm::ArrayRef<const std::vector<parser::NodePtr> *> bodies,
    const llvm::StringMap<mlir::Type> *inferenceHints,
    const llvm::StringMap<mlir::Type> *boundWithKnownType) {
  llvm::StringSet<> assigned;
  for (const std::vector<parser::NodePtr> *body : bodies)
    collectAssignedNames(body, assigned);
  if (boundWithKnownType)
    for (const auto &entry : *boundWithKnownType)
      assigned.insert(entry.getKey());
  // ⭐ A NAME A NESTED CALLABLE READS NEEDS ITS SLOT HERE EVEN IF NOTHING
  // AFTER THE REGION READS IT. The boxed path makes the cell at the first
  // BINDING, and inside a loop that site runs once per trip, so each closure
  // got a cell of its own:
  //
  //     for i in range(3):
  //         k = i * 10
  //         fs.append(lambda: k)     # [0, 10, 20]; CPython [20, 20, 20]
  //
  // The slot made here is before the region, which is where the frame's cell
  // belongs. This is why such a name is no longer skipped for being boxed:
  // being boxed is the reason it needs the slot, not a reason to leave it.
  llvm::StringSet<> readByNested;
  for (const std::vector<parser::NodePtr> *body : bodies)
    for (const auto &entry : namesReadByNestedCallables(body))
      readByNested.insert(entry.getKey());
  llvm::SmallVector<std::string, 4> names;
  for (const auto &entry : assigned) {
    llvm::StringRef name = entry.getKey();
    if (values.find(name) != values.end())
      continue;
    // A name the module already owns is a global, not a local of this scope.
    if (isModuleGlobalRead(name) || moduleGlobals.count(name) ||
        (currentBoxedLocals.contains(name) && !readByNested.contains(name)) ||
        types.lookupSymbol(name) || types.lookupClass(name))
      continue;
    // ⭐ ONLY A NAME SOMETHING LATER READS. Every name a region binds could be
    // given a slot, and giving one to a name used only INSIDE the region
    // changes the representation of code that already works: a plain SSA local
    // becomes a heap slot with a guarded read, and 42 tests failed on the
    // ownership and typing that follows from that. The defect is a read the
    // scope cannot reach, so the fix is scoped to exactly those reads.
    //
    // ⛔ THE SAME SUITE ONLY. A read in a suite further out (the loop is
    // inside an `if` and the read is after the `if`) is not seen here and
    // keeps the unresolved-name diagnostic. Widening it means threading the
    // enclosing suites, which is a bigger change than the one this makes.
    if (!(boundWithKnownType && boundWithKnownType->count(name)) &&
        !readByNested.contains(name) && !nameIsReadAfterCurrentStatement(name))
      continue;
    names.push_back(name.str());
  }
  llvm::sort(names);
  // ⭐ THE LOOP TARGET IS A HINT AND NOT A BINDING. `for v in xs: last = v`
  // types `last` from `v`, which is not in scope until the body runs -- and
  // the slot has to exist BEFORE the loop. The target's type is known from
  // the iterable, so it is bound for the length of the inference and dropped:
  // binding it for real here would leak the loop variable into the scope
  // around the loop with no value behind it.
  llvm::SmallVector<std::pair<std::string, mlir::Type>, 4> contents;
  {
    TypeSystem::Scope hintScope = types.pushScope();
    if (inferenceHints)
      for (const auto &hint : *inferenceHints)
        types.bindSymbol(hint.getKey(), hint.getValue());
    for (const std::string &name : names) {
      // The known type wins: the loop target's type is the iterable's element
      // type, and a body that reassigns the target to something else would
      // otherwise decide the slot.
      if (boundWithKnownType) {
        auto known = boundWithKnownType->find(name);
        if (known != boundWithKnownType->end()) {
          // The same storability rule `inferConditionalLocalType` ends with:
          // `for x in []` has an erased element type, and a slot for it
          // accepts every write and refuses every read -- the program left
          // the emitter and died in the lowering as "a type-erased `object`
          // value cannot be stored in field 'class.v'". With no slot it keeps
          // the unresolved-name refusal, which is what it had before.
          if (mlir::isa_and_nonnull<py::ContractType>(known->getValue()) &&
              !py::isPyObjectType(known->getValue()))
            contents.emplace_back(name, known->getValue());
          continue;
        }
      }
      if (mlir::Type content = inferConditionalLocalType(bodies, name))
        contents.emplace_back(name, content);
    }
  }
  for (const auto &[name, content] : contents) {
    auto unbound = py::UnboundOp::create(builder, loc(anchor), content);
    Value slot = emitCellAlloc(anchor, Value{unbound.getResult(), content},
                               /*tracksBinding=*/true);
    values[name] = slot;
    types.bindSymbol(name, content);
  }
}

} // namespace lython::emitter
