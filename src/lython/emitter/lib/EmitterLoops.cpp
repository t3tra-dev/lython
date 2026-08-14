#include "AstSynth.h"
#include "EmitterCore.h"
#include "EmitterPyOps.h"
#include "EmitterSupport.h"

#include "AstAccess.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"

namespace lython::emitter {

namespace {

// ⭐ The loop's own token for a UNION carried local, acquired on the entry
// edge. Every carried local is released once per iteration by
// carriedLoopEdgeOperands ("the loop-carried ownership token stays balanced
// on every edge that leaves the body"), and the matching acquisition on the
// way IN is placed by insertOwnedBlockArgumentReleases -- for every type
// except a union, whose release is guarded by its tag and which that pass
// skips (`g.condition`, Passes/Runtime/Passes/Ownership.cpp).
//
// With neither half, the first back edge released a value the loop never
// owned: `def f(s: str | None): while s is not None: s = None; return
// "start"` freed the CALLER's argument and printed an empty line, or aborted
// with "Ly_DecRef observed non-positive refcount" depending on what reused
// the hole. The pre-loop value's own token stays with the pre-loop binding,
// exactly as the "defined before the loop covers none" rule in that function
// says, so this is an acquisition and not a duplicate.
//
// ⛔ Why here and not there: emitting it in the pass needs a TAG-GUARDED
// call (`cmpi eq(tag, activeTag)` around the retain), and that pass emits
// only unguarded ones. py.incref on a union already lowers to the guarded
// form, so the emitter can say it in one op. When the pass grows a guarded
// emitter this belongs there, with the release half.
void acquireUnionCarriedTokens(mlir::OpBuilder &builder, mlir::Location loc,
                               llvm::ArrayRef<CarriedLoopLocal> carried,
                               llvm::ArrayRef<mlir::Value> initialValues) {
  for (auto [index, local] : llvm::enumerate(carried)) {
    if (index >= initialValues.size() || !initialValues[index])
      continue;
    if (!mlir::isa<py::UnionType>(local.type))
      continue;
    py::IncRefOp::create(builder, loc, initialValues[index]);
  }
}

// The other half of the same token: the loop acquired one on the way in, every
// edge that leaves the body keeps it balanced, and this discharges it on the
// way out.
//
// ⭐ Why releasing whatever is carried AT EXIT is right in all three shapes,
// which is what a reading of `carriedLoopEdgeOperands` settles: that function
// does not only release the lane the body replaced, it RE-ACQUIRES the
// replacement (`acquiringLanes`). So the loop's token rides the carry.
//
//   body never runs .......... exit carries v0; release balances the entry retain
//   body keeps v0 ............ no edge release, no reacquire; same
//   body rebinds to v1 ....... back edge released v0 and retained v1; the exit
//                              release discharges THAT token, and v1's own
//                              producer token is still the pass's to place
//
// ⛔ Why NOT gate this on the name being unread after the loop, which is what
// the earlier record proposed on the reasoning that a rebound-to-fresh value
// would dangle: it would not. The token released here is the loop's, minted by
// the reacquire, never the value's only one. The gate was written before
// `acquiringLanes` was read.
//
// ⛔ And why the after-block start rather than the exit edge: with a break the
// after-block has several predecessors and each carries the token, so one
// release at the join covers them where an edge release would need one per
// edge and would still miss the else block's.
void releaseUnionCarriedTokens(mlir::OpBuilder &builder, mlir::Location loc,
                               llvm::ArrayRef<CarriedLoopLocal> carried,
                               mlir::Block *afterBlock,
                               bool afterForwardsCarried,
                               llvm::ArrayRef<mlir::Value> headerValues) {
  for (auto [index, local] : llvm::enumerate(carried)) {
    if (!mlir::isa<py::UnionType>(local.type))
      continue;
    mlir::Value carriedAtExit;
    if (afterForwardsCarried) {
      if (index < afterBlock->getNumArguments())
        carriedAtExit = afterBlock->getArgument(index);
    } else if (index < headerValues.size()) {
      carriedAtExit = headerValues[index];
    }
    if (!carriedAtExit)
      continue;
    py::DecRefOp::create(builder, loc, carriedAtExit);
  }
}

} // namespace

namespace {

Value stripLocalProtocolView(Value value) {
  if (!value.value)
    return value;
  auto view = value.value.getDefiningOp<py::ProtocolViewOp>();
  if (!view)
    return value;
  return Value{view.getInput(), view.getInput().getType()};
}

} // namespace

namespace {

// Break statements that target this loop, not one nested inside it.
bool containsLoopLevelBreak(const parser::Node *node) {
  if (!node)
    return false;
  if (node->kind == "Break")
    return true;
  if (node->kind == "For" || node->kind == "While" ||
      node->kind == "AsyncFor" || node->kind == "FunctionDef" ||
      node->kind == "AsyncFunctionDef" || node->kind == "ClassDef" ||
      node->kind == "Lambda")
    return false;
  for (const parser::Field &field : node->fields) {
    if (const auto *child = std::get_if<parser::NodePtr>(&field.value)) {
      if (containsLoopLevelBreak(child->get()))
        return true;
    } else if (const auto *children =
                   std::get_if<std::vector<parser::NodePtr>>(&field.value)) {
      for (const parser::NodePtr &item : *children)
        if (containsLoopLevelBreak(item.get()))
          return true;
    }
  }
  return false;
}

bool containsLoopLevelBreak(const std::vector<parser::NodePtr> *body) {
  if (!body)
    return false;
  for (const parser::NodePtr &item : *body)
    if (containsLoopLevelBreak(item.get()))
      return true;
  return false;
}

bool containsNamedExpr(const parser::Node *node) {
  if (!node)
    return false;
  if (node->kind == "NamedExpr")
    return true;
  for (const parser::Field &field : node->fields) {
    if (const auto *child = std::get_if<parser::NodePtr>(&field.value)) {
      if (containsNamedExpr(child->get()))
        return true;
    } else if (const auto *children =
                   std::get_if<std::vector<parser::NodePtr>>(&field.value)) {
      for (const parser::NodePtr &item : *children)
        if (containsNamedExpr(item.get()))
          return true;
    }
  }
  return false;
}

} // namespace

llvm::SmallVector<CarriedLoopLocal, 4> ModuleEmitter::collectCarriedLoopLocals(
    const parser::Node &statement, const llvm::StringSet<> *excludedNames,
    llvm::SmallVectorImpl<mlir::Value> &initialValues) {
  llvm::StringSet<> assignedInBody;
  collectAssignedNames(ast::nodeList(statement, "body"), assignedInBody);
  // ⭐ The loop TARGET is rebound once per iteration too, and CPython leaves
  // it bound after the loop. It is not a body statement, so this scan never
  // saw it and the post-loop read got the pre-loop value (`i = -1; for i in
  // range(1): pass; print(i)` printed -1 where CPython prints 0). A name the
  // loop introduces is filtered out below by not being in `values` yet, which
  // is the right answer for it as well: with zero iterations CPython has
  // nothing bound there either.
  if (const parser::Node *target = ast::node(statement, "target"))
    collectAssignedNameTargets(target, assignedInBody);
  // A walrus in a while condition rebinds per iteration exactly like a body
  // assignment (the field is absent on for/async-for statements).
  if (const parser::Node *test = ast::node(statement, "test"))
    collectAssignedNames(test, assignedInBody);
  // ⭐ The `else` clause writes on the loop's EXIT edge, so its writes have
  // to reach the after-block the same way the body's do. A name written only
  // there was not carried, and the else block's own scope closed over it:
  // `for i in [1, 2]: pass` / `else: acc += 7` printed 7 inside the else and
  // the PRE-LOOP value after it, with no diagnostic. The break edges skip the
  // else and carry the pre-loop value, which is what CPython does too.
  collectAssignedNames(ast::nodeList(statement, "orelse"), assignedInBody);
  llvm::SmallVector<std::string, 4> names;
  for (const auto &assigned : assignedInBody) {
    if (excludedNames && excludedNames->contains(assigned.getKey()))
      continue;
    auto found = values.find(assigned.getKey());
    if (found == values.end() || !found->second.value)
      continue;
    names.push_back(assigned.getKey().str());
  }
  llvm::sort(names);
  llvm::SmallVector<CarriedLoopLocal, 4> carried;
  carried.reserve(names.size());
  for (const std::string &name : names) {
    Value initial = values.find(name)->second;
    mlir::Type carriedType = types.widenLiteral(initial.type);
    mlir::Value initialValue =
        coerceValue(initial, carriedType, statement).value;
    mlir::Value pinBuffer;
    if (auto tensorType =
            mlir::dyn_cast<mlir::RankedTensorType>(carriedType)) {
      // A break edge hands the carried buffer to the after-block while later
      // iterations keep rewriting it; the bufferization analysis on
      // unstructured control flow cannot separate those paths and rejects
      // the whole function. Diagnose here, where the local can still be
      // named.
      if (containsLoopLevelBreak(ast::nodeList(statement, "body"))) {
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, statement.range.start,
            "primitive tensor local '" + name +
                "' cannot be reassigned inside a loop that breaks: break "
                "would hand the carried buffer out of the loop"});
        continue;
      }
      // A nested loop must keep writing the enclosing loop's buffer; a
      // second allocation would make the outer back-edge forward a buffer
      // that is not equivalent to the outer entry's.
      for (auto context = loopControlContexts.rbegin();
           !pinBuffer && context != loopControlContexts.rend(); ++context)
        for (const CarriedLoopLocal &enclosing : context->carriedLocals)
          if (enclosing.name == name && enclosing.pinBuffer) {
            pinBuffer = enclosing.pinBuffer;
            break;
          }
      if (!pinBuffer) {
        auto alloc = mlir::bufferization::AllocTensorOp::create(
            builder, loc(statement), tensorType, mlir::ValueRange{},
            initialValue);
        pinBuffer = alloc.getResult();
        initialValue = pinBuffer;
      }
    }
    carried.push_back(CarriedLoopLocal{name, carriedType, pinBuffer});
    initialValues.push_back(initialValue);
  }
  return carried;
}

Value ModuleEmitter::pinLoopCarriedTensor(llvm::StringRef name, Value value,
                                          const parser::Node &anchor) {
  if (!value.value ||
      !mlir::isa<mlir::RankedTensorType>(value.value.getType()))
    return value;
  for (auto context = loopControlContexts.rbegin();
       context != loopControlContexts.rend(); ++context) {
    for (const CarriedLoopLocal &local : context->carriedLocals) {
      if (local.name != name)
        continue;
      if (!local.pinBuffer ||
          local.pinBuffer.getType() != value.value.getType())
        return value;
      // A rebound shaped primitive is copied into the loop's pre-loop buffer
      // at the assignment instead of being forwarded fresh at each loop
      // edge: forwarding would allocate per iteration and hand ownership
      // through block arguments, which the static release plan cannot
      // follow. Pinning at the (single) assignment also keeps every loop
      // edge forwarding values of one equivalent buffer.
      if (value.value == local.pinBuffer)
        return value;
      if (auto materialize =
              value.value.getDefiningOp<
                  mlir::bufferization::MaterializeInDestinationOp>())
        if (materialize.getDest() == local.pinBuffer)
          return value;
      mlir::Value pinned =
          mlir::bufferization::MaterializeInDestinationOp::create(
              builder, loc(anchor), value.value, local.pinBuffer)
              ->getResult(0);
      return Value{pinned, value.type};
    }
  }
  return value;
}

void ModuleEmitter::bindCarriedLoopLocals(
    llvm::ArrayRef<CarriedLoopLocal> carried, mlir::Block *block) {
  for (auto [index, local] : llvm::enumerate(carried)) {
    values[local.name] = Value{block->getArgument(index), local.type};
    types.bindSymbol(local.name, local.type);
  }
}

llvm::SmallVector<mlir::Value, 4> ModuleEmitter::carriedLoopEdgeOperands(
    const parser::Node &anchor, llvm::ArrayRef<CarriedLoopLocal> carried,
    mlir::Block *headerBlock, llvm::ArrayRef<mlir::Value> baselineValues,
    bool toHeader) {
  llvm::SmallVector<mlir::Value, 4> operands;
  operands.reserve(carried.size());
  // Releases are not emitted lane-by-lane: an alias assignment (`m = i`)
  // makes two lanes forward one SSA token, or forwards a token another lane
  // is abandoning on the same edge, so the edge's token ledger has to be
  // settled after every lane's forwarded/previous pair is known. The ledger
  // is keyed on the value behind upcast/protocol views: a per-edge coerce
  // mints a fresh view op, not a fresh object, and refcounts act on the
  // object.
  auto tokenRoot = [](mlir::Value value) {
    while (value) {
      if (auto upcast = value.getDefiningOp<py::ClassUpcastOp>()) {
        value = upcast.getInput();
        continue;
      }
      if (auto refine = value.getDefiningOp<py::ClassRefineOp>()) {
        value = refine.getInput();
        continue;
      }
      if (auto view = value.getDefiningOp<py::ProtocolViewOp>()) {
        value = view.getInput();
        continue;
      }
      break;
    }
    return value;
  };
  llvm::SmallVector<mlir::Value, 4> pendingReleases;
  llvm::SmallDenseSet<mlir::Value, 4> keptRoots;
  struct AcquireLedger {
    mlir::Value representative; // contract-typed view to retain through
    unsigned laneCount = 0;
  };
  llvm::MapVector<mlir::Value, AcquireLedger> acquiringLanes;
  for (auto [index, local] : llvm::enumerate(carried)) {
    auto found = values.find(local.name);
    if (found == values.end() || !found->second.value) {
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, anchor.range.start,
          "loop lost carried local '" + local.name + "'"});
      continue;
    }
    mlir::Value value = coerceValue(found->second, local.type, anchor).value;
    // Release the current-iteration header value when the body replaced it, so
    // the loop-carried ownership token stays balanced on every edge that
    // leaves the body (back-edge, break, continue, async-for try yield).
    // Locals that are not contract terms (primitive scalars and tensors) carry
    // no token to balance, and py.decref does not accept them.
    mlir::Value previous;
    if (index < baselineValues.size() && baselineValues[index])
      previous = baselineValues[index];
    else if (headerBlock && index < headerBlock->getNumArguments())
      previous = headerBlock->getArgument(index);
    if (previous && py::isPyContractType(local.type)) {
      mlir::Value valueRoot = tokenRoot(value);
      if (valueRoot != tokenRoot(previous) &&
          !derivesViaStructuralMutation(value, previous)) {
        pendingReleases.push_back(previous);
        AcquireLedger &ledger = acquiringLanes[valueRoot];
        ledger.representative = value;
        ++ledger.laneCount;
      } else {
        keptRoots.insert(valueRoot);
      }
    }
    operands.push_back(value);
  }
  for (auto &[root, ledger] : acquiringLanes) {
    unsigned laneCount = ledger.laneCount;
    // A lane abandoning this value on the same edge hands its token straight
    // to a lane acquiring it (decref + reacquire would touch a dead token).
    unsigned transferred = 0;
    for (mlir::Value &release : pendingReleases)
      if (release && tokenRoot(release) == root && transferred < laneCount) {
        release = nullptr;
        ++transferred;
      }
    // Header edges only: each header lane owns its own token per iteration
    // (the per-lane decrefs above assume it), so duplicate lanes retain. A
    // value produced this iteration brings its own creation token; a block
    // argument's token belongs to whichever lane (this loop's, an enclosing
    // loop's, or the entry ABI) already owns it, so it covers no lane here.
    // Exit edges (break, loop-else) are NOT retained: the ownership planner
    // folds aliased after-block arguments into one resource, and a dead
    // alias lane needs no token of its own there.
    if (!toHeader)
      continue;
    // How many of the acquiring lanes the value's own token can cover:
    // - a block argument's token is free to move into ONE lane unless the
    //   argument's own lane keeps it this edge (then a sibling alias lane
    //   must retain — the token stays with the keeping lane);
    // - an op result minted this iteration covers one lane;
    // - a value defined before the loop covers none: its creation token
    //   stays claimed by the pre-loop local that still binds it (`t = base`
    //   inside the loop must not steal base's token).
    bool bringsOwnToken;
    if (mlir::isa<mlir::BlockArgument>(root)) {
      bringsOwnToken = !keptRoots.contains(root);
    } else {
      bringsOwnToken = true;
      if (headerBlock) {
        mlir::Block *defBlock = root.getParentBlock();
        while (defBlock && defBlock->getParent() != headerBlock->getParent())
          defBlock = defBlock->getParentOp()
                         ? defBlock->getParentOp()->getBlock()
                         : nullptr;
        if (defBlock && defBlock != headerBlock)
          for (mlir::Block &blk : *headerBlock->getParent()) {
            if (&blk == defBlock) {
              bringsOwnToken = false; // defined before the loop header
              break;
            }
            if (&blk == headerBlock)
              break;
          }
      }
    }
    unsigned selfTokens = bringsOwnToken ? 1 : 0;
    for (unsigned acquired = transferred + selfTokens; acquired < laneCount;
         ++acquired)
      py::IncRefOp::create(builder, loc(anchor), ledger.representative);
  }
  for (mlir::Value release : pendingReleases)
    if (release)
      py::DecRefOp::create(builder, loc(anchor), release);
  return operands;
}

llvm::SmallVector<mlir::Value, 4>
ModuleEmitter::loopCarriedBranchOperands(const parser::Node &anchor,
                                         const LoopControlContext &loop,
                                         mlir::Block *target) {
  return carriedLoopEdgeOperands(anchor, loop.carriedLocals, loop.headerBlock,
                                 loop.baselineValues,
                                 /*toHeader=*/target == loop.headerBlock);
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
    llvm::SmallVector<std::string, 2> targetNames;
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
    // ⭐ A TUPLE target is what emitFor already binds -- `for a, b in zip(..)`
    // is an ordinary loop -- so the fusion only has to know which names to
    // scope. Rejecting it here made `sum(a * b for a, b in zip(xs, ys))`
    // "generator expression target must be a simple name" while the list
    // comprehension spelling of the same thing worked.
    if (!entry.target || !entry.iter)
      return reject("malformed generator expression");
    auto appendTargetName = [&](const parser::Node *node, auto &&self) -> bool {
      if (!node)
        return false;
      if (node->kind == "Name") {
        entry.targetNames.push_back(std::string(ast::nameSpelling(*node)));
        return true;
      }
      if (node->kind != "Tuple" && node->kind != "List")
        return false;
      const auto *elts = ast::nodeList(*node, "elts");
      if (!elts || elts->empty())
        return false;
      for (const parser::NodePtr &elt : *elts)
        if (!self(elt.get(), self))
          return false;
      return true;
    };
    if (!appendTargetName(entry.target.get(), appendTargetName))
      return reject("generator expression target must be a name or a tuple "
                    "of names");
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
  parser::NodePtr assign = synth::assign(forTarget, elt, statement.range);
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
  for (const GenEntry &entry : chain)
    for (const std::string &name : entry.targetNames) {
      std::optional<Value> prior;
      if (auto found = values.find(name); found != values.end())
        prior = found->second;
      priorTargets.push_back({name, prior});
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
  const auto *orelse = ast::nodeList(statement, "orelse");
  bool hasElse = orelse && !orelse->empty();
  if (const parser::Node *iterNode = ast::node(statement, "iter");
      iterNode && iterNode->kind == "GeneratorExp") {
    if (hasElse) {
      // The genexpr fusion rewrites the loop into nested loops, which would
      // silently re-scope a break away from the else's no-break condition.
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, statement.range.start,
          "for/else over a generator expression is not supported yet"});
      return;
    }
    emitGeneratorExpFor(statement, *iterNode);
    return;
  }
  // Lazy iterator builtins / dict views consumed directly by the loop fuse
  // into equivalent rewritten loops (EmitterIterators.cpp) — no iterator
  // object is materialized, matching CPython's per-element evaluation order.
  if (const parser::Node *iterNode = ast::node(statement, "iter");
      iterNode && iterNode->kind == "Call" &&
      tryEmitLazyIteratorFor(statement, *iterNode))
    return;
  // itertools calls consumed directly by the loop fuse the same way; names
  // without a fusion fall through to the value synthesis via emitFor.
  if (const parser::Node *iterNode = ast::node(statement, "iter");
      iterNode && iterNode->kind == "Call" &&
      tryEmitItertoolsFor(statement, *iterNode))
    return;
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
    if (emptyLiteral) {
      // Zero iterations means no break: the else body always runs.
      if (hasElse)
        emitStatements(orelse);
      return;
    }
  }
  // Loop-carried locals: pre-existing locals reassigned in the body (an
  // accumulator carried across iterations). Thread them as loop-header /
  // after-block arguments so the mutated value flows across the back-edge, and
  // release the replaced header value on the back-edge to keep ownership
  // balanced. Combining carried locals with break/continue is not yet
  // supported (break/continue would need to forward the carried values too).
  // ⭐ A for target that ALREADY EXISTS is a loop-carried local like any
  // other. CPython leaves the target bound after the loop; excluding every
  // target name from the carried set left the post-loop read seeing the
  // pre-loop value:
  //
  //     i = -1
  //     for i in range(1):
  //         pass
  //     print(i)      # printed -1; CPython prints 0
  //
  // A name the loop INTRODUCES stays excluded: with zero iterations CPython
  // has nothing bound there either (NameError), and this walk has no value to
  // carry, so the post-loop read keeps its present refusal rather than
  // inventing one.
  llvm::StringSet<> targetNames;
  llvm::SmallVector<mlir::Value, 4> carriedInitial;
  llvm::SmallVector<CarriedLoopLocal, 4> carried =
      collectCarriedLoopLocals(statement, /*excludedNames=*/nullptr,
                               carriedInitial);

  Value iterable = emitExpr(ast::node(statement, "iter"));
  // ⭐ A SOURCE class's __iter__ is called, not looked up in the manifest.
  // py.iter resolves its target against the runtime manifest, so
  // `for v in Box(...)` over a class that defines __iter__ died in the
  // lowering as "runtime manifest has no Box.__iter__ method" -- while
  // __len__, __getitem__ and __contains__ on the same class all worked, and
  // so did `async for` over a class's __aiter__, which is this same shape
  // twelve hundred lines down.
  Value concreteIterable = stripLocalProtocolView(iterable);
  Value methodIterable = concreteIterable.value ? concreteIterable : iterable;
  Value iteratorValue;
  mlir::Type iteratorType;
  bool iterRefused = false;
  std::optional<Value> sourceIterator =
      tryEmitClassDunder(statement, methodIterable, "__iter__", {},
                         &iterRefused);
  if (iterRefused)
    return;
  if (sourceIterator) {
    iteratorValue = *sourceIterator;
    iteratorType = iteratorValue.type;
  } else {
    CallInferenceResult iterInference =
        types.inferMethodCallWithEvidence(iterable.type, "__iter__", {});
    if (!requireStaticEvidence(statement, iterInference))
      return;
    iteratorType = iterInference.resultType;
    mlir::UnitAttr returnedSelf = iteratorType == iterable.type
                                      ? builder.getUnitAttr()
                                      : mlir::UnitAttr();
    auto iterator = py::IterOp::create(
        builder, loc(statement), iteratorType, "__iter__",
        callProtocolFor(iterInference), iterable.value, returnedSelf);
    iteratorValue = Value{iterator.getResult(), iteratorType};
  }
  CallInferenceResult nextInference =
      types.inferMethodCallWithEvidence(iteratorType, "__next__", {});
  if (!requireStaticEvidence(statement, nextInference))
    return;
  mlir::Type elem = nextInference.resultType;

  mlir::Block *entry = builder.getInsertionBlock();
  mlir::Region *region = entry->getParent();
  mlir::Block *afterBlock = entry->splitBlock(builder.getInsertionPoint());
  mlir::Block *checkBlock =
      builder.createBlock(region, afterBlock->getIterator());
  mlir::Block *bodyBlock =
      builder.createBlock(region, afterBlock->getIterator());
  // else: the no-break exhaustion edge runs the else body before joining the
  // after-block; break edges skip it by targeting the after-block directly.
  mlir::Block *elseBlock =
      hasElse ? builder.createBlock(region, afterBlock->getIterator())
              : nullptr;
  // Without a break the after-block's one predecessor is the header, whose
  // arguments dominate it: forwarding them again through after-block
  // arguments would give those arguments a single incoming edge that sits on
  // the bufferization inference stack whenever the header is being resolved,
  // making shaped-primitive buffer types order-dependent.
  bool breakForwardsCarried =
      containsLoopLevelBreak(ast::nodeList(statement, "body"));
  bool afterForwardsCarried = breakForwardsCarried || hasElse;
  for (const CarriedLoopLocal &local : carried) {
    checkBlock->addArgument(local.type, loc(statement));
    if (elseBlock)
      elseBlock->addArgument(local.type, loc(statement));
    if (afterForwardsCarried)
      afterBlock->addArgument(local.type, loc(statement));
  }

  builder.setInsertionPointToEnd(entry);
  acquireUnionCarriedTokens(builder, loc(statement), carried, carriedInitial);
  mlir::cf::BranchOp::create(builder, loc(statement), checkBlock, carriedInitial);

  builder.setInsertionPointToStart(checkBlock);
  bindCarriedLoopLocals(carried, checkBlock);
  llvm::SmallVector<mlir::Value, 4> checkArgs;
  if (afterForwardsCarried)
    for (unsigned index = 0; index < carried.size(); ++index)
      checkArgs.push_back(checkBlock->getArgument(index));
  // The next op carries the iterator expression's location, not the whole
  // for statement: an exception surfacing from the iterator anchors the
  // traceback carets under that expression, matching CPython's FOR_ITER
  // instruction position.
  const parser::Node *iterLocNode = ast::node(statement, "iter");
  mlir::Location nextLoc = iterLocNode ? loc(*iterLocNode) : loc(statement);
  auto next = py::NextOp::create(
      builder, nextLoc, elem, builder.getI1Type(), iteratorType,
      "__next__", callProtocolFor(nextInference), iteratorValue.value);
  mlir::Block *exhaustedTarget = elseBlock ? elseBlock : afterBlock;
  mlir::cf::CondBranchOp::create(builder, loc(statement), next.getValid(),
                                 bodyBlock, mlir::ValueRange{},
                                 exhaustedTarget, checkArgs);

  builder.setInsertionPointToStart(bodyBlock);
  {
    ScopedEmitterScope scope(values, types);
    bindCarriedLoopLocals(carried, checkBlock);
    LoopControlContext loop{afterBlock, checkBlock};
    loop.carriedLocals.assign(carried.begin(), carried.end());
    loop.headerBlock = checkBlock;
    loopControlContexts.push_back(loop);
    emitAssignTarget(*ast::node(statement, "target"),
                     Value{next.getElement(), elem});
    emitStatements(ast::nodeList(statement, "body"));
    loopControlContexts.pop_back();
    if (!insertionBlockTerminated(builder))
      mlir::cf::BranchOp::create(
          builder, loc(statement), checkBlock,
          carriedLoopEdgeOperands(statement, carried, checkBlock));
  }

  if (elseBlock) {
    builder.setInsertionPointToStart(elseBlock);
    // ⛔ Why the else body is NOT scoped when the loop cannot break: a name
    // FIRST bound in the else has no pre-loop value to carry, so the carried
    // machinery cannot thread it -- but without a break the else block is the
    // after-block's only predecessor, so it dominates every later read and
    // the value needs no edge at all. Scoping it away made `else: fresh = 5`
    // followed by `return fresh` read as "unresolved name 'fresh'" where
    // CPython prints 5. With a break in the loop the scope stays: that path
    // skips the else, and CPython raises NameError there.
    std::optional<ScopedEmitterScope> scope;
    if (breakForwardsCarried)
      scope.emplace(values, types);
    bindCarriedLoopLocals(carried, elseBlock);
    emitStatements(orelse);
    if (!insertionBlockTerminated(builder))
      mlir::cf::BranchOp::create(
          builder, loc(statement), afterBlock,
          carriedLoopEdgeOperands(statement, carried, elseBlock, {},
                                  /*toHeader=*/false));
  }

  builder.setInsertionPointToStart(afterBlock);
  bindCarriedLoopLocals(carried,
                        afterForwardsCarried ? afterBlock : checkBlock);
}

void ModuleEmitter::emitWhile(const parser::Node &statement) {
  const auto *orelse = ast::nodeList(statement, "orelse");
  bool hasElse = orelse && !orelse->empty();
  const parser::Node *test = ast::node(statement, "test");
  // A walrus in the condition rebinds locals in the loop HEADER, which the
  // carried-local machinery (built around body assignments) cannot balance.
  // Desugar to the equivalent body-assignment form instead:
  //   while TEST: BODY else: ELSE
  //   ==> while True:
  //         if TEST: BODY
  //         else: ELSE; break
  // (break/continue in BODY still target the while; break in BODY correctly
  // skips ELSE.)
  if (test && containsNamedExpr(test)) {
    const parser::Field *testField = parser::findField(statement, "test");
    const parser::Field *bodyField = parser::findField(statement, "body");
    if (testField && bodyField &&
        std::holds_alternative<parser::NodePtr>(testField->value) &&
        std::holds_alternative<std::vector<parser::NodePtr>>(
            bodyField->value)) {
      parser::NodePtr guard = parser::makeNode("If", statement.range);
      parser::addField(*guard, "test",
                       std::get<parser::NodePtr>(testField->value));
      parser::addField(
          *guard, "body",
          std::get<std::vector<parser::NodePtr>>(bodyField->value));
      std::vector<parser::NodePtr> exitBody;
      if (orelse)
        exitBody.assign(orelse->begin(), orelse->end());
      exitBody.push_back(parser::makeNode("Break", statement.range));
      parser::addField(*guard, "orelse", std::move(exitBody));

      parser::NodePtr trueConstant = synth::boolConstant(true, statement.range);
      parser::NodePtr loop = parser::makeNode("While", statement.range);
      parser::addField(*loop, "test", trueConstant);
      parser::addField(*loop, "body", std::vector<parser::NodePtr>{guard});
      parser::addField(*loop, "orelse", std::vector<parser::NodePtr>{});
      emitWhile(*loop);
      return;
    }
  }
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
  llvm::SmallVector<mlir::Value, 4> carriedInitial;
  llvm::SmallVector<CarriedLoopLocal, 4> carried =
      collectCarriedLoopLocals(statement, /*excludedNames=*/nullptr,
                               carriedInitial);

  mlir::Block *entry = builder.getInsertionBlock();
  mlir::Region *region = entry->getParent();
  mlir::Block *afterBlock = entry->splitBlock(builder.getInsertionPoint());
  mlir::Block *headerBlock =
      builder.createBlock(region, afterBlock->getIterator());
  mlir::Block *bodyBlock =
      builder.createBlock(region, afterBlock->getIterator());
  // else: the condition-false edge runs the else body before joining the
  // after-block; break edges skip it by targeting the after-block directly.
  mlir::Block *elseBlock =
      hasElse ? builder.createBlock(region, afterBlock->getIterator())
              : nullptr;
  // Without a break the after-block's one predecessor is the header, whose
  // arguments dominate it: forwarding them again through after-block
  // arguments would give those arguments a single incoming edge that sits on
  // the bufferization inference stack whenever the header is being resolved,
  // making shaped-primitive buffer types order-dependent.
  bool breakForwardsCarried =
      containsLoopLevelBreak(ast::nodeList(statement, "body"));
  bool afterForwardsCarried = breakForwardsCarried || hasElse;
  for (const CarriedLoopLocal &local : carried) {
    headerBlock->addArgument(local.type, loc(statement));
    if (elseBlock)
      elseBlock->addArgument(local.type, loc(statement));
    if (afterForwardsCarried)
      afterBlock->addArgument(local.type, loc(statement));
  }

  builder.setInsertionPointToEnd(entry);
  acquireUnionCarriedTokens(builder, loc(statement), carried, carriedInitial);
  mlir::cf::BranchOp::create(builder, loc(statement), headerBlock,
                             carriedInitial);

  // Header: bind carried locals, evaluate the condition, and on false forward
  // the current header values to the else block (when present) or the
  // after-block.
  builder.setInsertionPointToStart(headerBlock);
  bindCarriedLoopLocals(carried, headerBlock);
  mlir::Value condition = emitBoolValue(emitExpr(test), statement);
  // A walrus in the condition may have rebound carried locals in the header:
  // the replaced header argument is released HERE (the rebinding happened
  // unconditionally), and every edge leaving the header forwards / compares
  // against the post-condition value instead of the raw argument.
  llvm::SmallVector<mlir::Value, 4> postTestValues;
  postTestValues.reserve(carried.size());
  for (auto [index, local] : llvm::enumerate(carried)) {
    mlir::Value argument = headerBlock->getArgument(index);
    mlir::Value current = argument;
    if (auto found = values.find(local.name);
        found != values.end() && found->second.value)
      current = coerceValue(found->second, local.type, statement).value;
    if (current != argument && py::isPyContractType(local.type) &&
        !derivesViaStructuralMutation(current, argument))
      py::DecRefOp::create(builder, loc(statement), argument);
    postTestValues.push_back(current);
  }
  llvm::SmallVector<mlir::Value, 4> headerArgs;
  if (afterForwardsCarried)
    headerArgs.append(postTestValues.begin(), postTestValues.end());
  mlir::Block *conditionFalseTarget = elseBlock ? elseBlock : afterBlock;
  mlir::cf::CondBranchOp::create(builder, loc(statement), condition, bodyBlock,
                                 mlir::ValueRange{}, conditionFalseTarget,
                                 headerArgs);

  builder.setInsertionPointToStart(bodyBlock);
  {
    ScopedEmitterScope scope(values, types);
    for (auto [index, local] : llvm::enumerate(carried)) {
      values[local.name] = Value{postTestValues[index], local.type};
      types.bindSymbol(local.name, local.type);
    }
    // The body sees what the condition proves: `while n is not None:`
    // narrows n for its own body, the same fact the if statement and the
    // conditional expression apply, through the same applyBranchNarrowing.
    if (test)
      if (std::optional<BranchTypeNarrowing> narrowing =
              optionalBranchTypeNarrowing(*test, types, module))
        applyBranchNarrowing(statement, *narrowing, /*conditionIsTrue=*/true);
    // ⛔ What is still missing, measured: the OTHER half of the same
    // ownership hole. The acquisition above balances the per-iteration
    // release; nothing releases the LAST incarnation on the exit edge,
    // because that release is the same `g.condition` skip in
    // insertOwnedBlockArgumentReleases. So a loop that rebinds a union
    // carried local to a freshly owned value --
    //
    //     while n is not None:
    //         total += n
    //         n = n - 1          # or `v = "done"` for a str member
    //
    // -- is refused with "owned resource from @LyLong_Sub result 0 reaches
    // function exit without release". It is a refusal, not a wrong answer,
    // and the shapes that rebind to None or to another borrowed union run.
    LoopControlContext loop{afterBlock, headerBlock};
    loop.carriedLocals.assign(carried.begin(), carried.end());
    loop.headerBlock = headerBlock;
    loop.baselineValues.assign(postTestValues.begin(), postTestValues.end());
    loopControlContexts.push_back(loop);
    emitStatements(ast::nodeList(statement, "body"));
    loopControlContexts.pop_back();
    if (!insertionBlockTerminated(builder))
      mlir::cf::BranchOp::create(
          builder, loc(statement), headerBlock,
          carriedLoopEdgeOperands(statement, carried, headerBlock,
                                  postTestValues));
  }

  if (elseBlock) {
    builder.setInsertionPointToStart(elseBlock);
    // ⛔ Why the else body is NOT scoped when the loop cannot break: a name
    // FIRST bound in the else has no pre-loop value to carry, so the carried
    // machinery cannot thread it -- but without a break the else block is the
    // after-block's only predecessor, so it dominates every later read and
    // the value needs no edge at all. Scoping it away made `else: fresh = 5`
    // followed by `return fresh` read as "unresolved name 'fresh'" where
    // CPython prints 5. With a break in the loop the scope stays: that path
    // skips the else, and CPython raises NameError there.
    std::optional<ScopedEmitterScope> scope;
    if (breakForwardsCarried)
      scope.emplace(values, types);
    bindCarriedLoopLocals(carried, elseBlock);
    emitStatements(orelse);
    if (!insertionBlockTerminated(builder))
      mlir::cf::BranchOp::create(
          builder, loc(statement), afterBlock,
          carriedLoopEdgeOperands(statement, carried, elseBlock, {},
                                  /*toHeader=*/false));
  }

  builder.setInsertionPointToStart(afterBlock);
  releaseUnionCarriedTokens(builder, loc(statement), carried, afterBlock,
                            afterForwardsCarried, postTestValues);
  if (afterForwardsCarried) {
    bindCarriedLoopLocals(carried, afterBlock);
  } else {
    for (auto [index, local] : llvm::enumerate(carried)) {
      values[local.name] = Value{postTestValues[index], local.type};
      types.bindSymbol(local.name, local.type);
    }
  }
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

  llvm::SmallVector<mlir::Value, 4> carriedInitialValues;
  llvm::SmallVector<CarriedLoopLocal, 4> carriedLocals =
      collectCarriedLoopLocals(statement, /*excludedNames=*/nullptr,
                               carriedInitialValues);

  mlir::Block *afterBlock = entryBlock->splitBlock(builder.getInsertionPoint());
  mlir::Block *loopBlock =
      builder.createBlock(region, afterBlock->getIterator());
  for (const CarriedLoopLocal &local : carriedLocals) {
    loopBlock->addArgument(local.type, loc(statement));
    afterBlock->addArgument(local.type, loc(statement));
  }
  builder.setInsertionPointToEnd(entryBlock);
  acquireUnionCarriedTokens(builder, loc(statement), carriedLocals,
                            carriedInitialValues);
  mlir::cf::BranchOp::create(builder, loc(statement), loopBlock,
                             carriedInitialValues);
  builder.setInsertionPointToStart(loopBlock);
  bindCarriedLoopLocals(carriedLocals, loopBlock);

  mlir::OperationState tryState(loc(statement), py::TryOp::getOperationName());
  tryState.addTypes(builder.getI1Type());
  for (const CarriedLoopLocal &local : carriedLocals)
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
      llvm::SmallVector<mlir::Value, 4> carriedOperands =
          carriedLoopEdgeOperands(statement, carriedLocals, loopBlock);
      yieldValues.append(carriedOperands.begin(), carriedOperands.end());
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
  bindCarriedLoopLocals(carriedLocals, afterBlock);
}

} // namespace lython::emitter
