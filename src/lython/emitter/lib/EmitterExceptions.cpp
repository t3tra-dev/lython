#include "EmitterCore.h"
#include "EmitterPyOps.h"
#include "EmitterSupport.h"

#include "AstAccess.h"
#include "Contracts.h"

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

using py::contracts::isIntegerLiteralSpelling;

enum class FinallyCompletion {
  Fallthrough,
  Return,
  Break,
  Continue,
};

bool isSupportedFinallyReturnCarrierType(mlir::Type type) {
  if (!type)
    return false;
  if (auto literal = mlir::dyn_cast<py::LiteralType>(type)) {
    llvm::StringRef spelling = literal.getSpelling();
    return spelling == "None" || spelling == "True" || spelling == "False" ||
           isIntegerLiteralSpelling(spelling) ||
           (spelling.size() >= 2 && spelling.front() == '"' &&
            spelling.back() == '"');
  }
  if (auto contract = mlir::dyn_cast<py::ContractType>(type)) {
    llvm::StringRef name = contract.getContractName();
    return name == "types.NoneType" || name == "builtins.bool" ||
           name == "builtins.int" || name == "builtins.float" ||
           name == "builtins.str" || name == "builtins.object";
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

// Names a statement list rebinds through a NAME target. Deliberately narrower
// than `collectAssignedNames`, which also reports the receiver of a
// structural mutation (`xs.append(v)`, `d[k] = v`) because that rebinds the
// receiver's SSA value too. Why the narrow set here: those receivers must
// keep their sequence/mapping evidence, and evidence lives on the SSA value,
// not in storage -- promoting one to a cell turns its in-place mutation into
// a rejection ("in-place dict assignment requires a box-fronted field
// container"). Structural mutation inside a try keeps travelling the post-try
// result lanes instead.
void collectRebindNames(const parser::Node *node, llvm::StringSet<> &names) {
  if (!node)
    return;
  if (node->kind == "FunctionDef" || node->kind == "AsyncFunctionDef" ||
      node->kind == "ClassDef" || node->kind == "Lambda")
    return;
  if (node->kind == "Assign") {
    if (const auto *targets = ast::nodeList(*node, "targets"))
      for (const parser::NodePtr &target : *targets)
        collectAssignedNameTargets(target.get(), names);
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
        collectRebindNames(child->get(), names);
    } else if (const auto *children =
                   std::get_if<std::vector<parser::NodePtr>>(&field.value)) {
      for (const parser::NodePtr &child : *children)
        if (child)
          collectRebindNames(child.get(), names);
    }
  }
}

void collectRebindNames(const std::vector<parser::NodePtr> *statements,
                        llvm::StringSet<> &names) {
  if (!statements)
    return;
  for (const parser::NodePtr &statement : *statements)
    collectRebindNames(statement.get(), names);
}

// The complement of the above within `collectAssignedNames`: receivers whose
// SSA value a structural mutation rebinds in place. These write
// `values[name]` directly (the `ly.structural_mutation` call's second
// result), which would overwrite a storage promotion instead of storing
// through it -- so a name in this set is never promoted.
void collectStructuralReceiverNames(const parser::Node *node,
                                    llvm::StringSet<> &names) {
  if (!node)
    return;
  if (node->kind == "FunctionDef" || node->kind == "AsyncFunctionDef" ||
      node->kind == "ClassDef" || node->kind == "Lambda")
    return;
  if (node->kind == "Call") {
    if (const parser::Node *func = ast::node(*node, "func"))
      if (func->kind == "Attribute")
        if (const parser::Node *value = ast::node(*func, "value"))
          if (value->kind == "Name")
            names.insert(ast::nameSpelling(*value));
  } else if (node->kind == "Assign") {
    if (const auto *targets = ast::nodeList(*node, "targets"))
      for (const parser::NodePtr &target : *targets)
        if (target && target->kind == "Subscript")
          if (const parser::Node *container = ast::node(*target, "value"))
            if (container->kind == "Name")
              names.insert(ast::nameSpelling(*container));
  } else if (node->kind == "Delete") {
    if (const auto *targets = ast::nodeList(*node, "targets"))
      for (const parser::NodePtr &target : *targets)
        if (target && target->kind == "Subscript")
          if (const parser::Node *container = ast::node(*target, "value"))
            if (container->kind == "Name")
              names.insert(ast::nameSpelling(*container));
  }
  for (const parser::Field &field : node->fields) {
    if (const auto *child = std::get_if<parser::NodePtr>(&field.value)) {
      if (*child)
        collectStructuralReceiverNames(child->get(), names);
    } else if (const auto *children =
                   std::get_if<std::vector<parser::NodePtr>>(&field.value)) {
      for (const parser::NodePtr &child : *children)
        if (child)
          collectStructuralReceiverNames(child.get(), names);
    }
  }
}

void collectStructuralReceiverNames(
    const std::vector<parser::NodePtr> *statements, llvm::StringSet<> &names) {
  if (!statements)
    return;
  for (const parser::NodePtr &statement : *statements)
    collectStructuralReceiverNames(statement.get(), names);
}

// Name targets of an augmented assignment. On a container these desugar to
// the structural mutator (`d |= o` is `d.update(o)`), which rebinds through
// the same direct `values[name]` write a promotion cannot see; on a scalar
// the same spelling is an ordinary rebind. The caller resolves which by the
// operand's contract, so the two cannot be merged into one set here.
void collectAugAssignTargetNames(const parser::Node *node,
                                 llvm::StringSet<> &names) {
  if (!node)
    return;
  if (node->kind == "FunctionDef" || node->kind == "AsyncFunctionDef" ||
      node->kind == "ClassDef" || node->kind == "Lambda")
    return;
  if (node->kind == "AugAssign")
    collectAssignedNameTargets(ast::node(*node, "target"), names);
  for (const parser::Field &field : node->fields) {
    if (const auto *child = std::get_if<parser::NodePtr>(&field.value)) {
      if (*child)
        collectAugAssignTargetNames(child->get(), names);
    } else if (const auto *children =
                   std::get_if<std::vector<parser::NodePtr>>(&field.value)) {
      for (const parser::NodePtr &child : *children)
        if (child)
          collectAugAssignTargetNames(child.get(), names);
    }
  }
}

void collectAugAssignTargetNames(const std::vector<parser::NodePtr> *statements,
                                 llvm::StringSet<> &names) {
  if (!statements)
    return;
  for (const parser::NodePtr &statement : *statements)
    collectAugAssignTargetNames(statement.get(), names);
}

// Contracts whose augmented assignment goes through a structural mutator.
bool isMutableContainerContract(mlir::Type type) {
  auto contract = mlir::dyn_cast_if_present<py::ContractType>(type);
  if (!contract)
    return false;
  llvm::StringRef name = contract.getContractName();
  return name == "builtins.list" || name == "builtins.dict" ||
         name == "builtins.set" || name == "builtins.frozenset" ||
         name == "builtins.bytearray";
}

} // namespace

mlir::Type ModuleEmitter::postTryLaneCarrierType(mlir::Type type) const {
  mlir::Type widened = types.widenLiteral(type);
  auto contract = mlir::dyn_cast_if_present<py::ContractType>(widened);
  if (!contract)
    return {};
  // Why exceptions are excluded even though they are objects like the rest:
  // an `except E as e` binding is a BORROW of the current-exception slot
  // (lowerExceptCurrentValue hands out OwnershipKind::Borrow), and the handler
  // exit that would feed the lane runs LyEH_DiscardCurrentException BEFORE the
  // branch that carries it. A lane therefore publishes a pointer whose last
  // reference the discard just dropped: measured as an empty str() under the
  // system allocator and SIGSEGV under libgmalloc for a user exception class,
  // which reached here as an ordinary user class. Storage carries them
  // instead -- an aggregate slot store retains, and it retains inside the
  // handler, ahead of the discard.
  if (isExceptionContractType(widened))
    return {};
  llvm::StringRef name = contract.getContractName();
  if (name == "builtins.int" || name == "builtins.str" ||
      name == "builtins.bool" || name == "builtins.float" ||
      name == "builtins.tuple" || name == "builtins.list" ||
      name == "builtins.dict" || name == "builtins.set" ||
      name == "builtins.frozenset" || name == "builtins.bytes")
    return widened;
  // A user class instance carries the same way. Why not leave it out:
  // a non-carrier candidate used to be skipped SILENTLY, so `obj = C(...)`
  // inside a handler was discarded and the post-try read answered the
  // pre-try object -- and a desugared enum member or NamedTuple is a
  // user class, which makes that the common shape rather than a corner.
  if (classFieldOrders.contains(name))
    return widened;
  return {};
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
  // without a finally the flags simply dispatch right after the op. An else
  // block coexists: its normal-completion flag stays result 0 and the
  // completion flags follow it.
  const auto *handlersForEligibility = ast::nodeList(statement, "handlers");
  bool completionEligible =
      hasFinally ||
      (handlersForEligibility && !handlersForEligibility->empty());
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
    // CPython's evaluation order (body -> handlers/else -> finally) nests
    // exactly, so the combined form desugars instead of teaching the
    // mutually-exclusive else / finally-completion op suffixes about each
    // other:
    //   try: B except...: H else: E finally: F
    //   ==> try: (try: B except...: H else: E) finally: F
    const parser::Field *bodyField = parser::findField(statement, "body");
    const parser::Field *handlersField =
        parser::findField(statement, "handlers");
    const parser::Field *orelseField = parser::findField(statement, "orelse");
    const parser::Field *finalField = parser::findField(statement, "finalbody");
    if (bodyField && handlersField && orelseField && finalField &&
        std::holds_alternative<std::vector<parser::NodePtr>>(
            bodyField->value) &&
        std::holds_alternative<std::vector<parser::NodePtr>>(
            handlersField->value) &&
        std::holds_alternative<std::vector<parser::NodePtr>>(
            orelseField->value) &&
        std::holds_alternative<std::vector<parser::NodePtr>>(
            finalField->value)) {
      parser::NodePtr inner = parser::makeNode("Try", statement.range);
      parser::addField(*inner, "body",
                       std::get<std::vector<parser::NodePtr>>(bodyField->value));
      parser::addField(
          *inner, "handlers",
          std::get<std::vector<parser::NodePtr>>(handlersField->value));
      parser::addField(
          *inner, "orelse",
          std::get<std::vector<parser::NodePtr>>(orelseField->value));
      parser::addField(*inner, "finalbody", std::vector<parser::NodePtr>{});
      parser::NodePtr outer = parser::makeNode("Try", statement.range);
      parser::addField(*outer, "body", std::vector<parser::NodePtr>{inner});
      parser::addField(*outer, "handlers", std::vector<parser::NodePtr>{});
      parser::addField(*outer, "orelse", std::vector<parser::NodePtr>{});
      parser::addField(
          *outer, "finalbody",
          std::get<std::vector<parser::NodePtr>>(finalField->value));
      emitTry(*outer);
      return;
    }
    diagnostics.push_back(
        parser::Diagnostic{parser::Severity::Error, statement.range.start,
                           "malformed try/else/finally statement"});
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
  if (supportsLoopControlThroughFinally &&
      !loopControlContexts.back().carriedLocals.empty()) {
    // The break/continue completion branches after the op cannot see the try
    // region's SSA values, so they could only forward STALE pre-try carried
    // values — reject instead of silently mis-executing.
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, statement.range.start,
        "break/continue through try/finally in a loop with carried "
        "(reassigned) locals is not implemented yet"});
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

  // A local the try body REBINDS is observed by the handler, by the finally
  // body, and by the continuation with the value it held at the raise point.
  // For the extent of the statement such a local is promoted out of SSA into
  // an R6 cell: the body's rebinds become stores, and every later read is a
  // load.
  //
  // Why NOT a single static binding at the handler entry (block argument,
  // extra py.try result, or "the value at the end of the body"): CPython
  // makes that unrepresentable. Two raise points in one try body make the
  // SAME handler answer differently depending on which fired
  // (`n=2; if c: raise; n=3; raise` yields 2 or 3), and a store AFTER the
  // raise is not observed at all (`x=2; raise; x=3` yields 2). The merge is
  // therefore per-raise-point, and its join must be a memory cell rather
  // than a phi -- the emitter has no phi to write into, because the unwind
  // edge is a dynamic marker edge here and only becomes a real landing pad
  // in the LLVM cleanup phase (Passes/Runtime/Cleanup/EH.cpp).
  //
  // Why the R6 cell and not a fresh construct: a cell is an ordinary
  // one-field class, so the aggregate slot contract of
  // rfc/memory-safety-proof.md already covers the store (retain new,
  // release previous) and the affine verifier needs no new rule.
  //
  // The promotion is undone right after the statement so that the
  // continuation keeps the SSA fast paths (int lanes, list/sequence
  // evidence); leaving a container cell-bound would demote every later
  // `xs.append(...)` into a non-evidence-backed receiver.
  //
  // The same promotion covers a rebind in a HANDLER, the ELSE body or the
  // FINALLY body. Those bodies are each emitted under a ScopedEmitterScope
  // that restores `values` wholesale, so a rebind inside one reaches the
  // continuation only through storage or through a lane; with an else or a
  // finally present there are no lanes, and the rebind was silently dropped
  // (`seen = "unset"` after `except: seen = "handled"; finally: ...`, and
  // `outcome` unchanged by an else body). Where a lane CAN carry the name it
  // stays in charge: a lane keeps the value in SSA, which the continuation
  // prefers. Where it cannot -- an exception entity is the case that brought
  // this here, and postTryLaneCarrierType says why -- storage is the only
  // channel, so the name promotes even though lanes exist for its neighbours.
  bool postTryLanesAvailable = !hasElse && !hasFinally &&
                               !usesFinallyCompletion && handlers &&
                               !handlers->empty();
  llvm::SmallVector<std::string, 4> storagePromotedNames;
  // Names left to the lanes on purpose: pre-existing bindings a handler rebinds
  // whose type a lane CAN carry, mapped to the value they held before the
  // statement. Checked again once the lanes are known, because a lane is also
  // dropped for reasons only the emitted region shape shows, and then nothing
  // carries the rebind at all. The pre-try value is what tells a dropped lane
  // apart from a handler that never changed the binding.
  llvm::StringMap<mlir::Value> laneRequiredNames;
  if (hasFinally || (handlers && !handlers->empty())) {
    llvm::StringSet<> reboundInBody;
    collectRebindNames(ast::nodeList(statement, "body"), reboundInBody);
    llvm::StringSet<> reboundOutsideBody;
    if (handlers)
      for (const parser::NodePtr &handler : *handlers)
        if (handler)
          collectRebindNames(ast::nodeList(*handler, "body"),
                             reboundOutsideBody);
    collectRebindNames(ast::nodeList(statement, "orelse"), reboundOutsideBody);
    collectRebindNames(finalbody, reboundOutsideBody);
    for (const auto &entry : reboundOutsideBody) {
      llvm::StringRef name = entry.getKey();
      if (postTryLanesAvailable) {
        auto bound = values.find(name);
        if (bound == values.end() || !bound->second.value)
          continue;
        if (postTryLaneCarrierType(bound->second.type)) {
          laneRequiredNames[name] = bound->second.value;
          continue;
        }
      }
      reboundInBody.insert(name);
    }
    // A structural mutation ANYWHERE in the statement disqualifies the name:
    // in the body it would need the cell to carry evidence it cannot, and in
    // a handler or the finally body its direct `values[name]` write would
    // silently bypass the cell the continuation then reads.
    llvm::StringSet<> structuralReceivers;
    collectStructuralReceiverNames(ast::nodeList(statement, "body"),
                                   structuralReceivers);
    if (handlers)
      for (const parser::NodePtr &handler : *handlers)
        if (handler)
          collectStructuralReceiverNames(ast::nodeList(*handler, "body"),
                                         structuralReceivers);
    collectStructuralReceiverNames(finalbody, structuralReceivers);
    collectStructuralReceiverNames(ast::nodeList(statement, "orelse"),
                                   structuralReceivers);
    llvm::StringSet<> augAssignTargets;
    collectAugAssignTargetNames(ast::nodeList(statement, "body"),
                                augAssignTargets);
    if (handlers)
      for (const parser::NodePtr &handler : *handlers)
        if (handler)
          collectAugAssignTargetNames(ast::nodeList(*handler, "body"),
                                      augAssignTargets);
    collectAugAssignTargetNames(finalbody, augAssignTargets);
    collectAugAssignTargetNames(ast::nodeList(statement, "orelse"),
                                augAssignTargets);
    llvm::SmallVector<llvm::StringRef, 4> orderedNames;
    for (const auto &entry : reboundInBody)
      orderedNames.push_back(entry.getKey());
    llvm::sort(orderedNames);
    for (llvm::StringRef name : orderedNames) {
      if (structuralReceivers.contains(name))
        continue;
      auto bound = values.find(name);
      // Not bound before the try: the handler cannot observe a value the body
      // may never have produced (CPython raises UnboundLocalError there), so
      // there is nothing to merge. Such a name still travels the post-try
      // lanes below, which require it bound on every way out.
      if (bound == values.end() || !bound->second.value)
        continue;
      // Already storage: a nonlocal-shared cell reads and writes through the
      // same channel, so the shape is correct as it stands.
      if (isCellContract(bound->second.type))
        continue;
      mlir::Type content = types.widenLiteral(bound->second.type);
      // A cell field needs a contract-typed slot. A union (isinstance
      // narrowing) and a primitive tensor have none, so the rebind cannot
      // reach the handler or the continuation at all -- both would read the
      // pre-try value. Reject instead of answering with it.
      if (!mlir::isa_and_nonnull<py::ContractType>(content)) {
        std::string typeText;
        {
          llvm::raw_string_ostream stream(typeText);
          stream << content;
        }
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, statement.range.start,
            "local '" + name.str() +
                "' is reassigned inside this try and its type " + typeText +
                " cannot be carried out of the statement; bind the "
                "reassignment to a new name inside the block, or narrow the "
                "local to a single type before the try"});
        continue;
      }
      // `xs += other` / `d |= other` on a container is the structural mutator
      // in disguise, so it belongs with the receivers above.
      if (isMutableContainerContract(content) && augAssignTargets.contains(name))
        continue;
      // A loop-carried local arrives as a loop block argument, and moving that
      // incarnation's token into an aggregate slot inside the same iteration
      // is mis-tracked downstream: the release insertion accepts it and the
      // program then double-frees (a segfault, not a diagnostic). Until that
      // hole is closed the name keeps the post-try result lanes, which handle
      // the loop shape. Detected by name against every enclosing loop's
      // carried set rather than by asking whether the value is a BlockArgument,
      // because a promoted name is re-bound to the cell before the next
      // enclosing try sees it.
      bool loopCarried = false;
      for (const LoopControlContext &context : loopControlContexts)
        for (const CarriedLoopLocal &carried : context.carriedLocals)
          if (carried.name == name)
            loopCarried = true;
      if (loopCarried) {
        // ... but with no lane able to carry the value either, the two
        // channels' exclusions meet and the rebind reaches nothing. The
        // ownership verifier does catch the exception-entity spelling of this
        // (a promotion it would accept is the double-free above), yet only
        // with verifiers ON: --release turns them off and the same program
        // crashes in the JIT. Rejecting here instead makes the answer the same
        // in both configurations.
        if (!postTryLaneCarrierType(content))
          diagnostics.push_back(parser::Diagnostic{
              parser::Severity::Error, statement.range.start,
              "local '" + name.str() +
                  "' is reassigned inside this try and is also carried by an "
                  "enclosing loop; its type cannot travel either channel out "
                  "of the statement. Bind the reassignment to a new name "
                  "inside the block, or move the try out of the loop"});
        continue;
      }
      storagePromotedNames.push_back(std::string(name));
    }
    // A receiver whose SSA value a structural mutation rebinds is not a lane
    // question: it writes `values[name]` directly and the container it names is
    // mutated in place, so the lane check below must not speak for it.
    for (const auto &entry : structuralReceivers)
      laneRequiredNames.erase(entry.getKey());
    for (const auto &entry : augAssignTargets)
      laneRequiredNames.erase(entry.getKey());
    for (const std::string &name : storagePromotedNames) {
      Value cell = emitCellAlloc(statement, values.find(name)->second);
      if (!isCellContract(cell.type))
        continue;
      values[name] = cell;
      types.bindSymbol(name, cellContentType(cell.type));
    }
  }
  llvm::StringSet<> storagePromoted;
  for (const std::string &name : storagePromotedNames) {
    storagePromoted.insert(name);
    // Promoted after all -- a rebind in the try BODY promotes the name whatever
    // its handler rebind could have done, and a promoted name is deliberately
    // kept out of the lanes, so it must not be held to having one.
    laneRequiredNames.erase(name);
  }

  postTryEligible = postTryLanesAvailable;
  if (postTryEligible) {
    llvm::StringSet<> assignedNames;
    collectAssignedNames(ast::nodeList(statement, "body"), assignedNames);
    for (const parser::NodePtr &handler : *handlers)
      if (handler)
        collectAssignedNames(ast::nodeList(*handler, "body"), assignedNames);
    for (const auto &entry : assignedNames) {
      // A promoted name's authority is its cell, not a result lane: a lane
      // would merge the cell POINTER (identical on every edge) and then the
      // unboxing below would read it twice.
      if (storagePromoted.contains(entry.getKey()))
        continue;
      postCandidateNames.push_back(entry.getKey().str());
    }
    llvm::sort(postCandidateNames);
    if (postCandidateNames.empty())
      postTryEligible = false;
  }

  mlir::OperationState state(loc(statement), py::TryOp::getOperationName());
  if (hasElse)
    state.addTypes(builder.getI1Type());
  if (usesFinallyCompletion) {
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
        // Early completions never run the else block.
        if (hasElse)
          appendBoolYield(yieldValues, false);
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
      if (hasElse && !usesFinallyCompletion) {
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
                if (hasElse)
                  appendBoolYield(yieldValues, true);
                if (usesFinallyCompletion)
                  appendFallthroughReturnPayload(yieldValues);
                else if (hasElse)
                  for (const ElseCarriedLocal &local : elseCarriedLocals)
                    yieldValues.push_back(local.value);
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
            // A generic exception class has one contract per instantiation and
            // no class of its own, so an unsubscripted handler has no single
            // class id to test. Named here rather than left to the class-id
            // lookup, which cannot say why the class is missing.
            if (auto typeObject =
                    mlir::dyn_cast_if_present<py::TypeType>(candidateType))
              if (candidate &&
                  rejectGenericClassObject(*candidate,
                                           typeObject.getInstanceType())) {
                handlerTypes.clear();
                handlerTypeLocs.clear();
                break;
              }
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
        // `except (A, B) as e`: the binding's static type is the nearest
        // common ancestor of the tuple members (the runtime object is
        // whichever matched; only the static view needs one nominal type).
        mlir::Type boundHandlerType = handlerTypes.front();
        if (handlerName && handlerTypes.size() != 1) {
          auto instanceOf = [&](mlir::Type type) -> mlir::Type {
            auto typeObject = mlir::dyn_cast<py::TypeType>(type);
            return typeObject ? typeObject.getInstanceType() : mlir::Type();
          };
          mlir::Type common = instanceOf(handlerTypes.front());
          for (mlir::Type candidate :
               llvm::ArrayRef<mlir::Type>(handlerTypes).drop_front()) {
            mlir::Type instance = instanceOf(candidate);
            if (!common || !instance) {
              common = {};
              break;
            }
            if (isAssignableWithStaticEvidence(instance, common, module))
              continue;
            if (isAssignableWithStaticEvidence(common, instance, module)) {
              common = instance;
              continue;
            }
            common = types.contract("builtins.BaseException");
          }
          if (!common) {
            diagnostics.push_back(parser::Diagnostic{
                parser::Severity::Error, handler.range.start,
                "except-as binding requires resolvable exception types"});
            continue;
          }
          boundHandlerType = types.typeObject(common);
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
            auto handlerType = mlir::cast<py::TypeType>(boundHandlerType);
            mlir::Type exceptionType = handlerType.getInstanceType();
            auto current = py::ExceptCurrentValueOp::create(
                               builder, loc(handler), exceptionType,
                               mlir::TypeAttr::get(boundHandlerType))
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
          // Recorded whether or not the lanes are still eligible: an empty
          // record set is also the fact "no handler falls through", which the
          // no-channel diagnostic below needs in order not to speak about a
          // continuation nothing reaches (a handler ending in `raise` observes
          // its own rebind nowhere).
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
        // all lanes carrier-typed (postTryLaneCarrierType, the same question
        // the promotion above asked); the lane type is the widened join.
        // A try body that always raises (`try: raise X` / every path returns)
        // contributes NO fall-through lane: its bindings are unreachable after
        // the try. Requiring one here dropped every lane for that shape, so
        // `out = 7` inside the handler was silently discarded and the post-try
        // read answered the pre-try value. The handler exits alone are the
        // complete set of ways out in that case.
        for (const std::string &name : postCandidateNames) {
          llvm::SmallVector<mlir::Type, 4> parts;
          if (postTryFallThrough) {
            auto tryBound = postTryEndBindings.find(name);
            if (tryBound == postTryEndBindings.end())
              continue;
            mlir::Type tryPart =
                postTryLaneCarrierType(tryBound->second.type);
            if (!tryPart)
              continue;
            parts.push_back(tryPart);
          }
          bool everywhere = true;
          for (const HandlerExit &exit : postHandlerExits) {
            auto found = exit.bindings.find(name);
            mlir::Type part =
                found != exit.bindings.end()
                    ? postTryLaneCarrierType(found->second.type)
                    : mlir::Type();
            if (!part) {
              everywhere = false;
              break;
            }
            parts.push_back(part);
          }
          if (!everywhere || parts.empty())
            continue;
          mlir::Type merged = types.join(parts);
          if (!merged || !postTryLaneCarrierType(merged))
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
                  if (hasElse)
                    appendBoolYield(yieldValues, false);
                  if (usesFinallyCompletion) {
                    appendFallthroughReturnPayload(yieldValues);
                  } else if (hasElse) {
                    // Inert defaults: the else block (the only reader of the
                    // carried lanes) is unreachable on this path.
                    for (const ElseCarriedLocal &local : elseCarriedLocals)
                      yieldValues.push_back(
                          emitDefaultReturnValue(local.type).value);
                  }
                }) > 0;
      }
    }
  }

  // The promotion above stood aside for these names because a lane can carry
  // their type; if the lanes then went away for a region-shape reason, nothing
  // carries the rebind and the continuation would answer the pre-try value.
  // Say so rather than answer it.
  llvm::SmallVector<llvm::StringRef, 4> laneRequiredOrder;
  for (const auto &entry : laneRequiredNames)
    laneRequiredOrder.push_back(entry.getKey());
  llvm::sort(laneRequiredOrder); // map iteration is hash order; diagnostics
                                 // must come out the same way every run
  for (llvm::StringRef name : laneRequiredOrder) {
    if (llvm::any_of(postCarriedLocals, [&](const PostCarriedLocal &local) {
          return local.name == name;
        }))
      continue;
    // Only a handler that FALLS THROUGH with a different value than the one the
    // name arrived with can be answered wrongly afterwards. A handler that
    // raises or returns instead observes its own rebind nowhere, and so does one
    // that left the binding alone -- diagnosing either would reject a program
    // that has no way to notice.
    mlir::Value beforeTry = laneRequiredNames.lookup(name);
    bool reachesContinuation = false;
    for (const HandlerExit &exit : postHandlerExits) {
      auto found = exit.bindings.find(name);
      if (found != exit.bindings.end() && found->second.value != beforeTry) {
        reachesContinuation = true;
        break;
      }
    }
    if (!reachesContinuation)
      continue;
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, statement.range.start,
        "local '" + name.str() +
            "' is reassigned inside an except handler of this try, and the "
            "shape of this statement leaves no way to carry the new value to "
            "the code after it; bind the reassignment to a new name inside the "
            "handler, or split the statement"});
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
    const unsigned flagBase = hasElse ? 1u : 0u;
    const unsigned returnFlagIndex = flagBase;
    const unsigned breakFlagIndex = flagBase + 1;
    const unsigned continueFlagIndex = flagBase + 2;
    const unsigned returnPayloadIndex = flagBase + 3;
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
  }
  if (hasElse) {
    mlir::Block *dispatchBlock = builder.getInsertionBlock();
    mlir::Block *afterElseBlock =
        dispatchBlock->splitBlock(builder.getInsertionPoint());
    mlir::Block *elseBlock = new mlir::Block;
    dispatchBlock->getParent()->getBlocks().insert(
        afterElseBlock->getIterator(), elseBlock);
    builder.setInsertionPointToEnd(dispatchBlock);
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
  // Undo the storage promotion: the statement is over, so the cell's content
  // is the one value the continuation can see, and rebinding to it lets the
  // cell die here instead of demoting the rest of the scope to loads.
  //
  // Why HERE and not right after py.try, which is where the block that still
  // holds the op would dominate the most: the else body and the finally
  // completion dispatch are emitted between the two points, and the else body
  // rebinds through the cell. Loading before it ran published the pre-else
  // value as the continuation's, which is exactly the silent drop the
  // promotion exists to prevent. This block is the single continuation every
  // path inside the statement branches to, so it dominates every later use;
  // nothing emitted in between reads a promoted name through SSA, because
  // while the promotion stands the name resolves to the cell.
  for (const std::string &name : storagePromotedNames) {
    auto bound = values.find(name);
    if (bound == values.end() || !isCellContract(bound->second.type))
      continue;
    Value content = emitCellLoad(statement, bound->second);
    values[name] = content;
    types.bindSymbol(name, content.type);
  }
}


// except* (PEP 654). The emitted except region differs fundamentally from
// the regular form: clauses are not mutually exclusive (each may consume a
// slice of one exception group), every clause body is wrapped in an inner
// collect-try (a raise inside a clause is parked, later clauses still run),
// and a trailing finish step rethrows whatever is left. The star frame
// bookkeeping lives in the runtime; here only the clause skeleton is built.
void ModuleEmitter::emitTryStar(const parser::Node &statement) {
  const auto *handlers = ast::nodeList(statement, "handlers");
  const auto *finalbody = ast::nodeList(statement, "finalbody");
  const auto *orelse = ast::nodeList(statement, "orelse");
  bool hasFinally = finalbody && !finalbody->empty();

  if (hasFinally) {
    // Same nesting CPython's evaluation order licenses for try/except/else/
    // finally: run the star clauses inside a plain try/finally.
    const parser::Field *bodyField = parser::findField(statement, "body");
    const parser::Field *handlersField =
        parser::findField(statement, "handlers");
    const parser::Field *orelseField = parser::findField(statement, "orelse");
    if (!bodyField || !handlersField || !orelseField ||
        !std::holds_alternative<std::vector<parser::NodePtr>>(
            bodyField->value) ||
        !std::holds_alternative<std::vector<parser::NodePtr>>(
            handlersField->value) ||
        !std::holds_alternative<std::vector<parser::NodePtr>>(
            orelseField->value)) {
      diagnostics.push_back(
          parser::Diagnostic{parser::Severity::Error, statement.range.start,
                             "malformed try/except*/finally statement"});
      return;
    }
    parser::NodePtr inner = parser::makeNode("TryStar", statement.range);
    parser::addField(*inner, "body",
                     std::get<std::vector<parser::NodePtr>>(bodyField->value));
    parser::addField(
        *inner, "handlers",
        std::get<std::vector<parser::NodePtr>>(handlersField->value));
    parser::addField(*inner, "orelse",
                     std::get<std::vector<parser::NodePtr>>(orelseField->value));
    parser::addField(*inner, "finalbody", std::vector<parser::NodePtr>{});
    parser::NodePtr outer = parser::makeNode("Try", statement.range);
    parser::addField(*outer, "body", std::vector<parser::NodePtr>{inner});
    parser::addField(*outer, "handlers", std::vector<parser::NodePtr>{});
    parser::addField(*outer, "orelse", std::vector<parser::NodePtr>{});
    parser::addField(
        *outer, "finalbody",
        std::vector<parser::NodePtr>(finalbody->begin(), finalbody->end()));
    emitTry(*outer);
    return;
  }
  if (orelse && !orelse->empty()) {
    diagnostics.push_back(
        parser::Diagnostic{parser::Severity::Error, statement.range.start,
                           "else with except* is not implemented yet"});
    return;
  }
  if (!handlers || handlers->empty()) {
    diagnostics.push_back(
        parser::Diagnostic{parser::Severity::Error, statement.range.start,
                           "except* requires at least one handler"});
    return;
  }
  if (containsReturnStatement(ast::nodeList(statement, "body")) ||
      containsBreakOrContinueStatement(ast::nodeList(statement, "body"))) {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, statement.range.start,
        "return/break/continue inside a try with except* is not implemented "
        "yet"});
    return;
  }
  // The star-frame regions run in isolated emitter scopes (unlike plain
  // try/except there are no post-carried result lanes yet), so a rebind of a
  // pre-existing local inside the body or a clause would silently revert
  // after the statement. Reject it loudly instead of mis-executing; fresh
  // names created inside are unaffected (a later outside read of one already
  // fails as unresolved).
  {
    llvm::StringSet<> rebound;
    collectAssignedNames(ast::nodeList(statement, "body"), rebound);
    if (handlers)
      for (const parser::NodePtr &handlerPtr : *handlers)
        if (handlerPtr)
          collectAssignedNames(ast::nodeList(*handlerPtr, "body"), rebound);
    for (const auto &entry : rebound) {
      llvm::StringRef name = entry.getKey();
      if (values.find(std::string(name)) == values.end())
        continue;
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, statement.range.start,
          "assignment to local '" + name.str() +
              "' inside try/except* is not implemented yet (the value would "
              "not survive past the statement); bind a new name inside the "
              "clause or restructure without except*"});
      return;
    }
  }

  struct StarHandler {
    const parser::Node *node = nullptr;
    mlir::Type handlerType;
    std::optional<std::string_view> name;
  };
  llvm::SmallVector<StarHandler, 4> starHandlers;
  for (const parser::NodePtr &handlerPtr : *handlers) {
    if (!handlerPtr)
      continue;
    const parser::Node &handler = *handlerPtr;
    const parser::Node *typeNode = ast::node(handler, "type");
    if (!typeNode) {
      diagnostics.push_back(
          parser::Diagnostic{parser::Severity::Error, handler.range.start,
                             "except* requires an exception type"});
      return;
    }
    if (typeNode->kind == "Tuple") {
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, typeNode->range.start,
          "except* with a tuple of exception types is not implemented yet"});
      return;
    }
    // PEP 654: continue/break/return are a SyntaxError inside except*.
    if (containsReturnStatement(ast::nodeList(handler, "body")) ||
        containsBreakOrContinueStatement(ast::nodeList(handler, "body"))) {
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, handler.range.start,
          "'return', 'break' and 'continue' are not allowed in an except* "
          "block"});
      return;
    }
    mlir::Type candidateType = types.inferExpr(typeNode);
    if (!mlir::isa_and_nonnull<py::TypeType>(candidateType)) {
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, typeNode->range.start,
          "except* handler must resolve to a Python type object"});
      return;
    }
    starHandlers.push_back(StarHandler{&handler, candidateType,
                                       ast::string(handler, "name")});
  }
  if (starHandlers.empty()) {
    diagnostics.push_back(
        parser::Diagnostic{parser::Severity::Error, statement.range.start,
                           "except* requires at least one handler"});
    return;
  }

  mlir::OperationState state(loc(statement), py::TryOp::getOperationName());
  state.addRegion();
  state.addRegion();
  state.addRegion();
  mlir::Operation *rawTry = builder.create(state);
  auto tryOp = mlir::cast<py::TryOp>(rawTry);

  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    auto *tryBlock = new mlir::Block;
    tryOp.getTryRegion().push_back(tryBlock);
    builder.setInsertionPointToStart(tryBlock);
    {
      ScopedEmitterScope scope(values, types);
      emitStatements(ast::nodeList(statement, "body"));
    }
    terminateOpenRegionBlocks<py::TryYieldOp>(builder, loc(statement),
                                              tryOp.getTryRegion());
  }

  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    auto *entryBlock = new mlir::Block;
    tryOp.getExceptRegion().push_back(entryBlock);
    llvm::SmallVector<mlir::Block *, 4> checkBlocks;
    llvm::SmallVector<mlir::Block *, 4> bodyBlocks;
    for (std::size_t index = 0; index < starHandlers.size(); ++index) {
      checkBlocks.push_back(new mlir::Block);
      bodyBlocks.push_back(new mlir::Block);
      tryOp.getExceptRegion().push_back(checkBlocks.back());
      tryOp.getExceptRegion().push_back(bodyBlocks.back());
    }
    auto *finishBlock = new mlir::Block;
    tryOp.getExceptRegion().push_back(finishBlock);

    builder.setInsertionPointToStart(entryBlock);
    {
      mlir::OperationState beginState(loc(statement),
                                      py::StarBeginOp::getOperationName());
      builder.create(beginState);
    }
    mlir::cf::BranchOp::create(builder, loc(statement), checkBlocks.front());

    for (auto [index, starHandler] : llvm::enumerate(starHandlers)) {
      const parser::Node &handler = *starHandler.node;
      mlir::Block *next = index + 1 == starHandlers.size()
                              ? finishBlock
                              : checkBlocks[index + 1];
      builder.setInsertionPointToStart(checkBlocks[index]);
      mlir::OperationState matchState(
          loc(handler), py::ExceptStarMatchOp::getOperationName());
      matchState.addTypes(builder.getI1Type());
      matchState.addAttribute("handler",
                              mlir::TypeAttr::get(starHandler.handlerType));
      auto match =
          mlir::cast<py::ExceptStarMatchOp>(builder.create(matchState));
      mlir::cf::CondBranchOp::create(builder, loc(handler), match.getResult(),
                                     bodyBlocks[index], mlir::ValueRange{},
                                     next, mlir::ValueRange{});

      builder.setInsertionPointToStart(bodyBlocks[index]);
      {
        ScopedEmitterScope scope(values, types);
        if (starHandler.name) {
          // The binding is the matched SLICE: always an exception group
          // (naked matches arrive pre-wrapped by the runtime split).
          mlir::Type groupContract = types.contract("builtins.ExceptionGroup");
          mlir::Type bindingType = types.typeObject(groupContract);
          auto current = py::ExceptCurrentValueOp::create(
                             builder, loc(handler), groupContract,
                             mlir::TypeAttr::get(bindingType))
                             .getResult();
          std::string name(*starHandler.name);
          values[name] = Value{current, groupContract};
          types.bindSymbol(name, groupContract);
        }

        // Inner collect-try: a raise in the clause body parks the exception
        // in the star frame and the remaining clauses still run.
        mlir::OperationState innerState(loc(handler),
                                        py::TryOp::getOperationName());
        innerState.addRegion();
        innerState.addRegion();
        innerState.addRegion();
        mlir::Operation *rawInner = builder.create(innerState);
        auto innerTry = mlir::cast<py::TryOp>(rawInner);
        {
          mlir::OpBuilder::InsertionGuard innerGuard(builder);
          auto *innerBody = new mlir::Block;
          innerTry.getTryRegion().push_back(innerBody);
          builder.setInsertionPointToStart(innerBody);
          {
            ScopedEmitterScope bodyScope(values, types);
            emitStatements(ast::nodeList(handler, "body"));
          }
          terminateOpenRegionBlocks<py::TryYieldOp>(builder, loc(handler),
                                                    innerTry.getTryRegion());
          auto *collectBlock = new mlir::Block;
          innerTry.getExceptRegion().push_back(collectBlock);
          builder.setInsertionPointToStart(collectBlock);
          mlir::OperationState collectState(
              loc(handler), py::StarCollectOp::getOperationName());
          builder.create(collectState);
          py::ExceptYieldOp::create(builder, loc(handler), mlir::ValueRange{});
        }
        builder.setInsertionPointAfter(innerTry);
        mlir::OperationState bodyEndState(
            loc(handler), py::StarBodyEndOp::getOperationName());
        builder.create(bodyEndState);
      }
      mlir::cf::BranchOp::create(builder, loc(handler), next);
    }

    builder.setInsertionPointToStart(finishBlock);
    {
      mlir::OperationState finishState(loc(statement),
                                       py::StarFinishOp::getOperationName());
      builder.create(finishState);
    }
    py::ExceptYieldOp::create(builder, loc(statement), mlir::ValueRange{});
  }

  builder.setInsertionPointAfter(tryOp);
}

} // namespace lython::emitter
