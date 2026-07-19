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
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"

namespace lython::emitter {

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
  // A walrus in a while condition rebinds per iteration exactly like a body
  // assignment (the field is absent on for/async-for statements).
  if (const parser::Node *test = ast::node(statement, "test"))
    collectAssignedNames(test, assignedInBody);
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
    mlir::Block *headerBlock, llvm::ArrayRef<mlir::Value> baselineValues) {
  llvm::SmallVector<mlir::Value, 4> operands;
  operands.reserve(carried.size());
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
      if (value != previous && !derivesViaStructuralMutation(value, previous))
        py::DecRefOp::create(builder, loc(anchor), previous);
    }
    operands.push_back(value);
  }
  return operands;
}

llvm::SmallVector<mlir::Value, 4>
ModuleEmitter::loopCarriedBranchOperands(const parser::Node &anchor,
                                         const LoopControlContext &loop,
                                         mlir::Block *target) {
  (void)target;
  return carriedLoopEdgeOperands(anchor, loop.carriedLocals, loop.headerBlock,
                                 loop.baselineValues);
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
  llvm::StringSet<> targetNames;
  collectAssignedNameTargets(ast::node(statement, "target"), targetNames);
  llvm::SmallVector<mlir::Value, 4> carriedInitial;
  llvm::SmallVector<CarriedLoopLocal, 4> carried =
      collectCarriedLoopLocals(statement, &targetNames, carriedInitial);

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
      "__next__", callProtocolFor(nextInference), iterator.getResult());
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
    ScopedEmitterScope scope(values, types);
    bindCarriedLoopLocals(carried, elseBlock);
    emitStatements(orelse);
    if (!insertionBlockTerminated(builder))
      mlir::cf::BranchOp::create(
          builder, loc(statement), afterBlock,
          carriedLoopEdgeOperands(statement, carried, elseBlock));
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

      parser::NodePtr trueConstant =
          parser::makeNode("Constant", statement.range);
      parser::addField(*trueConstant, "value", true);
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
    ScopedEmitterScope scope(values, types);
    bindCarriedLoopLocals(carried, elseBlock);
    emitStatements(orelse);
    if (!insertionBlockTerminated(builder))
      mlir::cf::BranchOp::create(
          builder, loc(statement), afterBlock,
          carriedLoopEdgeOperands(statement, carried, elseBlock));
  }

  builder.setInsertionPointToStart(afterBlock);
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
