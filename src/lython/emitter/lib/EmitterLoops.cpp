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

Value stripLocalProtocolView(Value value) {
  if (!value.value)
    return value;
  auto view = value.value.getDefiningOp<py::ProtocolViewOp>();
  if (!view)
    return value;
  return Value{view.getInput(), view.getInput().getType()};
}

} // namespace

llvm::SmallVector<CarriedLoopLocal, 4> ModuleEmitter::collectCarriedLoopLocals(
    const parser::Node &statement, const llvm::StringSet<> *excludedNames,
    llvm::SmallVectorImpl<mlir::Value> &initialValues) {
  llvm::StringSet<> assignedInBody;
  collectAssignedNames(ast::nodeList(statement, "body"), assignedInBody);
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
    carried.push_back(CarriedLoopLocal{name, carriedType});
    initialValues.push_back(coerceValue(initial, carriedType, statement).value);
  }
  return carried;
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
    mlir::Block *headerBlock) {
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
    if (headerBlock && index < headerBlock->getNumArguments()) {
      mlir::Value previous = headerBlock->getArgument(index);
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
  return carriedLoopEdgeOperands(anchor, loop.carriedLocals, loop.headerBlock);
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
  for (const CarriedLoopLocal &local : carried) {
    checkBlock->addArgument(local.type, loc(statement));
    afterBlock->addArgument(local.type, loc(statement));
  }

  builder.setInsertionPointToEnd(entry);
  mlir::cf::BranchOp::create(builder, loc(statement), checkBlock, carriedInitial);

  builder.setInsertionPointToStart(checkBlock);
  bindCarriedLoopLocals(carried, checkBlock);
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

  builder.setInsertionPointToStart(afterBlock);
  bindCarriedLoopLocals(carried, afterBlock);
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
  for (const CarriedLoopLocal &local : carried) {
    headerBlock->addArgument(local.type, loc(statement));
    afterBlock->addArgument(local.type, loc(statement));
  }

  builder.setInsertionPointToEnd(entry);
  mlir::cf::BranchOp::create(builder, loc(statement), headerBlock,
                             carriedInitial);

  // Header: bind carried locals, evaluate the condition, and on false forward
  // the current header values to the after-block.
  builder.setInsertionPointToStart(headerBlock);
  bindCarriedLoopLocals(carried, headerBlock);
  llvm::SmallVector<mlir::Value, 4> headerArgs;
  for (unsigned index = 0; index < carried.size(); ++index)
    headerArgs.push_back(headerBlock->getArgument(index));
  mlir::Value condition = emitBoolValue(emitExpr(test), statement);
  mlir::cf::CondBranchOp::create(builder, loc(statement), condition, bodyBlock,
                                 mlir::ValueRange{}, afterBlock, headerArgs);

  builder.setInsertionPointToStart(bodyBlock);
  {
    ScopedEmitterScope scope(values, types);
    bindCarriedLoopLocals(carried, headerBlock);
    LoopControlContext loop{afterBlock, headerBlock};
    loop.carriedLocals.assign(carried.begin(), carried.end());
    loop.headerBlock = headerBlock;
    loopControlContexts.push_back(loop);
    emitStatements(ast::nodeList(statement, "body"));
    loopControlContexts.pop_back();
    if (!insertionBlockTerminated(builder))
      mlir::cf::BranchOp::create(
          builder, loc(statement), headerBlock,
          carriedLoopEdgeOperands(statement, carried, headerBlock));
  }

  builder.setInsertionPointToStart(afterBlock);
  bindCarriedLoopLocals(carried, afterBlock);
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
