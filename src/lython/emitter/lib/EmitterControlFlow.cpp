#include "EmitterCore.h"
#include "EmitterPyOps.h"
#include "EmitterSupport.h"

#include "AstAccess.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"

namespace lython::emitter {
namespace {

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

void collectAssignedNames(const parser::Node *node, llvm::StringSet<> &names) {
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

} // namespace

void ModuleEmitter::emitIf(const parser::Node &statement) {
  const parser::Node *test = ast::node(statement, "test");
  std::optional<NoneComparisonNarrowing> narrowing =
      test ? optionalNoneComparison(*test, types) : std::nullopt;
  auto applyNarrowing = [&](const NoneComparisonNarrowing &fact,
                            bool conditionIsTrue) {
    mlir::Type narrowed = conditionIsTrue == fact.trueBranchIsNone
                              ? types.none()
                              : fact.payloadType;
    if (!narrowed)
      return;
    auto found = values.find(fact.name);
    if (found != values.end()) {
      if (mlir::isa<py::UnionType>(found->second.value.getType()) &&
          found->second.value.getType() != narrowed) {
        auto unwrap = builder.create<py::UnionUnwrapOp>(
            loc(statement), narrowed, found->second.value);
        found->second.value = unwrap.getResult();
      }
      found->second.type = narrowed;
    }
    types.bindSymbol(fact.name, narrowed);
  };

  mlir::Value condition = emitBoolValue(emitExpr(test), statement);
  const auto *orelse = ast::nodeList(statement, "orelse");
  bool hasElse = orelse && !orelse->empty();
  mlir::Block *entry = builder.getInsertionBlock();
  mlir::Region *region = entry->getParent();
  mlir::Block *continuation = entry->splitBlock(builder.getInsertionPoint());
  mlir::Block *thenBlock =
      builder.createBlock(region, continuation->getIterator());
  mlir::Block *elseBlock =
      hasElse ? builder.createBlock(region, continuation->getIterator())
              : continuation;

  builder.setInsertionPointToEnd(entry);
  builder.create<mlir::cf::CondBranchOp>(loc(statement), condition, thenBlock,
                                         mlir::ValueRange{}, elseBlock,
                                         mlir::ValueRange{});

  builder.setInsertionPointToStart(thenBlock);
  {
    ScopedEmitterScope scope(values, types);
    if (narrowing)
      applyNarrowing(*narrowing, /*conditionIsTrue=*/true);
    emitStatements(ast::nodeList(statement, "body"));
  }
  bool thenTerminates = insertionBlockTerminated(builder);
  if (!thenTerminates)
    builder.create<mlir::cf::BranchOp>(loc(statement), continuation);

  bool elseTerminates = false;
  if (hasElse) {
    builder.setInsertionPointToStart(elseBlock);
    {
      ScopedEmitterScope scope(values, types);
      if (narrowing)
        applyNarrowing(*narrowing, /*conditionIsTrue=*/false);
      emitStatements(orelse);
    }
    elseTerminates = insertionBlockTerminated(builder);
    if (!elseTerminates)
      builder.create<mlir::cf::BranchOp>(loc(statement), continuation);
  }
  setInsertionBeforeTerminator(builder, *continuation);
  if (narrowing && thenTerminates && !elseTerminates)
    applyNarrowing(*narrowing, /*conditionIsTrue=*/false);
  else if (narrowing && hasElse && elseTerminates && !thenTerminates)
    applyNarrowing(*narrowing, /*conditionIsTrue=*/true);
}

void ModuleEmitter::emitFor(const parser::Node &statement) {
  Value iterable = emitExpr(ast::node(statement, "iter"));
  mlir::Type elem = elementType(iterable.type, types);
  mlir::Type iteratorType = types.iteratorOf(elem);
  CallInferenceResult iterInference =
      types.inferMethodCallWithEvidence(iterable.type, "__iter__", {});
  if (iterInference)
    iteratorType = iterInference.resultType;
  mlir::UnitAttr returnedSelf =
      iteratorType == iterable.type ? builder.getUnitAttr() : mlir::UnitAttr();
  auto iterator = builder.create<py::IterOp>(
      loc(statement), iteratorType, "__iter__", callProtocolFor(iterInference),
      iterable.value, returnedSelf);
  auto whileOp = builder.create<mlir::scf::WhileOp>(
      loc(statement), mlir::TypeRange{}, mlir::ValueRange{});

  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    mlir::Block *before = builder.createBlock(&whileOp.getBefore());
    builder.setInsertionPointToStart(before);
    CallInferenceResult nextInference =
        types.inferMethodCallWithEvidence(iteratorType, "__next__", {});
    if (nextInference)
      elem = nextInference.resultType;
    auto next = builder.create<py::NextOp>(
        loc(statement), elem, builder.getI1Type(), iteratorType, "__next__",
        callProtocolFor(nextInference), iterator.getResult());

    auto bodyIf =
        builder.create<mlir::scf::IfOp>(loc(statement), next.getValid(), false);
    mlir::Block &thenBlock = bodyIf.getThenRegion().front();
    setInsertionBeforeTerminator(builder, thenBlock);
    {
      ScopedEmitterScope scope(values, types);
      emitAssignTarget(*ast::node(statement, "target"),
                       Value{next.getElement(), elem});
      emitStatements(ast::nodeList(statement, "body"));
    }
    ensureYield(builder, loc(statement), thenBlock);

    builder.setInsertionPointAfter(bodyIf);
    builder.create<mlir::scf::ConditionOp>(loc(statement), next.getValid(),
                                           mlir::ValueRange{});
  }

  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    mlir::Block *after = builder.createBlock(&whileOp.getAfter());
    builder.setInsertionPointToStart(after);
    builder.create<mlir::scf::YieldOp>(loc(statement));
  }

  builder.setInsertionPointAfter(whileOp);
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
  mlir::Type iteratorType = types.protocol("AsyncIterator", {types.object()});
  Value iteratorValue;
  Value sourceIteratorReceiver;
  if (std::optional<MethodBinding> sourceAiter =
          lookupClassMethod(methodIterable.type, "__aiter__")) {
    if (sourceAiter->async) {
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, statement.range.start,
          "async __aiter__ methods are not supported; __aiter__ must return an "
          "AsyncIterator directly"});
      iteratorValue = emitNone(statement);
    } else {
      iteratorValue =
          emitInlineMethodCall(statement, methodIterable, *sourceAiter);
      iteratorType = iteratorValue.type;
      if (lookupClassMethod(methodIterable.type, "__anext__"))
        sourceIteratorReceiver = methodIterable;
    }
  } else {
    CallInferenceResult iterInference =
        types.inferMethodCallWithEvidence(iterable.type, "__aiter__", {});
    if (iterInference)
      iteratorType = iterInference.resultType;
    mlir::UnitAttr returnedSelf = iteratorType == iterable.type
                                      ? builder.getUnitAttr()
                                      : mlir::UnitAttr();
    auto iterator = builder.create<py::AIterOp>(
        loc(statement), iteratorType, "__aiter__",
        callProtocolFor(iterInference), iterable.value, returnedSelf);
    iteratorValue = Value{iterator.getResult(), iteratorType};
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
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, statement.range.start,
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
  builder.create<mlir::cf::BranchOp>(loc(statement), loopBlock,
                                     carriedInitialValues);
  builder.setInsertionPointToStart(loopBlock);
  for (auto [index, local] : llvm::enumerate(carriedLocals)) {
    Value loopValue{loopBlock->getArgument(index), local.type};
    values[local.name] = loopValue;
    types.bindSymbol(local.name, local.type);
  }

    mlir::OperationState tryState(loc(statement),
                                  py::TryOp::getOperationName());
    tryState.addTypes(types.boolType());
    for (const CarriedLocal &local : carriedLocals)
      tryState.addTypes(local.type);
    tryState.addRegion();
    tryState.addRegion();
    tryState.addRegion();
    auto tryOp = mlir::cast<py::TryOp>(builder.create(tryState));

    mlir::Block *tryBlock = new mlir::Block;
    tryOp.getTryRegion().push_back(tryBlock);
    builder.setInsertionPointToStart(tryBlock);
    mlir::Type awaitableType = types.protocol("Awaitable", {types.object()});
    Value awaitable;
    if (sourceAnextMethod) {
      if (sourceAnextMethod->async) {
        if (sourceAnextMethod->symbolName.empty()) {
          diagnostics.push_back(parser::Diagnostic{
              parser::Severity::Error, statement.range.start,
              "async __anext__ method has no lowered callable symbol"});
          awaitable = emitNone(statement);
        } else {
          Value sourceAnextCallable =
              emitMethodObject(statement, sourceAnextReceiver,
                               *sourceAnextMethod);
          awaitable = emitCallableDispatch(
              statement, sourceAnextCallable,
              emitCallOperands(statement, {}, /*includeAstArguments=*/false));
        }
      } else {
        awaitable =
            emitInlineMethodCall(statement, sourceAnextReceiver,
                                 *sourceAnextMethod);
      }
      awaitableType = awaitable.type;
    } else {
      CallInferenceResult nextInference =
          types.inferMethodCallWithEvidence(iteratorType, "__anext__", {});
      if (nextInference)
        awaitableType = nextInference.resultType;
      auto next = builder.create<py::ANextOp>(
          loc(statement), awaitableType, "__anext__",
          callProtocolFor(nextInference), iteratorValue.value);
      awaitable = Value{next.getAwaitable(), awaitableType};
    }
    Value item = emitAwaitValue(statement, awaitable);
    {
      ScopedEmitterScope scope(values, types);
      emitAssignTarget(*ast::node(statement, "target"), item);
      emitStatements(ast::nodeList(statement, "body"));
      if (!blockHasTerminator(*tryBlock)) {
        mlir::Type trueType = types.literal("True");
        Value trueValue{
            builder
                .create<py::BoolConstantOp>(loc(statement), trueType,
                                            builder.getBoolAttr(true))
                .getResult(),
            trueType};
        llvm::SmallVector<mlir::Value, 4> yieldValues;
        yieldValues.push_back(
            coerceValue(trueValue, types.boolType(), statement).value);
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
          if (carried.value && carried.value != previous)
            builder.create<py::DecRefOp>(loc(statement), previous);
          yieldValues.push_back(coerceValue(carried, local.type, statement)
                                    .value);
        }
        builder.create<py::TryYieldOp>(loc(statement), yieldValues);
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
    mlir::OperationState matchState(
        loc(statement), py::ExceptCurrentMatchOp::getOperationName());
    matchState.addTypes(builder.getI1Type());
    matchState.addAttribute("handler", mlir::TypeAttr::get(stopAsyncIteration));
    auto match =
        mlir::cast<py::ExceptCurrentMatchOp>(builder.create(matchState));
    builder.create<mlir::cf::CondBranchOp>(loc(statement), match.getResult(),
                                           stopBlock, mlir::ValueRange{},
                                           rethrowBlock, mlir::ValueRange{});

    builder.setInsertionPointToStart(stopBlock);
    mlir::Type falseType = types.literal("False");
    Value falseValue{builder
                         .create<py::BoolConstantOp>(loc(statement), falseType,
                                                     builder.getBoolAttr(false))
                         .getResult(),
                     falseType};
    llvm::SmallVector<mlir::Value, 4> stopValues;
    stopValues.push_back(coerceValue(falseValue, types.boolType(), statement)
                             .value);
    for (auto [index, local] : llvm::enumerate(carriedLocals))
      stopValues.push_back(loopBlock->getArgument(index));
    builder.create<py::ExceptYieldOp>(loc(statement), stopValues);

    builder.setInsertionPointToStart(rethrowBlock);
    builder.create<py::RaiseCurrentOp>(loc(statement));

    builder.setInsertionPointAfter(tryOp);
    mlir::Value keepGoing =
        emitBoolValue(Value{tryOp.getResult(0), types.boolType()}, statement);
    llvm::SmallVector<mlir::Value, 4> nextCarriedValues;
    nextCarriedValues.reserve(carriedLocals.size());
    for (auto [index, local] : llvm::enumerate(carriedLocals)) {
      mlir::Value result = tryOp.getResult(static_cast<unsigned>(index) + 1);
      nextCarriedValues.push_back(result);
      values[local.name] = Value{result, local.type};
      types.bindSymbol(local.name, local.type);
    }
    builder.create<mlir::cf::CondBranchOp>(loc(statement), keepGoing,
                                           loopBlock, nextCarriedValues,
                                           afterBlock, nextCarriedValues);

  builder.setInsertionPointToStart(afterBlock);
  for (auto [index, local] : llvm::enumerate(carriedLocals)) {
    values[local.name] = Value{afterBlock->getArgument(index), local.type};
    types.bindSymbol(local.name, local.type);
  }
}

void ModuleEmitter::emitTry(const parser::Node &statement) {
  const auto *handlers = ast::nodeList(statement, "handlers");
  if (!handlers || handlers->empty()) {
    diagnostics.push_back(
        parser::Diagnostic{parser::Severity::Error, statement.range.start,
                           "try without except is not implemented yet"});
    return;
  }
  if (const auto *orelse = ast::nodeList(statement, "orelse")) {
    if (!orelse->empty()) {
      diagnostics.push_back(
          parser::Diagnostic{parser::Severity::Error, statement.range.start,
                             "try/else is not implemented yet"});
      return;
    }
  }
  if (const auto *finalbody = ast::nodeList(statement, "finalbody")) {
    if (!finalbody->empty()) {
      diagnostics.push_back(
          parser::Diagnostic{parser::Severity::Error, statement.range.start,
                             "try/finally is not implemented yet"});
      return;
    }
  }
  if (containsReturnStatement(ast::nodeList(statement, "body"))) {
    diagnostics.push_back(
        parser::Diagnostic{parser::Severity::Error, statement.range.start,
                           "return inside try is not implemented yet"});
    return;
  }
  if (const auto *handlersForReturn = ast::nodeList(statement, "handlers")) {
    for (const parser::NodePtr &handler : *handlersForReturn) {
      if (handler && containsReturnStatement(ast::nodeList(*handler, "body"))) {
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, handler->range.start,
            "return inside except handler is not implemented yet"});
        return;
      }
    }
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
    if (!blockHasTerminator(*tryBlock))
      builder.create<py::TryYieldOp>(loc(statement), mlir::ValueRange{});
  }

  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    llvm::SmallVector<mlir::Block *, 8> checkBlocks;
    llvm::SmallVector<mlir::Block *, 8> bodyBlocks;
    checkBlocks.reserve(handlers->size());
    bodyBlocks.reserve(handlers->size());
    for (std::size_t index = 0; index < handlers->size(); ++index) {
      checkBlocks.push_back(new mlir::Block);
      bodyBlocks.push_back(new mlir::Block);
      tryOp.getExceptRegion().push_back(checkBlocks.back());
      tryOp.getExceptRegion().push_back(bodyBlocks.back());
    }
    mlir::Block *rethrowBlock = new mlir::Block;
    tryOp.getExceptRegion().push_back(rethrowBlock);

    for (auto [index, handlerPtr] : llvm::enumerate(*handlers)) {
      const parser::Node &handler = *handlerPtr;
      if (auto name = ast::string(handler, "name")) {
        diagnostics.push_back(
            parser::Diagnostic{parser::Severity::Error, handler.range.start,
                               "except-as binding is not implemented yet"});
        continue;
      }

      const parser::Node *typeNode = ast::node(handler, "type");
      if (!typeNode && index + 1 != handlers->size()) {
        diagnostics.push_back(
            parser::Diagnostic{parser::Severity::Error, handler.range.start,
                               "bare except must be the last handler"});
        continue;
      }
      if (typeNode && typeNode->kind == "Tuple") {
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, typeNode->range.start,
            "tuple exception handlers are not implemented yet"});
        continue;
      }

      mlir::Type handlerType =
          typeNode ? types.inferExpr(typeNode)
                   : types.typeObject(types.contract("builtins.BaseException"));
      if (!mlir::isa_and_nonnull<py::TypeType>(handlerType)) {
        handlerType = types.typeObject(types.annotationType(typeNode));
      }
      if (!mlir::isa_and_nonnull<py::TypeType>(handlerType)) {
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error,
            typeNode ? typeNode->range.start : handler.range.start,
            "except handler must resolve to a Python type object"});
        continue;
      }

      builder.setInsertionPointToStart(checkBlocks[index]);
      mlir::OperationState matchState(
          loc(handler), py::ExceptCurrentMatchOp::getOperationName());
      matchState.addTypes(builder.getI1Type());
      matchState.addAttribute("handler", mlir::TypeAttr::get(handlerType));
      auto match =
          mlir::cast<py::ExceptCurrentMatchOp>(builder.create(matchState));
      mlir::Block *miss =
          index + 1 == handlers->size() ? rethrowBlock : checkBlocks[index + 1];
      builder.create<mlir::cf::CondBranchOp>(
          loc(handler), match.getResult(), bodyBlocks[index],
          mlir::ValueRange{}, miss, mlir::ValueRange{});

      builder.setInsertionPointToStart(bodyBlocks[index]);
      {
        ScopedEmitterScope scope(values, types);
        emitStatements(ast::nodeList(handler, "body"));
      }
      if (!blockHasTerminator(*bodyBlocks[index]))
        builder.create<py::ExceptYieldOp>(loc(handler), mlir::ValueRange{});
    }

    builder.setInsertionPointToStart(rethrowBlock);
    builder.create<py::RaiseCurrentOp>(loc(statement));
  }

  builder.setInsertionPointAfter(tryOp);
}

} // namespace lython::emitter
