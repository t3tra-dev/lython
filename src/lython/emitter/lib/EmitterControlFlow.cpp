#include "EmitterCore.h"
#include "EmitterPyOps.h"
#include "EmitterSupport.h"

#include "AstAccess.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/STLExtras.h"

namespace lython::emitter {

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
  mlir::Type iteratorType = types.protocol("AsyncIterator", {types.object()});
  CallInferenceResult iterInference =
      types.inferMethodCallWithEvidence(iterable.type, "__aiter__", {});
  if (iterInference)
    iteratorType = iterInference.resultType;
  mlir::UnitAttr returnedSelf =
      iteratorType == iterable.type ? builder.getUnitAttr() : mlir::UnitAttr();
  auto iterator = builder.create<py::AIterOp>(
      loc(statement), iteratorType, "__aiter__", callProtocolFor(iterInference),
      iterable.value, returnedSelf);

  auto whileOp = builder.create<mlir::scf::WhileOp>(
      loc(statement), mlir::TypeRange{}, mlir::ValueRange{});

  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    mlir::Block *before = builder.createBlock(&whileOp.getBefore());
    builder.setInsertionPointToStart(before);

    mlir::OperationState tryState(loc(statement),
                                  py::TryOp::getOperationName());
    tryState.addTypes(types.boolType());
    tryState.addRegion();
    tryState.addRegion();
    tryState.addRegion();
    auto tryOp = mlir::cast<py::TryOp>(builder.create(tryState));

    mlir::Block *tryBlock = new mlir::Block;
    tryOp.getTryRegion().push_back(tryBlock);
    builder.setInsertionPointToStart(tryBlock);
    CallInferenceResult nextInference =
        types.inferMethodCallWithEvidence(iteratorType, "__anext__", {});
    mlir::Type awaitableType =
        nextInference ? nextInference.resultType
                      : types.protocol("Awaitable", {types.object()});
    auto next = builder.create<py::ANextOp>(
        loc(statement), awaitableType, "__anext__",
        callProtocolFor(nextInference), iterator.getResult());
    Value item =
        emitAwaitValue(statement, Value{next.getAwaitable(), awaitableType});
    {
      ScopedEmitterScope scope(values, types);
      emitAssignTarget(*ast::node(statement, "target"), item);
      emitStatements(ast::nodeList(statement, "body"));
    }
    if (!blockHasTerminator(*tryBlock)) {
      mlir::Type trueType = types.literal("True");
      Value trueValue{builder
                          .create<py::BoolConstantOp>(loc(statement), trueType,
                                                      builder.getBoolAttr(true))
                          .getResult(),
                      trueType};
      mlir::Value keepGoing =
          coerceValue(trueValue, types.boolType(), statement).value;
      builder.create<py::TryYieldOp>(loc(statement),
                                     mlir::ValueRange{keepGoing});
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
    mlir::Value stop =
        coerceValue(falseValue, types.boolType(), statement).value;
    builder.create<py::ExceptYieldOp>(loc(statement), mlir::ValueRange{stop});

    builder.setInsertionPointToStart(rethrowBlock);
    builder.create<py::RaiseCurrentOp>(loc(statement));

    builder.setInsertionPointAfter(tryOp);
    auto keepGoing = builder.create<py::BoolOp>(
        loc(statement), builder.getI1Type(),
        mlir::FlatSymbolRefAttr::get(&context, "__bool__"), boolProtocol(),
        tryOp.getResult(0));
    builder.create<mlir::scf::ConditionOp>(
        loc(statement), keepGoing.getResult(), mlir::ValueRange{});
  }

  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    mlir::Block *after = builder.createBlock(&whileOp.getAfter());
    builder.setInsertionPointToStart(after);
    builder.create<mlir::scf::YieldOp>(loc(statement));
  }

  builder.setInsertionPointAfter(whileOp);
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
