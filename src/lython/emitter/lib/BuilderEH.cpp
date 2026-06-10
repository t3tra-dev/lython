#include "BuilderImpl.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

namespace lython::emitter {

namespace {

struct ExceptHandlerPlan {
  const parser::Node *node = nullptr;
  const std::vector<parser::NodePtr> *body = nullptr;
  std::vector<std::string> classNames;
};

struct EmittedExceptHandler {
  const ExceptHandlerPlan *plan = nullptr;
  mlir::Block *block = nullptr;
};

bool isNoneConstant(const parser::NodePtr *node) {
  if (!node || !*node || (*node)->kind != "Constant")
    return false;
  const parser::FieldValue *value = valueField(**node, "value");
  return value && std::holds_alternative<std::monostate>(*value);
}

void emitYieldIfOpen(mlir::OpBuilder &builder, mlir::Location location,
                     bool blockTerminated, bool isExcept,
                     mlir::ValueRange operands = {}) {
  if (blockTerminated)
    return;
  if (isExcept)
    builder.create<py::ExceptYieldOp>(location, operands);
  else
    builder.create<py::TryYieldOp>(location, operands);
}

} // namespace

void Builder::Impl::emitTry(const parser::Node &stmt) {
  if (inNativeFunction) {
    error(stmt, "try/except is not supported inside @native functions");
    return;
  }

  const std::vector<parser::NodePtr> *body = nodeListField(stmt, "body");
  const std::vector<parser::NodePtr> *handlers =
      nodeListField(stmt, "handlers");
  const std::vector<parser::NodePtr> *orelse = nodeListField(stmt, "orelse");
  const std::vector<parser::NodePtr> *finalbody =
      nodeListField(stmt, "finalbody");
  if (!body || body->empty()) {
    error(stmt, "Try.body is missing");
    return;
  }
  bool hasHandlers = handlers && !handlers->empty();
  bool hasFinally = finalbody && !finalbody->empty();
  bool hasElse = orelse && !orelse->empty();
  if (!hasHandlers && !hasFinally) {
    error(stmt, "try requires except handlers or a finally body");
    return;
  }
  if (hasElse && !hasHandlers) {
    error(stmt, "try/else requires an except handler");
    return;
  }
  std::vector<ExceptHandlerPlan> handlerPlans;
  auto collectHandlerClassNames =
      [&](const parser::Node &handlerNode, const parser::NodePtr *typeNode,
          std::vector<std::string> &classNames) -> bool {
    if (!typeNode || !*typeNode)
      return true;
    auto collectOne = [&](const parser::Node &node) -> bool {
      if (node.kind != "Name") {
        error(node, "except handler type must be a statically known builtin "
                    "exception class");
        return false;
      }
      const std::string *name = stringField(node, "id");
      if (!name || !isBuiltinExceptionClass(*name)) {
        error(node, "except handler type must be a builtin BaseException "
                    "subclass in the current C++ emitter");
        return false;
      }
      classNames.push_back(*name);
      return true;
    };

    if ((*typeNode)->kind == "Tuple") {
      const std::vector<parser::NodePtr> *elements =
          nodeListField(**typeNode, "elts");
      if (!elements || elements->empty()) {
        error(**typeNode, "except handler tuple must not be empty");
        return false;
      }
      for (const parser::NodePtr &element : *elements) {
        if (!element || !collectOne(*element))
          return false;
      }
      return true;
    }
    return collectOne(**typeNode);
  };

  auto emitExceptDispatchChain = [&](mlir::Block *exceptBlock,
                                     llvm::ArrayRef<ExceptHandlerPlan> plans)
      -> std::vector<EmittedExceptHandler> {
    std::vector<EmittedExceptHandler> emitted;
    if (plans.empty())
      return emitted;

    mlir::Region *region = exceptBlock->getParent();
    mlir::Value caught = exceptBlock->getArgument(0);
    mlir::Block *dispatchBlock = exceptBlock;

    for (const ExceptHandlerPlan &plan : plans) {
      mlir::Block *handlerBlock = new mlir::Block();
      region->push_back(handlerBlock);
      emitted.push_back(EmittedExceptHandler{&plan, handlerBlock});

      builder.setInsertionPointToStart(dispatchBlock);
      if (plan.classNames.empty()) {
        builder.create<mlir::cf::BranchOp>(loc(*plan.node), handlerBlock);
        return emitted;
      }

      mlir::Block *nextDispatchBlock = new mlir::Block();
      region->push_back(nextDispatchBlock);
      mlir::Value matched =
          builder.create<mlir::arith::ConstantIntOp>(loc(*plan.node), 0, 1);
      for (const std::string &className : plan.classNames) {
        auto match = builder.create<py::ExceptMatchOp>(
            loc(*plan.node), i1Type(), caught,
            mlir::TypeAttr::get(classType(className)));
        matched = builder.create<mlir::arith::OrIOp>(loc(*plan.node), matched,
                                                     match.getResult());
      }
      builder.create<mlir::cf::CondBranchOp>(loc(*plan.node), matched,
                                             handlerBlock, nextDispatchBlock);
      dispatchBlock = nextDispatchBlock;
    }

    const parser::Node &lastHandler = *plans.back().node;
    builder.setInsertionPointToStart(dispatchBlock);
    builder.create<py::RaiseOp>(loc(lastHandler), caught);
    return emitted;
  };

  auto emitExceptHandlerBody =
      [&](mlir::Block *exceptBlock, const EmittedExceptHandler &handler,
          const std::map<std::string, Value> &baseSymbols,
          const std::map<std::string, FunctionInfo> &baseCallableAliases,
          bool yieldsCompletionFlag) {
        builder.setInsertionPointToStart(handler.block);
        symbols = baseSymbols;
        callableAliases = baseCallableAliases;
        const std::string *alias = stringField(*handler.plan->node, "name");
        if (alias && !alias->empty())
          symbols[*alias] = Value{exceptBlock->getArgument(0), exceptionType()};
        blockTerminated = false;
        ++exceptionContextDepth;
        for (const parser::NodePtr &child : *handler.plan->body)
          if (child && !blockTerminated)
            emitStatement(*child);
        --exceptionContextDepth;
        if (blockTerminated)
          return;
        if (yieldsCompletionFlag) {
          mlir::Value completedFalse =
              builder.create<mlir::arith::ConstantIntOp>(
                  loc(*handler.plan->node), 0, 1);
          emitYieldIfOpen(builder, loc(*handler.plan->node),
                          /*blockTerminated=*/false,
                          /*isExcept=*/true, mlir::ValueRange{completedFalse});
          return;
        }
        emitYieldIfOpen(builder, loc(*handler.plan->node),
                        /*blockTerminated=*/false,
                        /*isExcept=*/true);
      };

  if (hasHandlers) {
    handlerPlans.reserve(handlers->size());
    for (auto [index, handlerPtr] : llvm::enumerate(*handlers)) {
      if (!handlerPtr || handlerPtr->kind != "ExceptHandler") {
        error(stmt, "Try.handlers must contain ExceptHandler nodes");
        return;
      }
      const parser::NodePtr *handlerType = nodeField(*handlerPtr, "type");
      ExceptHandlerPlan plan;
      plan.node = handlerPtr.get();
      if (!collectHandlerClassNames(*handlerPtr, handlerType, plan.classNames))
        return;
      if (!handlerType && index + 1 != handlers->size()) {
        error(*handlerPtr, "default 'except:' must be last");
        return;
      }
      plan.body = nodeListField(*handlerPtr, "body");
      if (!plan.body || plan.body->empty()) {
        error(*handlerPtr, "ExceptHandler.body is missing");
        return;
      }
      handlerPlans.push_back(std::move(plan));
    }
  }

  if (hasElse && hasFinally) {
    py::TryOp outerTry =
        builder.create<py::TryOp>(loc(stmt), mlir::TypeRange{});
    std::map<std::string, Value> outerSymbols = symbols;
    std::map<std::string, FunctionInfo> outerCallableAliases = callableAliases;

    mlir::Block *outerTryBlock = new mlir::Block();
    outerTry.getTryRegion().push_back(outerTryBlock);
    builder.setInsertionPointToStart(outerTryBlock);
    symbols = outerSymbols;
    callableAliases = outerCallableAliases;
    blockTerminated = false;

    py::TryOp innerTry =
        builder.create<py::TryOp>(loc(stmt), mlir::TypeRange{i1Type()});
    mlir::Block *innerTryBlock = new mlir::Block();
    innerTry.getTryRegion().push_back(innerTryBlock);
    builder.setInsertionPointToStart(innerTryBlock);
    symbols = outerSymbols;
    callableAliases = outerCallableAliases;
    blockTerminated = false;
    for (const parser::NodePtr &child : *body)
      if (child && !blockTerminated)
        emitStatement(*child);
    bool innerTryTerminated = blockTerminated;
    mlir::Value completedTrue;
    if (!innerTryTerminated)
      completedTrue =
          builder.create<mlir::arith::ConstantIntOp>(loc(stmt), 1, 1);
    emitYieldIfOpen(builder, loc(stmt), innerTryTerminated,
                    /*isExcept=*/false,
                    innerTryTerminated ? mlir::ValueRange{}
                                       : mlir::ValueRange{completedTrue});

    mlir::Block *innerExceptBlock = new mlir::Block();
    innerExceptBlock->addArgument(exceptionType(),
                                  loc(*handlerPlans.front().node));
    innerTry.getExceptRegion().push_back(innerExceptBlock);
    std::vector<EmittedExceptHandler> emittedHandlers =
        emitExceptDispatchChain(innerExceptBlock, handlerPlans);
    for (const EmittedExceptHandler &emitted : emittedHandlers)
      emitExceptHandlerBody(innerExceptBlock, emitted, outerSymbols,
                            outerCallableAliases,
                            /*yieldsCompletionFlag=*/true);

    builder.setInsertionPointAfter(innerTry);
    mlir::Region *outerTryRegion = builder.getBlock()->getParent();
    mlir::Block *elseBlock = new mlir::Block();
    mlir::Block *mergeBlock = new mlir::Block();
    outerTryRegion->push_back(elseBlock);
    outerTryRegion->push_back(mergeBlock);
    builder.create<mlir::cf::CondBranchOp>(loc(stmt), innerTry.getResult(0),
                                           elseBlock, mergeBlock);

    builder.setInsertionPointToStart(elseBlock);
    symbols = outerSymbols;
    callableAliases = outerCallableAliases;
    blockTerminated = false;
    for (const parser::NodePtr &child : *orelse)
      if (child && !blockTerminated)
        emitStatement(*child);
    if (!blockTerminated)
      builder.create<mlir::cf::BranchOp>(loc(stmt), mergeBlock);

    builder.setInsertionPointToStart(mergeBlock);
    symbols = outerSymbols;
    callableAliases = outerCallableAliases;
    blockTerminated = false;
    emitYieldIfOpen(builder, loc(stmt), /*blockTerminated=*/false,
                    /*isExcept=*/false);

    mlir::Block *finallyBlock = new mlir::Block();
    outerTry.getFinallyRegion().push_back(finallyBlock);
    builder.setInsertionPointToStart(finallyBlock);
    symbols = outerSymbols;
    callableAliases = outerCallableAliases;
    blockTerminated = false;
    for (const parser::NodePtr &child : *finalbody)
      if (child && !blockTerminated)
        emitStatement(*child);
    if (!blockTerminated)
      builder.create<py::FinallyYieldOp>(loc(stmt), mlir::ValueRange{});

    builder.setInsertionPointAfter(outerTry);
    symbols = std::move(outerSymbols);
    callableAliases = std::move(outerCallableAliases);
    blockTerminated = false;
    return;
  }

  mlir::Type completedType = i1Type();
  py::TryOp tryOp = builder.create<py::TryOp>(
      loc(stmt), hasElse ? mlir::TypeRange{completedType} : mlir::TypeRange{});
  mlir::OpBuilder::InsertPoint afterTry = builder.saveInsertionPoint();
  std::map<std::string, Value> outerSymbols = symbols;
  std::map<std::string, FunctionInfo> outerCallableAliases = callableAliases;

  mlir::Block *tryBlock = new mlir::Block();
  tryOp.getTryRegion().push_back(tryBlock);
  builder.setInsertionPointToStart(tryBlock);
  blockTerminated = false;
  for (const parser::NodePtr &child : *body)
    if (child && !blockTerminated)
      emitStatement(*child);
  bool tryTerminated = blockTerminated;
  mlir::Value completedTrue;
  if (hasElse && !tryTerminated)
    completedTrue = builder.create<mlir::arith::ConstantIntOp>(loc(stmt), 1, 1);
  emitYieldIfOpen(builder, loc(stmt), tryTerminated, /*isExcept=*/false,
                  hasElse && !tryTerminated ? mlir::ValueRange{completedTrue}
                                            : mlir::ValueRange{});

  if (hasHandlers) {
    mlir::Block *exceptBlock = new mlir::Block();
    exceptBlock->addArgument(exceptionType(), loc(*handlerPlans.front().node));
    tryOp.getExceptRegion().push_back(exceptBlock);
    std::vector<EmittedExceptHandler> emittedHandlers =
        emitExceptDispatchChain(exceptBlock, handlerPlans);
    for (const EmittedExceptHandler &emitted : emittedHandlers)
      emitExceptHandlerBody(exceptBlock, emitted, outerSymbols,
                            outerCallableAliases,
                            /*yieldsCompletionFlag=*/hasElse);
  }

  if (hasFinally) {
    mlir::Block *finallyBlock = new mlir::Block();
    tryOp.getFinallyRegion().push_back(finallyBlock);
    builder.setInsertionPointToStart(finallyBlock);
    symbols = outerSymbols;
    callableAliases = outerCallableAliases;
    blockTerminated = false;
    for (const parser::NodePtr &child : *finalbody)
      if (child && !blockTerminated)
        emitStatement(*child);
    if (!blockTerminated)
      builder.create<py::FinallyYieldOp>(loc(stmt), mlir::ValueRange{});
  }

  if (hasElse) {
    builder.setInsertionPointAfter(tryOp);
    mlir::Region *region = builder.getBlock()->getParent();
    mlir::Block *elseBlock = new mlir::Block();
    mlir::Block *mergeBlock = new mlir::Block();
    region->push_back(elseBlock);
    region->push_back(mergeBlock);
    builder.create<mlir::cf::CondBranchOp>(loc(stmt), tryOp.getResult(0),
                                           elseBlock, mergeBlock);

    builder.setInsertionPointToStart(elseBlock);
    symbols = outerSymbols;
    callableAliases = outerCallableAliases;
    blockTerminated = false;
    for (const parser::NodePtr &child : *orelse)
      if (child && !blockTerminated)
        emitStatement(*child);
    if (!blockTerminated)
      builder.create<mlir::cf::BranchOp>(loc(stmt), mergeBlock);

    symbols = outerSymbols;
    callableAliases = outerCallableAliases;
    builder.setInsertionPointToStart(mergeBlock);
  } else {
    builder.restoreInsertionPoint(afterTry);
    symbols = outerSymbols;
    callableAliases = outerCallableAliases;
  }
  blockTerminated = false;
}

void Builder::Impl::emitRaise(const parser::Node &stmt) {
  const parser::NodePtr *excNode = nodeField(stmt, "exc");
  const parser::NodePtr *causeNode = nodeField(stmt, "cause");
  if (causeNode && *causeNode && !isNoneConstant(causeNode)) {
    error(stmt, "raise ... from <cause> is not supported by the C++ emitter "
                "yet; use 'from None' to suppress exception context");
    return;
  }
  if (!excNode || !*excNode) {
    if (exceptionContextDepth == 0) {
      error(stmt, "bare raise requires an active exception handler");
      return;
    }
    builder.create<py::RaiseCurrentOp>(loc(stmt));
    blockTerminated = true;
    return;
  }

  Value value;
  if ((*excNode)->kind == "Name") {
    const std::string *name = stringField(**excNode, "id");
    if (name && isBuiltinExceptionClass(*name)) {
      const std::vector<parser::NodePtr> noArgs;
      value = emitExceptionCall(**excNode, *name, noArgs);
    }
  }
  if (!value.value)
    value = emitExpression(**excNode);
  if (value.type != exceptionType()) {
    error(**excNode, "raise expression must have static !py.exception type");
    return;
  }

  builder.create<py::RaiseOp>(loc(stmt), value.value);
  blockTerminated = true;
}

void Builder::Impl::emitAssert(const parser::Node &stmt) {
  if (inNativeFunction) {
    error(stmt, "assert is not supported inside @native functions yet");
    return;
  }

  const parser::NodePtr *testNode = nodeField(stmt, "test");
  const parser::NodePtr *msgNode = nodeField(stmt, "msg");
  if (!testNode || !*testNode) {
    error(stmt, "Assert.test is missing");
    return;
  }

  Value condition = emitCondition(**testNode);
  if (!condition.value)
    return;

  mlir::Region *region = builder.getBlock()->getParent();
  mlir::Block *okBlock = new mlir::Block();
  mlir::Block *raiseBlock = new mlir::Block();
  region->push_back(okBlock);
  region->push_back(raiseBlock);
  builder.create<mlir::cf::CondBranchOp>(loc(stmt), condition.value, okBlock,
                                         raiseBlock);
  blockTerminated = true;

  builder.setInsertionPointToStart(raiseBlock);
  llvm::SmallVector<mlir::Value> messageArgs;
  if (msgNode && *msgNode) {
    Value message = emitExpression(**msgNode);
    if (!message.value)
      return;
    if (message.type != strType()) {
      error(**msgNode, "assert message must currently be !py.str");
      return;
    }
    messageArgs.push_back(message.value);
  }
  auto exception = builder.create<py::ExceptionNewOp>(
      loc(stmt), exceptionType(), mlir::ValueRange(messageArgs));
  exception->setAttr("py.exception.class",
                     builder.getStringAttr("AssertionError"));
  builder.create<py::RaiseOp>(loc(stmt), exception.getResult());

  builder.setInsertionPointToStart(okBlock);
  blockTerminated = false;
}

} // namespace lython::emitter
