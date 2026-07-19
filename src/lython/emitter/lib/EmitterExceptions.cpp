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

} // namespace

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
}


} // namespace lython::emitter
