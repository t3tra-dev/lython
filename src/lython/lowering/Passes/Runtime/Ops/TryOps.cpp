#include "Runtime/Core/Lowerer.h"

#include "Runtime/Model/Contracts.h"

#include "llvm/ADT/STLExtras.h"

#include <iterator>

namespace py::lowering {
namespace {

// Region walks below must not touch yields/rethrows that belong to a py.try
// NESTED inside `region`: tries lower outermost-first (handler ids must exist
// before inner constructs look up their enclosing handler), so at this point
// nested py.try ops are still structured. Rewriting their terminators here
// would wire the inner construct's exits straight to the outer targets — a
// nested try/finally's normal path would branch past its own finally region.
template <typename OpT>
void collectOwnedOps(mlir::Region &region, llvm::SmallVectorImpl<OpT> &ops) {
  mlir::Operation *owner = region.getParentOp();
  region.walk([&](OpT op) {
    if (op->template getParentOfType<py::TryOp>() == owner)
      ops.push_back(op);
  });
}

template <typename YieldOp>
void replaceYields(mlir::OpBuilder &builder, mlir::Region &region,
                   mlir::Block *continuation) {
  llvm::SmallVector<YieldOp, 8> yields;
  collectOwnedOps(region, yields);
  for (YieldOp yield : yields) {
    builder.setInsertionPoint(yield);
    mlir::cf::BranchOp::create(builder, yield.getLoc(), continuation,
                               yield.getOperands());
    yield.erase();
  }
}

void replaceExceptYields(mlir::OpBuilder &builder, mlir::Region &region,
                         mlir::Block *continuation,
                         mlir::func::FuncOp discardCurrentException) {
  llvm::SmallVector<py::ExceptYieldOp, 8> yields;
  collectOwnedOps(region, yields);
  for (py::ExceptYieldOp yield : yields) {
    builder.setInsertionPoint(yield);
    mlir::func::CallOp::create(builder, yield.getLoc(), discardCurrentException,
                               mlir::ValueRange{});
    mlir::cf::BranchOp::create(builder, yield.getLoc(), continuation,
                               yield.getOperands());
    yield.erase();
  }
}

mlir::Value boolConstant(mlir::OpBuilder &builder, mlir::Location loc,
                         bool value) {
  return mlir::arith::ConstantIntOp::create(builder, loc, value ? 1 : 0, 1)
      .getResult();
}

mlir::FailureOr<mlir::Value> pythonDefault(mlir::OpBuilder &builder,
                                           mlir::Operation *anchor,
                                           mlir::Type type) {
  if (type.isInteger(1))
    return boolConstant(builder, anchor->getLoc(), false);
  if (auto literal = mlir::dyn_cast<py::LiteralType>(type)) {
    llvm::StringRef spelling = literal.getSpelling();
    if (spelling == "None")
      return py::NoneOp::create(builder, anchor->getLoc(), type).getResult();
    if (spelling == "True" || spelling == "False")
      return py::BoolConstantOp::create(builder, anchor->getLoc(), type,
                                        builder.getBoolAttr(spelling == "True"))
          .getResult();
    if (isIntegerLiteralSpelling(spelling))
      return py::IntConstantOp::create(builder, anchor->getLoc(), type,
                                       builder.getStringAttr(spelling))
          .getResult();
    if (spelling.size() >= 2 && spelling.front() == '"' &&
        spelling.back() == '"')
      return py::StrConstantOp::create(
                 builder, anchor->getLoc(), type,
                 builder.getStringAttr(spelling.drop_front().drop_back()))
          .getResult();
  }
  if (auto contract = mlir::dyn_cast<py::ContractType>(type)) {
    llvm::StringRef name = contract.getContractName();
    if (name == "types.NoneType" || name == "builtins.object") {
      auto none = py::NoneOp::create(
          builder, anchor->getLoc(),
          py::LiteralType::get(builder.getContext(), "None"));
      auto upcast = py::ClassUpcastOp::create(builder, anchor->getLoc(), type,
                                              none.getResult());
      return upcast.getResult();
    }
    if (name == "builtins.bool") {
      mlir::Type falseType =
          py::LiteralType::get(builder.getContext(), "False");
      auto falseValue = py::BoolConstantOp::create(
          builder, anchor->getLoc(), falseType, builder.getBoolAttr(false));
      auto upcast = py::ClassUpcastOp::create(builder, anchor->getLoc(), type,
                                              falseValue.getResult());
      return upcast.getResult();
    }
    if (name == "builtins.int") {
      mlir::Type zeroType = py::LiteralType::get(builder.getContext(), "0");
      auto zero = py::IntConstantOp::create(builder, anchor->getLoc(), zeroType,
                                            builder.getStringAttr("0"));
      auto upcast = py::ClassUpcastOp::create(builder, anchor->getLoc(), type,
                                              zero.getResult());
      return upcast.getResult();
    }
    if (name == "builtins.float")
      return py::FloatConstantOp::create(builder, anchor->getLoc(), type,
                                         builder.getF64FloatAttr(0.0))
          .getResult();
    if (name == "builtins.str") {
      mlir::Type emptyType = py::LiteralType::get(builder.getContext(), "\"\"");
      auto empty = py::StrConstantOp::create(
          builder, anchor->getLoc(), emptyType, builder.getStringAttr(""));
      auto upcast = py::ClassUpcastOp::create(builder, anchor->getLoc(), type,
                                              empty.getResult());
      return upcast.getResult();
    }
  }
  return anchor->emitError()
         << "py.try finally lowering can only synthesize exceptional defaults "
            "for statically defaultable completion results, got "
         << type;
}

mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>>
exceptionalFinallyOperands(mlir::OpBuilder &builder, py::TryOp op) {
  llvm::SmallVector<mlir::Value, 4> operands;
  operands.reserve(op.getNumResults() + 1);
  operands.push_back(boolConstant(builder, op.getLoc(), true));
  for (mlir::Value result : op.getResults()) {
    mlir::FailureOr<mlir::Value> value =
        pythonDefault(builder, op.getOperation(), result.getType());
    if (mlir::failed(value))
      return mlir::failure();
    operands.push_back(*value);
  }
  return operands;
}

template <typename YieldOp>
void replaceYieldsWithFinallyEntry(mlir::OpBuilder &builder,
                                   mlir::Region &region,
                                   mlir::Block *finallyEntry,
                                   bool exceptional) {
  llvm::SmallVector<YieldOp, 8> yields;
  collectOwnedOps(region, yields);
  for (YieldOp yield : yields) {
    builder.setInsertionPoint(yield);
    llvm::SmallVector<mlir::Value, 4> operands;
    mlir::Value mode = boolConstant(builder, yield.getLoc(), exceptional);
    operands.push_back(mode);
    operands.append(yield.getOperands().begin(), yield.getOperands().end());
    mlir::cf::BranchOp::create(builder, yield.getLoc(), finallyEntry, operands);
    yield.erase();
  }
}

void replaceExceptYieldsWithFinallyEntry(
    mlir::OpBuilder &builder, mlir::Region &region, mlir::Block *finallyEntry,
    mlir::func::FuncOp discardCurrentException) {
  llvm::SmallVector<py::ExceptYieldOp, 8> yields;
  collectOwnedOps(region, yields);
  for (py::ExceptYieldOp yield : yields) {
    builder.setInsertionPoint(yield);
    mlir::func::CallOp::create(builder, yield.getLoc(), discardCurrentException,
                               mlir::ValueRange{});
    llvm::SmallVector<mlir::Value, 4> operands;
    mlir::Value mode = boolConstant(builder, yield.getLoc(), false);
    operands.push_back(mode);
    operands.append(yield.getOperands().begin(), yield.getOperands().end());
    mlir::cf::BranchOp::create(builder, yield.getLoc(), finallyEntry, operands);
    yield.erase();
  }
}

mlir::LogicalResult
replaceRaiseCurrentWithFinallyEntry(mlir::OpBuilder &builder,
                                    mlir::Region &region,
                                    mlir::Block *finallyEntry, py::TryOp op) {
  llvm::SmallVector<py::RaiseCurrentOp, 4> raises;
  collectOwnedOps(region, raises);
  for (py::RaiseCurrentOp raise : raises) {
    builder.setInsertionPoint(raise);
    mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>> operands =
        exceptionalFinallyOperands(builder, op);
    if (mlir::failed(operands))
      return mlir::failure();
    mlir::cf::BranchOp::create(builder, raise.getLoc(), finallyEntry,
                               *operands);
    raise.erase();
  }
  return mlir::success();
}

void replaceFinallyYields(mlir::OpBuilder &builder, mlir::Region &region,
                          mlir::Value mode, mlir::ValueRange carriedValues,
                          mlir::Block *dispatch) {
  llvm::SmallVector<py::FinallyYieldOp, 8> yields;
  collectOwnedOps(region, yields);
  for (py::FinallyYieldOp yield : yields) {
    builder.setInsertionPoint(yield);
    llvm::SmallVector<mlir::Value, 4> operands;
    if (yield->getNumOperands() == 0) {
      operands.push_back(mode);
      operands.append(carriedValues.begin(), carriedValues.end());
    } else {
      operands.push_back(boolConstant(builder, yield.getLoc(), false));
      operands.append(yield.getOperands().begin(), yield.getOperands().end());
    }
    mlir::cf::BranchOp::create(builder, yield.getLoc(), dispatch, operands);
    yield.erase();
  }
}

} // namespace

mlir::func::FuncOp RuntimeBundleLowerer::getOrCreateTryCallSiteMarker() {
  return getOrCreatePrivateFunction(
      module, builder, "LyEH_TryCallSiteMarker",
      builder.getFunctionType({builder.getI64Type()}, {}));
}

mlir::func::FuncOp RuntimeBundleLowerer::getOrCreateTryCatchMarker() {
  return getOrCreatePrivateFunction(
      module, builder, "LyEH_TryCatchMarker",
      builder.getFunctionType({builder.getI64Type()}, {}));
}

mlir::func::FuncOp RuntimeBundleLowerer::getOrCreateTryCatchAnchor() {
  return getOrCreatePrivateFunction(
      module, builder, "LyEH_TryCatchAnchor",
      builder.getFunctionType({builder.getI64Type()}, {builder.getI1Type()}));
}

std::optional<std::int64_t> RuntimeBundleLowerer::currentTryHandlerId() const {
  mlir::Block *block = builder.getInsertionBlock();
  while (block) {
    auto found = tryHandlerIds.find(block);
    if (found != tryHandlerIds.end())
      return found->second;
    mlir::Operation *parent = block->getParentOp();
    block = parent ? parent->getBlock() : nullptr;
  }
  return std::nullopt;
}

void RuntimeBundleLowerer::emitTryCallSiteMarker(mlir::Location loc,
                                                 std::int64_t id) {
  mlir::Value idValue =
      mlir::arith::ConstantIntOp::create(builder, loc, id, 64).getResult();
  mlir::func::CallOp::create(builder, loc, getOrCreateTryCallSiteMarker(),
                             mlir::ValueRange{idValue});
}

void RuntimeBundleLowerer::emitTryCallSiteMarkerIfNeeded(mlir::Location loc) {
  if (std::optional<std::int64_t> id = currentTryHandlerId())
    emitTryCallSiteMarker(loc, *id);
}

mlir::LogicalResult RuntimeBundleLowerer::lowerTry(py::TryOp op) {
  bool hasExcept = !op.getExceptRegion().empty();
  bool hasFinally = !op.getFinallyRegion().empty();
  if (!hasExcept && !hasFinally)
    return op.emitError()
           << "py.try lowering requires an except or finally region";
  if (op.getTryRegion().empty())
    return op.emitError() << "py.try lowering requires a try region";

  mlir::Operation *tryOperation = op.getOperation();
  mlir::Block *parentBlock = tryOperation->getBlock();
  mlir::Region *parentRegion = parentBlock->getParent();
  mlir::Location loc = op.getLoc();
  std::optional<std::int64_t> enclosingHandlerId;
  for (mlir::Block *block = parentBlock; block;) {
    auto found = tryHandlerIds.find(block);
    if (found != tryHandlerIds.end()) {
      enclosingHandlerId = found->second;
      break;
    }
    mlir::Operation *parent = block->getParentOp();
    block = parent ? parent->getBlock() : nullptr;
  }

  mlir::Block *tryEntry = &op.getTryRegion().front();
  mlir::Block *exceptEntry =
      hasExcept ? &op.getExceptRegion().front() : nullptr;
  mlir::Block *finallyEntry =
      hasFinally ? &op.getFinallyRegion().front() : nullptr;
  std::int64_t handlerId = nextTryHandlerId++;
  std::optional<std::int64_t> finalHandlerId;
  if (hasFinally && hasExcept)
    finalHandlerId = nextTryHandlerId++;
  mlir::func::FuncOp discardCurrentException;
  if (hasExcept)
    discardCurrentException =
        getOrCreateDiscardCurrentException(module, builder);

  for (mlir::Block &block : op.getTryRegion())
    tryHandlerIds.try_emplace(&block, handlerId);
  if (finalHandlerId) {
    for (mlir::Block &block : op.getExceptRegion())
      tryHandlerIds.try_emplace(&block, *finalHandlerId);
  } else if (enclosingHandlerId) {
    for (mlir::Block &block : op.getExceptRegion())
      tryHandlerIds.try_emplace(&block, *enclosingHandlerId);
  }
  if (enclosingHandlerId)
    for (mlir::Block &block : op.getFinallyRegion())
      tryHandlerIds.try_emplace(&block, *enclosingHandlerId);

  mlir::Block *continuation =
      parentBlock->splitBlock(std::next(tryOperation->getIterator()));
  if (enclosingHandlerId)
    tryHandlerIds.try_emplace(continuation, *enclosingHandlerId);
  llvm::SmallVector<mlir::Value, 4> continuationArgs;
  continuationArgs.reserve(op.getNumResults());
  for (mlir::Value result : op.getResults()) {
    mlir::BlockArgument arg =
        continuation->addArgument(result.getType(), result.getLoc());
    continuationArgs.push_back(arg);
    result.replaceAllUsesWith(arg);
  }

  mlir::Block *exceptionalFinallyEntry = nullptr;
  mlir::Block *finallyDispatch = nullptr;
  mlir::Block *finallyRethrow = nullptr;
  if (hasFinally) {
    mlir::Region::iterator continuationIt = continuation->getIterator();
    exceptionalFinallyEntry = builder.createBlock(parentRegion, continuationIt);
    llvm::SmallVector<mlir::Type, 4> dispatchTypes;
    llvm::SmallVector<mlir::Location, 4> dispatchLocs;
    dispatchTypes.push_back(builder.getI1Type());
    dispatchLocs.push_back(loc);
    for (mlir::Value result : op.getResults()) {
      dispatchTypes.push_back(result.getType());
      dispatchLocs.push_back(result.getLoc());
    }
    finallyDispatch = builder.createBlock(parentRegion, continuationIt,
                                          dispatchTypes, dispatchLocs);
    finallyRethrow = builder.createBlock(parentRegion, continuationIt);
    if (enclosingHandlerId) {
      tryHandlerIds.try_emplace(finallyDispatch, *enclosingHandlerId);
      tryHandlerIds.try_emplace(finallyRethrow, *enclosingHandlerId);
    }

    mlir::Value finallyMode =
        finallyEntry->addArgument(builder.getI1Type(), loc);
    llvm::SmallVector<mlir::Value, 4> finallyCarriedValues;
    finallyCarriedValues.reserve(op.getNumResults());
    for (mlir::Value result : op.getResults())
      finallyCarriedValues.push_back(
          finallyEntry->addArgument(result.getType(), result.getLoc()));
    replaceYieldsWithFinallyEntry<py::TryYieldOp>(
        builder, op.getTryRegion(), finallyEntry, /*exceptional=*/false);
    if (hasExcept)
      replaceExceptYieldsWithFinallyEntry(
          builder, op.getExceptRegion(), finallyEntry, discardCurrentException);
    if (mlir::failed(replaceRaiseCurrentWithFinallyEntry(
            builder, op.getExceptRegion(), finallyEntry, op)))
      return mlir::failure();
    replaceFinallyYields(builder, op.getFinallyRegion(), finallyMode,
                         finallyCarriedValues, finallyDispatch);

    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(exceptionalFinallyEntry);
      mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>> operands =
          exceptionalFinallyOperands(builder, op);
      if (mlir::failed(operands))
        return mlir::failure();
      mlir::cf::BranchOp::create(builder, loc, finallyEntry, *operands);
    }
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(finallyDispatch);
      mlir::Value mode = finallyDispatch->getArgument(0);
      llvm::SmallVector<mlir::Value, 4> carriedValues;
      for (unsigned index = 0; index < op.getNumResults(); ++index)
        carriedValues.push_back(finallyDispatch->getArgument(index + 1));
      mlir::cf::CondBranchOp::create(builder, loc, mode, finallyRethrow,
                                     mlir::ValueRange{}, continuation,
                                     carriedValues);
    }
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(finallyRethrow);
      emitTryCallSiteMarkerIfNeeded(loc);
      mlir::func::CallOp::create(builder, loc,
                                 getOrCreateRethrowCurrent(module, builder),
                                 mlir::ValueRange{});
      mlir::cf::BranchOp::create(builder, loc, finallyRethrow);
    }
  } else {
    replaceYields<py::TryYieldOp>(builder, op.getTryRegion(), continuation);
    replaceExceptYields(builder, op.getExceptRegion(), continuation,
                        discardCurrentException);
  }

  auto continuationIt = continuation->getIterator();
  parentRegion->getBlocks().splice(continuationIt,
                                   op.getTryRegion().getBlocks());
  parentRegion->getBlocks().splice(continuationIt,
                                   op.getExceptRegion().getBlocks());
  parentRegion->getBlocks().splice(continuationIt,
                                   op.getFinallyRegion().getBlocks());

  if (hasExcept) {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(exceptEntry);
    mlir::Value idValue =
        mlir::arith::ConstantIntOp::create(builder, loc, handlerId, 64)
            .getResult();
    mlir::func::CallOp marker = mlir::func::CallOp::create(
        builder, loc, getOrCreateTryCatchMarker(), mlir::ValueRange{idValue});
    if (finalHandlerId) {
      mlir::Block *matchEntry =
          exceptEntry->splitBlock(std::next(marker->getIterator()));
      builder.setInsertionPointToEnd(exceptEntry);
      mlir::Value finalIdValue =
          mlir::arith::ConstantIntOp::create(builder, loc, *finalHandlerId, 64)
              .getResult();
      auto finalAnchor =
          mlir::func::CallOp::create(builder, loc, getOrCreateTryCatchAnchor(),
                                     mlir::ValueRange{finalIdValue});
      mlir::cf::CondBranchOp::create(
          builder, loc, finalAnchor.getResult(0), exceptionalFinallyEntry,
          mlir::ValueRange{}, matchEntry, mlir::ValueRange{});
    }
  }
  if (hasFinally) {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(exceptionalFinallyEntry);
    std::int64_t catchId = finalHandlerId ? *finalHandlerId : handlerId;
    mlir::Value idValue =
        mlir::arith::ConstantIntOp::create(builder, loc, catchId, 64)
            .getResult();
    mlir::func::CallOp::create(builder, loc, getOrCreateTryCatchMarker(),
                               mlir::ValueRange{idValue});
  }

  builder.setInsertionPoint(tryOperation);
  mlir::Value idValue =
      mlir::arith::ConstantIntOp::create(builder, loc, handlerId, 64)
          .getResult();
  auto anchor = mlir::func::CallOp::create(
      builder, loc, getOrCreateTryCatchAnchor(), mlir::ValueRange{idValue});
  mlir::Block *unwindEntry = hasExcept ? exceptEntry : exceptionalFinallyEntry;
  mlir::cf::CondBranchOp::create(builder, loc, anchor.getResult(0), unwindEntry,
                                 mlir::ValueRange{}, tryEntry,
                                 mlir::ValueRange{});
  tryOperation->erase();
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerStructuredTryOps() {
  llvm::SmallVector<py::TryOp, 8> tries;
  module.walk([&](py::TryOp op) { tries.push_back(op); });

  for (py::TryOp op : llvm::reverse(tries))
    if (mlir::failed(lowerTry(op)))
      return mlir::failure();
  return mlir::success();
}

} // namespace py::lowering
