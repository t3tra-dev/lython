#include "RuntimeLowering/RuntimeBundleLowerer.h"

#include <iterator>

namespace py::runtime_lowering {
namespace {

mlir::func::FuncOp getOrCreatePrivateFunction(mlir::ModuleOp module,
                                              mlir::OpBuilder &builder,
                                              llvm::StringRef name,
                                              mlir::FunctionType type) {
  if (auto existing = module.lookupSymbol<mlir::func::FuncOp>(name))
    return existing;

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(module.getBody());
  auto function = builder.create<mlir::func::FuncOp>(module.getLoc(), name,
                                                     type);
  function.setPrivate();
  return function;
}

template <typename YieldOp>
void replaceYields(mlir::OpBuilder &builder, mlir::Region &region,
                   mlir::Block *continuation) {
  llvm::SmallVector<YieldOp, 8> yields;
  region.walk([&](YieldOp yield) { yields.push_back(yield); });
  for (YieldOp yield : yields) {
    builder.setInsertionPoint(yield);
    builder.create<mlir::cf::BranchOp>(yield.getLoc(), continuation,
                                       yield.getOperands());
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
  if (!block)
    return std::nullopt;
  auto found = tryHandlerIds.find(block);
  if (found == tryHandlerIds.end())
    return std::nullopt;
  return found->second;
}

void RuntimeBundleLowerer::emitTryCallSiteMarker(mlir::Location loc,
                                                 std::int64_t id) {
  mlir::Value idValue =
      builder.create<mlir::arith::ConstantIntOp>(loc, id, 64).getResult();
  builder.create<mlir::func::CallOp>(loc, getOrCreateTryCallSiteMarker(),
                                     mlir::ValueRange{idValue});
}

void RuntimeBundleLowerer::emitTryCallSiteMarkerIfNeeded(mlir::Location loc) {
  if (std::optional<std::int64_t> id = currentTryHandlerId())
    emitTryCallSiteMarker(loc, *id);
}

mlir::LogicalResult RuntimeBundleLowerer::lowerTry(py::TryOp op) {
  if (!op.getFinallyRegion().empty())
    return op.emitError()
           << "py.try finally lowering is not implemented yet";
  if (op.getExceptRegion().empty())
    return op.emitError() << "py.try lowering requires an except region";
  if (op.getTryRegion().empty())
    return op.emitError() << "py.try lowering requires a try region";

  mlir::Operation *tryOperation = op.getOperation();
  mlir::Block *parentBlock = tryOperation->getBlock();
  mlir::Region *parentRegion = parentBlock->getParent();
  mlir::Location loc = op.getLoc();

  mlir::Block *tryEntry = &op.getTryRegion().front();
  mlir::Block *exceptEntry = &op.getExceptRegion().front();
  std::int64_t handlerId = nextTryHandlerId++;

  for (mlir::Block &block : op.getTryRegion())
    tryHandlerIds.try_emplace(&block, handlerId);

  mlir::Block *continuation =
      parentBlock->splitBlock(std::next(tryOperation->getIterator()));
  llvm::SmallVector<mlir::Value, 4> continuationArgs;
  continuationArgs.reserve(op.getNumResults());
  for (mlir::Value result : op.getResults()) {
    mlir::BlockArgument arg =
        continuation->addArgument(result.getType(), result.getLoc());
    continuationArgs.push_back(arg);
    result.replaceAllUsesWith(arg);
  }

  replaceYields<py::TryYieldOp>(builder, op.getTryRegion(), continuation);
  replaceYields<py::ExceptYieldOp>(builder, op.getExceptRegion(),
                                   continuation);

  auto continuationIt = continuation->getIterator();
  parentRegion->getBlocks().splice(continuationIt,
                                   op.getTryRegion().getBlocks());
  parentRegion->getBlocks().splice(continuationIt,
                                   op.getExceptRegion().getBlocks());

  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(exceptEntry);
    mlir::Value idValue =
        builder.create<mlir::arith::ConstantIntOp>(loc, handlerId, 64)
            .getResult();
    builder.create<mlir::func::CallOp>(loc, getOrCreateTryCatchMarker(),
                                       mlir::ValueRange{idValue});
  }

  builder.setInsertionPoint(tryOperation);
  mlir::Value idValue =
      builder.create<mlir::arith::ConstantIntOp>(loc, handlerId, 64)
          .getResult();
  auto anchor = builder.create<mlir::func::CallOp>(
      loc, getOrCreateTryCatchAnchor(), mlir::ValueRange{idValue});
  builder.create<mlir::cf::CondBranchOp>(
      loc, anchor.getResult(0), exceptEntry, mlir::ValueRange{}, tryEntry,
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

} // namespace py::runtime_lowering
