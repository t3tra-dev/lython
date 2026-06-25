#include "RuntimeLowering/RuntimeLowering.h"

namespace py::runtime_lowering {
namespace {

void getLocInfo(mlir::Location loc, llvm::StringRef &filename,
                std::int64_t &line, std::int64_t &column) {
  if (auto fileLoc = mlir::dyn_cast<mlir::FileLineColLoc>(loc)) {
    filename = fileLoc.getFilename().getValue();
    line = static_cast<std::int64_t>(fileLoc.getLine());
    column = static_cast<std::int64_t>(fileLoc.getColumn());
    return;
  }
  if (auto nameLoc = mlir::dyn_cast<mlir::NameLoc>(loc)) {
    getLocInfo(nameLoc.getChildLoc(), filename, line, column);
    return;
  }
  if (auto fused = mlir::dyn_cast<mlir::FusedLoc>(loc)) {
    for (mlir::Location subloc : fused.getLocations()) {
      if (mlir::isa<mlir::FileLineColLoc>(subloc)) {
        getLocInfo(subloc, filename, line, column);
        return;
      }
    }
  }
  filename = "<unknown>";
  line = 0;
  column = 0;
}

llvm::StringRef currentCallableName(mlir::Operation *op) {
  if (!op)
    return "<unknown>";
  if (auto function = op->getParentOfType<mlir::func::FuncOp>()) {
    llvm::StringRef name = function.getName();
    return name == "__main__" ? "<module>" : name;
  }
  return "<unknown>";
}

void createDeadContinuation(mlir::OpBuilder &builder, mlir::Operation *op) {
  mlir::Block *current = op->getBlock();
  mlir::Block *dead = builder.createBlock(current->getParent(),
                                          std::next(current->getIterator()));
  builder.setInsertionPoint(op);
  builder.create<mlir::cf::BranchOp>(op->getLoc(), dead);
  builder.setInsertionPointToStart(dead);
  builder.create<mlir::cf::BranchOp>(op->getLoc(), dead);
}

mlir::func::FuncOp getOrCreateRethrow(mlir::ModuleOp module,
                                      mlir::OpBuilder &builder) {
  constexpr llvm::StringLiteral kName{"LyEH_RethrowCurrent"};
  if (auto existing = module.lookupSymbol<mlir::func::FuncOp>(kName))
    return existing;
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(module.getBody());
  auto functionType = builder.getFunctionType({}, {});
  auto function =
      builder.create<mlir::func::FuncOp>(module.getLoc(), kName, functionType);
  function.setPrivate();
  return function;
}

mlir::func::FuncOp getOrCreateTracebackPush(mlir::ModuleOp module,
                                            mlir::OpBuilder &builder) {
  constexpr llvm::StringLiteral kName{"LyTraceback_Push"};
  if (auto existing = module.lookupSymbol<mlir::func::FuncOp>(kName))
    return existing;
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(module.getBody());
  auto bytesType =
      mlir::MemRefType::get({mlir::ShapedType::kDynamic}, builder.getI8Type());
  auto functionType = builder.getFunctionType(
      {bytesType, bytesType, builder.getI32Type(), builder.getI32Type()}, {});
  auto function =
      builder.create<mlir::func::FuncOp>(module.getLoc(), kName, functionType);
  function.setPrivate();
  return function;
}

} // namespace

mlir::LogicalResult
RuntimeBundleLowerer::emitTracebackFrame(mlir::Operation *op) {
  llvm::StringRef filename;
  std::int64_t line = 0;
  std::int64_t column = 0;
  getLocInfo(op->getLoc(), filename, line, column);

  mlir::func::FuncOp tracebackPush = getOrCreateTracebackPush(module, builder);
  mlir::Value file = materializeByteBuffer(op->getLoc(), filename);
  mlir::Value function =
      materializeByteBuffer(op->getLoc(), currentCallableName(op));
  mlir::Value lineValue =
      builder.create<mlir::arith::ConstantIntOp>(op->getLoc(), line, 32)
          .getResult();
  mlir::Value columnValue =
      builder.create<mlir::arith::ConstantIntOp>(op->getLoc(), column, 32)
          .getResult();
  builder.create<mlir::func::CallOp>(
      op->getLoc(), tracebackPush,
      mlir::ValueRange{file, function, lineValue, columnValue});
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerRaise(py::RaiseOp op) {
  const RuntimeBundle *exception = bundleFor(op.getException());
  if (!exception)
    return op.emitError() << "raised exception has no lowered runtime bundle";

  std::optional<RuntimeSymbol> symbol =
      manifest.primitive(exception->contractName(), "raise");
  if (!symbol)
    return op.emitError() << "runtime manifest has no "
                          << exception->contractName() << ".raise primitive";

  llvm::SmallVector<const RuntimeBundle *, 1> sources{exception};
  llvm::SmallVector<mlir::Value, 8> operands;
  builder.setInsertionPoint(op);
  if (mlir::failed(emitTracebackFrame(op.getOperation())))
    return mlir::failure();
  if (mlir::failed(buildRuntimeCallOperands(op, *symbol, sources, operands,
                                            /*allowUnusedSources=*/false)))
    return mlir::failure();

  RuntimeBundleLowerer::createRuntimeCall(op.getLoc(), *symbol, operands);
  createDeadContinuation(builder, op.getOperation());
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult
RuntimeBundleLowerer::lowerRaiseCurrent(py::RaiseCurrentOp op) {
  mlir::func::FuncOp rethrow = getOrCreateRethrow(module, builder);
  builder.setInsertionPoint(op);
  builder.create<mlir::func::CallOp>(op.getLoc(), rethrow, mlir::ValueRange{});
  createDeadContinuation(builder, op.getOperation());
  erase.push_back(op);
  return mlir::success();
}

} // namespace py::runtime_lowering
