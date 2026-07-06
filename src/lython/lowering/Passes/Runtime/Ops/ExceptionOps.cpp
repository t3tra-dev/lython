#include "Runtime/Core/Lowerer.h"

namespace py::lowering {
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
  mlir::cf::BranchOp::create(builder, op->getLoc(), dead);
  builder.setInsertionPointToStart(dead);
  mlir::cf::BranchOp::create(builder, op->getLoc(), dead);
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
      mlir::func::FuncOp::create(builder, module.getLoc(), kName, functionType);
  function.setPrivate();
  return function;
}

mlir::func::FuncOp getOrCreateClassIdMatches(mlir::ModuleOp module,
                                             mlir::OpBuilder &builder) {
  constexpr llvm::StringLiteral kName{"LyEH_ClassIdMatches"};
  if (auto existing = module.lookupSymbol<mlir::func::FuncOp>(kName))
    return existing;
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(module.getBody());
  auto functionType = builder.getFunctionType(
      {builder.getI64Type(), builder.getI64Type()}, {builder.getI1Type()});
  auto function =
      mlir::func::FuncOp::create(builder, module.getLoc(), kName, functionType);
  function.setPrivate();
  return function;
}

mlir::func::FuncOp
getOrCreateDiscardCurrentExceptionIfMatches(mlir::ModuleOp module,
                                            mlir::OpBuilder &builder) {
  constexpr llvm::StringLiteral kName{"LyEH_DiscardCurrentExceptionIfMatches"};
  if (auto existing = module.lookupSymbol<mlir::func::FuncOp>(kName))
    return existing;
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(module.getBody());
  auto functionType =
      builder.getFunctionType({builder.getI64Type()}, {builder.getI1Type()});
  auto function =
      mlir::func::FuncOp::create(builder, module.getLoc(), kName, functionType);
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
      mlir::func::FuncOp::create(builder, module.getLoc(), kName, functionType);
  function.setPrivate();
  return function;
}

mlir::FailureOr<std::int64_t>
handlerClassId(mlir::Operation *op, mlir::Type handler,
               const RuntimeManifestIndex &manifest) {
  auto handlerType = mlir::dyn_cast<py::TypeType>(handler);
  if (!handlerType)
    return op->emitError() << "exception handler must be type[T]";
  auto handlerContract =
      mlir::dyn_cast<py::ContractType>(handlerType.getInstanceType());
  if (!handlerContract)
    return op->emitError() << "exception handler must name a manifest contract";
  std::optional<std::int64_t> classId =
      manifest.classId(handlerContract.getContractName());
  if (!classId)
    return op->emitError() << "runtime manifest has no class id for exception "
                           << "handler " << handlerContract.getContractName();
  return *classId;
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
      mlir::arith::ConstantIntOp::create(builder, op->getLoc(), line, 32)
          .getResult();
  mlir::Value columnValue =
      mlir::arith::ConstantIntOp::create(builder, op->getLoc(), column, 32)
          .getResult();
  mlir::func::CallOp::create(
      builder, op->getLoc(), tracebackPush,
      mlir::ValueRange{file, function, lineValue, columnValue});
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::emitRuntimeException(
    mlir::Operation *op, llvm::StringRef contract, llvm::StringRef message) {
  RuntimeBundle messageObject;
  if (mlir::failed(RuntimeBundleLowerer::materializeStringObject(
          op, message, messageObject)))
    return mlir::failure();
  return RuntimeBundleLowerer::emitRuntimeExceptionFromMessageObject(
      op, contract, messageObject);
}

mlir::LogicalResult RuntimeBundleLowerer::emitRuntimeExceptionFromMessageObject(
    mlir::Operation *op, llvm::StringRef contract,
    const RuntimeBundle &messageObject) {
  mlir::Type exceptionType = runtimeContractType(context, contract);
  RuntimeBundle classObject = RuntimeBundle::typeObject(
      runtimeContractType(context, "builtins.type"), exceptionType);

  std::optional<RuntimeSymbol> initializer =
      manifest.initializer(contract, "__new__");
  if (!initializer)
    return op->emitError() << "runtime manifest has no " << contract
                           << ".__new__ initializer";

  llvm::SmallVector<mlir::Value, 8> newOperands;
  if (mlir::failed(buildRuntimeCallOperands(op, *initializer, {}, newOperands,
                                            /*allowUnusedSources=*/true,
                                            &classObject)))
    return mlir::failure();
  mlir::func::CallOp newCall = RuntimeBundleLowerer::createRuntimeCall(
      op->getLoc(), *initializer, newOperands);
  RuntimeBundle exception;
  if (mlir::failed(RuntimeBundleLowerer::bundleRuntimeResults(
          op, exceptionType, newCall, exception)))
    return mlir::failure();

  RuntimeBundle message = messageObject;
  if (message.contractName() != "builtins.str")
    return op->emitError()
           << contract
           << " runtime exception message must be builtins.str, "
              "got "
           << message.contractName();

  mlir::Location loc = op->getLoc();
  std::optional<RuntimeSymbol> init = manifest.method(contract, "__init__");
  if (!init)
    return op->emitError() << "runtime manifest has no " << contract
                           << ".__init__ method";
  llvm::SmallVector<const RuntimeBundle *, 2> initSources{&exception, &message};
  llvm::SmallVector<mlir::Value, 8> initOperands;
  if (mlir::failed(buildRuntimeCallOperands(op, *init, initSources,
                                            initOperands,
                                            /*allowUnusedSources=*/true)))
    return mlir::failure();
  mlir::func::CallOp initCall =
      RuntimeBundleLowerer::createRuntimeCall(loc, *init, initOperands);
  if (initCall.getNumResults() != 0 &&
      mlir::failed(RuntimeBundleLowerer::bundleRuntimeResults(
          op, exceptionType, initCall, exception)))
    return mlir::failure();

  std::optional<RuntimeSymbol> raise = manifest.primitive(contract, "raise");
  if (!raise)
    return op->emitError() << "runtime manifest has no " << contract
                           << ".raise primitive";
  llvm::SmallVector<const RuntimeBundle *, 1> raiseSources{&exception};
  llvm::SmallVector<mlir::Value, 8> raiseOperands;
  if (mlir::failed(emitTracebackFrame(op)))
    return mlir::failure();
  if (mlir::failed(buildRuntimeCallOperands(op, *raise, raiseSources,
                                            raiseOperands,
                                            /*allowUnusedSources=*/false)))
    return mlir::failure();
  RuntimeBundleLowerer::createRuntimeCall(loc, *raise, raiseOperands);
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
  op.erase();
  return mlir::success();
}

mlir::LogicalResult
RuntimeBundleLowerer::lowerRaiseCurrent(py::RaiseCurrentOp op) {
  mlir::func::FuncOp rethrow = getOrCreateRethrow(module, builder);
  builder.setInsertionPoint(op);
  mlir::func::CallOp::create(builder, op.getLoc(), rethrow, mlir::ValueRange{});
  createDeadContinuation(builder, op.getOperation());
  op.erase();
  return mlir::success();
}

mlir::LogicalResult
RuntimeBundleLowerer::lowerExceptMatch(py::ExceptMatchOp op) {
  const RuntimeBundle *exception = bundleFor(op.getException());
  if (!exception)
    return op.emitError() << "except.match exception has no lowered runtime "
                             "bundle";

  mlir::FailureOr<std::int64_t> handlerClassIdValue =
      handlerClassId(op.getOperation(), op.getHandler(), manifest);
  if (mlir::failed(handlerClassIdValue))
    return mlir::failure();

  llvm::ArrayRef<mlir::Value> values = exception->physicalValues();
  if (values.empty())
    return op.emitError() << exception->contractName()
                          << " exception has no physical header";
  auto headerType = mlir::dyn_cast<mlir::MemRefType>(values.front().getType());
  if (!headerType || headerType.getRank() != 1 ||
      !headerType.getElementType().isInteger(64))
    return op.emitError() << exception->contractName()
                          << " exception header is not a rank-1 i64 memref";

  builder.setInsertionPoint(op);
  mlir::Value classSlot =
      mlir::arith::ConstantIndexOp::create(builder, op.getLoc(), 2).getResult();
  mlir::Value exceptionClassId =
      mlir::memref::LoadOp::create(builder, op.getLoc(), values.front(),
                                   mlir::ValueRange{classSlot})
          .getResult();
  mlir::Value handlerId = mlir::arith::ConstantIntOp::create(
                              builder, op.getLoc(), *handlerClassIdValue, 64)
                              .getResult();
  mlir::func::FuncOp classIdMatches =
      getOrCreateClassIdMatches(module, builder);
  auto call =
      mlir::func::CallOp::create(builder, op.getLoc(), classIdMatches,
                                 mlir::ValueRange{exceptionClassId, handlerId});
  op.getResult().replaceAllUsesWith(call.getResult(0));
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult
RuntimeBundleLowerer::lowerExceptCurrentMatch(py::ExceptCurrentMatchOp op) {
  mlir::FailureOr<std::int64_t> handlerId =
      handlerClassId(op.getOperation(), op.getHandler(), manifest);
  if (mlir::failed(handlerId))
    return mlir::failure();

  builder.setInsertionPoint(op);
  mlir::Value handler =
      mlir::arith::ConstantIntOp::create(builder, op.getLoc(), *handlerId, 64)
          .getResult();
  mlir::func::FuncOp discardIfMatches =
      getOrCreateDiscardCurrentExceptionIfMatches(module, builder);
  auto call = mlir::func::CallOp::create(builder, op.getLoc(), discardIfMatches,
                                         mlir::ValueRange{handler});
  op.getResult().replaceAllUsesWith(call.getResult(0));
  erase.push_back(op);
  return mlir::success();
}

} // namespace py::lowering
