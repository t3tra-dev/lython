#include "Common/PythonSourceRange.h"
#include "Runtime/Core/Lowerer.h"

#include "mlir/Dialect/SCF/IR/SCF.h"

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
    if (name == "__main__")
      return "<module>";
    // Specialization and generator state-machine clones carry ABI suffixes;
    // the traceback shows the Python-level name.
    for (llvm::StringRef marker : {"__lyrt_prim", "__lyrt_gen"}) {
      std::size_t suffix = name.find(marker);
      if (suffix != llvm::StringRef::npos)
        name = name.take_front(suffix);
    }
    return name;
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

mlir::func::FuncOp getOrCreateClassIdMatches(mlir::ModuleOp module,
                                             mlir::OpBuilder &builder) {
  return getOrCreatePrivateFunction(
      module, builder, "LyEH_ClassIdMatches",
      builder.getFunctionType({builder.getI64Type(), builder.getI64Type()},
                              {builder.getI1Type()}));
}

mlir::func::FuncOp
getOrCreateCurrentExceptionMatches(mlir::ModuleOp module,
                                   mlir::OpBuilder &builder) {
  return getOrCreatePrivateFunction(
      module, builder, "LyEH_CurrentExceptionMatches",
      builder.getFunctionType({builder.getI64Type()}, {builder.getI1Type()}));
}

mlir::func::FuncOp getOrCreateTracebackPush(mlir::ModuleOp module,
                                            mlir::OpBuilder &builder) {
  auto bytesType =
      mlir::MemRefType::get({mlir::ShapedType::kDynamic}, builder.getI8Type());
  mlir::Type i32 = builder.getI32Type();
  return getOrCreatePrivateFunction(
      module, builder, "LyTraceback_Push",
      builder.getFunctionType({bytesType, bytesType, i32, i32, i32, i32, i32},
                              {}));
}

} // namespace

mlir::FailureOr<std::int64_t>
RuntimeBundleLowerer::handlerClassId(mlir::Operation *op,
                                     mlir::Type handler) const {
  auto handlerType = mlir::dyn_cast<py::TypeType>(handler);
  if (!handlerType)
    return op->emitError() << "exception handler must be type[T]";
  auto handlerContract =
      mlir::dyn_cast<py::ContractType>(handlerType.getInstanceType());
  if (!handlerContract)
    return op->emitError() << "exception handler must name a manifest contract";
  // Manifest exceptions and source exception classes both resolve here (the
  // latter through their compiler-assigned ids).
  std::optional<std::int64_t> classId =
      RuntimeBundleLowerer::runtimeClassIdForContract(handlerContract);
  if (!classId)
    return op->emitError() << "runtime manifest has no class id for exception "
                           << "handler " << handlerContract.getContractName();
  return *classId;
}

mlir::LogicalResult
RuntimeBundleLowerer::emitTracebackFrame(mlir::Operation *op,
                                         bool stashCurrentException) {
  llvm::StringRef filename;
  std::int64_t line = 0;
  std::int64_t column = 0;
  getLocInfo(op->getLoc(), filename, line, column);

  // Sub-line source ranges become caret anchors (`d[k]` -> `~^^^`). Raise
  // statements never carry anchors: CPython only extracts anchors from
  // expression segments (calls, subscripts, operators), and a raise
  // statement's position is the whole statement.
  std::int64_t endLine = line;
  std::int64_t endColumn = 0;
  std::int64_t hasMarker = 0;
  if (!mlir::isa<py::RaiseOp, py::RaiseCurrentOp>(op)) {
    if (std::optional<PythonSourceRange> range =
            pythonSourceRange(op->getLoc())) {
      if (range->endLine == range->line && range->endColumn > range->column) {
        endLine = range->endLine;
        endColumn = range->endColumn;
        // Marker 2 = plain range carets: CPython extracts `~`/`^` anchors
        // only from call/binop/subscript nodes, so a suspension frame
        // (throw() delivery at a yield) must not split at punctuation its
        // operand happens to contain.
        hasMarker = mlir::isa<py::YieldValueOp>(op) ? 2 : 1;
      }
    }
  }

  // A frame push announces a fresh raise. If another exception is still being
  // handled here, it becomes the new exception's implicit __context__; the
  // stash must run before the push so the handled exception's traceback
  // snapshot does not swallow the new raise-site frame. Re-raises of the
  // current exception skip this (their traceback must stay in place).
  if (stashCurrentException)
    mlir::func::CallOp::create(builder, op->getLoc(),
                               getOrCreateStashCurrentAsContext(module,
                                                                builder),
                               mlir::ValueRange{});

  mlir::func::FuncOp tracebackPush = getOrCreateTracebackPush(module, builder);
  mlir::Value file = materializeByteBuffer(op->getLoc(), filename);
  mlir::Value function =
      materializeByteBuffer(op->getLoc(), currentCallableName(op));
  auto i32Const = [&](std::int64_t value) {
    return mlir::arith::ConstantIntOp::create(builder, op->getLoc(), value, 32)
        .getResult();
  };
  mlir::func::CallOp::create(
      builder, op->getLoc(), tracebackPush,
      mlir::ValueRange{file, function, i32Const(line), i32Const(column),
                       i32Const(endLine), i32Const(endColumn),
                       i32Const(hasMarker)});
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

// LyEH_SetCurrentCause(exception triple): records the raised exception's
// explicit `__cause__` (raise ... from <expr>). The runtime borrows the
// operands and retains what it stores, so the call is a plain use for the
// ownership machinery.
mlir::LogicalResult
RuntimeBundleLowerer::emitSetCurrentCause(mlir::Operation *op,
                                          const RuntimeBundle &cause) {
  llvm::ArrayRef<mlir::Value> values = cause.physicalValues();
  auto headerType = values.empty()
                        ? mlir::MemRefType()
                        : mlir::dyn_cast<mlir::MemRefType>(
                              values.front().getType());
  if (values.size() != 3 || !headerType || headerType.getRank() != 1 ||
      !headerType.getElementType().isInteger(64) ||
      !manifest.classId(cause.contractName()))
    return op->emitError()
           << "raise ... from cause must be a runtime exception instance, got "
           << cause.contractName();
  mlir::func::FuncOp setCause = getOrCreatePrivateFunction(
      module, builder, "LyEH_SetCurrentCause",
      builder.getFunctionType({values[0].getType(), values[1].getType(),
                               values[2].getType()},
                              {}));
  mlir::func::CallOp::create(builder, op->getLoc(), setCause, values);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerRaise(py::RaiseOp op) {
  const RuntimeBundle *exception = bundleFor(op.getException());
  if (!exception)
    return op.emitError() << "raised exception has no lowered runtime bundle";

  const RuntimeBundle *cause = nullptr;
  if (mlir::Value causeValue = op.getCause()) {
    cause = bundleFor(causeValue);
    if (!cause)
      return op.emitError() << "raise cause has no lowered runtime bundle";
  }

  if (exception->objectEvidence.hasFlag(kCurrentExceptionBorrowFlag)) {
    // Re-raise of the exception being handled: its traceback and context stay
    // in place; only an explicit cause / from None annotation is recorded.
    mlir::func::FuncOp rethrow = getOrCreateRethrowCurrent(module, builder);
    builder.setInsertionPoint(op);
    if (cause) {
      if (mlir::failed(emitSetCurrentCause(op.getOperation(), *cause)))
        return mlir::failure();
    } else if (op.getFromNone()) {
      mlir::func::CallOp::create(builder, op.getLoc(),
                                 getOrCreateSetCurrentSuppress(module, builder),
                                 mlir::ValueRange{});
    }
    if (mlir::failed(emitTracebackFrame(op.getOperation(),
                                        /*stashCurrentException=*/false)))
      return mlir::failure();
    emitTryCallSiteMarkerIfNeeded(op.getLoc());
    mlir::func::CallOp::create(builder, op.getLoc(), rethrow,
                               mlir::ValueRange{});
    createDeadContinuation(builder, op.getOperation());
    op.erase();
    return mlir::success();
  }

  if (cause || op.getFromNone()) {
    // The stash must precede the cause annotation: `raise X from e` where `e`
    // is the exception being handled shares the freshly stashed context node
    // as the cause instead of building a second reference chain.
    builder.setInsertionPoint(op);
    mlir::func::CallOp::create(builder, op.getLoc(),
                               getOrCreateStashCurrentAsContext(module,
                                                                builder),
                               mlir::ValueRange{});
    if (cause) {
      if (mlir::failed(emitSetCurrentCause(op.getOperation(), *cause)))
        return mlir::failure();
    } else {
      mlir::func::CallOp::create(builder, op.getLoc(),
                                 getOrCreateSetCurrentSuppress(module, builder),
                                 mlir::ValueRange{});
    }
  }

  if (mlir::failed(RuntimeBundleLowerer::emitRaiseExceptionBundle(
          op.getOperation(), *exception)))
    return mlir::failure();
  createDeadContinuation(builder, op.getOperation());
  op.erase();
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::emitRaiseExceptionBundle(
    mlir::Operation *op, const RuntimeBundle &exception) {
  std::optional<RuntimeSymbol> symbol =
      manifest.primitive(exception.contractName(), "raise");
  if (!symbol)
    // User exception classes raise through the builtin ancestor: the object
    // already carries the source class's id in its header.
    if (std::optional<std::string> ancestor =
            RuntimeBundleLowerer::exceptionAncestorContractFor(
                exception.contract))
      symbol = manifest.primitive(*ancestor, "raise");
  if (!symbol)
    return op->emitError() << "runtime manifest has no "
                           << exception.contractName() << ".raise primitive";

  llvm::SmallVector<const RuntimeBundle *, 1> sources{&exception};
  llvm::SmallVector<mlir::Value, 8> operands;
  builder.setInsertionPoint(op);
  if (mlir::failed(emitTracebackFrame(op)))
    return mlir::failure();
  if (mlir::failed(buildRuntimeCallOperands(op, *symbol, sources, operands,
                                            /*allowUnusedSources=*/false)))
    return mlir::failure();

  RuntimeBundleLowerer::createRuntimeCall(op->getLoc(), *symbol, operands);
  return mlir::success();
}

mlir::LogicalResult
RuntimeBundleLowerer::lowerRaiseCurrent(py::RaiseCurrentOp op) {
  mlir::func::FuncOp rethrow = getOrCreateRethrowCurrent(module, builder);
  builder.setInsertionPoint(op);
  emitTryCallSiteMarkerIfNeeded(op.getLoc());
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
      handlerClassId(op.getOperation(), op.getHandler());
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
      handlerClassId(op.getOperation(), op.getHandler());
  if (mlir::failed(handlerId))
    return mlir::failure();

  builder.setInsertionPoint(op);
  mlir::Value handler =
      mlir::arith::ConstantIntOp::create(builder, op.getLoc(), *handlerId, 64)
          .getResult();
  mlir::func::FuncOp currentMatches =
      getOrCreateCurrentExceptionMatches(module, builder);
  auto call = mlir::func::CallOp::create(builder, op.getLoc(), currentMatches,
                                         mlir::ValueRange{handler});
  op.getResult().replaceAllUsesWith(call.getResult(0));
  erase.push_back(op);
  return mlir::success();
}

// except* (PEP 654). The star frame lives in the native runtime; the split
// and combine steps are manifest primitives so the group machinery stays in
// builtins.mlir, and this glue only branches between them.

namespace {

mlir::func::FuncOp getOrCreateStarVoidFn(mlir::ModuleOp module,
                                         mlir::OpBuilder &builder,
                                         llvm::StringRef name) {
  return getOrCreatePrivateFunction(module, builder, name,
                                    builder.getFunctionType({}, {}));
}

llvm::SmallVector<mlir::Type, 3> exceptionTripleTypes(mlir::OpBuilder &b) {
  return {mlir::MemRefType::get({3}, b.getI64Type()),
          mlir::MemRefType::get({2}, b.getI64Type()),
          mlir::MemRefType::get({mlir::ShapedType::kDynamic}, b.getI8Type())};
}

} // namespace

mlir::LogicalResult RuntimeBundleLowerer::lowerStarBegin(py::StarBeginOp op) {
  builder.setInsertionPoint(op);
  mlir::func::CallOp::create(
      builder, op.getLoc(), getOrCreateStarVoidFn(module, builder,
                                                  "LyEH_StarBegin"),
      mlir::ValueRange{});
  op.erase();
  return mlir::success();
}

mlir::LogicalResult
RuntimeBundleLowerer::lowerExceptStarMatch(py::ExceptStarMatchOp op) {
  mlir::FailureOr<std::int64_t> handlerId =
      handlerClassId(op.getOperation(), op.getHandler());
  if (mlir::failed(handlerId))
    return mlir::failure();
  std::optional<RuntimeSymbol> split =
      manifest.primitive("builtins.BaseException", "star_split");
  if (!split)
    return op.emitError()
           << "runtime manifest has no BaseException star_split primitive";

  context->loadDialect<mlir::scf::SCFDialect>();
  mlir::Location loc = op.getLoc();
  builder.setInsertionPoint(op);
  llvm::SmallVector<mlir::Type, 3> triple = exceptionTripleTypes(builder);
  mlir::func::FuncOp residualParts = getOrCreatePrivateFunction(
      module, builder, "LyEH_StarResidualParts",
      builder.getFunctionType({}, triple));
  llvm::SmallVector<mlir::Type, 8> applyInputs{builder.getI1Type()};
  applyInputs.append(triple.begin(), triple.end());
  applyInputs.append(triple.begin(), triple.end());
  mlir::func::FuncOp applyMatch = getOrCreatePrivateFunction(
      module, builder, "LyEH_StarApplyMatch",
      builder.getFunctionType(applyInputs, {}));
  mlir::func::FuncOp hasResidual = getOrCreatePrivateFunction(
      module, builder, "LyEH_StarHasResidual",
      builder.getFunctionType({}, {builder.getI1Type()}));
  mlir::func::FuncOp discardSplit = getOrCreatePrivateFunction(
      module, builder, "LyEH_StarDiscardSplit",
      builder.getFunctionType(triple, {}));

  mlir::Value residual =
      mlir::func::CallOp::create(builder, loc, hasResidual, mlir::ValueRange{})
          .getResult(0);
  auto outer = mlir::scf::IfOp::create(builder, loc,
                                       mlir::TypeRange{builder.getI1Type()},
                                       residual, /*withElseRegion=*/true);
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&outer.getThenRegion().front());
    mlir::func::CallOp parts = mlir::func::CallOp::create(
        builder, loc, residualParts, mlir::ValueRange{});
    mlir::Value handler =
        mlir::arith::ConstantIntOp::create(builder, loc, *handlerId, 64)
            .getResult();
    mlir::func::CallOp splitCall = RuntimeBundleLowerer::createRuntimeCall(
        loc, *split,
        {parts.getResult(0), parts.getResult(1), parts.getResult(2), handler});
    if (splitCall.getNumResults() != 8)
      return op.emitError() << "star_split must return two flagged triples";
    mlir::Value matched = splitCall.getResult(0);
    auto applyIf = mlir::scf::IfOp::create(builder, loc, mlir::TypeRange{},
                                           matched, /*withElseRegion=*/true);
    {
      mlir::OpBuilder::InsertionGuard applyGuard(builder);
      builder.setInsertionPointToStart(&applyIf.getThenRegion().front());
      mlir::func::CallOp::create(
          builder, loc, applyMatch,
          mlir::ValueRange{splitCall.getResult(4), splitCall.getResult(1),
                           splitCall.getResult(2), splitCall.getResult(3),
                           splitCall.getResult(5), splitCall.getResult(6),
                           splitCall.getResult(7)});

      // Nothing matched: `applyMatch` -- the only consumer of the halves
      // star_split retained -- does not run, so the leftover it handed back is
      // this clause's to discharge. The frame's residual still holds its own
      // reference and is unchanged, which is why the half is DROPPED rather
      // than installed.
      builder.setInsertionPointToStart(&applyIf.getElseRegion().front());
      mlir::func::CallOp::create(
          builder, loc, discardSplit,
          mlir::ValueRange{splitCall.getResult(5), splitCall.getResult(6),
                           splitCall.getResult(7)});
    }
    mlir::scf::YieldOp::create(builder, loc, mlir::ValueRange{matched});
    builder.setInsertionPointToStart(&outer.getElseRegion().front());
    mlir::Value noMatch =
        mlir::arith::ConstantIntOp::create(builder, loc, 0, 1).getResult();
    mlir::scf::YieldOp::create(builder, loc, mlir::ValueRange{noMatch});
  }
  op.getResult().replaceAllUsesWith(outer.getResult(0));
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult
RuntimeBundleLowerer::lowerStarCollect(py::StarCollectOp op) {
  builder.setInsertionPoint(op);
  mlir::func::CallOp::create(
      builder, op.getLoc(), getOrCreateStarVoidFn(module, builder,
                                                  "LyEH_StarCollect"),
      mlir::ValueRange{});
  op.erase();
  return mlir::success();
}

mlir::LogicalResult
RuntimeBundleLowerer::lowerStarBodyEnd(py::StarBodyEndOp op) {
  builder.setInsertionPoint(op);
  mlir::func::CallOp::create(builder, op.getLoc(),
                             getOrCreateDiscardCurrentException(module,
                                                                builder),
                             mlir::ValueRange{});
  op.erase();
  return mlir::success();
}

mlir::LogicalResult
RuntimeBundleLowerer::lowerStarFinish(py::StarFinishOp op) {
  std::optional<RuntimeSymbol> combine =
      manifest.primitive("builtins.BaseException", "star_combine");
  if (!combine)
    return op.emitError()
           << "runtime manifest has no BaseException star_combine primitive";

  context->loadDialect<mlir::scf::SCFDialect>();
  mlir::Location loc = op.getLoc();
  builder.setInsertionPoint(op);
  llvm::SmallVector<mlir::Type, 3> triple = exceptionTripleTypes(builder);
  mlir::Type i64 = builder.getI64Type();
  mlir::func::FuncOp collectedCount = getOrCreatePrivateFunction(
      module, builder, "LyEH_StarCollectedCount",
      builder.getFunctionType({}, {i64}));
  mlir::func::FuncOp hasResidual = getOrCreatePrivateFunction(
      module, builder, "LyEH_StarHasResidual",
      builder.getFunctionType({}, {builder.getI1Type()}));
  mlir::func::FuncOp nodesPtr = getOrCreatePrivateFunction(
      module, builder, "LyEH_StarNodesPtr", builder.getFunctionType({}, {i64}));
  mlir::func::FuncOp residualParts = getOrCreatePrivateFunction(
      module, builder, "LyEH_StarResidualParts",
      builder.getFunctionType({}, triple));
  llvm::SmallVector<mlir::Type, 4> throwInputs(triple.begin(), triple.end());
  mlir::func::FuncOp throwCombined = getOrCreatePrivateFunction(
      module, builder, "LyEH_StarThrowCombined",
      builder.getFunctionType(throwInputs, {}));

  mlir::Value count =
      mlir::func::CallOp::create(builder, loc, collectedCount,
                                 mlir::ValueRange{})
          .getResult(0);
  mlir::Value residual =
      mlir::func::CallOp::create(builder, loc, hasResidual, mlir::ValueRange{})
          .getResult(0);
  mlir::Value zero =
      mlir::arith::ConstantIntOp::create(builder, loc, 0, 64).getResult();
  mlir::Value one =
      mlir::arith::ConstantIntOp::create(builder, loc, 1, 64).getResult();
  mlir::Value trueBit =
      mlir::arith::ConstantIntOp::create(builder, loc, 1, 1).getResult();
  mlir::Value noCollected = mlir::arith::CmpIOp::create(
      builder, loc, mlir::arith::CmpIPredicate::eq, count, zero);
  mlir::Value noResidual =
      mlir::arith::XOrIOp::create(builder, loc, residual, trueBit).getResult();
  mlir::Value clean =
      mlir::arith::AndIOp::create(builder, loc, noCollected, noResidual)
          .getResult();
  auto cleanIf = mlir::scf::IfOp::create(builder, loc, mlir::TypeRange{},
                                         clean, /*withElseRegion=*/true);
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&cleanIf.getThenRegion().front());
    mlir::func::CallOp::create(
        builder, loc, getOrCreateStarVoidFn(module, builder, "LyEH_StarPop"),
        mlir::ValueRange{});

    builder.setInsertionPointToStart(&cleanIf.getElseRegion().front());
    auto residualOnlyIf = mlir::scf::IfOp::create(
        builder, loc, mlir::TypeRange{}, noCollected,
        /*withElseRegion=*/true);
    builder.setInsertionPointToStart(&residualOnlyIf.getThenRegion().front());
    emitTryCallSiteMarkerIfNeeded(loc);
    mlir::func::CallOp::create(
        builder, loc,
        getOrCreateStarVoidFn(module, builder, "LyEH_StarRethrowResidual"),
        mlir::ValueRange{});

    builder.setInsertionPointToStart(&residualOnlyIf.getElseRegion().front());
    mlir::Value soleCount = mlir::arith::CmpIOp::create(
        builder, loc, mlir::arith::CmpIPredicate::eq, count, one);
    mlir::Value sole =
        mlir::arith::AndIOp::create(builder, loc, soleCount, noResidual)
            .getResult();
    auto soleIf = mlir::scf::IfOp::create(builder, loc, mlir::TypeRange{},
                                          sole, /*withElseRegion=*/true);
    builder.setInsertionPointToStart(&soleIf.getThenRegion().front());
    emitTryCallSiteMarkerIfNeeded(loc);
    mlir::func::CallOp::create(
        builder, loc,
        getOrCreateStarVoidFn(module, builder,
                              "LyEH_StarRethrowSoleCollected"),
        mlir::ValueRange{});

    builder.setInsertionPointToStart(&soleIf.getElseRegion().front());
    mlir::Value nodes =
        mlir::func::CallOp::create(builder, loc, nodesPtr, mlir::ValueRange{})
            .getResult(0);
    mlir::func::CallOp parts = mlir::func::CallOp::create(
        builder, loc, residualParts, mlir::ValueRange{});
    mlir::func::CallOp combined = RuntimeBundleLowerer::createRuntimeCall(
        loc, *combine,
        {nodes, count, residual, parts.getResult(0), parts.getResult(1),
         parts.getResult(2)});
    if (combined.getNumResults() != 3)
      return op.emitError() << "star_combine must return an exception triple";
    emitTryCallSiteMarkerIfNeeded(loc);
    mlir::func::CallOp::create(builder, loc, throwCombined,
                               combined.getResults());
  }
  op.erase();
  return mlir::success();
}

mlir::LogicalResult
RuntimeBundleLowerer::lowerExceptCurrentValue(py::ExceptCurrentValueOp op) {
  mlir::FailureOr<std::int64_t> handlerId =
      handlerClassId(op.getOperation(), op.getHandler());
  if (mlir::failed(handlerId))
    return mlir::failure();

  std::optional<RuntimeSymbol> borrow =
      manifest.primitive("builtins.BaseException", "borrow_current");
  if (!borrow)
    return op.emitError()
           << "runtime manifest has no builtins.BaseException.borrow_current "
              "primitive";

  builder.setInsertionPoint(op);
  mlir::func::CallOp call =
      RuntimeBundleLowerer::createRuntimeCall(op.getLoc(), *borrow, {});
  RuntimeBundle result;
  if (mlir::failed(RuntimeBundleLowerer::makeObjectBundleWithOwnership(
          op.getOperation(), op.getResult().getType(), call.getResults(),
          result, ownership::OwnershipKind::Borrow)))
    return mlir::failure();
  result.objectEvidence.setFlag(kCurrentExceptionBorrowFlag);
  valueBundles[op.getResult()] = std::move(result);
  erase.push_back(op);
  return mlir::success();
}

} // namespace py::lowering
