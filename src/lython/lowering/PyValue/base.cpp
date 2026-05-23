#include "Common/RuntimeSupport.h"
#include "Passes/OwnershipAnalysis.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"

#include <cerrno>
#include <cstdint>
#include <cstdlib>

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

namespace py {
namespace {

static void getLocInfo(mlir::Location loc, mlir::MLIRContext *ctx,
                       mlir::StringAttr &fileAttr, std::int64_t &line,
                       std::int64_t &col) {
  if (auto fileLoc = llvm::dyn_cast<mlir::FileLineColLoc>(loc)) {
    fileAttr = fileLoc.getFilename();
    line = static_cast<std::int64_t>(fileLoc.getLine());
    col = static_cast<std::int64_t>(fileLoc.getColumn());
    return;
  }
  if (auto nameLoc = llvm::dyn_cast<mlir::NameLoc>(loc)) {
    getLocInfo(nameLoc.getChildLoc(), ctx, fileAttr, line, col);
    return;
  }
  if (auto fused = llvm::dyn_cast<mlir::FusedLoc>(loc)) {
    for (auto subloc : fused.getLocations()) {
      if (auto subfile = llvm::dyn_cast<mlir::FileLineColLoc>(subloc)) {
        fileAttr = subfile.getFilename();
        line = static_cast<std::int64_t>(subfile.getLine());
        col = static_cast<std::int64_t>(subfile.getColumn());
        return;
      }
    }
  }
  fileAttr = mlir::StringAttr::get(ctx, "<unknown>");
  line = 0;
  col = 0;
}

static mlir::StringAttr getFuncNameAttr(mlir::func::FuncOp func,
                                        mlir::MLIRContext *ctx) {
  if (!func)
    return mlir::StringAttr::get(ctx, "<unknown>");
  llvm::StringRef name = func.getName();
  if (name == "main")
    return mlir::StringAttr::get(ctx, "<module>");
  return mlir::StringAttr::get(ctx, name);
}

static bool isExceptionCell(mlir::Value value) {
  return value && mlir::isa<mlir::LLVM::LLVMPointerType>(value.getType());
}

static mlir::Value getCurrentAsyncExceptionCell(mlir::Operation *op) {
  auto asyncFunc = op->getParentOfType<mlir::async::FuncOp>();
  if (!asyncFunc || asyncFunc.getBody().empty())
    return {};
  mlir::Block &entry = asyncFunc.getBody().front();
  if (entry.getNumArguments() == 0)
    return {};
  mlir::Value candidate = entry.getArgument(entry.getNumArguments() - 1);
  return isExceptionCell(candidate) ? candidate : mlir::Value();
}

static void storeExceptionCell(mlir::Location loc, mlir::Value cell,
                               mlir::Value exception, mlir::ModuleOp module,
                               mlir::PatternRewriter &rewriter,
                               const PyLLVMTypeConverter &typeConverter) {
  if (!isExceptionCell(cell))
    return;
  async_runtime::ExceptionCell::storeFirst(loc, cell, exception, module,
                                           rewriter, typeConverter);
}

static mlir::Value loadExceptionCell(mlir::Location loc, mlir::Value cell,
                                     mlir::OpBuilder &builder) {
  return async_runtime::ExceptionCell::load(loc, cell, builder);
}

static void copyExceptionCell(mlir::Location loc, mlir::Value sourceCell,
                              mlir::Value destCell, mlir::ModuleOp module,
                              mlir::PatternRewriter &rewriter,
                              const PyLLVMTypeConverter &typeConverter) {
  if (!isExceptionCell(sourceCell) || !isExceptionCell(destCell) ||
      sourceCell == destCell)
    return;
  mlir::Value exception = loadExceptionCell(loc, sourceCell, rewriter);
  storeExceptionCell(loc, destCell, exception, module, rewriter, typeConverter);
}

static void consumeAwaitedDescriptor(mlir::Location loc, mlir::Value awaitable,
                                     mlir::PatternRewriter &rewriter) {
  auto dec = rewriter.create<DecRefOp>(loc, awaitable);
  dec->setAttr("ly.async.await_consumed_descriptor", rewriter.getUnitAttr());
}

static void consumeAwaitedDescriptorWithLoadedException(
    mlir::Location loc, mlir::Value awaitable, mlir::ValueRange descriptor,
    mlir::Type awaitableType, mlir::Value loadedException,
    mlir::ModuleOp module, mlir::PatternRewriter &rewriter,
    const PyLLVMTypeConverter &typeConverter) {
  if (descriptor.size() >= 2 && isExceptionCell(descriptor[1])) {
    mlir::Value exception =
        loadedException ? loadedException
                        : loadExceptionCell(loc, descriptor[1], rewriter);
    async_runtime::ExceptionCell::releaseLoaded(loc, module, rewriter,
                                                typeConverter, exception);
    async_runtime::ExceptionCell::free(loc, module, rewriter, typeConverter,
                                       descriptor[1]);
  }

  if (mlir::isa<TaskType>(awaitableType) && descriptor.size() == 3 &&
      mlir::isa<mlir::MemRefType>(descriptor[2].getType()))
    rewriter.create<mlir::memref::DeallocOp>(loc, descriptor[2]);

  if (!descriptor.empty() &&
      mlir::isa<mlir::async::ValueType>(descriptor.front().getType()))
    rewriter.create<mlir::async::RuntimeDropRefOp>(
        loc, descriptor.front(), rewriter.getI64IntegerAttr(1));

  auto witness = rewriter.create<DecRefOp>(loc, awaitable);
  witness->setAttr("ly.ownership.lowered_witness", rewriter.getUnitAttr());
}

static void drainGatherCleanupDescriptor(mlir::Location loc,
                                         mlir::Value awaitable,
                                         mlir::PatternRewriter &rewriter) {
  auto dec = rewriter.create<DecRefOp>(loc, awaitable);
  dec->setAttr("ly.async.gather_drain_descriptor", rewriter.getUnitAttr());
}

static llvm::SmallVector<mlir::Value> unpackAsyncPayload(
    mlir::Location loc, mlir::Type logicalType, mlir::Value storage,
    const PyLLVMTypeConverter &typeConverter, mlir::PatternRewriter &rewriter) {
  llvm::SmallVector<mlir::Type> resultTypes;
  if (mlir::failed(typeConverter.convertType(logicalType, resultTypes)) ||
      resultTypes.empty())
    return {};
  if (resultTypes.size() == 1 && resultTypes.front() == storage.getType())
    return {storage};
  auto cast = rewriter.create<mlir::UnrealizedConversionCastOp>(
      loc, resultTypes, storage);
  llvm::SmallVector<mlir::Value> results;
  results.append(cast.getResults().begin(), cast.getResults().end());
  return results;
}

static mlir::FailureOr<mlir::Value>
materializeLogicalValue(mlir::Location loc, mlir::Type logicalType,
                        mlir::ValueRange parts,
                        mlir::PatternRewriter &rewriter) {
  if (parts.empty())
    return mlir::failure();
  if (parts.size() == 1 && parts.front().getType() == logicalType)
    return parts.front();
  auto cast = rewriter.create<mlir::UnrealizedConversionCastOp>(
      loc, mlir::TypeRange{logicalType}, parts);
  if (isPyOwnershipTrackedType(logicalType)) {
    llvm::SmallVector<mlir::Attribute, 1> owned{rewriter.getI64IntegerAttr(0)};
    cast->setAttr(OwnershipContractAttrs::kOwnedResults,
                  rewriter.getArrayAttr(owned));
  }
  return cast.getResult(0);
}

static void cloneFinallyBlockBody(mlir::Block *finallyBlock,
                                  mlir::PatternRewriter &rewriter,
                                  mlir::Attribute markerId = {}) {
  if (!finallyBlock)
    return;
  mlir::IRMapping mapping;
  for (mlir::Operation &nested : finallyBlock->without_terminator()) {
    mlir::Operation *clone = rewriter.clone(nested, mapping);
    if (markerId)
      clone->setAttr("ly.finally.error_block_id", markerId);
  }
}

static void cloneFinallyBody(TryOp tryOp, mlir::PatternRewriter &rewriter,
                             mlir::Attribute markerId = {}) {
  if (!tryOp || tryOp.getFinallyRegion().empty())
    return;
  cloneFinallyBlockBody(&tryOp.getFinallyRegion().front(), rewriter, markerId);
}

static bool isFinallyEscapeTerminator(mlir::Operation *terminator) {
  if (terminator && terminator->hasAttr("ly.redirected_to_except"))
    return false;
  return mlir::isa<RaiseOp, ReturnOp, mlir::async::ReturnOp,
                   mlir::func::ReturnOp>(terminator);
}

static void eraseOpsAfter(mlir::Operation *op,
                          mlir::PatternRewriter &rewriter) {
  mlir::Block *block = op ? op->getBlock() : nullptr;
  if (!block)
    return;
  for (auto it = std::next(op->getIterator()); it != block->end();) {
    mlir::Operation *dead = &*it++;
    rewriter.eraseOp(dead);
  }
}

static void eraseOpsAfterTerminators(llvm::ArrayRef<mlir::Block *> blocks,
                                     mlir::PatternRewriter &rewriter) {
  for (mlir::Block *block : blocks) {
    for (mlir::Operation &op : *block) {
      if (!op.hasTrait<mlir::OpTrait::IsTerminator>())
        continue;
      eraseOpsAfter(&op, rewriter);
      break;
    }
  }
}

static bool hasFinallyFallthrough(TryOp op) {
  for (mlir::Block &block : op.getTryRegion())
    if (mlir::isa_and_nonnull<TryYieldOp>(block.getTerminator()))
      return true;
  for (mlir::Block &block : op.getExceptRegion())
    if (mlir::isa_and_nonnull<ExceptYieldOp>(block.getTerminator()))
      return true;
  return false;
}

static void
cloneFinallyBeforeEscapes(mlir::Block *finallyBlock,
                          llvm::ArrayRef<mlir::Block *> protectedBlocks,
                          mlir::PatternRewriter &rewriter) {
  if (!finallyBlock)
    return;

  llvm::SmallVector<mlir::Operation *> escapes;
  for (mlir::Block *block : protectedBlocks) {
    for (mlir::Operation &candidate : *block) {
      if (!isFinallyEscapeTerminator(&candidate))
        continue;
      escapes.push_back(&candidate);
      break;
    }
  }

  for (mlir::Operation *terminator : escapes) {
    rewriter.setInsertionPoint(terminator);
    cloneFinallyBlockBody(finallyBlock, rewriter);
  }
}

static mlir::FailureOr<mlir::Value> lowerAwaitInFinallyTry(
    AwaitOp awaitOp, TryOp tryOp, const PyLLVMTypeConverter &typeConverter,
    mlir::PatternRewriter &rewriter, mlir::ValueRange cleanupPayloads = {},
    mlir::ValueRange cleanupAwaitables = {}) {
  mlir::Value awaitable = awaitOp.getAwaitable();
  llvm::SmallVector<mlir::Type> awaitableTypes;
  if (mlir::failed(
          typeConverter.convertType(awaitable.getType(), awaitableTypes)) ||
      awaitableTypes.empty())
    return rewriter.notifyMatchFailure(awaitOp,
                                       "failed to convert awaitable type");

  rewriter.setInsertionPoint(awaitOp);
  llvm::SmallVector<mlir::Value> convertedAwaitableParts;
  if (awaitableTypes.size() == 1 &&
      awaitable.getType() == awaitableTypes.front()) {
    convertedAwaitableParts.push_back(awaitable);
  } else {
    auto cast = rewriter.create<mlir::UnrealizedConversionCastOp>(
        awaitOp.getLoc(), mlir::TypeRange(awaitableTypes), awaitable);
    convertedAwaitableParts.append(cast.getResults().begin(),
                                   cast.getResults().end());
  }
  mlir::Value convertedAwaitable = convertedAwaitableParts.front();
  auto asyncValueType =
      mlir::dyn_cast<mlir::async::ValueType>(convertedAwaitable.getType());
  if (!asyncValueType)
    return rewriter.notifyMatchFailure(
        awaitOp, "awaitable did not lower to async.value");

  mlir::Value currentExceptionCell = getCurrentAsyncExceptionCell(awaitOp);
  mlir::Value awaitableExceptionCell = convertedAwaitableParts.size() >= 2
                                           ? convertedAwaitableParts[1]
                                           : mlir::Value();

  auto markerId = rewriter.getI64IntegerAttr(static_cast<int64_t>(
      reinterpret_cast<std::uintptr_t>(awaitOp.getOperation())));

  auto isError = rewriter.create<mlir::async::RuntimeIsErrorOp>(
      awaitOp.getLoc(), rewriter.getI1Type(), convertedAwaitable);
  mlir::Value shouldRunFinally = isError.getIsError();
  if (mlir::isa<TaskType>(awaitable.getType()) &&
      convertedAwaitableParts.size() == 3 &&
      mlir::isa<mlir::MemRefType>(convertedAwaitableParts[2].getType())) {
    mlir::Value zeroIndex =
        rewriter.create<mlir::arith::ConstantIndexOp>(awaitOp.getLoc(), 0);
    mlir::Value flag = rewriter.create<mlir::memref::LoadOp>(
        awaitOp.getLoc(), convertedAwaitableParts[2], zeroIndex);
    mlir::Value zero =
        rewriter.create<mlir::arith::ConstantIntOp>(awaitOp.getLoc(), 0, 8);
    mlir::Value isCancelled = rewriter.create<mlir::arith::CmpIOp>(
        awaitOp.getLoc(), mlir::arith::CmpIPredicate::ne, flag, zero);
    shouldRunFinally = rewriter.create<mlir::arith::OrIOp>(
        awaitOp.getLoc(), shouldRunFinally, isCancelled);
  }

  mlir::Block *currentBlock = rewriter.getInsertionBlock();
  mlir::Block *awaitBlock =
      rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
  mlir::Block *errorBlock =
      rewriter.createBlock(awaitBlock->getParent(), awaitBlock->getIterator());
  rewriter.setInsertionPointToEnd(currentBlock);
  rewriter.create<mlir::cf::CondBranchOp>(awaitOp.getLoc(), shouldRunFinally,
                                          errorBlock, awaitBlock);

  rewriter.setInsertionPointToStart(errorBlock);
  copyExceptionCell(
      awaitOp.getLoc(), awaitableExceptionCell, currentExceptionCell,
      awaitOp->getParentOfType<mlir::ModuleOp>(), rewriter, typeConverter);
  consumeAwaitedDescriptor(awaitOp.getLoc(), awaitable, rewriter);
  for (mlir::Value payload : cleanupPayloads)
    if (isPyOwnershipTrackedType(payload.getType()))
      rewriter.create<DecRefOp>(awaitOp.getLoc(), payload);
  for (mlir::Value awaitableToCleanup : cleanupAwaitables)
    if (isPyOwnershipTrackedType(awaitableToCleanup.getType()))
      drainGatherCleanupDescriptor(awaitOp.getLoc(), awaitableToCleanup,
                                   rewriter);
  cloneFinallyBody(tryOp, rewriter, markerId);
  mlir::Value falseValue = rewriter.create<mlir::arith::ConstantOp>(
      awaitOp.getLoc(), rewriter.getBoolAttr(false));
  rewriter.create<mlir::cf::AssertOp>(awaitOp.getLoc(), falseValue,
                                      "task cancelled");
  rewriter.create<mlir::cf::BranchOp>(awaitOp.getLoc(), errorBlock);

  rewriter.setInsertionPoint(awaitOp);
  auto asyncAwait = rewriter.create<mlir::async::AwaitOp>(awaitOp.getLoc(),
                                                          convertedAwaitable);
  asyncAwait->setAttr("ly.finally.error_id", markerId);
  rewriter.setInsertionPointAfter(asyncAwait);
  consumeAwaitedDescriptor(awaitOp.getLoc(), awaitable, rewriter);
  llvm::SmallVector<mlir::Value> results =
      unpackAsyncPayload(awaitOp.getLoc(), awaitOp.getResult().getType(),
                         asyncAwait.getResult(), typeConverter, rewriter);
  if (results.empty())
    return rewriter.notifyMatchFailure(awaitOp,
                                       "failed to unpack await result");
  mlir::FailureOr<mlir::Value> logicalResult = materializeLogicalValue(
      awaitOp.getLoc(), awaitOp.getResult().getType(), results, rewriter);
  if (mlir::failed(logicalResult))
    return rewriter.notifyMatchFailure(awaitOp,
                                       "failed to rematerialize await result");
  rewriter.replaceOp(awaitOp, mlir::ValueRange{*logicalResult});
  return *logicalResult;
}

static mlir::FailureOr<mlir::Value>
lowerAwaitInExceptTry(AwaitOp awaitOp, mlir::Block *exceptEntry,
                      const PyLLVMTypeConverter &typeConverter,
                      mlir::PatternRewriter &rewriter,
                      mlir::ValueRange cleanupPayloads = {},
                      mlir::ValueRange cleanupAwaitables = {}) {
  if (!exceptEntry)
    return mlir::failure();

  mlir::Value awaitable = awaitOp.getAwaitable();
  llvm::SmallVector<mlir::Type> awaitableTypes;
  if (mlir::failed(
          typeConverter.convertType(awaitable.getType(), awaitableTypes)) ||
      awaitableTypes.empty())
    return rewriter.notifyMatchFailure(awaitOp,
                                       "failed to convert awaitable type");

  rewriter.setInsertionPoint(awaitOp);
  llvm::SmallVector<mlir::Value> convertedAwaitableParts;
  if (awaitableTypes.size() == 1 &&
      awaitable.getType() == awaitableTypes.front()) {
    convertedAwaitableParts.push_back(awaitable);
  } else {
    auto cast = rewriter.create<mlir::UnrealizedConversionCastOp>(
        awaitOp.getLoc(), mlir::TypeRange(awaitableTypes), awaitable);
    convertedAwaitableParts.append(cast.getResults().begin(),
                                   cast.getResults().end());
  }
  mlir::Value convertedAwaitable = convertedAwaitableParts.front();
  auto asyncValueType =
      mlir::dyn_cast<mlir::async::ValueType>(convertedAwaitable.getType());
  if (!asyncValueType)
    return rewriter.notifyMatchFailure(
        awaitOp, "awaitable did not lower to async.value");

  mlir::Value awaitableExceptionCell = convertedAwaitableParts.size() >= 2
                                           ? convertedAwaitableParts[1]
                                           : mlir::Value();
  auto markerId = rewriter.getI64IntegerAttr(static_cast<int64_t>(
      reinterpret_cast<std::uintptr_t>(awaitOp.getOperation())));

  auto isError = rewriter.create<mlir::async::RuntimeIsErrorOp>(
      awaitOp.getLoc(), rewriter.getI1Type(), convertedAwaitable);
  mlir::Value shouldCatch = isError.getIsError();
  if (mlir::isa<TaskType>(awaitable.getType()) &&
      convertedAwaitableParts.size() == 3 &&
      mlir::isa<mlir::MemRefType>(convertedAwaitableParts[2].getType())) {
    mlir::Value zeroIndex =
        rewriter.create<mlir::arith::ConstantIndexOp>(awaitOp.getLoc(), 0);
    mlir::Value flag = rewriter.create<mlir::memref::LoadOp>(
        awaitOp.getLoc(), convertedAwaitableParts[2], zeroIndex);
    mlir::Value zero =
        rewriter.create<mlir::arith::ConstantIntOp>(awaitOp.getLoc(), 0, 8);
    mlir::Value isCancelled = rewriter.create<mlir::arith::CmpIOp>(
        awaitOp.getLoc(), mlir::arith::CmpIPredicate::ne, flag, zero);
    shouldCatch = rewriter.create<mlir::arith::OrIOp>(awaitOp.getLoc(),
                                                      shouldCatch, isCancelled);
  }

  mlir::Block *currentBlock = rewriter.getInsertionBlock();
  mlir::Block *awaitBlock =
      rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
  mlir::Block *errorBlock =
      rewriter.createBlock(awaitBlock->getParent(), awaitBlock->getIterator());
  rewriter.setInsertionPointToEnd(currentBlock);
  rewriter.create<mlir::cf::CondBranchOp>(awaitOp.getLoc(), shouldCatch,
                                          errorBlock, awaitBlock);

  rewriter.setInsertionPointToStart(errorBlock);
  mlir::Value loadedException;
  mlir::Value exception;
  if (isExceptionCell(awaitableExceptionCell)) {
    loadedException =
        loadExceptionCell(awaitOp.getLoc(), awaitableExceptionCell, rewriter);
    exception = loadedException;
  } else {
    exception = rewriter.create<ExceptionNullOp>(
        awaitOp.getLoc(), ExceptionType::get(awaitOp.getContext()));
  }
  exception.getDefiningOp()->setAttr("ly.finally.error_block_id", markerId);
  if (!mlir::isa<ExceptionType>(exception.getType())) {
    exception =
        rewriter
            .create<mlir::UnrealizedConversionCastOp>(
                awaitOp.getLoc(),
                mlir::TypeRange{ExceptionType::get(awaitOp.getContext())},
                exception)
            .getResult(0);
  }
  mlir::Type exceptArgType = exceptEntry->getNumArguments() == 0
                                 ? mlir::Type()
                                 : exceptEntry->getArgument(0).getType();
  if (exceptArgType && exception.getType() != exceptArgType) {
    exception =
        rewriter
            .create<mlir::UnrealizedConversionCastOp>(
                awaitOp.getLoc(), mlir::TypeRange{exceptArgType}, exception)
            .getResult(0);
  }
  auto retain = rewriter.create<IncRefOp>(awaitOp.getLoc(), exception);
  if (loadedException)
    threadsafe::Retain::premise(retain.getOperation(),
                                ThreadSafetyAttrs::kPremiseAggregateBorrow);
  consumeAwaitedDescriptorWithLoadedException(
      awaitOp.getLoc(), awaitable, convertedAwaitableParts, awaitable.getType(),
      loadedException, awaitOp->getParentOfType<mlir::ModuleOp>(), rewriter,
      typeConverter);
  for (mlir::Value payload : cleanupPayloads)
    if (isPyOwnershipTrackedType(payload.getType()))
      rewriter.create<DecRefOp>(awaitOp.getLoc(), payload);
  for (mlir::Value awaitableToCleanup : cleanupAwaitables)
    if (isPyOwnershipTrackedType(awaitableToCleanup.getType()))
      drainGatherCleanupDescriptor(awaitOp.getLoc(), awaitableToCleanup,
                                   rewriter);
  rewriter.create<mlir::cf::BranchOp>(awaitOp.getLoc(), exceptEntry,
                                      mlir::ValueRange{exception});

  rewriter.setInsertionPoint(awaitOp);
  auto asyncAwait = rewriter.create<mlir::async::AwaitOp>(awaitOp.getLoc(),
                                                          convertedAwaitable);
  asyncAwait->setAttr("ly.finally.error_id", markerId);
  rewriter.setInsertionPointAfter(asyncAwait);
  consumeAwaitedDescriptor(awaitOp.getLoc(), awaitable, rewriter);
  llvm::SmallVector<mlir::Value> results =
      unpackAsyncPayload(awaitOp.getLoc(), awaitOp.getResult().getType(),
                         asyncAwait.getResult(), typeConverter, rewriter);
  if (results.empty())
    return rewriter.notifyMatchFailure(awaitOp,
                                       "failed to unpack await result");
  mlir::FailureOr<mlir::Value> logicalResult = materializeLogicalValue(
      awaitOp.getLoc(), awaitOp.getResult().getType(), results, rewriter);
  if (mlir::failed(logicalResult))
    return rewriter.notifyMatchFailure(awaitOp,
                                       "failed to rematerialize await result");
  rewriter.replaceOp(awaitOp, mlir::ValueRange{*logicalResult});
  return *logicalResult;
}

static mlir::FailureOr<mlir::Value>
convertExceptEntryArgument(mlir::Block *exceptEntry,
                           const PyLLVMTypeConverter &typeConverter,
                           mlir::PatternRewriter &rewriter) {
  if (!exceptEntry || exceptEntry->getNumArguments() != 1)
    return mlir::failure();

  mlir::BlockArgument oldArg = exceptEntry->getArgument(0);
  mlir::Type convertedType = typeConverter.convertType(oldArg.getType());
  if (!convertedType)
    return mlir::failure();
  if (convertedType == oldArg.getType())
    return oldArg;

  mlir::BlockArgument convertedArg =
      exceptEntry->addArgument(convertedType, oldArg.getLoc());
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(exceptEntry);
  mlir::Value logicalArg =
      rewriter
          .create<mlir::UnrealizedConversionCastOp>(
              oldArg.getLoc(), mlir::TypeRange{oldArg.getType()}, convertedArg)
          .getResult(0);
  if (isPyOwnershipTrackedType(oldArg.getType())) {
    llvm::SmallVector<mlir::Attribute, 1> owned{rewriter.getI64IntegerAttr(0)};
    logicalArg.getDefiningOp()->setAttr(OwnershipContractAttrs::kOwnedResults,
                                        rewriter.getArrayAttr(owned));
  }
  oldArg.replaceAllUsesWith(logicalArg);
  exceptEntry->eraseArgument(0);
  return logicalArg;
}

static mlir::LogicalResult convertFinallyEntryArguments(
    TryOp op, mlir::Block *finallyEntry,
    llvm::ArrayRef<llvm::SmallVector<mlir::Type>> convertedResultTypes,
    llvm::ArrayRef<mlir::Type> flatResultTypes,
    mlir::PatternRewriter &rewriter) {
  if (!finallyEntry)
    return mlir::failure();

  if (finallyEntry->getNumArguments() == 0) {
    llvm::SmallVector<mlir::Location> locs(flatResultTypes.size(), op.getLoc());
    finallyEntry->addArguments(flatResultTypes, locs);
    return mlir::success();
  }

  if (finallyEntry->getNumArguments() == flatResultTypes.size()) {
    bool alreadyConverted = true;
    for (auto [arg, type] :
         llvm::zip(finallyEntry->getArguments(), flatResultTypes)) {
      if (arg.getType() != type) {
        alreadyConverted = false;
        break;
      }
    }
    if (alreadyConverted)
      return mlir::success();
  }

  if (finallyEntry->getNumArguments() != op.getNumResults())
    return rewriter.notifyMatchFailure(
        op, "finally entry argument count must match py.try results");

  llvm::SmallVector<mlir::BlockArgument> oldArgs;
  llvm::append_range(oldArgs, finallyEntry->getArguments());
  for (auto [arg, type] : llvm::zip(oldArgs, op.getResultTypes()))
    if (arg.getType() != type)
      return rewriter.notifyMatchFailure(
          op, "finally entry argument type must match py.try result");

  llvm::SmallVector<mlir::Location> locs(flatResultTypes.size(), op.getLoc());
  finallyEntry->addArguments(flatResultTypes, locs);

  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(finallyEntry);
  unsigned flatIndex = oldArgs.size();
  for (auto [oldArg, convertedTypes] :
       llvm::zip(oldArgs, convertedResultTypes)) {
    llvm::SmallVector<mlir::Value> convertedArgs;
    for (unsigned offset = 0; offset < convertedTypes.size(); ++offset)
      convertedArgs.push_back(finallyEntry->getArgument(flatIndex + offset));
    mlir::Value logicalArg;
    if (convertedArgs.size() == 1 &&
        convertedArgs.front().getType() == oldArg.getType()) {
      logicalArg = convertedArgs.front();
    } else {
      logicalArg = rewriter
                       .create<mlir::UnrealizedConversionCastOp>(
                           oldArg.getLoc(), mlir::TypeRange{oldArg.getType()},
                           mlir::ValueRange(convertedArgs))
                       .getResult(0);
    }
    oldArg.replaceAllUsesWith(logicalArg);
    flatIndex += convertedTypes.size();
  }

  for (int64_t index = static_cast<int64_t>(oldArgs.size()) - 1; index >= 0;
       --index)
    finallyEntry->eraseArgument(static_cast<unsigned>(index));
  return mlir::success();
}

static void
replaceRaiseCurrentInExceptBlocks(llvm::ArrayRef<mlir::Block *> exceptBlocks,
                                  mlir::Value caughtException,
                                  mlir::PatternRewriter &rewriter) {
  llvm::SmallVector<RaiseCurrentOp> raises;
  mlir::Region *exceptRegion = caughtException.getParentRegion();
  for (mlir::Block *block : exceptBlocks) {
    block->walk([&](RaiseCurrentOp raiseCurrent) {
      if (raiseCurrent->getBlock()->getParent() == exceptRegion)
        raises.push_back(raiseCurrent);
    });
  }

  mlir::Type exceptionType = ExceptionType::get(rewriter.getContext());
  for (RaiseCurrentOp raiseCurrent : raises) {
    rewriter.setInsertionPoint(raiseCurrent);
    mlir::Value exception = caughtException;
    if (exception.getType() != exceptionType) {
      exception = rewriter
                      .create<mlir::UnrealizedConversionCastOp>(
                          raiseCurrent.getLoc(), mlir::TypeRange{exceptionType},
                          exception)
                      .getResult(0);
    }
    rewriter.replaceOpWithNewOp<RaiseOp>(raiseCurrent, exception);
  }
}

static llvm::SmallVector<mlir::Block *> collectBlockSpan(mlir::Block *begin,
                                                         mlir::Block *end) {
  llvm::SmallVector<mlir::Block *> blocks;
  if (!begin || !end || begin->getParent() != end->getParent())
    return blocks;
  mlir::Region *region = begin->getParent();
  for (auto it = begin->getIterator(), endIt = end->getIterator();
       it != region->end() && it != endIt; ++it)
    blocks.push_back(&*it);
  return blocks;
}

static void
collectAwaitsExcludingNestedTry(llvm::ArrayRef<mlir::Block *> blocks,
                                llvm::SmallVectorImpl<AwaitOp> &awaits) {
  for (mlir::Block *block : blocks) {
    for (mlir::Operation &root : *block) {
      if (mlir::isa<TryOp>(root))
        continue;
      root.walk<mlir::WalkOrder::PreOrder>(
          [&](mlir::Operation *nested) -> mlir::WalkResult {
            if (nested != &root && mlir::isa<TryOp>(nested))
              return mlir::WalkResult::skip();
            if (auto awaitOp = mlir::dyn_cast<AwaitOp>(nested))
              awaits.push_back(awaitOp);
            return mlir::WalkResult::advance();
          });
    }
  }
}

static void collectGathersExcludingNestedTry(
    llvm::ArrayRef<mlir::Block *> blocks,
    llvm::SmallVectorImpl<AsyncGatherOp> &gathers) {
  for (mlir::Block *block : blocks) {
    for (mlir::Operation &root : *block) {
      if (mlir::isa<TryOp>(root))
        continue;
      root.walk<mlir::WalkOrder::PreOrder>(
          [&](mlir::Operation *nested) -> mlir::WalkResult {
            if (nested != &root && mlir::isa<TryOp>(nested))
              return mlir::WalkResult::skip();
            if (auto gatherOp = mlir::dyn_cast<AsyncGatherOp>(nested))
              gathers.push_back(gatherOp);
            return mlir::WalkResult::advance();
          });
    }
  }
}

static bool collectOnlyDecrefUsers(mlir::Value value,
                                   llvm::SmallVectorImpl<DecRefOp> &decrefs) {
  for (mlir::Operation *user : value.getUsers()) {
    auto decref = mlir::dyn_cast<DecRefOp>(user);
    if (!decref)
      return false;
    decrefs.push_back(decref);
  }
  return !decrefs.empty();
}

static mlir::FailureOr<llvm::SmallVector<mlir::Value>>
lowerGatherOperandsInFinallyTry(AsyncGatherOp gatherOp, TryOp tryOp,
                                const PyLLVMTypeConverter &typeConverter,
                                mlir::PatternRewriter &rewriter,
                                llvm::ArrayRef<mlir::Type> elementTypes) {
  llvm::SmallVector<mlir::Value> payloads;
  payloads.reserve(elementTypes.size());
  llvm::SmallVector<mlir::Value> awaitables(gatherOp.getAwaitables());
  for (auto [index, awaitable] : llvm::enumerate(awaitables)) {
    mlir::Type payloadType = elementTypes[index];
    llvm::ArrayRef<mlir::Value> remainingAwaitables =
        llvm::ArrayRef<mlir::Value>(awaitables).drop_front(index + 1);
    rewriter.setInsertionPoint(gatherOp);
    auto awaitOp =
        rewriter.create<AwaitOp>(gatherOp.getLoc(), payloadType, awaitable);
    mlir::FailureOr<mlir::Value> payload =
        lowerAwaitInFinallyTry(awaitOp, tryOp, typeConverter, rewriter,
                               mlir::ValueRange(payloads), remainingAwaitables);
    if (mlir::failed(payload))
      return mlir::failure();
    payloads.push_back(*payload);
  }
  return payloads;
}

static mlir::FailureOr<llvm::SmallVector<mlir::Value>>
lowerGatherOperandsInExceptTry(AsyncGatherOp gatherOp, mlir::Block *exceptEntry,
                               const PyLLVMTypeConverter &typeConverter,
                               mlir::PatternRewriter &rewriter,
                               llvm::ArrayRef<mlir::Type> elementTypes,
                               bool releaseEachPayload = false) {
  llvm::SmallVector<mlir::Value> payloads;
  llvm::SmallVector<mlir::Value> livePayloads;
  payloads.reserve(elementTypes.size());
  livePayloads.reserve(elementTypes.size());
  llvm::SmallVector<mlir::Value> awaitables(gatherOp.getAwaitables());
  for (auto [index, awaitable] : llvm::enumerate(awaitables)) {
    mlir::Type payloadType = elementTypes[index];
    llvm::ArrayRef<mlir::Value> remainingAwaitables =
        llvm::ArrayRef<mlir::Value>(awaitables).drop_front(index + 1);
    rewriter.setInsertionPoint(gatherOp);
    auto awaitOp =
        rewriter.create<AwaitOp>(gatherOp.getLoc(), payloadType, awaitable);
    mlir::FailureOr<mlir::Value> payload = lowerAwaitInExceptTry(
        awaitOp, exceptEntry, typeConverter, rewriter,
        mlir::ValueRange(livePayloads), remainingAwaitables);
    if (mlir::failed(payload))
      return mlir::failure();
    rewriter.setInsertionPoint(gatherOp);
    if (releaseEachPayload && isPyOwnershipTrackedType(payload->getType())) {
      rewriter.create<DecRefOp>(gatherOp.getLoc(), *payload);
    } else {
      livePayloads.push_back(*payload);
    }
    payloads.push_back(*payload);
  }
  return payloads;
}

static mlir::LogicalResult
lowerGatherInFinallyTry(AsyncGatherOp gatherOp, TryOp tryOp,
                        const PyLLVMTypeConverter &typeConverter,
                        mlir::PatternRewriter &rewriter) {
  auto tupleType = mlir::dyn_cast<TupleType>(gatherOp.getResult().getType());
  if (!tupleType)
    return rewriter.notifyMatchFailure(gatherOp,
                                       "gather result is not a tuple");

  auto elementTypes = tupleType.getElementTypes();
  if (elementTypes.size() != gatherOp.getAwaitables().size())
    return rewriter.notifyMatchFailure(
        gatherOp, "gather awaitable count must match tuple arity");

  mlir::FailureOr<llvm::SmallVector<mlir::Value>> payloads =
      lowerGatherOperandsInFinallyTry(gatherOp, tryOp, typeConverter, rewriter,
                                      elementTypes);
  if (mlir::failed(payloads))
    return mlir::failure();

  rewriter.setInsertionPoint(gatherOp);
  auto tuple = rewriter.create<TupleCreateOp>(
      gatherOp.getLoc(), gatherOp.getResult().getType(), *payloads);
  tuple->setAttr("ly.async.gather_tuple", rewriter.getUnitAttr());
  rewriter.replaceOp(gatherOp, tuple.getResult());
  return mlir::success();
}

static mlir::LogicalResult
lowerGatherInExceptTry(AsyncGatherOp gatherOp, mlir::Block *exceptEntry,
                       const PyLLVMTypeConverter &typeConverter,
                       mlir::PatternRewriter &rewriter) {
  auto tupleType = mlir::dyn_cast<TupleType>(gatherOp.getResult().getType());
  if (!tupleType)
    return rewriter.notifyMatchFailure(gatherOp,
                                       "gather result is not a tuple");

  auto elementTypes = tupleType.getElementTypes();
  if (elementTypes.size() != gatherOp.getAwaitables().size())
    return rewriter.notifyMatchFailure(
        gatherOp, "gather awaitable count must match tuple arity");

  llvm::SmallVector<DecRefOp> discardUsers;
  if (collectOnlyDecrefUsers(gatherOp.getResult(), discardUsers)) {
    if (mlir::failed(lowerGatherOperandsInExceptTry(
            gatherOp, exceptEntry, typeConverter, rewriter, elementTypes,
            /*releaseEachPayload=*/true)))
      return mlir::failure();
    for (DecRefOp decref : discardUsers)
      rewriter.eraseOp(decref);
    rewriter.eraseOp(gatherOp);
    return mlir::success();
  }

  mlir::FailureOr<llvm::SmallVector<mlir::Value>> payloads =
      lowerGatherOperandsInExceptTry(gatherOp, exceptEntry, typeConverter,
                                     rewriter, elementTypes);
  if (mlir::failed(payloads))
    return mlir::failure();

  rewriter.setInsertionPoint(gatherOp);
  auto tuple = rewriter.create<TupleCreateOp>(
      gatherOp.getLoc(), gatherOp.getResult().getType(), *payloads);
  tuple->setAttr("ly.async.gather_tuple", rewriter.getUnitAttr());
  rewriter.replaceOp(gatherOp, tuple.getResult());
  return mlir::success();
}

struct ExceptionNullLowering
    : public mlir::OpConversionPattern<ExceptionNullOp> {
  ExceptionNullLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<ExceptionNullOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(ExceptionNullOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    mlir::Type resultType =
        typeConverter->convertType(op.getResult().getType());
    rewriter.replaceOpWithNewOp<mlir::LLVM::ZeroOp>(op, resultType);
    return mlir::success();
  }
};

struct TracebackNullLowering
    : public mlir::OpConversionPattern<TracebackNullOp> {
  TracebackNullLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<TracebackNullOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(TracebackNullOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    mlir::Type resultType =
        typeConverter->convertType(op.getResult().getType());
    rewriter.replaceOpWithNewOp<mlir::LLVM::ZeroOp>(op, resultType);
    return mlir::success();
  }
};

struct LocationCurrentLowering
    : public mlir::OpConversionPattern<LocationCurrentOp> {
  LocationCurrentLowering(PyLLVMTypeConverter &converter,
                          mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<LocationCurrentOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(LocationCurrentOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    mlir::Type resultType =
        typeConverter->convertType(op.getResult().getType());
    rewriter.replaceOpWithNewOp<mlir::LLVM::ZeroOp>(op, resultType);
    return mlir::success();
  }
};

struct ExceptionNewLowering : public mlir::OpConversionPattern<ExceptionNewOp> {
  ExceptionNewLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<ExceptionNewOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(ExceptionNewOp op, OneToNOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return mlir::failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *typeConverter);
    mlir::Type resultType =
        typeConverter->convertType(op.getResult().getType());
    llvm::ArrayRef<mlir::ValueRange> operands = adaptor.getOperands();
    if (operands.size() != op->getNumOperands())
      return rewriter.notifyMatchFailure(
          op, "exception.new operand conversion arity mismatch");

    auto firstOperand = [&](unsigned index) -> mlir::Value {
      if (index >= operands.size() || operands[index].empty())
        return {};
      return operands[index].front();
    };
    mlir::Value type = firstOperand(0);
    mlir::Value message = firstOperand(1);
    mlir::Value cause = firstOperand(3);
    mlir::Value context = firstOperand(4);
    mlir::Value traceback = firstOperand(5);
    mlir::Value location = firstOperand(6);
    if (!type || !message || !cause || !context || !traceback || !location)
      return rewriter.notifyMatchFailure(
          op, "exception.new requires scalar runtime bridge operands");

    mlir::Value nullPtr = rewriter.create<mlir::LLVM::ZeroOp>(
        op.getLoc(), mlir::LLVM::LLVMPointerType::get(rewriter.getContext()));
    auto call =
        runtime.call(op.getLoc(), RuntimeSymbols::kExceptionNew, resultType,
                     mlir::ValueRange{type, message, nullPtr, cause, context,
                                      traceback, location, nullPtr});
    rewriter.replaceOp(op, call.getResults());
    return mlir::success();
  }
};

struct TryLowering : public mlir::OpRewritePattern<TryOp> {
  TryLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpRewritePattern<TryOp>(ctx, mlir::PatternBenefit(10)),
        typeConverter(converter) {}

  mlir::LogicalResult
  matchAndRewrite(TryOp op, mlir::PatternRewriter &rewriter) const override {
    if (op.getTryRegion().empty())
      return rewriter.notifyMatchFailure(op, "empty try region");

    bool hasExcept = !op.getExceptRegion().empty();
    bool hasFinally = !op.getFinallyRegion().empty();
    bool needsFinallyEntry = hasFinally && hasFinallyFallthrough(op);

    llvm::SmallVector<llvm::SmallVector<mlir::Type>> convertedResultTypes;
    llvm::SmallVector<mlir::Type> flatResultTypes;
    for (mlir::Type resultType : op.getResultTypes()) {
      llvm::SmallVector<mlir::Type> converted;
      if (mlir::failed(typeConverter.convertType(resultType, converted)) ||
          converted.empty())
        return rewriter.notifyMatchFailure(op, "failed to convert try result");
      flatResultTypes.append(converted.begin(), converted.end());
      convertedResultTypes.push_back(std::move(converted));
    }

    auto convertYieldOperands =
        [&](mlir::Operation *yieldOp, mlir::ValueRange operands,
            llvm::SmallVectorImpl<mlir::Value> &convertedOperands)
        -> mlir::LogicalResult {
      if (operands.size() != convertedResultTypes.size())
        return rewriter.notifyMatchFailure(
            yieldOp, "yield operand count must match py.try results");
      rewriter.setInsertionPoint(yieldOp);
      for (auto [operand, convertedTypes] :
           llvm::zip(operands, convertedResultTypes)) {
        if (isPyOwnershipTrackedType(operand.getType())) {
          if (mlir::Operation *def = operand.getDefiningOp())
            if (isPyOwnershipImmortalOp(def))
              rewriter.create<IncRefOp>(yieldOp->getLoc(), operand);
        }
        if (convertedTypes.size() == 1 &&
            operand.getType() == convertedTypes.front()) {
          convertedOperands.push_back(operand);
          continue;
        }
        auto cast = rewriter.create<mlir::UnrealizedConversionCastOp>(
            yieldOp->getLoc(), mlir::TypeRange(convertedTypes), operand);
        llvm::append_range(convertedOperands, cast.getResults());
      }
      return mlir::success();
    };

    if (hasFinally && !hasExcept) {
      llvm::SmallVector<mlir::Block *> tryBlocksForAwait;
      for (mlir::Block &block : op.getTryRegion())
        tryBlocksForAwait.push_back(&block);
      llvm::SmallVector<AsyncGatherOp> gathers;
      collectGathersExcludingNestedTry(tryBlocksForAwait, gathers);
      for (AsyncGatherOp gatherOp : gathers)
        if (mlir::failed(
                lowerGatherInFinallyTry(gatherOp, op, typeConverter, rewriter)))
          return mlir::failure();
      tryBlocksForAwait.clear();
      for (mlir::Block &block : op.getTryRegion())
        tryBlocksForAwait.push_back(&block);
      llvm::SmallVector<AwaitOp> awaits;
      collectAwaitsExcludingNestedTry(tryBlocksForAwait, awaits);
      for (AwaitOp awaitOp : awaits)
        if (mlir::failed(
                lowerAwaitInFinallyTry(awaitOp, op, typeConverter, rewriter)))
          return mlir::failure();
    }

    mlir::Region *parent = op->getParentRegion();
    mlir::Block *parentBlock = op->getBlock();
    auto insertIt = std::next(mlir::Region::iterator(parentBlock));
    mlir::Block *mergeBlock = rewriter.createBlock(parent, insertIt);
    for (mlir::Type resultType : flatResultTypes)
      mergeBlock->addArgument(resultType, op.getLoc());

    mlir::Block *tryEntry = &op.getTryRegion().front();
    mlir::Block *exceptEntry =
        hasExcept ? &op.getExceptRegion().front() : nullptr;
    mlir::Block *finallyEntry =
        hasFinally ? &op.getFinallyRegion().front() : nullptr;
    if (needsFinallyEntry &&
        mlir::failed(convertFinallyEntryArguments(
            op, finallyEntry, convertedResultTypes, flatResultTypes, rewriter)))
      return mlir::failure();

    llvm::SmallVector<mlir::Block *> tryBlocks;
    llvm::SmallVector<mlir::Block *> exceptBlocks;
    llvm::SmallVector<mlir::Block *> finallyBlocks;
    for (mlir::Block &block : op.getTryRegion())
      tryBlocks.push_back(&block);
    if (hasExcept)
      for (mlir::Block &block : op.getExceptRegion())
        exceptBlocks.push_back(&block);
    if (hasFinally)
      for (mlir::Block &block : op.getFinallyRegion())
        finallyBlocks.push_back(&block);

    // Move operations after py.try into merge block.
    for (auto it = std::next(op->getIterator()); it != parentBlock->end();) {
      mlir::Operation *move = &*it++;
      move->moveBefore(mergeBlock, mergeBlock->end());
    }

    // Inline regions into parent before merge block.
    rewriter.inlineRegionBefore(op.getTryRegion(), *parent,
                                mergeBlock->getIterator());
    if (hasExcept)
      rewriter.inlineRegionBefore(op.getExceptRegion(), *parent,
                                  mergeBlock->getIterator());
    if (needsFinallyEntry)
      rewriter.inlineRegionBefore(op.getFinallyRegion(), *parent,
                                  mergeBlock->getIterator());

    if (hasExcept) {
      mlir::FailureOr<mlir::Value> caughtException =
          convertExceptEntryArgument(exceptEntry, typeConverter, rewriter);
      if (mlir::failed(caughtException))
        return mlir::failure();
      replaceRaiseCurrentInExceptBlocks(exceptBlocks, *caughtException,
                                        rewriter);
      llvm::SmallVector<AsyncGatherOp> gathers;
      collectGathersExcludingNestedTry(tryBlocks, gathers);
      for (AsyncGatherOp gatherOp : gathers)
        if (mlir::failed(lowerGatherInExceptTry(gatherOp, exceptEntry,
                                                typeConverter, rewriter)))
          return mlir::failure();
      llvm::SmallVector<AwaitOp> awaits;
      collectAwaitsExcludingNestedTry(collectBlockSpan(tryEntry, exceptEntry),
                                      awaits);
      for (AwaitOp awaitOp : awaits)
        if (mlir::failed(lowerAwaitInExceptTry(awaitOp, exceptEntry,
                                               typeConverter, rewriter)))
          return mlir::failure();
    }

    // Redirect invokes inside try region to except entry.
    if (hasExcept) {
      llvm::SmallVector<mlir::Block *> activeTryBlocks =
          collectBlockSpan(tryEntry, exceptEntry);
      for (mlir::Block *block : activeTryBlocks) {
        for (auto invoke : block->getOps<InvokeOp>()) {
          mlir::OpBuilder builder(invoke);
          mlir::Type exceptArgType = exceptEntry->getArgument(0).getType();
          mlir::Value excNull;
          if (mlir::isa<mlir::LLVM::LLVMPointerType>(exceptArgType)) {
            excNull = builder.create<mlir::LLVM::ZeroOp>(invoke.getLoc(),
                                                         exceptArgType);
          } else {
            excNull = builder.create<ExceptionNullOp>(
                invoke.getLoc(), ExceptionType::get(op.getContext()));
            if (excNull.getType() != exceptArgType)
              excNull = builder
                            .create<mlir::UnrealizedConversionCastOp>(
                                invoke.getLoc(), mlir::TypeRange{exceptArgType},
                                excNull)
                            .getResult(0);
          }
          invoke.getUnwindDestOperandsMutable().assign(excNull);
          invoke->setSuccessor(exceptEntry, 1);
        }
        if (auto raise = mlir::dyn_cast<RaiseOp>(block->getTerminator())) {
          rewriter.setInsertionPoint(raise);
          mlir::Value exception = raise.getException();
          mlir::Type exceptArgType = exceptEntry->getArgument(0).getType();
          if (exception.getType() != exceptArgType)
            exception = rewriter
                            .create<mlir::UnrealizedConversionCastOp>(
                                raise.getLoc(), mlir::TypeRange{exceptArgType},
                                exception)
                            .getResult(0);
          eraseOpsAfter(raise, rewriter);
          raise->setAttr("ly.redirected_to_except", rewriter.getUnitAttr());
          rewriter.create<mlir::cf::BranchOp>(raise.getLoc(), exceptEntry,
                                              mlir::ValueRange{exception});
          rewriter.eraseOp(raise);
        }
      }
    }

    if (hasFinally) {
      llvm::SmallVector<mlir::Block *> protectedBlocks = collectBlockSpan(
          tryEntry, needsFinallyEntry ? finallyEntry : mergeBlock);
      cloneFinallyBeforeEscapes(finallyEntry, protectedBlocks, rewriter);
      eraseOpsAfterTerminators(protectedBlocks, rewriter);
    }

    // Replace yields with branches. Await lowering may split the original
    // region blocks, so scan the inlined span for this py.try instead of the
    // saved pre-split block lists. Do not scan the whole parent region: nested
    // try lowering can otherwise consume an outer py.try.yield.
    llvm::SmallVector<TryYieldOp> tryYields;
    llvm::SmallVector<ExceptYieldOp> exceptYields;
    llvm::SmallVector<FinallyYieldOp> finallyYields;
    llvm::SmallVector<mlir::Block *> yieldBlocks =
        collectBlockSpan(tryEntry, mergeBlock);
    for (mlir::Block *block : yieldBlocks) {
      if (auto yield = mlir::dyn_cast<TryYieldOp>(block->getTerminator()))
        tryYields.push_back(yield);
      if (auto yield = mlir::dyn_cast<ExceptYieldOp>(block->getTerminator()))
        exceptYields.push_back(yield);
      if (auto yield = mlir::dyn_cast<FinallyYieldOp>(block->getTerminator()))
        finallyYields.push_back(yield);
    }
    for (TryYieldOp yield : tryYields) {
      rewriter.setInsertionPoint(yield);
      llvm::SmallVector<mlir::Value> convertedOperands;
      if (mlir::failed(convertYieldOperands(yield, yield.getOperands(),
                                            convertedOperands)))
        return mlir::failure();
      if (needsFinallyEntry) {
        rewriter.create<mlir::cf::BranchOp>(yield.getLoc(), finallyEntry,
                                            convertedOperands);
      } else {
        rewriter.create<mlir::cf::BranchOp>(yield.getLoc(), mergeBlock,
                                            convertedOperands);
      }
      rewriter.eraseOp(yield);
    }
    for (ExceptYieldOp yield : exceptYields) {
      rewriter.setInsertionPoint(yield);
      llvm::SmallVector<mlir::Value> convertedOperands;
      if (mlir::failed(convertYieldOperands(yield, yield.getOperands(),
                                            convertedOperands)))
        return mlir::failure();
      if (needsFinallyEntry) {
        rewriter.create<mlir::cf::BranchOp>(yield.getLoc(), finallyEntry,
                                            convertedOperands);
      } else {
        rewriter.create<mlir::cf::BranchOp>(yield.getLoc(), mergeBlock,
                                            convertedOperands);
      }
      rewriter.eraseOp(yield);
    }
    for (FinallyYieldOp yield : finallyYields) {
      rewriter.setInsertionPoint(yield);
      llvm::SmallVector<mlir::Value> forwarded;
      if (yield->getNumOperands() > 0) {
        if (mlir::failed(
                convertYieldOperands(yield, yield.getOperands(), forwarded)))
          return mlir::failure();
      } else if (needsFinallyEntry) {
        llvm::append_range(forwarded, finallyEntry->getArguments());
      }
      rewriter.create<mlir::cf::BranchOp>(yield.getLoc(), mergeBlock,
                                          mlir::ValueRange{forwarded});
      rewriter.eraseOp(yield);
    }

    // Connect parent block to try entry.
    rewriter.setInsertionPointToEnd(parentBlock);
    rewriter.create<mlir::cf::BranchOp>(op.getLoc(), tryEntry);

    rewriter.setInsertionPointToStart(mergeBlock);
    unsigned flatIndex = 0;
    for (auto [result, convertedTypes] :
         llvm::zip(op.getResults(), convertedResultTypes)) {
      llvm::SmallVector<mlir::Value> replacement;
      for (unsigned offset = 0; offset < convertedTypes.size(); ++offset)
        replacement.push_back(mergeBlock->getArgument(flatIndex + offset));
      flatIndex += convertedTypes.size();
      mlir::FailureOr<mlir::Value> logicalReplacement = materializeLogicalValue(
          op.getLoc(), result.getType(), replacement, rewriter);
      if (mlir::failed(logicalReplacement))
        return rewriter.notifyMatchFailure(
            op, "failed to rematerialize try result");
      result.replaceAllUsesWith(*logicalReplacement);
    }
    rewriter.eraseOp(op);
    return mlir::success();
  }

private:
  const PyLLVMTypeConverter &typeConverter;
};

struct RaiseLowering : public mlir::OpConversionPattern<RaiseOp> {
  RaiseLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<RaiseOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(RaiseOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return mlir::failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *typeConverter);
    if (mlir::Value exceptionCell = getCurrentAsyncExceptionCell(op)) {
      storeExceptionCell(op.getLoc(), exceptionCell, adaptor.getException(),
                         module, rewriter, *typeConverter);
      runtime.call(op.getLoc(), RuntimeSymbols::kDecRef, mlir::Type(),
                   mlir::ValueRange{adaptor.getException()});
      mlir::Value falseValue = rewriter.create<mlir::arith::ConstantOp>(
          op.getLoc(), rewriter.getBoolAttr(false));
      rewriter.create<mlir::cf::AssertOp>(op.getLoc(), falseValue,
                                          "lython async exception");
      mlir::Block *currentBlock = rewriter.getInsertionBlock();
      mlir::Block *deadBlock = rewriter.createBlock(
          currentBlock->getParent(), std::next(currentBlock->getIterator()));
      rewriter.setInsertionPoint(op);
      rewriter.create<mlir::cf::BranchOp>(op.getLoc(), deadBlock);
      rewriter.setInsertionPointToStart(deadBlock);
      rewriter.create<mlir::cf::BranchOp>(op.getLoc(), deadBlock);
      rewriter.eraseOp(op);
      return mlir::success();
    }
    mlir::StringAttr fileAttr;
    std::int64_t line = 0;
    std::int64_t col = 0;
    getLocInfo(op.getLoc(), op.getContext(), fileAttr, line, col);
    mlir::StringAttr funcAttr = getFuncNameAttr(
        op->getParentOfType<mlir::func::FuncOp>(), op.getContext());
    mlir::Value filePtr = runtime.getStringLiteral(op.getLoc(), fileAttr);
    mlir::Value funcPtr = runtime.getStringLiteral(op.getLoc(), funcAttr);
    mlir::Value lineConst = rewriter.create<mlir::LLVM::ConstantOp>(
        op.getLoc(), rewriter.getI32Type(),
        rewriter.getI32IntegerAttr(static_cast<int32_t>(line)));
    mlir::Value colConst = rewriter.create<mlir::LLVM::ConstantOp>(
        op.getLoc(), rewriter.getI32Type(),
        rewriter.getI32IntegerAttr(static_cast<int32_t>(col)));
    runtime.call(op.getLoc(), RuntimeSymbols::kTracebackPush, mlir::Type(),
                 mlir::ValueRange{filePtr, funcPtr, lineConst, colConst});
    runtime.call(op.getLoc(), RuntimeSymbols::kEHThrow, mlir::Type(),
                 mlir::ValueRange{adaptor.getException()});
    rewriter.create<mlir::LLVM::UnreachableOp>(op.getLoc());
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct RaiseCurrentLowering : public mlir::OpConversionPattern<RaiseCurrentOp> {
  RaiseCurrentLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<RaiseCurrentOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(RaiseCurrentOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return mlir::failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *typeConverter);
    mlir::Type pyObject = runtime.getPyObjectPtrType();
    auto current =
        runtime.call(op.getLoc(), RuntimeSymbols::kExceptionGetCurrent,
                     pyObject, mlir::ValueRange{});
    auto retain = runtime.call(op.getLoc(), RuntimeSymbols::kIncRef,
                               mlir::Type(), current.getResults());
    threadsafe::Retain::premise(retain.getOperation(),
                                ThreadSafetyAttrs::kPremiseOwnedToken);
    runtime.call(op.getLoc(), RuntimeSymbols::kEHThrow, mlir::Type(),
                 current.getResults());
    rewriter.create<mlir::LLVM::UnreachableOp>(op.getLoc());
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

} // namespace

namespace lowering::value::base::Patterns {
void populate(PyLLVMTypeConverter &typeConverter,
              mlir::RewritePatternSet &patterns) {
  auto *ctx = patterns.getContext();
  patterns.add<ExceptionNullLowering, TracebackNullLowering,
               LocationCurrentLowering, ExceptionNewLowering, RaiseLowering,
               RaiseCurrentLowering>(typeConverter, ctx);
}
} // namespace lowering::value::base::Patterns

namespace lowering::try_ops::Patterns {
void populate(PyLLVMTypeConverter &typeConverter,
              mlir::RewritePatternSet &patterns) {
  auto *ctx = patterns.getContext();
  patterns.add<TryLowering>(typeConverter, ctx);
}
} // namespace lowering::try_ops::Patterns

} // namespace py
