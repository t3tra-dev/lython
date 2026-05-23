#include "Common/Container.h"
#include "Common/LoweringUtils.h"
#include "Common/RuntimeSupport.h"
#include "Common/SlotUtils.h"
#include "Passes/Runtime/Helpers.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"

#include <cstdint>
#include <string>

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

namespace py {
namespace {

static bool isExceptionCell(mlir::Value value);

static mlir::Type
getAsyncPayloadStorageType(mlir::Location loc, mlir::Type logicalType,
                           const PyLLVMTypeConverter &typeConverter,
                           mlir::ConversionPatternRewriter &rewriter) {
  llvm::SmallVector<mlir::Type> parts;
  if (mlir::failed(typeConverter.convertType(logicalType, parts)) ||
      parts.empty())
    return {};

  llvm::SmallVector<mlir::Type> storageParts;
  for (mlir::Type part : parts) {
    mlir::Type storage = typeConverter.convertType(part);
    if (!storage)
      storage = part;
    storageParts.push_back(storage);
  }
  if (storageParts.size() == 1)
    return storageParts.front();
  return mlir::LLVM::LLVMStructType::getLiteral(rewriter.getContext(),
                                                storageParts);
}

static mlir::Value packAsyncPayload(mlir::Location loc, mlir::Type logicalType,
                                    mlir::ValueRange convertedValues,
                                    const PyLLVMTypeConverter &typeConverter,
                                    mlir::ConversionPatternRewriter &rewriter) {
  mlir::Type storageType =
      getAsyncPayloadStorageType(loc, logicalType, typeConverter, rewriter);
  if (!storageType)
    return {};
  if (convertedValues.size() == 1 &&
      convertedValues.front().getType() == storageType)
    return convertedValues.front();
  auto cast = rewriter.create<mlir::UnrealizedConversionCastOp>(
      loc, mlir::TypeRange{storageType}, convertedValues);
  return cast.getResult(0);
}

static mlir::Value stripDescriptorCasts(mlir::Value value) {
  while (value) {
    if (auto cast = value.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
      if (cast->getNumOperands() == 1) {
        value = cast.getOperand(0);
        continue;
      }
    }
    if (auto cast = value.getDefiningOp<mlir::memref::CastOp>()) {
      value = cast.getSource();
      continue;
    }
    break;
  }
  return value;
}

static bool
needsAsyncReturnContainerPromotion(mlir::ValueRange convertedValues) {
  if (convertedValues.empty())
    return false;
  mlir::Value header = stripDescriptorCasts(convertedValues.front());
  return header && header.getDefiningOp<mlir::memref::AllocaOp>();
}

static mlir::FailureOr<llvm::SmallVector<mlir::Value>>
prepareAsyncReturnPayload(mlir::Location loc, mlir::Type logicalType,
                          mlir::ValueRange convertedValues,
                          mlir::ModuleOp module,
                          const PyLLVMTypeConverter &typeConverter,
                          mlir::ConversionPatternRewriter &rewriter) {
  if ((isCompilerOwnedMemRefListType(logicalType) ||
       isCompilerOwnedMemRefDictType(logicalType) ||
       isCompilerOwnedMemRefTupleType(logicalType)) &&
      needsAsyncReturnContainerPromotion(convertedValues)) {
    return container::Descriptor::promote(loc, logicalType, convertedValues,
                                          module, rewriter, typeConverter,
                                          /*cloneReferenceSlots=*/false);
  }

  llvm::SmallVector<mlir::Value> prepared(convertedValues.begin(),
                                          convertedValues.end());
  return prepared;
}

static llvm::SmallVector<mlir::Value>
unpackAsyncPayload(mlir::Location loc, mlir::Type logicalType,
                   mlir::Value storage,
                   const PyLLVMTypeConverter &typeConverter,
                   mlir::ConversionPatternRewriter &rewriter) {
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

static mlir::LLVM::LLVMFuncOp
getOrInsertRuntimeFunc(mlir::Location loc, mlir::ModuleOp module,
                       mlir::OpBuilder &builder, llvm::StringRef name,
                       mlir::Type resultType,
                       llvm::ArrayRef<mlir::Type> argTypes) {
  if (auto fn = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(name))
    return fn;

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());
  auto fnType = mlir::LLVM::LLVMFunctionType::get(resultType, argTypes, false);
  return builder.create<mlir::LLVM::LLVMFuncOp>(loc, name, fnType);
}

static constexpr llvm::StringLiteral kAsyncExceptionEdgeMarker{
    "__lython_async_exception_edge_marker"};
static constexpr llvm::StringLiteral kAsyncTaskCancelMarker{
    "__lython_async_task_cancel_marker"};

static mlir::func::FuncOp
getOrInsertAsyncExceptionEdgeMarker(mlir::Location loc, mlir::ModuleOp module,
                                    mlir::OpBuilder &builder) {
  if (auto fn =
          module.lookupSymbol<mlir::func::FuncOp>(kAsyncExceptionEdgeMarker))
    return fn;

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());
  auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  auto fnType = builder.getFunctionType(mlir::TypeRange{ptrType, ptrType}, {});
  auto fn = builder.create<mlir::func::FuncOp>(loc, kAsyncExceptionEdgeMarker,
                                               fnType);
  fn.setPrivate();
  async_runtime::ExceptionCell::markArgument(fn.getOperation(), 0);
  async_runtime::ExceptionCell::markArgument(fn.getOperation(), 1);
  return fn;
}

static void emitAwaitExceptionMarker(mlir::Location loc,
                                     mlir::ConversionPatternRewriter &rewriter,
                                     mlir::ModuleOp module,
                                     mlir::Value sourceCell,
                                     mlir::Value destCell) {
  if (!isExceptionCell(sourceCell) || !isExceptionCell(destCell) ||
      sourceCell == destCell)
    return;
  auto marker = getOrInsertAsyncExceptionEdgeMarker(loc, module, rewriter);
  rewriter.create<mlir::func::CallOp>(loc, marker.getSymName(),
                                      mlir::TypeRange{},
                                      mlir::ValueRange{sourceCell, destCell});
}

static mlir::func::FuncOp
getOrInsertAsyncTaskCancelMarker(mlir::Location loc, mlir::ModuleOp module,
                                 mlir::OpBuilder &builder) {
  if (auto fn = module.lookupSymbol<mlir::func::FuncOp>(kAsyncTaskCancelMarker))
    return fn;

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());
  auto flagType = mlir::MemRefType::get({1}, builder.getI8Type());
  auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  auto fnType =
      builder.getFunctionType(mlir::TypeRange{flagType, ptrType, ptrType}, {});
  auto fn =
      builder.create<mlir::func::FuncOp>(loc, kAsyncTaskCancelMarker, fnType);
  fn.setPrivate();
  async_runtime::CancelFlag::markArgument(fn.getOperation(), 0);
  async_runtime::ExceptionCell::markArgument(fn.getOperation(), 1);
  async_runtime::ExceptionCell::markArgument(fn.getOperation(), 2);
  return fn;
}

static void emitTaskCancelMarker(mlir::Location loc,
                                 mlir::ConversionPatternRewriter &rewriter,
                                 mlir::ModuleOp module,
                                 mlir::ValueRange taskDescriptor,
                                 mlir::Value destCell) {
  if (taskDescriptor.size() != 3 || !isExceptionCell(taskDescriptor[1]) ||
      !isExceptionCell(destCell) ||
      !mlir::isa<mlir::MemRefType>(taskDescriptor[2].getType()))
    return;
  auto marker = getOrInsertAsyncTaskCancelMarker(loc, module, rewriter);
  rewriter.create<mlir::func::CallOp>(
      loc, marker.getSymName(), mlir::TypeRange{},
      mlir::ValueRange{taskDescriptor[2], taskDescriptor[1], destCell});
}

static void getLocInfo(mlir::Location loc, mlir::MLIRContext *ctx,
                       mlir::StringAttr &fileAttr, std::int64_t &line,
                       std::int64_t &col) {
  if (auto fileLoc = mlir::dyn_cast<mlir::FileLineColLoc>(loc)) {
    fileAttr = fileLoc.getFilename();
    line = static_cast<std::int64_t>(fileLoc.getLine());
    col = static_cast<std::int64_t>(fileLoc.getColumn());
    return;
  }
  if (auto nameLoc = mlir::dyn_cast<mlir::NameLoc>(loc)) {
    getLocInfo(nameLoc.getChildLoc(), ctx, fileAttr, line, col);
    return;
  }
  if (auto fused = mlir::dyn_cast<mlir::FusedLoc>(loc)) {
    for (mlir::Location subloc : fused.getLocations()) {
      if (auto subfile = mlir::dyn_cast<mlir::FileLineColLoc>(subloc)) {
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

static mlir::StringAttr getCurrentCallableNameAttr(mlir::Operation *op,
                                                   mlir::MLIRContext *ctx) {
  if (!op)
    return mlir::StringAttr::get(ctx, "<unknown>");
  if (auto func = mlir::dyn_cast<mlir::func::FuncOp>(op)) {
    llvm::StringRef name = func.getName();
    return mlir::StringAttr::get(ctx, name == "main" ? "<module>" : name);
  }
  if (auto func = op->getParentOfType<mlir::func::FuncOp>()) {
    llvm::StringRef name = func.getName();
    return mlir::StringAttr::get(ctx, name == "main" ? "<module>" : name);
  }
  if (auto asyncFunc = mlir::dyn_cast<mlir::async::FuncOp>(op))
    return mlir::StringAttr::get(ctx, asyncFunc.getName());
  if (auto asyncFunc = op->getParentOfType<mlir::async::FuncOp>())
    return mlir::StringAttr::get(ctx, asyncFunc.getName());
  return mlir::StringAttr::get(ctx, "<unknown>");
}

static void emitTracebackPush(mlir::Location loc, mlir::ModuleOp module,
                              mlir::ConversionPatternRewriter &rewriter,
                              const PyLLVMTypeConverter &typeConverter) {
  RuntimeAPI runtime(module, rewriter, typeConverter);
  mlir::StringAttr fileAttr;
  std::int64_t line = 0;
  std::int64_t col = 0;
  getLocInfo(loc, rewriter.getContext(), fileAttr, line, col);
  mlir::Operation *parentOp = rewriter.getInsertionBlock()
                                  ? rewriter.getInsertionBlock()->getParentOp()
                                  : nullptr;
  mlir::StringAttr funcAttr =
      getCurrentCallableNameAttr(parentOp, rewriter.getContext());
  mlir::Value filePtr = runtime.getStringLiteral(loc, fileAttr);
  mlir::Value funcPtr = runtime.getStringLiteral(loc, funcAttr);
  mlir::Value lineConst = rewriter.create<mlir::LLVM::ConstantOp>(
      loc, rewriter.getI32Type(),
      rewriter.getI32IntegerAttr(static_cast<int32_t>(line)));
  mlir::Value colConst = rewriter.create<mlir::LLVM::ConstantOp>(
      loc, rewriter.getI32Type(),
      rewriter.getI32IntegerAttr(static_cast<int32_t>(col)));
  runtime.call(loc, RuntimeSymbols::kTracebackPush, mlir::Type(),
               mlir::ValueRange{filePtr, funcPtr, lineConst, colConst});
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

static mlir::Value getAwaitableExceptionCell(mlir::ValueRange awaitable) {
  if (awaitable.size() < 2)
    return {};
  return isExceptionCell(awaitable[1]) ? awaitable[1] : mlir::Value();
}

static mlir::ValueRange
expandConvertedAsyncDescriptor(mlir::ValueRange descriptor,
                               llvm::SmallVectorImpl<mlir::Value> &sink) {
  if (descriptor.size() != 1)
    return descriptor;
  auto cast =
      descriptor.front().getDefiningOp<mlir::UnrealizedConversionCastOp>();
  if (!cast || cast->getNumResults() != 1 ||
      cast.getResult(0) != descriptor.front())
    return descriptor;
  if (!mlir::isa<CoroutineType, FutureType, TaskType>(
          cast.getResult(0).getType()))
    return descriptor;
  sink.append(cast.getOperands().begin(), cast.getOperands().end());
  return mlir::ValueRange(sink);
}

static mlir::Value
createExceptionCell(mlir::Location loc, mlir::ModuleOp module,
                    mlir::ConversionPatternRewriter &rewriter,
                    const PyLLVMTypeConverter &typeConverter) {
  RuntimeAPI runtime(module, rewriter, typeConverter);
  auto ptrType = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
  mlir::Value cellSize = runtime.getI64Constant(loc, 8);
  mlir::Value cell = runtime
                         .call(loc, RuntimeSymbols::kMemAlloc, ptrType,
                               mlir::ValueRange{cellSize})
                         .getResult();
  async_runtime::ExceptionCell::mark(cell);
  mlir::Value nullPtr = rewriter.create<mlir::LLVM::ZeroOp>(loc, ptrType);
  rewriter.create<mlir::LLVM::StoreOp>(loc, nullPtr, cell);
  return cell;
}

static mlir::Value loadExceptionCell(mlir::Location loc, mlir::Value cell,
                                     mlir::OpBuilder &builder) {
  return async_runtime::ExceptionCell::load(loc, cell, builder);
}

static void
emitReleaseAggregateLoadedException(mlir::Location loc, mlir::ModuleOp module,
                                    mlir::ConversionPatternRewriter &rewriter,
                                    const PyLLVMTypeConverter &typeConverter,
                                    mlir::Value exception) {
  async_runtime::ExceptionCell::releaseLoaded(loc, module, rewriter,
                                              typeConverter, exception);
}

static void emitReleaseNoneAwaitPayload(
    mlir::Location loc, mlir::Type logicalType, mlir::ModuleOp module,
    mlir::ConversionPatternRewriter &rewriter,
    const PyLLVMTypeConverter &typeConverter, mlir::Value payload) {
  if (!mlir::isa<NoneType>(logicalType))
    return;
  if (!payload || !mlir::isa<mlir::LLVM::LLVMPointerType>(payload.getType()))
    return;

  RuntimeAPI runtime(module, rewriter, typeConverter);
  runtime.call(loc, RuntimeSymbols::kDecRef, mlir::Type(),
               mlir::ValueRange{payload});
}

static void emitFreeExceptionCell(mlir::Location loc, mlir::ModuleOp module,
                                  mlir::ConversionPatternRewriter &rewriter,
                                  const PyLLVMTypeConverter &typeConverter,
                                  mlir::Value exceptionCell) {
  if (!isExceptionCell(exceptionCell))
    return;
  async_runtime::ExceptionCell::free(loc, module, rewriter, typeConverter,
                                     exceptionCell);
}

static void emitDestroyExceptionCell(mlir::Location loc, mlir::ModuleOp module,
                                     mlir::ConversionPatternRewriter &rewriter,
                                     const PyLLVMTypeConverter &typeConverter,
                                     mlir::Value exceptionCell) {
  if (!isExceptionCell(exceptionCell))
    return;
  async_runtime::ExceptionCell::destroy(loc, module, rewriter, typeConverter,
                                        exceptionCell);
}

static void
emitDestroyTaskCancelFlag(mlir::Location loc, mlir::ValueRange descriptor,
                          mlir::ConversionPatternRewriter &rewriter) {
  if (descriptor.size() != 3)
    return;
  if (!mlir::isa<mlir::MemRefType>(descriptor[2].getType()))
    return;
  rewriter.create<mlir::memref::DeallocOp>(loc, descriptor[2]);
}

static void emitDropAsyncHandle(mlir::Location loc, mlir::Value asyncValue,
                                mlir::ConversionPatternRewriter &rewriter) {
  if (!mlir::isa<mlir::async::ValueType>(asyncValue.getType()))
    return;
  rewriter.create<mlir::async::RuntimeDropRefOp>(loc, asyncValue,
                                                 rewriter.getI64IntegerAttr(1));
}

static void storeExceptionCell(mlir::Location loc, mlir::Value cell,
                               mlir::Value exception, mlir::ModuleOp module,
                               mlir::ConversionPatternRewriter &rewriter,
                               const PyLLVMTypeConverter &typeConverter) {
  if (!isExceptionCell(cell))
    return;
  async_runtime::ExceptionCell::storeFirst(loc, cell, exception, module,
                                           rewriter, typeConverter);
}

static mlir::Value
createExceptionWithMessage(mlir::Location loc, mlir::ModuleOp module,
                           mlir::ConversionPatternRewriter &rewriter,
                           const PyLLVMTypeConverter &typeConverter,
                           llvm::StringRef messageText) {
  RuntimeAPI runtime(module, rewriter, typeConverter);
  mlir::Type pyObject = runtime.getPyObjectPtrType();
  mlir::Value nullPtr = rewriter.create<mlir::LLVM::ZeroOp>(loc, pyObject);
  mlir::Value messageBytes =
      runtime.getStringLiteral(loc, rewriter.getStringAttr(messageText));
  mlir::Value messageLength = runtime.getI64Constant(
      loc, static_cast<std::int64_t>(messageText.size()));
  auto message = runtime.call(loc, RuntimeSymbols::kStrFromUtf8, pyObject,
                              mlir::ValueRange{messageBytes, messageLength});
  auto exception = runtime.call(loc, RuntimeSymbols::kExceptionNew, pyObject,
                                mlir::ValueRange{nullPtr, message.getResult(),
                                                 nullPtr, nullPtr, nullPtr,
                                                 nullPtr, nullPtr, nullPtr});
  runtime.call(loc, RuntimeSymbols::kDecRef, mlir::Type(),
               message.getResults());
  return exception.getResult();
}

static bool isKnownAvailableAsyncValueBefore(mlir::Value asyncValue,
                                             mlir::Operation *before) {
  if (!asyncValue || !before || !asyncValue.getDefiningOp())
    return false;
  if (asyncValue.getDefiningOp()->getBlock() != before->getBlock())
    return false;

  for (mlir::Operation *cursor = before->getPrevNode(); cursor;
       cursor = cursor->getPrevNode()) {
    if (auto setAvailable =
            mlir::dyn_cast<mlir::async::RuntimeSetAvailableOp>(cursor))
      if (setAvailable.getOperand() == asyncValue)
        return true;
    if (auto setError = mlir::dyn_cast<mlir::async::RuntimeSetErrorOp>(cursor))
      if (setError.getOperand() == asyncValue)
        return false;
    if (cursor == asyncValue.getDefiningOp())
      return false;
  }
  return false;
}

static void emitThrowException(mlir::Location loc, mlir::ModuleOp module,
                               mlir::ConversionPatternRewriter &rewriter,
                               const PyLLVMTypeConverter &typeConverter,
                               mlir::Value exception) {
  RuntimeAPI runtime(module, rewriter, typeConverter);
  runtime.call(loc, RuntimeSymbols::kEHThrow, mlir::Type(),
               mlir::ValueRange{exception});
  rewriter.create<mlir::LLVM::UnreachableOp>(loc);
}

static void
emitFallbackAsyncAwaitErrorThrow(mlir::Location loc, mlir::ModuleOp module,
                                 mlir::ConversionPatternRewriter &rewriter,
                                 const PyLLVMTypeConverter &typeConverter) {
  mlir::Value exception = createExceptionWithMessage(
      loc, module, rewriter, typeConverter, "task cancelled");
  emitThrowException(loc, module, rewriter, typeConverter, exception);
}

static void emitAsyncAwaitErrorThrow(mlir::Location loc, mlir::ModuleOp module,
                                     mlir::ConversionPatternRewriter &rewriter,
                                     const PyLLVMTypeConverter &typeConverter,
                                     mlir::Value exceptionCell) {
  if (!isExceptionCell(exceptionCell)) {
    emitTracebackPush(loc, module, rewriter, typeConverter);
    emitFallbackAsyncAwaitErrorThrow(loc, module, rewriter, typeConverter);
    return;
  }

  auto ptrType = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
  mlir::Value exception = loadExceptionCell(loc, exceptionCell, rewriter);
  mlir::Value nullPtr = rewriter.create<mlir::LLVM::ZeroOp>(loc, ptrType);
  mlir::Value isNull = rewriter.create<mlir::LLVM::ICmpOp>(
      loc, mlir::LLVM::ICmpPredicate::eq, exception, nullPtr);
  emitTracebackPush(loc, module, rewriter, typeConverter);

  mlir::Block *currentBlock = rewriter.getInsertionBlock();
  mlir::Block *fallbackBlock = rewriter.createBlock(
      currentBlock->getParent(), std::next(currentBlock->getIterator()));
  mlir::Block *throwBlock = rewriter.createBlock(
      currentBlock->getParent(), std::next(fallbackBlock->getIterator()));
  rewriter.setInsertionPointToEnd(currentBlock);
  rewriter.create<mlir::cf::CondBranchOp>(loc, isNull, fallbackBlock,
                                          throwBlock);

  rewriter.setInsertionPointToStart(fallbackBlock);
  emitFreeExceptionCell(loc, module, rewriter, typeConverter, exceptionCell);
  emitFallbackAsyncAwaitErrorThrow(loc, module, rewriter, typeConverter);

  rewriter.setInsertionPointToStart(throwBlock);
  RuntimeAPI runtime(module, rewriter, typeConverter);
  auto retain = runtime.call(loc, RuntimeSymbols::kIncRef, mlir::Type(),
                             mlir::ValueRange{exception});
  threadsafe::Retain::premise(retain.getOperation(),
                              ThreadSafetyAttrs::kPremiseAggregateBorrow);
  emitReleaseAggregateLoadedException(loc, module, rewriter, typeConverter,
                                      exception);
  emitFreeExceptionCell(loc, module, rewriter, typeConverter, exceptionCell);
  emitThrowException(loc, module, rewriter, typeConverter, exception);
}

static void
emitStoreCancellationException(mlir::Location loc, mlir::ModuleOp module,
                               mlir::ConversionPatternRewriter &rewriter,
                               const PyLLVMTypeConverter &typeConverter,
                               mlir::Value exceptionCell) {
  mlir::Value exception = createExceptionWithMessage(
      loc, module, rewriter, typeConverter, "task cancelled");
  storeExceptionCell(loc, exceptionCell, exception, module, rewriter,
                     typeConverter);
  RuntimeAPI runtime(module, rewriter, typeConverter);
  runtime.call(loc, RuntimeSymbols::kDecRef, mlir::Type(),
               mlir::ValueRange{exception});
}

static void emitThrowFirstErroredAwaitable(
    mlir::Location loc, mlir::ModuleOp module,
    mlir::ConversionPatternRewriter &rewriter,
    const PyLLVMTypeConverter &typeConverter,
    llvm::ArrayRef<std::pair<mlir::Value, mlir::Value>> awaitables,
    llvm::ArrayRef<llvm::SmallVector<mlir::Value>> descriptors,
    llvm::ArrayRef<mlir::Type> awaitableTypes) {
  auto emitDestroyAwaitableResources = [&]() {
    for (auto [index, awaitableAndCell] : llvm::enumerate(awaitables)) {
      auto [asyncValue, cell] = awaitableAndCell;
      emitDestroyExceptionCell(loc, module, rewriter, typeConverter, cell);
      if (index < descriptors.size() && index < awaitableTypes.size() &&
          mlir::isa<TaskType>(awaitableTypes[index]))
        emitDestroyTaskCancelFlag(loc, descriptors[index], rewriter);
      emitDropAsyncHandle(loc, asyncValue, rewriter);
    }
  };

  mlir::Block *currentBlock = rewriter.getInsertionBlock();
  emitTracebackPush(loc, module, rewriter, typeConverter);
  mlir::Block *fallbackBlock = rewriter.createBlock(
      currentBlock->getParent(), std::next(currentBlock->getIterator()));
  mlir::Block *nextCheck = fallbackBlock;

  auto ptrType = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
  for (auto [asyncValue, exceptionCell] : llvm::reverse(awaitables)) {
    if (!isExceptionCell(exceptionCell))
      continue;
    mlir::Block *throwBlock = rewriter.createBlock(
        fallbackBlock->getParent(), fallbackBlock->getIterator());
    throwBlock->addArgument(ptrType, loc);
    rewriter.setInsertionPointToStart(throwBlock);
    mlir::Value thrownException = throwBlock->getArgument(0);
    RuntimeAPI runtime(module, rewriter, typeConverter);
    auto retain = runtime.call(loc, RuntimeSymbols::kIncRef, mlir::Type(),
                               mlir::ValueRange{thrownException});
    threadsafe::Retain::premise(retain.getOperation(),
                                ThreadSafetyAttrs::kPremiseAggregateBorrow);
    emitDestroyAwaitableResources();
    emitThrowException(loc, module, rewriter, typeConverter, thrownException);

    mlir::Block *checkBlock = rewriter.createBlock(fallbackBlock->getParent(),
                                                   throwBlock->getIterator());
    rewriter.setInsertionPointToStart(checkBlock);
    (void)asyncValue;
    mlir::Value exception = loadExceptionCell(loc, exceptionCell, rewriter);
    mlir::Value nullPtr = rewriter.create<mlir::LLVM::ZeroOp>(loc, ptrType);
    mlir::Value isNull = rewriter.create<mlir::LLVM::ICmpOp>(
        loc, mlir::LLVM::ICmpPredicate::eq, exception, nullPtr);
    rewriter.create<mlir::cf::CondBranchOp>(loc, isNull, nextCheck,
                                            mlir::ValueRange{}, throwBlock,
                                            mlir::ValueRange{exception});
    nextCheck = checkBlock;
  }

  rewriter.setInsertionPointToEnd(currentBlock);
  rewriter.create<mlir::cf::BranchOp>(loc, nextCheck);

  rewriter.setInsertionPointToStart(fallbackBlock);
  emitDestroyAwaitableResources();
  mlir::Value fallbackException = createExceptionWithMessage(
      loc, module, rewriter, typeConverter, "task cancelled");
  emitThrowException(loc, module, rewriter, typeConverter, fallbackException);
}

static void emitReleaseLoadedPayloads(mlir::Location loc, mlir::ModuleOp module,
                                      mlir::ConversionPatternRewriter &rewriter,
                                      const PyLLVMTypeConverter &typeConverter,
                                      llvm::ArrayRef<mlir::Value> payloads,
                                      llvm::ArrayRef<mlir::Type> elementTypes) {
  for (auto [index, payload] : llvm::enumerate(payloads)) {
    if (index >= elementTypes.size())
      break;
    Slot::releaseSource(loc, payload, elementTypes[index], module, rewriter,
                        typeConverter);
  }
}

struct CoroCreateLowering : public mlir::OpConversionPattern<CoroCreateOp> {
  CoroCreateLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<CoroCreateOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(CoroCreateOp op, OneToNOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    llvm::SmallVector<mlir::Type> resultTypes;
    if (mlir::failed(typeConverter->convertType(op.getResult().getType(),
                                                resultTypes)) ||
        resultTypes.size() != 2)
      return rewriter.notifyMatchFailure(op,
                                         "failed to convert coroutine type");

    mlir::Value exceptionCell =
        createExceptionCell(op.getLoc(), op->getParentOfType<mlir::ModuleOp>(),
                            rewriter, *typeConverter);

    llvm::SmallVector<mlir::Value> callOperands;
    for (mlir::ValueRange arg : adaptor.getArgs()) {
      llvm::SmallVector<mlir::Value> expandedStorage;
      mlir::ValueRange expanded =
          expandConvertedAsyncDescriptor(arg, expandedStorage);
      callOperands.append(expanded.begin(), expanded.end());
    }
    callOperands.push_back(exceptionCell);
    auto call = rewriter.create<mlir::async::CallOp>(
        op.getLoc(), op.getTargetAttr(), mlir::TypeRange{resultTypes.front()},
        callOperands);
    llvm::SmallVector<mlir::Value> descriptor{call.getResult(0), exceptionCell};
    rewriter.replaceOpWithMultiple(
        op, llvm::ArrayRef<mlir::ValueRange>{mlir::ValueRange{descriptor}});
    return mlir::success();
  }
};

struct CoroStartLowering : public mlir::OpConversionPattern<CoroStartOp> {
  CoroStartLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<CoroStartOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(CoroStartOp op, OneToNOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<mlir::Value> coroutineStorage;
    mlir::ValueRange coroutine = expandConvertedAsyncDescriptor(
        adaptor.getCoroutine(), coroutineStorage);
    if (coroutine.empty())
      return mlir::failure();
    rewriter.replaceOp(op, coroutine.front());
    return mlir::success();
  }
};

struct AsyncCallTypeLowering
    : public mlir::OpConversionPattern<mlir::async::CallOp> {
  AsyncCallTypeLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<mlir::async::CallOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::async::CallOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<mlir::Type> resultTypes;
    if (mlir::failed(
            getTypeConverter()->convertTypes(op.getResultTypes(), resultTypes)))
      return rewriter.notifyMatchFailure(op,
                                         "failed to convert async call types");

    rewriter.replaceOpWithNewOp<mlir::async::CallOp>(
        op, op.getCalleeAttr(), resultTypes, adaptor.getOperands());
    return mlir::success();
  }
};

struct AsyncFuncSignatureLowering
    : public mlir::OpConversionPattern<mlir::async::FuncOp> {
  AsyncFuncSignatureLowering(PyLLVMTypeConverter &converter,
                             mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<mlir::async::FuncOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::async::FuncOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    mlir::FunctionType oldType = op.getFunctionType();
    mlir::TypeConverter::SignatureConversion signature(oldType.getNumInputs());
    llvm::SmallVector<mlir::Type> resultTypes;
    if (mlir::failed(typeConverter->convertSignatureArgs(oldType.getInputs(),
                                                         signature)) ||
        mlir::failed(
            typeConverter->convertTypes(oldType.getResults(), resultTypes)))
      return rewriter.notifyMatchFailure(op,
                                         "failed to convert async signature");

    if (!op.isDeclaration()) {
      if (mlir::failed(rewriter.convertRegionTypes(&op.getBody(),
                                                   *typeConverter, &signature)))
        return mlir::failure();
    }

    auto newType =
        rewriter.getFunctionType(signature.getConvertedTypes(), resultTypes);
    rewriter.modifyOpInPlace(op, [&] { op.setType(newType); });
    lowering::runtime::async_args::mark(op.getOperation(), oldType.getInputs(),
                                        *typeConverter,
                                        /*trailingExceptionCell=*/true);
    return mlir::success();
  }
};

struct AwaitLowering : public mlir::OpConversionPattern<AwaitOp> {
  AwaitLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<AwaitOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(AwaitOp op, OneToNOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return mlir::failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    llvm::SmallVector<mlir::Value> awaitableStorage;
    mlir::ValueRange awaitable = expandConvertedAsyncDescriptor(
        adaptor.getAwaitable(), awaitableStorage);
    if (awaitable.empty())
      return mlir::failure();
    mlir::Value asyncValue = awaitable.front();
    mlir::Value exceptionCell = getAwaitableExceptionCell(awaitable);

    bool insideAsyncFunc =
        op->getParentOfType<mlir::async::FuncOp>() != nullptr;
    if (insideAsyncFunc) {
      if (isKnownAvailableAsyncValueBefore(asyncValue, op.getOperation())) {
        auto asyncValueType =
            mlir::dyn_cast<mlir::async::ValueType>(asyncValue.getType());
        if (!asyncValueType)
          return rewriter.notifyMatchFailure(
              op, "awaitable did not lower to value");
        rewriter.create<mlir::async::RuntimeAwaitOp>(op.getLoc(), asyncValue);
        auto awaitOp = rewriter.create<mlir::async::RuntimeLoadOp>(
            op.getLoc(), asyncValueType.getValueType(), asyncValue);
        llvm::SmallVector<mlir::Value> results =
            unpackAsyncPayload(op.getLoc(), op.getResult().getType(),
                               awaitOp.getResult(), *typeConverter, rewriter);
        if (results.empty())
          return mlir::failure();
        emitReleaseNoneAwaitPayload(op.getLoc(), op.getResult().getType(),
                                    module, rewriter, *typeConverter,
                                    results.front());
        emitDestroyExceptionCell(op.getLoc(), module, rewriter, *typeConverter,
                                 exceptionCell);
        if (mlir::isa<TaskType>(op.getAwaitable().getType()))
          emitDestroyTaskCancelFlag(op.getLoc(), awaitable, rewriter);
        emitDropAsyncHandle(op.getLoc(), asyncValue, rewriter);
        rewriter.replaceOpWithMultiple(
            op, llvm::ArrayRef<mlir::ValueRange>{mlir::ValueRange{results}});
        return mlir::success();
      }

      auto asyncAwait =
          rewriter.create<mlir::async::AwaitOp>(op.getLoc(), asyncValue);
      rewriter.setInsertionPointAfter(asyncAwait);
      emitAwaitExceptionMarker(op.getLoc(), rewriter, module, exceptionCell,
                               getCurrentAsyncExceptionCell(op));
      if (mlir::isa<TaskType>(op.getAwaitable().getType()))
        emitTaskCancelMarker(op.getLoc(), rewriter, module, awaitable,
                             getCurrentAsyncExceptionCell(op));
      llvm::SmallVector<mlir::Value> results =
          unpackAsyncPayload(op.getLoc(), op.getResult().getType(),
                             asyncAwait.getResult(), *typeConverter, rewriter);
      if (results.empty())
        return mlir::failure();
      emitReleaseNoneAwaitPayload(op.getLoc(), op.getResult().getType(), module,
                                  rewriter, *typeConverter, results.front());
      emitDestroyExceptionCell(op.getLoc(), module, rewriter, *typeConverter,
                               exceptionCell);
      if (mlir::isa<TaskType>(op.getAwaitable().getType()))
        emitDestroyTaskCancelFlag(op.getLoc(), awaitable, rewriter);
      emitDropAsyncHandle(op.getLoc(), asyncValue, rewriter);
      rewriter.replaceOpWithMultiple(
          op, llvm::ArrayRef<mlir::ValueRange>{mlir::ValueRange{results}});
      return mlir::success();
    }

    auto asyncValueType =
        mlir::dyn_cast<mlir::async::ValueType>(asyncValue.getType());
    if (!asyncValueType)
      return rewriter.notifyMatchFailure(op,
                                         "awaitable did not lower to value");

    rewriter.create<mlir::async::RuntimeAwaitOp>(op.getLoc(), asyncValue);
    auto isError = rewriter.create<mlir::async::RuntimeIsErrorOp>(
        op.getLoc(), rewriter.getI1Type(), asyncValue);

    mlir::Block *currentBlock = rewriter.getInsertionBlock();
    mlir::Block *successBlock =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    mlir::Block *errorBlock = rewriter.createBlock(successBlock->getParent(),
                                                   successBlock->getIterator());
    rewriter.setInsertionPointToEnd(currentBlock);
    rewriter.create<mlir::cf::CondBranchOp>(op.getLoc(), isError.getIsError(),
                                            errorBlock, successBlock);

    rewriter.setInsertionPointToStart(errorBlock);
    if (mlir::isa<TaskType>(op.getAwaitable().getType()))
      emitDestroyTaskCancelFlag(op.getLoc(), awaitable, rewriter);
    emitDropAsyncHandle(op.getLoc(), asyncValue, rewriter);
    emitAsyncAwaitErrorThrow(op.getLoc(), module, rewriter, *typeConverter,
                             exceptionCell);

    rewriter.setInsertionPoint(op);
    auto awaitOp = rewriter.create<mlir::async::RuntimeLoadOp>(
        op.getLoc(), asyncValueType.getValueType(), asyncValue);
    llvm::SmallVector<mlir::Value> results =
        unpackAsyncPayload(op.getLoc(), op.getResult().getType(),
                           awaitOp.getResult(), *typeConverter, rewriter);
    if (results.empty())
      return mlir::failure();
    emitReleaseNoneAwaitPayload(op.getLoc(), op.getResult().getType(), module,
                                rewriter, *typeConverter, results.front());
    emitDestroyExceptionCell(op.getLoc(), module, rewriter, *typeConverter,
                             exceptionCell);
    if (mlir::isa<TaskType>(op.getAwaitable().getType()))
      emitDestroyTaskCancelFlag(op.getLoc(), awaitable, rewriter);
    emitDropAsyncHandle(op.getLoc(), asyncValue, rewriter);
    rewriter.replaceOpWithMultiple(
        op, llvm::ArrayRef<mlir::ValueRange>{mlir::ValueRange{results}});
    return mlir::success();
  }
};

struct AsyncReturnLowering
    : public mlir::OpConversionPattern<mlir::async::ReturnOp> {
  AsyncReturnLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<mlir::async::ReturnOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::async::ReturnOp op, OneToNOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return mlir::failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    llvm::SmallVector<mlir::Value> operands;
    for (auto [original, converted] :
         llvm::zip(op.getOperands(), adaptor.getOperands())) {
      mlir::FailureOr<llvm::SmallVector<mlir::Value>> prepared =
          prepareAsyncReturnPayload(op.getLoc(), original.getType(), converted,
                                    module, *typeConverter, rewriter);
      if (mlir::failed(prepared))
        return mlir::failure();
      mlir::Value packed = packAsyncPayload(
          op.getLoc(), original.getType(), *prepared, *typeConverter, rewriter);
      if (!packed)
        return mlir::failure();
      operands.push_back(packed);
    }
    rewriter.replaceOpWithNewOp<mlir::async::ReturnOp>(op, operands);
    return mlir::success();
  }
};

struct TaskCreateLowering : public mlir::ConversionPattern {
  TaskCreateLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::ConversionPattern(converter, TaskCreateOp::getOperationName(),
                                mlir::PatternBenefit(1), ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *operation,
                  llvm::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<mlir::ValueRange> ranges{mlir::ValueRange{operands}};
    return matchAndRewrite(operation, llvm::ArrayRef<mlir::ValueRange>{ranges},
                           rewriter);
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *operation,
                  llvm::ArrayRef<mlir::ValueRange> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto op = mlir::cast<TaskCreateOp>(operation);
    if (operands.size() != 1)
      return rewriter.notifyMatchFailure(
          operation, "task.create expected one operand range");
    llvm::SmallVector<mlir::Value> coroutineStorage;
    mlir::ValueRange coroutine =
        expandConvertedAsyncDescriptor(operands.front(), coroutineStorage);
    if (coroutine.size() != 2)
      return rewriter.notifyMatchFailure(
          operation, "task.create expected one converted coroutine");

    auto taskType = mlir::dyn_cast<TaskType>(op.getResult().getType());
    if (!taskType)
      return rewriter.notifyMatchFailure(operation,
                                         "task.create result is not !py.task");

    auto flagType = mlir::MemRefType::get({1}, rewriter.getI8Type());
    auto cancelFlag =
        rewriter.create<mlir::memref::AllocOp>(op.getLoc(), flagType);
    async_runtime::CancelFlag::mark(cancelFlag.getResult());
    rewriter.create<mlir::memref::StoreOp>(
        op.getLoc(),
        rewriter.create<mlir::arith::ConstantIntOp>(op.getLoc(), 0, 8),
        cancelFlag, createIndexConstant(op.getLoc(), rewriter, 0));

    llvm::SmallVector<mlir::Value> descriptor{coroutine.front(), coroutine[1],
                                              cancelFlag.getResult()};
    llvm::SmallVector<mlir::ValueRange> replacements{
        mlir::ValueRange{descriptor}};
    rewriter.replaceOpWithMultiple(op, replacements);
    return mlir::success();
  }
};

struct TaskCancelLowering : public mlir::OpConversionPattern<TaskCancelOp> {
  TaskCancelLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<TaskCancelOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(TaskCancelOp op, OneToNOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return mlir::failure();
    mlir::ValueRange task = adaptor.getTask();
    llvm::SmallVector<mlir::Value> taskStorage;
    task = expandConvertedAsyncDescriptor(task, taskStorage);
    if (task.size() != 3)
      return mlir::failure();

    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    mlir::Type resultType =
        typeConverter->convertType(op.getAccepted().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "failed to convert bool type");

    auto alreadyError = rewriter.create<mlir::async::RuntimeIsErrorOp>(
        op.getLoc(), rewriter.getI1Type(), task.front());
    mlir::Value trueValue = rewriter.create<mlir::arith::ConstantOp>(
        op.getLoc(), rewriter.getBoolAttr(true));
    mlir::Value notAlreadyError = rewriter.create<mlir::arith::XOrIOp>(
        op.getLoc(), alreadyError.getIsError(), trueValue);
    mlir::Value requested =
        rewriter.create<mlir::arith::ConstantIntOp>(op.getLoc(), 1, 8);
    mlir::Value previousRequest = rewriter.create<mlir::memref::AtomicRMWOp>(
        op.getLoc(), mlir::arith::AtomicRMWKind::maxu, requested, task[2],
        mlir::ValueRange{createIndexConstant(op.getLoc(), rewriter, 0)});
    threadsafe::Atomic::set(previousRequest.getDefiningOp(),
                            ThreadSafetyAttrs::kRoleAsyncCancelRequest,
                            ThreadSafetyAttrs::kOrderingAcqRel);
    mlir::Value zero =
        rewriter.create<mlir::arith::ConstantIntOp>(op.getLoc(), 0, 8);
    mlir::Value notAlreadyRequested = rewriter.create<mlir::arith::CmpIOp>(
        op.getLoc(), mlir::arith::CmpIPredicate::eq, previousRequest, zero);
    mlir::Value accepted = rewriter.create<mlir::arith::AndIOp>(
        op.getLoc(), notAlreadyError, notAlreadyRequested);

    mlir::Block *currentBlock = rewriter.getInsertionBlock();
    mlir::Block *afterStore =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    mlir::Block *storeBlock = rewriter.createBlock(afterStore->getParent(),
                                                   afterStore->getIterator());
    rewriter.setInsertionPointToEnd(currentBlock);
    rewriter.create<mlir::cf::CondBranchOp>(op.getLoc(), accepted, storeBlock,
                                            afterStore);

    rewriter.setInsertionPointToStart(storeBlock);
    emitStoreCancellationException(op.getLoc(), module, rewriter,
                                   *typeConverter, task[1]);
    rewriter.create<mlir::cf::BranchOp>(op.getLoc(), afterStore);

    rewriter.setInsertionPoint(op);
    llvm::SmallVector<mlir::Type> argTypes{rewriter.getI1Type()};
    auto callee = getOrInsertRuntimeFunc(op.getLoc(), module, rewriter,
                                         RuntimeSymbols::kBoolFromBool,
                                         resultType, argTypes);
    auto call = rewriter.create<mlir::LLVM::CallOp>(
        op.getLoc(), mlir::TypeRange{resultType},
        mlir::SymbolRefAttr::get(callee), mlir::ValueRange{accepted});
    rewriter.replaceOp(op, call.getResults());
    return mlir::success();
  }
};

struct AsyncSleepLowering : public mlir::OpConversionPattern<AsyncSleepOp> {
  AsyncSleepLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<AsyncSleepOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(AsyncSleepOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return mlir::failure();

    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    llvm::SmallVector<mlir::Type> futureTypes;
    if (mlir::failed(typeConverter->convertType(op.getResult().getType(),
                                                futureTypes)) ||
        futureTypes.size() != 2)
      return rewriter.notifyMatchFailure(op, "failed to convert future type");
    auto asyncValueType =
        mlir::dyn_cast<mlir::async::ValueType>(futureTypes.front());
    if (!asyncValueType)
      return rewriter.notifyMatchFailure(op, "future did not convert to value");

    mlir::Value storage = rewriter.create<mlir::async::RuntimeCreateOp>(
        op.getLoc(), futureTypes.front());
    mlir::Value exceptionCell =
        createExceptionCell(op.getLoc(), module, rewriter, *typeConverter);
    mlir::Type payloadType = asyncValueType.getValueType();
    llvm::SmallVector<mlir::Type> argTypes;
    auto callee =
        getOrInsertRuntimeFunc(op.getLoc(), module, rewriter,
                               RuntimeSymbols::kGetNone, payloadType, argTypes);
    auto none = rewriter.create<mlir::LLVM::CallOp>(
        op.getLoc(), mlir::TypeRange{payloadType},
        mlir::SymbolRefAttr::get(callee), mlir::ValueRange{});
    rewriter.create<mlir::async::RuntimeStoreOp>(op.getLoc(), none.getResult(),
                                                 storage);
    rewriter.create<mlir::async::RuntimeSetAvailableOp>(op.getLoc(), storage);
    llvm::SmallVector<mlir::Value> descriptor{storage, exceptionCell};
    rewriter.replaceOpWithMultiple(
        op, llvm::ArrayRef<mlir::ValueRange>{mlir::ValueRange{descriptor}});
    return mlir::success();
  }
};

struct AsyncGatherLowering : public mlir::OpConversionPattern<AsyncGatherOp> {
  AsyncGatherLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<AsyncGatherOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(AsyncGatherOp op, OneToNOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return mlir::failure();

    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    auto tupleType = mlir::dyn_cast<TupleType>(op.getResult().getType());
    if (!tupleType)
      return mlir::failure();

    llvm::SmallVector<mlir::Type, 2> resultTypes;
    if (mlir::failed(typeConverter->convertType(op.getResult().getType(),
                                                resultTypes)) ||
        resultTypes.size() != 2)
      return mlir::failure();
    auto headerType = mlir::dyn_cast<mlir::MemRefType>(resultTypes[0]);
    auto itemsType = mlir::dyn_cast<mlir::MemRefType>(resultTypes[1]);
    if (!headerType || !itemsType)
      return mlir::failure();

    llvm::SmallVector<std::pair<mlir::Value, mlir::Value>> awaitables;
    llvm::SmallVector<llvm::SmallVector<mlir::Value>> awaitableDescriptors;
    llvm::SmallVector<mlir::Type> awaitableTypes;
    for (mlir::ValueRange awaitable : adaptor.getAwaitables()) {
      llvm::SmallVector<mlir::Value> awaitableStorage;
      awaitable = expandConvertedAsyncDescriptor(awaitable, awaitableStorage);
      if (awaitable.empty())
        return mlir::failure();
      awaitables.push_back(
          {awaitable.front(), getAwaitableExceptionCell(awaitable)});
      awaitableDescriptors.emplace_back(awaitable.begin(), awaitable.end());
    }
    for (mlir::Value awaitable : op.getAwaitables())
      awaitableTypes.push_back(awaitable.getType());

    mlir::Location loc = op.getLoc();
    bool insideAsyncFunc =
        op->getParentOfType<mlir::async::FuncOp>() != nullptr;
    mlir::Block *topLevelErrorBlock = nullptr;
    llvm::SmallVector<mlir::Value> payloads;
    auto elementTypes = tupleType.getElementTypes();
    if (!insideAsyncFunc) {
      for (auto [awaitable, exceptionCell] : awaitables) {
        (void)exceptionCell;
        rewriter.create<mlir::async::RuntimeAwaitOp>(loc, awaitable);
      }
      mlir::Block *currentBlock = rewriter.getInsertionBlock();
      mlir::Block *continuationBlock =
          rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
      mlir::Block *errorBlock = rewriter.createBlock(
          continuationBlock->getParent(), continuationBlock->getIterator());
      topLevelErrorBlock = errorBlock;

      rewriter.setInsertionPointToStart(errorBlock);
      emitThrowFirstErroredAwaitable(loc, module, rewriter, *typeConverter,
                                     awaitables, awaitableDescriptors,
                                     awaitableTypes);

      rewriter.setInsertionPointToEnd(currentBlock);
      for (auto [awaitable, exceptionCell] : awaitables) {
        (void)exceptionCell;
        auto asyncValueType =
            mlir::dyn_cast<mlir::async::ValueType>(awaitable.getType());
        if (!asyncValueType)
          return rewriter.notifyMatchFailure(
              op, "gather awaitable did not lower to async.value");
        auto childError = rewriter.create<mlir::async::RuntimeIsErrorOp>(
            loc, rewriter.getI1Type(), awaitable);
        mlir::Block *errorTarget = topLevelErrorBlock;
        if (!payloads.empty()) {
          mlir::Block *cleanupBlock = rewriter.createBlock(
              continuationBlock->getParent(), continuationBlock->getIterator());
          rewriter.setInsertionPointToStart(cleanupBlock);
          emitReleaseLoadedPayloads(loc, module, rewriter, *typeConverter,
                                    payloads, elementTypes);
          rewriter.create<mlir::cf::BranchOp>(loc, topLevelErrorBlock);
          errorTarget = cleanupBlock;
        }
        mlir::Block *nextBlock = rewriter.createBlock(
            continuationBlock->getParent(), continuationBlock->getIterator());
        rewriter.setInsertionPointToEnd(currentBlock);
        rewriter.create<mlir::cf::CondBranchOp>(loc, childError.getIsError(),
                                                errorTarget, nextBlock);
        rewriter.setInsertionPointToStart(nextBlock);
        auto loadOp = rewriter.create<mlir::async::RuntimeLoadOp>(
            loc, asyncValueType.getValueType(), awaitable);
        payloads.push_back(loadOp.getResult());
        currentBlock = nextBlock;
        rewriter.setInsertionPointToEnd(currentBlock);
      }
      rewriter.create<mlir::cf::BranchOp>(loc, continuationBlock);
      rewriter.setInsertionPointToStart(continuationBlock);
    }

    for (auto [index, awaitableAndCell] : llvm::enumerate(awaitables)) {
      auto [awaitable, exceptionCell] = awaitableAndCell;
      if (insideAsyncFunc) {
        if (isKnownAvailableAsyncValueBefore(awaitable, op.getOperation())) {
          auto asyncValueType =
              mlir::dyn_cast<mlir::async::ValueType>(awaitable.getType());
          if (!asyncValueType)
            return rewriter.notifyMatchFailure(
                op, "gather awaitable did not lower to async.value");
          rewriter.create<mlir::async::RuntimeAwaitOp>(loc, awaitable);
          auto loadOp = rewriter.create<mlir::async::RuntimeLoadOp>(
              loc, asyncValueType.getValueType(), awaitable);
          payloads.push_back(loadOp.getResult());
          continue;
        }

        auto awaitOp = rewriter.create<mlir::async::AwaitOp>(loc, awaitable);
        rewriter.setInsertionPointAfter(awaitOp);
        emitAwaitExceptionMarker(loc, rewriter, module, exceptionCell,
                                 getCurrentAsyncExceptionCell(op));
        emitTaskCancelMarker(loc, rewriter, module, awaitableDescriptors[index],
                             getCurrentAsyncExceptionCell(op));
        payloads.push_back(awaitOp.getResult());
        continue;
      }
    }

    mlir::Value allocSize = createIndexConstant(
        loc, rewriter, static_cast<int64_t>(payloads.size()));
    auto header = rewriter.create<mlir::memref::AllocaOp>(loc, headerType);
    auto items = rewriter.create<mlir::memref::AllocaOp>(
        loc, itemsType, mlir::ValueRange{allocSize});
    std::string descriptorGroup =
        container::descriptor::Group::make(op.getOperation(), "tuple");
    container::descriptor::Component::mark(
        header.getResult(), descriptorGroup,
        ContainerSafetyAttrs::kComponentHeader);
    container::descriptor::Component::mark(
        items.getResult(), descriptorGroup,
        ContainerSafetyAttrs::kComponentItems);
    rewriter.create<mlir::memref::StoreOp>(
        loc,
        createI64Constant(loc, rewriter, static_cast<int64_t>(payloads.size())),
        header, createIndexConstant(loc, rewriter, 0));
    for (int64_t slot = 1; slot < kTupleHeaderSize; ++slot) {
      rewriter.create<mlir::memref::StoreOp>(
          loc, createI64Constant(loc, rewriter, 0), header,
          createIndexConstant(loc, rewriter, slot));
    }

    for (auto [index, payload] : llvm::enumerate(payloads)) {
      mlir::Value stored = Slot::storage(loc, payload, elementTypes[index],
                                         module, rewriter, *typeConverter);
      if (!stored)
        return mlir::failure();
      auto store = rewriter.create<mlir::memref::StoreOp>(
          loc, stored, items,
          createIndexConstant(loc, rewriter, static_cast<int64_t>(index)));
      Slot::markTransfer(store.getOperation());
      Slot::releaseSource(loc, payload, elementTypes[index], module, rewriter,
                          *typeConverter);
    }
    for (auto [index, awaitableAndCell] : llvm::enumerate(awaitables)) {
      auto [_, exceptionCell] = awaitableAndCell;
      emitDestroyExceptionCell(loc, module, rewriter, *typeConverter,
                               exceptionCell);
      if (mlir::isa<TaskType>(op.getAwaitables()[index].getType()))
        emitDestroyTaskCancelFlag(loc, awaitableDescriptors[index], rewriter);
      emitDropAsyncHandle(loc, awaitableAndCell.first, rewriter);
    }

    rewriter.replaceOpWithMultiple(
        op, llvm::ArrayRef<mlir::ValueRange>{
                mlir::ValueRange{header.getResult(), items.getResult()}});
    return mlir::success();
  }
};

} // namespace

namespace lowering::async_runtime::Patterns {
void populate(PyLLVMTypeConverter &typeConverter,
              mlir::RewritePatternSet &patterns) {
  auto *ctx = patterns.getContext();
  patterns.add<AsyncFuncSignatureLowering, CoroCreateLowering,
               CoroStartLowering, AsyncCallTypeLowering, AwaitLowering,
               AsyncReturnLowering, TaskCreateLowering, TaskCancelLowering,
               AsyncSleepLowering, AsyncGatherLowering>(typeConverter, ctx);
}
} // namespace lowering::async_runtime::Patterns

} // namespace py
