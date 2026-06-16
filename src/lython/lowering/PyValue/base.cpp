#include "Common/Object.h"
#include "Common/RuntimeSupport.h"
#include "Passes/OwnershipAnalysis.h"
#include "cpp/PyTypeObject.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include <cerrno>
#include <cstdint>
#include <cstdlib>

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

namespace py {
namespace {

struct BuiltinExceptionClass {
  llvm::StringLiteral name;
  std::int64_t id;
};

static constexpr BuiltinExceptionClass kBuiltinExceptionClasses[] = {
    {type_object::kBaseException, 1},
    {type_object::kException, 2},
    {"RuntimeError", 3},
    {"TypeError", 4},
    {"ValueError", 5},
    {"KeyError", 6},
    {"IndexError", 7},
    {"AssertionError", 8},
    {"StopIteration", 9},
    {"StopAsyncIteration", 10},
};

static mlir::FailureOr<std::int64_t>
builtinExceptionClassId(mlir::Operation *op, llvm::StringRef className) {
  for (const BuiltinExceptionClass &entry : kBuiltinExceptionClasses)
    if (entry.name == className)
      return entry.id;
  op->emitOpError("unsupported builtin exception class '") << className << "'";
  return mlir::failure();
}

static mlir::FailureOr<llvm::SmallVector<std::int64_t, 8>>
matchingBuiltinExceptionClassIds(mlir::Operation *op,
                                 llvm::StringRef handlerName) {
  llvm::SmallVector<std::int64_t, 8> ids;
  for (const BuiltinExceptionClass &entry : kBuiltinExceptionClasses) {
    mlir::FailureOr<bool> matches =
        type_object::isSubclassOf(op, entry.name, handlerName);
    if (mlir::failed(matches))
      return mlir::failure();
    if (*matches)
      ids.push_back(entry.id);
  }
  return ids;
}

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
  return async_runtime::ExceptionCell::hasProvenance(value);
}

static bool isLoweredExceptionCellCarrierType(mlir::Type type) {
  return async_runtime::isExceptionCellType(type) ||
         async_runtime::isLoweredExceptionCellType(type);
}

static mlir::Value trailingExceptionCell(mlir::Region &body,
                                         bool allowAsyncSignatureCarrier) {
  if (body.empty())
    return {};
  mlir::Block &entry = body.front();
  if (entry.getNumArguments() == 0)
    return {};
  mlir::Value candidate = entry.getArgument(entry.getNumArguments() - 1);
  if (isExceptionCell(candidate))
    return candidate;
  if (allowAsyncSignatureCarrier &&
      isLoweredExceptionCellCarrierType(candidate.getType()))
    return candidate;
  return {};
}

static mlir::Value getCurrentAsyncExceptionCell(mlir::Operation *op) {
  if (auto asyncFunc = op ? op->getParentOfType<mlir::async::FuncOp>()
                          : mlir::async::FuncOp()) {
    mlir::Value cell =
        trailingExceptionCell(asyncFunc.getBody(),
                              /*allowAsyncSignatureCarrier=*/true);
    if (cell && !async_runtime::ExceptionCell::hasProvenance(cell)) {
      mlir::Block &entry = asyncFunc.getBody().front();
      async_runtime::ExceptionCell::markArgument(asyncFunc.getOperation(),
                                                 entry.getNumArguments() - 1);
    }
    return cell;
  }

  auto function = op ? op->getParentOfType<mlir::FunctionOpInterface>()
                     : mlir::FunctionOpInterface();
  if (!function || function.getFunctionBody().empty())
    return {};
  return trailingExceptionCell(function.getFunctionBody(),
                               /*allowAsyncSignatureCarrier=*/false);
}

static mlir::LogicalResult
storeExceptionCell(mlir::Location loc, mlir::Value cell, mlir::Value exception,
                   mlir::ModuleOp module, mlir::PatternRewriter &rewriter,
                   const PyLLVMTypeConverter &typeConverter) {
  if (!isExceptionCell(cell))
    return mlir::success();
  return async_runtime::ExceptionCell::storeFirst(loc, cell, exception, module,
                                                  rewriter, typeConverter);
}

static mlir::Value loadExceptionCell(mlir::Location loc, mlir::Value cell,
                                     mlir::RewriterBase &rewriter) {
  return async_runtime::ExceptionCell::load(loc, cell, rewriter);
}

static mlir::LLVM::LLVMStructType
loweredExceptionDescriptorType(mlir::MLIRContext *ctx) {
  return object_abi::Type::loweredStorage(ctx);
}

static mlir::LLVM::LLVMStructType
loweredExceptionPartsDescriptorType(mlir::MLIRContext *ctx) {
  auto descriptor = loweredExceptionDescriptorType(ctx);
  return mlir::LLVM::LLVMStructType::getLiteral(
      ctx, llvm::ArrayRef<mlir::Type>{descriptor, descriptor, descriptor});
}

static mlir::Value
packExceptionPartsForAsyncCell(mlir::Location loc, mlir::ValueRange parts,
                               mlir::PatternRewriter &rewriter) {
  if (parts.size() != 3)
    return {};
  return rewriter
      .create<mlir::UnrealizedConversionCastOp>(
          loc,
          mlir::TypeRange{
              loweredExceptionPartsDescriptorType(rewriter.getContext())},
          parts)
      .getResult(0);
}

static void markEHExceptionPart(mlir::Value value, unsigned index) {
  ownership::aggregate::Slot::markLoad(value, "eh.current_exception",
                                       "exception", index);
}

static mlir::Value captureLandingpadException(
    mlir::Location loc, mlir::LLVM::LandingpadOp landingpad,
    mlir::Type targetType, mlir::ModuleOp module,
    mlir::PatternRewriter &rewriter, const PyLLVMTypeConverter &typeConverter) {
  if (!landingpad)
    return {};
  auto ptrType = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());

  auto partsType = loweredExceptionPartsDescriptorType(rewriter.getContext());
  auto i64Type = rewriter.getI64Type();
  mlir::Value one = rewriter.create<mlir::LLVM::ConstantOp>(
      loc, i64Type, rewriter.getI64IntegerAttr(1));
  mlir::Value descriptorStorage = rewriter.create<mlir::LLVM::AllocaOp>(
      loc, ptrType, partsType, one, /*alignment=*/0);
  ownership::Pointer::markNonObject(descriptorStorage);
  RuntimeAPI runtime(module, rewriter, typeConverter);
  mlir::Value captured =
      runtime
          .call(loc, RuntimeSymbols::kEHTakeCurrentDescriptor,
                rewriter.getI1Type(), mlir::ValueRange{descriptorStorage})
          .getResult();
  rewriter.create<mlir::cf::AssertOp>(
      loc, captured, "current exception is not a parts descriptor");

  mlir::Value split =
      rewriter.create<mlir::LLVM::LoadOp>(loc, partsType, descriptorStorage);
  ownership::aggregate::Slot::markLoad(split, "eh.current_exception",
                                       "exception", std::nullopt);
  for (unsigned index = 0; index < 3; ++index) {
    auto descriptorType =
        mlir::cast<mlir::LLVM::LLVMStructType>(partsType.getBody()[index]);
    mlir::Value descriptor = rewriter.create<mlir::LLVM::ExtractValueOp>(
        loc, descriptorType, split,
        rewriter.getDenseI64ArrayAttr({static_cast<int64_t>(index)}));
    markEHExceptionPart(descriptor, index);
  }
  return rewriter
      .create<mlir::UnrealizedConversionCastOp>(
          loc, mlir::TypeRange{targetType}, split)
      .getResult(0);
}

static mlir::LogicalResult
copyExceptionCell(mlir::Location loc, mlir::Value sourceCell,
                  mlir::Value destCell, mlir::ModuleOp module,
                  mlir::PatternRewriter &rewriter,
                  const PyLLVMTypeConverter &typeConverter) {
  if (!isExceptionCell(sourceCell) || !isExceptionCell(destCell) ||
      sourceCell == destCell)
    return mlir::success();
  mlir::Value exception = loadExceptionCell(loc, sourceCell, rewriter);
  return async_runtime::ExceptionCell::storeFirst(
      loc, destCell, exception, module, rewriter, typeConverter,
      ThreadSafetyAttrs::kPremiseAggregateBorrow);
}

static void consumeAwaitedDescriptor(mlir::Location loc, mlir::Value awaitable,
                                     mlir::PatternRewriter &rewriter) {
  auto dec = rewriter.create<DecRefOp>(loc, awaitable);
  dec->setAttr("ly.async.await_consumed_descriptor", rewriter.getUnitAttr());
}

static mlir::LogicalResult consumeAwaitedDescriptorWithLoadedException(
    mlir::Location loc, mlir::Value awaitable, mlir::ValueRange descriptor,
    mlir::Type awaitableType, mlir::Value loadedException,
    mlir::ModuleOp module, mlir::PatternRewriter &rewriter,
    const PyLLVMTypeConverter &typeConverter) {
  (void)awaitableType;
  if (descriptor.size() >= 2 && isExceptionCell(descriptor[1])) {
    mlir::Value exception =
        loadedException ? loadedException
                        : loadExceptionCell(loc, descriptor[1], rewriter);
    if (mlir::failed(async_runtime::ExceptionCell::releaseLoaded(
            loc, module, rewriter, typeConverter, exception)))
      return mlir::failure();
    if (mlir::failed(async_runtime::ExceptionCell::free(
            loc, module, rewriter, typeConverter, descriptor[1])))
      return mlir::failure();
  }

  if (!descriptor.empty() &&
      mlir::isa<mlir::async::ValueType>(descriptor.front().getType()))
    rewriter.create<mlir::async::RuntimeDropRefOp>(
        loc, descriptor.front(), rewriter.getI64IntegerAttr(1));

  auto witness = rewriter.create<DecRefOp>(loc, awaitable);
  witness->setAttr("ly.ownership.lowered_witness", rewriter.getUnitAttr());
  return mlir::success();
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

static llvm::SmallVector<mlir::Type> blockArgumentTypes(mlir::Block *block) {
  llvm::SmallVector<mlir::Type> types;
  if (!block)
    return types;
  for (mlir::BlockArgument argument : block->getArguments())
    types.push_back(argument.getType());
  return types;
}

static mlir::FailureOr<llvm::SmallVector<mlir::Value>>
convertLogicalValueForBlock(mlir::Location loc, mlir::Value value,
                            mlir::Block *dest,
                            mlir::PatternRewriter &rewriter) {
  llvm::SmallVector<mlir::Type> destTypes = blockArgumentTypes(dest);
  if (destTypes.empty())
    return llvm::SmallVector<mlir::Value>{};
  if (destTypes.size() == 1 && value.getType() == destTypes.front())
    return llvm::SmallVector<mlir::Value>{value};
  auto cast = rewriter.create<mlir::UnrealizedConversionCastOp>(
      loc, mlir::TypeRange(destTypes), value);
  llvm::SmallVector<mlir::Value> results;
  results.append(cast.getResults().begin(), cast.getResults().end());
  return results;
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
    mlir::PatternRewriter &rewriter, mlir::ValueRange cleanupPayloads = {}) {
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

  mlir::Block *currentBlock = rewriter.getInsertionBlock();
  mlir::Block *awaitBlock =
      rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
  mlir::Block *errorBlock =
      rewriter.createBlock(awaitBlock->getParent(), awaitBlock->getIterator());
  rewriter.setInsertionPointToEnd(currentBlock);
  rewriter.create<mlir::cf::CondBranchOp>(awaitOp.getLoc(), shouldRunFinally,
                                          errorBlock, awaitBlock);

  rewriter.setInsertionPointToStart(errorBlock);
  if (mlir::failed(copyExceptionCell(
          awaitOp.getLoc(), awaitableExceptionCell, currentExceptionCell,
          awaitOp->getParentOfType<mlir::ModuleOp>(), rewriter, typeConverter)))
    return mlir::failure();
  consumeAwaitedDescriptor(awaitOp.getLoc(), awaitable, rewriter);
  for (mlir::Value payload : cleanupPayloads)
    if (isPyOwnershipTrackedType(payload.getType()))
      rewriter.create<DecRefOp>(awaitOp.getLoc(), payload);
  cloneFinallyBody(tryOp, rewriter, markerId);
  mlir::Value falseValue = rewriter.create<mlir::arith::ConstantOp>(
      awaitOp.getLoc(), rewriter.getBoolAttr(false));
  rewriter.create<mlir::cf::AssertOp>(awaitOp.getLoc(), falseValue,
                                      "awaitable failed");
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
                      mlir::ValueRange cleanupPayloads = {}) {
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
  auto retain = rewriter.create<IncRefOp>(awaitOp.getLoc(), exception);
  if (loadedException)
    threadsafe::Retain::premise(retain.getOperation(),
                                ThreadSafetyAttrs::kPremiseAggregateBorrow);
  mlir::FailureOr<llvm::SmallVector<mlir::Value>> branchArgs =
      convertLogicalValueForBlock(awaitOp.getLoc(), exception, exceptEntry,
                                  rewriter);
  if (mlir::failed(branchArgs))
    return mlir::failure();
  if (mlir::failed(consumeAwaitedDescriptorWithLoadedException(
          awaitOp.getLoc(), awaitable, convertedAwaitableParts,
          awaitable.getType(), loadedException,
          awaitOp->getParentOfType<mlir::ModuleOp>(), rewriter, typeConverter)))
    return mlir::failure();
  for (mlir::Value payload : cleanupPayloads)
    if (isPyOwnershipTrackedType(payload.getType()))
      rewriter.create<DecRefOp>(awaitOp.getLoc(), payload);
  rewriter.create<mlir::cf::BranchOp>(awaitOp.getLoc(), exceptEntry,
                                      mlir::ValueRange{*branchArgs});

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
  llvm::SmallVector<mlir::Type> convertedTypes;
  if (mlir::failed(
          typeConverter.convertType(oldArg.getType(), convertedTypes)) ||
      convertedTypes.empty())
    return mlir::failure();
  if (convertedTypes.size() == 1 && convertedTypes.front() == oldArg.getType())
    return oldArg;

  llvm::SmallVector<mlir::BlockArgument> convertedArgs;
  for (mlir::Type convertedType : convertedTypes)
    convertedArgs.push_back(
        exceptEntry->addArgument(convertedType, oldArg.getLoc()));
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(exceptEntry);
  llvm::SmallVector<mlir::Value> convertedValues;
  for (mlir::BlockArgument argument : convertedArgs)
    convertedValues.push_back(argument);
  mlir::Value logicalArg =
      rewriter
          .create<mlir::UnrealizedConversionCastOp>(
              oldArg.getLoc(), mlir::TypeRange{oldArg.getType()},
              convertedValues)
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

static void erasePrecedingExceptionDrops(mlir::Value exception,
                                         mlir::Operation *anchor,
                                         mlir::PatternRewriter &rewriter) {
  if (!exception || !anchor)
    return;
  llvm::SmallVector<mlir::Operation *> drops;
  for (mlir::OpOperand &use : llvm::make_early_inc_range(exception.getUses())) {
    mlir::Operation *user = use.getOwner();
    if (!mlir::isa<DecRefOp>(user))
      continue;
    if (user->getBlock() != anchor->getBlock())
      continue;
    if (!user->isBeforeInBlock(anchor))
      continue;
    drops.push_back(user);
  }
  for (mlir::Operation *drop : drops)
    rewriter.eraseOp(drop);
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
    erasePrecedingExceptionDrops(exception, raiseCurrent, rewriter);
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

static bool isDropOnlyExceptionUse(mlir::Operation *user) {
  return mlir::isa<DecRefOp>(user);
}

static bool caughtExceptionOnlyDropped(mlir::Block *exceptEntry) {
  if (!exceptEntry || exceptEntry->getNumArguments() != 1)
    return false;
  mlir::BlockArgument exception = exceptEntry->getArgument(0);
  return llvm::all_of(exception.getUsers(), isDropOnlyExceptionUse);
}

static bool hasRaiseCurrent(llvm::ArrayRef<mlir::Block *> blocks) {
  for (mlir::Block *block : blocks) {
    if (block
            ->walk([](RaiseCurrentOp) { return mlir::WalkResult::interrupt(); })
            .wasInterrupted())
      return true;
  }
  return false;
}

static bool hasNestedTry(TryOp op) {
  bool nested = false;
  for (mlir::Region &region : op->getRegions()) {
    region.walk([&](TryOp candidate) {
      if (candidate != op)
        nested = true;
    });
    if (nested)
      return true;
  }
  return false;
}

static void eraseDropUsers(mlir::Value value, mlir::PatternRewriter &rewriter) {
  llvm::SmallVector<mlir::Operation *> drops;
  for (mlir::OpOperand &use : llvm::make_early_inc_range(value.getUses())) {
    mlir::Operation *user = use.getOwner();
    if (isDropOnlyExceptionUse(user))
      drops.push_back(user);
  }
  llvm::SmallPtrSet<mlir::Operation *, 8> seen;
  for (mlir::Operation *drop : drops) {
    if (!seen.insert(drop).second)
      continue;
    if (!drop->hasTrait<mlir::OpTrait::IsTerminator>())
      rewriter.eraseOp(drop);
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

struct ExceptionNullLowering
    : public mlir::OpConversionPattern<ExceptionNullOp> {
  ExceptionNullLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<ExceptionNullOp>(converter, ctx) {}

  static mlir::Value zeroMemRef(mlir::Location loc, mlir::MemRefType type,
                                mlir::ConversionPatternRewriter &rewriter) {
    if (!type.hasStaticShape() || type.getRank() != 1)
      return {};
    mlir::Value storage = rewriter.create<mlir::memref::AllocaOp>(loc, type);
    mlir::Type elementType = type.getElementType();
    mlir::Value zero;
    if (auto intType = mlir::dyn_cast<mlir::IntegerType>(elementType)) {
      zero = rewriter.create<mlir::arith::ConstantIntOp>(loc, 0,
                                                         intType.getWidth());
    } else {
      return {};
    }
    for (int64_t index = 0; index < type.getShape().front(); ++index) {
      mlir::Value iv =
          rewriter.create<mlir::arith::ConstantIndexOp>(loc, index);
      rewriter.create<mlir::memref::StoreOp>(loc, zero, storage,
                                             mlir::ValueRange{iv});
    }
    return storage;
  }

  mlir::LogicalResult
  matchAndRewrite(ExceptionNullOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    llvm::SmallVector<mlir::Type> resultTypes;
    if (mlir::failed(
            typeConverter->convertType(op.getResult().getType(), resultTypes)))
      return rewriter.notifyMatchFailure(op, "failed to convert exception");
    llvm::SmallVector<mlir::Value> results;
    for (mlir::Type resultType : resultTypes) {
      if (auto memref = mlir::dyn_cast<mlir::MemRefType>(resultType)) {
        mlir::Value zero = zeroMemRef(op.getLoc(), memref, rewriter);
        if (!zero)
          return rewriter.notifyMatchFailure(
              op, "exception null requires static integer memref storage");
        results.push_back(zero);
        continue;
      }
      auto structType = mlir::dyn_cast<mlir::LLVM::LLVMStructType>(resultType);
      if (!structType ||
          (!object_abi::Type::isLoweredStorage(structType) &&
           structType != loweredExceptionPartsDescriptorType(op.getContext())))
        return rewriter.notifyMatchFailure(
            op, "exception null requires lowered exception descriptor ABI");
      results.push_back(
          rewriter.create<mlir::LLVM::ZeroOp>(op.getLoc(), resultType));
    }
    rewriter.replaceOp(op, results);
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
    if (auto memref = mlir::dyn_cast<mlir::MemRefType>(resultType)) {
      mlir::Value zero =
          ExceptionNullLowering::zeroMemRef(op.getLoc(), memref, rewriter);
      if (!zero)
        return rewriter.notifyMatchFailure(
            op, "traceback null requires static integer memref storage");
      rewriter.replaceOp(op, zero);
      return mlir::success();
    }
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
    llvm::SmallVector<mlir::Type> resultTypes;
    if (mlir::failed(
            typeConverter->convertType(op.getResult().getType(), resultTypes)))
      return rewriter.notifyMatchFailure(op, "failed to convert exception");
    if (resultTypes.size() != 3)
      return rewriter.notifyMatchFailure(
          op, "exception split ABI requires header and unicode message");

    llvm::SmallVector<mlir::Value, 2> messageParts;
    if (adaptor.getArgs().size() > 1)
      return rewriter.notifyMatchFailure(op, "too many exception arguments");
    if (adaptor.getArgs().empty()) {
      mlir::Value bytes =
          runtime.getByteLiteral(op.getLoc(), rewriter.getStringAttr(""));
      mlir::Value start =
          rewriter.create<mlir::arith::ConstantIndexOp>(op.getLoc(), 0);
      mlir::Value length = runtime.getI64Constant(op.getLoc(), 0);
      llvm::SmallVector<mlir::Type, 2> unicodeTypes;
      object_abi::str_abi::Parts::storageTypes(rewriter.getContext(),
                                               unicodeTypes);
      auto message =
          runtime.call(op.getLoc(), RuntimeSymbols::kUnicodeFromBytes,
                       mlir::TypeRange(unicodeTypes),
                       mlir::ValueRange{bytes, start, length});
      messageParts.append(message.getResults().begin(),
                          message.getResults().end());
    } else {
      mlir::ValueRange message = adaptor.getArgs().front();
      messageParts.append(message.begin(), message.end());
    }
    if (messageParts.size() != 2)
      return rewriter.notifyMatchFailure(op,
                                         "message must lower to unicode parts");

    llvm::StringRef className = type_object::kException;
    if (auto classAttr =
            op->getAttrOfType<mlir::StringAttr>("py.exception.class"))
      className = classAttr.getValue();
    mlir::FailureOr<std::int64_t> classId =
        builtinExceptionClassId(op.getOperation(), className);
    if (mlir::failed(classId))
      return mlir::failure();
    mlir::Value classIdValue = runtime.getI64Constant(op.getLoc(), *classId);
    auto storage = runtime.call(op.getLoc(), RuntimeSymbols::kExceptionNew,
                                mlir::TypeRange{resultTypes.front()},
                                mlir::ValueRange{classIdValue});
    llvm::SmallVector<mlir::Value, 3> exceptionParts;
    exceptionParts.push_back(storage.getResult());
    exceptionParts.append(messageParts.begin(), messageParts.end());
    rewriter.replaceOpWithMultiple(
        op, llvm::ArrayRef<mlir::ValueRange>{mlir::ValueRange{exceptionParts}});
    return mlir::success();
  }
};

struct ExceptMatchLowering : public mlir::OpConversionPattern<ExceptMatchOp> {
  ExceptMatchLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<ExceptMatchOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(ExceptMatchOp op, OneToNOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto handlerType =
        mlir::dyn_cast<ClassType>(op.getHandlerAttr().getValue());
    if (!handlerType)
      return rewriter.notifyMatchFailure(op, "handler is not a class type");

    mlir::ValueRange exceptionParts = adaptor.getException();
    if (exceptionParts.size() != 3)
      return rewriter.notifyMatchFailure(
          op, "except.match requires exception parts descriptor ABI");
    mlir::Value header = exceptionParts.front();
    if (!object_abi::exception_abi::Header::isOwned(header.getType()))
      return rewriter.notifyMatchFailure(
          op, "except.match requires exception header descriptor");

    mlir::FailureOr<llvm::SmallVector<std::int64_t, 8>> matchingIds =
        matchingBuiltinExceptionClassIds(op.getOperation(),
                                         handlerType.getClassName());
    if (mlir::failed(matchingIds))
      return mlir::failure();
    mlir::Value classIdSlot = rewriter.create<mlir::arith::ConstantIndexOp>(
        op.getLoc(), object_abi::exception_abi::kClassIdSlot);
    mlir::Value classId =
        rewriter.create<mlir::memref::LoadOp>(op.getLoc(), header, classIdSlot);
    mlir::Value result =
        rewriter.create<mlir::arith::ConstantIntOp>(op.getLoc(), 0, 1);
    for (std::int64_t id : *matchingIds) {
      mlir::Value expected =
          rewriter.create<mlir::arith::ConstantIntOp>(op.getLoc(), id, 64);
      mlir::Value match = rewriter.create<mlir::arith::CmpIOp>(
          op.getLoc(), mlir::arith::CmpIPredicate::eq, classId, expected);
      result = rewriter.create<mlir::arith::OrIOp>(op.getLoc(), result, match);
    }
    rewriter.replaceOp(op, result);
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
    if (hasNestedTry(op))
      return rewriter.notifyMatchFailure(op, "nested py.try must lower first");

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
    bool exceptDropsCaughtException = hasExcept &&
                                      caughtExceptionOnlyDropped(exceptEntry) &&
                                      !hasRaiseCurrent(exceptBlocks);

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
      if (exceptDropsCaughtException)
        eraseDropUsers(exceptEntry->getArgument(0), rewriter);
      mlir::FailureOr<mlir::Value> caughtException =
          convertExceptEntryArgument(exceptEntry, typeConverter, rewriter);
      if (mlir::failed(caughtException))
        return mlir::failure();
      replaceRaiseCurrentInExceptBlocks(exceptBlocks, *caughtException,
                                        rewriter);
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
          mlir::Value excNull;
          excNull = builder.create<ExceptionNullOp>(
              invoke.getLoc(), ExceptionType::get(op.getContext()));
          llvm::SmallVector<mlir::Type> exceptArgTypes =
              blockArgumentTypes(exceptEntry);
          auto cast = builder.create<mlir::UnrealizedConversionCastOp>(
              invoke.getLoc(), mlir::TypeRange(exceptArgTypes), excNull);
          invoke.getUnwindDestOperandsMutable().assign(cast.getResults());
          invoke->setSuccessor(exceptEntry, 1);
        }
        if (auto raise = mlir::dyn_cast<RaiseOp>(block->getTerminator())) {
          rewriter.setInsertionPoint(raise);
          mlir::Value raisedException = raise.getException();
          mlir::Value exception = raisedException;
          if (exceptDropsCaughtException) {
            exception = rewriter.create<ExceptionNullOp>(
                raise.getLoc(), ExceptionType::get(op.getContext()));
          }
          mlir::FailureOr<llvm::SmallVector<mlir::Value>> branchArgs =
              convertLogicalValueForBlock(raise.getLoc(), exception,
                                          exceptEntry, rewriter);
          if (mlir::failed(branchArgs))
            return mlir::failure();
          if (exceptDropsCaughtException &&
              isPyOwnershipTrackedType(raisedException.getType()))
            rewriter.create<DecRefOp>(raise.getLoc(), raisedException);
          eraseOpsAfter(raise, rewriter);
          raise->setAttr("ly.redirected_to_except", rewriter.getUnitAttr());
          rewriter.create<mlir::cf::BranchOp>(raise.getLoc(), exceptEntry,
                                              mlir::ValueRange{*branchArgs});
          rewriter.eraseOp(raise);
        }
        if (auto raiseCurrent =
                mlir::dyn_cast<RaiseCurrentOp>(block->getTerminator())) {
          rewriter.setInsertionPoint(raiseCurrent);
          mlir::Value exception;
          auto landingpad = mlir::dyn_cast_or_null<mlir::LLVM::LandingpadOp>(
              block->empty() ? nullptr : &block->front());
          if (exceptDropsCaughtException) {
            exception = rewriter.create<ExceptionNullOp>(
                raiseCurrent.getLoc(), ExceptionType::get(op.getContext()));
          } else if (landingpad) {
            exception = captureLandingpadException(
                raiseCurrent.getLoc(), landingpad,
                ExceptionType::get(op.getContext()),
                op->getParentOfType<mlir::ModuleOp>(), rewriter, typeConverter);
          } else if (block->getNumArguments() ==
                     exceptEntry->getNumArguments()) {
            llvm::SmallVector<mlir::Value> parts;
            for (mlir::BlockArgument argument : block->getArguments())
              parts.push_back(argument);
            mlir::FailureOr<mlir::Value> logicalException =
                materializeLogicalValue(raiseCurrent.getLoc(),
                                        ExceptionType::get(op.getContext()),
                                        parts, rewriter);
            if (mlir::failed(logicalException))
              return mlir::failure();
            exception = *logicalException;
          } else if (block->getNumArguments() == 1) {
            exception = block->getArgument(0);
          } else {
            return rewriter.notifyMatchFailure(
                op, "raise.current unwind block must carry one exception");
          }
          if (!exception)
            return rewriter.notifyMatchFailure(
                op, "failed to capture current exception");
          erasePrecedingExceptionDrops(exception, raiseCurrent, rewriter);
          mlir::FailureOr<llvm::SmallVector<mlir::Value>> branchArgs =
              convertLogicalValueForBlock(raiseCurrent.getLoc(), exception,
                                          exceptEntry, rewriter);
          if (mlir::failed(branchArgs))
            return mlir::failure();
          eraseOpsAfter(raiseCurrent, rewriter);
          raiseCurrent->setAttr("ly.redirected_to_except",
                                rewriter.getUnitAttr());
          rewriter.create<mlir::cf::BranchOp>(raiseCurrent.getLoc(),
                                              exceptEntry,
                                              mlir::ValueRange{*branchArgs});
          rewriter.eraseOp(raiseCurrent);
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
  matchAndRewrite(RaiseOp op, OneToNOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return mlir::failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *typeConverter);
    mlir::ValueRange exceptionParts = adaptor.getException();
    if (mlir::Value exceptionCell = getCurrentAsyncExceptionCell(op)) {
      mlir::Value exceptionForCell =
          exceptionParts.size() == 3
              ? packExceptionPartsForAsyncCell(op.getLoc(), exceptionParts,
                                               rewriter)
              : mlir::Value();
      if (!exceptionForCell) {
        mlir::FailureOr<mlir::Value> exception = materializeLogicalValue(
            op.getLoc(), op.getException().getType(), exceptionParts, rewriter);
        if (mlir::failed(exception))
          return mlir::failure();
        exceptionForCell = *exception;
      }
      if (mlir::failed(storeExceptionCell(op.getLoc(), exceptionCell,
                                          exceptionForCell, module, rewriter,
                                          *typeConverter)))
        return mlir::failure();
      if (mlir::failed(async_runtime::ExceptionCell::releasePayload(
              op.getLoc(), module, rewriter, *typeConverter, exceptionForCell)))
        return mlir::failure();
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
    mlir::Value fileBytes = runtime.getByteLiteral(op.getLoc(), fileAttr);
    mlir::Value funcBytes = runtime.getByteLiteral(op.getLoc(), funcAttr);
    mlir::Value lineConst = rewriter.create<mlir::arith::ConstantIntOp>(
        op.getLoc(), static_cast<int32_t>(line), 32);
    mlir::Value colConst = rewriter.create<mlir::arith::ConstantIntOp>(
        op.getLoc(), static_cast<int32_t>(col), 32);
    runtime.call(op.getLoc(), RuntimeSymbols::kTracebackPush, mlir::Type(),
                 mlir::ValueRange{fileBytes, funcBytes, lineConst, colConst});
    if (exceptionParts.size() == 3) {
      runtime.call(op.getLoc(), RuntimeSymbols::kEHThrowException, mlir::Type(),
                   exceptionParts);
      rewriter.create<mlir::LLVM::UnreachableOp>(op.getLoc());
      rewriter.eraseOp(op);
      return mlir::success();
    }
    return rewriter.notifyMatchFailure(
        op, "py.raise requires exception parts descriptor ABI");
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
    runtime.call(op.getLoc(), RuntimeSymbols::kEHRethrowCurrent, mlir::Type(),
                 mlir::ValueRange{});
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
               RaiseCurrentLowering, ExceptMatchLowering>(typeConverter, ctx);
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
