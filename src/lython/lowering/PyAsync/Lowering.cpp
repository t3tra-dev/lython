#include "Common/Container.h"
#include "Common/LoweringUtils.h"
#include "Common/Object.h"
#include "Common/RuntimeSupport.h"
#include "Common/SlotUtils.h"
#include "Passes/OwnershipAnalysis.h"
#include "Passes/Runtime/Helpers.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

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

static void markAsyncContainerPayload(mlir::Type logicalType,
                                      mlir::ValueRange values,
                                      mlir::Operation *seed) {
  if (values.empty())
    return;

  llvm::StringRef kind;
  llvm::SmallVector<llvm::StringRef, 4> components;
  if (mlir::isa<TupleType>(logicalType)) {
    kind = ContainerSafetyAttrs::kKindTuple;
    components = {ContainerSafetyAttrs::kComponentHeader,
                  ContainerSafetyAttrs::kComponentItems};
  } else if (mlir::isa<ListType>(logicalType)) {
    kind = ContainerSafetyAttrs::kKindList;
    components = {ContainerSafetyAttrs::kComponentHeader,
                  ContainerSafetyAttrs::kComponentLock,
                  ContainerSafetyAttrs::kComponentItems};
  } else if (mlir::isa<DictType>(logicalType)) {
    kind = ContainerSafetyAttrs::kKindDict;
    components = {ContainerSafetyAttrs::kComponentHeader,
                  ContainerSafetyAttrs::kComponentLock,
                  ContainerSafetyAttrs::kComponentKeys,
                  ContainerSafetyAttrs::kComponentValues,
                  ContainerSafetyAttrs::kComponentStates};
  } else {
    return;
  }

  std::string group = container::descriptor::Group::make(seed, kind);
  for (auto [index, value] : llvm::enumerate(values)) {
    if (index >= components.size())
      break;
    container::descriptor::Component::mark(value, group, components[index]);
  }
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
                                          /*cloneReferenceSlots=*/true);
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
  markAsyncContainerPayload(logicalType, results, cast.getOperation());
  return results;
}

static constexpr llvm::StringLiteral kAsyncExceptionEdgeMarker{
    "__lython_async_exception_edge_marker"};
static constexpr llvm::StringLiteral kAsyncTaskCancelMarker{
    "__lython_async_task_cancel_marker"};

static llvm::StringRef exceptionCellMarkerTypeCode(mlir::Type type) {
  if (async_runtime::isExceptionCellType(type))
    return "memref";
  if (async_runtime::isLoweredExceptionCellType(type))
    return "lowered";
  return "invalid";
}

static std::string asyncExceptionEdgeMarkerName(mlir::Type sourceType,
                                                mlir::Type destType) {
  return (llvm::Twine(kAsyncExceptionEdgeMarker) + "_" +
          exceptionCellMarkerTypeCode(sourceType) + "_" +
          exceptionCellMarkerTypeCode(destType))
      .str();
}

namespace async_descriptor {

mlir::Value asyncValue(mlir::ValueRange descriptor);
mlir::Value exceptionCell(mlir::ValueRange descriptor);
mlir::Value cancelFlag(mlir::ValueRange descriptor);
llvm::SmallVector<mlir::Value>
pack(mlir::Value value, mlir::Value exceptionCell, mlir::Value cancelFlag = {});

} // namespace async_descriptor

static mlir::func::FuncOp getOrInsertAsyncExceptionEdgeMarker(
    mlir::Location loc, mlir::ModuleOp module, mlir::OpBuilder &builder,
    mlir::Type sourceType, mlir::Type destType) {
  std::string markerName = asyncExceptionEdgeMarkerName(sourceType, destType);
  if (auto fn = module.lookupSymbol<mlir::func::FuncOp>(markerName))
    return fn;

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());
  auto fnType =
      builder.getFunctionType(mlir::TypeRange{sourceType, destType}, {});
  auto fn = builder.create<mlir::func::FuncOp>(loc, markerName, fnType);
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
  auto marker = getOrInsertAsyncExceptionEdgeMarker(
      loc, module, rewriter, sourceCell.getType(), destCell.getType());
  rewriter.create<mlir::func::CallOp>(loc, marker.getSymName(),
                                      mlir::TypeRange{},
                                      mlir::ValueRange{sourceCell, destCell});
}

static mlir::func::FuncOp
getOrInsertAsyncTaskCancelMarker(mlir::Location loc, mlir::ModuleOp module,
                                 mlir::OpBuilder &builder, mlir::Type flagType,
                                 mlir::Type cellType) {
  if (auto fn = module.lookupSymbol<mlir::func::FuncOp>(kAsyncTaskCancelMarker))
    return fn;

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());
  auto fnType = builder.getFunctionType(
      mlir::TypeRange{flagType, cellType, cellType}, {});
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
  mlir::Value taskCell = async_descriptor::exceptionCell(taskDescriptor);
  mlir::Value flag = async_descriptor::cancelFlag(taskDescriptor);
  if (taskDescriptor.size() != 3 || !taskCell || !isExceptionCell(destCell) ||
      !flag)
    return;
  auto marker = getOrInsertAsyncTaskCancelMarker(
      loc, module, rewriter, flag.getType(), taskCell.getType());
  rewriter.create<mlir::func::CallOp>(
      loc, marker.getSymName(), mlir::TypeRange{},
      mlir::ValueRange{flag, taskCell, destCell});
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
  mlir::Value fileBytes = runtime.getByteLiteral(loc, fileAttr);
  mlir::Value funcBytes = runtime.getByteLiteral(loc, funcAttr);
  mlir::Value lineConst = rewriter.create<mlir::arith::ConstantIntOp>(
      loc, static_cast<int32_t>(line), 32);
  mlir::Value colConst = rewriter.create<mlir::arith::ConstantIntOp>(
      loc, static_cast<int32_t>(col), 32);
  runtime.call(loc, RuntimeSymbols::kTracebackPush, mlir::Type(),
               mlir::ValueRange{fileBytes, funcBytes, lineConst, colConst});
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

namespace async_descriptor {

static constexpr unsigned kValueIndex = 0;
static constexpr unsigned kExceptionCellIndex = 1;
static constexpr unsigned kCancelFlagIndex = 2;

mlir::Value asyncValue(mlir::ValueRange descriptor) {
  if (descriptor.empty())
    return {};
  return descriptor[kValueIndex];
}

mlir::Value exceptionCell(mlir::ValueRange descriptor) {
  if (descriptor.size() <= kExceptionCellIndex)
    return {};
  mlir::Value cell = descriptor[kExceptionCellIndex];
  return isExceptionCell(cell) ? cell : mlir::Value();
}

mlir::Value cancelFlag(mlir::ValueRange descriptor) {
  if (descriptor.size() <= kCancelFlagIndex)
    return {};
  mlir::Value flag = descriptor[kCancelFlagIndex];
  return mlir::isa<mlir::MemRefType>(flag.getType()) ? flag : mlir::Value();
}

llvm::SmallVector<mlir::Value>
pack(mlir::Value value, mlir::Value exceptionCell, mlir::Value cancelFlag) {
  llvm::SmallVector<mlir::Value> descriptor{value, exceptionCell};
  if (cancelFlag)
    descriptor.push_back(cancelFlag);
  return descriptor;
}

} // namespace async_descriptor

static mlir::Value getAwaitableExceptionCell(mlir::ValueRange awaitable) {
  return async_descriptor::exceptionCell(awaitable);
}

static mlir::Value
createExceptionCell(mlir::Location loc, mlir::ModuleOp module,
                    mlir::ConversionPatternRewriter &rewriter,
                    const PyLLVMTypeConverter &typeConverter,
                    mlir::Type targetType = {}) {
  (void)module;
  (void)typeConverter;
  (void)targetType;

  auto cellStorageType =
      async_runtime::getExceptionCellType(rewriter.getContext());
  mlir::Value cellStorage =
      rewriter.create<mlir::memref::AllocOp>(loc, cellStorageType);
  async_runtime::ExceptionCell::mark(cellStorage);
  mlir::Value zero = createI64Constant(loc, rewriter, 0);
  rewriter.create<mlir::memref::StoreOp>(loc, zero, cellStorage,
                                         createIndexConstant(loc, rewriter, 0));
  return cellStorage;
}

static mlir::Value loadExceptionCell(mlir::Location loc, mlir::Value cell,
                                     mlir::RewriterBase &rewriter) {
  return async_runtime::ExceptionCell::load(loc, cell, rewriter);
}

static mlir::LogicalResult
emitReleaseAggregateLoadedException(mlir::Location loc, mlir::ModuleOp module,
                                    mlir::ConversionPatternRewriter &rewriter,
                                    const PyLLVMTypeConverter &typeConverter,
                                    mlir::Value exception) {
  return async_runtime::ExceptionCell::releasePayload(loc, module, rewriter,
                                                      typeConverter, exception,
                                                      /*aggregateLoaded=*/true);
}

static void emitReleaseNoneAwaitPayload(
    mlir::Location loc, mlir::Type logicalType, mlir::ModuleOp module,
    mlir::ConversionPatternRewriter &rewriter,
    const PyLLVMTypeConverter &typeConverter, mlir::Value payload) {
  (void)loc;
  (void)logicalType;
  (void)module;
  (void)rewriter;
  (void)typeConverter;
  (void)payload;
}

static mlir::LogicalResult
emitFreeExceptionCell(mlir::Location loc, mlir::ModuleOp module,
                      mlir::ConversionPatternRewriter &rewriter,
                      const PyLLVMTypeConverter &typeConverter,
                      mlir::Value exceptionCell) {
  if (!isExceptionCell(exceptionCell))
    return mlir::success();
  return async_runtime::ExceptionCell::free(loc, module, rewriter,
                                            typeConverter, exceptionCell);
}

static mlir::LogicalResult
emitDestroyExceptionCell(mlir::Location loc, mlir::ModuleOp module,
                         mlir::ConversionPatternRewriter &rewriter,
                         const PyLLVMTypeConverter &typeConverter,
                         mlir::Value exceptionCell) {
  if (!isExceptionCell(exceptionCell))
    return mlir::success();
  return async_runtime::ExceptionCell::destroy(loc, module, rewriter,
                                               typeConverter, exceptionCell);
}

static mlir::LogicalResult
emitDestroyKnownEmptyExceptionCell(mlir::Location loc, mlir::ModuleOp module,
                                   mlir::ConversionPatternRewriter &rewriter,
                                   const PyLLVMTypeConverter &typeConverter,
                                   mlir::Value exceptionCell) {
  if (!isExceptionCell(exceptionCell))
    return mlir::success();
  return async_runtime::ExceptionCell::destroyKnownEmpty(
      loc, module, rewriter, typeConverter, exceptionCell);
}

static void
emitDestroyTaskCancelFlag(mlir::Location loc, mlir::ValueRange descriptor,
                          mlir::ConversionPatternRewriter &rewriter) {
  mlir::Value flag = async_descriptor::cancelFlag(descriptor);
  if (!flag)
    return;
  rewriter.create<mlir::memref::DeallocOp>(loc, flag);
}

static void emitDropAsyncHandle(mlir::Location loc, mlir::Value asyncValue,
                                mlir::ConversionPatternRewriter &rewriter) {
  if (!mlir::isa<mlir::async::ValueType>(asyncValue.getType()))
    return;
  rewriter.create<mlir::async::RuntimeDropRefOp>(loc, asyncValue,
                                                 rewriter.getI64IntegerAttr(1));
}

static mlir::LogicalResult
storeExceptionCell(mlir::Location loc, mlir::Value cell, mlir::Value exception,
                   mlir::ModuleOp module,
                   mlir::ConversionPatternRewriter &rewriter,
                   const PyLLVMTypeConverter &typeConverter) {
  if (!isExceptionCell(cell))
    return mlir::success();
  return async_runtime::ExceptionCell::storeFirst(loc, cell, exception, module,
                                                  rewriter, typeConverter);
}

static mlir::Value
createExceptionWithMessage(mlir::Location loc, mlir::ModuleOp module,
                           mlir::ConversionPatternRewriter &rewriter,
                           const PyLLVMTypeConverter &typeConverter,
                           llvm::StringRef messageText) {
  static constexpr int64_t kRuntimeErrorClassId = 3;
  RuntimeAPI runtime(module, rewriter, typeConverter);
  llvm::SmallVector<mlir::Type> resultTypes;
  object_abi::exception_abi::Parts::storageTypes(module.getContext(),
                                                 resultTypes);
  if (resultTypes.size() != 3)
    return {};
  mlir::Value bytes =
      runtime.getByteLiteral(loc, rewriter.getStringAttr(messageText));
  mlir::Value start = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
  mlir::Value messageLength =
      runtime.getI64Constant(loc, static_cast<int64_t>(messageText.size()));
  llvm::SmallVector<mlir::Type, 2> unicodeTypes;
  object_abi::str_abi::Parts::storageTypes(module.getContext(), unicodeTypes);
  auto message = runtime.call(loc, RuntimeSymbols::kUnicodeFromBytes,
                              mlir::TypeRange(unicodeTypes),
                              mlir::ValueRange{bytes, start, messageLength});
  llvm::SmallVector<mlir::Value, 3> parts;
  mlir::Value classId = runtime.getI64Constant(loc, kRuntimeErrorClassId);
  auto header = runtime.call(loc, RuntimeSymbols::kExceptionNew,
                             mlir::TypeRange{resultTypes.front()},
                             mlir::ValueRange{classId});
  parts.push_back(header.getResult());
  parts.append(message.getResults().begin(), message.getResults().end());
  return packAsyncPayload(loc, ExceptionType::get(module.getContext()), parts,
                          typeConverter, rewriter);
}

static bool isNoThrowFunction(mlir::Operation *op) {
  return op && op->hasAttr("nothrow") && !op->hasAttr("maythrow");
}

static bool isNoThrowAsyncTarget(mlir::ModuleOp module,
                                 mlir::FlatSymbolRefAttr target) {
  if (!module || !target)
    return false;
  if (auto asyncFunc =
          module.lookupSymbol<mlir::async::FuncOp>(target.getValue()))
    return isNoThrowFunction(asyncFunc.getOperation());
  if (auto func = module.lookupSymbol<mlir::func::FuncOp>(target.getValue()))
    return isNoThrowFunction(func.getOperation());
  return false;
}

static bool isNoThrowAwaitable(mlir::Value awaitable, mlir::ModuleOp module) {
  if (!awaitable)
    return false;
  if (auto coro = awaitable.getDefiningOp<CoroCreateOp>())
    return isNoThrowAsyncTarget(module, coro.getTargetAttr());
  if (auto gather = awaitable.getDefiningOp<AsyncGatherOp>()) {
    for (mlir::Value child : gather.getAwaitables())
      if (!isNoThrowAwaitable(child, module))
        return false;
    return true;
  }
  return false;
}

static bool isNoThrowAsyncValue(mlir::Value asyncValue, mlir::ModuleOp module) {
  mlir::Value producer = stripDescriptorCasts(asyncValue);
  if (auto call = producer.getDefiningOp<mlir::async::CallOp>())
    return isNoThrowAsyncTarget(module, call.getCalleeAttr());
  return false;
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
  auto aggregate =
      mlir::dyn_cast<mlir::LLVM::LLVMStructType>(exception.getType());
  if (mlir::isa<ExceptionType>(exception.getType()) ||
      (aggregate && !aggregate.isOpaque() && aggregate.getBody().size() == 3)) {
    llvm::SmallVector<mlir::Value> parts =
        unpackAsyncPayload(loc, ExceptionType::get(module.getContext()),
                           exception, typeConverter, rewriter);
    if (parts.size() == 3) {
      runtime.call(loc, RuntimeSymbols::kEHThrowException, mlir::Type(), parts);
      rewriter.create<mlir::LLVM::UnreachableOp>(loc);
      return;
    }
  }
  mlir::emitError(loc)
      << "async exception throw requires exception parts descriptor ABI";
  rewriter.create<mlir::LLVM::UnreachableOp>(loc);
}

static void emitAsyncExceptionPayloadInvariantAbort(
    mlir::Location loc, mlir::ModuleOp module,
    mlir::ConversionPatternRewriter &rewriter) {
  auto abortFn = getOrInsertLLVMFunc(
      loc, module, rewriter, "abort",
      mlir::LLVM::LLVMVoidType::get(rewriter.getContext()), {});
  auto call = rewriter.create<mlir::LLVM::CallOp>(
      loc, mlir::TypeRange{},
      mlir::SymbolRefAttr::get(module.getContext(), abortFn.getName()),
      mlir::ValueRange{});
  call->setAttr(ControlFlowContractAttrs::kNoReturn,
                mlir::UnitAttr::get(module.getContext()));
  rewriter.create<mlir::LLVM::UnreachableOp>(loc);
}

static mlir::LogicalResult
emitAsyncAwaitErrorThrow(mlir::Location loc, mlir::ModuleOp module,
                         mlir::ConversionPatternRewriter &rewriter,
                         const PyLLVMTypeConverter &typeConverter,
                         mlir::Value exceptionCell) {
  if (!isExceptionCell(exceptionCell)) {
    emitTracebackPush(loc, module, rewriter, typeConverter);
    emitAsyncExceptionPayloadInvariantAbort(loc, module, rewriter);
    return mlir::success();
  }

  mlir::Value exception = loadExceptionCell(loc, exceptionCell, rewriter);
  mlir::Value isNull =
      async_runtime::ExceptionCell::isNull(loc, exception, rewriter);
  if (!isNull) {
    emitTracebackPush(loc, module, rewriter, typeConverter);
    emitAsyncExceptionPayloadInvariantAbort(loc, module, rewriter);
    return mlir::success();
  }
  emitTracebackPush(loc, module, rewriter, typeConverter);

  mlir::Block *currentBlock = rewriter.getInsertionBlock();
  mlir::Block *abortBlock = rewriter.createBlock(
      currentBlock->getParent(), std::next(currentBlock->getIterator()));
  mlir::Block *throwBlock = rewriter.createBlock(
      currentBlock->getParent(), std::next(abortBlock->getIterator()));
  rewriter.setInsertionPointToEnd(currentBlock);
  rewriter.create<mlir::cf::CondBranchOp>(loc, isNull, abortBlock, throwBlock);

  rewriter.setInsertionPointToStart(abortBlock);
  if (mlir::failed(emitFreeExceptionCell(loc, module, rewriter, typeConverter,
                                         exceptionCell)))
    return mlir::failure();
  emitAsyncExceptionPayloadInvariantAbort(loc, module, rewriter);

  rewriter.setInsertionPointToStart(throwBlock);
  if (mlir::failed(async_runtime::ExceptionCell::retainPayload(
          loc, module, rewriter, typeConverter, exception,
          ThreadSafetyAttrs::kPremiseAggregateBorrow)))
    return mlir::failure();
  if (mlir::failed(emitReleaseAggregateLoadedException(
          loc, module, rewriter, typeConverter, exception)))
    return mlir::failure();
  if (mlir::failed(emitFreeExceptionCell(loc, module, rewriter, typeConverter,
                                         exceptionCell)))
    return mlir::failure();
  emitThrowException(loc, module, rewriter, typeConverter, exception);
  return mlir::success();
}

static mlir::LogicalResult
emitStoreCancellationException(mlir::Location loc, mlir::ModuleOp module,
                               mlir::ConversionPatternRewriter &rewriter,
                               const PyLLVMTypeConverter &typeConverter,
                               mlir::Value exceptionCell) {
  mlir::Value exception = createExceptionWithMessage(
      loc, module, rewriter, typeConverter, "task cancelled");
  if (!exception) {
    emitAsyncExceptionPayloadInvariantAbort(loc, module, rewriter);
    return mlir::success();
  }
  if (mlir::failed(storeExceptionCell(loc, exceptionCell, exception, module,
                                      rewriter, typeConverter)))
    return mlir::failure();
  return async_runtime::ExceptionCell::releasePayload(loc, module, rewriter,
                                                      typeConverter, exception);
}

static mlir::LogicalResult emitThrowFirstErroredAwaitable(
    mlir::Location loc, mlir::ModuleOp module,
    mlir::ConversionPatternRewriter &rewriter,
    const PyLLVMTypeConverter &typeConverter,
    llvm::ArrayRef<std::pair<mlir::Value, mlir::Value>> awaitables,
    llvm::ArrayRef<llvm::SmallVector<mlir::Value>> descriptors,
    llvm::ArrayRef<mlir::Type> awaitableTypes) {
  auto emitDestroyAwaitableResources = [&]() -> mlir::LogicalResult {
    for (auto [index, awaitableAndCell] : llvm::enumerate(awaitables)) {
      auto [asyncValue, cell] = awaitableAndCell;
      if (mlir::failed(emitDestroyExceptionCell(loc, module, rewriter,
                                                typeConverter, cell)))
        return mlir::failure();
      if (index < descriptors.size() && index < awaitableTypes.size() &&
          mlir::isa<TaskType>(awaitableTypes[index]))
        emitDestroyTaskCancelFlag(loc, descriptors[index], rewriter);
      emitDropAsyncHandle(loc, asyncValue, rewriter);
    }
    return mlir::success();
  };

  mlir::Block *currentBlock = rewriter.getInsertionBlock();
  emitTracebackPush(loc, module, rewriter, typeConverter);
  mlir::Block *abortBlock = rewriter.createBlock(
      currentBlock->getParent(), std::next(currentBlock->getIterator()));
  mlir::Block *nextCheck = abortBlock;

  mlir::Type exceptionStorageType = getAsyncPayloadStorageType(
      loc, ExceptionType::get(module.getContext()), typeConverter, rewriter);
  if (!exceptionStorageType) {
    if (mlir::failed(emitDestroyAwaitableResources()))
      return mlir::failure();
    emitAsyncExceptionPayloadInvariantAbort(loc, module, rewriter);
    return mlir::success();
  }
  for (auto [asyncValue, exceptionCell] : llvm::reverse(awaitables)) {
    if (!isExceptionCell(exceptionCell))
      continue;
    mlir::Block *throwBlock = rewriter.createBlock(abortBlock->getParent(),
                                                   abortBlock->getIterator());
    throwBlock->addArgument(exceptionStorageType, loc);
    rewriter.setInsertionPointToStart(throwBlock);
    mlir::Value thrownException = throwBlock->getArgument(0);
    if (mlir::failed(async_runtime::ExceptionCell::retainPayload(
            loc, module, rewriter, typeConverter, thrownException,
            ThreadSafetyAttrs::kPremiseAggregateBorrow)))
      return mlir::failure();
    if (mlir::failed(emitDestroyAwaitableResources()))
      return mlir::failure();
    emitThrowException(loc, module, rewriter, typeConverter, thrownException);

    mlir::Block *checkBlock = rewriter.createBlock(abortBlock->getParent(),
                                                   throwBlock->getIterator());
    rewriter.setInsertionPointToStart(checkBlock);
    (void)asyncValue;
    mlir::Value exception = loadExceptionCell(loc, exceptionCell, rewriter);
    mlir::Value isNull =
        async_runtime::ExceptionCell::isNull(loc, exception, rewriter);
    if (!isNull) {
      if (mlir::failed(emitDestroyAwaitableResources()))
        return mlir::failure();
      emitAsyncExceptionPayloadInvariantAbort(loc, module, rewriter);
      return mlir::success();
    }
    rewriter.create<mlir::cf::CondBranchOp>(loc, isNull, nextCheck,
                                            mlir::ValueRange{}, throwBlock,
                                            mlir::ValueRange{exception});
    nextCheck = checkBlock;
  }

  rewriter.setInsertionPointToEnd(currentBlock);
  rewriter.create<mlir::cf::BranchOp>(loc, nextCheck);

  rewriter.setInsertionPointToStart(abortBlock);
  if (mlir::failed(emitDestroyAwaitableResources()))
    return mlir::failure();
  emitAsyncExceptionPayloadInvariantAbort(loc, module, rewriter);
  return mlir::success();
}

static mlir::Value
storageFromUnpackedAsyncPayload(mlir::Location loc, mlir::ValueRange payload,
                                mlir::Type logicalType, mlir::Type storageType,
                                mlir::ModuleOp module,
                                mlir::ConversionPatternRewriter &rewriter,
                                const PyLLVMTypeConverter &typeConverter) {
  if (payload.empty())
    return {};
  if (mlir::isa<IntType>(logicalType) && storageType == rewriter.getI64Type()) {
    if (payload.size() == 3) {
      RuntimeAPI runtime(module, rewriter, typeConverter);
      return runtime
          .call(loc, RuntimeSymbols::kLongAsI64, rewriter.getI64Type(), payload)
          .getResult();
    }
    if (payload.size() == 1)
      return Slot::storage(loc, payload.front(), logicalType, storageType,
                           module, rewriter, typeConverter);
    return {};
  }
  if (payload.size() > 1)
    return Slot::storage(loc, payload, logicalType, storageType, module,
                         rewriter, typeConverter);
  if (payload.size() != 1)
    return {};
  return Slot::storage(loc, payload.front(), logicalType, storageType, module,
                       rewriter, typeConverter);
}

static void
releaseStoredUnpackedAsyncPayload(mlir::Location loc, mlir::ValueRange payload,
                                  mlir::Type logicalType, mlir::ModuleOp module,
                                  mlir::ConversionPatternRewriter &rewriter,
                                  const PyLLVMTypeConverter &typeConverter) {
  if (mlir::isa<IntType>(logicalType) && payload.size() == 3) {
    RuntimeAPI runtime(module, rewriter, typeConverter);
    runtime.call(loc, RuntimeSymbols::kLongDecRef, mlir::Type(), payload);
    return;
  }
  if (mlir::isa<StrType>(logicalType) && payload.size() == 2) {
    RuntimeAPI runtime(module, rewriter, typeConverter);
    runtime.call(loc, RuntimeSymbols::kUnicodeDecRef, mlir::Type(), payload);
    return;
  }
  if (mlir::isa<ExceptionType>(logicalType) && payload.size() == 3) {
    RuntimeAPI runtime(module, rewriter, typeConverter);
    auto release = runtime.call(loc, RuntimeSymbols::kExceptionDecRef,
                                mlir::Type(), payload);
    release->setAttr(OwnershipContractAttrs::kAggregateRelease,
                     rewriter.getUnitAttr());
    return;
  }
}

static void emitReleaseLoadedPayloads(
    mlir::Location loc, mlir::ModuleOp module,
    mlir::ConversionPatternRewriter &rewriter,
    const PyLLVMTypeConverter &typeConverter,
    llvm::ArrayRef<llvm::SmallVector<mlir::Value>> payloads,
    llvm::ArrayRef<mlir::Type> elementTypes) {
  for (auto [index, payload] : llvm::enumerate(payloads)) {
    if (index >= elementTypes.size())
      break;
    releaseStoredUnpackedAsyncPayload(loc, payload, elementTypes[index], module,
                                      rewriter, typeConverter);
  }
}

struct CoroCreateLowering : public mlir::OpConversionPattern<CoroCreateOp> {
  CoroCreateLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<CoroCreateOp>(converter, ctx,
                                                mlir::PatternBenefit(2)) {}

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
                            rewriter, *typeConverter, resultTypes[1]);

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
    llvm::SmallVector<mlir::Value> descriptor =
        async_descriptor::pack(call.getResult(0), exceptionCell);
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
    mlir::Value coroutineValue = async_descriptor::asyncValue(coroutine);
    if (!coroutineValue)
      return mlir::failure();
    rewriter.replaceOp(op, coroutineValue);
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
    mlir::Value asyncValue = async_descriptor::asyncValue(awaitable);
    mlir::Value exceptionCell = getAwaitableExceptionCell(awaitable);
    if (!isExceptionCell(exceptionCell)) {
      mlir::Value producer = stripDescriptorCasts(asyncValue);
      if (auto call = producer.getDefiningOp<mlir::async::CallOp>()) {
        if (call->getNumOperands() > 0) {
          mlir::Value candidate = call->getOperand(call->getNumOperands() - 1);
          if (isExceptionCell(candidate))
            exceptionCell = candidate;
        }
      }
    }
    if (!isExceptionCell(exceptionCell)) {
      for (mlir::Operation *cursor = op->getPrevNode(); cursor;
           cursor = cursor->getPrevNode()) {
        if (cursor->hasTrait<mlir::OpTrait::IsTerminator>())
          break;
        auto call = mlir::dyn_cast<mlir::async::CallOp>(cursor);
        if (!call || call->getNumOperands() == 0)
          continue;
        mlir::Value candidate = call->getOperand(call->getNumOperands() - 1);
        if (!isExceptionCell(candidate))
          continue;
        exceptionCell = candidate;
        break;
      }
    }

    bool insideAsyncFunc =
        op->getParentOfType<mlir::async::FuncOp>() != nullptr;
    bool noThrowAwaitable = isNoThrowAwaitable(op.getAwaitable(), module) ||
                            isNoThrowAsyncValue(asyncValue, module);
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
        if (mlir::failed(emitDestroyKnownEmptyExceptionCell(
                op.getLoc(), module, rewriter, *typeConverter, exceptionCell)))
          return mlir::failure();
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
      if (!noThrowAwaitable)
        emitAwaitExceptionMarker(op.getLoc(), rewriter, module, exceptionCell,
                                 getCurrentAsyncExceptionCell(op));
      if (!noThrowAwaitable && mlir::isa<TaskType>(op.getAwaitable().getType()))
        emitTaskCancelMarker(op.getLoc(), rewriter, module, awaitable,
                             getCurrentAsyncExceptionCell(op));
      llvm::SmallVector<mlir::Value> results =
          unpackAsyncPayload(op.getLoc(), op.getResult().getType(),
                             asyncAwait.getResult(), *typeConverter, rewriter);
      if (results.empty())
        return mlir::failure();
      emitReleaseNoneAwaitPayload(op.getLoc(), op.getResult().getType(), module,
                                  rewriter, *typeConverter, results.front());
      if (mlir::failed(emitDestroyKnownEmptyExceptionCell(
              op.getLoc(), module, rewriter, *typeConverter, exceptionCell)))
        return mlir::failure();
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
    if (noThrowAwaitable) {
      auto awaitOp = rewriter.create<mlir::async::RuntimeLoadOp>(
          op.getLoc(), asyncValueType.getValueType(), asyncValue);
      llvm::SmallVector<mlir::Value> results =
          unpackAsyncPayload(op.getLoc(), op.getResult().getType(),
                             awaitOp.getResult(), *typeConverter, rewriter);
      if (results.empty())
        return mlir::failure();
      if (mlir::failed(Slot::refcount(
              op.getLoc(), results.front(), op.getResult().getType(), module,
              rewriter, *typeConverter, "incref",
              /*aggregateEffect=*/true, ThreadSafetyAttrs::kPremiseOwnedToken)))
        return mlir::failure();
      emitReleaseNoneAwaitPayload(op.getLoc(), op.getResult().getType(), module,
                                  rewriter, *typeConverter, results.front());
      if (mlir::failed(emitDestroyKnownEmptyExceptionCell(
              op.getLoc(), module, rewriter, *typeConverter, exceptionCell)))
        return mlir::failure();
      emitDropAsyncHandle(op.getLoc(), asyncValue, rewriter);
      rewriter.replaceOpWithMultiple(
          op, llvm::ArrayRef<mlir::ValueRange>{mlir::ValueRange{results}});
      return mlir::success();
    }

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
    if (mlir::failed(emitAsyncAwaitErrorThrow(op.getLoc(), module, rewriter,
                                              *typeConverter, exceptionCell)))
      return mlir::failure();

    rewriter.setInsertionPoint(op);
    auto awaitOp = rewriter.create<mlir::async::RuntimeLoadOp>(
        op.getLoc(), asyncValueType.getValueType(), asyncValue);
    llvm::SmallVector<mlir::Value> results =
        unpackAsyncPayload(op.getLoc(), op.getResult().getType(),
                           awaitOp.getResult(), *typeConverter, rewriter);
    if (results.empty())
      return mlir::failure();
    if (mlir::failed(Slot::refcount(
            op.getLoc(), results.front(), op.getResult().getType(), module,
            rewriter, *typeConverter, "incref",
            /*aggregateEffect=*/true, ThreadSafetyAttrs::kPremiseOwnedToken)))
      return mlir::failure();
    emitReleaseNoneAwaitPayload(op.getLoc(), op.getResult().getType(), module,
                                rewriter, *typeConverter, results.front());
    if (mlir::failed(emitDestroyKnownEmptyExceptionCell(
            op.getLoc(), module, rewriter, *typeConverter, exceptionCell)))
      return mlir::failure();
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

struct TaskCreateLowering : public mlir::OpConversionPattern<TaskCreateOp> {
  TaskCreateLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<TaskCreateOp>(converter, ctx,
                                                mlir::PatternBenefit(3)) {}

  mlir::LogicalResult
  rewriteTaskCreate(TaskCreateOp op, mlir::ValueRange operand,
                    mlir::ConversionPatternRewriter &rewriter) const {
    llvm::SmallVector<mlir::Value> coroutineStorage;
    mlir::ValueRange coroutine =
        expandConvertedAsyncDescriptor(operand, coroutineStorage);
    mlir::Value coroutineValue = async_descriptor::asyncValue(coroutine);
    mlir::Value exceptionCell = async_descriptor::exceptionCell(coroutine);
    if (!coroutineValue || !exceptionCell)
      return rewriter.notifyMatchFailure(
          op, "task.create expected one converted coroutine");

    auto taskType = mlir::dyn_cast<TaskType>(op.getResult().getType());
    if (!taskType)
      return rewriter.notifyMatchFailure(op,
                                         "task.create result is not !py.task");

    auto flagType = mlir::MemRefType::get({1}, rewriter.getI8Type());
    auto cancelFlag =
        rewriter.create<mlir::memref::AllocOp>(op.getLoc(), flagType);
    async_runtime::CancelFlag::mark(cancelFlag.getResult());
    rewriter.create<mlir::memref::StoreOp>(
        op.getLoc(),
        rewriter.create<mlir::arith::ConstantIntOp>(op.getLoc(), 0, 8),
        cancelFlag, createIndexConstant(op.getLoc(), rewriter, 0));

    llvm::SmallVector<mlir::Value> descriptor = async_descriptor::pack(
        coroutineValue, exceptionCell, cancelFlag.getResult());
    llvm::SmallVector<mlir::ValueRange> replacements{
        mlir::ValueRange{descriptor}};
    rewriter.replaceOpWithMultiple(op, replacements);
    return mlir::success();
  }

  mlir::LogicalResult
  matchAndRewrite(TaskCreateOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<mlir::Value, 1> operand{adaptor.getCoroutine()};
    return rewriteTaskCreate(op, mlir::ValueRange(operand), rewriter);
  }

  mlir::LogicalResult
  matchAndRewrite(TaskCreateOp op, OneToNOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    return rewriteTaskCreate(op, adaptor.getCoroutine(), rewriter);
  }
};

struct TaskCancelLowering : public mlir::OpConversionPattern<TaskCancelOp> {
  TaskCancelLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<TaskCancelOp>(converter, ctx,
                                                mlir::PatternBenefit(3)) {}

  mlir::LogicalResult
  replaceWithAccepted(TaskCancelOp op, mlir::Value accepted,
                      const PyLLVMTypeConverter &typeConverter,
                      mlir::ModuleOp module,
                      mlir::ConversionPatternRewriter &rewriter) const {
    mlir::Type resultType =
        typeConverter.convertType(op.getAccepted().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "failed to convert bool type");
    if (resultType == accepted.getType()) {
      rewriter.replaceOp(op, accepted);
      return mlir::success();
    }

    (void)module;
    if (!object_abi::Type::isStorageLike(resultType))
      return rewriter.notifyMatchFailure(
          op, "task.cancel result requires memref bool storage");
    return rewriter.notifyMatchFailure(
        op, "task.cancel bool object storage fallback is unsupported");
  }

  mlir::FailureOr<mlir::Value>
  emitTaskCancelRequest(TaskCancelOp op, mlir::ValueRange task,
                        mlir::ModuleOp module,
                        const PyLLVMTypeConverter &typeConverter,
                        mlir::ConversionPatternRewriter &rewriter) const {
    mlir::Value taskValue = async_descriptor::asyncValue(task);
    mlir::Value exceptionCell = async_descriptor::exceptionCell(task);
    mlir::Value cancelFlag = async_descriptor::cancelFlag(task);
    if (!taskValue || !exceptionCell || !cancelFlag)
      return mlir::failure();

    auto alreadyError = rewriter.create<mlir::async::RuntimeIsErrorOp>(
        op.getLoc(), rewriter.getI1Type(), taskValue);
    mlir::Value trueValue = rewriter.create<mlir::arith::ConstantOp>(
        op.getLoc(), rewriter.getBoolAttr(true));
    mlir::Value notAlreadyError = rewriter.create<mlir::arith::XOrIOp>(
        op.getLoc(), alreadyError.getIsError(), trueValue);
    mlir::Value requested =
        rewriter.create<mlir::arith::ConstantIntOp>(op.getLoc(), 1, 8);
    mlir::Value previousRequest = rewriter.create<mlir::memref::AtomicRMWOp>(
        op.getLoc(), mlir::arith::AtomicRMWKind::maxu, requested, cancelFlag,
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
    if (mlir::failed(emitStoreCancellationException(
            op.getLoc(), module, rewriter, typeConverter, exceptionCell)))
      return mlir::failure();
    rewriter.create<mlir::cf::BranchOp>(op.getLoc(), afterStore);

    rewriter.setInsertionPointToStart(afterStore);
    return accepted;
  }

  mlir::LogicalResult
  rewriteTaskCancel(TaskCancelOp op, mlir::ValueRange operand,
                    mlir::ConversionPatternRewriter &rewriter) const {
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return mlir::failure();
    mlir::ValueRange task = operand;
    llvm::SmallVector<mlir::Value> taskStorage;
    task = expandConvertedAsyncDescriptor(task, taskStorage);
    if (task.size() != 3)
      return rewriter.notifyMatchFailure(
          op, "task.cancel expected one converted task descriptor");

    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    mlir::FailureOr<mlir::Value> accepted =
        emitTaskCancelRequest(op, task, module, *typeConverter, rewriter);
    if (mlir::failed(accepted))
      return mlir::failure();
    return replaceWithAccepted(op, *accepted, *typeConverter, module, rewriter);
  }

  mlir::LogicalResult
  matchAndRewrite(TaskCancelOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<mlir::Value, 1> operand{adaptor.getTask()};
    return rewriteTaskCancel(op, mlir::ValueRange(operand), rewriter);
  }

  mlir::LogicalResult
  matchAndRewrite(TaskCancelOp op, OneToNOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    return rewriteTaskCancel(op, adaptor.getTask(), rewriter);
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
    mlir::Value exceptionCell = createExceptionCell(
        op.getLoc(), module, rewriter, *typeConverter, futureTypes[1]);
    mlir::Type payloadType = asyncValueType.getValueType();
    RuntimeAPI runtime(module, rewriter, *typeConverter);
    mlir::Value none = runtime.getNoneValue(op.getLoc());
    if (none.getType() != payloadType)
      none = rewriter
                 .create<mlir::UnrealizedConversionCastOp>(
                     op.getLoc(), mlir::TypeRange{payloadType},
                     mlir::ValueRange{none})
                 .getResult(0);
    rewriter.create<mlir::async::RuntimeStoreOp>(op.getLoc(), none, storage);
    rewriter.create<mlir::async::RuntimeSetAvailableOp>(op.getLoc(), storage);
    llvm::SmallVector<mlir::Value> descriptor =
        async_descriptor::pack(storage, exceptionCell);
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

    mlir::MemRefType headerType = getTupleHeaderMemRefType(op.getContext());
    mlir::MemRefType itemsType =
        typeConverter->getTupleItemsMemRefType(tupleType);
    if (!headerType || !itemsType)
      return mlir::failure();

    llvm::SmallVector<std::pair<mlir::Value, mlir::Value>> awaitables;
    llvm::SmallVector<llvm::SmallVector<mlir::Value>> awaitableDescriptors;
    llvm::SmallVector<mlir::Type> awaitableTypes;
    llvm::SmallVector<char> noThrowAwaitables;
    for (mlir::ValueRange awaitable : adaptor.getAwaitables()) {
      llvm::SmallVector<mlir::Value> awaitableStorage;
      awaitable = expandConvertedAsyncDescriptor(awaitable, awaitableStorage);
      if (awaitable.empty())
        return mlir::failure();
      awaitables.push_back({async_descriptor::asyncValue(awaitable),
                            getAwaitableExceptionCell(awaitable)});
      awaitableDescriptors.emplace_back(awaitable.begin(), awaitable.end());
    }
    for (mlir::Value awaitable : op.getAwaitables()) {
      awaitableTypes.push_back(awaitable.getType());
      noThrowAwaitables.push_back(isNoThrowAwaitable(awaitable, module));
    }
    for (auto [index, awaitableAndCell] : llvm::enumerate(awaitables))
      if (index < noThrowAwaitables.size() &&
          isNoThrowAsyncValue(awaitableAndCell.first, module))
        noThrowAwaitables[index] = true;
    bool allNoThrowAwaitables =
        llvm::all_of(noThrowAwaitables, [](char value) { return value != 0; });

    mlir::Location loc = op.getLoc();
    auto elementTypes = tupleType.getElementTypes();
    bool insideAsyncFunc =
        op->getParentOfType<mlir::async::FuncOp>() != nullptr;
    bool canStorePrimitivePayloadsImmediately =
        insideAsyncFunc && allNoThrowAwaitables &&
        llvm::all_of(elementTypes,
                     [](mlir::Type type) { return !Slot::refcounted(type); });

    if (canStorePrimitivePayloadsImmediately) {
      mlir::Value allocSize = createIndexConstant(
          loc, rewriter, static_cast<int64_t>(awaitables.size()));
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
          createI64Constant(loc, rewriter,
                            static_cast<int64_t>(awaitables.size())),
          header, createIndexConstant(loc, rewriter, 0));
      for (int64_t slot = 1; slot < kTupleHeaderSize; ++slot) {
        rewriter.create<mlir::memref::StoreOp>(
            loc, createI64Constant(loc, rewriter, 0), header,
            createIndexConstant(loc, rewriter, slot));
      }

      for (auto [index, awaitableAndCell] : llvm::enumerate(awaitables)) {
        auto [awaitable, exceptionCell] = awaitableAndCell;
        mlir::Value packedPayload;
        if (isKnownAvailableAsyncValueBefore(awaitable, op.getOperation())) {
          auto asyncValueType =
              mlir::dyn_cast<mlir::async::ValueType>(awaitable.getType());
          if (!asyncValueType)
            return rewriter.notifyMatchFailure(
                op, "gather awaitable did not lower to async.value");
          rewriter.create<mlir::async::RuntimeAwaitOp>(loc, awaitable);
          packedPayload = rewriter
                              .create<mlir::async::RuntimeLoadOp>(
                                  loc, asyncValueType.getValueType(), awaitable)
                              .getResult();
        } else {
          packedPayload =
              rewriter.create<mlir::async::AwaitOp>(loc, awaitable).getResult();
        }

        llvm::SmallVector<mlir::Value> payload = unpackAsyncPayload(
            loc, elementTypes[index], packedPayload, *typeConverter, rewriter);
        if (payload.empty())
          return mlir::failure();
        mlir::Value stored = storageFromUnpackedAsyncPayload(
            loc, payload, elementTypes[index], itemsType.getElementType(),
            module, rewriter, *typeConverter);
        if (!stored)
          return mlir::failure();
        auto store = rewriter.create<mlir::memref::StoreOp>(
            loc, stored, items,
            createIndexConstant(loc, rewriter, static_cast<int64_t>(index)));
        Slot::markTransfer(store.getOperation());
        if (mlir::isa<IntType>(elementTypes[index]) && payload.size() == 3)
          releaseStoredUnpackedAsyncPayload(loc, payload, elementTypes[index],
                                            module, rewriter, *typeConverter);

        if (mlir::failed(emitDestroyKnownEmptyExceptionCell(
                loc, module, rewriter, *typeConverter, exceptionCell)))
          return mlir::failure();
        if (mlir::isa<TaskType>(op.getAwaitables()[index].getType()))
          emitDestroyTaskCancelFlag(loc, awaitableDescriptors[index], rewriter);
        emitDropAsyncHandle(loc, awaitable, rewriter);
      }

      rewriter.replaceOpWithMultiple(
          op, llvm::ArrayRef<mlir::ValueRange>{
                  mlir::ValueRange{header.getResult(), items.getResult()}});
      return mlir::success();
    }

    mlir::Block *topLevelErrorBlock = nullptr;
    llvm::SmallVector<llvm::SmallVector<mlir::Value>> payloads;
    if (!insideAsyncFunc) {
      for (auto [awaitable, exceptionCell] : awaitables) {
        (void)exceptionCell;
        rewriter.create<mlir::async::RuntimeAwaitOp>(loc, awaitable);
      }
      if (allNoThrowAwaitables) {
        for (auto [awaitable, exceptionCell] : awaitables) {
          (void)exceptionCell;
          auto asyncValueType =
              mlir::dyn_cast<mlir::async::ValueType>(awaitable.getType());
          if (!asyncValueType)
            return rewriter.notifyMatchFailure(
                op, "gather awaitable did not lower to async.value");
          auto loadOp = rewriter.create<mlir::async::RuntimeLoadOp>(
              loc, asyncValueType.getValueType(), awaitable);
          llvm::SmallVector<mlir::Value> payload =
              unpackAsyncPayload(loc, elementTypes[payloads.size()],
                                 loadOp.getResult(), *typeConverter, rewriter);
          if (payload.empty())
            return mlir::failure();
          payloads.push_back(std::move(payload));
        }
      } else {
        mlir::Block *currentBlock = rewriter.getInsertionBlock();
        mlir::Block *continuationBlock =
            rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
        mlir::Block *errorBlock = rewriter.createBlock(
            continuationBlock->getParent(), continuationBlock->getIterator());
        topLevelErrorBlock = errorBlock;

        rewriter.setInsertionPointToStart(errorBlock);
        if (mlir::failed(emitThrowFirstErroredAwaitable(
                loc, module, rewriter, *typeConverter, awaitables,
                awaitableDescriptors, awaitableTypes)))
          return mlir::failure();

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
            mlir::Block *cleanupBlock =
                rewriter.createBlock(continuationBlock->getParent(),
                                     continuationBlock->getIterator());
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
          llvm::SmallVector<mlir::Value> payload =
              unpackAsyncPayload(loc, elementTypes[payloads.size()],
                                 loadOp.getResult(), *typeConverter, rewriter);
          if (payload.empty())
            return mlir::failure();
          payloads.push_back(std::move(payload));
          currentBlock = nextBlock;
          rewriter.setInsertionPointToEnd(currentBlock);
        }
        rewriter.create<mlir::cf::BranchOp>(loc, continuationBlock);
        rewriter.setInsertionPointToStart(continuationBlock);
      }
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
          llvm::SmallVector<mlir::Value> payload =
              unpackAsyncPayload(loc, elementTypes[payloads.size()],
                                 loadOp.getResult(), *typeConverter, rewriter);
          if (payload.empty())
            return mlir::failure();
          payloads.push_back(std::move(payload));
          continue;
        }

        auto awaitOp = rewriter.create<mlir::async::AwaitOp>(loc, awaitable);
        rewriter.setInsertionPointAfter(awaitOp);
        if (!noThrowAwaitables[index])
          emitAwaitExceptionMarker(loc, rewriter, module, exceptionCell,
                                   getCurrentAsyncExceptionCell(op));
        if (!noThrowAwaitables[index])
          emitTaskCancelMarker(loc, rewriter, module,
                               awaitableDescriptors[index],
                               getCurrentAsyncExceptionCell(op));
        llvm::SmallVector<mlir::Value> payload =
            unpackAsyncPayload(loc, elementTypes[index], awaitOp.getResult(),
                               *typeConverter, rewriter);
        if (payload.empty())
          return mlir::failure();
        payloads.push_back(std::move(payload));
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
      mlir::Value stored = storageFromUnpackedAsyncPayload(
          loc, payload, elementTypes[index], itemsType.getElementType(), module,
          rewriter, *typeConverter);
      if (!stored)
        return mlir::failure();
      auto store = rewriter.create<mlir::memref::StoreOp>(
          loc, stored, items,
          createIndexConstant(loc, rewriter, static_cast<int64_t>(index)));
      Slot::markTransfer(store.getOperation());
      if (mlir::isa<IntType>(elementTypes[index]) && payload.size() == 3)
        releaseStoredUnpackedAsyncPayload(loc, payload, elementTypes[index],
                                          module, rewriter, *typeConverter);
    }
    for (auto [index, awaitableAndCell] : llvm::enumerate(awaitables)) {
      auto [_, exceptionCell] = awaitableAndCell;
      if (mlir::failed(emitDestroyKnownEmptyExceptionCell(
              loc, module, rewriter, *typeConverter, exceptionCell)))
        return mlir::failure();
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
