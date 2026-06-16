#include "Common/Container.h"
#include "Common/LoweringUtils.h"
#include "Common/Object.h"
#include "Common/RuntimeSupport.h"
#include "Common/SlotUtils.h"
#include "Passes/Runtime/Helpers.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
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

mlir::LogicalResult appendFlattenedCallOperands(
    mlir::Location loc, mlir::ValueRange elements, mlir::FunctionType funcType,
    unsigned directInputCount, llvm::SmallVectorImpl<mlir::Value> &operands,
    mlir::RewriterBase &rewriter, const PyLLVMTypeConverter &typeConverter);

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
  if (!isCoroutineProtocolType(cast.getResult(0).getType()))
    return descriptor;
  sink.append(cast.getOperands().begin(), cast.getOperands().end());
  return mlir::ValueRange(sink);
}

namespace async_descriptor {

static constexpr unsigned kValueIndex = 0;
static constexpr unsigned kExceptionCellIndex = 1;

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

} // namespace async_descriptor

static mlir::Value getAwaitableExceptionCell(mlir::ValueRange awaitable) {
  return async_descriptor::exceptionCell(awaitable);
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
emitDestroyKnownEmptyExceptionCell(mlir::Location loc, mlir::ModuleOp module,
                                   mlir::ConversionPatternRewriter &rewriter,
                                   const PyLLVMTypeConverter &typeConverter,
                                   mlir::Value exceptionCell) {
  if (!isExceptionCell(exceptionCell))
    return mlir::success();
  return async_runtime::ExceptionCell::destroyKnownEmpty(
      loc, module, rewriter, typeConverter, exceptionCell);
}

static void emitDropAsyncHandle(mlir::Location loc, mlir::Value asyncValue,
                                mlir::ConversionPatternRewriter &rewriter) {
  if (!mlir::isa<mlir::async::ValueType>(asyncValue.getType()))
    return;
  rewriter.create<mlir::async::RuntimeDropRefOp>(loc, asyncValue,
                                                 rewriter.getI64IntegerAttr(1));
}

static mlir::Value materializeLogicalValueFromConverted(
    mlir::Location loc, mlir::Type logicalType, mlir::ValueRange values,
    mlir::ConversionPatternRewriter &rewriter) {
  if (values.size() == 1 && values.front().getType() == logicalType)
    return values.front();
  return rewriter
      .create<mlir::UnrealizedConversionCastOp>(
          loc, mlir::TypeRange{logicalType}, values)
      .getResult(0);
}

static mlir::Value
loadExceptionClassIdFromPayload(mlir::Location loc, mlir::Value exception,
                                mlir::ConversionPatternRewriter &rewriter) {
  auto partsType = mlir::dyn_cast<mlir::LLVM::LLVMStructType>(
      exception ? exception.getType() : mlir::Type{});
  if (!partsType || partsType.isOpaque() || partsType.getBody().size() != 3)
    return {};
  auto headerType =
      mlir::dyn_cast<mlir::LLVM::LLVMStructType>(partsType.getBody()[0]);
  if (!object_abi::Type::isLoweredStorage(headerType))
    return {};
  mlir::Value header = rewriter.create<mlir::LLVM::ExtractValueOp>(
      loc, headerType, exception, rewriter.getDenseI64ArrayAttr({0}));
  llvm::ArrayRef<mlir::Type> body = headerType.getBody();
  mlir::Value aligned = rewriter.create<mlir::LLVM::ExtractValueOp>(
      loc, body[1], header, rewriter.getDenseI64ArrayAttr({1}));
  mlir::Value offset = rewriter.create<mlir::LLVM::ExtractValueOp>(
      loc, rewriter.getI64Type(), header, rewriter.getDenseI64ArrayAttr({2}));
  mlir::Value stride = rewriter.create<mlir::LLVM::ExtractValueOp>(
      loc, rewriter.getI64Type(), header,
      rewriter.getDenseI64ArrayAttr({4, 0}));
  mlir::Value slot = rewriter.create<mlir::LLVM::ConstantOp>(
      loc, rewriter.getI64Type(),
      rewriter.getI64IntegerAttr(object_abi::exception_abi::kClassIdSlot));
  mlir::Value scaled = rewriter.create<mlir::LLVM::MulOp>(loc, slot, stride);
  mlir::Value index = rewriter.create<mlir::LLVM::AddOp>(loc, offset, scaled);
  auto ptrType = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
  mlir::Value address = rewriter.create<mlir::LLVM::GEPOp>(
      loc, ptrType, rewriter.getI64Type(), aligned,
      llvm::ArrayRef<mlir::LLVM::GEPArg>{index});
  return rewriter.create<mlir::LLVM::LoadOp>(loc, rewriter.getI64Type(),
                                             address);
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
  (void)module;
  if (!awaitable)
    return false;
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

    auto lowered = rewriter.create<mlir::async::CallOp>(
        op.getLoc(), op.getCalleeAttr(), resultTypes, adaptor.getOperands());
    lowered->setAttrs(op->getAttrs());
    rewriter.replaceOp(op, lowered.getResults());
    return mlir::success();
  }
};

struct AsyncFuncSignatureLowering
    : public mlir::OpConversionPattern<mlir::async::FuncOp> {
  AsyncFuncSignatureLowering(PyLLVMTypeConverter &converter,
                             mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<mlir::async::FuncOp>(
            converter, ctx, mlir::PatternBenefit(4)) {}

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
    emitDropAsyncHandle(op.getLoc(), asyncValue, rewriter);
    rewriter.replaceOpWithMultiple(
        op, llvm::ArrayRef<mlir::ValueRange>{mlir::ValueRange{results}});
    return mlir::success();
  }
};

struct AsyncNextLowering : public mlir::OpConversionPattern<AsyncNextOp> {
  AsyncNextLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<AsyncNextOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(AsyncNextOp op, OneToNOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return mlir::failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    bool insideAsyncFunc =
        op->getParentOfType<mlir::async::FuncOp>() != nullptr;
    if (!insideAsyncFunc)
      return op.emitOpError("lowering is valid only inside async functions");

    llvm::SmallVector<mlir::Value> awaitableStorage;
    mlir::ValueRange awaitable = expandConvertedAsyncDescriptor(
        adaptor.getAwaitable(), awaitableStorage);
    if (awaitable.empty())
      return mlir::failure();
    mlir::Value asyncValue = async_descriptor::asyncValue(awaitable);
    mlir::Value exceptionCell = getAwaitableExceptionCell(awaitable);
    if (!asyncValue)
      return rewriter.notifyMatchFailure(op,
                                         "awaitable did not lower to value");

    bool noThrowAwaitable = isNoThrowAwaitable(op.getAwaitable(), module) ||
                            isNoThrowAsyncValue(asyncValue, module);
    if (!noThrowAwaitable) {
      auto branch = mlir::dyn_cast_or_null<mlir::cf::CondBranchOp>(
          op->getBlock() ? op->getBlock()->getTerminator() : nullptr);
      if (!branch || branch.getCondition() != op.getValid())
        return op.emitOpError("may-throw lowering requires direct "
                              "cf.cond_br use of the valid result");
      for (mlir::Operation *user : op.getValid().getUsers())
        if (user != branch.getOperation())
          return op.emitOpError("may-throw lowering requires valid to be used "
                                "only by the following cf.cond_br");

      mlir::Block *trueDest = branch.getTrueDest();
      mlir::Block *falseDest = branch.getFalseDest();
      llvm::SmallVector<mlir::Value> trueOperands(
          branch.getTrueDestOperands().begin(),
          branch.getTrueDestOperands().end());
      llvm::SmallVector<mlir::Value> falseOperands(
          branch.getFalseDestOperands().begin(),
          branch.getFalseDestOperands().end());
      mlir::Value trueElementArgument;
      for (auto [index, value] : llvm::enumerate(trueOperands)) {
        if (value == op.getElement() && index < trueDest->getNumArguments()) {
          trueElementArgument = trueDest->getArgument(index);
          break;
        }
      }
      for (mlir::Value value : falseOperands)
        if (value == op.getElement() || value == op.getValid())
          return op.emitOpError("may-throw lowering cannot pass async.next "
                                "results to the false successor");

      llvm::SmallVector<std::pair<DecRefOp, unsigned>> trueElementDecRefs;
      for (mlir::OpOperand &use :
           llvm::make_early_inc_range(op.getElement().getUses())) {
        mlir::Operation *user = use.getOwner();
        if (user == branch.getOperation())
          continue;
        if (auto decRef = mlir::dyn_cast<DecRefOp>(user)) {
          if (decRef->getBlock() == trueDest) {
            if (trueElementArgument)
              decRef->setOperand(use.getOperandNumber(), trueElementArgument);
            else
              trueElementDecRefs.push_back(
                  {decRef, static_cast<unsigned>(use.getOperandNumber())});
            continue;
          }
          if (decRef->getBlock() == falseDest) {
            rewriter.eraseOp(decRef);
            continue;
          }
        }
        return op.emitOpError("may-throw lowering requires element to be "
                              "passed only to the following cf.cond_br");
      }

      rewriter.setInsertionPoint(op);
      rewriter.create<mlir::async::RuntimeAwaitOp>(op.getLoc(), asyncValue);
      auto isError = rewriter.create<mlir::async::RuntimeIsErrorOp>(
          op.getLoc(), rewriter.getI1Type(), asyncValue);

      mlir::Block *currentBlock = op->getBlock();
      mlir::Region *region = currentBlock->getParent();
      auto insertIt = std::next(mlir::Region::iterator(currentBlock));
      mlir::Block *errorBlock = rewriter.createBlock(region, insertIt);
      mlir::Block *successBlock = rewriter.createBlock(region, insertIt);

      rewriter.eraseOp(branch);
      rewriter.eraseOp(op);
      rewriter.setInsertionPointToEnd(currentBlock);
      rewriter.create<mlir::cf::CondBranchOp>(
          isError.getLoc(), isError.getIsError(), errorBlock, successBlock);

      rewriter.setInsertionPointToStart(successBlock);
      auto asyncValueType =
          mlir::dyn_cast<mlir::async::ValueType>(asyncValue.getType());
      if (!asyncValueType)
        return rewriter.notifyMatchFailure(
            op, "awaitable did not lower to async.value");
      mlir::Value packedPayload =
          rewriter
              .create<mlir::async::RuntimeLoadOp>(
                  op.getLoc(), asyncValueType.getValueType(), asyncValue)
              .getResult();
      llvm::SmallVector<mlir::Value> element =
          unpackAsyncPayload(op.getLoc(), op.getElement().getType(),
                             packedPayload, *typeConverter, rewriter);
      if (element.empty())
        return mlir::failure();
      emitReleaseNoneAwaitPayload(op.getLoc(), op.getElement().getType(),
                                  module, rewriter, *typeConverter,
                                  element.front());
      if (mlir::failed(emitDestroyKnownEmptyExceptionCell(
              op.getLoc(), module, rewriter, *typeConverter, exceptionCell)))
        return mlir::failure();
      emitDropAsyncHandle(op.getLoc(), asyncValue, rewriter);
      mlir::Value logicalElement = materializeLogicalValueFromConverted(
          op.getLoc(), op.getElement().getType(), element, rewriter);
      for (mlir::Value &operand : trueOperands)
        if (operand == op.getElement())
          operand = logicalElement;
      for (auto [decRef, operandNumber] : trueElementDecRefs)
        decRef->setOperand(operandNumber, logicalElement);
      rewriter.create<mlir::cf::BranchOp>(op.getLoc(), trueDest, trueOperands);

      rewriter.setInsertionPointToStart(errorBlock);
      mlir::Value exception =
          loadExceptionCell(op.getLoc(), exceptionCell, rewriter);
      mlir::Value isNull = async_runtime::ExceptionCell::isNull(
          op.getLoc(), exception, rewriter);
      mlir::Block *errorDispatchBlock = rewriter.getInsertionBlock();
      mlir::Block *classifyBlock = rewriter.createBlock(
          region, std::next(errorDispatchBlock->getIterator()));
      mlir::Block *abortBlock =
          rewriter.createBlock(region, std::next(classifyBlock->getIterator()));
      rewriter.setInsertionPointToEnd(errorDispatchBlock);
      rewriter.create<mlir::cf::CondBranchOp>(op.getLoc(), isNull, abortBlock,
                                              classifyBlock);

      rewriter.setInsertionPointToStart(abortBlock);
      if (mlir::failed(emitFreeExceptionCell(op.getLoc(), module, rewriter,
                                             *typeConverter, exceptionCell)))
        return mlir::failure();
      emitAsyncExceptionPayloadInvariantAbort(op.getLoc(), module, rewriter);

      rewriter.setInsertionPointToStart(classifyBlock);
      mlir::Value classId =
          loadExceptionClassIdFromPayload(op.getLoc(), exception, rewriter);
      if (!classId)
        return op.emitOpError("failed to read async exception class id");
      mlir::Value stopAsyncIterationId =
          rewriter.create<mlir::LLVM::ConstantOp>(
              op.getLoc(), rewriter.getI64Type(),
              rewriter.getI64IntegerAttr(10));
      mlir::Value isStopAsyncIteration = rewriter.create<mlir::LLVM::ICmpOp>(
          op.getLoc(), mlir::LLVM::ICmpPredicate::eq, classId,
          stopAsyncIterationId);
      mlir::Block *stopBlock =
          rewriter.createBlock(region, std::next(classifyBlock->getIterator()));
      mlir::Block *propagateBlock =
          rewriter.createBlock(region, std::next(stopBlock->getIterator()));
      rewriter.setInsertionPointToEnd(classifyBlock);
      rewriter.create<mlir::cf::CondBranchOp>(op.getLoc(), isStopAsyncIteration,
                                              stopBlock, propagateBlock);

      rewriter.setInsertionPointToStart(stopBlock);
      if (mlir::failed(emitReleaseAggregateLoadedException(
              op.getLoc(), module, rewriter, *typeConverter, exception)))
        return mlir::failure();
      if (mlir::failed(emitFreeExceptionCell(op.getLoc(), module, rewriter,
                                             *typeConverter, exceptionCell)))
        return mlir::failure();
      emitDropAsyncHandle(op.getLoc(), asyncValue, rewriter);
      rewriter.create<mlir::cf::BranchOp>(op.getLoc(), falseDest,
                                          falseOperands);

      rewriter.setInsertionPointToStart(propagateBlock);
      emitTracebackPush(op.getLoc(), module, rewriter, *typeConverter);
      if (mlir::failed(async_runtime::ExceptionCell::retainPayload(
              op.getLoc(), module, rewriter, *typeConverter, exception,
              ThreadSafetyAttrs::kPremiseAggregateBorrow)))
        return mlir::failure();
      if (mlir::failed(emitReleaseAggregateLoadedException(
              op.getLoc(), module, rewriter, *typeConverter, exception)))
        return mlir::failure();
      if (mlir::failed(emitFreeExceptionCell(op.getLoc(), module, rewriter,
                                             *typeConverter, exceptionCell)))
        return mlir::failure();
      emitDropAsyncHandle(op.getLoc(), asyncValue, rewriter);
      emitAsyncExceptionPayloadInvariantAbort(op.getLoc(), module, rewriter);
      return mlir::success();
    }

    mlir::Value packedPayload;
    if (isKnownAvailableAsyncValueBefore(asyncValue, op.getOperation())) {
      auto asyncValueType =
          mlir::dyn_cast<mlir::async::ValueType>(asyncValue.getType());
      if (!asyncValueType)
        return rewriter.notifyMatchFailure(
            op, "awaitable did not lower to async.value");
      rewriter.create<mlir::async::RuntimeAwaitOp>(op.getLoc(), asyncValue);
      packedPayload =
          rewriter
              .create<mlir::async::RuntimeLoadOp>(
                  op.getLoc(), asyncValueType.getValueType(), asyncValue)
              .getResult();
    } else {
      auto asyncAwait =
          rewriter.create<mlir::async::AwaitOp>(op.getLoc(), asyncValue);
      rewriter.setInsertionPointAfter(asyncAwait);
      packedPayload = asyncAwait.getResult();
    }

    llvm::SmallVector<mlir::Value> element =
        unpackAsyncPayload(op.getLoc(), op.getElement().getType(),
                           packedPayload, *typeConverter, rewriter);
    if (element.empty())
      return mlir::failure();
    emitReleaseNoneAwaitPayload(op.getLoc(), op.getElement().getType(), module,
                                rewriter, *typeConverter, element.front());
    if (mlir::failed(emitDestroyKnownEmptyExceptionCell(
            op.getLoc(), module, rewriter, *typeConverter, exceptionCell)))
      return mlir::failure();
    emitDropAsyncHandle(op.getLoc(), asyncValue, rewriter);

    mlir::Value valid = rewriter.create<mlir::arith::ConstantIntOp>(
        op.getLoc(), 1, op.getValid().getType());
    rewriter.replaceOpWithMultiple(
        op, llvm::ArrayRef<mlir::ValueRange>{mlir::ValueRange{element},
                                             mlir::ValueRange{valid}});
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

} // namespace

namespace lowering::async_runtime::Patterns {
void populate(PyLLVMTypeConverter &typeConverter,
              mlir::RewritePatternSet &patterns) {
  auto *ctx = patterns.getContext();
  patterns.add<AsyncFuncSignatureLowering, AsyncCallTypeLowering, AwaitLowering,
               AsyncNextLowering, AsyncReturnLowering>(typeConverter, ctx);
}
} // namespace lowering::async_runtime::Patterns

} // namespace py
