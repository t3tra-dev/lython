#include "Common/ClassLayout.h"
#include "Common/Container.h"
#include "Common/LoweringUtils.h"
#include "Common/Object.h"
#include "Common/RuntimeSupport.h"
#include "Common/SlotUtils.h"
#include "Passes/OwnershipAnalysis.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/STLExtras.h"

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

namespace py {
namespace {

namespace lowering::refcount::Value {
mlir::LogicalResult dec(mlir::Location loc, mlir::Type logicalType,
                        mlir::ValueRange values, mlir::ModuleOp module,
                        mlir::ConversionPatternRewriter &rewriter,
                        const PyLLVMTypeConverter &typeConverter);
} // namespace lowering::refcount::Value
namespace lowering::refcount::Class {

mlir::LogicalResult
emit(mlir::Operation *op, mlir::ValueRange values, ClassType classType,
     mlir::ModuleOp module, mlir::ConversionPatternRewriter &rewriter,
     const PyLLVMTypeConverter &typeConverter, llvm::StringRef suffix,
     llvm::StringRef retainPremise = {}, bool verifiedOwnedToken = false) {
  auto layout = class_layout::get(op, classType, typeConverter);
  if (mlir::failed(layout))
    return mlir::failure();
  llvm::SmallVector<mlir::Type, 2> helperInputTypes;
  class_layout::partsValueTypes(*layout, helperInputTypes);
  std::string helperName = getClassHelperName(classType, suffix);
  auto helper = module.lookupSymbol<mlir::func::FuncOp>(helperName);
  if (!helper) {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    helper = rewriter.create<mlir::func::FuncOp>(
        op->getLoc(), helperName,
        mlir::FunctionType::get(rewriter.getContext(), helperInputTypes, {}));
    helper.setVisibility(mlir::SymbolTable::Visibility::Private);
  }
  if (suffix == "incref")
    ownership::effect::retain(helper.getOperation(), {0, 1});
  else if (suffix == "decref")
    ownership::effect::release(helper.getOperation(), {0, 1});

  auto applyCallContracts = [&](mlir::func::CallOp call) {
    if (!call)
      return;
    if (suffix == "incref") {
      threadsafe::Retain::premise(call.getOperation(), retainPremise);
      if (retainPremise == ThreadSafetyAttrs::kPremiseOwnedToken &&
          verifiedOwnedToken)
        threadsafe::Retain::verifyOwnedToken(call.getOperation());
    }
    if (suffix == "decref") {
      mlir::Attribute attr =
          op->getAttr(OwnershipContractAttrs::kAggregateRelease);
      call->setAttr(OwnershipContractAttrs::kAggregateRelease,
                    attr ? attr : rewriter.getUnitAttr());
    }
    if (suffix == "destroy_local")
      call->setAttr(OwnershipContractAttrs::kLocalDestroy,
                    rewriter.getUnitAttr());
  };

  if (values.size() == helperInputTypes.size()) {
    llvm::SmallVector<mlir::Value, 4> args;
    for (auto [value, expected] : llvm::zip(values, helperInputTypes)) {
      if (value.getType() == expected) {
        args.push_back(value);
        continue;
      }
      auto valueMemRef = mlir::dyn_cast<mlir::MemRefType>(value.getType());
      auto expectedMemRef = mlir::dyn_cast<mlir::MemRefType>(expected);
      if (!valueMemRef || !expectedMemRef)
        return mlir::failure();
      args.push_back(rewriter.create<mlir::memref::CastOp>(
          op->getLoc(), expectedMemRef, value));
    }
    auto call = rewriter.create<mlir::func::CallOp>(op->getLoc(), helper, args);
    applyCallContracts(call);
    return mlir::success();
  }

  if (values.size() == 1) {
    auto objectStruct = class_layout::objectCarrierType(layout->objectType);
    if (!objectStruct)
      return mlir::failure();
    mlir::Value carrier = values.front();
    if (carrier.getType() != layout->objectType) {
      if (suffix == "destroy_local")
        return mlir::failure();
      carrier = Slot::classCarrierFromValues(op->getLoc(), values, objectStruct,
                                             rewriter);
      if (!carrier)
        return mlir::failure();
    }
    auto call = Slot::classCarrierRefcount(
        op->getLoc(), carrier, classType, module, rewriter, suffix,
        /*aggregateEffect=*/false, retainPremise);
    if (!call)
      return mlir::failure();
    applyCallContracts(call);
    return mlir::success();
  }
  return mlir::failure();
}

} // namespace lowering::refcount::Class

namespace lowering::refcount::InsertionPoint {

void set(mlir::Operation *op, mlir::ConversionPatternRewriter &rewriter) {
  mlir::Operation *terminator = op->getBlock()->getTerminator();
  if (terminator && terminator != op && terminator->isBeforeInBlock(op)) {
    rewriter.setInsertionPoint(terminator);
    return;
  }
  rewriter.setInsertionPoint(op);
}

} // namespace lowering::refcount::InsertionPoint

namespace lowering::refcount::Immortal {

bool known(mlir::Value value) {
  mlir::Operation *def = value.getDefiningOp();
  if (!def)
    return false;
  if (def->hasAttr(OwnershipContractAttrs::kImmortalObject))
    return true;
  if (mlir::isa<NoneOp, FuncObjectOp, TupleEmptyOp, IntConstantOp,
                ExceptionNullOp, TracebackNullOp, LocationCurrentOp>(def))
    return true;
  if (auto classNew = mlir::dyn_cast<ClassNewOp>(def))
    return classNew.getClassNameAttr().getValue() == "Exception";
  if (auto cast = mlir::dyn_cast<mlir::UnrealizedConversionCastOp>(def))
    if (cast->getNumOperands() == 1)
      return known(cast.getOperand(0));
  return false;
}

} // namespace lowering::refcount::Immortal

namespace lowering::refcount::managed_container::Atomic {

mlir::Value load(mlir::Location loc, mlir::Value header, int64_t refcountSlot,
                 mlir::ConversionPatternRewriter &rewriter) {
  auto op = rewriter.create<mlir::memref::AtomicRMWOp>(
      loc, mlir::arith::AtomicRMWKind::addi,
      createI64Constant(loc, rewriter, 0), header,
      mlir::ValueRange{createIndexConstant(loc, rewriter, refcountSlot)});
  threadsafe::Atomic::set(op.getOperation(),
                          ThreadSafetyAttrs::kRoleContainerRefcountLoad,
                          ThreadSafetyAttrs::kOrderingAcquire);
  container::Header::markAtomicResource(op.getOperation(), header,
                                        refcountSlot);
  return op;
}

mlir::Value add(mlir::Location loc, mlir::Value header, int64_t refcountSlot,
                int64_t delta, llvm::StringRef role,
                llvm::StringRef retainPremise,
                mlir::ConversionPatternRewriter &rewriter) {
  auto op = rewriter.create<mlir::memref::AtomicRMWOp>(
      loc, mlir::arith::AtomicRMWKind::addi,
      createI64Constant(loc, rewriter, delta), header,
      mlir::ValueRange{createIndexConstant(loc, rewriter, refcountSlot)});
  threadsafe::Atomic::set(op.getOperation(), role,
                          delta > 0 ? ThreadSafetyAttrs::kOrderingMonotonic
                                    : ThreadSafetyAttrs::kOrderingAcqRel,
                          retainPremise);
  container::Header::markAtomicResource(op.getOperation(), header,
                                        refcountSlot);
  return op;
}

} // namespace lowering::refcount::managed_container::Atomic

namespace lowering::refcount::managed_container {

static bool borrowedEntryDescriptor(mlir::Value header) {
  return isEntryBorrowedValue(header);
}

mlir::LogicalResult inc(mlir::Location loc, mlir::Type logicalType,
                        mlir::ValueRange descriptor,
                        mlir::ConversionPatternRewriter &rewriter,
                        llvm::StringRef retainPremise,
                        bool verifiedOwnedToken) {
  auto refcountSlot = container::Refcount::slotForLogicalType(logicalType);
  if (!refcountSlot || descriptor.empty())
    return mlir::failure();

  mlir::Value header = descriptor.front();
  if (header.getDefiningOp<mlir::memref::AllocaOp>())
    return mlir::success();

  mlir::Value refcount = Atomic::load(loc, header, *refcountSlot, rewriter);
  mlir::Value zero = createI64Constant(loc, rewriter, 0);
  mlir::Value isManaged = rewriter.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::ne, refcount, zero);
  auto ifOp = rewriter.create<mlir::scf::IfOp>(loc, isManaged,
                                               /*withElseRegion=*/false);
  {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(ifOp.thenBlock());
    mlir::Value retain =
        Atomic::add(loc, header, *refcountSlot, 1,
                    ThreadSafetyAttrs::kRoleContainerRefcountRetain,
                    retainPremise, rewriter);
    if (retainPremise == ThreadSafetyAttrs::kPremiseOwnedToken &&
        verifiedOwnedToken)
      threadsafe::Retain::verifyOwnedToken(retain.getDefiningOp());
  }
  return mlir::success();
}

mlir::LogicalResult dec(mlir::Location loc, mlir::Type logicalType,
                        mlir::ValueRange descriptor, mlir::ModuleOp module,
                        mlir::ConversionPatternRewriter &rewriter,
                        const PyLLVMTypeConverter &typeConverter) {
  auto refcountSlot = container::Refcount::slotForLogicalType(logicalType);
  if (!refcountSlot || descriptor.empty())
    return mlir::failure();

  mlir::Value header = descriptor.front();
  if (borrowedEntryDescriptor(header))
    return mlir::success();
  if (header.getDefiningOp<mlir::memref::AllocaOp>()) {
    return container::Elements::refcount(loc, logicalType, descriptor, module,
                                         rewriter, typeConverter, "decref");
  }

  mlir::Value refcount = Atomic::load(loc, header, *refcountSlot, rewriter);
  mlir::Value zero = createI64Constant(loc, rewriter, 0);
  mlir::Value isManaged = rewriter.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::ne, refcount, zero);
  rewriter.create<mlir::cf::AssertOp>(
      loc, isManaged, "managed container decref observed zero refcount");
  mlir::Value previous = Atomic::add(
      loc, header, *refcountSlot, -1,
      ThreadSafetyAttrs::kRoleContainerRefcountRelease, {}, rewriter);
  mlir::Value one = createI64Constant(loc, rewriter, 1);
  mlir::Value shouldFree = rewriter.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::eq, previous, one);
  auto free = rewriter.create<mlir::scf::IfOp>(loc, shouldFree,
                                               /*withElseRegion=*/false);
  {
    mlir::OpBuilder::InsertionGuard freeGuard(rewriter);
    rewriter.setInsertionPointToStart(free.thenBlock());
    if (mlir::failed(container::Elements::refcount(loc, logicalType, descriptor,
                                                   module, rewriter,
                                                   typeConverter, "decref")))
      return mlir::failure();
    for (mlir::Value memref : descriptor)
      rewriter.create<mlir::memref::DeallocOp>(loc, memref);
  }
  return mlir::success();
}

} // namespace lowering::refcount::managed_container

namespace lowering::refcount::async_descriptor {

mlir::ValueRange expand(mlir::ValueRange descriptor,
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

mlir::Type payloadType(mlir::Type descriptorType) {
  if (auto coroType = mlir::dyn_cast<CoroutineType>(descriptorType))
    return coroType.getResultType();
  if (auto futureType = mlir::dyn_cast<FutureType>(descriptorType))
    return futureType.getResultType();
  if (auto taskType = mlir::dyn_cast<TaskType>(descriptorType))
    return taskType.getResultType();
  return {};
}

void drop(mlir::Location loc, mlir::Value asyncValue,
          mlir::ConversionPatternRewriter &rewriter) {
  if (!mlir::isa<mlir::async::ValueType>(asyncValue.getType()))
    return;
  rewriter.create<mlir::async::RuntimeDropRefOp>(loc, asyncValue,
                                                 rewriter.getI64IntegerAttr(1));
}

llvm::SmallVector<mlir::Value>
unpackPayload(mlir::Location loc, mlir::Type logicalType, mlir::Value storage,
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
  return llvm::SmallVector<mlir::Value>(cast.getResults().begin(),
                                        cast.getResults().end());
}

mlir::LogicalResult drain(DecRefOp op, mlir::ValueRange descriptor,
                          mlir::ModuleOp module,
                          mlir::ConversionPatternRewriter &rewriter,
                          const PyLLVMTypeConverter &typeConverter,
                          mlir::Type payload, bool isTask,
                          mlir::Value asyncValue, mlir::Value exceptionCell) {
  auto asyncValueType =
      mlir::dyn_cast<mlir::async::ValueType>(asyncValue.getType());
  if (!asyncValueType)
    return op->emitError("async descriptor does not carry an async.value");

  rewriter.create<mlir::async::RuntimeAwaitOp>(op.getLoc(), asyncValue);
  bool payloadNeedsRelease = isPyOwnershipTrackedType(payload);
  if (payloadNeedsRelease) {
    auto isError = rewriter.create<mlir::async::RuntimeIsErrorOp>(
        op.getLoc(), rewriter.getI1Type(), asyncValue);
    mlir::Block *currentBlock = rewriter.getInsertionBlock();
    mlir::Block *afterBlock =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    mlir::Block *successBlock = rewriter.createBlock(afterBlock->getParent(),
                                                     afterBlock->getIterator());
    rewriter.setInsertionPointToEnd(currentBlock);
    auto branch = rewriter.create<mlir::cf::CondBranchOp>(
        op.getLoc(), isError.getIsError(), afterBlock, successBlock);
    branch->setAttr("ly.async.cleanup_error_check", rewriter.getUnitAttr());

    rewriter.setInsertionPointToStart(successBlock);
    auto load = rewriter.create<mlir::async::RuntimeLoadOp>(
        op.getLoc(), asyncValueType.getValueType(), asyncValue);
    llvm::SmallVector<mlir::Value> payloadValues = unpackPayload(
        op.getLoc(), payload, load.getResult(), typeConverter, rewriter);
    if (payloadValues.empty())
      return mlir::failure();
    if (mlir::failed(lowering::refcount::Value::dec(op.getLoc(), payload,
                                                    payloadValues, module,
                                                    rewriter, typeConverter)))
      return mlir::failure();
    rewriter.create<mlir::cf::BranchOp>(op.getLoc(), afterBlock);

    rewriter.setInsertionPointToStart(afterBlock);
  }

  if (mlir::failed(async_runtime::ExceptionCell::destroy(
          op.getLoc(), module, rewriter, typeConverter, exceptionCell)))
    return mlir::failure();
  if (isTask)
    rewriter.create<mlir::memref::DeallocOp>(op.getLoc(), descriptor[2]);
  drop(op.getLoc(), asyncValue, rewriter);
  return mlir::success();
}

} // namespace lowering::refcount::async_descriptor

namespace lowering::refcount::Value {

mlir::LogicalResult dec(mlir::Location loc, mlir::Type logicalType,
                        mlir::ValueRange values, mlir::ModuleOp module,
                        mlir::ConversionPatternRewriter &rewriter,
                        const PyLLVMTypeConverter &typeConverter) {
  if (!isPyOwnershipTrackedType(logicalType))
    return mlir::success();

  if (mlir::isa<IntType>(logicalType) && values.size() == 3) {
    RuntimeAPI runtime(module, rewriter, typeConverter);
    auto release = runtime.call(loc, RuntimeSymbols::kLongDecRef,
                                /*resultType=*/nullptr, values);
    release->setAttr(OwnershipContractAttrs::kAggregateRelease,
                     rewriter.getUnitAttr());
    return mlir::success();
  }

  if (mlir::isa<StrType>(logicalType) && values.size() == 2) {
    RuntimeAPI runtime(module, rewriter, typeConverter);
    auto release = runtime.call(loc, RuntimeSymbols::kUnicodeDecRef,
                                /*resultType=*/nullptr, values);
    release->setAttr(OwnershipContractAttrs::kAggregateRelease,
                     rewriter.getUnitAttr());
    return mlir::success();
  }

  if (isCompilerOwnedMemRefListType(logicalType) ||
      isCompilerOwnedMemRefDictType(logicalType) ||
      isCompilerOwnedMemRefTupleType(logicalType)) {
    return lowering::refcount::managed_container::dec(
        loc, logicalType, values, module, rewriter, typeConverter);
  }

  if (auto classType = mlir::dyn_cast<ClassType>(logicalType)) {
    if (values.empty())
      return mlir::failure();
    return lowering::refcount::Class::emit(
        rewriter.getInsertionBlock()->getParentOp(), values, classType, module,
        rewriter, typeConverter, "decref");
  }

  if (mlir::isa<ExceptionType>(logicalType)) {
    if (values.size() == 3) {
      RuntimeAPI runtime(module, rewriter, typeConverter);
      auto release = runtime.call(loc, RuntimeSymbols::kExceptionDecRef,
                                  /*resultType=*/nullptr, values);
      release->setAttr(OwnershipContractAttrs::kAggregateRelease,
                       rewriter.getUnitAttr());
      return mlir::success();
    }
    if (values.size() == 1) {
      return async_runtime::ExceptionCell::releasePayload(
          loc, module, rewriter, typeConverter, values.front());
    }
    return mlir::failure();
  }

  if (mlir::isa<FuncType, PrimFuncType>(logicalType))
    return mlir::success();

  return mlir::failure();
}

} // namespace lowering::refcount::Value

namespace lowering::refcount::async_descriptor {

mlir::LogicalResult dec(DecRefOp op, mlir::ValueRange descriptor,
                        mlir::ModuleOp module,
                        mlir::ConversionPatternRewriter &rewriter,
                        const PyLLVMTypeConverter &typeConverter) {
  mlir::Type payload = payloadType(op.getObject().getType());
  if (!payload)
    return mlir::failure();

  bool isTask = mlir::isa<TaskType>(op.getObject().getType());
  unsigned expectedParts = isTask ? 3 : 2;
  if (descriptor.size() != expectedParts)
    return op->emitError("async descriptor refcount lowering received ")
           << descriptor.size() << " parts, expected " << expectedParts;

  mlir::Value asyncValue = descriptor.front();
  mlir::Value exceptionCell = descriptor[1];
  auto asyncValueType =
      mlir::dyn_cast<mlir::async::ValueType>(asyncValue.getType());
  if (!asyncValueType)
    return op->emitError("async descriptor does not carry an async.value");

  if (op->hasAttr("ly.async.gather_drain_descriptor"))
    return lowering::refcount::async_descriptor::drain(
        op, descriptor, module, rewriter, typeConverter, payload, isTask,
        asyncValue, exceptionCell);

  if (op->hasAttr("ly.async.await_consumed_descriptor")) {
    if (mlir::failed(async_runtime::ExceptionCell::destroy(
            op.getLoc(), module, rewriter, typeConverter, exceptionCell)))
      return mlir::failure();
    if (isTask)
      rewriter.create<mlir::memref::DeallocOp>(op.getLoc(), descriptor[2]);
    drop(op.getLoc(), asyncValue, rewriter);
    return mlir::success();
  }

  rewriter.create<mlir::async::RuntimeAwaitOp>(op.getLoc(), asyncValue);
  bool payloadNeedsRelease = isPyOwnershipTrackedType(payload);
  if (payloadNeedsRelease) {
    auto isError = rewriter.create<mlir::async::RuntimeIsErrorOp>(
        op.getLoc(), rewriter.getI1Type(), asyncValue);
    mlir::Block *currentBlock = rewriter.getInsertionBlock();
    mlir::Block *afterBlock =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    mlir::Block *successBlock = rewriter.createBlock(afterBlock->getParent(),
                                                     afterBlock->getIterator());
    rewriter.setInsertionPointToEnd(currentBlock);
    auto branch = rewriter.create<mlir::cf::CondBranchOp>(
        op.getLoc(), isError.getIsError(), afterBlock, successBlock);
    branch->setAttr("ly.async.cleanup_error_check", rewriter.getUnitAttr());

    rewriter.setInsertionPointToStart(successBlock);
    auto load = rewriter.create<mlir::async::RuntimeLoadOp>(
        op.getLoc(), asyncValueType.getValueType(), asyncValue);
    llvm::SmallVector<mlir::Value> payloadValues = unpackPayload(
        op.getLoc(), payload, load.getResult(), typeConverter, rewriter);
    if (payloadValues.empty())
      return mlir::failure();
    if (mlir::failed(lowering::refcount::Value::dec(op.getLoc(), payload,
                                                    payloadValues, module,
                                                    rewriter, typeConverter)))
      return mlir::failure();
    rewriter.create<mlir::cf::BranchOp>(op.getLoc(), afterBlock);

    rewriter.setInsertionPoint(op);
  }

  if (mlir::failed(async_runtime::ExceptionCell::destroy(
          op.getLoc(), module, rewriter, typeConverter, exceptionCell)))
    return mlir::failure();
  if (isTask)
    rewriter.create<mlir::memref::DeallocOp>(op.getLoc(), descriptor[2]);
  drop(op.getLoc(), asyncValue, rewriter);
  return mlir::success();
}

} // namespace lowering::refcount::async_descriptor

namespace lowering::refcount::task_cancel {

bool canCleanupPayload(mlir::Type logicalType,
                       const PyLLVMTypeConverter &typeConverter) {
  if (!isPyOwnershipTrackedType(logicalType))
    return true;

  llvm::SmallVector<mlir::Type> convertedTypes;
  if (mlir::failed(typeConverter.convertType(logicalType, convertedTypes)) ||
      convertedTypes.size() != 1)
    return false;

  mlir::Type convertedType = convertedTypes.front();
  return object_abi::Type::isStorageLike(convertedType);
}

bool isFreshCleanupWitness(DecRefOp op,
                           const PyLLVMTypeConverter &typeConverter) {
  auto taskCreate = op.getObject().getDefiningOp<TaskCreateOp>();
  if (!taskCreate)
    return false;

  auto taskType = mlir::dyn_cast<TaskType>(op.getObject().getType());
  if (!taskType || !canCleanupPayload(taskType.getResultType(), typeConverter))
    return false;

  auto coroCreate = taskCreate.getCoroutine().getDefiningOp<CoroCreateOp>();
  if (!coroCreate || !coroCreate.getArgs().empty())
    return false;

  bool hasCancel = false;
  for (mlir::Operation *user : taskCreate.getResult().getUsers()) {
    if (user == op.getOperation())
      continue;
    if (mlir::isa<DecRefOp>(user))
      continue;
    if (mlir::isa<TaskCancelOp>(user)) {
      hasCancel = true;
      continue;
    }
    return false;
  }
  return hasCancel;
}

} // namespace lowering::refcount::task_cancel

struct IncRefLowering : public mlir::OpConversionPattern<IncRefOp> {
  IncRefLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<IncRefOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(IncRefOp op, OneToNOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    lowering::refcount::InsertionPoint::set(op, rewriter);
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return mlir::failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    if (lowering::refcount::Immortal::known(op.getObject())) {
      rewriter.eraseOp(op);
      return mlir::success();
    }
    if (isPyType(op.getObject().getType()) &&
        !isPyOwnershipTrackedType(op.getObject().getType())) {
      rewriter.eraseOp(op);
      return mlir::success();
    }
    llvm::StringRef retainPremise = ThreadSafetyAttrs::kPremiseOwnedToken;
    if (auto premise = op->getAttrOfType<mlir::StringAttr>(
            ThreadSafetyAttrs::kRetainPremise))
      retainPremise = premise.getValue();
    if (isCompilerOwnedMemRefListType(op.getObject().getType()) ||
        isCompilerOwnedMemRefDictType(op.getObject().getType()) ||
        isCompilerOwnedMemRefTupleType(op.getObject().getType())) {
      if (mlir::failed(lowering::refcount::managed_container::inc(
              op.getLoc(), op.getObject().getType(), adaptor.getObject(),
              rewriter, retainPremise,
              op->hasAttr(ThreadSafetyAttrs::kOwnedTokenVerified))))
        return mlir::failure();
      rewriter.eraseOp(op);
      return mlir::success();
    }
    if (mlir::isa<FuncType, PrimFuncType>(op.getObject().getType())) {
      rewriter.eraseOp(op);
      return mlir::success();
    }
    if (mlir::isa<CoroutineType, TaskType, FutureType>(
            op.getObject().getType()))
      return op->emitError("async descriptors are linear resources; retaining "
                           "a borrowed awaitable is not supported");
    if (auto classType = mlir::dyn_cast<ClassType>(op.getObject().getType())) {
      (void)typeConverter;
      if (adaptor.getObject().empty())
        return mlir::failure();
      if (mlir::failed(lowering::refcount::Class::emit(
              op, adaptor.getObject(), classType, module, rewriter,
              *typeConverter, "incref", retainPremise,
              op->hasAttr(ThreadSafetyAttrs::kOwnedTokenVerified))))
        return mlir::failure();
      rewriter.eraseOp(op);
      return mlir::success();
    }
    if (mlir::isa<ExceptionType>(op.getObject().getType())) {
      if (adaptor.getObject().size() == 3) {
        RuntimeAPI runtime(module, rewriter, *typeConverter);
        auto retain =
            runtime.call(op.getLoc(), RuntimeSymbols::kIncRef,
                         /*resultType=*/nullptr,
                         mlir::ValueRange{adaptor.getObject().front()});
        threadsafe::Retain::premise(retain.getOperation(), retainPremise);
        if (retainPremise == ThreadSafetyAttrs::kPremiseOwnedToken &&
            op->hasAttr(ThreadSafetyAttrs::kOwnedTokenVerified))
          threadsafe::Retain::verifyOwnedToken(retain.getOperation());
      } else if (adaptor.getObject().size() == 1) {
        if (mlir::failed(async_runtime::ExceptionCell::retainPayload(
                op.getLoc(), module, rewriter, *typeConverter,
                adaptor.getObject().front(), retainPremise)))
          return mlir::failure();
      } else {
        return mlir::failure();
      }
      rewriter.eraseOp(op);
      return mlir::success();
    }
    if ((mlir::isa<IntType>(op.getObject().getType()) &&
         adaptor.getObject().size() == 3) ||
        (mlir::isa<StrType>(op.getObject().getType()) &&
         adaptor.getObject().size() == 2)) {
      RuntimeAPI runtime(module, rewriter, *typeConverter);
      auto retain = runtime.call(op.getLoc(), RuntimeSymbols::kIncRef,
                                 /*resultType=*/nullptr,
                                 mlir::ValueRange{adaptor.getObject().front()});
      threadsafe::Retain::premise(retain.getOperation(), retainPremise);
      if (retainPremise == ThreadSafetyAttrs::kPremiseOwnedToken &&
          op->hasAttr(ThreadSafetyAttrs::kOwnedTokenVerified))
        threadsafe::Retain::verifyOwnedToken(retain.getOperation());
      rewriter.eraseOp(op);
      return mlir::success();
    }
    if (adaptor.getObject().size() == 1 &&
        object_abi::Type::isStorageLike(
            adaptor.getObject().front().getType())) {
      RuntimeAPI runtime(module, rewriter, *typeConverter);
      auto retain = runtime.call(op.getLoc(), RuntimeSymbols::kIncRef,
                                 /*resultType=*/nullptr, adaptor.getObject());
      threadsafe::Retain::premise(retain.getOperation(), retainPremise);
      if (retainPremise == ThreadSafetyAttrs::kPremiseOwnedToken &&
          op->hasAttr(ThreadSafetyAttrs::kOwnedTokenVerified))
        threadsafe::Retain::verifyOwnedToken(retain.getOperation());
      rewriter.eraseOp(op);
      return mlir::success();
    }
    return op->emitError("py.incref requires a typed memref descriptor; raw "
                         "pointer/object fallback is not part of the ABI");
  }
};

struct DecRefLowering : public mlir::OpConversionPattern<DecRefOp> {
  DecRefLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<DecRefOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(DecRefOp op, OneToNOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    lowering::refcount::InsertionPoint::set(op, rewriter);
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return mlir::failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    if (op->hasAttr("ly.ownership.lowered_witness")) {
      rewriter.eraseOp(op);
      return mlir::success();
    }
    if (lowering::refcount::Immortal::known(op.getObject())) {
      rewriter.eraseOp(op);
      return mlir::success();
    }
    if (isPyType(op.getObject().getType()) &&
        !isPyOwnershipTrackedType(op.getObject().getType())) {
      rewriter.eraseOp(op);
      return mlir::success();
    }
    if (isCompilerOwnedMemRefListType(op.getObject().getType()) ||
        isCompilerOwnedMemRefDictType(op.getObject().getType()) ||
        isCompilerOwnedMemRefTupleType(op.getObject().getType())) {
      if (mlir::failed(lowering::refcount::managed_container::dec(
              op.getLoc(), op.getObject().getType(), adaptor.getObject(),
              module, rewriter, *typeConverter)))
        return mlir::failure();
      rewriter.eraseOp(op);
      return mlir::success();
    }
    if (mlir::isa<FuncType, PrimFuncType>(op.getObject().getType())) {
      rewriter.eraseOp(op);
      return mlir::success();
    }
    if (mlir::isa<CoroutineType, TaskType, FutureType>(
            op.getObject().getType())) {
      llvm::SmallVector<mlir::Value> descriptorStorage;
      mlir::ValueRange descriptor =
          lowering::refcount::async_descriptor::expand(adaptor.getObject(),
                                                       descriptorStorage);
      if (mlir::failed(lowering::refcount::async_descriptor::dec(
              op, descriptor, module, rewriter, *typeConverter)))
        return mlir::failure();
      rewriter.eraseOp(op);
      return mlir::success();
    }
    if (auto classType = mlir::dyn_cast<ClassType>(op.getObject().getType())) {
      (void)typeConverter;
      llvm::StringRef suffix = op->hasAttr("ly.final_local_class_decref")
                                   ? "destroy_local"
                                   : "decref";
      if (adaptor.getObject().empty())
        return mlir::failure();
      if (mlir::failed(lowering::refcount::Class::emit(
              op, adaptor.getObject(), classType, module, rewriter,
              *typeConverter, suffix)))
        return mlir::failure();
      rewriter.eraseOp(op);
      return mlir::success();
    }
    if (mlir::isa<ExceptionType>(op.getObject().getType())) {
      if (adaptor.getObject().size() == 3) {
        RuntimeAPI runtime(module, rewriter, *typeConverter);
        auto release =
            runtime.call(op.getLoc(), RuntimeSymbols::kExceptionDecRef,
                         /*resultType=*/nullptr, adaptor.getObject());
        release->setAttr(OwnershipContractAttrs::kAggregateRelease,
                         rewriter.getUnitAttr());
      } else if (adaptor.getObject().size() == 1) {
        if (mlir::failed(async_runtime::ExceptionCell::releasePayload(
                op.getLoc(), module, rewriter, *typeConverter,
                adaptor.getObject().front(),
                op->hasAttr(OwnershipContractAttrs::kAggregateRelease))))
          return mlir::failure();
      } else {
        return mlir::failure();
      }
      rewriter.eraseOp(op);
      return mlir::success();
    }
    if (mlir::isa<IntType>(op.getObject().getType()) &&
        adaptor.getObject().size() == 3) {
      RuntimeAPI runtime(module, rewriter, *typeConverter);
      auto release = runtime.call(op.getLoc(), RuntimeSymbols::kLongDecRef,
                                  /*resultType=*/nullptr, adaptor.getObject());
      mlir::Attribute attr =
          op->getAttr(OwnershipContractAttrs::kAggregateRelease);
      release->setAttr(OwnershipContractAttrs::kAggregateRelease,
                       attr ? attr : rewriter.getUnitAttr());
      rewriter.eraseOp(op);
      return mlir::success();
    }
    if (mlir::isa<StrType>(op.getObject().getType()) &&
        adaptor.getObject().size() == 2) {
      RuntimeAPI runtime(module, rewriter, *typeConverter);
      auto release = runtime.call(op.getLoc(), RuntimeSymbols::kUnicodeDecRef,
                                  /*resultType=*/nullptr, adaptor.getObject());
      mlir::Attribute attr =
          op->getAttr(OwnershipContractAttrs::kAggregateRelease);
      release->setAttr(OwnershipContractAttrs::kAggregateRelease,
                       attr ? attr : rewriter.getUnitAttr());
      rewriter.eraseOp(op);
      return mlir::success();
    }
    return op->emitError("py.decref requires a typed memref descriptor; raw "
                         "pointer/object fallback is not part of the ABI");
  }
};

struct AsyncDecRefLowering : public mlir::OpConversionPattern<DecRefOp> {
  AsyncDecRefLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<DecRefOp>(converter, ctx,
                                            mlir::PatternBenefit(2)) {}

  mlir::LogicalResult
  rewriteAsync(DecRefOp op, mlir::ValueRange object,
               mlir::ConversionPatternRewriter &rewriter) const {
    if (!mlir::isa<CoroutineType, TaskType, FutureType>(
            op.getObject().getType()))
      return mlir::failure();

    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return mlir::failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    if (mlir::isa<TaskType>(op.getObject().getType()) &&
        lowering::refcount::task_cancel::isFreshCleanupWitness(
            op, *typeConverter)) {
      rewriter.eraseOp(op);
      return mlir::success();
    }
    llvm::SmallVector<mlir::Value> descriptorStorage;
    mlir::ValueRange descriptor =
        lowering::refcount::async_descriptor::expand(object, descriptorStorage);
    if (mlir::failed(lowering::refcount::async_descriptor::dec(
            op, descriptor, module, rewriter, *typeConverter)))
      return mlir::failure();
    rewriter.eraseOp(op);
    return mlir::success();
  }

  mlir::LogicalResult
  matchAndRewrite(DecRefOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<mlir::Value, 1> object{adaptor.getObject()};
    return rewriteAsync(op, mlir::ValueRange(object), rewriter);
  }

  mlir::LogicalResult
  matchAndRewrite(DecRefOp op, OneToNOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    return rewriteAsync(op, adaptor.getObject(), rewriter);
  }
};

} // namespace

namespace lowering::refcount::Patterns {
void populate(PyLLVMTypeConverter &typeConverter,
              mlir::RewritePatternSet &patterns) {
  auto *ctx = patterns.getContext();
  patterns.add<IncRefLowering, AsyncDecRefLowering, DecRefLowering>(
      typeConverter, ctx);
}
} // namespace lowering::refcount::Patterns

} // namespace py
