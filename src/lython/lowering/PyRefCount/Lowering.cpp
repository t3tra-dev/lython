#include "Common/Container.h"
#include "Common/LoweringUtils.h"
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

mlir::LogicalResult emit(mlir::Operation *op, mlir::Value object,
                         ClassType classType, mlir::ModuleOp module,
                         mlir::ConversionPatternRewriter &rewriter,
                         llvm::StringRef suffix,
                         llvm::StringRef retainPremise = {},
                         bool verifiedOwnedToken = false) {
  auto ptrType = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
  auto helper = getOrInsertLLVMFunc(
      op->getLoc(), module, rewriter, getClassHelperName(classType, suffix),
      mlir::LLVM::LLVMVoidType::get(rewriter.getContext()), {ptrType});
  ownership::llvm_func::Contract::apply(helper, helper.getName());
  auto helperRef =
      mlir::SymbolRefAttr::get(module.getContext(), helper.getName());
  auto call = rewriter.create<mlir::LLVM::CallOp>(
      op->getLoc(), mlir::TypeRange{}, helperRef, mlir::ValueRange{object});
  if (suffix == "incref") {
    threadsafe::Retain::premise(call.getOperation(), retainPremise);
    if (retainPremise == ThreadSafetyAttrs::kPremiseOwnedToken &&
        verifiedOwnedToken)
      threadsafe::Retain::verifyOwnedToken(call.getOperation());
  }
  if (suffix == "destroy_local")
    call->setAttr(OwnershipContractAttrs::kLocalDestroy,
                  rewriter.getUnitAttr());
  return mlir::success();
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
  if (mlir::isa<NoneOp, FuncObjectOp, TupleEmptyOp, StrConstantOp,
                ExceptionNullOp, TracebackNullOp, LocationCurrentOp>(def))
    return true;
  if (auto classNew = mlir::dyn_cast<ClassNewOp>(def))
    return classNew.getClassNameAttr().getValue() == "Exception";
  if (auto cast = mlir::dyn_cast<mlir::UnrealizedConversionCastOp>(def))
    if (cast->getNumOperands() == 1)
      return known(cast.getOperand(0));
  if (auto upcast = mlir::dyn_cast<UpcastOp>(def)) {
    if (isPyOwnershipMaterializedObjectBridge(def))
      return false;
    return known(upcast.getInput());
  }
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
    container::Elements::refcount(loc, logicalType, descriptor, module,
                                  rewriter, typeConverter, "decref");
    return mlir::success();
  }

  mlir::Value refcount = Atomic::load(loc, header, *refcountSlot, rewriter);
  mlir::Value zero = createI64Constant(loc, rewriter, 0);
  mlir::Value isManaged = rewriter.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::ne, refcount, zero);
  auto managed = rewriter.create<mlir::scf::IfOp>(loc, isManaged,
                                                  /*withElseRegion=*/false);
  {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(managed.thenBlock());
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
      container::Elements::refcount(loc, logicalType, descriptor, module,
                                    rewriter, typeConverter, "decref");
      for (mlir::Value memref : descriptor)
        rewriter.create<mlir::memref::DeallocOp>(loc, memref);
    }
  }
  return mlir::success();
}

} // namespace lowering::refcount::managed_container

namespace lowering::refcount::async_descriptor {

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

  async_runtime::ExceptionCell::destroy(op.getLoc(), module, rewriter,
                                        typeConverter, exceptionCell);
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
        rewriter.getInsertionBlock()->getParentOp(), values.front(), classType,
        module, rewriter, "decref");
  }

  if (mlir::isa<FuncType, PrimFuncType>(logicalType))
    return mlir::success();

  if (values.size() != 1 ||
      !mlir::isa<mlir::LLVM::LLVMPointerType>(values.front().getType()))
    return mlir::failure();

  RuntimeAPI runtime(module, rewriter, typeConverter);
  runtime.call(loc, RuntimeSymbols::kDecRef, /*resultType=*/nullptr,
               mlir::ValueRange{values.front()});
  return mlir::success();
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
    async_runtime::ExceptionCell::destroy(op.getLoc(), module, rewriter,
                                          typeConverter, exceptionCell);
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

  async_runtime::ExceptionCell::destroy(op.getLoc(), module, rewriter,
                                        typeConverter, exceptionCell);
  if (isTask)
    rewriter.create<mlir::memref::DeallocOp>(op.getLoc(), descriptor[2]);
  drop(op.getLoc(), asyncValue, rewriter);
  return mlir::success();
}

} // namespace lowering::refcount::async_descriptor

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
              op, adaptor.getObject().front(), classType, module, rewriter,
              "incref", retainPremise,
              op->hasAttr(ThreadSafetyAttrs::kOwnedTokenVerified))))
        return mlir::failure();
      rewriter.eraseOp(op);
      return mlir::success();
    }
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
      if (mlir::failed(lowering::refcount::async_descriptor::dec(
              op, adaptor.getObject(), module, rewriter, *typeConverter)))
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
              op, adaptor.getObject().front(), classType, module, rewriter,
              suffix)))
        return mlir::failure();
      rewriter.eraseOp(op);
      return mlir::success();
    }
    RuntimeAPI runtime(module, rewriter, *typeConverter);

    runtime.call(op.getLoc(), RuntimeSymbols::kDecRef, /*resultType=*/nullptr,
                 adaptor.getObject());
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

} // namespace

namespace lowering::refcount::Patterns {
void populate(PyLLVMTypeConverter &typeConverter,
              mlir::RewritePatternSet &patterns) {
  auto *ctx = patterns.getContext();
  patterns.add<IncRefLowering, DecRefLowering>(typeConverter, ctx);
}
} // namespace lowering::refcount::Patterns

} // namespace py
