#include "Passes/Runtime/MemRefToLLVM.h"

#include "Common/Container.h"
#include "Common/Object.h"
#include "Common/RuntimeSupport.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/MemRefToLLVM/AllocLikeConversion.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

namespace py {
namespace {

void copyDiscardableAttrs(mlir::Operation *from, mlir::Operation *to) {
  for (const mlir::NamedAttribute &attr : from->getDiscardableAttrs())
    to->setDiscardableAttr(attr.getName(), attr.getValue());
}

void markDescriptorData(mlir::Value value) {
  mlir::Operation *def = value ? value.getDefiningOp() : nullptr;
  if (!def)
    return;
  def->setAttr(ContainerSafetyAttrs::kDescriptorData,
               mlir::UnitAttr::get(def->getContext()));
}

void sealObjectRefcountMemRefProvenance(mlir::Operation *op,
                                        mlir::OpBuilder &builder) {
  op->setAttr(
      ThreadSafetyAttrs::kAtomicProvenance,
      builder.getStringAttr(ThreadSafetyAttrs::kProvenanceMemRefDescriptor));
  op->setAttr(ThreadSafetyAttrs::kAtomicMemRefComponent,
              builder.getStringAttr(ContainerSafetyAttrs::kComponentHeader));
  op->setAttr(ThreadSafetyAttrs::kAtomicMemRefSlot,
              builder.getI64IntegerAttr(object_abi::kRefcountSlot));
}

void copyAsyncCellAttrs(mlir::Operation *from, mlir::Value to) {
  mlir::Operation *def = to ? to.getDefiningOp() : nullptr;
  if (!from || !def)
    return;
  for (llvm::StringRef attrName :
       {AsyncSafetyAttrs::kExceptionCell, AsyncSafetyAttrs::kCancelFlag})
    if (from->hasAttr(attrName))
      def->setAttr(attrName, mlir::UnitAttr::get(def->getContext()));
}

void markAsyncCellAllocated(mlir::Operation *from, mlir::Value value) {
  mlir::Operation *def = value ? value.getDefiningOp() : nullptr;
  if (!from || !def || !from->hasAttr(AsyncSafetyAttrs::kExceptionCell))
    return;
  def->setAttr(AsyncSafetyAttrs::kExceptionCellAllocated,
               mlir::UnitAttr::get(def->getContext()));
}

bool isOwnedObjectStorageAlloc(mlir::memref::AllocOp alloc) {
  return alloc->hasAttr(OwnershipContractAttrs::kOwnedLocalObject) ||
         alloc->hasAttr(OwnershipContractAttrs::kObjectHeader);
}

struct ContractAllocOpLowering : public mlir::AllocLikeOpLLVMLowering {
  explicit ContractAllocOpLowering(const mlir::LLVMTypeConverter &converter)
      : mlir::AllocLikeOpLLVMLowering(mlir::memref::AllocOp::getOperationName(),
                                      converter, mlir::PatternBenefit(2)) {}

  std::tuple<mlir::Value, mlir::Value>
  allocateBuffer(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
                 mlir::Value sizeBytes, mlir::Operation *op) const override {
    auto alloc = mlir::cast<mlir::memref::AllocOp>(op);
    auto [allocatedPtr, alignedPtr] = allocateBufferManuallyAlign(
        rewriter, loc, sizeBytes, op, getAlignment(rewriter, loc, alloc));
    if (mlir::Operation *def = allocatedPtr.getDefiningOp())
      copyDiscardableAttrs(op, def);
    if (isOwnedObjectStorageAlloc(alloc)) {
      if (mlir::Operation *def = allocatedPtr.getDefiningOp())
        def->setAttr(OwnershipContractAttrs::kOwnedLocalObject,
                     rewriter.getUnitAttr());
    } else {
      ownership::Pointer::markNonObject(allocatedPtr);
      ownership::Pointer::markNonObject(alignedPtr);
    }
    copyAsyncCellAttrs(op, allocatedPtr);
    markAsyncCellAllocated(op, allocatedPtr);
    copyAsyncCellAttrs(op, alignedPtr);
    return {allocatedPtr, alignedPtr};
  }
};

struct ContractAllocaOpLowering : public mlir::AllocLikeOpLLVMLowering {
  explicit ContractAllocaOpLowering(const mlir::LLVMTypeConverter &converter)
      : mlir::AllocLikeOpLLVMLowering(
            mlir::memref::AllocaOp::getOperationName(), converter,
            mlir::PatternBenefit(2)) {
    setRequiresNumElements();
  }

  std::tuple<mlir::Value, mlir::Value>
  allocateBuffer(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
                 mlir::Value size, mlir::Operation *op) const override {
    auto alloca = mlir::cast<mlir::memref::AllocaOp>(op);
    mlir::Type elementType =
        typeConverter->convertType(alloca.getType().getElementType());
    unsigned addressSpace =
        *getTypeConverter()->getMemRefAddressSpace(alloca.getType());
    auto ptrType =
        mlir::LLVM::LLVMPointerType::get(rewriter.getContext(), addressSpace);
    auto lowered = rewriter.create<mlir::LLVM::AllocaOp>(
        loc, ptrType, elementType, size, alloca.getAlignment().value_or(0));
    copyDiscardableAttrs(op, lowered.getOperation());
    if (alloca->hasAttr(OwnershipContractAttrs::kOwnedLocalObject)) {
      lowered->setAttr(OwnershipContractAttrs::kOwnedLocalObject,
                       rewriter.getUnitAttr());
    } else {
      ownership::Pointer::markNonObject(lowered.getResult());
    }
    copyAsyncCellAttrs(op, lowered.getResult());
    markAsyncCellAllocated(op, lowered.getResult());
    return {lowered.getResult(), lowered.getResult()};
  }
};

struct ContainerAccessLoadOpLowering
    : public mlir::ConvertOpToLLVMPattern<mlir::memref::LoadOp> {
  using Base = mlir::ConvertOpToLLVMPattern<mlir::memref::LoadOp>;
  using Base::Base;

  mlir::LogicalResult match(mlir::memref::LoadOp op) const override {
    return isConvertibleAndHasIdentityMaps(op.getMemRefType())
               ? mlir::success()
               : mlir::failure();
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::LoadOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::MemRefType type = op.getMemRefType();
    mlir::Value dataPtr = getStridedElementPtr(
        op.getLoc(), type, adaptor.getMemref(), adaptor.getIndices(), rewriter);
    markDescriptorData(dataPtr);
    if (op->hasAttr(AsyncSafetyAttrs::kExceptionCell))
      async_runtime::ExceptionCell::mark(dataPtr);
    auto lowered = rewriter.replaceOpWithNewOp<mlir::LLVM::LoadOp>(
        op, typeConverter->convertType(type.getElementType()), dataPtr, 0,
        false, op.getNontemporal());
    copyDiscardableAttrs(op.getOperation(), lowered.getOperation());
    return mlir::success();
  }
};

struct ContainerAccessStoreOpLowering
    : public mlir::ConvertOpToLLVMPattern<mlir::memref::StoreOp> {
  using Base = mlir::ConvertOpToLLVMPattern<mlir::memref::StoreOp>;
  using Base::Base;

  mlir::LogicalResult match(mlir::memref::StoreOp op) const override {
    return isConvertibleAndHasIdentityMaps(op.getMemRefType())
               ? mlir::success()
               : mlir::failure();
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::StoreOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::MemRefType type = op.getMemRefType();
    mlir::Value dataPtr = getStridedElementPtr(
        op.getLoc(), type, adaptor.getMemref(), adaptor.getIndices(), rewriter);
    markDescriptorData(dataPtr);
    if (op->hasAttr(AsyncSafetyAttrs::kExceptionCellPayloadStore) ||
        op->hasAttr(AsyncSafetyAttrs::kExceptionCell))
      async_runtime::ExceptionCell::mark(dataPtr);
    auto lowered = rewriter.replaceOpWithNewOp<mlir::LLVM::StoreOp>(
        op, adaptor.getValue(), dataPtr, 0, false, op.getNontemporal());
    copyDiscardableAttrs(op.getOperation(), lowered.getOperation());
    return mlir::success();
  }
};

std::optional<mlir::LLVM::AtomicBinOp>
matchSimpleAtomicOp(mlir::arith::AtomicRMWKind kind) {
  switch (kind) {
  case mlir::arith::AtomicRMWKind::addf:
    return mlir::LLVM::AtomicBinOp::fadd;
  case mlir::arith::AtomicRMWKind::addi:
    return mlir::LLVM::AtomicBinOp::add;
  case mlir::arith::AtomicRMWKind::assign:
    return mlir::LLVM::AtomicBinOp::xchg;
  case mlir::arith::AtomicRMWKind::maximumf:
    return mlir::LLVM::AtomicBinOp::fmax;
  case mlir::arith::AtomicRMWKind::maxs:
    return mlir::LLVM::AtomicBinOp::max;
  case mlir::arith::AtomicRMWKind::maxu:
    return mlir::LLVM::AtomicBinOp::umax;
  case mlir::arith::AtomicRMWKind::minimumf:
    return mlir::LLVM::AtomicBinOp::fmin;
  case mlir::arith::AtomicRMWKind::mins:
    return mlir::LLVM::AtomicBinOp::min;
  case mlir::arith::AtomicRMWKind::minu:
    return mlir::LLVM::AtomicBinOp::umin;
  case mlir::arith::AtomicRMWKind::ori:
    return mlir::LLVM::AtomicBinOp::_or;
  case mlir::arith::AtomicRMWKind::andi:
    return mlir::LLVM::AtomicBinOp::_and;
  default:
    return std::nullopt;
  }
}

std::optional<mlir::LLVM::AtomicOrdering>
orderingFromContract(mlir::Operation *op) {
  auto attr =
      op->getAttrOfType<mlir::StringAttr>(ThreadSafetyAttrs::kAtomicOrdering);
  if (!attr)
    return std::nullopt;
  llvm::StringRef ordering = attr.getValue();
  if (ordering == ThreadSafetyAttrs::kOrderingMonotonic)
    return mlir::LLVM::AtomicOrdering::monotonic;
  if (ordering == ThreadSafetyAttrs::kOrderingAcquire)
    return mlir::LLVM::AtomicOrdering::acquire;
  if (ordering == ThreadSafetyAttrs::kOrderingRelease)
    return mlir::LLVM::AtomicOrdering::release;
  if (ordering == ThreadSafetyAttrs::kOrderingAcqRel)
    return mlir::LLVM::AtomicOrdering::acq_rel;
  if (ordering == ThreadSafetyAttrs::kOrderingSeqCst)
    return mlir::LLVM::AtomicOrdering::seq_cst;
  return std::nullopt;
}

struct ContractAtomicRMWOpLowering
    : public mlir::ConvertOpToLLVMPattern<mlir::memref::AtomicRMWOp> {
  using Base = mlir::ConvertOpToLLVMPattern<mlir::memref::AtomicRMWOp>;
  using Base::Base;

  mlir::LogicalResult match(mlir::memref::AtomicRMWOp op) const override {
    if (!matchSimpleAtomicOp(op.getKind()))
      return mlir::failure();
    return isConvertibleAndHasIdentityMaps(op.getMemRefType())
               ? mlir::success()
               : mlir::failure();
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::AtomicRMWOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto kind = matchSimpleAtomicOp(op.getKind());
    if (!kind)
      return mlir::failure();
    auto ordering = orderingFromContract(op.getOperation());
    if (!ordering) {
      ordering = mlir::LLVM::AtomicOrdering::acq_rel;
    }
    mlir::MemRefType type = op.getMemRefType();
    mlir::Value dataPtr = getStridedElementPtr(
        op.getLoc(), type, adaptor.getMemref(), adaptor.getIndices(), rewriter);
    markDescriptorData(dataPtr);
    auto lowered = rewriter.replaceOpWithNewOp<mlir::LLVM::AtomicRMWOp>(
        op, *kind, dataPtr, adaptor.getValue(), *ordering);
    copyDiscardableAttrs(op.getOperation(), lowered.getOperation());
    auto role =
        op->getAttrOfType<mlir::StringAttr>(ThreadSafetyAttrs::kAtomicRole);
    if (role && role.getValue() == ThreadSafetyAttrs::kRoleAsyncExceptionLoad) {
      async_runtime::ExceptionCell::mark(dataPtr);
      lowered->setAttr(AsyncSafetyAttrs::kExceptionCell,
                       mlir::UnitAttr::get(rewriter.getContext()));
    }
    if (role && role.getValue() == ThreadSafetyAttrs::kRoleAsyncCancelRequest) {
      lowered->setAttr(AsyncSafetyAttrs::kCancelFlag,
                       mlir::UnitAttr::get(rewriter.getContext()));
      async_runtime::CancelFlag::mark(dataPtr);
    }
    return mlir::success();
  }
};

struct ContractGenericAtomicRMWOpLowering
    : public mlir::ConvertOpToLLVMPattern<mlir::memref::GenericAtomicRMWOp> {
  using Base = mlir::ConvertOpToLLVMPattern<mlir::memref::GenericAtomicRMWOp>;
  using Base::Base;

  mlir::LogicalResult
  match(mlir::memref::GenericAtomicRMWOp op) const override {
    return isConvertibleAndHasIdentityMaps(op.getMemRefType())
               ? mlir::success()
               : mlir::failure();
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::GenericAtomicRMWOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::Type valueType = typeConverter->convertType(op.getResult().getType());
    mlir::MemRefType memrefType = op.getMemRefType();
    mlir::Type elementType =
        typeConverter->convertType(memrefType.getElementType());

    mlir::Block *initBlock = rewriter.getInsertionBlock();
    mlir::Block *loopBlock =
        rewriter.splitBlock(initBlock, mlir::Block::iterator(op));
    loopBlock->addArgument(valueType, loc);
    mlir::Block *endBlock =
        rewriter.splitBlock(loopBlock, mlir::Block::iterator(op)++);

    rewriter.setInsertionPointToEnd(initBlock);
    mlir::Value dataPtr = getStridedElementPtr(
        loc, memrefType, adaptor.getMemref(), adaptor.getIndices(), rewriter);
    markDescriptorData(dataPtr);
    auto role =
        op->getAttrOfType<mlir::StringAttr>(ThreadSafetyAttrs::kAtomicRole);
    auto initialLoad = rewriter.create<mlir::LLVM::LoadOp>(
        loc, elementType, dataPtr, /*alignment=*/8, /*isVolatile=*/false,
        /*isNonTemporal=*/false, /*isInvariant=*/false,
        /*isInvariantGroup=*/false, mlir::LLVM::AtomicOrdering::acquire);
    copyDiscardableAttrs(op.getOperation(), initialLoad.getOperation());
    if (role &&
        role.getValue() == ThreadSafetyAttrs::kRoleAsyncExceptionStore) {
      initialLoad->setAttr(
          ThreadSafetyAttrs::kAtomicRole,
          rewriter.getStringAttr(ThreadSafetyAttrs::kRoleAsyncExceptionLoad));
      initialLoad->setAttr(
          ThreadSafetyAttrs::kAtomicOrdering,
          rewriter.getStringAttr(ThreadSafetyAttrs::kOrderingAcquire));
      async_runtime::ExceptionCell::mark(dataPtr);
      initialLoad->setAttr(AsyncSafetyAttrs::kExceptionCell,
                           mlir::UnitAttr::get(rewriter.getContext()));
    } else {
      initialLoad->setAttr(
          ThreadSafetyAttrs::kAtomicRole,
          rewriter.getStringAttr(ThreadSafetyAttrs::kRoleObjectRefcountLoad));
      initialLoad->setAttr(
          ThreadSafetyAttrs::kAtomicOrdering,
          rewriter.getStringAttr(ThreadSafetyAttrs::kOrderingAcquire));
      sealObjectRefcountMemRefProvenance(initialLoad.getOperation(), rewriter);
    }
    rewriter.create<mlir::LLVM::BrOp>(
        loc, mlir::ValueRange{initialLoad.getResult()}, loopBlock);

    rewriter.setInsertionPointToStart(loopBlock);
    mlir::IRMapping mapping;
    mapping.map(op.getCurrentValue(), loopBlock->getArgument(0));
    mlir::Block &entryBlock = op.getRegion().front();
    for (mlir::Operation &nested : entryBlock.without_terminator()) {
      mlir::Operation *clone = rewriter.clone(nested, mapping);
      mapping.map(nested.getResults(), clone->getResults());
    }
    mlir::Value desired =
        mapping.lookup(entryBlock.getTerminator()->getOperand(0));

    auto cmpxchg = rewriter.create<mlir::LLVM::AtomicCmpXchgOp>(
        loc, dataPtr, loopBlock->getArgument(0), desired,
        mlir::LLVM::AtomicOrdering::acq_rel,
        mlir::LLVM::AtomicOrdering::acquire, llvm::StringRef(),
        /*alignment=*/8);
    copyDiscardableAttrs(op.getOperation(), cmpxchg.getOperation());
    if (role &&
        role.getValue() == ThreadSafetyAttrs::kRoleAsyncExceptionStore) {
      async_runtime::ExceptionCell::mark(dataPtr);
      cmpxchg->setAttr(AsyncSafetyAttrs::kExceptionCell,
                       mlir::UnitAttr::get(rewriter.getContext()));
    } else {
      sealObjectRefcountMemRefProvenance(cmpxchg.getOperation(), rewriter);
    }
    mlir::Value loaded =
        rewriter.create<mlir::LLVM::ExtractValueOp>(loc, cmpxchg, 0);
    mlir::Value success =
        rewriter.create<mlir::LLVM::ExtractValueOp>(loc, cmpxchg, 1);
    rewriter.create<mlir::LLVM::CondBrOp>(loc, success, endBlock,
                                          mlir::ValueRange{}, loopBlock,
                                          mlir::ValueRange{loaded});

    rewriter.setInsertionPointToEnd(endBlock);
    rewriter.replaceOp(op, loaded);
    return mlir::success();
  }
};

struct ContractExtractAlignedPointerAsIndexLowering
    : public mlir::ConvertOpToLLVMPattern<
          mlir::memref::ExtractAlignedPointerAsIndexOp> {
  using Base = mlir::ConvertOpToLLVMPattern<
      mlir::memref::ExtractAlignedPointerAsIndexOp>;
  using Base::Base;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::ExtractAlignedPointerAsIndexOp op,
                  OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value alignedPtr;
    mlir::Type sourceType = op.getSource().getType();
    if (mlir::isa<mlir::MemRefType>(sourceType)) {
      mlir::MemRefDescriptor descriptor(adaptor.getSource());
      alignedPtr = descriptor.alignedPtr(rewriter, op.getLoc());
    } else if (auto unrankedType =
                   mlir::dyn_cast<mlir::UnrankedMemRefType>(sourceType)) {
      auto elementPtrType = mlir::LLVM::LLVMPointerType::get(
          rewriter.getContext(), unrankedType.getMemorySpaceAsInt());
      mlir::UnrankedMemRefDescriptor descriptor(adaptor.getSource());
      mlir::Value descriptorPtr =
          descriptor.memRefDescPtr(rewriter, op.getLoc());
      alignedPtr = mlir::UnrankedMemRefDescriptor::alignedPtr(
          rewriter, op.getLoc(), *getTypeConverter(), descriptorPtr,
          elementPtrType);
    } else {
      return mlir::failure();
    }

    if (mlir::Operation *def = alignedPtr.getDefiningOp())
      copyDiscardableAttrs(op.getOperation(), def);
    rewriter.replaceOpWithNewOp<mlir::LLVM::PtrToIntOp>(
        op, getTypeConverter()->getIndexType(), alignedPtr);
    return mlir::success();
  }
};

struct ContractReinterpretCastOpLowering
    : public mlir::ConvertOpToLLVMPattern<mlir::memref::ReinterpretCastOp> {
  using Base = mlir::ConvertOpToLLVMPattern<mlir::memref::ReinterpretCastOp>;
  using Base::Base;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::ReinterpretCastOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (!mlir::isa<mlir::MemRefType>(op.getSource().getType()))
      return mlir::failure();
    auto targetType = mlir::cast<mlir::MemRefType>(op.getResult().getType());
    auto descriptorType = mlir::dyn_cast_or_null<mlir::LLVM::LLVMStructType>(
        typeConverter->convertType(targetType));
    if (!descriptorType)
      return mlir::failure();

    mlir::Location loc = op.getLoc();
    mlir::MemRefDescriptor descriptor =
        mlir::MemRefDescriptor::undef(rewriter, loc, descriptorType);

    mlir::Value allocatedPtr;
    mlir::Value alignedPtr;
    mlir::MemRefDescriptor sourceDescriptor(adaptor.getSource());
    allocatedPtr = sourceDescriptor.allocatedPtr(rewriter, loc);
    alignedPtr = sourceDescriptor.alignedPtr(rewriter, loc);
    descriptor.setAllocatedPtr(rewriter, loc, allocatedPtr);
    descriptor.setAlignedPtr(rewriter, loc, alignedPtr);

    if (op.isDynamicOffset(0))
      descriptor.setOffset(rewriter, loc, adaptor.getOffsets()[0]);
    else
      descriptor.setConstantOffset(rewriter, loc, op.getStaticOffset(0));

    unsigned dynSizeId = 0;
    unsigned dynStrideId = 0;
    for (unsigned index = 0, rank = targetType.getRank(); index < rank;
         ++index) {
      if (op.isDynamicSize(index))
        descriptor.setSize(rewriter, loc, index,
                           adaptor.getSizes()[dynSizeId++]);
      else
        descriptor.setConstantSize(rewriter, loc, index,
                                   op.getStaticSize(index));

      if (op.isDynamicStride(index))
        descriptor.setStride(rewriter, loc, index,
                             adaptor.getStrides()[dynStrideId++]);
      else
        descriptor.setConstantStride(rewriter, loc, index,
                                     op.getStaticStride(index));
    }

    mlir::Value descriptorValue = descriptor;
    if (mlir::Operation *def = descriptorValue.getDefiningOp())
      copyDiscardableAttrs(op.getOperation(), def);
    rewriter.replaceOp(op, descriptorValue);
    return mlir::success();
  }
};

struct ContractDeallocOpLowering
    : public mlir::ConvertOpToLLVMPattern<mlir::memref::DeallocOp> {
  using Base = mlir::ConvertOpToLLVMPattern<mlir::memref::DeallocOp>;
  using Base::Base;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::DeallocOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::FailureOr<mlir::LLVM::LLVMFuncOp> freeFunc =
        mlir::LLVM::lookupOrCreateFreeFn(op->getParentOfType<mlir::ModuleOp>());
    if (mlir::failed(freeFunc))
      return mlir::failure();

    mlir::Value allocatedPtr;
    if (auto unrankedTy = mlir::dyn_cast<mlir::UnrankedMemRefType>(
            op.getMemref().getType())) {
      auto elementPtrTy = mlir::LLVM::LLVMPointerType::get(
          rewriter.getContext(), unrankedTy.getMemorySpaceAsInt());
      allocatedPtr = mlir::UnrankedMemRefDescriptor::allocatedPtr(
          rewriter, op.getLoc(),
          mlir::UnrankedMemRefDescriptor(adaptor.getMemref())
              .memRefDescPtr(rewriter, op.getLoc()),
          elementPtrTy);
    } else {
      allocatedPtr = mlir::MemRefDescriptor(adaptor.getMemref())
                         .allocatedPtr(rewriter, op.getLoc());
    }

    auto lowered = rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(
        op, freeFunc.value(), allocatedPtr);
    copyDiscardableAttrs(op.getOperation(), lowered.getOperation());
    return mlir::success();
  }
};

} // namespace

namespace lowering::runtime::memref_to_llvm::Patterns {

void populate(PyLLVMTypeConverter &typeConverter,
              mlir::RewritePatternSet &patterns) {
  patterns.add<ContractAllocOpLowering, ContractAllocaOpLowering>(
      typeConverter);
  patterns.add<ContainerAccessLoadOpLowering, ContainerAccessStoreOpLowering,
               ContractAtomicRMWOpLowering, ContractGenericAtomicRMWOpLowering,
               ContractExtractAlignedPointerAsIndexLowering,
               ContractReinterpretCastOpLowering, ContractDeallocOpLowering>(
      typeConverter, mlir::PatternBenefit(2));
}

} // namespace lowering::runtime::memref_to_llvm::Patterns

} // namespace py
