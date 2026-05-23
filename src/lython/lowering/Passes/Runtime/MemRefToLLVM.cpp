#include "Passes/Runtime/MemRefToLLVM.h"

#include "Common/Container.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/MemRefToLLVM/AllocLikeConversion.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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
    ownership::Pointer::markNonObject(allocatedPtr);
    ownership::Pointer::markNonObject(alignedPtr);
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
    ownership::Pointer::markNonObject(lowered.getResult());
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
    auto lowered = rewriter.replaceOpWithNewOp<mlir::LLVM::AtomicRMWOp>(
        op, *kind, dataPtr, adaptor.getValue(), *ordering);
    copyDiscardableAttrs(op.getOperation(), lowered.getOperation());
    auto role =
        op->getAttrOfType<mlir::StringAttr>(ThreadSafetyAttrs::kAtomicRole);
    if (role && role.getValue() == ThreadSafetyAttrs::kRoleAsyncCancelRequest) {
      lowered->setAttr(AsyncSafetyAttrs::kCancelFlag,
                       mlir::UnitAttr::get(rewriter.getContext()));
      async_runtime::CancelFlag::mark(dataPtr);
    }
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
               ContractAtomicRMWOpLowering, ContractDeallocOpLowering>(
      typeConverter, mlir::PatternBenefit(2));
}

} // namespace lowering::runtime::memref_to_llvm::Patterns

} // namespace py
