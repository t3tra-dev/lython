#include "Verifier.h"

#include "Common/ClassLayout.h"
#include "Common/LoweringUtils.h"
#include "Common/Object.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace py::threadsafe {

namespace {

static mlir::LogicalResult
verifyNoObjectHeaderPointerExtraction(mlir::ModuleOp module) {
  mlir::Operation *offender = nullptr;
  module.walk([&](mlir::Operation *op) {
    if (op->getName().getStringRef() !=
        "memref.extract_aligned_pointer_as_index")
      return mlir::WalkResult::advance();
    if (op->getNumOperands() == 0 ||
        !object_header::provenance(op->getOperand(0)))
      return mlir::WalkResult::advance();
    offender = op;
    return mlir::WalkResult::interrupt();
  });
  if (!offender)
    return mlir::success();
  return offender->emitOpError(
      "must not observe an object header as a raw pointer/index; use "
      "proven header memref operations instead");
}

static mlir::LogicalResult verifyObjectHeaderProducerCall(mlir::Operation *op) {
  bool hasObjectHeader = op->hasAttr(OwnershipContractAttrs::kObjectHeader);
  mlir::SmallVector<int64_t> ownedResults =
      attrs::i64Array(op, OwnershipContractAttrs::kOwnedResults);
  bool ownsObjectHeaderResult = false;
  for (int64_t index : ownedResults) {
    if (index < 0 || static_cast<unsigned>(index) >= op->getNumResults())
      continue;
    if (object_header::type(
            op->getResult(static_cast<unsigned>(index)).getType())) {
      ownsObjectHeaderResult = true;
      break;
    }
  }

  if (ownsObjectHeaderResult && !hasObjectHeader)
    return op->emitOpError()
           << "owned object-header result lacks explicit "
           << OwnershipContractAttrs::kObjectHeader << " producer provenance";

  return mlir::success();
}

static mlir::LogicalResult verifyClassCarrierContracts(mlir::ModuleOp module) {
  mlir::Operation *offender = nullptr;
  llvm::StringRef reason;

  auto hasCarrierProvenance = [](mlir::Value value) {
    mlir::Operation *def = value ? value.getDefiningOp() : nullptr;
    return def && class_layout::isObjectCarrierType(value.getType()) &&
           (def->hasAttr(ClassSafetyAttrs::kCarrierPack) ||
            def->hasAttr(ClassSafetyAttrs::kCarrierLoad));
  };
  auto hasCarrierPartProvenance = [](mlir::Operation *op) {
    return op && (op->hasAttr(ClassSafetyAttrs::kCarrierPart) ||
                  op->hasAttr(ClassSafetyAttrs::kCarrierPack));
  };

  module.walk([&](mlir::LLVM::ExtractValueOp op) {
    if (!hasCarrierProvenance(op.getContainer()))
      return mlir::WalkResult::advance();
    if (hasCarrierPartProvenance(op.getOperation()))
      return mlir::WalkResult::advance();
    offender = op.getOperation();
    reason = "class carrier part extraction lacks provenance";
    return mlir::WalkResult::interrupt();
  });
  if (offender)
    return offender->emitOpError(reason);

  module.walk([&](mlir::LLVM::InsertValueOp op) {
    if (!op->hasAttr(ClassSafetyAttrs::kCarrierPack))
      return mlir::WalkResult::advance();
    if (class_layout::isObjectCarrierType(op.getResult().getType()))
      return mlir::WalkResult::advance();
    offender = op.getOperation();
    reason = "class carrier pack is attached to a non-carrier value";
    return mlir::WalkResult::interrupt();
  });
  if (offender)
    return offender->emitOpError(reason);

  module.walk([&](mlir::LLVM::InsertValueOp op) {
    if (!op->hasAttr(ClassSafetyAttrs::kCarrierPack))
      return mlir::WalkResult::advance();
    auto part = attrs::i64(op.getOperation(), ClassSafetyAttrs::kCarrierPart);
    if (!part) {
      offender = op.getOperation();
      reason = "class carrier pack lacks part index";
      return mlir::WalkResult::interrupt();
    }
    if (*part == class_layout::Object::kHeaderIndex) {
      if (mlir::failed(class_layout::DescriptorShape::verify(
              op.getOperation(), object_abi::Header::owned(op.getContext()),
              "class carrier header descriptor"))) {
        offender = op.getOperation();
        reason = "";
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    }
    if (!class_layout::DescriptorShape::has(op.getOperation()))
      return mlir::WalkResult::advance();
    auto width = attrs::i64(op.getOperation(),
                            ClassSafetyAttrs::kCarrierPartElementWidth);
    auto size =
        attrs::i64(op.getOperation(), ClassSafetyAttrs::kCarrierPartStaticSize);
    if (width && size && *width == 64 && *size < 0) {
      offender = op.getOperation();
      reason = "class carrier payload uses dynamic memref<?xi64> without a "
               "more precise payload shape";
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  if (offender)
    return reason.empty() ? mlir::failure() : offender->emitOpError(reason);

  module.walk([&](mlir::memref::LoadOp op) {
    if (!op->hasAttr(ClassSafetyAttrs::kCarrierLoad))
      return mlir::WalkResult::advance();
    if (!class_layout::isObjectCarrierType(op.getResult().getType())) {
      offender = op.getOperation();
      reason = "class carrier load is attached to a non-carrier value";
      return mlir::WalkResult::interrupt();
    }
    if (op->hasAttr(OwnershipContractAttrs::kAggregateSlotLoad))
      return mlir::WalkResult::advance();
    offender = op.getOperation();
    reason = "class carrier load lacks aggregate-slot provenance";
    return mlir::WalkResult::interrupt();
  });
  if (offender)
    return offender->emitOpError(reason);

  return mlir::success();
}

static bool dynamicI64MemRef(mlir::Type type) {
  auto memref = mlir::dyn_cast<mlir::MemRefType>(type);
  if (!memref || memref.getRank() != 1 ||
      !memref.getElementType().isInteger(64))
    return false;
  return !memref.hasStaticShape() ||
         mlir::ShapedType::isDynamic(memref.getShape().front());
}

static bool hasDynamicI64SemanticRole(mlir::Value value) {
  if (!dynamicI64MemRef(value.getType()))
    return true;

  mlir::Operation *def = value.getDefiningOp();
  if (!def)
    return true;

  if (def->hasAttr(OwnershipContractAttrs::kObjectHeader) ||
      def->hasAttr(ClassSafetyAttrs::kPayloadPart) ||
      def->hasAttr(AsyncSafetyAttrs::kExceptionCell))
    return true;

  if (def->hasAttr(ContainerSafetyAttrs::kDescriptorGroup) &&
      def->hasAttr(ContainerSafetyAttrs::kDescriptorKind) &&
      def->hasAttr(ContainerSafetyAttrs::kDescriptorComponent))
    return true;

  return false;
}

static mlir::LogicalResult
verifyDynamicI64MemRefAllocRoles(mlir::ModuleOp module) {
  mlir::Operation *offender = nullptr;

  module.walk([&](mlir::memref::AllocOp op) {
    if (hasDynamicI64SemanticRole(op.getResult()))
      return mlir::WalkResult::advance();
    offender = op.getOperation();
    return mlir::WalkResult::interrupt();
  });
  if (offender)
    return offender->emitOpError(
        "dynamic memref<?xi64> allocation lacks a semantic width contract; "
        "keep i64 only when the producer carries object/header, container "
        "descriptor, class payload, async cell, or another explicit width "
        "role; otherwise use a narrower element type");

  module.walk([&](mlir::memref::AllocaOp op) {
    if (hasDynamicI64SemanticRole(op.getResult()))
      return mlir::WalkResult::advance();
    offender = op.getOperation();
    return mlir::WalkResult::interrupt();
  });
  if (offender)
    return offender->emitOpError(
        "dynamic memref<?xi64> stack allocation lacks a semantic width "
        "contract; keep i64 only when the producer carries object/header, "
        "container descriptor, class payload, async cell, or another explicit "
        "width role; otherwise use a narrower element type");

  return mlir::success();
}

} // namespace

namespace verifier::function_like {

static mlir::LogicalResult verify(mlir::Operation *funcLike,
                                  mlir::Region &body) {
  if (body.empty())
    return mlir::success();

  bool failedAny = false;
  if (mlir::failed(verifier::container::BorrowRetain::dominance(funcLike)))
    failedAny = true;
  if (mlir::failed(verifier::container::Access::regions(funcLike)))
    failedAny = true;
  if (mlir::failed(verifier::container::Access::coverage(funcLike)))
    failedAny = true;
  if (mlir::failed(verifier::container::Access::final(funcLike)))
    failedAny = true;
  if (mlir::failed(verifier::container::DescriptorAccess::final(funcLike)))
    failedAny = true;
  if (mlir::failed(verifier::async_runtime::Cells::verify(funcLike)))
    failedAny = true;
  if (mlir::failed(verifier::async_runtime::Handles::balance(funcLike, body)))
    failedAny = true;

  if (auto llvmFunc = mlir::dyn_cast<mlir::LLVM::LLVMFuncOp>(funcLike)) {
    if (mlir::failed(verifier::class_helper::Incref::verify(llvmFunc)))
      failedAny = true;
    if (mlir::failed(verifier::class_helper::Decref::verify(llvmFunc)))
      failedAny = true;
    if (mlir::failed(verifier::class_helper::DestroyLocal::verify(llvmFunc)))
      failedAny = true;
    if (mlir::failed(verifier::class_helper::Promote::verify(llvmFunc)))
      failedAny = true;
  }
  return mlir::failure(failedAny);
}

} // namespace verifier::function_like

mlir::LogicalResult verifier::module::verify(mlir::ModuleOp module) {
  if (mlir::failed(
          lowering::verifyNoUnrealizedCasts(module, "thread-safety verifier")))
    return mlir::failure();
  if (mlir::failed(verifyNoObjectHeaderPointerExtraction(module)))
    return mlir::failure();
  if (mlir::failed(verifyClassCarrierContracts(module)))
    return mlir::failure();
  if (mlir::failed(verifyDynamicI64MemRefAllocRoles(module)))
    return mlir::failure();

  bool failedAny = false;
  module.walk([&](mlir::memref::AtomicRMWOp op) {
    if (mlir::failed(verifier::memref::AtomicRMW::verify(op)))
      failedAny = true;
  });
  module.walk([&](mlir::memref::GenericAtomicRMWOp op) {
    if (mlir::failed(verifier::memref::GenericAtomicRMW::verify(op)))
      failedAny = true;
  });
  module.walk([&](mlir::memref::StoreOp op) {
    if (mlir::failed(verifier::memref::Store::verify(op)))
      failedAny = true;
  });
  module.walk([&](mlir::memref::DeallocOp op) {
    if (mlir::failed(verifier::memref::Dealloc::verify(op)))
      failedAny = true;
  });
  module.walk([&](mlir::memref::AllocaOp op) {
    if (mlir::failed(verifier::memref::Alloca::verify(op)))
      failedAny = true;
  });
  module.walk([&](mlir::LLVM::AtomicRMWOp op) {
    if (mlir::failed(verifier::llvm::AtomicRMW::verify(op)))
      failedAny = true;
  });
  module.walk([&](mlir::LLVM::CallOp op) {
    if (mlir::failed(verifier::llvm::RetainCall::verify(op)))
      failedAny = true;
    if (mlir::failed(verifyObjectHeaderProducerCall(op.getOperation())))
      failedAny = true;
    if (mlir::failed(verifier::async_runtime::RefcountCall::verify(
            op.getOperation(), op.getOperands())))
      failedAny = true;
    if (mlir::failed(verifier::llvm::FreeCall::verify(op)))
      failedAny = true;
  });
  module.walk([&](mlir::func::CallOp op) {
    if (mlir::failed(verifyObjectHeaderProducerCall(op.getOperation())))
      failedAny = true;
    if (mlir::failed(verifier::async_runtime::RefcountCall::verify(
            op.getOperation(), op.getOperands())))
      failedAny = true;
  });
  module.walk([&](mlir::LLVM::StoreOp op) {
    if (mlir::failed(verifier::llvm::Store::verify(op)))
      failedAny = true;
  });
  module.walk([&](mlir::LLVM::LoadOp op) {
    if (mlir::failed(verifier::llvm::Load::verify(op)))
      failedAny = true;
  });
  module.walk([&](mlir::LLVM::AtomicCmpXchgOp op) {
    if (mlir::failed(verifier::llvm::CmpXchg::verify(op)))
      failedAny = true;
  });
  module.walk([&](mlir::func::FuncOp func) {
    if (!func.isExternal() && mlir::failed(verifier::function_like::verify(
                                  func.getOperation(), func.getBody())))
      failedAny = true;
  });
  module.walk([&](mlir::async::FuncOp func) {
    if (mlir::failed(verifier::function_like::verify(func.getOperation(),
                                                     func.getBody())))
      failedAny = true;
  });
  module.walk([&](mlir::LLVM::LLVMFuncOp func) {
    if (!func.isDeclaration() && mlir::failed(verifier::function_like::verify(
                                     func.getOperation(), func.getBody())))
      failedAny = true;
  });
  return mlir::failure(failedAny);
}

} // namespace py::threadsafe
