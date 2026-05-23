#include "Verifier.h"

namespace py::threadsafe {

namespace {

mlir::LogicalResult verifyNoUnrealizedCasts(mlir::ModuleOp module) {
  mlir::UnrealizedConversionCastOp offender = nullptr;
  module.walk([&](mlir::UnrealizedConversionCastOp cast) {
    offender = cast;
    return mlir::WalkResult::interrupt();
  });
  if (!offender)
    return mlir::success();
  return offender.emitError(
      "unrealized conversion cast reached thread-safety verifier");
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
  if (mlir::failed(verifyNoUnrealizedCasts(module)))
    return mlir::failure();

  bool failedAny = false;
  module.walk([&](mlir::memref::AtomicRMWOp op) {
    if (mlir::failed(verifier::memref::AtomicRMW::verify(op)))
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
    if (auto callee = op.getCallee())
      if (mlir::failed(verifier::async_runtime::RefcountCall::verify(
              op.getOperation(), *callee, op.getOperands())))
        failedAny = true;
    if (mlir::failed(verifier::llvm::FreeCall::verify(op)))
      failedAny = true;
  });
  module.walk([&](mlir::func::CallOp op) {
    if (mlir::failed(verifier::async_runtime::RefcountCall::verify(
            op.getOperation(), op.getCallee(), op.getOperands())))
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
