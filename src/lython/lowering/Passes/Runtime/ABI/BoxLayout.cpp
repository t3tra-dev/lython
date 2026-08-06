#include "Runtime/Core/Lowerer.h"

#include "Runtime/ABI/BoxLayout.h"

namespace py::lowering {

// Rank-1 memref over a payload a POINTER already addresses. The contract →
// physical shape relation is static, so the descriptor is assembled inline
// (llvm.insertvalue chain reconciled with the memref world by the standard
// unrealized-cast materialization). Borrow-only: the result aliases storage
// owned by the boxed element.
//
// This is the one place a descriptor is built in the lowering passes. Callers
// that hold a pointer come here; the word entry point below is the same thing
// with a launder in front of it, and keeping them apart is what makes the
// launder countable.
mlir::Value RuntimeBundleLowerer::memrefFromBoxPointer(mlir::OpBuilder &builder,
                                                       mlir::Location loc,
                                                       mlir::Value pointer,
                                                       mlir::Value sizeWord,
                                                       mlir::MemRefType type) {
  mlir::MLIRContext *context = builder.getContext();
  auto ptrType = mlir::LLVM::LLVMPointerType::get(context);
  mlir::Type i64 = builder.getI64Type();
  auto arrayType = mlir::LLVM::LLVMArrayType::get(i64, 1);
  auto descriptorType = mlir::LLVM::LLVMStructType::getLiteral(
      context, {ptrType, ptrType, i64, arrayType, arrayType});
  mlir::Value zero =
      mlir::arith::ConstantIntOp::create(builder, loc, 0, 64).getResult();
  mlir::Value one =
      mlir::arith::ConstantIntOp::create(builder, loc, 1, 64).getResult();
  mlir::Value size =
      type.hasStaticShape()
          ? mlir::arith::ConstantIntOp::create(builder, loc,
                                               type.getDimSize(0), 64)
                .getResult()
          : sizeWord;
  mlir::Value descriptor =
      mlir::LLVM::UndefOp::create(builder, loc, descriptorType);
  descriptor = mlir::LLVM::InsertValueOp::create(
      builder, loc, descriptor, pointer, llvm::ArrayRef<std::int64_t>{0});
  descriptor = mlir::LLVM::InsertValueOp::create(
      builder, loc, descriptor, pointer, llvm::ArrayRef<std::int64_t>{1});
  descriptor = mlir::LLVM::InsertValueOp::create(
      builder, loc, descriptor, zero, llvm::ArrayRef<std::int64_t>{2});
  descriptor = mlir::LLVM::InsertValueOp::create(
      builder, loc, descriptor, size, llvm::ArrayRef<std::int64_t>{3, 0});
  descriptor = mlir::LLVM::InsertValueOp::create(
      builder, loc, descriptor, one, llvm::ArrayRef<std::int64_t>{4, 0});
  return mlir::UnrealizedConversionCastOp::create(builder, loc,
                                                  mlir::TypeRange{type},
                                                  descriptor)
      .getResult(0);
}

// Same view, from a WORD, because a boxed payload slot holds one.
//
// ⛔ Do not "fix" this the way the exception chain node and the module-global
// cell were fixed. Those held a word by choice; a box cannot hold anything
// else. A box is a `memref<16xi64>` (BoxLayout.h) and MLIR refuses a pointer
// element type outright -- `memref<4x!llvm.ptr>` is "invalid memref element
// type", checked, not assumed. Every reference a boxed object owns is an
// address in an integer, and that is the memref dialect's constraint rather
// than this compiler's decision.
//
// What makes it sound is not silence but `descFromAlignedPointer`
// (Proof.MemRef.Dialect), the model's rule for exactly this: an address may
// become a descriptor again given that the allocation is live and the
// generation matches. Neither obligation is left to a claim in a comment --
// `Proof.RC.Address.site-address-recovers` discharges both from the refcount
// invariant, so for as long as a site holds an object, this cannot produce a
// dangling view and what it produces is that object.
//
// So the remaining meter is not "is this called" but "is the slot's reference
// still held", which is the affine-ownership verifier's question, not this
// function's -- and the theorem above is what makes that the only question
// left.
mlir::Value RuntimeBundleLowerer::memrefFromBoxWords(mlir::OpBuilder &builder,
                                                     mlir::Location loc,
                                                     mlir::Value pointerWord,
                                                     mlir::Value sizeWord,
                                                     mlir::MemRefType type) {
  mlir::Value pointer = mlir::LLVM::IntToPtrOp::create(
      builder, loc, mlir::LLVM::LLVMPointerType::get(builder.getContext()),
      pointerWord);
  return RuntimeBundleLowerer::memrefFromBoxPointer(builder, loc, pointer,
                                                    sizeWord, type);
}

} // namespace py::lowering
