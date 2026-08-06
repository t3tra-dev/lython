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

// ⛔ Same view, from a WORD. Every use is a place the payload's address was
// stored as an integer and has to be widened back, which the memory model
// says produces something outside it: `extract_aligned_pointer_as_index` is
// documented there as where provenance is lost.
//
// It is still correct for what it is used for -- a boxed element's payload
// slot holds a word, and reading it is the only way to reach the payload. The
// fix is not here; it is the slot, which should hold a pointer the way the
// exception chain node's now does. Until then this is the meter: a call to
// this is a launder, a call to `memrefFromBoxPointer` is not.
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
