#include "Runtime/Core/Lowerer.h"

#include "Common/MemRef1D.h"

#include "Runtime/ABI/BoxLayout.h"

namespace py::lowering {

// Rank-1 memref over a payload a POINTER already addresses. The contract →
// physical shape relation is static, so the descriptor is assembled from
// constants; `Common/MemRef1D.h` does the assembling, shared with the runtime
// support builders. Borrow-only: the result aliases storage owned by the boxed
// element.
//
// Callers that hold a pointer come here; the word entry point below is the same
// thing with a launder in front of it, and keeping them apart is what makes the
// launder countable. What this is NOT any more is the only place a descriptor
// is built -- it said that until `buildMemRef1D` arrived for the exception
// triple and quietly falsified it.
//
// A box is a single allocation, so allocated and aligned are the same pointer
// and the offset is zero; `MemRef1DParts` keeps those separate fields precisely
// so that stays a statement about boxes rather than a shape baked into the
// assembler.
mlir::Value RuntimeBundleLowerer::memrefFromBoxPointer(mlir::OpBuilder &builder,
                                                       mlir::Location loc,
                                                       mlir::Value pointer,
                                                       mlir::Value sizeWord,
                                                       mlir::MemRefType type) {
  auto word = [&](std::int64_t value) {
    return mlir::arith::ConstantIntOp::create(builder, loc, value, 64)
        .getResult();
  };
  MemRef1DParts parts;
  parts.allocated = pointer;
  parts.aligned = pointer;
  parts.offset = word(0);
  parts.size = type.hasStaticShape() ? word(type.getDimSize(0)) : sizeWord;
  parts.stride = word(1);
  return buildMemRef1D(builder, loc, type, parts);
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
