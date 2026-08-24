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
  auto word = [&](std::int64_t value) { return constantI64(builder, loc, value); };
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

// ⭐ A VALUE'S LANES OUT OF A BOX, FROM THE ENTITY WORD ALONE. Lane 0 IS the
// entity -- its pointer is word 2 and its width is its static type -- and every
// other lane comes from the contract's `lane_words` primitive, which reads them
// out of the entity's own block.
//
// Why NOT the cached pointer and size words the box used to carry: they are two
// words per lane on every boxed element, and they are a SECOND copy of what the
// block already says. The int narrowing made that concrete -- a box that cached
// a stale lane after a reallocation is the defect shape this removes by not
// having the copy.
mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>>
RuntimeBundleLowerer::lanesFromBoxEntity(mlir::OpBuilder &builder,
                                         mlir::Location loc,
                                         mlir::Value entityWord,
                                         llvm::ArrayRef<mlir::Type> laneTypes,
                                         llvm::StringRef contract,
                                         mlir::Operation *reporter) {
  llvm::SmallVector<mlir::Value, 4> lanes;
  if (laneTypes.empty())
    return lanes;
  auto head = mlir::dyn_cast<mlir::MemRefType>(laneTypes.front());
  if (!head || !head.hasStaticShape() || head.getRank() != 1)
    return reporter->emitError()
           << contract << " has no statically sized entity lane to rebuild a "
           << "box from, got " << laneTypes.front();
  mlir::Value headSize = mlir::arith::ConstantIntOp::create(
                             builder, loc, head.getDimSize(0), 64)
                             .getResult();
  lanes.push_back(RuntimeBundleLowerer::memrefFromBoxWords(
      builder, loc, entityWord, headSize, head));
  if (laneTypes.size() == 1)
    return lanes;

  std::optional<RuntimeSymbol> laneWords =
      RuntimeBundleLowerer::laneWordsPrimitiveFor(contract);
  if (!laneWords)
    return reporter->emitError()
           << contract << " expands to " << laneTypes.size()
           << " physical values but has no `lane_words` primitive, so a box "
              "holding it cannot hand back anything past its entity";
  mlir::func::CallOp call = mlir::func::CallOp::create(
      builder, loc, laneWords->function, mlir::ValueRange{entityWord});
  if (call.getNumResults() != 2 * (laneTypes.size() - 1))
    return laneWords->function.emitError()
           << "`lane_words` must answer with a pointer and a size for each "
              "lane past the entity: " << contract << " has "
           << (laneTypes.size() - 1) << " of them and this returns "
           << call.getNumResults() << " values";
  for (unsigned index = 1; index < laneTypes.size(); ++index) {
    auto memref = mlir::dyn_cast<mlir::MemRefType>(laneTypes[index]);
    if (!memref)
      return reporter->emitError()
             << contract << " lane " << index << " is not a memref, got "
             << laneTypes[index];
    lanes.push_back(RuntimeBundleLowerer::memrefFromBoxWords(
        builder, loc, call.getResult(2 * (index - 1)),
        call.getResult(2 * (index - 1) + 1), memref));
  }
  return lanes;
}

// The taxonomy shares one message layout, so the 71 exception contracts share
// `builtins.BaseException`'s primitive rather than each declaring its own.
//
// ⛔ MATCHED BY SHAPE AND NOT BY NAME. `exceptionAncestorContractFor` answers
// for a SOURCE class and says nothing about `_io.UnsupportedOperation` or the
// rest of the manifest's own taxonomy, and there are 71 of them. What they
// share is the thing that matters here: every one is allocated by
// `LyBaseException_New`, so every one has its physical shape, and a layout
// primitive is about the layout.
std::optional<RuntimeSymbol>
RuntimeBundleLowerer::laneWordsPrimitiveFor(llvm::StringRef contract) {
  if (std::optional<RuntimeSymbol> direct =
          manifest.primitive(contract, "lane_words"))
    return direct;
  // Recursive, because a SOURCE exception class answers with its immediate
  // taxonomy ancestor (`Where` -> `builtins.ValueError`) and that ancestor is
  // one of the 71 that declare nothing of their own.
  if (std::optional<std::string> ancestor =
          RuntimeBundleLowerer::exceptionAncestorContractFor(
              runtimeContractType(context, contract)))
    if (*ancestor != contract)
      if (std::optional<RuntimeSymbol> inherited =
              RuntimeBundleLowerer::laneWordsPrimitiveFor(*ancestor))
        return inherited;
  const RuntimeValueShape *shape = manifest.valueShape(contract);
  const RuntimeValueShape *exception =
      manifest.valueShape("builtins.BaseException");
  if (shape && exception && shape->valueTypes == exception->valueTypes)
    return manifest.primitive("builtins.BaseException", "lane_words");
  return std::nullopt;
}

} // namespace py::lowering
