#pragma once

// Physical layout of a payload box slot: 12 i64 words per element. Words
// [4, 7) hold the pointer word of each physical memref (position i at
// kPointerWordBase + i), words [7, 10) the matching size words. The runtime
// support module (RuntimeSupportBuilder) and every lower* TU that probes or
// rebuilds boxed payloads must agree on these offsets; they are defined only
// here.
//
// ⛔ WHY THE POINTER WORDS ARE WORDS, since two other slots in this tree were
// changed to hold real pointers and this one cannot be.
//
// A box is a `memref<12xi64>`, and a memref's element type cannot be a pointer:
// MLIR rejects `memref<4x!llvm.ptr>` with "invalid memref element type"
// (checked with mlir-opt, not assumed). `memref<4xindex>` is accepted and is
// the same thing -- an integer. So an object graph cannot be built inside the
// memref dialect at all: every reference a boxed object owns is an address in
// an integer, and reading it back is `inttoptr` by construction. The exception
// chain node and the module-global cell were different -- they were LLVM
// globals and structs holding a word by choice, and they now hold pointers.
//
// This is governed rather than merely tolerated. `Proof.MemRef.Dialect` models
// the trip out (`extractAlignedPointerAsIndex`, yielding an identity rather
// than a number) and the trip back (`descFromAlignedPointer`), the second
// premised on the allocation being live and its generation current. This
// compiler discharges both structurally: the slot owns a retained reference,
// and `memref.realloc` appears nowhere, so no generation moves under a held
// word. `recovered-identity` is the theorem that what comes back names the same
// object -- which is what `field′` in the refcount layer means by "holds".
//
// The manifest signatures that spell the box as `memref<Nxi64>` are therefore
// not a migration waiting to happen. Changing them would mean pushing LLVM
// struct types through the manifest surface, and the reason to do it would have
// to be something other than the pointer words.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"

#include <cstdint>

namespace py::lowering::box_abi {

// ⭐ TWELVE, AND THE MEASUREMENT THAT SET IT. Words [0, 4) are the header
// (refcount, class id, entity pointer, lane count), [4, 7) the pointer word of
// each physical memref, [7, 10) the matching size words, word 10 the owned flag
// the deallocators consult and word 11 the cached hash the dict and set keep.
// A source class INSTANCE is a block with these words in front of it, and it
// reads word 3 as the address of its body rather than as a lane count.
//
// ⛔ IT WAS SIXTEEN, WITH ROOM FOR FIVE LANES, AND NO CONTRACT HAS MORE THAN
// THREE. Counted across every `ly.runtime.shape` in the manifests: 71 contracts
// expand to three physical values (the exception family), 3 to two, 18 to one.
// The two spare lanes were four words on every boxed element -- 32 bytes per
// list slot, per dict key AND value, per tuple element, per set member -- and
// `objectPayloadHandleWords` rejects a wider value with a diagnostic, so
// narrowing cannot silently truncate.
//
// ⛔ THE LANE COUNT IS NO LONGER A CLASS'S FIELD BUDGET. It was: a class
// expanded to one handle per field plus its own, so narrowing the lanes from
// five to three took `class P: x, y, z: float` out of every container. Fields
// now live in the instance BODY (Lowerer.h, classInstanceBody) and a class is
// one handle however many it has -- what still reaches
// `objectPayloadHandleWords` too wide is a UNION, whose members do not share a
// width and so cannot share a box.
//
// ⛔ AND TWELVE IS NOT THE FLOOR. The floor is FIVE, because the non-first lanes
// of every multi-lane contract are interior to the first one's allocation
// (`__ly_unicode_alloc` puts the bytes at +24 of the header's block, the way
// `__ly_long_parts` recovers an int's meta and digits at +16 and +32), so a box
// needs only the entity pointer and the lanes come back by contract. Getting
// there is one-laning what is left: 71 contracts (the exception family) expand
// to three, `builtins.str` and two `_asyncio` contracts to two.
inline constexpr std::int64_t kWordsPerBox = 12;
// Word 3 of a SOURCE CLASS INSTANCE's header holds the address of its body, the
// block every field's storage lives in (Lowerer.h, classInstanceBody). A
// container's box uses the same word for its lane count; the two never describe
// the same block, because an instance stored in a container is described by a
// box that POINTS at it.
inline constexpr std::int64_t kInstanceBodyWord = 3;
inline constexpr std::int64_t kPointerWordBase = 4;
inline constexpr std::int64_t kSizeWordBase = 7;
inline constexpr std::int64_t kPointerWordCount =
    kSizeWordBase - kPointerWordBase;
inline constexpr std::int64_t kOwnedFlagWord =
    kSizeWordBase + kPointerWordCount;

inline mlir::MemRefType boxWordsType(mlir::Builder &builder) {
  return mlir::MemRefType::get({kWordsPerBox}, builder.getI64Type());
}

// The stack slot for one transient payload box, placed in the entry block of
// the function the builder is currently writing into.
//
// Why NOT beside the call that consumes it, which is where the builder already
// points: `memref.alloca` outside a function's entry block becomes an
// `llvm.alloca` that LLVM classifies as dynamic (`AllocaInst::isStaticAlloca`
// requires the entry block), so it extends the frame at run time and nothing
// shrinks it before the function returns. Beside a call inside a loop that is
// kWordsPerBox * 8 bytes of frame per iteration.
//
// Why NOT leave it to an existing hoist: MLIR's buffer-loop hoisting matches
// loop-shaped *regions* and these loops are already unstructured `cf` blocks by
// phase 9, and SROA/mem2reg cannot touch a slot whose address is passed to a
// call. Neither runs on this at any optimization level.
//
// Why NOT one shared slot per function: two boxes are live at once wherever a
// key and a value are boxed for the same call, so the slot has to be per site.
// Reuse across executions of one site is safe for a different reason -- see
// RuntimeBundleLowerer::transientPayloadBox.
inline mlir::Value allocaBoxWords(mlir::OpBuilder &builder,
                                  mlir::Location loc) {
  mlir::MemRefType boxType = boxWordsType(builder);
  mlir::func::FuncOp function;
  mlir::Block *insertion = builder.getInsertionBlock();
  for (mlir::Operation *parent = insertion ? insertion->getParentOp() : nullptr;
       parent; parent = parent->getParentOp()) {
    if (auto candidate = mlir::dyn_cast<mlir::func::FuncOp>(parent)) {
      function = candidate;
      break;
    }
    // Why NOT keep walking: an entry block above an isolated-from-above
    // boundary does not dominate this insertion point, so hoisting across one
    // would produce a use before its definition rather than a smaller frame.
    if (parent->hasTrait<mlir::OpTrait::IsIsolatedFromAbove>())
      break;
  }
  if (!function || function.getBody().empty())
    return mlir::memref::AllocaOp::create(builder, loc, boxType).getResult();

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&function.getBody().front());
  return mlir::memref::AllocaOp::create(builder, loc, boxType).getResult();
}

} // namespace py::lowering::box_abi
