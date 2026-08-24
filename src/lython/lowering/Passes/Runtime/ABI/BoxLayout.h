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

// ⭐ FIVE, WHICH IS THE FLOOR. Word 0 is the refcount, word 1 the class id,
// word 2 the entity, word 3 the owned flag the deallocators consult and word 4
// the cached hash the dict and set keep. Nothing else is left: a value's other
// physical lanes come from the entity's own block, which is what every contract
// that has any answers with `lane_words`.
//
// ⛔ IT WAS SIXTEEN, AND SEVEN OF THOSE WORDS WERE A SECOND COPY. Words [4, 10)
// cached a pointer and a size for each of five lanes -- 32 bytes per list slot,
// per dict key AND value, per tuple element, per set member, describing storage
// the entity's own block already describes. Every contract that is more than
// one physical value now answers `lane_words` from its first lane's address:
// `__ly_unicode_alloc` puts a str's code units at +24 of the header's block and
// records their length in the shape word, the exception taxonomy records its
// message in extended word 6, and the iterators record what they walk. Reading
// the block instead of a copy is also the defect this removes rather than
// re-finds: a cached lane goes stale when the payload reallocates.
//
// ⛔ AND THE LANE COUNT WAS A CLASS'S FIELD BUDGET, which is why narrowing used
// to cost capability. A class expanded to one handle per field plus its own, so
// three lanes took `class P: x, y, z: float` out of every container. Fields
// live in the instance BODY now and a class is one lane however many it has;
// what `objectPayloadHandleWords` still refuses is a UNION, whose members do
// not share an entity, so no single address names them.
inline constexpr std::int64_t kWordsPerBox = 5;
// Word 2 is the ENTITY: the address of the object's first physical value, and
// the only one a box keeps. Everything else a contract expands to is reached
// from it through that contract's `lane_words` primitive.
//
// A SOURCE CLASS INSTANCE's own header is a block of these words and reads word
// 2 as the address of its body -- the block its fields live in (Lowerer.h,
// classInstanceBody). That is the same reading: an instance's entity IS its
// body, and a box holding the instance points at the header rather than at it.
inline constexpr std::int64_t kEntityWord = 2;
inline constexpr std::int64_t kOwnedFlagWord = 3;
inline constexpr std::int64_t kHashWord = 4;

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
