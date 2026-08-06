#pragma once

// Physical layout of a payload box slot: 16 i64 words per element. Words
// [4, 9) hold the pointer word of each physical memref (position i at
// kPointerWordBase + i), words [9, 14) the matching size words. The runtime
// support module (RuntimeSupportBuilder) and every lower* TU that probes or
// rebuilds boxed payloads must agree on these offsets; they are defined only
// here.
//
// ⛔ WHY THE POINTER WORDS ARE WORDS, since two other slots in this tree were
// changed to hold real pointers and this one cannot be.
//
// A box is a `memref<16xi64>`, and a memref's element type cannot be a pointer:
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
// The 97 manifest signatures that spell `memref<16xi64>` are therefore not a
// migration waiting to happen. Changing them would mean pushing LLVM struct
// types through the manifest surface, and the reason to do it would have to be
// something other than the pointer words.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"

#include <cstdint>

namespace py::lowering::box_abi {

inline constexpr std::int64_t kWordsPerBox = 16;
inline constexpr std::int64_t kPointerWordBase = 4;
inline constexpr std::int64_t kSizeWordBase = 9;
// Up to five physical memrefs per boxed entity (pointer words [4, 9), size
// words [9, 14)); word 14 is the owned flag the deallocators consult.
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
