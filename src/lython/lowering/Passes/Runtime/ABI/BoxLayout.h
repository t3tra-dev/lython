#pragma once

// Physical layout of a payload slot: one i64 word per element, the entity --
// the address of the value's object (None's is the None object), or for an
// int or float
// the value itself with a nonzero low two bits (an "immediate";
// `__ly_slot_word_is_immediate` in objects/object.mlir has the encodings). A
// slot owns a reference exactly when its entity is an address. The runtime
// support module (RuntimeSupportBuilder) and every lower* TU that probes or
// rebuilds boxed payloads must agree on this; it is defined only here.
//
// ⛔ WHY THE POINTER WORDS ARE WORDS, since two other slots in this tree were
// changed to hold real pointers and this one cannot be.
//
// A box is a `memref<Nxi64>`, and a memref's element type cannot be a pointer:
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

#include "ClassIds.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"

#include <cstdint>

namespace py::lowering::box_abi {

// ⭐ ONE WORD: a slot is its entity alone. The entity is the None object's
// address for None (0 is a slot nothing was stored in), an immediate for an
// int or float that has one (low two bits nonzero: int
// `...1`, float `...10`; see `__ly_slot_word_is_immediate`), and otherwise the
// address of the value's object -- whose header word 1 is its class id. So the
// class is never stored beside the entity: `slotClassFromEntity` reads it the
// way `__ly_slot_class` does in the manifest.
//
// ⛔ NO CLASS WORD, and it was not a second copy of nothing: it was a second
// copy of the object's header word 1, or of what the tag says. ⛔ NO REFCOUNT
// WORD: a slot is not an object. ⛔ NO OWNED FLAG: a slot owns its entity
// exactly when the entity is an address. ⛔ NO HASH WORD: the dict and the set
// keep their entries' hashes in their own arrays, as CPython keeps them in the
// entry beside the key.
//
// A STANDALONE box (an `object` value, `memref<5xi64>`) is an object of its
// own: refcount in word 0, class id in word 1, entity in word 2 (kBoxClassWord,
// kBoxEntityWord), words 3 and 4 unused. A pointer to its word 2 is a slot.
inline constexpr std::int64_t kWordsPerBox = 1;
inline constexpr std::int64_t kEntityWord = 0;
inline constexpr std::int64_t kBoxClassWord = 1;
inline constexpr std::int64_t kBoxEntityWord = 2;
// A SOURCE CLASS INSTANCE's header keeps the address of its body -- the block
// its fields live in (Lowerer.h, classInstanceBody) -- in word 2, as a
// standalone box keeps its entity there.
inline constexpr std::int64_t kInstanceBodyWord = 2;
// A standalone `object` box's width.
inline constexpr std::int64_t kStandaloneBoxWords = 5;

inline mlir::MemRefType boxWordsType(mlir::Builder &builder) {
  return mlir::MemRefType::get({kStandaloneBoxWords}, builder.getI64Type());
}

// One slot of a payload array, viewed on its own.
inline mlir::MemRefType slotWordsType(mlir::Builder &builder) {
  return mlir::MemRefType::get({kWordsPerBox}, builder.getI64Type());
}

// The class id a slot's entity word names (`__ly_slot_class`): int or float
// by an immediate's tag, else the object's header word 1 (0 for None's).
inline mlir::Value slotClassFromEntity(mlir::OpBuilder &builder,
                                       mlir::Location loc, mlir::Value entity) {
  mlir::Type i64 = builder.getI64Type();
  auto constant = [&](std::int64_t value) {
    return mlir::arith::ConstantIntOp::create(builder, loc, value, 64)
        .getResult();
  };
  mlir::Value zero = constant(0);
  mlir::Value tag =
      mlir::arith::AndIOp::create(builder, loc, entity, constant(3));
  mlir::Value isObject = mlir::arith::CmpIOp::create(
      builder, loc, mlir::arith::CmpIPredicate::eq, tag, zero);
  mlir::Value intTag =
      mlir::arith::AndIOp::create(builder, loc, entity, constant(1));
  mlir::Value isInt = mlir::arith::CmpIOp::create(
      builder, loc, mlir::arith::CmpIPredicate::ne, intTag, zero);
  mlir::Value immediateClass = mlir::arith::SelectOp::create(
      builder, loc, isInt, constant(py::class_ids::of("builtins.int")),
      constant(py::class_ids::of("builtins.float")));
  mlir::Value isNull = mlir::arith::CmpIOp::create(
      builder, loc, mlir::arith::CmpIPredicate::eq, entity, zero);
  // ⛔ A branch, not a select: the load must not run for None or an
  // immediate, whose word is no address.
  mlir::Value readable =
      mlir::arith::AndIOp::create(builder, loc, isObject,
                                  mlir::arith::XOrIOp::create(
                                      builder, loc, isNull,
                                      mlir::arith::ConstantIntOp::create(
                                          builder, loc, 1, 1)));
  auto ifOp = mlir::scf::IfOp::create(
      builder, loc, mlir::TypeRange{i64}, readable, /*withElseRegion=*/true);
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
    mlir::Value ptr = mlir::LLVM::IntToPtrOp::create(
        builder, loc, mlir::LLVM::LLVMPointerType::get(builder.getContext()),
        entity);
    mlir::Value classPtr = mlir::LLVM::GEPOp::create(
        builder, loc, mlir::LLVM::LLVMPointerType::get(builder.getContext()),
        i64, ptr, llvm::ArrayRef<mlir::LLVM::GEPArg>{mlir::LLVM::GEPArg(1)});
    mlir::Value loaded =
        mlir::LLVM::LoadOp::create(builder, loc, i64, classPtr).getResult();
    mlir::scf::YieldOp::create(builder, loc, loaded);
    builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
    mlir::Value other = mlir::arith::SelectOp::create(
        builder, loc, isObject, zero, immediateClass);
    mlir::scf::YieldOp::create(builder, loc, other);
  }
  return ifOp.getResult(0);
}

// One word of the box at `slotBase` inside a container's payload array. Every
// runtime-mode container read -- subscript, iteration, dict value -- rebuilds
// its element from these words, so the addressing is written once here.
inline mlir::Value loadContainerBoxWord(mlir::OpBuilder &builder,
                                        mlir::Location loc, mlir::Value array,
                                        mlir::Value slotBase,
                                        std::int64_t wordIndex) {
  mlir::Value offset =
      mlir::arith::ConstantIntOp::create(builder, loc, wordIndex, 64);
  mlir::Value word =
      mlir::arith::AddIOp::create(builder, loc, slotBase, offset).getResult();
  mlir::Value index = mlir::arith::IndexCastOp::create(
                          builder, loc, builder.getIndexType(), word)
                          .getResult();
  return mlir::memref::LoadOp::create(builder, loc, array, index).getResult();
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
