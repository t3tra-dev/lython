#pragma once

// "Are words 0 and 1 of this handle the entity's refcount/class prefix, ALREADY
// STORED, at the earliest program point where the handle is available?"
//
// Two questions live here, and separating them is the whole point of the file,
// because the tree spent a release believing they were one:
//
//   typeCarriesHeaderPrefix    -- LAYOUT. Is the type wide enough to hold the
//                                 two-word prefix at all?
//   prefixIsInitializedAtDefinition -- PROVENANCE. Does the DEFINITION establish
//                                 those two words, or merely produce storage
//                                 where they will later be written?
//
// ⛔ WHY THE LAYOUT HALF IS NOT THE ANSWER, measured on 2026-07-28 and recorded
// because the opposite was written down first. `rfc/stdlib-semantics.md` (family
// D) and the comment on `borrowEdgeRetainIsSpellable` both stated that the
// separator between the retain that MUST be written and the three that must not
// is layout -- "a one-lane `list` handle satisfies [the prefix] and a 16-word
// payload box does not". **A payload box does satisfy it.**
// `objectPayloadHandleWords` (Core/CollectionPayload.cpp) writes
// `words[0] = refcount` and `words[1] = payloadClass`, exactly the prefix, at
// exactly those indices. Both populations carry it, so no layout predicate --
// this one or any other -- can tell them apart, and the widths cannot either
// (`builtins.object` and the payload box are both 16 words:
// ABI/HandleWidthRegistry.h).
//
// What actually separates them is the DEFINING OP, keyed over the four programs
// that pin the two behaviours (denominator: four programs, exactly one widened
// borrow-edge site each):
//
//     the retain that must exist ... memref<9xi64>  by func.call
//     the three that must not ...... memref<16xi64> by memref.alloc
//
// A `func.call` result is an entity its callee finished initialising before
// returning. A `memref.alloc` result is raw storage: `boxRuntimeObject`
// (ABI/RuntimeABI.cpp) stores the prefix in the ops AFTER the alloc, so the
// earliest point at which the value exists is INSIDE the initialisation window,
// and a refcount read there sees uninitialised memory (`Ly_IncRef observed
// non-positive refcount` on cross_container_box_fronted_fields,
// dict_key_mutation, dict_key_mutation_str).
//
// So the three cases are not broken by this predicate being too narrow. They
// depend on a DIFFERENT invariant -- "the anchor may sit at the header's
// definition" -- which is false for raw storage and which no layout fact can
// repair. That is why the repair splits cleanly instead of trading twelve
// against three.
//
// Why NOT enumerate the producers that are known-bad (alloc/alloca) and accept
// the rest: the answer would then default to "initialised" for every producer
// nobody thought about, and the failure direction of a wrong "yes" is a retain
// on uninitialised memory. This accepts only the two producers whose
// initialisation is established by construction and declines everything else,
// so a new producer costs a counted omission
// (`LYTHON_OWNERSHIP_TRACE_RETAIN_OMISSIONS`) rather than an over-retain.
//
// Why NOT ask this of `materializeByteBuffer`'s block or any other constant
// data: those are `NonObject` -- no refcount word, no class id, no deallocator
// (ABI/ConstantData.h) -- and their element type is i8, so the layout half
// already excludes them. They are not in this predicate's population at all.

#include "Ownership.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"

#include <cstdint>

namespace py::lowering::entity_header {

// Words 0 and 1 of every entity: the refcount and the layout/destructor family
// id. The same pair ContainerLayout.h and StrBytesLayout.h name for their own
// contracts, stated once here for contracts that have no layout header of their
// own.
inline constexpr std::int64_t kRefcountWord = 0;
inline constexpr std::int64_t kClassIdWord = 1;
inline constexpr std::int64_t kPrefixWordCount = 2;

// LAYOUT half. Delegated rather than re-spelled: `isObjectHeaderLikeType` is
// already this exact test (rank-1 i64, extent >= 2 or dynamic), and a second
// copy would be a second thing to keep true.
inline bool typeCarriesHeaderPrefix(mlir::Type type) {
  return ownership::isObjectHeaderLikeType(type);
}

// Follow re-descriptions of one handle back to the value whose DEFINITION
// decides whether the prefix is stored. A cast and a zero-offset prefix subview
// re-describe the same words; a non-zero offset does not, so it stops the walk.
inline mlir::Value handleProvenanceRoot(mlir::Value handle) {
  while (mlir::Operation *definition = handle.getDefiningOp()) {
    if (auto cast = mlir::dyn_cast<mlir::memref::CastOp>(definition)) {
      handle = cast.getSource();
      continue;
    }
    if (auto view = mlir::dyn_cast<mlir::memref::SubViewOp>(definition)) {
      llvm::SmallVector<mlir::OpFoldResult, 1> offsets = view.getMixedOffsets();
      // Why the offset and not just the source: word 0 of a subview taken at a
      // non-zero offset is not word 0 of the entity, so the source's prefix says
      // nothing about this value's.
      if (offsets.size() != 1)
        return handle;
      auto attribute = llvm::dyn_cast_if_present<mlir::Attribute>(offsets.front());
      auto integer = llvm::dyn_cast_if_present<mlir::IntegerAttr>(attribute);
      if (!integer || integer.getInt() != 0)
        return handle;
      handle = view.getSource();
      continue;
    }
    if (auto cast =
            mlir::dyn_cast<mlir::UnrealizedConversionCastOp>(definition)) {
      // Identity-shaped ownership markers keep types and arity; anything else is
      // a real conversion and is not followed.
      if (cast.getInputs().size() != cast.getOutputs().size())
        return handle;
      handle = cast.getInputs()[mlir::cast<mlir::OpResult>(handle)
                                    .getResultNumber()];
      continue;
    }
    return handle;
  }
  return handle;
}

// PROVENANCE half, and the question a retain writer actually has.
inline bool prefixIsInitializedAtDefinition(mlir::Value handle) {
  if (!typeCarriesHeaderPrefix(handle.getType()))
    return false;
  // ⭐ An ownership marker answers this question directly.
  //
  // `ly.ownership.owned_local_object` is emitted only once the entity is
  // COMPLETE, so at the marker the prefix is written, which is exactly what
  // this predicate is asked. That is `proof/`'s
  // `no-dup-in-the-initialisation-window`, and it holds here because
  // `verifyInitialisationWindowIn` refuses any marker the word-0 store does
  // not dominate.
  //
  // Why NOT justify it by listing the producers, as this said until the gate
  // existed: it named the boxing path in `ABI/RuntimeABI.cpp` and
  // `Ops/GetItemOps.cpp`, two of the three sites that mint the attribute. The
  // missing one, `Core/ObjectBundles.cpp`, is the one that marks ordinary
  // entities -- the common case this predicate is asked about. Its ordering
  // was in fact correct, so the conclusion held; an argument that omits a
  // third of its cases just cannot be what holds it up.
  //
  // Why NOT let `handleProvenanceRoot` decide it: that walk deliberately
  // follows identity-shaped casts, so it walks straight THROUGH the marker to
  // the `memref.alloc` underneath and answers about raw storage -- the state
  // the marker exists to say the value has left. Asking the marker is asking
  // what is true at the point the retain will be inserted; walking past it is
  // asking how the storage was made.
  if (mlir::Operation *definition = handle.getDefiningOp())
    if (definition->hasAttr("ly.ownership.owned_local_object"))
      return true;
  mlir::Value root = handleProvenanceRoot(handle);
  if (!typeCarriesHeaderPrefix(root.getType()))
    return false;
  // A block argument is bound on entry, so the earliest point at which it
  // exists is already past whatever initialised it in every predecessor.
  if (mlir::isa<mlir::BlockArgument>(root))
    return true;
  // A call result is an entity the callee finished before returning. This is the
  // producer the shipped over-release needed and did not have: a one-lane
  // container handle comes out of its contract's allocator as a call result, not
  // as a block argument (rfc/stdlib-semantics.md, family D).
  return mlir::isa_and_nonnull<mlir::func::CallOp>(root.getDefiningOp());
}

} // namespace py::lowering::entity_header
