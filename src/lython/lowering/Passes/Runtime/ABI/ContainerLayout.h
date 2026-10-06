#pragma once

// Handle-word layout of a one-lane container entity, shared by the C++
// lowering and the runtime manifest (LyDict_Shape / the LyDict_* bodies in
// runtime/modules/builtins.mlir).
//
// Words 0 and 1 are the refcount and the layout/destructor family id of every
// entity. A container adds its length, its capacity, and the BASE ADDRESS of
// each interior array. The arrays are therefore reached by loading the base
// out of the handle at the point of use, which is what makes a reallocation a
// write through the root rather than a re-description of the entity
// (rfc/memory-safety-proof.md, `Interior`).
//
// Why raw address words and not memref lanes: a lane travels beside the root,
// so a holder can keep one past a reallocation. Why raw address words and not
// a nested memref: the manifest side may not assemble a memref descriptor (a
// `builtin.unrealized_conversion_cast` in the runtime-lowering INPUT is
// rejected by `requireResolvedInput`), so the only spelling available to both
// sides is an integer address -- the same one `_io.*` uses for its buffer.

#include <cstdint>

namespace py::lowering::container_abi {

inline constexpr std::int64_t kRefcountWord = 0;
inline constexpr std::int64_t kClassIdWord = 1;
inline constexpr std::int64_t kLengthWord = 2;
inline constexpr std::int64_t kCapacityWord = 3;
// Primary array: `items` for a sequence, `keys` for a mapping.
inline constexpr std::int64_t kPrimaryArrayWord = 4;
// Secondary array: `values` for a mapping; unused by a sequence.
inline constexpr std::int64_t kSecondaryArrayWord = 5;
// Per-slot occupancy flags; unused by a sequence.
inline constexpr std::int64_t kPresentArrayWord = 6;
// Base of the mapping's index table (word 0 of that block is the length the
// table was built for; the rest is 2*capacity slots of (state, hash)). Unused
// by a sequence, which is why it was the reserved word.
inline constexpr std::int64_t kTableWord = 7;
// A handle is as wide as the words it uses: a sequence uses 0..4 (a tuple's
// items follow in the same block), a mapping 0..7, a set or frozenset 0..8.
inline constexpr std::int64_t kHandleWordCount = 8;
inline constexpr std::int64_t kDictHandleWordCount = 8;
inline constexpr std::int64_t kListHandleWordCount = 5;
inline constexpr std::int64_t kSetHandleWordCount = 9;
inline constexpr std::int64_t kFrozenSetHandleWordCount = 9;
inline constexpr std::int64_t kTupleHandleWordCount = 5;

// Slot indices inside the {length, capacity} pair, which a multi-lane
// contract still carries as its own `meta` lane. Words 2/3 above are the same
// pair in the same order, so a view over them is a drop-in `meta`.
inline constexpr std::int64_t kMetaLengthSlot = 0;
inline constexpr std::int64_t kMetaCapacitySlot = 1;
inline constexpr std::int64_t kMetaWordCount = 2;

} // namespace py::lowering::container_abi
