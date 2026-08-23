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
// The narrowest container handle, and therefore the minimum width at which a
// rank-1 i64 memref can be one: `isContainerHandleType` uses it as a lower
// bound, not as an equality, because a converted container may be WIDER than
// its layout needs. `builtins.list` is nine words for that reason -- words 0..7
// are this layout (5 and 6 unused by a sequence) and word 8 is dead space that
// buys an untied single-input release interface (ABI/HandleWidthRegistry.h).
// `set` and `frozenset` pay the same way, three and five dead words, and their
// widths differ from each other on purpose: a shared width would leave the two
// tied with each other on release interface, which is the situation
// `range`/`range_iterator` is stuck in and the one thing a converter CAN avoid
// when the two contracts are actually distinguishable.
//
// `builtins.tuple` is fourteen for the same reason and a worse rate: words
// 8..13 are dead, because by the time it converted, 14 was the narrowest free
// single-input interface left. That is the width-scarcity arithmetic in
// HandleWidthRegistry.h showing up as bytes -- a 3-element tuple pays 48 B of
// padding for identity alone -- and it is why the registry's free list is now
// empty rather than merely short. Set and frozenset spent 11 and 13 on the way
// there, so the four container conversions between them consumed every width
// the free list had.
inline constexpr std::int64_t kHandleWordCount = 8;
inline constexpr std::int64_t kDictHandleWordCount = 8;
inline constexpr std::int64_t kListHandleWordCount = 9;
inline constexpr std::int64_t kSetHandleWordCount = 11;
inline constexpr std::int64_t kFrozenSetHandleWordCount = 13;
inline constexpr std::int64_t kTupleHandleWordCount = 14;

// Slot indices inside the {length, capacity} pair, which a multi-lane
// contract still carries as its own `meta` lane. Words 2/3 above are the same
// pair in the same order, so a view over them is a drop-in `meta`.
inline constexpr std::int64_t kMetaLengthSlot = 0;
inline constexpr std::int64_t kMetaCapacitySlot = 1;
inline constexpr std::int64_t kMetaWordCount = 2;

} // namespace py::lowering::container_abi
