#pragma once

// Handle-word layout of the one-lane byte-payload entities (`builtins.bytes`
// today; `builtins.str` keeps its two lanes until that contract converts),
// shared by the C++ lowering and the runtime manifest (LyBytes_Shape and the
// LyBytes_* bodies in runtime/modules/builtins.mlir).
//
// Words 0 and 1 are the refcount and the layout/destructor family id of every
// entity. A byte-payload entity adds the BASE ADDRESS of its payload and the
// payload's length in bytes, so the payload is reached by loading the base out
// of the handle at the point of use -- which is what makes a reallocation a
// write through the root rather than a re-description of the entity
// (rfc/memory-safety-proof.md, `Interior`).
//
// Why a raw address word and not a memref lane: a lane travels beside the
// root, so a holder can keep one past a reallocation. Why not a nested memref:
// the manifest side may not assemble a memref descriptor (a
// `builtin.unrealized_conversion_cast` in the runtime-lowering INPUT is
// rejected by `requireResolvedInput`), so the only spelling available to both
// sides is an integer address -- the same one `_io.*` uses for its buffer.
//
// Why the width differs from ContainerLayout.h's eight: `verifyReceiverShape`
// compares a PREFIX of a method's inputs against the shape, so a contract that
// reuses a width already in play lets a not-yet-converted method pass the check
// with stale trailing lanes outside the comparison window
// (rfc/lane-conversion-playbook.md step 1). Two is what `bytes` came from, so
// four is the smallest width that satisfies that rule for this contract.
//
// Why SIX and not four: four was the release-interface width of the lyrt
// counters, and while a release was chosen by shape a shared width tied and
// released nothing (28 lost groups on a three-line bytes program). A release
// is now chosen by contract name (`findDeallocatorForValueGroup`), so the
// width only has to satisfy the rule above.

#include <cstdint>

namespace py::lowering::bytes_abi {

inline constexpr std::int64_t kRefcountWord = 0;
inline constexpr std::int64_t kClassIdWord = 1;
inline constexpr std::int64_t kPayloadArrayWord = 2;
inline constexpr std::int64_t kPayloadLengthWord = 3;
inline constexpr std::int64_t kHandleWordCount = 6;

// Byte offset at which the payload starts inside the single allocation the
// entity lives in. Kept 16-byte aligned so `memref.view` over the tail keeps
// the block's alignment.
inline constexpr std::int64_t kPayloadByteOffset = 48;

} // namespace py::lowering::bytes_abi
