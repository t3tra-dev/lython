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
// Why SIX and not four. Four also satisfies the rule above, and four is what this
// contract was first converted to -- wrongly. Four is the release-interface width
// of `lyrt.Counter`, `lyrt.AsyncCounter` and `lyrt.ReadyAsyncCounter`, and
// `findDeallocatorForValueGroup`'s contract-less overload filters candidates by
// `inputTypes` and only then breaks the tie on `shapeMatch`. A tie there is
// resolved by NEITHER width nor shape when the scores are equal: every candidate
// after the first sets `ambiguous`, and the function returns nullptr. Scoring
// zero is not the safe case, it is the INDISTINGUISHABLE case.
//
// That was measured, not reasoned. With `bytes` at four, a three-line program
// (`b = b"hello"; len(b); b + b"!"`) reached the contract-less overload 28 times
// with a bare `memref<4xi64>` and got nullptr every time -- so a bytes handle had
// no owner group on those paths. Six is unused by every release interface in the
// tree (7 and 9-15 are too), and the same probe reports zero ambiguous hits for
// `memref<6xi64>`.
//
// The tie this replaced was the same mechanism with a positive score: `str` and
// `bytes` were both two-lane, both scored `shapeMatch` 2 against a real two-lane
// group, and the collector refused a group it had in hand -- which is what hid a
// double release of the exception message (see rfc/memory-safety-proof.md,
// `NonInstantiationIsNotConformance`). Positive-score and zero-score ties differ
// only in what is suppressed, not in whether suppression happens.

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
