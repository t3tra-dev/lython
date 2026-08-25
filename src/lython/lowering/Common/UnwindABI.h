#pragma once

// Which personality routine reads a Lython frame's LSDA, and what the raise
// puts in the carrier.
//
// Four places have to agree -- the pass that sets `personalityFn`
// (Passes/Runtime/Cleanup/EH.cpp), the builder that defines the routine and
// emits the raise (Common/RuntimeSupportBuilder.cpp), and the target rewrite
// in the driver (driver/lib/LLVMFinalize.cpp) -- so the decision is made here
// and nowhere else.

#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include "llvm/TargetParser/Triple.h"

namespace py::runtime_library {

// "LYTHPY01". The high half is a vendor tag, so a C++ runtime that meets this
// carrier on its way past knows not to read a `__cxa_exception` header in
// front of it.
inline constexpr std::uint64_t kLythonExceptionClass = 0x4C59544850593031ULL;

// The exception carrier: the 32 bytes `_Unwind_Exception` requires, followed by
// what the personality remembers about the frames it has already read.
//
// The call-site table is read linearly, so a raise pays for the entries in front
// of it; the answer depends on nothing but the return address, so it is worth
// keeping. Direct-mapped by return address, and IN THE CARRIER because a
// carrier belongs to one in-flight exception -- no lock, no thread-local, and
// the entries stay valid for the life of the process, so a carrier handed back
// by a catch is worth more than a fresh one.
namespace eh_carrier {
inline constexpr std::int64_t kHeaderBytes = 32;
inline constexpr std::int64_t kMemoOffset = kHeaderBytes;
inline constexpr std::int64_t kMemoEntries = 8;
// return address, landing pad address, action, action table start, type table
// base.
inline constexpr std::int64_t kMemoEntryBytes = 40;
inline constexpr std::int64_t kMemoBytes = kMemoEntries * kMemoEntryBytes;

// And after the memo, the machine state a walk of our own carries: the
// callee-saved registers an intervening frame may have spilled, plus where that
// frame is and where it came from. A landing pad entered without these is a
// function reading someone else's registers.
inline constexpr std::int64_t kContextOffset = kMemoOffset + kMemoBytes;
inline constexpr std::int64_t kContextIntRegs = 0;   // x19..x28
inline constexpr std::int64_t kContextFPRegs = 80;   // d8..d15
inline constexpr std::int64_t kContextFP = 144;
inline constexpr std::int64_t kContextSP = 152;
inline constexpr std::int64_t kContextPC = 160;
inline constexpr std::int64_t kContextLR = 168;
inline constexpr std::int64_t kContextPad = 176;
inline constexpr std::int64_t kContextSelector = 184;
inline constexpr std::int64_t kContextCarrier = 192;
inline constexpr std::int64_t kContextHandler = 200;
inline constexpr std::int64_t kContextActive = 208;
inline constexpr std::int64_t kContextBytes = 216;

inline constexpr std::int64_t kTotalBytes = kContextOffset + kContextBytes;
} // namespace eh_carrier

// The parts of `__unwind_info` (mach-o/compact_unwind_encoding.h) this reads.
namespace compact_unwind {
inline constexpr std::int64_t kHeaderCommonEncodingsOffset = 4;
inline constexpr std::int64_t kHeaderCommonEncodingsCount = 8;
inline constexpr std::int64_t kHeaderPersonalitiesOffset = 12;
inline constexpr std::int64_t kHeaderIndexOffset = 20;
inline constexpr std::int64_t kHeaderIndexCount = 24;
inline constexpr std::int64_t kIndexEntryBytes = 12;
inline constexpr std::int64_t kSecondLevelKindRegular = 2;
inline constexpr std::int64_t kSecondLevelKindCompressed = 3;

inline constexpr std::int64_t kHasLSDA = 0x40000000;
inline constexpr std::int64_t kPersonalityMask = 0x30000000;
inline constexpr std::int64_t kArm64ModeMask = 0x0F000000;
inline constexpr std::int64_t kArm64ModeFrame = 0x04000000;
inline constexpr std::int64_t kArm64StackSizeMask = 0x00FFF000;
} // namespace compact_unwind

inline constexpr llvm::StringRef kPythonPersonalityName = "LyEH_Personality";
inline constexpr llvm::StringRef kItaniumPersonalityName =
    "__gxx_personality_v0";
inline constexpr llvm::StringRef kSEHPersonalityName = "__gxx_personality_seh0";

// The DWARF register numbers a `landingpad` reads its two values out of: the
// personality writes the carrier into the first and the clause selector into
// the second, and LLVM's landing-pad lowering copies them from exactly these.
struct EHDataRegisters {
  int exception;
  int selector;
};

// ⛔ 64-bit only, and only where the pair above is known. The personality
// signature spells `_Unwind_Ptr` as i64 -- on a 32-bit target that is the
// wrong width for `_Unwind_SetIP`/`_Unwind_SetGR`, and a personality that
// installs a truncated landing pad address is exactly the silent
// mis-execution this compiler refuses. Anything else keeps the C++ ABI's
// personality, which is correct everywhere and only slower.
inline bool usePythonPersonality(const llvm::Triple &triple) {
  if (triple.isOSWindows())
    return false;
  switch (triple.getArch()) {
  case llvm::Triple::aarch64:
  case llvm::Triple::aarch64_be:
  case llvm::Triple::x86_64:
  case llvm::Triple::riscv64:
    return true;
  default:
    return false;
  }
}

// ⛔ Darwin/arm64 only, and this is a floor rather than a preference. The walk
// reads `__unwind_info` and decodes the arm64 compact encoding to restore
// x19-x28 and d8-d15; every one of those is a per-platform fact, and a walk
// that restores the wrong register jumps into a handler holding another
// function's values. Everywhere else keeps `_Unwind_RaiseException`, which is
// correct on every target and only slower.
inline bool useFastUnwinder(const llvm::Triple &triple) {
  return usePythonPersonality(triple) && triple.isOSDarwin() &&
         (triple.getArch() == llvm::Triple::aarch64 ||
          triple.getArch() == llvm::Triple::aarch64_be);
}

inline EHDataRegisters ehDataRegisters(const llvm::Triple &triple) {
  if (triple.getArch() == llvm::Triple::riscv64)
    return {10, 11}; // x10, x11
  if (triple.getArch() == llvm::Triple::x86_64)
    return {0, 1}; // RAX, RDX
  return {0, 1};   // x0, x1
}

inline llvm::StringRef personalityNameFor(const llvm::Triple &triple) {
  if (usePythonPersonality(triple))
    return kPythonPersonalityName;
  if (triple.isWindowsGNUEnvironment())
    return kSEHPersonalityName;
  return kItaniumPersonalityName;
}

} // namespace py::runtime_library
