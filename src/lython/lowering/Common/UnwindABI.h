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
#include "llvm/TargetParser/Triple.h"

namespace py::runtime_library {

// "LYTHPY01". The high half is a vendor tag, so a C++ runtime that meets this
// carrier on its way past knows not to read a `__cxa_exception` header in
// front of it.
inline constexpr std::uint64_t kLythonExceptionClass = 0x4C59544850593031ULL;

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
