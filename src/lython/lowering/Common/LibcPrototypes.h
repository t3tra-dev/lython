#pragma once

// The C library as the target's C compiler would call it.
//
// Every runtime builder, manifest and MLIR conversion pattern declares libc
// in the runtime's own vocabulary: sizes are i64, `long` is i64, `time_t` is
// i64, and each function has one name. That is the C prototype on an LP64
// target and nothing else. `declareLibcWithTargetPrototypes` runs once the
// runtime is linked and makes every libc declaration the one C gives it on the
// module's target -- widths from the data layout, the symbol the target's
// headers would have picked -- adapting each call to it, so nothing upstream
// has to know which target it is building for.

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {
class Module;
class raw_ostream;
} // namespace llvm

namespace py::runtime_library {

// Function attribute on a declaration ctypes made for a foreign symbol: its
// prototype is the one the program wrote, so a name the table does not know
// is the program's business rather than a gap in the table.
inline constexpr llvm::StringLiteral kCtypesForeignSymbolAttr =
    "ly-ctypes-foreign";

// Fails, naming the functions, when a target that needs exact prototypes
// (a 32-bit one, or one that links calls by signature) calls a C library
// function the table has no prototype for.
mlir::LogicalResult declareLibcWithTargetPrototypes(llvm::Module &module,
                                                    llvm::raw_ostream &diag);

} // namespace py::runtime_library
