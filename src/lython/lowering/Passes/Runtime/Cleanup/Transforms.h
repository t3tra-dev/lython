#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringRef.h"

#include <memory>

namespace mlir {
class ConversionTarget;
class LLVMTypeConverter;
class Pass;
class RewritePatternSet;
} // namespace mlir

namespace py::lowering::runtime::cleanup {

bool pointerRoundTrips(mlir::ModuleOp module);

// Empties the body of every private function no public symbol can reach,
// leaving the symbol and its attributes in place. Answers how many it emptied.
unsigned stripUnreachableManifestBodies(mlir::ModuleOp module);

// Routes each cond_br whose two successors are ONE block (a canonicalized
// empty-arm conditional) through two fresh edge blocks, so every edge into a
// block has a terminator of its own. Answers how many it split.
unsigned splitDualEdges(mlir::ModuleOp module);

// Marks every function body from which no Python exception can propagate
// with `ownership::kCannotRaiseAttr`, by a fixpoint over the call graph.
// Answers how many it marked.
unsigned markCannotRaiseFunctions(mlir::ModuleOp module);

// Gives every converted function so marked LLVM's `nounwind`.
void markCannotRaiseNoUnwind(mlir::ModuleOp module);

// On a runtime func.func that must not be inlined into its callers.
inline constexpr llvm::StringLiteral kOutOfLineAttr{"ly.runtime.out_of_line"};
// Gives every converted function so marked LLVM's `noinline`.
void markOutOfLineNoInline(mlir::ModuleOp module);

// On the casts between a memref and the narrow form that carries it across a
// call (MemRefCallABI.cpp).
inline constexpr llvm::StringLiteral kNarrowCastAttr{"ly.abi.narrow"};
// On such a cast whose memref the analysis found at its allocation's start:
// LYTHON_VERIFY_BARE_POINTER_ABI=1 checks it there.
inline constexpr llvm::StringLiteral kNarrowAtStartAttr{"ly.abi.at_start"};
// On a native runtime function that spells memref descriptors out as their
// five words: [first word, static size or -1] per descriptor.
inline constexpr llvm::StringLiteral kExpandedDescriptorsAttr{
    "ly.abi.expanded_descriptors"};
inline constexpr llvm::StringLiteral kNarrowCheck{
    "__ly_abi_check_bare_pointer"};

// Which functions besides the private ones only called directly carry their
// memrefs narrow: a program's declarations (the native runtime's functions),
// and the native runtime's own public definitions.
struct NarrowingScope {
  bool declarations;
  bool publicDefinitions;
};

// Passes and returns memrefs narrow where `NarrowingScope` and the module's
// own calls allow. Answers how many functions changed.
unsigned narrowMemRefCalls(mlir::ModuleOp module, NarrowingScope scope);

// Rewrites each function carrying `kExpandedDescriptorsAttr` to take the
// narrow form in place of the five words. Answers how many changed.
unsigned narrowExpandedDescriptors(mlir::ModuleOp module);

bool verifyingNarrowCalls();
void populateNarrowCastLowering(mlir::LLVMTypeConverter &converter,
                                mlir::RewritePatternSet &patterns,
                                mlir::ConversionTarget &target, bool verify);
// The function LYTHON_VERIFY_BARE_POINTER_ABI's checks call.
void defineNarrowCheck(mlir::ModuleOp module);
// The casts' lowering on its own, for a pipeline that converts func and
// memref in separate passes (the native runtime's).
std::unique_ptr<mlir::Pass> createLowerNarrowCastsPass();

} // namespace py::lowering::runtime::cleanup
