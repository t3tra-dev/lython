#pragma once

#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

#include <cstddef>
#include <string>

namespace py::contracts {

inline constexpr llvm::StringLiteral kManifestContractsAttr{
    "ly.runtime.contracts"};
inline constexpr llvm::StringLiteral kManifestContractAttr{
    "ly.runtime.contract"};
inline constexpr llvm::StringLiteral kManifestMethodAttr{"ly.runtime.method"};
inline constexpr llvm::StringLiteral kManifestInitializerAttr{
    "ly.runtime.initializer"};
inline constexpr llvm::StringLiteral kManifestPrimitiveAttr{
    "ly.runtime.primitive"};
// A primitive whose results are INTERIOR words of the entity its operands
// reach (a block pointer hung off an exception header, a word of a box inside
// that block). Ownership treats those results the way it treats box-word
// reconstructions: they pin the entity's liveness, because the entity's
// release frees the storage they address. Declared here rather than inferred
// because only the manifest knows which primitives dereference their operands.
inline constexpr llvm::StringLiteral kManifestInteriorWordAttr{
    "ly.runtime.interior_word"};
inline constexpr llvm::StringLiteral kManifestBuiltinAttr{"ly.runtime.builtin"};
inline constexpr llvm::StringLiteral kManifestBuiltinLoweringAttr{
    "ly.runtime.builtin_lowering"};
inline constexpr llvm::StringLiteral kManifestBuiltinMethodAttr{
    "ly.runtime.builtin_method"};
inline constexpr llvm::StringLiteral kManifestBuiltinSinkContractAttr{
    "ly.runtime.builtin_sink_contract"};
inline constexpr llvm::StringLiteral kManifestShapeAttr{"ly.runtime.shape"};
inline constexpr llvm::StringLiteral kManifestDeallocatorAttr{
    "ly.runtime.deallocator"};
inline constexpr llvm::StringLiteral kManifestClassAttr{
    "ly.runtime.class"};
inline constexpr llvm::StringLiteral kManifestClassArgumentAttr{
    "ly.runtime.class_argument"};
inline constexpr llvm::StringLiteral kManifestDefaultI64Attr{
    "ly.runtime.default_i64"};
// An i64 input that takes an int the way CPython reads an index with
// PyNumber_AsSsize_t(v, NULL): past the word it is the nearest end of the
// word, not an OverflowError (a slice bound, str.find's window).
inline constexpr llvm::StringLiteral kManifestClipI64Attr{
    "ly.runtime.clip_i64"};
inline constexpr llvm::StringLiteral kManifestDefaultF64Attr{
    "ly.runtime.default_f64"};
inline constexpr llvm::StringLiteral kManifestDefaultStrAttr{
    "ly.runtime.default_str"};
inline constexpr llvm::StringLiteral kManifestDefaultBytesAttr{
    "ly.runtime.default_bytes"};
inline constexpr llvm::StringLiteral kManifestResultContractAttr{
    "ly.runtime.result_contract"};
inline constexpr llvm::StringLiteral kManifestResultEvidenceAttr{
    "ly.runtime.result_evidence"};
inline constexpr llvm::StringLiteral kManifestResultEvidenceSlotsAttr{
    "ly.runtime.result_evidence_slots"};
inline constexpr llvm::StringLiteral kManifestResultEvidenceContractsAttr{
    "ly.runtime.result_evidence_contracts"};
inline constexpr llvm::StringLiteral kManifestElementContractAttr{
    "ly.runtime.element_contract"};
inline constexpr llvm::StringLiteral kManifestNextContractAttr{
    "ly.runtime.next_contract"};
inline constexpr llvm::StringLiteral kManifestNextEvidenceAttr{
    "ly.runtime.next_evidence"};
inline constexpr llvm::StringLiteral kManifestValidResultIndexAttr{
    "ly.runtime.valid_result_index"};
inline constexpr llvm::StringLiteral kManifestRequiredAttr{
    "ly.runtime.required"};
inline constexpr llvm::StringLiteral kManifestRequiredInitializersAttr{
    "ly.runtime.required_initializers"};
inline constexpr llvm::StringLiteral kManifestRequiredMethodsAttr{
    "ly.runtime.required_methods"};
inline constexpr llvm::StringLiteral kManifestRequiredPrimitivesAttr{
    "ly.runtime.required_primitives"};
inline constexpr llvm::StringLiteral kManifestRequiredDeallocatorAttr{
    "ly.runtime.required_deallocator"};

inline bool isIntegerLiteralSpelling(llvm::StringRef spelling) {
  if (spelling.empty())
    return false;
  if (spelling.front() == '-')
    spelling = spelling.drop_front();
  return !spelling.empty() &&
         llvm::all_of(spelling, [](char c) { return c >= '0' && c <= '9'; });
}

// Strips the manifest module namespace off a contract name ("builtins.int"
// -> "int"); the prefix list must match the modules that publish manifest
// classes.
inline std::string manifestClassNameForContract(llvm::StringRef name) {
  for (llvm::StringRef prefix :
       {"builtins.", "typing.", "types.", "contextlib.", "contextvars.",
        "ctypes.", "_ctypes.", "_typeshed."}) {
    if (name.consume_front(prefix))
      return name.str();
  }
  return name.str();
}

// The class name a PROGRAM sees. A monomorphized generic class contract is
// named "<class>$spec<N>", an internal symbol that must never surface: repr,
// exception names and diagnostics all have to read as the class the source
// wrote. Kept separate from manifestClassNameForContract because that one also
// feeds manifest LOOKUPS, which need the specialization's own name.
inline std::string displayClassNameForContract(llvm::StringRef name) {
  std::string display = manifestClassNameForContract(name);
  std::size_t marker = display.rfind("$spec");
  if (marker == std::string::npos || marker == 0)
    return display;
  llvm::StringRef index(display.c_str() + marker + 5);
  if (index.empty() || !llvm::all_of(index, [](char c) {
        return c >= '0' && c <= '9';
      }))
    return display;
  display.resize(marker);
  return display;
}

std::string runtimeContractName(mlir::Type type);

// The reflected binary special methods: the receiver is the RIGHT operand.
// ⛔ Not "starts with `__r`": `__rshift__` is the forward right shift, and
// `__repr__` / `__round__` are not binary at all.
inline bool isReflectedBinaryMethodName(llvm::StringRef name) {
  return llvm::is_contained(
      {llvm::StringRef("__radd__"), llvm::StringRef("__rsub__"),
       llvm::StringRef("__rmul__"), llvm::StringRef("__rmatmul__"),
       llvm::StringRef("__rtruediv__"), llvm::StringRef("__rfloordiv__"),
       llvm::StringRef("__rmod__"), llvm::StringRef("__rdivmod__"),
       llvm::StringRef("__rpow__"), llvm::StringRef("__rlshift__"),
       llvm::StringRef("__rrshift__"), llvm::StringRef("__rand__"),
       llvm::StringRef("__ror__"), llvm::StringRef("__rxor__")},
      name);
}

} // namespace py::contracts
