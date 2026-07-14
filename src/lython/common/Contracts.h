#pragma once

#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

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
inline constexpr llvm::StringLiteral kManifestClassIdAttr{
    "ly.runtime.class_id"};
inline constexpr llvm::StringLiteral kManifestClassIdArgumentAttr{
    "ly.runtime.class_id_argument"};
inline constexpr llvm::StringLiteral kManifestDefaultI64Attr{
    "ly.runtime.default_i64"};
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
       {"builtins.", "typing.", "types.", "contextlib.", "_asyncio.",
        "asyncio.", "contextvars.", "ctypes.", "_ctypes.", "_typeshed."}) {
    if (name.consume_front(prefix))
      return name.str();
  }
  return name.str();
}
std::string runtimeContractName(mlir::Type type);

} // namespace py::contracts
