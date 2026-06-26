#pragma once

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <string>

namespace py::runtime_lowering {

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
inline constexpr llvm::StringLiteral kManifestResultContractAttr{
    "ly.runtime.result_contract"};
inline constexpr llvm::StringLiteral kManifestResultEvidenceAttr{
    "ly.runtime.result_evidence"};
inline constexpr llvm::StringLiteral kManifestElementContractAttr{
    "ly.runtime.element_contract"};
inline constexpr llvm::StringLiteral kManifestNextContractAttr{
    "ly.runtime.next_contract"};
inline constexpr llvm::StringLiteral kManifestValidResultIndexAttr{
    "ly.runtime.valid_result_index"};
inline constexpr llvm::StringLiteral kCallableDefaultValuesAttr{
    "callable_default_values"};
inline constexpr llvm::StringLiteral kCallableVarargValueTypeAttr{
    "callable_vararg_value_type"};
inline constexpr llvm::StringLiteral kCallableKwargValueTypeAttr{
    "callable_kwarg_value_type"};
inline constexpr llvm::StringLiteral kPackUnpackedOperandsAttr{
    "ly.unpack_operands"};

std::string runtimeKey(llvm::StringRef contract, llvm::StringRef role,
                       llvm::StringRef name);
bool isIntegerLiteralSpelling(llvm::StringRef spelling);
std::string runtimeContractName(mlir::Type type);
std::string runtimeShapeContractName(mlir::Type type);
bool compatibleRuntimeObjectEvidenceContract(mlir::Type resultType,
                                             mlir::Type evidenceType);
mlir::Type runtimeContractType(mlir::MLIRContext *context,
                               llvm::StringRef contract);
bool sameTypeSequence(llvm::ArrayRef<mlir::Type> lhs,
                      llvm::ArrayRef<mlir::Type> rhs);
std::string describeTypeSequence(llvm::ArrayRef<mlir::Type> types);
std::string describeValueTypes(mlir::ValueRange values);
llvm::SmallVector<mlir::Type, 4> takePrefix(llvm::ArrayRef<mlir::Type> types,
                                            unsigned count);
llvm::SmallVector<mlir::Type, 4> takeSlice(llvm::ArrayRef<mlir::Type> types,
                                           unsigned begin, unsigned end);

} // namespace py::runtime_lowering
