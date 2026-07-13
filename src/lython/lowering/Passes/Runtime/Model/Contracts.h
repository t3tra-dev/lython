#pragma once

#include <Contracts.h>

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <string>

namespace py::lowering {

using py::contracts::kManifestBuiltinAttr;
using py::contracts::kManifestBuiltinLoweringAttr;
using py::contracts::kManifestBuiltinMethodAttr;
using py::contracts::kManifestBuiltinSinkContractAttr;
using py::contracts::kManifestClassIdArgumentAttr;
using py::contracts::kManifestClassIdAttr;
using py::contracts::kManifestContractAttr;
using py::contracts::kManifestContractsAttr;
using py::contracts::kManifestDeallocatorAttr;
using py::contracts::kManifestDefaultBytesAttr;
using py::contracts::kManifestDefaultF64Attr;
using py::contracts::kManifestDefaultI64Attr;
using py::contracts::kManifestDefaultStrAttr;
using py::contracts::kManifestElementContractAttr;
using py::contracts::kManifestInitializerAttr;
using py::contracts::kManifestMethodAttr;
using py::contracts::kManifestNextContractAttr;
using py::contracts::kManifestNextEvidenceAttr;
using py::contracts::kManifestPrimitiveAttr;
using py::contracts::kManifestResultContractAttr;
using py::contracts::kManifestResultEvidenceAttr;
using py::contracts::kManifestResultEvidenceContractsAttr;
using py::contracts::kManifestResultEvidenceSlotsAttr;
using py::contracts::kManifestShapeAttr;
using py::contracts::kManifestValidResultIndexAttr;
using py::contracts::isIntegerLiteralSpelling;
using py::contracts::runtimeContractName;
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

} // namespace py::lowering
