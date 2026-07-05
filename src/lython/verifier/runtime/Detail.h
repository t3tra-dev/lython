#pragma once

#include "Support.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace py::lowering {

using py::verifier::VerificationResult;
using py::verifier::isIntegerType;
using py::verifier::readOptionalStringArrayAttr;
using py::verifier::readRequiredBoolAttr;
using py::verifier::readRequiredStringArrayAttr;
using py::verifier::readRequiredStringAttr;
using py::verifier::readRequiredUnsignedIntegerAttr;
using py::verifier::walkVerify;
using py::verifier::walkVerifyOperations;

mlir::LogicalResult verifyOwnershipContractShapes(mlir::ModuleOp module);
mlir::LogicalResult verifyLLVMOwnershipContractShapes(mlir::ModuleOp module);
mlir::LogicalResult verifyLLVMCallOwnershipContracts(mlir::ModuleOp module);
mlir::LogicalResult verifyFuncCallOwnershipContracts(mlir::ModuleOp module);

} // namespace py::lowering
