#include "runtime/Detail.h"
#include "runtime/Verification.h"

#include "Runtime/Manifest/Index.h"
#include "Contracts.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

#include <memory>

namespace py::lowering {
namespace {

namespace contracts = py::contracts;

bool declaredMethod(py::ClassOp classOp, llvm::StringRef methodName) {
  mlir::ArrayAttr methodNames = classOp.getMethodNamesAttr();
  if (!methodNames)
    return false;
  for (mlir::Attribute attr : methodNames) {
    auto string = mlir::dyn_cast<mlir::StringAttr>(attr);
    if (string && string.getValue() == methodName)
      return true;
  }
  return false;
}

bool hasRuntimeDeallocator(mlir::ModuleOp module, llvm::StringRef contractName) {
  bool found = false;
  module.walk([&](mlir::func::FuncOp function) {
    auto contract = function->getAttrOfType<mlir::StringAttr>(
        contracts::kManifestContractAttr);
    if (!contract || contract.getValue() != contractName)
      return mlir::WalkResult::advance();
    if (!function->hasAttr(contracts::kManifestDeallocatorAttr))
      return mlir::WalkResult::advance();
    found = true;
    return mlir::WalkResult::interrupt();
  });
  return found;
}

mlir::LogicalResult
verifyRuntimeRequiredDeallocator(py::ClassOp classOp, mlir::ModuleOp module,
                                 llvm::StringRef contractName) {
  if (!classOp->hasAttr(contracts::kManifestRequiredDeallocatorAttr))
    return mlir::success();
  if (hasRuntimeDeallocator(module, contractName))
    return mlir::success();
  return classOp.emitError()
         << "runtime-required contract " << contractName
         << " requires a deallocator but has no runtime deallocator symbol";
}

mlir::LogicalResult verifyRequiredMethodsDeclared(py::ClassOp classOp,
                                                  llvm::StringRef attrName) {
  mlir::FailureOr<llvm::SmallVector<llvm::StringRef, 8>> required =
      readOptionalStringArrayAttr(classOp.getOperation(), attrName);
  if (mlir::failed(required))
    return mlir::failure();

  VerificationResult verified;
  for (llvm::StringRef methodName : *required) {
    if (declaredMethod(classOp, methodName))
      continue;
    classOp.emitError() << attrName << " names " << methodName
                        << ", but the class method_names manifest does not "
                           "declare it";
    verified.fail();
  }
  return verified.get();
}

mlir::LogicalResult
verifyRuntimeRequiredClass(py::ClassOp classOp,
                           mlir::ModuleOp module,
                           const RuntimeManifestIndex &manifest) {
  VerificationResult verified;

  auto contract = classOp->getAttrOfType<mlir::StringAttr>(
      contracts::kManifestContractAttr);
  if (!contract) {
    classOp.emitError() << contracts::kManifestRequiredAttr
                        << " classes must declare "
                        << contracts::kManifestContractAttr;
    return mlir::failure();
  }

  llvm::StringRef contractName = contract.getValue();
  if (!manifest.valueShape(contractName)) {
    classOp.emitError() << "runtime-required contract " << contractName
                        << " has no ABI shape";
    verified.fail();
  }
  if (!manifest.classId(contractName)) {
    classOp.emitError() << "runtime-required contract " << contractName
                        << " has no runtime class id";
    verified.fail();
  }

  auto checkSymbols = [&](llvm::StringRef attrName,
                          llvm::StringRef role) -> mlir::LogicalResult {
    mlir::FailureOr<llvm::SmallVector<llvm::StringRef, 8>> names =
        readOptionalStringArrayAttr(classOp.getOperation(), attrName);
    if (mlir::failed(names))
      return mlir::failure();

    VerificationResult symbolsVerified;
    for (llvm::StringRef name : *names) {
      bool present = false;
      if (role == "initializer")
        present = manifest.initializer(contractName, name).has_value();
      else if (role == "method")
        present = manifest.method(contractName, name).has_value();
      else if (role == "primitive")
        present = manifest.primitive(contractName, name).has_value();

      if (present)
        continue;
      classOp.emitError() << "runtime-required " << role << " "
                          << contractName << "." << name
                          << " has no lowering manifest symbol";
      symbolsVerified.fail();
    }
    return symbolsVerified.get();
  };

  verified.check(verifyRequiredMethodsDeclared(
      classOp, contracts::kManifestRequiredMethodsAttr));
  verified.check(
      verifyRuntimeRequiredDeallocator(classOp, module, contractName));
  verified.check(checkSymbols(contracts::kManifestRequiredInitializersAttr,
                              "initializer"));
  verified.check(
      checkSymbols(contracts::kManifestRequiredMethodsAttr, "method"));
  verified.check(
      checkSymbols(contracts::kManifestRequiredPrimitivesAttr, "primitive"));

  return verified.get();
}

mlir::LogicalResult
verifyRuntimeManifestCompletenessImpl(mlir::ModuleOp module) {
  RuntimeManifestIndex manifest(module);
  if (mlir::failed(manifest.verify()))
    return mlir::failure();

  return walkVerify<py::ClassOp>(module, [&](py::ClassOp classOp) {
    if (!classOp->hasAttr(contracts::kManifestRequiredAttr))
      return mlir::success();
    return verifyRuntimeRequiredClass(classOp, module, manifest);
  });
}

class RuntimeManifestCompletenessVerifierPass
    : public mlir::PassWrapper<RuntimeManifestCompletenessVerifierPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      RuntimeManifestCompletenessVerifierPass)

  llvm::StringRef getArgument() const final {
    return "lython-runtime-manifest-completeness-verifier";
  }
  llvm::StringRef getDescription() const final {
    return "verify runtime-required typing contracts against runtime ABI "
           "manifest symbols";
  }

  void runOnOperation() final {
    if (mlir::failed(verifyRuntimeManifestCompletenessImpl(getOperation())))
      signalPassFailure();
  }
};

} // namespace

mlir::LogicalResult verifyRuntimeManifestCompleteness(mlir::ModuleOp module) {
  return verifyRuntimeManifestCompletenessImpl(module);
}

} // namespace py::lowering

namespace py {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createRuntimeManifestCompletenessVerifierPass() {
  return std::make_unique<
      lowering::RuntimeManifestCompletenessVerifierPass>();
}

mlir::LogicalResult verifyRuntimeManifestCompleteness(mlir::ModuleOp module) {
  return lowering::verifyRuntimeManifestCompleteness(module);
}

} // namespace py
