#include "runtime/Verification.h"

#include "Contracts.h"
#include "Ownership.h"
#include "runtime/Detail.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

#include <memory>
#include <optional>

namespace py::lowering {
namespace {

namespace own = py::ownership;
namespace contracts = py::contracts;

bool isRawObjectHeaderABI(mlir::Type type) {
  auto memref = mlir::dyn_cast<mlir::MemRefType>(type);
  if (!memref || memref.getRank() != 1 || !memref.hasStaticShape() ||
      memref.getDimSize(0) != 2)
    return false;
  return isIntegerType(memref.getElementType(), 64);
}

mlir::LogicalResult verifyAggregateOwnershipCallee(
    mlir::Operation *op, const own::AggregateOwnershipMarker &marker,
    llvm::StringRef calleeName, mlir::Operation *callee) {
  if (!callee)
    return op->emitError() << "aggregate ownership call target @" << calleeName
                           << " is not a function";

  if (marker.action == own::AggregateOwnershipAction::Retain) {
    auto retained = own::parseIndexSetAttr(callee, own::kRetainArgsAttr);
    if (mlir::failed(retained))
      return mlir::failure();
    if (!retained->contains(0))
      return op->emitError() << own::kAggregateRetainAttr
                             << " call target must retain operand 0";
    return mlir::success();
  }

  auto released = own::parseIndexSetAttr(callee, own::kReleaseArgsAttr);
  if (mlir::failed(released))
    return mlir::failure();
  auto transferred = own::parseIndexSetAttr(callee, own::kTransferArgsAttr);
  if (mlir::failed(transferred))
    return mlir::failure();
  if (!released->contains(0) && !transferred->contains(0))
    return op->emitError() << own::kAggregateReleaseAttr
                           << " call target must release or transfer operand 0";
  return mlir::success();
}

mlir::LogicalResult verifyFunctionOwnershipShape(mlir::func::FuncOp function) {
  auto contract = own::readFunctionContract(function);
  if (mlir::failed(contract))
    return mlir::failure();

  for (unsigned index : contract->ownedResults.values) {
    if (contract->borrowedResults.contains(index))
      return function.emitError()
             << "result " << index
             << " cannot be both owned_results and borrowed_results";
    if (!own::isObjectHeaderLikeType(
            function.getFunctionType().getResult(index)))
      return function.emitError()
             << own::kOwnedResultsAttr << " result " << index
             << " must start an object-header-like result group";
  }

  for (unsigned index : contract->borrowedResults.values) {
    if (!own::isObjectHeaderLikeType(
            function.getFunctionType().getResult(index)))
      return function.emitError()
             << own::kBorrowedResultsAttr << " result " << index
             << " must start an object-header-like result group";
  }

  auto verifyObjectArg = [&](unsigned index,
                             llvm::StringRef attrName) -> mlir::LogicalResult {
    if (!own::isObjectHeaderLikeType(function.getFunctionType().getInput(index)))
      return function.emitError() << attrName << " argument " << index
                                  << " must be an object-header-like memref";
    if (!function.getArgAttr(index, own::kObjectHeaderAttr))
      return function.emitError() << attrName << " argument " << index
                                  << " must carry " << own::kObjectHeaderAttr;
    return mlir::success();
  };

  auto verifyConcreteConsumerArg =
      [&](unsigned index, llvm::StringRef attrName) -> mlir::LogicalResult {
    auto manifestContract =
        function->getAttrOfType<mlir::StringAttr>(
            contracts::kManifestContractAttr);
    if (!manifestContract || manifestContract.getValue() != "builtins.object")
      return mlir::success();
    if (!isRawObjectHeaderABI(function.getFunctionType().getInput(index)))
      return mlir::success();
    return function.emitError()
           << attrName << " argument " << index
           << " consumes only a raw builtins.object header; consuming object "
              "ownership requires a concrete runtime value group or a boxed "
              "object handle";
  };

  for (unsigned index : contract->retainArgs.values)
    if (mlir::failed(verifyObjectArg(index, own::kRetainArgsAttr)))
      return mlir::failure();
  for (unsigned index : contract->releaseArgs.values) {
    if (mlir::failed(verifyObjectArg(index, own::kReleaseArgsAttr)))
      return mlir::failure();
    if (mlir::failed(
            verifyConcreteConsumerArg(index, own::kReleaseArgsAttr)))
      return mlir::failure();
  }
  for (unsigned index : contract->transferArgs.values) {
    if (mlir::failed(verifyObjectArg(index, own::kTransferArgsAttr)))
      return mlir::failure();
    if (mlir::failed(
            verifyConcreteConsumerArg(index, own::kTransferArgsAttr)))
      return mlir::failure();
  }

  if (function->hasAttr(contracts::kManifestDeallocatorAttr) &&
      contract->releaseArgs.empty())
    return function.emitError()
           << "runtime deallocator must declare release_args";

  if (contract->objectReleaseToZero) {
    if (function.getFunctionType().getNumResults() != 1 ||
        !isIntegerType(function.getFunctionType().getResult(0), 1))
      return function.emitError()
             << own::kObjectReleaseToZeroAttr
             << " function must return one i1 release-to-zero flag";
  }

  return mlir::success();
}

mlir::LogicalResult verifyOperationOwnershipShape(mlir::Operation *op) {
  if (op->hasAttr(own::kObjectDeallocPartAttr)) {
    if (op->getName().getStringRef() != "memref.dealloc")
      return op->emitError() << own::kObjectDeallocPartAttr
                             << " is only valid on memref.dealloc";
    if (!mlir::isa<mlir::StringAttr>(op->getAttr(own::kObjectDeallocPartAttr)))
      return op->emitError()
             << own::kObjectDeallocPartAttr << " must be a string attribute";
  }

  if (op->hasAttr(own::kOwnedLocalObjectAttr)) {
    if (op->getNumResults() == 0 ||
        !own::isObjectHeaderLikeType(op->getResult(0).getType()))
      return op->emitError()
             << own::kOwnedLocalObjectAttr
             << " must mark an operation producing an object header";
  }

  auto aggregate = own::readAggregateOwnershipMarker(op);
  if (mlir::failed(aggregate))
    return mlir::failure();
  if (*aggregate) {
    auto call = mlir::dyn_cast<mlir::func::CallOp>(op);
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (call) {
      if (call.getNumOperands() == 0)
        return op->emitError()
               << "aggregate ownership call must have at least one operand";
      mlir::Operation *callee =
          module ? module.lookupSymbol(call.getCallee()) : nullptr;
      return verifyAggregateOwnershipCallee(op, **aggregate, call.getCallee(),
                                            callee);
    }

    auto llvmCall = mlir::dyn_cast<mlir::LLVM::CallOp>(op);
    if (!llvmCall)
      return op->emitError()
             << "aggregate ownership marker must be attached to a call";
    if (llvmCall.getNumOperands() == 0)
      return op->emitError()
             << "aggregate ownership call must have at least one operand";
    std::optional<llvm::StringRef> calleeName = llvmCall.getCallee();
    if (!calleeName)
      return op->emitError()
             << "aggregate ownership marker requires a direct call target";
    mlir::Operation *callee =
        module ? module.lookupSymbol(*calleeName) : nullptr;
    return verifyAggregateOwnershipCallee(op, **aggregate, *calleeName, callee);
  }

  return mlir::success();
}

mlir::LogicalResult verifyOwnershipContractShapesImpl(mlir::ModuleOp module) {
  if (mlir::failed(
          walkVerify<mlir::func::FuncOp>(module, verifyFunctionOwnershipShape)))
    return mlir::failure();
  return walkVerifyOperations(module, verifyOperationOwnershipShape);
}

class OwnershipVerifierPass
    : public mlir::PassWrapper<OwnershipVerifierPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OwnershipVerifierPass)

  llvm::StringRef getArgument() const final {
    return "lython-ownership-verifier";
  }
  llvm::StringRef getDescription() const final {
    return "verify Lython ownership contracts and local affine sinks";
  }

  void runOnOperation() final {
    if (mlir::failed(verifyOwnership(getOperation())))
      signalPassFailure();
  }
};

class LLVMCallOwnershipVerifierPass
    : public mlir::PassWrapper<LLVMCallOwnershipVerifierPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LLVMCallOwnershipVerifierPass)

  llvm::StringRef getArgument() const final {
    return "lython-llvm-call-ownership-verifier";
  }
  llvm::StringRef getDescription() const final {
    return "verify lowered call ownership contracts";
  }

  void runOnOperation() final {
    if (mlir::failed(verifyLLVMCallOwnership(getOperation())))
      signalPassFailure();
  }
};

} // namespace

mlir::LogicalResult verifyOwnershipContractShapes(mlir::ModuleOp module) {
  return verifyOwnershipContractShapesImpl(module);
}

} // namespace py::lowering

namespace py {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createOwnershipVerifierPass() {
  return std::make_unique<lowering::OwnershipVerifierPass>();
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createLLVMCallOwnershipVerifierPass() {
  return std::make_unique<lowering::LLVMCallOwnershipVerifierPass>();
}

mlir::LogicalResult verifyOwnership(mlir::ModuleOp module) {
  if (mlir::failed(lowering::verifyOwnershipContractShapes(module)))
    return mlir::failure();
  return mlir::success();
}

mlir::LogicalResult verifyLLVMCallOwnership(mlir::ModuleOp module) {
  if (mlir::failed(lowering::verifyOwnershipContractShapes(module)))
    return mlir::failure();
  if (mlir::failed(
          lowering::verifyLLVMOwnershipContractShapes(module)))
    return mlir::failure();
  if (mlir::failed(lowering::verifyLLVMCallOwnershipContracts(module)))
    return mlir::failure();
  return lowering::verifyFuncCallOwnershipContracts(module);
}

} // namespace py
