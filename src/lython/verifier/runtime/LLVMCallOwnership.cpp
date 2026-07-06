#include "runtime/Detail.h"

#include "Contracts.h"
#include "Ownership.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/StringMap.h"

#include <optional>

namespace py::lowering {
namespace {

namespace own = py::ownership;
namespace contracts = py::contracts;

mlir::FailureOr<own::FunctionContract>
readLLVMFunctionContract(mlir::LLVM::LLVMFuncOp function) {
  own::FunctionContract contract;
  unsigned numInputs = function.getFunctionType().getNumParams();

  auto owned =
      own::parseIndexSetAttr(function, own::kOwnedResultsAttr,
                             /*upperBound=*/std::nullopt);
  if (mlir::failed(owned))
    return mlir::failure();
  contract.ownedResults = *owned;

  auto borrowed =
      own::parseIndexSetAttr(function, own::kBorrowedResultsAttr,
                             /*upperBound=*/std::nullopt);
  if (mlir::failed(borrowed))
    return mlir::failure();
  contract.borrowedResults = *borrowed;

  auto retained =
      own::parseIndexSetAttr(function, own::kRetainArgsAttr, numInputs);
  if (mlir::failed(retained))
    return mlir::failure();
  contract.retainArgs = *retained;

  auto released =
      own::parseIndexSetAttr(function, own::kReleaseArgsAttr, numInputs);
  if (mlir::failed(released))
    return mlir::failure();
  contract.releaseArgs = *released;

  auto transferred =
      own::parseIndexSetAttr(function, own::kTransferArgsAttr, numInputs);
  if (mlir::failed(transferred))
    return mlir::failure();
  contract.transferArgs = *transferred;

  contract.objectReleaseToZero =
      function->hasAttr(own::kObjectReleaseToZeroAttr);

  for (unsigned index : contract.releaseArgs.values) {
    if (contract.transferArgs.contains(index))
      return function.emitError()
             << "argument " << index
             << " cannot be both release_args and transfer_args";
  }

  return contract;
}

bool isSingleAggregateLLVMResult(llvm::ArrayRef<mlir::Type> resultTypes) {
  return resultTypes.size() == 1 &&
         mlir::isa<mlir::LLVM::LLVMStructType, mlir::LLVM::LLVMArrayType>(
             resultTypes.front());
}

mlir::LogicalResult
verifyLLVMResultOwnershipIndices(mlir::Operation *op,
                                 llvm::ArrayRef<mlir::Type> resultTypes,
                                 const own::IndexSet &indices,
                                 llvm::StringRef attrName) {
  if (indices.empty())
    return mlir::success();
  if (resultTypes.empty())
    return op->emitError() << attrName
                           << " cannot be attached to a void LLVM result";

  bool flattenedAggregate = isSingleAggregateLLVMResult(resultTypes);
  for (unsigned index : indices.values) {
    if (resultTypes.size() > 1 && index >= resultTypes.size())
      return op->emitError() << attrName << " index " << index
                             << " is out of range [0, "
                             << resultTypes.size() << ")";
    if (resultTypes.size() == 1 && index > 0 && !flattenedAggregate)
      return op->emitError()
             << attrName << " index " << index
             << " requires a flattened aggregate LLVM result";
  }
  return mlir::success();
}

bool hasLLVMOwnershipSurface(mlir::LLVM::LLVMFuncOp function,
                             const own::FunctionContract &contract) {
  return contract.hasAnyOwnershipAttr() ||
         function->hasAttr(contracts::kManifestDeallocatorAttr);
}

struct CachedLLVMFunctionContract {
  mlir::LLVM::LLVMFuncOp function;
  own::FunctionContract contract;
  bool hasOwnershipSurface = false;
};

class LLVMFunctionOwnershipCache {
public:
  explicit LLVMFunctionOwnershipCache(mlir::ModuleOp module) {
    module.walk([&](mlir::LLVM::LLVMFuncOp function) {
      functions.insert({function.getSymName(), function});
    });
  }

  mlir::FailureOr<const CachedLLVMFunctionContract *>
  lookup(llvm::StringRef name) {
    auto cached = contracts.find(name);
    if (cached != contracts.end())
      return &cached->second;

    auto function = functions.find(name);
    if (function == functions.end())
      return static_cast<const CachedLLVMFunctionContract *>(nullptr);

    auto contract = readLLVMFunctionContract(function->second);
    if (mlir::failed(contract))
      return mlir::failure();

    CachedLLVMFunctionContract entry{function->second, *contract,
                                     hasLLVMOwnershipSurface(function->second,
                                                            *contract)};
    auto inserted = contracts.insert({name, std::move(entry)});
    return &inserted.first->second;
  }

private:
  llvm::StringMap<mlir::LLVM::LLVMFuncOp> functions;
  llvm::StringMap<CachedLLVMFunctionContract> contracts;
};

mlir::LogicalResult
verifyLLVMFunctionOwnershipShape(mlir::LLVM::LLVMFuncOp function) {
  auto contract = readLLVMFunctionContract(function);
  if (mlir::failed(contract))
    return mlir::failure();

  llvm::ArrayRef<mlir::Type> resultTypes = function.getResultTypes();
  for (unsigned index : contract->ownedResults.values) {
    if (contract->borrowedResults.contains(index))
      return function.emitError()
             << "result " << index
             << " cannot be both owned_results and borrowed_results";
  }
  if (mlir::failed(verifyLLVMResultOwnershipIndices(
          function, resultTypes, contract->ownedResults,
          own::kOwnedResultsAttr)))
    return mlir::failure();
  if (mlir::failed(verifyLLVMResultOwnershipIndices(
          function, resultTypes, contract->borrowedResults,
          own::kBorrowedResultsAttr)))
    return mlir::failure();

  if (function->hasAttr(contracts::kManifestDeallocatorAttr) &&
      contract->releaseArgs.empty())
    return function.emitError()
           << "runtime deallocator must declare release_args";

  if (contract->objectReleaseToZero) {
    if (resultTypes.size() != 1 || !isIntegerType(resultTypes.front(), 1))
      return function.emitError()
             << own::kObjectReleaseToZeroAttr
             << " function must return one i1 release-to-zero flag";
  }

  return mlir::success();
}

mlir::LogicalResult verifyLLVMCallArgumentCount(mlir::LLVM::CallOp call,
                                                mlir::LLVM::LLVMFuncOp callee) {
  mlir::LLVM::LLVMFunctionType calleeType = callee.getFunctionType();
  unsigned numParams = calleeType.getNumParams();
  unsigned numOperands = call.getNumOperands();
  if (calleeType.isVarArg()) {
    if (numOperands < numParams)
      return call.emitError()
             << "direct call to @" << callee.getSymName()
             << " supplies " << numOperands << " operands for at least "
             << numParams << " LLVM parameters";
    return mlir::success();
  }

  if (numOperands != numParams)
    return call.emitError()
           << "direct call to @" << callee.getSymName() << " supplies "
           << numOperands << " operands for " << numParams
           << " LLVM parameters";
  return mlir::success();
}

mlir::LogicalResult verifyLLVMCallResultCount(mlir::LLVM::CallOp call,
                                              mlir::LLVM::LLVMFuncOp callee) {
  unsigned expected = callee.getResultTypes().size();
  if (call.getNumResults() != expected)
    return call.emitError()
           << "direct call to @" << callee.getSymName() << " produces "
           << call.getNumResults() << " results for " << expected
           << " LLVM callee results";
  return mlir::success();
}

mlir::LogicalResult
verifyLLVMConsumedArgumentIndices(mlir::LLVM::CallOp call,
                                  const own::IndexSet &indices,
                                  llvm::StringRef attrName) {
  for (unsigned index : indices.values) {
    if (index >= call.getNumOperands())
      return call.emitError() << attrName << " argument " << index
                              << " has no corresponding call operand";
  }
  return mlir::success();
}

mlir::LogicalResult
verifyLLVMCallOwnershipContract(LLVMFunctionOwnershipCache &cache,
                                mlir::LLVM::CallOp call) {
  std::optional<llvm::StringRef> calleeName = call.getCallee();
  if (!calleeName)
    return mlir::success();

  mlir::FailureOr<const CachedLLVMFunctionContract *> cached =
      cache.lookup(*calleeName);
  if (mlir::failed(cached))
    return mlir::failure();
  if (!*cached)
    return mlir::success();

  mlir::LLVM::LLVMFuncOp callee = (*cached)->function;
  const own::FunctionContract &contract = (*cached)->contract;
  if (!(*cached)->hasOwnershipSurface)
    return mlir::success();

  if (mlir::failed(verifyLLVMCallArgumentCount(call, callee)))
    return mlir::failure();
  if (mlir::failed(verifyLLVMCallResultCount(call, callee)))
    return mlir::failure();

  llvm::ArrayRef<mlir::Type> resultTypes = callee.getResultTypes();
  if (mlir::failed(verifyLLVMResultOwnershipIndices(
          call, resultTypes, contract.ownedResults, own::kOwnedResultsAttr)))
    return mlir::failure();
  if (mlir::failed(verifyLLVMResultOwnershipIndices(
          call, resultTypes, contract.borrowedResults,
          own::kBorrowedResultsAttr)))
    return mlir::failure();

  if (mlir::failed(verifyLLVMConsumedArgumentIndices(
          call, contract.retainArgs, own::kRetainArgsAttr)))
    return mlir::failure();
  if (mlir::failed(verifyLLVMConsumedArgumentIndices(
          call, contract.releaseArgs, own::kReleaseArgsAttr)))
    return mlir::failure();
  if (mlir::failed(verifyLLVMConsumedArgumentIndices(
          call, contract.transferArgs, own::kTransferArgsAttr)))
    return mlir::failure();

  return mlir::success();
}

} // namespace

mlir::LogicalResult verifyLLVMOwnershipContractShapes(mlir::ModuleOp module) {
  return walkVerify<mlir::LLVM::LLVMFuncOp>(
      module, verifyLLVMFunctionOwnershipShape);
}

mlir::LogicalResult verifyLLVMCallOwnershipContracts(mlir::ModuleOp module) {
  LLVMFunctionOwnershipCache cache(module);
  return walkVerify<mlir::LLVM::CallOp>(module, [&](mlir::LLVM::CallOp call) {
    return verifyLLVMCallOwnershipContract(cache, call);
  });
}

} // namespace py::lowering
