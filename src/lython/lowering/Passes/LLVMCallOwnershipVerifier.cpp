#include "Common/LoweringUtils.h"
#include "Common/RuntimeSupport.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace py {
namespace {

struct LLVMCallOwnershipVerifierPass
    : public mlir::PassWrapper<LLVMCallOwnershipVerifierPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LLVMCallOwnershipVerifierPass)

  llvm::StringRef getArgument() const override {
    return "py-llvm-call-ownership-verify";
  }

  llvm::StringRef getDescription() const override {
    return "Verify the minimal lowered ownership-effect kernel";
  }

  void runOnOperation() override {
    if (mlir::failed(verifyLLVMCallOwnership(getOperation())))
      signalPassFailure();
  }
};

static bool hasAttr(mlir::Operation *op, llvm::StringRef attr) {
  return op && op->hasAttr(attr);
}

static llvm::SmallVector<unsigned> effectIndices(mlir::Operation *op,
                                                 llvm::StringRef attrName) {
  llvm::SmallVector<unsigned> result;
  for (int64_t index : lowering::attrs::i64Array(op, attrName))
    if (index >= 0)
      result.push_back(static_cast<unsigned>(index));
  return result;
}

static llvm::SmallVector<mlir::Value> callOperands(mlir::Operation *op) {
  llvm::SmallVector<mlir::Value> operands;
  if (auto call = mlir::dyn_cast<mlir::LLVM::CallOp>(op)) {
    operands.append(call.getCalleeOperands().begin(),
                    call.getCalleeOperands().end());
    return operands;
  }
  if (auto invoke = mlir::dyn_cast<mlir::LLVM::InvokeOp>(op)) {
    operands.append(invoke.getCalleeOperands().begin(),
                    invoke.getCalleeOperands().end());
    return operands;
  }
  operands.append(op->operand_begin(), op->operand_end());
  return operands;
}

static bool rawPointer(mlir::Type type) {
  return mlir::isa<mlir::LLVM::LLVMPointerType>(type);
}

static mlir::Value canonical(mlir::Value value) {
  while (value) {
    if (auto cast = value.getDefiningOp<mlir::memref::CastOp>()) {
      value = cast.getSource();
      continue;
    }
    if (auto bitcast = value.getDefiningOp<mlir::LLVM::BitcastOp>()) {
      value = bitcast.getArg();
      continue;
    }
    return value;
  }
  return value;
}

static bool hasRefcountEffect(mlir::Operation *op) {
  return !effectIndices(op, OwnershipContractAttrs::kRetainArgs).empty() ||
         !effectIndices(op, OwnershipContractAttrs::kReleaseArgs).empty() ||
         !effectIndices(op, OwnershipContractAttrs::kTransferArgs).empty();
}

static mlir::LogicalResult verifyEffectShape(mlir::Operation *op) {
  llvm::SmallVector<mlir::Value> operands = callOperands(op);
  unsigned operandCount = operands.size();
  unsigned resultCount = op->getNumResults();
  if (auto function = mlir::dyn_cast<mlir::FunctionOpInterface>(op)) {
    operandCount = function.getNumArguments();
    resultCount = function.getNumResults();
  }

  if (hasRefcountEffect(op)) {
    bool rawSinglePointer =
        operands.size() == 1 && rawPointer(operands.front().getType());
    if (auto function = mlir::dyn_cast<mlir::FunctionOpInterface>(op))
      rawSinglePointer =
          rawSinglePointer || (function.getNumArguments() == 1 &&
                               rawPointer(function.getArgumentTypes().front()));
    if (rawSinglePointer)
      return op->emitError()
             << "raw single-pointer retain/release/transfer ABI reached "
                "ownership verifier; object ownership must be represented as "
                "a typed memref descriptor before LLVM pointer-lane lowering";
  }

  auto verifyOperandIndices = [&](llvm::StringRef attr) -> mlir::LogicalResult {
    for (unsigned index : effectIndices(op, attr)) {
      if (index >= operandCount)
        return op->emitOpError() << attr << " index " << index
                                 << " exceeds operand count " << operandCount;
    }
    return mlir::success();
  };

  auto verifyResultIndices = [&](llvm::StringRef attr) -> mlir::LogicalResult {
    for (unsigned index : effectIndices(op, attr))
      if (index >= resultCount)
        return op->emitOpError() << attr << " index " << index
                                 << " exceeds result count " << resultCount;
    return mlir::success();
  };

  if (mlir::failed(verifyOperandIndices(OwnershipContractAttrs::kRetainArgs)) ||
      mlir::failed(
          verifyOperandIndices(OwnershipContractAttrs::kReleaseArgs)) ||
      mlir::failed(
          verifyOperandIndices(OwnershipContractAttrs::kTransferArgs)) ||
      mlir::failed(
          verifyOperandIndices(OwnershipContractAttrs::kSetFieldValueArg)) ||
      mlir::failed(
          verifyOperandIndices(OwnershipContractAttrs::kSetFieldRetainArg)) ||
      mlir::failed(
          verifyOperandIndices(OwnershipContractAttrs::kGetFieldBorrowArg)) ||
      mlir::failed(
          verifyResultIndices(OwnershipContractAttrs::kOwnedResults)) ||
      mlir::failed(
          verifyResultIndices(OwnershipContractAttrs::kBorrowedResults)) ||
      mlir::failed(
          verifyResultIndices(OwnershipContractAttrs::kGetFieldOwnedResult)))
    return mlir::failure();
  return mlir::success();
}

static mlir::LogicalResult verifyBlockQuantities(mlir::Block &block) {
  llvm::DenseMap<mlir::Value, int> produced;
  llvm::DenseSet<mlir::Value> externallyConsumed;

  auto produce = [&](mlir::Value value) { ++produced[canonical(value)]; };
  auto consume = [&](mlir::Operation *op,
                     mlir::Value value) -> mlir::LogicalResult {
    mlir::Value key = canonical(value);
    int &available = produced[key];
    if (available > 0) {
      --available;
      return mlir::success();
    }
    if (!externallyConsumed.insert(key).second)
      return op->emitOpError("consumes the same ownership carrier twice in one "
                             "basic block without an intervening retain or "
                             "owned-result producer");
    return mlir::success();
  };

  for (mlir::Operation &op : block) {
    if (mlir::failed(verifyEffectShape(&op)))
      return mlir::failure();

    llvm::SmallVector<mlir::Value> operands = callOperands(&op);

    if (hasAttr(&op, OwnershipContractAttrs::kOwnedLocalObject))
      for (mlir::Value result : op.getResults())
        produce(result);

    for (unsigned index :
         effectIndices(&op, OwnershipContractAttrs::kOwnedResults))
      if (index < op.getNumResults())
        produce(op.getResult(index));

    for (unsigned index :
         effectIndices(&op, OwnershipContractAttrs::kRetainArgs))
      if (index < operands.size())
        produce(operands[index]);

    for (unsigned index :
         effectIndices(&op, OwnershipContractAttrs::kReleaseArgs))
      if (index < operands.size() &&
          mlir::failed(consume(&op, operands[index])))
        return mlir::failure();

    for (unsigned index :
         effectIndices(&op, OwnershipContractAttrs::kTransferArgs))
      if (index < operands.size() &&
          mlir::failed(consume(&op, operands[index])))
        return mlir::failure();
  }

  return mlir::success();
}

static mlir::LogicalResult verifyRegion(mlir::Region &region) {
  for (mlir::Block &block : region) {
    if (mlir::failed(verifyBlockQuantities(block)))
      return mlir::failure();
    for (mlir::Operation &op : block)
      for (mlir::Region &nested : op.getRegions())
        if (mlir::failed(verifyRegion(nested)))
          return mlir::failure();
  }
  return mlir::success();
}

} // namespace

mlir::LogicalResult verifyLLVMCallOwnership(mlir::ModuleOp module) {
  if (mlir::failed(
          lowering::verifyNoUnrealizedCasts(module, "LLVM ownership verifier")))
    return mlir::failure();
  for (mlir::Region &region : module->getRegions())
    if (mlir::failed(verifyRegion(region)))
      return mlir::failure();
  return mlir::success();
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createLLVMCallOwnershipVerifierPass() {
  return std::make_unique<LLVMCallOwnershipVerifierPass>();
}

} // namespace py
