#include "Passes/Runtime/Async.h"

#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

namespace py {
namespace {

bool stackMemRef(mlir::Value value, llvm::SmallPtrSetImpl<mlir::Value> &seen) {
  if (!value || !seen.insert(value).second)
    return false;
  if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    mlir::Block *owner = arg.getOwner();
    if (!owner)
      return false;
    for (mlir::Block *predecessor : owner->getPredecessors()) {
      mlir::Operation *terminator = predecessor->getTerminator();
      auto branch = mlir::dyn_cast_or_null<mlir::BranchOpInterface>(terminator);
      if (!branch)
        continue;
      for (unsigned i = 0, e = branch->getNumSuccessors(); i != e; ++i) {
        if (branch->getSuccessor(i) != owner)
          continue;
        mlir::SuccessorOperands operands = branch.getSuccessorOperands(i);
        unsigned argIndex = arg.getArgNumber();
        if (argIndex >= operands.size() || operands.isOperandProduced(argIndex))
          continue;
        if (stackMemRef(operands[argIndex], seen))
          return true;
      }
    }
    return false;
  }
  mlir::Operation *def = value.getDefiningOp();
  if (!def)
    return false;
  if (mlir::isa<mlir::memref::AllocaOp>(def))
    return true;
  if (auto cast = mlir::dyn_cast<mlir::UnrealizedConversionCastOp>(def)) {
    for (mlir::Value operand : cast.getOperands())
      if (stackMemRef(operand, seen))
        return true;
  }
  if (auto cast = mlir::dyn_cast<mlir::memref::CastOp>(def))
    return stackMemRef(cast.getSource(), seen);
  if (auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(def)) {
    auto result = mlir::dyn_cast<mlir::OpResult>(value);
    if (!result)
      return false;
    unsigned resultIndex = result.getResultNumber();
    bool captures = false;
    auto checkYield = [&](mlir::scf::YieldOp yield) {
      if (resultIndex >= yield.getNumOperands())
        return;
      if (stackMemRef(yield.getOperand(resultIndex), seen))
        captures = true;
    };
    ifOp.getThenRegion().walk(checkYield);
    ifOp.getElseRegion().walk(checkYield);
    return captures;
  }
  return false;
}

} // namespace

namespace lowering::runtime::async {

mlir::LogicalResult verifyReturnPayloads(mlir::ModuleOp module) {
  bool failedAny = false;
  module.walk([&](mlir::async::ReturnOp op) {
    for (mlir::Value operand : op.getOperands()) {
      llvm::SmallPtrSet<mlir::Value, 8> seen;
      if (!stackMemRef(operand, seen))
        continue;
      op->emitOpError("captures a stack memref in an async result payload; "
                      "escaping async payloads must be promoted to managed "
                      "heap descriptors before async.return");
      failedAny = true;
      return;
    }
  });
  return mlir::failure(failedAny);
}

} // namespace lowering::runtime::async

} // namespace py
