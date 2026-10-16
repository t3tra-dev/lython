#include "Passes/Runtime/Cleanup/Transforms.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/SmallVector.h"

namespace py::lowering::runtime::cleanup {

unsigned splitDualEdges(mlir::ModuleOp module) {
  llvm::SmallVector<mlir::cf::CondBranchOp, 8> dualEdges;
  module.walk([&](mlir::cf::CondBranchOp cond) {
    if (cond.getTrueDest() == cond.getFalseDest())
      dualEdges.push_back(cond);
  });
  for (mlir::cf::CondBranchOp cond : dualEdges) {
    mlir::Block *dest = cond.getTrueDest();
    mlir::Region *region = dest->getParent();
    mlir::OpBuilder builder(cond);
    auto makeEdgeBlock = [&](mlir::ValueRange operands) {
      auto *edge = new mlir::Block;
      region->getBlocks().insert(dest->getIterator(), edge);
      mlir::OpBuilder edgeBuilder(edge, edge->begin());
      mlir::cf::BranchOp::create(edgeBuilder, cond.getLoc(), dest, operands);
      return edge;
    };
    mlir::Block *trueEdge = makeEdgeBlock(cond.getTrueDestOperands());
    mlir::Block *falseEdge = makeEdgeBlock(cond.getFalseDestOperands());
    mlir::cf::CondBranchOp::create(builder, cond.getLoc(), cond.getCondition(),
                                   trueEdge, mlir::ValueRange{}, falseEdge,
                                   mlir::ValueRange{});
    cond.erase();
  }
  return dualEdges.size();
}

} // namespace py::lowering::runtime::cleanup
