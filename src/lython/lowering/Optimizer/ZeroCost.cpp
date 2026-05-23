#include "Optimizer/Utils.h"

namespace py::optimizer {
namespace {

constexpr char kProvenLocalClassAccessAttr[] = "ly.proven_local_class_access";

void dropProofOnlyAttrs(mlir::Operation *op) {
  op->removeAttr(kProvenLocalClassAccessAttr);
  op->removeAttr("ly.known_local_class_access");
}

} // namespace

void zero_cost::rewriteLocalAccess(mlir::ModuleOp module) {
  llvm::SmallVector<AttrGetOp> attrGets;
  llvm::SmallVector<AttrSetOp> attrSets;

  module.walk([&](AttrGetOp op) {
    if (op->hasAttr(kProvenLocalClassAccessAttr))
      attrGets.push_back(op);
  });
  module.walk([&](AttrSetOp op) {
    if (op->hasAttr(kProvenLocalClassAccessAttr))
      attrSets.push_back(op);
  });

  for (AttrGetOp op : attrGets) {
    mlir::OpBuilder builder(op);
    auto local =
        builder.create<AttrGetLocalOp>(op.getLoc(), op.getResult().getType(),
                                       op.getObject(), op.getNameAttr());
    local->setAttrs(op->getAttrs());
    dropProofOnlyAttrs(local.getOperation());
    op.getResult().replaceAllUsesWith(local.getResult());
    op.erase();
  }

  for (AttrSetOp op : attrSets) {
    mlir::OpBuilder builder(op);
    auto local = builder.create<AttrSetLocalOp>(
        op.getLoc(), op.getObject(), op.getNameAttr(), op.getValue());
    local->setAttrs(op->getAttrs());
    dropProofOnlyAttrs(local.getOperation());
    op.erase();
  }
}

void pipeline::zeroCostProofPre(mlir::ModuleOp module) {
  class_state::proveLocalAccess(module);
}

void pipeline::zeroCostRewritePre(mlir::ModuleOp module) {
  zero_cost::rewriteLocalAccess(module);
}

} // namespace py::optimizer
