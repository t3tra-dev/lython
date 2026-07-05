#include "Runtime/Model/Contracts.h"
#include "Ownership.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"

#include <memory>

namespace py::lowering {
namespace {

namespace own = py::ownership;

bool sameHeaderOperand(own::AliasAnalysis &aliases, mlir::func::CallOp retain,
                       mlir::func::CallOp release) {
  if (retain.getNumOperands() != 1 || release.getNumOperands() == 0)
    return false;
  return aliases.same(retain.getOperand(0), release.getOperand(0));
}

bool isRetainCall(mlir::ModuleOp module, mlir::func::CallOp call) {
  mlir::func::FuncOp callee =
      module.lookupSymbol<mlir::func::FuncOp>(call.getCallee());
  return callee && own::functionRetainsOperandAt(callee, 0);
}

bool isReleaseCall(mlir::ModuleOp module, mlir::func::CallOp call) {
  mlir::func::FuncOp callee =
      module.lookupSymbol<mlir::func::FuncOp>(call.getCallee());
  return callee && callee->hasAttr(kManifestDeallocatorAttr) &&
         own::functionReleasesOperandAt(callee, 0);
}

bool hasAggregateOwnershipMarker(mlir::func::CallOp call) {
  auto marker = own::readAggregateOwnershipMarker(call);
  return mlir::succeeded(marker) && marker->has_value();
}

void elideAdjacentRefcountPairs(mlir::ModuleOp module) {
  own::AliasAnalysis aliases;
  aliases.build(module);

  llvm::SmallVector<mlir::Operation *, 8> erase;
  module.walk([&](mlir::func::FuncOp function) {
    for (mlir::Block &block : function) {
      for (auto it = block.begin(), e = block.end(); it != e;) {
        mlir::Operation *current = &*it++;
        auto retain = mlir::dyn_cast<mlir::func::CallOp>(current);
        if (!retain || !isRetainCall(module, retain) || it == e)
          continue;
        if (hasAggregateOwnershipMarker(retain))
          continue;
        mlir::Operation *next = &*it;
        auto release = mlir::dyn_cast<mlir::func::CallOp>(next);
        if (!release || !isReleaseCall(module, release))
          continue;
        if (hasAggregateOwnershipMarker(release))
          continue;
        if (!sameHeaderOperand(aliases, retain, release))
          continue;
        if (retain.getResultTypes().empty() &&
            release.getResultTypes().empty()) {
          erase.push_back(retain.getOperation());
          erase.push_back(release.getOperation());
        }
      }
    }
  });

  for (mlir::Operation *op : llvm::reverse(erase))
    op->erase();
}

class RefCountPairElisionPass
    : public mlir::PassWrapper<RefCountPairElisionPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RefCountPairElisionPass)

  llvm::StringRef getArgument() const final {
    return "lython-refcount-elision";
  }
  llvm::StringRef getDescription() const final {
    return "elide adjacent retain/release pairs proven by shared alias "
           "analysis";
  }

  void runOnOperation() final { elideAdjacentRefcountPairs(getOperation()); }
};

} // namespace
} // namespace py::lowering

namespace py {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createRefCountPairElisionPass() {
  return std::make_unique<lowering::RefCountPairElisionPass>();
}

} // namespace py
