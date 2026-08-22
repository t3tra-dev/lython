#include "Runtime/Cleanup/Transforms.h"

#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace py::lowering::runtime::cleanup {

namespace {

// Every symbol named anywhere inside `op`, including from attributes -- the
// same surface MLIR's own symbol DCE consults, so a function this walk calls
// unreachable is one DCE would delete.
void collectSymbolUses(mlir::Operation *op, llvm::StringSet<> &named) {
  op->walk([&](mlir::Operation *nested) {
    for (mlir::NamedAttribute attr : nested->getAttrs()) {
      attr.getValue().walk([&](mlir::SymbolRefAttr symbol) {
        named.insert(symbol.getRootReference().getValue());
      });
    }
  });
}

} // namespace

unsigned stripUnreachableManifestBodies(mlir::ModuleOp module) {
  llvm::DenseSet<mlir::Operation *> reachable;
  llvm::SmallVector<mlir::func::FuncOp, 16> pending;
  for (mlir::func::FuncOp function : module.getOps<mlir::func::FuncOp>()) {
    if (mlir::SymbolTable::getSymbolVisibility(function) ==
            mlir::SymbolTable::Visibility::Public &&
        reachable.insert(function.getOperation()).second)
      pending.push_back(function);
  }
  // Anything a non-function op names is a root too: a global initializer or a
  // strategy attribute can be the only reference a body has.
  llvm::StringSet<> rootNames;
  for (mlir::Operation &op : module.getBody()->getOperations())
    if (!mlir::isa<mlir::func::FuncOp>(op))
      collectSymbolUses(&op, rootNames);
  for (mlir::func::FuncOp function : module.getOps<mlir::func::FuncOp>())
    if (rootNames.contains(function.getSymName()) &&
        reachable.insert(function.getOperation()).second)
      pending.push_back(function);

  while (!pending.empty()) {
    mlir::func::FuncOp current = pending.pop_back_val();
    llvm::StringSet<> named;
    collectSymbolUses(current.getOperation(), named);
    for (const auto &entry : named) {
      auto callee = module.lookupSymbol<mlir::func::FuncOp>(entry.getKey());
      if (callee && reachable.insert(callee.getOperation()).second)
        pending.push_back(callee);
    }
  }

  unsigned stripped = 0;
  for (mlir::func::FuncOp function : module.getOps<mlir::func::FuncOp>()) {
    if (function.isDeclaration() || reachable.contains(function.getOperation()))
      continue;
    function.eraseBody();
    ++stripped;
  }
  return stripped;
}

} // namespace py::lowering::runtime::cleanup
