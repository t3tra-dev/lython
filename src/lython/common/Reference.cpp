#include "Reference.h"

#include "Contracts.h"

namespace py::ownership {

bool createsReferenceAtResult(FuncContractCache &contracts, mlir::Operation *op,
                              unsigned index) {
  auto call = mlir::dyn_cast_or_null<mlir::func::CallOp>(op);
  if (!call)
    return false;
  auto cached = contracts.lookup(call.getCallee());
  if (mlir::failed(cached) || !*cached)
    return false;
  return llvm::is_contained((*cached)->contract.ownedResults.values, index);
}

mlir::Operation *mintedReferenceOf(mlir::Operation *marker,
                                   AliasAnalysis &aliases) {
  // One predicate, one implementation. `ownedLocalMarkerIsRetainRooted` already
  // decides mint-versus-republish and is what the placer and the verifier have
  // agreed on since the token split; asking it again differently here is the
  // divergence the whole file exists to remove.
  if (!ownedLocalMarkerIsRetainRooted(marker, aliases))
    return nullptr;
  return marker->getPrevNode();
}

bool preservesReference(mlir::Operation *op) {
  if (!op)
    return false;
  if (op->hasAttr(kOwnedLocalObjectAttr))
    return false;
  llvm::StringRef name = op->getName().getStringRef();
  return name == "py.union.wrap" || name == "py.union.unwrap" ||
         name == "py.class.upcast" || name == "py.class.refine" ||
         name == "py.protocol.view" || name == "memref.cast" ||
         name == "memref.subview" ||
         name == "builtin.unrealized_conversion_cast";
}

mlir::Value continuedReferenceOf(FuncContractCache &contracts,
                                 mlir::Operation *op, unsigned index) {
  auto call = mlir::dyn_cast_or_null<mlir::func::CallOp>(op);
  if (!call)
    return {};
  auto cached = contracts.lookup(call.getCallee());
  if (mlir::failed(cached) || !*cached)
    return {};
  const FunctionContract &contract = (*cached)->contract;
  if (!llvm::is_contained(contract.ownedResults.values, index) ||
      contract.transferArgs.values.size() != 1)
    return {};
  unsigned transferred = contract.transferArgs.values.front();
  if (transferred >= call.getNumOperands())
    return {};
  return call.getOperand(transferred);
}

bool ReferenceMap::isMinted(Reference reference) const {
  auto call = mlir::dyn_cast_or_null<mlir::func::CallOp>(reference.creator);
  if (!call)
    return false;
  auto cached = contracts.lookup(call.getCallee());
  if (mlir::failed(cached) || !*cached)
    return false;
  auto primitive = (*cached)->function->getAttrOfType<mlir::StringAttr>(
      contracts::kManifestPrimitiveAttr);
  return primitive && primitive.getValue() == "retain";
}

Reference ReferenceMap::of(mlir::Value value) const {
  if (!value)
    return {};
  auto cached = cache.find(value);
  if (cached != cache.end())
    return cached->second;
  // Seeded before the recursion so a cyclic renaming answers "no claim" once
  // instead of recursing forever. Nothing in the shapes below is cyclic today;
  // the seed is what keeps that from being a precondition on the IR.
  cache[value] = Reference{};
  Reference computed = compute(value);
  cache[value] = computed;
  return computed;
}

Reference ReferenceMap::compute(mlir::Value value) const {
  auto result = mlir::dyn_cast<mlir::OpResult>(value);
  if (!result)
    return {}; // a block argument: which reference arrives is an edge property
  mlir::Operation *op = result.getOwner();

  if (createsReferenceAtResult(contracts, op, result.getResultNumber())) {
    // A move first: the same obligation continued, not a second one.
    if (mlir::Value continued =
            continuedReferenceOf(contracts, op, result.getResultNumber()))
      return of(continued);
    return Reference{op, result.getResultNumber()};
  }

  if (op->hasAttr(kOwnedLocalObjectAttr)) {
    if (mlir::Operation *retain = mintedReferenceOf(op, aliases))
      return Reference{retain, 0};
    // Republishes: adds no increment, so it denotes what it was given.
    unsigned index = result.getResultNumber();
    return index < op->getNumOperands() ? of(op->getOperand(index))
                                        : Reference{};
  }

  if (preservesReference(op)) {
    // A view narrows the spelling, not the reference: `memref.subview` takes a
    // header prefix and has one operand for many results' worth of shape, so the
    // positional map only holds where the arities agree.
    unsigned index = result.getResultNumber();
    if (op->getNumResults() == op->getNumOperands())
      return of(op->getOperand(index));
    if (op->getNumOperands() >= 1)
      return of(op->getOperand(0));
  }
  return {};
}

} // namespace py::ownership
