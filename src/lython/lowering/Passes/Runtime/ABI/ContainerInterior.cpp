#include "Ownership.h"

#include "Runtime/Core/Lowerer.h"

#include "Runtime/ABI/BoxLayout.h"
#include "Runtime/ABI/CollectionPayload.h"
#include "Runtime/ABI/ContainerLayout.h"

#include <cstddef>

namespace py::lowering {
namespace {

// A container handle is a rank-1 i64 memref wide enough for the layout in
// ContainerLayout.h. Checked rather than assumed: `builtins.object` handles
// and boxed slots are also rank-1 i64 memrefs, and width is not a proof of
// kind (rfc/memory-safety-proof.md, `Provenance`) -- this only rules out
// reading past the end of something that is not a container handle at all.
bool isContainerHandleType(mlir::Type type) {
  if (!ownership::isRankOneI64MemRef(type))
    return false;
  auto memref = mlir::cast<mlir::MemRefType>(type);
  return memref.hasStaticShape() &&
         memref.getDimSize(0) >= container_abi::kHandleWordCount;
}

} // namespace

bool RuntimeBundleLowerer::containerIsHandleFronted(
    const RuntimeBundle &container) const {
  llvm::ArrayRef<mlir::Value> values = container.physicalValues();
  return values.size() == 1 && isContainerHandleType(values.front().getType());
}

// True when `container` carries a runtime payload the lowering can read or
// write. This is the gate every runtime-mode container path gets to before it
// touches the payload, and it is spelled once because the answer is a LANE
// COUNT for a lane-carrying contract and a HANDLE LAYOUT for a handle-fronted
// one. A gate that spelled the lane count itself would not fail when a
// contract converts -- it would quietly decide the container has no payload
// and take the evidence path, which is the "silently mis-execute" shape.
bool RuntimeBundleLowerer::containerHasRuntimePayload(
    const RuntimeBundle &container) const {
  if (RuntimeBundleLowerer::containerIsHandleFronted(container))
    return true;
  std::size_t lanes = container.contractName() == "builtins.dict" ? 5u : 3u;
  return container.physicalValues().size() >= lanes;
}

// Slot of the {length, capacity} pair, and the memref it lives in. A
// lane-carrying contract keeps the pair in its own `meta` lane; a
// handle-fronted one keeps it in words 2/3 of the handle, in the same order,
// so only the base and the slot offset differ.
mlir::FailureOr<std::pair<mlir::Value, mlir::Value>>
RuntimeBundleLowerer::containerMetaSlot(mlir::Operation *op,
                                        const RuntimeBundle &container,
                                        std::int64_t slot,
                                        llvm::StringRef label) {
  llvm::ArrayRef<mlir::Value> values = container.physicalValues();
  mlir::Location loc = op->getLoc();
  if (RuntimeBundleLowerer::containerIsHandleFronted(container)) {
    mlir::Value word = mlir::arith::ConstantIndexOp::create(
                           builder, loc, container_abi::kLengthWord + slot)
                           .getResult();
    return std::make_pair(values.front(), word);
  }
  if (values.size() < 2)
    return op->emitError() << label
                           << " collection has no physical length metadata";
  if (!collection_abi::isCollectionMetaType(values[1].getType()))
    return op->emitError() << label
                           << " collection length metadata has invalid type "
                           << values[1].getType();
  mlir::Value index =
      mlir::arith::ConstantIndexOp::create(builder, loc, slot).getResult();
  return std::make_pair(values[1], index);
}

mlir::FailureOr<mlir::Value>
RuntimeBundleLowerer::loadContainerLength(mlir::Operation *op,
                                          const RuntimeBundle &container,
                                          llvm::StringRef label) {
  mlir::FailureOr<std::pair<mlir::Value, mlir::Value>> slot =
      RuntimeBundleLowerer::containerMetaSlot(
          op, container, container_abi::kMetaLengthSlot, label);
  if (mlir::failed(slot))
    return mlir::failure();
  return mlir::memref::LoadOp::create(builder, op->getLoc(), slot->first,
                                      slot->second)
      .getResult();
}

// Stamp a dict's index table stale when the length being written is SMALLER
// than the one already there.
//
// ⭐ The one thing the manifest cannot work out for itself. `__ly_dict_table_sync`
// compares the table's stamp against the length and rebuilds when they differ,
// which catches every dict this pass GROWS -- a literal, an append -- and costs
// only the new tail. A delete is invisible to it: the dense array shifts down,
// so every index the table holds past the hole is wrong, and a delete that
// follows an insert puts the length back exactly where it started. The
// comparison then sees a table that matches and a table that lies.
//
// Why NOT announce every write instead: this pass writes a dict's arrays from
// six places and the manifest from six more, and the announcement would have to
// be remembered at each. The length is the one word all twelve go through.
mlir::LogicalResult RuntimeBundleLowerer::invalidateMappingTableOnShrink(
    mlir::Operation *op, const RuntimeBundle &container, mlir::Value length) {
  if (container.contractName() != "builtins.dict")
    return mlir::success();
  if (!RuntimeBundleLowerer::containerIsHandleFronted(container))
    return mlir::success();
  llvm::ArrayRef<mlir::Value> values = container.physicalValues();
  if (values.empty())
    return mlir::success();
  mlir::Location loc = op->getLoc();
  mlir::Type i64 = builder.getI64Type();
  mlir::Value handle = values.front();
  mlir::Value lengthWord = mlir::arith::ConstantIndexOp::create(
                               builder, loc, container_abi::kLengthWord)
                               .getResult();
  mlir::Value previous =
      mlir::memref::LoadOp::create(builder, loc, handle, lengthWord)
          .getResult();
  mlir::Value tableWord = mlir::arith::ConstantIndexOp::create(
                              builder, loc, container_abi::kTableWord)
                              .getResult();
  mlir::Value tableAddress =
      mlir::memref::LoadOp::create(builder, loc, handle, tableWord).getResult();
  mlir::Value one = mlir::arith::ConstantIntOp::create(builder, loc, 1, 64);
  mlir::Value stamp = RuntimeBundleLowerer::memrefFromBoxWords(
      builder, loc, tableAddress, one,
      mlir::MemRefType::get({mlir::ShapedType::kDynamic}, i64));
  mlir::Value zero = mlir::arith::ConstantIndexOp::create(builder, loc, 0)
                         .getResult();
  mlir::Value current =
      mlir::memref::LoadOp::create(builder, loc, stamp, zero).getResult();
  // Branch-free: the stale marker is -1, which `__ly_dict_table_sync` reads as
  // "below every dense index" and answers with a full rebuild.
  mlir::Value shrinking = mlir::arith::CmpIOp::create(
      builder, loc, mlir::arith::CmpIPredicate::slt, length, previous);
  mlir::Value stale = mlir::arith::ConstantIntOp::create(builder, loc, -1, 64);
  mlir::Value next = mlir::arith::SelectOp::create(builder, loc, shrinking,
                                                   stale, current);
  mlir::memref::StoreOp::create(builder, loc, next, stamp, zero);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::storeContainerLength(
    mlir::Operation *op, const RuntimeBundle &container, mlir::Value length,
    llvm::StringRef label) {
  mlir::FailureOr<std::pair<mlir::Value, mlir::Value>> slot =
      RuntimeBundleLowerer::containerMetaSlot(
          op, container, container_abi::kMetaLengthSlot, label);
  if (mlir::failed(slot))
    return mlir::failure();
  if (mlir::failed(RuntimeBundleLowerer::invalidateMappingTableOnShrink(
          op, container, length)))
    return mlir::failure();
  mlir::memref::StoreOp::create(builder, op->getLoc(), length, slot->first,
                                slot->second);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::adjustContainerLength(
    mlir::Operation *op, const RuntimeBundle &container, std::int64_t delta,
    llvm::StringRef label) {
  mlir::FailureOr<std::pair<mlir::Value, mlir::Value>> slot =
      RuntimeBundleLowerer::containerMetaSlot(
          op, container, container_abi::kMetaLengthSlot, label);
  if (mlir::failed(slot))
    return mlir::failure();
  mlir::Location loc = op->getLoc();
  mlir::Value current = mlir::memref::LoadOp::create(builder, loc, slot->first,
                                                     slot->second)
                            .getResult();
  mlir::Value one = mlir::arith::ConstantIntOp::create(builder, loc, 1, 64);
  mlir::Value next =
      delta >= 0
          ? mlir::arith::AddIOp::create(builder, loc, current, one).getResult()
          : mlir::arith::SubIOp::create(builder, loc, current, one).getResult();
  if (mlir::failed(RuntimeBundleLowerer::invalidateMappingTableOnShrink(
          op, container, next)))
    return mlir::failure();
  mlir::memref::StoreOp::create(builder, loc, next, slot->first, slot->second);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::touchContainerEvidenceUse(
    mlir::Operation *op, const RuntimeBundle &container,
    llvm::StringRef label) {
  return mlir::failed(
             RuntimeBundleLowerer::loadContainerLength(op, container, label))
             ? mlir::failure()
             : mlir::success();
}

mlir::FailureOr<mlir::Value> RuntimeBundleLowerer::containerInteriorView(
    mlir::Operation *op, const RuntimeBundle &container,
    ContainerInterior which, llvm::StringRef label) {
  llvm::ArrayRef<mlir::Value> values = container.physicalValues();
  mlir::Location loc = op->getLoc();
  mlir::Type i64 = builder.getI64Type();

  if (!RuntimeBundleLowerer::containerIsHandleFronted(container)) {
    // Lane-carrying contract: the view IS the lane. Lane order is the
    // manifest shape order (header, meta, primary[, secondary, present]).
    std::size_t lane = 0;
    switch (which) {
    case ContainerInterior::Primary:
      lane = 2;
      break;
    case ContainerInterior::Secondary:
      lane = 3;
      break;
    case ContainerInterior::Present:
      lane = 4;
      break;
    }
    if (values.size() <= lane)
      return op->emitError()
             << label << " container has no physical interior lane " << lane
             << " (contract " << container.contractName() << " expands to "
             << values.size() << " physical values)";
    if (!ownership::isRankOneI64MemRef(values[lane].getType()))
      return op->emitError() << label << " container interior lane " << lane
                             << " has invalid type " << values[lane].getType();
    return values[lane];
  }

  std::int64_t baseWord = container_abi::kPrimaryArrayWord;
  bool wordPerSlot = false;
  switch (which) {
  case ContainerInterior::Primary:
    baseWord = container_abi::kPrimaryArrayWord;
    break;
  case ContainerInterior::Secondary:
    baseWord = container_abi::kSecondaryArrayWord;
    break;
  case ContainerInterior::Present:
    baseWord = container_abi::kPresentArrayWord;
    wordPerSlot = true;
    break;
  }

  // The `memref.load` from the handle followed by the descriptor assembly in
  // memrefFromBoxWords is the exact chain `collectBoxWordDerivedViews`
  // (common/Ownership.cpp) walks, so the resulting view pins the entity until
  // the view's last use. Deriving the base any other way (pointer arithmetic
  // on the handle, a helper call without `ly.runtime.interior_word`) would
  // leave the walk with nothing to follow and under-pin the handle.
  mlir::Value handle = values.front();
  mlir::Value baseSlot =
      mlir::arith::ConstantIndexOp::create(builder, loc, baseWord).getResult();
  mlir::Value arrayAddress =
      mlir::memref::LoadOp::create(builder, loc, handle, baseSlot).getResult();
  // The descriptor's size only feeds memref.dim/bounds queries -- every reader
  // indexes with a slot the container's own length or capacity produced. It is
  // still filled in honestly so a bounds query cannot read a lie.
  mlir::Value capacitySlot =
      mlir::arith::ConstantIndexOp::create(builder, loc,
                                           container_abi::kCapacityWord)
          .getResult();
  mlir::Value capacity =
      mlir::memref::LoadOp::create(builder, loc, handle, capacitySlot)
          .getResult();
  mlir::Value size =
      wordPerSlot
          ? capacity
          : mlir::arith::MulIOp::create(
                builder, loc, capacity,
                mlir::arith::ConstantIntOp::create(builder, loc,
                                                   box_abi::kWordsPerBox, 64))
                .getResult();
  return RuntimeBundleLowerer::memrefFromBoxWords(
      builder, loc, arrayAddress, size,
      mlir::MemRefType::get({mlir::ShapedType::kDynamic}, i64));
}

} // namespace py::lowering
