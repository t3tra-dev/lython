#include "Runtime/Core/Lowerer.h"

#include "ExceptionTaxonomy.h"
#include "Runtime/ABI/BoxLayout.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"

namespace py::lowering {

namespace {

constexpr unsigned kPrimitiveFieldSlotBase =
    static_cast<unsigned>(box_abi::kPointerWordBase);
constexpr unsigned kPrimitiveFieldSlotLimit =
    static_cast<unsigned>(box_abi::kWordsPerBox);

void appendValueSlice(mlir::ValueRange values, unsigned begin, unsigned count,
                      llvm::SmallVectorImpl<mlir::Value> &out) {
  for (unsigned index = 0; index < count; ++index)
    out.push_back(values[begin + index]);
}

bool isMethodDescriptorKind(py::AttrGetOp op) {
  auto kind = op->getAttrOfType<mlir::StringAttr>("ly.attr.kind");
  if (!kind)
    return false;
  llvm::StringRef value = kind.getValue();
  return value == "instance" || value == "static" || value == "class" ||
         value == "classmethod";
}

// int and bool are the contracts whose whole value fits in one i64, so their
// field storage IS an instance-header word — a heap slot every frame holding
// the instance reaches through the same pointer. Words [0, 4) are the header's
// own (refcount, class id, value count), so field i takes word 4 + i and a
// class with more than kPrimitiveFieldSlotLimit - kPrimitiveFieldSlotBase of
// them falls back to the contract's own lanes.
std::optional<unsigned> primitiveFieldSlot(mlir::Type fieldType,
                                           unsigned fieldIndex) {
  std::string contract = runtimeContractName(fieldType);
  if (contract != "builtins.int" && contract != "builtins.bool")
    return std::nullopt;
  unsigned slot = kPrimitiveFieldSlotBase + fieldIndex;
  if (slot >= kPrimitiveFieldSlotLimit)
    return std::nullopt;
  return slot;
}

bool isBoolFieldType(mlir::Type fieldType) {
  return runtimeContractName(fieldType) == "builtins.bool";
}

// Does this field read feed an IN-PLACE MUTATION of the value it loaded?
//
// Selects how the read is spelled, not whether it is safe -- both spellings are
// sound now that the transfer manufactures its own reference
// (`promoteInteriorViewForTransfer`). A read that is about to be mutated stays
// pinned to the slot because a reallocating mutation renames the lane tuple, and
// a resource whose tuple is renamed cannot also be released under its old names
// on a loop back edge. See the branch in `lowerAttrGet` for the measurement.
//
// Syntactic, and narrow on purpose: it asks what the read is FOR, at the one
// point where both the load and its uses are visible. Do not widen it to decide
// anything else -- the split exists only because the representation has lanes,
// and it has no successor once the payload is behind the handle.
bool fieldReadFeedsInPlaceMutation(mlir::Value read) {
  for (mlir::Operation *user : read.getUsers()) {
    if (auto setItem = mlir::dyn_cast<py::SetItemOp>(user))
      if (setItem.getContainer() == read)
        return true;
    if (auto delItem = mlir::dyn_cast<py::DelItemOp>(user))
      if (delItem.getContainer() == read)
        return true;
    // A method call reaches its receiver through an attr.get, and whether the
    // method mutates is not known here: treat any method lookup on the loaded
    // value as a possible mutation. A nested FIELD read (`t.m.i`) is not one.
    if (auto attrGet = mlir::dyn_cast<py::AttrGetOp>(user)) {
      auto kind = attrGet->getAttrOfType<mlir::StringAttr>("ly.attr.kind");
      if (attrGet.getObject() == read &&
          !(kind && kind.getValue() == "field"))
        return true;
    }
  }
  return false;
}

std::optional<mlir::Attribute> classStaticValue(py::ClassOp classOp,
                                                llvm::StringRef name) {
  auto names =
      classOp->getAttrOfType<mlir::ArrayAttr>("class_static_attr_names");
  auto values =
      classOp->getAttrOfType<mlir::ArrayAttr>("class_static_attr_values");
  if (!names || !values || names.size() != values.size())
    return std::nullopt;
  for (auto [index, attr] : llvm::enumerate(names)) {
    auto stringAttr = mlir::dyn_cast<mlir::StringAttr>(attr);
    if (stringAttr && stringAttr.getValue() == name)
      return values[index];
  }
  return std::nullopt;
}

} // namespace

std::optional<unsigned>
RuntimeBundleLowerer::classFieldIndex(py::ClassOp classOp,
                                      llvm::StringRef name) const {
  auto fieldNames = classOp->getAttrOfType<mlir::ArrayAttr>("field_names");
  if (!fieldNames)
    return std::nullopt;
  for (auto [index, attr] : llvm::enumerate(fieldNames)) {
    auto stringAttr = mlir::dyn_cast<mlir::StringAttr>(attr);
    if (stringAttr && stringAttr.getValue() == name)
      return static_cast<unsigned>(index);
  }
  return std::nullopt;
}

mlir::FailureOr<unsigned> RuntimeBundleLowerer::classFieldValueOffset(
    mlir::Operation *op, py::ClassOp classOp, unsigned fieldIndex,
    llvm::StringRef purpose) const {
  const RuntimeValueShape *objectShape = manifest.valueShape("builtins.object");
  if (!objectShape)
    return op->emitError()
           << "runtime manifest has no builtins.object ABI shape for "
           << purpose;
  llvm::SmallVector<mlir::Type, 8> fieldTypes =
      RuntimeBundleLowerer::classFieldContractTypes(classOp);
  if (fieldIndex >= fieldTypes.size())
    return op->emitError() << purpose << " field index " << fieldIndex
                           << " is outside " << classOp.getSymName();

  unsigned offset = static_cast<unsigned>(objectShape->valueTypes.size());
  for (unsigned index = 0; index < fieldIndex; ++index) {
    mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> valueTypes =
        RuntimeBundleLowerer::classFieldStorageValueTypes(op, fieldTypes[index],
                                                          index, purpose);
    if (mlir::failed(valueTypes))
      return mlir::failure();
    offset += static_cast<unsigned>(valueTypes->size());
  }
  return offset;
}

bool RuntimeBundleLowerer::classFieldStoredBoxed(
    mlir::Type fieldContract) const {
  // runtimeShapeContractName returns by value; a StringRef binding would
  // dangle past this declaration statement.
  std::string contractName = runtimeShapeContractName(fieldContract);
  // A union-typed field has no single contract: its tag plus every member's
  // lanes stay inline, because the box words hold ONE payload handle and a
  // union is not one object.
  if (contractName.empty())
    return false;
  // Zero-lane contracts have nothing that could go stale. Adding a box would
  // be an allocation whose only content is the absence of a value.
  if (contractName == "types.NoneType")
    return false;
  // int/bool are the two contracts whose value is stored IN the instance
  // header (primitiveFieldSlot), which is already a stable heap slot; their
  // contract lanes are a placeholder the store never reads. Boxing them would
  // add a second storage for the same field and force the load to choose.
  if (contractName == "builtins.int" || contractName == "builtins.bool")
    return false;
  return true;
}

// Swaps the payload held by an existing box16 slot without re-rooting the
// slot itself: retain the new payload, release whatever the box owned (a
// no-op while the owned flag is zero), overwrite the handle words. The box
// pointer must stay stable for the instance's lifetime — external snapshots
// (dict/set key boxes, runtime-method self boxes) reach the field only
// through it. Returns the stored payload bundle (owned by the box).
mlir::FailureOr<RuntimeBundle>
RuntimeBundleLowerer::storeBoxedFieldPayloadInPlace(mlir::Operation *op,
                                                    mlir::Value box,
                                                    const RuntimeBundle &value,
                                                    llvm::StringRef slotName) {
  if (!mlir::isa<mlir::MemRefType>(box.getType()))
    return op->emitError() << slotName << " box-fronted slot is not a box16 "
                           << "lane, got " << box.getType();
  builder.setInsertionPoint(op);
  mlir::Location loc = op->getLoc();
  mlir::FailureOr<RuntimeBundle> payload =
      RuntimeBundleLowerer::materializePayloadObjectBundle(op, value);
  if (mlir::failed(payload))
    return mlir::failure();
  // The slot holds a canonical payload handle, so the value needs a concrete
  // shape to describe. `objectPayloadHandleWords` refuses an erased `object`
  // below, but its message is written for a container element; say it in terms
  // of the field, which is what the author wrote.
  //
  // This REJECTS one shape that used to run. Before the store moved into the
  // slot, an `object`-annotated field took the handle-store path, which wrote
  // the erased handle into the instance's lane; that was silently wrong across a
  // function boundary for a str/float/list/dict payload and happened to work for
  // an int. Keeping the one accidental case would mean keeping a path whose
  // correctness depends on the payload's width, and the project does not
  // implement runtime operations on `object` at all, so the whole shape is
  // refused at the earliest boundary instead.
  if (const RuntimeBundle *concrete =
          RuntimeBundleLowerer::concreteObjectForOwnership(*payload))
    if (concrete->contractName() == "builtins.object")
      return op->emitError()
             << "a type-erased `object` value cannot be stored in field '"
             << slotName
             << "'; annotate the field with the concrete type it holds";
  if (mlir::failed(RuntimeBundleLowerer::retainAggregateSlot(op, *payload,
                                                             slotName)))
    return mlir::failure();
  mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>> words =
      RuntimeBundleLowerer::objectPayloadHandleWords(op, *payload,
                                                     /*ownsPayload=*/true);
  if (mlir::failed(words))
    return mlir::failure();

  auto releaseBoxed = module.lookupSymbol<mlir::func::FuncOp>(
      "LyObject_ReleaseBoxedPayloadRaw");
  if (!releaseBoxed)
    return op->emitError()
           << "runtime support has no LyObject_ReleaseBoxedPayloadRaw";
  mlir::Value releaseOperand = box;
  mlir::Type expectedBox = releaseBoxed.getFunctionType().getInput(0);
  if (releaseOperand.getType() != expectedBox)
    releaseOperand =
        mlir::memref::CastOp::create(builder, loc, expectedBox, releaseOperand)
            .getResult();
  // ⭐ ALWAYS release what the box held. A store is a replace: retain the new
  // reference (above), give up the old one.
  //
  // This release used to be guarded by a runtime comparison of the payload header
  // pointer, so a SELF-store (`ks = self._kids; ks.append(v); self._kids = ks`)
  // took the retain and skipped the release. That is +1 per self-store, and it
  // leaked the whole payload: measured 2 allocations / 8264 B for a one-element
  // list -- 8192 of it the list payload at `minimum_capacity` -- and it is where
  // `class_field_list_mutation` spent 16736 B.
  //
  // The guard's reasoning was that releasing here "plus the caller's release of
  // the owned result drops it to zero while the program still reads through it".
  // Recount it for the self-store: the box holds one, the read that produced the
  // local retained a second, this retain makes three, this release makes two, and
  // the frame's release of the local makes one -- which is what the box holds. The
  // arithmetic works BECAUSE the retain above is unconditional; the guard made
  // only one half of the pair conditional, which is the shape that cannot balance.
  //
  // Why NOT make the retain conditional instead, to match: tried, and it is wrong
  // in two ways the suite names. `retainAggregateSlot` is not only a refcount
  // bump -- it is the accounting that parks the incoming token in the container --
  // so skipping it leaves the token neither released nor parked, and
  // `try_loop_carried_entity_rebind` and `try_inner_handler_rebind_loop_carried`
  // both fail with "reaches function exit without release, transfer, or owned
  // return". `cross_container_box_fronted_fields` fails harder, with a
  // use-after-free. Those three are the red-check for this repair.
  //
  // Word 2 is zero on the constructor's placeholder, so the very first store
  // releases a null payload -- a no-op inside `LyObject_ReleaseBoxedPayloadRaw`,
  // which is why an unconditional release is safe on a fresh instance.
  mlir::func::CallOp::create(builder, loc, releaseBoxed,
                             mlir::ValueRange{releaseOperand});
  for (auto [wordIndex, word] : llvm::enumerate(*words)) {
    mlir::Value slot = mlir::arith::ConstantIndexOp::create(
        builder, loc, static_cast<std::int64_t>(wordIndex));
    mlir::memref::StoreOp::create(builder, loc, word, box, slot);
  }
  RuntimeBundle stored = *payload;
  stored.setObjectLogicalOwnership(/*ownsObject=*/true);
  return stored;
}

mlir::LogicalResult RuntimeBundleLowerer::storePrimitiveFieldSlot(
    mlir::Operation *op, const RuntimeBundle &object,
    const RuntimeBundle &value, mlir::Type fieldType, unsigned fieldIndex,
    llvm::StringRef fieldName) {
  std::optional<unsigned> slot = primitiveFieldSlot(fieldType, fieldIndex);
  if (!slot)
    return op->emitError() << "field '" << fieldName << "' of " << fieldType
                           << " has no instance header word";
  builder.setInsertionPoint(op);
  mlir::Location loc = op->getLoc();
  mlir::Value word;
  if (isBoolFieldType(fieldType)) {
    if (value.physicalValues().size() != 1 ||
        !value.physicalValues().front().getType().isInteger(1))
      return op->emitError() << "attribute value " << value.contractName()
                             << " has no i1 lane for bool field '" << fieldName
                             << "'";
    word = mlir::arith::ExtUIOp::create(builder, loc, builder.getI64Type(),
                                        value.physicalValues().front())
               .getResult();
  } else if (primitiveI64LaneKnownValid(value.primitiveI64)) {
    word = value.primitiveI64->value;
  } else if (value.physicalValues().empty() && value.primitiveI64 &&
             value.primitiveI64->value) {
    // No boxed payload to fall back to (primitive-i64 clone lanes carry only
    // the (value, valid) pair): the lane is the sole carrier.
    word = value.primitiveI64->value;
  } else {
    std::optional<RuntimeSymbol> unbox =
        manifest.primitive(value.contractName(), "unbox.i64");
    if (!unbox)
      return op->emitError() << "attribute value " << value.contractName()
                             << " has no unbox.i64 primitive for field '"
                             << fieldName << "'";
    llvm::SmallVector<const RuntimeBundle *, 1> unboxSources{&value};
    llvm::SmallVector<mlir::Value, 4> unboxOperands;
    if (mlir::failed(buildRuntimeCallOperands(op, *unbox, unboxSources,
                                              unboxOperands,
                                              /*allowUnusedSources=*/false)))
      return mlir::failure();
    mlir::func::CallOp unboxCall =
        RuntimeBundleLowerer::createRuntimeCall(loc, *unbox, unboxOperands);
    if (unboxCall.getNumResults() != 1 ||
        !unboxCall.getResult(0).getType().isInteger(64))
      return unbox->function.emitError()
             << "unbox.i64 primitive must return one i64";
    word = unboxCall.getResult(0);
  }

  mlir::FailureOr<mlir::Value> header =
      RuntimeBundleLowerer::objectPhysicalHeader(op, object.objectValue);
  if (mlir::failed(header))
    return mlir::failure();
  mlir::Value slotIndex =
      mlir::arith::ConstantIndexOp::create(builder, loc, *slot).getResult();
  mlir::memref::StoreOp::create(builder, loc, word, *header, slotIndex);
  return mlir::success();
}

// Re-describes the payload a box already owns, for an in-place mutation that
// REALLOCATED its arrays (list.append, dict insert). Only the descriptor words
// move: the payload is the same logical object, the box holds the same single
// reference to it, and the box pointer never changed — so no retain, no
// release, and above all no re-root of the instance's lanes. Releasing the
// box's old payload here would hand the deallocator storage the mutation
// primitive already freed.
mlir::LogicalResult RuntimeBundleLowerer::updateBoxedFieldPayloadWords(
    mlir::Operation *op, mlir::Value box, const RuntimeBundle &payload,
    llvm::StringRef slotName) {
  if (!mlir::isa<mlir::MemRefType>(box.getType()))
    return op->emitError() << slotName << " box-fronted slot is not a box16 "
                           << "lane, got " << box.getType();
  const RuntimeBundle *concrete =
      RuntimeBundleLowerer::concreteObjectForOwnership(payload);
  if (!concrete || concrete->kind != RuntimeBundle::Kind::Object)
    return op->emitError() << slotName << " write-back needs an object bundle";
  if (concrete->physicalValues().empty())
    return mlir::success();
  builder.setInsertionPoint(op);
  mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>> words =
      RuntimeBundleLowerer::objectPayloadHandleWords(op, *concrete,
                                                    /*ownsPayload=*/true);
  if (mlir::failed(words))
    return mlir::failure();
  mlir::Location loc = op->getLoc();
  // Words 0 and 14 (refcount, owned flag) are the box's own bookkeeping and
  // must survive: rewriting them would reset a reference count the program is
  // still using. Everything from word 1 up describes the payload.
  for (unsigned index = 1; index < words->size(); ++index) {
    if (index == static_cast<unsigned>(box_abi::kOwnedFlagWord))
      continue;
    mlir::Value slot = mlir::arith::ConstantIndexOp::create(
        builder, loc, static_cast<std::int64_t>(index));
    mlir::memref::StoreOp::create(builder, loc, (*words)[index], box, slot);
  }
  return mlir::success();
}

mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>>
RuntimeBundleLowerer::classFieldStorageValueTypes(
    mlir::Operation *op, mlir::Type fieldContract, unsigned fieldIndex,
    llvm::StringRef purpose) const {
  // A header-word field occupies NO lane: the word IS the storage. It used to
  // carry the contract's full expansion as a placeholder nothing ever read --
  // three memrefs allocated and released per int field per instance -- which
  // also made a class of four int fields expand to thirteen handles and so too
  // wide to sit in another class's box.
  if (primitiveFieldSlot(fieldContract, fieldIndex))
    return llvm::SmallVector<mlir::Type, 8>{};
  if (classFieldStoredBoxed(fieldContract)) {
    const RuntimeValueShape *objectShape =
        manifest.valueShape("builtins.object");
    if (!objectShape)
      return op->emitError()
             << "runtime manifest has no builtins.object ABI shape for "
             << purpose;
    return llvm::SmallVector<mlir::Type, 8>(objectShape->valueTypes.begin(),
                                            objectShape->valueTypes.end());
  }
  return RuntimeBundleLowerer::runtimeValueTypesFor(op, fieldContract, purpose);
}

mlir::LogicalResult
RuntimeBundleLowerer::writeBackFieldAlias(mlir::Operation *op,
                                          const RuntimeBundle &updatedField) {
  if (!updatedField.fieldAliasOwner || updatedField.fieldAliasName.empty())
    return mlir::success();
  auto owner = valueBundles.find(updatedField.fieldAliasOwner);
  if (owner == valueBundles.end())
    return mlir::success();

  RuntimeBundle ownerBundle = owner->second;
  RuntimeBundle storedField = updatedField.withObjectOwnership(
      ownership::logicalOwnershipKind(updatedField.objectValue.contract,
                                      /*ownsObject=*/true));
  ownerBundle.fieldBundles[updatedField.fieldAliasName] =
      std::make_shared<RuntimeBundle>(storedField);
  if (ownerBundle.kind != RuntimeBundle::Kind::Object) {
    valueBundles[updatedField.fieldAliasOwner] = std::move(ownerBundle);
    return mlir::success();
  }

  py::ClassOp classOp =
      RuntimeBundleLowerer::classForContract(ownerBundle.objectValue.contract);
  if (!classOp)
    return op->emitError() << "field alias owner has no class schema";
  std::optional<unsigned> fieldIndex = RuntimeBundleLowerer::classFieldIndex(
      classOp, updatedField.fieldAliasName);
  if (!fieldIndex)
    return op->emitError() << "class " << classOp.getSymName()
                           << " has no field '" << updatedField.fieldAliasName
                           << "'";
  llvm::SmallVector<mlir::Type, 8> fieldTypes =
      RuntimeBundleLowerer::classFieldContractTypes(classOp);
  if (*fieldIndex >= fieldTypes.size())
    return op->emitError() << "class field metadata is malformed for "
                           << classOp.getSymName();
  mlir::FailureOr<unsigned> offset =
      RuntimeBundleLowerer::classFieldValueOffset(op, classOp, *fieldIndex,
                                                  "field alias writeback ABI");
  if (mlir::failed(offset))
    return mlir::failure();

  // A box-fronted field's write-back is a WORD UPDATE, not a re-root. The
  // mutation reallocated the payload's arrays, so the box's descriptor words
  // have to name the new ones; the box pointer, the box's reference, and the
  // instance's lanes are all unchanged. Nothing here needs the owned-local
  // marker republished — which is the whole reason the marker existed on this
  // path, and why re-rooting a field alias inside a branch used to produce a
  // value that did not dominate the later read.
  if (RuntimeBundleLowerer::classFieldStoredBoxed(fieldTypes[*fieldIndex])) {
    if (*offset >= ownerBundle.objectValue.values.size())
      return op->emitError() << "field alias update exceeds owner payload";
    if (mlir::failed(RuntimeBundleLowerer::updateBoxedFieldPayloadWords(
            op, ownerBundle.objectValue.values[*offset], updatedField,
            updatedField.fieldAliasName)))
      return mlir::failure();
    RuntimeBundle boxedOwnerView = ownerBundle;
    valueBundles[updatedField.fieldAliasOwner] = std::move(ownerBundle);
    // The box POINTER does not move, but the owner's cached bundle for this
    // field just changed, and an owner that is ITSELF a field of something
    // else has that stale copy one level up. `t.mid.leaves.append` twice read
    // `t`'s cached `mid` the second time and grew a list described by the
    // first append.
    if (boxedOwnerView.fieldAliasOwner &&
        !boxedOwnerView.fieldAliasName.empty() &&
        boxedOwnerView.fieldAliasOwner != updatedField.fieldAliasOwner)
      return RuntimeBundleLowerer::writeBackFieldAlias(op, boxedOwnerView);
    return mlir::success();
  }

  mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> fieldValueTypes =
      RuntimeBundleLowerer::runtimeValueTypesFor(op, fieldTypes[*fieldIndex],
                                                 "field alias writeback ABI");
  if (mlir::failed(fieldValueTypes))
    return mlir::failure();
  if (fieldValueTypes->size() != updatedField.physicalValues().size())
    return op->emitError() << "field alias update has "
                           << updatedField.physicalValues().size()
                           << " physical values, but field expects "
                           << fieldValueTypes->size();
  if (*offset + fieldValueTypes->size() > ownerBundle.objectValue.values.size())
    return op->emitError() << "field alias update exceeds owner payload";
  for (auto [index, replacement] :
       llvm::enumerate(updatedField.physicalValues()))
    ownerBundle.objectValue.values[*offset + index] = replacement;

  // The owner's OWN lanes just changed, so if the owner is itself an interior
  // view of something (`a.b.c`, where `b` is a lane-stored field), the storage
  // above it still names the pre-mutation lanes. Recurse before publishing, so
  // the chain is repaired root-first regardless of depth. Only the lane branch
  // needs this: a box-fronted field's box POINTER is what its owner's slot
  // holds, and that pointer never moves.
  RuntimeBundle ownerView = ownerBundle;
  valueBundles[updatedField.fieldAliasOwner] = std::move(ownerBundle);
  if (ownerView.fieldAliasOwner && !ownerView.fieldAliasName.empty() &&
      ownerView.fieldAliasOwner != updatedField.fieldAliasOwner)
    return RuntimeBundleLowerer::writeBackFieldAlias(op, ownerView);
  return mlir::success();
}

// You cannot transfer a borrow. A mutation primitive declared
// `transfer_args = [0]` consumes a reference to its receiver; when the receiver
// is an interior view (a field slot's container, read without a reference of its
// own) the reference it consumes is the SLOT's, and the entity is freed while
// the field still names it -- the one memory-safety defect left after 4a, and
// the reason `dict read` was unfilled in both boundaries of the 12-cell grid.
//
// So manufacture one here, adjacent to the transfer. Deliberately an
// aggregate-slot retain rather than an owned-local resource: a tracked resource
// whose tuple the transfer then renames is exactly what the affine verifier
// cannot follow across a loop back edge, and this reference has no lifetime of
// its own -- it exists to be consumed by the very next call. The call's
// `owned_results` group carries the obligation onward from there.
mlir::LogicalResult RuntimeBundleLowerer::promoteInteriorViewForTransfer(
    mlir::Operation *op, const RuntimeBundle &receiver, llvm::StringRef slotName,
    mlir::func::FuncOp mutation) {
  // Ask the mutation whether it actually takes a reference. A contract whose
  // interior state lives behind the handle has nothing to rename, so its
  // mutations are void and non-transfer -- and then there is no reference to
  // manufacture, because none is consumed. Asking the callee rather than
  // naming the converted contracts means this whole workaround retires one
  // contract at a time, on its own, as each conversion lands.
  if (mutation && !ownership::functionConsumesOperandAt(mutation, 0))
    return mlir::success();
  if (receiver.kind != RuntimeBundle::Kind::Object ||
      receiver.physicalValues().empty())
    return mlir::success();
  // An owned receiver already has a token the transfer can take, and doubling it
  // would leak: the result group's single release cannot answer for two.
  if (receiver.objectValue.ownership == ownership::OwnershipKind::Own)
    return mlir::success();
  // The bundle's own ownership field is not the whole answer. A read that DID
  // take a reference is spelled as a borrow bundle over values rooted by an
  // owned-local marker (`retainEvidenceElement`), because the reference belongs
  // to the marker's resource rather than to the bundle -- so the field reads
  // Borrow while a token exists. Retaining again there leaked the whole payload
  // once per call (8.1 kB/iteration on `leak_mutate_call_append`), because the
  // one release the result group plans cannot answer for two retains. Ask the
  // producer instead of the label.
  if (mlir::Operation *root = receiver.physicalValues().front().getDefiningOp())
    if (root->hasAttr(ownership::kOwnedLocalObjectAttr))
      return mlir::success();
  return RuntimeBundleLowerer::retainAggregateSlot(op, receiver, slotName);
}

// Binds the re-description a "consume the container and hand back another one"
// mutation primitive returns. The entity's identity did not change, so the
// interior storage that names it has to be told the new descriptor before
// anything else derives one from it -- that is `Interior`'s obligation in the
// lane era, where the payload still travels as SSA values alongside the handle.
//
// One function on purpose. The obligation used to be spelled out at each
// mutation site, and of the five sites that hand back a re-description, three
// did not discharge it (runtime dict setitem, runtime list append, set add):
// the local was rebound and the field slot kept naming storage the primitive had
// already reallocated. `writeBackFieldAlias` reached exactly one combination --
// an evidence-backed list one level below a field -- and every other
// (container kind x depth x acquisition path) silently went without.
// ⭐ A STORE INTO A SLOT IS NOT THE SOURCE'S DEATH when the source is still
// read afterwards. `self.xs = xs` retains for the slot and then releases the
// value's own token, which is right for a temporary (`self.xs = [1, 2, 3]`):
// the two cancel and the slot inherits the one reference. When the source is a
// LOCAL the caller keeps reading, the release is premature, and every later
// read through that name is a use-after-free:
//
//     seed = [3, 1, 2]
//     b = Bag(seed)          # __init__ does self.xs = xs
//     i = 0
//     while i < 3:
//         print(len(seed))   # printed 0 0 0; CPython prints 3 3 3
//         i += 1
//
// `for v in seed` over the same list SIGSEGV'd, `[v for v in seed]` answered
// `[]`, and `max(seed)` raised "max() iterable argument is empty" -- all three
// are len-then-index. It reads as intact whenever nothing allocates between the
// release and the read, which is why `print(len(seed))` at module scope, or a
// second read through the field in the same statement, made it look fine.
//
// So ask whether the value has a use this store DOMINATES. Dominance rather
// than block membership: a read in a loop body or a branch arm is a later use
// even though it is not in this block, and that is the shape that failed.
//
// ⛔ Why NOT keep the release and re-root the local's binding on the slot's
// reference: the slot's reference dies with the OBJECT, and the local outliving
// the object is the ordinary case (`b = Bag(seed)` inside a function, `seed`
// returned). Skipping the release leaves the frame group with its own token and
// the exit release the insertion pass already plans for it.
//
// ⛔ The PY-level operand, not the bundle's physical values. The walk lowers in
// program order, so a later read is still an unlowered `py.len` / `py.getitem`
// over the same py value -- the physical handle has no uses past this point yet,
// and asking it answered "nothing outlives this" for every shape above.
bool RuntimeBundleLowerer::storedSourceOutlivesStore(mlir::Operation *op,
                                                     mlir::Value source) {
  if (!source)
    return false;
  // ⭐ A LOOP-CARRIED VALUE OUTLIVES EVERY STORE INSIDE OR AFTER THE LOOP, and
  // the dominance walk below cannot see it: its other uses are the loop's own,
  // which the store does not dominate, so the store read "nobody else needs
  // this" and took the token. The loop then released the same reference again:
  //
  //     class B:
  //         def __init__(self, n: int) -> None:
  //             raw = ""
  //             for i in range(n):
  //                 raw = raw + "x"
  //             self.raw: str = raw
  //
  //     # owned resource from @LyUnicode_FromBytes result 0 is released or
  //     # transferred more than once on one CFG path
  //
  // ⛔ A BLOCK ARGUMENT IS THE SIGNAL, not the loop: the same is true of an
  // if/else merge, whose two arms each hand over a reference the frame still
  // owns. Both are values the store did not produce, and the store may only
  // move a token it can see the whole life of.
  if (mlir::isa<mlir::BlockArgument>(source))
    return true;
  mlir::Operation *function = op->getParentOfType<mlir::func::FuncOp>();
  if (!function)
    return false;
  mlir::DominanceInfo dominance(function);
  for (mlir::OpOperand &use : source.getUses()) {
    mlir::Operation *user = use.getOwner();
    if (user == op)
      continue;
    if (dominance.properlyDominates(op, user))
      return true;
  }
  return false;
}

mlir::LogicalResult RuntimeBundleLowerer::rebindMutatedContainer(
    mlir::Operation *op, const RuntimeBundle &receiver, mlir::ValueRange values,
    RuntimeBundle &rebound) {
  // No results means the mutation published no re-description: it wrote through
  // the handle the receiver already names. There is nothing to rebind and
  // nothing to write back, and saying so here is what makes the callers of a
  // converted contract collapse to a plain call.
  if (values.empty()) {
    rebound = receiver;
    return mlir::success();
  }
  if (mlir::failed(RuntimeBundleLowerer::makeObjectBundle(
          op, receiver.objectValue.contract, values, rebound)))
    return mlir::failure();
  rebound.fieldAliasOwner = receiver.fieldAliasOwner;
  rebound.fieldAliasName = receiver.fieldAliasName;
  return RuntimeBundleLowerer::writeBackFieldAlias(op, rebound);
}

mlir::LogicalResult RuntimeBundleLowerer::lowerAttrGet(py::AttrGetOp op) {
  const RuntimeBundle *object = RuntimeBundleLowerer::bundleFor(op.getObject());
  if (!object)
    return op.emitError() << "attr.get object has no lowered runtime bundle";
  if (object->kind == RuntimeBundle::Kind::TypeObject) {
    if (isMethodDescriptorKind(op) &&
        RuntimeBundleLowerer::classDefinesMethod(object->instanceContract,
                                                 op.getName())) {
      RuntimeBundle result =
          RuntimeBundle::object(op.getResult().getType(), mlir::ValueRange{});
      result.boundMethodReceiver = std::make_shared<RuntimeBundle>(*object);
      result.boundMethodName = op.getName().str();
      valueBundles[op.getResult()] = std::move(result);
      erase.push_back(op);
      return mlir::success();
    }
    if (py::ClassOp classOp =
            RuntimeBundleLowerer::classForContract(object->instanceContract)) {
      if (std::optional<mlir::Attribute> staticValue =
              classStaticValue(classOp, op.getName())) {
        auto dict = mlir::dyn_cast<mlir::DictionaryAttr>(*staticValue);
        if (!dict)
          return op.emitError() << "static class attribute metadata for '"
                                << op.getName() << "' is malformed";
        auto kind = dict.getAs<mlir::StringAttr>("kind");
        if (!kind)
          return op.emitError() << "static class attribute '" << op.getName()
                                << "' has no metadata kind";
        llvm::StringRef spelling = kind.getValue();
        llvm::StringRef defaultKind = spelling;
        if (spelling == "constant.none")
          defaultKind = "none";
        else if (spelling == "constant.bool")
          defaultKind = "bool";
        else if (spelling == "constant.int")
          defaultKind = "int";
        else if (spelling == "constant.float")
          defaultKind = "float";
        else if (spelling == "constant.str")
          defaultKind = "str";
        else
          return op.emitError()
                 << "unsupported static class attribute expression for '"
                 << op.getName() << "'";

        llvm::SmallVector<mlir::NamedAttribute, 4> attrs;
        attrs.push_back(
            builder.getNamedAttr("kind", builder.getStringAttr(defaultKind)));
        if (mlir::Attribute value = dict.get("value"))
          attrs.push_back(builder.getNamedAttr("value", value));
        mlir::DictionaryAttr defaultValue = builder.getDictionaryAttr(attrs);

        builder.setInsertionPoint(op);
        RuntimeBundle result;
        if (mlir::failed(RuntimeBundleLowerer::materializeDefaultValue(
                op, op.getResult().getType(), defaultValue, result)))
          return mlir::failure();
        valueBundles[op.getResult()] = std::move(result);
        erase.push_back(op);
        return mlir::success();
      }
    }
    mlir::LogicalResult descriptorResult =
        RuntimeBundleLowerer::lowerStaticCtypesTypeFieldDescriptorGet(op,
                                                                      *object);
    if (mlir::succeeded(descriptorResult))
      return mlir::success();
    return op.emitError()
           << "attr.get type object has no static runtime attribute '"
           << op.getName() << "'";
  }
  if (object->kind != RuntimeBundle::Kind::Object)
    return op.emitError() << "attr.get object has no lowered runtime bundle";

  if (object->ctypes &&
      object->ctypes->kind == RuntimeCtypesEvidence::Kind::Library)
    return RuntimeBundleLowerer::lowerStaticCtypesAttrGet(op, *object);
  if (object->ctypes &&
      object->ctypes->kind == RuntimeCtypesEvidence::Kind::Module)
    return RuntimeBundleLowerer::lowerStaticCtypesModuleAttrGet(op, *object);
  if (object->ctypes &&
      object->ctypes->kind == RuntimeCtypesEvidence::Kind::FieldDescriptor)
    return RuntimeBundleLowerer::lowerStaticCtypesFieldDescriptorAttrGet(
        op, *object);
  if (object->ctypes &&
      object->ctypes->kind == RuntimeCtypesEvidence::Kind::Cell) {
    mlir::LogicalResult fieldResult =
        RuntimeBundleLowerer::lowerStaticCtypesFieldAttrGet(op, *object);
    if (mlir::succeeded(fieldResult))
      return mlir::success();
  }
  if (object->ctypes &&
      object->ctypes->kind == RuntimeCtypesEvidence::Kind::Cell &&
      op.getName() == "value")
    return RuntimeBundleLowerer::lowerStaticCtypesValueAttrGet(op, *object);

  if (object->kind == RuntimeBundle::Kind::Object &&
      runtimeContractName(op.getObject().getType()) ==
          "builtins.StopIteration" &&
      op.getName() == "value") {
    if (object->physicalValues().size() < 3)
      return op.emitError()
             << "StopIteration.value requires exception message storage";
    mlir::Type stringType = runtimeContractType(context, "builtins.str");
    RuntimeBundle result = RuntimeBundle::objectWithOwnership(
        stringType,
        mlir::ValueRange{object->physicalValues()[1],
                         object->physicalValues()[2]},
        ownership::logicalOwnershipKind(stringType, /*ownsObject=*/false));
    if (!py::isAssignableTo(result.objectValue.contract,
                            op.getResult().getType(), op))
      return op.emitError() << "attribute evidence "
                            << result.objectValue.contract
                            << " is not assignable to result "
                            << op.getResult().getType();
    valueBundles[op.getResult()] = std::move(result);
    erase.push_back(op);
    return mlir::success();
  }

  // exception.args materializes from the message payload through the
  // BaseException manifest primitive: builtin exceptions have no field
  // storage to read (a 3-word header plus the message pair).
  if (object->kind == RuntimeBundle::Kind::Object && op.getName() == "args") {
    std::string contract = runtimeContractName(op.getObject().getType());
    llvm::StringRef leaf = llvm::StringRef(contract).rsplit('.').second;
    if (leaf.empty())
      leaf = contract;
    // User exception classes route through the same primitive: they share
    // the taxonomy's 3-word-header-plus-message shape (field declarations
    // are rejected at emit time), so the ancestor check is the class analog
    // of the taxonomy name lookup.
    bool isExceptionShaped =
        py::exceptions::findByName(leaf) != nullptr ||
        RuntimeBundleLowerer::exceptionAncestorContractFor(
            op.getObject().getType())
            .has_value();
    if (isExceptionShaped && object->physicalValues().size() == 3) {
      std::optional<RuntimeSymbol> argsPrimitive =
          manifest.primitive("builtins.BaseException", "args");
      if (!argsPrimitive)
        return op.emitError()
               << "runtime manifest has no BaseException args primitive";
      llvm::SmallVector<const RuntimeBundle *, 1> sources{object};
      llvm::SmallVector<mlir::Value, 4> operands;
      builder.setInsertionPoint(op);
      if (mlir::failed(buildRuntimeCallOperands(op, *argsPrimitive, sources,
                                                operands,
                                                /*allowUnusedSources=*/false)))
        return mlir::failure();
      mlir::func::CallOp call = RuntimeBundleLowerer::createRuntimeCall(
          op.getLoc(), *argsPrimitive, operands);
      RuntimeBundle result;
      if (mlir::failed(RuntimeBundleLowerer::bundleRuntimeResults(
              op, op.getResult().getType(), call, result)))
        return mlir::failure();
      valueBundles[op.getResult()] = std::move(result);
      erase.push_back(op);
      return mlir::success();
    }
  }

  // ExceptionGroup.message / .exceptions read the message lane and the
  // extended member block through manifest primitives — like args, there is
  // no field slot to load, so this must run before the class-field paths.
  if (object->kind == RuntimeBundle::Kind::Object &&
      (op.getName() == "message" || op.getName() == "exceptions")) {
    std::string contract = runtimeContractName(op.getObject().getType());
    // BaseException/Exception join in: a group member read back through
    // .exceptions is statically the tuple's BaseException element type while
    // the dynamic class stays the group's. Non-exception contracts still
    // fall through to the class-schema diagnostic.
    bool groupShaped = contract == "builtins.BaseExceptionGroup" ||
                       contract == "builtins.ExceptionGroup" ||
                       contract == "builtins.BaseException" ||
                       contract == "builtins.Exception";
    if (groupShaped && object->physicalValues().size() == 3) {
      std::optional<RuntimeSymbol> primitive =
          manifest.primitive(contract, op.getName());
      if (!primitive)
        primitive =
            manifest.primitive("builtins.BaseExceptionGroup", op.getName());
      if (!primitive)
        return op.emitError()
               << "runtime manifest has no BaseExceptionGroup " << op.getName()
               << " primitive";
      llvm::SmallVector<const RuntimeBundle *, 1> sources{object};
      llvm::SmallVector<mlir::Value, 4> operands;
      builder.setInsertionPoint(op);
      if (mlir::failed(buildRuntimeCallOperands(op, *primitive, sources,
                                                operands,
                                                /*allowUnusedSources=*/false)))
        return mlir::failure();
      mlir::func::CallOp call = RuntimeBundleLowerer::createRuntimeCall(
          op.getLoc(), *primitive, operands);
      RuntimeBundle result;
      if (mlir::failed(RuntimeBundleLowerer::bundleRuntimeResults(
              op, op.getResult().getType(), call, result)))
        return mlir::failure();
      valueBundles[op.getResult()] = std::move(result);
      erase.push_back(op);
      return mlir::success();
    }
  }

  if (isMethodDescriptorKind(op) &&
      RuntimeBundleLowerer::classDefinesMethod(op.getObject().getType(),
                                               op.getName())) {
    RuntimeBundle result =
        RuntimeBundle::object(op.getResult().getType(), mlir::ValueRange{});
    result.boundMethodReceiver = std::make_shared<RuntimeBundle>(*object);
    result.boundMethodName = op.getName().str();
    valueBundles[op.getResult()] = std::move(result);
    erase.push_back(op);
    return mlir::success();
  }

  py::ClassOp classOp =
      RuntimeBundleLowerer::classForContract(op.getObject().getType());
  std::optional<unsigned> fieldIndex;
  llvm::SmallVector<mlir::Type, 8> fieldTypes;
  if (classOp) {
    fieldIndex = RuntimeBundleLowerer::classFieldIndex(classOp, op.getName());
    fieldTypes = RuntimeBundleLowerer::classFieldContractTypes(classOp);
  }
  if (classOp && fieldIndex && RuntimeBundleLowerer::isCellClassOp(classOp))
    return RuntimeBundleLowerer::lowerCellAttrGet(op, *object, classOp,
                                                  *fieldIndex);
  // Before ANY layout-derived path: an exception-backed class has no field
  // lanes at all (its ABI is the taxonomy's header + message), so a header
  // slot or payload offset computed from the field index would land on the
  // extended words that hold the group/field blocks.
  if (classOp && fieldIndex &&
      RuntimeBundleLowerer::exceptionAncestorContract(classOp))
    return RuntimeBundleLowerer::lowerExceptionFieldAttrGet(op, *object,
                                                            classOp,
                                                            *fieldIndex);
  if (fieldIndex) {
    if (*fieldIndex >= fieldTypes.size())
      return op.emitError() << "class field metadata is malformed for "
                            << classOp.getSymName();
    mlir::Type fieldType = fieldTypes[*fieldIndex];
    // The store wrote nothing, so the read loads nothing: a `type[X]` field's
    // value is its declared type, and the bundle is rebuilt from it.
    if (auto typeField = mlir::dyn_cast<py::TypeType>(fieldType)) {
      valueBundles[op.getResult()] =
          RuntimeBundle::typeObject(fieldType, typeField.getInstanceType());
      erase.push_back(op);
      return mlir::success();
    }
    if (std::optional<unsigned> primitiveSlot =
            primitiveFieldSlot(fieldType, *fieldIndex)) {
      builder.setInsertionPoint(op);
      mlir::FailureOr<mlir::Value> header =
          RuntimeBundleLowerer::objectPhysicalHeader(op, object->objectValue);
      if (mlir::failed(header))
        return mlir::failure();
      mlir::Value slotIndex = mlir::arith::ConstantIndexOp::create(
          builder, op.getLoc(), *primitiveSlot);
      mlir::Value raw =
          mlir::memref::LoadOp::create(builder, op.getLoc(), *header, slotIndex)
              .getResult();
      // bool's physical lane IS the i1, so the word is narrowed back into one
      // rather than carried as primitive-i64 evidence (which only int has).
      if (isBoolFieldType(fieldType)) {
        mlir::Value zero =
            mlir::arith::ConstantIntOp::create(builder, op.getLoc(), 0, 64)
                .getResult();
        mlir::Value flag =
            mlir::arith::CmpIOp::create(builder, op.getLoc(),
                                        mlir::arith::CmpIPredicate::ne, raw,
                                        zero)
                .getResult();
        RuntimeBundle result = RuntimeBundle::objectWithOwnership(
            fieldType, mlir::ValueRange{flag},
            ownership::logicalOwnershipKind(fieldType, /*ownsObject=*/false));
        if (!py::isAssignableTo(result.objectValue.contract,
                                op.getResult().getType(), op))
          return op.emitError()
                 << "attribute evidence " << result.objectValue.contract
                 << " is not assignable to result " << op.getResult().getType();
        valueBundles[op.getResult()] = std::move(result);
        erase.push_back(op);
        return mlir::success();
      }
      mlir::Value valid =
          mlir::arith::ConstantIntOp::create(builder, op.getLoc(), 1, 1)
              .getResult();
      RuntimeBundle result = RuntimeBundle::objectWithOwnership(
          fieldType, mlir::ValueRange{},
          ownership::logicalOwnershipKind(fieldType,
                                          /*ownsObject=*/false));
      result.primitiveI64 = RuntimePrimitiveI64Evidence{raw, valid};
      if (!py::isAssignableTo(result.objectValue.contract,
                              op.getResult().getType(), op))
        return op.emitError()
               << "attribute evidence " << result.objectValue.contract
               << " is not assignable to result " << op.getResult().getType();
      valueBundles[op.getResult()] = std::move(result);
      erase.push_back(op);
      return mlir::success();
    }
  }

  bool boxedField =
      fieldIndex && *fieldIndex < fieldTypes.size() &&
      RuntimeBundleLowerer::classFieldStoredBoxed(fieldTypes[*fieldIndex]);

  // A box-fronted field's LANES always come from the box words, never from the
  // recorded bundle, for two independent reasons. (1) The words are the storage:
  // a store from any other frame lands there and nowhere else. (2) Loading from
  // the box is what keeps the INSTANCE live at the read — with the field's value
  // no longer flattened into the instance's lane list, an evidence-only read
  // uses none of the instance's lanes, so the release planner sees the instance
  // die at the store and `__ly_dealloc_Stack` lands before the read.
  //
  // The recorded bundle is still consulted, for the facts that are not lanes
  // (element/key evidence, a more specific contract than the field's
  // annotation). It is a CACHE: `dropObjectFieldEvidence` clears it at every
  // boundary this walk cannot see a store through, so a hit means the box still
  // holds the object the cache describes and the two are consistent.
  // ⛔ AND THE CACHE IS NOT CONSULTED WHEN THE READ WANTS A UNION AND THE
  // CACHE HOLDS A MEMBER. The entry records what was last STORED --
  // `self.name: str | None = "a"` caches a `builtins.str` -- and
  // `isAssignableTo(str, str | None)` is true, so the narrower bundle was
  // handed back where the union's own lanes were expected. Every consumer of a
  // union reads the TAG from lane 0, so `union.test` took the str's header
  // memref for the tag and `arith.cmpi` inferred its result from the operand:
  //
  //     class H:
  //         def __init__(self) -> None:
  //             self.name: str | None = "a"
  //     print(H().name is None)
  //     # runtime bundle value 0 for 'builtins.bool' has type 'memref<2xi1>',
  //     # but ABI expects 'i1'
  //
  // Loud, and it refuses `x.f is None` on any Optional field whose member has
  // more than one lane -- `int | None` reaches the same code with one lane and
  // happens to survive. The lane slice below is authoritative here: the inline
  // splice writes the whole union into the instance on every store.
  //
  // ⭐ Keyed on the RESULT type, not on the field's. A narrowed read
  // (`attr.get` typed `str` after an isinstance) is exactly what the cache
  // exists for and the slice would hand it the union instead; a read typed as
  // the union needs the union whatever the cache last saw stored.
  //
  // ⛔ Why NOT compare the cached contract against the result type and keep
  // the entry when they agree, which is the narrower-looking guard: the entry
  // whose contract IS the union can still carry the member's lanes through
  // `boxedObject`, and that spelling reaches the same `arith.cmpi`. Measured --
  // it was written that way first and all four programs stayed red.
  bool unionRead = mlir::isa<py::UnionType>(op.getResult().getType());
  auto fieldBundle = object->fieldBundles.find(op.getName());
  if (!boxedField && !unionRead &&
      fieldBundle != object->fieldBundles.end()) {
    if (!fieldBundle->second)
      return op.emitError()
             << "attribute evidence for '" << op.getName() << "' is empty";
    RuntimeBundle result = *fieldBundle->second;
    if (result.boxedObject &&
        py::isAssignableTo(result.boxedObject->objectValue.contract,
                           op.getResult().getType(), op))
      result = *result.boxedObject;
    if (!py::isAssignableTo(result.objectValue.contract,
                            op.getResult().getType(), op))
      return op.emitError()
             << "attribute evidence " << result.objectValue.contract
             << " is not assignable to result " << op.getResult().getType();
    result.setObjectLogicalOwnership(/*ownsObject=*/false);
    result.fieldAliasOwner = op.getObject();
    result.fieldAliasName = op.getName().str();
    valueBundles[op.getResult()] = std::move(result);
    erase.push_back(op);
    return mlir::success();
  }

  auto rebuildBoxedFieldLanes =
      [&](llvm::ArrayRef<mlir::Type> laneTypes, mlir::Value box)
      -> mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>> {
    builder.setInsertionPoint(op);
    llvm::SmallVector<mlir::Value, 4> rebuilt;
    for (auto [index, type] : llvm::enumerate(laneTypes)) {
      auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type);
      if (!memrefType)
        return op.emitError()
               << "box-fronted field '" << op.getName()
               << "' expects memref physical values, got " << type;
      mlir::Value ptrIndex = mlir::arith::ConstantIndexOp::create(
          builder, op.getLoc(),
          box_abi::kPointerWordBase + static_cast<std::int64_t>(index));
      mlir::Value sizeIndex = mlir::arith::ConstantIndexOp::create(
          builder, op.getLoc(),
          box_abi::kSizeWordBase + static_cast<std::int64_t>(index));
      mlir::Value ptrWord =
          mlir::memref::LoadOp::create(builder, op.getLoc(), box, ptrIndex)
              .getResult();
      mlir::Value sizeWord =
          mlir::memref::LoadOp::create(builder, op.getLoc(), box, sizeIndex)
              .getResult();
      rebuilt.push_back(RuntimeBundleLowerer::memrefFromBoxWords(
          builder, op.getLoc(), ptrWord, sizeWord, memrefType));
    }
    return rebuilt;
  };
  auto rebuildBoxedFieldValues =
      [&](mlir::Type fieldContract, mlir::Value box)
      -> mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>> {
    mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> arrayTypes =
        RuntimeBundleLowerer::runtimeValueTypesFor(op, fieldContract,
                                                   "class field ABI");
    if (mlir::failed(arrayTypes))
      return mlir::failure();
    return rebuildBoxedFieldLanes(*arrayTypes, box);
  };

  if (auto unionType = mlir::dyn_cast<py::UnionType>(op.getObject().getType())) {
    if (object->physicalValues().empty())
      return op.emitError() << "union attribute input has no runtime tag";

    mlir::Type commonFieldType;
    llvm::SmallVector<mlir::Type, 8> commonValueTypes;
    llvm::SmallVector<mlir::Value, 4> selectedValues;
    mlir::Value inputTag = object->physicalValues().front();

    builder.setInsertionPoint(op);
    for (auto [memberIndex, memberType] :
         llvm::enumerate(unionType.getMemberTypes())) {
      py::ClassOp memberClass =
          RuntimeBundleLowerer::classForContract(memberType);
      if (!memberClass)
        return op.emitError() << "union member " << memberType
                              << " has no class schema for attribute '"
                              << op.getName() << "'";
      std::optional<unsigned> memberFieldIndex =
          RuntimeBundleLowerer::classFieldIndex(memberClass, op.getName());
      if (!memberFieldIndex)
        return op.emitError() << "class " << memberClass.getSymName()
                              << " has no field '" << op.getName()
                              << "' for union attribute access";
      llvm::SmallVector<mlir::Type, 8> memberFieldTypes =
          RuntimeBundleLowerer::classFieldContractTypes(memberClass);
      if (*memberFieldIndex >= memberFieldTypes.size())
        return op.emitError()
               << "class field metadata is malformed for "
               << memberClass.getSymName();
      mlir::Type memberFieldType = memberFieldTypes[*memberFieldIndex];
      if (primitiveFieldSlot(memberFieldType, *memberFieldIndex))
        return op.emitError()
               << "primitive union field attribute access is not supported";

      // Slice with STORAGE types: a box-fronted member field occupies one
      // box16 slot, and slicing by the contract's array shape would read the
      // neighbouring fields' lanes as if they were payload arrays.
      mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> memberValueTypes =
          RuntimeBundleLowerer::classFieldStorageValueTypes(
              op, memberFieldType, *memberFieldIndex, "union field ABI");
      if (mlir::failed(memberValueTypes))
        return mlir::failure();
      if (!commonFieldType) {
        commonFieldType = memberFieldType;
        commonValueTypes = *memberValueTypes;
      } else if (commonFieldType != memberFieldType ||
                 commonValueTypes != *memberValueTypes) {
        return op.emitError()
               << "union field '" << op.getName()
               << "' has incompatible member field types";
      }

      mlir::FailureOr<unsigned> memberOffset =
          RuntimeBundleLowerer::unionMemberValueOffset(
              op, unionType, static_cast<unsigned>(memberIndex),
              "union field member ABI");
      if (mlir::failed(memberOffset))
        return mlir::failure();
      mlir::FailureOr<unsigned> fieldOffset =
          RuntimeBundleLowerer::classFieldValueOffset(
              op, memberClass, *memberFieldIndex, "union field ABI");
      if (mlir::failed(fieldOffset))
        return mlir::failure();
      unsigned offset = *memberOffset + *fieldOffset;
      if (offset + commonValueTypes.size() > object->physicalValues().size())
        return op.emitError() << "union field ABI exceeds object payload";

      llvm::SmallVector<mlir::Value, 4> memberValues;
      appendValueSlice(object->physicalValues(), offset,
                       static_cast<unsigned>(commonValueTypes.size()),
                       memberValues);
      if (selectedValues.empty()) {
        selectedValues = memberValues;
        continue;
      }

      mlir::Value tag = mlir::arith::ConstantIntOp::create(
          builder, op.getLoc(), static_cast<std::int64_t>(memberIndex), 64);
      mlir::Value active = mlir::arith::CmpIOp::create(
          builder, op.getLoc(), mlir::arith::CmpIPredicate::eq, inputTag, tag);
      for (auto [index, memberValue] : llvm::enumerate(memberValues))
        selectedValues[index] =
            mlir::arith::SelectOp::create(builder, op.getLoc(), active,
                                          memberValue, selectedValues[index])
                .getResult();
    }

    if (commonFieldType &&
        RuntimeBundleLowerer::classFieldStoredBoxed(commonFieldType)) {
      if (selectedValues.empty())
        return op.emitError() << "box-fronted union field has no box slot";
      mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>> rebuilt =
          rebuildBoxedFieldValues(commonFieldType, selectedValues.front());
      if (mlir::failed(rebuilt))
        return mlir::failure();
      selectedValues = std::move(*rebuilt);
      if (!py::isAssignableTo(commonFieldType, op.getResult().getType(), op))
        return op.emitError() << "attribute evidence " << commonFieldType
                              << " is not assignable to result "
                              << op.getResult().getType();
      RuntimeValue element{commonFieldType, selectedValues,
                           ownership::logicalOwnershipKind(
                               commonFieldType, /*ownsObject=*/false)};
      return bindRetainedEvidenceValue(op, op.getResult(),
                                       "box-fronted union field load", element);
    }

    RuntimeBundle result = RuntimeBundle::objectWithOwnership(
        commonFieldType, selectedValues,
        ownership::logicalOwnershipKind(commonFieldType,
                                        /*ownsObject=*/false));
    if (!py::isAssignableTo(result.objectValue.contract,
                            op.getResult().getType(), op))
      return op.emitError() << "attribute evidence "
                            << result.objectValue.contract
                            << " is not assignable to result "
                            << op.getResult().getType();
    valueBundles[op.getResult()] = std::move(result);
    erase.push_back(op);
    return mlir::success();
  }

  if (!classOp)
    return op.emitError() << "attr.get object type has no class schema";
  if (!fieldIndex)
    return op.emitError() << "class " << classOp.getSymName()
                          << " has no field '" << op.getName() << "'";
  if (*fieldIndex >= fieldTypes.size())
    return op.emitError() << "class field metadata is malformed for "
                          << classOp.getSymName();

  mlir::Type fieldType = fieldTypes[*fieldIndex];
  mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> valueTypes =
      RuntimeBundleLowerer::classFieldStorageValueTypes(op, fieldType,
                                                        *fieldIndex,
                                                        "class field ABI");
  if (mlir::failed(valueTypes))
    return mlir::failure();
  mlir::FailureOr<unsigned> offset =
      RuntimeBundleLowerer::classFieldValueOffset(op, classOp, *fieldIndex,
                                                  "class field ABI");
  if (mlir::failed(offset))
    return mlir::failure();
  if (*offset + valueTypes->size() > object->physicalValues().size())
    return op.emitError() << "class field ABI exceeds object payload";

  llvm::SmallVector<mlir::Value, 4> values;
  appendValueSlice(object->physicalValues(), *offset,
                   static_cast<unsigned>(valueTypes->size()), values);
  // The cache entry, when it is still valid, supplies the facts that are not
  // lanes AND the concrete contract, which for an erased/protocol-typed field is
  // more specific than the annotation. Its lane TYPES then say how many box
  // words to read back, so an erased field reads back as its concrete object
  // rather than as one opaque handle.
  const RuntimeBundle *cached = nullptr;
  if (boxedField && fieldBundle != object->fieldBundles.end() &&
      fieldBundle->second) {
    cached = fieldBundle->second.get();
    if (cached->boxedObject)
      cached = cached->boxedObject.get();
    if (cached->kind != RuntimeBundle::Kind::Object ||
        cached->physicalValues().empty() ||
        !py::isAssignableTo(cached->objectValue.contract,
                            op.getResult().getType(), op))
      cached = nullptr;
  }
  mlir::Type loadedContract = cached ? cached->objectValue.contract : fieldType;
  if (boxedField) {
    if (values.empty())
      return op.emitError() << "box-fronted field has no box slot";
    mlir::Value box = values.front();
    llvm::SmallVector<mlir::Type, 8> laneTypes;
    if (cached) {
      for (mlir::Value lane : cached->physicalValues())
        laneTypes.push_back(lane.getType());
    } else {
      mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> contractTypes =
          RuntimeBundleLowerer::runtimeValueTypesFor(op, fieldType,
                                                    "class field ABI");
      if (mlir::failed(contractTypes))
        return mlir::failure();
      laneTypes = std::move(*contractTypes);
    }
    mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>> rebuilt =
        rebuildBoxedFieldLanes(laneTypes, box);
    if (mlir::failed(rebuilt))
      return mlir::failure();
    values = std::move(*rebuilt);
    // A read TAKES A REFERENCE. The reconstructed values are fresh SSA with no
    // borrowed-entry provenance the planner could trace, and the slot's own
    // reference is not the reader's to rely on: `old = o.f; o.f = fresh` drops
    // it, and CPython keeps `old` alive because the read is a new reference.
    // This is the aggregate-slot contract, and the caller-owns-result return
    // convention then holds. The field alias rides along so an in-place
    // mutation's write-back still reaches the box.
    //
    // Except when the read is about to be mutated in place, where it stays
    // pinned to the slot with no reference of its own. The reason is the lane
    // tuple, not the read: a reallocating mutation primitive spells itself as
    // "consume the container and hand back another one" (`transfer_args = [0]`),
    // which RENAMES the tuple, and the affine verifier's resource identity IS
    // the tuple. Give this reader a token and inside a loop its release names a
    // tuple the transfer already consumed ("released through a value already
    // consumed by an ownership transfer" -- measured on
    // collections.Counter.update, five goldens).
    //
    // What CHANGED is that the pinned branch is no longer the unsafe one. The
    // transfer used to consume the SLOT's reference, since a borrow has none to
    // give, and freed the entity while the field still named it; the reference it
    // consumes is now manufactured at the transfer
    // (promoteInteriorViewForTransfer). So this condition selects an encoding,
    // not a safety level -- both branches are sound, which is why it must not be
    // widened to decide anything else.
    //
    // It stays syntactic for the same reason it cannot be deleted: only a lane
    // tuple can be renamed, so only a representation with lanes needs the split
    // at all. Behind the handle a mutation renames nothing, this branch has no
    // work to do, and a container read becomes an ordinary read.
    if (!RuntimeBundleLowerer::isMutableContainerContractName(
            runtimeShapeContractName(loadedContract)) ||
        !fieldReadFeedsInPlaceMutation(op.getResult())) {
      if (!py::isAssignableTo(loadedContract, op.getResult().getType(), op))
        return op.emitError() << "attribute evidence " << loadedContract
                              << " is not assignable to result "
                              << op.getResult().getType();
      RuntimeBundle read;
      if (cached) {
        read = *cached;
        read.objectValue.values.assign(values.begin(), values.end());
        read.setObjectLogicalOwnership(/*ownsObject=*/false);
      } else {
        read = RuntimeBundle::objectWithOwnership(
            loadedContract, values,
            ownership::logicalOwnershipKind(loadedContract,
                                           /*ownsObject=*/false));
      }
      read.fieldAliasOwner = op.getObject();
      read.fieldAliasName = op.getName().str();
      return bindRetainedEvidenceBundle(op, op.getResult(), std::move(read));
    }
  }
  RuntimeBundle result;
  if (cached) {
    result = *cached;
    result.objectValue.values.assign(values.begin(), values.end());
    result.setObjectLogicalOwnership(/*ownsObject=*/false);
  } else {
    result = RuntimeBundle::objectWithOwnership(
        loadedContract, values,
        ownership::logicalOwnershipKind(loadedContract,
                                        /*ownsObject=*/false));
  }
  result.fieldAliasOwner = op.getObject();
  result.fieldAliasName = op.getName().str();
  if (!py::isAssignableTo(result.objectValue.contract, op.getResult().getType(),
                          op))
    return op.emitError() << "attribute evidence "
                          << result.objectValue.contract
                          << " is not assignable to result "
                          << op.getResult().getType();
  valueBundles[op.getResult()] = std::move(result);
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerClassTest(py::ClassTestOp op) {
  const RuntimeBundle *object = RuntimeBundleLowerer::bundleFor(op.getInput());
  if (!object || object->kind != RuntimeBundle::Kind::Object)
    return op.emitError() << "class.test input has no lowered object bundle";

  mlir::FailureOr<llvm::SmallVector<std::int64_t, 8>> targetIds =
      RuntimeBundleLowerer::runtimeClassIdsForNominalTarget(op, op.getTarget());
  if (mlir::failed(targetIds))
    return mlir::failure();

  mlir::FailureOr<mlir::Value> header =
      RuntimeBundleLowerer::objectPhysicalHeader(op, object->objectValue);
  if (mlir::failed(header))
    return mlir::failure();

  // Every sibling lowering in this file sets the insertion point first; this
  // one did not, so it emitted wherever the builder happened to point. When
  // the test is the FIRST py op in its function there is nowhere sensible,
  // and the operands came out detached: `def f(a: A) -> bool: return
  // isinstance(a, B)` failed with "operation's operand is unlinked".
  builder.setInsertionPoint(op);
  mlir::Location loc = op.getLoc();
  mlir::Value storage = *header;
  mlir::Type dynamicHeaderType =
      mlir::MemRefType::get({mlir::ShapedType::kDynamic}, builder.getI64Type());
  if (storage.getType() != dynamicHeaderType)
    storage =
        mlir::memref::CastOp::create(builder, loc, dynamicHeaderType, storage)
            .getResult();

  mlir::Value classIdSlot =
      mlir::arith::ConstantIndexOp::create(builder, loc, 1);
  mlir::Value actualClassId =
      mlir::memref::LoadOp::create(builder, loc, storage, classIdSlot)
          .getResult();
  mlir::Value result = mlir::arith::ConstantIntOp::create(builder, loc, 0, 1);
  for (std::int64_t targetId : *targetIds) {
    mlir::Value expected =
        mlir::arith::ConstantIntOp::create(builder, loc, targetId, 64);
    mlir::Value match = mlir::arith::CmpIOp::create(
        builder, loc, mlir::arith::CmpIPredicate::eq, actualClassId, expected);
    result = mlir::arith::OrIOp::create(builder, loc, result, match);
  }

  op.getResult().replaceAllUsesWith(result);
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerAttrSet(py::AttrSetOp op) {
  const RuntimeBundle *object = RuntimeBundleLowerer::bundleFor(op.getObject());
  const RuntimeBundle *value = RuntimeBundleLowerer::bundleFor(op.getValue());
  if (object && object->kind == RuntimeBundle::Kind::TypeObject)
    return op.emitError()
           << "class static attribute mutation is not supported; declare "
              "static attributes in the class body";
  if (!object || object->kind != RuntimeBundle::Kind::Object)
    return op.emitError() << "attr.set object has no lowered runtime bundle";
  if (object->ctypes &&
      object->ctypes->kind == RuntimeCtypesEvidence::Kind::Symbol)
    return RuntimeBundleLowerer::lowerStaticCtypesAttrSet(op, *object, value);
  if (object->ctypes &&
      object->ctypes->kind == RuntimeCtypesEvidence::Kind::Cell) {
    mlir::LogicalResult fieldResult =
        RuntimeBundleLowerer::lowerStaticCtypesFieldAttrSet(op, *object, value);
    if (mlir::succeeded(fieldResult))
      return mlir::success();
    if (op.getName() == "value")
      return RuntimeBundleLowerer::lowerStaticCtypesValueAttrSet(op, *object,
                                                                 value);
  }
  // ⭐ Storing a `type[X]` STORES NOTHING. The field's physical shape is empty
  // because which class the value names is decided by its type, so the write
  // has no slot to write and the read reconstructs it from the field's
  // declared type. The assignability the emitter already checked is what makes
  // the two agree.
  if (value && value->kind == RuntimeBundle::Kind::TypeObject &&
      mlir::isa<py::TypeType>(op.getValue().getType())) {
    erase.push_back(op);
    return mlir::success();
  }
  if (!value || value->kind != RuntimeBundle::Kind::Object)
    return op.emitError() << "attr.set value has no lowered runtime bundle";

  py::ClassOp classOp =
      RuntimeBundleLowerer::classForContract(op.getObject().getType());
  if (!classOp)
    return op.emitError() << "attr.set object type has no class schema";
  std::optional<unsigned> fieldIndex =
      RuntimeBundleLowerer::classFieldIndex(classOp, op.getName());
  if (!fieldIndex)
    return op.emitError() << "class " << classOp.getSymName()
                          << " has no field '" << op.getName() << "'";
  llvm::SmallVector<mlir::Type, 8> fieldTypes =
      RuntimeBundleLowerer::classFieldContractTypes(classOp);
  if (*fieldIndex >= fieldTypes.size())
    return op.emitError() << "class field metadata is malformed for "
                          << classOp.getSymName();
  if (RuntimeBundleLowerer::isCellClassOp(classOp))
    return RuntimeBundleLowerer::lowerCellAttrSet(op, *object, *value, classOp,
                                                  *fieldIndex);
  // ⭐ A FUNCTION VALUE'S CONTRACT IS ITS TARGET'S, NOT ITS REPRESENTATION'S.
  // A lowered function reference is `builtins.function` whatever it points at,
  // so comparing that name against a `Callable[...]` field asks a runtime
  // representation a logical question and always answers no:
  //
  //     self._f: Callable[[], int] = f
  //     attribute value '!py.contract<"builtins.function">' is not assignable
  //     to field '!py.callable<[], returns = [!py.contract<"builtins.int">]>'
  //
  // The bundle names the target it holds, and that target declares a callable.
  // Asking THAT one is the same question the emitter already answered when it
  // accepted the assignment.
  //
  // ⛔ Why the lane shapes do not have to be reconciled first, which is what
  // this note said when the layer below was found: a Callable field is stored
  // boxed, and the boxed path writes the value's PAYLOAD into the slot's box16
  // rather than splicing a fixed lane tuple into the instance. There is no
  // tuple here to disagree about.
  bool assignable = py::isAssignableTo(value->objectValue.contract,
                                       fieldTypes[*fieldIndex], op);
  if (!assignable && !value->functionTarget.empty() &&
      mlir::isa<py::CallableType>(fieldTypes[*fieldIndex])) {
    if (mlir::func::FuncOp target =
            module.lookupSymbol<mlir::func::FuncOp>(value->functionTarget))
      if (py::CallableType declared = callableTypeOf(target))
        assignable =
            py::isAssignableTo(declared, fieldTypes[*fieldIndex], op);
  }
  if (!assignable)
    return op.emitError() << "attribute value " << value->objectValue.contract
                          << " is not assignable to field "
                          << fieldTypes[*fieldIndex];

  // Same reason as the read side: an exception-backed class has no field
  // lanes, so this must precede every layout-derived store.
  if (RuntimeBundleLowerer::exceptionAncestorContract(classOp))
    return RuntimeBundleLowerer::lowerExceptionFieldAttrSet(
        op, *object, *value, classOp, *fieldIndex);

  if (primitiveFieldSlot(fieldTypes[*fieldIndex], *fieldIndex)) {
    if (mlir::failed(RuntimeBundleLowerer::storePrimitiveFieldSlot(
            op, *object, *value, fieldTypes[*fieldIndex], *fieldIndex,
            op.getName())))
      return mlir::failure();
    erase.push_back(op);
    return mlir::success();
  }

  // THE store path for every object-contract field: the slot's box16 pointer is
  // fixed at construction, so a set swaps the payload the box holds and writes
  // nothing into the instance's SSA lanes. That is what makes the effect
  // observable to every other frame holding the instance — a callee taking it
  // as a parameter, a caller reading after the call, an arm of a branch — and it
  // is why `PathIsHeap` (rfc/object-ownership-kernel.md §2.2) needs no lane
  // width change: `dict` already carried a store through a call at FIVE lanes
  // while one-lane `io.StringIO` lost one, so the discriminator was never the
  // width, only whether the destination was a heap slot.
  if (RuntimeBundleLowerer::classFieldStoredBoxed(fieldTypes[*fieldIndex])) {
    mlir::FailureOr<unsigned> offset =
        RuntimeBundleLowerer::classFieldValueOffset(op, classOp, *fieldIndex,
                                                    "class field ABI");
    if (mlir::failed(offset))
      return mlir::failure();
    if (*offset >= object->physicalValues().size())
      return op.emitError() << "class field ABI exceeds object payload";
    mlir::Value box = object->physicalValues()[*offset];
    std::string slotName = (llvm::Twine("class.") + op.getName()).str();
    bool releaseOwnedSource = false;
    if (const RuntimeBundle *source =
            RuntimeBundleLowerer::concreteObjectForOwnership(*value)) {
      releaseOwnedSource =
          source->kind == RuntimeBundle::Kind::Object &&
          source->objectValue.ownership == ownership::OwnershipKind::Own &&
          !source->physicalValues().empty() &&
          !RuntimeBundleLowerer::storedSourceOutlivesStore(op, op.getValue());
    }
    mlir::FailureOr<RuntimeBundle> stored =
        RuntimeBundleLowerer::storeBoxedFieldPayloadInPlace(op, box, *value,
                                                            slotName);
    if (mlir::failed(stored))
      return mlir::failure();
    if (releaseOwnedSource &&
        mlir::failed(RuntimeBundleLowerer::releaseAggregateSlot(
            op, *value, llvm::Twine(slotName).concat(".source").str())))
      return mlir::failure();
    // Refresh the field-evidence cache. The instance's LANES are untouched --
    // that is the point of storing into the box -- so this republishes the
    // bundle without re-rooting anything, and no owned-local marker has to
    // follow.
    RuntimeBundle updated = *object;
    updated.fieldBundles[op.getName()] =
        std::make_shared<RuntimeBundle>(std::move(*stored));
    valueBundles[op.getObject()] = std::move(updated);
    erase.push_back(op);
    return mlir::success();
  }

  // Residual: a field with no single object contract to put behind a handle —
  // a union (tag plus every member's lanes), a zero-lane contract, or an
  // int/bool past the last header word. These keep the pre-4a lane splice, and
  // with it the pre-4a defect: a store here is only visible where these lanes
  // are. Union fields are the only shape that reaches it in practice.
  //
  // ⛔ WHICH IS A WRONG ANSWER WHEN THE RECEIVER CAME FROM A CALLER, so that
  // is refused here. The splice writes the receiver's own SSA expansion; a
  // caller holds its own and never sees it:
  //
  //     def rebind(b: Box) -> None:
  //         b.f = 5                   # b.f: Optional[int]
  //     rebind(o); print(o.f is None) # printed True; CPython prints False
  //
  // Silent, and reachable only since Optional stores began lowering at all --
  // before that this arm refused every union store, so the hole opened with
  // the feature. Refusing is the floor until the field is stored behind a
  // handle the way every other field is; a splice cannot cross a frame.
  //
  // ⭐ Gated on the function actually being CALLED, which is what keeps `self`
  // and `__init__` accepted. The emitter inlines a call to a known method, so
  // the store the caller executes is the inlined one, on the caller's own
  // value; the method body still exists as a symbol and is lowered here, but
  // nothing reaches it and symbol DCE removes it later. Refusing on "receiver
  // is a parameter" alone rejected every constructor in the suite.
  //
  // ⛔ Why the reference scan and not `SymbolTable::symbolKnownUseEmpty`,
  // which is the obvious spelling: a Python-level call names its target
  // through `py.binding.ref`, whose binding is a plain StringAttr and not a
  // SymbolRef, so the symbol table sees no use and every callee looked dead.
  // Measured -- the refusal never fired.
  if (auto receiver = mlir::dyn_cast<mlir::BlockArgument>(op.getObject())) {
    mlir::Block *owner = receiver.getOwner();
    auto enclosing =
        mlir::dyn_cast_if_present<mlir::func::FuncOp>(owner->getParentOp());
    if (owner->isEntryBlock() && enclosing) {
      llvm::StringRef name = enclosing.getSymName();
      bool reached = false;
      mlir::ModuleOp moduleOp = module;
      moduleOp.walk([&](mlir::Operation *user) {
        if (auto call = mlir::dyn_cast<mlir::func::CallOp>(user)) {
          if (call.getCallee() == name)
            reached = true;
        } else if (auto ref = mlir::dyn_cast<py::BindingRefOp>(user)) {
          if (ref.getBinding() == name)
            reached = true;
        }
        return reached ? mlir::WalkResult::interrupt()
                       : mlir::WalkResult::advance();
      });
      if (reached)
        return op.emitError()
               << "storing into field '" << op.getName()
               << "' of a receiver that arrived as a parameter is not "
                  "supported for this field's type: the store writes the "
                  "receiver's own value lanes, so the caller would not see it";
    }
  }
  RuntimeBundle slotValue;
  {
    mlir::FailureOr<RuntimeBundle> storageValue =
        RuntimeBundleLowerer::materializeObjectBundleForStorage(
            op, *value, fieldTypes[*fieldIndex], "attribute value ABI");
    if (mlir::failed(storageValue))
      return mlir::failure();
    slotValue = std::move(*storageValue);
  }

  mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> fieldValueTypes =
      RuntimeBundleLowerer::classFieldStorageValueTypes(
          op, fieldTypes[*fieldIndex], *fieldIndex, "class field ABI");
  if (mlir::failed(fieldValueTypes))
    return mlir::failure();
  mlir::FailureOr<unsigned> offset =
      RuntimeBundleLowerer::classFieldValueOffset(op, classOp, *fieldIndex,
                                                  "class field ABI");
  if (mlir::failed(offset))
    return mlir::failure();

  llvm::SmallVector<mlir::Value, 8> values(object->physicalValues().begin(),
                                           object->physicalValues().end());
  if (*offset + fieldValueTypes->size() > values.size())
    return op.emitError() << "class field ABI exceeds object payload";
  llvm::SmallVector<mlir::Value, 4> oldValues;
  appendValueSlice(values, *offset,
                   static_cast<unsigned>(fieldValueTypes->size()), oldValues);
  builder.setInsertionPoint(op);
  std::string slotName = (llvm::Twine("class.") + op.getName()).str();
  // ⭐ A LAZILY-BOXED int has no physical values until the store materializes
  // one, and the question here is whether the value arrives OWNED -- which it
  // does either way. Asking `physicalValues()` before the materialization
  // answered "no" and skipped the `.source` release below, so the object the
  // widening had just boxed was retained for the slot and never released:
  // `self.v: Optional[int]` then `n.v = 7; n.v = None` leaked 52 B, while the
  // str member, which is never lazy, was correct. Same shape as the union
  // return's double materialization -- a decision read off a bundle before the
  // step that gives it values.
  bool releaseOwnedSource = false;
  if (const RuntimeBundle *source =
          RuntimeBundleLowerer::concreteObjectForOwnership(*value)) {
    releaseOwnedSource =
        source->kind == RuntimeBundle::Kind::Object &&
        source->objectValue.ownership == ownership::OwnershipKind::Own &&
        (!source->physicalValues().empty() ||
         RuntimeBundleLowerer::hasLazyPrimitiveI64Object(*source));
  }
  // And the release names the object the STORE materialized, not the lazy
  // bundle it was handed: releasing through the latter would box a SECOND int
  // and discharge that one instead.
  const RuntimeBundle &ownedSource =
      slotValue.unionActiveMember ? *slotValue.unionActiveMember : *value;
  const RuntimeBundle *oldSlotValue = nullptr;
  auto oldFieldBundle = object->fieldBundles.find(op.getName());
  if (oldFieldBundle != object->fieldBundles.end())
    oldSlotValue = oldFieldBundle->second.get();
  // This arm re-roots the field's lanes in the object's expansion (see the loop
  // at the end of this function), so it needs an answer to "will a later
  // release of this object still name the pre-store lanes?".
  bool markerFollows =
      RuntimeBundleLowerer::ownedLocalObjectMarkerFollowsExpansion(
          op.getObject());
  // A SELF-store (`ks = self._kids; ks.append(v); self._kids = ks`): the growth
  // primitive's transfer already moved the slot's token out and its owned
  // result handed one back, so the retain inside `replaceAggregateSlot`
  // restores the slot's single reference and there is no second one to give up.
  // The old lanes are also the pre-realloc ones, so the release would hand the
  // deallocator storage the primitive has already freed.
  bool selfStore = RuntimeBundleLowerer::aggregateSlotStoreIsSelfStore(
      oldValues, slotValue.physicalValues());
  if (mlir::failed(RuntimeBundleLowerer::replaceAggregateSlot(
          op, fieldTypes[*fieldIndex], oldValues, oldSlotValue,
          fieldTypes[*fieldIndex], slotValue, slotName,
          /*releaseMissingOldObjectSlot=*/true,
          /*releaseOldSlot=*/markerFollows && !selfStore)))
    return mlir::failure();
  if (releaseOwnedSource &&
      mlir::failed(RuntimeBundleLowerer::releaseAggregateSlot(
          op, ownedSource, llvm::Twine(slotName).concat(".source").str())))
    return mlir::failure();
  for (auto [index, replacement] : llvm::enumerate(slotValue.physicalValues()))
    values[*offset + index] = replacement;
  slotValue.setObjectLogicalOwnership(/*ownsObject=*/true);

  RuntimeBundle updated;
  if (mlir::failed(RuntimeBundleLowerer::makeObjectBundleWithOwnership(
          op, op.getObject().getType(), values, updated,
          object->objectValue.ownership)))
    return mlir::failure();
  updated.copyEvidenceFrom(*object);
  updated.fieldBundles[op.getName()] =
      std::make_shared<RuntimeBundle>(std::move(slotValue));
  if (mlir::failed(RuntimeBundleLowerer::markOwnedLocalObjectBundle(
          op, op.getObject(), updated)))
    return mlir::failure();
  valueBundles[op.getObject()] = std::move(updated);
  erase.push_back(op);
  return mlir::success();
}

bool RuntimeBundleLowerer::isCellClassOp(py::ClassOp classOp) {
  return classOp && classOp.getSymName().starts_with("__ly_cell$");
}

// A cell load rebuilds the content's value group from the slot box words and
// retains it: the content can be replaced through ANY frame holding the cell
// (that is the point of a cell), so a borrow pinned to the box would dangle
// across the next store.
mlir::LogicalResult RuntimeBundleLowerer::lowerCellAttrGet(
    py::AttrGetOp op, const RuntimeBundle &objectRef, py::ClassOp classOp,
    unsigned fieldIndex) {
  // Copy: binding the result below writes valueBundles.
  RuntimeBundle object = objectRef;
  mlir::Type content = op.getResult().getType();
  mlir::FailureOr<unsigned> offset = RuntimeBundleLowerer::classFieldValueOffset(
      op, classOp, fieldIndex, "nonlocal cell ABI");
  if (mlir::failed(offset))
    return mlir::failure();
  if (*offset >= object.physicalValues().size())
    return op.emitError() << "nonlocal cell ABI exceeds object payload";
  mlir::Value box = object.physicalValues()[*offset];
  mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> shapes =
      RuntimeBundleLowerer::slotStorageShapesFor(op, content,
                                                 "nonlocal cell load");
  if (mlir::failed(shapes))
    return mlir::failure();
  for (mlir::Type shape : *shapes) {
    auto memref = mlir::dyn_cast<mlir::MemRefType>(shape);
    if (!memref || memref.getRank() != 1)
      return op.emitError()
             << "nonlocal over " << content
             << " is not supported yet (content has no boxable value group)";
  }
  builder.setInsertionPoint(op);
  mlir::Location loc = op.getLoc();
  llvm::SmallVector<mlir::Value, 4> elementValues;
  for (auto [position, shape] : llvm::enumerate(*shapes)) {
    mlir::Value ptrIndex = mlir::arith::ConstantIndexOp::create(
        builder, loc,
        box_abi::kPointerWordBase + static_cast<std::int64_t>(position));
    mlir::Value sizeIndex = mlir::arith::ConstantIndexOp::create(
        builder, loc,
        box_abi::kSizeWordBase + static_cast<std::int64_t>(position));
    mlir::Value ptrWord =
        mlir::memref::LoadOp::create(builder, loc, box, ptrIndex).getResult();
    mlir::Value sizeWord =
        mlir::memref::LoadOp::create(builder, loc, box, sizeIndex).getResult();
    elementValues.push_back(RuntimeBundleLowerer::memrefFromBoxWords(
        builder, loc, ptrWord, sizeWord, mlir::cast<mlir::MemRefType>(shape)));
  }
  mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>> canonical =
      RuntimeBundleLowerer::unboxSlotElementValues(op, content, elementValues);
  if (mlir::failed(canonical))
    return mlir::failure();
  RuntimeValue element{content, *canonical,
                       ownership::logicalOwnershipKind(content,
                                                       /*ownsObject=*/false)};
  if (mlir::failed(bindRetainedEvidenceValue(op, op.getResult(),
                                             "nonlocal cell load", element)))
    return mlir::failure();
  erase.push_back(op);
  return mlir::success();
}

// A cell store swaps the slot box's content IN PLACE: retain the new content
// into the slot, release whatever the box held (a no-op while the owned flag
// is still zero), overwrite the handle words. The instance bundle is NOT
// respliced — the box is the shared mutable state, and the instance may be a
// borrowed capture whose lanes belong to another frame.
mlir::LogicalResult RuntimeBundleLowerer::lowerCellAttrSet(
    py::AttrSetOp op, const RuntimeBundle &objectRef,
    const RuntimeBundle &valueRef, py::ClassOp classOp, unsigned fieldIndex) {
  RuntimeBundle object = objectRef;
  RuntimeBundle value = valueRef;
  mlir::FailureOr<unsigned> offset = RuntimeBundleLowerer::classFieldValueOffset(
      op, classOp, fieldIndex, "nonlocal cell ABI");
  if (mlir::failed(offset))
    return mlir::failure();
  if (*offset >= object.physicalValues().size())
    return op.emitError() << "nonlocal cell ABI exceeds object payload";
  mlir::Value box = object.physicalValues()[*offset];
  if (mlir::failed(RuntimeBundleLowerer::storeBoxedFieldPayloadInPlace(
          op, box, value, "nonlocal.cell")))
    return mlir::failure();
  erase.push_back(op);
  return mlir::success();
}

// The field block of an exception-backed instance, allocated on first use.
// Word 4 of the extended exception header holds a [count, count x box16]
// block — the same shape word 3 uses for group members — so the taxonomy's
// fixed 3-word/message layout stays untouched while a subclass adds fields.
mlir::FailureOr<mlir::Value>
RuntimeBundleLowerer::exceptionFieldBlockWord(mlir::Operation *op,
                                              const RuntimeBundle &object,
                                              py::ClassOp classOp) {
  std::optional<RuntimeSymbol> fieldsBlock =
      manifest.primitive("builtins.BaseException", "fields_block");
  if (!fieldsBlock)
    return op->emitError()
           << "runtime manifest has no BaseException fields_block primitive";
  mlir::FailureOr<mlir::Value> header =
      RuntimeBundleLowerer::objectPhysicalHeader(op, object.objectValue);
  if (mlir::failed(header))
    return mlir::failure();
  mlir::Type expectedHeader = fieldsBlock->function.getFunctionType().getInput(0);
  mlir::Value headerValue = *header;
  if (headerValue.getType() != expectedHeader)
    return op->emitError() << "exception field access needs a "
                           << expectedHeader << " header, got "
                           << headerValue.getType();
  std::size_t fieldCount =
      RuntimeBundleLowerer::classFieldContractTypes(classOp).size();
  mlir::Value count = mlir::arith::ConstantIntOp::create(
      builder, op->getLoc(), static_cast<std::int64_t>(fieldCount), 64);
  mlir::func::CallOp call = RuntimeBundleLowerer::createRuntimeCall(
      op->getLoc(), *fieldsBlock, mlir::ValueRange{headerValue, count});
  return call.getResult(0);
}

// A field read rebuilds the payload's value group from the slot box words and
// takes a reference: the block is shared mutable state reachable from every
// handler binding the exception, so a borrow pinned to the box words would
// dangle across the next store (the cell rule, for the same reason).
mlir::LogicalResult RuntimeBundleLowerer::lowerExceptionFieldAttrGet(
    py::AttrGetOp op, const RuntimeBundle &objectRef, py::ClassOp classOp,
    unsigned fieldIndex) {
  RuntimeBundle object = objectRef;
  llvm::SmallVector<mlir::Type, 8> fieldTypes =
      RuntimeBundleLowerer::classFieldContractTypes(classOp);
  if (fieldIndex >= fieldTypes.size())
    return op.emitError() << "class field metadata is malformed for "
                          << classOp.getSymName();
  mlir::Type fieldType = fieldTypes[fieldIndex];
  if (!py::isAssignableTo(fieldType, op.getResult().getType(), op))
    return op.emitError() << "attribute evidence " << fieldType
                          << " is not assignable to result "
                          << op.getResult().getType();
  // An erased-`object` field's value IS the slot box (its words are the
  // canonical object handle), so this read takes the box address instead of
  // reconstructing a payload group from the box words. It stays a borrow: the
  // box belongs to the exception's field block, and releasing it here would
  // dispatch the payload's deallocator while the exception still owns it.
  if (RuntimeBundleLowerer::isBuiltinsObjectContract(fieldType)) {
    std::optional<RuntimeSymbol> boxPtr =
        manifest.primitive("builtins.BaseException", "payload_box_ptr");
    if (!boxPtr)
      return op.emitError() << "runtime manifest has no BaseException "
                               "payload_box_ptr primitive";
    mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> handleTypes =
        RuntimeBundleLowerer::runtimeValueTypesFor(op, fieldType,
                                                   "exception field load");
    if (mlir::failed(handleTypes) || handleTypes->size() != 1)
      return op.emitError()
             << "erased object field expects a single box handle lane";
    auto boxType = mlir::dyn_cast<mlir::MemRefType>(handleTypes->front());
    if (!boxType)
      return op.emitError() << "erased object field handle " << *handleTypes
                            << " is not a box lane";
    builder.setInsertionPoint(op);
    mlir::FailureOr<mlir::Value> block =
        RuntimeBundleLowerer::exceptionFieldBlockWord(op, object, classOp);
    if (mlir::failed(block))
      return mlir::failure();
    mlir::Value slot = mlir::arith::ConstantIntOp::create(
        builder, op.getLoc(), static_cast<std::int64_t>(fieldIndex), 64);
    mlir::Value boxWord =
        RuntimeBundleLowerer::createRuntimeCall(op.getLoc(), *boxPtr,
                                               mlir::ValueRange{*block, slot})
            .getResult(0);
    mlir::Value size = mlir::arith::ConstantIntOp::create(
        builder, op.getLoc(), box_abi::kWordsPerBox, 64);
    mlir::Value box = RuntimeBundleLowerer::memrefFromBoxWords(
        builder, op.getLoc(), boxWord, size, boxType);
    RuntimeBundle result = RuntimeBundle::objectWithOwnership(
        fieldType, mlir::ValueRange{box},
        ownership::logicalOwnershipKind(fieldType, /*ownsObject=*/false));
    valueBundles[op.getResult()] = std::move(result);
    erase.push_back(op);
    return mlir::success();
  }
  mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> shapes =
      RuntimeBundleLowerer::slotStorageShapesFor(op, fieldType,
                                                 "exception field load");
  if (mlir::failed(shapes))
    return mlir::failure();
  for (mlir::Type shape : *shapes) {
    auto memref = mlir::dyn_cast<mlir::MemRefType>(shape);
    if (!memref || memref.getRank() != 1)
      return op.emitError() << "exception field '" << op.getName() << "' of "
                            << fieldType
                            << " has no boxable value group yet";
  }
  if (shapes->size() > static_cast<std::size_t>(box_abi::kPointerWordCount))
    return op.emitError() << "exception field '" << op.getName()
                          << "' needs more box slots than the payload box has";
  std::optional<RuntimeSymbol> boxWord =
      manifest.primitive("builtins.BaseException", "payload_box_word");
  if (!boxWord)
    return op.emitError()
           << "runtime manifest has no BaseException payload_box_word primitive";

  builder.setInsertionPoint(op);
  mlir::Location loc = op.getLoc();
  mlir::FailureOr<mlir::Value> block =
      RuntimeBundleLowerer::exceptionFieldBlockWord(op, object, classOp);
  if (mlir::failed(block))
    return mlir::failure();
  mlir::Value slot = mlir::arith::ConstantIntOp::create(
      builder, loc, static_cast<std::int64_t>(fieldIndex), 64);
  auto loadWord = [&](std::int64_t word) {
    mlir::Value wordIndex =
        mlir::arith::ConstantIntOp::create(builder, loc, word, 64);
    return RuntimeBundleLowerer::createRuntimeCall(
               loc, *boxWord, mlir::ValueRange{*block, slot, wordIndex})
        .getResult(0);
  };
  llvm::SmallVector<mlir::Value, 4> elementValues;
  for (auto [position, shape] : llvm::enumerate(*shapes)) {
    mlir::Value ptrWord =
        loadWord(box_abi::kPointerWordBase + static_cast<std::int64_t>(position));
    mlir::Value sizeWord =
        loadWord(box_abi::kSizeWordBase + static_cast<std::int64_t>(position));
    elementValues.push_back(RuntimeBundleLowerer::memrefFromBoxWords(
        builder, loc, ptrWord, sizeWord, mlir::cast<mlir::MemRefType>(shape)));
  }
  mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>> canonical =
      RuntimeBundleLowerer::unboxSlotElementValues(op, fieldType, elementValues);
  if (mlir::failed(canonical))
    return mlir::failure();
  RuntimeValue element{fieldType, *canonical,
                       ownership::logicalOwnershipKind(fieldType,
                                                       /*ownsObject=*/false)};
  if (mlir::failed(bindRetainedEvidenceValue(
          op, op.getResult(), "exception field load", element)))
    return mlir::failure();

  // An `int` result must also carry the primitive (value, valid) lane the
  // int ABI pairs with every boxed int: a bundle without it rides the return
  // and call boundaries as "not a valid primitive", and the boxed payload is
  // then ignored in favour of the zero placeholder — a silently wrong value.
  if (RuntimeBundleLowerer::hasPrimitiveI64ABI(fieldType)) {
    std::optional<RuntimeSymbol> unbox =
        manifest.primitive("builtins.int", "unbox.i64");
    if (!unbox)
      return op.emitError()
             << "runtime manifest has no builtins.int unbox.i64 primitive";
    auto bound = valueBundles.find(op.getResult());
    if (bound == valueBundles.end() || bound->second.physicalValues().empty())
      return op.emitError() << "exception field load produced no bundle";
    // Copy before building: createRuntimeCall may declare the callee, and the
    // assignment below can rehash valueBundles, so neither the iterator nor
    // the ArrayRef into the bundle may be held across those steps.
    llvm::SmallVector<mlir::Value, 4> boxedValues(
        bound->second.physicalValues().begin(),
        bound->second.physicalValues().end());
    builder.setInsertionPointAfterValue(boxedValues.back());
    mlir::func::CallOp call =
        RuntimeBundleLowerer::createRuntimeCall(loc, *unbox, boxedValues);
    mlir::Value valid =
        mlir::arith::ConstantIntOp::create(builder, loc, 1, 1).getResult();
    valueBundles[op.getResult()].primitiveI64 =
        RuntimePrimitiveI64Evidence{call.getResult(0), valid};
  }
  return mlir::success();
}

// A field store swaps the slot box's payload in place (retain new, release
// old, overwrite the 16 handle words). The instance bundle is NOT respliced:
// the block is the shared mutable state, and the exception may be a borrowed
// handler binding whose lanes belong to another frame.
mlir::LogicalResult RuntimeBundleLowerer::lowerExceptionFieldAttrSet(
    py::AttrSetOp op, const RuntimeBundle &objectRef,
    const RuntimeBundle &valueRef, py::ClassOp classOp, unsigned fieldIndex) {
  RuntimeBundle object = objectRef;
  RuntimeBundle value = valueRef;
  std::optional<RuntimeSymbol> storeWords =
      manifest.primitive("builtins.BaseException", "payload_store_words");
  std::optional<RuntimeSymbol> releaseSlot =
      manifest.primitive("builtins.BaseException", "payload_release_slot");
  if (!storeWords || !releaseSlot)
    return op.emitError() << "runtime manifest has no BaseException payload "
                             "slot store/release primitives";

  builder.setInsertionPoint(op);
  mlir::Location loc = op.getLoc();
  mlir::FailureOr<RuntimeBundle> payload =
      RuntimeBundleLowerer::materializePayloadObjectBundle(op, value);
  if (mlir::failed(payload))
    return mlir::failure();
  if (mlir::failed(RuntimeBundleLowerer::retainAggregateSlot(op, *payload,
                                                             "exception.field")))
    return mlir::failure();
  mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>> words =
      RuntimeBundleLowerer::objectPayloadHandleWords(op, *payload,
                                                     /*ownsPayload=*/true);
  if (mlir::failed(words))
    return mlir::failure();
  if (words->size() != static_cast<std::size_t>(box_abi::kWordsPerBox))
    return op.emitError() << "exception field store expects "
                          << box_abi::kWordsPerBox << " box words, got "
                          << words->size();

  builder.setInsertionPoint(op);
  mlir::FailureOr<mlir::Value> block =
      RuntimeBundleLowerer::exceptionFieldBlockWord(op, object, classOp);
  if (mlir::failed(block))
    return mlir::failure();
  mlir::Value slot = mlir::arith::ConstantIntOp::create(
      builder, loc, static_cast<std::int64_t>(fieldIndex), 64);
  RuntimeBundleLowerer::createRuntimeCall(loc, *releaseSlot,
                                          mlir::ValueRange{*block, slot});
  llvm::SmallVector<mlir::Value, 20> operands{*block, slot};
  operands.append(words->begin(), words->end());
  RuntimeBundleLowerer::createRuntimeCall(loc, *storeWords, operands);
  // The source keeps its token: the retain above gave the slot its OWN
  // reference. Consuming the source here instead would cancel that retain, so
  // an exception whose message is the same value as a field (the
  // `super().__init__(msg)` + `self.msg = msg` shape) would end with two
  // owners sharing one reference and underflow at teardown.
  erase.push_back(op);
  return mlir::success();
}

} // namespace py::lowering
