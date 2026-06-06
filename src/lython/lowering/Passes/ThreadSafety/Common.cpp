#include "Verifier.h"

#include "Common/LoweringUtils.h"
#include "Common/Object.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace py::threadsafe {

std::optional<::llvm::StringRef> attrs::str(mlir::Operation *op,
                                            ::llvm::StringRef attrName) {
  if (!op)
    return std::nullopt;
  auto attr = op->getAttrOfType<mlir::StringAttr>(attrName);
  if (!attr)
    return std::nullopt;
  return attr.getValue();
}

std::optional<int64_t> attrs::i64(mlir::Operation *op,
                                  ::llvm::StringRef attrName) {
  if (!op)
    return std::nullopt;
  auto attr = op->getAttrOfType<mlir::IntegerAttr>(attrName);
  if (!attr)
    return std::nullopt;
  return attr.getInt();
}

mlir::SmallVector<int64_t> attrs::i64Array(mlir::Operation *op,
                                           ::llvm::StringRef attrName) {
  return lowering::attrs::i64Array(op, attrName);
}

bool constant::memrefInt(mlir::Value value, int64_t expected) {
  if (auto constant = value.getDefiningOp<mlir::arith::ConstantIntOp>())
    return constant.value() == expected;
  if (auto constant = value.getDefiningOp<mlir::arith::ConstantOp>()) {
    auto attr = mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue());
    return attr && attr.getInt() == expected;
  }
  return false;
}

bool constant::llvmInt(mlir::Value value, int64_t expected) {
  if (auto constant = value.getDefiningOp<mlir::LLVM::ConstantOp>()) {
    auto attr = mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue());
    return attr && attr.getInt() == expected;
  }
  return false;
}

bool constant::llvmBoolOrInt(mlir::Value value, int64_t expected) {
  if (auto constant = value.getDefiningOp<mlir::LLVM::ConstantOp>()) {
    mlir::Attribute attr = constant.getValue();
    if (auto boolAttr = mlir::dyn_cast<mlir::BoolAttr>(attr))
      return static_cast<int64_t>(boolAttr.getValue()) == expected;
    if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr))
      return intAttr.getInt() == expected;
  }
  return false;
}

std::optional<int64_t> constant::anyInt(mlir::Value value) {
  if (auto constant = value.getDefiningOp<mlir::arith::ConstantIndexOp>())
    return constant.value();
  if (auto constant = value.getDefiningOp<mlir::arith::ConstantIntOp>())
    return constant.value();
  if (auto constant = value.getDefiningOp<mlir::arith::ConstantOp>()) {
    auto attr = mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue());
    if (attr)
      return attr.getInt();
  }
  if (auto constant = value.getDefiningOp<mlir::LLVM::ConstantOp>()) {
    auto attr = mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue());
    if (attr)
      return attr.getInt();
  }
  return std::nullopt;
}

std::optional<int64_t> constant::index(mlir::Value value) {
  if (auto constant = value.getDefiningOp<mlir::arith::ConstantIndexOp>())
    return constant.value();
  if (auto constant = value.getDefiningOp<mlir::arith::ConstantOp>()) {
    auto attr = mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue());
    if (attr)
      return attr.getInt();
  }
  return std::nullopt;
}

std::optional<int64_t> header_slot::refcount(mlir::Value header) {
  return container::Refcount::slot(header);
}

std::optional<int64_t> header_slot::lock(mlir::Value header) {
  auto component = descriptor::component(header);
  if (component && *component == ContainerSafetyAttrs::kComponentLock)
    return kTypedContainerLockSlot;
  return std::nullopt;
}

bool memref_value::alloca(mlir::Value value) {
  return value.getDefiningOp<mlir::memref::AllocaOp>() != nullptr;
}

bool memref_value::alloc(mlir::Value value) {
  return value.getDefiningOp<mlir::memref::AllocOp>() != nullptr;
}

bool object_header::type(mlir::Type type) {
  return object_abi::Header::isOwned(type) || object_abi::Header::isView(type);
}

bool object_header::runtimeArg(mlir::BlockArgument arg) {
  if (!object_header::type(arg.getType()))
    return false;
  mlir::Operation *parent =
      arg.getOwner() ? arg.getOwner()->getParentOp() : nullptr;
  auto func = mlir::dyn_cast_or_null<mlir::func::FuncOp>(parent);
  if (!func)
    return false;
  if (arg.getArgNumber() >= func.getNumArguments())
    return false;
  return func.getArgAttr(arg.getArgNumber(),
                         OwnershipContractAttrs::kObjectHeader) != nullptr;
}

static bool objectHeaderProvenance(mlir::Value value,
                                   ::llvm::SmallPtrSetImpl<mlir::Value> &seen) {
  if (!value || !seen.insert(value).second ||
      !object_header::type(value.getType()))
    return false;
  if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(value))
    return object_header::runtimeArg(arg);
  mlir::Operation *def = value.getDefiningOp();
  if (!def)
    return false;
  if (def->hasAttr(OwnershipContractAttrs::kObjectHeader))
    return true;
  if (auto cast = mlir::dyn_cast<mlir::memref::CastOp>(def))
    return objectHeaderProvenance(cast.getSource(), seen);
  return false;
}

bool object_header::provenance(mlir::Value value) {
  ::llvm::SmallPtrSet<mlir::Value, 8> seen;
  return objectHeaderProvenance(value, seen);
}

bool local_container::escapeUser(mlir::Operation *op);
bool local_container::use(mlir::Operation *op, mlir::Value value);
mlir::Operation *
local_container::escape(mlir::Value value,
                        ::llvm::SmallPtrSetImpl<mlir::Value> &seen);

bool provenance::gep(mlir::Value pointer) {
  while (true) {
    if (auto bitcast = pointer.getDefiningOp<mlir::LLVM::BitcastOp>()) {
      pointer = bitcast.getArg();
      continue;
    }
    return pointer.getDefiningOp<mlir::LLVM::GEPOp>() != nullptr;
  }
}

mlir::Value pointer::stripCasts(mlir::Value value) {
  while (true) {
    if (auto cast = value.getDefiningOp<mlir::memref::CastOp>()) {
      value = cast.getSource();
      continue;
    }
    if (auto bitcast = value.getDefiningOp<mlir::LLVM::BitcastOp>()) {
      value = bitcast.getArg();
      continue;
    }
    if (auto intToPtr = value.getDefiningOp<mlir::LLVM::IntToPtrOp>()) {
      value = intToPtr.getArg();
      continue;
    }
    if (auto ptrToInt = value.getDefiningOp<mlir::LLVM::PtrToIntOp>()) {
      value = ptrToInt.getArg();
      continue;
    }
    return value;
  }
}

std::optional<::llvm::StringRef> descriptor::group(mlir::Value value) {
  value = pointer::stripCasts(value);
  mlir::Operation *def = value.getDefiningOp();
  if (!def)
    return std::nullopt;
  return attrs::str(def, ContainerSafetyAttrs::kDescriptorGroup);
}

std::optional<::llvm::StringRef> descriptor::kind(mlir::Value value) {
  value = pointer::stripCasts(value);
  mlir::Operation *def = value.getDefiningOp();
  if (!def)
    return std::nullopt;
  return attrs::str(def, ContainerSafetyAttrs::kDescriptorKind);
}

std::optional<::llvm::StringRef> descriptor::component(mlir::Value value) {
  value = pointer::stripCasts(value);
  mlir::Operation *def = value.getDefiningOp();
  if (!def)
    return std::nullopt;
  return attrs::str(def, ContainerSafetyAttrs::kDescriptorComponent);
}

std::optional<::llvm::StringRef> descriptor::Kind::infer(mlir::Value header) {
  header = pointer::stripCasts(header);
  return container::Kind::nameFromHeader(header);
}

std::optional<::llvm::StringRef> descriptor::Kind::get(mlir::Value header) {
  if (auto kind = descriptor::kind(header))
    return kind;
  return infer(header);
}

std::optional<int64_t> header_slot::expectedRefcount(::llvm::StringRef kind) {
  return container::Refcount::slotForKindName(kind);
}

std::optional<int64_t> header_slot::expectedLock(::llvm::StringRef kind) {
  auto parsed = container::Kind::fromName(kind);
  if (!parsed || *parsed == KindId::Tuple)
    return std::nullopt;
  return kTypedContainerLockSlot;
}

std::string resource::group(mlir::Value header) {
  header = pointer::stripCasts(header);
  if (auto group = descriptor::group(header))
    return group->str();
  if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(header))
    return "blockarg." + std::to_string(arg.getArgNumber());
  return "value." + std::to_string(reinterpret_cast<std::uintptr_t>(
                        header.getAsOpaquePointer()));
}

void resource::sealAtomic(mlir::Operation *op, mlir::Value header,
                          int64_t slot) {
  auto kind = descriptor::Kind::get(header);
  if (!kind)
    return;
  threadsafe::memref::Atomic::set(op, ContainerSafetyAttrs::kComponentHeader,
                                  slot, resource::group(header), *kind);
}

void resource::sealAccess(mlir::Operation *op, mlir::Value header,
                          mlir::Value target) {
  if (!op || !header || !target)
    return;
  std::string group = resource::group(header);
  auto component = descriptor::component(target);
  mlir::OpBuilder builder(op->getContext());
  op->setAttr(ContainerSafetyAttrs::kAccessGroup, builder.getStringAttr(group));
  if (component)
    op->setAttr(ContainerSafetyAttrs::kAccessComponent,
                builder.getStringAttr(*component));
}

unsigned descriptor::componentCount(mlir::Value header) {
  header = pointer::stripCasts(header);
  return container::Descriptor::componentCount(header);
}

bool descriptor::siblingIndex(mlir::Value header, mlir::Value component) {
  header = pointer::stripCasts(header);
  component = pointer::stripCasts(component);
  unsigned count = descriptor::componentCount(header);
  if (count <= 1)
    return false;

  if (auto headerArg = mlir::dyn_cast<mlir::BlockArgument>(header)) {
    auto componentArg = mlir::dyn_cast<mlir::BlockArgument>(component);
    if (!componentArg || headerArg.getOwner() != componentArg.getOwner())
      return false;
    unsigned headerNo = headerArg.getArgNumber();
    unsigned componentNo = componentArg.getArgNumber();
    return componentNo >= headerNo && componentNo < headerNo + count;
  }

  if (auto headerResult = mlir::dyn_cast<mlir::OpResult>(header)) {
    auto componentResult = mlir::dyn_cast<mlir::OpResult>(component);
    if (!componentResult ||
        headerResult.getOwner() != componentResult.getOwner())
      return false;
    unsigned headerNo = headerResult.getResultNumber();
    unsigned componentNo = componentResult.getResultNumber();
    return componentNo >= headerNo && componentNo < headerNo + count;
  }

  return false;
}

bool descriptor::sameResource(mlir::Value header, mlir::Value component) {
  header = pointer::stripCasts(header);
  component = pointer::stripCasts(component);
  if (!header || !component)
    return false;
  if (header == component)
    return true;

  std::optional<::llvm::StringRef> headerGroup = descriptor::group(header);
  std::optional<::llvm::StringRef> componentGroup =
      descriptor::group(component);
  if (headerGroup && componentGroup)
    return *headerGroup == *componentGroup;

  return descriptor::siblingIndex(header, component);
}

mlir::Value descriptor::headerSibling(mlir::Value value) {
  value = pointer::stripCasts(value);
  if (!value)
    return {};
  if (descriptor::Kind::get(value))
    return value;

  if (auto result = mlir::dyn_cast<mlir::OpResult>(value)) {
    mlir::Operation *owner = result.getOwner();
    for (mlir::Value candidate : owner->getResults())
      if (descriptor::Kind::get(candidate) &&
          descriptor::sameResource(candidate, value))
        return candidate;
  }

  if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    mlir::Block *owner = arg.getOwner();
    for (mlir::BlockArgument candidate : owner->getArguments())
      if (descriptor::Kind::get(candidate) &&
          descriptor::sameResource(candidate, value))
        return candidate;
  }

  return {};
}

bool descriptor::value(mlir::Value value) {
  return static_cast<bool>(descriptor::headerSibling(value));
}

bool function_arg::hasAttr(mlir::Value value, ::llvm::StringRef attrName);

bool value_type::llvmPointer(mlir::Type type) {
  return mlir::isa<mlir::LLVM::LLVMPointerType>(type);
}

bool function_arg::loweredRank1MemRefDescriptor(mlir::BlockArgument arg,
                                                unsigned componentIndex) {
  mlir::Block *entry = arg.getOwner();
  if (!entry)
    return false;
  mlir::Operation *parent = entry->getParentOp();
  auto function = mlir::dyn_cast_or_null<mlir::FunctionOpInterface>(parent);
  if (!function || parent->getNumRegions() == 0 ||
      parent->getRegion(0).empty() || entry != &parent->getRegion(0).front())
    return false;

  unsigned argIndex = arg.getArgNumber();
  if (argIndex < componentIndex)
    return false;
  unsigned base = argIndex - componentIndex;
  if (base + 4 >= entry->getNumArguments())
    return false;

  return value_type::llvmPointer(entry->getArgument(base).getType()) &&
         value_type::llvmPointer(entry->getArgument(base + 1).getType()) &&
         entry->getArgument(base + 2).getType().isInteger(64) &&
         entry->getArgument(base + 3).getType().isInteger(64) &&
         entry->getArgument(base + 4).getType().isInteger(64);
}

bool provenance::descriptorData(mlir::Value ptr) {
  ptr = pointer::stripCasts(ptr);
  if (mlir::Operation *def = ptr.getDefiningOp())
    if (def->hasAttr(ContainerSafetyAttrs::kDescriptorData))
      return true;
  if (ptr.getDefiningOp<mlir::LLVM::AllocaOp>())
    return true;
  if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(ptr))
    return function_arg::loweredRank1MemRefDescriptor(arg,
                                                      /*componentIndex=*/1);
  if (auto extract = ptr.getDefiningOp<mlir::LLVM::ExtractValueOp>())
    return extract.getPosition() == ::llvm::ArrayRef<int64_t>{1};
  auto gep = ptr.getDefiningOp<mlir::LLVM::GEPOp>();
  if (!gep)
    return false;
  mlir::Value base = pointer::stripCasts(gep.getBase());
  if (mlir::Operation *def = base.getDefiningOp())
    if (def->hasAttr(ContainerSafetyAttrs::kDescriptorData))
      return true;
  if (base.getDefiningOp<mlir::LLVM::AllocaOp>())
    return true;
  if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(base))
    return function_arg::loweredRank1MemRefDescriptor(arg,
                                                      /*componentIndex=*/1);
  auto extract = base.getDefiningOp<mlir::LLVM::ExtractValueOp>();
  if (!extract)
    return false;
  return extract.getPosition() == ::llvm::ArrayRef<int64_t>{1};
}

bool provenance::descriptorAllocated(mlir::Value ptr) {
  ptr = pointer::stripCasts(ptr);
  if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(ptr))
    return function_arg::loweredRank1MemRefDescriptor(arg,
                                                      /*componentIndex=*/0);
  auto extract = ptr.getDefiningOp<mlir::LLVM::ExtractValueOp>();
  if (!extract)
    return false;
  return extract.getPosition() == ::llvm::ArrayRef<int64_t>{0};
}

bool provenance::asyncExceptionCellAllocated(mlir::Value value) {
  value = pointer::stripCasts(value);
  if (provenance::descriptorAllocated(value))
    return true;
  if (mlir::Operation *def = value.getDefiningOp())
    return def->hasAttr(AsyncSafetyAttrs::kExceptionCellAllocated);
  return false;
}

bool function_arg::hasAttr(mlir::Value value, ::llvm::StringRef attrName) {
  auto arg = mlir::dyn_cast<mlir::BlockArgument>(value);
  if (!arg)
    return false;

  mlir::Operation *parent = arg.getOwner()->getParentOp();
  if (!parent || parent->getNumRegions() == 0)
    return false;

  mlir::Region &entryRegion = parent->getRegion(0);
  if (entryRegion.empty() || arg.getOwner() != &entryRegion.front())
    return false;

  auto function = mlir::dyn_cast<mlir::FunctionOpInterface>(parent);
  if (!function || arg.getArgNumber() >= function.getNumArguments())
    return false;
  return function.getArgAttr(arg.getArgNumber(), attrName) != nullptr;
}

bool provenance::entryArgRoot(mlir::Value value) {
  value = pointer::stripCasts(value);
  if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    mlir::Operation *parent = arg.getOwner()->getParentOp();
    auto function = mlir::dyn_cast_or_null<mlir::FunctionOpInterface>(parent);
    return function && parent->getNumRegions() != 0 &&
           !parent->getRegion(0).empty() &&
           arg.getOwner() == &parent->getRegion(0).front();
  }
  if (auto gep = value.getDefiningOp<mlir::LLVM::GEPOp>())
    return provenance::entryArgRoot(gep.getBase());
  if (auto extract = value.getDefiningOp<mlir::LLVM::ExtractValueOp>())
    return provenance::entryArgRoot(extract.getContainer());
  return false;
}

mlir::Value pointer::gepRoot(mlir::Value ptr) {
  ptr = pointer::stripCasts(ptr);
  while (auto gep = ptr.getDefiningOp<mlir::LLVM::GEPOp>())
    ptr = pointer::stripCasts(gep.getBase());
  return ptr;
}

mlir::Value atomic::memrefHeader(mlir::Operation *op) {
  if (auto atomic = mlir::dyn_cast<mlir::memref::AtomicRMWOp>(op))
    return pointer::stripCasts(atomic.getMemref());
  return {};
}

mlir::Value atomic::llvmPointer(mlir::Operation *op) {
  if (auto atomic = mlir::dyn_cast<mlir::LLVM::AtomicRMWOp>(op))
    return atomic.getPtr();
  if (auto store = mlir::dyn_cast<mlir::LLVM::StoreOp>(op))
    return store.getAddr();
  return {};
}

mlir::Value atomic::llvmRoot(mlir::Operation *op) {
  mlir::Value ptr = atomic::llvmPointer(op);
  if (!ptr)
    return {};
  return pointer::gepRoot(ptr);
}

bool constant::llvmNullPtr(mlir::Value value) {
  return value && value.getDefiningOp<mlir::LLVM::ZeroOp>() != nullptr;
}

mlir::LogicalResult
verifier::llvm::Ordering::verify(mlir::LLVM::AtomicRMWOp op,
                                 mlir::LLVM::AtomicOrdering actualOrdering) {
  auto orderingAttr =
      attrs::str(op.getOperation(), ThreadSafetyAttrs::kAtomicOrdering);
  if (!orderingAttr)
    return op->emitOpError(
        "LLVM atomic with Lython role is missing ly.atomic.ordering");

  if (*orderingAttr == ThreadSafetyAttrs::kOrderingMonotonic) {
    if (actualOrdering == mlir::LLVM::AtomicOrdering::monotonic ||
        actualOrdering == mlir::LLVM::AtomicOrdering::acquire ||
        actualOrdering == mlir::LLVM::AtomicOrdering::acq_rel ||
        actualOrdering == mlir::LLVM::AtomicOrdering::seq_cst)
      return mlir::success();
    return op->emitOpError(
        "LLVM atomic ordering is weaker than monotonic contract");
  }

  if (*orderingAttr == ThreadSafetyAttrs::kOrderingAcquire) {
    if (ordering::atLeastAcquire(actualOrdering))
      return mlir::success();
    return op->emitOpError(
        "LLVM atomic ordering is weaker than acquire contract");
  }

  if (*orderingAttr == ThreadSafetyAttrs::kOrderingRelease) {
    if (ordering::atLeastRelease(actualOrdering))
      return mlir::success();
    return op->emitOpError(
        "LLVM atomic ordering is weaker than release contract");
  }

  if (*orderingAttr == ThreadSafetyAttrs::kOrderingAcqRel) {
    if (ordering::atLeastAcqRel(actualOrdering))
      return mlir::success();
    return op->emitOpError(
        "LLVM atomic ordering is weaker than acq_rel contract");
  }

  if (*orderingAttr == ThreadSafetyAttrs::kOrderingSeqCst) {
    if (actualOrdering == mlir::LLVM::AtomicOrdering::seq_cst)
      return mlir::success();
    return op->emitOpError(
        "LLVM atomic ordering is weaker than seq_cst contract");
  }

  return op->emitOpError("unsupported LLVM atomic ordering contract: ")
         << *orderingAttr;
}

bool compare::llvmZero(mlir::LLVM::ICmpOp cmp, mlir::Value value,
                       mlir::LLVM::ICmpPredicate expected) {
  if (cmp.getPredicate() != expected)
    return false;
  return (cmp.getLhs() == value && constant::llvmInt(cmp.getRhs(), 0)) ||
         (cmp.getRhs() == value && constant::llvmInt(cmp.getLhs(), 0));
}

bool control::noReturn(mlir::Operation *terminator) {
  if (!mlir::isa_and_nonnull<mlir::LLVM::UnreachableOp>(terminator))
    return false;
  auto call =
      mlir::dyn_cast_or_null<mlir::LLVM::CallOp>(terminator->getPrevNode());
  if (!call)
    return false;
  return call->hasAttr(ControlFlowContractAttrs::kNoReturn);
}

} // namespace py::threadsafe
