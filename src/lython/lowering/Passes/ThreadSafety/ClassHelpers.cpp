#include "Verifier.h"

namespace py::threadsafe {

namespace class_field {

namespace field_load {

static bool aggregateSlot(mlir::LLVM::LoadOp load) {
  if (!load || !load->hasAttr(OwnershipContractAttrs::kAggregateSlotLoad))
    return false;
  auto slot = attrs::i64(load.getOperation(),
                         OwnershipContractAttrs::kAggregateSlotIndex);
  return slot.has_value();
}

} // namespace field_load

static std::optional<int64_t> slot(mlir::Value value) {
  value = pointer::stripCasts(value);
  auto load = value.getDefiningOp<mlir::LLVM::LoadOp>();
  if (!field_load::aggregateSlot(load))
    return std::nullopt;
  return attrs::i64(load.getOperation(),
                    OwnershipContractAttrs::kAggregateSlotIndex);
}

struct Slots {
  static ::llvm::DenseMap<int64_t, unsigned> calls(mlir::LLVM::LLVMFuncOp fn,
                                                   ::llvm::StringRef attrName) {
    ::llvm::DenseMap<int64_t, unsigned> counts;
    fn.walk([&](mlir::LLVM::CallOp call) {
      if (!call->hasAttr(attrName) || call.getNumOperands() == 0)
        return;
      if (auto fieldSlot = slot(call.getOperand(0)))
        ++counts[*fieldSlot];
    });
    return counts;
  }

  static ::llvm::DenseMap<int64_t, unsigned> loads(mlir::LLVM::LLVMFuncOp fn) {
    ::llvm::DenseMap<int64_t, unsigned> counts;
    fn.walk([&](mlir::LLVM::LoadOp load) {
      if (!field_load::aggregateSlot(load))
        return;
      auto fieldSlot = attrs::i64(load.getOperation(),
                                  OwnershipContractAttrs::kAggregateSlotIndex);
      if (fieldSlot)
        ++counts[*fieldSlot];
    });
    return counts;
  }
};

static std::optional<int64_t>
slotInProvenance(mlir::Value value,
                 ::llvm::SmallPtrSetImpl<mlir::Value> &seen) {
  value = pointer::stripCasts(value);
  if (!value || !seen.insert(value).second)
    return std::nullopt;

  if (auto load = value.getDefiningOp<mlir::LLVM::LoadOp>())
    if (field_load::aggregateSlot(load))
      return attrs::i64(load.getOperation(),
                        OwnershipContractAttrs::kAggregateSlotIndex);
  if (auto load = value.getDefiningOp<mlir::memref::LoadOp>())
    if (load->hasAttr(OwnershipContractAttrs::kAggregateSlotLoad))
      return attrs::i64(load.getOperation(),
                        OwnershipContractAttrs::kAggregateSlotIndex);
  if (auto gep = value.getDefiningOp<mlir::LLVM::GEPOp>())
    return slotInProvenance(gep.getBase(), seen);
  if (auto extract = value.getDefiningOp<mlir::LLVM::ExtractValueOp>())
    return slotInProvenance(extract.getContainer(), seen);
  return std::nullopt;
}

} // namespace class_field

namespace release {

static mlir::Value target(mlir::Operation *op) {
  if (auto atomic = mlir::dyn_cast<mlir::memref::AtomicRMWOp>(op))
    return atomic.getMemref();
  if (auto atomic = mlir::dyn_cast<mlir::LLVM::AtomicRMWOp>(op))
    return atomic.getPtr();
  return {};
}

} // namespace release

namespace slot_counts {

using Map = ::llvm::DenseMap<int64_t, unsigned>;

static Map expected(::llvm::ArrayRef<int64_t> slots, unsigned countPerSlot) {
  Map result;
  for (int64_t slot : slots)
    result[slot] += countPerSlot;
  return result;
}

static void add(Map &expected, ::llvm::ArrayRef<int64_t> slots,
                unsigned countPerSlot) {
  for (int64_t slot : slots)
    expected[slot] += countPerSlot;
}

static mlir::LogicalResult exact(mlir::LLVM::LLVMFuncOp fn, const Map &got,
                                 const Map &expected,
                                 ::llvm::StringRef description) {
  for (auto [slot, count] : got) {
    auto it = expected.find(slot);
    unsigned expectedCount = it == expected.end() ? 0 : it->second;
    if (count != expectedCount)
      return fn.emitOpError()
             << description << " count for field slot " << slot << " is "
             << count << ", expected " << expectedCount;
  }
  for (auto [slot, expectedCount] : expected) {
    unsigned count = got.lookup(slot);
    if (count != expectedCount)
      return fn.emitOpError()
             << description << " count for field slot " << slot << " is "
             << count << ", expected " << expectedCount;
  }
  return mlir::success();
}

} // namespace slot_counts

namespace container_field {

struct Slots {
  static slot_counts::Map releases(mlir::LLVM::LLVMFuncOp fn) {
    slot_counts::Map counts;
    fn.walk([&](mlir::Operation *op) {
      auto role = attrs::str(op, ThreadSafetyAttrs::kAtomicRole);
      if (!role || *role != ThreadSafetyAttrs::kRoleContainerRefcountRelease)
        return;
      ::llvm::SmallPtrSet<mlir::Value, 8> seen;
      auto slot = class_field::slotInProvenance(release::target(op), seen);
      counts[slot.value_or(-1)]++;
    });
    return counts;
  }
};

} // namespace container_field

namespace schema {

struct Fields {
  mlir::SmallVector<int64_t, 8> direct;
  mlir::SmallVector<int64_t, 8> containers;
};

static mlir::LogicalResult counts(mlir::LLVM::LLVMFuncOp fn,
                                  int64_t &directFields,
                                  int64_t &containerFields) {
  auto direct = attrs::i64(fn.getOperation(),
                           ClassSafetyAttrs::kHelperDirectRefcountFields);
  auto containers =
      attrs::i64(fn.getOperation(), ClassSafetyAttrs::kHelperContainerFields);
  if (!direct || !containers)
    return fn.emitOpError("class helper is missing field schema metadata");
  directFields = *direct;
  containerFields = *containers;
  return mlir::success();
}

static mlir::LogicalResult read(mlir::LLVM::LLVMFuncOp fn, Fields &fields) {
  int64_t directCount = 0;
  int64_t containerCount = 0;
  if (mlir::failed(counts(fn, directCount, containerCount)))
    return mlir::failure();
  mlir::SmallVector<int64_t> directAttrs = attrs::i64Array(
      fn.getOperation(), ClassSafetyAttrs::kHelperDirectRefcountFieldIndices);
  mlir::SmallVector<int64_t> containerAttrs = attrs::i64Array(
      fn.getOperation(), ClassSafetyAttrs::kHelperContainerFieldIndices);
  fields.direct.append(directAttrs.begin(), directAttrs.end());
  fields.containers.append(containerAttrs.begin(), containerAttrs.end());
  if (fields.direct.size() != static_cast<size_t>(directCount))
    return fn.emitOpError("class helper direct field index schema does not "
                          "match direct field count");
  if (fields.containers.size() != static_cast<size_t>(containerCount))
    return fn.emitOpError("class helper container field index schema does not "
                          "match container field count");
  return mlir::success();
}

} // namespace schema

mlir::LogicalResult
verifier::class_helper::Incref::verify(mlir::LLVM::LLVMFuncOp fn) {
  auto helperKind =
      attrs::str(fn.getOperation(), ClassSafetyAttrs::kHelperKind);
  auto helperClass =
      attrs::str(fn.getOperation(), ClassSafetyAttrs::kHelperClass);
  if (!helperKind) {
    if (fn.getName().starts_with("__ly_class_incref_"))
      return fn.emitOpError("class incref helper is missing helper metadata");
    return mlir::success();
  }
  if (*helperKind != ClassSafetyAttrs::kKindIncref)
    return mlir::success();
  if (!helperClass)
    return fn.emitOpError("class incref helper is missing class metadata");
  if (fn.isDeclaration())
    return fn.emitOpError("class incref helper contract has no body to verify");
  schema::Fields fields;
  if (mlir::failed(schema::read(fn, fields)))
    return mlir::failure();
  auto &directFields = fields.direct;

  unsigned verifiedRetains = 0;
  fn.walk([&](mlir::Operation *op) {
    auto role = attrs::str(op, ThreadSafetyAttrs::kAtomicRole);
    if (role && *role == ThreadSafetyAttrs::kRoleClassRefcountRetain &&
        op->hasAttr(ThreadSafetyAttrs::kOwnedTokenVerified))
      ++verifiedRetains;
  });
  if (verifiedRetains != 1)
    return fn.emitOpError("class incref helper must contain exactly one "
                          "verified class refcount retain");

  if (mlir::failed(
          slot_counts::exact(fn,
                             class_field::Slots::calls(
                                 fn, OwnershipContractAttrs::kAggregateRetain),
                             slot_counts::expected(directFields, 1),
                             "class incref helper direct field retain")))
    return mlir::failure();

  if (mlir::failed(slot_counts::exact(fn, class_field::Slots::loads(fn),
                                      slot_counts::expected(directFields, 1),
                                      "class incref helper direct field load")))
    return mlir::failure();
  if (mlir::failed(slot_counts::exact(
          fn, container_field::Slots::releases(fn), slot_counts::Map{},
          "class incref helper container field release")))
    return mlir::failure();
  return mlir::success();
}

static mlir::LLVM::AtomicRMWOp
getClassRefcountReleaseAtomic(mlir::Value value) {
  auto atomic = value.getDefiningOp<mlir::LLVM::AtomicRMWOp>();
  if (!atomic)
    return {};
  auto role = attrs::str(atomic.getOperation(), ThreadSafetyAttrs::kAtomicRole);
  if (role && *role == ThreadSafetyAttrs::kRoleClassRefcountRelease)
    return atomic;
  return {};
}

namespace class_release {

static bool toZero(mlir::Value condition) {
  auto cmp = condition.getDefiningOp<mlir::LLVM::ICmpOp>();
  if (!cmp || cmp.getPredicate() != mlir::LLVM::ICmpPredicate::eq)
    return false;

  auto matches = [&](mlir::Value lhs, mlir::Value rhs) {
    auto release = getClassRefcountReleaseAtomic(lhs);
    return release && constant::llvmInt(rhs, 1);
  };
  return matches(cmp.getLhs(), cmp.getRhs()) ||
         matches(cmp.getRhs(), cmp.getLhs());
}

static bool memoryFree(mlir::LLVM::CallOp call) {
  auto callee = call.getCallee();
  return callee && (*callee == RuntimeSymbols::kMemFree || *callee == "free");
}

static bool storageFree(mlir::LLVM::LLVMFuncOp fn, mlir::LLVM::CallOp call) {
  if (!memoryFree(call) || fn.getBody().empty() || call.getNumOperands() < 1)
    return false;
  mlir::Block &entry = fn.getBody().front();
  if (entry.getNumArguments() < 1)
    return false;
  return pointer::stripCasts(call.getOperand(0)) == entry.getArgument(0);
}

static bool guardsFree(mlir::LLVM::LLVMFuncOp fn, mlir::LLVM::CallOp call) {
  mlir::DominanceInfo dominance(fn.getOperation());
  bool guarded = false;
  fn.walk([&](mlir::Operation *terminator) {
    if (guarded || terminator->getNumSuccessors() < 2)
      return;
    mlir::Value condition = control::condition(terminator);
    if (!condition || !toZero(condition))
      return;
    mlir::Block *destroySuccessor = terminator->getSuccessor(0);
    guarded = ::py::threadsafe::dominance::block(
        destroySuccessor, call.getOperation(), dominance);
  });
  return guarded;
}

} // namespace class_release

mlir::LogicalResult
verifier::class_helper::Decref::verify(mlir::LLVM::LLVMFuncOp fn) {
  auto helperKind =
      attrs::str(fn.getOperation(), ClassSafetyAttrs::kHelperKind);
  auto helperClass =
      attrs::str(fn.getOperation(), ClassSafetyAttrs::kHelperClass);
  if (!helperKind) {
    if (fn.getName().starts_with("__ly_class_decref_"))
      return fn.emitOpError("class decref helper is missing helper metadata");
    return mlir::success();
  }
  if (*helperKind != ClassSafetyAttrs::kKindDecref)
    return mlir::success();
  if (!helperClass)
    return fn.emitOpError("class decref helper is missing class metadata");
  if (fn.isDeclaration())
    return fn.emitOpError("class decref helper contract has no body to verify");
  schema::Fields fields;
  if (mlir::failed(schema::read(fn, fields)))
    return mlir::failure();
  auto &directFields = fields.direct;
  auto &containerFields = fields.containers;

  unsigned releases = 0;
  bool failedAny = false;
  fn.walk([&](mlir::Operation *op) {
    auto role = attrs::str(op, ThreadSafetyAttrs::kAtomicRole);
    if (role && *role == ThreadSafetyAttrs::kRoleClassRefcountRelease)
      ++releases;
    if (auto call = mlir::dyn_cast<mlir::LLVM::CallOp>(op)) {
      if (!class_release::storageFree(fn, call))
        return;
      if (!class_release::guardsFree(fn, call)) {
        call.emitOpError("class memory free is not guarded by class refcount "
                         "release returning 1");
        failedAny = true;
      }
    }
  });
  if (releases != 1)
    return fn.emitOpError("class decref helper must contain exactly one class "
                          "refcount release");
  if (mlir::failed(
          slot_counts::exact(fn,
                             class_field::Slots::calls(
                                 fn, OwnershipContractAttrs::kAggregateRelease),
                             slot_counts::expected(directFields, 2),
                             "class decref helper direct field release")))
    return mlir::failure();

  slot_counts::Map expectedLoads = slot_counts::expected(directFields, 2);
  slot_counts::add(expectedLoads, containerFields, 1);
  if (mlir::failed(slot_counts::exact(fn, class_field::Slots::loads(fn),
                                      expectedLoads,
                                      "class decref helper field load")))
    return mlir::failure();

  if (mlir::failed(
          slot_counts::exact(fn, container_field::Slots::releases(fn),
                             slot_counts::expected(containerFields, 1),
                             "class decref helper container field release")))
    return mlir::failure();
  return mlir::failure(failedAny);
}

mlir::LogicalResult
verifier::class_helper::DestroyLocal::verify(mlir::LLVM::LLVMFuncOp fn) {
  auto helperKind =
      attrs::str(fn.getOperation(), ClassSafetyAttrs::kHelperKind);
  if (!helperKind) {
    if (fn.getName().starts_with("__ly_class_destroy_local_"))
      return fn.emitOpError(
          "class destroy_local helper is missing helper metadata");
    return mlir::success();
  }
  if (*helperKind != ClassSafetyAttrs::kKindDestroyLocal)
    return mlir::success();

  auto helperClass =
      attrs::str(fn.getOperation(), ClassSafetyAttrs::kHelperClass);
  if (!helperClass)
    return fn.emitOpError(
        "class destroy_local helper is missing class metadata");
  if (fn.isDeclaration())
    return fn.emitOpError(
        "class destroy_local helper contract has no body to verify");
  schema::Fields fields;
  if (mlir::failed(schema::read(fn, fields)))
    return mlir::failure();
  auto &directFields = fields.direct;
  auto &containerFields = fields.containers;

  bool failedAny = false;
  fn.walk([&](mlir::Operation *op) {
    auto role = attrs::str(op, ThreadSafetyAttrs::kAtomicRole);
    if (role && *role == ThreadSafetyAttrs::kRoleClassRefcountRelease) {
      op->emitOpError("destroy_local helper must not mutate managed class "
                      "refcount");
      failedAny = true;
    }
    if (auto call = mlir::dyn_cast<mlir::LLVM::CallOp>(op)) {
      if (class_release::storageFree(fn, call)) {
        call.emitOpError("destroy_local helper must not free stack/local class "
                         "storage");
        failedAny = true;
      }
    }
  });
  if (mlir::failed(slot_counts::exact(
          fn,
          class_field::Slots::calls(fn,
                                    OwnershipContractAttrs::kAggregateRelease),
          slot_counts::expected(directFields, 1),
          "class destroy_local helper direct field release")))
    return mlir::failure();
  slot_counts::Map expectedLoads = slot_counts::expected(directFields, 1);
  slot_counts::add(expectedLoads, containerFields, 1);
  if (mlir::failed(slot_counts::exact(fn, class_field::Slots::loads(fn),
                                      expectedLoads,
                                      "class destroy_local helper field load")))
    return mlir::failure();
  if (mlir::failed(slot_counts::exact(
          fn, container_field::Slots::releases(fn),
          slot_counts::expected(containerFields, 1),
          "class destroy_local helper container field release")))
    return mlir::failure();
  return mlir::failure(failedAny);
}

namespace derived {

static bool from(mlir::Value value, mlir::Value root) {
  value = pointer::stripCasts(value);
  root = pointer::stripCasts(root);
  if (!value || !root)
    return false;
  if (value == root)
    return true;
  if (auto gep = value.getDefiningOp<mlir::LLVM::GEPOp>())
    return from(gep.getBase(), root);
  return false;
}

static bool fromAny(mlir::Value value, ::llvm::ArrayRef<mlir::Value> roots) {
  return ::llvm::any_of(roots,
                        [&](mlir::Value root) { return from(value, root); });
}

} // namespace derived

namespace promote {

static mlir::SmallVector<mlir::Value> freshObjects(mlir::LLVM::LLVMFuncOp fn) {
  mlir::SmallVector<mlir::Value> roots;
  fn.walk([&](mlir::LLVM::CallOp call) {
    if (!call->hasAttr(ClassSafetyAttrs::kPromoteFreshObject))
      return;
    if (call.getNumResults() == 1)
      roots.push_back(call.getResult());
  });
  return roots;
}

static mlir::LogicalResult init(mlir::LLVM::LLVMFuncOp fn,
                                ::llvm::ArrayRef<mlir::Value> freshObjects,
                                ::llvm::StringRef attrName,
                                int64_t expectedValue,
                                ::llvm::StringRef description) {
  unsigned count = 0;
  bool failedAny = false;
  fn.walk(
      [&](mlir::LLVM::StoreOp store) {
        if (!store->hasAttr(attrName))
          return;
        ++count;
        if (!constant::llvmBoolOrInt(store.getValue(), expectedValue)) {
          store->emitOpError()
              << description << " store writes unexpected initial value";
          failedAny = true;
        }
        if (!derived::fromAny(store.getAddr(), freshObjects)) {
          store->emitOpError()
              << description
              << " store does not target the fresh promoted object";
          failedAny = true;
        }
      });
  if (count != 1)
    return fn.emitOpError() << "class promote helper must contain exactly one "
                            << description << " store, found " << count;
  return mlir::failure(failedAny);
}

static mlir::LogicalResult returns(mlir::LLVM::LLVMFuncOp fn,
                                   ::llvm::ArrayRef<mlir::Value> freshObjects) {
  if (fn.getBody().empty())
    return mlir::success();
  mlir::Block &entry = fn.getBody().front();
  mlir::Value entryObject =
      entry.getNumArguments() ? entry.getArgument(0) : mlir::Value();
  bool failedAny = false;
  fn.walk([&](mlir::LLVM::ReturnOp ret) {
    for (mlir::Value operand : ret.getOperands()) {
      if (!mlir::isa<mlir::LLVM::LLVMPointerType>(operand.getType()))
        continue;
      if (operand.getDefiningOp<mlir::LLVM::ZeroOp>())
        continue;
      if (entryObject && derived::from(operand, entryObject))
        continue;
      if (derived::fromAny(operand, freshObjects))
        continue;
      ret->emitOpError("class promote helper returns a pointer that is neither "
                       "null, the retained entry object, nor the fresh managed "
                       "object");
      failedAny = true;
    }
  });
  return mlir::failure(failedAny);
}

static unsigned containerInits(mlir::LLVM::LLVMFuncOp fn) {
  unsigned count = 0;
  fn.walk([&](mlir::LLVM::StoreOp store) {
    auto init = store->getAttrOfType<mlir::IntegerAttr>(
        ContainerSafetyAttrs::kRefcountInit);
    if (!init || init.getInt() != 1)
      return;
    auto state =
        attrs::str(store.getOperation(), ContainerSafetyAttrs::kRefcountState);
    if (state && *state == ContainerSafetyAttrs::kStateManaged)
      ++count;
  });
  return count;
}

static unsigned retainCalls(mlir::LLVM::LLVMFuncOp fn) {
  unsigned count = 0;
  fn.walk([&](mlir::LLVM::CallOp call) {
    auto callee = call.getCallee();
    if (!callee || !callee->starts_with("__ly_class_incref_"))
      return;
    ++count;
  });
  return count;
}

} // namespace promote

mlir::LogicalResult
verifier::class_helper::Promote::verify(mlir::LLVM::LLVMFuncOp fn) {
  auto helperKind =
      attrs::str(fn.getOperation(), ClassSafetyAttrs::kHelperKind);
  auto helperClass =
      attrs::str(fn.getOperation(), ClassSafetyAttrs::kHelperClass);
  if (!helperKind) {
    if (fn.getName().starts_with("__ly_class_promote_"))
      return fn.emitOpError("class promote helper is missing helper metadata");
    return mlir::success();
  }
  if (*helperKind != ClassSafetyAttrs::kKindPromote)
    return mlir::success();
  if (!helperClass)
    return fn.emitOpError("class promote helper is missing class metadata");
  if (fn.isDeclaration())
    return fn.emitOpError(
        "class promote helper contract has no body to verify");

  schema::Fields fields;
  if (mlir::failed(schema::read(fn, fields)))
    return mlir::failure();
  auto &directFields = fields.direct;
  auto &containerFields = fields.containers;

  mlir::SmallVector<mlir::Value> freshObjects = promote::freshObjects(fn);
  if (freshObjects.size() != 1)
    return fn.emitOpError("class promote helper must mark exactly one fresh "
                          "managed object allocation");

  if (mlir::failed(promote::init(fn, freshObjects,
                                 ClassSafetyAttrs::kPromoteManagedInit, 1,
                                 "managed flag initialization")))
    return mlir::failure();
  if (mlir::failed(promote::init(fn, freshObjects,
                                 ClassSafetyAttrs::kPromoteLockInit, 0,
                                 "lock initialization")))
    return mlir::failure();
  if (mlir::failed(promote::init(fn, freshObjects,
                                 ClassSafetyAttrs::kPromoteRefcountInit, 1,
                                 "class refcount initialization")))
    return mlir::failure();

  if (promote::retainCalls(fn) != 1)
    return fn.emitOpError(
        "class promote helper must retain the already-managed "
        "entry object on exactly one branch");

  if (mlir::failed(promote::returns(fn, freshObjects)))
    return mlir::failure();

  if (mlir::failed(
          slot_counts::exact(fn,
                             class_field::Slots::calls(
                                 fn, OwnershipContractAttrs::kAggregateRetain),
                             slot_counts::expected(directFields, 1),
                             "class promote helper direct field retain")))
    return mlir::failure();
  if (mlir::failed(
          slot_counts::exact(fn, class_field::Slots::loads(fn),
                             slot_counts::expected(directFields, 1),
                             "class promote helper direct field load")))
    return mlir::failure();
  if (mlir::failed(slot_counts::exact(
          fn,
          class_field::Slots::calls(fn,
                                    OwnershipContractAttrs::kAggregateRelease),
          slot_counts::Map{}, "class promote helper direct field release")))
    return mlir::failure();
  if (mlir::failed(slot_counts::exact(
          fn, container_field::Slots::releases(fn), slot_counts::Map{},
          "class promote helper container field release")))
    return mlir::failure();

  unsigned containerInits = promote::containerInits(fn);
  if (containerInits != containerFields.size())
    return fn.emitOpError()
           << "class promote helper managed container refcount initialization "
              "count is "
           << containerInits << ", expected " << containerFields.size();
  return mlir::success();
}

} // namespace py::threadsafe
