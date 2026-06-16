#include "Verifier.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

namespace py::threadsafe {

namespace class_field {

namespace field_load {

static bool aggregateSlot(mlir::Operation *op) {
  if (!op || !op->hasAttr(OwnershipContractAttrs::kAggregateSlotLoad))
    return false;
  auto group = attrs::str(op, OwnershipContractAttrs::kAggregateSlotGroup);
  if (!group || *group != "class.field")
    return false;
  auto slot = attrs::i64(op, OwnershipContractAttrs::kAggregateSlotIndex);
  return slot.has_value();
}

} // namespace field_load

static std::optional<int64_t>
slotInProvenance(mlir::Value value, ::llvm::SmallPtrSetImpl<mlir::Value> &seen);

static bool hasHelperKind(mlir::LLVM::CallOp call,
                          ::llvm::StringRef expectedKind) {
  auto helperKind =
      attrs::str(call.getOperation(), ClassSafetyAttrs::kHelperKind);
  return helperKind && *helperKind == expectedKind;
}

struct Slots {
  static ::llvm::DenseMap<int64_t, unsigned> calls(mlir::LLVM::LLVMFuncOp fn,
                                                   ::llvm::StringRef attrName) {
    ::llvm::DenseMap<int64_t, unsigned> counts;
    fn.walk([&](mlir::LLVM::CallOp call) {
      if (!call->hasAttr(attrName) || call.getNumOperands() == 0)
        return;
      ::llvm::SmallPtrSet<mlir::Value, 8> seen;
      if (auto fieldSlot = slotInProvenance(call.getOperand(0), seen))
        ++counts[*fieldSlot];
    });
    return counts;
  }

  static ::llvm::DenseMap<int64_t, unsigned>
  calls(mlir::LLVM::LLVMFuncOp fn, ::llvm::StringRef attrName,
        ::llvm::ArrayRef<int64_t> allowedSlots) {
    ::llvm::DenseSet<int64_t> allowed(allowedSlots.begin(), allowedSlots.end());
    ::llvm::DenseMap<int64_t, unsigned> counts;
    fn.walk([&](mlir::LLVM::CallOp call) {
      if (!call->hasAttr(attrName) || call.getNumOperands() == 0)
        return;
      ::llvm::SmallPtrSet<mlir::Value, 8> seen;
      auto fieldSlot = slotInProvenance(call.getOperand(0), seen);
      if (!fieldSlot || !allowed.contains(*fieldSlot))
        return;
      ++counts[*fieldSlot];
    });
    return counts;
  }

  static ::llvm::DenseMap<int64_t, unsigned>
  promotes(mlir::LLVM::LLVMFuncOp fn, ::llvm::ArrayRef<int64_t> allowedSlots) {
    ::llvm::DenseSet<int64_t> allowed(allowedSlots.begin(), allowedSlots.end());
    ::llvm::DenseMap<int64_t, unsigned> counts;
    fn.walk([&](mlir::LLVM::CallOp call) {
      if (!hasHelperKind(call, ClassSafetyAttrs::kKindPromote) ||
          call.getNumOperands() == 0)
        return;
      ::llvm::SmallPtrSet<mlir::Value, 8> seen;
      auto fieldSlot = slotInProvenance(call.getOperand(0), seen);
      if (!fieldSlot || !allowed.contains(*fieldSlot))
        return;
      ++counts[*fieldSlot];
    });
    return counts;
  }

  static ::llvm::DenseMap<int64_t, unsigned> loads(mlir::LLVM::LLVMFuncOp fn) {
    ::llvm::DenseMap<int64_t, unsigned> counts;
    fn.walk([&](mlir::Operation *op) {
      if (mlir::isa<mlir::LLVM::CallOp>(op))
        return;
      if (!field_load::aggregateSlot(op))
        return;
      auto fieldSlot =
          attrs::i64(op, OwnershipContractAttrs::kAggregateSlotIndex);
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

  if (mlir::Operation *def = value.getDefiningOp())
    if (field_load::aggregateSlot(def))
      return attrs::i64(def, OwnershipContractAttrs::kAggregateSlotIndex);
  if (auto load = value.getDefiningOp<mlir::LLVM::LoadOp>())
    if (field_load::aggregateSlot(load.getOperation()))
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

static Map only(const Map &counts, ::llvm::ArrayRef<int64_t> slots) {
  ::llvm::DenseSet<int64_t> allowed(slots.begin(), slots.end());
  Map result;
  for (auto [slot, count] : counts)
    if (allowed.contains(slot))
      result[slot] = count;
  return result;
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

static mlir::LogicalResult atLeast(mlir::LLVM::LLVMFuncOp fn, const Map &got,
                                   const Map &minimum,
                                   ::llvm::StringRef description) {
  for (auto [slot, count] : got)
    if (!minimum.contains(slot))
      return fn.emitOpError()
             << description << " count for unexpected field slot " << slot
             << " is " << count;
  for (auto [slot, minimumCount] : minimum) {
    unsigned count = got.lookup(slot);
    if (count < minimumCount)
      return fn.emitOpError()
             << description << " count for field slot " << slot << " is "
             << count << ", expected at least " << minimumCount;
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
  int64_t count = 0;
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
  auto fieldCount =
      attrs::i64(fn.getOperation(), ClassSafetyAttrs::kHelperFieldCount);
  if (!direct || !containers || !fieldCount)
    return fn.emitOpError("class helper is missing field schema metadata");
  if (*fieldCount < 0)
    return fn.emitOpError("class helper field count must be non-negative");
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
  if (auto fieldCount =
          attrs::i64(fn.getOperation(), ClassSafetyAttrs::kHelperFieldCount))
    fields.count = *fieldCount;
  fields.direct.append(directAttrs.begin(), directAttrs.end());
  fields.containers.append(containerAttrs.begin(), containerAttrs.end());
  if (fields.direct.size() != static_cast<size_t>(directCount))
    return fn.emitOpError("class helper direct field index schema does not "
                          "match direct field count");
  if (fields.containers.size() != static_cast<size_t>(containerCount))
    return fn.emitOpError("class helper container field index schema does not "
                          "match container field count");
  auto inRange = [&](int64_t slot) { return slot >= 0 && slot < fields.count; };
  if (!llvm::all_of(fields.direct, inRange) ||
      !llvm::all_of(fields.containers, inRange))
    return fn.emitOpError(
        "class helper field index schema contains an out-of-range slot");
  return mlir::success();
}

} // namespace schema

namespace helper_contract {

static mlir::LogicalResult enter(mlir::LLVM::LLVMFuncOp fn,
                                 ::llvm::StringRef expectedKind,
                                 ::llvm::StringRef description,
                                 bool &shouldVerify) {
  shouldVerify = false;
  auto helperKind =
      attrs::str(fn.getOperation(), ClassSafetyAttrs::kHelperKind);
  if (!helperKind || *helperKind != expectedKind)
    return mlir::success();

  shouldVerify = true;
  auto helperClass =
      attrs::str(fn.getOperation(), ClassSafetyAttrs::kHelperClass);
  if (!helperClass)
    return fn.emitOpError() << description << " is missing class metadata";
  if (fn.isDeclaration())
    return fn.emitOpError() << description << " contract has no body to verify";
  return mlir::success();
}

} // namespace helper_contract

mlir::LogicalResult
verifier::class_helper::Incref::verify(mlir::LLVM::LLVMFuncOp fn) {
  bool shouldVerify = false;
  if (mlir::failed(helper_contract::enter(fn, ClassSafetyAttrs::kKindIncref,
                                          "class incref helper", shouldVerify)))
    return mlir::failure();
  if (!shouldVerify)
    return mlir::success();

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

  if (mlir::failed(slot_counts::exact(
          fn,
          class_field::Slots::calls(
              fn, OwnershipContractAttrs::kAggregateRetain, directFields),
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
  return call->hasAttr(ClassSafetyAttrs::kDeallocPart);
}

static bool storageFree(mlir::LLVM::LLVMFuncOp fn, mlir::LLVM::CallOp call) {
  if (!memoryFree(call) || fn.getBody().empty() || call.getNumOperands() < 1)
    return false;
  mlir::Block &entry = fn.getBody().front();
  if (entry.getNumArguments() < 1)
    return false;
  if (entry.getNumArguments() >= 10)
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
  bool shouldVerify = false;
  if (mlir::failed(helper_contract::enter(fn, ClassSafetyAttrs::kKindDecref,
                                          "class decref helper", shouldVerify)))
    return mlir::failure();
  if (!shouldVerify)
    return mlir::success();

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
      if (!class_release::memoryFree(call))
        return;
      if (class_release::storageFree(fn, call)) {
        call.emitOpError("class decref helper must not free caller-owned "
                         "class carrier storage");
        failedAny = true;
        return;
      }
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
  if (mlir::failed(slot_counts::exact(
          fn,
          class_field::Slots::calls(
              fn, OwnershipContractAttrs::kAggregateRelease, directFields),
          slot_counts::expected(directFields, 2),
          "class decref helper direct field release")))
    return mlir::failure();

  slot_counts::Map expectedLoads = slot_counts::expected(directFields, 2);
  slot_counts::add(expectedLoads, containerFields, 1);
  if (mlir::failed(slot_counts::atLeast(fn, class_field::Slots::loads(fn),
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
  bool shouldVerify = false;
  if (mlir::failed(
          helper_contract::enter(fn, ClassSafetyAttrs::kKindDestroyLocal,
                                 "class destroy_local helper", shouldVerify)))
    return mlir::failure();
  if (!shouldVerify)
    return mlir::success();

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
          class_field::Slots::calls(
              fn, OwnershipContractAttrs::kAggregateRelease, directFields),
          slot_counts::expected(directFields, 1),
          "class destroy_local helper direct field release")))
    return mlir::failure();
  slot_counts::Map expectedLoads = slot_counts::expected(directFields, 1);
  slot_counts::add(expectedLoads, containerFields, 1);
  if (mlir::failed(
          slot_counts::atLeast(fn, class_field::Slots::loads(fn), expectedLoads,
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

static bool from(mlir::Value value, mlir::Value root,
                 ::llvm::SmallPtrSetImpl<mlir::Value> &seen) {
  if (!value || !root || !seen.insert(value).second)
    return false;
  if (value == root)
    return true;
  if (auto bitcast = value.getDefiningOp<mlir::LLVM::BitcastOp>())
    return from(bitcast.getArg(), root, seen);
  if (auto gep = value.getDefiningOp<mlir::LLVM::GEPOp>())
    return from(gep.getBase(), root, seen);
  if (auto ptrToInt = value.getDefiningOp<mlir::LLVM::PtrToIntOp>())
    return from(ptrToInt.getArg(), root, seen);
  if (auto intToPtr = value.getDefiningOp<mlir::LLVM::IntToPtrOp>())
    return from(intToPtr.getArg(), root, seen);
  mlir::Operation *def = value.getDefiningOp();
  if (def &&
      mlir::isa<mlir::LLVM::AddOp, mlir::LLVM::SubOp, mlir::LLVM::MulOp,
                mlir::LLVM::UDivOp, mlir::LLVM::URemOp, mlir::LLVM::AndOp>(
          def)) {
    for (mlir::Value operand : def->getOperands())
      if (from(operand, root, seen))
        return true;
  }
  return false;
}

static bool from(mlir::Value value, mlir::Value root) {
  ::llvm::SmallPtrSet<mlir::Value, 8> seen;
  return from(value, root, seen);
}

static bool fromAny(mlir::Value value, ::llvm::ArrayRef<mlir::Value> roots) {
  return ::llvm::any_of(roots, [&](mlir::Value root) {
    ::llvm::SmallPtrSet<mlir::Value, 8> seen;
    return from(value, root, seen);
  });
}

} // namespace derived

namespace promote {

static bool samePosition(::llvm::ArrayRef<int64_t> lhs,
                         ::llvm::ArrayRef<int64_t> rhs) {
  return lhs.size() == rhs.size() && ::llvm::equal(lhs, rhs);
}

static bool positionPrefix(::llvm::ArrayRef<int64_t> prefix,
                           ::llvm::ArrayRef<int64_t> position) {
  return prefix.size() <= position.size() &&
         ::llvm::equal(prefix, position.take_front(prefix.size()));
}

static mlir::Value insertedAt(mlir::Value value,
                              ::llvm::ArrayRef<int64_t> position,
                              ::llvm::SmallPtrSetImpl<mlir::Value> &seen) {
  if (!value || !seen.insert(value).second)
    return {};
  if (position.empty())
    return value;
  auto insert = value.getDefiningOp<mlir::LLVM::InsertValueOp>();
  if (!insert)
    return {};
  ::llvm::ArrayRef<int64_t> insertedPosition = insert.getPosition();
  if (samePosition(insertedPosition, position))
    return insert.getValue();
  if (positionPrefix(insertedPosition, position)) {
    ::llvm::ArrayRef<int64_t> nestedPosition =
        position.drop_front(insertedPosition.size());
    return insertedAt(insert.getValue(), nestedPosition, seen);
  }
  return insertedAt(insert.getContainer(), position, seen);
}

static bool initValue(mlir::Value value, int64_t expectedValue,
                      ::llvm::ArrayRef<int64_t> position) {
  if (constant::llvmBoolOrInt(value, expectedValue))
    return true;
  ::llvm::SmallPtrSet<mlir::Value, 8> seen;
  mlir::Value slotValue = insertedAt(value, position, seen);
  return slotValue && constant::llvmBoolOrInt(slotValue, expectedValue);
}

static mlir::SmallVector<mlir::Value> freshObjects(mlir::LLVM::LLVMFuncOp fn) {
  mlir::SmallVector<mlir::Value> roots;
  fn.walk([&](mlir::Operation *op) {
    if (!op->hasAttr(ClassSafetyAttrs::kPromoteFreshObject))
      return;
    if (op->getNumResults() == 1)
      roots.push_back(op->getResult(0));
  });
  return roots;
}

static mlir::LogicalResult
freshAllocationContract(mlir::LLVM::LLVMFuncOp fn,
                        ::llvm::ArrayRef<mlir::Value> freshObjects) {
  bool failedAny = false;

  for (mlir::Value fresh : freshObjects) {
    mlir::Operation *def = fresh.getDefiningOp();
    if (!def || !def->hasAttr(ClassSafetyAttrs::kPromoteFreshObject) ||
        def->getNumResults() != 1) {
      (def ? def : fn.getOperation())
          ->emitOpError("class promote fresh object must be produced by the "
                        "memref allocation bridge and marked as fresh");
      failedAny = true;
    }
  }
  return mlir::failure(failedAny);
}

static mlir::LogicalResult
init(mlir::LLVM::LLVMFuncOp fn, ::llvm::ArrayRef<mlir::Value> freshObjects,
     ::llvm::StringRef attrName, int64_t expectedValue,
     ::llvm::StringRef description, ::llvm::ArrayRef<int64_t> position) {
  unsigned count = 0;
  bool failedAny = false;
  fn.walk(
      [&](mlir::LLVM::StoreOp store) {
        if (!store->hasAttr(attrName))
          return;
        ++count;
        if (!initValue(store.getValue(), expectedValue, position)) {
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
    if (!class_field::hasHelperKind(call, ClassSafetyAttrs::kKindIncref))
      return;
    if (call->hasAttr(OwnershipContractAttrs::kAggregateRetain))
      return;
    ++count;
  });
  return count;
}

} // namespace promote

mlir::LogicalResult
verifier::class_helper::Promote::verify(mlir::LLVM::LLVMFuncOp fn) {
  bool shouldVerify = false;
  if (mlir::failed(helper_contract::enter(fn, ClassSafetyAttrs::kKindPromote,
                                          "class promote helper",
                                          shouldVerify)))
    return mlir::failure();
  if (!shouldVerify)
    return mlir::success();

  schema::Fields fields;
  if (mlir::failed(schema::read(fn, fields)))
    return mlir::failure();
  auto &directFields = fields.direct;
  auto &containerFields = fields.containers;

  mlir::SmallVector<mlir::Value> freshObjects = promote::freshObjects(fn);
  // The erased class ABI returns only header + payload-table descriptors, while
  // the copy path still allocates every concrete payload slot plus the table.
  // The exact count is therefore a layout property, not derivable from the
  // helper signature after ABI erasure.
  if (freshObjects.size() < 2)
    return fn.emitOpError()
           << "class promote helper fresh managed allocation count is "
           << freshObjects.size()
           << ", expected at least header plus payload table";
  unsigned freshHeaderCount = 0;
  for (mlir::Value fresh : freshObjects) {
    mlir::Operation *def = fresh.getDefiningOp();
    if (!def)
      continue;
    if (def->hasAttr(OwnershipContractAttrs::kObjectHeader)) {
      ++freshHeaderCount;
      continue;
    }
    if (!def->hasAttr(ClassSafetyAttrs::kPayloadPart))
      return def->emitOpError(
          "class promote fresh allocation is neither the object header nor "
          "a payload part");
  }
  if (freshHeaderCount != 1)
    return fn.emitOpError()
           << "class promote helper must allocate exactly one fresh object "
              "header, found "
           << freshHeaderCount;
  if (mlir::failed(promote::freshAllocationContract(fn, freshObjects)))
    return mlir::failure();

  if (mlir::failed(promote::init(fn, freshObjects,
                                 ClassSafetyAttrs::kPromoteLockInit, 0,
                                 "lock initialization", {})))
    return mlir::failure();
  if (mlir::failed(promote::init(fn, freshObjects,
                                 ClassSafetyAttrs::kPromoteRefcountInit, 1,
                                 "class refcount initialization", {})))
    return mlir::failure();

  if (promote::retainCalls(fn) != 1)
    return fn.emitOpError(
        "class promote helper must retain the already-managed "
        "entry object on exactly one branch");

  if (mlir::failed(promote::returns(fn, freshObjects)))
    return mlir::failure();

  slot_counts::Map directRetains = class_field::Slots::calls(
      fn, OwnershipContractAttrs::kAggregateRetain, directFields);
  slot_counts::Map directPromotes =
      class_field::Slots::promotes(fn, directFields);
  slot_counts::Map directOwnership = directRetains;
  for (auto [slot, count] : directPromotes)
    directOwnership[slot] += count;
  if (mlir::failed(slot_counts::exact(
          fn, directOwnership, slot_counts::expected(directFields, 1),
          "class promote helper direct field ownership acquisition")))
    return mlir::failure();
  for (int64_t slot : directFields) {
    if (directRetains.lookup(slot) != 0 && directPromotes.lookup(slot) != 0)
      return fn.emitOpError() << "class promote helper direct field slot "
                              << slot << " uses both retain and promote";
  }
  if (mlir::failed(slot_counts::atLeast(
          fn, slot_counts::only(class_field::Slots::loads(fn), directFields),
          slot_counts::expected(directFields, 1),
          "class promote helper direct field load")))
    return mlir::failure();
  if (mlir::failed(slot_counts::exact(
          fn,
          class_field::Slots::calls(
              fn, OwnershipContractAttrs::kAggregateRelease, directFields),
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
