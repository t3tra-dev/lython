#include "Common/RuntimeSupport.h"

#include "Common/ClassLayout.h"
#include "Common/Container.h"
#include "Common/LoweringUtils.h"
#include "Common/Object.h"
#include "Common/ThreadSafetyKernel.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/FormatVariadic.h"

#include <atomic>
#include <cstdint>
#include <optional>
#include <string>

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

namespace py {

namespace {

static constexpr llvm::StringLiteral kMemRefAtomicContractId{
    "ly.internal.memref_atomic_contract_id"};
static constexpr llvm::StringLiteral kMemRefAggregateLoadContractId{
    "ly.internal.memref_aggregate_load_contract_id"};
static constexpr llvm::StringLiteral kMemRefContainerAccessContractId{
    "ly.internal.memref_container_access_contract_id"};
static constexpr llvm::StringLiteral kMemRefDeallocContractId{
    "ly.internal.memref_dealloc_contract_id"};

static void setIndexArrayAttr(mlir::Operation *op, llvm::StringRef name,
                              llvm::ArrayRef<int64_t> indices) {
  mlir::Builder builder(op->getContext());
  llvm::SmallVector<mlir::Attribute, 4> attrs;
  attrs.reserve(indices.size());
  for (int64_t index : indices)
    attrs.push_back(builder.getI64IntegerAttr(index));
  op->setAttr(name, builder.getArrayAttr(attrs));
}

namespace storage_cast {

static mlir::Value source(mlir::Value value) {
  llvm::SmallPtrSet<mlir::Operation *, 4> seen;
  while (value) {
    mlir::Operation *def = value.getDefiningOp();
    if (!def || !seen.insert(def).second)
      return value;

    if (auto cast = mlir::dyn_cast<mlir::memref::CastOp>(def)) {
      mlir::Value input = cast.getSource();
      if (!object_abi::Type::isStorage(input.getType()))
        return value;
      value = input;
      continue;
    }

    if (auto cast = mlir::dyn_cast<mlir::UnrealizedConversionCastOp>(def)) {
      if (cast->getNumOperands() != 1)
        return value;
      mlir::Value input = cast.getOperand(0);
      if (!object_abi::Type::isStorage(input.getType()))
        return value;
      value = input;
      continue;
    }

    return value;
  }
  return {};
}

} // namespace storage_cast

} // namespace

namespace runtime::mlir_async {
namespace {
enum CalleeFlag : unsigned {
  kValueStorage = 1u << 0,
  kRefcount = 1u << 1,
  kCreateHandle = 1u << 2,
  kIsError = 1u << 3,
  kAwaitAndExecute = 1u << 4,
  kBorrowOperand0 = 1u << 5,
  kBorrowOperand1 = 1u << 6,
};

struct Info {
  llvm::StringLiteral name;
  unsigned flags;
  int64_t refcountDelta = 0;
};

static constexpr Info kCallees[] = {
    {"mlirAsyncRuntimeCreateToken", kCreateHandle},
    {"mlirAsyncRuntimeCreateValue", kCreateHandle},
    {"mlirAsyncRuntimeCreateGroup", kCreateHandle},
    {"mlirAsyncRuntimeAddRef", kRefcount | kBorrowOperand0, 1},
    {"mlirAsyncRuntimeDropRef", kRefcount | kBorrowOperand0, -1},
    {"mlirAsyncRuntimeEmplaceToken", kBorrowOperand0},
    {"mlirAsyncRuntimeEmplaceValue", kBorrowOperand0},
    {"mlirAsyncRuntimeSetTokenError", kBorrowOperand0},
    {"mlirAsyncRuntimeSetValueError", kBorrowOperand0},
    {"mlirAsyncRuntimeIsTokenError", kIsError | kBorrowOperand0},
    {"mlirAsyncRuntimeIsValueError", kIsError | kBorrowOperand0},
    {"mlirAsyncRuntimeIsGroupError", kIsError | kBorrowOperand0},
    {"mlirAsyncRuntimeAwaitToken", kBorrowOperand0},
    {"mlirAsyncRuntimeAwaitValue", kBorrowOperand0},
    {"mlirAsyncRuntimeAwaitAllInGroup", kBorrowOperand0},
    {"mlirAsyncRuntimeAwaitTokenAndExecute",
     kAwaitAndExecute | kBorrowOperand0},
    {"mlirAsyncRuntimeAwaitValueAndExecute",
     kAwaitAndExecute | kBorrowOperand0},
    {"mlirAsyncRuntimeAwaitAllInGroupAndExecute",
     kAwaitAndExecute | kBorrowOperand0},
    {"mlirAsyncRuntimeAddTokenToGroup", kBorrowOperand0 | kBorrowOperand1},
    {"mlirAsyncRuntimeGetValueStorage", kValueStorage | kBorrowOperand0},
};

static const Info *lookup(llvm::StringRef name) {
  for (const Info &info : kCallees)
    if (name == info.name)
      return &info;
  return nullptr;
}
} // namespace

bool Callee::valueStorage(llvm::StringRef name) {
  const auto *info = lookup(name);
  return info && (info->flags & kValueStorage);
}

bool Callee::known(llvm::StringRef name) { return lookup(name) != nullptr; }

bool Callee::refcount(llvm::StringRef name) {
  const auto *info = lookup(name);
  return info && (info->flags & kRefcount);
}

bool Callee::createHandle(llvm::StringRef name) {
  const auto *info = lookup(name);
  return info && (info->flags & kCreateHandle);
}

bool Callee::isError(llvm::StringRef name) {
  const auto *info = lookup(name);
  return info && (info->flags & kIsError);
}

bool Callee::awaitAndExecute(llvm::StringRef name) {
  const auto *info = lookup(name);
  return info && (info->flags & kAwaitAndExecute);
}

bool Callee::executeEntry(llvm::StringRef name) {
  return name.starts_with("async_execute_fn");
}

std::optional<int64_t> Callee::refcountDelta(llvm::StringRef name) {
  const auto *info = lookup(name);
  if (!info || !(info->flags & kRefcount))
    return std::nullopt;
  return info->refcountDelta;
}

llvm::SmallVector<unsigned, 2>
Callee::borrowedHandleOperands(llvm::StringRef name) {
  llvm::SmallVector<unsigned, 2> indices;
  const auto *info = lookup(name);
  if (!info)
    return indices;
  if (info->flags & kBorrowOperand0)
    indices.push_back(0);
  if (info->flags & kBorrowOperand1)
    indices.push_back(1);
  return indices;
}
} // namespace runtime::mlir_async

namespace ownership::effect {
void retain(mlir::Operation *op, llvm::ArrayRef<int64_t> indices) {
  setIndexArrayAttr(op, OwnershipContractAttrs::kRetainArgs, indices);
}

void release(mlir::Operation *op, llvm::ArrayRef<int64_t> indices) {
  setIndexArrayAttr(op, OwnershipContractAttrs::kReleaseArgs, indices);
}

void transfer(mlir::Operation *op, llvm::ArrayRef<int64_t> indices) {
  setIndexArrayAttr(op, OwnershipContractAttrs::kTransferArgs, indices);
}

void ownedResults(mlir::Operation *op, llvm::ArrayRef<int64_t> indices) {
  setIndexArrayAttr(op, OwnershipContractAttrs::kOwnedResults, indices);
}
} // namespace ownership::effect

namespace publication::borrow::Attr {
std::string name(unsigned argIndex) {
  return "ly.published_borrow_helper_arg" + std::to_string(argIndex);
}
} // namespace publication::borrow::Attr

void threadsafe::Atomic::set(mlir::Operation *op, llvm::StringRef role,
                             llvm::StringRef ordering,
                             llvm::StringRef retainPremise) {
  if (!op)
    return;
  mlir::Builder builder(op->getContext());
  op->setAttr(ThreadSafetyAttrs::kAtomicRole, builder.getStringAttr(role));
  op->setAttr(ThreadSafetyAttrs::kAtomicOrdering,
              builder.getStringAttr(ordering));
  if (!retainPremise.empty())
    op->setAttr(ThreadSafetyAttrs::kRetainPremise,
                builder.getStringAttr(retainPremise));
}

void threadsafe::memref::Atomic::set(mlir::Operation *op,
                                     llvm::StringRef component,
                                     std::optional<int64_t> slot,
                                     llvm::StringRef group,
                                     llvm::StringRef containerKind) {
  if (!op)
    return;
  mlir::Builder builder(op->getContext());
  if (!component.empty())
    op->setAttr(ThreadSafetyAttrs::kAtomicMemRefComponent,
                builder.getStringAttr(component));
  if (slot)
    op->setAttr(ThreadSafetyAttrs::kAtomicMemRefSlot,
                builder.getI64IntegerAttr(*slot));
  if (!group.empty())
    op->setAttr(ThreadSafetyAttrs::kAtomicMemRefGroup,
                builder.getStringAttr(group));
  if (!containerKind.empty())
    op->setAttr(ThreadSafetyAttrs::kAtomicContainerKind,
                builder.getStringAttr(containerKind));
}

static std::string makeContainerResourceGroup(mlir::Value header);

static llvm::StringRef inferContainerKind(mlir::Value header) {
  auto kind = container::Kind::nameFromHeader(header);
  if (!kind)
    return {};
  return *kind;
}

void container::Header::markAtomicResource(mlir::Operation *op,
                                           mlir::Value header,
                                           std::optional<int64_t> slot) {
  if (!op || !header)
    return;
  threadsafe::memref::Atomic::set(op, ContainerSafetyAttrs::kComponentHeader,
                                  slot, makeContainerResourceGroup(header),
                                  inferContainerKind(header));
}

void threadsafe::Retain::premise(mlir::Operation *op,
                                 llvm::StringRef retainPremise) {
  if (!op || retainPremise.empty())
    return;
  mlir::Builder builder(op->getContext());
  op->setAttr(ThreadSafetyAttrs::kRetainPremise,
              builder.getStringAttr(retainPremise));
}

void threadsafe::Retain::verifyOwnedToken(mlir::Operation *op,
                                          llvm::StringRef proof) {
  if (!op)
    return;
  mlir::Builder builder(op->getContext());
  op->setAttr(ThreadSafetyAttrs::kOwnedTokenVerified, builder.getUnitAttr());
  if (!proof.empty())
    op->setAttr(ThreadSafetyAttrs::kOwnedTokenProof,
                builder.getStringAttr(proof));
}

static std::optional<llvm::StringRef>
getAggregateValueDefStringAttr(mlir::Value value, llvm::StringRef attrName) {
  if (!value)
    return std::nullopt;
  mlir::Operation *def = value.getDefiningOp();
  if (!def)
    return std::nullopt;
  auto attr = def->getAttrOfType<mlir::StringAttr>(attrName);
  if (!attr)
    return std::nullopt;
  return attr.getValue();
}

static std::optional<int64_t> getAggregateConstantIndex(mlir::Value value) {
  if (auto constant = value.getDefiningOp<mlir::arith::ConstantIndexOp>())
    return constant.value();
  if (auto constant = value.getDefiningOp<mlir::arith::ConstantIntOp>())
    return constant.value();
  if (auto constant = value.getDefiningOp<mlir::arith::ConstantOp>())
    if (auto attr = mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue()))
      return attr.getInt();
  if (auto constant = value.getDefiningOp<mlir::LLVM::ConstantOp>())
    if (auto attr = mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue()))
      return attr.getInt();
  return std::nullopt;
}

static mlir::Value stripAggregatePointerCasts(mlir::Value value) {
  while (value) {
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
    if (auto cast = value.getDefiningOp<mlir::memref::CastOp>()) {
      value = cast.getSource();
      continue;
    }
    if (auto subview = value.getDefiningOp<mlir::memref::SubViewOp>()) {
      value = subview.getSource();
      continue;
    }
    if (auto cast = value.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
      if (cast->getNumOperands() == 1) {
        value = cast.getOperand(0);
        continue;
      }
    }
    return value;
  }
  return {};
}

static mlir::Value getAggregateAddressRoot(mlir::Value address) {
  address = stripAggregatePointerCasts(address);
  while (auto gep = address.getDefiningOp<mlir::LLVM::GEPOp>())
    address = stripAggregatePointerCasts(gep.getBase());
  return address;
}

static std::optional<int64_t> getAggregateAddressSlot(mlir::Value address) {
  address = stripAggregatePointerCasts(address);
  std::optional<int64_t> slot;
  while (auto gep = address.getDefiningOp<mlir::LLVM::GEPOp>()) {
    for (int32_t raw : gep.getRawConstantIndices()) {
      if (raw == mlir::LLVM::GEPOp::kDynamicIndex)
        continue;
      slot = static_cast<int64_t>(raw);
    }
    address = stripAggregatePointerCasts(gep.getBase());
  }
  return slot;
}

static std::string makeAggregateValueGroup(mlir::Value value,
                                           llvm::StringRef prefix) {
  value = stripAggregatePointerCasts(value);
  if (auto group = getAggregateValueDefStringAttr(
          value, ContainerSafetyAttrs::kDescriptorGroup))
    return group->str();
  if (auto group = getAggregateValueDefStringAttr(
          value, OwnershipContractAttrs::kAggregateSlotGroup))
    return group->str();
  if (value)
    if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(value))
      return llvm::formatv("{0}.blockarg.{1}", prefix, arg.getArgNumber())
          .str();
  return llvm::formatv("{0}.value.{1}", prefix,
                       reinterpret_cast<std::uintptr_t>(
                           value ? value.getAsOpaquePointer() : nullptr))
      .str();
}

static void setAggregateSlotProvenance(mlir::Operation *op,
                                       llvm::StringRef group,
                                       llvm::StringRef component,
                                       std::optional<int64_t> slot) {
  if (!op)
    return;
  mlir::Builder builder(op->getContext());
  if (!group.empty())
    op->setAttr(OwnershipContractAttrs::kAggregateSlotGroup,
                builder.getStringAttr(group));
  if (!component.empty())
    op->setAttr(OwnershipContractAttrs::kAggregateSlotComponent,
                builder.getStringAttr(component));
  if (slot)
    op->setAttr(OwnershipContractAttrs::kAggregateSlotIndex,
                builder.getI64IntegerAttr(*slot));
}

static void copyAggregateSlotProvenance(mlir::Operation *from,
                                        mlir::Operation *to) {
  if (!from || !to)
    return;
  llvm::StringRef attrs[] = {OwnershipContractAttrs::kAggregateSlotLoad,
                             OwnershipContractAttrs::kAggregateSlotGroup,
                             OwnershipContractAttrs::kAggregateSlotComponent,
                             OwnershipContractAttrs::kAggregateSlotIndex,
                             OwnershipContractAttrs::kAggregateSlotPartIndex};
  for (llvm::StringRef attr : attrs) {
    if (mlir::Attribute value = from->getAttr(attr))
      to->setAttr(attr, value);
  }
}

static void setMemRefAggregateSlotProvenance(mlir::Operation *op,
                                             mlir::Value memref,
                                             mlir::ValueRange indices) {
  std::string group = makeAggregateValueGroup(memref, "memref");
  llvm::StringRef component = ContainerSafetyAttrs::kComponentItems;
  if (auto inferred = getAggregateValueDefStringAttr(
          memref, ContainerSafetyAttrs::kDescriptorComponent))
    component = *inferred;
  std::optional<int64_t> slot;
  if (indices.size() == 1)
    slot = getAggregateConstantIndex(indices.front());
  setAggregateSlotProvenance(op, group, component, slot);
}

static void setMemRefViewAggregateSlotProvenance(mlir::Operation *op,
                                                 mlir::Value memref) {
  std::string group = makeAggregateValueGroup(memref, "memref");
  llvm::StringRef component = ContainerSafetyAttrs::kComponentItems;
  if (auto inferred = getAggregateValueDefStringAttr(
          memref, ContainerSafetyAttrs::kDescriptorComponent))
    component = *inferred;
  setAggregateSlotProvenance(op, group, component, std::nullopt);
}

static void setLLVMAggregateSlotProvenance(mlir::Operation *op,
                                           mlir::Value address) {
  mlir::Value root = getAggregateAddressRoot(address);
  llvm::StringRef component = "llvm-slot";
  if (auto inferred = getAggregateValueDefStringAttr(
          root, OwnershipContractAttrs::kAggregateSlotComponent))
    component = *inferred;
  setAggregateSlotProvenance(op, makeAggregateValueGroup(root, "llvm"),
                             component, getAggregateAddressSlot(address));
}

void ownership::aggregate::Slot::markLoad(mlir::Value value,
                                          llvm::StringRef group,
                                          llvm::StringRef component,
                                          std::optional<int64_t> slot) {
  if (!value)
    return;
  mlir::Operation *def = value.getDefiningOp();
  if (!def)
    return;
  if (auto bitcast = mlir::dyn_cast<mlir::LLVM::BitcastOp>(def)) {
    def->setAttr(OwnershipContractAttrs::kAggregateSlotLoad,
                 mlir::UnitAttr::get(def->getContext()));
    setAggregateSlotProvenance(def, group, component, slot);
    ownership::aggregate::Slot::markLoad(bitcast.getArg(), group, component,
                                         slot);
    return;
  }
  if (auto intToPtr = mlir::dyn_cast<mlir::LLVM::IntToPtrOp>(def)) {
    def->setAttr(OwnershipContractAttrs::kAggregateSlotLoad,
                 mlir::UnitAttr::get(def->getContext()));
    setAggregateSlotProvenance(def, group, component, slot);
    ownership::aggregate::Slot::markLoad(intToPtr.getArg(), group, component,
                                         slot);
    return;
  }
  if (auto ptrToInt = mlir::dyn_cast<mlir::LLVM::PtrToIntOp>(def)) {
    def->setAttr(OwnershipContractAttrs::kAggregateSlotLoad,
                 mlir::UnitAttr::get(def->getContext()));
    setAggregateSlotProvenance(def, group, component, slot);
    ownership::aggregate::Slot::markLoad(ptrToInt.getArg(), group, component,
                                         slot);
    return;
  }
  if (auto indexCast = mlir::dyn_cast<mlir::arith::IndexCastOp>(def)) {
    def->setAttr(OwnershipContractAttrs::kAggregateSlotLoad,
                 mlir::UnitAttr::get(def->getContext()));
    setAggregateSlotProvenance(def, group, component, slot);
    ownership::aggregate::Slot::markLoad(indexCast.getIn(), group, component,
                                         slot);
    return;
  }
  if (auto cast = mlir::dyn_cast<mlir::UnrealizedConversionCastOp>(def)) {
    if (cast->getNumOperands() == 1) {
      def->setAttr(OwnershipContractAttrs::kAggregateSlotLoad,
                   mlir::UnitAttr::get(def->getContext()));
      setAggregateSlotProvenance(def, group, component, slot);
      ownership::aggregate::Slot::markLoad(cast.getOperand(0), group, component,
                                           slot);
    }
    return;
  }
  if (auto gep = mlir::dyn_cast<mlir::LLVM::GEPOp>(def)) {
    def->setAttr(OwnershipContractAttrs::kAggregateSlotLoad,
                 mlir::UnitAttr::get(def->getContext()));
    setAggregateSlotProvenance(def, group, component,
                               slot ? slot : getAggregateAddressSlot(value));
    ownership::aggregate::Slot::markLoad(gep.getBase(), group, component, slot);
    return;
  }
  if (auto subview = mlir::dyn_cast<mlir::memref::SubViewOp>(def)) {
    def->setAttr(OwnershipContractAttrs::kAggregateSlotLoad,
                 mlir::UnitAttr::get(def->getContext()));
    setAggregateSlotProvenance(def, group, component, slot);
    ownership::aggregate::Slot::markLoad(subview.getSource(), group, component,
                                         slot);
    return;
  }
  if (mlir::isa<mlir::LLVM::ExtractValueOp>(def)) {
    def->setAttr(OwnershipContractAttrs::kAggregateSlotLoad,
                 mlir::UnitAttr::get(def->getContext()));
    setAggregateSlotProvenance(def, group, component, slot);
    return;
  }
  if (mlir::isa<mlir::memref::ExtractAlignedPointerAsIndexOp>(def)) {
    def->setAttr(OwnershipContractAttrs::kAggregateSlotLoad,
                 mlir::UnitAttr::get(def->getContext()));
    setAggregateSlotProvenance(def, group, component, slot);
    return;
  }
  if (mlir::isa<mlir::memref::LoadOp, mlir::LLVM::LoadOp>(def)) {
    def->setAttr(OwnershipContractAttrs::kAggregateSlotLoad,
                 mlir::UnitAttr::get(def->getContext()));
    setAggregateSlotProvenance(def, group, component, slot);
  }
}

void ownership::aggregate::Slot::markLoad(mlir::Value value) {
  if (!value)
    return;
  mlir::Operation *def = value.getDefiningOp();
  if (!def)
    return;
  if (def->hasAttr(OwnershipContractAttrs::kAggregateSlotLoad) &&
      def->hasAttr(OwnershipContractAttrs::kAggregateSlotIndex))
    return;
  if (auto bitcast = mlir::dyn_cast<mlir::LLVM::BitcastOp>(def)) {
    def->setAttr(OwnershipContractAttrs::kAggregateSlotLoad,
                 mlir::UnitAttr::get(def->getContext()));
    setLLVMAggregateSlotProvenance(def, value);
    ownership::aggregate::Slot::markLoad(bitcast.getArg());
    return;
  }
  if (auto intToPtr = mlir::dyn_cast<mlir::LLVM::IntToPtrOp>(def)) {
    def->setAttr(OwnershipContractAttrs::kAggregateSlotLoad,
                 mlir::UnitAttr::get(def->getContext()));
    setLLVMAggregateSlotProvenance(def, value);
    ownership::aggregate::Slot::markLoad(intToPtr.getArg());
    return;
  }
  if (auto ptrToInt = mlir::dyn_cast<mlir::LLVM::PtrToIntOp>(def)) {
    def->setAttr(OwnershipContractAttrs::kAggregateSlotLoad,
                 mlir::UnitAttr::get(def->getContext()));
    setLLVMAggregateSlotProvenance(def, value);
    ownership::aggregate::Slot::markLoad(ptrToInt.getArg());
    return;
  }
  if (auto indexCast = mlir::dyn_cast<mlir::arith::IndexCastOp>(def)) {
    def->setAttr(OwnershipContractAttrs::kAggregateSlotLoad,
                 mlir::UnitAttr::get(def->getContext()));
    setLLVMAggregateSlotProvenance(def, value);
    ownership::aggregate::Slot::markLoad(indexCast.getIn());
    return;
  }
  if (auto cast = mlir::dyn_cast<mlir::UnrealizedConversionCastOp>(def)) {
    if (cast->getNumOperands() == 1) {
      def->setAttr(OwnershipContractAttrs::kAggregateSlotLoad,
                   mlir::UnitAttr::get(def->getContext()));
      setLLVMAggregateSlotProvenance(def, value);
      ownership::aggregate::Slot::markLoad(cast.getOperand(0));
    }
    return;
  }
  if (auto gep = mlir::dyn_cast<mlir::LLVM::GEPOp>(def)) {
    def->setAttr(OwnershipContractAttrs::kAggregateSlotLoad,
                 mlir::UnitAttr::get(def->getContext()));
    setLLVMAggregateSlotProvenance(def, value);
    ownership::aggregate::Slot::markLoad(gep.getBase());
    return;
  }
  if (auto subview = mlir::dyn_cast<mlir::memref::SubViewOp>(def)) {
    def->setAttr(OwnershipContractAttrs::kAggregateSlotLoad,
                 mlir::UnitAttr::get(def->getContext()));
    setMemRefViewAggregateSlotProvenance(def, subview.getSource());
    ownership::aggregate::Slot::markLoad(subview.getSource());
    return;
  }
  if (mlir::isa<mlir::LLVM::ExtractValueOp>(def)) {
    def->setAttr(OwnershipContractAttrs::kAggregateSlotLoad,
                 mlir::UnitAttr::get(def->getContext()));
    setLLVMAggregateSlotProvenance(def, value);
    return;
  }
  if (auto extract =
          mlir::dyn_cast<mlir::memref::ExtractAlignedPointerAsIndexOp>(def)) {
    def->setAttr(OwnershipContractAttrs::kAggregateSlotLoad,
                 mlir::UnitAttr::get(def->getContext()));
    setMemRefAggregateSlotProvenance(def, extract.getSource(), {});
    return;
  }
  if (auto load = mlir::dyn_cast<mlir::memref::LoadOp>(def)) {
    def->setAttr(OwnershipContractAttrs::kAggregateSlotLoad,
                 mlir::UnitAttr::get(def->getContext()));
    setMemRefAggregateSlotProvenance(def, load.getMemref(), load.getIndices());
    return;
  }
  if (auto load = mlir::dyn_cast<mlir::LLVM::LoadOp>(def)) {
    def->setAttr(OwnershipContractAttrs::kAggregateSlotLoad,
                 mlir::UnitAttr::get(def->getContext()));
    setLLVMAggregateSlotProvenance(def, load.getAddr());
    return;
  }
}

void ownership::aggregate::Slot::copyLoad(mlir::Value from, mlir::Value to) {
  mlir::Operation *target = to ? to.getDefiningOp() : nullptr;
  if (!target)
    return;

  mlir::Operation *source = from ? from.getDefiningOp() : nullptr;
  if (!source || !source->hasAttr(OwnershipContractAttrs::kAggregateSlotLoad)) {
    ownership::aggregate::Slot::markLoad(from);
    source = from ? from.getDefiningOp() : nullptr;
  }
  if (!source || !source->hasAttr(OwnershipContractAttrs::kAggregateSlotLoad))
    return;
  copyAggregateSlotProvenance(source, target);
}

void ownership::aggregate::Slot::markStore(mlir::Operation *op) {
  if (!op)
    return;
  op->setAttr(OwnershipContractAttrs::kMemRefSlotTransfer,
              mlir::UnitAttr::get(op->getContext()));
  if (auto store = mlir::dyn_cast<mlir::memref::StoreOp>(op)) {
    setMemRefAggregateSlotProvenance(op, store.getMemref(), store.getIndices());
    return;
  }
  if (auto store = mlir::dyn_cast<mlir::LLVM::StoreOp>(op))
    setLLVMAggregateSlotProvenance(op, store.getAddr());
}

void ownership::aggregate::Slot::markStore(mlir::Operation *op,
                                           llvm::StringRef group,
                                           llvm::StringRef component,
                                           std::optional<int64_t> slot) {
  if (!op)
    return;
  op->setAttr(OwnershipContractAttrs::kMemRefSlotTransfer,
              mlir::UnitAttr::get(op->getContext()));
  setAggregateSlotProvenance(op, group, component, slot);
}

void ownership::Pointer::markNonObject(mlir::Value value) {
  if (!value)
    return;
  mlir::Operation *def = value.getDefiningOp();
  if (!def)
    return;
  def->setAttr(OwnershipContractAttrs::kNonObjectPointer,
               mlir::UnitAttr::get(def->getContext()));
  if (auto bitcast = mlir::dyn_cast<mlir::LLVM::BitcastOp>(def)) {
    ownership::Pointer::markNonObject(bitcast.getArg());
    return;
  }
  if (auto gep = mlir::dyn_cast<mlir::LLVM::GEPOp>(def)) {
    ownership::Pointer::markNonObject(gep.getBase());
    return;
  }
  if (auto cast = mlir::dyn_cast<mlir::UnrealizedConversionCastOp>(def)) {
    if (cast->getNumOperands() == 1)
      ownership::Pointer::markNonObject(cast.getOperand(0));
  }
}

std::string container::descriptor::Group::make(mlir::Operation *owner,
                                               llvm::StringRef kind) {
  (void)owner;
  static std::atomic<std::uint64_t> nextGroupId{0};
  return llvm::formatv("{0}.{1}", kind,
                       nextGroupId.fetch_add(1, std::memory_order_relaxed))
      .str();
}

static void setValueDefStringAttr(mlir::Value value, llvm::StringRef attrName,
                                  llvm::StringRef attrValue) {
  if (!value || attrValue.empty())
    return;
  mlir::Operation *def = value.getDefiningOp();
  if (!def)
    return;
  mlir::Builder builder(def->getContext());
  def->setAttr(attrName, builder.getStringAttr(attrValue));
}

static std::optional<llvm::StringRef>
getValueDefStringAttr(mlir::Value value, llvm::StringRef attrName) {
  if (!value)
    return std::nullopt;
  mlir::Operation *def = value.getDefiningOp();
  if (!def)
    return std::nullopt;
  auto attr = def->getAttrOfType<mlir::StringAttr>(attrName);
  if (!attr)
    return std::nullopt;
  return attr.getValue();
}

static std::string makeContainerResourceGroup(mlir::Value header) {
  if (auto group =
          getValueDefStringAttr(header, ContainerSafetyAttrs::kDescriptorGroup))
    return group->str();
  if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(header))
    return "blockarg." + std::to_string(arg.getArgNumber());
  return "value." + std::to_string(reinterpret_cast<std::uintptr_t>(
                        header.getAsOpaquePointer()));
}

void container::descriptor::Component::mark(mlir::Value value,
                                            llvm::StringRef group,
                                            llvm::StringRef component) {
  setValueDefStringAttr(value, ContainerSafetyAttrs::kDescriptorGroup, group);
  llvm::StringRef kind = group.take_until([](char c) { return c == '.'; });
  setValueDefStringAttr(value, ContainerSafetyAttrs::kDescriptorKind, kind);
  setValueDefStringAttr(value, ContainerSafetyAttrs::kDescriptorComponent,
                        component);
}

void container::access::Contract::mark(mlir::Operation *op, mlir::Value header,
                                       mlir::Value target) {
  if (!op || !header || !target)
    return;
  mlir::Builder builder(op->getContext());
  op->setAttr(ContainerSafetyAttrs::kAccessGroup,
              builder.getStringAttr(makeContainerResourceGroup(header)));
  if (auto component = getValueDefStringAttr(
          target, ContainerSafetyAttrs::kDescriptorComponent))
    op->setAttr(ContainerSafetyAttrs::kAccessComponent,
                builder.getStringAttr(*component));
}

// Async exception cells are not Python objects, but they follow the same
// header/payload split: one pointer-sized atomic header word publishes the
// payload, and the payload caches the lowered header/payload memref
// descriptors that make up a LyException value.  The header remains i64
// because it must hold both sentinel values and the published aligned header
// pointer for cmpxchg.  Descriptor word 0 is that same published pointer, so
// the payload cache stores only descriptor words 1..9.
static constexpr unsigned kExceptionCellHeaderSlots = 1;
static constexpr unsigned kExceptionCellStateSlot = 0;
static constexpr unsigned kExceptionCellPayloadBaseSlot =
    kExceptionCellHeaderSlots;
static constexpr unsigned kExceptionPayloadAlignedSlot = 0;
static constexpr unsigned kExceptionPayloadAllocatedSlot = 1;
static constexpr unsigned kExceptionPayloadOffsetSlot = 2;
static constexpr unsigned kExceptionPayloadSizeSlot = 3;
static constexpr unsigned kExceptionPayloadStrideSlot = 4;
static constexpr unsigned kExceptionCellDescriptorWords = 5;
static constexpr unsigned kExceptionCellDescriptorCount = 3;
static constexpr unsigned kExceptionPayloadWordCount =
    kExceptionCellDescriptorWords * kExceptionCellDescriptorCount;
static constexpr unsigned kExceptionCellPayloadSlots =
    kExceptionPayloadWordCount - 1;
static constexpr unsigned kExceptionCellTotalSlots =
    kExceptionCellHeaderSlots + kExceptionCellPayloadSlots;
static constexpr int64_t kExceptionCellEmpty = 0;
static constexpr int64_t kExceptionCellPublishing = 1;

mlir::MemRefType async_runtime::getExceptionCellType(mlir::MLIRContext *ctx) {
  return mlir::MemRefType::get({kExceptionCellTotalSlots},
                               mlir::IntegerType::get(ctx, 64));
}

bool async_runtime::isExceptionCellType(mlir::Type type) {
  auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type);
  if (!memrefType || memrefType.getRank() != 1 ||
      memrefType.getShape()[0] != kExceptionCellTotalSlots)
    return false;
  auto intType = mlir::dyn_cast<mlir::IntegerType>(memrefType.getElementType());
  return intType && intType.getWidth() == 64;
}

bool async_runtime::isLoweredExceptionCellType(mlir::Type type) {
  return object_abi::Type::isLoweredStorage(type);
}

static bool hasFunctionArgAttr(mlir::Value value, llvm::StringRef attrName) {
  auto arg = mlir::dyn_cast<mlir::BlockArgument>(value);
  if (!arg || !arg.getOwner())
    return false;
  mlir::Operation *parent = arg.getOwner()->getParentOp();
  auto function = mlir::dyn_cast_or_null<mlir::FunctionOpInterface>(parent);
  if (function && arg.getArgNumber() < function.getNumArguments() &&
      function.getArgAttr(arg.getArgNumber(), attrName))
    return true;

  auto argAttrs = parent ? parent->getAttrOfType<mlir::ArrayAttr>("arg_attrs")
                         : mlir::ArrayAttr();
  if (!argAttrs || arg.getArgNumber() >= argAttrs.size())
    return false;
  auto dict =
      mlir::dyn_cast<mlir::DictionaryAttr>(argAttrs[arg.getArgNumber()]);
  return dict && static_cast<bool>(dict.get(attrName));
}

static mlir::Value getExceptionCellAddress(mlir::Location loc, mlir::Value cell,
                                           mlir::OpBuilder &builder) {
  if (!cell)
    return {};
  auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  if (async_runtime::ExceptionCell::hasProvenance(cell) &&
      async_runtime::isExceptionCellType(cell.getType())) {
    mlir::Value index =
        builder.create<mlir::memref::ExtractAlignedPointerAsIndexOp>(loc, cell);
    mlir::Value address = builder.create<mlir::arith::IndexCastOp>(
        loc, builder.getI64Type(), index);
    auto ptr = builder.create<mlir::LLVM::IntToPtrOp>(loc, ptrType, address);
    ptr->setAttr(AsyncSafetyAttrs::kExceptionCell,
                 mlir::UnitAttr::get(builder.getContext()));
    return ptr;
  }
  if (auto descriptorType =
          mlir::dyn_cast<mlir::LLVM::LLVMStructType>(cell.getType())) {
    if (!async_runtime::isLoweredExceptionCellType(descriptorType) ||
        !async_runtime::ExceptionCell::hasProvenance(cell))
      return {};
    auto aligned = builder.create<mlir::LLVM::ExtractValueOp>(
        loc, descriptorType.getBody()[1], cell,
        builder.getDenseI64ArrayAttr({1}));
    aligned->setAttr(AsyncSafetyAttrs::kExceptionCell,
                     mlir::UnitAttr::get(builder.getContext()));
    return aligned;
  }
  return {};
}

static mlir::Value gepExceptionCellSlot(mlir::Location loc, mlir::Value base,
                                        unsigned slot,
                                        mlir::OpBuilder &builder) {
  auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  auto i64Type = builder.getI64Type();
  mlir::Value slotValue = builder.create<mlir::LLVM::ConstantOp>(
      loc, i64Type, builder.getI64IntegerAttr(slot));
  return builder.create<mlir::LLVM::GEPOp>(
      loc, ptrType, i64Type, base,
      llvm::ArrayRef<mlir::LLVM::GEPArg>{slotValue});
}

static mlir::Value loadExceptionCellSlot(mlir::Location loc, mlir::Value base,
                                         unsigned slot,
                                         mlir::OpBuilder &builder) {
  mlir::Value address = gepExceptionCellSlot(loc, base, slot, builder);
  auto load = builder.create<mlir::LLVM::LoadOp>(
      loc, builder.getI64Type(), address, /*alignment=*/8,
      /*isVolatile=*/false, /*isNonTemporal=*/false, /*isInvariant=*/false,
      /*isInvariantGroup=*/false,
      slot == kExceptionCellStateSlot ? mlir::LLVM::AtomicOrdering::acquire
                                      : mlir::LLVM::AtomicOrdering::not_atomic);
  if (slot == kExceptionCellStateSlot)
    threadsafe::Atomic::set(load.getOperation(),
                            ThreadSafetyAttrs::kRoleAsyncExceptionLoad,
                            ThreadSafetyAttrs::kOrderingAcquire);
  return load.getResult();
}

static mlir::Value loadExceptionCellMemRefSlot(mlir::Location loc,
                                               mlir::Value cell, unsigned slot,
                                               mlir::OpBuilder &builder) {
  mlir::Value index = builder.create<mlir::arith::ConstantIndexOp>(loc, slot);
  if (slot == kExceptionCellStateSlot) {
    mlir::Value zero = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 64);
    auto load = builder.create<mlir::memref::AtomicRMWOp>(
        loc, mlir::arith::AtomicRMWKind::addi, zero, cell,
        mlir::ValueRange{index});
    threadsafe::Atomic::set(load.getOperation(),
                            ThreadSafetyAttrs::kRoleAsyncExceptionLoad,
                            ThreadSafetyAttrs::kOrderingAcquire);
    load->setAttr(AsyncSafetyAttrs::kExceptionCell, builder.getUnitAttr());
    return load.getResult();
  }
  auto load = builder.create<mlir::memref::LoadOp>(loc, cell, index);
  if (slot >= kExceptionCellPayloadBaseSlot)
    ownership::aggregate::Slot::markLoad(load.getResult(),
                                         "async.exception_cell", "exception",
                                         slot - kExceptionCellPayloadBaseSlot);
  return load.getResult();
}

static void storeExceptionCellMemRefSlot(mlir::Location loc, mlir::Value cell,
                                         unsigned slot, mlir::Value value,
                                         mlir::OpBuilder &builder) {
  mlir::Value index = builder.create<mlir::arith::ConstantIndexOp>(loc, slot);
  auto store = builder.create<mlir::memref::StoreOp>(loc, value, cell, index);
  if (slot >= kExceptionCellPayloadBaseSlot)
    store->setAttr(AsyncSafetyAttrs::kExceptionCellPayloadStore,
                   builder.getUnitAttr());
}

static mlir::Value
transitionExceptionCellState(mlir::Location loc, mlir::Value cell,
                             mlir::Value expected, mlir::Value replacement,
                             bool reservation, mlir::OpBuilder &builder) {
  mlir::Value index = builder.create<mlir::arith::ConstantIndexOp>(
      loc, kExceptionCellStateSlot);
  auto atomic = builder.create<mlir::memref::GenericAtomicRMWOp>(
      loc, cell, mlir::ValueRange{index});
  threadsafe::Atomic::set(atomic.getOperation(),
                          ThreadSafetyAttrs::kRoleAsyncExceptionStore,
                          ThreadSafetyAttrs::kOrderingAcqRel);
  atomic->setAttr(AsyncSafetyAttrs::kExceptionCellConditionalStore,
                  builder.getUnitAttr());
  if (reservation)
    atomic->setAttr(AsyncSafetyAttrs::kExceptionCellReservation,
                    builder.getUnitAttr());
  async_runtime::ExceptionCell::mark(cell);

  mlir::OpBuilder bodyBuilder =
      mlir::OpBuilder::atBlockEnd(atomic.getBody(), builder.getListener());
  mlir::Value current = atomic.getCurrentValue();
  mlir::Value matches = bodyBuilder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::eq, current, expected);
  mlir::Value next = bodyBuilder.create<mlir::arith::SelectOp>(
      loc, matches, replacement, current);
  bodyBuilder.create<mlir::memref::AtomicYieldOp>(loc, next);
  return atomic.getResult();
}

static void storeExceptionCellSlot(mlir::Location loc, mlir::Value base,
                                   unsigned slot, mlir::Value value,
                                   mlir::OpBuilder &builder) {
  mlir::Value address = gepExceptionCellSlot(loc, base, slot, builder);
  auto store =
      builder.create<mlir::LLVM::StoreOp>(loc, value, address, /*alignment=*/8);
  if (slot >= kExceptionCellPayloadBaseSlot)
    store->setAttr(AsyncSafetyAttrs::kExceptionCellPayloadStore,
                   builder.getUnitAttr());
}

static void assumeExceptionCellEmpty(mlir::Location loc, mlir::Value cell,
                                     mlir::OpBuilder &builder) {
  if (!cell)
    return;
  if (async_runtime::isExceptionCellType(cell.getType())) {
    mlir::Value slot = builder.create<mlir::arith::ConstantIndexOp>(
        loc, kExceptionCellStateSlot);
    mlir::Value word =
        builder.create<mlir::memref::LoadOp>(loc, cell, slot).getResult();
    mlir::Value zero = builder.create<mlir::arith::ConstantOp>(
        loc, builder.getI64IntegerAttr(kExceptionCellEmpty));
    mlir::Value empty = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::eq, word, zero);
    builder.create<mlir::LLVM::AssumeOp>(loc, empty);
    return;
  }
  if (async_runtime::ExceptionCell::hasProvenance(cell) &&
      async_runtime::isLoweredExceptionCellType(cell.getType())) {
    mlir::Value address = getExceptionCellAddress(loc, cell, builder);
    mlir::Value word =
        loadExceptionCellSlot(loc, address, kExceptionCellStateSlot, builder);
    mlir::Value zero = builder.create<mlir::LLVM::ConstantOp>(
        loc, builder.getI64Type(),
        builder.getI64IntegerAttr(kExceptionCellEmpty));
    mlir::Value empty = builder.create<mlir::LLVM::ICmpOp>(
        loc, mlir::LLVM::ICmpPredicate::eq, word, zero);
    builder.create<mlir::LLVM::AssumeOp>(loc, empty);
  }
}

namespace exception_payload {

static mlir::LLVM::LLVMStructType descriptorType(mlir::MLIRContext *ctx) {
  return object_abi::Type::loweredStorage(ctx);
}

static mlir::LLVM::LLVMStructType partsDescriptorType(mlir::MLIRContext *ctx) {
  auto single = descriptorType(ctx);
  return mlir::LLVM::LLVMStructType::getLiteral(
      ctx, llvm::ArrayRef<mlir::Type>{single, single, single});
}

static bool isPartsDescriptorType(mlir::Type type) {
  if (!type)
    return false;
  auto aggregate = mlir::dyn_cast<mlir::LLVM::LLVMStructType>(type);
  if (!aggregate || aggregate.isOpaque() ||
      aggregate.getBody().size() != kExceptionCellDescriptorCount)
    return false;
  return llvm::all_of(aggregate.getBody(), [](mlir::Type part) {
    return object_abi::Type::isLoweredStorage(part);
  });
}

static bool isExceptionPartType(unsigned index, mlir::Type type) {
  if (object_abi::Type::isLoweredStorage(type))
    return true;
  if (index == 0)
    return object_abi::exception_abi::Parts::isHeader(type);
  if (index == 1)
    return object_abi::exception_abi::Parts::isMessageHeader(type);
  if (index == 2)
    return object_abi::exception_abi::Parts::isMessageBytes(type);
  return false;
}

static mlir::Value ptrToI64(mlir::Location loc, mlir::Value pointer,
                            mlir::OpBuilder &builder) {
  if (pointer.getType().isInteger(64))
    return pointer;
  return builder.create<mlir::LLVM::PtrToIntOp>(loc, builder.getI64Type(),
                                                pointer);
}

static mlir::Value nonObjectPtrToI64(mlir::Location loc, mlir::Value pointer,
                                     mlir::OpBuilder &builder) {
  mlir::Value bits = ptrToI64(loc, pointer, builder);
  ownership::Pointer::markNonObject(bits);
  return bits;
}

static mlir::Value i64ToPtr(mlir::Location loc, mlir::Value value,
                            mlir::OpBuilder &builder) {
  auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  if (value.getType() == ptrType)
    return value;
  return builder.create<mlir::LLVM::IntToPtrOp>(loc, ptrType, value);
}

static mlir::Value descriptorSizeForMemRef(mlir::Location loc,
                                           mlir::MemRefType memrefType,
                                           mlir::Value memref,
                                           mlir::OpBuilder &builder) {
  if (memrefType.getRank() != 1)
    return {};
  if (memrefType.hasStaticShape()) {
    return builder.create<mlir::LLVM::ConstantOp>(
        loc, builder.getI64Type(),
        builder.getI64IntegerAttr(memrefType.getShape()[0]));
  }
  mlir::Value dim =
      builder.create<mlir::memref::DimOp>(loc, memref, /*index=*/0);
  return builder.create<mlir::arith::IndexCastOp>(loc, builder.getI64Type(),
                                                  dim);
}

static mlir::Value alignedPointer(mlir::Location loc, mlir::Value exception,
                                  mlir::OpBuilder &builder) {
  if (!exception)
    return {};
  auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  if (auto cast = exception.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
    if (cast->getNumOperands() == kExceptionCellDescriptorCount)
      return alignedPointer(loc, cast.getOperand(0), builder);
  }
  if (mlir::isa<mlir::MemRefType>(exception.getType())) {
    mlir::Value index =
        builder.create<mlir::memref::ExtractAlignedPointerAsIndexOp>(loc,
                                                                     exception);
    mlir::Value address = builder.create<mlir::arith::IndexCastOp>(
        loc, builder.getI64Type(), index);
    return builder.create<mlir::LLVM::IntToPtrOp>(loc, ptrType, address);
  }
  if (auto descriptor =
          mlir::dyn_cast<mlir::LLVM::LLVMStructType>(exception.getType())) {
    if (isPartsDescriptorType(descriptor)) {
      mlir::Value root = builder.create<mlir::LLVM::ExtractValueOp>(
          loc, descriptor.getBody()[0], exception,
          builder.getDenseI64ArrayAttr({0}));
      return alignedPointer(loc, root, builder);
    }
    if (!object_abi::Type::isLoweredStorage(descriptor))
      return {};
    return builder.create<mlir::LLVM::ExtractValueOp>(
        loc, descriptor.getBody()[1], exception,
        builder.getDenseI64ArrayAttr({1}));
  }
  return {};
}

static void appendOneDescriptorWords(mlir::Location loc, mlir::Value storage,
                                     bool rootDescriptor,
                                     llvm::SmallVectorImpl<mlir::Value> &words,
                                     mlir::OpBuilder &builder) {
  auto i64Type = builder.getI64Type();
  auto zero = builder.create<mlir::LLVM::ConstantOp>(
      loc, i64Type, builder.getI64IntegerAttr(0));
  auto stride = builder.create<mlir::LLVM::ConstantOp>(
      loc, i64Type, builder.getI64IntegerAttr(1));

  if (auto memrefType = mlir::dyn_cast<mlir::MemRefType>(storage.getType())) {
    mlir::Value aligned = alignedPointer(loc, storage, builder);
    mlir::Value size =
        descriptorSizeForMemRef(loc, memrefType, storage, builder);
    if (!aligned || !size)
      return;
    mlir::Value alignedInt = rootDescriptor
                                 ? ptrToI64(loc, aligned, builder)
                                 : nonObjectPtrToI64(loc, aligned, builder);
    mlir::Value allocatedInt = nonObjectPtrToI64(loc, aligned, builder);
    words.append({alignedInt, allocatedInt, zero, size, stride});
    return;
  }

  if (auto descriptor =
          mlir::dyn_cast<mlir::LLVM::LLVMStructType>(storage.getType())) {
    if (!object_abi::Type::isLoweredStorage(descriptor))
      return;
    auto body = descriptor.getBody();
    mlir::Value allocated = builder.create<mlir::LLVM::ExtractValueOp>(
        loc, body[0], storage, builder.getDenseI64ArrayAttr({0}));
    mlir::Value aligned = builder.create<mlir::LLVM::ExtractValueOp>(
        loc, body[1], storage, builder.getDenseI64ArrayAttr({1}));
    mlir::Value offset = builder.create<mlir::LLVM::ExtractValueOp>(
        loc, body[2], storage, builder.getDenseI64ArrayAttr({2}));
    mlir::Value size = builder.create<mlir::LLVM::ExtractValueOp>(
        loc, i64Type, storage, builder.getDenseI64ArrayAttr({3, 0}));
    mlir::Value descriptorStride = builder.create<mlir::LLVM::ExtractValueOp>(
        loc, i64Type, storage, builder.getDenseI64ArrayAttr({4, 0}));
    words.push_back(rootDescriptor ? ptrToI64(loc, aligned, builder)
                                   : nonObjectPtrToI64(loc, aligned, builder));
    words.push_back(nonObjectPtrToI64(loc, allocated, builder));
    words.push_back(offset);
    words.push_back(size);
    words.push_back(descriptorStride);
    return;
  }

  (void)zero;
  (void)stride;
}

static void appendDescriptorWords(mlir::Location loc, mlir::Value exception,
                                  llvm::SmallVectorImpl<mlir::Value> &words,
                                  mlir::OpBuilder &builder) {
  words.clear();
  if (!exception)
    return;

  if (auto cast = exception.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
    if (cast->getNumOperands() == kExceptionCellDescriptorCount) {
      for (auto [index, operand] : llvm::enumerate(cast.getOperands()))
        appendOneDescriptorWords(loc, operand, index == 0, words, builder);
      return;
    }
  }

  if (auto aggregate =
          mlir::dyn_cast<mlir::LLVM::LLVMStructType>(exception.getType())) {
    if (isPartsDescriptorType(aggregate)) {
      for (unsigned index = 0; index < kExceptionCellDescriptorCount; ++index) {
        mlir::Value part = builder.create<mlir::LLVM::ExtractValueOp>(
            loc, aggregate.getBody()[index], exception,
            builder.getDenseI64ArrayAttr({index}));
        appendOneDescriptorWords(loc, part, index == 0, words, builder);
      }
      return;
    }
  }

  appendOneDescriptorWords(loc, exception, /*rootDescriptor=*/true, words,
                           builder);
}

static mlir::Value descriptorFromWords(mlir::Location loc, mlir::Value aligned,
                                       mlir::Value allocated,
                                       mlir::Value offset, mlir::Value size,
                                       mlir::Value stride,
                                       mlir::OpBuilder &builder) {
  mlir::MLIRContext *ctx = builder.getContext();
  auto ptrType = mlir::LLVM::LLVMPointerType::get(ctx);
  auto i64Type = builder.getI64Type();
  auto memrefDescriptorType = descriptorType(ctx);
  mlir::Value allocatedPtr = i64ToPtr(loc, allocated, builder);
  mlir::Value alignedPtr = i64ToPtr(loc, aligned, builder);
  ownership::aggregate::Slot::markLoad(allocatedPtr, "async.exception_cell",
                                       "exception", std::nullopt);
  ownership::aggregate::Slot::markLoad(alignedPtr, "async.exception_cell",
                                       "exception", std::nullopt);
  mlir::Value descriptor =
      builder.create<mlir::LLVM::UndefOp>(loc, memrefDescriptorType);
  descriptor = builder.create<mlir::LLVM::InsertValueOp>(
      loc, memrefDescriptorType, descriptor, allocatedPtr,
      builder.getDenseI64ArrayAttr({0}));
  ownership::aggregate::Slot::markLoad(descriptor, "async.exception_cell",
                                       "exception", std::nullopt);
  descriptor = builder.create<mlir::LLVM::InsertValueOp>(
      loc, memrefDescriptorType, descriptor, alignedPtr,
      builder.getDenseI64ArrayAttr({1}));
  ownership::aggregate::Slot::markLoad(descriptor, "async.exception_cell",
                                       "exception", std::nullopt);
  descriptor = builder.create<mlir::LLVM::InsertValueOp>(
      loc, memrefDescriptorType, descriptor, offset,
      builder.getDenseI64ArrayAttr({2}));
  ownership::aggregate::Slot::markLoad(descriptor, "async.exception_cell",
                                       "exception", std::nullopt);
  descriptor = builder.create<mlir::LLVM::InsertValueOp>(
      loc, memrefDescriptorType, descriptor, size,
      builder.getDenseI64ArrayAttr({3, 0}));
  ownership::aggregate::Slot::markLoad(descriptor, "async.exception_cell",
                                       "exception", std::nullopt);
  descriptor = builder.create<mlir::LLVM::InsertValueOp>(
      loc, memrefDescriptorType, descriptor, stride,
      builder.getDenseI64ArrayAttr({4, 0}));
  ownership::aggregate::Slot::markLoad(descriptor, "async.exception_cell",
                                       "exception", std::nullopt);
  (void)ptrType;
  (void)i64Type;
  return descriptor;
}

static mlir::Value partsDescriptorFromWords(mlir::Location loc,
                                            llvm::ArrayRef<mlir::Value> words,
                                            mlir::OpBuilder &builder) {
  if (words.size() != kExceptionPayloadWordCount)
    return {};
  mlir::MLIRContext *ctx = builder.getContext();
  auto partsType = partsDescriptorType(ctx);
  mlir::Value aggregate = builder.create<mlir::LLVM::UndefOp>(loc, partsType);
  for (unsigned index = 0; index < kExceptionCellDescriptorCount; ++index) {
    unsigned base = index * kExceptionCellDescriptorWords;
    mlir::Value descriptor =
        descriptorFromWords(loc, words[base + kExceptionPayloadAlignedSlot],
                            words[base + kExceptionPayloadAllocatedSlot],
                            words[base + kExceptionPayloadOffsetSlot],
                            words[base + kExceptionPayloadSizeSlot],
                            words[base + kExceptionPayloadStrideSlot], builder);
    aggregate = builder.create<mlir::LLVM::InsertValueOp>(
        loc, partsType, aggregate, descriptor,
        builder.getDenseI64ArrayAttr({index}));
    ownership::aggregate::Slot::markLoad(aggregate, "async.exception_cell",
                                         "exception", std::nullopt);
  }
  return aggregate;
}

static bool partsDescriptorParts(mlir::Location loc, mlir::Value exception,
                                 mlir::OpBuilder &builder,
                                 llvm::SmallVectorImpl<mlir::Value> &parts) {
  parts.clear();
  if (!exception)
    return false;

  if (auto cast = exception.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
    if (cast->getNumOperands() == kExceptionCellDescriptorCount) {
      for (auto [index, operand] : llvm::enumerate(cast.getOperands()))
        if (!isExceptionPartType(static_cast<unsigned>(index),
                                 operand.getType()))
          return false;
      parts.append(cast.getOperands().begin(), cast.getOperands().end());
      return true;
    }
  }

  auto aggregate =
      mlir::dyn_cast<mlir::LLVM::LLVMStructType>(exception.getType());
  if (!isPartsDescriptorType(aggregate))
    return false;
  for (unsigned index = 0; index < kExceptionCellDescriptorCount; ++index) {
    mlir::Value part = builder.create<mlir::LLVM::ExtractValueOp>(
        loc, aggregate.getBody()[index], exception,
        builder.getDenseI64ArrayAttr({index}));
    ownership::aggregate::Slot::markLoad(part, "async.exception_cell",
                                         "exception", index);
    parts.push_back(part);
  }
  return true;
}

} // namespace exception_payload

void async_runtime::ExceptionCell::mark(mlir::Value value) {
  if (!value)
    return;
  mlir::Operation *def = value.getDefiningOp();
  if (!def)
    return;
  def->setAttr(AsyncSafetyAttrs::kExceptionCell,
               mlir::UnitAttr::get(def->getContext()));
}

void async_runtime::CancelFlag::mark(mlir::Value value) {
  if (!value)
    return;
  mlir::Operation *def = value.getDefiningOp();
  if (!def)
    return;
  def->setAttr(AsyncSafetyAttrs::kCancelFlag,
               mlir::UnitAttr::get(def->getContext()));
}

void async_runtime::RuntimeHandle::mark(mlir::Value value) {
  if (!value)
    return;
  mlir::Operation *def = value.getDefiningOp();
  if (!def)
    return;
  def->setAttr(AsyncSafetyAttrs::kRuntimeHandle,
               mlir::UnitAttr::get(def->getContext()));
}

static void markFunctionArgument(mlir::Operation *funcLike, unsigned argIndex,
                                 llvm::StringRef attrName) {
  if (!funcLike)
    return;
  auto function = mlir::dyn_cast<mlir::FunctionOpInterface>(funcLike);
  if (function && argIndex < function.getNumArguments()) {
    function.setArgAttr(argIndex, attrName,
                        mlir::UnitAttr::get(funcLike->getContext()));
    return;
  }

  if (funcLike->getNumRegions() == 0 || funcLike->getRegion(0).empty())
    return;
  unsigned numArgs = funcLike->getRegion(0).front().getNumArguments();
  if (argIndex >= numArgs)
    return;

  mlir::Builder builder(funcLike->getContext());
  auto existing = funcLike->getAttrOfType<mlir::ArrayAttr>("arg_attrs");
  llvm::SmallVector<mlir::Attribute> attrs;
  attrs.reserve(numArgs);
  for (unsigned index = 0; index < numArgs; ++index) {
    if (existing && index < existing.size())
      attrs.push_back(existing[index]);
    else
      attrs.push_back(builder.getDictionaryAttr({}));
  }

  auto dict = mlir::dyn_cast<mlir::DictionaryAttr>(attrs[argIndex]);
  mlir::NamedAttrList named(dict ? dict : builder.getDictionaryAttr({}));
  named.set(attrName, builder.getUnitAttr());
  attrs[argIndex] = named.getDictionary(funcLike->getContext());
  funcLike->setAttr("arg_attrs", builder.getArrayAttr(attrs));
}

void async_runtime::ExceptionCell::markArgument(mlir::Operation *funcLike,
                                                unsigned argIndex) {
  markFunctionArgument(funcLike, argIndex, AsyncSafetyAttrs::kExceptionCell);
}

static bool
hasExceptionCellPointerProvenance(mlir::Value value,
                                  llvm::SmallPtrSetImpl<mlir::Value> &seen) {
  if (!value || !seen.insert(value).second)
    return false;
  if (hasFunctionArgAttr(value, AsyncSafetyAttrs::kExceptionCell))
    return true;
  mlir::Operation *def = value.getDefiningOp();
  if (!def)
    return false;
  if (def->hasAttr(AsyncSafetyAttrs::kExceptionCell) ||
      def->hasAttr(AsyncSafetyAttrs::kExceptionCellAllocated))
    return true;
  if (auto cast = mlir::dyn_cast<mlir::UnrealizedConversionCastOp>(def)) {
    return cast->getNumOperands() == 1 &&
           hasExceptionCellPointerProvenance(cast.getOperand(0), seen);
  }
  if (auto bitcast = mlir::dyn_cast<mlir::LLVM::BitcastOp>(def))
    return hasExceptionCellPointerProvenance(bitcast.getArg(), seen);
  if (auto extract = mlir::dyn_cast<mlir::LLVM::ExtractValueOp>(def))
    return hasExceptionCellPointerProvenance(extract.getContainer(), seen);
  return false;
}

static bool
hasExceptionCellProvenance(mlir::Value value,
                           llvm::SmallPtrSetImpl<mlir::Value> &seen) {
  if (!value)
    return false;
  if (!seen.insert(value).second)
    return false;
  if (!mlir::isa<ExceptionCellType>(value.getType()) &&
      !async_runtime::isExceptionCellType(value.getType()) &&
      !async_runtime::isLoweredExceptionCellType(value.getType()))
    return false;
  if (hasFunctionArgAttr(value, AsyncSafetyAttrs::kExceptionCell))
    return true;
  mlir::Operation *def = value.getDefiningOp();
  if (!def)
    return false;
  if (def->hasAttr(AsyncSafetyAttrs::kExceptionCell) ||
      def->hasAttr(AsyncSafetyAttrs::kExceptionCellAllocated))
    return true;
  if (auto cast = mlir::dyn_cast<mlir::memref::CastOp>(def))
    return hasExceptionCellProvenance(cast.getSource(), seen);
  if (auto cast = mlir::dyn_cast<mlir::UnrealizedConversionCastOp>(def)) {
    return cast->getNumOperands() == 1 &&
           hasExceptionCellProvenance(cast.getOperand(0), seen);
  }
  if (auto insert = mlir::dyn_cast<mlir::LLVM::InsertValueOp>(def)) {
    llvm::SmallPtrSet<mlir::Value, 8> pointerSeen;
    return hasExceptionCellProvenance(insert.getContainer(), seen) ||
           hasExceptionCellProvenance(insert.getValue(), seen) ||
           hasExceptionCellPointerProvenance(insert.getValue(), pointerSeen);
  }
  if (auto extract = mlir::dyn_cast<mlir::LLVM::ExtractValueOp>(def))
    return hasExceptionCellProvenance(extract.getContainer(), seen);
  return false;
}

bool async_runtime::ExceptionCell::hasProvenance(mlir::Value value) {
  llvm::SmallPtrSet<mlir::Value, 8> seen;
  return hasExceptionCellProvenance(value, seen);
}

void async_runtime::CancelFlag::markArgument(mlir::Operation *funcLike,
                                             unsigned argIndex) {
  markFunctionArgument(funcLike, argIndex, AsyncSafetyAttrs::kCancelFlag);
}

void async_runtime::RuntimeHandle::markArgument(mlir::Operation *funcLike,
                                                unsigned argIndex) {
  markFunctionArgument(funcLike, argIndex, AsyncSafetyAttrs::kRuntimeHandle);
}

static mlir::Value sourceExceptionCellMemRef(mlir::Value cell) {
  if (async_runtime::ExceptionCell::hasProvenance(cell) &&
      async_runtime::isExceptionCellType(cell.getType()))
    return cell;
  auto cast = cell.getDefiningOp<mlir::UnrealizedConversionCastOp>();
  if (!cast || cast->getNumOperands() != 1)
    return {};
  mlir::Value source = cast.getOperand(0);
  return async_runtime::ExceptionCell::hasProvenance(source) &&
                 async_runtime::isExceptionCellType(source.getType())
             ? source
             : mlir::Value();
}

mlir::Value async_runtime::ExceptionCell::load(mlir::Location loc,
                                               mlir::Value cell,
                                               mlir::RewriterBase &rewriter) {
  mlir::Value memrefCell = sourceExceptionCellMemRef(cell);
  auto loadSlot = [&](unsigned slot) -> mlir::Value {
    if (memrefCell)
      return loadExceptionCellMemRefSlot(loc, memrefCell, slot, rewriter);
    mlir::Value address = getExceptionCellAddress(loc, cell, rewriter);
    return loadExceptionCellSlot(loc, address, slot, rewriter);
  };

  if (memrefCell || async_runtime::ExceptionCell::hasProvenance(cell)) {
    mlir::Value state = loadSlot(kExceptionCellStateSlot);
    mlir::Value empty = rewriter.create<mlir::arith::ConstantIntOp>(
        loc, kExceptionCellEmpty, 64);
    mlir::Value publishing = rewriter.create<mlir::arith::ConstantIntOp>(
        loc, kExceptionCellPublishing, 64);
    mlir::Value notEmpty = rewriter.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::ne, state, empty);
    mlir::Value notPublishing = rewriter.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::ne, state, publishing);
    mlir::Value isPublished =
        rewriter.create<mlir::arith::AndIOp>(loc, notEmpty, notPublishing);
    auto descriptorType =
        exception_payload::partsDescriptorType(rewriter.getContext());

    mlir::Value zero = rewriter.create<mlir::arith::ConstantIntOp>(loc, 0, 64);
    llvm::SmallVector<mlir::Value, kExceptionPayloadWordCount> nullWords(
        kExceptionPayloadWordCount, zero);
    mlir::Value nullDescriptor =
        exception_payload::partsDescriptorFromWords(loc, nullWords, rewriter);

    mlir::Block *currentBlock = rewriter.getInsertionBlock();
    mlir::Block *afterBlock =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    afterBlock->addArgument(descriptorType, loc);
    mlir::Block *publishedBlock = rewriter.createBlock(
        afterBlock->getParent(), afterBlock->getIterator());

    rewriter.setInsertionPointToEnd(currentBlock);
    rewriter.create<mlir::cf::CondBranchOp>(loc, isPublished, publishedBlock,
                                            mlir::ValueRange{}, afterBlock,
                                            mlir::ValueRange{nullDescriptor});

    rewriter.setInsertionPointToStart(publishedBlock);
    llvm::SmallVector<mlir::Value, kExceptionPayloadWordCount> words;
    words.resize(kExceptionPayloadWordCount);
    // The acquire state load is the synchronization point.  It also carries
    // the published root header pointer, so keep it as the canonical aligned
    // word and load the remaining payload descriptor words from the payload
    // area.
    words[kExceptionPayloadAlignedSlot] = state;
    for (unsigned word = 1; word < kExceptionPayloadWordCount; ++word) {
      words[word] = loadSlot(kExceptionCellPayloadBaseSlot + word - 1);
    }
    mlir::Value descriptor =
        exception_payload::partsDescriptorFromWords(loc, words, rewriter);
    rewriter.create<mlir::cf::BranchOp>(loc, afterBlock,
                                        mlir::ValueRange{descriptor});

    rewriter.setInsertionPointToStart(afterBlock);
    return afterBlock->getArgument(afterBlock->getNumArguments() - 1);
  }
  mlir::emitError(loc)
      << "async exception cell load requires memref descriptor ABI";
  return {};
}

mlir::Value async_runtime::ExceptionCell::isNull(mlir::Location loc,
                                                 mlir::Value exception,
                                                 mlir::OpBuilder &builder) {
  auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  if (auto descriptor =
          mlir::dyn_cast<mlir::LLVM::LLVMStructType>(exception.getType())) {
    if (exception_payload::isPartsDescriptorType(descriptor)) {
      mlir::Value root = builder.create<mlir::LLVM::ExtractValueOp>(
          loc, descriptor.getBody()[0], exception,
          builder.getDenseI64ArrayAttr({0}));
      return isNull(loc, root, builder);
    }
    if (!object_abi::Type::isLoweredStorage(descriptor))
      return {};
    mlir::Value aligned = builder.create<mlir::LLVM::ExtractValueOp>(
        loc, descriptor.getBody()[1], exception,
        builder.getDenseI64ArrayAttr({1}));
    mlir::Value nullPtr = builder.create<mlir::LLVM::ZeroOp>(loc, ptrType);
    return builder.create<mlir::LLVM::ICmpOp>(
        loc, mlir::LLVM::ICmpPredicate::eq, aligned, nullPtr);
  }
  return {};
}

mlir::LogicalResult async_runtime::ExceptionCell::releaseLoaded(
    mlir::Location loc, mlir::ModuleOp module, mlir::RewriterBase &rewriter,
    const PyLLVMTypeConverter &typeConverter, mlir::Value exception) {
  mlir::Value isNull =
      async_runtime::ExceptionCell::isNull(loc, exception, rewriter);
  if (!isNull) {
    if (mlir::failed(releasePayload(loc, module, rewriter, typeConverter,
                                    exception,
                                    /*aggregateLoaded=*/true)))
      return mlir::failure();
    return mlir::success();
  }

  mlir::Block *currentBlock = rewriter.getInsertionBlock();
  mlir::Block *afterBlock =
      rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
  mlir::Block *releaseBlock =
      rewriter.createBlock(afterBlock->getParent(), afterBlock->getIterator());
  rewriter.setInsertionPointToEnd(currentBlock);
  rewriter.create<mlir::cf::CondBranchOp>(loc, isNull, afterBlock,
                                          releaseBlock);

  rewriter.setInsertionPointToStart(releaseBlock);
  if (mlir::failed(releasePayload(loc, module, rewriter, typeConverter,
                                  exception,
                                  /*aggregateLoaded=*/true)))
    return mlir::failure();
  rewriter.create<mlir::cf::BranchOp>(loc, afterBlock);

  rewriter.setInsertionPointToStart(afterBlock);
  return mlir::success();
}

mlir::LogicalResult async_runtime::ExceptionCell::retainPayload(
    mlir::Location loc, mlir::ModuleOp module, mlir::OpBuilder &builder,
    const PyLLVMTypeConverter &typeConverter, mlir::Value exception,
    llvm::StringRef retainPremise) {
  (void)module;
  (void)typeConverter;
  RuntimeAPI runtime(module, builder, typeConverter);
  llvm::SmallVector<mlir::Value, 3> parts;
  if (exception_payload::partsDescriptorParts(loc, exception, builder, parts)) {
    auto call = runtime.call(loc, RuntimeSymbols::kIncRef, mlir::Type(),
                             mlir::ValueRange{parts.front()});
    threadsafe::Retain::premise(call.getOperation(), retainPremise);
    return mlir::success();
  }
  mlir::emitError(loc)
      << "async exception payload retain requires parts descriptor ABI";
  return mlir::failure();
}

mlir::FailureOr<mlir::Operation *> async_runtime::ExceptionCell::releasePayload(
    mlir::Location loc, mlir::ModuleOp module, mlir::OpBuilder &builder,
    const PyLLVMTypeConverter &typeConverter, mlir::Value exception,
    bool aggregateLoaded) {
  RuntimeAPI runtime(module, builder, typeConverter);
  llvm::SmallVector<mlir::Value, 3> parts;
  if (exception_payload::partsDescriptorParts(loc, exception, builder, parts)) {
    auto call = runtime.call(loc, RuntimeSymbols::kExceptionDecRef,
                             mlir::Type(), parts);
    if (aggregateLoaded)
      call->setAttr(OwnershipContractAttrs::kAggregateRelease,
                    builder.getUnitAttr());
    return call.getOperation();
  }
  mlir::emitError(loc)
      << "async exception payload release requires parts descriptor ABI";
  return mlir::failure();
}

mlir::LogicalResult async_runtime::ExceptionCell::free(
    mlir::Location loc, mlir::ModuleOp module, mlir::OpBuilder &builder,
    const PyLLVMTypeConverter &typeConverter, mlir::Value exceptionCell) {
  (void)typeConverter;
  if (async_runtime::isExceptionCellType(exceptionCell.getType())) {
    auto dealloc = builder.create<mlir::memref::DeallocOp>(loc, exceptionCell);
    dealloc->setAttr(AsyncSafetyAttrs::kExceptionCellFree,
                     builder.getUnitAttr());
    return mlir::success();
  }
  module.emitError("async exception cell dealloc requires high-level memref "
                   "cell ABI before memref-to-LLVM conversion");
  return mlir::failure();
}

mlir::LogicalResult async_runtime::ExceptionCell::destroy(
    mlir::Location loc, mlir::ModuleOp module, mlir::RewriterBase &rewriter,
    const PyLLVMTypeConverter &typeConverter, mlir::Value exceptionCell) {
  mlir::Value exception = load(loc, exceptionCell, rewriter);
  if (mlir::failed(
          releaseLoaded(loc, module, rewriter, typeConverter, exception)))
    return mlir::failure();
  return free(loc, module, rewriter, typeConverter, exceptionCell);
}

mlir::LogicalResult async_runtime::ExceptionCell::destroyKnownEmpty(
    mlir::Location loc, mlir::ModuleOp module, mlir::OpBuilder &builder,
    const PyLLVMTypeConverter &typeConverter, mlir::Value exceptionCell) {
  assumeExceptionCellEmpty(loc, exceptionCell, builder);
  return free(loc, module, builder, typeConverter, exceptionCell);
}

mlir::LogicalResult async_runtime::ExceptionCell::storeFirst(
    mlir::Location loc, mlir::Value cell, mlir::Value exception,
    mlir::ModuleOp module, mlir::RewriterBase &rewriter,
    const PyLLVMTypeConverter &typeConverter, llvm::StringRef retainPremise) {
  llvm::SmallVector<mlir::Value, kExceptionPayloadWordCount> descriptorWords;
  exception_payload::appendDescriptorWords(loc, exception, descriptorWords,
                                           rewriter);
  mlir::Value stored =
      exception_payload::alignedPointer(loc, exception, rewriter);
  if (!stored) {
    mlir::emitError(loc) << "async exception cell payload is not a memref "
                            "exception storage value";
    return mlir::failure();
  }

  if (async_runtime::ExceptionCell::hasProvenance(cell)) {
    if (descriptorWords.size() != kExceptionPayloadWordCount) {
      mlir::emitError(loc) << "async exception cell payload descriptor is "
                              "not available";
      return mlir::failure();
    }

    if (mlir::Value memrefCell = sourceExceptionCellMemRef(cell)) {
      mlir::Value empty = rewriter.create<mlir::arith::ConstantIntOp>(
          loc, kExceptionCellEmpty, 64);
      mlir::Value publishing = rewriter.create<mlir::arith::ConstantIntOp>(
          loc, kExceptionCellPublishing, 64);

      mlir::Value previous = transitionExceptionCellState(
          loc, memrefCell, empty, publishing, /*reservation=*/true, rewriter);
      mlir::Value reserved = rewriter.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::eq, previous, empty);

      mlir::Block *currentBlock = rewriter.getInsertionBlock();
      mlir::Block *afterBlock =
          rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
      mlir::Block *publishBlock = rewriter.createBlock(
          afterBlock->getParent(), afterBlock->getIterator());
      rewriter.setInsertionPointToEnd(currentBlock);
      rewriter.create<mlir::cf::CondBranchOp>(loc, reserved, publishBlock,
                                              afterBlock);

      rewriter.setInsertionPointToStart(publishBlock);
      if (mlir::failed(retainPayload(loc, module, rewriter, typeConverter,
                                     exception, retainPremise)))
        return mlir::failure();
      for (unsigned word = 1; word < descriptorWords.size(); ++word) {
        storeExceptionCellMemRefSlot(loc, memrefCell,
                                     kExceptionCellPayloadBaseSlot +
                                         static_cast<unsigned>(word - 1),
                                     descriptorWords[word], rewriter);
      }
      mlir::Value published = descriptorWords[kExceptionPayloadAlignedSlot];
      ownership::aggregate::Slot::copyLoad(exception, published);
      mlir::Value previousPublish =
          transitionExceptionCellState(loc, memrefCell, publishing, published,
                                       /*reservation=*/false, rewriter);
      mlir::Value publishSucceeded = rewriter.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::eq, previousPublish, publishing);
      mlir::Block *publishFailedBlock = rewriter.createBlock(
          afterBlock->getParent(), afterBlock->getIterator());
      rewriter.setInsertionPointAfter(publishSucceeded.getDefiningOp());
      rewriter.create<mlir::cf::CondBranchOp>(loc, publishSucceeded, afterBlock,
                                              publishFailedBlock);

      rewriter.setInsertionPointToStart(publishFailedBlock);
      if (mlir::failed(
              releasePayload(loc, module, rewriter, typeConverter, exception)))
        return mlir::failure();
      rewriter.create<mlir::cf::BranchOp>(loc, afterBlock);

      rewriter.setInsertionPointToStart(afterBlock);
      return mlir::success();
    }

    mlir::Value address = getExceptionCellAddress(loc, cell, rewriter);
    mlir::Value empty = rewriter.create<mlir::LLVM::ConstantOp>(
        loc, rewriter.getI64Type(),
        rewriter.getI64IntegerAttr(kExceptionCellEmpty));
    mlir::Value publishing = rewriter.create<mlir::LLVM::ConstantOp>(
        loc, rewriter.getI64Type(),
        rewriter.getI64IntegerAttr(kExceptionCellPublishing));

    auto reserve = rewriter.create<mlir::LLVM::AtomicCmpXchgOp>(
        loc, address, empty, publishing, mlir::LLVM::AtomicOrdering::acq_rel,
        mlir::LLVM::AtomicOrdering::acquire, llvm::StringRef(),
        /*alignment=*/8);
    threadsafe::Atomic::set(reserve.getOperation(),
                            ThreadSafetyAttrs::kRoleAsyncExceptionStore,
                            ThreadSafetyAttrs::kOrderingAcqRel);
    reserve->setAttr(AsyncSafetyAttrs::kExceptionCellReservation,
                     rewriter.getUnitAttr());
    mlir::Value reserved = rewriter.create<mlir::LLVM::ExtractValueOp>(
        loc, rewriter.getI1Type(), reserve.getRes(),
        rewriter.getDenseI64ArrayAttr({1}));

    mlir::Block *currentBlock = rewriter.getInsertionBlock();
    mlir::Block *afterBlock =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    mlir::Block *publishBlock = rewriter.createBlock(afterBlock->getParent(),
                                                     afterBlock->getIterator());
    rewriter.setInsertionPointToEnd(currentBlock);
    rewriter.create<mlir::cf::CondBranchOp>(loc, reserved, publishBlock,
                                            afterBlock);

    rewriter.setInsertionPointToStart(publishBlock);
    if (mlir::failed(retainPayload(loc, module, rewriter, typeConverter,
                                   exception, retainPremise)))
      return mlir::failure();
    for (unsigned word = 1; word < descriptorWords.size(); ++word) {
      storeExceptionCellSlot(loc, address,
                             kExceptionCellPayloadBaseSlot +
                                 static_cast<unsigned>(word - 1),
                             descriptorWords[word], rewriter);
    }
    mlir::Value published = descriptorWords[kExceptionPayloadAlignedSlot];
    ownership::aggregate::Slot::copyLoad(exception, published);
    auto publish = rewriter.create<mlir::LLVM::AtomicCmpXchgOp>(
        loc, address, publishing, published,
        mlir::LLVM::AtomicOrdering::acq_rel,
        mlir::LLVM::AtomicOrdering::acquire, llvm::StringRef(),
        /*alignment=*/8);
    threadsafe::Atomic::set(publish.getOperation(),
                            ThreadSafetyAttrs::kRoleAsyncExceptionStore,
                            ThreadSafetyAttrs::kOrderingAcqRel);
    mlir::Value publishSucceeded = rewriter.create<mlir::LLVM::ExtractValueOp>(
        loc, rewriter.getI1Type(), publish.getRes(),
        rewriter.getDenseI64ArrayAttr({1}));
    mlir::Block *publishFailedBlock = rewriter.createBlock(
        afterBlock->getParent(), afterBlock->getIterator());
    rewriter.setInsertionPointAfter(publishSucceeded.getDefiningOp());
    rewriter.create<mlir::cf::CondBranchOp>(loc, publishSucceeded, afterBlock,
                                            publishFailedBlock);

    rewriter.setInsertionPointToStart(publishFailedBlock);
    if (mlir::failed(
            releasePayload(loc, module, rewriter, typeConverter, exception)))
      return mlir::failure();
    rewriter.create<mlir::cf::BranchOp>(loc, afterBlock);

    rewriter.setInsertionPointToStart(afterBlock);
    return mlir::success();
  }

  mlir::emitError(loc)
      << "async exception cell store requires memref descriptor ABI";
  return mlir::failure();
}

static void copyContractAttr(mlir::Operation *from, mlir::Operation *to,
                             llvm::StringRef attrName) {
  if (mlir::Attribute attr = from->getAttr(attrName))
    to->setAttr(attrName, attr);
}

namespace container::access::Contract {
bool has(mlir::Operation *op) {
  return op && op->hasAttr(ContainerSafetyAttrs::kAccessGroup);
}

void copy(mlir::Operation *from, mlir::Operation *to) {
  copyContractAttr(from, to, ContainerSafetyAttrs::kAccessGroup);
  copyContractAttr(from, to, ContainerSafetyAttrs::kAccessComponent);
  copyContractAttr(from, to, OwnershipContractAttrs::kAggregateSlotLoad);
  copyContractAttr(from, to, OwnershipContractAttrs::kAggregateSlotGroup);
  copyContractAttr(from, to, OwnershipContractAttrs::kAggregateSlotComponent);
  copyContractAttr(from, to, OwnershipContractAttrs::kAggregateSlotIndex);
  copyContractAttr(from, to, OwnershipContractAttrs::kAggregateSlotPartIndex);
  copyContractAttr(from, to, OwnershipContractAttrs::kMemRefSlotTransfer);
  copyContractAttr(from, to, kMemRefContainerAccessContractId);
  copyContractAttr(from, to, kMemRefAggregateLoadContractId);
}
} // namespace container::access::Contract

static mlir::Value stripLLVMPointerCasts(mlir::Value value) {
  while (true) {
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
    if (auto cast = value.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
      if (cast->getNumOperands() == 1) {
        value = cast.getOperand(0);
        continue;
      }
    }
    return value;
  }
}

static std::optional<int64_t> getArithConstantInt(mlir::Value value) {
  if (auto constant = value.getDefiningOp<mlir::arith::ConstantIntOp>())
    return constant.value();
  if (auto constant = value.getDefiningOp<mlir::arith::ConstantOp>())
    if (auto attr = mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue()))
      return attr.getInt();
  return std::nullopt;
}

static std::optional<llvm::StringRef>
getLoweringValueDefStringAttr(mlir::Value value, llvm::StringRef attrName) {
  value = stripLLVMPointerCasts(value);
  mlir::Operation *def = value.getDefiningOp();
  if (!def)
    return std::nullopt;
  auto attr = def->getAttrOfType<mlir::StringAttr>(attrName);
  if (!attr)
    return std::nullopt;
  return attr.getValue();
}

static mlir::StringAttr inferContainerKindAttr(mlir::Value memref,
                                               mlir::OpBuilder &builder) {
  auto kind = container::Kind::nameFromHeader(memref);
  if (!kind)
    return {};
  return builder.getStringAttr(*kind);
}

static bool isArithConstantInt(mlir::Value value, int64_t expected) {
  if (auto constant = value.getDefiningOp<mlir::arith::ConstantIntOp>())
    return constant.value() == expected;
  if (auto constant = value.getDefiningOp<mlir::arith::ConstantOp>())
    if (auto attr = mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue()))
      return attr.getInt() == expected;
  return false;
}

static unsigned getLoweringDescriptorComponentCount(mlir::Value header) {
  header = stripLLVMPointerCasts(header);
  return container::Descriptor::componentCount(header);
}

static bool isLoweringDescriptorSiblingIndex(mlir::Value header,
                                             mlir::Value component) {
  header = stripLLVMPointerCasts(header);
  component = stripLLVMPointerCasts(component);
  unsigned count = getLoweringDescriptorComponentCount(header);
  if (count <= 1)
    return false;

  if (auto headerResult = mlir::dyn_cast<mlir::OpResult>(header)) {
    auto componentResult = mlir::dyn_cast<mlir::OpResult>(component);
    if (!componentResult ||
        headerResult.getOwner() != componentResult.getOwner())
      return false;
    unsigned headerNo = headerResult.getResultNumber();
    unsigned componentNo = componentResult.getResultNumber();
    return componentNo >= headerNo && componentNo < headerNo + count;
  }

  if (auto headerArg = mlir::dyn_cast<mlir::BlockArgument>(header)) {
    auto componentArg = mlir::dyn_cast<mlir::BlockArgument>(component);
    if (!componentArg || headerArg.getOwner() != componentArg.getOwner())
      return false;
    unsigned headerNo = headerArg.getArgNumber();
    unsigned componentNo = componentArg.getArgNumber();
    return componentNo >= headerNo && componentNo < headerNo + count;
  }

  return false;
}

static bool sameLoweringContainerDescriptorResource(mlir::Value header,
                                                    mlir::Value component) {
  header = stripLLVMPointerCasts(header);
  component = stripLLVMPointerCasts(component);
  if (!header || !component)
    return false;
  if (header == component)
    return true;
  auto headerGroup =
      getValueDefStringAttr(header, ContainerSafetyAttrs::kDescriptorGroup);
  auto componentGroup =
      getValueDefStringAttr(component, ContainerSafetyAttrs::kDescriptorGroup);
  if (headerGroup && componentGroup)
    return *headerGroup == *componentGroup;
  return isLoweringDescriptorSiblingIndex(header, component);
}

static unsigned getFinalLLVMFunctionInputCount(mlir::Type inputType) {
  if (auto memref = mlir::dyn_cast<mlir::MemRefType>(inputType))
    return 3 + 2 * static_cast<unsigned>(memref.getRank());
  if (mlir::isa<mlir::UnrankedMemRefType>(inputType))
    return 2;
  return 1;
}

static bool hasEmitCInterface(mlir::func::FuncOp func) {
  return func->hasAttr(mlir::LLVM::LLVMDialect::getEmitCWrapperAttrName());
}

static std::string cInterfaceName(llvm::StringRef name) {
  return ("_mlir_ciface_" + name).str();
}

static mlir::Attribute getFuncArgAttr(mlir::func::FuncOp func,
                                      unsigned argIndex,
                                      llvm::StringRef attrName) {
  if (argIndex >= func.getNumArguments())
    return {};
  return func.getArgAttr(argIndex, attrName);
}

static std::optional<unsigned>
flattenedInputCount(mlir::Type inputType,
                    const PyLLVMTypeConverter *typeConverter,
                    bool preserveUnrankedMemRefAbi) {
  if (!typeConverter)
    return getFinalLLVMFunctionInputCount(inputType);

  llvm::SmallVector<mlir::Type, 4> converted;
  if (mlir::failed(typeConverter->convertType(inputType, converted)) ||
      converted.empty())
    return std::nullopt;
  if (mlir::isa<mlir::MemRefType>(inputType) ||
      (preserveUnrankedMemRefAbi &&
       mlir::isa<mlir::UnrankedMemRefType>(inputType)))
    return getFinalLLVMFunctionInputCount(inputType);
  return static_cast<unsigned>(converted.size());
}

template <typename Fn>
static void forEachFlattenedFuncArg(mlir::ModuleOp module,
                                    const PyLLVMTypeConverter *typeConverter,
                                    bool preserveUnrankedMemRefAbi,
                                    Fn callback) {
  module.walk([&](mlir::func::FuncOp func) {
    llvm::StringRef name = func.getSymName();
    unsigned flattenedIndex = 0;
    for (unsigned arg = 0, e = func.getFunctionType().getNumInputs(); arg < e;
         ++arg) {
      mlir::Type inputType = func.getFunctionType().getInput(arg);
      std::optional<unsigned> inputCount = flattenedInputCount(
          inputType, typeConverter, preserveUnrankedMemRefAbi);
      if (!inputCount)
        return;
      callback(name, func, arg, flattenedIndex, inputType);
      flattenedIndex += *inputCount;
    }
  });
}

static void collectMemRefNonObjectArgs(
    llvm::StringRef name, mlir::func::FuncOp func, unsigned sourceArgIndex,
    unsigned flattenedIndex, mlir::Type inputType,
    llvm::SmallVectorImpl<NonObjectArgContract> &contracts) {
  if (mlir::isa<mlir::MemRefType>(inputType)) {
    contracts.push_back({name.str(), flattenedIndex});
    contracts.push_back({name.str(), flattenedIndex + 1});
    if (hasEmitCInterface(func))
      contracts.push_back({cInterfaceName(name), sourceArgIndex});
    return;
  }
  if (mlir::isa<mlir::UnrankedMemRefType>(inputType)) {
    contracts.push_back({name.str(), flattenedIndex + 1});
    if (hasEmitCInterface(func))
      contracts.push_back({cInterfaceName(name), sourceArgIndex});
  }
}

static void collectAsyncArgProvenance(
    llvm::StringRef name, mlir::func::FuncOp func, unsigned sourceArgIndex,
    unsigned flattenedIndex, mlir::Type inputType,
    llvm::SmallVectorImpl<AsyncArgProvenanceContract> &contracts) {
  if (getFuncArgAttr(func, sourceArgIndex, AsyncSafetyAttrs::kRuntimeHandle))
    contracts.push_back(
        {name.str(), flattenedIndex, AsyncArgProvenanceKind::RuntimeHandle});
  if (getFuncArgAttr(func, sourceArgIndex, AsyncSafetyAttrs::kExceptionCell))
    contracts.push_back(
        {name.str(), flattenedIndex, AsyncArgProvenanceKind::ExceptionCell});
  if (getFuncArgAttr(func, sourceArgIndex, AsyncSafetyAttrs::kCancelFlag)) {
    unsigned targetIndex = flattenedIndex;
    if (mlir::isa<mlir::MemRefType>(inputType))
      targetIndex = flattenedIndex + 1;
    contracts.push_back(
        {name.str(), targetIndex, AsyncArgProvenanceKind::CancelFlag});
  }
}

void collectAsyncArgProvenanceContracts(
    mlir::ModuleOp module,
    llvm::SmallVectorImpl<AsyncArgProvenanceContract> &contracts) {
  forEachFlattenedFuncArg(
      module, /*typeConverter=*/nullptr, /*preserveUnrankedMemRefAbi=*/false,
      [&](llvm::StringRef name, mlir::func::FuncOp func, unsigned arg,
          unsigned flattenedIndex, mlir::Type inputType) {
        collectAsyncArgProvenance(name, func, arg, flattenedIndex, inputType,
                                  contracts);
      });
}

void collectAsyncArgProvenanceContracts(
    mlir::ModuleOp module, const PyLLVMTypeConverter &typeConverter,
    llvm::SmallVectorImpl<AsyncArgProvenanceContract> &contracts) {
  forEachFlattenedFuncArg(
      module, &typeConverter, /*preserveUnrankedMemRefAbi=*/false,
      [&](llvm::StringRef name, mlir::func::FuncOp func, unsigned arg,
          unsigned flattenedIndex, mlir::Type inputType) {
        collectAsyncArgProvenance(name, func, arg, flattenedIndex, inputType,
                                  contracts);
      });
}

void collectNonObjectArgContracts(
    mlir::ModuleOp module,
    llvm::SmallVectorImpl<NonObjectArgContract> &contracts) {
  forEachFlattenedFuncArg(
      module, /*typeConverter=*/nullptr, /*preserveUnrankedMemRefAbi=*/false,
      [&](llvm::StringRef name, mlir::func::FuncOp func, unsigned arg,
          unsigned flattenedIndex, mlir::Type inputType) {
        collectMemRefNonObjectArgs(name, func, arg, flattenedIndex, inputType,
                                   contracts);
      });
}

void collectNonObjectArgContracts(
    mlir::ModuleOp module, const PyLLVMTypeConverter &typeConverter,
    llvm::SmallVectorImpl<NonObjectArgContract> &contracts) {
  forEachFlattenedFuncArg(
      module, &typeConverter, /*preserveUnrankedMemRefAbi=*/true,
      [&](llvm::StringRef name, mlir::func::FuncOp func, unsigned arg,
          unsigned flattenedIndex, mlir::Type inputType) {
        collectMemRefNonObjectArgs(name, func, arg, flattenedIndex, inputType,
                                   contracts);
      });
}

static mlir::Operation *lookupFuncLike(mlir::ModuleOp module,
                                       llvm::StringRef symbolName) {
  if (auto func = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(symbolName))
    return func.getOperation();
  if (auto func = module.lookupSymbol<mlir::func::FuncOp>(symbolName))
    return func.getOperation();
  if (auto func = module.lookupSymbol<mlir::async::FuncOp>(symbolName))
    return func.getOperation();
  return nullptr;
}

template <typename Contract, typename Apply>
static mlir::LogicalResult
preserveFuncArgContracts(mlir::ModuleOp module,
                         llvm::ArrayRef<Contract> contracts,
                         llvm::StringRef label, Apply apply) {
  bool failedAny = false;
  for (const Contract &contract : contracts) {
    mlir::Operation *funcLike = lookupFuncLike(module, contract.symbolName);
    if (!funcLike) {
      mlir::emitError(module.getLoc())
          << label << " for @" << contract.symbolName
          << " was not preserved through LLVM lowering";
      failedAny = true;
      continue;
    }
    apply(funcLike, contract);
  }
  return mlir::failure(failedAny);
}

mlir::LogicalResult preserveLLVMNonObjectArgContracts(
    mlir::ModuleOp module, llvm::ArrayRef<NonObjectArgContract> contracts) {
  return preserveFuncArgContracts(
      module, contracts, "non-object argument contract",
      [&](mlir::Operation *funcLike, const NonObjectArgContract &contract) {
        markFunctionArgument(funcLike, contract.argIndex,
                             OwnershipContractAttrs::kNonObjectPointer);

        module.walk([&](mlir::LLVM::CallOp call) {
          auto callee = call.getCallee();
          if (!callee || *callee != contract.symbolName)
            return;
          if (contract.argIndex >= call.getNumOperands())
            return;
          mlir::Value operand = call.getCalleeOperands()[contract.argIndex];
          if (mlir::isa<mlir::LLVM::LLVMPointerType>(operand.getType()))
            ownership::Pointer::markNonObject(operand);
        });
      });
}

mlir::LogicalResult preserveLLVMAsyncArgProvenanceContracts(
    mlir::ModuleOp module,
    llvm::ArrayRef<AsyncArgProvenanceContract> contracts) {
  return preserveFuncArgContracts(
      module, contracts, "async argument provenance contract",
      [](mlir::Operation *funcLike,
         const AsyncArgProvenanceContract &contract) {
        switch (contract.kind) {
        case AsyncArgProvenanceKind::RuntimeHandle:
          async_runtime::RuntimeHandle::markArgument(funcLike,
                                                     contract.argIndex);
          break;
        case AsyncArgProvenanceKind::ExceptionCell:
          async_runtime::ExceptionCell::markArgument(funcLike,
                                                     contract.argIndex);
          break;
        case AsyncArgProvenanceKind::CancelFlag:
          async_runtime::CancelFlag::markArgument(funcLike, contract.argIndex);
          break;
        }
      });
}

void collectMemRefAtomicContracts(
    mlir::ModuleOp module,
    llvm::SmallVectorImpl<MemRefAtomicContract> &contracts) {
  mlir::OpBuilder builder(module.getContext());
  int64_t nextId = 0;
  module.walk([&](mlir::memref::AtomicRMWOp atomic) {
    auto role =
        atomic->getAttrOfType<mlir::StringAttr>(ThreadSafetyAttrs::kAtomicRole);
    auto ordering = atomic->getAttrOfType<mlir::StringAttr>(
        ThreadSafetyAttrs::kAtomicOrdering);
    if (!role || !ordering)
      return;

    std::optional<int64_t> value = getArithConstantInt(atomic.getValue());
    if (!value)
      return;

    auto premise = atomic->getAttrOfType<mlir::StringAttr>(
        ThreadSafetyAttrs::kRetainPremise);
    auto group = atomic->getAttrOfType<mlir::StringAttr>(
        ThreadSafetyAttrs::kAtomicMemRefGroup);
    if (!group)
      if (auto groupName = getLoweringValueDefStringAttr(
              atomic.getMemref(), ContainerSafetyAttrs::kDescriptorGroup))
        group = builder.getStringAttr(*groupName);
    auto containerKind = atomic->getAttrOfType<mlir::StringAttr>(
        ThreadSafetyAttrs::kAtomicContainerKind);
    if (!containerKind)
      if (auto kindName = getLoweringValueDefStringAttr(
              atomic.getMemref(), ContainerSafetyAttrs::kDescriptorKind))
        containerKind = builder.getStringAttr(*kindName);
    if (!containerKind)
      containerKind = inferContainerKindAttr(atomic.getMemref(), builder);
    auto component = atomic->getAttrOfType<mlir::StringAttr>(
        ThreadSafetyAttrs::kAtomicMemRefComponent);
    if (!component)
      if (auto componentName = getLoweringValueDefStringAttr(
              atomic.getMemref(), ContainerSafetyAttrs::kDescriptorComponent))
        component = builder.getStringAttr(*componentName);
    std::optional<int64_t> slot;
    if (atomic.getIndices().size() == 1)
      slot = getArithConstantInt(atomic.getIndices().front());
    int64_t id = nextId++;
    atomic->setAttr(kMemRefAtomicContractId, builder.getI64IntegerAttr(id));
    contracts.push_back(MemRefAtomicContract{
        id, atomic.getLoc(), atomic.getKind(), *value, role, ordering, premise,
        group, containerKind, component, slot,
        atomic->hasAttr(ThreadSafetyAttrs::kOwnedTokenVerified),
        atomic->getAttrOfType<mlir::StringAttr>(
            ThreadSafetyAttrs::kOwnedTokenProof)});
  });
}

template <typename Contract> class ContractPreserver {
public:
  ContractPreserver(llvm::ArrayRef<Contract> contracts,
                    llvm::StringRef missingMessage)
      : contracts(contracts), used(contracts.size(), false),
        missingMessage(missingMessage) {}

  template <typename Accept, typename Validate, typename Apply>
  void apply(mlir::Operation *op, llvm::StringRef idAttrName,
             llvm::StringRef unknownMessage, Accept accept, Validate validate,
             Apply materialize) {
    auto idAttr = op->getAttrOfType<mlir::IntegerAttr>(idAttrName);
    if (!idAttr)
      return;

    int64_t id = idAttr.getInt();
    for (auto indexed : llvm::enumerate(contracts)) {
      const Contract &contract = indexed.value();
      if (contract.id != id || !accept(contract))
        continue;
      if (mlir::failed(validate(contract))) {
        failedAny = true;
        return;
      }
      materialize(contract);
      used[indexed.index()] = true;
      return;
    }
    op->emitOpError(unknownMessage);
    failedAny = true;
  }

  mlir::LogicalResult finish() {
    for (auto indexed : llvm::enumerate(contracts)) {
      if (used[indexed.index()])
        continue;
      mlir::emitError(indexed.value().location) << missingMessage;
      failedAny = true;
    }
    return mlir::failure(failedAny);
  }

private:
  llvm::ArrayRef<Contract> contracts;
  llvm::SmallVector<bool> used;
  llvm::StringRef missingMessage;
  bool failedAny = false;
};

mlir::LogicalResult preserveLoweredMemRefAtomicContracts(
    mlir::ModuleOp module, llvm::ArrayRef<MemRefAtomicContract> contracts) {
  auto applyContract = [](mlir::LLVM::AtomicRMWOp atomic,
                          const MemRefAtomicContract &contract) {
    threadsafe::Atomic::set(atomic.getOperation(), contract.role.getValue(),
                            contract.ordering.getValue(),
                            contract.retainPremise
                                ? contract.retainPremise.getValue()
                                : llvm::StringRef{});
    llvm::StringRef component =
        contract.component ? contract.component.getValue() : llvm::StringRef{};
    bool isContainerAtomic = role::containerAtomic(contract.role.getValue());
    bool isObjectAtomic = role::objectAtomic(contract.role.getValue());
    bool isClassAtomic = role::classAtomic(contract.role.getValue());
    if (component.empty() && isContainerAtomic)
      component = ContainerSafetyAttrs::kComponentHeader;
    if (component.empty() && isClassAtomic)
      component = ContainerSafetyAttrs::kComponentHeader;
    if (isContainerAtomic) {
      threadsafe::memref::Atomic::set(
          atomic.getOperation(), component, contract.slot,
          contract.group ? contract.group.getValue() : llvm::StringRef{},
          contract.containerKind ? contract.containerKind.getValue()
                                 : llvm::StringRef{});
    }
    if (isObjectAtomic)
      threadsafe::memref::Atomic::set(atomic.getOperation(), component,
                                      contract.slot, llvm::StringRef{},
                                      llvm::StringRef{});
    if (isClassAtomic)
      threadsafe::memref::Atomic::set(
          atomic.getOperation(), component, contract.slot,
          contract.group ? contract.group.getValue() : llvm::StringRef{},
          llvm::StringRef{});
    if (isContainerAtomic || isObjectAtomic || isClassAtomic) {
      mlir::OpBuilder builder(atomic.getContext());
      atomic->setAttr(ThreadSafetyAttrs::kAtomicProvenance,
                      builder.getStringAttr(
                          ThreadSafetyAttrs::kProvenanceMemRefDescriptor));
    }
    if (contract.role.getValue() ==
        ThreadSafetyAttrs::kRoleAsyncCancelRequest) {
      atomic->setAttr(AsyncSafetyAttrs::kCancelFlag,
                      mlir::UnitAttr::get(atomic.getContext()));
      async_runtime::CancelFlag::mark(atomic.getPtr());
    }
    if (contract.ownedTokenVerified)
      threadsafe::Retain::verifyOwnedToken(
          atomic.getOperation(),
          contract.ownedTokenProof
              ? contract.ownedTokenProof.getValue()
              : llvm::StringRef(ThreadSafetyAttrs::kProofOwnershipVerifier));
    atomic->removeAttr(kMemRefAtomicContractId);
  };

  ContractPreserver<MemRefAtomicContract> preserver(
      contracts, "memref atomic contract was not preserved through LLVM "
                 "lowering");
  module.walk([&](mlir::LLVM::AtomicRMWOp atomic) {
    preserver.apply(
        atomic.getOperation(), kMemRefAtomicContractId,
        "carries an unknown memref atomic contract id",
        [](const MemRefAtomicContract &) { return true; },
        [](const MemRefAtomicContract &) { return mlir::success(); },
        [&](const MemRefAtomicContract &contract) {
          applyContract(atomic, contract);
        });
  });
  return preserver.finish();
}

void collectMemRefAggregateLoadContracts(
    mlir::ModuleOp module,
    llvm::SmallVectorImpl<MemRefAggregateLoadContract> &contracts) {
  mlir::OpBuilder builder(module.getContext());
  int64_t nextId = 0;
  auto collect = [&](mlir::Operation *op, mlir::Value memref,
                     mlir::ValueRange indices) {
    if (!op->hasAttr(OwnershipContractAttrs::kAggregateSlotLoad))
      return;
    auto group = op->getAttrOfType<mlir::StringAttr>(
        OwnershipContractAttrs::kAggregateSlotGroup);
    if (!group)
      if (auto groupName = getLoweringValueDefStringAttr(
              memref, ContainerSafetyAttrs::kDescriptorGroup))
        group = builder.getStringAttr(*groupName);
    auto component = op->getAttrOfType<mlir::StringAttr>(
        OwnershipContractAttrs::kAggregateSlotComponent);
    if (!component)
      if (auto componentName = getLoweringValueDefStringAttr(
              memref, ContainerSafetyAttrs::kDescriptorComponent))
        component = builder.getStringAttr(*componentName);
    std::optional<int64_t> slot;
    if (auto slotAttr = op->getAttrOfType<mlir::IntegerAttr>(
            OwnershipContractAttrs::kAggregateSlotIndex))
      slot = slotAttr.getInt();
    else if (indices.size() == 1)
      slot = getArithConstantInt(indices.front());
    int64_t id = nextId++;
    op->setAttr(kMemRefAggregateLoadContractId, builder.getI64IntegerAttr(id));
    contracts.push_back(
        MemRefAggregateLoadContract{id, op->getLoc(), group, component, slot});
  };

  module.walk([&](mlir::memref::LoadOp load) {
    collect(load.getOperation(), load.getMemref(), load.getIndices());
  });
  module.walk([&](mlir::memref::ExtractAlignedPointerAsIndexOp extract) {
    collect(extract.getOperation(), extract.getSource(), {});
  });
}

mlir::LogicalResult preserveLoweredMemRefAggregateLoadContracts(
    mlir::ModuleOp module,
    llvm::ArrayRef<MemRefAggregateLoadContract> contracts) {
  auto applyContract = [](mlir::Operation *op,
                          const MemRefAggregateLoadContract &contract) {
    mlir::OpBuilder builder(op->getContext());
    op->setAttr(OwnershipContractAttrs::kAggregateSlotLoad,
                builder.getUnitAttr());
    if (contract.group)
      op->setAttr(OwnershipContractAttrs::kAggregateSlotGroup, contract.group);
    if (contract.component)
      op->setAttr(OwnershipContractAttrs::kAggregateSlotComponent,
                  contract.component);
    if (contract.slot)
      op->setAttr(OwnershipContractAttrs::kAggregateSlotIndex,
                  builder.getI64IntegerAttr(*contract.slot));
    op->removeAttr(kMemRefAggregateLoadContractId);
  };

  ContractPreserver<MemRefAggregateLoadContract> preserver(
      contracts, "aggregate slot load contract was not preserved through LLVM "
                 "lowering");
  module.walk([&](mlir::Operation *op) {
    preserver.apply(
        op, kMemRefAggregateLoadContractId,
        "carries an unknown aggregate load contract id",
        [](const MemRefAggregateLoadContract &) { return true; },
        [](const MemRefAggregateLoadContract &) { return mlir::success(); },
        [&](const MemRefAggregateLoadContract &contract) {
          applyContract(op, contract);
        });
  });
  return preserver.finish();
}

namespace container::access::Contract {
void collect(mlir::ModuleOp module,
             llvm::SmallVectorImpl<MemRefContainerAccessContract> &contracts) {
  mlir::OpBuilder builder(module.getContext());
  int64_t nextId = 0;
  auto collect = [&](mlir::Operation *op, bool store) {
    auto group =
        op->getAttrOfType<mlir::StringAttr>(ContainerSafetyAttrs::kAccessGroup);
    if (!group)
      return;
    auto component = op->getAttrOfType<mlir::StringAttr>(
        ContainerSafetyAttrs::kAccessComponent);
    int64_t id = nextId++;
    op->setAttr(kMemRefContainerAccessContractId,
                builder.getI64IntegerAttr(id));
    contracts.push_back(MemRefContainerAccessContract{id, op->getLoc(), store,
                                                      group, component});
  };
  module.walk(
      [&](mlir::memref::LoadOp load) { collect(load.getOperation(), false); });
  module.walk([&](mlir::memref::StoreOp store) {
    collect(store.getOperation(), true);
  });
}

mlir::LogicalResult
preserve(mlir::ModuleOp module,
         llvm::ArrayRef<MemRefContainerAccessContract> contracts) {
  auto applyContract = [](mlir::Operation *op,
                          const MemRefContainerAccessContract &contract) {
    op->setAttr(ContainerSafetyAttrs::kAccessGroup, contract.group);
    if (contract.component)
      op->setAttr(ContainerSafetyAttrs::kAccessComponent, contract.component);
    op->removeAttr(kMemRefContainerAccessContractId);
  };

  ContractPreserver<MemRefContainerAccessContract> preserver(
      contracts,
      "managed container access contract was not preserved through LLVM "
      "lowering");
  auto match = [&](mlir::Operation *op, bool store) {
    preserver.apply(
        op, kMemRefContainerAccessContractId,
        "carries an unknown container access contract id",
        [&](const MemRefContainerAccessContract &contract) {
          return contract.store == store;
        },
        [](const MemRefContainerAccessContract &) { return mlir::success(); },
        [&](const MemRefContainerAccessContract &contract) {
          applyContract(op, contract);
        });
  };

  module.walk([&](mlir::LLVM::LoadOp load) {
    match(load.getOperation(), /*store=*/false);
  });
  module.walk([&](mlir::LLVM::StoreOp store) {
    match(store.getOperation(), /*store=*/true);
  });

  return preserver.finish();
}
} // namespace container::access::Contract

void collectMemRefDeallocContracts(
    mlir::ModuleOp module,
    llvm::SmallVectorImpl<MemRefDeallocContract> &contracts) {
  mlir::OpBuilder builder(module.getContext());
  int64_t nextId = 0;
  auto getReleaseAtomic = [](mlir::Value value) -> mlir::memref::AtomicRMWOp {
    auto atomic = value.getDefiningOp<mlir::memref::AtomicRMWOp>();
    if (!atomic)
      return {};
    auto role =
        atomic->getAttrOfType<mlir::StringAttr>(ThreadSafetyAttrs::kAtomicRole);
    if (role &&
        role.getValue() == ThreadSafetyAttrs::kRoleContainerRefcountRelease)
      return atomic;
    return {};
  };
  auto getGuardingRelease =
      [&](mlir::Value condition,
          mlir::Value deallocated) -> mlir::memref::AtomicRMWOp {
    auto cmp = condition.getDefiningOp<mlir::arith::CmpIOp>();
    if (!cmp || cmp.getPredicate() != mlir::arith::CmpIPredicate::eq)
      return {};
    auto matches = [&](mlir::Value lhs,
                       mlir::Value rhs) -> mlir::memref::AtomicRMWOp {
      auto release = getReleaseAtomic(lhs);
      if (!release || !isArithConstantInt(rhs, 1))
        return {};
      if (!sameLoweringContainerDescriptorResource(release.getMemref(),
                                                   deallocated))
        return {};
      return release;
    };
    if (auto release = matches(cmp.getLhs(), cmp.getRhs()))
      return release;
    return matches(cmp.getRhs(), cmp.getLhs());
  };
  auto inferGuardedGroup =
      [&](mlir::memref::DeallocOp dealloc) -> mlir::StringAttr {
    for (mlir::Operation *parent = dealloc->getParentOp(); parent;
         parent = parent->getParentOp()) {
      auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(parent);
      if (!ifOp)
        continue;
      auto release =
          getGuardingRelease(ifOp.getCondition(), dealloc.getMemref());
      if (!release)
        continue;
      if (auto group = release->getAttrOfType<mlir::StringAttr>(
              ThreadSafetyAttrs::kAtomicMemRefGroup))
        return group;
      return builder.getStringAttr(
          makeContainerResourceGroup(release.getMemref()));
    }
    return {};
  };
  module.walk([&](mlir::memref::DeallocOp dealloc) {
    auto explicitGroup = dealloc->getAttrOfType<mlir::StringAttr>(
        ContainerSafetyAttrs::kDeallocGroup);
    auto classPart = dealloc->getAttrOfType<mlir::StringAttr>(
        ClassSafetyAttrs::kDeallocPart);
    auto objectPart = dealloc->getAttrOfType<mlir::StringAttr>(
        OwnershipContractAttrs::kObjectDeallocPart);
    bool exceptionCellFree =
        dealloc->hasAttr(AsyncSafetyAttrs::kExceptionCellFree);
    mlir::StringAttr guardedGroup;
    std::optional<llvm::StringRef> inferredGroup;
    if (!explicitGroup) {
      guardedGroup = inferGuardedGroup(dealloc);
      inferredGroup = getLoweringValueDefStringAttr(
          dealloc.getMemref(), ContainerSafetyAttrs::kDescriptorGroup);
    }
    llvm::StringRef groupName =
        explicitGroup
            ? explicitGroup.getValue()
            : (guardedGroup
                   ? guardedGroup.getValue()
                   : (inferredGroup ? *inferredGroup : llvm::StringRef{}));
    if (groupName.empty() && !classPart && !objectPart && !exceptionCellFree)
      return;
    auto explicitComponent = dealloc->getAttrOfType<mlir::StringAttr>(
        ContainerSafetyAttrs::kDeallocComponent);
    std::optional<llvm::StringRef> inferredComponent;
    if (!explicitComponent)
      inferredComponent = getLoweringValueDefStringAttr(
          dealloc.getMemref(), ContainerSafetyAttrs::kDescriptorComponent);
    int64_t id = nextId++;
    dealloc->setAttr(kMemRefDeallocContractId, builder.getI64IntegerAttr(id));
    contracts.push_back(MemRefDeallocContract{
        id, dealloc.getLoc(),
        groupName.empty() ? mlir::StringAttr{}
                          : builder.getStringAttr(groupName),
        explicitComponent
            ? explicitComponent
            : (inferredComponent ? builder.getStringAttr(*inferredComponent)
                                 : mlir::StringAttr{}),
        classPart, objectPart, exceptionCellFree});
  });
}

static bool deallocShape(mlir::LLVM::CallOp call) {
  return call.getNumResults() == 0 && call.getNumOperands() == 1 &&
         mlir::isa<mlir::LLVM::LLVMPointerType>(call.getOperand(0).getType());
}

mlir::LogicalResult preserveLoweredMemRefDeallocContracts(
    mlir::ModuleOp module, llvm::ArrayRef<MemRefDeallocContract> contracts) {
  auto applyContract = [](mlir::LLVM::CallOp call,
                          const MemRefDeallocContract &contract) {
    if (contract.group)
      call->setAttr(ContainerSafetyAttrs::kDeallocGroup, contract.group);
    if (contract.component)
      call->setAttr(ContainerSafetyAttrs::kDeallocComponent,
                    contract.component);
    if (contract.classPart)
      call->setAttr(ClassSafetyAttrs::kDeallocPart, contract.classPart);
    if (contract.objectPart)
      call->setAttr(OwnershipContractAttrs::kObjectDeallocPart,
                    contract.objectPart);
    if (contract.exceptionCellFree)
      call->setAttr(AsyncSafetyAttrs::kExceptionCellFree,
                    mlir::UnitAttr::get(call.getContext()));
    call->removeAttr(kMemRefDeallocContractId);
  };

  ContractPreserver<MemRefDeallocContract> preserver(
      contracts, "memref dealloc safety contract was not preserved through "
                 "LLVM lowering");
  module.walk([&](mlir::LLVM::CallOp call) {
    preserver.apply(
        call.getOperation(), kMemRefDeallocContractId,
        "carries an unknown memref dealloc contract id",
        [](const MemRefDeallocContract &) { return true; },
        [&](const MemRefDeallocContract &) -> mlir::LogicalResult {
          if (!deallocShape(call))
            return call->emitOpError("memref dealloc contract lowered to a "
                                     "non-dealloc call shape");
          return mlir::success();
        },
        [&](const MemRefDeallocContract &contract) {
          applyContract(call, contract);
        });
  });

  return preserver.finish();
}

void collectLoweredSafetyContracts(mlir::ModuleOp module,
                                   LoweredSafetyContracts &contracts) {
  collectAsyncArgProvenanceContracts(module, contracts.asyncArgs);
  collectNonObjectArgContracts(module, contracts.nonObjectArgs);
  collectMemRefAtomicContracts(module, contracts.memRefAtomics);
  collectMemRefAggregateLoadContracts(module, contracts.aggregateLoads);
  container::access::Contract::collect(module, contracts.containerAccesses);
  collectMemRefDeallocContracts(module, contracts.deallocs);
}

void collectLoweredSafetyContracts(mlir::ModuleOp module,
                                   const PyLLVMTypeConverter &typeConverter,
                                   LoweredSafetyContracts &contracts) {
  collectAsyncArgProvenanceContracts(module, typeConverter,
                                     contracts.asyncArgs);
  collectNonObjectArgContracts(module, typeConverter, contracts.nonObjectArgs);
  collectMemRefAtomicContracts(module, contracts.memRefAtomics);
  collectMemRefAggregateLoadContracts(module, contracts.aggregateLoads);
  container::access::Contract::collect(module, contracts.containerAccesses);
  collectMemRefDeallocContracts(module, contracts.deallocs);
}

mlir::LogicalResult
preserveLoweredSafetyContracts(mlir::ModuleOp module,
                               const LoweredSafetyContracts &contracts) {
  if (mlir::failed(
          preserveLLVMAsyncArgProvenanceContracts(module, contracts.asyncArgs)))
    return mlir::failure();
  if (mlir::failed(
          preserveLLVMNonObjectArgContracts(module, contracts.nonObjectArgs)))
    return mlir::failure();
  if (mlir::failed(preserveLoweredMemRefAtomicContracts(
          module, contracts.memRefAtomics)))
    return mlir::failure();
  if (mlir::failed(preserveLoweredMemRefAggregateLoadContracts(
          module, contracts.aggregateLoads)))
    return mlir::failure();
  if (mlir::failed(container::access::Contract::preserve(
          module, contracts.containerAccesses)))
    return mlir::failure();
  if (mlir::failed(
          preserveLoweredMemRefDeallocContracts(module, contracts.deallocs)))
    return mlir::failure();
  return mlir::success();
}

static bool isPyRuntimeBridgeType(mlir::Type type) {
  return isPyType(type) || mlir::isa<FuncType>(type) ||
         mlir::isa<TupleType>(type) || mlir::isa<ListType>(type) ||
         mlir::isa<ClassType>(type) || mlir::isa<DictType>(type) ||
         mlir::isa<CoroutineType>(type) || mlir::isa<FutureType>(type) ||
         mlir::isa<TaskType>(type);
}

static bool isConvertedAsyncDescriptorBridge(mlir::Type resultType,
                                             mlir::ValueRange inputs) {
  if (!mlir::isa<CoroutineType, FutureType, TaskType>(resultType))
    return false;
  bool expectsCancelFlag = mlir::isa<TaskType>(resultType);
  if (inputs.size() != (expectsCancelFlag ? 3 : 2))
    return false;
  if (!mlir::isa<mlir::async::ValueType>(inputs[0].getType()))
    return false;
  if (!async_runtime::ExceptionCell::hasProvenance(inputs[1]))
    return false;
  return !expectsCancelFlag || mlir::isa<mlir::MemRefType>(inputs[2].getType());
}

PyLLVMTypeConverter::PyLLVMTypeConverter(mlir::MLIRContext *ctx,
                                         mlir::ModuleOp module)
    : mlir::LLVMTypeConverter(ctx), module(module) {
  addConversion([](mlir::Type type) -> std::optional<mlir::Type> {
    if (mlir::isa<mlir::IntegerType, mlir::FloatType, mlir::RankedTensorType>(
            type))
      return type;
    return std::nullopt;
  });

  auto convertAsyncPayload =
      [this, ctx](mlir::Type payloadType) -> std::optional<mlir::Type> {
    mlir::SmallVector<mlir::Type> parts;
    if (failed(convertType(payloadType, parts)) || parts.empty())
      return std::nullopt;
    mlir::SmallVector<mlir::Type> storageParts;
    for (mlir::Type part : parts) {
      mlir::Type storage = convertType(part);
      if (!storage)
        storage = part;
      storageParts.push_back(storage);
    }
    if (storageParts.size() == 1)
      return storageParts.front();
    return mlir::LLVM::LLVMStructType::getLiteral(ctx, storageParts);
  };

  addConversion([ctx, convertAsyncPayload](
                    mlir::Type type) -> std::optional<mlir::Type> {
    if (mlir::isa<ListType, DictType, TupleType, CoroutineType, TaskType,
                  FutureType>(type))
      return std::nullopt;
    if (mlir::isa<BoolType>(type))
      return mlir::IntegerType::get(ctx, 1);
    if (mlir::isa<NoneType>(type))
      return mlir::IntegerType::get(ctx, 1);
    if (mlir::isa<ExceptionCellType>(type))
      return async_runtime::getExceptionCellType(ctx);
    if (mlir::isa<FloatType>(type))
      return mlir::Float64Type::get(ctx);
    if (mlir::isa<FuncType>(type))
      return mlir::FunctionType::get(ctx, {}, {});
    if (auto asyncValueType = mlir::dyn_cast<mlir::async::ValueType>(type)) {
      auto valueType = convertAsyncPayload(asyncValueType.getValueType());
      if (!valueType)
        return std::nullopt;
      return mlir::async::ValueType::get(*valueType);
    }
    return std::nullopt;
  });

  addConversion([this, ctx, convertAsyncPayload](
                    mlir::Type type, mlir::SmallVectorImpl<mlir::Type> &results)
                    -> std::optional<mlir::LogicalResult> {
    if (mlir::isa<IntType>(type)) {
      object_abi::long_abi::Parts::storageTypes(ctx, results);
      return mlir::success();
    }
    if (mlir::isa<StrType>(type)) {
      object_abi::str_abi::Parts::storageTypes(ctx, results);
      return mlir::success();
    }
    if (mlir::isa<ExceptionType>(type)) {
      object_abi::exception_abi::Parts::storageTypes(ctx, results);
      return mlir::success();
    }
    if (auto classType = mlir::dyn_cast<ClassType>(type)) {
      mlir::FailureOr<class_layout::Layout> layout =
          class_layout::get(this->module.getOperation(), classType, *this);
      if (mlir::failed(layout))
        return std::nullopt;
      class_layout::partsValueTypes(*layout, results);
      return mlir::success();
    }

    mlir::Type payloadType;
    bool includeCancelFlag = false;
    if (auto coroType = mlir::dyn_cast<CoroutineType>(type)) {
      payloadType = coroType.getResultType();
    } else if (auto futureType = mlir::dyn_cast<FutureType>(type)) {
      payloadType = futureType.getResultType();
    } else if (auto taskType = mlir::dyn_cast<TaskType>(type)) {
      payloadType = taskType.getResultType();
      includeCancelFlag = true;
    } else {
      return std::nullopt;
    }
    auto resultType = convertAsyncPayload(payloadType);
    if (!resultType)
      return std::nullopt;
    results.push_back(mlir::async::ValueType::get(*resultType));
    results.push_back(async_runtime::getExceptionCellType(ctx));
    if (!includeCancelFlag)
      return mlir::success();
    results.push_back(
        mlir::MemRefType::get({1}, mlir::IntegerType::get(ctx, 8)));
    return mlir::success();
  });

  addConversion(
      [this](ListType listType, mlir::SmallVectorImpl<mlir::Type> &results)
          -> std::optional<mlir::LogicalResult> {
        auto itemsType = getListItemsMemRefType(listType);
        if (!itemsType)
          return std::nullopt;
        results.push_back(getListHeaderMemRefType(&getContext()));
        results.push_back(getContainerLockMemRefType(&getContext()));
        results.push_back(itemsType);
        return mlir::success();
      });

  addConversion(
      [this](TupleType tupleType, mlir::SmallVectorImpl<mlir::Type> &results)
          -> std::optional<mlir::LogicalResult> {
        auto itemsType = getTupleItemsMemRefType(tupleType);
        if (!itemsType)
          return std::nullopt;
        results.push_back(getTupleHeaderMemRefType(&getContext()));
        results.push_back(itemsType);
        return mlir::success();
      });

  addConversion(
      [this](DictType dictType, mlir::SmallVectorImpl<mlir::Type> &results)
          -> std::optional<mlir::LogicalResult> {
        if (container::Slot::supported(dictType.getKeyType()) &&
            container::Slot::supported(dictType.getValueType())) {
          auto keysType = getDictKeysMemRefType(dictType);
          auto valuesType = getDictValuesMemRefType(dictType);
          if (!keysType || !valuesType)
            return std::nullopt;
          results.push_back(getDictHeaderMemRefType(&getContext()));
          results.push_back(getContainerLockMemRefType(&getContext()));
          results.push_back(keysType);
          results.push_back(valuesType);
          results.push_back(getDictStatesMemRefType(&getContext()));
          return mlir::success();
        }
        return std::nullopt;
      });

  auto materializeBridge = [](mlir::OpBuilder &builder, mlir::Type resultType,
                              mlir::ValueRange inputs,
                              mlir::Location loc) -> mlir::Value {
    if (inputs.empty())
      return {};
    mlir::Type inputType = inputs.front().getType();
    bool inputIsPyBridge =
        isPyRuntimeBridgeType(inputType) ||
        isConvertedAsyncDescriptorBridge(resultType, inputs) ||
        llvm::all_of(inputs, [](mlir::Value input) {
          return mlir::isa<mlir::MemRefType>(input.getType());
        });
    if (!isPyRuntimeBridgeType(resultType) && !inputIsPyBridge)
      return {};
    return builder
        .create<mlir::UnrealizedConversionCastOp>(loc, resultType, inputs)
        .getResult(0);
  };

  auto materializeTargetBridge =
      [](mlir::OpBuilder &builder, mlir::TypeRange resultTypes,
         mlir::ValueRange inputs, mlir::Location loc,
         mlir::Type originalType) -> mlir::SmallVector<mlir::Value> {
    if (inputs.size() != 1 || resultTypes.empty())
      return {};
    if (!isPyRuntimeBridgeType(originalType))
      return {};
    auto cast = builder.create<mlir::UnrealizedConversionCastOp>(
        loc, resultTypes, inputs);
    mlir::SmallVector<mlir::Value> results;
    results.append(cast.getResults().begin(), cast.getResults().end());
    return results;
  };

  addSourceMaterialization(materializeBridge);
  addTargetMaterialization(materializeBridge);
  addTargetMaterialization(materializeTargetBridge);
}

mlir::Type
PyLLVMTypeConverter::getContainerStorageType(mlir::Type logicalType) const {
  if (auto classType = mlir::dyn_cast<ClassType>(logicalType)) {
    if (mlir::Type objectType = class_layout::carrierStorageType(
            module, classType, *this, &getContext()))
      return objectType;
  }
  if (mlir::isa<StrType, ExceptionType>(logicalType)) {
    llvm::SmallVector<mlir::Type, 2> storageTypes;
    if (mlir::isa<StrType>(logicalType))
      object_abi::str_abi::Parts::storageTypes(&getContext(), storageTypes);
    else
      object_abi::exception_abi::Parts::storageTypes(&getContext(),
                                                     storageTypes);
    llvm::SmallVector<mlir::MemRefType, 2> partTypes;
    for (mlir::Type storageType : storageTypes) {
      auto memrefType = mlir::dyn_cast<mlir::MemRefType>(storageType);
      if (!memrefType)
        return {};
      partTypes.push_back(memrefType);
    }
    return class_layout::objectCarrierType(&getContext(), partTypes);
  }
  return container::Slot::storageType(logicalType, &getContext());
}

mlir::MemRefType
PyLLVMTypeConverter::getListItemsMemRefType(ListType listType) const {
  if (!listType)
    return {};
  mlir::Type storageType = getContainerStorageType(listType.getElementType());
  if (!storageType)
    return {};
  return mlir::MemRefType::get({mlir::ShapedType::kDynamic}, storageType);
}

mlir::Type
PyLLVMTypeConverter::getTupleItemsStorageType(TupleType tupleType) const {
  if (!tupleType)
    return {};
  auto elementTypes = tupleType.getElementTypes();
  if (elementTypes.empty())
    return mlir::IntegerType::get(&getContext(), 8);
  mlir::Type firstStorage = getContainerStorageType(elementTypes.front());
  if (!firstStorage)
    return {};
  for (mlir::Type elementType : elementTypes.drop_front()) {
    mlir::Type storage = getContainerStorageType(elementType);
    if (!storage || storage != firstStorage)
      return {};
  }
  return firstStorage;
}

mlir::MemRefType
PyLLVMTypeConverter::getTupleItemsMemRefType(TupleType tupleType) const {
  mlir::Type storageType = getTupleItemsStorageType(tupleType);
  if (!storageType)
    return {};
  return mlir::MemRefType::get(
      {static_cast<int64_t>(tupleType.getElementTypes().size())}, storageType);
}

mlir::MemRefType
PyLLVMTypeConverter::getDictKeysMemRefType(DictType dictType) const {
  if (!dictType)
    return {};
  mlir::Type storageType = getContainerStorageType(dictType.getKeyType());
  if (!storageType)
    return {};
  return mlir::MemRefType::get({mlir::ShapedType::kDynamic}, storageType);
}

mlir::MemRefType
PyLLVMTypeConverter::getDictValuesMemRefType(DictType dictType) const {
  if (!dictType)
    return {};
  mlir::Type storageType = getContainerStorageType(dictType.getValueType());
  if (!storageType)
    return {};
  return mlir::MemRefType::get({mlir::ShapedType::kDynamic}, storageType);
}

RuntimeAPI::RuntimeAPI(mlir::ModuleOp module, mlir::OpBuilder &rewriter,
                       const PyLLVMTypeConverter &typeConverter)
    : module(module), rewriter(rewriter) {
  (void)typeConverter;
}

static mlir::LLVM::LLVMFuncOp
declareRuntimeFunc(mlir::Location loc, mlir::ModuleOp module,
                   mlir::OpBuilder &rewriter, llvm::StringRef name,
                   mlir::Type resultType, llvm::ArrayRef<mlir::Type> argTypes) {
  if (auto fn = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(name))
    return fn;

  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  auto funcType =
      mlir::LLVM::LLVMFunctionType::get(resultType, argTypes, false);
  return rewriter.create<mlir::LLVM::LLVMFuncOp>(loc, name, funcType);
}

static bool requiresFuncRuntimeCall(mlir::Type type) {
  return mlir::isa<mlir::MemRefType>(type);
}

static void appendLoweredObjectMemRefDescriptor(
    mlir::Location loc, mlir::Value descriptor,
    llvm::SmallVectorImpl<mlir::Value> &operands, mlir::OpBuilder &rewriter) {
  auto descriptorType =
      mlir::cast<mlir::LLVM::LLVMStructType>(descriptor.getType());
  auto body = descriptorType.getBody();
  auto i64Type = rewriter.getI64Type();
  operands.push_back(rewriter.create<mlir::LLVM::ExtractValueOp>(
      loc, body[0], descriptor, rewriter.getDenseI64ArrayAttr({0})));
  operands.push_back(rewriter.create<mlir::LLVM::ExtractValueOp>(
      loc, body[1], descriptor, rewriter.getDenseI64ArrayAttr({1})));
  operands.push_back(rewriter.create<mlir::LLVM::ExtractValueOp>(
      loc, body[2], descriptor, rewriter.getDenseI64ArrayAttr({2})));
  operands.push_back(rewriter.create<mlir::LLVM::ExtractValueOp>(
      loc, i64Type, descriptor, rewriter.getDenseI64ArrayAttr({3, 0})));
  operands.push_back(rewriter.create<mlir::LLVM::ExtractValueOp>(
      loc, i64Type, descriptor, rewriter.getDenseI64ArrayAttr({4, 0})));
}

static mlir::Value loweredObjectStorageSource(mlir::Value value) {
  llvm::SmallPtrSet<mlir::Operation *, 4> seen;
  while (auto cast = value.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
    if (!seen.insert(cast.getOperation()).second || cast->getNumOperands() != 1)
      return {};
    value = cast.getOperand(0);
    if (object_abi::Type::isLoweredStorage(value.getType()))
      return value;
  }
  return {};
}

static bool compatibleMemRefCast(mlir::MemRefType from, mlir::MemRefType to) {
  if (!from || !to || from.getRank() != to.getRank() ||
      from.getElementType() != to.getElementType())
    return false;
  for (auto [lhs, rhs] : llvm::zip(from.getShape(), to.getShape())) {
    if (lhs == rhs || lhs == mlir::ShapedType::kDynamic ||
        rhs == mlir::ShapedType::kDynamic)
      continue;
    return false;
  }
  return true;
}

static mlir::Value adaptRuntimeOperand(mlir::Location loc, mlir::Value operand,
                                       mlir::Type expected,
                                       mlir::OpBuilder &rewriter) {
  if (!operand || !expected || operand.getType() == expected)
    return operand;
  if (mlir::Value source = storage_cast::source(operand))
    operand = source;
  if (operand.getType() == expected)
    return operand;

  auto from = mlir::dyn_cast<mlir::MemRefType>(operand.getType());
  auto to = mlir::dyn_cast<mlir::MemRefType>(expected);
  if (!compatibleMemRefCast(from, to))
    return operand;

  mlir::Value casted = rewriter.create<mlir::memref::CastOp>(loc, to, operand);
  if (object_abi::Header::isOwned(operand.getType()) ||
      object_abi::Header::isView(expected))
    casted.getDefiningOp()->setAttr(OwnershipContractAttrs::kObjectHeader,
                                    rewriter.getUnitAttr());
  return casted;
}

static mlir::func::FuncOp
declareFuncRuntimeFunc(mlir::Location loc, mlir::ModuleOp module,
                       mlir::OpBuilder &rewriter, llvm::StringRef name,
                       mlir::TypeRange resultTypes,
                       llvm::ArrayRef<mlir::Type> argTypes) {
  auto markObjectHeaderArgs = [&](mlir::func::FuncOp fn) {
    for (auto [index, type] : llvm::enumerate(argTypes)) {
      if (!object_abi::Header::isOwned(type) &&
          !object_abi::Header::isView(type))
        continue;
      if (static_cast<unsigned>(index) >= fn.getNumArguments())
        continue;
      fn.setArgAttr(static_cast<unsigned>(index),
                    OwnershipContractAttrs::kObjectHeader,
                    rewriter.getUnitAttr());
    }
  };

  if (auto fn = module.lookupSymbol<mlir::func::FuncOp>(name)) {
    markObjectHeaderArgs(fn);
    return fn;
  }

  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  auto funcType =
      mlir::FunctionType::get(module.getContext(), argTypes, resultTypes);
  auto fn = rewriter.create<mlir::func::FuncOp>(loc, name, funcType);
  markObjectHeaderArgs(fn);
  return fn;
}

static void markRuntimeCallResults(mlir::Operation *call,
                                   mlir::TypeRange resultTypes,
                                   mlir::OpBuilder &rewriter) {
  if (!call)
    return;
  if (!resultTypes.empty() && object_abi::Header::isOwned(resultTypes.front()))
    call->setAttr(OwnershipContractAttrs::kObjectHeader,
                  rewriter.getUnitAttr());
}

RuntimeAPI::Call RuntimeAPI::call(mlir::Location loc, llvm::StringRef name,
                                  mlir::TypeRange resultTypes,
                                  mlir::ValueRange operands) {
  mlir::SmallVector<mlir::Value> callOperands;
  callOperands.reserve(operands.size() * 2);

  mlir::func::FuncOp existingFunc =
      module.lookupSymbol<mlir::func::FuncOp>(name);
  mlir::FunctionType existingType =
      existingFunc ? existingFunc.getFunctionType() : mlir::FunctionType();
  if (existingType && existingType.getNumInputs() == operands.size()) {
    for (auto [operand, expected] :
         llvm::zip(operands, existingType.getInputs())) {
      callOperands.push_back(
          adaptRuntimeOperand(loc, operand, expected, rewriter));
    }
  }

  if (callOperands.empty()) {
    for (mlir::Value operand : operands) {
      if (mlir::Value lowered = loweredObjectStorageSource(operand))
        operand = lowered;
      if (object_abi::Type::isLoweredStorage(operand.getType())) {
        appendLoweredObjectMemRefDescriptor(loc, operand, callOperands,
                                            rewriter);
        continue;
      }
      callOperands.push_back(operand);
    }
  }

  mlir::SmallVector<mlir::Type> operandTypes;
  operandTypes.reserve(callOperands.size());
  for (mlir::Value operand : callOperands)
    operandTypes.push_back(operand.getType());

  bool isVoid = resultTypes.empty();

  bool useFuncCall =
      static_cast<bool>(module.lookupSymbol<mlir::func::FuncOp>(name));
  if (!useFuncCall) {
    useFuncCall = resultTypes.size() > 1 ||
                  llvm::any_of(resultTypes, requiresFuncRuntimeCall) ||
                  llvm::any_of(operandTypes, requiresFuncRuntimeCall);
  }
  if (useFuncCall) {
    auto callee = declareFuncRuntimeFunc(loc, module, rewriter, name,
                                         resultTypes, operandTypes);
    auto symbolRef =
        mlir::SymbolRefAttr::get(module.getContext(), callee.getName());
    auto call = rewriter.create<mlir::func::CallOp>(loc, symbolRef, resultTypes,
                                                    callOperands);
    markRuntimeCallResults(call.getOperation(), resultTypes, rewriter);
    return RuntimeAPI::Call(call.getOperation());
  }

  mlir::Type actualResult =
      isVoid ? mlir::LLVM::LLVMVoidType::get(module.getContext())
             : resultTypes.front();
  auto callee = declareRuntimeFunc(loc, module, rewriter, name, actualResult,
                                   operandTypes);
  auto symbolRef =
      mlir::SymbolRefAttr::get(module.getContext(), callee.getName());
  llvm::SmallVector<mlir::Type, 1> resultStorage;
  if (!isVoid)
    resultStorage.push_back(actualResult);
  mlir::TypeRange results(resultStorage);
  auto call = rewriter.create<mlir::LLVM::CallOp>(loc, results, symbolRef,
                                                  callOperands);
  markRuntimeCallResults(call.getOperation(), resultTypes, rewriter);
  return RuntimeAPI::Call(call.getOperation());
}

RuntimeAPI::Call RuntimeAPI::call(mlir::Location loc, llvm::StringRef name,
                                  mlir::Type resultType,
                                  mlir::ValueRange operands) {
  llvm::SmallVector<mlir::Type, 1> resultTypes;
  if (resultType && !mlir::isa<mlir::LLVM::LLVMVoidType>(resultType))
    resultTypes.push_back(resultType);
  return call(loc, name, mlir::TypeRange(resultTypes), operands);
}

RuntimeAPI::Call RuntimeAPI::call(mlir::Location loc, llvm::StringRef name,
                                  std::nullptr_t, mlir::ValueRange operands) {
  return call(loc, name, mlir::Type(), operands);
}

mlir::Value RuntimeAPI::getByteLiteral(mlir::Location loc,
                                       mlir::StringAttr literal) {
  mlir::Type byteType = rewriter.getI8Type();
  auto staticType = mlir::MemRefType::get(
      {static_cast<int64_t>(literal.getValue().size())}, byteType);
  mlir::Value storage =
      rewriter.create<mlir::memref::AllocaOp>(loc, staticType);
  for (auto indexed : llvm::enumerate(literal.getValue())) {
    auto byte = static_cast<uint8_t>(indexed.value());
    mlir::Value value =
        rewriter.create<mlir::arith::ConstantIntOp>(loc, byte, byteType);
    mlir::Value index =
        rewriter.create<mlir::arith::ConstantIndexOp>(loc, indexed.index());
    rewriter.create<mlir::memref::StoreOp>(loc, value, storage, index);
  }

  auto dynamicType =
      mlir::MemRefType::get({mlir::ShapedType::kDynamic}, byteType);
  return rewriter.create<mlir::memref::CastOp>(loc, dynamicType, storage);
}

mlir::Value RuntimeAPI::getUnicodeLiteral(mlir::Location loc,
                                          mlir::StringAttr literal) {
  mlir::emitError(loc)
      << "single-result unicode literal lowering is unavailable; use split "
         "unicode memref ABI";
  return {};
}

mlir::Value RuntimeAPI::getUnicodeLiteral(mlir::Location loc,
                                          mlir::StringAttr literal,
                                          mlir::Type resultType) {
  (void)literal;
  (void)resultType;
  mlir::emitError(loc)
      << "single-result unicode literal lowering is unavailable; use split "
         "unicode memref ABI";
  return {};
}

mlir::Value RuntimeAPI::getNoneValue(mlir::Location loc) {
  return rewriter.create<mlir::arith::ConstantIntOp>(loc, 0, 1);
}

mlir::Value RuntimeAPI::getI64Constant(mlir::Location loc, std::int64_t value) {
  return rewriter.create<mlir::LLVM::ConstantOp>(
      loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(value));
}

mlir::Value RuntimeAPI::getF64Constant(mlir::Location loc, double value) {
  return rewriter.create<mlir::LLVM::ConstantOp>(
      loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(value));
}

} // namespace py
