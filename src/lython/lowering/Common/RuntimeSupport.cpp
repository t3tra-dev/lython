#include "Common/RuntimeSupport.h"

#include "Common/Container.h"
#include "Common/LoweringUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/FormatVariadic.h"

#include <atomic>
#include <cstdint>
#include <optional>
#include <string>

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

static void setI64Attr(mlir::Operation *op, llvm::StringRef name,
                       int64_t value) {
  mlir::Builder builder(op->getContext());
  op->setAttr(name, builder.getI64IntegerAttr(value));
}

} // namespace

bool runtime::Callee::retain(llvm::StringRef name) {
  return name == RuntimeSymbols::kIncRef ||
         name.starts_with("__ly_class_incref_");
}

bool runtime::Callee::release(llvm::StringRef name) {
  return name == RuntimeSymbols::kDecRef ||
         name.starts_with("__ly_class_decref_");
}

bool runtime::Callee::transfer(llvm::StringRef name) {
  return name == RuntimeSymbols::kEHThrow;
}

bool runtime::Callee::setField(llvm::StringRef name) {
  return name.starts_with("__ly_class_setfield_");
}

bool runtime::Callee::alwaysOwnedResult(llvm::StringRef name) {
  return name == RuntimeSymbols::kStrFromUtf8 ||
         name == RuntimeSymbols::kUnicodeConcat ||
         name == RuntimeSymbols::kFloatFromDouble ||
         name == RuntimeSymbols::kNumberAdd ||
         name == RuntimeSymbols::kNumberSub ||
         name == RuntimeSymbols::kObjectRepr ||
         name == RuntimeSymbols::kClassReprNamed ||
         name == RuntimeSymbols::kExceptionNew ||
         name == RuntimeSymbols::kLongFromString ||
         name == RuntimeSymbols::kLongAdd || name == RuntimeSymbols::kLongSub ||
         name.starts_with("__ly_class_promote_");
}

bool runtime::Callee::conditionalOwnedResult(llvm::StringRef name) {
  return name == RuntimeSymbols::kLongFromI64 ||
         name.starts_with("__ly_class_getfield_");
}

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
};

static constexpr Info kCallees[] = {
    {"mlirAsyncRuntimeCreateToken", kCreateHandle},
    {"mlirAsyncRuntimeCreateValue", kCreateHandle},
    {"mlirAsyncRuntimeCreateGroup", kCreateHandle},
    {"mlirAsyncRuntimeAddRef", kRefcount | kBorrowOperand0},
    {"mlirAsyncRuntimeDropRef", kRefcount | kBorrowOperand0},
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

namespace ownership::llvm_func::Contract {
void apply(mlir::LLVM::LLVMFuncOp fn, llvm::StringRef name) {
  if (!fn)
    return;

  mlir::Operation *op = fn.getOperation();

  if (runtime::Callee::retain(name)) {
    setIndexArrayAttr(op, OwnershipContractAttrs::kRetainArgs, {0});
    return;
  }

  if (runtime::Callee::release(name)) {
    setIndexArrayAttr(op, OwnershipContractAttrs::kReleaseArgs, {0});
    return;
  }

  if (runtime::Callee::transfer(name)) {
    setIndexArrayAttr(op, OwnershipContractAttrs::kTransferArgs, {0});
    return;
  }

  if (runtime::Callee::setField(name)) {
    setI64Attr(op, OwnershipContractAttrs::kSetFieldValueArg, 1);
    setI64Attr(op, OwnershipContractAttrs::kSetFieldRetainArg, 2);
    return;
  }

  if (name.starts_with("__ly_class_getfield_")) {
    setI64Attr(op, OwnershipContractAttrs::kGetFieldBorrowArg, 1);
    setI64Attr(op, OwnershipContractAttrs::kGetFieldOwnedResult, 0);
    return;
  }

  if (runtime::Callee::alwaysOwnedResult(name)) {
    setIndexArrayAttr(op, OwnershipContractAttrs::kOwnedResults, {0});
    return;
  }
}
} // namespace ownership::llvm_func::Contract

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

static void setLLVMAggregateSlotProvenance(mlir::Operation *op,
                                           mlir::Value address) {
  mlir::Value root = getAggregateAddressRoot(address);
  setAggregateSlotProvenance(op, makeAggregateValueGroup(root, "llvm"),
                             "llvm-slot", getAggregateAddressSlot(address));
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
    ownership::aggregate::Slot::markLoad(bitcast.getArg(), group, component,
                                         slot);
    return;
  }
  if (auto intToPtr = mlir::dyn_cast<mlir::LLVM::IntToPtrOp>(def)) {
    ownership::aggregate::Slot::markLoad(intToPtr.getArg(), group, component,
                                         slot);
    return;
  }
  if (auto ptrToInt = mlir::dyn_cast<mlir::LLVM::PtrToIntOp>(def)) {
    ownership::aggregate::Slot::markLoad(ptrToInt.getArg(), group, component,
                                         slot);
    return;
  }
  if (auto cast = mlir::dyn_cast<mlir::UnrealizedConversionCastOp>(def)) {
    if (cast->getNumOperands() == 1)
      ownership::aggregate::Slot::markLoad(cast.getOperand(0), group, component,
                                           slot);
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
  if (auto bitcast = mlir::dyn_cast<mlir::LLVM::BitcastOp>(def)) {
    ownership::aggregate::Slot::markLoad(bitcast.getArg());
    return;
  }
  if (auto intToPtr = mlir::dyn_cast<mlir::LLVM::IntToPtrOp>(def)) {
    ownership::aggregate::Slot::markLoad(intToPtr.getArg());
    return;
  }
  if (auto ptrToInt = mlir::dyn_cast<mlir::LLVM::PtrToIntOp>(def)) {
    ownership::aggregate::Slot::markLoad(ptrToInt.getArg());
    return;
  }
  if (auto cast = mlir::dyn_cast<mlir::UnrealizedConversionCastOp>(def)) {
    if (cast->getNumOperands() == 1)
      ownership::aggregate::Slot::markLoad(cast.getOperand(0));
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
  if (!function || argIndex >= function.getNumArguments())
    return;
  function.setArgAttr(argIndex, attrName,
                      mlir::UnitAttr::get(funcLike->getContext()));
}

void async_runtime::ExceptionCell::markArgument(mlir::Operation *funcLike,
                                                unsigned argIndex) {
  markFunctionArgument(funcLike, argIndex, AsyncSafetyAttrs::kExceptionCell);
}

void async_runtime::CancelFlag::markArgument(mlir::Operation *funcLike,
                                             unsigned argIndex) {
  markFunctionArgument(funcLike, argIndex, AsyncSafetyAttrs::kCancelFlag);
}

void async_runtime::RuntimeHandle::markArgument(mlir::Operation *funcLike,
                                                unsigned argIndex) {
  markFunctionArgument(funcLike, argIndex, AsyncSafetyAttrs::kRuntimeHandle);
}

mlir::Value async_runtime::ExceptionCell::load(mlir::Location loc,
                                               mlir::Value cell,
                                               mlir::OpBuilder &builder) {
  auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  auto load = builder.create<mlir::LLVM::LoadOp>(
      loc, ptrType, cell, /*alignment=*/8, /*isVolatile=*/false,
      /*isNonTemporal=*/false, /*isInvariant=*/false,
      /*isInvariantGroup=*/false, mlir::LLVM::AtomicOrdering::acquire);
  threadsafe::Atomic::set(load.getOperation(),
                          ThreadSafetyAttrs::kRoleAsyncExceptionLoad,
                          ThreadSafetyAttrs::kOrderingAcquire);
  return load;
}

void async_runtime::ExceptionCell::releaseLoaded(
    mlir::Location loc, mlir::ModuleOp module, mlir::OpBuilder &builder,
    const PyLLVMTypeConverter &typeConverter, mlir::Value exception) {
  RuntimeAPI runtime(module, builder, typeConverter);
  auto call = runtime.call(loc, RuntimeSymbols::kDecRef, mlir::Type(),
                           mlir::ValueRange{exception});
  call->setAttr(OwnershipContractAttrs::kAggregateRelease,
                builder.getUnitAttr());
}

void async_runtime::ExceptionCell::free(
    mlir::Location loc, mlir::ModuleOp module, mlir::OpBuilder &builder,
    const PyLLVMTypeConverter &typeConverter, mlir::Value exceptionCell) {
  RuntimeAPI runtime(module, builder, typeConverter);
  runtime.call(loc, RuntimeSymbols::kMemFree, mlir::Type(),
               mlir::ValueRange{exceptionCell});
}

void async_runtime::ExceptionCell::destroy(
    mlir::Location loc, mlir::ModuleOp module, mlir::OpBuilder &builder,
    const PyLLVMTypeConverter &typeConverter, mlir::Value exceptionCell) {
  mlir::Value exception = load(loc, exceptionCell, builder);
  releaseLoaded(loc, module, builder, typeConverter, exception);
  free(loc, module, builder, typeConverter, exceptionCell);
}

void async_runtime::ExceptionCell::storeFirst(
    mlir::Location loc, mlir::Value cell, mlir::Value exception,
    mlir::ModuleOp module, mlir::RewriterBase &rewriter,
    const PyLLVMTypeConverter &typeConverter, llvm::StringRef retainPremise) {
  RuntimeAPI runtime(module, rewriter, typeConverter);
  auto retain = runtime.call(loc, RuntimeSymbols::kIncRef, mlir::Type(),
                             mlir::ValueRange{exception});
  threadsafe::Retain::premise(retain.getOperation(), retainPremise);
  auto ptrType = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
  mlir::Value nullPtr = rewriter.create<mlir::LLVM::ZeroOp>(loc, ptrType);
  auto cmpxchg = rewriter.create<mlir::LLVM::AtomicCmpXchgOp>(
      loc, cell, nullPtr, exception, mlir::LLVM::AtomicOrdering::acq_rel,
      mlir::LLVM::AtomicOrdering::acquire, llvm::StringRef(), /*alignment=*/8);
  threadsafe::Atomic::set(cmpxchg.getOperation(),
                          ThreadSafetyAttrs::kRoleAsyncExceptionStore,
                          ThreadSafetyAttrs::kOrderingAcqRel);
  mlir::Value success = rewriter.create<mlir::LLVM::ExtractValueOp>(
      loc, rewriter.getI1Type(), cmpxchg.getRes(),
      rewriter.getDenseI64ArrayAttr({1}));

  mlir::Block *currentBlock = rewriter.getInsertionBlock();
  mlir::Block *afterBlock =
      rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
  mlir::Block *failedBlock =
      rewriter.createBlock(afterBlock->getParent(), afterBlock->getIterator());
  rewriter.setInsertionPointToEnd(currentBlock);
  rewriter.create<mlir::cf::CondBranchOp>(loc, success, afterBlock,
                                          failedBlock);

  rewriter.setInsertionPointToStart(failedBlock);
  runtime.call(loc, RuntimeSymbols::kDecRef, mlir::Type(),
               mlir::ValueRange{exception});
  rewriter.create<mlir::cf::BranchOp>(loc, afterBlock);

  rewriter.setInsertionPointToStart(afterBlock);
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

void collectAsyncArgProvenanceContracts(
    mlir::ModuleOp module,
    llvm::SmallVectorImpl<AsyncArgProvenanceContract> &contracts) {
  module.walk([&](mlir::func::FuncOp func) {
    llvm::StringRef name = func.getSymName();
    unsigned flattenedIndex = 0;
    for (unsigned arg = 0, e = func.getFunctionType().getNumInputs(); arg < e;
         ++arg) {
      mlir::Type inputType = func.getFunctionType().getInput(arg);
      if (func.getArgAttr(arg, AsyncSafetyAttrs::kRuntimeHandle))
        contracts.push_back({name.str(), flattenedIndex,
                             AsyncArgProvenanceKind::RuntimeHandle});
      if (func.getArgAttr(arg, AsyncSafetyAttrs::kExceptionCell))
        contracts.push_back({name.str(), flattenedIndex,
                             AsyncArgProvenanceKind::ExceptionCell});
      if (func.getArgAttr(arg, AsyncSafetyAttrs::kCancelFlag)) {
        unsigned targetIndex = flattenedIndex;
        if (mlir::isa<mlir::MemRefType>(inputType))
          targetIndex = flattenedIndex + 1;
        contracts.push_back(
            {name.str(), targetIndex, AsyncArgProvenanceKind::CancelFlag});
      }
      flattenedIndex += getFinalLLVMFunctionInputCount(inputType);
    }
  });
}

void collectAsyncArgProvenanceContracts(
    mlir::ModuleOp module, const PyLLVMTypeConverter &typeConverter,
    llvm::SmallVectorImpl<AsyncArgProvenanceContract> &contracts) {
  module.walk([&](mlir::func::FuncOp func) {
    llvm::StringRef name = func.getSymName();
    unsigned flattenedIndex = 0;
    for (unsigned arg = 0, e = func.getFunctionType().getNumInputs(); arg < e;
         ++arg) {
      mlir::Type inputType = func.getFunctionType().getInput(arg);
      llvm::SmallVector<mlir::Type, 4> converted;
      if (mlir::failed(typeConverter.convertType(inputType, converted)) ||
          converted.empty())
        return;

      if (func.getArgAttr(arg, AsyncSafetyAttrs::kRuntimeHandle))
        contracts.push_back({name.str(), flattenedIndex,
                             AsyncArgProvenanceKind::RuntimeHandle});
      if (func.getArgAttr(arg, AsyncSafetyAttrs::kExceptionCell))
        contracts.push_back({name.str(), flattenedIndex,
                             AsyncArgProvenanceKind::ExceptionCell});
      if (func.getArgAttr(arg, AsyncSafetyAttrs::kCancelFlag)) {
        unsigned targetIndex = flattenedIndex;
        if (mlir::isa<mlir::MemRefType>(inputType))
          targetIndex = flattenedIndex + 1;
        contracts.push_back(
            {name.str(), targetIndex, AsyncArgProvenanceKind::CancelFlag});
      }
      flattenedIndex += mlir::isa<mlir::MemRefType>(inputType)
                            ? getFinalLLVMFunctionInputCount(inputType)
                            : static_cast<unsigned>(converted.size());
    }
  });
}

void collectNonObjectArgContracts(
    mlir::ModuleOp module,
    llvm::SmallVectorImpl<NonObjectArgContract> &contracts) {
  module.walk([&](mlir::func::FuncOp func) {
    llvm::StringRef name = func.getSymName();
    unsigned flattenedIndex = 0;
    for (unsigned arg = 0, e = func.getFunctionType().getNumInputs(); arg < e;
         ++arg) {
      mlir::Type inputType = func.getFunctionType().getInput(arg);
      collectMemRefNonObjectArgs(name, func, arg, flattenedIndex, inputType,
                                 contracts);
      flattenedIndex += getFinalLLVMFunctionInputCount(inputType);
    }
  });
}

void collectNonObjectArgContracts(
    mlir::ModuleOp module, const PyLLVMTypeConverter &typeConverter,
    llvm::SmallVectorImpl<NonObjectArgContract> &contracts) {
  module.walk([&](mlir::func::FuncOp func) {
    llvm::StringRef name = func.getSymName();
    unsigned flattenedIndex = 0;
    for (unsigned arg = 0, e = func.getFunctionType().getNumInputs(); arg < e;
         ++arg) {
      mlir::Type inputType = func.getFunctionType().getInput(arg);
      llvm::SmallVector<mlir::Type, 4> converted;
      if (mlir::failed(typeConverter.convertType(inputType, converted)) ||
          converted.empty())
        return;

      collectMemRefNonObjectArgs(name, func, arg, flattenedIndex, inputType,
                                 contracts);
      flattenedIndex +=
          mlir::isa<mlir::MemRefType, mlir::UnrankedMemRefType>(inputType)
              ? getFinalLLVMFunctionInputCount(inputType)
              : static_cast<unsigned>(converted.size());
    }
  });
}

mlir::LogicalResult preserveLLVMNonObjectArgContracts(
    mlir::ModuleOp module, llvm::ArrayRef<NonObjectArgContract> contracts) {
  bool failedAny = false;
  for (const NonObjectArgContract &contract : contracts) {
    mlir::Operation *funcLike = nullptr;
    if (auto func =
            module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(contract.symbolName))
      funcLike = func.getOperation();
    else if (auto func =
                 module.lookupSymbol<mlir::func::FuncOp>(contract.symbolName))
      funcLike = func.getOperation();
    else if (auto func =
                 module.lookupSymbol<mlir::async::FuncOp>(contract.symbolName))
      funcLike = func.getOperation();

    if (!funcLike) {
      mlir::emitError(module.getLoc())
          << "non-object argument contract for @" << contract.symbolName
          << " was not preserved through LLVM lowering";
      failedAny = true;
      continue;
    }
    markFunctionArgument(funcLike, contract.argIndex,
                         OwnershipContractAttrs::kNonObjectPointer);
  }
  return mlir::failure(failedAny);
}

mlir::LogicalResult preserveLLVMAsyncArgProvenanceContracts(
    mlir::ModuleOp module,
    llvm::ArrayRef<AsyncArgProvenanceContract> contracts) {
  bool failedAny = false;
  for (const AsyncArgProvenanceContract &contract : contracts) {
    mlir::Operation *funcLike = nullptr;
    if (auto func =
            module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(contract.symbolName))
      funcLike = func.getOperation();
    else if (auto func =
                 module.lookupSymbol<mlir::func::FuncOp>(contract.symbolName))
      funcLike = func.getOperation();
    else if (auto func =
                 module.lookupSymbol<mlir::async::FuncOp>(contract.symbolName))
      funcLike = func.getOperation();

    if (!funcLike) {
      mlir::emitError(module.getLoc())
          << "async argument provenance contract for @" << contract.symbolName
          << " was not preserved through LLVM lowering";
      failedAny = true;
      continue;
    }
    switch (contract.kind) {
    case AsyncArgProvenanceKind::RuntimeHandle:
      async_runtime::RuntimeHandle::markArgument(funcLike, contract.argIndex);
      break;
    case AsyncArgProvenanceKind::ExceptionCell:
      async_runtime::ExceptionCell::markArgument(funcLike, contract.argIndex);
      break;
    case AsyncArgProvenanceKind::CancelFlag:
      async_runtime::CancelFlag::markArgument(funcLike, contract.argIndex);
      break;
    }
  }
  return mlir::failure(failedAny);
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

mlir::LogicalResult preserveLoweredMemRefAtomicContracts(
    mlir::ModuleOp module, llvm::ArrayRef<MemRefAtomicContract> contracts) {
  llvm::SmallVector<bool> used(contracts.size(), false);
  auto applyContract = [](mlir::LLVM::AtomicRMWOp atomic,
                          const MemRefAtomicContract &contract) {
    threadsafe::Atomic::set(atomic.getOperation(), contract.role.getValue(),
                            contract.ordering.getValue(),
                            contract.retainPremise
                                ? contract.retainPremise.getValue()
                                : llvm::StringRef{});
    llvm::StringRef component =
        contract.component ? contract.component.getValue() : llvm::StringRef{};
    bool isContainerAtomic = contract.role.getValue().starts_with("container.");
    if (component.empty() && isContainerAtomic)
      component = ContainerSafetyAttrs::kComponentHeader;
    if (isContainerAtomic) {
      threadsafe::memref::Atomic::set(
          atomic.getOperation(), component, contract.slot,
          contract.group ? contract.group.getValue() : llvm::StringRef{},
          contract.containerKind ? contract.containerKind.getValue()
                                 : llvm::StringRef{});
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

  bool failedAny = false;
  module.walk([&](mlir::LLVM::AtomicRMWOp atomic) {
    if (auto idAttr =
            atomic->getAttrOfType<mlir::IntegerAttr>(kMemRefAtomicContractId)) {
      int64_t id = idAttr.getInt();
      for (auto indexed : llvm::enumerate(contracts)) {
        unsigned index = indexed.index();
        if (indexed.value().id != id)
          continue;
        applyContract(atomic, indexed.value());
        used[index] = true;
        return;
      }
      atomic->emitOpError("carries an unknown memref atomic contract id");
      failedAny = true;
      return;
    }
  });

  for (auto indexed : llvm::enumerate(contracts)) {
    if (used[indexed.index()])
      continue;
    mlir::emitError(indexed.value().location)
        << "memref atomic contract was not preserved through LLVM lowering";
    failedAny = true;
  }
  return mlir::failure(failedAny);
}

void collectMemRefAggregateLoadContracts(
    mlir::ModuleOp module,
    llvm::SmallVectorImpl<MemRefAggregateLoadContract> &contracts) {
  mlir::OpBuilder builder(module.getContext());
  int64_t nextId = 0;
  module.walk([&](mlir::memref::LoadOp load) {
    if (!load->hasAttr(OwnershipContractAttrs::kAggregateSlotLoad))
      return;
    auto group = load->getAttrOfType<mlir::StringAttr>(
        OwnershipContractAttrs::kAggregateSlotGroup);
    if (!group)
      if (auto groupName = getLoweringValueDefStringAttr(
              load.getMemref(), ContainerSafetyAttrs::kDescriptorGroup))
        group = builder.getStringAttr(*groupName);
    auto component = load->getAttrOfType<mlir::StringAttr>(
        OwnershipContractAttrs::kAggregateSlotComponent);
    if (!component)
      if (auto componentName = getLoweringValueDefStringAttr(
              load.getMemref(), ContainerSafetyAttrs::kDescriptorComponent))
        component = builder.getStringAttr(*componentName);
    std::optional<int64_t> slot;
    if (auto slotAttr = load->getAttrOfType<mlir::IntegerAttr>(
            OwnershipContractAttrs::kAggregateSlotIndex))
      slot = slotAttr.getInt();
    else if (load.getIndices().size() == 1)
      slot = getArithConstantInt(load.getIndices().front());
    int64_t id = nextId++;
    load->setAttr(kMemRefAggregateLoadContractId,
                  builder.getI64IntegerAttr(id));
    contracts.push_back(
        MemRefAggregateLoadContract{id, load.getLoc(), group, component, slot});
  });
}

mlir::LogicalResult preserveLoweredMemRefAggregateLoadContracts(
    mlir::ModuleOp module,
    llvm::ArrayRef<MemRefAggregateLoadContract> contracts) {
  llvm::SmallVector<bool> used(contracts.size(), false);
  auto applyContract = [](mlir::LLVM::LoadOp load,
                          const MemRefAggregateLoadContract &contract) {
    mlir::OpBuilder builder(load.getContext());
    load->setAttr(OwnershipContractAttrs::kAggregateSlotLoad,
                  builder.getUnitAttr());
    if (contract.group)
      load->setAttr(OwnershipContractAttrs::kAggregateSlotGroup,
                    contract.group);
    if (contract.component)
      load->setAttr(OwnershipContractAttrs::kAggregateSlotComponent,
                    contract.component);
    if (contract.slot)
      load->setAttr(OwnershipContractAttrs::kAggregateSlotIndex,
                    builder.getI64IntegerAttr(*contract.slot));
    load->removeAttr(kMemRefAggregateLoadContractId);
  };

  bool failedAny = false;
  module.walk([&](mlir::LLVM::LoadOp load) {
    if (auto idAttr = load->getAttrOfType<mlir::IntegerAttr>(
            kMemRefAggregateLoadContractId)) {
      int64_t id = idAttr.getInt();
      for (auto indexed : llvm::enumerate(contracts)) {
        unsigned index = indexed.index();
        if (indexed.value().id != id)
          continue;
        applyContract(load, indexed.value());
        used[index] = true;
        return;
      }
      load->emitOpError("carries an unknown aggregate load contract id");
      failedAny = true;
      return;
    }
    if (!load->hasAttr(OwnershipContractAttrs::kAggregateSlotLoad))
      return;
  });

  for (auto indexed : llvm::enumerate(contracts)) {
    if (used[indexed.index()])
      continue;
    mlir::emitError(indexed.value().location)
        << "aggregate slot load contract was not preserved through LLVM "
           "lowering";
    failedAny = true;
  }
  return mlir::failure(failedAny);
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
  llvm::SmallVector<bool> used(contracts.size(), false);
  auto applyContract = [](mlir::Operation *op,
                          const MemRefContainerAccessContract &contract) {
    op->setAttr(ContainerSafetyAttrs::kAccessGroup, contract.group);
    if (contract.component)
      op->setAttr(ContainerSafetyAttrs::kAccessComponent, contract.component);
    op->removeAttr(kMemRefContainerAccessContractId);
  };

  bool failedAny = false;
  auto match = [&](mlir::Operation *op, bool store, mlir::Value addr) {
    if (auto idAttr = op->getAttrOfType<mlir::IntegerAttr>(
            kMemRefContainerAccessContractId)) {
      int64_t id = idAttr.getInt();
      for (auto indexed : llvm::enumerate(contracts)) {
        unsigned index = indexed.index();
        if (indexed.value().id != id || indexed.value().store != store)
          continue;
        applyContract(op, indexed.value());
        used[index] = true;
        return;
      }
      op->emitOpError("carries an unknown container access contract id");
      failedAny = true;
      return;
    }
    (void)addr;
  };

  module.walk([&](mlir::LLVM::LoadOp load) {
    match(load.getOperation(), /*store=*/false, load.getAddr());
  });
  module.walk([&](mlir::LLVM::StoreOp store) {
    match(store.getOperation(), /*store=*/true, store.getAddr());
  });

  for (auto indexed : llvm::enumerate(contracts)) {
    if (used[indexed.index()])
      continue;
    mlir::emitError(indexed.value().location)
        << "managed container access contract was not preserved through LLVM "
           "lowering";
    failedAny = true;
  }
  return mlir::failure(failedAny);
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
    if (groupName.empty())
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
        id, dealloc.getLoc(), builder.getStringAttr(groupName),
        explicitComponent
            ? explicitComponent
            : (inferredComponent ? builder.getStringAttr(*inferredComponent)
                                 : mlir::StringAttr{})});
  });
}

static bool isFreeCall(mlir::LLVM::CallOp call) {
  auto callee = call.getCallee();
  return callee && *callee == "free";
}

mlir::LogicalResult preserveLoweredMemRefDeallocContracts(
    mlir::ModuleOp module, llvm::ArrayRef<MemRefDeallocContract> contracts) {
  llvm::SmallVector<bool> used(contracts.size(), false);
  auto applyContract = [](mlir::LLVM::CallOp call,
                          const MemRefDeallocContract &contract) {
    call->setAttr(ContainerSafetyAttrs::kDeallocGroup, contract.group);
    if (contract.component)
      call->setAttr(ContainerSafetyAttrs::kDeallocComponent,
                    contract.component);
    call->removeAttr(kMemRefDeallocContractId);
  };

  bool failedAny = false;
  module.walk([&](mlir::LLVM::CallOp call) {
    if (!isFreeCall(call))
      return;
    if (auto idAttr =
            call->getAttrOfType<mlir::IntegerAttr>(kMemRefDeallocContractId)) {
      int64_t id = idAttr.getInt();
      for (auto indexed : llvm::enumerate(contracts)) {
        unsigned index = indexed.index();
        if (indexed.value().id != id)
          continue;
        applyContract(call, indexed.value());
        used[index] = true;
        return;
      }
      call->emitOpError("carries an unknown memref dealloc contract id");
      failedAny = true;
      return;
    }
  });

  for (auto indexed : llvm::enumerate(contracts)) {
    if (used[indexed.index()])
      continue;
    mlir::emitError(indexed.value().location)
        << "managed container dealloc contract was not preserved through LLVM "
           "lowering";
    failedAny = true;
  }
  return mlir::failure(failedAny);
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
         mlir::isa<ObjectType>(type) || mlir::isa<CoroutineType>(type) ||
         mlir::isa<FutureType>(type) || mlir::isa<TaskType>(type);
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
  if (!mlir::isa<mlir::LLVM::LLVMPointerType>(inputs[1].getType()))
    return false;
  return !expectsCancelFlag || mlir::isa<mlir::MemRefType>(inputs[2].getType());
}

PyLLVMTypeConverter::PyLLVMTypeConverter(mlir::MLIRContext *ctx)
    : mlir::LLVMTypeConverter(ctx) {
  pyObjectPtrType = mlir::LLVM::LLVMPointerType::get(ctx);

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

  addConversion([this, convertAsyncPayload](
                    mlir::Type type) -> std::optional<mlir::Type> {
    if (mlir::isa<ListType, DictType, TupleType, CoroutineType, TaskType,
                  FutureType>(type))
      return std::nullopt;
    if (auto asyncValueType = mlir::dyn_cast<mlir::async::ValueType>(type)) {
      auto valueType = convertAsyncPayload(asyncValueType.getValueType());
      if (!valueType)
        return std::nullopt;
      return mlir::async::ValueType::get(*valueType);
    }
    if ((isPyType(type) &&
         !mlir::isa<CoroutineType, TaskType, FutureType>(type)) ||
        mlir::isa<FuncType>(type) || mlir::isa<TupleType>(type) ||
        mlir::isa<ClassType>(type) || mlir::isa<DictType>(type) ||
        mlir::isa<ObjectType>(type))
      return pyObjectPtrType;
    if (mlir::isa<NoneType>(type))
      return pyObjectPtrType;
    return std::nullopt;
  });

  addConversion([ctx, convertAsyncPayload](
                    mlir::Type type, mlir::SmallVectorImpl<mlir::Type> &results)
                    -> std::optional<mlir::LogicalResult> {
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
    results.push_back(mlir::LLVM::LLVMPointerType::get(ctx));
    if (!includeCancelFlag)
      return mlir::success();
    results.push_back(
        mlir::MemRefType::get({1}, mlir::IntegerType::get(ctx, 8)));
    return mlir::success();
  });

  addConversion(
      [ctx](ListType listType, mlir::SmallVectorImpl<mlir::Type> &results)
          -> std::optional<mlir::LogicalResult> {
        auto itemsType = getListItemsMemRefType(listType.getElementType(), ctx);
        if (!itemsType)
          return std::nullopt;
        results.push_back(getListHeaderMemRefType(ctx));
        results.push_back(itemsType);
        return mlir::success();
      });

  addConversion(
      [ctx](TupleType tupleType, mlir::SmallVectorImpl<mlir::Type> &results)
          -> std::optional<mlir::LogicalResult> {
        results.push_back(getTupleHeaderMemRefType(ctx));
        results.push_back(getTupleItemsMemRefType(tupleType, ctx));
        return mlir::success();
      });

  addConversion(
      [ctx](DictType dictType, mlir::SmallVectorImpl<mlir::Type> &results)
          -> std::optional<mlir::LogicalResult> {
        if (container::Slot::supported(dictType.getKeyType()) &&
            container::Slot::supported(dictType.getValueType())) {
          results.push_back(getDictHeaderMemRefType(ctx));
          results.push_back(getDictKeysMemRefType(dictType, ctx));
          results.push_back(getDictValuesMemRefType(dictType, ctx));
          results.push_back(getDictStatesMemRefType(ctx));
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

RuntimeAPI::RuntimeAPI(mlir::ModuleOp module, mlir::OpBuilder &rewriter,
                       const PyLLVMTypeConverter &typeConverter)
    : module(module), rewriter(rewriter),
      pyObjectPtrType(typeConverter.getPyObjectPtrType()) {}

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

mlir::LLVM::CallOp RuntimeAPI::call(mlir::Location loc, llvm::StringRef name,
                                    mlir::Type resultType,
                                    mlir::ValueRange operands) {
  mlir::SmallVector<mlir::Type> operandTypes;
  operandTypes.reserve(operands.size());
  for (mlir::Value operand : operands)
    operandTypes.push_back(operand.getType());

  mlir::Type actualResult =
      resultType ? resultType
                 : mlir::LLVM::LLVMVoidType::get(module.getContext());
  auto callee = declareRuntimeFunc(loc, module, rewriter, name, actualResult,
                                   operandTypes);
  ownership::llvm_func::Contract::apply(callee, name);
  auto symbolRef =
      mlir::SymbolRefAttr::get(module.getContext(), callee.getName());
  bool isVoid = llvm::isa<mlir::LLVM::LLVMVoidType>(actualResult);
  llvm::SmallVector<mlir::Type, 1> resultStorage;
  if (!isVoid)
    resultStorage.push_back(actualResult);
  mlir::TypeRange results(resultStorage);
  return rewriter.create<mlir::LLVM::CallOp>(loc, results, symbolRef, operands);
}

mlir::Value RuntimeAPI::getStringLiteral(mlir::Location loc,
                                         mlir::StringAttr literal) {
  llvm::SmallString<32> symbolName("__ly_str_");
  auto hashValue = static_cast<uint64_t>(llvm::hash_value(literal.getValue()));
  symbolName += llvm::formatv("{0:X}", hashValue).str();

  mlir::LLVM::GlobalOp global =
      module.lookupSymbol<mlir::LLVM::GlobalOp>(symbolName);
  if (!global) {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    auto arrayType = mlir::LLVM::LLVMArrayType::get(
        rewriter.getI8Type(), literal.getValue().size() + 1);

    llvm::SmallString<32> storage(literal.getValue());
    storage.push_back('\0');
    global = rewriter.create<mlir::LLVM::GlobalOp>(
        loc, arrayType, /*isConstant=*/true, mlir::LLVM::Linkage::Internal,
        symbolName, rewriter.getStringAttr(storage));
  }

  auto ptrType = mlir::LLVM::LLVMPointerType::get(module.getContext());
  mlir::Value addr = rewriter.create<mlir::LLVM::AddressOfOp>(
      loc, ptrType, global.getSymNameAttr());
  mlir::Value zero = rewriter.create<mlir::LLVM::ConstantOp>(
      loc, rewriter.getI64Type(), rewriter.getIndexAttr(0));
  return rewriter.create<mlir::LLVM::GEPOp>(
      loc, ptrType, global.getType(), addr,
      llvm::ArrayRef<mlir::Value>{zero, zero});
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
