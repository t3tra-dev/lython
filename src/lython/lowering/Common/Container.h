#pragma once

#include "Common/RuntimeSupport.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"

#include <cstdint>
#include <optional>

namespace py {

static inline std::optional<llvm::StringRef>
functionCallResultStringAttr(mlir::OpResult result, llvm::StringRef attrName) {
  auto call = mlir::dyn_cast<mlir::func::CallOp>(result.getOwner());
  if (!call)
    return std::nullopt;
  mlir::ModuleOp module = call->getParentOfType<mlir::ModuleOp>();
  if (!module)
    return std::nullopt;
  auto callee = module.lookupSymbol<mlir::func::FuncOp>(call.getCallee());
  if (!callee || result.getResultNumber() >= callee.getNumResults())
    return std::nullopt;
  auto attr = mlir::dyn_cast_or_null<mlir::StringAttr>(
      callee.getResultAttr(result.getResultNumber(), attrName));
  if (!attr)
    return std::nullopt;
  return attr.getValue();
}

enum class KindId {
  Tuple,
  List,
  Dict,
};

static constexpr int64_t kTupleHeaderSize = 2;
static constexpr int64_t kListHeaderSize = 3;
static constexpr int64_t kDictHeaderSize = 4;

static constexpr int64_t kTypedTupleSizeSlot = 0;
static constexpr int64_t kTypedTupleRefcountSlot = 1;

static constexpr int64_t kTypedListSizeSlot = 0;
static constexpr int64_t kTypedListCapacitySlot = 1;
static constexpr int64_t kTypedListRefcountSlot = 2;

static constexpr int64_t kTypedDictSizeSlot = 0;
static constexpr int64_t kTypedDictCapacitySlot = 1;
static constexpr int64_t kTypedDictTombstoneSlot = 2;
static constexpr int64_t kTypedDictRefcountSlot = 3;

static constexpr int64_t kTypedContainerLockSlot = 0;

static constexpr unsigned kTupleHeaderComponent = 0;
static constexpr unsigned kTupleItemsComponent = 1;

static constexpr unsigned kListHeaderComponent = 0;
static constexpr unsigned kListLockComponent = 1;
static constexpr unsigned kListItemsComponent = 2;

static constexpr unsigned kDictHeaderComponent = 0;
static constexpr unsigned kDictLockComponent = 1;
static constexpr unsigned kDictKeysComponent = 2;
static constexpr unsigned kDictValuesComponent = 3;
static constexpr unsigned kDictStatesComponent = 4;

namespace container {
struct Kind {
  static llvm::StringRef name(KindId kind) {
    switch (kind) {
    case KindId::Tuple:
      return ContainerSafetyAttrs::kKindTuple;
    case KindId::List:
      return ContainerSafetyAttrs::kKindList;
    case KindId::Dict:
      return ContainerSafetyAttrs::kKindDict;
    }
    llvm_unreachable("unknown typed container kind");
  }

  static std::optional<KindId> fromName(llvm::StringRef kind) {
    if (kind == ContainerSafetyAttrs::kKindTuple)
      return KindId::Tuple;
    if (kind == ContainerSafetyAttrs::kKindList)
      return KindId::List;
    if (kind == ContainerSafetyAttrs::kKindDict)
      return KindId::Dict;
    return std::nullopt;
  }

  static std::optional<KindId>
  fromHeader(mlir::Value header, llvm::SmallPtrSetImpl<mlir::Value> &seen) {
    if (!header || !seen.insert(header).second)
      return std::nullopt;

    // Fixed header width is not a semantic proof. Container headers carry
    // descriptor attrs because their slots encode container state, not generic
    // object-header state.
    // Function/block arguments must carry descriptor attrs; otherwise the value
    // is not a proven container header.
    if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(header)) {
      mlir::Operation *parent =
          arg.getOwner() ? arg.getOwner()->getParentOp() : nullptr;
      auto function = mlir::dyn_cast_or_null<mlir::FunctionOpInterface>(parent);
      if (function) {
        unsigned index = arg.getArgNumber();
        if (index >= function.getNumArguments())
          return std::nullopt;
        if (function.getArgAttr(index, OwnershipContractAttrs::kObjectHeader))
          return std::nullopt;
        if (auto kind =
                mlir::dyn_cast_or_null<mlir::StringAttr>(function.getArgAttr(
                    index, ContainerSafetyAttrs::kDescriptorKind)))
          return fromName(kind.getValue());
        if (auto group =
                mlir::dyn_cast_or_null<mlir::StringAttr>(function.getArgAttr(
                    index, ContainerSafetyAttrs::kDescriptorGroup))) {
          llvm::StringRef kind =
              group.getValue().take_until([](char c) { return c == '.'; });
          if (auto parsed = fromName(kind))
            return parsed;
        }
      }
      return std::nullopt;
    }

    mlir::Operation *def = header.getDefiningOp();
    if (!def)
      return std::nullopt;

    if (def->hasAttr(OwnershipContractAttrs::kObjectHeader))
      return std::nullopt;

    if (auto kind = def->getAttrOfType<mlir::StringAttr>(
            ContainerSafetyAttrs::kDescriptorKind))
      return fromName(kind.getValue());

    if (auto result = mlir::dyn_cast<mlir::OpResult>(header))
      if (auto kind = functionCallResultStringAttr(
              result, ContainerSafetyAttrs::kDescriptorKind))
        return fromName(*kind);

    if (auto group = def->getAttrOfType<mlir::StringAttr>(
            ContainerSafetyAttrs::kDescriptorGroup)) {
      llvm::StringRef kind =
          group.getValue().take_until([](char c) { return c == '.'; });
      if (auto parsed = fromName(kind))
        return parsed;
    }

    if (auto result = mlir::dyn_cast<mlir::OpResult>(header)) {
      if (auto group = functionCallResultStringAttr(
              result, ContainerSafetyAttrs::kDescriptorGroup)) {
        llvm::StringRef kind =
            group->take_until([](char c) { return c == '.'; });
        if (auto parsed = fromName(kind))
          return parsed;
      }
    }

    if (auto cast = mlir::dyn_cast<mlir::memref::CastOp>(def))
      return fromHeader(cast.getSource(), seen);

    if (auto subview = mlir::dyn_cast<mlir::memref::SubViewOp>(def))
      return fromHeader(subview.getSource(), seen);

    if (auto cast = mlir::dyn_cast<mlir::UnrealizedConversionCastOp>(def))
      if (cast->getNumOperands() == 1)
        return fromHeader(cast.getOperand(0), seen);

    return std::nullopt;
  }

  static std::optional<KindId> fromHeader(mlir::Value header) {
    llvm::SmallPtrSet<mlir::Value, 8> seen;
    return fromHeader(header, seen);
  }

  static std::optional<llvm::StringRef> nameFromHeader(mlir::Value header) {
    auto kind = fromHeader(header);
    if (!kind)
      return std::nullopt;
    return name(*kind);
  }
};

struct Refcount {
  static std::optional<int64_t> slot(KindId kind) {
    switch (kind) {
    case KindId::Tuple:
      return kTypedTupleRefcountSlot;
    case KindId::List:
      return kTypedListRefcountSlot;
    case KindId::Dict:
      return kTypedDictRefcountSlot;
    }
    llvm_unreachable("unknown typed container kind");
  }

  static std::optional<int64_t> slot(mlir::Value header) {
    if (!header)
      return std::nullopt;
    auto kind = Kind::fromHeader(header);
    if (!kind)
      return std::nullopt;
    return slot(*kind);
  }

  static std::optional<int64_t> slotForLogicalType(mlir::Type type) {
    if (mlir::isa<ListType>(type))
      return kTypedListRefcountSlot;
    if (mlir::isa<TupleType>(type))
      return kTypedTupleRefcountSlot;
    if (mlir::isa<DictType>(type))
      return kTypedDictRefcountSlot;
    return std::nullopt;
  }

  static std::optional<int64_t> slotForKindName(llvm::StringRef kindName) {
    auto kind = Kind::fromName(kindName);
    if (!kind)
      return std::nullopt;
    return slot(*kind);
  }
};

struct Descriptor {
  static std::optional<llvm::StringRef>
  kindNameForLogicalType(mlir::Type type) {
    if (mlir::isa<ListType>(type))
      return ContainerSafetyAttrs::kKindList;
    if (mlir::isa<TupleType>(type))
      return ContainerSafetyAttrs::kKindTuple;
    if (mlir::isa<DictType>(type))
      return ContainerSafetyAttrs::kKindDict;
    return std::nullopt;
  }

  static llvm::StringRef componentForLogicalType(mlir::Type type,
                                                 unsigned slot) {
    if (slot == kTupleHeaderComponent)
      return ContainerSafetyAttrs::kComponentHeader;
    if (mlir::isa<TupleType>(type)) {
      if (slot == kTupleItemsComponent)
        return ContainerSafetyAttrs::kComponentItems;
      return {};
    }
    if (mlir::isa<ListType>(type)) {
      switch (slot) {
      case kListLockComponent:
        return ContainerSafetyAttrs::kComponentLock;
      case kListItemsComponent:
        return ContainerSafetyAttrs::kComponentItems;
      default:
        break;
      }
    }
    if (mlir::isa<DictType>(type)) {
      switch (slot) {
      case kDictLockComponent:
        return ContainerSafetyAttrs::kComponentLock;
      case kDictKeysComponent:
        return ContainerSafetyAttrs::kComponentKeys;
      case kDictValuesComponent:
        return ContainerSafetyAttrs::kComponentValues;
      case kDictStatesComponent:
        return ContainerSafetyAttrs::kComponentStates;
      default:
        break;
      }
    }
    return {};
  }

  static unsigned componentCount(mlir::Value header) {
    if (!header)
      return 1;
    auto kind = Kind::fromHeader(header);
    if (!kind)
      return 1;

    switch (*kind) {
    case KindId::Tuple:
      return 2;
    case KindId::List:
      return 3;
    case KindId::Dict:
      return 5;
    }
    llvm_unreachable("unknown typed container kind");
  }

  static mlir::FailureOr<mlir::SmallVector<mlir::Value>>
  promote(mlir::Location loc, mlir::Type logicalType, mlir::ValueRange input,
          mlir::ModuleOp module, mlir::ConversionPatternRewriter &rewriter,
          const PyLLVMTypeConverter &typeConverter, bool cloneReferenceSlots);
};

} // namespace container

} // namespace py
