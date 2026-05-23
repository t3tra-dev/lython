#pragma once

#include "Common/RuntimeSupport.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"

#include <cstdint>
#include <optional>

namespace py {

enum class KindId {
  Tuple,
  List,
  Dict,
};

static constexpr int64_t kTupleHeaderSize = 3;
static constexpr int64_t kListHeaderSize = 4;
static constexpr int64_t kDictHeaderSize = 5;

static constexpr int64_t kTypedTupleSizeSlot = 0;
static constexpr int64_t kTypedTupleRefcountSlot = 2;

static constexpr int64_t kTypedListSizeSlot = 0;
static constexpr int64_t kTypedListCapacitySlot = 1;
static constexpr int64_t kTypedListLockSlot = 2;
static constexpr int64_t kTypedListRefcountSlot = 3;

static constexpr int64_t kTypedDictSizeSlot = 0;
static constexpr int64_t kTypedDictCapacitySlot = 1;
static constexpr int64_t kTypedDictTombstoneSlot = 2;
static constexpr int64_t kTypedDictLockSlot = 3;
static constexpr int64_t kTypedDictRefcountSlot = 4;

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

  static std::optional<KindId> fromHeaderType(mlir::Type type) {
    auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type);
    if (!memrefType || memrefType.getRank() != 1 ||
        !memrefType.hasStaticShape() ||
        !memrefType.getElementType().isInteger(64))
      return std::nullopt;

    switch (memrefType.getDimSize(0)) {
    case kTupleHeaderSize:
      return KindId::Tuple;
    case kListHeaderSize:
      return KindId::List;
    case kDictHeaderSize:
      return KindId::Dict;
    default:
      return std::nullopt;
    }
  }

  static std::optional<KindId> fromHeader(mlir::Value header) {
    if (!header)
      return std::nullopt;
    return fromHeaderType(header.getType());
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

  static std::optional<int64_t> slot(mlir::Type type) {
    auto kind = Kind::fromHeaderType(type);
    if (!kind)
      return std::nullopt;
    return slot(*kind);
  }

  static std::optional<int64_t> slot(mlir::Value header) {
    if (!header)
      return std::nullopt;
    return slot(header.getType());
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

struct Lock {
  static std::optional<int64_t> slot(KindId kind) {
    switch (kind) {
    case KindId::Tuple:
      return std::nullopt;
    case KindId::List:
      return kTypedListLockSlot;
    case KindId::Dict:
      return kTypedDictLockSlot;
    }
    llvm_unreachable("unknown typed container kind");
  }

  static std::optional<int64_t> slot(mlir::Type type) {
    auto kind = Kind::fromHeaderType(type);
    if (!kind)
      return std::nullopt;
    return slot(*kind);
  }

  static std::optional<int64_t> slot(mlir::Value header) {
    if (!header)
      return std::nullopt;
    return slot(header.getType());
  }

  static std::optional<int64_t> slotForKindName(llvm::StringRef kindName) {
    auto kind = Kind::fromName(kindName);
    if (!kind)
      return std::nullopt;
    return slot(*kind);
  }
};

struct Descriptor {
  static unsigned componentCount(mlir::Type type) {
    auto kind = Kind::fromHeaderType(type);
    if (!kind)
      return 1;

    switch (*kind) {
    case KindId::Tuple:
    case KindId::List:
      return 2;
    case KindId::Dict:
      return 4;
    }
    llvm_unreachable("unknown typed container kind");
  }

  static unsigned componentCount(mlir::Value header) {
    if (!header)
      return 1;
    return componentCount(header.getType());
  }

  static mlir::FailureOr<mlir::SmallVector<mlir::Value>>
  promote(mlir::Location loc, mlir::Type logicalType, mlir::ValueRange input,
          mlir::ModuleOp module, mlir::ConversionPatternRewriter &rewriter,
          const PyLLVMTypeConverter &typeConverter, bool cloneReferenceSlots);
};

} // namespace container

} // namespace py
