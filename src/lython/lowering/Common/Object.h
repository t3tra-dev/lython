#pragma once

#include "PyDialectTypes.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"

#include <cstdint>
#include <optional>

namespace py::object_abi {

// Object-family values must carry their allocation descriptor until the final
// release. The common header is intentionally small: an atomic refcount plus a
// destructor/layout family id for typed-erased release boundaries.
static constexpr int64_t kHeaderSlots = 2;
static constexpr int64_t kRefcountSlot = 0;
static constexpr int64_t kLayoutIdSlot = 1;

static constexpr int64_t kImmortalRefcount = 9223372036854775807LL;

enum class KindId : int64_t {
  None = 0,
  Bool = 1,
  Int = 2,
  Float = 3,
  Str = 4,
  Exception = 5,
  Object = 6,
};

struct Kind {
  static std::optional<KindId> fromType(mlir::Type type) {
    if (mlir::isa<NoneType>(type))
      return KindId::None;
    if (mlir::isa<BoolType>(type))
      return KindId::Bool;
    if (mlir::isa<IntType>(type))
      return KindId::Int;
    if (mlir::isa<FloatType>(type))
      return KindId::Float;
    if (mlir::isa<StrType>(type))
      return KindId::Str;
    if (mlir::isa<ExceptionType>(type))
      return KindId::Exception;
    if (mlir::isa<ClassType>(type))
      return KindId::Object;
    return std::nullopt;
  }

  static llvm::StringRef name(KindId kind) {
    switch (kind) {
    case KindId::None:
      return "none";
    case KindId::Bool:
      return "bool";
    case KindId::Int:
      return "int";
    case KindId::Float:
      return "float";
    case KindId::Str:
      return "str";
    case KindId::Exception:
      return "exception";
    case KindId::Object:
      return "object";
    }
    llvm_unreachable("unknown object ABI kind");
  }
};

struct Type {
  static bool isStorage(mlir::Type type) {
    auto memref = mlir::dyn_cast<mlir::MemRefType>(type);
    if (!memref || memref.getRank() != 1 ||
        !memref.getElementType().isInteger(64))
      return false;
    if (!memref.hasStaticShape() || memref.getShape()[0] != kHeaderSlots)
      return false;
    if (!memref.getLayout())
      return true;
    return mlir::isa<mlir::StridedLayoutAttr>(memref.getLayout());
  }

  static mlir::LLVM::LLVMStructType loweredStorage(mlir::MLIRContext *ctx) {
    auto ptrType = mlir::LLVM::LLVMPointerType::get(ctx);
    auto i64Type = mlir::IntegerType::get(ctx, 64);
    auto arrayType = mlir::LLVM::LLVMArrayType::get(i64Type, 1);
    return mlir::LLVM::LLVMStructType::getLiteral(
        ctx, llvm::ArrayRef<mlir::Type>{ptrType, ptrType, i64Type, arrayType,
                                        arrayType});
  }

  static bool isLoweredStorage(mlir::Type type) {
    auto descriptor = mlir::dyn_cast<mlir::LLVM::LLVMStructType>(type);
    if (!descriptor || descriptor.isOpaque() ||
        descriptor.getBody().size() != 5)
      return false;
    auto body = descriptor.getBody();
    auto sizes = mlir::dyn_cast<mlir::LLVM::LLVMArrayType>(body[3]);
    auto strides = mlir::dyn_cast<mlir::LLVM::LLVMArrayType>(body[4]);
    return mlir::isa<mlir::LLVM::LLVMPointerType>(body[0]) &&
           mlir::isa<mlir::LLVM::LLVMPointerType>(body[1]) &&
           body[2].isInteger(64) && sizes && strides &&
           sizes.getNumElements() == 1 && strides.getNumElements() == 1 &&
           sizes.getElementType().isInteger(64) &&
           strides.getElementType().isInteger(64);
  }

  static bool isStorageLike(mlir::Type type) {
    return isStorage(type) || isLoweredStorage(type);
  }

  static mlir::Value castStorage(mlir::Location loc, mlir::Value value,
                                 mlir::Type targetType,
                                 mlir::OpBuilder &builder) {
    if (!value || !targetType)
      return {};
    if (value.getType() == targetType)
      return value;
    if (mlir::isa<mlir::MemRefType>(value.getType()) && isStorage(targetType))
      return builder.create<mlir::memref::CastOp>(loc, targetType, value);
    return {};
  }
};

struct Header {
  static mlir::MemRefType owned(mlir::MLIRContext *ctx) {
    return mlir::MemRefType::get({kHeaderSlots},
                                 mlir::IntegerType::get(ctx, 64));
  }

  static mlir::MemRefType view(mlir::MLIRContext *ctx) {
    auto layout = mlir::StridedLayoutAttr::get(ctx, mlir::ShapedType::kDynamic,
                                               llvm::ArrayRef<int64_t>{1});
    return mlir::MemRefType::get({kHeaderSlots},
                                 mlir::IntegerType::get(ctx, 64), layout);
  }

  static bool isOwned(mlir::Type type) {
    auto memref = mlir::dyn_cast<mlir::MemRefType>(type);
    return memref && memref.getRank() == 1 && memref.hasStaticShape() &&
           memref.getShape()[0] == kHeaderSlots &&
           memref.getElementType().isInteger(64);
  }

  static bool isView(mlir::Type type) {
    auto memref = mlir::dyn_cast<mlir::MemRefType>(type);
    if (!memref || memref.getRank() != 1 ||
        !memref.getElementType().isInteger(64))
      return false;
    if (!memref.hasStaticShape() || memref.getShape()[0] != kHeaderSlots)
      return false;
    return static_cast<bool>(
        mlir::dyn_cast_or_null<mlir::StridedLayoutAttr>(memref.getLayout()));
  }
};

namespace long_abi {

static constexpr int64_t kLayoutId = 1;
static constexpr int64_t kSignSlot = 0;
static constexpr int64_t kDigitCountSlot = 1;
static constexpr int64_t kMetaSlots = 2;
static constexpr int64_t kDigitsBaseSlot = 0;
static constexpr int64_t kInlineDigits = 3;
static constexpr int64_t kDigitsSlots = kDigitsBaseSlot + kInlineDigits;
static constexpr int64_t kPayloadSlots = kMetaSlots + kDigitsSlots;
static constexpr int64_t kDigitBits = 30;
static constexpr int64_t kDigitMask = (1LL << kDigitBits) - 1;

// Compact LyLong separates narrow metadata from arithmetic digits.  sign and
// digit_count fit in i8, while 30-bit digits stay i32 to avoid byte packing on
// arithmetic hot paths.
struct Meta {
  static mlir::MemRefType storage(mlir::MLIRContext *ctx) {
    return mlir::MemRefType::get({kMetaSlots}, mlir::IntegerType::get(ctx, 8));
  }

  static bool isStorage(mlir::Type type) {
    return type == storage(type.getContext());
  }
};

struct Digits {
  static mlir::MemRefType storage(mlir::MLIRContext *ctx) {
    return mlir::MemRefType::get({kDigitsSlots},
                                 mlir::IntegerType::get(ctx, 32));
  }

  static bool isStorage(mlir::Type type) {
    return type == storage(type.getContext());
  }
};

struct Payload {
  static bool isStorage(mlir::Type type) {
    return Meta::isStorage(type) || Digits::isStorage(type);
  }
};

struct Parts {
  static void storageTypes(mlir::MLIRContext *ctx,
                           llvm::SmallVectorImpl<mlir::Type> &types) {
    types.push_back(Header::owned(ctx));
    types.push_back(Meta::storage(ctx));
    types.push_back(Digits::storage(ctx));
  }

  static bool isHeader(mlir::Type type) { return Header::isOwned(type); }
  static bool isMeta(mlir::Type type) { return Meta::isStorage(type); }
  static bool isDigits(mlir::Type type) { return Digits::isStorage(type); }
  static bool isPayload(mlir::Type type) { return Payload::isStorage(type); }
};

} // namespace long_abi

namespace str_abi {

// Compact Unicode, stripped to the statically needed fields:
// refcount/layout + UTF-8 bytes. The byte payload memref descriptor is the
// authoritative length carrier.
static constexpr int64_t kLayoutId = 4;

struct Bytes {
  static mlir::MemRefType storage(mlir::MLIRContext *ctx) {
    return mlir::MemRefType::get({mlir::ShapedType::kDynamic},
                                 mlir::IntegerType::get(ctx, 8));
  }
};

struct Parts {
  static void storageTypes(mlir::MLIRContext *ctx,
                           llvm::SmallVectorImpl<mlir::Type> &types) {
    types.push_back(Header::owned(ctx));
    types.push_back(Bytes::storage(ctx));
  }

  static bool isHeader(mlir::Type type) { return Header::isOwned(type); }
  static bool isBytes(mlir::Type type) {
    return type == Bytes::storage(type.getContext());
  }
  static bool isPayload(mlir::Type type) { return isBytes(type); }
};

} // namespace str_abi

namespace exception_abi {

static constexpr int64_t kLayoutId = 5;

struct Parts {
  static void storageTypes(mlir::MLIRContext *ctx,
                           llvm::SmallVectorImpl<mlir::Type> &types) {
    types.push_back(Header::owned(ctx));
    types.push_back(Header::owned(ctx));
    types.push_back(str_abi::Bytes::storage(ctx));
  }

  static bool isHeader(mlir::Type type) { return Header::isOwned(type); }
  static bool isMessageHeader(mlir::Type type) { return Header::isOwned(type); }
  static bool isMessageBytes(mlir::Type type) {
    return str_abi::Bytes::storage(type.getContext()) == type;
  }
  static bool isPayload(mlir::Type type) {
    return isMessageHeader(type) || isMessageBytes(type);
  }
};

} // namespace exception_abi

} // namespace py::object_abi
