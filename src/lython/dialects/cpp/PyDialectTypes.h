#ifndef LYTHON_PYDIALECTTYPES_H
#define LYTHON_PYDIALECTTYPES_H

#include <tuple>
#include <utility>

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
class Operation;
} // namespace mlir

namespace py {

enum class TypeKind : unsigned {
  Int = 0,
  Float,
  Bool,
  Str,
  None,
  Tuple,
  Dict,
  List,
  Class, // User-defined class type
  Exception,
  ExceptionCell,
  Traceback,
  Location,
  Callable,
  Union,
  Object,
  Type,
  Protocol,
  Self
};

namespace detail {

struct SimpleTypeStorage : public mlir::TypeStorage {
  using KeyTy = TypeKind;

  explicit SimpleTypeStorage(unsigned kind) : kind(kind) {}

  bool operator==(const KeyTy &key) const {
    return kind == static_cast<unsigned>(key);
  }

  static SimpleTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                      const KeyTy &key);

  unsigned kind;
};

struct TupleTypeStorage : public mlir::TypeStorage {
  using KeyTy = mlir::ArrayRef<mlir::Type>;

  explicit TupleTypeStorage(mlir::ArrayRef<mlir::Type> elements)
      : elementTypes(elements) {}

  bool operator==(const KeyTy &key) const { return key == elementTypes; }

  static TupleTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                     const KeyTy &key);

  mlir::ArrayRef<mlir::Type> elementTypes;
};

struct DictTypeStorage : public mlir::TypeStorage {
  using KeyTy = std::pair<mlir::Type, mlir::Type>;

  DictTypeStorage(mlir::Type key, mlir::Type value)
      : keyType(key), valueType(value) {}

  bool operator==(const KeyTy &key) const {
    return key.first == keyType && key.second == valueType;
  }

  static DictTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                    const KeyTy &key);

  mlir::Type keyType;
  mlir::Type valueType;
};

struct ListTypeStorage : public mlir::TypeStorage {
  using KeyTy = mlir::Type;

  explicit ListTypeStorage(mlir::Type element) : elementType(element) {}

  bool operator==(const KeyTy &key) const { return key == elementType; }

  static ListTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                    const KeyTy &key);

  mlir::Type elementType;
};

struct ClassTypeStorage : public mlir::TypeStorage {
  using KeyTy = ::llvm::StringRef;

  explicit ClassTypeStorage(::llvm::StringRef name) : className(name) {}

  bool operator==(const KeyTy &key) const { return key == className; }

  static ClassTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                     const KeyTy &key);

  ::llvm::StringRef className;
};

struct CallableTypeStorage : public mlir::TypeStorage {
  // Key order: positional, vararg, kwonly, kwargs, results, positional-only
  // count, positional names, kwonly names, positional default flags, kwonly
  // default flags, vararg name, kwargs name.
  using KeyTy = std::tuple<
      mlir::ArrayRef<mlir::Type>, mlir::Type, mlir::ArrayRef<mlir::Type>,
      mlir::Type, mlir::ArrayRef<mlir::Type>, unsigned,
      mlir::ArrayRef<mlir::StringAttr>, mlir::ArrayRef<mlir::StringAttr>,
      mlir::ArrayRef<mlir::BoolAttr>, mlir::ArrayRef<mlir::BoolAttr>,
      mlir::StringAttr, mlir::StringAttr>;

  CallableTypeStorage(mlir::ArrayRef<mlir::Type> positional, mlir::Type vararg,
                      mlir::ArrayRef<mlir::Type> kwonly, mlir::Type kwargs,
                      mlir::ArrayRef<mlir::Type> results,
                      unsigned positionalOnlyCount,
                      mlir::ArrayRef<mlir::StringAttr> positionalNames,
                      mlir::ArrayRef<mlir::StringAttr> kwOnlyNames,
                      mlir::ArrayRef<mlir::BoolAttr> positionalDefaults,
                      mlir::ArrayRef<mlir::BoolAttr> kwOnlyDefaults,
                      mlir::StringAttr varargName, mlir::StringAttr kwargsName)
      : positionalTypes(positional), varargType(vararg), kwOnlyTypes(kwonly),
        kwargsType(kwargs), resultTypes(results),
        positionalOnlyCount(positionalOnlyCount),
        positionalNames(positionalNames), kwOnlyNames(kwOnlyNames),
        positionalDefaults(positionalDefaults), kwOnlyDefaults(kwOnlyDefaults),
        varargName(varargName), kwargsName(kwargsName) {}

  bool operator==(const KeyTy &key) const {
    return std::get<0>(key) == positionalTypes &&
           std::get<1>(key) == varargType && std::get<2>(key) == kwOnlyTypes &&
           std::get<3>(key) == kwargsType && std::get<4>(key) == resultTypes &&
           std::get<5>(key) == positionalOnlyCount &&
           std::get<6>(key) == positionalNames &&
           std::get<7>(key) == kwOnlyNames &&
           std::get<8>(key) == positionalDefaults &&
           std::get<9>(key) == kwOnlyDefaults &&
           std::get<10>(key) == varargName && std::get<11>(key) == kwargsName;
  }

  static CallableTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                        const KeyTy &key);

  mlir::ArrayRef<mlir::Type> positionalTypes;
  mlir::Type varargType;
  mlir::ArrayRef<mlir::Type> kwOnlyTypes;
  mlir::Type kwargsType;
  mlir::ArrayRef<mlir::Type> resultTypes;
  unsigned positionalOnlyCount;
  mlir::ArrayRef<mlir::StringAttr> positionalNames;
  mlir::ArrayRef<mlir::StringAttr> kwOnlyNames;
  mlir::ArrayRef<mlir::BoolAttr> positionalDefaults;
  mlir::ArrayRef<mlir::BoolAttr> kwOnlyDefaults;
  mlir::StringAttr varargName;
  mlir::StringAttr kwargsName;
};

struct UnaryTypeStorage : public mlir::TypeStorage {
  using KeyTy = mlir::Type;

  explicit UnaryTypeStorage(mlir::Type value) : valueType(value) {}

  bool operator==(const KeyTy &key) const { return key == valueType; }

  static UnaryTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                     const KeyTy &key);

  mlir::Type valueType;
};

struct UnionTypeStorage : public mlir::TypeStorage {
  using KeyTy = mlir::ArrayRef<mlir::Type>;

  explicit UnionTypeStorage(mlir::ArrayRef<mlir::Type> members)
      : memberTypes(members) {}

  bool operator==(const KeyTy &key) const { return key == memberTypes; }

  static UnionTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                     const KeyTy &key);

  mlir::ArrayRef<mlir::Type> memberTypes;
};

struct ProtocolTypeStorage : public mlir::TypeStorage {
  using KeyTy = std::pair<::llvm::StringRef, mlir::ArrayRef<mlir::Type>>;

  ProtocolTypeStorage(::llvm::StringRef name, mlir::ArrayRef<mlir::Type> args)
      : protocolName(name), arguments(args) {}

  bool operator==(const KeyTy &key) const {
    return key.first == protocolName && key.second == arguments;
  }

  static ProtocolTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                        const KeyTy &key);

  ::llvm::StringRef protocolName;
  mlir::ArrayRef<mlir::Type> arguments;
};

} // namespace detail

class IntType : public mlir::Type::TypeBase<IntType, mlir::Type,
                                            detail::SimpleTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name{"py.int"};

  static IntType get(mlir::MLIRContext *ctx);
  static bool kindof(unsigned kind) {
    return kind == static_cast<unsigned>(TypeKind::Int);
  }
};

class FloatType : public mlir::Type::TypeBase<FloatType, mlir::Type,
                                              detail::SimpleTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name{"py.float"};

  static FloatType get(mlir::MLIRContext *ctx);
  static bool kindof(unsigned kind) {
    return kind == static_cast<unsigned>(TypeKind::Float);
  }
};

class BoolType : public mlir::Type::TypeBase<BoolType, mlir::Type,
                                             detail::SimpleTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name{"py.bool"};

  static BoolType get(mlir::MLIRContext *ctx);
  static bool kindof(unsigned kind) {
    return kind == static_cast<unsigned>(TypeKind::Bool);
  }
};

class StrType : public mlir::Type::TypeBase<StrType, mlir::Type,
                                            detail::SimpleTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name{"py.str"};

  static StrType get(mlir::MLIRContext *ctx);
  static bool kindof(unsigned kind) {
    return kind == static_cast<unsigned>(TypeKind::Str);
  }
};

class NoneType : public mlir::Type::TypeBase<NoneType, mlir::Type,
                                             detail::SimpleTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name{"py.none"};

  static NoneType get(mlir::MLIRContext *ctx);
  static bool kindof(unsigned kind) {
    return kind == static_cast<unsigned>(TypeKind::None);
  }
};

class ObjectType : public mlir::Type::TypeBase<ObjectType, mlir::Type,
                                               detail::SimpleTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name{"py.object"};

  static ObjectType get(mlir::MLIRContext *ctx);
  static bool kindof(unsigned kind) {
    return kind == static_cast<unsigned>(TypeKind::Object);
  }
};

class TupleType : public mlir::Type::TypeBase<TupleType, mlir::Type,
                                              detail::TupleTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name{"py.tuple"};

  static TupleType get(mlir::MLIRContext *ctx,
                       mlir::ArrayRef<mlir::Type> elementTypes);
  static bool kindof(unsigned kind) {
    return kind == static_cast<unsigned>(TypeKind::Tuple);
  }

  mlir::ArrayRef<mlir::Type> getElementTypes() const;
};

class DictType : public mlir::Type::TypeBase<DictType, mlir::Type,
                                             detail::DictTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name{"py.dict"};

  static DictType get(mlir::MLIRContext *ctx, mlir::Type keyType,
                      mlir::Type valueType);
  static bool kindof(unsigned kind) {
    return kind == static_cast<unsigned>(TypeKind::Dict);
  }

  mlir::Type getKeyType() const;
  mlir::Type getValueType() const;
};

class ClassType : public mlir::Type::TypeBase<ClassType, mlir::Type,
                                              detail::ClassTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name{"py.class"};

  static ClassType get(mlir::MLIRContext *ctx, ::llvm::StringRef className);
  static bool kindof(unsigned kind) {
    return kind == static_cast<unsigned>(TypeKind::Class);
  }

  ::llvm::StringRef getClassName() const;
};

class TypeType : public mlir::Type::TypeBase<TypeType, mlir::Type,
                                             detail::UnaryTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name{"py.type"};

  static TypeType get(mlir::MLIRContext *ctx, mlir::Type instanceType);
  static bool kindof(unsigned kind) {
    return kind == static_cast<unsigned>(TypeKind::Type);
  }

  mlir::Type getInstanceType() const;
};

class ProtocolType : public mlir::Type::TypeBase<ProtocolType, mlir::Type,
                                                 detail::ProtocolTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name{"py.protocol"};

  static ProtocolType get(mlir::MLIRContext *ctx,
                          ::llvm::StringRef protocolName,
                          mlir::ArrayRef<mlir::Type> arguments);
  static bool kindof(unsigned kind) {
    return kind == static_cast<unsigned>(TypeKind::Protocol);
  }

  ::llvm::StringRef getProtocolName() const;
  mlir::ArrayRef<mlir::Type> getArguments() const;
};

class SelfType : public mlir::Type::TypeBase<SelfType, mlir::Type,
                                             detail::SimpleTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name{"py.self"};

  static SelfType get(mlir::MLIRContext *ctx);
  static bool kindof(unsigned kind) {
    return kind == static_cast<unsigned>(TypeKind::Self);
  }
};

class ListType : public mlir::Type::TypeBase<ListType, mlir::Type,
                                             detail::ListTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name{"py.list"};

  static ListType get(mlir::MLIRContext *ctx, mlir::Type elementType);
  static bool kindof(unsigned kind) {
    return kind == static_cast<unsigned>(TypeKind::List);
  }

  mlir::Type getElementType() const;
};

class ExceptionType : public mlir::Type::TypeBase<ExceptionType, mlir::Type,
                                                  detail::SimpleTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name{"py.exception"};

  static ExceptionType get(mlir::MLIRContext *ctx);
  static bool kindof(unsigned kind) {
    return kind == static_cast<unsigned>(TypeKind::Exception);
  }
};

class ExceptionCellType
    : public mlir::Type::TypeBase<ExceptionCellType, mlir::Type,
                                  detail::SimpleTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name{"py.exception_cell"};

  static ExceptionCellType get(mlir::MLIRContext *ctx);
  static bool kindof(unsigned kind) {
    return kind == static_cast<unsigned>(TypeKind::ExceptionCell);
  }
};

class TracebackType : public mlir::Type::TypeBase<TracebackType, mlir::Type,
                                                  detail::SimpleTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name{"py.traceback"};

  static TracebackType get(mlir::MLIRContext *ctx);
  static bool kindof(unsigned kind) {
    return kind == static_cast<unsigned>(TypeKind::Traceback);
  }
};

class LocationType : public mlir::Type::TypeBase<LocationType, mlir::Type,
                                                 detail::SimpleTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name{"py.location"};

  static LocationType get(mlir::MLIRContext *ctx);
  static bool kindof(unsigned kind) {
    return kind == static_cast<unsigned>(TypeKind::Location);
  }
};

class CallableType : public mlir::Type::TypeBase<CallableType, mlir::Type,
                                                 detail::CallableTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name{"py.callable"};

  static CallableType get(mlir::MLIRContext *ctx,
                          mlir::ArrayRef<mlir::Type> positional,
                          mlir::ArrayRef<mlir::Type> kwonly,
                          mlir::Type varargType, mlir::Type kwargsType,
                          mlir::ArrayRef<mlir::Type> results);
  static CallableType
  get(mlir::MLIRContext *ctx, mlir::ArrayRef<mlir::Type> positional,
      mlir::ArrayRef<mlir::Type> kwonly, mlir::Type varargType,
      mlir::Type kwargsType, mlir::ArrayRef<mlir::Type> results,
      mlir::ArrayRef<mlir::StringAttr> positionalNames,
      mlir::ArrayRef<mlir::StringAttr> kwOnlyNames,
      mlir::ArrayRef<mlir::BoolAttr> positionalDefaults,
      mlir::ArrayRef<mlir::BoolAttr> kwOnlyDefaults,
      mlir::StringAttr varargName = {}, mlir::StringAttr kwargsName = {},
      unsigned positionalOnlyCount = 0);
  static bool kindof(unsigned kind) {
    return kind == static_cast<unsigned>(TypeKind::Callable);
  }

  mlir::ArrayRef<mlir::Type> getPositionalTypes() const;
  mlir::ArrayRef<mlir::Type> getKwOnlyTypes() const;
  mlir::ArrayRef<mlir::Type> getResultTypes() const;
  bool hasVararg() const;
  mlir::Type getVarargType() const;
  bool hasKwarg() const;
  mlir::Type getKwargType() const;
  mlir::ArrayRef<mlir::StringAttr> getPositionalNames() const;
  mlir::ArrayRef<mlir::StringAttr> getKwOnlyNames() const;
  mlir::ArrayRef<mlir::BoolAttr> getPositionalDefaults() const;
  mlir::ArrayRef<mlir::BoolAttr> getKwOnlyDefaults() const;
  mlir::StringAttr getVarargName() const;
  mlir::StringAttr getKwargName() const;
  unsigned getPositionalOnlyCount() const;
  bool hasParameterMetadata() const;
};

class UnionType : public mlir::Type::TypeBase<UnionType, mlir::Type,
                                              detail::UnionTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name{"py.union"};

  // Returns the normalized union of `memberTypes`: nested unions are
  // flattened, duplicates removed, and members sorted by a deterministic
  // canonical key. A single distinct member collapses to that member (the
  // result is then not a UnionType). Zero members returns a null type.
  static mlir::Type getNormalized(mlir::MLIRContext *ctx,
                                  mlir::ArrayRef<mlir::Type> memberTypes);
  static bool kindof(unsigned kind) {
    return kind == static_cast<unsigned>(TypeKind::Union);
  }

  mlir::ArrayRef<mlir::Type> getMemberTypes() const;
  bool hasMember(mlir::Type member) const;

  // Optional shape: exactly two members, one of which is !py.none.
  bool isOptional() const;
  // The non-None member of an Optional-shaped union; null otherwise.
  mlir::Type getOptionalPayloadType() const;
};

bool isPyIntType(mlir::Type type);
bool isPyFloatType(mlir::Type type);
bool isPyBoolType(mlir::Type type);
bool isPyStrType(mlir::Type type);
bool isPyNoneType(mlir::Type type);
bool isPyObjectType(mlir::Type type);
bool isPyTupleType(mlir::Type type);
bool isPyDictType(mlir::Type type);
bool isPyListType(mlir::Type type);
bool isPyClassType(mlir::Type type);
bool isPyTypeType(mlir::Type type);
bool isPyProtocolType(mlir::Type type);
bool isPySelfType(mlir::Type type);
bool isPyExceptionType(mlir::Type type);
bool isPyExceptionCellType(mlir::Type type);
bool isPyTracebackType(mlir::Type type);
bool isPyLocationType(mlir::Type type);
bool isPyUnionType(mlir::Type type);

// Callable contract checking. A Callable contract owns the complete static
// __call__ shape, including parameter and return metadata.
bool isCallableType(mlir::Type type);
CallableType getCallableContract(mlir::Type type);

bool isPyType(mlir::Type type);

// Awaitable contract helpers. These cover the dialect-level awaitable
// primitives and the manifest protocol spellings whose payload is encoded in
// the type arguments.
bool isCoroutineProtocolType(mlir::Type type);
mlir::Type awaitablePayloadType(mlir::Type type);

// Subtype checking (v2.1)
bool isSubtypeOf(mlir::Type subtype, mlir::Type supertype);
bool isSubtypeOf(mlir::Type subtype, mlir::Type supertype,
                 mlir::Operation *from);

} // namespace py

#endif // LYTHON_PYDIALECTTYPES_H
