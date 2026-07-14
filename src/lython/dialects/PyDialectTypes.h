#ifndef LYTHON_PYDIALECTTYPES_H
#define LYTHON_PYDIALECTTYPES_H

#include <optional>
#include <tuple>
#include <utility>

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
class Operation;
} // namespace mlir

namespace py {

namespace attrs {
inline constexpr ::llvm::StringLiteral kPublicContractType{
    "ly.public_contract_type"};
} // namespace attrs

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
  Self,
  IteratorState,
  Overload,
  Contract,
  Literal,
  TypeVar,
  ParamSpec,
  TypeVarTuple,
  Unpack
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

struct TypeListStorage : public mlir::TypeStorage {
  using KeyTy = mlir::ArrayRef<mlir::Type>;

  explicit TypeListStorage(mlir::ArrayRef<mlir::Type> types) : types(types) {}

  bool operator==(const KeyTy &key) const { return key == types; }

  static TypeListStorage *construct(mlir::TypeStorageAllocator &allocator,
                                    const KeyTy &key);

  mlir::ArrayRef<mlir::Type> types;
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

class ContractType : public mlir::Type::TypeBase<ContractType, mlir::Type,
                                                 detail::ProtocolTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name{"py.contract"};

  static ContractType get(mlir::MLIRContext *ctx, ::llvm::StringRef name,
                          mlir::ArrayRef<mlir::Type> arguments = {});
  static bool kindof(unsigned kind) {
    return kind == static_cast<unsigned>(TypeKind::Contract);
  }

  ::llvm::StringRef getContractName() const;
  mlir::ArrayRef<mlir::Type> getArguments() const;
};

class LiteralType : public mlir::Type::TypeBase<LiteralType, mlir::Type,
                                                detail::ClassTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name{"py.literal"};

  static LiteralType get(mlir::MLIRContext *ctx, ::llvm::StringRef spelling);
  static bool kindof(unsigned kind) {
    return kind == static_cast<unsigned>(TypeKind::Literal);
  }

  ::llvm::StringRef getSpelling() const;
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

class TypeVarType : public mlir::Type::TypeBase<TypeVarType, mlir::Type,
                                                detail::ClassTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name{"py.typevar"};

  static TypeVarType get(mlir::MLIRContext *ctx, ::llvm::StringRef name);
  static bool kindof(unsigned kind) {
    return kind == static_cast<unsigned>(TypeKind::TypeVar);
  }

  ::llvm::StringRef getName() const;
};

class ParamSpecType : public mlir::Type::TypeBase<ParamSpecType, mlir::Type,
                                                  detail::ClassTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name{"py.paramspec"};

  static ParamSpecType get(mlir::MLIRContext *ctx, ::llvm::StringRef name);
  static bool kindof(unsigned kind) {
    return kind == static_cast<unsigned>(TypeKind::ParamSpec);
  }

  ::llvm::StringRef getName() const;
};

class TypeVarTupleType
    : public mlir::Type::TypeBase<TypeVarTupleType, mlir::Type,
                                  detail::ClassTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name{"py.typevartuple"};

  static TypeVarTupleType get(mlir::MLIRContext *ctx, ::llvm::StringRef name);
  static bool kindof(unsigned kind) {
    return kind == static_cast<unsigned>(TypeKind::TypeVarTuple);
  }

  ::llvm::StringRef getName() const;
};

class UnpackType : public mlir::Type::TypeBase<UnpackType, mlir::Type,
                                               detail::UnaryTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name{"py.unpack"};

  static UnpackType get(mlir::MLIRContext *ctx, mlir::Type packedType);
  static bool kindof(unsigned kind) {
    return kind == static_cast<unsigned>(TypeKind::Unpack);
  }

  mlir::Type getPackedType() const;
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
                                              detail::TypeListStorage> {
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

class OverloadType : public mlir::Type::TypeBase<OverloadType, mlir::Type,
                                                 detail::TypeListStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name{"py.overload"};

  static OverloadType get(mlir::MLIRContext *ctx,
                          mlir::ArrayRef<mlir::Type> candidateTypes);
  static bool kindof(unsigned kind) {
    return kind == static_cast<unsigned>(TypeKind::Overload);
  }

  mlir::ArrayRef<mlir::Type> getCandidateTypes() const;
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
bool isPyIteratorStateType(mlir::Type type);
mlir::Type primitiveIteratorStateType(mlir::MLIRContext *ctx,
                                      mlir::Type sourceType,
                                      mlir::Type elementType);
bool isPrimitiveIteratorStateType(mlir::Type type);
mlir::Type primitiveIteratorStateSourceType(mlir::Type type);
mlir::Type primitiveIteratorStateElementType(mlir::Type type);
mlir::Type primitiveIteratorSourceElementType(mlir::Type sourceType);
mlir::Type iteratorProtocolType(mlir::MLIRContext *ctx, mlir::Type elementType);
mlir::Type iteratorDescriptorElementType(mlir::Type type);
bool isIteratorDescriptorType(mlir::Type type);
bool isPyClassType(mlir::Type type);
bool isPyTypeType(mlir::Type type);
bool isPyProtocolType(mlir::Type type);
bool isPyContractType(mlir::Type type);
bool isPyLiteralType(mlir::Type type);
bool isPySelfType(mlir::Type type);
bool isPyTypeVarType(mlir::Type type);
bool isPyParamSpecType(mlir::Type type);
bool isPyTypeVarTupleType(mlir::Type type);
bool isPyUnpackType(mlir::Type type);
bool isPyExceptionType(mlir::Type type);
bool isPyExceptionCellType(mlir::Type type);
bool isPyTracebackType(mlir::Type type);
bool isPyLocationType(mlir::Type type);
bool isPyUnionType(mlir::Type type);
bool isPyOverloadType(mlir::Type type);
bool isPyContractNamed(mlir::Type type, llvm::StringRef name);
mlir::Type pyObjectContractType(mlir::MLIRContext *ctx);
mlir::Type pyBoolContractType(mlir::MLIRContext *ctx);
mlir::Type pyIntContractType(mlir::MLIRContext *ctx);
mlir::Type pyFloatContractType(mlir::MLIRContext *ctx);
mlir::Type pyStrContractType(mlir::MLIRContext *ctx);
mlir::Type pyNoneContractType(mlir::MLIRContext *ctx);

// Callable contract checking. A Callable contract owns the complete static
// __call__ shape, including parameter and return metadata.
bool isCallableType(mlir::Type type);
CallableType getCallableContract(mlir::Type type);
bool isCallableEllipsisContract(CallableType signature);

bool isPyType(mlir::Type type);
bool isStaticTypeParameter(mlir::Type type);
mlir::Type eraseStaticTypeParameters(mlir::Type type);

// Structural map over the py type tree. `transform` runs on every node
// first: an engaged non-null result replaces the node without recursing into
// it, an engaged null result aborts the whole map ({}), and nullopt recurses
// into container arguments (Contract / Protocol / Union / Callable /
// Overload / Unpack / TypeObject) and rebuilds only when a child changed.
// Every bespoke recursive walk over these containers must be a callback on
// this mapper — six hand-rolled copies drifted apart (one skipped
// OverloadType entirely) before it existed.
mlir::Type mapPyTypeStructure(
    mlir::Type type,
    llvm::function_ref<std::optional<mlir::Type>(mlir::Type)> transform);

// Rebuilds a Callable with each child type passed through `mapChild`
// (12-argument reconstruction with null propagation); the shared shape
// under mapPyTypeStructure and the protocol-table signature substitution.
CallableType rebuildCallableWith(
    CallableType callable,
    llvm::function_ref<mlir::Type(mlir::Type)> mapChild);

// Protocol descriptor helpers. Manifest protocol instantiations can be queried
// through their base graph without adding a dedicated dialect type for each
// high-level typing contract.
std::optional<llvm::SmallVector<mlir::Type, 3>>
protocolDescriptorArguments(mlir::Type type, llvm::StringRef protocolName);
mlir::Type unaryProtocolDescriptorPayloadType(mlir::Type type,
                                              llvm::StringRef protocolName);

// Awaitable protocol helpers. Manifest protocol instantiations whose base graph
// reaches Awaitable[T] use the low-level descriptor ABI. Native async.value<T>
// remains a single MLIR async value and is handled explicitly by await users.
mlir::Type awaitableProtocolType(mlir::MLIRContext *ctx,
                                 mlir::Type payloadType);
mlir::Type awaitableDescriptorPayloadType(mlir::Type type);
bool isAwaitableDescriptorType(mlir::Type type);

// Subtype checking (v2.1)
struct SubtypeBindings {
  mlir::Type self;
};

bool isSubtypeOf(mlir::Type subtype, mlir::Type supertype);
bool isSubtypeOf(mlir::Type subtype, mlir::Type supertype,
                 mlir::Operation *from);
bool isSubtypeOf(mlir::Type subtype, mlir::Type supertype,
                 mlir::Operation *from, SubtypeBindings *bindings);

// Assignment compatibility at the dialect boundary. This keeps the pure
// subtype relation separate from ABI conveniences such as builtin MLIR integer
// and float values flowing into their Python primitive contracts.
bool isAssignableTo(mlir::Type actual, mlir::Type expected);
bool isAssignableTo(mlir::Type actual, mlir::Type expected,
                    mlir::Operation *from);
bool isAssignableTo(mlir::Type actual, mlir::Type expected,
                    mlir::Operation *from, SubtypeBindings *bindings);

} // namespace py

#endif // LYTHON_PYDIALECTTYPES_H
