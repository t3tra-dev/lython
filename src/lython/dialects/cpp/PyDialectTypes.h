#ifndef LYTHON_PYDIALECTTYPES_H
#define LYTHON_PYDIALECTTYPES_H

#include <tuple>
#include <utility>

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/StringRef.h"

namespace py {

enum class TypeKind : unsigned {
  Int = 0,
  Float,
  Bool,
  Str,
  Object,
  None,
  Tuple,
  Dict,
  Class, // User-defined class type
  FuncSig,
  Func,
  PrimFunc
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

struct ClassTypeStorage : public mlir::TypeStorage {
  using KeyTy = ::llvm::StringRef;

  explicit ClassTypeStorage(::llvm::StringRef name) : className(name) {}

  bool operator==(const KeyTy &key) const { return key == className; }

  static ClassTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                     const KeyTy &key);

  ::llvm::StringRef className;
};

struct FuncSignatureStorage : public mlir::TypeStorage {
  // Key order: positional, vararg, kwonly, kwargs, results
  using KeyTy = std::tuple<mlir::ArrayRef<mlir::Type>, mlir::Type,
                           mlir::ArrayRef<mlir::Type>, mlir::Type,
                           mlir::ArrayRef<mlir::Type>>;

  FuncSignatureStorage(mlir::ArrayRef<mlir::Type> positional, mlir::Type vararg,
                       mlir::ArrayRef<mlir::Type> kwonly, mlir::Type kwargs,
                       mlir::ArrayRef<mlir::Type> results)
      : positionalTypes(positional), varargType(vararg), kwOnlyTypes(kwonly),
        kwargsType(kwargs), resultTypes(results) {}

  bool operator==(const KeyTy &key) const {
    return std::get<0>(key) == positionalTypes &&
           std::get<1>(key) == varargType && std::get<2>(key) == kwOnlyTypes &&
           std::get<3>(key) == kwargsType && std::get<4>(key) == resultTypes;
  }

  static FuncSignatureStorage *construct(mlir::TypeStorageAllocator &allocator,
                                         const KeyTy &key);

  mlir::ArrayRef<mlir::Type> positionalTypes;
  mlir::Type varargType;
  mlir::ArrayRef<mlir::Type> kwOnlyTypes;
  mlir::Type kwargsType;
  mlir::ArrayRef<mlir::Type> resultTypes;
};

struct FuncTypeStorage : public mlir::TypeStorage {
  using KeyTy = mlir::Type;

  explicit FuncTypeStorage(mlir::Type signature) : signature(signature) {}

  bool operator==(const KeyTy &key) const { return key == signature; }

  static FuncTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                    const KeyTy &key);

  mlir::Type signature;
};

struct FunctionTypeStorage : public mlir::TypeStorage {
  using KeyTy = mlir::FunctionType;

  explicit FunctionTypeStorage(mlir::FunctionType signature)
      : signature(signature) {}

  bool operator==(const KeyTy &key) const { return key == signature; }

  static FunctionTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                        const KeyTy &key);

  mlir::FunctionType signature;
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

class FuncSignatureType
    : public mlir::Type::TypeBase<FuncSignatureType, mlir::Type,
                                  detail::FuncSignatureStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name{"py.funcsig"};

  static FuncSignatureType get(mlir::MLIRContext *ctx,
                               mlir::ArrayRef<mlir::Type> positional,
                               mlir::ArrayRef<mlir::Type> kwonly,
                               mlir::Type varargType, mlir::Type kwargsType,
                               mlir::ArrayRef<mlir::Type> results);
  static bool kindof(unsigned kind) {
    return kind == static_cast<unsigned>(TypeKind::FuncSig);
  }

  mlir::ArrayRef<mlir::Type> getPositionalTypes() const;
  mlir::ArrayRef<mlir::Type> getKwOnlyTypes() const;
  mlir::ArrayRef<mlir::Type> getResultTypes() const;
  bool hasVararg() const;
  mlir::Type getVarargType() const;
  bool hasKwarg() const;
  mlir::Type getKwargType() const;
};

class FuncType : public mlir::Type::TypeBase<FuncType, mlir::Type,
                                             detail::FuncTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name{"py.func"};

  static FuncType get(mlir::MLIRContext *ctx, FuncSignatureType signature);
  static bool kindof(unsigned kind) {
    return kind == static_cast<unsigned>(TypeKind::Func);
  }

  FuncSignatureType getSignature() const;
};

class PrimFuncType : public mlir::Type::TypeBase<PrimFuncType, mlir::Type,
                                                 detail::FunctionTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name{"py.prim.func"};

  static PrimFuncType get(mlir::MLIRContext *ctx, mlir::FunctionType signature);
  static bool kindof(unsigned kind) {
    return kind == static_cast<unsigned>(TypeKind::PrimFunc);
  }

  mlir::FunctionType getSignature() const;
};

bool isPyIntType(mlir::Type type);
bool isPyFloatType(mlir::Type type);
bool isPyBoolType(mlir::Type type);
bool isPyStrType(mlir::Type type);
bool isPyObjectType(mlir::Type type);
bool isPyNoneType(mlir::Type type);
bool isPyTupleType(mlir::Type type);
bool isPyDictType(mlir::Type type);
bool isPyClassType(mlir::Type type);
bool isPyFuncSigType(mlir::Type type);
bool isPyFuncType(mlir::Type type);
bool isPyPrimFuncType(mlir::Type type);

// v2: Callable type checking (!py.func or !py.class with __call__)
bool isCallableType(mlir::Type type);

bool isPyType(mlir::Type type);

// Subtype checking (v2.1)
bool isSubtypeOf(mlir::Type subtype, mlir::Type supertype);

} // namespace py

#endif // LYTHON_PYDIALECTTYPES_H
