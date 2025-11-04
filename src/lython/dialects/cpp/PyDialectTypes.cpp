#include "PyDialectTypes.h"

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

namespace py {

namespace detail {

SimpleTypeStorage *SimpleTypeStorage::construct(TypeStorageAllocator &allocator,
                                                const KeyTy &key) {
  return new (allocator.allocate<SimpleTypeStorage>())
      SimpleTypeStorage(static_cast<unsigned>(key));
}

TupleTypeStorage *TupleTypeStorage::construct(TypeStorageAllocator &allocator,
                                              const KeyTy &key) {
  ArrayRef<Type> copied = allocator.copyInto(key);
  return new (allocator.allocate<TupleTypeStorage>()) TupleTypeStorage(copied);
}

DictTypeStorage *DictTypeStorage::construct(TypeStorageAllocator &allocator,
                                            const KeyTy &key) {
  return new (allocator.allocate<DictTypeStorage>())
      DictTypeStorage(key.first, key.second);
}

ClassTypeStorage *ClassTypeStorage::construct(TypeStorageAllocator &allocator,
                                              const KeyTy &key) {
  return new (allocator.allocate<ClassTypeStorage>())
      ClassTypeStorage(allocator.copyInto(key));
}

FuncSignatureStorage *
FuncSignatureStorage::construct(TypeStorageAllocator &allocator,
                                const KeyTy &key) {
  // Key order: positional, vararg, kwonly, kwargs, results
  ArrayRef<Type> positional = allocator.copyInto(std::get<0>(key));
  Type vararg = std::get<1>(key);
  ArrayRef<Type> kwonly = allocator.copyInto(std::get<2>(key));
  Type kwargs = std::get<3>(key);
  ArrayRef<Type> results = allocator.copyInto(std::get<4>(key));
  return new (allocator.allocate<FuncSignatureStorage>())
      FuncSignatureStorage(positional, vararg, kwonly, kwargs, results);
}

FuncTypeStorage *FuncTypeStorage::construct(TypeStorageAllocator &allocator,
                                            const KeyTy &key) {
  return new (allocator.allocate<FuncTypeStorage>()) FuncTypeStorage(key);
}

FunctionTypeStorage *
FunctionTypeStorage::construct(TypeStorageAllocator &allocator,
                               const KeyTy &key) {
  return new (allocator.allocate<FunctionTypeStorage>())
      FunctionTypeStorage(key);
}

} // namespace detail

//===----------------------------------------------------------------------===//
// Simple types
//===----------------------------------------------------------------------===//

IntType IntType::get(MLIRContext *ctx) { return Base::get(ctx, TypeKind::Int); }

FloatType FloatType::get(MLIRContext *ctx) {
  return Base::get(ctx, TypeKind::Float);
}

BoolType BoolType::get(MLIRContext *ctx) {
  return Base::get(ctx, TypeKind::Bool);
}

StrType StrType::get(MLIRContext *ctx) { return Base::get(ctx, TypeKind::Str); }

ObjectType ObjectType::get(MLIRContext *ctx) {
  return Base::get(ctx, TypeKind::Object);
}

NoneType NoneType::get(MLIRContext *ctx) {
  return Base::get(ctx, TypeKind::None);
}

//===----------------------------------------------------------------------===//
// Composite types
//===----------------------------------------------------------------------===//

TupleType TupleType::get(MLIRContext *ctx, ArrayRef<Type> elementTypes) {
  return Base::get(ctx, elementTypes);
}

ArrayRef<Type> TupleType::getElementTypes() const {
  return getImpl()->elementTypes;
}

DictType DictType::get(MLIRContext *ctx, Type keyType, Type valueType) {
  return Base::get(ctx, std::make_pair(keyType, valueType));
}

Type DictType::getKeyType() const { return getImpl()->keyType; }

Type DictType::getValueType() const { return getImpl()->valueType; }

ClassType ClassType::get(MLIRContext *ctx, ::llvm::StringRef className) {
  return Base::get(ctx, className);
}

::llvm::StringRef ClassType::getClassName() const {
  return getImpl()->className;
}

FuncSignatureType FuncSignatureType::get(MLIRContext *ctx,
                                         ArrayRef<Type> positional,
                                         ArrayRef<Type> kwonly, Type varargType,
                                         Type kwargsType,
                                         ArrayRef<Type> results) {
  // Key order: positional, vararg, kwonly, kwargs, results
  return Base::get(ctx, std::make_tuple(positional, varargType, kwonly,
                                        kwargsType, results));
}

ArrayRef<Type> FuncSignatureType::getPositionalTypes() const {
  return getImpl()->positionalTypes;
}

ArrayRef<Type> FuncSignatureType::getKwOnlyTypes() const {
  return getImpl()->kwOnlyTypes;
}

ArrayRef<Type> FuncSignatureType::getResultTypes() const {
  return getImpl()->resultTypes;
}

bool FuncSignatureType::hasVararg() const {
  return static_cast<bool>(getImpl()->varargType);
}

Type FuncSignatureType::getVarargType() const { return getImpl()->varargType; }

bool FuncSignatureType::hasKwarg() const {
  return static_cast<bool>(getImpl()->kwargsType);
}

Type FuncSignatureType::getKwargType() const { return getImpl()->kwargsType; }

FuncType FuncType::get(MLIRContext *ctx, FuncSignatureType signature) {
  return Base::get(ctx, signature);
}

FuncSignatureType FuncType::getSignature() const {
  return mlir::cast<FuncSignatureType>(getImpl()->signature);
}

PrimFuncType PrimFuncType::get(MLIRContext *ctx, FunctionType signature) {
  return Base::get(ctx, signature);
}

FunctionType PrimFuncType::getSignature() const { return getImpl()->signature; }

//===----------------------------------------------------------------------===//
// Helper predicates
//===----------------------------------------------------------------------===//

bool isPyIntType(Type type) { return mlir::isa<IntType>(type); }

bool isPyFloatType(Type type) { return mlir::isa<FloatType>(type); }

bool isPyBoolType(Type type) { return mlir::isa<BoolType>(type); }

bool isPyStrType(Type type) { return mlir::isa<StrType>(type); }

bool isPyObjectType(Type type) { return mlir::isa<ObjectType>(type); }

bool isPyNoneType(Type type) { return mlir::isa<NoneType>(type); }

bool isPyTupleType(Type type) { return mlir::isa<TupleType>(type); }

bool isPyDictType(Type type) { return mlir::isa<DictType>(type); }

bool isPyClassType(Type type) { return mlir::isa<ClassType>(type); }

bool isPyFuncSigType(Type type) { return mlir::isa<FuncSignatureType>(type); }

bool isPyFuncType(Type type) { return mlir::isa<FuncType>(type); }

bool isPyPrimFuncType(Type type) { return mlir::isa<PrimFuncType>(type); }

bool isCallableType(Type type) {
  // v2: Callable = !py.func or !py.class (with __call__ method)
  // TODO(v2): Add runtime check for __call__ method in !py.class
  // For now, accept all !py.func and !py.class
  return mlir::isa<FuncType>(type) || mlir::isa<ClassType>(type);
}

bool isPyType(Type type) {
  return llvm::TypeSwitch<Type, bool>(type)
      .Case<IntType, FloatType, BoolType, StrType, ObjectType, NoneType,
            TupleType, DictType, ClassType, FuncType, PrimFuncType>(
          [](auto) { return true; })
      .Default([](Type) { return false; });
}

//===----------------------------------------------------------------------===//
// Subtype checking (v2.1)
//===----------------------------------------------------------------------===//

bool isSubtypeOf(Type subtype, Type supertype) {
  // Reflexive: T <: T
  if (subtype == supertype)
    return true;

  // Top type: T <: !py.object for all T
  if (mlir::isa<ObjectType>(supertype))
    return isPyType(subtype);

  // Tuple covariance: !py.tuple<S> <: !py.tuple<T> if S <: T
  auto subtypeTuple = mlir::dyn_cast<TupleType>(subtype);
  auto supertypeTuple = mlir::dyn_cast<TupleType>(supertype);
  if (subtypeTuple && supertypeTuple) {
    auto subElems = subtypeTuple.getElementTypes();
    auto superElems = supertypeTuple.getElementTypes();
    if (subElems.size() != superElems.size())
      return false;
    for (auto [sub, sup] : llvm::zip(subElems, superElems)) {
      if (!isSubtypeOf(sub, sup))
        return false;
    }
    return true;
  }

  // Dict covariance: !py.dict<K1,V1> <: !py.dict<K2,V2> if K1<:K2 and V1<:V2
  auto subtypeDict = mlir::dyn_cast<DictType>(subtype);
  auto supertypeDict = mlir::dyn_cast<DictType>(supertype);
  if (subtypeDict && supertypeDict) {
    return isSubtypeOf(subtypeDict.getKeyType(), supertypeDict.getKeyType()) &&
           isSubtypeOf(subtypeDict.getValueType(),
                       supertypeDict.getValueType());
  }

  // No other subtype relations in v2.1
  // TODO(v3): Add class hierarchy support
  return false;
}

} // namespace py
