#include "PyDialectTypes.h"

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/TypeSwitch.h"

namespace py {

namespace detail {

SimpleTypeStorage *
SimpleTypeStorage::construct(mlir::TypeStorageAllocator &allocator,
                             const KeyTy &key) {
  return new (allocator.allocate<SimpleTypeStorage>())
      SimpleTypeStorage(static_cast<unsigned>(key));
}

TupleTypeStorage *
TupleTypeStorage::construct(mlir::TypeStorageAllocator &allocator,
                            const KeyTy &key) {
  llvm::ArrayRef<mlir::Type> copied = allocator.copyInto(key);
  return new (allocator.allocate<TupleTypeStorage>()) TupleTypeStorage(copied);
}

DictTypeStorage *
DictTypeStorage::construct(mlir::TypeStorageAllocator &allocator,
                           const KeyTy &key) {
  return new (allocator.allocate<DictTypeStorage>())
      DictTypeStorage(key.first, key.second);
}

ListTypeStorage *
ListTypeStorage::construct(mlir::TypeStorageAllocator &allocator,
                           const KeyTy &key) {
  return new (allocator.allocate<ListTypeStorage>()) ListTypeStorage(key);
}

ClassTypeStorage *
ClassTypeStorage::construct(mlir::TypeStorageAllocator &allocator,
                            const KeyTy &key) {
  return new (allocator.allocate<ClassTypeStorage>())
      ClassTypeStorage(allocator.copyInto(key));
}

FuncSignatureStorage *
FuncSignatureStorage::construct(mlir::TypeStorageAllocator &allocator,
                                const KeyTy &key) {
  // Key order: positional, vararg, kwonly, kwargs, results
  llvm::ArrayRef<mlir::Type> positional = allocator.copyInto(std::get<0>(key));
  mlir::Type vararg = std::get<1>(key);
  llvm::ArrayRef<mlir::Type> kwonly = allocator.copyInto(std::get<2>(key));
  mlir::Type kwargs = std::get<3>(key);
  llvm::ArrayRef<mlir::Type> results = allocator.copyInto(std::get<4>(key));
  return new (allocator.allocate<FuncSignatureStorage>())
      FuncSignatureStorage(positional, vararg, kwonly, kwargs, results);
}

FuncTypeStorage *
FuncTypeStorage::construct(mlir::TypeStorageAllocator &allocator,
                           const KeyTy &key) {
  return new (allocator.allocate<FuncTypeStorage>()) FuncTypeStorage(key);
}

FunctionTypeStorage *
FunctionTypeStorage::construct(mlir::TypeStorageAllocator &allocator,
                               const KeyTy &key) {
  return new (allocator.allocate<FunctionTypeStorage>())
      FunctionTypeStorage(key);
}

AwaitableTypeStorage *
AwaitableTypeStorage::construct(mlir::TypeStorageAllocator &allocator,
                                const KeyTy &key) {
  return new (allocator.allocate<AwaitableTypeStorage>())
      AwaitableTypeStorage(key);
}

} // namespace detail

//===----------------------------------------------------------------------===//
// Simple types
//===----------------------------------------------------------------------===//

IntType IntType::get(mlir::MLIRContext *ctx) {
  return Base::get(ctx, TypeKind::Int);
}

FloatType FloatType::get(mlir::MLIRContext *ctx) {
  return Base::get(ctx, TypeKind::Float);
}

BoolType BoolType::get(mlir::MLIRContext *ctx) {
  return Base::get(ctx, TypeKind::Bool);
}

StrType StrType::get(mlir::MLIRContext *ctx) {
  return Base::get(ctx, TypeKind::Str);
}

ObjectType ObjectType::get(mlir::MLIRContext *ctx) {
  return Base::get(ctx, TypeKind::Object);
}

NoneType NoneType::get(mlir::MLIRContext *ctx) {
  return Base::get(ctx, TypeKind::None);
}

//===----------------------------------------------------------------------===//
// Composite types
//===----------------------------------------------------------------------===//

TupleType TupleType::get(mlir::MLIRContext *ctx,
                         llvm::ArrayRef<mlir::Type> elementTypes) {
  return Base::get(ctx, elementTypes);
}

llvm::ArrayRef<mlir::Type> TupleType::getElementTypes() const {
  return getImpl()->elementTypes;
}

DictType DictType::get(mlir::MLIRContext *ctx, mlir::Type keyType,
                       mlir::Type valueType) {
  return Base::get(ctx, std::make_pair(keyType, valueType));
}

mlir::Type DictType::getKeyType() const { return getImpl()->keyType; }

mlir::Type DictType::getValueType() const { return getImpl()->valueType; }

ListType ListType::get(mlir::MLIRContext *ctx, mlir::Type elementType) {
  return Base::get(ctx, elementType);
}

mlir::Type ListType::getElementType() const { return getImpl()->elementType; }

ClassType ClassType::get(mlir::MLIRContext *ctx, ::llvm::StringRef className) {
  return Base::get(ctx, className);
}

::llvm::StringRef ClassType::getClassName() const {
  return getImpl()->className;
}

ExceptionType ExceptionType::get(mlir::MLIRContext *ctx) {
  return Base::get(ctx, TypeKind::Exception);
}

TracebackType TracebackType::get(mlir::MLIRContext *ctx) {
  return Base::get(ctx, TypeKind::Traceback);
}

LocationType LocationType::get(mlir::MLIRContext *ctx) {
  return Base::get(ctx, TypeKind::Location);
}

FuncSignatureType FuncSignatureType::get(mlir::MLIRContext *ctx,
                                         llvm::ArrayRef<mlir::Type> positional,
                                         llvm::ArrayRef<mlir::Type> kwonly,
                                         mlir::Type varargType,
                                         mlir::Type kwargsType,
                                         llvm::ArrayRef<mlir::Type> results) {
  // Key order: positional, vararg, kwonly, kwargs, results
  return Base::get(ctx, std::make_tuple(positional, varargType, kwonly,
                                        kwargsType, results));
}

llvm::ArrayRef<mlir::Type> FuncSignatureType::getPositionalTypes() const {
  return getImpl()->positionalTypes;
}

llvm::ArrayRef<mlir::Type> FuncSignatureType::getKwOnlyTypes() const {
  return getImpl()->kwOnlyTypes;
}

llvm::ArrayRef<mlir::Type> FuncSignatureType::getResultTypes() const {
  return getImpl()->resultTypes;
}

bool FuncSignatureType::hasVararg() const {
  return static_cast<bool>(getImpl()->varargType);
}

mlir::Type FuncSignatureType::getVarargType() const {
  return getImpl()->varargType;
}

bool FuncSignatureType::hasKwarg() const {
  return static_cast<bool>(getImpl()->kwargsType);
}

mlir::Type FuncSignatureType::getKwargType() const {
  return getImpl()->kwargsType;
}

FuncType FuncType::get(mlir::MLIRContext *ctx, FuncSignatureType signature) {
  return Base::get(ctx, signature);
}

FuncSignatureType FuncType::getSignature() const {
  return mlir::cast<FuncSignatureType>(getImpl()->signature);
}

PrimFuncType PrimFuncType::get(mlir::MLIRContext *ctx,
                               mlir::FunctionType signature) {
  return Base::get(ctx, signature);
}

mlir::FunctionType PrimFuncType::getSignature() const {
  return getImpl()->signature;
}

CoroutineType CoroutineType::get(mlir::MLIRContext *ctx,
                                 mlir::Type resultType) {
  return Base::get(ctx, resultType);
}

mlir::Type CoroutineType::getResultType() const {
  return getImpl()->resultType;
}

TaskType TaskType::get(mlir::MLIRContext *ctx, mlir::Type resultType) {
  return Base::get(ctx, resultType);
}

mlir::Type TaskType::getResultType() const { return getImpl()->resultType; }

FutureType FutureType::get(mlir::MLIRContext *ctx, mlir::Type resultType) {
  return Base::get(ctx, resultType);
}

mlir::Type FutureType::getResultType() const { return getImpl()->resultType; }

//===----------------------------------------------------------------------===//
// Helper predicates
//===----------------------------------------------------------------------===//

bool isPyIntType(mlir::Type type) { return mlir::isa<IntType>(type); }

bool isPyFloatType(mlir::Type type) { return mlir::isa<FloatType>(type); }

bool isPyBoolType(mlir::Type type) { return mlir::isa<BoolType>(type); }

bool isPyStrType(mlir::Type type) { return mlir::isa<StrType>(type); }

bool isPyObjectType(mlir::Type type) { return mlir::isa<ObjectType>(type); }

bool isPyNoneType(mlir::Type type) { return mlir::isa<NoneType>(type); }

bool isPyTupleType(mlir::Type type) { return mlir::isa<TupleType>(type); }

bool isPyDictType(mlir::Type type) { return mlir::isa<DictType>(type); }

bool isPyListType(mlir::Type type) { return mlir::isa<ListType>(type); }

bool isPyClassType(mlir::Type type) { return mlir::isa<ClassType>(type); }

bool isPyExceptionType(mlir::Type type) {
  return mlir::isa<ExceptionType>(type);
}

bool isPyTracebackType(mlir::Type type) {
  return mlir::isa<TracebackType>(type);
}

bool isPyLocationType(mlir::Type type) { return mlir::isa<LocationType>(type); }

bool isPyFuncSigType(mlir::Type type) {
  return mlir::isa<FuncSignatureType>(type);
}

bool isPyFuncType(mlir::Type type) { return mlir::isa<FuncType>(type); }

bool isPyPrimFuncType(mlir::Type type) { return mlir::isa<PrimFuncType>(type); }

bool isPyCoroutineType(mlir::Type type) {
  return mlir::isa<CoroutineType>(type);
}

bool isPyTaskType(mlir::Type type) { return mlir::isa<TaskType>(type); }

bool isPyFutureType(mlir::Type type) { return mlir::isa<FutureType>(type); }

bool isCallableType(mlir::Type type) { return mlir::isa<FuncType>(type); }

bool isAwaitableType(mlir::Type type) {
  return mlir::isa<CoroutineType, TaskType, FutureType>(type);
}

bool isPyType(mlir::Type type) {
  return llvm::TypeSwitch<mlir::Type, bool>(type)
      .Case<IntType, FloatType, BoolType, StrType, ObjectType, NoneType,
            TupleType, DictType, ListType, ClassType, ExceptionType,
            TracebackType, LocationType, FuncType, CoroutineType, TaskType,
            FutureType>([](auto) { return true; })
      .Default([](mlir::Type) { return false; });
}

//===----------------------------------------------------------------------===//
// Subtype checking (v2.1)
//===----------------------------------------------------------------------===//

bool isSubtypeOf(mlir::Type subtype, mlir::Type supertype) {
  // Reflexive: T <: T
  if (subtype == supertype)
    return true;

  // Top type: object-world T <: !py.object
  if (mlir::isa<ObjectType>(supertype))
    return isPyType(subtype) && !mlir::isa<ClassType>(subtype);

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

  auto subtypeList = mlir::dyn_cast<ListType>(subtype);
  auto supertypeList = mlir::dyn_cast<ListType>(supertype);
  if (subtypeList && supertypeList) {
    return isSubtypeOf(subtypeList.getElementType(),
                       supertypeList.getElementType());
  }

  auto subtypeCoro = mlir::dyn_cast<CoroutineType>(subtype);
  auto supertypeCoro = mlir::dyn_cast<CoroutineType>(supertype);
  if (subtypeCoro && supertypeCoro)
    return isSubtypeOf(subtypeCoro.getResultType(),
                       supertypeCoro.getResultType());

  auto subtypeTask = mlir::dyn_cast<TaskType>(subtype);
  auto supertypeTask = mlir::dyn_cast<TaskType>(supertype);
  if (subtypeTask && supertypeTask)
    return isSubtypeOf(subtypeTask.getResultType(),
                       supertypeTask.getResultType());

  auto subtypeFuture = mlir::dyn_cast<FutureType>(subtype);
  auto supertypeFuture = mlir::dyn_cast<FutureType>(supertype);
  if (subtypeFuture && supertypeFuture)
    return isSubtypeOf(subtypeFuture.getResultType(),
                       supertypeFuture.getResultType());

  // No other subtype relations in v2.1
  // TODO(v3): Add class hierarchy support
  return false;
}

} // namespace py
