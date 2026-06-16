#include "PyDialectTypes.h"

#include "PyTypeObject.h"

#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/STLExtras.h"
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

CallableTypeStorage *
CallableTypeStorage::construct(mlir::TypeStorageAllocator &allocator,
                               const KeyTy &key) {
  llvm::ArrayRef<mlir::Type> positional = allocator.copyInto(std::get<0>(key));
  mlir::Type vararg = std::get<1>(key);
  llvm::ArrayRef<mlir::Type> kwonly = allocator.copyInto(std::get<2>(key));
  mlir::Type kwargs = std::get<3>(key);
  llvm::ArrayRef<mlir::Type> results = allocator.copyInto(std::get<4>(key));
  unsigned positionalOnlyCount = std::get<5>(key);
  llvm::ArrayRef<mlir::StringAttr> positionalNames =
      allocator.copyInto(std::get<6>(key));
  llvm::ArrayRef<mlir::StringAttr> kwOnlyNames =
      allocator.copyInto(std::get<7>(key));
  llvm::ArrayRef<mlir::BoolAttr> positionalDefaults =
      allocator.copyInto(std::get<8>(key));
  llvm::ArrayRef<mlir::BoolAttr> kwOnlyDefaults =
      allocator.copyInto(std::get<9>(key));
  mlir::StringAttr varargName = std::get<10>(key);
  mlir::StringAttr kwargsName = std::get<11>(key);
  return new (allocator.allocate<CallableTypeStorage>()) CallableTypeStorage(
      positional, vararg, kwonly, kwargs, results, positionalOnlyCount,
      positionalNames, kwOnlyNames, positionalDefaults, kwOnlyDefaults,
      varargName, kwargsName);
}

UnaryTypeStorage *
UnaryTypeStorage::construct(mlir::TypeStorageAllocator &allocator,
                            const KeyTy &key) {
  return new (allocator.allocate<UnaryTypeStorage>()) UnaryTypeStorage(key);
}

UnionTypeStorage *
UnionTypeStorage::construct(mlir::TypeStorageAllocator &allocator,
                            const KeyTy &key) {
  llvm::ArrayRef<mlir::Type> copied = allocator.copyInto(key);
  return new (allocator.allocate<UnionTypeStorage>()) UnionTypeStorage(copied);
}

ProtocolTypeStorage *
ProtocolTypeStorage::construct(mlir::TypeStorageAllocator &allocator,
                               const KeyTy &key) {
  llvm::StringRef name = allocator.copyInto(key.first);
  llvm::ArrayRef<mlir::Type> arguments = allocator.copyInto(key.second);
  return new (allocator.allocate<ProtocolTypeStorage>())
      ProtocolTypeStorage(name, arguments);
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

NoneType NoneType::get(mlir::MLIRContext *ctx) {
  return Base::get(ctx, TypeKind::None);
}

ObjectType ObjectType::get(mlir::MLIRContext *ctx) {
  return Base::get(ctx, TypeKind::Object);
}

SelfType SelfType::get(mlir::MLIRContext *ctx) {
  return Base::get(ctx, TypeKind::Self);
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

TypeType TypeType::get(mlir::MLIRContext *ctx, mlir::Type instanceType) {
  return Base::get(ctx, instanceType);
}

mlir::Type TypeType::getInstanceType() const { return getImpl()->valueType; }

ProtocolType ProtocolType::get(mlir::MLIRContext *ctx,
                               ::llvm::StringRef protocolName,
                               llvm::ArrayRef<mlir::Type> arguments) {
  return Base::get(ctx, std::make_pair(protocolName, arguments));
}

::llvm::StringRef ProtocolType::getProtocolName() const {
  return getImpl()->protocolName;
}

llvm::ArrayRef<mlir::Type> ProtocolType::getArguments() const {
  return getImpl()->arguments;
}

ExceptionType ExceptionType::get(mlir::MLIRContext *ctx) {
  return Base::get(ctx, TypeKind::Exception);
}

ExceptionCellType ExceptionCellType::get(mlir::MLIRContext *ctx) {
  return Base::get(ctx, TypeKind::ExceptionCell);
}

TracebackType TracebackType::get(mlir::MLIRContext *ctx) {
  return Base::get(ctx, TypeKind::Traceback);
}

LocationType LocationType::get(mlir::MLIRContext *ctx) {
  return Base::get(ctx, TypeKind::Location);
}

CallableType CallableType::get(mlir::MLIRContext *ctx,
                               llvm::ArrayRef<mlir::Type> positional,
                               llvm::ArrayRef<mlir::Type> kwonly,
                               mlir::Type varargType, mlir::Type kwargsType,
                               llvm::ArrayRef<mlir::Type> results) {
  return get(ctx, positional, kwonly, varargType, kwargsType, results, {}, {},
             {}, {});
}

CallableType
CallableType::get(mlir::MLIRContext *ctx, llvm::ArrayRef<mlir::Type> positional,
                  llvm::ArrayRef<mlir::Type> kwonly, mlir::Type varargType,
                  mlir::Type kwargsType, llvm::ArrayRef<mlir::Type> results,
                  llvm::ArrayRef<mlir::StringAttr> positionalNames,
                  llvm::ArrayRef<mlir::StringAttr> kwOnlyNames,
                  llvm::ArrayRef<mlir::BoolAttr> positionalDefaults,
                  llvm::ArrayRef<mlir::BoolAttr> kwOnlyDefaults,
                  mlir::StringAttr varargName, mlir::StringAttr kwargsName,
                  unsigned positionalOnlyCount) {
  return Base::get(
      ctx, std::make_tuple(positional, varargType, kwonly, kwargsType, results,
                           positionalOnlyCount, positionalNames, kwOnlyNames,
                           positionalDefaults, kwOnlyDefaults, varargName,
                           kwargsName));
}

llvm::ArrayRef<mlir::Type> CallableType::getPositionalTypes() const {
  return getImpl()->positionalTypes;
}

llvm::ArrayRef<mlir::Type> CallableType::getKwOnlyTypes() const {
  return getImpl()->kwOnlyTypes;
}

llvm::ArrayRef<mlir::Type> CallableType::getResultTypes() const {
  return getImpl()->resultTypes;
}

bool CallableType::hasVararg() const {
  return static_cast<bool>(getImpl()->varargType);
}

mlir::Type CallableType::getVarargType() const { return getImpl()->varargType; }

bool CallableType::hasKwarg() const {
  return static_cast<bool>(getImpl()->kwargsType);
}

mlir::Type CallableType::getKwargType() const { return getImpl()->kwargsType; }

llvm::ArrayRef<mlir::StringAttr> CallableType::getPositionalNames() const {
  return getImpl()->positionalNames;
}

llvm::ArrayRef<mlir::StringAttr> CallableType::getKwOnlyNames() const {
  return getImpl()->kwOnlyNames;
}

llvm::ArrayRef<mlir::BoolAttr> CallableType::getPositionalDefaults() const {
  return getImpl()->positionalDefaults;
}

llvm::ArrayRef<mlir::BoolAttr> CallableType::getKwOnlyDefaults() const {
  return getImpl()->kwOnlyDefaults;
}

mlir::StringAttr CallableType::getVarargName() const {
  return getImpl()->varargName;
}

mlir::StringAttr CallableType::getKwargName() const {
  return getImpl()->kwargsName;
}

unsigned CallableType::getPositionalOnlyCount() const {
  return getImpl()->positionalOnlyCount;
}

bool CallableType::hasParameterMetadata() const {
  return !getPositionalNames().empty() || !getKwOnlyNames().empty() ||
         !getPositionalDefaults().empty() || !getKwOnlyDefaults().empty() ||
         static_cast<bool>(getVarargName()) ||
         static_cast<bool>(getKwargName()) || getPositionalOnlyCount() != 0;
}

mlir::Type UnionType::getNormalized(mlir::MLIRContext *ctx,
                                    llvm::ArrayRef<mlir::Type> memberTypes) {
  llvm::SmallVector<mlir::Type> flattened;
  llvm::SmallVector<mlir::Type> worklist(memberTypes.begin(),
                                         memberTypes.end());
  while (!worklist.empty()) {
    mlir::Type member = worklist.pop_back_val();
    if (!member)
      return {};
    if (auto nested = mlir::dyn_cast<UnionType>(member)) {
      worklist.append(nested.getMemberTypes().begin(),
                      nested.getMemberTypes().end());
      continue;
    }
    if (!llvm::is_contained(flattened, member))
      flattened.push_back(member);
  }
  if (flattened.empty())
    return {};
  if (flattened.size() == 1)
    return flattened.front();

  auto spelling = [](mlir::Type type) {
    std::string text;
    llvm::raw_string_ostream os(text);
    type.print(os);
    return text;
  };
  llvm::sort(flattened, [&](mlir::Type lhs, mlir::Type rhs) {
    return spelling(lhs) < spelling(rhs);
  });
  return Base::get(ctx, flattened);
}

llvm::ArrayRef<mlir::Type> UnionType::getMemberTypes() const {
  return getImpl()->memberTypes;
}

bool UnionType::hasMember(mlir::Type member) const {
  return llvm::is_contained(getMemberTypes(), member);
}

bool UnionType::isOptional() const {
  llvm::ArrayRef<mlir::Type> members = getMemberTypes();
  return members.size() == 2 &&
         (mlir::isa<NoneType>(members[0]) || mlir::isa<NoneType>(members[1]));
}

mlir::Type UnionType::getOptionalPayloadType() const {
  if (!isOptional())
    return {};
  llvm::ArrayRef<mlir::Type> members = getMemberTypes();
  return mlir::isa<NoneType>(members[0]) ? members[1] : members[0];
}

//===----------------------------------------------------------------------===//
// Helper predicates
//===----------------------------------------------------------------------===//

bool isPyIntType(mlir::Type type) { return mlir::isa<IntType>(type); }

bool isPyFloatType(mlir::Type type) { return mlir::isa<FloatType>(type); }

bool isPyBoolType(mlir::Type type) { return mlir::isa<BoolType>(type); }

bool isPyStrType(mlir::Type type) { return mlir::isa<StrType>(type); }

bool isPyNoneType(mlir::Type type) { return mlir::isa<NoneType>(type); }

bool isPyObjectType(mlir::Type type) { return mlir::isa<ObjectType>(type); }

bool isPyTupleType(mlir::Type type) { return mlir::isa<TupleType>(type); }

bool isPyDictType(mlir::Type type) { return mlir::isa<DictType>(type); }

bool isPyListType(mlir::Type type) { return mlir::isa<ListType>(type); }

bool isPyClassType(mlir::Type type) { return mlir::isa<ClassType>(type); }

bool isPyTypeType(mlir::Type type) { return mlir::isa<TypeType>(type); }

bool isPyProtocolType(mlir::Type type) { return mlir::isa<ProtocolType>(type); }

bool isPySelfType(mlir::Type type) { return mlir::isa<SelfType>(type); }

bool isPyExceptionType(mlir::Type type) {
  return mlir::isa<ExceptionType>(type);
}

bool isPyExceptionCellType(mlir::Type type) {
  return mlir::isa<ExceptionCellType>(type);
}

bool isPyTracebackType(mlir::Type type) {
  return mlir::isa<TracebackType>(type);
}

bool isPyLocationType(mlir::Type type) { return mlir::isa<LocationType>(type); }

bool isPyUnionType(mlir::Type type) { return mlir::isa<UnionType>(type); }

CallableType getCallableContract(mlir::Type type) {
  return mlir::dyn_cast_if_present<CallableType>(type);
}

bool isCallableType(mlir::Type type) {
  return static_cast<bool>(getCallableContract(type));
}

bool isPyType(mlir::Type type) {
  return llvm::TypeSwitch<mlir::Type, bool>(type)
      .Case<IntType, FloatType, BoolType, StrType, NoneType, ObjectType,
            TupleType, DictType, ListType, ClassType, TypeType, ProtocolType,
            CallableType, ExceptionType, ExceptionCellType, TracebackType,
            LocationType, UnionType>([](auto) { return true; })
      .Default([](mlir::Type) { return false; });
}

bool isCoroutineProtocolType(mlir::Type type) {
  auto protocol = mlir::dyn_cast<ProtocolType>(type);
  return protocol && protocol.getProtocolName() == "Coroutine" &&
         protocol.getArguments().size() == 3;
}

mlir::Type awaitablePayloadType(mlir::Type type) {
  if (!type)
    return {};
  if (auto asyncValue = mlir::dyn_cast<mlir::async::ValueType>(type))
    return asyncValue.getValueType();
  auto protocol = mlir::dyn_cast<ProtocolType>(type);
  if (!protocol)
    return {};
  llvm::StringRef name = protocol.getProtocolName();
  auto args = protocol.getArguments();
  if (name == "Awaitable" && args.size() == 1)
    return args[0];
  if (name == "Coroutine" && args.size() == 3)
    return args[2];
  if ((name == "Future" || name == "Task") && args.size() == 1)
    return args[0];
  return {};
}

//===----------------------------------------------------------------------===//
// Subtype checking (v2.1)
//===----------------------------------------------------------------------===//

namespace {

bool isSubtypeOfImpl(mlir::Type subtype, mlir::Type supertype,
                     mlir::Operation *from);

bool isMemberOfUnion(mlir::Type subtype, UnionType supertype,
                     mlir::Operation *from) {
  return llvm::any_of(supertype.getMemberTypes(), [&](mlir::Type member) {
    return isSubtypeOfImpl(subtype, member, from);
  });
}

bool isCallableEllipsisContract(CallableType signature) {
  if (!signature.getPositionalTypes().empty() ||
      !signature.getKwOnlyTypes().empty() || !signature.hasVararg() ||
      !signature.hasKwarg() || signature.hasParameterMetadata())
    return false;
  auto varargTuple = mlir::dyn_cast<TupleType>(signature.getVarargType());
  if (!varargTuple || varargTuple.getElementTypes().size() != 1 ||
      !mlir::isa<ObjectType>(varargTuple.getElementTypes().front()))
    return false;
  auto kwargsDict = mlir::dyn_cast<DictType>(signature.getKwargType());
  return kwargsDict && mlir::isa<StrType>(kwargsDict.getKeyType()) &&
         mlir::isa<ObjectType>(kwargsDict.getValueType());
}

mlir::Type callableVarargElementType(mlir::Type varargType) {
  auto tuple = mlir::dyn_cast_if_present<TupleType>(varargType);
  if (!tuple || tuple.getElementTypes().size() != 1)
    return {};
  return tuple.getElementTypes().front();
}

bool isCallableContractSubtypeOf(CallableType subtypeSig,
                                 CallableType supertypeSig,
                                 mlir::Operation *from) {
  if (isCallableEllipsisContract(supertypeSig)) {
    if (subtypeSig.getResultTypes().size() !=
        supertypeSig.getResultTypes().size())
      return false;
    for (auto [subResult, superResult] :
         llvm::zip(subtypeSig.getResultTypes(), supertypeSig.getResultTypes()))
      if (!isSubtypeOfImpl(subResult, superResult, from))
        return false;
    return true;
  }
  if (isCallableEllipsisContract(subtypeSig))
    return false;
  if ((!subtypeSig.hasVararg() && supertypeSig.hasVararg()) ||
      (supertypeSig.hasKwarg() && !subtypeSig.hasKwarg()) ||
      subtypeSig.getPositionalOnlyCount() >
          supertypeSig.getPositionalOnlyCount())
    return false;
  if (subtypeSig.getKwOnlyTypes().size() !=
          supertypeSig.getKwOnlyTypes().size() ||
      subtypeSig.getResultTypes().size() !=
          supertypeSig.getResultTypes().size())
    return false;

  llvm::ArrayRef<mlir::Type> subtypePositional =
      subtypeSig.getPositionalTypes();
  llvm::ArrayRef<mlir::Type> supertypePositional =
      supertypeSig.getPositionalTypes();
  if (!subtypeSig.hasVararg() &&
      subtypePositional.size() != supertypePositional.size())
    return false;
  if (subtypeSig.hasVararg() && !supertypeSig.hasVararg() &&
      subtypePositional.size() > supertypePositional.size())
    return false;
  mlir::Type subtypeVarargElement =
      subtypeSig.hasVararg()
          ? callableVarargElementType(subtypeSig.getVarargType())
          : mlir::Type();
  if (subtypeSig.hasVararg() && !subtypeVarargElement)
    return false;
  for (auto [index, superArg] : llvm::enumerate(supertypePositional)) {
    mlir::Type subArg = index < subtypePositional.size()
                            ? subtypePositional[index]
                            : subtypeVarargElement;
    if (!subArg)
      return false;
    if (index < subtypePositional.size() && subArg != superArg)
      return false;
    if (index >= subtypePositional.size() &&
        !isSubtypeOfImpl(superArg, subArg, from))
      return false;
  }
  for (auto [subArg, superArg] :
       llvm::zip(subtypeSig.getKwOnlyTypes(), supertypeSig.getKwOnlyTypes()))
    if (!isSubtypeOfImpl(superArg, subArg, from))
      return false;
  if (supertypeSig.hasVararg()) {
    mlir::Type supertypeVarargElement =
        callableVarargElementType(supertypeSig.getVarargType());
    if (!supertypeVarargElement ||
        !isSubtypeOfImpl(supertypeVarargElement, subtypeVarargElement, from))
      return false;
  }
  if (supertypeSig.hasKwarg() &&
      !isSubtypeOfImpl(supertypeSig.getKwargType(), subtypeSig.getKwargType(),
                       from))
    return false;
  for (auto [subResult, superResult] :
       llvm::zip(subtypeSig.getResultTypes(), supertypeSig.getResultTypes()))
    if (!isSubtypeOfImpl(subResult, superResult, from))
      return false;
  return true;
}

bool isSubtypeOfImpl(mlir::Type subtype, mlir::Type supertype,
                     mlir::Operation *from) {
  // Reflexive: T <: T
  if (subtype == supertype)
    return true;

  // The manifest-only object contract is the static top type.
  if (mlir::isa<ObjectType>(supertype))
    return isPyType(subtype);

  // Union membership: T <: union<...Ts> iff T in Ts, and
  // union<...Ss> <: union<...Ts> iff Ss is a subset of Ts.
  if (auto supertypeUnion = mlir::dyn_cast<UnionType>(supertype)) {
    if (auto subtypeUnion = mlir::dyn_cast<UnionType>(subtype)) {
      return llvm::all_of(
          subtypeUnion.getMemberTypes(), [&](mlir::Type member) {
            return isMemberOfUnion(member, supertypeUnion, from);
          });
    }
    return isMemberOfUnion(subtype, supertypeUnion, from);
  }

  if (auto subtypeUnion = mlir::dyn_cast<UnionType>(subtype)) {
    return llvm::all_of(subtypeUnion.getMemberTypes(), [&](mlir::Type member) {
      return isSubtypeOfImpl(member, supertype, from);
    });
  }

  auto subtypeClass = mlir::dyn_cast<ClassType>(subtype);
  auto supertypeClass = mlir::dyn_cast<ClassType>(supertype);
  if (subtypeClass && supertypeClass) {
    if (!from)
      return false;
    mlir::FailureOr<bool> result = type_object::isSubclassOf(
        from, subtypeClass.getClassName(), supertypeClass.getClassName());
    return mlir::succeeded(result) && *result;
  }

  auto subtypeMeta = mlir::dyn_cast<TypeType>(subtype);
  auto supertypeMeta = mlir::dyn_cast<TypeType>(supertype);
  if (subtypeMeta && supertypeMeta)
    return isSubtypeOfImpl(subtypeMeta.getInstanceType(),
                           supertypeMeta.getInstanceType(), from);

  auto subtypeProtocol = mlir::dyn_cast<ProtocolType>(subtype);
  auto supertypeProtocol = mlir::dyn_cast<ProtocolType>(supertype);
  if (subtypeProtocol && supertypeProtocol) {
    if (subtypeProtocol.getProtocolName() !=
        supertypeProtocol.getProtocolName())
      return false;
    auto subtypeArgs = subtypeProtocol.getArguments();
    auto supertypeArgs = supertypeProtocol.getArguments();
    if (subtypeArgs.size() != supertypeArgs.size())
      return false;
    for (auto [sub, sup] : llvm::zip(subtypeArgs, supertypeArgs))
      if (!isSubtypeOfImpl(sub, sup, from))
        return false;
    return true;
  }

  // Tuple covariance: !py.tuple<S> <: !py.tuple<T> if S <: T
  auto subtypeTuple = mlir::dyn_cast<TupleType>(subtype);
  auto supertypeTuple = mlir::dyn_cast<TupleType>(supertype);
  if (subtypeTuple && supertypeTuple) {
    auto subElems = subtypeTuple.getElementTypes();
    auto superElems = supertypeTuple.getElementTypes();
    if (subElems.size() != superElems.size())
      return false;
    for (auto [sub, sup] : llvm::zip(subElems, superElems)) {
      if (!isSubtypeOfImpl(sub, sup, from))
        return false;
    }
    return true;
  }

  auto subtypeDict = mlir::dyn_cast<DictType>(subtype);
  auto supertypeDict = mlir::dyn_cast<DictType>(supertype);
  // Mutable containers are invariant: accepting dict[K1, V1] as dict[K2, V2]
  // would let later writes violate the original element schema.
  if (subtypeDict && supertypeDict)
    return subtypeDict.getKeyType() == supertypeDict.getKeyType() &&
           subtypeDict.getValueType() == supertypeDict.getValueType();

  auto subtypeList = mlir::dyn_cast<ListType>(subtype);
  auto supertypeList = mlir::dyn_cast<ListType>(supertype);
  // Lists are mutable, so list[Cat] is not a subtype of list[Animal].
  if (subtypeList && supertypeList)
    return subtypeList.getElementType() == supertypeList.getElementType();

  CallableType subtypeCallable = getCallableContract(subtype);
  CallableType supertypeCallable = getCallableContract(supertype);
  if (subtypeCallable && supertypeCallable)
    return isCallableContractSubtypeOf(subtypeCallable, supertypeCallable,
                                       from);

  // No other subtype relations in v2.1
  return false;
}

} // namespace

bool isSubtypeOf(mlir::Type subtype, mlir::Type supertype) {
  return isSubtypeOfImpl(subtype, supertype, nullptr);
}

bool isSubtypeOf(mlir::Type subtype, mlir::Type supertype,
                 mlir::Operation *from) {
  return isSubtypeOfImpl(subtype, supertype, from);
}

} // namespace py
