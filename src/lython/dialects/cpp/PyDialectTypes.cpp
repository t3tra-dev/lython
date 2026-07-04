#include "PyDialectTypes.h"

#include "CallableArgumentMatcher.h"
#include "PyCallableShape.h"
#include "PyProtocols.h"
#include "PyTypeObject.h"

#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

#include <optional>
#include <string>

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

namespace py {

namespace detail {

SimpleTypeStorage *
SimpleTypeStorage::construct(mlir::TypeStorageAllocator &allocator,
                             const KeyTy &key) {
  return new (allocator.allocate<SimpleTypeStorage>())
      SimpleTypeStorage(static_cast<unsigned>(key));
}

TypeListStorage *
TypeListStorage::construct(mlir::TypeStorageAllocator &allocator,
                           const KeyTy &key) {
  llvm::ArrayRef<mlir::Type> copied = allocator.copyInto(key);
  return new (allocator.allocate<TypeListStorage>()) TypeListStorage(copied);
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

IteratorStateTypeStorage *
IteratorStateTypeStorage::construct(mlir::TypeStorageAllocator &allocator,
                                    const KeyTy &key) {
  return new (allocator.allocate<IteratorStateTypeStorage>())
      IteratorStateTypeStorage(key.first, key.second);
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

ProtocolTypeStorage *
ProtocolTypeStorage::construct(mlir::TypeStorageAllocator &allocator,
                               const KeyTy &key) {
  llvm::StringRef name = allocator.copyInto(key.first);
  llvm::ArrayRef<mlir::Type> arguments = allocator.copyInto(key.second);
  return new (allocator.allocate<ProtocolTypeStorage>())
      ProtocolTypeStorage(name, arguments);
}

} // namespace detail

static llvm::StringRef contractLeafName(llvm::StringRef contract) {
  std::pair<llvm::StringRef, llvm::StringRef> split = contract.rsplit('.');
  return split.second.empty() ? split.first : split.second;
}

static llvm::StringRef builtinExceptionBase(llvm::StringRef name) {
  if (name == "Exception")
    return "BaseException";
  if (name == "RuntimeError" || name == "TypeError" || name == "ValueError" ||
      name == "ArithmeticError" || name == "LookupError" ||
      name == "AssertionError" || name == "StopIteration" ||
      name == "StopAsyncIteration")
    return "Exception";
  if (name == "ZeroDivisionError")
    return "ArithmeticError";
  if (name == "KeyError" || name == "IndexError")
    return "LookupError";
  return {};
}

static bool isBuiltinExceptionSubclassOf(llvm::StringRef subtype,
                                         llvm::StringRef supertype) {
  subtype = contractLeafName(subtype);
  supertype = contractLeafName(supertype);
  for (unsigned depth = 0; depth < 8 && !subtype.empty(); ++depth) {
    if (subtype == supertype)
      return true;
    subtype = builtinExceptionBase(subtype);
  }
  return false;
}

static bool isContractSubclassOf(ContractType subtype, ContractType supertype) {
  if (subtype.getContractName() == supertype.getContractName())
    return true;
  return isBuiltinExceptionSubclassOf(subtype.getContractName(),
                                      supertype.getContractName());
}

static bool isIntegerLiteralSpelling(llvm::StringRef spelling) {
  if (spelling.empty())
    return false;
  if (spelling.front() == '-')
    spelling = spelling.drop_front();
  return !spelling.empty() &&
         llvm::all_of(spelling, [](char ch) { return ch >= '0' && ch <= '9'; });
}

static std::optional<llvm::StringRef> literalContractName(LiteralType literal) {
  llvm::StringRef spelling = literal.getSpelling();
  if (spelling == "True" || spelling == "False")
    return llvm::StringRef("builtins.bool");
  if (spelling == "None")
    return llvm::StringRef("types.NoneType");
  if (spelling.starts_with("\"") && spelling.ends_with("\""))
    return llvm::StringRef("builtins.str");
  if (isIntegerLiteralSpelling(spelling))
    return llvm::StringRef("builtins.int");
  return std::nullopt;
}

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

TypeVarType TypeVarType::get(mlir::MLIRContext *ctx, ::llvm::StringRef name) {
  return Base::get(ctx, name);
}

::llvm::StringRef TypeVarType::getName() const { return getImpl()->className; }

ParamSpecType ParamSpecType::get(mlir::MLIRContext *ctx,
                                 ::llvm::StringRef name) {
  return Base::get(ctx, name);
}

::llvm::StringRef ParamSpecType::getName() const {
  return getImpl()->className;
}

TypeVarTupleType TypeVarTupleType::get(mlir::MLIRContext *ctx,
                                       ::llvm::StringRef name) {
  return Base::get(ctx, name);
}

::llvm::StringRef TypeVarTupleType::getName() const {
  return getImpl()->className;
}

UnpackType UnpackType::get(mlir::MLIRContext *ctx, mlir::Type packedType) {
  return Base::get(ctx, packedType);
}

mlir::Type UnpackType::getPackedType() const { return getImpl()->valueType; }

//===----------------------------------------------------------------------===//
// Composite types
//===----------------------------------------------------------------------===//

TupleType TupleType::get(mlir::MLIRContext *ctx,
                         llvm::ArrayRef<mlir::Type> elementTypes) {
  return Base::get(ctx, elementTypes);
}

llvm::ArrayRef<mlir::Type> TupleType::getElementTypes() const {
  return getImpl()->types;
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

IteratorStateType IteratorStateType::get(mlir::MLIRContext *ctx,
                                         mlir::Type sourceType,
                                         mlir::Type elementType) {
  return Base::get(ctx, std::make_pair(sourceType, elementType));
}

mlir::Type IteratorStateType::getSourceType() const {
  return getImpl()->sourceType;
}

mlir::Type IteratorStateType::getElementType() const {
  return getImpl()->elementType;
}

ClassType ClassType::get(mlir::MLIRContext *ctx, ::llvm::StringRef className) {
  return Base::get(ctx, className);
}

::llvm::StringRef ClassType::getClassName() const {
  return getImpl()->className;
}

ContractType ContractType::get(mlir::MLIRContext *ctx, ::llvm::StringRef name,
                               llvm::ArrayRef<mlir::Type> arguments) {
  return Base::get(ctx, std::make_pair(name, arguments));
}

::llvm::StringRef ContractType::getContractName() const {
  return getImpl()->protocolName;
}

llvm::ArrayRef<mlir::Type> ContractType::getArguments() const {
  return getImpl()->arguments;
}

LiteralType LiteralType::get(mlir::MLIRContext *ctx,
                             ::llvm::StringRef spelling) {
  return Base::get(ctx, spelling);
}

::llvm::StringRef LiteralType::getSpelling() const {
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
  return getImpl()->types;
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

OverloadType OverloadType::get(mlir::MLIRContext *ctx,
                               llvm::ArrayRef<mlir::Type> candidateTypes) {
  return Base::get(ctx, candidateTypes);
}

llvm::ArrayRef<mlir::Type> OverloadType::getCandidateTypes() const {
  return getImpl()->types;
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

bool isPyIteratorStateType(mlir::Type type) {
  return mlir::isa<IteratorStateType>(type);
}

mlir::Type primitiveIteratorStateType(mlir::MLIRContext *ctx,
                                      mlir::Type sourceType,
                                      mlir::Type elementType) {
  return IteratorStateType::get(ctx, sourceType, elementType);
}

bool isPrimitiveIteratorStateType(mlir::Type type) {
  return mlir::isa<IteratorStateType>(type);
}

mlir::Type primitiveIteratorStateSourceType(mlir::Type type) {
  if (auto iterator = mlir::dyn_cast_if_present<IteratorStateType>(type))
    return iterator.getSourceType();
  return {};
}

mlir::Type primitiveIteratorStateElementType(mlir::Type type) {
  if (auto iterator = mlir::dyn_cast_if_present<IteratorStateType>(type))
    return iterator.getElementType();
  return {};
}

mlir::Type primitiveIteratorSourceElementType(mlir::Type sourceType) {
  if (auto list = mlir::dyn_cast_if_present<ListType>(sourceType))
    return list.getElementType();
  auto tuple = mlir::dyn_cast_if_present<TupleType>(sourceType);
  if (!tuple)
    return {};

  llvm::ArrayRef<mlir::Type> elementTypes = tuple.getElementTypes();
  if (elementTypes.empty())
    return {};
  mlir::Type elementType = elementTypes.front();
  for (mlir::Type current : elementTypes.drop_front())
    if (current != elementType)
      return {};
  return elementType;
}

mlir::Type iteratorProtocolType(mlir::MLIRContext *ctx,
                                mlir::Type elementType) {
  return ProtocolType::get(ctx, "Iterator", {elementType});
}

mlir::Type iteratorDescriptorElementType(mlir::Type type) {
  if (mlir::Type element = primitiveIteratorStateElementType(type))
    return element;
  return unaryProtocolDescriptorPayloadType(type, "Iterator");
}

bool isIteratorDescriptorType(mlir::Type type) {
  return static_cast<bool>(iteratorDescriptorElementType(type));
}

bool isPyClassType(mlir::Type type) { return mlir::isa<ClassType>(type); }

bool isPyTypeType(mlir::Type type) { return mlir::isa<TypeType>(type); }

bool isPyProtocolType(mlir::Type type) {
  return mlir::isa<ProtocolType, CallableType>(type);
}

bool isPyContractType(mlir::Type type) {
  return llvm::TypeSwitch<mlir::Type, bool>(type)
      .Case<ContractType, ProtocolType, CallableType, UnionType, LiteralType,
            OverloadType, TypeType, SelfType, TypeVarType, ParamSpecType,
            TypeVarTupleType, UnpackType, ClassType, IntType, FloatType,
            BoolType, StrType, NoneType, ObjectType, TupleType, DictType,
            ListType, ExceptionType, TracebackType>(
          [](auto) { return true; })
      .Default([](mlir::Type) { return false; });
}

bool isPyLiteralType(mlir::Type type) { return mlir::isa<LiteralType>(type); }

bool isPySelfType(mlir::Type type) { return mlir::isa<SelfType>(type); }

bool isPyTypeVarType(mlir::Type type) { return mlir::isa<TypeVarType>(type); }

bool isPyParamSpecType(mlir::Type type) {
  return mlir::isa<ParamSpecType>(type);
}

bool isPyTypeVarTupleType(mlir::Type type) {
  return mlir::isa<TypeVarTupleType>(type);
}

bool isPyUnpackType(mlir::Type type) { return mlir::isa<UnpackType>(type); }

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

bool isPyOverloadType(mlir::Type type) { return mlir::isa<OverloadType>(type); }

CallableType getCallableContract(mlir::Type type) {
  return mlir::dyn_cast_if_present<CallableType>(type);
}

bool isCallableType(mlir::Type type) {
  return static_cast<bool>(getCallableContract(type));
}

bool isPyType(mlir::Type type) {
  return llvm::TypeSwitch<mlir::Type, bool>(type)
      .Case<IntType, FloatType, BoolType, StrType, NoneType, ObjectType,
            TupleType, DictType, ListType, IteratorStateType, ClassType,
            ContractType, LiteralType, TypeVarType, ParamSpecType,
            TypeVarTupleType, UnpackType, TypeType, ProtocolType, SelfType,
            CallableType, ExceptionType, ExceptionCellType, TracebackType,
            LocationType, UnionType, OverloadType>([](auto) { return true; })
      .Default([](mlir::Type) { return false; });
}

bool isStaticTypeParameter(mlir::Type type) {
  if (mlir::isa<SelfType, TypeVarType, ParamSpecType, TypeVarTupleType>(type))
    return true;
  if (auto unpack = mlir::dyn_cast<UnpackType>(type))
    return isStaticTypeParameter(unpack.getPackedType());
  if (auto contract = mlir::dyn_cast_if_present<ContractType>(type))
    return contract.getContractName().starts_with("$");
  auto classType = mlir::dyn_cast_if_present<ClassType>(type);
  return classType && classType.getClassName().starts_with("$");
}

mlir::Type eraseStaticTypeParameters(mlir::Type type) {
  if (!type)
    return type;
  if (auto asyncValue = mlir::dyn_cast<mlir::async::ValueType>(type)) {
    return mlir::async::ValueType::get(
        eraseStaticTypeParameters(asyncValue.getValueType()));
  }
  if (!isPyType(type))
    return type;

  mlir::MLIRContext *context = type.getContext();
  if (auto unpack = mlir::dyn_cast<UnpackType>(type))
    return UnpackType::get(context,
                           eraseStaticTypeParameters(unpack.getPackedType()));

  if (isStaticTypeParameter(type))
    return ObjectType::get(context);

  if (auto tuple = mlir::dyn_cast<TupleType>(type)) {
    llvm::SmallVector<mlir::Type> elements;
    elements.reserve(tuple.getElementTypes().size());
    for (mlir::Type element : tuple.getElementTypes())
      elements.push_back(eraseStaticTypeParameters(element));
    return TupleType::get(context, elements);
  }
  if (auto list = mlir::dyn_cast<ListType>(type))
    return ListType::get(context,
                         eraseStaticTypeParameters(list.getElementType()));
  if (auto dict = mlir::dyn_cast<DictType>(type)) {
    return DictType::get(context, eraseStaticTypeParameters(dict.getKeyType()),
                         eraseStaticTypeParameters(dict.getValueType()));
  }
  if (auto typeType = mlir::dyn_cast<TypeType>(type)) {
    return TypeType::get(context,
                         eraseStaticTypeParameters(typeType.getInstanceType()));
  }
  if (auto protocol = mlir::dyn_cast<ProtocolType>(type)) {
    llvm::SmallVector<mlir::Type> arguments;
    arguments.reserve(protocol.getArguments().size());
    for (mlir::Type argument : protocol.getArguments())
      arguments.push_back(eraseStaticTypeParameters(argument));
    return ProtocolType::get(context, protocol.getProtocolName(), arguments);
  }
  if (auto contract = mlir::dyn_cast<ContractType>(type)) {
    llvm::SmallVector<mlir::Type> arguments;
    arguments.reserve(contract.getArguments().size());
    for (mlir::Type argument : contract.getArguments())
      arguments.push_back(eraseStaticTypeParameters(argument));
    return ContractType::get(context, contract.getContractName(), arguments);
  }
  if (auto unionType = mlir::dyn_cast<UnionType>(type)) {
    llvm::SmallVector<mlir::Type> members;
    members.reserve(unionType.getMemberTypes().size());
    for (mlir::Type member : unionType.getMemberTypes())
      members.push_back(eraseStaticTypeParameters(member));
    return UnionType::getNormalized(context, members);
  }
  if (auto signature = mlir::dyn_cast<CallableType>(type)) {
    llvm::SmallVector<mlir::Type> positional;
    positional.reserve(signature.getPositionalTypes().size());
    for (mlir::Type argument : signature.getPositionalTypes())
      positional.push_back(eraseStaticTypeParameters(argument));

    llvm::SmallVector<mlir::Type> kwonly;
    kwonly.reserve(signature.getKwOnlyTypes().size());
    for (mlir::Type argument : signature.getKwOnlyTypes())
      kwonly.push_back(eraseStaticTypeParameters(argument));

    llvm::SmallVector<mlir::Type> results;
    results.reserve(signature.getResultTypes().size());
    for (mlir::Type result : signature.getResultTypes())
      results.push_back(eraseStaticTypeParameters(result));

    mlir::Type vararg;
    if (signature.hasVararg())
      vararg = eraseStaticTypeParameters(signature.getVarargType());
    mlir::Type kwarg;
    if (signature.hasKwarg())
      kwarg = eraseStaticTypeParameters(signature.getKwargType());

    return CallableType::get(
        context, positional, kwonly, vararg, kwarg, results,
        signature.getPositionalNames(), signature.getKwOnlyNames(),
        signature.getPositionalDefaults(), signature.getKwOnlyDefaults(),
        signature.getVarargName(), signature.getKwargName(),
        signature.getPositionalOnlyCount());
  }
  if (auto overload = mlir::dyn_cast<OverloadType>(type)) {
    llvm::SmallVector<mlir::Type> candidates;
    candidates.reserve(overload.getCandidateTypes().size());
    for (mlir::Type candidate : overload.getCandidateTypes())
      candidates.push_back(eraseStaticTypeParameters(candidate));
    return OverloadType::get(context, candidates);
  }
  return type;
}

mlir::Type awaitableProtocolType(mlir::MLIRContext *ctx,
                                 mlir::Type payloadType) {
  return ProtocolType::get(ctx, "Awaitable", {payloadType});
}

std::optional<llvm::SmallVector<mlir::Type, 3>>
protocolDescriptorArguments(mlir::Type type, llvm::StringRef protocolName) {
  if (!type)
    return std::nullopt;

  std::optional<std::vector<mlir::Type>> args =
      protocols::Table::get(*type.getContext())
          .protocolArgumentsFor(type, protocolName);
  if (!args)
    return std::nullopt;
  return llvm::SmallVector<mlir::Type, 3>(args->begin(), args->end());
}

mlir::Type unaryProtocolDescriptorPayloadType(mlir::Type type,
                                              llvm::StringRef protocolName) {
  std::optional<llvm::SmallVector<mlir::Type, 3>> args =
      protocolDescriptorArguments(type, protocolName);
  if (!args || args->size() != 1)
    return {};
  return args->front();
}

mlir::Type awaitableDescriptorPayloadType(mlir::Type type) {
  return unaryProtocolDescriptorPayloadType(type, "Awaitable");
}

bool isAwaitableDescriptorType(mlir::Type type) {
  return static_cast<bool>(awaitableDescriptorPayloadType(type));
}

//===----------------------------------------------------------------------===//
// Subtype checking (v2.1)
//===----------------------------------------------------------------------===//

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

namespace {

struct SubtypeContext {
  mlir::Operation *from = nullptr;
  SubtypeBindings *bindings = nullptr;
};

bool isSubtypeOfImpl(mlir::Type subtype, mlir::Type supertype,
                     SubtypeContext &context);

bool bindSelfType(mlir::Type bound, SubtypeContext &context) {
  if (!bound || !context.bindings)
    return false;
  if (!context.bindings->self) {
    context.bindings->self = bound;
    return true;
  }
  return bound == context.bindings->self ||
         isSubtypeOfImpl(bound, context.bindings->self, context);
}

bool isMemberOfUnion(mlir::Type subtype, UnionType supertype,
                     SubtypeContext &context) {
  return llvm::any_of(supertype.getMemberTypes(), [&](mlir::Type member) {
    return isSubtypeOfImpl(subtype, member, context);
  });
}

bool subtypeWithBindings(mlir::Type subtype, mlir::Type supertype,
                         SubtypeContext &context) {
  if (!context.bindings)
    return isSubtypeOfImpl(subtype, supertype, context);
  SubtypeBindings candidate = *context.bindings;
  SubtypeContext candidateContext{context.from, &candidate};
  if (!isSubtypeOfImpl(subtype, supertype, candidateContext))
    return false;
  *context.bindings = candidate;
  return true;
}

bool overloadProvides(OverloadType subtype, mlir::Type supertype,
                      SubtypeContext &context) {
  return llvm::any_of(subtype.getCandidateTypes(), [&](mlir::Type candidate) {
    return subtypeWithBindings(candidate, supertype, context);
  });
}

bool overloadCoversAllExpected(mlir::Type subtype, OverloadType supertype,
                               SubtypeContext &context) {
  auto coversAll = [&](SubtypeContext &candidateContext) {
    for (mlir::Type expectedCandidate : supertype.getCandidateTypes())
      if (!subtypeWithBindings(subtype, expectedCandidate, candidateContext))
        return false;
    return true;
  };
  if (!context.bindings)
    return coversAll(context);
  SubtypeBindings candidateBindings = *context.bindings;
  SubtypeContext candidateContext{context.from, &candidateBindings};
  if (!coversAll(candidateContext))
    return false;
  *context.bindings = candidateBindings;
  return true;
}

bool callableAcceptsInvocation(CallableType callable,
                               const CallableInvocation &invocation,
                               SubtypeContext &context) {
  std::optional<CallableSignatureShape> shape =
      callableSignatureShape(callable);
  if (!shape)
    return false;

  SubtypeBindings candidateBindings;
  if (context.bindings)
    candidateBindings = *context.bindings;
  SubtypeContext candidateContext{
      context.from, context.bindings ? &candidateBindings : nullptr};
  std::optional<CallableApplicationShapeResolution> application =
      resolveCallableApplicationShape(
          *shape, llvm::ArrayRef<mlir::Type>(invocation.positional),
          llvm::ArrayRef<CallableKeyword>(invocation.keywords),
          [&](mlir::Type expected, mlir::Type actual) {
            return subtypeWithBindings(actual, expected, candidateContext);
          },
          [](const CallableKeyword &keyword) -> llvm::StringRef {
            return keyword.name;
          },
          [](const CallableKeyword &keyword) { return keyword.type; });
  if (!application)
    return false;
  if (context.bindings)
    *context.bindings = candidateBindings;
  return true;
}

bool callableSubtypeCoversVariadicTail(CallableType subtypeSig,
                                       CallableType supertypeSig,
                                       SubtypeContext &context) {
  if (!supertypeSig.hasVararg())
    return true;
  if (!subtypeSig.hasVararg())
    return false;
  return lython::callable::matchVarargContainment(
      callableVarargShape(subtypeSig.getVarargType()),
      callableVarargShape(supertypeSig.getVarargType()), true, false,
      [&](mlir::Type supertypeElement, mlir::Type subtypeElement) {
        return subtypeWithBindings(supertypeElement, subtypeElement, context);
      },
      [](bool lhs, bool rhs) { return lhs && rhs; });
}

bool callableSubtypeCoversKeywordTail(CallableType subtypeSig,
                                      CallableType supertypeSig,
                                      SubtypeContext &context) {
  if (!supertypeSig.hasKwarg())
    return true;
  if (!subtypeSig.hasKwarg())
    return false;
  std::optional<mlir::Type> superValue =
      callableKwargValueType(supertypeSig.getKwargType());
  std::optional<mlir::Type> subValue =
      callableKwargValueType(subtypeSig.getKwargType());
  return superValue && subValue &&
         subtypeWithBindings(*superValue, *subValue, context);
}

bool callableResultsCovariant(CallableType subtypeSig,
                              CallableType supertypeSig,
                              SubtypeContext &context) {
  if (subtypeSig.getResultTypes().size() !=
      supertypeSig.getResultTypes().size())
    return false;
  for (auto [subResult, superResult] :
       llvm::zip(subtypeSig.getResultTypes(), supertypeSig.getResultTypes()))
    if (!subtypeWithBindings(subResult, superResult, context))
      return false;
  return true;
}

bool isCallableContractSubtypeOf(CallableType subtypeSig,
                                 CallableType supertypeSig,
                                 SubtypeContext &context) {
  if (isCallableEllipsisContract(supertypeSig) ||
      isCallableEllipsisContract(subtypeSig))
    return callableResultsCovariant(subtypeSig, supertypeSig, context);

  SubtypeBindings candidateBindings;
  if (context.bindings)
    candidateBindings = *context.bindings;
  SubtypeContext candidateContext{
      context.from, context.bindings ? &candidateBindings : nullptr};

  llvm::SmallVector<CallableInvocation, 4> samples;
  if (!appendCallableAcceptanceSamples(supertypeSig, samples))
    return false;
  for (const CallableInvocation &sample : samples)
    if (!callableAcceptsInvocation(subtypeSig, sample, candidateContext))
      return false;

  if (!callableSubtypeCoversVariadicTail(subtypeSig, supertypeSig,
                                         candidateContext))
    return false;
  if (!callableSubtypeCoversKeywordTail(subtypeSig, supertypeSig,
                                        candidateContext))
    return false;
  if (!callableResultsCovariant(subtypeSig, supertypeSig, candidateContext))
    return false;

  if (context.bindings)
    *context.bindings = candidateBindings;
  return true;
}

bool protocolArgumentMatchesVariance(mlir::Type subtypeArg,
                                     mlir::Type supertypeArg,
                                     protocols::Variance variance,
                                     SubtypeContext &context) {
  if (variance == protocols::Variance::Invariant)
    return subtypeArg == supertypeArg;
  if (variance == protocols::Variance::Contravariant)
    return isSubtypeOfImpl(supertypeArg, subtypeArg, context);
  return isSubtypeOfImpl(subtypeArg, supertypeArg, context);
}

bool isProtocolSubtypeOf(ProtocolType subtype, ProtocolType supertype,
                         SubtypeContext &context) {
  return protocols::Table::get(*subtype.getContext())
      .isProtocolSubtypeOf(subtype, supertype,
                           [&](mlir::Type subtypeArg, mlir::Type supertypeArg,
                               protocols::Variance variance) {
                             return protocolArgumentMatchesVariance(
                                 subtypeArg, supertypeArg, variance, context);
                           });
}

bool isSubtypeOfImpl(mlir::Type subtype, mlir::Type supertype,
                     SubtypeContext &context) {
  // Reflexive: T <: T
  if (subtype == supertype)
    return true;

  if (mlir::isa<SelfType>(supertype))
    return bindSelfType(subtype, context);
  if (mlir::isa<SelfType>(subtype))
    return bindSelfType(supertype, context);

  if (mlir::isa<TypeVarType, ParamSpecType, TypeVarTupleType>(supertype))
    return bindSelfType(subtype, context);
  if (mlir::isa<TypeVarType, ParamSpecType, TypeVarTupleType>(subtype))
    return bindSelfType(supertype, context);

  auto subtypeContract = mlir::dyn_cast<ContractType>(subtype);
  auto supertypeContract = mlir::dyn_cast<ContractType>(supertype);
  if (supertypeContract &&
      (supertypeContract.getContractName() == "typing.Any" ||
       supertypeContract.getContractName() == "builtins.object"))
    return isPyType(subtype);
  if (auto subtypeLiteral = mlir::dyn_cast<LiteralType>(subtype)) {
    if (supertypeContract) {
      std::optional<llvm::StringRef> literalContract =
          literalContractName(subtypeLiteral);
      if (literalContract &&
          *literalContract == supertypeContract.getContractName())
        return true;
    }
  }
  if (subtypeContract && supertypeContract) {
    if (!isContractSubclassOf(subtypeContract, supertypeContract))
      return false;
    if (subtypeContract.getContractName() !=
        supertypeContract.getContractName())
      return true;
    llvm::ArrayRef<mlir::Type> subtypeArgs = subtypeContract.getArguments();
    llvm::ArrayRef<mlir::Type> supertypeArgs = supertypeContract.getArguments();
    if (subtypeArgs.size() != supertypeArgs.size())
      return false;
    for (auto [subArg, superArg] : llvm::zip(subtypeArgs, supertypeArgs))
      if (!isSubtypeOfImpl(subArg, superArg, context))
        return false;
    return true;
  }

  // The manifest-only object contract is the static top type.
  if (mlir::isa<ObjectType>(supertype))
    return isPyType(subtype);

  // Union membership: T <: union<...Ts> iff T in Ts, and
  // union<...Ss> <: union<...Ts> iff Ss is a subset of Ts.
  if (auto supertypeUnion = mlir::dyn_cast<UnionType>(supertype)) {
    if (auto subtypeUnion = mlir::dyn_cast<UnionType>(subtype)) {
      return llvm::all_of(
          subtypeUnion.getMemberTypes(), [&](mlir::Type member) {
            return isMemberOfUnion(member, supertypeUnion, context);
          });
    }
    return isMemberOfUnion(subtype, supertypeUnion, context);
  }

  if (auto subtypeUnion = mlir::dyn_cast<UnionType>(subtype)) {
    return llvm::all_of(subtypeUnion.getMemberTypes(), [&](mlir::Type member) {
      return isSubtypeOfImpl(member, supertype, context);
    });
  }

  auto subtypeOverload = mlir::dyn_cast<OverloadType>(subtype);
  auto supertypeOverload = mlir::dyn_cast<OverloadType>(supertype);
  if (subtypeOverload && supertypeOverload)
    return overloadCoversAllExpected(subtypeOverload, supertypeOverload,
                                     context);
  if (subtypeOverload)
    return overloadProvides(subtypeOverload, supertype, context);
  if (supertypeOverload)
    return overloadCoversAllExpected(subtype, supertypeOverload, context);

  if (mlir::isa<ExceptionType>(subtype)) {
    auto supertypeClass = mlir::dyn_cast<ClassType>(supertype);
    if (supertypeClass && supertypeClass.getClassName() == "BaseException")
      return true;
  }

  auto subtypeClass = mlir::dyn_cast<ClassType>(subtype);
  auto supertypeClass = mlir::dyn_cast<ClassType>(supertype);
  if (subtypeClass && supertypeClass) {
    if (!context.from)
      return false;
    mlir::FailureOr<bool> result =
        type_object::isSubclassOf(context.from, subtypeClass.getClassName(),
                                  supertypeClass.getClassName());
    return mlir::succeeded(result) && *result;
  }

  auto subtypeMeta = mlir::dyn_cast<TypeType>(subtype);
  auto supertypeMeta = mlir::dyn_cast<TypeType>(supertype);
  if (subtypeMeta && supertypeMeta)
    return isSubtypeOfImpl(subtypeMeta.getInstanceType(),
                           supertypeMeta.getInstanceType(), context);

  auto subtypeProtocol = mlir::dyn_cast<ProtocolType>(subtype);
  auto supertypeProtocol = mlir::dyn_cast<ProtocolType>(supertype);
  if (subtypeProtocol && supertypeProtocol) {
    return isProtocolSubtypeOf(subtypeProtocol, supertypeProtocol, context);
  }
  if (supertypeProtocol) {
    std::optional<llvm::SmallVector<mlir::Type, 3>> subtypeArgs =
        protocolDescriptorArguments(subtype,
                                    supertypeProtocol.getProtocolName());
    if (subtypeArgs) {
      ProtocolType subtypeView =
          ProtocolType::get(subtype.getContext(),
                            supertypeProtocol.getProtocolName(), *subtypeArgs);
      return isProtocolSubtypeOf(subtypeView, supertypeProtocol, context);
    }
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
      if (!isSubtypeOfImpl(sub, sup, context))
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
                                       context);

  // No other subtype relations in v2.1
  return false;
}

} // namespace

bool isSubtypeOf(mlir::Type subtype, mlir::Type supertype) {
  SubtypeContext context;
  return isSubtypeOfImpl(subtype, supertype, context);
}

bool isSubtypeOf(mlir::Type subtype, mlir::Type supertype,
                 mlir::Operation *from) {
  SubtypeContext context{from, nullptr};
  return isSubtypeOfImpl(subtype, supertype, context);
}

bool isSubtypeOf(mlir::Type subtype, mlir::Type supertype,
                 mlir::Operation *from, SubtypeBindings *bindings) {
  SubtypeContext context{from, bindings};
  return isSubtypeOfImpl(subtype, supertype, context);
}

bool isAssignableToImpl(mlir::Type actual, mlir::Type expected,
                        SubtypeContext &context) {
  if (!actual || !expected)
    return false;
  if (isSubtypeOfImpl(actual, expected, context))
    return true;
  if (mlir::isa<IntType>(expected) &&
      (mlir::isa<mlir::IntegerType>(actual) || actual.isIndex()))
    return true;
  if (mlir::isa<BoolType>(expected)) {
    auto integer = mlir::dyn_cast<mlir::IntegerType>(actual);
    return integer && integer.getWidth() == 1;
  }
  if (mlir::isa<FloatType>(expected) && mlir::isa<mlir::FloatType>(actual))
    return true;
  return false;
}

bool isAssignableTo(mlir::Type actual, mlir::Type expected) {
  SubtypeContext context;
  return isAssignableToImpl(actual, expected, context);
}

bool isAssignableTo(mlir::Type actual, mlir::Type expected,
                    mlir::Operation *from) {
  SubtypeContext context{from, nullptr};
  return isAssignableToImpl(actual, expected, context);
}

bool isAssignableTo(mlir::Type actual, mlir::Type expected,
                    mlir::Operation *from, SubtypeBindings *bindings) {
  SubtypeContext context{from, bindings};
  return isAssignableToImpl(actual, expected, context);
}

} // namespace py
