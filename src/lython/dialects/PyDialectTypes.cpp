#include "PyDialectTypes.h"

#include "Contracts.h"
#include "ExceptionTaxonomy.h"

#include "CallableArgumentMatcher.h"
#include "PyCallableShape.h"
#include "PyProtocols.h"
#include "PyTypeObject.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

#include <optional>
#include <string>

namespace py {

using contracts::isIntegerLiteralSpelling;
using contracts::manifestClassNameForContract;

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

ClassTypeStorage *
ClassTypeStorage::construct(mlir::TypeStorageAllocator &allocator,
                            const KeyTy &key) {
  return new (allocator.allocate<ClassTypeStorage>())
      ClassTypeStorage(allocator.copyInto(key));
}

IdTypeStorage *IdTypeStorage::construct(mlir::TypeStorageAllocator &allocator,
                                        const KeyTy &key) {
  return new (allocator.allocate<IdTypeStorage>()) IdTypeStorage(key);
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

static bool isBuiltinExceptionSubclassOf(llvm::StringRef subtype,
                                         llvm::StringRef supertype) {
  subtype = contractLeafName(subtype);
  supertype = contractLeafName(supertype);
  if (subtype == supertype)
    return true;
  // The shared taxonomy walk covers the primary chain and the
  // multiple-inheritance extra edges (ExceptionGroup -> Exception).
  return py::exceptions::isBuiltinExceptionSubclassName(subtype, supertype);
}

static bool isContractSubclassOf(ContractType subtype, ContractType supertype) {
  if (subtype.getContractName() == supertype.getContractName())
    return true;
  return isBuiltinExceptionSubclassOf(subtype.getContractName(),
                                      supertype.getContractName());
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

static bool isLiteralBool(LiteralType literal) {
  llvm::StringRef spelling = literal.getSpelling();
  return spelling == "True" || spelling == "False";
}

static bool isLiteralString(LiteralType literal) {
  llvm::StringRef spelling = literal.getSpelling();
  return spelling.starts_with("\"") && spelling.ends_with("\"");
}

//===----------------------------------------------------------------------===//
// Simple types
//===----------------------------------------------------------------------===//

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

InferVarType InferVarType::get(mlir::MLIRContext *ctx, unsigned id) {
  return Base::get(ctx, id);
}

unsigned InferVarType::getId() const { return getImpl()->id; }

UnpackType UnpackType::get(mlir::MLIRContext *ctx, mlir::Type packedType) {
  return Base::get(ctx, packedType);
}

mlir::Type UnpackType::getPackedType() const { return getImpl()->valueType; }

//===----------------------------------------------------------------------===//
// Composite types
//===----------------------------------------------------------------------===//

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
         (isPyNoneType(members[0]) || isPyNoneType(members[1]));
}

mlir::Type UnionType::getOptionalPayloadType() const {
  if (!isOptional())
    return {};
  llvm::ArrayRef<mlir::Type> members = getMemberTypes();
  return isPyNoneType(members[0]) ? members[1] : members[0];
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

bool isPyContractNamed(mlir::Type type, llvm::StringRef name) {
  auto contract = mlir::dyn_cast_if_present<ContractType>(type);
  return contract && contract.getContractName() == name;
}

mlir::Type pyObjectContractType(mlir::MLIRContext *ctx) {
  return ContractType::get(ctx, "builtins.object");
}

mlir::Type pyBoolContractType(mlir::MLIRContext *ctx) {
  return ContractType::get(ctx, "builtins.bool");
}

mlir::Type pyIntContractType(mlir::MLIRContext *ctx) {
  return ContractType::get(ctx, "builtins.int");
}

mlir::Type pyFloatContractType(mlir::MLIRContext *ctx) {
  return ContractType::get(ctx, "builtins.float");
}

mlir::Type pyStrContractType(mlir::MLIRContext *ctx) {
  return ContractType::get(ctx, "builtins.str");
}

mlir::Type pyNoneContractType(mlir::MLIRContext *ctx) {
  return ContractType::get(ctx, "types.NoneType");
}

bool isPyIntType(mlir::Type type) {
  if (isPyContractNamed(type, "builtins.int"))
    return true;
  auto literal = mlir::dyn_cast_if_present<LiteralType>(type);
  return literal && isIntegerLiteralSpelling(literal.getSpelling());
}

bool isPyFloatType(mlir::Type type) {
  return isPyContractNamed(type, "builtins.float");
}

bool isPyBoolType(mlir::Type type) {
  if (isPyContractNamed(type, "builtins.bool"))
    return true;
  auto literal = mlir::dyn_cast_if_present<LiteralType>(type);
  return literal && isLiteralBool(literal);
}

bool isPyStrType(mlir::Type type) {
  if (isPyContractNamed(type, "builtins.str"))
    return true;
  auto literal = mlir::dyn_cast_if_present<LiteralType>(type);
  return literal && isLiteralString(literal);
}

bool isPyNoneType(mlir::Type type) {
  if (isPyContractNamed(type, "types.NoneType"))
    return true;
  auto literal = mlir::dyn_cast_if_present<LiteralType>(type);
  return literal && literal.getSpelling() == "None";
}

bool isPyObjectType(mlir::Type type) {
  return isPyContractNamed(type, "builtins.object");
}

bool isPyTupleType(mlir::Type type) {
  return isPyContractNamed(type, "builtins.tuple");
}

bool isPyDictType(mlir::Type type) {
  return isPyContractNamed(type, "builtins.dict");
}

bool isPyListType(mlir::Type type) {
  return isPyContractNamed(type, "builtins.list");
}

bool isPyIteratorStateType(mlir::Type type) {
  return isIteratorDescriptorType(type);
}

mlir::Type primitiveIteratorStateType(mlir::MLIRContext *ctx,
                                      mlir::Type sourceType,
                                      mlir::Type elementType) {
  (void)sourceType;
  return iteratorProtocolType(ctx, elementType);
}

bool isPrimitiveIteratorStateType(mlir::Type type) {
  return isIteratorDescriptorType(type);
}

mlir::Type primitiveIteratorStateSourceType(mlir::Type type) {
  (void)type;
  return {};
}

mlir::Type primitiveIteratorStateElementType(mlir::Type type) {
  return iteratorDescriptorElementType(type);
}

mlir::Type primitiveIteratorSourceElementType(mlir::Type sourceType) {
  auto contract = mlir::dyn_cast_if_present<ContractType>(sourceType);
  if (!contract)
    return {};
  llvm::ArrayRef<mlir::Type> arguments = contract.getArguments();
  if (contract.getContractName() == "builtins.list")
    return arguments.empty() ? mlir::Type() : arguments.front();
  if (contract.getContractName() != "builtins.tuple")
    return {};

  if (arguments.empty())
    return {};
  mlir::Type elementType = arguments.front();
  for (mlir::Type current : arguments.drop_front()) {
    if (current != elementType)
      return UnionType::getNormalized(sourceType.getContext(), arguments);
  }
  return elementType;
}

mlir::Type iteratorProtocolType(mlir::MLIRContext *ctx,
                                mlir::Type elementType) {
  return ProtocolType::get(ctx, "Iterator", {elementType});
}

mlir::Type iteratorDescriptorElementType(mlir::Type type) {
  return unaryProtocolDescriptorPayloadType(type, "Iterator");
}

bool isIteratorDescriptorType(mlir::Type type) {
  return static_cast<bool>(iteratorDescriptorElementType(type));
}

bool isPyClassType(mlir::Type type) {
  (void)type;
  return false;
}

bool isPyTypeType(mlir::Type type) { return mlir::isa<TypeType>(type); }

bool isPyProtocolType(mlir::Type type) {
  return mlir::isa<ProtocolType, CallableType>(type);
}

bool isPyContractType(mlir::Type type) {
  return llvm::TypeSwitch<mlir::Type, bool>(type)
      .Case<ContractType, ProtocolType, CallableType, UnionType, LiteralType,
            OverloadType, TypeType, SelfType, TypeVarType, ParamSpecType,
            TypeVarTupleType, UnpackType, ExceptionType, ExceptionCellType,
            TracebackType, LocationType>(
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

bool isPyInferVarType(mlir::Type type) {
  return mlir::isa_and_present<InferVarType>(type);
}

bool containsPyInferVar(mlir::Type type) {
  bool found = false;
  mapPyTypeStructure(type, [&](mlir::Type node) -> std::optional<mlir::Type> {
    if (mlir::isa<InferVarType>(node)) {
      found = true;
      return node;
    }
    return std::nullopt;
  });
  return found;
}

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
      .Case<ContractType, LiteralType, TypeVarType, ParamSpecType,
            TypeVarTupleType, UnpackType, TypeType, ProtocolType, SelfType,
            CallableType, ExceptionType, ExceptionCellType, TracebackType,
            LocationType, UnionType, OverloadType>(
          [](auto) { return true; })
      .Default([](mlir::Type) { return false; });
}

bool isStaticTypeParameter(mlir::Type type) {
  if (mlir::isa<SelfType, TypeVarType, ParamSpecType, TypeVarTupleType>(type))
    return true;
  if (auto unpack = mlir::dyn_cast<UnpackType>(type))
    return isStaticTypeParameter(unpack.getPackedType());
  if (auto contract = mlir::dyn_cast_if_present<ContractType>(type))
    return contract.getContractName().starts_with("$");
  return false;
}

CallableType rebuildCallableWith(
    CallableType callable,
    llvm::function_ref<mlir::Type(mlir::Type)> mapChild) {
  bool changed = false;
  bool failed = false;
  auto mapOne = [&](mlir::Type child) -> mlir::Type {
    mlir::Type mapped = mapChild(child);
    if (child && !mapped)
      failed = true;
    changed |= mapped != child;
    return mapped;
  };
  llvm::SmallVector<mlir::Type, 8> positional;
  positional.reserve(callable.getPositionalTypes().size());
  for (mlir::Type argument : callable.getPositionalTypes())
    positional.push_back(mapOne(argument));
  llvm::SmallVector<mlir::Type, 4> kwonly;
  kwonly.reserve(callable.getKwOnlyTypes().size());
  for (mlir::Type argument : callable.getKwOnlyTypes())
    kwonly.push_back(mapOne(argument));
  llvm::SmallVector<mlir::Type, 1> results;
  results.reserve(callable.getResultTypes().size());
  for (mlir::Type result : callable.getResultTypes())
    results.push_back(mapOne(result));
  mlir::Type vararg;
  if (callable.hasVararg())
    vararg = mapOne(callable.getVarargType());
  mlir::Type kwarg;
  if (callable.hasKwarg())
    kwarg = mapOne(callable.getKwargType());
  if (failed)
    return {};
  if (!changed)
    return callable;
  return CallableType::get(
      callable.getContext(), positional, kwonly, vararg, kwarg, results,
      callable.getPositionalNames(), callable.getKwOnlyNames(),
      callable.getPositionalDefaults(), callable.getKwOnlyDefaults(),
      callable.getVarargName(), callable.getKwargName(),
      callable.getPositionalOnlyCount());
}

mlir::Type mapPyTypeStructure(
    mlir::Type type,
    llvm::function_ref<std::optional<mlir::Type>(mlir::Type)> transform) {
  if (!type)
    return type;
  if (std::optional<mlir::Type> replaced = transform(type))
    return *replaced;

  mlir::MLIRContext *context = type.getContext();
  bool failed = false;
  bool changed = false;
  auto mapChild = [&](mlir::Type child) -> mlir::Type {
    mlir::Type mapped = mapPyTypeStructure(child, transform);
    if (child && !mapped)
      failed = true;
    changed |= mapped != child;
    return mapped;
  };
  auto mapList = [&](llvm::ArrayRef<mlir::Type> children,
                     llvm::SmallVectorImpl<mlir::Type> &out) {
    out.reserve(children.size());
    for (mlir::Type child : children)
      out.push_back(mapChild(child));
  };

  if (auto unpack = mlir::dyn_cast<UnpackType>(type)) {
    mlir::Type packed = mapChild(unpack.getPackedType());
    if (failed)
      return {};
    return changed ? UnpackType::get(context, packed) : type;
  }
  if (auto typeType = mlir::dyn_cast<TypeType>(type)) {
    mlir::Type instance = mapChild(typeType.getInstanceType());
    if (failed)
      return {};
    return changed ? TypeType::get(context, instance) : type;
  }
  if (auto contract = mlir::dyn_cast<ContractType>(type)) {
    llvm::SmallVector<mlir::Type, 4> arguments;
    mapList(contract.getArguments(), arguments);
    if (failed)
      return {};
    return changed
               ? ContractType::get(context, contract.getContractName(),
                                   arguments)
               : type;
  }
  if (auto protocol = mlir::dyn_cast<ProtocolType>(type)) {
    llvm::SmallVector<mlir::Type, 4> arguments;
    mapList(protocol.getArguments(), arguments);
    if (failed)
      return {};
    return changed
               ? ProtocolType::get(context, protocol.getProtocolName(),
                                   arguments)
               : type;
  }
  if (auto unionType = mlir::dyn_cast<UnionType>(type)) {
    llvm::SmallVector<mlir::Type, 4> members;
    mapList(unionType.getMemberTypes(), members);
    if (failed)
      return {};
    return changed ? UnionType::getNormalized(context, members) : type;
  }
  if (auto callable = mlir::dyn_cast<CallableType>(type))
    return rebuildCallableWith(callable, [&](mlir::Type child) {
      return mapPyTypeStructure(child, transform);
    });
  if (auto overload = mlir::dyn_cast<OverloadType>(type)) {
    llvm::SmallVector<mlir::Type, 4> candidates;
    mapList(overload.getCandidateTypes(), candidates);
    if (failed)
      return {};
    return changed ? OverloadType::get(context, candidates) : type;
  }
  return type;
}

mlir::Type eraseStaticTypeParameters(mlir::Type type) {
  return mapPyTypeStructure(
      type, [](mlir::Type node) -> std::optional<mlir::Type> {
        if (!isPyType(node))
          return node;
        if (isStaticTypeParameter(node))
          return pyObjectContractType(node.getContext());
        return std::nullopt;
      });
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
  auto varargTuple = mlir::dyn_cast<ContractType>(signature.getVarargType());
  if (!varargTuple || varargTuple.getContractName() != "builtins.tuple" ||
      varargTuple.getArguments().size() != 1 ||
      !isPyObjectType(varargTuple.getArguments().front()))
    return false;
  auto kwargsDict = mlir::dyn_cast<ContractType>(signature.getKwargType());
  return kwargsDict && kwargsDict.getContractName() == "builtins.dict" &&
         kwargsDict.getArguments().size() == 2 &&
         isPyStrType(kwargsDict.getArguments()[0]) &&
         isPyObjectType(kwargsDict.getArguments()[1]);
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
    if (subtypeContract.getContractName() !=
        supertypeContract.getContractName()) {
      bool subclass =
          isContractSubclassOf(subtypeContract, supertypeContract);
      if (!subclass && context.from)
        // Quiet query: this is a lattice check, not a verification site —
        // names without py.class symbols (manifest builtins) are just not
        // subclasses.
        subclass = type_object::isKnownSubclassOf(
            context.from,
            manifestClassNameForContract(subtypeContract.getContractName()),
            manifestClassNameForContract(supertypeContract.getContractName()));
      if (!subclass)
        return false;
      return true;
    }
    llvm::ArrayRef<mlir::Type> subtypeArgs = subtypeContract.getArguments();
    llvm::ArrayRef<mlir::Type> supertypeArgs = supertypeContract.getArguments();
    if (subtypeArgs.size() != supertypeArgs.size())
      return false;
    if (subtypeContract.getContractName() == "builtins.list" ||
        subtypeContract.getContractName() == "builtins.dict")
      return subtypeArgs == supertypeArgs;
    for (auto [subArg, superArg] : llvm::zip(subtypeArgs, supertypeArgs))
      if (!isSubtypeOfImpl(subArg, superArg, context))
        return false;
    return true;
  }

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

  if (mlir::isa<ExceptionType>(subtype) && supertypeContract &&
      manifestClassNameForContract(supertypeContract.getContractName()) ==
          "BaseException")
    return true;

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
  if (isPyIntType(expected) &&
      (mlir::isa<mlir::IntegerType>(actual) || actual.isIndex()))
    return true;
  if (isPyBoolType(expected)) {
    auto integer = mlir::dyn_cast<mlir::IntegerType>(actual);
    return integer && integer.getWidth() == 1;
  }
  if (isPyFloatType(expected) && mlir::isa<mlir::FloatType>(actual))
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
