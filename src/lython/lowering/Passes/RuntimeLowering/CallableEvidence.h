#pragma once

#include "RuntimeLowering/RuntimeLowering.h"

#include "cpp/PyCallableShape.h"

namespace py::runtime_lowering::callable_evidence {

inline void sortKeywordEvidence(llvm::SmallVectorImpl<std::string> &keys,
                                llvm::SmallVectorImpl<mlir::Type> &types) {
  llvm::SmallVector<unsigned, 8> order;
  order.reserve(keys.size());
  for (unsigned index = 0, end = static_cast<unsigned>(keys.size());
       index < end; ++index)
    order.push_back(index);

  llvm::sort(order,
             [&](unsigned lhs, unsigned rhs) { return keys[lhs] < keys[rhs]; });

  llvm::SmallVector<std::string, 8> sortedKeys;
  llvm::SmallVector<mlir::Type, 8> sortedTypes;
  sortedKeys.reserve(keys.size());
  sortedTypes.reserve(types.size());
  for (unsigned index : order) {
    sortedKeys.push_back(keys[index]);
    sortedTypes.push_back(types[index]);
  }
  keys.assign(sortedKeys.begin(), sortedKeys.end());
  types.assign(sortedTypes.begin(), sortedTypes.end());
}

inline unsigned fixedCallableInputCount(py::CallableType callable) {
  return static_cast<unsigned>(callable.getPositionalTypes().size() +
                               callable.getKwOnlyTypes().size());
}

inline bool mergePrefixCompatibleTypes(llvm::SmallVectorImpl<mlir::Type> &into,
                                       llvm::ArrayRef<mlir::Type> candidate) {
  unsigned common = std::min<unsigned>(into.size(), candidate.size());
  for (unsigned index = 0; index < common; ++index)
    if (into[index] != candidate[index])
      return false;
  if (candidate.size() > into.size())
    into.append(candidate.begin() + into.size(), candidate.end());
  return true;
}

inline bool
mergeSortedKeywordEvidence(llvm::SmallVectorImpl<std::string> &keys,
                           llvm::SmallVectorImpl<mlir::Type> &types,
                           llvm::ArrayRef<std::string> candidateKeys,
                           llvm::ArrayRef<mlir::Type> candidateTypes) {
  if (candidateKeys.size() != candidateTypes.size() ||
      keys.size() != types.size())
    return false;

  llvm::StringMap<mlir::Type> merged;
  for (auto [key, type] : llvm::zip(keys, types))
    merged[key] = type;
  for (auto [key, type] : llvm::zip(candidateKeys, candidateTypes)) {
    auto inserted = merged.try_emplace(key, type);
    if (!inserted.second && inserted.first->second != type)
      return false;
  }

  llvm::SmallVector<std::string, 8> mergedKeys;
  mergedKeys.reserve(merged.size());
  for (const auto &entry : merged)
    mergedKeys.push_back(entry.getKey().str());
  llvm::sort(mergedKeys);

  llvm::SmallVector<mlir::Type, 8> mergedTypes;
  mergedTypes.reserve(mergedKeys.size());
  for (llvm::StringRef key : mergedKeys)
    mergedTypes.push_back(merged.lookup(key));
  keys.assign(mergedKeys.begin(), mergedKeys.end());
  types.assign(mergedTypes.begin(), mergedTypes.end());
  return true;
}

inline std::optional<std::int64_t> integerLiteralFromValue(mlir::Value value) {
  auto constant = value.getDefiningOp<py::IntConstantOp>();
  if (!constant)
    return std::nullopt;
  std::int64_t parsed = 0;
  if (constant.getValue().getAsInteger(10, parsed))
    return std::nullopt;
  return parsed;
}

inline bool packOperandIsUnpacked(py::PackOp pack, unsigned index) {
  auto flags = pack->getAttrOfType<mlir::ArrayAttr>(kPackUnpackedOperandsAttr);
  if (!flags || index >= flags.size())
    return false;
  auto boolAttr = mlir::dyn_cast<mlir::BoolAttr>(flags[index]);
  return boolAttr && boolAttr.getValue();
}

inline mlir::Type runtimeEvidenceType(mlir::Type type) {
  if (auto literal = mlir::dyn_cast_if_present<py::LiteralType>(type)) {
    std::string contract = runtimeContractName(literal);
    if (!contract.empty())
      return runtimeContractType(type.getContext(), contract);
  }
  if (type && mlir::isa<py::ObjectType>(type))
    return runtimeContractType(type.getContext(), "builtins.object");
  return type;
}

inline bool containsStaticEvidenceParameter(mlir::Type type) {
  if (!type)
    return false;
  if (mlir::isa<py::TypeVarType, py::ParamSpecType, py::TypeVarTupleType>(type))
    return true;
  if (auto unpack = mlir::dyn_cast_if_present<py::UnpackType>(type))
    return containsStaticEvidenceParameter(unpack.getPackedType());
  if (auto callable = mlir::dyn_cast_if_present<py::CallableType>(type))
    return llvm::any_of(callable.getPositionalTypes(),
                        containsStaticEvidenceParameter) ||
           llvm::any_of(callable.getKwOnlyTypes(),
                        containsStaticEvidenceParameter) ||
           llvm::any_of(callable.getResultTypes(),
                        containsStaticEvidenceParameter) ||
           (callable.hasVararg() &&
            containsStaticEvidenceParameter(callable.getVarargType())) ||
           (callable.hasKwarg() &&
            containsStaticEvidenceParameter(callable.getKwargType()));
  if (auto contract = mlir::dyn_cast_if_present<py::ContractType>(type))
    return llvm::any_of(contract.getArguments(),
                        containsStaticEvidenceParameter);
  if (auto protocol = mlir::dyn_cast_if_present<py::ProtocolType>(type))
    return llvm::any_of(protocol.getArguments(),
                        containsStaticEvidenceParameter);
  if (auto unionType = mlir::dyn_cast_if_present<py::UnionType>(type))
    return llvm::any_of(unionType.getMemberTypes(),
                        containsStaticEvidenceParameter);
  if (auto typeType = mlir::dyn_cast_if_present<py::TypeType>(type))
    return containsStaticEvidenceParameter(typeType.getInstanceType());
  if (auto tuple = mlir::dyn_cast_if_present<py::TupleType>(type))
    return llvm::any_of(tuple.getElementTypes(),
                        containsStaticEvidenceParameter);
  if (auto list = mlir::dyn_cast_if_present<py::ListType>(type))
    return containsStaticEvidenceParameter(list.getElementType());
  if (auto dict = mlir::dyn_cast_if_present<py::DictType>(type))
    return containsStaticEvidenceParameter(dict.getKeyType()) ||
           containsStaticEvidenceParameter(dict.getValueType());
  return false;
}

inline std::optional<llvm::SmallVector<mlir::Type, 8>>
actualVarargEvidenceTypes(llvm::ArrayRef<mlir::Type> actualTypes,
                          llvm::ArrayRef<std::int64_t> indices) {
  llvm::SmallVector<mlir::Type, 8> types;
  types.reserve(indices.size());
  for (std::int64_t rawIndex : indices) {
    std::int64_t normalized = rawIndex;
    if (normalized < 0)
      normalized += static_cast<std::int64_t>(actualTypes.size());
    if (normalized < 0 ||
        normalized >= static_cast<std::int64_t>(actualTypes.size()))
      return std::nullopt;
    types.push_back(
        runtimeEvidenceType(actualTypes[static_cast<unsigned>(normalized)]));
  }
  return types;
}

inline std::optional<llvm::SmallVector<mlir::Type, 8>>
expectedVarargEvidenceTypes(py::CallableType callable,
                            llvm::ArrayRef<std::int64_t> indices,
                            unsigned actualCount) {
  if (!callable.hasVararg())
    return std::nullopt;

  lython::callable::VarargShape<mlir::Type> shape =
      py::callableVarargShape(callable.getVarargType());
  if (!shape.valid())
    return std::nullopt;

  llvm::SmallVector<mlir::Type, 8> types;
  types.reserve(indices.size());
  if (shape.kind == lython::callable::VarargShape<mlir::Type>::Kind::Repeated) {
    for (std::int64_t rawIndex : indices) {
      std::int64_t normalized = rawIndex;
      if (normalized < 0)
        normalized += actualCount;
      if (normalized < 0 ||
          normalized >= static_cast<std::int64_t>(actualCount))
        return std::nullopt;
      types.push_back(runtimeEvidenceType(*shape.repeated));
    }
    return types;
  }

  for (std::int64_t rawIndex : indices) {
    std::int64_t normalized = rawIndex;
    if (normalized < 0)
      normalized += actualCount;
    if (normalized < 0 ||
        normalized >= static_cast<std::int64_t>(shape.exact.size()))
      return std::nullopt;
    types.push_back(
        runtimeEvidenceType(shape.exact[static_cast<unsigned>(normalized)]));
  }
  return types;
}

inline std::optional<llvm::SmallVector<mlir::Type, 8>>
expectedDenseVarargEvidenceTypes(py::CallableType callable, unsigned count) {
  if (!callable.hasVararg())
    return std::nullopt;

  lython::callable::VarargShape<mlir::Type> shape =
      py::callableVarargShape(callable.getVarargType());
  if (!shape.valid())
    return std::nullopt;

  llvm::SmallVector<mlir::Type, 8> types;
  types.reserve(count);
  if (shape.kind == lython::callable::VarargShape<mlir::Type>::Kind::Repeated) {
    for (unsigned index = 0; index < count; ++index)
      types.push_back(runtimeEvidenceType(*shape.repeated));
    return types;
  }

  if (count > shape.exact.size())
    return std::nullopt;
  for (mlir::Type type :
       llvm::ArrayRef<mlir::Type>(shape.exact).take_front(count))
    types.push_back(runtimeEvidenceType(type));
  return types;
}

inline std::optional<llvm::SmallVector<mlir::Type, 8>>
expectedKwargEvidenceTypes(py::CallableType callable, unsigned count) {
  if (!callable.hasKwarg())
    return std::nullopt;

  std::optional<mlir::Type> valueType =
      py::callableKwargValueType(callable.getKwargType());
  if (!valueType)
    return std::nullopt;

  llvm::SmallVector<mlir::Type, 8> types;
  types.reserve(count);
  for (unsigned index = 0; index < count; ++index)
    types.push_back(*valueType);
  return types;
}
} // namespace py::runtime_lowering::callable_evidence
