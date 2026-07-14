#include "TypeInference.h"

#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <functional>

namespace lython::emitter {

namespace {

std::string typeText(mlir::Type type) {
  if (!type)
    return "<unknown>";
  std::string result;
  llvm::raw_string_ostream stream(result);
  stream << type;
  return stream.str();
}

} // namespace

mlir::Type InferenceContext::freshVar(VarKind kind, const parser::Node *origin,
                                      llvm::StringRef role) {
  unsigned id = static_cast<unsigned>(vars.size());
  vars.push_back(VarInfo{id, 0, {}, level, kind, origin, role.str()});
  return py::InferVarType::get(&context, id);
}

unsigned InferenceContext::findRep(unsigned id) {
  while (vars[id].parent != id) {
    // Path compression must be journaled too: compressing across a link a
    // speculative merge created would otherwise survive that merge's
    // rollback and silently move the variable to the wrong class.
    unsigned grandparent = vars[vars[id].parent].parent;
    if (vars[id].parent != grandparent) {
      recordMutation(id);
      vars[id].parent = grandparent;
    }
    id = vars[id].parent;
  }
  return id;
}

mlir::Type InferenceContext::resolveShallow(mlir::Type type) {
  while (auto var = mlir::dyn_cast_if_present<py::InferVarType>(type)) {
    unsigned rep = findRep(var.getId());
    if (!vars[rep].binding)
      return py::InferVarType::get(&context, rep);
    type = vars[rep].binding;
  }
  return type;
}

mlir::Type InferenceContext::zonk(mlir::Type type) {
  return py::mapPyTypeStructure(
      type, [&](mlir::Type node) -> std::optional<mlir::Type> {
        auto var = mlir::dyn_cast<py::InferVarType>(node);
        if (!var)
          return std::nullopt;
        mlir::Type resolved = resolveShallow(var);
        if (mlir::isa<py::InferVarType>(resolved))
          return resolved;
        return zonk(resolved);
      });
}

bool InferenceContext::occursAndAdjustLevels(unsigned rep, int varLevel,
                                             mlir::Type type) {
  bool occurs = false;
  py::mapPyTypeStructure(
      type, [&](mlir::Type node) -> std::optional<mlir::Type> {
        auto var = mlir::dyn_cast<py::InferVarType>(node);
        if (!var)
          return std::nullopt;
        mlir::Type resolved = resolveShallow(var);
        if (auto unbound = mlir::dyn_cast<py::InferVarType>(resolved)) {
          unsigned other = findRep(unbound.getId());
          if (other == rep) {
            occurs = true;
          } else if (varLevel < vars[other].level) {
            recordMutation(other);
            vars[other].level = varLevel;
          }
          return node;
        }
        if (occursAndAdjustLevels(rep, varLevel, resolved))
          occurs = true;
        return node;
      });
  return occurs;
}

InferenceContext::UnifyResult InferenceContext::bindVar(unsigned rep,
                                                        mlir::Type type) {
  if (occursAndAdjustLevels(rep, vars[rep].level, type)) {
    std::string subject = vars[rep].role.empty()
                              ? std::string("inferred type")
                              : ("type of " + vars[rep].role);
    return UnifyResult{false, "infinite type: " + subject + " occurs in " +
                                  typeText(zonk(type))};
  }
  recordMutation(rep);
  vars[rep].binding = type;
  ++generationCounter;
  return {};
}

InferenceContext::UnifyResult InferenceContext::mismatch(mlir::Type a,
                                                         mlir::Type b) {
  return UnifyResult{false, "cannot unify " + typeText(zonk(a)) + " with " +
                                typeText(zonk(b))};
}

InferenceContext::UnifyResult
InferenceContext::unifyLists(mlir::ArrayRef<mlir::Type> as,
                             mlir::ArrayRef<mlir::Type> bs, mlir::Type a,
                             mlir::Type b) {
  if (as.size() != bs.size())
    return mismatch(a, b);
  for (auto [elementA, elementB] : llvm::zip(as, bs))
    if (UnifyResult result = unify(elementA, elementB); !result)
      return result;
  return {};
}

InferenceContext::UnifyResult InferenceContext::unify(mlir::Type a,
                                                      mlir::Type b) {
  a = resolveShallow(a);
  b = resolveShallow(b);
  if (a == b)
    return {};
  if (!a || !b)
    return UnifyResult{false, "cannot unify an unknown type"};

  auto varA = mlir::dyn_cast<py::InferVarType>(a);
  auto varB = mlir::dyn_cast<py::InferVarType>(b);
  if (varA && varB) {
    unsigned repA = findRep(varA.getId());
    unsigned repB = findRep(varB.getId());
    if (repA == repB)
      return {};
    if (vars[repA].rank < vars[repB].rank)
      std::swap(repA, repB);
    recordMutation(repA);
    recordMutation(repB);
    vars[repB].parent = repA;
    if (vars[repA].rank == vars[repB].rank)
      ++vars[repA].rank;
    vars[repA].level = std::min(vars[repA].level, vars[repB].level);
    // Inference strictness must survive a merge with an Instantiation
    // variable: relaxing the merged variable would launder user unknowns
    // through the legacy join path.
    if (vars[repB].kind == VarKind::Inference)
      vars[repA].kind = VarKind::Inference;
    if (!vars[repA].origin) {
      vars[repA].origin = vars[repB].origin;
      vars[repA].role = vars[repB].role;
    }
    ++generationCounter;
    return {};
  }
  if (varA)
    return bindVar(findRep(varA.getId()), b);
  if (varB)
    return bindVar(findRep(varB.getId()), a);

  if (auto contractA = mlir::dyn_cast<py::ContractType>(a)) {
    auto contractB = mlir::dyn_cast<py::ContractType>(b);
    if (!contractB ||
        contractA.getContractName() != contractB.getContractName())
      return mismatch(a, b);
    return unifyLists(contractA.getArguments(), contractB.getArguments(), a,
                      b);
  }
  if (auto protocolA = mlir::dyn_cast<py::ProtocolType>(a)) {
    auto protocolB = mlir::dyn_cast<py::ProtocolType>(b);
    if (!protocolB ||
        protocolA.getProtocolName() != protocolB.getProtocolName())
      return mismatch(a, b);
    return unifyLists(protocolA.getArguments(), protocolB.getArguments(), a,
                      b);
  }
  if (auto typeA = mlir::dyn_cast<py::TypeType>(a)) {
    auto typeB = mlir::dyn_cast<py::TypeType>(b);
    if (!typeB)
      return mismatch(a, b);
    return unify(typeA.getInstanceType(), typeB.getInstanceType());
  }
  if (auto unpackA = mlir::dyn_cast<py::UnpackType>(a)) {
    auto unpackB = mlir::dyn_cast<py::UnpackType>(b);
    if (!unpackB)
      return mismatch(a, b);
    return unify(unpackA.getPackedType(), unpackB.getPackedType());
  }
  if (auto callableA = mlir::dyn_cast<py::CallableType>(a)) {
    auto callableB = mlir::dyn_cast<py::CallableType>(b);
    if (!callableB || callableA.hasVararg() != callableB.hasVararg() ||
        callableA.hasKwarg() != callableB.hasKwarg())
      return mismatch(a, b);
    if (UnifyResult result =
            unifyLists(callableA.getPositionalTypes(),
                       callableB.getPositionalTypes(), a, b);
        !result)
      return result;
    if (UnifyResult result = unifyLists(callableA.getKwOnlyTypes(),
                                        callableB.getKwOnlyTypes(), a, b);
        !result)
      return result;
    if (callableA.hasVararg())
      if (UnifyResult result =
              unify(callableA.getVarargType(), callableB.getVarargType());
          !result)
        return result;
    if (callableA.hasKwarg())
      if (UnifyResult result =
              unify(callableA.getKwargType(), callableB.getKwargType());
          !result)
        return result;
    return unifyLists(callableA.getResultTypes(), callableB.getResultTypes(),
                      a, b);
  }
  if (auto unionA = mlir::dyn_cast<py::UnionType>(a)) {
    auto unionB = mlir::dyn_cast<py::UnionType>(b);
    if (!unionB)
      return mismatch(a, b);
    // Members are canonically sorted by getNormalized, so pointwise matching
    // is deterministic. Solving member permutations would need backtracking;
    // callers that want subset acceptance go through subsumption instead.
    return unifyLists(unionA.getMemberTypes(), unionB.getMemberTypes(), a, b);
  }

  return mismatch(a, b);
}

mlir::Type InferenceContext::generalize(int baseLevel, mlir::Type type) {
  llvm::DenseMap<unsigned, mlir::Type> quantified;
  std::function<mlir::Type(mlir::Type)> walk = [&](mlir::Type node) {
    return py::mapPyTypeStructure(
        node, [&](mlir::Type inner) -> std::optional<mlir::Type> {
          auto var = mlir::dyn_cast<py::InferVarType>(inner);
          if (!var)
            return std::nullopt;
          mlir::Type resolved = resolveShallow(var);
          if (auto unbound = mlir::dyn_cast<py::InferVarType>(resolved)) {
            unsigned rep = findRep(unbound.getId());
            if (vars[rep].level <= baseLevel)
              return resolved;
            auto [it, inserted] = quantified.try_emplace(rep);
            if (inserted)
              it->second = py::TypeVarType::get(
                  &context, (generalizedPrefix +
                             llvm::Twine(generalizedCounter++))
                                .str());
            return it->second;
          }
          return walk(resolved);
        });
  };
  return walk(type);
}

mlir::Type InferenceContext::instantiate(mlir::Type scheme,
                                         const parser::Node *origin) {
  llvm::DenseMap<mlir::Type, mlir::Type> fresh;
  return py::mapPyTypeStructure(
      scheme, [&](mlir::Type node) -> std::optional<mlir::Type> {
        auto typeVar = mlir::dyn_cast<py::TypeVarType>(node);
        if (!typeVar)
          return std::nullopt;
        auto [it, inserted] = fresh.try_emplace(typeVar);
        if (inserted)
          it->second =
              freshVar(VarKind::Instantiation, origin, typeVar.getName());
        return it->second;
      });
}

} // namespace lython::emitter
