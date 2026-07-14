#pragma once

// Internal solver surface of TypeSystem: unification over TypeBindingMap,
// substitution and overload selection (TypeConstraintSolver.cpp), shared with
// the inference entry points that remain in TypeSystem.cpp. Not part of the
// public emitter API.

#include "TypeSystem.h"

#include "llvm/ADT/STLExtras.h"

#include <cstdint>
#include <map>
#include <optional>
#include <string>

namespace lython::emitter {

inline std::optional<std::int64_t> literalIntegerFromType(mlir::Type type) {
  auto literal = mlir::dyn_cast_if_present<py::LiteralType>(type);
  if (!literal)
    return std::nullopt;
  std::int64_t value = 0;
  if (literal.getSpelling().getAsInteger(10, value))
    return std::nullopt;
  return value;
}

inline bool isObjectTop(const TypeSystem &types, mlir::Type type) {
  if (!type)
    return false;
  if (type == types.object())
    return true;
  if (auto contract = mlir::dyn_cast_if_present<py::ContractType>(type))
    return contract.getContractName() == "typing.Any";
  return false;
}

inline std::string manifestNameForContract(llvm::StringRef name) {
  for (llvm::StringRef prefix :
       {"builtins.", "typing.", "types.", "contextlib.", "_asyncio.",
        "asyncio.", "contextvars."}) {
    if (name.consume_front(prefix))
      return name.str();
  }
  return name.str();
}

using TypeBindingMap = std::map<std::string, mlir::Type>;

struct CallSolution {
  mlir::Type result;
  TypeBindingMap bindings;
  py::CallableType callableContract;
  std::string methodName;
  std::optional<std::string> receiverManifestClass;
  int score = 0;
};

mlir::Type substituteType(const TypeSystem &types, mlir::Type type,
                          const TypeBindingMap &bindings,
                          bool eraseUnbound = false);

// Directed match: binds the static type parameters occurring in `expected`
// against `actual`. Monomorphization uses it to recover the instantiation of
// a generic callable from a resolved ground contract.
bool bindExpectedType(const TypeSystem &types, mlir::Type expected,
                      mlir::Type actual, TypeBindingMap &bindings);

int unboundStaticParameterCount(mlir::Type type);

std::optional<CallSolution>
tryCallableApplication(const TypeSystem &types, py::CallableType callable,
                       mlir::ArrayRef<mlir::Type> positional,
                       mlir::ArrayRef<CallKeywordType> keywords,
                       TypeBindingMap bindings = {},
                       std::size_t firstParameter = 0);

std::optional<CallSolution> selectCallableApplication(
    const TypeSystem &types, llvm::ArrayRef<py::CallableType> candidates,
    mlir::ArrayRef<mlir::Type> positional,
    mlir::ArrayRef<CallKeywordType> keywords, TypeBindingMap bindings = {},
    std::size_t firstParameter = 0);

std::optional<CallSolution>
tryManifestMethod(const TypeSystem &types, mlir::Type receiverType,
                  llvm::StringRef methodName,
                  mlir::ArrayRef<mlir::Type> positional,
                  mlir::ArrayRef<CallKeywordType> keywords = {});

} // namespace lython::emitter
