#ifndef LYTHON_PY_EQUALITY_H
#define LYTHON_PY_EQUALITY_H

#include "PyDialectTypes.h"

namespace py {

inline bool exactScalarEqualityLoweringSupported(mlir::Type lhs,
                                                 mlir::Type rhs) {
  if (lhs != rhs)
    return false;
  return mlir::isa<IntType, FloatType, BoolType, StrType>(lhs);
}

inline bool exactScalarEqualityAlwaysFalse(mlir::Type lhs, mlir::Type rhs) {
  if (lhs == rhs)
    return false;
  auto isNumeric = [](mlir::Type type) {
    return mlir::isa<IntType, FloatType, BoolType>(type);
  };
  auto isExactScalar = [&](mlir::Type type) {
    return isNumeric(type) || mlir::isa<StrType, NoneType>(type);
  };
  if (!isExactScalar(lhs) || !isExactScalar(rhs))
    return false;
  return !(isNumeric(lhs) && isNumeric(rhs));
}

} // namespace py

#endif // LYTHON_PY_EQUALITY_H
