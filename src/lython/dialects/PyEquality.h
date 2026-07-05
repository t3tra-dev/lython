#ifndef LYTHON_PY_EQUALITY_H
#define LYTHON_PY_EQUALITY_H

#include "PyDialectTypes.h"

namespace py {

inline bool exactScalarEqualityLoweringSupported(mlir::Type lhs,
                                                 mlir::Type rhs) {
  if (lhs != rhs)
    return false;
  return isPyIntType(lhs) || isPyFloatType(lhs) || isPyBoolType(lhs) ||
         isPyStrType(lhs);
}

inline bool exactScalarEqualityAlwaysFalse(mlir::Type lhs, mlir::Type rhs) {
  if (lhs == rhs)
    return false;
  auto isNumeric = [](mlir::Type type) {
    return isPyIntType(type) || isPyFloatType(type) || isPyBoolType(type);
  };
  auto isExactScalar = [&](mlir::Type type) {
    return isNumeric(type) || isPyStrType(type) || isPyNoneType(type);
  };
  if (!isExactScalar(lhs) || !isExactScalar(rhs))
    return false;
  return !(isNumeric(lhs) && isNumeric(rhs));
}

} // namespace py

#endif // LYTHON_PY_EQUALITY_H
