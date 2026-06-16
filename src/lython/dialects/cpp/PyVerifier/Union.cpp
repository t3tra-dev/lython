#include "cpp/PyVerifier/Common.h"

namespace py {

mlir::LogicalResult UnionWrapOp::verify() {
  mlir::Type inputType = getInput().getType();
  auto unionType = mlir::dyn_cast<UnionType>(getResult().getType());
  if (!unionType)
    return emitOpError("result must be a !py.union type");
  if (auto inputUnion = mlir::dyn_cast<UnionType>(inputType)) {
    // Subset injection: widening a union into a superset union.
    if (!isSubtypeOf(inputUnion, unionType, getOperation()))
      return emitOpError("input union ")
             << inputType << " is not a subset of " << unionType;
    return mlir::success();
  }
  if (!unionType.hasMember(inputType))
    return emitOpError("input type ")
           << inputType << " is not a member of " << unionType;
  return mlir::success();
}

mlir::LogicalResult UnionTestOp::verify() {
  auto unionType = mlir::dyn_cast<UnionType>(getInput().getType());
  if (!unionType)
    return emitOpError("input must be a !py.union type");
  mlir::Type member = getMember();
  if (!member || !unionType.hasMember(member))
    return emitOpError("tested type ")
           << member << " is not a member of " << unionType;
  return mlir::success();
}

mlir::LogicalResult UnionUnwrapOp::verify() {
  auto unionType = mlir::dyn_cast<UnionType>(getInput().getType());
  if (!unionType)
    return emitOpError("input must be a !py.union type");
  mlir::Type resultType = getResult().getType();
  if (auto resultUnion = mlir::dyn_cast<UnionType>(resultType)) {
    // Subset projection: the proven complement of a member test.
    if (!isSubtypeOf(resultUnion, unionType, getOperation()))
      return emitOpError("result union ")
             << resultType << " is not a subset of " << unionType;
    return mlir::success();
  }
  if (isSubtypeOf(unionType, resultType, getOperation()))
    return mlir::success();
  if (auto resultClass = mlir::dyn_cast<ClassType>(resultType)) {
    for (mlir::Type member : unionType.getMemberTypes()) {
      auto memberClass = mlir::dyn_cast<ClassType>(member);
      if (memberClass && isSubtypeOf(resultClass, memberClass, getOperation()))
        return mlir::success();
    }
  }
  if (!unionType.hasMember(resultType))
    return emitOpError("result type ")
           << resultType << " is not a member of " << unionType;
  return mlir::success();
}

} // namespace py
