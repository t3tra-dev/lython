#include "cpp/PyVerifier/Common.h"

using namespace mlir;

namespace py {

LogicalResult CastFromPrimOp::verify() {
  Type inputType = getInput().getType();
  Type resultType = getResult().getType();

  if (isPyType(inputType))
    return emitOpError("input must be a primitive type, not !py.* type");
  if (!isPyType(resultType))
    return emitOpError("result must be a !py.* type");

  mlir::MLIRContext *ctx = getContext();

  // Allow any integer type to !py.int conversion
  if (auto intType = llvm::dyn_cast<::mlir::IntegerType>(inputType)) {
    if (resultType == IntType::get(ctx))
      return success();
    // i1 -> !py.bool is also allowed
    if (intType.getWidth() == 1 && resultType == BoolType::get(ctx))
      return success();
  }

  // Allow float types to !py.float conversion
  if (llvm::isa<::mlir::Float16Type, ::mlir::Float32Type, ::mlir::Float64Type>(
          inputType)) {
    if (resultType == FloatType::get(ctx))
      return success();
  }

  // Allow ranked tensor types to !py.str conversion (repr carrier)
  if (llvm::isa<::mlir::RankedTensorType>(inputType)) {
    if (resultType == StrType::get(ctx))
      return success();
  }

  return emitOpError("unsupported type conversion from ")
         << inputType << " to " << resultType;
}

LogicalResult CastToPrimOp::verify() {
  Type inputType = getInput().getType();
  Type resultType = getResult().getType();

  if (!isPyType(inputType))
    return emitOpError("input must be a !py.* type");
  if (isPyType(resultType))
    return emitOpError("result must be a primitive type, not !py.* type");

  StringAttr modeAttr = getModeAttr();
  if (!modeAttr)
    return emitOpError("requires 'mode' attribute");
  StringRef mode = modeAttr.getValue();
  if (mode != "exact" && mode != "truncate" && mode != "saturate")
    return emitOpError("mode must be 'exact', 'truncate', or 'saturate'");

  auto checkConversion = [&](Type pyType, Type prim) -> bool {
    return inputType == pyType && resultType == prim;
  };

  mlir::MLIRContext *ctx = getContext();
  if (checkConversion(IntType::get(ctx), ::mlir::IntegerType::get(ctx, 32)))
    return success();
  if (checkConversion(IntType::get(ctx), ::mlir::IntegerType::get(ctx, 64)))
    return success();
  if (checkConversion(FloatType::get(ctx), ::mlir::Float64Type::get(ctx)))
    return success();
  if (checkConversion(BoolType::get(ctx), ::mlir::IntegerType::get(ctx, 1)))
    return success();

  return emitOpError("unsupported type conversion from ")
         << inputType << " to " << resultType;
}

LogicalResult CastIdentityOp::verify() {
  Type inputType = getInput().getType();
  Type resultType = getResult().getType();

  bool inputIsPy = isPyType(inputType);
  bool resultIsPy = isPyType(resultType);
  if (!inputIsPy && !resultIsPy)
    return emitOpError("at least one of input or result must be a !py.* type");

  if (inputType == resultType)
    return success();

  if (inputIsPy && resultIsPy)
    return emitOpError(
        "when both sides are !py.* types the element types must match");

  return success();
}

LogicalResult UpcastOp::verify() {
  Type inputType = getInput().getType();
  Type resultType = getResult().getType();

  if (!isPyType(inputType))
    return emitOpError("input must be a !py.* type");
  if (isa<ClassType>(inputType))
    return emitOpError("static class instances cannot be upcast to !py.object");
  if (!isPyObjectType(resultType))
    return emitOpError("result must be of type !py.object");

  return success();
}

} // namespace py
