#include "cpp/PyVerifier/Common.h"

namespace py {

mlir::LogicalResult CastFromPrimOp::verify() {
  mlir::Type inputType = getInput().getType();
  mlir::Type resultType = getResult().getType();

  if (isPyType(inputType))
    return emitOpError("input must be a primitive type, not !py.* type");
  if (!isPyType(resultType))
    return emitOpError("result must be a !py.* type");

  mlir::MLIRContext *ctx = getContext();

  // Allow any integer type to !py.int conversion
  if (auto intType = llvm::dyn_cast<::mlir::IntegerType>(inputType)) {
    if (resultType == IntType::get(ctx))
      return mlir::success();
    // i1 -> !py.bool is also allowed
    if (intType.getWidth() == 1 && resultType == BoolType::get(ctx))
      return mlir::success();
  }

  // Allow float types to !py.float conversion
  if (llvm::isa<::mlir::Float16Type, ::mlir::Float32Type, ::mlir::Float64Type>(
          inputType)) {
    if (resultType == FloatType::get(ctx))
      return mlir::success();
  }

  // Allow ranked tensor types to !py.str conversion (repr carrier)
  if (llvm::isa<::mlir::RankedTensorType>(inputType)) {
    if (resultType == StrType::get(ctx))
      return mlir::success();
  }

  return emitOpError("unsupported type conversion from ")
         << inputType << " to " << resultType;
}

mlir::LogicalResult CastToPrimOp::verify() {
  mlir::Type inputType = getInput().getType();
  mlir::Type resultType = getResult().getType();

  if (!isPyType(inputType))
    return emitOpError("input must be a !py.* type");
  if (isPyType(resultType))
    return emitOpError("result must be a primitive type, not !py.* type");

  mlir::StringAttr modeAttr = getModeAttr();
  if (!modeAttr)
    return emitOpError("requires 'mode' attribute");
  llvm::StringRef mode = modeAttr.getValue();
  if (mode != "exact" && mode != "truncate" && mode != "saturate")
    return emitOpError("mode must be 'exact', 'truncate', or 'saturate'");

  auto checkConversion = [&](mlir::Type pyType, mlir::Type prim) -> bool {
    return inputType == pyType && resultType == prim;
  };

  mlir::MLIRContext *ctx = getContext();
  if (checkConversion(IntType::get(ctx), ::mlir::IntegerType::get(ctx, 32)))
    return mlir::success();
  if (checkConversion(IntType::get(ctx), ::mlir::IntegerType::get(ctx, 64)))
    return mlir::success();
  if (checkConversion(FloatType::get(ctx), ::mlir::Float64Type::get(ctx)))
    return mlir::success();
  if (checkConversion(BoolType::get(ctx), ::mlir::IntegerType::get(ctx, 1)))
    return mlir::success();

  return emitOpError("unsupported type conversion from ")
         << inputType << " to " << resultType;
}

} // namespace py
