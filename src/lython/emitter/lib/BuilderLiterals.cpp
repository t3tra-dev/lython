#include "BuilderImpl.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

#include <cstdint>

namespace lython::emitter {

Value Builder::Impl::emitConstant(const parser::Node &expr) {
  const parser::FieldValue *value = valueField(expr, "value");
  if (!value) {
    error(expr, "Constant.value is missing");
    return Value{{}, noneType()};
  }
  if (const auto *text = std::get_if<std::string>(value)) {
    mlir::Value result =
        builder.create<py::StrConstantOp>(loc(), strType(), *text);
    return Value{result, strType()};
  }
  if (const auto *integer = std::get_if<std::int64_t>(value)) {
    mlir::Value result = builder.create<py::IntConstantOp>(
        loc(), intType(), std::to_string(*integer));
    return Value{result, intType()};
  }
  if (const auto *integer = std::get_if<parser::BigInteger>(value)) {
    mlir::Value result =
        builder.create<py::IntConstantOp>(loc(), intType(), integer->decimal);
    return Value{result, intType()};
  }
  if (const auto *number = std::get_if<double>(value)) {
    mlir::Value result = builder.create<py::FloatConstantOp>(
        loc(), floatType(), builder.getF64FloatAttr(*number));
    return Value{result, floatType()};
  }
  if (std::holds_alternative<std::complex<double>>(*value)) {
    error(expr, "complex literals are parsed as CPython Constant values but "
                "complex lowering is not implemented in the C++ emitter yet");
    return Value{{}, noneType()};
  }
  if (const auto *boolean = std::get_if<bool>(value)) {
    mlir::Value prim = builder.create<mlir::arith::ConstantIntOp>(
        loc(expr), *boolean ? 1 : 0, 1);
    mlir::Value result =
        builder.create<py::CastFromPrimOp>(loc(expr), boolType(), prim);
    return Value{result, boolType()};
  }
  if (std::holds_alternative<std::monostate>(*value)) {
    mlir::Value result = builder.create<py::NoneOp>(loc(), noneType());
    return Value{result, noneType()};
  }
  error(expr, "C++ emitter does not support this constant kind yet");
  return Value{{}, noneType()};
}

} // namespace lython::emitter
