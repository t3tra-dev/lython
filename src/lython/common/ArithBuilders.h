#pragma once

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"

#include <cstdint>

namespace lython::common {

// ⭐ ONE HOME FOR THE ARITH CONSTANTS AND BOOLEAN COMBINATORS every layer
// builds. The emitter, the runtime lowering and the ctypes lowering each grew
// their own `constantBool` / `constantI64` / `logicalAnd`; they were the same
// op, and a caller reaching for one had to know which file it was in.

inline mlir::Value constantBool(mlir::OpBuilder &builder, mlir::Location loc,
                                bool value) {
  return mlir::arith::ConstantIntOp::create(builder, loc, value ? 1 : 0, 1)
      .getResult();
}

inline mlir::Value constantInt(mlir::OpBuilder &builder,
                                   mlir::Location loc, mlir::IntegerType type,
                                   std::int64_t value) {
  return mlir::arith::ConstantOp::create(builder, loc, type,
                                         builder.getIntegerAttr(type, value))
      .getResult();
}

inline mlir::Value constantI64(mlir::OpBuilder &builder, mlir::Location loc,
                               std::int64_t value) {
  return mlir::arith::ConstantIntOp::create(builder, loc, value, 64)
      .getResult();
}

inline mlir::Value constantIndex(mlir::OpBuilder &builder, mlir::Location loc,
                                 std::int64_t value) {
  return mlir::arith::ConstantIndexOp::create(builder, loc, value).getResult();
}

inline mlir::Value logicalAnd(mlir::OpBuilder &builder, mlir::Location loc,
                              mlir::Value lhs, mlir::Value rhs) {
  return mlir::arith::AndIOp::create(builder, loc, lhs, rhs).getResult();
}

inline mlir::Value logicalNot(mlir::OpBuilder &builder, mlir::Location loc,
                              mlir::Value value) {
  return mlir::arith::XOrIOp::create(builder, loc, value,
                                     constantBool(builder, loc, true))
      .getResult();
}

// Did a signed add or subtract wrap? The two differ in one predicate: an ADD
// overflows only when the operands share a sign, a SUBTRACT only when they do
// not, and both then ask whether the result's sign left the left operand's.
// Spelling that as two functions meant maintaining the same six comparisons
// twice; spelling it in two layers meant maintaining them four times.
enum class SignedArith { Add, Subtract };

inline mlir::Value signedOverflow(mlir::OpBuilder &builder, mlir::Location loc,
                                  mlir::Value lhs, mlir::Value rhs,
                                  mlir::Value result,
                                  mlir::IntegerType integerType,
                                  SignedArith arithmetic) {
  mlir::Value zero = constantInt(builder, loc, integerType, 0);
  auto negative = [&](mlir::Value value) {
    return mlir::arith::CmpIOp::create(builder, loc,
                                       mlir::arith::CmpIPredicate::slt, value,
                                       zero)
        .getResult();
  };
  mlir::Value lhsNegative = negative(lhs);
  mlir::Value rhsNegative = negative(rhs);
  mlir::Value resultNegative = negative(result);
  mlir::Value operandSigns = mlir::arith::CmpIOp::create(
                                 builder, loc,
                                 arithmetic == SignedArith::Add
                                     ? mlir::arith::CmpIPredicate::eq
                                     : mlir::arith::CmpIPredicate::ne,
                                 lhsNegative, rhsNegative)
                                 .getResult();
  mlir::Value signChanged =
      mlir::arith::CmpIOp::create(builder, loc, mlir::arith::CmpIPredicate::ne,
                                  resultNegative, lhsNegative)
          .getResult();
  return logicalAnd(builder, loc, operandSigns, signChanged);
}

} // namespace lython::common
