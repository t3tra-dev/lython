#pragma once

#include "EmitterCore.h"
#include "EmitterPyOps.h"

#include "mlir/IR/BuiltinAttributes.h"

#include <functional>

namespace lython::emitter {

template <typename Op>
Value ModuleEmitter::emitBinarySpecial(const parser::Node &anchor,
                                       llvm::StringRef method, Value lhs,
                                       Value rhs, mlir::Type resultType) {
  // ⭐ A UNION LEFT OPERAND DISPATCHES ON ITS TAG. Every member answers the
  // operator or none does (`TypeSystem::unionOperatorResult`), so the value
  // this builds is the join of the arms -- which is exactly what CPython
  // computes without knowing which member is live:
  //
  //     def mk(n: int):
  //         if n < 0:
  //             return 1.5
  //         return n
  //     print(mk(-1) + 1)
  //     # static type !py.union<int, float> does not provide manifest
  //     # method '__add__'
  //
  // ⛔ The inactive members' lanes hold zeroed placeholders, so the arms are
  // BRANCHES and not a select over eagerly-computed values -- the same reason
  // `emitUnionStringify` is a chain rather than a fold.
  if (auto lhsUnion =
          mlir::dyn_cast_if_present<py::UnionType>(types.widenLiteral(lhs.type)))
    if (mlir::Type joined =
            types.unionOperatorResult(lhs.type, method, {rhs.type})) {
      llvm::ArrayRef<mlir::Type> members = lhsUnion.getMemberTypes();
      std::function<mlir::Value(unsigned)> arm =
          [&](unsigned index) -> mlir::Value {
        mlir::Type member = members[index];
        auto emitArm = [&]() -> mlir::Value {
          auto unwrap = py::UnionUnwrapOp::create(builder, loc(anchor), member,
                                                  lhs.value);
          Value armLhs{unwrap.getResult(), member};
          Value armRhs = rhs;
          // The operand promotion `emitBinary` makes for a mixed pair, made
          // here for the member: this arm's receiver is the member, not the
          // union the outer promotion looked at.
          mlir::Type armRight = types.widenLiteral(rhs.type);
          if (types.widenLiteral(member) == types.intType() &&
              armRight == types.floatType())
            armLhs = emitFloatFromInt(anchor, armLhs);
          else if (types.widenLiteral(member) == types.floatType() &&
                   armRight == types.intType())
            armRhs = emitFloatFromInt(anchor, armRhs);
          Value applied =
              emitBinarySpecial<Op>(anchor, method, armLhs, armRhs, joined);
          return coerceValue(applied, joined, anchor).value;
        };
        if (index + 1 >= members.size())
          return emitArm();
        auto test = py::UnionTestOp::create(builder, loc(anchor),
                                            builder.getI1Type(), lhs.value,
                                            mlir::TypeAttr::get(member));
        return emitValueDiamond(loc(anchor), test.getResult(), joined, emitArm,
                                [&] { return arm(index + 1); });
      };
      return Value{arm(0), joined};
    }
  if (auto rhsUnion =
          mlir::dyn_cast_if_present<py::UnionType>(types.widenLiteral(rhs.type)))
    if (mlir::Type joined =
            types.unionArgumentOperatorResult(lhs.type, method, rhs.type)) {
      llvm::ArrayRef<mlir::Type> members = rhsUnion.getMemberTypes();
      std::function<mlir::Value(unsigned)> arm =
          [&](unsigned index) -> mlir::Value {
        mlir::Type member = members[index];
        auto emitArm = [&]() -> mlir::Value {
          auto unwrap = py::UnionUnwrapOp::create(builder, loc(anchor), member,
                                                  rhs.value);
          Value armLhs = lhs;
          Value armRhs{unwrap.getResult(), member};
          mlir::Type armLeft = types.widenLiteral(lhs.type);
          if (armLeft == types.intType() &&
              types.widenLiteral(member) == types.floatType())
            armLhs = emitFloatFromInt(anchor, armLhs);
          else if (armLeft == types.floatType() &&
                   types.widenLiteral(member) == types.intType())
            armRhs = emitFloatFromInt(anchor, armRhs);
          Value applied =
              emitBinarySpecial<Op>(anchor, method, armLhs, armRhs, joined);
          return coerceValue(applied, joined, anchor).value;
        };
        if (index + 1 >= members.size())
          return emitArm();
        auto test = py::UnionTestOp::create(builder, loc(anchor),
                                            builder.getI1Type(), rhs.value,
                                            mlir::TypeAttr::get(member));
        return emitValueDiamond(loc(anchor), test.getResult(), joined, emitArm,
                                [&] { return arm(index + 1); });
      };
      return Value{arm(0), joined};
    }
  // Source-class operator methods (including MRO-inherited and dataclass-
  // synthesized ones) inline like any other source method call -- through the
  // same gate `x.m()` goes through, since `a == b` on a base-typed `a` is the
  // same unresolvable dispatch written differently.
  if (dispatchIsUnresolvable(lhs, method, /*receiverNode=*/nullptr,
                             /*throughSuper=*/false)) {
    if (std::optional<Value> dispatched =
            tryEmitVirtualDispatchWithValues(anchor, lhs, method, {rhs}))
      return *dispatched;
    if (refuseUnresolvableDispatch(anchor, lhs, method))
      return emitNone(anchor);
  }
  if (std::optional<MethodBinding> binding =
          lookupClassMethod(lhs.type, method);
      binding && binding->method)
    return emitInlineOperatorCall(anchor, lhs, *binding, {rhs});
  CallInferenceResult inference =
      types.inferMethodCallWithEvidence(lhs.type, method, {rhs.type});
  // The left operand has no answer of its own: CPython asks the right one for
  // the reflected operator before giving up (tryEmitReflectedBinary, which
  // decides what "no answer" means for a manifest left operand).
  if (std::optional<Value> reflected = tryEmitReflectedBinary(
          anchor, method, lhs, rhs, static_cast<bool>(inference)))
    return *reflected;
  if (!requireStaticEvidence(anchor, inference))
    return emitNone(anchor);
  if (inference)
    resultType = inference.resultType;
  auto op = Op::create(builder, loc(anchor), resultType,
                       mlir::FlatSymbolRefAttr::get(&context, method), method,
                       callProtocolFor(inference), lhs.value, rhs.value);
  return {op.getResult(), resultType};
}

template <typename Op>
Value ModuleEmitter::emitUnarySpecial(const parser::Node &anchor,
                                      llvm::StringRef method, Value input,
                                      mlir::Type resultType) {
  // The same tag dispatch the binary form takes; see the note there.
  if (auto inputUnion = mlir::dyn_cast_if_present<py::UnionType>(
          types.widenLiteral(input.type)))
    if (mlir::Type joined = types.unionOperatorResult(input.type, method, {})) {
      llvm::ArrayRef<mlir::Type> members = inputUnion.getMemberTypes();
      std::function<mlir::Value(unsigned)> arm =
          [&](unsigned index) -> mlir::Value {
        mlir::Type member = members[index];
        auto emitArm = [&]() -> mlir::Value {
          auto unwrap = py::UnionUnwrapOp::create(builder, loc(anchor), member,
                                                  input.value);
          Value applied = emitUnarySpecial<Op>(
              anchor, method, Value{unwrap.getResult(), member}, joined);
          return coerceValue(applied, joined, anchor).value;
        };
        if (index + 1 >= members.size())
          return emitArm();
        auto test = py::UnionTestOp::create(builder, loc(anchor),
                                            builder.getI1Type(), input.value,
                                            mlir::TypeAttr::get(member));
        return emitValueDiamond(loc(anchor), test.getResult(), joined, emitArm,
                                [&] { return arm(index + 1); });
      };
      return Value{arm(0), joined};
    }
  CallInferenceResult inference =
      types.inferMethodCallWithEvidence(input.type, method, {});
  if (!requireStaticEvidence(anchor, inference))
    return emitNone(anchor);
  if (inference)
    resultType = inference.resultType;
  auto op = Op::create(builder, loc(anchor), resultType,
                       mlir::FlatSymbolRefAttr::get(&context, method), method,
                       callProtocolFor(inference), input.value);
  return {op.getResult(), resultType};
}

} // namespace lython::emitter
