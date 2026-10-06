// The float lane of a primitive clone: float arithmetic, comparisons and the
// int-to-float conversion on f64 values, inside a clone that can fall back.
//
// A clone is a rehearsal (Primitive/I64Calls.cpp): whatever it cannot vouch
// for it reports as "cannot say", and the call site re-runs the boxed
// original, which raises or answers exactly. So everything here is either
// CPython's answer bit for bit, or a lane marked invalid:
//
//   + - *          arith.addf/subf/mulf, which is float_add & co.
//   /              arith.divf; a zero divisor is invalid (the original raises
//                  ZeroDivisionError). A NaN divisor is not zero, as in CPython.
//   == != < ...    arith.cmpf oeq/une/olt/ole/ogt/oge: NaN compares as Python's
//   float(i)       arith.sitofp, which rounds to nearest-even as
//                  PyLong_AsDouble does for every i64
//   float vs int   exact only while the int is within +-2^53, where the
//                  conversion is exact; otherwise invalid (CPython compares
//                  such a pair exactly, not through a rounded double)
//   - + abs        arith.negf, identity, math.absf
//
// ⛔ Not in a generator resume: its caller is the runtime's `next`, which has
// no boxed original to re-run, so "cannot say" would become a branch taken.

#include "Runtime/Core/Lowerer.h"

#include "ArithBuilders.h"
#include "Contracts.h"

#include "mlir/Dialect/Math/IR/Math.h"
#include "llvm/ADT/StringSwitch.h"

namespace py::lowering {
namespace {

using lython::common::constantBool;
using lython::common::constantI64;
using lython::common::logicalAnd;

std::optional<mlir::arith::CmpFPredicate>
floatComparePredicate(llvm::StringRef method) {
  return llvm::StringSwitch<std::optional<mlir::arith::CmpFPredicate>>(method)
      .Case("__eq__", mlir::arith::CmpFPredicate::OEQ)
      .Case("__ne__", mlir::arith::CmpFPredicate::UNE)
      .Case("__lt__", mlir::arith::CmpFPredicate::OLT)
      .Case("__le__", mlir::arith::CmpFPredicate::OLE)
      .Case("__gt__", mlir::arith::CmpFPredicate::OGT)
      .Case("__ge__", mlir::arith::CmpFPredicate::OGE)
      .Default(std::nullopt);
}

bool isFloatArithmetic(llvm::StringRef method) {
  return method == "__add__" || method == "__sub__" || method == "__mul__" ||
         method == "__truediv__";
}

bool isFloatUnary(llvm::StringRef method) {
  return method == "__neg__" || method == "__pos__" || method == "__abs__" ||
         method == "__float__";
}

} // namespace

bool RuntimeBundleLowerer::floatLaneApplies(
    mlir::Operation *op, llvm::StringRef method,
    llvm::ArrayRef<const RuntimeBundle *> sources) const {
  auto function = op->getParentOfType<mlir::func::FuncOp>();
  if (!RuntimeBundleLowerer::isPrimitiveI64CallableClone(function) ||
      function->hasAttr("ly.generator.resume"))
    return false;
  auto isFloat = [&](const RuntimeBundle *source) {
    return RuntimeBundleLowerer::hasPrimitiveF64Evidence(source);
  };
  auto isInt = [&](const RuntimeBundle *source) {
    return RuntimeBundleLowerer::hasPrimitiveI64Evidence(source);
  };
  if (sources.size() == 1) {
    if (!isFloatUnary(method))
      return false;
    return isFloat(sources[0]) || (method == "__float__" && isInt(sources[0]));
  }
  if (sources.size() != 2)
    return false;
  if (isFloatArithmetic(method))
    return isFloat(sources[0]) && isFloat(sources[1]);
  if (floatComparePredicate(method))
    return (isFloat(sources[0]) && (isFloat(sources[1]) || isInt(sources[1]))) ||
           (isInt(sources[0]) && isFloat(sources[1]));
  return false;
}

bool RuntimeBundleLowerer::readsFloatLanes(mlir::Operation *op) const {
  // A lazy float has no reference to drop and nothing to keep alive.
  if (mlir::isa<py::DecRefOp, py::KeepAliveOp>(op))
    return true;
  auto function = op->getParentOfType<mlir::func::FuncOp>();
  bool inClone = RuntimeBundleLowerer::isPrimitiveI64CallableClone(function) &&
                 !function->hasAttr("ly.generator.resume");
  if (!inClone)
    return false;
  // A call to a function with a clone is a call to the clone here, which
  // takes the lane -- and so is the argument pack that only feeds such calls.
  auto callsAClone = [&](py::CallOp call) {
    auto binding = stripReturnedObjectView(call.getCallable())
                       .getDefiningOp<py::BindingRefOp>();
    return binding &&
           RuntimeBundleLowerer::primitiveI64CloneFor(binding.getBinding())
               .has_value();
  };
  if (auto call = mlir::dyn_cast<py::CallOp>(op))
    return callsAClone(call);
  if (auto pack = mlir::dyn_cast<py::PackOp>(op))
    return packIsOnlyCallArguments(pack) &&
           llvm::all_of(pack.getResult().getUsers(), [&](mlir::Operation *user) {
             auto call = mlir::dyn_cast<py::CallOp>(user);
             return call && callsAClone(call);
           });
  auto method = op->getAttrOfType<mlir::StringAttr>("method_name");
  if (!method || op->getNumOperands() == 0 || op->getNumOperands() > 2)
    return false;
  llvm::SmallVector<const RuntimeBundle *, 2> sources;
  for (mlir::Value operand : op->getOperands()) {
    const RuntimeBundle *bundle = RuntimeBundleLowerer::bundleFor(operand);
    if (!bundle)
      return false;
    sources.push_back(bundle);
  }
  if (py::contracts::isReflectedBinaryMethodName(method.getValue()) &&
      sources.size() == 2)
    std::swap(sources[0], sources[1]);
  return RuntimeBundleLowerer::floatLaneApplies(op, method.getValue(), sources);
}

mlir::LogicalResult RuntimeBundleLowerer::lowerPrimitiveF64Special(
    mlir::Operation *op, llvm::StringRef method,
    llvm::ArrayRef<const RuntimeBundle *> sources, mlir::Value resultValue) {
  builder.setInsertionPoint(op);
  mlir::Location loc = op->getLoc();
  mlir::FloatType f64 = builder.getF64Type();
  auto laneOf = [&](const RuntimeBundle *source) {
    return source->primitiveF64 ? *source->primitiveF64 : *source->primitiveI64;
  };

  // An int operand, as a double: exact only within +-2^53.
  auto asDouble = [&](const RuntimeBundle *source)
      -> std::pair<mlir::Value, mlir::Value> {
    RuntimePrimitiveI64Evidence lane = laneOf(source);
    if (RuntimeBundleLowerer::hasPrimitiveF64Evidence(source))
      return {lane.value, lane.valid};
    mlir::Value converted =
        mlir::arith::SIToFPOp::create(builder, loc, f64, lane.value).getResult();
    return {converted, lane.valid};
  };
  auto exactInt = [&](const RuntimeBundle *source) -> mlir::Value {
    if (RuntimeBundleLowerer::hasPrimitiveF64Evidence(source))
      return constantBool(builder, loc, true);
    mlir::Value value = laneOf(source).value;
    mlir::Value limit = constantI64(builder, loc, std::int64_t{1} << 53);
    mlir::Value negativeLimit = constantI64(builder, loc, -(std::int64_t{1} << 53));
    mlir::Value low = mlir::arith::CmpIOp::create(
        builder, loc, mlir::arith::CmpIPredicate::sge, value, negativeLimit);
    mlir::Value high = mlir::arith::CmpIOp::create(
        builder, loc, mlir::arith::CmpIPredicate::sle, value, limit);
    return logicalAnd(builder, loc, low, high);
  };

  if (sources.size() == 1) {
    auto [value, valid] = asDouble(sources[0]);
    mlir::Value result = value;
    if (method == "__neg__")
      result = mlir::arith::NegFOp::create(builder, loc, value).getResult();
    else if (method == "__abs__")
      result = mlir::math::AbsFOp::create(builder, loc, value).getResult();
    RuntimeBundle bundle;
    RuntimeBundleLowerer::makePrimitiveF64Bundle(resultValue.getType(), result,
                                                 valid, bundle);
    valueBundles[resultValue] = std::move(bundle);
    return mlir::success();
  }

  auto [lhs, lhsValid] = asDouble(sources[0]);
  auto [rhs, rhsValid] = asDouble(sources[1]);
  mlir::Value valid = logicalAnd(builder, loc, lhsValid, rhsValid);

  if (std::optional<mlir::arith::CmpFPredicate> predicate =
          floatComparePredicate(method)) {
    valid = logicalAnd(builder, loc, valid, exactInt(sources[0]));
    valid = logicalAnd(builder, loc, valid, exactInt(sources[1]));
    mlir::Value compared =
        mlir::arith::CmpFOp::create(builder, loc, *predicate, lhs, rhs)
            .getResult();
    mlir::Value answer = compared;
    if (!isPinnedTrueFlag(valid)) {
      // Parked like an int comparison (I64Calls.cpp): the clone then answers
      // "cannot say" and the boxed original decides.
      RuntimeBundleLowerer::parkPrimitiveI64CloneDecision(op, valid);
      answer = logicalAnd(builder, loc, valid, compared);
    }
    RuntimeBundle bundle;
    if (mlir::failed(RuntimeBundleLowerer::makeObjectBundle(
            op, resultValue.getType(), mlir::ValueRange{answer}, bundle)))
      return mlir::failure();
    valueBundles[resultValue] = std::move(bundle);
    return mlir::success();
  }

  mlir::Value result;
  if (method == "__add__") {
    result = mlir::arith::AddFOp::create(builder, loc, lhs, rhs).getResult();
  } else if (method == "__sub__") {
    result = mlir::arith::SubFOp::create(builder, loc, lhs, rhs).getResult();
  } else if (method == "__mul__") {
    result = mlir::arith::MulFOp::create(builder, loc, lhs, rhs).getResult();
  } else if (method == "__truediv__") {
    mlir::Value zero =
        mlir::arith::ConstantFloatOp::create(builder, loc, f64,
                                             llvm::APFloat(0.0))
            .getResult();
    mlir::Value nonzero = mlir::arith::CmpFOp::create(
        builder, loc, mlir::arith::CmpFPredicate::UNE, rhs, zero);
    valid = logicalAnd(builder, loc, valid, nonzero);
    result = mlir::arith::DivFOp::create(builder, loc, lhs, rhs).getResult();
  } else {
    return op->emitError() << "unsupported float lane method " << method;
  }
  RuntimeBundle bundle;
  RuntimeBundleLowerer::makePrimitiveF64Bundle(resultValue.getType(), result,
                                               valid, bundle);
  valueBundles[resultValue] = std::move(bundle);
  return mlir::success();
}

} // namespace py::lowering
