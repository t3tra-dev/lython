// The float lane: float arithmetic, comparisons and the int-to-float
// conversion on f64 values, in every function but a generator resume.
//
// Every answer is CPython's bit for bit, or the lane says it cannot answer:
//
//   + - *          arith.addf/subf/mulf, which is float_add & co.
//   /              a zero divisor cannot be answered (CPython raises
//                  ZeroDivisionError). A NaN divisor is not zero, as in CPython.
//   == != < ...    arith.cmpf oeq/une/olt/ole/ogt/oge: NaN compares as Python's
//   float(i)       arith.sitofp, which rounds to nearest-even as
//                  PyLong_AsDouble does for every i64
//   float vs int   exact only while the int is within +-2^53, where the
//                  conversion is exact; otherwise not answered (CPython
//                  compares such a pair exactly, not through a rounded double)
//   - + abs        arith.negf, identity, math.absf
//
// What happens to "cannot answer" depends on where the lane is. In a clone
// (Primitive/I64Calls.cpp) it is the clone's validity, and the call site
// re-runs the boxed original. Anywhere else the op answers on the spot: `/`
// through `truediv.f64`, which raises, and the rest through the runtime
// method on the boxed operands.
//
// ⛔ Not in a generator resume: its caller is the runtime's `next`, which has
// no boxed original to re-run, and its frame carries no float lane.

#include "Runtime/Core/Lowerer.h"

#include "ArithBuilders.h"
#include "Contracts.h"

#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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

// A float lane is a value of its own everywhere but a generator resume.
static bool floatLanesEnabled(mlir::func::FuncOp function) {
  return function && !function->hasAttr("ly.generator.resume");
}

// A boxed float reads into a lane with one load; it is a lane source too.
static bool isBoxedFloat(const RuntimeBundle *source) {
  return source && source->kind == RuntimeBundle::Kind::Object &&
         source->contractName() == "builtins.float" &&
         !source->physicalValues().empty();
}

bool RuntimeBundleLowerer::floatLaneApplies(
    mlir::Operation *op, llvm::StringRef method,
    llvm::ArrayRef<const RuntimeBundle *> sources) const {
  if (!floatLanesEnabled(op->getParentOfType<mlir::func::FuncOp>()))
    return false;
  auto isFloat = [&](const RuntimeBundle *source) {
    return RuntimeBundleLowerer::hasPrimitiveF64Evidence(source) ||
           isBoxedFloat(source);
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
  // A lazy float has no reference to take, drop or keep alive.
  if (mlir::isa<py::IncRefOp, py::DecRefOp, py::KeepAliveOp>(op))
    return true;
  auto function = op->getParentOfType<mlir::func::FuncOp>();
  if (!floatLanesEnabled(function))
    return false;
  // A call to a function with a clone is a call to the clone in a clone, which
  // takes the lane -- and so is the argument pack that only feeds such calls.
  // ⛔ Not outside a clone: there the call keeps the boxed original as its
  // fallback, and that one takes objects.
  if (RuntimeBundleLowerer::isPrimitiveI64CallableClone(function)) {
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
             llvm::all_of(pack.getResult().getUsers(),
                          [&](mlir::Operation *user) {
                            auto call = mlir::dyn_cast<py::CallOp>(user);
                            return call && callsAClone(call);
                          });
  }
  // A method call adapts each argument to its callee's input where it builds
  // the operands (`appendRuntimeSource`: an f64 input takes the lane, an
  // object input a box made there), and a store makes the slot word from the
  // lane (`materializePayloadObjectBundle`), so neither needs a box made for
  // it. ⛔ Not a call to a named function: its arguments meet union and object
  // parameters through `collectFunctionTargetRuntimeSources`, which reads
  // objects.
  if (mlir::isa<py::SetItemOp>(op))
    return true;
  if (auto call = mlir::dyn_cast<py::CallOp>(op))
    return !stripReturnedObjectView(call.getCallable())
                .getDefiningOp<py::BindingRefOp>();
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
  std::optional<RuntimeSymbol> unboxFloat =
      manifest.primitive("builtins.float", "unbox.f64");

  // Each operand as a lane: its own, or a load out of its box.
  llvm::SmallVector<RuntimePrimitiveI64Evidence, 2> lanes;
  for (const RuntimeBundle *source : sources) {
    if (source->primitiveF64) {
      lanes.push_back(*source->primitiveF64);
    } else if (source->primitiveI64) {
      lanes.push_back(*source->primitiveI64);
    } else {
      if (!unboxFloat || unboxFloat->function.getNumArguments() !=
                             source->physicalValues().size())
        return op->emitError() << "float lane operand cannot be read";
      lanes.push_back(RuntimePrimitiveI64Evidence{
          RuntimeBundleLowerer::createRuntimeCall(loc, *unboxFloat,
                                                  source->physicalValues())
              .getResult(0),
          constantBool(builder, loc, true)});
    }
  }
  auto isIntLane = [&](unsigned index) {
    return lanes[index].value.getType().isInteger(64);
  };
  // An int operand, as a double: exact only within +-2^53.
  auto asDouble = [&](unsigned index) -> mlir::Value {
    if (!isIntLane(index))
      return lanes[index].value;
    return mlir::arith::SIToFPOp::create(builder, loc, f64, lanes[index].value)
        .getResult();
  };
  auto exactInt = [&](unsigned index) -> mlir::Value {
    if (!isIntLane(index))
      return constantBool(builder, loc, true);
    mlir::Value value = lanes[index].value;
    mlir::Value limit = constantI64(builder, loc, std::int64_t{1} << 53);
    mlir::Value negativeLimit =
        constantI64(builder, loc, -(std::int64_t{1} << 53));
    mlir::Value low = mlir::arith::CmpIOp::create(
        builder, loc, mlir::arith::CmpIPredicate::sge, value, negativeLimit);
    mlir::Value high = mlir::arith::CmpIOp::create(
        builder, loc, mlir::arith::CmpIPredicate::sle, value, limit);
    return logicalAnd(builder, loc, low, high);
  };

  std::optional<mlir::arith::CmpFPredicate> predicate =
      floatComparePredicate(method);
  mlir::Value valid = lanes[0].valid;
  for (unsigned index = 1; index < lanes.size(); ++index)
    valid = logicalAnd(builder, loc, valid, lanes[index].valid);
  mlir::Value result;
  if (sources.size() == 1) {
    mlir::Value value = asDouble(0);
    result = value;
    if (method == "__neg__")
      result = mlir::arith::NegFOp::create(builder, loc, value).getResult();
    else if (method == "__abs__")
      result = mlir::math::AbsFOp::create(builder, loc, value).getResult();
  } else if (predicate) {
    valid = logicalAnd(builder, loc, valid, exactInt(0));
    valid = logicalAnd(builder, loc, valid, exactInt(1));
    result = mlir::arith::CmpFOp::create(builder, loc, *predicate, asDouble(0),
                                         asDouble(1))
                 .getResult();
  } else {
    mlir::Value lhs = asDouble(0);
    mlir::Value rhs = asDouble(1);
    if (method == "__add__") {
      result = mlir::arith::AddFOp::create(builder, loc, lhs, rhs).getResult();
    } else if (method == "__sub__") {
      result = mlir::arith::SubFOp::create(builder, loc, lhs, rhs).getResult();
    } else if (method == "__mul__") {
      result = mlir::arith::MulFOp::create(builder, loc, lhs, rhs).getResult();
    } else if (method == "__truediv__" &&
               !RuntimeBundleLowerer::isPrimitiveI64CallableClone(
                   op->getParentOfType<mlir::func::FuncOp>())) {
      // ⛔ Not the slow arm below: it boxes the operands for the runtime's
      // `__truediv__`, which then raises with the boxes live, and a box made
      // inside the arm is not released on that unwind. The f64 division
      // raises with nothing to release.
      std::optional<RuntimeSymbol> divide =
          manifest.primitive("builtins.float", "truediv.f64");
      if (!divide)
        return op->emitError() << "runtime manifest has no float truediv.f64";
      result = RuntimeBundleLowerer::createRuntimeCall(
                   loc, *divide, mlir::ValueRange{lhs, rhs})
                   .getResult(0);
    } else if (method == "__truediv__") {
      mlir::Value zero = mlir::arith::ConstantFloatOp::create(
                             builder, loc, f64, llvm::APFloat(0.0))
                             .getResult();
      mlir::Value nonzero = mlir::arith::CmpFOp::create(
          builder, loc, mlir::arith::CmpFPredicate::UNE, rhs, zero);
      valid = logicalAnd(builder, loc, valid, nonzero);
      result = mlir::arith::DivFOp::create(builder, loc, lhs, rhs).getResult();
    } else {
      return op->emitError() << "unsupported float lane method " << method;
    }
  }

  auto bindResult = [&](mlir::Value answer, mlir::Value answerValid)
      -> mlir::LogicalResult {
    RuntimeBundle bundle;
    if (predicate) {
      if (mlir::failed(RuntimeBundleLowerer::makeObjectBundle(
              op, resultValue.getType(), mlir::ValueRange{answer}, bundle)))
        return mlir::failure();
    } else {
      RuntimeBundleLowerer::makePrimitiveF64Bundle(resultValue.getType(),
                                                   answer, answerValid, bundle);
    }
    valueBundles[resultValue] = std::move(bundle);
    return mlir::success();
  };

  auto function = op->getParentOfType<mlir::func::FuncOp>();
  if (isPinnedTrueFlag(valid))
    return bindResult(result, valid);
  if (RuntimeBundleLowerer::isPrimitiveI64CallableClone(function)) {
    // A rehearsal says "cannot say" and the call site re-runs the boxed
    // original (Primitive/I64Calls.cpp); a comparison parks that, having no
    // lane to carry it in.
    if (!predicate)
      return bindResult(result, valid);
    RuntimeBundleLowerer::parkPrimitiveI64CloneDecision(op, valid);
    return bindResult(logicalAnd(builder, loc, valid, result), valid);
  }

  // Anywhere else nobody re-runs anything: what the lane cannot answer is
  // answered here, by the runtime method on the boxed operands, which raises
  // (a zero divisor) or answers exactly (an int past 2^53).
  context->loadDialect<mlir::scf::SCFDialect>();
  auto ifOp = mlir::scf::IfOp::create(builder, loc,
                                      mlir::TypeRange{result.getType()}, valid,
                                      /*withElseRegion=*/true);
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  mlir::scf::YieldOp::create(builder, loc, mlir::ValueRange{result});
  builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
  llvm::SmallVector<RuntimeBundle, 2> boxed;
  boxed.reserve(sources.size());
  llvm::SmallVector<const RuntimeBundle *, 2> slowSources;
  for (const RuntimeBundle *source : sources) {
    if (!source->physicalValues().empty()) {
      slowSources.push_back(source);
      continue;
    }
    mlir::FailureOr<RuntimeValue> object =
        RuntimeBundleLowerer::hasLazyPrimitiveF64Object(*source)
            ? RuntimeBundleLowerer::
                  materializePrimitiveF64ObjectAtCurrentInsertion(op, *source)
            : RuntimeBundleLowerer::
                  materializePrimitiveI64ObjectAtCurrentInsertion(op, *source);
    if (mlir::failed(object))
      return mlir::failure();
    boxed.push_back(
        RuntimeBundle::object(source->objectValue.contract, object->values));
    slowSources.push_back(&boxed.back());
  }
  // ⛔ Selected only now, from the boxed operands: an overload is chosen by
  // the operands' physical shapes, and a lane has none -- `1 < x` chose
  // int.__lt__(int) for a float x.
  mlir::FailureOr<RuntimeSymbol> selected =
      RuntimeBundleLowerer::selectManifestMethod(op, *slowSources.front(),
                                                 method, slowSources,
                                                 /*allowUnusedSources=*/false);
  if (mlir::failed(selected))
    return mlir::failure();
  llvm::SmallVector<mlir::Value, 8> operands;
  if (mlir::failed(RuntimeBundleLowerer::buildRuntimeCallOperands(
          op, *selected, slowSources, operands,
          /*allowUnusedSources=*/false)))
    return mlir::failure();
  builder.setInsertionPointToEnd(&ifOp.getElseRegion().front());
  mlir::func::CallOp call =
      RuntimeBundleLowerer::createRuntimeCall(loc, *selected, operands);
  mlir::Value slow;
  if (predicate) {
    if (call.getNumResults() != 1 || !call.getResult(0).getType().isInteger(1))
      return op->emitError() << "float comparison " << method
                             << " fallback must answer one i1";
    slow = call.getResult(0);
  } else {
    if (!unboxFloat || unboxFloat->function.getNumArguments() !=
                           call.getNumResults())
      return op->emitError() << "float " << method
                             << " fallback result cannot be read as a lane";
    slow = RuntimeBundleLowerer::createRuntimeCall(loc, *unboxFloat,
                                                   call.getResults())
               .getResult(0);
  }
  mlir::scf::YieldOp::create(builder, loc, mlir::ValueRange{slow});
  builder.setInsertionPointAfter(ifOp);
  return bindResult(ifOp.getResult(0), constantBool(builder, loc, true));
}

} // namespace py::lowering
