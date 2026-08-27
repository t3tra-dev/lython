#include "Runtime/Core/Lowerer.h"

#include "ArithBuilders.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ErrorHandling.h"

#include <limits>

namespace py::lowering {

namespace {

using lython::common::constantBool;
using lython::common::constantI64;
using lython::common::logicalAnd;
using lython::common::logicalNot;
using lython::common::SignedArith;
using lython::common::signedOverflow;

enum class PrimitiveI64ArithmeticKind {
  Add,
  Sub,
  Mul,
  FloorDiv,
  Mod,
  LShift,
  RShift,
  And,
  Or,
  Xor
};

std::optional<PrimitiveI64ArithmeticKind>
primitiveI64ArithmeticKind(llvm::StringRef methodName) {
  return llvm::StringSwitch<std::optional<PrimitiveI64ArithmeticKind>>(
             methodName)
      .Case("__add__", PrimitiveI64ArithmeticKind::Add)
      .Case("__sub__", PrimitiveI64ArithmeticKind::Sub)
      .Case("__mul__", PrimitiveI64ArithmeticKind::Mul)
      .Case("__floordiv__", PrimitiveI64ArithmeticKind::FloorDiv)
      .Case("__mod__", PrimitiveI64ArithmeticKind::Mod)
      .Case("__lshift__", PrimitiveI64ArithmeticKind::LShift)
      .Case("__rshift__", PrimitiveI64ArithmeticKind::RShift)
      .Case("__and__", PrimitiveI64ArithmeticKind::And)
      .Case("__or__", PrimitiveI64ArithmeticKind::Or)
      .Case("__xor__", PrimitiveI64ArithmeticKind::Xor)
      .Default(std::nullopt);
}

// `&`, `|` and `^` are the three that hand a BOOL back when both operands are
// bools (`True & True` is `True`, while `True + True` is `2` and `True // True`
// is `1`), and this path only produces `builtins.int`. Asked here rather than
// in the builder because the routing decision is made before a result contract
// exists.
bool primitiveI64ArithmeticKeepsBool(PrimitiveI64ArithmeticKind kind) {
  return kind == PrimitiveI64ArithmeticKind::And ||
         kind == PrimitiveI64ArithmeticKind::Or ||
         kind == PrimitiveI64ArithmeticKind::Xor;
}

std::optional<mlir::arith::CmpIPredicate>
primitiveI64ComparePredicate(llvm::StringRef methodName) {
  return llvm::StringSwitch<std::optional<mlir::arith::CmpIPredicate>>(
             methodName)
      .Case("__eq__", mlir::arith::CmpIPredicate::eq)
      .Case("__ne__", mlir::arith::CmpIPredicate::ne)
      .Case("__lt__", mlir::arith::CmpIPredicate::slt)
      .Case("__le__", mlir::arith::CmpIPredicate::sle)
      .Case("__gt__", mlir::arith::CmpIPredicate::sgt)
      .Case("__ge__", mlir::arith::CmpIPredicate::sge)
      .Default(std::nullopt);
}

std::pair<mlir::Value, mlir::Value>
buildPrimitiveI64Arithmetic(mlir::OpBuilder &builder, mlir::Location loc,
                            PrimitiveI64ArithmeticKind kind, mlir::Value lhs,
                            mlir::Value rhs) {
  switch (kind) {
  case PrimitiveI64ArithmeticKind::Add: {
    mlir::Value result =
        mlir::arith::AddIOp::create(builder, loc, lhs, rhs).getResult();
    return {result, signedOverflow(builder, loc, lhs, rhs, result,
                           builder.getI64Type(), SignedArith::Add)};
  }
  case PrimitiveI64ArithmeticKind::Sub: {
    mlir::Value result =
        mlir::arith::SubIOp::create(builder, loc, lhs, rhs).getResult();
    return {result, signedOverflow(builder, loc, lhs, rhs, result,
                           builder.getI64Type(), SignedArith::Subtract)};
  }
  case PrimitiveI64ArithmeticKind::Mul: {
    auto extended =
        mlir::arith::MulSIExtendedOp::create(builder, loc, lhs, rhs);
    mlir::Value shift = constantI64(builder, loc, 63);
    mlir::Value expectedHigh =
        mlir::arith::ShRSIOp::create(builder, loc, extended.getLow(), shift)
            .getResult();
    mlir::Value overflow = mlir::arith::CmpIOp::create(
                               builder, loc, mlir::arith::CmpIPredicate::ne,
                               extended.getHigh(), expectedHigh)
                               .getResult();
    return {extended.getLow(), overflow};
  }
  case PrimitiveI64ArithmeticKind::FloorDiv:
  case PrimitiveI64ArithmeticKind::Mod: {
    // ⛔ PYTHON DIVIDES TOWARD MINUS INFINITY AND LLVM TOWARD ZERO. `-7 // 2`
    // is -4 in Python and -3 for `arith.divsi`, and `-7 % 2` is 1 in Python
    // against -1 for `arith.remsi`: the remainder takes the DIVISOR's sign.
    // Correcting by one when the truncated remainder is non-zero and the signs
    // disagree is the identity CPython's `l_divmod` applies for the same
    // reason.
    mlir::Value zero = constantI64(builder, loc, 0);
    mlir::Value one = constantI64(builder, loc, 1);
    mlir::Value minusOne = constantI64(builder, loc, -1);
    mlir::Value intMin = constantI64(builder, loc,
                                     std::numeric_limits<std::int64_t>::min());
    mlir::Value byZero =
        mlir::arith::CmpIOp::create(builder, loc,
                                    mlir::arith::CmpIPredicate::eq, rhs, zero)
            .getResult();
    mlir::Value minOverMinusOne = mlir::arith::AndIOp::create(
        builder, loc,
        mlir::arith::CmpIOp::create(builder, loc,
                                    mlir::arith::CmpIPredicate::eq, lhs, intMin)
            .getResult(),
        mlir::arith::CmpIOp::create(builder, loc,
                                    mlir::arith::CmpIPredicate::eq, rhs,
                                    minusOne)
            .getResult());
    // The two pairs this cannot answer: a zero divisor, whose Python answer is
    // a ZeroDivisionError the boxed path raises, and INT64_MIN // -1, whose
    // quotient does not fit. Both are also the two that would TRAP, so the
    // division is given a benign divisor rather than guarded by a branch --
    // the clone must be able to run this as a rehearsal and observe nothing.
    mlir::Value refused =
        mlir::arith::OrIOp::create(builder, loc, byZero, minOverMinusOne)
            .getResult();
    mlir::Value divisor =
        mlir::arith::SelectOp::create(builder, loc, refused, one, rhs)
            .getResult();
    mlir::Value truncated =
        mlir::arith::DivSIOp::create(builder, loc, lhs, divisor).getResult();
    mlir::Value remainder =
        mlir::arith::RemSIOp::create(builder, loc, lhs, divisor).getResult();
    mlir::Value remainderNonZero =
        mlir::arith::CmpIOp::create(builder, loc,
                                    mlir::arith::CmpIPredicate::ne, remainder,
                                    zero)
            .getResult();
    mlir::Value signsDiffer = mlir::arith::CmpIOp::create(
        builder, loc, mlir::arith::CmpIPredicate::ne,
        mlir::arith::CmpIOp::create(builder, loc,
                                    mlir::arith::CmpIPredicate::slt, remainder,
                                    zero)
            .getResult(),
        mlir::arith::CmpIOp::create(builder, loc,
                                    mlir::arith::CmpIPredicate::slt, divisor,
                                    zero)
            .getResult())
                                  .getResult();
    mlir::Value adjust =
        mlir::arith::AndIOp::create(builder, loc, remainderNonZero, signsDiffer)
            .getResult();
    mlir::Value result;
    if (kind == PrimitiveI64ArithmeticKind::FloorDiv) {
      mlir::Value lowered =
          mlir::arith::SubIOp::create(builder, loc, truncated, one).getResult();
      result = mlir::arith::SelectOp::create(builder, loc, adjust, lowered,
                                             truncated)
                   .getResult();
    } else {
      mlir::Value shifted =
          mlir::arith::AddIOp::create(builder, loc, remainder, divisor)
              .getResult();
      result = mlir::arith::SelectOp::create(builder, loc, adjust, shifted,
                                             remainder)
                   .getResult();
    }
    return {result, refused};
  }
  case PrimitiveI64ArithmeticKind::LShift: {
    // A negative shift count is a ValueError in Python and undefined in LLVM,
    // and a count at or past the width is undefined too; both go to the boxed
    // path, which raises or widens as CPython does.
    mlir::Value zero = constantI64(builder, loc, 0);
    mlir::Value width = constantI64(builder, loc, 64);
    mlir::Value outOfRange = mlir::arith::OrIOp::create(
        builder, loc,
        mlir::arith::CmpIOp::create(builder, loc,
                                    mlir::arith::CmpIPredicate::slt, rhs, zero)
            .getResult(),
        mlir::arith::CmpIOp::create(builder, loc,
                                    mlir::arith::CmpIPredicate::sge, rhs, width)
            .getResult())
                                 .getResult();
    mlir::Value count =
        mlir::arith::SelectOp::create(builder, loc, outOfRange, zero, rhs)
            .getResult();
    mlir::Value shifted =
        mlir::arith::ShLIOp::create(builder, loc, lhs, count).getResult();
    // Python widens instead of wrapping, so a shift that loses bits is not an
    // answer: shifting back has to reproduce the operand.
    mlir::Value lost = mlir::arith::CmpIOp::create(
        builder, loc, mlir::arith::CmpIPredicate::ne,
        mlir::arith::ShRSIOp::create(builder, loc, shifted, count).getResult(),
        lhs)
                           .getResult();
    return {shifted,
            mlir::arith::OrIOp::create(builder, loc, outOfRange, lost)
                .getResult()};
  }
  case PrimitiveI64ArithmeticKind::RShift: {
    // Python's `>>` on a negative int floors, which is an ARITHMETIC shift, and
    // a count past the width saturates at the sign bit rather than being
    // undefined -- so the count is clamped and only a negative one is refused.
    mlir::Value zero = constantI64(builder, loc, 0);
    mlir::Value last = constantI64(builder, loc, 63);
    mlir::Value negative =
        mlir::arith::CmpIOp::create(builder, loc,
                                    mlir::arith::CmpIPredicate::slt, rhs, zero)
            .getResult();
    mlir::Value tooWide =
        mlir::arith::CmpIOp::create(builder, loc,
                                    mlir::arith::CmpIPredicate::sgt, rhs, last)
            .getResult();
    mlir::Value safeCount = mlir::arith::SelectOp::create(
        builder, loc,
        mlir::arith::OrIOp::create(builder, loc, negative, tooWide).getResult(),
        mlir::arith::SelectOp::create(builder, loc, negative, zero, last)
            .getResult(),
        rhs)
                                .getResult();
    return {mlir::arith::ShRSIOp::create(builder, loc, lhs, safeCount)
                .getResult(),
            negative};
  }
  case PrimitiveI64ArithmeticKind::And:
    return {mlir::arith::AndIOp::create(builder, loc, lhs, rhs).getResult(),
            constantBool(builder, loc, false)};
  case PrimitiveI64ArithmeticKind::Or:
    return {mlir::arith::OrIOp::create(builder, loc, lhs, rhs).getResult(),
            constantBool(builder, loc, false)};
  case PrimitiveI64ArithmeticKind::Xor:
    return {mlir::arith::XOrIOp::create(builder, loc, lhs, rhs).getResult(),
            constantBool(builder, loc, false)};
  }
  llvm_unreachable("unknown primitive i64 arithmetic kind");
}

} // namespace

mlir::FailureOr<RuntimePrimitiveI64Evidence>
RuntimeBundleLowerer::emitPrimitiveI64ArithmeticEvidence(
    mlir::Operation *op, llvm::StringRef methodName,
    const RuntimePrimitiveI64Evidence &lhs,
    const RuntimePrimitiveI64Evidence &rhs) {
  std::optional<PrimitiveI64ArithmeticKind> arithmetic =
      primitiveI64ArithmeticKind(methodName);
  if (!arithmetic)
    return op->emitError() << "unsupported primitive i64 arithmetic method "
                           << methodName;

  mlir::Location loc = op->getLoc();
  mlir::Value operandsValid = logicalAnd(builder, loc, lhs.valid, rhs.valid);
  auto [rawResult, overflow] = buildPrimitiveI64Arithmetic(
      builder, loc, *arithmetic, lhs.value, rhs.value);
  mlir::Value valid = logicalAnd(builder, loc, operandsValid,
                                 logicalNot(builder, loc, overflow));
  return RuntimePrimitiveI64Evidence{rawResult, valid};
}

mlir::LogicalResult RuntimeBundleLowerer::lowerPrimitiveI64BinarySpecial(
    mlir::Operation *op, llvm::StringRef methodName,
    llvm::ArrayRef<const RuntimeBundle *> sources, mlir::Value resultValue) {
  if (sources.size() != 2 ||
      !RuntimeBundleLowerer::hasPrimitiveI64Evidence(sources[0]) ||
      !RuntimeBundleLowerer::hasPrimitiveI64Evidence(sources[1]))
    return op->emitError()
           << "primitive i64 lowering requires two int operands with evidence";

  std::optional<PrimitiveI64ArithmeticKind> arithmetic =
      primitiveI64ArithmeticKind(methodName);
  std::optional<mlir::arith::CmpIPredicate> compare =
      primitiveI64ComparePredicate(methodName);
  if (!arithmetic && !compare)
    return op->emitError() << "unsupported primitive i64 special method "
                           << methodName;

  builder.setInsertionPoint(op);
  mlir::Location loc = op->getLoc();
  const RuntimePrimitiveI64Evidence &lhs = *sources[0]->primitiveI64;
  const RuntimePrimitiveI64Evidence &rhs = *sources[1]->primitiveI64;
  mlir::Value operandsValid = logicalAnd(builder, loc, lhs.valid, rhs.valid);

  if (RuntimeBundleLowerer::isPrimitiveI64CallableClone(
          op->getParentOfType<mlir::func::FuncOp>())) {
    if (arithmetic) {
      mlir::FailureOr<RuntimePrimitiveI64Evidence> fastEvidence =
          RuntimeBundleLowerer::emitPrimitiveI64ArithmeticEvidence(
              op, methodName, lhs, rhs);
      if (mlir::failed(fastEvidence))
        return mlir::failure();
      RuntimeBundle result;
      if (mlir::failed(RuntimeBundleLowerer::makePrimitiveI64Bundle(
              op, resultValue.getType(), fastEvidence->value,
              fastEvidence->valid, result)))
        return mlir::failure();
      valueBundles[resultValue] = std::move(result);
      return mlir::success();
    }

    mlir::Value compared = mlir::arith::CmpIOp::create(builder, loc, *compare,
                                                       lhs.value, rhs.value)
                               .getResult();
    mlir::Value fastResult = compared;
    if (!isPinnedTrueFlag(operandsValid)) {
      // A comparison answers i1: unlike arithmetic it has nowhere to carry
      // "the raw operands were not the true values" forward. WHY NOT just
      // and-ing the validity in: that maps "unknown" onto a definite `false`,
      // and the clone then BRANCHES on it. For `if n <= 1: return n` the false
      // arm is the recursive one, so an overflowed lane recursed until the
      // stack guard fired -- measured on fib(93), which hung.
      //
      // So the bit is parked in the clone's decision flag and AND-ed into
      // whatever the clone returns: the answer becomes "I cannot say", and
      // the call site takes the boxed original. The comparison itself still
      // reads false, which for the loop this shape usually is means "leave",
      // and no clone is ever ENTERED with a stale lane (the call guard), so
      // the wrong branch cannot recurse without end.
      RuntimeBundleLowerer::parkPrimitiveI64CloneDecision(op, operandsValid);
      fastResult = logicalAnd(builder, loc, operandsValid, compared);
    }
    RuntimeBundle result;
    if (mlir::failed(RuntimeBundleLowerer::makeObjectBundle(
            op, resultValue.getType(), mlir::ValueRange{fastResult}, result)))
      return mlir::failure();
    valueBundles[resultValue] = std::move(result);
    return mlir::success();
  }

  mlir::FailureOr<RuntimeSymbol> selected =
      RuntimeBundleLowerer::selectManifestMethod(op, *sources.front(),
                                                 methodName, sources,
                                                 /*allowUnusedSources=*/false);
  if (mlir::failed(selected))
    return mlir::failure();

  std::string resultContract = RuntimeBundleLowerer::resultContractFor(
      resultValue, *selected, /*preferManifestObjectResult=*/true);
  if (resultContract.empty())
    return op->emitError() << "primitive i64 " << methodName
                           << " result needs a concrete runtime contract";
  if (arithmetic && resultContract != "builtins.int")
    return op->emitError() << "primitive i64 arithmetic " << methodName
                           << " must produce builtins.int, got "
                           << resultContract;
  if (compare && resultContract != "builtins.bool")
    return op->emitError() << "primitive i64 comparison " << methodName
                           << " must produce builtins.bool, got "
                           << resultContract;

  mlir::Type resultType = runtimeContractType(context, resultContract);
  mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> resultTypes =
      RuntimeBundleLowerer::runtimeValueTypesFor(
          op, resultType, "primitive i64 guarded result ABI");
  if (mlir::failed(resultTypes))
    return mlir::failure();

  auto checkPhysicalTypes = [&](mlir::ValueRange values,
                                llvm::StringRef label) -> mlir::LogicalResult {
    if (values.size() != resultTypes->size())
      return op->emitError()
             << label << " produced " << values.size()
             << " values, but primitive guarded result ABI expects "
             << resultTypes->size();
    for (auto [index, value] : llvm::enumerate(values)) {
      if (value.getType() != (*resultTypes)[index])
        return op->emitError()
               << label << " result " << index << " has type "
               << value.getType() << ", but primitive guarded result ABI "
               << "expects " << (*resultTypes)[index];
    }
    return mlir::success();
  };

  context->loadDialect<mlir::scf::SCFDialect>();

  auto emitFallbackYield = [&]() -> mlir::LogicalResult {
    mlir::Block *fallbackBlock = builder.getInsertionBlock();
    builder.setInsertionPointToEnd(fallbackBlock);
    llvm::SmallVector<RuntimeBundle, 2> materializedSources;
    llvm::SmallVector<const RuntimeBundle *, 2> fallbackSources;
    materializedSources.reserve(sources.size());
    fallbackSources.reserve(sources.size());
    for (const RuntimeBundle *source : sources) {
      if (!source ||
          !RuntimeBundleLowerer::hasLazyPrimitiveI64Object(*source)) {
        fallbackSources.push_back(source);
        continue;
      }
      mlir::FailureOr<RuntimeValue> materialized =
          RuntimeBundleLowerer::materializePrimitiveI64ObjectAtCurrentInsertion(
              op, *source);
      if (mlir::failed(materialized))
        return mlir::failure();
      RuntimeBundle updated = *source;
      updated.contract = materialized->contract;
      updated.objectValue = *materialized;
      materializedSources.push_back(std::move(updated));
      fallbackSources.push_back(&materializedSources.back());
    }
    llvm::SmallVector<mlir::Value, 8> operands;
    if (mlir::failed(RuntimeBundleLowerer::buildRuntimeCallOperands(
            op, *selected, fallbackSources, operands,
            /*allowUnusedSources=*/false)))
      return mlir::failure();
    builder.setInsertionPointToEnd(fallbackBlock);
    mlir::func::CallOp call =
        RuntimeBundleLowerer::createRuntimeCall(loc, *selected, operands);
    if (mlir::failed(checkPhysicalTypes(call.getResults(), "fallback call")))
      return mlir::failure();
    mlir::scf::YieldOp::create(builder, loc, call.getResults());
    return mlir::success();
  };

  if (arithmetic) {
    mlir::FailureOr<RuntimePrimitiveI64Evidence> fastEvidence =
        RuntimeBundleLowerer::emitPrimitiveI64ArithmeticEvidence(
            op, methodName, lhs, rhs);
    if (mlir::failed(fastEvidence))
      return mlir::failure();
    mlir::Value rawResult = fastEvidence->value;
    mlir::Value fastValid = fastEvidence->valid;
    auto ifOp = mlir::scf::IfOp::create(builder, loc, *resultTypes, fastValid,
                                        /*withElseRegion=*/true);

    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
    RuntimeBundle fastBundle;
    if (mlir::failed(RuntimeBundleLowerer::initializeObjectFromRawValues(
            op, resultType, mlir::ValueRange{rawResult}, fastBundle)))
      return mlir::failure();
    if (mlir::failed(checkPhysicalTypes(fastBundle.physicalValues(),
                                        "primitive i64 fast path")))
      return mlir::failure();
    mlir::scf::YieldOp::create(builder, loc, fastBundle.physicalValues());

    builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
    if (mlir::failed(emitFallbackYield()))
      return mlir::failure();

    builder.setInsertionPointAfter(ifOp);
    RuntimeBundle result;
    if (mlir::failed(RuntimeBundleLowerer::bundleRuntimeResults(
            op, resultType, ifOp.getResults(), result)))
      return mlir::failure();
    result.primitiveI64 = RuntimePrimitiveI64Evidence{rawResult, fastValid};
    valueBundles[resultValue] = std::move(result);
    return mlir::success();
  }

  if (resultTypes->size() != 1 || !resultTypes->front().isInteger(1))
    return op->emitError() << "primitive i64 comparison " << methodName
                           << " expects a single i1 bool ABI result";
  mlir::Value fastResult =
      mlir::arith::CmpIOp::create(builder, loc, *compare, lhs.value, rhs.value)
          .getResult();
  if (isPinnedTrueFlag(operandsValid)) {
    RuntimeBundle result;
    if (mlir::failed(RuntimeBundleLowerer::makeObjectBundle(
            op, resultType, mlir::ValueRange{fastResult}, result)))
      return mlir::failure();
    valueBundles[resultValue] = std::move(result);
    return mlir::success();
  }
  auto ifOp = mlir::scf::IfOp::create(builder, loc, *resultTypes, operandsValid,
                                      /*withElseRegion=*/true);

  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  mlir::scf::YieldOp::create(builder, loc, mlir::ValueRange{fastResult});

  builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
  if (mlir::failed(emitFallbackYield()))
    return mlir::failure();

  builder.setInsertionPointAfter(ifOp);
  RuntimeBundle result;
  if (mlir::failed(RuntimeBundleLowerer::bundleRuntimeResults(
          op, resultType, ifOp.getResults(), result)))
    return mlir::failure();
  valueBundles[resultValue] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerBinarySpecial(
    mlir::Operation *op, mlir::Value lhs, mlir::Value rhs,
    llvm::StringRef methodName, mlir::Value resultValue) {
  llvm::SmallVector<mlir::Value, 2> inputs{lhs, rhs};
  llvm::SmallVector<const RuntimeBundle *, 2> sources;
  if (mlir::failed(collectObjectSources(
          op, inputs, "binary special method operands need runtime bundles",
          sources)))
    return mlir::failure();
  std::optional<PrimitiveI64ArithmeticKind> primitiveArithmetic =
      primitiveI64ArithmeticKind(methodName);
  // A bool-preserving bitwise op is only routed here when neither operand is a
  // bool, because this path answers `builtins.int` and `True & True` is `True`.
  bool boolOperand =
      sources.size() == 2 && (sources[0]->contractName() == "builtins.bool" ||
                              sources[1]->contractName() == "builtins.bool");
  if (primitiveArithmetic && boolOperand &&
      primitiveI64ArithmeticKeepsBool(*primitiveArithmetic))
    primitiveArithmetic.reset();
  if (sources.size() == 2 &&
      RuntimeBundleLowerer::hasPrimitiveI64Evidence(sources[0]) &&
      RuntimeBundleLowerer::hasPrimitiveI64Evidence(sources[1]) &&
      (primitiveArithmetic || primitiveI64ComparePredicate(methodName))) {
    if (mlir::failed(RuntimeBundleLowerer::lowerPrimitiveI64BinarySpecial(
            op, methodName, sources, resultValue)))
      return mlir::failure();
    erase.push_back(op);
    return mlir::success();
  }
  if (methodName == "__mul__" && sources.size() == 2)
    if (mlir::succeeded(RuntimeBundleLowerer::lowerStaticCtypesArrayTypeMul(
            op, *sources[0], *sources[1], resultValue))) {
      erase.push_back(op);
      return mlir::success();
    }
  if (mlir::failed(lowerManifestMethodResult(
          op, resultValue, *sources.front(), methodName, sources,
          /*allowUnusedSources=*/false,
          /*preferManifestObjectResult=*/true)))
    return mlir::failure();
  erase.push_back(op);
  return mlir::success();
}

} // namespace py::lowering
