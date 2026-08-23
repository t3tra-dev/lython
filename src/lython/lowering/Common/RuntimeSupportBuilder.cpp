#include "Common/RuntimeSupportBuilder.h"

#include "Common/SupportBuilder.h"
#include "ExceptionTaxonomy.h"
#include "Runtime/ABI/BoxLayout.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"

#include <limits>

namespace py::runtime_library {
namespace {


// double LyFloat_RoundToI64(double x): round-half-to-even then narrow to i64,
// trapping on NaN/inf and out-of-range results.
void buildFloatRoundToI64(SupportBuilder &b) {
  auto fn = b.beginFunction(
      "LyFloat_RoundToI64", b.builder.getFunctionType({b.f64()}, {b.i64()}));
  mlir::Block *entry = fn.addEntryBlock();
  mlir::Region &body = fn.getBody();
  mlir::Value x = entry->getArgument(0);

  mlir::Block *okBlock = b.builder.createBlock(&body);
  mlir::Block *convBlock = b.builder.createBlock(&body);
  mlir::Block *trapBlock = b.builder.createBlock(&body);

  b.builder.setInsertionPointToEnd(entry);
  mlir::Value inf =
      b.fconst(std::numeric_limits<double>::infinity());
  mlir::Value isNan = b.cmpf(mlir::arith::CmpFPredicate::UNO, x, x);
  mlir::Value absX = mlir::math::AbsFOp::create(b.builder, b.loc, x);
  mlir::Value isInf = b.cmpf(mlir::arith::CmpFPredicate::OEQ, absX, inf);
  mlir::Value bad = b.orBit(isNan, isInf);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, bad, trapBlock,
                                 mlir::ValueRange{}, okBlock,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(okBlock);
  mlir::Value rounded = mlir::math::RoundEvenOp::create(b.builder, b.loc, x);
  // -2^63 and 2^63 as the representable i64 window.
  mlir::Value lo = b.fconst(-9223372036854775808.0);
  mlir::Value hi = b.fconst(9223372036854775808.0);
  mlir::Value tooLow = b.cmpf(mlir::arith::CmpFPredicate::OLT, rounded, lo);
  mlir::Value tooHigh = b.cmpf(mlir::arith::CmpFPredicate::OGE, rounded, hi);
  mlir::Value outOfRange = b.orBit(tooLow, tooHigh);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, outOfRange, trapBlock,
                                 mlir::ValueRange{}, convBlock,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(convBlock);
  mlir::Value narrowed =
      mlir::arith::FPToSIOp::create(b.builder, b.loc, b.i64(), rounded);
  mlir::func::ReturnOp::create(b.builder, b.loc, narrowed);

  b.builder.setInsertionPointToEnd(trapBlock);
  b.emitTrap(b.i64());
}

// double LyFloat_Round(double x, i64 ndigits): round x to ndigits decimal
// places (round-half-to-even), passing NaN/inf and extreme exponents through.
void buildFloatRound(SupportBuilder &b) {
  auto fn = b.beginFunction(
      "LyFloat_Round",
      b.builder.getFunctionType({b.f64(), b.i64()}, {b.f64()}));
  mlir::Block *entry = fn.addEntryBlock();
  mlir::Region &body = fn.getBody();
  mlir::Value x = entry->getArgument(0);
  mlir::Value ndigits = entry->getArgument(1);

  mlir::Block *rangeBlock = b.builder.createBlock(&body);
  mlir::Block *scaleBlock = b.builder.createBlock(&body);
  mlir::Block *dirBlock = b.builder.createBlock(&body);
  mlir::Block *upBlock = b.builder.createBlock(&body);
  mlir::Block *downBlock = b.builder.createBlock(&body);
  mlir::Block *passBlock = b.builder.createBlock(&body);
  mlir::Block *trapBlock = b.builder.createBlock(&body);

  b.builder.setInsertionPointToEnd(entry);
  mlir::Value inf = b.fconst(std::numeric_limits<double>::infinity());
  mlir::Value isNan = b.cmpf(mlir::arith::CmpFPredicate::UNO, x, x);
  mlir::Value absX = mlir::math::AbsFOp::create(b.builder, b.loc, x);
  mlir::Value isInf = b.cmpf(mlir::arith::CmpFPredicate::OEQ, absX, inf);
  mlir::Value special = b.orBit(isNan, isInf);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, special, trapBlock,
                                 mlir::ValueRange{}, rangeBlock,
                                 mlir::ValueRange{});

  // |ndigits| beyond the double exponent range leaves x unchanged.
  b.builder.setInsertionPointToEnd(rangeBlock);
  mlir::Value hiDigits = b.iconst(308);
  mlir::Value loDigits = b.iconst(-308);
  mlir::Value aboveHi =
      b.cmpi(mlir::arith::CmpIPredicate::sgt, ndigits, hiDigits);
  mlir::Value belowLo =
      b.cmpi(mlir::arith::CmpIPredicate::slt, ndigits, loDigits);
  mlir::Value extreme = b.orBit(aboveHi, belowLo);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, extreme, passBlock,
                                 mlir::ValueRange{}, scaleBlock,
                                 mlir::ValueRange{});

  // scale = 10 ** |ndigits|.
  b.builder.setInsertionPointToEnd(scaleBlock);
  mlir::Value zero64 = b.iconst(0);
  mlir::Value ten = b.fconst(10.0);
  mlir::Value negDigits =
      b.cmpi(mlir::arith::CmpIPredicate::slt, ndigits, zero64);
  mlir::Value absDigitsInt = mlir::arith::SubIOp::create(
      b.builder, b.loc, zero64, ndigits);
  mlir::Value magnitude = mlir::arith::SelectOp::create(
      b.builder, b.loc, negDigits, absDigitsInt, ndigits);
  mlir::Value magnitudeF =
      mlir::arith::UIToFPOp::create(b.builder, b.loc, b.f64(), magnitude);
  mlir::Value scale =
      mlir::math::PowFOp::create(b.builder, b.loc, ten, magnitudeF);
  mlir::Value scaleNan = b.cmpf(mlir::arith::CmpFPredicate::UNO, scale, scale);
  mlir::Value scaleAbs = mlir::math::AbsFOp::create(b.builder, b.loc, scale);
  mlir::Value scaleInf = b.cmpf(mlir::arith::CmpFPredicate::OEQ, scaleAbs, inf);
  mlir::Value zeroF = b.fconst(0.0);
  mlir::Value scaleZero = b.cmpf(mlir::arith::CmpFPredicate::OEQ, scale, zeroF);
  mlir::Value scaleBad = b.orBit(b.orBit(scaleNan, scaleInf), scaleZero);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, scaleBad, trapBlock,
                                 mlir::ValueRange{}, dirBlock,
                                 mlir::ValueRange{});

  // ndigits > -1 multiplies before rounding; otherwise divides.
  b.builder.setInsertionPointToEnd(dirBlock);
  mlir::Value negOne = b.iconst(-1);
  mlir::Value multiplyFirst =
      b.cmpi(mlir::arith::CmpIPredicate::sgt, ndigits, negOne);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, multiplyFirst, upBlock,
                                 mlir::ValueRange{}, downBlock,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(upBlock);
  mlir::Value scaledUp = mlir::arith::MulFOp::create(b.builder, b.loc, x, scale);
  mlir::Value roundedUp =
      mlir::math::RoundEvenOp::create(b.builder, b.loc, scaledUp);
  mlir::Value resultUp =
      mlir::arith::DivFOp::create(b.builder, b.loc, roundedUp, scale);
  mlir::func::ReturnOp::create(b.builder, b.loc, resultUp);

  b.builder.setInsertionPointToEnd(downBlock);
  mlir::Value scaledDown =
      mlir::arith::DivFOp::create(b.builder, b.loc, x, scale);
  mlir::Value roundedDown =
      mlir::math::RoundEvenOp::create(b.builder, b.loc, scaledDown);
  mlir::Value resultDown =
      mlir::arith::MulFOp::create(b.builder, b.loc, roundedDown, scale);
  mlir::func::ReturnOp::create(b.builder, b.loc, resultDown);

  b.builder.setInsertionPointToEnd(passBlock);
  mlir::func::ReturnOp::create(b.builder, b.loc, x);

  b.builder.setInsertionPointToEnd(trapBlock);
  b.emitTrap(b.f64());
}

// i64 LyInt_Round(i64 value, i64 ndigits): CPython `round(int, ndigits)` — a
// no-op for ndigits >= 0, otherwise round-half-to-even to the 10**(-ndigits)
// place, trapping on i64 overflow.
void buildIntRound(SupportBuilder &b) {
  auto fn = b.beginFunction(
      "LyInt_Round",
      b.builder.getFunctionType({b.i64(), b.i64()}, {b.i64()}));
  mlir::Block *entry = fn.addEntryBlock();
  mlir::Region &body = fn.getBody();
  mlir::Value value = entry->getArgument(0);
  mlir::Value ndigits = entry->getArgument(1);

  mlir::Block *negBlock = b.builder.createBlock(&body);
  mlir::Block *loopHeader = b.builder.createBlock(&body);
  loopHeader->addArgument(b.i64(), b.loc); // exponent counter
  loopHeader->addArgument(b.i64(), b.loc); // running power of ten
  mlir::Block *loopCheck = b.builder.createBlock(&body);
  mlir::Block *loopStep = b.builder.createBlock(&body);
  mlir::Block *roundBlock = b.builder.createBlock(&body);
  mlir::Block *posBlock = b.builder.createBlock(&body);
  mlir::Block *posReturn = b.builder.createBlock(&body);
  mlir::Block *negCheck = b.builder.createBlock(&body);
  mlir::Block *negOverflow = b.builder.createBlock(&body);
  mlir::Block *negReturn = b.builder.createBlock(&body);
  mlir::Block *minBlock = b.builder.createBlock(&body);
  mlir::Block *zeroBlock = b.builder.createBlock(&body);
  mlir::Block *passBlock = b.builder.createBlock(&body);
  mlir::Block *trapBlock = b.builder.createBlock(&body);

  mlir::Value negOne = b.iconst(-1);
  mlir::Value zero = b.iconst(0);
  mlir::Value one = b.iconst(1);
  mlir::Value ten = b.iconst(10);
  mlir::Value nineteen = b.iconst(19);
  mlir::Value powLimit = b.iconst(1844674407370955161); // floor(UINT64_MAX/10)
  mlir::Value intMin = b.iconst(-9223372036854775807LL - 1);
  mlir::Value intMax = b.iconst(9223372036854775807LL);

  b.builder.setInsertionPointToEnd(entry);
  mlir::Value nonNegative =
      b.cmpi(mlir::arith::CmpIPredicate::sgt, ndigits, negOne);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, nonNegative, passBlock,
                                 mlir::ValueRange{}, negBlock,
                                 mlir::ValueRange{});

  // places = -ndigits; more than 19 digits rounds everything to zero.
  b.builder.setInsertionPointToEnd(negBlock);
  mlir::Value places = mlir::arith::SubIOp::create(b.builder, b.loc, zero,
                                                   ndigits);
  mlir::Value tooManyPlaces =
      b.cmpi(mlir::arith::CmpIPredicate::ugt, places, nineteen);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, tooManyPlaces, zeroBlock,
                                 mlir::ValueRange{}, loopHeader,
                                 mlir::ValueRange{zero, one});

  // Accumulate divisor = 10**places, trapping if it would overflow u64.
  b.builder.setInsertionPointToEnd(loopHeader);
  mlir::Value counter = loopHeader->getArgument(0);
  mlir::Value power = loopHeader->getArgument(1);
  mlir::Value reached = b.cmpi(mlir::arith::CmpIPredicate::eq, counter, places);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, reached, roundBlock,
                                 mlir::ValueRange{}, loopCheck,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(loopCheck);
  mlir::Value powerHuge =
      b.cmpi(mlir::arith::CmpIPredicate::ugt, power, powLimit);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, powerHuge, zeroBlock,
                                 mlir::ValueRange{}, loopStep,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(loopStep);
  mlir::Value nextPower = mlir::arith::MulIOp::create(b.builder, b.loc, power,
                                                      ten);
  mlir::Value nextCounter =
      mlir::arith::AddIOp::create(b.builder, b.loc, counter, one);
  mlir::cf::BranchOp::create(b.builder, b.loc, loopHeader,
                             mlir::ValueRange{nextCounter, nextPower});

  // round-half-to-even of |value| / divisor, then restore the sign.
  b.builder.setInsertionPointToEnd(roundBlock);
  mlir::Value divisor = power;
  mlir::Value positive =
      b.cmpi(mlir::arith::CmpIPredicate::sgt, value, negOne);
  mlir::Value valuePlusOne =
      mlir::arith::AddIOp::create(b.builder, b.loc, value, one);
  mlir::Value negated =
      mlir::arith::SubIOp::create(b.builder, b.loc, zero, valuePlusOne);
  mlir::Value magnitude =
      mlir::arith::AddIOp::create(b.builder, b.loc, negated, one);
  mlir::Value absValue = mlir::arith::SelectOp::create(b.builder, b.loc,
                                                       positive, value,
                                                       magnitude);
  mlir::Value quotient =
      mlir::arith::DivUIOp::create(b.builder, b.loc, absValue, divisor);
  mlir::Value remainder =
      mlir::arith::RemUIOp::create(b.builder, b.loc, absValue, divisor);
  mlir::Value complement =
      mlir::arith::SubIOp::create(b.builder, b.loc, divisor, remainder);
  mlir::Value remGreater =
      b.cmpi(mlir::arith::CmpIPredicate::ugt, remainder, complement);
  mlir::Value halfway =
      b.cmpi(mlir::arith::CmpIPredicate::eq, remainder, complement);
  mlir::Value quotientOdd =
      mlir::arith::AndIOp::create(b.builder, b.loc, quotient, one);
  mlir::Value isOdd = b.cmpi(mlir::arith::CmpIPredicate::ne, quotientOdd, zero);
  mlir::Value halfwayToEven =
      mlir::arith::AndIOp::create(b.builder, b.loc, halfway, isOdd);
  mlir::Value roundUp = b.orBit(remGreater, halfwayToEven);
  mlir::Value roundUpInt =
      mlir::arith::ExtUIOp::create(b.builder, b.loc, b.i64(), roundUp);
  mlir::Value roundedQuotient =
      mlir::arith::AddIOp::create(b.builder, b.loc, quotient, roundUpInt);
  mlir::Value scaled = mlir::arith::MulIOp::create(b.builder, b.loc,
                                                   roundedQuotient, divisor);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, positive, posBlock,
                                 mlir::ValueRange{}, negCheck,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(posBlock);
  mlir::Value posOverflow =
      b.cmpi(mlir::arith::CmpIPredicate::ugt, scaled, intMax);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, posOverflow, trapBlock,
                                 mlir::ValueRange{}, posReturn,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(posReturn);
  mlir::func::ReturnOp::create(b.builder, b.loc, scaled);

  b.builder.setInsertionPointToEnd(negCheck);
  mlir::Value isIntMin = b.cmpi(mlir::arith::CmpIPredicate::eq, scaled, intMin);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, isIntMin, minBlock,
                                 mlir::ValueRange{}, negOverflow,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(negOverflow);
  mlir::Value negTooBig =
      b.cmpi(mlir::arith::CmpIPredicate::ugt, scaled, intMax);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, negTooBig, trapBlock,
                                 mlir::ValueRange{}, negReturn,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(negReturn);
  mlir::Value negatedResult =
      mlir::arith::SubIOp::create(b.builder, b.loc, zero, scaled);
  mlir::func::ReturnOp::create(b.builder, b.loc, negatedResult);

  b.builder.setInsertionPointToEnd(minBlock);
  mlir::func::ReturnOp::create(b.builder, b.loc, intMin);

  b.builder.setInsertionPointToEnd(zeroBlock);
  mlir::func::ReturnOp::create(b.builder, b.loc, zero);

  b.builder.setInsertionPointToEnd(passBlock);
  mlir::func::ReturnOp::create(b.builder, b.loc, value);

  b.builder.setInsertionPointToEnd(trapBlock);
  b.emitTrap(b.i64());
}

// i64 exception_base_class_id(i64 class_id): one step up the builtin exception
// hierarchy (class id -> its base class id), 0 for a root/unknown id. Pure
// `cf.switch` over the fixed builtin exception class-id table; ids outside it
// consult the per-program user-exception hook (source class ids live above
// 2^32 and cannot be known when this module is built).
void buildExceptionBaseClassId(SupportBuilder &b) {
  auto fn = b.beginFunction("exception_base_class_id",
                            b.builder.getFunctionType({b.i64()}, {b.i64()}),
                            /*isPrivate=*/true);
  mlir::Block *entry = fn.addEntryBlock();
  mlir::Region &body = fn.getBody();
  mlir::Value classId = entry->getArgument(0);

  // One return block per distinct base class id in the shared taxonomy;
  // unknown ids (the switch default) ask the user-exception hook, which
  // returns the root (0) for anything it does not own.
  llvm::SmallDenseMap<std::int64_t, mlir::Block *, 8> returnBlocks;
  auto returnBlockFor = [&](std::int64_t value) {
    mlir::Block *&block = returnBlocks[value];
    if (!block) {
      block = b.builder.createBlock(&body);
      b.builder.setInsertionPointToEnd(block);
      mlir::func::ReturnOp::create(b.builder, b.loc, b.iconst(value));
    }
    return block;
  };
  mlir::Block *toRoot;
  {
    toRoot = b.builder.createBlock(&body);
    b.builder.setInsertionPointToEnd(toRoot);
    auto userBase = mlir::func::CallOp::create(
        b.builder, b.loc, "__ly_user_exception_base_class_id", b.i64(),
        mlir::ValueRange{classId});
    mlir::func::ReturnOp::create(b.builder, b.loc, userBase.getResult(0));
  }

  llvm::SmallVector<llvm::APInt, 16> caseValues;
  llvm::SmallVector<mlir::Block *, 16> caseDests;
  llvm::SmallVector<mlir::ValueRange, 16> caseOperands;
  for (const py::exceptions::BuiltinExceptionInfo &info :
       py::exceptions::kBuiltinExceptions) {
    if (info.baseClassId == py::exceptions::kRootClassId)
      continue;
    caseValues.emplace_back(64, static_cast<std::uint64_t>(info.classId));
    caseDests.push_back(returnBlockFor(info.baseClassId));
    caseOperands.push_back(mlir::ValueRange{});
  }

  b.builder.setInsertionPointToEnd(entry);
  mlir::cf::SwitchOp::create(b.builder, b.loc, classId, toRoot,
                             mlir::ValueRange{}, caseValues, caseDests,
                             caseOperands);
}

// i1 LyEH_ClassIdMatches(i64 raised, i64 handler): whether a raised exception's
// class id is `handler` or a subclass of it, by walking base class ids up to
// the root. Pure `cf` loop.
void buildEHClassIdMatches(SupportBuilder &b) {
  auto fn = b.beginFunction(
      "LyEH_ClassIdMatches",
      b.builder.getFunctionType({b.i64(), b.i64()}, {b.i1()}));
  mlir::Block *entry = fn.addEntryBlock();
  mlir::Region &body = fn.getBody();
  mlir::Value raised = entry->getArgument(0);
  mlir::Value handler = entry->getArgument(1);

  mlir::Block *loop = b.builder.createBlock(&body);
  loop->addArgument(b.i64(), b.loc); // current class id
  mlir::Block *checkHandler = b.builder.createBlock(&body);
  mlir::Block *stepUp = b.builder.createBlock(&body);
  mlir::Block *matched = b.builder.createBlock(&body);
  mlir::Block *exhausted = b.builder.createBlock(&body);

  b.builder.setInsertionPointToEnd(entry);
  mlir::cf::BranchOp::create(b.builder, b.loc, loop, mlir::ValueRange{raised});

  b.builder.setInsertionPointToEnd(loop);
  mlir::Value current = loop->getArgument(0);
  mlir::Value isRoot =
      b.cmpi(mlir::arith::CmpIPredicate::eq, current, b.iconst(0));
  mlir::cf::CondBranchOp::create(b.builder, b.loc, isRoot, exhausted,
                                 mlir::ValueRange{}, checkHandler,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(checkHandler);
  mlir::Value hit =
      b.cmpi(mlir::arith::CmpIPredicate::eq, current, handler);
  // Multiple-inheritance extra edges (ExceptionGroup -> Exception): the
  // chain walk only follows the primary base, so accept `handler` being any
  // ancestor of an extra base when the walk passes through the edge's owner.
  for (const py::exceptions::BuiltinExceptionExtraEdge &edge :
       py::exceptions::kBuiltinExceptionExtraEdges) {
    mlir::Value atEdgeOwner =
        b.cmpi(mlir::arith::CmpIPredicate::eq, current, b.iconst(edge.classId));
    std::int64_t ancestor = edge.extraBaseClassId;
    while (ancestor != py::exceptions::kRootClassId) {
      mlir::Value handlerIsAncestor = b.cmpi(mlir::arith::CmpIPredicate::eq,
                                             handler, b.iconst(ancestor));
      mlir::Value viaEdge = mlir::arith::AndIOp::create(
          b.builder, b.loc, atEdgeOwner, handlerIsAncestor);
      hit = mlir::arith::OrIOp::create(b.builder, b.loc, hit, viaEdge);
      const py::exceptions::BuiltinExceptionInfo *info =
          py::exceptions::findByClassId(ancestor);
      ancestor = info ? info->baseClassId : py::exceptions::kRootClassId;
    }
  }
  mlir::cf::CondBranchOp::create(b.builder, b.loc, hit, matched,
                                 mlir::ValueRange{}, stepUp,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(stepUp);
  auto base = mlir::func::CallOp::create(b.builder, b.loc,
                                         "exception_base_class_id", b.i64(),
                                         mlir::ValueRange{current});
  mlir::cf::BranchOp::create(b.builder, b.loc, loop,
                             mlir::ValueRange{base.getResult(0)});

  b.builder.setInsertionPointToEnd(matched);
  mlir::Value trueVal =
      mlir::arith::ConstantIntOp::create(b.builder, b.loc, b.i1(), 1);
  mlir::func::ReturnOp::create(b.builder, b.loc, trueVal);

  b.builder.setInsertionPointToEnd(exhausted);
  mlir::Value falseVal =
      mlir::arith::ConstantIntOp::create(b.builder, b.loc, b.i1(), 0);
  mlir::func::ReturnOp::create(b.builder, b.loc, falseVal);
}

// i1 raw_bytes_equal(i64 p1, i64 n1, i64 p2, i64 n2): byte-equality of two raw
// buffers. Control/logic in scf/arith; the raw byte loads are the irreducible
// pointer part (llvm dialect). Shared with builtins.mlir (dict key compare).
void buildRawBytesEqual(SupportBuilder &b) {
  auto fn = b.beginFunction(
      "raw_bytes_equal",
      b.builder.getFunctionType({b.i64(), b.i64(), b.i64(), b.i64()},
                                {b.i1()}));
  mlir::Block *entry = fn.addEntryBlock();
  mlir::Region &body = fn.getBody();
  mlir::Value p1 = entry->getArgument(0);
  mlir::Value n1 = entry->getArgument(1);
  mlir::Value p2 = entry->getArgument(2);
  mlir::Value n2 = entry->getArgument(3);

  mlir::Block *scan = b.builder.createBlock(&body);
  mlir::Block *unequal = b.builder.createBlock(&body);

  b.builder.setInsertionPointToEnd(entry);
  mlir::Value sameLen = b.cmpi(mlir::arith::CmpIPredicate::eq, n1, n2);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, sameLen, scan,
                                 mlir::ValueRange{}, unequal,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(scan);
  auto ptrType = mlir::LLVM::LLVMPointerType::get(b.builder.getContext());
  auto i8 = b.builder.getIntegerType(8);
  mlir::Value a = mlir::LLVM::IntToPtrOp::create(b.builder, b.loc, ptrType, p1);
  mlir::Value bPtr =
      mlir::LLVM::IntToPtrOp::create(b.builder, b.loc, ptrType, p2);
  mlir::Value zeroIdx = mlir::arith::ConstantIndexOp::create(b.builder, b.loc, 0);
  mlir::Value oneIdx = mlir::arith::ConstantIndexOp::create(b.builder, b.loc, 1);
  mlir::Value count =
      mlir::arith::IndexCastOp::create(b.builder, b.loc, b.builder.getIndexType(),
                                       n1);
  mlir::Value trueVal =
      mlir::arith::ConstantIntOp::create(b.builder, b.loc, b.i1(), 1);
  auto loop = mlir::scf::ForOp::create(
      b.builder, b.loc, zeroIdx, count, oneIdx, mlir::ValueRange{trueVal},
      [&](mlir::OpBuilder &nested, mlir::Location nestedLoc, mlir::Value iv,
          mlir::ValueRange iter) {
        mlir::Value index = mlir::arith::IndexCastOp::create(
            nested, nestedLoc, b.i64(), iv);
        mlir::Value pa = mlir::LLVM::GEPOp::create(
            nested, nestedLoc, ptrType, i8, a, mlir::ValueRange{index});
        mlir::Value pb = mlir::LLVM::GEPOp::create(
            nested, nestedLoc, ptrType, i8, bPtr, mlir::ValueRange{index});
        mlir::Value va = mlir::LLVM::LoadOp::create(nested, nestedLoc, i8, pa,
                                                    /*alignment=*/1);
        mlir::Value vb = mlir::LLVM::LoadOp::create(nested, nestedLoc, i8, pb,
                                                    /*alignment=*/1);
        mlir::Value equalByte = mlir::arith::CmpIOp::create(
            nested, nestedLoc, mlir::arith::CmpIPredicate::eq, va, vb);
        mlir::Value next = mlir::arith::AndIOp::create(nested, nestedLoc,
                                                       iter.front(), equalByte);
        mlir::scf::YieldOp::create(nested, nestedLoc, mlir::ValueRange{next});
      });
  mlir::func::ReturnOp::create(b.builder, b.loc, loop.getResult(0));

  b.builder.setInsertionPointToEnd(unequal);
  mlir::Value falseVal =
      mlir::arith::ConstantIntOp::create(b.builder, b.loc, b.i1(), 0);
  mlir::func::ReturnOp::create(b.builder, b.loc, falseVal);
}

// ---------------------------------------------------------------------------
// Boxed-object deallocation (bottom slice). Raw address / pointer plumbing, so
// these use the llvm dialect directly; control/logic stays in arith/cf/scf.
// ---------------------------------------------------------------------------

// !llvm.ptr boxed_slot_ptr(!llvm.ptr base, i64 slot): &base[slot] (i64 units).
void buildBoxedSlotPtr(SupportBuilder &b) {
  auto fn = b.beginFunction(
      "boxed_slot_ptr", b.builder.getFunctionType({b.ptr(), b.i64()}, {b.ptr()}),
      /*isPrivate=*/true);
  mlir::Block *entry = fn.addEntryBlock();
  b.builder.setInsertionPointToEnd(entry);
  mlir::Value slot =
      b.gepI64(entry->getArgument(0), entry->getArgument(1));
  mlir::func::ReturnOp::create(b.builder, b.loc, slot);
}

// i64 boxed_load_i64(!llvm.ptr base, i64 slot): load base[slot].
void buildBoxedLoadI64(SupportBuilder &b) {
  auto fn = b.beginFunction(
      "boxed_load_i64", b.builder.getFunctionType({b.ptr(), b.i64()}, {b.i64()}));
  mlir::Block *entry = fn.addEntryBlock();
  b.builder.setInsertionPointToEnd(entry);
  auto slot = mlir::func::CallOp::create(
      b.builder, b.loc, "boxed_slot_ptr", b.ptr(),
      mlir::ValueRange{entry->getArgument(0), entry->getArgument(1)});
  mlir::func::ReturnOp::create(b.builder, b.loc, b.loadI64(slot.getResult(0)));
}

// ⭐ CPython's PyObject_Malloc (Objects/obmalloc.c), which is what every Python
// object goes through there and what this runtime did NOT have: `memref.alloc`
// lowers to a bare `malloc`, so a loop that boxes an int paid the system
// allocator twice per iteration. Measured before writing this: 55-70% of the
// time in the container benchmarks was malloc, free and the zeroing they do.
//
// The port keeps obmalloc's shape -- a size class per 16 bytes up to a small
// threshold, a free list per class, blocks carved from arenas taken from the
// system allocator in bulk -- and drops two things it does not need:
//
// ⛔ NO ARENA RELEASE. CPython returns an empty arena to the system; this does
// not, so peak RSS is a high-water mark.
//
// It was built and measured before being left out. The shape that returns
// memory is obmalloc's pool -- an aligned span carved into blocks of ONE size
// class, holding its own free list and a live count, released the moment its
// last block dies -- because a free list spanning every pool can return none of
// them. Ported here (16 KB pools, `aligned_alloc`, the free head's null doing
// double duty as the "off the class list" flag, the cold paths outlined so
// `LyMem_Alloc` still inlines) it cost 10-30% on the container benchmarks
// (b_class 37.0 -> 48.0 ms, b_tuple2 49.6 -> 61.0, b_smalllist 52.1 -> 62.0,
// 2M-list 112.8 -> 125.9) for the live count alone: three memory ops on a path
// that is five, and this allocator is fast enough that three is a third of it.
//
// ⭐ AND IT RETURNED ALMOST NOTHING, which is the actual reason. Measured on a
// program that builds 20,000 two-key dicts, drops them, and keeps running:
// resident memory afterwards was 66.5 MB without pools and 67.0 MB with them.
// The bytes are not in pooled blocks. A container's payload is an array of
// 16-word element boxes, so a dict at PyDict_MINSIZE is 8 x 16 x 8 = 1 KB per
// array -- past the 512-byte class ceiling and already going straight to the
// system allocator, which is 94% of that program's heap. Only a two-phase
// workload whose objects all fit the classes showed anything, and that was 10%
// of peak.
//
// So the ordering is: shrink the element box first (`box_abi::kWordsPerBox` is
// 16 for a maximum of 3 lanes across every contract), which puts container
// payloads back under the ceiling, and then the pool layer has something to
// give back. Doing it in the other order buys a 10-30% regression for 0.5 MB.
//
// One thing the port found that outlives it: `redirectAllocationsToObjectAllocator`
// skips functions named `LyMem_*`, so an allocator helper spelled any other way
// has its `free(pool)` rewritten into `LyMem_Free(pool)` -- which reads the
// pool's own header as a block prefix and pushes the pool onto its own free
// chain. It reaches a program as a hang, not as a crash.
//
// ⛔ NO `address_in_range`. CPython derives the pool (and so the size class)
// from the address by masking, which lets it keep zero per-object overhead but
// costs it a probe into the arena table on every free. This carries a 16-byte
// prefix holding the class instead: one store per allocation, one load per
// free, and no read of memory that may not be ours.
//
// ⛔ NOT THREAD-SAFE, exactly like `g_current_parts` beside it: the free lists
// and the bump pointer are plain globals. The runtime's object model is
// single-threaded (the fork-join used by matmul allocates nothing), and
// `--fsanitize=address|leak|thread` bypasses this allocator entirely, which is
// CPython's PYTHONMALLOC=malloc.
constexpr std::int64_t kObjectAllocatorPrefixBytes = 16;
constexpr std::int64_t kObjectAllocatorGranularity = 16;
constexpr std::int64_t kObjectAllocatorClasses = 32;   // 16..512 bytes
constexpr std::int64_t kObjectAllocatorArenaBytes = 1 << 20;

void buildObjectAllocator(SupportBuilder &b) {
  b.declareExternal("malloc", b.builder.getFunctionType({b.i64()}, {b.ptr()}));
  b.declareExternal("free", b.builder.getFunctionType({b.ptr()}, {}));
  b.declareExternal("realloc",
                    b.builder.getFunctionType({b.ptr(), b.i64()}, {b.ptr()}));
  b.declareExternal(
      "memcpy", b.builder.getFunctionType({b.ptr(), b.ptr(), b.i64()},
                                          {b.ptr()}));

  auto zeroInitGlobal = [&](llvm::StringRef name, mlir::Type type) {
    if (b.module.lookupSymbol(name))
      return;
    mlir::OpBuilder::InsertionGuard guard(b.builder);
    b.builder.setInsertionPointToEnd(b.module.getBody());
    auto global = mlir::LLVM::GlobalOp::create(
        b.builder, b.loc, type, /*isConstant=*/false,
        mlir::LLVM::Linkage::Internal, name, mlir::Attribute(),
        /*alignment=*/8);
    global.setDsoLocal(true);
    mlir::Block *init = b.builder.createBlock(&global.getInitializerRegion());
    b.builder.setInsertionPointToEnd(init);
    mlir::Value zero = mlir::LLVM::ZeroOp::create(b.builder, b.loc, type);
    mlir::LLVM::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{zero});
  };
  // Head of each class's free list, indexed by class (1..32); slot 0 unused.
  zeroInitGlobal("g_lymem_class_heads",
                 mlir::LLVM::LLVMArrayType::get(
                     b.ptr(), kObjectAllocatorClasses + 1));
  // [next address, bytes remaining] of the arena being carved.
  zeroInitGlobal("g_lymem_arena",
                 mlir::LLVM::LLVMArrayType::get(b.i64(), 2));

  auto storeI64At = [&](mlir::Value value, mlir::Value pointer) {
    mlir::LLVM::StoreOp::create(b.builder, b.loc, value, pointer,
                                /*alignment=*/8);
  };
  auto storePtrAt = [&](mlir::Value value, mlir::Value pointer) {
    mlir::LLVM::StoreOp::create(b.builder, b.loc, value, pointer,
                                /*alignment=*/8);
  };
  auto classSlot = [&](mlir::Value classIndex) {
    return mlir::LLVM::GEPOp::create(b.builder, b.loc, b.ptr(), b.ptr(),
                                     b.addrOf("g_lymem_class_heads"),
                                     mlir::ValueRange{classIndex});
  };

  // ---- ptr LyMem_Alloc(i64 size) -------------------------------------------
  {
    auto fn = b.beginFunction(
        "LyMem_Alloc", b.builder.getFunctionType({b.i64()}, {b.ptr()}));
    mlir::Block *entry = fn.addEntryBlock();
    mlir::Region &body = fn.getBody();
    mlir::Block *large = b.builder.createBlock(&body);
    mlir::Block *small = b.builder.createBlock(&body);
    mlir::Block *pop = b.builder.createBlock(&body);
    mlir::Block *carve = b.builder.createBlock(&body);
    mlir::Block *fresh = b.builder.createBlock(&body);
    mlir::Block *take = b.builder.createBlock(&body, {}, {b.i64(), b.ptr()},
                                              {b.loc, b.loc});
    mlir::Block *publish = b.builder.createBlock(&body);
    mlir::Block *fromArena = b.builder.createBlock(&body);
    mlir::Block *stamp = b.builder.createBlock(&body, {}, {b.ptr()}, {b.loc});
    mlir::Block *fail = b.builder.createBlock(&body);

    b.builder.setInsertionPointToEnd(entry);
    mlir::Value total = mlir::arith::AddIOp::create(
        b.builder, b.loc, entry->getArgument(0),
        b.iconst(kObjectAllocatorPrefixBytes));
    mlir::Value isLarge =
        b.cmpi(mlir::arith::CmpIPredicate::sgt, total,
               b.iconst(kObjectAllocatorGranularity * kObjectAllocatorClasses));
    mlir::cf::CondBranchOp::create(b.builder, b.loc, isLarge, large,
                                   mlir::ValueRange{}, small,
                                   mlir::ValueRange{});

    // Above the threshold the system allocator answers directly; the prefix
    // records that so the free knows which way to go back.
    b.builder.setInsertionPointToEnd(large);
    mlir::Value block = b.call("malloc", b.ptr(), mlir::ValueRange{total})
                            .front();
    mlir::cf::CondBranchOp::create(
        b.builder, b.loc, b.ptrEq(block, b.nullPtr()), fail,
        mlir::ValueRange{}, take, mlir::ValueRange{b.iconst(-1), block});

    b.builder.setInsertionPointToEnd(small);
    mlir::Value rounded = mlir::arith::AddIOp::create(
        b.builder, b.loc, total, b.iconst(kObjectAllocatorGranularity - 1));
    mlir::Value classIndex =
        mlir::arith::ShRSIOp::create(b.builder, b.loc, rounded, b.iconst(4));
    mlir::Value head = b.loadPtrVal(classSlot(classIndex));
    mlir::cf::CondBranchOp::create(b.builder, b.loc,
                                   b.ptrEq(head, b.nullPtr()), carve,
                                   mlir::ValueRange{}, pop, mlir::ValueRange{});

    // The free list holds the NEXT pointer in the prefix's second word; the
    // first is the class, rewritten below because the pop overwrites nothing.
    b.builder.setInsertionPointToEnd(pop);
    mlir::Value poppedHead = b.loadPtrVal(classSlot(classIndex));
    mlir::Value next = b.loadPtrVal(b.gepI64(poppedHead, b.iconst(1)));
    storePtrAt(next, classSlot(classIndex));
    storeI64At(classIndex, poppedHead);
    mlir::func::ReturnOp::create(
        b.builder, b.loc,
        mlir::ValueRange{b.gepI8(poppedHead,
                                 b.iconst(kObjectAllocatorPrefixBytes))});

    b.builder.setInsertionPointToEnd(carve);
    mlir::Value bytes = mlir::arith::MulIOp::create(
        b.builder, b.loc, classIndex, b.iconst(kObjectAllocatorGranularity));
    mlir::Value arena = b.addrOf("g_lymem_arena");
    mlir::Value remaining = b.loadI64(b.gepI64(arena, b.iconst(1)));
    mlir::Value fits =
        b.cmpi(mlir::arith::CmpIPredicate::sge, remaining, bytes);
    mlir::cf::CondBranchOp::create(
        b.builder, b.loc, fits, take,
        mlir::ValueRange{classIndex, b.nullPtr()}, fresh, mlir::ValueRange{});

    // A new arena. The remainder of the old one is abandoned -- at most one
    // class width, which is why the classes are the granularity.
    b.builder.setInsertionPointToEnd(fresh);
    mlir::Value chunk =
        b.call("malloc", b.ptr(),
               mlir::ValueRange{b.iconst(kObjectAllocatorArenaBytes)})
            .front();
    mlir::cf::CondBranchOp::create(b.builder, b.loc,
                                   b.ptrEq(chunk, b.nullPtr()), fail,
                                   mlir::ValueRange{}, publish,
                                   mlir::ValueRange{});
    b.builder.setInsertionPointToEnd(publish);
    storeI64At(b.ptrToInt(chunk), arena);
    storeI64At(b.iconst(kObjectAllocatorArenaBytes),
               b.gepI64(arena, b.iconst(1)));
    mlir::cf::BranchOp::create(b.builder, b.loc, take,
                               mlir::ValueRange{classIndex, b.nullPtr()});

    // `take` serves both the carve and the large path: a negative class means
    // the block is already in hand (the `malloc` above) and only needs its
    // prefix stamped.
    b.builder.setInsertionPointToEnd(take);
    mlir::Value takenClass = take->getArgument(0);
    mlir::cf::CondBranchOp::create(
        b.builder, b.loc,
        b.cmpi(mlir::arith::CmpIPredicate::slt, takenClass, b.iconst(0)), stamp,
        mlir::ValueRange{take->getArgument(1)}, fromArena,
        mlir::ValueRange{});

    b.builder.setInsertionPointToEnd(fromArena);
    mlir::Value takeBytes = mlir::arith::MulIOp::create(
        b.builder, b.loc, takenClass, b.iconst(kObjectAllocatorGranularity));
    mlir::Value cursor = b.loadI64(arena);
    mlir::Value left = b.loadI64(b.gepI64(arena, b.iconst(1)));
    storeI64At(mlir::arith::AddIOp::create(b.builder, b.loc, cursor, takeBytes),
               arena);
    storeI64At(mlir::arith::SubIOp::create(b.builder, b.loc, left, takeBytes),
               b.gepI64(arena, b.iconst(1)));
    mlir::cf::BranchOp::create(b.builder, b.loc, stamp,
                               mlir::ValueRange{b.intToPtr(cursor)});

    b.builder.setInsertionPointToEnd(stamp);
    storeI64At(takenClass, stamp->getArgument(0));
    mlir::func::ReturnOp::create(
        b.builder, b.loc,
        mlir::ValueRange{b.gepI8(stamp->getArgument(0),
                                 b.iconst(kObjectAllocatorPrefixBytes))});

    b.builder.setInsertionPointToEnd(fail);
    mlir::func::ReturnOp::create(b.builder, b.loc,
                                 mlir::ValueRange{b.nullPtr()});
  }

  // ---- void LyMem_Free(ptr block) ------------------------------------------
  {
    auto fn = b.beginFunction("LyMem_Free",
                              b.builder.getFunctionType({b.ptr()}, {}));
    mlir::Block *entry = fn.addEntryBlock();
    mlir::Region &body = fn.getBody();
    mlir::Block *live = b.builder.createBlock(&body);
    mlir::Block *large = b.builder.createBlock(&body);
    mlir::Block *small = b.builder.createBlock(&body);
    mlir::Block *done = b.builder.createBlock(&body);

    b.builder.setInsertionPointToEnd(entry);
    mlir::cf::CondBranchOp::create(
        b.builder, b.loc, b.ptrEq(entry->getArgument(0), b.nullPtr()), done,
        mlir::ValueRange{}, live, mlir::ValueRange{});

    b.builder.setInsertionPointToEnd(live);
    mlir::Value prefix =
        b.gepI8(entry->getArgument(0), b.iconst(-kObjectAllocatorPrefixBytes));
    mlir::Value classIndex = b.loadI64(prefix);
    mlir::cf::CondBranchOp::create(
        b.builder, b.loc,
        b.cmpi(mlir::arith::CmpIPredicate::slt, classIndex, b.iconst(0)), large,
        mlir::ValueRange{}, small, mlir::ValueRange{});

    b.builder.setInsertionPointToEnd(large);
    b.call("free", mlir::TypeRange{}, mlir::ValueRange{prefix});
    mlir::cf::BranchOp::create(b.builder, b.loc, done, mlir::ValueRange{});

    b.builder.setInsertionPointToEnd(small);
    mlir::Value slot = classSlot(classIndex);
    storePtrAt(b.loadPtrVal(slot), b.gepI64(prefix, b.iconst(1)));
    storePtrAt(prefix, slot);
    mlir::cf::BranchOp::create(b.builder, b.loc, done, mlir::ValueRange{});

    b.builder.setInsertionPointToEnd(done);
    mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
  }

  // ---- ptr LyMem_Realloc(ptr block, i64 size) ------------------------------
  {
    auto fn = b.beginFunction(
        "LyMem_Realloc",
        b.builder.getFunctionType({b.ptr(), b.i64()}, {b.ptr()}));
    mlir::Block *entry = fn.addEntryBlock();
    mlir::Region &body = fn.getBody();
    mlir::Block *fresh = b.builder.createBlock(&body);
    mlir::Block *held = b.builder.createBlock(&body);
    mlir::Block *large = b.builder.createBlock(&body);
    mlir::Block *small = b.builder.createBlock(&body);
    mlir::Block *keep = b.builder.createBlock(&body);
    mlir::Block *move = b.builder.createBlock(&body);

    b.builder.setInsertionPointToEnd(entry);
    mlir::cf::CondBranchOp::create(
        b.builder, b.loc, b.ptrEq(entry->getArgument(0), b.nullPtr()), fresh,
        mlir::ValueRange{}, held, mlir::ValueRange{});

    b.builder.setInsertionPointToEnd(fresh);
    mlir::func::ReturnOp::create(
        b.builder, b.loc,
        b.call("LyMem_Alloc", b.ptr(),
               mlir::ValueRange{entry->getArgument(1)}));

    b.builder.setInsertionPointToEnd(held);
    mlir::Value prefix =
        b.gepI8(entry->getArgument(0), b.iconst(-kObjectAllocatorPrefixBytes));
    mlir::Value classIndex = b.loadI64(prefix);
    mlir::cf::CondBranchOp::create(
        b.builder, b.loc,
        b.cmpi(mlir::arith::CmpIPredicate::slt, classIndex, b.iconst(0)), large,
        mlir::ValueRange{}, small, mlir::ValueRange{});

    b.builder.setInsertionPointToEnd(large);
    mlir::Value grown =
        b.call("realloc", b.ptr(),
               mlir::ValueRange{prefix,
                                mlir::arith::AddIOp::create(
                                    b.builder, b.loc, entry->getArgument(1),
                                    b.iconst(kObjectAllocatorPrefixBytes))})
            .front();
    mlir::func::ReturnOp::create(
        b.builder, b.loc,
        mlir::ValueRange{
            b.gepI8(grown, b.iconst(kObjectAllocatorPrefixBytes))});

    // A pooled block cannot grow where it lies: its neighbours belong to other
    // objects. It keeps its block while the request still fits the class it is
    // already in, which is what makes an appending loop amortise.
    b.builder.setInsertionPointToEnd(small);
    mlir::Value capacity = mlir::arith::SubIOp::create(
        b.builder, b.loc,
        mlir::arith::MulIOp::create(b.builder, b.loc, classIndex,
                                    b.iconst(kObjectAllocatorGranularity)),
        b.iconst(kObjectAllocatorPrefixBytes));
    mlir::cf::CondBranchOp::create(
        b.builder, b.loc,
        b.cmpi(mlir::arith::CmpIPredicate::sle, entry->getArgument(1),
               capacity),
        keep, mlir::ValueRange{}, move, mlir::ValueRange{});

    b.builder.setInsertionPointToEnd(keep);
    mlir::func::ReturnOp::create(b.builder, b.loc,
                                 mlir::ValueRange{entry->getArgument(0)});

    b.builder.setInsertionPointToEnd(move);
    mlir::Value moved =
        b.call("LyMem_Alloc", b.ptr(),
               mlir::ValueRange{entry->getArgument(1)})
            .front();
    b.call("memcpy", b.ptr(),
           mlir::ValueRange{moved, entry->getArgument(0), capacity});
    b.call("LyMem_Free", mlir::TypeRange{},
           mlir::ValueRange{entry->getArgument(0)});
    mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{moved});
  }
}

// void free_raw_i64_ptr(i64 address): free() a non-null raw address.
void buildFreeRawI64Ptr(SupportBuilder &b) {
  auto fn = b.beginFunction("free_raw_i64_ptr",
                            b.builder.getFunctionType({b.i64()}, {}));
  mlir::Block *entry = fn.addEntryBlock();
  mlir::Region &body = fn.getBody();
  mlir::Block *doFree = b.builder.createBlock(&body);
  mlir::Block *done = b.builder.createBlock(&body);
  b.builder.setInsertionPointToEnd(entry);
  mlir::Value isNull =
      b.cmpi(mlir::arith::CmpIPredicate::eq, entry->getArgument(0), b.iconst(0));
  mlir::cf::CondBranchOp::create(b.builder, b.loc, isNull, done,
                                 mlir::ValueRange{}, doFree,
                                 mlir::ValueRange{});
  b.builder.setInsertionPointToEnd(doFree);
  mlir::func::CallOp::create(b.builder, b.loc, "free", mlir::TypeRange{},
                             mlir::ValueRange{b.intToPtr(entry->getArgument(0))});
  mlir::cf::BranchOp::create(b.builder, b.loc, done, mlir::ValueRange{});
  b.builder.setInsertionPointToEnd(done);
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
}

// i64 realloc_raw_i64_ptr(i64 address, i64 bytes): CPython's list_resize hands
// the items block to PyMem_Realloc rather than allocating a new one and
// copying, and the mild 1.125x over-allocation it uses is only affordable
// because of that -- realloc usually extends the block where it lies.
// Answers the new base address.
void buildReallocRawI64Ptr(SupportBuilder &b) {
  b.declareExternal("realloc",
                    b.builder.getFunctionType({b.ptr(), b.i64()}, {b.ptr()}));
  auto fn = b.beginFunction(
      "realloc_raw_i64_ptr",
      b.builder.getFunctionType({b.i64(), b.i64()}, {b.i64()}));
  mlir::Block *entry = fn.addEntryBlock();
  b.builder.setInsertionPointToEnd(entry);
  mlir::Value grown =
      b.call("realloc", b.ptr(),
             mlir::ValueRange{b.intToPtr(entry->getArgument(0)),
                              entry->getArgument(1)})
          .front();
  mlir::func::ReturnOp::create(b.builder, b.loc, b.ptrToInt(grown));
}

// i1 release_storage_raw_to_zero(ptr storage): atomically decrement the
// refcount word at address; return whether it dropped to zero. Skips null,
// tagged (odd), and immortal (INT64_MAX) storages.
void buildReleaseStorageRawToZero(SupportBuilder &b) {
  auto fn = b.beginFunction(
      "release_storage_raw_to_zero",
      b.builder.getFunctionType({b.ptr()}, {b.i1()}));
  mlir::Block *entry = fn.addEntryBlock();
  mlir::Region &body = fn.getBody();
  mlir::Value storage = entry->getArgument(0);

  mlir::Block *tagCheck = b.builder.createBlock(&body);
  mlir::Block *probe = b.builder.createBlock(&body);
  mlir::Block *positive = b.builder.createBlock(&body);
  mlir::Block *done = b.builder.createBlock(&body);

  mlir::Value zero = b.iconst(0);
  mlir::Value one = b.iconst(1);
  mlir::Value immortal = b.iconst(9223372036854775807LL);
  mlir::Value falseVal =
      mlir::arith::ConstantIntOp::create(b.builder, b.loc, b.i1(), 0);

  b.builder.setInsertionPointToEnd(entry);
  // ⛔ Narrowed HERE and only here: null and the odd-tag test are questions
  // about the bit pattern, and a tagged reference is an immediate rather than
  // an address. The pointer itself is what the load and the atomic use, so a
  // reference that IS an allocation reaches memory with its provenance intact
  // -- widening the word back, which this did, threw that away at the one
  // access every owned object passes through.
  mlir::Value address = b.ptrToInt(storage);
  mlir::Value isNull = b.cmpi(mlir::arith::CmpIPredicate::eq, address, zero);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, isNull, done,
                                 mlir::ValueRange{}, tagCheck,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(tagCheck);
  mlir::Value tag = mlir::arith::AndIOp::create(b.builder, b.loc, address, one);
  mlir::Value isTagged = b.cmpi(mlir::arith::CmpIPredicate::eq, tag, one);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, isTagged, done,
                                 mlir::ValueRange{}, probe,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(probe);
  mlir::Value observed = b.loadI64(storage);
  mlir::Value preImmortal =
      b.cmpi(mlir::arith::CmpIPredicate::eq, observed, immortal);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, preImmortal, done,
                                 mlir::ValueRange{}, positive,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(positive);
  mlir::Value observedPositive =
      b.cmpi(mlir::arith::CmpIPredicate::sgt, observed, zero);
  mlir::cf::AssertOp::create(
      b.builder, b.loc, observedPositive,
      "release_storage_raw_to_zero observed non-positive refcount");
  mlir::Value previous = mlir::LLVM::AtomicRMWOp::create(
      b.builder, b.loc, mlir::LLVM::AtomicBinOp::sub, storage, one,
      mlir::LLVM::AtomicOrdering::acq_rel);
  mlir::Value previousPositive =
      b.cmpi(mlir::arith::CmpIPredicate::sgt, previous, zero);
  mlir::cf::AssertOp::create(
      b.builder, b.loc, previousPositive,
      "release_storage_raw_to_zero raced with non-positive refcount");
  mlir::Value becameZero =
      b.cmpi(mlir::arith::CmpIPredicate::eq, previous, one);
  mlir::func::ReturnOp::create(b.builder, b.loc, becameZero);

  b.builder.setInsertionPointToEnd(done);
  mlir::func::ReturnOp::create(b.builder, b.loc, falseVal);
}

// void retain_storage_raw(ptr storage): atomically increment the refcount
// word it points at. Mirror of release_storage_raw_to_zero: skips null, tagged
// (odd), and immortal (INT64_MAX) storages.
void buildRetainStorageRaw(SupportBuilder &b) {
  auto fn = b.beginFunction("retain_storage_raw",
                            b.builder.getFunctionType({b.ptr()}, {}));
  mlir::Block *entry = fn.addEntryBlock();
  mlir::Region &body = fn.getBody();
  mlir::Value storage = entry->getArgument(0);

  mlir::Block *tagCheck = b.builder.createBlock(&body);
  mlir::Block *probe = b.builder.createBlock(&body);
  mlir::Block *bump = b.builder.createBlock(&body);
  mlir::Block *done = b.builder.createBlock(&body);

  mlir::Value zero = b.iconst(0);
  mlir::Value one = b.iconst(1);
  mlir::Value immortal = b.iconst(9223372036854775807LL);

  b.builder.setInsertionPointToEnd(entry);
  // ⛔ Narrowed HERE and only here: null and the odd-tag test are questions
  // about the bit pattern, and a tagged reference is an immediate rather than
  // an address. The pointer itself is what the load and the atomic use, so a
  // reference that IS an allocation reaches memory with its provenance intact
  // -- widening the word back, which this did, threw that away at the one
  // access every owned object passes through.
  mlir::Value address = b.ptrToInt(storage);
  mlir::Value isNull = b.cmpi(mlir::arith::CmpIPredicate::eq, address, zero);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, isNull, done,
                                 mlir::ValueRange{}, tagCheck,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(tagCheck);
  mlir::Value tag = mlir::arith::AndIOp::create(b.builder, b.loc, address, one);
  mlir::Value isTagged = b.cmpi(mlir::arith::CmpIPredicate::eq, tag, one);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, isTagged, done,
                                 mlir::ValueRange{}, probe,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(probe);
  mlir::Value observed = b.loadI64(storage);
  mlir::Value preImmortal =
      b.cmpi(mlir::arith::CmpIPredicate::eq, observed, immortal);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, preImmortal, done,
                                 mlir::ValueRange{}, bump,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(bump);
  mlir::LLVM::AtomicRMWOp::create(b.builder, b.loc,
                                  mlir::LLVM::AtomicBinOp::add, storage, one,
                                  mlir::LLVM::AtomicOrdering::acq_rel);
  mlir::cf::BranchOp::create(b.builder, b.loc, done, mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(done);
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
}

// Shared shape: release a single-allocation storage rooted at a POINTER,
// freeing it if the refcount hit zero. `release_unicode_raw`'s second argument
// (an interior bytes view) needs no separate free.
void buildReleaseSingleAllocation(SupportBuilder &b, llvm::StringRef name,
                                  bool twoArgs) {
  llvm::SmallVector<mlir::Type, 2> inputs = {b.ptr()};
  if (twoArgs)
    inputs.push_back(b.ptr());
  auto fn = b.beginFunction(name, b.builder.getFunctionType(inputs, {}));
  mlir::Block *entry = fn.addEntryBlock();
  mlir::Region &body = fn.getBody();
  mlir::Block *freeBlock = b.builder.createBlock(&body);
  mlir::Block *done = b.builder.createBlock(&body);
  b.builder.setInsertionPointToEnd(entry);
  // The one narrowing this path takes: the tag check needs a word because a
  // tagged reference is an immediate, not an address. Everything after it is
  // known to be a real allocation, so the pointer carries on unbroken.
  auto becameZero = mlir::func::CallOp::create(
      b.builder, b.loc, "release_storage_raw_to_zero", b.i1(),
      mlir::ValueRange{entry->getArgument(0)});
  mlir::cf::CondBranchOp::create(b.builder, b.loc, becameZero.getResult(0),
                                 freeBlock, mlir::ValueRange{}, done,
                                 mlir::ValueRange{});
  b.builder.setInsertionPointToEnd(freeBlock);
  mlir::func::CallOp::create(b.builder, b.loc, "free", mlir::TypeRange{},
                             mlir::ValueRange{entry->getArgument(0)});
  mlir::cf::BranchOp::create(b.builder, b.loc, done, mlir::ValueRange{});
  b.builder.setInsertionPointToEnd(done);
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
}

// void write_len(i32 fd, ptr data, i64 len): raw write(2) guarded by len > 0
// and data != null. Public: the traceback printer still in the native module
// calls it through a bridge declaration.
void buildWriteLen(SupportBuilder &b) {
  auto fn = b.beginFunction(
      "write_len", b.builder.getFunctionType({b.i32(), b.ptr(), b.i64()}, {}));
  mlir::Block *entry = fn.addEntryBlock();
  mlir::Region &body = fn.getBody();
  mlir::Block *doWrite = b.builder.createBlock(&body);
  mlir::Block *done = b.builder.createBlock(&body);
  b.builder.setInsertionPointToEnd(entry);
  mlir::Value zero = b.iconst(0);
  mlir::Value null =
      mlir::LLVM::ZeroOp::create(b.builder, b.loc, b.ptr()).getResult();
  mlir::Value lenPositive =
      b.cmpi(mlir::arith::CmpIPredicate::sgt, entry->getArgument(2), zero);
  mlir::Value dataOk = mlir::LLVM::ICmpOp::create(
      b.builder, b.loc, mlir::LLVM::ICmpPredicate::ne, entry->getArgument(1),
      null);
  mlir::Value shouldWrite =
      mlir::arith::AndIOp::create(b.builder, b.loc, lenPositive, dataOk);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, shouldWrite, doWrite,
                                 mlir::ValueRange{}, done, mlir::ValueRange{});
  b.builder.setInsertionPointToEnd(doWrite);
  mlir::func::CallOp::create(
      b.builder, b.loc, "write", b.i64(),
      mlir::ValueRange{entry->getArgument(0), entry->getArgument(1),
                       entry->getArgument(2)});
  mlir::cf::BranchOp::create(b.builder, b.loc, done, mlir::ValueRange{});
  b.builder.setInsertionPointToEnd(done);
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
}

// void write_cstr(i32 fd, ptr cstr): strlen + write_len for non-null cstr.
void buildWriteCStr(SupportBuilder &b) {
  auto fn = b.beginFunction(
      "write_cstr", b.builder.getFunctionType({b.i32(), b.ptr()}, {}));
  mlir::Block *entry = fn.addEntryBlock();
  mlir::Region &body = fn.getBody();
  mlir::Block *doWrite = b.builder.createBlock(&body);
  mlir::Block *done = b.builder.createBlock(&body);
  b.builder.setInsertionPointToEnd(entry);
  mlir::Value null =
      mlir::LLVM::ZeroOp::create(b.builder, b.loc, b.ptr()).getResult();
  mlir::Value isNull = mlir::LLVM::ICmpOp::create(
      b.builder, b.loc, mlir::LLVM::ICmpPredicate::eq, entry->getArgument(1),
      null);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, isNull, done,
                                 mlir::ValueRange{}, doWrite,
                                 mlir::ValueRange{});
  b.builder.setInsertionPointToEnd(doWrite);
  auto length =
      mlir::func::CallOp::create(b.builder, b.loc, "strlen", b.i64(),
                                 mlir::ValueRange{entry->getArgument(1)});
  mlir::func::CallOp::create(
      b.builder, b.loc, "write_len", mlir::TypeRange{},
      mlir::ValueRange{entry->getArgument(0), entry->getArgument(1),
                       length.getResult(0)});
  mlir::cf::BranchOp::create(b.builder, b.loc, done, mlir::ValueRange{});
  b.builder.setInsertionPointToEnd(done);
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
}

// void write_char(i32 fd, i8 ch): one-byte stack buffer + write_len.
void buildWriteChar(SupportBuilder &b) {
  auto fn = b.beginFunction(
      "write_char", b.builder.getFunctionType({b.i32(), b.i8()}, {}));
  mlir::Block *entry = fn.addEntryBlock();
  b.builder.setInsertionPointToEnd(entry);
  mlir::Value one = b.iconst(1);
  mlir::Value buffer = mlir::LLVM::AllocaOp::create(b.builder, b.loc, b.ptr(),
                                                    b.i8(), one,
                                                    /*alignment=*/1);
  mlir::LLVM::StoreOp::create(b.builder, b.loc, entry->getArgument(1), buffer,
                              /*alignment=*/1);
  mlir::func::CallOp::create(
      b.builder, b.loc, "write_len", mlir::TypeRange{},
      mlir::ValueRange{entry->getArgument(0), buffer, one});
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
}

// void write_buffered(i32 fd, ptr data, i32 len): snprintf-style results —
// negative means an encoding error (skipped), otherwise clamp to the 1023-byte
// buffer capacity and write.
void buildWriteBuffered(SupportBuilder &b) {
  auto fn = b.beginFunction(
      "write_buffered",
      b.builder.getFunctionType({b.i32(), b.ptr(), b.i32()}, {}));
  mlir::Block *entry = fn.addEntryBlock();
  mlir::Region &body = fn.getBody();
  mlir::Block *doWrite = b.builder.createBlock(&body);
  mlir::Block *done = b.builder.createBlock(&body);
  b.builder.setInsertionPointToEnd(entry);
  mlir::Value zero32 = mlir::arith::ConstantIntOp::create(b.builder, b.loc,
                                                          b.i32(), 0);
  mlir::Value negative = b.cmpi(mlir::arith::CmpIPredicate::slt,
                                entry->getArgument(2), zero32);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, negative, done,
                                 mlir::ValueRange{}, doWrite,
                                 mlir::ValueRange{});
  b.builder.setInsertionPointToEnd(doWrite);
  mlir::Value extended = mlir::arith::ExtSIOp::create(b.builder, b.loc, b.i64(),
                                                      entry->getArgument(2));
  mlir::Value capacity = b.iconst(1023);
  mlir::Value overflow =
      b.cmpi(mlir::arith::CmpIPredicate::sgt, extended, capacity);
  mlir::Value clamped = mlir::arith::SelectOp::create(b.builder, b.loc,
                                                      overflow, capacity,
                                                      extended);
  mlir::func::CallOp::create(
      b.builder, b.loc, "write_len", mlir::TypeRange{},
      mlir::ValueRange{entry->getArgument(0), entry->getArgument(1), clamped});
  mlir::cf::BranchOp::create(b.builder, b.loc, done, mlir::ValueRange{});
  b.builder.setInsertionPointToEnd(done);
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
}

// i64 boxed_int_value(i64 meta_bits, i64 digits_bits): decode a boxed int
// payload (sign at meta[0], digit count at meta[1], base-2^30 digits) into a
// signed i64 — the runtime's small-int envelope. Shared ABI: builtins.mlir
// declares and calls it.
void buildBoxedIntValue(SupportBuilder &b) {
  auto fn = b.beginFunction(
      "boxed_int_value", b.builder.getFunctionType({b.i64(), b.i64()},
                                                   {b.i64()}));
  mlir::Block *entry = fn.addEntryBlock();
  mlir::Region &body = fn.getBody();
  mlir::Block *zeroBlock = b.builder.createBlock(&body);
  mlir::Block *decode = b.builder.createBlock(&body);
  mlir::Block *digits = b.builder.createBlock(&body);
  b.builder.setInsertionPointToEnd(entry);
  mlir::Value zero = b.iconst(0);
  mlir::Value metaMissing = b.cmpi(mlir::arith::CmpIPredicate::eq,
                                   entry->getArgument(0), zero);
  mlir::Value digitsMissing = b.cmpi(mlir::arith::CmpIPredicate::eq,
                                     entry->getArgument(1), zero);
  mlir::Value missing = b.orBit(metaMissing, digitsMissing);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, missing, zeroBlock,
                                 mlir::ValueRange{}, decode,
                                 mlir::ValueRange{});
  b.builder.setInsertionPointToEnd(zeroBlock);
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{zero});
  b.builder.setInsertionPointToEnd(decode);
  mlir::Value metaPtr = b.intToPtr(entry->getArgument(0));
  mlir::Value digitsPtr = b.intToPtr(entry->getArgument(1));
  mlir::Value sign = b.loadI64(b.gepI64(metaPtr, zero));
  mlir::Value count = b.loadI64(b.gepI64(metaPtr, b.iconst(1)));
  mlir::Value countEmpty =
      b.cmpi(mlir::arith::CmpIPredicate::sle, count, zero);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, countEmpty, zeroBlock,
                                 mlir::ValueRange{}, digits,
                                 mlir::ValueRange{});
  b.builder.setInsertionPointToEnd(digits);
  mlir::Value countIndex = mlir::arith::IndexCastOp::create(
      b.builder, b.loc, b.builder.getIndexType(), count);
  mlir::Value zeroIndex =
      mlir::arith::ConstantIndexOp::create(b.builder, b.loc, 0);
  mlir::Value oneIndex =
      mlir::arith::ConstantIndexOp::create(b.builder, b.loc, 1);
  mlir::Value one = b.iconst(1);
  mlir::Value base = b.iconst(1073741824);
  auto loop = mlir::scf::ForOp::create(b.builder, b.loc, zeroIndex, countIndex,
                                       oneIndex, mlir::ValueRange{zero});
  {
    mlir::OpBuilder::InsertionGuard guard(b.builder);
    b.builder.setInsertionPointToStart(loop.getBody());
    mlir::Value position = mlir::arith::IndexCastOp::create(
        b.builder, b.loc, b.i64(), loop.getInductionVar());
    mlir::Value fromEnd =
        mlir::arith::AddIOp::create(b.builder, b.loc, position, one);
    mlir::Value digitIndex =
        mlir::arith::SubIOp::create(b.builder, b.loc, count, fromEnd);
    mlir::Value digitPtr = mlir::LLVM::GEPOp::create(
        b.builder, b.loc, b.ptr(), b.i32(), digitsPtr,
        mlir::ValueRange{digitIndex});
    mlir::Value digit = mlir::LLVM::LoadOp::create(b.builder, b.loc, b.i32(),
                                                   digitPtr, /*alignment=*/4);
    mlir::Value wide =
        mlir::arith::ExtUIOp::create(b.builder, b.loc, b.i64(), digit);
    mlir::Value shifted = mlir::arith::MulIOp::create(
        b.builder, b.loc, loop.getRegionIterArg(0), base);
    mlir::Value accumulated =
        mlir::arith::AddIOp::create(b.builder, b.loc, shifted, wide);
    mlir::scf::YieldOp::create(b.builder, b.loc,
                               mlir::ValueRange{accumulated});
  }
  mlir::Value magnitude = loop.getResult(0);
  mlir::Value isNegative = b.cmpi(mlir::arith::CmpIPredicate::slt, sign, zero);
  mlir::Value negated =
      mlir::arith::SubIOp::create(b.builder, b.loc, zero, magnitude);
  mlir::Value result = mlir::arith::SelectOp::create(b.builder, b.loc,
                                                     isNegative, negated,
                                                     magnitude);
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{result});
}

// void print_bytes(i32 fd, ptr data, i64 offset, i64 size, i64 stride,
// i64 len): validated memref-view write. Contiguous views write in one call;
// strided views write per element. Invalid descriptors abort.
void buildPrintBytes(SupportBuilder &b) {
  auto fn = b.beginFunction(
      "print_bytes",
      b.builder.getFunctionType(
          {b.i32(), b.ptr(), b.i64(), b.i64(), b.i64(), b.i64()}, {}),
      /*isPrivate=*/true);
  mlir::Block *entry = fn.addEntryBlock();
  mlir::Region &body = fn.getBody();
  mlir::Block *checkDescriptor = b.builder.createBlock(&body);
  mlir::Block *validate = b.builder.createBlock(&body);
  mlir::Block *dispatch = b.builder.createBlock(&body);
  mlir::Block *contiguous = b.builder.createBlock(&body);
  mlir::Block *strided = b.builder.createBlock(&body);
  mlir::Block *done = b.builder.createBlock(&body);
  mlir::Block *trap = b.builder.createBlock(&body);
  mlir::Value fd = entry->getArgument(0);
  mlir::Value data = entry->getArgument(1);
  mlir::Value offset = entry->getArgument(2);
  mlir::Value size = entry->getArgument(3);
  mlir::Value stride = entry->getArgument(4);
  mlir::Value len = entry->getArgument(5);

  b.builder.setInsertionPointToEnd(entry);
  mlir::Value zero = b.iconst(0);
  mlir::Value one = b.iconst(1);
  mlir::Value lenNegative = b.cmpi(mlir::arith::CmpIPredicate::slt, len, zero);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, lenNegative, trap,
                                 mlir::ValueRange{}, checkDescriptor,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(checkDescriptor);
  mlir::Value lenZero = b.cmpi(mlir::arith::CmpIPredicate::eq, len, zero);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, lenZero, done,
                                 mlir::ValueRange{}, validate,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(validate);
  mlir::Value null =
      mlir::LLVM::ZeroOp::create(b.builder, b.loc, b.ptr()).getResult();
  mlir::Value offsetNegative =
      b.cmpi(mlir::arith::CmpIPredicate::slt, offset, zero);
  mlir::Value sizeNegative =
      b.cmpi(mlir::arith::CmpIPredicate::slt, size, zero);
  mlir::Value strideInvalid =
      b.cmpi(mlir::arith::CmpIPredicate::slt, stride, one);
  mlir::Value dataNull = mlir::LLVM::ICmpOp::create(
      b.builder, b.loc, mlir::LLVM::ICmpPredicate::eq, data, null);
  mlir::Value lenOverSize = b.cmpi(mlir::arith::CmpIPredicate::sgt, len, size);
  mlir::Value invalid = b.orBit(
      b.orBit(b.orBit(offsetNegative, sizeNegative),
              b.orBit(strideInvalid, dataNull)),
      lenOverSize);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, invalid, trap,
                                 mlir::ValueRange{}, dispatch,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(dispatch);
  mlir::Value unitStride = b.cmpi(mlir::arith::CmpIPredicate::eq, stride, one);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, unitStride, contiguous,
                                 mlir::ValueRange{}, strided,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(contiguous);
  mlir::Value start = mlir::LLVM::GEPOp::create(
      b.builder, b.loc, b.ptr(), b.i8(), data, mlir::ValueRange{offset});
  mlir::func::CallOp::create(b.builder, b.loc, "write_len", mlir::TypeRange{},
                             mlir::ValueRange{fd, start, len});
  mlir::cf::BranchOp::create(b.builder, b.loc, done, mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(strided);
  mlir::Value zeroIndex =
      mlir::arith::ConstantIndexOp::create(b.builder, b.loc, 0);
  mlir::Value oneIndex =
      mlir::arith::ConstantIndexOp::create(b.builder, b.loc, 1);
  mlir::Value lenIndex = mlir::arith::IndexCastOp::create(
      b.builder, b.loc, b.builder.getIndexType(), len);
  auto loop = mlir::scf::ForOp::create(b.builder, b.loc, zeroIndex, lenIndex,
                                       oneIndex);
  {
    mlir::OpBuilder::InsertionGuard guard(b.builder);
    b.builder.setInsertionPointToStart(loop.getBody());
    mlir::Value position = mlir::arith::IndexCastOp::create(
        b.builder, b.loc, b.i64(), loop.getInductionVar());
    mlir::Value scaled =
        mlir::arith::MulIOp::create(b.builder, b.loc, position, stride);
    mlir::Value elementIndex =
        mlir::arith::AddIOp::create(b.builder, b.loc, offset, scaled);
    mlir::Value elementPtr = mlir::LLVM::GEPOp::create(
        b.builder, b.loc, b.ptr(), b.i8(), data,
        mlir::ValueRange{elementIndex});
    mlir::Value byte = mlir::LLVM::LoadOp::create(b.builder, b.loc, b.i8(),
                                                  elementPtr, /*alignment=*/1);
    mlir::func::CallOp::create(b.builder, b.loc, "write_char",
                               mlir::TypeRange{}, mlir::ValueRange{fd, byte});
  }
  mlir::cf::BranchOp::create(b.builder, b.loc, done, mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(done);
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(trap);
  mlir::func::CallOp::create(b.builder, b.loc, "abort", mlir::TypeRange{},
                             mlir::ValueRange{});
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
}

// void release_payload_slot_ptr(ptr slot): release an owned boxed container
// slot through the per-program `__ly_release_boxed_by_contract` hook (the
// manifest deallocators, generated in the user module and resolved at link).
void buildReleasePayloadSlotPtr(SupportBuilder &b) {
  auto fn = b.beginFunction(
      "release_payload_slot_ptr", b.builder.getFunctionType({b.ptr()}, {}),
      /*isPrivate=*/true);
  mlir::Block *entry = fn.addEntryBlock();
  mlir::Region &body = fn.getBody();
  mlir::Block *owned = b.builder.createBlock(&body);
  mlir::Block *done = b.builder.createBlock(&body);
  b.builder.setInsertionPointToEnd(entry);
  mlir::Value slot = entry->getArgument(0);
  mlir::Value zero = b.iconst(0);
  auto ownedWord = mlir::func::CallOp::create(
      b.builder, b.loc, "boxed_load_i64", b.i64(),
      mlir::ValueRange{slot, b.iconst(py::lowering::box_abi::kOwnedFlagWord)});
  mlir::Value notOwned = b.cmpi(mlir::arith::CmpIPredicate::eq,
                                ownedWord.getResult(0), zero);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, notOwned, done,
                                 mlir::ValueRange{}, owned,
                                 mlir::ValueRange{});
  b.builder.setInsertionPointToEnd(owned);
  auto classWord = mlir::func::CallOp::create(
      b.builder, b.loc, "boxed_load_i64", b.i64(),
      mlir::ValueRange{slot, b.iconst(1)});
  mlir::func::CallOp::create(
      b.builder, b.loc, "__ly_release_boxed_by_contract", b.i1(),
      mlir::ValueRange{slot, classWord.getResult(0)});
  mlir::cf::BranchOp::create(b.builder, b.loc, done, mlir::ValueRange{});
  b.builder.setInsertionPointToEnd(done);
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
}

// void retain_payload_slot_ptr(ptr slot): retain an owned boxed container
// slot by bumping the refcount of the boxed entity's object header (pointer
// word kPointerWordBase). Class dispatch is unnecessary — every heap header
// keeps its refcount in word 0 — so unlike the release path there is no
// per-program by-contract hook. Tagged (bit-0) headers own no memory and
// immortal headers are never written (they may live in read-only sections),
// mirroring Ly_IncRef.
void buildRetainPayloadSlotPtr(SupportBuilder &b) {
  auto fn = b.beginFunction(
      "retain_payload_slot_ptr", b.builder.getFunctionType({b.ptr()}, {}),
      /*isPrivate=*/true);
  mlir::Block *entry = fn.addEntryBlock();
  mlir::Region &body = fn.getBody();
  mlir::Block *checkHeader = b.builder.createBlock(&body);
  mlir::Block *checkImmortal = b.builder.createBlock(&body);
  mlir::Block *bump = b.builder.createBlock(&body);
  mlir::Block *done = b.builder.createBlock(&body);

  b.builder.setInsertionPointToEnd(entry);
  mlir::Value slot = entry->getArgument(0);
  mlir::Value zero = b.iconst(0);
  auto ownedWord = mlir::func::CallOp::create(
      b.builder, b.loc, "boxed_load_i64", b.i64(),
      mlir::ValueRange{slot, b.iconst(py::lowering::box_abi::kOwnedFlagWord)});
  mlir::Value notOwned = b.cmpi(mlir::arith::CmpIPredicate::eq,
                                ownedWord.getResult(0), zero);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, notOwned, done,
                                 mlir::ValueRange{}, checkHeader,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(checkHeader);
  auto headerWord = mlir::func::CallOp::create(
      b.builder, b.loc, "boxed_load_i64", b.i64(),
      mlir::ValueRange{slot,
                       b.iconst(py::lowering::box_abi::kPointerWordBase)});
  mlir::Value header = headerWord.getResult(0);
  mlir::Value isNull =
      b.cmpi(mlir::arith::CmpIPredicate::eq, header, zero);
  mlir::Value tagBit = mlir::arith::AndIOp::create(b.builder, b.loc, header,
                                                   b.iconst(1));
  mlir::Value isTagged =
      b.cmpi(mlir::arith::CmpIPredicate::ne, tagBit, zero);
  mlir::Value skip = b.orBit(isNull, isTagged);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, skip, done,
                                 mlir::ValueRange{}, checkImmortal,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(checkImmortal);
  mlir::Value headerPtr = b.intToPtr(header);
  mlir::Value observed =
      mlir::LLVM::LoadOp::create(b.builder, b.loc, b.i64(), headerPtr)
          .getResult();
  mlir::Value immortal =
      b.iconst(std::numeric_limits<std::int64_t>::max());
  mlir::Value isImmortal =
      b.cmpi(mlir::arith::CmpIPredicate::eq, observed, immortal);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, isImmortal, done,
                                 mlir::ValueRange{}, bump, mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(bump);
  mlir::LLVM::AtomicRMWOp::create(b.builder, b.loc,
                                  mlir::LLVM::AtomicBinOp::add, headerPtr,
                                  b.iconst(1),
                                  mlir::LLVM::AtomicOrdering::acq_rel);
  mlir::cf::BranchOp::create(b.builder, b.loc, done, mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(done);
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
}

// LyObject_ReleaseBoxedPayloadRaw(memref<16xi64>) /
// LyObject_ReleaseBoxedPayloadArraySlotRaw(memref<?xi64>, i64): shared-ABI
// wrappers the lib manifests call to release a boxed slot (whole box, or the
// index-th 16-word slot of an items array).
void buildReleaseBoxedPayloadRaw(SupportBuilder &b) {
  auto boxType = mlir::MemRefType::get(
      {py::lowering::box_abi::kWordsPerBox}, b.i64());
  auto fn = b.beginFunction(
      "LyObject_ReleaseBoxedPayloadRaw",
      b.builder.getFunctionType({boxType}, {}));
  mlir::Block *entry = fn.addEntryBlock();
  b.builder.setInsertionPointToEnd(entry);
  mlir::Value pointerIndex =
      mlir::memref::ExtractAlignedPointerAsIndexOp::create(
          b.builder, b.loc, entry->getArgument(0));
  mlir::Value pointerWord = mlir::arith::IndexCastOp::create(
      b.builder, b.loc, b.i64(), pointerIndex);
  mlir::Value slot = b.intToPtr(pointerWord);
  mlir::func::CallOp::create(b.builder, b.loc, "release_payload_slot_ptr",
                             mlir::TypeRange{}, mlir::ValueRange{slot});
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
}

// LyObject_{Retain,Release}BoxedPayloadArraySlotRaw(memref<?xi64>, i64):
// shared-ABI wrappers the lib manifests call on the index-th 16-word slot of
// an items array (a container copy duplicating element boxes, or dropping
// them). The two differ only in the payload-slot helper they forward to.
void buildBoxedPayloadArraySlotRaw(SupportBuilder &b, llvm::StringRef name,
                                   llvm::StringRef callee) {
  auto itemsType = mlir::MemRefType::get({mlir::ShapedType::kDynamic}, b.i64());
  auto fn = b.beginFunction(
      name, b.builder.getFunctionType({itemsType, b.i64()}, {}));
  mlir::Block *entry = fn.addEntryBlock();
  b.builder.setInsertionPointToEnd(entry);
  mlir::Value pointerIndex =
      mlir::memref::ExtractAlignedPointerAsIndexOp::create(
          b.builder, b.loc, entry->getArgument(0));
  mlir::Value pointerWord = mlir::arith::IndexCastOp::create(
      b.builder, b.loc, b.i64(), pointerIndex);
  mlir::Value base = b.intToPtr(pointerWord);
  mlir::Value wordOffset = mlir::arith::MulIOp::create(
      b.builder, b.loc, entry->getArgument(1), b.iconst(16));
  mlir::Value slot = b.gepI64(base, wordOffset);
  mlir::func::CallOp::create(b.builder, b.loc, callee, mlir::TypeRange{},
                             mlir::ValueRange{slot});
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
}

void declareLLVMExternal(SupportBuilder &b, llvm::StringRef name,
                         mlir::Type result, llvm::ArrayRef<mlir::Type> inputs,
                         bool isVarArg = false) {
  if (b.module.lookupSymbol(name))
    return;
  mlir::OpBuilder::InsertionGuard guard(b.builder);
  b.builder.setInsertionPointToEnd(b.module.getBody());
  mlir::Type resultType =
      result ? result : mlir::LLVM::LLVMVoidType::get(b.builder.getContext());
  mlir::LLVM::LLVMFuncOp::create(
      b.builder, b.loc, name,
      mlir::LLVM::LLVMFunctionType::get(resultType, inputs, isVarArg));
}

void declareEHSupport(SupportBuilder &b) {
  declareLLVMExternal(b, "LyRt_InstallStackGuard", {}, {});
  declareLLVMExternal(b, "__cxa_allocate_exception", b.ptr(), {b.i64()});
  declareLLVMExternal(b, "__cxa_throw", {}, {b.ptr(), b.ptr(), b.ptr()});
  declareLLVMExternal(b, "__cxa_begin_catch", b.ptr(), {b.ptr()});
  declareLLVMExternal(b, "__cxa_end_catch", {}, {});
  declareLLVMExternal(b, "__gxx_personality_v0", b.i32(), {},
                      /*isVarArg=*/true);

  mlir::OpBuilder::InsertionGuard guard(b.builder);
  b.builder.setInsertionPointToEnd(b.module.getBody());
  auto boolGlobal = [&](llvm::StringRef name) {
    auto global = mlir::LLVM::GlobalOp::create(
        b.builder, b.loc, b.i1(), /*isConstant=*/false,
        mlir::LLVM::Linkage::Internal, name,
        b.builder.getIntegerAttr(b.i1(), 0), /*alignment=*/4);
    global.setDsoLocal(true);
  };
  boolGlobal("g_current_exception");
  boolGlobal("g_native_catch_active");
  // The in-flight exception. `ExceptionParts` -- so the three descriptors are
  // stored as descriptors and their pointer members round-trip as pointers.
  //
  // ⛔ Do not reach into this global by word offset. It is 120 contiguous bytes
  // and an `i64` view of it type-checks, which is how the star and chain-node
  // paths used to read it; every such read has to turn the aligned pointer back
  // into a pointer to be useful, and that is the direction the memory model
  // refuses (`extract_aligned_pointer_as_index` is documented there as where
  // provenance is lost). `partsField` gives the same slot with its real type.
  {
    auto parts = mlir::LLVM::GlobalOp::create(
        b.builder, b.loc, exceptionPartsType(b), /*isConstant=*/false,
        mlir::LLVM::Linkage::Internal, "g_current_parts", mlir::Attribute(),
        /*alignment=*/8);
    parts.setDsoLocal(true);
    mlir::OpBuilder::InsertionGuard initGuard(b.builder);
    mlir::Block *init = b.builder.createBlock(&parts.getInitializerRegion());
    b.builder.setInsertionPointToEnd(init);
    mlir::Value zero =
        mlir::LLVM::ZeroOp::create(b.builder, b.loc, exceptionPartsType(b));
    mlir::LLVM::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{zero});
  }
  b.stringGlobal(".native_exception",
                 "error: uncaught native exception during Python execution\n");

  // Itanium typeinfo for LyPythonException: vtable slot from the C++ ABI's
  // __class_type_info, name from the mangled-string global.
  mlir::LLVM::GlobalOp::create(
      b.builder, b.loc, mlir::LLVM::LLVMArrayType::get(b.ptr(), 0),
      /*isConstant=*/false, mlir::LLVM::Linkage::External,
      "_ZTVN10__cxxabiv117__class_type_infoE", mlir::Attribute());
  {
    std::string mangled = "17LyPythonException";
    mangled.push_back('\0');
    auto nameType = mlir::LLVM::LLVMArrayType::get(b.i8(), mangled.size());
    auto nameGlobal = mlir::LLVM::GlobalOp::create(
        b.builder, b.loc, nameType, /*isConstant=*/true,
        mlir::LLVM::Linkage::LinkonceODR, "_ZTS17LyPythonException",
        b.builder.getStringAttr(mangled), /*alignment=*/1);
    nameGlobal.setDsoLocal(true);
    nameGlobal.setVisibility_(mlir::LLVM::Visibility::Hidden);
  }
  {
    auto typeInfoType = mlir::LLVM::LLVMStructType::getLiteral(
        b.builder.getContext(), {b.ptr(), b.ptr()});
    auto typeInfo = mlir::LLVM::GlobalOp::create(
        b.builder, b.loc, typeInfoType, /*isConstant=*/true,
        mlir::LLVM::Linkage::LinkonceODR, "_ZTI17LyPythonException",
        mlir::Attribute(), /*alignment=*/8);
    typeInfo.setDsoLocal(true);
    typeInfo.setVisibility_(mlir::LLVM::Visibility::Hidden);
    mlir::OpBuilder::InsertionGuard initGuard(b.builder);
    mlir::Block *init = b.builder.createBlock(&typeInfo.getInitializerRegion());
    b.builder.setInsertionPointToEnd(init);
    mlir::Value name = b.addrOf("_ZTS17LyPythonException");
    mlir::Value vtable = b.addrOf("_ZTVN10__cxxabiv117__class_type_infoE");
    mlir::Value vtableEntry = mlir::LLVM::GEPOp::create(
        b.builder, b.loc, b.ptr(), b.ptr(), vtable,
        llvm::ArrayRef<mlir::LLVM::GEPArg>{mlir::LLVM::GEPArg(2)},
        mlir::LLVM::GEPNoWrapFlags::inbounds);
    mlir::Value undef =
        mlir::LLVM::UndefOp::create(b.builder, b.loc, typeInfoType);
    mlir::Value withVtable = mlir::LLVM::InsertValueOp::create(
        b.builder, b.loc, undef, vtableEntry, llvm::ArrayRef<std::int64_t>{0});
    mlir::Value complete = mlir::LLVM::InsertValueOp::create(
        b.builder, b.loc, withVtable, name, llvm::ArrayRef<std::int64_t>{1});
    mlir::LLVM::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{complete});
  }
}

mlir::LLVM::LLVMFuncOp beginLLVMFunction(SupportBuilder &b,
                                         llvm::StringRef name,
                                         mlir::Type result,
                                         llvm::ArrayRef<mlir::Type> inputs) {
  mlir::OpBuilder::InsertionGuard guard(b.builder);
  b.builder.setInsertionPointToEnd(b.module.getBody());
  mlir::Type resultType =
      result ? result : mlir::LLVM::LLVMVoidType::get(b.builder.getContext());
  return mlir::LLVM::LLVMFuncOp::create(
      b.builder, b.loc, name,
      mlir::LLVM::LLVMFunctionType::get(resultType, inputs, false));
}

void emitLLVMTrap(SupportBuilder &b) {
  mlir::func::CallOp::create(b.builder, b.loc, "abort", mlir::TypeRange{},
                             mlir::ValueRange{});
  mlir::LLVM::UnreachableOp::create(b.builder, b.loc);
}

// i64 current_exception_class_id_unchecked(): class id word of the stored
// exception header (aligned[offset + 2*stride]); aborts on a null header.
void buildCurrentExceptionClassIdUnchecked(SupportBuilder &b) {
  auto fn = b.beginFunction("current_exception_class_id_unchecked",
                            b.builder.getFunctionType({}, {b.i64()}),
                            /*isPrivate=*/true);
  mlir::Block *entry = fn.addEntryBlock();
  mlir::Region &body = fn.getBody();
  mlir::Block *load = b.builder.createBlock(&body);
  mlir::Block *trap = b.builder.createBlock(&body);
  b.builder.setInsertionPointToEnd(entry);
  mlir::Value parts = b.addrOf("g_current_parts");
  mlir::Value aligned = b.loadPtrVal(partsField(b, parts, 0, 1));
  mlir::Value offset = mlir::LLVM::LoadOp::create(
      b.builder, b.loc, b.i64(), partsField(b, parts, 0, 2), /*alignment=*/8);
  mlir::Value stride = mlir::LLVM::LoadOp::create(
      b.builder, b.loc, b.i64(), partsField(b, parts, 0, 4), /*alignment=*/8);
  mlir::Value missing = b.ptrEq(aligned, b.nullPtr());
  mlir::cf::CondBranchOp::create(b.builder, b.loc, missing, trap,
                                 mlir::ValueRange{}, load, mlir::ValueRange{});
  b.builder.setInsertionPointToEnd(load);
  mlir::Value scaled =
      mlir::arith::MulIOp::create(b.builder, b.loc, stride, b.iconst(2));
  mlir::Value index =
      mlir::arith::AddIOp::create(b.builder, b.loc, offset, scaled);
  mlir::Value classId = b.loadI64(b.gepI64(aligned, index));
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{classId});
  b.builder.setInsertionPointToEnd(trap);
  b.emitTrap(b.i64());
}

// void end_native_catch_if_active(): closes the __cxa catch scope opened by
// LyEH_BeginCatch, once.
void buildEndNativeCatchIfActive(SupportBuilder &b) {
  auto fn = b.beginFunction("end_native_catch_if_active",
                            b.builder.getFunctionType({}, {}),
                            /*isPrivate=*/true);
  mlir::Block *entry = fn.addEntryBlock();
  b.builder.setInsertionPointToEnd(entry);
  mlir::Value flagSlot = b.addrOf("g_native_catch_active");
  mlir::Value active = mlir::LLVM::LoadOp::create(b.builder, b.loc, b.i1(),
                                                  flagSlot, /*alignment=*/4);
  auto endIf = mlir::scf::IfOp::create(b.builder, b.loc, mlir::TypeRange{},
                                       active, /*withElseRegion=*/false);
  {
    mlir::OpBuilder::InsertionGuard guard(b.builder);
    b.builder.setInsertionPointToStart(&endIf.getThenRegion().front());
    mlir::LLVM::CallOp::create(b.builder, b.loc, mlir::TypeRange{},
                               "__cxa_end_catch", mlir::ValueRange{});
    mlir::LLVM::StoreOp::create(
        b.builder, b.loc,
        mlir::arith::ConstantIntOp::create(b.builder, b.loc, 0, 1).getResult(),
        flagSlot, /*alignment=*/4);
  }
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
}

// LyEH_ThrowException(exception triple): stores the payload in the process
// slot, then throws the 1-byte C++ carrier. A still-pending exception (a raise
// while another exception is handled that did not go through the lowering's
// explicit stash — e.g. a runtime-internal raise) becomes the new exception's
// implicit __context__.
void buildThrowException(SupportBuilder &b) {
  auto fn = b.beginFunction(
      "LyEH_ThrowException",
      b.builder.getFunctionType(exceptionTripleTypes(b.builder), {}));
  mlir::Block *entry = fn.addEntryBlock();
  mlir::Region &body = fn.getBody();
  mlir::Block *stash = b.builder.createBlock(&body);
  mlir::Block *store = b.builder.createBlock(&body);
  b.builder.setInsertionPointToEnd(entry);
  mlir::Value flagSlot = b.addrOf("g_current_exception");
  mlir::Value pending = mlir::LLVM::LoadOp::create(b.builder, b.loc, b.i1(),
                                                   flagSlot, /*alignment=*/4);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, pending, stash,
                                 mlir::ValueRange{}, store,
                                 mlir::ValueRange{});
  b.builder.setInsertionPointToEnd(stash);
  b.call("LyEH_StashCurrentAsContext", mlir::TypeRange{}, {});
  mlir::cf::BranchOp::create(b.builder, b.loc, store, mlir::ValueRange{});
  b.builder.setInsertionPointToEnd(store);
  storeExceptionTriple(b, b.addrOf("g_current_parts"), entry->getArguments());
  mlir::LLVM::StoreOp::create(
      b.builder, b.loc,
      mlir::arith::ConstantIntOp::create(b.builder, b.loc, 1, 1).getResult(),
      flagSlot, /*alignment=*/4);
  auto carrier = mlir::LLVM::CallOp::create(
      b.builder, b.loc, mlir::TypeRange{b.ptr()}, "__cxa_allocate_exception",
      mlir::ValueRange{b.iconst(1)});
  mlir::LLVM::CallOp::create(
      b.builder, b.loc, mlir::TypeRange{}, "__cxa_throw",
      mlir::ValueRange{carrier.getResult(),
                       b.addrOf("_ZTI17LyPythonException"), b.nullPtr()});
  // `__cxa_throw` does not return; the trailing return only satisfies the
  // verifier, which is why this is not `llvm.unreachable`.
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
}

// LyEH_BeginCatch(ptr exceptionObject): opens the __cxa catch scope for a
// pending Python exception.
void buildBeginCatch(SupportBuilder &b) {
  auto fn = beginLLVMFunction(b, "LyEH_BeginCatch", {}, {b.ptr()});
  mlir::Block *entry = fn.addEntryBlock(b.builder);
  mlir::Region &body = fn.getBody();
  mlir::Block *begin = b.builder.createBlock(&body);
  mlir::Block *trap = b.builder.createBlock(&body);
  b.builder.setInsertionPointToEnd(entry);
  mlir::Value objectNull = b.ptrEq(entry->getArgument(0), b.nullPtr());
  mlir::Value pending = mlir::LLVM::LoadOp::create(
      b.builder, b.loc, b.i1(), b.addrOf("g_current_exception"),
      /*alignment=*/4);
  mlir::Value notPending = mlir::arith::XOrIOp::create(
      b.builder, b.loc, pending,
      mlir::arith::ConstantIntOp::create(b.builder, b.loc, 1, 1).getResult());
  mlir::Value invalid =
      mlir::arith::OrIOp::create(b.builder, b.loc, objectNull, notPending);
  mlir::LLVM::CondBrOp::create(b.builder, b.loc, invalid, trap, begin);
  b.builder.setInsertionPointToEnd(begin);
  mlir::LLVM::CallOp::create(b.builder, b.loc, mlir::TypeRange{b.ptr()},
                             "__cxa_begin_catch",
                             mlir::ValueRange{entry->getArgument(0)});
  mlir::LLVM::StoreOp::create(
      b.builder, b.loc,
      mlir::arith::ConstantIntOp::create(b.builder, b.loc, 1, 1).getResult(),
      b.addrOf("g_native_catch_active"), /*alignment=*/4);
  mlir::LLVM::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
  b.builder.setInsertionPointToEnd(trap);
  emitLLVMTrap(b);
}

// (memref<3xi64>, memref<2xi64>, memref<?xi8>) LyEH_BorrowCurrentException():
// the stored payload as three borrowed views. Unlike the star paths' views,
// these carry the stored descriptor whole -- offset and stride included --
// because there is nothing here to reconstruct them from.
void buildBorrowCurrentException(SupportBuilder &b) {
  auto fn = b.beginFunction(
      "LyEH_BorrowCurrentException",
      b.builder.getFunctionType({}, exceptionTripleTypes(b.builder)));
  mlir::Block *entry = fn.addEntryBlock();
  mlir::Region &body = fn.getBody();
  mlir::Block *borrow = b.builder.createBlock(&body);
  mlir::Block *trap = b.builder.createBlock(&body);
  b.builder.setInsertionPointToEnd(entry);
  mlir::Value pending = mlir::LLVM::LoadOp::create(
      b.builder, b.loc, b.i1(), b.addrOf("g_current_exception"),
      /*alignment=*/4);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, pending, borrow,
                                 mlir::ValueRange{}, trap, mlir::ValueRange{});
  b.builder.setInsertionPointToEnd(borrow);
  mlir::Value parts = b.addrOf("g_current_parts");
  llvm::SmallVector<mlir::Type, 3> types = exceptionTripleTypes(b.builder);
  llvm::SmallVector<mlir::Value, 3> results;
  for (std::int32_t section = 0; section < 3; ++section)
    results.push_back(buildMemRef1D(
        b, types[section],
        MemRef1DParts{b.loadPtrVal(partsField(b, parts, section, 0)),
                      b.loadPtrVal(partsField(b, parts, section, 1)),
                      b.loadI64(partsField(b, parts, section, 2)),
                      b.loadI64(partsField(b, parts, section, 3)),
                      b.loadI64(partsField(b, parts, section, 4))}));
  mlir::func::ReturnOp::create(b.builder, b.loc, results);
  b.builder.setInsertionPointToEnd(trap);
  // Three poison results, not one: `emitTrap` returns a single value.
  mlir::func::CallOp::create(b.builder, b.loc, "abort", mlir::TypeRange{},
                             mlir::ValueRange{});
  llvm::SmallVector<mlir::Value, 3> poison;
  for (mlir::Type type : types)
    poison.push_back(mlir::ub::PoisonOp::create(b.builder, b.loc, type,
                                                nullptr));
  mlir::func::ReturnOp::create(b.builder, b.loc, poison);
}

void buildCurrentExceptionClassId(SupportBuilder &b) {
  auto fn = b.beginFunction("LyEH_CurrentExceptionClassId",
                            b.builder.getFunctionType({}, {b.i64()}));
  mlir::Block *entry = fn.addEntryBlock();
  b.builder.setInsertionPointToEnd(entry);
  mlir::Value pending = mlir::LLVM::LoadOp::create(
      b.builder, b.loc, b.i1(), b.addrOf("g_current_exception"),
      /*alignment=*/4);
  auto classIf = mlir::scf::IfOp::create(b.builder, b.loc,
                                         mlir::TypeRange{b.i64()}, pending,
                                         /*withElseRegion=*/true);
  {
    mlir::OpBuilder::InsertionGuard guard(b.builder);
    b.builder.setInsertionPointToStart(&classIf.getThenRegion().front());
    mlir::Value classId =
        b.call("current_exception_class_id_unchecked", b.i64(), {}).front();
    mlir::scf::YieldOp::create(b.builder, b.loc, mlir::ValueRange{classId});
    b.builder.setInsertionPointToStart(&classIf.getElseRegion().front());
    mlir::scf::YieldOp::create(b.builder, b.loc, mlir::ValueRange{b.iconst(0)});
  }
  mlir::func::ReturnOp::create(b.builder, b.loc,
                               mlir::ValueRange{classIf.getResult(0)});
}

void buildCurrentExceptionMatches(SupportBuilder &b) {
  auto fn = b.beginFunction("LyEH_CurrentExceptionMatches",
                            b.builder.getFunctionType({b.i64()}, {b.i1()}));
  mlir::Block *entry = fn.addEntryBlock();
  b.builder.setInsertionPointToEnd(entry);
  mlir::Value raised =
      b.call("LyEH_CurrentExceptionClassId", b.i64(), {}).front();
  mlir::Value matches =
      b.call("LyEH_ClassIdMatches", b.i1(),
             mlir::ValueRange{raised, entry->getArgument(0)})
          .front();
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{matches});
}

// void release_exception_extras(i64 header): drop the extended exception
// words before the 3-word storage itself is freed — word 3 (ExceptionGroup
// members / multi-value args) and word 4 (user-exception fields) each hold a
// [count, count x box16] i64 block whose slots own one reference apiece.
// Lives here (not in builtins.mlir) because every raw free site — the
// manifest deallocator, the discard path, and chain-node destruction — must
// share one implementation.
void buildReleaseExceptionExtras(SupportBuilder &b) {
  auto fn = b.beginFunction("release_exception_extras",
                            b.builder.getFunctionType({b.ptr()}, {}),
                            /*isPrivate=*/true);
  mlir::Block *entry = fn.addEntryBlock();
  b.builder.setInsertionPointToEnd(entry);
  mlir::Value header = entry->getArgument(0);
  for (std::int64_t word : {std::int64_t(3), std::int64_t(4)}) {
    mlir::Value slotPtr = b.gepI64(header, b.iconst(word));
    mlir::Value block64 = b.loadI64(slotPtr);
    mlir::Value present =
        b.cmpi(mlir::arith::CmpIPredicate::ne, block64, b.iconst(0));
    auto blockIf = mlir::scf::IfOp::create(b.builder, b.loc, mlir::TypeRange{},
                                           present, /*withElseRegion=*/false);
    {
      mlir::OpBuilder::InsertionGuard guard(b.builder);
      b.builder.setInsertionPointToStart(&blockIf.getThenRegion().front());
      mlir::Value blockPtr = b.intToPtr(block64);
      // Bit 62 of the count word is the tuple-repr flag.
      mlir::Value rawCount = b.loadI64(blockPtr);
      mlir::Value count = mlir::arith::AndIOp::create(
          b.builder, b.loc, rawCount, b.iconst(0x3FFFFFFFFFFFFFFFLL));
      mlir::Value zeroIndex =
          mlir::arith::ConstantIndexOp::create(b.builder, b.loc, 0);
      mlir::Value oneIndex =
          mlir::arith::ConstantIndexOp::create(b.builder, b.loc, 1);
      mlir::Value countIndex = mlir::arith::IndexCastOp::create(
          b.builder, b.loc, b.builder.getIndexType(), count);
      auto loop = mlir::scf::ForOp::create(b.builder, b.loc, zeroIndex,
                                           countIndex, oneIndex);
      {
        mlir::OpBuilder::InsertionGuard loopGuard(b.builder);
        b.builder.setInsertionPointToStart(loop.getBody());
        mlir::Value position = mlir::arith::IndexCastOp::create(
            b.builder, b.loc, b.i64(), loop.getInductionVar());
        mlir::Value boxWords = mlir::arith::MulIOp::create(
            b.builder, b.loc, position, b.iconst(16));
        mlir::Value boxBase = mlir::arith::AddIOp::create(
            b.builder, b.loc, boxWords, b.iconst(1));
        mlir::Value boxPtr = b.gepI64(blockPtr, boxBase);
        b.call("release_payload_slot_ptr", mlir::TypeRange{},
               mlir::ValueRange{boxPtr});
      }
      b.call("free", mlir::TypeRange{}, mlir::ValueRange{blockPtr});
      mlir::LLVM::StoreOp::create(b.builder, b.loc, b.iconst(0), slotPtr,
                                  /*alignment=*/8);
    }
  }
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
}

// void LyEH_DiscardCurrentException(): consumes the stored exception token
// (refcount decrement; frees message + header at zero), clears the slot and
// the traceback. An escaping handler binding was retained by the
// borrowed-return machinery, so its token survives this release.
//
// Chaining: the discarded exception's __cause__ node is released with it; a
// __context__ node is *restored* as the pending exception instead — handler
// completion returns to handling the outer exception (CPython's exception
// stack pop), so a bare `raise` after a nested try re-raises the right one.
void buildDiscardCurrentException(SupportBuilder &b) {
  auto fn = b.beginFunction("LyEH_DiscardCurrentException",
                            b.builder.getFunctionType({}, {}));
  mlir::Block *entry = fn.addEntryBlock();
  mlir::Region &body = fn.getBody();
  mlir::Block *release = b.builder.createBlock(&body);
  mlir::Block *freeBlocks = b.builder.createBlock(&body);
  mlir::Block *clear = b.builder.createBlock(&body);
  mlir::Block *restore = b.builder.createBlock(&body);
  mlir::Block *finish = b.builder.createBlock(&body);
  b.builder.setInsertionPointToEnd(entry);
  b.call("end_native_catch_if_active", mlir::TypeRange{}, {});
  mlir::Value causeSlot = b.addrOf("g_exc_cause_node");
  b.call("release_chain_node", mlir::TypeRange{},
         mlir::ValueRange{b.loadPtrVal(causeSlot)});
  mlir::LLVM::StoreOp::create(b.builder, b.loc, b.nullPtr(), causeSlot,
                              /*alignment=*/8);
  mlir::Value flagSlot = b.addrOf("g_current_exception");
  mlir::Value parts = b.addrOf("g_current_parts");
  mlir::Value pending = mlir::LLVM::LoadOp::create(b.builder, b.loc, b.i1(),
                                                   flagSlot, /*alignment=*/4);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, pending, release,
                                 mlir::ValueRange{}, clear,
                                 mlir::ValueRange{});
  b.builder.setInsertionPointToEnd(release);
  mlir::Value exceptionAligned = b.loadPtrVal(partsField(b, parts, 0, 1));
  mlir::Value messageHeader = b.loadPtrVal(partsField(b, parts, 1, 1));
  mlir::Value messageBytes = b.loadPtrVal(partsField(b, parts, 2, 1));
  mlir::Value becameZero = b.call("release_storage_raw_to_zero", b.i1(),
                                  mlir::ValueRange{exceptionAligned})
                               .front();
  mlir::cf::CondBranchOp::create(b.builder, b.loc, becameZero, freeBlocks,
                                 mlir::ValueRange{}, clear,
                                 mlir::ValueRange{});
  b.builder.setInsertionPointToEnd(freeBlocks);
  b.call("release_exception_extras", mlir::TypeRange{},
         mlir::ValueRange{exceptionAligned});
  b.call("release_unicode_raw", mlir::TypeRange{},
         mlir::ValueRange{messageHeader, messageBytes});
  b.call("free", mlir::TypeRange{}, mlir::ValueRange{exceptionAligned});
  mlir::cf::BranchOp::create(b.builder, b.loc, clear, mlir::ValueRange{});
  b.builder.setInsertionPointToEnd(clear);
  mlir::LLVM::StoreOp::create(
      b.builder, b.loc,
      mlir::arith::ConstantIntOp::create(b.builder, b.loc, 0, 1).getResult(),
      flagSlot, /*alignment=*/4);
  clearExceptionParts(b, parts);
  b.call("LyTraceback_Clear", mlir::TypeRange{}, {});
  mlir::Value contextSlot = b.addrOf("g_exc_context_node");
  mlir::Value node = b.loadPtrVal(contextSlot);
  mlir::Value haveContext = b.ptrNe(node, b.nullPtr());
  mlir::cf::CondBranchOp::create(b.builder, b.loc, haveContext, restore,
                                 mlir::ValueRange{}, finish,
                                 mlir::ValueRange{});

  // Destructive restore: after the cause release above the node has exactly
  // one owner (only the discarded exception could have shared it), so its
  // members move back into the globals and only the shell is freed.
  b.builder.setInsertionPointToEnd(restore);
  mlir::LLVM::StoreOp::create(b.builder, b.loc,
                              b.loadPtrVal(nodeMember(b, node, kNodeCause)),
                              causeSlot, /*alignment=*/8);
  mlir::LLVM::StoreOp::create(b.builder, b.loc,
                              b.loadPtrVal(nodeMember(b, node, kNodeContext)),
                              contextSlot, /*alignment=*/8);
  mlir::LLVM::StoreOp::create(b.builder, b.loc,
                              b.loadI64(nodeMember(b, node, kNodeSuppress)),
                              b.addrOf("g_exc_suppress_context"),
                              /*alignment=*/8);
  storeExceptionParts(b, parts,
                      loadExceptionParts(b, nodeMember(b, node, kNodePayload)));
  mlir::LLVM::StoreOp::create(
      b.builder, b.loc,
      mlir::arith::ConstantIntOp::create(b.builder, b.loc, 1, 1).getResult(),
      flagSlot, /*alignment=*/4);
  mlir::Value frames = b.loadPtrVal(nodeMember(b, node, kNodeFrames));
  mlir::Value count = b.loadI64(nodeMember(b, node, kNodeFrameCount));
  mlir::Value haveFrames =
      b.cmpi(mlir::arith::CmpIPredicate::sgt, count, b.iconst(0));
  auto framesIf = mlir::scf::IfOp::create(b.builder, b.loc, mlir::TypeRange{},
                                          haveFrames, /*withElseRegion=*/false);
  {
    mlir::OpBuilder::InsertionGuard guard(b.builder);
    b.builder.setInsertionPointToStart(&framesIf.getThenRegion().front());
    mlir::Value bytes =
        mlir::arith::MulIOp::create(b.builder, b.loc, count, b.iconst(40));
    mlir::LLVM::MemcpyOp::create(b.builder, b.loc,
                                 b.addrOf("g_traceback_stack"), frames, bytes,
                                 /*isVolatile=*/false);
    mlir::LLVM::StoreOp::create(b.builder, b.loc, count,
                                b.addrOf("g_traceback_size"), /*alignment=*/8);
  }
  // free(), not free_raw_i64_ptr(): the null guard that wrapper exists for is
  // free()'s own contract, and going through it would mean handing the node a
  // word back.
  b.call("free", mlir::TypeRange{}, mlir::ValueRange{frames});
  freeSoleChainNode(b, node, "LyEH_DiscardCurrentException");
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(finish);
  mlir::LLVM::StoreOp::create(b.builder, b.loc, b.iconst(0),
                              b.addrOf("g_exc_suppress_context"),
                              /*alignment=*/8);
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
}

// LyEH_RethrowCurrent(): rethrows the still-stored payload with a fresh C++
// carrier (the previous catch scope is closed first).
void buildRethrowCurrent(SupportBuilder &b) {
  auto fn = beginLLVMFunction(b, "LyEH_RethrowCurrent", {}, {});
  mlir::Block *entry = fn.addEntryBlock(b.builder);
  mlir::Region &body = fn.getBody();
  mlir::Block *rethrow = b.builder.createBlock(&body);
  mlir::Block *trap = b.builder.createBlock(&body);
  b.builder.setInsertionPointToEnd(entry);
  mlir::Value pending = mlir::LLVM::LoadOp::create(
      b.builder, b.loc, b.i1(), b.addrOf("g_current_exception"),
      /*alignment=*/4);
  mlir::LLVM::CondBrOp::create(b.builder, b.loc, pending, rethrow, trap);
  b.builder.setInsertionPointToEnd(rethrow);
  mlir::func::CallOp::create(b.builder, b.loc, "end_native_catch_if_active",
                             mlir::TypeRange{}, mlir::ValueRange{});
  auto carrier = mlir::LLVM::CallOp::create(
      b.builder, b.loc, mlir::TypeRange{b.ptr()}, "__cxa_allocate_exception",
      mlir::ValueRange{b.iconst(1)});
  mlir::LLVM::CallOp::create(
      b.builder, b.loc, mlir::TypeRange{}, "__cxa_throw",
      mlir::ValueRange{carrier.getResult(),
                       b.addrOf("_ZTI17LyPythonException"), b.nullPtr()});
  mlir::LLVM::UnreachableOp::create(b.builder, b.loc);
  b.builder.setInsertionPointToEnd(trap);
  emitLLVMTrap(b);
}

// i1 LyEH_TakeCurrentDescriptor(ptr out): moves the stored payload into the
// caller's ExceptionParts buffer and clears the slot.
void buildTakeCurrentDescriptor(SupportBuilder &b) {
  auto fn = b.beginFunction("LyEH_TakeCurrentDescriptor",
                            b.builder.getFunctionType({b.ptr()}, {b.i1()}));
  mlir::Block *entry = fn.addEntryBlock();
  mlir::Region &body = fn.getBody();
  mlir::Block *take = b.builder.createBlock(&body);
  mlir::Block *miss = b.builder.createBlock(&body);
  b.builder.setInsertionPointToEnd(entry);
  mlir::Value outOk = b.ptrNe(entry->getArgument(0), b.nullPtr());
  mlir::Value pending = mlir::LLVM::LoadOp::create(
      b.builder, b.loc, b.i1(), b.addrOf("g_current_exception"),
      /*alignment=*/4);
  mlir::Value usable =
      mlir::arith::AndIOp::create(b.builder, b.loc, outOk, pending);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, usable, take,
                                 mlir::ValueRange{}, miss, mlir::ValueRange{});
  b.builder.setInsertionPointToEnd(take);
  b.call("end_native_catch_if_active", mlir::TypeRange{}, {});
  mlir::Value parts = b.addrOf("g_current_parts");
  storeExceptionParts(b, entry->getArgument(0), loadExceptionParts(b, parts));
  clearExceptionParts(b, parts);
  mlir::LLVM::StoreOp::create(
      b.builder, b.loc,
      mlir::arith::ConstantIntOp::create(b.builder, b.loc, 0, 1).getResult(),
      b.addrOf("g_current_exception"), /*alignment=*/4);
  mlir::func::ReturnOp::create(
      b.builder, b.loc,
      mlir::ValueRange{
          mlir::arith::ConstantIntOp::create(b.builder, b.loc, 1, 1)
              .getResult()});
  b.builder.setInsertionPointToEnd(miss);
  mlir::func::ReturnOp::create(
      b.builder, b.loc,
      mlir::ValueRange{
          mlir::arith::ConstantIntOp::create(b.builder, b.loc, 0, 1)
              .getResult()});
}

// Generator suspension EH stash (rfc/stdlib-semantics.md R3 / item: TLS
// token save/restore). The process exception slot is a single token; a
// generator body suspended INSIDE an exception handler still owns its
// in-flight token, which must not occupy the slot while unrelated code runs
// between resumptions. The token's identity spans more than the 120-byte
// parts: its traceback snapshot and __cause__/__context__ chain live in
// separate globals, and leaving those in place while other code raises
// would misattach them to an unrelated exception (whose handler would then
// destructively "restore" them as a phantom pending token). A stash
// therefore parks the WHOLE identity as one exception chain node
// (TracebackSupportBuilder layout); the area holds just that node's
// address in word 0 (0 = empty; the remaining words of the 16-word areas
// stay reserved). Why not memcpy the globals into the area: the traceback
// snapshot is variable-length, and LyEH_StashCurrentAsContext already
// packages exactly this state. Areas live in the generator storage (word
// 48) for the suspended body's token, and in a resume driver's stack frame
// for the resumer's own context.

// void LyEH_StashCurrentException(ptr cell): move the pending token (if any)
// out of the process slot (parts + traceback + chain globals) into a chain
// node parked in the cell. Closes the native catch scope like any other slot
// consumer.
void buildStashCurrentException(SupportBuilder &b) {
  auto fn = b.beginFunction("LyEH_StashCurrentException",
                            b.builder.getFunctionType({b.ptr()}, {}));
  mlir::Block *entry = fn.addEntryBlock();
  mlir::Region &body = fn.getBody();
  mlir::Block *park = b.builder.createBlock(&body);
  mlir::Block *empty = b.builder.createBlock(&body);
  mlir::Block *done = b.builder.createBlock(&body);
  b.builder.setInsertionPointToEnd(entry);
  mlir::Value areaPtr = entry->getArgument(0);
  mlir::Value pending = mlir::LLVM::LoadOp::create(
      b.builder, b.loc, b.i1(), b.addrOf("g_current_exception"),
      /*alignment=*/4);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, pending, park,
                                 mlir::ValueRange{}, empty,
                                 mlir::ValueRange{});
  b.builder.setInsertionPointToEnd(park);
  b.call("LyEH_StashCurrentAsContext", mlir::TypeRange{}, {});
  mlir::Value contextSlot = b.addrOf("g_exc_context_node");
  // The area is a raw word inside generator storage, so the node's identity
  // narrows to an address here and widens again in the unstash. That pair is
  // the star frame's, not the chain's: it goes when the frame becomes a box.
  mlir::LLVM::StoreOp::create(b.builder, b.loc, b.loadPtrVal(contextSlot),
                              areaPtr, /*alignment=*/8);
  mlir::LLVM::StoreOp::create(b.builder, b.loc, b.nullPtr(), contextSlot,
                              /*alignment=*/8);
  mlir::cf::BranchOp::create(b.builder, b.loc, done, mlir::ValueRange{});
  b.builder.setInsertionPointToEnd(empty);
  mlir::LLVM::StoreOp::create(b.builder, b.loc, b.nullPtr(), areaPtr,
                              /*alignment=*/8);
  mlir::cf::BranchOp::create(b.builder, b.loc, done, mlir::ValueRange{});
  b.builder.setInsertionPointToEnd(done);
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
}

// void LyEH_UnstashException(i64 area): destructively restore a parked
// node into the process slot (parts + traceback + chain globals) —
// LyEH_DiscardCurrentException's __context__ restore, applied to a parked
// node instead. Sole ownership holds because stash nodes are never shared.
// Restoring over a pending token would silently drop one of the two, so
// that is a trap (the resume drivers order their stash/unstash calls so it
// cannot happen).
void buildUnstashException(SupportBuilder &b) {
  auto fn = b.beginFunction("LyEH_UnstashException",
                            b.builder.getFunctionType({b.ptr()}, {}));
  mlir::Block *entry = fn.addEntryBlock();
  mlir::Region &body = fn.getBody();
  mlir::Block *check = b.builder.createBlock(&body);
  mlir::Block *restore = b.builder.createBlock(&body);
  mlir::Block *done = b.builder.createBlock(&body);
  mlir::Block *trap = b.builder.createBlock(&body);
  b.builder.setInsertionPointToEnd(entry);
  mlir::Value areaPtr = entry->getArgument(0);
  mlir::Value node = b.loadPtrVal(areaPtr);
  mlir::Value stashed = b.ptrNe(node, b.nullPtr());
  mlir::cf::CondBranchOp::create(b.builder, b.loc, stashed, check,
                                 mlir::ValueRange{}, done, mlir::ValueRange{});
  b.builder.setInsertionPointToEnd(check);
  mlir::Value pending = mlir::LLVM::LoadOp::create(
      b.builder, b.loc, b.i1(), b.addrOf("g_current_exception"),
      /*alignment=*/4);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, pending, trap,
                                 mlir::ValueRange{}, restore,
                                 mlir::ValueRange{});
  b.builder.setInsertionPointToEnd(restore);
  mlir::LLVM::StoreOp::create(b.builder, b.loc,
                              b.loadPtrVal(nodeMember(b, node, kNodeCause)),
                              b.addrOf("g_exc_cause_node"), /*alignment=*/8);
  mlir::LLVM::StoreOp::create(b.builder, b.loc,
                              b.loadPtrVal(nodeMember(b, node, kNodeContext)),
                              b.addrOf("g_exc_context_node"),
                              /*alignment=*/8);
  mlir::LLVM::StoreOp::create(b.builder, b.loc,
                              b.loadI64(nodeMember(b, node, kNodeSuppress)),
                              b.addrOf("g_exc_suppress_context"),
                              /*alignment=*/8);
  storeExceptionParts(b, b.addrOf("g_current_parts"),
                      loadExceptionParts(b, nodeMember(b, node, kNodePayload)));
  mlir::LLVM::StoreOp::create(
      b.builder, b.loc,
      mlir::arith::ConstantIntOp::create(b.builder, b.loc, 1, 1).getResult(),
      b.addrOf("g_current_exception"), /*alignment=*/4);
  mlir::Value frames = b.loadPtrVal(nodeMember(b, node, kNodeFrames));
  mlir::Value count = b.loadI64(nodeMember(b, node, kNodeFrameCount));
  mlir::Value haveFrames =
      b.cmpi(mlir::arith::CmpIPredicate::sgt, count, b.iconst(0));
  auto framesIf = mlir::scf::IfOp::create(b.builder, b.loc, mlir::TypeRange{},
                                          haveFrames, /*withElseRegion=*/false);
  {
    mlir::OpBuilder::InsertionGuard guard(b.builder);
    b.builder.setInsertionPointToStart(&framesIf.getThenRegion().front());
    mlir::Value bytes =
        mlir::arith::MulIOp::create(b.builder, b.loc, count, b.iconst(40));
    mlir::LLVM::MemcpyOp::create(b.builder, b.loc,
                                 b.addrOf("g_traceback_stack"), frames, bytes,
                                 /*isVolatile=*/false);
    mlir::LLVM::StoreOp::create(b.builder, b.loc, count,
                                b.addrOf("g_traceback_size"), /*alignment=*/8);
  }
  b.call("free", mlir::TypeRange{}, mlir::ValueRange{frames});
  freeSoleChainNode(b, node, "LyEH_UnstashException");
  mlir::LLVM::StoreOp::create(b.builder, b.loc, b.nullPtr(), areaPtr,
                              /*alignment=*/8);
  mlir::cf::BranchOp::create(b.builder, b.loc, done, mlir::ValueRange{});
  b.builder.setInsertionPointToEnd(done);
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
  b.builder.setInsertionPointToEnd(trap);
  emitLLVMTrap(b);
}

// void LyEH_AdoptStashedAsContext(i64 area): hand a parked token to the
// pending exception as the TAIL of its __context__ chain. This models
// CPython's per-frame exc-state stack at the generator boundary: an
// exception crossing the boundary (escaping the body, or injected by
// throw()) was raised above the parked handler exception, so the parked
// state belongs at the bottom of the new exception's chain — both for the
// "During handling of ..." report and so the handler-completion restore
// (LyEH_DiscardCurrentException) pops back to it, CPython's exception
// stack pop. Requires a pending exception; adopting into an empty slot
// would strand the node in globals no raise path owns, so that is a trap.
void buildAdoptStashedAsContext(SupportBuilder &b) {
  auto fn = b.beginFunction("LyEH_AdoptStashedAsContext",
                            b.builder.getFunctionType({b.ptr()}, {}));
  mlir::Block *entry = fn.addEntryBlock();
  mlir::Region &body = fn.getBody();
  mlir::Block *check = b.builder.createBlock(&body);
  mlir::Block *walk =
      b.builder.createBlock(&body, body.end(), {b.ptr()}, {b.loc});
  mlir::Block *step =
      b.builder.createBlock(&body, body.end(), {b.ptr()}, {b.loc});
  mlir::Block *attach =
      b.builder.createBlock(&body, body.end(), {b.ptr()}, {b.loc});
  mlir::Block *done = b.builder.createBlock(&body);
  mlir::Block *trap = b.builder.createBlock(&body);
  b.builder.setInsertionPointToEnd(entry);
  mlir::Value areaPtr = entry->getArgument(0);
  mlir::Value node = b.loadPtrVal(areaPtr);
  mlir::LLVM::StoreOp::create(b.builder, b.loc, b.nullPtr(), areaPtr,
                              /*alignment=*/8);
  mlir::Value stashed = b.ptrNe(node, b.nullPtr());
  mlir::cf::CondBranchOp::create(b.builder, b.loc, stashed, check,
                                 mlir::ValueRange{}, done, mlir::ValueRange{});
  b.builder.setInsertionPointToEnd(check);
  mlir::Value pending = mlir::LLVM::LoadOp::create(
      b.builder, b.loc, b.i1(), b.addrOf("g_current_exception"),
      /*alignment=*/4);
  mlir::cf::CondBranchOp::create(
      b.builder, b.loc, pending, walk,
      mlir::ValueRange{b.addrOf("g_exc_context_node")}, trap,
      mlir::ValueRange{});
  b.builder.setInsertionPointToEnd(walk);
  mlir::Value slotPtr = walk->getArgument(0);
  mlir::Value current = b.loadPtrVal(slotPtr);
  mlir::Value occupied = b.ptrNe(current, b.nullPtr());
  mlir::cf::CondBranchOp::create(b.builder, b.loc, occupied, step,
                                 mlir::ValueRange{current}, attach,
                                 mlir::ValueRange{slotPtr});
  b.builder.setInsertionPointToEnd(step);
  mlir::Value nextSlot = nodeMember(b, step->getArgument(0), kNodeContext);
  mlir::cf::BranchOp::create(b.builder, b.loc, walk,
                             mlir::ValueRange{nextSlot});
  b.builder.setInsertionPointToEnd(attach);
  mlir::LLVM::StoreOp::create(b.builder, b.loc, node,
                              attach->getArgument(0), /*alignment=*/8);
  mlir::cf::BranchOp::create(b.builder, b.loc, done, mlir::ValueRange{});
  b.builder.setInsertionPointToEnd(done);
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
  b.builder.setInsertionPointToEnd(trap);
  emitLLVMTrap(b);
}

// void LyEH_ReleaseCurrentException(): consume the pending token together
// with its traceback and chain, restoring nothing. Why not
// LyEH_DiscardCurrentException: Discard implements handler completion (the
// __context__ pop), which would resurrect a generator-side chain as the
// CALLER's pending exception when a swallowed injection carries one; the
// swallow paths need the whole identity gone instead.
void buildReleaseCurrentException(SupportBuilder &b) {
  auto fn = b.beginFunction("LyEH_ReleaseCurrentException",
                            b.builder.getFunctionType({}, {}));
  mlir::Block *entry = fn.addEntryBlock();
  mlir::Region &body = fn.getBody();
  mlir::Block *park = b.builder.createBlock(&body);
  mlir::Block *done = b.builder.createBlock(&body);
  b.builder.setInsertionPointToEnd(entry);
  mlir::Value pending = mlir::LLVM::LoadOp::create(
      b.builder, b.loc, b.i1(), b.addrOf("g_current_exception"),
      /*alignment=*/4);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, pending, park,
                                 mlir::ValueRange{}, done,
                                 mlir::ValueRange{});
  b.builder.setInsertionPointToEnd(park);
  b.call("LyEH_StashCurrentAsContext", mlir::TypeRange{}, {});
  mlir::Value contextSlot = b.addrOf("g_exc_context_node");
  mlir::Value node = b.loadPtrVal(contextSlot);
  mlir::LLVM::StoreOp::create(b.builder, b.loc, b.nullPtr(), contextSlot,
                              /*alignment=*/8);
  b.call("release_chain_node", mlir::TypeRange{}, mlir::ValueRange{node});
  mlir::cf::BranchOp::create(b.builder, b.loc, done, mlir::ValueRange{});
  b.builder.setInsertionPointToEnd(done);
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
}

// __ly_global_view_*(i64 pointer_word, i64 size) -> memref<?xT>: a rank-1
// view over a payload a WORD addresses (allocated == aligned, offset 0,
// stride 1 -- the runtime's single-allocation entity convention). One per
// element type because the result type differs; static shapes narrow through
// memref.cast at the call site.
//
// ⛔ The integer argument is the point of the function and also what is wrong
// with it. It exists so a MANIFEST body can get a descriptor through a call
// rather than through a cast, which this pipeline rejects in its input
// (Passes/Runtime/Passes/Lowering.cpp says why); the manifests that call it
// hold a pointer WORD read out of a boxed element's payload slot, and widening
// it is the direction the memory model refuses.
//
// The compiler's own module-global path used to come through here too and no
// longer does -- its cell holds a pointer. What is left is the box slot, which
// should hold one the same way; then the argument here becomes `!llvm.ptr` and
// the widen goes with it.
void buildGlobalViewFunction(SupportBuilder &b, llvm::StringRef name,
                             mlir::Type element) {
  auto resultType =
      mlir::MemRefType::get({mlir::ShapedType::kDynamic}, element);
  auto fn = b.beginFunction(
      name, b.builder.getFunctionType({b.i64(), b.i64()}, {resultType}));
  mlir::Block *entry = fn.addEntryBlock();
  b.builder.setInsertionPointToEnd(entry);
  mlir::Value pointer = b.intToPtr(entry->getArgument(0));
  mlir::func::ReturnOp::create(
      b.builder, b.loc,
      buildMemRef1D(b, resultType,
                    MemRef1DParts{pointer, pointer, b.iconst(0),
                                  entry->getArgument(1), b.iconst(1)}));
}

// i32 LyRunPythonMain(ptr entry): installs the stack guard, invokes the
// program body under the C++ personality, and prints the Python traceback (or
// the native-exception notice) for anything that unwinds out.
void buildRunPythonMain(SupportBuilder &b) {
  auto fn = beginLLVMFunction(b, "LyRunPythonMain", b.i32(), {b.ptr()});
  fn.setPersonalityAttr(mlir::FlatSymbolRefAttr::get(b.builder.getContext(),
                                                     "__gxx_personality_v0"));
  mlir::Block *entry = fn.addEntryBlock(b.builder);
  mlir::Region &body = fn.getBody();
  mlir::Block *run = b.builder.createBlock(&body);
  mlir::Block *ok = b.builder.createBlock(&body);
  mlir::Block *nullEntry = b.builder.createBlock(&body);
  mlir::Block *landing = b.builder.createBlock(&body);
  mlir::Block *native = b.builder.createBlock(&body);
  mlir::Block *python = b.builder.createBlock(&body);
  mlir::Block *printTraceback = b.builder.createBlock(&body);
  mlir::Block *systemExit = b.builder.createBlock(&body);
  mlir::Block *systemExitArgs = b.builder.createBlock(&body);
  mlir::Block *exitWithStatus = b.builder.createBlock(&body);
  mlir::Block *exitSilently = b.builder.createBlock(&body);
  mlir::Block *exitWithMessage = b.builder.createBlock(&body);

  b.builder.setInsertionPointToEnd(entry);
  mlir::LLVM::CallOp::create(b.builder, b.loc, mlir::TypeRange{},
                             "LyRt_InstallStackGuard", mlir::ValueRange{});
  mlir::Value entryNull = b.ptrEq(entry->getArgument(0), b.nullPtr());
  mlir::LLVM::CondBrOp::create(b.builder, b.loc, entryNull, nullEntry, run);

  b.builder.setInsertionPointToEnd(run);
  auto bodyType = mlir::LLVM::LLVMFunctionType::get(
      mlir::LLVM::LLVMVoidType::get(b.builder.getContext()), {}, false);
  mlir::LLVM::InvokeOp::create(b.builder, b.loc, bodyType,
                               /*callee=*/mlir::FlatSymbolRefAttr(),
                               mlir::ValueRange{entry->getArgument(0)}, ok,
                               mlir::ValueRange{}, landing,
                               mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(ok);
  mlir::LLVM::ReturnOp::create(b.builder, b.loc,
                               mlir::ValueRange{b.iconst32(0)});

  b.builder.setInsertionPointToEnd(nullEntry);
  mlir::LLVM::ReturnOp::create(b.builder, b.loc,
                               mlir::ValueRange{b.iconst32(1)});

  b.builder.setInsertionPointToEnd(landing);
  auto landingType = mlir::LLVM::LLVMStructType::getLiteral(
      b.builder.getContext(), {b.ptr(), b.i32()});
  mlir::Value pad = mlir::LLVM::LandingpadOp::create(
      b.builder, b.loc, landingType, /*cleanup=*/false,
      mlir::ValueRange{b.nullPtr()});
  mlir::Value exceptionObject = mlir::LLVM::ExtractValueOp::create(
      b.builder, b.loc, pad, llvm::ArrayRef<std::int64_t>{0});
  mlir::LLVM::CallOp::create(b.builder, b.loc, mlir::TypeRange{b.ptr()},
                             "__cxa_begin_catch",
                             mlir::ValueRange{exceptionObject});
  mlir::Value descriptor = mlir::LLVM::AllocaOp::create(
      b.builder, b.loc, b.ptr(), exceptionPartsType(b), b.iconst32(1),
      /*alignment=*/8);
  mlir::Value isPython =
      mlir::func::CallOp::create(b.builder, b.loc,
                                 "LyEH_TakeCurrentDescriptor",
                                 mlir::TypeRange{b.i1()},
                                 mlir::ValueRange{descriptor})
          .getResult(0);
  mlir::LLVM::CondBrOp::create(b.builder, b.loc, isPython, python, native);

  b.builder.setInsertionPointToEnd(native);
  mlir::func::CallOp::create(
      b.builder, b.loc, "write_cstr", mlir::TypeRange{},
      mlir::ValueRange{b.iconst32(2), b.addrOf(".native_exception")});
  mlir::func::CallOp::create(b.builder, b.loc, "release_current_chain",
                             mlir::TypeRange{}, mlir::ValueRange{});
  mlir::func::CallOp::create(b.builder, b.loc, "LyTraceback_Clear",
                             mlir::TypeRange{}, mlir::ValueRange{});
  mlir::LLVM::CallOp::create(b.builder, b.loc, mlir::TypeRange{},
                             "__cxa_end_catch", mlir::ValueRange{});
  mlir::LLVM::ReturnOp::create(b.builder, b.loc,
                               mlir::ValueRange{b.iconst32(1)});

  b.builder.setInsertionPointToEnd(python);
  mlir::Value aligned = b.loadPtrVal(partsField(b, descriptor, 0, 1));
  mlir::Value offset = mlir::LLVM::LoadOp::create(
      b.builder, b.loc, b.i64(), partsField(b, descriptor, 0, 2),
      /*alignment=*/8);
  mlir::Value stride = mlir::LLVM::LoadOp::create(
      b.builder, b.loc, b.i64(), partsField(b, descriptor, 0, 4),
      /*alignment=*/8);
  auto headerWord = [&](std::int64_t slot) {
    mlir::Value scaled =
        mlir::LLVM::MulOp::create(b.builder, b.loc, stride, b.iconst(slot));
    mlir::Value index =
        mlir::LLVM::AddOp::create(b.builder, b.loc, offset, scaled);
    return b.loadI64(b.gepI64(aligned, index));
  };
  mlir::Value classId = headerWord(2);
  // The two words SystemExit answers with: the payload block (absent means the
  // exception was raised with no argument at all) and the exit code, biased by
  // one so that a zero slot is "no int code" rather than "exit 0".
  mlir::Value payloadBlock = headerWord(3);
  mlir::Value exitCodeBiased = headerWord(5);
  mlir::Value messageHeader = b.loadPtrVal(partsField(b, descriptor, 1, 1));
  mlir::Value messageData = b.loadPtrVal(partsField(b, descriptor, 2, 1));
  mlir::Value messageOffset = mlir::LLVM::LoadOp::create(
      b.builder, b.loc, b.i64(), partsField(b, descriptor, 2, 2),
      /*alignment=*/8);
  mlir::Value messageLen = mlir::LLVM::LoadOp::create(
      b.builder, b.loc, b.i64(), partsField(b, descriptor, 2, 3),
      /*alignment=*/8);
  mlir::Value messageStride = mlir::LLVM::LoadOp::create(
      b.builder, b.loc, b.i64(), partsField(b, descriptor, 2, 4),
      /*alignment=*/8);
  // The reference `LyEH_TakeCurrentDescriptor` transferred to this frame. Every
  // path out of here discharges it (`release_taken_exception`); the words are
  // read now because the descriptor's storage is an alloca the exit paths still
  // read from, and taking them once keeps the three paths spelling the same
  // thing.
  mlir::Value takenHeader = aligned;
  mlir::Value takenMessageHeader = messageHeader;
  mlir::Value takenMessageData = messageData;
  auto releaseTaken = [&]() {
    mlir::func::CallOp::create(
        b.builder, b.loc, "release_taken_exception", mlir::TypeRange{},
        mlir::ValueRange{takenHeader, takenMessageHeader, takenMessageData});
  };
  mlir::Value isSystemExit =
      b.cmpi(mlir::arith::CmpIPredicate::eq, classId, b.iconst(64));
  mlir::LLVM::CondBrOp::create(b.builder, b.loc, isSystemExit, systemExit,
                               printTraceback);

  b.builder.setInsertionPointToEnd(printTraceback);
  mlir::func::CallOp::create(
      b.builder, b.loc, "LyTraceback_PrintMessage", mlir::TypeRange{},
      mlir::ValueRange{classId, aligned, messageHeader, messageData,
                       messageOffset, messageLen, messageStride});
  mlir::func::CallOp::create(b.builder, b.loc, "release_current_chain",
                             mlir::TypeRange{}, mlir::ValueRange{});
  mlir::func::CallOp::create(b.builder, b.loc, "LyTraceback_Clear",
                             mlir::TypeRange{}, mlir::ValueRange{});
  releaseTaken();
  mlir::LLVM::CallOp::create(b.builder, b.loc, mlir::TypeRange{},
                             "__cxa_end_catch", mlir::ValueRange{});
  mlir::LLVM::ReturnOp::create(b.builder, b.loc,
                               mlir::ValueRange{b.iconst32(1)});

  // SystemExit never prints a traceback (CPython semantics), and CPython's three
  // answers are read off the object rather than guessed from the message: an int
  // code exits WITH it in silence, no argument at all exits 0 in silence, and
  // anything else goes to stderr and exits 1.
  //
  // ⛔ THE OLD TEST WAS "the message is empty", a proxy for "this came from
  // sys.exit", and it was wrong at both edges. `SystemExit("")` has an empty
  // message and is NOT a status (CPython prints the blank line and exits 1),
  // and an int argument could not be a status at all, because the code lived in
  // a process global that only sys.exit wrote.
  b.builder.setInsertionPointToEnd(systemExit);
  mlir::func::CallOp::create(b.builder, b.loc, "release_current_chain",
                             mlir::TypeRange{}, mlir::ValueRange{});
  mlir::func::CallOp::create(b.builder, b.loc, "LyTraceback_Clear",
                             mlir::TypeRange{}, mlir::ValueRange{});
  mlir::Value hasCode =
      b.cmpi(mlir::arith::CmpIPredicate::ne, exitCodeBiased, b.iconst(0));
  mlir::LLVM::CondBrOp::create(b.builder, b.loc, hasCode, exitWithStatus,
                               systemExitArgs);

  b.builder.setInsertionPointToEnd(systemExitArgs);
  mlir::Value hasArgument =
      b.cmpi(mlir::arith::CmpIPredicate::ne, payloadBlock, b.iconst(0));
  mlir::LLVM::CondBrOp::create(b.builder, b.loc, hasArgument, exitWithMessage,
                               exitSilently);

  b.builder.setInsertionPointToEnd(exitWithStatus);
  releaseTaken();
  mlir::LLVM::CallOp::create(b.builder, b.loc, mlir::TypeRange{},
                             "__cxa_end_catch", mlir::ValueRange{});
  mlir::Value status = mlir::LLVM::SubOp::create(b.builder, b.loc,
                                                 exitCodeBiased, b.iconst(1));
  mlir::Value status32 =
      mlir::arith::TruncIOp::create(b.builder, b.loc, b.i32(), status);
  mlir::LLVM::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{status32});

  b.builder.setInsertionPointToEnd(exitSilently);
  releaseTaken();
  mlir::LLVM::CallOp::create(b.builder, b.loc, mlir::TypeRange{},
                             "__cxa_end_catch", mlir::ValueRange{});
  mlir::LLVM::ReturnOp::create(b.builder, b.loc,
                               mlir::ValueRange{b.iconst32(0)});

  b.builder.setInsertionPointToEnd(exitWithMessage);
  mlir::Value messageCStr =
      mlir::func::CallOp::create(
          b.builder, b.loc, "utf8_message_cstr", mlir::TypeRange{b.ptr()},
          mlir::ValueRange{messageHeader, messageData, messageOffset,
                           messageLen, messageStride})
          .getResult(0);
  mlir::func::CallOp::create(b.builder, b.loc, "write_cstr", mlir::TypeRange{},
                             mlir::ValueRange{b.iconst32(2), messageCStr});
  mlir::func::CallOp::create(
      b.builder, b.loc, "write_char", mlir::TypeRange{},
      mlir::ValueRange{b.iconst32(2), b.iconst8(10)});
  b.call("free", mlir::TypeRange{}, mlir::ValueRange{messageCStr});
  releaseTaken();
  mlir::LLVM::CallOp::create(b.builder, b.loc, mlir::TypeRange{},
                             "__cxa_end_catch", mlir::ValueRange{});
  mlir::LLVM::ReturnOp::create(b.builder, b.loc,
                               mlir::ValueRange{b.iconst32(1)});
}

} // namespace

// ---------------------------------------------------------------------------
// The exception payload and the chain node (declared in SupportBuilder.h).
// ---------------------------------------------------------------------------

namespace {

mlir::Type memRefPartsStruct(SupportBuilder &b, llvm::StringRef name) {
  auto type =
      mlir::LLVM::LLVMStructType::getIdentified(b.builder.getContext(), name);
  if (type.getBody().empty())
    (void)type.setBody({b.ptr(), b.ptr(), b.i64(), b.i64(), b.i64()},
                       /*isPacked=*/false);
  return type;
}

} // namespace

mlir::Type exceptionPartsType(SupportBuilder &b) {
  auto type = mlir::LLVM::LLVMStructType::getIdentified(b.builder.getContext(),
                                                        "ExceptionParts");
  if (type.getBody().empty())
    (void)type.setBody({memRefPartsStruct(b, "I64MemRef"),
                        memRefPartsStruct(b, "I64MemRef"),
                        memRefPartsStruct(b, "I8MemRef")},
                       /*isPacked=*/false);
  return type;
}

mlir::Value partsField(SupportBuilder &b, mlir::Value parts,
                       std::int32_t section, std::int32_t field) {
  return mlir::LLVM::GEPOp::create(
      b.builder, b.loc, b.ptr(), exceptionPartsType(b), parts,
      llvm::ArrayRef<mlir::LLVM::GEPArg>{mlir::LLVM::GEPArg(0),
                                         mlir::LLVM::GEPArg(section),
                                         mlir::LLVM::GEPArg(field)},
      mlir::LLVM::GEPNoWrapFlags::inbounds);
}

mlir::Value loadExceptionParts(SupportBuilder &b, mlir::Value parts) {
  return mlir::LLVM::LoadOp::create(b.builder, b.loc, exceptionPartsType(b),
                                    parts, /*alignment=*/8);
}

void storeExceptionParts(SupportBuilder &b, mlir::Value parts,
                         mlir::Value value) {
  mlir::LLVM::StoreOp::create(b.builder, b.loc, value, parts,
                              /*alignment=*/8);
}

void clearExceptionParts(SupportBuilder &b, mlir::Value parts) {
  storeExceptionParts(
      b, parts,
      mlir::LLVM::ZeroOp::create(b.builder, b.loc, exceptionPartsType(b)));
}

mlir::Type exceptionChainNodeType(SupportBuilder &b) {
  auto type = mlir::LLVM::LLVMStructType::getIdentified(
      b.builder.getContext(), "ExceptionChainNode");
  if (type.getBody().empty())
    (void)type.setBody({b.i64(), exceptionPartsType(b), b.ptr(), b.i64(),
                        b.ptr(), b.ptr(), b.i64()},
                       /*isPacked=*/false);
  return type;
}

mlir::Value nodeMember(SupportBuilder &b, mlir::Value node,
                       std::int32_t member) {
  return mlir::LLVM::GEPOp::create(
      b.builder, b.loc, b.ptr(), exceptionChainNodeType(b), node,
      llvm::ArrayRef<mlir::LLVM::GEPArg>{mlir::LLVM::GEPArg(0),
                                         mlir::LLVM::GEPArg(member)},
      mlir::LLVM::GEPNoWrapFlags::inbounds);
}

mlir::Value nodePartsField(SupportBuilder &b, mlir::Value node,
                           std::int32_t section, std::int32_t field) {
  return mlir::LLVM::GEPOp::create(
      b.builder, b.loc, b.ptr(), exceptionChainNodeType(b), node,
      llvm::ArrayRef<mlir::LLVM::GEPArg>{
          mlir::LLVM::GEPArg(0), mlir::LLVM::GEPArg(kNodePayload),
          mlir::LLVM::GEPArg(section), mlir::LLVM::GEPArg(field)},
      mlir::LLVM::GEPNoWrapFlags::inbounds);
}

MemRef1DParts explodeMemRef1D(SupportBuilder &b, mlir::Value memref) {
  return explodeMemRef1D(b.builder, b.loc, memref);
}

mlir::Value buildMemRef1D(SupportBuilder &b, mlir::Type memrefType,
                          const MemRef1DParts &parts) {
  return buildMemRef1D(b.builder, b.loc, memrefType, parts);
}

void freeSoleChainNode(SupportBuilder &b, mlir::Value node,
                       llvm::StringRef site) {
  mlir::Value refcount = b.loadI64(nodeMember(b, node, kNodeRefcount));
  mlir::cf::AssertOp::create(
      b.builder, b.loc,
      b.cmpi(mlir::arith::CmpIPredicate::eq, refcount, b.iconst(1)),
      ("chain node still shared at " + site).str());
  b.call("free", mlir::TypeRange{}, mlir::ValueRange{node});
}

mlir::Value typeSizeBytes(SupportBuilder &b, mlir::Type type) {
  mlir::Value end = mlir::LLVM::GEPOp::create(
      b.builder, b.loc, b.ptr(), type, b.nullPtr(),
      llvm::ArrayRef<mlir::LLVM::GEPArg>{mlir::LLVM::GEPArg(1)});
  return b.ptrToInt(end);
}

mlir::Type starFrameType(SupportBuilder &b) {
  auto type = mlir::LLVM::LLVMStructType::getIdentified(b.builder.getContext(),
                                                        "ExceptStarFrame");
  if (type.getBody().empty())
    (void)type.setBody(
        {stashCellType(b), b.i64(), b.i64(),
         mlir::LLVM::LLVMArrayType::get(stashCellType(b), kStarClauseLimit)},
        /*isPacked=*/false);
  return type;
}

mlir::Value starFrameMember(SupportBuilder &b, mlir::Value frame,
                            std::int32_t member) {
  return mlir::LLVM::GEPOp::create(
      b.builder, b.loc, b.ptr(), starFrameType(b), frame,
      llvm::ArrayRef<mlir::LLVM::GEPArg>{mlir::LLVM::GEPArg(0),
                                         mlir::LLVM::GEPArg(member)},
      mlir::LLVM::GEPNoWrapFlags::inbounds);
}

mlir::Value starClauseCell(SupportBuilder &b, mlir::Value frame,
                           mlir::Value index) {
  return mlir::LLVM::GEPOp::create(
      b.builder, b.loc, b.ptr(), starFrameType(b), frame,
      llvm::ArrayRef<mlir::LLVM::GEPArg>{mlir::LLVM::GEPArg(0),
                                         mlir::LLVM::GEPArg(kStarClauses),
                                         mlir::LLVM::GEPArg(index)},
      mlir::LLVM::GEPNoWrapFlags::inbounds);
}

void storeExceptionTriple(SupportBuilder &b, mlir::Value parts,
                          mlir::ValueRange triple) {
  for (std::int32_t section = 0; section < 3; ++section) {
    MemRef1DParts members = explodeMemRef1D(b, triple[section]);
    mlir::Value fields[5] = {members.allocated, members.aligned,
                             members.offset, members.size, members.stride};
    for (std::int32_t field = 0; field < 5; ++field)
      mlir::LLVM::StoreOp::create(b.builder, b.loc, fields[field],
                                  partsField(b, parts, section, field),
                                  /*alignment=*/8);
  }
}

mlir::OwningOpRef<mlir::ModuleOp>
buildNativeRuntimeSupportModule(mlir::MLIRContext &context,
                                const llvm::Triple &triple) {
  mlir::OpBuilder builder(&context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::ModuleOp::create(builder.getUnknownLoc());

  SupportBuilder support(*module, triple);
  support.declareExternal("abort", builder.getFunctionType({}, {}));
  support.declareExternal(
      "free", builder.getFunctionType({support.ptr()}, {}));
  support.declareExternal(
      "write", builder.getFunctionType(
                   {support.i32(), support.ptr(), support.i64()},
                   {support.i64()}));
  support.declareExternal(
      "strlen",
      builder.getFunctionType({support.ptr()}, {support.i64()}));
  support.declareExternal(
      "realloc", builder.getFunctionType({support.ptr(), support.i64()},
                                         {support.ptr()}));
  support.declareExternal(
      "fopen", builder.getFunctionType({support.ptr(), support.ptr()},
                                       {support.ptr()}));
  support.declareExternal(
      "fread",
      builder.getFunctionType(
          {support.ptr(), support.i64(), support.i64(), support.ptr()},
          {support.i64()}));
  support.declareExternal(
      "fgetc", builder.getFunctionType({support.ptr()}, {support.i32()}));
  support.declareExternal("getchar",
                          builder.getFunctionType({}, {support.i32()}));
  support.declareExternal(
      "fwrite",
      builder.getFunctionType(
          {support.ptr(), support.i64(), support.i64(), support.ptr()},
          {support.i64()}));
  support.declareExternal(
      "fclose", builder.getFunctionType({support.ptr()}, {support.i32()}));
  support.declareExternal(
      "fflush", builder.getFunctionType({support.ptr()}, {support.i32()}));
  support.declareExternal(
      "fileno", builder.getFunctionType({support.ptr()}, {support.i32()}));
  support.declareExternal(
      "fseek",
      builder.getFunctionType({support.ptr(), support.i64(), support.i32()},
                              {support.i32()}));
  support.declareExternal(
      "ftell", builder.getFunctionType({support.ptr()}, {support.i64()}));
  support.declareExternal(
      "ungetc", builder.getFunctionType({support.i32(), support.ptr()},
                                        {support.i32()}));
  support.declareExternal(
      "ftruncate", builder.getFunctionType({support.i32(), support.i64()},
                                           {support.i32()}));
  // Generated per program in the user module (the manifest deallocators);
  // resolved at link time.
  support.declareExternal(
      "__ly_release_boxed_by_contract",
      builder.getFunctionType({support.ptr(), support.i64()}, {support.i1()}));
  // Per-program user-exception hooks (source-class exception hierarchy and
  // names); same link-time resolution scheme.
  support.declareExternal(
      "__ly_user_exception_base_class_id",
      builder.getFunctionType({support.i64()}, {support.i64()}));
  support.declareExternal(
      "__ly_user_exception_class_name",
      builder.getFunctionType({support.i64()}, {support.ptr()}));

  buildFloatRoundToI64(support);
  buildFloatRound(support);
  buildIntRound(support);
  buildExceptionBaseClassId(support);
  buildEHClassIdMatches(support);
  buildRawBytesEqual(support);
  buildBoxedSlotPtr(support);
  buildBoxedLoadI64(support);
  buildObjectAllocator(support);
  buildFreeRawI64Ptr(support);
  buildReallocRawI64Ptr(support);
  buildReleaseStorageRawToZero(support);
  buildRetainStorageRaw(support);
  buildReleaseSingleAllocation(support, "release_unicode_raw", /*twoArgs=*/true);
  buildWriteLen(support);
  buildWriteCStr(support);
  buildWriteChar(support);
  buildWriteBuffered(support);
  buildBoxedIntValue(support);
  buildPrintBytes(support);
  buildHostSupport(support);
  buildOsSupport(support);
  buildGlobalViewFunction(support, "__ly_global_view_i8", support.i8());
  buildGlobalViewFunction(support, "__ly_global_view_i32", support.i32());
  buildGlobalViewFunction(support, "__ly_global_view_i64", support.i64());
  buildGlobalViewFunction(support, "__ly_global_view_f64", support.f64());
  buildReleasePayloadSlotPtr(support);
  buildReleaseExceptionExtras(support);
  buildReleaseBoxedPayloadRaw(support);
  buildBoxedPayloadArraySlotRaw(
      support, "LyObject_ReleaseBoxedPayloadArraySlotRaw",
      "release_payload_slot_ptr");
  buildRetainPayloadSlotPtr(support);
  buildBoxedPayloadArraySlotRaw(
      support, "LyObject_RetainBoxedPayloadArraySlotRaw",
      "retain_payload_slot_ptr");
  buildTracebackSupport(support);
  declareEHSupport(support);
  buildCurrentExceptionClassIdUnchecked(support);
  buildEndNativeCatchIfActive(support);
  buildThrowException(support);
  buildBeginCatch(support);
  buildBorrowCurrentException(support);
  buildCurrentExceptionClassId(support);
  buildCurrentExceptionMatches(support);
  buildDiscardCurrentException(support);
  buildRethrowCurrent(support);
  buildTakeCurrentDescriptor(support);
  buildStashCurrentException(support);
  buildUnstashException(support);
  buildAdoptStashedAsContext(support);
  buildReleaseCurrentException(support);
  buildRunPythonMain(support);

  return module;
}

} // namespace py::runtime_library
