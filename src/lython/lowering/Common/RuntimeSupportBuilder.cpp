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
// `cf.switch` over the fixed builtin exception class-id table.
void buildExceptionBaseClassId(SupportBuilder &b) {
  auto fn = b.beginFunction("exception_base_class_id",
                            b.builder.getFunctionType({b.i64()}, {b.i64()}),
                            /*isPrivate=*/true);
  mlir::Block *entry = fn.addEntryBlock();
  mlir::Region &body = fn.getBody();
  mlir::Value classId = entry->getArgument(0);

  // One return block per distinct base class id in the shared taxonomy;
  // the root (BaseException's base, 0) is the switch default.
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
  mlir::Block *toRoot = returnBlockFor(py::exceptions::kRootClassId);

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

// i1 release_storage_raw_to_zero(i64 address): atomically decrement the
// refcount word at address; return whether it dropped to zero. Skips null,
// tagged (odd), and immortal (INT64_MAX) storages.
void buildReleaseStorageRawToZero(SupportBuilder &b) {
  auto fn = b.beginFunction(
      "release_storage_raw_to_zero",
      b.builder.getFunctionType({b.i64()}, {b.i1()}));
  mlir::Block *entry = fn.addEntryBlock();
  mlir::Region &body = fn.getBody();
  mlir::Value address = entry->getArgument(0);

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
  mlir::Value pointer = b.intToPtr(address);
  mlir::Value observed = b.loadI64(pointer);
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
      b.builder, b.loc, mlir::LLVM::AtomicBinOp::sub, pointer, one,
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

// void retain_storage_raw(i64 address): atomically increment the refcount
// word at address. Mirror of release_storage_raw_to_zero: skips null, tagged
// (odd), and immortal (INT64_MAX) storages.
void buildRetainStorageRaw(SupportBuilder &b) {
  auto fn = b.beginFunction("retain_storage_raw",
                            b.builder.getFunctionType({b.i64()}, {}));
  mlir::Block *entry = fn.addEntryBlock();
  mlir::Region &body = fn.getBody();
  mlir::Value address = entry->getArgument(0);

  mlir::Block *tagCheck = b.builder.createBlock(&body);
  mlir::Block *probe = b.builder.createBlock(&body);
  mlir::Block *bump = b.builder.createBlock(&body);
  mlir::Block *done = b.builder.createBlock(&body);

  mlir::Value zero = b.iconst(0);
  mlir::Value one = b.iconst(1);
  mlir::Value immortal = b.iconst(9223372036854775807LL);

  b.builder.setInsertionPointToEnd(entry);
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
  mlir::Value pointer = b.intToPtr(address);
  mlir::Value observed = b.loadI64(pointer);
  mlir::Value preImmortal =
      b.cmpi(mlir::arith::CmpIPredicate::eq, observed, immortal);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, preImmortal, done,
                                 mlir::ValueRange{}, bump,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(bump);
  mlir::LLVM::AtomicRMWOp::create(b.builder, b.loc,
                                  mlir::LLVM::AtomicBinOp::add, pointer, one,
                                  mlir::LLVM::AtomicOrdering::acq_rel);
  mlir::cf::BranchOp::create(b.builder, b.loc, done, mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(done);
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
}

// Shared shape: release a single-allocation storage rooted at `address`,
// freeing it if the refcount hit zero. `release_unicode_raw`'s second argument
// (an interior bytes view) needs no separate free.
void buildReleaseSingleAllocation(SupportBuilder &b, llvm::StringRef name,
                                  bool twoArgs) {
  llvm::SmallVector<mlir::Type, 2> inputs = {b.i64()};
  if (twoArgs)
    inputs.push_back(b.i64());
  auto fn = b.beginFunction(name, b.builder.getFunctionType(inputs, {}));
  mlir::Block *entry = fn.addEntryBlock();
  mlir::Region &body = fn.getBody();
  mlir::Block *freeBlock = b.builder.createBlock(&body);
  mlir::Block *done = b.builder.createBlock(&body);
  b.builder.setInsertionPointToEnd(entry);
  auto becameZero = mlir::func::CallOp::create(
      b.builder, b.loc, "release_storage_raw_to_zero", b.i1(),
      mlir::ValueRange{entry->getArgument(0)});
  mlir::cf::CondBranchOp::create(b.builder, b.loc, becameZero.getResult(0),
                                 freeBlock, mlir::ValueRange{}, done,
                                 mlir::ValueRange{});
  b.builder.setInsertionPointToEnd(freeBlock);
  mlir::func::CallOp::create(b.builder, b.loc, "free_raw_i64_ptr",
                             mlir::TypeRange{},
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

void buildReleaseBoxedPayloadArraySlotRaw(SupportBuilder &b) {
  auto itemsType =
      mlir::MemRefType::get({mlir::ShapedType::kDynamic}, b.i64());
  auto fn = b.beginFunction(
      "LyObject_ReleaseBoxedPayloadArraySlotRaw",
      b.builder.getFunctionType({itemsType, b.i64()}, {}));
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
  mlir::func::CallOp::create(b.builder, b.loc, "release_payload_slot_ptr",
                             mlir::TypeRange{}, mlir::ValueRange{slot});
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
}


mlir::Type memRefPartsStruct(SupportBuilder &b, llvm::StringRef name) {
  auto type = mlir::LLVM::LLVMStructType::getIdentified(
      b.builder.getContext(), name);
  if (type.getBody().empty())
    (void)type.setBody({b.ptr(), b.ptr(), b.i64(), b.i64(), b.i64()},
                       /*isPacked=*/false);
  return type;
}

mlir::Type exceptionPartsType(SupportBuilder &b) {
  auto type = mlir::LLVM::LLVMStructType::getIdentified(
      b.builder.getContext(), "ExceptionParts");
  if (type.getBody().empty())
    (void)type.setBody({memRefPartsStruct(b, "I64MemRef"),
                        memRefPartsStruct(b, "I64MemRef"),
                        memRefPartsStruct(b, "I8MemRef")},
                       /*isPacked=*/false);
  return type;
}

mlir::Type memRef1DType(SupportBuilder &b) {
  auto arrayOne = mlir::LLVM::LLVMArrayType::get(b.i64(), 1);
  return mlir::LLVM::LLVMStructType::getLiteral(
      b.builder.getContext(), {b.ptr(), b.ptr(), b.i64(), arrayOne, arrayOne});
}

mlir::Type exceptionBorrowPartsType(SupportBuilder &b) {
  mlir::Type part = memRef1DType(b);
  return mlir::LLVM::LLVMStructType::getLiteral(b.builder.getContext(),
                                                {part, part, part});
}

// parts[0, section, field]
mlir::Value partsField(SupportBuilder &b, mlir::Value parts,
                       std::int32_t section, std::int32_t field) {
  return mlir::LLVM::GEPOp::create(
      b.builder, b.loc, b.ptr(), exceptionPartsType(b), parts,
      llvm::ArrayRef<mlir::LLVM::GEPArg>{mlir::LLVM::GEPArg(0),
                                         mlir::LLVM::GEPArg(section),
                                         mlir::LLVM::GEPArg(field)},
      mlir::LLVM::GEPNoWrapFlags::inbounds);
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

// LyEH_ThrowException(exception view words x5, message header x5, bytes x5):
// stores the payload in the process slot, then throws the 1-byte C++ carrier.
// A still-pending exception (a raise while another exception is handled that
// did not go through the lowering's explicit stash — e.g. a runtime-internal
// raise) becomes the new exception's implicit __context__.
void buildThrowException(SupportBuilder &b) {
  llvm::SmallVector<mlir::Type, 15> inputs;
  for (int section = 0; section < 3; ++section) {
    inputs.push_back(b.ptr());
    inputs.push_back(b.ptr());
    inputs.push_back(b.i64());
    inputs.push_back(b.i64());
    inputs.push_back(b.i64());
  }
  auto fn = beginLLVMFunction(b, "LyEH_ThrowException", {}, inputs);
  mlir::Block *entry = fn.addEntryBlock(b.builder);
  mlir::Region &body = fn.getBody();
  mlir::Block *stash = b.builder.createBlock(&body);
  mlir::Block *store = b.builder.createBlock(&body);
  b.builder.setInsertionPointToEnd(entry);
  mlir::Value flagSlot = b.addrOf("g_current_exception");
  mlir::Value pending = mlir::LLVM::LoadOp::create(b.builder, b.loc, b.i1(),
                                                   flagSlot, /*alignment=*/4);
  mlir::LLVM::CondBrOp::create(b.builder, b.loc, pending, stash, store);
  b.builder.setInsertionPointToEnd(stash);
  mlir::func::CallOp::create(b.builder, b.loc, "LyEH_StashCurrentAsContext",
                             mlir::TypeRange{}, mlir::ValueRange{});
  mlir::LLVM::BrOp::create(b.builder, b.loc, store);
  b.builder.setInsertionPointToEnd(store);
  mlir::Value parts = b.addrOf("g_current_parts");
  for (int section = 0; section < 3; ++section)
    for (int field = 0; field < 5; ++field)
      mlir::LLVM::StoreOp::create(b.builder, b.loc,
                                  entry->getArgument(section * 5 + field),
                                  partsField(b, parts, section, field),
                                  /*alignment=*/8);
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
  mlir::LLVM::UnreachableOp::create(b.builder, b.loc);
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

// ExceptionBorrowParts LyEH_BorrowCurrentException(): the stored payload as
// three rank-1 memref descriptors (borrowed views).
void buildBorrowCurrentException(SupportBuilder &b) {
  auto fn = beginLLVMFunction(b, "LyEH_BorrowCurrentException",
                              exceptionBorrowPartsType(b), {});
  mlir::Block *entry = fn.addEntryBlock(b.builder);
  mlir::Region &body = fn.getBody();
  mlir::Block *borrow = b.builder.createBlock(&body);
  mlir::Block *trap = b.builder.createBlock(&body);
  b.builder.setInsertionPointToEnd(entry);
  mlir::Value pending = mlir::LLVM::LoadOp::create(
      b.builder, b.loc, b.i1(), b.addrOf("g_current_exception"),
      /*alignment=*/4);
  mlir::LLVM::CondBrOp::create(b.builder, b.loc, pending, borrow, trap);
  b.builder.setInsertionPointToEnd(borrow);
  mlir::Value parts = b.addrOf("g_current_parts");
  auto arrayOne = mlir::LLVM::LLVMArrayType::get(b.i64(), 1);
  mlir::Value result =
      mlir::LLVM::UndefOp::create(b.builder, b.loc,
                                  exceptionBorrowPartsType(b));
  for (int section = 0; section < 3; ++section) {
    mlir::Value alloc = b.loadPtrVal(partsField(b, parts, section, 0));
    mlir::Value aligned = b.loadPtrVal(partsField(b, parts, section, 1));
    mlir::Value offset = mlir::LLVM::LoadOp::create(
        b.builder, b.loc, b.i64(), partsField(b, parts, section, 2),
        /*alignment=*/8);
    mlir::Value size = mlir::LLVM::LoadOp::create(
        b.builder, b.loc, b.i64(), partsField(b, parts, section, 3),
        /*alignment=*/8);
    mlir::Value stride = mlir::LLVM::LoadOp::create(
        b.builder, b.loc, b.i64(), partsField(b, parts, section, 4),
        /*alignment=*/8);
    mlir::Value sizeArray = mlir::LLVM::InsertValueOp::create(
        b.builder, b.loc,
        mlir::LLVM::UndefOp::create(b.builder, b.loc, arrayOne).getResult(),
        size, llvm::ArrayRef<std::int64_t>{0});
    mlir::Value strideArray = mlir::LLVM::InsertValueOp::create(
        b.builder, b.loc,
        mlir::LLVM::UndefOp::create(b.builder, b.loc, arrayOne).getResult(),
        stride, llvm::ArrayRef<std::int64_t>{0});
    mlir::Value descriptor =
        mlir::LLVM::UndefOp::create(b.builder, b.loc, memRef1DType(b));
    descriptor = mlir::LLVM::InsertValueOp::create(
        b.builder, b.loc, descriptor, alloc, llvm::ArrayRef<std::int64_t>{0});
    descriptor = mlir::LLVM::InsertValueOp::create(
        b.builder, b.loc, descriptor, aligned, llvm::ArrayRef<std::int64_t>{1});
    descriptor = mlir::LLVM::InsertValueOp::create(
        b.builder, b.loc, descriptor, offset, llvm::ArrayRef<std::int64_t>{2});
    descriptor = mlir::LLVM::InsertValueOp::create(
        b.builder, b.loc, descriptor, sizeArray,
        llvm::ArrayRef<std::int64_t>{3});
    descriptor = mlir::LLVM::InsertValueOp::create(
        b.builder, b.loc, descriptor, strideArray,
        llvm::ArrayRef<std::int64_t>{4});
    result = mlir::LLVM::InsertValueOp::create(
        b.builder, b.loc, result, descriptor,
        llvm::ArrayRef<std::int64_t>{section});
  }
  mlir::LLVM::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{result});
  b.builder.setInsertionPointToEnd(trap);
  emitLLVMTrap(b);
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
         mlir::ValueRange{b.loadI64(causeSlot)});
  mlir::LLVM::StoreOp::create(b.builder, b.loc, b.iconst(0), causeSlot,
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
  mlir::Value exceptionWord = mlir::LLVM::PtrToIntOp::create(
      b.builder, b.loc, b.i64(), exceptionAligned);
  mlir::Value becameZero = b.call("release_storage_raw_to_zero", b.i1(),
                                  mlir::ValueRange{exceptionWord})
                               .front();
  mlir::cf::CondBranchOp::create(b.builder, b.loc, becameZero, freeBlocks,
                                 mlir::ValueRange{}, clear,
                                 mlir::ValueRange{});
  b.builder.setInsertionPointToEnd(freeBlocks);
  mlir::Value headerWord =
      mlir::LLVM::PtrToIntOp::create(b.builder, b.loc, b.i64(), messageHeader);
  mlir::Value bytesWord =
      mlir::LLVM::PtrToIntOp::create(b.builder, b.loc, b.i64(), messageBytes);
  b.call("release_unicode_raw", mlir::TypeRange{},
         mlir::ValueRange{headerWord, bytesWord});
  b.call("free_raw_i64_ptr", mlir::TypeRange{}, mlir::ValueRange{exceptionWord});
  mlir::cf::BranchOp::create(b.builder, b.loc, clear, mlir::ValueRange{});
  b.builder.setInsertionPointToEnd(clear);
  mlir::LLVM::StoreOp::create(
      b.builder, b.loc,
      mlir::arith::ConstantIntOp::create(b.builder, b.loc, 0, 1).getResult(),
      flagSlot, /*alignment=*/4);
  mlir::LLVM::MemsetOp::create(b.builder, b.loc, parts, b.iconst8(0),
                               b.iconst(120), /*isVolatile=*/false);
  b.call("LyTraceback_Clear", mlir::TypeRange{}, {});
  mlir::Value contextSlot = b.addrOf("g_exc_context_node");
  mlir::Value context64 = b.loadI64(contextSlot);
  mlir::Value haveContext =
      b.cmpi(mlir::arith::CmpIPredicate::ne, context64, b.iconst(0));
  mlir::cf::CondBranchOp::create(b.builder, b.loc, haveContext, restore,
                                 mlir::ValueRange{}, finish,
                                 mlir::ValueRange{});

  // Destructive restore: after the cause release above the node has exactly
  // one owner (only the discarded exception could have shared it), so its
  // members move back into the globals and only the shell is freed.
  b.builder.setInsertionPointToEnd(restore);
  mlir::Value node = b.intToPtr(context64);
  auto nodeSlotAt = [&](std::int64_t slot) {
    return b.gepI64(node, b.iconst(slot));
  };
  mlir::LLVM::StoreOp::create(b.builder, b.loc, b.loadI64(nodeSlotAt(18)),
                              causeSlot, /*alignment=*/8);
  mlir::LLVM::StoreOp::create(b.builder, b.loc, b.loadI64(nodeSlotAt(19)),
                              contextSlot, /*alignment=*/8);
  mlir::LLVM::StoreOp::create(b.builder, b.loc, b.loadI64(nodeSlotAt(20)),
                              b.addrOf("g_exc_suppress_context"),
                              /*alignment=*/8);
  mlir::LLVM::MemcpyOp::create(b.builder, b.loc, parts, nodeSlotAt(1),
                               b.iconst(120), /*isVolatile=*/false);
  mlir::LLVM::StoreOp::create(
      b.builder, b.loc,
      mlir::arith::ConstantIntOp::create(b.builder, b.loc, 1, 1).getResult(),
      flagSlot, /*alignment=*/4);
  mlir::Value frames64 = b.loadI64(nodeSlotAt(16));
  mlir::Value count = b.loadI64(nodeSlotAt(17));
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
                                 b.addrOf("g_traceback_stack"),
                                 b.intToPtr(frames64), bytes,
                                 /*isVolatile=*/false);
    mlir::LLVM::StoreOp::create(b.builder, b.loc, count,
                                b.addrOf("g_traceback_size"), /*alignment=*/8);
  }
  b.call("free_raw_i64_ptr", mlir::TypeRange{}, mlir::ValueRange{frames64});
  b.call("free_raw_i64_ptr", mlir::TypeRange{}, mlir::ValueRange{context64});
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(finish);
  mlir::LLVM::StoreOp::create(b.builder, b.loc, b.iconst(0),
                              b.addrOf("g_exc_suppress_context"),
                              /*alignment=*/8);
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
}

void buildDiscardCurrentExceptionIfMatches(SupportBuilder &b) {
  auto fn = b.beginFunction("LyEH_DiscardCurrentExceptionIfMatches",
                            b.builder.getFunctionType({b.i64()}, {b.i1()}));
  mlir::Block *entry = fn.addEntryBlock();
  b.builder.setInsertionPointToEnd(entry);
  mlir::Value matches = b.call("LyEH_CurrentExceptionMatches", b.i1(),
                               mlir::ValueRange{entry->getArgument(0)})
                            .front();
  auto discardIf = mlir::scf::IfOp::create(b.builder, b.loc, mlir::TypeRange{},
                                           matches, /*withElseRegion=*/false);
  {
    mlir::OpBuilder::InsertionGuard guard(b.builder);
    b.builder.setInsertionPointToStart(&discardIf.getThenRegion().front());
    b.call("LyEH_DiscardCurrentException", mlir::TypeRange{}, {});
  }
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{matches});
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
  mlir::LLVM::MemcpyOp::create(b.builder, b.loc, entry->getArgument(0), parts,
                               b.iconst(120), /*isVolatile=*/false);
  mlir::LLVM::MemsetOp::create(b.builder, b.loc, parts, b.iconst8(0),
                               b.iconst(120), /*isVolatile=*/false);
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
// between resumptions. A stash area is 16 i64 words: word 0 is the
// occupancy flag, words 1..15 hold the ExceptionParts payload (the same
// 120-byte layout as g_current_parts). Areas live in the generator storage
// (words 48..63) for the suspended body's token, and in a resume driver's
// stack frame for the resumer's own context.

// void LyEH_StashCurrentException(i64 area): move the pending token (if
// any) out of the process slot into the area; records occupancy in word 0.
// Closes the native catch scope like any other slot consumer.
void buildStashCurrentException(SupportBuilder &b) {
  auto fn = b.beginFunction("LyEH_StashCurrentException",
                            b.builder.getFunctionType({b.i64()}, {}));
  mlir::Block *entry = fn.addEntryBlock();
  b.builder.setInsertionPointToEnd(entry);
  mlir::Value area = entry->getArgument(0);
  mlir::Value partsAddress = mlir::arith::AddIOp::create(
      b.builder, b.loc, area, b.iconst(8));
  mlir::Value partsPtr = b.intToPtr(partsAddress);
  mlir::Value taken = b.call("LyEH_TakeCurrentDescriptor", b.i1(),
                             mlir::ValueRange{partsPtr})
                          .front();
  mlir::Value flagWord =
      mlir::arith::ExtUIOp::create(b.builder, b.loc, b.i64(), taken);
  mlir::LLVM::StoreOp::create(b.builder, b.loc, flagWord, b.intToPtr(area),
                              /*alignment=*/8);
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
}

// void LyEH_UnstashException(i64 area): move a stashed token back into the
// process slot. Restoring over a pending token would silently drop one of
// the two, so that is a trap (the resume drivers order their stash/unstash
// calls so it cannot happen).
void buildUnstashException(SupportBuilder &b) {
  auto fn = beginLLVMFunction(b, "LyEH_UnstashException", {}, {b.i64()});
  mlir::Block *entry = fn.addEntryBlock(b.builder);
  mlir::Region &body = fn.getBody();
  mlir::Block *check = b.builder.createBlock(&body);
  mlir::Block *restore = b.builder.createBlock(&body);
  mlir::Block *done = b.builder.createBlock(&body);
  mlir::Block *trap = b.builder.createBlock(&body);
  b.builder.setInsertionPointToEnd(entry);
  mlir::Value area = entry->getArgument(0);
  mlir::Value areaPtr = b.intToPtr(area);
  mlir::Value flagWord = mlir::LLVM::LoadOp::create(b.builder, b.loc, b.i64(),
                                                    areaPtr, /*alignment=*/8);
  mlir::Value stashed =
      b.cmpi(mlir::arith::CmpIPredicate::ne, flagWord, b.iconst(0));
  mlir::LLVM::CondBrOp::create(b.builder, b.loc, stashed, check, done);
  b.builder.setInsertionPointToEnd(check);
  mlir::Value pending = mlir::LLVM::LoadOp::create(
      b.builder, b.loc, b.i1(), b.addrOf("g_current_exception"),
      /*alignment=*/4);
  mlir::LLVM::CondBrOp::create(b.builder, b.loc, pending, trap, restore);
  b.builder.setInsertionPointToEnd(restore);
  mlir::Value partsAddress = mlir::arith::AddIOp::create(
      b.builder, b.loc, area, b.iconst(8));
  mlir::LLVM::MemcpyOp::create(b.builder, b.loc, b.addrOf("g_current_parts"),
                               b.intToPtr(partsAddress), b.iconst(120),
                               /*isVolatile=*/false);
  mlir::LLVM::StoreOp::create(
      b.builder, b.loc,
      mlir::arith::ConstantIntOp::create(b.builder, b.loc, 1, 1).getResult(),
      b.addrOf("g_current_exception"), /*alignment=*/4);
  mlir::LLVM::StoreOp::create(b.builder, b.loc, b.iconst(0), areaPtr,
                              /*alignment=*/8);
  mlir::LLVM::BrOp::create(b.builder, b.loc, mlir::ValueRange{}, done);
  b.builder.setInsertionPointToEnd(done);
  mlir::LLVM::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
  b.builder.setInsertionPointToEnd(trap);
  emitLLVMTrap(b);
}

// void LyEH_ReleaseStashedException(i64 area): consume a stashed token
// without restoring it (the suspended handler context it belonged to is
// being replaced by an injected exception, or discarded with the
// generator). Mirrors LyEH_DiscardCurrentException's release, minus the
// slot/traceback bookkeeping that only applies to the active token.
void buildReleaseStashedException(SupportBuilder &b) {
  auto fn = b.beginFunction("LyEH_ReleaseStashedException",
                            b.builder.getFunctionType({b.i64()}, {}));
  mlir::Block *entry = fn.addEntryBlock();
  mlir::Region &body = fn.getBody();
  mlir::Block *release = b.builder.createBlock(&body);
  mlir::Block *freeBlocks = b.builder.createBlock(&body);
  mlir::Block *clear = b.builder.createBlock(&body);
  mlir::Block *done = b.builder.createBlock(&body);
  b.builder.setInsertionPointToEnd(entry);
  mlir::Value area = entry->getArgument(0);
  mlir::Value areaPtr = b.intToPtr(area);
  mlir::Value flagWord = mlir::LLVM::LoadOp::create(b.builder, b.loc, b.i64(),
                                                    areaPtr, /*alignment=*/8);
  mlir::Value stashed =
      b.cmpi(mlir::arith::CmpIPredicate::ne, flagWord, b.iconst(0));
  mlir::cf::CondBranchOp::create(b.builder, b.loc, stashed, release,
                                 mlir::ValueRange{}, done, mlir::ValueRange{});
  b.builder.setInsertionPointToEnd(release);
  mlir::Value partsAddress = mlir::arith::AddIOp::create(
      b.builder, b.loc, area, b.iconst(8));
  mlir::Value parts = b.intToPtr(partsAddress);
  mlir::Value exceptionAligned = b.loadPtrVal(partsField(b, parts, 0, 1));
  mlir::Value messageHeader = b.loadPtrVal(partsField(b, parts, 1, 1));
  mlir::Value messageBytes = b.loadPtrVal(partsField(b, parts, 2, 1));
  mlir::Value exceptionWord = mlir::LLVM::PtrToIntOp::create(
      b.builder, b.loc, b.i64(), exceptionAligned);
  mlir::Value becameZero = b.call("release_storage_raw_to_zero", b.i1(),
                                  mlir::ValueRange{exceptionWord})
                               .front();
  mlir::cf::CondBranchOp::create(b.builder, b.loc, becameZero, freeBlocks,
                                 mlir::ValueRange{}, clear, mlir::ValueRange{});
  b.builder.setInsertionPointToEnd(freeBlocks);
  mlir::Value headerWord =
      mlir::LLVM::PtrToIntOp::create(b.builder, b.loc, b.i64(), messageHeader);
  mlir::Value bytesWord =
      mlir::LLVM::PtrToIntOp::create(b.builder, b.loc, b.i64(), messageBytes);
  b.call("release_unicode_raw", mlir::TypeRange{},
         mlir::ValueRange{headerWord, bytesWord});
  b.call("free_raw_i64_ptr", mlir::TypeRange{},
         mlir::ValueRange{exceptionWord});
  mlir::cf::BranchOp::create(b.builder, b.loc, clear, mlir::ValueRange{});
  b.builder.setInsertionPointToEnd(clear);
  mlir::LLVM::StoreOp::create(b.builder, b.loc, b.iconst(0), areaPtr,
                              /*alignment=*/8);
  mlir::LLVM::MemsetOp::create(b.builder, b.loc, parts, b.iconst8(0),
                               b.iconst(120), /*isVolatile=*/false);
  mlir::cf::BranchOp::create(b.builder, b.loc, done, mlir::ValueRange{});
  b.builder.setInsertionPointToEnd(done);
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
}

// __ly_global_view_*(i64 ptr, i64 size): reassemble a rank-1 memref
// descriptor from the raw pointer/size words a module-global object cell
// stores (allocated == aligned, offset 0, stride 1 -- the runtime's
// single-allocation entity convention). One wrapper per element type so the
// func-level signature type-checks against the caller's memref world;
// static shapes narrow through memref.cast at the call site.
void buildGlobalViewFunction(SupportBuilder &b, llvm::StringRef name) {
  auto fn = beginLLVMFunction(b, name, memRef1DType(b), {b.i64(), b.i64()});
  mlir::Block *entry = fn.addEntryBlock(b.builder);
  b.builder.setInsertionPointToEnd(entry);
  auto arrayOne = mlir::LLVM::LLVMArrayType::get(b.i64(), 1);
  mlir::Value pointer = b.intToPtr(entry->getArgument(0));
  mlir::Value size = entry->getArgument(1);
  mlir::Value zero = b.iconst(0);
  mlir::Value one = b.iconst(1);
  mlir::Value sizeArray = mlir::LLVM::InsertValueOp::create(
      b.builder, b.loc,
      mlir::LLVM::UndefOp::create(b.builder, b.loc, arrayOne).getResult(),
      size, llvm::ArrayRef<std::int64_t>{0});
  mlir::Value strideArray = mlir::LLVM::InsertValueOp::create(
      b.builder, b.loc,
      mlir::LLVM::UndefOp::create(b.builder, b.loc, arrayOne).getResult(),
      one, llvm::ArrayRef<std::int64_t>{0});
  mlir::Value descriptor =
      mlir::LLVM::UndefOp::create(b.builder, b.loc, memRef1DType(b));
  descriptor = mlir::LLVM::InsertValueOp::create(
      b.builder, b.loc, descriptor, pointer, llvm::ArrayRef<std::int64_t>{0});
  descriptor = mlir::LLVM::InsertValueOp::create(
      b.builder, b.loc, descriptor, pointer, llvm::ArrayRef<std::int64_t>{1});
  descriptor = mlir::LLVM::InsertValueOp::create(
      b.builder, b.loc, descriptor, zero, llvm::ArrayRef<std::int64_t>{2});
  descriptor = mlir::LLVM::InsertValueOp::create(
      b.builder, b.loc, descriptor, sizeArray,
      llvm::ArrayRef<std::int64_t>{3});
  descriptor = mlir::LLVM::InsertValueOp::create(
      b.builder, b.loc, descriptor, strideArray,
      llvm::ArrayRef<std::int64_t>{4});
  mlir::LLVM::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{descriptor});
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
  mlir::Block *exitWithStatus = b.builder.createBlock(&body);
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
  mlir::Value scaled =
      mlir::LLVM::MulOp::create(b.builder, b.loc, stride, b.iconst(2));
  mlir::Value classIndex =
      mlir::LLVM::AddOp::create(b.builder, b.loc, offset, scaled);
  mlir::Value classId = b.loadI64(b.gepI64(aligned, classIndex));
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
  mlir::Value isSystemExit =
      b.cmpi(mlir::arith::CmpIPredicate::eq, classId, b.iconst(64));
  mlir::LLVM::CondBrOp::create(b.builder, b.loc, isSystemExit, systemExit,
                               printTraceback);

  b.builder.setInsertionPointToEnd(printTraceback);
  mlir::func::CallOp::create(
      b.builder, b.loc, "LyTraceback_PrintMessage", mlir::TypeRange{},
      mlir::ValueRange{classId, messageHeader, messageData, messageOffset,
                       messageLen, messageStride});
  mlir::func::CallOp::create(b.builder, b.loc, "release_current_chain",
                             mlir::TypeRange{}, mlir::ValueRange{});
  mlir::func::CallOp::create(b.builder, b.loc, "LyTraceback_Clear",
                             mlir::TypeRange{}, mlir::ValueRange{});
  mlir::LLVM::CallOp::create(b.builder, b.loc, mlir::TypeRange{},
                             "__cxa_end_catch", mlir::ValueRange{});
  mlir::LLVM::ReturnOp::create(b.builder, b.loc,
                               mlir::ValueRange{b.iconst32(1)});

  // SystemExit never prints a traceback (CPython semantics): an empty
  // message reports the last LyHost_SetExitStatus value, a non-empty message
  // goes to stderr with exit status 1 (the non-int `SystemExit(code)` path).
  b.builder.setInsertionPointToEnd(systemExit);
  mlir::func::CallOp::create(b.builder, b.loc, "release_current_chain",
                             mlir::TypeRange{}, mlir::ValueRange{});
  mlir::func::CallOp::create(b.builder, b.loc, "LyTraceback_Clear",
                             mlir::TypeRange{}, mlir::ValueRange{});
  mlir::Value messageEmpty =
      b.cmpi(mlir::arith::CmpIPredicate::eq, messageLen, b.iconst(0));
  mlir::LLVM::CondBrOp::create(b.builder, b.loc, messageEmpty, exitWithStatus,
                               exitWithMessage);

  b.builder.setInsertionPointToEnd(exitWithStatus);
  mlir::LLVM::CallOp::create(b.builder, b.loc, mlir::TypeRange{},
                             "__cxa_end_catch", mlir::ValueRange{});
  mlir::Value status = mlir::LLVM::LoadOp::create(
      b.builder, b.loc, b.i64(), b.addrOf("g_sys_exit_status"),
      /*alignment=*/8);
  mlir::Value status32 =
      mlir::arith::TruncIOp::create(b.builder, b.loc, b.i32(), status);
  mlir::LLVM::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{status32});

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
  mlir::LLVM::CallOp::create(b.builder, b.loc, mlir::TypeRange{},
                             "__cxa_end_catch", mlir::ValueRange{});
  mlir::LLVM::ReturnOp::create(b.builder, b.loc,
                               mlir::ValueRange{b.iconst32(1)});
}

} // namespace

mlir::OwningOpRef<mlir::ModuleOp>
buildNativeRuntimeSupportModule(mlir::MLIRContext &context) {
  mlir::OpBuilder builder(&context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::ModuleOp::create(builder.getUnknownLoc());

  SupportBuilder support(*module);
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

  buildFloatRoundToI64(support);
  buildFloatRound(support);
  buildIntRound(support);
  buildExceptionBaseClassId(support);
  buildEHClassIdMatches(support);
  buildRawBytesEqual(support);
  buildBoxedSlotPtr(support);
  buildBoxedLoadI64(support);
  buildFreeRawI64Ptr(support);
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
  buildGlobalViewFunction(support, "__ly_global_view_i8");
  buildGlobalViewFunction(support, "__ly_global_view_i32");
  buildGlobalViewFunction(support, "__ly_global_view_i64");
  buildGlobalViewFunction(support, "__ly_global_view_f64");
  buildReleasePayloadSlotPtr(support);
  buildReleaseBoxedPayloadRaw(support);
  buildReleaseBoxedPayloadArraySlotRaw(support);
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
  buildDiscardCurrentExceptionIfMatches(support);
  buildRethrowCurrent(support);
  buildTakeCurrentDescriptor(support);
  buildStashCurrentException(support);
  buildUnstashException(support);
  buildReleaseStashedException(support);
  buildRunPythonMain(support);

  return module;
}

} // namespace py::runtime_library
