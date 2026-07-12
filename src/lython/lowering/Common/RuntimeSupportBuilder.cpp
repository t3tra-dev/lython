#include "Common/RuntimeSupportBuilder.h"

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

// Small builder facade that mirrors the shape of the hand-written support IR
// while keeping the C++ concise. Every runtime routine is composed from the
// high-level dialects (func/arith/math/cf/ub) so the existing native-runtime
// lowering pipeline finalizes it to LLVM.
struct SupportBuilder {
  mlir::OpBuilder builder;
  mlir::Location loc;
  mlir::ModuleOp module;

  explicit SupportBuilder(mlir::ModuleOp module)
      : builder(module.getContext()), loc(builder.getUnknownLoc()),
        module(module) {}

  mlir::Type f64() { return builder.getF64Type(); }
  mlir::Type i64() { return builder.getIntegerType(64); }
  mlir::Type i32() { return builder.getIntegerType(32); }
  mlir::Type i8() { return builder.getIntegerType(8); }
  mlir::Type i1() { return builder.getIntegerType(1); }
  mlir::Type ptr() { return mlir::LLVM::LLVMPointerType::get(builder.getContext()); }

  mlir::Value intToPtr(mlir::Value address) {
    return mlir::LLVM::IntToPtrOp::create(builder, loc, ptr(), address);
  }
  // base[index] as an i64-element GEP (index in i64 units unless noted).
  mlir::Value gepI64(mlir::Value base, mlir::Value index) {
    return mlir::LLVM::GEPOp::create(builder, loc, ptr(), i64(), base,
                                     mlir::ValueRange{index});
  }
  mlir::Value loadI64(mlir::Value pointer) {
    return mlir::LLVM::LoadOp::create(builder, loc, i64(), pointer,
                                      /*alignment=*/8);
  }

  mlir::Value fconst(double value) {
    return mlir::arith::ConstantOp::create(builder, loc,
                                           builder.getF64FloatAttr(value));
  }
  mlir::Value iconst(std::int64_t value) {
    return mlir::arith::ConstantIntOp::create(builder, loc, i64(), value);
  }
  mlir::Value iconst32(std::int32_t value) {
    return mlir::arith::ConstantIntOp::create(builder, loc, i32(), value);
  }
  mlir::Value iconst8(std::int8_t value) {
    return mlir::arith::ConstantIntOp::create(builder, loc, i8(), value);
  }
  mlir::Value nullPtr() {
    return mlir::LLVM::ZeroOp::create(builder, loc, ptr()).getResult();
  }
  mlir::Value addrOf(llvm::StringRef name) {
    return mlir::LLVM::AddressOfOp::create(builder, loc, ptr(), name)
        .getResult();
  }
  mlir::Value ptrEq(mlir::Value a, mlir::Value b) {
    return mlir::LLVM::ICmpOp::create(builder, loc,
                                      mlir::LLVM::ICmpPredicate::eq, a, b);
  }
  mlir::Value ptrNe(mlir::Value a, mlir::Value b) {
    return mlir::LLVM::ICmpOp::create(builder, loc,
                                      mlir::LLVM::ICmpPredicate::ne, a, b);
  }
  mlir::Value gepI8(mlir::Value base, mlir::Value index) {
    return mlir::LLVM::GEPOp::create(builder, loc, ptr(), i8(), base,
                                     mlir::ValueRange{index});
  }
  mlir::Value loadI8(mlir::Value pointer) {
    return mlir::LLVM::LoadOp::create(builder, loc, i8(), pointer,
                                      /*alignment=*/1);
  }
  void storeI8(mlir::Value value, mlir::Value pointer) {
    mlir::LLVM::StoreOp::create(builder, loc, value, pointer, /*alignment=*/1);
  }
  mlir::Value loadPtrVal(mlir::Value pointer) {
    return mlir::LLVM::LoadOp::create(builder, loc, ptr(), pointer,
                                      /*alignment=*/8);
  }
  mlir::Value loadI32(mlir::Value pointer) {
    return mlir::LLVM::LoadOp::create(builder, loc, i32(), pointer,
                                      /*alignment=*/4);
  }
  // Fields of a TracebackFrame slot: frame[0, index].
  mlir::Value frameField(mlir::Type frameType, mlir::Value frame,
                         std::int32_t index) {
    return mlir::LLVM::GEPOp::create(
        builder, loc, ptr(), frameType, frame,
        llvm::ArrayRef<mlir::LLVM::GEPArg>{mlir::LLVM::GEPArg(0),
                                           mlir::LLVM::GEPArg(index)},
        mlir::LLVM::GEPNoWrapFlags::inbounds);
  }
  mlir::ValueRange call(llvm::StringRef callee, mlir::TypeRange results,
                        mlir::ValueRange args) {
    return mlir::func::CallOp::create(builder, loc, callee, results, args)
        .getResults();
  }
  // Internal constant C string global (idempotent); NUL is appended.
  void stringGlobal(llvm::StringRef name, llvm::StringRef text) {
    if (module.lookupSymbol(name))
      return;
    std::string data = text.str();
    data.push_back('\0');
    auto type = mlir::LLVM::LLVMArrayType::get(i8(), data.size());
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(module.getBody());
    mlir::LLVM::GlobalOp::create(builder, loc, type, /*isConstant=*/true,
                                 mlir::LLVM::Linkage::Internal, name,
                                 builder.getStringAttr(data),
                                 /*alignment=*/1);
  }
  mlir::Value cmpf(mlir::arith::CmpFPredicate pred, mlir::Value a,
                   mlir::Value b) {
    return mlir::arith::CmpFOp::create(builder, loc, pred, a, b);
  }
  mlir::Value cmpi(mlir::arith::CmpIPredicate pred, mlir::Value a,
                   mlir::Value b) {
    return mlir::arith::CmpIOp::create(builder, loc, pred, a, b);
  }
  mlir::Value orBit(mlir::Value a, mlir::Value b) {
    return mlir::arith::OrIOp::create(builder, loc, a, b);
  }

  // Declares an external libc symbol (resolved at final link).
  void declareExternal(llvm::StringRef name, mlir::FunctionType type) {
    if (module.lookupSymbol(name))
      return;
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(module.getBody());
    auto fn = mlir::func::FuncOp::create(builder, loc, name, type);
    fn.setPrivate();
  }

  mlir::func::FuncOp beginFunction(llvm::StringRef name, mlir::FunctionType type,
                                   bool isPrivate = false) {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(module.getBody());
    auto fn = mlir::func::FuncOp::create(builder, loc, name, type);
    fn.setVisibility(isPrivate ? mlir::SymbolTable::Visibility::Private
                               : mlir::SymbolTable::Visibility::Public);
    return fn;
  }

  // Terminates a trap block: abort() then a poison return (abort is noreturn,
  // so the return is dead but keeps the block well-formed without dropping to
  // the llvm dialect's `unreachable`).
  void emitTrap(mlir::Type resultType) {
    mlir::func::CallOp::create(builder, loc, "abort", mlir::TypeRange{},
                               mlir::ValueRange{});
    mlir::Value poison =
        mlir::ub::PoisonOp::create(builder, loc, resultType, nullptr);
    mlir::func::ReturnOp::create(builder, loc, poison);
  }
};

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

  // Return blocks, one per distinct base class id in the table.
  auto returnBlock = [&](std::int64_t value) {
    mlir::Block *block = b.builder.createBlock(&body);
    b.builder.setInsertionPointToEnd(block);
    mlir::func::ReturnOp::create(b.builder, b.loc, b.iconst(value));
    return block;
  };
  mlir::Block *toException = returnBlock(5);        // 50/62 -> Exception
  mlir::Block *toBaseException = returnBlock(50);   // most concrete -> Exception
  mlir::Block *toLookupError = returnBlock(60);     // 54/55 -> LookupError
  mlir::Block *toArithmeticError = returnBlock(59); // 61 -> ArithmeticError
  mlir::Block *toRoot = returnBlock(0);             // default -> root

  struct Case {
    std::int64_t value;
    mlir::Block *dest;
  };
  const Case cases[] = {
      {50, toException},        {62, toException},
      {51, toBaseException},    {52, toBaseException},
      {53, toBaseException},    {56, toBaseException},
      {57, toBaseException},    {58, toBaseException},
      {59, toBaseException},    {60, toBaseException},
      {54, toLookupError},      {55, toLookupError},
      {61, toArithmeticError},
  };
  llvm::SmallVector<llvm::APInt, 16> caseValues;
  llvm::SmallVector<mlir::Block *, 16> caseDests;
  llvm::SmallVector<mlir::ValueRange, 16> caseOperands;
  for (const Case &entryCase : cases) {
    caseValues.emplace_back(64, static_cast<std::uint64_t>(entryCase.value));
    caseDests.push_back(entryCase.dest);
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
      mlir::ValueRange{slot, b.iconst(14)});
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
  auto boxType =
      mlir::MemRefType::get({16}, b.i64());
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

// LyHost_Print/LyHost_PrintLine(memref<?xi8> descriptor, i64 len) at the
// lowered ABI (alloc ptr, aligned ptr, offset, size, stride, len): stdout
// writes through print_bytes; PrintLine appends the newline byte.
void buildHostPrint(SupportBuilder &b, llvm::StringRef name, bool newline) {
  auto fn = b.beginFunction(
      name, b.builder.getFunctionType(
                {b.ptr(), b.ptr(), b.i64(), b.i64(), b.i64(), b.i64()}, {}));
  mlir::Block *entry = fn.addEntryBlock();
  b.builder.setInsertionPointToEnd(entry);
  mlir::Value stdoutFd = mlir::arith::ConstantIntOp::create(b.builder, b.loc,
                                                            b.i32(), 1);
  mlir::func::CallOp::create(
      b.builder, b.loc, "print_bytes", mlir::TypeRange{},
      mlir::ValueRange{stdoutFd, entry->getArgument(1), entry->getArgument(2),
                       entry->getArgument(3), entry->getArgument(4),
                       entry->getArgument(5)});
  if (newline) {
    mlir::Value newlineByte = mlir::arith::ConstantIntOp::create(
        b.builder, b.loc, b.i8(), 10);
    mlir::func::CallOp::create(b.builder, b.loc, "write_char",
                               mlir::TypeRange{},
                               mlir::ValueRange{stdoutFd, newlineByte});
  }
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
}

// ---------------------------------------------------------------------------
// Traceback cluster: the per-process frame stack, push/pop accounting, and the
// uncaught-exception printer (CPython-style traceback with source lines and
// `~~~^^` markers). Faithful translation of the former native module.
// ---------------------------------------------------------------------------

mlir::Type tracebackFrameType(SupportBuilder &b) {
  auto frame = mlir::LLVM::LLVMStructType::getIdentified(
      b.builder.getContext(), "TracebackFrame");
  if (frame.getBody().empty())
    (void)frame.setBody({b.ptr(), b.ptr(), b.i32(), b.i32(), b.i32(), b.i32(),
                         b.i32(), b.i32()},
                        /*isPacked=*/false);
  return frame;
}

mlir::Type tracebackStackType(SupportBuilder &b) {
  return mlir::LLVM::LLVMArrayType::get(tracebackFrameType(b), 1024);
}

void declareTracebackSupport(SupportBuilder &b) {
  b.declareExternal("malloc",
                    b.builder.getFunctionType({b.i64()}, {b.ptr()}));
  b.declareExternal("fopen", b.builder.getFunctionType({b.ptr(), b.ptr()},
                                                       {b.ptr()}));
  b.declareExternal("fgets", b.builder.getFunctionType(
                                 {b.ptr(), b.i32(), b.ptr()}, {b.ptr()}));
  b.declareExternal("fclose",
                    b.builder.getFunctionType({b.ptr()}, {b.i32()}));
  // Variadic: must be an llvm.func so the call carries the vararg callee type.
  if (!b.module.lookupSymbol("snprintf")) {
    mlir::OpBuilder::InsertionGuard guard(b.builder);
    b.builder.setInsertionPointToEnd(b.module.getBody());
    mlir::LLVM::LLVMFuncOp::create(
        b.builder, b.loc, "snprintf",
        mlir::LLVM::LLVMFunctionType::get(b.i32(),
                                          {b.ptr(), b.i64(), b.ptr()},
                                          /*isVarArg=*/true));
  }

  {
    mlir::OpBuilder::InsertionGuard guard(b.builder);
    b.builder.setInsertionPointToEnd(b.module.getBody());
    mlir::LLVM::GlobalOp::create(b.builder, b.loc, b.i64(),
                                 /*isConstant=*/false,
                                 mlir::LLVM::Linkage::Internal,
                                 "g_traceback_size",
                                 b.builder.getIntegerAttr(b.i64(), 0),
                                 /*alignment=*/8);
    auto stack = mlir::LLVM::GlobalOp::create(
        b.builder, b.loc, tracebackStackType(b), /*isConstant=*/false,
        mlir::LLVM::Linkage::Internal, "g_traceback_stack", mlir::Attribute(),
        /*alignment=*/8);
    mlir::Block *init = b.builder.createBlock(&stack.getInitializerRegion());
    b.builder.setInsertionPointToEnd(init);
    mlir::Value zero =
        mlir::LLVM::ZeroOp::create(b.builder, b.loc, tracebackStackType(b));
    mlir::LLVM::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{zero});
  }

  b.stringGlobal(".tb_read_mode", "r");
  b.stringGlobal(".tb_indent", "    ");
  b.stringGlobal(".tb_newline", "\n");
  b.stringGlobal(".tb_header", "Traceback (most recent call last):\n");
  b.stringGlobal(".tb_fmt_frame", "  File \"%s\", line %d, in %s\n");
  b.stringGlobal(".tb_fmt_class", "%s\n");
  b.stringGlobal(".tb_fmt_invalid", "%s: <invalid>\n");
  b.stringGlobal(".tb_fmt_unknown", "%s: <unknown>\n");
  b.stringGlobal(".tb_fmt_message", "%s: %s\n");
  for (llvm::StringRef name :
       {"BaseException", "Exception", "RuntimeError", "TypeError",
        "ValueError", "KeyError", "IndexError", "AssertionError",
        "StopIteration", "StopAsyncIteration", "ArithmeticError",
        "LookupError", "ZeroDivisionError", "CancelledError"})
    b.stringGlobal((".tb_class." + name).str(), name);
}

// ptr copy_cstr(ptr cstr): malloc'd NUL-terminated copy ("" for null input).
void buildCopyCStr(SupportBuilder &b) {
  auto fn = b.beginFunction("copy_cstr",
                            b.builder.getFunctionType({b.ptr()}, {b.ptr()}),
                            /*isPrivate=*/true);
  mlir::Block *entry = fn.addEntryBlock();
  mlir::Region &body = fn.getBody();
  mlir::Block *emptyCopy = b.builder.createBlock(&body);
  mlir::Block *emptyStore = b.builder.createBlock(&body);
  mlir::Block *realCopy = b.builder.createBlock(&body);
  mlir::Block *copyBytes = b.builder.createBlock(&body);
  mlir::Block *terminate = b.builder.createBlock(&body);
  mlir::Block *trap = b.builder.createBlock(&body);

  b.builder.setInsertionPointToEnd(entry);
  mlir::Value source = entry->getArgument(0);
  mlir::Value isNull = b.ptrEq(source, b.nullPtr());
  mlir::cf::CondBranchOp::create(b.builder, b.loc, isNull, emptyCopy,
                                 mlir::ValueRange{}, realCopy,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(emptyCopy);
  mlir::Value one = b.iconst(1);
  mlir::Value emptyBlock =
      b.call("malloc", b.ptr(), mlir::ValueRange{one}).front();
  mlir::Value emptyFailed = b.ptrEq(emptyBlock, b.nullPtr());
  mlir::cf::CondBranchOp::create(b.builder, b.loc, emptyFailed, trap,
                                 mlir::ValueRange{}, emptyStore,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(emptyStore);
  b.storeI8(b.iconst8(0), emptyBlock);
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{emptyBlock});

  b.builder.setInsertionPointToEnd(realCopy);
  mlir::Value length =
      b.call("strlen", b.i64(), mlir::ValueRange{source}).front();
  mlir::Value withNul =
      mlir::arith::AddIOp::create(b.builder, b.loc, length, b.iconst(1));
  mlir::Value block =
      b.call("malloc", b.ptr(), mlir::ValueRange{withNul}).front();
  mlir::Value failed = b.ptrEq(block, b.nullPtr());
  mlir::cf::CondBranchOp::create(b.builder, b.loc, failed, trap,
                                 mlir::ValueRange{}, copyBytes,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(copyBytes);
  mlir::Value hasBytes = b.cmpi(mlir::arith::CmpIPredicate::ne, length,
                                b.iconst(0));
  auto copyIf = mlir::scf::IfOp::create(b.builder, b.loc, mlir::TypeRange{},
                                        hasBytes, /*withElseRegion=*/false);
  {
    mlir::OpBuilder::InsertionGuard guard(b.builder);
    b.builder.setInsertionPointToStart(&copyIf.getThenRegion().front());
    mlir::LLVM::MemcpyOp::create(b.builder, b.loc, block, source, length,
                                 /*isVolatile=*/false);
  }
  mlir::cf::BranchOp::create(b.builder, b.loc, terminate, mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(terminate);
  b.storeI8(b.iconst8(0), b.gepI8(block, length));
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{block});

  b.builder.setInsertionPointToEnd(trap);
  b.emitTrap(b.ptr());
}

// ptr copy_i8_memref(ptr data, i64 offset, i64 len, i64 stride): malloc'd
// NUL-terminated copy of a strided byte view; invalid descriptors abort.
void buildCopyI8MemRef(SupportBuilder &b) {
  auto fn = b.beginFunction(
      "copy_i8_memref",
      b.builder.getFunctionType({b.ptr(), b.i64(), b.i64(), b.i64()},
                                {b.ptr()}),
      /*isPrivate=*/true);
  mlir::Block *entry = fn.addEntryBlock();
  mlir::Region &body = fn.getBody();
  mlir::Block *checkNull = b.builder.createBlock(&body);
  mlir::Block *allocate = b.builder.createBlock(&body);
  mlir::Block *loopHead = b.builder.createBlock(&body, body.end(), {b.i64()}, {b.loc});
  mlir::Block *loopBody = b.builder.createBlock(&body);
  mlir::Block *terminate = b.builder.createBlock(&body);
  mlir::Block *trap = b.builder.createBlock(&body);
  mlir::Value data = entry->getArgument(0);
  mlir::Value offset = entry->getArgument(1);
  mlir::Value len = entry->getArgument(2);
  mlir::Value stride = entry->getArgument(3);

  b.builder.setInsertionPointToEnd(entry);
  mlir::Value zero = b.iconst(0);
  mlir::Value one = b.iconst(1);
  mlir::Value offsetNeg =
      b.cmpi(mlir::arith::CmpIPredicate::slt, offset, zero);
  mlir::Value lenNeg = b.cmpi(mlir::arith::CmpIPredicate::slt, len, zero);
  mlir::Value strideBad =
      b.cmpi(mlir::arith::CmpIPredicate::slt, stride, one);
  mlir::Value invalid = b.orBit(b.orBit(offsetNeg, lenNeg), strideBad);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, invalid, trap,
                                 mlir::ValueRange{}, checkNull,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(checkNull);
  mlir::Value lenZero = b.cmpi(mlir::arith::CmpIPredicate::eq, len, zero);
  mlir::Value dataNull = b.ptrEq(data, b.nullPtr());
  mlir::Value lenNonZero = mlir::arith::XOrIOp::create(
      b.builder, b.loc, lenZero,
      mlir::arith::ConstantIntOp::create(b.builder, b.loc, 1, 1).getResult());
  mlir::Value nullWithBytes =
      mlir::arith::AndIOp::create(b.builder, b.loc, dataNull, lenNonZero);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, nullWithBytes, trap,
                                 mlir::ValueRange{}, allocate,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(allocate);
  mlir::Value withNul =
      mlir::arith::AddIOp::create(b.builder, b.loc, len, one);
  mlir::Value block =
      b.call("malloc", b.ptr(), mlir::ValueRange{withNul}).front();
  mlir::Value failed = b.ptrEq(block, b.nullPtr());
  mlir::cf::CondBranchOp::create(b.builder, b.loc, failed, trap,
                                 mlir::ValueRange{}, loopHead,
                                 mlir::ValueRange{zero});

  b.builder.setInsertionPointToEnd(loopHead);
  mlir::Value index = loopHead->getArgument(0);
  mlir::Value doneCopying =
      b.cmpi(mlir::arith::CmpIPredicate::eq, index, len);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, doneCopying, terminate,
                                 mlir::ValueRange{}, loopBody,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(loopBody);
  mlir::Value scaled =
      mlir::arith::MulIOp::create(b.builder, b.loc, index, stride);
  mlir::Value sourceIndex =
      mlir::arith::AddIOp::create(b.builder, b.loc, offset, scaled);
  mlir::Value byte = b.loadI8(b.gepI8(data, sourceIndex));
  b.storeI8(byte, b.gepI8(block, index));
  mlir::Value next = mlir::arith::AddIOp::create(b.builder, b.loc, index, one);
  mlir::cf::BranchOp::create(b.builder, b.loc, loopHead,
                             mlir::ValueRange{next});

  b.builder.setInsertionPointToEnd(terminate);
  b.storeI8(b.iconst8(0), b.gepI8(block, len));
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{block});

  b.builder.setInsertionPointToEnd(trap);
  b.emitTrap(b.ptr());
}

// ptr frame_at(i64 index): address of g_traceback_stack[index].
void buildFrameAt(SupportBuilder &b) {
  auto fn = b.beginFunction("frame_at",
                            b.builder.getFunctionType({b.i64()}, {b.ptr()}),
                            /*isPrivate=*/true);
  mlir::Block *entry = fn.addEntryBlock();
  b.builder.setInsertionPointToEnd(entry);
  mlir::Value stack = b.addrOf("g_traceback_stack");
  mlir::Value frame = mlir::LLVM::GEPOp::create(
      b.builder, b.loc, b.ptr(), tracebackStackType(b), stack,
      llvm::ArrayRef<mlir::LLVM::GEPArg>{mlir::LLVM::GEPArg(0),
                                         mlir::LLVM::GEPArg(
                                             entry->getArgument(0))},
      mlir::LLVM::GEPNoWrapFlags::inbounds);
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{frame});
}

// void free_frame(ptr frame): frees the two owned name copies.
void buildFreeFrame(SupportBuilder &b) {
  auto fn = b.beginFunction("free_frame",
                            b.builder.getFunctionType({b.ptr()}, {}),
                            /*isPrivate=*/true);
  mlir::Block *entry = fn.addEntryBlock();
  b.builder.setInsertionPointToEnd(entry);
  mlir::Type frameType = tracebackFrameType(b);
  mlir::Value frame = entry->getArgument(0);
  for (std::int32_t field : {0, 1}) {
    mlir::Value pointer = b.loadPtrVal(b.frameField(frameType, frame, field));
    mlir::Value present = b.ptrNe(pointer, b.nullPtr());
    auto freeIf = mlir::scf::IfOp::create(b.builder, b.loc, mlir::TypeRange{},
                                          present, /*withElseRegion=*/false);
    mlir::OpBuilder::InsertionGuard guard(b.builder);
    b.builder.setInsertionPointToStart(&freeIf.getThenRegion().front());
    b.call("free", mlir::TypeRange{}, mlir::ValueRange{pointer});
  }
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
}

// Shared frame-store tail for the two push entry points: writes the copied
// names, the five i32 words, bumps the stack size.
void emitFramePush(SupportBuilder &b, mlir::Value size, mlir::Value fileCopy,
                   mlir::Value functionCopy,
                   llvm::ArrayRef<mlir::Value> words) {
  mlir::Type frameType = tracebackFrameType(b);
  mlir::Value frame =
      b.call("frame_at", b.ptr(), mlir::ValueRange{size}).front();
  mlir::LLVM::StoreOp::create(b.builder, b.loc, fileCopy,
                              b.frameField(frameType, frame, 0),
                              /*alignment=*/8);
  mlir::LLVM::StoreOp::create(b.builder, b.loc, functionCopy,
                              b.frameField(frameType, frame, 1),
                              /*alignment=*/8);
  for (auto [index, word] : llvm::enumerate(words))
    mlir::LLVM::StoreOp::create(
        b.builder, b.loc, word,
        b.frameField(frameType, frame, 2 + static_cast<std::int32_t>(index)),
        /*alignment=*/4);
  mlir::Value bumped =
      mlir::arith::AddIOp::create(b.builder, b.loc, size, b.iconst(1));
  mlir::LLVM::StoreOp::create(b.builder, b.loc, bumped,
                              b.addrOf("g_traceback_size"), /*alignment=*/8);
}

mlir::Value loadTracebackSize(SupportBuilder &b) {
  return mlir::LLVM::LoadOp::create(b.builder, b.loc, b.i64(),
                                    b.addrOf("g_traceback_size"),
                                    /*alignment=*/8);
}

// LyTraceback_Push(file view: ptr/offset/size/stride via two descriptor arg
// groups, i32 line, i32 col): pushes a frame with copied names.
void buildTracebackPush(SupportBuilder &b) {
  auto fn = b.beginFunction(
      "LyTraceback_Push",
      b.builder.getFunctionType({b.ptr(), b.ptr(), b.i64(), b.i64(), b.i64(),
                                 b.ptr(), b.ptr(), b.i64(), b.i64(), b.i64(),
                                 b.i32(), b.i32()},
                                {}));
  mlir::Block *entry = fn.addEntryBlock();
  mlir::Region &body = fn.getBody();
  mlir::Block *push = b.builder.createBlock(&body);
  mlir::Block *trap = b.builder.createBlock(&body);

  b.builder.setInsertionPointToEnd(entry);
  mlir::Value size = loadTracebackSize(b);
  mlir::Value full =
      b.cmpi(mlir::arith::CmpIPredicate::uge, size, b.iconst(1024));
  mlir::cf::CondBranchOp::create(b.builder, b.loc, full, trap,
                                 mlir::ValueRange{}, push,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(push);
  mlir::Value fileCopy =
      b.call("copy_i8_memref", b.ptr(),
             mlir::ValueRange{entry->getArgument(1), entry->getArgument(2),
                              entry->getArgument(3), entry->getArgument(4)})
          .front();
  mlir::Value functionCopy =
      b.call("copy_i8_memref", b.ptr(),
             mlir::ValueRange{entry->getArgument(6), entry->getArgument(7),
                              entry->getArgument(8), entry->getArgument(9)})
          .front();
  mlir::Value zero32 = b.iconst32(0);
  emitFramePush(b, size, fileCopy, functionCopy,
                {entry->getArgument(10), entry->getArgument(11), zero32,
                 zero32, zero32});
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(trap);
  mlir::func::CallOp::create(b.builder, b.loc, "abort", mlir::TypeRange{},
                             mlir::ValueRange{});
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
}

// LyTraceback_PushCStringRange(file, function, line, col, endCol, colValid):
// C-string push carrying the caret range; marker flag set.
void buildTracebackPushCStringRange(SupportBuilder &b) {
  auto fn = b.beginFunction(
      "LyTraceback_PushCStringRange",
      b.builder.getFunctionType(
          {b.ptr(), b.ptr(), b.i32(), b.i32(), b.i32(), b.i32()}, {}));
  mlir::Block *entry = fn.addEntryBlock();
  mlir::Region &body = fn.getBody();
  mlir::Block *push = b.builder.createBlock(&body);
  mlir::Block *trap = b.builder.createBlock(&body);

  b.builder.setInsertionPointToEnd(entry);
  mlir::Value size = loadTracebackSize(b);
  mlir::Value full =
      b.cmpi(mlir::arith::CmpIPredicate::uge, size, b.iconst(1024));
  mlir::cf::CondBranchOp::create(b.builder, b.loc, full, trap,
                                 mlir::ValueRange{}, push,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(push);
  mlir::Value fileCopy =
      b.call("copy_cstr", b.ptr(), mlir::ValueRange{entry->getArgument(0)})
          .front();
  mlir::Value functionCopy =
      b.call("copy_cstr", b.ptr(), mlir::ValueRange{entry->getArgument(1)})
          .front();
  emitFramePush(b, size, fileCopy, functionCopy,
                {entry->getArgument(2), entry->getArgument(3),
                 entry->getArgument(4), entry->getArgument(5),
                 b.iconst32(1)});
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(trap);
  mlir::func::CallOp::create(b.builder, b.loc, "abort", mlir::TypeRange{},
                             mlir::ValueRange{});
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
}

void buildTracebackPushCString(SupportBuilder &b) {
  auto fn = b.beginFunction(
      "LyTraceback_PushCString",
      b.builder.getFunctionType({b.ptr(), b.ptr(), b.i32(), b.i32()}, {}));
  mlir::Block *entry = fn.addEntryBlock();
  b.builder.setInsertionPointToEnd(entry);
  b.call("LyTraceback_PushCStringRange", mlir::TypeRange{},
         mlir::ValueRange{entry->getArgument(0), entry->getArgument(1),
                          entry->getArgument(2), entry->getArgument(3),
                          entry->getArgument(2), b.iconst32(0)});
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
}

void buildTracebackPop(SupportBuilder &b) {
  auto fn =
      b.beginFunction("LyTraceback_Pop", b.builder.getFunctionType({}, {}));
  mlir::Block *entry = fn.addEntryBlock();
  b.builder.setInsertionPointToEnd(entry);
  mlir::Value size = loadTracebackSize(b);
  mlir::Value hasFrames =
      b.cmpi(mlir::arith::CmpIPredicate::ne, size, b.iconst(0));
  auto popIf = mlir::scf::IfOp::create(b.builder, b.loc, mlir::TypeRange{},
                                       hasFrames, /*withElseRegion=*/false);
  {
    mlir::OpBuilder::InsertionGuard guard(b.builder);
    b.builder.setInsertionPointToStart(&popIf.getThenRegion().front());
    mlir::Value top =
        mlir::arith::SubIOp::create(b.builder, b.loc, size, b.iconst(1));
    mlir::Value frame =
        b.call("frame_at", b.ptr(), mlir::ValueRange{top}).front();
    b.call("free_frame", mlir::TypeRange{}, mlir::ValueRange{frame});
    mlir::LLVM::StoreOp::create(b.builder, b.loc, top,
                                b.addrOf("g_traceback_size"),
                                /*alignment=*/8);
  }
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
}

void buildTracebackClear(SupportBuilder &b) {
  auto fn =
      b.beginFunction("LyTraceback_Clear", b.builder.getFunctionType({}, {}));
  mlir::Block *entry = fn.addEntryBlock();
  mlir::Region &body = fn.getBody();
  mlir::Block *head = b.builder.createBlock(&body);
  mlir::Block *popOne = b.builder.createBlock(&body);
  mlir::Block *done = b.builder.createBlock(&body);
  b.builder.setInsertionPointToEnd(entry);
  mlir::cf::BranchOp::create(b.builder, b.loc, head, mlir::ValueRange{});
  b.builder.setInsertionPointToEnd(head);
  mlir::Value size = loadTracebackSize(b);
  mlir::Value empty =
      b.cmpi(mlir::arith::CmpIPredicate::eq, size, b.iconst(0));
  mlir::cf::CondBranchOp::create(b.builder, b.loc, empty, done,
                                 mlir::ValueRange{}, popOne,
                                 mlir::ValueRange{});
  b.builder.setInsertionPointToEnd(popOne);
  b.call("LyTraceback_Pop", mlir::TypeRange{}, mlir::ValueRange{});
  mlir::cf::BranchOp::create(b.builder, b.loc, head, mlir::ValueRange{});
  b.builder.setInsertionPointToEnd(done);
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
}

// ptr read_source_line(ptr path, i32 line): malloc'd copy of the line-th
// source line ("" when unavailable), trailing newline characters stripped.
void buildReadSourceLine(SupportBuilder &b) {
  auto fn = b.beginFunction(
      "read_source_line",
      b.builder.getFunctionType({b.ptr(), b.i32()}, {b.ptr()}),
      /*isPrivate=*/true);
  mlir::Block *entry = fn.addEntryBlock();
  mlir::Region &body = fn.getBody();
  mlir::Block *checkArgs = b.builder.createBlock(&body);
  mlir::Block *open = b.builder.createBlock(&body);
  mlir::Block *readHead = b.builder.createBlock(&body, body.end(), {b.i32()}, {b.loc});
  mlir::Block *readCheck = b.builder.createBlock(&body);
  mlir::Block *readNext = b.builder.createBlock(&body);
  mlir::Block *eof = b.builder.createBlock(&body);
  mlir::Block *found = b.builder.createBlock(&body);
  mlir::Block *trimHead = b.builder.createBlock(&body, body.end(), {b.i64()}, {b.loc});
  mlir::Block *trimCheck = b.builder.createBlock(&body);
  mlir::Block *trimOne = b.builder.createBlock(&body);
  mlir::Block *finish = b.builder.createBlock(&body);
  mlir::Block *trap = b.builder.createBlock(&body);
  mlir::Value path = entry->getArgument(0);
  mlir::Value line = entry->getArgument(1);

  b.builder.setInsertionPointToEnd(entry);
  mlir::Value buffer =
      b.call("malloc", b.ptr(), mlir::ValueRange{b.iconst(512)}).front();
  mlir::Value failed = b.ptrEq(buffer, b.nullPtr());
  mlir::cf::CondBranchOp::create(b.builder, b.loc, failed, trap,
                                 mlir::ValueRange{}, checkArgs,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(checkArgs);
  b.storeI8(b.iconst8(0), buffer);
  mlir::Value pathNull = b.ptrEq(path, b.nullPtr());
  mlir::Value lineBad =
      b.cmpi(mlir::arith::CmpIPredicate::slt, line, b.iconst32(1));
  mlir::Value unusable = b.orBit(pathNull, lineBad);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, unusable, finish,
                                 mlir::ValueRange{}, open, mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(open);
  mlir::Value file = b.call("fopen", b.ptr(),
                            mlir::ValueRange{path, b.addrOf(".tb_read_mode")})
                         .front();
  mlir::Value openFailed = b.ptrEq(file, b.nullPtr());
  mlir::cf::CondBranchOp::create(b.builder, b.loc, openFailed, finish,
                                 mlir::ValueRange{}, readHead,
                                 mlir::ValueRange{b.iconst32(1)});

  b.builder.setInsertionPointToEnd(readHead);
  mlir::Value current = readHead->getArgument(0);
  mlir::Value got = b.call("fgets", b.ptr(),
                           mlir::ValueRange{buffer, b.iconst32(512), file})
                        .front();
  mlir::Value readFailed = b.ptrEq(got, b.nullPtr());
  mlir::cf::CondBranchOp::create(b.builder, b.loc, readFailed, eof,
                                 mlir::ValueRange{}, readCheck,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(readCheck);
  mlir::Value atLine = b.cmpi(mlir::arith::CmpIPredicate::eq, current, line);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, atLine, found,
                                 mlir::ValueRange{}, readNext,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(readNext);
  mlir::Value nextLine =
      mlir::arith::AddIOp::create(b.builder, b.loc, current, b.iconst32(1));
  mlir::cf::BranchOp::create(b.builder, b.loc, readHead,
                             mlir::ValueRange{nextLine});

  b.builder.setInsertionPointToEnd(eof);
  b.storeI8(b.iconst8(0), buffer);
  b.call("fclose", b.i32(), mlir::ValueRange{file});
  mlir::cf::BranchOp::create(b.builder, b.loc, finish, mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(found);
  b.call("fclose", b.i32(), mlir::ValueRange{file});
  mlir::Value initialLength =
      b.call("strlen", b.i64(), mlir::ValueRange{buffer}).front();
  mlir::cf::BranchOp::create(b.builder, b.loc, trimHead,
                             mlir::ValueRange{initialLength});

  b.builder.setInsertionPointToEnd(trimHead);
  mlir::Value remaining = trimHead->getArgument(0);
  mlir::Value trimDone =
      b.cmpi(mlir::arith::CmpIPredicate::eq, remaining, b.iconst(0));
  mlir::cf::CondBranchOp::create(b.builder, b.loc, trimDone, finish,
                                 mlir::ValueRange{}, trimCheck,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(trimCheck);
  mlir::Value lastIndex =
      mlir::arith::SubIOp::create(b.builder, b.loc, remaining, b.iconst(1));
  mlir::Value lastPtr = b.gepI8(buffer, lastIndex);
  mlir::Value last = b.loadI8(lastPtr);
  mlir::Value isNewline =
      b.cmpi(mlir::arith::CmpIPredicate::eq, last, b.iconst8(10));
  mlir::Value isReturn =
      b.cmpi(mlir::arith::CmpIPredicate::eq, last, b.iconst8(13));
  mlir::Value trimIt = b.orBit(isNewline, isReturn);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, trimIt, trimOne,
                                 mlir::ValueRange{}, finish,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(trimOne);
  b.storeI8(b.iconst8(0), lastPtr);
  mlir::cf::BranchOp::create(b.builder, b.loc, trimHead,
                             mlir::ValueRange{lastIndex});

  b.builder.setInsertionPointToEnd(finish);
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{buffer});

  b.builder.setInsertionPointToEnd(trap);
  b.emitTrap(b.ptr());
}

// ptr exception_class_name(i64 class_id): builtin exception-class name table
// (value selection; unknown ids display as "Exception").
void buildExceptionClassName(SupportBuilder &b) {
  auto fn = b.beginFunction(
      "exception_class_name",
      b.builder.getFunctionType({b.i64()}, {b.ptr()}), /*isPrivate=*/true);
  mlir::Block *entry = fn.addEntryBlock();
  b.builder.setInsertionPointToEnd(entry);
  mlir::Value classId = entry->getArgument(0);
  const std::pair<std::int64_t, const char *> table[] = {
      {5, ".tb_class.BaseException"},   {50, ".tb_class.Exception"},
      {51, ".tb_class.RuntimeError"},   {52, ".tb_class.TypeError"},
      {53, ".tb_class.ValueError"},     {54, ".tb_class.KeyError"},
      {55, ".tb_class.IndexError"},     {56, ".tb_class.AssertionError"},
      {57, ".tb_class.StopIteration"},  {58, ".tb_class.StopAsyncIteration"},
      {59, ".tb_class.ArithmeticError"}, {60, ".tb_class.LookupError"},
      {61, ".tb_class.ZeroDivisionError"}, {62, ".tb_class.CancelledError"},
  };
  mlir::Value name = b.addrOf(".tb_class.Exception");
  for (const auto &[id, global] : table) {
    mlir::Value matches =
        b.cmpi(mlir::arith::CmpIPredicate::eq, classId, b.iconst(id));
    name = mlir::arith::SelectOp::create(b.builder, b.loc, matches,
                                         b.addrOf(global), name);
  }
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{name});
}

// i64 leading_whitespace(ptr line): count of leading spaces/tabs.
void buildLeadingWhitespace(SupportBuilder &b) {
  auto fn = b.beginFunction(
      "leading_whitespace",
      b.builder.getFunctionType({b.ptr()}, {b.i64()}), /*isPrivate=*/true);
  mlir::Block *entry = fn.addEntryBlock();
  mlir::Region &body = fn.getBody();
  mlir::Block *head = b.builder.createBlock(&body, body.end(), {b.i64()}, {b.loc});
  mlir::Block *check = b.builder.createBlock(&body);
  mlir::Block *advance = b.builder.createBlock(&body);
  mlir::Block *done = b.builder.createBlock(&body, body.end(), {b.i64()}, {b.loc});
  mlir::Value line = entry->getArgument(0);

  b.builder.setInsertionPointToEnd(entry);
  mlir::Value length =
      b.call("strlen", b.i64(), mlir::ValueRange{line}).front();
  mlir::cf::BranchOp::create(b.builder, b.loc, head,
                             mlir::ValueRange{b.iconst(0)});

  b.builder.setInsertionPointToEnd(head);
  mlir::Value index = head->getArgument(0);
  mlir::Value atEnd = b.cmpi(mlir::arith::CmpIPredicate::eq, index, length);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, atEnd, done,
                                 mlir::ValueRange{index}, check,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(check);
  mlir::Value ch = b.loadI8(b.gepI8(line, index));
  mlir::Value isSpace =
      b.cmpi(mlir::arith::CmpIPredicate::eq, ch, b.iconst8(32));
  mlir::Value isTab = b.cmpi(mlir::arith::CmpIPredicate::eq, ch, b.iconst8(9));
  mlir::Value isBlank = b.orBit(isSpace, isTab);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, isBlank, advance,
                                 mlir::ValueRange{}, done,
                                 mlir::ValueRange{index});

  b.builder.setInsertionPointToEnd(advance);
  mlir::Value next =
      mlir::arith::AddIOp::create(b.builder, b.loc, index, b.iconst(1));
  mlir::cf::BranchOp::create(b.builder, b.loc, head, mlir::ValueRange{next});

  b.builder.setInsertionPointToEnd(done);
  mlir::func::ReturnOp::create(b.builder, b.loc,
                               mlir::ValueRange{done->getArgument(0)});
}

// void print_marker(ptr line, i32 col, i32 endCol): the CPython-style
// `    ~~~~~^^` underline for the failing range on stderr.
void buildPrintMarker(SupportBuilder &b) {
  auto fn = b.beginFunction(
      "print_marker",
      b.builder.getFunctionType({b.ptr(), b.i32(), b.i32()}, {}),
      /*isPrivate=*/true);
  mlir::Block *entry = fn.addEntryBlock();
  mlir::Region &body = fn.getBody();
  mlir::Block *findStart = b.builder.createBlock(&body);
  mlir::Block *scanHead = b.builder.createBlock(&body, body.end(), {b.i64()}, {b.loc});
  mlir::Block *scanCheck = b.builder.createBlock(&body);
  mlir::Block *scanNext = b.builder.createBlock(&body);
  mlir::Block *haveStart = b.builder.createBlock(&body, body.end(), {b.i64()}, {b.loc});
  mlir::Block *emit = b.builder.createBlock(&body);
  mlir::Block *padHead = b.builder.createBlock(&body, body.end(), {b.i64()}, {b.loc});
  mlir::Block *padOne = b.builder.createBlock(&body);
  mlir::Block *markers = b.builder.createBlock(&body);
  mlir::Block *caretsHead = b.builder.createBlock(&body, body.end(), {b.i64()}, {b.loc});
  mlir::Block *caretsOne = b.builder.createBlock(&body);
  mlir::Block *tildes = b.builder.createBlock(&body);
  mlir::Block *tildesHead = b.builder.createBlock(&body, body.end(), {b.i64()}, {b.loc});
  mlir::Block *tildesOne = b.builder.createBlock(&body);
  mlir::Block *tildesEnd = b.builder.createBlock(&body);
  mlir::Block *newline = b.builder.createBlock(&body);
  mlir::Block *done = b.builder.createBlock(&body);
  mlir::Value line = entry->getArgument(0);
  mlir::Value col = entry->getArgument(1);
  mlir::Value endCol = entry->getArgument(2);

  b.builder.setInsertionPointToEnd(entry);
  mlir::Value length =
      b.call("strlen", b.i64(), mlir::ValueRange{line}).front();
  mlir::Value emptyLine =
      b.cmpi(mlir::arith::CmpIPredicate::eq, length, b.iconst(0));
  mlir::cf::CondBranchOp::create(b.builder, b.loc, emptyLine, done,
                                 mlir::ValueRange{}, findStart,
                                 mlir::ValueRange{});

  // Marker start: the given column when it lands inside the line, otherwise
  // the first non-blank character.
  b.builder.setInsertionPointToEnd(findStart);
  mlir::Value colPositive =
      b.cmpi(mlir::arith::CmpIPredicate::sgt, col, b.iconst32(0));
  mlir::Value colWide =
      mlir::arith::ExtSIOp::create(b.builder, b.loc, b.i64(), col);
  mlir::Value colInLine =
      b.cmpi(mlir::arith::CmpIPredicate::slt, colWide, length);
  mlir::Value useColumn =
      mlir::arith::AndIOp::create(b.builder, b.loc, colPositive, colInLine);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, useColumn, haveStart,
                                 mlir::ValueRange{colWide}, scanHead,
                                 mlir::ValueRange{b.iconst(0)});

  b.builder.setInsertionPointToEnd(scanHead);
  mlir::Value scanIndex = scanHead->getArgument(0);
  mlir::Value scanEnd =
      b.cmpi(mlir::arith::CmpIPredicate::eq, scanIndex, length);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, scanEnd, haveStart,
                                 mlir::ValueRange{scanIndex}, scanCheck,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(scanCheck);
  mlir::Value scanCh = b.loadI8(b.gepI8(line, scanIndex));
  mlir::Value scanSpace =
      b.cmpi(mlir::arith::CmpIPredicate::eq, scanCh, b.iconst8(32));
  mlir::Value scanTab =
      b.cmpi(mlir::arith::CmpIPredicate::eq, scanCh, b.iconst8(9));
  mlir::Value scanBlank = b.orBit(scanSpace, scanTab);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, scanBlank, scanNext,
                                 mlir::ValueRange{}, haveStart,
                                 mlir::ValueRange{scanIndex});

  b.builder.setInsertionPointToEnd(scanNext);
  mlir::Value scanAdvance =
      mlir::arith::AddIOp::create(b.builder, b.loc, scanIndex, b.iconst(1));
  mlir::cf::BranchOp::create(b.builder, b.loc, scanHead,
                             mlir::ValueRange{scanAdvance});

  b.builder.setInsertionPointToEnd(haveStart);
  mlir::Value start = haveStart->getArgument(0);
  mlir::Value startPastEnd =
      b.cmpi(mlir::arith::CmpIPredicate::uge, start, length);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, startPastEnd, done,
                                 mlir::ValueRange{}, emit,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(emit);
  // Marker end: endCol when it is a usable range end, clamped to the line;
  // degenerate ranges underline a single character.
  mlir::Value endAfterCol =
      b.cmpi(mlir::arith::CmpIPredicate::sgt, endCol, col);
  mlir::Value endPositive =
      b.cmpi(mlir::arith::CmpIPredicate::sgt, endCol, b.iconst32(0));
  mlir::Value endUsable =
      mlir::arith::AndIOp::create(b.builder, b.loc, endAfterCol, endPositive);
  mlir::Value endWide =
      mlir::arith::ExtSIOp::create(b.builder, b.loc, b.i64(), endCol);
  mlir::Value endOverLength =
      b.cmpi(mlir::arith::CmpIPredicate::sgt, endWide, length);
  mlir::Value endClamped = mlir::arith::SelectOp::create(
      b.builder, b.loc, endOverLength, length, endWide);
  mlir::Value endOrLength = mlir::arith::SelectOp::create(
      b.builder, b.loc, endUsable, endClamped, length);
  mlir::Value endTooSmall =
      b.cmpi(mlir::arith::CmpIPredicate::ule, endOrLength, start);
  mlir::Value startPlusOne =
      mlir::arith::AddIOp::create(b.builder, b.loc, start, b.iconst(1));
  mlir::Value plusOneOver =
      b.cmpi(mlir::arith::CmpIPredicate::sgt, startPlusOne, length);
  mlir::Value plusOneClamped = mlir::arith::SelectOp::create(
      b.builder, b.loc, plusOneOver, length, startPlusOne);
  mlir::Value markerEnd = mlir::arith::SelectOp::create(
      b.builder, b.loc, endTooSmall, plusOneClamped, endOrLength);
  mlir::Value markerWidth =
      mlir::arith::SubIOp::create(b.builder, b.loc, markerEnd, start);
  mlir::Value stderrFd = b.iconst32(2);
  b.call("write_cstr", mlir::TypeRange{},
         mlir::ValueRange{stderrFd, b.addrOf(".tb_indent")});
  mlir::cf::BranchOp::create(b.builder, b.loc, padHead,
                             mlir::ValueRange{b.iconst(0)});

  // Alignment padding: tabs stay tabs so the marker lines up under the code.
  b.builder.setInsertionPointToEnd(padHead);
  mlir::Value padIndex = padHead->getArgument(0);
  mlir::Value padDone =
      b.cmpi(mlir::arith::CmpIPredicate::eq, padIndex, start);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, padDone, markers,
                                 mlir::ValueRange{}, padOne,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(padOne);
  mlir::Value padCh = b.loadI8(b.gepI8(line, padIndex));
  mlir::Value padIsTab =
      b.cmpi(mlir::arith::CmpIPredicate::eq, padCh, b.iconst8(9));
  mlir::Value padOut = mlir::arith::SelectOp::create(
      b.builder, b.loc, padIsTab, b.iconst8(9), b.iconst8(32));
  b.call("write_char", mlir::TypeRange{},
         mlir::ValueRange{b.iconst32(2), padOut});
  mlir::Value padNext =
      mlir::arith::AddIOp::create(b.builder, b.loc, padIndex, b.iconst(1));
  mlir::cf::BranchOp::create(b.builder, b.loc, padHead,
                             mlir::ValueRange{padNext});

  // Width <= 2 renders carets only; wider ranges render tildes with a two
  // caret tail.
  b.builder.setInsertionPointToEnd(markers);
  mlir::Value narrow =
      b.cmpi(mlir::arith::CmpIPredicate::ule, markerWidth, b.iconst(2));
  mlir::cf::CondBranchOp::create(b.builder, b.loc, narrow, caretsHead,
                                 mlir::ValueRange{b.iconst(0)}, tildes,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(caretsHead);
  mlir::Value caretIndex = caretsHead->getArgument(0);
  mlir::Value caretsDone =
      b.cmpi(mlir::arith::CmpIPredicate::eq, caretIndex, markerWidth);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, caretsDone, newline,
                                 mlir::ValueRange{}, caretsOne,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(caretsOne);
  b.call("write_char", mlir::TypeRange{},
         mlir::ValueRange{b.iconst32(2), b.iconst8(94)});
  mlir::Value caretNext =
      mlir::arith::AddIOp::create(b.builder, b.loc, caretIndex, b.iconst(1));
  mlir::cf::BranchOp::create(b.builder, b.loc, caretsHead,
                             mlir::ValueRange{caretNext});

  b.builder.setInsertionPointToEnd(tildes);
  mlir::Value tildeCount =
      mlir::arith::SubIOp::create(b.builder, b.loc, markerWidth, b.iconst(2));
  mlir::cf::BranchOp::create(b.builder, b.loc, tildesHead,
                             mlir::ValueRange{b.iconst(0)});

  b.builder.setInsertionPointToEnd(tildesHead);
  mlir::Value tildeIndex = tildesHead->getArgument(0);
  mlir::Value tildesDone =
      b.cmpi(mlir::arith::CmpIPredicate::eq, tildeIndex, tildeCount);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, tildesDone, tildesEnd,
                                 mlir::ValueRange{}, tildesOne,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(tildesOne);
  b.call("write_char", mlir::TypeRange{},
         mlir::ValueRange{b.iconst32(2), b.iconst8(126)});
  mlir::Value tildeNext =
      mlir::arith::AddIOp::create(b.builder, b.loc, tildeIndex, b.iconst(1));
  mlir::cf::BranchOp::create(b.builder, b.loc, tildesHead,
                             mlir::ValueRange{tildeNext});

  b.builder.setInsertionPointToEnd(tildesEnd);
  b.call("write_char", mlir::TypeRange{},
         mlir::ValueRange{b.iconst32(2), b.iconst8(94)});
  b.call("write_char", mlir::TypeRange{},
         mlir::ValueRange{b.iconst32(2), b.iconst8(94)});
  mlir::cf::BranchOp::create(b.builder, b.loc, newline, mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(newline);
  b.call("write_len", mlir::TypeRange{},
         mlir::ValueRange{b.iconst32(2), b.addrOf(".tb_newline"),
                          b.iconst(1)});
  mlir::cf::BranchOp::create(b.builder, b.loc, done, mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(done);
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
}

// void print_trace_frame(ptr frame): "  File ..., line N, in fn" + the source
// line + optional marker, on stderr.
void buildPrintTraceFrame(SupportBuilder &b) {
  auto fn = b.beginFunction("print_trace_frame",
                            b.builder.getFunctionType({b.ptr()}, {}),
                            /*isPrivate=*/true);
  mlir::Block *entry = fn.addEntryBlock();
  mlir::Region &body = fn.getBody();
  mlir::Block *sourceShown = b.builder.createBlock(&body);
  mlir::Block *marker = b.builder.createBlock(&body);
  mlir::Block *done = b.builder.createBlock(&body);
  mlir::Value frame = entry->getArgument(0);

  b.builder.setInsertionPointToEnd(entry);
  mlir::Type frameType = tracebackFrameType(b);
  auto bufferType = mlir::LLVM::LLVMArrayType::get(b.i8(), 1024);
  mlir::Value bufferSlot = mlir::LLVM::AllocaOp::create(
      b.builder, b.loc, b.ptr(), bufferType, b.iconst32(1), /*alignment=*/1);
  mlir::Value buffer = mlir::LLVM::GEPOp::create(
      b.builder, b.loc, b.ptr(), bufferType, bufferSlot,
      llvm::ArrayRef<mlir::LLVM::GEPArg>{mlir::LLVM::GEPArg(0),
                                         mlir::LLVM::GEPArg(0)},
      mlir::LLVM::GEPNoWrapFlags::inbounds);
  mlir::Value file = b.loadPtrVal(b.frameField(frameType, frame, 0));
  mlir::Value function = b.loadPtrVal(b.frameField(frameType, frame, 1));
  mlir::Value lineNo = b.loadI32(b.frameField(frameType, frame, 2));
  mlir::Value col = b.loadI32(b.frameField(frameType, frame, 3));
  mlir::Value endCol = b.loadI32(b.frameField(frameType, frame, 5));
  mlir::Value hasMarker = b.loadI32(b.frameField(frameType, frame, 6));
  auto snprintfType = mlir::LLVM::LLVMFunctionType::get(
      b.i32(), {b.ptr(), b.i64(), b.ptr()}, /*isVarArg=*/true);
  auto formatted = mlir::LLVM::CallOp::create(
      b.builder, b.loc, snprintfType, "snprintf",
      mlir::ValueRange{buffer, b.iconst(1024), b.addrOf(".tb_fmt_frame"),
                       file, lineNo, function});
  b.call("write_buffered", mlir::TypeRange{},
         mlir::ValueRange{b.iconst32(2), buffer, formatted.getResult()});
  mlir::Value sourceLine =
      b.call("read_source_line", b.ptr(), mlir::ValueRange{file, lineNo})
          .front();
  mlir::Value sourceLength =
      b.call("strlen", b.i64(), mlir::ValueRange{sourceLine}).front();
  mlir::Value haveSource =
      b.cmpi(mlir::arith::CmpIPredicate::ne, sourceLength, b.iconst(0));
  mlir::cf::CondBranchOp::create(b.builder, b.loc, haveSource, sourceShown,
                                 mlir::ValueRange{}, done,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(sourceShown);
  mlir::Value indentWidth =
      b.call("leading_whitespace", b.i64(), mlir::ValueRange{sourceLine})
          .front();
  mlir::Value trimmed = b.gepI8(sourceLine, indentWidth);
  mlir::Value trimmedLength =
      b.call("strlen", b.i64(), mlir::ValueRange{trimmed}).front();
  b.call("write_cstr", mlir::TypeRange{},
         mlir::ValueRange{b.iconst32(2), b.addrOf(".tb_indent")});
  b.call("write_len", mlir::TypeRange{},
         mlir::ValueRange{b.iconst32(2), trimmed, trimmedLength});
  b.call("write_len", mlir::TypeRange{},
         mlir::ValueRange{b.iconst32(2), b.addrOf(".tb_newline"),
                          b.iconst(1)});
  mlir::Value wantMarker =
      b.cmpi(mlir::arith::CmpIPredicate::ne, hasMarker, b.iconst32(0));
  mlir::cf::CondBranchOp::create(b.builder, b.loc, wantMarker, marker,
                                 mlir::ValueRange{}, done,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(marker);
  // Columns are absolute; the printed line lost its indentation.
  mlir::Value indent32 =
      mlir::arith::TruncIOp::create(b.builder, b.loc, b.i32(), indentWidth);
  mlir::Value colPast =
      b.cmpi(mlir::arith::CmpIPredicate::sgt, col, indent32);
  mlir::Value colShift =
      mlir::arith::SubIOp::create(b.builder, b.loc, col, indent32);
  mlir::Value colAdjusted = mlir::arith::SelectOp::create(
      b.builder, b.loc, colPast, colShift, b.iconst32(0));
  mlir::Value endPast =
      b.cmpi(mlir::arith::CmpIPredicate::sgt, endCol, indent32);
  mlir::Value endShift =
      mlir::arith::SubIOp::create(b.builder, b.loc, endCol, indent32);
  mlir::Value endAdjusted = mlir::arith::SelectOp::create(
      b.builder, b.loc, endPast, endShift, b.iconst32(0));
  b.call("print_marker", mlir::TypeRange{},
         mlir::ValueRange{trimmed, colAdjusted, endAdjusted});
  mlir::cf::BranchOp::create(b.builder, b.loc, done, mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(done);
  b.call("free", mlir::TypeRange{}, mlir::ValueRange{sourceLine});
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
}

// void print_exception_summary(i64 class_id, message view): the final
// "Class: message" line (or the class-only / invalid / unknown forms).
void buildPrintExceptionSummary(SupportBuilder &b) {
  auto fn = b.beginFunction(
      "print_exception_summary",
      b.builder.getFunctionType({b.i64(), b.ptr(), b.i64(), b.i64(), b.i64()},
                                {}),
      /*isPrivate=*/true);
  mlir::Block *entry = fn.addEntryBlock();
  mlir::Region &body = fn.getBody();
  mlir::Block *checkEmpty = b.builder.createBlock(&body);
  mlir::Block *checkNull = b.builder.createBlock(&body);
  mlir::Block *withMessage = b.builder.createBlock(&body);
  mlir::Block *classOnly = b.builder.createBlock(&body);
  mlir::Block *invalid = b.builder.createBlock(&body);
  mlir::Block *unknown = b.builder.createBlock(&body);
  mlir::Value classId = entry->getArgument(0);
  mlir::Value data = entry->getArgument(1);
  mlir::Value offset = entry->getArgument(2);
  mlir::Value len = entry->getArgument(3);
  mlir::Value stride = entry->getArgument(4);

  b.builder.setInsertionPointToEnd(entry);
  auto bufferType = mlir::LLVM::LLVMArrayType::get(b.i8(), 1024);
  mlir::Value bufferSlot = mlir::LLVM::AllocaOp::create(
      b.builder, b.loc, b.ptr(), bufferType, b.iconst32(1), /*alignment=*/1);
  mlir::Value buffer = mlir::LLVM::GEPOp::create(
      b.builder, b.loc, b.ptr(), bufferType, bufferSlot,
      llvm::ArrayRef<mlir::LLVM::GEPArg>{mlir::LLVM::GEPArg(0),
                                         mlir::LLVM::GEPArg(0)},
      mlir::LLVM::GEPNoWrapFlags::inbounds);
  mlir::Value className =
      b.call("exception_class_name", b.ptr(), mlir::ValueRange{classId})
          .front();
  auto snprintfType = mlir::LLVM::LLVMFunctionType::get(
      b.i32(), {b.ptr(), b.i64(), b.ptr()}, /*isVarArg=*/true);
  auto emitBuffered = [&](mlir::Value formattedLength) {
    b.call("write_buffered", mlir::TypeRange{},
           mlir::ValueRange{b.iconst32(2), buffer, formattedLength});
    mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
  };
  mlir::Value zero = b.iconst(0);
  mlir::Value offsetNeg =
      b.cmpi(mlir::arith::CmpIPredicate::slt, offset, zero);
  mlir::Value lenNeg = b.cmpi(mlir::arith::CmpIPredicate::slt, len, zero);
  mlir::Value strideBad =
      b.cmpi(mlir::arith::CmpIPredicate::slt, stride, b.iconst(1));
  mlir::Value badView = b.orBit(b.orBit(offsetNeg, lenNeg), strideBad);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, badView, invalid,
                                 mlir::ValueRange{}, checkEmpty,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(checkEmpty);
  mlir::Value emptyMessage =
      b.cmpi(mlir::arith::CmpIPredicate::eq, len, zero);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, emptyMessage, classOnly,
                                 mlir::ValueRange{}, checkNull,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(checkNull);
  mlir::Value dataNull = b.ptrEq(data, b.nullPtr());
  mlir::cf::CondBranchOp::create(b.builder, b.loc, dataNull, unknown,
                                 mlir::ValueRange{}, withMessage,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(withMessage);
  mlir::Value message =
      b.call("copy_i8_memref", b.ptr(),
             mlir::ValueRange{data, offset, len, stride})
          .front();
  auto formattedMessage = mlir::LLVM::CallOp::create(
      b.builder, b.loc, snprintfType, "snprintf",
      mlir::ValueRange{buffer, b.iconst(1024), b.addrOf(".tb_fmt_message"),
                       className, message});
  b.call("write_buffered", mlir::TypeRange{},
         mlir::ValueRange{b.iconst32(2), buffer,
                          formattedMessage.getResult()});
  b.call("free", mlir::TypeRange{}, mlir::ValueRange{message});
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(classOnly);
  auto formattedClass = mlir::LLVM::CallOp::create(
      b.builder, b.loc, snprintfType, "snprintf",
      mlir::ValueRange{buffer, b.iconst(1024), b.addrOf(".tb_fmt_class"),
                       className});
  emitBuffered(formattedClass.getResult());

  b.builder.setInsertionPointToEnd(invalid);
  auto formattedInvalid = mlir::LLVM::CallOp::create(
      b.builder, b.loc, snprintfType, "snprintf",
      mlir::ValueRange{buffer, b.iconst(1024), b.addrOf(".tb_fmt_invalid"),
                       className});
  emitBuffered(formattedInvalid.getResult());

  b.builder.setInsertionPointToEnd(unknown);
  auto formattedUnknown = mlir::LLVM::CallOp::create(
      b.builder, b.loc, snprintfType, "snprintf",
      mlir::ValueRange{buffer, b.iconst(1024), b.addrOf(".tb_fmt_unknown"),
                       className});
  emitBuffered(formattedUnknown.getResult());
}

// LyTraceback_PrintMessage(i64 class_id, ptr unused, message view): header +
// frames (most recent last, printed from the top of the stack downwards) +
// summary line, on stderr.
void buildTracebackPrintMessage(SupportBuilder &b) {
  auto fn = b.beginFunction(
      "LyTraceback_PrintMessage",
      b.builder.getFunctionType(
          {b.i64(), b.ptr(), b.ptr(), b.i64(), b.i64(), b.i64()}, {}));
  mlir::Block *entry = fn.addEntryBlock();
  mlir::Region &body = fn.getBody();
  mlir::Block *head = b.builder.createBlock(&body, body.end(), {b.i64()}, {b.loc});
  mlir::Block *printOne = b.builder.createBlock(&body);
  mlir::Block *summary = b.builder.createBlock(&body);

  b.builder.setInsertionPointToEnd(entry);
  b.call("write_cstr", mlir::TypeRange{},
         mlir::ValueRange{b.iconst32(2), b.addrOf(".tb_header")});
  mlir::cf::BranchOp::create(b.builder, b.loc, head,
                             mlir::ValueRange{loadTracebackSize(b)});

  b.builder.setInsertionPointToEnd(head);
  mlir::Value remaining = head->getArgument(0);
  mlir::Value doneFrames =
      b.cmpi(mlir::arith::CmpIPredicate::eq, remaining, b.iconst(0));
  mlir::cf::CondBranchOp::create(b.builder, b.loc, doneFrames, summary,
                                 mlir::ValueRange{}, printOne,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(printOne);
  mlir::Value top =
      mlir::arith::SubIOp::create(b.builder, b.loc, remaining, b.iconst(1));
  mlir::Value frame =
      b.call("frame_at", b.ptr(), mlir::ValueRange{top}).front();
  b.call("print_trace_frame", mlir::TypeRange{}, mlir::ValueRange{frame});
  mlir::cf::BranchOp::create(b.builder, b.loc, head, mlir::ValueRange{top});

  b.builder.setInsertionPointToEnd(summary);
  b.call("print_exception_summary", mlir::TypeRange{},
         mlir::ValueRange{entry->getArgument(0), entry->getArgument(2),
                          entry->getArgument(3), entry->getArgument(4),
                          entry->getArgument(5)});
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
}

// ---------------------------------------------------------------------------
// Exception-handling core: the Itanium C++ ABI bridge (LyPythonException as a
// 1-byte C++ exception carrying its payload in process globals), the current
// exception slot, and the program entry (LyRunPythonMain). Irreducibly llvm
// dialect: personality, invoke/landingpad, __cxa_* and typeinfo globals.
// ---------------------------------------------------------------------------

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
  mlir::Block *store = b.builder.createBlock(&body);
  mlir::Block *trap = b.builder.createBlock(&body);
  b.builder.setInsertionPointToEnd(entry);
  mlir::Value flagSlot = b.addrOf("g_current_exception");
  mlir::Value pending = mlir::LLVM::LoadOp::create(b.builder, b.loc, b.i1(),
                                                   flagSlot, /*alignment=*/4);
  mlir::LLVM::CondBrOp::create(b.builder, b.loc, pending, trap, store);
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
  b.builder.setInsertionPointToEnd(trap);
  emitLLVMTrap(b);
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
void buildDiscardCurrentException(SupportBuilder &b) {
  auto fn = b.beginFunction("LyEH_DiscardCurrentException",
                            b.builder.getFunctionType({}, {}));
  mlir::Block *entry = fn.addEntryBlock();
  mlir::Region &body = fn.getBody();
  mlir::Block *release = b.builder.createBlock(&body);
  mlir::Block *freeBlocks = b.builder.createBlock(&body);
  mlir::Block *clear = b.builder.createBlock(&body);
  b.builder.setInsertionPointToEnd(entry);
  b.call("end_native_catch_if_active", mlir::TypeRange{}, {});
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
  mlir::func::CallOp::create(
      b.builder, b.loc, "LyTraceback_PrintMessage", mlir::TypeRange{},
      mlir::ValueRange{classId, b.nullPtr(), messageData, messageOffset,
                       messageLen, messageStride});
  mlir::func::CallOp::create(b.builder, b.loc, "LyTraceback_Clear",
                             mlir::TypeRange{}, mlir::ValueRange{});
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
  buildReleaseSingleAllocation(support, "release_unicode_raw", /*twoArgs=*/true);
  buildWriteLen(support);
  buildWriteCStr(support);
  buildWriteChar(support);
  buildWriteBuffered(support);
  buildBoxedIntValue(support);
  buildPrintBytes(support);
  buildHostPrint(support, "LyHost_Print", /*newline=*/false);
  buildHostPrint(support, "LyHost_PrintLine", /*newline=*/true);
  buildReleasePayloadSlotPtr(support);
  buildReleaseBoxedPayloadRaw(support);
  buildReleaseBoxedPayloadArraySlotRaw(support);
  declareTracebackSupport(support);
  buildCopyCStr(support);
  buildCopyI8MemRef(support);
  buildFrameAt(support);
  buildFreeFrame(support);
  buildTracebackPush(support);
  buildTracebackPushCStringRange(support);
  buildTracebackPushCString(support);
  buildTracebackPop(support);
  buildTracebackClear(support);
  buildReadSourceLine(support);
  buildExceptionClassName(support);
  buildLeadingWhitespace(support);
  buildPrintMarker(support);
  buildPrintTraceFrame(support);
  buildPrintExceptionSummary(support);
  buildTracebackPrintMessage(support);
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
  buildRunPythonMain(support);

  return module;
}

} // namespace py::runtime_library
