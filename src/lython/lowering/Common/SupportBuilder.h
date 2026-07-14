#pragma once

// Shared facade for the native runtime support builders (RuntimeSupportBuilder
// and TracebackSupportBuilder compose the same module).

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include <cstdint>
#include <string>

namespace py::runtime_library {

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

// Emits the host-boundary cluster (raw write / exit status / argv / FILE*
// and buffer wrappers); implemented in HostSupportBuilder.cpp.
void buildHostSupport(SupportBuilder &b);

// Emits the traceback cluster (frame stack, push/pop accounting, uncaught
// exception printer) into the module; implemented in
// TracebackSupportBuilder.cpp.
void buildTracebackSupport(SupportBuilder &b);

} // namespace py::runtime_library
