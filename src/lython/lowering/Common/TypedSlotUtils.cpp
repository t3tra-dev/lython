#include "Common/TypedSlotUtils.h"

#include "Common/LoweringUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

using namespace mlir;

namespace py {

FailureOr<Value> packTypedSlot(Location loc, Value value, Type logicalType,
                               ModuleOp module,
                               ConversionPatternRewriter &rewriter,
                               const PyLLVMTypeConverter &typeConverter) {
  RuntimeAPI runtime(module, rewriter, typeConverter);
  auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

  if (value.getType() == rewriter.getI64Type())
    return value;
  if (value.getType() == ptrType)
    return rewriter.create<LLVM::PtrToIntOp>(loc, rewriter.getI64Type(), value)
        .getResult();

  if (isa<IntType>(logicalType))
    return runtime
        .call(loc, RuntimeSymbols::kLongAsI64, rewriter.getI64Type(),
              ValueRange{value})
        .getResult();

  if (isa<BoolType>(logicalType)) {
    if (value.getType() == rewriter.getI1Type())
      return rewriter.create<arith::ExtUIOp>(loc, rewriter.getI64Type(), value)
          .getResult();
    Value raw = runtime
                    .call(loc, RuntimeSymbols::kBoolAsBool,
                          rewriter.getI1Type(), ValueRange{value})
                    .getResult();
    return rewriter.create<arith::ExtUIOp>(loc, rewriter.getI64Type(), raw)
        .getResult();
  }

  if (isa<FloatType>(logicalType)) {
    Value asDouble = value;
    if (value.getType() != rewriter.getF64Type())
      asDouble = runtime
                     .call(loc, RuntimeSymbols::kFloatAsDouble,
                           rewriter.getF64Type(), ValueRange{value})
                     .getResult();
    return rewriter
        .create<arith::BitcastOp>(loc, rewriter.getI64Type(), asDouble)
        .getResult();
  }

  if (isa<StrType, ObjectType, ClassType>(logicalType)) {
    Value asPtr = value;
    if (asPtr.getType() != ptrType)
      asPtr = rewriter.create<CastIdentityOp>(loc, ptrType, asPtr);
    return rewriter.create<LLVM::PtrToIntOp>(loc, rewriter.getI64Type(), asPtr)
        .getResult();
  }

  return failure();
}

FailureOr<Value> boxTypedSlot(Location loc, Value value, Type logicalType,
                              ModuleOp module,
                              ConversionPatternRewriter &rewriter,
                              const PyLLVMTypeConverter &typeConverter) {
  RuntimeAPI runtime(module, rewriter, typeConverter);

  if (isa<IntType>(logicalType))
    return runtime
        .call(loc, RuntimeSymbols::kLongFromI64,
              typeConverter.getPyObjectPtrType(), ValueRange{value})
        .getResult();

  if (isa<BoolType>(logicalType)) {
    Value zero = createI64Constant(loc, rewriter, 0);
    Value asBool = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
                                                  value, zero);
    return runtime
        .call(loc, RuntimeSymbols::kBoolFromBool,
              typeConverter.getPyObjectPtrType(), ValueRange{asBool})
        .getResult();
  }

  if (isa<FloatType>(logicalType))
    return runtime
        .call(loc, RuntimeSymbols::kFloatFromDouble,
              typeConverter.getPyObjectPtrType(),
              ValueRange{rewriter.create<arith::BitcastOp>(
                  loc, rewriter.getF64Type(), value)})
        .getResult();

  if (isa<StrType, ObjectType, ClassType>(logicalType)) {
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    return rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, value).getResult();
  }

  return failure();
}

void emitClassSlotRefcount(Location loc, Value slot, ClassType type,
                           ModuleOp module, ConversionPatternRewriter &rewriter,
                           StringRef suffix) {
  auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
  Value asPtr = rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, slot);
  auto helper = getOrInsertLLVMFunc(
      loc, module, rewriter, getClassHelperName(type, suffix),
      LLVM::LLVMVoidType::get(rewriter.getContext()), {ptrType});
  auto helperRef = SymbolRefAttr::get(module.getContext(), helper.getName());
  rewriter.create<LLVM::CallOp>(loc, TypeRange{}, helperRef, ValueRange{asPtr});
}

} // namespace py
