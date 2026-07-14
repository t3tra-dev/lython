#include "Runtime/Core/Lowerer.h"

#include "Runtime/ABI/BoxLayout.h"

namespace py::lowering {

// Rebuild a rank-1 memref from a payload box pointer/size word pair. The
// contract → physical shape relation is static, so the descriptor is
// assembled inline (llvm.insertvalue chain reconciled with the memref world
// by the standard unrealized-cast materialization). Borrow-only: the result
// aliases storage owned by the boxed element.
mlir::Value RuntimeBundleLowerer::memrefFromBoxWords(mlir::OpBuilder &builder,
                                                     mlir::Location loc,
                                                     mlir::Value pointerWord,
                                                     mlir::Value sizeWord,
                                                     mlir::MemRefType type) {
  mlir::MLIRContext *context = builder.getContext();
  auto ptrType = mlir::LLVM::LLVMPointerType::get(context);
  mlir::Type i64 = builder.getI64Type();
  auto arrayType = mlir::LLVM::LLVMArrayType::get(i64, 1);
  auto descriptorType = mlir::LLVM::LLVMStructType::getLiteral(
      context, {ptrType, ptrType, i64, arrayType, arrayType});
  mlir::Value pointer =
      mlir::LLVM::IntToPtrOp::create(builder, loc, ptrType, pointerWord);
  mlir::Value zero =
      mlir::arith::ConstantIntOp::create(builder, loc, 0, 64).getResult();
  mlir::Value one =
      mlir::arith::ConstantIntOp::create(builder, loc, 1, 64).getResult();
  mlir::Value size =
      type.hasStaticShape()
          ? mlir::arith::ConstantIntOp::create(builder, loc,
                                               type.getDimSize(0), 64)
                .getResult()
          : sizeWord;
  mlir::Value descriptor =
      mlir::LLVM::UndefOp::create(builder, loc, descriptorType);
  descriptor = mlir::LLVM::InsertValueOp::create(
      builder, loc, descriptor, pointer, llvm::ArrayRef<std::int64_t>{0});
  descriptor = mlir::LLVM::InsertValueOp::create(
      builder, loc, descriptor, pointer, llvm::ArrayRef<std::int64_t>{1});
  descriptor = mlir::LLVM::InsertValueOp::create(
      builder, loc, descriptor, zero, llvm::ArrayRef<std::int64_t>{2});
  descriptor = mlir::LLVM::InsertValueOp::create(
      builder, loc, descriptor, size, llvm::ArrayRef<std::int64_t>{3, 0});
  descriptor = mlir::LLVM::InsertValueOp::create(
      builder, loc, descriptor, one, llvm::ArrayRef<std::int64_t>{4, 0});
  return mlir::UnrealizedConversionCastOp::create(builder, loc,
                                                  mlir::TypeRange{type},
                                                  descriptor)
      .getResult(0);
}

} // namespace py::lowering
