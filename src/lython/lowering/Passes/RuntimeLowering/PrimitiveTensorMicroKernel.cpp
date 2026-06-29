#include "PrimitiveTensorMicroKernel.h"
#include "PrimitiveTensorGemm.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace py::runtime_lowering {
namespace {

constexpr int64_t kMaxMicroM = 8;
constexpr int64_t kMaxMicroN = 32;
constexpr int64_t kRegisterKUnroll = 4;

bool isPrimitiveElementType(mlir::Type type) {
  return mlir::isa<mlir::FloatType, mlir::IntegerType>(type);
}

std::optional<mlir::Value> zeroValueForElementType(mlir::OpBuilder &builder,
                                                   mlir::Location loc,
                                                   mlir::Type type) {
  if (auto floatType = mlir::dyn_cast<mlir::FloatType>(type)) {
    return builder.create<mlir::arith::ConstantOp>(
        loc, floatType, builder.getFloatAttr(floatType, 0.0));
  }
  if (auto intType = mlir::dyn_cast<mlir::IntegerType>(type)) {
    return builder.create<mlir::arith::ConstantOp>(
        loc, intType, builder.getIntegerAttr(intType, 0));
  }
  return std::nullopt;
}

std::optional<mlir::Value> createPrimitiveMul(mlir::OpBuilder &builder,
                                              mlir::Location loc,
                                              mlir::Value lhs,
                                              mlir::Value rhs) {
  auto vectorType = mlir::dyn_cast<mlir::VectorType>(lhs.getType());
  mlir::Type elementType =
      vectorType ? vectorType.getElementType() : lhs.getType();
  if (mlir::isa<mlir::FloatType>(elementType))
    return builder
        .create<mlir::arith::MulFOp>(loc, lhs, rhs,
                                     mlir::arith::FastMathFlags::contract)
        .getResult();
  if (mlir::isa<mlir::IntegerType>(elementType))
    return builder.create<mlir::arith::MulIOp>(loc, lhs, rhs).getResult();
  return std::nullopt;
}

std::optional<mlir::Value> createPrimitiveAdd(mlir::OpBuilder &builder,
                                              mlir::Location loc,
                                              mlir::Value lhs,
                                              mlir::Value rhs) {
  auto vectorType = mlir::dyn_cast<mlir::VectorType>(lhs.getType());
  mlir::Type elementType =
      vectorType ? vectorType.getElementType() : lhs.getType();
  if (mlir::isa<mlir::FloatType>(elementType))
    return builder
        .create<mlir::arith::AddFOp>(loc, lhs, rhs,
                                     mlir::arith::FastMathFlags::contract)
        .getResult();
  if (mlir::isa<mlir::IntegerType>(elementType))
    return builder.create<mlir::arith::AddIOp>(loc, lhs, rhs).getResult();
  return std::nullopt;
}

mlir::Value createIndexConstant(mlir::OpBuilder &builder, mlir::Location loc,
                                int64_t value) {
  return builder.create<mlir::arith::ConstantIndexOp>(loc, value).getResult();
}

mlir::Value addIndexOffset(mlir::OpBuilder &builder, mlir::Location loc,
                           mlir::Value base, int64_t offset) {
  if (offset == 0)
    return base;
  return builder
      .create<mlir::arith::AddIOp>(
          loc, base, builder.create<mlir::arith::ConstantIndexOp>(loc, offset))
      .getResult();
}

mlir::Value createZeroVector(mlir::OpBuilder &builder, mlir::Location loc,
                             mlir::VectorType rowVectorType,
                             mlir::Value scalarZero) {
  return builder.create<mlir::vector::BroadcastOp>(loc, rowVectorType,
                                                   scalarZero);
}

mlir::Value loadAccumulatorRow(mlir::OpBuilder &builder, mlir::Location loc,
                               mlir::Value out, mlir::VectorType rowVectorType,
                               mlir::Value padding,
                               llvm::ArrayRef<bool> inBounds,
                               mlir::Value rowIndex, mlir::Value columnIndex) {
  return builder
      .create<mlir::vector::TransferReadOp>(
          loc, rowVectorType, out, mlir::ValueRange{rowIndex, columnIndex},
          padding, inBounds)
      .getVector();
}

void loadAccumulatorRows(mlir::OpBuilder &builder, mlir::Location loc,
                         mlir::Value out, mlir::VectorType rowVectorType,
                         mlir::Value padding, llvm::ArrayRef<bool> inBounds,
                         llvm::ArrayRef<mlir::Value> rowIndices,
                         llvm::SmallVectorImpl<mlir::Value> &acc) {
  acc.clear();
  acc.reserve(rowIndices.size());
  for (mlir::Value rowIndex : rowIndices) {
    acc.push_back(loadAccumulatorRow(builder, loc, out, rowVectorType, padding,
                                     inBounds, rowIndex, rowIndices.front()));
  }
}

mlir::LogicalResult
initializeAccumulatorRows(mlir::linalg::MatmulOp matmul,
                          mlir::IRRewriter &rewriter, mlir::Location loc,
                          mlir::Value out, mlir::VectorType rowVectorType,
                          mlir::Value padding, llvm::ArrayRef<bool> inBounds,
                          llvm::ArrayRef<mlir::Value> rowIndices,
                          llvm::SmallVectorImpl<mlir::Value> &acc) {
  if (matmul->hasAttr(kMatmulZeroInitAttr)) {
    acc.clear();
    acc.reserve(rowIndices.size());
    for (std::size_t index = 0; index < rowIndices.size(); ++index)
      acc.push_back(createZeroVector(rewriter, loc, rowVectorType, padding));
  } else {
    loadAccumulatorRows(rewriter, loc, out, rowVectorType, padding, inBounds,
                        rowIndices, acc);
  }
  return mlir::success();
}

mlir::LogicalResult
accumulateKStep(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value lhs,
                mlir::Value rhs, llvm::SmallVectorImpl<mlir::Value> &acc,
                mlir::VectorType rowVectorType, mlir::Value padding,
                llvm::ArrayRef<bool> inBounds,
                llvm::ArrayRef<mlir::Value> rowIndices, mlir::Value kIndex) {
  mlir::Value b =
      builder
          .create<mlir::vector::TransferReadOp>(
              loc, rowVectorType, rhs, mlir::ValueRange{kIndex, rowIndices[0]},
              padding, inBounds)
          .getVector();

  for (auto [row, rowIndex] :
       llvm::enumerate(rowIndices.take_front(acc.size()))) {
    mlir::Value a = builder
                        .create<mlir::memref::LoadOp>(
                            loc, lhs, mlir::ValueRange{rowIndex, kIndex})
                        .getResult();
    mlir::Value splat =
        builder.create<mlir::vector::BroadcastOp>(loc, rowVectorType, a);
    std::optional<mlir::Value> product =
        createPrimitiveMul(builder, loc, splat, b);
    if (!product)
      return mlir::failure();
    std::optional<mlir::Value> sum =
        createPrimitiveAdd(builder, loc, acc[row], *product);
    if (!sum)
      return mlir::failure();
    acc[row] = *sum;
  }

  return mlir::success();
}

mlir::LogicalResult accumulateStaticKRange(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value lhs,
    mlir::Value rhs, llvm::SmallVectorImpl<mlir::Value> &acc,
    mlir::VectorType rowVectorType, mlir::Value padding,
    llvm::ArrayRef<bool> inBounds, llvm::ArrayRef<mlir::Value> rowIndices,
    int64_t begin, int64_t end) {
  for (int64_t k = begin; k < end; ++k) {
    if (mlir::failed(accumulateKStep(builder, loc, lhs, rhs, acc, rowVectorType,
                                     padding, inBounds, rowIndices,
                                     createIndexConstant(builder, loc, k))))
      return mlir::failure();
  }
  return mlir::success();
}

} // namespace

mlir::LogicalResult lowerMatmulMicroKernel(mlir::linalg::MatmulOp matmul,
                                           mlir::IRRewriter &rewriter) {
  if (matmul->getNumResults() != 0)
    return mlir::failure();

  auto lhsType = mlir::dyn_cast<mlir::MemRefType>(
      matmul.getDpsInputOperand(0)->get().getType());
  auto rhsType = mlir::dyn_cast<mlir::MemRefType>(
      matmul.getDpsInputOperand(1)->get().getType());
  auto outType = mlir::dyn_cast<mlir::MemRefType>(
      matmul.getDpsInitOperand(0)->get().getType());
  if (!lhsType || !rhsType || !outType || lhsType.getRank() != 2 ||
      rhsType.getRank() != 2 || outType.getRank() != 2 ||
      !lhsType.hasStaticShape() || !rhsType.hasStaticShape() ||
      !outType.hasStaticShape())
    return mlir::failure();

  int64_t microM = outType.getDimSize(0);
  int64_t microN = outType.getDimSize(1);
  int64_t microK = lhsType.getDimSize(1);
  if (microM <= 0 || microN <= 0 || microK <= 0 || microM > kMaxMicroM ||
      microN > kMaxMicroN || lhsType.getDimSize(0) != microM ||
      rhsType.getDimSize(0) != microK || rhsType.getDimSize(1) != microN)
    return mlir::failure();

  mlir::Type elementType = outType.getElementType();
  if (lhsType.getElementType() != elementType ||
      rhsType.getElementType() != elementType ||
      !isPrimitiveElementType(elementType))
    return mlir::failure();

  mlir::Location loc = matmul.getLoc();
  mlir::VectorType rowVectorType = mlir::VectorType::get({microN}, elementType);
  std::optional<mlir::Value> padding =
      zeroValueForElementType(rewriter, loc, elementType);
  if (!padding)
    return mlir::failure();

  llvm::SmallVector<mlir::Value, 16> indexConstants;
  indexConstants.reserve(microM);
  for (int64_t index = 0; index < microM; ++index)
    indexConstants.push_back(createIndexConstant(rewriter, loc, index));

  bool inBoundsValue = true;
  llvm::ArrayRef<bool> inBounds(inBoundsValue);
  mlir::Value lhs = matmul.getDpsInputOperand(0)->get();
  mlir::Value rhs = matmul.getDpsInputOperand(1)->get();
  mlir::Value out = matmul.getDpsInitOperand(0)->get();

  llvm::SmallVector<mlir::Value, 8> acc;
  if (mlir::failed(initializeAccumulatorRows(matmul, rewriter, loc, out,
                                             rowVectorType, *padding, inBounds,
                                             indexConstants, acc)))
    return mlir::failure();

  int64_t unrolledLimit = (microK / kRegisterKUnroll) * kRegisterKUnroll;
  if (unrolledLimit > 0) {
    mlir::Value lower = createIndexConstant(rewriter, loc, 0);
    mlir::Value upper = createIndexConstant(rewriter, loc, unrolledLimit);
    mlir::Value step = createIndexConstant(rewriter, loc, kRegisterKUnroll);
    auto kLoop = rewriter.create<mlir::scf::ForOp>(
        loc, lower, upper, step, acc,
        [&](mlir::OpBuilder &builder, mlir::Location nestedLoc, mlir::Value iv,
            mlir::ValueRange iterArgs) {
          llvm::SmallVector<mlir::Value, 8> loopAcc(iterArgs.begin(),
                                                    iterArgs.end());
          for (int64_t offset = 0; offset < kRegisterKUnroll; ++offset) {
            (void)accumulateKStep(
                builder, nestedLoc, lhs, rhs, loopAcc, rowVectorType, *padding,
                inBounds, indexConstants,
                addIndexOffset(builder, nestedLoc, iv, offset));
          }
          builder.create<mlir::scf::YieldOp>(nestedLoc, loopAcc);
        });

    for (int64_t row = 0; row < microM; ++row)
      acc[row] = kLoop.getResult(row);
  }

  if (mlir::failed(accumulateStaticKRange(
          rewriter, loc, lhs, rhs, acc, rowVectorType, *padding, inBounds,
          indexConstants, unrolledLimit, microK)))
    return mlir::failure();

  for (int64_t row = 0; row < microM; ++row) {
    rewriter.create<mlir::vector::TransferWriteOp>(
        loc, acc[row], out,
        mlir::ValueRange{indexConstants[row], indexConstants[0]}, inBounds);
  }

  rewriter.eraseOp(matmul);
  return mlir::success();
}

} // namespace py::runtime_lowering
