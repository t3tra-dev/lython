#include "../../Primitive/TensorGemm.h"
#include "../../Primitive/TensorMicroKernel.h"
#include "../../Primitive/TensorSupport.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace py::lowering::arch::generic {
namespace {

constexpr int64_t kMaxMicroM = 8;
constexpr int64_t kMaxMicroN = 32;
constexpr int64_t kRegisterKUnroll = 4;

std::optional<mlir::Value> createPrimitiveMul(mlir::OpBuilder &builder,
                                              mlir::Location loc,
                                              mlir::Value lhs,
                                              mlir::Value rhs) {
  auto vectorType = mlir::dyn_cast<mlir::VectorType>(lhs.getType());
  mlir::Type elementType =
      vectorType ? vectorType.getElementType() : lhs.getType();
  if (mlir::isa<mlir::FloatType>(elementType))
    return mlir::arith::MulFOp::create(builder, loc, lhs, rhs,
                                       mlir::arith::FastMathFlags::contract)
        .getResult();
  if (mlir::isa<mlir::IntegerType>(elementType))
    return mlir::arith::MulIOp::create(builder, loc, lhs, rhs).getResult();
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
    return mlir::arith::AddFOp::create(builder, loc, lhs, rhs,
                                       mlir::arith::FastMathFlags::contract)
        .getResult();
  if (mlir::isa<mlir::IntegerType>(elementType))
    return mlir::arith::AddIOp::create(builder, loc, lhs, rhs).getResult();
  return std::nullopt;
}

std::optional<mlir::Value>
createPrimitiveFusedMultiplyAdd(mlir::OpBuilder &builder, mlir::Location loc,
                                mlir::Value lhs, mlir::Value rhs,
                                mlir::Value acc) {
  auto vectorType = mlir::dyn_cast<mlir::VectorType>(lhs.getType());
  mlir::Type elementType =
      vectorType ? vectorType.getElementType() : lhs.getType();
  if (mlir::isa<mlir::FloatType>(elementType))
    return mlir::vector::FMAOp::create(builder, loc, lhs, rhs, acc).getResult();

  std::optional<mlir::Value> product =
      createPrimitiveMul(builder, loc, lhs, rhs);
  if (!product)
    return std::nullopt;
  return createPrimitiveAdd(builder, loc, acc, *product);
}

mlir::Value addIndexOffset(mlir::OpBuilder &builder, mlir::Location loc,
                           mlir::Value base, int64_t offset) {
  if (offset == 0)
    return base;
  return mlir::arith::AddIOp::create(
             builder, loc, base,
             mlir::arith::ConstantIndexOp::create(builder, loc, offset))
      .getResult();
}

mlir::Value createZeroVector(mlir::OpBuilder &builder, mlir::Location loc,
                             mlir::VectorType rowVectorType,
                             mlir::Value scalarZero) {
  return mlir::vector::BroadcastOp::create(builder, loc, rowVectorType,
                                           scalarZero);
}

mlir::Value loadAccumulatorRow(mlir::OpBuilder &builder, mlir::Location loc,
                               mlir::Value out, mlir::VectorType rowVectorType,
                               mlir::Value rowIndex, mlir::Value columnIndex) {
  return mlir::vector::LoadOp::create(builder, loc, rowVectorType, out,
                                      mlir::ValueRange{rowIndex, columnIndex})
      .getResult();
}

void loadAccumulatorRows(mlir::OpBuilder &builder, mlir::Location loc,
                         mlir::Value out, mlir::VectorType rowVectorType,
                         llvm::ArrayRef<mlir::Value> rowIndices,
                         llvm::SmallVectorImpl<mlir::Value> &acc) {
  acc.clear();
  acc.reserve(rowIndices.size());
  for (mlir::Value rowIndex : rowIndices) {
    acc.push_back(loadAccumulatorRow(builder, loc, out, rowVectorType, rowIndex,
                                     rowIndices.front()));
  }
}

std::optional<mlir::Value>
createFirstReductionTilePredicate(mlir::OpBuilder &builder, mlir::Location loc,
                                  mlir::linalg::MatmulOp matmul) {
  std::optional<mlir::Value> reductionOffset = createSubviewOffsetForDimension(
      builder, loc, matmul.getDpsInputOperand(0)->get(), /*dimension=*/1);
  if (!reductionOffset)
    return std::nullopt;
  mlir::Value zero = createIndexConstant(builder, loc, 0);
  return mlir::arith::CmpIOp::create(builder, loc,
                                     mlir::arith::CmpIPredicate::eq,
                                     *reductionOffset, zero)
      .getResult();
}

void yieldZeroAccumulatorRows(mlir::OpBuilder &builder, mlir::Location loc,
                              mlir::VectorType rowVectorType,
                              mlir::Value padding,
                              llvm::ArrayRef<mlir::Value> rowIndices) {
  llvm::SmallVector<mlir::Value, 8> zeros;
  zeros.reserve(rowIndices.size());
  for ([[maybe_unused]] mlir::Value rowIndex : rowIndices)
    zeros.push_back(createZeroVector(builder, loc, rowVectorType, padding));
  mlir::scf::YieldOp::create(builder, loc, zeros);
}

void yieldLoadedAccumulatorRows(mlir::OpBuilder &builder, mlir::Location loc,
                                mlir::Value out, mlir::VectorType rowVectorType,
                                llvm::ArrayRef<mlir::Value> rowIndices) {
  llvm::SmallVector<mlir::Value, 8> loaded;
  loadAccumulatorRows(builder, loc, out, rowVectorType, rowIndices, loaded);
  mlir::scf::YieldOp::create(builder, loc, loaded);
}

mlir::LogicalResult initializeAccumulatorRows(
    mlir::linalg::MatmulOp matmul, mlir::IRRewriter &rewriter,
    mlir::Location loc, mlir::Value out, mlir::VectorType rowVectorType,
    mlir::Value padding, llvm::ArrayRef<mlir::Value> rowIndices,
    llvm::SmallVectorImpl<mlir::Value> &acc) {
  if (matmul->hasAttr(kMatmulZeroInitAttr)) {
    acc.clear();
    acc.reserve(rowIndices.size());
    for (std::size_t index = 0; index < rowIndices.size(); ++index)
      acc.push_back(createZeroVector(rewriter, loc, rowVectorType, padding));
  } else if (matmul->hasAttr(kMatmulZeroInitFirstReductionAttr)) {
    std::optional<mlir::Value> firstReductionTile =
        createFirstReductionTilePredicate(rewriter, loc, matmul);
    if (!firstReductionTile)
      return matmul.emitError()
             << "cannot resolve first reduction tile for zero-init matmul";

    llvm::SmallVector<mlir::Type, 8> resultTypes(rowIndices.size(),
                                                 rowVectorType);
    auto ifOp =
        mlir::scf::IfOp::create(rewriter, loc, resultTypes, *firstReductionTile,
                                /*withElseRegion=*/true);
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
      yieldZeroAccumulatorRows(rewriter, loc, rowVectorType, padding,
                               rowIndices);
    }
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
      yieldLoadedAccumulatorRows(rewriter, loc, out, rowVectorType, rowIndices);
    }
    acc.assign(ifOp.getResults().begin(), ifOp.getResults().end());
  } else {
    loadAccumulatorRows(rewriter, loc, out, rowVectorType, rowIndices, acc);
  }
  return mlir::success();
}

mlir::LogicalResult
accumulateKStep(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value lhs,
                mlir::Value rhs, llvm::SmallVectorImpl<mlir::Value> &acc,
                mlir::VectorType rowVectorType,
                llvm::ArrayRef<mlir::Value> rowIndices, mlir::Value kIndex) {
  mlir::Value b =
      mlir::vector::LoadOp::create(builder, loc, rowVectorType, rhs,
                                   mlir::ValueRange{kIndex, rowIndices[0]})
          .getResult();

  for (auto [row, rowIndex] :
       llvm::enumerate(rowIndices.take_front(acc.size()))) {
    mlir::Value a = mlir::memref::LoadOp::create(
                        builder, loc, lhs, mlir::ValueRange{rowIndex, kIndex})
                        .getResult();
    mlir::Value splat =
        mlir::vector::BroadcastOp::create(builder, loc, rowVectorType, a);
    std::optional<mlir::Value> sum =
        createPrimitiveFusedMultiplyAdd(builder, loc, splat, b, acc[row]);
    if (!sum)
      return mlir::failure();
    acc[row] = *sum;
  }

  return mlir::success();
}

mlir::LogicalResult accumulateStaticKRange(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value lhs,
    mlir::Value rhs, llvm::SmallVectorImpl<mlir::Value> &acc,
    mlir::VectorType rowVectorType, llvm::ArrayRef<mlir::Value> rowIndices,
    int64_t begin, int64_t end) {
  for (int64_t k = begin; k < end; ++k) {
    if (mlir::failed(accumulateKStep(builder, loc, lhs, rhs, acc, rowVectorType,
                                     rowIndices,
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
      createPrimitiveZeroValue(rewriter, loc, elementType);
  if (!padding)
    return mlir::failure();

  llvm::SmallVector<mlir::Value, 16> indexConstants;
  indexConstants.reserve(microM);
  for (int64_t index = 0; index < microM; ++index)
    indexConstants.push_back(createIndexConstant(rewriter, loc, index));

  mlir::Value lhs = matmul.getDpsInputOperand(0)->get();
  mlir::Value rhs = matmul.getDpsInputOperand(1)->get();
  mlir::Value out = matmul.getDpsInitOperand(0)->get();

  llvm::SmallVector<mlir::Value, 8> acc;
  if (mlir::failed(initializeAccumulatorRows(matmul, rewriter, loc, out,
                                             rowVectorType, *padding,
                                             indexConstants, acc)))
    return mlir::failure();

  int64_t unrolledLimit = (microK / kRegisterKUnroll) * kRegisterKUnroll;
  if (unrolledLimit > 0) {
    mlir::Value lower = createIndexConstant(rewriter, loc, 0);
    mlir::Value upper = createIndexConstant(rewriter, loc, unrolledLimit);
    mlir::Value step = createIndexConstant(rewriter, loc, kRegisterKUnroll);
    auto kLoop = mlir::scf::ForOp::create(
        rewriter, loc, lower, upper, step, acc,
        [&](mlir::OpBuilder &builder, mlir::Location nestedLoc, mlir::Value iv,
            mlir::ValueRange iterArgs) {
          llvm::SmallVector<mlir::Value, 8> loopAcc(iterArgs.begin(),
                                                    iterArgs.end());
          for (int64_t offset = 0; offset < kRegisterKUnroll; ++offset) {
            (void)accumulateKStep(
                builder, nestedLoc, lhs, rhs, loopAcc, rowVectorType,
                indexConstants, addIndexOffset(builder, nestedLoc, iv, offset));
          }
          mlir::scf::YieldOp::create(builder, nestedLoc, loopAcc);
        });

    for (int64_t row = 0; row < microM; ++row)
      acc[row] = kLoop.getResult(row);
  }

  if (mlir::failed(accumulateStaticKRange(rewriter, loc, lhs, rhs, acc,
                                          rowVectorType, indexConstants,
                                          unrolledLimit, microK)))
    return mlir::failure();

  for (int64_t row = 0; row < microM; ++row) {
    mlir::vector::StoreOp::create(
        rewriter, loc, acc[row], out,
        mlir::ValueRange{indexConstants[row], indexConstants[0]});
  }

  rewriter.eraseOp(matmul);
  return mlir::success();
}

} // namespace py::lowering::arch::generic
