#include "PrimitiveTensorArmSME.h"

#include "../../PrimitiveTensorGemm.h"
#include "../../PrimitiveTensorSupport.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ArmSME/IR/ArmSME.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <memory>
#include <optional>

namespace py::runtime_lowering::arch::arm {
namespace {

struct StaticMatmulMemRefs {
  mlir::Value lhs;
  mlir::Value rhs;
  mlir::Value out;
  mlir::MemRefType lhsType;
  mlir::MemRefType rhsType;
  mlir::MemRefType outType;
  int64_t m;
  int64_t n;
  int64_t k;
};

struct SMEMatmulViews {
  mlir::Value lhs;
  mlir::Value rhs;
  mlir::Value out;
};

mlir::Value createIndexConstant(mlir::OpBuilder &builder, mlir::Location loc,
                                int64_t value) {
  return builder.create<mlir::arith::ConstantIndexOp>(loc, value).getResult();
}

mlir::VectorType scalableVectorType(mlir::Type elementType) {
  return mlir::VectorType::get({4}, elementType, {true});
}

mlir::VectorType scalableTileType(mlir::Type elementType) {
  return mlir::VectorType::get({4, 4}, elementType, {true, true});
}

mlir::VectorType scalableTileMaskType(mlir::MLIRContext *context) {
  return mlir::VectorType::get({4, 4}, mlir::IntegerType::get(context, 1),
                               {true, true});
}

std::optional<int64_t> staticStride(mlir::MemRefType type, unsigned dimension) {
  llvm::SmallVector<int64_t, 4> strides;
  int64_t offset = 0;
  if (mlir::failed(type.getStridesAndOffset(strides, offset)) ||
      dimension >= strides.size() ||
      mlir::ShapedType::isDynamic(strides[dimension]))
    return std::nullopt;
  return strides[dimension];
}

std::optional<StaticMatmulMemRefs>
matchStaticF32Matmul(mlir::linalg::MatmulOp matmul) {
  if (matmul->getNumResults() != 0 || matmul.getNumDpsInputs() != 2 ||
      matmul.getNumDpsInits() != 1)
    return std::nullopt;

  mlir::Value lhs = matmul.getDpsInputOperand(0)->get();
  mlir::Value rhs = matmul.getDpsInputOperand(1)->get();
  mlir::Value out = matmul.getDpsInitOperand(0)->get();
  auto lhsType = mlir::dyn_cast<mlir::MemRefType>(lhs.getType());
  auto rhsType = mlir::dyn_cast<mlir::MemRefType>(rhs.getType());
  auto outType = mlir::dyn_cast<mlir::MemRefType>(out.getType());
  if (!lhsType || !rhsType || !outType || lhsType.getRank() != 2 ||
      rhsType.getRank() != 2 || outType.getRank() != 2 ||
      !lhsType.hasStaticShape() || !rhsType.hasStaticShape() ||
      !outType.hasStaticShape())
    return std::nullopt;

  mlir::Type elementType = outType.getElementType();
  if (!elementType.isF32() || lhsType.getElementType() != elementType ||
      rhsType.getElementType() != elementType)
    return std::nullopt;

  int64_t m = outType.getDimSize(0);
  int64_t n = outType.getDimSize(1);
  int64_t k = lhsType.getDimSize(1);
  if (m <= 0 || n <= 0 || k <= 0 || lhsType.getDimSize(0) != m ||
      rhsType.getDimSize(0) != k || rhsType.getDimSize(1) != n)
    return std::nullopt;

  std::optional<int64_t> rhsInnerStride = staticStride(rhsType, 1);
  std::optional<int64_t> outInnerStride = staticStride(outType, 1);
  if (!rhsInnerStride || !outInnerStride || *rhsInnerStride != 1 ||
      *outInnerStride != 1)
    return std::nullopt;

  return StaticMatmulMemRefs{lhs, rhs, out, lhsType, rhsType, outType, m, n, k};
}

mlir::Value createRemaining(mlir::OpBuilder &builder, mlir::Location loc,
                            int64_t upper, mlir::Value iv) {
  return builder
      .create<mlir::arith::SubIOp>(loc,
                                   createIndexConstant(builder, loc, upper), iv)
      .getResult();
}

mlir::Value createMask(mlir::OpBuilder &builder, mlir::Location loc,
                       mlir::VectorType maskType, mlir::Value extent) {
  return builder
      .create<mlir::vector::CreateMaskOp>(loc, maskType,
                                          mlir::ValueRange{extent})
      .getResult();
}

mlir::Value createTileMask(mlir::OpBuilder &builder, mlir::Location loc,
                           mlir::VectorType maskType, mlir::Value rows,
                           mlir::Value cols) {
  return builder
      .create<mlir::vector::CreateMaskOp>(loc, maskType,
                                          mlir::ValueRange{rows, cols})
      .getResult();
}

mlir::ArrayAttr transferInBoundsAttr(mlir::OpBuilder &builder,
                                     unsigned vectorRank) {
  llvm::SmallVector<mlir::Attribute, 4> attrs;
  attrs.reserve(vectorRank);
  for (unsigned i = 0; i < vectorRank; ++i)
    attrs.push_back(builder.getBoolAttr(false));
  return builder.getArrayAttr(attrs);
}

mlir::Value createDynamicRank2MemRefCast(mlir::OpBuilder &builder,
                                         mlir::Location loc, mlir::Value value,
                                         mlir::MemRefType type) {
  mlir::MemRefType dynamicType = mlir::MemRefType::get(
      {mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic},
      type.getElementType(), type.getLayout(), type.getMemorySpace());
  if (dynamicType == type)
    return value;
  return builder.create<mlir::memref::CastOp>(loc, dynamicType, value)
      .getResult();
}

mlir::Value createZeroVector(mlir::OpBuilder &builder, mlir::Location loc,
                             mlir::VectorType type, mlir::Value scalarZero) {
  return builder.create<mlir::vector::BroadcastOp>(loc, type, scalarZero)
      .getResult();
}

std::optional<mlir::Value>
createFirstReductionTilePredicate(mlir::OpBuilder &builder, mlir::Location loc,
                                  mlir::linalg::MatmulOp matmul) {
  std::optional<mlir::Value> reductionOffset = createSubviewOffsetForDimension(
      builder, loc, matmul.getDpsInputOperand(0)->get(), /*dimension=*/1);
  if (!reductionOffset)
    return std::nullopt;
  return builder
      .create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq,
                                   *reductionOffset,
                                   createIndexConstant(builder, loc, 0))
      .getResult();
}

mlir::LogicalResult
createAccumulatorInit(mlir::linalg::MatmulOp matmul, mlir::OpBuilder &builder,
                      mlir::Location loc, const StaticMatmulMemRefs &refs,
                      const SMEMatmulViews &views, mlir::VectorType tileType,
                      mlir::Value scalarZero, mlir::Value row, mlir::Value col,
                      mlir::Value tileMask, mlir::Value &acc) {
  mlir::Value zeroTile = createZeroVector(builder, loc, tileType, scalarZero);
  if (matmul->hasAttr(kMatmulZeroInitAttr)) {
    acc = zeroTile;
    return mlir::success();
  }

  auto loadOut = [&]() {
    return builder
        .create<mlir::arm_sme::TileLoadOp>(
            loc, tileType, views.out, mlir::ValueRange{row, col}, scalarZero,
            tileMask, mlir::arm_sme::TileSliceLayout::Horizontal)
        .getResult();
  };

  if (!matmul->hasAttr(kMatmulZeroInitFirstReductionAttr)) {
    acc = loadOut();
    return mlir::success();
  }

  std::optional<mlir::Value> firstReductionTile =
      createFirstReductionTilePredicate(builder, loc, matmul);
  if (!firstReductionTile)
    return mlir::failure();

  auto ifOp = builder.create<mlir::scf::IfOp>(
      loc, tileType, *firstReductionTile, /*withElseRegion=*/true);
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
    builder.create<mlir::scf::YieldOp>(loc, zeroTile);
  }
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
    builder.create<mlir::scf::YieldOp>(loc, loadOut());
  }

  acc = ifOp.getResult(0);
  return mlir::success();
}

mlir::Value createPackedLhsPanel(mlir::OpBuilder &builder, mlir::Location loc,
                                 const StaticMatmulMemRefs &refs) {
  mlir::MemRefType packedType = mlir::MemRefType::get(
      {refs.k, refs.m}, refs.lhsType.getElementType(),
      mlir::MemRefLayoutAttrInterface{}, refs.lhsType.getMemorySpace());
  mlir::Value packed =
      builder
          .create<mlir::memref::AllocOp>(loc, packedType, mlir::ValueRange{},
                                         builder.getI64IntegerAttr(64))
          .getResult();

  mlir::Value zero = createIndexConstant(builder, loc, 0);
  mlir::Value one = createIndexConstant(builder, loc, 1);
  mlir::Value kUpper = createIndexConstant(builder, loc, refs.k);
  mlir::Value mUpper = createIndexConstant(builder, loc, refs.m);

  auto kLoop = builder.create<mlir::scf::ForOp>(loc, zero, kUpper, one);
  {
    mlir::OpBuilder::InsertionGuard kGuard(builder);
    builder.setInsertionPointToStart(kLoop.getBody());
    auto rowLoop = builder.create<mlir::scf::ForOp>(loc, zero, mUpper, one);
    {
      mlir::OpBuilder::InsertionGuard rowGuard(builder);
      builder.setInsertionPointToStart(rowLoop.getBody());
      mlir::Value source = builder
                               .create<mlir::memref::LoadOp>(
                                   loc, refs.lhs,
                                   mlir::ValueRange{rowLoop.getInductionVar(),
                                                    kLoop.getInductionVar()})
                               .getResult();
      builder.create<mlir::memref::StoreOp>(
          loc, source, packed,
          mlir::ValueRange{kLoop.getInductionVar(), rowLoop.getInductionVar()});
    }
  }

  return packed;
}

mlir::Value createLhsVector(mlir::OpBuilder &builder, mlir::Location loc,
                            const SMEMatmulViews &views,
                            mlir::VectorType vectorType, mlir::Value scalarZero,
                            mlir::Value row, mlir::Value k,
                            mlir::Value rowMask) {
  mlir::AffineMap minorIdentity =
      mlir::AffineMap::getMinorIdentityMap(2, 1, builder.getContext());
  return builder
      .create<mlir::vector::TransferReadOp>(
          loc, vectorType, views.lhs, mlir::ValueRange{k, row}, minorIdentity,
          scalarZero, rowMask, transferInBoundsAttr(builder, 1))
      .getResult();
}

mlir::Value createRhsVector(mlir::OpBuilder &builder, mlir::Location loc,
                            const SMEMatmulViews &views,
                            mlir::VectorType vectorType, mlir::Value scalarZero,
                            mlir::Value k, mlir::Value col,
                            mlir::Value colMask) {
  mlir::AffineMap minorIdentity =
      mlir::AffineMap::getMinorIdentityMap(2, 1, builder.getContext());
  return builder
      .create<mlir::vector::TransferReadOp>(
          loc, vectorType, views.rhs, mlir::ValueRange{k, col}, minorIdentity,
          scalarZero, colMask, transferInBoundsAttr(builder, 1))
      .getResult();
}

mlir::LogicalResult lowerStaticF32MatmulToSME(mlir::linalg::MatmulOp matmul,
                                              mlir::IRRewriter &rewriter) {
  std::optional<StaticMatmulMemRefs> refs = matchStaticF32Matmul(matmul);
  if (!refs)
    return mlir::failure();

  rewriter.setInsertionPoint(matmul);
  mlir::Location loc = matmul.getLoc();
  mlir::MLIRContext *context = matmul.getContext();
  mlir::Type elementType = refs->outType.getElementType();
  mlir::VectorType vectorType = scalableVectorType(elementType);
  mlir::VectorType tileType = scalableTileType(elementType);
  mlir::VectorType maskType =
      mlir::VectorType::get({4}, rewriter.getI1Type(), {true});
  mlir::VectorType tileMaskType = scalableTileMaskType(context);

  std::optional<mlir::Value> scalarZero =
      createPrimitiveZeroValue(rewriter, loc, elementType);
  if (!scalarZero)
    return mlir::failure();

  mlir::Value zero = createIndexConstant(rewriter, loc, 0);
  mlir::Value one = createIndexConstant(rewriter, loc, 1);
  mlir::Value mUpper = createIndexConstant(rewriter, loc, refs->m);
  mlir::Value nUpper = createIndexConstant(rewriter, loc, refs->n);
  mlir::Value kUpper = createIndexConstant(rewriter, loc, refs->k);
  mlir::Value vl =
      rewriter
          .create<mlir::arm_sme::StreamingVLOp>(loc, rewriter.getIndexType(),
                                                mlir::arm_sme::TypeSize::Word)
          .getResult();
  mlir::Value packedLhs = createPackedLhsPanel(rewriter, loc, *refs);
  auto packedLhsType = mlir::cast<mlir::MemRefType>(packedLhs.getType());
  SMEMatmulViews views{
      createDynamicRank2MemRefCast(rewriter, loc, packedLhs, packedLhsType),
      createDynamicRank2MemRefCast(rewriter, loc, refs->rhs, refs->rhsType),
      createDynamicRank2MemRefCast(rewriter, loc, refs->out, refs->outType)};

  auto rowLoop = rewriter.create<mlir::scf::ForOp>(loc, zero, mUpper, vl);
  {
    mlir::OpBuilder::InsertionGuard rowGuard(rewriter);
    rewriter.setInsertionPointToStart(rowLoop.getBody());
    mlir::Value row = rowLoop.getInductionVar();
    mlir::Value rowsRemaining = createRemaining(rewriter, loc, refs->m, row);
    mlir::Value rowMask = createMask(rewriter, loc, maskType, rowsRemaining);

    auto colLoop = rewriter.create<mlir::scf::ForOp>(loc, zero, nUpper, vl);
    {
      mlir::OpBuilder::InsertionGuard colGuard(rewriter);
      rewriter.setInsertionPointToStart(colLoop.getBody());
      mlir::Value col = colLoop.getInductionVar();
      mlir::Value colsRemaining = createRemaining(rewriter, loc, refs->n, col);
      mlir::Value colMask = createMask(rewriter, loc, maskType, colsRemaining);
      mlir::Value tileMask = createTileMask(rewriter, loc, tileMaskType,
                                            rowsRemaining, colsRemaining);

      mlir::Value initialAcc;
      if (mlir::failed(createAccumulatorInit(matmul, rewriter, loc, *refs,
                                             views, tileType, *scalarZero, row,
                                             col, tileMask, initialAcc)))
        return mlir::failure();

      auto kLoop = rewriter.create<mlir::scf::ForOp>(
          loc, zero, kUpper, one, mlir::ValueRange{initialAcc},
          [&](mlir::OpBuilder &builder, mlir::Location nestedLoc, mlir::Value k,
              mlir::ValueRange iterArgs) {
            mlir::Value lhsVector =
                createLhsVector(builder, nestedLoc, views, vectorType,
                                *scalarZero, row, k, rowMask);
            mlir::Value rhsVector =
                createRhsVector(builder, nestedLoc, views, vectorType,
                                *scalarZero, k, col, colMask);
            mlir::Value nextAcc =
                builder
                    .create<mlir::vector::OuterProductOp>(
                        nestedLoc, tileType, lhsVector, rhsVector, iterArgs[0],
                        mlir::vector::CombiningKind::ADD)
                    .getResult();
            builder.create<mlir::scf::YieldOp>(nestedLoc, nextAcc);
          });

      rewriter.create<mlir::arm_sme::TileStoreOp>(
          loc, kLoop.getResult(0), views.out, mlir::ValueRange{row, col},
          tileMask, mlir::arm_sme::TileSliceLayout::Horizontal);
    }
  }

  rewriter.eraseOp(matmul);
  return mlir::success();
}

class MatmulSMELoweringPass
    : public mlir::PassWrapper<MatmulSMELoweringPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MatmulSMELoweringPass)

  llvm::StringRef getArgument() const final {
    return "lython-arm-sme-matmul-lowering";
  }

  llvm::StringRef getDescription() const final {
    return "lower supported primitive linalg.matmul ops to Arm SME outer "
           "products before generic micro-tiling";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<mlir::arith::ArithDialect, mlir::arm_sme::ArmSMEDialect,
                    mlir::linalg::LinalgDialect, mlir::memref::MemRefDialect,
                    mlir::scf::SCFDialect, mlir::vector::VectorDialect>();
  }

  void runOnOperation() final {
    llvm::SmallVector<mlir::linalg::MatmulOp, 16> matmuls;
    getOperation().walk([&](mlir::linalg::MatmulOp matmul) {
      if (matchStaticF32Matmul(matmul))
        matmuls.push_back(matmul);
    });

    mlir::IRRewriter rewriter(&getContext());
    for (mlir::linalg::MatmulOp matmul : matmuls) {
      if (!matmul->getBlock())
        continue;
      if (mlir::failed(lowerStaticF32MatmulToSME(matmul, rewriter))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createMatmulSMELoweringPass() {
  return std::make_unique<MatmulSMELoweringPass>();
}

} // namespace py::runtime_lowering::arch::arm
