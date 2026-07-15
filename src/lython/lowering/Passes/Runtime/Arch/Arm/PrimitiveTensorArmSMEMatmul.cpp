#include "PrimitiveTensorArmSME.h"

#include "../../Primitive/TensorGemm.h"
#include "../../Primitive/TensorSupport.h"

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
#include <limits>
#include <memory>
#include <optional>

namespace py::lowering::arch::arm {
namespace {

// A ZA tile is SVL x SVL bits, so the lane count per scalable vector — and with
// it the tile shape and the unit the streaming vector length is measured in —
// follows the element width.
struct SMEElementLayout {
  int64_t minLanes;
  mlir::arm_sme::TypeSize streamingVLUnit;
};

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
  SMEElementLayout layout;
};

struct SMEMatmulViews {
  mlir::Value lhs;
  mlir::Value rhs;
  mlir::Value out;
};

// SME setup and LHS panel packing dominate very small contractions. Keep those
// on the generic scalar/vector path, which already has small-matmul handling.
constexpr std::uint64_t kSMEMatmulMinWork = 1024;

std::optional<SMEElementLayout>
smeElementLayout(mlir::Type elementType, const py::TensorLoweringTarget &target) {
  if (elementType.isF32())
    return SMEElementLayout{4, mlir::arm_sme::TypeSize::Word};
  if (elementType.isF64() && target.usesArmSMEF64())
    return SMEElementLayout{2, mlir::arm_sme::TypeSize::Double};
  // Why not integers: SME has no non-widening integer outer product, so an
  // i32 x i32 -> i32 linalg.matmul has no single-instruction ZA form. Widening
  // (SMOPA i8->i32) changes the contraction's operand types, which the generic
  // vector path already covers without a shape-changing rewrite.
  return std::nullopt;
}

mlir::VectorType scalableVectorType(mlir::Type elementType,
                                    const SMEElementLayout &layout) {
  return mlir::VectorType::get({layout.minLanes}, elementType, {true});
}

mlir::VectorType scalableTileType(mlir::Type elementType,
                                  const SMEElementLayout &layout) {
  return mlir::VectorType::get({layout.minLanes, layout.minLanes}, elementType,
                               {true, true});
}

mlir::VectorType scalableMaskType(mlir::MLIRContext *context,
                                  const SMEElementLayout &layout) {
  return mlir::VectorType::get({layout.minLanes},
                               mlir::IntegerType::get(context, 1), {true});
}

mlir::VectorType scalableTileMaskType(mlir::MLIRContext *context,
                                      const SMEElementLayout &layout) {
  return mlir::VectorType::get({layout.minLanes, layout.minLanes},
                               mlir::IntegerType::get(context, 1),
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
matchStaticSMEMatmul(mlir::linalg::MatmulOp matmul,
                     const py::TensorLoweringTarget &target) {
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
  if (lhsType.getElementType() != elementType ||
      rhsType.getElementType() != elementType)
    return std::nullopt;
  std::optional<SMEElementLayout> layout =
      smeElementLayout(elementType, target);
  if (!layout)
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

  return StaticMatmulMemRefs{lhs,     rhs, out, lhsType, rhsType,
                             outType, m,   n,   k,       *layout};
}

std::uint64_t saturatedMatmulWork(const StaticMatmulMemRefs &refs) {
  constexpr std::uint64_t max = std::numeric_limits<std::uint64_t>::max();
  std::uint64_t work = 1;
  for (int64_t dim : {refs.m, refs.n, refs.k}) {
    if (dim <= 0)
      return 0;
    std::uint64_t value = static_cast<std::uint64_t>(dim);
    if (work > max / value)
      return max;
    work *= value;
  }
  return work;
}

bool isProfitableStaticSMEMatmul(const StaticMatmulMemRefs &refs) {
  return saturatedMatmulWork(refs) >= kSMEMatmulMinWork;
}

mlir::Value createRemaining(mlir::OpBuilder &builder, mlir::Location loc,
                            int64_t upper, mlir::Value iv) {
  return mlir::arith::SubIOp::create(
             builder, loc, createIndexConstant(builder, loc, upper), iv)
      .getResult();
}

mlir::Value createMask(mlir::OpBuilder &builder, mlir::Location loc,
                       mlir::VectorType maskType, mlir::Value extent) {
  return mlir::vector::CreateMaskOp::create(builder, loc, maskType,
                                            mlir::ValueRange{extent})
      .getResult();
}

mlir::Value createTileMask(mlir::OpBuilder &builder, mlir::Location loc,
                           mlir::VectorType maskType, mlir::Value rows,
                           mlir::Value cols) {
  return mlir::vector::CreateMaskOp::create(builder, loc, maskType,
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
  return mlir::memref::CastOp::create(builder, loc, dynamicType, value)
      .getResult();
}

mlir::Value createZeroVector(mlir::OpBuilder &builder, mlir::Location loc,
                             mlir::VectorType type, mlir::Value scalarZero) {
  return mlir::vector::BroadcastOp::create(builder, loc, type, scalarZero)
      .getResult();
}

std::optional<mlir::Value>
createFirstReductionTilePredicate(mlir::OpBuilder &builder, mlir::Location loc,
                                  mlir::linalg::MatmulOp matmul) {
  std::optional<mlir::Value> reductionOffset = createSubviewOffsetForDimension(
      builder, loc, matmul.getDpsInputOperand(0)->get(), /*dimension=*/1);
  if (!reductionOffset)
    return std::nullopt;
  return mlir::arith::CmpIOp::create(
             builder, loc, mlir::arith::CmpIPredicate::eq, *reductionOffset,
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
    return mlir::arm_sme::TileLoadOp::create(
               builder, loc, tileType, views.out, mlir::ValueRange{row, col},
               scalarZero, tileMask, mlir::arm_sme::TileSliceLayout::Horizontal)
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

  auto ifOp = mlir::scf::IfOp::create(
      builder, loc, tileType, *firstReductionTile, /*withElseRegion=*/true);
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
    mlir::scf::YieldOp::create(builder, loc, zeroTile);
  }
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
    mlir::scf::YieldOp::create(builder, loc, loadOut());
  }

  acc = ifOp.getResult(0);
  return mlir::success();
}

mlir::Value createPackedLhsPanel(mlir::OpBuilder &builder, mlir::Location loc,
                                 const StaticMatmulMemRefs &refs) {
  mlir::MemRefType packedType = mlir::MemRefType::get(
      {refs.k, refs.m}, refs.lhsType.getElementType(),
      mlir::MemRefLayoutAttrInterface{}, refs.lhsType.getMemorySpace());
  mlir::Value packed = mlir::memref::AllocOp::create(
                           builder, loc, packedType, mlir::ValueRange{},
                           builder.getI64IntegerAttr(64))
                           .getResult();

  mlir::Value zero = createIndexConstant(builder, loc, 0);
  mlir::Value one = createIndexConstant(builder, loc, 1);
  mlir::Value kUpper = createIndexConstant(builder, loc, refs.k);
  mlir::Value mUpper = createIndexConstant(builder, loc, refs.m);

  auto kLoop = mlir::scf::ForOp::create(builder, loc, zero, kUpper, one);
  {
    mlir::OpBuilder::InsertionGuard kGuard(builder);
    builder.setInsertionPointToStart(kLoop.getBody());
    auto rowLoop = mlir::scf::ForOp::create(builder, loc, zero, mUpper, one);
    {
      mlir::OpBuilder::InsertionGuard rowGuard(builder);
      builder.setInsertionPointToStart(rowLoop.getBody());
      mlir::Value source = mlir::memref::LoadOp::create(
                               builder, loc, refs.lhs,
                               mlir::ValueRange{rowLoop.getInductionVar(),
                                                kLoop.getInductionVar()})
                               .getResult();
      mlir::memref::StoreOp::create(
          builder, loc, source, packed,
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
  return mlir::vector::TransferReadOp::create(
             builder, loc, vectorType, views.lhs, mlir::ValueRange{k, row},
             minorIdentity, scalarZero, rowMask,
             transferInBoundsAttr(builder, 1))
      .getResult();
}

mlir::Value createRhsVector(mlir::OpBuilder &builder, mlir::Location loc,
                            const SMEMatmulViews &views,
                            mlir::VectorType vectorType, mlir::Value scalarZero,
                            mlir::Value k, mlir::Value col,
                            mlir::Value colMask) {
  mlir::AffineMap minorIdentity =
      mlir::AffineMap::getMinorIdentityMap(2, 1, builder.getContext());
  return mlir::vector::TransferReadOp::create(
             builder, loc, vectorType, views.rhs, mlir::ValueRange{k, col},
             minorIdentity, scalarZero, colMask,
             transferInBoundsAttr(builder, 1))
      .getResult();
}

mlir::LogicalResult lowerStaticMatmulToSME(mlir::linalg::MatmulOp matmul,
                                           mlir::IRRewriter &rewriter,
                                           const py::TensorLoweringTarget &target) {
  std::optional<StaticMatmulMemRefs> refs = matchStaticSMEMatmul(matmul, target);
  if (!refs || !isProfitableStaticSMEMatmul(*refs))
    return mlir::failure();

  rewriter.setInsertionPoint(matmul);
  mlir::Location loc = matmul.getLoc();
  mlir::MLIRContext *context = matmul.getContext();
  mlir::Type elementType = refs->outType.getElementType();
  mlir::VectorType vectorType = scalableVectorType(elementType, refs->layout);
  mlir::VectorType tileType = scalableTileType(elementType, refs->layout);
  mlir::VectorType maskType = scalableMaskType(context, refs->layout);
  mlir::VectorType tileMaskType = scalableTileMaskType(context, refs->layout);

  std::optional<mlir::Value> scalarZero =
      createPrimitiveZeroValue(rewriter, loc, elementType);
  if (!scalarZero)
    return mlir::failure();

  mlir::Value zero = createIndexConstant(rewriter, loc, 0);
  mlir::Value one = createIndexConstant(rewriter, loc, 1);
  mlir::Value mUpper = createIndexConstant(rewriter, loc, refs->m);
  mlir::Value nUpper = createIndexConstant(rewriter, loc, refs->n);
  mlir::Value kUpper = createIndexConstant(rewriter, loc, refs->k);
  mlir::Value vl = mlir::arm_sme::StreamingVLOp::create(
                       rewriter, loc, rewriter.getIndexType(),
                       refs->layout.streamingVLUnit)
                       .getResult();
  mlir::Value packedLhs = createPackedLhsPanel(rewriter, loc, *refs);
  auto packedLhsType = mlir::cast<mlir::MemRefType>(packedLhs.getType());
  SMEMatmulViews views{
      createDynamicRank2MemRefCast(rewriter, loc, packedLhs, packedLhsType),
      createDynamicRank2MemRefCast(rewriter, loc, refs->rhs, refs->rhsType),
      createDynamicRank2MemRefCast(rewriter, loc, refs->out, refs->outType)};

  auto rowLoop = mlir::scf::ForOp::create(rewriter, loc, zero, mUpper, vl);
  {
    mlir::OpBuilder::InsertionGuard rowGuard(rewriter);
    rewriter.setInsertionPointToStart(rowLoop.getBody());
    mlir::Value row = rowLoop.getInductionVar();
    mlir::Value rowsRemaining = createRemaining(rewriter, loc, refs->m, row);
    mlir::Value rowMask = createMask(rewriter, loc, maskType, rowsRemaining);

    auto colLoop = mlir::scf::ForOp::create(rewriter, loc, zero, nUpper, vl);
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

      auto kLoop = mlir::scf::ForOp::create(
          rewriter, loc, zero, kUpper, one, mlir::ValueRange{initialAcc},
          [&](mlir::OpBuilder &builder, mlir::Location nestedLoc, mlir::Value k,
              mlir::ValueRange iterArgs) {
            mlir::Value lhsVector =
                createLhsVector(builder, nestedLoc, views, vectorType,
                                *scalarZero, row, k, rowMask);
            mlir::Value rhsVector =
                createRhsVector(builder, nestedLoc, views, vectorType,
                                *scalarZero, k, col, colMask);
            mlir::Value nextAcc =
                mlir::vector::OuterProductOp::create(
                    builder, nestedLoc, tileType, lhsVector, rhsVector,
                    iterArgs[0], mlir::vector::CombiningKind::ADD)
                    .getResult();
            mlir::scf::YieldOp::create(builder, nestedLoc, nextAcc);
          });

      mlir::arm_sme::TileStoreOp::create(
          rewriter, loc, kLoop.getResult(0), views.out,
          mlir::ValueRange{row, col}, tileMask,
          mlir::arm_sme::TileSliceLayout::Horizontal);
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

  explicit MatmulSMELoweringPass(py::TensorLoweringTarget target = {})
      : target(target) {}

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
      std::optional<StaticMatmulMemRefs> refs =
          matchStaticSMEMatmul(matmul, target);
      if (refs && isProfitableStaticSMEMatmul(*refs))
        matmuls.push_back(matmul);
    });

    mlir::IRRewriter rewriter(&getContext());
    for (mlir::linalg::MatmulOp matmul : matmuls) {
      if (!matmul->getBlock())
        continue;
      if (mlir::failed(lowerStaticMatmulToSME(matmul, rewriter, target))) {
        signalPassFailure();
        return;
      }
    }
  }

private:
  py::TensorLoweringTarget target;
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createMatmulSMELoweringPass(py::TensorLoweringTarget target) {
  return std::make_unique<MatmulSMELoweringPass>(target);
}

} // namespace py::lowering::arch::arm
