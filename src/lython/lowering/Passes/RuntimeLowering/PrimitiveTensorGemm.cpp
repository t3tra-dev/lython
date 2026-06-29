#include "PrimitiveTensorGemm.h"
#include "PrimitiveTensorMicroKernel.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>

namespace py::runtime_lowering {
namespace {

constexpr int64_t kMatmulMTile = 64;
constexpr int64_t kMatmulNTile = 64;
constexpr int64_t kMatmulKTile = 32;
constexpr int64_t kMatmulPackedMTile = 128;
constexpr int64_t kMatmulPackedNTile = 64;
constexpr int64_t kMatmulPackedKTileMax = 256;
constexpr int64_t kMatmulMicroMTile = 4;
constexpr int64_t kMatmulMicroNTile = 8;
constexpr int64_t kMatmulWideMicroNTile = 16;
constexpr int64_t kMatmulPackedMicroMTile = 2;
constexpr int64_t kMatmulPackedMicroNTile = 32;
constexpr int64_t kMatmulScalarVectorKLimit = 4;
constexpr int64_t kMatmulKeepKTile = 0;
constexpr std::uint64_t kMatmulPackedMinWork = 384ull * 384ull * 128ull;
constexpr std::uint64_t kMatmulPackedMinSourceElements = 384ull * 384ull;
constexpr int64_t kPackedPanelAlignment = 64;
constexpr int64_t kPackedCopyVectorBits = 512;
constexpr llvm::StringLiteral kPackLhsCandidateAttr{
    "ly.prim_tensor.pack_lhs_candidate"};
constexpr llvm::StringLiteral kPackRhsCandidateAttr{
    "ly.prim_tensor.pack_rhs_candidate"};
constexpr llvm::StringLiteral kPackedLhsAttr{"ly.prim_tensor.packed_lhs"};
constexpr llvm::StringLiteral kPackedRhsAttr{"ly.prim_tensor.packed_rhs"};
constexpr llvm::StringLiteral kPackedPanelAttr{"ly.prim_tensor.packed_panel"};
constexpr llvm::StringLiteral kMicroMAttr{"ly.prim_tensor.micro_m"};
constexpr llvm::StringLiteral kMicroNAttr{"ly.prim_tensor.micro_n"};
constexpr llvm::StringLiteral kMicroKAttr{"ly.prim_tensor.micro_k"};

struct MatmulTileShape {
  int64_t m;
  int64_t n;
  int64_t k;
};

enum class MatmulLoweringPlan {
  Conservative,
  ScalarOrVector,
  TiledVector,
  TiledPackedVector,
};

enum class MatmulPanel {
  Lhs,
  Rhs,
};

struct MatmulPackPolicy {
  bool lhs;
  bool rhs;
};

struct GemmTargetModel {
  MatmulTileShape outerTile;
  MatmulTileShape microTile;
  MatmulPackPolicy pack;
  llvm::SmallVector<unsigned, 3> outerInterchange;
};

struct MatmulLoweringPolicy {
  MatmulLoweringPlan plan;
  GemmTargetModel target;
};

GemmTargetModel defaultGemmTargetModel() {
  return GemmTargetModel{
      MatmulTileShape{kMatmulMTile, kMatmulNTile, kMatmulKTile},
      MatmulTileShape{kMatmulMicroMTile, kMatmulMicroNTile, kMatmulKeepKTile},
      MatmulPackPolicy{false, false},
      {}};
}

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

std::optional<int64_t> primitiveElementBitWidth(mlir::Type type) {
  if (auto floatType = mlir::dyn_cast<mlir::FloatType>(type))
    return floatType.getWidth();
  if (auto intType = mlir::dyn_cast<mlir::IntegerType>(type))
    return intType.getWidth();
  return std::nullopt;
}

bool hasPrimitiveStaticShape(mlir::Value value) {
  auto shapedType = mlir::dyn_cast<mlir::ShapedType>(value.getType());
  if (!shapedType || !shapedType.hasStaticShape())
    return false;
  return isPrimitiveElementType(shapedType.getElementType());
}

bool hasPrimitiveMatmulContract(mlir::linalg::MatmulOp matmul) {
  return hasPrimitiveStaticShape(matmul.getDpsInputOperand(0)->get()) &&
         hasPrimitiveStaticShape(matmul.getDpsInputOperand(1)->get()) &&
         hasPrimitiveStaticShape(matmul.getDpsInitOperand(0)->get());
}

std::optional<MatmulTileShape>
staticMatmulShape(mlir::linalg::MatmulOp matmul) {
  if (matmul.getNumDpsInputs() != 2 || matmul.getNumDpsInits() != 1)
    return std::nullopt;

  auto lhsType = mlir::dyn_cast<mlir::ShapedType>(
      matmul.getDpsInputOperand(0)->get().getType());
  auto rhsType = mlir::dyn_cast<mlir::ShapedType>(
      matmul.getDpsInputOperand(1)->get().getType());
  auto initType = mlir::dyn_cast<mlir::ShapedType>(
      matmul.getDpsInitOperand(0)->get().getType());
  if (!lhsType || !rhsType || !initType || !lhsType.hasStaticShape() ||
      !rhsType.hasStaticShape() || !initType.hasStaticShape())
    return std::nullopt;
  if (lhsType.getRank() != 2 || rhsType.getRank() != 2 ||
      initType.getRank() != 2)
    return std::nullopt;

  return MatmulTileShape{lhsType.getDimSize(0), rhsType.getDimSize(1),
                         lhsType.getDimSize(1)};
}

std::uint64_t saturatedMatmulWork(MatmulTileShape shape) {
  constexpr std::uint64_t max = std::numeric_limits<std::uint64_t>::max();
  std::uint64_t work = 1;
  for (int64_t dim : {shape.m, shape.n, shape.k}) {
    if (dim <= 0)
      return 0;
    std::uint64_t value = static_cast<std::uint64_t>(dim);
    if (work > max / value)
      return max;
    work *= value;
  }
  return work;
}

int64_t selectPackedKTile(MatmulTileShape shape) {
  // Keep packed K panels large enough to reduce partial C traffic, but require
  // exact tiling for now: the affine/vector cleanup pipeline is deliberately
  // simpler and does not need partial-panel special cases.
  int64_t candidate =
      shape.k < kMatmulPackedKTileMax ? shape.k : kMatmulPackedKTileMax;
  while (candidate > 1 && shape.k % candidate != 0)
    --candidate;
  return candidate;
}

MatmulLoweringPlan classifyMatmulLowering(mlir::linalg::MatmulOp matmul) {
  if (!hasPrimitiveMatmulContract(matmul))
    return MatmulLoweringPlan::Conservative;

  std::optional<MatmulTileShape> shape = staticMatmulShape(matmul);
  if (!shape)
    return MatmulLoweringPlan::Conservative;

  if (shape->m <= kMatmulMicroMTile && shape->n <= kMatmulMicroNTile &&
      shape->k <= kMatmulScalarVectorKLimit)
    return MatmulLoweringPlan::ScalarOrVector;

  if (saturatedMatmulWork(*shape) >= kMatmulPackedMinWork)
    return MatmulLoweringPlan::TiledPackedVector;

  return MatmulLoweringPlan::TiledVector;
}

MatmulLoweringPolicy selectMatmulLoweringPolicy(mlir::linalg::MatmulOp matmul) {
  MatmulLoweringPlan plan = classifyMatmulLowering(matmul);
  GemmTargetModel target = defaultGemmTargetModel();
  std::optional<MatmulTileShape> shape = staticMatmulShape(matmul);

  if (plan == MatmulLoweringPlan::TiledPackedVector) {
    target.outerTile = MatmulTileShape{kMatmulPackedMTile, kMatmulPackedNTile,
                                       shape ? selectPackedKTile(*shape)
                                             : kMatmulPackedKTileMax};
    target.microTile = MatmulTileShape{
        kMatmulPackedMicroMTile, kMatmulPackedMicroNTile, kMatmulKeepKTile};
    target.pack = MatmulPackPolicy{false, true};
    // Mirror the BLAS macro-kernel shape where it already pays off: keep the
    // packed RHS panel scoped by (N, K) and reuse it for all M tiles. LHS stays
    // on source rows until an interleaved panel can lower to a single
    // contiguous load and avoid scalar copy overhead.
    target.outerInterchange = {1, 2, 0};
  }

  return MatmulLoweringPolicy{plan, std::move(target)};
}

bool usesTiledMatmulPath(MatmulLoweringPlan plan) {
  return plan == MatmulLoweringPlan::TiledVector ||
         plan == MatmulLoweringPlan::TiledPackedVector;
}

bool hasStaticShapeLargerThanTile(mlir::linalg::MatmulOp matmul,
                                  MatmulTileShape tile) {
  std::optional<MatmulTileShape> shape = staticMatmulShape(matmul);
  if (!shape)
    return false;
  return (tile.m > 0 && shape->m > tile.m) ||
         (tile.n > 0 && shape->n > tile.n) || (tile.k > 0 && shape->k > tile.k);
}

bool memrefHasRowMajorStrides(mlir::MemRefType type) {
  if (!type.hasStaticShape())
    return false;

  llvm::SmallVector<int64_t, 4> strides;
  int64_t offset = 0;
  if (mlir::failed(type.getStridesAndOffset(strides, offset)))
    return false;

  llvm::SmallVector<int64_t, 4> expectedStrides(type.getRank(), 1);
  int64_t stride = 1;
  for (int64_t dim = type.getRank() - 1; dim >= 0; --dim) {
    expectedStrides[dim] = stride;
    stride *= type.getDimSize(dim);
  }
  return llvm::equal(strides, expectedStrides);
}

bool memrefHasContiguousInnerDimension(mlir::MemRefType type) {
  if (!type.hasStaticShape() || type.getRank() == 0)
    return false;

  llvm::SmallVector<int64_t, 4> strides;
  int64_t offset = 0;
  if (mlir::failed(type.getStridesAndOffset(strides, offset)))
    return false;
  return strides.back() == 1;
}

std::optional<int64_t> selectPackedCopyVectorLanes(mlir::Type elementType,
                                                   int64_t columns) {
  std::optional<int64_t> bitWidth = primitiveElementBitWidth(elementType);
  if (!bitWidth || *bitWidth <= 0 || columns <= 0)
    return std::nullopt;

  int64_t lanes = kPackedCopyVectorBits / *bitWidth;
  if (lanes < 1)
    lanes = 1;
  if (lanes > columns)
    lanes = columns;
  while (lanes > 1 && columns % lanes != 0)
    --lanes;
  return lanes;
}

std::uint64_t staticElementCount(mlir::MemRefType type) {
  if (!type.hasStaticShape())
    return 0;

  constexpr std::uint64_t max = std::numeric_limits<std::uint64_t>::max();
  std::uint64_t count = 1;
  for (int64_t dim : type.getShape()) {
    if (dim <= 0)
      return 0;
    std::uint64_t value = static_cast<std::uint64_t>(dim);
    if (count > max / value)
      return max;
    count *= value;
  }
  return count;
}

mlir::Value sourceMemrefForPanel(mlir::Value value) {
  if (auto subview = value.getDefiningOp<mlir::memref::SubViewOp>())
    return subview.getSource();
  if (auto cast = value.getDefiningOp<mlir::memref::ReinterpretCastOp>())
    return cast.getSource();
  return value;
}

bool sourceIsLargePrimitiveTensor(mlir::Value value) {
  auto sourceType =
      mlir::dyn_cast<mlir::MemRefType>(sourceMemrefForPanel(value).getType());
  if (!sourceType || sourceType.getRank() != 2 ||
      !isPrimitiveElementType(sourceType.getElementType()))
    return false;
  return staticElementCount(sourceType) >= kMatmulPackedMinSourceElements;
}

llvm::StringLiteral candidateAttrForPanel(MatmulPanel panel) {
  return panel == MatmulPanel::Lhs ? kPackLhsCandidateAttr
                                   : kPackRhsCandidateAttr;
}

llvm::StringLiteral packedAttrForPanel(MatmulPanel panel) {
  return panel == MatmulPanel::Lhs ? kPackedLhsAttr : kPackedRhsAttr;
}

mlir::OpOperand *operandForPanel(mlir::linalg::MatmulOp matmul,
                                 MatmulPanel panel) {
  return matmul.getDpsInputOperand(panel == MatmulPanel::Lhs ? 0 : 1);
}

bool shouldPackPanel(mlir::linalg::MatmulOp matmul, MatmulPanel panel) {
  if (matmul->hasAttr(packedAttrForPanel(panel)))
    return false;
  if (!matmul->hasAttr(candidateAttrForPanel(panel)))
    return false;

  mlir::Value panelValue = operandForPanel(matmul, panel)->get();
  auto panelType = mlir::dyn_cast<mlir::MemRefType>(panelValue.getType());
  if (!panelType || panelType.getRank() != 2 || !panelType.hasStaticShape())
    return false;
  if (memrefHasRowMajorStrides(panelType))
    return false;

  return sourceIsLargePrimitiveTensor(panelValue);
}

mlir::Value packPanel(mlir::linalg::MatmulOp matmul, MatmulPanel panel,
                      mlir::IRRewriter &rewriter) {
  mlir::OpOperand *operand = operandForPanel(matmul, panel);
  mlir::Value source = operand->get();
  auto sourceType = mlir::cast<mlir::MemRefType>(source.getType());
  mlir::MemRefType packedType = mlir::MemRefType::get(
      sourceType.getShape(), sourceType.getElementType(),
      mlir::MemRefLayoutAttrInterface{}, sourceType.getMemorySpace());

  rewriter.setInsertionPoint(matmul);
  mlir::Value packed =
      rewriter
          .create<mlir::memref::AllocOp>(
              matmul.getLoc(), packedType, mlir::ValueRange{},
              rewriter.getI64IntegerAttr(kPackedPanelAlignment))
          .getResult();
  packed.getDefiningOp()->setAttr(kPackedPanelAttr, rewriter.getUnitAttr());
  rewriter.create<mlir::linalg::CopyOp>(
      matmul.getLoc(), mlir::ValueRange{source}, mlir::ValueRange{packed},
      llvm::ArrayRef<mlir::NamedAttribute>{});
  operand->set(packed);
  matmul->setAttr(packedAttrForPanel(panel), rewriter.getUnitAttr());
  matmul->removeAttr(candidateAttrForPanel(panel));
  return packed;
}

void setMicroTileAttrs(mlir::Operation *op, MatmulTileShape tile);

mlir::LogicalResult
tileMatmul(mlir::linalg::MatmulOp matmul, MatmulTileShape tile,
           mlir::IRRewriter &rewriter, MatmulPackPolicy pack = {false, false},
           llvm::ArrayRef<unsigned> interchange = {},
           std::optional<MatmulTileShape> nextMicroTile = std::nullopt) {
  mlir::linalg::LinalgTilingOptions options;
  options.setTileSizes({tile.m, tile.n, tile.k});
  if (!interchange.empty())
    options.setInterchange(interchange);
  options.setLoopType(mlir::linalg::LinalgTilingLoopType::Loops);

  rewriter.setInsertionPoint(matmul);
  mlir::FailureOr<mlir::linalg::TiledLinalgOp> tiled =
      mlir::linalg::tileLinalgOp(rewriter, matmul, options);
  if (mlir::failed(tiled))
    return mlir::failure();
  if (tiled->tensorResults.size() != matmul->getNumResults())
    return matmul.emitError()
           << "matmul tiling produced an unexpected result count";
  if (pack.lhs)
    tiled->op->setAttr(kPackLhsCandidateAttr, rewriter.getUnitAttr());
  if (pack.rhs)
    tiled->op->setAttr(kPackRhsCandidateAttr, rewriter.getUnitAttr());
  if (matmul->hasAttr(kMatmulZeroInitAttr) &&
      (!tile.k || !hasStaticShapeLargerThanTile(matmul, {0, 0, tile.k})))
    tiled->op->setAttr(kMatmulZeroInitAttr, rewriter.getUnitAttr());
  if (nextMicroTile)
    setMicroTileAttrs(tiled->op.getOperation(), *nextMicroTile);

  rewriter.replaceOp(matmul, tiled->tensorResults);
  return mlir::success();
}

void setMicroTileAttrs(mlir::Operation *op, MatmulTileShape tile) {
  mlir::Builder builder(op->getContext());
  op->setAttr(kMicroMAttr, builder.getI64IntegerAttr(tile.m));
  op->setAttr(kMicroNAttr, builder.getI64IntegerAttr(tile.n));
  op->setAttr(kMicroKAttr, builder.getI64IntegerAttr(tile.k));
}

std::optional<int64_t> getI64Attr(mlir::Operation *op,
                                  llvm::StringLiteral name) {
  if (auto attr = op->getAttrOfType<mlir::IntegerAttr>(name))
    return attr.getInt();
  return std::nullopt;
}

std::optional<MatmulTileShape> microTileAttr(mlir::Operation *op) {
  std::optional<int64_t> m = getI64Attr(op, kMicroMAttr);
  std::optional<int64_t> n = getI64Attr(op, kMicroNAttr);
  std::optional<int64_t> k = getI64Attr(op, kMicroKAttr);
  if (!m || !n || !k)
    return std::nullopt;
  return MatmulTileShape{*m, *n, *k};
}

MatmulTileShape selectMicroTile(mlir::linalg::MatmulOp matmul) {
  if (std::optional<MatmulTileShape> tile =
          microTileAttr(matmul.getOperation()))
    return *tile;

  std::optional<MatmulTileShape> shape = staticMatmulShape(matmul);
  if (shape && shape->n >= kMatmulPackedNTile && shape->m >= kMatmulMTile)
    return MatmulTileShape{kMatmulMicroMTile, kMatmulWideMicroNTile,
                           kMatmulKeepKTile};
  return defaultGemmTargetModel().microTile;
}

bool isPrimitiveZeroConstant(mlir::Value value) {
  auto constant = value.getDefiningOp<mlir::arith::ConstantOp>();
  if (!constant)
    return false;
  mlir::Attribute attr = constant.getValue();
  if (auto integer = mlir::dyn_cast<mlir::IntegerAttr>(attr))
    return integer.getValue().isZero();
  if (auto floating = mlir::dyn_cast<mlir::FloatAttr>(attr))
    return floating.getValue().isZero();
  return false;
}

bool hasSingleReductionTile(mlir::linalg::MatmulOp matmul) {
  std::optional<MatmulTileShape> shape = staticMatmulShape(matmul);
  if (!shape)
    return false;

  MatmulLoweringPolicy policy = selectMatmulLoweringPolicy(matmul);
  if (!usesTiledMatmulPath(policy.plan))
    return true;
  return policy.target.outerTile.k <= 0 ||
         shape->k <= policy.target.outerTile.k;
}

bool canAbsorbZeroFill(mlir::linalg::MatmulOp matmul,
                       mlir::linalg::FillOp fill) {
  if (!hasPrimitiveMatmulContract(matmul) || fill->getNumResults() != 1 ||
      fill.getNumDpsInputs() != 1 || fill.getNumDpsInits() != 1 ||
      !hasSingleReductionTile(matmul))
    return false;

  mlir::Value fillResult = fill->getResult(0);
  if (matmul.getDpsInitOperand(0)->get() != fillResult ||
      !fillResult.hasOneUse())
    return false;

  return isPrimitiveZeroConstant(fill.getDpsInputOperand(0)->get());
}

class MatmulZeroInitElisionPass
    : public mlir::PassWrapper<MatmulZeroInitElisionPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MatmulZeroInitElisionPass)

  llvm::StringRef getArgument() const final {
    return "lython-matmul-zero-init-elision";
  }
  llvm::StringRef getDescription() const final {
    return "fold zero-filled matmul outs into matmul lowering evidence";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<mlir::arith::ArithDialect, mlir::linalg::LinalgDialect,
                    mlir::tensor::TensorDialect>();
  }

  void runOnOperation() final {
    llvm::SmallVector<mlir::linalg::MatmulOp, 16> matmuls;
    getOperation().walk([&](mlir::linalg::MatmulOp matmul) {
      mlir::Value init = matmul.getDpsInitOperand(0)->get();
      auto fill = init.getDefiningOp<mlir::linalg::FillOp>();
      if (fill && canAbsorbZeroFill(matmul, fill))
        matmuls.push_back(matmul);
    });

    for (mlir::linalg::MatmulOp matmul : matmuls) {
      if (!matmul->getBlock())
        continue;
      auto fill = matmul.getDpsInitOperand(0)
                      ->get()
                      .getDefiningOp<mlir::linalg::FillOp>();
      if (!fill || !canAbsorbZeroFill(matmul, fill))
        continue;
      matmul->setAttr(kMatmulZeroInitAttr,
                      mlir::UnitAttr::get(matmul.getContext()));
      matmul.getDpsInitOperand(0)->set(fill.getDpsInitOperand(0)->get());
      fill.erase();
    }
  }
};

class MatmulTilingPass
    : public mlir::PassWrapper<MatmulTilingPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MatmulTilingPass)

  llvm::StringRef getArgument() const final { return "lython-matmul-tiling"; }
  llvm::StringRef getDescription() const final {
    return "tile large primitive linalg.matmul ops after bufferization";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<mlir::linalg::LinalgDialect, mlir::scf::SCFDialect,
                    mlir::memref::MemRefDialect, mlir::tensor::TensorDialect>();
  }

  void runOnOperation() final {
    struct TilingTarget {
      mlir::linalg::MatmulOp matmul;
      MatmulLoweringPolicy policy;
    };

    llvm::SmallVector<TilingTarget, 8> matmuls;
    getOperation().walk([&](mlir::linalg::MatmulOp matmul) {
      MatmulLoweringPolicy policy = selectMatmulLoweringPolicy(matmul);
      if (usesTiledMatmulPath(policy.plan) &&
          hasStaticShapeLargerThanTile(matmul, policy.target.outerTile))
        matmuls.push_back(TilingTarget{matmul, policy});
    });

    mlir::IRRewriter rewriter(&getContext());
    for (const TilingTarget &target : matmuls) {
      if (!target.matmul->getBlock())
        continue;
      if (mlir::failed(tileMatmul(target.matmul, target.policy.target.outerTile,
                                  rewriter, target.policy.target.pack,
                                  target.policy.target.outerInterchange,
                                  target.policy.target.microTile))) {
        signalPassFailure();
        return;
      }
    }
  }
};

class MatmulPackingPass
    : public mlir::PassWrapper<MatmulPackingPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MatmulPackingPass)

  llvm::StringRef getArgument() const final { return "lython-matmul-packing"; }
  llvm::StringRef getDescription() const final {
    return "pack non-contiguous primitive matmul panels";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<mlir::linalg::LinalgDialect, mlir::memref::MemRefDialect>();
  }

  void runOnOperation() final {
    llvm::SmallVector<mlir::linalg::MatmulOp, 16> matmuls;
    getOperation().walk([&](mlir::linalg::MatmulOp matmul) {
      if (shouldPackPanel(matmul, MatmulPanel::Lhs) ||
          shouldPackPanel(matmul, MatmulPanel::Rhs))
        matmuls.push_back(matmul);
    });

    mlir::IRRewriter rewriter(&getContext());
    for (mlir::linalg::MatmulOp matmul : matmuls) {
      if (!matmul->getBlock())
        continue;
      if (shouldPackPanel(matmul, MatmulPanel::Lhs))
        packPanel(matmul, MatmulPanel::Lhs, rewriter);
      if (shouldPackPanel(matmul, MatmulPanel::Rhs))
        packPanel(matmul, MatmulPanel::Rhs, rewriter);
    }
  }
};

class PackedPanelHoistingPass
    : public mlir::PassWrapper<PackedPanelHoistingPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PackedPanelHoistingPass)

  llvm::StringRef getArgument() const final {
    return "lython-packed-panel-hoisting";
  }
  llvm::StringRef getDescription() const final {
    return "hoist static packed-panel buffers out of tiled matmul loops";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<mlir::memref::MemRefDialect, mlir::scf::SCFDialect>();
  }

  void runOnOperation() final {
    llvm::SmallVector<mlir::memref::AllocOp, 16> packedPanels;
    getOperation().walk([&](mlir::memref::AllocOp alloc) {
      if (!alloc->hasAttr(kPackedPanelAttr))
        return;
      if (!alloc.getType().hasStaticShape() || !alloc.getDynamicSizes().empty())
        return;
      packedPanels.push_back(alloc);
    });

    for (mlir::memref::AllocOp alloc : packedPanels) {
      while (auto loop = alloc->getParentOfType<mlir::scf::ForOp>())
        alloc->moveBefore(loop);
    }
  }
};

bool isPackedPanel(mlir::Value value) {
  mlir::Operation *def = value.getDefiningOp();
  return def && def->hasAttr(kPackedPanelAttr);
}

bool copiesIntoPackedPanel(mlir::linalg::CopyOp copy) {
  return copy.getNumDpsInits() == 1 &&
         isPackedPanel(copy.getDpsInitOperand(0)->get());
}

bool isBlockArgumentDefinedInside(mlir::Value value, mlir::Operation *scope) {
  auto argument = mlir::dyn_cast<mlir::BlockArgument>(value);
  if (!argument)
    return false;

  mlir::Operation *owner = argument.getOwner()->getParentOp();
  return owner && scope->isProperAncestor(owner);
}

bool collectLoopInvariantDefs(
    mlir::Value value, mlir::scf::ForOp loop,
    llvm::SmallPtrSetImpl<mlir::Operation *> &visited,
    llvm::SmallVectorImpl<mlir::Operation *> &orderedDefs) {
  if (value == loop.getInductionVar() ||
      isBlockArgumentDefinedInside(value, loop.getOperation()))
    return false;

  mlir::Operation *def = value.getDefiningOp();
  if (!def || !loop->isProperAncestor(def))
    return true;
  if (!visited.insert(def).second)
    return true;
  if (def->getNumRegions() != 0 || !mlir::isMemoryEffectFree(def))
    return false;

  for (mlir::Value operand : def->getOperands()) {
    if (!collectLoopInvariantDefs(operand, loop, visited, orderedDefs))
      return false;
  }
  orderedDefs.push_back(def);
  return true;
}

bool hoistCopyAcrossInvariantLoop(mlir::linalg::CopyOp copy,
                                  mlir::scf::ForOp loop) {
  llvm::SmallPtrSet<mlir::Operation *, 8> visited;
  llvm::SmallVector<mlir::Operation *, 8> defs;
  for (mlir::Value operand : copy->getOperands()) {
    if (!collectLoopInvariantDefs(operand, loop, visited, defs))
      return false;
  }

  for (mlir::Operation *def : defs)
    def->moveBefore(loop);
  copy->moveBefore(loop);
  return true;
}

class PackedPanelCopyHoistingPass
    : public mlir::PassWrapper<PackedPanelCopyHoistingPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PackedPanelCopyHoistingPass)

  llvm::StringRef getArgument() const final {
    return "lython-packed-panel-copy-hoisting";
  }
  llvm::StringRef getDescription() const final {
    return "hoist packed-panel copy operations across loop-invariant tile "
           "loops";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<mlir::arith::ArithDialect, mlir::linalg::LinalgDialect,
                    mlir::memref::MemRefDialect, mlir::scf::SCFDialect>();
  }

  void runOnOperation() final {
    llvm::SmallVector<mlir::linalg::CopyOp, 16> copies;
    getOperation().walk([&](mlir::linalg::CopyOp copy) {
      if (copiesIntoPackedPanel(copy))
        copies.push_back(copy);
    });

    for (mlir::linalg::CopyOp copy : copies) {
      while (copy->getBlock()) {
        auto loop = copy->getParentOfType<mlir::scf::ForOp>();
        if (!loop || !hoistCopyAcrossInvariantLoop(copy, loop))
          break;
      }
    }
  }
};

bool lowerInnerContiguousPackedPanelCopy(mlir::linalg::CopyOp copy,
                                         mlir::Value source, mlir::Value target,
                                         mlir::MemRefType sourceType,
                                         mlir::MemRefType targetType,
                                         mlir::IRRewriter &rewriter) {
  if (!memrefHasContiguousInnerDimension(sourceType) ||
      !memrefHasContiguousInnerDimension(targetType))
    return false;
  mlir::Type elementType = targetType.getElementType();
  std::optional<int64_t> lanes =
      selectPackedCopyVectorLanes(elementType, targetType.getDimSize(1));
  if (!lanes || *lanes <= 1)
    return false;

  mlir::Location loc = copy.getLoc();
  rewriter.setInsertionPoint(copy);
  mlir::Value rowBegin = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
  mlir::Value rowEnd = rewriter.create<mlir::arith::ConstantIndexOp>(
      loc, targetType.getDimSize(0));
  mlir::Value rowStep = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
  mlir::Value colBegin = rowBegin;
  mlir::Value colEnd = rewriter.create<mlir::arith::ConstantIndexOp>(
      loc, targetType.getDimSize(1));
  mlir::Value colStep =
      rewriter.create<mlir::arith::ConstantIndexOp>(loc, *lanes);
  std::optional<mlir::Value> padding =
      zeroValueForElementType(rewriter, loc, elementType);
  if (!padding)
    return false;

  auto rowLoop =
      rewriter.create<mlir::scf::ForOp>(loc, rowBegin, rowEnd, rowStep);
  {
    mlir::OpBuilder::InsertionGuard rowGuard(rewriter);
    rewriter.setInsertionPointToStart(rowLoop.getBody());
    auto colLoop =
        rewriter.create<mlir::scf::ForOp>(loc, colBegin, colEnd, colStep);
    {
      mlir::OpBuilder::InsertionGuard colGuard(rewriter);
      rewriter.setInsertionPointToStart(colLoop.getBody());
      mlir::VectorType vectorType =
          mlir::VectorType::get({*lanes}, elementType);
      bool inBoundsValue = true;
      llvm::ArrayRef<bool> inBounds(inBoundsValue);
      mlir::Value vector = rewriter
                               .create<mlir::vector::TransferReadOp>(
                                   loc, vectorType, source,
                                   mlir::ValueRange{rowLoop.getInductionVar(),
                                                    colLoop.getInductionVar()},
                                   *padding, inBounds)
                               .getVector();
      rewriter.create<mlir::vector::TransferWriteOp>(
          loc, vector, target,
          mlir::ValueRange{rowLoop.getInductionVar(),
                           colLoop.getInductionVar()},
          inBounds);
    }
  }

  rewriter.eraseOp(copy);
  return true;
}

bool lowerPackedPanelCopy(mlir::linalg::CopyOp copy,
                          mlir::IRRewriter &rewriter) {
  if (!copiesIntoPackedPanel(copy) || copy.getNumDpsInputs() != 1 ||
      copy.getNumDpsInits() != 1)
    return false;

  mlir::Value source = copy.getDpsInputOperand(0)->get();
  mlir::Value target = copy.getDpsInitOperand(0)->get();
  auto sourceType = mlir::dyn_cast<mlir::MemRefType>(source.getType());
  auto targetType = mlir::dyn_cast<mlir::MemRefType>(target.getType());
  if (!sourceType || !targetType || sourceType.getRank() != 2 ||
      targetType.getRank() != 2 || !sourceType.hasStaticShape() ||
      !targetType.hasStaticShape() ||
      sourceType.getShape() != targetType.getShape() ||
      sourceType.getElementType() != targetType.getElementType())
    return false;

  return lowerInnerContiguousPackedPanelCopy(copy, source, target, sourceType,
                                             targetType, rewriter);
}

class PackedPanelCopyVectorizationPass
    : public mlir::PassWrapper<PackedPanelCopyVectorizationPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PackedPanelCopyVectorizationPass)

  llvm::StringRef getArgument() const final {
    return "lython-packed-panel-copy-vectorization";
  }
  llvm::StringRef getDescription() const final {
    return "lower packed-panel copy operations to contiguous vector transfers";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<mlir::arith::ArithDialect, mlir::linalg::LinalgDialect,
                    mlir::memref::MemRefDialect, mlir::scf::SCFDialect,
                    mlir::vector::VectorDialect>();
  }

  void runOnOperation() final {
    llvm::SmallVector<mlir::linalg::CopyOp, 16> copies;
    getOperation().walk([&](mlir::linalg::CopyOp copy) {
      if (copiesIntoPackedPanel(copy))
        copies.push_back(copy);
    });

    mlir::IRRewriter rewriter(&getContext());
    for (mlir::linalg::CopyOp copy : copies) {
      if (copy->getBlock())
        lowerPackedPanelCopy(copy, rewriter);
    }
  }
};

class MatmulMicroTilingPass
    : public mlir::PassWrapper<MatmulMicroTilingPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MatmulMicroTilingPass)

  llvm::StringRef getArgument() const final {
    return "lython-matmul-micro-tiling";
  }
  llvm::StringRef getDescription() const final {
    return "tile primitive linalg.matmul ops to vector-contract sized kernels";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<mlir::linalg::LinalgDialect, mlir::memref::MemRefDialect,
                    mlir::scf::SCFDialect>();
  }

  void runOnOperation() final {
    llvm::SmallVector<mlir::linalg::MatmulOp, 16> matmuls;
    getOperation().walk([&](mlir::linalg::MatmulOp matmul) {
      MatmulTileShape microTile = selectMicroTile(matmul);
      if (classifyMatmulLowering(matmul) != MatmulLoweringPlan::Conservative &&
          hasStaticShapeLargerThanTile(matmul, microTile))
        matmuls.push_back(matmul);
    });

    mlir::IRRewriter rewriter(&getContext());
    for (mlir::linalg::MatmulOp matmul : matmuls) {
      if (!matmul->getBlock())
        continue;
      MatmulTileShape microTile = selectMicroTile(matmul);
      if (mlir::failed(tileMatmul(matmul, microTile, rewriter))) {
        signalPassFailure();
        return;
      }
    }
  }
};

class MatmulVectorizationPass
    : public mlir::PassWrapper<MatmulVectorizationPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MatmulVectorizationPass)

  llvm::StringRef getArgument() const final {
    return "lython-matmul-vectorization";
  }
  llvm::StringRef getDescription() const final {
    return "vectorize primitive linalg.matmul micro kernels";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<mlir::arith::ArithDialect, mlir::linalg::LinalgDialect,
                    mlir::memref::MemRefDialect, mlir::vector::VectorDialect>();
  }

  void runOnOperation() final {
    llvm::SmallVector<mlir::linalg::MatmulOp, 32> matmuls;
    getOperation().walk([&](mlir::linalg::MatmulOp matmul) {
      MatmulTileShape microTile = selectMicroTile(matmul);
      if (classifyMatmulLowering(matmul) != MatmulLoweringPlan::Conservative &&
          !hasStaticShapeLargerThanTile(matmul, microTile))
        matmuls.push_back(matmul);
    });

    mlir::IRRewriter rewriter(&getContext());
    for (mlir::linalg::MatmulOp matmul : matmuls) {
      if (!matmul->getBlock())
        continue;
      rewriter.setInsertionPoint(matmul);
      if (mlir::succeeded(lowerMatmulMicroKernel(matmul, rewriter)))
        continue;
      if (mlir::failed(mlir::linalg::vectorize(rewriter, matmul))) {
        matmul.emitError() << "failed to vectorize primitive matmul micro tile";
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createMatmulZeroInitElisionPass() {
  return std::make_unique<MatmulZeroInitElisionPass>();
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createMatmulTilingPass() {
  return std::make_unique<MatmulTilingPass>();
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createMatmulPackingPass() {
  return std::make_unique<MatmulPackingPass>();
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createPackedPanelHoistingPass() {
  return std::make_unique<PackedPanelHoistingPass>();
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createPackedPanelCopyHoistingPass() {
  return std::make_unique<PackedPanelCopyHoistingPass>();
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createPackedPanelCopyVectorizationPass() {
  return std::make_unique<PackedPanelCopyVectorizationPass>();
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createMatmulMicroTilingPass() {
  return std::make_unique<MatmulMicroTilingPass>();
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createMatmulVectorizationPass() {
  return std::make_unique<MatmulVectorizationPass>();
}

} // namespace py::runtime_lowering
