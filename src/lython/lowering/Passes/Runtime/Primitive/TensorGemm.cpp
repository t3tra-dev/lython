#include "TensorGemm.h"
#include "TensorMicroKernel.h"
#include "TensorPacking.h"
#include "TensorSupport.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
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

namespace py::lowering {
namespace {

constexpr int64_t kMatmulConservativeMC = 64;
constexpr int64_t kMatmulConservativeNC = 64;
constexpr int64_t kMatmulConservativeKC = 32;
constexpr int64_t kMatmulPackedMC = 128;
constexpr int64_t kMatmulPackedNC = 64;
// Keep the packed panels below the point where a 512-wide reduction panel hurts
// locality on Apple Silicon, while still allowing 768-wide reductions to use
// two 384 panels instead of three 256 panels. KC=512 re-measured 2026-07 on
// apple-m1 NEON f32 512^3/1024^3: 15-18% slower than 384.
constexpr int64_t kMatmulPackedKCMax = 384;
constexpr int64_t kMatmulConservativeMR = 4;
constexpr int64_t kMatmulConservativeNR = 8;
constexpr int64_t kMatmulWideNR = 16;
// Why not MR=4 x NR=16 (same accumulator budget, half the B-vector loads):
// measured 2026-07 on apple-m1 NEON f32, 512^3 was 2% slower and 1024^3 flat
// -- the wide-NR shape keeps the fewer, longer B streams the prefetcher
// prefers.
constexpr int64_t kMatmulPackedMR = 2;
constexpr int64_t kMatmulPackedNR = 32;
constexpr int64_t kMatmulScalarVectorKLimit = 4;
constexpr int64_t kMatmulKeepKC = 0;
// Packing starts paying off for the current single-threaded RHS-prepack path at
// 256x256 source panels. Smaller panels can still win for selected shapes, but
// the generic threshold becomes noisy below this point.
constexpr std::uint64_t kMatmulPackedMinWork = 256ull * 256ull * 128ull;
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

struct GemmSchedule {
  MatmulTileShape macroTile;
  MatmulTileShape registerTile;
  bool packRhs;
  llvm::SmallVector<unsigned, 3> outerInterchange;
};

struct MatmulLoweringPolicy {
  MatmulLoweringPlan plan;
  GemmSchedule schedule;
};

GemmSchedule defaultGemmSchedule() {
  return GemmSchedule{MatmulTileShape{kMatmulConservativeMC,
                                      kMatmulConservativeNC,
                                      kMatmulConservativeKC},
                      MatmulTileShape{kMatmulConservativeMR,
                                      kMatmulConservativeNR, kMatmulKeepKC},
                      /*packRhs=*/false,
                      {}};
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
      shape.k < kMatmulPackedKCMax ? shape.k : kMatmulPackedKCMax;
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

  if (shape->m <= kMatmulConservativeMR && shape->n <= kMatmulConservativeNR &&
      shape->k <= kMatmulScalarVectorKLimit)
    return MatmulLoweringPlan::ScalarOrVector;

  if (saturatedMatmulWork(*shape) >= kMatmulPackedMinWork)
    return MatmulLoweringPlan::TiledPackedVector;

  return MatmulLoweringPlan::TiledVector;
}

MatmulLoweringPolicy selectMatmulLoweringPolicy(mlir::linalg::MatmulOp matmul) {
  MatmulLoweringPlan plan = classifyMatmulLowering(matmul);
  GemmSchedule schedule = defaultGemmSchedule();
  std::optional<MatmulTileShape> shape = staticMatmulShape(matmul);

  if (plan == MatmulLoweringPlan::TiledPackedVector) {
    schedule.macroTile =
        MatmulTileShape{kMatmulPackedMC, kMatmulPackedNC,
                        shape ? selectPackedKTile(*shape) : kMatmulPackedKCMax};
    schedule.registerTile =
        MatmulTileShape{kMatmulPackedMR, kMatmulPackedNR, kMatmulKeepKC};
    schedule.packRhs = true;
    // BLAS-style loop ordering: keep the RHS B panel scoped by (N, K) and reuse
    // it for all M tiles. A-side packing is intentionally absent from this
    // schedule: the previous strided-MxK implementation was consistently slower
    // in measured 512/768 f32 cases. Reintroduce it only together with a real
    // Ap[k][ir] micro-kernel contract.
    schedule.outerInterchange = {1, 2, 0};
  }

  return MatmulLoweringPolicy{plan, std::move(schedule)};
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

mlir::Value packRhsPanel(mlir::linalg::MatmulOp matmul,
                         mlir::IRRewriter &rewriter) {
  mlir::OpOperand *operand = matmul.getDpsInputOperand(1);
  mlir::Value source = operand->get();
  auto sourceType = mlir::cast<mlir::MemRefType>(source.getType());
  mlir::MemRefType packedType = mlir::MemRefType::get(
      sourceType.getShape(), sourceType.getElementType(),
      mlir::MemRefLayoutAttrInterface{}, sourceType.getMemorySpace());

  rewriter.setInsertionPoint(matmul);
  mlir::Value packed =
      mlir::memref::AllocOp::create(
          rewriter, matmul.getLoc(), packedType, mlir::ValueRange{},
          rewriter.getI64IntegerAttr(kPackedPanelAlignment))
          .getResult();
  packed.getDefiningOp()->setAttr(kPackedPanelAttr, rewriter.getUnitAttr());
  mlir::linalg::CopyOp::create(
      rewriter, matmul.getLoc(), mlir::ValueRange{source},
      mlir::ValueRange{packed}, llvm::ArrayRef<mlir::NamedAttribute>{});
  operand->set(packed);
  matmul->setAttr(kPackedRhsAttr, rewriter.getUnitAttr());
  matmul->removeAttr(kPackRhsCandidateAttr);
  return packed;
}

void setMicroTileAttrs(mlir::Operation *op, MatmulTileShape tile);

bool loopHasPartialTile(mlir::scf::ForOp forOp) {
  std::optional<int64_t> lower =
      mlir::getConstantIntValue(forOp.getLowerBound());
  std::optional<int64_t> upper =
      mlir::getConstantIntValue(forOp.getUpperBound());
  std::optional<int64_t> step = mlir::getConstantIntValue(forOp.getStep());
  if (!lower || !upper || !step || *step <= 0 || *upper <= *lower)
    return false;
  return (*upper - *lower) % *step != 0;
}

// A tile that does not divide its extent leaves the trailing iteration with an
// affine.min extent, which makes the tiled matmul read as dynamically shaped.
// staticMatmulShape then declines it, classifyMatmulLowering calls it
// Conservative, so neither the micro-kernel nor the vectorizer claims it, and
// the affine loop conversion finally rejects the affine.min as a bound
// operand. Peeling splits the extent so both the main and the trailing nest
// carry static tiles. Innermost first: peeling a loop clones the loops nested
// inside it, and those clones are not in `loops`.
void peelPartialTileLoops(llvm::ArrayRef<mlir::Operation *> loops,
                          mlir::IRRewriter &rewriter) {
  for (mlir::Operation *loop : llvm::reverse(loops)) {
    auto forOp = mlir::dyn_cast_or_null<mlir::scf::ForOp>(loop);
    if (forOp && loopHasPartialTile(forOp))
      mlir::linalg::peelLoop(rewriter, forOp);
  }
}

mlir::LogicalResult
tileMatmul(mlir::linalg::MatmulOp matmul, MatmulTileShape tile,
           mlir::IRRewriter &rewriter, bool packRhs = false,
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
  tiled->op->removeAttr(kMatmulZeroInitAttr);
  tiled->op->removeAttr(kMatmulZeroInitFirstReductionAttr);
  if (packRhs)
    tiled->op->setAttr(kPackRhsCandidateAttr, rewriter.getUnitAttr());
  if (matmul->hasAttr(kMatmulZeroInitFirstReductionAttr)) {
    tiled->op->setAttr(kMatmulZeroInitFirstReductionAttr,
                       rewriter.getUnitAttr());
  } else if (matmul->hasAttr(kMatmulZeroInitAttr)) {
    if (!tile.k || !hasStaticShapeLargerThanTile(matmul, {0, 0, tile.k})) {
      tiled->op->setAttr(kMatmulZeroInitAttr, rewriter.getUnitAttr());
    } else {
      tiled->op->setAttr(kMatmulZeroInitFirstReductionAttr,
                         rewriter.getUnitAttr());
    }
  }
  if (nextMicroTile)
    setMicroTileAttrs(tiled->op.getOperation(), *nextMicroTile);

  rewriter.replaceOp(matmul, tiled->tensorResults);
  peelPartialTileLoops(tiled->loops, rewriter);
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
  if (shape && shape->n >= kMatmulPackedNC && shape->m >= kMatmulConservativeMC)
    return MatmulTileShape{kMatmulConservativeMR, kMatmulWideNR, kMatmulKeepKC};
  return defaultGemmSchedule().registerTile;
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

bool evenlyTiled(int64_t extent, int64_t tile) {
  return tile <= 0 || (extent > 0 && extent % tile == 0);
}

bool zeroInitElisionCoversAllOutputTiles(mlir::linalg::MatmulOp matmul) {
  std::optional<MatmulTileShape> shape = staticMatmulShape(matmul);
  if (!shape)
    return false;

  MatmulLoweringPolicy policy = selectMatmulLoweringPolicy(matmul);
  if (policy.plan == MatmulLoweringPlan::Conservative)
    return false;

  if (!usesTiledMatmulPath(policy.plan)) {
    MatmulTileShape microTile = defaultGemmSchedule().registerTile;
    return shape->m <= microTile.m && shape->n <= microTile.n &&
           shape->k <= kMatmulScalarVectorKLimit;
  }

  MatmulTileShape macroTile = policy.schedule.macroTile;
  MatmulTileShape registerTile = policy.schedule.registerTile;
  return evenlyTiled(shape->m, macroTile.m) &&
         evenlyTiled(shape->n, macroTile.n) &&
         evenlyTiled(shape->k, macroTile.k) &&
         evenlyTiled(macroTile.m, registerTile.m) &&
         evenlyTiled(macroTile.n, registerTile.n) &&
         evenlyTiled(macroTile.k, registerTile.k);
}

bool canAbsorbZeroFill(mlir::linalg::MatmulOp matmul,
                       mlir::linalg::FillOp fill) {
  if (!hasPrimitiveMatmulContract(matmul) || fill->getNumResults() != 1 ||
      fill.getNumDpsInputs() != 1 || fill.getNumDpsInits() != 1 ||
      !zeroInitElisionCoversAllOutputTiles(matmul))
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
          hasStaticShapeLargerThanTile(matmul, policy.schedule.macroTile))
        matmuls.push_back(TilingTarget{matmul, policy});
    });

    mlir::IRRewriter rewriter(&getContext());
    for (const TilingTarget &target : matmuls) {
      if (!target.matmul->getBlock())
        continue;
      if (mlir::failed(tileMatmul(target.matmul,
                                  target.policy.schedule.macroTile, rewriter,
                                  target.policy.schedule.packRhs,
                                  target.policy.schedule.outerInterchange,
                                  target.policy.schedule.registerTile))) {
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
    return "pack primitive matmul panels for the selected GEMM schedule";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<mlir::linalg::LinalgDialect, mlir::memref::MemRefDialect>();
  }

  void runOnOperation() final {
    llvm::SmallVector<mlir::linalg::MatmulOp, 16> matmuls;
    getOperation().walk([&](mlir::linalg::MatmulOp matmul) {
      if (shouldPackRhsPanel(matmul))
        matmuls.push_back(matmul);
    });

    mlir::IRRewriter rewriter(&getContext());
    llvm::SmallVector<RhsPrepackPlan, 8> rhsPrepackPlans;
    for (mlir::linalg::MatmulOp matmul : matmuls) {
      if (!matmul->getBlock())
        continue;
      if (shouldPackRhsPanel(matmul) &&
          !tryPrepackFullRhsPanel(matmul, rewriter, rhsPrepackPlans))
        packRhsPanel(matmul, rewriter);
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
      selectDivisibleVectorLanes(elementType, targetType.getDimSize(1),
                                 kPackedCopyVectorBits, /*minLanes=*/1);
  if (!lanes || *lanes <= 1)
    return false;

  mlir::Location loc = copy.getLoc();
  rewriter.setInsertionPoint(copy);
  mlir::Value rowBegin = createIndexConstant(rewriter, loc, 0);
  mlir::Value rowEnd =
      createIndexConstant(rewriter, loc, targetType.getDimSize(0));
  mlir::Value rowStep = createIndexConstant(rewriter, loc, 1);
  mlir::Value colBegin = rowBegin;
  mlir::Value colEnd =
      createIndexConstant(rewriter, loc, targetType.getDimSize(1));
  mlir::Value colStep = createIndexConstant(rewriter, loc, *lanes);

  auto rowLoop =
      mlir::scf::ForOp::create(rewriter, loc, rowBegin, rowEnd, rowStep);
  {
    mlir::OpBuilder::InsertionGuard rowGuard(rewriter);
    rewriter.setInsertionPointToStart(rowLoop.getBody());
    auto colLoop =
        mlir::scf::ForOp::create(rewriter, loc, colBegin, colEnd, colStep);
    {
      mlir::OpBuilder::InsertionGuard colGuard(rewriter);
      rewriter.setInsertionPointToStart(colLoop.getBody());
      mlir::VectorType vectorType =
          mlir::VectorType::get({*lanes}, elementType);
      mlir::Value vector = mlir::vector::LoadOp::create(
                               rewriter, loc, vectorType, source,
                               mlir::ValueRange{rowLoop.getInductionVar(),
                                                colLoop.getInductionVar()})
                               .getResult();
      mlir::vector::StoreOp::create(
          rewriter, loc, vector, target,
          mlir::ValueRange{rowLoop.getInductionVar(),
                           colLoop.getInductionVar()});
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
    return "tile primitive linalg.matmul ops to register-blocked kernels";
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
      if (mlir::succeeded(
              arch::generic::lowerMatmulMicroKernel(matmul, rewriter)))
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

} // namespace py::lowering
