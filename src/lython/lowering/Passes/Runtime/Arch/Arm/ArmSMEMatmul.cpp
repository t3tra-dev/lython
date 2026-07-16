#include "ArmSME.h"

#include "../../Primitive/TensorGemm.h"
#include "../../Primitive/TensorParallel.h"
#include "../../Primitive/TensorSupport.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ArmSME/IR/ArmSME.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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
  // True when `lhs` already holds A as [K][M]. The kernel wants A's column k as
  // a contiguous run, so this is the layout it packs towards; when the LHS
  // arrives transposed the packing step drops out entirely.
  bool lhsTransposed;
};

struct SMEMatmulViews {
  mlir::Value lhs;
  mlir::Value rhs;
  mlir::Value out;
};

// SME setup and LHS panel packing dominate very small contractions. Keep those
// on the generic scalar/vector path, which already has small-matmul handling.
constexpr std::uint64_t kSMEMatmulMinWork = 1024;

// Marks a matmul whose rhs operand is already the panel buffer, disguised as
// [K][N] by a reinterpret_cast. Only the SME kernel may consume such a matmul:
// any other reader would take the panels for rows and silently mis-execute,
// which is why MatmulSMELoweringPass refuses loudly when it cannot take one.
constexpr llvm::StringLiteral kSMERhsPanelsAttr{
    "ly.prim_tensor.sme_rhs_panels"};
// The lhs twin: A's panels were materialised chunk-relative at A's definition
// (hoisted out of any loop that merely re-reads A), and the operand is the
// panel buffer behind a [K][M] reinterpret_cast. Same commitment as the rhs
// mark: only the SME kernel may read it.
constexpr llvm::StringLiteral kSMELhsPanelsAttr{
    "ly.prim_tensor.sme_lhs_panels"};


// Register-block the ZA accumulator: each (row, col) macro step drives a
// rows x cols grid of ZA tiles, loading (rows + cols) scalable vectors and
// issuing (rows * cols) outer products per K step. f32 has exactly four ZA.S
// tiles, so the grid must not exceed four.
struct SMETileGrid {
  int64_t rows;
  int64_t cols;
};

// The balanced block loads the fewest vectors per outer product: 2x2 feeds four
// FMOPAs from 2+2 loads, while 1x4 needs 1+4 for the same four. Both sides being
// a power-of-two block also puts each on one multi-vector load, so the two
// shapes cost the same two load instructions and 2x2 simply moves less.
//
// Why not pick the block from N: a wide N was worth the lopsided 1x4 only while
// the LHS arrived scalar-packed and could not ride a multi-vector load at all.
// Once the transpose moved to the tensor stage (see MatmulLhsTransposePass) the
// crossover went away -- re-measured on M4 Max f32 single-threaded, 2x2 against
// 1x4 runs 1619 vs 1468 at 512^3, 1592 vs 1612 at 1024^3 and 1443 vs 1405 at
// 2048^3. Only the narrow shape still separates them; the rest is a wash, and
// nothing pays for a switch. Both beat the 726 GFLOP/s this kernel started at.
constexpr SMETileGrid kSMETileGrid{2, 2};

// Panel-pack engages when B reaches this size. Below it the strided kernel is
// simply faster and the pack is pure overhead; at and above it the panels win
// on both thread counts. Measured on M4 Max f32 against the strided kernel
// (1T / MT), with A's panels hoisted to its definition:
//   1536^2 B=9.4MB:  1722 vs 1791 / 3264 vs 3349  -- strided
//   2000^2 B=15.3MB: 1540 vs 1431 / 2962 vs 2689  -- packed
//   2048^2 B=16MB:   1650 vs 1469 / 3191 vs 2841  -- packed
//   2560^2 B=26MB:   1675 vs 1438 / 3200 vs 2769  -- packed
// The line sits between the measured points; 12MB splits them. It tracks the
// 16MB L2 a P-cluster owns: once B stops fitting, the strided kernel's
// re-streaming pays full price and the panels' contiguity starts earning.
constexpr std::uint64_t kSMEPanelMinRhsBytes = 12u << 20;

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

  // The kernel handles both LHS layouts: [M][K] (it transposes into a packed
  // panel) and [K][M] (already in the shape it wants).
  bool lhsTransposed = hasTransposedLhsMatmulMaps(matmul);
  if (!hasDefaultMatmulMaps(matmul) && !lhsTransposed)
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
  int64_t k = lhsTransposed ? lhsType.getDimSize(0) : lhsType.getDimSize(1);
  int64_t lhsM = lhsTransposed ? lhsType.getDimSize(1) : lhsType.getDimSize(0);
  if (m <= 0 || n <= 0 || k <= 0 || lhsM != m ||
      rhsType.getDimSize(0) != k || rhsType.getDimSize(1) != n)
    return std::nullopt;

  std::optional<int64_t> rhsInnerStride = staticStride(rhsType, 1);
  std::optional<int64_t> outInnerStride = staticStride(outType, 1);
  if (!rhsInnerStride || !outInnerStride || *rhsInnerStride != 1 ||
      *outInnerStride != 1)
    return std::nullopt;
  // A transposed LHS is read directly, so it needs the contiguous innermost
  // run the packed panel would otherwise have guaranteed.
  if (lhsTransposed) {
    std::optional<int64_t> lhsInnerStride = staticStride(lhsType, 1);
    if (!lhsInnerStride || *lhsInnerStride != 1)
      return std::nullopt;
  }

  return StaticMatmulMemRefs{lhs,     rhs, out, lhsType,      rhsType,
                             outType, m,   n,   k,   *layout, lhsTransposed};
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

// Lanes left at `offset`, floored at zero. The floor matters for the trailing
// tiles of a register block: their offset can start past the extent, and a
// negative bound would reach vector.create_mask as a huge unsigned lane count
// (all lanes true) instead of the empty mask the tile needs.
mlir::Value createRemaining(mlir::OpBuilder &builder, mlir::Location loc,
                            int64_t upper, mlir::Value offset) {
  mlir::Value remaining =
      mlir::arith::SubIOp::create(
          builder, loc, createIndexConstant(builder, loc, upper), offset)
          .getResult();
  return mlir::arith::MaxSIOp::create(builder, loc, remaining,
                                      createIndexConstant(builder, loc, 0))
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

// The zero-init-on-first-reduction contract is carried by A's subview offset
// along K, so which dimension that is follows the LHS layout: [M][K] keeps K
// innermost, [K][M] puts it first. Reading the wrong one here silently answers
// with M's offset, which an untiled M pins at zero -- every K block would then
// re-zero the accumulator and only the last one would survive.
std::optional<mlir::Value>
createFirstReductionTilePredicate(mlir::OpBuilder &builder, mlir::Location loc,
                                  mlir::linalg::MatmulOp matmul,
                                  bool lhsTransposed) {
  std::optional<mlir::Value> reductionOffset = createSubviewOffsetForDimension(
      builder, loc, matmul.getDpsInputOperand(0)->get(),
      /*dimension=*/lhsTransposed ? 0 : 1);
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
      createFirstReductionTilePredicate(builder, loc, matmul,
                                        refs.lhsTransposed);
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

// The kernel needs A's column k as 16 contiguous f32 (LD1W moves contiguous
// bytes), so A is transposed into Ap[k][m] once per contraction.
//
// The transpose runs through ZA: a tile loaded with horizontal slices and
// stored with vertical ones comes back transposed, which is a whole 16x16
// block per pair of instructions. Why not linalg.transpose or a scalar nest:
// the nest moves one element per iteration and measured as expensive as the
// contraction itself (1024^3 f32 went from 821 to 1629 GFLOP/s with the
// packing removed), and linalg.transpose reaches the affine vectorizer as a
// non-identity layout map -- a chunked matmul's LHS is a strided subview --
// which it rejects outright.
mlir::Value createPackedLhsPanel(mlir::OpBuilder &builder, mlir::Location loc,
                                 const StaticMatmulMemRefs &refs,
                                 mlir::VectorType tileType,
                                 mlir::VectorType tileMaskType,
                                 mlir::Value scalarZero, mlir::Value vl) {
  mlir::MemRefType packedType = mlir::MemRefType::get(
      {refs.k, refs.m}, refs.lhsType.getElementType(),
      mlir::MemRefLayoutAttrInterface{}, refs.lhsType.getMemorySpace());
  mlir::Value packed = mlir::memref::AllocOp::create(
                           builder, loc, packedType, mlir::ValueRange{},
                           builder.getI64IntegerAttr(64))
                           .getResult();

  // tile_load/tile_store need the fully dynamic rank-2 form.
  mlir::Value lhsView =
      createDynamicRank2MemRefCast(builder, loc, refs.lhs, refs.lhsType);
  mlir::Value packedView =
      createDynamicRank2MemRefCast(builder, loc, packed, packedType);

  mlir::Value zero = createIndexConstant(builder, loc, 0);
  mlir::Value mUpper = createIndexConstant(builder, loc, refs.m);
  mlir::Value kUpper = createIndexConstant(builder, loc, refs.k);

  auto rowLoop = mlir::scf::ForOp::create(builder, loc, zero, mUpper, vl);
  {
    mlir::OpBuilder::InsertionGuard rowGuard(builder);
    builder.setInsertionPointToStart(rowLoop.getBody());
    mlir::Value m0 = rowLoop.getInductionVar();
    mlir::Value rowsLeft = createRemaining(builder, loc, refs.m, m0);

    auto kLoop = mlir::scf::ForOp::create(builder, loc, zero, kUpper, vl);
    {
      mlir::OpBuilder::InsertionGuard kGuard(builder);
      builder.setInsertionPointToStart(kLoop.getBody());
      mlir::Value k0 = kLoop.getInductionVar();
      mlir::Value colsLeft = createRemaining(builder, loc, refs.k, k0);

      // Load A[m0..][k0..] by rows, store it into Ap[k0..][m0..] by columns.
      mlir::Value loadMask =
          createTileMask(builder, loc, tileMaskType, rowsLeft, colsLeft);
      mlir::Value tile =
          mlir::arm_sme::TileLoadOp::create(
              builder, loc, tileType, lhsView, mlir::ValueRange{m0, k0},
              scalarZero, loadMask, mlir::arm_sme::TileSliceLayout::Horizontal)
              .getResult();
      mlir::Value storeMask =
          createTileMask(builder, loc, tileMaskType, colsLeft, rowsLeft);
      mlir::arm_sme::TileStoreOp::create(
          builder, loc, tile, packedView, mlir::ValueRange{k0, m0}, storeMask,
          mlir::arm_sme::TileSliceLayout::Vertical);
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

// SME2 folds a block's contiguous vectors into one LD1W, keeping the per-K
// instruction count flat as the register block widens. It needs a
// predicate-as-counter rather than an ordinary predicate, and the destinations
// are fixed strided registers, so it is reachable only through the intrinsic --
// no vector dialect op models it.
//
// Only for a fully in-bounds block: the counter form a partial block would need
// (WHILELT with a vlx2/vlx4 counter) has no intrinsic here, so those stay on
// the masked single-vector path.
// Raw address of `memref`'s element `linear`. The multi-vector intrinsics take
// an address rather than a memref, so every caller needs this.
mlir::Value createElementAddress(mlir::OpBuilder &builder, mlir::Location loc,
                                 mlir::Value memref, mlir::Value linear) {
  auto i64 = builder.getI64Type();
  auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  mlir::Value base = mlir::memref::ExtractAlignedPointerAsIndexOp::create(
      builder, loc, memref);
  mlir::Value baseI64 =
      mlir::arith::IndexCastOp::create(builder, loc, i64, base);
  mlir::Value linearI64 =
      mlir::arith::IndexCastOp::create(builder, loc, i64, linear);
  mlir::Value bytes = mlir::arith::ConstantIntOp::create(builder, loc, i64, 4);
  mlir::Value addr = mlir::arith::AddIOp::create(
      builder, loc, baseI64,
      mlir::arith::MulIOp::create(builder, loc, linearI64, bytes));
  return mlir::LLVM::IntToPtrOp::create(builder, loc, ptrType, addr);
}

// The element index of a [K][X] operand's (row, col), using the layout's own
// pitch: a view's rows are spaced by the parent's, not by its own width.
// Callers gate on that pitch being static (see multiVectorLoadable).
mlir::Value createStridedIndex(mlir::OpBuilder &builder, mlir::Location loc,
                               mlir::MemRefType memrefType, mlir::Value row,
                               mlir::Value col) {
  std::optional<int64_t> rowStride = staticStride(memrefType, 0);
  mlir::Value stride = createIndexConstant(builder, loc, *rowStride);
  return mlir::arith::AddIOp::create(
      builder, loc, mlir::arith::MulIOp::create(builder, loc, row, stride),
      col);
}

llvm::SmallVector<mlir::Value, 4>
createMultiVectorLoad(mlir::OpBuilder &builder, mlir::Location loc,
                      mlir::Value memref, mlir::VectorType vectorType,
                      mlir::Value linear, int64_t count) {
  mlir::MLIRContext *context = builder.getContext();
  auto countType = mlir::LLVM::LLVMTargetExtType::get(
      context, "aarch64.svcount", /*typeParams=*/{}, /*intParams=*/{});
  mlir::Value ptr = createElementAddress(builder, loc, memref, linear);

  mlir::Value pn =
      mlir::LLVM::CallIntrinsicOp::create(
          builder, loc, mlir::TypeRange{countType},
          builder.getStringAttr("llvm.aarch64.sve.ptrue.c32"),
          mlir::ValueRange{}, mlir::LLVM::FastmathFlagsAttr{})
          .getResult(0);
  llvm::SmallVector<mlir::Type, 4> fields(count, vectorType);
  auto structType = mlir::LLVM::LLVMStructType::getLiteral(context, fields);
  llvm::StringRef intrinsic = count == 2
                                  ? "llvm.aarch64.sve.ld1.pn.x2.nxv4f32"
                                  : "llvm.aarch64.sve.ld1.pn.x4.nxv4f32";
  mlir::Value loaded = mlir::LLVM::CallIntrinsicOp::create(
                           builder, loc, mlir::TypeRange{structType},
                           builder.getStringAttr(intrinsic),
                           mlir::ValueRange{pn, ptr},
                           mlir::LLVM::FastmathFlagsAttr{})
                           .getResult(0);

  llvm::SmallVector<mlir::Value, 4> result;
  for (int64_t index = 0; index < count; ++index)
    result.push_back(
        mlir::LLVM::ExtractValueOp::create(builder, loc, loaded, index));
  return result;
}

void createMultiVectorStore(mlir::OpBuilder &builder, mlir::Location loc,
                            mlir::Value memref, mlir::Value linear,
                            llvm::ArrayRef<mlir::Value> vectors) {
  mlir::MLIRContext *context = builder.getContext();
  auto countType = mlir::LLVM::LLVMTargetExtType::get(
      context, "aarch64.svcount", /*typeParams=*/{}, /*intParams=*/{});
  mlir::Value ptr = createElementAddress(builder, loc, memref, linear);
  mlir::Value pn = mlir::LLVM::CallIntrinsicOp::create(
                       builder, loc, mlir::TypeRange{countType},
                       builder.getStringAttr("llvm.aarch64.sve.ptrue.c32"),
                       mlir::ValueRange{}, mlir::LLVM::FastmathFlagsAttr{})
                       .getResult(0);
  // Mirror of the load's operand order, not a copy of it: st1.pn takes the
  // vectors first and the counter after them, where ld1.pn leads with the
  // counter.
  llvm::SmallVector<mlir::Value, 4> operands(vectors.begin(), vectors.end());
  operands.push_back(pn);
  operands.push_back(ptr);
  llvm::StringRef intrinsic = vectors.size() == 2
                                  ? "llvm.aarch64.sve.st1.pn.x2.nxv4f32"
                                  : "llvm.aarch64.sve.st1.pn.x4.nxv4f32";
  mlir::LLVM::CallIntrinsicOp::create(builder, loc, mlir::TypeRange{},
                                      builder.getStringAttr(intrinsic), operands,
                                      mlir::LLVM::FastmathFlagsAttr{});
}


// Streaming vector length in f32 lanes, readable outside streaming mode: the
// intrinsic wraps RDSVL, which is specified to work in either mode. The
// arm_sme.streaming_vl op would say the same thing, but emitting an arm_sme op
// into the caller risks the streaming-mode attribution pass treating the whole
// caller as SME code. cntsd (doubles per streaming vector) is the one counting
// form LLVM still defines, so f32 lanes are twice it.
mlir::Value createStreamingLaneCount(mlir::OpBuilder &builder,
                                     mlir::Location loc) {
  auto i64 = builder.getI64Type();
  mlir::Value doubles =
      mlir::LLVM::CallIntrinsicOp::create(
          builder, loc, mlir::TypeRange{i64},
          builder.getStringAttr("llvm.aarch64.sme.cntsd"), mlir::ValueRange{},
          mlir::LLVM::FastmathFlagsAttr{})
          .getResult(0);
  mlir::Value words = mlir::LLVM::MulOp::create(
      builder, loc, doubles,
      mlir::LLVM::ConstantOp::create(builder, loc, i64,
                                     builder.getI64IntegerAttr(2)));
  return mlir::arith::IndexCastOp::create(builder, loc,
                                          builder.getIndexType(), words);
}

// Materialise [K][N] B as ceildiv(N, width) panels, each [K][width] contiguous
// and zero-filled past N. Runs in the caller, outside any streaming region, so
// it sticks to fixed-width vectors: Apple silicon has no non-streaming SVE to
// reach for, and packing is memory-bound anyway.
// Pack `source` ([K][X], non-streaming caller) as panels of `width` columns
// into `dest` starting at element `destBase`. Factored from the rhs pack so
// the hoisted lhs pack can lay several chunks' panels into one buffer.
void createPanelBufferInto(mlir::OpBuilder &builder, mlir::Location loc,
                           mlir::Value dest, mlir::Value destBase,
                           mlir::Value source, int64_t k, int64_t x,
                           mlir::Value width) {
  auto f32 = builder.getF32Type();
  auto vec4 = mlir::VectorType::get({4}, f32);
  mlir::Value zero = createIndexConstant(builder, loc, 0);
  mlir::Value one = createIndexConstant(builder, loc, 1);
  mlir::Value four = createIndexConstant(builder, loc, 4);
  mlir::Value kValue = createIndexConstant(builder, loc, k);
  mlir::Value nValue = createIndexConstant(builder, loc, x);
  mlir::Value rhs = source;
  mlir::Value packed = dest;
  mlir::Value panels =
      mlir::arith::CeilDivSIOp::create(builder, loc, nValue, width);

  auto panelLoop = mlir::scf::ForOp::create(builder, loc, zero, panels, one);
  // Panels are disjoint, so the copy forks like the kernel does. Serial, this
  // loop was the multi-threaded mode's whole deficit: it has to run before the
  // kernel's own fork (every chunk shares its output), and Amdahl charged the
  // full copy against the parallel speedup -- measured 2819 against 3053
  // GFLOP/s at 2048^3 MT with the copy serial.
  panelLoop->setAttr(kParallelDispatchAttrName,
                     mlir::UnitAttr::get(builder.getContext()));
  {
    mlir::OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToStart(panelLoop.getBody());
    mlir::Value panel = panelLoop.getInductionVar();
    mlir::Value colBase =
        mlir::arith::MulIOp::create(builder, loc, panel, width);
    mlir::Value panelEnd =
        mlir::arith::AddIOp::create(builder, loc, colBase, width);
    mlir::Value whole = mlir::arith::CmpIOp::create(
        builder, loc, mlir::arith::CmpIPredicate::sle, panelEnd, nValue);
    mlir::Value panelTimesK =
        mlir::arith::MulIOp::create(builder, loc, panel, kValue);
    auto ifOp =
        mlir::scf::IfOp::create(builder, loc, whole, /*withElseRegion=*/true);
    {
      // Whole panel. The row copy is the entire cost of this pass, and a
      // dynamic-trip 4-wide loop ran ~11 GB/s -- the loop overhead was most of
      // it. The panel width is 2*SVL/32, so on every shipping Apple SME part it
      // is exactly 32: specialise that case to straight-line code (eight
      // vector<4> moves the backend can pair into ldp/stp q) plus the L2
      // prefetch Accelerate's own pack loop runs, and keep the dynamic loop
      // as the fallback for any other SVL.
      mlir::OpBuilder::InsertionGuard g2(builder);
      builder.setInsertionPointToStart(ifOp.thenBlock());
      mlir::Value w32 = mlir::arith::CmpIOp::create(
          builder, loc, mlir::arith::CmpIPredicate::eq, width,
          createIndexConstant(builder, loc, 32));
      auto wIf =
          mlir::scf::IfOp::create(builder, loc, w32, /*withElseRegion=*/true);
      {
        mlir::OpBuilder::InsertionGuard g3(builder);
        builder.setInsertionPointToStart(wIf.thenBlock());
        auto kLoop = mlir::scf::ForOp::create(builder, loc, zero, kValue, one);
        builder.setInsertionPointToStart(kLoop.getBody());
        mlir::Value step = kLoop.getInductionVar();
        mlir::Value dstRow = mlir::arith::AddIOp::create(
            builder, loc, destBase,
            mlir::arith::MulIOp::create(
                builder, loc,
                mlir::arith::AddIOp::create(builder, loc, panelTimesK, step),
                width));
        // A few rows ahead, clamped: PRFM cannot fault, but keep the index
        // arithmetic in bounds anyway.
        mlir::Value ahead = mlir::arith::MinSIOp::create(
            builder, loc,
            mlir::arith::AddIOp::create(builder, loc, step,
                                        createIndexConstant(builder, loc, 16)),
            mlir::arith::SubIOp::create(builder, loc, kValue, one));
        mlir::memref::PrefetchOp::create(builder, loc, rhs,
                                         mlir::ValueRange{ahead, colBase},
                                         /*isWrite=*/false,
                                         /*localityHint=*/2,
                                         /*isDataCache=*/true);
        llvm::SmallVector<mlir::Value, 8> vs;
        for (int64_t jj = 0; jj < 32; jj += 4) {
          mlir::Value col = mlir::arith::AddIOp::create(
              builder, loc, colBase, createIndexConstant(builder, loc, jj));
          vs.push_back(mlir::vector::LoadOp::create(
              builder, loc, vec4, rhs, mlir::ValueRange{step, col}));
        }
        for (int64_t jj = 0; jj < 32; jj += 4) {
          mlir::Value dst = mlir::arith::AddIOp::create(
              builder, loc, dstRow, createIndexConstant(builder, loc, jj));
          mlir::vector::StoreOp::create(builder, loc, vs[jj / 4], packed,
                                        mlir::ValueRange{dst});
        }
      }
      {
        mlir::OpBuilder::InsertionGuard g3(builder);
        builder.setInsertionPointToStart(wIf.elseBlock());
        auto kLoop = mlir::scf::ForOp::create(builder, loc, zero, kValue, one);
        builder.setInsertionPointToStart(kLoop.getBody());
        mlir::Value step = kLoop.getInductionVar();
        mlir::Value dstRow = mlir::arith::AddIOp::create(
            builder, loc, destBase,
            mlir::arith::MulIOp::create(
                builder, loc,
                mlir::arith::AddIOp::create(builder, loc, panelTimesK, step),
                width));
        auto jLoop = mlir::scf::ForOp::create(builder, loc, zero, width, four);
        builder.setInsertionPointToStart(jLoop.getBody());
        mlir::Value j = jLoop.getInductionVar();
        mlir::Value col = mlir::arith::AddIOp::create(builder, loc, colBase, j);
        mlir::Value v = mlir::vector::LoadOp::create(
            builder, loc, vec4, rhs, mlir::ValueRange{step, col});
        mlir::vector::StoreOp::create(
            builder, loc, v, packed,
            mlir::ValueRange{
                mlir::arith::AddIOp::create(builder, loc, dstRow, j)});
      }
    }
    {
      // Trailing panel: scalar with a bounds check, zero past N. At most one
      // panel per contraction ever takes this path.
      mlir::OpBuilder::InsertionGuard g2(builder);
      builder.setInsertionPointToStart(ifOp.elseBlock());
      mlir::Value zeroF = mlir::arith::ConstantOp::create(
          builder, loc, f32, builder.getF32FloatAttr(0.0f));
      auto kLoop = mlir::scf::ForOp::create(builder, loc, zero, kValue, one);
      builder.setInsertionPointToStart(kLoop.getBody());
      mlir::Value step = kLoop.getInductionVar();
      mlir::Value dstRow = mlir::arith::AddIOp::create(
          builder, loc, destBase,
          mlir::arith::MulIOp::create(
              builder, loc,
              mlir::arith::AddIOp::create(builder, loc, panelTimesK, step),
              width));
      auto jLoop = mlir::scf::ForOp::create(builder, loc, zero, width, one);
      builder.setInsertionPointToStart(jLoop.getBody());
      mlir::Value j = jLoop.getInductionVar();
      mlir::Value col = mlir::arith::AddIOp::create(builder, loc, colBase, j);
      mlir::Value inBounds = mlir::arith::CmpIOp::create(
          builder, loc, mlir::arith::CmpIPredicate::slt, col, nValue);
      auto pick = mlir::scf::IfOp::create(builder, loc, mlir::TypeRange{f32},
                                          inBounds, /*withElseRegion=*/true);
      {
        mlir::OpBuilder::InsertionGuard g3(builder);
        builder.setInsertionPointToStart(pick.thenBlock());
        mlir::Value v = mlir::memref::LoadOp::create(
            builder, loc, rhs, mlir::ValueRange{step, col});
        mlir::scf::YieldOp::create(builder, loc, v);
        builder.setInsertionPointToStart(pick.elseBlock());
        mlir::scf::YieldOp::create(builder, loc, zeroF);
      }
      mlir::memref::StoreOp::create(
          builder, loc, pick.getResult(0), packed,
          mlir::ValueRange{
              mlir::arith::AddIOp::create(builder, loc, dstRow, j)});
    }
  }
}

mlir::Value createRhsPanels(mlir::OpBuilder &builder, mlir::Location loc,
                            mlir::Value rhs, int64_t k, int64_t n,
                            mlir::Value width) {
  auto f32 = builder.getF32Type();
  mlir::Value kValue = createIndexConstant(builder, loc, k);
  mlir::Value nValue = createIndexConstant(builder, loc, n);
  mlir::Value panels =
      mlir::arith::CeilDivSIOp::create(builder, loc, nValue, width);
  mlir::Value total = mlir::arith::MulIOp::create(
      builder, loc, mlir::arith::MulIOp::create(builder, loc, panels, kValue),
      width);
  auto packedType = mlir::MemRefType::get({mlir::ShapedType::kDynamic}, f32);
  mlir::Value packed = mlir::memref::AllocOp::create(builder, loc, packedType,
                                                     mlir::ValueRange{total});
  createPanelBufferInto(builder, loc, packed,
                        createIndexConstant(builder, loc, 0), rhs, k, n,
                        width);
  return packed;
}

// Copy a [K][X] operand into panel-major order: panel p of width `width` holds
// [K][width], so a reduction step walks it contiguously instead of jumping the
// source's row pitch.
//
// This is what the kernel's throughput actually turns on. Measured on M4 Max
// f32 single-threaded with the same 2x2 FMOPA loop, reading both operands
// packed runs a flat 1999-2020 GFLOP/s from 512^3 to 2048^3, while reading them
// at their row pitch decays from 2019 to 1690 -- and packing only one side buys
// nothing (1.00-1.07x). Accelerate's kernel does the same thing: its operand
// pointers advance by exactly the two vectors it just read (`add x24, x24,
// #0x80`), and it spends 7.5% of a 2000^3 sgemm in the two pack loops that earn
// that.
//
// `width` is the run the kernel loads at once (2*VL), so it is a run-time value
// and the packed buffer is a dynamic memref. Why not fix it at 32 the way
// Accelerate does: that is a bet on SVL being 512 bits, which holds on every
// Apple SME part but is not what the architecture promises, and the fallback it
// would need is a second copy of this kernel.
mlir::Value createPanelPack(mlir::OpBuilder &builder, mlir::Location loc,
                            mlir::Value source, mlir::MemRefType sourceType,
                            mlir::VectorType vectorType, int64_t k, int64_t x,
                            mlir::Value width, int64_t vectorsPerPanel) {
  mlir::Type elementType = sourceType.getElementType();
  mlir::Value zero = createIndexConstant(builder, loc, 0);
  mlir::Value one = createIndexConstant(builder, loc, 1);
  mlir::Value kValue = createIndexConstant(builder, loc, k);
  mlir::Value xValue = createIndexConstant(builder, loc, x);

  // ceildiv(x, width) panels, each [K][width]; the last one runs past x and is
  // zero-filled, which is what lets the kernel read every panel unmasked.
  mlir::Value panels = mlir::arith::CeilDivSIOp::create(builder, loc, xValue,
                                                        width);
  mlir::Value padded = mlir::arith::MulIOp::create(builder, loc, panels, width);
  mlir::Value total =
      mlir::arith::MulIOp::create(builder, loc, padded, kValue);
  auto packedType = mlir::MemRefType::get({mlir::ShapedType::kDynamic},
                                          elementType);
  mlir::Value packed = mlir::memref::AllocOp::create(builder, loc, packedType,
                                                     mlir::ValueRange{total});

  auto panelLoop = mlir::scf::ForOp::create(builder, loc, zero, panels, one);
  {
    mlir::OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToStart(panelLoop.getBody());
    mlir::Value panel = panelLoop.getInductionVar();
    mlir::Value col = mlir::arith::MulIOp::create(builder, loc, panel, width);
    auto kLoop = mlir::scf::ForOp::create(builder, loc, zero, kValue, one);
    {
      mlir::OpBuilder::InsertionGuard g2(builder);
      builder.setInsertionPointToStart(kLoop.getBody());
      mlir::Value step = kLoop.getInductionVar();
      // Masked reads: the trailing panel is short, and its padding must read as
      // zero so the outer products it feeds contribute nothing.
      llvm::SmallVector<mlir::Value, 4> vectors;
      mlir::Value vl = mlir::arith::DivSIOp::create(
          builder, loc, width, createIndexConstant(builder, loc,
                                                   vectorsPerPanel));
      for (int64_t i = 0; i < vectorsPerPanel; ++i) {
        mlir::Value offset = mlir::arith::AddIOp::create(
            builder, loc, col,
            mlir::arith::MulIOp::create(builder, loc, vl,
                                        createIndexConstant(builder, loc, i)));
        mlir::Value remaining =
            createRemaining(builder, loc, x, offset);
        mlir::Value mask = createMask(
            builder, loc, scalableMaskType(builder.getContext(),
                                           SMEElementLayout{
                                               vectorType.getShape()[0],
                                               mlir::arm_sme::TypeSize::Word}),
            remaining);
        std::optional<mlir::Value> zeroScalar =
            createPrimitiveZeroValue(builder, loc, elementType);
        mlir::Value pad = createZeroVector(builder, loc, vectorType,
                                           *zeroScalar);
        vectors.push_back(mlir::vector::MaskedLoadOp::create(
                              builder, loc, vectorType, source,
                              mlir::ValueRange{step, offset}, mask, pad)
                              .getResult());
      }
      mlir::Value panelBase = mlir::arith::MulIOp::create(
          builder, loc,
          mlir::arith::AddIOp::create(
              builder, loc,
              mlir::arith::MulIOp::create(builder, loc, panel, kValue), step),
          width);
      createMultiVectorStore(builder, loc, packed, panelBase, vectors);
    }
  }
  return packed;
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
  mlir::Value mUpper = createIndexConstant(rewriter, loc, refs->m);
  mlir::Value nUpper = createIndexConstant(rewriter, loc, refs->n);
  mlir::Value kUpper = createIndexConstant(rewriter, loc, refs->k);
  mlir::Value vl = mlir::arm_sme::StreamingVLOp::create(
                       rewriter, loc, rewriter.getIndexType(),
                       refs->layout.streamingVLUnit)
                       .getResult();
  SMETileGrid grid = kSMETileGrid;
  mlir::Value rowBlock =
      mlir::arith::MulIOp::create(
          rewriter, loc, vl, createIndexConstant(rewriter, loc, grid.rows))
          .getResult();
  mlir::Value colBlock =
      mlir::arith::MulIOp::create(
          rewriter, loc, vl, createIndexConstant(rewriter, loc, grid.cols))
          .getResult();
  // Panel-pack both operands when the multi-vector loads are reachable: it is
  // what makes the reduction step read contiguously (see createPanelPack).
  // Packing once per contraction costs O(M*K + K*N) against O(M*K*N) of
  // arithmetic -- 0.1% at 2000^3 -- so unlike a BLIS-style schedule there is
  // nothing to amortise and no reason to re-pack per block.
  auto multiVectorLoadable = [](mlir::MemRefType type) {
    std::optional<int64_t> rowStride = staticStride(type, 0);
    std::optional<int64_t> innerStride = staticStride(type, 1);
    return rowStride && innerStride && *innerStride == 1;
  };
  // Panels engage only when MatmulSMERhsPackPass marked the matmul: the mark
  // is a commitment (the rhs type lies about a panel buffer, so the strided
  // path must never read it), and the pass owns the policy of when packing
  // pays. Without the mark this kernel reads both operands strided, exactly as
  // before packing existed.
  bool rhsPrePacked = matmul->hasAttr(kSMERhsPanelsAttr);
  bool lhsPrePacked = matmul->hasAttr(kSMELhsPanelsAttr);
  bool packOperands = rhsPrePacked;

  // An already-transposed LHS is the packed panel: nothing to do.
  mlir::Value packedLhs =
      refs->lhsTransposed
          ? refs->lhs
          : createPackedLhsPanel(rewriter, loc, *refs, tileType, tileMaskType,
                                 *scalarZero, vl);
  auto packedLhsType = mlir::cast<mlir::MemRefType>(packedLhs.getType());
  mlir::Value panelLhs, panelRhs;
  if (packOperands) {
    // A is chunked, so each worker packs its own disjoint slice here. B is
    // shared by every chunk, which is why MatmulSMERhsPackPass packs it once
    // in the caller instead: inline it re-copied the whole of B per chunk.
    panelLhs = lhsPrePacked
                   ? refs->lhs
                   : createPanelPack(rewriter, loc, packedLhs, packedLhsType,
                                     vectorType, refs->k, refs->m, rowBlock,
                                     grid.rows);
    panelRhs = rhsPrePacked
                   ? refs->rhs
                   : createPanelPack(rewriter, loc, refs->rhs, refs->rhsType,
                                     vectorType, refs->k, refs->n, colBlock,
                                     grid.cols);
  }
  // Where this kernel's rows sit in the hoisted panel buffer. The buffer is
  // chunk-relative ([chunk][panels][K][W]); the chunk index falls out of the
  // output subview's row offset and the (static) chunk height m. The lhs
  // operand itself is a subview of the disguised [K][M] type -- taken only so
  // linalg.matmul's shape inference holds -- and reads deliberately go through
  // ExtractAlignedPointerAsIndexOp, which returns the allocation base and
  // ignores subview offsets: the very footgun that once freed these buffers
  // early is what lets one chunk-relative buffer serve every chunk.
  mlir::Value lhsPanelBase;
  if (lhsPrePacked) {
    std::optional<mlir::Value> outRow = createSubviewOffsetForDimension(
        rewriter, loc, refs->out, /*dimension=*/0);
    mlir::Value rowOffset =
        outRow ? *outRow : createIndexConstant(rewriter, loc, 0);
    mlir::Value chunk = mlir::arith::DivSIOp::create(
        rewriter, loc, rowOffset, createIndexConstant(rewriter, loc, refs->m));
    mlir::Value panelsPerChunk = mlir::arith::CeilDivSIOp::create(
        rewriter, loc, createIndexConstant(rewriter, loc, refs->m), rowBlock);
    mlir::Value chunkStride = mlir::arith::MulIOp::create(
        rewriter, loc,
        mlir::arith::MulIOp::create(rewriter, loc, panelsPerChunk,
                                    createIndexConstant(rewriter, loc,
                                                        refs->k)),
        rowBlock);
    lhsPanelBase =
        mlir::arith::MulIOp::create(rewriter, loc, chunk, chunkStride);
  }
  SMEMatmulViews views{
      createDynamicRank2MemRefCast(rewriter, loc, packedLhs, packedLhsType),
      createDynamicRank2MemRefCast(rewriter, loc, refs->rhs, refs->rhsType),
      createDynamicRank2MemRefCast(rewriter, loc, refs->out, refs->outType)};

  // Trailing tiles of a block may start past the extent; their masks come out
  // empty, so the reads pad with zero, the outer products contribute nothing,
  // and the stores drop. No separate remainder nest is needed.
  auto tileOffsets = [&](mlir::Value base, int64_t count) {
    llvm::SmallVector<mlir::Value, 2> offsets;
    offsets.push_back(base);
    for (int64_t index = 1; index < count; ++index)
      offsets.push_back(mlir::arith::AddIOp::create(
                            rewriter, loc, base,
                            mlir::arith::MulIOp::create(
                                rewriter, loc, vl,
                                createIndexConstant(rewriter, loc, index))
                                .getResult())
                            .getResult());
    return offsets;
  };

  // Panel mode adds one loop above the pair: column-panel groups sized so a
  // group's B panels sit in L2 across the whole row sweep. Without it every
  // row block re-streams all of B from DRAM -- 64 x 16MB at 2048^3, and the
  // contraction pins at the ~1.7 TFLOP/s that bandwidth allows (which is also
  // where Accelerate's own 1T numbers sit at these sizes). Grouping costs an
  // extra pass over A per group instead: A x ceil(N/NC), a fraction of B's
  // former bill. This is the N-blocking that was useless for the strided
  // kernel (a strided column strip scatters across the row pitch and measured
  // no better than nothing) -- panels made it contiguous, which is what makes
  // it work.
  //
  // Group width: the largest colBlock multiple keeping K*NC*4 within a 4MB
  // slice of L2. Static shape, so it folds to a constant; when NC >= N the
  // group loop runs once and canonicalises away.
  mlir::Value colGroup = nUpper;
  if (packOperands) {
    int64_t nc = (4 << 20) / (refs->k * 4);
    nc = std::max<int64_t>(nc - nc % 32, 32);
    colGroup = createIndexConstant(rewriter, loc, nc);
    colGroup = mlir::arith::MaxSIOp::create(rewriter, loc, colGroup, colBlock);
  }
  auto groupLoop =
      mlir::scf::ForOp::create(rewriter, loc, zero, nUpper, colGroup);
  {
  // Everything through the row/col nest lives in the group body; the guard is
  // what keeps the panel deallocs below OUTSIDE it. Without it they land at
  // the end of the first group's iteration and every later group reads freed
  // panels -- which is invisible at shapes small enough to fit one group,
  // exactly the shapes the smoke tests use.
  mlir::OpBuilder::InsertionGuard groupGuard(rewriter);
  rewriter.setInsertionPointToStart(groupLoop.getBody());
  mlir::Value groupLo = groupLoop.getInductionVar();
  mlir::Value groupHi = mlir::arith::MinSIOp::create(
      rewriter, loc,
      mlir::arith::AddIOp::create(rewriter, loc, groupLo, colGroup), nUpper);

  auto rowLoop = mlir::scf::ForOp::create(rewriter, loc, zero, mUpper, rowBlock);
  {
    mlir::OpBuilder::InsertionGuard rowGuard(rewriter);
    rewriter.setInsertionPointToStart(rowLoop.getBody());
    llvm::SmallVector<mlir::Value, 2> rows =
        tileOffsets(rowLoop.getInductionVar(), grid.rows);
    llvm::SmallVector<mlir::Value, 2> rowsRemaining;
    llvm::SmallVector<mlir::Value, 2> rowMasks;
    for (mlir::Value row : rows) {
      rowsRemaining.push_back(createRemaining(rewriter, loc, refs->m, row));
      rowMasks.push_back(
          createMask(rewriter, loc, maskType, rowsRemaining.back()));
    }

    auto colLoop =
        mlir::scf::ForOp::create(rewriter, loc, groupLo, groupHi, colBlock);
    {
      mlir::OpBuilder::InsertionGuard colGuard(rewriter);
      rewriter.setInsertionPointToStart(colLoop.getBody());
      llvm::SmallVector<mlir::Value, 2> cols =
          tileOffsets(colLoop.getInductionVar(), grid.cols);
      llvm::SmallVector<mlir::Value, 2> colsRemaining;
      llvm::SmallVector<mlir::Value, 2> colMasks;
      for (mlir::Value col : cols) {
        colsRemaining.push_back(createRemaining(rewriter, loc, refs->n, col));
        colMasks.push_back(
            createMask(rewriter, loc, maskType, colsRemaining.back()));
      }

      // ld1.pn.x2/x4 only: the intrinsic comes in those two widths, and f32
      // is the only element type wired up here.
      bool useMultiVector = packOperands ||
                            (target.usesArmSME2() &&
                            (grid.cols == 2 || grid.cols == 4) &&
                            (grid.rows == 1 || grid.rows == 2) &&
                            refs->outType.getElementType().isF32() &&
                             multiVectorLoadable(refs->rhsType) &&
                             multiVectorLoadable(packedLhsType));

      llvm::SmallVector<mlir::Value, 4> tileMasks;
      llvm::SmallVector<mlir::Value, 4> initialAccs;
      for (auto [rowIndex, row] : llvm::enumerate(rows)) {
        for (auto [colIndex, col] : llvm::enumerate(cols)) {
          mlir::Value tileMask =
              createTileMask(rewriter, loc, tileMaskType,
                             rowsRemaining[rowIndex], colsRemaining[colIndex]);
          tileMasks.push_back(tileMask);
          mlir::Value acc;
          if (mlir::failed(createAccumulatorInit(matmul, rewriter, loc, *refs,
                                                 views, tileType, *scalarZero,
                                                 row, col, tileMask, acc)))
            return mlir::failure();
          initialAccs.push_back(acc);
        }
      }

      // The multi-vector load needs the whole column block in bounds: its
      // counter form covers all four vectors at once and has no masked
      // equivalent here. Pick per block at run time -- SVL is only known then,
      // so which blocks are whole cannot be decided while compiling.
      // One reduction step: the loads for `k` and the outer products they feed.
      auto emitKStep = [&](mlir::OpBuilder &builder, mlir::Location nestedLoc,
                           mlir::Value k, bool multiVector,
                           llvm::SmallVectorImpl<mlir::Value> &accs) {
        // The packed LHS is [K][M], so a row block is contiguous there too:
        // both operands can ride the multi-vector load.
        // Packed: panel p of width `block` holds [K][block] contiguously, so a
        // reduction step is (p*K + k)*block elements in. Every panel is full
        // (the pack zero-fills the trailing one), so the load never needs the
        // mask the strided form does.
        auto panelIndex = [&](mlir::Value base, mlir::Value block) {
            mlir::Value panel =
                mlir::arith::DivSIOp::create(builder, nestedLoc, base, block);
            return mlir::arith::MulIOp::create(
                builder, nestedLoc,
                mlir::arith::AddIOp::create(
                    builder, nestedLoc,
                    mlir::arith::MulIOp::create(builder, nestedLoc, panel,
                                                kUpper),
                    k),
                block);
        };
        llvm::SmallVector<mlir::Value, 4> lhsVectors;
        if (packOperands) {
          mlir::Value lhsLinear = panelIndex(rows.front(), rowBlock);
          if (lhsPrePacked)
            lhsLinear = mlir::arith::AddIOp::create(builder, nestedLoc,
                                                    lhsPanelBase, lhsLinear);
          lhsVectors = createMultiVectorLoad(builder, nestedLoc, panelLhs,
                                             vectorType, lhsLinear, grid.rows);
        } else if (multiVector && grid.rows > 1) {
          lhsVectors = createMultiVectorLoad(
              builder, nestedLoc, packedLhs, vectorType,
              createStridedIndex(builder, nestedLoc, packedLhsType, k,
                                 rows.front()),
              grid.rows);
        } else {
          for (auto [rowIndex, row] : llvm::enumerate(rows))
            lhsVectors.push_back(createLhsVector(builder, nestedLoc, views,
                                                 vectorType, *scalarZero, row,
                                                 k, rowMasks[rowIndex]));
        }
        llvm::SmallVector<mlir::Value, 4> rhsVectors;
        if (packOperands) {
          rhsVectors = createMultiVectorLoad(
              builder, nestedLoc, panelRhs, vectorType,
              panelIndex(cols.front(), colBlock), grid.cols);
        } else if (multiVector) {
          rhsVectors = createMultiVectorLoad(
              builder, nestedLoc, refs->rhs, vectorType,
              createStridedIndex(builder, nestedLoc, refs->rhsType, k,
                                 cols.front()),
              grid.cols);
        } else {
          for (auto [colIndex, col] : llvm::enumerate(cols))
            rhsVectors.push_back(
                createRhsVector(builder, nestedLoc, views, vectorType,
                                *scalarZero, k, col, colMasks[colIndex]));
        }
        for (auto [rowIndex, lhsVector] : llvm::enumerate(lhsVectors)) {
          for (auto [colIndex, rhsVector] : llvm::enumerate(rhsVectors)) {
            std::size_t tile = rowIndex * rhsVectors.size() + colIndex;
            accs[tile] = mlir::vector::OuterProductOp::create(
                             builder, nestedLoc, tileType, lhsVector, rhsVector,
                             accs[tile], mlir::vector::CombiningKind::ADD)
                             .getResult();
          }
        }
      };

      // Why not unroll the reduction: it was worth 774 -> 821 GFLOP/s while the
      // LHS arrived scalar-packed and the loads were what starved the outer
      // products. Once both operands ride the multi-vector load, that load
      // already supplies the overlap unrolling used to buy, and re-measured on
      // M4 Max f32 single-threaded it is neutral -- 1024^3 runs 1585 / 1605 /
      // 1594 / 1596 GFLOP/s and 2048^3 runs 1474 / 1474 / 1452 / 1456 at an
      // unroll of 1 / 2 / 4 / 8. Unrolling also constrained K, since a factor
      // that did not divide it needed a residue loop.
      mlir::Value one = createIndexConstant(rewriter, loc, 1);
      auto buildKLoop = [&](mlir::OpBuilder &b, bool multiVector) {
        return mlir::scf::ForOp::create(
            b, loc, zero, kUpper, one, initialAccs,
            [&](mlir::OpBuilder &builder, mlir::Location nestedLoc,
                mlir::Value k, mlir::ValueRange iterArgs) {
              llvm::SmallVector<mlir::Value, 4> accs(iterArgs.begin(),
                                                     iterArgs.end());
              emitKStep(builder, nestedLoc, k, multiVector, accs);
              mlir::scf::YieldOp::create(builder, nestedLoc, accs);
            });
      };

      llvm::SmallVector<mlir::Value, 4> accResults;
      if (packOperands) {
        auto loop = buildKLoop(rewriter, /*multiVector=*/true);
        accResults.assign(loop.getResults().begin(), loop.getResults().end());
      } else if (useMultiVector) {
        mlir::Value blockEnd =
            mlir::arith::AddIOp::create(rewriter, loc, cols.back(), vl);
        mlir::Value whole = mlir::arith::CmpIOp::create(
            rewriter, loc, mlir::arith::CmpIPredicate::sle, blockEnd, nUpper);
        // The row block gates the same branch because the LHS rides the
        // multi-vector load too, and that load has no mask: on a partial row
        // block it would read past M. Both operands share one predicate, so a
        // partial block on either side puts both back on the masked path.
        if (grid.rows > 1) {
          mlir::Value rowEnd =
              mlir::arith::AddIOp::create(rewriter, loc, rows.back(), vl);
          whole = mlir::arith::AndIOp::create(
              rewriter, loc, whole,
              mlir::arith::CmpIOp::create(rewriter, loc,
                                          mlir::arith::CmpIPredicate::sle,
                                          rowEnd, mUpper));
        }
        llvm::SmallVector<mlir::Type, 4> accTypes(initialAccs.size(), tileType);
        auto ifOp = mlir::scf::IfOp::create(rewriter, loc, accTypes, whole,
                                            /*withElseRegion=*/true);
        {
          mlir::OpBuilder::InsertionGuard g(rewriter);
          rewriter.setInsertionPointToStart(ifOp.thenBlock());
          auto loop = buildKLoop(rewriter, /*multiVector=*/true);
          mlir::scf::YieldOp::create(rewriter, loc, loop.getResults());
        }
        {
          mlir::OpBuilder::InsertionGuard g(rewriter);
          rewriter.setInsertionPointToStart(ifOp.elseBlock());
          auto loop = buildKLoop(rewriter, /*multiVector=*/false);
          mlir::scf::YieldOp::create(rewriter, loc, loop.getResults());
        }
        accResults.assign(ifOp.getResults().begin(), ifOp.getResults().end());
      } else {
        auto loop = buildKLoop(rewriter, /*multiVector=*/false);
        accResults.assign(loop.getResults().begin(), loop.getResults().end());
      }

      for (auto [rowIndex, row] : llvm::enumerate(rows)) {
        for (auto [colIndex, col] : llvm::enumerate(cols)) {
          std::size_t tile = rowIndex * cols.size() + colIndex;
          mlir::arm_sme::TileStoreOp::create(
              rewriter, loc, accResults[tile], views.out,
              mlir::ValueRange{row, col}, tileMasks[tile],
              mlir::arm_sme::TileSliceLayout::Horizontal);
        }
      }
    }
  }
  }

  // Free the panels here, not via the deallocation pass. The kernel reads them
  // through raw pointers (ExtractAlignedPointerAsIndexOp feeding intrinsics),
  // and CSE folds those extracts into one right after the alloc -- at which
  // point the buffer's last visible memref use IS that extract, and the
  // deallocation pass frees it before a single byte is packed. Single-threaded
  // that was silent (nothing reused the freed pages between free and use);
  // under dispatch another worker's malloc reuses them mid-kernel, which showed
  // up as whole M-chunks of C computed from the neighbour's RHS values. An
  // explicit dealloc is a memref use at the right point, and the deallocation
  // pass skips allocs that already have one.
  if (packOperands) {
    if (!lhsPrePacked)
      mlir::memref::DeallocOp::create(rewriter, loc, panelLhs);
    if (!rhsPrePacked)
      mlir::memref::DeallocOp::create(rewriter, loc, panelRhs);
  }

  rewriter.eraseOp(matmul);
  return mlir::success();
}

// Whether the SME kernel will take this contraction once it reaches memref.
// Kept next to matchStaticSMEMatmul on purpose: if the two drift apart, a
// transposed matmul would find no kernel and fall all the way to scalar loops.
bool smeWillLowerTensorMatmul(mlir::linalg::MatmulOp matmul,
                              const py::TensorLoweringTarget &target) {
  if (!target.usesArmSME() || matmul.getNumDpsInputs() != 2 ||
      matmul.getNumDpsInits() != 1 || !hasDefaultMatmulMaps(matmul))
    return false;
  auto lhsType = mlir::dyn_cast<mlir::RankedTensorType>(
      matmul.getDpsInputOperand(0)->get().getType());
  auto rhsType = mlir::dyn_cast<mlir::RankedTensorType>(
      matmul.getDpsInputOperand(1)->get().getType());
  auto outType = mlir::dyn_cast<mlir::RankedTensorType>(
      matmul.getDpsInitOperand(0)->get().getType());
  if (!lhsType || !rhsType || !outType || lhsType.getRank() != 2 ||
      rhsType.getRank() != 2 || outType.getRank() != 2 ||
      !lhsType.hasStaticShape() || !rhsType.hasStaticShape() ||
      !outType.hasStaticShape())
    return false;
  mlir::Type elementType = outType.getElementType();
  if (lhsType.getElementType() != elementType ||
      rhsType.getElementType() != elementType)
    return false;
  if (!smeElementLayout(elementType, target))
    return false;

  int64_t m = outType.getDimSize(0);
  int64_t n = outType.getDimSize(1);
  int64_t k = lhsType.getDimSize(1);
  if (m <= 0 || n <= 0 || k <= 0 || lhsType.getDimSize(0) != m ||
      rhsType.getDimSize(0) != k || rhsType.getDimSize(1) != n)
    return false;
  std::uint64_t work = static_cast<std::uint64_t>(m) *
                       static_cast<std::uint64_t>(n) *
                       static_cast<std::uint64_t>(k);
  return work >= kSMEMatmulMinWork;
}

mlir::ArrayAttr transposedLhsMatmulMaps(mlir::OpBuilder &builder) {
  mlir::MLIRContext *context = builder.getContext();
  mlir::AffineExpr m, n, k;
  mlir::bindDims(context, m, n, k);
  return builder.getArrayAttr(
      {mlir::AffineMapAttr::get(mlir::AffineMap::get(3, 0, {k, m}, context)),
       mlir::AffineMapAttr::get(mlir::AffineMap::get(3, 0, {k, n}, context)),
       mlir::AffineMapAttr::get(mlir::AffineMap::get(3, 0, {m, n}, context))});
}

// How deep a K block the SME kernel wants, in bytes of one reduction step's
// slice of a single row (so it scales with the element type rather than being an
// f32 element count).
//
// Measured on M4 Max f32 single-threaded over a 120-point sweep of K in
// {1024..2560} x N in {256..2048} x every KC dividing K, scored as regret
// against each shape's own best KC. The optimum is a wide plateau -- every
// target from 480 to 928 elements holds worst-case regret above 0.88 and mean
// above 0.95 -- so the exact value matters far less than being on it. 640 sits
// where the worst-case (0.913) and mean (0.973) plateaus overlap.
//
// This is a property of one machine's memory hierarchy, not of the contraction:
// it removes no arithmetic, and nothing here would transfer to another core. It
// lives on the Arm side for that reason, next to the kernel whose streams it
// bounds.
constexpr int64_t kSMEReductionBlockBytes = 640 * 4;

// Cut the reduction of a matmul the SME kernel would otherwise take whole.
//
// The kernel sweeps the entire reduction per output block, so both operands
// stream past at their parent row pitch for K steps before anything is reused.
// Past some depth that stops fitting whatever the hardware was keeping it in,
// and the contraction falls off a cliff -- measured on M4 Max f32
// single-threaded, K=2048 runs 675-718 GFLOP/s unblocked against 1526-1841 for
// a blocked K, across every N from 256 to 2048. Cutting K bounds the stream.
//
// What sets the depth is only measured, not explained. Several plausible models
// -- B's block staying in L2, the per-step slices fitting L1, A's block staying
// resident -- each contradicted the sweep.
//
// Why not derive the depth from N, which is what B's block size depends on: that
// reads the reduction's depth off the wrong extent. It leaves K unblocked
// whenever B happens to be small, which is exactly what a narrow N does, and
// measured 0.369 regret at K=2048, N<=1024. Only the reduction's own extent
// decides this.
//
// Why not cut N instead: B is [K][N], so a K block is whole rows -- a contiguous
// slab of exactly the bytes it spans. An N block is a column strip of the same
// byte count scattered across B's full extent at the parent row pitch, which on
// a power-of-two pitch collides in the cache sets it lands on. Measured at
// 2048^3, cutting N alone ran 554 GFLOP/s against 576 for no blocking at all.
//
// Why not cut N *as well*, which that does not rule out: measured, it is worth
// between nothing and a little -- 1380 -> 1393 at 2048^3, 1554 -> 1568 at
// K=2048/N=1024, 1774 -> 1834 at 1536^3. All of it sits inside the ~6% this
// machine drifts between runs, and none of it pays for the extra loop.
class MatmulSMEReductionBlockPass
    : public mlir::PassWrapper<MatmulSMEReductionBlockPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MatmulSMEReductionBlockPass)

  explicit MatmulSMEReductionBlockPass(py::TensorLoweringTarget target = {})
      : target(target) {}

  llvm::StringRef getArgument() const final {
    return "lython-matmul-sme-reduction-block";
  }
  llvm::StringRef getDescription() const final {
    return "cut the reduction of SME-bound matmuls to bound their operand "
           "streams";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<mlir::linalg::LinalgDialect, mlir::scf::SCFDialect,
                    mlir::memref::MemRefDialect, mlir::tensor::TensorDialect>();
  }

  void runOnOperation() final {
    if (!target.usesArmSME())
      return;
    llvm::SmallVector<std::pair<mlir::linalg::MatmulOp, int64_t>, 4> blocked;
    getOperation().walk([&](mlir::linalg::MatmulOp matmul) {
      if (std::optional<int64_t> kc = selectReductionBlock(matmul))
        blocked.push_back({matmul, *kc});
    });

    mlir::IRRewriter rewriter(&getContext());
    for (auto [matmul, kc] : blocked) {
      if (!matmul->getBlock())
        continue;
      if (mlir::failed(tileMatmulReduction(matmul, kc, rewriter))) {
        signalPassFailure();
        return;
      }
    }
  }

private:
  // Why only a transposed LHS: it is the mark MatmulLhsTransposePass leaves on
  // the matmuls this kernel will take whole. A default-layout matmul reaches
  // MatmulTilingPass with a macro tile of its own, and blocking it here would
  // nest one block inside another.
  //
  // Nothing here is a run-time branch: the extents are static type parameters,
  // so the divisor search runs once at compile time. The library this competes
  // with has to re-derive its block per call from arguments it cannot see
  // coming.
  std::optional<int64_t> selectReductionBlock(mlir::linalg::MatmulOp matmul) {
    if (!hasTransposedLhsMatmulMaps(matmul))
      return std::nullopt;
    // Safety net, not policy: the pipeline runs this pass before the rhs pack,
    // so a marked matmul here means the order was broken. Tiling one would
    // subview the disguised panel buffer, which addresses garbage.
    if (matmul->hasAttr(kSMERhsPanelsAttr) ||
        matmul->hasAttr(kSMELhsPanelsAttr))
      return std::nullopt;
    auto rhsType = mlir::dyn_cast<mlir::ShapedType>(
        matmul.getDpsInputOperand(1)->get().getType());
    if (!rhsType || !rhsType.hasStaticShape() || rhsType.getRank() != 2)
      return std::nullopt;
    int64_t k = rhsType.getDimSize(0);
    unsigned width = rhsType.getElementType().getIntOrFloatBitWidth() / 8;
    if (k <= 0 || width == 0)
      return std::nullopt;

    int64_t target = kSMEReductionBlockBytes / static_cast<int64_t>(width);
    if (target <= 0)
      return std::nullopt;
    int64_t kc = divisorNearest(k, target);
    if (kc >= k || kc <= 0)
      return std::nullopt;
    return kc;
  }

  py::TensorLoweringTarget target;
};

// Pack B's panels once, before the parallel chunker multiplies the matmul.
//
// The kernel wants both operands panel-major (see createPanelPack), but by the
// time it runs, the matmul has been cut into M chunks and outlined: packing
// inline re-copied the whole of B once per chunk, and measured as a regression
// -- 1T f32 512^3 ran 1134 GFLOP/s against 1620 unpacked, all of it redundant
// copy traffic. The chunks partition A and C but share B, so B's panels belong
// to the caller, made once per contraction. A stays packed inline: its chunks
// are disjoint, so there is nothing to hoist.
//
// The buffer rides into the kernel as the rhs operand itself, reinterpret_cast
// back to [K][N]. That type is a lie about the layout, held together by
// kSMERhsPanelsAttr: the chunker passes B through whole, the reduction blocker
// is taught to skip, and the SME lowering refuses loudly if it ever cannot
// take the marked matmul, because any other consumer would read panels as rows
// and silently mis-execute.
class MatmulSMERhsPackPass
    : public mlir::PassWrapper<MatmulSMERhsPackPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MatmulSMERhsPackPass)

  explicit MatmulSMERhsPackPass(py::TensorLoweringTarget target = {})
      : target(target) {}

  llvm::StringRef getArgument() const final {
    return "lython-matmul-sme-rhs-pack";
  }
  llvm::StringRef getDescription() const final {
    return "pack the rhs of SME-bound matmuls into panels once, ahead of the "
           "parallel chunker";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<mlir::arith::ArithDialect, mlir::LLVM::LLVMDialect,
                    mlir::memref::MemRefDialect, mlir::scf::SCFDialect,
                    mlir::vector::VectorDialect>();
  }

  void runOnOperation() final {
    if (!target.usesArmSME2())
      return;
    // Automatic: packs when B is big enough to win (kSMEPanelMinRhsBytes).
    // LY_SME_PACK=1 forces it everywhere, =0 disables it -- measurement
    // overrides, not configuration.
    const char *mode = getenv("LY_SME_PACK");
    if (mode && mode[0] == '0')
      return;
    bool force = mode && mode[0] == '1';
    llvm::SmallVector<mlir::linalg::MatmulOp, 8> targets;
    getOperation().walk([&](mlir::linalg::MatmulOp matmul) {
      std::optional<StaticMatmulMemRefs> refs =
          matchStaticSMEMatmul(matmul, target);
      // Mirror of the kernel's packOperands conditions: anything marked here
      // that the kernel then refuses is a hard error over there.
      if (!refs || !isProfitableStaticSMEMatmul(*refs) ||
          !refs->lhsTransposed || !refs->outType.getElementType().isF32())
        return;
      if (kSMETileGrid.rows != 2 || kSMETileGrid.cols != 2)
        return;
      auto contiguous = [](mlir::MemRefType type) {
        std::optional<int64_t> rowStride = staticStride(type, 0);
        std::optional<int64_t> innerStride = staticStride(type, 1);
        return rowStride && innerStride && *innerStride == 1;
      };
      if (!contiguous(refs->lhsType) || !contiguous(refs->rhsType))
        return;
      std::uint64_t rhsBytes = static_cast<std::uint64_t>(refs->k) *
                               static_cast<std::uint64_t>(refs->n) * 4;
      if (!force && rhsBytes < kSMEPanelMinRhsBytes)
        return;
      targets.push_back(matmul);
    });

    mlir::OpBuilder builder(&getContext());
    for (mlir::linalg::MatmulOp matmul : targets) {
      std::optional<StaticMatmulMemRefs> refs =
          matchStaticSMEMatmul(matmul, target);
      mlir::Location loc = matmul.getLoc();
      builder.setInsertionPoint(matmul);
      mlir::Value lanes = createStreamingLaneCount(builder, loc);
      mlir::Value width = mlir::arith::MulIOp::create(
          builder, loc, lanes,
          createIndexConstant(builder, loc, kSMETileGrid.cols));
      mlir::Value packed = createRhsPanels(builder, loc, refs->rhs, refs->k,
                                           refs->n, width);
      auto lieType = mlir::MemRefType::get({refs->k, refs->n},
                                           refs->rhsType.getElementType());
      mlir::Value lie = mlir::memref::ReinterpretCastOp::create(
          builder, loc, lieType, packed, /*offset=*/0,
          llvm::ArrayRef<int64_t>{refs->k, refs->n},
          llvm::ArrayRef<int64_t>{refs->n, 1});
      matmul.getDpsInputOperand(1)->set(lie);
      matmul->setAttr(kSMERhsPanelsAttr, builder.getUnitAttr());
      builder.setInsertionPointAfter(matmul);
      mlir::memref::DeallocOp::create(builder, loc, packed);

      // A's panels, hoisted. B changes per call, so its pack just ran here;
      // A usually does not -- in an `x = A @ x` chain it never does -- so its
      // panels are materialised right after the transpose that defines A_T,
      // which MatmulLhsTransposePass already anchored at A's definition,
      // outside any loop that merely re-reads it. Accelerate re-packs A on
      // every sgemm call (measured 4.5% of a 2000^3 run); a compiler that can
      // see the loop pays once.
      //
      // The layout is chunk-relative -- chunk c's rows start a fresh panel
      // group -- because the parallel chunker's equal-height cuts need not be
      // multiples of the runtime panel width, and a panel straddling a chunk
      // boundary could not be read contiguously by either side. The kernel
      // recovers c from its output subview's row offset.
      //
      // No dealloc on purpose: the panels outlive every reader of A_T, and
      // the deallocation pass frees function-exit-lived buffers itself.
      packLhsPanels(builder, matmul, *refs, width);
    }
  }

private:
  // Emits the hoisted pack when A_T's defining write is visible; a matmul
  // whose lhs has no traceable transpose keeps the kernel's inline pack.
  void packLhsPanels(mlir::OpBuilder &builder, mlir::linalg::MatmulOp matmul,
                     const StaticMatmulMemRefs &refs, mlir::Value width) {
    mlir::Value root = refs.lhs;
    while (auto cast = root.getDefiningOp<mlir::memref::CastOp>())
      root = cast.getSource();
    mlir::linalg::TransposeOp writer;
    for (mlir::Operation *user : root.getUsers()) {
      auto transpose = mlir::dyn_cast<mlir::linalg::TransposeOp>(user);
      if (transpose && transpose.getDpsInitOperand(0)->get() == root) {
        writer = transpose;
        break;
      }
    }
    if (!writer)
      return;

    int64_t chunks =
        selectSMEMatmulChunkCount(refs.m, refs.n, refs.k).value_or(1);
    int64_t rows = refs.m / chunks;
    mlir::Location loc = writer.getLoc();
    builder.setInsertionPointAfter(writer);
    // The pass entry point computed `width` next to the matmul; the pack
    // lives at A_T's definition, so it derives its own.
    mlir::Value lanes = createStreamingLaneCount(builder, loc);
    mlir::Value panelWidth = mlir::arith::MulIOp::create(
        builder, loc, lanes,
        createIndexConstant(builder, loc, kSMETileGrid.cols));
    mlir::Value rowsValue = createIndexConstant(builder, loc, rows);
    mlir::Value kValue = createIndexConstant(builder, loc, refs.k);
    mlir::Value chunksValue = createIndexConstant(builder, loc, chunks);
    mlir::Value panelsPerChunk = mlir::arith::CeilDivSIOp::create(
        builder, loc, rowsValue, panelWidth);
    mlir::Value chunkStride = mlir::arith::MulIOp::create(
        builder, loc,
        mlir::arith::MulIOp::create(builder, loc, panelsPerChunk, kValue),
        panelWidth);
    mlir::Value total =
        mlir::arith::MulIOp::create(builder, loc, chunksValue, chunkStride);
    auto packedType = mlir::MemRefType::get({mlir::ShapedType::kDynamic},
                                            builder.getF32Type());
    mlir::Value panels = mlir::memref::AllocOp::create(
        builder, loc, packedType, mlir::ValueRange{total});

    mlir::Value zero = createIndexConstant(builder, loc, 0);
    mlir::Value one = createIndexConstant(builder, loc, 1);
    auto chunkLoop =
        mlir::scf::ForOp::create(builder, loc, zero, chunksValue, one);
    {
      mlir::OpBuilder::InsertionGuard g(builder);
      builder.setInsertionPointToStart(chunkLoop.getBody());
      mlir::Value c = chunkLoop.getInductionVar();
      mlir::Value rowStart =
          mlir::arith::MulIOp::create(builder, loc, c, rowsValue);
      mlir::Value destBase =
          mlir::arith::MulIOp::create(builder, loc, c, chunkStride);
      mlir::Value source = mlir::memref::SubViewOp::create(
          builder, loc, root,
          llvm::ArrayRef<mlir::OpFoldResult>{builder.getIndexAttr(0),
                                             mlir::OpFoldResult(rowStart)},
          llvm::ArrayRef<mlir::OpFoldResult>{builder.getIndexAttr(refs.k),
                                             builder.getIndexAttr(rows)},
          llvm::ArrayRef<mlir::OpFoldResult>{builder.getIndexAttr(1),
                                             builder.getIndexAttr(1)});
      createPanelBufferInto(builder, loc, panels, destBase, source, refs.k,
                            rows, panelWidth);
    }

    builder.setInsertionPoint(matmul);
    auto lieType = mlir::MemRefType::get({refs.k, refs.m},
                                         refs.lhsType.getElementType());
    mlir::Value lie = mlir::memref::ReinterpretCastOp::create(
        builder, matmul.getLoc(), lieType, panels, /*offset=*/0,
        llvm::ArrayRef<int64_t>{refs.k, refs.m},
        llvm::ArrayRef<int64_t>{refs.m, 1});
    matmul.getDpsInputOperand(0)->set(lie);
    matmul->setAttr(kSMELhsPanelsAttr, builder.getUnitAttr());
  }

  py::TensorLoweringTarget target;
};

class MatmulLhsTransposePass
    : public mlir::PassWrapper<MatmulLhsTransposePass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MatmulLhsTransposePass)

  explicit MatmulLhsTransposePass(py::TensorLoweringTarget target = {})
      : target(target) {}

  llvm::StringRef getArgument() const final {
    return "lython-matmul-lhs-transpose";
  }
  llvm::StringRef getDescription() const final {
    return "hoist the SME kernel's LHS transpose out of the contraction";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect>();
  }

  void runOnOperation() final {
    if (!target.usesArmSME())
      return;
    llvm::SmallVector<mlir::linalg::MatmulOp, 8> targets;
    getOperation().walk([&](mlir::linalg::MatmulOp matmul) {
      if (smeWillLowerTensorMatmul(matmul, target))
        targets.push_back(matmul);
    });

    mlir::OpBuilder builder(&getContext());
    for (mlir::linalg::MatmulOp matmul : targets) {
      mlir::Value lhs = matmul.getDpsInputOperand(0)->get();
      auto lhsType = mlir::cast<mlir::RankedTensorType>(lhs.getType());
      mlir::Location loc = matmul.getLoc();

      // Anchor at A's definition. A tensor is a value: once defined it cannot
      // be written to, so this position is always safe and is as far out as
      // the transpose can go. When A is loop-carried its definition is the
      // block argument, and the transpose correctly stays in the loop.
      mlir::OpBuilder::InsertionGuard guard(builder);
      if (mlir::Operation *def = lhs.getDefiningOp())
        builder.setInsertionPointAfter(def);
      else
        builder.setInsertionPointToStart(lhs.getParentBlock());

      mlir::Value empty = mlir::tensor::EmptyOp::create(
          builder, loc,
          llvm::ArrayRef<int64_t>{lhsType.getDimSize(1), lhsType.getDimSize(0)},
          lhsType.getElementType());
      mlir::Value transposed =
          mlir::linalg::TransposeOp::create(builder, loc, lhs, empty,
                                            llvm::ArrayRef<int64_t>{1, 0})
              .getResult()[0];

      matmul.getDpsInputOperand(0)->set(transposed);
      matmul.setIndexingMapsAttr(transposedLhsMatmulMaps(builder));
    }
  }

private:
  py::TensorLoweringTarget target;
};

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
    bool poisoned = false;
    getOperation().walk([&](mlir::linalg::MatmulOp matmul) {
      std::optional<StaticMatmulMemRefs> refs =
          matchStaticSMEMatmul(matmul, target);
      if (refs && isProfitableStaticSMEMatmul(*refs)) {
        matmuls.push_back(matmul);
      } else if (matmul->hasAttr(kSMERhsPanelsAttr)) {
        // The rhs operand's type lies about a panel buffer. Nothing but this
        // kernel can read it, so falling through to the generic path would
        // silently mis-execute -- refuse instead.
        matmul.emitError()
            << "matmul carries a panel-packed rhs but the SME kernel cannot "
               "take it";
        poisoned = true;
      }
    });
    if (poisoned) {
      signalPassFailure();
      return;
    }

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
createMatmulSMEReductionBlockPass(py::TensorLoweringTarget target) {
  return std::make_unique<MatmulSMEReductionBlockPass>(target);
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createMatmulSMERhsPackPass(py::TensorLoweringTarget target) {
  return std::make_unique<MatmulSMERhsPackPass>(target);
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createMatmulLhsTransposePass(py::TensorLoweringTarget target) {
  return std::make_unique<MatmulLhsTransposePass>(target);
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createMatmulSMELoweringPass(py::TensorLoweringTarget target) {
  return std::make_unique<MatmulSMELoweringPass>(target);
}

} // namespace py::lowering::arch::arm
