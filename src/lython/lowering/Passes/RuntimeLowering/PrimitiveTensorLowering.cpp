#include "Arch/Arm/PrimitiveTensorArmSME.h"
#include "Arch/X86/PrimitiveTensorX86.h"
#include "Common/RuntimeSupport.h"
#include "PrimitiveTensorGemm.h"
#include "PrimitiveTensorSupport.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/STLExtras.h"

#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <utility>

namespace py::runtime_lowering {
namespace {

// Keep reductions at a width that lowers to efficient target vectors without
// over-extending the reduction tree. Empirically, 256-bit reductions beat the
// previous 512-bit width on the current matmul/sum and 3-D elementwise benches.
constexpr int64_t kReductionVectorBits = 256;

bool isRankPreservingUnitStrideSubview(mlir::memref::SubViewOp subview) {
  mlir::MemRefType sourceType = subview.getSourceType();
  mlir::MemRefType resultType = subview.getType();
  if (sourceType.getRank() != resultType.getRank())
    return false;

  for (int64_t stride : subview.getStaticStrides()) {
    if (stride != 1)
      return false;
  }
  return true;
}

struct ViewAccess {
  mlir::Value source;
  llvm::SmallVector<mlir::Value, 4> sourceIndices;
  mlir::AffineMap permutationMap;
};

bool accessRankMatchesMemRef(mlir::MemRefType memrefType,
                             mlir::ValueRange accessIndices) {
  return accessIndices.size() == static_cast<std::size_t>(memrefType.getRank());
}

llvm::SmallVector<mlir::Value, 4>
addAccessIndices(mlir::OpBuilder &builder, mlir::Location loc,
                 llvm::SmallVector<mlir::Value, 4> baseIndices,
                 mlir::ValueRange accessIndices) {
  for (auto [index, accessIndex] : llvm::enumerate(accessIndices)) {
    baseIndices[index] = builder.create<mlir::arith::AddIOp>(
        loc, baseIndices[index], accessIndex);
  }
  return baseIndices;
}

llvm::SmallVector<int64_t, 4> rowMajorStrides(mlir::MemRefType type) {
  llvm::SmallVector<int64_t, 4> strides(type.getRank(), 1);
  int64_t stride = 1;
  for (int64_t dim = type.getRank() - 1; dim >= 0; --dim) {
    strides[dim] = stride;
    stride *= type.getDimSize(dim);
  }
  return strides;
}

std::optional<llvm::SmallVector<int64_t, 4>>
staticRowMajorSourceStrides(mlir::MemRefType sourceType) {
  if (!sourceType.hasStaticShape())
    return std::nullopt;

  llvm::SmallVector<int64_t, 4> sourceStrides;
  int64_t sourceOffset = 0;
  if (mlir::failed(sourceType.getStridesAndOffset(sourceStrides, sourceOffset)))
    return std::nullopt;
  if (sourceOffset != 0)
    return std::nullopt;

  llvm::SmallVector<int64_t, 4> expectedStrides = rowMajorStrides(sourceType);
  if (!llvm::equal(sourceStrides, expectedStrides))
    return std::nullopt;
  return sourceStrides;
}

std::optional<ViewAccess> buildSubviewView(mlir::memref::SubViewOp subview,
                                           mlir::ValueRange accessIndices,
                                           mlir::AffineMap permutationMap,
                                           mlir::OpBuilder &builder) {
  if (!isRankPreservingUnitStrideSubview(subview) ||
      !accessRankMatchesMemRef(subview.getType(), accessIndices))
    return std::nullopt;

  llvm::ArrayRef<int64_t> staticOffsets = subview.getStaticOffsets();
  mlir::Operation::operand_range dynamicOffsets = subview.getOffsets();
  auto dynamicOffset = dynamicOffsets.begin();

  llvm::SmallVector<mlir::Value, 4> baseIndices;
  baseIndices.reserve(accessIndices.size());
  for (std::size_t index = 0; index < accessIndices.size(); ++index) {
    int64_t staticOffset = staticOffsets[index];
    if (mlir::ShapedType::isDynamic(staticOffset)) {
      baseIndices.push_back(*dynamicOffset);
      ++dynamicOffset;
    } else {
      baseIndices.push_back(builder.create<mlir::arith::ConstantIndexOp>(
          subview.getLoc(), staticOffset));
    }
  }
  llvm::SmallVector<mlir::Value, 4> sourceIndices = addAccessIndices(
      builder, subview.getLoc(), std::move(baseIndices), accessIndices);
  return ViewAccess{subview.getSource(), std::move(sourceIndices),
                    permutationMap};
}

std::optional<mlir::Value>
buildReinterpretCastLinearOffset(mlir::memref::ReinterpretCastOp cast,
                                 mlir::OpBuilder &builder) {
  llvm::ArrayRef<int64_t> staticOffsets = cast.getStaticOffsets();
  if (staticOffsets.size() != 1)
    return std::nullopt;

  int64_t staticOffset = staticOffsets.front();
  if (mlir::ShapedType::isDynamic(staticOffset)) {
    if (cast.getOffsets().size() != 1)
      return std::nullopt;
    return *cast.getOffsets().begin();
  }

  return builder
      .create<mlir::arith::ConstantIndexOp>(cast.getLoc(), staticOffset)
      .getResult();
}

std::optional<int64_t> constantIndexValue(mlir::Value value) {
  mlir::IntegerAttr integer;
  if (mlir::matchPattern(value, mlir::m_Constant(&integer)))
    return integer.getInt();
  return std::nullopt;
}

struct StaticIndexRange {
  int64_t min;
  int64_t max;
};

std::optional<int64_t> checkedAdd(int64_t lhs, int64_t rhs) {
  if ((rhs > 0 && lhs > std::numeric_limits<int64_t>::max() - rhs) ||
      (rhs < 0 && lhs < std::numeric_limits<int64_t>::min() - rhs))
    return std::nullopt;
  return lhs + rhs;
}

std::optional<StaticIndexRange> addRanges(StaticIndexRange lhs,
                                          StaticIndexRange rhs) {
  std::optional<int64_t> min = checkedAdd(lhs.min, rhs.min);
  std::optional<int64_t> max = checkedAdd(lhs.max, rhs.max);
  if (!min || !max)
    return std::nullopt;
  return StaticIndexRange{*min, *max};
}

std::optional<StaticIndexRange> staticIndexRange(mlir::Value value) {
  if (std::optional<int64_t> constant = constantIndexValue(value))
    return StaticIndexRange{*constant, *constant};

  if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    auto loop =
        mlir::dyn_cast_or_null<mlir::scf::ForOp>(arg.getOwner()->getParentOp());
    if (!loop || loop.getInductionVar() != value)
      return std::nullopt;

    std::optional<int64_t> lower = constantIndexValue(loop.getLowerBound());
    std::optional<int64_t> upper = constantIndexValue(loop.getUpperBound());
    std::optional<int64_t> step = constantIndexValue(loop.getStep());
    if (!lower || !upper || !step || *step <= 0)
      return std::nullopt;
    if (*upper <= *lower)
      return StaticIndexRange{*lower, *lower};
    return StaticIndexRange{*lower, *upper - *step};
  }

  if (auto add = value.getDefiningOp<mlir::arith::AddIOp>()) {
    std::optional<StaticIndexRange> lhs = staticIndexRange(add.getLhs());
    std::optional<StaticIndexRange> rhs = staticIndexRange(add.getRhs());
    if (!lhs || !rhs)
      return std::nullopt;
    return addRanges(*lhs, *rhs);
  }

  return std::nullopt;
}

bool matchMulByConstant(mlir::Value value, int64_t expected,
                        mlir::Value &other) {
  auto mul = value.getDefiningOp<mlir::arith::MulIOp>();
  if (!mul)
    return false;

  if (std::optional<int64_t> lhs = constantIndexValue(mul.getLhs());
      lhs && *lhs == expected) {
    other = mul.getRhs();
    return true;
  }
  if (std::optional<int64_t> rhs = constantIndexValue(mul.getRhs());
      rhs && *rhs == expected) {
    other = mul.getLhs();
    return true;
  }
  return false;
}

std::optional<std::pair<mlir::Value, mlir::Value>>
decomposeRank2RowMajorOffset(mlir::Value linearOffset, int64_t rowStride,
                             mlir::OpBuilder &builder, mlir::Location loc) {
  if (rowStride == 1)
    return std::make_pair(
        linearOffset,
        builder.create<mlir::arith::ConstantIndexOp>(loc, 0).getResult());

  mlir::Value row;
  if (matchMulByConstant(linearOffset, rowStride, row)) {
    mlir::Value zero =
        builder.create<mlir::arith::ConstantIndexOp>(loc, 0).getResult();
    return std::make_pair(row, zero);
  }

  auto add = linearOffset.getDefiningOp<mlir::arith::AddIOp>();
  if (!add)
    return std::nullopt;

  if (matchMulByConstant(add.getLhs(), rowStride, row))
    return std::make_pair(row, add.getRhs());
  if (matchMulByConstant(add.getRhs(), rowStride, row))
    return std::make_pair(row, add.getLhs());

  return std::nullopt;
}

bool isKnownNonNegativeAndLessThan(mlir::Value value, int64_t upperBound) {
  if (upperBound <= 0)
    return false;

  std::optional<StaticIndexRange> range = staticIndexRange(value);
  return range && range->min >= 0 && range->max < upperBound;
}

bool foldRowMajorDiv(mlir::arith::DivUIOp div, mlir::IRRewriter &rewriter) {
  std::optional<int64_t> stride = constantIndexValue(div.getRhs());
  if (!stride)
    return false;

  std::optional<std::pair<mlir::Value, mlir::Value>> indices =
      decomposeRank2RowMajorOffset(div.getLhs(), *stride, rewriter,
                                   div.getLoc());
  if (!indices || !isKnownNonNegativeAndLessThan(indices->second, *stride))
    return false;

  rewriter.replaceOp(div, indices->first);
  return true;
}

bool foldRowMajorRem(mlir::arith::RemUIOp rem, mlir::IRRewriter &rewriter) {
  std::optional<int64_t> stride = constantIndexValue(rem.getRhs());
  if (!stride)
    return false;

  std::optional<std::pair<mlir::Value, mlir::Value>> indices =
      decomposeRank2RowMajorOffset(rem.getLhs(), *stride, rewriter,
                                   rem.getLoc());
  if (!indices || !isKnownNonNegativeAndLessThan(indices->second, *stride))
    return false;

  rewriter.replaceOp(rem, indices->second);
  return true;
}

void foldRowMajorDelinearization(mlir::ModuleOp module,
                                 mlir::IRRewriter &rewriter) {
  llvm::SmallVector<mlir::Operation *, 16> ops;
  module.walk([&](mlir::Operation *op) {
    if (mlir::isa<mlir::arith::DivUIOp, mlir::arith::RemUIOp>(op))
      ops.push_back(op);
  });

  for (mlir::Operation *op : ops) {
    if (!op->getBlock())
      continue;
    rewriter.setInsertionPoint(op);
    if (auto div = mlir::dyn_cast<mlir::arith::DivUIOp>(op)) {
      foldRowMajorDiv(div, rewriter);
      continue;
    }
    foldRowMajorRem(mlir::cast<mlir::arith::RemUIOp>(op), rewriter);
  }
}

llvm::SmallVector<mlir::Value, 4>
delinearizeRowMajorOffset(mlir::OpBuilder &builder, mlir::Location loc,
                          mlir::Value linearOffset,
                          llvm::ArrayRef<int64_t> strides) {
  if (strides.size() == 2) {
    if (std::optional<std::pair<mlir::Value, mlir::Value>> indices =
            decomposeRank2RowMajorOffset(linearOffset, strides.front(), builder,
                                         loc)) {
      return llvm::SmallVector<mlir::Value, 4>{indices->first, indices->second};
    }
  }

  llvm::SmallVector<mlir::Value, 4> indices;
  indices.reserve(strides.size());

  mlir::Value remaining = linearOffset;
  for (std::size_t dim = 0; dim < strides.size(); ++dim) {
    int64_t stride = strides[dim];
    if (dim + 1 == strides.size()) {
      indices.push_back(remaining);
      break;
    }

    mlir::Value strideValue =
        builder.create<mlir::arith::ConstantIndexOp>(loc, stride);
    indices.push_back(
        builder.create<mlir::arith::DivUIOp>(loc, remaining, strideValue));
    remaining =
        builder.create<mlir::arith::RemUIOp>(loc, remaining, strideValue);
  }
  return indices;
}

std::optional<llvm::SmallVector<unsigned, 4>>
resultToSourceDimPermutation(mlir::memref::ReinterpretCastOp cast,
                             mlir::MemRefType sourceType) {
  mlir::MemRefType resultType = cast.getType();
  if (sourceType.getRank() != resultType.getRank() ||
      !sourceType.hasStaticShape() || !resultType.hasStaticShape())
    return std::nullopt;

  std::optional<llvm::SmallVector<int64_t, 4>> sourceStrides =
      staticRowMajorSourceStrides(sourceType);
  if (!sourceStrides)
    return std::nullopt;

  llvm::ArrayRef<int64_t> resultStrides = cast.getStaticStrides();
  if (resultStrides.size() != static_cast<std::size_t>(resultType.getRank()))
    return std::nullopt;

  llvm::SmallVector<unsigned, 4> resultToSource(resultType.getRank());
  llvm::SmallVector<bool, 4> usedSourceDims(sourceType.getRank(), false);
  for (auto [resultDim, stride] : llvm::enumerate(resultStrides)) {
    if (mlir::ShapedType::isDynamic(stride))
      return std::nullopt;

    std::optional<unsigned> sourceDim;
    for (auto [candidateDim, sourceStride] : llvm::enumerate(*sourceStrides)) {
      if (!usedSourceDims[candidateDim] && sourceStride == stride) {
        sourceDim = candidateDim;
        break;
      }
    }
    if (!sourceDim)
      return std::nullopt;
    resultToSource[resultDim] = *sourceDim;
    usedSourceDims[*sourceDim] = true;
  }
  return resultToSource;
}

std::optional<mlir::AffineMap>
remapTransferPermutationMap(mlir::AffineMap map,
                            llvm::ArrayRef<unsigned> resultToSource,
                            unsigned sourceRank) {
  if (!map)
    return mlir::AffineMap();

  llvm::SmallVector<mlir::AffineExpr, 4> sourceExprs;
  sourceExprs.reserve(map.getNumResults());
  for (mlir::AffineExpr expr : map.getResults()) {
    auto dim = mlir::dyn_cast<mlir::AffineDimExpr>(expr);
    if (!dim || dim.getPosition() >= resultToSource.size())
      return std::nullopt;
    sourceExprs.push_back(mlir::getAffineDimExpr(
        resultToSource[dim.getPosition()], map.getContext()));
  }
  return mlir::AffineMap::get(sourceRank, 0, sourceExprs, map.getContext());
}

std::optional<ViewAccess>
buildPermutationReinterpretCastView(mlir::memref::ReinterpretCastOp cast,
                                    mlir::ValueRange accessIndices,
                                    mlir::AffineMap permutationMap) {
  llvm::ArrayRef<int64_t> staticOffsets = cast.getStaticOffsets();
  if (staticOffsets.size() != 1 || staticOffsets.front() != 0)
    return std::nullopt;

  auto sourceType =
      mlir::dyn_cast<mlir::MemRefType>(cast.getSource().getType());
  mlir::MemRefType resultType = cast.getType();
  if (!sourceType || !accessRankMatchesMemRef(resultType, accessIndices))
    return std::nullopt;

  std::optional<llvm::SmallVector<unsigned, 4>> resultToSource =
      resultToSourceDimPermutation(cast, sourceType);
  if (!resultToSource)
    return std::nullopt;

  llvm::SmallVector<mlir::Value, 4> sourceIndices(sourceType.getRank());
  for (auto [resultDim, accessIndex] : llvm::enumerate(accessIndices))
    sourceIndices[(*resultToSource)[resultDim]] = accessIndex;

  std::optional<mlir::AffineMap> sourceMap = remapTransferPermutationMap(
      permutationMap, *resultToSource, sourceType.getRank());
  if (!sourceMap)
    return std::nullopt;
  return ViewAccess{cast.getSource(), std::move(sourceIndices), *sourceMap};
}

std::optional<ViewAccess> buildReinterpretCastView(
    mlir::memref::ReinterpretCastOp cast, mlir::ValueRange accessIndices,
    mlir::AffineMap permutationMap, mlir::IRRewriter &rewriter) {
  auto sourceType =
      mlir::dyn_cast<mlir::MemRefType>(cast.getSource().getType());
  mlir::MemRefType resultType = cast.getType();
  if (!sourceType || !accessRankMatchesMemRef(resultType, accessIndices))
    return std::nullopt;

  if (std::optional<ViewAccess> permutationView =
          buildPermutationReinterpretCastView(cast, accessIndices,
                                              permutationMap))
    return permutationView;

  if (sourceType.getRank() != resultType.getRank())
    return std::nullopt;

  std::optional<llvm::SmallVector<int64_t, 4>> sourceStrides =
      staticRowMajorSourceStrides(sourceType);
  if (!sourceStrides || !llvm::equal(cast.getStaticStrides(), *sourceStrides))
    return std::nullopt;

  std::optional<mlir::Value> linearOffset =
      buildReinterpretCastLinearOffset(cast, rewriter);
  if (!linearOffset)
    return std::nullopt;

  llvm::SmallVector<mlir::Value, 4> baseIndices = delinearizeRowMajorOffset(
      rewriter, cast.getLoc(), *linearOffset, *sourceStrides);
  llvm::SmallVector<mlir::Value, 4> sourceIndices = addAccessIndices(
      rewriter, cast.getLoc(), std::move(baseIndices), accessIndices);
  return ViewAccess{cast.getSource(), std::move(sourceIndices), permutationMap};
}

std::optional<ViewAccess> buildViewAccess(mlir::Value view,
                                          mlir::ValueRange accessIndices,
                                          mlir::AffineMap permutationMap,
                                          mlir::IRRewriter &rewriter) {
  if (auto subview = view.getDefiningOp<mlir::memref::SubViewOp>())
    return buildSubviewView(subview, accessIndices, permutationMap, rewriter);
  if (auto cast = view.getDefiningOp<mlir::memref::ReinterpretCastOp>())
    return buildReinterpretCastView(cast, accessIndices, permutationMap,
                                    rewriter);
  return std::nullopt;
}

bool rewriteViewLoad(mlir::memref::LoadOp load, mlir::IRRewriter &rewriter) {
  rewriter.setInsertionPoint(load);
  std::optional<ViewAccess> view = buildViewAccess(
      load.getMemRef(), load.getIndices(), mlir::AffineMap(), rewriter);
  if (!view)
    return false;

  rewriter.setInsertionPoint(load);
  mlir::Value replacement = rewriter.create<mlir::memref::LoadOp>(
      load.getLoc(), view->source, view->sourceIndices);
  rewriter.replaceOp(load, replacement);
  return true;
}

bool rewriteViewLoad(mlir::affine::AffineLoadOp load,
                     mlir::IRRewriter &rewriter) {
  if (!load.getMap().isIdentity())
    return false;

  rewriter.setInsertionPoint(load);
  std::optional<ViewAccess> view = buildViewAccess(
      load.getMemref(), load.getIndices(), mlir::AffineMap(), rewriter);
  if (!view)
    return false;

  rewriter.setInsertionPoint(load);
  mlir::Value replacement = rewriter.create<mlir::memref::LoadOp>(
      load.getLoc(), view->source, view->sourceIndices);
  rewriter.replaceOp(load, replacement);
  return true;
}

bool rewriteViewStore(mlir::memref::StoreOp store, mlir::IRRewriter &rewriter) {
  rewriter.setInsertionPoint(store);
  std::optional<ViewAccess> view = buildViewAccess(
      store.getMemRef(), store.getIndices(), mlir::AffineMap(), rewriter);
  if (!view)
    return false;

  rewriter.setInsertionPoint(store);
  rewriter.create<mlir::memref::StoreOp>(store.getLoc(),
                                         store.getValueToStore(), view->source,
                                         view->sourceIndices);
  rewriter.eraseOp(store);
  return true;
}

bool rewriteViewStore(mlir::affine::AffineStoreOp store,
                      mlir::IRRewriter &rewriter) {
  if (!store.getMap().isIdentity())
    return false;

  rewriter.setInsertionPoint(store);
  std::optional<ViewAccess> view = buildViewAccess(
      store.getMemref(), store.getIndices(), mlir::AffineMap(), rewriter);
  if (!view)
    return false;

  rewriter.setInsertionPoint(store);
  rewriter.create<mlir::memref::StoreOp>(store.getLoc(), store.getValue(),
                                         view->source, view->sourceIndices);
  rewriter.eraseOp(store);
  return true;
}

bool rewriteViewTransferRead(mlir::vector::TransferReadOp read,
                             mlir::IRRewriter &rewriter) {
  rewriter.setInsertionPoint(read);
  std::optional<ViewAccess> view = buildViewAccess(
      read.getSource(), read.getIndices(), read.getPermutationMap(), rewriter);
  if (!view)
    return false;

  rewriter.setInsertionPoint(read);
  mlir::AffineMapAttr permutationMapAttr =
      view->permutationMap ? mlir::AffineMapAttr::get(view->permutationMap)
                           : read.getPermutationMapAttr();
  mlir::Value replacement =
      rewriter
          .create<mlir::vector::TransferReadOp>(
              read.getLoc(), read.getVectorType(), view->source,
              view->sourceIndices, permutationMapAttr, read.getPadding(),
              read.getMask(), read.getInBoundsAttr())
          .getVector();
  rewriter.replaceOp(read, replacement);
  return true;
}

bool rewriteViewTransferWrite(mlir::vector::TransferWriteOp write,
                              mlir::IRRewriter &rewriter) {
  rewriter.setInsertionPoint(write);
  std::optional<ViewAccess> view =
      buildViewAccess(write.getSource(), write.getIndices(),
                      write.getPermutationMap(), rewriter);
  if (!view)
    return false;

  rewriter.setInsertionPoint(write);
  mlir::AffineMapAttr permutationMapAttr =
      view->permutationMap ? mlir::AffineMapAttr::get(view->permutationMap)
                           : write.getPermutationMapAttr();
  mlir::Operation *replacement =
      rewriter
          .create<mlir::vector::TransferWriteOp>(
              write.getLoc(), write->getResultTypes(), write.getVector(),
              view->source, view->sourceIndices, permutationMapAttr,
              write.getMask(), write.getInBoundsAttr())
          .getOperation();
  rewriter.replaceOp(write, replacement->getResults());
  return true;
}

bool rewriteViewVectorLoad(mlir::vector::LoadOp load,
                           mlir::IRRewriter &rewriter) {
  rewriter.setInsertionPoint(load);
  std::optional<ViewAccess> view = buildViewAccess(
      load.getBase(), load.getIndices(), mlir::AffineMap(), rewriter);
  if (!view)
    return false;

  rewriter.setInsertionPoint(load);
  mlir::Value replacement =
      rewriter
          .create<mlir::vector::LoadOp>(
              load.getLoc(), load.getResult().getType(), view->source,
              view->sourceIndices, load.getNontemporal())
          .getResult();
  rewriter.replaceOp(load, replacement);
  return true;
}

bool rewriteViewVectorStore(mlir::vector::StoreOp store,
                            mlir::IRRewriter &rewriter) {
  rewriter.setInsertionPoint(store);
  std::optional<ViewAccess> view = buildViewAccess(
      store.getBase(), store.getIndices(), mlir::AffineMap(), rewriter);
  if (!view)
    return false;

  rewriter.setInsertionPoint(store);
  rewriter.create<mlir::vector::StoreOp>(
      store.getLoc(), store.getValueToStore(), view->source,
      view->sourceIndices, store.getNontemporal());
  rewriter.eraseOp(store);
  return true;
}

class ViewAccessLoweringPass
    : public mlir::PassWrapper<ViewAccessLoweringPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ViewAccessLoweringPass)

  llvm::StringRef getArgument() const final { return "lython-view-access"; }
  llvm::StringRef getDescription() const final {
    return "fold rank-preserving view accesses into source memrefs";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<mlir::affine::AffineDialect, mlir::arith::ArithDialect,
                    mlir::memref::MemRefDialect, mlir::vector::VectorDialect>();
  }

  void runOnOperation() final {
    llvm::SmallVector<mlir::Operation *, 32> accesses;
    getOperation().walk([&](mlir::Operation *op) {
      if (mlir::isa<mlir::affine::AffineLoadOp, mlir::affine::AffineStoreOp,
                    mlir::memref::LoadOp, mlir::memref::StoreOp,
                    mlir::vector::LoadOp, mlir::vector::StoreOp,
                    mlir::vector::TransferReadOp,
                    mlir::vector::TransferWriteOp>(op))
        accesses.push_back(op);
    });

    mlir::IRRewriter rewriter(&getContext());
    for (mlir::Operation *access : accesses) {
      if (!access->getBlock())
        continue;
      if (auto load = mlir::dyn_cast<mlir::memref::LoadOp>(access)) {
        rewriteViewLoad(load, rewriter);
        continue;
      }
      if (auto load = mlir::dyn_cast<mlir::affine::AffineLoadOp>(access)) {
        rewriteViewLoad(load, rewriter);
        continue;
      }
      if (auto store = mlir::dyn_cast<mlir::memref::StoreOp>(access)) {
        rewriteViewStore(store, rewriter);
        continue;
      }
      if (auto store = mlir::dyn_cast<mlir::affine::AffineStoreOp>(access)) {
        rewriteViewStore(store, rewriter);
        continue;
      }
      if (auto read = mlir::dyn_cast<mlir::vector::TransferReadOp>(access)) {
        rewriteViewTransferRead(read, rewriter);
        continue;
      }
      if (auto load = mlir::dyn_cast<mlir::vector::LoadOp>(access)) {
        rewriteViewVectorLoad(load, rewriter);
        continue;
      }
      if (auto store = mlir::dyn_cast<mlir::vector::StoreOp>(access)) {
        rewriteViewVectorStore(store, rewriter);
        continue;
      }
      rewriteViewTransferWrite(
          mlir::cast<mlir::vector::TransferWriteOp>(access), rewriter);
    }

    llvm::SmallVector<mlir::memref::SubViewOp, 16> deadSubviews;
    getOperation().walk([&](mlir::memref::SubViewOp subview) {
      if (subview->use_empty())
        deadSubviews.push_back(subview);
    });
    for (mlir::memref::SubViewOp subview : deadSubviews)
      subview.erase();

    llvm::SmallVector<mlir::memref::ReinterpretCastOp, 16> deadCasts;
    getOperation().walk([&](mlir::memref::ReinterpretCastOp cast) {
      if (cast->use_empty())
        deadCasts.push_back(cast);
    });
    for (mlir::memref::ReinterpretCastOp cast : deadCasts)
      cast.erase();

    foldRowMajorDelinearization(getOperation(), rewriter);
  }
};

class RowMajorDelinearizationFoldPass
    : public mlir::PassWrapper<RowMajorDelinearizationFoldPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RowMajorDelinearizationFoldPass)

  llvm::StringRef getArgument() const final {
    return "lython-row-major-delinearization-fold";
  }
  llvm::StringRef getDescription() const final {
    return "fold row-major div/rem delinearization back to source indices";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<mlir::arith::ArithDialect, mlir::scf::SCFDialect>();
  }

  void runOnOperation() final {
    mlir::IRRewriter rewriter(&getContext());
    foldRowMajorDelinearization(getOperation(), rewriter);
  }
};

std::optional<int64_t> selectReductionVectorLanes(mlir::Type elementType,
                                                  int64_t tripCount) {
  return selectDivisibleVectorLanes(elementType, tripCount,
                                    kReductionVectorBits, /*minLanes=*/2);
}

std::optional<mlir::arith::FastMathFlags>
matchAddReductionOp(mlir::Operation *op, mlir::Value reductionArg,
                    mlir::memref::LoadOp &load) {
  if (auto add = mlir::dyn_cast_or_null<mlir::arith::AddFOp>(op)) {
    if (add.getLhs() == reductionArg) {
      load = add.getRhs().getDefiningOp<mlir::memref::LoadOp>();
      if (load)
        return add.getFastmath();
    }
    if (add.getRhs() == reductionArg) {
      load = add.getLhs().getDefiningOp<mlir::memref::LoadOp>();
      if (load)
        return add.getFastmath();
    }
  }

  if (auto add = mlir::dyn_cast_or_null<mlir::arith::AddIOp>(op)) {
    if (add.getLhs() == reductionArg) {
      load = add.getRhs().getDefiningOp<mlir::memref::LoadOp>();
      if (load)
        return mlir::arith::FastMathFlags::none;
    }
    if (add.getRhs() == reductionArg) {
      load = add.getLhs().getDefiningOp<mlir::memref::LoadOp>();
      if (load)
        return mlir::arith::FastMathFlags::none;
    }
  }

  return std::nullopt;
}

bool bodyContainsOnlyReduction(mlir::scf::ForOp loop, mlir::memref::LoadOp load,
                               mlir::Operation *add) {
  for (mlir::Operation &op : loop.getBody()->without_terminator()) {
    if (&op != load.getOperation() && &op != add)
      return false;
  }
  return true;
}

struct ContiguousReductionMatch {
  mlir::memref::LoadOp load;
  mlir::arith::FastMathFlags fastMath;
  int64_t lanes;
};

std::optional<mlir::Value>
createPrimitiveVectorAdd(mlir::OpBuilder &builder, mlir::Location loc,
                         mlir::Value lhs, mlir::Value rhs,
                         mlir::Type elementType,
                         mlir::arith::FastMathFlags fastMath) {
  if (mlir::isa<mlir::FloatType>(elementType))
    return builder.create<mlir::arith::AddFOp>(loc, lhs, rhs, fastMath)
        .getResult();
  if (mlir::isa<mlir::IntegerType>(elementType))
    return builder.create<mlir::arith::AddIOp>(loc, lhs, rhs).getResult();
  return std::nullopt;
}

std::optional<ContiguousReductionMatch>
matchContiguousInnerReduction(mlir::scf::ForOp loop) {
  if (loop.getNumResults() != 1 || loop.getNumRegionIterArgs() != 1)
    return std::nullopt;

  std::optional<int64_t> lower = constantIndexValue(loop.getLowerBound());
  std::optional<int64_t> upper = constantIndexValue(loop.getUpperBound());
  std::optional<int64_t> step = constantIndexValue(loop.getStep());
  if (!lower || !upper || !step || *step != 1 || *upper <= *lower)
    return std::nullopt;

  auto yield =
      mlir::dyn_cast<mlir::scf::YieldOp>(loop.getBody()->getTerminator());
  if (!yield || yield.getNumOperands() != 1)
    return std::nullopt;

  mlir::Operation *add = yield.getOperand(0).getDefiningOp();
  if (!add || add->getBlock() != loop.getBody())
    return std::nullopt;

  mlir::memref::LoadOp load;
  std::optional<mlir::arith::FastMathFlags> fastMath =
      matchAddReductionOp(add, loop.getRegionIterArgs().front(), load);
  if (!fastMath || !load || load->getBlock() != loop.getBody() ||
      !bodyContainsOnlyReduction(loop, load, add))
    return std::nullopt;

  mlir::MemRefType memrefType =
      mlir::dyn_cast<mlir::MemRefType>(load.getMemRef().getType());
  if (!memrefType || load.getIndices().empty() ||
      load.getIndices().size() !=
          static_cast<std::size_t>(memrefType.getRank()) ||
      !memrefHasContiguousInnerDimension(memrefType))
    return std::nullopt;

  if (load.getIndices().back() != loop.getInductionVar())
    return std::nullopt;

  int64_t innerDim = memrefType.getDimSize(memrefType.getRank() - 1);
  if (*lower < 0 || *upper > innerDim)
    return std::nullopt;

  std::optional<int64_t> lanes =
      selectReductionVectorLanes(memrefType.getElementType(), *upper - *lower);
  if (!lanes)
    return std::nullopt;

  return ContiguousReductionMatch{load, *fastMath, *lanes};
}

bool vectorizeContiguousReduction(mlir::scf::ForOp loop,
                                  ContiguousReductionMatch match,
                                  mlir::IRRewriter &rewriter) {
  mlir::Location loc = loop.getLoc();
  mlir::MemRefType memrefType =
      mlir::cast<mlir::MemRefType>(match.load.getMemRef().getType());
  mlir::Type elementType = memrefType.getElementType();

  rewriter.setInsertionPoint(loop);
  std::optional<mlir::Value> padding =
      createPrimitiveZeroValue(rewriter, loc, elementType);
  if (!padding)
    return false;

  mlir::Value vectorStep =
      rewriter.create<mlir::arith::ConstantIndexOp>(loc, match.lanes);
  mlir::VectorType vectorType =
      mlir::VectorType::get({match.lanes}, elementType);
  mlir::Value initialVector =
      rewriter.create<mlir::vector::BroadcastOp>(loc, vectorType, *padding);

  auto vectorLoop = rewriter.create<mlir::scf::ForOp>(
      loc, loop.getLowerBound(), loop.getUpperBound(), vectorStep,
      mlir::ValueRange{initialVector},
      [&](mlir::OpBuilder &builder, mlir::Location nestedLoc, mlir::Value iv,
          mlir::ValueRange iterArgs) {
        llvm::SmallVector<mlir::Value, 4> indices =
            llvm::to_vector<4>(match.load.getIndices());
        indices.back() = iv;

        mlir::Value vector =
            builder
                .create<mlir::vector::LoadOp>(nestedLoc, vectorType,
                                              match.load.getMemRef(), indices)
                .getResult();
        std::optional<mlir::Value> next =
            createPrimitiveVectorAdd(builder, nestedLoc, iterArgs.front(),
                                     vector, elementType, match.fastMath);
        if (!next)
          return;
        builder.create<mlir::scf::YieldOp>(nestedLoc, *next);
      });

  rewriter.setInsertionPointAfter(vectorLoop);
  mlir::Value reduced = rewriter.create<mlir::vector::ReductionOp>(
      loc, mlir::vector::CombiningKind::ADD, vectorLoop.getResult(0),
      loop.getInitArgs().front(), match.fastMath);
  rewriter.replaceOp(loop, reduced);
  return true;
}

class ContiguousReductionVectorizationPass
    : public mlir::PassWrapper<ContiguousReductionVectorizationPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      ContiguousReductionVectorizationPass)

  llvm::StringRef getArgument() const final {
    return "lython-contiguous-reduction-vectorization";
  }
  llvm::StringRef getDescription() const final {
    return "vectorize static contiguous primitive reductions";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<mlir::arith::ArithDialect, mlir::memref::MemRefDialect,
                    mlir::scf::SCFDialect, mlir::vector::VectorDialect>();
  }

  void runOnOperation() final {
    llvm::SmallVector<mlir::scf::ForOp, 16> loops;
    getOperation().walk([&](mlir::scf::ForOp loop) { loops.push_back(loop); });

    mlir::IRRewriter rewriter(&getContext());
    for (mlir::scf::ForOp loop : loops) {
      if (!loop->getBlock())
        continue;
      std::optional<ContiguousReductionMatch> match =
          matchContiguousInnerReduction(loop);
      if (match)
        vectorizeContiguousReduction(loop, *match, rewriter);
    }
  }
};

bool provesVectorDimInBounds(mlir::Value start, int64_t vectorSize,
                             int64_t dimSize) {
  if (vectorSize <= 0 || dimSize <= 0)
    return false;

  std::optional<StaticIndexRange> range = staticIndexRange(start);
  if (!range || range->min < 0)
    return false;

  std::optional<int64_t> last = checkedAdd(range->max, vectorSize - 1);
  return last && *last < dimSize;
}

bool sourceShapeCanCoverBroadcastVectorDim(mlir::ShapedType sourceType,
                                           int64_t vectorSize) {
  if (vectorSize <= 1)
    return true;
  for (int64_t dimSize : sourceType.getShape()) {
    if (dimSize <= 0 || dimSize < vectorSize)
      return false;
  }
  return true;
}

bool sourceShapeCanCoverVectorTransfer(mlir::ShapedType sourceType,
                                       mlir::VectorType vectorType) {
  int64_t maxVectorDim = 1;
  for (int64_t vectorDim : vectorType.getShape())
    maxVectorDim = std::max(maxVectorDim, vectorDim);
  return sourceShapeCanCoverBroadcastVectorDim(sourceType, maxVectorDim);
}

std::optional<mlir::ArrayAttr> computeStaticInBoundsAttr(
    mlir::MLIRContext *context, mlir::VectorType vectorType,
    mlir::ShapedType sourceType, mlir::ValueRange indices,
    mlir::AffineMap permutationMap, mlir::ArrayAttr current) {
  if (!sourceType.hasStaticShape() || sourceType.getRank() == 0 ||
      permutationMap.getNumResults() != vectorType.getRank() ||
      indices.size() != static_cast<std::size_t>(sourceType.getRank()))
    return std::nullopt;
  if (!sourceShapeCanCoverVectorTransfer(sourceType, vectorType))
    return std::nullopt;

  llvm::SmallVector<bool, 4> inBounds(vectorType.getRank(), false);
  if (current) {
    for (auto [index, attr] : llvm::enumerate(current.getValue())) {
      if (index >= inBounds.size())
        break;
      if (auto flag = mlir::dyn_cast<mlir::BoolAttr>(attr))
        inBounds[index] = flag.getValue();
    }
  }

  bool changed = false;
  for (auto [vectorDim, expr] : llvm::enumerate(permutationMap.getResults())) {
    if (mlir::isa<mlir::AffineConstantExpr>(expr)) {
      int64_t vectorSize = vectorType.getDimSize(vectorDim);
      if (inBounds[vectorDim] &&
          !sourceShapeCanCoverBroadcastVectorDim(sourceType, vectorSize)) {
        inBounds[vectorDim] = false;
        changed = true;
      }
      continue;
    }

    auto dimExpr = mlir::dyn_cast<mlir::AffineDimExpr>(expr);
    if (!dimExpr)
      continue;

    unsigned memrefDim = dimExpr.getPosition();
    if (memrefDim >= sourceType.getRank())
      continue;

    int64_t dimSize = sourceType.getDimSize(memrefDim);
    int64_t vectorSize = vectorType.getDimSize(vectorDim);
    bool proven =
        provesVectorDimInBounds(indices[memrefDim], vectorSize, dimSize);
    if (inBounds[vectorDim] != proven) {
      inBounds[vectorDim] = proven;
      changed = true;
    }
  }

  if (!changed)
    return std::nullopt;

  mlir::Builder builder(context);
  llvm::SmallVector<mlir::Attribute, 4> attrs;
  attrs.reserve(inBounds.size());
  for (bool flag : inBounds)
    attrs.push_back(builder.getBoolAttr(flag));
  return builder.getArrayAttr(attrs);
}

bool markTransferReadInBounds(mlir::vector::TransferReadOp read) {
  auto sourceType =
      mlir::dyn_cast<mlir::ShapedType>(read.getSource().getType());
  if (!sourceType)
    return false;

  std::optional<mlir::ArrayAttr> inBounds = computeStaticInBoundsAttr(
      read.getContext(), read.getVectorType(), sourceType, read.getIndices(),
      read.getPermutationMap(), read.getInBoundsAttr());
  if (!inBounds)
    return false;

  read.setInBoundsAttr(*inBounds);
  return true;
}

bool markTransferWriteInBounds(mlir::vector::TransferWriteOp write) {
  auto sourceType =
      mlir::dyn_cast<mlir::ShapedType>(write.getSource().getType());
  if (!sourceType)
    return false;

  auto vectorType = mlir::cast<mlir::VectorType>(write.getVector().getType());
  std::optional<mlir::ArrayAttr> inBounds = computeStaticInBoundsAttr(
      write.getContext(), vectorType, sourceType, write.getIndices(),
      write.getPermutationMap(), write.getInBoundsAttr());
  if (!inBounds)
    return false;

  write.setInBoundsAttr(*inBounds);
  return true;
}

class StaticTransferInBoundsPass
    : public mlir::PassWrapper<StaticTransferInBoundsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StaticTransferInBoundsPass)

  llvm::StringRef getArgument() const final {
    return "lython-static-transfer-in-bounds";
  }
  llvm::StringRef getDescription() const final {
    return "prove in-bounds vector transfers from static loop ranges";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<mlir::vector::VectorDialect, mlir::scf::SCFDialect>();
  }

  void runOnOperation() final {
    getOperation().walk([](mlir::Operation *op) {
      if (auto read = mlir::dyn_cast<mlir::vector::TransferReadOp>(op)) {
        markTransferReadInBounds(read);
        return;
      }
      if (auto write = mlir::dyn_cast<mlir::vector::TransferWriteOp>(op))
        markTransferWriteInBounds(write);
    });
  }
};

class LinalgLoweringPass
    : public mlir::PassWrapper<LinalgLoweringPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinalgLoweringPass)

  explicit LinalgLoweringPass(py::TensorLoweringTarget target = {})
      : target(target) {}

  llvm::StringRef getArgument() const final { return "lython-linalg-lowering"; }
  llvm::StringRef getDescription() const final {
    return "lower primitive tensor contractions through linalg/affine/vector";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<mlir::affine::AffineDialect, mlir::arith::ArithDialect,
                    mlir::bufferization::BufferizationDialect,
                    mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                    mlir::memref::MemRefDialect, mlir::scf::SCFDialect,
                    mlir::tensor::TensorDialect, mlir::vector::VectorDialect>();
    if (arch::arm::usesSME(target))
      arch::arm::registerSMEDialects(registry);
    if (arch::x86::usesX86(target))
      arch::x86::registerX86Dialects(registry);
  }

  void runOnOperation() final {
    mlir::bufferization::OneShotBufferizationOptions bufferizeOptions;
    bufferizeOptions.allowUnknownOps = true;

    mlir::affine::AffineVectorizeOptions vectorizeOptions;
    vectorizeOptions.vectorSizes = {4};
    vectorizeOptions.vectorizeReductions = true;

    mlir::OpPassManager pipeline(mlir::ModuleOp::getOperationName());
    pipeline.addPass(createMatmulZeroInitElisionPass());
    pipeline.addPass(
        mlir::bufferization::createOneShotBufferizePass(bufferizeOptions));
    if (arch::arm::usesSME(target))
      pipeline.addPass(arch::arm::createMatmulSMELoweringPass());
    pipeline.addPass(createMatmulTilingPass());
    pipeline.addPass(createMatmulPackingPass());
    pipeline.addPass(createPackedPanelHoistingPass());
    pipeline.addPass(createPackedPanelCopyHoistingPass());
    pipeline.addPass(createPackedPanelCopyVectorizationPass());
    pipeline.addPass(mlir::createLoopInvariantCodeMotionPass());
    pipeline.addPass(mlir::createCSEPass());
    pipeline.addPass(createMatmulMicroTilingPass());
    pipeline.addPass(createMatmulVectorizationPass());
    if (arch::arm::usesSME(target))
      arch::arm::addSMELinalgPipeline(pipeline);
    if (arch::x86::usesX86(target))
      arch::x86::addX86LinalgPipeline(pipeline);
    pipeline.addPass(mlir::createLoopInvariantCodeMotionPass());
    pipeline.addPass(mlir::createCSEPass());
    pipeline.addPass(std::make_unique<ViewAccessLoweringPass>());
    pipeline.addPass(std::make_unique<ContiguousReductionVectorizationPass>());
    pipeline.addPass(mlir::createConvertLinalgToAffineLoopsPass());
    pipeline.addPass(mlir::memref::createExpandStridedMetadataPass());
    pipeline.addPass(std::make_unique<ViewAccessLoweringPass>());
    pipeline.addNestedPass<mlir::func::FuncOp>(
        mlir::affine::createAffineVectorize(vectorizeOptions));
    pipeline.addPass(mlir::createLowerAffinePass());
    pipeline.addPass(std::make_unique<RowMajorDelinearizationFoldPass>());
    pipeline.addPass(std::make_unique<StaticTransferInBoundsPass>());
    pipeline.addPass(mlir::createCanonicalizerPass());
    pipeline.addPass(mlir::createCSEPass());

    if (mlir::failed(runPipeline(pipeline, getOperation())))
      signalPassFailure();
  }

private:
  py::TensorLoweringTarget target;
};

} // namespace
} // namespace py::runtime_lowering

namespace py {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createLinalgLoweringPass(TensorLoweringTarget target) {
  return std::make_unique<runtime_lowering::LinalgLoweringPass>(target);
}

} // namespace py
