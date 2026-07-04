#include "TensorPacking.h"
#include "TensorSupport.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"

#include <optional>

namespace py::runtime_lowering {
namespace {

struct RhsPanelSlice {
  mlir::Value base;
  mlir::OpFoldResult kOffset;
  mlir::OpFoldResult nOffset;
};

mlir::Value createIndexConstant(mlir::OpBuilder &builder, mlir::Location loc,
                                int64_t value) {
  return builder.create<mlir::arith::ConstantIndexOp>(loc, value).getResult();
}

mlir::Value addIndexValues(mlir::OpBuilder &builder, mlir::Location loc,
                           mlir::Value lhs, mlir::Value rhs) {
  return builder.create<mlir::arith::AddIOp>(loc, lhs, rhs).getResult();
}

mlir::Value multiplyIndexByConstant(mlir::OpBuilder &builder,
                                    mlir::Location loc, mlir::Value value,
                                    int64_t factor) {
  if (factor == 1)
    return value;
  return builder
      .create<mlir::arith::MulIOp>(loc, value,
                                   createIndexConstant(builder, loc, factor))
      .getResult();
}

std::optional<int64_t> selectPackedCopyVectorLanes(mlir::Type elementType,
                                                   int64_t columns) {
  return selectDivisibleVectorLanes(elementType, columns, kPackedCopyVectorBits,
                                    /*minLanes=*/1);
}

bool isBlockArgumentDefinedInside(mlir::Value value, mlir::Operation *scope) {
  auto argument = mlir::dyn_cast<mlir::BlockArgument>(value);
  if (!argument)
    return false;

  mlir::Operation *owner = argument.getOwner()->getParentOp();
  return owner && scope->isProperAncestor(owner);
}

bool operationWritesToMemrefRoot(mlir::Operation *op, mlir::Value root) {
  if (auto store = mlir::dyn_cast<mlir::memref::StoreOp>(op))
    return sameMemrefRoot(store.getMemRef(), root);
  if (auto store = mlir::dyn_cast<mlir::vector::StoreOp>(op))
    return sameMemrefRoot(store.getBase(), root);
  if (auto write = mlir::dyn_cast<mlir::vector::TransferWriteOp>(op))
    return sameMemrefRoot(write.getSource(), root);
  if (auto linalg = mlir::dyn_cast<mlir::linalg::LinalgOp>(op)) {
    for (int64_t index = 0; index < linalg.getNumDpsInits(); ++index) {
      mlir::OpOperand *init = linalg.getDpsInitOperand(index);
      if (sameMemrefRoot(init->get(), root))
        return true;
    }
  }
  return false;
}

bool loopWritesToMemrefRoot(mlir::scf::ForOp loop, mlir::Value root) {
  bool writes = false;
  loop.walk([&](mlir::Operation *op) {
    if (!writes && operationWritesToMemrefRoot(op, root))
      writes = true;
  });
  return writes;
}

bool sourceIsLargePrimitiveTensor(mlir::Value value) {
  auto sourceType =
      mlir::dyn_cast<mlir::MemRefType>(sourceMemrefRoot(value).getType());
  if (!sourceType || sourceType.getRank() != 2 ||
      !isPrimitiveElementType(sourceType.getElementType()))
    return false;
  return staticElementCount(sourceType) >=
         kPrimitiveTensorPackedMinSourceElements;
}

std::optional<RhsPanelSlice> rhsPanelSlice(mlir::Value source) {
  if (auto subview = source.getDefiningOp<mlir::memref::SubViewOp>()) {
    if (subview.getType().getRank() != 2 ||
        subview.getStaticOffsets().size() != 2 ||
        subview.getStaticSizes().size() != 2 ||
        subview.getStaticStrides().size() != 2)
      return std::nullopt;
    if (subview.getStaticStrides()[0] != 1 ||
        subview.getStaticStrides()[1] != 1)
      return std::nullopt;
    llvm::SmallVector<mlir::OpFoldResult, 2> offsets =
        subview.getMixedOffsets();
    return RhsPanelSlice{subview.getSource(), offsets[0], offsets[1]};
  }

  auto type = mlir::dyn_cast<mlir::MemRefType>(source.getType());
  if (!type || type.getRank() != 2)
    return std::nullopt;
  mlir::OpFoldResult zero =
      mlir::getAsIndexOpFoldResult(source.getContext(), 0);
  return RhsPanelSlice{source, zero, zero};
}

bool isLoopInvariantFor(mlir::Value value, mlir::scf::ForOp loop) {
  if (value == loop.getInductionVar())
    return false;
  if (isBlockArgumentDefinedInside(value, loop.getOperation()))
    return false;
  mlir::Operation *def = value.getDefiningOp();
  return !def || !loop->isProperAncestor(def);
}

std::optional<mlir::scf::ForOp>
prepackInsertionAnchor(mlir::linalg::MatmulOp matmul, mlir::Value sourceRoot) {
  std::optional<mlir::scf::ForOp> anchor;
  for (mlir::Operation *parent = matmul->getParentOp(); parent;
       parent = parent->getParentOp()) {
    auto loop = mlir::dyn_cast<mlir::scf::ForOp>(parent);
    if (!loop)
      continue;
    if (!isLoopInvariantFor(sourceRoot, loop))
      break;
    if (loopWritesToMemrefRoot(loop, sourceRoot))
      break;
    anchor = loop;
  }
  return anchor;
}

bool rhsPrepackShapeIsExact(mlir::MemRefType baseType,
                            mlir::MemRefType panelType) {
  if (baseType.getRank() != 2 || panelType.getRank() != 2 ||
      !baseType.hasStaticShape() || !panelType.hasStaticShape())
    return false;
  if (baseType.getElementType() != panelType.getElementType())
    return false;
  if (!memrefHasContiguousInnerDimension(baseType) ||
      !memrefHasContiguousInnerDimension(panelType))
    return false;
  int64_t totalK = baseType.getDimSize(0);
  int64_t totalN = baseType.getDimSize(1);
  int64_t panelK = panelType.getDimSize(0);
  int64_t panelN = panelType.getDimSize(1);
  return totalK > 0 && totalN > 0 && panelK > 0 && panelN > 0 &&
         totalK % panelK == 0 && totalN % panelN == 0;
}

mlir::Value linearRhsPanelOffset(mlir::OpBuilder &builder, mlir::Location loc,
                                 mlir::Value kOffset, mlir::Value nOffset,
                                 int64_t totalK, int64_t panelN) {
  mlir::Value panelColumnBase =
      multiplyIndexByConstant(builder, loc, nOffset, totalK);
  mlir::Value panelRowBase =
      multiplyIndexByConstant(builder, loc, kOffset, panelN);
  return addIndexValues(builder, loc, panelColumnBase, panelRowBase);
}

void buildFullRhsPrepackLoops(mlir::Value base, mlir::Value storage,
                              mlir::MemRefType baseType,
                              mlir::MemRefType panelType,
                              mlir::Operation *anchor,
                              mlir::IRRewriter &rewriter) {
  mlir::Location loc = anchor->getLoc();
  mlir::Type elementType = baseType.getElementType();
  int64_t totalK = baseType.getDimSize(0);
  int64_t totalN = baseType.getDimSize(1);
  int64_t panelK = panelType.getDimSize(0);
  int64_t panelN = panelType.getDimSize(1);
  std::optional<int64_t> lanes =
      selectPackedCopyVectorLanes(elementType, panelN);
  if (!lanes || *lanes <= 1)
    return;

  rewriter.setInsertionPointAfter(storage.getDefiningOp());
  mlir::Value zero = createIndexConstant(rewriter, loc, 0);
  mlir::Value one = createIndexConstant(rewriter, loc, 1);
  mlir::Value totalKValue = createIndexConstant(rewriter, loc, totalK);
  mlir::Value totalNValue = createIndexConstant(rewriter, loc, totalN);
  mlir::Value panelKValue = createIndexConstant(rewriter, loc, panelK);
  mlir::Value panelNValue = createIndexConstant(rewriter, loc, panelN);
  mlir::Value lanesValue = createIndexConstant(rewriter, loc, *lanes);

  auto nPanelLoop =
      rewriter.create<mlir::scf::ForOp>(loc, zero, totalNValue, panelNValue);
  {
    mlir::OpBuilder::InsertionGuard nGuard(rewriter);
    rewriter.setInsertionPointToStart(nPanelLoop.getBody());
    auto kPanelLoop =
        rewriter.create<mlir::scf::ForOp>(loc, zero, totalKValue, panelKValue);
    {
      mlir::OpBuilder::InsertionGuard kPanelGuard(rewriter);
      rewriter.setInsertionPointToStart(kPanelLoop.getBody());
      mlir::Value panelOffset =
          linearRhsPanelOffset(rewriter, loc, kPanelLoop.getInductionVar(),
                               nPanelLoop.getInductionVar(), totalK, panelN);
      auto kLoop =
          rewriter.create<mlir::scf::ForOp>(loc, zero, panelKValue, one);
      {
        mlir::OpBuilder::InsertionGuard kGuard(rewriter);
        rewriter.setInsertionPointToStart(kLoop.getBody());
        auto nLoop = rewriter.create<mlir::scf::ForOp>(loc, zero, panelNValue,
                                                       lanesValue);
        {
          mlir::OpBuilder::InsertionGuard nInnerGuard(rewriter);
          rewriter.setInsertionPointToStart(nLoop.getBody());
          mlir::VectorType vectorType =
              mlir::VectorType::get({*lanes}, elementType);
          mlir::Value sourceK =
              addIndexValues(rewriter, loc, kPanelLoop.getInductionVar(),
                             kLoop.getInductionVar());
          mlir::Value sourceN =
              addIndexValues(rewriter, loc, nPanelLoop.getInductionVar(),
                             nLoop.getInductionVar());
          mlir::Value vector =
              rewriter
                  .create<mlir::vector::LoadOp>(
                      loc, vectorType, base, mlir::ValueRange{sourceK, sourceN})
                  .getResult();
          mlir::Value panelRowOffset = multiplyIndexByConstant(
              rewriter, loc, kLoop.getInductionVar(), panelN);
          mlir::Value linearOffset =
              addIndexValues(rewriter, loc, panelOffset, panelRowOffset);
          linearOffset = addIndexValues(rewriter, loc, linearOffset,
                                        nLoop.getInductionVar());
          rewriter.create<mlir::vector::StoreOp>(
              loc, vector, storage, mlir::ValueRange{linearOffset});
        }
      }
    }
  }
}

RhsPrepackPlan *findRhsPrepackPlan(llvm::SmallVectorImpl<RhsPrepackPlan> &plans,
                                   mlir::Value base, mlir::Operation *anchor,
                                   mlir::MemRefType baseType,
                                   mlir::MemRefType panelType) {
  for (RhsPrepackPlan &plan : plans) {
    if (plan.base == base && plan.anchor == anchor &&
        plan.totalK == baseType.getDimSize(0) &&
        plan.totalN == baseType.getDimSize(1) &&
        plan.panelK == panelType.getDimSize(0) &&
        plan.panelN == panelType.getDimSize(1))
      return &plan;
  }
  return nullptr;
}

RhsPrepackPlan *
getOrCreateRhsPrepackPlan(llvm::SmallVectorImpl<RhsPrepackPlan> &plans,
                          mlir::Value base, mlir::Operation *anchor,
                          mlir::MemRefType baseType, mlir::MemRefType panelType,
                          mlir::IRRewriter &rewriter) {
  if (RhsPrepackPlan *existing =
          findRhsPrepackPlan(plans, base, anchor, baseType, panelType))
    return existing;

  mlir::Location loc = anchor->getLoc();
  mlir::MemRefType storageType = mlir::MemRefType::get(
      {static_cast<int64_t>(staticElementCount(baseType))},
      baseType.getElementType(), mlir::MemRefLayoutAttrInterface{},
      baseType.getMemorySpace());
  rewriter.setInsertionPoint(anchor);
  mlir::Value storage =
      rewriter
          .create<mlir::memref::AllocOp>(
              loc, storageType, mlir::ValueRange{},
              rewriter.getI64IntegerAttr(kPackedPanelAlignment))
          .getResult();
  storage.getDefiningOp()->setAttr(kPackedPanelAttr, rewriter.getUnitAttr());
  storage.getDefiningOp()->setAttr(kPrepackedRhsAttr, rewriter.getUnitAttr());
  buildFullRhsPrepackLoops(base, storage, baseType, panelType, anchor,
                           rewriter);

  plans.push_back(RhsPrepackPlan{
      base, storage, anchor, baseType.getDimSize(0), baseType.getDimSize(1),
      panelType.getDimSize(0), panelType.getDimSize(1)});
  return &plans.back();
}

} // namespace

bool shouldPackRhsPanel(mlir::linalg::MatmulOp matmul) {
  if (matmul->hasAttr(kPackedRhsAttr) ||
      !matmul->hasAttr(kPackRhsCandidateAttr))
    return false;

  mlir::Value panelValue = matmul.getDpsInputOperand(1)->get();
  auto panelType = mlir::dyn_cast<mlir::MemRefType>(panelValue.getType());
  if (!panelType || panelType.getRank() != 2 || !panelType.hasStaticShape())
    return false;

  return sourceIsLargePrimitiveTensor(panelValue);
}

bool tryPrepackFullRhsPanel(mlir::linalg::MatmulOp matmul,
                            mlir::IRRewriter &rewriter,
                            llvm::SmallVectorImpl<RhsPrepackPlan> &plans) {
  if (!shouldPackRhsPanel(matmul))
    return false;

  mlir::OpOperand *operand = matmul.getDpsInputOperand(1);
  mlir::Value source = operand->get();
  auto panelType = mlir::dyn_cast<mlir::MemRefType>(source.getType());
  std::optional<RhsPanelSlice> slice = rhsPanelSlice(source);
  if (!panelType || !slice)
    return false;

  mlir::Value base = sourceMemrefRoot(slice->base);
  auto baseType = mlir::dyn_cast<mlir::MemRefType>(base.getType());
  if (slice->base != base || !baseType ||
      !rhsPrepackShapeIsExact(baseType, panelType))
    return false;

  std::optional<mlir::scf::ForOp> anchor = prepackInsertionAnchor(matmul, base);
  if (!anchor)
    return false;

  RhsPrepackPlan *plan = getOrCreateRhsPrepackPlan(
      plans, base, anchor->getOperation(), baseType, panelType, rewriter);
  if (!plan)
    return false;

  mlir::Location loc = matmul.getLoc();
  rewriter.setInsertionPoint(matmul);
  mlir::Value kOffset =
      mlir::getValueOrCreateConstantIndexOp(rewriter, loc, slice->kOffset);
  mlir::Value nOffset =
      mlir::getValueOrCreateConstantIndexOp(rewriter, loc, slice->nOffset);
  mlir::Value linearOffset = linearRhsPanelOffset(
      rewriter, loc, kOffset, nOffset, plan->totalK, plan->panelN);

  llvm::SmallVector<int64_t, 2> strides{plan->panelN, 1};
  mlir::MemRefType packedType = mlir::MemRefType::get(
      {plan->panelK, plan->panelN}, baseType.getElementType(),
      mlir::StridedLayoutAttr::get(matmul.getContext(),
                                   mlir::ShapedType::kDynamic, strides),
      baseType.getMemorySpace());
  llvm::SmallVector<mlir::OpFoldResult, 2> sizes{
      rewriter.getIndexAttr(plan->panelK), rewriter.getIndexAttr(plan->panelN)};
  llvm::SmallVector<mlir::OpFoldResult, 2> castStrides{
      rewriter.getIndexAttr(plan->panelN), rewriter.getIndexAttr(1)};
  mlir::Value packed =
      rewriter
          .create<mlir::memref::ReinterpretCastOp>(
              loc, packedType, plan->storage, mlir::OpFoldResult(linearOffset),
              sizes, castStrides, llvm::ArrayRef<mlir::NamedAttribute>{})
          .getResult();
  packed.getDefiningOp()->setAttr(kPackedPanelAttr, rewriter.getUnitAttr());
  packed.getDefiningOp()->setAttr(kPrepackedRhsAttr, rewriter.getUnitAttr());
  operand->set(packed);
  matmul->setAttr(kPackedRhsAttr, rewriter.getUnitAttr());
  matmul->removeAttr(kPackRhsCandidateAttr);
  return true;
}

} // namespace py::runtime_lowering
