#include "Passes/Runtime/Cleanup.h"

#include "Common/LoweringUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

namespace py {
namespace {

bool eraseUnreachableBlocksInRegion(mlir::Region &region) {
  if (region.empty())
    return false;

  bool changed = false;
  bool localChanged = false;
  do {
    localChanged = false;
    for (mlir::Block &block :
         llvm::make_early_inc_range(llvm::drop_begin(region.getBlocks()))) {
      if (!block.hasNoPredecessors())
        continue;
      block.dropAllDefinedValueUses();
      block.dropAllReferences();
      block.erase();
      localChanged = true;
      changed = true;
    }
  } while (localChanged);
  return changed;
}

namespace memref_descriptor_cast {

struct ExtractSource {
  mlir::UnrealizedConversionCastOp insertionPoint;
  mlir::Location loc;
  mlir::Value aggregate;
  mlir::LLVM::LLVMStructType type;
};

mlir::LLVM::LLVMStructType structTypeOf(mlir::Value value) {
  auto type = mlir::dyn_cast<mlir::LLVM::LLVMStructType>(value.getType());
  if (!type || type.isOpaque())
    return {};
  return type;
}

mlir::Value extract(mlir::OpResult result, const ExtractSource &source,
                    mlir::OpBuilder &builder) {
  unsigned index = result.getResultNumber();
  if (index >= source.type.getBody().size())
    return {};
  builder.setInsertionPoint(source.insertionPoint);
  return builder.create<mlir::LLVM::ExtractValueOp>(
      source.loc, source.type.getBody()[index], source.aggregate,
      builder.getDenseI64ArrayAttr({static_cast<int64_t>(index)}));
}

std::optional<ExtractSource>
directExtractSource(mlir::UnrealizedConversionCastOp cast) {
  if (cast->getNumOperands() != 1 || cast->getNumResults() <= 1)
    return std::nullopt;
  auto sourceType = structTypeOf(cast.getOperand(0));
  if (!sourceType)
    return std::nullopt;
  return ExtractSource{cast, cast.getLoc(), cast.getOperand(0), sourceType};
}

std::optional<ExtractSource>
nestedExtractSource(mlir::UnrealizedConversionCastOp cast) {
  if (cast->getNumOperands() != 1 || cast->getNumResults() <= 1)
    return std::nullopt;
  mlir::Value nested = cast.getOperand(0);
  auto nestedResult = mlir::dyn_cast<mlir::OpResult>(nested);
  auto nestedCast = nested.getDefiningOp<mlir::UnrealizedConversionCastOp>();
  if (!nestedResult || !nestedCast || nestedCast->getNumOperands() != 1)
    return std::nullopt;
  auto sourceType = structTypeOf(nestedCast.getOperand(0));
  if (!sourceType)
    return std::nullopt;
  return ExtractSource{nestedCast, cast.getLoc(), nestedCast.getOperand(0),
                       sourceType};
}

mlir::Value materialize(mlir::Value value, mlir::OpBuilder &builder) {
  auto cast = value.getDefiningOp<mlir::UnrealizedConversionCastOp>();
  auto result = mlir::dyn_cast<mlir::OpResult>(value);
  if (!cast || !result)
    return {};

  if (cast->getNumOperands() == 1 && cast->getNumResults() == 1)
    if (structTypeOf(cast.getOperand(0)))
      return cast.getOperand(0);

  if (auto source = directExtractSource(cast))
    return extract(result, *source, builder);
  if (auto source = nestedExtractSource(cast))
    return extract(result, *source, builder);
  return {};
}

bool canMaterialize(mlir::Value value) {
  auto cast = value.getDefiningOp<mlir::UnrealizedConversionCastOp>();
  auto result = mlir::dyn_cast<mlir::OpResult>(value);
  if (!cast || !result)
    return false;

  if (cast->getNumOperands() == 1 && cast->getNumResults() == 1)
    if (structTypeOf(cast.getOperand(0)))
      return true;

  auto hasInBoundsResult = [&](const std::optional<ExtractSource> &source) {
    return source && result.getResultNumber() < source->type.getBody().size();
  };
  return hasInBoundsResult(directExtractSource(cast)) ||
         hasInBoundsResult(nestedExtractSource(cast));
}

bool rebuildAggregate(mlir::UnrealizedConversionCastOp cast,
                      mlir::OpBuilder &builder) {
  if (cast->getNumResults() != 1)
    return false;
  auto resultType =
      mlir::dyn_cast<mlir::LLVM::LLVMStructType>(cast.getResult(0).getType());
  if (!resultType || resultType.isOpaque() ||
      resultType.getBody().size() != cast->getNumOperands())
    return false;
  if (!llvm::all_of(cast.getOperands(), canMaterialize))
    return false;

  mlir::Value aggregate =
      builder.create<mlir::LLVM::UndefOp>(cast.getLoc(), resultType);
  for (auto [index, operand] : llvm::enumerate(cast.getOperands())) {
    mlir::Value descriptor = materialize(operand, builder);
    if (!descriptor)
      return false;
    builder.setInsertionPoint(cast);
    aggregate = builder.create<mlir::LLVM::InsertValueOp>(
        cast.getLoc(), resultType, aggregate, descriptor,
        builder.getDenseI64ArrayAttr({static_cast<int64_t>(index)}));
  }

  cast.getResult(0).replaceAllUsesWith(aggregate);
  cast.erase();
  return true;
}

bool foldSingle(mlir::UnrealizedConversionCastOp cast,
                mlir::OpBuilder &builder) {
  if (cast->getNumOperands() != 1 || cast->getNumResults() != 1 ||
      !mlir::isa<mlir::LLVM::LLVMStructType>(cast.getResult(0).getType()))
    return false;
  mlir::Value descriptor = materialize(cast.getOperand(0), builder);
  if (!descriptor)
    return false;
  cast.getResult(0).replaceAllUsesWith(descriptor);
  cast.erase();
  return true;
}

bool eraseDead(mlir::Operation *container) {
  llvm::SmallVector<mlir::UnrealizedConversionCastOp> deadCasts;
  container->walk([&](mlir::UnrealizedConversionCastOp cast) {
    if (cast->use_empty())
      deadCasts.push_back(cast);
  });
  for (auto cast : deadCasts)
    cast.erase();
  return !deadCasts.empty();
}

bool cleanup(mlir::Operation *container) {
  llvm::SmallVector<mlir::UnrealizedConversionCastOp> casts;
  container->walk(
      [&](mlir::UnrealizedConversionCastOp cast) { casts.push_back(cast); });

  bool changed = false;
  for (auto cast : casts) {
    if (!cast || cast->use_empty())
      continue;
    mlir::OpBuilder builder(cast);
    bool rewritten = rebuildAggregate(cast, builder);
    if (!rewritten)
      rewritten = foldSingle(cast, builder);
    changed |= rewritten;
  }

  return eraseDead(container) || changed;
}

} // namespace memref_descriptor_cast

} // namespace

namespace lowering::runtime::cleanup {

bool unreachableBlocks(mlir::ModuleOp module) {
  bool changed = false;
  bool everChanged = false;
  do {
    changed = false;
    llvm::SmallVector<mlir::Region *> regions;
    module.walk([&](mlir::Operation *op) {
      for (mlir::Region &region : op->getRegions())
        regions.push_back(&region);
    });
    for (mlir::Region *region : regions)
      changed |= eraseUnreachableBlocksInRegion(*region);
    everChanged |= changed;
  } while (changed);
  return everChanged;
}

bool pyBridgeCasts(mlir::Operation *container) {
  llvm::SmallVector<mlir::UnrealizedConversionCastOp> pending;

  container->walk([&](mlir::UnrealizedConversionCastOp cast) {
    if (cast->getNumOperands() != 1 || cast->getNumResults() != 1)
      return;
    mlir::Type inputType = cast->getOperand(0).getType();
    mlir::Type resultType = cast->getResultTypes().front();
    bool involvesPy = isPyType(inputType) || isPyType(resultType);
    if (!involvesPy)
      return;
    pending.push_back(cast);
  });

  for (auto cast : pending)
    cast.getResult(0).replaceAllUsesWith(cast.getOperand(0));
  for (auto cast : pending)
    if (cast && cast->use_empty())
      cast->erase();

  return !pending.empty();
}

bool pyMultiCasts(mlir::Operation *container) {
  llvm::SmallVector<mlir::UnrealizedConversionCastOp> casts;
  container->walk(
      [&](mlir::UnrealizedConversionCastOp cast) { casts.push_back(cast); });

  bool changed = false;
  for (auto cast : casts) {
    if (!cast)
      continue;
    if (cast->getNumOperands() == 1 && isPyType(cast.getOperand(0).getType()) &&
        cast->getNumResults() > 1) {
      auto source =
          cast.getOperand(0).getDefiningOp<mlir::UnrealizedConversionCastOp>();
      if (!source || source->getNumResults() != 1 ||
          !isPyType(source.getResult(0).getType()) ||
          source->getNumOperands() != cast->getNumResults())
        continue;
      bool compatible = true;
      for (auto [operand, result] :
           llvm::zip(source.getOperands(), cast.getResults())) {
        if (operand.getType() != result.getType()) {
          compatible = false;
          break;
        }
      }
      if (!compatible)
        continue;
      for (auto [result, operand] :
           llvm::zip(cast.getResults(), source.getOperands()))
        result.replaceAllUsesWith(operand);
      cast.erase();
      if (source->use_empty())
        source.erase();
      changed = true;
      continue;
    }
  }
  return changed;
}

bool voidPyReturns(mlir::Operation *container) {
  llvm::SmallVector<ReturnOp> pending;
  container->walk([&](ReturnOp ret) {
    auto parentFunc = ret->getParentOfType<mlir::func::FuncOp>();
    if (parentFunc && parentFunc.getFunctionType().getNumResults() == 0)
      pending.push_back(ret);
  });

  for (ReturnOp ret : pending) {
    mlir::OpBuilder builder(ret);
    NoneOp noneOp = nullptr;
    if (ret.getNumOperands() == 1)
      noneOp = ret.getOperand(0).getDefiningOp<NoneOp>();
    builder.create<mlir::func::ReturnOp>(ret.getLoc());
    ret.erase();
    if (noneOp && noneOp->use_empty())
      noneOp.erase();
  }

  return !pending.empty();
}

bool memrefDescriptorCasts(mlir::Operation *container) {
  return memref_descriptor_cast::cleanup(container);
}

} // namespace lowering::runtime::cleanup

} // namespace py
