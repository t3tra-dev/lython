#include "Passes/Runtime/Cleanup.h"

#include "Common/LoweringUtils.h"
#include "Common/Object.h"
#include "Common/RuntimeSupport.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
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

  llvm::SmallPtrSet<mlir::Block *, 16> reachable;
  llvm::SmallVector<mlir::Block *, 16> worklist;
  worklist.push_back(&region.front());
  while (!worklist.empty()) {
    mlir::Block *block = worklist.pop_back_val();
    if (!reachable.insert(block).second)
      continue;
    mlir::Operation *terminator = block->getTerminator();
    if (!terminator)
      continue;
    for (mlir::Block *successor : terminator->getSuccessors())
      worklist.push_back(successor);
  }

  bool changed = false;
  bool localChanged = false;
  do {
    localChanged = false;
    for (mlir::Block &block :
         llvm::make_early_inc_range(llvm::drop_begin(region.getBlocks()))) {
      if (reachable.contains(&block))
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

bool terminateEmptyFallthroughBlocksInRegion(mlir::Region &region) {
  if (region.empty())
    return false;

  mlir::Operation *parent = region.getParentOp();
  bool isLLVMRegion = mlir::isa_and_nonnull<mlir::LLVM::LLVMFuncOp>(parent);
  bool isFuncRegion = mlir::isa_and_nonnull<mlir::func::FuncOp>(parent);
  if (!isLLVMRegion && !isFuncRegion)
    return false;

  bool changed = false;
  for (mlir::Block &block : region.getBlocks()) {
    if (!block.empty())
      continue;
    auto next = std::next(block.getIterator());
    if (next == region.end() || next->getNumArguments() != 0)
      continue;

    mlir::OpBuilder builder(region.getContext());
    builder.setInsertionPointToEnd(&block);
    mlir::Location loc = parent ? parent->getLoc() : builder.getUnknownLoc();
    if (isLLVMRegion)
      builder.create<mlir::LLVM::BrOp>(loc, mlir::ValueRange{}, &*next);
    else
      builder.create<mlir::cf::BranchOp>(loc, &*next);
    changed = true;
  }
  return changed;
}

mlir::LLVM::LLVMFuncOp lookupCallee(mlir::LLVM::CallOp call) {
  auto callee = call.getCallee();
  if (!callee)
    return {};
  mlir::ModuleOp module = call->getParentOfType<mlir::ModuleOp>();
  if (!module)
    return {};
  return module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(*callee);
}

bool descriptorFieldsAt(llvm::ArrayRef<mlir::Type> types, unsigned base) {
  if (types.size() < base + 5)
    return false;
  return mlir::isa<mlir::LLVM::LLVMPointerType>(types[base]) &&
         mlir::isa<mlir::LLVM::LLVMPointerType>(types[base + 1]) &&
         types[base + 2].isInteger(64) && types[base + 3].isInteger(64) &&
         types[base + 4].isInteger(64);
}

bool acceptsDescriptorAt(mlir::LLVM::LLVMFuncOp callee, unsigned base) {
  return callee &&
         descriptorFieldsAt(callee.getFunctionType().getParams(), base);
}

bool acceptsDescriptorAt(mlir::LLVM::CallOp call, unsigned base) {
  return acceptsDescriptorAt(lookupCallee(call), base);
}

bool acceptsDescriptorChunks(mlir::LLVM::CallOp call,
                             llvm::ArrayRef<unsigned> bases) {
  mlir::LLVM::LLVMFuncOp callee = lookupCallee(call);
  if (!callee)
    return false;
  return llvm::all_of(
      bases, [&](unsigned base) { return acceptsDescriptorAt(callee, base); });
}

bool hasReleaseEffect(mlir::LLVM::CallOp call) {
  return call->hasAttr(OwnershipContractAttrs::kReleaseArgs) ||
         call->hasAttr(OwnershipContractAttrs::kAggregateRelease);
}

namespace memref_descriptor_cast {

struct ExtractSource {
  mlir::UnrealizedConversionCastOp insertionPoint;
  mlir::Location loc;
  mlir::Value aggregate;
  mlir::LLVM::LLVMStructType type;
};

void copyDiscardableAttrs(mlir::Operation *from, mlir::Operation *to) {
  if (!from || !to)
    return;
  for (const mlir::NamedAttribute &attr : from->getDiscardableAttrs())
    to->setDiscardableAttr(attr.getName(), attr.getValue());
}

void copyValueDiscardableAttrs(mlir::Value from, mlir::Value to) {
  copyDiscardableAttrs(from ? from.getDefiningOp() : nullptr,
                       to ? to.getDefiningOp() : nullptr);
}

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
  auto extract = builder.create<mlir::LLVM::ExtractValueOp>(
      source.loc, source.type.getBody()[index], source.aggregate,
      builder.getDenseI64ArrayAttr({static_cast<int64_t>(index)}));
  copyValueDiscardableAttrs(source.aggregate, extract.getResult());
  return extract;
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
    copyValueDiscardableAttrs(operand, aggregate);
  }
  copyDiscardableAttrs(cast.getOperation(), aggregate.getDefiningOp());

  cast.getResult(0).replaceAllUsesWith(aggregate);
  cast.erase();
  return true;
}

bool foldSingle(mlir::UnrealizedConversionCastOp cast,
                mlir::OpBuilder &builder) {
  if (cast->getNumOperands() != 1 || cast->getNumResults() != 1 ||
      !mlir::isa<mlir::LLVM::LLVMStructType>(cast.getResult(0).getType()))
    return false;
  auto resultType =
      mlir::cast<mlir::LLVM::LLVMStructType>(cast.getResult(0).getType());
  if (auto extract =
          cast.getOperand(0).getDefiningOp<mlir::LLVM::ExtractValueOp>()) {
    builder.setInsertionPoint(cast);
    auto replacement = builder.create<mlir::LLVM::ExtractValueOp>(
        cast.getLoc(), resultType, extract.getContainer(),
        extract.getPositionAttr());
    copyDiscardableAttrs(extract.getOperation(), replacement.getOperation());
    cast.getResult(0).replaceAllUsesWith(replacement);
    cast.erase();
    return true;
  }
  mlir::Value descriptor = materialize(cast.getOperand(0), builder);
  if (!descriptor)
    return false;
  copyDiscardableAttrs(cast.getOperation(), descriptor.getDefiningOp());
  cast.getResult(0).replaceAllUsesWith(descriptor);
  cast.erase();
  return true;
}

bool foldIdentity(mlir::UnrealizedConversionCastOp cast) {
  if (cast->getNumOperands() != 1 || cast->getNumResults() != 1)
    return false;
  if (cast.getOperand(0).getType() != cast.getResult(0).getType())
    return false;
  copyDiscardableAttrs(cast.getOperation(), cast.getOperand(0).getDefiningOp());
  cast.getResult(0).replaceAllUsesWith(cast.getOperand(0));
  cast.erase();
  return true;
}

bool foldExtractedDataPointerToDescriptor(
    mlir::UnrealizedConversionCastOp cast) {
  if (cast->getNumOperands() != 1 || cast->getNumResults() != 1)
    return false;
  auto resultType =
      mlir::dyn_cast<mlir::LLVM::LLVMStructType>(cast.getResult(0).getType());
  if (!resultType || resultType.isOpaque())
    return false;
  auto extract = cast.getOperand(0).getDefiningOp<mlir::LLVM::ExtractValueOp>();
  if (!extract || extract.getContainer().getType() != resultType ||
      !llvm::ArrayRef<int64_t>(extract.getPosition()).equals({1}))
    return false;
  copyDiscardableAttrs(cast.getOperation(),
                       extract.getContainer().getDefiningOp());
  cast.getResult(0).replaceAllUsesWith(extract.getContainer());
  cast.erase();
  return true;
}

bool foldBareMemRefPointer(mlir::UnrealizedConversionCastOp cast,
                           mlir::OpBuilder &builder) {
  if (cast->getNumOperands() != 1 || cast->getNumResults() != 1 ||
      !mlir::isa<mlir::LLVM::LLVMPointerType>(cast.getResult(0).getType()))
    return false;
  for (mlir::OpOperand &use : cast.getResult(0).getUses()) {
    auto call = mlir::dyn_cast<mlir::LLVM::CallOp>(use.getOwner());
    if (!call)
      continue;
    if (acceptsDescriptorAt(call, use.getOperandNumber()))
      return false;
  }
  auto sourceCast =
      cast.getOperand(0).getDefiningOp<mlir::UnrealizedConversionCastOp>();
  if (!sourceCast || sourceCast->getNumOperands() != 1 ||
      sourceCast->getNumResults() != 1 ||
      !object_abi::Type::isStorage(sourceCast.getResult(0).getType()))
    return false;
  mlir::Value descriptor = sourceCast.getOperand(0);
  if (!object_abi::Type::isLoweredStorage(descriptor.getType()))
    return false;
  auto descriptorType =
      mlir::cast<mlir::LLVM::LLVMStructType>(descriptor.getType());
  builder.setInsertionPoint(cast);
  mlir::Value alignedPtr = builder.create<mlir::LLVM::ExtractValueOp>(
      cast.getLoc(), descriptorType.getBody()[1], descriptor,
      builder.getDenseI64ArrayAttr({1}));
  cast.getResult(0).replaceAllUsesWith(alignedPtr);
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
    bool rewritten = foldIdentity(cast);
    if (!rewritten)
      rewritten = foldBareMemRefPointer(cast, builder);
    if (!rewritten)
      rewritten = foldExtractedDataPointerToDescriptor(cast);
    if (!rewritten)
      rewritten = rebuildAggregate(cast, builder);
    if (!rewritten)
      rewritten = foldSingle(cast, builder);
    changed |= rewritten;
  }

  return eraseDead(container) || changed;
}

} // namespace memref_descriptor_cast

namespace memref_atomic_bridge {

std::optional<mlir::LLVM::AtomicBinOp>
matchSimpleAtomicOp(mlir::arith::AtomicRMWKind kind) {
  switch (kind) {
  case mlir::arith::AtomicRMWKind::addf:
    return mlir::LLVM::AtomicBinOp::fadd;
  case mlir::arith::AtomicRMWKind::addi:
    return mlir::LLVM::AtomicBinOp::add;
  case mlir::arith::AtomicRMWKind::assign:
    return mlir::LLVM::AtomicBinOp::xchg;
  case mlir::arith::AtomicRMWKind::maximumf:
    return mlir::LLVM::AtomicBinOp::fmax;
  case mlir::arith::AtomicRMWKind::maxs:
    return mlir::LLVM::AtomicBinOp::max;
  case mlir::arith::AtomicRMWKind::maxu:
    return mlir::LLVM::AtomicBinOp::umax;
  case mlir::arith::AtomicRMWKind::minimumf:
    return mlir::LLVM::AtomicBinOp::fmin;
  case mlir::arith::AtomicRMWKind::mins:
    return mlir::LLVM::AtomicBinOp::min;
  case mlir::arith::AtomicRMWKind::minu:
    return mlir::LLVM::AtomicBinOp::umin;
  case mlir::arith::AtomicRMWKind::ori:
    return mlir::LLVM::AtomicBinOp::_or;
  case mlir::arith::AtomicRMWKind::andi:
    return mlir::LLVM::AtomicBinOp::_and;
  default:
    return std::nullopt;
  }
}

mlir::LLVM::AtomicOrdering ordering(mlir::Operation *op) {
  auto attr =
      op->getAttrOfType<mlir::StringAttr>(ThreadSafetyAttrs::kAtomicOrdering);
  if (!attr)
    return mlir::LLVM::AtomicOrdering::acq_rel;
  llvm::StringRef value = attr.getValue();
  if (value == ThreadSafetyAttrs::kOrderingMonotonic)
    return mlir::LLVM::AtomicOrdering::monotonic;
  if (value == ThreadSafetyAttrs::kOrderingAcquire)
    return mlir::LLVM::AtomicOrdering::acquire;
  if (value == ThreadSafetyAttrs::kOrderingRelease)
    return mlir::LLVM::AtomicOrdering::release;
  if (value == ThreadSafetyAttrs::kOrderingAcqRel)
    return mlir::LLVM::AtomicOrdering::acq_rel;
  if (value == ThreadSafetyAttrs::kOrderingSeqCst)
    return mlir::LLVM::AtomicOrdering::seq_cst;
  return mlir::LLVM::AtomicOrdering::acq_rel;
}

mlir::Value indexAsI64(mlir::Location loc, mlir::Value index,
                       mlir::OpBuilder &builder) {
  if (index.getType().isInteger(64))
    return index;
  if (!index.getType().isIndex())
    return {};
  if (auto constant = index.getDefiningOp<mlir::arith::ConstantIndexOp>()) {
    return builder.create<mlir::LLVM::ConstantOp>(
        loc, builder.getI64Type(), builder.getI64IntegerAttr(constant.value()));
  }
  return {};
}

bool lower(mlir::memref::AtomicRMWOp atomic) {
  auto kind = matchSimpleAtomicOp(atomic.getKind());
  if (!kind)
    return false;

  auto memrefType = mlir::dyn_cast<mlir::MemRefType>(atomic.getMemRefType());
  if (!memrefType || memrefType.getRank() != 1 ||
      atomic.getIndices().size() != 1)
    return false;

  auto cast =
      atomic.getMemref().getDefiningOp<mlir::UnrealizedConversionCastOp>();
  if (!cast || cast->getNumOperands() != 1 || cast->getNumResults() != 1)
    return false;

  mlir::Value descriptor = cast.getOperand(0);
  auto descriptorType =
      mlir::dyn_cast<mlir::LLVM::LLVMStructType>(descriptor.getType());
  if (!descriptorType || descriptorType.isOpaque() ||
      descriptorType.getBody().size() != 5)
    return false;

  mlir::OpBuilder builder(atomic);
  mlir::Location loc = atomic.getLoc();
  mlir::Type elementType = memrefType.getElementType();
  mlir::Value aligned = builder.create<mlir::LLVM::ExtractValueOp>(
      loc, descriptorType.getBody()[1], descriptor,
      builder.getDenseI64ArrayAttr({1}));
  mlir::Value offset = builder.create<mlir::LLVM::ExtractValueOp>(
      loc, builder.getI64Type(), descriptor, builder.getDenseI64ArrayAttr({2}));
  mlir::Value stride = builder.create<mlir::LLVM::ExtractValueOp>(
      loc, builder.getI64Type(), descriptor,
      builder.getDenseI64ArrayAttr({4, 0}));
  mlir::Value index = indexAsI64(loc, atomic.getIndices().front(), builder);
  if (!index)
    return false;

  mlir::Value scaled = builder.create<mlir::LLVM::MulOp>(loc, index, stride);
  mlir::Value linear = builder.create<mlir::LLVM::AddOp>(loc, offset, scaled);
  auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  mlir::Value address = builder.create<mlir::LLVM::GEPOp>(
      loc, ptrType, elementType, aligned,
      llvm::ArrayRef<mlir::LLVM::GEPArg>{linear});
  address.getDefiningOp()->setAttr(ContainerSafetyAttrs::kDescriptorData,
                                   mlir::UnitAttr::get(builder.getContext()));

  auto lowered = builder.create<mlir::LLVM::AtomicRMWOp>(
      loc, *kind, address, atomic.getValue(), ordering(atomic.getOperation()));
  memref_descriptor_cast::copyDiscardableAttrs(atomic.getOperation(),
                                               lowered.getOperation());
  atomic.getResult().replaceAllUsesWith(lowered.getResult());
  atomic.erase();
  if (cast && cast->use_empty())
    cast.erase();
  return true;
}

mlir::Value elementAddress(mlir::Location loc, mlir::Value descriptor,
                           mlir::MemRefType memrefType, mlir::Value index,
                           mlir::OpBuilder &builder) {
  auto descriptorType =
      mlir::dyn_cast<mlir::LLVM::LLVMStructType>(descriptor.getType());
  if (!descriptorType || descriptorType.isOpaque() ||
      descriptorType.getBody().size() != 5)
    return {};
  mlir::Value index64 = indexAsI64(loc, index, builder);
  if (!index64)
    return {};
  mlir::Value aligned = builder.create<mlir::LLVM::ExtractValueOp>(
      loc, descriptorType.getBody()[1], descriptor,
      builder.getDenseI64ArrayAttr({1}));
  mlir::Value offset = builder.create<mlir::LLVM::ExtractValueOp>(
      loc, builder.getI64Type(), descriptor, builder.getDenseI64ArrayAttr({2}));
  mlir::Value stride = builder.create<mlir::LLVM::ExtractValueOp>(
      loc, builder.getI64Type(), descriptor,
      builder.getDenseI64ArrayAttr({4, 0}));
  mlir::Value scaled = builder.create<mlir::LLVM::MulOp>(loc, index64, stride);
  mlir::Value linear = builder.create<mlir::LLVM::AddOp>(loc, offset, scaled);
  auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  mlir::Value address = builder.create<mlir::LLVM::GEPOp>(
      loc, ptrType, memrefType.getElementType(), aligned,
      llvm::ArrayRef<mlir::LLVM::GEPArg>{linear});
  address.getDefiningOp()->setAttr(ContainerSafetyAttrs::kDescriptorData,
                                   mlir::UnitAttr::get(builder.getContext()));
  return address;
}

mlir::Value bridgeDescriptor(mlir::Value memref) {
  auto cast = memref.getDefiningOp<mlir::UnrealizedConversionCastOp>();
  if (!cast || cast->getNumOperands() != 1 || cast->getNumResults() != 1)
    return {};
  return mlir::isa<mlir::LLVM::LLVMStructType>(cast.getOperand(0).getType())
             ? cast.getOperand(0)
             : mlir::Value{};
}

mlir::Value i64Constant(mlir::Location loc, int64_t value,
                        mlir::OpBuilder &builder) {
  return builder.create<mlir::LLVM::ConstantOp>(
      loc, builder.getI64Type(), builder.getI64IntegerAttr(value));
}

mlir::Value indexLikeAsI64(mlir::Location loc, mlir::Value value,
                           mlir::OpBuilder &builder) {
  if (value.getType().isInteger(64))
    return value;
  if (value.getType().isIndex())
    return builder.create<mlir::arith::IndexCastOp>(loc, builder.getI64Type(),
                                                    value);
  return {};
}

mlir::Value i64AsIndex(mlir::Location loc, mlir::Value value,
                       mlir::OpBuilder &builder) {
  if (value.getType().isIndex())
    return value;
  if (!value.getType().isInteger(64))
    return {};
  return builder.create<mlir::arith::IndexCastOp>(loc, builder.getIndexType(),
                                                  value);
}

mlir::Value extractDescriptorField(mlir::Location loc, mlir::Value descriptor,
                                   llvm::ArrayRef<int64_t> position,
                                   mlir::Type type, mlir::OpBuilder &builder) {
  return builder.create<mlir::LLVM::ExtractValueOp>(
      loc, type, descriptor, builder.getDenseI64ArrayAttr(position));
}

mlir::Value extractDescriptorI64(mlir::Location loc, mlir::Value descriptor,
                                 llvm::ArrayRef<int64_t> position,
                                 mlir::OpBuilder &builder) {
  return extractDescriptorField(loc, descriptor, position, builder.getI64Type(),
                                builder);
}

mlir::Value extractDescriptorIndex(mlir::Location loc, mlir::Value descriptor,
                                   llvm::ArrayRef<int64_t> position,
                                   mlir::OpBuilder &builder) {
  return i64AsIndex(
      loc, extractDescriptorI64(loc, descriptor, position, builder), builder);
}

mlir::Value alignedPointerIndex(mlir::Location loc, mlir::Value descriptor,
                                mlir::OpBuilder &builder) {
  auto descriptorType =
      mlir::dyn_cast<mlir::LLVM::LLVMStructType>(descriptor.getType());
  if (!descriptorType || descriptorType.isOpaque() ||
      descriptorType.getBody().size() != 5)
    return {};
  mlir::Value aligned = extractDescriptorField(
      loc, descriptor, {1}, descriptorType.getBody()[1], builder);
  ownership::Pointer::markNonObject(aligned);
  mlir::Value integer = builder.create<mlir::LLVM::PtrToIntOp>(
      loc, builder.getI64Type(), aligned);
  return i64AsIndex(loc, integer, builder);
}

mlir::Value
metadataDescriptor(mlir::memref::ExtractStridedMetadataOp metadata) {
  return bridgeDescriptor(metadata.getSource());
}

mlir::Value
pointerIndexDescriptor(mlir::memref::ExtractAlignedPointerAsIndexOp op) {
  if (mlir::Value descriptor = bridgeDescriptor(op.getSource()))
    return descriptor;
  auto metadata =
      op.getSource().getDefiningOp<mlir::memref::ExtractStridedMetadataOp>();
  if (!metadata || metadata.getBaseBuffer() != op.getSource())
    return {};
  return metadataDescriptor(metadata);
}

void copyAttrs(mlir::Operation *from, mlir::Operation *to) {
  if (!from || !to)
    return;
  for (const mlir::NamedAttribute &attr : from->getAttrs())
    to->setAttr(attr.getName(), attr.getValue());
}

bool lower(mlir::memref::ReinterpretCastOp cast) {
  auto resultType =
      mlir::dyn_cast<mlir::MemRefType>(cast.getResult().getType());
  if (!resultType || resultType.getRank() != 1)
    return false;
  mlir::Value sourceDescriptor = bridgeDescriptor(cast.getSource());
  if (!sourceDescriptor)
    return false;
  auto descriptorType =
      mlir::dyn_cast<mlir::LLVM::LLVMStructType>(sourceDescriptor.getType());
  if (!descriptorType || descriptorType.isOpaque() ||
      descriptorType.getBody().size() != 5)
    return false;

  mlir::OpBuilder builder(cast);
  mlir::Location loc = cast.getLoc();
  mlir::Value descriptor =
      builder.create<mlir::LLVM::UndefOp>(loc, descriptorType);
  mlir::Value allocated = builder.create<mlir::LLVM::ExtractValueOp>(
      loc, descriptorType.getBody()[0], sourceDescriptor,
      builder.getDenseI64ArrayAttr({0}));
  mlir::Value aligned = builder.create<mlir::LLVM::ExtractValueOp>(
      loc, descriptorType.getBody()[1], sourceDescriptor,
      builder.getDenseI64ArrayAttr({1}));
  descriptor = builder.create<mlir::LLVM::InsertValueOp>(
      loc, descriptorType, descriptor, allocated,
      builder.getDenseI64ArrayAttr({0}));
  descriptor = builder.create<mlir::LLVM::InsertValueOp>(
      loc, descriptorType, descriptor, aligned,
      builder.getDenseI64ArrayAttr({1}));

  mlir::Value offset = cast.isDynamicOffset(0)
                           ? indexLikeAsI64(loc, cast.getOffsets()[0], builder)
                           : i64Constant(loc, cast.getStaticOffset(0), builder);
  if (!offset)
    return false;
  descriptor = builder.create<mlir::LLVM::InsertValueOp>(
      loc, descriptorType, descriptor, offset,
      builder.getDenseI64ArrayAttr({2}));

  mlir::Value size = cast.isDynamicSize(0)
                         ? indexLikeAsI64(loc, cast.getSizes()[0], builder)
                         : i64Constant(loc, cast.getStaticSize(0), builder);
  mlir::Value stride = cast.isDynamicStride(0)
                           ? indexLikeAsI64(loc, cast.getStrides()[0], builder)
                           : i64Constant(loc, cast.getStaticStride(0), builder);
  if (!size || !stride)
    return false;
  descriptor = builder.create<mlir::LLVM::InsertValueOp>(
      loc, descriptorType, descriptor, size,
      builder.getDenseI64ArrayAttr({3, 0}));
  descriptor = builder.create<mlir::LLVM::InsertValueOp>(
      loc, descriptorType, descriptor, stride,
      builder.getDenseI64ArrayAttr({4, 0}));

  auto replacement = builder.create<mlir::UnrealizedConversionCastOp>(
      loc, cast.getResult().getType(), mlir::ValueRange{descriptor});
  copyAttrs(cast.getOperation(), replacement.getOperation());
  cast.getResult().replaceAllUsesWith(replacement.getResult(0));
  cast.erase();
  return true;
}

bool lower(mlir::memref::ExtractAlignedPointerAsIndexOp op) {
  mlir::Value descriptor = pointerIndexDescriptor(op);
  if (!descriptor)
    return false;
  mlir::OpBuilder builder(op);
  mlir::Value index = alignedPointerIndex(op.getLoc(), descriptor, builder);
  if (!index)
    return false;
  op.getResult().replaceAllUsesWith(index);
  op.erase();
  return true;
}

bool lower(mlir::memref::ExtractStridedMetadataOp metadata) {
  mlir::Value descriptor = metadataDescriptor(metadata);
  if (!descriptor)
    return false;
  if (!llvm::all_of(
          metadata.getBaseBuffer().getUsers(), [](mlir::Operation *user) {
            return mlir::isa<mlir::memref::ExtractAlignedPointerAsIndexOp>(
                user);
          }))
    return false;

  mlir::OpBuilder builder(metadata);
  mlir::Location loc = metadata.getLoc();
  mlir::Value offset = extractDescriptorIndex(loc, descriptor, {2}, builder);
  if (!offset)
    return false;
  metadata.getOffset().replaceAllUsesWith(offset);

  for (auto [index, size] : llvm::enumerate(metadata.getSizes())) {
    mlir::Value value = extractDescriptorIndex(
        loc, descriptor, {3, static_cast<int64_t>(index)}, builder);
    if (!value)
      return false;
    size.replaceAllUsesWith(value);
  }
  for (auto [index, stride] : llvm::enumerate(metadata.getStrides())) {
    mlir::Value value = extractDescriptorIndex(
        loc, descriptor, {4, static_cast<int64_t>(index)}, builder);
    if (!value)
      return false;
    stride.replaceAllUsesWith(value);
  }

  llvm::SmallVector<mlir::memref::ExtractAlignedPointerAsIndexOp> pointerUsers;
  for (mlir::Operation *user : metadata.getBaseBuffer().getUsers()) {
    if (auto pointer =
            mlir::dyn_cast<mlir::memref::ExtractAlignedPointerAsIndexOp>(user))
      pointerUsers.push_back(pointer);
  }
  for (mlir::memref::ExtractAlignedPointerAsIndexOp pointer : pointerUsers)
    if (pointer)
      lower(pointer);

  if (!metadata.getBaseBuffer().use_empty())
    return false;
  metadata.erase();
  return true;
}

bool lower(mlir::memref::LoadOp load) {
  auto memrefType = mlir::dyn_cast<mlir::MemRefType>(load.getMemRefType());
  if (!memrefType || memrefType.getRank() != 1 || load.getIndices().size() != 1)
    return false;
  mlir::Value descriptor = bridgeDescriptor(load.getMemref());
  if (!descriptor)
    return false;
  mlir::OpBuilder builder(load);
  mlir::Value address = elementAddress(load.getLoc(), descriptor, memrefType,
                                       load.getIndices().front(), builder);
  if (!address)
    return false;
  auto lowered = builder.create<mlir::LLVM::LoadOp>(
      load.getLoc(), memrefType.getElementType(), address, 0, false,
      load.getNontemporal());
  memref_descriptor_cast::copyDiscardableAttrs(load.getOperation(),
                                               lowered.getOperation());
  load.getResult().replaceAllUsesWith(lowered.getResult());
  load.erase();
  return true;
}

bool lower(mlir::memref::StoreOp store) {
  auto memrefType = mlir::dyn_cast<mlir::MemRefType>(store.getMemRefType());
  if (!memrefType || memrefType.getRank() != 1 ||
      store.getIndices().size() != 1)
    return false;
  mlir::Value descriptor = bridgeDescriptor(store.getMemref());
  if (!descriptor)
    return false;
  mlir::OpBuilder builder(store);
  mlir::Value address = elementAddress(store.getLoc(), descriptor, memrefType,
                                       store.getIndices().front(), builder);
  if (!address)
    return false;
  auto lowered = builder.create<mlir::LLVM::StoreOp>(store.getLoc(),
                                                     store.getValue(), address);
  memref_descriptor_cast::copyDiscardableAttrs(store.getOperation(),
                                               lowered.getOperation());
  store.erase();
  return true;
}

bool lower(mlir::memref::DeallocOp dealloc) {
  mlir::Value descriptor = bridgeDescriptor(dealloc.getMemref());
  if (!descriptor)
    return false;
  auto descriptorType =
      mlir::dyn_cast<mlir::LLVM::LLVMStructType>(descriptor.getType());
  if (!descriptorType || descriptorType.isOpaque() ||
      descriptorType.getBody().size() != 5)
    return false;

  mlir::ModuleOp module = dealloc->getParentOfType<mlir::ModuleOp>();
  if (!module)
    return false;
  mlir::FailureOr<mlir::LLVM::LLVMFuncOp> freeFunc =
      mlir::LLVM::lookupOrCreateFreeFn(module);
  if (mlir::failed(freeFunc))
    return false;

  mlir::OpBuilder builder(dealloc);
  mlir::Value allocated = builder.create<mlir::LLVM::ExtractValueOp>(
      dealloc.getLoc(), descriptorType.getBody()[0], descriptor,
      builder.getDenseI64ArrayAttr({0}));
  auto call = builder.create<mlir::LLVM::CallOp>(dealloc.getLoc(), *freeFunc,
                                                 mlir::ValueRange{allocated});
  copyAttrs(dealloc.getOperation(), call.getOperation());
  auto cast =
      dealloc.getMemref().getDefiningOp<mlir::UnrealizedConversionCastOp>();
  dealloc.erase();
  if (cast && cast->use_empty())
    cast.erase();
  return true;
}

bool cleanup(mlir::Operation *container) {
  llvm::SmallVector<mlir::memref::AtomicRMWOp> atomics;
  container->walk(
      [&](mlir::memref::AtomicRMWOp atomic) { atomics.push_back(atomic); });
  llvm::SmallVector<mlir::memref::LoadOp> loads;
  container->walk([&](mlir::memref::LoadOp load) { loads.push_back(load); });
  llvm::SmallVector<mlir::memref::StoreOp> stores;
  container->walk(
      [&](mlir::memref::StoreOp store) { stores.push_back(store); });
  llvm::SmallVector<mlir::memref::DeallocOp> deallocs;
  container->walk(
      [&](mlir::memref::DeallocOp dealloc) { deallocs.push_back(dealloc); });
  llvm::SmallVector<mlir::memref::ReinterpretCastOp> reinterpretCasts;
  container->walk([&](mlir::memref::ReinterpretCastOp cast) {
    reinterpretCasts.push_back(cast);
  });
  llvm::SmallVector<mlir::memref::ExtractAlignedPointerAsIndexOp>
      alignedPointers;
  container->walk([&](mlir::memref::ExtractAlignedPointerAsIndexOp pointer) {
    alignedPointers.push_back(pointer);
  });
  llvm::SmallVector<mlir::memref::ExtractStridedMetadataOp> metadataOps;
  container->walk([&](mlir::memref::ExtractStridedMetadataOp metadata) {
    metadataOps.push_back(metadata);
  });

  bool changed = false;
  for (mlir::memref::ReinterpretCastOp cast : reinterpretCasts)
    if (cast)
      changed |= lower(cast);
  for (mlir::memref::ExtractAlignedPointerAsIndexOp pointer : alignedPointers)
    if (pointer)
      changed |= lower(pointer);
  for (mlir::memref::ExtractStridedMetadataOp metadata : metadataOps)
    if (metadata)
      changed |= lower(metadata);
  for (mlir::memref::AtomicRMWOp atomic : atomics)
    if (atomic)
      changed |= lower(atomic);
  for (mlir::memref::LoadOp load : loads)
    if (load)
      changed |= lower(load);
  for (mlir::memref::StoreOp store : stores)
    if (store)
      changed |= lower(store);
  for (mlir::memref::DeallocOp dealloc : deallocs)
    if (dealloc)
      changed |= lower(dealloc);
  return changed;
}

} // namespace memref_atomic_bridge

namespace scalar_bridge_cast {

bool foldRoundTrip(mlir::UnrealizedConversionCastOp cast) {
  if (cast->getNumOperands() != 1 || cast->getNumResults() != 1)
    return false;
  auto source =
      cast.getOperand(0).getDefiningOp<mlir::UnrealizedConversionCastOp>();
  if (!source || source->getNumOperands() != 1 || source->getNumResults() != 1)
    return false;
  if (source.getOperand(0).getType() != cast.getResult(0).getType())
    return false;
  if (!(cast.getResult(0).getType().isInteger(64) ||
        cast.getResult(0).getType().isIndex()))
    return false;
  cast.getResult(0).replaceAllUsesWith(source.getOperand(0));
  cast.erase();
  if (source && source->use_empty())
    source.erase();
  return true;
}

bool cleanup(mlir::Operation *container) {
  llvm::SmallVector<mlir::UnrealizedConversionCastOp> casts;
  container->walk(
      [&](mlir::UnrealizedConversionCastOp cast) { casts.push_back(cast); });
  bool changed = false;
  for (mlir::UnrealizedConversionCastOp cast : casts)
    if (cast)
      changed |= foldRoundTrip(cast);
  return changed;
}

} // namespace scalar_bridge_cast

namespace memref_runtime_call {

llvm::StringRef canonicalCallee(llvm::StringRef name) { return name; }

mlir::Value descriptorSource(mlir::Value operand) {
  if (object_abi::Type::isLoweredStorage(operand.getType()))
    return operand;
  if (mlir::isa<mlir::LLVM::LLVMPointerType>(operand.getType())) {
    auto ptrCast = operand.getDefiningOp<mlir::UnrealizedConversionCastOp>();
    if (!ptrCast || ptrCast->getNumOperands() != 1 ||
        ptrCast->getNumResults() != 1)
      return {};
    mlir::Value memrefValue = ptrCast.getOperand(0);
    if (!object_abi::Type::isStorage(memrefValue.getType()))
      return {};
    auto memrefCast =
        memrefValue.getDefiningOp<mlir::UnrealizedConversionCastOp>();
    if (!memrefCast || memrefCast->getNumOperands() != 1 ||
        memrefCast->getNumResults() != 1)
      return {};
    mlir::Value source = memrefCast.getOperand(0);
    if (object_abi::Type::isLoweredStorage(source.getType()))
      return source;
    return {};
  }
  auto cast = operand.getDefiningOp<mlir::UnrealizedConversionCastOp>();
  if (!cast || cast->getNumOperands() != 1 || cast->getNumResults() != 1)
    return {};
  if (!object_abi::Type::isStorage(cast.getResult(0).getType()))
    return {};
  mlir::Value source = cast.getOperand(0);
  if (!object_abi::Type::isLoweredStorage(source.getType()))
    return {};
  return source;
}

void appendDescriptorFields(mlir::Location loc, mlir::Value descriptor,
                            llvm::SmallVectorImpl<mlir::Value> &operands,
                            mlir::OpBuilder &builder) {
  auto type = mlir::cast<mlir::LLVM::LLVMStructType>(descriptor.getType());
  auto body = type.getBody();
  auto i64Type = builder.getI64Type();
  mlir::Value allocated = builder.create<mlir::LLVM::ExtractValueOp>(
      loc, body[0], descriptor, builder.getDenseI64ArrayAttr({0}));
  mlir::Value aligned = builder.create<mlir::LLVM::ExtractValueOp>(
      loc, body[1], descriptor, builder.getDenseI64ArrayAttr({1}));
  ownership::Pointer::markNonObject(allocated);
  ownership::Pointer::markNonObject(aligned);
  operands.push_back(allocated);
  operands.push_back(aligned);
  operands.push_back(builder.create<mlir::LLVM::ExtractValueOp>(
      loc, body[2], descriptor, builder.getDenseI64ArrayAttr({2})));
  operands.push_back(builder.create<mlir::LLVM::ExtractValueOp>(
      loc, i64Type, descriptor, builder.getDenseI64ArrayAttr({3, 0})));
  operands.push_back(builder.create<mlir::LLVM::ExtractValueOp>(
      loc, i64Type, descriptor, builder.getDenseI64ArrayAttr({4, 0})));
}

bool descriptorComponentPosition(unsigned component,
                                 llvm::ArrayRef<int64_t> position) {
  switch (component) {
  case 0:
    return position == llvm::ArrayRef<int64_t>{0};
  case 1:
    return position == llvm::ArrayRef<int64_t>{1};
  case 2:
    return position == llvm::ArrayRef<int64_t>{2};
  case 3:
    return position == llvm::ArrayRef<int64_t>{3, 0};
  case 4:
    return position == llvm::ArrayRef<int64_t>{4, 0};
  default:
    return false;
  }
}

mlir::Value descriptorFieldSource(mlir::OperandRange operands, unsigned base,
                                  unsigned fieldIndex,
                                  mlir::Value expectedDescriptor) {
  if (operands.size() < base + 5)
    return {};

  mlir::Value fieldValue;
  for (unsigned component = 0; component != 5; ++component) {
    auto extract =
        operands[base + component].getDefiningOp<mlir::LLVM::ExtractValueOp>();
    if (!extract ||
        !descriptorComponentPosition(component, extract.getPosition()))
      return {};
    if (fieldValue && fieldValue != extract.getContainer())
      return {};
    fieldValue = extract.getContainer();
  }

  auto field = fieldValue.getDefiningOp<mlir::LLVM::ExtractValueOp>();
  if (!field || field.getPosition() != llvm::ArrayRef<int64_t>{fieldIndex})
    return {};
  mlir::Value descriptor = field.getContainer();
  if (!object_abi::Type::isLoweredStorage(descriptor.getType()))
    return {};
  if (expectedDescriptor && descriptor != expectedDescriptor)
    return {};
  return descriptor;
}

void eraseDeadExtractTree(mlir::Value value) {
  auto extract = value.getDefiningOp<mlir::LLVM::ExtractValueOp>();
  if (!extract || !extract->use_empty())
    return;
  mlir::Value container = extract.getContainer();
  extract.erase();
  eraseDeadExtractTree(container);
}

bool rewritePartsDescriptorFromFields(mlir::LLVM::CallOp call) {
  auto callee = call.getCallee();
  if (!callee || call.getNumOperands() != 15 ||
      !acceptsDescriptorChunks(call, {0, 5, 10}))
    return false;

  mlir::Value descriptor =
      descriptorFieldSource(call.getOperands(), /*base=*/0, /*fieldIndex=*/0,
                            /*expectedDescriptor=*/{});
  if (!descriptor)
    return false;
  if (!descriptorFieldSource(call.getOperands(), /*base=*/5, /*fieldIndex=*/1,
                             descriptor) ||
      !descriptorFieldSource(call.getOperands(), /*base=*/10, /*fieldIndex=*/2,
                             descriptor))
    return false;

  auto descriptorExtract =
      descriptor.getDefiningOp<mlir::LLVM::ExtractValueOp>();
  if (!descriptorExtract ||
      descriptorExtract.getPosition() != llvm::ArrayRef<int64_t>{0})
    return false;
  mlir::Value aggregate = descriptorExtract.getContainer();
  auto aggregateType =
      mlir::dyn_cast<mlir::LLVM::LLVMStructType>(aggregate.getType());
  if (!aggregateType || aggregateType.isOpaque() ||
      aggregateType.getBody().size() < 3)
    return false;
  auto descriptorType = descriptor.getType();
  if (aggregateType.getBody()[0] != descriptorType ||
      aggregateType.getBody()[1] != descriptorType ||
      aggregateType.getBody()[2] != descriptorType)
    return false;

  mlir::OpBuilder builder(call);
  llvm::SmallVector<mlir::Value, 15> operands;
  appendDescriptorFields(call.getLoc(), descriptor, operands, builder);
  for (int64_t index : {1, 2}) {
    mlir::Value part = builder.create<mlir::LLVM::ExtractValueOp>(
        call.getLoc(), descriptorType, aggregate,
        builder.getDenseI64ArrayAttr({index}));
    appendDescriptorFields(call.getLoc(), part, operands, builder);
  }

  auto replacement = builder.create<mlir::LLVM::CallOp>(
      call.getLoc(), call->getResultTypes(), call.getCalleeAttr(), operands);
  memref_descriptor_cast::copyDiscardableAttrs(call.getOperation(),
                                               replacement.getOperation());
  if (hasReleaseEffect(call))
    replacement->setAttr(OwnershipContractAttrs::kAggregateRelease,
                         builder.getUnitAttr());
  llvm::SmallVector<mlir::Value, 15> oldOperands(call.getOperands());
  call->replaceAllUsesWith(replacement.getOperation());
  call.erase();
  for (mlir::Value oldOperand : oldOperands)
    eraseDeadExtractTree(oldOperand);
  return true;
}

bool rewrite(mlir::LLVM::CallOp call) {
  auto callee = call.getCallee();
  if (!callee)
    return false;

  if (rewritePartsDescriptorFromFields(call))
    return true;

  bool needsRewrite = false;
  unsigned calleeArg = 0;
  for (mlir::Value operand : call.getOperands()) {
    if (descriptorSource(operand) && acceptsDescriptorAt(call, calleeArg)) {
      needsRewrite = true;
      break;
    }
    ++calleeArg;
  }
  if (!needsRewrite)
    return false;

  mlir::OpBuilder builder(call);
  llvm::SmallVector<mlir::Value, 10> operands;
  calleeArg = 0;
  for (mlir::Value operand : call.getOperands()) {
    if (mlir::Value descriptor = descriptorSource(operand);
        descriptor && acceptsDescriptorAt(call, calleeArg)) {
      appendDescriptorFields(call.getLoc(), descriptor, operands, builder);
      calleeArg += 5;
      continue;
    }
    operands.push_back(operand);
    ++calleeArg;
  }

  auto calleeAttr = mlir::SymbolRefAttr::get(
      call->getContext(), canonicalCallee(*call.getCallee()));
  auto replacement = builder.create<mlir::LLVM::CallOp>(
      call.getLoc(), call->getResultTypes(), calleeAttr, operands);
  memref_descriptor_cast::copyDiscardableAttrs(call.getOperation(),
                                               replacement.getOperation());
  call->replaceAllUsesWith(replacement.getOperation());
  call.erase();
  return true;
}

mlir::LLVM::LLVMFuncOp lookupLLVMFunc(mlir::func::CallOp call) {
  mlir::ModuleOp module = call->getParentOfType<mlir::ModuleOp>();
  if (!module)
    return {};
  return module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(call.getCallee());
}

mlir::Value genericDescriptorSource(mlir::Value operand) {
  if (mlir::Value descriptor = descriptorSource(operand))
    return descriptor;
  return memref_atomic_bridge::bridgeDescriptor(operand);
}

bool rewrite(mlir::func::CallOp call) {
  if (call.getNumResults() != 0)
    return false;

  mlir::LLVM::LLVMFuncOp callee = lookupLLVMFunc(call);
  if (!callee)
    return false;

  bool needsRewrite = false;
  unsigned calleeArg = 0;
  for (mlir::Value operand : call.getOperands()) {
    if (genericDescriptorSource(operand) &&
        acceptsDescriptorAt(callee, calleeArg)) {
      needsRewrite = true;
      break;
    }
    ++calleeArg;
  }
  if (!needsRewrite)
    return false;

  mlir::OpBuilder builder(call);
  llvm::SmallVector<mlir::Value, 16> operands;
  calleeArg = 0;
  for (mlir::Value operand : call.getOperands()) {
    if (mlir::Value descriptor = genericDescriptorSource(operand);
        descriptor && acceptsDescriptorAt(callee, calleeArg)) {
      appendDescriptorFields(call.getLoc(), descriptor, operands, builder);
      calleeArg += 5;
      continue;
    }
    operands.push_back(operand);
    ++calleeArg;
  }

  auto calleeAttr = mlir::SymbolRefAttr::get(call->getContext(),
                                             canonicalCallee(call.getCallee()));
  auto replacement = builder.create<mlir::LLVM::CallOp>(
      call.getLoc(), mlir::TypeRange{}, calleeAttr, operands);
  memref_descriptor_cast::copyDiscardableAttrs(call.getOperation(),
                                               replacement.getOperation());
  call.erase();
  return true;
}

bool cleanup(mlir::Operation *container) {
  llvm::SmallVector<mlir::LLVM::CallOp> calls;
  container->walk([&](mlir::LLVM::CallOp call) { calls.push_back(call); });
  llvm::SmallVector<mlir::func::CallOp> funcCalls;
  container->walk([&](mlir::func::CallOp call) { funcCalls.push_back(call); });

  bool changed = false;
  for (mlir::LLVM::CallOp call : calls) {
    if (call)
      changed |= rewrite(call);
  }
  for (mlir::func::CallOp call : funcCalls) {
    if (call)
      changed |= rewrite(call);
  }
  return changed;
}

} // namespace memref_runtime_call

namespace pointer_roundtrip {

void copyAttr(mlir::Operation *from, mlir::Operation *to,
              llvm::StringRef attrName, bool overwrite) {
  if (!from || !to)
    return;
  if (!overwrite && to->hasAttr(attrName))
    return;
  if (mlir::Attribute attr = from->getAttr(attrName))
    to->setAttr(attrName, attr);
}

void copyPointerProvenance(mlir::Operation *from, mlir::Operation *to,
                           bool overwrite) {
  copyAttr(from, to, OwnershipContractAttrs::kAggregateSlotLoad, overwrite);
  copyAttr(from, to, OwnershipContractAttrs::kAggregateSlotGroup, overwrite);
  copyAttr(from, to, OwnershipContractAttrs::kAggregateSlotComponent,
           overwrite);
  copyAttr(from, to, OwnershipContractAttrs::kAggregateSlotIndex, overwrite);
  copyAttr(from, to, OwnershipContractAttrs::kNonObjectPointer, overwrite);
}

bool carriesPointerProvenance(mlir::Operation *op) {
  return op && (op->hasAttr(OwnershipContractAttrs::kAggregateSlotLoad) ||
                op->hasAttr(OwnershipContractAttrs::kAggregateSlotGroup) ||
                op->hasAttr(OwnershipContractAttrs::kNonObjectPointer));
}

bool canReceivePointerProvenance(mlir::Operation *op) {
  return mlir::isa<mlir::LLVM::GEPOp, mlir::LLVM::LoadOp, mlir::LLVM::StoreOp,
                   mlir::LLVM::CallOp, mlir::LLVM::InvokeOp>(op);
}

void propagateUserProvenance(mlir::Value replacement,
                             mlir::LLVM::IntToPtrOp intToPtr) {
  mlir::Operation *sourceDef = replacement.getDefiningOp();
  mlir::Operation *castOp = intToPtr.getOperation();
  for (mlir::Operation *user :
       llvm::make_early_inc_range(intToPtr.getResult().getUsers())) {
    if (!canReceivePointerProvenance(user))
      continue;
    if (carriesPointerProvenance(sourceDef))
      copyPointerProvenance(sourceDef, user, /*overwrite=*/true);
    copyPointerProvenance(castOp, user, /*overwrite=*/false);
  }
}

bool repairZeroSlotBase(mlir::LLVM::GEPOp gep) {
  if (!gep->hasAttr(OwnershipContractAttrs::kAggregateSlotLoad) ||
      !gep->hasAttr(OwnershipContractAttrs::kAggregateSlotGroup) ||
      !gep->hasAttr(OwnershipContractAttrs::kAggregateSlotComponent))
    return false;

  mlir::Operation *baseDef = gep.getBase().getDefiningOp();
  if (!baseDef || baseDef->hasAttr(OwnershipContractAttrs::kAggregateSlotLoad))
    return false;

  copyPointerProvenance(gep.getOperation(), baseDef, /*overwrite=*/false);
  baseDef->setAttr(
      OwnershipContractAttrs::kAggregateSlotIndex,
      mlir::IntegerAttr::get(mlir::IntegerType::get(gep->getContext(), 64), 0));
  return true;
}

bool fold(mlir::LLVM::IntToPtrOp intToPtr) {
  auto ptrToInt = intToPtr.getArg().getDefiningOp<mlir::LLVM::PtrToIntOp>();
  if (!ptrToInt)
    return false;

  mlir::Value replacement = ptrToInt.getArg();
  if (replacement.getType() != intToPtr.getResult().getType())
    return false;
  if (!replacement.getDefiningOp() &&
      carriesPointerProvenance(intToPtr.getOperation()))
    return false;

  copyPointerProvenance(intToPtr.getOperation(), replacement.getDefiningOp(),
                        /*overwrite=*/false);
  propagateUserProvenance(replacement, intToPtr);
  intToPtr.getResult().replaceAllUsesWith(replacement);
  intToPtr.erase();
  if (ptrToInt && ptrToInt->use_empty())
    ptrToInt.erase();
  return true;
}

bool cleanup(mlir::Operation *container) {
  llvm::SmallVector<mlir::LLVM::IntToPtrOp> casts;
  container->walk(
      [&](mlir::LLVM::IntToPtrOp intToPtr) { casts.push_back(intToPtr); });
  llvm::SmallVector<mlir::LLVM::GEPOp> geps;
  container->walk([&](mlir::LLVM::GEPOp gep) { geps.push_back(gep); });

  bool changed = false;
  for (mlir::LLVM::IntToPtrOp intToPtr : casts) {
    if (intToPtr)
      changed |= fold(intToPtr);
  }
  for (mlir::LLVM::GEPOp gep : geps) {
    if (gep)
      changed |= repairZeroSlotBase(gep);
  }
  return changed;
}

} // namespace pointer_roundtrip

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
      changed |= terminateEmptyFallthroughBlocksInRegion(*region);
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
  bool changed = memref_atomic_bridge::cleanup(container);
  changed |= scalar_bridge_cast::cleanup(container);
  return memref_descriptor_cast::cleanup(container) || changed;
}

bool memrefRuntimeCalls(mlir::Operation *container) {
  return memref_runtime_call::cleanup(container);
}

bool pointerRoundTrips(mlir::Operation *container) {
  return pointer_roundtrip::cleanup(container);
}

bool llvmFuncReturns(mlir::Operation *container) {
  llvm::SmallVector<mlir::func::ReturnOp> returns;
  container->walk([&](mlir::func::ReturnOp ret) {
    auto parentFunc = ret->getParentOfType<mlir::LLVM::LLVMFuncOp>();
    if (parentFunc && ret->getParentRegion() == &parentFunc.getBody())
      returns.push_back(ret);
  });

  for (mlir::func::ReturnOp ret : returns) {
    auto next = std::next(ret->getIterator());
    if (next != ret->getBlock()->end())
      ret->getBlock()->splitBlock(next);
    mlir::OpBuilder builder(ret);
    auto replacement =
        builder.create<mlir::LLVM::ReturnOp>(ret.getLoc(), ret.getOperands());
    memref_descriptor_cast::copyDiscardableAttrs(ret.getOperation(),
                                                 replacement.getOperation());
    ret.erase();
  }
  return !returns.empty();
}

bool finalBoundary(mlir::ModuleOp module) {
  bool changed = false;
  bool everChanged = false;
  do {
    changed = false;
    changed |= pyMultiCasts(module);
    changed |= memrefDescriptorCasts(module);
    changed |= memrefRuntimeCalls(module);
    // Runtime-call rewriting can expose descriptor materialization casts.
    changed |= memrefDescriptorCasts(module);
    changed |= pointerRoundTrips(module);
    changed |= llvmFuncReturns(module);
    everChanged |= changed;
  } while (changed);

  unreachableBlocks(module);
  return everChanged;
}

} // namespace lowering::runtime::cleanup

} // namespace py
