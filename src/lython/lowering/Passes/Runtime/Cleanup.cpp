#include "Passes/Runtime/Cleanup.h"

#include "Common/LoweringUtils.h"
#include "Common/Object.h"
#include "Common/RuntimeSupport.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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

bool cleanup(mlir::Operation *container) {
  llvm::SmallVector<mlir::LLVM::CallOp> calls;
  container->walk([&](mlir::LLVM::CallOp call) { calls.push_back(call); });

  bool changed = false;
  for (mlir::LLVM::CallOp call : calls) {
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

bool memrefRuntimeCalls(mlir::Operation *container) {
  return memref_runtime_call::cleanup(container);
}

bool pointerRoundTrips(mlir::Operation *container) {
  return pointer_roundtrip::cleanup(container);
}

} // namespace lowering::runtime::cleanup

} // namespace py
