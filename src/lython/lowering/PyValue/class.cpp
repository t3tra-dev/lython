#include "Common/LoweringUtils.h"
#include "Common/RuntimeSupport.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FormatVariadic.h"

#include <cstdint>
#include <string>

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

using namespace mlir;

namespace py {
namespace {

static ClassOp lookupStaticClass(Operation *from, ClassType classType) {
  StringAttr nameAttr =
      StringAttr::get(from->getContext(), classType.getClassName());
  for (Operation *symbolTableOp = from; symbolTableOp;
       symbolTableOp = symbolTableOp->getParentOp()) {
    if (!symbolTableOp->hasTrait<OpTrait::SymbolTable>())
      continue;
    if (Operation *symbol =
            SymbolTable::lookupSymbolIn(symbolTableOp, nameAttr))
      if (ClassOp classOp = dyn_cast<ClassOp>(symbol))
        return classOp;
  }
  return nullptr;
}

struct StaticClassFieldInfo {
  StringAttr name;
  Type logicalType;
  Type storageType;
};

struct StaticClassLayout {
  LLVM::LLVMStructType storageType;
  LLVM::LLVMStructType objectType;
  SmallVector<StaticClassFieldInfo, 8> fields;
};

static Type getStaticFieldStorageType(Type logicalType,
                                      const PyLLVMTypeConverter &typeConverter,
                                      MLIRContext *ctx) {
  auto getMemRefDescriptorType = [&](MemRefType memrefType) -> Type {
    if (memrefType.getRank() != 1)
      return Type();
    auto ptrType = LLVM::LLVMPointerType::get(ctx);
    auto i64Type = IntegerType::get(ctx, 64);
    auto sizesType = LLVM::LLVMArrayType::get(i64Type, 1);
    return LLVM::LLVMStructType::getLiteral(
        ctx, ArrayRef<Type>{ptrType, ptrType, i64Type, sizesType, sizesType});
  };

  SmallVector<Type, 4> convertedTypes;
  if (failed(typeConverter.convertType(logicalType, convertedTypes)) ||
      convertedTypes.empty())
    return Type();

  if (convertedTypes.size() == 1) {
    if (auto memrefType = dyn_cast<MemRefType>(convertedTypes.front()))
      return getMemRefDescriptorType(memrefType);
    return convertedTypes.front();
  }

  SmallVector<Type, 4> descriptorTypes;
  descriptorTypes.reserve(convertedTypes.size());
  for (Type converted : convertedTypes) {
    auto memrefType = dyn_cast<MemRefType>(converted);
    if (!memrefType)
      return Type();
    Type descriptorType = getMemRefDescriptorType(memrefType);
    if (!descriptorType)
      return Type();
    descriptorTypes.push_back(descriptorType);
  }
  return LLVM::LLVMStructType::getLiteral(ctx, descriptorTypes);
}

static bool needsStaticFieldRefCount(Type logicalType, Type storageType,
                                     const PyLLVMTypeConverter &typeConverter) {
  return isPyType(logicalType) &&
         !isa<ClassType, NoneType, BoolType>(logicalType) &&
         storageType == typeConverter.getPyObjectPtrType();
}

static ClassType getClassElementListType(Type logicalType) {
  auto listType = dyn_cast<ListType>(logicalType);
  if (!listType)
    return {};
  return dyn_cast<ClassType>(listType.getElementType());
}

static void collectClassElementTupleSlots(
    Type logicalType, SmallVectorImpl<std::pair<unsigned, ClassType>> &slots) {
  auto tupleType = dyn_cast<TupleType>(logicalType);
  if (!tupleType)
    return;
  for (auto [index, type] : llvm::enumerate(tupleType.getElementTypes()))
    if (auto classType = dyn_cast<ClassType>(type))
      slots.push_back({static_cast<unsigned>(index), classType});
}

static std::pair<ClassType, ClassType>
getClassElementDictTypes(Type logicalType) {
  auto dictType = dyn_cast<DictType>(logicalType);
  if (!dictType)
    return {};
  return {dyn_cast<ClassType>(dictType.getKeyType()),
          dyn_cast<ClassType>(dictType.getValueType())};
}

static FailureOr<StaticClassLayout>
getStaticClassLayout(Operation *op, ClassType classType,
                     const PyLLVMTypeConverter &typeConverter) {
  ClassOp classOp = lookupStaticClass(op, classType);
  if (!classOp) {
    op->emitError("unable to resolve class '")
        << classType.getClassName() << "'";
    return failure();
  }

  ArrayAttr fieldNamesAttr = classOp.getFieldNamesAttr();
  ArrayAttr fieldTypesAttr = classOp.getFieldTypesAttr();

  StaticClassLayout layout;
  if (!fieldNamesAttr && !fieldTypesAttr) {
    layout.storageType = LLVM::LLVMStructType::getLiteral(op->getContext(), {});
    layout.objectType = LLVM::LLVMStructType::getLiteral(
        op->getContext(),
        ArrayRef<Type>{layout.storageType,
                       IntegerType::get(op->getContext(), 1),
                       IntegerType::get(op->getContext(), 32),
                       IntegerType::get(op->getContext(), 64)});
    return layout;
  }
  if (!fieldNamesAttr || !fieldTypesAttr ||
      fieldNamesAttr.size() != fieldTypesAttr.size()) {
    op->emitError("class '")
        << classType.getClassName() << "' has malformed static field schema";
    return failure();
  }

  SmallVector<Type, 8> loweredFieldTypes;
  for (auto [nameAttr, typeAttr] : llvm::zip(fieldNamesAttr, fieldTypesAttr)) {
    auto stringAttr = dyn_cast<StringAttr>(nameAttr);
    auto mlirTypeAttr = dyn_cast<TypeAttr>(typeAttr);
    if (!stringAttr || !mlirTypeAttr) {
      op->emitError("class '")
          << classType.getClassName() << "' has malformed static field schema";
      return failure();
    }
    Type logicalType = mlirTypeAttr.getValue();
    Type storageType =
        getStaticFieldStorageType(logicalType, typeConverter, op->getContext());
    if (!storageType) {
      op->emitError("failed to convert field type ")
          << logicalType << " in class '" << classType.getClassName() << "'";
      return failure();
    }
    layout.fields.push_back({stringAttr, logicalType, storageType});
    loweredFieldTypes.push_back(storageType);
  }

  layout.storageType =
      LLVM::LLVMStructType::getLiteral(op->getContext(), loweredFieldTypes);
  layout.objectType = LLVM::LLVMStructType::getLiteral(
      op->getContext(),
      ArrayRef<Type>{layout.storageType, IntegerType::get(op->getContext(), 1),
                     IntegerType::get(op->getContext(), 32),
                     IntegerType::get(op->getContext(), 64)});
  return layout;
}

static FailureOr<std::pair<unsigned, StaticClassFieldInfo>>
lookupStaticClassField(Operation *op, ClassType classType,
                       const PyLLVMTypeConverter &typeConverter,
                       StringRef fieldName) {
  FailureOr<StaticClassLayout> layoutOr =
      getStaticClassLayout(op, classType, typeConverter);
  if (failed(layoutOr))
    return failure();

  for (auto [index, fieldInfo] : llvm::enumerate(layoutOr->fields))
    if (fieldInfo.name.getValue() == fieldName)
      return std::make_pair(static_cast<unsigned>(index), fieldInfo);

  op->emitError("class '") << classType.getClassName() << "' has no field '"
                           << fieldName << "'";
  return failure();
}

static std::string getStaticClassFieldHelperName(ClassType classType,
                                                 StringRef action,
                                                 unsigned fieldIndex) {
  return ("__ly_class_" + action + "_" + classType.getClassName() + "_" +
          std::to_string(fieldIndex))
      .str();
}

static Value getOrCreateStringLiteralPtr(Location loc, ModuleOp module,
                                         OpBuilder &builder,
                                         StringRef literal) {
  llvm::SmallString<32> symbolName("__ly_str_");
  auto hashValue = static_cast<uint64_t>(llvm::hash_value(literal));
  symbolName += llvm::formatv("{0:X}", hashValue).str();

  LLVM::GlobalOp global = module.lookupSymbol<LLVM::GlobalOp>(symbolName);
  if (!global) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(module.getBody());
    auto arrayType =
        LLVM::LLVMArrayType::get(builder.getI8Type(), literal.size() + 1);
    llvm::SmallString<32> storage(literal);
    storage.push_back('\0');
    global = builder.create<LLVM::GlobalOp>(loc, arrayType, /*isConstant=*/true,
                                            LLVM::Linkage::Internal, symbolName,
                                            builder.getStringAttr(storage));
  }

  auto ptrType = LLVM::LLVMPointerType::get(module.getContext());
  Value addr =
      builder.create<LLVM::AddressOfOp>(loc, ptrType, global.getSymNameAttr());
  Value zero = builder.create<LLVM::ConstantOp>(loc, builder.getI64Type(),
                                                builder.getI64IntegerAttr(0));
  return builder.create<LLVM::GEPOp>(loc, ptrType, global.getType(), addr,
                                     llvm::ArrayRef<Value>{zero, zero});
}

static Value createStaticClassSlot(Location loc,
                                   LLVM::LLVMStructType objectType,
                                   ConversionPatternRewriter &rewriter,
                                   Operation *anchor) {
  auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
  auto i64Type = rewriter.getI64Type();
  auto oneAttr = rewriter.getI64IntegerAttr(1);

  auto parentFunc = anchor->getParentOfType<func::FuncOp>();
  if (!parentFunc)
    return Value();

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(&parentFunc.getBody().front());
  Value one = rewriter.create<LLVM::ConstantOp>(loc, i64Type, oneAttr);
  return rewriter.create<LLVM::AllocaOp>(loc, ptrType, objectType, one,
                                         /*alignment=*/0);
}

static Value getStaticClassManagedFlagPtr(Location loc, Value object,
                                          const StaticClassLayout &layout,
                                          OpBuilder &builder) {
  auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());
  return builder.create<LLVM::GEPOp>(loc, ptrType, layout.objectType, object,
                                     ArrayRef<LLVM::GEPArg>{0, 1});
}

static Value getStaticClassRefcountPtr(Location loc, Value object,
                                       const StaticClassLayout &layout,
                                       OpBuilder &builder) {
  auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());
  return builder.create<LLVM::GEPOp>(loc, ptrType, layout.objectType, object,
                                     ArrayRef<LLVM::GEPArg>{0, 3});
}

static Value getStaticClassLockPtr(Location loc, Value object,
                                   const StaticClassLayout &layout,
                                   OpBuilder &builder) {
  auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());
  return builder.create<LLVM::GEPOp>(loc, ptrType, layout.objectType, object,
                                     ArrayRef<LLVM::GEPArg>{0, 2});
}

static LLVM::CallOp emitRuntimeCall(Location loc, ModuleOp module,
                                    OpBuilder &builder, StringRef runtimeSymbol,
                                    Type resultType, ValueRange args) {
  SmallVector<Type, 4> argTypes;
  argTypes.reserve(args.size());
  for (Value arg : args)
    argTypes.push_back(arg.getType());

  auto callee = getOrInsertLLVMFunc(loc, module, builder, runtimeSymbol,
                                    resultType, argTypes);
  auto calleeRef = SymbolRefAttr::get(module.getContext(), callee.getName());
  SmallVector<Type, 1> results;
  if (!isa<LLVM::LLVMVoidType>(resultType))
    results.push_back(resultType);
  return builder.create<LLVM::CallOp>(loc, TypeRange(results), calleeRef, args);
}

static void emitRuntimeVoidCall(Location loc, ModuleOp module,
                                OpBuilder &builder, StringRef runtimeSymbol,
                                ValueRange args) {
  emitRuntimeCall(loc, module, builder, runtimeSymbol,
                  LLVM::LLVMVoidType::get(builder.getContext()), args);
}

static void copyMemRefPrefix(Location loc, Value source, Value target,
                             Value count, OpBuilder &builder) {
  Value lower = createIndexConstant(loc, builder, 0);
  Value step = createIndexConstant(loc, builder, 1);
  auto loop = builder.create<scf::ForOp>(loc, lower, count, step);
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(loop.getBody());
    Value iv = loop.getInductionVar();
    Value value = builder.create<memref::LoadOp>(loc, source, iv);
    builder.create<memref::StoreOp>(loc, value, target, iv);
  }
}

static Value loadI64HeaderSlot(Location loc, Value header, int64_t slot,
                               OpBuilder &builder) {
  return builder.create<memref::LoadOp>(
      loc, header, createIndexConstant(loc, builder, slot));
}

static void storeI64HeaderSlot(Location loc, Value header, int64_t slot,
                               int64_t value, OpBuilder &builder) {
  builder.create<memref::StoreOp>(loc, createI64Constant(loc, builder, value),
                                  header,
                                  createIndexConstant(loc, builder, slot));
}

static Value staticMemRefSize(Location loc, MemRefType type,
                              OpBuilder &builder) {
  if (type.getRank() != 1 || ShapedType::isDynamic(type.getShape()[0]))
    return {};
  return createIndexConstant(loc, builder, type.getShape()[0]);
}

static FailureOr<SmallVector<Value>>
promoteListDescriptor(Location loc, ValueRange input,
                      ConversionPatternRewriter &rewriter) {
  if (input.size() != 2)
    return failure();
  auto headerType = dyn_cast<MemRefType>(input[0].getType());
  auto itemsType = dyn_cast<MemRefType>(input[1].getType());
  if (!headerType || !itemsType)
    return failure();
  Value headerSize = staticMemRefSize(loc, headerType, rewriter);
  if (!headerSize)
    return failure();
  Value managedHeader = rewriter.create<memref::AllocOp>(loc, headerType);
  Value size = loadI64HeaderSlot(loc, input[0], 0, rewriter);
  Value capacity = loadI64HeaderSlot(loc, input[0], 1, rewriter);
  Value sizeIndex =
      rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), size);
  Value capacityIndex = rewriter.create<arith::IndexCastOp>(
      loc, rewriter.getIndexType(), capacity);
  Value managedItems = rewriter.create<memref::AllocOp>(
      loc, itemsType, ValueRange{capacityIndex});
  copyMemRefPrefix(loc, input[0], managedHeader, headerSize, rewriter);
  copyMemRefPrefix(loc, input[1], managedItems, sizeIndex, rewriter);
  storeI64HeaderSlot(loc, managedHeader, 3, 1, rewriter);
  return SmallVector<Value>{managedHeader, managedItems};
}

static FailureOr<SmallVector<Value>>
promoteTupleDescriptor(Location loc, ValueRange input,
                       ConversionPatternRewriter &rewriter) {
  if (input.size() != 2)
    return failure();
  auto headerType = dyn_cast<MemRefType>(input[0].getType());
  auto itemsType = dyn_cast<MemRefType>(input[1].getType());
  if (!headerType || !itemsType)
    return failure();
  Value headerSize = staticMemRefSize(loc, headerType, rewriter);
  if (!headerSize)
    return failure();
  Value managedHeader = rewriter.create<memref::AllocOp>(loc, headerType);
  Value size = loadI64HeaderSlot(loc, input[0], 0, rewriter);
  Value sizeIndex =
      rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), size);
  Value managedItems =
      rewriter.create<memref::AllocOp>(loc, itemsType, ValueRange{sizeIndex});
  copyMemRefPrefix(loc, input[0], managedHeader, headerSize, rewriter);
  copyMemRefPrefix(loc, input[1], managedItems, sizeIndex, rewriter);
  storeI64HeaderSlot(loc, managedHeader, 2, 1, rewriter);
  return SmallVector<Value>{managedHeader, managedItems};
}

static FailureOr<SmallVector<Value>>
promoteDictDescriptor(Location loc, ValueRange input,
                      ConversionPatternRewriter &rewriter) {
  if (input.size() != 4)
    return failure();
  auto headerType = dyn_cast<MemRefType>(input[0].getType());
  auto keysType = dyn_cast<MemRefType>(input[1].getType());
  auto valuesType = dyn_cast<MemRefType>(input[2].getType());
  auto statesType = dyn_cast<MemRefType>(input[3].getType());
  if (!headerType || !keysType || !valuesType || !statesType)
    return failure();
  Value headerSize = staticMemRefSize(loc, headerType, rewriter);
  if (!headerSize)
    return failure();
  Value managedHeader = rewriter.create<memref::AllocOp>(loc, headerType);
  Value capacity = loadI64HeaderSlot(loc, input[0], 1, rewriter);
  Value capacityIndex = rewriter.create<arith::IndexCastOp>(
      loc, rewriter.getIndexType(), capacity);
  Value managedKeys = rewriter.create<memref::AllocOp>(
      loc, keysType, ValueRange{capacityIndex});
  Value managedValues = rewriter.create<memref::AllocOp>(
      loc, valuesType, ValueRange{capacityIndex});
  Value managedStates = rewriter.create<memref::AllocOp>(
      loc, statesType, ValueRange{capacityIndex});
  copyMemRefPrefix(loc, input[0], managedHeader, headerSize, rewriter);
  copyMemRefPrefix(loc, input[1], managedKeys, capacityIndex, rewriter);
  copyMemRefPrefix(loc, input[2], managedValues, capacityIndex, rewriter);
  copyMemRefPrefix(loc, input[3], managedStates, capacityIndex, rewriter);
  storeI64HeaderSlot(loc, managedHeader, 4, 1, rewriter);
  return SmallVector<Value>{managedHeader, managedKeys, managedValues,
                            managedStates};
}

// Atomic lowering policy:
// - Prefer memref.atomic_rmw for memref-backed storage when acq_rel RMW is the
//   intended semantics. This keeps buffer-like storage in higher-level MLIR.
// - Use LLVM atomics when the value is a raw LLVM pointer or when a specific
//   ordering weaker/stronger than memref.atomic_rmw's current acq_rel lowering
//   is required.
static Value emitMemRefAtomicRMW(Location loc, arith::AtomicRMWKind kind,
                                 Value memref, Value value, ValueRange indices,
                                 OpBuilder &builder) {
  if (!isa<MemRefType>(memref.getType()))
    return {};
  return builder.create<memref::AtomicRMWOp>(loc, kind, value, memref, indices);
}

static Value
emitOrderedIntegerAtomicRMW(Location loc, LLVM::AtomicBinOp llvmKind,
                            arith::AtomicRMWKind memrefKind, Value storage,
                            Value value, ValueRange indices,
                            LLVM::AtomicOrdering ordering, OpBuilder &builder) {
  if (ordering == LLVM::AtomicOrdering::acq_rel)
    if (Value result = emitMemRefAtomicRMW(loc, memrefKind, storage, value,
                                           indices, builder))
      return result;
  return builder.create<LLVM::AtomicRMWOp>(loc, llvmKind, storage, value,
                                           ordering);
}

static void emitOrderedAtomicStore(Location loc, Value storage, Value value,
                                   ValueRange indices,
                                   LLVM::AtomicOrdering ordering,
                                   OpBuilder &builder) {
  if (isa<MemRefType>(storage.getType()) &&
      ordering == LLVM::AtomicOrdering::acq_rel) {
    (void)emitMemRefAtomicRMW(loc, arith::AtomicRMWKind::assign, storage, value,
                              indices, builder);
    return;
  }
  unsigned alignment = 0;
  if (auto intType = dyn_cast<IntegerType>(value.getType()))
    alignment = std::max<unsigned>(1, intType.getWidth() / 8);
  builder.create<LLVM::StoreOp>(loc, value, storage, alignment,
                                /*isVolatile=*/false,
                                /*isNonTemporal=*/false,
                                /*isInvariantGroup=*/false, ordering);
}

static void emitStaticClassLockAcquire(Location loc, Value lockPtr,
                                       ModuleOp module, OpBuilder &builder) {
  (void)module;
  Block *currentBlock = builder.getInsertionBlock();
  auto insertionPoint = builder.getInsertionPoint();
  Region *region = builder.getInsertionBlock()->getParent();
  Block *loopBlock = builder.createBlock(region);
  Block *acquiredBlock = builder.createBlock(region);

  builder.setInsertionPoint(currentBlock, insertionPoint);
  builder.create<LLVM::BrOp>(loc, ValueRange{}, loopBlock);

  builder.setInsertionPointToStart(loopBlock);
  Value one = builder.create<LLVM::ConstantOp>(loc, builder.getI32Type(),
                                               builder.getI32IntegerAttr(1));
  Value previous = emitOrderedIntegerAtomicRMW(
      loc, LLVM::AtomicBinOp::xchg, arith::AtomicRMWKind::assign, lockPtr, one,
      ValueRange{}, LLVM::AtomicOrdering::acquire, builder);
  Value zero = builder.create<LLVM::ConstantOp>(loc, builder.getI32Type(),
                                                builder.getI32IntegerAttr(0));
  Value acquired = builder.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq,
                                                previous, zero);
  builder.create<LLVM::CondBrOp>(loc, acquired, acquiredBlock, loopBlock);

  builder.setInsertionPointToStart(acquiredBlock);
}

static void emitStaticClassLockRelease(Location loc, Value lockPtr,
                                       ModuleOp module, OpBuilder &builder) {
  (void)module;
  Value zero = builder.create<LLVM::ConstantOp>(loc, builder.getI32Type(),
                                                builder.getI32IntegerAttr(0));
  emitOrderedAtomicStore(loc, lockPtr, zero, ValueRange{},
                         LLVM::AtomicOrdering::release, builder);
}

static void emitStaticClassAtomicRefInc(Location loc, Value refcountPtr,
                                        ModuleOp module, OpBuilder &builder) {
  (void)module;
  Value one = builder.create<LLVM::ConstantOp>(loc, builder.getI64Type(),
                                               builder.getI64IntegerAttr(1));
  emitOrderedIntegerAtomicRMW(
      loc, LLVM::AtomicBinOp::add, arith::AtomicRMWKind::addi, refcountPtr, one,
      ValueRange{}, LLVM::AtomicOrdering::monotonic, builder);
}

static Value emitStaticClassAtomicRefDecAndTestZero(Location loc,
                                                    Value refcountPtr,
                                                    ModuleOp module,
                                                    OpBuilder &builder) {
  (void)module;
  Value one = builder.create<LLVM::ConstantOp>(loc, builder.getI64Type(),
                                               builder.getI64IntegerAttr(1));
  Value negOne = builder.create<LLVM::ConstantOp>(
      loc, builder.getI64Type(), builder.getI64IntegerAttr(-1));
  Value previous = emitOrderedIntegerAtomicRMW(
      loc, LLVM::AtomicBinOp::add, arith::AtomicRMWKind::addi, refcountPtr,
      negOne, ValueRange{}, LLVM::AtomicOrdering::acq_rel, builder);
  return builder.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, previous,
                                      one);
}

static void
emitStaticListFieldClassElementRefcount(Location loc, Value descriptor,
                                        ClassType elementType, ModuleOp module,
                                        OpBuilder &builder, StringRef suffix) {
  auto *parent = builder.getInsertionBlock()->getParent();
  auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());
  auto i64Type = builder.getI64Type();
  auto voidType = LLVM::LLVMVoidType::get(builder.getContext());
  auto helper = getOrInsertLLVMFunc(loc, module, builder,
                                    getClassHelperName(elementType, suffix),
                                    voidType, {ptrType});
  auto helperRef = SymbolRefAttr::get(module.getContext(), helper.getName());

  Value headerDescriptor = descriptor;
  Value itemsDescriptor = descriptor;
  if (auto structType = dyn_cast<LLVM::LLVMStructType>(descriptor.getType())) {
    if (!structType.isOpaque() && structType.getBody().size() == 2) {
      headerDescriptor = builder.create<LLVM::ExtractValueOp>(
          loc, structType.getBody()[0], descriptor,
          builder.getDenseI64ArrayAttr({0}));
      itemsDescriptor = builder.create<LLVM::ExtractValueOp>(
          loc, structType.getBody()[1], descriptor,
          builder.getDenseI64ArrayAttr({1}));
    }
  }

  Value headerData = builder.create<LLVM::ExtractValueOp>(
      loc, ptrType, headerDescriptor, builder.getDenseI64ArrayAttr({1}));
  Value itemData = builder.create<LLVM::ExtractValueOp>(
      loc, ptrType, itemsDescriptor, builder.getDenseI64ArrayAttr({1}));
  Value size = builder.create<LLVM::LoadOp>(loc, i64Type, headerData);
  Value zero = builder.create<LLVM::ConstantOp>(loc, i64Type,
                                                builder.getI64IntegerAttr(0));
  Value one = builder.create<LLVM::ConstantOp>(loc, i64Type,
                                               builder.getI64IntegerAttr(1));

  Block *currentBlock = builder.getInsertionBlock();
  Block *condBlock = builder.createBlock(parent);
  condBlock->addArgument(i64Type, loc);
  Block *bodyBlock = builder.createBlock(parent);
  Block *afterBlock = builder.createBlock(parent);
  builder.setInsertionPointToEnd(currentBlock);
  builder.create<LLVM::BrOp>(loc, ValueRange{zero}, condBlock);

  builder.setInsertionPointToStart(condBlock);
  Value iv = condBlock->getArgument(0);
  Value inBounds =
      builder.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::slt, iv, size);
  builder.create<LLVM::CondBrOp>(loc, inBounds, bodyBlock, afterBlock);

  builder.setInsertionPointToStart(bodyBlock);
  Value itemPtr = builder.create<LLVM::GEPOp>(loc, ptrType, i64Type, itemData,
                                              ArrayRef<LLVM::GEPArg>{iv});
  Value slot = builder.create<LLVM::LoadOp>(loc, i64Type, itemPtr);
  Value object = builder.create<LLVM::IntToPtrOp>(loc, ptrType, slot);
  builder.create<LLVM::CallOp>(loc, TypeRange{}, helperRef, ValueRange{object});
  Value next = builder.create<LLVM::AddOp>(loc, iv, one);
  builder.create<LLVM::BrOp>(loc, ValueRange{next}, condBlock);

  builder.setInsertionPointToStart(afterBlock);
}

static void emitClassSlotRefcountFromI64(Location loc, Value slot,
                                         ClassType elementType, ModuleOp module,
                                         OpBuilder &builder, StringRef suffix) {
  auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());
  auto voidType = LLVM::LLVMVoidType::get(builder.getContext());
  auto helper = getOrInsertLLVMFunc(loc, module, builder,
                                    getClassHelperName(elementType, suffix),
                                    voidType, {ptrType});
  auto helperRef = SymbolRefAttr::get(module.getContext(), helper.getName());
  Value object = builder.create<LLVM::IntToPtrOp>(loc, ptrType, slot);
  builder.create<LLVM::CallOp>(loc, TypeRange{}, helperRef, ValueRange{object});
}

static void emitStaticTupleFieldClassElementRefcount(
    Location loc, Value descriptor,
    ArrayRef<std::pair<unsigned, ClassType>> elementSlots, ModuleOp module,
    OpBuilder &builder, StringRef suffix) {
  auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());
  auto i64Type = builder.getI64Type();
  Value data = builder.create<LLVM::ExtractValueOp>(
      loc, ptrType, descriptor, builder.getDenseI64ArrayAttr({1}));
  for (auto [index, elementType] : elementSlots) {
    Value offset = builder.create<LLVM::ConstantOp>(
        loc, i64Type, builder.getI64IntegerAttr(3 + index));
    Value itemPtr = builder.create<LLVM::GEPOp>(loc, ptrType, i64Type, data,
                                                ArrayRef<LLVM::GEPArg>{offset});
    Value slot = builder.create<LLVM::LoadOp>(loc, i64Type, itemPtr);
    emitClassSlotRefcountFromI64(loc, slot, elementType, module, builder,
                                 suffix);
  }
}

static void emitStaticDictFieldClassElementRefcount(
    Location loc, Value descriptor, ClassType keyClassType,
    ClassType valueClassType, ModuleOp module, OpBuilder &builder,
    StringRef suffix) {
  auto *parent = builder.getInsertionBlock()->getParent();
  auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());
  auto i64Type = builder.getI64Type();
  auto extractMemRefDataPtr = [&](Value memrefDescriptor) -> Value {
    return builder.create<LLVM::ExtractValueOp>(
        loc, ptrType, memrefDescriptor, builder.getDenseI64ArrayAttr({1}));
  };
  Value keysDescriptor = builder.create<LLVM::ExtractValueOp>(
      loc, cast<LLVM::LLVMStructType>(descriptor.getType()).getBody()[1],
      descriptor, builder.getDenseI64ArrayAttr({1}));
  Value valuesDescriptor = builder.create<LLVM::ExtractValueOp>(
      loc, cast<LLVM::LLVMStructType>(descriptor.getType()).getBody()[2],
      descriptor, builder.getDenseI64ArrayAttr({2}));
  Value statesDescriptor = builder.create<LLVM::ExtractValueOp>(
      loc, cast<LLVM::LLVMStructType>(descriptor.getType()).getBody()[3],
      descriptor, builder.getDenseI64ArrayAttr({3}));
  Value keysData = extractMemRefDataPtr(keysDescriptor);
  Value valuesData = extractMemRefDataPtr(valuesDescriptor);
  Value statesData = extractMemRefDataPtr(statesDescriptor);
  auto i8Type = builder.getI8Type();
  Value occupiedState = builder.create<LLVM::ConstantOp>(
      loc, i8Type, builder.getIntegerAttr(i8Type, 1));
  Value zero = builder.create<LLVM::ConstantOp>(loc, i64Type,
                                                builder.getI64IntegerAttr(0));
  Value one = builder.create<LLVM::ConstantOp>(loc, i64Type,
                                               builder.getI64IntegerAttr(1));
  Value capacity = builder.create<LLVM::ConstantOp>(
      loc, i64Type, builder.getI64IntegerAttr(64));

  Block *currentBlock = builder.getInsertionBlock();
  Block *condBlock = builder.createBlock(parent);
  condBlock->addArgument(i64Type, loc);
  Block *bodyBlock = builder.createBlock(parent);
  Block *occupiedBlock = builder.createBlock(parent);
  Block *nextBlock = builder.createBlock(parent);
  Block *afterBlock = builder.createBlock(parent);
  builder.setInsertionPointToEnd(currentBlock);
  builder.create<LLVM::BrOp>(loc, ValueRange{zero}, condBlock);

  builder.setInsertionPointToStart(condBlock);
  Value iv = condBlock->getArgument(0);
  Value inBounds =
      builder.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::slt, iv, capacity);
  builder.create<LLVM::CondBrOp>(loc, inBounds, bodyBlock, afterBlock);

  builder.setInsertionPointToStart(bodyBlock);
  Value statePtr = builder.create<LLVM::GEPOp>(loc, ptrType, i8Type, statesData,
                                               ArrayRef<LLVM::GEPArg>{iv});
  Value state = builder.create<LLVM::LoadOp>(loc, i8Type, statePtr);
  Value occupied = builder.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq,
                                                state, occupiedState);
  builder.create<LLVM::CondBrOp>(loc, occupied, occupiedBlock, nextBlock);

  builder.setInsertionPointToStart(occupiedBlock);
  if (keyClassType) {
    Value keyPtr = builder.create<LLVM::GEPOp>(loc, ptrType, i64Type, keysData,
                                               ArrayRef<LLVM::GEPArg>{iv});
    Value keySlot = builder.create<LLVM::LoadOp>(loc, i64Type, keyPtr);
    emitClassSlotRefcountFromI64(loc, keySlot, keyClassType, module, builder,
                                 suffix);
  }
  if (valueClassType) {
    Value valuePtr = builder.create<LLVM::GEPOp>(
        loc, ptrType, i64Type, valuesData, ArrayRef<LLVM::GEPArg>{iv});
    Value valueSlot = builder.create<LLVM::LoadOp>(loc, i64Type, valuePtr);
    emitClassSlotRefcountFromI64(loc, valueSlot, valueClassType, module,
                                 builder, suffix);
  }
  builder.create<LLVM::BrOp>(loc, ValueRange{}, nextBlock);

  builder.setInsertionPointToStart(nextBlock);
  Value next = builder.create<LLVM::AddOp>(loc, iv, one);
  builder.create<LLVM::BrOp>(loc, ValueRange{next}, condBlock);

  builder.setInsertionPointToStart(afterBlock);
}

static void emitStaticClassFieldRuntimeCall(
    Location loc, Value object, const StaticClassLayout &layout,
    ModuleOp module, OpBuilder &builder,
    const PyLLVMTypeConverter &typeConverter, StringRef runtimeSymbol,
    bool includeContainerFields = true) {
  auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());
  auto voidType = LLVM::LLVMVoidType::get(builder.getContext());
  auto runtimeFn = getOrInsertLLVMFunc(loc, module, builder, runtimeSymbol,
                                       voidType, {ptrType});
  auto runtimeRef =
      SymbolRefAttr::get(module.getContext(), runtimeFn.getName());

  for (auto [index, fieldInfo] : llvm::enumerate(layout.fields)) {
    bool needsObjectRefcount = needsStaticFieldRefCount(
        fieldInfo.logicalType, fieldInfo.storageType, typeConverter);
    ClassType listElementClass =
        includeContainerFields ? getClassElementListType(fieldInfo.logicalType)
                               : ClassType{};
    SmallVector<std::pair<unsigned, ClassType>, 4> tupleElementClasses;
    if (includeContainerFields)
      collectClassElementTupleSlots(fieldInfo.logicalType, tupleElementClasses);
    auto [dictKeyClass, dictValueClass] =
        includeContainerFields ? getClassElementDictTypes(fieldInfo.logicalType)
                               : std::pair<ClassType, ClassType>{};
    if (!needsObjectRefcount && !listElementClass &&
        tupleElementClasses.empty() && !dictKeyClass && !dictValueClass)
      continue;

    Value fieldPtr = builder.create<LLVM::GEPOp>(
        loc, ptrType, layout.storageType, object,
        ArrayRef<LLVM::GEPArg>{0, static_cast<int32_t>(index)});
    Value fieldValue =
        builder.create<LLVM::LoadOp>(loc, fieldInfo.storageType, fieldPtr);
    if (needsObjectRefcount) {
      builder.create<LLVM::CallOp>(loc, TypeRange{}, runtimeRef,
                                   ValueRange{fieldValue});
      continue;
    }

    StringRef suffix;
    if (runtimeSymbol == RuntimeSymbols::kIncRef) {
      suffix = "incref";
    } else if (runtimeSymbol == RuntimeSymbols::kDecRef) {
      suffix = "decref";
    } else {
      continue;
    }
    if (listElementClass)
      emitStaticListFieldClassElementRefcount(loc, fieldValue, listElementClass,
                                              module, builder, suffix);
    if (!tupleElementClasses.empty())
      emitStaticTupleFieldClassElementRefcount(
          loc, fieldValue, tupleElementClasses, module, builder, suffix);
    if (dictKeyClass || dictValueClass)
      emitStaticDictFieldClassElementRefcount(loc, fieldValue, dictKeyClass,
                                              dictValueClass, module, builder,
                                              suffix);
  }
}

static void emitStaticFieldRuntimeCall(Location loc, Value fieldValue,
                                       ModuleOp module, OpBuilder &builder,
                                       StringRef runtimeSymbol) {
  emitRuntimeVoidCall(loc, module, builder, runtimeSymbol,
                      ValueRange{fieldValue});
}

static Value getStaticClassObjectSize(Location loc,
                                      const StaticClassLayout &layout,
                                      OpBuilder &builder) {
  auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());
  auto i64Type = builder.getI64Type();
  Value nullPtr = builder.create<LLVM::ZeroOp>(loc, ptrType);
  Value one = builder.create<LLVM::ConstantOp>(loc, i64Type,
                                               builder.getI64IntegerAttr(1));
  Value endPtr = builder.create<LLVM::GEPOp>(loc, ptrType, layout.objectType,
                                             nullPtr, ArrayRef<Value>{one});
  return builder.create<LLVM::PtrToIntOp>(loc, i64Type, endPtr);
}

static LLVM::LLVMFuncOp getOrCreateStaticClassRetainHelper(
    Location loc, ModuleOp module, ClassType classType,
    const StaticClassLayout &layout, OpBuilder &builder,
    const PyLLVMTypeConverter &typeConverter) {
  auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());
  auto voidType = LLVM::LLVMVoidType::get(builder.getContext());
  std::string helperName = getClassHelperName(classType, "incref");
  auto fn = getOrInsertLLVMFunc(loc, module, builder, helperName, voidType,
                                {ptrType});
  fn.setVisibility(SymbolTable::Visibility::Private);
  if (!fn.getBody().empty())
    return fn;

  Block *entry = fn.addEntryBlock(builder);
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(entry);

  Value object = entry->getArgument(0);
  Value nullPtr = builder.create<LLVM::ZeroOp>(loc, ptrType);
  Value isNull = builder.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq,
                                              object, nullPtr);
  Block *retBlock = builder.createBlock(&fn.getBody());
  Block *dispatchBlock = builder.createBlock(&fn.getBody());
  builder.setInsertionPointToEnd(entry);
  builder.create<LLVM::CondBrOp>(loc, isNull, retBlock, dispatchBlock);

  builder.setInsertionPointToStart(dispatchBlock);
  Value managedPtr = getStaticClassManagedFlagPtr(loc, object, layout, builder);
  Value managed =
      builder.create<LLVM::LoadOp>(loc, builder.getI1Type(), managedPtr);
  Block *managedBlock = builder.createBlock(&fn.getBody());
  Block *localBlock = builder.createBlock(&fn.getBody());
  builder.setInsertionPointToEnd(dispatchBlock);
  builder.create<LLVM::CondBrOp>(loc, managed, managedBlock, localBlock);

  builder.setInsertionPointToStart(managedBlock);
  Value refcountPtr = getStaticClassRefcountPtr(loc, object, layout, builder);
  emitStaticClassAtomicRefInc(loc, refcountPtr, module, builder);
  builder.create<LLVM::BrOp>(loc, ValueRange{}, retBlock);

  builder.setInsertionPointToStart(localBlock);
  emitStaticClassFieldRuntimeCall(loc, object, layout, module, builder,
                                  typeConverter, RuntimeSymbols::kIncRef,
                                  /*includeContainerFields=*/false);
  builder.create<LLVM::BrOp>(loc, ValueRange{}, retBlock);

  builder.setInsertionPointToStart(retBlock);
  builder.create<LLVM::ReturnOp>(loc, ValueRange{});
  return fn;
}

static LLVM::LLVMFuncOp getOrCreateStaticClassReleaseHelper(
    Location loc, ModuleOp module, ClassType classType,
    const StaticClassLayout &layout, OpBuilder &builder,
    const PyLLVMTypeConverter &typeConverter) {
  auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());
  auto voidType = LLVM::LLVMVoidType::get(builder.getContext());
  std::string helperName = getClassHelperName(classType, "decref");
  auto fn = getOrInsertLLVMFunc(loc, module, builder, helperName, voidType,
                                {ptrType});
  fn.setVisibility(SymbolTable::Visibility::Private);
  if (!fn.getBody().empty())
    return fn;

  Block *entry = fn.addEntryBlock(builder);
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(entry);

  Value object = entry->getArgument(0);
  Value nullPtr = builder.create<LLVM::ZeroOp>(loc, ptrType);
  Value isNull = builder.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq,
                                              object, nullPtr);
  Block *retBlock = builder.createBlock(&fn.getBody());
  Block *dispatchBlock = builder.createBlock(&fn.getBody());
  builder.setInsertionPointToEnd(entry);
  builder.create<LLVM::CondBrOp>(loc, isNull, retBlock, dispatchBlock);

  builder.setInsertionPointToStart(dispatchBlock);
  Value managedPtr = getStaticClassManagedFlagPtr(loc, object, layout, builder);
  Value managed =
      builder.create<LLVM::LoadOp>(loc, builder.getI1Type(), managedPtr);
  Block *managedBlock = builder.createBlock(&fn.getBody());
  Block *localBlock = builder.createBlock(&fn.getBody());
  builder.setInsertionPointToEnd(dispatchBlock);
  builder.create<LLVM::CondBrOp>(loc, managed, managedBlock, localBlock);

  builder.setInsertionPointToStart(localBlock);
  emitStaticClassFieldRuntimeCall(loc, object, layout, module, builder,
                                  typeConverter, RuntimeSymbols::kDecRef,
                                  /*includeContainerFields=*/false);
  builder.create<LLVM::BrOp>(loc, ValueRange{}, retBlock);

  builder.setInsertionPointToStart(managedBlock);
  Value refcountPtr = getStaticClassRefcountPtr(loc, object, layout, builder);
  Value isZero =
      emitStaticClassAtomicRefDecAndTestZero(loc, refcountPtr, module, builder);
  Block *destroyBlock = builder.createBlock(&fn.getBody());
  Block *managedRetBlock = builder.createBlock(&fn.getBody());
  builder.setInsertionPointToEnd(managedBlock);
  builder.create<LLVM::CondBrOp>(loc, isZero, destroyBlock, managedRetBlock);

  builder.setInsertionPointToStart(destroyBlock);
  emitStaticClassFieldRuntimeCall(loc, object, layout, module, builder,
                                  typeConverter, RuntimeSymbols::kDecRef);
  auto freeFn = getOrInsertLLVMFunc(
      loc, module, builder, RuntimeSymbols::kMemFree, voidType, {ptrType});
  auto freeRef = SymbolRefAttr::get(module.getContext(), freeFn.getName());
  builder.create<LLVM::CallOp>(loc, TypeRange{}, freeRef, ValueRange{object});
  builder.create<LLVM::BrOp>(loc, ValueRange{}, retBlock);

  builder.setInsertionPointToStart(managedRetBlock);
  builder.create<LLVM::BrOp>(loc, ValueRange{}, retBlock);

  builder.setInsertionPointToStart(retBlock);
  builder.create<LLVM::ReturnOp>(loc, ValueRange{});
  return fn;
}

static LLVM::LLVMFuncOp getOrCreateStaticClassLocalDestroyHelper(
    Location loc, ModuleOp module, ClassType classType,
    const StaticClassLayout &layout, OpBuilder &builder,
    const PyLLVMTypeConverter &typeConverter) {
  auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());
  auto voidType = LLVM::LLVMVoidType::get(builder.getContext());
  std::string helperName = getClassHelperName(classType, "destroy_local");
  auto fn = getOrInsertLLVMFunc(loc, module, builder, helperName, voidType,
                                {ptrType});
  fn.setVisibility(SymbolTable::Visibility::Private);
  if (!fn.getBody().empty())
    return fn;

  Block *entry = fn.addEntryBlock(builder);
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(entry);

  Value object = entry->getArgument(0);
  Value nullPtr = builder.create<LLVM::ZeroOp>(loc, ptrType);
  Value isNull = builder.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq,
                                              object, nullPtr);
  Block *retBlock = builder.createBlock(&fn.getBody());
  Block *destroyBlock = builder.createBlock(&fn.getBody());
  builder.setInsertionPointToEnd(entry);
  builder.create<LLVM::CondBrOp>(loc, isNull, retBlock, destroyBlock);

  builder.setInsertionPointToStart(destroyBlock);
  emitStaticClassFieldRuntimeCall(loc, object, layout, module, builder,
                                  typeConverter, RuntimeSymbols::kDecRef);
  builder.create<LLVM::BrOp>(loc, ValueRange{}, retBlock);

  builder.setInsertionPointToStart(retBlock);
  builder.create<LLVM::ReturnOp>(loc, ValueRange{});
  return fn;
}

static void getOrCreateStaticClassReprHelper(Location loc, ModuleOp module,
                                             ClassType classType,
                                             OpBuilder &builder) {
  auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());
  std::string helperName = getClassHelperName(classType, "repr");
  std::string customName = (classType.getClassName() + ".__repr__").str();
  if (auto customFunc = module.lookupSymbol<func::FuncOp>(customName)) {
    if (module.lookupSymbol<func::FuncOp>(helperName))
      return;
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(module.getBody());
    auto funcType =
        FunctionType::get(module.getContext(), {ptrType}, {ptrType});
    auto wrapper = builder.create<func::FuncOp>(loc, helperName, funcType);
    wrapper.setVisibility(SymbolTable::Visibility::Private);
    wrapper->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
    Block *entry = wrapper.addEntryBlock();
    builder.setInsertionPointToStart(entry);
    auto call = builder.create<func::CallOp>(loc, customFunc,
                                             ValueRange{entry->getArgument(0)});
    builder.create<func::ReturnOp>(loc, call.getResults());
    return;
  }

  auto fn =
      getOrInsertLLVMFunc(loc, module, builder, helperName, ptrType, {ptrType});
  fn.setVisibility(SymbolTable::Visibility::Private);
  if (!fn.getBody().empty())
    return;

  Block *entry = fn.addEntryBlock(builder);
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(entry);
  Value namePtr = getOrCreateStringLiteralPtr(loc, module, builder,
                                              classType.getClassName());
  auto reprFn =
      getOrInsertLLVMFunc(loc, module, builder, RuntimeSymbols::kClassReprNamed,
                          ptrType, {ptrType, ptrType});
  auto reprRef = SymbolRefAttr::get(module.getContext(), reprFn.getName());
  auto call =
      builder.create<LLVM::CallOp>(loc, TypeRange{ptrType}, reprRef,
                                   ValueRange{namePtr, entry->getArgument(0)});
  builder.create<LLVM::ReturnOp>(loc, call.getResults());
}

static LLVM::LLVMFuncOp getOrCreateStaticClassEqHelper(Location loc,
                                                       ModuleOp module,
                                                       ClassType classType,
                                                       OpBuilder &builder) {
  auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());
  auto i1Type = builder.getI1Type();
  std::string helperName = getClassHelperName(classType, "eq");
  auto fn = getOrInsertLLVMFunc(loc, module, builder, helperName, i1Type,
                                {ptrType, ptrType});
  fn.setVisibility(SymbolTable::Visibility::Private);
  if (!fn.getBody().empty())
    return fn;

  Block *entry = fn.addEntryBlock(builder);
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(entry);
  Value equal = builder.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq,
                                             entry->getArgument(0),
                                             entry->getArgument(1));
  builder.create<LLVM::ReturnOp>(loc, equal);
  return fn;
}

static int64_t getLLVMStorageElementBytes(Type elementType) {
  if (isa<IntegerType>(elementType))
    return std::max<int64_t>(1, cast<IntegerType>(elementType).getWidth() / 8);
  if (isa<mlir::FloatType>(elementType))
    return std::max<int64_t>(1,
                             cast<mlir::FloatType>(elementType).getWidth() / 8);
  return 0;
}

static Value cloneLLVMRank1MemRefDescriptor(Location loc, Value descriptor,
                                            MemRefType memrefType,
                                            FlatSymbolRefAttr allocRef,
                                            OpBuilder &builder) {
  auto descriptorType = cast<LLVM::LLVMStructType>(descriptor.getType());
  Type elementType = memrefType.getElementType();
  int64_t elementBytes = getLLVMStorageElementBytes(elementType);
  if (elementBytes <= 0)
    return descriptor;

  auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());
  auto i64Type = builder.getI64Type();
  Value oldData = builder.create<LLVM::ExtractValueOp>(
      loc, ptrType, descriptor, builder.getDenseI64ArrayAttr({1}));
  Value size = builder.create<LLVM::ExtractValueOp>(
      loc, i64Type, descriptor, builder.getDenseI64ArrayAttr({3, 0}));
  Value bytesPerElement = builder.create<LLVM::ConstantOp>(
      loc, i64Type, builder.getI64IntegerAttr(elementBytes));
  Value byteSize = builder.create<LLVM::MulOp>(loc, size, bytesPerElement);
  auto allocCall = builder.create<LLVM::CallOp>(loc, TypeRange{ptrType},
                                                allocRef, ValueRange{byteSize});
  Value newData = allocCall.getResult();

  Value zero = builder.create<LLVM::ConstantOp>(loc, i64Type,
                                                builder.getI64IntegerAttr(0));
  Value one = builder.create<LLVM::ConstantOp>(loc, i64Type,
                                               builder.getI64IntegerAttr(1));
  Value cloned = builder.create<LLVM::UndefOp>(loc, descriptorType);
  cloned = builder.create<LLVM::InsertValueOp>(
      loc, descriptorType, cloned, newData, builder.getDenseI64ArrayAttr({0}));
  cloned = builder.create<LLVM::InsertValueOp>(
      loc, descriptorType, cloned, newData, builder.getDenseI64ArrayAttr({1}));
  cloned = builder.create<LLVM::InsertValueOp>(
      loc, descriptorType, cloned, zero, builder.getDenseI64ArrayAttr({2}));
  cloned = builder.create<LLVM::InsertValueOp>(
      loc, descriptorType, cloned, size, builder.getDenseI64ArrayAttr({3, 0}));
  cloned = builder.create<LLVM::InsertValueOp>(
      loc, descriptorType, cloned, one, builder.getDenseI64ArrayAttr({4, 0}));

  auto *parent = builder.getInsertionBlock()->getParent();
  Block *currentBlock = builder.getInsertionBlock();
  Block *condBlock = builder.createBlock(parent);
  condBlock->addArgument(i64Type, loc);
  Block *bodyBlock = builder.createBlock(parent);
  Block *afterBlock = builder.createBlock(parent);
  builder.setInsertionPointToEnd(currentBlock);
  builder.create<LLVM::BrOp>(loc, ValueRange{zero}, condBlock);

  builder.setInsertionPointToStart(condBlock);
  Value iv = condBlock->getArgument(0);
  Value inBounds =
      builder.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::slt, iv, size);
  builder.create<LLVM::CondBrOp>(loc, inBounds, bodyBlock, afterBlock);

  builder.setInsertionPointToStart(bodyBlock);
  Value srcPtr = builder.create<LLVM::GEPOp>(loc, ptrType, elementType, oldData,
                                             ArrayRef<LLVM::GEPArg>{iv});
  Value dstPtr = builder.create<LLVM::GEPOp>(loc, ptrType, elementType, newData,
                                             ArrayRef<LLVM::GEPArg>{iv});
  Value item = builder.create<LLVM::LoadOp>(loc, elementType, srcPtr);
  builder.create<LLVM::StoreOp>(loc, item, dstPtr);
  Value next = builder.create<LLVM::AddOp>(loc, iv, one);
  builder.create<LLVM::BrOp>(loc, ValueRange{next}, condBlock);

  builder.setInsertionPointToStart(afterBlock);
  return cloned;
}

static void setClonedContainerOwnershipMarker(Location loc,
                                              Value headerDescriptor,
                                              int64_t markerSlot,
                                              OpBuilder &builder) {
  auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());
  auto i64Type = builder.getI64Type();
  Value data = builder.create<LLVM::ExtractValueOp>(
      loc, ptrType, headerDescriptor, builder.getDenseI64ArrayAttr({1}));
  Value slot = builder.create<LLVM::ConstantOp>(
      loc, i64Type, builder.getI64IntegerAttr(markerSlot));
  Value ptr = builder.create<LLVM::GEPOp>(loc, ptrType, i64Type, data,
                                          ArrayRef<LLVM::GEPArg>{slot});
  Value one = builder.create<LLVM::ConstantOp>(loc, i64Type,
                                               builder.getI64IntegerAttr(1));
  builder.create<LLVM::StoreOp>(loc, one, ptr);
}

static Value cloneStaticContainerFieldStorage(Location loc, Value storageValue,
                                              Type logicalType,
                                              FlatSymbolRefAttr allocRef,
                                              OpBuilder &builder) {
  SmallVector<Type, 4> convertedTypes;
  auto appendDescriptor = [&](unsigned index, MemRefType memrefType,
                              Value &result) {
    auto storageStruct = cast<LLVM::LLVMStructType>(result.getType());
    Type partType = storageStruct.getBody()[index];
    Value descriptor = builder.create<LLVM::ExtractValueOp>(
        loc, partType, result, builder.getDenseI64ArrayAttr({index}));
    Value cloned = cloneLLVMRank1MemRefDescriptor(loc, descriptor, memrefType,
                                                  allocRef, builder);
    result = builder.create<LLVM::InsertValueOp>(
        loc, storageStruct, result, cloned,
        builder.getDenseI64ArrayAttr({index}));
    return cloned;
  };

  Value result = storageValue;
  if (auto listType = dyn_cast<ListType>(logicalType)) {
    auto headerType = getListHeaderMemRefType(builder.getContext());
    auto itemsType =
        getListItemsMemRefType(listType.getElementType(), builder.getContext());
    if (!itemsType)
      return storageValue;
    Value header = appendDescriptor(0, headerType, result);
    appendDescriptor(1, itemsType, result);
    setClonedContainerOwnershipMarker(loc, header, 3, builder);
    return result;
  }
  if (auto tupleType = dyn_cast<TupleType>(logicalType)) {
    Value header = appendDescriptor(
        0, getTupleHeaderMemRefType(builder.getContext()), result);
    appendDescriptor(
        1, getTupleItemsMemRefType(tupleType, builder.getContext()), result);
    setClonedContainerOwnershipMarker(loc, header, 2, builder);
    return result;
  }
  if (auto dictType = dyn_cast<DictType>(logicalType)) {
    auto keysType = getDictKeysMemRefType(dictType, builder.getContext());
    auto valuesType = getDictValuesMemRefType(dictType, builder.getContext());
    if (!keysType || !valuesType)
      return storageValue;
    Value header = appendDescriptor(
        0, getDictHeaderMemRefType(builder.getContext()), result);
    appendDescriptor(1, keysType, result);
    appendDescriptor(2, valuesType, result);
    appendDescriptor(3, getDictStatesMemRefType(builder.getContext()), result);
    setClonedContainerOwnershipMarker(loc, header, 4, builder);
    return result;
  }
  return storageValue;
}

static void deepCloneStaticContainerFields(Location loc, Value object,
                                           const StaticClassLayout &layout,
                                           FlatSymbolRefAttr allocRef,
                                           OpBuilder &builder) {
  auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());
  for (auto [index, field] : llvm::enumerate(layout.fields)) {
    if (!isa<ListType, TupleType, DictType>(field.logicalType))
      continue;
    Value fieldPtr = builder.create<LLVM::GEPOp>(
        loc, ptrType, layout.storageType, object,
        ArrayRef<LLVM::GEPArg>{0, static_cast<int32_t>(index)});
    Value fieldValue =
        builder.create<LLVM::LoadOp>(loc, field.storageType, fieldPtr);
    Value cloned = cloneStaticContainerFieldStorage(
        loc, fieldValue, field.logicalType, allocRef, builder);
    builder.create<LLVM::StoreOp>(loc, cloned, fieldPtr);
  }
}

static LLVM::LLVMFuncOp getOrCreateStaticClassPromoteHelper(
    Location loc, ModuleOp module, ClassType classType,
    const StaticClassLayout &layout, OpBuilder &builder,
    const PyLLVMTypeConverter &typeConverter) {
  auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());
  auto i64Type = builder.getI64Type();
  std::string helperName = getClassHelperName(classType, "promote");
  auto fn =
      getOrInsertLLVMFunc(loc, module, builder, helperName, ptrType, {ptrType});
  fn.setVisibility(SymbolTable::Visibility::Private);
  if (!fn.getBody().empty())
    return fn;

  auto retainHelper = getOrCreateStaticClassRetainHelper(
      loc, module, classType, layout, builder, typeConverter);
  auto retainRef =
      SymbolRefAttr::get(module.getContext(), retainHelper.getName());
  auto allocFn = getOrInsertLLVMFunc(
      loc, module, builder, RuntimeSymbols::kMemAlloc, ptrType, {i64Type});
  auto allocRef =
      FlatSymbolRefAttr::get(module.getContext(), allocFn.getName());

  Block *entry = fn.addEntryBlock(builder);
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(entry);

  Value object = entry->getArgument(0);
  Value nullPtr = builder.create<LLVM::ZeroOp>(loc, ptrType);
  Value isNull = builder.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq,
                                              object, nullPtr);
  Block *retNullBlock = builder.createBlock(&fn.getBody());
  Block *dispatchBlock = builder.createBlock(&fn.getBody());
  builder.setInsertionPointToEnd(entry);
  builder.create<LLVM::CondBrOp>(loc, isNull, retNullBlock, dispatchBlock);

  builder.setInsertionPointToStart(retNullBlock);
  builder.create<LLVM::ReturnOp>(loc, nullPtr);

  builder.setInsertionPointToStart(dispatchBlock);
  Value managedPtr = getStaticClassManagedFlagPtr(loc, object, layout, builder);
  Value managed =
      builder.create<LLVM::LoadOp>(loc, builder.getI1Type(), managedPtr);
  Block *managedBlock = builder.createBlock(&fn.getBody());
  Block *copyBlock = builder.createBlock(&fn.getBody());
  builder.setInsertionPointToEnd(dispatchBlock);
  builder.create<LLVM::CondBrOp>(loc, managed, managedBlock, copyBlock);

  builder.setInsertionPointToStart(managedBlock);
  builder.create<LLVM::CallOp>(loc, TypeRange{}, retainRef, ValueRange{object});
  builder.create<LLVM::ReturnOp>(loc, object);

  builder.setInsertionPointToStart(copyBlock);
  Value sizeBytes = getStaticClassObjectSize(loc, layout, builder);
  auto allocCall = builder.create<LLVM::CallOp>(
      loc, TypeRange{ptrType}, allocRef, ValueRange{sizeBytes});
  Value managedObject = allocCall.getResult();
  Value storageValue =
      builder.create<LLVM::LoadOp>(loc, layout.storageType, object);
  builder.create<LLVM::StoreOp>(loc, storageValue, managedObject);
  deepCloneStaticContainerFields(loc, managedObject, layout, allocRef, builder);
  Value newManagedPtr =
      getStaticClassManagedFlagPtr(loc, managedObject, layout, builder);
  Value trueValue = builder.create<LLVM::ConstantOp>(loc, builder.getI1Type(),
                                                     builder.getBoolAttr(true));
  builder.create<LLVM::StoreOp>(loc, trueValue, newManagedPtr);
  Value lockPtr = getStaticClassLockPtr(loc, managedObject, layout, builder);
  Value zeroLock = builder.create<LLVM::ConstantOp>(
      loc, builder.getI32Type(), builder.getI32IntegerAttr(0));
  builder.create<LLVM::StoreOp>(loc, zeroLock, lockPtr);
  Value refcountPtr =
      getStaticClassRefcountPtr(loc, managedObject, layout, builder);
  Value one = builder.create<LLVM::ConstantOp>(loc, i64Type,
                                               builder.getI64IntegerAttr(1));
  builder.create<LLVM::StoreOp>(loc, one, refcountPtr);
  emitStaticClassFieldRuntimeCall(loc, managedObject, layout, module, builder,
                                  typeConverter, RuntimeSymbols::kIncRef);
  builder.create<LLVM::ReturnOp>(loc, managedObject);
  return fn;
}

static LLVM::LLVMFuncOp getOrCreateStaticClassGetFieldHelper(
    Location loc, ModuleOp module, ClassType classType,
    const StaticClassLayout &layout, unsigned fieldIndex, OpBuilder &builder,
    const PyLLVMTypeConverter &typeConverter) {
  assert(fieldIndex < layout.fields.size() && "field index out of range");
  const StaticClassFieldInfo &fieldInfo = layout.fields[fieldIndex];
  auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());
  auto i1Type = builder.getI1Type();
  std::string helperName =
      getStaticClassFieldHelperName(classType, "getfield", fieldIndex);
  auto fn = getOrInsertLLVMFunc(loc, module, builder, helperName,
                                fieldInfo.storageType, {ptrType, i1Type});
  fn.setVisibility(SymbolTable::Visibility::Private);
  if (!fn.getBody().empty())
    return fn;

  bool needsRefcount = needsStaticFieldRefCount(
      fieldInfo.logicalType, fieldInfo.storageType, typeConverter);

  Block *entry = fn.addEntryBlock(builder);
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(entry);

  Value object = entry->getArgument(0);
  Value borrowLocal = entry->getArgument(1);
  Value managedPtr = getStaticClassManagedFlagPtr(loc, object, layout, builder);
  Value managed = builder.create<LLVM::LoadOp>(loc, i1Type, managedPtr);
  Block *managedBlock = builder.createBlock(&fn.getBody());
  Block *localBlock = builder.createBlock(&fn.getBody());
  builder.setInsertionPointToEnd(entry);
  builder.create<LLVM::CondBrOp>(loc, managed, managedBlock, localBlock);

  builder.setInsertionPointToStart(localBlock);
  Value localFieldPtr = builder.create<LLVM::GEPOp>(
      loc, ptrType, layout.storageType, object,
      ArrayRef<LLVM::GEPArg>{0, static_cast<int32_t>(fieldIndex)});
  Value localValue =
      builder.create<LLVM::LoadOp>(loc, fieldInfo.storageType, localFieldPtr);
  if (!needsRefcount) {
    builder.create<LLVM::ReturnOp>(loc, localValue);
  } else {
    Block *localBorrowBlock = builder.createBlock(&fn.getBody());
    Block *localRetainBlock = builder.createBlock(&fn.getBody());
    builder.setInsertionPointToEnd(localBlock);
    builder.create<LLVM::CondBrOp>(loc, borrowLocal, localBorrowBlock,
                                   localRetainBlock);

    builder.setInsertionPointToStart(localBorrowBlock);
    builder.create<LLVM::ReturnOp>(loc, localValue);

    builder.setInsertionPointToStart(localRetainBlock);
    emitStaticFieldRuntimeCall(loc, localValue, module, builder,
                               RuntimeSymbols::kIncRef);
    builder.create<LLVM::ReturnOp>(loc, localValue);
  }

  builder.setInsertionPointToStart(managedBlock);
  Value lockPtr = getStaticClassLockPtr(loc, object, layout, builder);
  emitStaticClassLockAcquire(loc, lockPtr, module, builder);
  Value managedFieldPtr = builder.create<LLVM::GEPOp>(
      loc, ptrType, layout.storageType, object,
      ArrayRef<LLVM::GEPArg>{0, static_cast<int32_t>(fieldIndex)});
  Value managedValue =
      builder.create<LLVM::LoadOp>(loc, fieldInfo.storageType, managedFieldPtr);
  if (needsRefcount) {
    emitStaticFieldRuntimeCall(loc, managedValue, module, builder,
                               RuntimeSymbols::kIncRef);
  }
  emitStaticClassLockRelease(loc, lockPtr, module, builder);
  builder.create<LLVM::ReturnOp>(loc, managedValue);
  return fn;
}

static LLVM::LLVMFuncOp getOrCreateStaticClassSetFieldHelper(
    Location loc, ModuleOp module, ClassType classType,
    const StaticClassLayout &layout, unsigned fieldIndex, OpBuilder &builder,
    const PyLLVMTypeConverter &typeConverter) {
  assert(fieldIndex < layout.fields.size() && "field index out of range");
  const StaticClassFieldInfo &fieldInfo = layout.fields[fieldIndex];
  auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());
  auto i1Type = builder.getI1Type();
  auto voidType = LLVM::LLVMVoidType::get(builder.getContext());
  std::string helperName =
      getStaticClassFieldHelperName(classType, "setfield", fieldIndex);
  auto fn =
      getOrInsertLLVMFunc(loc, module, builder, helperName, voidType,
                          {ptrType, fieldInfo.storageType, i1Type, i1Type});
  fn.setVisibility(SymbolTable::Visibility::Private);
  if (!fn.getBody().empty())
    return fn;

  bool needsRefcount = needsStaticFieldRefCount(
      fieldInfo.logicalType, fieldInfo.storageType, typeConverter);

  Block *entry = fn.addEntryBlock(builder);
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(entry);

  Value object = entry->getArgument(0);
  Value newValue = entry->getArgument(1);
  Value retainNewValue = entry->getArgument(2);
  Value skipOldValueDrop = entry->getArgument(3);
  Block *dispatchBlock = builder.createBlock(&fn.getBody());
  if (needsRefcount) {
    Block *retainBlock = builder.createBlock(&fn.getBody());
    builder.setInsertionPointToEnd(entry);
    builder.create<LLVM::CondBrOp>(loc, retainNewValue, retainBlock,
                                   dispatchBlock);
    builder.setInsertionPointToStart(retainBlock);
    emitStaticFieldRuntimeCall(loc, newValue, module, builder,
                               RuntimeSymbols::kIncRef);
    builder.create<LLVM::BrOp>(loc, ValueRange{}, dispatchBlock);
  } else {
    builder.setInsertionPointToEnd(entry);
    builder.create<LLVM::BrOp>(loc, ValueRange{}, dispatchBlock);
  }

  builder.setInsertionPointToStart(dispatchBlock);
  Value managedPtr = getStaticClassManagedFlagPtr(loc, object, layout, builder);
  Value managed = builder.create<LLVM::LoadOp>(loc, i1Type, managedPtr);
  Block *managedBlock = builder.createBlock(&fn.getBody());
  Block *localBlock = builder.createBlock(&fn.getBody());
  builder.setInsertionPointToEnd(dispatchBlock);
  builder.create<LLVM::CondBrOp>(loc, managed, managedBlock, localBlock);

  builder.setInsertionPointToStart(localBlock);
  Value localFieldPtr = builder.create<LLVM::GEPOp>(
      loc, ptrType, layout.storageType, object,
      ArrayRef<LLVM::GEPArg>{0, static_cast<int32_t>(fieldIndex)});
  if (!needsRefcount) {
    builder.create<LLVM::StoreOp>(loc, newValue, localFieldPtr);
    builder.create<LLVM::ReturnOp>(loc, ValueRange{});
  } else {
    Block *localSkipOldBlock = builder.createBlock(&fn.getBody());
    Block *localLoadOldBlock = builder.createBlock(&fn.getBody());
    builder.setInsertionPointToEnd(localBlock);
    builder.create<LLVM::CondBrOp>(loc, skipOldValueDrop, localSkipOldBlock,
                                   localLoadOldBlock);

    builder.setInsertionPointToStart(localSkipOldBlock);
    builder.create<LLVM::StoreOp>(loc, newValue, localFieldPtr);
    builder.create<LLVM::ReturnOp>(loc, ValueRange{});

    builder.setInsertionPointToStart(localLoadOldBlock);
    Value oldValue =
        builder.create<LLVM::LoadOp>(loc, fieldInfo.storageType, localFieldPtr);
    builder.create<LLVM::StoreOp>(loc, newValue, localFieldPtr);
    emitStaticFieldRuntimeCall(loc, oldValue, module, builder,
                               RuntimeSymbols::kDecRef);
    builder.create<LLVM::ReturnOp>(loc, ValueRange{});
  }

  builder.setInsertionPointToStart(managedBlock);
  Value lockPtr = getStaticClassLockPtr(loc, object, layout, builder);
  if (!needsRefcount) {
    emitStaticClassLockAcquire(loc, lockPtr, module, builder);
    Value managedFieldPtr = builder.create<LLVM::GEPOp>(
        loc, ptrType, layout.storageType, object,
        ArrayRef<LLVM::GEPArg>{0, static_cast<int32_t>(fieldIndex)});
    builder.create<LLVM::StoreOp>(loc, newValue, managedFieldPtr);
    emitStaticClassLockRelease(loc, lockPtr, module, builder);
    builder.create<LLVM::ReturnOp>(loc, ValueRange{});
    return fn;
  }

  Block *managedSkipOldBlock = builder.createBlock(&fn.getBody());
  Block *managedLoadOldBlock = builder.createBlock(&fn.getBody());
  builder.setInsertionPointToEnd(managedBlock);
  builder.create<LLVM::CondBrOp>(loc, skipOldValueDrop, managedSkipOldBlock,
                                 managedLoadOldBlock);

  builder.setInsertionPointToStart(managedSkipOldBlock);
  emitStaticClassLockAcquire(loc, lockPtr, module, builder);
  Value managedSkipFieldPtr = builder.create<LLVM::GEPOp>(
      loc, ptrType, layout.storageType, object,
      ArrayRef<LLVM::GEPArg>{0, static_cast<int32_t>(fieldIndex)});
  builder.create<LLVM::StoreOp>(loc, newValue, managedSkipFieldPtr);
  emitStaticClassLockRelease(loc, lockPtr, module, builder);
  builder.create<LLVM::ReturnOp>(loc, ValueRange{});

  builder.setInsertionPointToStart(managedLoadOldBlock);
  emitStaticClassLockAcquire(loc, lockPtr, module, builder);
  Value managedLoadFieldPtr = builder.create<LLVM::GEPOp>(
      loc, ptrType, layout.storageType, object,
      ArrayRef<LLVM::GEPArg>{0, static_cast<int32_t>(fieldIndex)});
  Value oldValue = builder.create<LLVM::LoadOp>(loc, fieldInfo.storageType,
                                                managedLoadFieldPtr);
  builder.create<LLVM::StoreOp>(loc, newValue, managedLoadFieldPtr);
  emitStaticClassLockRelease(loc, lockPtr, module, builder);
  emitStaticFieldRuntimeCall(loc, oldValue, module, builder,
                             RuntimeSymbols::kDecRef);
  builder.create<LLVM::ReturnOp>(loc, ValueRange{});
  return fn;
}

static LLVM::LLVMFuncOp getOrCreateStaticClassCopyHelper(
    Location loc, ModuleOp module, ClassType classType,
    const StaticClassLayout &layout, OpBuilder &builder,
    const PyLLVMTypeConverter &typeConverter) {
  auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());
  auto voidType = LLVM::LLVMVoidType::get(builder.getContext());
  std::string helperName = getClassHelperName(classType, "copy");
  auto fn = getOrInsertLLVMFunc(loc, module, builder, helperName, voidType,
                                {ptrType, ptrType});
  fn.setVisibility(SymbolTable::Visibility::Private);
  if (!fn.getBody().empty())
    return fn;

  Block *entry = fn.addEntryBlock(builder);
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(entry);

  Value dest = entry->getArgument(0);
  Value src = entry->getArgument(1);
  Value borrowLocal = builder.create<LLVM::ConstantOp>(
      loc, builder.getI1Type(), builder.getBoolAttr(false));
  Value retainNewValue = builder.create<LLVM::ConstantOp>(
      loc, builder.getI1Type(), builder.getBoolAttr(false));
  Value skipOldDrop = builder.create<LLVM::ConstantOp>(
      loc, builder.getI1Type(), builder.getBoolAttr(true));

  for (auto [fieldIndex, fieldInfo] : llvm::enumerate(layout.fields)) {
    auto getHelper = getOrCreateStaticClassGetFieldHelper(
        loc, module, classType, layout, fieldIndex, builder, typeConverter);
    auto getHelperRef =
        SymbolRefAttr::get(module.getContext(), getHelper.getName());
    auto fieldValue = builder.create<LLVM::CallOp>(
        loc, TypeRange{fieldInfo.storageType}, getHelperRef,
        ValueRange{src, borrowLocal});

    auto setHelper = getOrCreateStaticClassSetFieldHelper(
        loc, module, classType, layout, fieldIndex, builder, typeConverter);
    auto setHelperRef =
        SymbolRefAttr::get(module.getContext(), setHelper.getName());
    builder.create<LLVM::CallOp>(
        loc, TypeRange{}, setHelperRef,
        ValueRange{dest, fieldValue.getResult(), retainNewValue, skipOldDrop});
  }

  builder.create<LLVM::ReturnOp>(loc, ValueRange{});
  return fn;
}

static void
ensureStaticClassHelperBodies(ClassOp classOp, const StaticClassLayout &layout,
                              ModuleOp module, OpBuilder &builder,
                              const PyLLVMTypeConverter &typeConverter) {
  ClassType classType =
      ClassType::get(classOp.getContext(), classOp.getSymNameAttr().getValue());
  getOrCreateStaticClassRetainHelper(classOp.getLoc(), module, classType,
                                     layout, builder, typeConverter);
  getOrCreateStaticClassReleaseHelper(classOp.getLoc(), module, classType,
                                      layout, builder, typeConverter);
  getOrCreateStaticClassLocalDestroyHelper(classOp.getLoc(), module, classType,
                                           layout, builder, typeConverter);
  getOrCreateStaticClassPromoteHelper(classOp.getLoc(), module, classType,
                                      layout, builder, typeConverter);
  getOrCreateStaticClassReprHelper(classOp.getLoc(), module, classType,
                                   builder);
  getOrCreateStaticClassEqHelper(classOp.getLoc(), module, classType, builder);
  getOrCreateStaticClassCopyHelper(classOp.getLoc(), module, classType, layout,
                                   builder, typeConverter);
}

static FailureOr<Value>
boxStaticFieldValue(Location loc, Value storageValue, Type logicalType,
                    ModuleOp module, ConversionPatternRewriter &rewriter,
                    const PyLLVMTypeConverter &typeConverter,
                    bool borrowExistingRef = false) {
  RuntimeAPI runtime(module, rewriter, typeConverter);
  if (isa<IntType>(logicalType) && isa<IntegerType>(storageValue.getType()))
    return runtime
        .call(loc, RuntimeSymbols::kLongFromI64,
              typeConverter.getPyObjectPtrType(), ValueRange{storageValue})
        .getResult();
  if (isa<FloatType>(logicalType) &&
      isa<mlir::FloatType>(storageValue.getType()))
    return runtime
        .call(loc, RuntimeSymbols::kFloatFromDouble,
              typeConverter.getPyObjectPtrType(), ValueRange{storageValue})
        .getResult();
  if (isa<BoolType>(logicalType) && isa<IntegerType>(storageValue.getType()))
    return runtime
        .call(loc, RuntimeSymbols::kBoolFromBool,
              typeConverter.getPyObjectPtrType(), ValueRange{storageValue})
        .getResult();
  if (needsStaticFieldRefCount(logicalType, storageValue.getType(),
                               typeConverter)) {
    if (!borrowExistingRef) {
      runtime.call(loc, RuntimeSymbols::kIncRef, /*resultType=*/nullptr,
                   ValueRange{storageValue});
    }
    return storageValue;
  }
  Type convertedLogicalType = typeConverter.convertType(logicalType);
  if (convertedLogicalType &&
      isa<MemRefType, UnrankedMemRefType>(convertedLogicalType)) {
    return rewriter
        .create<UnrealizedConversionCastOp>(loc, convertedLogicalType,
                                            storageValue)
        .getResult(0);
  }
  if (storageValue.getType() != convertedLogicalType) {
    mlir::emitError(loc) << "unsupported static field boxing from "
                         << storageValue.getType() << " to " << logicalType;
    return failure();
  }
  return storageValue;
}

static FailureOr<SmallVector<Value>>
boxStaticFieldValues(Location loc, Value storageValue, Type logicalType,
                     ModuleOp module, ConversionPatternRewriter &rewriter,
                     const PyLLVMTypeConverter &typeConverter,
                     bool borrowExistingRef = false) {
  SmallVector<Type, 4> convertedTypes;
  if (failed(typeConverter.convertType(logicalType, convertedTypes)) ||
      convertedTypes.empty())
    return failure();
  if (convertedTypes.size() > 1) {
    SmallVector<Value> values = typeConverter.materializeTargetConversion(
        rewriter, loc, TypeRange(convertedTypes), ValueRange{storageValue},
        logicalType);
    if (values.empty())
      return failure();
    return values;
  }

  FailureOr<Value> boxed =
      boxStaticFieldValue(loc, storageValue, logicalType, module, rewriter,
                          typeConverter, borrowExistingRef);
  if (failed(boxed))
    return failure();
  return SmallVector<Value>{*boxed};
}

static bool isBorrowOnlyAttrGetUser(Operation *user) {
  return isa<NumAddOp, NumSubOp, NumLeOp, NumLtOp, NumGtOp, NumGeOp, NumEqOp,
             NumNeOp, ListAppendOp, ListRemoveOp, ListGetOp>(user);
}

static DecRefOp getBorrowedAttrGetDrop(AttrGetOp op) {
  DecRefOp drop;
  Operation *borrowUser = nullptr;

  for (Operation *user : op.getResult().getUsers()) {
    if (auto decref = dyn_cast<DecRefOp>(user)) {
      if (drop)
        return nullptr;
      drop = decref;
      continue;
    }
    if (!isBorrowOnlyAttrGetUser(user))
      return nullptr;
    if (borrowUser)
      return nullptr;
    borrowUser = user;
  }

  if (!borrowUser || !drop)
    return nullptr;
  if (borrowUser->getBlock() != drop->getBlock())
    return nullptr;
  if (!borrowUser->isBeforeInBlock(drop))
    return nullptr;
  return drop;
}

static FailureOr<Value>
unboxStaticFieldValue(Location loc, Value boxedValue, Type logicalType,
                      Type storageType, ModuleOp module,
                      ConversionPatternRewriter &rewriter,
                      const PyLLVMTypeConverter &typeConverter) {
  RuntimeAPI runtime(module, rewriter, typeConverter);
  if (isa<IntType>(logicalType) && isa<IntegerType>(storageType))
    return runtime
        .call(loc, RuntimeSymbols::kLongAsI64, storageType,
              ValueRange{boxedValue})
        .getResult();
  if (isa<FloatType>(logicalType) && isa<mlir::FloatType>(storageType))
    return runtime
        .call(loc, RuntimeSymbols::kFloatAsDouble, storageType,
              ValueRange{boxedValue})
        .getResult();
  if (isa<BoolType>(logicalType) && isa<IntegerType>(storageType))
    return runtime
        .call(loc, RuntimeSymbols::kBoolAsBool, storageType,
              ValueRange{boxedValue})
        .getResult();
  if (isa<MemRefType, UnrankedMemRefType>(boxedValue.getType())) {
    return rewriter
        .create<UnrealizedConversionCastOp>(loc, storageType, boxedValue)
        .getResult(0);
  }
  if (boxedValue.getType() != storageType) {
    mlir::emitError(loc) << "unsupported static field unboxing from "
                         << boxedValue.getType() << " to " << storageType;
    return failure();
  }
  return boxedValue;
}

static FailureOr<Value>
unboxStaticFieldValue(Location loc, ValueRange boxedValues, Type logicalType,
                      Type storageType, ModuleOp module,
                      ConversionPatternRewriter &rewriter,
                      const PyLLVMTypeConverter &typeConverter) {
  if (boxedValues.size() > 1) {
    return rewriter
        .create<UnrealizedConversionCastOp>(loc, storageType, boxedValues)
        .getResult(0);
  }
  if (boxedValues.empty())
    return failure();
  return unboxStaticFieldValue(loc, boxedValues.front(), logicalType,
                               storageType, module, rewriter, typeConverter);
}

// Static class instances lower to function-local stack slots.
struct ClassNewLowering : public OpConversionPattern<ClassNewOp> {
  ClassNewLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<ClassNewOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(ClassNewOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto classType = dyn_cast<ClassType>(op.getResult().getType());
    if (!classType)
      return failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    FailureOr<StaticClassLayout> layoutOr =
        getStaticClassLayout(op, classType, *typeConverter);
    if (failed(layoutOr))
      return failure();

    Value slot =
        createStaticClassSlot(op.getLoc(), layoutOr->objectType, rewriter, op);
    if (!slot)
      return failure();
    Value zero =
        rewriter.create<LLVM::ZeroOp>(op.getLoc(), layoutOr->objectType);
    rewriter.create<LLVM::StoreOp>(op.getLoc(), zero, slot);
    rewriter.replaceOp(op, slot);
    return success();
  }
};

struct ClassPromoteLowering : public OpConversionPattern<ClassPromoteOp> {
  ClassPromoteLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<ClassPromoteOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(ClassPromoteOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto classType = dyn_cast<ClassType>(op.getResult().getType());
    if (!classType)
      return failure();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    std::string helperName = getClassHelperName(classType, "promote");
    auto helper = getOrInsertLLVMFunc(op.getLoc(), module, rewriter, helperName,
                                      ptrType, {ptrType});
    auto helperRef = SymbolRefAttr::get(module.getContext(), helper.getName());
    auto call = rewriter.create<LLVM::CallOp>(op.getLoc(), TypeRange{ptrType},
                                              helperRef,
                                              ValueRange{adaptor.getInput()});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};

struct PublishLowering : public OpConversionPattern<PublishOp> {
  PublishLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<PublishOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(PublishOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.setInsertionPoint(op);
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    if (auto classType = dyn_cast<ClassType>(op.getResult().getType())) {
      if (adaptor.getInput().size() != 1)
        return failure();
      auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
      std::string helperName = getClassHelperName(classType, "promote");
      auto helper = getOrInsertLLVMFunc(op.getLoc(), module, rewriter,
                                        helperName, ptrType, {ptrType});
      auto helperRef =
          SymbolRefAttr::get(module.getContext(), helper.getName());
      auto call = rewriter.create<LLVM::CallOp>(
          op.getLoc(), TypeRange{ptrType}, helperRef,
          ValueRange{adaptor.getInput().front()});
      rewriter.replaceOp(op, call.getResults());
      return success();
    }

    FailureOr<SmallVector<Value>> promoted = failure();
    Type resultType = op.getResult().getType();
    if (isa<ListType>(resultType)) {
      promoted =
          promoteListDescriptor(op.getLoc(), adaptor.getInput(), rewriter);
    } else if (isa<TupleType>(resultType)) {
      promoted =
          promoteTupleDescriptor(op.getLoc(), adaptor.getInput(), rewriter);
    } else if (isa<DictType>(resultType)) {
      promoted =
          promoteDictDescriptor(op.getLoc(), adaptor.getInput(), rewriter);
    }
    if (succeeded(promoted)) {
      rewriter.replaceOpWithMultiple(op, ArrayRef<ValueRange>{*promoted});
      return success();
    }

    if (adaptor.getInput().size() != 1)
      return failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *typeConverter);
    runtime.call(op.getLoc(), RuntimeSymbols::kIncRef, /*resultType=*/nullptr,
                 ValueRange{adaptor.getInput().front()});

    rewriter.replaceOp(op, adaptor.getInput().front());
    return success();
  }
};

// AttrGetOp lowers to direct field load from the class slot.
struct AttrGetLowering : public OpConversionPattern<AttrGetOp> {
  AttrGetLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<AttrGetOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(AttrGetOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();
    auto classType = dyn_cast<ClassType>(op.getObject().getType());
    if (!classType)
      return failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    FailureOr<StaticClassLayout> layoutOr =
        getStaticClassLayout(op, classType, *typeConverter);
    if (failed(layoutOr))
      return failure();
    FailureOr<std::pair<unsigned, StaticClassFieldInfo>> fieldOr =
        lookupStaticClassField(op, classType, *typeConverter,
                               op.getNameAttr().getValue());
    if (failed(fieldOr))
      return failure();

    bool knownLocal =
        static_cast<bool>(op->getAttr("lython.known_local_class_access"));
    if (knownLocal) {
      DecRefOp borrowedDrop = nullptr;
      if (needsStaticFieldRefCount(fieldOr->second.logicalType,
                                   fieldOr->second.storageType,
                                   *typeConverter)) {
        borrowedDrop = getBorrowedAttrGetDrop(op);
      }

      auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
      Value fieldPtr = rewriter.create<LLVM::GEPOp>(
          op.getLoc(), ptrType, layoutOr->storageType,
          adaptor.getObject().front(),
          ArrayRef<LLVM::GEPArg>{0, static_cast<int32_t>(fieldOr->first)});
      Value loaded = rewriter.create<LLVM::LoadOp>(
          op.getLoc(), fieldOr->second.storageType, fieldPtr);
      FailureOr<SmallVector<Value>> boxed = boxStaticFieldValues(
          op.getLoc(), loaded, fieldOr->second.logicalType, module, rewriter,
          *typeConverter, static_cast<bool>(borrowedDrop));
      if (failed(boxed))
        return failure();
      if (borrowedDrop)
        rewriter.eraseOp(borrowedDrop);
      SmallVector<ValueRange, 1> replacements{ValueRange(*boxed)};
      rewriter.replaceOpWithMultiple(op, replacements);
      return success();
    }

    auto helper = getOrCreateStaticClassGetFieldHelper(
        op.getLoc(), module, classType, *layoutOr, fieldOr->first, rewriter,
        *typeConverter);
    auto helperRef = SymbolRefAttr::get(module.getContext(), helper.getName());
    Value borrowLocal = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), rewriter.getI1Type(), rewriter.getBoolAttr(false));
    auto call = rewriter.create<LLVM::CallOp>(
        op.getLoc(), TypeRange{fieldOr->second.storageType}, helperRef,
        ValueRange{adaptor.getObject().front(), borrowLocal});
    FailureOr<SmallVector<Value>> boxed = boxStaticFieldValues(
        op.getLoc(), call.getResult(), fieldOr->second.logicalType, module,
        rewriter, *typeConverter, /*borrowExistingRef=*/true);
    if (failed(boxed))
      return failure();
    SmallVector<ValueRange, 1> replacements{ValueRange(*boxed)};
    rewriter.replaceOpWithMultiple(op, replacements);
    return success();
  }
};

// AttrSetOp lowers to direct field store into the class slot.
struct AttrSetLowering : public OpConversionPattern<AttrSetOp> {
  AttrSetLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<AttrSetOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(AttrSetOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();
    auto classType = dyn_cast<ClassType>(op.getObject().getType());
    if (!classType)
      return failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    FailureOr<StaticClassLayout> layoutOr =
        getStaticClassLayout(op, classType, *typeConverter);
    if (failed(layoutOr))
      return failure();
    FailureOr<std::pair<unsigned, StaticClassFieldInfo>> fieldOr =
        lookupStaticClassField(op, classType, *typeConverter,
                               op.getNameAttr().getValue());
    if (failed(fieldOr))
      return failure();

    FailureOr<Value> unboxed = unboxStaticFieldValue(
        op.getLoc(), adaptor.getValue(), fieldOr->second.logicalType,
        fieldOr->second.storageType, module, rewriter, *typeConverter);
    if (failed(unboxed))
      return failure();
    bool consumeValue = static_cast<bool>(op->getAttr("lython.consume_value"));
    bool skipOldValueLoad =
        static_cast<bool>(op->getAttr("lython.zero_init_first_store"));
    bool knownLocal =
        static_cast<bool>(op->getAttr("lython.known_local_class_access"));
    if (knownLocal) {
      auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
      Value fieldPtr = rewriter.create<LLVM::GEPOp>(
          op.getLoc(), ptrType, layoutOr->storageType,
          adaptor.getObject().front(),
          ArrayRef<LLVM::GEPArg>{0, static_cast<int32_t>(fieldOr->first)});
      Value oldValue;
      if (needsStaticFieldRefCount(fieldOr->second.logicalType,
                                   fieldOr->second.storageType,
                                   *typeConverter)) {
        RuntimeAPI runtime(module, rewriter, *typeConverter);
        if (!skipOldValueLoad) {
          oldValue = rewriter.create<LLVM::LoadOp>(
              op.getLoc(), fieldOr->second.storageType, fieldPtr);
        }
        if (!consumeValue) {
          runtime.call(op.getLoc(), RuntimeSymbols::kIncRef,
                       /*resultType=*/nullptr, ValueRange{*unboxed});
        }
      }
      rewriter.create<LLVM::StoreOp>(op.getLoc(), *unboxed, fieldPtr);
      if (oldValue) {
        RuntimeAPI runtime(module, rewriter, *typeConverter);
        runtime.call(op.getLoc(), RuntimeSymbols::kDecRef,
                     /*resultType=*/nullptr, ValueRange{oldValue});
      }
      rewriter.eraseOp(op);
      return success();
    }

    auto helper = getOrCreateStaticClassSetFieldHelper(
        op.getLoc(), module, classType, *layoutOr, fieldOr->first, rewriter,
        *typeConverter);
    auto helperRef = SymbolRefAttr::get(module.getContext(), helper.getName());
    Value retainNewValue = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), rewriter.getI1Type(), rewriter.getBoolAttr(!consumeValue));
    Value skipOldDrop = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), rewriter.getI1Type(),
        rewriter.getBoolAttr(skipOldValueLoad));
    rewriter.create<LLVM::CallOp>(op.getLoc(), TypeRange{}, helperRef,
                                  ValueRange{adaptor.getObject().front(),
                                             *unboxed, retainNewValue,
                                             skipOldDrop});
    rewriter.eraseOp(op);
    return success();
  }
};

// ClassOp is just a container - erase it and keep its methods at module scope
struct ClassOpLowering : public OpConversionPattern<ClassOp> {
  ClassOpLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<ClassOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(ClassOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();
    ClassType classType =
        ClassType::get(op.getContext(), op.getSymNameAttr().getValue());
    FailureOr<StaticClassLayout> layoutOr =
        getStaticClassLayout(op, classType, *typeConverter);
    if (failed(layoutOr))
      return failure();
    ensureStaticClassHelperBodies(op, *layoutOr, module, rewriter,
                                  *typeConverter);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populatePyClassValueLoweringPatterns(PyLLVMTypeConverter &typeConverter,
                                          RewritePatternSet &patterns) {
  auto *ctx = patterns.getContext();
  patterns.add<ClassNewLowering, ClassPromoteLowering, PublishLowering,
               AttrGetLowering, AttrSetLowering, ClassOpLowering>(typeConverter,
                                                                  ctx);
}

} // namespace py
