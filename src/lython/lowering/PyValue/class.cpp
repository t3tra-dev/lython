#include "Common/Container.h"
#include "Common/LoweringUtils.h"
#include "Common/RuntimeSupport.h"
#include "Common/SlotUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FormatVariadic.h"

#include <cstdint>
#include <string>

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

namespace py {
namespace {

static ClassOp lookupStaticClass(mlir::Operation *from, ClassType classType) {
  mlir::StringAttr nameAttr =
      mlir::StringAttr::get(from->getContext(), classType.getClassName());
  for (mlir::Operation *symbolTableOp = from; symbolTableOp;
       symbolTableOp = symbolTableOp->getParentOp()) {
    if (!symbolTableOp->hasTrait<mlir::OpTrait::SymbolTable>())
      continue;
    if (mlir::Operation *symbol =
            mlir::SymbolTable::lookupSymbolIn(symbolTableOp, nameAttr))
      if (ClassOp classOp = mlir::dyn_cast<ClassOp>(symbol))
        return classOp;
  }
  return nullptr;
}

struct StaticClassFieldInfo {
  mlir::StringAttr name;
  mlir::Type logicalType;
  mlir::Type storageType;
};

struct StaticClassLayout {
  mlir::LLVM::LLVMStructType storageType;
  mlir::LLVM::LLVMStructType objectType;
  llvm::SmallVector<StaticClassFieldInfo, 8> fields;
};

static mlir::Type
getStaticFieldStorageType(mlir::Type logicalType,
                          const PyLLVMTypeConverter &typeConverter,
                          mlir::MLIRContext *ctx) {
  auto getMemRefDescriptorType =
      [&](mlir::MemRefType memrefType) -> mlir::Type {
    if (memrefType.getRank() != 1)
      return mlir::Type();
    auto ptrType = mlir::LLVM::LLVMPointerType::get(ctx);
    auto i64Type = mlir::IntegerType::get(ctx, 64);
    auto sizesType = mlir::LLVM::LLVMArrayType::get(i64Type, 1);
    return mlir::LLVM::LLVMStructType::getLiteral(
        ctx, llvm::ArrayRef<mlir::Type>{ptrType, ptrType, i64Type, sizesType,
                                        sizesType});
  };

  llvm::SmallVector<mlir::Type, 4> convertedTypes;
  if (mlir::failed(typeConverter.convertType(logicalType, convertedTypes)) ||
      convertedTypes.empty())
    return mlir::Type();

  if (convertedTypes.size() == 1) {
    if (auto memrefType =
            mlir::dyn_cast<mlir::MemRefType>(convertedTypes.front()))
      return getMemRefDescriptorType(memrefType);
    return convertedTypes.front();
  }

  llvm::SmallVector<mlir::Type, 4> descriptorTypes;
  descriptorTypes.reserve(convertedTypes.size());
  for (mlir::Type converted : convertedTypes) {
    auto memrefType = mlir::dyn_cast<mlir::MemRefType>(converted);
    if (!memrefType)
      return mlir::Type();
    mlir::Type descriptorType = getMemRefDescriptorType(memrefType);
    if (!descriptorType)
      return mlir::Type();
    descriptorTypes.push_back(descriptorType);
  }
  return mlir::LLVM::LLVMStructType::getLiteral(ctx, descriptorTypes);
}

namespace class_field::Refcount {
bool needed(mlir::Type logicalType, mlir::Type storageType,
            const PyLLVMTypeConverter &typeConverter) {
  return isPyType(logicalType) &&
         !mlir::isa<ClassType, NoneType, BoolType>(logicalType) &&
         storageType == typeConverter.getPyObjectPtrType();
}
} // namespace class_field::Refcount

namespace class_element::List {
ClassType type(mlir::Type logicalType) {
  auto listType = mlir::dyn_cast<ListType>(logicalType);
  if (!listType)
    return {};
  return mlir::dyn_cast<ClassType>(listType.getElementType());
}
} // namespace class_element::List

namespace class_element::Tuple {
void slots(mlir::Type logicalType,
           llvm::SmallVectorImpl<std::pair<unsigned, ClassType>> &slots) {
  auto tupleType = mlir::dyn_cast<TupleType>(logicalType);
  if (!tupleType)
    return;
  for (auto [index, type] : llvm::enumerate(tupleType.getElementTypes()))
    if (auto classType = mlir::dyn_cast<ClassType>(type))
      slots.push_back({static_cast<unsigned>(index), classType});
}
} // namespace class_element::Tuple

namespace class_element::Dict {
std::pair<ClassType, ClassType> types(mlir::Type logicalType) {
  auto dictType = mlir::dyn_cast<DictType>(logicalType);
  if (!dictType)
    return {};
  return {mlir::dyn_cast<ClassType>(dictType.getKeyType()),
          mlir::dyn_cast<ClassType>(dictType.getValueType())};
}
} // namespace class_element::Dict

static mlir::FailureOr<StaticClassLayout>
getStaticClassLayout(mlir::Operation *op, ClassType classType,
                     const PyLLVMTypeConverter &typeConverter) {
  ClassOp classOp = lookupStaticClass(op, classType);
  if (!classOp) {
    op->emitError("unable to resolve class '")
        << classType.getClassName() << "'";
    return mlir::failure();
  }

  mlir::ArrayAttr fieldNamesAttr = classOp.getFieldNamesAttr();
  mlir::ArrayAttr fieldTypesAttr = classOp.getFieldTypesAttr();

  StaticClassLayout layout;
  if (!fieldNamesAttr && !fieldTypesAttr) {
    layout.storageType =
        mlir::LLVM::LLVMStructType::getLiteral(op->getContext(), {});
    layout.objectType = mlir::LLVM::LLVMStructType::getLiteral(
        op->getContext(),
        llvm::ArrayRef<mlir::Type>{
            layout.storageType, mlir::IntegerType::get(op->getContext(), 1),
            mlir::IntegerType::get(op->getContext(), 32),
            mlir::IntegerType::get(op->getContext(), 64)});
    return layout;
  }
  if (!fieldNamesAttr || !fieldTypesAttr ||
      fieldNamesAttr.size() != fieldTypesAttr.size()) {
    op->emitError("class '")
        << classType.getClassName() << "' has malformed static field schema";
    return mlir::failure();
  }

  llvm::SmallVector<mlir::Type, 8> loweredFieldTypes;
  for (auto [nameAttr, typeAttr] : llvm::zip(fieldNamesAttr, fieldTypesAttr)) {
    auto stringAttr = mlir::dyn_cast<mlir::StringAttr>(nameAttr);
    auto mlirTypeAttr = mlir::dyn_cast<mlir::TypeAttr>(typeAttr);
    if (!stringAttr || !mlirTypeAttr) {
      op->emitError("class '")
          << classType.getClassName() << "' has malformed static field schema";
      return mlir::failure();
    }
    mlir::Type logicalType = mlirTypeAttr.getValue();
    mlir::Type storageType =
        getStaticFieldStorageType(logicalType, typeConverter, op->getContext());
    if (!storageType) {
      op->emitError("failed to convert field type ")
          << logicalType << " in class '" << classType.getClassName() << "'";
      return mlir::failure();
    }
    layout.fields.push_back({stringAttr, logicalType, storageType});
    loweredFieldTypes.push_back(storageType);
  }

  layout.storageType = mlir::LLVM::LLVMStructType::getLiteral(
      op->getContext(), loweredFieldTypes);
  layout.objectType = mlir::LLVM::LLVMStructType::getLiteral(
      op->getContext(),
      llvm::ArrayRef<mlir::Type>{layout.storageType,
                                 mlir::IntegerType::get(op->getContext(), 1),
                                 mlir::IntegerType::get(op->getContext(), 32),
                                 mlir::IntegerType::get(op->getContext(), 64)});
  return layout;
}

static mlir::FailureOr<std::pair<unsigned, StaticClassFieldInfo>>
lookupStaticClassField(mlir::Operation *op, ClassType classType,
                       const PyLLVMTypeConverter &typeConverter,
                       llvm::StringRef fieldName) {
  mlir::FailureOr<StaticClassLayout> layoutOr =
      getStaticClassLayout(op, classType, typeConverter);
  if (mlir::failed(layoutOr))
    return mlir::failure();

  for (auto [index, fieldInfo] : llvm::enumerate(layoutOr->fields))
    if (fieldInfo.name.getValue() == fieldName)
      return std::make_pair(static_cast<unsigned>(index), fieldInfo);

  op->emitError("class '") << classType.getClassName() << "' has no field '"
                           << fieldName << "'";
  return mlir::failure();
}

namespace class_helper::Field {
std::string name(ClassType classType, llvm::StringRef action,
                 unsigned fieldIndex) {
  return ("__ly_class_" + action + "_" + classType.getClassName() + "_" +
          std::to_string(fieldIndex))
      .str();
}
} // namespace class_helper::Field

namespace class_helper::Kind {
void mark(mlir::LLVM::LLVMFuncOp fn, ClassType classType,
          llvm::StringRef kind) {
  if (!fn)
    return;
  mlir::OpBuilder builder(fn.getContext());
  fn->setAttr(ClassSafetyAttrs::kHelperKind, builder.getStringAttr(kind));
  fn->setAttr(ClassSafetyAttrs::kHelperClass,
              builder.getStringAttr(classType.getClassName()));
}
} // namespace class_helper::Kind

namespace class_helper::Schema {
void mark(mlir::LLVM::LLVMFuncOp fn, const StaticClassLayout &layout,
          const PyLLVMTypeConverter &typeConverter) {
  if (!fn)
    return;
  int64_t directRefcountFields = 0;
  int64_t containerFields = 0;
  llvm::SmallVector<int64_t, 8> directRefcountFieldIndices;
  llvm::SmallVector<int64_t, 8> containerFieldIndices;
  for (auto [index, field] : llvm::enumerate(layout.fields)) {
    if (class_field::Refcount::needed(field.logicalType, field.storageType,
                                      typeConverter)) {
      ++directRefcountFields;
      directRefcountFieldIndices.push_back(static_cast<int64_t>(index));
      continue;
    }
    if (mlir::isa<ListType, TupleType, DictType>(field.logicalType)) {
      ++containerFields;
      containerFieldIndices.push_back(static_cast<int64_t>(index));
    }
  }
  mlir::OpBuilder builder(fn.getContext());
  fn->setAttr(ClassSafetyAttrs::kHelperDirectRefcountFields,
              builder.getI64IntegerAttr(directRefcountFields));
  fn->setAttr(ClassSafetyAttrs::kHelperContainerFields,
              builder.getI64IntegerAttr(containerFields));
  auto makeIndexArray = [&](llvm::ArrayRef<int64_t> indices) {
    llvm::SmallVector<mlir::Attribute, 8> attrs;
    attrs.reserve(indices.size());
    for (int64_t index : indices)
      attrs.push_back(builder.getI64IntegerAttr(index));
    return builder.getArrayAttr(attrs);
  };
  fn->setAttr(ClassSafetyAttrs::kHelperDirectRefcountFieldIndices,
              makeIndexArray(directRefcountFieldIndices));
  fn->setAttr(ClassSafetyAttrs::kHelperContainerFieldIndices,
              makeIndexArray(containerFieldIndices));
}
} // namespace class_helper::Schema

namespace class_helper::Field {
void mark(mlir::LLVM::LLVMFuncOp fn, ClassType classType, llvm::StringRef kind,
          unsigned fieldIndex) {
  class_helper::Kind::mark(fn, classType, kind);
  if (!fn)
    return;
  mlir::OpBuilder builder(fn.getContext());
  fn->setAttr(ClassSafetyAttrs::kHelperFieldIndex,
              builder.getI64IntegerAttr(fieldIndex));
}
} // namespace class_helper::Field

namespace string_literal {
mlir::Value ptr(mlir::Location loc, mlir::ModuleOp module,
                mlir::OpBuilder &builder, llvm::StringRef literal) {
  llvm::SmallString<32> symbolName("__ly_str_");
  auto hashValue = static_cast<uint64_t>(llvm::hash_value(literal));
  symbolName += llvm::formatv("{0:X}", hashValue).str();

  mlir::LLVM::GlobalOp global =
      module.lookupSymbol<mlir::LLVM::GlobalOp>(symbolName);
  if (!global) {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(module.getBody());
    auto arrayType =
        mlir::LLVM::LLVMArrayType::get(builder.getI8Type(), literal.size() + 1);
    llvm::SmallString<32> storage(literal);
    storage.push_back('\0');
    global = builder.create<mlir::LLVM::GlobalOp>(
        loc, arrayType, /*isConstant=*/true, mlir::LLVM::Linkage::Internal,
        symbolName, builder.getStringAttr(storage));
  }

  auto ptrType = mlir::LLVM::LLVMPointerType::get(module.getContext());
  mlir::Value addr = builder.create<mlir::LLVM::AddressOfOp>(
      loc, ptrType, global.getSymNameAttr());
  mlir::Value zero = builder.create<mlir::LLVM::ConstantOp>(
      loc, builder.getI64Type(), builder.getI64IntegerAttr(0));
  return builder.create<mlir::LLVM::GEPOp>(
      loc, ptrType, global.getType(), addr,
      llvm::ArrayRef<mlir::Value>{zero, zero});
}
} // namespace string_literal

namespace class_object::Slot {
mlir::Value create(mlir::Location loc, mlir::LLVM::LLVMStructType objectType,
                   mlir::ConversionPatternRewriter &rewriter,
                   mlir::Operation *anchor) {
  auto ptrType = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
  auto i64Type = rewriter.getI64Type();
  auto oneAttr = rewriter.getI64IntegerAttr(1);

  auto parentFunc = anchor->getParentOfType<mlir::func::FuncOp>();
  if (!parentFunc)
    return mlir::Value();

  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(&parentFunc.getBody().front());
  mlir::Value one =
      rewriter.create<mlir::LLVM::ConstantOp>(loc, i64Type, oneAttr);
  auto slot =
      rewriter.create<mlir::LLVM::AllocaOp>(loc, ptrType, objectType, one,
                                            /*alignment=*/0);
  slot->setAttr(OwnershipContractAttrs::kOwnedLocalObject,
                rewriter.getUnitAttr());
  return slot;
}
} // namespace class_object::Slot

namespace class_object::Managed {
mlir::Value flagPtr(mlir::Location loc, mlir::Value object,
                    const StaticClassLayout &layout, mlir::OpBuilder &builder) {
  auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  return builder.create<mlir::LLVM::GEPOp>(
      loc, ptrType, layout.objectType, object,
      llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 1});
}
} // namespace class_object::Managed

namespace class_object::Refcount {
mlir::Value ptr(mlir::Location loc, mlir::Value object,
                const StaticClassLayout &layout, mlir::OpBuilder &builder) {
  auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  return builder.create<mlir::LLVM::GEPOp>(
      loc, ptrType, layout.objectType, object,
      llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 3});
}
} // namespace class_object::Refcount

namespace class_object::Lock {
mlir::Value ptr(mlir::Location loc, mlir::Value object,
                const StaticClassLayout &layout, mlir::OpBuilder &builder) {
  auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  return builder.create<mlir::LLVM::GEPOp>(
      loc, ptrType, layout.objectType, object,
      llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 2});
}
} // namespace class_object::Lock

static mlir::LLVM::CallOp
emitRuntimeCall(mlir::Location loc, mlir::ModuleOp module,
                mlir::OpBuilder &builder, llvm::StringRef runtimeSymbol,
                mlir::Type resultType, mlir::ValueRange args) {
  llvm::SmallVector<mlir::Type, 4> argTypes;
  argTypes.reserve(args.size());
  for (mlir::Value arg : args)
    argTypes.push_back(arg.getType());

  auto callee = getOrInsertLLVMFunc(loc, module, builder, runtimeSymbol,
                                    resultType, argTypes);
  ownership::llvm_func::Contract::apply(callee, runtimeSymbol);
  auto calleeRef =
      mlir::SymbolRefAttr::get(module.getContext(), callee.getName());
  llvm::SmallVector<mlir::Type, 1> results;
  if (!mlir::isa<mlir::LLVM::LLVMVoidType>(resultType))
    results.push_back(resultType);
  return builder.create<mlir::LLVM::CallOp>(loc, mlir::TypeRange(results),
                                            calleeRef, args);
}

static mlir::LLVM::CallOp emitRuntimeVoidCall(mlir::Location loc,
                                              mlir::ModuleOp module,
                                              mlir::OpBuilder &builder,
                                              llvm::StringRef runtimeSymbol,
                                              mlir::ValueRange args) {
  return emitRuntimeCall(loc, module, builder, runtimeSymbol,
                         mlir::LLVM::LLVMVoidType::get(builder.getContext()),
                         args);
}

// Atomic lowering policy:
// - Prefer memref.atomic_rmw for memref-backed storage when acq_rel RMW is the
//   intended semantics. This keeps buffer-like storage in higher-level MLIR.
// - Use LLVM atomics when the value is a raw LLVM pointer or when a specific
//   ordering weaker/stronger than memref.atomic_rmw's current acq_rel lowering
//   is required.
namespace atomic_storage::MemRef {
mlir::Value rmw(mlir::Location loc, mlir::arith::AtomicRMWKind kind,
                mlir::Value memref, mlir::Value value, mlir::ValueRange indices,
                mlir::OpBuilder &builder) {
  if (!mlir::isa<mlir::MemRefType>(memref.getType()))
    return {};
  return builder.create<mlir::memref::AtomicRMWOp>(loc, kind, value, memref,
                                                   indices);
}
} // namespace atomic_storage::MemRef

namespace atomic_storage::Integer {
mlir::Value rmw(mlir::Location loc, mlir::LLVM::AtomicBinOp llvmKind,
                mlir::arith::AtomicRMWKind memrefKind, mlir::Value storage,
                mlir::Value value, mlir::ValueRange indices,
                mlir::LLVM::AtomicOrdering ordering, mlir::OpBuilder &builder) {
  if (ordering == mlir::LLVM::AtomicOrdering::acq_rel)
    if (mlir::Value result = atomic_storage::MemRef::rmw(
            loc, memrefKind, storage, value, indices, builder))
      return result;
  return builder.create<mlir::LLVM::AtomicRMWOp>(loc, llvmKind, storage, value,
                                                 ordering);
}
} // namespace atomic_storage::Integer

namespace atomic_storage::Store {
mlir::Operation *ordered(mlir::Location loc, mlir::Value storage,
                         mlir::Value value, mlir::ValueRange indices,
                         mlir::LLVM::AtomicOrdering ordering,
                         mlir::OpBuilder &builder) {
  if (mlir::isa<mlir::MemRefType>(storage.getType()) &&
      ordering == mlir::LLVM::AtomicOrdering::acq_rel) {
    mlir::Value result =
        atomic_storage::MemRef::rmw(loc, mlir::arith::AtomicRMWKind::assign,
                                    storage, value, indices, builder);
    return result.getDefiningOp();
  }
  unsigned alignment = 0;
  if (auto intType = mlir::dyn_cast<mlir::IntegerType>(value.getType()))
    alignment = std::max<unsigned>(1, intType.getWidth() / 8);
  return builder
      .create<mlir::LLVM::StoreOp>(loc, value, storage, alignment,
                                   /*isVolatile=*/false,
                                   /*isNonTemporal=*/false,
                                   /*isInvariantGroup=*/false, ordering)
      .getOperation();
}
} // namespace atomic_storage::Store

namespace class_container::Kind {
std::optional<llvm::StringRef> name(mlir::Type logicalType) {
  if (mlir::isa<ListType>(logicalType))
    return ContainerSafetyAttrs::kKindList;
  if (mlir::isa<TupleType>(logicalType))
    return ContainerSafetyAttrs::kKindTuple;
  if (mlir::isa<DictType>(logicalType))
    return ContainerSafetyAttrs::kKindDict;
  return std::nullopt;
}
} // namespace class_container::Kind

namespace class_container::Atomic {
std::string group(mlir::Value descriptor) {
  if (!descriptor)
    return {};
  if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(descriptor))
    return "blockarg." + std::to_string(arg.getArgNumber());
  return "value." + std::to_string(reinterpret_cast<std::uintptr_t>(
                        descriptor.getAsOpaquePointer()));
}

void markHeader(mlir::Operation *op, mlir::Value descriptor,
                std::optional<int64_t> slot, mlir::Type logicalType = {}) {
  auto kind = logicalType ? class_container::Kind::name(logicalType)
                          : std::optional<llvm::StringRef>{};
  threadsafe::memref::Atomic::set(op, ContainerSafetyAttrs::kComponentHeader,
                                  slot,
                                  class_container::Atomic::group(descriptor),
                                  kind ? *kind : llvm::StringRef{});
  if (op) {
    mlir::OpBuilder builder(op->getContext());
    op->setAttr(
        ThreadSafetyAttrs::kAtomicProvenance,
        builder.getStringAttr(ThreadSafetyAttrs::kProvenanceMemRefDescriptor));
  }
}
} // namespace class_container::Atomic

namespace class_object::Lock {
void acquire(mlir::Location loc, mlir::Value lockPtr, mlir::ModuleOp module,
             mlir::OpBuilder &builder) {
  (void)module;
  mlir::Block *currentBlock = builder.getInsertionBlock();
  auto insertionPoint = builder.getInsertionPoint();
  mlir::Region *region = builder.getInsertionBlock()->getParent();
  mlir::Block *loopBlock = builder.createBlock(region);
  mlir::Block *acquiredBlock = builder.createBlock(region);

  builder.setInsertionPoint(currentBlock, insertionPoint);
  builder.create<mlir::LLVM::BrOp>(loc, mlir::ValueRange{}, loopBlock);

  builder.setInsertionPointToStart(loopBlock);
  mlir::Value one = builder.create<mlir::LLVM::ConstantOp>(
      loc, builder.getI32Type(), builder.getI32IntegerAttr(1));
  mlir::Value previous = atomic_storage::Integer::rmw(
      loc, mlir::LLVM::AtomicBinOp::xchg, mlir::arith::AtomicRMWKind::assign,
      lockPtr, one, mlir::ValueRange{}, mlir::LLVM::AtomicOrdering::acquire,
      builder);
  threadsafe::Atomic::set(previous.getDefiningOp(),
                          ThreadSafetyAttrs::kRoleClassLockAcquire,
                          ThreadSafetyAttrs::kOrderingAcquire);
  mlir::Value zero = builder.create<mlir::LLVM::ConstantOp>(
      loc, builder.getI32Type(), builder.getI32IntegerAttr(0));
  mlir::Value acquired = builder.create<mlir::LLVM::ICmpOp>(
      loc, mlir::LLVM::ICmpPredicate::eq, previous, zero);
  builder.create<mlir::LLVM::CondBrOp>(loc, acquired, acquiredBlock, loopBlock);

  builder.setInsertionPointToStart(acquiredBlock);
}

void release(mlir::Location loc, mlir::Value lockPtr, mlir::ModuleOp module,
             mlir::OpBuilder &builder) {
  (void)module;
  mlir::Value zero = builder.create<mlir::LLVM::ConstantOp>(
      loc, builder.getI32Type(), builder.getI32IntegerAttr(0));
  mlir::Operation *store = atomic_storage::Store::ordered(
      loc, lockPtr, zero, mlir::ValueRange{},
      mlir::LLVM::AtomicOrdering::release, builder);
  threadsafe::Atomic::set(store, ThreadSafetyAttrs::kRoleClassLockRelease,
                          ThreadSafetyAttrs::kOrderingRelease);
}
} // namespace class_object::Lock

namespace class_object::Refcount {
void inc(mlir::Location loc, mlir::Value refcountPtr, mlir::ModuleOp module,
         mlir::OpBuilder &builder) {
  (void)module;
  mlir::Value one = builder.create<mlir::LLVM::ConstantOp>(
      loc, builder.getI64Type(), builder.getI64IntegerAttr(1));
  mlir::Value previous = atomic_storage::Integer::rmw(
      loc, mlir::LLVM::AtomicBinOp::add, mlir::arith::AtomicRMWKind::addi,
      refcountPtr, one, mlir::ValueRange{},
      mlir::LLVM::AtomicOrdering::monotonic, builder);
  threadsafe::Atomic::set(previous.getDefiningOp(),
                          ThreadSafetyAttrs::kRoleClassRefcountRetain,
                          ThreadSafetyAttrs::kOrderingMonotonic,
                          ThreadSafetyAttrs::kPremiseOwnedToken);
  threadsafe::Retain::verifyOwnedToken(
      previous.getDefiningOp(), ThreadSafetyAttrs::kProofClassFieldHelper);
}

mlir::Value decAndIsZero(mlir::Location loc, mlir::Value refcountPtr,
                         mlir::ModuleOp module, mlir::OpBuilder &builder) {
  (void)module;
  mlir::Value one = builder.create<mlir::LLVM::ConstantOp>(
      loc, builder.getI64Type(), builder.getI64IntegerAttr(1));
  mlir::Value negOne = builder.create<mlir::LLVM::ConstantOp>(
      loc, builder.getI64Type(), builder.getI64IntegerAttr(-1));
  mlir::Value previous = atomic_storage::Integer::rmw(
      loc, mlir::LLVM::AtomicBinOp::add, mlir::arith::AtomicRMWKind::addi,
      refcountPtr, negOne, mlir::ValueRange{},
      mlir::LLVM::AtomicOrdering::acq_rel, builder);
  threadsafe::Atomic::set(previous.getDefiningOp(),
                          ThreadSafetyAttrs::kRoleClassRefcountRelease,
                          ThreadSafetyAttrs::kOrderingAcqRel);
  return builder.create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::eq,
                                            previous, one);
}
} // namespace class_object::Refcount

namespace aggregate_refcount::Call {
void mark(mlir::LLVM::CallOp call, llvm::StringRef suffix) {
  if (!call)
    return;
  if (call.getNumOperands() >= 1)
    ownership::aggregate::Slot::markLoad(call.getOperand(0));
  call->setAttr(suffix == "incref" ? OwnershipContractAttrs::kAggregateRetain
                                   : OwnershipContractAttrs::kAggregateRelease,
                mlir::UnitAttr::get(call.getContext()));
  if (suffix == "incref")
    threadsafe::Retain::premise(call.getOperation(),
                                ThreadSafetyAttrs::kPremiseAggregateBorrow);
}
} // namespace aggregate_refcount::Call

namespace class_container::Elements {
void list(mlir::Location loc, mlir::Value descriptor, ClassType elementType,
          mlir::ModuleOp module, mlir::OpBuilder &builder,
          llvm::StringRef suffix) {
  auto *parent = builder.getInsertionBlock()->getParent();
  auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  auto i64Type = builder.getI64Type();
  auto voidType = mlir::LLVM::LLVMVoidType::get(builder.getContext());
  auto helper = getOrInsertLLVMFunc(loc, module, builder,
                                    getClassHelperName(elementType, suffix),
                                    voidType, {ptrType});
  ownership::llvm_func::Contract::apply(helper, helper.getName());
  auto helperRef =
      mlir::SymbolRefAttr::get(module.getContext(), helper.getName());

  mlir::Value headerDescriptor = descriptor;
  mlir::Value itemsDescriptor = descriptor;
  if (auto structType =
          mlir::dyn_cast<mlir::LLVM::LLVMStructType>(descriptor.getType())) {
    if (!structType.isOpaque() && structType.getBody().size() == 2) {
      headerDescriptor = builder.create<mlir::LLVM::ExtractValueOp>(
          loc, structType.getBody()[0], descriptor,
          builder.getDenseI64ArrayAttr({0}));
      itemsDescriptor = builder.create<mlir::LLVM::ExtractValueOp>(
          loc, structType.getBody()[1], descriptor,
          builder.getDenseI64ArrayAttr({1}));
    }
  }

  mlir::Value headerData = builder.create<mlir::LLVM::ExtractValueOp>(
      loc, ptrType, headerDescriptor, builder.getDenseI64ArrayAttr({1}));
  mlir::Value itemData = builder.create<mlir::LLVM::ExtractValueOp>(
      loc, ptrType, itemsDescriptor, builder.getDenseI64ArrayAttr({1}));
  mlir::Value size =
      builder.create<mlir::LLVM::LoadOp>(loc, i64Type, headerData);
  mlir::Value zero = builder.create<mlir::LLVM::ConstantOp>(
      loc, i64Type, builder.getI64IntegerAttr(0));
  mlir::Value one = builder.create<mlir::LLVM::ConstantOp>(
      loc, i64Type, builder.getI64IntegerAttr(1));

  mlir::Block *currentBlock = builder.getInsertionBlock();
  mlir::Block *condBlock = builder.createBlock(parent);
  condBlock->addArgument(i64Type, loc);
  mlir::Block *bodyBlock = builder.createBlock(parent);
  mlir::Block *afterBlock = builder.createBlock(parent);
  builder.setInsertionPointToEnd(currentBlock);
  builder.create<mlir::LLVM::BrOp>(loc, mlir::ValueRange{zero}, condBlock);

  builder.setInsertionPointToStart(condBlock);
  mlir::Value iv = condBlock->getArgument(0);
  mlir::Value inBounds = builder.create<mlir::LLVM::ICmpOp>(
      loc, mlir::LLVM::ICmpPredicate::slt, iv, size);
  builder.create<mlir::LLVM::CondBrOp>(loc, inBounds, bodyBlock, afterBlock);

  builder.setInsertionPointToStart(bodyBlock);
  mlir::Value itemPtr = builder.create<mlir::LLVM::GEPOp>(
      loc, ptrType, i64Type, itemData, llvm::ArrayRef<mlir::LLVM::GEPArg>{iv});
  mlir::Value slot = builder.create<mlir::LLVM::LoadOp>(loc, i64Type, itemPtr);
  mlir::Value object =
      builder.create<mlir::LLVM::IntToPtrOp>(loc, ptrType, slot);
  auto call = builder.create<mlir::LLVM::CallOp>(
      loc, mlir::TypeRange{}, helperRef, mlir::ValueRange{object});
  aggregate_refcount::Call::mark(call, suffix);
  mlir::Value next = builder.create<mlir::LLVM::AddOp>(loc, iv, one);
  builder.create<mlir::LLVM::BrOp>(loc, mlir::ValueRange{next}, condBlock);

  builder.setInsertionPointToStart(afterBlock);
}

void slotI64(mlir::Location loc, mlir::Value slot, ClassType elementType,
             mlir::ModuleOp module, mlir::OpBuilder &builder,
             llvm::StringRef suffix) {
  auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  auto voidType = mlir::LLVM::LLVMVoidType::get(builder.getContext());
  auto helper = getOrInsertLLVMFunc(loc, module, builder,
                                    getClassHelperName(elementType, suffix),
                                    voidType, {ptrType});
  ownership::llvm_func::Contract::apply(helper, helper.getName());
  auto helperRef =
      mlir::SymbolRefAttr::get(module.getContext(), helper.getName());
  mlir::Value object =
      builder.create<mlir::LLVM::IntToPtrOp>(loc, ptrType, slot);
  auto call = builder.create<mlir::LLVM::CallOp>(
      loc, mlir::TypeRange{}, helperRef, mlir::ValueRange{object});
  aggregate_refcount::Call::mark(call, suffix);
}

void tuple(mlir::Location loc, mlir::Value descriptor,
           llvm::ArrayRef<std::pair<unsigned, ClassType>> elementSlots,
           mlir::ModuleOp module, mlir::OpBuilder &builder,
           llvm::StringRef suffix) {
  auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  auto i64Type = builder.getI64Type();
  mlir::Value data = builder.create<mlir::LLVM::ExtractValueOp>(
      loc, ptrType, descriptor, builder.getDenseI64ArrayAttr({1}));
  for (auto [index, elementType] : elementSlots) {
    mlir::Value offset = builder.create<mlir::LLVM::ConstantOp>(
        loc, i64Type, builder.getI64IntegerAttr(3 + index));
    mlir::Value itemPtr = builder.create<mlir::LLVM::GEPOp>(
        loc, ptrType, i64Type, data,
        llvm::ArrayRef<mlir::LLVM::GEPArg>{offset});
    mlir::Value slot =
        builder.create<mlir::LLVM::LoadOp>(loc, i64Type, itemPtr);
    class_container::Elements::slotI64(loc, slot, elementType, module, builder,
                                       suffix);
  }
}

void dict(mlir::Location loc, mlir::Value descriptor, ClassType keyClassType,
          ClassType valueClassType, mlir::ModuleOp module,
          mlir::OpBuilder &builder, llvm::StringRef suffix) {
  auto *parent = builder.getInsertionBlock()->getParent();
  auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  auto i64Type = builder.getI64Type();
  auto extractMemRefDataPtr = [&](mlir::Value memrefDescriptor) -> mlir::Value {
    return builder.create<mlir::LLVM::ExtractValueOp>(
        loc, ptrType, memrefDescriptor, builder.getDenseI64ArrayAttr({1}));
  };
  mlir::Value keysDescriptor = builder.create<mlir::LLVM::ExtractValueOp>(
      loc,
      mlir::cast<mlir::LLVM::LLVMStructType>(descriptor.getType()).getBody()[1],
      descriptor, builder.getDenseI64ArrayAttr({1}));
  mlir::Value valuesDescriptor = builder.create<mlir::LLVM::ExtractValueOp>(
      loc,
      mlir::cast<mlir::LLVM::LLVMStructType>(descriptor.getType()).getBody()[2],
      descriptor, builder.getDenseI64ArrayAttr({2}));
  mlir::Value statesDescriptor = builder.create<mlir::LLVM::ExtractValueOp>(
      loc,
      mlir::cast<mlir::LLVM::LLVMStructType>(descriptor.getType()).getBody()[3],
      descriptor, builder.getDenseI64ArrayAttr({3}));
  mlir::Value keysData = extractMemRefDataPtr(keysDescriptor);
  mlir::Value valuesData = extractMemRefDataPtr(valuesDescriptor);
  mlir::Value statesData = extractMemRefDataPtr(statesDescriptor);
  auto i8Type = builder.getI8Type();
  mlir::Value occupiedState = builder.create<mlir::LLVM::ConstantOp>(
      loc, i8Type, builder.getIntegerAttr(i8Type, 1));
  mlir::Value zero = builder.create<mlir::LLVM::ConstantOp>(
      loc, i64Type, builder.getI64IntegerAttr(0));
  mlir::Value one = builder.create<mlir::LLVM::ConstantOp>(
      loc, i64Type, builder.getI64IntegerAttr(1));
  mlir::Value capacity = builder.create<mlir::LLVM::ConstantOp>(
      loc, i64Type, builder.getI64IntegerAttr(64));

  mlir::Block *currentBlock = builder.getInsertionBlock();
  mlir::Block *condBlock = builder.createBlock(parent);
  condBlock->addArgument(i64Type, loc);
  mlir::Block *bodyBlock = builder.createBlock(parent);
  mlir::Block *occupiedBlock = builder.createBlock(parent);
  mlir::Block *nextBlock = builder.createBlock(parent);
  mlir::Block *afterBlock = builder.createBlock(parent);
  builder.setInsertionPointToEnd(currentBlock);
  builder.create<mlir::LLVM::BrOp>(loc, mlir::ValueRange{zero}, condBlock);

  builder.setInsertionPointToStart(condBlock);
  mlir::Value iv = condBlock->getArgument(0);
  mlir::Value inBounds = builder.create<mlir::LLVM::ICmpOp>(
      loc, mlir::LLVM::ICmpPredicate::slt, iv, capacity);
  builder.create<mlir::LLVM::CondBrOp>(loc, inBounds, bodyBlock, afterBlock);

  builder.setInsertionPointToStart(bodyBlock);
  mlir::Value statePtr = builder.create<mlir::LLVM::GEPOp>(
      loc, ptrType, i8Type, statesData, llvm::ArrayRef<mlir::LLVM::GEPArg>{iv});
  mlir::Value state = builder.create<mlir::LLVM::LoadOp>(loc, i8Type, statePtr);
  mlir::Value occupied = builder.create<mlir::LLVM::ICmpOp>(
      loc, mlir::LLVM::ICmpPredicate::eq, state, occupiedState);
  builder.create<mlir::LLVM::CondBrOp>(loc, occupied, occupiedBlock, nextBlock);

  builder.setInsertionPointToStart(occupiedBlock);
  if (keyClassType) {
    mlir::Value keyPtr = builder.create<mlir::LLVM::GEPOp>(
        loc, ptrType, i64Type, keysData,
        llvm::ArrayRef<mlir::LLVM::GEPArg>{iv});
    mlir::Value keySlot =
        builder.create<mlir::LLVM::LoadOp>(loc, i64Type, keyPtr);
    class_container::Elements::slotI64(loc, keySlot, keyClassType, module,
                                       builder, suffix);
  }
  if (valueClassType) {
    mlir::Value valuePtr = builder.create<mlir::LLVM::GEPOp>(
        loc, ptrType, i64Type, valuesData,
        llvm::ArrayRef<mlir::LLVM::GEPArg>{iv});
    mlir::Value valueSlot =
        builder.create<mlir::LLVM::LoadOp>(loc, i64Type, valuePtr);
    class_container::Elements::slotI64(loc, valueSlot, valueClassType, module,
                                       builder, suffix);
  }
  builder.create<mlir::LLVM::BrOp>(loc, mlir::ValueRange{}, nextBlock);

  builder.setInsertionPointToStart(nextBlock);
  mlir::Value next = builder.create<mlir::LLVM::AddOp>(loc, iv, one);
  builder.create<mlir::LLVM::BrOp>(loc, mlir::ValueRange{next}, condBlock);

  builder.setInsertionPointToStart(afterBlock);
}
} // namespace class_container::Elements

namespace class_container::Refcount {
std::optional<int64_t> slot(mlir::Type logicalType) {
  return container::Refcount::slotForLogicalType(logicalType);
}
} // namespace class_container::Refcount

namespace class_container::Descriptor {
mlir::Value part(mlir::Location loc, mlir::Value descriptor, unsigned index,
                 mlir::OpBuilder &builder) {
  auto structType =
      mlir::dyn_cast<mlir::LLVM::LLVMStructType>(descriptor.getType());
  if (!structType || structType.isOpaque() ||
      index >= structType.getBody().size())
    return {};
  return builder.create<mlir::LLVM::ExtractValueOp>(
      loc, structType.getBody()[index], descriptor,
      builder.getDenseI64ArrayAttr({static_cast<int64_t>(index)}));
}

mlir::Value dataPtr(mlir::Location loc, mlir::Value descriptor, unsigned index,
                    mlir::OpBuilder &builder) {
  auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  return builder.create<mlir::LLVM::ExtractValueOp>(
      loc, ptrType, descriptor,
      builder.getDenseI64ArrayAttr({static_cast<int64_t>(index)}));
}

mlir::Value header(mlir::Location loc, mlir::Value descriptor,
                   mlir::OpBuilder &builder) {
  if (mlir::Value header =
          class_container::Descriptor::part(loc, descriptor, 0, builder))
    return header;
  return descriptor;
}

llvm::SmallVector<mlir::Value, 4> parts(mlir::Location loc,
                                        mlir::Value descriptor,
                                        mlir::Type logicalType,
                                        mlir::OpBuilder &builder) {
  llvm::SmallVector<mlir::Value, 4> descriptors;
  unsigned parts = 0;
  if (mlir::isa<ListType, TupleType>(logicalType))
    parts = 2;
  else if (mlir::isa<DictType>(logicalType))
    parts = 4;
  if (parts == 0)
    return descriptors;

  for (unsigned index = 0; index < parts; ++index) {
    mlir::Value part =
        class_container::Descriptor::part(loc, descriptor, index, builder);
    if (!part)
      return {};
    descriptors.push_back(part);
  }
  return descriptors;
}
} // namespace class_container::Descriptor

namespace class_container::Elements {
void refcount(mlir::Location loc, mlir::Value descriptor,
              mlir::Type logicalType, mlir::ModuleOp module,
              mlir::OpBuilder &builder, llvm::StringRef suffix) {
  if (ClassType listElementClass = class_element::List::type(logicalType)) {
    class_container::Elements::list(loc, descriptor, listElementClass, module,
                                    builder, suffix);
    return;
  }

  llvm::SmallVector<std::pair<unsigned, ClassType>, 4> tupleElementClasses;
  class_element::Tuple::slots(logicalType, tupleElementClasses);
  if (!tupleElementClasses.empty()) {
    class_container::Elements::tuple(loc, descriptor, tupleElementClasses,
                                     module, builder, suffix);
    return;
  }

  auto [dictKeyClass, dictValueClass] = class_element::Dict::types(logicalType);
  if (dictKeyClass || dictValueClass)
    class_container::Elements::dict(loc, descriptor, dictKeyClass,
                                    dictValueClass, module, builder, suffix);
}
} // namespace class_container::Elements

namespace class_container::MemRef {
void free(mlir::Location loc, mlir::Value descriptor, mlir::Type logicalType,
          mlir::ModuleOp module, mlir::OpBuilder &builder,
          llvm::StringRef deallocGroup = {}) {
  auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  auto voidType = mlir::LLVM::LLVMVoidType::get(builder.getContext());
  auto freeFn = getOrInsertLLVMFunc(
      loc, module, builder, RuntimeSymbols::kMemFree, voidType, {ptrType});
  auto freeRef =
      mlir::SymbolRefAttr::get(module.getContext(), freeFn.getName());
  std::string group = deallocGroup.str();
  if (group.empty()) {
    mlir::Value header =
        class_container::Descriptor::header(loc, descriptor, builder);
    group = class_container::Atomic::group(header);
  }
  for (mlir::Value part : class_container::Descriptor::parts(
           loc, descriptor, logicalType, builder)) {
    mlir::Value allocated =
        class_container::Descriptor::dataPtr(loc, part, 0, builder);
    auto call = builder.create<mlir::LLVM::CallOp>(
        loc, mlir::TypeRange{}, freeRef, mlir::ValueRange{allocated});
    if (!group.empty())
      call->setAttr(ContainerSafetyAttrs::kDeallocGroup,
                    builder.getStringAttr(group));
  }
}
} // namespace class_container::MemRef

namespace class_container::Field {
void retainIfManaged(mlir::Location loc, mlir::Value descriptor,
                     mlir::Type logicalType, mlir::ModuleOp module,
                     mlir::OpBuilder &builder, llvm::StringRef premise) {
  (void)module;
  auto refcountSlot = class_container::Refcount::slot(logicalType);
  if (!refcountSlot)
    return;

  auto *parent = builder.getInsertionBlock()->getParent();
  auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  auto i64Type = builder.getI64Type();
  mlir::Block *currentBlock = builder.getInsertionBlock();
  mlir::Block *nonNullBlock = builder.createBlock(parent);
  mlir::Block *retainBlock = builder.createBlock(parent);
  mlir::Block *afterBlock = builder.createBlock(parent);

  builder.setInsertionPointToEnd(currentBlock);
  mlir::Value header =
      class_container::Descriptor::header(loc, descriptor, builder);
  mlir::Value headerData =
      class_container::Descriptor::dataPtr(loc, header, 1, builder);
  mlir::Value nullPtr = builder.create<mlir::LLVM::ZeroOp>(loc, ptrType);
  mlir::Value hasHeader = builder.create<mlir::LLVM::ICmpOp>(
      loc, mlir::LLVM::ICmpPredicate::ne, headerData, nullPtr);
  builder.create<mlir::LLVM::CondBrOp>(loc, hasHeader, nonNullBlock,
                                       afterBlock);

  builder.setInsertionPointToStart(nonNullBlock);
  mlir::Value slot = builder.create<mlir::LLVM::ConstantOp>(
      loc, i64Type, builder.getI64IntegerAttr(*refcountSlot));
  mlir::Value refcountPtr = builder.create<mlir::LLVM::GEPOp>(
      loc, ptrType, i64Type, headerData,
      llvm::ArrayRef<mlir::LLVM::GEPArg>{slot});
  mlir::Value zero = builder.create<mlir::LLVM::ConstantOp>(
      loc, i64Type, builder.getI64IntegerAttr(0));
  mlir::Value refcount = atomic_storage::Integer::rmw(
      loc, mlir::LLVM::AtomicBinOp::add, mlir::arith::AtomicRMWKind::addi,
      refcountPtr, zero, mlir::ValueRange{},
      mlir::LLVM::AtomicOrdering::acquire, builder);
  threadsafe::Atomic::set(refcount.getDefiningOp(),
                          ThreadSafetyAttrs::kRoleContainerRefcountLoad,
                          ThreadSafetyAttrs::kOrderingAcquire);
  class_container::Atomic::markHeader(refcount.getDefiningOp(), header,
                                      refcountSlot, logicalType);
  mlir::Value isManaged = builder.create<mlir::LLVM::ICmpOp>(
      loc, mlir::LLVM::ICmpPredicate::ne, refcount, zero);
  builder.create<mlir::LLVM::CondBrOp>(loc, isManaged, retainBlock, afterBlock);

  builder.setInsertionPointToStart(retainBlock);
  mlir::Value one = builder.create<mlir::LLVM::ConstantOp>(
      loc, i64Type, builder.getI64IntegerAttr(1));
  mlir::Value previous = atomic_storage::Integer::rmw(
      loc, mlir::LLVM::AtomicBinOp::add, mlir::arith::AtomicRMWKind::addi,
      refcountPtr, one, mlir::ValueRange{},
      mlir::LLVM::AtomicOrdering::monotonic, builder);
  threadsafe::Atomic::set(previous.getDefiningOp(),
                          ThreadSafetyAttrs::kRoleContainerRefcountRetain,
                          ThreadSafetyAttrs::kOrderingMonotonic, premise);
  if (premise == ThreadSafetyAttrs::kPremiseOwnedToken)
    threadsafe::Retain::verifyOwnedToken(
        previous.getDefiningOp(), ThreadSafetyAttrs::kProofClassFieldHelper);
  class_container::Atomic::markHeader(previous.getDefiningOp(), header,
                                      refcountSlot, logicalType);
  builder.create<mlir::LLVM::BrOp>(loc, mlir::ValueRange{}, afterBlock);

  builder.setInsertionPointToStart(afterBlock);
}

void destroy(mlir::Location loc, mlir::Value descriptor, mlir::Type logicalType,
             mlir::ModuleOp module, mlir::OpBuilder &builder) {
  auto refcountSlot = class_container::Refcount::slot(logicalType);
  if (!refcountSlot)
    return;

  auto *parent = builder.getInsertionBlock()->getParent();
  auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  auto i64Type = builder.getI64Type();
  mlir::Block *currentBlock = builder.getInsertionBlock();
  mlir::Block *nonNullBlock = builder.createBlock(parent);
  mlir::Block *managedBlock = builder.createBlock(parent);
  mlir::Block *localBlock = builder.createBlock(parent);
  mlir::Block *freeBlock = builder.createBlock(parent);
  mlir::Block *afterBlock = builder.createBlock(parent);

  builder.setInsertionPointToEnd(currentBlock);
  mlir::Value header =
      class_container::Descriptor::header(loc, descriptor, builder);
  mlir::Value headerData =
      class_container::Descriptor::dataPtr(loc, header, 1, builder);
  mlir::Value nullPtr = builder.create<mlir::LLVM::ZeroOp>(loc, ptrType);
  mlir::Value hasHeader = builder.create<mlir::LLVM::ICmpOp>(
      loc, mlir::LLVM::ICmpPredicate::ne, headerData, nullPtr);
  builder.create<mlir::LLVM::CondBrOp>(loc, hasHeader, nonNullBlock,
                                       afterBlock);

  builder.setInsertionPointToStart(nonNullBlock);
  mlir::Value slot = builder.create<mlir::LLVM::ConstantOp>(
      loc, i64Type, builder.getI64IntegerAttr(*refcountSlot));
  mlir::Value refcountPtr = builder.create<mlir::LLVM::GEPOp>(
      loc, ptrType, i64Type, headerData,
      llvm::ArrayRef<mlir::LLVM::GEPArg>{slot});
  mlir::Value zero = builder.create<mlir::LLVM::ConstantOp>(
      loc, i64Type, builder.getI64IntegerAttr(0));
  mlir::Value refcount = atomic_storage::Integer::rmw(
      loc, mlir::LLVM::AtomicBinOp::add, mlir::arith::AtomicRMWKind::addi,
      refcountPtr, zero, mlir::ValueRange{},
      mlir::LLVM::AtomicOrdering::acquire, builder);
  threadsafe::Atomic::set(refcount.getDefiningOp(),
                          ThreadSafetyAttrs::kRoleContainerRefcountLoad,
                          ThreadSafetyAttrs::kOrderingAcquire);
  class_container::Atomic::markHeader(refcount.getDefiningOp(), header,
                                      refcountSlot, logicalType);
  mlir::Value isManaged = builder.create<mlir::LLVM::ICmpOp>(
      loc, mlir::LLVM::ICmpPredicate::ne, refcount, zero);
  builder.create<mlir::LLVM::CondBrOp>(loc, isManaged, managedBlock,
                                       localBlock);

  builder.setInsertionPointToStart(managedBlock);
  mlir::Value one = builder.create<mlir::LLVM::ConstantOp>(
      loc, i64Type, builder.getI64IntegerAttr(1));
  mlir::Value negOne = builder.create<mlir::LLVM::ConstantOp>(
      loc, i64Type, builder.getI64IntegerAttr(-1));
  mlir::Value previous = atomic_storage::Integer::rmw(
      loc, mlir::LLVM::AtomicBinOp::add, mlir::arith::AtomicRMWKind::addi,
      refcountPtr, negOne, mlir::ValueRange{},
      mlir::LLVM::AtomicOrdering::acq_rel, builder);
  threadsafe::Atomic::set(previous.getDefiningOp(),
                          ThreadSafetyAttrs::kRoleContainerRefcountRelease,
                          ThreadSafetyAttrs::kOrderingAcqRel);
  class_container::Atomic::markHeader(previous.getDefiningOp(), header,
                                      refcountSlot, logicalType);
  mlir::Value shouldFree = builder.create<mlir::LLVM::ICmpOp>(
      loc, mlir::LLVM::ICmpPredicate::eq, previous, one);
  builder.create<mlir::LLVM::CondBrOp>(loc, shouldFree, freeBlock, afterBlock);

  builder.setInsertionPointToStart(localBlock);
  class_container::Elements::refcount(loc, descriptor, logicalType, module,
                                      builder, "decref");
  builder.create<mlir::LLVM::BrOp>(loc, mlir::ValueRange{}, afterBlock);

  builder.setInsertionPointToStart(freeBlock);
  class_container::Elements::refcount(loc, descriptor, logicalType, module,
                                      builder, "decref");
  class_container::MemRef::free(loc, descriptor, logicalType, module, builder,
                                class_container::Atomic::group(header));
  builder.create<mlir::LLVM::BrOp>(loc, mlir::ValueRange{}, afterBlock);

  builder.setInsertionPointToStart(afterBlock);
}
} // namespace class_container::Field

namespace class_field::Runtime {
void forEach(mlir::Location loc, mlir::Value object,
             const StaticClassLayout &layout, mlir::ModuleOp module,
             mlir::OpBuilder &builder, const PyLLVMTypeConverter &typeConverter,
             llvm::StringRef runtimeSymbol,
             bool includeContainerFields = true) {
  auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  auto voidType = mlir::LLVM::LLVMVoidType::get(builder.getContext());
  auto runtimeFn = getOrInsertLLVMFunc(loc, module, builder, runtimeSymbol,
                                       voidType, {ptrType});
  auto runtimeRef =
      mlir::SymbolRefAttr::get(module.getContext(), runtimeFn.getName());

  for (auto [index, fieldInfo] : llvm::enumerate(layout.fields)) {
    bool needsObjectRefcount = class_field::Refcount::needed(
        fieldInfo.logicalType, fieldInfo.storageType, typeConverter);
    ClassType listElementClass =
        includeContainerFields
            ? class_element::List::type(fieldInfo.logicalType)
            : ClassType{};
    llvm::SmallVector<std::pair<unsigned, ClassType>, 4> tupleElementClasses;
    if (includeContainerFields)
      class_element::Tuple::slots(fieldInfo.logicalType, tupleElementClasses);
    auto [dictKeyClass, dictValueClass] =
        includeContainerFields
            ? class_element::Dict::types(fieldInfo.logicalType)
            : std::pair<ClassType, ClassType>{};
    if (!needsObjectRefcount && !listElementClass &&
        tupleElementClasses.empty() && !dictKeyClass && !dictValueClass)
      continue;

    mlir::Value fieldPtr = builder.create<mlir::LLVM::GEPOp>(
        loc, ptrType, layout.storageType, object,
        llvm::ArrayRef<mlir::LLVM::GEPArg>{0, static_cast<int32_t>(index)});
    mlir::Value fieldValue = builder.create<mlir::LLVM::LoadOp>(
        loc, fieldInfo.storageType, fieldPtr);
    if (needsObjectRefcount) {
      auto call = builder.create<mlir::LLVM::CallOp>(
          loc, mlir::TypeRange{}, runtimeRef, mlir::ValueRange{fieldValue});
      if (runtimeSymbol == RuntimeSymbols::kIncRef)
        ownership::aggregate::Slot::markLoad(fieldValue);
      if (runtimeSymbol == RuntimeSymbols::kIncRef)
        call->setAttr(OwnershipContractAttrs::kAggregateRetain,
                      builder.getUnitAttr());
      if (runtimeSymbol == RuntimeSymbols::kIncRef)
        threadsafe::Retain::premise(call.getOperation(),
                                    ThreadSafetyAttrs::kPremiseAggregateBorrow);
      else if (runtimeSymbol == RuntimeSymbols::kDecRef) {
        ownership::aggregate::Slot::markLoad(fieldValue);
        call->setAttr(OwnershipContractAttrs::kAggregateRelease,
                      builder.getUnitAttr());
      }
      continue;
    }

    llvm::StringRef suffix;
    if (runtimeSymbol == RuntimeSymbols::kIncRef) {
      suffix = "incref";
    } else if (runtimeSymbol == RuntimeSymbols::kDecRef) {
      suffix = "decref";
    } else {
      continue;
    }
    if (listElementClass)
      class_container::Elements::list(loc, fieldValue, listElementClass, module,
                                      builder, suffix);
    if (!tupleElementClasses.empty())
      class_container::Elements::tuple(loc, fieldValue, tupleElementClasses,
                                       module, builder, suffix);
    if (dictKeyClass || dictValueClass)
      class_container::Elements::dict(loc, fieldValue, dictKeyClass,
                                      dictValueClass, module, builder, suffix);
  }
}

void single(mlir::Location loc, mlir::Value fieldValue, mlir::ModuleOp module,
            mlir::OpBuilder &builder, llvm::StringRef runtimeSymbol,
            llvm::StringRef retainPremise =
                ThreadSafetyAttrs::kPremiseAggregateBorrow) {
  auto call = emitRuntimeVoidCall(loc, module, builder, runtimeSymbol,
                                  mlir::ValueRange{fieldValue});
  if (runtimeSymbol == RuntimeSymbols::kIncRef) {
    ownership::aggregate::Slot::markLoad(fieldValue);
    call->setAttr(OwnershipContractAttrs::kAggregateRetain,
                  builder.getUnitAttr());
    threadsafe::Retain::premise(call.getOperation(), retainPremise);
  } else if (runtimeSymbol == RuntimeSymbols::kDecRef) {
    ownership::aggregate::Slot::markLoad(fieldValue);
    call->setAttr(OwnershipContractAttrs::kAggregateRelease,
                  builder.getUnitAttr());
  }
}
} // namespace class_field::Runtime

namespace class_field::Destroy {
void all(mlir::Location loc, mlir::Value object,
         const StaticClassLayout &layout, mlir::ModuleOp module,
         mlir::OpBuilder &builder, const PyLLVMTypeConverter &typeConverter) {
  auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  for (auto [index, fieldInfo] : llvm::enumerate(layout.fields)) {
    mlir::Value fieldPtr = builder.create<mlir::LLVM::GEPOp>(
        loc, ptrType, layout.storageType, object,
        llvm::ArrayRef<mlir::LLVM::GEPArg>{0, static_cast<int32_t>(index)});
    mlir::Value fieldValue = builder.create<mlir::LLVM::LoadOp>(
        loc, fieldInfo.storageType, fieldPtr);
    if (class_field::Refcount::needed(fieldInfo.logicalType,
                                      fieldInfo.storageType, typeConverter)) {
      class_field::Runtime::single(loc, fieldValue, module, builder,
                                   RuntimeSymbols::kDecRef);
      continue;
    }
    if (mlir::isa<ListType, TupleType, DictType>(fieldInfo.logicalType)) {
      ownership::aggregate::Slot::markLoad(fieldValue);
      class_container::Field::destroy(loc, fieldValue, fieldInfo.logicalType,
                                      module, builder);
    }
  }
}
} // namespace class_field::Destroy

namespace class_object::Size {
mlir::Value bytes(mlir::Location loc, const StaticClassLayout &layout,
                  mlir::OpBuilder &builder) {
  auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  auto i64Type = builder.getI64Type();
  mlir::Value nullPtr = builder.create<mlir::LLVM::ZeroOp>(loc, ptrType);
  mlir::Value one = builder.create<mlir::LLVM::ConstantOp>(
      loc, i64Type, builder.getI64IntegerAttr(1));
  mlir::Value endPtr = builder.create<mlir::LLVM::GEPOp>(
      loc, ptrType, layout.objectType, nullPtr,
      llvm::ArrayRef<mlir::Value>{one});
  return builder.create<mlir::LLVM::PtrToIntOp>(loc, i64Type, endPtr);
}
} // namespace class_object::Size

namespace class_helper::Retain {
mlir::LLVM::LLVMFuncOp get(mlir::Location loc, mlir::ModuleOp module,
                           ClassType classType, const StaticClassLayout &layout,
                           mlir::OpBuilder &builder,
                           const PyLLVMTypeConverter &typeConverter) {
  auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  auto voidType = mlir::LLVM::LLVMVoidType::get(builder.getContext());
  std::string helperName = getClassHelperName(classType, "incref");
  auto fn = getOrInsertLLVMFunc(loc, module, builder, helperName, voidType,
                                {ptrType});
  ownership::llvm_func::Contract::apply(fn, helperName);
  class_helper::Kind::mark(fn, classType, ClassSafetyAttrs::kKindIncref);
  class_helper::Schema::mark(fn, layout, typeConverter);
  fn.setVisibility(mlir::SymbolTable::Visibility::Private);
  if (!fn.getBody().empty())
    return fn;

  mlir::Block *entry = fn.addEntryBlock(builder);
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(entry);

  mlir::Value object = entry->getArgument(0);
  mlir::Value nullPtr = builder.create<mlir::LLVM::ZeroOp>(loc, ptrType);
  mlir::Value isNull = builder.create<mlir::LLVM::ICmpOp>(
      loc, mlir::LLVM::ICmpPredicate::eq, object, nullPtr);
  mlir::Block *retBlock = builder.createBlock(&fn.getBody());
  mlir::Block *dispatchBlock = builder.createBlock(&fn.getBody());
  builder.setInsertionPointToEnd(entry);
  builder.create<mlir::LLVM::CondBrOp>(loc, isNull, retBlock, dispatchBlock);

  builder.setInsertionPointToStart(dispatchBlock);
  mlir::Value managedPtr =
      class_object::Managed::flagPtr(loc, object, layout, builder);
  mlir::Value managed =
      builder.create<mlir::LLVM::LoadOp>(loc, builder.getI1Type(), managedPtr);
  mlir::Block *managedBlock = builder.createBlock(&fn.getBody());
  mlir::Block *localBlock = builder.createBlock(&fn.getBody());
  builder.setInsertionPointToEnd(dispatchBlock);
  builder.create<mlir::LLVM::CondBrOp>(loc, managed, managedBlock, localBlock);

  builder.setInsertionPointToStart(managedBlock);
  mlir::Value refcountPtr =
      class_object::Refcount::ptr(loc, object, layout, builder);
  class_object::Refcount::inc(loc, refcountPtr, module, builder);
  builder.create<mlir::LLVM::BrOp>(loc, mlir::ValueRange{}, retBlock);

  builder.setInsertionPointToStart(localBlock);
  class_field::Runtime::forEach(loc, object, layout, module, builder,
                                typeConverter, RuntimeSymbols::kIncRef,
                                /*includeContainerFields=*/false);
  builder.create<mlir::LLVM::BrOp>(loc, mlir::ValueRange{}, retBlock);

  builder.setInsertionPointToStart(retBlock);
  builder.create<mlir::LLVM::ReturnOp>(loc, mlir::ValueRange{});
  return fn;
}
} // namespace class_helper::Retain

namespace class_helper::Release {
mlir::LLVM::LLVMFuncOp get(mlir::Location loc, mlir::ModuleOp module,
                           ClassType classType, const StaticClassLayout &layout,
                           mlir::OpBuilder &builder,
                           const PyLLVMTypeConverter &typeConverter) {
  auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  auto voidType = mlir::LLVM::LLVMVoidType::get(builder.getContext());
  std::string helperName = getClassHelperName(classType, "decref");
  auto fn = getOrInsertLLVMFunc(loc, module, builder, helperName, voidType,
                                {ptrType});
  ownership::llvm_func::Contract::apply(fn, helperName);
  class_helper::Kind::mark(fn, classType, ClassSafetyAttrs::kKindDecref);
  class_helper::Schema::mark(fn, layout, typeConverter);
  fn.setVisibility(mlir::SymbolTable::Visibility::Private);
  if (!fn.getBody().empty())
    return fn;

  mlir::Block *entry = fn.addEntryBlock(builder);
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(entry);

  mlir::Value object = entry->getArgument(0);
  mlir::Value nullPtr = builder.create<mlir::LLVM::ZeroOp>(loc, ptrType);
  mlir::Value isNull = builder.create<mlir::LLVM::ICmpOp>(
      loc, mlir::LLVM::ICmpPredicate::eq, object, nullPtr);
  mlir::Block *retBlock = builder.createBlock(&fn.getBody());
  mlir::Block *dispatchBlock = builder.createBlock(&fn.getBody());
  builder.setInsertionPointToEnd(entry);
  builder.create<mlir::LLVM::CondBrOp>(loc, isNull, retBlock, dispatchBlock);

  builder.setInsertionPointToStart(dispatchBlock);
  mlir::Value managedPtr =
      class_object::Managed::flagPtr(loc, object, layout, builder);
  mlir::Value managed =
      builder.create<mlir::LLVM::LoadOp>(loc, builder.getI1Type(), managedPtr);
  mlir::Block *managedBlock = builder.createBlock(&fn.getBody());
  mlir::Block *localBlock = builder.createBlock(&fn.getBody());
  builder.setInsertionPointToEnd(dispatchBlock);
  builder.create<mlir::LLVM::CondBrOp>(loc, managed, managedBlock, localBlock);

  builder.setInsertionPointToStart(localBlock);
  class_field::Runtime::forEach(loc, object, layout, module, builder,
                                typeConverter, RuntimeSymbols::kDecRef,
                                /*includeContainerFields=*/false);
  builder.create<mlir::LLVM::BrOp>(loc, mlir::ValueRange{}, retBlock);

  builder.setInsertionPointToStart(managedBlock);
  mlir::Value refcountPtr =
      class_object::Refcount::ptr(loc, object, layout, builder);
  mlir::Value isZero =
      class_object::Refcount::decAndIsZero(loc, refcountPtr, module, builder);
  mlir::Block *destroyBlock = builder.createBlock(&fn.getBody());
  mlir::Block *managedRetBlock = builder.createBlock(&fn.getBody());
  builder.setInsertionPointToEnd(managedBlock);
  builder.create<mlir::LLVM::CondBrOp>(loc, isZero, destroyBlock,
                                       managedRetBlock);

  builder.setInsertionPointToStart(destroyBlock);
  class_field::Destroy::all(loc, object, layout, module, builder,
                            typeConverter);
  auto freeFn = getOrInsertLLVMFunc(
      loc, module, builder, RuntimeSymbols::kMemFree, voidType, {ptrType});
  auto freeRef =
      mlir::SymbolRefAttr::get(module.getContext(), freeFn.getName());
  builder.create<mlir::LLVM::CallOp>(loc, mlir::TypeRange{}, freeRef,
                                     mlir::ValueRange{object});
  builder.create<mlir::LLVM::BrOp>(loc, mlir::ValueRange{}, retBlock);

  builder.setInsertionPointToStart(managedRetBlock);
  builder.create<mlir::LLVM::BrOp>(loc, mlir::ValueRange{}, retBlock);

  builder.setInsertionPointToStart(retBlock);
  builder.create<mlir::LLVM::ReturnOp>(loc, mlir::ValueRange{});
  return fn;
}
} // namespace class_helper::Release

namespace class_helper::LocalDestroy {
mlir::LLVM::LLVMFuncOp get(mlir::Location loc, mlir::ModuleOp module,
                           ClassType classType, const StaticClassLayout &layout,
                           mlir::OpBuilder &builder,
                           const PyLLVMTypeConverter &typeConverter) {
  auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  auto voidType = mlir::LLVM::LLVMVoidType::get(builder.getContext());
  std::string helperName = getClassHelperName(classType, "destroy_local");
  auto fn = getOrInsertLLVMFunc(loc, module, builder, helperName, voidType,
                                {ptrType});
  ownership::llvm_func::Contract::apply(fn, helperName);
  class_helper::Kind::mark(fn, classType, ClassSafetyAttrs::kKindDestroyLocal);
  class_helper::Schema::mark(fn, layout, typeConverter);
  fn.setVisibility(mlir::SymbolTable::Visibility::Private);
  if (!fn.getBody().empty())
    return fn;

  mlir::Block *entry = fn.addEntryBlock(builder);
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(entry);

  mlir::Value object = entry->getArgument(0);
  mlir::Value nullPtr = builder.create<mlir::LLVM::ZeroOp>(loc, ptrType);
  mlir::Value isNull = builder.create<mlir::LLVM::ICmpOp>(
      loc, mlir::LLVM::ICmpPredicate::eq, object, nullPtr);
  mlir::Block *retBlock = builder.createBlock(&fn.getBody());
  mlir::Block *destroyBlock = builder.createBlock(&fn.getBody());
  builder.setInsertionPointToEnd(entry);
  builder.create<mlir::LLVM::CondBrOp>(loc, isNull, retBlock, destroyBlock);

  builder.setInsertionPointToStart(destroyBlock);
  class_field::Destroy::all(loc, object, layout, module, builder,
                            typeConverter);
  builder.create<mlir::LLVM::BrOp>(loc, mlir::ValueRange{}, retBlock);

  builder.setInsertionPointToStart(retBlock);
  builder.create<mlir::LLVM::ReturnOp>(loc, mlir::ValueRange{});
  return fn;
}
} // namespace class_helper::LocalDestroy

namespace class_helper::Repr {
void ensure(mlir::Location loc, mlir::ModuleOp module, ClassType classType,
            mlir::OpBuilder &builder) {
  auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  std::string helperName = getClassHelperName(classType, "repr");
  std::string customName = (classType.getClassName() + ".__repr__").str();
  if (auto customFunc = module.lookupSymbol<mlir::func::FuncOp>(customName)) {
    if (module.lookupSymbol<mlir::func::FuncOp>(helperName))
      return;
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(module.getBody());
    auto funcType =
        mlir::FunctionType::get(module.getContext(), {ptrType}, {ptrType});
    auto wrapper =
        builder.create<mlir::func::FuncOp>(loc, helperName, funcType);
    wrapper.setVisibility(mlir::SymbolTable::Visibility::Private);
    wrapper->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
    wrapper->setAttr(OwnershipContractAttrs::kOwnedResults,
                     builder.getArrayAttr({builder.getI64IntegerAttr(0)}));
    mlir::Block *entry = wrapper.addEntryBlock();
    builder.setInsertionPointToStart(entry);
    auto call = builder.create<mlir::func::CallOp>(
        loc, customFunc, mlir::ValueRange{entry->getArgument(0)});
    builder.create<mlir::func::ReturnOp>(loc, call.getResults());
    return;
  }

  auto fn =
      getOrInsertLLVMFunc(loc, module, builder, helperName, ptrType, {ptrType});
  ownership::llvm_func::Contract::apply(fn, helperName);
  fn->setAttr(OwnershipContractAttrs::kOwnedResults,
              builder.getArrayAttr({builder.getI64IntegerAttr(0)}));
  fn.setVisibility(mlir::SymbolTable::Visibility::Private);
  if (!fn.getBody().empty())
    return;

  mlir::Block *entry = fn.addEntryBlock(builder);
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(entry);
  mlir::Value namePtr =
      string_literal::ptr(loc, module, builder, classType.getClassName());
  auto reprFn =
      getOrInsertLLVMFunc(loc, module, builder, RuntimeSymbols::kClassReprNamed,
                          ptrType, {ptrType, ptrType});
  auto reprRef =
      mlir::SymbolRefAttr::get(module.getContext(), reprFn.getName());
  auto call = builder.create<mlir::LLVM::CallOp>(
      loc, mlir::TypeRange{ptrType}, reprRef,
      mlir::ValueRange{namePtr, entry->getArgument(0)});
  builder.create<mlir::LLVM::ReturnOp>(loc, call.getResults());
}
} // namespace class_helper::Repr

namespace class_helper::Eq {
mlir::LLVM::LLVMFuncOp get(mlir::Location loc, mlir::ModuleOp module,
                           ClassType classType, mlir::OpBuilder &builder) {
  auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  auto i1Type = builder.getI1Type();
  std::string helperName = getClassHelperName(classType, "eq");
  auto fn = getOrInsertLLVMFunc(loc, module, builder, helperName, i1Type,
                                {ptrType, ptrType});
  fn.setVisibility(mlir::SymbolTable::Visibility::Private);
  if (!fn.getBody().empty())
    return fn;

  mlir::Block *entry = fn.addEntryBlock(builder);
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(entry);
  mlir::Value equal = builder.create<mlir::LLVM::ICmpOp>(
      loc, mlir::LLVM::ICmpPredicate::eq, entry->getArgument(0),
      entry->getArgument(1));
  builder.create<mlir::LLVM::ReturnOp>(loc, equal);
  return fn;
}
} // namespace class_helper::Eq

namespace class_container::Clone {
int64_t elementBytes(mlir::Type elementType) {
  if (mlir::isa<mlir::IntegerType>(elementType))
    return std::max<int64_t>(
        1, mlir::cast<mlir::IntegerType>(elementType).getWidth() / 8);
  if (mlir::isa<mlir::FloatType>(elementType))
    return std::max<int64_t>(
        1, mlir::cast<mlir::FloatType>(elementType).getWidth() / 8);
  return 0;
}

mlir::Value rank1Descriptor(mlir::Location loc, mlir::Value descriptor,
                            mlir::MemRefType memrefType,
                            mlir::FlatSymbolRefAttr allocRef,
                            mlir::OpBuilder &builder) {
  auto descriptorType =
      mlir::cast<mlir::LLVM::LLVMStructType>(descriptor.getType());
  mlir::Type elementType = memrefType.getElementType();
  int64_t elementBytes = class_container::Clone::elementBytes(elementType);
  if (elementBytes <= 0)
    return descriptor;

  auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  auto i64Type = builder.getI64Type();
  mlir::Value oldData = builder.create<mlir::LLVM::ExtractValueOp>(
      loc, ptrType, descriptor, builder.getDenseI64ArrayAttr({1}));
  mlir::Value size = builder.create<mlir::LLVM::ExtractValueOp>(
      loc, i64Type, descriptor, builder.getDenseI64ArrayAttr({3, 0}));
  mlir::Value bytesPerElement = builder.create<mlir::LLVM::ConstantOp>(
      loc, i64Type, builder.getI64IntegerAttr(elementBytes));
  mlir::Value byteSize =
      builder.create<mlir::LLVM::MulOp>(loc, size, bytesPerElement);
  auto allocCall = builder.create<mlir::LLVM::CallOp>(
      loc, mlir::TypeRange{ptrType}, allocRef, mlir::ValueRange{byteSize});
  mlir::Value newData = allocCall.getResult();

  mlir::Value zero = builder.create<mlir::LLVM::ConstantOp>(
      loc, i64Type, builder.getI64IntegerAttr(0));
  mlir::Value one = builder.create<mlir::LLVM::ConstantOp>(
      loc, i64Type, builder.getI64IntegerAttr(1));
  mlir::Value cloned = builder.create<mlir::LLVM::UndefOp>(loc, descriptorType);
  cloned = builder.create<mlir::LLVM::InsertValueOp>(
      loc, descriptorType, cloned, newData, builder.getDenseI64ArrayAttr({0}));
  cloned = builder.create<mlir::LLVM::InsertValueOp>(
      loc, descriptorType, cloned, newData, builder.getDenseI64ArrayAttr({1}));
  cloned = builder.create<mlir::LLVM::InsertValueOp>(
      loc, descriptorType, cloned, zero, builder.getDenseI64ArrayAttr({2}));
  cloned = builder.create<mlir::LLVM::InsertValueOp>(
      loc, descriptorType, cloned, size, builder.getDenseI64ArrayAttr({3, 0}));
  cloned = builder.create<mlir::LLVM::InsertValueOp>(
      loc, descriptorType, cloned, one, builder.getDenseI64ArrayAttr({4, 0}));

  auto *parent = builder.getInsertionBlock()->getParent();
  mlir::Block *currentBlock = builder.getInsertionBlock();
  mlir::Block *condBlock = builder.createBlock(parent);
  condBlock->addArgument(i64Type, loc);
  mlir::Block *bodyBlock = builder.createBlock(parent);
  mlir::Block *afterBlock = builder.createBlock(parent);
  builder.setInsertionPointToEnd(currentBlock);
  builder.create<mlir::LLVM::BrOp>(loc, mlir::ValueRange{zero}, condBlock);

  builder.setInsertionPointToStart(condBlock);
  mlir::Value iv = condBlock->getArgument(0);
  mlir::Value inBounds = builder.create<mlir::LLVM::ICmpOp>(
      loc, mlir::LLVM::ICmpPredicate::slt, iv, size);
  builder.create<mlir::LLVM::CondBrOp>(loc, inBounds, bodyBlock, afterBlock);

  builder.setInsertionPointToStart(bodyBlock);
  mlir::Value srcPtr =
      builder.create<mlir::LLVM::GEPOp>(loc, ptrType, elementType, oldData,
                                        llvm::ArrayRef<mlir::LLVM::GEPArg>{iv});
  mlir::Value dstPtr =
      builder.create<mlir::LLVM::GEPOp>(loc, ptrType, elementType, newData,
                                        llvm::ArrayRef<mlir::LLVM::GEPArg>{iv});
  mlir::Value item =
      builder.create<mlir::LLVM::LoadOp>(loc, elementType, srcPtr);
  builder.create<mlir::LLVM::StoreOp>(loc, item, dstPtr);
  mlir::Value next = builder.create<mlir::LLVM::AddOp>(loc, iv, one);
  builder.create<mlir::LLVM::BrOp>(loc, mlir::ValueRange{next}, condBlock);

  builder.setInsertionPointToStart(afterBlock);
  return cloned;
}

void markOwnership(mlir::Location loc, mlir::Value headerDescriptor,
                   int64_t markerSlot, int64_t markerValue,
                   mlir::OpBuilder &builder) {
  auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  auto i64Type = builder.getI64Type();
  mlir::Value data = builder.create<mlir::LLVM::ExtractValueOp>(
      loc, ptrType, headerDescriptor, builder.getDenseI64ArrayAttr({1}));
  mlir::Value slot = builder.create<mlir::LLVM::ConstantOp>(
      loc, i64Type, builder.getI64IntegerAttr(markerSlot));
  mlir::Value ptr = builder.create<mlir::LLVM::GEPOp>(
      loc, ptrType, i64Type, data, llvm::ArrayRef<mlir::LLVM::GEPArg>{slot});
  mlir::Value value = builder.create<mlir::LLVM::ConstantOp>(
      loc, i64Type, builder.getI64IntegerAttr(markerValue));
  auto store = builder.create<mlir::LLVM::StoreOp>(loc, value, ptr);
  store->setAttr(ContainerSafetyAttrs::kRefcountInit,
                 builder.getI64IntegerAttr(markerValue));
  store->setAttr(ContainerSafetyAttrs::kRefcountState,
                 builder.getStringAttr(ContainerSafetyAttrs::kStateManaged));
}

mlir::Value storage(mlir::Location loc, mlir::Value storageValue,
                    mlir::Type logicalType, mlir::FlatSymbolRefAttr allocRef,
                    mlir::OpBuilder &builder, int64_t initialRefcount = 1) {
  llvm::SmallVector<mlir::Type, 4> convertedTypes;
  auto appendDescriptor = [&](unsigned index, mlir::MemRefType memrefType,
                              mlir::Value &result) {
    auto storageStruct =
        mlir::cast<mlir::LLVM::LLVMStructType>(result.getType());
    mlir::Type partType = storageStruct.getBody()[index];
    mlir::Value descriptor = builder.create<mlir::LLVM::ExtractValueOp>(
        loc, partType, result, builder.getDenseI64ArrayAttr({index}));
    mlir::Value cloned = class_container::Clone::rank1Descriptor(
        loc, descriptor, memrefType, allocRef, builder);
    result = builder.create<mlir::LLVM::InsertValueOp>(
        loc, storageStruct, result, cloned,
        builder.getDenseI64ArrayAttr({index}));
    return cloned;
  };

  mlir::Value result = storageValue;
  if (auto listType = mlir::dyn_cast<ListType>(logicalType)) {
    auto headerType = getListHeaderMemRefType(builder.getContext());
    auto itemsType =
        getListItemsMemRefType(listType.getElementType(), builder.getContext());
    if (!itemsType)
      return storageValue;
    mlir::Value header = appendDescriptor(0, headerType, result);
    appendDescriptor(1, itemsType, result);
    class_container::Clone::markOwnership(loc, header, 3, initialRefcount,
                                          builder);
    return result;
  }
  if (auto tupleType = mlir::dyn_cast<TupleType>(logicalType)) {
    mlir::Value header = appendDescriptor(
        0, getTupleHeaderMemRefType(builder.getContext()), result);
    appendDescriptor(
        1, getTupleItemsMemRefType(tupleType, builder.getContext()), result);
    class_container::Clone::markOwnership(loc, header, 2, initialRefcount,
                                          builder);
    return result;
  }
  if (auto dictType = mlir::dyn_cast<DictType>(logicalType)) {
    auto keysType = getDictKeysMemRefType(dictType, builder.getContext());
    auto valuesType = getDictValuesMemRefType(dictType, builder.getContext());
    if (!keysType || !valuesType)
      return storageValue;
    mlir::Value header = appendDescriptor(
        0, getDictHeaderMemRefType(builder.getContext()), result);
    appendDescriptor(1, keysType, result);
    appendDescriptor(2, valuesType, result);
    appendDescriptor(3, getDictStatesMemRefType(builder.getContext()), result);
    class_container::Clone::markOwnership(loc, header, 4, initialRefcount,
                                          builder);
    return result;
  }
  return storageValue;
}

mlir::Value
retainOrCloneLocalAlias(mlir::Location loc, mlir::Value descriptor,
                        mlir::Type logicalType, mlir::Value fieldPtr,
                        mlir::FlatSymbolRefAttr allocRef, mlir::ModuleOp module,
                        mlir::OpBuilder &builder, llvm::StringRef premise) {
  (void)module;
  auto refcountSlot = class_container::Refcount::slot(logicalType);
  if (!refcountSlot)
    return descriptor;

  auto *parent = builder.getInsertionBlock()->getParent();
  auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  auto i64Type = builder.getI64Type();
  mlir::Block *currentBlock = builder.getInsertionBlock();
  mlir::Block *nonNullBlock = builder.createBlock(parent);
  mlir::Block *retainBlock = builder.createBlock(parent);
  mlir::Block *cloneBlock = builder.createBlock(parent);
  mlir::Block *afterBlock = builder.createBlock(parent);
  afterBlock->addArgument(descriptor.getType(), loc);

  builder.setInsertionPointToEnd(currentBlock);
  mlir::Value header =
      class_container::Descriptor::header(loc, descriptor, builder);
  mlir::Value headerData =
      class_container::Descriptor::dataPtr(loc, header, 1, builder);
  mlir::Value nullPtr = builder.create<mlir::LLVM::ZeroOp>(loc, ptrType);
  mlir::Value hasHeader = builder.create<mlir::LLVM::ICmpOp>(
      loc, mlir::LLVM::ICmpPredicate::ne, headerData, nullPtr);
  builder.create<mlir::LLVM::CondBrOp>(loc, hasHeader, nonNullBlock,
                                       mlir::ValueRange{}, afterBlock,
                                       mlir::ValueRange{descriptor});

  builder.setInsertionPointToStart(nonNullBlock);
  mlir::Value slot = builder.create<mlir::LLVM::ConstantOp>(
      loc, i64Type, builder.getI64IntegerAttr(*refcountSlot));
  mlir::Value refcountPtr = builder.create<mlir::LLVM::GEPOp>(
      loc, ptrType, i64Type, headerData,
      llvm::ArrayRef<mlir::LLVM::GEPArg>{slot});
  mlir::Value zero = builder.create<mlir::LLVM::ConstantOp>(
      loc, i64Type, builder.getI64IntegerAttr(0));
  mlir::Value refcount = atomic_storage::Integer::rmw(
      loc, mlir::LLVM::AtomicBinOp::add, mlir::arith::AtomicRMWKind::addi,
      refcountPtr, zero, mlir::ValueRange{},
      mlir::LLVM::AtomicOrdering::acquire, builder);
  threadsafe::Atomic::set(refcount.getDefiningOp(),
                          ThreadSafetyAttrs::kRoleContainerRefcountLoad,
                          ThreadSafetyAttrs::kOrderingAcquire);
  class_container::Atomic::markHeader(refcount.getDefiningOp(), header,
                                      refcountSlot, logicalType);
  mlir::Value isManaged = builder.create<mlir::LLVM::ICmpOp>(
      loc, mlir::LLVM::ICmpPredicate::ne, refcount, zero);
  builder.create<mlir::LLVM::CondBrOp>(loc, isManaged, retainBlock, cloneBlock);

  builder.setInsertionPointToStart(retainBlock);
  mlir::Value one = builder.create<mlir::LLVM::ConstantOp>(
      loc, i64Type, builder.getI64IntegerAttr(1));
  mlir::Value previous = atomic_storage::Integer::rmw(
      loc, mlir::LLVM::AtomicBinOp::add, mlir::arith::AtomicRMWKind::addi,
      refcountPtr, one, mlir::ValueRange{},
      mlir::LLVM::AtomicOrdering::monotonic, builder);
  threadsafe::Atomic::set(previous.getDefiningOp(),
                          ThreadSafetyAttrs::kRoleContainerRefcountRetain,
                          ThreadSafetyAttrs::kOrderingMonotonic, premise);
  if (premise == ThreadSafetyAttrs::kPremiseOwnedToken)
    threadsafe::Retain::verifyOwnedToken(
        previous.getDefiningOp(), ThreadSafetyAttrs::kProofClassFieldHelper);
  class_container::Atomic::markHeader(previous.getDefiningOp(), header,
                                      refcountSlot, logicalType);
  builder.create<mlir::LLVM::BrOp>(loc, mlir::ValueRange{descriptor},
                                   afterBlock);

  builder.setInsertionPointToStart(cloneBlock);
  mlir::Value cloned = class_container::Clone::storage(
      loc, descriptor, logicalType, allocRef, builder,
      /*initialRefcount=*/2);
  builder.create<mlir::LLVM::StoreOp>(loc, cloned, fieldPtr);
  builder.create<mlir::LLVM::BrOp>(loc, mlir::ValueRange{cloned}, afterBlock);

  builder.setInsertionPointToStart(afterBlock);
  return afterBlock->getArgument(0);
}
} // namespace class_container::Clone

namespace class_container::Clone {
void fields(mlir::Location loc, mlir::Value object,
            const StaticClassLayout &layout, mlir::FlatSymbolRefAttr allocRef,
            mlir::OpBuilder &builder) {
  auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  for (auto [index, field] : llvm::enumerate(layout.fields)) {
    if (!mlir::isa<ListType, TupleType, DictType>(field.logicalType))
      continue;
    mlir::Value fieldPtr = builder.create<mlir::LLVM::GEPOp>(
        loc, ptrType, layout.storageType, object,
        llvm::ArrayRef<mlir::LLVM::GEPArg>{0, static_cast<int32_t>(index)});
    mlir::Value fieldValue =
        builder.create<mlir::LLVM::LoadOp>(loc, field.storageType, fieldPtr);
    mlir::Value cloned = class_container::Clone::storage(
        loc, fieldValue, field.logicalType, allocRef, builder);
    builder.create<mlir::LLVM::StoreOp>(loc, cloned, fieldPtr);
  }
}
} // namespace class_container::Clone

namespace class_helper::Promote {
mlir::LLVM::LLVMFuncOp get(mlir::Location loc, mlir::ModuleOp module,
                           ClassType classType, const StaticClassLayout &layout,
                           mlir::OpBuilder &builder,
                           const PyLLVMTypeConverter &typeConverter) {
  auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  auto i64Type = builder.getI64Type();
  std::string helperName = getClassHelperName(classType, "promote");
  auto fn =
      getOrInsertLLVMFunc(loc, module, builder, helperName, ptrType, {ptrType});
  ownership::llvm_func::Contract::apply(fn, helperName);
  class_helper::Kind::mark(fn, classType, ClassSafetyAttrs::kKindPromote);
  class_helper::Schema::mark(fn, layout, typeConverter);
  fn.setVisibility(mlir::SymbolTable::Visibility::Private);
  if (!fn.getBody().empty())
    return fn;

  auto retainHelper = class_helper::Retain::get(loc, module, classType, layout,
                                                builder, typeConverter);
  auto retainRef =
      mlir::SymbolRefAttr::get(module.getContext(), retainHelper.getName());
  auto allocFn = getOrInsertLLVMFunc(
      loc, module, builder, RuntimeSymbols::kMemAlloc, ptrType, {i64Type});
  auto allocRef =
      mlir::FlatSymbolRefAttr::get(module.getContext(), allocFn.getName());

  mlir::Block *entry = fn.addEntryBlock(builder);
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(entry);

  mlir::Value object = entry->getArgument(0);
  mlir::Value nullPtr = builder.create<mlir::LLVM::ZeroOp>(loc, ptrType);
  mlir::Value isNull = builder.create<mlir::LLVM::ICmpOp>(
      loc, mlir::LLVM::ICmpPredicate::eq, object, nullPtr);
  mlir::Block *retNullBlock = builder.createBlock(&fn.getBody());
  mlir::Block *dispatchBlock = builder.createBlock(&fn.getBody());
  builder.setInsertionPointToEnd(entry);
  builder.create<mlir::LLVM::CondBrOp>(loc, isNull, retNullBlock,
                                       dispatchBlock);

  builder.setInsertionPointToStart(retNullBlock);
  builder.create<mlir::LLVM::ReturnOp>(loc, nullPtr);

  builder.setInsertionPointToStart(dispatchBlock);
  mlir::Value managedPtr =
      class_object::Managed::flagPtr(loc, object, layout, builder);
  mlir::Value managed =
      builder.create<mlir::LLVM::LoadOp>(loc, builder.getI1Type(), managedPtr);
  mlir::Block *managedBlock = builder.createBlock(&fn.getBody());
  mlir::Block *copyBlock = builder.createBlock(&fn.getBody());
  builder.setInsertionPointToEnd(dispatchBlock);
  builder.create<mlir::LLVM::CondBrOp>(loc, managed, managedBlock, copyBlock);

  builder.setInsertionPointToStart(managedBlock);
  auto retain = builder.create<mlir::LLVM::CallOp>(
      loc, mlir::TypeRange{}, retainRef, mlir::ValueRange{object});
  threadsafe::Retain::premise(retain.getOperation(),
                              ThreadSafetyAttrs::kPremiseEntryBorrowed);
  builder.create<mlir::LLVM::ReturnOp>(loc, object);

  builder.setInsertionPointToStart(copyBlock);
  mlir::Value sizeBytes = class_object::Size::bytes(loc, layout, builder);
  auto allocCall = builder.create<mlir::LLVM::CallOp>(
      loc, mlir::TypeRange{ptrType}, allocRef, mlir::ValueRange{sizeBytes});
  allocCall->setAttr(ClassSafetyAttrs::kPromoteFreshObject,
                     builder.getUnitAttr());
  mlir::Value managedObject = allocCall.getResult();
  mlir::Value storageValue =
      builder.create<mlir::LLVM::LoadOp>(loc, layout.storageType, object);
  builder.create<mlir::LLVM::StoreOp>(loc, storageValue, managedObject);
  class_container::Clone::fields(loc, managedObject, layout, allocRef, builder);
  mlir::Value newManagedPtr =
      class_object::Managed::flagPtr(loc, managedObject, layout, builder);
  mlir::Value trueValue = builder.create<mlir::LLVM::ConstantOp>(
      loc, builder.getI1Type(), builder.getBoolAttr(true));
  auto managedStore =
      builder.create<mlir::LLVM::StoreOp>(loc, trueValue, newManagedPtr);
  managedStore->setAttr(ClassSafetyAttrs::kPromoteManagedInit,
                        builder.getUnitAttr());
  mlir::Value lockPtr =
      class_object::Lock::ptr(loc, managedObject, layout, builder);
  mlir::Value zeroLock = builder.create<mlir::LLVM::ConstantOp>(
      loc, builder.getI32Type(), builder.getI32IntegerAttr(0));
  auto lockStore = builder.create<mlir::LLVM::StoreOp>(loc, zeroLock, lockPtr);
  lockStore->setAttr(ClassSafetyAttrs::kPromoteLockInit, builder.getUnitAttr());
  mlir::Value refcountPtr =
      class_object::Refcount::ptr(loc, managedObject, layout, builder);
  mlir::Value one = builder.create<mlir::LLVM::ConstantOp>(
      loc, i64Type, builder.getI64IntegerAttr(1));
  auto refcountStore =
      builder.create<mlir::LLVM::StoreOp>(loc, one, refcountPtr);
  refcountStore->setAttr(ClassSafetyAttrs::kPromoteRefcountInit,
                         builder.getUnitAttr());
  class_field::Runtime::forEach(loc, managedObject, layout, module, builder,
                                typeConverter, RuntimeSymbols::kIncRef);
  builder.create<mlir::LLVM::ReturnOp>(loc, managedObject);
  return fn;
}
} // namespace class_helper::Promote

namespace class_helper::GetField {
mlir::LLVM::LLVMFuncOp get(mlir::Location loc, mlir::ModuleOp module,
                           ClassType classType, const StaticClassLayout &layout,
                           unsigned fieldIndex, mlir::OpBuilder &builder,
                           const PyLLVMTypeConverter &typeConverter) {
  assert(fieldIndex < layout.fields.size() && "field index out of range");
  const StaticClassFieldInfo &fieldInfo = layout.fields[fieldIndex];
  auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  auto i1Type = builder.getI1Type();
  std::string helperName =
      class_helper::Field::name(classType, "getfield", fieldIndex);
  auto fn = getOrInsertLLVMFunc(loc, module, builder, helperName,
                                fieldInfo.storageType, {ptrType, i1Type});
  ownership::llvm_func::Contract::apply(fn, helperName);
  class_helper::Field::mark(fn, classType, ClassSafetyAttrs::kKindGetField,
                            fieldIndex);
  fn.setVisibility(mlir::SymbolTable::Visibility::Private);
  if (!fn.getBody().empty())
    return fn;

  bool needsRefcount = class_field::Refcount::needed(
      fieldInfo.logicalType, fieldInfo.storageType, typeConverter);
  bool needsContainerRefcount =
      mlir::isa<ListType, TupleType, DictType>(fieldInfo.logicalType);
  mlir::FlatSymbolRefAttr allocRef;
  if (needsContainerRefcount) {
    auto allocFn =
        getOrInsertLLVMFunc(loc, module, builder, RuntimeSymbols::kMemAlloc,
                            ptrType, {builder.getI64Type()});
    allocRef =
        mlir::FlatSymbolRefAttr::get(module.getContext(), allocFn.getName());
  }

  mlir::Block *entry = fn.addEntryBlock(builder);
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(entry);

  mlir::Value object = entry->getArgument(0);
  mlir::Value borrowLocal = entry->getArgument(1);
  mlir::Value managedPtr =
      class_object::Managed::flagPtr(loc, object, layout, builder);
  mlir::Value managed =
      builder.create<mlir::LLVM::LoadOp>(loc, i1Type, managedPtr);
  mlir::Block *managedBlock = builder.createBlock(&fn.getBody());
  mlir::Block *localBlock = builder.createBlock(&fn.getBody());
  builder.setInsertionPointToEnd(entry);
  builder.create<mlir::LLVM::CondBrOp>(loc, managed, managedBlock, localBlock);

  builder.setInsertionPointToStart(localBlock);
  mlir::Value localFieldPtr = builder.create<mlir::LLVM::GEPOp>(
      loc, ptrType, layout.storageType, object,
      llvm::ArrayRef<mlir::LLVM::GEPArg>{0, static_cast<int32_t>(fieldIndex)});
  mlir::Value localValue = builder.create<mlir::LLVM::LoadOp>(
      loc, fieldInfo.storageType, localFieldPtr);
  if (!needsRefcount && !needsContainerRefcount) {
    builder.create<mlir::LLVM::ReturnOp>(loc, localValue);
  } else {
    mlir::Block *localBorrowBlock = builder.createBlock(&fn.getBody());
    mlir::Block *localRetainBlock = builder.createBlock(&fn.getBody());
    builder.setInsertionPointToEnd(localBlock);
    builder.create<mlir::LLVM::CondBrOp>(loc, borrowLocal, localBorrowBlock,
                                         localRetainBlock);

    builder.setInsertionPointToStart(localBorrowBlock);
    builder.create<mlir::LLVM::ReturnOp>(loc, localValue);

    builder.setInsertionPointToStart(localRetainBlock);
    if (needsRefcount) {
      class_field::Runtime::single(loc, localValue, module, builder,
                                   RuntimeSymbols::kIncRef);
    } else {
      ownership::aggregate::Slot::markLoad(localValue);
      localValue = class_container::Clone::retainOrCloneLocalAlias(
          loc, localValue, fieldInfo.logicalType, localFieldPtr, allocRef,
          module, builder, ThreadSafetyAttrs::kPremiseAggregateBorrow);
    }
    builder.create<mlir::LLVM::ReturnOp>(loc, localValue);
  }

  builder.setInsertionPointToStart(managedBlock);
  mlir::Value lockPtr = class_object::Lock::ptr(loc, object, layout, builder);
  class_object::Lock::acquire(loc, lockPtr, module, builder);
  mlir::Value managedFieldPtr = builder.create<mlir::LLVM::GEPOp>(
      loc, ptrType, layout.storageType, object,
      llvm::ArrayRef<mlir::LLVM::GEPArg>{0, static_cast<int32_t>(fieldIndex)});
  mlir::Value managedValue = builder.create<mlir::LLVM::LoadOp>(
      loc, fieldInfo.storageType, managedFieldPtr);
  if (needsRefcount) {
    class_field::Runtime::single(loc, managedValue, module, builder,
                                 RuntimeSymbols::kIncRef,
                                 ThreadSafetyAttrs::kPremiseLockedBorrow);
  } else if (needsContainerRefcount) {
    class_container::Field::retainIfManaged(
        loc, managedValue, fieldInfo.logicalType, module, builder,
        ThreadSafetyAttrs::kPremiseLockedBorrow);
  }
  class_object::Lock::release(loc, lockPtr, module, builder);
  builder.create<mlir::LLVM::ReturnOp>(loc, managedValue);
  return fn;
}
} // namespace class_helper::GetField

namespace class_helper::SetField {
mlir::LLVM::LLVMFuncOp get(mlir::Location loc, mlir::ModuleOp module,
                           ClassType classType, const StaticClassLayout &layout,
                           unsigned fieldIndex, mlir::OpBuilder &builder,
                           const PyLLVMTypeConverter &typeConverter) {
  assert(fieldIndex < layout.fields.size() && "field index out of range");
  const StaticClassFieldInfo &fieldInfo = layout.fields[fieldIndex];
  auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  auto i1Type = builder.getI1Type();
  auto voidType = mlir::LLVM::LLVMVoidType::get(builder.getContext());
  std::string helperName =
      class_helper::Field::name(classType, "setfield", fieldIndex);
  auto fn =
      getOrInsertLLVMFunc(loc, module, builder, helperName, voidType,
                          {ptrType, fieldInfo.storageType, i1Type, i1Type});
  ownership::llvm_func::Contract::apply(fn, helperName);
  class_helper::Field::mark(fn, classType, ClassSafetyAttrs::kKindSetField,
                            fieldIndex);
  fn.setVisibility(mlir::SymbolTable::Visibility::Private);
  if (!fn.getBody().empty())
    return fn;

  bool needsRefcount = class_field::Refcount::needed(
      fieldInfo.logicalType, fieldInfo.storageType, typeConverter);
  bool needsContainerRefcount =
      mlir::isa<ListType, TupleType, DictType>(fieldInfo.logicalType);

  mlir::Block *entry = fn.addEntryBlock(builder);
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(entry);

  mlir::Value object = entry->getArgument(0);
  mlir::Value newValue = entry->getArgument(1);
  mlir::Value retainNewValue = entry->getArgument(2);
  mlir::Value skipOldValueDrop = entry->getArgument(3);
  mlir::Block *dispatchBlock = builder.createBlock(&fn.getBody());
  if (needsRefcount || needsContainerRefcount) {
    mlir::Block *retainBlock = builder.createBlock(&fn.getBody());
    builder.setInsertionPointToEnd(entry);
    builder.create<mlir::LLVM::CondBrOp>(loc, retainNewValue, retainBlock,
                                         dispatchBlock);
    builder.setInsertionPointToStart(retainBlock);
    if (needsRefcount) {
      class_field::Runtime::single(loc, newValue, module, builder,
                                   RuntimeSymbols::kIncRef,
                                   ThreadSafetyAttrs::kPremiseEntryBorrowed);
    } else {
      class_container::Field::retainIfManaged(
          loc, newValue, fieldInfo.logicalType, module, builder,
          ThreadSafetyAttrs::kPremiseEntryBorrowed);
    }
    builder.create<mlir::LLVM::BrOp>(loc, mlir::ValueRange{}, dispatchBlock);
  } else {
    builder.setInsertionPointToEnd(entry);
    builder.create<mlir::LLVM::BrOp>(loc, mlir::ValueRange{}, dispatchBlock);
  }

  builder.setInsertionPointToStart(dispatchBlock);
  mlir::Value managedPtr =
      class_object::Managed::flagPtr(loc, object, layout, builder);
  mlir::Value managed =
      builder.create<mlir::LLVM::LoadOp>(loc, i1Type, managedPtr);
  mlir::Block *managedBlock = builder.createBlock(&fn.getBody());
  mlir::Block *localBlock = builder.createBlock(&fn.getBody());
  builder.setInsertionPointToEnd(dispatchBlock);
  builder.create<mlir::LLVM::CondBrOp>(loc, managed, managedBlock, localBlock);

  builder.setInsertionPointToStart(localBlock);
  mlir::Value localFieldPtr = builder.create<mlir::LLVM::GEPOp>(
      loc, ptrType, layout.storageType, object,
      llvm::ArrayRef<mlir::LLVM::GEPArg>{0, static_cast<int32_t>(fieldIndex)});
  if (!needsRefcount && !needsContainerRefcount) {
    builder.create<mlir::LLVM::StoreOp>(loc, newValue, localFieldPtr);
    builder.create<mlir::LLVM::ReturnOp>(loc, mlir::ValueRange{});
  } else {
    mlir::Block *localSkipOldBlock = builder.createBlock(&fn.getBody());
    mlir::Block *localLoadOldBlock = builder.createBlock(&fn.getBody());
    builder.setInsertionPointToEnd(localBlock);
    builder.create<mlir::LLVM::CondBrOp>(loc, skipOldValueDrop,
                                         localSkipOldBlock, localLoadOldBlock);

    builder.setInsertionPointToStart(localSkipOldBlock);
    auto store =
        builder.create<mlir::LLVM::StoreOp>(loc, newValue, localFieldPtr);
    ownership::aggregate::Slot::markStore(store.getOperation());
    builder.create<mlir::LLVM::ReturnOp>(loc, mlir::ValueRange{});

    builder.setInsertionPointToStart(localLoadOldBlock);
    mlir::Value oldValue = builder.create<mlir::LLVM::LoadOp>(
        loc, fieldInfo.storageType, localFieldPtr);
    store = builder.create<mlir::LLVM::StoreOp>(loc, newValue, localFieldPtr);
    ownership::aggregate::Slot::markStore(store.getOperation());
    if (needsRefcount) {
      class_field::Runtime::single(loc, oldValue, module, builder,
                                   RuntimeSymbols::kDecRef);
    } else {
      class_container::Field::destroy(loc, oldValue, fieldInfo.logicalType,
                                      module, builder);
    }
    builder.create<mlir::LLVM::ReturnOp>(loc, mlir::ValueRange{});
  }

  builder.setInsertionPointToStart(managedBlock);
  mlir::Value lockPtr = class_object::Lock::ptr(loc, object, layout, builder);
  if (!needsRefcount && !needsContainerRefcount) {
    class_object::Lock::acquire(loc, lockPtr, module, builder);
    mlir::Value managedFieldPtr = builder.create<mlir::LLVM::GEPOp>(
        loc, ptrType, layout.storageType, object,
        llvm::ArrayRef<mlir::LLVM::GEPArg>{0,
                                           static_cast<int32_t>(fieldIndex)});
    builder.create<mlir::LLVM::StoreOp>(loc, newValue, managedFieldPtr);
    class_object::Lock::release(loc, lockPtr, module, builder);
    builder.create<mlir::LLVM::ReturnOp>(loc, mlir::ValueRange{});
    return fn;
  }

  mlir::Block *managedSkipOldBlock = builder.createBlock(&fn.getBody());
  mlir::Block *managedLoadOldBlock = builder.createBlock(&fn.getBody());
  builder.setInsertionPointToEnd(managedBlock);
  builder.create<mlir::LLVM::CondBrOp>(
      loc, skipOldValueDrop, managedSkipOldBlock, managedLoadOldBlock);

  builder.setInsertionPointToStart(managedSkipOldBlock);
  class_object::Lock::acquire(loc, lockPtr, module, builder);
  mlir::Value managedSkipFieldPtr = builder.create<mlir::LLVM::GEPOp>(
      loc, ptrType, layout.storageType, object,
      llvm::ArrayRef<mlir::LLVM::GEPArg>{0, static_cast<int32_t>(fieldIndex)});
  auto store =
      builder.create<mlir::LLVM::StoreOp>(loc, newValue, managedSkipFieldPtr);
  ownership::aggregate::Slot::markStore(store.getOperation());
  class_object::Lock::release(loc, lockPtr, module, builder);
  builder.create<mlir::LLVM::ReturnOp>(loc, mlir::ValueRange{});

  builder.setInsertionPointToStart(managedLoadOldBlock);
  class_object::Lock::acquire(loc, lockPtr, module, builder);
  mlir::Value managedLoadFieldPtr = builder.create<mlir::LLVM::GEPOp>(
      loc, ptrType, layout.storageType, object,
      llvm::ArrayRef<mlir::LLVM::GEPArg>{0, static_cast<int32_t>(fieldIndex)});
  mlir::Value oldValue = builder.create<mlir::LLVM::LoadOp>(
      loc, fieldInfo.storageType, managedLoadFieldPtr);
  store =
      builder.create<mlir::LLVM::StoreOp>(loc, newValue, managedLoadFieldPtr);
  ownership::aggregate::Slot::markStore(store.getOperation());
  class_object::Lock::release(loc, lockPtr, module, builder);
  if (needsRefcount) {
    class_field::Runtime::single(loc, oldValue, module, builder,
                                 RuntimeSymbols::kDecRef);
  } else {
    class_container::Field::destroy(loc, oldValue, fieldInfo.logicalType,
                                    module, builder);
  }
  builder.create<mlir::LLVM::ReturnOp>(loc, mlir::ValueRange{});
  return fn;
}
} // namespace class_helper::SetField

namespace class_helper::Copy {
mlir::LLVM::LLVMFuncOp get(mlir::Location loc, mlir::ModuleOp module,
                           ClassType classType, const StaticClassLayout &layout,
                           mlir::OpBuilder &builder,
                           const PyLLVMTypeConverter &typeConverter) {
  auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  auto voidType = mlir::LLVM::LLVMVoidType::get(builder.getContext());
  std::string helperName = getClassHelperName(classType, "copy");
  auto fn = getOrInsertLLVMFunc(loc, module, builder, helperName, voidType,
                                {ptrType, ptrType});
  fn.setVisibility(mlir::SymbolTable::Visibility::Private);
  if (!fn.getBody().empty())
    return fn;

  mlir::Block *entry = fn.addEntryBlock(builder);
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(entry);

  mlir::Value dest = entry->getArgument(0);
  mlir::Value src = entry->getArgument(1);
  mlir::Value borrowLocal = builder.create<mlir::LLVM::ConstantOp>(
      loc, builder.getI1Type(), builder.getBoolAttr(false));
  mlir::Value retainNewValue = builder.create<mlir::LLVM::ConstantOp>(
      loc, builder.getI1Type(), builder.getBoolAttr(false));
  mlir::Value skipOldDrop = builder.create<mlir::LLVM::ConstantOp>(
      loc, builder.getI1Type(), builder.getBoolAttr(true));

  for (auto [fieldIndex, fieldInfo] : llvm::enumerate(layout.fields)) {
    auto getHelper = class_helper::GetField::get(
        loc, module, classType, layout, fieldIndex, builder, typeConverter);
    auto getHelperRef =
        mlir::SymbolRefAttr::get(module.getContext(), getHelper.getName());
    auto fieldValue = builder.create<mlir::LLVM::CallOp>(
        loc, mlir::TypeRange{fieldInfo.storageType}, getHelperRef,
        mlir::ValueRange{src, borrowLocal});

    auto setHelper = class_helper::SetField::get(
        loc, module, classType, layout, fieldIndex, builder, typeConverter);
    auto setHelperRef =
        mlir::SymbolRefAttr::get(module.getContext(), setHelper.getName());
    builder.create<mlir::LLVM::CallOp>(
        loc, mlir::TypeRange{}, setHelperRef,
        mlir::ValueRange{dest, fieldValue.getResult(), retainNewValue,
                         skipOldDrop});
  }

  builder.create<mlir::LLVM::ReturnOp>(loc, mlir::ValueRange{});
  return fn;
}
} // namespace class_helper::Copy

static void
ensureStaticClassHelperBodies(ClassOp classOp, const StaticClassLayout &layout,
                              mlir::ModuleOp module, mlir::OpBuilder &builder,
                              const PyLLVMTypeConverter &typeConverter) {
  ClassType classType =
      ClassType::get(classOp.getContext(), classOp.getSymNameAttr().getValue());
  class_helper::Retain::get(classOp.getLoc(), module, classType, layout,
                            builder, typeConverter);
  class_helper::Release::get(classOp.getLoc(), module, classType, layout,
                             builder, typeConverter);
  class_helper::LocalDestroy::get(classOp.getLoc(), module, classType, layout,
                                  builder, typeConverter);
  class_helper::Promote::get(classOp.getLoc(), module, classType, layout,
                             builder, typeConverter);
  class_helper::Repr::ensure(classOp.getLoc(), module, classType, builder);
  class_helper::Eq::get(classOp.getLoc(), module, classType, builder);
  class_helper::Copy::get(classOp.getLoc(), module, classType, layout, builder,
                          typeConverter);
}

static mlir::FailureOr<mlir::Value> boxStaticFieldValue(
    mlir::Location loc, mlir::Value storageValue, mlir::Type logicalType,
    mlir::ModuleOp module, mlir::ConversionPatternRewriter &rewriter,
    const PyLLVMTypeConverter &typeConverter, bool borrowExistingRef = false) {
  RuntimeAPI runtime(module, rewriter, typeConverter);
  if (mlir::isa<IntType>(logicalType) &&
      mlir::isa<mlir::IntegerType>(storageValue.getType()))
    return runtime
        .call(loc, RuntimeSymbols::kLongFromI64,
              typeConverter.getPyObjectPtrType(),
              mlir::ValueRange{storageValue})
        .getResult();
  if (mlir::isa<mlir::FloatType>(logicalType) &&
      mlir::isa<mlir::FloatType>(storageValue.getType()))
    return runtime
        .call(loc, RuntimeSymbols::kFloatFromDouble,
              typeConverter.getPyObjectPtrType(),
              mlir::ValueRange{storageValue})
        .getResult();
  if (mlir::isa<BoolType>(logicalType) &&
      mlir::isa<mlir::IntegerType>(storageValue.getType()))
    return runtime
        .call(loc, RuntimeSymbols::kBoolFromBool,
              typeConverter.getPyObjectPtrType(),
              mlir::ValueRange{storageValue})
        .getResult();
  if (class_field::Refcount::needed(logicalType, storageValue.getType(),
                                    typeConverter)) {
    if (!borrowExistingRef) {
      auto retain =
          runtime.call(loc, RuntimeSymbols::kIncRef, /*resultType=*/nullptr,
                       mlir::ValueRange{storageValue});
      threadsafe::Retain::premise(retain.getOperation(),
                                  ThreadSafetyAttrs::kPremiseAggregateBorrow);
    }
    return storageValue;
  }
  mlir::Type convertedLogicalType = typeConverter.convertType(logicalType);
  if (convertedLogicalType &&
      mlir::isa<mlir::MemRefType, mlir::UnrankedMemRefType>(
          convertedLogicalType)) {
    return rewriter
        .create<mlir::UnrealizedConversionCastOp>(loc, convertedLogicalType,
                                                  storageValue)
        .getResult(0);
  }
  if (storageValue.getType() != convertedLogicalType) {
    mlir::emitError(loc) << "unsupported static field boxing from "
                         << storageValue.getType() << " to " << logicalType;
    return mlir::failure();
  }
  return storageValue;
}

static mlir::FailureOr<llvm::SmallVector<mlir::Value>> boxStaticFieldValues(
    mlir::Location loc, mlir::Value storageValue, mlir::Type logicalType,
    mlir::ModuleOp module, mlir::ConversionPatternRewriter &rewriter,
    const PyLLVMTypeConverter &typeConverter, bool borrowExistingRef = false) {
  llvm::SmallVector<mlir::Type, 4> convertedTypes;
  if (mlir::failed(typeConverter.convertType(logicalType, convertedTypes)) ||
      convertedTypes.empty())
    return mlir::failure();
  if (convertedTypes.size() > 1) {
    llvm::SmallVector<mlir::Value> values =
        typeConverter.materializeTargetConversion(
            rewriter, loc, mlir::TypeRange(convertedTypes),
            mlir::ValueRange{storageValue}, logicalType);
    if (values.empty())
      return mlir::failure();
    return values;
  }

  mlir::FailureOr<mlir::Value> boxed =
      boxStaticFieldValue(loc, storageValue, logicalType, module, rewriter,
                          typeConverter, borrowExistingRef);
  if (mlir::failed(boxed))
    return mlir::failure();
  return llvm::SmallVector<mlir::Value>{*boxed};
}

namespace attr_get::Borrow {
bool onlyUser(mlir::Operation *user) {
  return mlir::isa<AddOp, SubOp, LeOp, LtOp, GtOp, GeOp, EqOp, NeOp, ReprOp,
                   ListAppendOp, ListRemoveOp, ListGetOp>(user);
}

template <typename AttrGetLikeOp> DecRefOp drop(AttrGetLikeOp op) {
  DecRefOp drop;
  mlir::Operation *borrowUser = nullptr;

  for (mlir::Operation *user : op.getResult().getUsers()) {
    if (auto decref = mlir::dyn_cast<DecRefOp>(user)) {
      if (drop)
        return nullptr;
      drop = decref;
      continue;
    }
    if (!attr_get::Borrow::onlyUser(user))
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
} // namespace attr_get::Borrow

static mlir::FailureOr<mlir::Value>
unboxStaticFieldValue(mlir::Location loc, mlir::Value boxedValue,
                      mlir::Type logicalType, mlir::Type storageType,
                      mlir::ModuleOp module,
                      mlir::ConversionPatternRewriter &rewriter,
                      const PyLLVMTypeConverter &typeConverter) {
  RuntimeAPI runtime(module, rewriter, typeConverter);
  if (mlir::isa<IntType>(logicalType) &&
      mlir::isa<mlir::IntegerType>(storageType))
    return runtime
        .call(loc, RuntimeSymbols::kLongAsI64, storageType,
              mlir::ValueRange{boxedValue})
        .getResult();
  if (mlir::isa<mlir::FloatType>(logicalType) &&
      mlir::isa<mlir::FloatType>(storageType))
    return runtime
        .call(loc, RuntimeSymbols::kFloatAsDouble, storageType,
              mlir::ValueRange{boxedValue})
        .getResult();
  if (mlir::isa<BoolType>(logicalType) &&
      mlir::isa<mlir::IntegerType>(storageType))
    return runtime
        .call(loc, RuntimeSymbols::kBoolAsBool, storageType,
              mlir::ValueRange{boxedValue})
        .getResult();
  if (mlir::isa<mlir::MemRefType, mlir::UnrankedMemRefType>(
          boxedValue.getType())) {
    return rewriter
        .create<mlir::UnrealizedConversionCastOp>(loc, storageType, boxedValue)
        .getResult(0);
  }
  if (boxedValue.getType() != storageType) {
    mlir::emitError(loc) << "unsupported static field unboxing from "
                         << boxedValue.getType() << " to " << storageType;
    return mlir::failure();
  }
  return boxedValue;
}

static mlir::FailureOr<mlir::Value>
unboxStaticFieldValue(mlir::Location loc, mlir::ValueRange boxedValues,
                      mlir::Type logicalType, mlir::Type storageType,
                      mlir::ModuleOp module,
                      mlir::ConversionPatternRewriter &rewriter,
                      const PyLLVMTypeConverter &typeConverter) {
  if (boxedValues.size() > 1) {
    return rewriter
        .create<mlir::UnrealizedConversionCastOp>(loc, storageType, boxedValues)
        .getResult(0);
  }
  if (boxedValues.empty())
    return mlir::failure();
  return unboxStaticFieldValue(loc, boxedValues.front(), logicalType,
                               storageType, module, rewriter, typeConverter);
}

// Static class instances lower to function-local stack slots.
struct ClassNewLowering : public mlir::OpConversionPattern<ClassNewOp> {
  ClassNewLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<ClassNewOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(ClassNewOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto classType = mlir::dyn_cast<ClassType>(op.getResult().getType());
    if (!classType)
      return mlir::failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    if (classType.getClassName() == "Exception") {
      mlir::Type resultType =
          typeConverter->convertType(op.getResult().getType());
      if (!resultType)
        return mlir::failure();
      rewriter.replaceOpWithNewOp<mlir::LLVM::ZeroOp>(op, resultType);
      return mlir::success();
    }
    mlir::FailureOr<StaticClassLayout> layoutOr =
        getStaticClassLayout(op, classType, *typeConverter);
    if (mlir::failed(layoutOr))
      return mlir::failure();

    mlir::Value slot = class_object::Slot::create(
        op.getLoc(), layoutOr->objectType, rewriter, op);
    if (!slot)
      return mlir::failure();
    mlir::Value zero =
        rewriter.create<mlir::LLVM::ZeroOp>(op.getLoc(), layoutOr->objectType);
    rewriter.create<mlir::LLVM::StoreOp>(op.getLoc(), zero, slot);
    rewriter.replaceOp(op, slot);
    return mlir::success();
  }
};

struct ClassPromoteLowering : public mlir::OpConversionPattern<ClassPromoteOp> {
  ClassPromoteLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<ClassPromoteOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(ClassPromoteOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto classType = mlir::dyn_cast<ClassType>(op.getResult().getType());
    if (!classType)
      return mlir::failure();
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return mlir::failure();

    auto ptrType = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
    std::string helperName = getClassHelperName(classType, "promote");
    auto helper = getOrInsertLLVMFunc(op.getLoc(), module, rewriter, helperName,
                                      ptrType, {ptrType});
    ownership::llvm_func::Contract::apply(helper, helperName);
    auto helperRef =
        mlir::SymbolRefAttr::get(module.getContext(), helper.getName());
    auto call = rewriter.create<mlir::LLVM::CallOp>(
        op.getLoc(), mlir::TypeRange{ptrType}, helperRef,
        mlir::ValueRange{adaptor.getInput()});
    rewriter.replaceOp(op, call.getResults());
    return mlir::success();
  }
};

struct PublishLowering : public mlir::OpConversionPattern<PublishOp> {
  PublishLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<PublishOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(PublishOp op, OneToNOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.setInsertionPoint(op);
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return mlir::failure();

    if (auto classType = mlir::dyn_cast<ClassType>(op.getResult().getType())) {
      if (adaptor.getInput().size() != 1)
        return mlir::failure();
      auto ptrType = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
      std::string helperName = getClassHelperName(classType, "promote");
      auto helper = getOrInsertLLVMFunc(op.getLoc(), module, rewriter,
                                        helperName, ptrType, {ptrType});
      ownership::llvm_func::Contract::apply(helper, helperName);
      auto helperRef =
          mlir::SymbolRefAttr::get(module.getContext(), helper.getName());
      auto call = rewriter.create<mlir::LLVM::CallOp>(
          op.getLoc(), mlir::TypeRange{ptrType}, helperRef,
          mlir::ValueRange{adaptor.getInput().front()});
      rewriter.replaceOp(op, call.getResults());
      return mlir::success();
    }

    mlir::FailureOr<llvm::SmallVector<mlir::Value>> promoted = mlir::failure();
    mlir::Type resultType = op.getResult().getType();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    promoted = container::Descriptor::promote(
        op.getLoc(), resultType, adaptor.getInput(), module, rewriter,
        *typeConverter, /*cloneReferenceSlots=*/true);
    if (mlir::succeeded(promoted)) {
      rewriter.replaceOpWithMultiple(
          op, llvm::ArrayRef<mlir::ValueRange>{*promoted});
      return mlir::success();
    }

    if (adaptor.getInput().size() != 1)
      return mlir::failure();
    RuntimeAPI runtime(module, rewriter, *typeConverter);
    auto retain = runtime.call(op.getLoc(), RuntimeSymbols::kIncRef,
                               /*resultType=*/nullptr,
                               mlir::ValueRange{adaptor.getInput().front()});
    threadsafe::Retain::premise(retain.getOperation(),
                                ThreadSafetyAttrs::kPremiseOwnedToken);

    rewriter.replaceOp(op, adaptor.getInput().front());
    return mlir::success();
  }
};

// AttrGetOp lowers to direct field load from the class slot.
struct AttrGetLowering : public mlir::OpConversionPattern<AttrGetOp> {
  AttrGetLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<AttrGetOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(AttrGetOp op, OneToNOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return mlir::failure();
    auto classType = mlir::dyn_cast<ClassType>(op.getObject().getType());
    if (!classType)
      return mlir::failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    mlir::FailureOr<StaticClassLayout> layoutOr =
        getStaticClassLayout(op, classType, *typeConverter);
    if (mlir::failed(layoutOr))
      return mlir::failure();
    mlir::FailureOr<std::pair<unsigned, StaticClassFieldInfo>> fieldOr =
        lookupStaticClassField(op, classType, *typeConverter,
                               op.getNameAttr().getValue());
    if (mlir::failed(fieldOr))
      return mlir::failure();

    auto helper =
        class_helper::GetField::get(op.getLoc(), module, classType, *layoutOr,
                                    fieldOr->first, rewriter, *typeConverter);
    auto helperRef =
        mlir::SymbolRefAttr::get(module.getContext(), helper.getName());
    mlir::Value borrowLocal = rewriter.create<mlir::LLVM::ConstantOp>(
        op.getLoc(), rewriter.getI1Type(), rewriter.getBoolAttr(false));
    auto call = rewriter.create<mlir::LLVM::CallOp>(
        op.getLoc(), mlir::TypeRange{fieldOr->second.storageType}, helperRef,
        mlir::ValueRange{adaptor.getObject().front(), borrowLocal});
    mlir::FailureOr<llvm::SmallVector<mlir::Value>> boxed =
        boxStaticFieldValues(op.getLoc(), call.getResult(),
                             fieldOr->second.logicalType, module, rewriter,
                             *typeConverter, /*borrowExistingRef=*/true);
    if (mlir::failed(boxed))
      return mlir::failure();
    llvm::SmallVector<mlir::ValueRange, 1> replacements{
        mlir::ValueRange(*boxed)};
    rewriter.replaceOpWithMultiple(op, replacements);
    return mlir::success();
  }
};

// AttrGetLocalOp is introduced only by the zero-cost rewrite pass after a
// local/non-shared proof. It can therefore lower to direct field access.
struct AttrGetLocalLowering : public mlir::OpConversionPattern<AttrGetLocalOp> {
  AttrGetLocalLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<AttrGetLocalOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(AttrGetLocalOp op, OneToNOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return mlir::failure();
    auto classType = mlir::dyn_cast<ClassType>(op.getObject().getType());
    if (!classType)
      return mlir::failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    mlir::FailureOr<StaticClassLayout> layoutOr =
        getStaticClassLayout(op, classType, *typeConverter);
    if (mlir::failed(layoutOr))
      return mlir::failure();
    mlir::FailureOr<std::pair<unsigned, StaticClassFieldInfo>> fieldOr =
        lookupStaticClassField(op, classType, *typeConverter,
                               op.getNameAttr().getValue());
    if (mlir::failed(fieldOr))
      return mlir::failure();

    DecRefOp borrowedDrop = nullptr;
    bool needsRefcount = class_field::Refcount::needed(
        fieldOr->second.logicalType, fieldOr->second.storageType,
        *typeConverter);
    if (needsRefcount)
      borrowedDrop = attr_get::Borrow::drop(op);

    auto ptrType = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
    mlir::Value fieldPtr = rewriter.create<mlir::LLVM::GEPOp>(
        op.getLoc(), ptrType, layoutOr->storageType,
        adaptor.getObject().front(),
        llvm::ArrayRef<mlir::LLVM::GEPArg>{
            0, static_cast<int32_t>(fieldOr->first)});
    mlir::Value loaded = rewriter.create<mlir::LLVM::LoadOp>(
        op.getLoc(), fieldOr->second.storageType, fieldPtr);
    if (needsRefcount)
      ownership::aggregate::Slot::markLoad(loaded);
    mlir::FailureOr<llvm::SmallVector<mlir::Value>> boxed =
        boxStaticFieldValues(op.getLoc(), loaded, fieldOr->second.logicalType,
                             module, rewriter, *typeConverter,
                             static_cast<bool>(borrowedDrop));
    if (mlir::failed(boxed))
      return mlir::failure();
    if (borrowedDrop)
      rewriter.eraseOp(borrowedDrop);
    llvm::SmallVector<mlir::ValueRange, 1> replacements{
        mlir::ValueRange(*boxed)};
    rewriter.replaceOpWithMultiple(op, replacements);
    return mlir::success();
  }
};

// AttrSetOp lowers through the synchronized helper path.
struct AttrSetLowering : public mlir::OpConversionPattern<AttrSetOp> {
  AttrSetLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<AttrSetOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(AttrSetOp op, OneToNOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return mlir::failure();
    auto classType = mlir::dyn_cast<ClassType>(op.getObject().getType());
    if (!classType)
      return mlir::failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    mlir::FailureOr<StaticClassLayout> layoutOr =
        getStaticClassLayout(op, classType, *typeConverter);
    if (mlir::failed(layoutOr))
      return mlir::failure();
    mlir::FailureOr<std::pair<unsigned, StaticClassFieldInfo>> fieldOr =
        lookupStaticClassField(op, classType, *typeConverter,
                               op.getNameAttr().getValue());
    if (mlir::failed(fieldOr))
      return mlir::failure();

    mlir::FailureOr<mlir::Value> unboxed = unboxStaticFieldValue(
        op.getLoc(), adaptor.getValue(), fieldOr->second.logicalType,
        fieldOr->second.storageType, module, rewriter, *typeConverter);
    if (mlir::failed(unboxed))
      return mlir::failure();
    bool consumeValue = static_cast<bool>(op->getAttr("ly.consume_value"));
    bool skipOldValueLoad =
        static_cast<bool>(op->getAttr("ly.zero_init_first_store"));

    auto helper =
        class_helper::SetField::get(op.getLoc(), module, classType, *layoutOr,
                                    fieldOr->first, rewriter, *typeConverter);
    auto helperRef =
        mlir::SymbolRefAttr::get(module.getContext(), helper.getName());
    mlir::Value retainNewValue = rewriter.create<mlir::LLVM::ConstantOp>(
        op.getLoc(), rewriter.getI1Type(), rewriter.getBoolAttr(!consumeValue));
    mlir::Value skipOldDrop = rewriter.create<mlir::LLVM::ConstantOp>(
        op.getLoc(), rewriter.getI1Type(),
        rewriter.getBoolAttr(skipOldValueLoad));
    rewriter.create<mlir::LLVM::CallOp>(
        op.getLoc(), mlir::TypeRange{}, helperRef,
        mlir::ValueRange{adaptor.getObject().front(), *unboxed, retainNewValue,
                         skipOldDrop});
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

// AttrSetLocalOp is introduced only by the zero-cost rewrite pass after a
// local/non-shared proof. It can therefore lower to direct field mutation.
struct AttrSetLocalLowering : public mlir::OpConversionPattern<AttrSetLocalOp> {
  AttrSetLocalLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<AttrSetLocalOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(AttrSetLocalOp op, OneToNOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return mlir::failure();
    auto classType = mlir::dyn_cast<ClassType>(op.getObject().getType());
    if (!classType)
      return mlir::failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    mlir::FailureOr<StaticClassLayout> layoutOr =
        getStaticClassLayout(op, classType, *typeConverter);
    if (mlir::failed(layoutOr))
      return mlir::failure();
    mlir::FailureOr<std::pair<unsigned, StaticClassFieldInfo>> fieldOr =
        lookupStaticClassField(op, classType, *typeConverter,
                               op.getNameAttr().getValue());
    if (mlir::failed(fieldOr))
      return mlir::failure();

    mlir::FailureOr<mlir::Value> unboxed = unboxStaticFieldValue(
        op.getLoc(), adaptor.getValue(), fieldOr->second.logicalType,
        fieldOr->second.storageType, module, rewriter, *typeConverter);
    if (mlir::failed(unboxed))
      return mlir::failure();
    bool consumeValue = static_cast<bool>(op->getAttr("ly.consume_value"));
    bool skipOldValueLoad =
        static_cast<bool>(op->getAttr("ly.zero_init_first_store"));

    auto ptrType = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
    mlir::Value fieldPtr = rewriter.create<mlir::LLVM::GEPOp>(
        op.getLoc(), ptrType, layoutOr->storageType,
        adaptor.getObject().front(),
        llvm::ArrayRef<mlir::LLVM::GEPArg>{
            0, static_cast<int32_t>(fieldOr->first)});
    mlir::Value oldValue;
    bool needsRefcount = class_field::Refcount::needed(
        fieldOr->second.logicalType, fieldOr->second.storageType,
        *typeConverter);
    if (needsRefcount) {
      RuntimeAPI runtime(module, rewriter, *typeConverter);
      if (!skipOldValueLoad) {
        oldValue = rewriter.create<mlir::LLVM::LoadOp>(
            op.getLoc(), fieldOr->second.storageType, fieldPtr);
      }
      if (!consumeValue) {
        auto retain =
            runtime.call(op.getLoc(), RuntimeSymbols::kIncRef,
                         /*resultType=*/nullptr, mlir::ValueRange{*unboxed});
        retain->setAttr(OwnershipContractAttrs::kAggregateRetain,
                        rewriter.getUnitAttr());
        threadsafe::Retain::premise(
            retain.getOperation(),
            isEntryBorrowedValue(*unboxed)
                ? ThreadSafetyAttrs::kPremiseEntryBorrowed
                : ThreadSafetyAttrs::kPremiseOwnedToken);
      }
    }
    auto store =
        rewriter.create<mlir::LLVM::StoreOp>(op.getLoc(), *unboxed, fieldPtr);
    if (needsRefcount)
      ownership::aggregate::Slot::markStore(store.getOperation());
    if (oldValue) {
      RuntimeAPI runtime(module, rewriter, *typeConverter);
      auto release =
          runtime.call(op.getLoc(), RuntimeSymbols::kDecRef,
                       /*resultType=*/nullptr, mlir::ValueRange{oldValue});
      ownership::aggregate::Slot::markLoad(oldValue);
      release->setAttr(OwnershipContractAttrs::kAggregateRelease,
                       rewriter.getUnitAttr());
    }
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

// ClassOp is just a container - erase it and keep its methods at module scope
struct ClassOpLowering : public mlir::OpConversionPattern<ClassOp> {
  ClassOpLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<ClassOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(ClassOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return mlir::failure();
    ClassType classType =
        ClassType::get(op.getContext(), op.getSymNameAttr().getValue());
    mlir::FailureOr<StaticClassLayout> layoutOr =
        getStaticClassLayout(op, classType, *typeConverter);
    if (mlir::failed(layoutOr))
      return mlir::failure();
    ensureStaticClassHelperBodies(op, *layoutOr, module, rewriter,
                                  *typeConverter);
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

} // namespace

namespace lowering::value::class_::Patterns {
void populate(PyLLVMTypeConverter &typeConverter,
              mlir::RewritePatternSet &patterns) {
  auto *ctx = patterns.getContext();
  patterns.add<ClassNewLowering, ClassPromoteLowering, PublishLowering,
               AttrGetLowering, AttrGetLocalLowering, AttrSetLowering,
               AttrSetLocalLowering, ClassOpLowering>(typeConverter, ctx);
}
} // namespace lowering::value::class_::Patterns

} // namespace py
