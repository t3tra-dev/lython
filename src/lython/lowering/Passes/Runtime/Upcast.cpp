#include "Passes/Runtime/Upcast.h"

#include "Common/Container.h"
#include "Common/LoweringUtils.h"
#include "Common/SlotUtils.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <string>

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

namespace py {
namespace {

static llvm::StringRef getTypedListReprHelperName(ListType listType) {
  mlir::Type elementType = listType.getElementType();
  if (mlir::isa<BoolType>(elementType))
    return "LyListBool_Repr";
  if (mlir::isa<mlir::FloatType>(elementType))
    return "LyListF64Bits_Repr";
  if (mlir::isa<ClassType>(elementType))
    return "LyListPtr_Repr";
  return "LyListI64_Repr";
}

static llvm::StringRef getTypedTupleReprHelperName(TupleType tupleType) {
  auto elements = tupleType.getElementTypes();
  if (!elements.empty() && llvm::all_of(elements, [](mlir::Type type) {
        return mlir::isa<BoolType>(type);
      }))
    return "LyTupleBool_Repr";
  if (!elements.empty() && llvm::all_of(elements, [](mlir::Type type) {
        return mlir::isa<mlir::FloatType>(type);
      }))
    return "LyTupleF64Bits_Repr";
  if (!elements.empty() && llvm::all_of(elements, [](mlir::Type type) {
        return mlir::isa<ClassType>(type);
      }))
    return "LyTuplePtr_Repr";
  return "LyTupleI64_Repr";
}

static void markFuncOwnedResult(mlir::func::FuncOp fn,
                                unsigned resultIndex = 0) {
  if (!fn)
    return;
  mlir::Builder builder(fn.getContext());
  fn->setAttr(OwnershipContractAttrs::kOwnedResults,
              builder.getArrayAttr({builder.getI64IntegerAttr(resultIndex)}));
}

static mlir::func::FuncOp
getOrInsertTypedListReprFunc(mlir::Location loc, mlir::ModuleOp module,
                             ListType listType, mlir::Type memrefType,
                             llvm::ArrayRef<mlir::Type> extraArgTypes,
                             mlir::Type resultType, mlir::OpBuilder &builder) {
  llvm::StringRef name = getTypedListReprHelperName(listType);
  if (auto fn = module.lookupSymbol<mlir::func::FuncOp>(name)) {
    markFuncOwnedResult(fn);
    return fn;
  }

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());
  llvm::SmallVector<mlir::Type> inputTypes;
  inputTypes.push_back(memrefType);
  inputTypes.append(extraArgTypes.begin(), extraArgTypes.end());
  auto fnType =
      mlir::FunctionType::get(module.getContext(), inputTypes, {resultType});
  auto fn = builder.create<mlir::func::FuncOp>(loc, name, fnType);
  fn->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
  fn.setVisibility(mlir::SymbolTable::Visibility::Private);
  markFuncOwnedResult(fn);
  return fn;
}

static mlir::func::FuncOp
getOrInsertTypedTupleReprFunc(mlir::Location loc, mlir::ModuleOp module,
                              TupleType tupleType, mlir::Type memrefType,
                              llvm::ArrayRef<mlir::Type> extraArgTypes,
                              mlir::Type resultType, mlir::OpBuilder &builder) {
  llvm::StringRef name = getTypedTupleReprHelperName(tupleType);
  if (auto fn = module.lookupSymbol<mlir::func::FuncOp>(name)) {
    markFuncOwnedResult(fn);
    return fn;
  }

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());
  llvm::SmallVector<mlir::Type> inputTypes;
  inputTypes.push_back(memrefType);
  inputTypes.append(extraArgTypes.begin(), extraArgTypes.end());
  auto fnType =
      mlir::FunctionType::get(module.getContext(), inputTypes, {resultType});
  auto fn = builder.create<mlir::func::FuncOp>(loc, name, fnType);
  fn->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
  fn.setVisibility(mlir::SymbolTable::Visibility::Private);
  markFuncOwnedResult(fn);
  return fn;
}

static mlir::func::FuncOp
getOrInsertTypedDictReprFunc(mlir::Location loc, mlir::ModuleOp module,
                             mlir::Type memrefType, mlir::Type resultType,
                             mlir::OpBuilder &builder) {
  llvm::StringRef name = "LyDictPacked_Repr";
  if (auto fn = module.lookupSymbol<mlir::func::FuncOp>(name)) {
    markFuncOwnedResult(fn);
    return fn;
  }

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());
  auto ptrType = mlir::LLVM::LLVMPointerType::get(module.getContext());
  mlir::Type i64Type = builder.getI64Type();
  auto fnType = mlir::FunctionType::get(
      module.getContext(), {memrefType, i64Type, i64Type, ptrType, ptrType},
      {resultType});
  auto fn = builder.create<mlir::func::FuncOp>(loc, name, fnType);
  fn->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
  fn.setVisibility(mlir::SymbolTable::Visibility::Private);
  markFuncOwnedResult(fn);
  return fn;
}

static std::string getStaticClassReprCallbackName(ClassType classType) {
  return ("__ly_class_repr_" + classType.getClassName()).str();
}

enum class PackedReprSlotKind : int64_t {
  I64 = 0,
  Bool = 1,
  F64Bits = 2,
  PyObject = 3,
  CallbackPtr = 4,
};

static int64_t getPackedReprSlotKind(mlir::Type type) {
  if (mlir::isa<BoolType>(type))
    return static_cast<int64_t>(PackedReprSlotKind::Bool);
  if (mlir::isa<mlir::FloatType>(type))
    return static_cast<int64_t>(PackedReprSlotKind::F64Bits);
  if (mlir::isa<ClassType>(type))
    return static_cast<int64_t>(PackedReprSlotKind::CallbackPtr);
  if (mlir::isa<NoneType, StrType, ObjectType, ExceptionType, TracebackType,
                LocationType>(type))
    return static_cast<int64_t>(PackedReprSlotKind::PyObject);
  return static_cast<int64_t>(PackedReprSlotKind::I64);
}

static mlir::Value getPackedReprCallback(mlir::Location loc, mlir::Type type,
                                         mlir::ModuleOp module,
                                         mlir::OpBuilder &builder) {
  auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  if (auto classType = mlir::dyn_cast<ClassType>(type)) {
    std::string reprName = getStaticClassReprCallbackName(classType);
    return builder.create<mlir::LLVM::AddressOfOp>(
        loc, ptrType, mlir::StringAttr::get(module.getContext(), reprName));
  }
  return builder.create<mlir::LLVM::ZeroOp>(loc, ptrType);
}

static mlir::Value widenReprSlot(mlir::Location loc, mlir::Value value,
                                 mlir::OpBuilder &builder) {
  mlir::Type type = value.getType();
  mlir::Type i64Type = builder.getI64Type();
  if (type == i64Type)
    return value;
  if (type == builder.getF64Type())
    return builder.create<mlir::arith::BitcastOp>(loc, i64Type, value);
  if (auto intType = mlir::dyn_cast<mlir::IntegerType>(type)) {
    if (intType.getWidth() < 64)
      return builder.create<mlir::arith::ExtUIOp>(loc, i64Type, value);
    if (intType.getWidth() > 64)
      return builder.create<mlir::arith::TruncIOp>(loc, i64Type, value);
  }
  return {};
}

struct ReprSnapshot {
  mlir::Value flat;
  mlir::Value size;
};

static mlir::FailureOr<ReprSnapshot> materializeListReprFlat(
    mlir::Location loc, mlir::Value header, mlir::Value items,
    mlir::Type elementType, mlir::ModuleOp module, mlir::OpBuilder &builder,
    const PyLLVMTypeConverter &typeConverter, llvm::StringRef retainPremise,
    bool markManagedAccesses) {
  auto itemsType = mlir::dyn_cast<mlir::MemRefType>(items.getType());
  if (!itemsType)
    return mlir::failure();

  mlir::Value sizeI64 = builder.create<mlir::memref::LoadOp>(
      loc, header, createIndexConstant(loc, builder, 0));
  if (markManagedAccesses)
    container::access::Contract::mark(sizeI64.getDefiningOp(), header, header);
  mlir::Value size = builder.create<mlir::arith::IndexCastOp>(
      loc, builder.getIndexType(), sizeI64);
  mlir::Value headerSlots = createIndexConstant(loc, builder, kListHeaderSize);
  mlir::Value flatSize =
      builder.create<mlir::arith::AddIOp>(loc, size, headerSlots);
  auto flatType =
      mlir::MemRefType::get({mlir::ShapedType::kDynamic}, builder.getI64Type());
  mlir::Value flat = builder.create<mlir::memref::AllocaOp>(
      loc, flatType, mlir::ValueRange{flatSize});
  for (int64_t slot = 0; slot < kListHeaderSize; ++slot) {
    mlir::Value index = createIndexConstant(loc, builder, slot);
    mlir::Value value =
        builder.create<mlir::memref::LoadOp>(loc, header, index);
    if (markManagedAccesses)
      container::access::Contract::mark(value.getDefiningOp(), header, header);
    builder.create<mlir::memref::StoreOp>(loc, value, flat, index);
  }

  mlir::Value zero = createIndexConstant(loc, builder, 0);
  mlir::Value one = createIndexConstant(loc, builder, 1);
  auto loop = builder.create<mlir::scf::ForOp>(loc, zero, size, one);
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(loop.getBody());
    mlir::Value iv = loop.getInductionVar();
    mlir::Value value = builder.create<mlir::memref::LoadOp>(loc, items, iv);
    if (markManagedAccesses)
      container::access::Contract::mark(value.getDefiningOp(), header, items);
    value = widenReprSlot(loc, value, builder);
    if (!value)
      return mlir::failure();
    mlir::Value destIndex =
        builder.create<mlir::arith::AddIOp>(loc, iv, headerSlots);
    Slot::refcount(loc, value, elementType, module, builder, typeConverter,
                   "incref", /*aggregateEffect=*/true, retainPremise);
    auto store =
        builder.create<mlir::memref::StoreOp>(loc, value, flat, destIndex);
    if (Slot::refcounted(elementType))
      Slot::markTransfer(store.getOperation());
  }
  return ReprSnapshot{flat, size};
}

static void releaseListReprSnapshot(mlir::Location loc, mlir::Value flat,
                                    mlir::Value size, mlir::Type elementType,
                                    mlir::ModuleOp module,
                                    mlir::OpBuilder &builder,
                                    const PyLLVMTypeConverter &typeConverter) {
  if (!Slot::refcounted(elementType))
    return;

  mlir::Value headerSlots = createIndexConstant(loc, builder, kListHeaderSize);
  mlir::Value zero = createIndexConstant(loc, builder, 0);
  mlir::Value one = createIndexConstant(loc, builder, 1);
  auto loop = builder.create<mlir::scf::ForOp>(loc, zero, size, one);
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(loop.getBody());
    mlir::Value iv = loop.getInductionVar();
    mlir::Value sourceIndex =
        builder.create<mlir::arith::AddIOp>(loc, iv, headerSlots);
    mlir::Value slot =
        builder.create<mlir::memref::LoadOp>(loc, flat, sourceIndex);
    Slot::refcount(loc, slot, elementType, module, builder, typeConverter,
                   "decref", /*aggregateEffect=*/true);
  }
}

static mlir::FailureOr<ReprSnapshot> materializeDictReprFlat(
    mlir::Location loc, mlir::Value header, mlir::Value keys,
    mlir::Value values, mlir::Value states, DictType dictType,
    mlir::ModuleOp module, mlir::OpBuilder &builder,
    const PyLLVMTypeConverter &typeConverter, llvm::StringRef retainPremise,
    bool markManagedAccesses) {
  if (!mlir::dyn_cast<mlir::MemRefType>(keys.getType()) ||
      !mlir::dyn_cast<mlir::MemRefType>(values.getType()) ||
      !mlir::dyn_cast<mlir::MemRefType>(states.getType()))
    return mlir::failure();

  mlir::Value capacityI64 = builder.create<mlir::memref::LoadOp>(
      loc, header, createIndexConstant(loc, builder, kTypedDictCapacitySlot));
  if (markManagedAccesses)
    container::access::Contract::mark(capacityI64.getDefiningOp(), header,
                                      header);
  mlir::Value capacity = builder.create<mlir::arith::IndexCastOp>(
      loc, builder.getIndexType(), capacityI64);
  mlir::Value headerSlots = createIndexConstant(loc, builder, kDictHeaderSize);
  mlir::Value three = createIndexConstant(loc, builder, 3);
  mlir::Value payloadSlots =
      builder.create<mlir::arith::MulIOp>(loc, capacity, three);
  mlir::Value flatSize =
      builder.create<mlir::arith::AddIOp>(loc, headerSlots, payloadSlots);
  auto flatType =
      mlir::MemRefType::get({mlir::ShapedType::kDynamic}, builder.getI64Type());
  mlir::Value flat = builder.create<mlir::memref::AllocaOp>(
      loc, flatType, mlir::ValueRange{flatSize});

  for (int64_t slot = 0; slot < kDictHeaderSize; ++slot) {
    mlir::Value index = createIndexConstant(loc, builder, slot);
    mlir::Value value =
        builder.create<mlir::memref::LoadOp>(loc, header, index);
    if (markManagedAccesses)
      container::access::Contract::mark(value.getDefiningOp(), header, header);
    builder.create<mlir::memref::StoreOp>(loc, value, flat, index);
  }

  mlir::Value keysBase = headerSlots;
  mlir::Value valuesBase =
      builder.create<mlir::arith::AddIOp>(loc, keysBase, capacity);
  mlir::Value statesBase =
      builder.create<mlir::arith::AddIOp>(loc, valuesBase, capacity);
  mlir::Value zero = createIndexConstant(loc, builder, 0);
  mlir::Value one = createIndexConstant(loc, builder, 1);
  auto loop = builder.create<mlir::scf::ForOp>(loc, zero, capacity, one);
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(loop.getBody());
    mlir::Value iv = loop.getInductionVar();
    auto stateLoad = builder.create<mlir::memref::LoadOp>(loc, states, iv);
    if (markManagedAccesses)
      container::access::Contract::mark(stateLoad.getOperation(), header,
                                        states);
    mlir::Value state = stateLoad;
    mlir::Value stateI64 = state;
    if (auto intType = mlir::dyn_cast<mlir::IntegerType>(state.getType())) {
      if (intType.getWidth() < 64)
        stateI64 = builder.create<mlir::arith::ExtUIOp>(
            loc, builder.getI64Type(), state);
      else if (intType.getWidth() > 64)
        stateI64 = builder.create<mlir::arith::TruncIOp>(
            loc, builder.getI64Type(), state);
    }
    mlir::Value stateIndex =
        builder.create<mlir::arith::AddIOp>(loc, statesBase, iv);
    builder.create<mlir::memref::StoreOp>(loc, stateI64, flat, stateIndex);

    mlir::Value occupied = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::eq, state,
        builder.create<mlir::arith::ConstantIntOp>(loc, 1, state.getType()));
    auto copyIfOccupied = builder.create<mlir::scf::IfOp>(
        loc, occupied, /*withElseRegion=*/false);
    {
      mlir::OpBuilder::InsertionGuard ifGuard(builder);
      builder.setInsertionPointToStart(copyIfOccupied.thenBlock());

      auto keyLoad = builder.create<mlir::memref::LoadOp>(loc, keys, iv);
      if (markManagedAccesses)
        container::access::Contract::mark(keyLoad.getOperation(), header, keys);
      mlir::Value key = widenReprSlot(loc, keyLoad, builder);
      if (!key)
        return mlir::failure();
      mlir::Value keyIndex =
          builder.create<mlir::arith::AddIOp>(loc, keysBase, iv);
      Slot::refcount(loc, key, dictType.getKeyType(), module, builder,
                     typeConverter, "incref",
                     /*aggregateEffect=*/true, retainPremise);
      auto keyStore =
          builder.create<mlir::memref::StoreOp>(loc, key, flat, keyIndex);
      if (Slot::refcounted(dictType.getKeyType()))
        Slot::markTransfer(keyStore.getOperation());

      auto valueLoad = builder.create<mlir::memref::LoadOp>(loc, values, iv);
      if (markManagedAccesses)
        container::access::Contract::mark(valueLoad.getOperation(), header,
                                          values);
      mlir::Value value = widenReprSlot(loc, valueLoad, builder);
      if (!value)
        return mlir::failure();
      mlir::Value valueIndex =
          builder.create<mlir::arith::AddIOp>(loc, valuesBase, iv);
      Slot::refcount(loc, value, dictType.getValueType(), module, builder,
                     typeConverter, "incref",
                     /*aggregateEffect=*/true, retainPremise);
      auto valueStore =
          builder.create<mlir::memref::StoreOp>(loc, value, flat, valueIndex);
      if (Slot::refcounted(dictType.getValueType()))
        Slot::markTransfer(valueStore.getOperation());
    }
  }

  return ReprSnapshot{flat, capacity};
}

static void releaseDictReprSnapshot(mlir::Location loc, mlir::Value flat,
                                    mlir::Value capacity, DictType dictType,
                                    mlir::ModuleOp module,
                                    mlir::OpBuilder &builder,
                                    const PyLLVMTypeConverter &typeConverter) {
  bool keyRefcounted = Slot::refcounted(dictType.getKeyType());
  bool valueRefcounted = Slot::refcounted(dictType.getValueType());
  if (!keyRefcounted && !valueRefcounted)
    return;

  mlir::Value keysBase = createIndexConstant(loc, builder, kDictHeaderSize);
  mlir::Value valuesBase =
      builder.create<mlir::arith::AddIOp>(loc, keysBase, capacity);
  mlir::Value statesBase =
      builder.create<mlir::arith::AddIOp>(loc, valuesBase, capacity);
  mlir::Value zero = createIndexConstant(loc, builder, 0);
  mlir::Value one = createIndexConstant(loc, builder, 1);
  auto loop = builder.create<mlir::scf::ForOp>(loc, zero, capacity, one);
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(loop.getBody());
    mlir::Value iv = loop.getInductionVar();
    mlir::Value stateIndex =
        builder.create<mlir::arith::AddIOp>(loc, statesBase, iv);
    mlir::Value state =
        builder.create<mlir::memref::LoadOp>(loc, flat, stateIndex);
    mlir::Value occupied = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::eq, state,
        createI64Constant(loc, builder, 1));
    auto releaseIfOccupied = builder.create<mlir::scf::IfOp>(
        loc, occupied, /*withElseRegion=*/false);
    {
      mlir::OpBuilder::InsertionGuard ifGuard(builder);
      builder.setInsertionPointToStart(releaseIfOccupied.thenBlock());
      if (keyRefcounted) {
        mlir::Value keyIndex =
            builder.create<mlir::arith::AddIOp>(loc, keysBase, iv);
        mlir::Value key =
            builder.create<mlir::memref::LoadOp>(loc, flat, keyIndex);
        Slot::refcount(loc, key, dictType.getKeyType(), module, builder,
                       typeConverter, "decref",
                       /*aggregateEffect=*/true);
      }
      if (valueRefcounted) {
        mlir::Value valueIndex =
            builder.create<mlir::arith::AddIOp>(loc, valuesBase, iv);
        mlir::Value value =
            builder.create<mlir::memref::LoadOp>(loc, flat, valueIndex);
        Slot::refcount(loc, value, dictType.getValueType(), module, builder,
                       typeConverter, "decref",
                       /*aggregateEffect=*/true);
      }
    }
  }
}

/// py.upcast forwards pointer-backed py.* operands. Compiler-owned typed
/// containers are not pointer-compatible; upcasting them materializes an owned
/// object bridge from a borrowed container snapshot.
struct UpcastLowering : public mlir::OpConversionPattern<UpcastOp> {
  UpcastLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<UpcastOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(UpcastOp op, OneToNOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (adaptor.getInput().empty())
      return mlir::failure();
    if (auto listType = mlir::dyn_cast<ListType>(op.getInput().getType())) {
      if (isCompilerOwnedMemRefListType(listType)) {
        mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
        if (!module)
          return mlir::failure();
        auto *converter =
            static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
        auto unrankedType = mlir::UnrankedMemRefType::get(rewriter.getI64Type(),
                                                          /*memorySpace=*/0);
        mlir::Value reprMemref;
        mlir::Value reprSnapshotSize;
        bool releaseSnapshot = false;
        if (adaptor.getInput().size() == 2) {
          mlir::Value header = adaptor.getInput().front();
          mlir::Value items = adaptor.getInput().back();
          auto flatType = mlir::MemRefType::get({mlir::ShapedType::kDynamic},
                                                rewriter.getI64Type());

          mlir::Value isManaged = container::Managed::predicate(
              op.getLoc(), header, kTypedListRefcountSlot, rewriter);
          auto lockedCopy = rewriter.create<mlir::scf::IfOp>(
              op.getLoc(), mlir::TypeRange{flatType, rewriter.getIndexType()},
              isManaged,
              /*withElseRegion=*/true);
          {
            mlir::OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(lockedCopy.thenBlock());
            container::Managed::lock(op.getLoc(), header, kTypedListLockSlot,
                                     rewriter);
            mlir::FailureOr<ReprSnapshot> snapshot = materializeListReprFlat(
                op.getLoc(), header, items, listType.getElementType(), module,
                rewriter, *converter, ThreadSafetyAttrs::kPremiseLockedBorrow,
                /*markManagedAccesses=*/true);
            if (mlir::failed(snapshot))
              return mlir::failure();
            container::Managed::unlock(op.getLoc(), header, kTypedListLockSlot,
                                       rewriter);
            rewriter.create<mlir::scf::YieldOp>(
                op.getLoc(), mlir::ValueRange{snapshot->flat, snapshot->size});
          }
          {
            mlir::OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(lockedCopy.elseBlock());
            mlir::FailureOr<ReprSnapshot> snapshot = materializeListReprFlat(
                op.getLoc(), header, items, listType.getElementType(), module,
                rewriter, *converter,
                ThreadSafetyAttrs::kPremiseAggregateBorrow,
                /*markManagedAccesses=*/false);
            if (mlir::failed(snapshot))
              return mlir::failure();
            rewriter.create<mlir::scf::YieldOp>(
                op.getLoc(), mlir::ValueRange{snapshot->flat, snapshot->size});
          }
          reprMemref = lockedCopy.getResult(0);
          reprSnapshotSize = lockedCopy.getResult(1);
          releaseSnapshot = Slot::refcounted(listType.getElementType());
        } else if (adaptor.getInput().size() == 1) {
          reprMemref = adaptor.getInput().front();
        } else {
          return mlir::failure();
        }
        mlir::Value unranked = rewriter.create<mlir::memref::CastOp>(
            op.getLoc(), unrankedType, reprMemref);
        llvm::SmallVector<mlir::Type> extraTypes;
        llvm::SmallVector<mlir::Value> extraOperands;
        if (auto classType =
                mlir::dyn_cast<ClassType>(listType.getElementType())) {
          auto ptrType =
              mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
          extraTypes.push_back(ptrType);
          std::string reprName = getStaticClassReprCallbackName(classType);
          extraOperands.push_back(rewriter.create<mlir::LLVM::AddressOfOp>(
              op.getLoc(), ptrType,
              mlir::StringAttr::get(module.getContext(), reprName)));
        }
        auto reprFunc = getOrInsertTypedListReprFunc(
            op.getLoc(), module, listType, unrankedType, extraTypes,
            converter->getPyObjectPtrType(), rewriter);
        llvm::SmallVector<mlir::Value> operands;
        operands.push_back(unranked);
        operands.append(extraOperands.begin(), extraOperands.end());
        auto call = rewriter.create<mlir::func::CallOp>(op.getLoc(), reprFunc,
                                                        operands);
        if (releaseSnapshot)
          releaseListReprSnapshot(op.getLoc(), reprMemref, reprSnapshotSize,
                                  listType.getElementType(), module, rewriter,
                                  *converter);
        rewriter.replaceOp(op, call.getResults());
        return mlir::success();
      }
    }
    if (auto tupleType = mlir::dyn_cast<TupleType>(op.getInput().getType())) {
      if (isCompilerOwnedMemRefTupleType(tupleType)) {
        mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
        if (!module)
          return mlir::failure();
        auto *converter =
            static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
        auto unrankedType = mlir::UnrankedMemRefType::get(rewriter.getI64Type(),
                                                          /*memorySpace=*/0);
        if (adaptor.getInput().size() != 2)
          return mlir::failure();

        mlir::Value header = adaptor.getInput().front();
        mlir::Value items = adaptor.getInput().back();
        auto itemsType = mlir::dyn_cast<mlir::MemRefType>(items.getType());
        if (!itemsType)
          return rewriter.notifyMatchFailure(
              op, "tuple repr requires memref item storage");
        auto elements = tupleType.getElementTypes();

        mlir::Value sizeI64 = rewriter.create<mlir::memref::LoadOp>(
            op.getLoc(), header, createIndexConstant(op.getLoc(), rewriter, 0));
        mlir::Value size = rewriter.create<mlir::arith::IndexCastOp>(
            op.getLoc(), rewriter.getIndexType(), sizeI64);
        mlir::Value headerSlots =
            createIndexConstant(op.getLoc(), rewriter, kTupleHeaderSize);
        mlir::Value flatSize = rewriter.create<mlir::arith::AddIOp>(
            op.getLoc(), size, headerSlots);
        auto flatType = mlir::MemRefType::get({mlir::ShapedType::kDynamic},
                                              rewriter.getI64Type());
        mlir::Value flat = rewriter.create<mlir::memref::AllocaOp>(
            op.getLoc(), flatType, mlir::ValueRange{flatSize});
        for (int64_t slot = 0; slot < kTupleHeaderSize; ++slot) {
          mlir::Value index = createIndexConstant(op.getLoc(), rewriter, slot);
          mlir::Value value =
              rewriter.create<mlir::memref::LoadOp>(op.getLoc(), header, index);
          rewriter.create<mlir::memref::StoreOp>(op.getLoc(), value, flat,
                                                 index);
        }

        for (auto [index, elementType] : llvm::enumerate(elements)) {
          mlir::Value sourceIndex = createIndexConstant(
              op.getLoc(), rewriter, static_cast<int64_t>(index));
          mlir::Value value = rewriter.create<mlir::memref::LoadOp>(
              op.getLoc(), items, sourceIndex);
          value = widenReprSlot(op.getLoc(), value, rewriter);
          if (!value)
            return mlir::failure();
          mlir::Value destIndex = createIndexConstant(
              op.getLoc(), rewriter,
              kTupleHeaderSize + static_cast<int64_t>(index));
          Slot::refcount(op.getLoc(), value, elementType, module, rewriter,
                         *converter, "incref",
                         /*aggregateEffect=*/true,
                         ThreadSafetyAttrs::kPremiseAggregateBorrow);
          auto store = rewriter.create<mlir::memref::StoreOp>(
              op.getLoc(), value, flat, destIndex);
          if (Slot::refcounted(elementType))
            Slot::markTransfer(store.getOperation());
        }

        mlir::Value unranked = rewriter.create<mlir::memref::CastOp>(
            op.getLoc(), unrankedType, flat);
        llvm::SmallVector<mlir::Type> extraTypes;
        llvm::SmallVector<mlir::Value> extraOperands;
        if (!elements.empty() && llvm::all_of(elements, [](mlir::Type type) {
              return mlir::isa<ClassType>(type);
            })) {
          auto classType = mlir::cast<ClassType>(elements.front());
          auto ptrType =
              mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
          extraTypes.push_back(ptrType);
          std::string reprName = getStaticClassReprCallbackName(classType);
          extraOperands.push_back(rewriter.create<mlir::LLVM::AddressOfOp>(
              op.getLoc(), ptrType,
              mlir::StringAttr::get(module.getContext(), reprName)));
        }
        auto reprFunc = getOrInsertTypedTupleReprFunc(
            op.getLoc(), module, tupleType, unrankedType, extraTypes,
            converter->getPyObjectPtrType(), rewriter);
        llvm::SmallVector<mlir::Value> operands;
        operands.push_back(unranked);
        operands.append(extraOperands.begin(), extraOperands.end());
        auto call = rewriter.create<mlir::func::CallOp>(op.getLoc(), reprFunc,
                                                        operands);
        for (auto [index, elementType] : llvm::enumerate(elements)) {
          if (!Slot::refcounted(elementType))
            continue;
          mlir::Value sourceIndex = createIndexConstant(
              op.getLoc(), rewriter,
              kTupleHeaderSize + static_cast<int64_t>(index));
          mlir::Value slot = rewriter.create<mlir::memref::LoadOp>(
              op.getLoc(), flat, sourceIndex);
          Slot::refcount(op.getLoc(), slot, elementType, module, rewriter,
                         *converter, "decref",
                         /*aggregateEffect=*/true);
        }
        rewriter.replaceOp(op, call.getResults());
        return mlir::success();
      }
    }
    if (auto dictType = mlir::dyn_cast<DictType>(op.getInput().getType())) {
      if (isCompilerOwnedMemRefDictType(dictType)) {
        mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
        if (!module)
          return mlir::failure();
        auto *converter =
            static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
        auto unrankedType = mlir::UnrankedMemRefType::get(rewriter.getI64Type(),
                                                          /*memorySpace=*/0);
        auto flatType = mlir::MemRefType::get({mlir::ShapedType::kDynamic},
                                              rewriter.getI64Type());
        if (adaptor.getInput().size() != 4)
          return mlir::failure();

        mlir::Value header = adaptor.getInput()[0];
        mlir::Value keys = adaptor.getInput()[1];
        mlir::Value values = adaptor.getInput()[2];
        mlir::Value states = adaptor.getInput()[3];
        mlir::Value isManaged = container::Managed::predicate(
            op.getLoc(), header, kTypedDictRefcountSlot, rewriter);
        auto lockedCopy = rewriter.create<mlir::scf::IfOp>(
            op.getLoc(), mlir::TypeRange{flatType, rewriter.getIndexType()},
            isManaged,
            /*withElseRegion=*/true);
        {
          mlir::OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointToStart(lockedCopy.thenBlock());
          container::Managed::lock(op.getLoc(), header, kTypedDictLockSlot,
                                   rewriter);
          mlir::FailureOr<ReprSnapshot> snapshot = materializeDictReprFlat(
              op.getLoc(), header, keys, values, states, dictType, module,
              rewriter, *converter, ThreadSafetyAttrs::kPremiseLockedBorrow,
              /*markManagedAccesses=*/true);
          if (mlir::failed(snapshot))
            return mlir::failure();
          container::Managed::unlock(op.getLoc(), header, kTypedDictLockSlot,
                                     rewriter);
          rewriter.create<mlir::scf::YieldOp>(
              op.getLoc(), mlir::ValueRange{snapshot->flat, snapshot->size});
        }
        {
          mlir::OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointToStart(lockedCopy.elseBlock());
          mlir::FailureOr<ReprSnapshot> snapshot = materializeDictReprFlat(
              op.getLoc(), header, keys, values, states, dictType, module,
              rewriter, *converter, ThreadSafetyAttrs::kPremiseAggregateBorrow,
              /*markManagedAccesses=*/false);
          if (mlir::failed(snapshot))
            return mlir::failure();
          rewriter.create<mlir::scf::YieldOp>(
              op.getLoc(), mlir::ValueRange{snapshot->flat, snapshot->size});
        }

        mlir::Value unranked = rewriter.create<mlir::memref::CastOp>(
            op.getLoc(), unrankedType, lockedCopy.getResult(0));
        auto reprFunc = getOrInsertTypedDictReprFunc(
            op.getLoc(), module, unrankedType, converter->getPyObjectPtrType(),
            rewriter);
        mlir::Value keyKind =
            createI64Constant(op.getLoc(), rewriter,
                              getPackedReprSlotKind(dictType.getKeyType()));
        mlir::Value valueKind =
            createI64Constant(op.getLoc(), rewriter,
                              getPackedReprSlotKind(dictType.getValueType()));
        mlir::Value keyCallback = getPackedReprCallback(
            op.getLoc(), dictType.getKeyType(), module, rewriter);
        mlir::Value valueCallback = getPackedReprCallback(
            op.getLoc(), dictType.getValueType(), module, rewriter);
        auto call = rewriter.create<mlir::func::CallOp>(
            op.getLoc(), reprFunc,
            mlir::ValueRange{unranked, keyKind, valueKind, keyCallback,
                             valueCallback});
        releaseDictReprSnapshot(op.getLoc(), lockedCopy.getResult(0),
                                lockedCopy.getResult(1), dictType, module,
                                rewriter, *converter);
        rewriter.replaceOp(op, call.getResults());
        return mlir::success();
      }
    }
    rewriter.replaceOp(op, adaptor.getInput().front());
    return mlir::success();
  }
};

} // namespace

namespace lowering::runtime::upcast::Patterns {

void populate(PyLLVMTypeConverter &typeConverter,
              mlir::RewritePatternSet &patterns) {
  patterns.add<UpcastLowering>(typeConverter, patterns.getContext());
}

} // namespace lowering::runtime::upcast::Patterns

} // namespace py
