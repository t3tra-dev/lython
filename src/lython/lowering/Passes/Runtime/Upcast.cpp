#include "Passes/Runtime/Upcast.h"

#include "Common/ClassLayout.h"
#include "Common/Container.h"
#include "Common/LoweringUtils.h"
#include "Common/Object.h"
#include "Common/SlotUtils.h"
#include "PyValue/ClassHelpers.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <string>

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

namespace py {
namespace {

static std::string getStaticClassReprCallbackName(ClassType classType) {
  return ("__ly_class_repr_" + classType.getClassName()).str();
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

namespace unicode {

using Value = llvm::SmallVector<mlir::Value, 2>;

llvm::SmallVector<mlir::Type, 2> types(mlir::MLIRContext *ctx) {
  llvm::SmallVector<mlir::Type, 2> result;
  object_abi::str_abi::Parts::storageTypes(ctx, result);
  return result;
}

Value take(mlir::ValueRange values, size_t count) {
  Value result;
  result.reserve(count);
  for (size_t index = 0; index < count && index < values.size(); ++index)
    result.push_back(values[index]);
  return result;
}

Value literal(mlir::Location loc, llvm::StringRef text, mlir::ModuleOp module,
              mlir::OpBuilder &builder,
              const PyLLVMTypeConverter &typeConverter, bool owned) {
  (void)owned;
  RuntimeAPI runtime(module, builder, typeConverter);
  mlir::Value bytes = runtime.getByteLiteral(loc, builder.getStringAttr(text));
  mlir::Value start = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
  mlir::Value length =
      runtime.getI64Constant(loc, static_cast<int64_t>(text.size()));
  auto resultTypes = types(builder.getContext());
  auto call = runtime.call(loc, RuntimeSymbols::kUnicodeFromBytes,
                           mlir::TypeRange(resultTypes),
                           mlir::ValueRange{bytes, start, length});
  return Value(call.getResults());
}

Value concat(mlir::Location loc, mlir::ValueRange lhs, mlir::ValueRange rhs,
             mlir::ModuleOp module, mlir::OpBuilder &builder,
             const PyLLVMTypeConverter &typeConverter) {
  RuntimeAPI runtime(module, builder, typeConverter);
  auto resultTypes = types(builder.getContext());
  llvm::SmallVector<mlir::Value, 6> operands;
  operands.append(lhs.begin(), lhs.end());
  operands.append(rhs.begin(), rhs.end());
  auto call = runtime.call(loc, RuntimeSymbols::kUnicodeConcat,
                           mlir::TypeRange(resultTypes), operands);
  return Value(call.getResults());
}

Value concat3(mlir::Location loc, mlir::ValueRange lhs, mlir::ValueRange middle,
              mlir::ValueRange rhs, mlir::ModuleOp module,
              mlir::OpBuilder &builder,
              const PyLLVMTypeConverter &typeConverter) {
  RuntimeAPI runtime(module, builder, typeConverter);
  auto resultTypes = types(builder.getContext());
  llvm::SmallVector<mlir::Value, 9> operands;
  operands.append(lhs.begin(), lhs.end());
  operands.append(middle.begin(), middle.end());
  operands.append(rhs.begin(), rhs.end());
  auto call = runtime.call(loc, RuntimeSymbols::kUnicodeConcat3,
                           mlir::TypeRange(resultTypes), operands);
  return Value(call.getResults());
}

void release(mlir::Location loc, mlir::ValueRange value, mlir::ModuleOp module,
             mlir::OpBuilder &builder, const PyLLVMTypeConverter &typeConverter,
             bool aggregateEffect = false) {
  (void)aggregateEffect;
  RuntimeAPI runtime(module, builder, typeConverter);
  runtime.call(loc, RuntimeSymbols::kUnicodeDecRef, mlir::Type(), value);
}

void yield(mlir::Location loc, mlir::ValueRange value,
           mlir::OpBuilder &builder) {
  builder.create<mlir::scf::YieldOp>(loc, value);
}

} // namespace unicode

static mlir::Value classPartMemRef(mlir::Location loc, mlir::Value descriptor,
                                   mlir::MemRefType targetType,
                                   mlir::OpBuilder &builder) {
  if (!descriptor || !targetType)
    return {};
  if (descriptor.getType() == targetType)
    return descriptor;
  if (mlir::Operation *def = descriptor.getDefiningOp())
    if (class_layout::DescriptorShape::has(def) &&
        mlir::failed(class_layout::DescriptorShape::verify(
            def, targetType, "class carrier descriptor")))
      return {};
  if (mlir::isa<mlir::MemRefType>(descriptor.getType()))
    return builder.create<mlir::memref::CastOp>(loc, targetType, descriptor);
  auto cast = builder.create<mlir::UnrealizedConversionCastOp>(
      loc, targetType, mlir::ValueRange{descriptor});
  class_layout::DescriptorShape::mark(cast.getOperation(), targetType);
  return cast.getResult(0);
}

static mlir::Value
loadedClassObjectDescriptor(mlir::Location loc, mlir::Value objectDescriptor,
                            mlir::LLVM::LLVMStructType objectType,
                            mlir::OpBuilder &builder) {
  if (objectDescriptor.getType() == objectType)
    return objectDescriptor;
  auto memrefType =
      mlir::dyn_cast<mlir::MemRefType>(objectDescriptor.getType());
  if (!memrefType || memrefType.getRank() != 1 ||
      memrefType.getElementType() != objectType)
    return {};
  auto load = builder.create<mlir::memref::LoadOp>(
      loc, objectDescriptor, createIndexConstant(loc, builder, 0));
  load->setAttr(ClassSafetyAttrs::kCarrierLoad, builder.getUnitAttr());
  ownership::aggregate::Slot::markLoad(load.getResult());
  return load.getResult();
}

static mlir::FailureOr<unicode::Value>
unicodePartsFromCarrier(mlir::Location loc, mlir::Value value,
                        mlir::OpBuilder &builder) {
  auto objectType = class_layout::objectCarrierType(value.getType());
  if (!objectType)
    return mlir::failure();
  mlir::Value object =
      loadedClassObjectDescriptor(loc, value, objectType, builder);
  if (!object)
    return mlir::failure();
  llvm::SmallVector<mlir::Type, 2> storageTypes;
  object_abi::str_abi::Parts::storageTypes(builder.getContext(), storageTypes);
  if (storageTypes.size() !=
      static_cast<size_t>(class_layout::Object::partCount(objectType)))
    return mlir::failure();
  unicode::Value parts;
  for (auto [index, storageType] : llvm::enumerate(storageTypes)) {
    auto memrefType = mlir::dyn_cast<mlir::MemRefType>(storageType);
    if (!memrefType)
      return mlir::failure();
    mlir::Value descriptor = class_layout::Object::descriptor(
        loc, objectType, object, static_cast<int64_t>(index), builder);
    mlir::Value part = classPartMemRef(loc, descriptor, memrefType, builder);
    if (!part)
      return mlir::failure();
    parts.push_back(part);
  }
  return parts;
}

static mlir::LogicalResult
appendClassReprArgs(mlir::Location loc, ClassType classType,
                    mlir::Value objectDescriptor, mlir::func::FuncOp func,
                    mlir::ModuleOp module, mlir::OpBuilder &builder,
                    const PyLLVMTypeConverter &typeConverter,
                    llvm::SmallVectorImpl<mlir::Value> &args) {
  mlir::FunctionType fnType = func.getFunctionType();
  if (fnType.getNumInputs() == 1) {
    mlir::Type argType = fnType.getInput(0);
    if (objectDescriptor.getType() != argType) {
      auto descriptorType =
          mlir::dyn_cast<mlir::MemRefType>(objectDescriptor.getType());
      auto targetType = mlir::dyn_cast<mlir::MemRefType>(argType);
      if (descriptorType && descriptorType.getRank() == 1 &&
          descriptorType.getElementType() == argType) {
        auto objectType = class_layout::objectCarrierType(argType);
        if (!objectType)
          return mlir::failure();
        objectDescriptor = loadedClassObjectDescriptor(loc, objectDescriptor,
                                                       objectType, builder);
      } else if (!descriptorType || !targetType ||
                 descriptorType.getRank() != targetType.getRank() ||
                 descriptorType.getElementType() !=
                     targetType.getElementType()) {
        return mlir::failure();
      } else {
        objectDescriptor = builder.create<mlir::memref::CastOp>(
            loc, targetType, objectDescriptor);
      }
    }
    if (!objectDescriptor)
      return mlir::failure();
    args.push_back(objectDescriptor);
    return mlir::success();
  }

  mlir::FailureOr<class_layout::Layout> layout =
      class_layout::get(module, classType, typeConverter);
  if (mlir::failed(layout))
    return mlir::failure();
  llvm::SmallVector<mlir::Type, 4> expectedTypes;
  class_layout::partsValueTypes(*layout, expectedTypes);
  if (!llvm::equal(fnType.getInputs(), expectedTypes))
    return mlir::failure();
  auto headerType = mlir::dyn_cast<mlir::MemRefType>(expectedTypes[0]);
  auto objectType = layout->objectType;
  if (!headerType || class_layout::Object::partCount(objectType) !=
                         static_cast<int64_t>(expectedTypes.size()))
    return mlir::failure();
  mlir::Value object =
      loadedClassObjectDescriptor(loc, objectDescriptor, objectType, builder);
  if (!object)
    return mlir::failure();

  for (auto [index, expected] : llvm::enumerate(expectedTypes)) {
    auto memrefType = mlir::dyn_cast<mlir::MemRefType>(expected);
    if (!memrefType)
      return mlir::failure();
    mlir::Value descriptor = class_layout::Object::descriptor(
        loc, objectType, object, static_cast<int64_t>(index), builder);
    mlir::Value memref = classPartMemRef(loc, descriptor, memrefType, builder);
    if (!memref)
      return mlir::failure();
    args.push_back(memref);
  }
  return mlir::success();
}

static mlir::FailureOr<unicode::Value>
callClassReprSymbol(mlir::Location loc, llvm::StringRef symbol,
                    ClassType classType, mlir::Value objectDescriptor,
                    mlir::ModuleOp module, mlir::OpBuilder &builder,
                    const PyLLVMTypeConverter &typeConverter) {
  auto func = module.lookupSymbol<mlir::func::FuncOp>(symbol);
  auto strTypes = unicode::types(builder.getContext());
  if (!func || func.getFunctionType().getNumResults() != strTypes.size())
    return mlir::failure();
  if (!llvm::all_of(
          llvm::zip(func.getFunctionType().getResults(), strTypes),
          [](auto pair) { return std::get<0>(pair) == std::get<1>(pair); }))
    return mlir::failure();

  llvm::SmallVector<mlir::Value, 2> args;
  if (mlir::failed(appendClassReprArgs(loc, classType, objectDescriptor, func,
                                       module, builder, typeConverter, args)))
    return mlir::failure();

  auto call = builder.create<mlir::func::CallOp>(loc, func, args);
  call->setAttr(OwnershipContractAttrs::kOwnedResults,
                builder.getArrayAttr({builder.getI64IntegerAttr(0)}));
  if (call.getNumResults() != strTypes.size())
    return mlir::failure();

  unicode::Value result;
  result.reserve(strTypes.size());
  result.append(call.getResults().begin(), call.getResults().end());
  return result;
}

static mlir::FailureOr<unicode::Value>
callStaticClassRepr(mlir::Location loc, ClassType classType,
                    mlir::Value objectDescriptor, mlir::ModuleOp module,
                    mlir::OpBuilder &builder,
                    const PyLLVMTypeConverter &typeConverter) {
  (void)typeConverter;
  std::string reprName = getStaticClassReprCallbackName(classType);
  if (auto repr =
          callClassReprSymbol(loc, reprName, classType, objectDescriptor,
                              module, builder, typeConverter);
      mlir::succeeded(repr))
    return repr;

  std::string customName = (classType.getClassName() + ".__repr__").str();
  if (auto repr =
          callClassReprSymbol(loc, customName, classType, objectDescriptor,
                              module, builder, typeConverter);
      mlir::succeeded(repr))
    return repr;

  return mlir::failure();
}

static mlir::Value
classDescriptorForRepr(mlir::Location loc, ClassType classType,
                       mlir::Value value, mlir::ModuleOp module,
                       mlir::OpBuilder &builder,
                       const PyLLVMTypeConverter &typeConverter) {
  auto objectType =
      class_layout::objectCarrierType(class_layout::carrierStorageType(
          module, classType, typeConverter, builder.getContext()));
  if (!objectType)
    return {};

  if (value.getType() == objectType)
    return value;

  if (auto memrefType = mlir::dyn_cast<mlir::MemRefType>(value.getType())) {
    if (memrefType.getRank() == 1 &&
        memrefType.getElementType() == objectType) {
      auto targetType =
          class_layout::carrierType(objectType, builder.getContext());
      if (value.getType() == targetType)
        return value;
      return builder.create<mlir::memref::CastOp>(loc, targetType, value)
          .getResult();
    }
  }

  return {};
}

struct ReprSnapshot {
  mlir::Value storage;
  mlir::Value size;
};

struct DictReprSnapshot {
  mlir::Value keys;
  mlir::Value values;
  mlir::Value states;
  mlir::Value capacity;
};

static mlir::FailureOr<ReprSnapshot> materializeListReprSnapshot(
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
  mlir::Value snapshot = builder.create<mlir::memref::AllocaOp>(
      loc, itemsType, mlir::ValueRange{size});
  ClassType inlineClassType = mlir::dyn_cast<ClassType>(elementType);
  auto inlineCarrierType = class_layout::objectCarrierType(itemsType);
  bool inlineClass = inlineClassType && inlineCarrierType;

  mlir::Value zero = createIndexConstant(loc, builder, 0);
  mlir::Value one = createIndexConstant(loc, builder, 1);
  auto loop = builder.create<mlir::scf::ForOp>(loc, zero, size, one);
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(loop.getBody());
    mlir::Value iv = loop.getInductionVar();
    mlir::Value value;
    if (inlineClass) {
      mlir::Value sourceSlot =
          Slot::classCarrierView(loc, items, iv, inlineCarrierType, builder);
      if (!sourceSlot)
        return mlir::failure();
      mlir::Value destSlot =
          Slot::classCarrierView(loc, snapshot, iv, inlineCarrierType, builder);
      if (!destSlot)
        return mlir::failure();
      if (mlir::failed(Slot::classCarrierInitialize(
              loc, destSlot, inlineClassType, module, builder, typeConverter)))
        return mlir::failure();
      if (mlir::failed(::py::lowering::value::class_::Copy::ensure(
              loc, module, inlineClassType, builder, typeConverter)))
        return mlir::failure();
      if (mlir::failed(Slot::classCarrierCopy(
              loc, destSlot, sourceSlot, inlineClassType, module, builder)))
        return mlir::failure();
    } else {
      value = builder.create<mlir::memref::LoadOp>(loc, items, iv);
      if (markManagedAccesses)
        container::access::Contract::mark(value.getDefiningOp(), header, items);
      if (mlir::failed(Slot::refcount(loc, value, elementType, module, builder,
                                      typeConverter, "incref",
                                      /*aggregateEffect=*/true, retainPremise)))
        return mlir::failure();
      auto store =
          builder.create<mlir::memref::StoreOp>(loc, value, snapshot, iv);
      if (Slot::refcounted(elementType))
        Slot::markTransfer(store.getOperation());
    }
  }
  return ReprSnapshot{snapshot, size};
}

static mlir::LogicalResult
releaseListReprSnapshot(mlir::Location loc, mlir::Value items, mlir::Value size,
                        mlir::Type elementType, mlir::ModuleOp module,
                        mlir::OpBuilder &builder,
                        const PyLLVMTypeConverter &typeConverter) {
  if (!Slot::refcounted(elementType))
    return mlir::success();

  auto itemsType = mlir::dyn_cast<mlir::MemRefType>(items.getType());
  if (!itemsType)
    return mlir::success();
  ClassType inlineClassType = mlir::dyn_cast<ClassType>(elementType);
  auto inlineCarrierType = class_layout::objectCarrierType(itemsType);
  bool inlineClass = inlineClassType && inlineCarrierType;
  mlir::Value zero = createIndexConstant(loc, builder, 0);
  mlir::Value one = createIndexConstant(loc, builder, 1);
  auto loop = builder.create<mlir::scf::ForOp>(loc, zero, size, one);
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(loop.getBody());
    mlir::Value iv = loop.getInductionVar();
    if (inlineClass) {
      mlir::Value objectSlot =
          Slot::classCarrierView(loc, items, iv, inlineCarrierType, builder);
      if (!objectSlot)
        return mlir::failure();
      Slot::classCarrierRefcount(loc, objectSlot, inlineClassType, module,
                                 builder, "decref",
                                 /*aggregateEffect=*/true);
    } else {
      mlir::Value slot = builder.create<mlir::memref::LoadOp>(loc, items, iv);
      if (mlir::failed(Slot::refcount(loc, slot, elementType, module, builder,
                                      typeConverter, "decref",
                                      /*aggregateEffect=*/true)))
        return mlir::failure();
    }
  }
  return mlir::success();
}

static mlir::FailureOr<unicode::Value>
buildReprSlot(mlir::Location loc, mlir::Value value, mlir::Type logicalType,
              mlir::ModuleOp module, mlir::OpBuilder &builder,
              const PyLLVMTypeConverter &typeConverter) {
  RuntimeAPI runtime(module, builder, typeConverter);
  auto strTypes = unicode::types(builder.getContext());

  if (auto classType = mlir::dyn_cast<ClassType>(logicalType)) {
    mlir::Value objectDescriptor = classDescriptorForRepr(
        loc, classType, value, module, builder, typeConverter);
    if (!objectDescriptor)
      return mlir::failure();
    ownership::aggregate::Slot::markLoad(objectDescriptor);
    return callStaticClassRepr(loc, classType, objectDescriptor, module,
                               builder, typeConverter);
  }

  if (mlir::isa<BoolType>(logicalType)) {
    if (value.getType() == builder.getI1Type()) {
      auto select =
          builder.create<mlir::scf::IfOp>(loc, mlir::TypeRange(strTypes), value,
                                          /*withElseRegion=*/true);
      {
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(select.thenBlock());
        unicode::Value trueText =
            unicode::literal(loc, "True", module, builder, typeConverter,
                             /*owned=*/true);
        unicode::yield(loc, trueText, builder);
      }
      {
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(select.elseBlock());
        unicode::Value falseText =
            unicode::literal(loc, "False", module, builder, typeConverter,
                             /*owned=*/true);
        unicode::yield(loc, falseText, builder);
      }
      return unicode::Value(select.getResults());
    }
    auto intType = mlir::dyn_cast<mlir::IntegerType>(value.getType());
    if (!intType)
      return mlir::failure();
    mlir::Value zero =
        builder.create<mlir::arith::ConstantIntOp>(loc, 0, intType);
    mlir::Value isTrue = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::ne, value, zero);
    auto select =
        builder.create<mlir::scf::IfOp>(loc, mlir::TypeRange(strTypes), isTrue,
                                        /*withElseRegion=*/true);
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(select.thenBlock());
      unicode::Value trueText =
          unicode::literal(loc, "True", module, builder, typeConverter,
                           /*owned=*/true);
      unicode::yield(loc, trueText, builder);
    }
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(select.elseBlock());
      unicode::Value falseText =
          unicode::literal(loc, "False", module, builder, typeConverter,
                           /*owned=*/true);
      unicode::yield(loc, falseText, builder);
    }
    return unicode::Value(select.getResults());
  }

  if (mlir::isa<IntType>(logicalType)) {
    mlir::Value bits = widenReprSlot(loc, value, builder);
    if (!bits)
      return mlir::failure();
    auto repr = runtime.call(loc, RuntimeSymbols::kUnicodeFromI64,
                             mlir::TypeRange(strTypes), mlir::ValueRange{bits});
    return unicode::Value(repr.getResults());
  }

  if (mlir::isa<NoneType>(logicalType))
    return unicode::literal(loc, "None", module, builder, typeConverter,
                            /*owned=*/true);

  if (mlir::isa<StrType>(logicalType)) {
    mlir::FailureOr<unicode::Value> parts =
        unicodePartsFromCarrier(loc, value, builder);
    if (mlir::failed(parts))
      return mlir::failure();
    unicode::Value quote = unicode::literal(loc, "'", module, builder,
                                            typeConverter, /*owned=*/true);
    unicode::Value result = unicode::concat3(loc, quote, *parts, quote, module,
                                             builder, typeConverter);
    unicode::release(loc, quote, module, builder, typeConverter);
    return result;
  }

  return mlir::failure();
}

static mlir::FailureOr<unicode::Value>
buildListReprFromSnapshot(mlir::Location loc, mlir::Value items,
                          mlir::Value size, mlir::Type elementType,
                          mlir::ModuleOp module, mlir::OpBuilder &builder,
                          const PyLLVMTypeConverter &typeConverter) {
  auto itemsType = mlir::dyn_cast<mlir::MemRefType>(items.getType());
  if (!itemsType)
    return mlir::failure();

  auto strTypes = unicode::types(builder.getContext());
  unicode::Value initial =
      unicode::literal(loc, "[", module, builder, typeConverter,
                       /*owned=*/true);

  mlir::Value zero = createIndexConstant(loc, builder, 0);
  mlir::Value one = createIndexConstant(loc, builder, 1);
  ClassType inlineClassType = mlir::dyn_cast<ClassType>(elementType);
  auto inlineCarrierType = class_layout::objectCarrierType(itemsType);
  bool inlineClass = inlineClassType && inlineCarrierType;
  auto loop = builder.create<mlir::scf::ForOp>(loc, zero, size, one,
                                               mlir::ValueRange(initial));
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(loop.getBody());
    mlir::Value iv = loop.getInductionVar();
    unicode::Value current =
        unicode::take(loop.getRegionIterArgs(), strTypes.size());
    mlir::Value first = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::eq, iv, zero);
    auto separatorSelect =
        builder.create<mlir::scf::IfOp>(loc, mlir::TypeRange(strTypes), first,
                                        /*withElseRegion=*/true);
    {
      mlir::OpBuilder::InsertionGuard thenGuard(builder);
      builder.setInsertionPointToStart(separatorSelect.thenBlock());
      unicode::Value empty =
          unicode::literal(loc, "", module, builder, typeConverter,
                           /*owned=*/true);
      unicode::yield(loc, empty, builder);
    }
    {
      mlir::OpBuilder::InsertionGuard elseGuard(builder);
      builder.setInsertionPointToStart(separatorSelect.elseBlock());
      unicode::Value comma =
          unicode::literal(loc, ", ", module, builder, typeConverter,
                           /*owned=*/true);
      unicode::yield(loc, comma, builder);
    }
    unicode::Value separator(separatorSelect.getResults());

    mlir::Value value;
    if (inlineClass) {
      value =
          Slot::classCarrierView(loc, items, iv, inlineCarrierType, builder);
      if (!value)
        return mlir::failure();
    } else {
      value = builder.create<mlir::memref::LoadOp>(loc, items, iv);
      ownership::aggregate::Slot::markLoad(value);
    }
    mlir::FailureOr<unicode::Value> itemRepr =
        buildReprSlot(loc, value, elementType, module, builder, typeConverter);
    if (mlir::failed(itemRepr))
      return mlir::failure();
    unicode::Value next = unicode::concat3(loc, current, separator, *itemRepr,
                                           module, builder, typeConverter);
    unicode::release(loc, current, module, builder, typeConverter,
                     /*aggregateEffect=*/true);
    unicode::release(loc, separator, module, builder, typeConverter);
    unicode::release(loc, *itemRepr, module, builder, typeConverter);
    unicode::yield(loc, next, builder);
  }

  unicode::Value current(loop.getResults());
  unicode::Value close = unicode::literal(loc, "]", module, builder,
                                          typeConverter, /*owned=*/true);
  unicode::Value result =
      unicode::concat(loc, current, close, module, builder, typeConverter);
  unicode::release(loc, current, module, builder, typeConverter,
                   /*aggregateEffect=*/true);
  unicode::release(loc, close, module, builder, typeConverter);
  return result;
}

static mlir::FailureOr<unicode::Value>
buildTupleReprFromSnapshot(mlir::Location loc, mlir::Value items,
                           llvm::ArrayRef<mlir::Type> elementTypes,
                           mlir::ModuleOp module, mlir::OpBuilder &builder,
                           const PyLLVMTypeConverter &typeConverter) {
  auto itemsType = mlir::dyn_cast<mlir::MemRefType>(items.getType());
  if (!itemsType)
    return mlir::failure();

  unicode::Value current =
      unicode::literal(loc, "(", module, builder, typeConverter,
                       /*owned=*/true);
  auto inlineCarrierType = class_layout::objectCarrierType(itemsType);

  for (auto [index, elementType] : llvm::enumerate(elementTypes)) {
    unicode::Value separator =
        unicode::literal(loc, index == 0 ? "" : ", ", module, builder,
                         typeConverter, /*owned=*/true);

    mlir::Value sourceIndex =
        createIndexConstant(loc, builder, static_cast<int64_t>(index));
    ClassType inlineClassType = mlir::dyn_cast<ClassType>(elementType);
    bool inlineClass = inlineClassType && inlineCarrierType;
    mlir::Value value;
    if (inlineClass) {
      value = Slot::classCarrierView(loc, items, sourceIndex, inlineCarrierType,
                                     builder);
      if (!value)
        return mlir::failure();
      Slot::classCarrierRefcount(
          loc, value, inlineClassType, module, builder, "incref",
          /*aggregateEffect=*/true, ThreadSafetyAttrs::kPremiseAggregateBorrow);
    } else {
      value = builder.create<mlir::memref::LoadOp>(loc, items, sourceIndex);
      ownership::aggregate::Slot::markLoad(value);
      if (mlir::failed(
              Slot::refcount(loc, value, elementType, module, builder,
                             typeConverter, "incref", /*aggregateEffect=*/true,
                             ThreadSafetyAttrs::kPremiseAggregateBorrow)))
        return mlir::failure();
    }
    mlir::FailureOr<unicode::Value> itemRepr =
        buildReprSlot(loc, value, elementType, module, builder, typeConverter);
    if (mlir::failed(itemRepr))
      return mlir::failure();
    if (inlineClass) {
      Slot::classCarrierRefcount(loc, value, inlineClassType, module, builder,
                                 "decref", /*aggregateEffect=*/true);
    } else {
      if (mlir::failed(Slot::refcount(loc, value, elementType, module, builder,
                                      typeConverter, "decref",
                                      /*aggregateEffect=*/true)))
        return mlir::failure();
    }
    unicode::Value next = unicode::concat3(loc, current, separator, *itemRepr,
                                           module, builder, typeConverter);
    unicode::release(loc, current, module, builder, typeConverter,
                     /*aggregateEffect=*/true);
    unicode::release(loc, separator, module, builder, typeConverter);
    unicode::release(loc, *itemRepr, module, builder, typeConverter);
    current = next;
  }

  if (elementTypes.size() == 1) {
    unicode::Value comma = unicode::literal(loc, ",", module, builder,
                                            typeConverter, /*owned=*/true);
    unicode::Value close =
        unicode::literal(loc, ")", module, builder, typeConverter,
                         /*owned=*/true);
    unicode::Value result = unicode::concat3(loc, current, comma, close, module,
                                             builder, typeConverter);
    unicode::release(loc, current, module, builder, typeConverter,
                     /*aggregateEffect=*/true);
    unicode::release(loc, comma, module, builder, typeConverter);
    unicode::release(loc, close, module, builder, typeConverter);
    return result;
  }

  unicode::Value close = unicode::literal(loc, ")", module, builder,
                                          typeConverter, /*owned=*/true);
  unicode::Value result =
      unicode::concat(loc, current, close, module, builder, typeConverter);
  unicode::release(loc, current, module, builder, typeConverter,
                   /*aggregateEffect=*/true);
  unicode::release(loc, close, module, builder, typeConverter);
  return result;
}

static mlir::FailureOr<DictReprSnapshot> materializeDictReprSnapshot(
    mlir::Location loc, mlir::Value header, mlir::Value keys,
    mlir::Value values, mlir::Value states, DictType dictType,
    mlir::ModuleOp module, mlir::OpBuilder &builder,
    const PyLLVMTypeConverter &typeConverter, llvm::StringRef retainPremise,
    bool markManagedAccesses) {
  auto keysType = mlir::dyn_cast<mlir::MemRefType>(keys.getType());
  auto valuesType = mlir::dyn_cast<mlir::MemRefType>(values.getType());
  auto statesType = mlir::dyn_cast<mlir::MemRefType>(states.getType());
  if (!keysType || !valuesType || !statesType)
    return mlir::failure();

  mlir::Value capacityI64 = builder.create<mlir::memref::LoadOp>(
      loc, header, createIndexConstant(loc, builder, kTypedDictCapacitySlot));
  if (markManagedAccesses)
    container::access::Contract::mark(capacityI64.getDefiningOp(), header,
                                      header);
  mlir::Value capacity = builder.create<mlir::arith::IndexCastOp>(
      loc, builder.getIndexType(), capacityI64);
  mlir::Value snapshotKeys = builder.create<mlir::memref::AllocaOp>(
      loc, keysType, mlir::ValueRange{capacity});
  mlir::Value snapshotValues = builder.create<mlir::memref::AllocaOp>(
      loc, valuesType, mlir::ValueRange{capacity});
  mlir::Value snapshotStates = builder.create<mlir::memref::AllocaOp>(
      loc, statesType, mlir::ValueRange{capacity});

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
    builder.create<mlir::memref::StoreOp>(loc, state, snapshotStates, iv);

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
      mlir::Value key = keyLoad;
      if (mlir::failed(Slot::refcount(loc, key, dictType.getKeyType(), module,
                                      builder, typeConverter, "incref",
                                      /*aggregateEffect=*/true, retainPremise)))
        return mlir::failure();
      auto keyStore =
          builder.create<mlir::memref::StoreOp>(loc, key, snapshotKeys, iv);
      if (Slot::refcounted(dictType.getKeyType()))
        Slot::markTransfer(keyStore.getOperation());

      auto valueLoad = builder.create<mlir::memref::LoadOp>(loc, values, iv);
      if (markManagedAccesses)
        container::access::Contract::mark(valueLoad.getOperation(), header,
                                          values);
      mlir::Value value = valueLoad;
      if (mlir::failed(Slot::refcount(loc, value, dictType.getValueType(),
                                      module, builder, typeConverter, "incref",
                                      /*aggregateEffect=*/true, retainPremise)))
        return mlir::failure();
      auto valueStore =
          builder.create<mlir::memref::StoreOp>(loc, value, snapshotValues, iv);
      if (Slot::refcounted(dictType.getValueType()))
        Slot::markTransfer(valueStore.getOperation());
    }
  }

  return DictReprSnapshot{snapshotKeys, snapshotValues, snapshotStates,
                          capacity};
}

static mlir::LogicalResult
releaseDictReprSnapshot(mlir::Location loc, mlir::Value keys,
                        mlir::Value values, mlir::Value states,
                        mlir::Value capacity, DictType dictType,
                        mlir::ModuleOp module, mlir::OpBuilder &builder,
                        const PyLLVMTypeConverter &typeConverter) {
  bool keyRefcounted = Slot::refcounted(dictType.getKeyType());
  bool valueRefcounted = Slot::refcounted(dictType.getValueType());
  if (!keyRefcounted && !valueRefcounted)
    return mlir::success();

  mlir::Value zero = createIndexConstant(loc, builder, 0);
  mlir::Value one = createIndexConstant(loc, builder, 1);
  auto loop = builder.create<mlir::scf::ForOp>(loc, zero, capacity, one);
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(loop.getBody());
    mlir::Value iv = loop.getInductionVar();
    mlir::Value state = builder.create<mlir::memref::LoadOp>(loc, states, iv);
    mlir::Value occupied = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::eq, state,
        builder.create<mlir::arith::ConstantIntOp>(loc, 1, state.getType()));
    auto releaseIfOccupied = builder.create<mlir::scf::IfOp>(
        loc, occupied, /*withElseRegion=*/false);
    {
      mlir::OpBuilder::InsertionGuard ifGuard(builder);
      builder.setInsertionPointToStart(releaseIfOccupied.thenBlock());
      if (keyRefcounted) {
        mlir::Value key = builder.create<mlir::memref::LoadOp>(loc, keys, iv);
        if (mlir::failed(Slot::refcount(loc, key, dictType.getKeyType(), module,
                                        builder, typeConverter, "decref",
                                        /*aggregateEffect=*/true)))
          return mlir::failure();
      }
      if (valueRefcounted) {
        mlir::Value value =
            builder.create<mlir::memref::LoadOp>(loc, values, iv);
        if (mlir::failed(Slot::refcount(loc, value, dictType.getValueType(),
                                        module, builder, typeConverter,
                                        "decref",
                                        /*aggregateEffect=*/true)))
          return mlir::failure();
      }
    }
  }
  return mlir::success();
}

static mlir::FailureOr<unicode::Value>
buildDictReprFromSnapshot(mlir::Location loc, mlir::Value keys,
                          mlir::Value values, mlir::Value states,
                          mlir::Value capacity, DictType dictType,
                          mlir::ModuleOp module, mlir::OpBuilder &builder,
                          const PyLLVMTypeConverter &typeConverter) {
  if (!mlir::dyn_cast<mlir::MemRefType>(keys.getType()) ||
      !mlir::dyn_cast<mlir::MemRefType>(values.getType()) ||
      !mlir::dyn_cast<mlir::MemRefType>(states.getType()))
    return mlir::failure();

  auto strTypes = unicode::types(builder.getContext());
  unicode::Value initial =
      unicode::literal(loc, "{", module, builder, typeConverter,
                       /*owned=*/true);
  mlir::Value emittedInit = createI64Constant(loc, builder, 0);

  mlir::Value zero = createIndexConstant(loc, builder, 0);
  mlir::Value one = createIndexConstant(loc, builder, 1);
  llvm::SmallVector<mlir::Value, 4> initArgs;
  initArgs.append(initial.begin(), initial.end());
  initArgs.push_back(emittedInit);
  auto loop = builder.create<mlir::scf::ForOp>(loc, zero, capacity, one,
                                               mlir::ValueRange(initArgs));
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(loop.getBody());
    mlir::Value iv = loop.getInductionVar();
    unicode::Value current =
        unicode::take(loop.getRegionIterArgs(), strTypes.size());
    mlir::Value emitted = loop.getRegionIterArgs()[strTypes.size()];
    mlir::Value state = builder.create<mlir::memref::LoadOp>(loc, states, iv);
    mlir::Value occupied = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::eq, state,
        builder.create<mlir::arith::ConstantIntOp>(loc, 1, state.getType()));
    llvm::SmallVector<mlir::Type, 3> ifResultTypes(strTypes.begin(),
                                                   strTypes.end());
    ifResultTypes.push_back(builder.getI64Type());
    auto emitIfOccupied = builder.create<mlir::scf::IfOp>(
        loc, mlir::TypeRange(ifResultTypes), occupied,
        /*withElseRegion=*/true);
    {
      mlir::OpBuilder::InsertionGuard ifGuard(builder);
      builder.setInsertionPointToStart(emitIfOccupied.thenBlock());
      mlir::Value first = builder.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::eq, emitted,
          createI64Constant(loc, builder, 0));
      auto separatorSelect =
          builder.create<mlir::scf::IfOp>(loc, mlir::TypeRange(strTypes), first,
                                          /*withElseRegion=*/true);
      {
        mlir::OpBuilder::InsertionGuard thenGuard(builder);
        builder.setInsertionPointToStart(separatorSelect.thenBlock());
        unicode::Value empty =
            unicode::literal(loc, "", module, builder, typeConverter,
                             /*owned=*/true);
        unicode::yield(loc, empty, builder);
      }
      {
        mlir::OpBuilder::InsertionGuard elseGuard(builder);
        builder.setInsertionPointToStart(separatorSelect.elseBlock());
        unicode::Value comma =
            unicode::literal(loc, ", ", module, builder, typeConverter,
                             /*owned=*/true);
        unicode::yield(loc, comma, builder);
      }
      unicode::Value separator(separatorSelect.getResults());

      mlir::Value keyBits = builder.create<mlir::memref::LoadOp>(loc, keys, iv);
      ownership::aggregate::Slot::markLoad(keyBits);
      mlir::FailureOr<unicode::Value> keyRepr = buildReprSlot(
          loc, keyBits, dictType.getKeyType(), module, builder, typeConverter);
      if (mlir::failed(keyRepr))
        return mlir::failure();
      unicode::Value withKey = unicode::concat3(
          loc, current, separator, *keyRepr, module, builder, typeConverter);
      unicode::release(loc, current, module, builder, typeConverter,
                       /*aggregateEffect=*/true);
      unicode::release(loc, separator, module, builder, typeConverter);
      unicode::release(loc, *keyRepr, module, builder, typeConverter);

      unicode::Value colon = unicode::literal(loc, ": ", module, builder,
                                              typeConverter, /*owned=*/true);

      mlir::Value valueBits =
          builder.create<mlir::memref::LoadOp>(loc, values, iv);
      ownership::aggregate::Slot::markLoad(valueBits);
      mlir::FailureOr<unicode::Value> valueRepr =
          buildReprSlot(loc, valueBits, dictType.getValueType(), module,
                        builder, typeConverter);
      if (mlir::failed(valueRepr))
        return mlir::failure();
      unicode::Value next = unicode::concat3(loc, withKey, colon, *valueRepr,
                                             module, builder, typeConverter);
      unicode::release(loc, withKey, module, builder, typeConverter);
      unicode::release(loc, colon, module, builder, typeConverter);
      unicode::release(loc, *valueRepr, module, builder, typeConverter);
      mlir::Value emittedNext = builder.create<mlir::arith::AddIOp>(
          loc, emitted, createI64Constant(loc, builder, 1));
      llvm::SmallVector<mlir::Value, 4> yielded;
      yielded.append(next.begin(), next.end());
      yielded.push_back(emittedNext);
      builder.create<mlir::scf::YieldOp>(loc, yielded);
    }
    {
      mlir::OpBuilder::InsertionGuard elseGuard(builder);
      builder.setInsertionPointToStart(emitIfOccupied.elseBlock());
      llvm::SmallVector<mlir::Value, 4> yielded;
      yielded.append(current.begin(), current.end());
      yielded.push_back(emitted);
      builder.create<mlir::scf::YieldOp>(loc, yielded);
    }
    builder.create<mlir::scf::YieldOp>(loc, emitIfOccupied.getResults());
  }

  unicode::Value current = unicode::take(loop.getResults(), strTypes.size());
  unicode::Value close = unicode::literal(loc, "}", module, builder,
                                          typeConverter, /*owned=*/true);
  unicode::Value result =
      unicode::concat(loc, current, close, module, builder, typeConverter);
  unicode::release(loc, current, module, builder, typeConverter,
                   /*aggregateEffect=*/true);
  unicode::release(loc, close, module, builder, typeConverter);
  return result;
}

/// Container repr lowering is still housed in this file for historical reasons.
/// Generic object upcast has been removed from the dialect; repr lowering must
/// keep static element types throughout.
struct ContainerReprLowering : public mlir::OpConversionPattern<ReprOp> {
  ContainerReprLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<ReprOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(ReprOp op, OneToNOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (!module || adaptor.getInput().empty())
      return mlir::failure();
    auto *converter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    mlir::Type inputType = op.getInput().getType();

    if (auto listType = mlir::dyn_cast<ListType>(inputType)) {
      if (!isCompilerOwnedMemRefListType(listType))
        return mlir::failure();

      mlir::Value reprMemref;
      mlir::Value reprSnapshotSize;
      bool releaseSnapshot = false;
      if (adaptor.getInput().size() == 3) {
        mlir::Value header = adaptor.getInput()[kListHeaderComponent];
        mlir::Value lock = adaptor.getInput()[kListLockComponent];
        mlir::Value items = adaptor.getInput()[kListItemsComponent];
        auto snapshotType = items.getType();
        mlir::Value isManaged = container::Managed::predicate(
            op.getLoc(), header, kTypedListRefcountSlot, rewriter);
        auto lockedCopy = rewriter.create<mlir::scf::IfOp>(
            op.getLoc(), mlir::TypeRange{snapshotType, rewriter.getIndexType()},
            isManaged,
            /*withElseRegion=*/true);
        {
          mlir::OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointToStart(lockedCopy.thenBlock());
          container::Managed::lock(op.getLoc(), lock, rewriter);
          mlir::FailureOr<ReprSnapshot> snapshot = materializeListReprSnapshot(
              op.getLoc(), header, items, listType.getElementType(), module,
              rewriter, *converter, ThreadSafetyAttrs::kPremiseLockedBorrow,
              /*markManagedAccesses=*/true);
          if (mlir::failed(snapshot))
            return mlir::failure();
          container::Managed::unlock(op.getLoc(), lock, rewriter);
          rewriter.create<mlir::scf::YieldOp>(
              op.getLoc(), mlir::ValueRange{snapshot->storage, snapshot->size});
        }
        {
          mlir::OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointToStart(lockedCopy.elseBlock());
          mlir::FailureOr<ReprSnapshot> snapshot = materializeListReprSnapshot(
              op.getLoc(), header, items, listType.getElementType(), module,
              rewriter, *converter, ThreadSafetyAttrs::kPremiseAggregateBorrow,
              /*markManagedAccesses=*/false);
          if (mlir::failed(snapshot))
            return mlir::failure();
          rewriter.create<mlir::scf::YieldOp>(
              op.getLoc(), mlir::ValueRange{snapshot->storage, snapshot->size});
        }
        reprMemref = lockedCopy.getResult(0);
        reprSnapshotSize = lockedCopy.getResult(1);
        releaseSnapshot = Slot::refcounted(listType.getElementType());
      } else if (adaptor.getInput().size() == 1) {
        reprMemref = adaptor.getInput().front();
      } else {
        return mlir::failure();
      }
      mlir::Value snapshotSize = reprSnapshotSize;
      if (!snapshotSize) {
        auto sizeLoad = rewriter.create<mlir::memref::LoadOp>(
            op.getLoc(), reprMemref,
            createIndexConstant(op.getLoc(), rewriter, 0));
        snapshotSize = rewriter.create<mlir::arith::IndexCastOp>(
            op.getLoc(), rewriter.getIndexType(), sizeLoad);
      }
      mlir::FailureOr<unicode::Value> repr = buildListReprFromSnapshot(
          op.getLoc(), reprMemref, snapshotSize, listType.getElementType(),
          module, rewriter, *converter);
      if (mlir::failed(repr))
        return mlir::failure();
      if (releaseSnapshot)
        if (mlir::failed(releaseListReprSnapshot(
                op.getLoc(), reprMemref, snapshotSize,
                listType.getElementType(), module, rewriter, *converter)))
          return mlir::failure();
      rewriter.replaceOpWithMultiple(
          op, llvm::ArrayRef<mlir::ValueRange>{mlir::ValueRange(*repr)});
      return mlir::success();
    }

    if (auto tupleType = mlir::dyn_cast<TupleType>(inputType)) {
      if (!isCompilerOwnedMemRefTupleType(tupleType) ||
          adaptor.getInput().size() != 2)
        return mlir::failure();

      mlir::Value items = adaptor.getInput().back();
      auto itemsType = mlir::dyn_cast<mlir::MemRefType>(items.getType());
      if (!itemsType)
        return rewriter.notifyMatchFailure(
            op, "tuple repr requires memref item storage");
      auto elements = tupleType.getElementTypes();

      mlir::FailureOr<unicode::Value> repr = buildTupleReprFromSnapshot(
          op.getLoc(), items, elements, module, rewriter, *converter);
      if (mlir::failed(repr))
        return mlir::failure();
      rewriter.replaceOpWithMultiple(
          op, llvm::ArrayRef<mlir::ValueRange>{mlir::ValueRange(*repr)});
      return mlir::success();
    }

    if (auto dictType = mlir::dyn_cast<DictType>(inputType)) {
      if (!isCompilerOwnedMemRefDictType(dictType) ||
          adaptor.getInput().size() != 5)
        return mlir::failure();

      mlir::Value header = adaptor.getInput()[kDictHeaderComponent];
      mlir::Value lock = adaptor.getInput()[kDictLockComponent];
      mlir::Value keys = adaptor.getInput()[kDictKeysComponent];
      mlir::Value values = adaptor.getInput()[kDictValuesComponent];
      mlir::Value states = adaptor.getInput()[kDictStatesComponent];
      mlir::Value isManaged = container::Managed::predicate(
          op.getLoc(), header, kTypedDictRefcountSlot, rewriter);
      auto lockedCopy = rewriter.create<mlir::scf::IfOp>(
          op.getLoc(),
          mlir::TypeRange{keys.getType(), values.getType(), states.getType(),
                          rewriter.getIndexType()},
          isManaged, /*withElseRegion=*/true);
      {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(lockedCopy.thenBlock());
        container::Managed::lock(op.getLoc(), lock, rewriter);
        mlir::FailureOr<DictReprSnapshot> snapshot =
            materializeDictReprSnapshot(
                op.getLoc(), header, keys, values, states, dictType, module,
                rewriter, *converter, ThreadSafetyAttrs::kPremiseLockedBorrow,
                /*markManagedAccesses=*/true);
        if (mlir::failed(snapshot))
          return mlir::failure();
        container::Managed::unlock(op.getLoc(), lock, rewriter);
        rewriter.create<mlir::scf::YieldOp>(
            op.getLoc(),
            mlir::ValueRange{snapshot->keys, snapshot->values, snapshot->states,
                             snapshot->capacity});
      }
      {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(lockedCopy.elseBlock());
        mlir::FailureOr<DictReprSnapshot> snapshot =
            materializeDictReprSnapshot(
                op.getLoc(), header, keys, values, states, dictType, module,
                rewriter, *converter,
                ThreadSafetyAttrs::kPremiseAggregateBorrow,
                /*markManagedAccesses=*/false);
        if (mlir::failed(snapshot))
          return mlir::failure();
        rewriter.create<mlir::scf::YieldOp>(
            op.getLoc(),
            mlir::ValueRange{snapshot->keys, snapshot->values, snapshot->states,
                             snapshot->capacity});
      }

      mlir::FailureOr<unicode::Value> repr = buildDictReprFromSnapshot(
          op.getLoc(), lockedCopy.getResult(0), lockedCopy.getResult(1),
          lockedCopy.getResult(2), lockedCopy.getResult(3), dictType, module,
          rewriter, *converter);
      if (mlir::failed(repr))
        return mlir::failure();
      if (mlir::failed(releaseDictReprSnapshot(
              op.getLoc(), lockedCopy.getResult(0), lockedCopy.getResult(1),
              lockedCopy.getResult(2), lockedCopy.getResult(3), dictType,
              module, rewriter, *converter)))
        return mlir::failure();
      rewriter.replaceOpWithMultiple(
          op, llvm::ArrayRef<mlir::ValueRange>{mlir::ValueRange(*repr)});
      return mlir::success();
    }

    return mlir::failure();
  }
};

} // namespace

namespace lowering::runtime::upcast::Patterns {

void populate(PyLLVMTypeConverter &typeConverter,
              mlir::RewritePatternSet &patterns) {
  patterns.add<ContainerReprLowering>(typeConverter, patterns.getContext());
}

} // namespace lowering::runtime::upcast::Patterns

} // namespace py
