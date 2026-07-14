#include "Runtime/Core/Lowerer.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace py::lowering {

// Module-level int globals are backed by a single process-lifetime i64 cell.
// Reads/writes are a plain llvm.load/llvm.store, so accessing a module global
// never allocates -- an async-signal-safe channel for signal handlers to
// exchange primitive state. The stored
// representation is the UNBOXED i64 value; the boxed int object is
// reconstructed on demand at each read (box-on-read), and the value is
// unboxed at each write (unbox-on-write).
mlir::LLVM::GlobalOp
RuntimeBundleLowerer::moduleGlobalStorage(mlir::Operation *op,
                                          llvm::StringRef name) {
  std::string symbol = ("__ly_module_global_" + name).str();
  if (auto existing = module.lookupSymbol<mlir::LLVM::GlobalOp>(symbol))
    return existing;
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(module.getBody());
  mlir::Type i64 = builder.getI64Type();
  auto global = mlir::LLVM::GlobalOp::create(
      builder, op->getLoc(), i64, /*isConstant=*/false,
      mlir::LLVM::Linkage::Internal, symbol,
      builder.getI64IntegerAttr(0), /*alignment=*/8);
  return global;
}

namespace {

// Element type of a rank-1 memref, or null when the value is not one.
mlir::Type rankOneElementType(mlir::Type type) {
  auto memref = mlir::dyn_cast<mlir::MemRefType>(type);
  if (!memref || memref.getRank() != 1)
    return {};
  return memref.getElementType();
}

llvm::StringRef globalViewSymbolFor(mlir::Type element) {
  if (element.isInteger(8))
    return "__ly_global_view_i8";
  if (element.isInteger(32))
    return "__ly_global_view_i32";
  if (element.isInteger(64))
    return "__ly_global_view_i64";
  if (element.isF64())
    return "__ly_global_view_f64";
  return {};
}

} // namespace

// Cells backing one module-global OBJECT value: one i64 llvm.global per
// stored word (`_init` bound flag, then `_p<i>`/`_s<i>` pointer and size
// words per rank-1 memref physical value, or `_v<i>` for scalar physical
// values such as a union tag). Objects park here with one retained
// reference; rebinding releases the previous holder.
mlir::LLVM::GlobalOp
RuntimeBundleLowerer::moduleObjectGlobalCell(mlir::Operation *op,
                                             llvm::StringRef name,
                                             llvm::StringRef suffix) {
  std::string symbol =
      ("__ly_module_global_obj_" + name + "_" + suffix).str();
  if (auto existing = module.lookupSymbol<mlir::LLVM::GlobalOp>(symbol))
    return existing;
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(module.getBody());
  mlir::Type i64 = builder.getI64Type();
  return mlir::LLVM::GlobalOp::create(
      builder, module.getLoc(), i64, /*isConstant=*/false,
      mlir::LLVM::Linkage::Internal, symbol, builder.getI64IntegerAttr(0),
      /*alignment=*/8);
}

mlir::func::FuncOp
RuntimeBundleLowerer::globalViewFunction(mlir::Operation *op,
                                         mlir::Type element) {
  llvm::StringRef symbol = globalViewSymbolFor(element);
  if (symbol.empty())
    return {};
  if (auto existing = module.lookupSymbol<mlir::func::FuncOp>(symbol))
    return existing;
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(module.getBody());
  mlir::Type i64 = builder.getI64Type();
  auto resultType = mlir::MemRefType::get({mlir::ShapedType::kDynamic},
                                          element);
  auto fn = mlir::func::FuncOp::create(
      builder, module.getLoc(), symbol,
      builder.getFunctionType({i64, i64}, {resultType}));
  fn.setPrivate();
  return fn;
}

mlir::Value RuntimeBundleLowerer::loadObjectGlobalWord(
    mlir::Operation *op, llvm::StringRef name, llvm::StringRef suffix) {
  mlir::LLVM::GlobalOp cell =
      RuntimeBundleLowerer::moduleObjectGlobalCell(op, name, suffix);
  mlir::Value address =
      mlir::LLVM::AddressOfOp::create(builder, op->getLoc(), cell);
  return mlir::LLVM::LoadOp::create(builder, op->getLoc(),
                                    builder.getI64Type(), address);
}

void RuntimeBundleLowerer::storeObjectGlobalWord(mlir::Operation *op,
                                                 llvm::StringRef name,
                                                 llvm::StringRef suffix,
                                                 mlir::Value word) {
  mlir::LLVM::GlobalOp cell =
      RuntimeBundleLowerer::moduleObjectGlobalCell(op, name, suffix);
  mlir::Value address =
      mlir::LLVM::AddressOfOp::create(builder, op->getLoc(), cell);
  mlir::LLVM::StoreOp::create(builder, op->getLoc(), word, address);
}

// Reassemble the physical value group of a module-global object from its
// cells (the inverse of the store side in lowerGlobalSet).
mlir::LogicalResult RuntimeBundleLowerer::loadObjectGlobalValues(
    mlir::Operation *op, llvm::StringRef name,
    llvm::ArrayRef<mlir::Type> valueTypes,
    llvm::SmallVectorImpl<mlir::Value> &values) {
  mlir::Location loc = op->getLoc();
  for (auto [index, valueType] : llvm::enumerate(valueTypes)) {
    std::string slot = std::to_string(index);
    if (mlir::Type element = rankOneElementType(valueType)) {
      mlir::func::FuncOp view =
          RuntimeBundleLowerer::globalViewFunction(op, element);
      if (!view)
        return op->emitError()
               << "module global '" << name << "' value " << index
               << " has unsupported element type " << valueType;
      mlir::Value pointer =
          RuntimeBundleLowerer::loadObjectGlobalWord(op, name, "p" + slot);
      mlir::Value size =
          RuntimeBundleLowerer::loadObjectGlobalWord(op, name, "s" + slot);
      mlir::Value dynamic =
          mlir::func::CallOp::create(builder, loc, view,
                                     mlir::ValueRange{pointer, size})
              .getResult(0);
      auto memref = mlir::cast<mlir::MemRefType>(valueType);
      mlir::Value value = dynamic;
      if (memref.hasStaticShape())
        value = mlir::memref::CastOp::create(builder, loc, valueType, dynamic)
                    .getResult();
      values.push_back(value);
      continue;
    }
    mlir::Value word =
        RuntimeBundleLowerer::loadObjectGlobalWord(op, name, "v" + slot);
    if (valueType.isInteger(64)) {
      values.push_back(word);
    } else if (valueType.isInteger(1)) {
      values.push_back(
          mlir::arith::TruncIOp::create(builder, loc, builder.getI1Type(),
                                        word)
              .getResult());
    } else if (valueType.isF64()) {
      values.push_back(mlir::arith::BitcastOp::create(
                           builder, loc, builder.getF64Type(), word)
                           .getResult());
    } else {
      return op->emitError() << "module global '" << name << "' value "
                             << index << " has unsupported scalar type "
                             << valueType;
    }
  }
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerGlobalGet(py::GlobalGetOp op) {
  if (runtimeContractName(op.getResult().getType()) != "builtins.int")
    return RuntimeBundleLowerer::lowerObjectGlobalGet(op);
  mlir::LLVM::GlobalOp storage =
      RuntimeBundleLowerer::moduleGlobalStorage(op, op.getName());

  builder.setInsertionPoint(op);
  mlir::Value address =
      mlir::LLVM::AddressOfOp::create(builder, op.getLoc(), storage);
  mlir::Value raw = mlir::LLVM::LoadOp::create(builder, op.getLoc(),
                                               builder.getI64Type(), address);
  mlir::Value valid =
      mlir::arith::ConstantIntOp::create(builder, op.getLoc(), 1, 1);
  RuntimeBundle result;
  if (mlir::failed(RuntimeBundleLowerer::makePrimitiveI64Bundle(
          op, runtimeContractType(context, "builtins.int"), raw, valid,
          result)))
    return mlir::failure();
  valueBundles[op.getResult()] = std::move(result);
  erase.push_back(op);
  return mlir::success();
}

// The object path: cells hold the value group's raw words plus one
// retained reference. Reads reassemble the group, retain it for the reader
// (the refcount insertion releases it after use), and an unbound read
// raises RuntimeError through the shared manifest raise helper.
mlir::LogicalResult
RuntimeBundleLowerer::lowerObjectGlobalGet(py::GlobalGetOp op) {
  mlir::Type type = op.getResult().getType();
  mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> valueTypes =
      RuntimeBundleLowerer::runtimeValueTypesFor(op, type,
                                                 "module global object ABI");
  if (mlir::failed(valueTypes))
    return mlir::failure();

  builder.setInsertionPoint(op);
  mlir::Location loc = op.getLoc();
  mlir::Value bound = RuntimeBundleLowerer::loadObjectGlobalWord(
      op, op.getName(), "init");
  mlir::Value zero =
      mlir::arith::ConstantIntOp::create(builder, loc, 0, 64).getResult();
  mlir::Value unbound = mlir::arith::CmpIOp::create(
      builder, loc, mlir::arith::CmpIPredicate::eq, bound, zero);
  mlir::func::FuncOp raise =
      module.lookupSymbol<mlir::func::FuncOp>("__ly_long_raise_message");
  if (!raise)
    return op.emitError() << "runtime manifest raise helper is missing";
  auto guard = mlir::scf::IfOp::create(builder, loc, mlir::TypeRange{},
                                       unbound, /*withElseRegion=*/false);
  {
    mlir::OpBuilder::InsertionGuard insertionGuard(builder);
    builder.setInsertionPointToStart(&guard.getThenRegion().front());
    std::string message = ("module global '" + op.getName() +
                           "' referenced before assignment")
                              .str();
    mlir::Value messageBuffer =
        RuntimeBundleLowerer::materializeByteBuffer(loc, message);
    mlir::Value classId =
        mlir::arith::ConstantIntOp::create(builder, loc, 51, 64).getResult();
    mlir::Value length =
        mlir::arith::ConstantIntOp::create(
            builder, loc, static_cast<std::int64_t>(message.size()), 64)
            .getResult();
    mlir::func::CallOp::create(
        builder, loc, raise,
        mlir::ValueRange{classId, messageBuffer, length});
  }

  llvm::SmallVector<mlir::Value, 8> values;
  if (mlir::failed(RuntimeBundleLowerer::loadObjectGlobalValues(
          op, op.getName(), *valueTypes, values)))
    return mlir::failure();
  // The reader takes its own reference so a later rebinding of the global
  // cannot release the object out from under it.
  if (mlir::failed(RuntimeBundleLowerer::retainAggregateSlot(
          op.getOperation(), type, values, "module.global")))
    return mlir::failure();
  RuntimeBundle result;
  if (mlir::failed(RuntimeBundleLowerer::makeObjectBundleWithOwnership(
          op.getOperation(), type, values, result,
          ownership::OwnershipKind::Own)))
    return mlir::failure();
  valueBundles[op.getResult()] = std::move(result);
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult
RuntimeBundleLowerer::lowerObjectGlobalSet(py::GlobalSetOp op) {
  const RuntimeBundle *value = RuntimeBundleLowerer::bundleFor(op.getValue());
  if (!value)
    return op.emitError() << "module global assignment value has no bundle";
  mlir::Type type = op.getValue().getType();
  mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> valueTypes =
      RuntimeBundleLowerer::runtimeValueTypesFor(op, type,
                                                 "module global object ABI");
  if (mlir::failed(valueTypes))
    return mlir::failure();
  llvm::ArrayRef<mlir::Value> newValues = value->physicalValues();
  if (newValues.size() != valueTypes->size())
    return op.emitError() << "module global '" << op.getName()
                          << "' assignment value group has "
                          << newValues.size() << " values, expected "
                          << valueTypes->size();

  builder.setInsertionPoint(op);
  mlir::Location loc = op.getLoc();
  // Retain the new value before releasing the old one so a self-assignment
  // (X = X) never drops the object to zero in between.
  if (mlir::failed(RuntimeBundleLowerer::retainAggregateSlot(
          op.getOperation(), type, newValues, "module.global")))
    return mlir::failure();

  mlir::Value bound = RuntimeBundleLowerer::loadObjectGlobalWord(
      op, op.getName(), "init");
  mlir::Value zero =
      mlir::arith::ConstantIntOp::create(builder, loc, 0, 64).getResult();
  mlir::Value wasBound = mlir::arith::CmpIOp::create(
      builder, loc, mlir::arith::CmpIPredicate::ne, bound, zero);
  auto releaseOld = mlir::scf::IfOp::create(builder, loc, mlir::TypeRange{},
                                            wasBound,
                                            /*withElseRegion=*/false);
  {
    mlir::OpBuilder::InsertionGuard insertionGuard(builder);
    builder.setInsertionPointToStart(&releaseOld.getThenRegion().front());
    llvm::SmallVector<mlir::Value, 8> oldValues;
    if (mlir::failed(RuntimeBundleLowerer::loadObjectGlobalValues(
            op, op.getName(), *valueTypes, oldValues)))
      return mlir::failure();
    if (mlir::failed(RuntimeBundleLowerer::releaseAggregateSlot(
            op.getOperation(), type, oldValues, "module.global")))
      return mlir::failure();
  }

  for (auto [index, pair] : llvm::enumerate(llvm::zip(newValues,
                                                      *valueTypes))) {
    auto [newValue, valueType] = pair;
    std::string slot = std::to_string(index);
    if (mlir::Type element = rankOneElementType(valueType)) {
      mlir::Value pointerIndex =
          mlir::memref::ExtractAlignedPointerAsIndexOp::create(builder, loc,
                                                               newValue);
      mlir::Value pointer = mlir::arith::IndexCastOp::create(
          builder, loc, builder.getI64Type(), pointerIndex);
      RuntimeBundleLowerer::storeObjectGlobalWord(op, op.getName(),
                                                  "p" + slot, pointer);
      auto memref = mlir::cast<mlir::MemRefType>(valueType);
      mlir::Value size;
      if (memref.hasStaticShape()) {
        size = mlir::arith::ConstantIntOp::create(
                   builder, loc, memref.getDimSize(0), 64)
                   .getResult();
      } else {
        mlir::Value dim =
            mlir::memref::DimOp::create(builder, loc, newValue, 0);
        size = mlir::arith::IndexCastOp::create(builder, loc,
                                                builder.getI64Type(), dim);
      }
      RuntimeBundleLowerer::storeObjectGlobalWord(op, op.getName(),
                                                  "s" + slot, size);
      continue;
    }
    mlir::Value word;
    if (valueType.isInteger(64)) {
      word = newValue;
    } else if (valueType.isInteger(1)) {
      word = mlir::arith::ExtUIOp::create(builder, loc, builder.getI64Type(),
                                          newValue)
                 .getResult();
    } else if (valueType.isF64()) {
      word = mlir::arith::BitcastOp::create(builder, loc,
                                            builder.getI64Type(), newValue)
                 .getResult();
    } else {
      return op.emitError() << "module global '" << op.getName()
                            << "' value " << index
                            << " has unsupported scalar type " << valueType;
    }
    RuntimeBundleLowerer::storeObjectGlobalWord(op, op.getName(), "v" + slot,
                                                word);
  }
  mlir::Value one =
      mlir::arith::ConstantIntOp::create(builder, loc, 1, 64).getResult();
  RuntimeBundleLowerer::storeObjectGlobalWord(op, op.getName(), "init", one);
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerGlobalSet(py::GlobalSetOp op) {
  const RuntimeBundle *value = RuntimeBundleLowerer::bundleFor(op.getValue());
  if (!value)
    return op.emitError() << "module global assignment value has no bundle";
  if (runtimeContractName(op.getValue().getType()) != "builtins.int")
    return RuntimeBundleLowerer::lowerObjectGlobalSet(op);
  mlir::LLVM::GlobalOp storage =
      RuntimeBundleLowerer::moduleGlobalStorage(op, op.getName());

  builder.setInsertionPoint(op);
  mlir::Value raw;
  if (value->primitiveI64 && value->primitiveI64->value) {
    raw = value->primitiveI64->value;
  } else {
    std::optional<RuntimeSymbol> unbox =
        manifest.primitive(value->contractName(), "unbox.i64");
    if (!unbox ||
        unbox->function.getNumArguments() != value->physicalValues().size())
      return op.emitError() << "module global assignment value "
                            << value->contractName()
                            << " has no unbox.i64 primitive";
    mlir::func::CallOp unboxCall = RuntimeBundleLowerer::createRuntimeCall(
        op.getLoc(), *unbox, value->physicalValues());
    raw = unboxCall.getResult(0);
  }
  mlir::Value address =
      mlir::LLVM::AddressOfOp::create(builder, op.getLoc(), storage);
  mlir::LLVM::StoreOp::create(builder, op.getLoc(), raw, address);
  erase.push_back(op);
  return mlir::success();
}

} // namespace py::lowering
