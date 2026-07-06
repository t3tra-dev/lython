#include "Runtime/Core/Lowerer.h"

#include "mlir/Dialect/SCF/IR/SCF.h"

namespace py::lowering {

namespace {

namespace own = py::ownership;

bool isNoneLikeType(mlir::Type type) { return py::isPyNoneType(type); }

std::string describeSlotType(mlir::Type type) {
  std::string name = runtimeContractName(type);
  if (!name.empty())
    return name;
  std::string text;
  llvm::raw_string_ostream os(text);
  type.print(os);
  return os.str();
}

std::string markerName(mlir::Type type, llvm::StringRef slotName) {
  std::string marker = describeSlotType(type);
  if (!slotName.empty()) {
    marker += ":";
    marker += slotName.str();
  }
  return marker;
}

mlir::FailureOr<mlir::Value> buildHeaderView(mlir::Operation *op,
                                             mlir::OpBuilder &builder,
                                             mlir::Value header,
                                             mlir::Type targetType) {
  if (header.getType() == targetType)
    return header;

  auto sourceType = mlir::dyn_cast<mlir::MemRefType>(header.getType());
  auto targetMemRef = mlir::dyn_cast<mlir::MemRefType>(targetType);
  if (!sourceType || !targetMemRef || sourceType.getRank() != 1 ||
      targetMemRef.getRank() != 1 ||
      sourceType.getElementType() != targetMemRef.getElementType() ||
      sourceType.getMemorySpace() != targetMemRef.getMemorySpace())
    return op->emitError() << "aggregate slot object value " << header.getType()
                           << " cannot be viewed as " << targetType;

  if (sourceType.getDimSize(0) == targetMemRef.getDimSize(0))
    return mlir::memref::CastOp::create(builder, op->getLoc(), targetType,
                                        header)
        .getResult();

  if (sourceType.hasStaticShape() && targetMemRef.hasStaticShape() &&
      sourceType.getDimSize(0) >= targetMemRef.getDimSize(0)) {
    llvm::SmallVector<mlir::OpFoldResult, 1> offsets{builder.getIndexAttr(0)};
    llvm::SmallVector<mlir::OpFoldResult, 1> sizes{
        builder.getIndexAttr(targetMemRef.getDimSize(0))};
    llvm::SmallVector<mlir::OpFoldResult, 1> strides{builder.getIndexAttr(1)};
    llvm::SmallVector<int64_t, 1> resultShape{targetMemRef.getDimSize(0)};
    auto inferredType = mlir::cast<mlir::MemRefType>(
        mlir::memref::SubViewOp::inferRankReducedResultType(
            resultShape, sourceType, offsets, sizes, strides));
    mlir::Value view =
        mlir::memref::SubViewOp::create(builder, op->getLoc(), inferredType,
                                        header, offsets, sizes, strides)
            .getResult();
    if (view.getType() == targetMemRef)
      return view;
    return mlir::memref::CastOp::create(builder, op->getLoc(), targetMemRef,
                                        view)
        .getResult();
  }

  return op->emitError() << "aggregate slot object value " << header.getType()
                         << " cannot be viewed as " << targetType;
}

} // namespace

mlir::func::FuncOp RuntimeBundleLowerer::findRetainFunction() const {
  mlir::ModuleOp moduleOp = module;
  mlir::func::FuncOp retained;
  moduleOp.walk([&](mlir::func::FuncOp function) {
    auto primitive =
        function->getAttrOfType<mlir::StringAttr>(kManifestPrimitiveAttr);
    if (primitive && primitive.getValue() == "retain")
      retained = function;
  });
  return retained;
}

mlir::LogicalResult RuntimeBundleLowerer::retainAggregateSlot(
    mlir::Operation *op, mlir::Type slotType, mlir::ValueRange values,
    llvm::StringRef slotName) {
  return RuntimeBundleLowerer::retainAggregateSlot(op, slotType, values,
                                                   slotName, /*depth=*/0);
}

mlir::LogicalResult
RuntimeBundleLowerer::retainAggregateSlot(mlir::Operation *op,
                                          const RuntimeBundle &slotValue,
                                          llvm::StringRef slotName) {
  const RuntimeBundle *concrete =
      RuntimeBundleLowerer::concreteObjectForOwnership(slotValue);
  if (!concrete || concrete->kind != RuntimeBundle::Kind::Object)
    return op->emitError() << "aggregate slot retain requires an object bundle";
  return RuntimeBundleLowerer::retainAggregateSlot(
      op, concrete->objectValue.contract, concrete->physicalValues(), slotName);
}

mlir::LogicalResult RuntimeBundleLowerer::releaseAggregateSlot(
    mlir::Operation *op, mlir::Type slotType, mlir::ValueRange values,
    llvm::StringRef slotName) {
  llvm::SmallVector<own::RuntimeDeallocator, 8> deallocators =
      own::collectRuntimeDeallocators(module);
  return RuntimeBundleLowerer::releaseAggregateSlot(op, slotType, values,
                                                    slotName, deallocators,
                                                    /*depth=*/0);
}

mlir::LogicalResult
RuntimeBundleLowerer::releaseAggregateSlot(mlir::Operation *op,
                                           const RuntimeBundle &slotValue,
                                           llvm::StringRef slotName) {
  const RuntimeBundle *concrete =
      RuntimeBundleLowerer::concreteObjectForOwnership(slotValue);
  if (!concrete || concrete->kind != RuntimeBundle::Kind::Object)
    return op->emitError()
           << "aggregate slot release requires an object bundle";
  return RuntimeBundleLowerer::releaseAggregateSlot(
      op, concrete->objectValue.contract, concrete->physicalValues(), slotName);
}

mlir::LogicalResult RuntimeBundleLowerer::replaceAggregateSlot(
    mlir::Operation *op, mlir::Type oldType, mlir::ValueRange oldValues,
    mlir::Type newType, mlir::ValueRange newValues, llvm::StringRef slotName) {
  if (mlir::failed(RuntimeBundleLowerer::retainAggregateSlot(
          op, newType, newValues, slotName)))
    return mlir::failure();
  return RuntimeBundleLowerer::releaseAggregateSlot(op, oldType, oldValues,
                                                    slotName);
}

mlir::LogicalResult RuntimeBundleLowerer::replaceAggregateSlot(
    mlir::Operation *op, mlir::Type oldType, mlir::ValueRange oldValues,
    const RuntimeBundle *oldSlotValue, mlir::Type newType,
    const RuntimeBundle &newSlotValue, llvm::StringRef slotName,
    bool releaseMissingOldObjectSlot) {
  (void)newType;
  if (mlir::failed(RuntimeBundleLowerer::retainAggregateSlot(op, newSlotValue,
                                                             slotName)))
    return mlir::failure();

  if (oldSlotValue)
    return RuntimeBundleLowerer::releaseAggregateSlot(op, *oldSlotValue,
                                                      slotName);

  if (RuntimeBundleLowerer::isBuiltinsObjectContract(oldType) &&
      !releaseMissingOldObjectSlot)
    return mlir::success();

  return RuntimeBundleLowerer::releaseAggregateSlot(op, oldType, oldValues,
                                                    slotName);
}

mlir::LogicalResult RuntimeBundleLowerer::retainAggregateSlot(
    mlir::Operation *op, mlir::Type slotType, mlir::ValueRange values,
    llvm::StringRef slotName, unsigned depth) {
  if (depth > 16)
    return op->emitError()
           << "aggregate ownership retain recursion is too deep for "
           << slotType;
  if (values.empty() || isNoneLikeType(slotType))
    return mlir::success();

  if (auto unionType = mlir::dyn_cast<py::UnionType>(slotType)) {
    if (values.empty())
      return op->emitError() << "union aggregate retain value has no tag";
    context->loadDialect<mlir::scf::SCFDialect>();
    mlir::Value tag = values.front();
    unsigned offset = 1;
    for (auto [memberIndex, member] :
         llvm::enumerate(unionType.getMemberTypes())) {
      mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> memberTypes =
          RuntimeBundleLowerer::runtimeValueTypesFor(
              op, member, "union aggregate retain ABI");
      if (mlir::failed(memberTypes))
        return mlir::failure();
      unsigned size = static_cast<unsigned>(memberTypes->size());
      if (offset + size > values.size())
        return op->emitError()
               << "union aggregate retain member exceeds value group";
      if (size == 0) {
        offset += size;
        continue;
      }

      mlir::Value expected = mlir::arith::ConstantIntOp::create(
          builder, op->getLoc(), static_cast<std::int64_t>(memberIndex), 64);
      mlir::Value active = mlir::arith::CmpIOp::create(
          builder, op->getLoc(), mlir::arith::CmpIPredicate::eq, tag, expected);
      auto ifOp =
          mlir::scf::IfOp::create(builder, op->getLoc(), mlir::TypeRange{},
                                  active, /*withElseRegion=*/false);
      builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
      if (mlir::failed(RuntimeBundleLowerer::retainAggregateSlot(
              op, member, values.slice(offset, size), slotName, depth + 1)))
        return mlir::failure();
      mlir::scf::YieldOp::create(builder, op->getLoc());
      builder.setInsertionPointAfter(ifOp);
      offset += size;
    }
    return mlir::success();
  }

  if (!own::isObjectHeaderLikeType(values.front().getType()))
    return mlir::success();

  mlir::func::FuncOp retain = RuntimeBundleLowerer::findRetainFunction();
  if (!retain)
    return op->emitError()
           << "aggregate slot retain requires a runtime retain primitive";
  if (retain.getFunctionType().getNumInputs() != 1)
    return retain.emitError()
           << "runtime retain primitive must accept one object header";

  mlir::FailureOr<mlir::Value> header = buildHeaderView(
      op, builder, values.front(), retain.getFunctionType().getInput(0));
  if (mlir::failed(header))
    return mlir::failure();
  auto call =
      mlir::func::CallOp::create(builder, op->getLoc(), retain, *header);
  call->setAttr(own::kAggregateRetainAttr,
                builder.getStringAttr(markerName(slotType, slotName)));
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::releaseAggregateSlot(
    mlir::Operation *op, mlir::Type slotType, mlir::ValueRange values,
    llvm::StringRef slotName,
    llvm::ArrayRef<own::RuntimeDeallocator> deallocators, unsigned depth) {
  if (depth > 16)
    return op->emitError()
           << "aggregate ownership release recursion is too deep for "
           << slotType;
  if (values.empty() || isNoneLikeType(slotType))
    return mlir::success();

  if (auto unionType = mlir::dyn_cast<py::UnionType>(slotType)) {
    if (values.empty())
      return op->emitError() << "union aggregate release value has no tag";
    context->loadDialect<mlir::scf::SCFDialect>();
    mlir::Value tag = values.front();
    unsigned offset = 1;
    for (auto [memberIndex, member] :
         llvm::enumerate(unionType.getMemberTypes())) {
      mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> memberTypes =
          RuntimeBundleLowerer::runtimeValueTypesFor(
              op, member, "union aggregate release ABI");
      if (mlir::failed(memberTypes))
        return mlir::failure();
      unsigned size = static_cast<unsigned>(memberTypes->size());
      if (offset + size > values.size())
        return op->emitError()
               << "union aggregate release member exceeds value group";
      if (size == 0) {
        offset += size;
        continue;
      }

      mlir::Value expected = mlir::arith::ConstantIntOp::create(
          builder, op->getLoc(), static_cast<std::int64_t>(memberIndex), 64);
      mlir::Value active = mlir::arith::CmpIOp::create(
          builder, op->getLoc(), mlir::arith::CmpIPredicate::eq, tag, expected);
      auto ifOp =
          mlir::scf::IfOp::create(builder, op->getLoc(), mlir::TypeRange{},
                                  active, /*withElseRegion=*/false);
      builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
      if (mlir::failed(RuntimeBundleLowerer::releaseAggregateSlot(
              op, member, values.slice(offset, size), slotName, deallocators,
              depth + 1)))
        return mlir::failure();
      mlir::scf::YieldOp::create(builder, op->getLoc());
      builder.setInsertionPointAfter(ifOp);
      offset += size;
    }
    return mlir::success();
  }

  if (!own::isObjectHeaderLikeType(values.front().getType()))
    return mlir::success();

  std::string contract = runtimeContractName(slotType);
  const own::RuntimeDeallocator *deallocator =
      own::findDeallocatorForValueGroup(values, 0, deallocators, contract);
  if (!deallocator || deallocator->inputTypes.size() != values.size()) {
    if (RuntimeBundleLowerer::classForContract(slotType))
      return op->emitError() << "aggregate slot release for " << slotType
                             << " has no matching runtime deallocator";
    return mlir::success();
  }

  auto call = mlir::func::CallOp::create(builder, op->getLoc(),
                                         deallocator->function, values);
  call->setAttr(own::kAggregateReleaseAttr,
                builder.getStringAttr(markerName(slotType, slotName)));
  return mlir::success();
}

} // namespace py::lowering
