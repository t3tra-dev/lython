#include "Runtime/Core/Lowerer.h"

namespace py::lowering {

namespace {

namespace own = py::ownership;

bool compatibleRankOneMemRefStorage(mlir::Type source, mlir::Type target,
                                    bool targetMustBeDynamic) {
  auto sourceMemRef = mlir::dyn_cast<mlir::MemRefType>(source);
  auto targetMemRef = mlir::dyn_cast<mlir::MemRefType>(target);
  if (!sourceMemRef || !targetMemRef || sourceMemRef.getRank() != 1 ||
      targetMemRef.getRank() != 1 ||
      sourceMemRef.getElementType() != targetMemRef.getElementType() ||
      sourceMemRef.getMemorySpace() != targetMemRef.getMemorySpace())
    return false;
  if (targetMustBeDynamic)
    return targetMemRef.getDimSize(0) == mlir::ShapedType::kDynamic;
  return sourceMemRef.getShape() == targetMemRef.getShape();
}

mlir::Value boolConstant(mlir::OpBuilder &builder, mlir::Location loc,
                         bool value) {
  return builder.create<mlir::arith::ConstantIntOp>(loc, value ? 1 : 0, 1)
      .getResult();
}

bool canAppendExactValues(mlir::FunctionType functionType, unsigned inputIndex,
                          mlir::ValueRange values) {
  if (values.empty() ||
      inputIndex + values.size() > functionType.getNumInputs())
    return false;
  for (auto [offset, value] : llvm::enumerate(values))
    if (value.getType() != functionType.getInput(inputIndex + offset))
      return false;
  return true;
}

bool runtimeInputConsumesObject(const RuntimeSymbol &symbol,
                                unsigned inputIndex) {
  return symbol.function &&
         own::functionConsumesOperandAt(symbol.function, inputIndex);
}

mlir::LogicalResult rejectConsumingObjectView(mlir::Operation *op,
                                              const RuntimeSymbol &symbol,
                                              unsigned inputIndex,
                                              const RuntimeBundle &source,
                                              llvm::StringRef viewName) {
  return op->emitError()
         << "cannot pass " << source.contractName() << " to consuming input "
         << inputIndex << " of " << symbol.contract << "." << symbol.name
         << " through a " << viewName
         << "; object ownership requires the concrete runtime value group";
}

} // namespace

bool RuntimeBundleLowerer::canAppendExactValueSequence(
    mlir::FunctionType functionType, unsigned inputIndex,
    const RuntimeBundle &source) const {
  return canAppendExactValues(functionType, inputIndex,
                              source.physicalValues());
}

mlir::LogicalResult RuntimeBundleLowerer::appendRuntimeSource(
    mlir::Operation *op, const RuntimeSymbol &symbol,
    mlir::FunctionType functionType, unsigned &inputIndex,
    const RuntimeBundle &source, llvm::SmallVectorImpl<mlir::Value> &operands) {
  llvm::ArrayRef<mlir::Value> sourceValues = source.physicalValues();
  if (canAppendExactValues(functionType, inputIndex, sourceValues)) {
    if (source.contractName() == "builtins.object" &&
        sourceValues.size() == 1 &&
        isBuiltinsObjectHandleType(functionType.getInput(inputIndex)) &&
        runtimeInputConsumesObject(symbol, inputIndex))
      return rejectConsumingObjectView(op, symbol, inputIndex, source,
                                       "borrowed builtins.object handle");
    operands.append(sourceValues.begin(), sourceValues.end());
    inputIndex += static_cast<unsigned>(sourceValues.size());
    return mlir::success();
  }

  mlir::Type expected = functionType.getInput(inputIndex);
  std::optional<RuntimeValue> materializedObject;
  auto materializeLazySource = [&]() -> mlir::LogicalResult {
    if (materializedObject)
      return mlir::success();
    mlir::FailureOr<RuntimeValue> value =
        RuntimeBundleLowerer::materializePrimitiveI64ObjectAtCurrentInsertion(
            op, source);
    if (mlir::failed(value))
      return mlir::failure();
    materializedObject = std::move(*value);
    sourceValues = materializedObject->values;
    return mlir::success();
  };

  if (RuntimeBundleLowerer::hasLazyPrimitiveI64Object(source) &&
      expected.isInteger(64)) {
    operands.push_back(source.primitiveI64->value);
    ++inputIndex;
    return mlir::success();
  }

  if (RuntimeBundleLowerer::hasLazyPrimitiveI64Object(source)) {
    if (mlir::failed(materializeLazySource()))
      return mlir::failure();
    if (canAppendExactValues(functionType, inputIndex, sourceValues)) {
      operands.append(sourceValues.begin(), sourceValues.end());
      inputIndex += static_cast<unsigned>(sourceValues.size());
      return mlir::success();
    }
  }

  if (source.kind == RuntimeBundle::Kind::Object &&
      isErasedObjectStorageType(expected)) {
    std::optional<RuntimeBundle> boxedSource;
    if (!RuntimeBundleLowerer::isBuiltinsObjectContract(source.contract)) {
      mlir::FailureOr<RuntimeBundle> boxed =
          RuntimeBundleLowerer::boxRuntimeObjectAtCurrentInsertion(
              op, source, runtimeInputConsumesObject(symbol, inputIndex));
      if (mlir::failed(boxed))
        return mlir::failure();
      boxedSource = std::move(*boxed);
    }
    const RuntimeBundle &storageSource = boxedSource ? *boxedSource : source;
    if (runtimeInputConsumesObject(symbol, inputIndex) && !boxedSource)
      return rejectConsumingObjectView(op, symbol, inputIndex, source,
                                       "erased object storage view");
    const RuntimeValue &objectValue =
        boxedSource ? boxedSource->objectValue
                    : (materializedObject ? *materializedObject
                                          : storageSource.objectValue);
    mlir::FailureOr<mlir::Value> storage =
        RuntimeBundleLowerer::erasedObjectStorageView(op, objectValue,
                                                      expected);
    if (mlir::failed(storage))
      return mlir::failure();
    operands.push_back(*storage);
    ++inputIndex;
    return mlir::success();
  }

  if (source.kind == RuntimeBundle::Kind::Object &&
      isBuiltinsObjectHandleType(expected)) {
    if (!RuntimeBundleLowerer::isBuiltinsObjectContract(source.contract))
      return op->emitError()
             << "cannot pass concrete object " << source.contractName()
             << " as builtins.object runtime input " << inputIndex << " of "
             << symbol.contract << "." << symbol.name
             << "; box the object at the owning ABI boundary first";
    if (runtimeInputConsumesObject(symbol, inputIndex))
      return rejectConsumingObjectView(op, symbol, inputIndex, source,
                                       "borrowed builtins.object handle");
    if (sourceValues.empty())
      return op->emitError() << "builtins.object argument has no boxed handle";
    mlir::Value handle = sourceValues.front();
    if (handle.getType() != expected) {
      if (!compatibleRankOneMemRefStorage(handle.getType(), expected,
                                          /*targetMustBeDynamic=*/false))
        return op->emitError()
               << "builtins.object handle " << handle.getType()
               << " cannot be adapted to runtime input " << inputIndex
               << " of " << symbol.contract << "." << symbol.name;
      handle = builder.create<mlir::memref::CastOp>(op->getLoc(), expected,
                                                    handle);
    }
    operands.push_back(handle);
    ++inputIndex;
    return mlir::success();
  }

  if (source.kind == RuntimeBundle::Kind::TypeObject &&
      expected.isInteger(64)) {
    std::optional<std::int64_t> id =
        manifest.classId(source.instanceContractName());
    if (!id)
      return op->emitError() << "type object has no runtime class id for "
                             << source.instanceContractName();
    mlir::Value value =
        builder.create<mlir::arith::ConstantIntOp>(op->getLoc(), *id, 64)
            .getResult();
    operands.push_back(value);
    ++inputIndex;
    return mlir::success();
  }

  if (expected.isInteger(64)) {
    std::optional<RuntimeSymbol> unbox =
        manifest.primitive(source.contractName(), "unbox.i64");
    if (unbox) {
      mlir::func::CallOp call = RuntimeBundleLowerer::createRuntimeCall(
          op->getLoc(), *unbox, sourceValues);
      if (call.getNumResults() == 1 &&
          call.getResult(0).getType() == expected) {
        operands.push_back(call.getResult(0));
        ++inputIndex;
        return mlir::success();
      }
    }
  }

  if (expected == builder.getF64Type()) {
    std::optional<RuntimeSymbol> unbox =
        manifest.primitive(source.contractName(), "unbox.f64");
    if (unbox) {
      mlir::func::CallOp call = RuntimeBundleLowerer::createRuntimeCall(
          op->getLoc(), *unbox, sourceValues);
      if (call.getNumResults() == 1 &&
          call.getResult(0).getType() == expected) {
        operands.push_back(call.getResult(0));
        ++inputIndex;
        return mlir::success();
      }
    }
  }

  return op->emitError() << "cannot adapt " << source.contractName()
                         << " to runtime input " << inputIndex << " of "
                         << symbol.contract << "." << symbol.name;
}

mlir::LogicalResult RuntimeBundleLowerer::appendRuntimeSourceAs(
    mlir::Operation *op, const RuntimeSymbol &symbol,
    mlir::FunctionType functionType, unsigned &inputIndex,
    const RuntimeBundle &source, mlir::Type expected,
    llvm::SmallVectorImpl<mlir::Value> &operands) {
  if (auto expectedUnion = mlir::dyn_cast_if_present<py::UnionType>(expected)) {
    mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> expectedTypes =
        RuntimeBundleLowerer::runtimeValueTypesFor(
            op, expectedUnion, "callable union argument ABI");
    if (mlir::failed(expectedTypes))
      return mlir::failure();
    if (inputIndex + expectedTypes->size() > functionType.getNumInputs())
      return op->emitError() << "union argument ABI for " << symbol.contract
                             << "." << symbol.name << " exceeds runtime inputs";
    for (auto [offset, expectedType] : llvm::enumerate(*expectedTypes)) {
      mlir::Type physicalType = functionType.getInput(inputIndex + offset);
      if (physicalType != expectedType)
        return op->emitError()
               << "union argument ABI for " << symbol.contract << "."
               << symbol.name << " expects runtime input "
               << inputIndex + offset << " to be " << expectedType
               << ", but function ABI has " << physicalType;
    }

    llvm::SmallVector<mlir::Value, 8> unionValues;
    if (mlir::failed(RuntimeBundleLowerer::appendUnionRuntimeValues(
            op, expectedUnion, source, source.contract, unionValues)))
      return mlir::failure();
    if (unionValues.size() != expectedTypes->size())
      return op->emitError()
             << "union argument produced " << unionValues.size()
             << " physical values, but ABI expects " << expectedTypes->size();
    operands.append(unionValues.begin(), unionValues.end());
    inputIndex += static_cast<unsigned>(unionValues.size());
    return mlir::success();
  }

  return RuntimeBundleLowerer::appendRuntimeSource(
      op, symbol, functionType, inputIndex, source, operands);
}

mlir::LogicalResult RuntimeBundleLowerer::appendPrimitiveI64EvidenceOperand(
    mlir::Operation *op, mlir::FunctionType functionType, unsigned &inputIndex,
    const RuntimeBundle &source, llvm::SmallVectorImpl<mlir::Value> &operands) {
  if (source.contractName() != "builtins.int" ||
      inputIndex + 2 > functionType.getNumInputs() ||
      !functionType.getInput(inputIndex).isInteger(64) ||
      !functionType.getInput(inputIndex + 1).isInteger(1))
    return mlir::success();

  if (source.primitiveI64) {
    operands.push_back(source.primitiveI64->value);
    operands.push_back(source.primitiveI64->valid);
  } else {
    operands.push_back(
        builder.create<mlir::arith::ConstantIntOp>(op->getLoc(), 0, 64)
            .getResult());
    operands.push_back(boolConstant(builder, op->getLoc(), false));
  }
  inputIndex += 2;
  return mlir::success();
}

bool RuntimeBundleLowerer::canAppendRuntimeSource(
    const RuntimeSymbol &symbol, mlir::FunctionType functionType,
    unsigned &inputIndex, const RuntimeBundle &source) const {
  llvm::ArrayRef<mlir::Value> sourceValues = source.physicalValues();
  if (canAppendExactValues(functionType, inputIndex, sourceValues)) {
    if (source.contractName() == "builtins.object" &&
        sourceValues.size() == 1 &&
        isBuiltinsObjectHandleType(functionType.getInput(inputIndex)) &&
        runtimeInputConsumesObject(symbol, inputIndex))
      return false;
    inputIndex += static_cast<unsigned>(sourceValues.size());
    return true;
  }
  if (inputIndex >= functionType.getNumInputs())
    return false;

  mlir::Type expected = functionType.getInput(inputIndex);
  if (RuntimeBundleLowerer::hasLazyPrimitiveI64Object(source)) {
    if (expected.isInteger(64)) {
      ++inputIndex;
      return true;
    }
    const RuntimeValueShape *shape = manifest.valueShape("builtins.int");
    if (shape &&
        inputIndex + shape->valueTypes.size() <= functionType.getNumInputs()) {
      bool exact = true;
      for (auto [offset, type] : llvm::enumerate(shape->valueTypes)) {
        if (type != functionType.getInput(inputIndex + offset)) {
          exact = false;
          break;
        }
      }
      if (exact) {
        inputIndex += static_cast<unsigned>(shape->valueTypes.size());
        return true;
      }
      if (!shape->valueTypes.empty() && isBuiltinsObjectHandleType(expected) &&
          compatibleRankOneMemRefStorage(shape->valueTypes.front(), expected,
                                         /*targetMustBeDynamic=*/false)) {
        if (runtimeInputConsumesObject(symbol, inputIndex))
          return false;
        ++inputIndex;
        return true;
      }
      if (!shape->valueTypes.empty() && isErasedObjectStorageType(expected) &&
          compatibleRankOneMemRefStorage(shape->valueTypes.front(), expected,
                                         /*targetMustBeDynamic=*/true)) {
        if (runtimeInputConsumesObject(symbol, inputIndex))
          return false;
        ++inputIndex;
        return true;
      }
    }
  }
  if (source.kind == RuntimeBundle::Kind::Object &&
      isErasedObjectStorageType(expected) && !sourceValues.empty() &&
      compatibleRankOneMemRefStorage(sourceValues.front().getType(), expected,
                                     /*targetMustBeDynamic=*/true)) {
    if (!RuntimeBundleLowerer::isBuiltinsObjectContract(source.contract))
      return false;
    if (runtimeInputConsumesObject(symbol, inputIndex))
      return false;
    ++inputIndex;
    return true;
  }

  if (source.kind == RuntimeBundle::Kind::Object &&
      isBuiltinsObjectHandleType(expected) && !sourceValues.empty() &&
      (sourceValues.front().getType() == expected ||
       compatibleRankOneMemRefStorage(sourceValues.front().getType(), expected,
                                      /*targetMustBeDynamic=*/false))) {
    if (!RuntimeBundleLowerer::isBuiltinsObjectContract(source.contract))
      return false;
    if (runtimeInputConsumesObject(symbol, inputIndex))
      return false;
    ++inputIndex;
    return true;
  }

  if (source.kind == RuntimeBundle::Kind::TypeObject &&
      expected.isInteger(64) &&
      manifest.classId(source.instanceContractName())) {
    ++inputIndex;
    return true;
  }

  if (expected.isInteger(64)) {
    std::optional<RuntimeSymbol> unbox =
        manifest.primitive(source.contractName(), "unbox.i64");
    if (unbox && unbox->function.getFunctionType().getNumResults() == 1 &&
        unbox->function.getFunctionType().getResult(0) == expected) {
      ++inputIndex;
      return true;
    }
  }

  if (expected.isF64()) {
    std::optional<RuntimeSymbol> unbox =
        manifest.primitive(source.contractName(), "unbox.f64");
    if (unbox && unbox->function.getFunctionType().getNumResults() == 1 &&
        unbox->function.getFunctionType().getResult(0) == expected) {
      ++inputIndex;
      return true;
    }
  }

  (void)symbol;
  return false;
}

mlir::LogicalResult RuntimeBundleLowerer::appendImplicitRuntimeArgument(
    mlir::Operation *op, const RuntimeSymbol &symbol, unsigned &inputIndex,
    llvm::SmallVectorImpl<mlir::Value> &operands) {
  if (const RuntimeDefaultArgument *defaultArgument =
          symbol.defaultArgument(inputIndex)) {
    if (defaultArgument->kind == RuntimeDefaultArgument::Kind::I64) {
      auto integerDefault =
          mlir::cast<mlir::IntegerAttr>(defaultArgument->value);
      operands.push_back(builder
                             .create<mlir::arith::ConstantIntOp>(
                                 op->getLoc(), integerDefault.getInt(), 64)
                             .getResult());
      ++inputIndex;
      return mlir::success();
    }
    auto floatDefault = mlir::cast<mlir::FloatAttr>(defaultArgument->value);
    operands.push_back(
        builder
            .create<mlir::arith::ConstantFloatOp>(
                op->getLoc(), floatDefault.getValue(), builder.getF64Type())
            .getResult());
    ++inputIndex;
    return mlir::success();
  }
  return op->emitError() << "runtime call is missing input " << inputIndex
                         << " for " << symbol.contract << "." << symbol.name;
}

bool RuntimeBundleLowerer::canAppendImplicitRuntimeArgument(
    const RuntimeSymbol &symbol, unsigned &inputIndex) const {
  if (!symbol.defaultArgument(inputIndex))
    return false;
  ++inputIndex;
  return true;
}

bool RuntimeBundleLowerer::canBuildRuntimeCallOperands(
    const RuntimeSymbol &symbol, llvm::ArrayRef<const RuntimeBundle *> sources,
    bool allowUnusedSources, const RuntimeBundle *classObject) const {
  mlir::func::FuncOp function = symbol.function;
  mlir::FunctionType functionType = function.getFunctionType();
  unsigned inputIndex = 0;
  unsigned sourceIndex = 0;
  while (inputIndex < functionType.getNumInputs()) {
    if (symbol.hasClassIdArgument(inputIndex)) {
      if (!classObject)
        return false;
      if (!canAppendRuntimeSource(symbol, functionType, inputIndex,
                                  *classObject))
        return false;
      continue;
    }
    if (sourceIndex >= sources.size()) {
      if (!canAppendImplicitRuntimeArgument(symbol, inputIndex))
        return false;
      continue;
    }
    if (!sources[sourceIndex] ||
        !canAppendRuntimeSource(symbol, functionType, inputIndex,
                                *sources[sourceIndex]))
      return false;
    ++sourceIndex;
  }
  return allowUnusedSources || sourceIndex == sources.size();
}

mlir::LogicalResult RuntimeBundleLowerer::buildRuntimeCallOperands(
    mlir::Operation *op, const RuntimeSymbol &symbol,
    llvm::ArrayRef<const RuntimeBundle *> sources,
    llvm::SmallVectorImpl<mlir::Value> &operands, bool allowUnusedSources,
    const RuntimeBundle *classObject) {
  mlir::func::FuncOp function = symbol.function;
  mlir::FunctionType functionType = function.getFunctionType();
  unsigned inputIndex = 0;
  unsigned sourceIndex = 0;
  while (inputIndex < functionType.getNumInputs()) {
    if (symbol.hasClassIdArgument(inputIndex)) {
      if (!classObject)
        return op->emitError()
               << "runtime class id input " << inputIndex << " for "
               << symbol.contract << "." << symbol.name
               << " has no lowered class object source";
      if (mlir::failed(appendRuntimeSource(op, symbol, functionType, inputIndex,
                                           *classObject, operands)))
        return mlir::failure();
      continue;
    }
    if (sourceIndex >= sources.size()) {
      if (mlir::failed(
              appendImplicitRuntimeArgument(op, symbol, inputIndex, operands)))
        return mlir::failure();
      continue;
    }
    if (mlir::failed(appendRuntimeSource(op, symbol, functionType, inputIndex,
                                         *sources[sourceIndex], operands)))
      return mlir::failure();
    ++sourceIndex;
  }
  if (!allowUnusedSources && sourceIndex != sources.size())
    return op->emitError() << "too many runtime arguments for "
                           << symbol.contract << "." << symbol.name;
  return mlir::success();
}

} // namespace py::lowering
