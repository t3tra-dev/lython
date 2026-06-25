#include "RuntimeLowering/RuntimeLowering.h"

namespace py::runtime_lowering {

namespace {

bool compatibleMemRefView(mlir::Type source, mlir::Type target) {
  auto sourceMemRef = mlir::dyn_cast<mlir::MemRefType>(source);
  auto targetMemRef = mlir::dyn_cast<mlir::MemRefType>(target);
  return sourceMemRef && targetMemRef &&
         sourceMemRef.getShape() == targetMemRef.getShape() &&
         sourceMemRef.getElementType() == targetMemRef.getElementType() &&
         sourceMemRef.getMemorySpace() == targetMemRef.getMemorySpace();
}

} // namespace

bool RuntimeBundleLowerer::canAppendExactValueSequence(
    mlir::FunctionType functionType, unsigned inputIndex,
    const RuntimeBundle &source) const {
  llvm::ArrayRef<mlir::Value> sourceValues = source.physicalValues();
  if (sourceValues.empty() ||
      inputIndex + sourceValues.size() > functionType.getNumInputs())
    return false;
  for (auto [offset, value] : llvm::enumerate(sourceValues)) {
    if (value.getType() != functionType.getInput(inputIndex + offset))
      return false;
  }
  return true;
}

mlir::LogicalResult RuntimeBundleLowerer::appendRuntimeSource(
    mlir::Operation *op, const RuntimeSymbol &symbol,
    mlir::FunctionType functionType, unsigned &inputIndex,
    const RuntimeBundle &source, llvm::SmallVectorImpl<mlir::Value> &operands) {
  llvm::ArrayRef<mlir::Value> sourceValues = source.physicalValues();
  if (canAppendExactValueSequence(functionType, inputIndex, source)) {
    operands.append(sourceValues.begin(), sourceValues.end());
    inputIndex += static_cast<unsigned>(sourceValues.size());
    return mlir::success();
  }

  mlir::Type expected = functionType.getInput(inputIndex);
  if (source.kind == RuntimeBundle::Kind::Object) {
    const RuntimeValueShape *objectShape =
        manifest.valueShape("builtins.object");
    if (objectShape && objectShape->valueTypes.size() == 1 &&
        objectShape->valueTypes.front() == expected && !sourceValues.empty() &&
        sourceValues.front().getType() == expected) {
      operands.push_back(sourceValues.front());
      ++inputIndex;
      return mlir::success();
    }
    if (objectShape && objectShape->valueTypes.size() == 1 &&
        objectShape->valueTypes.front() == expected && !sourceValues.empty() &&
        compatibleMemRefView(sourceValues.front().getType(), expected)) {
      mlir::Value header = builder.create<mlir::memref::CastOp>(
          op->getLoc(), expected, sourceValues.front());
      operands.push_back(header);
      ++inputIndex;
      return mlir::success();
    }
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

mlir::LogicalResult RuntimeBundleLowerer::lowerBinarySpecial(
    mlir::Operation *op, mlir::Value lhs, mlir::Value rhs,
    llvm::StringRef methodName, mlir::Value resultValue) {
  const RuntimeBundle *lhsBundle = RuntimeBundleLowerer::bundleFor(lhs);
  const RuntimeBundle *rhsBundle = RuntimeBundleLowerer::bundleFor(rhs);
  if (!lhsBundle || !rhsBundle)
    return op->emitError()
           << "binary special method operands need runtime bundles";
  llvm::SmallVector<const RuntimeBundle *, 2> sources{lhsBundle, rhsBundle};
  if (mlir::failed(lowerManifestMethodResult(
          op, resultValue, *lhsBundle, methodName, sources,
          /*allowUnusedSources=*/false,
          /*preferManifestObjectResult=*/true)))
    return mlir::failure();
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::collectSingleBuiltinArgument(
    py::CallOp op, const RuntimeSymbol &symbol,
    const RuntimeBundle *&argument) const {
  const RuntimeBundle *posargs =
      RuntimeBundleLowerer::bundleFor(op.getPosargs());
  if (!posargs || posargs->kind != RuntimeBundle::Kind::Aggregate)
    return op.emitError() << "builtin callable '" << symbol.builtinName
                          << "' requires packed positional arguments";
  if (mlir::failed(requireEmptyAggregate(op, op.getKwnames(), "kw names")) ||
      mlir::failed(requireEmptyAggregate(op, op.getKwvalues(), "kw values")))
    return mlir::failure();
  if (posargs->aggregateOperands.size() != 1)
    return op.emitError() << "builtin callable '" << symbol.builtinName
                          << "' expects exactly one positional argument";

  argument = RuntimeBundleLowerer::bundleFor(posargs->aggregateOperands[0]);
  if (!argument)
    return op.emitError() << "builtin callable '" << symbol.builtinName
                          << "' argument has no lowered runtime bundle";
  return mlir::success();
}

mlir::LogicalResult
RuntimeBundleLowerer::lowerBuiltinMethodCall(py::CallOp op,
                                             const RuntimeSymbol &symbol) {
  if (op.getNumResults() != 1)
    return op.emitError() << "builtin callable '" << symbol.builtinName
                          << "' method lowering must produce one result";

  const RuntimeBundle *argument = nullptr;
  if (mlir::failed(collectSingleBuiltinArgument(op, symbol, argument)))
    return mlir::failure();

  llvm::SmallVector<const RuntimeBundle *, 1> sources{argument};
  std::optional<EmittedRuntimeCall> emitted;
  if (mlir::failed(RuntimeBundleLowerer::emitManifestMethodCall(
          op, *argument, symbol.builtinMethod, sources,
          /*allowUnusedSources=*/false, emitted)))
    return mlir::failure();

  std::string resultContract = runtimeContractName(op.getResult(0).getType());
  if (resultContract.empty() || resultContract == "builtins.object")
    resultContract = symbol.resultContract;
  if (resultContract.empty())
    return op.emitError() << "builtin callable '" << symbol.builtinName
                          << "' needs a concrete result contract";

  RuntimeBundle result;
  if (mlir::failed(
          bundleRuntimeResults(op, runtimeContractType(context, resultContract),
                               emitted->call, result)))
    return mlir::failure();
  valueBundles[op.getResult(0)] = std::move(result);
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult
RuntimeBundleLowerer::lowerBuiltinMethodSinkCall(py::CallOp op,
                                                 const RuntimeSymbol &symbol) {
  const RuntimeBundle *argument = nullptr;
  if (mlir::failed(collectSingleBuiltinArgument(op, symbol, argument)))
    return mlir::failure();

  RuntimeBundle printable = *argument;
  if (printable.contractName() != symbol.builtinSinkContract) {
    llvm::SmallVector<const RuntimeBundle *, 1> sources{argument};
    std::optional<EmittedRuntimeCall> emitted;
    if (mlir::failed(
            emitManifestMethodCall(op, *argument, symbol.builtinMethod, sources,
                                   /*allowUnusedSources=*/false, emitted)))
      return mlir::failure();
    if (mlir::failed(bundleRuntimeResults(
            op, runtimeContractType(context, symbol.builtinSinkContract),
            emitted->call, printable)))
      return mlir::failure();
  }
  if (printable.contractName() != symbol.builtinSinkContract)
    return op.emitError() << "builtin callable '" << symbol.builtinName
                          << "' requires a " << symbol.builtinSinkContract
                          << "-compatible argument";

  builder.setInsertionPoint(op);
  RuntimeBundleLowerer::createRuntimeCall(op.getLoc(), symbol,
                                          printable.physicalValues());
  std::string resultContract =
      symbol.resultContract.empty() ? "types.NoneType" : symbol.resultContract;
  for (mlir::Value result : op.getResults()) {
    if (mlir::failed(assignObjectBundle(
            op, result, runtimeContractType(context, resultContract), {})))
      return mlir::failure();
  }
  erase.push_back(op);
  return mlir::success();
}

} // namespace py::runtime_lowering
