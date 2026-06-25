#include "RuntimeLowering/RuntimeLowering.h"

namespace py::runtime_lowering {

mlir::LogicalResult
RuntimeBundleLowerer::lowerStrConstant(py::StrConstantOp op) {
  if (isStaticKeywordName(op)) {
    erase.push_back(op);
    return mlir::success();
  }

  builder.setInsertionPoint(op);
  llvm::StringRef text = op.getValue();
  mlir::Location loc = op.getLoc();
  mlir::Value bytes = RuntimeBundleLowerer::materializeByteBuffer(loc, text);
  mlir::Value start =
      builder.create<mlir::arith::ConstantIndexOp>(loc, 0).getResult();
  mlir::Value length = builder
                           .create<mlir::arith::ConstantIntOp>(
                               loc, static_cast<std::int64_t>(text.size()), 64)
                           .getResult();
  RuntimeBundle result;
  if (mlir::failed(initializeObjectFromRawValues(
          op, op.getResult().getType(), mlir::ValueRange{bytes, start, length},
          result)))
    return mlir::failure();
  valueBundles[op.getResult()] = std::move(result);
  erase.push_back(op);
  return mlir::success();
}

bool RuntimeBundleLowerer::isStaticKeywordName(py::StrConstantOp op) const {
  if (op.getResult().use_empty())
    return false;
  for (mlir::OpOperand &use : op.getResult().getUses()) {
    auto pack = mlir::dyn_cast<py::PackOp>(use.getOwner());
    if (!pack)
      return false;
    bool feedsKeywordNames = false;
    for (mlir::OpOperand &packUse : pack.getResult().getUses()) {
      auto call = mlir::dyn_cast<py::CallOp>(packUse.getOwner());
      if (call && call.getKwnames() == pack.getResult()) {
        feedsKeywordNames = true;
        break;
      }
    }
    if (!feedsKeywordNames)
      return false;
  }
  return true;
}

mlir::LogicalResult
RuntimeBundleLowerer::lowerIntConstant(py::IntConstantOp op) {
  std::int64_t parsed = 0;
  if (op.getValue().getAsInteger(10, parsed))
    return op.emitError()
           << "integer literal is outside the currently lowered i64 path";

  builder.setInsertionPoint(op);
  mlir::Location loc = op.getLoc();
  mlir::Value value =
      builder.create<mlir::arith::ConstantIntOp>(loc, parsed, 64).getResult();
  RuntimeBundle result;
  if (mlir::failed(initializeObjectFromRawValues(
          op, op.getResult().getType(), mlir::ValueRange{value}, result)))
    return mlir::failure();
  valueBundles[op.getResult()] = std::move(result);
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult
RuntimeBundleLowerer::lowerFloatConstant(py::FloatConstantOp op) {
  builder.setInsertionPoint(op);
  mlir::Value value = builder
                          .create<mlir::arith::ConstantFloatOp>(
                              op.getLoc(), op.getValue(), builder.getF64Type())
                          .getResult();
  RuntimeBundle result;
  if (mlir::failed(initializeObjectFromRawValues(
          op, op.getResult().getType(), mlir::ValueRange{value}, result)))
    return mlir::failure();
  valueBundles[op.getResult()] = std::move(result);
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult
RuntimeBundleLowerer::lowerBoolConstant(py::BoolConstantOp op) {
  builder.setInsertionPoint(op);
  mlir::Value bit = builder
                        .create<mlir::arith::ConstantIntOp>(
                            op.getLoc(), op.getValue() ? 1 : 0, 1)
                        .getResult();
  if (mlir::failed(assignObjectBundle(
          op, op.getResult(), runtimeContractType(context, "builtins.bool"),
          bit)))
    return mlir::failure();
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerNone(py::NoneOp op) {
  if (mlir::failed(assignObjectBundle(
          op, op.getResult(), runtimeContractType(context, "types.NoneType"),
          mlir::ValueRange{})))
    return mlir::failure();
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult
RuntimeBundleLowerer::lowerCastFromPrim(py::CastFromPrimOp op) {
  std::string expected = runtimeContractName(op.getResult().getType());
  if (expected.empty())
    return op.emitError() << "primitive cast result has no runtime contract";

  builder.setInsertionPoint(op);
  llvm::SmallVector<mlir::Value, 1> inputValues{op.getInput()};
  if (objectShapeMatches(expected, inputValues)) {
    if (mlir::failed(assignObjectBundle(op, op.getResult(),
                                        runtimeContractType(context, expected),
                                        inputValues)))
      return mlir::failure();
    erase.push_back(op);
    return mlir::success();
  }

  RuntimeBundle result;
  if (mlir::succeeded(initializeObjectFromRawValues(
          op, op.getResult().getType(), inputValues, result,
          /*emitErrors=*/false))) {
    valueBundles[op.getResult()] = std::move(result);
    erase.push_back(op);
    return mlir::success();
  }

  return op.emitError() << "primitive cast to " << expected
                        << " has no manifest-driven runtime lowering yet";
}

mlir::LogicalResult RuntimeBundleLowerer::lowerTypeObject(py::TypeObjectOp op) {
  valueBundles[op.getResult()] = RuntimeBundle::typeObject(
      op.getResult().getType(), op.getInstanceContract());
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerPack(py::PackOp op) {
  valueBundles[op.getResult()] =
      RuntimeBundle::aggregate(op.getResult().getType(), op.getValues());
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerBindingRef(py::BindingRefOp op) {
  std::optional<RuntimeSymbol> builtin =
      manifest.builtinCallable(op.getBinding());
  if (builtin) {
    valueBundles[op.getResult()] = RuntimeBundle::builtinCallable(
        op.getResult().getType(), op.getBinding());
    erase.push_back(op);
    return mlir::success();
  }

  if (auto function = module.lookupSymbol<mlir::func::FuncOp>(op.getBinding()))
    return RuntimeBundleLowerer::lowerFunctionBindingRef(op, function);

  return op.emitError() << "unresolved runtime binding '" << op.getBinding()
                        << "'";
}

mlir::LogicalResult
RuntimeBundleLowerer::lowerFunctionBindingRef(py::BindingRefOp op,
                                              mlir::func::FuncOp function) {
  auto callableType = function->getAttrOfType<mlir::TypeAttr>("callable_type");
  if (!callableType)
    return op.emitError() << "runtime binding '" << op.getBinding()
                          << "' names a func.func without callable_type";
  if (!mlir::isa<py::CallableType>(callableType.getValue()))
    return op.emitError()
           << "runtime binding '" << op.getBinding()
           << "' names a func.func whose callable_type is not Callable";

  std::optional<RuntimeSymbol> initializer =
      manifest.initializer("builtins.function", "__new__");
  if (!initializer)
    return op.emitError()
           << "runtime manifest has no builtins.function.__new__";

  builder.setInsertionPoint(op);
  llvm::SmallVector<mlir::Value, 6> operands;
  operands.push_back(
      builder
          .create<mlir::arith::ConstantIntOp>(
              op.getLoc(),
              RuntimeBundleLowerer::functionTargetId(function.getSymName()), 64)
          .getResult());
  for (unsigned index = 0; index < 5; ++index)
    operands.push_back(
        builder.create<mlir::arith::ConstantIntOp>(op.getLoc(), 0, 64)
            .getResult());

  mlir::func::CallOp call = RuntimeBundleLowerer::createRuntimeCall(
      op.getLoc(), *initializer, operands);
  mlir::Type functionContract =
      runtimeContractType(context, "builtins.function");
  RuntimeBundle bundle;
  if (mlir::failed(RuntimeBundleLowerer::makeObjectBundle(
          op, functionContract, call.getResults(), bundle)))
    return mlir::failure();
  bundle.functionTarget = function.getSymName().str();
  if (mlir::failed(appendClosureValues(op, function, bundle)))
    return mlir::failure();
  valueBundles[op.getResult()] = std::move(bundle);
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::appendClosureValues(
    py::BindingRefOp op, mlir::func::FuncOp function, RuntimeBundle &bundle) {
  llvm::SmallVector<mlir::Type, 4> closureTypes =
      callableClosureTypes(function);
  if (closureTypes.size() != op.getCaptures().size())
    return op.emitError() << "binding '" << op.getBinding() << "' captures "
                          << op.getCaptures().size()
                          << " values, but target declares "
                          << closureTypes.size() << " closure inputs";

  for (auto [index, capture] : llvm::enumerate(op.getCaptures())) {
    const RuntimeBundle *captureBundle =
        RuntimeBundleLowerer::bundleFor(capture);
    if (!captureBundle || captureBundle->kind != RuntimeBundle::Kind::Object)
      return op.emitError() << "closure capture " << index
                            << " must be a lowered Python object bundle";
    if (!py::isAssignableTo(captureBundle->contract, closureTypes[index],
                            op.getOperation()))
      return op.emitError()
             << "closure capture " << index << " has type "
             << captureBundle->contract << ", expected " << closureTypes[index];
    bundle.closureValues.push_back(captureBundle->objectValue);
  }
  return mlir::success();
}

mlir::LogicalResult
RuntimeBundleLowerer::lowerAliasView(mlir::Operation *op, mlir::Value input,
                                     mlir::Value resultValue) {
  const RuntimeBundle *inputBundle = RuntimeBundleLowerer::bundleFor(input);
  if (!inputBundle)
    return op->emitError()
           << "aliasing contract view input has no lowered runtime bundle";
  valueBundles[resultValue] = *inputBundle;
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::collectPackedObjectSources(
    mlir::Operation *op, mlir::Value packValue, llvm::StringRef label,
    llvm::SmallVectorImpl<const RuntimeBundle *> &sources) const {
  const RuntimeBundle *pack = RuntimeBundleLowerer::bundleFor(packValue);
  if (!pack || pack->kind != RuntimeBundle::Kind::Aggregate)
    return op->emitError() << label << " must be a lowered aggregate bundle";
  for (mlir::Value operand : pack->aggregateOperands) {
    const RuntimeBundle *source = RuntimeBundleLowerer::bundleFor(operand);
    if (!source)
      return op->emitError()
             << label << " operand has no lowered runtime bundle";
    sources.push_back(source);
  }
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::requireEmptyAggregate(
    mlir::Operation *op, mlir::Value packValue, llvm::StringRef label) const {
  const RuntimeBundle *pack = RuntimeBundleLowerer::bundleFor(packValue);
  if (!pack || pack->kind != RuntimeBundle::Kind::Aggregate)
    return op->emitError() << label << " must be a lowered aggregate bundle";
  if (!pack->aggregateOperands.empty())
    return op->emitError() << label << " lowering is not keyword-aware yet";
  return mlir::success();
}

} // namespace py::runtime_lowering
