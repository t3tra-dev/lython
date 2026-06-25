#include "RuntimeLowering/RuntimeLowering.h"

namespace py::runtime_lowering {

mlir::LogicalResult RuntimeBundleLowerer::bundleRuntimeResults(
    mlir::Operation *op, mlir::Type expectedContract, mlir::func::CallOp call,
    RuntimeBundle &result) {
  std::string expected = runtimeContractName(expectedContract);
  if (expected.empty())
    return op->emitError() << "runtime call result has no concrete contract";

  mlir::Type contract = runtimeContractType(context, expected);
  if (objectShapeMatches(expected, call.getResults()))
    return RuntimeBundleLowerer::makeObjectBundle(op, contract,
                                                  call.getResults(), result);
  if (mlir::succeeded(initializeObjectFromRawValues(
          op, contract, call.getResults(), result, /*emitErrors=*/false)))
    return mlir::success();
  return RuntimeBundleLowerer::makeObjectBundle(op, contract, call.getResults(),
                                                result);
}

const RuntimeBundle *RuntimeBundleLowerer::bundleFor(mlir::Value value) const {
  auto found = valueBundles.find(value);
  if (found == valueBundles.end())
    return nullptr;
  return &found->second;
}

mlir::Value RuntimeBundleLowerer::materializeByteBuffer(mlir::Location loc,
                                                        llvm::StringRef text) {
  mlir::Value dynamicSize = builder
                                .create<mlir::arith::ConstantIndexOp>(
                                    loc, static_cast<std::int64_t>(text.size()))
                                .getResult();
  auto memrefType =
      mlir::MemRefType::get({mlir::ShapedType::kDynamic}, builder.getI8Type());
  mlir::Value buffer = builder
                           .create<mlir::memref::AllocaOp>(
                               loc, memrefType, mlir::ValueRange{dynamicSize})
                           .getResult();
  for (auto [index, byte] : llvm::enumerate(text.bytes())) {
    mlir::Value position = builder
                               .create<mlir::arith::ConstantIndexOp>(
                                   loc, static_cast<std::int64_t>(index))
                               .getResult();
    mlir::Value value = builder
                            .create<mlir::arith::ConstantIntOp>(
                                loc, static_cast<std::int64_t>(byte), 8)
                            .getResult();
    builder.create<mlir::memref::StoreOp>(loc, value, buffer,
                                          mlir::ValueRange{position});
  }
  return buffer;
}

mlir::func::CallOp
RuntimeBundleLowerer::createRuntimeCall(mlir::Location loc,
                                        const RuntimeSymbol &symbol,
                                        mlir::ValueRange operands) {
  mlir::func::FuncOp function = symbol.function;
  return builder.create<mlir::func::CallOp>(
      loc, function.getSymName(), function.getFunctionType().getResults(),
      operands);
}

std::int64_t RuntimeBundleLowerer::functionTargetId(llvm::StringRef target) {
  auto inserted = functionTargetIds.try_emplace(target, nextFunctionTargetId);
  if (inserted.second)
    ++nextFunctionTargetId;
  return inserted.first->second;
}

mlir::LogicalResult RuntimeBundleLowerer::lowerFunctionReturns() {
  mlir::LogicalResult result = mlir::success();
  module.walk([&](mlir::func::ReturnOp op) {
    auto function = op->getParentOfType<mlir::func::FuncOp>();
    if (!function || !function->hasAttr("callable_type"))
      return mlir::WalkResult::advance();

    llvm::SmallVector<mlir::Value, 8> operands;
    for (mlir::Value operand : op.getOperands()) {
      const RuntimeBundle *bundle = RuntimeBundleLowerer::bundleFor(operand);
      if (!bundle) {
        op.emitError() << "callable function return value has no lowered "
                          "runtime bundle";
        result = mlir::failure();
        return mlir::WalkResult::interrupt();
      }
      llvm::ArrayRef<mlir::Value> values = bundle->physicalValues();
      operands.append(values.begin(), values.end());
    }

    builder.setInsertionPoint(op);
    builder.create<mlir::func::ReturnOp>(op.getLoc(), operands);
    erase.push_back(op);
    return mlir::WalkResult::advance();
  });
  return result;
}

mlir::LogicalResult RuntimeBundleLowerer::eraseCallableLogicalEntryArgs() {
  for (CallableLogicalEntryArgs entryArgs : callableLogicalEntryArgCounts) {
    if (entryArgs.function.isDeclaration() || entryArgs.count == 0)
      continue;
    mlir::Block &entry = entryArgs.function.getBody().front();
    for (unsigned index = 0; index < entryArgs.count; ++index) {
      mlir::BlockArgument argument = entry.getArgument(0);
      if (!argument.use_empty())
        return entryArgs.function.emitError()
               << "callable logical entry argument still has users after "
                  "runtime lowering";
      entry.eraseArgument(0);
    }
  }
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::eraseLoweredPyOps() {
  for (mlir::Operation *op : llvm::reverse(erase)) {
    for (mlir::Value result : op->getResults()) {
      if (!result.use_empty())
        return op->emitError()
               << "lowered Py value still has non-lowered users";
    }
    op->erase();
  }
  return mlir::success();
}

} // namespace py::runtime_lowering
