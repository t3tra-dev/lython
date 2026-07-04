#include "Runtime/Core/Lowerer.h"

namespace py::runtime_lowering {
namespace {

bool isPrimitiveOnlyCallableFunction(mlir::func::FuncOp function) {
  auto callableAttr = function->getAttrOfType<mlir::TypeAttr>("callable_type");
  auto callable = mlir::dyn_cast_if_present<py::CallableType>(
      callableAttr ? callableAttr.getValue() : mlir::Type());
  if (!callable || callable.hasVararg() || callable.hasKwarg())
    return false;
  auto isRuntimePrimitive = [](mlir::Type type) {
    return type && !py::isPyType(type);
  };
  return llvm::all_of(callable.getPositionalTypes(), isRuntimePrimitive) &&
         llvm::all_of(callable.getKwOnlyTypes(), isRuntimePrimitive) &&
         llvm::all_of(callable.getResultTypes(), isRuntimePrimitive);
}

} // namespace

mlir::LogicalResult RuntimeBundleLowerer::bundleRuntimeResults(
    mlir::Operation *op, mlir::Type expectedContract, mlir::func::CallOp call,
    RuntimeBundle &result) {
  return bundleRuntimeResults(op, expectedContract, call.getResults(), result);
}

mlir::LogicalResult RuntimeBundleLowerer::bundleRuntimeResults(
    mlir::Operation *op, mlir::Type expectedContract, mlir::ValueRange values,
    RuntimeBundle &result) {
  std::string expected = runtimeContractName(expectedContract);
  mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> expectedTypes =
      RuntimeBundleLowerer::runtimeValueTypesFor(op, expectedContract,
                                                 "runtime call result ABI");
  if (mlir::failed(expectedTypes))
    return mlir::failure();

  bool exact = values.size() == expectedTypes->size();
  if (exact) {
    for (auto [value, expectedType] : llvm::zip(values, *expectedTypes)) {
      if (value.getType() == expectedType)
        continue;
      exact = false;
      break;
    }
  }
  if (exact)
    return RuntimeBundleLowerer::makeObjectBundle(op, expectedContract, values,
                                                  result);
  if (!expected.empty() &&
      mlir::succeeded(initializeObjectFromRawValues(
          op, runtimeContractType(context, expected), values, result,
          /*emitErrors=*/false)))
    return mlir::success();
  return RuntimeBundleLowerer::makeObjectBundle(op, expectedContract, values,
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
  emitTryCallSiteMarkerIfNeeded(loc);
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
    if (isPrimitiveOnlyCallableFunction(function))
      return mlir::WalkResult::advance();

    auto returnedCoroutine =
        returnedCoroutineSummaries.find(function.getSymName());
    auto returnedObjectEvidence =
        returnedObjectEvidenceSummaries.find(function.getSymName());
    mlir::FunctionType functionType = function.getFunctionType();
    llvm::SmallVector<mlir::Value, 8> operands;
    unsigned resultIndex = 0;
    unsigned logicalResultIndex = 0;
    builder.setInsertionPoint(op);
    if (RuntimeBundleLowerer::isPrimitiveI64CallableClone(function)) {
      for (mlir::Value operand : op.getOperands()) {
        if (mlir::failed(
                RuntimeBundleLowerer::ensureValueBundle(op, operand))) {
          result = mlir::failure();
          return mlir::WalkResult::interrupt();
        }
        const RuntimeBundle *bundle = RuntimeBundleLowerer::bundleFor(operand);
        if (!bundle || bundle->kind != RuntimeBundle::Kind::Object ||
            bundle->contractName() != "builtins.int") {
          op.emitError()
              << "primitive i64 callable clone return needs builtins.int";
          result = mlir::failure();
          return mlir::WalkResult::interrupt();
        }
        if (resultIndex + 2 > functionType.getNumResults() ||
            !functionType.getResult(resultIndex).isInteger(64) ||
            !functionType.getResult(resultIndex + 1).isInteger(1)) {
          op.emitError()
              << "primitive i64 callable clone has malformed return ABI";
          result = mlir::failure();
          return mlir::WalkResult::interrupt();
        }
        if (bundle->primitiveI64) {
          operands.push_back(bundle->primitiveI64->value);
          operands.push_back(bundle->primitiveI64->valid);
        } else {
          operands.push_back(
              builder.create<mlir::arith::ConstantIntOp>(op.getLoc(), 0, 64)
                  .getResult());
          operands.push_back(
              builder.create<mlir::arith::ConstantIntOp>(op.getLoc(), 0, 1)
                  .getResult());
        }
        resultIndex += 2;
      }
      if (resultIndex != functionType.getNumResults()) {
        op.emitError() << "primitive i64 callable clone return ABI expected "
                       << functionType.getNumResults()
                       << " physical values, but lowering produced "
                       << resultIndex;
        result = mlir::failure();
        return mlir::WalkResult::interrupt();
      }
      builder.create<mlir::func::ReturnOp>(op.getLoc(), operands);
      erase.push_back(op);
      return mlir::WalkResult::advance();
    }
    auto appendPrimitiveReturnEvidence =
        [&](const RuntimeBundle &bundle) -> mlir::LogicalResult {
      if (bundle.contractName() != "builtins.int" ||
          resultIndex + 2 > functionType.getNumResults() ||
          !functionType.getResult(resultIndex).isInteger(64) ||
          !functionType.getResult(resultIndex + 1).isInteger(1))
        return mlir::success();
      if (bundle.primitiveI64) {
        operands.push_back(bundle.primitiveI64->value);
        operands.push_back(bundle.primitiveI64->valid);
      } else {
        operands.push_back(
            builder.create<mlir::arith::ConstantIntOp>(op.getLoc(), 0, 64)
                .getResult());
        operands.push_back(
            builder.create<mlir::arith::ConstantIntOp>(op.getLoc(), 0, 1)
                .getResult());
      }
      resultIndex += 2;
      return mlir::success();
    };
    auto appendReturnObject =
        [&](const RuntimeBundle &bundle,
            llvm::StringRef label) -> mlir::LogicalResult {
      llvm::ArrayRef<mlir::Value> values = bundle.physicalValues();
      std::optional<RuntimeValue> materializedObject;
      if (values.empty() &&
          RuntimeBundleLowerer::hasLazyPrimitiveI64Object(bundle)) {
        mlir::FailureOr<RuntimeValue> value =
            RuntimeBundleLowerer::materializePrimitiveI64Object(op, bundle);
        if (mlir::failed(value))
          return mlir::failure();
        materializedObject = std::move(*value);
        values = materializedObject->values;
      }
      if (resultIndex + values.size() <= functionType.getNumResults()) {
        bool exact = true;
        for (auto [offset, value] : llvm::enumerate(values)) {
          if (value.getType() != functionType.getResult(resultIndex + offset)) {
            exact = false;
            break;
          }
        }
        if (exact) {
          operands.append(values.begin(), values.end());
          resultIndex += static_cast<unsigned>(values.size());
          return appendPrimitiveReturnEvidence(bundle);
        }
      }

      if (resultIndex < functionType.getNumResults() &&
          bundle.kind == RuntimeBundle::Kind::Object &&
          isBuiltinsObjectHeaderType(functionType.getResult(resultIndex))) {
        const RuntimeValue &objectValue =
            materializedObject ? *materializedObject : bundle.objectValue;
        mlir::FailureOr<mlir::Value> header =
            RuntimeBundleLowerer::objectHeaderView(op, objectValue);
        if (mlir::failed(header))
          return mlir::failure();
        operands.push_back(*header);
        ++resultIndex;
        return appendPrimitiveReturnEvidence(bundle);
      }

      return op.emitError() << "cannot adapt " << bundle.contractName() << " "
                            << label << " to callable return ABI "
                            << resultIndex << " of " << function.getSymName();
    };

    for (mlir::Value operand : op.getOperands()) {
      if (mlir::failed(RuntimeBundleLowerer::ensureValueBundle(op, operand))) {
        result = mlir::failure();
        return mlir::WalkResult::interrupt();
      }
      const RuntimeBundle *bundle = RuntimeBundleLowerer::bundleFor(operand);
      if (!bundle) {
        op.emitError() << "callable function return value has no lowered "
                          "runtime bundle";
        result = mlir::failure();
        return mlir::WalkResult::interrupt();
      }
      if (mlir::failed(appendReturnObject(*bundle, "return value"))) {
        result = mlir::failure();
        return mlir::WalkResult::interrupt();
      }
      if (returnedCoroutine != returnedCoroutineSummaries.end() &&
          runtimeContractName(operand.getType()) == "types.CoroutineType") {
        if (bundle->coroutineTarget != returnedCoroutine->second.target) {
          op.emitError()
              << "returned coroutine target evidence does not match function "
                 "ABI summary";
          result = mlir::failure();
          return mlir::WalkResult::interrupt();
        }
        if (bundle->coroutineSources.size() !=
            returnedCoroutine->second.sourceContracts.size()) {
          op.emitError() << "returned coroutine frame source count does not "
                            "match function ABI summary";
          result = mlir::failure();
          return mlir::WalkResult::interrupt();
        }
        for (auto [index, source] : llvm::enumerate(bundle->coroutineSources)) {
          mlir::Type expected =
              returnedCoroutine->second.sourceContracts[index];
          if (runtimeContractName(source.contract) !=
              runtimeContractName(expected)) {
            op.emitError() << "returned coroutine frame source " << index
                           << " has contract " << source.contract
                           << ", expected " << expected;
            result = mlir::failure();
            return mlir::WalkResult::interrupt();
          }
          RuntimeBundle sourceBundle =
              RuntimeBundle::object(source.contract, source.values);
          if (mlir::failed(
                  appendReturnObject(sourceBundle, "coroutine frame source"))) {
            result = mlir::failure();
            return mlir::WalkResult::interrupt();
          }
        }
      }
      if (returnedObjectEvidence != returnedObjectEvidenceSummaries.end() &&
          returnedObjectEvidence->second.resultIndex == logicalResultIndex) {
        for (llvm::StringRef flag : returnedObjectEvidence->second.flags) {
          if (!bundle->objectEvidence.hasFlag(flag)) {
            op.emitError() << "returned object evidence for "
                           << function.getSymName() << " is missing flag '"
                           << flag << "'";
            result = mlir::failure();
            return mlir::WalkResult::interrupt();
          }
        }
        for (const ReturnedObjectEvidenceSlot &slot :
             returnedObjectEvidence->second.slots) {
          const RuntimeValue *value = bundle->objectEvidence.slot(slot.name);
          if (!value) {
            op.emitError() << "returned object evidence for "
                           << function.getSymName() << " is missing slot '"
                           << slot.name << "'";
            result = mlir::failure();
            return mlir::WalkResult::interrupt();
          }
          if (runtimeContractName(value->contract) !=
              runtimeContractName(slot.sourceContract)) {
            op.emitError() << "returned object evidence slot '" << slot.name
                           << "' has contract " << value->contract
                           << ", expected " << slot.sourceContract;
            result = mlir::failure();
            return mlir::WalkResult::interrupt();
          }
          RuntimeBundle valueBundle =
              RuntimeBundle::object(value->contract, value->values);
          if (mlir::failed(appendReturnObject(
                  valueBundle, "returned object evidence slot"))) {
            result = mlir::failure();
            return mlir::WalkResult::interrupt();
          }
        }
      }
      ++logicalResultIndex;
    }
    if (resultIndex != functionType.getNumResults()) {
      op.emitError() << "callable return ABI expected "
                     << functionType.getNumResults()
                     << " physical values, but lowering produced "
                     << resultIndex;
      result = mlir::failure();
      return mlir::WalkResult::interrupt();
    }

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
