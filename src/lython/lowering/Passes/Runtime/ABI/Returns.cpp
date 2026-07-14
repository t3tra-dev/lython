#include "Runtime/Core/Lowerer.h"

namespace py::lowering {
namespace {

bool isPrimitiveOnlyCallableFunction(mlir::func::FuncOp function) {
  py::CallableType callable = callableTypeOf(function);
  if (!callable || callable.hasVararg() || callable.hasKwarg())
    return false;
  auto isRuntimePrimitive = [](mlir::Type type) {
    return type && !py::isPyType(type);
  };
  return llvm::all_of(callable.getPositionalTypes(), isRuntimePrimitive) &&
         llvm::all_of(callable.getKwOnlyTypes(), isRuntimePrimitive) &&
         llvm::all_of(callable.getResultTypes(), isRuntimePrimitive);
}

bool isErasedObjectResult(mlir::Type type) {
  return runtimeContractName(type) == "builtins.object";
}

mlir::Value integerConstant(mlir::OpBuilder &builder, mlir::Location loc,
                            mlir::IntegerType type, std::int64_t value) {
  return mlir::arith::ConstantOp::create(
             builder, loc, builder.getIntegerAttr(type, value))
      .getResult();
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
  if (RuntimeBundleLowerer::hasPrimitiveI64ABI(expectedContract) &&
      values.size() == expectedTypes->size() + 2 &&
      values[expectedTypes->size()].getType().isInteger(64) &&
      values[expectedTypes->size() + 1].getType().isInteger(1)) {
    llvm::SmallVector<mlir::Value, 4> objectValues;
    objectValues.reserve(expectedTypes->size());
    for (unsigned index = 0, end = static_cast<unsigned>(expectedTypes->size());
         index < end; ++index)
      objectValues.push_back(values[index]);
    if (mlir::failed(RuntimeBundleLowerer::makeObjectBundle(
            op, expectedContract, objectValues, result)))
      return mlir::failure();
    result.primitiveI64 = RuntimePrimitiveI64Evidence{
        values[expectedTypes->size()], values[expectedTypes->size() + 1]};
    return mlir::success();
  }
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
  mlir::Value dynamicSize =
      mlir::arith::ConstantIndexOp::create(
          builder, loc, static_cast<std::int64_t>(text.size()))
          .getResult();
  auto memrefType =
      mlir::MemRefType::get({mlir::ShapedType::kDynamic}, builder.getI8Type());
  mlir::Value buffer =
      mlir::memref::AllocaOp::create(builder, loc, memrefType,
                                     mlir::ValueRange{dynamicSize})
          .getResult();
  for (auto [index, byte] : llvm::enumerate(text.bytes())) {
    mlir::Value position = mlir::arith::ConstantIndexOp::create(
                               builder, loc, static_cast<std::int64_t>(index))
                               .getResult();
    mlir::Value value = mlir::arith::ConstantIntOp::create(
                            builder, loc, static_cast<std::int64_t>(byte), 8)
                            .getResult();
    mlir::memref::StoreOp::create(builder, loc, value, buffer,
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
  return mlir::func::CallOp::create(builder, loc, function.getSymName(),
                                    function.getFunctionType().getResults(),
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
    if (RuntimeBundleLowerer::isCallableProtocolTemplate(function))
      return mlir::WalkResult::advance();
    if (isPrimitiveOnlyCallableFunction(function))
      return mlir::WalkResult::advance();

    py::CallableType callable = callableTypeOf(function);
    llvm::SmallVector<mlir::Type, 4> logicalResultTypes;
    if (callable)
      logicalResultTypes.append(callable.getResultTypes().begin(),
                                callable.getResultTypes().end());

    auto returnedCoroutine =
        returnedCoroutineSummaries.find(function.getSymName());
    auto returnedObjectEvidence =
        returnedObjectEvidenceSummaries.find(function.getSymName());
    auto returnedStaticObject =
        returnedStaticObjectSummaries.find(function.getSymName());
    mlir::FunctionType functionType = function.getFunctionType();
    llvm::SmallVector<mlir::Value, 8> operands;
    unsigned resultIndex = 0;
    unsigned logicalResultIndex = 0;
    builder.setInsertionPoint(op);
    if (RuntimeBundleLowerer::isPrimitiveI64CallableClone(function)) {
      auto releaseReturnedObjectIfOwned =
          [&](const RuntimeBundle &bundle) -> mlir::LogicalResult {
        if (bundle.kind != RuntimeBundle::Kind::Object ||
            bundle.objectValue.ownership != ownership::OwnershipKind::Own ||
            bundle.physicalValues().empty())
          return mlir::success();
        return RuntimeBundleLowerer::releaseAggregateSlot(
            op, bundle, "primitive_i64_return");
      };
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
          operands.push_back(integerConstant(
              builder, op.getLoc(), builder.getI64Type(), 0));
          operands.push_back(integerConstant(
              builder, op.getLoc(), builder.getI1Type(), 0));
        }
        if (mlir::failed(releaseReturnedObjectIfOwned(*bundle))) {
          result = mlir::failure();
          return mlir::WalkResult::interrupt();
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
      // Operand bundling may lower block arguments, which rewrites (erases)
      // predecessor terminators and dangles the saved insertion iterator:
      // re-anchor at the return being replaced.
      builder.setInsertionPoint(op);
      mlir::func::ReturnOp::create(builder, op.getLoc(), operands);
      op.erase();
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
            integerConstant(builder, op.getLoc(), builder.getI64Type(), 0));
        operands.push_back(
            integerConstant(builder, op.getLoc(), builder.getI1Type(), 0));
      }
      resultIndex += 2;
      return mlir::success();
    };
    auto appendReturnObject =
        [&](const RuntimeBundle &bundle, llvm::StringRef label,
            mlir::Type expectedLogicalType) -> mlir::LogicalResult {
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
      if (isErasedObjectResult(expectedLogicalType)) {
        if (bundle.contractName() == "builtins.object" &&
            resultIndex + values.size() <= functionType.getNumResults()) {
          bool exactBox = !values.empty();
          for (auto [offset, value] : llvm::enumerate(values)) {
            if (value.getType() !=
                functionType.getResult(resultIndex + offset)) {
              exactBox = false;
              break;
            }
          }
          if (exactBox) {
            operands.append(values.begin(), values.end());
            resultIndex += static_cast<unsigned>(values.size());
            return mlir::success();
          }
        }
        RuntimeBundle sourceForBox = bundle;
        if (materializedObject) {
          sourceForBox.objectValue = *materializedObject;
          sourceForBox.contract = materializedObject->contract;
        }
        mlir::FailureOr<RuntimeBundle> boxed =
            RuntimeBundleLowerer::boxRuntimeObject(op, sourceForBox,
                                                   /*retainPayload=*/true);
        if (mlir::failed(boxed))
          return mlir::failure();
        llvm::ArrayRef<mlir::Value> boxedValues = boxed->physicalValues();
        if (resultIndex + boxedValues.size() > functionType.getNumResults())
          return op.emitError()
                 << "callable function '" << function.getSymName()
                 << "' boxed object return exceeds ABI result list";
        for (auto [offset, value] : llvm::enumerate(boxedValues)) {
          mlir::Type expected = functionType.getResult(resultIndex + offset);
          if (value.getType() != expected)
            return op.emitError()
                   << "callable function '" << function.getSymName()
                   << "' boxed object return has type " << value.getType()
                   << ", but ABI expects " << expected;
        }
        operands.append(boxedValues.begin(), boxedValues.end());
        resultIndex += static_cast<unsigned>(boxedValues.size());
        return mlir::success();
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
          isBuiltinsObjectHandleType(functionType.getResult(resultIndex))) {
        if (!RuntimeBundleLowerer::isBuiltinsObjectContract(bundle.contract)) {
          mlir::FailureOr<RuntimeBundle> boxed =
              RuntimeBundleLowerer::boxRuntimeObject(op, bundle,
                                                     /*retainPayload=*/true);
          if (mlir::failed(boxed))
            return mlir::failure();
          llvm::ArrayRef<mlir::Value> values = boxed->physicalValues();
          if (values.size() == 1 &&
              values.front().getType() == functionType.getResult(resultIndex)) {
            operands.push_back(values.front());
            ++resultIndex;
            return appendPrimitiveReturnEvidence(bundle);
          }
          return op.emitError()
                 << "boxed return value for " << bundle.contractName()
                 << " does not match callable return ABI " << resultIndex
                 << " of " << function.getSymName();
        }
        if (values.empty())
          return op.emitError()
                 << "builtins.object return value has no boxed handle";
        if (values.front().getType() != functionType.getResult(resultIndex))
          return op.emitError()
                 << "builtins.object return handle " << values.front().getType()
                 << " does not match callable return ABI " << resultIndex
                 << " of " << function.getSymName();
        operands.push_back(values.front());
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
      builder.setInsertionPoint(op);
      const RuntimeBundle *bundle = RuntimeBundleLowerer::bundleFor(operand);
      if (!bundle) {
        op.emitError() << "callable function return value has no lowered "
                          "runtime bundle";
        result = mlir::failure();
        return mlir::WalkResult::interrupt();
      }
      mlir::Type logicalResultType =
          logicalResultIndex < logicalResultTypes.size()
              ? logicalResultTypes[logicalResultIndex]
              : operand.getType();
      if (mlir::failed(
              appendReturnObject(*bundle, "return value", logicalResultType))) {
        result = mlir::failure();
        return mlir::WalkResult::interrupt();
      }
      if (returnedStaticObject != returnedStaticObjectSummaries.end() &&
          returnedStaticObject->second.resultIndex == logicalResultIndex) {
        mlir::Type objectContract = returnedStaticObject->second.objectContract;
        const RuntimeBundle *staticObject = nullptr;
        if (bundle->kind == RuntimeBundle::Kind::Object &&
            py::isAssignableTo(bundle->objectValue.contract, objectContract,
                               op.getOperation())) {
          staticObject = bundle;
        } else if (bundle->boxedObject &&
                   py::isAssignableTo(bundle->boxedObject->objectValue.contract,
                                      objectContract, op.getOperation())) {
          staticObject = bundle->boxedObject.get();
        } else if (const RuntimeBundle *member =
                       bundle->unionActiveMember.get()) {
          // Union-wrapped concrete return: the active member holds the owned
          // token, so the owned evidence lane must alias its values. Inactive
          // branches keep immortal dead placeholders, whose release stays a
          // no-op in the caller.
          if (member->kind == RuntimeBundle::Kind::Object &&
              !py::isPyNoneType(member->objectValue.contract) &&
              py::isAssignableTo(member->objectValue.contract, objectContract,
                                 op.getOperation())) {
            staticObject = member;
          } else if (member->boxedObject &&
                     !py::isPyNoneType(
                         member->boxedObject->objectValue.contract) &&
                     py::isAssignableTo(
                         member->boxedObject->objectValue.contract,
                         objectContract, op.getOperation())) {
            staticObject = member->boxedObject.get();
          }
        }
        RuntimeBundle deadStaticObject;
        if (!staticObject) {
          mlir::FailureOr<RuntimeValue> dead =
              RuntimeBundleLowerer::materializeNonOwningDeadObjectValue(
                  op.getOperation(), objectContract,
                  "inactive static returned object evidence");
          if (mlir::failed(dead)) {
            result = mlir::failure();
            return mlir::WalkResult::interrupt();
          }
          deadStaticObject = RuntimeBundle::objectWithOwnership(
              objectContract, dead->values, dead->ownership);
          staticObject = &deadStaticObject;
        }
        if (mlir::failed(appendReturnObject(*staticObject,
                                            "static returned object evidence",
                                            objectContract))) {
          result = mlir::failure();
          return mlir::WalkResult::interrupt();
        }
      }
      if (returnedCoroutine != returnedCoroutineSummaries.end() &&
          bundle->kind == RuntimeBundle::Kind::Object &&
          !bundle->coroutineTarget.empty() &&
          (isCoroutineLikeResultType(logicalResultType) ||
           isAwaitIteratorLikeResultType(logicalResultType))) {
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
              index < bundle->coroutineSourceBundles.size() &&
                      bundle->coroutineSourceBundles[index]
                  ? *bundle->coroutineSourceBundles[index]
                  : RuntimeBundle::object(source.contract, source.values);
          sourceBundle.contract = source.contract;
          sourceBundle.objectValue = source;
          if (mlir::failed(appendReturnObject(
                  sourceBundle, "coroutine frame source", expected))) {
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
          if (mlir::failed(appendReturnObject(valueBundle,
                                              "returned object evidence slot",
                                              slot.sourceContract))) {
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

    // Same re-anchor as the primitive branch: operand bundling can rewrite
    // predecessor terminators and dangle the saved insertion iterator.
    builder.setInsertionPoint(op);
    mlir::func::ReturnOp::create(builder, op.getLoc(), operands);
    op.erase();
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

// A py.try result lane that the program never reads leaves a py-typed
// continuation block argument with no users; nothing registers it with the
// logical-argument expansion, so the forwarding branch operands (often a
// py.class.upcast emitted only for the lane) would survive to the final
// erase check. Drop such arguments and their forwarding operands, to a
// fixpoint (dropping a forward can make the forwarded value dead too).
mlir::LogicalResult RuntimeBundleLowerer::dropUnusedLogicalBlockArguments() {
  // Arguments already registered with the logical expansion are erased by
  // eraseControlFlowLogicalBlockArguments (by saved handle) -- touching them
  // here would double-erase.
  llvm::SmallPtrSet<void *, 16> registered;
  for (const ControlFlowLogicalBlockArgumentABI &abi :
       controlFlowLogicalBlockArguments)
    registered.insert(abi.argument.getAsOpaquePointer());
  bool changed = true;
  while (changed) {
    changed = false;
    llvm::SmallVector<mlir::BlockArgument, 8> unused;
    module.walk([&](mlir::func::FuncOp function) {
      if (function.isDeclaration())
        return;
      for (mlir::Block &block : function.getBody()) {
        if (block.isEntryBlock())
          continue; // entry args follow the callable ABI machinery
        for (mlir::BlockArgument argument : block.getArguments())
          if (argument.use_empty() &&
              argument.getType().getDialect().getNamespace() == "py" &&
              !registered.contains(argument.getAsOpaquePointer()))
            unused.push_back(argument);
      }
    });
    for (mlir::BlockArgument argument : unused) {
      mlir::Block *block = argument.getOwner();
      if (argument.getArgNumber() >= block->getNumArguments() ||
          block->getArgument(argument.getArgNumber()) != argument)
        continue; // shifted by an earlier erase this round; retry next pass
      unsigned index = argument.getArgNumber();
      // One rewrite per predecessor terminator handles ALL its edges into
      // the block (a dual-edge cond_br lists its predecessor twice).
      llvm::SmallPtrSet<mlir::Operation *, 4> seen;
      bool droppable = true;
      llvm::SmallVector<mlir::Operation *, 4> terminators;
      for (mlir::Block *predecessor : block->getPredecessors()) {
        mlir::Operation *terminator = predecessor->getTerminator();
        if (!seen.insert(terminator).second)
          continue;
        if (!mlir::isa<mlir::cf::BranchOp, mlir::cf::CondBranchOp>(terminator)) {
          droppable = false;
          break;
        }
        terminators.push_back(terminator);
      }
      if (!droppable)
        continue;
      for (mlir::Operation *terminator : terminators) {
        mlir::OpBuilder rebuild(terminator);
        if (auto branch = mlir::dyn_cast<mlir::cf::BranchOp>(terminator)) {
          llvm::SmallVector<mlir::Value, 8> operands(
              branch.getDestOperands().begin(), branch.getDestOperands().end());
          operands.erase(operands.begin() + index);
          mlir::cf::BranchOp::create(rebuild, branch.getLoc(),
                                     branch.getDest(), operands);
          branch.erase();
          continue;
        }
        auto cond = mlir::cast<mlir::cf::CondBranchOp>(terminator);
        llvm::SmallVector<mlir::Value, 8> trueOperands(
            cond.getTrueDestOperands().begin(),
            cond.getTrueDestOperands().end());
        llvm::SmallVector<mlir::Value, 8> falseOperands(
            cond.getFalseDestOperands().begin(),
            cond.getFalseDestOperands().end());
        if (cond.getTrueDest() == block)
          trueOperands.erase(trueOperands.begin() + index);
        if (cond.getFalseDest() == block)
          falseOperands.erase(falseOperands.begin() + index);
        mlir::cf::CondBranchOp::create(rebuild, cond.getLoc(),
                                       cond.getCondition(), cond.getTrueDest(),
                                       trueOperands, cond.getFalseDest(),
                                       falseOperands);
        cond.erase();
      }
      block->eraseArgument(index);
      changed = true;
    }
  }
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::eraseLoweredPyOps() {
  llvm::DenseSet<mlir::Operation *> erased;
  for (mlir::Operation *op : llvm::reverse(erase)) {
    if (!erased.insert(op).second)
      continue;
    // Class metadata (method symbols, source-class ids) feeds the boxed-method
    // hooks generated at the end of the pass; lowerModule erases the class ops
    // after the hooks are built.
    if (mlir::isa<py::ClassOp>(op))
      continue;
    for (auto [resultIndex, result] : llvm::enumerate(op->getResults())) {
      if (!result.use_empty()) {
        mlir::InFlightDiagnostic diagnostic =
            op->emitError()
            << "lowered Py value still has non-lowered users for "
            << op->getName() << " result #" << resultIndex << " : "
            << result.getType();
        for (mlir::Operation *user : result.getUsers())
          diagnostic << " user=" << user->getName();
        return mlir::failure();
      }
    }
    op->erase();
  }
  return mlir::success();
}

} // namespace py::lowering
