#include "RuntimeLowering/RuntimeLowering.h"

#include "Common/RuntimeSupport.h"
#include "Passes/Runtime/Cleanup.h"

#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/STLExtras.h"

#include <memory>
#include <string>

namespace py::runtime_lowering {
namespace {

inline constexpr llvm::StringLiteral kOwnershipOwnedResultsAttr{
    "ly.ownership.owned_results"};
inline constexpr llvm::StringLiteral kOwnershipReleaseArgsAttr{
    "ly.ownership.release_args"};
inline constexpr llvm::StringLiteral kOwnershipTransferArgsAttr{
    "ly.ownership.transfer_args"};
inline constexpr llvm::StringLiteral kCallableTypeAttr{"callable_type"};

bool isRuntimeManifestFunction(mlir::func::FuncOp function) {
  return function->hasAttr(kManifestContractAttr) ||
         function->hasAttr(kManifestPrimitiveAttr) ||
         function->hasAttr(kManifestMethodAttr) ||
         function->hasAttr(kManifestInitializerAttr) ||
         function->hasAttr(kManifestBuiltinAttr) ||
         function->hasAttr(kManifestShapeAttr) ||
         function->hasAttr(kManifestDeallocatorAttr);
}

bool integerListContains(mlir::Attribute attr, std::int64_t value) {
  if (!attr)
    return false;
  if (auto dense = mlir::dyn_cast<mlir::DenseI64ArrayAttr>(attr))
    return llvm::is_contained(dense.asArrayRef(), value);
  if (auto array = mlir::dyn_cast<mlir::ArrayAttr>(attr)) {
    for (mlir::Attribute element : array) {
      auto integer = mlir::dyn_cast<mlir::IntegerAttr>(element);
      if (integer && integer.getInt() == value)
        return true;
    }
  }
  return false;
}

bool functionOwnsResultAt(mlir::func::FuncOp function, unsigned resultIndex) {
  return integerListContains(function->getAttr(kOwnershipOwnedResultsAttr),
                             resultIndex);
}

bool functionUsesOwnedReturnABI(mlir::func::FuncOp function) {
  if (!function || function.isExternal() || isRuntimeManifestFunction(function))
    return false;
  return function->hasAttr(kCallableTypeAttr) ||
         function.getSymName() == "__main__";
}

bool consumesOwnershipAtOperand(mlir::func::FuncOp function,
                                unsigned operandIndex) {
  return integerListContains(function->getAttr(kOwnershipReleaseArgsAttr),
                             operandIndex) ||
         integerListContains(function->getAttr(kOwnershipTransferArgsAttr),
                             operandIndex);
}

struct RuntimeDeallocator {
  mlir::func::FuncOp function;
  llvm::SmallVector<mlir::Type, 4> inputTypes;
};

llvm::SmallVector<RuntimeDeallocator, 8>
collectRuntimeDeallocators(mlir::ModuleOp module) {
  llvm::SmallVector<RuntimeDeallocator, 8> deallocators;
  module.walk([&](mlir::func::FuncOp function) {
    if (!function->hasAttr(kManifestDeallocatorAttr))
      return;
    RuntimeDeallocator deallocator;
    deallocator.function = function;
    deallocator.inputTypes.append(function.getFunctionType().getInputs().begin(),
                                  function.getFunctionType().getInputs().end());
    deallocators.push_back(std::move(deallocator));
  });
  return deallocators;
}

bool valueRangeMatchesTypes(mlir::ValueRange values, unsigned offset,
                            llvm::ArrayRef<mlir::Type> types) {
  if (offset + types.size() > values.size())
    return false;
  for (auto [index, type] : llvm::enumerate(types)) {
    if (values[offset + index].getType() != type)
      return false;
  }
  return true;
}

const RuntimeDeallocator *
findDeallocatorForValueGroup(mlir::ValueRange values, unsigned offset,
                             llvm::ArrayRef<RuntimeDeallocator> deallocators) {
  const RuntimeDeallocator *matched = nullptr;
  bool ambiguous = false;
  for (const RuntimeDeallocator &deallocator : deallocators) {
    if (!valueRangeMatchesTypes(values, offset, deallocator.inputTypes))
      continue;
    if (!matched ||
        deallocator.inputTypes.size() > matched->inputTypes.size()) {
      matched = &deallocator;
      ambiguous = false;
      continue;
    }
    if (deallocator.inputTypes.size() == matched->inputTypes.size())
      ambiguous = true;
  }
  if (ambiguous)
    return nullptr;
  return matched;
}

llvm::SmallVector<mlir::Value, 4> valueSlice(mlir::ValueRange values,
                                             unsigned offset, unsigned size) {
  llvm::SmallVector<mlir::Value, 4> slice;
  slice.reserve(size);
  for (unsigned index = 0; index < size; ++index)
    slice.push_back(values[offset + index]);
  return slice;
}

bool valueGroupEqualsEntryArgumentGroup(mlir::func::FuncOp function,
                                        llvm::ArrayRef<mlir::Value> group) {
  if (function.empty() || group.empty())
    return false;
  mlir::Block &entry = function.front();
  if (entry.getNumArguments() < group.size())
    return false;

  for (unsigned start = 0; start + group.size() <= entry.getNumArguments();
       ++start) {
    bool matches = true;
    for (auto [index, value] : llvm::enumerate(group)) {
      if (value != entry.getArgument(start + index)) {
        matches = false;
        break;
      }
    }
    if (matches)
      return true;
  }
  return false;
}

mlir::func::FuncOp findRetainFunction(mlir::ModuleOp module) {
  mlir::func::FuncOp retained;
  module.walk([&](mlir::func::FuncOp function) {
    auto primitive =
        function->getAttrOfType<mlir::StringAttr>(kManifestPrimitiveAttr);
    if (!primitive || primitive.getValue() != "retain")
      return;
    retained = function;
  });
  return retained;
}

mlir::FailureOr<mlir::Value>
buildRetainHeaderView(mlir::OpBuilder &builder, mlir::Location loc,
                      mlir::Value header, mlir::Type retainInputType) {
  if (header.getType() == retainInputType)
    return header;

  auto sourceType = mlir::dyn_cast<mlir::MemRefType>(header.getType());
  auto targetType = mlir::dyn_cast<mlir::MemRefType>(retainInputType);
  if (!sourceType || !targetType)
    return mlir::failure();
  if (sourceType.getRank() != 1 || targetType.getRank() != 1)
    return mlir::failure();
  if (sourceType.getElementType() != targetType.getElementType())
    return mlir::failure();

  if (sourceType.getDimSize(0) == targetType.getDimSize(0))
    return builder.create<mlir::memref::CastOp>(loc, retainInputType, header)
        .getResult();

  if (sourceType.hasStaticShape() && targetType.hasStaticShape() &&
      sourceType.getDimSize(0) >= targetType.getDimSize(0)) {
    llvm::SmallVector<mlir::OpFoldResult, 1> offsets{
        builder.getIndexAttr(0)};
    llvm::SmallVector<mlir::OpFoldResult, 1> sizes{
        builder.getIndexAttr(targetType.getDimSize(0))};
    llvm::SmallVector<mlir::OpFoldResult, 1> strides{
        builder.getIndexAttr(1)};
    return builder
        .create<mlir::memref::SubViewOp>(loc, targetType, header, offsets,
                                         sizes, strides)
        .getResult();
  }

  return mlir::failure();
}

mlir::LogicalResult insertRetain(mlir::func::FuncOp retain,
                                 mlir::func::ReturnOp returnOp,
                                 mlir::Value header) {
  if (!retain)
    return returnOp.emitError()
           << "borrowed object return requires a runtime retain primitive";
  if (retain.getFunctionType().getNumInputs() != 1)
    return retain.emitError()
           << "runtime retain primitive must accept one object header";

  mlir::OpBuilder builder(returnOp);
  mlir::FailureOr<mlir::Value> headerView = buildRetainHeaderView(
      builder, returnOp.getLoc(), header, retain.getFunctionType().getInput(0));
  if (mlir::failed(headerView))
    return returnOp.emitError()
           << "cannot build object header view for borrowed return retain";

  builder.create<mlir::func::CallOp>(returnOp.getLoc(), retain, *headerView);
  return mlir::success();
}

mlir::LogicalResult insertBorrowedReturnRetains(
    mlir::ModuleOp module, mlir::func::FuncOp retain,
    llvm::ArrayRef<RuntimeDeallocator> deallocators) {
  mlir::LogicalResult result = mlir::success();
  module.walk([&](mlir::func::FuncOp function) {
    if (mlir::failed(result))
      return;
    if (!functionUsesOwnedReturnABI(function))
      return;

    function.walk([&](mlir::func::ReturnOp returnOp) {
      unsigned offset = 0;
      while (offset < returnOp.getNumOperands()) {
        const RuntimeDeallocator *deallocator =
            findDeallocatorForValueGroup(returnOp.getOperands(), offset,
                                         deallocators);
        if (!deallocator) {
          ++offset;
          continue;
        }

        llvm::SmallVector<mlir::Value, 4> group =
            valueSlice(returnOp.getOperands(), offset,
                       static_cast<unsigned>(deallocator->inputTypes.size()));
        if (valueGroupEqualsEntryArgumentGroup(function, group)) {
          if (mlir::failed(insertRetain(retain, returnOp, group.front()))) {
            result = mlir::failure();
            return;
          }
        }
        offset += static_cast<unsigned>(deallocator->inputTypes.size());
      }
    });
  });
  return result;
}

bool isOwnershipConsumingUse(mlir::ModuleOp module, mlir::OpOperand &use) {
  auto call = mlir::dyn_cast<mlir::func::CallOp>(use.getOwner());
  if (!call)
    return false;
  mlir::func::FuncOp callee =
      module.lookupSymbol<mlir::func::FuncOp>(call.getCallee());
  return callee && consumesOwnershipAtOperand(
                       callee, static_cast<unsigned>(use.getOperandNumber()));
}

mlir::Operation *latestUserInBlock(mlir::Operation *lhs,
                                   mlir::Operation *rhs) {
  if (!lhs)
    return rhs;
  return lhs->isBeforeInBlock(rhs) ? rhs : lhs;
}

mlir::Operation *
findReleaseInsertionPoint(mlir::ModuleOp module, mlir::func::CallOp owner,
                          llvm::ArrayRef<mlir::Value> group) {
  mlir::Block *block = owner->getBlock();
  mlir::Operation *lastUser = nullptr;
  for (mlir::Value result : group) {
    for (mlir::OpOperand &use : result.getUses()) {
      mlir::Operation *user = use.getOwner();
      if (user == owner)
        continue;
      if (user->getBlock() != block)
        return nullptr;
      if (user->hasTrait<mlir::OpTrait::IsTerminator>())
        return nullptr;
      if (isOwnershipConsumingUse(module, use))
        return nullptr;
      lastUser = latestUserInBlock(lastUser, user);
    }
  }
  return lastUser ? lastUser : owner.getOperation();
}

bool callResultGroupIsOwned(mlir::func::FuncOp callee, unsigned resultIndex) {
  return functionOwnsResultAt(callee, resultIndex) ||
         functionUsesOwnedReturnABI(callee);
}

mlir::LogicalResult insertOwnedResultReleases(
    mlir::ModuleOp module, mlir::func::CallOp call,
    llvm::ArrayRef<RuntimeDeallocator> deallocators) {
  mlir::func::FuncOp callee =
      module.lookupSymbol<mlir::func::FuncOp>(call.getCallee());
  if (!callee || call.getNumResults() == 0)
    return mlir::success();

  unsigned offset = 0;
  while (offset < call.getNumResults()) {
    const RuntimeDeallocator *deallocator =
        findDeallocatorForValueGroup(call.getResults(), offset, deallocators);
    if (!deallocator) {
      ++offset;
      continue;
    }

    unsigned groupSize = static_cast<unsigned>(deallocator->inputTypes.size());
    if (!callResultGroupIsOwned(callee, offset)) {
      ++offset;
      continue;
    }

    llvm::SmallVector<mlir::Value, 4> group =
        valueSlice(call.getResults(), offset, groupSize);
    mlir::Operation *insertionPoint =
        findReleaseInsertionPoint(module, call, group);
    if (insertionPoint) {
      mlir::OpBuilder builder(insertionPoint);
      builder.setInsertionPointAfter(insertionPoint);
      builder.create<mlir::func::CallOp>(call.getLoc(),
                                         deallocator->function, group);
    }
    offset += groupSize;
  }
  return mlir::success();
}

class NoOpModulePass
    : public mlir::PassWrapper<NoOpModulePass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  NoOpModulePass(llvm::StringRef argument, llvm::StringRef description)
      : argument(argument.str()), description(description.str()) {}

  NoOpModulePass(const NoOpModulePass &other)
      : mlir::PassWrapper<NoOpModulePass, mlir::OperationPass<mlir::ModuleOp>>(
            other),
        argument(other.argument), description(other.description) {}

  llvm::StringRef getArgument() const final { return argument; }
  llvm::StringRef getDescription() const final { return description; }
  void runOnOperation() final {}

private:
  std::string argument;
  std::string description;
};

class RefCountInsertionPass
    : public mlir::PassWrapper<RefCountInsertionPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RefCountInsertionPass)

  llvm::StringRef getArgument() const final {
    return "lython-refcount-insertion";
  }
  llvm::StringRef getDescription() const final {
    return "insert manifest-driven releases for runtime-owned call results";
  }

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();
    llvm::SmallVector<RuntimeDeallocator, 8> deallocators =
        collectRuntimeDeallocators(module);
    if (deallocators.empty())
      return;

    mlir::func::FuncOp retain = findRetainFunction(module);
    if (mlir::failed(insertBorrowedReturnRetains(module, retain,
                                                 deallocators))) {
      signalPassFailure();
      return;
    }

    llvm::SmallVector<mlir::func::CallOp, 32> calls;
    module.walk([&](mlir::func::FuncOp function) {
      if (isRuntimeManifestFunction(function))
        return;
      function.walk([&](mlir::func::CallOp call) { calls.push_back(call); });
    });

    for (mlir::func::CallOp call : calls) {
      if (mlir::failed(insertOwnedResultReleases(module, call, deallocators))) {
        signalPassFailure();
        return;
      }
    }
  }
};

class RuntimeLoweringPass
    : public mlir::PassWrapper<RuntimeLoweringPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RuntimeLoweringPass)

  llvm::StringRef getArgument() const final {
    return "lython-runtime-lowering";
  }
  llvm::StringRef getDescription() const final {
    return "lower resolved Py dialect operations to the runtime ABI";
  }

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();
    if (mlir::failed(requireResolvedInput(module))) {
      signalPassFailure();
      return;
    }

    if (mlir::failed(RuntimeBundleLowerer(module).lowerModule()))
      signalPassFailure();
  }

private:
  mlir::LogicalResult requireResolvedInput(mlir::ModuleOp module) {
    mlir::LogicalResult result = mlir::success();
    module.walk([&](mlir::UnrealizedConversionCastOp op) {
      op.emitError()
          << "lowering requires fully resolved Python IR; "
             "builtin.unrealized_conversion_cast is resolution evidence, not "
             "a runtime ABI value";
      result = mlir::failure();
      return mlir::WalkResult::interrupt();
    });
    return result;
  }
};

class AsyncThunkLoweringPass
    : public mlir::PassWrapper<AsyncThunkLoweringPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AsyncThunkLoweringPass)

  llvm::StringRef getArgument() const final {
    return "lython-async-thunk-lowering";
  }
  llvm::StringRef getDescription() const final {
    return "lower Lython-owned async dialect thunks to the runtime ABI";
  }

  void runOnOperation() final {
    llvm::SmallVector<mlir::async::ExecuteOp, 8> executes;
    getOperation().walk([&](mlir::async::ExecuteOp execute) {
      if (execute->hasAttr("ly.async.python_await"))
        executes.push_back(execute);
    });

    for (mlir::async::ExecuteOp execute : executes) {
      if (mlir::failed(lowerPythonAwaitThunk(execute))) {
        signalPassFailure();
        return;
      }
    }
  }

private:
  static bool isCoroutineFrameMarker(mlir::Operation *op) {
    return mlir::isa<mlir::async::CoroIdOp, mlir::async::CoroBeginOp,
                     mlir::async::CoroEndOp, mlir::async::CoroFreeOp,
                     mlir::async::CoroSaveOp>(op);
  }

  static bool canEraseCoroutineFrameMarker(mlir::Operation *op) {
    return llvm::all_of(op->getResults(),
                        [](mlir::Value result) { return result.use_empty(); });
  }

  static mlir::LogicalResult eraseCoroutineFrameMarkers(
      mlir::async::ExecuteOp execute,
      llvm::SmallVectorImpl<mlir::Operation *> &markers) {
    bool changed = true;
    while (changed) {
      changed = false;
      for (auto marker = markers.begin(); marker != markers.end();) {
        if (!canEraseCoroutineFrameMarker(*marker)) {
          ++marker;
          continue;
        }
        (*marker)->erase();
        marker = markers.erase(marker);
        changed = true;
      }
    }

    if (!markers.empty())
      return execute.emitError()
             << "Lython Python-await coroutine frame marker still has uses "
                "after thunk lowering";
    return mlir::success();
  }

  mlir::LogicalResult lowerPythonAwaitThunk(mlir::async::ExecuteOp execute) {
    if (!execute.getToken().use_empty())
      return execute.emitError()
             << "Lython Python-await async thunk token must be unused";

    mlir::Block &body = execute.getBodyRegion().front();
    auto yield = mlir::dyn_cast<mlir::async::YieldOp>(body.getTerminator());
    if (!yield)
      return execute.emitError()
             << "Lython Python-await async thunk must end with async.yield";
    if (yield.getOperands().size() != execute.getBodyResults().size())
      return execute.emitError()
             << "Lython Python-await async thunk yield/result arity mismatch";

    llvm::SmallVector<mlir::async::AwaitOp, 4> awaits;
    awaits.reserve(execute.getBodyResults().size());
    mlir::Operation *firstAwait = nullptr;
    for (mlir::Value result : execute.getBodyResults()) {
      if (!result.hasOneUse())
        return execute.emitError()
               << "Lython Python-await async thunk result must have one use";
      auto await =
          mlir::dyn_cast<mlir::async::AwaitOp>(*result.getUsers().begin());
      if (!await)
        return execute.emitError()
               << "Lython Python-await async thunk result must be consumed by "
                  "async.await";
      if (await->getBlock() != execute->getBlock())
        return await.emitError()
               << "Lython Python-await async thunk await must stay in the "
                  "producer block";
      if (!firstAwait || await->isBeforeInBlock(firstAwait))
        firstAwait = await;
      awaits.push_back(await);
    }

    mlir::Operation *anchor = firstAwait ? firstAwait : execute.getOperation();
    llvm::SmallVector<mlir::Operation *, 4> coroutineFrameMarkers;
    while (!body.empty() && &body.front() != yield) {
      mlir::Operation *moved = &body.front();
      moved->moveBefore(anchor);
      if (isCoroutineFrameMarker(moved))
        coroutineFrameMarkers.push_back(moved);
    }

    if (mlir::failed(
            eraseCoroutineFrameMarkers(execute, coroutineFrameMarkers)))
      return mlir::failure();

    for (auto [index, await] : llvm::enumerate(awaits)) {
      if (await.getNumResults() != 1)
        return await.emitError()
               << "Lython Python-await async thunk expects value awaits";
      await.getResult().replaceAllUsesWith(yield.getOperand(index));
      await.erase();
    }

    execute.erase();
    return mlir::success();
  }
};

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
makeNoOpPass(llvm::StringRef argument, llvm::StringRef description) {
  return std::make_unique<NoOpModulePass>(argument, description);
}

} // namespace
} // namespace py::runtime_lowering

namespace py {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createRuntimeLoweringPass() {
  return std::make_unique<runtime_lowering::RuntimeLoweringPass>();
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createPublicationPreparationPass() {
  return runtime_lowering::makeNoOpPass(
      "lython-publication-preparation",
      "prepare a resolved module for runtime lowering");
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createRefCountInsertionPass() {
  return std::make_unique<runtime_lowering::RefCountInsertionPass>();
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createRefCountPairElisionPass() {
  return runtime_lowering::makeNoOpPass(
      "lython-refcount-elision", "elide proven redundant ownership pairs");
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createPyOptimizationPass() {
  return runtime_lowering::makeNoOpPass(
      "lython-py-optimization", "optimize resolved Py dialect operations");
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createOwnershipVerifierPass() {
  return runtime_lowering::makeNoOpPass(
      "lython-ownership-verifier", "verify ownership after runtime lowering");
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createLLVMCallOwnershipVerifierPass() {
  return runtime_lowering::makeNoOpPass(
      "lython-llvm-call-ownership-verifier",
      "verify lowered LLVM call ownership metadata");
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createLLVMThreadSafetyVerifierPass() {
  return runtime_lowering::makeNoOpPass(
      "lython-llvm-thread-safety-verifier",
      "verify lowered LLVM thread-safety metadata");
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createNativeVerificationPass() {
  return runtime_lowering::makeNoOpPass("lython-native-verification",
                                        "verify native function declarations");
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createAsyncThunkLoweringPass() {
  return std::make_unique<runtime_lowering::AsyncThunkLoweringPass>();
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createLinalgLoweringPass() {
  return runtime_lowering::makeNoOpPass("lython-linalg-lowering",
                                        "lower linalg helper operations");
}

namespace lowering::runtime::cleanup {

bool unreachableBlocks(mlir::ModuleOp) { return false; }
bool pyBridgeCasts(mlir::Operation *) { return false; }
bool pyMultiCasts(mlir::Operation *) { return false; }
bool voidPyReturns(mlir::Operation *) { return false; }
bool memrefDescriptorCasts(mlir::Operation *) { return false; }
bool memrefRuntimeCalls(mlir::Operation *) { return false; }
bool pointerRoundTrips(mlir::ModuleOp) { return false; }
bool llvmFuncReturns(mlir::Operation *) { return false; }
bool finalBoundary(mlir::ModuleOp) { return false; }

} // namespace lowering::runtime::cleanup

void collectLoweredSafetyContracts(mlir::ModuleOp, LoweredSafetyContracts &) {}

void collectLoweredSafetyContracts(mlir::ModuleOp, const PyLLVMTypeConverter &,
                                   LoweredSafetyContracts &) {}

mlir::LogicalResult
preserveLoweredSafetyContracts(mlir::ModuleOp, const LoweredSafetyContracts &) {
  return mlir::success();
}

mlir::LogicalResult verifyOwnership(mlir::ModuleOp) { return mlir::success(); }

mlir::LogicalResult verifyLLVMCallOwnership(mlir::ModuleOp) {
  return mlir::success();
}

namespace optimizer::publication {
void prepare(mlir::ModuleOp) {}
} // namespace optimizer::publication

namespace optimizer::pipeline {
void preLowering(mlir::ModuleOp) {}
void postValueLowering(mlir::ModuleOp) {}
void finalLLVMCleanup(mlir::ModuleOp) {}
} // namespace optimizer::pipeline

} // namespace py
