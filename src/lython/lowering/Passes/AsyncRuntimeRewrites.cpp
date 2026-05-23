#include "Common/AsyncSafetyKernel.h"
#include "Common/RuntimeSupport.h"

#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

namespace py {
namespace {

constexpr llvm::StringLiteral kAsyncExceptionEdgeMarker =
    "__lython_async_exception_edge_marker";
constexpr llvm::StringLiteral kAsyncTaskCancelMarker =
    "__lython_async_task_cancel_marker";

namespace runtime_func {

mlir::LLVM::LLVMFuncOp getOrInsert(mlir::ModuleOp module, mlir::Location loc,
                                   mlir::OpBuilder &builder,
                                   llvm::StringRef name, mlir::Type resultType,
                                   llvm::ArrayRef<mlir::Type> argTypes) {
  if (auto fn = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(name))
    return fn;

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());
  auto fnType = mlir::LLVM::LLVMFunctionType::get(resultType, argTypes, false);
  return builder.create<mlir::LLVM::LLVMFuncOp>(loc, name, fnType);
}

} // namespace runtime_func

namespace cfedge {

mlir::Value condition(mlir::Operation *branch) {
  if (auto cfBranch = mlir::dyn_cast<mlir::cf::CondBranchOp>(branch))
    return cfBranch.getCondition();
  if (auto llvmBranch = mlir::dyn_cast<mlir::LLVM::CondBrOp>(branch))
    return llvmBranch.getCondition();
  return {};
}

} // namespace cfedge

namespace value_type {

bool pointerLike(mlir::Type type) {
  return mlir::isa<mlir::LLVM::LLVMPointerType>(type);
}

} // namespace value_type

namespace cancel_marker {

mlir::Value awaitedHandle(mlir::Operation *marker) {
  for (mlir::Operation *cursor = marker ? marker->getPrevNode() : nullptr;
       cursor; cursor = cursor->getPrevNode()) {
    if (cursor->hasTrait<mlir::OpTrait::IsTerminator>())
      return {};
    if (auto load = mlir::dyn_cast<mlir::async::RuntimeLoadOp>(cursor))
      return load.getOperand();
    if (auto load = mlir::dyn_cast<mlir::LLVM::LoadOp>(cursor)) {
      if (async_runtime::ValueStorage::isAddress(load.getAddr())) {
        if (auto storage = load.getAddr().getDefiningOp<mlir::LLVM::CallOp>()) {
          auto callee = storage.getCallee();
          if (callee && runtime::mlir_async::Callee::valueStorage(*callee) &&
              storage.getNumOperands() > 0)
            return storage.getOperand(0);
        }
      }
    }
    if (mlir::isa<mlir::async::RuntimeAwaitOp>(cursor) ||
        mlir::isa<mlir::async::RuntimeAwaitAndResumeOp>(cursor))
      return {};
  }
  return {};
}

mlir::Value discardedPayload(mlir::Operation *marker) {
  for (mlir::Operation *cursor = marker ? marker->getPrevNode() : nullptr;
       cursor; cursor = cursor->getPrevNode()) {
    if (cursor->hasTrait<mlir::OpTrait::IsTerminator>())
      return {};
    if (auto call = mlir::dyn_cast<mlir::LLVM::CallOp>(cursor)) {
      auto callee = call.getCallee();
      if (callee && *callee == kAsyncExceptionEdgeMarker)
        continue;
    }
    if (auto call = mlir::dyn_cast<mlir::func::CallOp>(cursor))
      if (call.getCallee() == kAsyncExceptionEdgeMarker)
        continue;
    if (auto load = mlir::dyn_cast<mlir::async::RuntimeLoadOp>(cursor)) {
      mlir::Value result = load.getResult();
      return value_type::pointerLike(result.getType()) ? result : mlir::Value();
    }
    if (auto load = mlir::dyn_cast<mlir::LLVM::LoadOp>(cursor)) {
      mlir::Value result = load.getResult();
      if (value_type::pointerLike(result.getType()) &&
          async_runtime::ValueStorage::isAddress(load.getAddr()))
        return result;
    }
    if (mlir::isa<mlir::async::RuntimeAwaitOp>(cursor) ||
        mlir::isa<mlir::async::RuntimeAwaitAndResumeOp>(cursor))
      return {};
  }
  return {};
}

} // namespace cancel_marker

namespace runtime_call {

mlir::LLVM::CallOp emitVoid(mlir::ModuleOp module, mlir::Location loc,
                            mlir::IRRewriter &rewriter, llvm::StringRef name,
                            mlir::ValueRange operands) {
  llvm::SmallVector<mlir::Type> argTypes;
  argTypes.reserve(operands.size());
  for (mlir::Value operand : operands)
    argTypes.push_back(operand.getType());
  auto callee = runtime_func::getOrInsert(
      module, loc, rewriter, name,
      mlir::LLVM::LLVMVoidType::get(module.getContext()), argTypes);
  return rewriter.create<mlir::LLVM::CallOp>(
      loc, mlir::TypeRange{}, mlir::SymbolRefAttr::get(callee), operands);
}

} // namespace runtime_call

namespace suspend_switch {

mlir::Value awaitedHandle(mlir::LLVM::SwitchOp switchOp) {
  for (mlir::Operation *cursor = switchOp->getPrevNode(); cursor;
       cursor = cursor->getPrevNode()) {
    if (auto call = mlir::dyn_cast<mlir::LLVM::CallOp>(cursor)) {
      auto callee = call.getCallee();
      if (callee && async_runtime::Callee::awaitAndExecute(*callee) &&
          call.getNumOperands() > 0)
        return call.getOperand(0);
    }
    if (auto call = mlir::dyn_cast<mlir::func::CallOp>(cursor)) {
      if (async_runtime::Callee::awaitAndExecute(call.getCallee()) &&
          call.getNumOperands() > 0)
        return call.getOperand(0);
    }
  }
  return {};
}

} // namespace suspend_switch

namespace async_ref {

bool dropFor(mlir::Operation &op, mlir::Value handle) {
  if (auto call = mlir::dyn_cast<mlir::LLVM::CallOp>(op)) {
    auto callee = call.getCallee();
    return callee && *callee == "mlirAsyncRuntimeDropRef" &&
           call.getNumOperands() >= 1 && call.getOperand(0) == handle;
  }
  if (auto call = mlir::dyn_cast<mlir::func::CallOp>(op))
    return call.getCallee() == "mlirAsyncRuntimeDropRef" &&
           call.getNumOperands() >= 1 && call.getOperand(0) == handle;
  return false;
}

} // namespace async_ref

namespace py_ref {

bool decFor(mlir::Operation &op, mlir::Value payload) {
  if (auto call = mlir::dyn_cast<mlir::LLVM::CallOp>(op)) {
    auto callee = call.getCallee();
    return callee && *callee == RuntimeSymbols::kDecRef &&
           call.getNumOperands() >= 1 && call.getOperand(0) == payload;
  }
  return false;
}

} // namespace py_ref

namespace value_set {

void add(llvm::SmallVectorImpl<mlir::Value> &payloads, mlir::Value payload) {
  if (!payload || llvm::is_contained(payloads, payload))
    return;
  payloads.push_back(payload);
}

void erase(llvm::SmallVectorImpl<mlir::Value> &payloads, mlir::Value payload) {
  auto it = llvm::find(payloads, payload);
  if (it != payloads.end())
    payloads.erase(it);
}

void intersect(llvm::SmallVectorImpl<mlir::Value> &lhs,
               llvm::ArrayRef<mlir::Value> rhs) {
  for (mlir::Value value : llvm::make_early_inc_range(lhs))
    if (!llvm::is_contained(rhs, value))
      erase(lhs, value);
}

bool same(llvm::ArrayRef<mlir::Value> lhs, llvm::ArrayRef<mlir::Value> rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (mlir::Value value : lhs)
    if (!llvm::is_contained(rhs, value))
      return false;
  return true;
}

} // namespace value_set

namespace successor_args {

void remap(mlir::Operation *terminator, unsigned successorIndex,
           llvm::SmallVectorImpl<mlir::Value> &values) {
  auto branch = mlir::dyn_cast<mlir::BranchOpInterface>(terminator);
  if (!branch || successorIndex >= terminator->getNumSuccessors())
    return;
  mlir::Block *successor = branch->getSuccessor(successorIndex);
  mlir::SuccessorOperands operands =
      branch.getSuccessorOperands(successorIndex);
  unsigned count =
      std::min<unsigned>(operands.size(), successor->getNumArguments());
  for (unsigned i = 0; i != count; ++i) {
    if (operands.isOperandProduced(i))
      continue;
    mlir::Value operand = operands[i];
    mlir::BlockArgument argument = successor->getArgument(i);
    for (mlir::Value &value : values)
      if (value == operand)
        value = argument;
  }
}

} // namespace successor_args

namespace liveness {

struct State {
  llvm::SmallVector<mlir::Value> values;
  bool initialized = false;
};

llvm::SmallVector<mlir::Value>
before(mlir::Operation *limit,
       llvm::function_ref<void(mlir::Operation &,
                               llvm::SmallVectorImpl<mlir::Value> &)>
           transfer) {
  if (!limit || !limit->getBlock())
    return {};
  mlir::Region *region = limit->getBlock()->getParent();
  if (!region || region->empty())
    return {};

  llvm::DenseMap<mlir::Block *, State> states;
  llvm::SmallVector<mlir::Block *, 16> worklist;
  llvm::SmallPtrSet<mlir::Block *, 16> queued;

  mlir::Block *entry = &region->front();
  states[entry].initialized = true;
  worklist.push_back(entry);
  queued.insert(entry);

  while (!worklist.empty()) {
    mlir::Block *block = worklist.pop_back_val();
    queued.erase(block);

    State blockState = states[block];
    if (!blockState.initialized)
      continue;

    llvm::SmallVector<mlir::Value> out = blockState.values;
    for (mlir::Operation &op : *block)
      transfer(op, out);

    mlir::Operation *terminator = block->getTerminator();
    if (!terminator)
      continue;
    for (unsigned i = 0, e = terminator->getNumSuccessors(); i != e; ++i) {
      mlir::Block *successor = terminator->getSuccessor(i);
      llvm::SmallVector<mlir::Value> incoming = out;
      successor_args::remap(terminator, i, incoming);

      State &successorState = states[successor];
      bool changed = false;
      if (!successorState.initialized) {
        successorState.values = std::move(incoming);
        successorState.initialized = true;
        changed = true;
      } else {
        llvm::SmallVector<mlir::Value> merged = successorState.values;
        value_set::intersect(merged, incoming);
        changed = !value_set::same(successorState.values, merged);
        successorState.values = std::move(merged);
      }
      if (changed && queued.insert(successor).second)
        worklist.push_back(successor);
    }
  }

  State limitEntry = states[limit->getBlock()];
  llvm::SmallVector<mlir::Value> values =
      limitEntry.initialized ? std::move(limitEntry.values)
                             : llvm::SmallVector<mlir::Value>();
  for (mlir::Operation &op : *limit->getBlock()) {
    if (&op == limit)
      break;
    transfer(op, values);
  }
  return values;
}

} // namespace liveness

namespace async_child {

void transfer(mlir::Operation &op,
              llvm::SmallVectorImpl<mlir::Value> &handles) {
  if (async_runtime::Handle::ownedChildProducer(&op)) {
    for (mlir::Value result : op.getResults())
      if (value_type::pointerLike(result.getType()))
        value_set::add(handles, result);
  }
  for (mlir::Value handle : llvm::make_early_inc_range(handles))
    if (async_ref::dropFor(op, handle))
      value_set::erase(handles, handle);
}

} // namespace async_child

namespace async_payload {

void transfer(mlir::Operation &op,
              llvm::SmallVectorImpl<mlir::Value> &payloads) {
  if (auto load = mlir::dyn_cast<mlir::LLVM::LoadOp>(op)) {
    mlir::Value result = load.getResult();
    if (value_type::pointerLike(result.getType()) &&
        async_runtime::ValueStorage::isAddress(load.getAddr())) {
      value_set::add(payloads, result);
      return;
    }
  }
  for (mlir::Value payload : llvm::make_early_inc_range(payloads))
    if (py_ref::decFor(op, payload))
      value_set::erase(payloads, payload);
}

} // namespace async_payload

namespace suspend_cleanup {

llvm::SmallVector<mlir::Value> childHandles(mlir::LLVM::SwitchOp switchOp) {
  return liveness::before(switchOp.getOperation(), async_child::transfer);
}

llvm::SmallVector<mlir::Value> payloads(mlir::LLVM::SwitchOp switchOp) {
  return liveness::before(switchOp.getOperation(), async_payload::transfer);
}

} // namespace suspend_cleanup

namespace await_cleanup {

mlir::LogicalResult rewrite(mlir::ModuleOp module) {
  struct CleanupDrops {
    mlir::LLVM::SwitchOp switchOp;
    unsigned successorIndex;
    llvm::SmallVector<mlir::Value> handles;
    llvm::SmallVector<mlir::Value> payloads;
  };
  llvm::SmallVector<CleanupDrops, 8> cleanupDrops;
  module.walk([&](mlir::LLVM::SwitchOp switchOp) {
    if (!isCoroSuspendStatus(switchOp.getValue()))
      return;
    auto cleanupIndex = findCoroSuspendCleanupSuccessorIndex(switchOp);
    if (!cleanupIndex)
      return;
    mlir::Value awaited = suspend_switch::awaitedHandle(switchOp);
    llvm::SmallVector<mlir::Value> handles =
        suspend_cleanup::childHandles(switchOp);
    value_set::add(handles, awaited);
    llvm::SmallVector<mlir::Value> payloads =
        suspend_cleanup::payloads(switchOp);
    if (handles.empty() && payloads.empty())
      return;
    cleanupDrops.push_back(CleanupDrops{
        switchOp, *cleanupIndex, std::move(handles), std::move(payloads)});
  });

  mlir::IRRewriter rewriter(module.getContext());
  for (auto &drop : cleanupDrops) {
    mlir::LLVM::SwitchOp switchOp = drop.switchOp;
    mlir::Block *cleanupBlock = switchOp->getSuccessor(drop.successorIndex);
    if (!cleanupBlock)
      continue;
    mlir::Block *dropBlock = rewriter.createBlock(cleanupBlock->getParent(),
                                                  cleanupBlock->getIterator());
    switchOp->setSuccessor(dropBlock, drop.successorIndex);
    rewriter.setInsertionPointToStart(dropBlock);
    mlir::Value one = rewriter.create<mlir::LLVM::ConstantOp>(
        switchOp.getLoc(), rewriter.getI64Type(),
        rewriter.getI64IntegerAttr(1));
    for (mlir::Value handle : drop.handles) {
      if (module.lookupSymbol<mlir::func::FuncOp>("mlirAsyncRuntimeDropRef")) {
        auto call = rewriter.create<mlir::func::CallOp>(
            switchOp.getLoc(), "mlirAsyncRuntimeDropRef", mlir::TypeRange{},
            mlir::ValueRange{handle, one});
        call->setAttr(OwnershipContractAttrs::kFrameTransfer,
                      mlir::UnitAttr::get(module.getContext()));
      } else {
        runtime_call::emitVoid(module, switchOp.getLoc(), rewriter,
                               "mlirAsyncRuntimeDropRef",
                               mlir::ValueRange{handle, one})
            ->setAttr(OwnershipContractAttrs::kFrameTransfer,
                      mlir::UnitAttr::get(module.getContext()));
      }
    }
    for (mlir::Value payload : drop.payloads)
      runtime_call::emitVoid(module, switchOp.getLoc(), rewriter,
                             RuntimeSymbols::kDecRef, mlir::ValueRange{payload})
          ->setAttr(OwnershipContractAttrs::kFrameTransfer,
                    mlir::UnitAttr::get(module.getContext()));
    rewriter.create<mlir::cf::BranchOp>(switchOp.getLoc(), cleanupBlock);
  }
  return mlir::success();
}

} // namespace await_cleanup

namespace async_error_branch {

constexpr llvm::StringLiteral kCleanupErrorCheck =
    "ly.async.cleanup_error_check";

bool runtimeCondition(mlir::Value condition) {
  if (!condition)
    return false;
  if (auto call = condition.getDefiningOp<mlir::func::CallOp>())
    return async_runtime::Callee::isError(call.getCallee());
  if (auto call = condition.getDefiningOp<mlir::LLVM::CallOp>()) {
    auto callee = call.getCallee();
    return callee && async_runtime::Callee::isError(*callee);
  }
  return false;
}

mlir::Value handle(mlir::Value condition) {
  if (!condition)
    return {};
  if (auto isError = condition.getDefiningOp<mlir::async::RuntimeIsErrorOp>())
    return isError.getOperand();
  if (auto call = condition.getDefiningOp<mlir::func::CallOp>())
    if (async_runtime::Callee::isError(call.getCallee()) &&
        call.getNumOperands() > 0)
      return call.getOperand(0);
  if (auto call = condition.getDefiningOp<mlir::LLVM::CallOp>()) {
    auto callee = call.getCallee();
    if (callee && async_runtime::Callee::isError(*callee) &&
        call.getNumOperands() > 0)
      return call.getOperand(0);
  }
  return {};
}

std::optional<unsigned> successorIndex(mlir::Operation *op) {
  if (op->hasAttr(kCleanupErrorCheck))
    return std::nullopt;
  mlir::Value condition = cfedge::condition(op);
  if (!runtimeCondition(condition))
    return std::nullopt;
  if (mlir::isa<mlir::cf::CondBranchOp, mlir::LLVM::CondBrOp>(op) &&
      op->getNumSuccessors() >= 2)
    return 0;
  return std::nullopt;
}

} // namespace async_error_branch

namespace cleanup_block {

bool hasDecRef(mlir::Block *block, mlir::Value payload) {
  if (!block)
    return false;
  for (mlir::Operation &op : *block)
    if (py_ref::decFor(op, payload))
      return true;
  return false;
}

bool hasAsyncDrop(mlir::Block *block, mlir::Value handle) {
  if (!block || !handle)
    return false;
  for (mlir::Operation &op : *block)
    if (async_ref::dropFor(op, handle))
      return true;
  return false;
}

bool usesAsyncHandle(mlir::Block *block, mlir::Value handle) {
  if (!block || !handle)
    return false;
  for (mlir::Operation &op : *block)
    for (mlir::Value operand : op.getOperands())
      if (operand == handle)
        return true;
  return false;
}

} // namespace cleanup_block

namespace error_handle_cleanup {

mlir::LogicalResult rewrite(mlir::ModuleOp module) {
  struct HandleCleanup {
    mlir::Operation *branch;
    unsigned successorIndex;
    llvm::SmallVector<mlir::Value> handles;
  };
  llvm::SmallVector<HandleCleanup, 8> cleanups;
  module.walk([&](mlir::Operation *op) {
    auto errorIndex = async_error_branch::successorIndex(op);
    if (!errorIndex)
      return;
    mlir::Value handle = async_error_branch::handle(cfedge::condition(op));
    if (!handle)
      return;
    mlir::Block *errorTarget = op->getSuccessor(*errorIndex);
    llvm::SmallVector<mlir::Value> handles =
        liveness::before(op, async_child::transfer);
    value_set::add(handles, handle);
    for (mlir::Value liveHandle : llvm::make_early_inc_range(handles))
      if (cleanup_block::hasAsyncDrop(errorTarget, liveHandle) ||
          cleanup_block::usesAsyncHandle(errorTarget, liveHandle))
        value_set::erase(handles, liveHandle);
    if (handles.empty())
      return;
    cleanups.push_back({op, *errorIndex, std::move(handles)});
  });

  mlir::IRRewriter rewriter(module.getContext());
  for (HandleCleanup &cleanup : cleanups) {
    mlir::Operation *branch = cleanup.branch;
    if (!branch || cleanup.successorIndex >= branch->getNumSuccessors())
      continue;
    mlir::Block *errorTarget = branch->getSuccessor(cleanup.successorIndex);
    if (!errorTarget)
      continue;
    mlir::Block *cleanupBlock = rewriter.createBlock(
        errorTarget->getParent(), errorTarget->getIterator());
    branch->setSuccessor(cleanupBlock, cleanup.successorIndex);
    rewriter.setInsertionPointToStart(cleanupBlock);
    auto one = rewriter.getI64IntegerAttr(1);
    for (mlir::Value handle : cleanup.handles) {
      if (mlir::isa<mlir::async::ValueType>(handle.getType())) {
        rewriter.create<mlir::async::RuntimeDropRefOp>(branch->getLoc(), handle,
                                                       one);
      } else {
        mlir::Value count = rewriter.create<mlir::LLVM::ConstantOp>(
            branch->getLoc(), rewriter.getI64Type(), one);
        if (module.lookupSymbol<mlir::func::FuncOp>(
                "mlirAsyncRuntimeDropRef")) {
          rewriter.create<mlir::func::CallOp>(
              branch->getLoc(), "mlirAsyncRuntimeDropRef", mlir::TypeRange{},
              mlir::ValueRange{handle, count});
        } else {
          runtime_call::emitVoid(module, branch->getLoc(), rewriter,
                                 "mlirAsyncRuntimeDropRef",
                                 mlir::ValueRange{handle, count});
        }
      }
    }
    rewriter.create<mlir::cf::BranchOp>(branch->getLoc(), errorTarget);
  }
  return mlir::success();
}

} // namespace error_handle_cleanup

namespace error_payload_cleanup {

mlir::LogicalResult rewrite(mlir::ModuleOp module) {
  struct ErrorCleanup {
    mlir::Operation *branch;
    unsigned successorIndex;
    llvm::SmallVector<mlir::Value> payloads;
  };
  llvm::SmallVector<ErrorCleanup, 8> cleanups;
  module.walk([&](mlir::Operation *op) {
    auto errorIndex = async_error_branch::successorIndex(op);
    if (!errorIndex)
      return;
    mlir::Block *errorTarget = op->getSuccessor(*errorIndex);
    llvm::SmallVector<mlir::Value> payloads =
        liveness::before(op, async_payload::transfer);
    for (mlir::Value payload : llvm::make_early_inc_range(payloads)) {
      if (cleanup_block::hasDecRef(errorTarget, payload))
        value_set::erase(payloads, payload);
    }
    if (!payloads.empty())
      cleanups.push_back({op, *errorIndex, std::move(payloads)});
  });

  mlir::IRRewriter rewriter(module.getContext());
  for (ErrorCleanup &cleanup : cleanups) {
    mlir::Operation *branch = cleanup.branch;
    if (!branch || cleanup.successorIndex >= branch->getNumSuccessors())
      continue;
    mlir::Block *errorTarget = branch->getSuccessor(cleanup.successorIndex);
    if (!errorTarget)
      continue;
    mlir::Block *cleanupBlock = rewriter.createBlock(
        errorTarget->getParent(), errorTarget->getIterator());
    branch->setSuccessor(cleanupBlock, cleanup.successorIndex);
    rewriter.setInsertionPointToStart(cleanupBlock);
    for (mlir::Value payload : cleanup.payloads)
      runtime_call::emitVoid(module, branch->getLoc(), rewriter,
                             RuntimeSymbols::kDecRef,
                             mlir::ValueRange{payload});
    rewriter.create<mlir::cf::BranchOp>(branch->getLoc(), errorTarget);
  }
  return mlir::success();
}

} // namespace error_payload_cleanup

namespace exception_cell {

void storeFirst(
    mlir::ModuleOp module, mlir::Location loc, mlir::IRRewriter &rewriter,
    mlir::Value destCell, mlir::Value exception,
    llvm::StringRef retainPremise = ThreadSafetyAttrs::kPremiseOwnedToken) {
  PyLLVMTypeConverter typeConverter(module.getContext());
  async_runtime::ExceptionCell::storeFirst(
      loc, destCell, exception, module, rewriter, typeConverter, retainPremise);
}

void copyAt(mlir::ModuleOp module, mlir::Location loc,
            mlir::IRRewriter &rewriter, mlir::Value sourceCell,
            mlir::Value destCell) {
  if (!sourceCell || !destCell || sourceCell == destCell)
    return;

  auto ptrType = mlir::LLVM::LLVMPointerType::get(module.getContext());
  mlir::Value exception =
      async_runtime::ExceptionCell::load(loc, sourceCell, rewriter);
  mlir::Value nullPtr = rewriter.create<mlir::LLVM::ZeroOp>(loc, ptrType);
  mlir::Value isNull = rewriter.create<mlir::LLVM::ICmpOp>(
      loc, mlir::LLVM::ICmpPredicate::eq, exception, nullPtr);

  mlir::Block *currentBlock = rewriter.getInsertionBlock();
  mlir::Block *afterBlock =
      rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
  mlir::Block *copyBlock =
      rewriter.createBlock(afterBlock->getParent(), afterBlock->getIterator());
  rewriter.setInsertionPointToEnd(currentBlock);
  rewriter.create<mlir::cf::CondBranchOp>(loc, isNull, afterBlock, copyBlock);

  rewriter.setInsertionPointToStart(copyBlock);
  storeFirst(module, loc, rewriter, destCell, exception,
             ThreadSafetyAttrs::kPremiseAggregateBorrow);
  rewriter.create<mlir::cf::BranchOp>(loc, afterBlock);

  rewriter.setInsertionPointToStart(afterBlock);
}

void destroyAt(mlir::ModuleOp module, mlir::Location loc,
               mlir::IRRewriter &rewriter, mlir::Value cell) {
  if (!cell)
    return;
  PyLLVMTypeConverter typeConverter(module.getContext());
  async_runtime::ExceptionCell::destroy(loc, module, rewriter, typeConverter,
                                        cell);
}

void moveAt(mlir::ModuleOp module, mlir::Location loc,
            mlir::IRRewriter &rewriter, mlir::Value sourceCell,
            mlir::Value destCell) {
  if (!sourceCell || !destCell || sourceCell == destCell)
    return;
  copyAt(module, loc, rewriter, sourceCell, destCell);
  destroyAt(module, loc, rewriter, sourceCell);
}

void moveFirst(mlir::ModuleOp module, mlir::Location loc,
               mlir::Block *errorBlock, mlir::Value sourceCell,
               mlir::Value destCell) {
  mlir::IRRewriter rewriter(module.getContext());
  rewriter.setInsertionPointToStart(errorBlock);
  moveAt(module, loc, rewriter, sourceCell, destCell);
}

} // namespace exception_cell

namespace await_error {

mlir::Block *blockForContinuation(mlir::Block *continuationBlock) {
  if (!continuationBlock)
    return nullptr;
  for (mlir::Block *pred : continuationBlock->getPredecessors()) {
    mlir::Operation *branch = pred->getTerminator();
    if (!branch || branch->getNumSuccessors() != 2)
      continue;
    mlir::Value condition = cfedge::condition(branch);
    if (!condition || !condition.getDefiningOp<mlir::async::RuntimeIsErrorOp>())
      continue;
    if (branch->getSuccessor(1) == continuationBlock)
      return branch->getSuccessor(0);
    if (branch->getSuccessor(0) == continuationBlock)
      return branch->getSuccessor(1);
  }
  return nullptr;
}

} // namespace await_error

namespace exception_payload {

mlir::LogicalResult rewrite(mlir::ModuleOp module) {
  struct Marker {
    mlir::Operation *op;
    mlir::Value sourceCell;
    mlir::Value destCell;
  };
  llvm::SmallVector<Marker> markers;
  module.walk([&](mlir::func::CallOp call) {
    if (call.getCallee() == kAsyncExceptionEdgeMarker)
      markers.push_back(
          {call.getOperation(), call.getOperand(0), call.getOperand(1)});
  });
  module.walk([&](mlir::LLVM::CallOp call) {
    auto callee = call.getCallee();
    if (callee && *callee == kAsyncExceptionEdgeMarker &&
        call.getNumOperands() == 2)
      markers.push_back(
          {call.getOperation(), call.getOperand(0), call.getOperand(1)});
  });

  mlir::LogicalResult result = mlir::success();
  for (Marker marker : markers) {
    mlir::Block *errorBlock =
        await_error::blockForContinuation(marker.op->getBlock());
    if (!errorBlock) {
      marker.op->emitError(
          "failed to find async await error branch for payload propagation");
      result = mlir::failure();
      continue;
    }
    exception_cell::moveFirst(module, marker.op->getLoc(), errorBlock,
                              marker.sourceCell, marker.destCell);
    marker.op->erase();
  }

  if (auto markerFunc =
          module.lookupSymbol<mlir::func::FuncOp>(kAsyncExceptionEdgeMarker)) {
    if (markerFunc.isPrivate() && markerFunc.isDeclaration())
      markerFunc.erase();
  }
  if (auto markerFunc = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(
          kAsyncExceptionEdgeMarker))
    markerFunc.erase();

  return result;
}

} // namespace exception_payload

namespace cancel_flag {

mlir::Value pointer(mlir::Location loc, mlir::IRRewriter &rewriter,
                    mlir::Value flagStorage) {
  if (!flagStorage)
    return {};
  if (mlir::isa<mlir::LLVM::LLVMPointerType>(flagStorage.getType()))
    return flagStorage;
  auto descriptorType =
      mlir::dyn_cast<mlir::LLVM::LLVMStructType>(flagStorage.getType());
  if (!descriptorType || descriptorType.getBody().size() < 2 ||
      !mlir::isa<mlir::LLVM::LLVMPointerType>(descriptorType.getBody()[1]))
    return {};
  return rewriter.create<mlir::LLVM::ExtractValueOp>(
      loc, descriptorType.getBody()[1], flagStorage,
      rewriter.getDenseI64ArrayAttr({1}));
}

} // namespace cancel_flag

namespace task_cancel {

mlir::LogicalResult rewrite(mlir::ModuleOp module) {
  struct Marker {
    mlir::Operation *op;
    mlir::Value flagStorage;
    mlir::Value sourceCell;
    mlir::Value destCell;
  };
  llvm::SmallVector<Marker> markers;
  mlir::LogicalResult result = mlir::success();
  module.walk([&](mlir::func::CallOp call) {
    if (call.getCallee() != kAsyncTaskCancelMarker)
      return;
    if (call.getNumOperands() != 3) {
      call.emitError("invalid async task cancel marker operand count");
      result = mlir::failure();
      return;
    }
    markers.push_back({call.getOperation(), call.getOperand(0),
                       call.getOperand(1), call.getOperand(2)});
  });
  module.walk([&](mlir::LLVM::CallOp call) {
    auto callee = call.getCallee();
    if (!callee || *callee != kAsyncTaskCancelMarker)
      return;
    if (call.getNumOperands() == 3) {
      markers.push_back({call.getOperation(), call.getOperand(0),
                         call.getOperand(1), call.getOperand(2)});
      return;
    }
    if (call.getNumOperands() >= 7) {
      markers.push_back({call.getOperation(), call.getOperand(1),
                         call.getOperand(call.getNumOperands() - 2),
                         call.getOperand(call.getNumOperands() - 1)});
      return;
    }
    call.emitError("invalid async task cancel marker operand count");
    result = mlir::failure();
  });

  for (Marker marker : markers) {
    mlir::Block *continuationBlock = marker.op->getBlock();
    mlir::Block *errorBlock =
        await_error::blockForContinuation(continuationBlock);
    if (!errorBlock) {
      marker.op->emitError(
          "failed to find async await error branch for task cancellation");
      result = mlir::failure();
      continue;
    }
    mlir::async::RuntimeSetErrorOp setError;
    for (mlir::Operation &op : errorBlock->getOperations()) {
      setError = mlir::dyn_cast<mlir::async::RuntimeSetErrorOp>(op);
      if (setError)
        break;
    }
    mlir::Operation *errorTerminator = errorBlock->getTerminator();
    if (!setError || !errorTerminator ||
        errorTerminator->getNumSuccessors() != 1) {
      marker.op->emitError(
          "failed to find async await error merge for task cancellation");
      result = mlir::failure();
      continue;
    }
    mlir::Block *errorMergeBlock = errorTerminator->getSuccessor(0);
    mlir::Value discardedPayload = cancel_marker::discardedPayload(marker.op);
    mlir::Value awaitedHandle = cancel_marker::awaitedHandle(marker.op);

    mlir::IRRewriter rewriter(module.getContext());
    rewriter.setInsertionPoint(marker.op);
    mlir::Value flagPtr =
        cancel_flag::pointer(marker.op->getLoc(), rewriter, marker.flagStorage);
    if (!flagPtr) {
      marker.op->emitError("failed to materialize async task cancel flag");
      result = mlir::failure();
      continue;
    }

    mlir::Block *currentBlock = marker.op->getBlock();
    mlir::Block *successBlock =
        rewriter.splitBlock(currentBlock, marker.op->getIterator());
    mlir::Block *cancelBlock = rewriter.createBlock(
        successBlock->getParent(), successBlock->getIterator());

    rewriter.setInsertionPointToEnd(currentBlock);
    mlir::Value flag = rewriter.create<mlir::LLVM::LoadOp>(
        marker.op->getLoc(), rewriter.getI8Type(), flagPtr,
        /*alignment=*/1, /*isVolatile=*/false, /*isNonTemporal=*/false,
        /*isInvariant=*/false, /*isInvariantGroup=*/false,
        mlir::LLVM::AtomicOrdering::acquire);
    threadsafe::Atomic::set(flag.getDefiningOp(),
                            ThreadSafetyAttrs::kRoleAsyncCancelLoad,
                            ThreadSafetyAttrs::kOrderingAcquire);
    flag.getDefiningOp()->setAttr(AsyncSafetyAttrs::kCancelFlag,
                                  mlir::UnitAttr::get(rewriter.getContext()));
    mlir::Value zero = rewriter.create<mlir::LLVM::ConstantOp>(
        marker.op->getLoc(), rewriter.getI8Type(),
        rewriter.getIntegerAttr(rewriter.getI8Type(), 0));
    mlir::Value isCancelled = rewriter.create<mlir::LLVM::ICmpOp>(
        marker.op->getLoc(), mlir::LLVM::ICmpPredicate::ne, flag, zero);
    rewriter.create<mlir::cf::CondBranchOp>(marker.op->getLoc(), isCancelled,
                                            cancelBlock, successBlock);

    rewriter.setInsertionPointToStart(cancelBlock);
    if (discardedPayload)
      runtime_call::emitVoid(module, marker.op->getLoc(), rewriter,
                             RuntimeSymbols::kDecRef,
                             mlir::ValueRange{discardedPayload});
    exception_cell::moveAt(module, marker.op->getLoc(), rewriter,
                           marker.sourceCell, marker.destCell);
    if (mlir::isa<mlir::LLVM::LLVMPointerType>(flagPtr.getType()))
      runtime_call::emitVoid(module, marker.op->getLoc(), rewriter, "free",
                             mlir::ValueRange{flagPtr});
    if (awaitedHandle) {
      auto one = rewriter.getI64IntegerAttr(1);
      if (mlir::isa<mlir::async::ValueType>(awaitedHandle.getType())) {
        rewriter.create<mlir::async::RuntimeDropRefOp>(marker.op->getLoc(),
                                                       awaitedHandle, one);
      } else {
        mlir::Value count = rewriter.create<mlir::LLVM::ConstantOp>(
            marker.op->getLoc(), rewriter.getI64Type(), one);
        if (module.lookupSymbol<mlir::func::FuncOp>(
                "mlirAsyncRuntimeDropRef")) {
          rewriter.create<mlir::func::CallOp>(
              marker.op->getLoc(), "mlirAsyncRuntimeDropRef", mlir::TypeRange{},
              mlir::ValueRange{awaitedHandle, count});
        } else {
          runtime_call::emitVoid(module, marker.op->getLoc(), rewriter,
                                 "mlirAsyncRuntimeDropRef",
                                 mlir::ValueRange{awaitedHandle, count});
        }
      }
    }
    rewriter.create<mlir::async::RuntimeSetErrorOp>(marker.op->getLoc(),
                                                    setError.getOperand());
    rewriter.create<mlir::cf::BranchOp>(marker.op->getLoc(), errorMergeBlock);
    marker.op->erase();
  }

  if (auto markerFunc =
          module.lookupSymbol<mlir::func::FuncOp>(kAsyncTaskCancelMarker)) {
    if (markerFunc.isPrivate() && markerFunc.isDeclaration())
      markerFunc.erase();
  }
  if (auto markerFunc =
          module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(kAsyncTaskCancelMarker))
    markerFunc.erase();

  return result;
}

} // namespace task_cancel

namespace finally_error {

mlir::LogicalResult rewrite(mlir::ModuleOp module) {
  constexpr llvm::StringLiteral kAwaitErrorId = "ly.finally.error_id";
  constexpr llvm::StringLiteral kErrorBlockId = "ly.finally.error_block_id";
  constexpr llvm::StringLiteral kCleanupErrorCheck =
      "ly.async.cleanup_error_check";

  llvm::SmallVector<std::pair<mlir::Attribute, mlir::Block *>> errorBlocks;
  llvm::SmallVector<mlir::Operation *> markerOps;
  module.walk([&](mlir::Operation *op) {
    mlir::Attribute id = op->getAttr(kErrorBlockId);
    if (!id)
      return;
    errorBlocks.push_back({id, op->getBlock()});
    markerOps.push_back(op);
  });

  auto findErrorBlock = [&](mlir::Attribute id) -> mlir::Block * {
    for (auto [candidate, block] : errorBlocks)
      if (candidate == id)
        return block;
    return nullptr;
  };

  mlir::LogicalResult result = mlir::success();
  llvm::SmallVector<mlir::Operation *> branches;
  module.walk([&](mlir::cf::CondBranchOp branch) {
    if (branch->hasAttr(kAwaitErrorId))
      branches.push_back(branch.getOperation());
  });
  module.walk([&](mlir::LLVM::CondBrOp branch) {
    if (branch->hasAttr(kAwaitErrorId))
      branches.push_back(branch.getOperation());
  });

  for (mlir::Operation *branch : branches) {
    mlir::Attribute id = branch->getAttr(kAwaitErrorId);
    mlir::Block *errorBlock = findErrorBlock(id);
    if (!errorBlock) {
      branch->emitError("missing Lython async finally error block");
      result = mlir::failure();
      continue;
    }
    branch->setSuccessor(errorBlock, 0);
    branch->removeAttr(kAwaitErrorId);
  }

  llvm::SmallVector<mlir::Operation *> allBranches;
  module.walk([&](mlir::cf::CondBranchOp branch) {
    allBranches.push_back(branch.getOperation());
  });
  module.walk([&](mlir::LLVM::CondBrOp branch) {
    allBranches.push_back(branch.getOperation());
  });
  for (auto [id, errorBlock] : errorBlocks) {
    for (mlir::Operation *precheck : allBranches) {
      if (precheck->hasAttr(kCleanupErrorCheck))
        continue;
      if (precheck->getNumSuccessors() < 2)
        continue;
      bool reachesError = precheck->getSuccessor(0) == errorBlock ||
                          precheck->getSuccessor(1) == errorBlock;
      if (!reachesError)
        continue;

      mlir::Value precheckCondition = cfedge::condition(precheck);
      if (!precheckCondition)
        continue;
      auto precheckIsError =
          precheckCondition.getDefiningOp<mlir::async::RuntimeIsErrorOp>();
      if (!precheckIsError)
        continue;
      mlir::Value awaitable = precheckIsError.getOperand();

      for (mlir::Operation *postAwait : allBranches) {
        if (postAwait->hasAttr(kCleanupErrorCheck))
          continue;
        if (postAwait == precheck || postAwait->getNumSuccessors() < 2)
          continue;
        mlir::Value postAwaitCondition = cfedge::condition(postAwait);
        if (!postAwaitCondition)
          continue;
        auto postAwaitIsError =
            postAwaitCondition.getDefiningOp<mlir::async::RuntimeIsErrorOp>();
        if (!postAwaitIsError || postAwaitIsError.getOperand() != awaitable)
          continue;
        if (postAwait->getSuccessor(0) == errorBlock)
          continue;
        postAwait->setSuccessor(errorBlock, 0);
      }
      (void)id;
    }
  }

  if (mlir::failed(result))
    return mlir::failure();

  for (mlir::Operation *marker : markerOps) {
    if (!marker || marker->getBlock() == nullptr)
      continue;
    marker->removeAttr(kErrorBlockId);
  }
  return mlir::success();
}

} // namespace finally_error

struct AsyncRuntimeRewritePass
    : public mlir::PassWrapper<AsyncRuntimeRewritePass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AsyncRuntimeRewritePass)

  llvm::StringRef getArgument() const override {
    return "py-async-runtime-rewrite";
  }

  llvm::StringRef getDescription() const override {
    return "Rewrite Lython async exception/cancel markers after async runtime "
           "conversion";
  }

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    if (mlir::failed(finally_error::rewrite(module)) ||
        mlir::failed(task_cancel::rewrite(module)) ||
        mlir::failed(exception_payload::rewrite(module)) ||
        mlir::failed(error_handle_cleanup::rewrite(module)) ||
        mlir::failed(error_payload_cleanup::rewrite(module)) ||
        mlir::failed(await_cleanup::rewrite(module)))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createAsyncRuntimeRewritePass() {
  return std::make_unique<AsyncRuntimeRewritePass>();
}

} // namespace py
