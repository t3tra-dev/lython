#include "Common/AsyncSafetyKernel.h"
#include "Common/LoweringUtils.h"
#include "Common/RuntimeSupport.h"
#include "Passes/OwnershipAnalysis.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

namespace py {
namespace {

using AliasAnalysis = OwnershipAliasAnalysis;

namespace value_type {

bool pointerLike(mlir::Type type) {
  return mlir::isa<mlir::LLVM::LLVMPointerType>(type);
}

} // namespace value_type

namespace carrier {

bool ownership(mlir::Value value) {
  mlir::Type type = value.getType();
  if (value_type::pointerLike(type))
    return true;
  if (!type.isInteger(64))
    return false;

  if (auto ptrToInt = value.getDefiningOp<mlir::LLVM::PtrToIntOp>())
    return value_type::pointerLike(ptrToInt.getArg().getType());
  if (auto load = value.getDefiningOp<mlir::LLVM::LoadOp>())
    return load->hasAttr(OwnershipContractAttrs::kAggregateSlotLoad);
  if (auto load = value.getDefiningOp<mlir::memref::LoadOp>())
    return load->hasAttr(OwnershipContractAttrs::kAggregateSlotLoad);
  return false;
}

} // namespace carrier

namespace function_arg {

bool hasAttr(mlir::Value value, llvm::StringRef attrName) {
  auto arg = mlir::dyn_cast<mlir::BlockArgument>(value);
  if (!arg)
    return false;
  auto func =
      mlir::dyn_cast<mlir::FunctionOpInterface>(arg.getOwner()->getParentOp());
  return func && func.getArgAttr(arg.getArgNumber(), attrName);
}

} // namespace function_arg

namespace lowered_identity {

bool transform(mlir::Operation *op) {
  if (mlir::isa<mlir::memref::CastOp, mlir::LLVM::BitcastOp,
                mlir::LLVM::IntToPtrOp, mlir::LLVM::PtrToIntOp>(op))
    return op->getNumOperands() == 1 && op->getNumResults() == 1;
  return false;
}

} // namespace lowered_identity

namespace callee_kind {

bool retain(llvm::StringRef callee) { return runtime::Callee::retain(callee); }

bool release(llvm::StringRef callee) {
  return runtime::Callee::release(callee);
}

bool setField(llvm::StringRef callee) {
  return runtime::Callee::setField(callee);
}

bool transfer(llvm::StringRef callee) {
  return runtime::Callee::transfer(callee);
}

bool localDestroy(llvm::StringRef callee) {
  return callee.starts_with("__ly_class_destroy_local_");
}

bool runtimeLike(llvm::StringRef callee) {
  return callee.starts_with("Ly") || callee.starts_with("__ly_") ||
         callee.starts_with("_mlir_ciface_") || callee == "builtin_print_impl";
}

bool runtimeComparison(llvm::StringRef callee) {
  return callee == RuntimeSymbols::kNumberLt ||
         callee == RuntimeSymbols::kNumberLe ||
         callee == RuntimeSymbols::kNumberGt ||
         callee == RuntimeSymbols::kNumberGe ||
         callee == RuntimeSymbols::kNumberEq ||
         callee == RuntimeSymbols::kNumberNe;
}

bool runtimeBorrowOnly(llvm::StringRef callee) {
  return callee == RuntimeSymbols::kLongAsI64 ||
         callee == RuntimeSymbols::kFloatAsDouble ||
         callee == RuntimeSymbols::kBoolAsBool ||
         callee == RuntimeSymbols::kLongCompare ||
         callee == RuntimeSymbols::kObjectEqBool ||
         callee == RuntimeSymbols::kTracebackPush ||
         callee == RuntimeSymbols::kTracebackPop ||
         callee == RuntimeSymbols::kEHReportUnhandled ||
         callee == RuntimeSymbols::kExceptionSetCurrent ||
         callee == RuntimeSymbols::kExceptionClear ||
         callee == RuntimeSymbols::kBuiltinPrintImpl ||
         runtimeComparison(callee);
}

bool asyncRewriteMarker(llvm::StringRef callee) {
  return callee == "__lython_async_exception_edge_marker" ||
         callee == "__lython_async_task_cancel_marker";
}

bool nonObjectAllocator(llvm::StringRef callee) {
  return callee == "aligned_alloc" || callee == "malloc" || callee == "free" ||
         callee == RuntimeSymbols::kMemAlloc ||
         callee == RuntimeSymbols::kMemFree;
}

} // namespace callee_kind

namespace constant {

bool boolFalse(mlir::Value value) {
  if (auto constant = value.getDefiningOp<mlir::LLVM::ConstantOp>()) {
    mlir::Attribute attr = constant.getValue();
    if (auto boolAttr = mlir::dyn_cast<mlir::BoolAttr>(attr))
      return !boolAttr.getValue();
    if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr))
      return intAttr.getInt() == 0;
  }
  if (auto constant = value.getDefiningOp<mlir::arith::ConstantOp>()) {
    mlir::Attribute attr = constant.getValue();
    if (auto boolAttr = mlir::dyn_cast<mlir::BoolAttr>(attr))
      return !boolAttr.getValue();
    if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr))
      return intAttr.getInt() == 0;
  }
  if (auto constant = value.getDefiningOp<mlir::arith::ConstantIntOp>())
    return constant.value() == 0;
  return false;
}

} // namespace constant

static mlir::LLVM::LLVMFuncOp lookupLLVMFunc(mlir::Operation *op,
                                             llvm::StringRef name) {
  auto module = op->getParentOfType<mlir::ModuleOp>();
  if (!module)
    return {};
  return module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(name);
}

namespace class_helper {

bool hasKind(mlir::Operation *op, llvm::StringRef expectedKind) {
  if (!op)
    return false;
  auto helperKind =
      op->getAttrOfType<mlir::StringAttr>(ClassSafetyAttrs::kHelperKind);
  auto helperClass =
      op->getAttrOfType<mlir::StringAttr>(ClassSafetyAttrs::kHelperClass);
  return helperKind && helperClass && helperKind.getValue() == expectedKind;
}

bool calleeHasKind(mlir::Operation *context, llvm::StringRef name,
                   llvm::StringRef expectedKind) {
  auto fn = lookupLLVMFunc(context, name);
  return fn && hasKind(fn.getOperation(), expectedKind);
}

} // namespace class_helper

static mlir::func::FuncOp lookupFunc(mlir::Operation *op,
                                     llvm::StringRef name) {
  auto module = op->getParentOfType<mlir::ModuleOp>();
  if (!module)
    return {};
  return module.lookupSymbol<mlir::func::FuncOp>(name);
}

static llvm::SmallVector<unsigned> getIndexArrayAttr(mlir::Operation *op,
                                                     llvm::StringRef attrName) {
  llvm::SmallVector<unsigned> result;
  if (!op)
    return result;
  auto array = op->getAttrOfType<mlir::ArrayAttr>(attrName);
  if (!array)
    return result;
  for (mlir::Attribute attr : array) {
    auto integer = mlir::dyn_cast<mlir::IntegerAttr>(attr);
    if (!integer || integer.getInt() < 0)
      continue;
    result.push_back(static_cast<unsigned>(integer.getInt()));
  }
  return result;
}

static std::optional<unsigned> getIndexAttr(mlir::Operation *op,
                                            llvm::StringRef attrName) {
  if (!op)
    return std::nullopt;
  auto attr = op->getAttrOfType<mlir::IntegerAttr>(attrName);
  if (!attr || attr.getInt() < 0)
    return std::nullopt;
  return static_cast<unsigned>(attr.getInt());
}

namespace constant {

bool smallInteger(mlir::Value value) {
  llvm::APInt intValue;
  if (mlir::matchPattern(value, mlir::m_ConstantInt(&intValue))) {
    int64_t raw = intValue.getSExtValue();
    return raw >= -5 && raw <= 256;
  }

  auto constant = value.getDefiningOp<mlir::LLVM::ConstantOp>();
  if (!constant)
    return false;
  auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue());
  if (!intAttr)
    return false;
  int64_t raw = intAttr.getInt();
  return raw >= -5 && raw <= 256;
}

} // namespace constant

namespace borrow {

bool immortal(mlir::Value value) {
  if (auto intToPtr = value.getDefiningOp<mlir::LLVM::IntToPtrOp>())
    return immortal(intToPtr.getArg());
  if (auto ptrToInt = value.getDefiningOp<mlir::LLVM::PtrToIntOp>())
    return immortal(ptrToInt.getArg());
  if (auto bitcast = value.getDefiningOp<mlir::LLVM::BitcastOp>())
    return immortal(bitcast.getArg());
  auto call = value.getDefiningOp<mlir::LLVM::CallOp>();
  if (!call)
    return false;
  auto callee = call.getCallee();
  if (!callee)
    return false;
  if (*callee == RuntimeSymbols::kGetNone)
    return true;
  if (*callee == RuntimeSymbols::kBoolFromBool)
    return true;
  if (*callee == RuntimeSymbols::kStrInternStaticUtf8)
    return true;
  if (*callee == RuntimeSymbols::kBuiltinPrintImpl)
    return true;
  if (*callee == RuntimeSymbols::kGetBuiltinPrint)
    return true;
  if (callee_kind::runtimeComparison(*callee))
    return true;
  return *callee == RuntimeSymbols::kLongFromI64 &&
         call.getNumOperands() == 1 &&
         constant::smallInteger(call.getOperand(0));
}

bool runtimeResult(mlir::Value value) {
  auto call = value.getDefiningOp<mlir::LLVM::CallOp>();
  if (!call)
    return false;
  auto callee = call.getCallee();
  if (!callee)
    return false;
  return *callee == RuntimeSymbols::kEHCapture ||
         *callee == RuntimeSymbols::kExceptionGetCurrent;
}

bool nonObject(mlir::Value value);

} // namespace borrow

namespace aggregate_slot {

bool transferContract(mlir::Operation *op) {
  return op && op->hasAttr(OwnershipContractAttrs::kMemRefSlotTransfer) &&
         op->hasAttr(OwnershipContractAttrs::kAggregateSlotGroup) &&
         op->hasAttr(OwnershipContractAttrs::kAggregateSlotComponent);
}

bool loadContract(mlir::Operation *op) {
  return op && op->hasAttr(OwnershipContractAttrs::kAggregateSlotLoad) &&
         op->hasAttr(OwnershipContractAttrs::kAggregateSlotGroup) &&
         op->hasAttr(OwnershipContractAttrs::kAggregateSlotComponent);
}

bool loaded(mlir::Value value) {
  if (auto intToPtr = value.getDefiningOp<mlir::LLVM::IntToPtrOp>())
    return loaded(intToPtr.getArg());
  if (auto bitcast = value.getDefiningOp<mlir::LLVM::BitcastOp>())
    return loaded(bitcast.getArg());

  if (auto load = value.getDefiningOp<mlir::LLVM::LoadOp>()) {
    auto role =
        load->getAttrOfType<mlir::StringAttr>(ThreadSafetyAttrs::kAtomicRole);
    if (role && role.getValue() == ThreadSafetyAttrs::kRoleAsyncExceptionLoad)
      return true;
    return loadContract(load.getOperation());
  }
  if (auto load = value.getDefiningOp<mlir::memref::LoadOp>())
    return loadContract(load.getOperation());
  return false;
}

bool releaseLoaded(mlir::Value value) {
  if (loaded(value))
    return true;
  if (auto intToPtr = value.getDefiningOp<mlir::LLVM::IntToPtrOp>())
    return releaseLoaded(intToPtr.getArg());
  if (auto bitcast = value.getDefiningOp<mlir::LLVM::BitcastOp>())
    return releaseLoaded(bitcast.getArg());
  return false;
}

bool priorTransferInBlock(mlir::LLVM::CallOp call, mlir::Value value,
                          const AliasAnalysis &aliases) {
  if (!aliases.isCarrier(value))
    return false;
  mlir::Value root = aliases.getRoot(value);
  for (mlir::Operation *previous = call->getPrevNode(); previous;
       previous = previous->getPrevNode()) {
    if (auto store = mlir::dyn_cast<mlir::LLVM::StoreOp>(previous)) {
      if (!transferContract(store.getOperation()))
        continue;
      mlir::Value stored = store.getValue();
      if (aliases.isCarrier(stored) && aliases.getRoot(stored) == root)
        return true;
    }
    if (auto store = mlir::dyn_cast<mlir::memref::StoreOp>(previous)) {
      if (!transferContract(store.getOperation()))
        continue;
      mlir::Value stored = store.getValue();
      if (aliases.isCarrier(stored) && aliases.getRoot(stored) == root)
        return true;
    }
  }
  return false;
}

bool loadedAlias(mlir::Value value, const AliasAnalysis &aliases);

} // namespace aggregate_slot

namespace aggregate_call {

mlir::LogicalResult verify(mlir::LLVM::CallOp call,
                           const AliasAnalysis &aliases) {
  bool aggregateRetain =
      call->hasAttr(OwnershipContractAttrs::kAggregateRetain);
  bool aggregateRelease =
      call->hasAttr(OwnershipContractAttrs::kAggregateRelease);
  if (!aggregateRetain && !aggregateRelease)
    return mlir::success();
  if (aggregateRetain && aggregateRelease)
    return call->emitOpError(
        "cannot carry both aggregate retain and aggregate release contracts");

  auto callee = call.getCallee();
  if (!callee)
    return call->emitOpError("aggregate ownership call must be direct");
  llvm::StringRef name = *callee;
  if (call.getNumOperands() != 1)
    return call->emitOpError("aggregate ownership call must have one operand");

  mlir::Value value = call.getOperand(0);
  if (!aliases.isCarrier(value))
    return call->emitOpError(
        "aggregate ownership operand must carry object ownership");

  if (aggregateRetain) {
    if (!callee_kind::retain(name) &&
        !class_helper::calleeHasKind(call.getOperation(), name,
                                     ClassSafetyAttrs::kKindIncref))
      return call->emitOpError("aggregate retain must call a retain helper");
    return mlir::success();
  }

  if (!callee_kind::release(name) &&
      !class_helper::calleeHasKind(call.getOperation(), name,
                                   ClassSafetyAttrs::kKindDecref))
    return call->emitOpError("aggregate release must call a release helper");
  if (aggregate_slot::releaseLoaded(value))
    return mlir::success();
  return call->emitOpError(
      "aggregate release lacks resource-keyed aggregate load provenance");
}

} // namespace aggregate_call

static bool callReturnsOwnedReference(mlir::LLVM::CallOp call) {
  auto callee = call.getCallee();
  if (!callee)
    return false;
  llvm::StringRef name = *callee;

  if (runtime::Callee::alwaysOwnedResult(name))
    return true;

  if (!runtime::Callee::conditionalOwnedResult(name))
    return false;

  if (name == RuntimeSymbols::kLongFromI64)
    return call.getNumOperands() != 1 ||
           !constant::smallInteger(call.getOperand(0));

  if (name.starts_with("__ly_class_getfield_"))
    return call.getNumOperands() >= 2 &&
           constant::boolFalse(call.getOperand(1));

  return false;
}

struct LLVMOwnershipState {
  llvm::DenseMap<mlir::Value, int64_t> balance;

  bool empty() const { return balance.empty(); }
};

static mlir::LogicalResult addTrackedToken(LLVMOwnershipState &state,
                                           mlir::Value value, int64_t delta,
                                           mlir::Operation *op,
                                           const AliasAnalysis &aliases) {
  if (!aliases.isCarrier(value))
    return mlir::success();
  mlir::Value root = aliases.getRoot(value);
  int64_t &count = state.balance[root];
  count += delta;
  if (count < 0)
    return op->emitOpError("LLVM ownership balance became negative for value ")
           << value;
  if (count == 0)
    state.balance.erase(root);
  return mlir::success();
}

static mlir::LogicalResult releaseIfTracked(LLVMOwnershipState &state,
                                            mlir::Value value,
                                            mlir::Operation *op,
                                            const AliasAnalysis &aliases,
                                            bool requireTracked = false) {
  if (!aliases.isCarrier(value))
    return mlir::success();
  mlir::Value root = aliases.getRoot(value);
  if (!state.balance.contains(root)) {
    if (requireTracked)
      return op->emitOpError("releases value without a proven ownership token ")
             << value;
    return mlir::success();
  }
  return addTrackedToken(state, value, -1, op, aliases);
}

namespace aggregate_slot {

bool declaredSetFieldEntryArg(mlir::Operation *op, mlir::Value value,
                              const AliasAnalysis &aliases) {
  auto fn = op->getParentOfType<mlir::LLVM::LLVMFuncOp>();
  if (!fn || fn.getBody().empty())
    return false;
  auto valueArg = getIndexAttr(fn.getOperation(),
                               OwnershipContractAttrs::kSetFieldValueArg);
  if (!valueArg)
    return false;
  mlir::Block &entry = fn.getBody().front();
  if (*valueArg >= entry.getNumArguments())
    return false;
  return aliases.getRoot(value) ==
         aliases.getRoot(entry.getArgument(*valueArg));
}

} // namespace aggregate_slot

namespace aggregate_call {

bool internalRetain(mlir::LLVM::CallOp call, mlir::Value value,
                    const AliasAnalysis &aliases) {
  auto fn = call->getParentOfType<mlir::LLVM::LLVMFuncOp>();
  if (!fn)
    return false;
  mlir::Operation *fnOp = fn.getOperation();
  if (class_helper::hasKind(fnOp, ClassSafetyAttrs::kKindIncref) ||
      class_helper::hasKind(fnOp, ClassSafetyAttrs::kKindPromote))
    return true;
  if (class_helper::hasKind(fnOp, ClassSafetyAttrs::kKindSetField) &&
      aggregate_slot::declaredSetFieldEntryArg(call.getOperation(), value,
                                               aliases))
    return true;
  if (aggregate_slot::priorTransferInBlock(call, value, aliases))
    return true;
  return false;
}

} // namespace aggregate_call

namespace aggregate_slot {

mlir::LogicalResult releaseTransferToken(LLVMOwnershipState &state,
                                         mlir::Value value, mlir::Operation *op,
                                         const AliasAnalysis &aliases) {
  if (!aliases.isCarrier(value))
    return mlir::success();
  mlir::Value root = aliases.getRoot(value);
  if (state.balance.contains(root))
    return addTrackedToken(state, value, -1, op, aliases);
  if (declaredSetFieldEntryArg(op, value, aliases))
    return mlir::success();
  if (loaded(value))
    return mlir::success();
  return op->emitOpError("aggregate slot transfer stores value without a "
                         "proven ownership token ")
         << value;
}

} // namespace aggregate_slot

static mlir::LogicalResult
verifyBorrowUsePremise(mlir::Operation *op, mlir::Value value,
                       const LLVMOwnershipState &state, mlir::Block &entry,
                       const AliasAnalysis &aliases);

static bool isLocalScratchAddress(mlir::Value value) {
  value = value ? value : mlir::Value();
  if (!value)
    return false;
  if (auto bitcast = value.getDefiningOp<mlir::LLVM::BitcastOp>())
    return isLocalScratchAddress(bitcast.getArg());
  if (auto gep = value.getDefiningOp<mlir::LLVM::GEPOp>())
    return isLocalScratchAddress(gep.getBase());
  return value.getDefiningOp<mlir::LLVM::AllocaOp>() != nullptr;
}

static mlir::LogicalResult applyStoreOwnership(mlir::LLVM::StoreOp store,
                                               LLVMOwnershipState &state,
                                               mlir::Block &entry,
                                               const AliasAnalysis &aliases) {
  mlir::Value stored = store.getValue();
  if (!aliases.isCarrier(stored))
    return mlir::success();

  if (aggregate_slot::transferContract(store.getOperation()))
    return aggregate_slot::releaseTransferToken(state, stored, store, aliases);

  if (async_runtime::ValueStorage::isAddress(store.getAddr())) {
    if (borrow::immortal(stored)) {
      if (!aliases.isCarrier(stored))
        return mlir::success();
      if (!state.balance.contains(aliases.getRoot(stored)))
        return mlir::success();
    }
    return releaseIfTracked(state, stored, store, aliases,
                            /*requireTracked=*/true);
  }

  if (isLocalScratchAddress(store.getAddr()))
    return verifyBorrowUsePremise(store.getOperation(), stored, state, entry,
                                  aliases);

  if (borrow::nonObject(stored) || borrow::immortal(stored))
    return mlir::success();

  return store->emitOpError("stores object pointer without an explicit "
                            "ownership transfer contract for ")
         << stored;
}

static mlir::LogicalResult
applyMemRefStoreOwnership(mlir::memref::StoreOp store,
                          LLVMOwnershipState &state,
                          const AliasAnalysis &aliases) {
  if (!store->hasAttr(OwnershipContractAttrs::kMemRefSlotTransfer))
    return mlir::success();
  mlir::Value stored = store.getValue();
  if (!aliases.isCarrier(stored))
    return mlir::success();

  auto memrefType =
      mlir::dyn_cast<mlir::MemRefType>(store.getMemref().getType());
  if (!memrefType || memrefType.getRank() != 1)
    return store->emitOpError(
        "memref slot ownership transfer must target rank-1 storage");

  // Typed memref containers own reference-counted payload slots only on stores
  // explicitly marked by container lowering. Once a tracked token is written
  // into such a slot, aggregate destruction owns the matching release
  // obligation.
  return releaseIfTracked(state, stored, store, aliases,
                          /*requireTracked=*/true);
}

static mlir::LogicalResult
applyLLVMLoadOwnership(mlir::LLVM::LoadOp load, LLVMOwnershipState &state,
                       const AliasAnalysis &aliases) {
  if (!value_type::pointerLike(load.getResult().getType()))
    return mlir::success();
  if (!async_runtime::ValueStorage::isAddress(load.getAddr()))
    return mlir::success();

  // async.runtime.load is a transfer from the async value storage to the
  // awaiter. After ConvertAsyncToLLVM this appears as a raw load from
  // mlirAsyncRuntimeGetValueStorage(), so preserve the same ownership fact.
  return addTrackedToken(state, load.getResult(), +1, load, aliases);
}

static mlir::LogicalResult
applyAsyncPayloadLoadOwnership(mlir::Operation &op, LLVMOwnershipState &state,
                               const AliasAnalysis &aliases) {
  if (!mlir::isa<mlir::async::AwaitOp, mlir::async::RuntimeLoadOp>(&op))
    return mlir::success();

  for (mlir::Value result : op.getResults())
    if (value_type::pointerLike(result.getType()))
      if (mlir::failed(addTrackedToken(state, result, +1, &op, aliases)))
        return mlir::failure();
  return mlir::success();
}

static bool isEntryArgument(mlir::Value value, mlir::Block &entry,
                            const AliasAnalysis &aliases) {
  mlir::Value root = aliases.getRoot(value);
  if (isEntryBorrowedValue(root))
    return true;
  for (mlir::BlockArgument arg : entry.getArguments())
    if (aliases.getRoot(arg) == root)
      return true;
  return false;
}

static bool stateHasOwnershipToken(const LLVMOwnershipState &state,
                                   mlir::Value value,
                                   const AliasAnalysis &aliases) {
  if (!aliases.isCarrier(value))
    return true;
  return state.balance.contains(aliases.getRoot(value));
}

namespace borrow {

bool nonObject(mlir::Value value) {
  if (!value)
    return true;
  if (function_arg::hasAttr(value, AsyncSafetyAttrs::kExceptionCell) ||
      function_arg::hasAttr(value, AsyncSafetyAttrs::kCancelFlag) ||
      function_arg::hasAttr(value, AsyncSafetyAttrs::kRuntimeHandle))
    return true;
  if (mlir::Operation *def = value.getDefiningOp())
    if (def->hasAttr(OwnershipContractAttrs::kNonObjectPointer) ||
        def->hasAttr(AsyncSafetyAttrs::kExceptionCell) ||
        def->hasAttr(AsyncSafetyAttrs::kCancelFlag) ||
        def->hasAttr(AsyncSafetyAttrs::kRuntimeHandle))
      return true;
  if (auto call = value.getDefiningOp<mlir::LLVM::CallOp>()) {
    auto callee = call.getCallee();
    if (callee && callee_kind::nonObjectAllocator(*callee))
      return true;
    if (callee && (async_runtime::Callee::known(*callee) ||
                   async_runtime::Callee::executeFunction(*callee)))
      return true;
  }
  if (auto call = value.getDefiningOp<mlir::func::CallOp>()) {
    llvm::StringRef callee = call.getCallee();
    if (callee_kind::nonObjectAllocator(callee) ||
        async_runtime::Callee::known(callee) ||
        async_runtime::Callee::executeFunction(callee))
      return true;
  }
  if (auto bitcast = value.getDefiningOp<mlir::LLVM::BitcastOp>())
    return nonObject(bitcast.getArg());
  if (auto gep = value.getDefiningOp<mlir::LLVM::GEPOp>())
    return nonObject(gep.getBase());
  if (value.getDefiningOp<mlir::LLVM::AddressOfOp>())
    return true;
  if (value.getDefiningOp<mlir::LLVM::ZeroOp>())
    return true;
  if (auto extract = value.getDefiningOp<mlir::LLVM::ExtractValueOp>()) {
    if (aggregate_slot::loadContract(extract.getOperation()))
      return false;
    return nonObject(extract.getContainer());
  }
  if (async_runtime::ValueStorage::isAddress(value))
    return true;
  return false;
}

} // namespace borrow

static mlir::LogicalResult
verifyBorrowUsePremise(mlir::Operation *op, mlir::Value value,
                       const LLVMOwnershipState &state, mlir::Block &entry,
                       const AliasAnalysis &aliases) {
  if (!aliases.isCarrier(value))
    return mlir::success();
  if (borrow::nonObject(value))
    return mlir::success();
  if (stateHasOwnershipToken(state, value, aliases))
    return mlir::success();
  if (isEntryArgument(value, entry, aliases))
    return mlir::success();
  if (aggregate_slot::loadedAlias(value, aliases))
    return mlir::success();
  if (borrow::immortal(value))
    return mlir::success();
  if (borrow::runtimeResult(value))
    return mlir::success();
  return op->emitOpError("borrow use lacks a live ownership token or explicit "
                         "borrow provenance for ")
         << value;
}

namespace aggregate_slot {

bool loadedAlias(mlir::Value value, const AliasAnalysis &aliases) {
  llvm::SmallVector<mlir::Value, 4> aliasValues;
  aliases.collectAliases(value, aliasValues);
  return llvm::any_of(aliasValues,
                      [](mlir::Value alias) { return loaded(alias); });
}

} // namespace aggregate_slot

static mlir::LogicalResult verifyRetainOwnedTokenPremise(
    mlir::LLVM::CallOp call, mlir::Value value, const LLVMOwnershipState &state,
    mlir::Block &entry, const AliasAnalysis &aliases) {
  auto premise =
      call->getAttrOfType<mlir::StringAttr>(ThreadSafetyAttrs::kRetainPremise);
  if (!premise || premise.getValue() != ThreadSafetyAttrs::kPremiseOwnedToken)
    return mlir::success();
  if (borrow::immortal(value))
    return mlir::success();
  if (stateHasOwnershipToken(state, value, aliases))
    return mlir::success();
  return call->emitOpError(
             "owned-token retain premise lacks a proven ownership token for ")
         << value;
}

static mlir::LogicalResult
verifyAggregateBorrowRetainPremise(mlir::LLVM::CallOp call, mlir::Value value,
                                   const AliasAnalysis &aliases) {
  auto premise =
      call->getAttrOfType<mlir::StringAttr>(ThreadSafetyAttrs::kRetainPremise);
  if (!premise ||
      premise.getValue() != ThreadSafetyAttrs::kPremiseAggregateBorrow)
    return mlir::success();
  if (aggregate_slot::loadedAlias(value, aliases))
    return mlir::success();
  return call->emitOpError("aggregate-borrow retain premise must originate "
                           "from an aggregate slot or exception-cell load for ")
         << value;
}

static mlir::LogicalResult
verifyEntryBorrowedRetainPremise(mlir::LLVM::CallOp call, mlir::Value value,
                                 mlir::Block &entry,
                                 const AliasAnalysis &aliases) {
  auto premise =
      call->getAttrOfType<mlir::StringAttr>(ThreadSafetyAttrs::kRetainPremise);
  if (!premise ||
      premise.getValue() != ThreadSafetyAttrs::kPremiseEntryBorrowed)
    return mlir::success();
  if (isEntryArgument(value, entry, aliases))
    return mlir::success();
  return call->emitOpError("entry-borrowed retain premise does not originate "
                           "from an entry borrowed argument for ")
         << value;
}

static mlir::LogicalResult verifyRetainPremise(mlir::LLVM::CallOp call,
                                               mlir::Value value,
                                               const LLVMOwnershipState &state,
                                               mlir::Block &entry,
                                               const AliasAnalysis &aliases) {
  if (mlir::failed(
          verifyRetainOwnedTokenPremise(call, value, state, entry, aliases)))
    return mlir::failure();
  if (mlir::failed(
          verifyEntryBorrowedRetainPremise(call, value, entry, aliases)))
    return mlir::failure();
  if (mlir::failed(verifyAggregateBorrowRetainPremise(call, value, aliases)))
    return mlir::failure();
  return mlir::success();
}

namespace aggregate_call {

mlir::LogicalResult verifyRetainPremise(mlir::LLVM::CallOp call,
                                        mlir::Value value,
                                        const LLVMOwnershipState &state,
                                        mlir::Block &entry,
                                        const AliasAnalysis &aliases) {
  auto premise =
      call->getAttrOfType<mlir::StringAttr>(ThreadSafetyAttrs::kRetainPremise);
  if (!premise)
    return call->emitOpError("aggregate retain is missing a retain premise");
  if (premise.getValue() == ThreadSafetyAttrs::kPremiseLockedBorrow) {
    if (aggregate_slot::loadedAlias(value, aliases))
      return mlir::success();
    return call->emitOpError("locked-borrow aggregate retain must originate "
                             "from an aggregate slot load");
  }
  if (premise.getValue() == ThreadSafetyAttrs::kPremiseAggregateBorrow)
    return verifyAggregateBorrowRetainPremise(call, value, aliases);
  if (premise.getValue() == ThreadSafetyAttrs::kPremiseOwnedToken)
    return verifyRetainOwnedTokenPremise(call, value, state, entry, aliases);
  if (premise.getValue() == ThreadSafetyAttrs::kPremiseEntryBorrowed)
    return verifyEntryBorrowedRetainPremise(call, value, entry, aliases);
  return call->emitOpError("aggregate retain has unsupported premise: ")
         << premise.getValue();
}

} // namespace aggregate_call

static bool indexIn(llvm::ArrayRef<unsigned> indices, unsigned index) {
  return llvm::is_contained(indices, index);
}

static bool callHasCarrierOperand(mlir::LLVM::CallOp call,
                                  const AliasAnalysis &aliases) {
  for (mlir::Value operand : call.getOperands())
    if (aliases.isCarrier(operand) && !borrow::nonObject(operand))
      return true;
  return false;
}

static bool callHasCarrierResult(mlir::LLVM::CallOp call) {
  for (mlir::Value result : call->getResults())
    if (value_type::pointerLike(result.getType()) &&
        !borrow::nonObject(result) && !borrow::immortal(result) &&
        !borrow::runtimeResult(result))
      return true;
  return false;
}

static bool invokeHasCarrierOperand(mlir::LLVM::InvokeOp invoke,
                                    const AliasAnalysis &aliases) {
  for (mlir::Value operand : invoke.getCalleeOperands())
    if (aliases.isCarrier(operand) && !borrow::nonObject(operand))
      return true;
  return false;
}

static bool invokeHasCarrierResult(mlir::LLVM::InvokeOp invoke) {
  for (mlir::Value result : invoke->getResults())
    if (value_type::pointerLike(result.getType()) &&
        !borrow::nonObject(result) && !borrow::immortal(result) &&
        !borrow::runtimeResult(result))
      return true;
  return false;
}

static bool isExternalDeclaration(mlir::Operation *op);
static bool isClosedExternalOwnershipCallee(llvm::StringRef name,
                                            mlir::Operation *calleeOp);

static mlir::LogicalResult
verifyUnknownCallContract(mlir::LLVM::CallOp call, mlir::Operation *calleeOp,
                          const AliasAnalysis &aliases) {
  auto callee = call.getCallee();
  bool hasCarrier =
      callHasCarrierOperand(call, aliases) || callHasCarrierResult(call);
  if (!callee) {
    if (!hasCarrier)
      return mlir::success();
    return call->emitOpError("indirect ownership-carrying call requires an "
                             "explicit contract");
  }
  if (async_runtime::Handle::isCallee(*callee, calleeOp))
    return mlir::success();
  if (calleeOp && !isExternalDeclaration(calleeOp))
    return mlir::success();
  if (isClosedExternalOwnershipCallee(*callee, calleeOp))
    return mlir::success();
  if (!hasCarrier)
    return mlir::success();
  return call->emitOpError("ownership-carrying external call is not in the "
                           "closed ownership/runtime contract table: ")
         << *callee;
}

static mlir::LogicalResult verifyCallBorrowUses(mlir::LLVM::CallOp call,
                                                const LLVMOwnershipState &state,
                                                mlir::Block &entry,
                                                const AliasAnalysis &aliases,
                                                mlir::Operation *calleeOp) {
  auto callee = call.getCallee();
  if (!callee)
    return mlir::success();
  if (async_runtime::Handle::isCallee(*callee, calleeOp))
    return mlir::success();
  if (callee_kind::nonObjectAllocator(*callee))
    return mlir::success();

  llvm::SmallVector<unsigned> releaseArgs =
      getIndexArrayAttr(calleeOp, OwnershipContractAttrs::kReleaseArgs);
  llvm::SmallVector<unsigned> transferArgs =
      getIndexArrayAttr(calleeOp, OwnershipContractAttrs::kTransferArgs);
  llvm::SmallVector<unsigned> retainArgs =
      getIndexArrayAttr(calleeOp, OwnershipContractAttrs::kRetainArgs);
  auto setFieldValueArg =
      getIndexAttr(calleeOp, OwnershipContractAttrs::kSetFieldValueArg);
  auto setFieldRetainArg =
      getIndexAttr(calleeOp, OwnershipContractAttrs::kSetFieldRetainArg);
  auto calleeArgNonObject = [&](unsigned argIndex) {
    auto calleeFunc =
        mlir::dyn_cast_or_null<mlir::FunctionOpInterface>(calleeOp);
    return calleeFunc &&
           static_cast<unsigned>(calleeFunc.getNumArguments()) > argIndex &&
           calleeFunc.getArgAttr(argIndex,
                                 OwnershipContractAttrs::kNonObjectPointer);
  };

  bool shouldCheck = callHasCarrierOperand(call, aliases) ||
                     callHasCarrierResult(call) ||
                     callee_kind::runtimeLike(*callee) || calleeOp ||
                     call->hasAttr(OwnershipContractAttrs::kAggregateRetain) ||
                     call->hasAttr(OwnershipContractAttrs::kAggregateRelease) ||
                     call->hasAttr(OwnershipContractAttrs::kLocalDestroy);
  if (!shouldCheck)
    return mlir::success();

  for (auto [index, operand] : llvm::enumerate(call.getOperands())) {
    unsigned argIndex = static_cast<unsigned>(index);
    if (!aliases.isCarrier(operand))
      continue;
    if (indexIn(releaseArgs, argIndex) || indexIn(transferArgs, argIndex))
      continue;
    if (setFieldValueArg && *setFieldValueArg == argIndex &&
        setFieldRetainArg && *setFieldRetainArg < call.getNumOperands() &&
        constant::boolFalse(call.getOperand(*setFieldRetainArg)))
      continue;
    if (calleeArgNonObject(argIndex))
      continue;
    if (indexIn(retainArgs, argIndex))
      continue;
    if (mlir::failed(verifyBorrowUsePremise(call.getOperation(), operand, state,
                                            entry, aliases)))
      return mlir::failure();
  }
  return mlir::success();
}

static mlir::LogicalResult verifyLocalDestroyCall(mlir::LLVM::CallOp call,
                                                  llvm::StringRef calleeName) {
  if (!call->hasAttr(OwnershipContractAttrs::kLocalDestroy))
    return mlir::success();
  if (!callee_kind::localDestroy(calleeName))
    return call->emitOpError("local_destroy contract must call a destroy_local "
                             "class helper");
  if (call.getNumOperands() != 1)
    return call->emitOpError("local_destroy contract must have one operand");
  auto calleeFunc = lookupLLVMFunc(call.getOperation(), calleeName);
  if (!calleeFunc ||
      !class_helper::hasKind(calleeFunc.getOperation(),
                             ClassSafetyAttrs::kKindDestroyLocal))
    return call->emitOpError("local_destroy callee lacks destroy_local helper "
                             "metadata");
  return mlir::success();
}

static mlir::Operation *lookupOwnershipCalleeOp(mlir::LLVM::CallOp call,
                                                llvm::StringRef name);

static bool isBorrowedCallResult(mlir::Value value,
                                 const AliasAnalysis &aliases) {
  llvm::SmallVector<mlir::Value, 4> values;
  aliases.collectAliases(value, values);
  values.push_back(value);
  for (mlir::Value candidate : values) {
    auto result = mlir::dyn_cast<mlir::OpResult>(candidate);
    if (!result)
      continue;
    auto call = mlir::dyn_cast<mlir::LLVM::CallOp>(result.getOwner());
    if (!call)
      continue;
    auto callee = call.getCallee();
    if (!callee)
      continue;
    mlir::Operation *calleeOp = lookupOwnershipCalleeOp(call, *callee);
    auto borrowedResults =
        getIndexArrayAttr(calleeOp, OwnershipContractAttrs::kBorrowedResults);
    if (indexIn(borrowedResults, result.getResultNumber()))
      return true;
  }
  return false;
}

static mlir::LogicalResult
consumeReturnOperand(LLVMOwnershipState &state, mlir::Value value,
                     mlir::Operation *op, unsigned resultIndex,
                     mlir::Block &entry, const AliasAnalysis &aliases) {
  if (!aliases.isCarrier(value))
    return mlir::success();
  mlir::Value root = aliases.getRoot(value);
  if (state.balance.contains(root))
    return addTrackedToken(state, value, -1, op, aliases);
  if (borrow::nonObject(value) || borrow::immortal(value))
    return mlir::success();
  if (auto fn = op->getParentOfType<mlir::LLVM::LLVMFuncOp>()) {
    auto borrowedResults = getIndexArrayAttr(
        fn.getOperation(), OwnershipContractAttrs::kBorrowedResults);
    if (indexIn(borrowedResults, resultIndex) &&
        isBorrowedCallResult(value, aliases))
      return mlir::success();
    if (class_helper::hasKind(fn.getOperation(),
                              ClassSafetyAttrs::kKindGetField) &&
        aggregate_slot::loadedAlias(value, aliases))
      return mlir::success();
  }
  if (isEntryArgument(value, entry, aliases))
    return op->emitOpError("returns borrowed entry pointer without a proven "
                           "retain: ")
           << value;
  if (aggregate_slot::loadedAlias(value, aliases) ||
      borrow::runtimeResult(value))
    return op->emitOpError("returns borrowed object pointer without a proven "
                           "retain: ")
           << value;
  return op->emitOpError("returns object pointer without a proven ownership "
                         "token or immortal provenance: ")
         << value;
}

static mlir::Operation *lookupOwnershipCalleeOp(mlir::LLVM::CallOp call,
                                                llvm::StringRef name) {
  if (auto calleeFunc = lookupLLVMFunc(call.getOperation(), name))
    return calleeFunc.getOperation();
  if (auto directFunc = lookupFunc(call.getOperation(), name))
    return directFunc.getOperation();
  return nullptr;
}

static mlir::Operation *lookupOwnershipCalleeOp(mlir::LLVM::InvokeOp invoke,
                                                llvm::StringRef name) {
  if (auto calleeFunc = lookupLLVMFunc(invoke.getOperation(), name))
    return calleeFunc.getOperation();
  if (auto directFunc = lookupFunc(invoke.getOperation(), name))
    return directFunc.getOperation();
  return nullptr;
}

static bool isExternalDeclaration(mlir::Operation *op) {
  if (auto fn = mlir::dyn_cast_or_null<mlir::LLVM::LLVMFuncOp>(op))
    return fn.isExternal();
  if (auto fn = mlir::dyn_cast_or_null<mlir::func::FuncOp>(op))
    return fn.isExternal();
  return false;
}

static bool hasAnyCalleeOwnershipContract(mlir::Operation *op) {
  if (!op)
    return false;
  return op->hasAttr(OwnershipContractAttrs::kRetainArgs) ||
         op->hasAttr(OwnershipContractAttrs::kReleaseArgs) ||
         op->hasAttr(OwnershipContractAttrs::kTransferArgs) ||
         op->hasAttr(OwnershipContractAttrs::kOwnedResults) ||
         op->hasAttr(OwnershipContractAttrs::kBorrowedResults) ||
         op->hasAttr(OwnershipContractAttrs::kSetFieldValueArg) ||
         op->hasAttr(OwnershipContractAttrs::kSetFieldRetainArg) ||
         op->hasAttr(OwnershipContractAttrs::kGetFieldBorrowArg) ||
         op->hasAttr(OwnershipContractAttrs::kGetFieldOwnedResult);
}

static bool isClosedExternalOwnershipCallee(llvm::StringRef name,
                                            mlir::Operation *calleeOp) {
  return hasAnyCalleeOwnershipContract(calleeOp) || callee_kind::retain(name) ||
         callee_kind::release(name) || callee_kind::transfer(name) ||
         callee_kind::setField(name) || callee_kind::runtimeBorrowOnly(name) ||
         callee_kind::asyncRewriteMarker(name) ||
         runtime::Callee::alwaysOwnedResult(name) ||
         runtime::Callee::conditionalOwnedResult(name) ||
         callee_kind::nonObjectAllocator(name);
}

static mlir::LogicalResult applyCallOwnership(mlir::LLVM::CallOp call,
                                              LLVMOwnershipState &state,
                                              mlir::Block &entry,
                                              const AliasAnalysis &aliases) {
  auto callee = call.getCallee();
  if (!callee)
    return mlir::success();
  llvm::StringRef name = *callee;
  if (callee_kind::nonObjectAllocator(name))
    return mlir::success();
  mlir::Operation *calleeOp = lookupOwnershipCalleeOp(call, name);
  if (async_runtime::Handle::isCallee(name, calleeOp))
    return mlir::success();

  if (mlir::failed(verifyUnknownCallContract(call, calleeOp, aliases)))
    return mlir::failure();

  if (mlir::failed(verifyLocalDestroyCall(call, name)))
    return mlir::failure();
  if (call->hasAttr(OwnershipContractAttrs::kLocalDestroy)) {
    if (call.getNumOperands() >= 1) {
      mlir::Value object = call.getOperand(0);
      if (borrow::nonObject(object) || borrow::immortal(object))
        return mlir::success();
      return releaseIfTracked(state, object, call, aliases,
                              /*requireTracked=*/true);
    }
    return mlir::success();
  }

  if (mlir::failed(verifyCallBorrowUses(call, state, entry, aliases, calleeOp)))
    return mlir::failure();

  if (call->hasAttr(OwnershipContractAttrs::kAggregateRetain) ||
      call->hasAttr(OwnershipContractAttrs::kAggregateRelease)) {
    if (mlir::failed(aggregate_call::verify(call, aliases)))
      return mlir::failure();
    if (call->hasAttr(OwnershipContractAttrs::kAggregateRetain)) {
      mlir::Value value = call.getOperand(0);
      if (mlir::failed(aggregate_call::verifyRetainPremise(call, value, state,
                                                           entry, aliases)))
        return mlir::failure();
      if (aggregate_call::internalRetain(call, value, aliases))
        return mlir::success();
      return addTrackedToken(state, value, +1, call, aliases);
    }
    return mlir::success();
  }

  bool hasExplicitContract = false;
  for (unsigned argIndex :
       getIndexArrayAttr(calleeOp, OwnershipContractAttrs::kRetainArgs)) {
    hasExplicitContract = true;
    if (argIndex < call.getNumOperands()) {
      if (mlir::failed(verifyRetainPremise(call, call.getOperand(argIndex),
                                           state, entry, aliases)))
        return mlir::failure();
      if (mlir::failed(addTrackedToken(state, call.getOperand(argIndex), +1,
                                       call, aliases)))
        return mlir::failure();
    }
  }

  for (unsigned argIndex :
       getIndexArrayAttr(calleeOp, OwnershipContractAttrs::kReleaseArgs)) {
    hasExplicitContract = true;
    if (argIndex < call.getNumOperands())
      if (mlir::failed(releaseIfTracked(state, call.getOperand(argIndex), call,
                                        aliases, /*requireTracked=*/true)))
        return mlir::failure();
  }

  for (unsigned argIndex :
       getIndexArrayAttr(calleeOp, OwnershipContractAttrs::kTransferArgs)) {
    hasExplicitContract = true;
    if (argIndex < call.getNumOperands())
      if (mlir::failed(releaseIfTracked(state, call.getOperand(argIndex), call,
                                        aliases, /*requireTracked=*/true)))
        return mlir::failure();
  }

  auto setFieldValueArg =
      getIndexAttr(calleeOp, OwnershipContractAttrs::kSetFieldValueArg);
  auto setFieldRetainArg =
      getIndexAttr(calleeOp, OwnershipContractAttrs::kSetFieldRetainArg);
  if (setFieldValueArg && setFieldRetainArg) {
    hasExplicitContract = true;
    if (*setFieldValueArg < call.getNumOperands() &&
        *setFieldRetainArg < call.getNumOperands() &&
        constant::boolFalse(call.getOperand(*setFieldRetainArg)))
      if (mlir::failed(
              releaseIfTracked(state, call.getOperand(*setFieldValueArg), call,
                               aliases, /*requireTracked=*/true)))
        return mlir::failure();
  }

  auto getFieldBorrowArg =
      getIndexAttr(calleeOp, OwnershipContractAttrs::kGetFieldBorrowArg);
  auto getFieldOwnedResult =
      getIndexAttr(calleeOp, OwnershipContractAttrs::kGetFieldOwnedResult);
  if (getFieldBorrowArg && getFieldOwnedResult) {
    hasExplicitContract = true;
    if (*getFieldBorrowArg < call.getNumOperands() &&
        *getFieldOwnedResult < call->getNumResults() &&
        constant::boolFalse(call.getOperand(*getFieldBorrowArg))) {
      mlir::Value result = call->getResult(*getFieldOwnedResult);
      if (value_type::pointerLike(result.getType()) &&
          !borrow::immortal(result))
        if (mlir::failed(addTrackedToken(state, result, +1, call, aliases)))
          return mlir::failure();
    }
  }

  for (unsigned resultIndex :
       getIndexArrayAttr(calleeOp, OwnershipContractAttrs::kOwnedResults)) {
    hasExplicitContract = true;
    if (resultIndex < call->getNumResults()) {
      mlir::Value result = call->getResult(resultIndex);
      if (value_type::pointerLike(result.getType()))
        if (mlir::failed(addTrackedToken(state, result, +1, call, aliases)))
          return mlir::failure();
    }
  }

  for (unsigned resultIndex :
       getIndexArrayAttr(calleeOp, OwnershipContractAttrs::kBorrowedResults)) {
    hasExplicitContract = true;
    if (resultIndex < call->getNumResults()) {
      mlir::Value result = call->getResult(resultIndex);
      if (!value_type::pointerLike(result.getType()))
        return call->emitOpError(
            "borrowed-result contract targets a non-pointer result");
    }
  }

  if (hasExplicitContract)
    return mlir::success();

  if (callee_kind::retain(name)) {
    if (call.getNumOperands() >= 1) {
      if (mlir::failed(verifyRetainPremise(call, call.getOperand(0), state,
                                           entry, aliases)))
        return mlir::failure();
      return addTrackedToken(state, call.getOperand(0), +1, call, aliases);
    }
    return mlir::success();
  }

  if (callee_kind::release(name) || callee_kind::transfer(name)) {
    if (call.getNumOperands() >= 1)
      return releaseIfTracked(state, call.getOperand(0), call, aliases,
                              /*requireTracked=*/true);
    return mlir::success();
  }

  if (callee_kind::setField(name)) {
    if (call.getNumOperands() >= 3 && constant::boolFalse(call.getOperand(2)))
      return releaseIfTracked(state, call.getOperand(1), call, aliases,
                              /*requireTracked=*/true);
    return mlir::success();
  }

  if (!callReturnsOwnedReference(call)) {
    if (callHasCarrierResult(call))
      return call->emitOpError("ownership-carrying call result lacks an "
                               "explicit owned-result, borrowed-result, or "
                               "non-object contract");
    return mlir::success();
  }
  for (mlir::Value result : call->getResults())
    if (value_type::pointerLike(result.getType()) && !borrow::immortal(result))
      if (mlir::failed(addTrackedToken(state, result, +1, call, aliases)))
        return mlir::failure();
  return mlir::success();
}

static mlir::LogicalResult
verifyUnknownInvokeContract(mlir::LLVM::InvokeOp invoke,
                            mlir::Operation *calleeOp,
                            const AliasAnalysis &aliases) {
  auto callee = invoke.getCallee();
  bool hasCarrier = invokeHasCarrierOperand(invoke, aliases) ||
                    invokeHasCarrierResult(invoke);
  if (!callee) {
    if (!hasCarrier)
      return mlir::success();
    return invoke->emitOpError("indirect ownership-carrying invoke requires an "
                               "explicit contract");
  }
  if (async_runtime::Handle::isCallee(*callee, calleeOp))
    return mlir::success();
  if (calleeOp && !isExternalDeclaration(calleeOp))
    return mlir::success();
  if (isClosedExternalOwnershipCallee(*callee, calleeOp))
    return mlir::success();
  if (!hasCarrier)
    return mlir::success();
  return invoke->emitOpError("ownership-carrying external invoke is not in the "
                             "closed ownership/runtime contract table: ")
         << *callee;
}

static mlir::LogicalResult
verifyInvokeBorrowUses(mlir::LLVM::InvokeOp invoke,
                       const LLVMOwnershipState &state, mlir::Block &entry,
                       const AliasAnalysis &aliases,
                       mlir::Operation *calleeOp) {
  auto callee = invoke.getCallee();
  if (!callee)
    return mlir::success();
  if (async_runtime::Handle::isCallee(*callee, calleeOp))
    return mlir::success();
  if (callee_kind::nonObjectAllocator(*callee))
    return mlir::success();

  llvm::SmallVector<unsigned> releaseArgs =
      getIndexArrayAttr(calleeOp, OwnershipContractAttrs::kReleaseArgs);
  llvm::SmallVector<unsigned> transferArgs =
      getIndexArrayAttr(calleeOp, OwnershipContractAttrs::kTransferArgs);
  llvm::SmallVector<unsigned> retainArgs =
      getIndexArrayAttr(calleeOp, OwnershipContractAttrs::kRetainArgs);
  auto setFieldValueArg =
      getIndexAttr(calleeOp, OwnershipContractAttrs::kSetFieldValueArg);
  auto setFieldRetainArg =
      getIndexAttr(calleeOp, OwnershipContractAttrs::kSetFieldRetainArg);
  auto calleeArgNonObject = [&](unsigned argIndex) {
    auto calleeFunc =
        mlir::dyn_cast_or_null<mlir::FunctionOpInterface>(calleeOp);
    return calleeFunc &&
           static_cast<unsigned>(calleeFunc.getNumArguments()) > argIndex &&
           calleeFunc.getArgAttr(argIndex,
                                 OwnershipContractAttrs::kNonObjectPointer);
  };

  bool shouldCheck = invokeHasCarrierOperand(invoke, aliases) ||
                     invokeHasCarrierResult(invoke) ||
                     callee_kind::runtimeLike(*callee) || calleeOp;
  if (!shouldCheck)
    return mlir::success();

  mlir::OperandRange operands = invoke.getCalleeOperands();
  for (auto [index, operand] : llvm::enumerate(operands)) {
    unsigned argIndex = static_cast<unsigned>(index);
    if (!aliases.isCarrier(operand))
      continue;
    if (indexIn(releaseArgs, argIndex) || indexIn(transferArgs, argIndex))
      continue;
    if (setFieldValueArg && *setFieldValueArg == argIndex &&
        setFieldRetainArg && *setFieldRetainArg < operands.size() &&
        constant::boolFalse(operands[*setFieldRetainArg]))
      continue;
    if (calleeArgNonObject(argIndex))
      continue;
    if (indexIn(retainArgs, argIndex))
      continue;
    if (mlir::failed(verifyBorrowUsePremise(invoke.getOperation(), operand,
                                            state, entry, aliases)))
      return mlir::failure();
  }
  return mlir::success();
}

static bool invokeReturnsOwnedReference(mlir::LLVM::InvokeOp invoke) {
  auto callee = invoke.getCallee();
  if (!callee)
    return false;
  llvm::StringRef name = *callee;

  if (runtime::Callee::alwaysOwnedResult(name))
    return true;

  if (!runtime::Callee::conditionalOwnedResult(name))
    return false;

  mlir::OperandRange operands = invoke.getCalleeOperands();
  if (name == RuntimeSymbols::kLongFromI64)
    return operands.size() != 1 || !constant::smallInteger(operands.front());

  if (name.starts_with("__ly_class_getfield_"))
    return operands.size() >= 2 && constant::boolFalse(operands[1]);

  return false;
}

static mlir::LogicalResult
applyInvokeOperandOwnership(mlir::LLVM::InvokeOp invoke,
                            LLVMOwnershipState &state, mlir::Block &entry,
                            const AliasAnalysis &aliases) {
  auto callee = invoke.getCallee();
  if (!callee)
    return mlir::success();
  llvm::StringRef name = *callee;
  if (callee_kind::nonObjectAllocator(name))
    return mlir::success();
  mlir::Operation *calleeOp = lookupOwnershipCalleeOp(invoke, name);
  if (async_runtime::Handle::isCallee(name, calleeOp))
    return mlir::success();

  if (mlir::failed(verifyUnknownInvokeContract(invoke, calleeOp, aliases)))
    return mlir::failure();
  if (mlir::failed(
          verifyInvokeBorrowUses(invoke, state, entry, aliases, calleeOp)))
    return mlir::failure();

  mlir::OperandRange operands = invoke.getCalleeOperands();
  for (unsigned argIndex :
       getIndexArrayAttr(calleeOp, OwnershipContractAttrs::kRetainArgs)) {
    if (argIndex < operands.size())
      if (mlir::failed(
              addTrackedToken(state, operands[argIndex], +1, invoke, aliases)))
        return mlir::failure();
  }

  for (unsigned argIndex :
       getIndexArrayAttr(calleeOp, OwnershipContractAttrs::kReleaseArgs)) {
    if (argIndex < operands.size())
      if (mlir::failed(releaseIfTracked(state, operands[argIndex], invoke,
                                        aliases, /*requireTracked=*/true)))
        return mlir::failure();
  }

  for (unsigned argIndex :
       getIndexArrayAttr(calleeOp, OwnershipContractAttrs::kTransferArgs)) {
    if (argIndex < operands.size())
      if (mlir::failed(releaseIfTracked(state, operands[argIndex], invoke,
                                        aliases, /*requireTracked=*/true)))
        return mlir::failure();
  }

  auto setFieldValueArg =
      getIndexAttr(calleeOp, OwnershipContractAttrs::kSetFieldValueArg);
  auto setFieldRetainArg =
      getIndexAttr(calleeOp, OwnershipContractAttrs::kSetFieldRetainArg);
  if (setFieldValueArg && setFieldRetainArg &&
      *setFieldValueArg < operands.size() &&
      *setFieldRetainArg < operands.size() &&
      constant::boolFalse(operands[*setFieldRetainArg]))
    if (mlir::failed(releaseIfTracked(state, operands[*setFieldValueArg],
                                      invoke, aliases,
                                      /*requireTracked=*/true)))
      return mlir::failure();

  if (callee_kind::retain(name)) {
    if (!operands.empty())
      return addTrackedToken(state, operands.front(), +1, invoke, aliases);
    return mlir::success();
  }

  if (callee_kind::release(name) || callee_kind::transfer(name)) {
    if (!operands.empty())
      return releaseIfTracked(state, operands.front(), invoke, aliases,
                              /*requireTracked=*/true);
    return mlir::success();
  }

  if (callee_kind::setField(name)) {
    if (operands.size() >= 3 && constant::boolFalse(operands[2]))
      return releaseIfTracked(state, operands[1], invoke, aliases,
                              /*requireTracked=*/true);
    return mlir::success();
  }

  if (!invokeReturnsOwnedReference(invoke) && invokeHasCarrierResult(invoke) &&
      getIndexArrayAttr(calleeOp, OwnershipContractAttrs::kOwnedResults)
          .empty() &&
      getIndexArrayAttr(calleeOp, OwnershipContractAttrs::kBorrowedResults)
          .empty())
    return invoke->emitOpError("ownership-carrying invoke result lacks an "
                               "explicit owned-result, borrowed-result, or "
                               "non-object contract");

  return mlir::success();
}

static mlir::LogicalResult
addInvokeNormalResultOwnership(mlir::LLVM::InvokeOp invoke,
                               LLVMOwnershipState &state,
                               const AliasAnalysis &aliases) {
  auto callee = invoke.getCallee();
  if (!callee)
    return mlir::success();
  mlir::Operation *calleeOp = lookupOwnershipCalleeOp(invoke, *callee);

  for (unsigned resultIndex :
       getIndexArrayAttr(calleeOp, OwnershipContractAttrs::kOwnedResults)) {
    if (resultIndex < invoke->getNumResults()) {
      mlir::Value result = invoke->getResult(resultIndex);
      if (value_type::pointerLike(result.getType()) &&
          !borrow::immortal(result))
        if (mlir::failed(addTrackedToken(state, result, +1, invoke, aliases)))
          return mlir::failure();
    }
  }

  if (!invokeReturnsOwnedReference(invoke))
    return mlir::success();
  for (mlir::Value result : invoke->getResults())
    if (value_type::pointerLike(result.getType()) && !borrow::immortal(result))
      if (mlir::failed(addTrackedToken(state, result, +1, invoke, aliases)))
        return mlir::failure();
  return mlir::success();
}

static mlir::LogicalResult
applyOperationOwnership(mlir::Operation &op, LLVMOwnershipState &state,
                        mlir::Block &entry, const AliasAnalysis &aliases) {
  if (op.hasAttr(OwnershipContractAttrs::kOwnedLocalObject)) {
    for (mlir::Value result : op.getResults())
      if (mlir::failed(addTrackedToken(state, result, +1, &op, aliases)))
        return mlir::failure();
  }

  if (auto call = mlir::dyn_cast<mlir::LLVM::CallOp>(&op))
    if (mlir::failed(applyCallOwnership(call, state, entry, aliases)))
      return mlir::failure();

  if (auto invoke = mlir::dyn_cast<mlir::LLVM::InvokeOp>(&op))
    if (mlir::failed(
            applyInvokeOperandOwnership(invoke, state, entry, aliases)))
      return mlir::failure();

  if (auto store = mlir::dyn_cast<mlir::LLVM::StoreOp>(&op))
    if (mlir::failed(applyStoreOwnership(store, state, entry, aliases)))
      return mlir::failure();

  if (auto store = mlir::dyn_cast<mlir::memref::StoreOp>(&op))
    if (mlir::failed(applyMemRefStoreOwnership(store, state, aliases)))
      return mlir::failure();

  if (auto load = mlir::dyn_cast<mlir::LLVM::LoadOp>(&op))
    if (mlir::failed(applyLLVMLoadOwnership(load, state, aliases)))
      return mlir::failure();

  if (mlir::failed(applyAsyncPayloadLoadOwnership(op, state, aliases)))
    return mlir::failure();

  if (mlir::isa<mlir::func::ReturnOp, mlir::LLVM::ReturnOp,
                mlir::async::ReturnOp>(&op)) {
    for (auto [index, operand] : llvm::enumerate(op.getOperands()))
      if (mlir::failed(consumeReturnOperand(state, operand, &op,
                                            static_cast<unsigned>(index), entry,
                                            aliases)))
        return mlir::failure();
  }

  return mlir::success();
}

namespace operation {

bool hasOwnershipEffect(mlir::Operation *op) {
  return mlir::isa<mlir::LLVM::CallOp, mlir::LLVM::InvokeOp,
                   mlir::LLVM::StoreOp, mlir::memref::StoreOp,
                   mlir::LLVM::LoadOp, mlir::async::AwaitOp,
                   mlir::async::RuntimeLoadOp, mlir::func::ReturnOp,
                   mlir::LLVM::ReturnOp, mlir::async::ReturnOp>(op);
}

} // namespace operation

namespace nested_effect {

mlir::LogicalResult verifyAbsent(mlir::Operation &op) {
  if (op.getNumRegions() == 0)
    return mlir::success();

  mlir::Operation *bad = nullptr;
  for (mlir::Region &region : op.getRegions()) {
    region.walk([&](mlir::Operation *nested) {
      if (bad || nested == &op)
        return;
      if (operation::hasOwnershipEffect(nested))
        bad = nested;
    });
  }
  if (!bad)
    return mlir::success();
  return bad->emitOpError("nested LLVM ownership effect is not represented in "
                          "the flat CFG ownership proof");
}

} // namespace nested_effect

namespace cfedge {

mlir::Value condition(mlir::Operation *terminator) {
  if (auto branch = mlir::dyn_cast<mlir::cf::CondBranchOp>(terminator))
    return branch.getCondition();
  if (auto branch = mlir::dyn_cast<mlir::LLVM::CondBrOp>(terminator))
    return branch.getCondition();
  return {};
}

} // namespace cfedge

namespace frame_transfer {

void collectRoots(mlir::Block *block, const AliasAnalysis &aliases,
                  llvm::SmallPtrSetImpl<mlir::Block *> &visited,
                  llvm::SmallVectorImpl<mlir::Value> &roots) {
  if (!block || !visited.insert(block).second)
    return;

  for (mlir::Operation &op : *block) {
    auto call = mlir::dyn_cast<mlir::LLVM::CallOp>(op);
    if (!call || !call->hasAttr(OwnershipContractAttrs::kFrameTransfer))
      continue;
    auto callee = call.getCallee();
    if (!callee || *callee != RuntimeSymbols::kDecRef ||
        call.getNumOperands() < 1)
      continue;
    mlir::Value value = call.getOperand(0);
    if (!aliases.isCarrier(value))
      continue;
    mlir::Value root = aliases.getRoot(value);
    if (!llvm::is_contained(roots, root))
      roots.push_back(root);
  }

  mlir::Operation *terminator = block->getTerminator();
  if (!terminator)
    return;
  for (mlir::Block *successor : terminator->getSuccessors())
    collectRoots(successor, aliases, visited, roots);
}

void dropOnSuspendDefault(mlir::Operation *terminator, mlir::Block *successor,
                          LLVMOwnershipState &state,
                          const AliasAnalysis &aliases) {
  auto switchOp = mlir::dyn_cast<mlir::LLVM::SwitchOp>(terminator);
  if (!switchOp || !isCoroSuspendStatus(switchOp.getValue()))
    return;
  if (successor != switchOp.getDefaultDestination())
    return;

  auto cleanupIndex = findCoroSuspendCleanupSuccessorIndex(switchOp);
  if (!cleanupIndex)
    return;
  mlir::Block *cleanupSuccessor = switchOp->getSuccessor(*cleanupIndex);
  llvm::SmallVector<mlir::Value, 4> transferredRoots;
  llvm::SmallPtrSet<mlir::Block *, 8> visited;
  collectRoots(cleanupSuccessor, aliases, visited, transferredRoots);
  for (mlir::Value root : transferredRoots)
    state.balance.erase(root);
}

} // namespace frame_transfer

static mlir::LLVM::AtomicCmpXchgOp
getAsyncExceptionStoreCmpXchg(mlir::Value condition) {
  if (!condition)
    return {};
  auto extract = condition.getDefiningOp<mlir::LLVM::ExtractValueOp>();
  if (!extract || extract.getPosition() != llvm::ArrayRef<int64_t>{1})
    return {};
  auto cmpxchg =
      extract.getContainer().getDefiningOp<mlir::LLVM::AtomicCmpXchgOp>();
  if (!cmpxchg)
    return {};
  auto role =
      cmpxchg->getAttrOfType<mlir::StringAttr>(ThreadSafetyAttrs::kAtomicRole);
  if (!role || role.getValue() != ThreadSafetyAttrs::kRoleAsyncExceptionStore)
    return {};
  return cmpxchg;
}

namespace successor_transfer {

void moveRoot(LLVMOwnershipState &state, mlir::Value from, mlir::Value to,
              const AliasAnalysis &aliases) {
  if (!aliases.isCarrier(from) || !aliases.isCarrier(to))
    return;
  mlir::Value fromRoot = aliases.getRoot(from);
  mlir::Value toRoot = aliases.getRoot(to);
  if (!fromRoot || !toRoot || fromRoot == toRoot)
    return;
  auto it = state.balance.find(fromRoot);
  if (it == state.balance.end())
    return;
  state.balance[toRoot] += it->second;
  state.balance.erase(it);
  if (state.balance.lookup(toRoot) == 0)
    state.balance.erase(toRoot);
}

void remapBlockArguments(
    mlir::Operation *terminator,
    llvm::SmallVectorImpl<LLVMOwnershipState> &successorStates,
    const AliasAnalysis &aliases) {
  auto branch = mlir::dyn_cast<mlir::BranchOpInterface>(terminator);
  if (!branch)
    return;

  for (auto [successorIndex, successorState] :
       llvm::enumerate(successorStates)) {
    mlir::Block *successor = branch->getSuccessor(successorIndex);
    mlir::SuccessorOperands operands =
        branch.getSuccessorOperands(successorIndex);
    unsigned count =
        std::min<unsigned>(operands.size(), successor->getNumArguments());
    for (unsigned i = 0; i != count; ++i) {
      if (operands.isOperandProduced(i))
        continue;
      moveRoot(successorState, operands[i], successor->getArgument(i), aliases);
    }
  }
}

mlir::LogicalResult
apply(mlir::Operation *terminator,
      llvm::SmallVectorImpl<LLVMOwnershipState> &successorStates,
      const AliasAnalysis &aliases) {
  for (auto it : llvm::enumerate(successorStates)) {
    auto invoke = mlir::dyn_cast<mlir::LLVM::InvokeOp>(terminator);
    if (!invoke ||
        terminator->getSuccessor(it.index()) != invoke.getNormalDest())
      continue;
    LLVMOwnershipState &successorState = it.value();
    if (mlir::failed(
            addInvokeNormalResultOwnership(invoke, successorState, aliases)))
      return mlir::failure();
  }

  mlir::Value condition = cfedge::condition(terminator);
  auto cmpxchg = getAsyncExceptionStoreCmpXchg(condition);
  if (!cmpxchg || successorStates.empty()) {
    remapBlockArguments(terminator, successorStates, aliases);
    return mlir::success();
  }

  // Success successor is index 0 for both cf.cond_br and llvm.cond_br. A
  // successful exception-cell cmpxchg publishes the retained exception token to
  // the cell; the failure successor keeps the token so the explicit rollback
  // DecRef can consume it.
  if (mlir::failed(releaseIfTracked(successorStates.front(), cmpxchg.getVal(),
                                    terminator, aliases,
                                    /*requireTracked=*/true)))
    return mlir::failure();
  remapBlockArguments(terminator, successorStates, aliases);
  return mlir::success();
}

} // namespace successor_transfer

static bool statesEqual(const LLVMOwnershipState &lhs,
                        const LLVMOwnershipState &rhs) {
  if (lhs.balance.size() != rhs.balance.size())
    return false;
  for (auto [root, count] : lhs.balance) {
    auto it = rhs.balance.find(root);
    if (it == rhs.balance.end() || it->second != count)
      return false;
  }
  return true;
}

namespace open_roots {

void collect(mlir::Operation *func, mlir::Region &region,
             const AliasAnalysis &aliases,
             llvm::DenseMap<mlir::Value, int64_t> &roots) {
  if (region.empty())
    return;
  if (class_helper::hasKind(func, ClassSafetyAttrs::kKindIncref))
    return;
  mlir::Block &entry = region.front();
  for (unsigned argIndex :
       getIndexArrayAttr(func, OwnershipContractAttrs::kRetainArgs)) {
    if (argIndex < entry.getNumArguments())
      ++roots[aliases.getRoot(entry.getArgument(argIndex))];
  }
}

bool match(const LLVMOwnershipState &state,
           const llvm::DenseMap<mlir::Value, int64_t> &roots) {
  for (auto [root, count] : state.balance) {
    auto it = roots.find(root);
    int64_t expected = it == roots.end() ? 0 : it->second;
    if (count != expected)
      return false;
  }
  for (auto [root, expected] : roots) {
    auto it = state.balance.find(root);
    int64_t count = it == state.balance.end() ? 0 : it->second;
    if (count != expected)
      return false;
  }
  return true;
}

} // namespace open_roots

namespace entry_state {

mlir::LogicalResult seed(mlir::Operation *func, mlir::Block &entry,
                         const AliasAnalysis &aliases,
                         LLVMOwnershipState &state) {
  auto setFieldValueArg =
      getIndexAttr(func, OwnershipContractAttrs::kSetFieldValueArg);
  if (!setFieldValueArg || *setFieldValueArg >= entry.getNumArguments())
    return mlir::success();
  return addTrackedToken(state, entry.getArgument(*setFieldValueArg), +1, func,
                         aliases);
}

} // namespace entry_state

namespace postcondition {

bool mayRetainArgs(mlir::Operation *func) {
  auto symbol = mlir::dyn_cast<mlir::SymbolOpInterface>(func);
  if (!symbol)
    return false;
  llvm::StringRef name = symbol.getName();
  return name == RuntimeSymbols::kIncRef ||
         class_helper::hasKind(func, ClassSafetyAttrs::kKindIncref);
}

mlir::LogicalResult verifyRetainArgsScope(mlir::Operation *func) {
  if (getIndexArrayAttr(func, OwnershipContractAttrs::kRetainArgs).empty())
    return mlir::success();
  if (mayRetainArgs(func))
    return mlir::success();
  return func->emitOpError("retain_args leaves ownership open but function is "
                           "not a known retain helper");
}

} // namespace postcondition

namespace state_report {

mlir::LogicalResult mismatch(mlir::Operation *func, mlir::Block *block,
                             const LLVMOwnershipState &existing,
                             const LLVMOwnershipState &incoming) {
  mlir::InFlightDiagnostic diag =
      func->emitOpError("LLVM ownership balance differs across predecessors");
  if (block) {
    std::string blockName;
    llvm::raw_string_ostream os(blockName);
    block->printAsOperand(os, /*printType=*/false);
    diag << " for block " << os.str();
  }
  diag << "; existing tokens=" << existing.balance.size()
       << ", incoming tokens=" << incoming.balance.size();
  for (auto [root, count] : existing.balance)
    diag << " [existing " << root << ":" << count << "]";
  for (auto [root, count] : incoming.balance)
    diag << " [incoming " << root << ":" << count << "]";
  return mlir::failure();
}

} // namespace state_report

namespace terminator {

bool abortLikeCallee(llvm::StringRef callee) {
  return callee == "abort" || callee == "llvm.trap";
}

bool noReturnUnreachable(mlir::Operation *terminatorOp) {
  if (!mlir::isa<mlir::LLVM::UnreachableOp>(terminatorOp))
    return false;
  mlir::Operation *previous = terminatorOp->getPrevNode();
  auto call = mlir::dyn_cast_or_null<mlir::LLVM::CallOp>(previous);
  if (!call)
    return false;
  auto callee = call.getCallee();
  return callee && abortLikeCallee(*callee);
}

mlir::LogicalResult
verifyExit(mlir::Operation *terminator, const LLVMOwnershipState &state,
           const llvm::DenseMap<mlir::Value, int64_t> &allowedOpenRoots) {
  if (noReturnUnreachable(terminator))
    return mlir::success();
  if (open_roots::match(state, allowedOpenRoots))
    return mlir::success();
  if (allowedOpenRoots.empty() && state.empty())
    return mlir::success();
  mlir::InFlightDiagnostic diag = terminator->emitOpError(
      "LLVM ownership balance is not closed on exit; remaining tokens=");
  diag << state.balance.size();
  for (auto [root, count] : state.balance)
    diag << " [" << root << ":" << count << "]";
  return mlir::failure();
}

} // namespace terminator

namespace function_like {

mlir::LogicalResult verify(mlir::Operation *func, mlir::Region &region) {
  if (region.empty())
    return mlir::success();
  if (mlir::failed(postcondition::verifyRetainArgsScope(func)))
    return mlir::failure();

  AliasAnalysis aliases(region, carrier::ownership,
                        lowered_identity::transform);
  llvm::DenseMap<mlir::Value, int64_t> allowedOpenRoots;
  open_roots::collect(func, region, aliases, allowedOpenRoots);
  llvm::DenseMap<mlir::Block *, LLVMOwnershipState> entryStates;
  llvm::SmallVector<mlir::Block *, 16> worklist;
  llvm::SmallPtrSet<mlir::Block *, 16> queued;

  mlir::Block *entry = &region.front();
  LLVMOwnershipState initialState;
  if (mlir::failed(entry_state::seed(func, *entry, aliases, initialState)))
    return mlir::failure();
  entryStates.try_emplace(entry, initialState);
  worklist.push_back(entry);
  queued.insert(entry);

  while (!worklist.empty()) {
    mlir::Block *block = worklist.pop_back_val();
    queued.erase(block);

    auto entryIt = entryStates.find(block);
    if (entryIt == entryStates.end())
      continue;
    LLVMOwnershipState state = entryIt->second;

    for (mlir::Operation &op : *block) {
      if (mlir::failed(nested_effect::verifyAbsent(op)))
        return mlir::failure();
      if (mlir::failed(applyOperationOwnership(op, state, *entry, aliases)))
        return mlir::failure();
    }

    mlir::Operation *terminator = block->getTerminator();
    if (!terminator)
      return func->emitOpError("contains a block without terminator");

    if (terminator->getNumSuccessors() == 0) {
      if (mlir::failed(
              terminator::verifyExit(terminator, state, allowedOpenRoots)))
        return mlir::failure();
      continue;
    }

    llvm::SmallVector<LLVMOwnershipState, 2> successorStates(
        terminator->getNumSuccessors(), state);
    if (mlir::failed(
            successor_transfer::apply(terminator, successorStates, aliases)))
      return mlir::failure();

    for (auto [index, successor] : llvm::enumerate(block->getSuccessors())) {
      LLVMOwnershipState &successorState = successorStates[index];
      frame_transfer::dropOnSuspendDefault(terminator, successor,
                                           successorState, aliases);
      auto [it, inserted] = entryStates.try_emplace(successor, successorState);
      if (!inserted) {
        if (!statesEqual(it->second, successorState))
          return state_report::mismatch(func, successor, it->second,
                                        successorState);
        continue;
      }
      if (queued.insert(successor).second)
        worklist.push_back(successor);
    }
  }

  return mlir::success();
}

} // namespace function_like

struct LLVMCallOwnershipVerifierPass
    : public mlir::PassWrapper<LLVMCallOwnershipVerifierPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LLVMCallOwnershipVerifierPass)

  llvm::StringRef getArgument() const override {
    return "py-llvm-call-ownership-verify";
  }

  llvm::StringRef getDescription() const override {
    return "Verify conservative ownership balance for lowered LLVM calls";
  }

  void runOnOperation() override {
    if (mlir::failed(verifyLLVMCallOwnership(getOperation())))
      signalPassFailure();
  }
};

} // namespace

static mlir::LogicalResult
verifyNoUnrealizedCastsForOwnership(mlir::ModuleOp module) {
  mlir::UnrealizedConversionCastOp offender = nullptr;
  module.walk([&](mlir::UnrealizedConversionCastOp cast) {
    offender = cast;
    return mlir::WalkResult::interrupt();
  });
  if (!offender)
    return mlir::success();
  return offender.emitError(
      "unrealized conversion cast reached LLVM ownership verifier");
}

mlir::LogicalResult verifyLLVMCallOwnership(mlir::ModuleOp module) {
  if (mlir::failed(verifyNoUnrealizedCastsForOwnership(module)))
    return mlir::failure();

  bool failedAny = false;

  module.walk([&](mlir::func::FuncOp func) {
    if (func.isExternal())
      return;
    if (mlir::failed(
            function_like::verify(func.getOperation(), func.getBody())))
      failedAny = true;
  });

  module.walk([&](mlir::async::FuncOp func) {
    if (func.getBody().empty())
      return;
    if (mlir::failed(
            function_like::verify(func.getOperation(), func.getBody())))
      failedAny = true;
  });

  module.walk([&](mlir::LLVM::LLVMFuncOp func) {
    if (func.isDeclaration())
      return;
    if (mlir::failed(
            function_like::verify(func.getOperation(), func.getBody())))
      failedAny = true;
  });

  return mlir::failure(failedAny);
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createLLVMCallOwnershipVerifierPass() {
  return std::make_unique<LLVMCallOwnershipVerifierPass>();
}

} // namespace py
