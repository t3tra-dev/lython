#include "Common/AsyncSafetyKernel.h"
#include "Common/Object.h"
#include "Common/RuntimeSupport.h"

#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
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

std::optional<llvm::StringRef> directCallee(mlir::Operation *op) {
  if (auto call = mlir::dyn_cast<mlir::LLVM::CallOp>(op))
    return call.getCallee();
  if (auto call = mlir::dyn_cast<mlir::func::CallOp>(op))
    return call.getCallee();
  return std::nullopt;
}

namespace async_marker {

enum class Kind { ExceptionEdge };

struct Spec {
  Kind kind;
  llvm::StringLiteral symbol;
  bool allowSuffix;
  bool eraseFuncDeclarationOnly;
};

static constexpr Spec kSpecs[] = {
    {Kind::ExceptionEdge, kAsyncExceptionEdgeMarker, /*allowSuffix=*/true,
     /*eraseFuncDeclarationOnly=*/true},
};

static const Spec *lookup(Kind kind) {
  for (const Spec &spec : kSpecs)
    if (spec.kind == kind)
      return &spec;
  return nullptr;
}

static std::optional<Kind> classify(llvm::StringRef callee) {
  for (const Spec &spec : kSpecs) {
    if (callee == spec.symbol)
      return spec.kind;
    if (!spec.allowSuffix)
      continue;
    llvm::StringRef suffix = callee;
    if (suffix.consume_front(spec.symbol) && suffix.starts_with("_"))
      return spec.kind;
  }
  return std::nullopt;
}

static bool is(llvm::StringRef callee, Kind kind) {
  std::optional<Kind> actual = classify(callee);
  return actual && *actual == kind;
}

static void eraseDeclarations(mlir::ModuleOp module, Kind kind) {
  const Spec *spec = lookup(kind);
  if (!spec)
    return;

  llvm::SmallVector<mlir::Operation *> markerFuncs;
  module.walk([&](mlir::func::FuncOp markerFunc) {
    if (!is(markerFunc.getName(), kind))
      return;
    if (!spec->eraseFuncDeclarationOnly ||
        (markerFunc.isPrivate() && markerFunc.isDeclaration()))
      markerFuncs.push_back(markerFunc.getOperation());
  });
  module.walk([&](mlir::LLVM::LLVMFuncOp markerFunc) {
    if (is(markerFunc.getName(), kind))
      markerFuncs.push_back(markerFunc.getOperation());
  });
  for (mlir::Operation *markerFunc : markerFuncs)
    markerFunc->erase();
}

} // namespace async_marker

mlir::Operation *lookupCallable(mlir::Operation *op, llvm::StringRef name) {
  if (!op)
    return nullptr;
  auto symbol = mlir::StringAttr::get(op->getContext(), name);
  if (auto func =
          mlir::SymbolTable::lookupNearestSymbolFrom<mlir::func::FuncOp>(
              op, symbol))
    return func.getOperation();
  if (auto func =
          mlir::SymbolTable::lookupNearestSymbolFrom<mlir::LLVM::LLVMFuncOp>(
              op, symbol))
    return func.getOperation();
  return nullptr;
}

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

namespace async_contract {

static std::optional<int64_t> i64(mlir::Operation *op,
                                  llvm::StringRef attrName) {
  if (!op)
    return std::nullopt;
  auto attr = op->getAttrOfType<mlir::IntegerAttr>(attrName);
  if (!attr)
    return std::nullopt;
  return attr.getInt();
}

static void setIndexArray(mlir::Operation *op, llvm::StringRef attrName,
                          llvm::ArrayRef<int64_t> values) {
  if (!op)
    return;
  mlir::Builder builder(op->getContext());
  llvm::SmallVector<mlir::Attribute, 2> attrs;
  attrs.reserve(values.size());
  for (int64_t value : values)
    attrs.push_back(builder.getI64IntegerAttr(value));
  op->setAttr(attrName, builder.getArrayAttr(attrs));
}

static bool has(mlir::Operation *op, llvm::StringRef attrName) {
  return async_runtime::contract::has(op, attrName);
}

} // namespace async_contract

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

bool ownedPayload(mlir::Type type) {
  if (pointerLike(type) || object_abi::Type::isLoweredStorage(type))
    return true;
  auto aggregate = mlir::dyn_cast<mlir::LLVM::LLVMStructType>(type);
  return aggregate && !aggregate.isOpaque() &&
         aggregate.getBody().size() >= 2 &&
         llvm::all_of(aggregate.getBody(), object_abi::Type::isLoweredStorage);
}

} // namespace value_type

namespace runtime_call {

static void appendMemRefDescriptor(mlir::Location loc, mlir::Value descriptor,
                                   llvm::SmallVectorImpl<mlir::Value> &operands,
                                   mlir::IRRewriter &rewriter) {
  auto descriptorType =
      mlir::cast<mlir::LLVM::LLVMStructType>(descriptor.getType());
  llvm::ArrayRef<mlir::Type> body = descriptorType.getBody();
  mlir::Type i64 = rewriter.getI64Type();
  operands.push_back(rewriter.create<mlir::LLVM::ExtractValueOp>(
      loc, body[0], descriptor, rewriter.getDenseI64ArrayAttr({0})));
  operands.push_back(rewriter.create<mlir::LLVM::ExtractValueOp>(
      loc, body[1], descriptor, rewriter.getDenseI64ArrayAttr({1})));
  operands.push_back(rewriter.create<mlir::LLVM::ExtractValueOp>(
      loc, body[2], descriptor, rewriter.getDenseI64ArrayAttr({2})));
  operands.push_back(rewriter.create<mlir::LLVM::ExtractValueOp>(
      loc, i64, descriptor, rewriter.getDenseI64ArrayAttr({3, 0})));
  operands.push_back(rewriter.create<mlir::LLVM::ExtractValueOp>(
      loc, i64, descriptor, rewriter.getDenseI64ArrayAttr({4, 0})));
}

mlir::LLVM::CallOp emitVoid(mlir::ModuleOp module, mlir::Location loc,
                            mlir::IRRewriter &rewriter, llvm::StringRef name,
                            mlir::ValueRange operands) {
  llvm::SmallVector<mlir::Value> callOperands;
  callOperands.reserve(operands.size() * 5);
  for (mlir::Value operand : operands) {
    if (object_abi::Type::isLoweredStorage(operand.getType())) {
      appendMemRefDescriptor(loc, operand, callOperands, rewriter);
      continue;
    }
    callOperands.push_back(operand);
  }

  llvm::SmallVector<mlir::Type> argTypes;
  argTypes.reserve(callOperands.size());
  for (mlir::Value operand : callOperands)
    argTypes.push_back(operand.getType());
  auto callee = runtime_func::getOrInsert(
      module, loc, rewriter, name,
      mlir::LLVM::LLVMVoidType::get(module.getContext()), argTypes);
  return rewriter.create<mlir::LLVM::CallOp>(
      loc, mlir::TypeRange{}, mlir::SymbolRefAttr::get(callee), callOperands);
}

} // namespace runtime_call

namespace async_handle_ref {

constexpr llvm::StringLiteral kDropRef = "mlirAsyncRuntimeDropRef";

static mlir::Operation *drop(mlir::ModuleOp module, mlir::Location loc,
                             mlir::IRRewriter &rewriter, mlir::Value handle,
                             bool allowAsyncOp, bool frameTransfer) {
  mlir::Operation *op = nullptr;
  auto one = rewriter.getI64IntegerAttr(1);
  if (allowAsyncOp && mlir::isa<mlir::async::ValueType>(handle.getType())) {
    op = rewriter.create<mlir::async::RuntimeDropRefOp>(loc, handle, one)
             .getOperation();
  } else {
    mlir::Value count = rewriter.create<mlir::LLVM::ConstantOp>(
        loc, rewriter.getI64Type(), one);
    if (module.lookupSymbol<mlir::func::FuncOp>(kDropRef)) {
      op = rewriter
               .create<mlir::func::CallOp>(loc, kDropRef, mlir::TypeRange{},
                                           mlir::ValueRange{handle, count})
               .getOperation();
    } else {
      op = runtime_call::emitVoid(module, loc, rewriter, kDropRef,
                                  mlir::ValueRange{handle, count});
    }
    op->setAttr(AsyncSafetyAttrs::kRuntimeRefcountDelta,
                rewriter.getI64IntegerAttr(-1));
  }

  if (frameTransfer)
    op->setAttr(OwnershipContractAttrs::kFrameTransfer,
                mlir::UnitAttr::get(module.getContext()));
  return op;
}

} // namespace async_handle_ref

namespace lowered_descriptor {

mlir::Value build(mlir::Location loc, mlir::IRRewriter &rewriter,
                  llvm::ArrayRef<mlir::Value> operands) {
  if (operands.size() != 5)
    return {};

  auto ptrType = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
  if (operands[0].getType() != ptrType || operands[1].getType() != ptrType ||
      !operands[2].getType().isInteger(64) ||
      !operands[3].getType().isInteger(64) ||
      !operands[4].getType().isInteger(64))
    return {};

  auto descriptorType = object_abi::Type::loweredStorage(rewriter.getContext());
  mlir::Value result =
      rewriter.create<mlir::LLVM::UndefOp>(loc, descriptorType);
  result = rewriter.create<mlir::LLVM::InsertValueOp>(
      loc, descriptorType, result, operands[0],
      rewriter.getDenseI64ArrayAttr({0}));
  result = rewriter.create<mlir::LLVM::InsertValueOp>(
      loc, descriptorType, result, operands[1],
      rewriter.getDenseI64ArrayAttr({1}));
  result = rewriter.create<mlir::LLVM::InsertValueOp>(
      loc, descriptorType, result, operands[2],
      rewriter.getDenseI64ArrayAttr({2}));
  result = rewriter.create<mlir::LLVM::InsertValueOp>(
      loc, descriptorType, result, operands[3],
      rewriter.getDenseI64ArrayAttr({3, 0}));
  result = rewriter.create<mlir::LLVM::InsertValueOp>(
      loc, descriptorType, result, operands[4],
      rewriter.getDenseI64ArrayAttr({4, 0}));
  return result;
}

} // namespace lowered_descriptor

namespace suspend_switch {

mlir::Value awaitedHandle(mlir::LLVM::SwitchOp switchOp) {
  for (mlir::Operation *cursor = switchOp->getPrevNode(); cursor;
       cursor = cursor->getPrevNode()) {
    if (auto call = mlir::dyn_cast<mlir::LLVM::CallOp>(cursor)) {
      if (async_contract::has(call.getOperation(),
                              AsyncSafetyAttrs::kRuntimeAwaitExecute) &&
          call.getNumOperands() > 0)
        return call.getOperand(0);
    }
    if (auto call = mlir::dyn_cast<mlir::func::CallOp>(cursor)) {
      if (async_contract::has(call.getOperation(),
                              AsyncSafetyAttrs::kRuntimeAwaitExecute) &&
          call.getNumOperands() > 0)
        return call.getOperand(0);
    }
  }
  return {};
}

} // namespace suspend_switch

namespace exception_cell {
mlir::LogicalResult destroyAt(mlir::ModuleOp module, mlir::Location loc,
                              mlir::IRRewriter &rewriter, mlir::Value cell);
} // namespace exception_cell

namespace exception_cell_operand {

bool arg(mlir::Operation *calleeOp, unsigned argIndex) {
  auto function = mlir::dyn_cast_or_null<mlir::FunctionOpInterface>(calleeOp);
  return function && argIndex < function.getNumArguments() &&
         function.getArgAttr(argIndex, AsyncSafetyAttrs::kExceptionCell);
}

bool marked(mlir::Value value) {
  if (!value)
    return false;
  if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    auto function = arg.getOwner()->getParentOp()
                        ? mlir::dyn_cast<mlir::FunctionOpInterface>(
                              arg.getOwner()->getParentOp())
                        : mlir::FunctionOpInterface();
    return function && arg.getArgNumber() < function.getNumArguments() &&
           function.getArgAttr(arg.getArgNumber(),
                               AsyncSafetyAttrs::kExceptionCell);
  }
  mlir::Operation *def = value.getDefiningOp();
  return def && def->hasAttr(AsyncSafetyAttrs::kExceptionCell);
}

mlir::Value call(mlir::ModuleOp module, mlir::Operation *op,
                 mlir::Value awaitedHandle) {
  if (!op || op->getNumResults() != 1 || op->getResult(0) != awaitedHandle ||
      op->getNumOperands() == 0)
    return {};
  auto callee = directCallee(op);
  if (!callee)
    return {};
  mlir::Operation *calleeOp = lookupCallable(op, *callee);
  unsigned last = op->getNumOperands() - 1;
  mlir::Value cell = op->getOperand(last);
  if (!arg(calleeOp, last) && !marked(cell))
    return {};
  return cell;
}

mlir::Value forHandle(mlir::ModuleOp module, mlir::Operation *anchor,
                      mlir::Value handle) {
  if (!handle)
    return {};
  if (mlir::Operation *def = handle.getDefiningOp())
    if (mlir::Value cell = call(module, def, handle))
      return cell;
  for (mlir::Operation *cursor = anchor ? anchor->getPrevNode() : nullptr;
       cursor; cursor = cursor->getPrevNode()) {
    if (mlir::Value cell = call(module, cursor, handle))
      return cell;
  }
  return {};
}

} // namespace exception_cell_operand

namespace async_ref {

bool dropFor(mlir::Operation &op, mlir::Value handle) {
  if (auto call = mlir::dyn_cast<mlir::LLVM::CallOp>(op)) {
    auto delta = async_contract::i64(call.getOperation(),
                                     AsyncSafetyAttrs::kRuntimeRefcountDelta);
    return delta && *delta < 0 && call.getNumOperands() >= 1 &&
           call.getOperand(0) == handle;
  }
  if (auto call = mlir::dyn_cast<mlir::func::CallOp>(op)) {
    auto delta = async_contract::i64(call.getOperation(),
                                     AsyncSafetyAttrs::kRuntimeRefcountDelta);
    return delta && *delta < 0 && call.getNumOperands() >= 1 &&
           call.getOperand(0) == handle;
  }
  return false;
}

} // namespace async_ref

namespace py_ref {

static bool releasesPayload(mlir::LLVM::CallOp call) {
  return call->hasAttr(OwnershipContractAttrs::kReleaseArgs) ||
         call->hasAttr(OwnershipContractAttrs::kAggregateRelease);
}

static bool derivedFrom(mlir::Value value, mlir::Value root) {
  if (!value || !root)
    return false;
  if (value == root)
    return true;
  if (auto extract = value.getDefiningOp<mlir::LLVM::ExtractValueOp>())
    return derivedFrom(extract.getContainer(), root);
  return false;
}

bool decFor(mlir::Operation &op, mlir::Value payload) {
  if (auto call = mlir::dyn_cast<mlir::LLVM::CallOp>(op)) {
    if (!releasesPayload(call))
      return false;
    if (call.getNumOperands() < 1)
      return false;
    if (call.getOperand(0) == payload)
      return true;
    return llvm::any_of(call.getOperands(), [&](mlir::Value operand) {
      return derivedFrom(operand, payload);
    });
  }
  return false;
}

} // namespace py_ref

namespace async_payload_ref {

static bool partsAggregate(mlir::Value payload) {
  if (!payload)
    return false;
  auto aggregate =
      mlir::dyn_cast<mlir::LLVM::LLVMStructType>(payload.getType());
  return aggregate && !aggregate.isOpaque() &&
         aggregate.getBody().size() >= 2 &&
         llvm::all_of(aggregate.getBody(), object_abi::Type::isLoweredStorage);
}

static llvm::SmallVector<mlir::Value, 4>
parts(mlir::Location loc, mlir::IRRewriter &rewriter, mlir::Value payload) {
  llvm::SmallVector<mlir::Value, 4> result;
  auto aggregate =
      mlir::dyn_cast<mlir::LLVM::LLVMStructType>(payload.getType());
  if (!aggregate || aggregate.isOpaque() || aggregate.getBody().size() < 2)
    return result;
  for (int64_t index = 0;
       index < static_cast<int64_t>(aggregate.getBody().size()); ++index)
    result.push_back(rewriter.create<mlir::LLVM::ExtractValueOp>(
        loc, aggregate.getBody()[index], payload,
        rewriter.getDenseI64ArrayAttr({index})));
  return result;
}

static mlir::Value layoutId(mlir::Location loc, mlir::IRRewriter &rewriter,
                            mlir::Value headerDescriptor) {
  auto descriptorType =
      mlir::dyn_cast<mlir::LLVM::LLVMStructType>(headerDescriptor.getType());
  if (!descriptorType || descriptorType.isOpaque() ||
      descriptorType.getBody().size() != 5)
    return {};
  auto ptrType = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
  mlir::Value data = rewriter.create<mlir::LLVM::ExtractValueOp>(
      loc, ptrType, headerDescriptor, rewriter.getDenseI64ArrayAttr({1}));
  mlir::Value slot = rewriter.create<mlir::LLVM::ConstantOp>(
      loc, rewriter.getI64Type(),
      rewriter.getI64IntegerAttr(object_abi::kLayoutIdSlot));
  mlir::Value address = rewriter.create<mlir::LLVM::GEPOp>(
      loc, ptrType, rewriter.getI64Type(), data,
      llvm::ArrayRef<mlir::LLVM::GEPArg>{slot});
  return rewriter.create<mlir::LLVM::LoadOp>(loc, rewriter.getI64Type(),
                                             address);
}

static mlir::Operation *emitReleaseCall(mlir::Location loc,
                                        mlir::ModuleOp module,
                                        mlir::IRRewriter &rewriter,
                                        llvm::StringRef symbol,
                                        mlir::ValueRange payloadParts) {
  PyLLVMTypeConverter typeConverter(module.getContext());
  RuntimeAPI runtime(module, rewriter, typeConverter);
  RuntimeAPI::Call call = runtime.call(loc, symbol, mlir::Type(), payloadParts);
  call->setAttr(OwnershipContractAttrs::kAggregateRelease,
                mlir::UnitAttr::get(module.getContext()));
  call->setAttr(OwnershipContractAttrs::kFrameTransfer,
                mlir::UnitAttr::get(module.getContext()));
  return call.getOperation();
}

static void emitAbort(mlir::ModuleOp module, mlir::Location loc,
                      mlir::IRRewriter &rewriter) {
  auto abortFn = runtime_func::getOrInsert(
      module, loc, rewriter, "abort",
      mlir::LLVM::LLVMVoidType::get(rewriter.getContext()), {});
  auto call = rewriter.create<mlir::LLVM::CallOp>(
      loc, mlir::TypeRange{},
      mlir::SymbolRefAttr::get(module.getContext(), abortFn.getName()),
      mlir::ValueRange{});
  call->setAttr(ControlFlowContractAttrs::kNoReturn,
                mlir::UnitAttr::get(module.getContext()));
  rewriter.create<mlir::LLVM::UnreachableOp>(loc);
}

static mlir::Operation *releasePartsAggregate(mlir::ModuleOp module,
                                              mlir::Location loc,
                                              mlir::IRRewriter &rewriter,
                                              mlir::Value payload) {
  llvm::SmallVector<mlir::Value, 4> payloadParts =
      parts(loc, rewriter, payload);
  if (payloadParts.size() != 2 && payloadParts.size() != 3)
    return nullptr;
  mlir::Value layout = layoutId(loc, rewriter, payloadParts.front());
  if (!layout)
    return nullptr;

  mlir::Block *dispatchBlock = rewriter.getInsertionBlock();
  mlir::Region *region = dispatchBlock->getParent();
  mlir::Block *afterBlock =
      rewriter.splitBlock(dispatchBlock, rewriter.getInsertionPoint());

  auto branchIfLayout = [&](mlir::Block *matchBlock, mlir::Block *nextBlock,
                            int64_t expected) {
    mlir::Value expectedValue = rewriter.create<mlir::LLVM::ConstantOp>(
        loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(expected));
    mlir::Value matched = rewriter.create<mlir::LLVM::ICmpOp>(
        loc, mlir::LLVM::ICmpPredicate::eq, layout, expectedValue);
    rewriter.create<mlir::cf::CondBranchOp>(loc, matched, matchBlock,
                                            nextBlock);
  };

  mlir::Block *longReleaseBlock =
      rewriter.createBlock(region, afterBlock->getIterator());
  mlir::Block *unicodeReleaseBlock =
      rewriter.createBlock(region, afterBlock->getIterator());
  mlir::Block *exceptionReleaseBlock =
      rewriter.createBlock(region, afterBlock->getIterator());
  mlir::Block *unicodeCheckBlock =
      rewriter.createBlock(region, afterBlock->getIterator());
  mlir::Block *exceptionCheckBlock =
      rewriter.createBlock(region, afterBlock->getIterator());
  mlir::Block *abortBlock =
      rewriter.createBlock(region, afterBlock->getIterator());

  rewriter.setInsertionPointToEnd(dispatchBlock);
  branchIfLayout(longReleaseBlock,
                 payloadParts.size() == 3 ? exceptionCheckBlock
                                          : unicodeCheckBlock,
                 object_abi::long_abi::kLayoutId);

  rewriter.setInsertionPointToStart(longReleaseBlock);
  mlir::Operation *release = nullptr;
  if (payloadParts.size() == 3) {
    release = emitReleaseCall(loc, module, rewriter,
                              RuntimeSymbols::kLongDecRef, payloadParts);
    rewriter.create<mlir::cf::BranchOp>(loc, afterBlock);
  } else {
    rewriter.create<mlir::cf::BranchOp>(loc, abortBlock);
  }

  rewriter.setInsertionPointToStart(unicodeCheckBlock);
  if (payloadParts.size() == 2)
    branchIfLayout(unicodeReleaseBlock, exceptionCheckBlock,
                   object_abi::str_abi::kLayoutId);
  else
    rewriter.create<mlir::cf::BranchOp>(loc, abortBlock);

  rewriter.setInsertionPointToStart(unicodeReleaseBlock);
  if (payloadParts.size() == 2)
    release = emitReleaseCall(loc, module, rewriter,
                              RuntimeSymbols::kUnicodeDecRef, payloadParts);
  rewriter.create<mlir::cf::BranchOp>(loc, afterBlock);

  rewriter.setInsertionPointToStart(exceptionCheckBlock);
  if (payloadParts.size() == 3)
    branchIfLayout(exceptionReleaseBlock, abortBlock,
                   object_abi::exception_abi::kLayoutId);
  else
    rewriter.create<mlir::cf::BranchOp>(loc, abortBlock);

  rewriter.setInsertionPointToStart(exceptionReleaseBlock);
  if (payloadParts.size() == 3)
    release = emitReleaseCall(loc, module, rewriter,
                              RuntimeSymbols::kExceptionDecRef, payloadParts);
  rewriter.create<mlir::cf::BranchOp>(loc, afterBlock);

  rewriter.setInsertionPointToStart(abortBlock);
  emitAbort(module, loc, rewriter);

  rewriter.setInsertionPointToStart(afterBlock);
  return release;
}

mlir::FailureOr<mlir::Operation *> release(mlir::ModuleOp module,
                                           mlir::Location loc,
                                           mlir::IRRewriter &rewriter,
                                           mlir::Value payload) {
  if (partsAggregate(payload))
    return releasePartsAggregate(module, loc, rewriter, payload);
  PyLLVMTypeConverter typeConverter(module.getContext());
  return async_runtime::ExceptionCell::releasePayload(loc, module, rewriter,
                                                      typeConverter, payload);
}

} // namespace async_payload_ref

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
  for (unsigned index : async_runtime::Handle::transferredOperands(&op))
    if (index < op.getNumOperands())
      value_set::erase(handles, op.getOperand(index));
  for (mlir::Value handle : llvm::make_early_inc_range(handles))
    if (async_ref::dropFor(op, handle))
      value_set::erase(handles, handle);
}

} // namespace async_child

namespace async_payload {

void transfer(mlir::Operation &op,
              llvm::SmallVectorImpl<mlir::Value> &payloads) {
  if (auto load = mlir::dyn_cast<mlir::async::RuntimeLoadOp>(op)) {
    mlir::Value result = load.getResult();
    if (value_type::ownedPayload(result.getType())) {
      value_set::add(payloads, result);
      return;
    }
  }
  if (auto load = mlir::dyn_cast<mlir::LLVM::LoadOp>(op)) {
    mlir::Value result = load.getResult();
    if (value_type::ownedPayload(result.getType()) &&
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
    llvm::SmallVector<mlir::Value> exceptionCells;
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
    llvm::SmallVector<mlir::Value> exceptionCells;
    for (mlir::Value handle : handles)
      value_set::add(exceptionCells,
                     exception_cell_operand::forHandle(
                         module, switchOp.getOperation(), handle));
    if (handles.empty() && payloads.empty() && exceptionCells.empty())
      return;
    cleanupDrops.push_back(CleanupDrops{switchOp, *cleanupIndex,
                                        std::move(handles), std::move(payloads),
                                        std::move(exceptionCells)});
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
    for (mlir::Value handle : drop.handles)
      async_handle_ref::drop(module, switchOp.getLoc(), rewriter, handle,
                             /*allowAsyncOp=*/false,
                             /*frameTransfer=*/true);
    for (mlir::Value payload : drop.payloads) {
      mlir::FailureOr<mlir::Operation *> release = async_payload_ref::release(
          module, switchOp.getLoc(), rewriter, payload);
      if (mlir::failed(release))
        return mlir::failure();
      if (*release)
        (*release)->setAttr(OwnershipContractAttrs::kFrameTransfer,
                            mlir::UnitAttr::get(module.getContext()));
    }
    for (mlir::Value cell : drop.exceptionCells)
      if (mlir::failed(exception_cell::destroyAt(module, switchOp.getLoc(),
                                                 rewriter, cell)))
        return mlir::failure();
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
  if (condition.getDefiningOp<mlir::async::RuntimeIsErrorOp>())
    return true;
  if (auto call = condition.getDefiningOp<mlir::func::CallOp>())
    return async_contract::has(call.getOperation(),
                               AsyncSafetyAttrs::kRuntimeErrorQuery);
  if (auto call = condition.getDefiningOp<mlir::LLVM::CallOp>())
    return async_contract::has(call.getOperation(),
                               AsyncSafetyAttrs::kRuntimeErrorQuery);
  return false;
}

mlir::Value handle(mlir::Value condition) {
  if (!condition)
    return {};
  if (auto isError = condition.getDefiningOp<mlir::async::RuntimeIsErrorOp>())
    return isError.getOperand();
  if (auto call = condition.getDefiningOp<mlir::func::CallOp>())
    if (async_contract::has(call.getOperation(),
                            AsyncSafetyAttrs::kRuntimeErrorQuery) &&
        call.getNumOperands() > 0)
      return call.getOperand(0);
  if (auto call = condition.getDefiningOp<mlir::LLVM::CallOp>()) {
    if (async_contract::has(call.getOperation(),
                            AsyncSafetyAttrs::kRuntimeErrorQuery) &&
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

mlir::Value remapToSuccessor(mlir::Operation *terminator,
                             unsigned successorIndex, mlir::Value value) {
  auto branch = mlir::dyn_cast_or_null<mlir::BranchOpInterface>(terminator);
  if (!branch || successorIndex >= terminator->getNumSuccessors())
    return value;
  mlir::Block *successor = terminator->getSuccessor(successorIndex);
  mlir::SuccessorOperands operands =
      branch.getSuccessorOperands(successorIndex);
  unsigned count =
      std::min<unsigned>(operands.size(), successor->getNumArguments());
  for (unsigned i = 0; i != count; ++i) {
    if (operands.isOperandProduced(i))
      continue;
    if (operands[i] == value)
      return successor->getArgument(i);
  }
  return value;
}

bool noReturn(mlir::Operation *terminator) {
  return mlir::isa_and_nonnull<mlir::LLVM::UnreachableOp>(terminator);
}

bool hasDecRefOnAllExits(mlir::Block *block, mlir::Value payload,
                         llvm::SmallPtrSetImpl<mlir::Block *> &seen) {
  if (!block)
    return false;
  for (mlir::Operation &op : *block)
    if (py_ref::decFor(op, payload))
      return true;
  mlir::Operation *terminator = block->getTerminator();
  if (!terminator)
    return false;
  if (noReturn(terminator))
    return true;
  if (terminator->getNumSuccessors() == 0)
    return false;
  if (!seen.insert(block).second)
    return true;
  for (unsigned i = 0, e = terminator->getNumSuccessors(); i != e; ++i) {
    mlir::Block *successor = terminator->getSuccessor(i);
    mlir::Value successorPayload = remapToSuccessor(terminator, i, payload);
    if (!hasDecRefOnAllExits(successor, successorPayload, seen))
      return false;
  }
  return true;
}

bool hasDecRef(mlir::Block *block, mlir::Value payload) {
  llvm::SmallPtrSet<mlir::Block *, 8> seen;
  return hasDecRefOnAllExits(block, payload, seen);
}

bool hasAsyncDropOnAllExits(mlir::Block *block, mlir::Value handle,
                            llvm::SmallPtrSetImpl<mlir::Block *> &seen) {
  if (!block || !handle)
    return false;
  for (mlir::Operation &op : *block)
    if (async_ref::dropFor(op, handle))
      return true;
  mlir::Operation *terminator = block->getTerminator();
  if (!terminator)
    return false;
  if (noReturn(terminator))
    return true;
  if (terminator->getNumSuccessors() == 0)
    return false;
  if (!seen.insert(block).second)
    return true;
  for (unsigned i = 0, e = terminator->getNumSuccessors(); i != e; ++i) {
    mlir::Block *successor = terminator->getSuccessor(i);
    mlir::Value successorHandle = remapToSuccessor(terminator, i, handle);
    if (!hasAsyncDropOnAllExits(successor, successorHandle, seen))
      return false;
  }
  return true;
}

bool hasAsyncDrop(mlir::Block *block, mlir::Value handle) {
  llvm::SmallPtrSet<mlir::Block *, 8> seen;
  return hasAsyncDropOnAllExits(block, handle, seen);
}

mlir::Value exceptionCellRoot(mlir::Value value) {
  while (value) {
    if (auto cast = value.getDefiningOp<mlir::memref::CastOp>()) {
      value = cast.getSource();
      continue;
    }
    if (auto bitcast = value.getDefiningOp<mlir::LLVM::BitcastOp>()) {
      value = bitcast.getArg();
      continue;
    }
    if (auto intToPtr = value.getDefiningOp<mlir::LLVM::IntToPtrOp>()) {
      value = intToPtr.getArg();
      continue;
    }
    if (auto ptrToInt = value.getDefiningOp<mlir::LLVM::PtrToIntOp>()) {
      value = ptrToInt.getArg();
      continue;
    }
    if (auto extract = value.getDefiningOp<mlir::LLVM::ExtractValueOp>()) {
      value = extract.getContainer();
      continue;
    }
    if (auto gep = value.getDefiningOp<mlir::LLVM::GEPOp>()) {
      value = gep.getBase();
      continue;
    }
    return value;
  }
  return {};
}

bool sameExceptionCell(mlir::Value lhs, mlir::Value rhs) {
  return exceptionCellRoot(lhs) == exceptionCellRoot(rhs);
}

bool freesExceptionCell(mlir::Operation &op, mlir::Value cell) {
  if (auto dealloc = mlir::dyn_cast<mlir::memref::DeallocOp>(op))
    return sameExceptionCell(cell, dealloc.getMemref());
  if (auto call = mlir::dyn_cast<mlir::LLVM::CallOp>(op))
    return call->hasAttr(AsyncSafetyAttrs::kExceptionCellFree) &&
           call.getNumOperands() > 0 &&
           sameExceptionCell(cell, call.getOperand(0));
  return false;
}

bool hasExceptionCellFreeOnAllExits(
    mlir::Block *block, mlir::Value cell,
    llvm::SmallPtrSetImpl<mlir::Block *> &seen) {
  if (!block || !cell)
    return false;
  for (mlir::Operation &op : *block)
    if (freesExceptionCell(op, cell))
      return true;
  mlir::Operation *terminator = block->getTerminator();
  if (!terminator)
    return false;
  if (noReturn(terminator))
    return true;
  if (terminator->getNumSuccessors() == 0)
    return false;
  if (!seen.insert(block).second)
    return true;
  for (unsigned i = 0, e = terminator->getNumSuccessors(); i != e; ++i) {
    mlir::Block *successor = terminator->getSuccessor(i);
    mlir::Value successorCell = remapToSuccessor(terminator, i, cell);
    if (!hasExceptionCellFreeOnAllExits(successor, successorCell, seen))
      return false;
  }
  return true;
}

bool hasExceptionCellFree(mlir::Block *block, mlir::Value cell) {
  llvm::SmallPtrSet<mlir::Block *, 8> seen;
  return hasExceptionCellFreeOnAllExits(block, cell, seen);
}

bool usesAsyncHandleBeforeDrop(mlir::Block *block, mlir::Value handle,
                               llvm::SmallPtrSetImpl<mlir::Block *> &seen) {
  if (!block || !handle)
    return false;
  if (!seen.insert(block).second)
    return false;
  for (mlir::Operation &op : *block) {
    if (async_ref::dropFor(op, handle))
      return false;
    for (mlir::Value operand : op.getOperands())
      if (operand == handle)
        return true;
  }
  mlir::Operation *terminator = block->getTerminator();
  if (!terminator)
    return false;
  for (unsigned i = 0, e = terminator->getNumSuccessors(); i != e; ++i) {
    mlir::Block *successor = terminator->getSuccessor(i);
    mlir::Value successorHandle = remapToSuccessor(terminator, i, handle);
    if (usesAsyncHandleBeforeDrop(successor, successorHandle, seen))
      return true;
  }
  return false;
}

bool usesAsyncHandle(mlir::Block *block, mlir::Value handle) {
  llvm::SmallPtrSet<mlir::Block *, 8> seen;
  return usesAsyncHandleBeforeDrop(block, handle, seen);
}

} // namespace cleanup_block

namespace error_handle_cleanup {

mlir::LogicalResult rewrite(mlir::ModuleOp module) {
  struct HandleCleanup {
    mlir::Operation *branch;
    unsigned successorIndex;
    llvm::SmallVector<mlir::Value> handles;
    llvm::SmallVector<mlir::Value> exceptionCells;
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
    llvm::SmallVector<mlir::Value> exceptionCells;
    for (mlir::Value liveHandle : handles) {
      if (liveHandle == handle)
        continue;
      value_set::add(exceptionCells,
                     exception_cell_operand::forHandle(module, op, liveHandle));
    }
    value_set::add(handles, handle);
    for (mlir::Value liveHandle : llvm::make_early_inc_range(handles))
      if (cleanup_block::hasAsyncDrop(errorTarget, liveHandle) ||
          cleanup_block::usesAsyncHandle(errorTarget, liveHandle))
        value_set::erase(handles, liveHandle);
    for (mlir::Value cell : llvm::make_early_inc_range(exceptionCells))
      if (cleanup_block::hasExceptionCellFree(errorTarget, cell))
        value_set::erase(exceptionCells, cell);
    if (handles.empty() && exceptionCells.empty())
      return;
    cleanups.push_back(
        {op, *errorIndex, std::move(handles), std::move(exceptionCells)});
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
    for (mlir::Value cell : cleanup.exceptionCells)
      if (mlir::failed(exception_cell::destroyAt(module, branch->getLoc(),
                                                 rewriter, cell)))
        return mlir::failure();
    for (mlir::Value handle : cleanup.handles) {
      async_handle_ref::drop(module, branch->getLoc(), rewriter, handle,
                             /*allowAsyncOp=*/true,
                             /*frameTransfer=*/false);
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
    for (mlir::Value payload : cleanup.payloads) {
      mlir::FailureOr<mlir::Operation *> release = async_payload_ref::release(
          module, branch->getLoc(), rewriter, payload);
      if (mlir::failed(release))
        return mlir::failure();
    }
    rewriter.create<mlir::cf::BranchOp>(branch->getLoc(), errorTarget);
  }
  return mlir::success();
}

} // namespace error_payload_cleanup

namespace exception_cell {

mlir::LogicalResult storeFirst(
    mlir::ModuleOp module, mlir::Location loc, mlir::IRRewriter &rewriter,
    mlir::Value destCell, mlir::Value exception,
    llvm::StringRef retainPremise = ThreadSafetyAttrs::kPremiseOwnedToken) {
  PyLLVMTypeConverter typeConverter(module.getContext());
  return async_runtime::ExceptionCell::storeFirst(
      loc, destCell, exception, module, rewriter, typeConverter, retainPremise);
}

mlir::LogicalResult freeLoweredAt(mlir::ModuleOp module, mlir::Location loc,
                                  mlir::IRRewriter &rewriter, mlir::Value cell);

mlir::LogicalResult copyAt(mlir::ModuleOp module, mlir::Location loc,
                           mlir::IRRewriter &rewriter, mlir::Value sourceCell,
                           mlir::Value destCell) {
  if (!sourceCell || !destCell || sourceCell == destCell)
    return mlir::success();

  mlir::Value exception =
      async_runtime::ExceptionCell::load(loc, sourceCell, rewriter);
  mlir::Value isNull =
      async_runtime::ExceptionCell::isNull(loc, exception, rewriter);
  if (!isNull)
    return mlir::success();

  mlir::Block *currentBlock = rewriter.getInsertionBlock();
  mlir::Block *afterBlock =
      rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
  mlir::Block *copyBlock =
      rewriter.createBlock(afterBlock->getParent(), afterBlock->getIterator());
  rewriter.setInsertionPointToEnd(currentBlock);
  rewriter.create<mlir::cf::CondBranchOp>(loc, isNull, afterBlock, copyBlock);

  rewriter.setInsertionPointToStart(copyBlock);
  if (mlir::failed(storeFirst(module, loc, rewriter, destCell, exception,
                              ThreadSafetyAttrs::kPremiseAggregateBorrow)))
    return mlir::failure();
  rewriter.create<mlir::cf::BranchOp>(loc, afterBlock);

  rewriter.setInsertionPointToStart(afterBlock);
  return mlir::success();
}

mlir::LogicalResult destroyAt(mlir::ModuleOp module, mlir::Location loc,
                              mlir::IRRewriter &rewriter, mlir::Value cell) {
  if (!cell)
    return mlir::success();
  PyLLVMTypeConverter typeConverter(module.getContext());
  if (async_runtime::ExceptionCell::hasProvenance(cell) &&
      async_runtime::isLoweredExceptionCellType(cell.getType())) {
    mlir::Value exception =
        async_runtime::ExceptionCell::load(loc, cell, rewriter);
    if (mlir::failed(async_runtime::ExceptionCell::releaseLoaded(
            loc, module, rewriter, typeConverter, exception)))
      return mlir::failure();
    return freeLoweredAt(module, loc, rewriter, cell);
  }
  return async_runtime::ExceptionCell::destroy(loc, module, rewriter,
                                               typeConverter, cell);
}

mlir::LogicalResult freeLoweredAt(mlir::ModuleOp module, mlir::Location loc,
                                  mlir::IRRewriter &rewriter,
                                  mlir::Value cell) {
  auto descriptorType =
      mlir::dyn_cast<mlir::LLVM::LLVMStructType>(cell.getType());
  if (!descriptorType || !async_runtime::ExceptionCell::hasProvenance(cell) ||
      !async_runtime::isLoweredExceptionCellType(cell.getType())) {
    mlir::emitError(loc)
        << "async exception cell lowered free requires memref descriptor ABI";
    return mlir::failure();
  }
  mlir::Value allocated = rewriter.create<mlir::LLVM::ExtractValueOp>(
      loc, descriptorType.getBody()[0], cell,
      rewriter.getDenseI64ArrayAttr({0}));
  if (mlir::Operation *def = allocated.getDefiningOp())
    def->setAttr(AsyncSafetyAttrs::kExceptionCellAllocated,
                 rewriter.getUnitAttr());
  mlir::LLVM::CallOp call = runtime_call::emitVoid(
      module, loc, rewriter, "free", mlir::ValueRange{allocated});
  call->setAttr(AsyncSafetyAttrs::kExceptionCellFree, rewriter.getUnitAttr());
  return mlir::success();
}

mlir::LogicalResult freeAt(mlir::ModuleOp module, mlir::Location loc,
                           mlir::IRRewriter &rewriter, mlir::Value cell) {
  if (!cell)
    return mlir::success();
  if (async_runtime::ExceptionCell::hasProvenance(cell) &&
      async_runtime::isLoweredExceptionCellType(cell.getType())) {
    return freeLoweredAt(module, loc, rewriter, cell);
  }
  if (!async_runtime::ExceptionCell::hasProvenance(cell) ||
      !async_runtime::isExceptionCellType(cell.getType()))
    return mlir::success();
  PyLLVMTypeConverter typeConverter(module.getContext());
  return async_runtime::ExceptionCell::free(loc, module, rewriter,
                                            typeConverter, cell);
}

mlir::LogicalResult moveAt(mlir::ModuleOp module, mlir::Location loc,
                           mlir::IRRewriter &rewriter, mlir::Value sourceCell,
                           mlir::Value destCell) {
  if (!sourceCell || !destCell || sourceCell == destCell)
    return mlir::success();
  if (mlir::failed(copyAt(module, loc, rewriter, sourceCell, destCell)))
    return mlir::failure();
  return destroyAt(module, loc, rewriter, sourceCell);
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
    llvm::SmallVector<mlir::Value, 5> sourceParts;
    llvm::SmallVector<mlir::Value, 5> destParts;
  };
  llvm::SmallVector<Marker> markers;
  mlir::LogicalResult result = mlir::success();
  module.walk([&](mlir::func::CallOp call) {
    if (async_marker::is(call.getCallee(), async_marker::Kind::ExceptionEdge))
      markers.push_back({call.getOperation(),
                         call.getOperand(0),
                         call.getOperand(1),
                         {},
                         {}});
  });
  module.walk([&](mlir::LLVM::CallOp call) {
    auto callee = call.getCallee();
    if (!callee ||
        !async_marker::is(*callee, async_marker::Kind::ExceptionEdge))
      return;
    if (call.getNumOperands() == 2) {
      markers.push_back({call.getOperation(),
                         call.getOperand(0),
                         call.getOperand(1),
                         {},
                         {}});
      return;
    }
    if (call.getNumOperands() == 6) {
      mlir::Value sourceCell =
          cleanup_block::exceptionCellRoot(call.getOperand(0));
      llvm::SmallVector<mlir::Value, 5> sourceParts;
      if (!sourceCell ||
          !async_runtime::isLoweredExceptionCellType(sourceCell.getType())) {
        sourceCell = {};
        for (unsigned index = 0; index < 5; ++index)
          sourceParts.push_back(call.getOperand(index));
      }
      if ((!sourceCell && sourceParts.size() != 5) ||
          !async_runtime::isLoweredExceptionCellType(
              call.getOperand(5).getType())) {
        call.emitError("invalid async exception edge marker descriptor");
        result = mlir::failure();
        return;
      }
      async_runtime::ExceptionCell::mark(sourceCell);
      markers.push_back({call.getOperation(),
                         sourceCell,
                         call.getOperand(5),
                         std::move(sourceParts),
                         {}});
      return;
    }
    if (call.getNumOperands() == 10) {
      mlir::Value sourceCell =
          cleanup_block::exceptionCellRoot(call.getOperand(0));
      mlir::Value destCell =
          cleanup_block::exceptionCellRoot(call.getOperand(5));
      llvm::SmallVector<mlir::Value, 5> sourceParts;
      llvm::SmallVector<mlir::Value, 5> destParts;
      if (!sourceCell ||
          !async_runtime::isLoweredExceptionCellType(sourceCell.getType())) {
        sourceCell = {};
        for (unsigned index = 0; index < 5; ++index)
          sourceParts.push_back(call.getOperand(index));
      }
      if (!destCell ||
          !async_runtime::isLoweredExceptionCellType(destCell.getType())) {
        destCell = {};
        for (unsigned index = 5; index < 10; ++index)
          destParts.push_back(call.getOperand(index));
      }
      if ((!sourceCell && sourceParts.size() != 5) ||
          (!destCell && destParts.size() != 5)) {
        call.emitError("invalid async exception edge marker descriptors");
        result = mlir::failure();
        return;
      }
      async_runtime::ExceptionCell::mark(sourceCell);
      async_runtime::ExceptionCell::mark(destCell);
      markers.push_back({call.getOperation(), sourceCell, destCell,
                         std::move(sourceParts), std::move(destParts)});
      return;
    }
    call.emitError("invalid async exception edge marker operand count");
    result = mlir::failure();
  });

  for (Marker marker : markers) {
    mlir::Block *errorBlock =
        await_error::blockForContinuation(marker.op->getBlock());
    if (!errorBlock) {
      marker.op->emitError(
          "failed to find async await error branch for payload propagation");
      result = mlir::failure();
      continue;
    }
    mlir::IRRewriter rewriter(module.getContext());
    rewriter.setInsertionPointToStart(errorBlock);
    auto materializeCell =
        [&](mlir::Value cell,
            llvm::ArrayRef<mlir::Value> parts) -> mlir::Value {
      if (cell)
        return cell;
      mlir::Value descriptor =
          lowered_descriptor::build(marker.op->getLoc(), rewriter, parts);
      async_runtime::ExceptionCell::mark(descriptor);
      return descriptor;
    };
    mlir::Value sourceCell =
        materializeCell(marker.sourceCell, marker.sourceParts);
    mlir::Value destCell = materializeCell(marker.destCell, marker.destParts);
    if (!sourceCell || !destCell) {
      marker.op->emitError(
          "failed to materialize async exception edge marker descriptor");
      result = mlir::failure();
      continue;
    }
    if (mlir::failed(exception_cell::moveAt(module, marker.op->getLoc(),
                                            rewriter, sourceCell, destCell))) {
      result = mlir::failure();
      continue;
    }
    marker.op->erase();
  }

  async_marker::eraseDeclarations(module, async_marker::Kind::ExceptionEdge);

  return result;
}

} // namespace exception_payload

namespace await_exception_cell {

constexpr llvm::StringLiteral kRewritten =
    "ly.async.await_exception_cell_rewritten";

bool noThrowHandle(mlir::Value handle) {
  if (!handle)
    return false;
  mlir::Operation *def = handle.getDefiningOp();
  if (!def)
    return false;
  return def->hasAttr("nothrow") && !def->hasAttr("maythrow");
}

bool replaceWithSuccessBranch(mlir::Operation *branch, unsigned successIndex) {
  if (!branch || successIndex >= branch->getNumSuccessors())
    return false;
  mlir::Value condition = cfedge::condition(branch);
  mlir::Operation *conditionDef =
      condition ? condition.getDefiningOp() : nullptr;
  bool eraseCondition =
      conditionDef && async_error_branch::runtimeCondition(condition);
  auto branchInterface = mlir::dyn_cast<mlir::BranchOpInterface>(branch);
  if (!branchInterface)
    return false;
  mlir::SuccessorOperands successorOperands =
      branchInterface.getSuccessorOperands(successIndex);
  llvm::SmallVector<mlir::Value> operands;
  operands.reserve(successorOperands.size());
  for (unsigned index = 0; index != successorOperands.size(); ++index) {
    if (successorOperands.isOperandProduced(index))
      return false;
    operands.push_back(successorOperands[index]);
  }

  mlir::IRRewriter rewriter(branch->getContext());
  rewriter.setInsertionPoint(branch);
  mlir::Block *successor = branch->getSuccessor(successIndex);
  if (mlir::isa<mlir::cf::CondBranchOp>(branch)) {
    rewriter.create<mlir::cf::BranchOp>(branch->getLoc(), successor, operands);
  } else if (mlir::isa<mlir::LLVM::CondBrOp>(branch)) {
    rewriter.create<mlir::LLVM::BrOp>(branch->getLoc(), operands, successor);
  } else {
    return false;
  }
  rewriter.eraseOp(branch);
  if (eraseCondition && conditionDef->use_empty())
    rewriter.eraseOp(conditionDef);
  return true;
}

bool hasExceptionCellArg(mlir::FunctionOpInterface function,
                         unsigned argIndex) {
  return function &&
         exception_cell_operand::arg(function.getOperation(), argIndex);
}

bool markedCell(mlir::Value value) {
  return exception_cell_operand::marked(value);
}

mlir::Value currentCell(mlir::Operation *op) {
  auto function = op ? op->getParentOfType<mlir::FunctionOpInterface>()
                     : mlir::FunctionOpInterface();
  if (!function || function.getFunctionBody().empty())
    return {};
  mlir::Block &entry = function.getFunctionBody().front();
  if (entry.getNumArguments() == 0)
    return {};
  unsigned index = entry.getNumArguments() - 1;
  if (!hasExceptionCellArg(function, index))
    return {};
  return entry.getArgument(index);
}

mlir::Value callCell(mlir::ModuleOp module, mlir::func::CallOp call) {
  if (!call || call.getNumResults() != 1 ||
      !mlir::isa<mlir::async::ValueType>(call.getResult(0).getType()) ||
      call.getNumOperands() == 0)
    return {};
  mlir::Operation *calleeOp = module.lookupSymbol(call.getCallee());
  auto callee = mlir::dyn_cast_or_null<mlir::FunctionOpInterface>(calleeOp);
  unsigned last = call.getNumOperands() - 1;
  if (!hasExceptionCellArg(callee, last) && !markedCell(call.getOperand(last)))
    return {};
  return call.getOperand(last);
}

llvm::DenseMap<mlir::Value, mlir::Value>
collectCallCells(mlir::ModuleOp module) {
  llvm::DenseMap<mlir::Value, mlir::Value> cells;
  module.walk([&](mlir::func::CallOp call) {
    mlir::Value cell = callCell(module, call);
    if (cell)
      cells.try_emplace(call.getResult(0), cell);
  });
  return cells;
}

mlir::Block::iterator afterInitialDrop(mlir::Block *block, mlir::Value handle) {
  if (!block)
    return {};
  for (auto it = block->begin(), end = block->end(); it != end; ++it) {
    if (it->hasTrait<mlir::OpTrait::IsTerminator>())
      return it;
    if (!async_ref::dropFor(*it, handle))
      return it;
    return std::next(it);
  }
  return block->end();
}

void abortAsyncExceptionPayloadInvariant(mlir::ModuleOp module,
                                         mlir::Location loc,
                                         mlir::IRRewriter &rewriter) {
  auto abortFn = runtime_func::getOrInsert(
      module, loc, rewriter, "abort",
      mlir::LLVM::LLVMVoidType::get(module.getContext()), {});
  auto call = rewriter.create<mlir::LLVM::CallOp>(
      loc, mlir::TypeRange{},
      mlir::SymbolRefAttr::get(module.getContext(), abortFn.getName()),
      mlir::ValueRange{});
  call->setAttr(ControlFlowContractAttrs::kNoReturn,
                mlir::UnitAttr::get(module.getContext()));
  rewriter.create<mlir::LLVM::UnreachableOp>(loc);
}

bool splitExceptionPayload(mlir::Value value) {
  auto aggregate = mlir::dyn_cast<mlir::LLVM::LLVMStructType>(value.getType());
  if (!aggregate || aggregate.isOpaque() || aggregate.getBody().size() != 3)
    return false;
  return llvm::all_of(aggregate.getBody(), [](mlir::Type part) {
    return object_abi::Type::isLoweredStorage(part);
  });
}

llvm::SmallVector<mlir::Value, 3>
splitExceptionParts(mlir::Location loc, mlir::Value value,
                    mlir::IRRewriter &rewriter) {
  llvm::SmallVector<mlir::Value, 3> parts;
  auto aggregate = mlir::cast<mlir::LLVM::LLVMStructType>(value.getType());
  for (unsigned index = 0; index < 3; ++index) {
    mlir::Value part = rewriter.create<mlir::LLVM::ExtractValueOp>(
        loc, aggregate.getBody()[index], value,
        rewriter.getDenseI64ArrayAttr({index}));
    ownership::aggregate::Slot::markLoad(part, "async.exception_cell",
                                         "exception", index);
    parts.push_back(part);
  }
  return parts;
}

void throwExceptionPayload(mlir::ModuleOp module, mlir::Location loc,
                           mlir::IRRewriter &rewriter,
                           const PyLLVMTypeConverter &typeConverter,
                           mlir::Value exception) {
  RuntimeAPI runtime(module, rewriter, typeConverter);
  if (splitExceptionPayload(exception)) {
    runtime.call(loc, RuntimeSymbols::kEHThrowException, mlir::Type(),
                 splitExceptionParts(loc, exception, rewriter));
  } else {
    abortAsyncExceptionPayloadInvariant(module, loc, rewriter);
    return;
  }
  rewriter.create<mlir::LLVM::UnreachableOp>(loc);
}

mlir::LogicalResult throwStoredOrAbort(mlir::ModuleOp module,
                                       mlir::Location loc,
                                       mlir::IRRewriter &rewriter,
                                       mlir::Value cell,
                                       mlir::Block *resumeBlock) {
  PyLLVMTypeConverter typeConverter(module.getContext());
  mlir::Value exception =
      async_runtime::ExceptionCell::load(loc, cell, rewriter);
  mlir::Value isNull =
      async_runtime::ExceptionCell::isNull(loc, exception, rewriter);
  if (!isNull) {
    if (mlir::failed(exception_cell::freeAt(module, loc, rewriter, cell)))
      return mlir::failure();
    abortAsyncExceptionPayloadInvariant(module, loc, rewriter);
    return mlir::success();
  }

  mlir::Block *currentBlock = rewriter.getInsertionBlock();
  mlir::Block *throwBlock = rewriter.createBlock(resumeBlock->getParent(),
                                                 resumeBlock->getIterator());
  mlir::Block *abortBlock =
      rewriter.createBlock(resumeBlock->getParent(), throwBlock->getIterator());
  rewriter.setInsertionPointToEnd(currentBlock);
  rewriter.create<mlir::cf::CondBranchOp>(loc, isNull, abortBlock, throwBlock);

  rewriter.setInsertionPointToStart(abortBlock);
  if (mlir::failed(exception_cell::freeAt(module, loc, rewriter, cell)))
    return mlir::failure();
  abortAsyncExceptionPayloadInvariant(module, loc, rewriter);

  rewriter.setInsertionPointToStart(throwBlock);
  if (mlir::failed(async_runtime::ExceptionCell::retainPayload(
          loc, module, rewriter, typeConverter, exception,
          ThreadSafetyAttrs::kPremiseAggregateBorrow)))
    return mlir::failure();
  if (mlir::failed(async_runtime::ExceptionCell::releaseLoaded(
          loc, module, rewriter, typeConverter, exception)))
    return mlir::failure();
  if (mlir::failed(exception_cell::freeAt(module, loc, rewriter, cell)))
    return mlir::failure();
  throwExceptionPayload(module, loc, rewriter, typeConverter, exception);
  return mlir::success();
}

mlir::LogicalResult insertErrorTransfer(mlir::ModuleOp module,
                                        mlir::Operation *branch,
                                        unsigned errorIndex, mlir::Value handle,
                                        mlir::Value cell) {
  mlir::Block *errorBlock = branch->getSuccessor(errorIndex);
  if (!errorBlock)
    return mlir::success();

  mlir::IRRewriter rewriter(module.getContext());
  mlir::Block::iterator splitPoint = afterInitialDrop(errorBlock, handle);
  mlir::Block *resumeBlock = rewriter.splitBlock(errorBlock, splitPoint);

  mlir::Value destCell = currentCell(branch);
  rewriter.setInsertionPointToEnd(errorBlock);
  if (destCell) {
    if (mlir::failed(exception_cell::moveAt(module, branch->getLoc(), rewriter,
                                            cell, destCell)))
      return mlir::failure();
    rewriter.create<mlir::cf::BranchOp>(branch->getLoc(), resumeBlock);
    return mlir::success();
  }

  return throwStoredOrAbort(module, branch->getLoc(), rewriter, cell,
                            resumeBlock);
}

mlir::LogicalResult insertSuccessFree(mlir::ModuleOp module,
                                      mlir::Operation *branch,
                                      unsigned successIndex, mlir::Value cell) {
  mlir::Block *successBlock = branch->getSuccessor(successIndex);
  if (!successBlock)
    return mlir::success();
  mlir::IRRewriter rewriter(module.getContext());
  rewriter.setInsertionPointToStart(successBlock);
  return exception_cell::freeAt(module, branch->getLoc(), rewriter, cell);
}

bool blockHasExceptionCellAtomic(mlir::Block *block,
                                 llvm::SmallPtrSetImpl<mlir::Block *> &seen) {
  if (!block || !seen.insert(block).second)
    return false;
  for (mlir::Operation &op : *block) {
    auto role =
        op.getAttrOfType<mlir::StringAttr>(ThreadSafetyAttrs::kAtomicRole);
    if (role &&
        (role.getValue() == ThreadSafetyAttrs::kRoleAsyncExceptionLoad ||
         role.getValue() == ThreadSafetyAttrs::kRoleAsyncExceptionStore))
      return true;
  }
  mlir::Operation *terminator = block->getTerminator();
  if (!terminator || terminator->getNumSuccessors() != 1)
    return false;
  return blockHasExceptionCellAtomic(terminator->getSuccessor(0), seen);
}

bool blockHasExceptionCellAtomic(mlir::Block *block) {
  llvm::SmallPtrSet<mlir::Block *, 4> seen;
  return blockHasExceptionCellAtomic(block, seen);
}

mlir::LogicalResult rewrite(mlir::ModuleOp module) {
  llvm::DenseMap<mlir::Value, mlir::Value> cells = collectCallCells(module);
  if (cells.empty())
    return mlir::success();

  llvm::SmallVector<mlir::Operation *> branches;
  module.walk([&](mlir::cf::CondBranchOp branch) {
    if (!branch->hasAttr(kRewritten))
      branches.push_back(branch.getOperation());
  });
  module.walk([&](mlir::LLVM::CondBrOp branch) {
    if (!branch->hasAttr(kRewritten))
      branches.push_back(branch.getOperation());
  });

  for (mlir::Operation *branch : branches) {
    auto errorIndex = async_error_branch::successorIndex(branch);
    if (!errorIndex || *errorIndex != 0 || branch->getNumSuccessors() < 2)
      continue;
    mlir::Value handle = async_error_branch::handle(cfedge::condition(branch));
    if (noThrowHandle(handle)) {
      replaceWithSuccessBranch(branch, /*successIndex=*/1);
      continue;
    }
    auto found = cells.find(handle);
    if (found == cells.end())
      continue;
    mlir::Value cell = found->second;
    if (blockHasExceptionCellAtomic(branch->getSuccessor(*errorIndex)) ||
        blockHasExceptionCellAtomic(branch->getSuccessor(1)))
      continue;
    if (mlir::failed(
            insertErrorTransfer(module, branch, *errorIndex, handle, cell)))
      return mlir::failure();
    if (mlir::failed(
            insertSuccessFree(module, branch, /*successIndex=*/1, cell)))
      return mlir::failure();
    branch->setAttr(kRewritten, mlir::UnitAttr::get(module.getContext()));
  }

  return mlir::success();
}

} // namespace await_exception_cell

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

namespace async_runtime_contract {

static void setUnit(mlir::Operation *op, llvm::StringRef attrName) {
  if (op)
    op->setAttr(attrName, mlir::UnitAttr::get(op->getContext()));
}

static void setI64(mlir::Operation *op, llvm::StringRef attrName,
                   int64_t value) {
  if (!op)
    return;
  mlir::Builder builder(op->getContext());
  op->setAttr(attrName, builder.getI64IntegerAttr(value));
}

static void setIndexArray(mlir::Operation *op, llvm::StringRef attrName,
                          llvm::ArrayRef<unsigned> unsignedValues) {
  llvm::SmallVector<int64_t, 2> values;
  values.reserve(unsignedValues.size());
  for (unsigned value : unsignedValues)
    values.push_back(static_cast<int64_t>(value));
  async_contract::setIndexArray(op, attrName, values);
}

static void annotate(mlir::Operation *op, llvm::StringRef callee) {
  if (!op)
    return;

  if (auto value = runtime::mlir_async::Callee::refcountDelta(callee))
    setI64(op, AsyncSafetyAttrs::kRuntimeRefcountDelta, *value);

  llvm::SmallVector<unsigned, 2> borrowed =
      runtime::mlir_async::Callee::borrowedHandleOperands(callee);
  if (!borrowed.empty())
    setIndexArray(op, AsyncSafetyAttrs::kRuntimeHandleBorrowArgs, borrowed);

  if (runtime::mlir_async::Callee::createHandle(callee))
    async_contract::setIndexArray(
        op, AsyncSafetyAttrs::kRuntimeHandleOwnedResults, {0});

  if (runtime::mlir_async::Callee::isError(callee))
    setUnit(op, AsyncSafetyAttrs::kRuntimeErrorQuery);

  if (runtime::mlir_async::Callee::awaitAndExecute(callee))
    setUnit(op, AsyncSafetyAttrs::kRuntimeAwaitExecute);

  if (runtime::mlir_async::Callee::valueStorage(callee))
    setUnit(op, AsyncSafetyAttrs::kRuntimeValueStorage);

  if (runtime::mlir_async::Callee::executeEntry(callee)) {
    setUnit(op, AsyncSafetyAttrs::kRuntimeExecuteEntry);
    async_contract::setIndexArray(
        op, AsyncSafetyAttrs::kRuntimeHandleOwnedResults, {0});
    async_contract::setIndexArray(
        op, AsyncSafetyAttrs::kRuntimeHandleTransferArgs, {0});
  }
}

static void copyPresplitCoroutineAttrs(mlir::Operation *callee,
                                       mlir::Operation *call) {
  if (!hasPresplitCoroutinePassthrough(callee) || !call)
    return;
  setUnit(call, AsyncSafetyAttrs::kRuntimeExecuteEntry);
  if (call->getNumResults() > 0)
    async_contract::setIndexArray(
        call, AsyncSafetyAttrs::kRuntimeHandleOwnedResults, {0});

  auto function = mlir::dyn_cast<mlir::FunctionOpInterface>(callee);
  if (!function)
    return;

  llvm::SmallVector<int64_t, 2> transferredHandles;
  llvm::SmallVector<int64_t, 2> transferredExceptionCells;
  for (unsigned index = 0, e = function.getNumArguments(); index < e; ++index) {
    if (!function.getArgAttr(index, AsyncSafetyAttrs::kRuntimeHandle))
      continue;
    if (index >= call->getNumOperands())
      continue;
    transferredHandles.push_back(static_cast<int64_t>(index));

    unsigned cellIndex = index + 1;
    if (cellIndex < e && cellIndex < call->getNumOperands() &&
        function.getArgAttr(cellIndex, AsyncSafetyAttrs::kExceptionCell))
      transferredExceptionCells.push_back(static_cast<int64_t>(cellIndex));
  }
  if (!transferredHandles.empty())
    async_contract::setIndexArray(
        call, AsyncSafetyAttrs::kRuntimeHandleTransferArgs, transferredHandles);
  if (!transferredExceptionCells.empty())
    async_contract::setIndexArray(call,
                                  AsyncSafetyAttrs::kExceptionCellTransferArgs,
                                  transferredExceptionCells);
}

static void copyThrowAttrs(mlir::Operation *from, mlir::Operation *to) {
  if (!from || !to)
    return;
  if (mlir::Attribute attr = from->getAttr("nothrow"))
    to->setAttr("nothrow", attr);
  if (mlir::Attribute attr = from->getAttr("maythrow"))
    to->setAttr("maythrow", attr);
}

static mlir::LogicalResult annotate(mlir::ModuleOp module) {
  module.walk([&](mlir::func::FuncOp func) {
    annotate(func.getOperation(), func.getSymName());
  });
  module.walk([&](mlir::LLVM::LLVMFuncOp func) {
    annotate(func.getOperation(), func.getSymName());
  });
  module.walk([&](mlir::func::CallOp call) {
    annotate(call.getOperation(), call.getCallee());
    mlir::Operation *callee = module.lookupSymbol(call.getCallee());
    copyPresplitCoroutineAttrs(callee, call.getOperation());
    copyThrowAttrs(callee, call.getOperation());
  });
  module.walk([&](mlir::LLVM::CallOp call) {
    auto callee = call.getCallee();
    if (callee) {
      annotate(call.getOperation(), *callee);
      mlir::Operation *calleeOp = module.lookupSymbol(*callee);
      copyPresplitCoroutineAttrs(calleeOp, call.getOperation());
      copyThrowAttrs(calleeOp, call.getOperation());
    }
  });
  return mlir::success();
}

} // namespace async_runtime_contract

struct AsyncRuntimeRewritePass
    : public mlir::PassWrapper<AsyncRuntimeRewritePass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AsyncRuntimeRewritePass)

  llvm::StringRef getArgument() const override {
    return "py-async-runtime-rewrite";
  }

  llvm::StringRef getDescription() const override {
    return "Rewrite Lython async exception markers after async runtime "
           "conversion";
  }

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    if (mlir::failed(async_runtime_contract::annotate(module)) ||
        mlir::failed(finally_error::rewrite(module)) ||
        mlir::failed(exception_payload::rewrite(module)) ||
        mlir::failed(error_handle_cleanup::rewrite(module)) ||
        mlir::failed(await_exception_cell::rewrite(module)) ||
        mlir::failed(error_payload_cleanup::rewrite(module)) ||
        mlir::failed(await_cleanup::rewrite(module)) ||
        mlir::failed(async_runtime_contract::annotate(module)))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createAsyncRuntimeRewritePass() {
  return std::make_unique<AsyncRuntimeRewritePass>();
}

} // namespace py
