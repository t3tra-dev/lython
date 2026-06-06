#include "Passes/Runtime/EH.h"

#include "Common/Object.h"
#include "Common/RuntimeSupport.h"
#include "Passes/OwnershipAnalysis.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace py {
namespace {

static mlir::LLVM::LLVMFuncOp
getOrCreateLLVMFunc(mlir::ModuleOp module, llvm::StringRef name,
                    mlir::Type resultType, llvm::ArrayRef<mlir::Type> argTypes,
                    bool varArg = false) {
  auto markNonObjectPointerArgs = [&](mlir::LLVM::LLVMFuncOp fn) {
    if (!fn)
      return;
    if (name != RuntimeSymbols::kTracebackPrintMessage &&
        name != RuntimeSymbols::kEHTakeCurrentDescriptor)
      return;
    mlir::Builder attrBuilder(fn.getContext());
    auto unit = attrBuilder.getUnitAttr();
    for (auto [index, type] : llvm::enumerate(fn.getFunctionType().getParams()))
      if (mlir::isa<mlir::LLVM::LLVMPointerType>(type))
        fn.setArgAttr(static_cast<unsigned>(index),
                      OwnershipContractAttrs::kNonObjectPointer, unit);
  };

  if (auto fn = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(name)) {
    markNonObjectPointerArgs(fn);
    return fn;
  }
  mlir::OpBuilder builder(module.getBody(), module.getBody()->begin());
  auto fnType = mlir::LLVM::LLVMFunctionType::get(resultType, argTypes, varArg);
  auto fn =
      builder.create<mlir::LLVM::LLVMFuncOp>(module.getLoc(), name, fnType);
  markNonObjectPointerArgs(fn);
  return fn;
}

static mlir::LLVM::LLVMFuncOp
getOrCreateLLVMPersonality(mlir::ModuleOp module) {
  mlir::OpBuilder builder(module.getContext());
  return getOrCreateLLVMFunc(module, "__gxx_personality_v0",
                             builder.getI32Type(), {}, true);
}

static mlir::LLVM::CallOp
emitLLVMRuntimeCall(mlir::ModuleOp module, mlir::OpBuilder &builder,
                    mlir::Location loc, llvm::StringRef name,
                    mlir::Type resultType, mlir::ValueRange operands) {
  llvm::SmallVector<mlir::Type> argTypes;
  argTypes.reserve(operands.size());
  for (mlir::Value operand : operands)
    argTypes.push_back(operand.getType());
  mlir::Type actualResult =
      resultType ? resultType
                 : mlir::LLVM::LLVMVoidType::get(module.getContext());
  auto callee = getOrCreateLLVMFunc(module, name, actualResult, argTypes);
  auto symbol = mlir::SymbolRefAttr::get(module.getContext(), callee.getName());
  llvm::SmallVector<mlir::Type> results;
  if (!mlir::isa<mlir::LLVM::LLVMVoidType>(actualResult))
    results.push_back(actualResult);
  return builder.create<mlir::LLVM::CallOp>(loc, results, symbol, operands);
}

static void emitTracebackClear(mlir::ModuleOp module, mlir::OpBuilder &builder,
                               mlir::Location loc) {
  emitLLVMRuntimeCall(module, builder, loc, RuntimeSymbols::kTracebackClear,
                      mlir::Type(), mlir::ValueRange{});
}

static mlir::LLVM::LLVMStructType memrefDescriptorType(mlir::MLIRContext *ctx) {
  return object_abi::Type::loweredStorage(ctx);
}

static mlir::LLVM::LLVMStructType
exceptionPartsDescriptorType(mlir::MLIRContext *ctx) {
  auto descriptor = memrefDescriptorType(ctx);
  return mlir::LLVM::LLVMStructType::getLiteral(
      ctx, llvm::ArrayRef<mlir::Type>{descriptor, descriptor, descriptor});
}

static mlir::Value extractValue(mlir::Location loc, mlir::Value aggregate,
                                mlir::Type resultType,
                                llvm::ArrayRef<int64_t> position,
                                mlir::OpBuilder &builder) {
  return builder.create<mlir::LLVM::ExtractValueOp>(
      loc, resultType, aggregate, builder.getDenseI64ArrayAttr(position));
}

static void markEHExceptionPart(mlir::Value value, unsigned index) {
  ownership::aggregate::Slot::markLoad(value, "eh.current_exception",
                                       "exception", index);
}

static void
appendMemRefDescriptorOperands(mlir::Location loc, mlir::Value descriptor,
                               mlir::LLVM::LLVMStructType descriptorType,
                               llvm::SmallVectorImpl<mlir::Value> &operands,
                               mlir::OpBuilder &builder) {
  llvm::ArrayRef<mlir::Type> body = descriptorType.getBody();
  mlir::Value allocated = extractValue(loc, descriptor, body[0], {0}, builder);
  mlir::Value aligned = extractValue(loc, descriptor, body[1], {1}, builder);
  ownership::Pointer::markNonObject(allocated);
  ownership::Pointer::markNonObject(aligned);
  operands.push_back(allocated);
  operands.push_back(aligned);
  operands.push_back(extractValue(loc, descriptor, body[2], {2}, builder));
  auto i64Type = builder.getI64Type();
  operands.push_back(extractValue(loc, descriptor, i64Type, {3, 0}, builder));
  operands.push_back(extractValue(loc, descriptor, i64Type, {4, 0}, builder));
}

} // namespace

namespace lowering::runtime::eh {

void ensureFuncPersonalities(mlir::ModuleOp module) {
  auto personality =
      mlir::FlatSymbolRefAttr::get(module.getContext(), "__gxx_personality_v0");
  module.walk([&](mlir::func::FuncOp func) {
    if (func->hasAttr("llvm.personality"))
      return;
    bool hasLandingpad = false;
    func.walk([&](mlir::LLVM::LandingpadOp) { hasLandingpad = true; });
    if (hasLandingpad)
      func->setAttr("llvm.personality", personality);
  });
}

void finalizeUnwindBlocks(mlir::ModuleOp module) {
  mlir::MLIRContext *ctx = module.getContext();
  auto ehTokenPtr = mlir::LLVM::LLVMPointerType::get(ctx);
  auto personality = getOrCreateLLVMPersonality(module);
  auto personalityRef =
      mlir::FlatSymbolRefAttr::get(ctx, personality.getName());

  module.walk([&](mlir::LLVM::InvokeOp invoke) {
    auto func = invoke->getParentOfType<mlir::LLVM::LLVMFuncOp>();
    if (func && !func->hasAttr("personality"))
      func->setAttr("personality", personalityRef);

    mlir::Block *unwind = invoke.getUnwindDest();
    if (!unwind)
      return;
    mlir::Operation *terminator = unwind->getTerminator();
    if (terminator && !mlir::isa<mlir::LLVM::ResumeOp>(terminator)) {
      if (unwind->empty() ||
          !mlir::isa<mlir::LLVM::LandingpadOp>(unwind->front())) {
        mlir::OpBuilder builder(ctx);
        builder.setInsertionPointToStart(unwind);
        auto lpType = mlir::LLVM::LLVMStructType::getLiteral(
            ctx, llvm::ArrayRef<mlir::Type>{ehTokenPtr, builder.getI32Type()});
        builder.create<mlir::LLVM::LandingpadOp>(
            invoke.getLoc(), lpType, builder.getUnitAttr(), mlir::ValueRange{});
      }
      return;
    }
    unwind->clear();
    mlir::OpBuilder builder(ctx);
    builder.setInsertionPointToStart(unwind);
    auto lpType = mlir::LLVM::LLVMStructType::getLiteral(
        ctx, llvm::ArrayRef<mlir::Type>{ehTokenPtr, builder.getI32Type()});
    auto lp = builder.create<mlir::LLVM::LandingpadOp>(
        invoke.getLoc(), lpType, builder.getUnitAttr(), mlir::ValueRange{});
    builder.create<mlir::LLVM::ResumeOp>(invoke.getLoc(), lp.getRes());
  });
}

void wrapTopLevelMain(mlir::ModuleOp module) {
  mlir::MLIRContext *ctx = module.getContext();
  auto mainFunc = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("main");
  if (!mainFunc)
    return;

  auto fnType = mainFunc.getFunctionType();
  if (fnType.getNumParams() != 0 ||
      mlir::isa<mlir::LLVM::LLVMVoidType>(fnType.getReturnType()))
    return;

  llvm::StringRef implName = "__lython_main";
  if (module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(implName))
    return;

  mainFunc.setName(implName);

  mlir::OpBuilder builder(ctx);
  builder.setInsertionPointToEnd(module.getBody());
  auto wrapper =
      builder.create<mlir::LLVM::LLVMFuncOp>(module.getLoc(), "main", fnType);

  auto personality = getOrCreateLLVMPersonality(module);
  auto personalityRef =
      mlir::FlatSymbolRefAttr::get(ctx, personality.getName());
  wrapper->setAttr("personality", personalityRef);

  mlir::Block *entry = wrapper.addEntryBlock(builder);
  mlir::Block *normal = builder.createBlock(&wrapper.getBody());
  mlir::Block *unwind = builder.createBlock(&wrapper.getBody());

  builder.setInsertionPointToStart(entry);
  auto ptrType = mlir::LLVM::LLVMPointerType::get(ctx);
  mlir::Value catchAll =
      builder.create<mlir::LLVM::ZeroOp>(module.getLoc(), ptrType);
  auto invoke = builder.create<mlir::LLVM::InvokeOp>(
      module.getLoc(), fnType.getReturnType(),
      mlir::FlatSymbolRefAttr::get(ctx, implName), mlir::ValueRange{}, normal,
      mlir::ValueRange{}, unwind, mlir::ValueRange{});

  builder.setInsertionPointToStart(normal);
  builder.create<mlir::LLVM::ReturnOp>(module.getLoc(), invoke.getResult());

  builder.setInsertionPointToStart(unwind);
  auto i32Type = builder.getI32Type();
  auto i64Type = builder.getI64Type();
  auto lpType = mlir::LLVM::LLVMStructType::getLiteral(
      ctx, llvm::ArrayRef<mlir::Type>{ptrType, i32Type});
  builder.create<mlir::LLVM::LandingpadOp>(module.getLoc(), lpType,
                                           builder.getUnitAttr(),
                                           mlir::ValueRange{catchAll});
  auto descriptorType = memrefDescriptorType(ctx);
  auto partsType = exceptionPartsDescriptorType(ctx);
  mlir::Value one = builder.create<mlir::LLVM::ConstantOp>(
      module.getLoc(), i64Type, builder.getI64IntegerAttr(1));
  mlir::Value descriptorStorage = builder.create<mlir::LLVM::AllocaOp>(
      module.getLoc(), ptrType, partsType, one, /*alignment=*/0);
  ownership::Pointer::markNonObject(descriptorStorage);
  mlir::Value captured =
      emitLLVMRuntimeCall(module, builder, module.getLoc(),
                          RuntimeSymbols::kEHTakeCurrentDescriptor,
                          builder.getI1Type(),
                          mlir::ValueRange{descriptorStorage})
          .getResult();

  mlir::Block *release = builder.createBlock(&wrapper.getBody());
  mlir::Block *abort = builder.createBlock(&wrapper.getBody());
  builder.setInsertionPointToEnd(unwind);
  builder.create<mlir::LLVM::CondBrOp>(module.getLoc(), captured, release,
                                       abort);

  builder.setInsertionPointToStart(release);
  mlir::Value partsDescriptor = builder.create<mlir::LLVM::LoadOp>(
      module.getLoc(), partsType, descriptorStorage);
  ownership::aggregate::Slot::markLoad(partsDescriptor, "eh.current_exception",
                                       "exception", std::nullopt);
  mlir::Value header = extractValue(module.getLoc(), partsDescriptor,
                                    descriptorType, {0}, builder);
  mlir::Value messageHeader = extractValue(module.getLoc(), partsDescriptor,
                                           descriptorType, {1}, builder);
  mlir::Value messageBytes = extractValue(module.getLoc(), partsDescriptor,
                                          descriptorType, {2}, builder);
  markEHExceptionPart(header, 0);
  markEHExceptionPart(messageHeader, 1);
  markEHExceptionPart(messageBytes, 2);
  llvm::SmallVector<mlir::Value, 5> printOperands;
  appendMemRefDescriptorOperands(module.getLoc(), messageBytes, descriptorType,
                                 printOperands, builder);
  emitLLVMRuntimeCall(module, builder, module.getLoc(),
                      RuntimeSymbols::kTracebackPrintMessage, mlir::Type(),
                      printOperands);
  emitTracebackClear(module, builder, module.getLoc());
  llvm::SmallVector<mlir::Value, 10> descriptorOperands;
  appendMemRefDescriptorOperands(module.getLoc(), header, descriptorType,
                                 descriptorOperands, builder);
  appendMemRefDescriptorOperands(module.getLoc(), messageHeader, descriptorType,
                                 descriptorOperands, builder);
  appendMemRefDescriptorOperands(module.getLoc(), messageBytes, descriptorType,
                                 descriptorOperands, builder);
  auto releaseCall = emitLLVMRuntimeCall(module, builder, module.getLoc(),
                                         RuntimeSymbols::kExceptionDecRef,
                                         mlir::Type(), descriptorOperands);
  releaseCall->setAttr(OwnershipContractAttrs::kAggregateRelease,
                       builder.getUnitAttr());
  mlir::Value failureCode = builder.create<mlir::LLVM::ConstantOp>(
      module.getLoc(), i32Type, builder.getI32IntegerAttr(1));
  builder.create<mlir::LLVM::ReturnOp>(module.getLoc(), failureCode);

  builder.setInsertionPointToStart(abort);
  emitTracebackClear(module, builder, module.getLoc());
  auto abortCall =
      emitLLVMRuntimeCall(module, builder, module.getLoc(), "abort",
                          mlir::Type(), mlir::ValueRange{});
  abortCall->setAttr(ControlFlowContractAttrs::kNoReturn,
                     builder.getUnitAttr());
  builder.create<mlir::LLVM::UnreachableOp>(module.getLoc());
}

} // namespace lowering::runtime::eh

} // namespace py
