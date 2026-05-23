#include "Passes/Runtime/EH.h"

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
  if (auto fn = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(name))
    return fn;
  mlir::OpBuilder builder(module.getBody(), module.getBody()->begin());
  auto fnType = mlir::LLVM::LLVMFunctionType::get(resultType, argTypes, varArg);
  return builder.create<mlir::LLVM::LLVMFuncOp>(module.getLoc(), name, fnType);
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
  auto pyObject = mlir::LLVM::LLVMPointerType::get(ctx);
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
    unwind->clear();
    mlir::OpBuilder builder(ctx);
    builder.setInsertionPointToStart(unwind);
    auto lpType = mlir::LLVM::LLVMStructType::getLiteral(
        ctx, llvm::ArrayRef<mlir::Type>{pyObject, builder.getI32Type()});
    auto lp = builder.create<mlir::LLVM::LandingpadOp>(
        invoke.getLoc(), lpType, builder.getUnitAttr(), mlir::ValueRange{});
    mlir::Value raw = builder.create<mlir::LLVM::ExtractValueOp>(
        invoke.getLoc(), pyObject, lp.getRes(),
        builder.getDenseI64ArrayAttr({0}));
    ownership::Pointer::markNonObject(raw);
    emitLLVMRuntimeCall(module, builder, invoke.getLoc(),
                        RuntimeSymbols::kEHCapture, pyObject,
                        mlir::ValueRange{raw});
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
  auto lpType = mlir::LLVM::LLVMStructType::getLiteral(
      ctx, llvm::ArrayRef<mlir::Type>{ptrType, i32Type});
  auto lp = builder.create<mlir::LLVM::LandingpadOp>(
      module.getLoc(), lpType, builder.getUnitAttr(),
      mlir::ValueRange{catchAll});
  mlir::Value raw = builder.create<mlir::LLVM::ExtractValueOp>(
      module.getLoc(), ptrType, lp.getRes(), builder.getDenseI64ArrayAttr({0}));
  ownership::Pointer::markNonObject(raw);
  auto captured = emitLLVMRuntimeCall(module, builder, module.getLoc(),
                                      RuntimeSymbols::kEHCapture, ptrType,
                                      mlir::ValueRange{raw});
  auto reported = emitLLVMRuntimeCall(
      module, builder, module.getLoc(), RuntimeSymbols::kEHReportUnhandled,
      i32Type, mlir::ValueRange{captured.getResult()});
  builder.create<mlir::LLVM::ReturnOp>(module.getLoc(), reported.getResult());
}

} // namespace lowering::runtime::eh

} // namespace py
