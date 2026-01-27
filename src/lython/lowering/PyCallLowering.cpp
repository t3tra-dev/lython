#include "RuntimeSupport.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Location.h"
#include "llvm/ADT/STLExtras.h"

#include <cstdint>

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

using namespace mlir;

namespace py {
namespace {

static void getLocInfo(Location loc, MLIRContext *ctx, StringAttr &fileAttr,
                       std::int64_t &line, std::int64_t &col) {
  if (auto fileLoc = llvm::dyn_cast<FileLineColLoc>(loc)) {
    fileAttr = fileLoc.getFilename();
    line = static_cast<std::int64_t>(fileLoc.getLine());
    col = static_cast<std::int64_t>(fileLoc.getColumn());
    return;
  }
  if (auto nameLoc = llvm::dyn_cast<NameLoc>(loc)) {
    getLocInfo(nameLoc.getChildLoc(), ctx, fileAttr, line, col);
    return;
  }
  if (auto fused = llvm::dyn_cast<FusedLoc>(loc)) {
    for (auto subloc : fused.getLocations()) {
      if (auto subfile = llvm::dyn_cast<FileLineColLoc>(subloc)) {
        fileAttr = subfile.getFilename();
        line = static_cast<std::int64_t>(subfile.getLine());
        col = static_cast<std::int64_t>(subfile.getColumn());
        return;
      }
    }
  }
  fileAttr = StringAttr::get(ctx, "<unknown>");
  line = 0;
  col = 0;
}

static StringAttr getFuncNameAttr(func::FuncOp func, MLIRContext *ctx) {
  if (!func)
    return StringAttr::get(ctx, "<unknown>");
  StringRef name = func.getName();
  if (name == "main")
    return StringAttr::get(ctx, "<module>");
  return StringAttr::get(ctx, name);
}

static void emitTracebackPush(Location loc, func::FuncOp func, RuntimeAPI &runtime,
                              ConversionPatternRewriter &rewriter) {
  StringAttr fileAttr;
  std::int64_t line = 0;
  std::int64_t col = 0;
  getLocInfo(loc, rewriter.getContext(), fileAttr, line, col);
  StringAttr funcAttr = getFuncNameAttr(func, rewriter.getContext());
  Value filePtr = runtime.getStringLiteral(loc, fileAttr);
  Value funcPtr = runtime.getStringLiteral(loc, funcAttr);
  Value lineConst = rewriter.create<LLVM::ConstantOp>(
      loc, rewriter.getI32Type(),
      rewriter.getI32IntegerAttr(static_cast<int32_t>(line)));
  Value colConst = rewriter.create<LLVM::ConstantOp>(
      loc, rewriter.getI32Type(),
      rewriter.getI32IntegerAttr(static_cast<int32_t>(col)));
  runtime.call(loc, RuntimeSymbols::kTracebackPush, Type(),
               ValueRange{filePtr, funcPtr, lineConst, colConst});
}

static void ensureLandingpad(Block *unwind, Location loc,
                             ConversionPatternRewriter &rewriter) {
  if (!unwind)
    return;
  if (!unwind->empty() && llvm::isa<LLVM::LandingpadOp>(unwind->front()))
    return;
  rewriter.setInsertionPointToStart(unwind);
  auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
  auto i32Type = rewriter.getI32Type();
  auto lpType =
      LLVM::LLVMStructType::getLiteral(rewriter.getContext(),
                                       ArrayRef<Type>{ptrType, i32Type});
  rewriter.create<LLVM::LandingpadOp>(loc, lpType, rewriter.getUnitAttr(),
                                      ValueRange{});
}

struct CallVectorLowering : public OpConversionPattern<CallVectorOp> {
  CallVectorLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<CallVectorOp>(converter, ctx) {}

  LogicalResult lowerViaRuntime(CallVectorOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    if (op.getNumResults() > 1)
      return rewriter.notifyMatchFailure(
          op, "runtime lowering supports at most one result");

    auto *converter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *converter);
    Type pyObject = runtime.getPyObjectPtrType();

    SmallVector<Value> operands{adaptor.getCallable(), adaptor.getPosargs(),
                                adaptor.getKwnames(), adaptor.getKwvalues()};
    auto call = runtime.call(op.getLoc(), RuntimeSymbols::kCallVectorcall,
                             pyObject, operands);

    if (op.getNumResults() == 0) {
      rewriter.eraseOp(op);
      return success();
    }

    rewriter.replaceOp(op, call.getResults());
    return success();
  }

  LogicalResult
  matchAndRewrite(CallVectorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    Value callable = op.getCallable();
    if (auto identity = callable.getDefiningOp<CastIdentityOp>())
      callable = identity.getInput();
    Operation *producer = callable.getDefiningOp();
    auto constOp = dyn_cast_or_null<func::ConstantOp>(producer);
    if (!constOp)
      return lowerViaRuntime(op, adaptor, rewriter);

    SymbolRefAttr symbolRef = constOp.getValueAttr();
    StringRef symbolName = symbolRef.getLeafReference().empty()
                               ? symbolRef.getRootReference().getValue()
                               : symbolRef.getLeafReference().getValue();
    auto func = module.lookupSymbol<func::FuncOp>(symbolName);
    if (!func)
      return lowerViaRuntime(op, adaptor, rewriter);

    if (!isa<TupleEmptyOp>(op.getKwnames().getDefiningOp()) ||
        !isa<TupleEmptyOp>(op.getKwvalues().getDefiningOp()))
      return lowerViaRuntime(op, adaptor, rewriter);

    auto funcType = func.getFunctionType();
    SmallVector<Value> callOperands;
    auto posargsOp = op.getPosargs().getDefiningOp();
    if (auto tupleCreate = dyn_cast_or_null<TupleCreateOp>(posargsOp)) {
      if (tupleCreate.getElements().size() != funcType.getNumInputs())
        return lowerViaRuntime(op, adaptor, rewriter);
      for (auto [idx, element] : llvm::enumerate(tupleCreate.getElements())) {
        Type expected = funcType.getInput(idx);
        Value operand = element;
        if (operand.getType() != expected)
          operand =
              rewriter.create<CastIdentityOp>(op.getLoc(), expected, operand);
        callOperands.push_back(operand);
      }
    } else if (isa<TupleEmptyOp>(posargsOp)) {
      if (funcType.getNumInputs() != 0)
        return lowerViaRuntime(op, adaptor, rewriter);
    } else {
      return lowerViaRuntime(op, adaptor, rewriter);
    }

    // Create direct call, bypassing the tuple-based vectorcall
    auto call = rewriter.create<func::CallOp>(op.getLoc(), func, callOperands);

    if (call.getNumResults() == 0) {
      rewriter.eraseOp(op);
    } else {
      rewriter.replaceOp(op, call.getResults());
    }

    // Dead tuple ops and their DecRefOps are cleaned up by
    // cleanupDeadTuples() in RuntimeLoweringPass after call conversion

    if (constOp->use_empty())
      rewriter.eraseOp(constOp);
    return success();
  }
};

struct CallOpLowering : public OpConversionPattern<CallOp> {
  CallOpLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<CallOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    if (op.getNumResults() > 1)
      return rewriter.notifyMatchFailure(
          op, "runtime lowering supports at most one result");

    auto *converter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *converter);
    Type pyObject = runtime.getPyObjectPtrType();

    auto call =
        runtime.call(op.getLoc(), RuntimeSymbols::kCall, pyObject,
                     ValueRange{adaptor.getCallable(), adaptor.getPosargs(),
                                adaptor.getKwargs()});
    if (op.getNumResults() == 0) {
      rewriter.eraseOp(op);
      return success();
    }

    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};

struct InvokeLowering : public OpConversionPattern<InvokeOp> {
  InvokeLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<InvokeOp>(converter, ctx) {}

  LLVM::LLVMFuncOp getOrCreatePersonality(ModuleOp module,
                                          ConversionPatternRewriter &rewriter,
                                          Location loc) const {
    StringRef name = "__gxx_personality_v0";
    if (auto fn = module.lookupSymbol<LLVM::LLVMFuncOp>(name))
      return fn;
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    auto fnType = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(module.getContext()), {});
    auto fn = rewriter.create<LLVM::LLVMFuncOp>(loc, name, fnType);
    fn.setVisibility(SymbolTable::Visibility::Private);
    return fn;
  }

  LLVM::LLVMFuncOp
  getOrCreateInvokeCallee(ModuleOp module,
                          ConversionPatternRewriter &rewriter, Location loc,
                          Type resultType, ArrayRef<Type> argTypes) const {
    StringRef name = RuntimeSymbols::kCallVectorcallInvoke;
    if (auto fn = module.lookupSymbol<LLVM::LLVMFuncOp>(name))
      return fn;
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    auto fnType = LLVM::LLVMFunctionType::get(resultType, argTypes, false);
    auto fn = rewriter.create<LLVM::LLVMFuncOp>(loc, name, fnType);
    fn.setVisibility(SymbolTable::Visibility::Private);
    return fn;
  }

  LogicalResult lowerViaRuntime(InvokeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto *converter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *converter);
    Type pyObject = runtime.getPyObjectPtrType();
    SmallVector<Value> operands{adaptor.getCallable(), adaptor.getPosargs(),
                                adaptor.getKwnames(), adaptor.getKwvalues()};

    emitTracebackPush(op.getLoc(), op->getParentOfType<func::FuncOp>(), runtime,
                      rewriter);

    auto func = op->getParentOfType<func::FuncOp>();
    if (func) {
      auto personality = FlatSymbolRefAttr::get(op.getContext(),
                                                getOrCreatePersonality(
                                                    module, rewriter, op.getLoc())
                                                    .getName());
      func->setAttr("llvm.personality", personality);
    }

    SmallVector<Type> argTypes;
    argTypes.reserve(operands.size());
    for (Value operand : operands)
      argTypes.push_back(operand.getType());
    auto callee =
        getOrCreateInvokeCallee(module, rewriter, op.getLoc(), pyObject,
                                ArrayRef<Type>(argTypes));

    SmallVector<Value> normalOperands;
    if (op.getNormalDestOperands().size() > 1)
      return rewriter.notifyMatchFailure(
          op, "invoke lowering supports at most one result");
    Value normalSeed = nullptr;
    if (!op.getNormalDestOperands().empty()) {
      normalSeed = rewriter.create<LLVM::ZeroOp>(op.getLoc(), pyObject);
      normalOperands.push_back(normalSeed);
    }

    Block *unwindBlock = op.getUnwindDest();
    SmallVector<Value> unwindOperands(adaptor.getUnwindDestOperands().begin(),
                                      adaptor.getUnwindDestOperands().end());
    rewriter.setInsertionPoint(op);
    auto invoke = rewriter.create<LLVM::InvokeOp>(
        op.getLoc(), pyObject,
        FlatSymbolRefAttr::get(op.getContext(), callee.getName()), operands,
        op.getNormalDest(), normalOperands, unwindBlock, unwindOperands);

    if (!op.getNormalDestOperands().empty())
      op.getNormalDest()->getArgument(0).replaceAllUsesWith(invoke.getResult());

    ensureLandingpad(unwindBlock, op.getLoc(), rewriter);

    if (op.getNormalDest()) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(op.getNormalDest());
      runtime.call(op.getLoc(), RuntimeSymbols::kTracebackPop, Type(),
                   ValueRange{});
    }

    rewriter.eraseOp(op);
    return success();
  }

  LogicalResult
  matchAndRewrite(InvokeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto *converter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *converter);

    // Direct invoke path for symbol callee when args/kwargs are representable.
    Value callable = op.getCallable();
    if (auto identity = callable.getDefiningOp<CastIdentityOp>())
      callable = identity.getInput();
    Operation *producer = callable.getDefiningOp();
    auto constOp = dyn_cast_or_null<func::ConstantOp>(producer);
    if (!constOp)
      return lowerViaRuntime(op, adaptor, rewriter);

    SymbolRefAttr symbolRef = constOp.getValueAttr();
    StringRef symbolName = symbolRef.getLeafReference().empty()
                               ? symbolRef.getRootReference().getValue()
                               : symbolRef.getLeafReference().getValue();
    auto func = module.lookupSymbol<func::FuncOp>(symbolName);
    if (!func)
      return lowerViaRuntime(op, adaptor, rewriter);

    auto unwrapCast = [](Value v) -> Value {
      if (auto id = v.getDefiningOp<CastIdentityOp>())
        return id.getInput();
      return v;
    };
    auto posargsOp = unwrapCast(op.getPosargs()).getDefiningOp();
    auto kwnamesOp = unwrapCast(op.getKwnames()).getDefiningOp();
    auto kwvaluesOp = unwrapCast(op.getKwvalues()).getDefiningOp();
    if (!isa<TupleEmptyOp>(kwnamesOp) || !isa<TupleEmptyOp>(kwvaluesOp))
      return lowerViaRuntime(op, adaptor, rewriter);

    SmallVector<Value> callOperands;
    auto funcType = func.getFunctionType();
    if (auto tupleCreate = dyn_cast_or_null<TupleCreateOp>(posargsOp)) {
      if (tupleCreate.getElements().size() != funcType.getNumInputs())
        return lowerViaRuntime(op, adaptor, rewriter);
      for (auto [idx, element] : llvm::enumerate(tupleCreate.getElements())) {
        Type expected = funcType.getInput(idx);
        Value operand = element;
        if (operand.getType() != expected)
          operand =
              rewriter.create<CastIdentityOp>(op.getLoc(), expected, operand);
        callOperands.push_back(operand);
      }
    } else if (isa<TupleEmptyOp>(posargsOp)) {
      if (funcType.getNumInputs() != 0)
        return lowerViaRuntime(op, adaptor, rewriter);
    } else {
      return lowerViaRuntime(op, adaptor, rewriter);
    }

    SmallVector<Type> results;
    for (Type res : funcType.getResults())
      results.push_back(converter->convertType(res));

    Block *unwind = op.getUnwindDest();
    emitTracebackPush(op.getLoc(), op->getParentOfType<func::FuncOp>(), runtime,
                      rewriter);

    auto invoke = rewriter.create<LLVM::InvokeOp>(
        op.getLoc(), results,
        FlatSymbolRefAttr::get(op.getContext(), symbolName), callOperands,
        op.getNormalDest(), adaptor.getNormalDestOperands(), unwind,
        adaptor.getUnwindDestOperands());

    if (!op.getNormalDestOperands().empty() && !results.empty())
      op.getNormalDest()->getArgument(0).replaceAllUsesWith(
          invoke.getResult());

    ensureLandingpad(unwind, op.getLoc(), rewriter);

    if (constOp->use_empty())
      rewriter.eraseOp(constOp);

    if (op.getNormalDest()) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(op.getNormalDest());
      runtime.call(op.getLoc(), RuntimeSymbols::kTracebackPop, Type(),
                   ValueRange{});
    }
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populatePyCallLoweringPatterns(PyLLVMTypeConverter &typeConverter,
                                    RewritePatternSet &patterns) {
  auto *ctx = patterns.getContext();
  patterns.add<CallVectorLowering, CallOpLowering, InvokeLowering>(typeConverter,
                                                                   ctx);
}

} // namespace py
