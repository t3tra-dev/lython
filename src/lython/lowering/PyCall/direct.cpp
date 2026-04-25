#include "PyCall/Utils.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>

using namespace mlir;

namespace py {
namespace {

struct CallVectorLowering : public OpConversionPattern<CallVectorOp> {
  CallVectorLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<CallVectorOp>(converter, ctx) {}

  LogicalResult lowerBuiltinPrint(CallVectorOp op,
                                  ConversionPatternRewriter &rewriter) const {
    if (!isa<TupleEmptyOp>(op.getKwnames().getDefiningOp()) ||
        !isa<TupleEmptyOp>(op.getKwvalues().getDefiningOp()))
      return rewriter.notifyMatchFailure(
          op, "builtin print keyword arguments are not supported");

    auto posargsOp = op.getPosargs().getDefiningOp();
    SmallVector<Value> elements;
    if (auto tupleCreate = dyn_cast_or_null<TupleCreateOp>(posargsOp)) {
      elements.append(tupleCreate.getElements().begin(),
                      tupleCreate.getElements().end());
    } else if (!isa<TupleEmptyOp>(posargsOp)) {
      return rewriter.notifyMatchFailure(op,
                                         "builtin print requires static args");
    }

    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i64Type = rewriter.getI64Type();
    auto i32Type = rewriter.getI32Type();
    Value argc = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i64Type, rewriter.getI64IntegerAttr(elements.size()));
    Value storageCount = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i64Type,
        rewriter.getI64IntegerAttr(std::max<std::size_t>(elements.size(), 1)));
    Value argsArray = rewriter.create<LLVM::AllocaOp>(
        op.getLoc(), ptrType, ptrType, storageCount, /*alignment=*/0);

    for (auto [index, element] : llvm::enumerate(elements)) {
      Value asPtr = element;
      if (asPtr.getType() != ptrType)
        asPtr = rewriter.create<CastIdentityOp>(op.getLoc(), ptrType, asPtr);
      Value idx = rewriter.create<LLVM::ConstantOp>(
          op.getLoc(), i64Type, rewriter.getI64IntegerAttr(index));
      Value slot = rewriter.create<LLVM::GEPOp>(op.getLoc(), ptrType, ptrType,
                                                argsArray, ValueRange{idx});
      rewriter.create<LLVM::StoreOp>(op.getLoc(), asPtr, slot);
    }

    Value nullPtr = rewriter.create<LLVM::ZeroOp>(op.getLoc(), ptrType);
    Value flush = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i32Type, rewriter.getI32IntegerAttr(0));
    auto printFn = getOrInsertLLVMFunc(
        op.getLoc(), module, rewriter, RuntimeSymbols::kBuiltinPrintImpl,
        ptrType,
        {ptrType, ptrType, i64Type, ptrType, ptrType, ptrType, i32Type});
    auto printRef =
        SymbolRefAttr::get(rewriter.getContext(), printFn.getName());
    auto call = rewriter.create<LLVM::CallOp>(
        op.getLoc(), TypeRange{ptrType}, printRef,
        ValueRange{nullPtr, argsArray, argc, nullPtr, nullPtr, nullPtr, flush});

    if (op.getNumResults() == 0) {
      rewriter.eraseOp(op);
      return success();
    }

    SmallVector<Value> logicalResults;
    materializeLogicalResults(op.getLoc(), op.getResultTypes(),
                              call.getResults(), logicalResults, rewriter);
    rewriter.replaceOp(op, logicalResults);
    return success();
  }

  LogicalResult lowerViaRuntime(CallVectorOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    (void)adaptor;
    return rewriter.notifyMatchFailure(
        op, "dynamic vectorcall runtime fallback is disabled");
  }

  LogicalResult
  matchAndRewrite(CallVectorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();
    auto *converter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());

    if (isBuiltinPrintCallable(op.getCallable()))
      return lowerBuiltinPrint(op, rewriter);

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

    if (canUseVoidHelper(op, func)) {
      auto posargsOp = op.getPosargs().getDefiningOp();
      SmallVector<Value> helperOperands;
      if (auto tupleCreate = dyn_cast_or_null<TupleCreateOp>(posargsOp)) {
        auto helperFunc = resolvePreferredDirectHelper(
            func, tupleCreate.getElements(), module, /*allowVoidHelper=*/true);
        if (!helperFunc)
          return rewriter.notifyMatchFailure(op, "missing void helper");
        auto helperType = helperFunc.getFunctionType();
        if (tupleCreate.getElements().size() != helperType.getNumInputs())
          return lowerViaRuntime(op, adaptor, rewriter);
        for (auto [idx, element] : llvm::enumerate(tupleCreate.getElements())) {
          Type expected = helperType.getInput(idx);
          Value operand = element;
          if (operand.getType() != expected)
            operand =
                rewriter.create<CastIdentityOp>(op.getLoc(), expected, operand);
          helperOperands.push_back(operand);
        }
        rewriter.create<func::CallOp>(op.getLoc(), helperFunc, helperOperands);
      } else if (isa<TupleEmptyOp>(posargsOp)) {
        auto helperFunc =
            resolvePreferredDirectHelper(func, ValueRange{}, module,
                                         /*allowVoidHelper=*/true);
        if (!helperFunc)
          return rewriter.notifyMatchFailure(op, "missing void helper");
        auto helperType = helperFunc.getFunctionType();
        if (helperType.getNumInputs() != 0)
          return lowerViaRuntime(op, adaptor, rewriter);
        rewriter.create<func::CallOp>(op.getLoc(), helperFunc, helperOperands);
      } else {
        return lowerViaRuntime(op, adaptor, rewriter);
      }
      eraseNoneResultUsers(op, rewriter);
      rewriter.eraseOp(op);
      if (constOp->use_empty())
        rewriter.eraseOp(constOp);
      return success();
    }

    if (!isa<TupleEmptyOp>(op.getKwnames().getDefiningOp()) ||
        !isa<TupleEmptyOp>(op.getKwvalues().getDefiningOp()))
      return lowerViaRuntime(op, adaptor, rewriter);

    auto posargsOp = op.getPosargs().getDefiningOp();
    if (auto tupleCreate = dyn_cast_or_null<TupleCreateOp>(posargsOp)) {
      func =
          resolvePreferredDirectHelper(func, tupleCreate.getElements(), module,
                                       /*allowVoidHelper=*/false);
    }

    auto funcType = func.getFunctionType();
    SmallVector<Value> callOperands;
    bool hiddenClassReturnOutArg = op.getNumResults() == 1 &&
                                   isa<ClassType>(op.getResult(0).getType()) &&
                                   funcType.getNumResults() == 0;
    unsigned directInputCount =
        funcType.getNumInputs() - (hiddenClassReturnOutArg ? 1u : 0u);
    if (auto tupleCreate = dyn_cast_or_null<TupleCreateOp>(posargsOp)) {
      if (tupleCreate.getElements().size() != directInputCount)
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
      if (directInputCount != 0)
        return lowerViaRuntime(op, adaptor, rewriter);
    } else {
      return lowerViaRuntime(op, adaptor, rewriter);
    }

    Value classReturnSlot;
    if (hiddenClassReturnOutArg) {
      auto classType = cast<ClassType>(op.getResult(0).getType());
      auto objectTypeOr = getStaticClassObjectType(op, classType, *converter);
      if (failed(objectTypeOr))
        return failure();
      classReturnSlot =
          createStaticClassSlot(op.getLoc(), *objectTypeOr, rewriter, op);
      if (!classReturnSlot)
        return failure();
      Value zero = rewriter.create<LLVM::ZeroOp>(op.getLoc(), *objectTypeOr);
      rewriter.create<LLVM::StoreOp>(op.getLoc(), zero, classReturnSlot);
      callOperands.push_back(classReturnSlot);
    }

    auto call = rewriter.create<func::CallOp>(op.getLoc(), func, callOperands);

    if (hiddenClassReturnOutArg) {
      rewriter.replaceOp(op, classReturnSlot);
    } else if (call.getNumResults() == 0) {
      rewriter.eraseOp(op);
    } else {
      SmallVector<Value> logicalResults;
      materializeLogicalResults(op.getLoc(), op.getResultTypes(),
                                call.getResults(), logicalResults, rewriter);
      rewriter.replaceOp(op, logicalResults);
    }

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
    (void)adaptor;
    return rewriter.notifyMatchFailure(
        op, "dynamic call runtime fallback is disabled");
  }
};

} // namespace

void populatePyDirectCallLoweringPatterns(PyLLVMTypeConverter &typeConverter,
                                          RewritePatternSet &patterns) {
  auto *ctx = patterns.getContext();
  patterns.add<CallVectorLowering, CallOpLowering>(typeConverter, ctx);
}

} // namespace py
