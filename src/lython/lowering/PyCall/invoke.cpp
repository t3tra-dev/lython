#include "PyCall/Utils.h"

#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace py {
namespace {

struct InvokeLowering : public OpConversionPattern<InvokeOp> {
  InvokeLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<InvokeOp>(converter, ctx) {}

  LogicalResult lowerViaRuntime(InvokeOp op,
                                ConversionPatternRewriter &rewriter) const {
    return rewriter.notifyMatchFailure(
        op, "dynamic invoke runtime fallback is disabled");
  }

  LogicalResult
  matchAndRewrite(InvokeOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto *converter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *converter);

    Value callable = op.getCallable();
    if (auto identity = callable.getDefiningOp<CastIdentityOp>())
      callable = identity.getInput();
    Operation *producer = callable.getDefiningOp();
    auto constOp = dyn_cast_or_null<func::ConstantOp>(producer);
    if (!constOp)
      return lowerViaRuntime(op, rewriter);

    SymbolRefAttr symbolRef = constOp.getValueAttr();
    StringRef symbolName = symbolRef.getLeafReference().empty()
                               ? symbolRef.getRootReference().getValue()
                               : symbolRef.getLeafReference().getValue();
    auto func = module.lookupSymbol<func::FuncOp>(symbolName);
    if (!func)
      return lowerViaRuntime(op, rewriter);

    auto unwrapCast = [](Value v) -> Value {
      if (auto id = v.getDefiningOp<CastIdentityOp>())
        return id.getInput();
      return v;
    };
    auto posargsOp = unwrapCast(op.getPosargs()).getDefiningOp();
    auto kwnamesOp = unwrapCast(op.getKwnames()).getDefiningOp();
    auto kwvaluesOp = unwrapCast(op.getKwvalues()).getDefiningOp();
    if (!isa<TupleEmptyOp>(kwnamesOp) || !isa<TupleEmptyOp>(kwvaluesOp))
      return lowerViaRuntime(op, rewriter);

    SmallVector<Value> callOperands;
    auto funcType = func.getFunctionType();
    bool hiddenClassReturnOutArg =
        op.getNormalDestOperands().size() == 1 &&
        isa<ClassType>(op.getNormalDestOperands().front().getType()) &&
        funcType.getNumResults() == 0;
    unsigned directInputCount =
        funcType.getNumInputs() - (hiddenClassReturnOutArg ? 1u : 0u);
    if (auto tupleCreate = dyn_cast_or_null<TupleCreateOp>(posargsOp)) {
      bool allowVoidHelper = op.getNormalDestOperands().empty();
      func = resolvePreferredDirectHelper(func, tupleCreate.getElements(),
                                          module, allowVoidHelper);
      funcType = func.getFunctionType();
      hiddenClassReturnOutArg =
          op.getNormalDestOperands().size() == 1 &&
          isa<ClassType>(op.getNormalDestOperands().front().getType()) &&
          funcType.getNumResults() == 0;
      directInputCount =
          funcType.getNumInputs() - (hiddenClassReturnOutArg ? 1u : 0u);
      if (tupleCreate.getElements().size() != directInputCount)
        return lowerViaRuntime(op, rewriter);
      for (auto [idx, element] : llvm::enumerate(tupleCreate.getElements())) {
        Type expected = funcType.getInput(idx);
        Value operand = element;
        if (operand.getType() != expected)
          operand =
              rewriter.create<CastIdentityOp>(op.getLoc(), expected, operand);
        callOperands.push_back(operand);
      }
    } else if (isa<TupleEmptyOp>(posargsOp)) {
      bool allowVoidHelper = op.getNormalDestOperands().empty();
      func = resolvePreferredDirectHelper(func, ValueRange{}, module,
                                          allowVoidHelper);
      funcType = func.getFunctionType();
      hiddenClassReturnOutArg =
          op.getNormalDestOperands().size() == 1 &&
          isa<ClassType>(op.getNormalDestOperands().front().getType()) &&
          funcType.getNumResults() == 0;
      directInputCount =
          funcType.getNumInputs() - (hiddenClassReturnOutArg ? 1u : 0u);
      if (directInputCount != 0)
        return lowerViaRuntime(op, rewriter);
    } else {
      return lowerViaRuntime(op, rewriter);
    }

    Block *normalDest = op.getNormalDest();
    Block *finalNormalDest = normalDest;
    Value hiddenClassReturnSeed = nullptr;
    if (hiddenClassReturnOutArg) {
      if (adaptor.getNormalDestOperands().size() != 1 ||
          adaptor.getNormalDestOperands().front().size() != 1)
        return rewriter.notifyMatchFailure(
            op, "class-return invoke expects a single normal destination seed");
      hiddenClassReturnSeed = adaptor.getNormalDestOperands().front().front();
      callOperands.push_back(hiddenClassReturnSeed);
      eraseInvokeNormalSeedDrops(op, op.getNormalDestOperands().front(),
                                 rewriter);
    }

    SmallVector<Type> results;
    for (Type res : funcType.getResults())
      if (failed(converter->convertType(res, results)))
        return failure();

    Block *unwind = op.getUnwindDest();
    emitTracebackPush(op.getLoc(), op->getParentOfType<func::FuncOp>(), runtime,
                      rewriter);

    SmallVector<Value> normalOperands;
    for (ValueRange values : adaptor.getNormalDestOperands())
      normalOperands.append(values.begin(), values.end());
    SmallVector<Value> unwindOperands;
    for (ValueRange values : adaptor.getUnwindDestOperands())
      unwindOperands.append(values.begin(), values.end());
    if (!op.getNormalDestOperands().empty() && finalNormalDest &&
        finalNormalDest->getNumArguments() == 1) {
      Type bridgeArgType = hiddenClassReturnOutArg
                               ? hiddenClassReturnSeed.getType()
                               : results.front();
      normalDest = createInvokeNormalBridge(finalNormalDest, bridgeArgType,
                                            op.getLoc(), rewriter);
      normalOperands.clear();
      if (hiddenClassReturnOutArg) {
        normalOperands.push_back(hiddenClassReturnSeed);
      } else {
        normalOperands.push_back(
            rewriter.create<LLVM::ZeroOp>(op.getLoc(), bridgeArgType));
      }
    }

    auto invoke = rewriter.create<LLVM::InvokeOp>(
        op.getLoc(), results,
        FlatSymbolRefAttr::get(op.getContext(), func.getName()), callOperands,
        normalDest, normalOperands, unwind, unwindOperands);

    if (!op.getNormalDestOperands().empty()) {
      if (normalDest != finalNormalDest) {
        Value forwarded = hiddenClassReturnOutArg ? normalDest->getArgument(0)
                                                  : invoke.getResult();
        finalizeInvokeNormalBridge(normalDest, finalNormalDest, forwarded,
                                   op.getLoc(), rewriter);
      } else if (hiddenClassReturnOutArg) {
        materializeInvokeNormalResult(op, hiddenClassReturnSeed, rewriter);
      } else if (!results.empty()) {
        materializeInvokeNormalResult(op, invoke.getResult(), rewriter);
      }
    }

    ensureLandingpad(unwind, op.getLoc(), rewriter);

    if (constOp->use_empty())
      rewriter.eraseOp(constOp);

    if (normalDest) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(normalDest);
      runtime.call(op.getLoc(), RuntimeSymbols::kTracebackPop, Type(),
                   ValueRange{});
    }
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populatePyInvokeLoweringPatterns(PyLLVMTypeConverter &typeConverter,
                                      RewritePatternSet &patterns) {
  auto *ctx = patterns.getContext();
  patterns.add<InvokeLowering>(typeConverter, ctx);
}

} // namespace py
