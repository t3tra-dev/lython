#include "PyCall/Utils.h"

#include "llvm/ADT/STLExtras.h"

namespace py {
namespace {

struct InvokeLowering : public mlir::OpConversionPattern<InvokeOp> {
  InvokeLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<InvokeOp>(converter, ctx) {}

  mlir::LogicalResult
  lowerViaRuntime(InvokeOp op,
                  mlir::ConversionPatternRewriter &rewriter) const {
    return rewriter.notifyMatchFailure(
        op, "dynamic invoke runtime fallback is disabled");
  }

  mlir::LogicalResult
  matchAndRewrite(InvokeOp op, OneToNOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return mlir::failure();

    auto *converter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *converter);

    mlir::Value callable = stripBridgeCasts(op.getCallable());
    mlir::Operation *producer = callable.getDefiningOp();
    auto constOp = mlir::dyn_cast_or_null<mlir::func::ConstantOp>(producer);
    if (!constOp)
      return lowerViaRuntime(op, rewriter);

    mlir::SymbolRefAttr symbolRef = constOp.getValueAttr();
    llvm::StringRef symbolName = symbolRef.getLeafReference().empty()
                                     ? symbolRef.getRootReference().getValue()
                                     : symbolRef.getLeafReference().getValue();
    auto func = module.lookupSymbol<mlir::func::FuncOp>(symbolName);
    if (!func)
      return lowerViaRuntime(op, rewriter);

    auto unwrapCast = [](mlir::Value v) -> mlir::Value {
      if (auto cast = v.getDefiningOp<mlir::UnrealizedConversionCastOp>())
        if (cast->getNumOperands() == 1)
          return cast.getOperand(0);
      return v;
    };
    auto posargsOp = unwrapCast(op.getPosargs()).getDefiningOp();
    auto kwnamesOp = unwrapCast(op.getKwnames()).getDefiningOp();
    auto kwvaluesOp = unwrapCast(op.getKwvalues()).getDefiningOp();
    if (!mlir::isa<TupleEmptyOp>(kwnamesOp) ||
        !mlir::isa<TupleEmptyOp>(kwvaluesOp))
      return lowerViaRuntime(op, rewriter);

    llvm::SmallVector<mlir::Value> callOperands;
    auto funcType = func.getFunctionType();
    bool hiddenClassReturnOutArg =
        op.getNormalDestOperands().size() == 1 &&
        mlir::isa<ClassType>(op.getNormalDestOperands().front().getType()) &&
        funcType.getNumResults() == 0;
    unsigned directInputCount =
        funcType.getNumInputs() - (hiddenClassReturnOutArg ? 1u : 0u);
    if (auto tupleCreate = mlir::dyn_cast_or_null<TupleCreateOp>(posargsOp)) {
      bool allowVoidHelper = op.getNormalDestOperands().empty();
      func = resolvePreferredDirectHelper(func, tupleCreate.getElements(),
                                          module, allowVoidHelper);
      funcType = func.getFunctionType();
      hiddenClassReturnOutArg =
          op.getNormalDestOperands().size() == 1 &&
          mlir::isa<ClassType>(op.getNormalDestOperands().front().getType()) &&
          funcType.getNumResults() == 0;
      directInputCount =
          funcType.getNumInputs() - (hiddenClassReturnOutArg ? 1u : 0u);
      if (tupleCreate.getElements().size() != directInputCount)
        return lowerViaRuntime(op, rewriter);
      for (auto [idx, element] : llvm::enumerate(tupleCreate.getElements())) {
        mlir::Type expected = funcType.getInput(idx);
        mlir::Value operand = element;
        if (operand.getType() != expected)
          operand = rewriter
                        .create<mlir::UnrealizedConversionCastOp>(
                            op.getLoc(), mlir::TypeRange{expected},
                            mlir::ValueRange{operand})
                        .getResult(0);
        callOperands.push_back(operand);
      }
    } else if (mlir::isa<TupleEmptyOp>(posargsOp)) {
      bool allowVoidHelper = op.getNormalDestOperands().empty();
      func = resolvePreferredDirectHelper(func, mlir::ValueRange{}, module,
                                          allowVoidHelper);
      funcType = func.getFunctionType();
      hiddenClassReturnOutArg =
          op.getNormalDestOperands().size() == 1 &&
          mlir::isa<ClassType>(op.getNormalDestOperands().front().getType()) &&
          funcType.getNumResults() == 0;
      directInputCount =
          funcType.getNumInputs() - (hiddenClassReturnOutArg ? 1u : 0u);
      if (directInputCount != 0)
        return lowerViaRuntime(op, rewriter);
    } else {
      return lowerViaRuntime(op, rewriter);
    }

    mlir::Block *normalDest = op.getNormalDest();
    mlir::Block *finalNormalDest = normalDest;
    mlir::Value hiddenClassReturnSeed = nullptr;
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

    llvm::SmallVector<mlir::Type> results;
    for (mlir::Type res : funcType.getResults())
      if (mlir::failed(converter->convertType(res, results)))
        return mlir::failure();

    mlir::Block *unwind = op.getUnwindDest();
    emitTracebackPush(op.getLoc(), op->getParentOfType<mlir::func::FuncOp>(),
                      runtime, rewriter);

    llvm::SmallVector<mlir::Value> normalOperands;
    for (mlir::ValueRange values : adaptor.getNormalDestOperands())
      normalOperands.append(values.begin(), values.end());
    llvm::SmallVector<mlir::Value> unwindOperands;
    for (mlir::ValueRange values : adaptor.getUnwindDestOperands())
      unwindOperands.append(values.begin(), values.end());
    if (!op.getNormalDestOperands().empty() && finalNormalDest &&
        finalNormalDest->getNumArguments() == 1) {
      mlir::Type bridgeArgType = hiddenClassReturnOutArg
                                     ? hiddenClassReturnSeed.getType()
                                     : results.front();
      normalDest = createInvokeNormalBridge(finalNormalDest, bridgeArgType,
                                            op.getLoc(), rewriter);
      normalOperands.clear();
      if (hiddenClassReturnOutArg) {
        normalOperands.push_back(hiddenClassReturnSeed);
      } else {
        normalOperands.push_back(
            rewriter.create<mlir::LLVM::ZeroOp>(op.getLoc(), bridgeArgType));
      }
    }

    auto invoke = rewriter.create<mlir::LLVM::InvokeOp>(
        op.getLoc(), results,
        mlir::FlatSymbolRefAttr::get(op.getContext(), func.getName()),
        callOperands, normalDest, normalOperands, unwind, unwindOperands);

    if (!op.getNormalDestOperands().empty()) {
      if (normalDest != finalNormalDest) {
        mlir::Value forwarded = hiddenClassReturnOutArg
                                    ? normalDest->getArgument(0)
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
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(normalDest);
      runtime.call(op.getLoc(), RuntimeSymbols::kTracebackPop, mlir::Type(),
                   mlir::ValueRange{});
    }
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

} // namespace

namespace lowering::call::invoke::Patterns {
void populate(PyLLVMTypeConverter &typeConverter,
              mlir::RewritePatternSet &patterns) {
  auto *ctx = patterns.getContext();
  patterns.add<InvokeLowering>(typeConverter, ctx);
}
} // namespace lowering::call::invoke::Patterns

} // namespace py
