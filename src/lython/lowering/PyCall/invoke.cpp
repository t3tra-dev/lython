#include "PyCall/Utils.h"

#include "Common/Object.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/ADT/STLExtras.h"

namespace py {
namespace {

static mlir::LogicalResult
appendLLVMInvokeOperand(mlir::Location loc, mlir::Value operand,
                        llvm::SmallVectorImpl<mlir::Value> &operands,
                        mlir::ConversionPatternRewriter &rewriter,
                        const PyLLVMTypeConverter &converter) {
  mlir::Type converted = converter.convertType(operand.getType());
  if (!converted)
    return mlir::failure();
  if (converted != operand.getType()) {
    operand =
        rewriter
            .create<mlir::UnrealizedConversionCastOp>(
                loc, mlir::TypeRange{converted}, mlir::ValueRange{operand})
            .getResult(0);
  }

  auto descriptor =
      mlir::dyn_cast<mlir::LLVM::LLVMStructType>(operand.getType());
  if (!object_abi::Type::isLoweredStorage(descriptor)) {
    operands.push_back(operand);
    return mlir::success();
  }

  llvm::ArrayRef<mlir::Type> body = descriptor.getBody();
  operands.push_back(rewriter.create<mlir::LLVM::ExtractValueOp>(
      loc, body[0], operand, rewriter.getDenseI64ArrayAttr({0})));
  operands.push_back(rewriter.create<mlir::LLVM::ExtractValueOp>(
      loc, body[1], operand, rewriter.getDenseI64ArrayAttr({1})));
  operands.push_back(rewriter.create<mlir::LLVM::ExtractValueOp>(
      loc, body[2], operand, rewriter.getDenseI64ArrayAttr({2})));
  operands.push_back(rewriter.create<mlir::LLVM::ExtractValueOp>(
      loc, rewriter.getI64Type(), operand,
      rewriter.getDenseI64ArrayAttr({3, 0})));
  operands.push_back(rewriter.create<mlir::LLVM::ExtractValueOp>(
      loc, rewriter.getI64Type(), operand,
      rewriter.getDenseI64ArrayAttr({4, 0})));
  return mlir::success();
}

static void collectEarlyArgDrops(TupleCreateOp tupleCreate,
                                 mlir::Operation *anchor,
                                 llvm::SmallVectorImpl<DecRefOp> &drops) {
  if (!tupleCreate || !anchor)
    return;
  for (mlir::Value element : tupleCreate.getElements()) {
    for (mlir::Operation *user : element.getUsers()) {
      auto drop = mlir::dyn_cast<DecRefOp>(user);
      if (!drop || drop->getBlock() != anchor->getBlock() ||
          !drop->isBeforeInBlock(anchor))
        continue;
      if (!llvm::is_contained(drops, drop))
        drops.push_back(drop);
    }
  }
}

static void collectTupleDrops(TupleCreateOp tupleCreate,
                              llvm::SmallVectorImpl<DecRefOp> &drops) {
  if (!tupleCreate)
    return;
  for (mlir::Operation *user : tupleCreate.getResult().getUsers()) {
    if (auto drop = mlir::dyn_cast<DecRefOp>(user))
      drops.push_back(drop);
  }
}

static void cloneDropAtBlockStart(DecRefOp drop, mlir::Block *block,
                                  mlir::ConversionPatternRewriter &rewriter) {
  if (!drop || !block)
    return;
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  if (!block->empty() && mlir::isa<mlir::LLVM::LandingpadOp>(block->front()))
    rewriter.setInsertionPointAfter(&block->front());
  else
    rewriter.setInsertionPointToStart(block);
  auto clone = rewriter.create<DecRefOp>(drop.getLoc(), drop.getObject());
  for (mlir::NamedAttribute attr : drop->getAttrs())
    clone->setAttr(attr.getName(), attr.getValue());
}

static void releaseArgsAfterInvoke(llvm::ArrayRef<DecRefOp> drops,
                                   mlir::Block *normalDest,
                                   mlir::Block *unwindDest,
                                   mlir::ConversionPatternRewriter &rewriter) {
  for (DecRefOp drop : drops) {
    cloneDropAtBlockStart(drop, normalDest, rewriter);
    cloneDropAtBlockStart(drop, unwindDest, rewriter);
    rewriter.eraseOp(drop);
  }
}

static void cleanupCallPack(TupleCreateOp tupleCreate,
                            llvm::ArrayRef<DecRefOp> tupleDrops,
                            mlir::ConversionPatternRewriter &rewriter) {
  for (DecRefOp drop : tupleDrops)
    rewriter.eraseOp(drop);
  if (tupleCreate && tupleCreate->use_empty())
    rewriter.eraseOp(tupleCreate);
}

struct InvokeLowering : public mlir::OpConversionPattern<InvokeOp> {
  InvokeLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<InvokeOp>(converter, ctx) {}

  mlir::LogicalResult
  lowerViaRuntime(InvokeOp op,
                  mlir::ConversionPatternRewriter &rewriter) const {
    return rewriter.notifyMatchFailure(
        op, "dynamic invoke requires static call resolution");
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
    TupleCreateOp staticPosargs;
    llvm::SmallVector<DecRefOp> argDrops;
    llvm::SmallVector<DecRefOp> tupleDrops;
    auto funcType = func.getFunctionType();
    unsigned directInputCount = funcType.getNumInputs();
    if (auto tupleCreate = mlir::dyn_cast_or_null<TupleCreateOp>(posargsOp)) {
      staticPosargs = tupleCreate;
      collectEarlyArgDrops(tupleCreate, op.getOperation(), argDrops);
      collectTupleDrops(tupleCreate, tupleDrops);
      bool allowVoidHelper = op.getNormalDestOperands().empty();
      func = resolvePreferredDirectHelper(func, tupleCreate.getElements(),
                                          module, allowVoidHelper);
      funcType = func.getFunctionType();
      directInputCount = funcType.getNumInputs();
      if (mlir::failed(::py::appendFlattenedCallOperands(
              op.getLoc(), tupleCreate.getElements(), funcType,
              directInputCount, callOperands, rewriter, *converter)))
        return lowerViaRuntime(op, rewriter);
    } else if (mlir::isa<TupleEmptyOp>(posargsOp)) {
      bool allowVoidHelper = op.getNormalDestOperands().empty();
      func = resolvePreferredDirectHelper(func, mlir::ValueRange{}, module,
                                          allowVoidHelper);
      funcType = func.getFunctionType();
      directInputCount = funcType.getNumInputs();
      if (directInputCount != 0)
        return lowerViaRuntime(op, rewriter);
    } else {
      return lowerViaRuntime(op, rewriter);
    }

    mlir::Block *normalDest = op.getNormalDest();
    mlir::Block *finalNormalDest = normalDest;
    llvm::SmallVector<mlir::Type> results;
    for (mlir::Type res : funcType.getResults())
      if (mlir::failed(converter->convertType(res, results)))
        return mlir::failure();
    if (!op.getNormalDestOperands().empty() && results.empty())
      return lowerViaRuntime(op, rewriter);

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
      mlir::Type bridgeArgType = results.front();
      normalDest = createInvokeNormalBridge(finalNormalDest, bridgeArgType,
                                            op.getLoc(), rewriter);
      normalOperands.clear();
      normalOperands.push_back(
          rewriter.create<mlir::LLVM::ZeroOp>(op.getLoc(), bridgeArgType));
    }

    llvm::SmallVector<mlir::Value> invokeOperands;
    for (mlir::Value operand : callOperands)
      if (mlir::failed(appendLLVMInvokeOperand(
              op.getLoc(), operand, invokeOperands, rewriter, *converter)))
        return mlir::failure();

    auto invoke = rewriter.create<mlir::LLVM::InvokeOp>(
        op.getLoc(), results,
        mlir::FlatSymbolRefAttr::get(op.getContext(), func.getName()),
        invokeOperands, normalDest, normalOperands, unwind, unwindOperands);

    if (!op.getNormalDestOperands().empty()) {
      if (normalDest != finalNormalDest) {
        mlir::Value forwarded = invoke.getResult();
        finalizeInvokeNormalBridge(normalDest, finalNormalDest, forwarded,
                                   op.getLoc(), rewriter);
      } else if (!results.empty()) {
        materializeInvokeNormalResult(op, invoke.getResult(), rewriter);
      }
    }

    ensureLandingpad(unwind, op.getLoc(), rewriter);

    if (constOp->use_empty())
      rewriter.eraseOp(constOp);

    releaseArgsAfterInvoke(argDrops,
                           finalNormalDest ? finalNormalDest : normalDest,
                           unwind, rewriter);
    cleanupCallPack(staticPosargs, tupleDrops, rewriter);

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
