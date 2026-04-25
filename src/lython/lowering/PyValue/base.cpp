#include "Common/RuntimeSupport.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Location.h"

#include <cerrno>
#include <cstdint>
#include <cstdlib>

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

struct ExceptionNullLowering : public OpConversionPattern<ExceptionNullOp> {
  ExceptionNullLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<ExceptionNullOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(ExceptionNullOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    Type resultType = typeConverter->convertType(op.getResult().getType());
    rewriter.replaceOpWithNewOp<LLVM::ZeroOp>(op, resultType);
    return success();
  }
};

struct TracebackNullLowering : public OpConversionPattern<TracebackNullOp> {
  TracebackNullLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<TracebackNullOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(TracebackNullOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    Type resultType = typeConverter->convertType(op.getResult().getType());
    rewriter.replaceOpWithNewOp<LLVM::ZeroOp>(op, resultType);
    return success();
  }
};

struct LocationCurrentLowering : public OpConversionPattern<LocationCurrentOp> {
  LocationCurrentLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<LocationCurrentOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(LocationCurrentOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    Type resultType = typeConverter->convertType(op.getResult().getType());
    rewriter.replaceOpWithNewOp<LLVM::ZeroOp>(op, resultType);
    return success();
  }
};

struct ExceptionNewLowering : public OpConversionPattern<ExceptionNewOp> {
  ExceptionNewLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<ExceptionNewOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(ExceptionNewOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *typeConverter);
    Type resultType = typeConverter->convertType(op.getResult().getType());
    Value nullPtr = rewriter.create<LLVM::ZeroOp>(
        op.getLoc(), LLVM::LLVMPointerType::get(rewriter.getContext()));
    auto call = runtime.call(
        op.getLoc(), RuntimeSymbols::kExceptionNew, resultType,
        ValueRange{adaptor.getType(), adaptor.getMessage(), nullPtr,
                   adaptor.getCause(), adaptor.getContext(),
                   adaptor.getTraceback(), adaptor.getLocation(), nullPtr});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};

struct TryLowering : public OpConversionPattern<TryOp> {
  TryLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<TryOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(TryOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getTryRegion().empty())
      return rewriter.notifyMatchFailure(op, "empty try region");

    bool hasExcept = !op.getExceptRegion().empty();
    bool hasFinally = !op.getFinallyRegion().empty();

    if (!hasExcept)
      return rewriter.notifyMatchFailure(op,
                                         "try lowering requires except region");

    if (hasFinally && op.getNumResults() > 0)
      return rewriter.notifyMatchFailure(
          op, "finally with results is not supported yet");

    Region *parent = op->getParentRegion();
    Block *parentBlock = op->getBlock();
    auto insertIt = std::next(
        parent->begin(),
        std::distance(parent->begin(), Region::iterator(parentBlock)));
    Block *mergeBlock = rewriter.createBlock(parent, insertIt);
    for (Type resultType : op.getResultTypes())
      mergeBlock->addArgument(resultType, op.getLoc());

    Block *tryEntry = &op.getTryRegion().front();
    Block *exceptEntry = &op.getExceptRegion().front();
    Block *finallyEntry = hasFinally ? &op.getFinallyRegion().front() : nullptr;

    SmallVector<Block *> tryBlocks;
    SmallVector<Block *> exceptBlocks;
    SmallVector<Block *> finallyBlocks;
    for (Block &block : op.getTryRegion())
      tryBlocks.push_back(&block);
    for (Block &block : op.getExceptRegion())
      exceptBlocks.push_back(&block);
    if (hasFinally)
      for (Block &block : op.getFinallyRegion())
        finallyBlocks.push_back(&block);

    // Move operations after py.try into merge block.
    for (auto it = std::next(op->getIterator()); it != parentBlock->end();) {
      Operation *move = &*it++;
      move->moveBefore(mergeBlock, mergeBlock->end());
    }

    // Inline regions into parent before merge block.
    rewriter.inlineRegionBefore(op.getTryRegion(), *parent,
                                mergeBlock->getIterator());
    rewriter.inlineRegionBefore(op.getExceptRegion(), *parent,
                                mergeBlock->getIterator());
    if (hasFinally)
      rewriter.inlineRegionBefore(op.getFinallyRegion(), *parent,
                                  mergeBlock->getIterator());

    // Redirect invokes inside try region to except entry.
    for (Block *block : tryBlocks) {
      for (auto invoke : block->getOps<InvokeOp>()) {
        OpBuilder builder(invoke);
        auto excNull = builder.create<ExceptionNullOp>(
            invoke.getLoc(), ExceptionType::get(op.getContext()));
        invoke.getUnwindDestOperandsMutable().assign(excNull.getResult());
        invoke->setSuccessor(exceptEntry, 1);
      }
    }

    // Replace yields with branches.
    for (Block *block : tryBlocks) {
      if (auto yield = dyn_cast<TryYieldOp>(block->getTerminator())) {
        if (hasFinally) {
          rewriter.setInsertionPoint(yield);
          rewriter.replaceOpWithNewOp<cf::BranchOp>(yield, finallyEntry,
                                                    ValueRange{});
        } else {
          rewriter.setInsertionPoint(yield);
          rewriter.replaceOpWithNewOp<cf::BranchOp>(yield, mergeBlock,
                                                    yield.getOperands());
        }
      }
    }
    for (Block *block : exceptBlocks) {
      if (auto yield = dyn_cast<ExceptYieldOp>(block->getTerminator())) {
        if (hasFinally) {
          rewriter.setInsertionPoint(yield);
          rewriter.replaceOpWithNewOp<cf::BranchOp>(yield, finallyEntry,
                                                    ValueRange{});
        } else {
          rewriter.setInsertionPoint(yield);
          rewriter.replaceOpWithNewOp<cf::BranchOp>(yield, mergeBlock,
                                                    yield.getOperands());
        }
      }
    }
    for (Block *block : finallyBlocks) {
      if (auto yield = dyn_cast<FinallyYieldOp>(block->getTerminator())) {
        rewriter.setInsertionPoint(yield);
        rewriter.replaceOpWithNewOp<cf::BranchOp>(yield, mergeBlock,
                                                  ValueRange{});
      }
    }

    // Connect parent block to try entry.
    rewriter.setInsertionPointToEnd(parentBlock);
    rewriter.create<cf::BranchOp>(op.getLoc(), tryEntry);

    for (auto [res, arg] :
         llvm::zip(op.getResults(), mergeBlock->getArguments()))
      res.replaceAllUsesWith(arg);

    rewriter.eraseOp(op);
    return success();
  }
};

struct RaiseLowering : public OpConversionPattern<RaiseOp> {
  RaiseLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<RaiseOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(RaiseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *typeConverter);
    StringAttr fileAttr;
    std::int64_t line = 0;
    std::int64_t col = 0;
    getLocInfo(op.getLoc(), op.getContext(), fileAttr, line, col);
    StringAttr funcAttr =
        getFuncNameAttr(op->getParentOfType<func::FuncOp>(), op.getContext());
    Value filePtr = runtime.getStringLiteral(op.getLoc(), fileAttr);
    Value funcPtr = runtime.getStringLiteral(op.getLoc(), funcAttr);
    Value lineConst = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), rewriter.getI32Type(),
        rewriter.getI32IntegerAttr(static_cast<int32_t>(line)));
    Value colConst = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), rewriter.getI32Type(),
        rewriter.getI32IntegerAttr(static_cast<int32_t>(col)));
    runtime.call(op.getLoc(), RuntimeSymbols::kTracebackPush, Type(),
                 ValueRange{filePtr, funcPtr, lineConst, colConst});
    runtime.call(op.getLoc(), RuntimeSymbols::kEHThrow, Type(),
                 ValueRange{adaptor.getException()});
    rewriter.create<LLVM::UnreachableOp>(op.getLoc());
    rewriter.eraseOp(op);
    return success();
  }
};

struct RaiseCurrentLowering : public OpConversionPattern<RaiseCurrentOp> {
  RaiseCurrentLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<RaiseCurrentOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(RaiseCurrentOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *typeConverter);
    Type pyObject = runtime.getPyObjectPtrType();
    auto current =
        runtime.call(op.getLoc(), RuntimeSymbols::kExceptionGetCurrent,
                     pyObject, ValueRange{});
    runtime.call(op.getLoc(), RuntimeSymbols::kEHThrow, Type(),
                 current.getResults());
    rewriter.create<LLVM::UnreachableOp>(op.getLoc());
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populatePyValueLoweringPatterns(PyLLVMTypeConverter &typeConverter,
                                     RewritePatternSet &patterns) {
  auto *ctx = patterns.getContext();
  patterns.add<ExceptionNullLowering, TracebackNullLowering,
               LocationCurrentLowering, ExceptionNewLowering, TryLowering,
               RaiseLowering, RaiseCurrentLowering>(typeConverter, ctx);
}

} // namespace py
