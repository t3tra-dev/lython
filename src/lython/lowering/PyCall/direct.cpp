#include "PyCall/Utils.h"

#include "Common/SlotUtils.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>

namespace py {
namespace {

struct CallVectorLowering : public mlir::OpConversionPattern<CallVectorOp> {
  CallVectorLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<CallVectorOp>(converter, ctx) {}

  mlir::LogicalResult
  lowerBuiltinPrint(CallVectorOp op,
                    mlir::ConversionPatternRewriter &rewriter) const {
    if (!mlir::isa<TupleEmptyOp>(op.getKwnames().getDefiningOp()) ||
        !mlir::isa<TupleEmptyOp>(op.getKwvalues().getDefiningOp()))
      return rewriter.notifyMatchFailure(
          op, "builtin print keyword arguments are not supported");

    auto posargsOp = op.getPosargs().getDefiningOp();
    llvm::SmallVector<mlir::Value> elements;
    if (auto tupleCreate = mlir::dyn_cast_or_null<TupleCreateOp>(posargsOp)) {
      elements.append(tupleCreate.getElements().begin(),
                      tupleCreate.getElements().end());
    } else if (!mlir::isa<TupleEmptyOp>(posargsOp)) {
      return rewriter.notifyMatchFailure(op,
                                         "builtin print requires static args");
    }

    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return mlir::failure();
    auto *converter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    auto ptrType = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i64Type = rewriter.getI64Type();
    auto i32Type = rewriter.getI32Type();
    mlir::Value argc = rewriter.create<mlir::LLVM::ConstantOp>(
        op.getLoc(), i64Type, rewriter.getI64IntegerAttr(elements.size()));
    mlir::Value storageCount = rewriter.create<mlir::LLVM::ConstantOp>(
        op.getLoc(), i64Type,
        rewriter.getI64IntegerAttr(std::max<std::size_t>(elements.size(), 1)));
    mlir::Value argsArray = rewriter.create<mlir::LLVM::AllocaOp>(
        op.getLoc(), ptrType, ptrType, storageCount, /*alignment=*/0);

    for (auto [index, element] : llvm::enumerate(elements)) {
      mlir::Value asPtr = Slot::bridgePointer(op.getLoc(), element, rewriter);
      mlir::Value idx = rewriter.create<mlir::LLVM::ConstantOp>(
          op.getLoc(), i64Type, rewriter.getI64IntegerAttr(index));
      mlir::Value slot = rewriter.create<mlir::LLVM::GEPOp>(
          op.getLoc(), ptrType, ptrType, argsArray, mlir::ValueRange{idx});
      rewriter.create<mlir::LLVM::StoreOp>(op.getLoc(), asPtr, slot);
    }

    mlir::Value nullPtr =
        rewriter.create<mlir::LLVM::ZeroOp>(op.getLoc(), ptrType);
    mlir::Value flush = rewriter.create<mlir::LLVM::ConstantOp>(
        op.getLoc(), i32Type, rewriter.getI32IntegerAttr(0));
    auto printFn = getOrInsertLLVMFunc(
        op.getLoc(), module, rewriter, RuntimeSymbols::kBuiltinPrintImpl,
        ptrType,
        {ptrType, ptrType, i64Type, ptrType, ptrType, ptrType, i32Type});
    auto printRef =
        mlir::SymbolRefAttr::get(rewriter.getContext(), printFn.getName());
    auto call = rewriter.create<mlir::LLVM::CallOp>(
        op.getLoc(), mlir::TypeRange{ptrType}, printRef,
        mlir::ValueRange{nullPtr, argsArray, argc, nullPtr, nullPtr, nullPtr,
                         flush});

    if (op.getNumResults() == 0) {
      rewriter.eraseOp(op);
      return mlir::success();
    }

    llvm::SmallVector<mlir::Value> logicalResults;
    materializeLogicalResults(op.getLoc(), op.getResultTypes(),
                              call.getResults(), logicalResults, *converter,
                              rewriter);
    rewriter.replaceOp(op, logicalResults);
    return mlir::success();
  }

  mlir::LogicalResult
  lowerViaRuntime(CallVectorOp op,
                  mlir::ConversionPatternRewriter &rewriter) const {
    return rewriter.notifyMatchFailure(
        op, "dynamic vectorcall runtime fallback is disabled");
  }

  mlir::LogicalResult
  appendFlattenedCallOperands(mlir::Location loc, mlir::ValueRange elements,
                              mlir::FunctionType funcType,
                              unsigned directInputCount,
                              llvm::SmallVectorImpl<mlir::Value> &operands,
                              mlir::ConversionPatternRewriter &rewriter) const {
    auto *converter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    unsigned expectedIndex = 0;
    for (mlir::Value element : elements) {
      llvm::SmallVector<mlir::Value> remapped;
      llvm::SmallVector<mlir::Type> convertedElementTypes;
      if (mlir::failed(converter->convertType(element.getType(),
                                              convertedElementTypes)) ||
          convertedElementTypes.empty())
        return mlir::failure();

      bool needsMaterializedExpansion = false;
      if (mlir::succeeded(rewriter.getRemappedValues(mlir::ValueRange{element},
                                                     remapped))) {
        needsMaterializedExpansion =
            remapped.size() != convertedElementTypes.size();
      } else {
        needsMaterializedExpansion = true;
      }
      if (!needsMaterializedExpansion) {
        for (auto [value, type] : llvm::zip(remapped, convertedElementTypes)) {
          if (value.getType() == element.getType() && value.getType() != type) {
            needsMaterializedExpansion = true;
            break;
          }
        }
      }
      if (needsMaterializedExpansion) {
        remapped = converter->materializeTargetConversion(
            rewriter, loc, mlir::TypeRange(convertedElementTypes),
            mlir::ValueRange{element}, element.getType());
        if (remapped.empty())
          return mlir::failure();
      }
      if (expectedIndex + remapped.size() > directInputCount)
        return mlir::failure();
      for (mlir::Value operand : remapped) {
        mlir::Type expected = funcType.getInput(expectedIndex++);
        if (operand.getType() != expected)
          operand =
              rewriter
                  .create<mlir::UnrealizedConversionCastOp>(
                      loc, mlir::TypeRange{expected}, mlir::ValueRange{operand})
                  .getResult(0);
        operands.push_back(operand);
      }
    }
    return mlir::success(expectedIndex == directInputCount);
  }

  mlir::LogicalResult
  matchAndRewrite(CallVectorOp op, OneToNOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return mlir::failure();
    auto *converter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());

    if (isBuiltinPrintCallable(op.getCallable()))
      return lowerBuiltinPrint(op, rewriter);

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

    if (canUseVoidHelper(op, func)) {
      auto posargsOp = op.getPosargs().getDefiningOp();
      llvm::SmallVector<mlir::Value> helperOperands;
      if (auto tupleCreate = mlir::dyn_cast_or_null<TupleCreateOp>(posargsOp)) {
        auto helperFunc = resolvePreferredDirectHelper(
            func, tupleCreate.getElements(), module, /*allowVoidHelper=*/true);
        if (!helperFunc)
          return rewriter.notifyMatchFailure(op, "missing void helper");
        auto helperType = helperFunc.getFunctionType();
        if (mlir::failed(appendFlattenedCallOperands(
                op.getLoc(), tupleCreate.getElements(), helperType,
                helperType.getNumInputs(), helperOperands, rewriter)))
          return lowerViaRuntime(op, rewriter);
        rewriter.create<mlir::func::CallOp>(op.getLoc(), helperFunc,
                                            helperOperands);
      } else if (mlir::isa<TupleEmptyOp>(posargsOp)) {
        auto helperFunc =
            resolvePreferredDirectHelper(func, mlir::ValueRange{}, module,
                                         /*allowVoidHelper=*/true);
        if (!helperFunc)
          return rewriter.notifyMatchFailure(op, "missing void helper");
        auto helperType = helperFunc.getFunctionType();
        if (helperType.getNumInputs() != 0)
          return lowerViaRuntime(op, rewriter);
        rewriter.create<mlir::func::CallOp>(op.getLoc(), helperFunc,
                                            helperOperands);
      } else {
        return lowerViaRuntime(op, rewriter);
      }
      eraseNoneResultUsers(op, rewriter);
      rewriter.eraseOp(op);
      if (constOp->use_empty())
        rewriter.eraseOp(constOp);
      return mlir::success();
    }

    if (!mlir::isa<TupleEmptyOp>(op.getKwnames().getDefiningOp()) ||
        !mlir::isa<TupleEmptyOp>(op.getKwvalues().getDefiningOp()))
      return lowerViaRuntime(op, rewriter);

    auto posargsOp = op.getPosargs().getDefiningOp();
    if (auto tupleCreate = mlir::dyn_cast_or_null<TupleCreateOp>(posargsOp)) {
      func =
          resolvePreferredDirectHelper(func, tupleCreate.getElements(), module,
                                       /*allowVoidHelper=*/false);
    }

    auto funcType = func.getFunctionType();
    llvm::SmallVector<mlir::Value> callOperands;
    bool hiddenClassReturnOutArg =
        op.getNumResults() == 1 &&
        mlir::isa<ClassType>(op.getResult(0).getType()) &&
        funcType.getNumResults() == 0;
    unsigned directInputCount =
        funcType.getNumInputs() - (hiddenClassReturnOutArg ? 1u : 0u);
    if (auto tupleCreate = mlir::dyn_cast_or_null<TupleCreateOp>(posargsOp)) {
      if (mlir::failed(appendFlattenedCallOperands(
              op.getLoc(), tupleCreate.getElements(), funcType,
              directInputCount, callOperands, rewriter)))
        return lowerViaRuntime(op, rewriter);
    } else if (mlir::isa<TupleEmptyOp>(posargsOp)) {
      if (directInputCount != 0)
        return lowerViaRuntime(op, rewriter);
    } else {
      return lowerViaRuntime(op, rewriter);
    }

    mlir::Value classReturnSlot;
    if (hiddenClassReturnOutArg) {
      auto classType = mlir::cast<ClassType>(op.getResult(0).getType());
      auto objectTypeOr = getStaticClassObjectType(op, classType, *converter);
      if (mlir::failed(objectTypeOr))
        return mlir::failure();
      classReturnSlot =
          createStaticClassSlot(op.getLoc(), *objectTypeOr, rewriter, op);
      if (!classReturnSlot)
        return mlir::failure();
      mlir::Value zero =
          rewriter.create<mlir::LLVM::ZeroOp>(op.getLoc(), *objectTypeOr);
      rewriter.create<mlir::LLVM::StoreOp>(op.getLoc(), zero, classReturnSlot);
      callOperands.push_back(classReturnSlot);
    }

    auto call =
        rewriter.create<mlir::func::CallOp>(op.getLoc(), func, callOperands);

    if (hiddenClassReturnOutArg) {
      rewriter.replaceOp(op, classReturnSlot);
    } else if (call.getNumResults() == 0) {
      rewriter.eraseOp(op);
    } else {
      llvm::SmallVector<mlir::Value> logicalResults;
      materializeLogicalResults(op.getLoc(), op.getResultTypes(),
                                call.getResults(), logicalResults, *converter,
                                rewriter);
      rewriter.replaceOp(op, logicalResults);
    }

    if (constOp->use_empty())
      rewriter.eraseOp(constOp);
    return mlir::success();
  }
};

struct CallOpLowering : public mlir::OpConversionPattern<CallOp> {
  CallOpLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<CallOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(CallOp op, OneToNOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    return rewriter.notifyMatchFailure(
        op, "dynamic call runtime fallback is disabled");
  }
};

} // namespace

namespace lowering::call::direct::Patterns {
void populate(PyLLVMTypeConverter &typeConverter,
              mlir::RewritePatternSet &patterns) {
  auto *ctx = patterns.getContext();
  patterns.add<CallVectorLowering, CallOpLowering>(typeConverter, ctx);
}
} // namespace lowering::call::direct::Patterns

} // namespace py
