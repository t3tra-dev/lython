// This file implements lowering patterns for py.func, py.return, and
// py.func_object operations. These patterns convert Python-style function
// definitions to standard MLIR func dialect operations.

#include "RuntimeSupport.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/StringMap.h"

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

using namespace mlir;

namespace py {
namespace {

LogicalResult translateFunctionSignature(
    FuncSignatureType sig, const PyLLVMTypeConverter &typeConverter,
    SmallVectorImpl<Type> &pyInputs, SmallVectorImpl<Type> &convertedInputs,
    SmallVectorImpl<Type> &convertedResults, Operation *emitOnError) {
  pyInputs.clear();
  convertedInputs.clear();
  convertedResults.clear();

  auto appendConverted = [&](Type ty,
                             SmallVectorImpl<Type> &storage) -> LogicalResult {
    Type converted = typeConverter.convertType(ty);
    if (!converted) {
      emitOnError->emitError("failed to convert type ") << ty;
      return failure();
    }
    storage.push_back(converted);
    return success();
  };

  auto positional = sig.getPositionalTypes();
  pyInputs.append(positional.begin(), positional.end());
  auto kwonly = sig.getKwOnlyTypes();
  pyInputs.append(kwonly.begin(), kwonly.end());
  if (sig.hasVararg())
    pyInputs.push_back(sig.getVarargType());
  if (sig.hasKwarg())
    pyInputs.push_back(sig.getKwargType());

  for (Type ty : pyInputs)
    if (failed(appendConverted(ty, convertedInputs)))
      return failure();

  for (Type result : sig.getResultTypes())
    if (failed(appendConverted(result, convertedResults)))
      return failure();

  return success();
}

// FuncOpLowering: py.func -> func.func

struct FuncOpLowering : public OpConversionPattern<FuncOp> {
  FuncOpLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<FuncOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto nameAttr = op->getAttrOfType<StringAttr>("sym_name");
    if (!nameAttr)
      return rewriter.notifyMatchFailure(op, "missing sym_name");

    // Skip builtin print function - it's handled specially
    if (nameAttr.getValue() == "__builtin_print") {
      rewriter.eraseOp(op);
      return success();
    }

    auto sigAttr = op->getAttrOfType<TypeAttr>("function_type");
    if (!sigAttr)
      return rewriter.notifyMatchFailure(op, "missing function_type attr");

    auto sig = dyn_cast<FuncSignatureType>(sigAttr.getValue());
    if (!sig)
      return rewriter.notifyMatchFailure(op, "expected FuncSignatureType");

    auto *tc = static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    SmallVector<Type, 8> pyInputTypes;
    SmallVector<Type, 8> llvmInputTypes;
    SmallVector<Type, 4> llvmResultTypes;
    if (failed(translateFunctionSignature(sig, *tc, pyInputTypes,
                                          llvmInputTypes, llvmResultTypes, op)))
      return failure();

    auto funcType =
        FunctionType::get(getContext(), llvmInputTypes, llvmResultTypes);
    auto newFunc = rewriter.create<func::FuncOp>(op.getLoc(),
                                                 nameAttr.getValue(), funcType);
    newFunc->setAttr("llvm.emit_c_interface", rewriter.getUnitAttr());

    if (op.getBody().empty())
      op.getBody().emplaceBlock();

    rewriter.inlineRegionBefore(op.getBody(), newFunc.getBody(),
                                newFunc.getBody().end());
    auto &entry = newFunc.getBody().front();
    TypeConverter::SignatureConversion conversion(pyInputTypes.size());
    for (auto [idx, ty] : llvm::enumerate(llvmInputTypes)) {
      SmallVector<Type, 1> packed{ty};
      conversion.addInputs(idx, packed);
    }
    auto *convertedEntry =
        rewriter.applySignatureConversion(&entry, conversion);
    if (!convertedEntry)
      return failure();

    rewriter.eraseOp(op);
    return success();
  }
};

// ReturnLowering: py.return -> func.return

struct ReturnLowering : public OpConversionPattern<ReturnOp> {
  ReturnLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<ReturnOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getOperands());
    return success();
  }
};

// FuncObjectLowering: py.func_object -> function reference

struct FuncObjectLowering : public OpConversionPattern<FuncObjectOp> {
  FuncObjectLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<FuncObjectOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(FuncObjectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    static llvm::StringMap<llvm::StringLiteral> builtinTable = {
        {"__builtin_print", RuntimeSymbols::kGetBuiltinPrint},
        {"print", RuntimeSymbols::kGetBuiltinPrint},
    };

    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    StringRef symbol = op.getTargetAttr().getValue();

    // Check if this is a builtin function
    if (auto it = builtinTable.find(symbol); it != builtinTable.end()) {
      auto *converter =
          static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
      RuntimeAPI runtime(module, rewriter, *converter);
      Type resultType = converter->convertType(op.getResult().getType());
      auto call =
          runtime.call(op.getLoc(), it->second, resultType, ValueRange{});
      rewriter.replaceOp(op, call.getResults());
      return success();
    }

    // Look up user-defined function
    auto func = module.lookupSymbol<func::FuncOp>(symbol);
    if (!func)
      return rewriter.notifyMatchFailure(op, "unknown function reference '" +
                                                 symbol + "'");

    auto constOp = rewriter.create<func::ConstantOp>(
        op.getLoc(), func.getFunctionType(),
        SymbolRefAttr::get(rewriter.getContext(), symbol));
    auto identity = rewriter.create<CastIdentityOp>(
        op.getLoc(), op.getResult().getType(), constOp.getResult());
    rewriter.replaceOp(op, identity.getResult());
    return success();
  }
};

} // namespace

void populatePyFuncLoweringPatterns(PyLLVMTypeConverter &typeConverter,
                                    RewritePatternSet &patterns) {
  auto *ctx = patterns.getContext();
  patterns.add<FuncOpLowering, ReturnLowering, FuncObjectLowering>(
      typeConverter, ctx);
}

} // namespace py
