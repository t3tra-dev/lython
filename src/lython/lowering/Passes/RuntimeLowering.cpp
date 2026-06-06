// This file implements the main RuntimeLoweringPass which orchestrates the
// complete lowering pipeline from Py dialect to LLVM dialect. It coordinates
// the various conversion phases:
//   1. Function conversion (py.func -> func.func)
//   2. Function object conversion (py.func_object -> references)
//   3. Call conversion (py.call_vector -> runtime calls or direct calls)
//   4. mlir::Value conversion (py.* ops -> LLVM ops via runtime calls)
//
// Individual lowering patterns are implemented in separate files:
//   - PyFunc/Lowering.cpp: Function and calling convention patterns
//   - PyCall/*.cpp: Call operation patterns
//   - PyValue*.cpp: mlir::Value, class, scalar, and typed container patterns
//   - PyRefCount/Lowering.cpp: Reference counting patterns
//
// Optimizations are implemented under Optimizer/.

#include "Common/LoweringUtils.h"
#include "Common/RuntimeLibrary.h"
#include "Common/RuntimeSupport.h"
#include "Passes/OwnershipAnalysis.h"
#include "Passes/Runtime/Async.h"
#include "Passes/Runtime/Cleanup.h"
#include "Passes/Runtime/Conversion.h"
#include "Passes/Runtime/EH.h"
#include "Passes/Runtime/Helpers.h"
#include "Passes/Runtime/MemRefToLLVM.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>

#include "PyDialect.h.inc"

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

namespace py {

namespace optimizer::scalar {
void foldIntConstants(mlir::ModuleOp module);
} // namespace optimizer::scalar

namespace {

// RuntimeLoweringPass: Main pipeline orchestration

void copyDiscardableAttrs(mlir::Operation *from, mlir::Operation *to) {
  if (!from || !to)
    return;
  for (const mlir::NamedAttribute &attr : from->getDiscardableAttrs())
    to->setDiscardableAttr(attr.getName(), attr.getValue());
}

mlir::Value materializeIndex(mlir::Location loc, mlir::OpFoldResult value,
                             mlir::PatternRewriter &rewriter) {
  if (auto dynamic = mlir::dyn_cast<mlir::Value>(value))
    return dynamic;
  auto attr = mlir::cast<mlir::IntegerAttr>(mlir::cast<mlir::Attribute>(value));
  return rewriter.create<mlir::arith::ConstantIndexOp>(loc, attr.getInt());
}

mlir::Value mulIndex(mlir::Location loc, mlir::Value lhs,
                     mlir::OpFoldResult rhs, mlir::PatternRewriter &rewriter) {
  if (auto attr = mlir::dyn_cast<mlir::Attribute>(rhs)) {
    int64_t value = mlir::cast<mlir::IntegerAttr>(attr).getInt();
    if (value == 1)
      return lhs;
    if (value == 0)
      return rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
  }
  return rewriter.create<mlir::arith::MulIOp>(
      loc, lhs, materializeIndex(loc, rhs, rewriter));
}

struct AttrPreservingRank1SubviewExpansion
    : public mlir::OpRewritePattern<mlir::memref::SubViewOp> {
  using mlir::OpRewritePattern<mlir::memref::SubViewOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::SubViewOp subview,
                  mlir::PatternRewriter &rewriter) const override {
    auto sourceType =
        mlir::dyn_cast<mlir::MemRefType>(subview.getSource().getType());
    auto resultType = mlir::dyn_cast<mlir::MemRefType>(subview.getType());
    if (!sourceType || !resultType || sourceType.getRank() != 1 ||
        resultType.getRank() != 1 || subview.getDroppedDims().any())
      return mlir::failure();

    mlir::Location loc = subview.getLoc();
    auto metadata = rewriter.create<mlir::memref::ExtractStridedMetadataOp>(
        loc, subview.getSource());
    copyDiscardableAttrs(subview.getOperation(), metadata.getOperation());

    mlir::OpFoldResult subOffset = subview.getMixedOffsets().front();
    mlir::OpFoldResult subSize = subview.getMixedSizes().front();
    mlir::OpFoldResult subStride = subview.getMixedStrides().front();
    mlir::Value sourceOffset = metadata.getOffset();
    mlir::Value sourceStride = metadata.getStrides().front();
    mlir::Value scaledOffset = mulIndex(loc, sourceStride, subOffset, rewriter);
    mlir::Value finalOffset =
        rewriter.create<mlir::arith::AddIOp>(loc, sourceOffset, scaledOffset);
    mlir::Value finalStride = mulIndex(loc, sourceStride, subStride, rewriter);

    llvm::SmallVector<mlir::OpFoldResult, 1> sizes{subSize};
    llvm::SmallVector<mlir::OpFoldResult, 1> strides{finalStride};
    llvm::SmallVector<mlir::NamedAttribute, 4> attrs(
        subview->getDiscardableAttrs().begin(),
        subview->getDiscardableAttrs().end());
    auto cast = rewriter.create<mlir::memref::ReinterpretCastOp>(
        loc, resultType, metadata.getBaseBuffer(), finalOffset, sizes, strides,
        attrs);
    rewriter.replaceOp(subview, cast.getResult());
    return mlir::success();
  }
};

bool containsPyRuntimeType(mlir::Type type) {
  if (isPyType(type))
    return true;
  if (auto asyncValue = mlir::dyn_cast<mlir::async::ValueType>(type))
    return containsPyRuntimeType(asyncValue.getValueType());
  if (auto memref = mlir::dyn_cast<mlir::MemRefType>(type))
    return containsPyRuntimeType(memref.getElementType());
  if (auto llvmStruct = mlir::dyn_cast<mlir::LLVM::LLVMStructType>(type)) {
    if (llvmStruct.isOpaque())
      return false;
    return llvm::any_of(llvmStruct.getBody(), containsPyRuntimeType);
  }
  if (auto llvmArray = mlir::dyn_cast<mlir::LLVM::LLVMArrayType>(type))
    return containsPyRuntimeType(llvmArray.getElementType());
  return false;
}

mlir::LogicalResult verifyPyTypesLowered(mlir::ModuleOp module) {
  mlir::Operation *offender = nullptr;
  mlir::Type offenderType;
  bool offenderIsOperand = false;
  unsigned offenderIndex = 0;
  module.walk([&](mlir::Operation *op) -> mlir::WalkResult {
    for (auto [index, type] : llvm::enumerate(op->getOperandTypes())) {
      if (!containsPyRuntimeType(type))
        continue;
      offender = op;
      offenderType = type;
      offenderIsOperand = true;
      offenderIndex = index;
      return mlir::WalkResult::interrupt();
    }
    for (auto [index, type] : llvm::enumerate(op->getResultTypes())) {
      if (!containsPyRuntimeType(type))
        continue;
      offender = op;
      offenderType = type;
      offenderIsOperand = false;
      offenderIndex = index;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  if (!offender)
    return mlir::success();
  auto diagnostic =
      offender->emitError("Py runtime type survived lowered ABI finalization");
  diagnostic << " in " << (offenderIsOperand ? "operand" : "result") << " #"
             << offenderIndex << " of type " << offenderType << " on op '"
             << offender->getName() << "'";
  return mlir::failure();
}

mlir::LogicalResult expandMemRefMetadata(mlir::ModuleOp module) {
  // LLVM helper bodies can temporarily contain memref ops plus lowered
  // descriptor casts while the generic conversion is still pending. Expanding
  // memref metadata there materializes memref.extract_strided_metadata on
  // LLVM descriptor structs, which is not a valid high-level memref operand.
  // Keep this pre-expansion scoped to high-level func.func/async.func bodies;
  // llvm.func bodies are handled by the later memref-to-LLVM conversion.
  for (mlir::func::FuncOp fn : module.getOps<mlir::func::FuncOp>()) {
    mlir::RewritePatternSet patterns(module.getContext());
    patterns.add<AttrPreservingRank1SubviewExpansion>(module.getContext(),
                                                      mlir::PatternBenefit(2));
    mlir::memref::populateExpandStridedMetadataPatterns(patterns);
    if (mlir::failed(mlir::applyPatternsGreedily(fn, std::move(patterns))))
      return mlir::failure();
  }
  for (mlir::async::FuncOp fn : module.getOps<mlir::async::FuncOp>()) {
    mlir::RewritePatternSet patterns(module.getContext());
    patterns.add<AttrPreservingRank1SubviewExpansion>(module.getContext(),
                                                      mlir::PatternBenefit(2));
    mlir::memref::populateExpandStridedMetadataPatterns(patterns);
    if (mlir::failed(mlir::applyPatternsGreedily(fn, std::move(patterns))))
      return mlir::failure();
  }
  return mlir::success();
}

mlir::LogicalResult verifyFrontendTypesFinalized(mlir::ModuleOp module) {
  mlir::Operation *unknownOp = nullptr;
  mlir::UnrealizedConversionCastOp materialization = nullptr;

  module.walk([&](mlir::Operation *op) -> mlir::WalkResult {
    if (!op->getName().isRegistered()) {
      unknownOp = op;
      return mlir::WalkResult::interrupt();
    }
    if (auto cast = mlir::dyn_cast<mlir::UnrealizedConversionCastOp>(op)) {
      materialization = cast;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });

  if (unknownOp)
    return unknownOp->emitError(
        "unregistered operation reached runtime lowering; frontend type "
        "inference must reject unknown constructs before lowering");
  if (materialization)
    return materialization.emitError(
        "frontend emitted builtin.unrealized_conversion_cast; all Python "
        "types must be inferred before runtime lowering");
  return mlir::success();
}

bool containsTensorType(mlir::Type type) {
  if (mlir::isa<mlir::TensorType>(type))
    return true;
  if (auto functionType = mlir::dyn_cast<mlir::FunctionType>(type)) {
    return llvm::any_of(functionType.getInputs(), containsTensorType) ||
           llvm::any_of(functionType.getResults(), containsTensorType);
  }
  return false;
}

bool containsMemRefType(mlir::Type type) {
  if (mlir::isa<mlir::MemRefType>(type))
    return true;
  if (auto functionType = mlir::dyn_cast<mlir::FunctionType>(type)) {
    return llvm::any_of(functionType.getInputs(), containsMemRefType) ||
           llvm::any_of(functionType.getResults(), containsMemRefType);
  }
  if (auto asyncValue = mlir::dyn_cast<mlir::async::ValueType>(type))
    return containsMemRefType(asyncValue.getValueType());
  return false;
}

bool opHasMemRefType(mlir::Operation *op) {
  return llvm::any_of(op->getOperandTypes(), containsMemRefType) ||
         llvm::any_of(op->getResultTypes(), containsMemRefType);
}

bool opHasTensorOrLinalgWork(mlir::Operation *op) {
  if (mlir::isa<mlir::linalg::LinalgOp>(op))
    return true;
  return llvm::any_of(op->getOperandTypes(), containsTensorType) ||
         llvm::any_of(op->getResultTypes(), containsTensorType);
}

bool functionHasTensorOrLinalgWork(mlir::FunctionOpInterface function) {
  if (llvm::any_of(function.getArgumentTypes(), containsTensorType) ||
      llvm::any_of(function.getResultTypes(), containsTensorType))
    return true;

  bool found = false;
  function->walk([&](mlir::Operation *op) -> mlir::WalkResult {
    if (op == function.getOperation())
      return mlir::WalkResult::advance();
    if (opHasTensorOrLinalgWork(op)) {
      found = true;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  return found;
}

llvm::SmallVector<mlir::Operation *, 4>
collectTensorWorkFunctions(mlir::ModuleOp module) {
  llvm::SmallVector<mlir::Operation *, 4> functions;
  module.walk([&](mlir::Operation *op) {
    if (auto function = mlir::dyn_cast<mlir::FunctionOpInterface>(op)) {
      if (functionHasTensorOrLinalgWork(function))
        functions.push_back(op);
    }
  });
  return functions;
}

bool isInTensorWorkFunction(
    mlir::Operation *op,
    llvm::ArrayRef<mlir::Operation *> tensorWorkFunctions) {
  auto contains = [&](mlir::Operation *candidate) {
    for (mlir::Operation *function : tensorWorkFunctions)
      if (function == candidate)
        return true;
    return false;
  };

  if (mlir::isa<mlir::ModuleOp>(op))
    return true;
  if (auto function = mlir::dyn_cast<mlir::FunctionOpInterface>(op))
    return contains(function.getOperation());
  if (auto parent = op->getParentOfType<mlir::func::FuncOp>())
    return contains(parent.getOperation());
  return false;
}

mlir::LogicalResult lowerTensorProgramLevelOps(mlir::ModuleOp module,
                                               mlir::MLIRContext *ctx) {
  llvm::SmallVector<mlir::Operation *, 4> tensorWorkFunctions =
      collectTensorWorkFunctions(module);
  if (tensorWorkFunctions.empty())
    return mlir::success();

  mlir::RewritePatternSet elementwisePatterns(ctx);
  mlir::linalg::populateElementwiseToLinalgConversionPatterns(
      elementwisePatterns);
  if (mlir::failed(
          applyPatternsGreedily(module, std::move(elementwisePatterns))))
    return mlir::failure();

  mlir::PassManager pm(ctx);
  mlir::bufferization::OneShotBufferizationOptions options;
  options.allowUnknownOps = true;
  options.bufferizeFunctionBoundaries = true;
  options.setFunctionBoundaryTypeConversion(
      mlir::bufferization::LayoutMapOption::IdentityLayoutMap);
  options.opFilter.allowOperation([tensorWorkFunctions](mlir::Operation *op) {
    return isInTensorWorkFunction(op, tensorWorkFunctions);
  });
  pm.addPass(mlir::bufferization::createOneShotBufferizePass(options));
  pm.addPass(mlir::createConvertLinalgToLoopsPass());
  return pm.run(module);
}

void populateGenericLLVMConversion(PyLLVMTypeConverter &typeConverter,
                                   mlir::RewritePatternSet &patterns) {
  lowering::async_runtime::Patterns::populate(typeConverter, patterns);
  populateSCFToControlFlowConversionPatterns(patterns);
  mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                        patterns);
  mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
  lowering::runtime::memref_to_llvm::Patterns::populate(typeConverter,
                                                        patterns);
  populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
  populateFuncToLLVMConversionPatterns(typeConverter, patterns);
}

void configureGenericLLVMTarget(mlir::ConversionTarget &target) {
  target.addLegalDialect<mlir::LLVM::LLVMDialect>();
  target.addLegalDialect<mlir::async::AsyncDialect>();
  target.addIllegalDialect<mlir::bufferization::BufferizationDialect>();
  target.addIllegalDialect<mlir::linalg::LinalgDialect>();
  target.addIllegalDialect<mlir::tensor::TensorDialect>();
  target.addIllegalDialect<mlir::func::FuncDialect>();
  target.addIllegalDialect<mlir::cf::ControlFlowDialect>();
  target.addIllegalDialect<mlir::scf::SCFDialect>();
  target.addIllegalDialect<mlir::memref::MemRefDialect>();
  target.addIllegalDialect<mlir::arith::ArithDialect>();
  target.addDynamicallyLegalOp<mlir::async::FuncOp>([](mlir::async::FuncOp op) {
    return !containsMemRefType(op.getFunctionType());
  });
  target.addDynamicallyLegalOp<mlir::async::CallOp>(
      [](mlir::async::CallOp op) { return !opHasMemRefType(op); });
  target.addDynamicallyLegalOp<mlir::async::ReturnOp>(
      [](mlir::async::ReturnOp op) { return !opHasMemRefType(op); });
  target.addLegalOp<mlir::ModuleOp>();
  target.addLegalOp<mlir::cf::AssertOp>();
}

struct RuntimeLoweringPass
    : public mlir::PassWrapper<RuntimeLoweringPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RuntimeLoweringPass)

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<PyDialect, mlir::LLVM::LLVMDialect,
                    mlir::async::AsyncDialect, mlir::arith::ArithDialect,
                    mlir::bufferization::BufferizationDialect,
                    mlir::func::FuncDialect, mlir::cf::ControlFlowDialect,
                    mlir::linalg::LinalgDialect, mlir::memref::MemRefDialect,
                    mlir::scf::SCFDialect, mlir::tensor::TensorDialect>();
  }

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::MLIRContext *ctx = module.getContext();
    ctx->loadDialect<PyDialect, mlir::async::AsyncDialect,
                     mlir::arith::ArithDialect,
                     mlir::bufferization::BufferizationDialect,
                     mlir::cf::ControlFlowDialect, mlir::func::FuncDialect,
                     mlir::linalg::LinalgDialect, mlir::memref::MemRefDialect,
                     mlir::scf::SCFDialect, mlir::tensor::TensorDialect>();
    PyLLVMTypeConverter typeConverter(ctx, module);
    bool dumpLowering = static_cast<bool>(
        llvm::sys::Process::GetEnv("LYTHON_DUMP_LOWERING_IR"));
    bool dumpInternalLowering = static_cast<bool>(
        llvm::sys::Process::GetEnv("LYTHON_DUMP_INTERNAL_LOWERING_IR"));

    auto materializationFilter = [](mlir::Diagnostic &) -> mlir::LogicalResult {
      return mlir::failure();
    };

    if (mlir::failed(verifyFrontendTypesFinalized(module))) {
      signalPassFailure();
      return;
    }

    // Phase 1a: Function definition conversion (py.func -> func.func).
    // Keep py.return conversion separate; moving a py.func body and replacing
    // nested returns in the same delayed conversion can invalidate commit
    // ordering for multi-result lowered values.

    auto runFuncConversion = [&]() -> mlir::LogicalResult {
      return lowering::runtime::conversion::runPartial(
          module, ctx,
          [&](mlir::RewritePatternSet &patterns) {
            lowering::runtime::conversion::function::definition::Patterns::
                populate(typeConverter, patterns);
          },
          [&](mlir::ConversionTarget &target) {
            lowering::runtime::conversion::configurePyTarget(target);
            target.addIllegalOp<FuncOp>();
          },
          materializationFilter);
    };

    if (mlir::failed(runFuncConversion())) {
      signalPassFailure();
      return;
    }

    // Phase 1b: Function return conversion (py.return -> func.return).
    auto runReturnConversion = [&]() -> mlir::LogicalResult {
      return lowering::runtime::conversion::runPartial(
          module, ctx,
          [&](mlir::RewritePatternSet &patterns) {
            lowering::runtime::conversion::function::returns::Patterns::
                populate(typeConverter, patterns);
          },
          [&](mlir::ConversionTarget &target) {
            lowering::runtime::conversion::configurePyTarget(target);
            target.addIllegalOp<ReturnOp>();
          },
          materializationFilter);
    };

    if (mlir::failed(runReturnConversion())) {
      signalPassFailure();
      return;
    }

    while (lowering::runtime::cleanup::voidPyReturns(module))
      ;
    lowering::runtime::helpers::synthesizeLocalSelf(module);
    lowering::runtime::helpers::synthesizePublishedBorrow(module);

    if (dumpInternalLowering) {
      llvm::errs() << "[After func conversion]\n";
      module.dump();
    }

    // Phase 2: Function object conversion (py.func_object -> references)

    auto runFuncObjectConversion = [&]() -> mlir::LogicalResult {
      return lowering::runtime::conversion::runPartial(
          module, ctx,
          [&](mlir::RewritePatternSet &patterns) {
            lowering::runtime::conversion::function::objects::Patterns::
                populate(typeConverter, patterns);
          },
          [&](mlir::ConversionTarget &target) {
            lowering::runtime::conversion::configurePyTarget(target);
            target.addIllegalOp<FuncObjectOp, MakeFunctionOp>();
          },
          materializationFilter);
    };

    if (mlir::failed(runFuncObjectConversion())) {
      signalPassFailure();
      return;
    }
    if (dumpInternalLowering) {
      llvm::errs() << "[After func object conversion]\n";
      module.dump();
    }

    optimizer::scalar::foldIntConstants(module);

    // Phase 3: Call conversion (py.call_vector/py.call -> calls)

    auto runCallConversion = [&]() -> mlir::LogicalResult {
      return lowering::runtime::conversion::runPartial(
          module, ctx,
          [&](mlir::RewritePatternSet &patterns) {
            lowering::runtime::conversion::call::Patterns::populate(
                typeConverter, patterns);
          },
          [&](mlir::ConversionTarget &target) {
            lowering::runtime::conversion::configurePyTarget(target);
            target.addIllegalOp<CallVectorOp, CallOp, InvokeOp>();
          },
          materializationFilter);
    };

    if (mlir::failed(runCallConversion())) {
      signalPassFailure();
      return;
    }
    // Apply pre-lowering optimizations
    optimizer::pipeline::preLowering(module);
    if (mlir::failed(verifyOwnership(module))) {
      signalPassFailure();
      return;
    }

    if (dumpInternalLowering) {
      llvm::errs() << "[After call conversion]\n";
      module.dump();
    }

    // Phase 4: Structured exception control-flow conversion. This must run
    // before general value conversion so container lowering cannot materialize
    // memref.alloca inside py.try regions, which are not allocation scopes.
    auto runTryConversion = [&]() -> mlir::LogicalResult {
      mlir::RewritePatternSet patterns(ctx);
      lowering::runtime::conversion::try_phase::Patterns::populate(
          typeConverter, patterns);
      if (mlir::failed(applyPatternsGreedily(module, std::move(patterns))))
        return mlir::failure();
      return mlir::success(
          !lowering::runtime::conversion::try_ops::hasStructured(module));
    };

    if (mlir::failed(runTryConversion())) {
      if (dumpInternalLowering) {
        llvm::errs() << "[Try conversion failed]\n";
        module.dump();
      }
      signalPassFailure();
      return;
    }
    if (std::getenv("LYTHON_DEBUG_TRY_LOWERING"))
      llvm::errs() << "[TryPhase] pre-lowering opts\n";
    optimizer::pipeline::preLowering(module);
    if (mlir::failed(verifyOwnership(module))) {
      signalPassFailure();
      return;
    }
    if (std::getenv("LYTHON_DEBUG_TRY_LOWERING"))
      llvm::errs() << "[TryPhase] erase unreachable\n";
    lowering::runtime::cleanup::unreachableBlocks(module);
    if (std::getenv("LYTHON_DEBUG_TRY_LOWERING"))
      llvm::errs() << "[TryPhase] done cleanup\n";

    if (dumpInternalLowering) {
      llvm::errs() << "[After try conversion]\n";
      module.dump();
    }

    // Runtime MLIR signatures are part of the ABI contract. Import them before
    // value conversion so RuntimeAPI can adapt operands by signature instead of
    // classifying callees by name. A second call after value conversion
    // materializes contracts on newly emitted calls.
    if (mlir::failed(runtime_library::embedObjectModules(module))) {
      signalPassFailure();
      return;
    }

    // Phase 5: mlir::Value conversion (py.* ops -> LLVM ops)

    auto runValueConversion = [&]() -> mlir::LogicalResult {
      return lowering::runtime::conversion::runPartial(
          module, ctx,
          [&](mlir::RewritePatternSet &patterns) {
            lowering::runtime::conversion::value::Patterns::populate(
                typeConverter, patterns);
          },
          [&](mlir::ConversionTarget &target) {
            lowering::runtime::conversion::types::configureValueTarget(
                target, typeConverter);
          },
          materializationFilter);
    };

    if (mlir::failed(runValueConversion())) {
      if (dumpInternalLowering) {
        llvm::errs() << "[mlir::Value conversion failed]\n";
        module.dump();
      }
      signalPassFailure();
      return;
    }

    // Apply Python-value cleanup at the boundary where object-level scalar
    // concepts have just been lowered to runtime/LLVM calls.
    optimizer::pipeline::postValueLowering(module);
    if (mlir::failed(runtime_library::embedObjectModules(module))) {
      signalPassFailure();
      return;
    }

    // A Python-typed unwind payload reaching LLVM invoke lowering is an ABI
    // leak. Do not hide it behind llvm.ptr; the exception lowering must choose
    // the descriptor shape before this boundary.
    bool hasUnloweredUnwindPyArg = false;
    module.walk(
        [&](mlir::LLVM::InvokeOp invoke) {
          mlir::Block *unwind = invoke.getUnwindDest();
          if (!unwind)
            return;
          for (mlir::BlockArgument arg : unwind->getArguments()) {
            if (!isPyType(arg.getType()))
              continue;
            invoke.emitError()
                << "unwind block argument kept high-level Python type "
                << arg.getType()
                << " after value conversion; lower it to the exception "
                   "descriptor ABI before LLVM invoke lowering";
            hasUnloweredUnwindPyArg = true;
          }
        });
    if (hasUnloweredUnwindPyArg) {
      signalPassFailure();
      return;
    }

    // Some passes may drop llvm.personality; restore it for landingpads.
    lowering::runtime::eh::ensureFuncPersonalities(module);

    if (dumpInternalLowering) {
      llvm::errs() << "[After optimizations]\n";
      module.dump();
    }

    // Async result storage outlives the coroutine frame. Ensure container
    // payload descriptors were promoted before memref-to-LLVM erases
    // stack-vs-heap allocation provenance.
    if (mlir::failed(lowering::runtime::async::verifyReturnPayloads(module))) {
      signalPassFailure();
      return;
    }

    // Phase 5: Freeze the lowered Py ABI. After this boundary all !py.* value
    // shapes must be represented by their fixed memref/LLVM forms.

    while (lowering::runtime::cleanup::pyBridgeCasts(module))
      ;
    while (lowering::runtime::cleanup::pyMultiCasts(module))
      ;
    if (mlir::failed(verifyPyTypesLowered(module))) {
      signalPassFailure();
      return;
    }

    // Bufferize tensor/linalg values before generic LLVM conversion. Leaving
    // tensor ops legal during arith-to-LLVM forces index casts back from i64,
    // which hides an ABI mismatch until the final cleanup verifier.
    if (mlir::failed(lowerTensorProgramLevelOps(module, ctx))) {
      signalPassFailure();
      return;
    }
    if (dumpInternalLowering) {
      llvm::errs() << "[After tensor bufferization]\n";
      module.dump();
    }

    // Header/payload object ABI uses memref.subview to expose the common
    // header. Expand it before memref-to-LLVM, where SubViewOp is intentionally
    // illegal.
    if (mlir::failed(expandMemRefMetadata(module))) {
      signalPassFailure();
      return;
    }

    // Phase 6: Convert func.func to llvm.func before final EH materialization.
    {
      if (dumpInternalLowering) {
        llvm::errs() << "[Before func-to-llvm conversion]\n";
        module.dump();
      }
      LoweredSafetyContracts safetyContracts;
      collectLoweredSafetyContracts(module, typeConverter, safetyContracts);

      mlir::RewritePatternSet patterns(ctx);
      populateGenericLLVMConversion(typeConverter, patterns);
      mlir::ConversionTarget target(*ctx);
      configureGenericLLVMTarget(target);
      if (mlir::failed(
              applyPartialConversion(module, target, std::move(patterns)))) {
        signalPassFailure();
        return;
      }
      if (mlir::failed(
              preserveLoweredSafetyContracts(module, safetyContracts))) {
        signalPassFailure();
        return;
      }
      if (dumpInternalLowering) {
        llvm::errs() << "[After func-to-llvm conversion]\n";
        module.dump();
      }
    }

    while (lowering::runtime::cleanup::pyMultiCasts(module))
      ;
    while (lowering::runtime::cleanup::memrefDescriptorCasts(module))
      ;
    while (lowering::runtime::cleanup::memrefRuntimeCalls(module))
      ;
    while (lowering::runtime::cleanup::memrefDescriptorCasts(module))
      ;
    while (lowering::runtime::cleanup::pointerRoundTrips(module))
      ;
    if (mlir::failed(lowering::verifyNoUnrealizedCasts(
            module, "runtime lowering boundary"))) {
      signalPassFailure();
      return;
    }
    if (dumpLowering) {
      llvm::errs() << "[After RuntimeLowering cleanup]\n";
      module.dump();
    }

    // Finalize unwind blocks with landingpad in LLVM world.
    lowering::runtime::eh::finalizeUnwindBlocks(module);
    if (dumpLowering) {
      llvm::errs() << "[After EH finalize]\n";
      module.dump();
    }

    // Insert a top-level exception handler wrapper for `main`.
    lowering::runtime::eh::wrapTopLevelMain(module);

    if (mlir::failed(verifyLLVMCallOwnership(module))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createRuntimeLoweringPass() {
  return std::make_unique<RuntimeLoweringPass>();
}

} // namespace py
