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
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
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

namespace optimizer::call {
void staticDefaults(mlir::ModuleOp module);
} // namespace optimizer::call

namespace {

// RuntimeLoweringPass: Main pipeline orchestration

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
  module.walk([&](mlir::Operation *op) -> mlir::WalkResult {
    for (mlir::Type type : op->getOperandTypes()) {
      if (!containsPyRuntimeType(type))
        continue;
      offender = op;
      return mlir::WalkResult::interrupt();
    }
    for (mlir::Type type : op->getResultTypes()) {
      if (!containsPyRuntimeType(type))
        continue;
      offender = op;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  if (!offender)
    return mlir::success();
  return offender->emitError(
      "Py runtime type survived lowered ABI finalization");
}

mlir::LogicalResult verifyNoUnrealizedCasts(mlir::ModuleOp module) {
  mlir::UnrealizedConversionCastOp offender = nullptr;
  module.walk([&](mlir::UnrealizedConversionCastOp cast) {
    offender = cast;
    return mlir::WalkResult::interrupt();
  });
  if (!offender)
    return mlir::success();
  return offender.emitError(
      "unrealized conversion cast survived lowering boundary");
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

struct RuntimeLoweringPass
    : public mlir::PassWrapper<RuntimeLoweringPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RuntimeLoweringPass)

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::LLVM::LLVMDialect, mlir::async::AsyncDialect,
                    mlir::arith::ArithDialect,
                    mlir::bufferization::BufferizationDialect,
                    mlir::func::FuncDialect, mlir::cf::ControlFlowDialect,
                    mlir::linalg::LinalgDialect, mlir::memref::MemRefDialect,
                    mlir::scf::SCFDialect, mlir::tensor::TensorDialect>();
  }

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::MLIRContext *ctx = module.getContext();
    ctx->loadDialect<mlir::async::AsyncDialect, mlir::arith::ArithDialect,
                     mlir::bufferization::BufferizationDialect,
                     mlir::cf::ControlFlowDialect, mlir::func::FuncDialect,
                     mlir::linalg::LinalgDialect, mlir::memref::MemRefDialect,
                     mlir::scf::SCFDialect, mlir::tensor::TensorDialect>();
    PyLLVMTypeConverter typeConverter(ctx);
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

    optimizer::call::staticDefaults(module);

    // Phase 1: Function conversion (py.func/py.return -> func.func/func.return)

    auto runFuncConversion = [&]() -> mlir::LogicalResult {
      return lowering::runtime::conversion::runPartial(
          module, ctx,
          [&](mlir::RewritePatternSet &patterns) {
            lowering::runtime::conversion::function::Patterns::populate(
                typeConverter, patterns);
          },
          [&](mlir::ConversionTarget &target) {
            lowering::runtime::conversion::configurePyTarget(target);
            target.addIllegalOp<FuncOp, ReturnOp>();
          },
          materializationFilter);
    };

    if (mlir::failed(runFuncConversion())) {
      signalPassFailure();
      return;
    }
    while (lowering::runtime::cleanup::voidPyReturns(module))
      ;
    lowering::runtime::helpers::retainBorrowedEntryBlockReturns(module);
    lowering::runtime::helpers::synthesizeLocalSelf(module);
    lowering::runtime::helpers::synthesizePublishedBorrow(module);

    if (dumpInternalLowering) {
      llvm::errs() << "[After func conversion]\n";
      module.dump();
    }

    optimizer::call::staticDefaults(module);

    // Phase 2: Function object conversion (py.func_object -> references)

    auto runFuncObjectConversion = [&]() -> mlir::LogicalResult {
      return lowering::runtime::conversion::runPartial(
          module, ctx,
          [&](mlir::RewritePatternSet &patterns) {
            lowering::runtime::conversion::function::Patterns::populate(
                typeConverter, patterns);
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

    // Apply post-lowering optimizations
    optimizer::pipeline::postLowering(module);

    // Normalize invoke unwind block arguments to LLVM pointer types.
    {
      auto pyObject = mlir::LLVM::LLVMPointerType::get(ctx);
      module.walk([&](mlir::LLVM::InvokeOp invoke) {
        mlir::Block *unwind = invoke.getUnwindDest();
        if (!unwind)
          return;
        for (mlir::BlockArgument arg :
             llvm::make_early_inc_range(unwind->getArguments())) {
          if (!isPyType(arg.getType()))
            continue;
          arg.setType(pyObject);
          for (auto &use : llvm::make_early_inc_range(arg.getUses())) {
            auto *owner = use.getOwner();
            auto cast = mlir::dyn_cast<mlir::UnrealizedConversionCastOp>(owner);
            if (!cast)
              continue;
            if (cast->getNumOperands() != 1 || cast->getNumResults() != 1)
              continue;
            if (cast.getResult(0).getType() != pyObject)
              continue;
            cast.getResult(0).replaceAllUsesWith(arg);
            cast.erase();
          }
        }
      });
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

    // Phase 6: Convert func.func to llvm.func before final EH materialization.
    {
      if (dumpInternalLowering) {
        llvm::errs() << "[Before func-to-llvm conversion]\n";
        module.dump();
      }
      LoweredSafetyContracts safetyContracts;
      collectLoweredSafetyContracts(module, typeConverter, safetyContracts);

      mlir::RewritePatternSet patterns(ctx);
      populateSCFToControlFlowConversionPatterns(patterns);
      mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                            patterns);
      mlir::arith::populateArithToLLVMConversionPatterns(typeConverter,
                                                         patterns);
      lowering::runtime::memref_to_llvm::Patterns::populate(typeConverter,
                                                            patterns);
      populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
      populateFuncToLLVMConversionPatterns(typeConverter, patterns);
      mlir::ConversionTarget target(*ctx);
      target.addLegalDialect<mlir::LLVM::LLVMDialect>();
      target.addLegalDialect<mlir::async::AsyncDialect>();
      target.addLegalDialect<mlir::bufferization::BufferizationDialect>();
      target.addLegalDialect<mlir::linalg::LinalgDialect>();
      target.addLegalDialect<mlir::tensor::TensorDialect>();
      target.addIllegalDialect<mlir::func::FuncDialect>();
      target.addIllegalDialect<mlir::cf::ControlFlowDialect>();
      target.addIllegalDialect<mlir::scf::SCFDialect>();
      target.addIllegalDialect<mlir::memref::MemRefDialect>();
      target.addIllegalDialect<mlir::arith::ArithDialect>();
      target.addLegalOp<mlir::ModuleOp>();
      target.addLegalOp<mlir::cf::AssertOp>();
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
    if (mlir::failed(verifyNoUnrealizedCasts(module))) {
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
