#include "Passes/Runtime/Conversion.h"

#include "Passes/Runtime/Upcast.h"

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
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"

#include "PyDialect.h.inc"

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

namespace py::lowering::runtime::conversion {

mlir::LogicalResult
runPartial(mlir::ModuleOp module, mlir::MLIRContext *ctx,
           llvm::function_ref<void(mlir::RewritePatternSet &)> populatePatterns,
           llvm::function_ref<void(mlir::ConversionTarget &)> configureTarget,
           llvm::function_ref<mlir::LogicalResult(mlir::Diagnostic &)>
               materializationFilter) {
  mlir::RewritePatternSet patterns(ctx);
  populatePatterns(patterns);

  mlir::ConversionTarget target(*ctx);
  configureTarget(target);

  mlir::ScopedDiagnosticHandler diagHandler(ctx, materializationFilter);
  return applyPartialConversion(module, target, std::move(patterns));
}

namespace function::Patterns {
void populate(PyLLVMTypeConverter &typeConverter,
              mlir::RewritePatternSet &patterns) {
  ::py::lowering::func::Patterns::populate(typeConverter, patterns);
}
} // namespace function::Patterns

void configurePyTarget(mlir::ConversionTarget &target) {
  target.addLegalDialect<
      ::py::PyDialect, mlir::LLVM::LLVMDialect, mlir::async::AsyncDialect,
      mlir::arith::ArithDialect, mlir::bufferization::BufferizationDialect,
      mlir::cf::ControlFlowDialect, mlir::func::FuncDialect,
      mlir::linalg::LinalgDialect, mlir::memref::MemRefDialect,
      mlir::scf::SCFDialect, mlir::tensor::TensorDialect>();
  target.addLegalOp<mlir::ModuleOp>();
  target.addLegalOp<mlir::UnrealizedConversionCastOp>();
}

namespace call::Patterns {
void populate(PyLLVMTypeConverter &typeConverter,
              mlir::RewritePatternSet &patterns) {
  ::py::lowering::call::Patterns::populate(typeConverter, patterns);
}
} // namespace call::Patterns

namespace value::Patterns {
void populate(PyLLVMTypeConverter &typeConverter,
              mlir::RewritePatternSet &patterns) {
  ::py::lowering::value::base::Patterns::populate(typeConverter, patterns);
  ::py::lowering::value::list::Patterns::populate(typeConverter, patterns);
  ::py::lowering::value::number::Patterns::populate(typeConverter, patterns);
  ::py::lowering::value::class_::Patterns::populate(typeConverter, patterns);
  ::py::lowering::value::tuple::Patterns::populate(typeConverter, patterns);
  ::py::lowering::value::dict::Patterns::populate(typeConverter, patterns);
  ::py::lowering::async_runtime::Patterns::populate(typeConverter, patterns);
  ::py::lowering::refcount::Patterns::populate(typeConverter, patterns);
  ::py::lowering::runtime::upcast::Patterns::populate(typeConverter, patterns);
}
} // namespace value::Patterns

namespace try_phase::Patterns {
void populate(PyLLVMTypeConverter &typeConverter,
              mlir::RewritePatternSet &patterns) {
  ::py::lowering::try_ops::Patterns::populate(typeConverter, patterns);
}
} // namespace try_phase::Patterns

namespace try_ops {

bool hasStructured(mlir::ModuleOp module) {
  bool found = false;
  module.walk([&](mlir::Operation *op) -> mlir::WalkResult {
    if (!mlir::isa<TryOp, TryYieldOp, ExceptYieldOp, FinallyYieldOp>(op))
      return mlir::WalkResult::advance();
    found = true;
    return mlir::WalkResult::interrupt();
  });
  return found;
}

} // namespace try_ops

namespace types {

bool containsPyRuntime(mlir::Type type) {
  if (isPyType(type) ||
      mlir::isa<FuncType, TupleType, ListType, ClassType, DictType, ObjectType,
                CoroutineType, FutureType, TaskType>(type))
    return true;
  if (auto asyncValue = mlir::dyn_cast<mlir::async::ValueType>(type))
    return containsPyRuntime(asyncValue.getValueType());
  if (auto memref = mlir::dyn_cast<mlir::MemRefType>(type))
    return containsPyRuntime(memref.getElementType());
  if (auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(type))
    return containsPyRuntime(tensor.getElementType());
  if (auto llvmStruct = mlir::dyn_cast<mlir::LLVM::LLVMStructType>(type)) {
    if (llvmStruct.isOpaque())
      return false;
    return llvm::any_of(llvmStruct.getBody(), containsPyRuntime);
  }
  if (auto llvmArray = mlir::dyn_cast<mlir::LLVM::LLVMArrayType>(type))
    return containsPyRuntime(llvmArray.getElementType());
  return false;
}

bool loweredRuntime(mlir::Type type) { return !containsPyRuntime(type); }

void configureValueTarget(mlir::ConversionTarget &target,
                          PyLLVMTypeConverter &typeConverter) {
  (void)typeConverter;
  target.addLegalDialect<
      mlir::LLVM::LLVMDialect, mlir::async::AsyncDialect,
      mlir::func::FuncDialect, mlir::arith::ArithDialect,
      mlir::cf::ControlFlowDialect, mlir::tensor::TensorDialect,
      mlir::linalg::LinalgDialect, mlir::memref::MemRefDialect,
      mlir::scf::SCFDialect, mlir::bufferization::BufferizationDialect>();
  target.addDynamicallyLegalOp<mlir::async::FuncOp>(
      [&](mlir::async::FuncOp op) {
        if (!llvm::all_of(op.getFunctionType().getInputs(), loweredRuntime) ||
            !llvm::all_of(op.getFunctionType().getResults(), loweredRuntime))
          return false;
        if (op.isDeclaration())
          return true;
        return llvm::all_of(op.getBody().front().getArgumentTypes(),
                            loweredRuntime);
      });
  target.addDynamicallyLegalOp<mlir::async::CallOp>(
      [&](mlir::async::CallOp op) {
        return llvm::all_of(op.getResultTypes(), loweredRuntime) &&
               llvm::all_of(op.getOperandTypes(), loweredRuntime);
      });
  target.addDynamicallyLegalOp<mlir::async::ReturnOp>(
      [&](mlir::async::ReturnOp op) {
        return llvm::all_of(op.getOperandTypes(), loweredRuntime);
      });
  target.addLegalOp<mlir::ModuleOp>();
  target.addLegalOp<mlir::UnrealizedConversionCastOp>();
  target.addIllegalOp<
      StrConstantOp, IntConstantOp, FloatConstantOp, TupleEmptyOp,
      TupleCreateOp, DictEmptyOp, DictInsertOp, DictGetOp, ListNewOp,
      ListAppendOp, ListRemoveOp, ListGetOp, NoneOp, FuncObjectOp,
      MakeFunctionOp, AddOp, SubOp, LtOp, LeOp, GtOp, GeOp, EqOp, NeOp, ReprOp,
      CastToPrimOp, CastFromPrimOp, UpcastOp, IncRefOp, DecRefOp, ClassNewOp,
      ClassPromoteOp, PublishOp, AttrGetOp, AttrSetOp, AttrGetLocalOp,
      AttrSetLocalOp, ClassOp, ExceptionNullOp, TracebackNullOp,
      LocationCurrentOp, ExceptionNewOp, RaiseOp, RaiseCurrentOp, TryOp,
      TryYieldOp, ExceptYieldOp, FinallyYieldOp, ExceptMatchOp, CoroCreateOp,
      CoroStartOp, AwaitOp, TaskCreateOp, TaskCancelOp, AsyncSleepOp,
      AsyncGatherOp>();
}

bool same(mlir::TypeRange lhs, llvm::ArrayRef<mlir::Type> rhs) {
  if (lhs.size() != rhs.size())
    return false;
  return llvm::all_of(llvm::zip(lhs, rhs), [](auto pair) {
    return std::get<0>(pair) == std::get<1>(pair);
  });
}

} // namespace types

} // namespace py::lowering::runtime::conversion
