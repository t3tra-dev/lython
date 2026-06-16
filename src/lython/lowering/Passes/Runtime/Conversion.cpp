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
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/ADT/STLExtras.h"

#include "PyDialect.h.inc"

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

namespace py::lowering::runtime::conversion {

namespace {

llvm::SmallVector<mlir::Value>
flatten(llvm::ArrayRef<mlir::ValueRange> ranges) {
  llvm::SmallVector<mlir::Value> values;
  for (mlir::ValueRange range : ranges)
    values.append(range.begin(), range.end());
  return values;
}

llvm::SmallVector<mlir::Type> typesOf(mlir::ValueRange values) {
  llvm::SmallVector<mlir::Type> types;
  for (mlir::Value value : values)
    types.push_back(value.getType());
  return types;
}

bool sameTypes(mlir::TypeRange lhs, llvm::ArrayRef<mlir::Type> rhs) {
  if (lhs.size() != rhs.size())
    return false;
  return llvm::all_of(llvm::zip(lhs, rhs), [](auto pair) {
    return std::get<0>(pair) == std::get<1>(pair);
  });
}

mlir::FailureOr<mlir::Block *>
convertSuccessorBlock(mlir::ConversionPatternRewriter &rewriter,
                      const mlir::TypeConverter &converter,
                      mlir::Operation *branchOp, mlir::Block *block,
                      mlir::ValueRange operands) {
  if (!block)
    return rewriter.notifyMatchFailure(branchOp, "missing successor block");
  if (block->isEntryBlock())
    return rewriter.notifyMatchFailure(
        branchOp, "entry block cannot be converted through a branch");

  llvm::SmallVector<mlir::Type> expected = typesOf(operands);
  if (sameTypes(block->getArgumentTypes(), expected))
    return block;

  std::optional<mlir::TypeConverter::SignatureConversion> conversion =
      converter.convertBlockSignature(block);
  if (!conversion)
    return rewriter.notifyMatchFailure(branchOp,
                                       "could not compute block signature");
  if (!sameTypes(conversion->getConvertedTypes(), expected))
    return rewriter.notifyMatchFailure(
        branchOp,
        "mismatch between converted branch operands and block signature");
  return rewriter.applySignatureConversion(block, *conversion, &converter);
}

struct BranchLowering : mlir::OpConversionPattern<mlir::cf::BranchOp> {
  BranchLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<mlir::cf::BranchOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::cf::BranchOp op, OneToNOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<mlir::Value> operands = flatten(adaptor.getOperands());
    mlir::FailureOr<mlir::Block *> dest = convertSuccessorBlock(
        rewriter, *getTypeConverter(), op, op.getDest(), operands);
    if (mlir::failed(dest))
      return mlir::failure();
    rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(op, *dest, operands);
    return mlir::success();
  }
};

struct CondBranchLowering : mlir::OpConversionPattern<mlir::cf::CondBranchOp> {
  CondBranchLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<mlir::cf::CondBranchOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::cf::CondBranchOp op, OneToNOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    llvm::ArrayRef<mlir::ValueRange> operandGroups = adaptor.getOperands();
    if (operandGroups.empty() || operandGroups.front().size() != 1)
      return rewriter.notifyMatchFailure(
          op, "condition must remain a single lowered value");

    unsigned trueCount = op.getNumTrueOperands();
    unsigned falseCount = op.getNumFalseOperands();
    if (operandGroups.size() != 1u + trueCount + falseCount)
      return rewriter.notifyMatchFailure(op, "unexpected operand group count");

    mlir::Value condition = operandGroups.front().front();
    llvm::ArrayRef<mlir::ValueRange> trueGroups =
        operandGroups.slice(1, trueCount);
    llvm::ArrayRef<mlir::ValueRange> falseGroups =
        operandGroups.slice(1 + trueCount, falseCount);
    llvm::SmallVector<mlir::Value> trueOperands = flatten(trueGroups);
    llvm::SmallVector<mlir::Value> falseOperands = flatten(falseGroups);

    mlir::FailureOr<mlir::Block *> trueDest = convertSuccessorBlock(
        rewriter, *getTypeConverter(), op, op.getTrueDest(), trueOperands);
    if (mlir::failed(trueDest))
      return mlir::failure();
    mlir::FailureOr<mlir::Block *> falseDest = convertSuccessorBlock(
        rewriter, *getTypeConverter(), op, op.getFalseDest(), falseOperands);
    if (mlir::failed(falseDest))
      return mlir::failure();

    rewriter.replaceOpWithNewOp<mlir::cf::CondBranchOp>(
        op, condition, *trueDest, trueOperands, *falseDest, falseOperands);
    return mlir::success();
  }
};

struct PySelectToScfIf : mlir::OpRewritePattern<mlir::arith::SelectOp> {
  using mlir::OpRewritePattern<mlir::arith::SelectOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::arith::SelectOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!isPyType(op.getResult().getType()))
      return mlir::failure();

    auto ifOp = rewriter.create<mlir::scf::IfOp>(
        op.getLoc(), mlir::TypeRange{op.getResult().getType()},
        op.getCondition(), /*withElseRegion=*/true);
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(ifOp.thenBlock());
      rewriter.create<mlir::scf::YieldOp>(op.getLoc(), op.getTrueValue());
    }
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(ifOp.elseBlock());
      rewriter.create<mlir::scf::YieldOp>(op.getLoc(), op.getFalseValue());
    }
    rewriter.replaceOp(op, ifOp.getResults());
    return mlir::success();
  }
};

} // namespace

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

namespace function::definition::Patterns {
void populate(PyLLVMTypeConverter &typeConverter,
              mlir::RewritePatternSet &patterns) {
  ::py::lowering::func::definition::Patterns::populate(typeConverter, patterns);
}
} // namespace function::definition::Patterns

namespace function::returns::Patterns {
void populate(PyLLVMTypeConverter &typeConverter,
              mlir::RewritePatternSet &patterns) {
  ::py::lowering::func::returns::Patterns::populate(typeConverter, patterns);
}
} // namespace function::returns::Patterns

namespace function::objects::Patterns {
void populate(PyLLVMTypeConverter &typeConverter,
              mlir::RewritePatternSet &patterns) {
  ::py::lowering::func::objects::Patterns::populate(typeConverter, patterns);
}
} // namespace function::objects::Patterns

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
  ::py::lowering::value::union_::Patterns::populate(typeConverter, patterns);
  ::py::lowering::async_runtime::Patterns::populate(typeConverter, patterns);
  ::py::lowering::refcount::Patterns::populate(typeConverter, patterns);
  ::py::lowering::runtime::upcast::Patterns::populate(typeConverter, patterns);
  mlir::scf::populateSCFStructuralTypeConversions(typeConverter, patterns);
  mlir::scf::populateSCFStructuralOneToNTypeConversions(typeConverter,
                                                        patterns);
  patterns.add<BranchLowering, CondBranchLowering>(typeConverter,
                                                   patterns.getContext());
  patterns.add<PySelectToScfIf>(patterns.getContext());
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
      mlir::isa<TupleType, ListType, ClassType, DictType>(type))
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
  target.addDynamicallyLegalOp<mlir::arith::SelectOp>(
      [](mlir::arith::SelectOp op) {
        return llvm::none_of(op.getOperandTypes(), containsPyRuntime) &&
               llvm::none_of(op->getResultTypes(), containsPyRuntime);
      });
  auto blockLowered = [](mlir::Block *block) {
    return block && llvm::all_of(block->getArgumentTypes(), loweredRuntime);
  };
  target.addDynamicallyLegalOp<mlir::cf::BranchOp>([&](mlir::cf::BranchOp op) {
    return llvm::all_of(op.getOperandTypes(), loweredRuntime) &&
           blockLowered(op.getDest());
  });
  target.addDynamicallyLegalOp<mlir::cf::CondBranchOp>(
      [&](mlir::cf::CondBranchOp op) {
        return llvm::all_of(op.getOperandTypes(), loweredRuntime) &&
               blockLowered(op.getTrueDest()) &&
               blockLowered(op.getFalseDest());
      });
  target.addLegalOp<mlir::ModuleOp>();
  target.addLegalOp<mlir::UnrealizedConversionCastOp>();
  target.addDynamicallyLegalOp<mlir::scf::IfOp>([](mlir::scf::IfOp op) {
    return llvm::none_of(op.getResultTypes(), containsPyRuntime);
  });
  target.addDynamicallyLegalOp<mlir::scf::ForOp>([](mlir::scf::ForOp op) {
    return llvm::none_of(op.getResultTypes(), containsPyRuntime) &&
           llvm::none_of(op.getInitArgs().getTypes(), containsPyRuntime);
  });
  target.addDynamicallyLegalOp<mlir::scf::WhileOp>([](mlir::scf::WhileOp op) {
    return llvm::none_of(op.getResultTypes(), containsPyRuntime) &&
           llvm::none_of(op.getOperandTypes(), containsPyRuntime);
  });
  target.addDynamicallyLegalOp<mlir::scf::YieldOp>([](mlir::scf::YieldOp op) {
    if (!mlir::isa<mlir::scf::IfOp, mlir::scf::ForOp, mlir::scf::WhileOp>(
            op->getParentOp()))
      return true;
    return llvm::none_of(op.getOperandTypes(), containsPyRuntime);
  });
  target.addIllegalOp<
      StrConstantOp, IntConstantOp, FloatConstantOp, TupleEmptyOp,
      TupleCreateOp, GetItemOp, ContainsOp, DictEmptyOp, DictInsertOp,
      ListNewOp, ListAppendOp, ListRemoveOp, NoneOp, CallableObjectOp,
      MakeFunctionOp, AddOp, SubOp, MulOp, DivOp, FloorDivOp, ModOp, LShiftOp,
      RShiftOp, BitAndOp, BitOrOp, BitXorOp, LtOp, LeOp, GtOp, GeOp, EqOp, NeOp,
      ReprOp, StrConcat3Op, CastToPrimOp, CastFromPrimOp, IncRefOp, DecRefOp,
      ClassNewOp, ClassPromoteOp, ClassUpcastOp, ClassRefineOp, ClassTestOp,
      ProtocolViewOp, PublishOp, AttrGetOp, AttrSetOp, AttrGetLocalOp,
      AttrSetLocalOp, ClassOp, ExceptionNullOp, TracebackNullOp,
      LocationCurrentOp, ExceptionNewOp, RaiseOp, RaiseCurrentOp, TryOp,
      TryYieldOp, ExceptYieldOp, FinallyYieldOp, ExceptMatchOp, EnterOp, ExitOp,
      AEnterOp, AExitOp, SendOp, ThrowOp, CloseOp, ASendOp, AThrowOp, ACloseOp,
      IterOp, NextOp, AwaitOp, AsyncNextOp, UnionWrapOp, UnionTestOp,
      UnionUnwrapOp, LenOp>();
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
