// This file implements the main RuntimeLoweringPass which orchestrates the
// complete lowering pipeline from Py dialect to LLVM dialect. It coordinates
// the various conversion phases:
//   1. Function conversion (py.callable.func -> func.func)
//   2. Callable object conversion (py.callable.object -> references)
//   3. Call conversion (py.call -> runtime calls or direct calls)
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
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>
#include <optional>
#include <string>

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

bool runtimePerfEnabled() {
  static const bool enabled = [] {
    auto value = llvm::sys::Process::GetEnv("LYTHON_PERF");
    if (!value)
      return false;
    llvm::StringRef text(*value);
    return text == "1" || text.equals_insensitive("true") ||
           text.equals_insensitive("yes") || text.equals_insensitive("on");
  }();
  return enabled;
}

class RuntimePerfScope {
public:
  explicit RuntimePerfScope(llvm::StringRef phase) : phase(phase.str()) {
    if (!runtimePerfEnabled())
      return;
    enabled = true;
    start = Clock::now();
  }

  ~RuntimePerfScope() {
    if (!enabled)
      return;
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                       Clock::now() - start)
                       .count();
    llvm::errs() << "[LYTHON_PERF] phase=" << phase << " wall_us=" << elapsed
                 << "\n";
  }

private:
  using Clock = std::chrono::steady_clock;
  std::string phase;
  bool enabled = false;
  Clock::time_point start;
};

template <typename Fn>
mlir::LogicalResult timedRuntimePhase(llvm::StringRef name, Fn run) {
  std::string phase = ("lowering.runtime-lowering." + name).str();
  RuntimePerfScope perf(phase);
  return run();
}

void copyDiscardableAttrs(mlir::Operation *from, mlir::Operation *to) {
  if (!from || !to)
    return;
  for (const mlir::NamedAttribute &attr : from->getDiscardableAttrs())
    to->setDiscardableAttr(attr.getName(), attr.getValue());
}

mlir::Value materializeIndex(mlir::Location loc, mlir::OpFoldResult value,
                             mlir::OpBuilder &builder) {
  if (auto dynamic = mlir::dyn_cast<mlir::Value>(value))
    return dynamic;
  auto attr = mlir::cast<mlir::IntegerAttr>(mlir::cast<mlir::Attribute>(value));
  return builder.create<mlir::arith::ConstantIndexOp>(loc, attr.getInt());
}

mlir::Value mulIndex(mlir::Location loc, mlir::Value lhs,
                     mlir::OpFoldResult rhs, mlir::OpBuilder &builder) {
  if (auto attr = mlir::dyn_cast<mlir::Attribute>(rhs)) {
    int64_t value = mlir::cast<mlir::IntegerAttr>(attr).getInt();
    if (value == 1)
      return lhs;
    if (value == 0)
      return builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
  }
  return builder.create<mlir::arith::MulIOp>(
      loc, lhs, materializeIndex(loc, rhs, builder));
}

bool getStaticRank1Layout(mlir::MemRefType type, int64_t &offset,
                          int64_t &stride) {
  llvm::SmallVector<int64_t, 1> strides;
  if (mlir::failed(type.getStridesAndOffset(strides, offset)) ||
      strides.size() != 1)
    return false;
  if (mlir::ShapedType::isDynamic(offset) ||
      mlir::ShapedType::isDynamic(strides.front()))
    return false;
  stride = strides.front();
  return true;
}

std::optional<int64_t> getStaticIndex(mlir::OpFoldResult value) {
  auto attr = mlir::dyn_cast<mlir::Attribute>(value);
  if (!attr)
    return std::nullopt;
  return mlir::cast<mlir::IntegerAttr>(attr).getInt();
}

mlir::OpFoldResult affineIndex(mlir::Location loc, int64_t base, int64_t scale,
                               mlir::OpFoldResult input,
                               mlir::OpBuilder &builder) {
  if (std::optional<int64_t> value = getStaticIndex(input))
    return builder.getIndexAttr(base + scale * *value);

  mlir::Value result = materializeIndex(loc, input, builder);
  if (scale != 1) {
    mlir::Value scaleValue =
        builder.create<mlir::arith::ConstantIndexOp>(loc, scale);
    result = builder.create<mlir::arith::MulIOp>(loc, result, scaleValue);
  }
  if (base != 0) {
    mlir::Value baseValue =
        builder.create<mlir::arith::ConstantIndexOp>(loc, base);
    result = builder.create<mlir::arith::AddIOp>(loc, baseValue, result);
  }
  return result;
}

mlir::OpFoldResult scaledIndex(mlir::Location loc, int64_t scale,
                               mlir::OpFoldResult input,
                               mlir::OpBuilder &builder) {
  if (std::optional<int64_t> value = getStaticIndex(input))
    return builder.getIndexAttr(scale * *value);
  if (scale == 1)
    return input;
  mlir::Value scaleValue =
      builder.create<mlir::arith::ConstantIndexOp>(loc, scale);
  return builder
      .create<mlir::arith::MulIOp>(loc, materializeIndex(loc, input, builder),
                                   scaleValue)
      .getResult();
}

mlir::FailureOr<mlir::Value> expandRank1Subview(mlir::memref::SubViewOp subview,
                                                mlir::OpBuilder &builder) {
  auto sourceType =
      mlir::dyn_cast<mlir::MemRefType>(subview.getSource().getType());
  auto resultType = mlir::dyn_cast<mlir::MemRefType>(subview.getType());
  if (!sourceType || !resultType || sourceType.getRank() != 1 ||
      resultType.getRank() != 1 || subview.getDroppedDims().any())
    return subview.emitError("unsupported memref.subview before LLVM lowering");

  mlir::Location loc = subview.getLoc();

  mlir::OpFoldResult subOffset = subview.getMixedOffsets().front();
  mlir::OpFoldResult subSize = subview.getMixedSizes().front();
  mlir::OpFoldResult subStride = subview.getMixedStrides().front();

  int64_t staticOffset = 0;
  int64_t staticStride = 1;
  if (getStaticRank1Layout(sourceType, staticOffset, staticStride)) {
    mlir::OpFoldResult finalOffset =
        affineIndex(loc, staticOffset, staticStride, subOffset, builder);
    mlir::OpFoldResult finalStride =
        scaledIndex(loc, staticStride, subStride, builder);

    llvm::SmallVector<mlir::OpFoldResult, 1> sizes{subSize};
    llvm::SmallVector<mlir::OpFoldResult, 1> strides{finalStride};
    llvm::SmallVector<mlir::NamedAttribute, 4> attrs(
        subview->getDiscardableAttrs().begin(),
        subview->getDiscardableAttrs().end());
    auto cast = builder.create<mlir::memref::ReinterpretCastOp>(
        loc, resultType, subview.getSource(), finalOffset, sizes, strides,
        attrs);
    return cast.getResult();
  }

  auto metadata = builder.create<mlir::memref::ExtractStridedMetadataOp>(
      loc, subview.getSource());
  copyDiscardableAttrs(subview.getOperation(), metadata.getOperation());

  mlir::Value sourceOffset = metadata.getOffset();
  mlir::Value sourceStride = metadata.getStrides().front();
  mlir::Value scaledOffset = mulIndex(loc, sourceStride, subOffset, builder);
  mlir::Value finalOffset =
      builder.create<mlir::arith::AddIOp>(loc, sourceOffset, scaledOffset);
  mlir::Value finalStride = mulIndex(loc, sourceStride, subStride, builder);

  llvm::SmallVector<mlir::OpFoldResult, 1> sizes{subSize};
  llvm::SmallVector<mlir::OpFoldResult, 1> strides{finalStride};
  llvm::SmallVector<mlir::NamedAttribute, 4> attrs(
      subview->getDiscardableAttrs().begin(),
      subview->getDiscardableAttrs().end());
  auto cast = builder.create<mlir::memref::ReinterpretCastOp>(
      loc, resultType, metadata.getBaseBuffer(), finalOffset, sizes, strides,
      attrs);
  return cast.getResult();
}

bool containsPyRuntimeType(mlir::Type type,
                           llvm::DenseMap<mlir::Type, bool> &cache) {
  if (auto cached = cache.find(type); cached != cache.end())
    return cached->second;

  bool contains = false;
  if (isPyType(type))
    contains = true;
  if (auto asyncValue = mlir::dyn_cast<mlir::async::ValueType>(type))
    contains =
        contains || containsPyRuntimeType(asyncValue.getValueType(), cache);
  if (auto memref = mlir::dyn_cast<mlir::MemRefType>(type))
    contains =
        contains || containsPyRuntimeType(memref.getElementType(), cache);
  if (auto llvmStruct = mlir::dyn_cast<mlir::LLVM::LLVMStructType>(type)) {
    if (!llvmStruct.isOpaque()) {
      contains = contains ||
                 llvm::any_of(llvmStruct.getBody(), [&](mlir::Type element) {
                   return containsPyRuntimeType(element, cache);
                 });
    }
  }
  if (auto llvmArray = mlir::dyn_cast<mlir::LLVM::LLVMArrayType>(type))
    contains =
        contains || containsPyRuntimeType(llvmArray.getElementType(), cache);
  cache[type] = contains;
  return contains;
}

mlir::LogicalResult verifyPyTypesLowered(mlir::ModuleOp module) {
  llvm::DenseMap<mlir::Type, bool> typeCache;
  mlir::Operation *offender = nullptr;
  mlir::Type offenderType;
  bool offenderIsOperand = false;
  unsigned offenderIndex = 0;
  module.walk([&](mlir::Operation *op) -> mlir::WalkResult {
    for (auto [index, type] : llvm::enumerate(op->getOperandTypes())) {
      if (!containsPyRuntimeType(type, typeCache))
        continue;
      offender = op;
      offenderType = type;
      offenderIsOperand = true;
      offenderIndex = index;
      return mlir::WalkResult::interrupt();
    }
    for (auto [index, type] : llvm::enumerate(op->getResultTypes())) {
      if (!containsPyRuntimeType(type, typeCache))
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
  if (auto cast = mlir::dyn_cast<mlir::UnrealizedConversionCastOp>(offender)) {
    diagnostic << "; operands = " << cast.getOperandTypes()
               << ", results = " << cast.getResultTypes()
               << ", op = " << *cast.getOperation();
    diagnostic << ", users = [";
    bool firstUser = true;
    for (mlir::Operation *user : cast.getResult(0).getUsers()) {
      if (!firstUser)
        diagnostic << ", ";
      firstUser = false;
      diagnostic << *user;
    }
    diagnostic << "]";
  }
  if (auto parentFunc = offender->getParentOfType<mlir::func::FuncOp>())
    diagnostic << ", parent func = " << parentFunc.getName();
  if (auto parentLLVMFunc = offender->getParentOfType<mlir::LLVM::LLVMFuncOp>())
    diagnostic << ", parent llvm func = " << parentLLVMFunc.getName();
  return mlir::failure();
}

mlir::LogicalResult expandMemRefMetadata(mlir::ModuleOp module) {
  // Expand only the subview operation that the header/payload object ABI uses.
  // Running the generic expand-strided-metadata rewrite greedily also invokes
  // region DCE; at this point class carriers may already contain temporary
  // LLVM descriptor bridges, and the generic DCE path is too broad for that
  // mixed-level IR. The remaining memref metadata ops are lowered by the
  // memref-to-LLVM conversion.
  llvm::SmallVector<mlir::memref::SubViewOp> subviews;
  module.walk(
      [&](mlir::memref::SubViewOp subview) { subviews.push_back(subview); });

  for (mlir::memref::SubViewOp subview : subviews) {
    if (!subview || !subview->getBlock())
      continue;
    mlir::OpBuilder builder(subview);
    mlir::FailureOr<mlir::Value> replacement =
        expandRank1Subview(subview, builder);
    if (mlir::failed(replacement))
      return mlir::failure();
    subview.replaceAllUsesWith(*replacement);
    subview.erase();
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
      // The runtime library fabricates descriptor values (tagged longs)
      // through explicitly marked bridge casts; only unmarked casts indicate
      // a frontend type-inference leak.
      if (cast->hasAttr("ly.runtime.descriptor_bridge"))
        return mlir::WalkResult::advance();
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

void collectReferencedSymbols(mlir::Attribute attr,
                              llvm::DenseSet<llvm::StringRef> &referenced) {
  if (auto symbol = mlir::dyn_cast<mlir::SymbolRefAttr>(attr)) {
    referenced.insert(symbol.getRootReference());
    for (mlir::FlatSymbolRefAttr nested : symbol.getNestedReferences())
      referenced.insert(nested.getValue());
    return;
  }
  if (auto array = mlir::dyn_cast<mlir::ArrayAttr>(attr)) {
    for (mlir::Attribute nested : array)
      collectReferencedSymbols(nested, referenced);
    return;
  }
  if (auto dict = mlir::dyn_cast<mlir::DictionaryAttr>(attr)) {
    for (mlir::NamedAttribute nested : dict)
      collectReferencedSymbols(nested.getValue(), referenced);
  }
}

bool eraseUnusedPrivateFuncSymbols(mlir::ModuleOp module) {
  auto hasOnlyEffectArg = [](mlir::func::FuncOp func, llvm::StringRef attrName,
                             unsigned argIndex) {
    auto attr = func->getAttrOfType<mlir::ArrayAttr>(attrName);
    if (!attr || attr.size() != 1)
      return false;
    auto integer = mlir::dyn_cast<mlir::IntegerAttr>(attr[0]);
    return integer && integer.getInt() == static_cast<int64_t>(argIndex);
  };

  auto hasObjectHeaderArg = [](mlir::func::FuncOp func, unsigned argIndex) {
    return static_cast<unsigned>(func.getNumArguments()) > argIndex &&
           static_cast<bool>(func.getArgAttr(
               argIndex, OwnershipContractAttrs::kObjectHeader));
  };

  auto keepForLateLowering = [](mlir::func::FuncOp func) {
    return func->hasAttr(OwnershipContractAttrs::kObjectReleaseToZero);
  };

  auto keepRuntimeObjectEffectRoot = [&](mlir::func::FuncOp func) {
    if (!hasObjectHeaderArg(func, /*argIndex=*/0))
      return false;
    return hasOnlyEffectArg(func, OwnershipContractAttrs::kRetainArgs,
                            /*argIndex=*/0) ||
           hasOnlyEffectArg(func, OwnershipContractAttrs::kReleaseArgs,
                            /*argIndex=*/0);
  };

  bool changed = false;
  for (;;) {
    llvm::DenseSet<llvm::StringRef> referenced;
    module.walk([&](mlir::Operation *op) {
      for (mlir::NamedAttribute attr : op->getAttrs()) {
        if (mlir::isa<mlir::SymbolOpInterface>(op) &&
            attr.getName() == mlir::SymbolTable::getSymbolAttrName())
          continue;
        collectReferencedSymbols(attr.getValue(), referenced);
      }
    });

    llvm::SmallVector<mlir::Operation *> unused;
    module.walk([&](mlir::func::FuncOp func) {
      if (func.getSymName() == "main" || !func.isPrivate())
        return;
      if (keepForLateLowering(func) || keepRuntimeObjectEffectRoot(func))
        return;
      if (referenced.contains(func.getSymName()))
        return;
      unused.push_back(func.getOperation());
    });
    module.walk([&](mlir::LLVM::LLVMFuncOp func) {
      auto symbol = llvm::cast<mlir::SymbolOpInterface>(func.getOperation());
      if (func.getSymName() == "main" ||
          symbol.getVisibility() != mlir::SymbolTable::Visibility::Private)
        return;
      if (referenced.contains(func.getSymName()))
        return;
      unused.push_back(func.getOperation());
    });
    if (unused.empty())
      return changed;
    for (mlir::Operation *op : unused)
      op->erase();
    changed = true;
  }
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
  target.addDynamicallyLegalOp<mlir::LLVM::LLVMFuncOp>(
      [](mlir::LLVM::LLVMFuncOp) { return true; });
  target.addLegalOp<mlir::ModuleOp>();
  target.addLegalOp<mlir::cf::AssertOp>();
  target.addLegalOp<mlir::UnrealizedConversionCastOp>();
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

    auto materializationFilter = [](mlir::Diagnostic &) -> mlir::LogicalResult {
      return mlir::failure();
    };

    if (mlir::failed(verifyFrontendTypesFinalized(module))) {
      signalPassFailure();
      return;
    }

    // Phase 1a: Function definition conversion (py.callable.func -> func.func).
    // Keep py.return conversion separate; moving a py.callable.func body and
    // replacing nested returns in the same delayed conversion can invalidate
    // commit ordering for multi-result lowered values.

    auto runFuncConversion = [&]() -> mlir::LogicalResult {
      return lowering::runtime::conversion::runPartial(
          module, ctx,
          [&](mlir::RewritePatternSet &patterns) {
            lowering::runtime::conversion::function::definition::Patterns::
                populate(typeConverter, patterns);
          },
          [&](mlir::ConversionTarget &target) {
            lowering::runtime::conversion::configurePyTarget(target);
            target.addIllegalOp<CallableFuncOp>();
          },
          materializationFilter);
    };

    if (mlir::failed(timedRuntimePhase("func-definition", runFuncConversion))) {
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

    if (mlir::failed(
            timedRuntimePhase("return-conversion", runReturnConversion))) {
      signalPassFailure();
      return;
    }

    {
      RuntimePerfScope perf("lowering.runtime-lowering.post-return-helpers");
      while (lowering::runtime::cleanup::voidPyReturns(module))
        ;
      lowering::runtime::helpers::synthesizeLocalSelf(module);
      lowering::runtime::helpers::synthesizePublishedBorrow(module);
    }

    // Phase 2: Callable object conversion (py.callable.object -> references)

    auto runCallableObjectConversion = [&]() -> mlir::LogicalResult {
      return lowering::runtime::conversion::runPartial(
          module, ctx,
          [&](mlir::RewritePatternSet &patterns) {
            lowering::runtime::conversion::function::objects::Patterns::
                populate(typeConverter, patterns);
          },
          [&](mlir::ConversionTarget &target) {
            lowering::runtime::conversion::configurePyTarget(target);
            target.addIllegalOp<CallableObjectOp, MakeFunctionOp>();
          },
          materializationFilter);
    };

    if (mlir::failed(timedRuntimePhase("callable-object",
                                       runCallableObjectConversion))) {
      signalPassFailure();
      return;
    }

    {
      RuntimePerfScope perf("lowering.runtime-lowering.fold-int-constants");
      optimizer::scalar::foldIntConstants(module);
    }

    // Phase 3: Call conversion (py.call -> calls)

    auto runCallConversion = [&]() -> mlir::LogicalResult {
      return lowering::runtime::conversion::runPartial(
          module, ctx,
          [&](mlir::RewritePatternSet &patterns) {
            lowering::runtime::conversion::call::Patterns::populate(
                typeConverter, patterns);
          },
          [&](mlir::ConversionTarget &target) {
            lowering::runtime::conversion::configurePyTarget(target);
            target.addIllegalOp<EnterOp, ExitOp, AEnterOp, AExitOp, SendOp,
                                ThrowOp, CloseOp, ASendOp, AThrowOp, ACloseOp,
                                IterOp, NextOp, CallOp, InvokeOp>();
          },
          materializationFilter);
    };

    if (mlir::failed(timedRuntimePhase("call-conversion", runCallConversion))) {
      signalPassFailure();
      return;
    }
    // Apply pre-lowering optimizations
    {
      RuntimePerfScope perf("lowering.runtime-lowering.pre-lowering-1");
      optimizer::pipeline::preLowering(module);
    }
    {
      RuntimePerfScope perf("lowering.runtime-lowering.verify-ownership-1");
      if (mlir::failed(verifyOwnership(module))) {
        signalPassFailure();
        return;
      }
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

    if (mlir::failed(timedRuntimePhase("try-conversion", runTryConversion))) {
      signalPassFailure();
      return;
    }
    {
      RuntimePerfScope perf("lowering.runtime-lowering.pre-lowering-2");
      optimizer::pipeline::preLowering(module);
    }
    {
      RuntimePerfScope perf("lowering.runtime-lowering.verify-ownership-2");
      if (mlir::failed(verifyOwnership(module))) {
        signalPassFailure();
        return;
      }
    }
    {
      RuntimePerfScope perf("lowering.runtime-lowering.unreachable-blocks-1");
      lowering::runtime::cleanup::unreachableBlocks(module);
    }

    // Runtime MLIR signatures are part of the ABI contract. Import them before
    // value conversion so RuntimeAPI can adapt operands by signature instead of
    // classifying callees by name. A second call after value conversion
    // materializes contracts on newly emitted calls.
    {
      RuntimePerfScope perf(
          "lowering.runtime-lowering.embed-runtime-before-value");
      if (mlir::failed(runtime_library::embedObjectModules(module))) {
        signalPassFailure();
        return;
      }
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

    if (mlir::failed(
            timedRuntimePhase("value-conversion", runValueConversion))) {
      signalPassFailure();
      return;
    }

    // Apply Python-value cleanup at the boundary where object-level scalar
    // concepts have just been lowered to runtime/LLVM calls.
    {
      RuntimePerfScope perf("lowering.runtime-lowering.post-value-lowering");
      optimizer::pipeline::postValueLowering(module);
    }
    {
      RuntimePerfScope perf(
          "lowering.runtime-lowering.embed-runtime-after-value");
      if (mlir::failed(runtime_library::embedObjectModules(module))) {
        signalPassFailure();
        return;
      }
    }
    {
      RuntimePerfScope perf("lowering.runtime-lowering.pre-generic-symbol-dce");
      eraseUnusedPrivateFuncSymbols(module);
    }

    // A Python-typed unwind payload reaching LLVM invoke lowering is an ABI
    // leak. Do not hide it behind llvm.ptr; the exception lowering must choose
    // the descriptor shape before this boundary.
    bool hasUnloweredUnwindPyArg = false;
    {
      RuntimePerfScope perf("lowering.runtime-lowering.verify-unwind-py-args");
      module.walk([&](mlir::LLVM::InvokeOp invoke) {
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
    }
    if (hasUnloweredUnwindPyArg) {
      signalPassFailure();
      return;
    }

    // Some passes may drop llvm.personality; restore it for landingpads.
    {
      RuntimePerfScope perf("lowering.runtime-lowering.ensure-personalities");
      lowering::runtime::eh::ensureFuncPersonalities(module);
    }

    // Async result storage outlives the coroutine frame. Ensure container
    // payload descriptors were promoted before memref-to-LLVM erases
    // stack-vs-heap allocation provenance.
    {
      RuntimePerfScope perf("lowering.runtime-lowering.verify-async-payloads");
      if (mlir::failed(
              lowering::runtime::async::verifyReturnPayloads(module))) {
        signalPassFailure();
        return;
      }
    }

    // Phase 5: Freeze the lowered Py ABI. After this boundary all !py.* value
    // shapes must be represented by their fixed memref/LLVM forms.

    {
      RuntimePerfScope perf("lowering.runtime-lowering.py-cast-cleanup");
      while (lowering::runtime::cleanup::pyBridgeCasts(module))
        ;
      while (lowering::runtime::cleanup::pyMultiCasts(module))
        ;
    }
    {
      RuntimePerfScope perf(
          "lowering.runtime-lowering.verify-py-types-lowered");
      if (mlir::failed(verifyPyTypesLowered(module))) {
        signalPassFailure();
        return;
      }
    }

    // Bufferize tensor/linalg values before generic LLVM conversion. Leaving
    // tensor ops legal during arith-to-LLVM forces index casts back from i64,
    // which hides an ABI mismatch until the final cleanup verifier.
    {
      RuntimePerfScope perf("lowering.runtime-lowering.tensor-lowering");
      if (mlir::failed(lowerTensorProgramLevelOps(module, ctx))) {
        signalPassFailure();
        return;
      }
    }

    // Header/payload object ABI uses memref.subview to expose the common
    // header. Expand it before memref-to-LLVM, where SubViewOp is intentionally
    // illegal.
    {
      RuntimePerfScope perf("lowering.runtime-lowering.expand-memref-metadata");
      if (mlir::failed(expandMemRefMetadata(module))) {
        signalPassFailure();
        return;
      }
    }

    auto runGenericLLVMConversion = [&](bool full) -> mlir::LogicalResult {
      LoweredSafetyContracts safetyContracts;
      collectLoweredSafetyContracts(module, typeConverter, safetyContracts);

      mlir::RewritePatternSet patterns(ctx);
      populateGenericLLVMConversion(typeConverter, patterns);
      mlir::ConversionTarget target(*ctx);
      configureGenericLLVMTarget(target);
      mlir::LogicalResult converted =
          full ? applyFullConversion(module, target, std::move(patterns))
               : applyPartialConversion(module, target, std::move(patterns));
      if (mlir::failed(converted))
        return mlir::failure();
      if (mlir::failed(preserveLoweredSafetyContracts(module, safetyContracts)))
        return mlir::failure();
      return mlir::success();
    };

    // Phase 6: Convert func.func to llvm.func before final EH materialization.
    {
      RuntimePerfScope perf(
          "lowering.runtime-lowering.generic-llvm-conversion");
      if (mlir::failed(runGenericLLVMConversion(/*full=*/false))) {
        signalPassFailure();
        return;
      }
    }

    {
      RuntimePerfScope perf("lowering.runtime-lowering.final-cleanups");
      lowering::runtime::cleanup::finalBoundary(module);
    }
    {
      RuntimePerfScope perf("lowering.runtime-lowering.verify-no-casts");
      if (mlir::failed(lowering::verifyNoUnrealizedCasts(
              module, "runtime lowering boundary"))) {
        signalPassFailure();
        return;
      }
    }

    // Finalize unwind blocks with landingpad in LLVM world.
    {
      RuntimePerfScope perf("lowering.runtime-lowering.finalize-unwind");
      lowering::runtime::eh::finalizeUnwindBlocks(module);
    }

    // Insert a top-level exception handler wrapper for `main`.
    {
      RuntimePerfScope perf("lowering.runtime-lowering.wrap-main");
      lowering::runtime::eh::wrapTopLevelMain(module);
    }

    {
      RuntimePerfScope perf("lowering.runtime-lowering.verify-llvm-ownership");
      if (mlir::failed(verifyLLVMCallOwnership(module))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createRuntimeLoweringPass() {
  return std::make_unique<RuntimeLoweringPass>();
}

} // namespace py
