// This file implements the main RuntimeLoweringPass which orchestrates the
// complete lowering pipeline from Py dialect to LLVM dialect. It coordinates
// the various conversion phases:
//   1. Function conversion (py.func -> func.func)
//   2. Function object conversion (py.func_object -> references)
//   3. Call conversion (py.call_vector -> runtime calls or direct calls)
//   4. Value conversion (py.* ops -> LLVM ops via runtime calls)
//
// Individual lowering patterns are implemented in separate files:
//   - PyFunc/Lowering.cpp: Function and calling convention patterns
//   - PyCall/*.cpp: Call operation patterns
//   - PyValue*.cpp: Value, class, scalar, and typed container patterns
//   - PyRefCount/Lowering.cpp: Reference counting patterns
//
// Optimizations are implemented under Optimizer/.

#include "Common/LoweringUtils.h"
#include "Common/RuntimeSupport.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"

#include "PyDialect.h.inc"

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

using namespace mlir;

namespace py {

namespace optimizer {
void applyStaticMakeFunctionDefaults(ModuleOp module);
} // namespace optimizer
namespace {

static constexpr int64_t kTypedListHeaderSlots = 4;
static constexpr int64_t kTypedTupleHeaderSlots = 3;

// Utility functions

/// Replaces UnrealizedConversionCastOp involving py.* types with
/// CastIdentityOp. Returns true if any replacements were made.
static bool replaceUnrealizedCastsWithIdentity(Operation *container) {
  SmallVector<UnrealizedConversionCastOp> pending;

  container->walk([&](UnrealizedConversionCastOp cast) {
    if (cast->getNumOperands() != 1 || cast->getNumResults() != 1)
      return;
    Type inputType = cast->getOperand(0).getType();
    Type resultType = cast->getResultTypes().front();
    bool involvesPy = isPyType(inputType) || isPyType(resultType);
    if (!involvesPy)
      return;
    pending.push_back(cast);
  });

  for (auto cast : pending) {
    OpBuilder builder(cast);
    auto identity = builder.create<CastIdentityOp>(
        cast.getLoc(), cast->getResultTypes().front(), cast->getOperand(0));
    cast->getResult(0).replaceAllUsesWith(identity.getResult());
    cast->erase();
  }

  return !pending.empty();
}

static bool foldPyMultiUnrealizedCasts(Operation *container) {
  SmallVector<UnrealizedConversionCastOp> casts;
  container->walk(
      [&](UnrealizedConversionCastOp cast) { casts.push_back(cast); });

  bool changed = false;
  for (auto cast : casts) {
    if (!cast)
      continue;
    if (cast->getNumOperands() == 1 && isPyType(cast.getOperand(0).getType()) &&
        cast->getNumResults() > 1) {
      auto source =
          cast.getOperand(0).getDefiningOp<UnrealizedConversionCastOp>();
      if (!source || source->getNumResults() != 1 ||
          !isPyType(source.getResult(0).getType()) ||
          source->getNumOperands() != cast->getNumResults())
        continue;
      bool compatible = true;
      for (auto [operand, result] :
           llvm::zip(source.getOperands(), cast.getResults())) {
        if (operand.getType() != result.getType()) {
          compatible = false;
          break;
        }
      }
      if (!compatible)
        continue;
      for (auto [result, operand] :
           llvm::zip(cast.getResults(), source.getOperands()))
        result.replaceAllUsesWith(operand);
      cast.erase();
      if (source->use_empty())
        source.erase();
      changed = true;
      continue;
    }
  }
  return changed;
}

static Value materializeDescriptorFromMemRefBridge(Value value,
                                                   OpBuilder &builder) {
  auto defining = value.getDefiningOp<UnrealizedConversionCastOp>();
  if (!defining)
    return {};
  auto result = dyn_cast<OpResult>(value);
  if (!result)
    return {};

  if (defining->getNumOperands() == 1 &&
      isa<LLVM::LLVMStructType>(defining.getOperand(0).getType()) &&
      defining->getNumResults() > 1) {
    auto sourceType =
        dyn_cast<LLVM::LLVMStructType>(defining.getOperand(0).getType());
    if (!sourceType || sourceType.isOpaque() ||
        result.getResultNumber() >= sourceType.getBody().size())
      return {};
    builder.setInsertionPoint(defining);
    return builder.create<LLVM::ExtractValueOp>(
        defining.getLoc(), sourceType.getBody()[result.getResultNumber()],
        defining.getOperand(0),
        builder.getDenseI64ArrayAttr(
            {static_cast<int64_t>(result.getResultNumber())}));
  }

  if (defining->getNumOperands() == 1 &&
      isa<LLVM::LLVMStructType>(defining.getOperand(0).getType()) &&
      defining->getNumResults() == 1) {
    return defining.getOperand(0);
  }

  if (defining->getNumOperands() == 1 && defining->getNumResults() > 1) {
    Value nested = defining.getOperand(0);
    auto nestedResult = dyn_cast<OpResult>(nested);
    auto nestedCast = nested.getDefiningOp<UnrealizedConversionCastOp>();
    if (!nestedResult || !nestedCast || nestedCast->getNumOperands() != 1 ||
        !isa<LLVM::LLVMStructType>(nestedCast.getOperand(0).getType()))
      return {};
    auto sourceType =
        dyn_cast<LLVM::LLVMStructType>(nestedCast.getOperand(0).getType());
    if (!sourceType || sourceType.isOpaque() ||
        result.getResultNumber() >= sourceType.getBody().size())
      return {};
    builder.setInsertionPoint(nestedCast);
    return builder.create<LLVM::ExtractValueOp>(
        defining.getLoc(), sourceType.getBody()[result.getResultNumber()],
        nestedCast.getOperand(0),
        builder.getDenseI64ArrayAttr(
            {static_cast<int64_t>(result.getResultNumber())}));
  }

  return {};
}

static bool cleanupMemRefDescriptorUnrealizedCasts(Operation *container) {
  SmallVector<UnrealizedConversionCastOp> casts;
  container->walk(
      [&](UnrealizedConversionCastOp cast) { casts.push_back(cast); });

  bool changed = false;
  for (auto cast : casts) {
    if (!cast || cast->use_empty())
      continue;

    OpBuilder builder(cast);
    if (cast->getNumResults() == 1 &&
        isa<LLVM::LLVMStructType>(cast.getResult(0).getType())) {
      auto resultType =
          mlir::cast<LLVM::LLVMStructType>(cast.getResult(0).getType());
      if (!resultType.isOpaque() &&
          resultType.getBody().size() == cast->getNumOperands()) {
        Value aggregate =
            builder.create<LLVM::UndefOp>(cast.getLoc(), resultType);
        bool ok = true;
        for (auto [index, operand] : llvm::enumerate(cast.getOperands())) {
          Value descriptor =
              materializeDescriptorFromMemRefBridge(operand, builder);
          if (!descriptor) {
            ok = false;
            break;
          }
          builder.setInsertionPoint(cast);
          aggregate = builder.create<LLVM::InsertValueOp>(
              cast.getLoc(), resultType, aggregate, descriptor,
              builder.getDenseI64ArrayAttr({static_cast<int64_t>(index)}));
        }
        if (!ok)
          continue;
        cast.getResult(0).replaceAllUsesWith(aggregate);
        cast.erase();
        changed = true;
        continue;
      }
    }

    if (cast->getNumOperands() == 1 && cast->getNumResults() == 1 &&
        isa<LLVM::LLVMStructType>(cast.getResult(0).getType())) {
      Value descriptor =
          materializeDescriptorFromMemRefBridge(cast.getOperand(0), builder);
      if (!descriptor)
        continue;
      cast.getResult(0).replaceAllUsesWith(descriptor);
      cast.erase();
      changed = true;
    }
  }

  SmallVector<UnrealizedConversionCastOp> deadCasts;
  container->walk([&](UnrealizedConversionCastOp cast) {
    if (cast->use_empty())
      deadCasts.push_back(cast);
  });
  for (auto cast : deadCasts) {
    cast.erase();
    changed = true;
  }
  return changed;
}

/// Replaces CastIdentityOp with their forwarded operand. Returns true if any
/// replacements were made.
static bool cleanupCastIdentityOps(Operation *container) {
  SmallVector<CastIdentityOp> pending;

  container->walk([&](CastIdentityOp cast) { pending.push_back(cast); });

  for (CastIdentityOp cast : pending) {
    if (cast.getInput())
      cast.getResult().replaceAllUsesWith(cast.getInput());
    cast.erase();
  }

  return !pending.empty();
}

/// Replaces remaining py.return in zero-result func.func helpers with
/// func.return. These can remain after function conversion because helper
/// func.func ops are dynamically legal during the conversion phase.
static bool cleanupVoidPyReturns(Operation *container) {
  SmallVector<ReturnOp> pending;
  container->walk([&](ReturnOp ret) {
    auto parentFunc = ret->getParentOfType<func::FuncOp>();
    if (parentFunc && parentFunc.getFunctionType().getNumResults() == 0)
      pending.push_back(ret);
  });

  for (ReturnOp ret : pending) {
    OpBuilder builder(ret);
    NoneOp noneOp = nullptr;
    if (ret.getNumOperands() == 1)
      noneOp = ret.getOperand(0).getDefiningOp<NoneOp>();
    builder.create<func::ReturnOp>(ret.getLoc());
    ret.erase();
    if (noneOp && noneOp->use_empty())
      noneOp.erase();
  }

  return !pending.empty();
}

static func::FuncOp lookupFuncFromSymbolAttr(ModuleOp module,
                                             SymbolRefAttr ref) {
  if (!ref)
    return nullptr;
  StringRef symbol = ref.getLeafReference().empty()
                         ? ref.getRootReference().getValue()
                         : ref.getLeafReference().getValue();
  return module.lookupSymbol<func::FuncOp>(symbol);
}

static func::FuncOp clonePrivateHelper(ModuleOp module, func::FuncOp base,
                                       StringRef newName) {
  if (!base)
    return nullptr;
  if (auto existing = module.lookupSymbol<func::FuncOp>(newName))
    return existing;

  auto cloned = cast<func::FuncOp>(base->clone());
  cloned.setName(newName);
  cloned.setVisibility(SymbolTable::Visibility::Private);
  SymbolTable symbolTable(module);
  symbolTable.insert(cloned);
  return cloned;
}

static LLVM::LLVMFuncOp getOrInsertLLVMRuntimeFunc(ModuleOp module,
                                                   StringRef name,
                                                   Type resultType,
                                                   ArrayRef<Type> argTypes) {
  if (auto fn = module.lookupSymbol<LLVM::LLVMFuncOp>(name))
    return fn;
  OpBuilder builder(module.getBody(), module.getBody()->begin());
  auto fnType = LLVM::LLVMFunctionType::get(resultType, argTypes, false);
  return builder.create<LLVM::LLVMFuncOp>(module.getLoc(), name, fnType);
}

static void retainBorrowedEntryBlockReturns(ModuleOp module) {
  MLIRContext *ctx = module.getContext();
  auto ptrType = LLVM::LLVMPointerType::get(ctx);
  auto voidType = LLVM::LLVMVoidType::get(ctx);
  auto incRef = getOrInsertLLVMRuntimeFunc(module, RuntimeSymbols::kIncRef,
                                           voidType, {ptrType});
  auto incRefRef = SymbolRefAttr::get(ctx, incRef.getName());

  module.walk([&](func::FuncOp func) {
    if (func.getBody().empty())
      return;
    Block &entry = func.getBody().front();
    SmallVector<func::ReturnOp> returns;
    func.walk([&](func::ReturnOp ret) { returns.push_back(ret); });
    for (func::ReturnOp ret : returns) {
      if (ret.getNumOperands() != 1)
        continue;
      Value returned = ret.getOperand(0);
      if (returned.getType() != ptrType)
        continue;
      auto blockArg = dyn_cast<BlockArgument>(returned);
      if (!blockArg || blockArg.getOwner() != &entry)
        continue;
      OpBuilder builder(ret);
      builder.create<LLVM::CallOp>(ret.getLoc(), TypeRange{}, incRefRef,
                                   ValueRange{returned});
    }
  });
}

static void synthesizeLocalSelfHelpers(ModuleOp module) {
  SmallVector<func::FuncOp> candidates;
  module.walk([&](func::FuncOp func) {
    if (!func->hasAttr("llvm.emit_c_interface"))
      return;
    if (!func->hasAttr("lython.self_receiver_arg0"))
      return;
    candidates.push_back(func);
  });

  MLIRContext *ctx = module.getContext();
  for (func::FuncOp func : candidates) {
    if (func->hasAttr("lython.void_helper") &&
        !func->hasAttr("lython.local_self_helper")) {
      auto baseHelper = lookupFuncFromSymbolAttr(
          module, func->getAttrOfType<SymbolRefAttr>("lython.void_helper"));
      if (auto localHelper = clonePrivateHelper(
              module, baseHelper, (func.getName() + "$local").str())) {
        localHelper->setAttr("lython.local_self_arg0", UnitAttr::get(ctx));
        func->setAttr("lython.local_self_helper",
                      SymbolRefAttr::get(ctx, localHelper.getName()));
      }
    }

    if (!func->hasAttr("lython.local_self_helper") &&
        !func->hasAttr("lython.void_helper") &&
        !func->hasAttr("lython.class_return_helper")) {
      if (auto localHelper = clonePrivateHelper(
              module, func, (func.getName() + "$local").str())) {
        localHelper->setAttr("lython.local_self_arg0", UnitAttr::get(ctx));
        func->setAttr("lython.local_self_helper",
                      SymbolRefAttr::get(ctx, localHelper.getName()));
      }
    }

    if (func->hasAttr("lython.class_return_helper")) {
      auto baseHelper = lookupFuncFromSymbolAttr(
          module,
          func->getAttrOfType<SymbolRefAttr>("lython.class_return_helper"));
      if (baseHelper && !baseHelper->hasAttr("lython.local_self_helper")) {
        if (auto localHelper = clonePrivateHelper(
                module, baseHelper, (baseHelper.getName() + "$local").str())) {
          localHelper->setAttr("lython.local_self_arg0", UnitAttr::get(ctx));
          baseHelper->setAttr("lython.local_self_helper",
                              SymbolRefAttr::get(ctx, localHelper.getName()));
        }
      }
    }
  }
}

static std::string getPublishedBorrowHelperAttrName(unsigned argIndex) {
  return "lython.published_borrow_helper_arg" + std::to_string(argIndex);
}

static bool arrayAttrContainsIndex(ArrayAttr attr, unsigned index) {
  if (!attr)
    return false;
  for (Attribute element : attr) {
    auto intAttr = dyn_cast<IntegerAttr>(element);
    if (!intAttr)
      continue;
    if (intAttr.getInt() == static_cast<int64_t>(index))
      return true;
  }
  return false;
}

static Value stripIdentityCasts(Value value) {
  while (auto identity = value.getDefiningOp<CastIdentityOp>())
    value = identity.getInput();
  return value;
}

static bool specializePublishedBorrowArg(func::FuncOp func, unsigned argIndex) {
  if (!func || func.getBody().empty())
    return false;

  Block &entry = func.getBody().front();
  if (argIndex >= entry.getNumArguments())
    return false;

  Value borrowedArg = entry.getArgument(argIndex);
  SmallVector<PublishOp> publishes;
  func.walk([&](PublishOp publish) {
    if (stripIdentityCasts(publish.getInput()) == borrowedArg)
      publishes.push_back(publish);
  });

  if (publishes.empty())
    return false;

  bool changed = false;
  for (PublishOp publish : publishes) {
    Value forwardedValue = publish.getInput();
    SmallVector<DecRefOp> decRefs;
    SmallVector<OpOperand *> forwardedUses;
    for (OpOperand &use : publish.getResult().getUses()) {
      if (auto decRef = dyn_cast<DecRefOp>(use.getOwner())) {
        decRefs.push_back(decRef);
        continue;
      }
      forwardedUses.push_back(&use);
    }

    for (OpOperand *use : forwardedUses)
      use->set(forwardedValue);
    for (DecRefOp decRef : decRefs)
      decRef.erase();
    publish.erase();
    changed = true;
  }

  return changed;
}

static void synthesizePublishedBorrowHelpers(ModuleOp module) {
  MLIRContext *ctx = module.getContext();
  bool changed = false;
  do {
    changed = false;
    SmallVector<func::FuncOp> candidates;
    module.walk([&](func::FuncOp func) {
      if (func->hasAttr("lython.publishes_args"))
        candidates.push_back(func);
    });

    for (func::FuncOp func : candidates) {
      auto publishesArgs =
          func->getAttrOfType<ArrayAttr>("lython.publishes_args");
      if (!publishesArgs)
        continue;

      for (unsigned argIndex = 0; argIndex < func.getNumArguments();
           ++argIndex) {
        if (!arrayAttrContainsIndex(publishesArgs, argIndex))
          continue;

        std::string attrName = getPublishedBorrowHelperAttrName(argIndex);
        if (func->hasAttr(attrName))
          continue;

        std::string helperName =
            (func.getName() + "$published_arg" + std::to_string(argIndex))
                .str();
        auto helper = clonePrivateHelper(module, func, helperName);
        if (!helper)
          continue;
        if (!specializePublishedBorrowArg(helper, argIndex)) {
          helper.erase();
          continue;
        }

        func->setAttr(attrName, SymbolRefAttr::get(ctx, helper.getName()));
        changed = true;
      }
    }
  } while (changed);
}

// Type cast lowering patterns

struct UnrealizedCastLowering
    : public OpConversionPattern<UnrealizedConversionCastOp> {
  UnrealizedCastLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<UnrealizedConversionCastOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(UnrealizedConversionCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "expected single operand");

    Type inputType = op.getOperandTypes().front();
    Type resultType = op.getResultTypes().front();
    if (!isPyType(inputType) && !isPyType(resultType))
      return rewriter.notifyMatchFailure(
          op, "unrelated to py.* types, keep default handling");

    auto identity = rewriter.create<CastIdentityOp>(
        op.getLoc(), resultType, adaptor.getOperands().front());
    rewriter.replaceOp(op, identity.getResult());
    return success();
  }
};

static StringRef getTypedListReprHelperName(ListType listType) {
  Type elementType = listType.getElementType();
  if (isa<BoolType>(elementType))
    return "LyListBool_Repr";
  if (isa<FloatType>(elementType))
    return "LyListF64Bits_Repr";
  if (isa<ClassType>(elementType))
    return "LyListPtr_Repr";
  return "LyListI64_Repr";
}

static StringRef getTypedTupleReprHelperName(TupleType tupleType) {
  auto elements = tupleType.getElementTypes();
  if (!elements.empty() &&
      llvm::all_of(elements, [](Type type) { return isa<BoolType>(type); }))
    return "LyTupleBool_Repr";
  if (!elements.empty() &&
      llvm::all_of(elements, [](Type type) { return isa<FloatType>(type); }))
    return "LyTupleF64Bits_Repr";
  if (!elements.empty() &&
      llvm::all_of(elements, [](Type type) { return isa<ClassType>(type); }))
    return "LyTuplePtr_Repr";
  return "LyTupleI64_Repr";
}

static func::FuncOp
getOrInsertTypedListReprFunc(Location loc, ModuleOp module, ListType listType,
                             Type memrefType, ArrayRef<Type> extraArgTypes,
                             Type resultType, OpBuilder &builder) {
  StringRef name = getTypedListReprHelperName(listType);
  if (auto fn = module.lookupSymbol<func::FuncOp>(name))
    return fn;

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());
  SmallVector<Type> inputTypes;
  inputTypes.push_back(memrefType);
  inputTypes.append(extraArgTypes.begin(), extraArgTypes.end());
  auto fnType =
      FunctionType::get(module.getContext(), inputTypes, {resultType});
  auto fn = builder.create<func::FuncOp>(loc, name, fnType);
  fn->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
  fn.setVisibility(SymbolTable::Visibility::Private);
  return fn;
}

static func::FuncOp getOrInsertTypedTupleReprFunc(
    Location loc, ModuleOp module, TupleType tupleType, Type memrefType,
    ArrayRef<Type> extraArgTypes, Type resultType, OpBuilder &builder) {
  StringRef name = getTypedTupleReprHelperName(tupleType);
  if (auto fn = module.lookupSymbol<func::FuncOp>(name))
    return fn;

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());
  SmallVector<Type> inputTypes;
  inputTypes.push_back(memrefType);
  inputTypes.append(extraArgTypes.begin(), extraArgTypes.end());
  auto fnType =
      FunctionType::get(module.getContext(), inputTypes, {resultType});
  auto fn = builder.create<func::FuncOp>(loc, name, fnType);
  fn->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
  fn.setVisibility(SymbolTable::Visibility::Private);
  return fn;
}

static std::string getStaticClassReprCallbackName(ClassType classType) {
  return ("__ly_class_repr_" + classType.getClassName()).str();
}

static Value widenTypedContainerReprSlot(Location loc, Value value,
                                         OpBuilder &builder) {
  Type type = value.getType();
  Type i64Type = builder.getI64Type();
  if (type == i64Type)
    return value;
  if (type == builder.getF64Type())
    return builder.create<arith::BitcastOp>(loc, i64Type, value);
  if (auto intType = dyn_cast<IntegerType>(type)) {
    if (intType.getWidth() < 64)
      return builder.create<arith::ExtUIOp>(loc, i64Type, value);
    if (intType.getWidth() > 64)
      return builder.create<arith::TruncIOp>(loc, i64Type, value);
  }
  return {};
}

static std::optional<unsigned> getTypedContainerRefcountSlot(Type type) {
  if (isa<ListType>(type))
    return 3u;
  if (isa<TupleType>(type))
    return 2u;
  if (isa<DictType>(type))
    return 4u;
  return std::nullopt;
}

static bool isBorrowedEntryBlockDescriptor(Value header) {
  if (auto cast = header.getDefiningOp<UnrealizedConversionCastOp>()) {
    for (Value operand : cast.getOperands())
      if (isBorrowedEntryBlockDescriptor(operand))
        return true;
    return false;
  }
  auto arg = dyn_cast<BlockArgument>(header);
  if (!arg)
    return false;
  Block *owner = arg.getOwner();
  if (!owner)
    return false;
  auto func = dyn_cast_or_null<func::FuncOp>(owner->getParentOp());
  return func && owner == &func.getBody().front();
}

static LogicalResult dropTypedContainerAfterRepr(Location loc, Type logicalType,
                                                 ValueRange descriptor,
                                                 OpBuilder &builder) {
  auto refcountSlot = getTypedContainerRefcountSlot(logicalType);
  if (!refcountSlot || descriptor.empty())
    return failure();

  Value header = descriptor.front();
  if (isBorrowedEntryBlockDescriptor(header))
    return success();
  if (header.getDefiningOp<memref::AllocaOp>())
    return success();

  Value slot =
      createIndexConstant(loc, builder, static_cast<int64_t>(*refcountSlot));
  Value refcount = builder.create<memref::LoadOp>(loc, header, slot);
  Value zero = createI64Constant(loc, builder, 0);
  Value isManaged = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
                                                  refcount, zero);
  auto managed =
      builder.create<scf::IfOp>(loc, isManaged, /*withElseRegion=*/false);
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(managed.thenBlock());
    Value next = builder.create<arith::SubIOp>(
        loc, refcount, createI64Constant(loc, builder, 1));
    builder.create<memref::StoreOp>(loc, next, header, slot);
    Value shouldFree = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, next, zero);
    auto free =
        builder.create<scf::IfOp>(loc, shouldFree, /*withElseRegion=*/false);
    {
      OpBuilder::InsertionGuard freeGuard(builder);
      builder.setInsertionPointToStart(free.thenBlock());
      for (Value memref : descriptor)
        builder.create<memref::DeallocOp>(loc, memref);
    }
  }
  return success();
}

/// py.upcast forwards the operand since all py.* types share the same
/// runtime representation (PyObject*).
struct UpcastLowering : public OpConversionPattern<UpcastOp> {
  UpcastLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<UpcastOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(UpcastOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (adaptor.getInput().empty())
      return failure();
    if (auto listType = dyn_cast<ListType>(op.getInput().getType())) {
      if (isCompilerOwnedMemRefListType(listType)) {
        ModuleOp module = op->getParentOfType<ModuleOp>();
        if (!module)
          return failure();
        auto *converter =
            static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
        auto unrankedType =
            UnrankedMemRefType::get(rewriter.getI64Type(), /*memorySpace=*/0);
        Value reprMemref;
        if (adaptor.getInput().size() == 2) {
          Value header = adaptor.getInput().front();
          Value items = adaptor.getInput().back();
          auto itemsType = dyn_cast<MemRefType>(items.getType());
          if (!itemsType)
            return rewriter.notifyMatchFailure(
                op, "list repr requires memref item storage");

          Value sizeI64 = rewriter.create<memref::LoadOp>(
              op.getLoc(), header,
              createIndexConstant(op.getLoc(), rewriter, 0));
          Value size = rewriter.create<arith::IndexCastOp>(
              op.getLoc(), rewriter.getIndexType(), sizeI64);
          Value headerSlots =
              createIndexConstant(op.getLoc(), rewriter, kTypedListHeaderSlots);
          Value flatSize =
              rewriter.create<arith::AddIOp>(op.getLoc(), size, headerSlots);
          auto flatType =
              MemRefType::get({ShapedType::kDynamic}, rewriter.getI64Type());
          Value flat = rewriter.create<memref::AllocaOp>(op.getLoc(), flatType,
                                                         ValueRange{flatSize});
          for (int64_t slot = 0; slot < kTypedListHeaderSlots; ++slot) {
            Value index = createIndexConstant(op.getLoc(), rewriter, slot);
            Value value =
                rewriter.create<memref::LoadOp>(op.getLoc(), header, index);
            rewriter.create<memref::StoreOp>(op.getLoc(), value, flat, index);
          }

          Value zero = createIndexConstant(op.getLoc(), rewriter, 0);
          Value one = createIndexConstant(op.getLoc(), rewriter, 1);
          auto loop = rewriter.create<scf::ForOp>(op.getLoc(), zero, size, one);
          {
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(loop.getBody());
            Value iv = loop.getInductionVar();
            Value value =
                rewriter.create<memref::LoadOp>(op.getLoc(), items, iv);
            value = widenTypedContainerReprSlot(op.getLoc(), value, rewriter);
            if (!value)
              return failure();
            Value destIndex =
                rewriter.create<arith::AddIOp>(op.getLoc(), iv, headerSlots);
            rewriter.create<memref::StoreOp>(op.getLoc(), value, flat,
                                             destIndex);
          }
          reprMemref = flat;
        } else if (adaptor.getInput().size() == 1) {
          reprMemref = adaptor.getInput().front();
        } else {
          return failure();
        }
        Value unranked = rewriter.create<memref::CastOp>(
            op.getLoc(), unrankedType, reprMemref);
        SmallVector<Type> extraTypes;
        SmallVector<Value> extraOperands;
        if (auto classType = dyn_cast<ClassType>(listType.getElementType())) {
          auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
          extraTypes.push_back(ptrType);
          std::string reprName = getStaticClassReprCallbackName(classType);
          extraOperands.push_back(rewriter.create<LLVM::AddressOfOp>(
              op.getLoc(), ptrType,
              StringAttr::get(module.getContext(), reprName)));
        }
        auto reprFunc = getOrInsertTypedListReprFunc(
            op.getLoc(), module, listType, unrankedType, extraTypes,
            converter->getPyObjectPtrType(), rewriter);
        SmallVector<Value> operands;
        operands.push_back(unranked);
        operands.append(extraOperands.begin(), extraOperands.end());
        auto call =
            rewriter.create<func::CallOp>(op.getLoc(), reprFunc, operands);
        if (failed(dropTypedContainerAfterRepr(op.getLoc(), listType,
                                               adaptor.getInput(), rewriter)))
          return failure();
        rewriter.replaceOp(op, call.getResults());
        return success();
      }
    }
    if (auto tupleType = dyn_cast<TupleType>(op.getInput().getType())) {
      if (isCompilerOwnedMemRefTupleType(tupleType)) {
        ModuleOp module = op->getParentOfType<ModuleOp>();
        if (!module)
          return failure();
        auto *converter =
            static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
        auto unrankedType =
            UnrankedMemRefType::get(rewriter.getI64Type(), /*memorySpace=*/0);
        if (adaptor.getInput().size() != 2)
          return failure();

        Value header = adaptor.getInput().front();
        Value items = adaptor.getInput().back();
        auto itemsType = dyn_cast<MemRefType>(items.getType());
        if (!itemsType)
          return rewriter.notifyMatchFailure(
              op, "tuple repr requires memref item storage");

        Value sizeI64 = rewriter.create<memref::LoadOp>(
            op.getLoc(), header, createIndexConstant(op.getLoc(), rewriter, 0));
        Value size = rewriter.create<arith::IndexCastOp>(
            op.getLoc(), rewriter.getIndexType(), sizeI64);
        Value headerSlots =
            createIndexConstant(op.getLoc(), rewriter, kTypedTupleHeaderSlots);
        Value flatSize =
            rewriter.create<arith::AddIOp>(op.getLoc(), size, headerSlots);
        auto flatType =
            MemRefType::get({ShapedType::kDynamic}, rewriter.getI64Type());
        Value flat = rewriter.create<memref::AllocaOp>(op.getLoc(), flatType,
                                                       ValueRange{flatSize});
        for (int64_t slot = 0; slot < kTypedTupleHeaderSlots; ++slot) {
          Value index = createIndexConstant(op.getLoc(), rewriter, slot);
          Value value =
              rewriter.create<memref::LoadOp>(op.getLoc(), header, index);
          rewriter.create<memref::StoreOp>(op.getLoc(), value, flat, index);
        }

        Value zero = createIndexConstant(op.getLoc(), rewriter, 0);
        Value one = createIndexConstant(op.getLoc(), rewriter, 1);
        auto loop = rewriter.create<scf::ForOp>(op.getLoc(), zero, size, one);
        {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointToStart(loop.getBody());
          Value iv = loop.getInductionVar();
          Value value = rewriter.create<memref::LoadOp>(op.getLoc(), items, iv);
          value = widenTypedContainerReprSlot(op.getLoc(), value, rewriter);
          if (!value)
            return failure();
          Value destIndex =
              rewriter.create<arith::AddIOp>(op.getLoc(), iv, headerSlots);
          rewriter.create<memref::StoreOp>(op.getLoc(), value, flat, destIndex);
        }

        Value unranked =
            rewriter.create<memref::CastOp>(op.getLoc(), unrankedType, flat);
        SmallVector<Type> extraTypes;
        SmallVector<Value> extraOperands;
        auto elements = tupleType.getElementTypes();
        if (!elements.empty() && llvm::all_of(elements, [](Type type) {
              return isa<ClassType>(type);
            })) {
          auto classType = cast<ClassType>(elements.front());
          auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
          extraTypes.push_back(ptrType);
          std::string reprName = getStaticClassReprCallbackName(classType);
          extraOperands.push_back(rewriter.create<LLVM::AddressOfOp>(
              op.getLoc(), ptrType,
              StringAttr::get(module.getContext(), reprName)));
        }
        auto reprFunc = getOrInsertTypedTupleReprFunc(
            op.getLoc(), module, tupleType, unrankedType, extraTypes,
            converter->getPyObjectPtrType(), rewriter);
        SmallVector<Value> operands;
        operands.push_back(unranked);
        operands.append(extraOperands.begin(), extraOperands.end());
        auto call =
            rewriter.create<func::CallOp>(op.getLoc(), reprFunc, operands);
        if (failed(dropTypedContainerAfterRepr(op.getLoc(), tupleType,
                                               adaptor.getInput(), rewriter)))
          return failure();
        rewriter.replaceOp(op, call.getResults());
        return success();
      }
    }
    rewriter.replaceOp(op, adaptor.getInput().front());
    return success();
  }
};

struct CastIdentityLowering : public OpConversionPattern<CastIdentityOp> {
  CastIdentityLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<CastIdentityOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(CastIdentityOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (adaptor.getInput())
      rewriter.replaceOp(op, adaptor.getInput());
    else
      rewriter.eraseOp(op);
    return success();
  }
};

static LogicalResult runIterativePartialConversion(
    ModuleOp module, MLIRContext *ctx,
    llvm::function_ref<void(RewritePatternSet &)> populatePatterns,
    llvm::function_ref<void(ConversionTarget &)> configureTarget,
    llvm::function_ref<LogicalResult(Diagnostic &)> materializationFilter) {
  while (true) {
    RewritePatternSet patterns(ctx);
    populatePatterns(patterns);

    ConversionTarget target(*ctx);
    configureTarget(target);

    ScopedDiagnosticHandler diagHandler(ctx, materializationFilter);
    auto result = applyPartialConversion(module, target, std::move(patterns));
    if (succeeded(result))
      return success();
    if (!replaceUnrealizedCastsWithIdentity(module))
      return failure();
  }
}

static void populateFunctionPhasePatterns(PyLLVMTypeConverter &typeConverter,
                                          RewritePatternSet &patterns) {
  auto *ctx = patterns.getContext();
  populatePyFuncLoweringPatterns(typeConverter, patterns);
  patterns.add<UnrealizedCastLowering>(typeConverter, ctx);
}

static void configurePyDialectConversionTarget(ConversionTarget &target) {
  target.addLegalDialect<py::PyDialect>();
  target.addLegalOp<ModuleOp>();
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
}

static void populateCallPhasePatterns(PyLLVMTypeConverter &typeConverter,
                                      RewritePatternSet &patterns) {
  auto *ctx = patterns.getContext();
  populatePyCallLoweringPatterns(typeConverter, patterns);
  patterns.add<UnrealizedCastLowering>(typeConverter, ctx);
}

static void populateValuePhasePatterns(PyLLVMTypeConverter &typeConverter,
                                       RewritePatternSet &patterns) {
  auto *ctx = patterns.getContext();
  populatePyValueLoweringPatterns(typeConverter, patterns);
  populatePyListValueLoweringPatterns(typeConverter, patterns);
  populatePyNumberValueLoweringPatterns(typeConverter, patterns);
  populatePyClassValueLoweringPatterns(typeConverter, patterns);
  populatePyTupleValueLoweringPatterns(typeConverter, patterns);
  populatePyDictValueLoweringPatterns(typeConverter, patterns);
  populatePyRefCountLoweringPatterns(typeConverter, patterns);
  patterns.add<UpcastLowering, CastIdentityLowering, UnrealizedCastLowering>(
      typeConverter, ctx);
}

static void configureValueConversionTarget(ConversionTarget &target) {
  target.addLegalDialect<
      LLVM::LLVMDialect, func::FuncDialect, arith::ArithDialect,
      tensor::TensorDialect, linalg::LinalgDialect, memref::MemRefDialect,
      scf::SCFDialect, bufferization::BufferizationDialect>();
  target.addLegalOp<ModuleOp>();
  target.addLegalOp<UnrealizedConversionCastOp>();
  target.addIllegalOp<
      StrConstantOp, IntConstantOp, FloatConstantOp, TupleEmptyOp,
      TupleCreateOp, DictEmptyOp, DictInsertOp, DictGetOp, ListNewOp,
      ListAppendOp, ListRemoveOp, ListGetOp, NoneOp, FuncObjectOp,
      MakeFunctionOp, NumAddOp, NumSubOp, NumLtOp, NumLeOp, NumGtOp, NumGeOp,
      NumEqOp, NumNeOp, CastToPrimOp, CastFromPrimOp, CastIdentityOp, UpcastOp,
      IncRefOp, DecRefOp, ClassNewOp, ClassPromoteOp, PublishOp, AttrGetOp,
      AttrSetOp, ClassOp, ExceptionNullOp, TracebackNullOp, LocationCurrentOp,
      ExceptionNewOp, RaiseOp, RaiseCurrentOp, TryOp, TryYieldOp, ExceptYieldOp,
      FinallyYieldOp, ExceptMatchOp>();
}

// RuntimeLoweringPass: Main pipeline orchestration

struct RuntimeLoweringPass
    : public PassWrapper<RuntimeLoweringPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RuntimeLoweringPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, func::FuncDialect,
                    cf::ControlFlowDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();
    ctx->loadDialect<bufferization::BufferizationDialect,
                     memref::MemRefDialect>();
    PyLLVMTypeConverter typeConverter(ctx);
    bool dumpLowering = static_cast<bool>(
        llvm::sys::Process::GetEnv("LYTHON_DUMP_LOWERING_IR"));

    auto materializationFilter = [&](Diagnostic &diag) -> LogicalResult {
      std::string message;
      llvm::raw_string_ostream os(message);
      diag.print(os);
      os.flush();
      if (message.find("unresolved materialization") != std::string::npos)
        return success();
      return failure();
    };

    optimizer::applyStaticMakeFunctionDefaults(module);

    // Phase 1: Function conversion (py.func/py.return -> func.func/func.return)

    auto runFuncConversion = [&]() -> LogicalResult {
      return runIterativePartialConversion(
          module, ctx,
          [&](RewritePatternSet &patterns) {
            populateFunctionPhasePatterns(typeConverter, patterns);
          },
          [&](ConversionTarget &target) {
            configurePyDialectConversionTarget(target);
            target.addIllegalOp<FuncOp, ReturnOp>();
          },
          materializationFilter);
    };

    if (failed(runFuncConversion())) {
      signalPassFailure();
      return;
    }
    replaceUnrealizedCastsWithIdentity(module);
    while (cleanupVoidPyReturns(module))
      ;
    retainBorrowedEntryBlockReturns(module);
    synthesizeLocalSelfHelpers(module);
    synthesizePublishedBorrowHelpers(module);

    if (dumpLowering) {
      llvm::errs() << "[After func conversion]\n";
      module.dump();
    }

    optimizer::applyStaticMakeFunctionDefaults(module);

    // Phase 2: Function object conversion (py.func_object -> references)

    auto runFuncObjectConversion = [&]() -> LogicalResult {
      return runIterativePartialConversion(
          module, ctx,
          [&](RewritePatternSet &patterns) {
            populateFunctionPhasePatterns(typeConverter, patterns);
          },
          [&](ConversionTarget &target) {
            configurePyDialectConversionTarget(target);
            target.addIllegalOp<FuncObjectOp, MakeFunctionOp>();
          },
          materializationFilter);
    };

    if (failed(runFuncObjectConversion())) {
      signalPassFailure();
      return;
    }
    replaceUnrealizedCastsWithIdentity(module);

    if (dumpLowering) {
      llvm::errs() << "[After func object conversion]\n";
      module.dump();
    }

    // Phase 3: Call conversion (py.call_vector/py.call -> calls)

    auto runCallConversion = [&]() -> LogicalResult {
      return runIterativePartialConversion(
          module, ctx,
          [&](RewritePatternSet &patterns) {
            populateCallPhasePatterns(typeConverter, patterns);
          },
          [&](ConversionTarget &target) {
            configurePyDialectConversionTarget(target);
            target.addIllegalOp<CallVectorOp, CallOp, InvokeOp>();
          },
          materializationFilter);
    };

    if (failed(runCallConversion())) {
      signalPassFailure();
      return;
    }
    replaceUnrealizedCastsWithIdentity(module);

    // Apply pre-lowering optimizations
    runPreLoweringOptimizations(module);

    if (dumpLowering) {
      llvm::errs() << "[After call conversion]\n";
      module.dump();
    }

    // Phase 4: Value conversion (py.* ops -> LLVM ops)

    auto runValueConversion = [&]() -> LogicalResult {
      return runIterativePartialConversion(
          module, ctx,
          [&](RewritePatternSet &patterns) {
            populateValuePhasePatterns(typeConverter, patterns);
          },
          configureValueConversionTarget, materializationFilter);
    };

    if (failed(runValueConversion())) {
      signalPassFailure();
      return;
    }

    // Apply post-lowering optimizations
    runPostLoweringOptimizations(module);

    // Normalize invoke unwind block arguments to LLVM pointer types.
    {
      auto pyObject = LLVM::LLVMPointerType::get(ctx);
      module.walk([&](LLVM::InvokeOp invoke) {
        Block *unwind = invoke.getUnwindDest();
        if (!unwind)
          return;
        for (BlockArgument arg :
             llvm::make_early_inc_range(unwind->getArguments())) {
          if (!isPyType(arg.getType()))
            continue;
          arg.setType(pyObject);
          for (auto &use : llvm::make_early_inc_range(arg.getUses())) {
            auto *owner = use.getOwner();
            auto cast = dyn_cast<UnrealizedConversionCastOp>(owner);
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
    auto ensurePersonalityForLandingpads = [&]() {
      auto personality = FlatSymbolRefAttr::get(ctx, "__gxx_personality_v0");
      module.walk([&](func::FuncOp func) {
        if (func->hasAttr("llvm.personality"))
          return;
        bool hasLandingpad = false;
        func.walk([&](LLVM::LandingpadOp) { hasLandingpad = true; });
        if (hasLandingpad)
          func->setAttr("llvm.personality", personality);
      });
    };
    ensurePersonalityForLandingpads();

    if (dumpLowering) {
      llvm::errs() << "[After optimizations]\n";
      module.dump();
    }

    // Phase 5: Cleanup remaining casts

    replaceUnrealizedCastsWithIdentity(module);
    while (foldPyMultiUnrealizedCasts(module))
      ;
    while (cleanupCastIdentityOps(module))
      ;

    // Phase 6: Convert func.func to llvm.func before final EH materialization.
    {
      if (dumpLowering) {
        llvm::errs() << "[Before func-to-llvm conversion]\n";
        module.dump();
      }
      RewritePatternSet patterns(ctx);
      populateSCFToControlFlowConversionPatterns(patterns);
      cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
      arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
      populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
      populateFuncToLLVMConversionPatterns(typeConverter, patterns);
      ConversionTarget target(*ctx);
      target.addLegalDialect<LLVM::LLVMDialect>();
      target.addIllegalDialect<func::FuncDialect>();
      target.addIllegalDialect<cf::ControlFlowDialect>();
      target.addIllegalDialect<scf::SCFDialect>();
      target.addIllegalDialect<memref::MemRefDialect>();
      target.addIllegalDialect<arith::ArithDialect>();
      target.addLegalOp<ModuleOp>();
      target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
      if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
        signalPassFailure();
        return;
      }
      if (dumpLowering) {
        llvm::errs() << "[After func-to-llvm conversion]\n";
        module.dump();
      }
    }

    while (foldPyMultiUnrealizedCasts(module))
      ;
    while (cleanupMemRefDescriptorUnrealizedCasts(module))
      ;

    // Finalize unwind blocks with landingpad in LLVM world.
    {
      auto pyObject = LLVM::LLVMPointerType::get(ctx);
      auto getOrCreatePersonality = [&]() {
        StringRef name = "__gxx_personality_v0";
        if (auto fn = module.lookupSymbol<LLVM::LLVMFuncOp>(name))
          return fn;
        OpBuilder builder(module.getBody(), module.getBody()->begin());
        auto fnType =
            LLVM::LLVMFunctionType::get(builder.getI32Type(), {}, true);
        return builder.create<LLVM::LLVMFuncOp>(module.getLoc(), name, fnType);
      };
      auto getOrCreateRuntimeFunc = [&](StringRef name, Type resultType,
                                        ArrayRef<Type> argTypes) {
        if (auto fn = module.lookupSymbol<LLVM::LLVMFuncOp>(name))
          return fn;
        OpBuilder builder(module.getBody(), module.getBody()->begin());
        auto fnType = LLVM::LLVMFunctionType::get(resultType, argTypes, false);
        return builder.create<LLVM::LLVMFuncOp>(module.getLoc(), name, fnType);
      };
      auto emitRuntimeCall = [&](OpBuilder &builder, Location loc,
                                 StringRef name, Type resultType,
                                 ValueRange operands) {
        SmallVector<Type> argTypes;
        argTypes.reserve(operands.size());
        for (Value operand : operands)
          argTypes.push_back(operand.getType());
        Type actualResult =
            resultType ? resultType : LLVM::LLVMVoidType::get(ctx);
        auto callee = getOrCreateRuntimeFunc(name, actualResult, argTypes);
        auto symbol = SymbolRefAttr::get(ctx, callee.getName());
        SmallVector<Type> results;
        if (!isa<LLVM::LLVMVoidType>(actualResult))
          results.push_back(actualResult);
        return builder.create<LLVM::CallOp>(loc, results, symbol, operands);
      };

      auto personality = getOrCreatePersonality();
      auto personalityRef = FlatSymbolRefAttr::get(ctx, personality.getName());

      module.walk([&](LLVM::InvokeOp invoke) {
        auto func = invoke->getParentOfType<LLVM::LLVMFuncOp>();
        if (func && !func->hasAttr("personality"))
          func->setAttr("personality", personalityRef);

        Block *unwind = invoke.getUnwindDest();
        if (!unwind)
          return;
        unwind->clear();
        OpBuilder builder(ctx);
        builder.setInsertionPointToStart(unwind);
        auto lpType = LLVM::LLVMStructType::getLiteral(
            ctx, ArrayRef<Type>{pyObject, builder.getI32Type()});
        auto lp = builder.create<LLVM::LandingpadOp>(
            invoke.getLoc(), lpType, builder.getUnitAttr(), ValueRange{});
        Value raw = builder.create<LLVM::ExtractValueOp>(
            invoke.getLoc(), pyObject, lp.getRes(),
            builder.getDenseI64ArrayAttr({0}));
        emitRuntimeCall(builder, invoke.getLoc(), RuntimeSymbols::kEHCapture,
                        pyObject, ValueRange{raw});
        builder.create<LLVM::ResumeOp>(invoke.getLoc(), lp.getRes());
      });
      if (dumpLowering) {
        llvm::errs() << "[After EH finalize]\n";
        module.dump();
      }
    }

    // Insert a top-level exception handler wrapper for `main`.
    {
      auto mainFunc = module.lookupSymbol<LLVM::LLVMFuncOp>("main");
      if (mainFunc) {
        auto fnType = mainFunc.getFunctionType();
        if (fnType.getNumParams() == 0 &&
            !isa<LLVM::LLVMVoidType>(fnType.getReturnType())) {
          StringRef implName = "__lython_main";
          if (!module.lookupSymbol<LLVM::LLVMFuncOp>(implName)) {
            mainFunc.setName(implName);

            auto getOrCreateRuntimeFunc = [&](StringRef name, Type resultType,
                                              ArrayRef<Type> argTypes) {
              if (auto fn = module.lookupSymbol<LLVM::LLVMFuncOp>(name))
                return fn;
              OpBuilder builder(module.getBody(), module.getBody()->begin());
              auto fnType =
                  LLVM::LLVMFunctionType::get(resultType, argTypes, false);
              return builder.create<LLVM::LLVMFuncOp>(module.getLoc(), name,
                                                      fnType);
            };
            auto emitRuntimeCall = [&](OpBuilder &builder, Location loc,
                                       StringRef name, Type resultType,
                                       ValueRange operands) {
              SmallVector<Type> argTypes;
              argTypes.reserve(operands.size());
              for (Value operand : operands)
                argTypes.push_back(operand.getType());
              Type actualResult =
                  resultType ? resultType : LLVM::LLVMVoidType::get(ctx);
              auto callee =
                  getOrCreateRuntimeFunc(name, actualResult, argTypes);
              auto symbol = SymbolRefAttr::get(ctx, callee.getName());
              SmallVector<Type> results;
              if (!isa<LLVM::LLVMVoidType>(actualResult))
                results.push_back(actualResult);
              return builder.create<LLVM::CallOp>(loc, results, symbol,
                                                  operands);
            };

            auto getOrCreatePersonality = [&]() {
              StringRef name = "__gxx_personality_v0";
              if (auto fn = module.lookupSymbol<LLVM::LLVMFuncOp>(name))
                return fn;
              OpBuilder builder(module.getBody(), module.getBody()->begin());
              auto fnType =
                  LLVM::LLVMFunctionType::get(builder.getI32Type(), {}, true);
              return builder.create<LLVM::LLVMFuncOp>(module.getLoc(), name,
                                                      fnType);
            };

            OpBuilder builder(ctx);
            builder.setInsertionPointToEnd(module.getBody());
            auto wrapper = builder.create<LLVM::LLVMFuncOp>(module.getLoc(),
                                                            "main", fnType);

            auto personality = getOrCreatePersonality();
            auto personalityRef =
                FlatSymbolRefAttr::get(ctx, personality.getName());
            wrapper->setAttr("personality", personalityRef);

            Block *entry = wrapper.addEntryBlock(builder);
            Block *normal = builder.createBlock(&wrapper.getBody());
            Block *unwind = builder.createBlock(&wrapper.getBody());

            builder.setInsertionPointToStart(entry);
            auto ptrType = LLVM::LLVMPointerType::get(ctx);
            Value catchAll =
                builder.create<LLVM::ZeroOp>(module.getLoc(), ptrType);
            auto invoke = builder.create<LLVM::InvokeOp>(
                module.getLoc(), fnType.getReturnType(),
                FlatSymbolRefAttr::get(ctx, implName), ValueRange{}, normal,
                ValueRange{}, unwind, ValueRange{});

            builder.setInsertionPointToStart(normal);
            builder.create<LLVM::ReturnOp>(module.getLoc(), invoke.getResult());

            builder.setInsertionPointToStart(unwind);
            auto i32Type = builder.getI32Type();
            auto lpType = LLVM::LLVMStructType::getLiteral(
                ctx, ArrayRef<Type>{ptrType, i32Type});
            auto lp = builder.create<LLVM::LandingpadOp>(
                module.getLoc(), lpType, builder.getUnitAttr(),
                ValueRange{catchAll});
            Value raw = builder.create<LLVM::ExtractValueOp>(
                module.getLoc(), ptrType, lp.getRes(),
                builder.getDenseI64ArrayAttr({0}));
            auto captured = emitRuntimeCall(builder, module.getLoc(),
                                            RuntimeSymbols::kEHCapture, ptrType,
                                            ValueRange{raw});
            auto reported = emitRuntimeCall(
                builder, module.getLoc(), RuntimeSymbols::kEHReportUnhandled,
                i32Type, ValueRange{captured.getResult()});
            builder.create<LLVM::ReturnOp>(module.getLoc(),
                                           reported.getResult());
          }
        }
      }
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createRuntimeLoweringPass() {
  return std::make_unique<RuntimeLoweringPass>();
}

} // namespace py
