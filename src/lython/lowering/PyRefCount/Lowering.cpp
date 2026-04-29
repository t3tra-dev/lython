#include "Common/LoweringUtils.h"
#include "Common/RuntimeSupport.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/STLExtras.h"

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

using namespace mlir;

namespace py {
namespace {

static LogicalResult emitStaticClassRefCountCall(
    Operation *op, Value object, ClassType classType, ModuleOp module,
    ConversionPatternRewriter &rewriter, StringRef suffix) {
  auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
  auto helper = getOrInsertLLVMFunc(
      op->getLoc(), module, rewriter, getClassHelperName(classType, suffix),
      LLVM::LLVMVoidType::get(rewriter.getContext()), {ptrType});
  auto helperRef = SymbolRefAttr::get(module.getContext(), helper.getName());
  rewriter.create<LLVM::CallOp>(op->getLoc(), TypeRange{}, helperRef,
                                ValueRange{object});
  return success();
}

static std::optional<unsigned> getContainerRefcountSlot(Type type) {
  if (isa<ListType>(type))
    return 3u;
  if (isa<TupleType>(type))
    return 2u;
  if (isa<DictType>(type))
    return 4u;
  return std::nullopt;
}

static Value loadContainerRefcount(Location loc, Value header,
                                   unsigned refcountSlot,
                                   ConversionPatternRewriter &rewriter) {
  return rewriter.create<memref::LoadOp>(
      loc, header,
      createIndexConstant(loc, rewriter, static_cast<int64_t>(refcountSlot)));
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
  auto parentFunc = dyn_cast_or_null<func::FuncOp>(owner->getParentOp());
  return parentFunc && owner == &parentFunc.getBody().front();
}

static void storeContainerRefcount(Location loc, Value header,
                                   unsigned refcountSlot, Value value,
                                   ConversionPatternRewriter &rewriter) {
  rewriter.create<memref::StoreOp>(
      loc, value, header,
      createIndexConstant(loc, rewriter, static_cast<int64_t>(refcountSlot)));
}

static LogicalResult
emitCompilerOwnedContainerIncRef(Location loc, Type logicalType,
                                 ValueRange descriptor,
                                 ConversionPatternRewriter &rewriter) {
  auto refcountSlot = getContainerRefcountSlot(logicalType);
  if (!refcountSlot || descriptor.empty())
    return failure();

  Value header = descriptor.front();
  if (header.getDefiningOp<memref::AllocaOp>())
    return success();

  Value refcount = loadContainerRefcount(loc, header, *refcountSlot, rewriter);
  Value zero = createI64Constant(loc, rewriter, 0);
  Value isManaged = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::ne, refcount, zero);
  auto ifOp =
      rewriter.create<scf::IfOp>(loc, isManaged, /*withElseRegion=*/false);
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(ifOp.thenBlock());
    Value next = rewriter.create<arith::AddIOp>(
        loc, refcount, createI64Constant(loc, rewriter, 1));
    storeContainerRefcount(loc, header, *refcountSlot, next, rewriter);
  }
  return success();
}

static LogicalResult
emitCompilerOwnedContainerDecRef(Location loc, Type logicalType,
                                 ValueRange descriptor,
                                 ConversionPatternRewriter &rewriter) {
  auto refcountSlot = getContainerRefcountSlot(logicalType);
  if (!refcountSlot || descriptor.empty())
    return failure();

  Value header = descriptor.front();
  if (isBorrowedEntryBlockDescriptor(header))
    return success();
  if (header.getDefiningOp<memref::AllocaOp>())
    return success();

  Value refcount = loadContainerRefcount(loc, header, *refcountSlot, rewriter);
  Value zero = createI64Constant(loc, rewriter, 0);
  Value isManaged = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::ne, refcount, zero);
  auto managed =
      rewriter.create<scf::IfOp>(loc, isManaged, /*withElseRegion=*/false);
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(managed.thenBlock());
    Value next = rewriter.create<arith::SubIOp>(
        loc, refcount, createI64Constant(loc, rewriter, 1));
    storeContainerRefcount(loc, header, *refcountSlot, next, rewriter);
    Value shouldFree = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, next, zero);
    auto free =
        rewriter.create<scf::IfOp>(loc, shouldFree, /*withElseRegion=*/false);
    {
      OpBuilder::InsertionGuard freeGuard(rewriter);
      rewriter.setInsertionPointToStart(free.thenBlock());
      for (Value memref : descriptor)
        rewriter.create<memref::DeallocOp>(loc, memref);
    }
  }
  return success();
}

struct IncRefLowering : public OpConversionPattern<IncRefOp> {
  IncRefLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<IncRefOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(IncRefOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.setInsertionPoint(op);
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    if (isCompilerOwnedMemRefListType(op.getObject().getType()) ||
        isCompilerOwnedMemRefDictType(op.getObject().getType()) ||
        isCompilerOwnedMemRefTupleType(op.getObject().getType())) {
      if (failed(emitCompilerOwnedContainerIncRef(
              op.getLoc(), op.getObject().getType(), adaptor.getObject(),
              rewriter)))
        return failure();
      rewriter.eraseOp(op);
      return success();
    }
    if (isa<FuncType, PrimFuncType>(op.getObject().getType())) {
      rewriter.eraseOp(op);
      return success();
    }
    if (auto classType = dyn_cast<ClassType>(op.getObject().getType())) {
      (void)typeConverter;
      if (adaptor.getObject().empty())
        return failure();
      if (failed(emitStaticClassRefCountCall(op, adaptor.getObject().front(),
                                             classType, module, rewriter,
                                             "incref")))
        return failure();
      rewriter.eraseOp(op);
      return success();
    }
    RuntimeAPI runtime(module, rewriter, *typeConverter);

    runtime.call(op.getLoc(), RuntimeSymbols::kIncRef, /*resultType=*/nullptr,
                 adaptor.getObject());
    rewriter.eraseOp(op);
    return success();
  }
};

struct DecRefLowering : public OpConversionPattern<DecRefOp> {
  DecRefLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<DecRefOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(DecRefOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.setInsertionPoint(op);
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    if (isCompilerOwnedMemRefListType(op.getObject().getType()) ||
        isCompilerOwnedMemRefDictType(op.getObject().getType()) ||
        isCompilerOwnedMemRefTupleType(op.getObject().getType())) {
      if (failed(emitCompilerOwnedContainerDecRef(
              op.getLoc(), op.getObject().getType(), adaptor.getObject(),
              rewriter)))
        return failure();
      rewriter.eraseOp(op);
      return success();
    }
    if (isa<FuncType, PrimFuncType>(op.getObject().getType())) {
      rewriter.eraseOp(op);
      return success();
    }
    if (auto classType = dyn_cast<ClassType>(op.getObject().getType())) {
      (void)typeConverter;
      StringRef suffix = op->hasAttr("lython.final_local_class_decref")
                             ? "destroy_local"
                             : "decref";
      if (adaptor.getObject().empty())
        return failure();
      if (failed(emitStaticClassRefCountCall(op, adaptor.getObject().front(),
                                             classType, module, rewriter,
                                             suffix)))
        return failure();
      rewriter.eraseOp(op);
      return success();
    }
    RuntimeAPI runtime(module, rewriter, *typeConverter);

    runtime.call(op.getLoc(), RuntimeSymbols::kDecRef, /*resultType=*/nullptr,
                 adaptor.getObject());
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populatePyRefCountLoweringPatterns(PyLLVMTypeConverter &typeConverter,
                                        RewritePatternSet &patterns) {
  auto *ctx = patterns.getContext();
  patterns.add<IncRefLowering, DecRefLowering>(typeConverter, ctx);
}

} // namespace py
