#include "PyCall/Utils.h"

#include "llvm/ADT/STLExtras.h"

#include <cstdint>

using namespace mlir;

namespace py {

ClassOp lookupClassSymbol(Operation *from, ClassType classType);

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

static func::FuncOp resolveLocalSelfHelper(func::FuncOp callee,
                                           ModuleOp module) {
  auto helperAttr =
      callee->getAttrOfType<SymbolRefAttr>("lython.local_self_helper");
  if (!helperAttr)
    return nullptr;
  auto helperName = helperAttr.getLeafReference().empty()
                        ? helperAttr.getRootReference().getValue()
                        : helperAttr.getLeafReference().getValue();
  return module.lookupSymbol<func::FuncOp>(helperName);
}

static func::FuncOp resolveFreshInitHelper(func::FuncOp callee,
                                           ModuleOp module) {
  auto helperAttr =
      callee->getAttrOfType<SymbolRefAttr>("lython.fresh_init_helper");
  if (!helperAttr)
    return nullptr;
  auto helperName = helperAttr.getLeafReference().empty()
                        ? helperAttr.getRootReference().getValue()
                        : helperAttr.getLeafReference().getValue();
  return module.lookupSymbol<func::FuncOp>(helperName);
}

static func::FuncOp resolveVoidHelper(func::FuncOp callee, ModuleOp module) {
  auto helperAttr = callee->getAttrOfType<SymbolRefAttr>("lython.void_helper");
  if (!helperAttr)
    return nullptr;
  auto helperName = helperAttr.getLeafReference().empty()
                        ? helperAttr.getRootReference().getValue()
                        : helperAttr.getLeafReference().getValue();
  return module.lookupSymbol<func::FuncOp>(helperName);
}

static func::FuncOp resolveClassReturnHelper(func::FuncOp callee,
                                             ModuleOp module) {
  auto helperAttr =
      callee->getAttrOfType<SymbolRefAttr>("lython.class_return_helper");
  if (!helperAttr)
    return nullptr;
  auto helperName = helperAttr.getLeafReference().empty()
                        ? helperAttr.getRootReference().getValue()
                        : helperAttr.getLeafReference().getValue();
  return module.lookupSymbol<func::FuncOp>(helperName);
}

static std::string getPublishedBorrowHelperAttrName(unsigned argIndex) {
  return "lython.published_borrow_helper_arg" + std::to_string(argIndex);
}

static func::FuncOp resolvePublishedBorrowHelper(func::FuncOp callee,
                                                 unsigned argIndex,
                                                 ModuleOp module) {
  auto helperAttr = callee->getAttrOfType<SymbolRefAttr>(
      getPublishedBorrowHelperAttrName(argIndex));
  if (!helperAttr)
    return nullptr;
  auto helperName = helperAttr.getLeafReference().empty()
                        ? helperAttr.getRootReference().getValue()
                        : helperAttr.getLeafReference().getValue();
  return module.lookupSymbol<func::FuncOp>(helperName);
}

static bool arrayAttrContainsIndex(ArrayAttr attr, unsigned index) {
  if (!attr)
    return false;
  for (Attribute element : attr) {
    auto intAttr = dyn_cast<IntegerAttr>(element);
    if (intAttr && intAttr.getInt() == static_cast<int64_t>(index))
      return true;
  }
  return false;
}

static FuncOp resolveDirectPyFuncSymbol(Operation *from, Value callable) {
  callable = stripIdentityCasts(callable);

  if (auto funcObject = callable.getDefiningOp<FuncObjectOp>()) {
    Operation *symbol =
        SymbolTable::lookupNearestSymbolFrom(from, funcObject.getTargetAttr());
    return dyn_cast_or_null<FuncOp>(symbol);
  }
  return nullptr;
}

static bool funcResultIsPublished(Operation *funcLike, unsigned resultIndex) {
  return funcLike && arrayAttrContainsIndex(funcLike->getAttrOfType<ArrayAttr>(
                                                "lython.returns_published"),
                                            resultIndex);
}

static bool isFreshClassInstanceValue(Value value) {
  return isa_and_nonnull<ClassNewOp>(stripIdentityCasts(value).getDefiningOp());
}

static bool isLocalClassInstanceValue(Value value) {
  value = stripIdentityCasts(value);
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    auto *owner = arg.getOwner();
    auto func = owner ? owner->getParentOp() : nullptr;
    auto funcOp = dyn_cast_or_null<func::FuncOp>(func);
    if (!funcOp || arg.getArgNumber() != 0)
      return false;
    return static_cast<bool>(funcOp->getAttr("lython.local_self_arg0")) ||
           static_cast<bool>(funcOp->getAttr("lython.zero_initialized_self"));
  }

  if (!isa<ClassType>(value.getType()))
    return false;
  return isa_and_nonnull<ClassNewOp>(value.getDefiningOp());
}

static bool isPublishedClassInstanceValue(Value value) {
  value = stripIdentityCasts(value);
  if (!isa<ClassType>(value.getType()))
    return false;

  if (auto arg = dyn_cast<BlockArgument>(value)) {
    Block *owner = arg.getOwner();
    if (!owner)
      return false;
    Block *pred = owner->getUniquePredecessor();
    if (!pred)
      return false;
    auto invoke = dyn_cast_or_null<InvokeOp>(pred->getTerminator());
    if (!invoke || invoke.getNormalDest() != owner)
      return false;
    auto callee = resolveDirectPyFuncSymbol(invoke, invoke.getCallable());
    return funcResultIsPublished(callee, arg.getArgNumber());
  }

  Operation *def = value.getDefiningOp();
  if (!def)
    return false;
  if (isa<PublishOp, ClassPromoteOp, ListGetOp>(def))
    return true;

  auto result = dyn_cast<OpResult>(value);
  if (!result)
    return false;

  if (auto call = dyn_cast<CallOp>(def))
    return funcResultIsPublished(
        resolveDirectPyFuncSymbol(call, call.getCallable()),
        result.getResultNumber());

  if (auto call = dyn_cast<CallVectorOp>(def))
    return funcResultIsPublished(
        resolveDirectPyFuncSymbol(call, call.getCallable()),
        result.getResultNumber());

  if (auto call = dyn_cast<func::CallOp>(def)) {
    ModuleOp module = call->getParentOfType<ModuleOp>();
    if (!module)
      return false;
    return funcResultIsPublished(
        module.lookupSymbol<func::FuncOp>(call.getCallee()),
        result.getResultNumber());
  }

  return false;
}

} // namespace

void emitTracebackPush(Location loc, func::FuncOp func, RuntimeAPI &runtime,
                       ConversionPatternRewriter &rewriter) {
  StringAttr fileAttr;
  std::int64_t line = 0;
  std::int64_t col = 0;
  getLocInfo(loc, rewriter.getContext(), fileAttr, line, col);
  StringAttr funcAttr = getFuncNameAttr(func, rewriter.getContext());
  Value filePtr = runtime.getStringLiteral(loc, fileAttr);
  Value funcPtr = runtime.getStringLiteral(loc, funcAttr);
  Value lineConst = rewriter.create<LLVM::ConstantOp>(
      loc, rewriter.getI32Type(),
      rewriter.getI32IntegerAttr(static_cast<int32_t>(line)));
  Value colConst = rewriter.create<LLVM::ConstantOp>(
      loc, rewriter.getI32Type(),
      rewriter.getI32IntegerAttr(static_cast<int32_t>(col)));
  runtime.call(loc, RuntimeSymbols::kTracebackPush, Type(),
               ValueRange{filePtr, funcPtr, lineConst, colConst});
}

bool isBuiltinPrintCallable(Value callable) {
  while (auto identity = callable.getDefiningOp<CastIdentityOp>())
    callable = identity.getInput();
  auto call = callable.getDefiningOp<LLVM::CallOp>();
  if (!call)
    return false;
  auto callee = call.getCallee();
  return callee && *callee == RuntimeSymbols::kGetBuiltinPrint;
}

void ensureLandingpad(Block *unwind, Location loc,
                      ConversionPatternRewriter &rewriter) {
  if (!unwind)
    return;
  if (!unwind->empty() && llvm::isa<LLVM::LandingpadOp>(unwind->front()))
    return;
  rewriter.setInsertionPointToStart(unwind);
  auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
  auto i32Type = rewriter.getI32Type();
  auto lpType = LLVM::LLVMStructType::getLiteral(
      rewriter.getContext(), ArrayRef<Type>{ptrType, i32Type});
  rewriter.create<LLVM::LandingpadOp>(loc, lpType, rewriter.getUnitAttr(),
                                      ValueRange{});
}

bool canUseVoidHelper(CallVectorOp op, func::FuncOp callee) {
  auto helperAttr = callee->getAttrOfType<SymbolRefAttr>("lython.void_helper");
  if (!helperAttr)
    return false;
  if (op.getNumResults() != 1 || !isa<NoneType>(op.getResult(0).getType()))
    return false;
  for (Operation *user : op.getResult(0).getUsers())
    if (!isa<DecRefOp>(user))
      return false;
  return true;
}

Value stripIdentityCasts(Value value) {
  while (auto identity = value.getDefiningOp<CastIdentityOp>())
    value = identity.getInput();
  return value;
}

FailureOr<LLVM::LLVMStructType>
getStaticClassObjectType(Operation *from, ClassType classType,
                         const PyLLVMTypeConverter &typeConverter) {
  ClassOp classOp = lookupClassSymbol(from, classType);
  if (!classOp) {
    from->emitError("unable to resolve class '")
        << classType.getClassName() << "'";
    return failure();
  }

  ArrayAttr fieldNamesAttr = classOp.getFieldNamesAttr();
  ArrayAttr fieldTypesAttr = classOp.getFieldTypesAttr();
  if (!fieldNamesAttr && !fieldTypesAttr) {
    auto emptyStorage =
        LLVM::LLVMStructType::getLiteral(from->getContext(), ArrayRef<Type>{});
    return LLVM::LLVMStructType::getLiteral(
        from->getContext(),
        ArrayRef<Type>{emptyStorage, IntegerType::get(from->getContext(), 1),
                       IntegerType::get(from->getContext(), 32),
                       IntegerType::get(from->getContext(), 64)});
  }
  if (!fieldNamesAttr || !fieldTypesAttr ||
      fieldNamesAttr.size() != fieldTypesAttr.size()) {
    from->emitError("class '")
        << classType.getClassName() << "' has malformed static field schema";
    return failure();
  }

  SmallVector<Type, 8> loweredFieldTypes;
  for (Attribute typeAttr : fieldTypesAttr) {
    auto mlirTypeAttr = dyn_cast<TypeAttr>(typeAttr);
    if (!mlirTypeAttr) {
      from->emitError("class '")
          << classType.getClassName() << "' has malformed static field schema";
      return failure();
    }
    Type lowered = typeConverter.convertType(mlirTypeAttr.getValue());
    if (!lowered) {
      from->emitError("failed to convert field type ")
          << mlirTypeAttr.getValue() << " in class '"
          << classType.getClassName() << "'";
      return failure();
    }
    loweredFieldTypes.push_back(lowered);
  }

  auto storageType =
      LLVM::LLVMStructType::getLiteral(from->getContext(), loweredFieldTypes);
  return LLVM::LLVMStructType::getLiteral(
      from->getContext(),
      ArrayRef<Type>{storageType, IntegerType::get(from->getContext(), 1),
                     IntegerType::get(from->getContext(), 32),
                     IntegerType::get(from->getContext(), 64)});
}

Value createStaticClassSlot(Location loc, LLVM::LLVMStructType objectType,
                            ConversionPatternRewriter &rewriter,
                            Operation *anchor) {
  auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
  auto i64Type = rewriter.getI64Type();
  auto parentFunc = anchor->getParentOfType<func::FuncOp>();
  if (!parentFunc)
    return Value();

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(&parentFunc.getBody().front());
  Value one = rewriter.create<LLVM::ConstantOp>(loc, i64Type,
                                                rewriter.getI64IntegerAttr(1));
  return rewriter.create<LLVM::AllocaOp>(loc, ptrType, objectType, one,
                                         /*alignment=*/0);
}

func::FuncOp resolvePreferredDirectHelper(func::FuncOp callee,
                                          ValueRange operands, ModuleOp module,
                                          bool allowVoidHelper) {
  func::FuncOp preferred = callee;
  if (!allowVoidHelper)
    if (auto classReturnHelper = resolveClassReturnHelper(callee, module))
      preferred = classReturnHelper;
  if (allowVoidHelper)
    if (auto voidHelper = resolveVoidHelper(callee, module))
      preferred = voidHelper;

  if (!operands.empty()) {
    bool usedFreshHelper = false;
    if (allowVoidHelper && isFreshClassInstanceValue(operands.front())) {
      if (auto freshHelper = resolveFreshInitHelper(callee, module))
        preferred = freshHelper, usedFreshHelper = true;
    }
    if (!usedFreshHelper && isLocalClassInstanceValue(operands.front())) {
      func::FuncOp localHelperOwner = allowVoidHelper ? callee : preferred;
      if (auto localHelper = resolveLocalSelfHelper(localHelperOwner, module))
        preferred = localHelper;
    }
  }

  for (unsigned idx = 1; idx < operands.size(); ++idx) {
    if (!isPublishedClassInstanceValue(operands[idx]))
      continue;
    if (auto publishedHelper =
            resolvePublishedBorrowHelper(preferred, idx, module))
      preferred = publishedHelper;
  }

  return preferred;
}

void eraseNoneResultUsers(CallVectorOp op,
                          ConversionPatternRewriter &rewriter) {
  for (Operation *user : llvm::make_early_inc_range(op.getResult(0).getUsers()))
    rewriter.eraseOp(user);
}

void materializeLogicalResults(Location loc, TypeRange logicalTypes,
                               ValueRange loweredResults,
                               SmallVectorImpl<Value> &results,
                               ConversionPatternRewriter &rewriter) {
  results.clear();
  results.reserve(logicalTypes.size());
  for (auto [logicalType, lowered] : llvm::zip(logicalTypes, loweredResults)) {
    if (lowered.getType() == logicalType) {
      results.push_back(lowered);
      continue;
    }
    auto cast = rewriter.create<CastIdentityOp>(loc, logicalType, lowered);
    results.push_back(cast.getResult());
  }
}

void materializeInvokeNormalResult(InvokeOp op, Value loweredResult,
                                   ConversionPatternRewriter &rewriter) {
  if (op.getNormalDestOperands().empty())
    return;
  Block *normalDest = op.getNormalDest();
  if (!normalDest || normalDest->getNumArguments() == 0)
    return;

  BlockArgument arg = normalDest->getArgument(0);
  Value replacement = loweredResult;
  if (loweredResult.getType() != arg.getType()) {
    Type logicalType = arg.getType();
    arg.setType(loweredResult.getType());
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(normalDest);
    replacement =
        rewriter.create<CastIdentityOp>(op.getLoc(), logicalType, loweredResult)
            .getResult();
  }
  arg.replaceAllUsesWith(replacement);
}

Block *createInvokeNormalBridge(Block *finalDest, Type bridgeArgType,
                                Location loc,
                                ConversionPatternRewriter &rewriter) {
  if (!finalDest)
    return nullptr;
  auto *bridge = new Block();
  bridge->addArgument(bridgeArgType, loc);
  bridge->insertBefore(finalDest);
  return bridge;
}

void finalizeInvokeNormalBridge(Block *bridge, Block *finalDest,
                                Value forwardedValue, Location loc,
                                ConversionPatternRewriter &rewriter) {
  if (!bridge || !finalDest || finalDest->getNumArguments() == 0)
    return;

  BlockArgument finalArg = finalDest->getArgument(0);
  if (forwardedValue.getType() != finalArg.getType()) {
    Type logicalType = finalArg.getType();
    finalArg.setType(forwardedValue.getType());
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(finalDest);
    Value replacement =
        rewriter.create<CastIdentityOp>(loc, logicalType, forwardedValue)
            .getResult();
    finalArg.replaceAllUsesWith(replacement);
  }
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToEnd(bridge);
  rewriter.create<cf::BranchOp>(loc, finalDest, ValueRange{forwardedValue});
}

void eraseInvokeNormalSeedDrops(InvokeOp op, Value logicalSeed,
                                ConversionPatternRewriter &rewriter) {
  Block *normalDest = op.getNormalDest();
  if (!normalDest)
    return;

  SmallVector<DecRefOp> decRefs;
  for (Operation &nested : *normalDest)
    if (auto decRef = dyn_cast<DecRefOp>(&nested))
      if (decRef.getObject() == logicalSeed)
        decRefs.push_back(decRef);

  for (DecRefOp decRef : decRefs)
    rewriter.eraseOp(decRef);
}

void populatePyCallLoweringPatterns(PyLLVMTypeConverter &typeConverter,
                                    RewritePatternSet &patterns) {
  populatePyDirectCallLoweringPatterns(typeConverter, patterns);
  populatePyInvokeLoweringPatterns(typeConverter, patterns);
}

} // namespace py
