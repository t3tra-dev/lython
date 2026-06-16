#include "PyCall/Utils.h"

#include "Common/Object.h"
#include "Common/RuntimeSupport.h"
#include "Passes/OwnershipAnalysis.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "llvm/ADT/STLExtras.h"

#include <cstdint>

namespace py {

namespace {

static void getLocInfo(mlir::Location loc, mlir::MLIRContext *ctx,
                       mlir::StringAttr &fileAttr, std::int64_t &line,
                       std::int64_t &col) {
  if (auto fileLoc = llvm::dyn_cast<mlir::FileLineColLoc>(loc)) {
    fileAttr = fileLoc.getFilename();
    line = static_cast<std::int64_t>(fileLoc.getLine());
    col = static_cast<std::int64_t>(fileLoc.getColumn());
    return;
  }
  if (auto nameLoc = llvm::dyn_cast<mlir::NameLoc>(loc)) {
    getLocInfo(nameLoc.getChildLoc(), ctx, fileAttr, line, col);
    return;
  }
  if (auto fused = llvm::dyn_cast<mlir::FusedLoc>(loc)) {
    for (auto subloc : fused.getLocations()) {
      if (auto subfile = llvm::dyn_cast<mlir::FileLineColLoc>(subloc)) {
        fileAttr = subfile.getFilename();
        line = static_cast<std::int64_t>(subfile.getLine());
        col = static_cast<std::int64_t>(subfile.getColumn());
        return;
      }
    }
  }
  fileAttr = mlir::StringAttr::get(ctx, "<unknown>");
  line = 0;
  col = 0;
}

static mlir::StringAttr getFuncNameAttr(mlir::func::FuncOp func,
                                        mlir::MLIRContext *ctx) {
  if (!func)
    return mlir::StringAttr::get(ctx, "<unknown>");
  llvm::StringRef name = func.getName();
  if (name == "main")
    return mlir::StringAttr::get(ctx, "<module>");
  return mlir::StringAttr::get(ctx, name);
}

static mlir::func::FuncOp resolveLocalSelfHelper(mlir::func::FuncOp callee,
                                                 mlir::ModuleOp module) {
  auto helperAttr =
      callee->getAttrOfType<mlir::SymbolRefAttr>("ly.local_self_helper");
  if (!helperAttr)
    return nullptr;
  auto helperName = helperAttr.getLeafReference().empty()
                        ? helperAttr.getRootReference().getValue()
                        : helperAttr.getLeafReference().getValue();
  return module.lookupSymbol<mlir::func::FuncOp>(helperName);
}

static mlir::func::FuncOp resolveFreshInitHelper(mlir::func::FuncOp callee,
                                                 mlir::ModuleOp module) {
  auto helperAttr =
      callee->getAttrOfType<mlir::SymbolRefAttr>("ly.fresh_init_helper");
  if (!helperAttr)
    return nullptr;
  auto helperName = helperAttr.getLeafReference().empty()
                        ? helperAttr.getRootReference().getValue()
                        : helperAttr.getLeafReference().getValue();
  return module.lookupSymbol<mlir::func::FuncOp>(helperName);
}

static mlir::func::FuncOp resolveVoidHelper(mlir::func::FuncOp callee,
                                            mlir::ModuleOp module) {
  auto helperAttr =
      callee->getAttrOfType<mlir::SymbolRefAttr>("ly.void_helper");
  if (!helperAttr)
    return nullptr;
  auto helperName = helperAttr.getLeafReference().empty()
                        ? helperAttr.getRootReference().getValue()
                        : helperAttr.getLeafReference().getValue();
  return module.lookupSymbol<mlir::func::FuncOp>(helperName);
}

static mlir::func::FuncOp
resolvePublishedBorrowHelper(mlir::func::FuncOp callee, unsigned argIndex,
                             mlir::ModuleOp module) {
  auto helperAttr = callee->getAttrOfType<mlir::SymbolRefAttr>(
      publication::borrow::Attr::name(argIndex));
  if (!helperAttr)
    return nullptr;
  auto helperName = helperAttr.getLeafReference().empty()
                        ? helperAttr.getRootReference().getValue()
                        : helperAttr.getLeafReference().getValue();
  return module.lookupSymbol<mlir::func::FuncOp>(helperName);
}

static bool arrayAttrContainsIndex(mlir::ArrayAttr attr, unsigned index) {
  if (!attr)
    return false;
  for (mlir::Attribute element : attr) {
    auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(element);
    if (intAttr && intAttr.getInt() == static_cast<int64_t>(index))
      return true;
  }
  return false;
}

static CallableFuncOp resolveDirectPyFuncSymbol(mlir::Operation *from,
                                                mlir::Value callable) {
  callable = stripBridgeCasts(callable);

  if (auto funcObject = callable.getDefiningOp<CallableObjectOp>()) {
    mlir::Operation *symbol = mlir::SymbolTable::lookupNearestSymbolFrom(
        from, funcObject.getTargetAttr());
    return mlir::dyn_cast_or_null<CallableFuncOp>(symbol);
  }
  return nullptr;
}

static bool funcResultIsPublished(mlir::Operation *funcLike,
                                  unsigned resultIndex) {
  return funcLike &&
         arrayAttrContainsIndex(
             funcLike->getAttrOfType<mlir::ArrayAttr>("ly.returns_published"),
             resultIndex);
}

static bool isFreshClassInstanceValue(mlir::Value value) {
  return mlir::isa_and_nonnull<ClassNewOp>(
      stripBridgeCasts(value).getDefiningOp());
}

static bool isLocalClassInstanceValue(mlir::Value value) {
  value = stripBridgeCasts(value);
  if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    auto *owner = arg.getOwner();
    auto func = owner ? owner->getParentOp() : nullptr;
    auto funcOp = mlir::dyn_cast_or_null<mlir::func::FuncOp>(func);
    if (!funcOp || arg.getArgNumber() != 0)
      return false;
    return static_cast<bool>(funcOp->getAttr("ly.local_self_arg0")) ||
           static_cast<bool>(funcOp->getAttr("ly.zero_initialized_self"));
  }

  if (!mlir::isa<ClassType>(value.getType()))
    return false;
  return mlir::isa_and_nonnull<ClassNewOp>(value.getDefiningOp());
}

static bool isPublishedClassInstanceValue(mlir::Value value) {
  value = stripBridgeCasts(value);
  if (!mlir::isa<ClassType>(value.getType()))
    return false;

  if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    mlir::Block *owner = arg.getOwner();
    if (!owner)
      return false;
    mlir::Block *pred = owner->getUniquePredecessor();
    if (!pred)
      return false;
    auto invoke = mlir::dyn_cast_or_null<InvokeOp>(pred->getTerminator());
    if (!invoke || invoke.getNormalDest() != owner)
      return false;
    auto callee = resolveDirectPyFuncSymbol(invoke, invoke.getCallable());
    return funcResultIsPublished(callee, arg.getArgNumber());
  }

  mlir::Operation *def = value.getDefiningOp();
  if (!def)
    return false;
  if (mlir::isa<PublishOp, ClassPromoteOp>(def))
    return true;
  if (auto getItem = mlir::dyn_cast<GetItemOp>(def))
    return mlir::isa<ListType>(getItem.getContainer().getType());

  auto result = mlir::dyn_cast<mlir::OpResult>(value);
  if (!result)
    return false;

  if (auto call = mlir::dyn_cast<CallOp>(def))
    return funcResultIsPublished(
        resolveDirectPyFuncSymbol(call, call.getCallable()),
        result.getResultNumber());

  if (auto call = mlir::dyn_cast<mlir::func::CallOp>(def)) {
    mlir::ModuleOp module = call->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return false;
    return funcResultIsPublished(
        module.lookupSymbol<mlir::func::FuncOp>(call.getCallee()),
        result.getResultNumber());
  }

  return false;
}

} // namespace

void emitTracebackPush(mlir::Location loc, mlir::func::FuncOp func,
                       RuntimeAPI &runtime,
                       mlir::ConversionPatternRewriter &rewriter) {
  mlir::StringAttr fileAttr;
  std::int64_t line = 0;
  std::int64_t col = 0;
  getLocInfo(loc, rewriter.getContext(), fileAttr, line, col);
  mlir::StringAttr funcAttr = getFuncNameAttr(func, rewriter.getContext());
  mlir::Value fileBytes = runtime.getByteLiteral(loc, fileAttr);
  mlir::Value funcBytes = runtime.getByteLiteral(loc, funcAttr);
  mlir::Value lineConst = rewriter.create<mlir::arith::ConstantIntOp>(
      loc, static_cast<int32_t>(line), 32);
  mlir::Value colConst = rewriter.create<mlir::arith::ConstantIntOp>(
      loc, static_cast<int32_t>(col), 32);
  runtime.call(loc, RuntimeSymbols::kTracebackPush, mlir::Type(),
               mlir::ValueRange{fileBytes, funcBytes, lineConst, colConst});
}

static std::optional<llvm::StringRef>
builtinCallableName(mlir::Value callable) {
  callable = stripBridgeCasts(callable);
  if (auto funcObject = callable.getDefiningOp<CallableObjectOp>())
    return funcObject.getTarget();
  if (auto constant = callable.getDefiningOp<mlir::func::ConstantOp>()) {
    mlir::SymbolRefAttr symbol = constant.getValueAttr();
    return symbol.getLeafReference().empty()
               ? symbol.getRootReference().getValue()
               : symbol.getLeafReference().getValue();
  }
  return std::nullopt;
}

bool isBuiltinPrintCallable(mlir::Value callable) {
  std::optional<llvm::StringRef> name = builtinCallableName(callable);
  return name && (*name == "__builtin_print" ||
                  *name == "__builtin_print_raw" || *name == "print");
}

bool isBuiltinPrintRawCallable(mlir::Value callable) {
  std::optional<llvm::StringRef> name = builtinCallableName(callable);
  return name && *name == "__builtin_print_raw";
}

void ensureLandingpad(mlir::Block *unwind, mlir::Location loc,
                      mlir::ConversionPatternRewriter &rewriter) {
  if (!unwind)
    return;
  if (!unwind->empty() && llvm::isa<mlir::LLVM::LandingpadOp>(unwind->front()))
    return;
  rewriter.setInsertionPointToStart(unwind);
  auto ptrType = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
  auto i32Type = rewriter.getI32Type();
  auto lpType = mlir::LLVM::LLVMStructType::getLiteral(
      rewriter.getContext(), llvm::ArrayRef<mlir::Type>{ptrType, i32Type});
  rewriter.create<mlir::LLVM::LandingpadOp>(loc, lpType, rewriter.getUnitAttr(),
                                            mlir::ValueRange{});
}

bool canUseVoidHelper(CallOp op, mlir::func::FuncOp callee) {
  auto helperAttr =
      callee->getAttrOfType<mlir::SymbolRefAttr>("ly.void_helper");
  if (!helperAttr)
    return false;
  if (op.getNumResults() != 1 ||
      !mlir::isa<NoneType>(op.getResult(0).getType()))
    return false;
  for (mlir::Operation *user : op.getResult(0).getUsers())
    if (!mlir::isa<DecRefOp>(user))
      return false;
  return true;
}

mlir::Value stripBridgeCasts(mlir::Value value) {
  while (value) {
    if (auto cast = value.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
      if (cast->getNumOperands() != 1)
        break;
      value = cast.getOperand(0);
      continue;
    }
    if (auto upcast = value.getDefiningOp<ClassUpcastOp>()) {
      value = upcast.getInput();
      continue;
    }
    break;
  }
  return value;
}

mlir::func::FuncOp resolvePreferredDirectHelper(mlir::func::FuncOp callee,
                                                mlir::ValueRange operands,
                                                mlir::ModuleOp module,
                                                bool allowVoidHelper) {
  mlir::func::FuncOp preferred = callee;
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
      mlir::func::FuncOp localHelperOwner =
          allowVoidHelper ? callee : preferred;
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

static bool canBridgeCallOperandType(mlir::Type type) {
  return isPyType(type) ||
         mlir::isa<TupleType, ListType, ClassType, DictType>(type);
}

static bool carriesObjectHeaderPrefix(mlir::Type type) {
  return mlir::isa<IntType, StrType, ExceptionType, ClassType, TypeType,
                   ObjectType, ProtocolType, TracebackType>(type);
}

static mlir::Value materializeNoneObjectHeader(mlir::Location loc,
                                               mlir::RewriterBase &rewriter) {
  mlir::MemRefType headerType =
      object_abi::Header::owned(rewriter.getContext());
  mlir::Value header = rewriter.create<mlir::memref::AllocaOp>(loc, headerType);
  mlir::Type i64Type = rewriter.getI64Type();
  auto storeI64 = [&](int64_t slot, int64_t value) {
    mlir::Value iv = rewriter.create<mlir::arith::ConstantIndexOp>(loc, slot);
    mlir::Value constant =
        rewriter.create<mlir::arith::ConstantIntOp>(loc, value, i64Type);
    rewriter.create<mlir::memref::StoreOp>(loc, constant, header,
                                           mlir::ValueRange{iv});
  };
  storeI64(object_abi::kRefcountSlot, object_abi::kImmortalRefcount);
  storeI64(object_abi::kLayoutIdSlot,
           static_cast<int64_t>(object_abi::KindId::None));
  if (mlir::Operation *def = header.getDefiningOp())
    def->setAttr(OwnershipContractAttrs::kImmortalObject,
                 rewriter.getUnitAttr());
  return header;
}

static std::optional<llvm::SmallVector<mlir::Value>>
materializeObjectHeaderView(mlir::Location loc, mlir::Value element,
                            mlir::RewriterBase &rewriter,
                            const PyLLVMTypeConverter &typeConverter) {
  if (!mlir::isa<ObjectType, ProtocolType>(element.getType()))
    return std::nullopt;
  auto cast = element.getDefiningOp<mlir::UnrealizedConversionCastOp>();
  if (!cast || cast.getNumOperands() != 1)
    return std::nullopt;
  mlir::Value source = cast.getOperand(0);
  if (mlir::isa<NoneType>(source.getType()))
    return llvm::SmallVector<mlir::Value>{
        materializeNoneObjectHeader(loc, rewriter)};
  if (source.getType() == element.getType() ||
      !carriesObjectHeaderPrefix(source.getType()))
    return std::nullopt;

  llvm::SmallVector<mlir::Type> sourceParts;
  if (mlir::failed(typeConverter.convertType(source.getType(), sourceParts)) ||
      sourceParts.empty())
    return std::nullopt;

  llvm::SmallVector<mlir::Value> loweredParts;
  if (sourceParts.size() == 1 && source.getType() == sourceParts.front()) {
    loweredParts.push_back(source);
  } else {
    auto lowered = rewriter.create<mlir::UnrealizedConversionCastOp>(
        loc, mlir::TypeRange(sourceParts), mlir::ValueRange{source});
    loweredParts.append(lowered.getResults().begin(),
                        lowered.getResults().end());
  }
  return llvm::SmallVector<mlir::Value>{loweredParts.front()};
}

mlir::LogicalResult appendFlattenedCallOperands(
    mlir::Location loc, mlir::ValueRange elements, mlir::FunctionType funcType,
    unsigned directInputCount, llvm::SmallVectorImpl<mlir::Value> &operands,
    mlir::RewriterBase &rewriter, const PyLLVMTypeConverter &typeConverter) {
  unsigned expectedIndex = 0;
  for (mlir::Value element : elements) {
    llvm::SmallVector<mlir::Value> remapped;
    if (std::optional<llvm::SmallVector<mlir::Value>> header =
            materializeObjectHeaderView(loc, element, rewriter,
                                        typeConverter)) {
      remapped = std::move(*header);
    } else {
      llvm::SmallVector<mlir::Type> convertedElementTypes;
      if (mlir::failed(typeConverter.convertType(element.getType(),
                                                 convertedElementTypes)) ||
          convertedElementTypes.empty())
        return mlir::failure();

      if (convertedElementTypes.size() == 1 &&
          element.getType() == convertedElementTypes.front()) {
        remapped.push_back(element);
      } else {
        if (!canBridgeCallOperandType(element.getType()))
          return mlir::failure();
        auto cast = rewriter.create<mlir::UnrealizedConversionCastOp>(
            loc, mlir::TypeRange(convertedElementTypes),
            mlir::ValueRange{element});
        remapped.assign(cast.getResults().begin(), cast.getResults().end());
      }
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

void eraseNoneResultUsers(CallOp op, mlir::RewriterBase &rewriter) {
  for (mlir::Operation *user :
       llvm::make_early_inc_range(op.getResult(0).getUsers()))
    rewriter.eraseOp(user);
}

void materializeLogicalResults(mlir::Location loc, mlir::TypeRange logicalTypes,
                               mlir::ValueRange loweredResults,
                               llvm::SmallVectorImpl<mlir::Value> &results,
                               const PyLLVMTypeConverter &typeConverter,
                               mlir::RewriterBase &rewriter) {
  results.clear();
  results.reserve(logicalTypes.size());
  unsigned loweredIndex = 0;
  for (mlir::Type logicalType : logicalTypes) {
    llvm::SmallVector<mlir::Type, 4> convertedTypes;
    if (mlir::failed(typeConverter.convertType(logicalType, convertedTypes)) ||
        convertedTypes.empty() ||
        loweredIndex + convertedTypes.size() > loweredResults.size())
      return;

    mlir::ValueRange loweredGroup =
        loweredResults.slice(loweredIndex, convertedTypes.size());
    loweredIndex += convertedTypes.size();
    if (loweredGroup.size() == 1 &&
        loweredGroup.front().getType() == logicalType) {
      results.push_back(loweredGroup.front());
      continue;
    }
    if (loweredGroup.size() == 1) {
      auto cast = rewriter.create<mlir::UnrealizedConversionCastOp>(
          loc, mlir::TypeRange{logicalType}, loweredGroup);
      if (isPyOwnershipTrackedType(logicalType)) {
        llvm::SmallVector<mlir::Attribute, 1> owned{
            rewriter.getI64IntegerAttr(0)};
        cast->setAttr(OwnershipContractAttrs::kOwnedResults,
                      rewriter.getArrayAttr(owned));
      }
      results.push_back(cast.getResult(0));
      continue;
    }
    auto cast = rewriter.create<mlir::UnrealizedConversionCastOp>(
        loc, mlir::TypeRange{logicalType}, loweredGroup);
    if (isPyOwnershipTrackedType(logicalType)) {
      llvm::SmallVector<mlir::Attribute, 1> owned{
          rewriter.getI64IntegerAttr(0)};
      cast->setAttr(OwnershipContractAttrs::kOwnedResults,
                    rewriter.getArrayAttr(owned));
    }
    results.push_back(cast.getResult(0));
  }
}

void materializeInvokeNormalResult(InvokeOp op, mlir::Value loweredResult,
                                   mlir::ConversionPatternRewriter &rewriter) {
  if (op.getNormalDestOperands().empty())
    return;
  mlir::Block *normalDest = op.getNormalDest();
  if (!normalDest || normalDest->getNumArguments() == 0)
    return;

  mlir::BlockArgument arg = normalDest->getArgument(0);
  mlir::Value replacement = loweredResult;
  if (loweredResult.getType() != arg.getType()) {
    mlir::Type logicalType = arg.getType();
    arg.setType(loweredResult.getType());
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(normalDest);
    replacement = rewriter
                      .create<mlir::UnrealizedConversionCastOp>(
                          op.getLoc(), mlir::TypeRange{logicalType},
                          mlir::ValueRange{loweredResult})
                      .getResult(0);
  }
  arg.replaceAllUsesWith(replacement);
}

mlir::Block *
createInvokeNormalBridge(mlir::Block *finalDest, mlir::Type bridgeArgType,
                         mlir::Location loc,
                         mlir::ConversionPatternRewriter &rewriter) {
  if (!finalDest)
    return nullptr;
  auto *bridge = new mlir::Block();
  bridge->addArgument(bridgeArgType, loc);
  bridge->insertBefore(finalDest);
  return bridge;
}

void finalizeInvokeNormalBridge(mlir::Block *bridge, mlir::Block *finalDest,
                                mlir::Value forwardedValue, mlir::Location loc,
                                mlir::ConversionPatternRewriter &rewriter) {
  if (!bridge || !finalDest || finalDest->getNumArguments() == 0)
    return;

  mlir::BlockArgument finalArg = finalDest->getArgument(0);
  if (forwardedValue.getType() != finalArg.getType()) {
    mlir::Type logicalType = finalArg.getType();
    finalArg.setType(forwardedValue.getType());
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(finalDest);
    mlir::Value replacement = rewriter
                                  .create<mlir::UnrealizedConversionCastOp>(
                                      loc, mlir::TypeRange{logicalType},
                                      mlir::ValueRange{forwardedValue})
                                  .getResult(0);
    finalArg.replaceAllUsesWith(replacement);
  }
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToEnd(bridge);
  rewriter.create<mlir::cf::BranchOp>(loc, finalDest,
                                      mlir::ValueRange{forwardedValue});
}

void eraseInvokeNormalSeedDrops(InvokeOp op, mlir::Value logicalSeed,
                                mlir::ConversionPatternRewriter &rewriter) {
  mlir::Block *normalDest = op.getNormalDest();
  if (!normalDest)
    return;

  llvm::SmallVector<DecRefOp> decRefs;
  for (mlir::Operation &nested : *normalDest)
    if (auto decRef = mlir::dyn_cast<DecRefOp>(&nested))
      if (decRef.getObject() == logicalSeed)
        decRefs.push_back(decRef);

  for (DecRefOp decRef : decRefs)
    rewriter.eraseOp(decRef);
}

namespace lowering::call::Patterns {
void populate(PyLLVMTypeConverter &typeConverter,
              mlir::RewritePatternSet &patterns) {
  lowering::call::direct::Patterns::populate(typeConverter, patterns);
  lowering::call::invoke::Patterns::populate(typeConverter, patterns);
}
} // namespace lowering::call::Patterns

} // namespace py
