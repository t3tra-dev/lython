#include "PyCall/Utils.h"

#include "Common/Object.h"
#include "Common/SlotUtils.h"
#include "Passes/OwnershipAnalysis.h"

#include "mlir/Dialect/SCF/IR/SCF.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <algorithm>
#include <optional>

namespace py::optimizer::class_state {
bool local(mlir::Value value);
} // namespace py::optimizer::class_state

namespace py {
namespace {

static bool canDropNoneResult(CallVectorOp op) {
  if (op.getNumResults() == 0)
    return true;
  if (op.getNumResults() != 1 ||
      !mlir::isa<NoneType>(op.getResult(0).getType()))
    return false;
  return llvm::all_of(op.getResult(0).getUsers(), [](mlir::Operation *user) {
    return mlir::isa<DecRefOp>(user);
  });
}

struct DirectCallable {
  mlir::func::FuncOp func;
  mlir::FunctionType functionType;
  std::string symbolName;
  mlir::Operation *reference = nullptr;
  mlir::ArrayAttr argNames;
  mlir::ArrayAttr kwonlyNames;
};

static mlir::ArrayAttr getArrayAttr(mlir::Operation *op,
                                    llvm::StringRef name) {
  return op ? op->getAttrOfType<mlir::ArrayAttr>(name) : mlir::ArrayAttr{};
}

static std::optional<llvm::StringRef> stringAttrAt(mlir::ArrayAttr attr,
                                                   std::size_t index) {
  if (!attr || index >= attr.size())
    return std::nullopt;
  auto str = mlir::dyn_cast<mlir::StringAttr>(attr[index]);
  if (!str)
    return std::nullopt;
  return str.getValue();
}

static std::optional<std::size_t>
findKeyword(llvm::ArrayRef<llvm::StringRef> names, llvm::StringRef target,
            llvm::ArrayRef<bool> consumed) {
  for (auto [index, name] : llvm::enumerate(names)) {
    if (consumed[index])
      continue;
    if (name == target)
      return index;
  }
  return std::nullopt;
}

static bool collectTupleElements(mlir::Value tuple,
                                 llvm::SmallVectorImpl<mlir::Value> &out) {
  mlir::Value stripped = stripBridgeCasts(tuple);
  if (auto create = stripped.getDefiningOp<TupleCreateOp>()) {
    out.append(create.getElements().begin(), create.getElements().end());
    return true;
  }
  return static_cast<bool>(stripped.getDefiningOp<TupleEmptyOp>());
}

static mlir::LogicalResult
appendConvertedType(mlir::Type type, const PyLLVMTypeConverter &typeConverter,
                    llvm::SmallVectorImpl<mlir::Type> &storage,
                    mlir::Operation *emitOnError) {
  llvm::SmallVector<mlir::Type, 4> converted;
  if (mlir::failed(typeConverter.convertType(type, converted)) ||
      converted.empty()) {
    emitOnError->emitError("failed to convert direct callable type ") << type;
    return mlir::failure();
  }
  storage.append(converted.begin(), converted.end());
  return mlir::success();
}

static mlir::FailureOr<mlir::FunctionType>
loweredPyFuncType(FuncOp func, const PyLLVMTypeConverter &typeConverter,
                  mlir::Operation *emitOnError) {
  auto sigAttr = func->getAttrOfType<mlir::TypeAttr>("function_type");
  if (!sigAttr)
    return mlir::failure();
  auto sig = mlir::dyn_cast<FuncSignatureType>(sigAttr.getValue());
  if (!sig)
    return mlir::failure();

  llvm::SmallVector<mlir::Type, 8> inputs;
  llvm::SmallVector<mlir::Type, 4> results;
  auto positional = sig.getPositionalTypes();
  for (mlir::Type type : positional)
    if (mlir::failed(
            appendConvertedType(type, typeConverter, inputs, emitOnError)))
      return mlir::failure();
  auto kwonly = sig.getKwOnlyTypes();
  for (mlir::Type type : kwonly)
    if (mlir::failed(
            appendConvertedType(type, typeConverter, inputs, emitOnError)))
      return mlir::failure();
  if (sig.hasVararg() &&
      mlir::failed(appendConvertedType(sig.getVarargType(), typeConverter,
                                       inputs, emitOnError)))
    return mlir::failure();
  if (sig.hasKwarg() &&
      mlir::failed(appendConvertedType(sig.getKwargType(), typeConverter,
                                       inputs, emitOnError)))
    return mlir::failure();

  if (mlir::ArrayAttr closureTypes = func.getClosureTypesAttr()) {
    for (mlir::Attribute attr : closureTypes) {
      auto typeAttr = mlir::dyn_cast<mlir::TypeAttr>(attr);
      if (!typeAttr)
        return mlir::failure();
      if (mlir::failed(appendConvertedType(typeAttr.getValue(), typeConverter,
                                           inputs, emitOnError)))
        return mlir::failure();
    }
  }

  for (mlir::Type type : sig.getResultTypes()) {
    if (mlir::isa<NoneType>(type))
      continue;
    if (mlir::failed(
            appendConvertedType(type, typeConverter, results, emitOnError)))
      return mlir::failure();
  }
  return mlir::FunctionType::get(func.getContext(), inputs, results);
}

static mlir::LogicalResult packVectorArgsForDirectCall(
    CallVectorOp op, mlir::ValueRange posElements,
    mlir::ValueRange kwNameElements, mlir::ValueRange kwValueElements,
    mlir::ArrayAttr argNamesAttr, mlir::ArrayAttr kwonlyNamesAttr,
    llvm::SmallVectorImpl<mlir::Value> &directElements,
    mlir::PatternRewriter &rewriter) {
  auto callableType = mlir::dyn_cast<FuncType>(op.getCallable().getType());
  if (!callableType)
    return mlir::failure();

  FuncSignatureType signature = callableType.getSignature();
  auto positionalTypes = signature.getPositionalTypes();
  auto kwonlyTypes = signature.getKwOnlyTypes();
  if (kwNameElements.size() != kwValueElements.size())
    return mlir::failure();

  llvm::SmallVector<llvm::StringRef, 4> keywordNames;
  keywordNames.reserve(kwNameElements.size());
  for (mlir::Value nameValue : kwNameElements) {
    auto name = stripBridgeCasts(nameValue).getDefiningOp<StrConstantOp>();
    if (!name)
      return mlir::failure();
    keywordNames.push_back(name.getValueAttr().getValue());
  }
  llvm::SmallVector<bool, 8> keywordConsumed(keywordNames.size(), false);

  if (signature.hasVararg() && !kwonlyTypes.empty())
    return mlir::failure();

  if (!signature.hasVararg() &&
      posElements.size() > positionalTypes.size() + kwonlyTypes.size())
    return mlir::failure();

  for (std::size_t index = 0; index < positionalTypes.size(); ++index) {
    if (index < posElements.size()) {
      if (std::optional<llvm::StringRef> name =
              stringAttrAt(argNamesAttr, index)) {
        if (findKeyword(keywordNames, *name, keywordConsumed))
          return mlir::failure();
      }
      directElements.push_back(posElements[index]);
      continue;
    }

    std::optional<llvm::StringRef> name = stringAttrAt(argNamesAttr, index);
    if (!name)
      return mlir::failure();
    std::optional<std::size_t> keywordIndex =
        findKeyword(keywordNames, *name, keywordConsumed);
    if (!keywordIndex)
      return mlir::failure();
    directElements.push_back(kwValueElements[*keywordIndex]);
    keywordConsumed[*keywordIndex] = true;
  }

  if (!signature.hasVararg()) {
    for (std::size_t index = 0; index < kwonlyTypes.size(); ++index) {
      std::size_t normalizedIndex = positionalTypes.size() + index;
      if (normalizedIndex < posElements.size()) {
        directElements.push_back(posElements[normalizedIndex]);
        continue;
      }

      std::optional<llvm::StringRef> name =
          stringAttrAt(kwonlyNamesAttr, index);
      if (!name)
        return mlir::failure();
      std::optional<std::size_t> keywordIndex =
          findKeyword(keywordNames, *name, keywordConsumed);
      if (!keywordIndex)
        return mlir::failure();
      directElements.push_back(kwValueElements[*keywordIndex]);
      keywordConsumed[*keywordIndex] = true;
    }
  }

  if (signature.hasVararg()) {
    std::size_t positionalCount = positionalTypes.size();
    if (posElements.size() < positionalCount)
      return mlir::failure();

    auto varargType = mlir::dyn_cast<TupleType>(signature.getVarargType());
    if (!varargType || varargType.getElementTypes().size() != 1)
      return mlir::failure();

    llvm::SmallVector<mlir::Value, 4> varargElements(
        posElements.begin() + positionalCount, posElements.end());
    mlir::Value varargs =
        rewriter.create<TupleCreateOp>(op.getLoc(), varargType, varargElements);
    directElements.push_back(varargs);
  }

  if (signature.hasKwarg()) {
    auto kwargsType = mlir::dyn_cast<DictType>(signature.getKwargType());
    if (!kwargsType)
      return mlir::failure();
    mlir::Value kwargs =
        rewriter.create<DictEmptyOp>(op.getLoc(), kwargsType);
    for (auto [index, consumed] : llvm::enumerate(keywordConsumed)) {
      if (consumed)
        continue;
      rewriter.create<DictInsertOp>(op.getLoc(), kwargs, kwNameElements[index],
                                    kwValueElements[index]);
    }
    directElements.push_back(kwargs);
  } else if (llvm::any_of(keywordConsumed, [](bool consumed) {
               return !consumed;
             })) {
    return mlir::failure();
  }
  return mlir::success();
}

static DirectCallable resolveDirectCallableBySymbol(
    CallVectorOp op, mlir::ModuleOp module, llvm::StringRef symbolName,
    mlir::Operation *reference, const PyLLVMTypeConverter &typeConverter) {
  if (auto func = module.lookupSymbol<mlir::func::FuncOp>(symbolName))
    return {func,
            func.getFunctionType(),
            symbolName.str(),
            reference,
            getArrayAttr(func.getOperation(), "arg_names"),
            getArrayAttr(func.getOperation(), "kwonly_names")};
  if (auto func = module.lookupSymbol<FuncOp>(symbolName)) {
    mlir::FailureOr<mlir::FunctionType> type =
        loweredPyFuncType(func, typeConverter, op.getOperation());
    if (mlir::succeeded(type))
      return {nullptr,
              *type,
              symbolName.str(),
              reference,
              func.getArgNamesAttr(),
              func.getKwonlyNamesAttr()};
  }
  return {};
}

static DirectCallable
resolveDirectCallable(CallVectorOp op, mlir::ModuleOp module,
                      const PyLLVMTypeConverter &typeConverter) {
  mlir::Value callable = stripBridgeCasts(op.getCallable());
  mlir::Operation *producer = callable.getDefiningOp();
  if (auto constOp = mlir::dyn_cast_or_null<mlir::func::ConstantOp>(producer)) {
    mlir::SymbolRefAttr symbolRef = constOp.getValueAttr();
    llvm::StringRef symbolName = symbolRef.getLeafReference().empty()
                                     ? symbolRef.getRootReference().getValue()
                                     : symbolRef.getLeafReference().getValue();
    return resolveDirectCallableBySymbol(op, module, symbolName, constOp,
                                         typeConverter);
  }

  if (auto funcObject = mlir::dyn_cast_or_null<FuncObjectOp>(producer)) {
    return resolveDirectCallableBySymbol(
        op, module, funcObject.getTargetAttr().getValue(),
        funcObject.getOperation(), typeConverter);
  }

  if (auto makeFunction = mlir::dyn_cast_or_null<MakeFunctionOp>(producer)) {
    if (makeFunction.getDefaults() || makeFunction.getKwdefaults() ||
        makeFunction.getClosure())
      return {};
    return resolveDirectCallableBySymbol(
        op, module, makeFunction.getTargetAttr().getValue(),
        makeFunction.getOperation(), typeConverter);
  }

  return {};
}

static TupleCreateOp
collectDeadStaticPosargs(CallVectorOp op,
                         llvm::SmallVectorImpl<DecRefOp> &drops) {
  auto tupleCreate =
      stripBridgeCasts(op.getPosargs()).getDefiningOp<TupleCreateOp>();
  if (!tupleCreate)
    return {};

  llvm::SmallPtrSet<mlir::Operation *, 8> seenDrops;
  auto collectDrop = [&](mlir::Operation *user) -> bool {
    if (user == op.getOperation())
      return true;
    if (auto drop = mlir::dyn_cast<DecRefOp>(user)) {
      if (seenDrops.insert(drop.getOperation()).second)
        drops.push_back(drop);
      return true;
    }
    return false;
  };

  for (mlir::Operation *user : tupleCreate.getResult().getUsers()) {
    if (collectDrop(user))
      continue;
    if (auto cast = mlir::dyn_cast<mlir::UnrealizedConversionCastOp>(user)) {
      bool onlyCallOrDrop = true;
      for (mlir::Value result : cast->getResults()) {
        for (mlir::Operation *castUser : result.getUsers()) {
          if (collectDrop(castUser))
            continue;
          onlyCallOrDrop = false;
          break;
        }
        if (!onlyCallOrDrop)
          break;
      }
      if (onlyCallOrDrop)
        continue;
    }
    drops.clear();
    return {};
  }
  return tupleCreate;
}

static void eraseDeadBridgeUsers(mlir::Value value,
                                 mlir::RewriterBase &rewriter) {
  for (mlir::Operation *user : llvm::make_early_inc_range(value.getUsers())) {
    auto cast = mlir::dyn_cast<mlir::UnrealizedConversionCastOp>(user);
    if (!cast || !cast->use_empty())
      continue;
    rewriter.eraseOp(cast);
  }
}

static void cleanupDeadCallPack(TupleCreateOp tuple,
                                llvm::ArrayRef<DecRefOp> drops,
                                mlir::RewriterBase &rewriter) {
  for (DecRefOp drop : drops)
    if (drop)
      rewriter.eraseOp(drop);
  if (!tuple)
    return;
  eraseDeadBridgeUsers(tuple.getResult(), rewriter);
  if (tuple->use_empty()) {
    rewriter.eraseOp(tuple);
  } else {
    tuple->setAttr("ly.dead_call_pack",
                   mlir::UnitAttr::get(tuple->getContext()));
  }
}

static void collectArgTransferDrops(TupleCreateOp tupleCreate,
                                    mlir::Operation *callOp,
                                    llvm::SmallVectorImpl<DecRefOp> &drops) {
  if (!tupleCreate || !callOp || tupleCreate->getBlock() != callOp->getBlock())
    return;
  llvm::SmallPtrSet<mlir::Operation *, 8> seen;
  for (mlir::Value element : tupleCreate.getElements()) {
    for (mlir::Operation *user : element.getUsers()) {
      auto drop = mlir::dyn_cast<DecRefOp>(user);
      if (!drop || drop->getBlock() != callOp->getBlock())
        continue;
      if (!tupleCreate->isBeforeInBlock(drop) || !drop->isBeforeInBlock(callOp))
        continue;
      if (seen.insert(drop.getOperation()).second)
        drops.push_back(drop);
    }
  }
}

static bool hasDecRefUser(mlir::Value value) {
  value = stripBridgeCasts(value);
  for (mlir::Operation *user : value.getUsers())
    if (mlir::isa<DecRefOp>(user))
      return true;
  return false;
}

static void releaseOwnedStringCallArgs(TupleCreateOp tupleCreate,
                                       mlir::Operation *anchor,
                                       mlir::RewriterBase &rewriter) {
  if (!tupleCreate || !anchor)
    return;
  mlir::Operation *cursor = anchor;
  for (mlir::Value element : tupleCreate.getElements()) {
    if (!mlir::isa<StrType>(element.getType()))
      continue;
    if (hasDecRefUser(element))
      continue;
    rewriter.setInsertionPointAfter(cursor);
    auto drop = rewriter.create<DecRefOp>(element.getLoc(), element);
    cursor = drop.getOperation();
  }
}

static void moveAfter(mlir::Operation *anchor, llvm::ArrayRef<DecRefOp> drops) {
  mlir::Operation *cursor = anchor;
  for (DecRefOp drop : drops) {
    if (!drop || !cursor || drop->getBlock() != cursor->getBlock())
      continue;
    drop->moveAfter(cursor);
    cursor = drop.getOperation();
  }
}

static mlir::Value tupleDropValue(mlir::Value element) { return element; }

static mlir::Value stripLocalClassForwarding(mlir::Value value) {
  while (value) {
    if (auto cast = value.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
      if (cast->getNumOperands() != 1)
        return value;
      value = cast.getOperand(0);
      continue;
    }
    return value;
  }
  return value;
}

static bool isAlreadyDroppedByArgTransfer(mlir::Value element,
                                          llvm::ArrayRef<DecRefOp> drops) {
  mlir::Value payload = stripLocalClassForwarding(tupleDropValue(element));
  mlir::Value root = stripLocalClassForwarding(element);
  for (DecRefOp drop : drops) {
    if (!drop)
      continue;
    mlir::Value dropped = stripLocalClassForwarding(drop.getObject());
    if (dropped == root || dropped == payload)
      return true;
  }
  return false;
}

static bool hasExistingElementDrop(mlir::Value element) {
  mlir::Value payload = stripLocalClassForwarding(tupleDropValue(element));
  mlir::Value root = stripLocalClassForwarding(element);
  for (mlir::Value candidate : {element, payload, root}) {
    if (!candidate)
      continue;
    for (mlir::Operation *user : candidate.getUsers())
      if (mlir::isa<DecRefOp>(user))
        return true;
  }
  return false;
}

static bool hasElementDropAfter(mlir::Operation *anchor, mlir::Value element) {
  if (!anchor)
    return hasExistingElementDrop(element);

  mlir::Value payload = stripLocalClassForwarding(tupleDropValue(element));
  mlir::Value root = stripLocalClassForwarding(element);
  for (mlir::Value candidate : {element, payload, root}) {
    if (!candidate)
      continue;
    for (mlir::Operation *user : candidate.getUsers()) {
      if (!mlir::isa<DecRefOp>(user))
        continue;
      if (user->getBlock() != anchor->getBlock())
        return true;
      if (anchor->isBeforeInBlock(user))
        return true;
    }
  }
  return false;
}

static bool classValueHasOwnedProducer(mlir::Value value) {
  value = stripLocalClassForwarding(value);
  mlir::Operation *def = value ? value.getDefiningOp() : nullptr;
  if (!def)
    return false;
  return mlir::isa<ClassNewOp, AttrGetOp, AttrGetLocalOp, ClassPromoteOp,
                   CallVectorOp, CallOp>(def);
}

static bool hasNonDropUseAfter(mlir::Operation *anchor, mlir::Value value) {
  if (!anchor)
    return true;

  value = stripLocalClassForwarding(value);
  for (mlir::Operation *user : value.getUsers()) {
    if (user == anchor || mlir::isa<DecRefOp>(user))
      continue;
    if (user->getBlock() != anchor->getBlock())
      return true;
    if (anchor->isBeforeInBlock(user))
      return true;
  }
  return false;
}

static bool tupleSlotCanOwnRef(TupleCreateOp tupleCreate, unsigned index) {
  auto tupleType = mlir::dyn_cast<TupleType>(tupleCreate.getType());
  if (!tupleType || index >= tupleType.getElementTypes().size())
    return true;
  switch (container::Slot::policy(tupleType.getElementTypes()[index])) {
  case container::SlotPolicy::NativeInteger:
  case container::SlotPolicy::NativeBool:
  case container::SlotPolicy::NativeFloat:
    return false;
  case container::SlotPolicy::ObjectParts:
  case container::SlotPolicy::Unsupported:
    return true;
  }
  return true;
}

static void
closeDeadLocalClassArguments(TupleCreateOp tupleCreate, mlir::Operation *anchor,
                             llvm::ArrayRef<DecRefOp> argTransferDrops,
                             mlir::PatternRewriter &rewriter) {
  if (!tupleCreate || !anchor)
    return;

  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfter(anchor);
  for (auto [index, element] : llvm::enumerate(tupleCreate.getElements())) {
    if (!tupleSlotCanOwnRef(tupleCreate, static_cast<unsigned>(index)))
      continue;
    if (isAlreadyDroppedByArgTransfer(element, argTransferDrops))
      continue;
    if (hasElementDropAfter(anchor, element))
      continue;
    if (!consumesPyOwnedOperand(tupleCreate.getOperation(), element))
      continue;
    mlir::Value value = tupleDropValue(element);
    if (!mlir::isa<ClassType>(value.getType()))
      continue;
    if (!classValueHasOwnedProducer(value))
      continue;
    if (hasNonDropUseAfter(anchor, value))
      continue;
    auto drop = rewriter.create<DecRefOp>(value.getLoc(), value);
    drop->setAttr("ly.dead_call_pack_decref", rewriter.getUnitAttr());
  }
}

struct LoweredString {
  llvm::SmallVector<mlir::Value, 3> values;
  bool owned = false;

  bool split() const {
    if (values.size() != 2)
      return false;
    return object_abi::str_abi::Parts::isHeader(values[0].getType()) &&
           object_abi::str_abi::Parts::isBytes(values[1].getType());
  }

  mlir::Value single() const {
    return values.size() == 1 ? values.front() : mlir::Value();
  }
};

static bool isLoweredStringStorage(mlir::Value value) {
  return value && (object_abi::Type::isStorage(value.getType()) ||
                   object_abi::Type::isLoweredStorage(value.getType()));
}

static RuntimeAPI::Call emitHostPrintLiteral(mlir::Location loc,
                                             llvm::StringRef text,
                                             RuntimeAPI &runtime,
                                             llvm::StringRef printSymbol) {
  mlir::OpBuilder &builder = runtime.getBuilder();
  mlir::StringAttr literal = builder.getStringAttr(text);
  mlir::Value bytes = runtime.getByteLiteral(loc, literal);
  mlir::Value start = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
  mlir::Value length =
      runtime.getI64Constant(loc, static_cast<int64_t>(text.size()));
  llvm::SmallVector<mlir::Type, 3> resultTypes;
  object_abi::str_abi::Parts::storageTypes(builder.getContext(), resultTypes);
  RuntimeAPI::Call unicode = runtime.call(
      loc, RuntimeSymbols::kUnicodeFromBytes, mlir::TypeRange(resultTypes),
      mlir::ValueRange{bytes, start, length});
  llvm::SmallVector<mlir::Value, 3> parts(unicode.getResults());
  runtime.call(loc, printSymbol, mlir::Type(), mlir::ValueRange(parts));
  RuntimeAPI::Call release =
      runtime.call(loc, RuntimeSymbols::kUnicodeDecRef, mlir::Type(),
                   mlir::ValueRange(parts));
  release->setAttr(OwnershipContractAttrs::kAggregateRelease,
                   builder.getUnitAttr());
  return release;
}

static mlir::FailureOr<mlir::Operation *>
emitHostPrintBool(mlir::Location loc, mlir::Value value,
                  mlir::PatternRewriter &rewriter, RuntimeAPI &runtime,
                  llvm::StringRef printSymbol) {
  mlir::Value bit = value;
  if (mlir::isa<BoolType>(value.getType())) {
    bit = rewriter
              .create<CastToPrimOp>(loc, rewriter.getI1Type(), value,
                                    rewriter.getStringAttr("exact"))
              .getResult();
  }
  if (bit.getType() != rewriter.getI1Type())
    return mlir::failure();

  auto branch =
      rewriter.create<mlir::scf::IfOp>(loc, bit, /*withElseRegion=*/true);
  {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(branch.thenBlock());
    emitHostPrintLiteral(loc, "True", runtime, printSymbol);
  }
  {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(branch.elseBlock());
    emitHostPrintLiteral(loc, "False", runtime, printSymbol);
  }
  return branch.getOperation();
}

static llvm::SmallVector<mlir::Value, 3>
emitUnicodePartsLiteral(mlir::Location loc, mlir::StringAttr literal,
                        mlir::PatternRewriter &rewriter, RuntimeAPI &runtime) {
  mlir::Value bytes = runtime.getByteLiteral(loc, literal);
  mlir::Value start = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
  mlir::Value length =
      runtime.getI64Constant(loc, static_cast<int64_t>(literal.size()));
  llvm::SmallVector<mlir::Type, 3> resultTypes;
  object_abi::str_abi::Parts::storageTypes(rewriter.getContext(), resultTypes);
  RuntimeAPI::Call call = runtime.call(loc, RuntimeSymbols::kUnicodeFromBytes,
                                       mlir::TypeRange(resultTypes),
                                       mlir::ValueRange{bytes, start, length});
  return llvm::SmallVector<mlir::Value, 3>(call.getResults());
}

static mlir::FailureOr<LoweredString> lowerStringForImmediatePrint(
    mlir::Location loc, mlir::Value value, mlir::ModuleOp module,
    mlir::PatternRewriter &rewriter, RuntimeAPI &runtime) {
  if (isLoweredStringStorage(value))
    return LoweredString{{value}, /*owned=*/false};

  if (auto literal = value.getDefiningOp<StrConstantOp>()) {
    return LoweredString{
        emitUnicodePartsLiteral(loc, literal.getValueAttr(), rewriter, runtime),
        /*owned=*/true};
  }

  if (auto add = value.getDefiningOp<AddOp>()) {
    if (!mlir::isa<StrType>(add.getLhs().getType()) ||
        !mlir::isa<StrType>(add.getRhs().getType()))
      return mlir::failure();
    auto lhs = lowerStringForImmediatePrint(loc, add.getLhs(), module, rewriter,
                                            runtime);
    if (mlir::failed(lhs))
      return mlir::failure();
    auto rhs = lowerStringForImmediatePrint(loc, add.getRhs(), module, rewriter,
                                            runtime);
    if (mlir::failed(rhs))
      return mlir::failure();
    if (!lhs->split() || !rhs->split())
      return mlir::failure();
    llvm::SmallVector<mlir::Value, 4> operands;
    operands.append(lhs->values.begin(), lhs->values.end());
    operands.append(rhs->values.begin(), rhs->values.end());
    llvm::SmallVector<mlir::Type, 3> resultTypes;
    object_abi::str_abi::Parts::storageTypes(module.getContext(), resultTypes);
    RuntimeAPI::Call result =
        runtime.call(loc, RuntimeSymbols::kUnicodeConcat,
                     mlir::TypeRange(resultTypes), mlir::ValueRange(operands));
    if (lhs->owned)
      runtime.call(loc, RuntimeSymbols::kUnicodeDecRef, mlir::Type(),
                   mlir::ValueRange(lhs->values));
    if (rhs->owned)
      runtime.call(loc, RuntimeSymbols::kUnicodeDecRef, mlir::Type(),
                   mlir::ValueRange(rhs->values));
    return LoweredString{llvm::SmallVector<mlir::Value, 3>(result.getResults()),
                         /*owned=*/true};
  }

  return mlir::failure();
}

struct CallVectorLowering : public mlir::OpConversionPattern<CallVectorOp> {
  CallVectorLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<CallVectorOp>(converter, ctx,
                                                mlir::PatternBenefit(10)),
        typeConverter(converter) {}

  mlir::LogicalResult lowerBuiltinPrint(CallVectorOp op,
                                        mlir::PatternRewriter &rewriter) const {
    if (!mlir::isa<TupleEmptyOp>(
            stripBridgeCasts(op.getKwnames()).getDefiningOp()) ||
        !mlir::isa<TupleEmptyOp>(
            stripBridgeCasts(op.getKwvalues()).getDefiningOp()))
      return rewriter.notifyMatchFailure(
          op, "builtin print keyword arguments are not supported");

    auto posargsOp = stripBridgeCasts(op.getPosargs()).getDefiningOp();
    TupleCreateOp staticPosargs;
    llvm::SmallVector<mlir::Value> elements;
    if (auto tupleCreate = mlir::dyn_cast_or_null<TupleCreateOp>(posargsOp)) {
      staticPosargs = tupleCreate;
      elements.append(tupleCreate.getElements().begin(),
                      tupleCreate.getElements().end());
    } else if (!mlir::isa<TupleEmptyOp>(posargsOp)) {
      return rewriter.notifyMatchFailure(op,
                                         "builtin print requires static args");
    }

    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return mlir::failure();
    RuntimeAPI runtime(module, rewriter, typeConverter);
    llvm::StringRef printSymbol = isBuiltinPrintRawCallable(op.getCallable())
                                      ? RuntimeSymbols::kUnicodePrint
                                      : RuntimeSymbols::kUnicodePrintLine;

    if (elements.size() != 1)
      return rewriter.notifyMatchFailure(
          op, "builtin print currently requires one statically resolved arg");

    mlir::Value literalCandidate = elements.front();
    auto emitStaticPrintLiteral =
        [&](llvm::StringRef text,
            mlir::Operation *producer) -> mlir::LogicalResult {
      llvm::SmallVector<DecRefOp> deadPosargDrops;
      llvm::SmallVector<DecRefOp> sourceTransferDrops;
      TupleCreateOp deadPosargs = collectDeadStaticPosargs(op, deadPosargDrops);
      if (!deadPosargs)
        deadPosargs = staticPosargs;
      collectArgTransferDrops(deadPosargs, op.getOperation(),
                              sourceTransferDrops);
      emitHostPrintLiteral(op.getLoc(), text, runtime, printSymbol);
      if (op.getNumResults() != 0) {
        if (canDropNoneResult(op)) {
          eraseNoneResultUsers(op, rewriter);
        } else {
          auto none = rewriter.create<NoneOp>(op.getLoc(),
                                              NoneType::get(op.getContext()));
          rewriter.replaceOp(op, none.getResult());
          cleanupDeadCallPack(deadPosargs, deadPosargDrops, rewriter);
          if (producer && producer->use_empty())
            rewriter.eraseOp(producer);
          return mlir::success();
        }
      }
      rewriter.eraseOp(op);
      for (DecRefOp drop : sourceTransferDrops)
        if (drop)
          rewriter.eraseOp(drop);
      cleanupDeadCallPack(deadPosargs, deadPosargDrops, rewriter);
      if (producer && producer->use_empty())
        rewriter.eraseOp(producer);
      return mlir::success();
    };
    if (auto literal = literalCandidate.getDefiningOp<StrConstantOp>()) {
      return emitStaticPrintLiteral(literal.getValue(), literal.getOperation());
    }
    if (auto literal = literalCandidate.getDefiningOp<IntConstantOp>()) {
      return emitStaticPrintLiteral(literal.getValue(), literal.getOperation());
    }
    if (auto repr = literalCandidate.getDefiningOp<ReprOp>()) {
      if (auto literal = stripBridgeCasts(repr.getInput())
                             .getDefiningOp<IntConstantOp>()) {
        return emitStaticPrintLiteral(literal.getValue(), repr.getOperation());
      }
    }

    {
      llvm::SmallVector<DecRefOp> deadPosargDrops;
      llvm::SmallVector<DecRefOp> sourceTransferDrops;
      TupleCreateOp deadPosargs = collectDeadStaticPosargs(op, deadPosargDrops);
      if (!deadPosargs)
        deadPosargs = staticPosargs;
      collectArgTransferDrops(deadPosargs, op.getOperation(),
                              sourceTransferDrops);
      mlir::Value printable = elements.front();
      bool releasePrintable = false;
      auto finishDirectPrint =
          [&](mlir::Operation *printAnchor) -> mlir::LogicalResult {
        moveAfter(printAnchor, sourceTransferDrops);
        if (op.getNumResults() != 0) {
          if (canDropNoneResult(op)) {
            eraseNoneResultUsers(op, rewriter);
          } else {
            auto none = rewriter.create<NoneOp>(op.getLoc(),
                                                NoneType::get(op.getContext()));
            rewriter.replaceOp(op, none.getResult());
            cleanupDeadCallPack(deadPosargs, deadPosargDrops, rewriter);
            return mlir::success();
          }
        }
        rewriter.eraseOp(op);
        cleanupDeadCallPack(deadPosargs, deadPosargDrops, rewriter);
        return mlir::success();
      };
      if (!isPyStrType(printable.getType())) {
        if (mlir::isa<BoolType>(printable.getType())) {
          mlir::FailureOr<mlir::Operation *> printAnchor = emitHostPrintBool(
              op.getLoc(), printable, rewriter, runtime, printSymbol);
          if (mlir::failed(printAnchor))
            return rewriter.notifyMatchFailure(
                op, "failed to lower bool print to host print");
          return finishDirectPrint(*printAnchor);
        }
        if (mlir::isa<ClassType>(printable.getType())) {
          printable =
              rewriter
                  .create<ReprOp>(op.getLoc(), StrType::get(op.getContext()),
                                  printable)
                  .getResult();
          releasePrintable = true;
        } else {
          printable =
              rewriter
                  .create<ReprOp>(op.getLoc(), StrType::get(op.getContext()),
                                  printable)
                  .getResult();
          releasePrintable = true;
        }
      }
      mlir::Operation *printAnchor = nullptr;
      mlir::Value strStorage = printable;
      llvm::SmallVector<mlir::Value, 3> strPartsStorage;
      bool partsPrintStorage = false;
      bool helperOwned = false;
      if (mlir::isa<StrType>(printable.getType())) {
        auto lowered = lowerStringForImmediatePrint(op.getLoc(), printable,
                                                    module, rewriter, runtime);
        if (mlir::succeeded(lowered)) {
          helperOwned = lowered->owned;
          if (lowered->split()) {
            strPartsStorage.append(lowered->values.begin(),
                                   lowered->values.end());
            partsPrintStorage = true;
          } else if (mlir::Value single = lowered->single()) {
            strStorage = single;
          } else {
            return rewriter.notifyMatchFailure(
                op, "lowered string has unsupported storage shape");
          }
        }
      }
      if (partsPrintStorage) {
        auto printCall = runtime.call(op.getLoc(), printSymbol, mlir::Type(),
                                      mlir::ValueRange(strPartsStorage));
        printAnchor = printCall.getOperation();
      } else if (strStorage == printable &&
                 mlir::isa<StrType>(printable.getType())) {
        llvm::SmallVector<mlir::Type, 3> strTypes;
        object_abi::str_abi::Parts::storageTypes(op.getContext(), strTypes);
        auto cast = rewriter.create<mlir::UnrealizedConversionCastOp>(
            op.getLoc(), mlir::TypeRange(strTypes),
            mlir::ValueRange{printable});
        strPartsStorage.append(cast.getResults().begin(),
                               cast.getResults().end());
        auto printCall = runtime.call(op.getLoc(), printSymbol, mlir::Type(),
                                      mlir::ValueRange(strPartsStorage));
        printAnchor = printCall.getOperation();
        partsPrintStorage = true;
      } else {
        return rewriter.notifyMatchFailure(
            op, "print requires unicode header/payload parts");
      }
      bool releasePrintStorage = helperOwned;
      if (!deadPosargDrops.empty() && !releasePrintable &&
          object_abi::Type::isStorageLike(strStorage.getType())) {
        releasePrintStorage = true;
      }
      if (releasePrintStorage) {
        if (printAnchor)
          rewriter.setInsertionPointAfter(printAnchor);
        if (!partsPrintStorage)
          return rewriter.notifyMatchFailure(
              op, "print release requires unicode header/payload parts");
        RuntimeAPI::Call releaseCall =
            runtime.call(op.getLoc(), RuntimeSymbols::kUnicodeDecRef,
                         mlir::Type(), mlir::ValueRange(strPartsStorage));
        releaseCall->setAttr(OwnershipContractAttrs::kAggregateRelease,
                             rewriter.getUnitAttr());
        printAnchor = releaseCall.getOperation();
      }
      moveAfter(printAnchor, sourceTransferDrops);
      if (op.getNumResults() != 0) {
        if (canDropNoneResult(op)) {
          eraseNoneResultUsers(op, rewriter);
        } else {
          auto none = rewriter.create<NoneOp>(op.getLoc(),
                                              NoneType::get(op.getContext()));
          rewriter.replaceOp(op, none.getResult());
          cleanupDeadCallPack(deadPosargs, deadPosargDrops, rewriter);
          return mlir::success();
        }
      }
      rewriter.eraseOp(op);
      cleanupDeadCallPack(deadPosargs, deadPosargDrops, rewriter);
      return mlir::success();
    }
  }

  mlir::LogicalResult lowerViaRuntime(CallVectorOp op,
                                      mlir::PatternRewriter &rewriter) const {
    return rewriter.notifyMatchFailure(
        op, "dynamic vectorcall requires static call resolution");
  }

  mlir::LogicalResult lowerCallVector(CallVectorOp op,
                                      mlir::PatternRewriter &rewriter) const {
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return mlir::failure();

    if (isBuiltinPrintCallable(op.getCallable()))
      return lowerBuiltinPrint(op, rewriter);

    DirectCallable direct = resolveDirectCallable(op, module, typeConverter);
    auto func = direct.func;
    if (!func && !direct.functionType)
      return lowerViaRuntime(op, rewriter);

    auto callableType = mlir::dyn_cast<FuncType>(op.getCallable().getType());
    if (!callableType)
      return lowerViaRuntime(op, rewriter);
    FuncSignatureType signature = callableType.getSignature();

    llvm::SmallVector<mlir::Value, 4> kwNameElements;
    llvm::SmallVector<mlir::Value, 4> kwValueElements;
    if (!collectTupleElements(op.getKwnames(), kwNameElements) ||
        !collectTupleElements(op.getKwvalues(), kwValueElements))
      return lowerViaRuntime(op, rewriter);
    if (kwNameElements.size() != kwValueElements.size())
      return lowerViaRuntime(op, rewriter);

    if (func && !signature.hasVararg() && !signature.hasKwarg() &&
        kwNameElements.empty() && canUseVoidHelper(op, func)) {
      auto posargsOp = stripBridgeCasts(op.getPosargs()).getDefiningOp();
      llvm::SmallVector<mlir::Value> helperOperands;
      llvm::SmallVector<DecRefOp> argTransferDrops;
      llvm::SmallVector<DecRefOp> deadPosargDrops;
      TupleCreateOp deadPosargs = collectDeadStaticPosargs(op, deadPosargDrops);
      if (auto tupleCreate = mlir::dyn_cast_or_null<TupleCreateOp>(posargsOp)) {
        if (!deadPosargs)
          deadPosargs = tupleCreate;
        auto helperFunc = resolvePreferredDirectHelper(
            func, tupleCreate.getElements(), module, /*allowVoidHelper=*/true);
        if (!helperFunc)
          return rewriter.notifyMatchFailure(op, "missing void helper");
        auto helperType = helperFunc.getFunctionType();
        if (mlir::failed(::py::appendFlattenedCallOperands(
                op.getLoc(), tupleCreate.getElements(), helperType,
                helperType.getNumInputs(), helperOperands, rewriter,
                typeConverter)))
          return lowerViaRuntime(op, rewriter);
        collectArgTransferDrops(tupleCreate, op.getOperation(),
                                argTransferDrops);
        auto helperCall = rewriter.create<mlir::func::CallOp>(
            op.getLoc(), helperFunc, helperOperands);
        moveAfter(helperCall.getOperation(), argTransferDrops);
        releaseOwnedStringCallArgs(tupleCreate, helperCall.getOperation(),
                                   rewriter);
        closeDeadLocalClassArguments(tupleCreate, helperCall.getOperation(),
                                     argTransferDrops, rewriter);
      } else if (mlir::isa<TupleEmptyOp>(posargsOp)) {
        auto helperFunc =
            resolvePreferredDirectHelper(func, mlir::ValueRange{}, module,
                                         /*allowVoidHelper=*/true);
        if (!helperFunc)
          return rewriter.notifyMatchFailure(op, "missing void helper");
        auto helperType = helperFunc.getFunctionType();
        if (helperType.getNumInputs() != 0)
          return lowerViaRuntime(op, rewriter);
        rewriter.create<mlir::func::CallOp>(op.getLoc(), helperFunc,
                                            helperOperands);
      } else {
        return lowerViaRuntime(op, rewriter);
      }
      eraseNoneResultUsers(op, rewriter);
      rewriter.eraseOp(op);
      cleanupDeadCallPack(deadPosargs, deadPosargDrops, rewriter);
      if (direct.reference && direct.reference->use_empty())
        rewriter.eraseOp(direct.reference);
      return mlir::success();
    }

    auto posargsOp = stripBridgeCasts(op.getPosargs()).getDefiningOp();
    if (auto tupleCreate = mlir::dyn_cast_or_null<TupleCreateOp>(posargsOp)) {
      if (func) {
        func = resolvePreferredDirectHelper(func, tupleCreate.getElements(),
                                            module,
                                            /*allowVoidHelper=*/false);
        if (func)
          direct.functionType = func.getFunctionType();
      }
    }

    auto funcType = direct.functionType;
    llvm::SmallVector<mlir::Value> callOperands;
    llvm::SmallVector<DecRefOp> argTransferDrops;
    unsigned directInputCount = funcType.getNumInputs();
    TupleCreateOp staticPosargs;
    llvm::SmallVector<DecRefOp> deadPosargDrops;
    TupleCreateOp deadPosargs = collectDeadStaticPosargs(op, deadPosargDrops);
    if (auto tupleCreate = mlir::dyn_cast_or_null<TupleCreateOp>(posargsOp)) {
      staticPosargs = tupleCreate;
      if (!deadPosargs)
        deadPosargs = tupleCreate;
      llvm::SmallVector<mlir::Value, 8> directElements;
      if (mlir::failed(packVectorArgsForDirectCall(
              op, tupleCreate.getElements(), kwNameElements, kwValueElements,
              direct.argNames, direct.kwonlyNames, directElements, rewriter)))
        return lowerViaRuntime(op, rewriter);
      if (mlir::failed(::py::appendFlattenedCallOperands(
              op.getLoc(), directElements, funcType, directInputCount,
              callOperands, rewriter, typeConverter)))
        return lowerViaRuntime(op, rewriter);
      collectArgTransferDrops(tupleCreate, op.getOperation(), argTransferDrops);
    } else if (mlir::isa<TupleEmptyOp>(posargsOp)) {
      llvm::SmallVector<mlir::Value, 1> directElements;
      if (mlir::failed(packVectorArgsForDirectCall(
              op, mlir::ValueRange{}, kwNameElements, kwValueElements,
              direct.argNames, direct.kwonlyNames, directElements, rewriter)))
        return lowerViaRuntime(op, rewriter);
      if (directElements.empty()) {
        if (directInputCount != 0)
          return lowerViaRuntime(op, rewriter);
      } else if (mlir::failed(::py::appendFlattenedCallOperands(
                     op.getLoc(), directElements, funcType, directInputCount,
                     callOperands, rewriter, typeConverter))) {
        return lowerViaRuntime(op, rewriter);
      }
    } else {
      return lowerViaRuntime(op, rewriter);
    }

    mlir::func::CallOp call;
    if (func) {
      call =
          rewriter.create<mlir::func::CallOp>(op.getLoc(), func, callOperands);
    } else {
      call = rewriter.create<mlir::func::CallOp>(
          op.getLoc(), direct.symbolName, funcType.getResults(), callOperands);
    }
    moveAfter(call.getOperation(), argTransferDrops);
    releaseOwnedStringCallArgs(staticPosargs, call.getOperation(), rewriter);
    closeDeadLocalClassArguments(staticPosargs, call.getOperation(),
                                 argTransferDrops, rewriter);

    if (call.getNumResults() == 0) {
      if (op.getNumResults() == 1 &&
          mlir::isa<NoneType>(op.getResult(0).getType())) {
        if (canDropNoneResult(op)) {
          eraseNoneResultUsers(op, rewriter);
          rewriter.eraseOp(op);
          cleanupDeadCallPack(deadPosargs, deadPosargDrops, rewriter);
        } else {
          auto none = rewriter.create<NoneOp>(op.getLoc(),
                                              NoneType::get(op.getContext()));
          rewriter.replaceOp(op, none.getResult());
          cleanupDeadCallPack(deadPosargs, deadPosargDrops, rewriter);
        }
      } else {
        rewriter.eraseOp(op);
        cleanupDeadCallPack(deadPosargs, deadPosargDrops, rewriter);
      }
    } else {
      llvm::SmallVector<mlir::Value> logicalResults;
      materializeLogicalResults(op.getLoc(), op.getResultTypes(),
                                call.getResults(), logicalResults,
                                typeConverter, rewriter);
      rewriter.replaceOp(op, logicalResults);
      cleanupDeadCallPack(deadPosargs, deadPosargDrops, rewriter);
    }

    if (direct.reference && direct.reference->use_empty())
      rewriter.eraseOp(direct.reference);
    return mlir::success();
  }

  mlir::LogicalResult
  matchAndRewrite(CallVectorOp op, OneToNOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    return lowerCallVector(op, rewriter);
  }

  PyLLVMTypeConverter &typeConverter;
};

struct CallVectorRewrite : public mlir::OpRewritePattern<CallVectorOp> {
  CallVectorRewrite(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpRewritePattern<CallVectorOp>(ctx, mlir::PatternBenefit(90)),
        typeConverter(converter) {}

  mlir::LogicalResult
  matchAndRewrite(CallVectorOp op,
                  mlir::PatternRewriter &rewriter) const override {
    CallVectorLowering lowerer(typeConverter, rewriter.getContext());
    return lowerer.lowerCallVector(op, rewriter);
  }

  PyLLVMTypeConverter &typeConverter;
};

struct BuiltinPrintVectorRewrite : public mlir::OpRewritePattern<CallVectorOp> {
  BuiltinPrintVectorRewrite(PyLLVMTypeConverter &converter,
                            mlir::MLIRContext *ctx)
      : mlir::OpRewritePattern<CallVectorOp>(ctx, mlir::PatternBenefit(100)),
        typeConverter(converter) {}

  mlir::LogicalResult
  matchAndRewrite(CallVectorOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!isBuiltinPrintCallable(op.getCallable()))
      return rewriter.notifyMatchFailure(op, "not a builtin print call");

    CallVectorLowering lowerer(typeConverter, rewriter.getContext());
    return lowerer.lowerBuiltinPrint(op, rewriter);
  }

  PyLLVMTypeConverter &typeConverter;
};

struct CallOpLowering : public mlir::OpConversionPattern<CallOp> {
  CallOpLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<CallOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(CallOp op, OneToNOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    return rewriter.notifyMatchFailure(
        op, "dynamic call requires static call resolution");
  }
};

} // namespace

namespace lowering::call::direct::Patterns {
void populate(PyLLVMTypeConverter &typeConverter,
              mlir::RewritePatternSet &patterns) {
  auto *ctx = patterns.getContext();
  patterns.add<BuiltinPrintVectorRewrite, CallVectorRewrite, CallVectorLowering,
               CallOpLowering>(typeConverter, ctx);
}
} // namespace lowering::call::direct::Patterns

} // namespace py
