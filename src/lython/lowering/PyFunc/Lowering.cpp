// This file implements lowering patterns for py.func, py.return, and
// py.func_object operations. These patterns convert Python-style function
// definitions to standard MLIR func dialect operations.

#include "Common/Container.h"
#include "Common/LoweringUtils.h"
#include "Common/Object.h"
#include "Common/RuntimeSupport.h"
#include "Common/SlotUtils.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/FormatVariadic.h"

#include <algorithm>

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

namespace py {
ClassOp lookupClassSymbol(mlir::Operation *from, ClassType classType);

namespace {

namespace lowering::func::signature {

mlir::LogicalResult
translate(FuncSignatureType sig, const PyLLVMTypeConverter &typeConverter,
          llvm::SmallVectorImpl<mlir::Type> &pyInputs,
          llvm::SmallVectorImpl<mlir::Type> &convertedInputs,
          llvm::SmallVectorImpl<mlir::Type> &convertedResults,
          mlir::Operation *emitOnError) {
  pyInputs.clear();
  convertedInputs.clear();
  convertedResults.clear();

  auto appendConverted =
      [&](mlir::Type ty,
          llvm::SmallVectorImpl<mlir::Type> &storage) -> mlir::LogicalResult {
    llvm::SmallVector<mlir::Type, 4> converted;
    if (mlir::failed(typeConverter.convertType(ty, converted)) ||
        converted.empty()) {
      emitOnError->emitError("failed to convert type ") << ty;
      return mlir::failure();
    }
    storage.append(converted.begin(), converted.end());
    return mlir::success();
  };

  auto positional = sig.getPositionalTypes();
  pyInputs.append(positional.begin(), positional.end());
  auto kwonly = sig.getKwOnlyTypes();
  pyInputs.append(kwonly.begin(), kwonly.end());
  if (sig.hasVararg())
    pyInputs.push_back(sig.getVarargType());
  if (sig.hasKwarg())
    pyInputs.push_back(sig.getKwargType());

  for (mlir::Type ty : pyInputs)
    if (mlir::failed(appendConverted(ty, convertedInputs)))
      return mlir::failure();

  for (mlir::Type result : sig.getResultTypes()) {
    if (mlir::isa<NoneType>(result))
      continue;
    if (mlir::failed(appendConverted(result, convertedResults)))
      return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult
appendClosureInputs(mlir::ArrayAttr closureTypesAttr,
                    const PyLLVMTypeConverter &typeConverter,
                    llvm::SmallVectorImpl<mlir::Type> &pyInputs,
                    llvm::SmallVectorImpl<mlir::Type> &convertedInputs,
                    mlir::Operation *emitOnError) {
  if (!closureTypesAttr)
    return mlir::success();

  auto appendConverted = [&](mlir::Type ty) -> mlir::LogicalResult {
    llvm::SmallVector<mlir::Type, 4> converted;
    if (mlir::failed(typeConverter.convertType(ty, converted)) ||
        converted.empty()) {
      emitOnError->emitError("failed to convert closure type ") << ty;
      return mlir::failure();
    }
    pyInputs.push_back(ty);
    convertedInputs.append(converted.begin(), converted.end());
    return mlir::success();
  };

  for (mlir::Attribute attr : closureTypesAttr) {
    auto typeAttr = mlir::dyn_cast<mlir::TypeAttr>(attr);
    if (!typeAttr) {
      emitOnError->emitError(
          "closure_types must contain only TypeAttr elements");
      return mlir::failure();
    }
    if (mlir::failed(appendConverted(typeAttr.getValue())))
      return mlir::failure();
  }

  return mlir::success();
}

} // namespace lowering::func::signature

namespace lowering::func::args::ABI {

static void setUnitAttr(mlir::func::FuncOp func, unsigned argIndex,
                        llvm::StringRef name) {
  if (argIndex >= func.getNumArguments())
    return;
  func.setArgAttr(argIndex, name, mlir::UnitAttr::get(func.getContext()));
}

static void setStringAttr(mlir::func::FuncOp func, unsigned argIndex,
                          llvm::StringRef name, llvm::StringRef value) {
  if (argIndex >= func.getNumArguments())
    return;
  func.setArgAttr(argIndex, name,
                  mlir::StringAttr::get(func.getContext(), value));
}

static bool hasObjectHeaderPart(mlir::Type type) {
  return mlir::isa<IntType, StrType, ExceptionType, ClassType>(type);
}

static std::optional<llvm::StringRef> containerKind(mlir::Type type) {
  if (mlir::isa<ListType>(type))
    return ContainerSafetyAttrs::kKindList;
  if (mlir::isa<TupleType>(type))
    return ContainerSafetyAttrs::kKindTuple;
  if (mlir::isa<DictType>(type))
    return ContainerSafetyAttrs::kKindDict;
  return std::nullopt;
}

static llvm::StringRef containerComponent(mlir::Type type, unsigned slot) {
  if (slot == kTupleHeaderComponent)
    return ContainerSafetyAttrs::kComponentHeader;
  if (mlir::isa<TupleType>(type)) {
    if (slot == kTupleItemsComponent)
      return ContainerSafetyAttrs::kComponentItems;
    return {};
  }
  if (mlir::isa<ListType>(type)) {
    switch (slot) {
    case kListLockComponent:
      return ContainerSafetyAttrs::kComponentLock;
    case kListItemsComponent:
      return ContainerSafetyAttrs::kComponentItems;
    default:
      break;
    }
  }
  if (mlir::isa<DictType>(type)) {
    switch (slot) {
    case kDictLockComponent:
      return ContainerSafetyAttrs::kComponentLock;
    case kDictKeysComponent:
      return ContainerSafetyAttrs::kComponentKeys;
    case kDictValuesComponent:
      return ContainerSafetyAttrs::kComponentValues;
    case kDictStatesComponent:
      return ContainerSafetyAttrs::kComponentStates;
    default:
      break;
    }
  }
  return {};
}

static void markObjectHeader(mlir::func::FuncOp func, mlir::Type logicalType,
                             llvm::ArrayRef<mlir::Type> converted,
                             unsigned flattenedIndex) {
  if (!hasObjectHeaderPart(logicalType) || converted.empty() ||
      (!object_abi::Header::isOwned(converted.front()) &&
       !object_abi::exception_abi::Header::isOwned(converted.front())))
    return;
  setUnitAttr(func, flattenedIndex, OwnershipContractAttrs::kObjectHeader);
}

static void markContainerDescriptor(mlir::func::FuncOp func,
                                    mlir::Type logicalType,
                                    unsigned logicalIndex,
                                    unsigned flattenedIndex,
                                    unsigned convertedWidth) {
  auto kind = containerKind(logicalType);
  if (!kind)
    return;
  std::string group =
      (llvm::Twine(*kind) + ".arg" + llvm::Twine(logicalIndex)).str();
  for (unsigned slot = 0; slot < convertedWidth; ++slot) {
    llvm::StringRef component = containerComponent(logicalType, slot);
    if (component.empty())
      continue;
    unsigned argIndex = flattenedIndex + slot;
    setStringAttr(func, argIndex, ContainerSafetyAttrs::kDescriptorGroup,
                  group);
    setStringAttr(func, argIndex, ContainerSafetyAttrs::kDescriptorKind, *kind);
    setStringAttr(func, argIndex, ContainerSafetyAttrs::kDescriptorComponent,
                  component);
  }
}

void mark(mlir::func::FuncOp func, llvm::ArrayRef<mlir::Type> pyInputs,
          const PyLLVMTypeConverter &typeConverter) {
  if (!func)
    return;

  unsigned flattenedIndex = 0;
  unsigned logicalIndex = 0;
  for (mlir::Type inputType : pyInputs) {
    llvm::SmallVector<mlir::Type, 4> converted;
    if (mlir::failed(typeConverter.convertType(inputType, converted)) ||
        converted.empty())
      return;

    markObjectHeader(func, inputType, converted, flattenedIndex);
    markContainerDescriptor(func, inputType, logicalIndex, flattenedIndex,
                            static_cast<unsigned>(converted.size()));

    if (mlir::isa<CoroutineType, FutureType, TaskType>(inputType) &&
        converted.size() >= 2) {
      async_runtime::RuntimeHandle::markArgument(func.getOperation(),
                                                 flattenedIndex);
      async_runtime::ExceptionCell::markArgument(func.getOperation(),
                                                 flattenedIndex + 1);
      if (mlir::isa<TaskType>(inputType) && converted.size() >= 3)
        async_runtime::CancelFlag::markArgument(func.getOperation(),
                                                flattenedIndex + 2);
    }
    if (isCompilerOwnedMemRefContainerType(inputType)) {
      mlir::Builder builder(func.getContext());
      for (auto [offset, convertedType] : llvm::enumerate(converted)) {
        if (!mlir::isa<mlir::LLVM::LLVMPointerType>(convertedType))
          continue;
        if (flattenedIndex + static_cast<unsigned>(offset) >=
            func.getNumArguments())
          continue;
        func.setArgAttr(flattenedIndex + static_cast<unsigned>(offset),
                        OwnershipContractAttrs::kNonObjectPointer,
                        builder.getUnitAttr());
      }
    }
    flattenedIndex += static_cast<unsigned>(converted.size());
    ++logicalIndex;
  }
}

} // namespace lowering::func::args::ABI

namespace lowering::func::result::Ownership {

static bool ownsMemRefResultSlot(mlir::Type logicalType, unsigned slot) {
  if (mlir::isa<IntType, StrType>(logicalType))
    return slot == 0;
  if (mlir::isa<ClassType>(logicalType))
    return slot == 0;
  return slot == 0 && isCompilerOwnedMemRefContainerType(logicalType);
}

void mark(mlir::func::FuncOp func, llvm::ArrayRef<mlir::Type> logicalResults,
          const PyLLVMTypeConverter &typeConverter) {
  if (!func)
    return;
  llvm::SmallVector<int64_t, 4> ownedResults;
  llvm::SmallVector<int64_t, 4> borrowedResults;
  unsigned flattenedIndex = 0;
  for (mlir::Type logicalType : logicalResults) {
    llvm::SmallVector<mlir::Type, 4> converted;
    if (mlir::failed(typeConverter.convertType(logicalType, converted)) ||
        converted.empty())
      return;

    bool immortalNone = mlir::isa<NoneType>(logicalType);
    for (auto [slot, loweredType] : llvm::enumerate(converted)) {
      // Raw LLVM pointer results are backend handles or host-boundary carriers,
      // not LyObject ownership. Object-family values expose ownership through
      // their header memref slot and payload descriptors.
      if (!immortalNone && mlir::isa<mlir::MemRefType>(loweredType) &&
          ownsMemRefResultSlot(logicalType, static_cast<unsigned>(slot))) {
        ownedResults.push_back(static_cast<int64_t>(flattenedIndex));
      }
      ++flattenedIndex;
    }
  }
  if (ownedResults.empty() && borrowedResults.empty())
    return;

  mlir::OpBuilder builder(func.getContext());
  auto setIndexArrayAttr = [&](llvm::StringRef name,
                               llvm::ArrayRef<int64_t> indices) {
    if (indices.empty())
      return;
    llvm::SmallVector<mlir::Attribute, 4> attrs;
    attrs.reserve(indices.size());
    for (int64_t index : indices)
      attrs.push_back(builder.getI64IntegerAttr(index));
    func->setAttr(name, builder.getArrayAttr(attrs));
  };
  setIndexArrayAttr(OwnershipContractAttrs::kOwnedResults, ownedResults);
  setIndexArrayAttr(OwnershipContractAttrs::kBorrowedResults, borrowedResults);
}

} // namespace lowering::func::result::Ownership

namespace lowering::func::void_helper {

bool shouldUse(FuncOp op, FuncSignatureType sig) {
  auto results = sig.getResultTypes();
  if (results.size() != 1 || !mlir::isa<NoneType>(results.front()))
    return false;
  return static_cast<bool>(op->getAttr("mutates_self")) ||
         static_cast<bool>(op->getAttr("init_method"));
}

} // namespace lowering::func::void_helper

namespace lowering::func::attrs {

static void copyNamed(mlir::Operation *from, mlir::Operation *to,
                      llvm::StringRef name) {
  if (mlir::Attribute attr = from->getAttr(name))
    to->setAttr(name, attr);
}

static void copyAll(mlir::Operation *from, mlir::Operation *to,
                    llvm::ArrayRef<llvm::StringLiteral> names) {
  for (llvm::StringLiteral name : names)
    copyNamed(from, to, name);
}

void copyPy(FuncOp from, mlir::func::FuncOp to) {
  static constexpr llvm::StringLiteral kAttrs[] = {"arg_names",
                                                   "mutates_self",
                                                   "init_method",
                                                   "nothrow",
                                                   "maythrow",
                                                   "ly.publishes_args",
                                                   "ly.captures_published",
                                                   "ly.returns_published",
                                                   "ly.readonly_args",
                                                   "ly.mutable_args",
                                                   "kwonly_names",
                                                   "closure_types"};
  copyAll(from.getOperation(), to.getOperation(), kAttrs);
}

void copySpecialized(mlir::func::FuncOp from, mlir::func::FuncOp to) {
  static constexpr llvm::StringLiteral kAttrs[] = {"arg_names",
                                                   "mutates_self",
                                                   "init_method",
                                                   "nothrow",
                                                   "maythrow",
                                                   "ly.publishes_args",
                                                   "ly.captures_published",
                                                   "ly.returns_published",
                                                   "ly.readonly_args",
                                                   "ly.mutable_args",
                                                   "ly.local_self_arg0",
                                                   "ly.zero_initialized_self",
                                                   "kwonly_names",
                                                   "closure_types"};
  copyAll(from.getOperation(), to.getOperation(), kAttrs);
}

} // namespace lowering::func::attrs

namespace lowering::func::published_borrow {

bool specialize(mlir::func::FuncOp func, unsigned argIndex) {
  if (func.getBody().empty())
    return false;

  mlir::Block &entry = func.getBody().front();
  if (argIndex >= entry.getNumArguments())
    return false;

  mlir::Value borrowedArg = entry.getArgument(argIndex);
  llvm::SmallVector<PublishOp> publishes;
  func.walk([&](PublishOp publish) {
    if (publish.getInput() == borrowedArg)
      publishes.push_back(publish);
  });

  if (publishes.empty())
    return false;

  bool changed = false;
  for (PublishOp publish : publishes) {
    llvm::SmallVector<DecRefOp> decRefs;
    llvm::SmallVector<mlir::OpOperand *> forwardedUses;
    for (mlir::OpOperand &use : publish.getResult().getUses()) {
      if (auto decRef = mlir::dyn_cast<DecRefOp>(use.getOwner())) {
        decRefs.push_back(decRef);
        continue;
      }
      forwardedUses.push_back(&use);
    }

    for (mlir::OpOperand *use : forwardedUses)
      use->set(borrowedArg);
    for (DecRefOp decRef : decRefs)
      decRef.erase();

    publish.erase();
    changed = true;
  }

  return changed;
}

} // namespace lowering::func::published_borrow

// FuncOpLowering: py.func -> func.func

struct FuncOpLowering : public mlir::OpConversionPattern<FuncOp> {
  FuncOpLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<FuncOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(FuncOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto nameAttr = op->getAttrOfType<mlir::StringAttr>("sym_name");
    if (!nameAttr)
      return rewriter.notifyMatchFailure(op, "missing sym_name");

    // Skip builtin print functions - they are handled specially.
    if (nameAttr.getValue() == "__builtin_print" ||
        nameAttr.getValue() == "__builtin_print_raw") {
      rewriter.eraseOp(op);
      return mlir::success();
    }

    auto sigAttr = op->getAttrOfType<mlir::TypeAttr>("function_type");
    if (!sigAttr)
      return rewriter.notifyMatchFailure(op, "missing function_type attr");

    auto sig = mlir::dyn_cast<FuncSignatureType>(sigAttr.getValue());
    if (!sig)
      return rewriter.notifyMatchFailure(op, "expected FuncSignatureType");

    auto *tc = static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    llvm::SmallVector<mlir::Type, 8> pyInputTypes;
    llvm::SmallVector<mlir::Type, 8> llvmInputTypes;
    llvm::SmallVector<mlir::Type, 4> llvmResultTypes;
    if (mlir::failed(lowering::func::signature::translate(
            sig, *tc, pyInputTypes, llvmInputTypes, llvmResultTypes, op)))
      return mlir::failure();
    llvm::SmallVector<mlir::Type, 4> pyResultTypes(sig.getResultTypes().begin(),
                                                   sig.getResultTypes().end());
    unsigned visibleInputCount = static_cast<unsigned>(pyInputTypes.size());
    if (mlir::failed(lowering::func::signature::appendClosureInputs(
            op.getClosureTypesAttr(), *tc, pyInputTypes, llvmInputTypes, op)))
      return mlir::failure();

    bool useVoidHelper = lowering::func::void_helper::shouldUse(op, sig);
    bool createFreshInitHelper =
        useVoidHelper && static_cast<bool>(op->getAttr("init_method"));
    bool hasSelfReceiver = false;
    if (auto argNames = op->getAttrOfType<mlir::ArrayAttr>("arg_names")) {
      if (!argNames.empty()) {
        if (auto firstName = mlir::dyn_cast<mlir::StringAttr>(argNames[0]))
          hasSelfReceiver = firstName.getValue() == "self";
      }
    }
    bool createLocalSelfHelper = hasSelfReceiver && !pyInputTypes.empty() &&
                                 mlir::isa<ClassType>(pyInputTypes.front());
    llvm::SmallVector<unsigned, 4> publishedClassArgIndices;
    for (auto [idx, ty] :
         llvm::enumerate(llvm::ArrayRef<mlir::Type>(pyInputTypes)
                             .take_front(visibleInputCount)))
      if ((idx != 0 || !hasSelfReceiver) && mlir::isa<ClassType>(ty))
        publishedClassArgIndices.push_back(idx);
    createLocalSelfHelper = false;
    publishedClassArgIndices.clear();
    if (static_cast<bool>(op->getAttr("maythrow")))
      publishedClassArgIndices.clear();
    std::string freshHelperName;
    std::string localSelfHelperName;
    if (createFreshInitHelper)
      freshHelperName = (nameAttr.getValue() + "$fresh").str();
    if (createLocalSelfHelper)
      localSelfHelperName = (nameAttr.getValue() + "$local").str();
    bool needsCInterface = nameAttr.getValue() == "main";

    auto createLoweredFuncWithInputs =
        [&](llvm::StringRef name, llvm::ArrayRef<mlir::Type> inputs,
            llvm::ArrayRef<mlir::Type> results) -> mlir::func::FuncOp {
      auto loweredType = mlir::FunctionType::get(getContext(), inputs, results);
      auto func =
          rewriter.create<mlir::func::FuncOp>(op.getLoc(), name, loweredType);
      lowering::func::args::ABI::mark(func, pyInputTypes, *tc);
      if (!results.empty())
        lowering::func::result::Ownership::mark(func, pyResultTypes, *tc);
      return func;
    };
    auto createLoweredFunc =
        [&](llvm::StringRef name,
            llvm::ArrayRef<mlir::Type> results) -> mlir::func::FuncOp {
      return createLoweredFuncWithInputs(name, llvmInputTypes, results);
    };
    mlir::func::FuncOp loweredFunc;
    mlir::func::FuncOp helperFunc;
    mlir::func::FuncOp freshHelperFunc;
    mlir::func::FuncOp localSelfHelperFunc;
    llvm::SmallVector<std::pair<unsigned, mlir::func::FuncOp>, 2>
        publishedBorrowHelpers;
    llvm::SmallVector<std::pair<unsigned, mlir::func::FuncOp>, 2>
        freshPublishedBorrowHelpers;
    llvm::SmallVector<std::pair<unsigned, mlir::func::FuncOp>, 2>
        localPublishedBorrowHelpers;
    if (useVoidHelper) {
      std::string helperName = (nameAttr.getValue() + "$void").str();
      helperFunc = createLoweredFunc(helperName, {});
      helperFunc.setVisibility(mlir::SymbolTable::Visibility::Private);
      lowering::func::attrs::copyPy(op, helperFunc);

      loweredFunc = createLoweredFunc(nameAttr.getValue(), llvmResultTypes);
      if (needsCInterface)
        loweredFunc->setAttr("llvm.emit_c_interface", rewriter.getUnitAttr());
      loweredFunc->setAttr("ly.void_helper",
                           mlir::SymbolRefAttr::get(rewriter.getContext(),
                                                    helperFunc.getName()));
      if (createLocalSelfHelper) {
        localSelfHelperFunc = createLoweredFunc(localSelfHelperName, {});
        localSelfHelperFunc.setVisibility(
            mlir::SymbolTable::Visibility::Private);
        lowering::func::attrs::copyPy(op, localSelfHelperFunc);
        localSelfHelperFunc->setAttr("ly.local_self_arg0",
                                     rewriter.getUnitAttr());
        loweredFunc->setAttr(
            "ly.local_self_helper",
            mlir::SymbolRefAttr::get(rewriter.getContext(),
                                     localSelfHelperFunc.getName()));
      }

      for (unsigned idx : publishedClassArgIndices) {
        std::string sharedHelperName =
            (nameAttr.getValue() + "$published_arg").str();
        sharedHelperName += std::to_string(idx);
        auto publishedHelper = createLoweredFunc(sharedHelperName, {});
        publishedHelper.setVisibility(mlir::SymbolTable::Visibility::Private);
        lowering::func::attrs::copyPy(op, publishedHelper);
        publishedBorrowHelpers.emplace_back(idx, publishedHelper);

        if (createLocalSelfHelper) {
          std::string localHelperName =
              (nameAttr.getValue() + "$local_published_arg").str();
          localHelperName += std::to_string(idx);
          auto localPublishedHelper = createLoweredFunc(localHelperName, {});
          localPublishedHelper.setVisibility(
              mlir::SymbolTable::Visibility::Private);
          lowering::func::attrs::copyPy(op, localPublishedHelper);
          localPublishedHelper->setAttr("ly.local_self_arg0",
                                        rewriter.getUnitAttr());
          localPublishedBorrowHelpers.emplace_back(idx, localPublishedHelper);
        }
        if (createFreshInitHelper) {
          std::string freshPublishedHelperName =
              (nameAttr.getValue() + "$fresh_published_arg").str();
          freshPublishedHelperName += std::to_string(idx);
          auto freshPublishedHelper =
              createLoweredFunc(freshPublishedHelperName, {});
          freshPublishedHelper.setVisibility(
              mlir::SymbolTable::Visibility::Private);
          lowering::func::attrs::copyPy(op, freshPublishedHelper);
          freshPublishedHelper->setAttr("ly.zero_initialized_self",
                                        rewriter.getUnitAttr());
          freshPublishedBorrowHelpers.emplace_back(idx, freshPublishedHelper);
        }
      }
      if (createFreshInitHelper) {
        freshHelperFunc = createLoweredFunc(freshHelperName, {});
        freshHelperFunc.setVisibility(mlir::SymbolTable::Visibility::Private);
        lowering::func::attrs::copyPy(op, freshHelperFunc);
        freshHelperFunc->setAttr("ly.zero_initialized_self",
                                 rewriter.getUnitAttr());
        loweredFunc->setAttr(
            "ly.fresh_init_helper",
            mlir::SymbolRefAttr::get(rewriter.getContext(),
                                     freshHelperFunc.getName()));
      }
      lowering::func::attrs::copyPy(op, loweredFunc);
    } else {
      loweredFunc = createLoweredFunc(nameAttr.getValue(), llvmResultTypes);
      if (needsCInterface)
        loweredFunc->setAttr("llvm.emit_c_interface", rewriter.getUnitAttr());
      lowering::func::attrs::copyPy(op, loweredFunc);
      if (createLocalSelfHelper) {
        localSelfHelperFunc =
            createLoweredFunc(localSelfHelperName, llvmResultTypes);
        localSelfHelperFunc.setVisibility(
            mlir::SymbolTable::Visibility::Private);
        lowering::func::attrs::copyPy(op, localSelfHelperFunc);
        localSelfHelperFunc->setAttr("ly.local_self_arg0",
                                     rewriter.getUnitAttr());
        loweredFunc->setAttr(
            "ly.local_self_helper",
            mlir::SymbolRefAttr::get(rewriter.getContext(),
                                     localSelfHelperFunc.getName()));
      }

      for (unsigned idx : publishedClassArgIndices) {
        std::string sharedHelperName =
            (nameAttr.getValue() + "$published_arg").str();
        sharedHelperName += std::to_string(idx);
        auto publishedHelper =
            createLoweredFunc(sharedHelperName, llvmResultTypes);
        publishedHelper.setVisibility(mlir::SymbolTable::Visibility::Private);
        lowering::func::attrs::copyPy(op, publishedHelper);
        publishedBorrowHelpers.emplace_back(idx, publishedHelper);

        if (createLocalSelfHelper) {
          std::string localHelperName =
              (nameAttr.getValue() + "$local_published_arg").str();
          localHelperName += std::to_string(idx);
          auto localPublishedHelper =
              createLoweredFunc(localHelperName, llvmResultTypes);
          localPublishedHelper.setVisibility(
              mlir::SymbolTable::Visibility::Private);
          lowering::func::attrs::copyPy(op, localPublishedHelper);
          localPublishedHelper->setAttr("ly.local_self_arg0",
                                        rewriter.getUnitAttr());
          localPublishedBorrowHelpers.emplace_back(idx, localPublishedHelper);
        }
      }
    }

    if (hasSelfReceiver && !pyInputTypes.empty() &&
        mlir::isa<ClassType>(pyInputTypes.front())) {
      auto selfAttr = rewriter.getUnitAttr();
      loweredFunc->setAttr("ly.self_receiver_arg0", selfAttr);
      if (helperFunc)
        helperFunc->setAttr("ly.self_receiver_arg0", selfAttr);
      if (freshHelperFunc)
        freshHelperFunc->setAttr("ly.self_receiver_arg0", selfAttr);
    }

    if (op.getBody().empty())
      op.getBody().emplaceBlock();

    mlir::func::FuncOp bodyTarget = useVoidHelper ? helperFunc : loweredFunc;
    if (freshHelperFunc) {
      mlir::IRMapping mapping;
      op.getBody().cloneInto(&freshHelperFunc.getBody(), mapping);
    }
    if (localSelfHelperFunc) {
      mlir::IRMapping mapping;
      op.getBody().cloneInto(&localSelfHelperFunc.getBody(), mapping);
    }
    for (auto &[idx, helper] : publishedBorrowHelpers) {
      mlir::IRMapping mapping;
      op.getBody().cloneInto(&helper.getBody(), mapping);
    }
    for (auto &[idx, helper] : freshPublishedBorrowHelpers) {
      mlir::IRMapping mapping;
      op.getBody().cloneInto(&helper.getBody(), mapping);
    }
    for (auto &[idx, helper] : localPublishedBorrowHelpers) {
      mlir::IRMapping mapping;
      op.getBody().cloneInto(&helper.getBody(), mapping);
    }

    llvm::SmallVector<std::pair<unsigned, mlir::func::FuncOp>, 2>
        activePublishedBorrowHelpers;
    for (auto &[idx, helper] : publishedBorrowHelpers) {
      if (!lowering::func::published_borrow::specialize(helper, idx)) {
        helper.erase();
        continue;
      }

      std::string attrName = publication::borrow::Attr::name(idx);
      auto helperRef =
          mlir::SymbolRefAttr::get(rewriter.getContext(), helper.getName());
      loweredFunc->setAttr(attrName, helperRef);
      if (helperFunc)
        helperFunc->setAttr(attrName, helperRef);
      activePublishedBorrowHelpers.emplace_back(idx, helper);
    }
    publishedBorrowHelpers = std::move(activePublishedBorrowHelpers);

    llvm::SmallVector<std::pair<unsigned, mlir::func::FuncOp>, 2>
        activeFreshPublishedBorrowHelpers;
    for (auto &[idx, helper] : freshPublishedBorrowHelpers) {
      if (!lowering::func::published_borrow::specialize(helper, idx)) {
        helper.erase();
        continue;
      }

      std::string attrName = publication::borrow::Attr::name(idx);
      auto helperRef =
          mlir::SymbolRefAttr::get(rewriter.getContext(), helper.getName());
      if (freshHelperFunc)
        freshHelperFunc->setAttr(attrName, helperRef);
      activeFreshPublishedBorrowHelpers.emplace_back(idx, helper);
    }
    freshPublishedBorrowHelpers = std::move(activeFreshPublishedBorrowHelpers);

    llvm::SmallVector<std::pair<unsigned, mlir::func::FuncOp>, 2>
        activeLocalPublishedBorrowHelpers;
    for (auto &[idx, helper] : localPublishedBorrowHelpers) {
      if (!lowering::func::published_borrow::specialize(helper, idx)) {
        helper.erase();
        continue;
      }

      std::string attrName = publication::borrow::Attr::name(idx);
      auto helperRef =
          mlir::SymbolRefAttr::get(rewriter.getContext(), helper.getName());
      if (localSelfHelperFunc)
        localSelfHelperFunc->setAttr(attrName, helperRef);
      activeLocalPublishedBorrowHelpers.emplace_back(idx, helper);
    }
    localPublishedBorrowHelpers = std::move(activeLocalPublishedBorrowHelpers);

    llvm::SmallVector<mlir::func::FuncOp, 8> nestedPublishedBorrowHelpers;
    auto attachNestedPublishedBorrowHelpers =
        [&](auto &&self, mlir::func::FuncOp parent, unsigned startPos) -> void {
      for (unsigned pos = startPos; pos < publishedClassArgIndices.size();
           ++pos) {
        unsigned argIndex = publishedClassArgIndices[pos];
        std::string childName = (parent.getName() + "_arg").str();
        childName += std::to_string(argIndex);
        auto child = rewriter.create<mlir::func::FuncOp>(
            op.getLoc(), childName, parent.getFunctionType());
        child.setVisibility(mlir::SymbolTable::Visibility::Private);
        lowering::func::attrs::copySpecialized(parent, child);

        mlir::IRMapping mapping;
        parent.getBody().cloneInto(&child.getBody(), mapping);
        if (!lowering::func::published_borrow::specialize(child, argIndex)) {
          child.erase();
          continue;
        }

        parent->setAttr(
            publication::borrow::Attr::name(argIndex),
            mlir::SymbolRefAttr::get(rewriter.getContext(), child.getName()));
        nestedPublishedBorrowHelpers.push_back(child);
        self(self, child, pos + 1);
      }
    };

    auto attachForRoots =
        [&](llvm::ArrayRef<std::pair<unsigned, mlir::func::FuncOp>> roots) {
          for (auto &[rootIndex, rootHelper] : roots) {
            auto it = llvm::find(publishedClassArgIndices, rootIndex);
            if (it == publishedClassArgIndices.end())
              continue;
            unsigned startPos = static_cast<unsigned>(std::distance(
                                    publishedClassArgIndices.begin(), it)) +
                                1;
            attachNestedPublishedBorrowHelpers(
                attachNestedPublishedBorrowHelpers, rootHelper, startPos);
          }
        };
    attachForRoots(publishedBorrowHelpers);
    attachForRoots(freshPublishedBorrowHelpers);
    attachForRoots(localPublishedBorrowHelpers);

    rewriter.inlineRegionBefore(op.getBody(), bodyTarget.getBody(),
                                bodyTarget.getBody().end());
    auto convertEntryBlock =
        [&](mlir::func::FuncOp func) -> mlir::LogicalResult {
      auto &entry = func.getBody().front();
      mlir::TypeConverter::SignatureConversion conversion(
          entry.getNumArguments());
      if (mlir::failed(tc->convertSignatureArgs(mlir::TypeRange(pyInputTypes),
                                                conversion)))
        return mlir::failure();
      return rewriter.applySignatureConversion(&entry, conversion,
                                               getTypeConverter())
                 ? mlir::success()
                 : mlir::failure();
    };
    if (mlir::failed(convertEntryBlock(bodyTarget)))
      return mlir::failure();
    if (freshHelperFunc && mlir::failed(convertEntryBlock(freshHelperFunc)))
      return mlir::failure();
    if (localSelfHelperFunc &&
        mlir::failed(convertEntryBlock(localSelfHelperFunc)))
      return mlir::failure();
    for (auto &[idx, helper] : publishedBorrowHelpers)
      if (mlir::failed(convertEntryBlock(helper)))
        return mlir::failure();
    for (auto &[idx, helper] : freshPublishedBorrowHelpers)
      if (mlir::failed(convertEntryBlock(helper)))
        return mlir::failure();
    for (auto &[idx, helper] : localPublishedBorrowHelpers)
      if (mlir::failed(convertEntryBlock(helper)))
        return mlir::failure();
    for (mlir::func::FuncOp helper : nestedPublishedBorrowHelpers)
      if (mlir::failed(convertEntryBlock(helper)))
        return mlir::failure();

    if (useVoidHelper) {
      mlir::Block *wrapperEntry = loweredFunc.addEntryBlock();
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(wrapperEntry);
      rewriter.create<mlir::func::CallOp>(op.getLoc(), helperFunc,
                                          wrapperEntry->getArguments());
      if (llvmResultTypes.empty()) {
        rewriter.create<mlir::func::ReturnOp>(op.getLoc());
        rewriter.eraseOp(op);
        return mlir::success();
      }
      mlir::ModuleOp module = loweredFunc->getParentOfType<mlir::ModuleOp>();
      if (!module)
        return mlir::failure();
      RuntimeAPI runtime(module, rewriter, *tc);
      mlir::Value noneValue = runtime.getNoneValue(op.getLoc());
      if (noneValue.getType() != llvmResultTypes.front())
        noneValue =
            rewriter
                .create<mlir::UnrealizedConversionCastOp>(
                    op.getLoc(), mlir::TypeRange{llvmResultTypes.front()},
                    mlir::ValueRange{noneValue})
                .getResult(0);
      rewriter.create<mlir::func::ReturnOp>(op.getLoc(), noneValue);
    }

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

// ReturnLowering: py.return -> func.return

struct ReturnLowering : public mlir::OpConversionPattern<ReturnOp> {
  ReturnLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<ReturnOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(ReturnOp op, OneToNOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto parentFunc = op->getParentOfType<mlir::func::FuncOp>();
    auto flattenOperands = [&]() {
      llvm::SmallVector<mlir::Value> flattened;
      for (mlir::ValueRange group : adaptor.getOperands())
        flattened.append(group.begin(), group.end());
      return flattened;
    };
    if (parentFunc && parentFunc.getFunctionType().getNumResults() == 0) {
      NoneOp noneOp = nullptr;
      llvm::SmallVector<mlir::Value> operands = flattenOperands();
      if (!operands.empty())
        noneOp = operands.front().getDefiningOp<NoneOp>();
      rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(op);
      if (noneOp && noneOp->use_empty())
        rewriter.eraseOp(noneOp);
      return mlir::success();
    }
    llvm::SmallVector<mlir::Value> operands = flattenOperands();
    rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(op, operands);
    return mlir::success();
  }
};

// FuncObjectLowering: py.func_object -> function reference

struct FuncObjectLowering : public mlir::OpConversionPattern<FuncObjectOp> {
  FuncObjectLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<FuncObjectOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(FuncObjectOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return mlir::failure();

    llvm::StringRef symbol = op.getTargetAttr().getValue();

    if (symbol == "__builtin_print" || symbol == "__builtin_print_raw" ||
        symbol == "print") {
      llvm::StringRef builtinSymbol = symbol == "__builtin_print_raw"
                                          ? "__builtin_print_raw"
                                          : "__builtin_print";
      auto func = module.lookupSymbol<mlir::func::FuncOp>(builtinSymbol);
      if (!func) {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(module.getBody());
        auto markerType =
            mlir::FunctionType::get(rewriter.getContext(), {}, {});
        func = rewriter.create<mlir::func::FuncOp>(op.getLoc(), builtinSymbol,
                                                   markerType);
        func.setPrivate();
      }
      auto constOp = rewriter.create<mlir::func::ConstantOp>(
          op.getLoc(), func.getFunctionType(),
          mlir::SymbolRefAttr::get(rewriter.getContext(), builtinSymbol));
      auto bridge = rewriter.create<mlir::UnrealizedConversionCastOp>(
          op.getLoc(), mlir::TypeRange{op.getResult().getType()},
          mlir::ValueRange{constOp.getResult()});
      rewriter.replaceOp(op, bridge.getResult(0));
      return mlir::success();
    }

    // Look up user-defined function
    auto func = module.lookupSymbol<mlir::func::FuncOp>(symbol);
    if (!func)
      return rewriter.notifyMatchFailure(op, "unknown function reference '" +
                                                 symbol + "'");

    auto constOp = rewriter.create<mlir::func::ConstantOp>(
        op.getLoc(), func.getFunctionType(),
        mlir::SymbolRefAttr::get(rewriter.getContext(), symbol));
    auto bridge = rewriter.create<mlir::UnrealizedConversionCastOp>(
        op.getLoc(), mlir::TypeRange{op.getResult().getType()},
        mlir::ValueRange{constOp.getResult()});
    rewriter.replaceOp(op, bridge.getResult(0));
    return mlir::success();
  }
};

struct MakeFunctionLowering : public mlir::OpConversionPattern<MakeFunctionOp> {
  MakeFunctionLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<MakeFunctionOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(MakeFunctionOp op, OneToNOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    rewriter.setInsertionPoint(op);
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return mlir::failure();

    FuncType funcType = mlir::dyn_cast<FuncType>(op.getResult().getType());
    if (!funcType)
      return rewriter.notifyMatchFailure(op, "result must be !py.func");

    llvm::StringRef targetName =
        op.getTargetAttr().getLeafReference().empty()
            ? op.getTargetAttr().getRootReference().getValue()
            : op.getTargetAttr().getLeafReference().getValue();
    auto targetFunc = module.lookupSymbol<mlir::func::FuncOp>(targetName);
    if (!targetFunc)
      return rewriter.notifyMatchFailure(op, "unknown lowered func target");

    auto constOp = rewriter.create<mlir::func::ConstantOp>(
        op.getLoc(), targetFunc.getFunctionType(),
        mlir::SymbolRefAttr::get(rewriter.getContext(), targetName));
    auto bridge = rewriter.create<mlir::UnrealizedConversionCastOp>(
        op.getLoc(), mlir::TypeRange{op.getResult().getType()},
        mlir::ValueRange{constOp.getResult()});
    rewriter.replaceOp(op, bridge.getResult(0));
    return mlir::success();
  }
};

} // namespace

namespace lowering::func::Patterns {
void populate(PyLLVMTypeConverter &typeConverter,
              mlir::RewritePatternSet &patterns) {
  auto *ctx = patterns.getContext();
  patterns.add<FuncOpLowering, ReturnLowering, FuncObjectLowering,
               MakeFunctionLowering>(typeConverter, ctx);
}
} // namespace lowering::func::Patterns

namespace lowering::func::definition::Patterns {
void populate(PyLLVMTypeConverter &typeConverter,
              mlir::RewritePatternSet &patterns) {
  patterns.add<FuncOpLowering>(typeConverter, patterns.getContext());
}
} // namespace lowering::func::definition::Patterns

namespace lowering::func::returns::Patterns {
void populate(PyLLVMTypeConverter &typeConverter,
              mlir::RewritePatternSet &patterns) {
  patterns.add<ReturnLowering>(typeConverter, patterns.getContext());
}
} // namespace lowering::func::returns::Patterns

namespace lowering::func::objects::Patterns {
void populate(PyLLVMTypeConverter &typeConverter,
              mlir::RewritePatternSet &patterns) {
  auto *ctx = patterns.getContext();
  patterns.add<FuncObjectLowering, MakeFunctionLowering>(typeConverter, ctx);
}
} // namespace lowering::func::objects::Patterns

} // namespace py
