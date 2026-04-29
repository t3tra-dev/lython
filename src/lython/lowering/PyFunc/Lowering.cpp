// This file implements lowering patterns for py.func, py.return, and
// py.func_object operations. These patterns convert Python-style function
// definitions to standard MLIR func dialect operations.

#include "Common/LoweringUtils.h"
#include "Common/RuntimeSupport.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/FormatVariadic.h"

#include <algorithm>

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

using namespace mlir;

namespace py {
ClassOp lookupClassSymbol(Operation *from, ClassType classType);
FailureOr<LLVM::LLVMStructType>
getStaticClassObjectType(Operation *from, ClassType classType,
                         const PyLLVMTypeConverter &typeConverter);

namespace {

LogicalResult translateFunctionSignature(
    FuncSignatureType sig, const PyLLVMTypeConverter &typeConverter,
    SmallVectorImpl<Type> &pyInputs, SmallVectorImpl<Type> &convertedInputs,
    SmallVectorImpl<Type> &convertedResults, Operation *emitOnError) {
  pyInputs.clear();
  convertedInputs.clear();
  convertedResults.clear();

  auto appendConverted = [&](Type ty,
                             SmallVectorImpl<Type> &storage) -> LogicalResult {
    SmallVector<Type, 4> converted;
    if (failed(typeConverter.convertType(ty, converted)) || converted.empty()) {
      emitOnError->emitError("failed to convert type ") << ty;
      return failure();
    }
    storage.append(converted.begin(), converted.end());
    return success();
  };

  auto positional = sig.getPositionalTypes();
  pyInputs.append(positional.begin(), positional.end());
  auto kwonly = sig.getKwOnlyTypes();
  pyInputs.append(kwonly.begin(), kwonly.end());
  if (sig.hasVararg())
    pyInputs.push_back(sig.getVarargType());
  if (sig.hasKwarg())
    pyInputs.push_back(sig.getKwargType());

  for (Type ty : pyInputs)
    if (failed(appendConverted(ty, convertedInputs)))
      return failure();

  for (Type result : sig.getResultTypes())
    if (failed(appendConverted(result, convertedResults)))
      return failure();

  return success();
}

static LogicalResult appendClosureInputTypes(
    ArrayAttr closureTypesAttr, const PyLLVMTypeConverter &typeConverter,
    SmallVectorImpl<Type> &pyInputs, SmallVectorImpl<Type> &convertedInputs,
    Operation *emitOnError) {
  if (!closureTypesAttr)
    return success();

  auto appendConverted = [&](Type ty) -> LogicalResult {
    SmallVector<Type, 4> converted;
    if (failed(typeConverter.convertType(ty, converted)) || converted.empty()) {
      emitOnError->emitError("failed to convert closure type ") << ty;
      return failure();
    }
    pyInputs.push_back(ty);
    convertedInputs.append(converted.begin(), converted.end());
    return success();
  };

  for (Attribute attr : closureTypesAttr) {
    auto typeAttr = dyn_cast<TypeAttr>(attr);
    if (!typeAttr) {
      emitOnError->emitError(
          "closure_types must contain only TypeAttr elements");
      return failure();
    }
    if (failed(appendConverted(typeAttr.getValue())))
      return failure();
  }

  return success();
}

static bool shouldLowerViaVoidHelper(FuncOp op, FuncSignatureType sig) {
  auto results = sig.getResultTypes();
  if (results.size() != 1 || !isa<NoneType>(results.front()))
    return false;
  return static_cast<bool>(op->getAttr("mutates_self")) ||
         static_cast<bool>(op->getAttr("init_method"));
}

static void copyPyFuncAttrs(FuncOp from, func::FuncOp to) {
  if (auto attr = from->getAttr("arg_names"))
    to->setAttr("arg_names", attr);
  if (auto attr = from->getAttr("mutates_self"))
    to->setAttr("mutates_self", attr);
  if (auto attr = from->getAttr("init_method"))
    to->setAttr("init_method", attr);
  if (auto attr = from->getAttr("nothrow"))
    to->setAttr("nothrow", attr);
  if (auto attr = from->getAttr("maythrow"))
    to->setAttr("maythrow", attr);
  if (auto attr = from->getAttr("lython.publishes_args"))
    to->setAttr("lython.publishes_args", attr);
  if (auto attr = from->getAttr("lython.captures_published"))
    to->setAttr("lython.captures_published", attr);
  if (auto attr = from->getAttr("lython.returns_published"))
    to->setAttr("lython.returns_published", attr);
  if (auto attr = from->getAttr("lython.readonly_args"))
    to->setAttr("lython.readonly_args", attr);
  if (auto attr = from->getAttr("lython.mutable_args"))
    to->setAttr("lython.mutable_args", attr);
  if (auto attr = from->getAttr("kwonly_names"))
    to->setAttr("kwonly_names", attr);
  if (auto attr = from->getAttr("closure_types"))
    to->setAttr("closure_types", attr);
}

static std::string getPublishedBorrowHelperAttrName(unsigned argIndex) {
  return "lython.published_borrow_helper_arg" + std::to_string(argIndex);
}

static bool specializePublishedBorrowArg(func::FuncOp func, unsigned argIndex) {
  if (func.getBody().empty())
    return false;

  Block &entry = func.getBody().front();
  if (argIndex >= entry.getNumArguments())
    return false;

  Value borrowedArg = entry.getArgument(argIndex);
  SmallVector<PublishOp> publishes;
  func.walk([&](PublishOp publish) {
    if (publish.getInput() == borrowedArg)
      publishes.push_back(publish);
  });

  if (publishes.empty())
    return false;

  bool changed = false;
  for (PublishOp publish : publishes) {
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
      use->set(borrowedArg);
    for (DecRefOp decRef : decRefs)
      decRef.erase();

    publish.erase();
    changed = true;
  }

  return changed;
}

static void copySpecializedHelperAttrs(func::FuncOp from, func::FuncOp to) {
  if (auto attr = from->getAttr("arg_names"))
    to->setAttr("arg_names", attr);
  if (auto attr = from->getAttr("mutates_self"))
    to->setAttr("mutates_self", attr);
  if (auto attr = from->getAttr("init_method"))
    to->setAttr("init_method", attr);
  if (auto attr = from->getAttr("nothrow"))
    to->setAttr("nothrow", attr);
  if (auto attr = from->getAttr("maythrow"))
    to->setAttr("maythrow", attr);
  if (auto attr = from->getAttr("lython.publishes_args"))
    to->setAttr("lython.publishes_args", attr);
  if (auto attr = from->getAttr("lython.captures_published"))
    to->setAttr("lython.captures_published", attr);
  if (auto attr = from->getAttr("lython.returns_published"))
    to->setAttr("lython.returns_published", attr);
  if (auto attr = from->getAttr("lython.readonly_args"))
    to->setAttr("lython.readonly_args", attr);
  if (auto attr = from->getAttr("lython.mutable_args"))
    to->setAttr("lython.mutable_args", attr);
  if (auto attr = from->getAttr("lython.local_self_arg0"))
    to->setAttr("lython.local_self_arg0", attr);
  if (auto attr = from->getAttr("lython.zero_initialized_self"))
    to->setAttr("lython.zero_initialized_self", attr);
  if (auto attr = from->getAttr("lython.class_return_outarg"))
    to->setAttr("lython.class_return_outarg", attr);
  if (auto attr = from->getAttr("kwonly_names"))
    to->setAttr("kwonly_names", attr);
  if (auto attr = from->getAttr("closure_types"))
    to->setAttr("closure_types", attr);
}

// FuncOpLowering: py.func -> func.func

struct FuncOpLowering : public OpConversionPattern<FuncOp> {
  FuncOpLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<FuncOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto nameAttr = op->getAttrOfType<StringAttr>("sym_name");
    if (!nameAttr)
      return rewriter.notifyMatchFailure(op, "missing sym_name");

    // Skip builtin print function - it's handled specially
    if (nameAttr.getValue() == "__builtin_print") {
      rewriter.eraseOp(op);
      return success();
    }

    auto sigAttr = op->getAttrOfType<TypeAttr>("function_type");
    if (!sigAttr)
      return rewriter.notifyMatchFailure(op, "missing function_type attr");

    auto sig = dyn_cast<FuncSignatureType>(sigAttr.getValue());
    if (!sig)
      return rewriter.notifyMatchFailure(op, "expected FuncSignatureType");

    auto *tc = static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    SmallVector<Type, 8> pyInputTypes;
    SmallVector<Type, 8> llvmInputTypes;
    SmallVector<Type, 4> llvmResultTypes;
    if (failed(translateFunctionSignature(sig, *tc, pyInputTypes,
                                          llvmInputTypes, llvmResultTypes, op)))
      return failure();
    unsigned visibleInputCount = static_cast<unsigned>(pyInputTypes.size());
    if (failed(appendClosureInputTypes(op.getClosureTypesAttr(), *tc,
                                       pyInputTypes, llvmInputTypes, op)))
      return failure();

    bool useVoidHelper = shouldLowerViaVoidHelper(op, sig);
    bool createClassReturnHelper = false;
    ClassType classReturnType;
    Type classReturnStorageType;
    LLVM::LLVMStructType classReturnObjectType;
    if (!useVoidHelper) {
      auto results = sig.getResultTypes();
      if (results.size() == 1)
        if (auto classType = dyn_cast<ClassType>(results.front())) {
          createClassReturnHelper = true;
          classReturnType = classType;
          classReturnStorageType = llvmResultTypes.front();
          auto objectTypeOr = getStaticClassObjectType(op, classType, *tc);
          if (failed(objectTypeOr))
            return failure();
          classReturnObjectType = *objectTypeOr;
        }
    }
    bool createFreshInitHelper =
        useVoidHelper && static_cast<bool>(op->getAttr("init_method"));
    bool hasSelfReceiver = false;
    if (auto argNames = op->getAttrOfType<ArrayAttr>("arg_names")) {
      if (!argNames.empty()) {
        if (auto firstName = dyn_cast<StringAttr>(argNames[0]))
          hasSelfReceiver = firstName.getValue() == "self";
      }
    }
    bool createLocalSelfHelper = hasSelfReceiver && !pyInputTypes.empty() &&
                                 isa<ClassType>(pyInputTypes.front());
    SmallVector<unsigned, 4> publishedClassArgIndices;
    for (auto [idx, ty] : llvm::enumerate(
             ArrayRef<Type>(pyInputTypes).take_front(visibleInputCount)))
      if ((idx != 0 || !hasSelfReceiver) && isa<ClassType>(ty))
        publishedClassArgIndices.push_back(idx);
    if (!std::getenv("LYTHON_ENABLE_CLASS_SPECIALIZED_HELPERS")) {
      createLocalSelfHelper = false;
      publishedClassArgIndices.clear();
    }
    if (static_cast<bool>(op->getAttr("maythrow")))
      publishedClassArgIndices.clear();
    std::string freshHelperName;
    std::string localSelfHelperName;
    std::string classReturnHelperName;
    if (createFreshInitHelper)
      freshHelperName = (nameAttr.getValue() + "$fresh").str();
    if (createLocalSelfHelper && !createClassReturnHelper)
      localSelfHelperName = (nameAttr.getValue() + "$local").str();
    if (createClassReturnHelper) {
      classReturnHelperName = (nameAttr.getValue() + "$sret").str();
      if (createLocalSelfHelper)
        localSelfHelperName = classReturnHelperName + "$local";
    }

    auto createLoweredFuncWithInputs =
        [&](StringRef name, ArrayRef<Type> inputs,
            ArrayRef<Type> results) -> func::FuncOp {
      auto loweredType = FunctionType::get(getContext(), inputs, results);
      return rewriter.create<func::FuncOp>(op.getLoc(), name, loweredType);
    };
    auto createLoweredFunc = [&](StringRef name,
                                 ArrayRef<Type> results) -> func::FuncOp {
      return createLoweredFuncWithInputs(name, llvmInputTypes, results);
    };
    SmallVector<Type, 8> classReturnHelperInputs(llvmInputTypes.begin(),
                                                 llvmInputTypes.end());
    if (createClassReturnHelper)
      classReturnHelperInputs.push_back(classReturnStorageType);

    func::FuncOp loweredFunc;
    func::FuncOp helperFunc;
    func::FuncOp freshHelperFunc;
    func::FuncOp localSelfHelperFunc;
    func::FuncOp classReturnHelperFunc;
    SmallVector<std::pair<unsigned, func::FuncOp>, 2> publishedBorrowHelpers;
    SmallVector<std::pair<unsigned, func::FuncOp>, 2>
        freshPublishedBorrowHelpers;
    SmallVector<std::pair<unsigned, func::FuncOp>, 2>
        localPublishedBorrowHelpers;
    if (useVoidHelper) {
      std::string helperName = (nameAttr.getValue() + "$void").str();
      helperFunc = createLoweredFunc(helperName, {});
      helperFunc.setVisibility(SymbolTable::Visibility::Private);
      copyPyFuncAttrs(op, helperFunc);

      loweredFunc = createLoweredFunc(nameAttr.getValue(), llvmResultTypes);
      loweredFunc->setAttr("llvm.emit_c_interface", rewriter.getUnitAttr());
      loweredFunc->setAttr(
          "lython.void_helper",
          SymbolRefAttr::get(rewriter.getContext(), helperFunc.getName()));
      if (createLocalSelfHelper) {
        localSelfHelperFunc = createLoweredFunc(localSelfHelperName, {});
        localSelfHelperFunc.setVisibility(SymbolTable::Visibility::Private);
        copyPyFuncAttrs(op, localSelfHelperFunc);
        localSelfHelperFunc->setAttr("lython.local_self_arg0",
                                     rewriter.getUnitAttr());
        loweredFunc->setAttr("lython.local_self_helper",
                             SymbolRefAttr::get(rewriter.getContext(),
                                                localSelfHelperFunc.getName()));
      }

      for (unsigned idx : publishedClassArgIndices) {
        std::string sharedHelperName =
            (nameAttr.getValue() + "$published_arg").str();
        sharedHelperName += std::to_string(idx);
        auto publishedHelper = createLoweredFunc(sharedHelperName, {});
        publishedHelper.setVisibility(SymbolTable::Visibility::Private);
        copyPyFuncAttrs(op, publishedHelper);
        publishedBorrowHelpers.emplace_back(idx, publishedHelper);

        if (createLocalSelfHelper) {
          std::string localHelperName =
              (nameAttr.getValue() + "$local_published_arg").str();
          localHelperName += std::to_string(idx);
          auto localPublishedHelper = createLoweredFunc(localHelperName, {});
          localPublishedHelper.setVisibility(SymbolTable::Visibility::Private);
          copyPyFuncAttrs(op, localPublishedHelper);
          localPublishedHelper->setAttr("lython.local_self_arg0",
                                        rewriter.getUnitAttr());
          localPublishedBorrowHelpers.emplace_back(idx, localPublishedHelper);
        }
        if (createFreshInitHelper) {
          std::string freshPublishedHelperName =
              (nameAttr.getValue() + "$fresh_published_arg").str();
          freshPublishedHelperName += std::to_string(idx);
          auto freshPublishedHelper =
              createLoweredFunc(freshPublishedHelperName, {});
          freshPublishedHelper.setVisibility(SymbolTable::Visibility::Private);
          copyPyFuncAttrs(op, freshPublishedHelper);
          freshPublishedHelper->setAttr("lython.zero_initialized_self",
                                        rewriter.getUnitAttr());
          freshPublishedBorrowHelpers.emplace_back(idx, freshPublishedHelper);
        }
      }
      if (createFreshInitHelper) {
        freshHelperFunc = createLoweredFunc(freshHelperName, {});
        freshHelperFunc.setVisibility(SymbolTable::Visibility::Private);
        copyPyFuncAttrs(op, freshHelperFunc);
        freshHelperFunc->setAttr("lython.zero_initialized_self",
                                 rewriter.getUnitAttr());
        loweredFunc->setAttr("lython.fresh_init_helper",
                             SymbolRefAttr::get(rewriter.getContext(),
                                                freshHelperFunc.getName()));
      }
      copyPyFuncAttrs(op, loweredFunc);
    } else if (createClassReturnHelper) {
      classReturnHelperFunc = createLoweredFuncWithInputs(
          classReturnHelperName, classReturnHelperInputs, {});
      classReturnHelperFunc.setVisibility(SymbolTable::Visibility::Private);
      copyPyFuncAttrs(op, classReturnHelperFunc);
      classReturnHelperFunc->setAttr("lython.class_return_outarg",
                                     rewriter.getUnitAttr());

      loweredFunc = createLoweredFunc(nameAttr.getValue(), llvmResultTypes);
      loweredFunc->setAttr("llvm.emit_c_interface", rewriter.getUnitAttr());
      loweredFunc->setAttr("lython.class_return_helper",
                           SymbolRefAttr::get(rewriter.getContext(),
                                              classReturnHelperFunc.getName()));
      copyPyFuncAttrs(op, loweredFunc);

      if (createLocalSelfHelper) {
        localSelfHelperFunc = createLoweredFuncWithInputs(
            localSelfHelperName, classReturnHelperInputs, {});
        localSelfHelperFunc.setVisibility(SymbolTable::Visibility::Private);
        copyPyFuncAttrs(op, localSelfHelperFunc);
        localSelfHelperFunc->setAttr("lython.local_self_arg0",
                                     rewriter.getUnitAttr());
        localSelfHelperFunc->setAttr("lython.class_return_outarg",
                                     rewriter.getUnitAttr());
        classReturnHelperFunc->setAttr(
            "lython.local_self_helper",
            SymbolRefAttr::get(rewriter.getContext(),
                               localSelfHelperFunc.getName()));
      }

      for (unsigned idx : publishedClassArgIndices) {
        std::string sharedHelperName = classReturnHelperName + "$published_arg";
        sharedHelperName += std::to_string(idx);
        auto publishedHelper = createLoweredFuncWithInputs(
            sharedHelperName, classReturnHelperInputs, {});
        publishedHelper.setVisibility(SymbolTable::Visibility::Private);
        copyPyFuncAttrs(op, publishedHelper);
        publishedHelper->setAttr("lython.class_return_outarg",
                                 rewriter.getUnitAttr());
        publishedBorrowHelpers.emplace_back(idx, publishedHelper);

        if (createLocalSelfHelper) {
          std::string localHelperName = localSelfHelperName + "_published_arg";
          localHelperName += std::to_string(idx);
          auto localPublishedHelper = createLoweredFuncWithInputs(
              localHelperName, classReturnHelperInputs, {});
          localPublishedHelper.setVisibility(SymbolTable::Visibility::Private);
          copyPyFuncAttrs(op, localPublishedHelper);
          localPublishedHelper->setAttr("lython.local_self_arg0",
                                        rewriter.getUnitAttr());
          localPublishedHelper->setAttr("lython.class_return_outarg",
                                        rewriter.getUnitAttr());
          localPublishedBorrowHelpers.emplace_back(idx, localPublishedHelper);
        }
      }
    } else {
      loweredFunc = createLoweredFunc(nameAttr.getValue(), llvmResultTypes);
      loweredFunc->setAttr("llvm.emit_c_interface", rewriter.getUnitAttr());
      copyPyFuncAttrs(op, loweredFunc);
      if (createLocalSelfHelper) {
        localSelfHelperFunc =
            createLoweredFunc(localSelfHelperName, llvmResultTypes);
        localSelfHelperFunc.setVisibility(SymbolTable::Visibility::Private);
        copyPyFuncAttrs(op, localSelfHelperFunc);
        localSelfHelperFunc->setAttr("lython.local_self_arg0",
                                     rewriter.getUnitAttr());
        loweredFunc->setAttr("lython.local_self_helper",
                             SymbolRefAttr::get(rewriter.getContext(),
                                                localSelfHelperFunc.getName()));
      }

      for (unsigned idx : publishedClassArgIndices) {
        std::string sharedHelperName =
            (nameAttr.getValue() + "$published_arg").str();
        sharedHelperName += std::to_string(idx);
        auto publishedHelper =
            createLoweredFunc(sharedHelperName, llvmResultTypes);
        publishedHelper.setVisibility(SymbolTable::Visibility::Private);
        copyPyFuncAttrs(op, publishedHelper);
        publishedBorrowHelpers.emplace_back(idx, publishedHelper);

        if (createLocalSelfHelper) {
          std::string localHelperName =
              (nameAttr.getValue() + "$local_published_arg").str();
          localHelperName += std::to_string(idx);
          auto localPublishedHelper =
              createLoweredFunc(localHelperName, llvmResultTypes);
          localPublishedHelper.setVisibility(SymbolTable::Visibility::Private);
          copyPyFuncAttrs(op, localPublishedHelper);
          localPublishedHelper->setAttr("lython.local_self_arg0",
                                        rewriter.getUnitAttr());
          localPublishedBorrowHelpers.emplace_back(idx, localPublishedHelper);
        }
      }
    }

    if (hasSelfReceiver && !pyInputTypes.empty() &&
        isa<ClassType>(pyInputTypes.front())) {
      auto selfAttr = rewriter.getUnitAttr();
      loweredFunc->setAttr("lython.self_receiver_arg0", selfAttr);
      if (helperFunc)
        helperFunc->setAttr("lython.self_receiver_arg0", selfAttr);
      if (freshHelperFunc)
        freshHelperFunc->setAttr("lython.self_receiver_arg0", selfAttr);
      if (classReturnHelperFunc)
        classReturnHelperFunc->setAttr("lython.self_receiver_arg0", selfAttr);
    }

    if (op.getBody().empty())
      op.getBody().emplaceBlock();

    func::FuncOp bodyTarget =
        useVoidHelper
            ? helperFunc
            : (createClassReturnHelper ? classReturnHelperFunc : loweredFunc);
    auto appendClassReturnOutArg = [&](func::FuncOp func) {
      if (!createClassReturnHelper || !func || func.getBody().empty())
        return;
      Block &entry = func.getBody().front();
      if (entry.getNumArguments() == func.getFunctionType().getNumInputs())
        return;
      entry.addArgument(classReturnStorageType, op.getLoc());
    };
    if (freshHelperFunc) {
      IRMapping mapping;
      op.getBody().cloneInto(&freshHelperFunc.getBody(), mapping);
    }
    if (localSelfHelperFunc) {
      IRMapping mapping;
      op.getBody().cloneInto(&localSelfHelperFunc.getBody(), mapping);
    }
    for (auto &[idx, helper] : publishedBorrowHelpers) {
      IRMapping mapping;
      op.getBody().cloneInto(&helper.getBody(), mapping);
    }
    for (auto &[idx, helper] : freshPublishedBorrowHelpers) {
      IRMapping mapping;
      op.getBody().cloneInto(&helper.getBody(), mapping);
    }
    for (auto &[idx, helper] : localPublishedBorrowHelpers) {
      IRMapping mapping;
      op.getBody().cloneInto(&helper.getBody(), mapping);
    }

    SmallVector<std::pair<unsigned, func::FuncOp>, 2>
        activePublishedBorrowHelpers;
    for (auto &[idx, helper] : publishedBorrowHelpers) {
      if (!specializePublishedBorrowArg(helper, idx)) {
        helper.erase();
        continue;
      }

      std::string attrName = getPublishedBorrowHelperAttrName(idx);
      auto helperRef =
          SymbolRefAttr::get(rewriter.getContext(), helper.getName());
      if (createClassReturnHelper) {
        classReturnHelperFunc->setAttr(attrName, helperRef);
      } else {
        loweredFunc->setAttr(attrName, helperRef);
      }
      if (helperFunc)
        helperFunc->setAttr(attrName, helperRef);
      activePublishedBorrowHelpers.emplace_back(idx, helper);
    }
    publishedBorrowHelpers = std::move(activePublishedBorrowHelpers);

    SmallVector<std::pair<unsigned, func::FuncOp>, 2>
        activeFreshPublishedBorrowHelpers;
    for (auto &[idx, helper] : freshPublishedBorrowHelpers) {
      if (!specializePublishedBorrowArg(helper, idx)) {
        helper.erase();
        continue;
      }

      std::string attrName = getPublishedBorrowHelperAttrName(idx);
      auto helperRef =
          SymbolRefAttr::get(rewriter.getContext(), helper.getName());
      if (freshHelperFunc)
        freshHelperFunc->setAttr(attrName, helperRef);
      activeFreshPublishedBorrowHelpers.emplace_back(idx, helper);
    }
    freshPublishedBorrowHelpers = std::move(activeFreshPublishedBorrowHelpers);

    SmallVector<std::pair<unsigned, func::FuncOp>, 2>
        activeLocalPublishedBorrowHelpers;
    for (auto &[idx, helper] : localPublishedBorrowHelpers) {
      if (!specializePublishedBorrowArg(helper, idx)) {
        helper.erase();
        continue;
      }

      std::string attrName = getPublishedBorrowHelperAttrName(idx);
      auto helperRef =
          SymbolRefAttr::get(rewriter.getContext(), helper.getName());
      if (localSelfHelperFunc)
        localSelfHelperFunc->setAttr(attrName, helperRef);
      activeLocalPublishedBorrowHelpers.emplace_back(idx, helper);
    }
    localPublishedBorrowHelpers = std::move(activeLocalPublishedBorrowHelpers);

    SmallVector<func::FuncOp, 8> nestedPublishedBorrowHelpers;
    auto attachNestedPublishedBorrowHelpers =
        [&](auto &&self, func::FuncOp parent, unsigned startPos) -> void {
      for (unsigned pos = startPos; pos < publishedClassArgIndices.size();
           ++pos) {
        unsigned argIndex = publishedClassArgIndices[pos];
        std::string childName = (parent.getName() + "_arg").str();
        childName += std::to_string(argIndex);
        auto child = rewriter.create<func::FuncOp>(op.getLoc(), childName,
                                                   parent.getFunctionType());
        child.setVisibility(SymbolTable::Visibility::Private);
        copySpecializedHelperAttrs(parent, child);

        IRMapping mapping;
        parent.getBody().cloneInto(&child.getBody(), mapping);
        if (!specializePublishedBorrowArg(child, argIndex)) {
          child.erase();
          continue;
        }

        parent->setAttr(
            getPublishedBorrowHelperAttrName(argIndex),
            SymbolRefAttr::get(rewriter.getContext(), child.getName()));
        nestedPublishedBorrowHelpers.push_back(child);
        self(self, child, pos + 1);
      }
    };

    auto attachForRoots =
        [&](ArrayRef<std::pair<unsigned, func::FuncOp>> roots) {
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
    appendClassReturnOutArg(bodyTarget);
    appendClassReturnOutArg(localSelfHelperFunc);
    for (auto &[idx, helper] : publishedBorrowHelpers)
      appendClassReturnOutArg(helper);
    for (auto &[idx, helper] : localPublishedBorrowHelpers)
      appendClassReturnOutArg(helper);
    for (func::FuncOp helper : nestedPublishedBorrowHelpers)
      appendClassReturnOutArg(helper);
    auto convertEntryBlock =
        [&](func::FuncOp func,
            bool preserveClassReturnOutArg) -> LogicalResult {
      auto &entry = func.getBody().front();
      TypeConverter::SignatureConversion conversion(entry.getNumArguments());
      if (failed(tc->convertSignatureArgs(TypeRange(pyInputTypes), conversion)))
        return failure();
      if (preserveClassReturnOutArg) {
        unsigned outArgIndex = static_cast<unsigned>(pyInputTypes.size());
        if (entry.getNumArguments() <= outArgIndex)
          return failure();
        SmallVector<Type, 1> packed{entry.getArgument(outArgIndex).getType()};
        conversion.addInputs(outArgIndex, packed);
      }
      return rewriter.applySignatureConversion(&entry, conversion,
                                               getTypeConverter())
                 ? success()
                 : failure();
    };
    if (failed(convertEntryBlock(bodyTarget, createClassReturnHelper)))
      return failure();
    if (freshHelperFunc && failed(convertEntryBlock(freshHelperFunc, false)))
      return failure();
    if (localSelfHelperFunc &&
        failed(convertEntryBlock(localSelfHelperFunc, createClassReturnHelper)))
      return failure();
    for (auto &[idx, helper] : publishedBorrowHelpers)
      if (failed(convertEntryBlock(helper, createClassReturnHelper)))
        return failure();
    for (auto &[idx, helper] : freshPublishedBorrowHelpers)
      if (failed(convertEntryBlock(helper, false)))
        return failure();
    for (auto &[idx, helper] : localPublishedBorrowHelpers)
      if (failed(convertEntryBlock(helper, createClassReturnHelper)))
        return failure();
    for (func::FuncOp helper : nestedPublishedBorrowHelpers)
      if (failed(convertEntryBlock(helper, createClassReturnHelper)))
        return failure();

    if (useVoidHelper) {
      Block *wrapperEntry = loweredFunc.addEntryBlock();
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(wrapperEntry);
      rewriter.create<func::CallOp>(op.getLoc(), helperFunc,
                                    wrapperEntry->getArguments());
      auto noneValue =
          rewriter.create<NoneOp>(op.getLoc(), NoneType::get(getContext()));
      auto castedNone = rewriter.create<CastIdentityOp>(
          op.getLoc(), llvmResultTypes.front(), noneValue.getResult());
      rewriter.create<func::ReturnOp>(op.getLoc(), castedNone.getResult());
    } else if (createClassReturnHelper) {
      Block *wrapperEntry = loweredFunc.addEntryBlock();
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(wrapperEntry);
      Value one = rewriter.create<LLVM::ConstantOp>(
          op.getLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(1));
      auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
      Value resultSlot = rewriter.create<LLVM::AllocaOp>(
          op.getLoc(), ptrType, classReturnObjectType, one, /*alignment=*/0);
      Value zero =
          rewriter.create<LLVM::ZeroOp>(op.getLoc(), classReturnObjectType);
      rewriter.create<LLVM::StoreOp>(op.getLoc(), zero, resultSlot);

      SmallVector<Value> helperOperands(wrapperEntry->getArguments().begin(),
                                        wrapperEntry->getArguments().end());
      helperOperands.push_back(resultSlot);
      rewriter.create<func::CallOp>(op.getLoc(), classReturnHelperFunc,
                                    helperOperands);

      auto module = loweredFunc->getParentOfType<ModuleOp>();
      if (!module)
        return failure();
      auto promoteHelper = getOrInsertLLVMFunc(
          op.getLoc(), module, rewriter,
          getClassHelperName(classReturnType, "promote"), ptrType, {ptrType});
      auto promoteRef =
          SymbolRefAttr::get(rewriter.getContext(), promoteHelper.getName());
      auto promoted = rewriter.create<LLVM::CallOp>(
          op.getLoc(), TypeRange{ptrType}, promoteRef, ValueRange{resultSlot});
      rewriter.create<func::ReturnOp>(op.getLoc(), promoted.getResult());
    }

    rewriter.eraseOp(op);
    return success();
  }
};

// ReturnLowering: py.return -> func.return

struct ReturnLowering : public OpConversionPattern<ReturnOp> {
  ReturnLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<ReturnOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(ReturnOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto parentFunc = op->getParentOfType<func::FuncOp>();
    auto flattenOperands = [&]() {
      SmallVector<Value> flattened;
      for (ValueRange group : adaptor.getOperands())
        flattened.append(group.begin(), group.end());
      return flattened;
    };
    auto stripForwardedValue = [](Value value) -> Value {
      while (true) {
        if (auto identity = value.getDefiningOp<CastIdentityOp>()) {
          value = identity.getInput();
          continue;
        }
        if (auto upcast = value.getDefiningOp<UpcastOp>()) {
          value = upcast.getInput();
          continue;
        }
        if (auto publish = value.getDefiningOp<PublishOp>()) {
          value = publish.getInput();
          continue;
        }
        return value;
      }
    };
    if (parentFunc && parentFunc->getAttr("lython.class_return_outarg")) {
      SmallVector<Value> operands = flattenOperands();
      if (operands.size() != 1)
        return rewriter.notifyMatchFailure(
            op, "class return helper expects exactly one return operand");
      auto classType = dyn_cast<ClassType>(op.getOperands().front().getType());
      if (!classType)
        return rewriter.notifyMatchFailure(
            op, "class return helper expects a static class return operand");

      auto *converter =
          static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
      auto objectTypeOr = getStaticClassObjectType(op, classType, *converter);
      if (failed(objectTypeOr))
        return failure();

      ModuleOp module = op->getParentOfType<ModuleOp>();
      if (!module)
        return failure();

      Block &entry = parentFunc.getBody().front();
      Value outArg = entry.getArgument(entry.getNumArguments() - 1);
      auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
      auto copyHelper = getOrInsertLLVMFunc(
          op.getLoc(), module, rewriter, getClassHelperName(classType, "copy"),
          LLVM::LLVMVoidType::get(rewriter.getContext()), {ptrType, ptrType});
      auto copyRef =
          SymbolRefAttr::get(rewriter.getContext(), copyHelper.getName());
      SmallVector<Value> copyOperands;
      copyOperands.push_back(outArg);
      copyOperands.append(operands.begin(), operands.end());
      rewriter.create<LLVM::CallOp>(op.getLoc(), TypeRange{}, copyRef,
                                    copyOperands);
      rewriter.replaceOpWithNewOp<func::ReturnOp>(op);
      return success();
    }
    if (parentFunc && parentFunc.getFunctionType().getNumResults() == 0) {
      NoneOp noneOp = nullptr;
      SmallVector<Value> operands = flattenOperands();
      if (!operands.empty())
        noneOp = operands.front().getDefiningOp<NoneOp>();
      rewriter.replaceOpWithNewOp<func::ReturnOp>(op);
      if (noneOp && noneOp->use_empty())
        rewriter.eraseOp(noneOp);
      return success();
    }
    if (parentFunc && op.getNumOperands() == 1 &&
        isPyType(op.getOperands().front().getType())) {
      Value original = stripForwardedValue(op.getOperands().front());
      auto blockArg = dyn_cast<BlockArgument>(original);
      if (blockArg && blockArg.getOwner() == &parentFunc.getBody().front())
        rewriter.create<IncRefOp>(op.getLoc(), original);
    }
    SmallVector<Value> operands = flattenOperands();
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, operands);
    return success();
  }
};

// FuncObjectLowering: py.func_object -> function reference

struct FuncObjectLowering : public OpConversionPattern<FuncObjectOp> {
  FuncObjectLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<FuncObjectOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(FuncObjectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    static llvm::StringMap<llvm::StringLiteral> builtinTable = {
        {"__builtin_print", RuntimeSymbols::kGetBuiltinPrint},
        {"print", RuntimeSymbols::kGetBuiltinPrint},
    };

    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    StringRef symbol = op.getTargetAttr().getValue();

    // Check if this is a builtin function
    if (auto it = builtinTable.find(symbol); it != builtinTable.end()) {
      auto *converter =
          static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
      RuntimeAPI runtime(module, rewriter, *converter);
      Type resultType = converter->convertType(op.getResult().getType());
      auto call =
          runtime.call(op.getLoc(), it->second, resultType, ValueRange{});
      rewriter.replaceOp(op, call.getResults());
      return success();
    }

    // Look up user-defined function
    auto func = module.lookupSymbol<func::FuncOp>(symbol);
    if (!func)
      return rewriter.notifyMatchFailure(op, "unknown function reference '" +
                                                 symbol + "'");

    auto constOp = rewriter.create<func::ConstantOp>(
        op.getLoc(), func.getFunctionType(),
        SymbolRefAttr::get(rewriter.getContext(), symbol));
    auto identity = rewriter.create<CastIdentityOp>(
        op.getLoc(), op.getResult().getType(), constOp.getResult());
    rewriter.replaceOp(op, identity.getResult());
    return success();
  }
};

struct MakeFunctionLowering : public OpConversionPattern<MakeFunctionOp> {
  MakeFunctionLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<MakeFunctionOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(MakeFunctionOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    rewriter.setInsertionPoint(op);
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    FuncType funcType = dyn_cast<FuncType>(op.getResult().getType());
    if (!funcType)
      return rewriter.notifyMatchFailure(op, "result must be !py.func");

    StringRef targetName =
        op.getTargetAttr().getLeafReference().empty()
            ? op.getTargetAttr().getRootReference().getValue()
            : op.getTargetAttr().getLeafReference().getValue();
    auto targetFunc = module.lookupSymbol<func::FuncOp>(targetName);
    if (!targetFunc)
      return rewriter.notifyMatchFailure(op, "unknown lowered func target");

    auto constOp = rewriter.create<func::ConstantOp>(
        op.getLoc(), targetFunc.getFunctionType(),
        SymbolRefAttr::get(rewriter.getContext(), targetName));
    auto identity = rewriter.create<CastIdentityOp>(
        op.getLoc(), op.getResult().getType(), constOp.getResult());
    rewriter.replaceOp(op, identity.getResult());
    return success();
  }
};

} // namespace

void populatePyFuncLoweringPatterns(PyLLVMTypeConverter &typeConverter,
                                    RewritePatternSet &patterns) {
  auto *ctx = patterns.getContext();
  patterns.add<FuncOpLowering, ReturnLowering, FuncObjectLowering,
               MakeFunctionLowering>(typeConverter, ctx);
}

} // namespace py
