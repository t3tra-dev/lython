#include "Passes/Runtime/Helpers.h"

#include "Common/LoweringUtils.h"
#include "Passes/OwnershipAnalysis.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <string>

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

namespace py {
namespace {

mlir::func::FuncOp lookupFuncFromSymbolAttr(mlir::ModuleOp module,
                                            mlir::SymbolRefAttr ref) {
  if (!ref)
    return nullptr;
  llvm::StringRef symbol = ref.getLeafReference().empty()
                               ? ref.getRootReference().getValue()
                               : ref.getLeafReference().getValue();
  return module.lookupSymbol<mlir::func::FuncOp>(symbol);
}

mlir::func::FuncOp clonePrivateHelper(mlir::ModuleOp module,
                                      mlir::func::FuncOp base,
                                      llvm::StringRef newName) {
  if (!base)
    return nullptr;
  if (auto existing = module.lookupSymbol<mlir::func::FuncOp>(newName))
    return existing;

  auto cloned = mlir::cast<mlir::func::FuncOp>(base->clone());
  cloned.setName(newName);
  cloned.setVisibility(mlir::SymbolTable::Visibility::Private);
  mlir::SymbolTable symbolTable(module);
  symbolTable.insert(cloned);
  return cloned;
}

namespace runtime_func {

mlir::LLVM::LLVMFuncOp getOrInsert(mlir::ModuleOp module, llvm::StringRef name,
                                   mlir::Type resultType,
                                   llvm::ArrayRef<mlir::Type> argTypes) {
  if (auto fn = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(name))
    return fn;
  mlir::OpBuilder builder(module.getBody(), module.getBody()->begin());
  auto fnType = mlir::LLVM::LLVMFunctionType::get(resultType, argTypes, false);
  return builder.create<mlir::LLVM::LLVMFuncOp>(module.getLoc(), name, fnType);
}

} // namespace runtime_func

bool arrayAttrContainsIndex(mlir::ArrayAttr attr, unsigned index) {
  if (!attr)
    return false;
  for (mlir::Attribute element : attr) {
    auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(element);
    if (!intAttr)
      continue;
    if (intAttr.getInt() == static_cast<int64_t>(index))
      return true;
  }
  return false;
}

mlir::Value stripPublishCasts(mlir::Value value) {
  while (auto cast = value.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
    if (cast->getNumOperands() != 1)
      break;
    value = cast.getOperand(0);
  }
  return value;
}

} // namespace

namespace lowering::runtime::async_args {

void mark(mlir::Operation *funcLike, llvm::ArrayRef<mlir::Type> inputs,
          const PyLLVMTypeConverter &typeConverter,
          bool trailingExceptionCell) {
  if (!funcLike)
    return;

  unsigned flattenedIndex = 0;
  for (auto [index, inputType] : llvm::enumerate(inputs)) {
    llvm::SmallVector<mlir::Type, 4> converted;
    if (mlir::failed(typeConverter.convertType(inputType, converted)) ||
        converted.empty())
      return;

    if (mlir::isa<CoroutineType, FutureType, TaskType>(inputType) &&
        converted.size() >= 2) {
      ::py::async_runtime::ExceptionCell::markArgument(funcLike,
                                                       flattenedIndex + 1);
      if (mlir::isa<TaskType>(inputType) && converted.size() >= 3)
        ::py::async_runtime::CancelFlag::markArgument(funcLike,
                                                      flattenedIndex + 2);
    } else if (trailingExceptionCell && index + 1 == inputs.size() &&
               mlir::isa<mlir::LLVM::LLVMPointerType>(inputType)) {
      ::py::async_runtime::ExceptionCell::markArgument(funcLike,
                                                       flattenedIndex);
    }

    flattenedIndex += static_cast<unsigned>(converted.size());
  }
}

} // namespace lowering::runtime::async_args

namespace lowering::runtime::published_borrow {

bool specialize(mlir::func::FuncOp func, unsigned argIndex) {
  if (!func || func.getBody().empty())
    return false;

  mlir::Block &entry = func.getBody().front();
  if (argIndex >= entry.getNumArguments())
    return false;

  mlir::Value borrowedArg = entry.getArgument(argIndex);
  llvm::SmallVector<PublishOp> publishes;
  func.walk([&](PublishOp publish) {
    if (stripPublishCasts(publish.getInput()) == borrowedArg)
      publishes.push_back(publish);
  });

  if (publishes.empty())
    return false;

  bool changed = false;
  for (PublishOp publish : publishes) {
    mlir::Value forwardedValue = publish.getInput();
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
      use->set(forwardedValue);
    for (DecRefOp decRef : decRefs)
      decRef.erase();
    publish.erase();
    changed = true;
  }

  return changed;
}

} // namespace lowering::runtime::published_borrow

namespace lowering::runtime::helpers {

void retainBorrowedEntryBlockReturns(mlir::ModuleOp module) {
  mlir::MLIRContext *ctx = module.getContext();
  auto ptrType = mlir::LLVM::LLVMPointerType::get(ctx);
  auto voidType = mlir::LLVM::LLVMVoidType::get(ctx);
  auto incRef = runtime_func::getOrInsert(module, RuntimeSymbols::kIncRef,
                                          voidType, {ptrType});
  auto incRefRef = mlir::SymbolRefAttr::get(ctx, incRef.getName());
  auto stripForwarding = [](mlir::Value value) {
    while (true) {
      if (auto cast = value.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
        if (cast->getNumOperands() != 1)
          return value;
        value = cast.getOperand(0);
        continue;
      }
      if (auto upcast = value.getDefiningOp<UpcastOp>()) {
        if (isPyOwnershipMaterializedObjectBridge(upcast))
          return value;
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

  module.walk([&](mlir::func::FuncOp func) {
    if (func.getBody().empty())
      return;
    mlir::Block &entry = func.getBody().front();
    llvm::SmallVector<mlir::func::ReturnOp> returns;
    func.walk([&](mlir::func::ReturnOp ret) { returns.push_back(ret); });
    for (mlir::func::ReturnOp ret : returns) {
      if (ret.getNumOperands() != 1)
        continue;
      mlir::Value returned = ret.getOperand(0);
      if (returned.getType() != ptrType)
        continue;
      auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(returned);
      if (!blockArg || blockArg.getOwner() != &entry)
        continue;
      if (mlir::Operation *prev = ret->getPrevNode()) {
        if (auto inc = mlir::dyn_cast<IncRefOp>(prev)) {
          if (stripForwarding(inc.getObject()) == returned)
            continue;
        }
        if (auto call = mlir::dyn_cast<mlir::LLVM::CallOp>(prev)) {
          if (call.getCallee() &&
              *call.getCallee() == RuntimeSymbols::kIncRef &&
              call.getNumOperands() == 1 && call.getOperand(0) == returned)
            continue;
        }
      }
      mlir::OpBuilder builder(ret);
      auto retain = builder.create<mlir::LLVM::CallOp>(
          ret.getLoc(), mlir::TypeRange{}, incRefRef,
          mlir::ValueRange{returned});
      threadsafe::Retain::premise(retain.getOperation(),
                                  ThreadSafetyAttrs::kPremiseEntryBorrowed);
    }
  });
}

void synthesizeLocalSelf(mlir::ModuleOp module) {
  llvm::SmallVector<mlir::func::FuncOp> candidates;
  module.walk([&](mlir::func::FuncOp func) {
    if (!func->hasAttr("llvm.emit_c_interface"))
      return;
    if (!func->hasAttr("ly.self_receiver_arg0"))
      return;
    candidates.push_back(func);
  });

  mlir::MLIRContext *ctx = module.getContext();
  for (mlir::func::FuncOp func : candidates) {
    if (func->hasAttr("ly.void_helper") &&
        !func->hasAttr("ly.local_self_helper")) {
      auto baseHelper = lookupFuncFromSymbolAttr(
          module, func->getAttrOfType<mlir::SymbolRefAttr>("ly.void_helper"));
      if (auto localHelper = clonePrivateHelper(
              module, baseHelper, (func.getName() + "$local").str())) {
        localHelper->setAttr("ly.local_self_arg0", mlir::UnitAttr::get(ctx));
        func->setAttr("ly.local_self_helper",
                      mlir::SymbolRefAttr::get(ctx, localHelper.getName()));
      }
    }

    if (!func->hasAttr("ly.local_self_helper") &&
        !func->hasAttr("ly.void_helper") &&
        !func->hasAttr("ly.class_return_helper")) {
      if (auto localHelper = clonePrivateHelper(
              module, func, (func.getName() + "$local").str())) {
        localHelper->setAttr("ly.local_self_arg0", mlir::UnitAttr::get(ctx));
        func->setAttr("ly.local_self_helper",
                      mlir::SymbolRefAttr::get(ctx, localHelper.getName()));
      }
    }

    if (func->hasAttr("ly.class_return_helper")) {
      auto baseHelper = lookupFuncFromSymbolAttr(
          module,
          func->getAttrOfType<mlir::SymbolRefAttr>("ly.class_return_helper"));
      if (baseHelper && !baseHelper->hasAttr("ly.local_self_helper")) {
        if (auto localHelper = clonePrivateHelper(
                module, baseHelper, (baseHelper.getName() + "$local").str())) {
          localHelper->setAttr("ly.local_self_arg0", mlir::UnitAttr::get(ctx));
          baseHelper->setAttr(
              "ly.local_self_helper",
              mlir::SymbolRefAttr::get(ctx, localHelper.getName()));
        }
      }
    }
  }
}

void synthesizePublishedBorrow(mlir::ModuleOp module) {
  mlir::MLIRContext *ctx = module.getContext();
  bool changed = false;
  do {
    changed = false;
    llvm::SmallVector<mlir::func::FuncOp> candidates;
    module.walk([&](mlir::func::FuncOp func) {
      if (func->hasAttr("ly.publishes_args"))
        candidates.push_back(func);
    });

    for (mlir::func::FuncOp func : candidates) {
      auto publishesArgs =
          func->getAttrOfType<mlir::ArrayAttr>("ly.publishes_args");
      if (!publishesArgs)
        continue;

      for (unsigned argIndex = 0; argIndex < func.getNumArguments();
           ++argIndex) {
        if (!arrayAttrContainsIndex(publishesArgs, argIndex))
          continue;

        std::string attrName = publication::borrow::Attr::name(argIndex);
        if (func->hasAttr(attrName))
          continue;

        std::string helperName =
            (func.getName() + "$published_arg" + std::to_string(argIndex))
                .str();
        auto helper = clonePrivateHelper(module, func, helperName);
        if (!helper)
          continue;
        if (!lowering::runtime::published_borrow::specialize(helper,
                                                             argIndex)) {
          helper.erase();
          continue;
        }

        func->setAttr(attrName,
                      mlir::SymbolRefAttr::get(ctx, helper.getName()));
        changed = true;
      }
    }
  } while (changed);
}

} // namespace lowering::runtime::helpers

} // namespace py
