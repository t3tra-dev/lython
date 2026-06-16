#include "Passes/Runtime/Helpers.h"

#include "Common/Container.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/STLExtras.h"

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

static void setFunctionArgumentStringAttr(mlir::Operation *funcLike,
                                          unsigned argIndex,
                                          llvm::StringRef attrName,
                                          llvm::StringRef value) {
  if (!funcLike)
    return;
  mlir::Builder builder(funcLike->getContext());
  if (auto function = mlir::dyn_cast<mlir::FunctionOpInterface>(funcLike)) {
    if (argIndex < function.getNumArguments())
      function.setArgAttr(argIndex, attrName, builder.getStringAttr(value));
    return;
  }

  if (funcLike->getNumRegions() == 0 || funcLike->getRegion(0).empty())
    return;
  unsigned numArgs = funcLike->getRegion(0).front().getNumArguments();
  if (argIndex >= numArgs)
    return;

  auto existing = funcLike->getAttrOfType<mlir::ArrayAttr>("arg_attrs");
  llvm::SmallVector<mlir::Attribute> attrs;
  attrs.reserve(numArgs);
  for (unsigned index = 0; index < numArgs; ++index) {
    if (existing && index < existing.size())
      attrs.push_back(existing[index]);
    else
      attrs.push_back(builder.getDictionaryAttr({}));
  }

  auto dict = mlir::dyn_cast<mlir::DictionaryAttr>(attrs[argIndex]);
  mlir::NamedAttrList named(dict ? dict : builder.getDictionaryAttr({}));
  named.set(attrName, builder.getStringAttr(value));
  attrs[argIndex] = named.getDictionary(funcLike->getContext());
  funcLike->setAttr("arg_attrs", builder.getArrayAttr(attrs));
}

static void markContainerDescriptor(mlir::Operation *funcLike,
                                    mlir::Type logicalType,
                                    unsigned logicalIndex,
                                    unsigned flattenedIndex,
                                    unsigned convertedWidth) {
  auto kind = container::Descriptor::kindNameForLogicalType(logicalType);
  if (!kind)
    return;
  std::string group =
      (llvm::Twine(*kind) + ".arg" + llvm::Twine(logicalIndex)).str();
  for (unsigned slot = 0; slot < convertedWidth; ++slot) {
    llvm::StringRef component =
        container::Descriptor::componentForLogicalType(logicalType, slot);
    if (component.empty())
      continue;
    unsigned argIndex = flattenedIndex + slot;
    setFunctionArgumentStringAttr(
        funcLike, argIndex, ContainerSafetyAttrs::kDescriptorGroup, group);
    setFunctionArgumentStringAttr(funcLike, argIndex,
                                  ContainerSafetyAttrs::kDescriptorKind, *kind);
    setFunctionArgumentStringAttr(funcLike, argIndex,
                                  ContainerSafetyAttrs::kDescriptorComponent,
                                  component);
  }
}

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

    markContainerDescriptor(funcLike, inputType, static_cast<unsigned>(index),
                            flattenedIndex,
                            static_cast<unsigned>(converted.size()));

    if (isCoroutineProtocolType(inputType) && converted.size() >= 2) {
      ::py::async_runtime::RuntimeHandle::markArgument(funcLike,
                                                       flattenedIndex);
      ::py::async_runtime::ExceptionCell::markArgument(funcLike,
                                                       flattenedIndex + 1);
    } else if (trailingExceptionCell && index + 1 == inputs.size() &&
               (::py::async_runtime::isExceptionCellType(inputType) ||
                ::py::async_runtime::isLoweredExceptionCellType(inputType))) {
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
        !func->hasAttr("ly.void_helper")) {
      if (auto localHelper = clonePrivateHelper(
              module, func, (func.getName() + "$local").str())) {
        localHelper->setAttr("ly.local_self_arg0", mlir::UnitAttr::get(ctx));
        func->setAttr("ly.local_self_helper",
                      mlir::SymbolRefAttr::get(ctx, localHelper.getName()));
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
