#include "Runtime/Js/Host.h"

#include "JsHost.h"
#include "PyDialectTypes.h"

#include "mlir/IR/AttrTypeSubElements.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace py::lowering::js {
namespace {

// ⛔ By hand and not through the replacer's own recursion: the py types do
// not expose their elements to it, so it would rewrite `js.Element` and leave
// `js.Element | None` and `Callable[[js.Event], None]` as they were.
mlir::Type erase(mlir::Type type) {
  if (!type)
    return type;
  mlir::MLIRContext *context = type.getContext();
  auto eraseAll = [](llvm::ArrayRef<mlir::Type> types) {
    return llvm::to_vector(llvm::map_range(types, erase));
  };
  if (auto contract = mlir::dyn_cast<py::ContractType>(type)) {
    if (contract.getContractName().starts_with(
            (py::kJsHostModule + ".").str()))
      return py::ContractType::get(context, py::kJsProxyContract, {});
    if (contract.getArguments().empty())
      return type;
    return py::ContractType::get(context, contract.getContractName(),
                                 eraseAll(contract.getArguments()));
  }
  if (auto unionType = mlir::dyn_cast<py::UnionType>(type))
    return py::UnionType::getNormalized(context,
                                        eraseAll(unionType.getMemberTypes()));
  if (auto protocol = mlir::dyn_cast<py::ProtocolType>(type))
    return py::ProtocolType::get(context, protocol.getProtocolName(),
                                 eraseAll(protocol.getArguments()));
  if (auto typeType = mlir::dyn_cast<py::TypeType>(type))
    return py::TypeType::get(context, erase(typeType.getInstanceType()));
  if (auto callable = mlir::dyn_cast<py::CallableType>(type))
    return py::CallableType::get(
        context, eraseAll(callable.getPositionalTypes()),
        eraseAll(callable.getKwOnlyTypes()), erase(callable.getVarargType()),
        erase(callable.getKwargType()), eraseAll(callable.getResultTypes()),
        callable.getPositionalNames(), callable.getKwOnlyNames(),
        callable.getPositionalDefaults(), callable.getKwOnlyDefaults(),
        callable.getVarargName(), callable.getKwargName(),
        callable.getPositionalOnlyCount());
  if (auto overload = mlir::dyn_cast<py::OverloadType>(type))
    return py::OverloadType::get(context,
                                 eraseAll(overload.getCandidateTypes()));
  return type;
}

} // namespace

void eraseJsHostContracts(mlir::ModuleOp module) {
  if (!module->hasAttr(py::kJsHostModuleAttr))
    return;
  mlir::AttrTypeReplacer replacer;
  replacer.addReplacement([](mlir::Type type) -> std::optional<mlir::Type> {
    if (!mlir::isa<py::ContractType, py::UnionType, py::ProtocolType,
                   py::TypeType, py::CallableType, py::OverloadType>(type))
      return std::nullopt;
    mlir::Type erased = erase(type);
    if (erased == type)
      return std::nullopt;
    return erased;
  });
  replacer.recursivelyReplaceElementsIn(module, /*replaceAttrs=*/true,
                                        /*replaceLocs=*/false,
                                        /*replaceTypes=*/true);
}

} // namespace py::lowering::js
