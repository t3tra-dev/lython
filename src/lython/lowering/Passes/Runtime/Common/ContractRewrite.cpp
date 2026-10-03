#include "Runtime/Common/ContractRewrite.h"

#include "mlir/IR/AttrTypeSubElements.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace py::lowering {
namespace {

// ⛔ By hand and not through the replacer's own recursion: the py types do
// not expose their elements to it, so it would rewrite `js.Element` and leave
// `js.Element | None` and `Callable[[js.Event], None]` as they were.
mlir::Type rewrite(mlir::Type type,
                   llvm::function_ref<mlir::Type(py::ContractType)> rename) {
  if (!type)
    return type;
  mlir::MLIRContext *context = type.getContext();
  auto all = [&](llvm::ArrayRef<mlir::Type> types) {
    return llvm::to_vector(llvm::map_range(
        types, [&](mlir::Type element) { return rewrite(element, rename); }));
  };
  if (auto contract = mlir::dyn_cast<py::ContractType>(type)) {
    if (!contract.getArguments().empty())
      contract = py::ContractType::get(context, contract.getContractName(),
                                       all(contract.getArguments()));
    return rename(contract);
  }
  if (auto unionType = mlir::dyn_cast<py::UnionType>(type))
    return py::UnionType::getNormalized(context,
                                        all(unionType.getMemberTypes()));
  if (auto protocol = mlir::dyn_cast<py::ProtocolType>(type))
    return py::ProtocolType::get(context, protocol.getProtocolName(),
                                 all(protocol.getArguments()));
  if (auto typeType = mlir::dyn_cast<py::TypeType>(type))
    return py::TypeType::get(context,
                             rewrite(typeType.getInstanceType(), rename));
  if (auto callable = mlir::dyn_cast<py::CallableType>(type))
    return py::CallableType::get(
        context, all(callable.getPositionalTypes()),
        all(callable.getKwOnlyTypes()),
        rewrite(callable.getVarargType(), rename),
        rewrite(callable.getKwargType(), rename),
        all(callable.getResultTypes()), callable.getPositionalNames(),
        callable.getKwOnlyNames(), callable.getPositionalDefaults(),
        callable.getKwOnlyDefaults(), callable.getVarargName(),
        callable.getKwargName(), callable.getPositionalOnlyCount());
  if (auto overload = mlir::dyn_cast<py::OverloadType>(type))
    return py::OverloadType::get(context, all(overload.getCandidateTypes()));
  return type;
}

} // namespace

void rewriteContracts(
    mlir::ModuleOp module,
    llvm::function_ref<mlir::Type(py::ContractType)> rename) {
  mlir::AttrTypeReplacer replacer;
  replacer.addReplacement([&](mlir::Type type) -> std::optional<mlir::Type> {
    if (!mlir::isa<py::ContractType, py::UnionType, py::ProtocolType,
                   py::TypeType, py::CallableType, py::OverloadType>(type))
      return std::nullopt;
    mlir::Type rewritten = rewrite(type, rename);
    if (rewritten == type)
      return std::nullopt;
    return rewritten;
  });
  replacer.recursivelyReplaceElementsIn(module, /*replaceAttrs=*/true,
                                        /*replaceLocs=*/false,
                                        /*replaceTypes=*/true);
}

} // namespace py::lowering
