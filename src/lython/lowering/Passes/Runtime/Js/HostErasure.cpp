#include "Runtime/Js/Host.h"

#include "Runtime/Common/ContractRewrite.h"

#include "JsHost.h"

namespace py::lowering::js {

void eraseJsHostContracts(mlir::ModuleOp module) {
  if (!module->hasAttr(py::kJsHostModuleAttr))
    return;
  rewriteContracts(module, [](py::ContractType contract) -> mlir::Type {
    if (contract.getContractName().starts_with(
            (py::kJsHostModule + ".").str()))
      return py::ContractType::get(contract.getContext(), py::kJsProxyContract,
                                   {});
    return contract;
  });
}

} // namespace py::lowering::js
