#include "Contracts.h"

#include "PyDialectTypes.h"

#include "llvm/ADT/STLExtras.h"

namespace py::contracts {



std::string runtimeContractName(mlir::Type type) {
  if (auto contract = mlir::dyn_cast<py::ContractType>(type))
    return contract.getContractName().str();
  if (mlir::isa<py::CallableType>(type))
    return "builtins.function";
  if (auto protocol = mlir::dyn_cast<py::ProtocolType>(type))
    if (protocol.getProtocolName() == "Callable")
      return "builtins.function";
  if (auto literal = mlir::dyn_cast<py::LiteralType>(type)) {
    llvm::StringRef spelling = literal.getSpelling();
    if (spelling == "True" || spelling == "False")
      return "builtins.bool";
    if (spelling == "None")
      return "types.NoneType";
    if (spelling.starts_with("\"") && spelling.ends_with("\""))
      return "builtins.str";
    if (isIntegerLiteralSpelling(spelling))
      return "builtins.int";
  }
  return "";
}

} // namespace py::contracts
