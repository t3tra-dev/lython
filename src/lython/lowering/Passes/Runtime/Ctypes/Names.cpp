#include "Runtime/Ctypes/Internal.h"

#include "PyProtocols.h"

#include "llvm/ADT/StringSwitch.h"

namespace py::lowering::ctypes {

llvm::StringRef stripCtypesModule(llvm::StringRef contract) {
  return py::native::stripCtypesModule(contract);
}

std::optional<std::string> ctypesModuleAttrContract(mlir::MLIRContext &context,
                                                    llvm::StringRef moduleName,
                                                    llvm::StringRef attr) {
  return py::protocols::Table::get(context).moduleClassExport(moduleName, attr);
}

std::optional<std::string> ctypesBareNameContract(mlir::MLIRContext &context,
                                                  llvm::StringRef name) {
  return py::protocols::Table::get(context).bareClassExport(name);
}

std::optional<std::string>
ctypesQualifiedNameContract(mlir::MLIRContext &context, llvm::StringRef name) {
  return py::protocols::Table::get(context).qualifiedClassExport(name);
}

bool isStaticCtypesFunctionName(mlir::MLIRContext &context,
                                llvm::StringRef name) {
  const py::protocols::Table &table = py::protocols::Table::get(context);
  if (name.contains('.')) {
    auto split = name.rsplit('.');
    if (split.first != "ctypes" && split.first != "_ctypes")
      return false; // other modules' exports lower via the builtin channel
    return table.isModuleCallableExport(split.first, split.second);
  }
  return table.isModuleCallableExport("ctypes", name);
}

mlir::Type ctypesContractType(mlir::MLIRContext *context,
                              llvm::StringRef contract) {
  return py::ContractType::get(context, contract);
}

RuntimeBundle makeCtypesModuleBundle(mlir::Type resultType,
                                     llvm::StringRef moduleName) {
  RuntimeBundle bundle = RuntimeBundle::object(resultType, {});
  RuntimeCtypesEvidence evidence;
  evidence.kind = RuntimeCtypesEvidence::Kind::Module;
  evidence.lifetime = RuntimeCtypesEvidence::Lifetime::Static;
  evidence.ctypeName = moduleName.str();
  evidence.ctype = resultType;
  bundle.ctypes = std::move(evidence);
  return bundle;
}

} // namespace py::lowering::ctypes
