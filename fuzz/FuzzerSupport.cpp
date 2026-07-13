#include "FuzzerSupport.h"

#include "Driver.h"
#include "embedded.h"

#include "llvm/Support/TargetSelect.h"

const mlir::DialectRegistry &lythonFuzzerRegistry() {
  static mlir::DialectRegistry *registry = [] {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    py::runtime_library::embedded::registerPyRuntimeEmbeddedModules();
    auto *result = new mlir::DialectRegistry();
    lython::driver::registerLythonDialects(*result);
    return result;
  }();
  return *registry;
}
