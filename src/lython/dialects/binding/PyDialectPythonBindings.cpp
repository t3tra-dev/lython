#include <pybind11/pybind11.h>

#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir/CAPI/IR.h"
#include "mlir/IR/MLIRContext.h"

#include "PyDialect.h.inc"

namespace py11 = pybind11;
namespace lython_py = ::py;

namespace {

constexpr const char *kCapiAttr = MLIR_PYTHON_CAPI_PTR_ATTR;

py11::object importContextCurrent(py11::handle contextObject) {
  if (!contextObject.is_none())
    return py11::reinterpret_borrow<py11::object>(contextObject);
  return py11::module::import(MAKE_MLIR_PYTHON_QUALNAME("ir"))
      .attr("Context")
      .attr("current");
}

MlirDialectRegistry unwrapDialectRegistry(const py11::object &registryObj) {
  py11::object capsule = registryObj.attr(kCapiAttr);
  return mlirPythonCapsuleToDialectRegistry(capsule.ptr());
}

MlirContext unwrapContext(const py11::object &contextObj) {
  py11::object capsule = contextObj.attr(kCapiAttr);
  return mlirPythonCapsuleToContext(capsule.ptr());
}

void registerPyDialect(const py11::object &registryObj) {
  MlirDialectRegistry registry = unwrapDialectRegistry(registryObj);
  unwrap(registry)->insert<lython_py::PyDialect>();
}

void ensurePyDialectLoaded(const py11::object &contextObj) {
  py11::object resolvedContext = importContextCurrent(contextObj);
  MlirContext ctx = unwrapContext(resolvedContext);
  unwrap(ctx)->loadDialect<lython_py::PyDialect>();
}

} // namespace

PYBIND11_MODULE(_site_initialize_0, m) {
  m.doc() = "Initialization hooks for the Lython py dialect.";
  m.def("register_dialects", &registerPyDialect, py11::arg("registry"));
  m.def("context_init_hook", &ensurePyDialectLoaded, py11::arg("context") = py11::none());
}
