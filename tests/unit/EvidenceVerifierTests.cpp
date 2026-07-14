#include "Driver.h"
#include "Emitter.h"
#include "Parser.h"
#include "PyDialectTypes.h"
#include "runtime/Verification.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/TargetParser/Host.h"

#include <gtest/gtest.h>

namespace {

const mlir::DialectRegistry &testRegistry() {
  static mlir::DialectRegistry *registry = [] {
    auto *result = new mlir::DialectRegistry();
    lython::driver::registerLythonDialects(*result);
    return result;
  }();
  return *registry;
}

lython::emitter::EmitResult emitSource(llvm::StringRef source,
                                       mlir::MLIRContext &context) {
  lython::parser::ParseOptions parseOptions;
  parseOptions.typeComments = true;
  lython::parser::ParseResult parsed =
      lython::parser::parse(source, "<test>.py", parseOptions);
  EXPECT_TRUE(parsed.ok());
  if (!parsed.ok())
    return {};

  lython::emitter::EmitOptions options;
  options.targetTriple = llvm::sys::getDefaultTargetTriple();
  return lython::emitter::emitModule(*parsed.tree, context, "__main__",
                                     "<test>.py", options);
}

TEST(EvidenceVerifierTest, CleanModulePasses) {
  mlir::MLIRContext context(testRegistry());
  lython::emitter::EmitResult emitted =
      emitSource("x = 1\nprint(x + 2)\n", context);
  ASSERT_TRUE(emitted.ok());
  EXPECT_TRUE(mlir::succeeded(py::verifyTypeEvidence(*emitted.module)));
}

TEST(EvidenceVerifierTest, RejectsEscapedInferenceVariableInTypeAttr) {
  mlir::MLIRContext context(testRegistry());
  lython::emitter::EmitResult emitted =
      emitSource("x = 1\nprint(x + 2)\n", context);
  ASSERT_TRUE(emitted.ok());

  // A nested occurrence: InferVarType itself is not a Py_ContractType, so
  // only container arguments and type-bearing attributes can smuggle it past
  // op verification -- exactly what the escape check must catch.
  mlir::Type contaminated = py::ContractType::get(
      &context, "builtins.list", {py::InferVarType::get(&context, 0)});

  bool attached = false;
  emitted.module->walk([&](mlir::Operation *op) {
    if (attached || op == emitted.module->getOperation())
      return;
    op->setAttr("test.contaminated", mlir::TypeAttr::get(contaminated));
    attached = true;
  });
  ASSERT_TRUE(attached);

  context.getDiagEngine().registerHandler(
      [](mlir::Diagnostic &) { return mlir::success(); });
  EXPECT_TRUE(mlir::failed(py::verifyTypeEvidence(*emitted.module)));
}

} // namespace
