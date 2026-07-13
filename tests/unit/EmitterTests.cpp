#include "Driver.h"
#include "Emitter.h"
#include "Parser.h"

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

TEST(EmitterTest, EmitsSimpleModule) {
  mlir::MLIRContext context(testRegistry());
  lython::emitter::EmitResult emitted =
      emitSource("x = 1\nprint(x + 2)\n", context);
  EXPECT_TRUE(emitted.ok());
  EXPECT_TRUE(emitted.module);
}

TEST(EmitterTest, ReportsUnresolvedName) {
  mlir::MLIRContext context(testRegistry());
  lython::emitter::EmitResult emitted =
      emitSource("x = eval(\"1\")\n", context);
  EXPECT_FALSE(emitted.ok());
  bool found = false;
  for (const lython::parser::Diagnostic &diagnostic : emitted.diagnostics)
    found = found ||
            diagnostic.message.find("unresolved name") != std::string::npos;
  EXPECT_TRUE(found);
}

TEST(EmitterTest, RepeatedEmitIsStable) {
  for (int round = 0; round < 5; ++round) {
    mlir::MLIRContext context(testRegistry());
    lython::emitter::EmitResult emitted =
        emitSource("def f(a: int, b: int) -> int:\n    return a * b\n"
                   "print(f(6, 7))\n",
                   context);
    EXPECT_TRUE(emitted.ok()) << "round " << round;
  }
}

} // namespace
