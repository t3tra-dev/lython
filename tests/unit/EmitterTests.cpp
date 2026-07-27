#include "Driver.h"
#include "Emitter.h"
#include "Parser.h"

#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"

#include <gtest/gtest.h>
#include <string>

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

std::string emittedIR(llvm::StringRef source, mlir::MLIRContext &context) {
  lython::emitter::EmitResult emitted = emitSource(source, context);
  EXPECT_TRUE(emitted.ok());
  if (!emitted.ok() || !emitted.module)
    return {};
  std::string text;
  llvm::raw_string_ostream os(text);
  emitted.module->print(os);
  return text;
}

// A module-level `def len` wins the call, and the builtin's fast path (py.len)
// is not emitted at all. The control below is what makes this assertion
// falsifiable: without it, "no py.len" could also mean "nothing was emitted".
TEST(EmitterTest, TopLevelDefOutranksBuiltinFastPath) {
  mlir::MLIRContext context(testRegistry());
  std::string ir = emittedIR("def len(a: list[int]) -> int:\n"
                             "    return 99\n"
                             "print(len([1, 2, 3]))\n",
                             context);
  EXPECT_EQ(ir.find("py.len"), std::string::npos);
  // The emitted symbol is renamed away from the manifest's `len` binding, and
  // the reference names the symbol: py.binding.ref carries a name, and the
  // runtime lowering resolves names against the manifest first.
  EXPECT_NE(ir.find("@len$user"), std::string::npos);
  EXPECT_NE(ir.find("py.binding.ref \"len$user\""), std::string::npos);
}

TEST(EmitterTest, UnshadowedBuiltinKeepsItsFastPath) {
  mlir::MLIRContext context(testRegistry());
  std::string ir = emittedIR("print(len([1, 2, 3]))\n", context);
  EXPECT_NE(ir.find("py.len"), std::string::npos);
  EXPECT_EQ(ir.find("len$user"), std::string::npos);
}

// A local binding shadows the builtin too, and this is the axis that used to
// depend on argument count: `len` and `round` had no shadowing gate at all,
// so a bound name lost to the fast path whenever the arity matched it.
TEST(EmitterTest, LocalBindingOutranksBuiltinFastPath) {
  mlir::MLIRContext context(testRegistry());
  std::string ir = emittedIR("def user(a: list[int]) -> int:\n"
                             "    return 99\n"
                             "def go() -> int:\n"
                             "    len = user\n"
                             "    return len([1, 2])\n"
                             "print(go())\n",
                             context);
  EXPECT_EQ(ir.find("py.len"), std::string::npos);
}

// `int` names a class contract as well as a conversion fast path, and neither
// may claim the call once a top-level `def int` exists. It needs no rename:
// nothing in the manifest binds the name `int`, so the reference resolves to
// the user's func.func on its own spelling.
//
// The discriminator is the conversion's target attribute rather than the
// absence of py.type.object: the constructor route only appears in an
// intermediate state of this fix, so asserting on it would have been an
// assertion that can never go red.
TEST(EmitterTest, TopLevelDefOutranksBuiltinConversion) {
  mlir::MLIRContext context(testRegistry());
  std::string ir = emittedIR("def int(a: float) -> int:\n"
                             "    return 99\n"
                             "print(int(1.5))\n",
                             context);
  EXPECT_EQ(ir.find("@__int__"), std::string::npos);
  EXPECT_EQ(ir.find("py.type.object"), std::string::npos);
  EXPECT_NE(ir.find("py.binding.ref \"int\""), std::string::npos);
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
