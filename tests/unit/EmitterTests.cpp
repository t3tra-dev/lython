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

bool reportsDiagnostic(const lython::emitter::EmitResult &emitted,
                       llvm::StringRef needle) {
  for (const lython::parser::Diagnostic &diagnostic : emitted.diagnostics)
    if (diagnostic.message.find(needle.str()) != std::string::npos)
      return true;
  return false;
}

// The lazy-iterator builtins used as VALUES compile to a synthesized generator
// that walks `x[0], x[1], ... < len(x)`, because a for loop's position cell
// cannot cross a generator suspension. That walk is only iteration for a
// position-addressable sequence, and the admission gate used to be exactly
// `__len__ && __getitem__(int)` -- which a `dict[int, V]` answers, since its
// key type IS int. `iter({1: 2, 3: 4})` therefore emitted `d[0]` and raised
// KeyError: 0 at runtime; with keys 0..n-1 present it would instead have
// yielded values where CPython yields keys.
//
// Emit-layer and not golden: the repair is a static rejection, so no program
// has to run to observe it. The `for k in d:` control that this defect did NOT
// affect is already asserted by four golden cases (dict_generic_keys,
// dict_iteration_views, dict_loop_carried_inplace_mutate{,_deep}).
TEST(EmitterTest, RejectsMappingAsIndexWalkedIteratorValue) {
  mlir::MLIRContext context(testRegistry());
  lython::emitter::EmitResult emitted =
      emitSource("d = {1: 2, 3: 4}\nit = iter(d)\nprint(next(it))\n", context);
  EXPECT_FALSE(emitted.ok());
  EXPECT_TRUE(reportsDiagnostic(emitted, "requires indexable sequences"));
}

// reversed() over a mapping shared the gate and was worse: the walk starts at
// len(d)-1, so `{1: 2, 3: 4}` found key 1 present and printed the VALUE 2
// where CPython prints the key 3 -- a wrong answer with exit 0, in both value
// and loop position.
TEST(EmitterTest, RejectsMappingAsReversedSequence) {
  mlir::MLIRContext context(testRegistry());
  lython::emitter::EmitResult loopForm = emitSource(
      "d = {1: 2, 3: 4}\nfor k in reversed(d):\n    print(k)\n", context);
  EXPECT_FALSE(loopForm.ok());
  EXPECT_TRUE(reportsDiagnostic(loopForm, "requires an indexable sequence"));

  lython::emitter::EmitResult valueForm = emitSource(
      "d = {1: 2, 3: 4}\nit = reversed(d)\nprint(next(it))\n", context);
  EXPECT_FALSE(valueForm.ok());
  EXPECT_TRUE(reportsDiagnostic(valueForm, "requires indexable sequences"));
}

// The control that keeps the two rejections above from being vacuous: the gate
// must still admit real sequences, and iterating the same dict with a for
// statement -- which never took the index walk -- must still emit. Without
// this, "the mapping is rejected" would also be satisfied by rejecting
// everything.
TEST(EmitterTest, IndexWalkGateStillAdmitsSequencesAndDictForLoops) {
  mlir::MLIRContext context(testRegistry());
  EXPECT_TRUE(
      emitSource("xs = [1, 2]\nit = iter(xs)\nprint(next(it))\n", context)
          .ok());
  EXPECT_TRUE(
      emitSource("t = (1, 2)\nit = reversed(t)\nprint(next(it))\n", context)
          .ok());
  EXPECT_TRUE(
      emitSource("d = {1: 2, 3: 4}\nfor k in d:\n    print(k)\n", context).ok());
  // The remedy the rejection recommends has to compile, or the diagnostic is
  // advice that does not work.
  EXPECT_TRUE(
      emitSource("d = {1: 2, 3: 4}\nit = iter(list(d))\nprint(next(it))\n",
                 context)
          .ok());
}

// What this pins: that a yield inside a `try/except*` is refused HERE, by the
// emitter, and that it is refused for the reason it is refused.
//
// The star frame became an SSA value produced by `except_star.begin`, so it is
// live across the whole statement; a yield would make it cross a suspension and
// the generator state machine carries only what it has a lane contract for.
// Before the frame was a value the shape compiled and ran correctly, so this is
// a deliberate narrowing and the test is what records it as deliberate.
//
// Not a golden: the program never reaches lowering, and the string is the whole
// content. The control below is what keeps the rejection from being vacuous --
// the same except* without the yield must still emit, and a yield outside any
// except* must still emit.
TEST(EmitterTest, RejectsYieldInsideExceptStar) {
  mlir::MLIRContext context(testRegistry());
  lython::emitter::EmitResult emitted = emitSource(
      "from collections.abc import Generator\n"
      "def g() -> Generator[int, None, None]:\n"
      "    try:\n"
      "        raise ValueError('x')\n"
      "    except* ValueError:\n"
      "        yield 1\n",
      context);
  EXPECT_FALSE(emitted.ok());
  EXPECT_TRUE(reportsDiagnostic(emitted, "yield inside a try with except*"));
  // The reason, not just the refusal: a message that named the generator
  // lowering's straight-line restriction would be true and misleading.
  EXPECT_TRUE(reportsDiagnostic(emitted, "cross the suspension"));
}

TEST(EmitterTest, ExceptStarAndYieldAreEachStillFineApart) {
  mlir::MLIRContext context(testRegistry());
  lython::emitter::EmitResult star = emitSource(
      "try:\n"
      "    raise ValueError('x')\n"
      "except* ValueError:\n"
      "    print('caught')\n",
      context);
  EXPECT_TRUE(star.ok());

  lython::emitter::EmitResult yielding = emitSource(
      "from collections.abc import Generator\n"
      "def g() -> Generator[int, None, None]:\n"
      "    try:\n"
      "        yield 1\n"
      "    except ValueError:\n"
      "        print('caught')\n",
      context);
  EXPECT_TRUE(yielding.ok());
}

// What this pins: a call to a user-defined method or constructor is checked
// against the DECLARED parameter types, and that the check does not refuse the
// three shapes that must keep passing.
//
// A method body is inlined at its call site with the argument VALUE bound to
// the parameter name, so an argument of the wrong type used to be substituted
// into the body and the program succeeded or failed on whatever the body
// happened to do with it. `collections.Counter` is where it was found:
// `__init__` declares `list[str] | None`, `Counter({"x": 2, "y": 1})` reached
// it with a dict, `update` iterated the dict's KEYS, and every count came out
// 1 where CPython gives 2 and 1. A free function of the same signature has
// always refused the same call.
//
// Emit-layer and not golden: the repair is a static rejection. The controls
// are the other half -- int against a float parameter must still pass (the
// annotation is inert at a parameter, so the value stays an int), a subclass
// must still reach a base-typed parameter, and a SYNTHESIZED signature is
// exempt because it is this compiler's spelling rather than the program's
// (`TupleA(1) == TupleB(1)` is True in CPython while the synthesized `__eq__`
// declares Self).
TEST(EmitterTest, RejectsMethodArgumentThatViolatesTheDeclaredParameter) {
  mlir::MLIRContext context(testRegistry());
  lython::emitter::EmitResult method = emitSource(
      "class C:\n"
      "    def take(self, xs: list[str]) -> int:\n"
      "        return len(xs)\n"
      "print(C().take({\"a\": 1}))\n",
      context);
  EXPECT_FALSE(method.ok());
  EXPECT_TRUE(reportsDiagnostic(method, "argument 'xs' of 'take' is declared"));

  lython::emitter::EmitResult constructor = emitSource(
      "class C:\n"
      "    def __init__(self, xs: list[str] | None = None) -> None:\n"
      "        self.n: int = 0 if xs is None else len(xs)\n"
      "print(C({\"a\": 1}).n)\n",
      context);
  EXPECT_FALSE(constructor.ok());
  EXPECT_TRUE(
      reportsDiagnostic(constructor, "argument 'xs' of '__init__' is declared"));
}

TEST(EmitterTest, DeclaredParameterCheckStillAdmitsTheseThree) {
  mlir::MLIRContext context(testRegistry());
  // The numeric tower: int is assignable to float, and stays an int.
  EXPECT_TRUE(emitSource("class C:\n"
                         "    def scale(self, x: float) -> float:\n"
                         "        return x\n"
                         "print(C().scale(3))\n",
                         context)
                  .ok());
  // A subclass reaching a base-typed parameter.
  EXPECT_TRUE(emitSource("class B:\n"
                         "    def __init__(self) -> None:\n"
                         "        self.n: int = 1\n"
                         "class D(B):\n"
                         "    pass\n"
                         "class Holder:\n"
                         "    def take(self, b: B) -> int:\n"
                         "        return b.n\n"
                         "print(Holder().take(D()))\n",
                         context)
                  .ok());
  // A BARE generic contract accepts any instantiation of itself, which is the
  // fourth admitted shape and the one this check got wrong: a generic class's
  // own methods spell the receiver WITHOUT arguments, so
  // `def __add__(self, other: "Counter")` reached by `Counter[str] + ...` was
  // refused ("declared Counter and this call gives it Counter[str]"). It is
  // pinned by tests/golden/cases/counter_views_and_bare_generic_self.py rather
  // than here: it needs a generic class from the embedded modules, and
  // `emitSource` carries none -- `typing.Generic` is not importable without
  // them, so a synthetic Box cannot stand in.
  // A synthesized signature: NamedTuple's __eq__ declares Self, and CPython
  // compares across classes.
  EXPECT_TRUE(emitSource("from typing import NamedTuple\n"
                         "class A(NamedTuple):\n"
                         "    v: int\n"
                         "class B(NamedTuple):\n"
                         "    v: int\n"
                         "print(A(1) == B(1))\n",
                         context)
                  .ok());
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
