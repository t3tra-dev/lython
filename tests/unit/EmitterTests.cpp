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

// What this pins: a recursive generator TYPES. Inside its own body a
// generator's name was bound to the BODY's callable, and a generator body
// returns None, so `for v in count(n - 1)` was refused at EMIT with
// "literal<None> does not provide manifest method '__iter__'" -- a message
// about nothing the reader wrote, on a program whose real obstacle is one
// stage later. The name denotes a GENERATOR there, which is what
// `publicCallable` says, and the annotated self-call resolves through the
// annotations while the signature that would answer it is still being
// computed.
//
// ⛔ The program is still refused, by the LOWERING: "yield from delegation
// exceeded the static inlining budget (recursive delegation has no static
// expansion)". That is the nested-generator frame, and it is the honest
// boundary -- which is the point of this test, since the emitter must stop
// hiding it behind a type error.
TEST(EmitterTest, ARecursiveGeneratorTypesAndLeavesTheRefusalToTheLowering) {
  mlir::MLIRContext context(testRegistry());
  lython::emitter::EmitResult emitted = emitSource(
      "from typing import Iterator\n"
      "def count(n: int) -> Iterator[int]:\n"
      "    yield n\n"
      "    if n > 0:\n"
      "        for v in count(n - 1):\n"
      "            yield v\n",
      context);
  EXPECT_TRUE(emitted.ok()) << emitted.diagnostics.size();
  for (const auto &diagnostic : emitted.diagnostics)
    EXPECT_EQ(diagnostic.message.find("literal<None>"), std::string::npos)
        << diagnostic.message;
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
TEST(EmitterTest, NamesThePrintKeywordItCannotTake) {
  mlir::MLIRContext context(testRegistry());
  // `sep` and `end` are this ladder's own join and terminator; every other
  // keyword is named rather than reported as an unmatched contract.
  lython::emitter::EmitResult end =
      emitSource("print(\"a\", end=\"\")\n", context);
  EXPECT_TRUE(end.ok());

  lython::emitter::EmitResult flush =
      emitSource("print(\"a\", flush=True)\n", context);
  EXPECT_FALSE(flush.ok());
  EXPECT_TRUE(
      reportsDiagnostic(flush, "does not take the keyword argument 'flush'"));

  lython::emitter::EmitResult unknown =
      emitSource("print(\"a\", file=None)\n", context);
  EXPECT_FALSE(unknown.ok());
  EXPECT_TRUE(
      reportsDiagnostic(unknown, "does not take the keyword argument 'file'"));
}

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

TEST(EmitterTest, ASetattrNameThatIsNotALiteralIsRefused) {
  mlir::MLIRContext context(testRegistry());
  lython::emitter::EmitResult emitted =
      emitSource("class C:\n"
                 "    def __init__(self) -> None:\n"
                 "        self.v = 1\n"
                 "c = C()\n"
                 "def rename(s: str) -> None:\n"
                 "    setattr(c, s, 2)\n",
                 context);
  EXPECT_FALSE(emitted.ok());
  bool found = false;
  for (const lython::parser::Diagnostic &diagnostic : emitted.diagnostics)
    found = found || diagnostic.message.find("literal attribute name") !=
                         std::string::npos;
  EXPECT_TRUE(found);
}

TEST(EmitterTest, ASetattrNameFoldedFromOneLiteralIsAccepted) {
  mlir::MLIRContext context(testRegistry());
  lython::emitter::EmitResult emitted =
      emitSource("class C:\n"
                 "    def __init__(self) -> None:\n"
                 "        self.v = 1\n"
                 "c = C()\n"
                 "name = \"v\"\n"
                 "setattr(c, name, 2)\n"
                 "print(c.v)\n",
                 context);
  EXPECT_TRUE(emitted.ok()) << emitted.diagnostics.size();
}

TEST(EmitterTest, AUserDefinedSetattrStillWins) {
  mlir::MLIRContext context(testRegistry());
  lython::emitter::EmitResult emitted =
      emitSource("def setattr(a: int, b: int, c: int) -> int:\n"
                 "    return a + b + c\n"
                 "print(setattr(1, 2, 3))\n",
                 context);
  EXPECT_TRUE(emitted.ok()) << emitted.diagnostics.size();
}

TEST(EmitterTest, TheExceptionChainAttributesAreRefusedWhereTheyAreWritten) {
  for (const char *attribute :
       {"__cause__", "__context__", "__suppress_context__"}) {
    mlir::MLIRContext context(testRegistry());
    lython::emitter::EmitResult emitted =
        emitSource(std::string("try:\n"
                               "    raise ValueError('v')\n"
                               "except ValueError as e:\n"
                               "    print(e.") +
                       attribute + ")\n",
                   context);
    EXPECT_FALSE(emitted.ok()) << attribute;
    bool found = false;
    for (const lython::parser::Diagnostic &diagnostic : emitted.diagnostics)
      found = found || diagnostic.message.find("no runtime implementation") !=
                           std::string::npos;
    EXPECT_TRUE(found) << attribute;
  }
}

// `__traceback__` left that group: its type is representable now and the
// traceback module builds the chain, so the refusal is about the wiring and has
// to say where the answer IS. A reader who follows the message finds a working
// call; the old one said the type did not exist.
TEST(EmitterTest, TheTracebackAttributeNamesWhatToWriteInstead) {
  mlir::MLIRContext context(testRegistry());
  lython::emitter::EmitResult emitted =
      emitSource("try:\n"
                 "    raise ValueError('v')\n"
                 "except ValueError as e:\n"
                 "    print(e.__traceback__)\n",
                 context);
  EXPECT_FALSE(emitted.ok());
  bool found = false;
  for (const lython::parser::Diagnostic &diagnostic : emitted.diagnostics)
    found = found ||
            diagnostic.message.find("traceback.format_exception(e)") !=
                std::string::npos;
  EXPECT_TRUE(found);
}

TEST(EmitterTest, AnExceptionsArgsAndItsOwnFieldsStillRead) {
  mlir::MLIRContext context(testRegistry());
  lython::emitter::EmitResult emitted =
      emitSource("class E(Exception):\n"
                 "    def __init__(self) -> None:\n"
                 "        super().__init__('x')\n"
                 "        self.code = 3\n"
                 "try:\n"
                 "    raise E()\n"
                 "except E as e:\n"
                 "    print(e.code, e.args)\n",
                 context);
  EXPECT_TRUE(emitted.ok()) << emitted.diagnostics.size();
}

TEST(EmitterTest, AnUnfrozenDataclassIsRefusedAsAKeyAndAsAnElement) {
  for (const char *use : {"d = {K(1): 2}", "s = {K(1)}"}) {
    mlir::MLIRContext context(testRegistry());
    lython::emitter::EmitResult emitted =
        emitSource(std::string("from dataclasses import dataclass\n"
                               "@dataclass\n"
                               "class K:\n"
                               "    a: int\n") +
                       use + "\n",
                   context);
    EXPECT_FALSE(emitted.ok()) << use;
    bool found = false;
    for (const lython::parser::Diagnostic &diagnostic : emitted.diagnostics)
      found = found ||
              diagnostic.message.find("unhashable type") != std::string::npos;
    EXPECT_TRUE(found) << use;
  }
}

TEST(EmitterTest, AFrozenDataclassFieldIsNotAssignableOutsideItsConstructor) {
  mlir::MLIRContext context(testRegistry());
  lython::emitter::EmitResult emitted =
      emitSource("from dataclasses import dataclass\n"
                 "@dataclass(frozen=True)\n"
                 "class K:\n"
                 "    a: int\n"
                 "k = K(1)\n"
                 "k.a = 5\n",
                 context);
  EXPECT_FALSE(emitted.ok());
  bool found = false;
  for (const lython::parser::Diagnostic &diagnostic : emitted.diagnostics)
    found = found || diagnostic.message.find("frozen dataclass") !=
                         std::string::npos;
  EXPECT_TRUE(found);
}

TEST(EmitterTest, AFrozenDataclassFillsItsFieldsInItsOwnConstructor) {
  mlir::MLIRContext context(testRegistry());
  lython::emitter::EmitResult emitted =
      emitSource("from dataclasses import dataclass\n"
                 "@dataclass(frozen=True)\n"
                 "class K:\n"
                 "    a: int\n"
                 "print(K(1).a, hash(K(1)) == hash(K(1)))\n",
                 context);
  EXPECT_TRUE(emitted.ok()) << emitted.diagnostics.size();
}

// A loop-carried name has ONE lane type. Rebinding it to an unrelated one used
// to reach the runtime lowering as "cannot adapt runtime bundle builtins.int
// with physical values (...) to expected ABI (...)", which names MLIR types and
// no source name; the refusal now happens at the loop and names the local.
TEST(EmitterTest, ALoopCarriedLocalRebornWithAnotherTypeIsRefusedAtTheLoop) {
  mlir::MLIRContext context(testRegistry());
  lython::emitter::EmitResult emitted =
      emitSource("def f() -> int:\n"
                 "    a = \"x\"\n"
                 "    t = 0\n"
                 "    for a in [1, 2]:\n"
                 "        t = t + a\n"
                 "    return t\n"
                 "print(f())\n",
                 context);
  EXPECT_FALSE(emitted.ok());
  bool found = false;
  for (const lython::parser::Diagnostic &diagnostic : emitted.diagnostics)
    found = found || diagnostic.message.find("loop-carried local 'a'") !=
                         std::string::npos;
  EXPECT_TRUE(found);
}

// The body assignment spelling of the same lane, so the check is not read as
// being about loop TARGETS.
//
// ⭐ THE READ AFTER THE LOOP IS PART OF THE SUBJECT, and was added when the
// refusal learned to ask for one. It used to end `return 0`, which retypes a
// name NOTHING observes -- and the refusal's own reason ("a loop that runs
// zero times leaves the earlier binding in place") is a claim about a read
// after the loop. With `print(a)` there the claim holds and the refusal is
// right; without it the program is legal Python that this compiler had no
// reason to reject, which the second half of this test now pins.
TEST(EmitterTest, ALoopBodyCannotRetypeACarriedLocalThatIsRead) {
  mlir::MLIRContext context(testRegistry());
  lython::emitter::EmitResult emitted =
      emitSource("def f() -> int:\n"
                 "    a = \"x\"\n"
                 "    for i in [1, 2]:\n"
                 "        a = i\n"
                 "    print(a)\n"
                 "    return 0\n"
                 "print(f())\n",
                 context);
  EXPECT_FALSE(emitted.ok());
  bool found = false;
  for (const lython::parser::Diagnostic &diagnostic : emitted.diagnostics)
    found = found || diagnostic.message.find("loop-carried local 'a'") !=
                         std::string::npos;
  EXPECT_TRUE(found);

  // Nothing observes the rebind: no read in the body, none after the loop.
  // The throwaway name is the everyday spelling of it --
  // `value, _, count = part.partition("x")` then `for _ in range(...)`.
  for (const char *source :
       {"def f() -> int:\n"
        "    a = \"x\"\n"
        "    for i in [1, 2]:\n"
        "        a = i\n"
        "    return 0\n"
        "print(f())\n",
        "def f() -> int:\n"
        "    _ = \"x\"\n"
        "    for _ in range(2):\n"
        "        pass\n"
        "    return 0\n"
        "print(f())\n",
        "def f(text: str) -> int:\n"
        "    total = 0\n"
        "    for part in text.split(\",\"):\n"
        "        value, _, count = part.partition(\"x\")\n"
        "        for _ in range(int(count)):\n"
        "            total += len(value)\n"
        "    return total\n"
        "print(f(\"ab x2\"))\n"}) {
    lython::emitter::EmitResult unobserved = emitSource(source, context);
    EXPECT_TRUE(unobserved.ok()) << source;
  }
}

// The negative control: widening a carried lane is not a retype. `n` is bound
// to a literal before the loop and to its widened contract inside it, and the
// numeric tower lets an int reach a float lane.
TEST(EmitterTest, ACarriedLaneStillTakesAWidenedValue) {
  mlir::MLIRContext context(testRegistry());
  lython::emitter::EmitResult emitted =
      emitSource("def f() -> float:\n"
                 "    total = 0.0\n"
                 "    n = 0\n"
                 "    for i in [1, 2]:\n"
                 "        n = i\n"
                 "        total = total + i\n"
                 "    return total + n\n"
                 "print(f())\n",
                 context);
  EXPECT_TRUE(emitted.ok()) << emitted.diagnostics.size();
}

// WHAT: `isinstance(o, int)` on an `object` answers, and does NOT hand the
// branch an int. The answer has to include a bool -- Python's bool is an int --
// and a boxed bool is not an int object, so viewing one as the other read
// `True + 1` as 1. The refusal is the boundary; the test above it is not.
TEST(EmitterTest, AnObjectNarrowedByAnIntTestIsStillAnObject) {
  mlir::MLIRContext context(testRegistry());
  lython::emitter::EmitResult emitted =
      emitSource("def f(o: object) -> int:\n"
                 "    if isinstance(o, int):\n"
                 "        return o + 1\n"
                 "    return -1\n"
                 "print(f(1))\n",
                 context);
  EXPECT_FALSE(emitted.ok());
  bool found = false;
  for (const lython::parser::Diagnostic &diagnostic : emitted.diagnostics)
    found = found || diagnostic.message.find("builtins.object") !=
                         std::string::npos;
  EXPECT_TRUE(found);
}

// The positive control: every other target DOES narrow, and the narrowed value
// is a view of the box's entity rather than the box.
TEST(EmitterTest, AnObjectNarrowedByAClassTestIsThatClass) {
  mlir::MLIRContext context(testRegistry());
  lython::emitter::EmitResult emitted =
      emitSource("class A:\n"
                 "    n: int\n"
                 "    def __init__(self, n: int) -> None:\n"
                 "        self.n = n\n"
                 "def f(o: object) -> int:\n"
                 "    if isinstance(o, A):\n"
                 "        return o.n\n"
                 "    return -1\n"
                 "print(f(A(1)))\n",
                 context);
  EXPECT_TRUE(emitted.ok());
}

// WHAT: a function that CAN reach its end, annotated with a result that cannot
// hold the None a fallthrough returns, is refused by name. The failure it
// replaces named neither the function nor the missing return: "callable return
// ABI expected 2 physical values, but lowering produced 0".
//
// Emit-layer and not golden: the repair is a static rejection. The controls
// are the other half -- a body that cannot reach its end has no fallthrough to
// answer for, and a result that CAN hold None still falls through.
TEST(EmitterTest, RefusesAFallthroughTheResultCannotHold) {
  mlir::MLIRContext context(testRegistry());
  lython::emitter::EmitResult missing =
      emitSource("def f(c: bool) -> str:\n"
                 "    if c:\n"
                 "        return \"a\"\n"
                 "print(f(True))\n",
                 context);
  EXPECT_FALSE(missing.ok());
  EXPECT_TRUE(reportsDiagnostic(missing, "can reach its end without returning"));

  lython::emitter::EmitResult raises =
      emitSource("def f() -> str:\n"
                 "    try:\n"
                 "        raise ValueError(\"x\")\n"
                 "    except ValueError:\n"
                 "        raise\n"
                 "f()\n",
                 context);
  EXPECT_TRUE(raises.ok());

  lython::emitter::EmitResult optional =
      emitSource("def f(c: bool) -> \"str | None\":\n"
                 "    if c:\n"
                 "        return \"a\"\n"
                 "print(f(False))\n",
                 context);
  EXPECT_TRUE(optional.ok());

  lython::emitter::EmitResult none =
      emitSource("def f(c: bool) -> None:\n"
                 "    if c:\n"
                 "        return\n"
                 "f(False)\n",
                 context);
  EXPECT_TRUE(none.ok());
}

// WHAT: `str(x)` where `x` is a name a `with` body rebinds. The name is
// promoted to storage for the duration of the statement, so its static type
// inside is the cell -- and the str/repr ladder asked THAT for a `__repr__`,
// found none, and fell through to a `repr` binding no program declares:
// "unresolved name 'repr'".
//
// Emit-layer and not golden: the failure was a static rejection of a program
// that has nothing wrong with it. The control is the other readers of the same
// name in the same position, which demote on the way in and always worked.
TEST(EmitterTest, StrSeesThroughAPromotedCell) {
  mlir::MLIRContext context(testRegistry());
  const char *body =
      "import sys\n"
      "class C:\n"
      "    def __enter__(self) -> str:\n"
      "        return \"x\"\n"
      "    def __exit__(self, a: object, b: object, c: object) -> bool:\n"
      "        return False\n"
      "count = 0\n"
      "with C() as s:\n"
      "    n = 0\n"
      "    while n < 2:\n"
      "        count += 1\n"
      "        n += 1\n";
  lython::emitter::EmitResult str =
      emitSource(std::string(body) +
                     "    sys.stdout.write(str(count) + \"\\n\")\n",
                 context);
  EXPECT_TRUE(str.ok());

  lython::emitter::EmitResult repr =
      emitSource(std::string(body) +
                     "    sys.stdout.write(repr(count) + \"\\n\")\n",
                 context);
  EXPECT_TRUE(repr.ok());

  lython::emitter::EmitResult others =
      emitSource(std::string(body) + "    print(abs(count) + 1)\n", context);
  EXPECT_TRUE(others.ok());
}

// WHAT: a `match` with an irrefutable case is exhaustive, so a function whose
// every case returns cannot reach its end. Treating a match as always
// completing asked for a return the program does not need.
//
// The controls are the two ways it is NOT exhaustive: no wildcard at all, and
// a wildcard whose guard can fail.
TEST(EmitterTest, AnExhaustiveMatchDoesNotFallThrough) {
  mlir::MLIRContext context(testRegistry());
  lython::emitter::EmitResult exhaustive =
      emitSource("def f(v: int) -> str:\n"
                 "    match v:\n"
                 "        case 0:\n"
                 "            return \"zero\"\n"
                 "        case _:\n"
                 "            return \"other\"\n"
                 "print(f(0))\n",
                 context);
  EXPECT_TRUE(exhaustive.ok());

  lython::emitter::EmitResult partial =
      emitSource("def f(v: int) -> str:\n"
                 "    match v:\n"
                 "        case 0:\n"
                 "            return \"zero\"\n"
                 "print(f(0))\n",
                 context);
  EXPECT_FALSE(partial.ok());
  EXPECT_TRUE(
      reportsDiagnostic(partial, "can reach its end without returning"));

  lython::emitter::EmitResult guarded =
      emitSource("def f(v: int) -> str:\n"
                 "    match v:\n"
                 "        case _ if v > 0:\n"
                 "            return \"pos\"\n"
                 "print(f(0))\n",
                 context);
  EXPECT_FALSE(guarded.ok());
  EXPECT_TRUE(
      reportsDiagnostic(guarded, "can reach its end without returning"));
}

// WHAT: `f(**mapping)` in every spelling -- a dict local, a dict display, a
// `**kwargs` parameter forwarded on -- against a callee that names its
// parameters and one that collects `**kwargs`. All are refused here, by the
// emitter, naming `**`. The control is the same calls with the keywords
// spelled out, which compile.
//
// Emit-layer and not golden: the refusal is the whole behaviour, and it used
// to happen eight phases later as "kw names and kw values must have the same
// size" against a fused location carrying no source line.
TEST(EmitterTest, RefusesAMappingUnpackedIntoACall) {
  mlir::MLIRContext context(testRegistry());
  const char *collector = "def show(**kwargs: int) -> int:\n"
                          "    return len(kwargs)\n";
  const char *named = "def f(a: int, b: int) -> int:\n"
                      "    return a * 10 + b\n";

  for (const std::string &source :
       {std::string(collector) + "opts = {\"a\": 7}\nprint(show(**opts))\n",
        std::string(collector) + "print(show(**{\"a\": 7}))\n",
        std::string(collector) + "def outer(**kw: int) -> int:\n"
                                 "    return show(**kw)\n"
                                 "print(outer(a=1))\n",
        std::string(named) + "opts = {\"a\": 1, \"b\": 2}\nprint(f(**opts))\n"}) {
    lython::emitter::EmitResult result = emitSource(source, context);
    EXPECT_FALSE(result.ok()) << source;
    EXPECT_TRUE(reportsDiagnostic(
        result, "`**` call arguments require statically known keyword names"))
        << source;
  }

  lython::emitter::EmitResult spelled = emitSource(
      std::string(collector) + "print(show(a=7, b=8))\n", context);
  EXPECT_TRUE(spelled.ok());
  lython::emitter::EmitResult positional =
      emitSource(std::string(named) + "print(f(1, 2))\n", context);
  EXPECT_TRUE(positional.ok());
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

// What: a class that declares `__del__` is refused, and the message says the
// finalizer would never run. Nothing calls it -- not at scope exit, not when a
// container drops its last reference, not at module teardown -- so accepting
// the class means printing one line fewer than CPython with no diagnostic.
TEST(EmitterTest, AClassWithAFinalizerIsRefused) {
  mlir::MLIRContext context(testRegistry());
  lython::emitter::EmitResult refused =
      emitSource("class R:\n"
                 "    def __init__(self, n: int) -> None:\n"
                 "        self.n = n\n"
                 "    def __del__(self) -> None:\n"
                 "        print(self.n)\n"
                 "R(1)\n",
                 context);
  EXPECT_FALSE(refused.ok());
  bool named = false;
  for (const lython::parser::Diagnostic &diagnostic : refused.diagnostics)
    named = named || diagnostic.message.find("__del__ is not supported") !=
                         std::string::npos;
  EXPECT_TRUE(named);

  mlir::MLIRContext accepting(testRegistry());
  lython::emitter::EmitResult accepted =
      emitSource("class R:\n"
                 "    def __init__(self, n: int) -> None:\n"
                 "        self.n = n\n"
                 "R(1)\n",
                 accepting);
  EXPECT_TRUE(accepted.ok());
}

// What: the two class-body hooks CPython calls implicitly. `__init_subclass__`
// runs for the subclass and needs no `@classmethod` to be written; a class
// attribute whose class defines `__set_name__` is refused, because the hook
// CPython calls at that assignment is never run here.
TEST(EmitterTest, TheImplicitClassHooksAreAccountedFor) {
  mlir::MLIRContext undecorated(testRegistry());
  lython::emitter::EmitResult accepted =
      emitSource("class Base:\n"
                 "    def __init_subclass__(cls) -> None:\n"
                 "        print(\"hook\")\n"
                 "class Sub(Base):\n"
                 "    pass\n",
                 undecorated);
  EXPECT_TRUE(accepted.ok());

  mlir::MLIRContext named(testRegistry());
  lython::emitter::EmitResult refused =
      emitSource("class Field:\n"
                 "    def __set_name__(self, owner: object, name: str) -> None:\n"
                 "        print(name)\n"
                 "class Holder:\n"
                 "    a = Field()\n",
                 named);
  EXPECT_FALSE(refused.ok());
  bool told = false;
  for (const lython::parser::Diagnostic &diagnostic : refused.diagnostics)
    told = told ||
           diagnostic.message.find("__set_name__") != std::string::npos;
  EXPECT_TRUE(told);
}

// What: three reads that used to reach the LOWERING and fail there with a
// sentence about the compiler. `(3).real` is a fold and now answers; `__doc__`
// and a class in `print()` cannot answer at all and say so about the PROGRAM.
TEST(EmitterTest, TheReadsThatUsedToFailInTheLowering) {
  mlir::MLIRContext folded(testRegistry());
  lython::emitter::EmitResult numeric =
      emitSource("print((3).real, (3).imag, (3).numerator, (3).denominator)\n",
                 folded);
  EXPECT_TRUE(numeric.ok());

  mlir::MLIRContext documented(testRegistry());
  lython::emitter::EmitResult doc =
      emitSource("print((1).__doc__)\n", documented);
  EXPECT_FALSE(doc.ok());
  bool saidDoc = false;
  for (const lython::parser::Diagnostic &diagnostic : doc.diagnostics)
    saidDoc = saidDoc ||
              diagnostic.message.find("docstrings are not retained") !=
                  std::string::npos;
  EXPECT_TRUE(saidDoc);

  mlir::MLIRContext classy(testRegistry());
  lython::emitter::EmitResult rendered =
      emitSource("print(type(1))\n", classy);
  EXPECT_FALSE(rendered.ok());
  bool saidClass = false;
  for (const lython::parser::Diagnostic &diagnostic : rendered.diagnostics)
    saidClass = saidClass ||
                diagnostic.message.find("cannot render a class") !=
                    std::string::npos;
  EXPECT_TRUE(saidClass);
}

TEST(EmitterTest, AModuleLevelLambdaCannotFreezeAReboundName) {
  // A module name bound more than once gets no cell, so a lambda that reads
  // it would carry the value it had when the lambda was built. The def
  // spelling of the same read is already refused; this one used to run and
  // print the stale value.
  mlir::MLIRContext rebound(testRegistry());
  lython::emitter::EmitResult stale =
      emitSource("x = 1\nf = lambda: x\nx = 2\nprint(f())\n", rebound);
  EXPECT_FALSE(stale.ok());
  bool saidRebound = false;
  for (const lython::parser::Diagnostic &diagnostic : stale.diagnostics)
    saidRebound = saidRebound ||
                  diagnostic.message.find("rebound after this point") !=
                      std::string::npos;
  EXPECT_TRUE(saidRebound);

  // The loop spelling is the same defect: the target is rebound every trip.
  mlir::MLIRContext looped(testRegistry());
  lython::emitter::EmitResult loop = emitSource(
      "fs = []\nfor i in range(3):\n    fs.append(lambda: i)\n", looped);
  EXPECT_FALSE(loop.ok());

  // A module name bound once is still capturable: there is nothing to go
  // stale against.
  mlir::MLIRContext once(testRegistry());
  lython::emitter::EmitResult stable =
      emitSource("xs = [1, 2, 3]\nf = lambda: len(xs)\nprint(f())\n", once);
  EXPECT_TRUE(stable.ok());
}
}
