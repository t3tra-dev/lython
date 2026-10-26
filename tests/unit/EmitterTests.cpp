#include "Driver.h"
#include "Mangling.h"
#include "Emitter.h"
#include "Parser.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"

#include <gtest/gtest.h>
#include <string>
#include <utility>
#include <vector>

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

// What: emit a two-module program. The helper goes on DISK rather than into a
// second buffer because import resolution is a driver stage that reads the
// importer's directory, and `emitMLIRFromSource` is the earliest entry point
// that runs it -- so a diagnostic about an imported module can still be
// asserted without paying for a lowering.
struct ImportedModuleEmit {
  bool succeeded = false;
  std::string diagnostics;
};

ImportedModuleEmit
emitWithImportedModules(
    llvm::ArrayRef<std::pair<std::string, std::string>> helpers,
    llvm::StringRef mainSource, llvm::StringRef helperExtension = ".py",
    llvm::StringRef targetTriple = {}, bool jsHost = false) {
  ImportedModuleEmit result;
  llvm::SmallString<128> dir;
  if (llvm::sys::fs::createUniqueDirectory("lython-emit-import", dir)) {
    result.diagnostics = "could not create a temporary import directory";
    return result;
  }
  std::vector<llvm::SmallString<128>> helperPaths;
  for (const auto &[helperName, helperSource] : helpers) {
    llvm::SmallString<128> helperPath(dir);
    llvm::sys::path::append(helperPath,
                            llvm::Twine(helperName) + helperExtension);
    helperPaths.push_back(helperPath);
    std::error_code error;
    llvm::raw_fd_ostream out(helperPath, error);
    if (error) {
      result.diagnostics = "could not write the helper module";
      return result;
    }
    out << helperSource;
  }
  llvm::SmallString<128> mainPath(dir);
  llvm::sys::path::append(mainPath, "main.py");

  mlir::MLIRContext context(testRegistry());
  mlir::OwningOpRef<mlir::ModuleOp> module;
  llvm::raw_string_ostream diag(result.diagnostics);
  lython::driver::DriverOptions options;
  options.targetTriple = targetTriple.empty()
                             ? llvm::sys::getDefaultTargetTriple()
                             : targetTriple.str();
  options.jsHost = jsHost;
  result.succeeded = mlir::succeeded(lython::driver::emitMLIRFromSource(
      mainSource, mainPath, dir, options, context, module, diag));
  for (const llvm::SmallString<128> &helperPath : helperPaths)
    llvm::sys::fs::remove(helperPath);
  llvm::sys::fs::remove(dir);
  return result;
}

ImportedModuleEmit
emitWithImportedModule(llvm::StringRef helperName, llvm::StringRef helperSource,
                       llvm::StringRef mainSource,
                       llvm::StringRef helperExtension = ".py",
                       llvm::StringRef targetTriple = {},
                       bool jsHost = false) {
  return emitWithImportedModules({{helperName.str(), helperSource.str()}},
                                 mainSource, helperExtension, targetTriple,
                                 jsHost);
}

// What: sixteen modules, each importing every module before it, emit. Each
// import used to read the signatures of everything the imported module imports
// again, level by level -- twelve such modules took over four minutes, and
// CPython's unittest package never finished.
TEST(EmitterTest, AModuleGraphWhereEachImportsAllEarlierOnesEmits) {
  std::vector<std::pair<std::string, std::string>> modules;
  for (int index = 0; index < 16; ++index) {
    std::string source;
    for (int earlier = 0; earlier < index; ++earlier)
      source += "import m" + std::to_string(earlier) + "\n";
    std::string self = std::to_string(index);
    source += "def f" + self + "(x: int) -> int:\n    return " +
              (index == 0 ? std::string("x")
                          : "m" + std::to_string(index - 1) + ".f" +
                                std::to_string(index - 1) + "(x) + 1") +
              "\n";
    source += "def g" + self + "(x: int) -> int:\n    return f" + self +
              "(x) * 2\n";
    modules.emplace_back("m" + self, std::move(source));
  }
  ImportedModuleEmit emitted =
      emitWithImportedModules(modules, "import m15\nprint(m15.g15(1))\n");
  EXPECT_TRUE(emitted.succeeded) << emitted.diagnostics;
}

// What: an imported module's body does not run, so a statement in it that WOULD
// have run is refused instead of dropped.
TEST(EmitterTest, RefusesAStatementInAnImportedModuleBody) {
  ImportedModuleEmit emitted = emitWithImportedModule(
      "sider", "ITEMS: \"list[int]\" = []\nITEMS.append(1)\n",
      "import sider\nprint(sider.ITEMS)\n");
  EXPECT_FALSE(emitted.succeeded);
  EXPECT_NE(emitted.diagnostics.find(
                "a module-level statement in an imported module is not "
                "supported"),
            std::string::npos)
      << emitted.diagnostics;
}

// What: an error in a method of the program that another module's code calls
// -- the hook a base class of that module calls -- names the program's file
// and line, and no line of the other module.
TEST(EmitterTest, NamesTheProgramForAnErrorInAMethodImportedCodeCalls) {
  ImportedModuleEmit emitted = emitWithImportedModule(
      "hookbase",
      "class Base:\n"
      "    def go(self) -> str:\n"
      "        out = \"base\"\n"
      "        return self.hook() + out\n"
      "\n"
      "    def hook(self) -> str:\n"
      "        return \"\"\n",
      "import hookbase\n"
      "\n"
      "\n"
      "class Sub(hookbase.Base):\n"
      "    def hook(self) -> str:\n"
      "        return undefined_name\n"
      "\n"
      "\n"
      "print(Sub().go())\n");
  EXPECT_FALSE(emitted.succeeded);
  EXPECT_NE(emitted.diagnostics.find(
                "main.py:6:15: emit error: unresolved name 'undefined_name'"),
            std::string::npos)
      << emitted.diagnostics;
  EXPECT_EQ(emitted.diagnostics.find("hookbase.py"), std::string::npos)
      << emitted.diagnostics;
}

// What: a module the program does not import is not a name in it, though a
// module it imports imports it: `re.sub(...)`, `re.sub` as a value and `re`
// alone are each an unresolved name; with `import re` they compile.
TEST(EmitterTest, AModuleAnImportedModuleImportsIsNotANameHere) {
  const char *helper = "import re\n\n\n"
                       "def tidy(s: str) -> str:\n"
                       "    return re.sub(\"a\", \"b\", s)\n";
  for (const char *use :
       {"print(re.sub(\"x\", \"y\", \"xx\"))\n", "f = re.sub\nprint(f)\n",
        "x = re\n"}) {
    ImportedModuleEmit emitted = emitWithImportedModule(
        "tidier", helper,
        std::string("import tidier\n\nprint(tidier.tidy(\"a\"))\n") + use);
    EXPECT_FALSE(emitted.succeeded) << use;
    EXPECT_NE(emitted.diagnostics.find("unresolved name 're'"),
              std::string::npos)
        << use << emitted.diagnostics;
  }
  ImportedModuleEmit imported = emitWithImportedModule(
      "tidier", helper,
      "import re\nimport tidier\n\nprint(tidier.tidy(\"a\"), "
      "re.sub(\"x\", \"y\", \"xx\"))\n");
  EXPECT_TRUE(imported.succeeded) << imported.diagnostics;
}

// What: an imported module's `assert` whose test the compiler can decide is
// decided: a true one compiles, a false one is refused as the AssertionError
// the import raises, and one over a run-time value is refused.
TEST(EmitterTest, DecidesAnAssertInAnImportedModuleBody) {
  ImportedModuleEmit holds = emitWithImportedModule(
      "asserts_true", "X = 80\nY = X - (5 + 12)\nassert Y >= 0 and X // 8 == 10\n",
      "import asserts_true\nprint(asserts_true.X)\n");
  EXPECT_TRUE(holds.succeeded) << holds.diagnostics;
  ImportedModuleEmit fails = emitWithImportedModule(
      "asserts_false", "X = 80\nassert X < 0\n",
      "import asserts_false\nprint(asserts_false.X)\n");
  EXPECT_FALSE(fails.succeeded);
  EXPECT_NE(fails.diagnostics.find(
                "this assert fails when the module is imported"),
            std::string::npos)
      << fails.diagnostics;
  ImportedModuleEmit runtime = emitWithImportedModule(
      "asserts_runtime", "import sys\nassert len(sys.argv) > 5\n",
      "import asserts_runtime\nprint(1)\n");
  EXPECT_FALSE(runtime.succeeded);
  EXPECT_NE(runtime.diagnostics.find(
                "an assert in an imported module needs a test this compiler "
                "can decide"),
            std::string::npos)
      << runtime.diagnostics;
}

// What: a union stored where one class is declared, with a member that is not
// that class, is refused naming the member; a union of its subclasses is not.
TEST(EmitterTest, RefusesAUnionStoredWhereOneOfItsMembersIsNotTheField) {
  mlir::MLIRContext context(testRegistry());
  lython::emitter::EmitResult refused = emitSource(
      "class A:\n    pass\n\nclass B:\n    pass\n\n"
      "class H:\n    def __init__(self, a: A) -> None:\n        self.a = a\n\n"
      "def f(v: A | B) -> None:\n    h = H(A())\n    h.a = v\n",
      context);
  EXPECT_FALSE(refused.ok());
  bool named = false;
  for (const lython::parser::Diagnostic &diagnostic : refused.diagnostics)
    named = named || diagnostic.message.find(
                         "its member B is not one") != std::string::npos;
  EXPECT_TRUE(named);
  mlir::MLIRContext second(testRegistry());
  lython::emitter::EmitResult subclasses = emitSource(
      "class A:\n    pass\n\nclass B(A):\n    pass\n\nclass C(A):\n    pass\n\n"
      "class H:\n    def __init__(self, a: A) -> None:\n        self.a = a\n\n"
      "def f(v: B | C) -> None:\n    h = H(A())\n    h.a = v\n",
      second);
  EXPECT_TRUE(subclasses.ok());
}

// What: what an imported module's decorated method or function-local class
// cannot be is refused in that module's file: a diagnostic with no file was
// printed against the main module's name.
TEST(EmitterTest, AnImportedModuleIsRefusedInItsOwnFile) {
  ImportedModuleEmit decorated = emitWithImportedModule(
      "decorated",
      "from typing import Callable\n\n"
      "def wrapped[T](f: Callable[[T, int], int]) -> Callable[[T, int], int]:\n"
      "    def inner(x: T, n: int) -> int:\n        return f(x, n)\n"
      "    return inner\n\n"
      "class A:\n"
      "    @wrapped\n"
      "    def m(self, n: int = 2) -> int:\n        return n\n",
      "from decorated import A\nprint(A().m(1))\n");
  EXPECT_FALSE(decorated.succeeded);
  EXPECT_NE(decorated.diagnostics.find(
                "decorated.py:9:5: emit error: method 'm' declares default "
                "values"),
            std::string::npos)
      << decorated.diagnostics;

  ImportedModuleEmit local = emitWithImportedModule(
      "local",
      "def f() -> int:\n"
      "    class A:\n"
      "        items: list[int] = []\n"
      "    return 1\n",
      "import local\nprint(local.f())\n");
  EXPECT_FALSE(local.succeeded);
  EXPECT_NE(local.diagnostics.find("local.py:3:8: emit error: class 'A' is "
                                   "written in a function"),
            std::string::npos)
      << local.diagnostics;
}

// What: a decorator on a function in an imported module. It was dropped and the
// undecorated function answered under the decorated name, so the refusal is
// what keeps the wrong answer from happening.
TEST(EmitterTest, RefusesADroppedDecoratorInAnImportedModule) {
  // A plain NAME decorator naming a def in the same module is `f = d(f)`,
  // which the module-global initializer queue runs at the start of `__main__`
  // -- so it compiles now, and the golden pins what it answers.
  ImportedModuleEmit applied = emitWithImportedModule(
      "wrapped",
      "from typing import Callable\n\n"
      "def twice(f: \"Callable[[int], int]\") -> \"Callable[[int], int]\":\n"
      "    def inner(n: int) -> int:\n        return f(n) * 2\n"
      "    return inner\n\n"
      "@twice\ndef scaled(n: int) -> int:\n    return n + 1\n",
      "import wrapped\nprint(wrapped.scaled(1))\n");
  EXPECT_TRUE(applied.succeeded) << applied.diagnostics;

  // ⛔ A decorator that is not a plain NAME is still refused: the application
  // is folded from the decorator's own signature, and a FACTORY (`@d(arg)`)
  // is one more call whose intermediate value has none to fold. The sentence
  // is what it always was.
  ImportedModuleEmit factory = emitWithImportedModule(
      "borrowed",
      "from typing import Callable\n\n"
      "def scale(k: int) -> \"Callable[[Callable[[int], int]], "
      "Callable[[int], int]]\":\n"
      "    def deco(f: \"Callable[[int], int]\") -> "
      "\"Callable[[int], int]\":\n"
      "        def inner(n: int) -> int:\n            return f(n) * k\n"
      "        return inner\n"
      "    return deco\n\n"
      "@scale(2)\ndef scaled(n: int) -> int:\n    return n + 1\n",
      "import borrowed\nprint(borrowed.scaled(1))\n");
  ImportedModuleEmit &foreign = factory;
  EXPECT_FALSE(foreign.succeeded);
  EXPECT_NE(foreign.diagnostics.find(
                "a decorator on a function in an imported module is not "
                "supported"),
            std::string::npos)
      << foreign.diagnostics;
}

// What: the floor the two refusals above stand on -- an imported module of
// plain declarations still emits, so what they refuse is what they name.
TEST(EmitterTest, EmitsAnImportedModuleOfDeclarations) {
  ImportedModuleEmit emitted = emitWithImportedModule(
      "plain",
      "\"a docstring\"\n\nCOUNT = 3\n\n"
      "def only(n: int) -> int:\n    return n + COUNT\n\n"
      "class Holder:\n    def __init__(self, v: int) -> None:\n"
      "        self.v = v\n",
      "import plain\nprint(plain.only(1), plain.Holder(2).v)\n");
  EXPECT_TRUE(emitted.succeeded) << emitted.diagnostics;
}

// What: the same refusal for the branch nobody can choose -- a module-level
// `if` whose test this compiler cannot decide selects no branch at all.
TEST(EmitterTest, RefusesAnUndecidableIfInAnImportedModuleBody) {
  ImportedModuleEmit emitted = emitWithImportedModule(
      "chooser",
      "import sys\n\n"
      "if len(sys.argv) > 1:\n"
      "    def pick() -> int:\n        return 1\n"
      "else:\n"
      "    def pick() -> int:\n        return 2\n",
      "import chooser\nprint(chooser.pick())\n");
  EXPECT_FALSE(emitted.succeeded);
  EXPECT_NE(emitted.diagnostics.find(
                "a module-level 'if' in an imported module needs a test this "
                "compiler can decide"),
            std::string::npos)
      << emitted.diagnostics;
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
  // The emitted symbol is the definition's mangled one, away from the
  // manifest's `len` binding, and the reference names the symbol:
  // py.binding.ref carries a name, and the runtime lowering resolves names
  // against the manifest first.
  std::string symbol = py::mangling::pythonDefinitionSymbol("__main__", "len");
  EXPECT_NE(ir.find("@" + symbol), std::string::npos);
  EXPECT_NE(ir.find("py.binding.ref \"" + symbol + "\""), std::string::npos);
  EXPECT_EQ(ir.find("py.binding.ref \"len\""), std::string::npos);
}

// What: a large method called on an object the call site constructed is a call
// of the method's own symbol, while a small one beside it is still copied into
// the caller -- which is what makes "no reference" mean "copied" rather than
// "nothing emitted".
TEST(EmitterTest, ALargeMethodIsCalledAndASmallOneCopied) {
  mlir::MLIRContext context(testRegistry());
  std::string ir = emittedIR("class Tally:\n"
                             "    def __init__(self) -> None:\n"
                             "        self.n = 0\n"
                             "    def tiny(self) -> int:\n"
                             "        return self.n\n"
                             "    def work(self, k: int) -> int:\n"
                             "        total = 0\n"
                             "        for i in range(k):\n"
                             "            if i % 2 == 0:\n"
                             "                total += i * 3 + 1\n"
                             "            elif i % 3 == 0:\n"
                             "                total -= i * 2 - 1\n"
                             "            else:\n"
                             "                total += self.n + i\n"
                             "        for j in range(k, 0, -1):\n"
                             "            if total > j * 10:\n"
                             "                total = total // 2 + j\n"
                             "            else:\n"
                             "                total = total * 2 - j\n"
                             "        self.n = total\n"
                             "        return total + self.n * 2 - k\n"
                             "t = Tally()\n"
                             "print(t.work(3))\n"
                             "print(t.tiny())\n",
                             context);
  std::string work =
      py::mangling::pythonDefinitionSymbol("__main__", "Tally.work");
  std::string tiny =
      py::mangling::pythonDefinitionSymbol("__main__", "Tally.tiny");
  EXPECT_NE(ir.find("@" + work), std::string::npos);
  EXPECT_NE(ir.find("@" + tiny), std::string::npos);
  EXPECT_NE(ir.find("py.binding.ref \"" + work + "\""), std::string::npos);
  EXPECT_EQ(ir.find("py.binding.ref \"" + tiny + "\""), std::string::npos);
}

// What: a function is refused where a callable of another numeric rung is
// declared -- its result (int held as `Callable[[], float]`) or a parameter (a
// float parameter the declaration passes an int) -- naming which; a bool result
// held as int is accepted.
TEST(EmitterTest, ACallableIsRefusedWhereAnotherNumericRungIsDeclared) {
  const char *prelude = "from collections.abc import Callable\n"
                        "def one() -> int:\n    return 1\n"
                        "def half(x: float) -> float:\n    return x / 2\n"
                        "def flag() -> bool:\n    return True\n";
  ImportedModuleEmit result = emitWithImportedModules(
      {}, std::string(prelude) + "f: Callable[[], float] = one\n");
  EXPECT_FALSE(result.succeeded);
  EXPECT_NE(result.diagnostics.find(
                "main.py:8:0: emit error: this function cannot be held as the "
                "callable declared here: its result is int and the "
                "declaration's is float"),
            std::string::npos)
      << result.diagnostics;
  ImportedModuleEmit parameter = emitWithImportedModules(
      {}, std::string(prelude) + "h: Callable[[int], float] = half\n");
  EXPECT_FALSE(parameter.succeeded);
  EXPECT_NE(parameter.diagnostics.find("its parameter 1 is float and the "
                                       "declaration passes int"),
            std::string::npos)
      << parameter.diagnostics;
  ImportedModuleEmit subclass = emitWithImportedModules(
      {}, std::string(prelude) + "b: Callable[[], int] = flag\nprint(b())\n");
  EXPECT_TRUE(subclass.succeeded) << subclass.diagnostics;
}

// What: a module-level class whose base is a class of the module defined
// further down is refused at the base, naming both classes; the same two
// classes in the other order emit.
TEST(EmitterTest, AClassBaseDefinedFurtherDownIsRefused) {
  ImportedModuleEmit later = emitWithImportedModules(
      {}, "class A(Later):\n    pass\n\n\nclass Later:\n    pass\n");
  EXPECT_FALSE(later.succeeded);
  EXPECT_NE(later.diagnostics.find("main.py:1:8: emit error: class 'A' names "
                                   "'Later' as a base before the class "
                                   "statement that defines 'Later' has run"),
            std::string::npos)
      << later.diagnostics;
  ImportedModuleEmit earlier = emitWithImportedModules(
      {}, "class Later:\n    pass\n\n\nclass A(Later):\n    pass\n");
  EXPECT_TRUE(earlier.succeeded) << earlier.diagnostics;
}

// What: a list, set or dict display of subclass instances is returned where
// the base's container is declared; an element of an unrelated class is still
// refused at the return.
TEST(EmitterTest, AReturnedDisplayIsJudgedByItsElements) {
  const char *classes = "class Base:\n    pass\n"
                        "class Sub(Base):\n    pass\n"
                        "class Other:\n    pass\n";
  ImportedModuleEmit displays = emitWithImportedModules(
      {}, std::string(classes) +
              "def l() -> list[Base]:\n    return [Sub()]\n"
              "def s() -> set[Base]:\n    return {Sub()}\n"
              "def d() -> dict[str, Base]:\n    return {'a': Sub()}\n"
              "print(len(l()), len(s()), len(d()))\n");
  EXPECT_TRUE(displays.succeeded) << displays.diagnostics;
  ImportedModuleEmit unrelated = emitWithImportedModules(
      {}, std::string(classes) +
              "def l() -> list[Base]:\n    return [Other()]\n"
              "print(len(l()))\n");
  EXPECT_FALSE(unrelated.succeeded);
  EXPECT_NE(unrelated.diagnostics.find("main.py:8:4: emit error: function is "
                                       "annotated to return"),
            std::string::npos)
      << unrelated.diagnostics;
}

// What: a generic function or method declared to return `Box[T]` returns
// `Box(v)`, the class built from its own type parameter.
TEST(EmitterTest, AGenericClassBuiltFromATypeParameterIsReturned) {
  ImportedModuleEmit result = emitWithImportedModules(
      {}, "class Box[T]:\n"
          "    def __init__(self, v: T) -> None:\n        self.v = v\n"
          "class Case:\n"
          "    def make[T](self, v: T) -> Box[T]:\n        return Box(v)\n"
          "def make[T](v: T) -> Box[T]:\n    return Box(v)\n"
          "print(make(3).v + 1, Case().make('x').v.upper())\n");
  EXPECT_TRUE(result.succeeded) << result.diagnostics;
}

// What: a generic class written through its module in an annotation
// (`types.GeneratorType[int, None, None]`) takes its arguments, as the same
// class imported by name does.
TEST(EmitterTest, ADottedGenericAnnotationTakesItsArguments) {
  mlir::MLIRContext context(testRegistry());
  std::string ir = emittedIR("import types\n"
                             "def gen() -> types.GeneratorType[int, None, None]:\n"
                             "    yield 1\n"
                             "g: types.GeneratorType[int, None, None] = gen()\n"
                             "print(next(g) + 1)\n",
                             context);
  EXPECT_NE(ir.find("py.global.set \"g\""), std::string::npos) << ir;
  EXPECT_NE(ir.find("!py.contract<\"types.GeneratorType\", "
                    "[!py.contract<\"builtins.int\">"),
            std::string::npos)
      << ir;
}

// What: a tuple display is passed to a generator method's `tuple[int, ...]`
// parameter, and a list of subclass instances to its `list[Base]` one.
TEST(EmitterTest, AGeneratorMethodTakesArgumentsAtItsParameterTypes) {
  ImportedModuleEmit result = emitWithImportedModules(
      {}, "from collections.abc import Iterator\n"
          "class Base:\n    pass\n"
          "class Sub(Base):\n    pass\n"
          "class G:\n"
          "    def walk(self, xs: tuple[int, ...]) -> Iterator[int]:\n"
          "        for x in xs:\n            yield x\n"
          "    def each(self, bs: list[Base]) -> Iterator[Base]:\n"
          "        for b in bs:\n            yield b\n"
          "print(list(G().walk((1, 2, 3))), len(list(G().each([Sub()]))))\n");
  EXPECT_TRUE(result.succeeded) << result.diagnostics;
}

// What: a lambda passed to a builtin container's method is typed by the
// parameter it reaches -- `append` and `insert` on a list of callables whose
// result is `object` -- where it has no type of its own.
TEST(EmitterTest, ALambdaPassedToAContainerMethodTakesTheParameterType) {
  ImportedModuleEmit result = emitWithImportedModules(
      {}, "from collections.abc import Callable\n"
          "def value() -> object:\n    return 3\n"
          "fs: list[Callable[[], object]] = []\n"
          "fs.append(lambda: value())\n"
          "fs.insert(0, lambda: value())\n"
          "print([f() for f in fs])\n");
  EXPECT_TRUE(result.succeeded) << result.diagnostics;
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
// may claim the call once a top-level `def int` exists: the reference names
// the user's func.func by its mangled symbol.
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
  EXPECT_NE(ir.find("py.binding.ref \"" +
                    py::mangling::pythonDefinitionSymbol("__main__", "int") +
                    "\""),
            std::string::npos);
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
// What: `sep`, `end`, `file` and `flush` are print's own keywords and compile
// (`file=None` is sys.stdout); any other keyword is named rather than reported
// as an unmatched contract.
TEST(EmitterTest, NamesThePrintKeywordItCannotTake) {
  mlir::MLIRContext context(testRegistry());
  for (const char *source :
       {"print(\"a\", end=\"\")\n", "print(\"a\", flush=True)\n",
        "print(\"a\", file=None)\n",
        "import sys\nprint(\"a\", 1, sep=\",\", file=sys.stderr)\n"}) {
    lython::emitter::EmitResult accepted = emitSource(source, context);
    EXPECT_TRUE(accepted.ok()) << source;
  }

  lython::emitter::EmitResult unknown =
      emitSource("print(\"a\", color=1)\n", context);
  EXPECT_FALSE(unknown.ok());
  EXPECT_TRUE(
      reportsDiagnostic(unknown, "does not take the keyword argument 'color'"));
}

// A computed attribute name is answered as a call or by hasattr/dir; each
// shape it cannot answer exactly is refused at the call, with the reason.
TEST(EmitterTest, AComputedAttributeNameIsRefusedWhereItCannotBeAnswered) {
  mlir::MLIRContext context(testRegistry());
  const char *classes = "class T:\n"
                        "    def go(self, a: int) -> int:\n"
                        "        return a\n"
                        "def f() -> int:\n"
                        "    return 1\n";
  for (const char *use :
       {"def use(t: T, n: str) -> None:\n"
        "    print(getattr(t, n)(1), hasattr(t, n))\n",
        "print(dir(T), dir(T()))\n"}) {
    std::string source = std::string(classes) + use;
    lython::emitter::EmitResult accepted = emitSource(source, context);
    EXPECT_TRUE(accepted.ok()) << source;
  }
  struct Refused {
    const char *use;
    const char *reason;
  };
  for (const Refused &refused : std::initializer_list<Refused>{
           {"def use(t: T, n: str) -> None:\n    m = getattr(t, n)\n",
            "is supported as the call `getattr(x, name)(...)`"},
           {"def use(t: T, n: str) -> None:\n    print(getattr(t, n)(f()))\n",
            "takes only names and constants as arguments"},
           {"def use(t: T, n: str) -> None:\n    print(getattr(t, n)(a=1))\n",
            "takes no keyword arguments"},
           {"def use(n: str) -> None:\n    print(getattr(5, n)())\n",
            "needs a receiver whose class this program declares"},
           {"print(dir())\n", "dir() without an argument"},
           {"from dataclasses import dataclass\n@dataclass\nclass D:\n"
            "    x: int\nprint(dir(D))\n",
            "has a decorator"},
           {"class E(Exception):\n    pass\nprint(dir(E))\n",
            "derives from 'Exception'"}}) {
    std::string source = std::string(classes) + refused.use;
    lython::emitter::EmitResult result = emitSource(source, context);
    EXPECT_FALSE(result.ok()) << source;
    EXPECT_TRUE(reportsDiagnostic(result, refused.reason)) << source;
  }
}

// object's own __eq__/__ne__ answer NotImplemented for two different
// objects, which has no value here; the call is refused with that reason.
TEST(EmitterTest, ObjectsOwnEqualityIsRefusedForItsNotImplemented) {
  mlir::MLIRContext context(testRegistry());
  lython::emitter::EmitResult accepted = emitSource(
      "class A:\n    pass\na = A()\n"
      "print(object.__repr__(a), object.__str__(a), object.__hash__(a))\n",
      context);
  EXPECT_TRUE(accepted.ok());
  for (const char *spelling : {"__eq__", "__ne__"}) {
    std::string source = std::string("class A:\n    pass\n"
                                     "print(object.") +
                         spelling + "(A(), A()))\n";
    lython::emitter::EmitResult result = emitSource(source, context);
    EXPECT_FALSE(result.ok()) << spelling;
    EXPECT_TRUE(reportsDiagnostic(result, "answers NotImplemented"))
        << spelling;
  }
}

// R1 refuses the truth test of an int the program typed as an int; a number
// held by a parameter declared `object` is tested as the object it is there.
TEST(EmitterTest, ImplicitTruthOfAnIntIsRefusedOnlyWhereTheIntIsDeclared) {
  mlir::MLIRContext context(testRegistry());
  lython::emitter::EmitResult held = emitSource(
      "class T:\n"
      "    def check(self, expr: object) -> bool:\n"
      "        return not expr\n"
      "print(T().check(3), T().check(0.0))\n",
      context);
  EXPECT_TRUE(held.ok());
  for (const char *declared : {"int", "float"}) {
    std::string source = std::string("class T:\n"
                                     "    def check(self, n: ") +
                         declared +
                         ") -> bool:\n"
                         "        return not n\n"
                         "print(T().check(3))\n";
    lython::emitter::EmitResult result = emitSource(source, context);
    EXPECT_FALSE(result.ok()) << declared;
    EXPECT_TRUE(reportsDiagnostic(result, "implicit truthiness of"))
        << declared;
  }
}

// What: a decorator whose application is not a function is refused by name at
// a module-level def -- the name is rebound to what the decorator returns,
// and leaving the undecorated body under it would answer the wrong thing.
TEST(EmitterTest, ADecoratorThatReturnsNoFunctionIsRefusedByName) {
  mlir::MLIRContext context(testRegistry());
  lython::emitter::EmitResult refused = emitSource(
      "from typing import Callable\n"
      "def to_int(f: Callable[[int], int]) -> int:\n"
      "    return f(10)\n"
      "@to_int\n"
      "def ten(n: int) -> int:\n"
      "    return n + 1\n"
      "print(ten)\n",
      context);
  EXPECT_FALSE(refused.ok());
  EXPECT_TRUE(reportsDiagnostic(
      refused, "decorator 'to_int' does not return a function here"));
}

// What: a class decorator that does not hand back the class it decorates is
// refused by name, where one that does compiles.
TEST(EmitterTest, AClassDecoratorMustReturnItsClass) {
  mlir::MLIRContext context(testRegistry());
  lython::emitter::EmitResult refused = emitSource(
      "def describe[T](cls: type[T]) -> str:\n"
      "    return 'x'\n"
      "@describe\n"
      "class A:\n"
      "    pass\n",
      context);
  EXPECT_FALSE(refused.ok());
  EXPECT_TRUE(reportsDiagnostic(
      refused, "class decorator 'describe' does not return the class it "
               "decorates"));
  lython::emitter::EmitResult kept = emitSource(
      "def keep[T](cls: type[T]) -> type[T]:\n"
      "    return cls\n"
      "@keep\n"
      "class B:\n"
      "    pass\n",
      context);
  EXPECT_TRUE(kept.ok());
}

// What: a decorated method compiles when its decorator returns a function,
// and one whose decorator returns anything else is refused by name, with no
// second sentence about the cell the rewrite reads. A method with defaults
// keeps them through a decorator that hands the function back, and is
// refused under one whose result does not take them.
TEST(EmitterTest, ADecoratedMethodIsWhatItsDecoratorReturns) {
  mlir::MLIRContext accepting(testRegistry());
  lython::emitter::EmitResult accepted = emitSource(
      "from typing import TypeVar\n"
      "F = TypeVar(\"F\")\n"
      "def mark(f: F) -> F:\n"
      "    return f\n"
      "class A:\n"
      "    @mark\n"
      "    def m(self, x: int = 1) -> int:\n"
      "        return x\n"
      "print(A().m(), A().m(2))\n",
      accepting);
  EXPECT_TRUE(accepted.ok());

  mlir::MLIRContext refusing(testRegistry());
  lython::emitter::EmitResult refused = emitSource(
      "def to_int(f: object) -> int:\n"
      "    return 1\n"
      "class A:\n"
      "    @to_int\n"
      "    def m(self) -> int:\n"
      "        return 1\n"
      "print(A().m())\n",
      refusing);
  EXPECT_FALSE(refused.ok());
  EXPECT_TRUE(reportsDiagnostic(
      refused, "decorator 'to_int' on method 'm' does not return a function"));
  EXPECT_FALSE(reportsDiagnostic(refused, "unresolved name"));

  mlir::MLIRContext wrapping(testRegistry());
  lython::emitter::EmitResult dropped = emitSource(
      "from typing import Callable\n"
      "def wrapped[T](f: Callable[[T, int], int]) -> Callable[[T, int], int]:\n"
      "    def inner(x: T, n: int) -> int:\n"
      "        return f(x, n)\n"
      "    return inner\n"
      "class A:\n"
      "    @wrapped\n"
      "    def m(self, n: int = 2) -> int:\n"
      "        return n\n"
      "print(A().m())\n",
      wrapping);
  EXPECT_FALSE(dropped.ok());
  EXPECT_TRUE(reportsDiagnostic(
      dropped, "method 'm' declares default values, and what the decorator "
               "returns does not take them"));
}

// What: a decorator factory generic in a parameter only its result names
// compiles where its result is applied at once -- a decorator, or `f(r)(g)`
// -- and a call that holds the factory's result is refused naming why. A
// decorator whose result changes the method's signature is refused naming
// both signatures.
TEST(EmitterTest, AGenericDecoratorFactoryIsSolvedWhereItIsApplied) {
  const char *factory =
      "from typing import Callable\n"
      "def skip[F](reason: str) -> Callable[[F], F]:\n"
      "    def decorator(f: F) -> F:\n"
      "        return f\n"
      "    return decorator\n";
  mlir::MLIRContext accepting(testRegistry());
  lython::emitter::EmitResult accepted = emitSource(
      std::string(factory) +
          "class Case:\n"
          "    @skip(\"why\")\n"
          "    def test(self) -> None:\n"
          "        print(1)\n"
          "@skip(\"module\")\n"
          "def g(x: int) -> int:\n"
          "    return x\n"
          "Case().test()\n"
          "print(g(1), skip(\"now\")(g)(2))\n",
      accepting);
  EXPECT_TRUE(accepted.ok());
  for (const lython::parser::Diagnostic &diagnostic : accepted.diagnostics)
    ADD_FAILURE() << diagnostic.message;

  mlir::MLIRContext holding(testRegistry());
  lython::emitter::EmitResult held = emitSource(
      std::string(factory) + "d = skip(\"later\")\n", holding);
  EXPECT_FALSE(held.ok());
  EXPECT_TRUE(reportsDiagnostic(
      held, "this call's result is generic in a type parameter no argument "
            "names"));

  mlir::MLIRContext changing(testRegistry());
  lython::emitter::EmitResult changed = emitSource(
      "from typing import Callable\n"
      "def as_text[T](f: Callable[[T], int]) -> Callable[[T], str]:\n"
      "    def wrapper(x: T) -> str:\n"
      "        return str(f(x))\n"
      "    return wrapper\n"
      "class Case:\n"
      "    @as_text\n"
      "    def count(self) -> int:\n"
      "        return 5\n"
      "print(Case().count())\n",
      changing);
  EXPECT_FALSE(changed.ok());
  EXPECT_TRUE(reportsDiagnostic(
      changed, "method 'count' is (Case) -> int and its decorator makes it "
               "(Case) -> str"));
}

// What: `sys.exc_info()` outside an `except` block is refused, saying where
// it is answered: which exception is being handled there is the handler
// active at run time.
// What: written in an except block's body, `sys.exc_info()` is the triple
// without None; anywhere else -- a function a handler calls -- it is
// typeshed's optional triple, answered at run time, so a member is narrowed
// before it is used.
TEST(EmitterTest, SysExcInfoIsOptionalOutsideAnExceptBlock) {
  lython::driver::DriverOptions native;
  native.targetTriple = llvm::sys::getDefaultTargetTriple();
  auto emits = [&](const char *source, std::string &diagnostics) {
    mlir::MLIRContext context(testRegistry());
    mlir::OwningOpRef<mlir::ModuleOp> module;
    llvm::raw_string_ostream diag(diagnostics);
    return mlir::succeeded(lython::driver::emitMLIRFromSource(
        source, "main.py", "<lython-no-import-dir>", native, context, module,
        diag));
  };
  std::string inside;
  EXPECT_TRUE(emits("import sys\ntry:\n    raise KeyError('k')\n"
                    "except KeyError:\n"
                    "    print(sys.exc_info()[0].__name__)\n",
                    inside))
      << inside;
  std::string narrowed;
  EXPECT_TRUE(emits("import sys\ndef report() -> str:\n"
                    "    etype, value, tb = sys.exc_info()\n"
                    "    if etype is None:\n        return 'none'\n"
                    "    return etype.__name__\nprint(report())\n",
                    narrowed))
      << narrowed;
  std::string unnarrowed;
  EXPECT_FALSE(emits("import sys\ndef report() -> str:\n"
                     "    etype, value, tb = sys.exc_info()\n"
                     "    return etype.__name__\nprint(report())\n",
                     unnarrowed));
}

// What: `Container` is covariant, as typeshed declares it, so a method taking
// `Container[object]` accepts a list[int], a dict, a set and a tuple, while a
// `Container[str]` parameter still refuses a list[int].
TEST(EmitterTest, AContainerOfObjectsTakesAContainerOfAnything) {
  mlir::MLIRContext accepting(testRegistry());
  lython::emitter::EmitResult accepted = emitSource(
      "from typing import Container\n"
      "class T:\n"
      "    def has(self, member: object, c: Container[object]) -> bool:\n"
      "        return member in c\n"
      "t = T()\n"
      "print(t.has(1, [1, 2]), t.has(\"k\", {\"k\": 1}), t.has(2, {2}),\n"
      "      t.has(3, (1, 2)))\n",
      accepting);
  EXPECT_TRUE(accepted.ok());

  mlir::MLIRContext refusing(testRegistry());
  lython::emitter::EmitResult refused = emitSource(
      "from typing import Container\n"
      "class T:\n"
      "    def has(self, member: str, c: Container[str]) -> bool:\n"
      "        return member in c\n"
      "print(T().has(\"a\", [1, 2]))\n",
      refusing);
  EXPECT_FALSE(refused.ok());
  EXPECT_TRUE(reportsDiagnostic(refused, "argument 'c' of 'has' is declared"));
}

// A method forwarding its packs is specialized at each call; what cannot be
// forwarded exactly -- a dict the body rewrites, a list starred into a method,
// the method taken as a value -- is refused with the reason.
TEST(EmitterTest, AMethodForwardingItsPacksIsRefusedWhereItCannotBeExact) {
  mlir::MLIRContext context(testRegistry());
  const char *prefix =
      "from typing import Callable\n"
      "def show(a: int, c: float = 1.5) -> str:\n"
      "    return str(a)\n"
      "class T:\n"
      "    def call(self, f: Callable[..., str], *args: object,\n"
      "             **kwargs: object) -> str:\n";
  lython::emitter::EmitResult forwarded = emitSource(
      std::string(prefix) + "        return f(*args, **kwargs)\n"
                            "print(T().call(show, 1), T().call(show, 2, c=0.5))\n",
      context);
  EXPECT_TRUE(forwarded.ok());
  struct Refused {
    const char *body;
    const char *reason;
  };
  for (const Refused &refused : std::initializer_list<Refused>{
           {"        kwargs[\"c\"] = 9.5\n"
            "        return f(*args, **kwargs)\n"
            "print(T().call(show, 1))\n",
            "its keys could name 'c' of 'show', which Python binds first"},
           {"        return f(*args, **kwargs)\n"
            "xs = [1]\n"
            "print(T().call(show, *xs))\n",
            "its first members would fill 'a' and 'c' of 'show'"},
           {"        return f(*args, **kwargs)\n"
            "m = T().call\n",
            "it forwards its *args or **kwargs"}}) {
    std::string source = std::string(prefix) + refused.body;
    lython::emitter::EmitResult result = emitSource(source, context);
    EXPECT_FALSE(result.ok()) << source;
    EXPECT_TRUE(reportsDiagnostic(result, refused.reason)) << source;
  }
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

// A method called on a union is answered by every member or refused: the
// refusal names the member that has no such method, and a method that can
// reallocate its receiver is refused by that reason.
TEST(EmitterTest, AMethodOnAUnionIsAnsweredByEveryMemberOrRefused) {
  {
    mlir::MLIRContext context(testRegistry());
    lython::emitter::EmitResult emitted =
        emitSource("class A:\n"
                   "    def size(self, k: int) -> int:\n"
                   "        return k\n"
                   "class B:\n"
                   "    def size(self, k: int) -> str:\n"
                   "        return str(k)\n"
                   "def pick(b: bool) -> A | B:\n"
                   "    if b:\n"
                   "        return A()\n"
                   "    return B()\n"
                   "print(pick(True).size(k=1))\n",
                   context);
    EXPECT_TRUE(emitted.ok()) << emitted.diagnostics.size();
  }
  {
    mlir::MLIRContext context(testRegistry());
    lython::emitter::EmitResult emitted =
        emitSource("def maybe(b: bool) -> str | None:\n"
                   "    if b:\n"
                   "        return 'x'\n"
                   "    return None\n"
                   "print(maybe(True).upper())\n",
                   context);
    EXPECT_FALSE(emitted.ok());
    bool found = false;
    for (const lython::parser::Diagnostic &diagnostic : emitted.diagnostics)
      found = found ||
              (diagnostic.message.find("for the member !py.literal<None>") !=
                   std::string::npos &&
               diagnostic.message.find("narrow the value") !=
                   std::string::npos);
    EXPECT_TRUE(found);
  }
  {
    mlir::MLIRContext context(testRegistry());
    lython::emitter::EmitResult emitted =
        emitSource("def seqs(b: bool) -> list[int] | bytearray:\n"
                   "    if b:\n"
                   "        return [1]\n"
                   "    return bytearray(b'a')\n"
                   "xs = seqs(True)\n"
                   "xs.append(2)\n",
                   context);
    EXPECT_FALSE(emitted.ok());
    bool found = false;
    for (const lython::parser::Diagnostic &diagnostic : emitted.diagnostics)
      found = found || diagnostic.message.find("can reallocate the object") !=
                           std::string::npos;
    EXPECT_TRUE(found);
  }
}

// A program that replaces sys.stdout refuses what would read the stream it
// replaced: an object of another kind assigned to it, a `from sys import
// stdout` snapshot, and a print() whose arguments are unpacked.
TEST(EmitterTest, AReplacedStreamRefusesWhatItCannotWriteThrough) {
  struct Case {
    const char *source;
    const char *message;
  };
  for (const Case &item : {
           Case{"import sys\n"
                "sys.stdout = 'not a stream'\n",
                "sys.stdout can be replaced here by an io.StringIO"},
           Case{"import sys\n"
                "from sys import stdout\n"
                "sys.stdout = sys.__stdout__\n",
                "would bind the stream as it is at the import"},
           Case{"import sys\n"
                "xs = [1, 2]\n"
                "sys.stdout = sys.__stdout__\n"
                "print(*xs)\n",
                "in a program that replaces sys.stdout"},
       }) {
    mlir::MLIRContext context(testRegistry());
    lython::emitter::EmitResult emitted = emitSource(item.source, context);
    EXPECT_FALSE(emitted.ok()) << item.source;
    bool found = false;
    for (const lython::parser::Diagnostic &diagnostic : emitted.diagnostics)
      found = found ||
              diagnostic.message.find(item.message) != std::string::npos;
    EXPECT_TRUE(found) << item.source;
  }
}

// The chain attributes read the words a raise left in the exception; an
// assignment to one is refused at the store, naming the raise that writes it.
TEST(EmitterTest, TheExceptionChainIsReadButWrittenOnlyByARaise) {
  for (const char *attribute :
       {"__cause__", "__context__", "__suppress_context__"}) {
    mlir::MLIRContext context(testRegistry());
    lython::emitter::EmitResult read =
        emitSource(std::string("try:\n"
                               "    raise ValueError('v')\n"
                               "except ValueError as e:\n"
                               "    print(e.") +
                       attribute + ")\n",
                   context);
    EXPECT_TRUE(read.ok()) << attribute;
  }
  for (const char *attribute : {"__cause__", "__context__",
                                "__suppress_context__", "__traceback__"}) {
    mlir::MLIRContext context(testRegistry());
    lython::emitter::EmitResult written =
        emitSource(std::string("e = ValueError('v')\n"
                               "e.") +
                       attribute + " = None\n",
                   context);
    EXPECT_FALSE(written.ok()) << attribute;
    bool found = false;
    for (const lython::parser::Diagnostic &diagnostic : written.diagnostics)
      found = found || diagnostic.message.find("cannot be assigned") !=
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

// WHAT: `isinstance(o, int)` and `isinstance(o, bool)` on an `object` narrow
// the branch to the class they test: `o + 1` takes an int and `not o` a bool,
// where both used to be refused as `builtins.object` -- an int test accepts a
// bool, and the int view of a bool is exact now that bool objects carry the
// int layout.
TEST(EmitterTest, AnObjectNarrowedByAnIntOrBoolTestIsThatClass) {
  mlir::MLIRContext context(testRegistry());
  lython::emitter::EmitResult emitted =
      emitSource("def f(o: object) -> int:\n"
                 "    if isinstance(o, bool):\n"
                 "        return 0 if not o else 1\n"
                 "    if isinstance(o, int):\n"
                 "        return o + 1\n"
                 "    return -1\n"
                 "print(f(1))\n",
                 context);
  EXPECT_TRUE(emitted.ok()) << emitted.diagnostics.size();
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
// A `**` mapping whose keys only the run time knows reaches a function's
// **kwargs -- a dict, a display, a forwarded **kwargs, a generic function's
// or method's -- and is refused where its keys could name a parameter, which
// is named in the message.
TEST(EmitterTest, AMappingOfUnknownKeysReachesOnlyKwargs) {
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
        std::string("def keys[V](**kwargs: V) -> int:\n"
                    "    return len(kwargs)\n"
                    "d = {\"a\": 1.5}\n"
                    "print(keys(**d))\n"),
        std::string("class Bag:\n"
                    "    def lines[V](self, **parts: V) -> list[str]:\n"
                    "        return [k + str(v) for k, v in parts.items()]\n"
                    "d = {\"a\": 1.5}\n"
                    "print(Bag().lines(**d), Bag().lines(z=1, **d))\n")}) {
    lython::emitter::EmitResult result = emitSource(source, context);
    EXPECT_TRUE(result.ok()) << source;
  }

  lython::emitter::EmitResult refused = emitSource(
      std::string(named) + "opts = {\"a\": 1, \"b\": 2}\nprint(f(**opts))\n",
      context);
  EXPECT_FALSE(refused.ok());
  EXPECT_TRUE(reportsDiagnostic(refused, "its keys could name 'a' and 'b' of "
                                         "'f', which Python binds first"));

  lython::emitter::EmitResult spelled = emitSource(
      std::string(collector) + "print(show(a=7, b=8))\n", context);
  EXPECT_TRUE(spelled.ok());
  lython::emitter::EmitResult positional =
      emitSource(std::string(named) + "print(f(1, 2))\n", context);
  EXPECT_TRUE(positional.ok());
}

// A starred sequence whose length only the run time knows reaches *args -- a
// generic function's or method's too -- and is refused where its first
// members would fill a positional parameter, after which nothing positional
// may follow it, and before a builtin.
TEST(EmitterTest, AStarredSequenceOfUnknownLengthReachesOnlyArgs) {
  mlir::MLIRContext context(testRegistry());
  struct Case {
    const char *source;
    const char *message;
  };
  for (const Case &item : {
           Case{"def f(a: int = 0, *rest: int) -> int:\n"
                "    return a\n"
                "xs = [1]\n"
                "print(f(*xs))\n",
                "its first members would fill 'a' of 'f'"},
           Case{"def h(*rest: int) -> int:\n"
                "    return len(rest)\n"
                "xs = [1]\n"
                "print(h(*xs, 2))\n",
                "must be the last positional argument"},
           Case{"xs = [1, 2]\n"
                "print(*xs)\n",
                "only to the *args of a function or method this program "
                "defines"},
       }) {
    mlir::MLIRContext caseContext(testRegistry());
    lython::emitter::EmitResult result = emitSource(item.source, caseContext);
    EXPECT_FALSE(result.ok()) << item.source;
    EXPECT_TRUE(reportsDiagnostic(result, item.message)) << item.source;
  }
  lython::emitter::EmitResult forwarded = emitSource(
      "def total(*args: int) -> int:\n"
      "    return len(args)\n"
      "def relay(*args: int) -> int:\n"
      "    return total(*args)\n"
      "def count[T](*args: T) -> int:\n"
      "    return len(args)\n"
      "class Bag:\n"
      "    def count[T](self, *items: T) -> int:\n"
      "        return len(items)\n"
      "xs = [1, 2]\n"
      "print(total(*xs), total(0, *xs), relay(3, 4), count(*xs),\n"
      "      Bag().count(*xs), Bag().count(0, *xs))\n",
      context);
  EXPECT_TRUE(forwarded.ok());
}

// A function forwarding `*args: P.args` and `**kwargs: P.kwargs` is compiled
// at each call's shape; a call whose shape only the run knows, a callable that
// does not take what the call passes on, and a use that is not a call, are
// refused, as is `Concatenate` outside `Callable`.
TEST(EmitterTest, AParamSpecForwarderIsSpecializedAtEachCall) {
  const char *forwarder =
      "from collections.abc import Callable\n"
      "def call[**P, R](f: Callable[P, R], *args: P.args,\n"
      "                 **kwargs: P.kwargs) -> R:\n"
      "    return f(*args, **kwargs)\n"
      "def add(a: int, b: int = 0) -> int:\n"
      "    return a + b\n";
  {
    mlir::MLIRContext context(testRegistry());
    lython::emitter::EmitResult called = emitSource(
        std::string(forwarder) + "print(call(add, 1), call(add, 1, b=2))\n",
        context);
    EXPECT_TRUE(called.ok());
  }
  struct Case {
    std::string source;
    const char *message;
  };
  for (const Case &item : {
           Case{std::string(forwarder) + "xs = [1, 2]\nprint(call(add, *xs))\n",
                "cannot fill `*args: P.args` or `**kwargs: P.kwargs`"},
           Case{std::string(forwarder) +
                    "d = {\"b\": 2}\nprint(call(add, 1, **d))\n",
                "cannot fill `*args: P.args` or `**kwargs: P.kwargs`"},
           Case{std::string(forwarder) + "g: Callable[..., int] = call\n",
                "specialized at a call and has no value of its own"},
           Case{std::string(forwarder) + "print(call(add, 1, c=2))\n",
                "the callable given for 'f' does not take what this call "
                "passes on to it"},
           Case{"from typing import Concatenate\n"
                "def f(x: Concatenate[int, str]) -> None:\n"
                "    pass\n",
                "`Concatenate[...]` is meaningful only as the first argument "
                "of `Callable`"},
       }) {
    mlir::MLIRContext context(testRegistry());
    lython::emitter::EmitResult result = emitSource(item.source, context);
    EXPECT_FALSE(result.ok()) << item.source;
    EXPECT_TRUE(reportsDiagnostic(result, item.message)) << item.source;
  }
}

// A generic function is specialized at every type it is called with, however
// many there are; what is refused is an instantiation that keeps nesting
// inside the one being emitted -- polymorphic recursion.
TEST(EmitterTest, AGenericIsSpecializedAtEveryTypeButNotWithoutEnd) {
  std::string many = "def ident[T](x: T) -> T:\n"
                     "    return x\n";
  for (int arity = 1; arity <= 40; ++arity) {
    std::string tuple = "(";
    for (int index = 0; index < arity; ++index)
      tuple += "1, ";
    many += "print(len(ident(" + tuple + "))))\n";
  }
  {
    mlir::MLIRContext context(testRegistry());
    lython::emitter::EmitResult result = emitSource(many, context);
    EXPECT_TRUE(result.ok());
  }
  mlir::MLIRContext context(testRegistry());
  lython::emitter::EmitResult mutual = emitSource(
      "def f[T](x: T, n: int) -> int:\n"
      "    if n == 0:\n"
      "        return 0\n"
      "    return g([x], n - 1)\n"
      "def g[T](x: T, n: int) -> int:\n"
      "    if n == 0:\n"
      "        return 0\n"
      "    return f([x], n - 1)\n"
      "print(f(1, 3))\n",
      context);
  EXPECT_FALSE(mutual.ok());
  EXPECT_TRUE(reportsDiagnostic(
      mutual, "too many nested instantiations (polymorphic recursion?)"));
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
// sentence about the compiler. `(3).real` is a fold and a class in `print()`
// renders through its type object; `(1).__doc__` is a builtin's docstring,
// which is not retained, and the refusal names the class.
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
              diagnostic.message.find(
                  "the docstring of 'int' is not retained") !=
                  std::string::npos;
  EXPECT_TRUE(saidDoc);

  mlir::MLIRContext classy(testRegistry());
  lython::emitter::EmitResult rendered =
      emitSource("print(type(1))\n", classy);
  EXPECT_TRUE(rendered.ok());
}

// What: every spelling that asks for a class's text -- print(), str(),
// repr(), an f-string, over a class name and over type(x) -- emits; the text
// they all produce is golden.cases.every_way_to_render_a_class.
TEST(EmitterTest, EveryWayToRenderAClassEmits) {
  for (const char *source : {"print(int)\n", "print(type([1]))\n",
                             "print(str(int))\n", "print(repr(int))\n",
                             "xs = [1]\nprint(str(type(xs)))\n",
                             "print(f\"{int}\")\n",
                             "xs = [1]\nprint(f\"{type(xs)}\")\n",
                             "class A:\n    pass\n\n\nprint(str(A))\n"}) {
    mlir::MLIRContext context(testRegistry());
    lython::emitter::EmitResult result = emitSource(source, context);
    EXPECT_TRUE(result.ok()) << source;
  }
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

  // The loop spelling is REPAIRED rather than refused: a target a callable in
  // the body reads gets a cell before the loop, so the capture is by
  // reference and reads the last value, as CPython does.
  mlir::MLIRContext looped(testRegistry());
  lython::emitter::EmitResult loop = emitSource(
      "fs = []\nfor i in range(3):\n    fs.append(lambda: i)\n", looped);
  EXPECT_TRUE(loop.ok());

  // A module name bound once is still capturable: there is nothing to go
  // stale against.
  mlir::MLIRContext once(testRegistry());
  lython::emitter::EmitResult stable =
      emitSource("xs = [1, 2, 3]\nf = lambda: len(xs)\nprint(f())\n", once);
  EXPECT_TRUE(stable.ok());
}

// What: a value of a class unrelated to the declared one is refused where it is
// stored -- an annotated name, and a field `__init__` typed int that another
// method stores a str into -- while a subclass is still stored where its base,
// or a builtin exception two levels above it, is declared. The unrelated store
// used to retype the value, and the str's payload printed as an integer.
TEST(EmitterTest, AValueOfAnUnrelatedClassIsRefusedWhereAnotherIsDeclared) {
  for (const char *source :
       {"x: int = 1\nx = \"hello\"\nprint(x)\n",
        "class C:\n    def __init__(self) -> None:\n        self.x = 1\n"
        "    def b(self) -> None:\n        self.x = \"s\"\n"
        "c = C()\nc.b()\nprint(c.x)\n"}) {
    mlir::MLIRContext context(testRegistry());
    lython::emitter::EmitResult result = emitSource(source, context);
    EXPECT_FALSE(result.ok()) << source;
    bool named = false;
    for (const lython::parser::Diagnostic &diagnostic : result.diagnostics)
      named = named ||
              diagnostic.message.find(
                  "a str value cannot be stored where int is declared") !=
                  std::string::npos;
    EXPECT_TRUE(named) << source;
  }
  mlir::MLIRContext context(testRegistry());
  lython::emitter::EmitResult subclass = emitSource(
      "class B:\n    pass\nclass D(B):\n    pass\nb: B = D()\n"
      "class Err(Exception):\n    pass\ne: BaseException = Err(\"x\")\n"
      "print(b, e)\n",
      context);
  EXPECT_TRUE(subclass.ok());
}

// What: `getattr(x, "name", default)` is refused when the receiver's class
// lacks the name and a subclass declares it -- the class line cannot say which
// one the instance is -- and names that subclass.
TEST(EmitterTest, AGetattrDefaultASubclassMayAnswerIsRefused) {
  mlir::MLIRContext context(testRegistry());
  lython::emitter::EmitResult result = emitSource(
      "class B:\n    pass\nclass D(B):\n    extra = 1\n"
      "def f(b: B) -> int:\n    return getattr(b, \"extra\", 0)\n"
      "print(f(D()))\n",
      context);
  EXPECT_FALSE(result.ok());
  bool named = false;
  for (const lython::parser::Diagnostic &diagnostic : result.diagnostics)
    named = named ||
            diagnostic.message.find(
                "getattr(x, 'extra', default) cannot be answered: 'D' "
                "derives from") != std::string::npos;
  EXPECT_TRUE(named);
}

// What: a call or an attribute read through a class value that may name ANY
// class (`t: type`) is refused -- every class of the program would be an arm
// -- while the same through a `type[Base]` compiles.
TEST(EmitterTest, ACallThroughAClassValueThatMayNameAnyClassIsRefused) {
  struct Case {
    const char *source;
    const char *message;
  };
  for (const Case &shape :
       {Case{"def f(t: type) -> object:\n    return t()\nprint(f(int))\n",
             "calling through a class value that may name any class is "
             "not supported"},
        Case{"def f(t: type) -> int:\n    return t.x\n",
             "reading 'x' through a class value that may name any class is "
             "not supported"}}) {
    mlir::MLIRContext context(testRegistry());
    lython::emitter::EmitResult result = emitSource(shape.source, context);
    EXPECT_FALSE(result.ok()) << shape.source;
    bool told = false;
    for (const lython::parser::Diagnostic &diagnostic : result.diagnostics)
      told = told ||
             diagnostic.message.find(shape.message) != std::string::npos;
    EXPECT_TRUE(told) << shape.source;
  }

  mlir::MLIRContext bounded(testRegistry());
  lython::emitter::EmitResult accepted = emitSource(
      "class A:\n    x = 1\nclass B(A):\n    x = 2\n"
      "def f(t: type[A]) -> int:\n    return t().x + t.x\n"
      "print(f(B))\n",
      bounded);
  EXPECT_TRUE(accepted.ok());
}

// What: a class value is stored only where a class value of one of its bases
// is declared: an unrelated class and a step along the numeric tower are
// refused (a `type[int]` naming bool would build an int), a subclass is not.
TEST(EmitterTest, AClassValueIsStoredOnlyWhereItsBaseIsDeclared) {
  for (const char *source :
       {"class A:\n    pass\nclass B:\n    pass\nt: type[A] = B\n",
        "def f() -> None:\n    t: type[int] = bool\n    print(t(1))\n"}) {
    mlir::MLIRContext context(testRegistry());
    lython::emitter::EmitResult result = emitSource(source, context);
    EXPECT_FALSE(result.ok()) << source;
    bool told = false;
    for (const lython::parser::Diagnostic &diagnostic : result.diagnostics)
      told = told || diagnostic.message.find("value cannot be stored where "
                                             "type[") != std::string::npos;
    EXPECT_TRUE(told) << source;
  }

  mlir::MLIRContext subclass(testRegistry());
  lython::emitter::EmitResult accepted = emitSource(
      "class A:\n    pass\nclass B(A):\n    pass\n"
      "def f() -> str:\n    t: type[A] = B\n    return t.__name__\n",
      subclass);
  EXPECT_TRUE(accepted.ok());
}

// What: a generic method has a body only where a call solves it, so the
// method read as a value is refused at emit with a sentence that says why;
// an ordinary call compiles, and so does the method calling itself, which is
// a call of the body its solution already has.
TEST(EmitterTest, AGenericMethodHasABodyOnlyWhereItIsCalled) {
  struct Case {
    const char *source;
    const char *message;
  };
  for (const Case &shape :
       {Case{"class C:\n"
             "    def show[T](self, x: T) -> str:\n"
             "        return repr(x)\n"
             "m = C().show\n",
             "the generic method 'show' cannot be used as a value"}}) {
    mlir::MLIRContext context(testRegistry());
    lython::emitter::EmitResult result = emitSource(shape.source, context);
    EXPECT_FALSE(result.ok()) << shape.source;
    bool told = false;
    for (const lython::parser::Diagnostic &diagnostic : result.diagnostics)
      told = told ||
             diagnostic.message.find(shape.message) != std::string::npos;
    EXPECT_TRUE(told) << shape.source;
  }

  mlir::MLIRContext called(testRegistry());
  lython::emitter::EmitResult accepted = emitSource(
      "class C:\n"
      "    def same[T](self, a: T, b: T) -> bool:\n"
      "        return a == b\n"
      "    def count[T](self, xs: list[T], i: int) -> int:\n"
      "        if i == len(xs):\n"
      "            return 0\n"
      "        return 1 + self.count(xs, i + 1)\n"
      "print(C().same(1, 1), C().same(\"a\", \"b\"), C().count([1, 2], 0))\n",
      called);
  EXPECT_TRUE(accepted.ok());
}

// What: `@contextmanager` on a function with spelled-out parameters compiles,
// and so does one with *args or **kwargs, whose packs the generator takes as
// values; one that also has positional-only parameters is refused by name.
// Through the driver, because `import contextlib` resolves lib/contextlib.py.
TEST(EmitterTest, AContextManagerFunctionHandsItsParametersOnByName) {
  lython::driver::DriverOptions native;
  native.targetTriple = llvm::sys::getDefaultTargetTriple();
  auto emits = [&](const char *source, std::string &diagnostics) {
    mlir::MLIRContext context(testRegistry());
    mlir::OwningOpRef<mlir::ModuleOp> module;
    llvm::raw_string_ostream diag(diagnostics);
    return mlir::succeeded(lython::driver::emitMLIRFromSource(
        source, "main.py", "<lython-no-import-dir>", native, context, module,
        diag));
  };
  std::string spelled;
  EXPECT_TRUE(emits("from contextlib import contextmanager\n"
                    "from typing import Iterator\n"
                    "@contextmanager\n"
                    "def tag(name: str, *, depth: int = 0) -> Iterator[int]:\n"
                    "    yield len(name) + depth\n"
                    "with tag(\"a\", depth=1) as n:\n"
                    "    print(n)\n",
                    spelled))
      << spelled;

  std::string packed;
  EXPECT_TRUE(emits("import contextlib\n"
                    "from typing import Iterator\n"
                    "@contextlib.contextmanager\n"
                    "def sub(*labels: str, **params: int) -> Iterator[int]:\n"
                    "    yield len(labels) + len(params)\n"
                    "with sub(\"a\", i=1) as n:\n"
                    "    print(n)\n",
                    packed))
      << packed;

  std::string refused;
  EXPECT_FALSE(emits("import contextlib\n"
                     "from typing import Iterator\n"
                     "@contextlib.contextmanager\n"
                     "def sub(a: int, /, **params: int) -> Iterator[int]:\n"
                     "    yield a\n",
                     refused));
  EXPECT_NE(refused.find("@contextmanager on 'sub' is not supported with both "
                         "positional-only parameters and *args or **kwargs"),
            std::string::npos)
      << refused;
}

TEST(EmitterTest, WhatAClassDefiningGetattributeCannotDoIsRefused) {
  // A method call on such an instance would be looked up through
  // __getattribute__ first; calling the declared method directly is refused.
  mlir::MLIRContext called(testRegistry());
  lython::emitter::EmitResult call = emitSource(
      "class C:\n    def __getattribute__(self, name: str) -> str:\n"
      "        return name\n    def m(self) -> int:\n        return 1\n"
      "print(C().m())\n",
      called);
  EXPECT_FALSE(call.ok());
  EXPECT_TRUE(reportsDiagnostic(
      call, "'C' defines __getattribute__, which CPython asks for 'm' before "
            "calling it; a method call cannot be resolved around it "
            "statically"));

  // With __getattr__ beside it, a read would need the fallback CPython takes
  // when __getattribute__ raises AttributeError.
  mlir::MLIRContext both(testRegistry());
  lython::emitter::EmitResult paired = emitSource(
      "class C:\n    def __getattribute__(self, name: str) -> str:\n"
      "        return name\n    def __getattr__(self, name: str) -> str:\n"
      "        return name\nprint(C().x)\n",
      both);
  EXPECT_FALSE(paired.ok());
  EXPECT_TRUE(reportsDiagnostic(
      paired, "'C' defines both __getattribute__ and __getattr__; reading "
              "'x' would need __getattr__ called when __getattribute__ raises "
              "AttributeError, which is not supported"));
}

TEST(EmitterTest, AnAttributeASourceClassDoesNotHaveIsRefusedAtEmit) {
  // Nothing resolves the read -- not a field, a class attribute, a method or
  // a property -- and it used to be emitted anyway and die in the lowering as
  // "class C has no field 'missing'".
  mlir::MLIRContext missing(testRegistry());
  lython::emitter::EmitResult absent = emitSource(
      "class C:\n    def __init__(self) -> None:\n        self.a = 1\n"
      "print(C().missing)\n",
      missing);
  EXPECT_FALSE(absent.ok());
  bool saidAttribute = false;
  for (const lython::parser::Diagnostic &diagnostic : absent.diagnostics)
    saidAttribute =
        saidAttribute ||
        diagnostic.message.find("has no attribute 'missing'") !=
            std::string::npos;
  EXPECT_TRUE(saidAttribute);

  // A class that declares __getattr__ is answered by it: the read is the
  // call, and is not refused.
  mlir::MLIRContext hooked(testRegistry());
  lython::emitter::EmitResult hook = emitSource(
      "class C:\n    def __init__(self) -> None:\n        self.a = 1\n"
      "    def __getattr__(self, name: str) -> int:\n        return 0\n"
      "print(C().missing)\n",
      hooked);
  EXPECT_TRUE(hook.ok());

  // An attribute a manifest BASE provides is not this: the lowering resolves
  // it from the base's schema.
  mlir::MLIRContext derived(testRegistry());
  lython::emitter::EmitResult inherited = emitSource(
      "class MyError(Exception):\n    pass\n"
      "try:\n    raise MyError(\"x\")\nexcept MyError as e:\n"
      "    print(e.args)\n",
      derived);
  EXPECT_TRUE(inherited.ok());
}

// What: a module name used where a value goes. It has no runtime object, and
// the refusal names the module rather than arriving as "unresolved runtime
// binding 'sys'" from the lowering.
TEST(EmitterTest, AModuleUsedAsAValueIsRefusedAtEmit) {
  for (const char *source :
       {"import sys\nL = sys\nprint(L.platform)\n",
        "import sys\nprint(sys)\n",
        "import math\nxs = [math]\n"}) {
    mlir::MLIRContext context(testRegistry());
    lython::emitter::EmitResult result = emitSource(source, context);
    EXPECT_FALSE(result.ok()) << source;
    bool named = false;
    for (const lython::parser::Diagnostic &diagnostic : result.diagnostics)
      named = named || diagnostic.message.find("is not a value: a module has "
                                               "no runtime object") !=
                           std::string::npos;
    EXPECT_TRUE(named) << source;
  }
}

// What: a local reassigned in one match case while a sibling case leaves the
// loop. The write is carried by a cell the match allocates, and a jump leaves
// the match without passing the point that reads it back -- so the shape is
// refused rather than answered with the pre-match value.
TEST(EmitterTest, RefusesAMatchWriteThatCannotReachAJump) {
  mlir::MLIRContext context(testRegistry());
  lython::emitter::EmitResult result = emitSource(
      "def f(xs: \"list[int]\") -> str:\n"
      "    out = \"\"\n"
      "    for v in xs:\n"
      "        match v:\n"
      "            case 1:\n                out = out + \"one\"\n"
      "            case 2:\n                break\n"
      "            case _:\n                out = out + \"?\"\n"
      "    return out\n\n\nprint(f([1, 2]))\n",
      context);
  EXPECT_FALSE(result.ok());
  bool named = false;
  for (const lython::parser::Diagnostic &diagnostic : result.diagnostics)
    named = named || diagnostic.message.find(
                         "is reassigned in a match case while another case "
                         "leaves the loop") != std::string::npos;
  EXPECT_TRUE(named);
}

// What: an Optional assigned to a loop-carried local whose lane type is the
// non-optional member. The None arm was dropped into that lane and the value
// read back as the member, which is how every walk of a linked structure
// answered as if the list never ended.
TEST(EmitterTest, RefusesAUnionRebindOfALoopCarriedLocal) {
  mlir::MLIRContext context(testRegistry());
  lython::emitter::EmitResult result = emitSource(
      "class Node:\n"
      "    def __init__(self, v: int) -> None:\n"
      "        self.v = v\n"
      "        self.nxt: \"Node | None\" = None\n"
      "\n"
      "\n"
      "def f() -> int:\n"
      "    head = Node(0)\n"
      "    cur = head\n"
      "    i = 0\n"
      "    while i < 1:\n"
      "        cur = cur.nxt\n"
      "        i = i + 1\n"
      "    if cur is None:\n"
      "        return 1\n"
      "    return 2\n"
      "\n"
      "\n"
      "print(f())\n",
      context);
  EXPECT_FALSE(result.ok());
  bool named = false;
  for (const lython::parser::Diagnostic &diagnostic : result.diagnostics)
    named = named || diagnostic.message.find(
                         "loop-carried local 'cur' is bound to") !=
                         std::string::npos;
  EXPECT_TRUE(named);
}

// What: a statement in a class body that would have run. A class body executes
// in CPython and declares here, so a call or a loop in one was dropped without
// a word -- `log.append("body")` in a class body appended nothing.
TEST(EmitterTest, RefusesAStatementInAClassBody) {
  for (const char *source :
       {"log: \"list[str]\" = []\n\n\nclass C:\n    log.append(\"body\")\n"
        "\n    def f(self) -> int:\n        return 1\n\n\nprint(log)\n",
        "class C:\n    for i in range(3):\n        pass\n\n"
        "    def f(self) -> int:\n        return 1\n\n\nprint(1)\n"}) {
    mlir::MLIRContext context(testRegistry());
    lython::emitter::EmitResult result = emitSource(source, context);
    EXPECT_FALSE(result.ok()) << source;
    bool named = false;
    for (const lython::parser::Diagnostic &diagnostic : result.diagnostics)
      named = named || diagnostic.message.find(
                           "statement in a class body is not supported") !=
                           std::string::npos;
    EXPECT_TRUE(named) << source;
  }
}

// What: a class declared inside another class. The refusals for a class
// statement elsewhere live where statements are emitted, which never sees a
// class body, so this one was accepted, silently dropped, and reported only at
// a use -- as a missing "manifest method".
TEST(EmitterTest, RefusesAClassDeclaredInsideAClass) {
  mlir::MLIRContext context(testRegistry());
  lython::emitter::EmitResult result = emitSource(
      "class Outer:\n"
      "    class Inner:\n"
      "        pass\n"
      "\n"
      "\n"
      "print(1)\n",
      context);
  EXPECT_FALSE(result.ok());
  bool named = false;
  for (const lython::parser::Diagnostic &diagnostic : result.diagnostics)
    named = named || diagnostic.message.find(
                         "a class defined inside another class") !=
                         std::string::npos;
  EXPECT_TRUE(named);
}

TEST(EmitterTest, AnAttributeAModuleDoesNotHaveIsRefusedAtEmit) {
  // Nothing resolved these, so they fell through to a dynamic attribute read
  // on the module object -- which no lowering can answer. The message came out
  // one phase later and named the MODULE rather than the attribute:
  // "unresolved runtime binding 'sys'".
  // ⛔ Manifest modules only: this harness emits a module on its own, without
  // the embedded stdlib SOURCE modules `lyc` links, so `os` and `json` do not
  // resolve here at all. The same refusal covers them in the compiler.
  for (const char *source :
       {"import sys\nprint(sys.version_info)\n",
        "import sys\nprint(sys.path)\n",
        "import math\nprint(math.nonexistent)\n"}) {
    mlir::MLIRContext context(testRegistry());
    lython::emitter::EmitResult result = emitSource(source, context);
    EXPECT_FALSE(result.ok()) << source;
    bool named = false;
    for (const lython::parser::Diagnostic &diagnostic : result.diagnostics)
      named = named || (diagnostic.message.find("has no attribute") !=
                            std::string::npos &&
                        diagnostic.message.find("module '") !=
                            std::string::npos);
    EXPECT_TRUE(named) << source;
  }

  // The attributes these modules DO have still resolve, and an attribute on a
  // module's non-module member is not this question: `sys.stderr` is an
  // ordinary object whose `write` the dispatch finds.
  for (const char *source :
       {"import math\nprint(math.sqrt(4.0))\n",
        "import sys\nprint(sys.maxsize > 0)\n"}) {
    mlir::MLIRContext context(testRegistry());
    lython::emitter::EmitResult result = emitSource(source, context);
    EXPECT_TRUE(result.ok()) << source;
  }
}

TEST(EmitterTest, NextWithAnUnrelatedDefaultIsRefusedInItsOwnTerms) {
  // A default that joins with the element at a wider union than an Optional
  // has no slot to be carried out of the desugared try, and the refusal used
  // to be reported on `__lynext1` -- a name the program never wrote, with
  // advice ("bind the reassignment to a new name") that cannot be followed for
  // a scratch the emitter owns.
  mlir::MLIRContext context(testRegistry());
  lython::emitter::EmitResult result =
      emitSource("xs = iter([1])\nprint(next(xs, \"z\"))\n", context);
  EXPECT_FALSE(result.ok());
  bool named = false;
  bool leakedScratch = false;
  for (const lython::parser::Diagnostic &diagnostic : result.diagnostics) {
    named = named || diagnostic.message.find("next(iterator, default)") !=
                         std::string::npos;
    leakedScratch = leakedScratch ||
                    diagnostic.message.find("__lynext") != std::string::npos;
  }
  EXPECT_TRUE(named);
  EXPECT_FALSE(leakedScratch);
}

TEST(EmitterTest, AUnionIsNotDisjointFromItsOwnMember) {
  // `is` against a member of the value's own union is refused, not answered.
  // The disjointness fold decided both `is` and `is not` from the whole type,
  // and a union is assignable to no member of itself in either direction, so
  // `classify(-1) is False` printed False where CPython prints True -- a wrong
  // answer with no diagnostic anywhere.
  for (const char *source :
       {"def classify(n: int):\n    if n < 0:\n        return False\n    "
        "return n * 2\nprint(classify(-1) is False)\n",
        "def classify(n: int):\n    if n < 0:\n        return False\n    "
        "return n * 2\nprint(classify(-1) is not False)\n"}) {
    mlir::MLIRContext context(testRegistry());
    lython::emitter::EmitResult result = emitSource(source, context);
    EXPECT_FALSE(result.ok()) << source;
    bool refused = false;
    for (const lython::parser::Diagnostic &diagnostic : result.diagnostics)
      refused = refused || diagnostic.message.find("reference-typed operands") !=
                               std::string::npos;
    EXPECT_TRUE(refused) << source;
  }

  // Two types that really are disjoint still fold, and a shared member is what
  // separates the two cases: a class instance is never a small int.
  mlir::MLIRContext disjoint(testRegistry());
  lython::emitter::EmitResult folded = emitSource(
      "class P:\n    pass\np = P()\nprint(p is 5)\n", disjoint);
  EXPECT_TRUE(folded.ok()) << folded.diagnostics.size();
}

TEST(EmitterTest, AFunctionValueSaysItHasNoRepr) {
  // Nothing resolves a `__repr__` for a callable, and the fall-through used
  // to try to resolve the NAME `repr` -- so all three spellings read
  // "unresolved name 'repr'", a builtin the program never mentioned.
  for (const char *source :
       {"f = lambda: 3\nprint(str(f))\n", "f = lambda: 3\nprint(repr(f))\n",
        "def g() -> int:\n    return 1\nprint(g)\n"}) {
    mlir::MLIRContext context(testRegistry());
    lython::emitter::EmitResult result = emitSource(source, context);
    EXPECT_FALSE(result.ok()) << source;
    bool saidFunction = false;
    for (const lython::parser::Diagnostic &diagnostic : result.diagnostics)
      saidFunction = saidFunction ||
                     diagnostic.message.find("a function value has no repr()") !=
                         std::string::npos;
    EXPECT_TRUE(saidFunction) << source;
  }

  // The ordinary receivers still render.
  mlir::MLIRContext values(testRegistry());
  lython::emitter::EmitResult ok =
      emitSource("print(repr(1), repr(\"a\"), repr([1]))\n", values);
  EXPECT_TRUE(ok.ok());
}

TEST(EmitterTest, ANestedClassNamesItsOwnLimit) {
  // The generic "unsupported statement kind 'ClassDef'" plus a misleading
  // "unresolved name" at every use read as a scoping bug rather than as the
  // one limitation it is. In a function the class compiles; in a block of the
  // module's top level it is still refused, by name.
  mlir::MLIRContext nested(testRegistry());
  lython::emitter::EmitResult inFunction = emitSource(
      "def run() -> None:\n    class D:\n        def __init__(self) -> None:\n"
      "            self.v = 1\n    print(D().v)\nrun()\n",
      nested);
  EXPECT_TRUE(inFunction.ok());
  mlir::MLIRContext block(testRegistry());
  lython::emitter::EmitResult inBlock = emitSource(
      "import sys\nif sys.platform:\n    class D:\n        pass\n", block);
  EXPECT_TRUE(reportsDiagnostic(
      inBlock, "a class defined in a block of the module's top level"));

  // At module scope the same class compiles.
  mlir::MLIRContext top(testRegistry());
  lython::emitter::EmitResult atModule = emitSource(
      "class D:\n    def __init__(self) -> None:\n        self.v = 1\n"
      "def run() -> None:\n    print(D().v)\nrun()\n",
      top);
  EXPECT_TRUE(atModule.ok());
}
}

// What: a class written in a function is one class, declared at the module's
// top level under its qualified name. Each program here has a class statement
// whose meaning one class cannot carry -- a use the statement may not have run
// before, a second binding of the name, a read of the function's locals, or
// something the statement evaluates that a later call could see differ -- and
// each is refused at the class, naming why.
TEST(EmitterTest, AClassInAFunctionIsRefusedWhereOneClassCannotBeIt) {
  struct Case {
    const char *source;
    const char *says;
  };
  const Case cases[] = {
      {"def f() -> None:\n    a = A()\n    class A:\n        pass\n",
       "is used where its class statement may not have run"},
      {"def f(flag: bool) -> None:\n    if flag:\n        class A:\n"
       "            pass\n    print(A())\n",
       "is used where its class statement may not have run"},
      {"def f() -> int:\n    def g() -> int:\n        return A().v()\n"
       "    class A:\n        def v(self) -> int:\n            return 1\n"
       "    return g()\n",
       "is used where its class statement may not have run"},
      {"def f() -> int:\n    class A:\n        pass\n    A = 3\n    return A\n",
       "is bound again in the function that defines the class"},
      {"def f() -> None:\n    class A:\n        pass\n    def g() -> None:\n"
       "        nonlocal A\n        A = int\n",
       "by a nested def's nonlocal"},
      {"def f(flag: bool) -> None:\n    if flag:\n        class A:\n"
       "            pass\n    else:\n        class A:\n            pass\n",
       "two classes named 'A' in one function"},
      {"def f() -> None:\n    global A\n    class A:\n        pass\n",
       "is declared global or nonlocal"},
      {"def f(n: int) -> int:\n    class A:\n        def get(self) -> int:\n"
       "            return n\n    return A().get()\n",
       "reads 'n', a local of the function that defines it"},
      {"def f() -> None:\n    class A:\n        items: list[int] = []\n",
       "the value of 'items' makes a new list for every call"},
      {"def g() -> int:\n    return 1\ndef f() -> None:\n    class A:\n"
       "        x = g()\n",
       "the value of 'x' calls 'g', which runs again for every call"},
      {"def f() -> None:\n    class A:\n        print(1)\n",
       "its body runs a statement"},
      {"from typing import Callable\n"
       "def deco(fn: Callable[[object], int]) -> Callable[[object], int]:\n"
       "    return fn\n"
       "def f() -> None:\n    class A:\n        @deco\n"
       "        def m(self) -> int:\n            return 1\n",
       "'m' applies 'deco', which runs again for every call"},
      {"def f() -> None:\n    class A:\n"
       "        def m(self, acc: list[int] = []) -> int:\n"
       "            return len(acc)\n",
       "a default of 'm' makes a new list for every call"},
      {"def f() -> int:\n    class A:\n        n = 0\n        @classmethod\n"
       "        def bump(cls) -> None:\n            cls.n += 1\n"
       "    A.bump()\n    return A.n\n",
       "class attribute 'n' of 'f.<locals>.A' is assigned outside its class "
       "body"},
  };
  for (const Case &each : cases) {
    mlir::MLIRContext context(testRegistry());
    lython::emitter::EmitResult emitted = emitSource(each.source, context);
    EXPECT_FALSE(emitted.ok()) << each.source;
    EXPECT_TRUE(reportsDiagnostic(emitted, each.says))
        << each.source << "\nexpected: " << each.says;
  }
}

// What: what a class statement in a function evaluates may be what every
// call would evaluate alike -- literals, a tuple of them, enum numbering, a
// dataclass field's spec, the descriptors a def is wrapped in, a name the
// module binds once -- and such a class compiles.
TEST(EmitterTest, AClassInAFunctionEvaluatingOnlyConstantsCompiles) {
  mlir::MLIRContext context(testRegistry());
  lython::emitter::EmitResult emitted = emitSource(
      "from dataclasses import dataclass, field\n"
      "from enum import Enum, auto\n"
      "LIMIT = 10\n"
      "def f() -> None:\n"
      "    class Color(Enum):\n"
      "        RED = auto()\n"
      "    @dataclass\n"
      "    class P:\n"
      "        x: int = 0\n"
      "        ys: list[int] = field(default_factory=list)\n"
      "    class A:\n"
      "        \"\"\"Doc.\"\"\"\n"
      "        name = \"a\"\n"
      "        pair = (1, -2)\n"
      "        limit = LIMIT\n"
      "        def __init__(self, v: int = 3) -> None:\n"
      "            self._v = v\n"
      "        @property\n"
      "        def v(self) -> int:\n"
      "            return self._v\n"
      "        @v.setter\n"
      "        def v(self, value: int) -> None:\n"
      "            self._v = value\n"
      "        @staticmethod\n"
      "        def twice(x: int) -> int:\n"
      "            return x * 2\n"
      "    print(Color.RED, P(), A().v, A.twice(A.limit), A.name, A.pair)\n"
      "f()\n",
      context);
  EXPECT_TRUE(emitted.ok());
  for (const lython::parser::Diagnostic &diagnostic : emitted.diagnostics)
    ADD_FAILURE() << diagnostic.message;
}

TEST(EmitterTest, AModuleGlobalContainerNamesItsElementRepresentation) {
  // A module global's cell reaches its elements: `xs: list[int] = [True]`
  // printed 0 and `ys: list[float] = [1]` printed 5e-324, because the store
  // went through at the value's own element type and the read came back at the
  // declaration's. Only the scalar shape was reported.
  const char *shapes[] = {
      "xs: list[int] = [True]\nprint(xs[0])\n",
      "ys: list[float] = [1]\nprint(ys[0])\n",
      "t: tuple[int, int] = (True, 2)\nprint(t[0])\n",
      "d: dict[str, int] = {\"a\": True}\nprint(d[\"a\"])\n",
      // The class attribute is the same cell with the same rule, and its
      // container spelling printed the reinterpretation too.
      "class P:\n    v: list[float] = [1]\nprint(P.v[0])\n",
      "class Q:\n    v: dict[str, int] = {\"a\": True}\nprint(Q.v[\"a\"])\n",
  };
  for (const char *shape : shapes) {
    mlir::MLIRContext context(testRegistry());
    lython::emitter::EmitResult result = emitSource(shape, context);
    EXPECT_FALSE(result.ok()) << shape;
    bool named = false;
    for (const lython::parser::Diagnostic &diagnostic : result.diagnostics)
      named = named || diagnostic.message.find("a container of") !=
                           std::string::npos;
    EXPECT_TRUE(named) << shape;
  }

  // Inside a function the same four compile: a local carries the type its
  // value has, so there is no cell to disagree with.
  mlir::MLIRContext local(testRegistry());
  lython::emitter::EmitResult inFunction = emitSource(
      "def run() -> None:\n    xs: list[int] = [True]\n"
      "    ys: list[float] = [1]\n    t: tuple[int, int] = (True, 2)\n"
      "    d: dict[str, int] = {\"a\": True}\n"
      "    print(xs[0], ys[0], t[0], d[\"a\"])\nrun()\n",
      local);
  EXPECT_TRUE(inFunction.ok());
}

TEST(EmitterTest, ACellsElementStorageNamesItsRepresentation) {
  // A cell's element type is its storage one level in, so a CONTAINER of
  // bools stored as an element of a `list[list[int]]` cell is refused, and the
  // refusal names the element assignment.
  const char *cells[] = {
      "xs: list[list[int]] = [[1]]\nxs.append([True])\nprint(xs)\n",
      "xs: list[list[int]] = [[1]]\nxs[0] = [True]\nprint(xs)\n",
      // Written from inside a function, where the name still resolves to the
      // module global's cell.
      ("xs: list[list[int]] = [[1]]\ndef go() -> None:\n    xs[0] = [True]\n"
       "go()\nprint(xs)\n"),
  };
  for (const char *cell : cells) {
    mlir::MLIRContext context(testRegistry());
    lython::emitter::EmitResult result = emitSource(cell, context);
    EXPECT_FALSE(result.ok()) << cell;
    bool named = false;
    for (const lython::parser::Diagnostic &diagnostic : result.diagnostics)
      named = named || diagnostic.message.find("element assignment") !=
                           std::string::npos;
    EXPECT_TRUE(named) << cell;
  }

  // A LOCAL is not a cell: the emitter refines the name's element type at the
  // store, so these are correct and must stay compiling.
  const char *locals[] = {
      "def go() -> None:\n    xs: list[int] = [1, 2]\n    xs[0] = True\n"
      "    print(xs[0])\ngo()\n",
      "def go() -> None:\n    xs: list[int] = [1, 2]\n    xs.append(True)\n"
      "    print(xs[2])\ngo()\n",
      "def go() -> None:\n    xs: list[int] = []\n    xs.append(True)\n"
      "    print(xs[0])\ngo()\n",
      // And a same-rung store into a cell is not this question at all.
      "xs: list[int] = [1, 2]\nxs[0] = 5\nprint(xs[0])\n",
      // A bool written into an int cell's element is kept as its own object,
      // which the int element reads back as the bool
      // (tests/python/test_bool_in_module_storage.py runs these).
      "xs: list[int] = [1, 2]\nxs[0] = True\nprint(xs[0])\n",
      "xs: list[int] = [1, 2]\nxs.append(True)\nprint(xs[2])\n",
      "d: dict[str, int] = {\"a\": 1}\nd[\"b\"] = True\nprint(d[\"b\"])\n",
      "class P:\n    v: list[int] = [1, 2]\nP.v[0] = True\nprint(P.v[0])\n",
      ("xs: list[int] = [1, 2]\ndef go() -> None:\n    xs[0] = True\n"
       "go()\nprint(xs[0])\n"),
      // ⛔ A dict KEY of a lower rung is correct and must stay correct:
      // CPython unifies `db[True]` with `db[1]` by hash and equality, and so
      // does this compiler (golden dict_generic_keys).
      "db: dict[int, str] = {}\ndb[True] = \"t\"\ndb[1] = \"one\"\n"
      "print(db, len(db))\n",
  };
  for (const char *local : locals) {
    mlir::MLIRContext context(testRegistry());
    EXPECT_TRUE(emitSource(local, context).ok()) << local;
  }
}

TEST(EmitterTest, AMethodThatExistsIsNotReportedAsMissing) {
  // `list[float]` provides `append`; what it does not have is an overload
  // taking an int. Both cases said "does not provide manifest method", and a
  // reader given that sentence looks for a missing method that is not there.
  mlir::MLIRContext wrongArguments(testRegistry());
  lython::emitter::EmitResult mismatched = emitSource(
      "xs: list[float] = [1.0]\nxs.append(1)\nprint(xs)\n", wrongArguments);
  EXPECT_FALSE(mismatched.ok());
  bool namedTheArguments = false;
  for (const lython::parser::Diagnostic &diagnostic : mismatched.diagnostics)
    namedTheArguments =
        namedTheArguments ||
        (diagnostic.message.find("has manifest method 'append'") !=
             std::string::npos &&
         diagnostic.message.find("no signature that accepts") !=
             std::string::npos);
  EXPECT_TRUE(namedTheArguments);

  // A method the receiver really does not have keeps the other sentence.
  mlir::MLIRContext noSuchMethod(testRegistry());
  lython::emitter::EmitResult missing =
      emitSource("xs: list[int] = [1]\nprint(xs.no_such_method())\n",
                 noSuchMethod);
  EXPECT_FALSE(missing.ok());
  bool namedTheMethod = false;
  for (const lython::parser::Diagnostic &diagnostic : missing.diagnostics)
    namedTheMethod = namedTheMethod ||
                     diagnostic.message.find("does not provide manifest "
                                             "method 'no_such_method'") !=
                         std::string::npos;
  EXPECT_TRUE(namedTheMethod);
}

TEST(EmitterTest, AnImportedModulesOwnBindingIsNotAnUnresolvedName) {
  // A container constant in an imported module IS storage now: it gets the
  // module global cell the main module's own container constant gets, filled
  // at the start of `__main__`. Read from a function in its own module, which
  // is the half that used to say "unresolved name 'ITEMS'" at a line where
  // the name is plainly in scope.
  ImportedModuleEmit container = emitWithImportedModule(
      "tabled",
      "ITEMS: \"list[int]\" = [1, 2, 3]\n\n\ndef total() -> int:\n"
      "    return len(ITEMS)\n",
      "import tabled\nprint(tabled.total())\n");
  EXPECT_TRUE(container.succeeded) << container.diagnostics;

  // ⛔ And only when the element type is RESOLVED: an erased `[]` has no
  // storage the readers can agree on, and that refusal names the assignment
  // rather than sending the reader looking for a typo.
  ImportedModuleEmit erased = emitWithImportedModule(
      "erased", "ITEMS = []\n\n\ndef total() -> int:\n    return len(ITEMS)\n",
      "import erased\nprint(erased.total())\n");
  EXPECT_FALSE(erased.succeeded);
  EXPECT_NE(erased.diagnostics.find("is assigned at the top level of this "
                                    "imported module"),
            std::string::npos)
      << erased.diagnostics;

  // A name the module really does not bind keeps the other sentence.
  ImportedModuleEmit typo = emitWithImportedModule(
      "typoed", "def go() -> int:\n    return NOPE\n",
      "import typoed\nprint(typoed.go())\n");
  EXPECT_FALSE(typo.succeeded);
  EXPECT_NE(typo.diagnostics.find("unresolved name 'NOPE'"), std::string::npos)
      << typo.diagnostics;

  // And a SCALAR constant read the same way still works, which is what makes
  // the container case a limitation of the channel rather than of the scope.
  ImportedModuleEmit scalar = emitWithImportedModule(
      "counted", "COUNT: int = 5\n\n\ndef get() -> int:\n    return COUNT\n",
      "import counted\nprint(counted.get())\n");
  EXPECT_TRUE(scalar.succeeded) << scalar.diagnostics;
}

// `Literal[a, b]` is the JOIN of its members. The tuple slice fell to a
// fallback that built `!py.literal<object>` -- a literal whose spelling is the
// word "object" -- and every call and every operation on the annotated name was
// then refused with a sentence about the Callable contract. Emit-level: once
// the annotation names a type the program can use, nothing about the running
// values is new.
TEST(EmitterTest, AMultiValueLiteralAnnotationIsTheJoinOfItsMembers) {
  auto diagnosticText = [](const lython::emitter::EmitResult &emitted) {
    std::string text;
    for (const lython::parser::Diagnostic &diagnostic : emitted.diagnostics)
      text += diagnostic.message + "\n";
    return text;
  };
  mlir::MLIRContext strings(testRegistry());
  lython::emitter::EmitResult twoStrings = emitSource(
      "from typing import Literal\n\n\n"
      "def go(mode: Literal[\"a\", \"b\"]) -> int:\n"
      "    return 1 if mode == \"a\" else 2\n\n\n"
      "print(go(\"a\"), go(\"b\"))\n",
      strings);
  EXPECT_TRUE(twoStrings.ok()) << diagnosticText(twoStrings);

  // The members are widened before the join, so an int Literal supports int
  // arithmetic rather than answering as a union of two literals.
  mlir::MLIRContext ints(testRegistry());
  lython::emitter::EmitResult twoInts = emitSource(
      "from typing import Literal\n\n\n"
      "def go(n: Literal[1, 2]) -> int:\n"
      "    return n * 10\n\n\n"
      "print(go(1), go(2))\n",
      ints);
  EXPECT_TRUE(twoInts.ok()) << diagnosticText(twoInts);

  // A one-member Literal keeps its exact literal type, which is what made the
  // tuple slice the gap rather than the annotation.
  mlir::MLIRContext single(testRegistry());
  lython::emitter::EmitResult one = emitSource(
      "from typing import Literal\n\n\n"
      "def go(mode: Literal[\"a\"]) -> str:\n"
      "    return mode\n\n\n"
      "print(go(\"a\"))\n",
      single);
  EXPECT_TRUE(one.ok()) << diagnosticText(one);
}

// A return whose type does not match the annotation is refused HERE, where both
// types have names. It used to reach the lowering and surface as "callable
// return ABI expected 2 physical values, but lowering produced 1" -- and inside
// a `try`, where the return leaves through a join, as "cannot adapt runtime
// bundle builtins.int ... to expected ABI".
TEST(EmitterTest, AReturnThatDoesNotMatchItsAnnotationIsNamed) {
  auto refusedNaming = [](lython::emitter::EmitResult &emitted) {
    for (const lython::parser::Diagnostic &diagnostic : emitted.diagnostics)
      if (diagnostic.message.find("annotated to return") != std::string::npos &&
          diagnostic.message.find("but this return gives") != std::string::npos)
        return true;
    return false;
  };

  mlir::MLIRContext plain(testRegistry());
  lython::emitter::EmitResult direct = emitSource(
      "def run() -> str:\n    return 1\n\n\nprint(run())\n", plain);
  EXPECT_FALSE(direct.ok());
  EXPECT_TRUE(refusedNaming(direct));

  // The same return inside a `try` leaves through a join rather than a
  // `func.return`, which is why the check sits before the paths diverge.
  mlir::MLIRContext guarded(testRegistry());
  lython::emitter::EmitResult inTry = emitSource(
      "def run() -> str:\n"
      "    try:\n"
      "        return 1\n"
      "    except ValueError:\n"
      "        return \"caught\"\n\n\n"
      "print(run())\n",
      guarded);
  EXPECT_FALSE(inTry.ok());
  EXPECT_TRUE(refusedNaming(inTry));

  // The three allowances the declared-PARAMETER check makes are kept: the
  // numeric tower, a subclass reaching a base, and a bare generic contract.
  mlir::MLIRContext admitted(testRegistry());
  lython::emitter::EmitResult tower = emitSource(
      "def run(x: int) -> float:\n    return x\n\n\nprint(run(3))\n", admitted);
  EXPECT_TRUE(tower.ok());

  mlir::MLIRContext subclass(testRegistry());
  lython::emitter::EmitResult base = emitSource(
      "class B:\n"
      "    def __init__(self) -> None:\n        self.n: int = 1\n"
      "class D(B):\n    pass\n"
      "def make() -> B:\n    return D()\n\n\nprint(make().n)\n",
      subclass);
  EXPECT_TRUE(base.ok()) << base.diagnostics.size();
}

// What: a `@classmethod` an override sits behind DISPATCHES when it is called,
// and its value spelling (`m = a.tag`) is what is left. The call was refused
// until the candidate set became every SUBCLASS rather than the redeclaring
// slice of it -- `cls` binds the runtime class, so a subclass that redeclares
// nothing still has its own answer. A staticmethod beside it dispatches
// through both spellings, which is what says the residue is about carrying
// `cls` into a value and not about the receiver being an instance.
TEST(EmitterTest, AClassmethodBehindAnOverrideDispatchesWhenItIsCalled) {
  auto refusedNaming = [](lython::emitter::EmitResult &emitted,
                          const char *member) {
    for (const lython::parser::Diagnostic &diagnostic : emitted.diagnostics)
      if (diagnostic.message.find(std::string("'") + member +
                                  "' is overridden by a subclass") !=
          std::string::npos)
        return true;
    return false;
  };
  const char *hierarchy =
      "class Base:\n"
      "    @classmethod\n"
      "    def tag(cls) -> str:\n        return \"base\"\n"
      "class Sub(Base):\n"
      "    @classmethod\n"
      "    def tag(cls) -> str:\n        return \"sub\"\n"
      "a: Base = Sub()\n";

  mlir::MLIRContext called(testRegistry());
  lython::emitter::EmitResult direct =
      emitSource(std::string(hierarchy) + "print(a.tag())\n", called);
  EXPECT_TRUE(direct.ok()) << direct.diagnostics.size();

  mlir::MLIRContext read(testRegistry());
  lython::emitter::EmitResult asValue =
      emitSource(std::string(hierarchy) + "m = a.tag\nprint(m())\n", read);
  EXPECT_FALSE(asValue.ok());
  EXPECT_TRUE(refusedNaming(asValue, "tag"));

  mlir::MLIRContext staticly(testRegistry());
  lython::emitter::EmitResult dispatched = emitSource(
      "class Base:\n"
      "    @staticmethod\n"
      "    def tag() -> str:\n        return \"base\"\n"
      "class Sub(Base):\n"
      "    @staticmethod\n"
      "    def tag() -> str:\n        return \"sub\"\n"
      "a: Base = Sub()\n"
      "m = a.tag\nprint(a.tag(), m())\n",
      staticly);
  EXPECT_TRUE(dispatched.ok()) << dispatched.diagnostics.size();
}

// What: the two class-header keyword shapes that have no answer are refused
// rather than dropped. Both compiled and ran before: `metaclass=` named a type
// that never ran, and a keyword no `__init_subclass__` declares is a TypeError
// in CPython and printed nothing here.
TEST(EmitterTest, AClassHeaderKeywordWithNoHookToTakeItIsRefused) {
  auto refusedSaying = [](lython::emitter::EmitResult &emitted,
                          const char *fragment) {
    for (const lython::parser::Diagnostic &diagnostic : emitted.diagnostics)
      if (diagnostic.message.find(fragment) != std::string::npos)
        return true;
    return false;
  };

  mlir::MLIRContext meta(testRegistry());
  lython::emitter::EmitResult withMetaclass =
      emitSource("class Meta(type):\n    pass\n"
                 "class Base(metaclass=Meta):\n    pass\n"
                 "print(\"ok\")\n",
                 meta);
  EXPECT_FALSE(withMetaclass.ok());
  EXPECT_TRUE(refusedSaying(withMetaclass, "'metaclass=' is not supported"));

  mlir::MLIRContext undeclared(testRegistry());
  lython::emitter::EmitResult noHook =
      emitSource("class Base:\n    pass\n"
                 "class Sub(Base, tag=\"s\"):\n    pass\n"
                 "print(\"ok\")\n",
                 undeclared);
  EXPECT_FALSE(noHook.ok());
  EXPECT_TRUE(refusedSaying(noHook, "does not declare"));

  // A parameter the hook requires and the header does not pass is the other
  // half of CPython's TypeError, and it is the one a default makes legal --
  // so the same hierarchy with a default compiles.
  mlir::MLIRContext missing(testRegistry());
  lython::emitter::EmitResult short_ =
      emitSource("class Base:\n"
                 "    @classmethod\n"
                 "    def __init_subclass__(cls, tag: str) -> None:\n"
                 "        print(tag)\n"
                 "class Sub(Base):\n    pass\n",
                 missing);
  EXPECT_FALSE(short_.ok());
  EXPECT_TRUE(refusedSaying(short_, "the header does not pass"));

  mlir::MLIRContext defaulted(testRegistry());
  lython::emitter::EmitResult ok =
      emitSource("class Base:\n"
                 "    @classmethod\n"
                 "    def __init_subclass__(cls, tag: str = \"d\") -> None:\n"
                 "        print(tag)\n"
                 "class Sub(Base):\n    pass\n",
                 defaulted);
  EXPECT_TRUE(ok.ok()) << ok.diagnostics.size();
}

// What: redeclaring a `@property` in a subclass replaces the whole descriptor,
// so the base's setter is not reachable through it -- CPython raises
// AttributeError and this inlined the base's setter, storing where Python
// stores nothing. The same hierarchy WITH a setter in the subclass compiles,
// which is what says the refusal is about the missing setter and not about the
// redeclaration.
TEST(EmitterTest, APropertyRedeclaredWithoutASetterRefusesTheWrite) {
  auto refusedSaying = [](lython::emitter::EmitResult &emitted) {
    for (const lython::parser::Diagnostic &diagnostic : emitted.diagnostics)
      if (diagnostic.message.find("has no setter") != std::string::npos)
        return true;
    return false;
  };
  const char *base =
      "class Base:\n"
      "    def __init__(self) -> None:\n        self._v = 0\n"
      "    @property\n"
      "    def v(self) -> int:\n        return self._v\n"
      "    @v.setter\n"
      "    def v(self, n: int) -> None:\n        self._v = n\n";

  mlir::MLIRContext replaced(testRegistry());
  lython::emitter::EmitResult noSetter =
      emitSource(std::string(base) +
                     "class Sub(Base):\n"
                     "    @property\n"
                     "    def v(self) -> int:\n        return self._v * 10\n"
                     "s = Sub()\ns.v = 5\nprint(s.v)\n",
                 replaced);
  EXPECT_FALSE(noSetter.ok());
  EXPECT_TRUE(refusedSaying(noSetter));

  mlir::MLIRContext kept(testRegistry());
  lython::emitter::EmitResult withSetter =
      emitSource(std::string(base) +
                     "class Sub(Base):\n"
                     "    @property\n"
                     "    def v(self) -> int:\n        return self._v * 10\n"
                     "    @v.setter\n"
                     "    def v(self, n: int) -> None:\n        self._v = n + 1\n"
                     "s = Sub()\ns.v = 5\nprint(s.v)\n",
                 kept);
  EXPECT_TRUE(withSetter.ok()) << withSetter.diagnostics.size();
}

// What: a generator whose ANNOTATION is genuinely wrong is refused at emit,
// naming the yield that does not match it: `-> Iterator[str]` with `yield v`
// for an int `v` under a guard.
TEST(EmitterTest, AGeneratorAnnotationThatIsWrongIsStillNamed) {
  auto refusedNaming = [](lython::emitter::EmitResult &emitted) {
    for (const lython::parser::Diagnostic &diagnostic : emitted.diagnostics)
      if (diagnostic.message.find("yield of !py.contract<\"builtins.int\"> "
                                  "in a generator annotated to yield") !=
          std::string::npos)
        return true;
    return false;
  };

  mlir::MLIRContext wrong(testRegistry());
  lython::emitter::EmitResult mismatched = emitSource(
      "from typing import Iterator\n"
      "def bad(xs: \"list[int | None]\") -> Iterator[str]:\n"
      "    for v in xs:\n"
      "        if v is not None:\n"
      "            yield v\n"
      "print(list(bad([1, None])))\n",
      wrong);
  EXPECT_FALSE(mismatched.ok());
  EXPECT_TRUE(refusedNaming(mismatched));

  // The same shape with the annotation the guard proves compiles, which is
  // what says the refusal is about the mismatch and not about the guard.
  mlir::MLIRContext right(testRegistry());
  lython::emitter::EmitResult narrowed = emitSource(
      "from typing import Iterator\n"
      "def good(xs: \"list[int | None]\") -> Iterator[int]:\n"
      "    for v in xs:\n"
      "        if v is not None:\n"
      "            yield v + 1\n"
      "print(list(good([1, None])))\n",
      right);
  EXPECT_TRUE(narrowed.ok()) << narrowed.diagnostics.size();
}

// A field read returned where a MEMBER of its union is declared. The guard
// that proved the member does not survive a call that assigns the field, so
// the return really does give the union -- and it reached the LOWERING as
// "cannot adapt runtime bundle with physical values (i64, memref, memref) to
// expected ABI (memref, memref)", a sentence about lanes for a return the
// author can see is wrong.
TEST(EmitterTest, AUnionFieldReturnedWhereAMemberIsDeclaredIsNamed) {
  auto namesTheReturn = [](lython::emitter::EmitResult &emitted) {
    for (const lython::parser::Diagnostic &diagnostic : emitted.diagnostics)
      if (diagnostic.message.find("is annotated to return") !=
          std::string::npos)
        return true;
    return false;
  };

  mlir::MLIRContext wrong(testRegistry());
  lython::emitter::EmitResult mismatched = emitSource(
      "class Box:\n"
      "    def __init__(self) -> None:\n"
      "        self.v: \"str | None\" = None\n"
      "    def clear(self) -> None:\n"
      "        self.v = None\n"
      "    def use(self) -> str:\n"
      "        if self.v is not None:\n"
      "            self.clear()\n"
      "            return self.v\n"
      "        return \"-\"\n"
      "print(Box().use())\n",
      wrong);
  EXPECT_FALSE(mismatched.ok());
  EXPECT_TRUE(namesTheReturn(mismatched));

  // The same read WITHOUT the call keeps the guard's proof and compiles, which
  // is what says the refusal is about the union and not about the field.
  mlir::MLIRContext right(testRegistry());
  lython::emitter::EmitResult proved = emitSource(
      "class Box:\n"
      "    def __init__(self) -> None:\n"
      "        self.v: \"str | None\" = None\n"
      "    def use(self) -> str:\n"
      "        if self.v is not None:\n"
      "            return self.v\n"
      "        return \"-\"\n"
      "print(Box().use())\n",
      right);
  EXPECT_TRUE(proved.ok()) << proved.diagnostics.size();
}

// What: a method call with NO arguments on a name bound to an empty container
// literal is still a call on that name, so it reaches the scan that decides
// the container's element type from what fills it. `xs = []` followed by
// `xs.pop()` -- and `clear`, `copy`, `sort`, `reverse`, `keys`, `values`,
// `items`, `popitem`, and every arity mistake -- crashed the compiler with no
// diagnostic at all: 54 of the 72 spellings measured, on `[]`, `{}` and
// `set()` alike, and only the ANNOTATED form escaped because an annotation
// means the scan never runs. The seed the scan is for has to survive beside
// them.
TEST(EmitterTest, AZeroArgumentMethodOnAnEmptyLiteralIsNotACrash) {
  auto emits = [](const char *source) {
    mlir::MLIRContext context(testRegistry());
    lython::emitter::EmitResult emitted = emitSource(source, context);
    return emitted.ok();
  };

  EXPECT_TRUE(emits("def f() -> int:\n"
                    "    xs = []\n"
                    "    xs.clear()\n"
                    "    return len(xs)\n"
                    "print(f())\n"));
  EXPECT_TRUE(emits("def f() -> int:\n"
                    "    d = {}\n"
                    "    d.clear()\n"
                    "    return len(d)\n"
                    "print(f())\n"));
  EXPECT_TRUE(emits("def f() -> int:\n"
                    "    s = set()\n"
                    "    s.copy()\n"
                    "    return len(s)\n"
                    "print(f())\n"));
  EXPECT_TRUE(emits("xs = []\n"
                    "xs.sort()\n"
                    "print(len(xs))\n"));

  // A call written with too few arguments reaches the same place, and the
  // program is refused for the argument count rather than by dying.
  EXPECT_FALSE(emits("def f() -> int:\n"
                     "    xs = []\n"
                     "    xs.append()\n"
                     "    return len(xs)\n"
                     "print(f())\n"));

  // And the seed still arrives past the zero-argument call: without it the
  // element stays erased and `\"/\".join(parts)` has nothing to join.
  EXPECT_TRUE(emits("def f(path: str) -> str:\n"
                    "    parts = []\n"
                    "    for seg in path.split(\"/\"):\n"
                    "        if seg == \"..\":\n"
                    "            parts.pop()\n"
                    "            continue\n"
                    "        parts.append(seg)\n"
                    "    return \"/\".join(parts)\n"
                    "print(f(\"a/b/../c\"))\n"));
}

// What: a stub's classes answer member questions -- a field, a method found
// through a base, an overload chosen by a literal argument -- and a literal no
// overload names is refused.
TEST(EmitterTest, AStubsClassesAreContracts) {
  const char *stub = "import typing as t\n"
                     "class Shape:\n"
                     "    sides: int\n"
                     "    def area(self, scale: float) -> float: ...\n"
                     "    @t.overload\n"
                     "    def pick(self, k: t.Literal[\"a\"]) -> int: ...\n"
                     "    @t.overload\n"
                     "    def pick(self, k: t.Literal[\"b\"]) -> str: ...\n"
                     "class Square(Shape):\n"
                     "    side: float\n";
  ImportedModuleEmit typed =
      emitWithImportedModule("shape", stub,
                             "import shape\n"
                             "def use(s: shape.Square) -> None:\n"
                             "    x: float = s.area(2.0) + s.side\n"
                             "    n: int = s.sides + s.pick(\"a\")\n"
                             "    b: str = s.pick(\"b\")\n",
                             ".pyi");
  EXPECT_TRUE(typed.succeeded) << typed.diagnostics;

  ImportedModuleEmit unnamed =
      emitWithImportedModule("shape", stub,
                             "import shape\n"
                             "def use(s: shape.Square) -> None:\n"
                             "    s.pick(\"c\")\n",
                             ".pyi");
  EXPECT_FALSE(unnamed.succeeded);
  EXPECT_NE(unnamed.diagnostics.find("has manifest method 'pick' but no "
                                     "signature that accepts"),
            std::string::npos)
      << unnamed.diagnostics;
}

// What: a stub's overloads are tried in declaration order and the first that
// accepts the call answers (PEP 484), where two that both accept would
// otherwise tie.
TEST(EmitterTest, AStubsFirstApplicableOverloadAnswers) {
  const char *stub = "import typing as t\n"
                     "class Box:\n"
                     "    @t.overload\n"
                     "    def get(self, k: object) -> int: ...\n"
                     "    @t.overload\n"
                     "    def get(self, k: object) -> str: ...\n";
  ImportedModuleEmit first =
      emitWithImportedModule("box", stub,
                             "import box\ndef use(b: box.Box) -> None:\n"
                             "    print(b.get(1).bit_length())\n",
                             ".pyi");
  EXPECT_TRUE(first.succeeded) << first.diagnostics;
  ImportedModuleEmit second =
      emitWithImportedModule("box", stub,
                             "import box\ndef use(b: box.Box) -> None:\n"
                             "    print(b.get(1).upper())\n",
                             ".pyi");
  EXPECT_FALSE(second.succeeded);
}

// What: `js` is the JavaScript host's module: refused where the target has no
// host, and on one that does, a local js.py may not take its name.
TEST(EmitterTest, TheJsModuleIsTheHosts) {
  mlir::MLIRContext context(testRegistry());
  mlir::OwningOpRef<mlir::ModuleOp> module;
  std::string diagnostics;
  llvm::raw_string_ostream diag(diagnostics);
  lython::driver::DriverOptions native;
  native.targetTriple = llvm::sys::getDefaultTargetTriple();
  EXPECT_TRUE(mlir::failed(lython::driver::emitMLIRFromSource(
      "from js import console\nconsole.log(1)\n", "main.py",
      "<lython-no-import-dir>", native, context, module, diag)));
  EXPECT_NE(diagnostics.find("module 'js' is the JavaScript host's and this "
                             "target has none"),
            std::string::npos)
      << diagnostics;

  ImportedModuleEmit shadowed = emitWithImportedModule(
      "js", "def log() -> None:\n    pass\n", "import js\njs.log()\n", ".py",
      "wasm32-wasip1", /*jsHost=*/true);
  EXPECT_FALSE(shadowed.succeeded);
  EXPECT_NE(shadowed.diagnostics.find("module 'js' is the host's; rename this "
                                      "file"),
            std::string::npos)
      << shadowed.diagnostics;
}

// What: a name imported from `js` is the host's global typed by the stub --
// `Math` the object, not the class of that name -- so its members are typed:
// a float where the stub says float, and a str refused where an int is
// declared.
TEST(EmitterTest, AJsGlobalIsTypedByTheStub) {
  auto emitFor = [](llvm::StringRef source, std::string &diagnostics) {
    mlir::MLIRContext context(testRegistry());
    mlir::OwningOpRef<mlir::ModuleOp> module;
    llvm::raw_string_ostream diag(diagnostics);
    lython::driver::DriverOptions options;
    options.targetTriple = "wasm32-wasip1";
    options.jsHost = true;
    return mlir::succeeded(lython::driver::emitMLIRFromSource(
        source, "main.py", "<lython-no-import-dir>", options, context, module,
        diag));
  };
  std::string typed;
  EXPECT_TRUE(emitFor("from js import Math, URLSearchParams\n"
                      "x: float = Math.sqrt(2.0) + Math.PI\n"
                      "v = URLSearchParams.new(\"a=1\").get(\"a\")\n"
                      "if v is not None:\n"
                      "    print(v + \"!\")\n",
                      typed))
      << typed;
  std::string refused;
  EXPECT_FALSE(emitFor("from js import JSON, URLSearchParams\n"
                       "print(JSON.stringify(1).bit_length())\n"
                       "v = URLSearchParams.new(\"a=1\").get(\"a\")\n"
                       "print(v + \"!\")\n",
                       refused));
  EXPECT_NE(refused.find("'bit_length'"), std::string::npos) << refused;
  EXPECT_NE(refused.find("__add__"), std::string::npos) << refused;
}

// What: an `Any` result is a bare JavaScript value with no members until an
// isinstance against a host class narrows it; isinstance of a union against
// one is refused, since the host can only test a JavaScript value.
TEST(EmitterTest, AnIsinstanceAgainstAHostClassNarrows) {
  auto emitFor = [](llvm::StringRef source, std::string &diagnostics) {
    mlir::MLIRContext context(testRegistry());
    mlir::OwningOpRef<mlir::ModuleOp> module;
    llvm::raw_string_ostream diag(diagnostics);
    lython::driver::DriverOptions options;
    options.targetTriple = "wasm32-wasip1";
    options.jsHost = true;
    return mlir::succeeded(lython::driver::emitMLIRFromSource(
        source, "main.py", "<lython-no-import-dir>", options, context, module,
        diag));
  };
  std::string narrowed;
  EXPECT_TRUE(emitFor("from js import JSON, URLSearchParams\n"
                      "v = JSON.parse(\"{}\")\n"
                      "if isinstance(v, URLSearchParams):\n"
                      "    print(v.toString())\n",
                      narrowed))
      << narrowed;
  std::string refused;
  EXPECT_FALSE(emitFor("from js import JSON, URLSearchParams\n"
                       "v = JSON.parse(\"{}\")\n"
                       "print(v.toString())\n"
                       "o = URLSearchParams.new(\"a=1\").get(\"a\")\n"
                       "print(isinstance(o, URLSearchParams))\n",
                       refused));
  EXPECT_NE(refused.find("does not provide manifest method 'toString'"),
            std::string::npos)
      << refused;
  EXPECT_NE(refused.find("needs a JavaScript value"), std::string::npos)
      << refused;
}

// A generator handed to a manifest method whose parameter takes the generator
// itself is passed as it is; only a parameter declared `Iterable` consumes it
// into a list.
TEST(EmitterTest, AGeneratorAppendedToAListStaysAGenerator) {
  mlir::MLIRContext context(testRegistry());
  lython::emitter::EmitResult appended = emitSource(
      "from typing import Generator\n\n\n"
      "def worker(n: int) -> Generator[int, None, None]:\n"
      "    yield n\n\n\n"
      "tasks = [worker(1)]\n"
      "task = tasks.pop(0)\n"
      "tasks.append(task)\n",
      context);
  EXPECT_TRUE(appended.ok());
  lython::emitter::EmitResult joined = emitSource(
      "from typing import Iterator\n\n\n"
      "def words() -> Iterator[str]:\n"
      "    yield \"a\"\n\n\n"
      "print(\"-\".join(words()))\n",
      context);
  EXPECT_TRUE(joined.ok());
}

// A yield whose expression is typed by the narrowing above it -- a `while`
// guard on an Optional -- takes the generator's annotation.
TEST(EmitterTest, AYieldUnderAWhileGuardTakesTheAnnotation) {
  mlir::MLIRContext context(testRegistry());
  lython::emitter::EmitResult emitted = emitSource(
      "from typing import Iterator\n"
      "class Node:\n"
      "    def __init__(self, name: str, nxt: \"Node | None\") -> None:\n"
      "        self.name = name\n"
      "        self.nxt = nxt\n"
      "def walk(head: Node | None) -> Iterator[str]:\n"
      "    cur = head\n"
      "    while cur is not None:\n"
      "        yield cur.name\n"
      "        cur = cur.nxt\n",
      context);
  EXPECT_TRUE(emitted.ok());
}

// What: `await` reaches only what can suspend -- a coroutine, or an object
// whose class defines `__await__` -- and only inside `async def`.
TEST(EmitterTest, AnAwaitTakesACoroutineInsideACoroutine) {
  mlir::MLIRContext context(testRegistry());
  lython::emitter::EmitResult outside = emitSource(
      "async def one() -> int:\n"
      "    return 1\n\n\n"
      "def f() -> None:\n"
      "    await one()\n",
      context);
  EXPECT_TRUE(reportsDiagnostic(outside, "'await' outside an async function"));
  lython::emitter::EmitResult notAwaitable = emitSource(
      "async def f() -> None:\n"
      "    await 3\n",
      context);
  EXPECT_TRUE(reportsDiagnostic(
      notAwaitable,
      "await needs a coroutine or an object whose class defines __await__"));
  lython::emitter::EmitResult awaited = emitSource(
      "from typing import Generator\n\n\n"
      "class Ready:\n"
      "    def __await__(self) -> Generator[object, None, int]:\n"
      "        yield None\n"
      "        return 2\n\n\n"
      "async def one() -> int:\n"
      "    return 1\n\n\n"
      "async def f() -> int:\n"
      "    return await one() + await Ready()\n",
      context);
  EXPECT_TRUE(awaited.ok());
}

// What: `async with` and `async for` call Python dunders; a class without
// them is refused by name, with the dunder it lacks.
TEST(EmitterTest, AsyncWithAndAsyncForNameTheMissingDunder) {
  mlir::MLIRContext context(testRegistry());
  lython::emitter::EmitResult syncManager = emitSource(
      "class Plain:\n"
      "    def __enter__(self) -> \"Plain\":\n"
      "        return self\n\n"
      "    def __exit__(self, a: object, b: object, c: object) -> None:\n"
      "        pass\n\n\n"
      "async def f() -> None:\n"
      "    async with Plain():\n"
      "        pass\n",
      context);
  EXPECT_TRUE(reportsDiagnostic(syncManager, "defines no __aenter__"));
  lython::emitter::EmitResult list = emitSource(
      "async def f() -> None:\n"
      "    async for x in [1, 2]:\n"
      "        print(x)\n",
      context);
  EXPECT_TRUE(reportsDiagnostic(list, "defines no __aiter__"));
}

// What: an async generator is refused where it is defined.
TEST(EmitterTest, AnAsyncGeneratorIsRefusedAtItsDefinition) {
  mlir::MLIRContext context(testRegistry());
  lython::emitter::EmitResult emitted = emitSource(
      "from typing import AsyncIterator\n\n\n"
      "async def ticks() -> AsyncIterator[int]:\n"
      "    yield 1\n",
      context);
  EXPECT_TRUE(reportsDiagnostic(
      emitted, "async generator function lowering is not implemented yet"));
}

namespace {

// The driver's emit entry point for one main source on `triple`, which is
// where `js` exists and where module-level platform branches are decided.
ImportedModuleEmit emitMainFor(llvm::StringRef source, llvm::StringRef triple,
                               bool jsHost = false) {
  ImportedModuleEmit result;
  mlir::MLIRContext context(testRegistry());
  mlir::OwningOpRef<mlir::ModuleOp> module;
  llvm::raw_string_ostream diag(result.diagnostics);
  lython::driver::DriverOptions options;
  options.targetTriple =
      triple.empty() ? llvm::sys::getDefaultTargetTriple() : triple.str();
  options.jsHost = jsHost;
  result.succeeded = mlir::succeeded(lython::driver::emitMLIRFromSource(
      source, "main.py", "<lython-no-import-dir>", options, context, module,
      diag));
  return result;
}

} // namespace

// What: a generic host class used with arguments -- `Promise[int]` as an
// annotation and as a call's result -- has its methods, read with T as int,
// and a host `number` parameter takes an int.
TEST(EmitterTest, AGenericHostClassIsReadWithItsArguments) {
  ImportedModuleEmit emitted = emitMainFor(
      "from js import Promise, setTimeout\n\n\n"
      "def show(p: Promise[int]) -> None:\n"
      "    def ok(v: int) -> None:\n"
      "        print(v)\n\n"
      "    p.then(ok)\n\n\n"
      "def later() -> None:\n"
      "    print(\"later\")\n\n\n"
      "show(Promise.resolve(4))\n"
      "setTimeout(later, 10)\n",
      "wasm32-wasip1", /*jsHost=*/true);
  EXPECT_TRUE(emitted.succeeded) << emitted.diagnostics;
}

// What: `await` on a host value takes a Promise, and asyncio, whose loop is
// what waits on it.
TEST(EmitterTest, AnAwaitOnAHostValueTakesAPromiseAndAsyncio) {
  ImportedModuleEmit noAsyncio = emitMainFor(
      "from js import Promise\n\n\n"
      "async def f() -> int:\n"
      "    return await Promise.resolve(1)\n",
      "wasm32-wasip1", /*jsHost=*/true);
  EXPECT_FALSE(noAsyncio.succeeded);
  EXPECT_NE(noAsyncio.diagnostics.find(
                "await on a JavaScript Promise needs asyncio"),
            std::string::npos)
      << noAsyncio.diagnostics;
  ImportedModuleEmit notAPromise = emitMainFor(
      "import asyncio\n"
      "from js import document\n\n\n"
      "async def f() -> None:\n"
      "    await document\n",
      "wasm32-wasip1", /*jsHost=*/true);
  EXPECT_FALSE(notAPromise.succeeded);
  EXPECT_NE(notAPromise.diagnostics.find(
                "await on a JavaScript value needs a Promise"),
            std::string::npos)
      << notAPromise.diagnostics;
}

// What: a module-level `if sys.platform == ...` branch for another platform
// imports nothing, so a module may import `js` where there is a host and
// still compile where there is none.
TEST(EmitterTest, ABranchForAnotherPlatformImportsNothing) {
  ImportedModuleEmit emitted = emitWithImportedModule(
      "hosted",
      "import sys\n\n"
      "if sys.platform == \"wasi\":\n"
      "    import js\n\n"
      "    def where() -> str:\n"
      "        return \"host\"\n"
      "else:\n\n"
      "    def where() -> str:\n"
      "        return \"native\"\n",
      "import hosted\nprint(hosted.where())\n");
  EXPECT_TRUE(emitted.succeeded) << emitted.diagnostics;
}

// What: a parameter shadows a module imported under its name, in the
// program and in a module it imports (time.py's `mktime(t)` under
// `import time as t`).
TEST(EmitterTest, AParameterShadowsAModuleOfItsName) {
  ImportedModuleEmit emitted = emitMainFor(
      "import time as t\n\n\n"
      "def f(t: list[int]) -> int:\n"
      "    return t.count(1)\n\n\n"
      "print(f([1, 2, 1]), t.monotonic() > 0)\n",
      {});
  EXPECT_TRUE(emitted.succeeded) << emitted.diagnostics;
}

// What: WASI has a JavaScript host only when it is asked for, and
// `sys._js_host` says which, statically: a branch on it imports `js` only
// where there is a host.
TEST(EmitterTest, AWASIProgramHasAJsHostOnlyWhenAskedFor) {
  const char *program = "from js import console\nconsole.log(1)\n";
  ImportedModuleEmit without = emitMainFor(program, "wasm32-wasip1");
  EXPECT_FALSE(without.succeeded);
  EXPECT_NE(without.diagnostics.find("wasm32-wasip1 with --js-host"),
            std::string::npos)
      << without.diagnostics;
  ImportedModuleEmit with =
      emitMainFor(program, "wasm32-wasip1", /*jsHost=*/true);
  EXPECT_TRUE(with.succeeded) << with.diagnostics;
  const char *guarded = "import sys\n\n"
                        "if sys._js_host:\n"
                        "    from js import console\n\n"
                        "    def say() -> None:\n"
                        "        console.log(\"host\")\n"
                        "else:\n\n"
                        "    def say() -> None:\n"
                        "        print(\"no host\")\n";
  for (auto [triple, jsHost] :
       {std::pair<const char *, bool>{"", false},
        {"wasm32-wasip1", false},
        {"wasm32-wasip1", true}}) {
    ImportedModuleEmit emitted =
        emitWithImportedModule("guarded", guarded,
                               "import guarded\nguarded.say()\n", ".py",
                               triple, jsHost);
    EXPECT_TRUE(emitted.succeeded) << triple << " " << emitted.diagnostics;
  }
}

// What: a codec error takes the codec's arguments -- UnicodeDecodeError and
// UnicodeEncodeError five, UnicodeTranslateError four, as typeshed declares
// them -- and a single message is refused here, where CPython raises
// TypeError when the call runs; the codec form compiles.
TEST(EmitterTest, ACodecErrorTakesItsCodecArguments) {
  lython::driver::DriverOptions native;
  native.targetTriple = llvm::sys::getDefaultTargetTriple();
  for (const char *refused :
       {"e = UnicodeDecodeError(\"bad\")\n",
        "e = UnicodeEncodeError(\"ascii\", \"x\", 0, 1)\n",
        "e = UnicodeTranslateError(\"x\", 0, 1)\n"}) {
    mlir::MLIRContext context(testRegistry());
    mlir::OwningOpRef<mlir::ModuleOp> module;
    std::string diagnostics;
    llvm::raw_string_ostream diag(diagnostics);
    EXPECT_TRUE(mlir::failed(lython::driver::emitMLIRFromSource(
        refused, "main.py", "<lython-no-import-dir>", native, context, module,
        diag)))
        << refused;
    EXPECT_NE(diagnostics.find("has manifest method '__init__' but no "
                               "signature that accepts"),
              std::string::npos)
        << refused << diagnostics;
  }
  mlir::MLIRContext context(testRegistry());
  mlir::OwningOpRef<mlir::ModuleOp> module;
  std::string diagnostics;
  llvm::raw_string_ostream diag(diagnostics);
  EXPECT_TRUE(mlir::succeeded(lython::driver::emitMLIRFromSource(
      "d = UnicodeDecodeError(\"utf-8\", b\"\\xff\", 0, 1, \"bad\")\n"
      "e = UnicodeEncodeError(\"ascii\", \"\\xe9\", 0, 1, \"bad\")\n"
      "t = UnicodeTranslateError(\"\\xe9\", 0, 1, \"bad\")\n"
      "print(d.start, e.reason, t.end, d.object, e.encoding, t.object)\n",
      "main.py", "<lython-no-import-dir>", native, context, module, diag)))
      << diagnostics;
}

// What: a constructor call is checked against the class's declared __init__
// arities whether it passes arguments or not -- slice() and range() take one to
// three, so none, four, or a str/float bound is refused at emit, and the
// arities CPython accepts (frozenset() included) emit.
TEST(EmitterTest, AConstructorCallIsCheckedAgainstItsInitArities) {
  lython::driver::DriverOptions native;
  native.targetTriple = llvm::sys::getDefaultTargetTriple();
  for (const char *refused :
       {"s = slice()\n", "s = slice(1, 2, 3, 4)\n", "s = slice(\"a\")\n",
        "s = slice(1.5)\n", "r = range()\n", "r = range(1, 2, 3, 4)\n",
        "r = range(1.5)\n"}) {
    mlir::MLIRContext context(testRegistry());
    mlir::OwningOpRef<mlir::ModuleOp> module;
    std::string diagnostics;
    llvm::raw_string_ostream diag(diagnostics);
    EXPECT_TRUE(mlir::failed(lython::driver::emitMLIRFromSource(
        refused, "main.py", "<lython-no-import-dir>", native, context, module,
        diag)))
        << refused;
    EXPECT_NE(diagnostics.find("has manifest method '__init__' but no "
                               "signature that accepts"),
              std::string::npos)
        << refused << diagnostics;
  }
  mlir::MLIRContext context(testRegistry());
  mlir::OwningOpRef<mlir::ModuleOp> module;
  std::string diagnostics;
  llvm::raw_string_ostream diag(diagnostics);
  EXPECT_TRUE(mlir::succeeded(lython::driver::emitMLIRFromSource(
      "a = slice(1)\nb = slice(None, 2)\nc = slice(1, None, -1)\n"
      "f = frozenset()\nr = range(3)\nq = range(1, 9, 2)\n"
      "print(a, b, c, f, r, q)\n",
      "main.py", "<lython-no-import-dir>", native, context, module, diag)))
      << diagnostics;
}

// What: a slice keeps the objects it is given and reads its bounds back as
// `int | None`, so a bool bound -- a literal, or a `bool | None` value -- is
// refused naming the argument, where reading it back as an int would print 1
// for CPython's True.
TEST(EmitterTest, ASliceBoundIsNotABool) {
  lython::driver::DriverOptions native;
  native.targetTriple = llvm::sys::getDefaultTargetTriple();
  for (auto [source, argument] :
       std::initializer_list<std::pair<const char *, const char *>>{
           {"s = slice(True)\n", "argument 1"},
           {"s = slice(1, False)\n", "argument 2"},
           {"def f(b: bool | None) -> slice:\n"
            "    return slice(None, 2, b)\n",
            "argument 3"}}) {
    mlir::MLIRContext context(testRegistry());
    mlir::OwningOpRef<mlir::ModuleOp> module;
    std::string diagnostics;
    llvm::raw_string_ostream diag(diagnostics);
    EXPECT_TRUE(mlir::failed(lython::driver::emitMLIRFromSource(
        source, "main.py", "<lython-no-import-dir>", native, context, module,
        diag)))
        << source;
    EXPECT_NE(diagnostics.find(std::string("slice() keeps ") + argument +
                               " as it is given and reads it back as int; "
                               "this call gives it bool"),
              std::string::npos)
        << source << diagnostics;
  }
}

// What: `slice[A, B, C]` -- typeshed's generic spelling -- names the slice
// type this compiler has, quoted or not, so a parameter annotated that way
// accepts the slices passed to it.
TEST(EmitterTest, ASubscriptedSliceAnnotationNamesTheSlice) {
  lython::driver::DriverOptions native;
  native.targetTriple = llvm::sys::getDefaultTargetTriple();
  mlir::MLIRContext context(testRegistry());
  mlir::OwningOpRef<mlir::ModuleOp> module;
  std::string diagnostics;
  llvm::raw_string_ostream diag(diagnostics);
  EXPECT_TRUE(mlir::succeeded(lython::driver::emitMLIRFromSource(
      "def f(s: slice[int, int, int]) -> slice:\n"
      "    return s\n"
      "def g(s: \"slice[int, None, None]\") -> int:\n"
      "    return 0 if s.start is None else s.start\n"
      "print(f(slice(1, 2, 3)), g(slice(4)))\n",
      "main.py", "<lython-no-import-dir>", native, context, module, diag)))
      << diagnostics;
}

// What: what CPython's bytearray refuses with a TypeError as it runs -- a float
// or a str without an encoding to build from, a list to `+=` -- is refused
// here when the program is compiled, and the spellings it accepts emit.
// A classmethod through an instance is re-spelled as the class's call only
// while the class name still names the class: a program that binds `bytes`
// keeps the refusal instead of calling whatever the name now holds.
// A union cell -- a module global a function uses, a class attribute -- holds
// each member as itself, so a numeric value of a rung the union does not name
// is refused where it is written, naming the union as the program spelled it.
TEST(EmitterTest, AUnionCellRefusesARungItDoesNotHold) {
  lython::driver::DriverOptions native;
  native.targetTriple = llvm::sys::getDefaultTargetTriple();
  for (auto [source, expected] :
       std::initializer_list<std::pair<const char *, const char *>>{
           {"o: int | None = None\no = True\ndef r() -> None:\n    print(o)\n",
            "module global 'o' holds int | None and this assignment gives it "
            "bool"},
           {"o: float | None = None\ndef w() -> None:\n    global o\n    o = "
            "1\n",
            "module global 'o' holds float | None and this assignment gives it "
            "int"},
           {"class P:\n    v: int | None = True\n",
            "class attribute 'P.v' holds int | None and this initializer gives "
            "it bool"}}) {
    mlir::MLIRContext context(testRegistry());
    mlir::OwningOpRef<mlir::ModuleOp> module;
    std::string diagnostics;
    llvm::raw_string_ostream diag(diagnostics);
    EXPECT_TRUE(mlir::failed(lython::driver::emitMLIRFromSource(
        source, "main.py", "<lython-no-import-dir>", native, context, module,
        diag)))
        << source;
    EXPECT_NE(diagnostics.find(expected), std::string::npos)
        << source << diagnostics;
  }
}

TEST(EmitterTest, AClassmethodThroughAnInstanceNeedsItsClassName) {
  lython::driver::DriverOptions native;
  native.targetTriple = llvm::sys::getDefaultTargetTriple();
  {
    mlir::MLIRContext context(testRegistry());
    mlir::OwningOpRef<mlir::ModuleOp> module;
    std::string diagnostics;
    llvm::raw_string_ostream diag(diagnostics);
    EXPECT_TRUE(mlir::succeeded(lython::driver::emitMLIRFromSource(
        "a = b\"x\".fromhex(\"41\")\nd: dict[str, int] = {}\n"
        "e = d.fromkeys([\"k\"], 1)\nt = \"s\".maketrans(\"a\", \"b\")\n"
        "print(a, e, t)\n",
        "main.py", "<lython-no-import-dir>", native, context, module, diag)))
        << diagnostics;
  }
  mlir::MLIRContext context(testRegistry());
  mlir::OwningOpRef<mlir::ModuleOp> module;
  std::string diagnostics;
  llvm::raw_string_ostream diag(diagnostics);
  EXPECT_TRUE(mlir::failed(lython::driver::emitMLIRFromSource(
      "def f() -> None:\n    bytes = 3\n    print(b\"x\".fromhex(\"41\"), "
      "bytes)\nf()\n",
      "main.py", "<lython-no-import-dir>", native, context, module, diag)));
  EXPECT_NE(diagnostics.find("has manifest method 'fromhex' but no signature"),
            std::string::npos)
      << diagnostics;
}

// A generic function's non-literal default is one value every call shares --
// CPython evaluates it once at the def -- so a call that leaves it out where
// its specialization types the parameter otherwise has no value to pass.
// Passing it, or a default whose type does not depend on the type parameter,
// compiles.
TEST(EmitterTest, AGenericDefaultIsOneValueForEveryCall) {
  lython::driver::DriverOptions native;
  native.targetTriple = llvm::sys::getDefaultTargetTriple();
  {
    mlir::MLIRContext context(testRegistry());
    mlir::OwningOpRef<mlir::ModuleOp> module;
    std::string diagnostics;
    llvm::raw_string_ostream diag(diagnostics);
    EXPECT_TRUE(mlir::succeeded(lython::driver::emitMLIRFromSource(
        "def g[T](x: T, xs: list[int] = [1]) -> int:\n    return len(xs)\n"
        "def h[T](x: T, ys: list[T] = []) -> int:\n    return len(ys)\n"
        "print(g(1), g(\"a\"), h(1, [2]))\n",
        "main.py", "<lython-no-import-dir>", native, context, module, diag)))
        << diagnostics;
  }
  mlir::MLIRContext context(testRegistry());
  mlir::OwningOpRef<mlir::ModuleOp> module;
  std::string diagnostics;
  llvm::raw_string_ostream diag(diagnostics);
  EXPECT_TRUE(mlir::failed(lython::driver::emitMLIRFromSource(
      "def h[T](x: T, ys: list[T] = []) -> int:\n    return len(ys)\n"
      "print(h(1))\n",
      "main.py", "<lython-no-import-dir>", native, context, module, diag)));
  EXPECT_NE(diagnostics.find("the default of 'ys' is one value every call "
                             "shares"),
            std::string::npos)
      << diagnostics;
  EXPECT_NE(diagnostics.find("where the parameter is list[int]"),
            std::string::npos)
      << diagnostics;
}

TEST(EmitterTest, ABytearrayTakesWhatCPythonTakes) {
  lython::driver::DriverOptions native;
  native.targetTriple = llvm::sys::getDefaultTargetTriple();
  for (auto [source, method] :
       std::initializer_list<std::pair<const char *, const char *>>{
           {"b = bytearray(1.5)\n", "'__init__'"},
           {"b = bytearray(\"abc\")\n", "'__init__'"},
           {"b = bytearray(b\"a\")\nb += [1]\n", "'__iadd__'"}}) {
    mlir::MLIRContext context(testRegistry());
    mlir::OwningOpRef<mlir::ModuleOp> module;
    std::string diagnostics;
    llvm::raw_string_ostream diag(diagnostics);
    EXPECT_TRUE(mlir::failed(lython::driver::emitMLIRFromSource(
        source, "main.py", "<lython-no-import-dir>", native, context, module,
        diag)))
        << source;
    EXPECT_NE(diagnostics.find(std::string("has manifest method ") + method +
                               " but no signature that accepts"),
              std::string::npos)
        << source << diagnostics;
  }
  mlir::MLIRContext context(testRegistry());
  mlir::OwningOpRef<mlir::ModuleOp> module;
  std::string diagnostics;
  llvm::raw_string_ostream diag(diagnostics);
  EXPECT_TRUE(mlir::succeeded(lython::driver::emitMLIRFromSource(
      "a = bytearray()\nb = bytearray(3)\nc = bytearray(b\"x\")\n"
      "d = bytearray(c)\ne = bytearray([1, 2])\nf = bytearray(\"x\", \"utf-8\")\n"
      "a += b\nb *= 2\nc[0:1] = [65]\ndel d[::2]\nprint(a, b, c, d, e, f, 2 * e)\n",
      "main.py", "<lython-no-import-dir>", native, context, module, diag)))
      << diagnostics;
}

// What: a memoryview views bytes, a bytearray or another view -- a str is
// refused as CPython refuses it at run time, and so are deleting an item
// (CPython's "cannot delete memory") and cast(), which this port leaves out;
// the spellings it accepts emit, attributes included.
TEST(EmitterTest, AMemoryviewTakesWhatItCanView) {
  lython::driver::DriverOptions native;
  native.targetTriple = llvm::sys::getDefaultTargetTriple();
  for (auto [source, expected] :
       std::initializer_list<std::pair<const char *, const char *>>{
           {"m = memoryview(\"abc\")\n", "but no signature that accepts"},
           {"m = memoryview(bytearray(b\"ab\"))\ndel m[0]\n",
            "'__delitem__'"},
           {"m = memoryview(b\"ab\")\nc = m.cast(\"B\")\n", "'cast'"}}) {
    mlir::MLIRContext context(testRegistry());
    mlir::OwningOpRef<mlir::ModuleOp> module;
    std::string diagnostics;
    llvm::raw_string_ostream diag(diagnostics);
    EXPECT_TRUE(mlir::failed(lython::driver::emitMLIRFromSource(
        source, "main.py", "<lython-no-import-dir>", native, context, module,
        diag)))
        << source;
    EXPECT_NE(diagnostics.find(expected), std::string::npos)
        << source << diagnostics;
  }
  mlir::MLIRContext context(testRegistry());
  mlir::OwningOpRef<mlir::ModuleOp> module;
  std::string diagnostics;
  llvm::raw_string_ostream diag(diagnostics);
  EXPECT_TRUE(mlir::succeeded(lython::driver::emitMLIRFromSource(
      "b = bytearray(b\"abc\")\nm = memoryview(b)\nr = memoryview(b\"x\")\n"
      "v = memoryview(m)\ns = m[::-1]\nm[0] = 1\nm[0:1] = b\"z\"\n"
      "print(len(m), m[0], s.tolist(), m.readonly, m.format, m.shape,\n"
      "      m.strides, m.nbytes, bytes(m), bytearray(v), m == b\"abc\")\n"
      "with memoryview(r) as w:\n    print(w.tobytes(), w.hex())\n"
      "m.release()\n",
      "main.py", "<lython-no-import-dir>", native, context, module, diag)))
      << diagnostics;
}

// id() is refused where `is` is -- a value type, a function value, a union --
// and the builtins with no value here (`vars`, the Ellipsis object) say what
// they are instead of reading as an unbound name.
TEST(EmitterTest, IdAndTheValuelessBuiltinsSayWhyTheyAreRefused) {
  lython::driver::DriverOptions native;
  native.targetTriple = llvm::sys::getDefaultTargetTriple();
  for (auto [source, expected] :
       std::initializer_list<std::pair<const char *, const char *>>{
           {"print(id(5))\n", "id() of an int/str/float/bytes/complex is "
                               "rejected"},
           {"s = \"a\"\nprint(id(s))\n",
            "id() of an int/str/float/bytes/complex is rejected"},
           {"def f() -> int:\n    return 1\nprint(id(f))\n",
            "id() needs an argument of one reference type"},
           {"class A:\n    pass\ndef g(u: A | None) -> int:\n    return "
            "id(u)\n",
            "id() needs an argument of one reference type"},
           {"class A:\n    pass\nprint(vars(A()))\n",
            "vars() is not supported: an instance here keeps its fields in "
            "fixed slots"},
           {"x = ...\n", "the Ellipsis object (`...` or `Ellipsis` as a "
                         "value) is not supported"},
           {"print(Ellipsis)\n", "the Ellipsis object (`...` or `Ellipsis` "
                                 "as a value) is not supported"}}) {
    mlir::MLIRContext context(testRegistry());
    mlir::OwningOpRef<mlir::ModuleOp> module;
    std::string diagnostics;
    llvm::raw_string_ostream diag(diagnostics);
    EXPECT_TRUE(mlir::failed(lython::driver::emitMLIRFromSource(
        source, "main.py", "<lython-no-import-dir>", native, context, module,
        diag)))
        << source;
    EXPECT_NE(diagnostics.find(expected), std::string::npos)
        << source << diagnostics;
  }
}

// A protocol is refused where CPython refuses it or where this compiler
// cannot give CPython's answer: built directly, generic, a class that misses
// a member passed for it (the message names the member), and isinstance()
// without @runtime_checkable, over a builtin value, or where a class has the
// method names but not the signatures.
TEST(EmitterTest, AProtocolSaysWhyItIsRefused) {
  lython::driver::DriverOptions native;
  native.targetTriple = llvm::sys::getDefaultTargetTriple();
  const std::string shape =
      "from typing import Protocol, runtime_checkable\n"
      "class Shape(Protocol):\n    def area(self) -> float: ...\n"
      "@runtime_checkable\n"
      "class Closer(Protocol):\n    def close(self) -> None: ...\n";
  for (auto [tail, expected] :
       std::initializer_list<std::pair<const char *, const char *>>{
           {"s = Shape()\n", "'Shape' is a Protocol, and protocols cannot be "
                              "instantiated"},
           {"class Blob:\n    def volume(self) -> float:\n        return "
            "1.0\ndef one(s: Shape) -> float:\n    return "
            "s.area()\nprint(one(Blob()))\n",
            "argument 1 ('Blob') does not satisfy the protocol 'Shape': it has "
            "no member 'area'"},
           {"class Odd:\n    def area(self, k: float) -> float:\n        "
            "return k\ndef one(s: Shape) -> float:\n    return "
            "s.area()\nprint(one(Odd()))\n",
            "its 'area' does not take the parameters the protocol's does"},
           {"class Sq:\n    def area(self) -> float:\n        return "
            "1.0\nprint(isinstance(Sq(), Shape))\n",
            "it is not @runtime_checkable"},
           {"x: int | str = 1\nprint(isinstance(x, Closer))\n",
            "is decided here only for values of this program's own classes"},
           {"class Shut:\n    def close(self, now: bool) -> None:\n        "
            "pass\nclass Door:\n    def close(self) -> None:\n        "
            "pass\nprint(isinstance(Door(), Closer))\n",
            "'Shut' has its method names, which CPython's check accepts"}}) {
    std::string source = shape + tail;
    mlir::MLIRContext context(testRegistry());
    mlir::OwningOpRef<mlir::ModuleOp> module;
    std::string diagnostics;
    llvm::raw_string_ostream diag(diagnostics);
    EXPECT_TRUE(mlir::failed(lython::driver::emitMLIRFromSource(
        source, "main.py", "<lython-no-import-dir>", native, context, module,
        diag)))
        << source;
    EXPECT_NE(diagnostics.find(expected), std::string::npos)
        << source << diagnostics;
  }
  {
    const char *source = "from typing import Protocol, TypeVar\nT = "
                         "TypeVar(\"T\")\nclass Box(Protocol[T]):\n    def "
                         "get(self) -> T: ...\n";
    mlir::MLIRContext context(testRegistry());
    mlir::OwningOpRef<mlir::ModuleOp> module;
    std::string diagnostics;
    llvm::raw_string_ostream diag(diagnostics);
    EXPECT_TRUE(mlir::failed(lython::driver::emitMLIRFromSource(
        source, "main.py", "<lython-no-import-dir>", native, context, module,
        diag)));
    EXPECT_NE(diagnostics.find("a generic Protocol (`Protocol[T]`) is not "
                               "supported yet"),
              std::string::npos)
        << diagnostics;
  }
}

// What: `__subclasses__()` is refused when a list it may answer would hold a
// class written inside a function -- asked of the class itself, and of a class
// value that may hold a class one level up from it -- and answered for the
// function's own class, which has no subclass to list.
TEST(EmitterTest, SubclassesThatMayHoldAFunctionLocalClassAreRefused) {
  lython::driver::DriverOptions native;
  native.targetTriple = llvm::sys::getDefaultTargetTriple();
  auto emit = [&](const std::string &source, std::string &diagnostics) {
    mlir::MLIRContext context(testRegistry());
    mlir::OwningOpRef<mlir::ModuleOp> module;
    llvm::raw_string_ostream diag(diagnostics);
    return mlir::succeeded(lython::driver::emitMLIRFromSource(
        source, "main.py", "<lython-no-import-dir>", native, context, module,
        diag));
  };
  const std::string classes = "class Base:\n    pass\nclass Mid(Base):\n"
                              "    pass\ndef make() -> type[Base]:\n"
                              "    class Local(Mid):\n        pass\n"
                              "    print(Local.__subclasses__())\n"
                              "    return Local\n";
  for (auto [tail, expected] :
       std::initializer_list<std::pair<const char *, const char *>>{
           {"print(Mid.__subclasses__())\n",
            "'Mid.__subclasses__()' would list 'make.<locals>.Local'"},
           {"k: type[Base] = Base\nprint(k.__subclasses__())\n",
            "'Mid.__subclasses__()' would list 'make.<locals>.Local'"}}) {
    std::string diagnostics;
    std::string source = classes + tail;
    EXPECT_FALSE(emit(source, diagnostics)) << source;
    EXPECT_NE(diagnostics.find(expected), std::string::npos)
        << source << diagnostics;
  }
  std::string diagnostics;
  std::string source = classes + "print(Base.__subclasses__(), make())\n";
  EXPECT_TRUE(emit(source, diagnostics)) << source << diagnostics;
}

// What: `__doc__` is refused where the docstring is not the program's -- a
// builtin function, a builtin class or value, a class a decorator or
// NamedTuple writes the docstring of -- and answered for the program's own
// function, class, method and module.
TEST(EmitterTest, ADocstringIsRefusedWhereItIsNotTheProgramsOwn) {
  lython::driver::DriverOptions native;
  native.targetTriple = llvm::sys::getDefaultTargetTriple();
  auto emit = [&](const std::string &source, std::string &diagnostics) {
    mlir::MLIRContext context(testRegistry());
    mlir::OwningOpRef<mlir::ModuleOp> module;
    llvm::raw_string_ostream diag(diagnostics);
    return mlir::succeeded(lython::driver::emitMLIRFromSource(
        source, "main.py", "<lython-no-import-dir>", native, context, module,
        diag));
  };
  for (auto [source, expected] :
       std::initializer_list<std::pair<const char *, const char *>>{
           {"print(len.__doc__)\n",
            "the docstring of builtin 'len' is not retained"},
           {"import math\nprint(math.sqrt.__doc__)\n",
            "the docstring of builtin 'math.sqrt' is not retained"},
           {"print(int.__doc__)\n", "the docstring of 'int' is not retained"},
           {"x = 'a'\nprint(x.__doc__)\n",
            "the docstring of 'str' is not retained"},
           {"from dataclasses import dataclass\n@dataclass\nclass P:\n"
            "    x: int\nprint(P.__doc__)\n",
            "'P' is decorated, and a decorator may write its docstring"},
           {"from typing import NamedTuple\nclass N(NamedTuple):\n"
            "    x: int\nprint(N.__doc__)\n",
            "the docstring of NamedTuple 'N' is generated"}}) {
    std::string diagnostics;
    EXPECT_FALSE(emit(source, diagnostics)) << source;
    EXPECT_NE(diagnostics.find(expected), std::string::npos)
        << source << diagnostics;
  }
  std::string diagnostics;
  EXPECT_TRUE(emit("\"\"\"Module.\"\"\"\nclass C:\n    \"\"\"C.\"\"\"\n"
                   "    def m(self) -> None:\n        \"\"\"m.\"\"\"\n"
                   "def f() -> None:\n    pass\n"
                   "print(__doc__, C.__doc__, C.m.__doc__, C().__doc__, "
                   "f.__doc__)\n",
                   diagnostics))
      << diagnostics;
}

// What: a manifest method that only the emitter calls (a slice, a sort with a
// key) is not an attribute of the builtin type in a program, as it is not in
// CPython -- read or called -- while a class of the program's own may still
// define a method spelled the same.
TEST(EmitterTest, AnInternalManifestMethodIsNotAnAttributeOfABuiltin) {
  mlir::MLIRContext context(testRegistry());
  auto refusedWith = [&](llvm::StringRef source, llvm::StringRef message) {
    lython::emitter::EmitResult emitted = emitSource(source, context);
    EXPECT_FALSE(emitted.ok()) << source.str();
    bool found = false;
    for (const lython::parser::Diagnostic &diagnostic : emitted.diagnostics)
      found = found || diagnostic.message.find(message.str()) != std::string::npos;
    EXPECT_TRUE(found) << source.str();
  };
  refusedWith("xs = [3, 1, 2]\nprint(xs.__getslice__(0, 2, 1, 3))\n",
              "'list' object has no attribute '__getslice__'");
  refusedWith("xs = [3, 1, 2]\nxs.__sort_by_keys__([0, 1, 2], False)\n",
              "'list' object has no attribute '__sort_by_keys__'");
  refusedWith("s = 'abc'\nf = s.__getslice__\n",
              "'str' object has no attribute '__getslice__'");
  refusedWith("xs = [3, 1, 2]\nprint(xs.__best_index__(True))\n",
              "'list' object has no attribute '__best_index__'");
  lython::emitter::EmitResult own = emitSource(
      "class C:\n"
      "    def __getslice__(self, a: int) -> int:\n"
      "        return a + 1\n"
      "print(C().__getslice__(4))\n",
      context);
  EXPECT_TRUE(own.ok());
}

// What: `@overload` declarations without the def that implements them are
// refused, for a module function and for a method, and a call that no
// overload accepts is refused even where the implementation's own signature
// would take it.
TEST(EmitterTest, AnOverloadIsTheWholeSignatureItsCallsSee) {
  mlir::MLIRContext context(testRegistry());
  auto refusedWith = [&](llvm::StringRef source, llvm::StringRef message) {
    lython::emitter::EmitResult emitted = emitSource(source, context);
    EXPECT_FALSE(emitted.ok()) << source.str();
    bool found = false;
    for (const lython::parser::Diagnostic &diagnostic : emitted.diagnostics)
      found = found || diagnostic.message.find(message.str()) != std::string::npos;
    EXPECT_TRUE(found) << source.str();
  };
  refusedWith("from typing import overload\n"
              "@overload\n"
              "def f(x: int) -> str: ...\n"
              "@overload\n"
              "def f(x: str) -> int: ...\n",
              "'f' is declared with @overload and has no implementation");
  refusedWith("from typing import overload\n"
              "class C:\n"
              "    @overload\n"
              "    def m(self, x: int) -> str: ...\n"
              "    def n(self) -> int:\n"
              "        return 1\n"
              "print(C().n())\n",
              "'m' is declared with @overload and has no implementation");
  const char *declared = "from typing import overload\n"
                         "@overload\n"
                         "def f(x: int) -> str: ...\n"
                         "@overload\n"
                         "def f(x: str) -> int: ...\n"
                         "def f(x: int | str | bytes) -> int | str:\n"
                         "    return 1\n"
                         "class C:\n"
                         "    @overload\n"
                         "    def m(self, x: int) -> str: ...\n"
                         "    @overload\n"
                         "    def m(self, x: str) -> int: ...\n"
                         "    def m(self, x: int | str | bytes) -> int | str:\n"
                         "        return 1\n";
  refusedWith(std::string(declared) + "print(f(b'x'))\n",
              "no @overload of 'f' accepts these arguments");
  refusedWith(std::string(declared) + "print(C().m(b'x'))\n",
              "no @overload of 'm' accepts these arguments");
  lython::emitter::EmitResult accepted = emitSource(
      std::string(declared) + "print(f(1).upper(), C().m('s') + 1)\n", context);
  EXPECT_TRUE(accepted.ok());
}
