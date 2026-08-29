// MODULE GLOBALS: which module-level names get storage, and what kind.
//
// Three answers, and the annotation is what picks between them (the lowering
// side is Passes/Runtime/Ops/GlobalOps.cpp):
//
//   an ADDRESS (`ctypes.c_void_p`) ... a native machine-word cell, which a
//       signal handler may read because touching it never allocates
//   a storage-backed object ......... any CONTRACT: a cell holding the value
//       group, rebindable from a function that declared `global NAME`
//   value-bound ..................... an annotation that is not a contract --
//       a union, a protocol, a callable, `type[X]`, a tensor
//
// ⛔ THE SECOND ANSWER USED TO EXCLUDE EVERY CONTAINER, on the ground that "a
// structural mutation reallocates the interior arrays through SSA rebinding,
// which a cell would go stale against". That describes a representation this
// compiler no longer has. `builtins.mlir` states the current one twice, once
// per container: "a growth writes the new address THROUGH the handle, so every
// holder observes it with no further action and a mutation has nothing to
// rename. That is what lets ensure_capacity / extend / __setslice__ /
// __delslice__ be void and non-transferring." A cell holds the handle, and the
// handle is what stays put.
//
// What that exclusion cost was not one construct: `T: dict[str, int] = {...}`
// read from any function was "unresolved name 'T'", and so were `list`,
// `tuple`, `set` and every stdlib contract. A module-level table read by a
// function is ordinary Python.
//
// ⛔ Why the residue is spelled as "not a contract" rather than enumerated:
// the enumeration is what went stale. A union stays value-bound so isinstance
// narrowing keeps working on the module flow, and the rest have no runtime
// value group to put in a cell; both are properties of the annotation, and
// neither is a list of names.
//
// Why NOT in EmitterClasses.cpp, where this lived: nothing here is about a
// class. It shared the file only because `collectStaticModuleAssignments`
// (still there) feeds the same static-attribute table the class bodies do.

#include "EmitterCore.h"
#include "EmitterPyOps.h"

#include "AstAccess.h"
#include "AstSynth.h"
#include "ClosureAnalysis.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/StringSet.h"

namespace lython::emitter {

void ModuleEmitter::collectModuleGlobals(const parser::Node &moduleNode) {
  // Opt-in: an annotated module-level assignment (`NAME: T = ...`) becomes a
  // storage-backed mutable global. Plain `NAME = expr` at module scope keeps
  // its value-binding behaviour (module-scope constants).
  const auto *body = ast::nodeList(moduleNode, "body");
  if (!body)
    return;
  // ⭐ A DECORATED NAME IS A MODULE CELL, because that is what CPython makes it.
  // `@d def f` rebinds the module name to `d(f)` and every later reference --
  // a recursion inside f's own body, another function calling f -- resolves
  // THAT at call time. A body here binds the name to the emitted SYMBOL, which
  // is the undecorated function, so a decorated `fib(6)` printed 9 where
  // CPython prints 33. Declaring the cell before any body is typed is what
  // makes those references read the wrapper.
  for (const parser::NodePtr &statement : *body) {
    if (!statement || (statement->kind != "FunctionDef" &&
                       statement->kind != "AsyncFunctionDef"))
      continue;
    auto decorated = ast::string(*statement, "name");
    const auto *decorators = ast::nodeList(*statement, "decorator_list");
    if (!decorated || !decorators || decorators->empty())
      continue;
    if (moduleGlobals.count(*decorated))
      continue;
    parser::NodePtr applied = synth::name(*decorated, statement->range);
    bool rebinding = false;
    for (const parser::NodePtr &decorator : llvm::reverse(*decorators)) {
      if (!decorator || decorator->kind != "Name" ||
          !moduleFunctionNames.count(ast::nameSpelling(*decorator)))
        continue;
      std::vector<parser::NodePtr> arguments;
      arguments.push_back(applied);
      applied = synth::call(
          synth::name(ast::nameSpelling(*decorator), statement->range),
          std::move(arguments), statement->range);
      rebinding = true;
    }
    if (!rebinding)
      continue;
    mlir::Type decoratedType = types.widenLiteral(types.inferExpr(applied.get()));
    if (!mlir::isa_and_nonnull<py::CallableType>(decoratedType))
      continue;
    decoratorApplications.push_back(applied);
    moduleGlobals[*decorated] = decoratedType;
    types.bindSymbol(*decorated, decoratedType);
  }
  for (const parser::NodePtr &statement : *body) {
    if (!statement || statement->kind != "AnnAssign")
      continue;
    const parser::Node *target = ast::node(*statement, "target");
    if (!target || target->kind != "Name")
      continue;
    mlir::Type annotated = types.widenLiteral(
        types.annotationType(ast::node(*statement, "annotation")));
    if (!annotated)
      continue;
    // Every contract gets a cell; the file header has the measurement that
    // replaced the container exclusion. `ctypes.c_void_p` is inside this set
    // as the ADDRESS spelling -- its cell is the machine word, which is what a
    // signal handler may read and what an `int` global deliberately no longer
    // is (see lowerGlobalGet).
    bool storageBacked = mlir::isa<py::ContractType>(annotated);
    // ⛔ EXCEPT a container whose ELEMENT type is a union. A cell hands back
    // the handle and nothing else, and a union-typed element read needs the
    // per-element evidence the literal recorded -- the runtime read declines a
    // union (its tag lane is an i64, not a memref) and the read is then
    // "runtime manifest has no builtins.list.__getitem__ method". So
    //
    //     xs: list[int | None] = [1, None, 3]
    //     print(xs[0])
    //
    // was refused at module scope while the same three lines inside a function
    // ran. Value binding keeps the evidence, which is the pre-existing
    // behaviour for every container and is strictly better than a cell here.
    //
    // The cost is the cell's own benefit: such a global is not readable from a
    // function, exactly as no container global was before it had cells. The
    // real repair is the runtime read learning to build a union from the
    // slot's payload class id, which would also close the second-read case
    // (tests/probe/wb_grid_leftovers_2026_08_16.py).
    if (storageBacked)
      if (auto contract = mlir::dyn_cast<py::ContractType>(annotated))
        for (mlir::Type argument : contract.getArguments())
          if (mlir::isa<py::UnionType>(argument)) {
            storageBacked = false;
            break;
          }
    if (!storageBacked)
      continue;
    llvm::StringRef name = ast::nameSpelling(*target);
    moduleGlobals[name] = annotated;
    types.bindSymbol(name, annotated);
  }

  // A plain `NAME = <literal>` is not a global CELL -- module-scope names
  // stay value-bound, which is why a container one is not visible from a
  // function -- but a name the module binds ONCE to a literal has nothing to
  // go stale against, so its references re-emit the literal. Reading
  // `N = 5` from a function was "unresolved name 'N'" while `N: int = 5`
  // worked, and CPython does not distinguish the two spellings.
  //
  // ⛔ Why NOT make it a storage cell like the annotated form: a cell is a
  // rebindable slot, and re-emitting the literal is strictly cheaper for a
  // name that is never rebound. (Until the boxing repair an int cell was also
  // an unboxed i64, so `fact = 1` grown past 2**63 raised where it printed --
  // measured at 4 goldens, one of them exactly that factorial. That reason is
  // gone; the cheapness one is not.)
  //
  // ⛔ EXCEPT a name some function declares `global`, which the loop below
  // excludes and the cell pass after it then claims. Re-emitting the literal
  // is a READ strategy and there is nothing to write into, so the counter
  // idiom -- `G = 0` with `global G; G += 1` in a function -- was refused:
  // "'global G' names a module global this compiler does not give storage
  // to". A `global` declaration is the strongest statement a program can make
  // that the binding needs storage.
  llvm::StringSet<> boundOnce = singleAssignmentNames(moduleNode);
  llvm::StringSet<> declaredGlobal = moduleGlobalDeclarations(moduleNode);
  for (const parser::NodePtr &statement : *body) {
    if (!statement || statement->kind != "Assign")
      continue;
    const auto *targets = ast::nodeList(*statement, "targets");
    if (!targets || targets->size() != 1 || !targets->front() ||
        targets->front()->kind != "Name")
      continue;
    const parser::Node *value = ast::node(*statement, "value");
    if (!value || value->kind != "Constant" ||
        ast::isNoneField(*value, "value"))
      continue;
    llvm::StringRef name = ast::nameSpelling(*targets->front());
    if (!boundOnce.contains(name) || moduleGlobals.count(name) ||
        declaredGlobal.contains(name))
      continue;
    moduleConstantBindings[name] = value;
  }

  // ⭐ And a plain `NAME = <not a literal>` bound once gets a CELL, which is
  // the same argument one step on: re-emitting is only cheaper when there is
  // something to re-emit, and a container or an instance cannot be re-emitted
  // at all -- a second `{"a": 1}` is a second dict, so a mutation through one
  // name would not be visible through the other. That left every unannotated
  // module-level table and singleton unreachable from a function:
  //
  //     T = {"a": 1}
  //     def look(k: str) -> int:
  //         return T[k]          # emit error: unresolved name 'T'
  //
  //     O = C(4)
  //     def get() -> int:
  //         return O.n           # emit error: unresolved name 'O'
  //
  // ⛔ Why BOUND ONCE and not every plain assignment: a cell has ONE runtime
  // representation, fixed at the declaration, and `x = 1` followed by
  // `x = "a"` is legal Python that a cell cannot hold. The annotated form
  // reports that as a diagnostic because the annotation is the promise; a
  // plain rebinding promises nothing, so it keeps its value binding instead of
  // becoming a new refusal. Names bound once are also what
  // `moduleConstantBindings` above already restricts itself to.
  //
  // ⛔ Why the value's INFERRED type and not an annotation: there is none.
  // `registerModule`'s fixpoint has run by the time this is called, so the
  // inference is the same one the module body will use; a name whose value
  // does not infer to a contract is skipped and stays value-bound, which is
  // the pre-existing behaviour rather than a refusal.
  //
  // ⛔ AND ONLY NAMES A FUNCTION ACTUALLY READS, which is the whole difference
  // between this and a change that broke 80 of 716 tests. A cell is an opaque
  // handle: the module body's own reads lose the bundle's evidence (a
  // sequence's element types, a literal's precision, a concrete contract
  // behind an erased annotation), and `data = b"hello"` followed by
  // `data + b" world"` at module scope then fails with "cannot pass concrete
  // object builtins.bytes as builtins.object". Value binding is strictly
  // better for a name nothing else can see, so the cell is only worth it where
  // a function would otherwise be unable to see the name at all.
  //
  // `lexicalCaptureNames` is the right question because it asks for FREE
  // names: a function with its own local `data` does not put `data` in this
  // set, so the module's `data` keeps its evidence.
  llvm::StringSet<> readFromAFunction;
  {
    llvm::SmallVector<const parser::Node *, 8> callables;
    auto collectCallables = [&](const parser::Node &scope,
                                auto &&recurse) -> void {
      const auto *statements = ast::nodeList(scope, "body");
      if (!statements)
        return;
      for (const parser::NodePtr &statement : *statements) {
        if (!statement)
          continue;
        if (statement->kind == "FunctionDef" ||
            statement->kind == "AsyncFunctionDef") {
          callables.push_back(statement.get());
          continue;
        }
        if (statement->kind == "ClassDef")
          recurse(*statement, recurse);
      }
    };
    collectCallables(moduleNode, collectCallables);
    for (const parser::Node *callable : callables)
      for (const std::string &capture : lexicalCaptureNames(*callable))
        readFromAFunction.insert(capture);
    // ⭐ AND EVERY `global NAME`, which `lexicalCaptureNames` does not report.
    // That walk answers about CLOSURES, and a name a function ASSIGNS is its
    // own local as far as closure capture goes -- `global G` says the opposite
    // and is not one of the things it subtracts.
    for (const auto &entry : declaredGlobal)
      readFromAFunction.insert(entry.getKey());
  }

  for (const parser::NodePtr &statement : *body) {
    if (!statement || statement->kind != "Assign")
      continue;
    const auto *targets = ast::nodeList(*statement, "targets");
    if (!targets || targets->size() != 1 || !targets->front() ||
        targets->front()->kind != "Name")
      continue;
    llvm::StringRef name = ast::nameSpelling(*targets->front());
    if (!boundOnce.contains(name) || moduleGlobals.count(name) ||
        moduleConstantBindings.count(name) ||
        !readFromAFunction.contains(name))
      continue;
    const parser::Node *value = ast::node(*statement, "value");
    if (!value)
      continue;
    mlir::Type inferred = types.widenLiteral(types.inferExpr(value));
    // ⭐ A FUNCTION VALUE IS A GLOBAL LIKE ANY OTHER. `CALLBACK = base` read
    // from a function body was "unresolved name 'CALLBACK'": the cell was
    // never declared, because a callable's static type is a `py.callable` and
    // this filter took contracts only. The VALUE is an ordinary object
    // (`builtins.function`, a header the cell boxes like a list or a class
    // instance); it is only the static type that is spelled differently, and
    // keeping that spelling is what lets the call through the global check its
    // arguments.
    if (!mlir::isa_and_nonnull<py::ContractType>(inferred) &&
        !mlir::isa_and_nonnull<py::CallableType>(inferred))
      continue;
    moduleGlobals[name] = inferred;
    types.bindSymbol(name, inferred);
  }
}

// Every name any function in this module declares `global`, at any depth.
llvm::StringSet<>
ModuleEmitter::moduleGlobalDeclarations(const parser::Node &scope) const {
  llvm::StringSet<> declared;
  auto walk = [&](const parser::Node &node, auto &&recurse) -> void {
    const auto *statements = ast::nodeList(node, "body");
    if (!statements)
      return;
    for (const parser::NodePtr &statement : *statements) {
      if (!statement)
        continue;
      if (statement->kind == "Global") {
        if (const auto *names = ast::stringList(*statement, "names"))
          for (const std::string &name : *names)
            declared.insert(name);
        continue;
      }
      recurse(*statement, recurse);
    }
  };
  walk(scope, walk);
  return declared;
}

void ModuleEmitter::markBoxedModuleGlobal(mlir::Operation *op) const {
  if (options.runtimeInternal)
    return;
  op->setAttr("ly.global.boxed", mlir::UnitAttr::get(&context));
}

bool ModuleEmitter::isModuleGlobalRead(llvm::StringRef name) const {
  // A read resolves to the module global unless a local (function-scope)
  // binding shadows it.
  return moduleGlobals.count(name) && values.find(name) == values.end();
}

bool ModuleEmitter::isModuleGlobalWrite(llvm::StringRef name) const {
  if (!moduleGlobals.count(name))
    return false;
  // Module scope always writes the global; a function writes it only when it
  // declared `global NAME` (otherwise the assignment makes a local).
  return atModuleScope || currentGlobalDecls.count(name);
}

// ⭐ A structural mutation is spelled as an SSA REASSIGNMENT of the receiver
// (the call carries an extra receiver-typed result), so it needs somewhere to
// put that result. A local has its binding; a module global has its cell, and
// the two questions below are the only difference between them at the three
// sites that mutate -- a bound method, a slice assignment, and `__setitem__`.
//
// ⛔ Why the cell is written even from a function that did NOT declare
// `global NAME`, which `isModuleGlobalWrite` refuses: this is not an
// assignment. Python rebinds no name when `L.append(x)` runs; what changes is
// the representation of the object the name already holds, and the cell is
// where that representation lives. Asking `isModuleGlobalWrite` here would
// make `def f(): L.append(1)` silently drop the update.
bool ModuleEmitter::isStructuralMutationRebindable(llvm::StringRef name,
                                                   mlir::Value receiver) const {
  auto bound = values.find(name);
  if (bound != values.end())
    return bound->second.value == receiver;
  return isModuleGlobalRead(name);
}

void ModuleEmitter::rebindStructuralMutation(const parser::Node &at,
                                             llvm::StringRef name,
                                             Value rebound) {
  if (values.find(name) != values.end()) {
    values[name] = rebound;
    return;
  }
  auto op = py::GlobalSetOp::create(builder, loc(at),
                                    builder.getStringAttr(name), rebound.value);
  markBoxedModuleGlobal(op);
}

} // namespace lython::emitter
