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
// compiler no longer has. The runtime states the current one twice, once per
// container (objects/list.mlir, objects/set.mlir): "a growth writes the new
// address THROUGH the handle, so every holder observes it with no further
// action and a mutation has nothing to rename. That is what lets
// ensure_capacity / extend / __setslice__ / __delslice__ be void and
// non-transferring." A cell holds the handle, and the handle is what stays
// put.
//
// What that exclusion cost was not one construct: `T: dict[str, int] = {...}`
// read from any function was "unresolved name 'T'", and so were `list`,
// `tuple`, `set` and every stdlib contract. A module-level table read by a
// function is ordinary Python.
//
// ⛔ Why the residue is spelled as "not a contract" rather than enumerated:
// the enumeration is what went stale -- twice. The container exclusion above
// was the first; a CALLABLE was the second (2026-09-06), and it was worse than
// a missing feature because ANNOTATING took a working program away:
// `DOUBLE = double` resolved through the alias binding and
// `DOUBLE: Callable[[int], int] = double` did not. A function object has a
// value group like anything else.
//
// ⛔ A UNION IS STILL NOT ONE, and the reason is now MEASURED rather than
// argued. Adding `isa<UnionType>` here turns the refusal into a LOWERING
// failure -- "cannot adapt runtime bundle with physical values (i64,
// memref<2xi64>) to expected ABI (memref<2xi64>)" -- so the store side reduces
// the union to one lane where the value has a tag and a payload. The cell
// machinery itself is ready (`moduleObjectGlobalCell` documents a `_v<i>` word
// "for scalar physical values such as a union tag"); what is not is the ABI
// the global's own type is adapted to. That is where the next attempt starts,
// and until then value binding also keeps isinstance narrowing working on the
// module flow. The rest have no runtime value group at all.
//
// Why NOT in EmitterClasses.cpp, where this lived: nothing here is about a
// class. It shared the file only because `collectStaticModuleAssignments`
// (still there) feeds the same static-attribute table the class bodies do.

#include "EmitterCore.h"
#include "EmitterPyOps.h"

#include "AstAccess.h"
#include "AstSynth.h"
#include "ClosureAnalysis.h"
#include "EmitterSupport.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/StringSet.h"

#include <functional>

namespace lython::emitter {

// Every name spelled inside a function, a lambda or a class body, at any
// depth: the module-level names a body may read or declare `global`.
//
// ⛔ A SPELLING, not a resolved reference: a body's local of the same name
// counts too. That only gives a union a cell it did not need, which costs a
// load per module-level read and changes no answer.
static llvm::StringSet<>
namesUsedInsideFunctions(const parser::Node &moduleNode) {
  llvm::StringSet<> names;
  auto walk = [&](const parser::Node &node, bool inside,
                  auto &&recurse) -> void {
    bool nested = inside || node.kind == "FunctionDef" ||
                  node.kind == "AsyncFunctionDef" || node.kind == "Lambda" ||
                  node.kind == "ClassDef";
    if (nested && node.kind == "Name")
      names.insert(ast::nameSpelling(node));
    if (nested && node.kind == "Global")
      if (const auto *declared = ast::stringList(node, "names"))
        for (const std::string &name : *declared)
          names.insert(name);
    for (const parser::Field &field : node.fields) {
      if (const auto *child = std::get_if<parser::NodePtr>(&field.value)) {
        if (*child)
          recurse(**child, nested, recurse);
        continue;
      }
      if (const auto *children =
              std::get_if<std::vector<parser::NodePtr>>(&field.value))
        for (const parser::NodePtr &child : *children)
          if (child)
            recurse(*child, nested, recurse);
    }
  };
  walk(moduleNode, false, walk);
  return names;
}

void ModuleEmitter::collectModuleGlobals(const parser::Node &moduleNode) {
  std::optional<llvm::StringSet<>> usedInsideFunctions;
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
      if (!decorator)
        continue;
      // ⭐ AND A DECORATOR FACTORY, `@deco(arg)`, which is `f = deco(arg)(f)`.
      // The refusal said the intermediate value is "a function the compiler
      // would have to see through", and the hand-written spelling --
      // `shown = tagged("v")(show)` -- has compiled all along, so what was
      // missing was this desugaring and not the seeing.
      parser::NodePtr callee = bindingDecoratorCallee(decorator);
      if (!callee)
        continue;
      std::vector<parser::NodePtr> arguments;
      arguments.push_back(applied);
      applied = synth::call(std::move(callee), std::move(arguments),
                            statement->range);
      rebinding = true;
    }
    if (!rebinding)
      continue;
    attemptedDecoratedDefs.insert(statement.get());
    mlir::Type decoratedType = types.widenLiteral(types.inferExpr(applied.get()));
    if (!mlir::isa_and_nonnull<py::CallableType>(decoratedType))
      continue;
    rebindingDecoratedDefs.insert(statement.get());
    decoratorApplications.push_back(applied);
    moduleGlobals[*decorated] = decoratedType;
    types.bindSymbol(*decorated, decoratedType);
  }
  // And a decorated METHOD's cell (desugarMethodDecorators): typed by its
  // application, like a decorated function's -- here when its decorators'
  // types are known already, else when its class is emitted.
  // An imported module's are typed in its own scope, where its decorators
  // are named (emitClassContract).
  for (DecoratedMethod &method : decoratedMethods)
    if (!method.source)
      typeDecoratedMethod(method, /*reportFailure=*/false);
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
    // A class value's cell holds its type object's reference, as a field of
    // that type does.
    bool storageBacked =
        mlir::isa<py::ContractType>(annotated) ||
        mlir::isa<py::CallableType>(annotated) ||
        (mlir::isa<py::TypeType>(annotated) &&
         mlir::isa<py::ContractType>(
             mlir::cast<py::TypeType>(annotated).getInstanceType()));
    // ⭐ A UNION GETS A CELL WHEN A FUNCTION NEEDS ONE -- reads the name or
    // declares it `global`. Without it `_cache: dict | None = None` read from
    // any function was "unresolved name '_cache'", and `global _cache;
    // _cache = {...}` was refused as having nothing to write to: the lazy
    // module cache, written exactly as CPython code writes it.
    //
    // ⛔ Not every union, which is what the first attempt did: a cell's read
    // hands back a union whose tag is a runtime fact, and module code that
    // relied on the value binding's STATIC member then changed meaning --
    // `first: "ValueError | KeyError" = ValueError(m); raise first` is a plain
    // raise while value-bound and a union raise through a cell, which is
    // refused (tests/probe/wb_raise_a_runtime_chosen_exception.py). A name
    // only the module body touches keeps that evidence.
    if (!storageBacked && mlir::isa<py::UnionType>(annotated)) {
      if (!usedInsideFunctions)
        usedInsideFunctions = namesUsedInsideFunctions(moduleNode);
      storageBacked = usedInsideFunctions->contains(ast::nameSpelling(*target));
    }
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
    // slot's payload class word, which would also close the second-read case
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
    // ⭐ `None` IS ONE OF THEM. It was excluded here with no stated reason,
    // so `V = None` read from a function was "unresolved name 'V'" while
    // `V = 3` beside it worked -- one channel with an arm missing, which is
    // the same shape the imported-module literal channel had for None. The
    // re-emission this channel does is `py.none`, which costs nothing and
    // cannot go stale: a name bound once to None is None everywhere.
    if (!value || value->kind != "Constant")
      continue;
    llvm::StringRef name = ast::nameSpelling(*targets->front());
    if (!boundOnce.contains(name) || moduleGlobals.count(name) ||
        declaredGlobal.contains(name))
      continue;
    moduleConstantBindings[name] = value;
  }

  // ⭐ AND THE TYPE SYSTEM HAS TO KNOW THEM. A reference to one of these names
  // re-emits the literal, so every USE site was right; but nothing bound the
  // name, so any question asked ABOUT it away from a use site went unanswered:
  //
  //     WIDTH = 80
  //     class C:
  //         half = WIDTH // 2
  //     print(C.half)      # unsupported static class attribute expression
  //
  // A class attribute is TYPED before any body is emitted, by `inferExpr`
  // alone, and an attribute whose type is unknown falls off the slot channel
  // onto the constant channel -- which has no arm for an expression and says
  // so from the lowering. The annotated spelling `WIDTH: int = 80` already
  // worked here, and CPython does not distinguish the two.
  //
  // ⛔ The LITERAL type, not the widened one: the binding has to describe what
  // `emitConstant` will produce at the use site, and a widened `int` would
  // make the two disagree wherever the literal's precision is what decides.
  //
  // ⛔ And only where nothing is bound already -- a name this pass reached is
  // bound once by construction, but a builtin or an import may own the
  // spelling, and this is a fallback for inference, not a rebinding.
  for (const auto &entry : moduleConstantBindings)
    if (!types.lookupSymbol(entry.getKey()))
      if (mlir::Type literal = types.inferExpr(entry.second))
        types.bindSymbol(entry.getKey(), literal);

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

  // ⭐ A CELL'S VALUE IS READ WITH THE NAMES BOUND ABOVE IT, as the module
  // body runs. A value-bound name is in no table this inference reads, so a
  // cell computed from one was `object`:
  //
  //     D = {1: 5, 4: 7}
  //     U = D[4]
  //     def f() -> int:
  //         return U + 1   # object does not provide manifest method '__add__'
  //
  // ⛔ Not bound into the type system: nothing outside the module body can
  // read a value-bound name, and binding it would say otherwise.
  llvm::StringMap<mlir::Type> boundAbove;
  const llvm::StringMap<mlir::Type> noCallables;
  auto inferBelow = [&](const parser::Node *value) {
    if (boundAbove.empty())
      return types.widenLiteral(types.inferExpr(value));
    return types.widenLiteral(types.inferExpr(
        value, ExprInferenceContext{noCallables, nullptr, &boundAbove,
                                    /*strict=*/false}));
  };
  for (const parser::NodePtr &statement : *body) {
    if (!statement || statement->kind != "Assign")
      continue;
    const auto *targets = ast::nodeList(*statement, "targets");
    if (!targets || targets->size() != 1 || !targets->front() ||
        targets->front()->kind != "Name")
      continue;
    llvm::StringRef name = ast::nameSpelling(*targets->front());
    const parser::Node *value = ast::node(*statement, "value");
    if (!boundOnce.contains(name) || moduleGlobals.count(name) ||
        moduleConstantBindings.count(name) ||
        !readFromAFunction.contains(name)) {
      if (value && !types.lookupSymbol(name))
        boundAbove[name] = inferBelow(value);
      continue;
    }
    if (!value)
      continue;
    mlir::Type inferred = inferBelow(value);
    // ⭐ AN EMPTY CONTAINER GLOBAL TAKES ITS ELEMENT FROM THE FUNCTIONS THAT
    // FILL IT. The cell's type is decided HERE, before any body is emitted, so
    // seeding it at the assignment (where a local is seeded) is too late --
    // the reads have already been typed against the erased element:
    //
    //     XS = []
    //     def put(n: int) -> None:
    //         XS.append(n)
    //     put(1)
    //     print(XS[0] + 1)
    //     # builtins.object does not provide manifest method '__add__'
    //
    // A global read from a function is exactly the shape this filter selects
    // for, so every one of them is a candidate.
    if (isEmptyContainerExpression(value)) {
      llvm::StringRef literalKind = value->kind;
      if (literalKind == "Call") {
        llvm::StringRef callee = ast::nameSpelling(*ast::node(*value, "func"));
        literalKind = callee == "dict"    ? "Dict"
                      : callee == "set"   ? "Set"
                      : callee == "tuple" ? "Tuple"
                                          : "List";
      }
      SuiteCursor cursor{body, 0};
      llvm::SmallVector<const parser::Node *, 4> callables;
      for (const parser::NodePtr &candidate : *body)
        if (candidate && (candidate->kind == "FunctionDef" ||
                          candidate->kind == "AsyncFunctionDef"))
          callables.push_back(candidate.get());
      if (mlir::Type seededGlobal =
              emptyLiteralSeedAcross(name, literalKind, cursor, callables)) {
        inferred = seededGlobal;
        // ⛔ AND THE SIGNATURES THAT WERE READ BEFORE THE ANSWER EXISTED. Every
        // top-level signature was memoized by `registerModule`, whose sweeps
        // saw this global at its ERASED element -- so a body that measures it
        // (`len(SAMPLES)`) recorded a failure there, and the memo would report
        // it at emission for a program that is now fine. Forgotten here, they
        // are recomputed where they are declared, which is what the generator
        // arm of `emitTopLevelDeclarations` already does for the same reason.
        for (const parser::Node *callable : callables) {
          types.forgetSignature(callable);
          types.forgetInferredResult(callable);
        }
      }
    }
    // ⭐ A FUNCTION VALUE IS A GLOBAL LIKE ANY OTHER. `CALLBACK = base` read
    // from a function body was "unresolved name 'CALLBACK'": the cell was
    // never declared, because a callable's static type is a `py.callable` and
    // this filter took contracts only. The VALUE is an ordinary object
    // (`builtins.function`, a header the cell boxes like a list or a class
    // instance); it is only the static type that is spelled differently, and
    // keeping that spelling is what lets the call through the global check its
    // arguments.
    // ⭐ AND A UNION, as an annotated one gets a cell when a function reads
    // it: every name here is one. `U = D[4]` over `{1: 5, 4: True}` is
    // `bool | int`, and without a cell `return U` was "unresolved name 'U'".
    if (!mlir::isa_and_nonnull<py::ContractType>(inferred) &&
        !mlir::isa_and_nonnull<py::CallableType>(inferred) &&
        !mlir::isa_and_nonnull<py::UnionType>(inferred))
      continue;
    moduleGlobals[name] = inferred;
    types.bindSymbol(name, inferred);
  }

  // ⭐ AND A NAME BOUND MORE THAN ONCE, WHICH A FUNCTION READS, gets a cell of
  // every value it is bound to -- their join, a union when they differ, which
  // a cell holds as an annotated `int | str` one does. `n = 1` followed by
  // `for i in ...: n = n + i`, a `tag = "g"` rebound under a module-level
  // `if`, `x = 1` then `x = 2`: each was "unresolved name" in every function
  // that read it, though CPython runs them all. The "bound once" rule above
  // stays the rule for a name no function reads, for the evidence reason it
  // gives.
  // ⛔ Only plain `NAME = value`, an augmented assignment and a `for NAME`
  // loop: a binding that unpacks, imports, catches or opens a context gives
  // no single value to type here, and such a name keeps its value binding.
  llvm::StringMap<llvm::SmallVector<mlir::Type, 4>> rebound;
  llvm::StringMap<unsigned> bindings;
  llvm::StringSet<> unpackable;
  auto candidate = [&](llvm::StringRef name) {
    return !boundOnce.contains(name) && !moduleGlobals.count(name) &&
           !moduleConstantBindings.count(name) &&
           readFromAFunction.contains(name);
  };
  auto bind = [&](llvm::StringRef name, mlir::Type type) {
    if (!candidate(name))
      return;
    ++bindings[name];
    rebound[name].push_back(type);
    if (!types.lookupSymbol(name))
      boundAbove[name] = type;
  };
  std::function<void(const parser::Node *)> disqualify =
      [&](const parser::Node *target) {
        if (!target)
          return;
        if (target->kind == "Name") {
          unpackable.insert(ast::nameSpelling(*target));
          return;
        }
        if (target->kind == "Starred") {
          disqualify(ast::node(*target, "value"));
          return;
        }
        if (const auto *elements = ast::nodeList(*target, "elts"))
          for (const parser::NodePtr &element : *elements)
            disqualify(element.get());
      };
  std::function<void(const std::vector<parser::NodePtr> *)> walk =
      [&](const std::vector<parser::NodePtr> *statements) {
        if (!statements)
          return;
        for (const parser::NodePtr &statement : *statements) {
          if (!statement)
            continue;
          llvm::StringRef kind = statement->kind;
          if (kind == "Assign") {
            const auto *targets = ast::nodeList(*statement, "targets");
            const parser::Node *value = ast::node(*statement, "value");
            if (targets && targets->size() == 1 && targets->front() &&
                targets->front()->kind == "Name" && value) {
              bind(ast::nameSpelling(*targets->front()), inferBelow(value));
            } else if (targets) {
              for (const parser::NodePtr &target : *targets)
                disqualify(target.get());
            }
          } else if (kind == "AugAssign") {
            // The same value kind as the name holds: counted, not typed.
            const parser::Node *target = ast::node(*statement, "target");
            if (!target || target->kind != "Name")
              disqualify(target);
            else if (candidate(ast::nameSpelling(*target)))
              ++bindings[ast::nameSpelling(*target)];
          } else if (kind == "AnnAssign") {
            disqualify(ast::node(*statement, "target"));
          } else if (kind == "For" || kind == "AsyncFor") {
            const parser::Node *target = ast::node(*statement, "target");
            if (target && target->kind == "Name") {
              mlir::Type element =
                  types.iterationElementType(ast::node(*statement, "iter"));
              bind(ast::nameSpelling(*target),
                   element ? types.widenLiteral(element) : types.object());
            } else {
              disqualify(target);
            }
          } else if (kind == "With" || kind == "AsyncWith") {
            if (const auto *items = ast::nodeList(*statement, "items"))
              for (const parser::NodePtr &item : *items)
                if (item)
                  disqualify(ast::node(*item, "optional_vars"));
          } else if (kind == "Try" || kind == "TryStar") {
            if (const auto *handlers = ast::nodeList(*statement, "handlers"))
              for (const parser::NodePtr &handler : *handlers)
                if (handler) {
                  if (auto caught = ast::string(*handler, "name"))
                    unpackable.insert(*caught);
                  walk(ast::nodeList(*handler, "body"));
                }
            walk(ast::nodeList(*statement, "finalbody"));
          } else if (kind == "Import" || kind == "ImportFrom" ||
                     kind == "FunctionDef" || kind == "AsyncFunctionDef" ||
                     kind == "ClassDef") {
            if (auto defined = ast::string(*statement, "name"))
              unpackable.insert(*defined);
            if (const auto *aliases = ast::nodeList(*statement, "names"))
              for (const parser::NodePtr &alias : *aliases) {
                if (!alias)
                  continue;
                std::optional<std::string_view> spelled =
                    ast::string(*alias, "asname");
                if (!spelled)
                  spelled = ast::string(*alias, "name");
                if (spelled)
                  unpackable.insert(
                      llvm::StringRef(*spelled).split('.').first);
              }
            continue;
          }
          if (kind == "FunctionDef" || kind == "AsyncFunctionDef" ||
              kind == "ClassDef")
            continue;
          walk(ast::nodeList(*statement, "body"));
          walk(ast::nodeList(*statement, "orelse"));
        }
      };
  walk(body);
  for (const auto &entry : rebound) {
    llvm::StringRef name = entry.getKey();
    if (unpackable.contains(name) || bindings.lookup(name) < 2)
      continue;
    mlir::Type joined = types.join(entry.getValue());
    if (!mlir::isa_and_nonnull<py::ContractType>(joined) &&
        !mlir::isa_and_nonnull<py::CallableType>(joined) &&
        !mlir::isa_and_nonnull<py::UnionType>(joined))
      continue;
    if (joined == types.object())
      continue;
    moduleGlobals[name] = joined;
    types.bindSymbol(name, joined);
  }
}

llvm::StringSet<>
ModuleEmitter::unboundModuleTargets(const parser::Node &moduleNode) const {
  llvm::StringSet<> unbound;
  const auto *body = ast::nodeList(moduleNode, "body");
  if (!body)
    return unbound;
  for (const parser::NodePtr &statement : *body) {
    if (!statement)
      continue;
    const parser::Node *target = nullptr;
    if (statement->kind == "AnnAssign") {
      target = ast::node(*statement, "target");
    } else if (statement->kind == "Assign") {
      const auto *targets = ast::nodeList(*statement, "targets");
      if (targets && targets->size() == 1)
        target = targets->front().get();
    }
    if (target && target->kind == "Name") {
      llvm::StringRef name = ast::nameSpelling(*target);
      if (!types.lookupSymbol(name))
        unbound.insert(name);
    }
  }
  return unbound;
}

// ⭐ A SIGNATURE READ BEFORE THE GLOBALS WERE BOUND IS READ AGAIN. Every
// top-level signature is memoized by `registerModule`, which has to run before
// this pass -- a global's value may call a function -- so a body that returns
// a global saw an unbound name:
//
//     B = True
//     def f() -> int:
//         return B
//     print(f())      # "type of return operand 0 ('!py.literal<True>')
//                     #  doesn't match function result type"
//
// The return walk keeps a lower rung (`belowDeclaredRung`) only when it can
// see it, and the body then returned the bool its signature said was an int.
// Forgotten, a signature is recomputed where it is declared, with the globals
// bound -- as the empty-container globals above and the generators in
// `emitTopLevelDeclarations` already are. And so is every function that reads
// a forgotten one: its own walk read the stale result. ⛔ Not every signature:
// one that reads no late global was read under the facts it has now.
void ModuleEmitter::forgetSignaturesReadingLateGlobals(
    const parser::Node &moduleNode, const llvm::StringSet<> &unbound) {
  llvm::StringSet<> late;
  for (const auto &entry : unbound)
    if (types.lookupSymbol(entry.getKey()))
      late.insert(entry.getKey());
  const auto *body = ast::nodeList(moduleNode, "body");
  if (late.empty() || !body)
    return;
  struct Reader {
    const parser::Node *function;
    llvm::StringRef name;
    llvm::StringSet<> names;
  };
  std::vector<Reader> readers;
  for (const parser::NodePtr &statement : *body) {
    if (!statement || (statement->kind != "FunctionDef" &&
                       statement->kind != "AsyncFunctionDef"))
      continue;
    Reader reader{statement.get(), ast::nameSpelling(*statement), {}};
    ast::walk(statement.get(), [&](const parser::Node &node) {
      if (node.kind == "Name")
        reader.names.insert(ast::nameSpelling(node));
      return ast::Walk::Continue;
    });
    readers.push_back(std::move(reader));
  }
  llvm::SmallPtrSet<const parser::Node *, 8> forgotten;
  for (bool changed = true; changed;) {
    changed = false;
    for (const Reader &reader : readers) {
      if (forgotten.contains(reader.function) ||
          llvm::none_of(reader.names, [&](const auto &entry) {
            return late.contains(entry.getKey());
          }))
        continue;
      types.forgetSignature(reader.function);
      types.forgetInferredResult(reader.function);
      forgotten.insert(reader.function);
      late.insert(reader.name);
      changed = true;
    }
  }
  // ...AND THEIR SYMBOLS ARE BOUND AGAIN NOW, until they hold still: a
  // function declared above the one it calls reads the callee's symbol, which
  // still held the stale answer (`def h() -> int: return k()` above
  // `def k() -> int: return B`). ⛔ Not memoized: the declaration reads each
  // one a last time, with the classes emitted. ⛔ Not a decorated name, which
  // this pass bound to what the decorator returns.
  for (std::size_t round = 0; round <= forgotten.size(); ++round) {
    bool moved = false;
    for (const Reader &reader : readers) {
      const auto *decorators =
          ast::nodeList(*reader.function, "decorator_list");
      if (!forgotten.contains(reader.function) ||
          (decorators && !decorators->empty()))
        continue;
      mlir::Type callable =
          types.functionSignature(*reader.function).publicCallable;
      std::optional<mlir::Type> bound = types.lookupSymbol(reader.name);
      if (!callable || (bound && *bound == callable))
        continue;
      types.bindSymbol(reader.name, callable);
      moved = true;
    }
    if (!moved)
      break;
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

// ⭐ A STREAM THE PROGRAM REPLACES IS A CELL, and one that it never replaces
// stays the runtime's own stream. `sys.stdout = buf` is how output is
// captured -- unittest's buffered result, `contextlib.redirect_stdout` -- and
// CPython's print writes to whatever `sys.stdout` holds when it runs, so a
// program that assigns it needs every read to see the assignment. One that
// never does keeps the direct write it has always had: the question is asked
// of the whole program, syntactically, before anything is bound.
//
// ⛔ BY THE SPELLING `sys.stdout = ...` (any alias of `import sys`), in any
// module the program compiles. A replacement the scan cannot see -- through
// `setattr`, or a name it was handed -- reaches the assignment's ordinary
// refusal, so it is loud rather than unseen.
void ModuleEmitter::collectReplacedStreams() {
  llvm::SmallVector<std::string, 2> found;
  auto scan = [&](const parser::Node &moduleNode) {
    llvm::StringSet<> sysNames;
    ast::walk(&moduleNode, [&](const parser::Node &node) {
      if (node.kind == "Import")
        if (const auto *names = ast::nodeList(node, "names"))
          for (const parser::NodePtr &alias : *names)
            if (alias && ast::string(*alias, "name") == "sys")
              sysNames.insert(
                  std::string(ast::string(*alias, "asname").value_or("sys")));
      return ast::Walk::Continue;
    });
    if (sysNames.empty())
      return;
    std::function<void(const parser::Node *)> target =
        [&](const parser::Node *node) {
          if (!node)
            return;
          if (node->kind == "Tuple" || node->kind == "List") {
            if (const auto *elements = ast::nodeList(*node, "elts"))
              for (const parser::NodePtr &element : *elements)
                target(element.get());
            return;
          }
          if (node->kind == "Starred") {
            target(ast::node(*node, "value"));
            return;
          }
          if (node->kind != "Attribute")
            return;
          const parser::Node *owner = ast::node(*node, "value");
          std::optional<std::string_view> attr = ast::string(*node, "attr");
          if (!owner || owner->kind != "Name" || !attr ||
              (*attr != "stdout" && *attr != "stderr") ||
              !sysNames.contains(ast::nameSpelling(*owner)))
            return;
          std::string canonical = "sys." + std::string(*attr);
          if (!llvm::is_contained(found, canonical))
            found.push_back(canonical);
        };
    ast::walk(&moduleNode, [&](const parser::Node &node) {
      if (node.kind == "Assign") {
        if (const auto *targets = ast::nodeList(node, "targets"))
          for (const parser::NodePtr &assigned : *targets)
            target(assigned.get());
      } else if (node.kind == "AnnAssign" || node.kind == "AugAssign") {
        target(ast::node(node, "target"));
      }
      return ast::Walk::Continue;
    });
  };
  for (const EmitOptions::SourceModule &source : options.sourceModules)
    if (source.moduleNode && !source.isStub)
      scan(*source.moduleNode);
  scan(moduleNode);
  llvm::sort(found);
  replacedStreams.assign(found.begin(), found.end());
  types.setReplacedStreams(found);
}

// Each replaced stream starts as the runtime's own, before any module's code
// can read it.
void ModuleEmitter::emitReplacedStreamInitializers(
    const parser::Node &anchor) {
  mlir::Type wrapper = types.contract("_io.TextIOWrapper");
  for (const std::string &stream : replacedStreams) {
    // ⛔ The op itself and not `emitBindingRef`, which reads a replaced
    // stream's CELL -- the one this fills.
    auto own = py::BindingRefOp::create(builder, loc(anchor), wrapper,
                                        builder.getStringAttr(stream),
                                        mlir::ValueRange{});
    Value initial = coerceValue(Value{own.getResult(), wrapper},
                                types.replacedStreamType(), anchor);
    auto store = py::GlobalSetOp::create(builder, loc(anchor),
                                         builder.getStringAttr(stream),
                                         initial.value);
    markBoxedModuleGlobal(store);
  }
}

std::optional<std::string>
ModuleEmitter::replacedStreamTarget(const parser::Node &target) const {
  if (target.kind != "Attribute" || replacedStreams.empty())
    return std::nullopt;
  std::optional<std::string> canonical =
      types.lookupCanonicalBinding(ast::qualifiedName(&target));
  if (!canonical || !llvm::is_contained(replacedStreams, *canonical))
    return std::nullopt;
  return canonical;
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
  // ⛔ In the CELL's type, not the receiver's: a union global's receiver is
  // the member a guard proved, and storing that bare put a list's lanes where
  // the cell keeps a tag -- `if buf is not None: buf.append(s)` segfaulted.
  Value stored = coerceValue(rebound, moduleGlobals.lookup(name), at);
  auto op = py::GlobalSetOp::create(builder, loc(at),
                                    builder.getStringAttr(name), stored.value);
  markBoxedModuleGlobal(op);
}


// ⛔ Typed through the method's own signature, not `C.__ly_undecorated$m`:
// the class's methods are not in its table until it is emitted, so the
// attribute read answered `object` and an identity decorator
// (`def mark(f: F) -> F`) handed that back as the method.
// ⛔ And asked twice: a decorator that is an INSTANCE (`@skip("why")` with a
// `__call__`) has no method table either until its class is emitted, which
// it is before the class it decorates -- so a miss here is final only there.
bool ModuleEmitter::typeDecoratedMethod(DecoratedMethod &method,
                                        bool reportFailure) {
  if (method.typed)
    return true;
  mlir::Type decoratedType;
  mlir::Type undecoratedType;
  if (std::optional<mlir::Type> classType =
          types.lookupClass(method.contractName);
      classType && method.hiddenDef) {
    std::optional<llvm::StringRef> selfName;
    if (const parser::Node *arguments = ast::node(*method.hiddenDef, "args"))
      for (llvm::StringRef group : {"posonlyargs", "args"})
        if (const auto *list = ast::nodeList(*arguments, group);
            !selfName && list && !list->empty() && list->front())
          selfName = ast::string(*list->front(), "arg");
    FunctionSignature sig = types.functionSignature(
        *method.hiddenDef, selfName, py::CallableType(), *classType);
    undecoratedType = sig.publicCallable;
    std::string subject = method.cellName + "$method";
    types.bindSymbol(subject, sig.publicCallable);
    parser::NodePtr applied = decoratedMethodApplication(
        method, synth::name(subject, method.range));
    synthesizedIteratorDefs.push_back(applied);
    decoratedType = types.widenLiteral(types.inferExpr(applied.get()));
  }
  // ⛔ A default the def declares is the method's only while what the
  // decorator returned still has it: the forwarder restates the def's
  // signature, and over a wrapper `(x, n)` it answered `m()` with the def's
  // default where CPython raises TypeError. A decorator that hands the
  // function back (`F -> F`) keeps the def's own type, defaults and all.
  std::string droppedDefaults;
  if (auto decorated = mlir::dyn_cast_if_present<py::CallableType>(decoratedType))
    if (const parser::Node *arguments =
            method.hiddenDef ? ast::node(*method.hiddenDef, "args") : nullptr)
      if (const auto *defaults = ast::nodeList(*arguments, "defaults");
          defaults && !defaults->empty()) {
        llvm::ArrayRef<mlir::BoolAttr> kept = decorated.getPositionalDefaults();
        std::size_t count = decorated.getPositionalTypes().size();
        bool keeps = kept.size() == count && count >= defaults->size();
        for (std::size_t index = count - std::min(count, defaults->size());
             keeps && index < count; ++index)
          keeps = kept[index] && kept[index].getValue();
        if (!keeps)
          droppedDefaults = "' declares default values, and what the "
                            "decorator returns does not take them: the method "
                            "is what the decorator returns";
      }
  // ⛔ Nor one whose result takes other parameters or returns another type:
  // the forwarder restates the def's signature, so `@as_text` turning
  // `-> int` into `-> str` read as "annotated to return int but this return
  // gives str", a sentence about a line the program did not write wrong.
  // Supporting it is the forwarder taking the result's signature instead.
  if (droppedDefaults.empty())
    if (auto decorated =
            mlir::dyn_cast_if_present<py::CallableType>(decoratedType))
      if (auto undecorated =
              mlir::dyn_cast_if_present<py::CallableType>(undecoratedType)) {
        auto same = [&](mlir::ArrayRef<mlir::Type> lhs,
                        mlir::ArrayRef<mlir::Type> rhs) {
          if (lhs.size() != rhs.size())
            return false;
          for (auto [left, right] : llvm::zip(lhs, rhs))
            if (types.widenLiteral(left) != types.widenLiteral(right))
              return false;
          return true;
        };
        auto spell = [&](py::CallableType callable) {
          std::string text = "(";
          for (auto [index, parameter] :
               llvm::enumerate(callable.getPositionalTypes()))
            text += (index ? ", " : "") + pythonTypeSpelling(parameter);
          text += ") -> ";
          text += callable.getResultTypes().empty()
                      ? std::string("None")
                      : pythonTypeSpelling(callable.getResultTypes().front());
          return text;
        };
        if (!same(decorated.getPositionalTypes(),
                  undecorated.getPositionalTypes()) ||
            !same(decorated.getResultTypes(), undecorated.getResultTypes()))
          droppedDefaults = "' is " + spell(undecorated) +
                            " and its decorator makes it " + spell(decorated) +
                            ": a decorated method keeps the signature its def "
                            "declares here";
      }
  if (!droppedDefaults.empty()) {
    if (!reportFailure)
      return false;
    const parser::Node *decorator =
        method.decorators.empty() ? nullptr : method.decorators.front().get();
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error,
        decorator ? decorator->range.start : method.range.start,
        "method '" +
            llvm::StringRef(method.hiddenName)
                .drop_front(llvm::StringRef("__ly_undecorated$").size())
                .str() +
            droppedDefaults});
    if (undecoratedType)
      bindDecoratedMethodCell(method, undecoratedType);
    return false;
  }
  if (!mlir::isa_and_nonnull<py::CallableType>(decoratedType)) {
    if (!reportFailure)
      return false;
    const parser::Node *decorator =
        method.decorators.empty() ? nullptr : method.decorators.front().get();
    const parser::Node *callee =
        decorator && decorator->kind == "Call" ? ast::node(*decorator, "func")
                                               : decorator;
    std::string spelling = callee ? ast::qualifiedName(callee) : "";
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error,
        decorator ? decorator->range.start : method.range.start,
        "decorator '" + spelling + "' on method '" +
            llvm::StringRef(method.hiddenName)
                .drop_front(llvm::StringRef("__ly_undecorated$").size())
                .str() +
            "' does not return a function here: the method is bound to "
            "what its decorators return, and only a callable can be"});
    // The forwarder still names the cell; typed as the method itself, it
    // adds no second sentence to the one above.
    if (undecoratedType)
      bindDecoratedMethodCell(method, undecoratedType);
    return false;
  }
  method.typed = true;
  bindDecoratedMethodCell(method, decoratedType);
  return true;
}

void ModuleEmitter::bindDecoratedMethodCell(const DecoratedMethod &method,
                                            mlir::Type type) {
  moduleGlobals[method.globalName] = type;
  if (method.source)
    types.bindCanonicalSymbol(method.cellName, method.globalName, type);
  else
    types.bindSymbol(method.cellName, type);
}

} // namespace lython::emitter
