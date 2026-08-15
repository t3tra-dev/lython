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
  llvm::StringSet<> boundOnce = singleAssignmentNames(moduleNode);
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
    if (!boundOnce.contains(name) || moduleGlobals.count(name))
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
    if (!mlir::isa_and_nonnull<py::ContractType>(inferred))
      continue;
    moduleGlobals[name] = inferred;
    types.bindSymbol(name, inferred);
  }
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
