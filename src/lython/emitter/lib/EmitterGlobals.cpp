// MODULE GLOBALS: which module-level names get storage, and what kind.
//
// Three answers, and the annotation is what picks between them (the lowering
// side is Passes/Runtime/Ops/GlobalOps.cpp):
//
//   an ADDRESS (`ctypes.c_void_p`) ... a native machine-word cell, which a
//       signal handler may read because touching it never allocates
//   a storage-backed object ......... an immutable scalar, bytes, or a user
//       class: a cell holding the value group, rebindable from a function
//       that declared `global NAME`
//   value-bound ..................... everything else, including every
//       container: a structural mutation reallocates the interior arrays
//       through SSA rebinding, which a cell would go stale against
//
// Why NOT in EmitterClasses.cpp, where this lived: nothing here is about a
// class. It shared the file only because `collectStaticModuleAssignments`
// (still there) feeds the same static-attribute table the class bodies do.

#include "EmitterCore.h"

#include "AstAccess.h"
#include "ClosureAnalysis.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/StringSet.h"

namespace lython::emitter {

void ModuleEmitter::collectModuleGlobals(const parser::Node &moduleNode) {
  // Opt-in: an annotated module-level assignment (`NAME: T = ...`) becomes
  // a storage-backed mutable global (int keeps its unboxed i64 cell for the
  // signal-safe channel; other contracts store their physical value words).
  // Plain `NAME = expr` at module scope keeps its value-binding behavior
  // (module-scope constants).
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
    // Storage-backed globals cover the immutable scalars plus user classes
    // (whose mutation happens in place on the heap). Containers stay
    // value-bound: their structural mutations reallocate the interior
    // arrays through SSA rebinding, which a storage cell would go stale
    // against; unions stay value-bound so isinstance narrowing keeps
    // working on the module flow.
    bool storageBacked =
        annotated == types.intType() || annotated == types.strType() ||
        annotated == types.floatType() || annotated == types.boolType();
    if (!storageBacked) {
      if (auto contract = mlir::dyn_cast<py::ContractType>(annotated)) {
        llvm::StringRef contractName = contract.getContractName();
        // `ctypes.c_void_p` is the ADDRESS spelling: its cell is the machine
        // word, which is what a signal handler may read and what an `int`
        // global deliberately no longer is (see lowerGlobalGet).
        storageBacked = contractName == "builtins.bytes" ||
                        contractName == "ctypes.c_void_p" ||
                        !contractName.contains('.');
      }
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
  // ⛔ Why NOT make it a storage cell like the annotated form: an int cell is
  // an UNBOXED i64 (the async-signal-safe channel), so a module `fact = 1`
  // grown past 2**63 by module-scope arithmetic would raise
  // "int too large to convert to a native 64-bit integer" where it prints
  // the value today. Measured: 4 goldens, one of them exactly that
  // factorial. A literal bound once has no such arithmetic on it.
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

} // namespace lython::emitter
