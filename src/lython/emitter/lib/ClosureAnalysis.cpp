#include "ClosureAnalysis.h"

#include "AstAccess.h"
#include "EmitterSupport.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"

namespace lython::emitter {
namespace {

bool hasContext(const parser::Node &node, llvm::StringRef expected) {
  const parser::Node *ctx = ast::node(node, "ctx");
  return ctx && ctx->kind == expected;
}

void collectParameterNames(const parser::Node *arguments,
                           llvm::StringSet<> &names) {
  if (!arguments)
    return;
  if (const auto *posOnly = ast::nodeList(*arguments, "posonlyargs"))
    for (const parser::NodePtr &arg : *posOnly)
      names.insert(ast::nameSpelling(*arg));
  if (const auto *args = ast::nodeList(*arguments, "args"))
    for (const parser::NodePtr &arg : *args)
      names.insert(ast::nameSpelling(*arg));
  if (const auto *kwonly = ast::nodeList(*arguments, "kwonlyargs"))
    for (const parser::NodePtr &arg : *kwonly)
      names.insert(ast::nameSpelling(*arg));
  if (const parser::Node *vararg = ast::node(*arguments, "vararg"))
    names.insert(ast::nameSpelling(*vararg));
  if (const parser::Node *kwarg = ast::node(*arguments, "kwarg"))
    names.insert(ast::nameSpelling(*kwarg));
}

// A nested `def`/`class` binds its own name in the enclosing function.
void collectLocalNames(const parser::Node *node, llvm::StringSet<> &names) {
  collectNameBindings(node, names, /*bindsNestedDefinitions=*/true);
}

void collectReadNames(const parser::Node *node, llvm::StringSet<> &names) {
  if (!node)
    return;
  if (node->kind == "Name") {
    if (!hasContext(*node, "Store") && !hasContext(*node, "Del"))
      names.insert(ast::nameSpelling(*node));
    return;
  }
  if (node->kind == "FunctionDef" || node->kind == "AsyncFunctionDef") {
    collectReadNames(ast::node(*node, "args"), names);
    collectReadNames(ast::node(*node, "returns"), names);
    if (const auto *decorators = ast::nodeList(*node, "decorator_list"))
      for (const parser::NodePtr &decorator : *decorators)
        collectReadNames(decorator.get(), names);
    if (const auto *body = ast::nodeList(*node, "body"))
      for (const parser::NodePtr &statement : *body)
        collectReadNames(statement.get(), names);
    return;
  }
  for (const parser::Field &field : node->fields) {
    if (field.name == "ctx")
      continue;
    if (const auto *child = std::get_if<parser::NodePtr>(&field.value)) {
      if (*child)
        collectReadNames(child->get(), names);
    } else if (const auto *children =
                   std::get_if<std::vector<parser::NodePtr>>(&field.value)) {
      for (const parser::NodePtr &child : *children)
        if (child)
          collectReadNames(child.get(), names);
    }
  }
}

// `nonlocal NAME` declarations at THIS function's statement level (if/while
// bodies included, nested def/lambda/class scopes excluded).
void collectOwnNonlocalNames(const parser::Node *node,
                             llvm::StringSet<> &names) {
  if (!node)
    return;
  if (node->kind == "FunctionDef" || node->kind == "AsyncFunctionDef" ||
      node->kind == "ClassDef" || node->kind == "Lambda")
    return;
  if (node->kind == "Nonlocal") {
    if (const auto *declared = ast::stringList(*node, "names"))
      for (const std::string &name : *declared)
        names.insert(name);
    return;
  }
  for (const parser::Field &field : node->fields) {
    if (const auto *child = std::get_if<parser::NodePtr>(&field.value)) {
      if (*child)
        collectOwnNonlocalNames(child->get(), names);
    } else if (const auto *children =
                   std::get_if<std::vector<parser::NodePtr>>(&field.value)) {
      for (const parser::NodePtr &child : *children)
        if (child)
          collectOwnNonlocalNames(child.get(), names);
    }
  }
}

// Function definitions nested directly in this scope (not crossing into
// deeper function scopes; class bodies are traversed because their methods
// close over the enclosing function scope).
void collectDirectNestedFunctions(
    const parser::Node *node,
    llvm::SmallVectorImpl<const parser::Node *> &nested) {
  if (!node)
    return;
  if (node->kind == "FunctionDef" || node->kind == "AsyncFunctionDef") {
    nested.push_back(node);
    return;
  }
  if (node->kind == "Lambda")
    return; // no statements, so no nonlocal declarations
  for (const parser::Field &field : node->fields) {
    if (const auto *child = std::get_if<parser::NodePtr>(&field.value)) {
      if (*child)
        collectDirectNestedFunctions(child->get(), nested);
    } else if (const auto *children =
                   std::get_if<std::vector<parser::NodePtr>>(&field.value)) {
      for (const parser::NodePtr &child : *children)
        if (child)
          collectDirectNestedFunctions(child.get(), nested);
    }
  }
}

// ⭐ Collected SEPARATELY from the nested defs above, whose list exists for
// `nonlocal` declarations -- which a lambda, having no statements, cannot
// make. That list is also the readers list for the boxing decision below, and
// there a lambda counts exactly as much as a def:
//
//     def run() -> None:
//         x = 1
//         f = lambda: x
//         x = 2
//         print(f())      # printed 1; CPython prints 2
//
// A def nested inside is skipped: what it reads is already collected as its
// own capture set, and a lambda inside THAT belongs to its scope, not this one.
void collectDirectNestedLambdas(
    const parser::Node *node,
    llvm::SmallVectorImpl<const parser::Node *> &nested) {
  if (!node)
    return;
  if (node->kind == "FunctionDef" || node->kind == "AsyncFunctionDef" ||
      node->kind == "ClassDef")
    return;
  if (node->kind == "Lambda") {
    nested.push_back(node);
    return;
  }
  for (const parser::Field &field : node->fields) {
    if (const auto *child = std::get_if<parser::NodePtr>(&field.value)) {
      if (*child)
        collectDirectNestedLambdas(child->get(), nested);
    } else if (const auto *children =
                   std::get_if<std::vector<parser::NodePtr>>(&field.value)) {
      for (const parser::NodePtr &child : *children)
        if (child)
          collectDirectNestedLambdas(child.get(), nested);
    }
  }
}

void collectFunctionLocalNames(const parser::Node &callable,
                               llvm::StringSet<> &locals) {
  collectParameterNames(ast::node(callable, "args"), locals);
  if (const auto *body = ast::nodeList(callable, "body"))
    for (const parser::NodePtr &statement : *body)
      collectLocalNames(statement.get(), locals);
  if (const parser::Node *body = ast::node(callable, "body"))
    collectLocalNames(body, locals);
}

// Names a nested function requires from SOME enclosing function scope: its
// own nonlocal declarations plus whatever its nested functions require and
// this function does not bind locally.
void collectNeededNonlocalNames(const parser::Node &callable,
                                llvm::StringSet<> &needed) {
  llvm::StringSet<> own;
  if (const auto *body = ast::nodeList(callable, "body"))
    for (const parser::NodePtr &statement : *body)
      collectOwnNonlocalNames(statement.get(), own);
  llvm::StringSet<> locals;
  collectFunctionLocalNames(callable, locals);
  for (const auto &entry : own)
    locals.erase(entry.getKey());

  llvm::SmallVector<const parser::Node *, 4> nested;
  if (const auto *body = ast::nodeList(callable, "body"))
    for (const parser::NodePtr &statement : *body)
      collectDirectNestedFunctions(statement.get(), nested);

  llvm::StringSet<> fromNested;
  for (const parser::Node *inner : nested)
    collectNeededNonlocalNames(*inner, fromNested);

  for (const auto &entry : own)
    needed.insert(entry.getKey());
  for (const auto &entry : fromNested)
    if (!locals.contains(entry.getKey()))
      needed.insert(entry.getKey());
}

} // namespace

llvm::SmallVector<std::string, 4>
lexicalCaptureNames(const parser::Node &callable) {
  llvm::StringSet<> locals;
  collectFunctionLocalNames(callable, locals);

  // A name this function declares nonlocal is a capture even when assigned
  // (the assignment targets the enclosing function's cell, not a new local).
  llvm::StringSet<> ownNonlocals;
  if (const auto *body = ast::nodeList(callable, "body"))
    for (const parser::NodePtr &statement : *body)
      collectOwnNonlocalNames(statement.get(), ownNonlocals);
  for (const auto &entry : ownNonlocals)
    locals.erase(entry.getKey());

  llvm::StringSet<> reads;
  if (const auto *body = ast::nodeList(callable, "body"))
    for (const parser::NodePtr &statement : *body)
      collectReadNames(statement.get(), reads);
  if (const parser::Node *body = ast::node(callable, "body"))
    collectReadNames(body, reads);
  for (const auto &entry : ownNonlocals)
    reads.insert(entry.getKey());
  // A deeper nested function's nonlocal target that this function does not
  // bind must ride this function's environment (write-only uses do not show
  // up as reads).
  llvm::StringSet<> needed;
  {
    llvm::SmallVector<const parser::Node *, 4> nested;
    if (const auto *body = ast::nodeList(callable, "body"))
      for (const parser::NodePtr &statement : *body)
        collectDirectNestedFunctions(statement.get(), nested);
    for (const parser::Node *inner : nested)
      collectNeededNonlocalNames(*inner, needed);
  }
  for (const auto &entry : needed)
    reads.insert(entry.getKey());

  llvm::SmallVector<std::string, 4> captures;
  for (const auto &entry : reads)
    if (!locals.contains(entry.getKey()))
      captures.push_back(entry.getKey().str());
  llvm::sort(captures);
  return captures;
}

namespace {
// How many times each name is ASSIGNED directly in this statement list,
// counting only bindings of the enclosing scope itself -- a nested function's
// own assignments bind its own locals.
void countNameAssignments(const std::vector<parser::NodePtr> *statements,
                          llvm::StringMap<unsigned> &counts,
                          unsigned weight = 1);

// ⭐ ONE ASSIGNMENT INSIDE A LOOP IS NOT ONE BINDING. The loop-target rule
// below said so for the target and nothing said it for the body, so
// `for i in ...: k = i * 10` left `k` bound once -- and a closure over it took
// a copy per trip instead of sharing the frame's cell:
//
//     def named(n: int):
//         for i in range(n):
//             k = i * 10
//             yield lambda: k
//     [f() for f in list(named(3))]   # gave [0, 10, 20]; CPython [20, 20, 20]
//
// The weight is what a loop multiplies, so a single write anywhere under one
// counts as the repetition it is.
void countNameAssignments(const parser::Node *node,
                          llvm::StringMap<unsigned> &counts,
                          unsigned weight = 1) {
  if (!node)
    return;
  if (node->kind == "FunctionDef" || node->kind == "AsyncFunctionDef" ||
      node->kind == "ClassDef" || node->kind == "Lambda")
    return;
  if (node->kind == "Assign" || node->kind == "AnnAssign" ||
      node->kind == "AugAssign" || node->kind == "NamedExpr") {
    llvm::StringSet<> targets;
    if (const parser::Node *target = ast::node(*node, "target"))
      collectAssignedNameTargets(target, targets);
    if (const auto *targetList = ast::nodeList(*node, "targets"))
      for (const parser::NodePtr &target : *targetList)
        collectAssignedNameTargets(target.get(), targets);
    for (const auto &entry : targets)
      counts[entry.getKey()] += weight;
  }
  if (node->kind == "For" || node->kind == "AsyncFor" ||
      node->kind == "comprehension")
    if (const parser::Node *target = ast::node(*node, "target")) {
      llvm::StringSet<> targets;
      collectAssignedNameTargets(target, targets);
      // A loop target is rebound every trip, which is more than once -- and a
      // comprehension's target is rebound every element, in a frame of its own
      // that every closure built in the body shares.
      for (const auto &entry : targets)
        counts[entry.getKey()] += 2 * weight;
    }
  const bool repeats = node->kind == "For" || node->kind == "AsyncFor" ||
                       node->kind == "While" || node->kind == "comprehension";
  unsigned childWeight = repeats ? weight * 2 : weight;
  for (const parser::Field &field : node->fields) {
    if (const auto *child = std::get_if<parser::NodePtr>(&field.value))
      countNameAssignments(child->get(), counts, childWeight);
    else if (const auto *children =
                 std::get_if<std::vector<parser::NodePtr>>(&field.value))
      countNameAssignments(children, counts, childWeight);
  }
}

void countNameAssignments(const std::vector<parser::NodePtr> *statements,
                          llvm::StringMap<unsigned> &counts, unsigned weight) {
  if (!statements)
    return;
  for (const parser::NodePtr &statement : *statements)
    countNameAssignments(statement.get(), counts, weight);
}
} // namespace

llvm::StringSet<> functionLocalNames(const parser::Node &callable) {
  llvm::StringSet<> locals;
  collectFunctionLocalNames(callable, locals);
  return locals;
}

llvm::StringSet<> nonlocalBoxedNames(const parser::Node &callable) {
  llvm::StringSet<> locals;
  collectFunctionLocalNames(callable, locals);
  llvm::StringSet<> ownNonlocals;
  if (const auto *body = ast::nodeList(callable, "body"))
    for (const parser::NodePtr &statement : *body)
      collectOwnNonlocalNames(statement.get(), ownNonlocals);
  for (const auto &entry : ownNonlocals)
    locals.erase(entry.getKey());

  llvm::SmallVector<const parser::Node *, 4> nested;
  if (const auto *body = ast::nodeList(callable, "body"))
    for (const parser::NodePtr &statement : *body)
      collectDirectNestedFunctions(statement.get(), nested);

  llvm::StringSet<> needed;
  for (const parser::Node *inner : nested)
    collectNeededNonlocalNames(*inner, needed);

  llvm::StringSet<> boxed;
  for (const auto &entry : needed)
    if (locals.contains(entry.getKey()))
      boxed.insert(entry.getKey());

  // ⭐ A name a nested function READS is a cell in CPython too, not just one
  // it declares `nonlocal`. Capturing it by value at the def site made the
  // enclosing scope's later writes invisible:
  //
  //     def run() -> None:
  //         n: int = 1
  //         def show() -> None:
  //             print(n)
  //         n = 2
  //         show()      # printed 1; CPython prints 2
  //
  // Boxed only when the enclosing scope assigns it MORE THAN ONCE: with a
  // single binding the cell and the copy hold the same thing forever, and
  // boxing every captured name would put a cell behind every read-only
  // capture in the program.
  llvm::SmallVector<const parser::Node *, 4> readers = nested;
  if (const auto *body = ast::nodeList(callable, "body"))
    for (const parser::NodePtr &statement : *body)
      collectDirectNestedLambdas(statement.get(), readers);
  llvm::StringSet<> readByNested;
  for (const parser::Node *inner : readers)
    for (const std::string &capture : lexicalCaptureNames(*inner))
      readByNested.insert(capture);
  llvm::StringMap<unsigned> assignments;
  // ⭐ A PARAMETER arrives already bound, so one assignment in the body is its
  // SECOND binding. Counting only the body's missed it:
  //
  //     def make(n: int) -> Callable[[], int]:
  //         def get() -> int: return n
  //         n = n * 2
  //         return get
  //     print(make(5)())      # printed 5; CPython prints 10
  {
    llvm::StringSet<> parameters;
    collectParameterNames(ast::node(callable, "args"), parameters);
    for (const auto &entry : parameters)
      ++assignments[entry.getKey()];
  }
  if (const auto *body = ast::nodeList(callable, "body"))
    countNameAssignments(body, assignments);
  for (const auto &entry : readByNested)
    if (locals.contains(entry.getKey()) &&
        assignments.lookup(entry.getKey()) > 1)
      boxed.insert(entry.getKey());

  // ⭐ AND A NAME BOUND ONLY AFTER THE DEF THAT READS IT, which the
  // more-than-once rule above cannot see: one binding means the cell and the
  // copy hold the same thing forever only when that binding comes FIRST.
  // Mutually recursive nested functions are the shape this refused --
  // `is_even` reads `is_odd`, whose def has not run yet -- and they were
  // "emit error: unresolved name 'is_odd'". The same pair written at module
  // scope or in a class body has always worked.
  for (const auto &entry : namesBoundAfterNestedReader(callable))
    boxed.insert(entry.getKey());
  return boxed;
}

llvm::StringSet<> namesBoundAfterNestedReader(const parser::Node &callable) {
  llvm::StringSet<> forward;
  const auto *body = ast::nodeList(callable, "body");
  if (!body)
    return forward;
  llvm::StringSet<> locals;
  collectFunctionLocalNames(callable, locals);
  llvm::StringSet<> parameters;
  collectParameterNames(ast::node(callable, "args"), parameters);

  llvm::StringMap<unsigned> firstRead;
  llvm::StringMap<unsigned> firstBinding;
  for (auto [index, statement] : llvm::enumerate(*body)) {
    llvm::SmallVector<const parser::Node *, 4> readers;
    collectDirectNestedFunctions(statement.get(), readers);
    collectDirectNestedLambdas(statement.get(), readers);
    for (const parser::Node *reader : readers)
      for (const std::string &capture : lexicalCaptureNames(*reader))
        firstRead.try_emplace(capture, static_cast<unsigned>(index));
    llvm::StringSet<> bound;
    collectLocalNames(statement.get(), bound);
    for (const auto &entry : bound)
      firstBinding.try_emplace(entry.getKey(), static_cast<unsigned>(index));
  }

  for (const auto &entry : firstRead) {
    llvm::StringRef name = entry.getKey();
    if (parameters.contains(name) || !locals.contains(name))
      continue;
    auto binding = firstBinding.find(name);
    if (binding == firstBinding.end())
      continue;
    if (binding->second > entry.getValue())
      forward.insert(name);
  }
  return forward;
}

llvm::StringSet<> singleAssignmentNames(const parser::Node &scope) {
  llvm::StringMap<unsigned> counts;
  countNameAssignments(ast::nodeList(scope, "body"), counts);
  llvm::StringSet<> once;
  for (const auto &entry : counts)
    if (entry.getValue() == 1)
      once.insert(entry.getKey());
  return once;
}

llvm::StringSet<> namesReadByNestedCallables(
    const std::vector<parser::NodePtr> *body) {
  llvm::SmallVector<const parser::Node *, 4> readers;
  if (body)
    for (const parser::NodePtr &statement : *body) {
      collectDirectNestedFunctions(statement.get(), readers);
      collectDirectNestedLambdas(statement.get(), readers);
    }
  llvm::StringSet<> names;
  for (const parser::Node *reader : readers)
    for (const std::string &capture : lexicalCaptureNames(*reader))
      names.insert(capture);
  return names;
}

llvm::StringSet<> reboundNames(const parser::Node &scope) {
  llvm::StringMap<unsigned> counts;
  countNameAssignments(ast::nodeList(scope, "body"), counts);
  llvm::StringSet<> rebound;
  for (const auto &entry : counts)
    if (entry.getValue() > 1)
      rebound.insert(entry.getKey());
  return rebound;
}

std::string sanitizedSymbolPart(llvm::StringRef text) {
  std::string result;
  result.reserve(text.size());
  for (char ch : text)
    result.push_back(llvm::isAlnum(ch) ? ch : '_');
  return result.empty() ? "callable" : result;
}

} // namespace lython::emitter
