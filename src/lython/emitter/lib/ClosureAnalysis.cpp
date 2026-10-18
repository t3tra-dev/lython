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

// Does this def's OWN body yield? Nested defs are skipped: a generator inside a
// plain def does not make the def a generator.
bool bodyYields(const parser::Node &node) {
  if (node.kind == "FunctionDef" || node.kind == "AsyncFunctionDef" ||
      node.kind == "Lambda" || node.kind == "ClassDef")
    return false;
  if (node.kind == "Yield" || node.kind == "YieldFrom")
    return true;
  for (const parser::Field &field : node.fields) {
    if (const auto *child = std::get_if<parser::NodePtr>(&field.value)) {
      if (*child && bodyYields(**child))
        return true;
      continue;
    }
    if (const auto *children =
            std::get_if<std::vector<parser::NodePtr>>(&field.value))
      for (const parser::NodePtr &child : *children)
        if (child && bodyYields(*child))
          return true;
  }
  return false;
}

bool defIsAGenerator(const parser::Node &def) {
  if (const auto *body = ast::nodeList(def, "body"))
    for (const parser::NodePtr &statement : *body)
      if (statement && bodyYields(*statement))
        return true;
  return false;
}

MutualNestedDefGroup mutualNestedDefGroup(const parser::Node &callable) {
  MutualNestedDefGroup group;
  const auto *body = ast::nodeList(callable, "body");
  if (!body)
    return group;
  // Only the defs the body declares DIRECTLY: a def nested one level deeper is
  // reached through its own enclosing def, whose captures this walk already
  // sees.
  llvm::StringMap<const parser::Node *> siblings;
  for (const parser::NodePtr &statement : *body) {
    if (!statement || statement->kind != "FunctionDef")
      continue;
    // ⛔ NOT A GENERATOR. Its frame is a state machine with its own capture
    // seeding, and putting one in the group swapped a readable refusal ("a
    // generator cannot carry a value of contract 'builtins.function' across a
    // suspension yet") for an internal one about clone entry seeding. A
    // generator that names a member still captures it as a value, which closes
    // no cycle: the member it captures holds no function object of its own.
    if (defIsAGenerator(*statement))
      continue;
    if (auto name = ast::string(*statement, "name"))
      siblings.try_emplace(*name, statement.get());
  }
  if (siblings.size() < 2)
    return group;
  llvm::StringSet<> enclosingNames;
  collectFunctionLocalNames(callable, enclosingNames);
  // ⛔ Against the ENCLOSING scope's names and not against "anything the walk
  // reports": a member may read a module global or a builtin freely -- those
  // resolve by symbol and are not captures. What the group has to carry is a
  // name the enclosing FRAME owns.
  llvm::StringSet<> external;
  bool referencesASibling = false;
  for (const auto &entry : siblings)
    for (const std::string &capture : lexicalCaptureNames(*entry.second)) {
      llvm::StringRef name(capture);
      if (siblings.count(name)) {
        referencesASibling = true;
        continue;
      }
      if (enclosingNames.contains(name))
        external.insert(name);
    }
  if (!referencesASibling)
    return group;
  for (const auto &entry : siblings)
    group.members.insert(entry.getKey());
  for (const auto &entry : external)
    group.captures.push_back(entry.getKey().str());
  llvm::sort(group.captures);
  return group;
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

namespace {

bool isComprehensionNode(const parser::Node &node) {
  return node.kind == "ListComp" || node.kind == "SetComp" ||
         node.kind == "DictComp" || node.kind == "GeneratorExp";
}

// Every spelling `node`'s subtree uses for a name: a Name, a parameter, a
// `global` or `nonlocal` declaration.
void countNameSpellings(const parser::Node *node,
                        llvm::StringMap<unsigned> &counts) {
  if (!node)
    return;
  if (node->kind == "Name") {
    if (auto id = ast::string(*node, "id"))
      ++counts[*id];
  } else if (node->kind == "arg") {
    if (auto arg = ast::string(*node, "arg"))
      ++counts[*arg];
  } else if (node->kind == "Global" || node->kind == "Nonlocal") {
    if (const auto *names = ast::stringList(*node, "names"))
      for (const std::string &name : *names)
        ++counts[name];
  }
  for (const parser::Field &field : node->fields) {
    if (const auto *child = std::get_if<parser::NodePtr>(&field.value))
      countNameSpellings(child->get(), counts);
    else if (const auto *children =
                 std::get_if<std::vector<parser::NodePtr>>(&field.value))
      for (const parser::NodePtr &element : *children)
        countNameSpellings(element.get(), counts);
  }
}

void appendTargetSpellings(const parser::Node *target,
                           llvm::SmallVectorImpl<std::string> &names) {
  if (!target)
    return;
  if (target->kind == "Name") {
    names.push_back(std::string(ast::nameSpelling(*target)));
    return;
  }
  if (target->kind == "Starred") {
    appendTargetSpellings(ast::node(*target, "value"), names);
    return;
  }
  if (target->kind == "Tuple" || target->kind == "List")
    if (const auto *elts = ast::nodeList(*target, "elts"))
      for (const parser::NodePtr &elt : *elts)
        appendTargetSpellings(elt.get(), names);
}

bool comprehensionBinds(const parser::Node &comprehension,
                        llvm::StringRef name) {
  const auto *generators = ast::nodeList(comprehension, "generators");
  if (!generators)
    return false;
  for (const parser::NodePtr &generator : *generators) {
    if (!generator)
      continue;
    llvm::SmallVector<std::string, 2> targets;
    appendTargetSpellings(ast::node(*generator, "target"), targets);
    if (llvm::is_contained(targets, name))
      return true;
  }
  return false;
}

void renameNameNode(parser::Node &node, llvm::StringRef to) {
  for (parser::Field &field : node.fields)
    if (field.name == "id")
      field.value = to.str();
}

// `from` becomes `to` wherever it still names the comprehension's binding:
// not inside a lambda or a nested comprehension that binds `from` again, past
// the nested comprehension's first iterable, which its ENCLOSING scope
// evaluates.
void renameBoundSpelling(parser::Node *node, llvm::StringRef from,
                         llvm::StringRef to) {
  if (!node)
    return;
  if (node->kind == "Name") {
    if (llvm::StringRef(ast::nameSpelling(*node)) == from)
      renameNameNode(*node, to);
    return;
  }
  // A lambda that takes `from` as a parameter binds it again -- but its
  // defaults are evaluated where the lambda is built, so `lambda x=x: x`
  // reads the comprehension's `x` there.
  if (node->kind == "Lambda") {
    const parser::Node *arguments = ast::node(*node, "args");
    bool parameter = false;
    if (arguments)
      for (const parser::Field &field : arguments->fields) {
        auto takes = [&](const parser::Node *arg) {
          return arg && arg->kind == "arg" &&
                 llvm::StringRef(ast::nameSpelling(*arg)) == from;
        };
        if (const auto *one = std::get_if<parser::NodePtr>(&field.value))
          parameter = parameter || takes(one->get());
        else if (const auto *many =
                     std::get_if<std::vector<parser::NodePtr>>(&field.value))
          for (const parser::NodePtr &arg : *many)
            parameter = parameter || takes(arg.get());
      }
    if (parameter) {
      if (arguments)
        for (llvm::StringRef defaults : {"defaults", "kw_defaults"})
          if (const auto *values = ast::nodeList(*arguments, defaults))
            for (const parser::NodePtr &value : *values)
              renameBoundSpelling(value.get(), from, to);
      return;
    }
  }
  if (isComprehensionNode(*node) && comprehensionBinds(*node, from)) {
    if (const auto *generators = ast::nodeList(*node, "generators"))
      if (!generators->empty() && generators->front())
        renameBoundSpelling(
            const_cast<parser::Node *>(ast::node(*generators->front(), "iter")),
            from, to);
    return;
  }
  for (parser::Field &field : node->fields) {
    if (auto *child = std::get_if<parser::NodePtr>(&field.value))
      renameBoundSpelling(child->get(), from, to);
    else if (auto *children =
                 std::get_if<std::vector<parser::NodePtr>>(&field.value))
      for (parser::NodePtr &element : *children)
        renameBoundSpelling(element.get(), from, to);
  }
}

void scopeTargetsOf(parser::Node &comprehension, const parser::Node &scope,
                    unsigned &counter) {
  auto *generators = const_cast<std::vector<parser::NodePtr> *>(
      ast::nodeList(comprehension, "generators"));
  if (!generators)
    return;
  llvm::StringMap<unsigned> inScope;
  llvm::StringMap<unsigned> inComprehension;
  countNameSpellings(&scope, inScope);
  countNameSpellings(&comprehension, inComprehension);
  llvm::SmallVector<std::string, 4> targets;
  for (const parser::NodePtr &generator : *generators)
    if (generator)
      appendTargetSpellings(ast::node(*generator, "target"), targets);
  llvm::StringSet<> renamed;
  for (const std::string &name : targets) {
    if (!renamed.insert(name).second ||
        inScope.lookup(name) <= inComprehension.lookup(name))
      continue;
    std::string fresh = name + "$comp" + std::to_string(++counter);
    // The comprehension's scope: every generator but the first one's
    // iterable, the filters and the element.
    for (auto [index, generator] : llvm::enumerate(*generators)) {
      if (!generator)
        continue;
      for (parser::Field &field : generator->fields) {
        if (field.name == "iter" && index == 0)
          continue;
        if (auto *child = std::get_if<parser::NodePtr>(&field.value))
          renameBoundSpelling(child->get(), name, fresh);
        else if (auto *children =
                     std::get_if<std::vector<parser::NodePtr>>(&field.value))
          for (parser::NodePtr &element : *children)
            renameBoundSpelling(element.get(), name, fresh);
      }
    }
    for (parser::Field &field : comprehension.fields)
      if (field.name == "elt" || field.name == "key" || field.name == "value")
        if (auto *child = std::get_if<parser::NodePtr>(&field.value))
          renameBoundSpelling(child->get(), name, fresh);
  }
}

void scopeComprehensionsUnder(parser::Node *node, const parser::Node *scope,
                              unsigned &counter) {
  if (!node)
    return;
  if (node->kind == "FunctionDef" || node->kind == "AsyncFunctionDef" ||
      node->kind == "Lambda")
    scope = node;
  if (isComprehensionNode(*node))
    scopeTargetsOf(*node, *scope, counter);
  for (parser::Field &field : node->fields) {
    if (auto *child = std::get_if<parser::NodePtr>(&field.value))
      scopeComprehensionsUnder(child->get(), scope, counter);
    else if (auto *children =
                 std::get_if<std::vector<parser::NodePtr>>(&field.value))
      for (parser::NodePtr &element : *children)
        scopeComprehensionsUnder(element.get(), scope, counter);
  }
}

} // namespace

// ⭐ A COMPREHENSION'S TARGETS ARE ITS OWN, as CPython's comprehension scope
// makes them. Every form is emitted INLINE here -- a loop in the enclosing
// frame -- so a target spelled like another binding of that frame WAS that
// binding:
//
//     k: int = 5
//     print([k for k in "xy"])     # a str value cannot be stored where int
//                                  # is declared
//
// and `sum(1 for k in s)` beside a local `k` was refused as one loop-carried
// local of two types, while a module `k = 5` with a comprehension over `k`
// anywhere stopped being bound once and became unreadable from a function.
//
// ⛔ Only on a collision, counted over the whole enclosing function (or
// module): an unrenamed target keeps its own spelling in every diagnostic
// about it.
void scopeComprehensionTargets(const parser::Node &moduleNode) {
  unsigned counter = 0;
  for (const parser::Field &field : moduleNode.fields)
    if (const auto *children =
            std::get_if<std::vector<parser::NodePtr>>(&field.value))
      for (const parser::NodePtr &statement : *children)
        scopeComprehensionsUnder(statement.get(), &moduleNode, counter);
}

std::string sanitizedSymbolPart(llvm::StringRef text) {
  std::string result;
  result.reserve(text.size());
  for (char ch : text)
    result.push_back(llvm::isAlnum(ch) ? ch : '_');
  return result.empty() ? "callable" : result;
}

} // namespace lython::emitter
