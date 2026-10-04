#include "EmptyLiteralSeed.h"

#include "AstAccess.h"
#include "PrimitiveTypes.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"

#include <functional>
#include <optional>
#include <string>
#include <string_view>

namespace lython::emitter {
namespace {

// The constant bindings in the rest of the suite, bound before the scan walks
// it. The seed usually mentions one and the walk has not reached it:
//
//     out = []          <- deciding here
//     k = 0
//     while k < n:
//         out.append(k + 1)
//
// `k` is not bound yet, so `k + 1` infers object and the whole scan declines.
// A CONSTANT's type depends on nothing, so binding it early is the same answer
// the walk will reach, just sooner -- and any name whose value is not a
// constant is left alone rather than guessed at.
const llvm::StringMap<mlir::Type> &noLocalCallables() {
  static const llvm::StringMap<mlir::Type> empty;
  return empty;
}

void preBindSuiteConstants(const TypeSystem &types, const SuiteCursor &cursor) {
  if (!cursor.suite || cursor.from > cursor.suite->size())
    return;
  auto walk = [&](const parser::Node &node, auto &&recurse) -> void {
    if (node.kind == "FunctionDef" || node.kind == "AsyncFunctionDef" ||
        node.kind == "ClassDef")
      return;
    if (node.kind == "Assign") {
      const parser::Node *value = ast::node(node, "value");
      const auto *targets = ast::nodeList(node, "targets");
      if (value && value->kind == "Constant" && targets &&
          targets->size() == 1 && targets->front() &&
          targets->front()->kind == "Name")
        if (mlir::Type constantType =
                types.widenLiteral(types.inferExpr(value)))
          types.bindLocalSymbol(ast::nameSpelling(*targets->front()),
                                constantType);
    }
    for (const parser::Field &field : node.fields) {
      if (const auto *child = std::get_if<parser::NodePtr>(&field.value)) {
        if (*child)
          recurse(**child, recurse);
        continue;
      }
      if (const auto *children =
              std::get_if<std::vector<parser::NodePtr>>(&field.value))
        for (const parser::NodePtr &child : *children)
          if (child)
            recurse(*child, recurse);
    }
  };
  for (std::size_t index = cursor.from; index < cursor.end(); ++index)
    if (cursor.keeps(index) && (*cursor.suite)[index])
      walk(*(*cursor.suite)[index], walk);
}

void collectStatementNames(const parser::Node &node, StatementNames &names) {
  if (node.kind == "Name") {
    llvm::StringRef spelling = ast::nameSpelling(node);
    names.mentioned.insert(spelling);
    if (const parser::Node *context = ast::node(node, "ctx");
        context && context->kind == "Store")
      names.stored.insert(spelling);
  } else if (node.kind == "Attribute") {
    if (std::optional<std::string_view> attr = ast::string(node, "attr"))
      names.mentioned.insert(*attr);
  } else if (node.kind == "FunctionDef" || node.kind == "AsyncFunctionDef" ||
             node.kind == "ClassDef" || node.kind == "ExceptHandler") {
    // A binding the AST spells as a string rather than a Name.
    if (std::optional<std::string_view> bound = ast::string(node, "name"))
      names.stored.insert(*bound);
  } else if (node.kind == "alias") {
    std::optional<std::string_view> bound = ast::string(node, "asname");
    if (!bound)
      bound = ast::string(node, "name");
    if (bound)
      names.stored.insert(llvm::StringRef(*bound).split('.').first);
  }
  for (const parser::Field &field : node.fields) {
    if (const auto *child = std::get_if<parser::NodePtr>(&field.value)) {
      if (*child)
        collectStatementNames(**child, names);
      continue;
    }
    if (const auto *children =
            std::get_if<std::vector<parser::NodePtr>>(&field.value))
      for (const parser::NodePtr &child : *children)
        if (child)
          collectStatementNames(*child, names);
  }
}

// The statements of the cursors that can matter to a scan for `name`, as a
// mask on each cursor. One RELEVANT to it spells it, or spells a name bound by
// a statement that does -- the locals `derivedFromName` and `subscriptAliases`
// follow, through which a later statement may fill the container without
// spelling it. One it NEEDS binds a name a kept statement reads, read
// backwards from the last relevant one. Nothing else seeds anything or leaves
// a binding a seed reads.
//
// ⛔ Every scan read every statement in its cursors. A module of N statements
// with M empty literals cost N x M tree walks -- 39 s of a 64-case golden
// batch -- and a module global is scanned from the module's FIRST statement,
// so cutting only the tail left the head of every scan in place.
llvm::SmallVector<SuiteCursor, 4>
slicedCursors(const TypeSystem &types, llvm::StringRef name,
              llvm::ArrayRef<SuiteCursor> suites) {
  llvm::SmallVector<SuiteCursor, 4> sliced(suites.begin(), suites.end());
  struct Position {
    std::size_t cursor;
    std::size_t index;
    const StatementNames *names;
  };
  // The statements in the order the scan reads them: innermost cursor first.
  llvm::SmallVector<Position, 64> order;
  for (auto [position, cursor] : llvm::enumerate(sliced)) {
    if (!cursor.suite)
      continue;
    const std::vector<StatementNames> &statements =
        types.statementNamesOf(*cursor.suite);
    for (std::size_t index = cursor.from; index < cursor.end(); ++index)
      order.push_back({position, index, &statements[index]});
  }
  auto meets = [](const llvm::StringSet<> &names,
                  const llvm::StringSet<> &wanted) {
    const llvm::StringSet<> &small = names.size() < wanted.size() ? names : wanted;
    const llvm::StringSet<> &large = &small == &names ? wanted : names;
    for (const auto &entry : small)
      if (large.contains(entry.getKey()))
        return true;
    return false;
  };
  // Relevant: forwards, to a fixpoint; four rounds, then the whole sequence.
  llvm::StringSet<> relevantNames;
  relevantNames.insert(name);
  std::vector<bool> relevant(order.size(), false);
  bool converged = false;
  for (unsigned round = 0; round < 4 && !converged; ++round) {
    std::size_t before = relevantNames.size();
    for (auto [at, position] : llvm::enumerate(order)) {
      if (!meets(position.names->mentioned, relevantNames))
        continue;
      relevant[at] = true;
      for (const auto &stored : position.names->stored)
        relevantNames.insert(stored.getKey());
    }
    converged = relevantNames.size() == before;
  }
  if (!converged)
    return llvm::SmallVector<SuiteCursor, 4>(suites.begin(), suites.end());
  // Needed: backwards from the last relevant statement.
  std::vector<bool> kept(order.size(), false);
  llvm::StringSet<> needed;
  for (std::size_t at = order.size(); at-- > 0;) {
    const StatementNames &names = *order[at].names;
    if (!relevant[at] && !meets(names.stored, needed))
      continue;
    kept[at] = true;
    for (const auto &mentioned : names.mentioned)
      needed.insert(mentioned.getKey());
  }
  for (SuiteCursor &cursor : sliced) {
    cursor.keep.assign(cursor.suite ? cursor.suite->size() : 0, false);
    cursor.to = cursor.from;
  }
  for (auto [at, position] : llvm::enumerate(order)) {
    if (!kept[at])
      continue;
    SuiteCursor &cursor = sliced[position.cursor];
    cursor.keep[position.index] = true;
    cursor.to = std::max(cursor.to, position.index + 1);
  }
  return sliced;
}

} // namespace

const std::vector<StatementNames> &
TypeSystem::statementNamesOf(const std::vector<parser::NodePtr> &suite) const {
  std::vector<StatementNames> &statements = suiteStatementNames[&suite];
  bool current = statements.size() == suite.size();
  for (std::size_t index = 0; current && index < suite.size(); ++index)
    current = statements[index].node == suite[index].get();
  if (current)
    return statements;
  // ⛔ Checked, not trusted: the key is an address, and a suite freed and
  // another allocated there would otherwise read the first one's names.
  statements.assign(suite.size(), StatementNames{});
  for (std::size_t index = 0; index < suite.size(); ++index) {
    statements[index].node = suite[index].get();
    if (suite[index])
      collectStatementNames(*suite[index], statements[index]);
  }
  return statements;
}

mlir::Type emptyLiteralSeedTypeIn(const TypeSystem &types,
                                  llvm::StringRef name,
                                  llvm::StringRef literalKind,
                                  llvm::ArrayRef<SuiteCursor> allSuites,
                                  const llvm::StringMap<mlir::Type> *localSymbols,
                                  unsigned depth, unsigned subscriptDepth,
                                  llvm::StringRef receiver) {
  if (allSuites.empty() || !allSuites.front().suite ||
      allSuites.front().from > allSuites.front().suite->size())
    return {};
  llvm::SmallVector<SuiteCursor, 4> boundedSuites =
      slicedCursors(types, name, allSuites);
  llvm::ArrayRef<SuiteCursor> suites = boundedSuites;
  // ⛔ The walk's names go through a CONTEXT rather than into the scope this
  // pushes: `bindLocalSymbol` would make them visible to everything the scan
  // calls, and the signature walk's bindings are provisional -- it is reading a
  // body whose parameters may still be variables.
  auto inferHere = [&](const parser::Node *expr) {
    return localSymbols ? types.inferExpr(
                              expr, ExprInferenceContext{noLocalCallables(),
                                                         nullptr, localSymbols,
                                                         /*strict=*/false})
                        : types.inferExpr(expr);
  };
  bool isMapping = literalKind == "Dict";
  // `dict()`/`set()`/`list()` spell the same literals as calls; the callers map
  // them before asking, and the nested ask below has to map them too.
  // ⭐ THE CONTAINER BEING DECIDED IS NOT ALWAYS SPELLED AS A NAME. At
  // `subscriptDepth` 1 it is `name[...]`, which is how every operation on a
  // container stored INSIDE another one reads:
  //
  //     out = {}
  //     for w in words:
  //         if w[0] not in out:
  //             out[w[0]] = []
  //         out[w[0]].append(w)
  //
  // The adjacency map, the grouping, the bucket table. Asking the same scan
  // one subscript deeper answers what fills the inner container, and the
  // outer element is that answer wrapped -- rather than the `list[object]`
  // reading the empty literal gives on its own.
  //
  // ⭐ AND A LOCAL BOUND TO IT IS IT. `bucket = out.setdefault(k, [])` names
  // `out[k]`, and the appends that follow go through THAT name:
  //
  //     out = {}
  //     for w in words:
  //         bucket = out.setdefault(w[0], [])
  //         bucket.append(w)
  //
  // which is the grouping written the short way. `bucket = out[k]` is the same
  // alias; both are collected below and only at a depth where they can mean
  // one -- at depth 0 the container IS the name, and a local bound to it is a
  // second reference this scan has no reason to follow.
  llvm::StringSet<> subscriptAliases;
  // The container itself, however it is spelled: a bare name, or `<receiver>.
  // <name>` when this scan was asked about a field.
  auto isTheContainerItself = [&](const parser::Node *node) -> bool {
    if (!node)
      return false;
    if (receiver.empty())
      return node->kind == "Name" &&
             llvm::StringRef(ast::nameSpelling(*node)) == name;
    if (node->kind != "Attribute")
      return false;
    const parser::Node *base = ast::node(*node, "value");
    std::optional<std::string_view> attr = ast::string(*node, "attr");
    return base && base->kind == "Name" &&
           llvm::StringRef(ast::nameSpelling(*base)) == receiver && attr &&
           llvm::StringRef(*attr) == name;
  };
  auto namesTheContainerAt = [&](const parser::Node *node,
                                 unsigned atDepth) -> bool {
    const parser::Node *current = node;
    for (unsigned level = 0; level < atDepth; ++level) {
      if (!current || current->kind != "Subscript")
        return false;
      current = ast::node(*current, "value");
    }
    return isTheContainerItself(current);
  };
  auto namesTheContainer = [&](const parser::Node *node) -> bool {
    if (subscriptDepth > 0 && node && node->kind == "Name" &&
        subscriptAliases.contains(ast::nameSpelling(*node)))
      return true;
    return namesTheContainerAt(node, subscriptDepth);
  };
  auto literalKindOf = [](const parser::Node &node) -> llvm::StringRef {
    if (node.kind != "Call")
      return node.kind;
    llvm::StringRef callee = ast::nameSpelling(*ast::node(node, "func"));
    return callee == "dict"    ? "Dict"
           : callee == "set"   ? "Set"
           : callee == "tuple" ? "Tuple"
                               : "List";
  };
  mlir::Type element;
  mlir::Type key;
  bool disagreed = false;

  // ⭐ Constant bindings in the rest of the suite are pre-bound, because the
  // seed usually mentions one and the walk has not reached it:
  //
  //     out = []          <- deciding here
  //     k = 0
  //     while k < n:
  //         out.append(k + 1)
  //
  // `k` is not bound yet, so `k + 1` infers object and the whole scan
  // declines. A CONSTANT's type depends on nothing, so binding it early is
  // the same answer the walk will reach, just sooner -- and any name whose
  // value is not a constant is left alone rather than guessed at.
  TypeSystem::Scope seedScope = types.pushScope();
  preBindSuiteConstants(types, suites.front());

  // ⛔ A seed that MENTIONS the name is unknowable here and is skipped rather
  // than counted as disagreement. `d[w] = d[w] + 1` beside `d[w] = 1` is the
  // frequency-count idiom, and the first of those two reads `d` at the type
  // this scan is trying to decide -- object -- so counting it would make the
  // pair disagree and leave the whole thing at object, which is where it
  // started.
  //
  // ⭐ AND A LOCAL THE NAME FLOWED INTO IS THE NAME. `derivedFromName` below
  // collects the assignments in the scanned suites whose value reads the name
  // -- directly or through another such local -- because a seed spelled with
  // one of those reads the type this scan is deciding, one binding removed:
  //
  //     stack = []
  //     for t in toks:
  //         if t == "+":
  //             b = stack.pop()
  //             a = stack.pop()
  //             stack.append(a + b)      <- reads `stack`, through a and b
  //         else:
  //             stack.append(int(t))     <- the seed
  //
  // Without it the first append answered `object`, the second answered `int`,
  // and the pair DISAGREED -- so the whole shape, which is every stack
  // machine, got no seed at all. Skipping it leaves the honest seed standing.
  llvm::StringSet<> derivedFromName;
  if (receiver.empty())
    derivedFromName.insert(name);
  auto mentionsName = [&](const parser::Node *node, auto &&recurse) -> bool {
    if (!node)
      return false;
    if (isTheContainerItself(node))
      return true;
    if (node->kind == "Name" &&
        derivedFromName.contains(ast::nameSpelling(*node)))
      return true;
    for (const parser::Field &field : node->fields) {
      if (const auto *child = std::get_if<parser::NodePtr>(&field.value)) {
        if (recurse(child->get(), recurse))
          return true;
        continue;
      }
      if (const auto *children =
              std::get_if<std::vector<parser::NodePtr>>(&field.value))
        for (const parser::NodePtr &child : *children)
          if (recurse(child.get(), recurse))
            return true;
    }
    return false;
  };
  auto noteExpr = [&](mlir::Type &slot, const parser::Node *expr,
                      auto &&noteType) {
    if (!expr || mentionsName(expr, mentionsName))
      return;
    noteType(slot, inferHere(expr));
  };
  // ⭐ AN EMPTY CONTAINER PUT INTO THIS ONE IS ASKED, NOT READ. Reading one
  // answers `list[builtins.object]` -- a type, so it was taken -- and the
  // inner element was lost:
  //
  //     out = {}
  //     out[k] = []
  //     out[k].append(w)          # nothing looked here
  //
  // Asking the same scan one subscript deeper looks exactly there. The same
  // shape written through a LOCAL (`bucket = []; out[k] = bucket`) was already
  // answered, one spelling over, by the local seeding beside this.
  //
  // ⛔ Falls back to reading it when the deeper scan finds nothing: the erased
  // container is what this returned before, and a program that only prints the
  // outer one never decodes an inner element.
  // ⭐ AND AN EMPTY ONE THAT NOTHING FILLS DOES NOT DISAGREE WITH A FULL ONE.
  // `emptyFallback` holds what reading an unfilled empty literal answers, and
  // it is used only when nothing else contributed -- the rule a sibling
  // literal already follows (`joinIgnoringEmptyLiterals`), which a store had
  // not:
  //
  //     table = {}
  //     table["start"] = {"a": "middle"}
  //     table["end"] = {}          <- read as dict[object, object]
  //     # the pair disagreed and the whole table stayed erased
  //
  // -- the shape of every transition table with a terminal state in it.
  mlir::Type emptyFallback;
  bool emptyFallbackDisagreed = false;
  auto noteMaybeContainer = [&](mlir::Type &slot, const parser::Node *expr,
                                auto &&noteType) {
    if (expr && depth < 3 && isEmptyContainerExpression(expr)) {
      if (mlir::Type inner = emptyLiteralSeedTypeIn(
              types, name, literalKindOf(*expr), allSuites, localSymbols,
              depth + 1, subscriptDepth + 1)) {
        noteType(slot, inner);
        return;
      }
      if (&slot == &element) {
        mlir::Type read = types.widenLiteral(inferHere(expr));
        if (!read)
          return;
        if (!emptyFallback)
          emptyFallback = read;
        else if (emptyFallback != read)
          emptyFallbackDisagreed = true;
        return;
      }
    }
    noteExpr(slot, expr, noteType);
  };
  auto note = [&](mlir::Type &slot, mlir::Type seen) {
    if (!seen || disagreed)
      return;
    mlir::Type widened = types.widenLiteral(seen);
    if (!widened || widened == types.object()) {
      disagreed = true;
      return;
    }
    if (!slot) {
      slot = widened;
      return;
    }
    if (slot != widened)
      disagreed = true;
  };

  // ⭐ THE COUNTING IDIOM SEEDS ITSELF THROUGH `get`, and it is the one shape
  // where the ONLY store mentions the name:
  //
  //     counts = {}
  //     for w in words:
  //         counts[w] = counts.get(w, 0) + 1
  //     # !py.union<int, object> does not provide manifest method '__add__'
  //
  // The skip above is right in general -- a seed that reads the name reads it
  // at the type being decided -- but a `.get(key, default)` on that same name
  // carries the answer in its DEFAULT: that is what the value is when the key
  // is absent. Binding it provisionally and re-inferring the whole stored
  // expression is what keeps `... + 1.5` a float instead of the default's int.
  mlir::Type deferredElement;
  auto getDefaultOnName = [&](const parser::Node *expr,
                              auto &&recurse) -> const parser::Node * {
    if (!expr)
      return nullptr;
    if (expr->kind == "Call") {
      const parser::Node *callee = ast::node(*expr, "func");
      const auto *callArgs = ast::nodeList(*expr, "args");
      const auto *callKeywords = ast::nodeList(*expr, "keywords");
      if (callee && callee->kind == "Attribute" && callArgs &&
          callArgs->size() == 2 && (!callKeywords || callKeywords->empty())) {
        const parser::Node *receiver = ast::node(*callee, "value");
        std::optional<std::string_view> method = ast::string(*callee, "attr");
        if (receiver && namesTheContainer(receiver) && method &&
            *method == "get")
          return (*callArgs)[1].get();
      }
    }
    for (const parser::Field &field : expr->fields) {
      if (const auto *child = std::get_if<parser::NodePtr>(&field.value)) {
        if (const parser::Node *found = recurse(child->get(), recurse))
          return found;
        continue;
      }
      if (const auto *children =
              std::get_if<std::vector<parser::NodePtr>>(&field.value))
        for (const parser::NodePtr &child : *children)
          if (const parser::Node *found = recurse(child.get(), recurse))
            return found;
    }
    return nullptr;
  };

  // What a statement leaves BEHIND for the rest of its suite: a nested def's
  // name, and a simple assignment's target. Bound after the statement is
  // scanned, so an assignment does not see itself.
  //
  // ⛔ Asked from BOTH walks. The generic recursion below reaches the suites
  // inside a statement; `scanRemainder` reaches the top-level ones, and
  // without this there it read `CACHE[n] = value` with `value` unbound -- the
  // memo table, whose store is the only thing that says what the table holds.
  std::function<void(const std::vector<parser::NodePtr> *, std::size_t,
                     std::optional<TypeSystem::Scope> &)>
      bindWhatItLeaves = [&](const std::vector<parser::NodePtr> *suite,
                             std::size_t index,
                             std::optional<TypeSystem::Scope> &scope) {
        if (!suite || index >= suite->size() || !(*suite)[index])
          return;
        const parser::Node &child = *(*suite)[index];
          // A nested def binds its name here too. The recursion above declines
          // to look INSIDE one (its own binding order), which is a different
          // question from what the name it leaves behind is worth:
          // `for i in ...: def f() -> int: return i` then `fs.append(f)` had
          // the element decided from a name the scan had no type for.
          if (child.kind == "FunctionDef" ||
              child.kind == "AsyncFunctionDef") {
            auto nestedName = ast::string(child, "name");
            if (!nestedName)
              return;
            FunctionSignature nested = types.functionSignature(child);
            if (!nested.publicCallable)
              return;
            if (!scope)
              scope.emplace(types.pushScope());
            types.bindLocalSymbol(*nestedName, nested.publicCallable);
            return;
          }
          if (child.kind != "Assign")
            return;
          const auto *assignTargets = ast::nodeList(child, "targets");
          const parser::Node *assigned = ast::node(child, "value");
          if (!assignTargets || assignTargets->size() != 1 ||
              !assignTargets->front() ||
              assignTargets->front()->kind != "Name" || !assigned)
            return;
          // ⭐ A LOCAL THAT IS ITSELF AN EMPTY LITERAL IS SEEDED, NOT READ.
          // Reading one answers `list[object]`, which is a type and therefore
          // passed the test below, so the OUTER container took it and the
          // inner element was lost two levels down:
          //
          //     def grid(rows: int, cols: int):
          //         out = []
          //         for _ in range(rows):
          //             line = []
          //             for _ in range(cols):
          //                 line.append(0)
          //             out.append(line)
          //         return out
          //     print(grid(2, 2)[0][0] + 1)
          //     # builtins.object does not provide manifest method '__add__'
          //
          // ⛔ Depth-bounded rather than cycle-detected: the recursion is a
          // container inside a container, which is two or three deep in real
          // programs, and a bound is cheaper to be sure of than a visited set
          // threaded through a scan that pushes type scopes.
          mlir::Type bound;
          llvm::StringRef assignedName =
              ast::nameSpelling(*assignTargets->front());
          if (depth < 3 && isEmptyContainerExpression(assigned) &&
              assignedName != name) {
            llvm::SmallVector<SuiteCursor, 4> nested;
            nested.push_back(SuiteCursor{suite, index + 1});
            // The whole sequence: `line` is filled where `name` is not
            // spelled, and the asked scan cuts its own.
            nested.append(allSuites.begin(), allSuites.end());
            bound = emptyLiteralSeedTypeIn(types, assignedName,
                                           literalKindOf(*assigned), nested,
                                           localSymbols, depth + 1);
          }
          if (!bound)
            bound = types.widenLiteral(inferHere(assigned));
          if (!bound || bound == types.object())
            return;
          if (!scope)
            scope.emplace(types.pushScope());
          types.bindLocalSymbol(assignedName, bound);
      };
  auto visit = [&](const parser::Node &node, auto &&recurse) -> void {
    if (disagreed)
      return;
    // A nested function has its own binding order; see the note above.
    if (node.kind == "FunctionDef" || node.kind == "AsyncFunctionDef" ||
        node.kind == "ClassDef")
      return;
    // ⭐ A FILLING CALL IS NOT ALWAYS A STATEMENT. This looked only at a bare
    // expression statement, so `bucket = out.setdefault(k, [])` -- the short
    // way to write a grouping -- was never seen: the call is the right-hand
    // side of an assignment. Every Call the walk reaches is asked now; the
    // walk descends into an assignment's value already, and the arms below
    // each require a method name and an arity, so a call that fills nothing
    // still notes nothing.
    {
      const parser::Node *call = &node;
      // ⭐ AND A CALLEE THAT DECLARES WHAT IT TAKES. Handing the container to
      // `def put(heap: "list[int]", value: int)` says its element as plainly
      // as an append does, and nothing looked:
      //
      //     h = []
      //     put(h, 5)
      //     print(h[0] + 1)
      //     # builtins.object does not provide manifest method '__add__'
      //
      // ⛔ Only a parameter that is a CONTAINER of this literal's kind with an
      // element of its own. `print(xs)` takes `object` and `len(xs)` takes a
      // structural bound; neither says anything about an element, and reading
      // them as if they did is the mistake the protocol repairs describe.
      if (call->kind == "Call" && !disagreed)
        if (const auto *callArgs = ast::nodeList(*call, "args"))
          for (auto [argIndex, argument] : llvm::enumerate(*callArgs)) {
            if (!argument || !namesTheContainer(argument.get()))
              continue;
            auto callable = mlir::dyn_cast_if_present<py::CallableType>(
                types.widenLiteral(inferHere(ast::node(*call, "func"))));
            if (!callable ||
                argIndex >= callable.getPositionalTypes().size())
              continue;
            auto declared = mlir::dyn_cast_if_present<py::ContractType>(
                types.widenLiteral(callable.getPositionalTypes()[argIndex]));
            if (!declared)
              continue;
            llvm::StringRef declaredName = declared.getContractName();
            llvm::StringRef wanted = isMapping         ? "builtins.dict"
                                     : literalKind == "Set" ? "builtins.set"
                                                            : "builtins.list";
            if (declaredName != wanted)
              continue;
            llvm::ArrayRef<mlir::Type> arguments = declared.getArguments();
            if (isMapping && arguments.size() == 2) {
              note(key, arguments[0]);
              note(element, arguments[1]);
            } else if (!isMapping && arguments.size() == 1) {
              note(element, arguments.front());
            }
          }
      if (call->kind == "Call") {
        const parser::Node *callee = ast::node(*call, "func");
        if (callee && callee->kind == "Attribute") {
          const parser::Node *receiver = ast::node(*callee, "value");
          std::optional<std::string_view> method =
              ast::string(*callee, "attr");
          const auto *args = ast::nodeList(*call, "args");
          // ⛔ `!empty()` BEFORE `front()`. A method call with no arguments
          // is still a call on this name, so it reaches here -- and
          // `args->front()` on an empty list read past the end and took the
          // compiler with it (SIGSEGV, no diagnostic). `xs = []` followed by
          // `xs.pop()` crashed, and so did `clear`, `copy`, `sort`, `reverse`,
          // `keys`, `values`, `items`, `popitem`, and every arity mistake
          // (`xs.append()`, `xs.insert()`), on `[]`, `{}` and `set()` alike:
          // 54 of 72 spellings measured. Only the ANNOTATED form escaped,
          // because an annotation means this scan never runs.
          //
          // ⛔ Nothing below wants a zero-argument call anyway -- each arm
          // asks for a size of its own -- so the guard costs no seeding.
          if (receiver && namesTheContainer(receiver) &&
              method && args && !args->empty() && args->front()) {
            // ⭐ EVERY OPERATION THAT PUTS SOMETHING IN IT SEEDS IT. Two were
            // recognised, and the rest of the ways Python fills a fresh
            // container left it erased -- each with the same message about a
            // `builtins.object` that no line of the program mentions:
            //
            //     xs = []; xs.extend([1]); xs[0] + 1
            //     xs = []; xs.insert(0, 1)
            //     s = set(); s.update([1])
            //     d = {}; d.update({"a": 1})
            //     d = {}; d.setdefault("a", 1)
            //
            // ⛔ An ITERABLE argument contributes its ELEMENT, not itself:
            // `extend([1])` puts an int in, where `append([1])` puts a list
            // in. That is why the two cannot share an arm.
            auto noteIterableElements = [&](const parser::Node *iterable) {
              if (!iterable || mentionsName(iterable, mentionsName))
                return;
              auto contract = mlir::dyn_cast_if_present<py::ContractType>(
                  types.widenLiteral(inferHere(iterable)));
              if (!contract) {
                disagreed = true;
                return;
              }
              llvm::ArrayRef<mlir::Type> arguments = contract.getArguments();
              if (isMapping) {
                if (arguments.size() != 2) {
                  disagreed = true;
                  return;
                }
                note(key, arguments[0]);
                note(element, arguments[1]);
                return;
              }
              if (arguments.size() != 1) {
                disagreed = true;
                return;
              }
              note(element, arguments.front());
            };
            if (args->size() == 1 &&
                (*method == "append" || *method == "add"))
              noteMaybeContainer(element, args->front().get(), note);
            else if (args->size() == 1 &&
                     (*method == "extend" || *method == "update"))
              noteIterableElements(args->front().get());
            else if (args->size() == 2 && *method == "insert" && !isMapping)
              noteMaybeContainer(element, (*args)[1].get(), note);
            else if (args->size() == 2 && *method == "setdefault" &&
                     isMapping) {
              noteExpr(key, args->front().get(), note);
              noteMaybeContainer(element, (*args)[1].get(), note);
            }
          }
        }
      }
    }
    // ⭐ AND `xs += [1]` IS `xs.extend([1])`, which is how an accumulator that
    // is built from slices rather than elements is written.
    if (node.kind == "AugAssign") {
      const parser::Node *target = ast::node(node, "target");
      const parser::Node *op = ast::node(node, "op");
      if (target && target->kind == "Name" &&
          llvm::StringRef(ast::nameSpelling(*target)) == name && op &&
          (op->kind == "Add" || op->kind == "BitOr")) {
        const parser::Node *operand = ast::node(node, "value");
        if (operand && !mentionsName(operand, mentionsName)) {
          auto contract = mlir::dyn_cast_if_present<py::ContractType>(
              types.widenLiteral(inferHere(operand)));
          llvm::ArrayRef<mlir::Type> arguments =
              contract ? contract.getArguments()
                       : llvm::ArrayRef<mlir::Type>();
          if (isMapping && arguments.size() == 2) {
            note(key, arguments[0]);
            note(element, arguments[1]);
          } else if (!isMapping && arguments.size() == 1) {
            note(element, arguments.front());
          } else {
            disagreed = true;
          }
        }
      }
    }
    if (node.kind == "Assign") {
      // ⭐ A LATER REBIND SEEDS IT TOO. `out = []` followed by `out = [1]` in
      // the same suite is the accumulator written the other way round, and
      // only the first assignment decided the name's type -- so the reversed
      // order (`out = [1]` then `out = []`) compiled and this one did not:
      //
      //     def f(flag: bool) -> "list[int]":
      //         out = []
      //         if flag:
      //             out = [1]
      //         return out         # cannot adapt return value
      //
      // ⛔ A rebind that MENTIONS the name is skipped for the same reason the
      // subscript seeds are: `out = out + [1]` reads `out` at the type this
      // scan is deciding.
      if (const auto *targets = ast::nodeList(node, "targets"))
        for (const parser::NodePtr &target : *targets) {
          if (!target || !isTheContainerItself(target.get()))
            continue;
          const parser::Node *rebound = ast::node(node, "value");
          if (!rebound || isEmptyContainerExpression(rebound) ||
              mentionsName(rebound, mentionsName))
            continue;
          llvm::StringRef wanted = isMapping ? "Dict" : literalKind;
          if (rebound->kind != wanted)
            continue;
          if (isMapping) {
            const auto *keys = ast::nodeList(*rebound, "keys");
            const auto *vals = ast::nodeList(*rebound, "values");
            if (keys && !keys->empty())
              noteExpr(key, keys->front().get(), note);
            if (vals && !vals->empty())
              noteExpr(element, vals->front().get(), note);
            continue;
          }
          if (const auto *elements = ast::nodeList(*rebound, "elts");
              elements && !elements->empty())
            noteExpr(element, elements->front().get(), note);
        }
      if (const auto *targets = ast::nodeList(node, "targets"))
        for (const parser::NodePtr &target : *targets) {
          if (!target || target->kind != "Subscript")
            continue;
          const parser::Node *receiver = ast::node(*target, "value");
          if (!receiver || !namesTheContainer(receiver))
            continue;
          if (!isMapping) {
            disagreed = true;
            return;
          }
          noteExpr(key, ast::node(*target, "slice"), note);
          const parser::Node *stored = ast::node(node, "value");
          // ⛔ Inside the walk, not after it: the stored expression usually
          // mentions the LOOP TARGET, which is bound by the scope this walk
          // pushes and gone by the time the walk returns.
          if (key && stored && !deferredElement &&
              mentionsName(stored, mentionsName))
            if (const parser::Node *fallback =
                    getDefaultOnName(stored, getDefaultOnName)) {
              mlir::Type provisional =
                  types.widenLiteral(inferHere(fallback));
              if (provisional && provisional != types.object()) {
                TypeSystem::Scope provisionalScope = types.pushScope();
                types.bindLocalSymbol(
                    name, types.contract("builtins.dict", {key, provisional}));
                mlir::Type seeded =
                    types.widenLiteral(inferHere(stored));
                if (seeded && seeded != types.object())
                  deferredElement = seeded;
              }
            }
          noteMaybeContainer(element, stored, note);
        }
    }
    // ⭐ A `for` target is BOUND while its body is scanned. The seed is
    // usually the loop variable -- `for i in range(3): xs.append(i)` and
    // `for w in words: d[w] = 1` are the two commonest shapes -- and nothing
    // has bound it yet at the point of the empty literal, so without this the
    // scan infers `object` from a name that is plainly an int or a str.
    std::optional<TypeSystem::Scope> loopScope;
    if (node.kind == "For" || node.kind == "AsyncFor") {
      const parser::Node *loopTarget = ast::node(node, "target");
      if (loopTarget && loopTarget->kind == "Name")
        if (mlir::Type item =
                types.iterationElementType(ast::node(node, "iter"))) {
          loopScope.emplace(types.pushScope());
          types.bindLocalSymbol(ast::nameSpelling(*loopTarget), item);
        }
    }
    for (const parser::Field &field : node.fields) {
      if (const auto *child = std::get_if<parser::NodePtr>(&field.value)) {
        if (*child)
          recurse(**child, recurse);
        continue;
      }
      if (const auto *children =
              std::get_if<std::vector<parser::NodePtr>>(&field.value)) {
        // ⭐ AND A NAME THE SUITE BINDS, for the reason above one line on: the
        // seed is often computed just before the append, and a scan that has
        // not bound it infers `object` from a name that is plainly an int --
        // which then decides the container's element type:
        //
        //     for i in range(3):
        //         k = i * 10
        //         fs.append(lambda: k)   # element was Callable[[], object]
        //
        // Bound AFTER the statement is scanned, so an assignment does not see
        // itself, and only for the rest of THIS suite.
        std::optional<TypeSystem::Scope> suiteScope;
        for (auto [childIndex, child] : llvm::enumerate(*children)) {
          if (!child)
            continue;
          recurse(*child, recurse);
          bindWhatItLeaves(children, childIndex, suiteScope);
        }
      }
    }
  };

  // ⭐ AND THE REST OF EVERY SUITE THIS ONE SITS INSIDE. The scan read the
  // current suite only, so a container decided inside a REGION never saw the
  // operations that seed it -- which are usually one suite out, because that
  // is where the two branches meet again:
  //
  //     def f(c: bool) -> int:
  //         if c:
  //             xs = []
  //         else:
  //             xs = [2]
  //         xs.append(1)          # the seed, in the enclosing suite
  //         return xs[0] + 1
  //     # static type !py.union<list[int], list[object]> ...
  //
  // The `try:` spelling of the same three lines failed the same way. This is
  // the walk `nameMayBeReadAfterCurrentStatement` already makes for the same
  // reason -- "after this statement" means the rest of this suite AND the rest
  // of every suite it sits inside -- and it stops at the same floor, because
  // the same name in the enclosing FUNCTION is a different binding.
  //
  // ⛔ Disagreement still decides: two seeds one suite apart that say
  // different things leave the element erased, which is where it started.
  std::optional<TypeSystem::Scope> remainderScope;
  auto scanRemainder = [&](const SuiteCursor &cursor) {
    const std::vector<parser::NodePtr> *suite = cursor.suite;
    if (!suite)
      return;
    for (std::size_t index = cursor.from; index < cursor.end(); ++index)
      if (cursor.keeps(index) && (*suite)[index]) {
        visit(*(*suite)[index], visit);
        bindWhatItLeaves(suite, index, remainderScope);
      }
  };
  // ⛔ To a FIXPOINT and before any scanning: `a = stack.pop()` has to be known
  // derived by the time `b = a` is read, and the two can be written in either
  // order across the suites this walks. Three rounds is a bound, not a count --
  // each round can only add names, and a chain longer than that leaves the
  // extra links reading as ordinary locals, which is what they did before.
  if (subscriptDepth > 0) {
    // The value shapes that BIND one: a subscript of the container one level
    // out, or its `setdefault`/`get`, which is the same read written as a call.
    auto bindsTheContainer = [&](const parser::Node *value) -> bool {
      if (!value)
        return false;
      if (value->kind == "Subscript")
        return namesTheContainerAt(ast::node(*value, "value"),
                                   subscriptDepth - 1);
      if (value->kind != "Call")
        return false;
      const parser::Node *callee = ast::node(*value, "func");
      if (!callee || callee->kind != "Attribute")
        return false;
      std::optional<std::string_view> method = ast::string(*callee, "attr");
      if (!method || (*method != "setdefault" && *method != "get"))
        return false;
      return namesTheContainerAt(ast::node(*callee, "value"),
                                 subscriptDepth - 1);
    };
    std::function<void(const parser::Node &)> collectAliases =
        [&](const parser::Node &node) {
          if (node.kind == "FunctionDef" || node.kind == "AsyncFunctionDef" ||
              node.kind == "ClassDef")
            return;
          if (node.kind == "Assign") {
            const auto *targets = ast::nodeList(node, "targets");
            if (targets && targets->size() == 1 && targets->front() &&
                targets->front()->kind == "Name" &&
                bindsTheContainer(ast::node(node, "value")))
              subscriptAliases.insert(ast::nameSpelling(*targets->front()));
          }
          for (const parser::Field &field : node.fields) {
            if (const auto *child = std::get_if<parser::NodePtr>(&field.value)) {
              if (*child)
                collectAliases(**child);
              continue;
            }
            if (const auto *children =
                    std::get_if<std::vector<parser::NodePtr>>(&field.value))
              for (const parser::NodePtr &child : *children)
                if (child)
                  collectAliases(*child);
          }
        };
    for (const SuiteCursor &cursor : suites) {
      if (!cursor.suite)
        continue;
      for (std::size_t index = cursor.from; index < cursor.end();
           ++index)
        if (cursor.keeps(index) && (*cursor.suite)[index])
          collectAliases(*(*cursor.suite)[index]);
    }
  }
  {
    std::function<void(const parser::Node &)> collect =
        [&](const parser::Node &node) {
          if (node.kind == "FunctionDef" || node.kind == "AsyncFunctionDef" ||
              node.kind == "ClassDef")
            return;
          if (node.kind == "Assign") {
            const auto *targets = ast::nodeList(node, "targets");
            const parser::Node *value = ast::node(node, "value");
            if (value && targets && targets->size() == 1 &&
                targets->front() && targets->front()->kind == "Name" &&
                mentionsName(value, mentionsName))
              derivedFromName.insert(ast::nameSpelling(*targets->front()));
          }
          for (const parser::Field &field : node.fields) {
            if (const auto *child = std::get_if<parser::NodePtr>(&field.value)) {
              if (*child)
                collect(**child);
              continue;
            }
            if (const auto *children =
                    std::get_if<std::vector<parser::NodePtr>>(&field.value))
              for (const parser::NodePtr &child : *children)
                if (child)
                  collect(*child);
          }
        };
    for (unsigned round = 0; round < 3; ++round) {
      std::size_t before = derivedFromName.size();
      for (const SuiteCursor &cursor : suites) {
        if (!cursor.suite)
          continue;
        for (std::size_t index = cursor.from; index < cursor.end();
             ++index)
          if (cursor.keeps(index) && (*cursor.suite)[index])
            collect(*(*cursor.suite)[index]);
      }
      if (derivedFromName.size() == before)
        break;
    }
  }

  for (const SuiteCursor &cursor : suites)
    scanRemainder(cursor);

  // The provisional seed answers only when nothing else did: a store that does
  // not mention the name is better evidence, and two of those that disagree is
  // still a disagreement.
  if (!disagreed && !element)
    element = deferredElement;
  if (!disagreed && !element && !emptyFallbackDisagreed)
    element = emptyFallback;
  if (disagreed || !element)
    return {};
  if (isMapping)
    return key ? types.contract("builtins.dict", {key, element}) : mlir::Type();
  if (literalKind == "Set")
    return types.contract("builtins.set", {element});
  return types.contract("builtins.list", {element});
}
} // namespace lython::emitter
