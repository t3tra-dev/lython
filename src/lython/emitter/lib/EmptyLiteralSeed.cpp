#include "EmptyLiteralSeed.h"

#include "AstAccess.h"
#include "PrimitiveTypes.h"

#include "llvm/ADT/SmallVector.h"

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
  for (std::size_t index = cursor.from; index < cursor.suite->size(); ++index)
    if ((*cursor.suite)[index])
      walk(*(*cursor.suite)[index], walk);
}

} // namespace

mlir::Type emptyLiteralSeedTypeIn(const TypeSystem &types,
                                  llvm::StringRef name,
                                  llvm::StringRef literalKind,
                                  llvm::ArrayRef<SuiteCursor> suites,
                                  const llvm::StringMap<mlir::Type> *localSymbols) {
  if (suites.empty() || !suites.front().suite ||
      suites.front().from > suites.front().suite->size())
    return {};
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
  auto mentionsName = [&](const parser::Node *node, auto &&recurse) -> bool {
    if (!node)
      return false;
    if (node->kind == "Name" &&
        llvm::StringRef(ast::nameSpelling(*node)) == name)
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
        if (receiver && receiver->kind == "Name" && method &&
            *method == "get" &&
            llvm::StringRef(ast::nameSpelling(*receiver)) == name)
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

  auto visit = [&](const parser::Node &node, auto &&recurse) -> void {
    if (disagreed)
      return;
    // A nested function has its own binding order; see the note above.
    if (node.kind == "FunctionDef" || node.kind == "AsyncFunctionDef" ||
        node.kind == "ClassDef")
      return;
    if (node.kind == "Expr") {
      const parser::Node *call = ast::node(node, "value");
      if (call && call->kind == "Call") {
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
          if (receiver && receiver->kind == "Name" &&
              llvm::StringRef(ast::nameSpelling(*receiver)) == name &&
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
              noteExpr(element, args->front().get(), note);
            else if (args->size() == 1 &&
                     (*method == "extend" || *method == "update"))
              noteIterableElements(args->front().get());
            else if (args->size() == 2 && *method == "insert" && !isMapping)
              noteExpr(element, (*args)[1].get(), note);
            else if (args->size() == 2 && *method == "setdefault" &&
                     isMapping) {
              noteExpr(key, args->front().get(), note);
              noteExpr(element, (*args)[1].get(), note);
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
          if (!target || target->kind != "Name" ||
              llvm::StringRef(ast::nameSpelling(*target)) != name)
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
          if (!receiver || receiver->kind != "Name" ||
              llvm::StringRef(ast::nameSpelling(*receiver)) != name)
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
          noteExpr(element, stored, note);
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
        for (const parser::NodePtr &child : *children) {
          if (!child)
            continue;
          recurse(*child, recurse);
          // A nested def binds its name here too. The recursion above declines
          // to look INSIDE one (its own binding order), which is a different
          // question from what the name it leaves behind is worth:
          // `for i in ...: def f() -> int: return i` then `fs.append(f)` had
          // the element decided from a name the scan had no type for.
          if (child->kind == "FunctionDef" ||
              child->kind == "AsyncFunctionDef") {
            auto nestedName = ast::string(*child, "name");
            if (!nestedName)
              continue;
            FunctionSignature nested = types.functionSignature(*child);
            if (!nested.publicCallable)
              continue;
            if (!suiteScope)
              suiteScope.emplace(types.pushScope());
            types.bindLocalSymbol(*nestedName, nested.publicCallable);
            continue;
          }
          if (child->kind != "Assign")
            continue;
          const auto *assignTargets = ast::nodeList(*child, "targets");
          const parser::Node *assigned = ast::node(*child, "value");
          if (!assignTargets || assignTargets->size() != 1 ||
              !assignTargets->front() ||
              assignTargets->front()->kind != "Name" || !assigned)
            continue;
          mlir::Type bound = types.widenLiteral(inferHere(assigned));
          if (!bound || bound == types.object())
            continue;
          if (!suiteScope)
            suiteScope.emplace(types.pushScope());
          types.bindLocalSymbol(ast::nameSpelling(*assignTargets->front()),
                                bound);
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
  auto scanRemainder = [&](const std::vector<parser::NodePtr> *suite,
                           std::size_t from) {
    if (!suite)
      return;
    for (std::size_t index = from; index < suite->size(); ++index)
      if ((*suite)[index])
        visit(*(*suite)[index], visit);
  };
  for (const SuiteCursor &cursor : suites)
    scanRemainder(cursor.suite, cursor.from);

  // The provisional seed answers only when nothing else did: a store that does
  // not mention the name is better evidence, and two of those that disagree is
  // still a disagreement.
  if (!disagreed && !element)
    element = deferredElement;
  if (disagreed || !element)
    return {};
  if (isMapping)
    return key ? types.contract("builtins.dict", {key, element}) : mlir::Type();
  if (literalKind == "Set")
    return types.contract("builtins.set", {element});
  return types.contract("builtins.list", {element});
}
} // namespace lython::emitter
