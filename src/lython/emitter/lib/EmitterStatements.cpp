#include "AstSynth.h"
#include "EmitterCore.h"

#include "llvm/ADT/ScopeExit.h"
#include "EmitterOps.h" // IWYU pragma: keep
#include "EmitterPyOps.h"
#include "EmitterSupport.h"

#include "AstAccess.h"
#include "Contracts.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/SaveAndRestore.h"

#include <cstddef>
#include <string>

namespace lython::emitter {

void ModuleEmitter::emitStatements(
    const std::vector<parser::NodePtr> *statements, bool skipDeclarations) {
  if (!statements)
    return;
  // The suite being emitted, so an EMPTY container literal can look forward
  // for the operations that seed it (`emptyLiteralSeedType`). Saved and
  // restored because suites nest.
  llvm::SaveAndRestore<const std::vector<parser::NodePtr> *> savedSuite(
      currentSuite, statements);
  llvm::SaveAndRestore<std::size_t> savedSuiteIndex(currentSuiteIndex, 0);
  suiteStack.push_back({statements, 0});
  auto popSuite = llvm::make_scope_exit([&] { suiteStack.pop_back(); });
  for (const parser::NodePtr &statement : *statements) {
    if (insertionBlockTerminated(builder))
      break;
    ++currentSuiteIndex;
    suiteStack.back().second = currentSuiteIndex;
    if (statement && (!skipDeclarations || !isTopLevelDecl(*statement)))
      emitStatement(*statement);
    else if (statement && skipDeclarations) {
      // The class contract was declared up front, but its attribute
      // initializers evaluate here -- at the class statement's position in
      // module flow, like CPython's class-body execution.
      if (statement->kind == "ClassDef") {
        emitClassAttrInitializers(*statement);
        emitInitSubclassHook(*statement);
      }
      // A skipped module-level def still EXECUTES here in CPython terms:
      // its non-constant defaults evaluate at this spot, once, into their
      // module-lifetime cells (R6). Not ClassDef-exclusive: method defaults
      // registered under a class statement flow through the same cells.
      emitPendingDefaultCells(*statement);
      if (statement->kind == "FunctionDef" ||
          statement->kind == "AsyncFunctionDef")
        applyFunctionDecorators(*statement);
    }
  }
}


// ⭐ `xs = []` LEARNS ITS ELEMENT TYPE FROM WHAT SEEDS IT. An empty literal has
// nothing of its own to infer from, so it typed as `list[object]` and the very
// next line stopped compiling:
//
//     xs = []
//     xs.append(1)
//     print(xs[0] + 1)   # 'builtins.object' does not provide '__add__'
//
// which is the most common way a Python program builds a list, a dict or a
// set. The annotated spelling (`xs: list[int] = []`) always worked, and so did
// an empty literal assigned into a name that already had a declared type --
// this is the same expectation, read from the operations that FOLLOW instead
// of from an annotation.
//
// ⛔ EVERY SEED MUST AGREE, and that is what keeps this from being a new
// refusal. `xs = []; xs.append(1); xs.append("s")` compiles today at
// `list[object]` and CPython allows it; committing to `list[int]` from the
// first append would reject the second. Disagreement -- or no seed at all --
// leaves the literal exactly as it was.
//
// ⛔ Why a syntactic forward scan and not the HM fixpoint, which is where an
// empty literal's element type WANTS to be a unification variable: the emitter
// decides the literal's type when it emits it, and the statements after it
// have not been emitted. Making the variable survive to the point of use is a
// representation change in the bundle, not an inference change. The scan is a
// bounded approximation of the answer that change would give, and it declines
// wherever it is unsure.
//
// ⛔ Nested FUNCTION bodies are not scanned. A `def` that appends to a name
// this suite binds is reading a closure variable whose seeding order this walk
// does not know, and the scan has no way to say "later" about it.
// Bind every `NAME = <constant>` in the rest of the current suite into the
// scope the caller has just pushed. A constant's type depends on nothing, so
// binding it early is the same answer the walk reaches later; anything that is
// not a constant is left alone rather than guessed at. Both forward scans need
// it, because both are asked about an expression that mentions a local the
// walk has not reached.
void ModuleEmitter::preBindSuiteConstants() {
  if (!currentSuite || currentSuiteIndex > currentSuite->size())
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
  for (std::size_t index = currentSuiteIndex; index < currentSuite->size();
       ++index)
    if ((*currentSuite)[index])
      walk(*(*currentSuite)[index], walk);
}

// ⭐ `f = lambda v: v * 2` LEARNS ITS PARAMETERS FROM THE CALL. An unannotated
// lambda has no type of its own -- "lambda requires a Callable annotation
// because its type contains unresolved Unknown" -- and the callee-position
// repair reads the parameter types off the arguments, which an ASSIGNMENT does
// not have. The call does, and it is in the same suite:
//
//     f = lambda v: v * 2
//     print(f(5))
//
// so the forward scan that decides an empty literal's element type
// (`emptyLiteralSeedType`) answers this one too, on the same terms.
//
// ⛔ Every call of the name must agree on the argument types, and there must be
// at least one. Disagreeing calls leave the lambda unannotated, which is the
// refusal it already had -- a lambda emitted at one call's types and used at
// another's would be a wrong program rather than a refused one.
//
// ⛔ Nested FUNCTION bodies are not scanned, for the reason the literal scan
// gives: a `def` that calls the name reads a closure variable whose binding
// order this walk does not know.
py::CallableType
ModuleEmitter::lambdaCallSeedContract(llvm::StringRef name,
                                      const parser::Node &lambda) {
  if (!currentSuite || currentSuiteIndex > currentSuite->size())
    return {};
  llvm::SmallVector<mlir::Type, 4> agreed;
  bool seen = false;
  bool disagreed = false;
  // The same pre-binding the literal scan needs, for the same reason: the call
  // usually mentions a local the walk has not reached. `step = lambda v: v + 1`
  // is followed by `total = 0` and then `step(total)`, and without `total`
  // bound the argument infers as object and the scan declines.
  TypeSystem::Scope seedScope = types.pushScope();
  preBindSuiteConstants();
  auto visit = [&](const parser::Node &node, auto &&recurse) -> void {
    if (disagreed)
      return;
    if (node.kind == "FunctionDef" || node.kind == "AsyncFunctionDef" ||
        node.kind == "ClassDef" || node.kind == "Lambda")
      return;
    if (node.kind == "Call") {
      const parser::Node *callee = ast::node(node, "func");
      if (callee && callee->kind == "Name" &&
          llvm::StringRef(ast::nameSpelling(*callee)) == name) {
        const auto *args = ast::nodeList(node, "args");
        const auto *keywords = ast::nodeList(node, "keywords");
        if (!args || (keywords && !keywords->empty())) {
          disagreed = true;
          return;
        }
        llvm::SmallVector<mlir::Type, 4> here;
        for (const parser::NodePtr &argument : *args) {
          if (!argument || argument->kind == "Starred") {
            disagreed = true;
            return;
          }
          mlir::Type argumentType =
              types.widenLiteral(types.inferExpr(argument.get()));
          if (!argumentType || argumentType == types.object()) {
            disagreed = true;
            return;
          }
          here.push_back(argumentType);
        }
        if (!seen) {
          agreed = here;
          seen = true;
        } else if (agreed != here) {
          disagreed = true;
          return;
        }
      }
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
  for (std::size_t index = currentSuiteIndex; index < currentSuite->size();
       ++index)
    if ((*currentSuite)[index])
      visit(*(*currentSuite)[index], visit);
  if (disagreed || !seen || agreed.empty())
    return {};
  py::CallableType parameters =
      py::CallableType::get(&context, agreed, {}, {}, {}, {});
  mlir::Type resultType =
      types.functionSignature(lambda, std::nullopt, parameters).resultType;
  if (!resultType)
    return {};
  return py::CallableType::get(&context, agreed, {}, {}, {}, {resultType});
}

// ⭐ `best = None` IS THE START OF AN ACCUMULATOR, not a declaration that the
// name is None forever. The idiom
//
//     best = None
//     for v in xs:
//         if best is None or v > best:
//             best = v
//
// bound `best` at `literal<None>`, so the comparison inside the `or` -- the
// half the guard exists to make safe -- was `int.__gt__(None)` and the whole
// program was refused. The annotated spelling (`best: int | None = None`)
// always worked, and this is the same expectation read from the bindings that
// follow instead of from an annotation.
//
// ⛔ A binding whose type cannot be settled leaves the name alone. Widening it
// to a union on a guess would make every later read need a narrowing the
// source does not have, which is a worse answer than the None it starts with.
static bool isNoneConstantNode(const parser::Node *node) {
  return node && node->kind == "Constant" && ast::isNoneField(*node, "value");
}

mlir::Type ModuleEmitter::noneSeedUnionType(llvm::StringRef name) {
  if (!currentSuite || currentSuiteIndex > currentSuite->size())
    return {};
  TypeSystem::Scope seedScope = types.pushScope();
  preBindSuiteConstants();
  llvm::SmallVector<mlir::Type, 4> members;
  bool opaque = false;
  // A loop TARGET is the commonest right-hand side here (`best = v`), and it
  // is not in scope at the point the seed is decided -- the loop has not been
  // emitted. Its type is the iterable's element type, which is known.
  const std::function<void(const parser::Node *)> *collectInline = nullptr;
  std::function<void(const parser::Node *)> bindLoopTargets =
      [&](const parser::Node *node) {
        if (!node || opaque)
          return;
        if (collectInline)
          (*collectInline)(node);
        if (node->kind == "FunctionDef" || node->kind == "AsyncFunctionDef" ||
            node->kind == "ClassDef" || node->kind == "Lambda")
          return;
        if (node->kind == "For" || node->kind == "AsyncFor")
          if (const parser::Node *target = ast::node(*node, "target"))
            if (target->kind == "Name")
              if (mlir::Type element =
                      types.iterationElementType(ast::node(*node, "iter")))
                types.bindSymbol(ast::nameSpelling(*target), element);
        // ⭐ AND THE ORDINARY LOCALS ALONG THE WAY. The binding this scan is
        // reading is usually `best = item`, where `item` came off the
        // container one line earlier -- inside a body this walk has not
        // emitted. Binding each simple assignment as the forward walk reaches
        // it is the same answer the emission will reach, just sooner; a name
        // whose right-hand side does not infer is left alone.
        if (node->kind == "Assign")
          if (const auto *targets = ast::nodeList(*node, "targets"))
            if (targets->size() == 1 && targets->front() &&
                targets->front()->kind == "Name" &&
                llvm::StringRef(ast::nameSpelling(*targets->front())) != name)
              if (const parser::Node *value = ast::node(*node, "value")) {
                mlir::Type bound = types.widenLiteral(types.inferExpr(value));
                if (bound && !py::isPyObjectType(bound))
                  types.bindSymbol(ast::nameSpelling(*targets->front()), bound);
              }
        // ⭐ AND THE NARROWING THE GUARD AROUND THEM ESTABLISHES. The walk of
        // an optional linked structure is `while cur is not None: ... best =
        // cur.value`, and `cur` is `Node | None` here -- so the right-hand
        // side infers as an attribute of a union and the seed is abandoned for
        // a program whose guard makes it exact. Only `is not None` on a bare
        // name, which is the guard this shape is written with.
        if (node->kind == "While" || node->kind == "If") {
          llvm::StringRef guarded;
          if (const parser::Node *test = ast::node(*node, "test"))
            if (test->kind == "Compare")
              if (const auto *ops = ast::nodeList(*test, "ops");
                  ops && ops->size() == 1 && ops->front() &&
                  ast::isOperator(ops->front().get(), "IsNot"))
                if (const auto *comparators =
                        ast::nodeList(*test, "comparators");
                    comparators && comparators->size() == 1 &&
                    isNoneConstantNode(comparators->front().get()))
                  if (const parser::Node *left = ast::node(*test, "left");
                      left && left->kind == "Name")
                    guarded = ast::nameSpelling(*left);
          if (!guarded.empty()) {
            TypeSystem::Scope guardScope = types.pushScope();
            if (auto bound = types.lookupSymbol(guarded))
              if (auto unionType =
                      mlir::dyn_cast_if_present<py::UnionType>(*bound))
                if (unionType.isOptional())
                  types.bindSymbol(guarded, unionType.getOptionalPayloadType());
            if (const auto *body = ast::nodeList(*node, "body"))
              for (const parser::NodePtr &child : *body)
                bindLoopTargets(child.get());
          } else if (const auto *body = ast::nodeList(*node, "body")) {
            for (const parser::NodePtr &child : *body)
              bindLoopTargets(child.get());
          }
          if (const auto *orelse = ast::nodeList(*node, "orelse"))
            for (const parser::NodePtr &child : *orelse)
              bindLoopTargets(child.get());
          return;
        }
        for (const parser::Field &field : node->fields) {
          if (const auto *child = std::get_if<parser::NodePtr>(&field.value))
            bindLoopTargets(child->get());
          else if (const auto *children =
                       std::get_if<std::vector<parser::NodePtr>>(&field.value))
            for (const parser::NodePtr &child : *children)
              bindLoopTargets(child.get());
        }
      };
  // Per NODE: the walk above drives the traversal, so this only answers about
  // the statement it is handed.
  std::function<void(const parser::Node *)> collectOne =
      [&](const parser::Node *node) {
        if (!node || opaque)
          return;
        llvm::StringSet<> written;
        collectAssignedNames(node, written);
        if (!written.contains(name))
          return;
        if (node->kind == "Assign") {
          if (const auto *targets = ast::nodeList(*node, "targets"))
            for (const parser::NodePtr &target : *targets)
              if (target && target->kind == "Name" &&
                  llvm::StringRef(ast::nameSpelling(*target)) == name) {
                const parser::Node *value = ast::node(*node, "value");
                // ⛔ A binding that READS the name says nothing about its
                // type. `acc = acc + v` is a rebinding, and inferring it with
                // the name still at None answered `object` and threw the whole
                // seed away -- which is the running-total idiom.
                if (value && containsNameLoad(value, name))
                  continue;
                mlir::Type bound =
                    value ? types.widenLiteral(types.inferExpr(value))
                          : mlir::Type();
                // ⛔ ANOTHER `= None` CONTRIBUTES NOTHING AND IS NOT A
                // FAILURE. `cur = None` inside the loop is how a scan starts
                // the next group, and reading it as an unusable type gave up
                // on the whole seed -- for the shape the seed is most needed
                // in.
                if (bound && isNoneTypeLike(bound))
                  continue;
                if (!bound || py::isPyObjectType(bound) ||
                    !mlir::isa<py::ContractType>(bound)) {
                  opaque = true;
                  return;
                }
                members.push_back(bound);
              }
          return;
        }
        // A binding this does not recognise (`for name in ...`, `name +=`,
        // `with ... as name`) is not a type this can read off the source.
        llvm::StringSet<> direct;
        collectAssignedNameTargets(ast::node(*node, "target"), direct);
        if (const auto *items = ast::nodeList(*node, "items"))
          for (const parser::NodePtr &item : *items)
            collectAssignedNameTargets(ast::node(*item, "optional_vars"),
                                       direct);
        if (direct.contains(name))
          opaque = true;
      };

  // ⛔ ONE WALK, not two. The narrowing a guard establishes lives only while
  // the walk is inside it, so collecting in a second pass read
  // `best = cur.value` with `cur` back at `Node | None` and abandoned the seed
  // for exactly the programs the guard makes exact.
  collectInline = &collectOne;
  for (std::size_t index = currentSuiteIndex; index < currentSuite->size();
       ++index)
    bindLoopTargets((*currentSuite)[index].get());
  if (opaque || members.empty())
    return {};
  mlir::Type joined = types.join(members);
  if (!joined || py::isPyObjectType(joined) ||
      !mlir::isa<py::ContractType>(joined))
    return {};
  llvm::SmallVector<mlir::Type, 2> memberTypes{joined,
                                               types.literal("None")};
  return py::UnionType::getNormalized(&context, memberTypes);
}

mlir::Type ModuleEmitter::emptyLiteralSeedType(llvm::StringRef name,
                                               llvm::StringRef literalKind) {
  if (!currentSuite || currentSuiteIndex > currentSuite->size())
    return {};
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
  preBindSuiteConstants();

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
    noteType(slot, types.inferExpr(expr));
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
          if (receiver && receiver->kind == "Name" &&
              llvm::StringRef(ast::nameSpelling(*receiver)) == name &&
              method && args &&
              args->size() == 1 && args->front() &&
              (*method == "append" || *method == "add"))
            noteExpr(element, args->front().get(), note);
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
                  types.widenLiteral(types.inferExpr(fallback));
              if (provisional && provisional != types.object()) {
                TypeSystem::Scope provisionalScope = types.pushScope();
                types.bindLocalSymbol(
                    name, types.contract("builtins.dict", {key, provisional}));
                mlir::Type seeded =
                    types.widenLiteral(types.inferExpr(stored));
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
          mlir::Type bound = types.widenLiteral(types.inferExpr(assigned));
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

  for (std::size_t index = currentSuiteIndex; index < currentSuite->size();
       ++index)
    if ((*currentSuite)[index])
      visit(*(*currentSuite)[index], visit);

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

void ModuleEmitter::emitPendingDefaultCells(const parser::Node &statement) {
  auto pending = pendingDefaultCells.find(&statement);
  if (pending == pendingDefaultCells.end())
    return;
  for (const PendingDefaultCell &cell : pending->second) {
    Value value = emitExprExpected(cell.expr.get(), cell.declaredType);
    Value coerced = coerceValue(value, cell.declaredType, statement);
    py::GlobalSetOp::create(builder, loc(statement),
                            builder.getStringAttr(cell.cellName),
                            coerced.value);
  }
}

// The call CPython draws no anchors under: the whole right side of `return
// f(...)` where `f` is a plain name, or of `x = <any call>`. CPython's
// `_should_show_carets` parses the source line back and refuses those two
// statement shapes by hand, on the grounds that the underline would repeat what
// the line already says. Anything else -- `return [f()][0]`, `f()` alone,
// `x.y = f()`, `x: T = f()` -- keeps its anchors.
//
// ⛔ THE CALLEE IS CHECKED FOR `Return` AND NOT FOR `Assign`, WHICH IS NOT A
// TYPO: CPython's `case ast.Return(value=ast.Call())` arm guards on
// `isinstance(statement.value.func, ast.Name)` and its `ast.Assign` arm guards
// only on the target. So `return b.method()` shows anchors and `y =
// b.method()` does not, and a differential against 3.14 catches the difference.
static const parser::Node *anchorlessCallOf(const parser::Node &statement) {
  const parser::Node *value = nullptr;
  bool calleeMustBeName = false;
  if (statement.kind == "Return") {
    value = ast::node(statement, "value");
    calleeMustBeName = true;
  } else if (statement.kind == "Assign") {
    const auto *targets = ast::nodeList(statement, "targets");
    if (!targets || targets->size() != 1 || !targets->front() ||
        targets->front()->kind != "Name")
      return nullptr;
    value = ast::node(statement, "value");
  }
  if (!value || value->kind != "Call")
    return nullptr;
  if (calleeMustBeName) {
    const parser::Node *callee = ast::node(*value, "func");
    if (!callee || callee->kind != "Name")
      return nullptr;
  }
  return value;
}

void ModuleEmitter::emitStatement(const parser::Node &statement) {
  // ⛔ SAVED AND RESTORED RATHER THAN JUST ASSIGNED. A call the emitter inlines
  // emits the callee's statements in the middle of the caller's expression, and
  // each of those would leave its own answer behind -- so by the time the outer
  // call op is built and asks `loc` for its location, the flag would be the
  // last inlined statement's.
  llvm::SaveAndRestore<const parser::Node *> anchorless(
      anchorlessCall, anchorlessCallOf(statement));
  if (statement.kind == "Expr") {
    emitExpr(ast::node(statement, "value"));
  } else if (statement.kind == "Import") {
    bindImportStatement(statement, /*diagnoseUnsupported=*/true);
  } else if (statement.kind == "ImportFrom") {
    bindImportStatement(statement, /*diagnoseUnsupported=*/true);
  } else if (statement.kind == "Assign") {
    const parser::Node *rhs = ast::node(statement, "value");
    // ⭐ `a, b = b, a + b` BUILDS NO TUPLE. Both sides are written out here, so
    // the general path's "materialize the right, then index it once per
    // target" makes an object whose whole life is the statement -- and in a
    // loop that object is what the ownership placement cannot resolve:
    //
    //     a, b = 0, 1
    //     while i < n:
    //         a, b = b, a + b     # "operand #0 does not dominate this use"
    //                             # or "released owned resource ... is used
    //                             #     after release" inside a function
    //
    // which is the fibonacci idiom and every loop-carried swap with it. The
    // same statement outside a loop worked, and the three-statement spelling
    // with an explicit temporary always did.
    //
    // Python evaluates the WHOLE right side before assigning any target, so
    // emitting every element first and then assigning is the semantics, not an
    // approximation -- it is also what CPython's ROT_TWO does with no tuple.
    //
    // ⛔ Arity must match exactly and neither side may be starred: `a, b = xs`
    // has no elements here to pair, and `a, *rest = 1, 2, 3` takes a
    // statically unknown number. Both fall through to the general path, which
    // is where the arity check and the starred refusal live.
    //
    // ⛔ AND EVERY TARGET MUST BE A BARE NAME, which is not tidiness. A
    // subscript target LEAKS on this path -- measured, 2 allocations / 104 B
    // for `grid[0] = 3; grid[1] = 4; grid[0], grid[1] = grid[1], grid[0]`,
    // against 0 through the tuple. The tuple is doing ownership work there:
    // it takes a reference to each element it materializes and gives it up
    // when it dies, so the element read out of it arrives at the store owned.
    // Handing the READ's borrow straight to the store instead leaves a
    // reference nobody accounts for. A bare name binds rather than stores, so
    // it has no such edge -- and the loop-carried swap of locals is the whole
    // defect. Widening this needs the store side to take the reference the
    // tuple used to.
    if (rhs && (rhs->kind == "Tuple" || rhs->kind == "List")) {
      const auto *targets = ast::nodeList(statement, "targets");
      const auto *sources = ast::nodeList(*rhs, "elts");
      if (targets && targets->size() == 1 && targets->front() &&
          (targets->front()->kind == "Tuple" ||
           targets->front()->kind == "List") &&
          sources) {
        const auto *elts = ast::nodeList(*targets->front(), "elts");
        auto noneStarred = [](const std::vector<parser::NodePtr> &nodes) {
          return llvm::all_of(nodes, [](const parser::NodePtr &node) {
            return node && node->kind != "Starred";
          });
        };
        auto allNames = [](const std::vector<parser::NodePtr> &nodes) {
          return llvm::all_of(nodes, [](const parser::NodePtr &node) {
            return node && node->kind == "Name";
          });
        };
        if (elts && elts->size() == sources->size() && !elts->empty() &&
            allNames(*elts) && noneStarred(*sources)) {
          llvm::SmallVector<Value, 4> parts;
          parts.reserve(sources->size());
          for (const parser::NodePtr &source : *sources)
            parts.push_back(emitExpr(source.get()));
          for (auto [index, elt] : llvm::enumerate(*elts))
            emitAssignTarget(*elt, parts[index]);
          return;
        }
      }
    }
    Value value{{}, {}};
    bool emittedWithContext = false;
    if (rhs && rhs->kind == "Lambda") {
      if (const auto *targets = ast::nodeList(statement, "targets")) {
        if (targets->size() == 1 && targets->front() &&
            targets->front()->kind == "Name") {
          llvm::StringRef name = ast::nameSpelling(*targets->front());
          py::CallableType expectedCallable;
          if (auto expectedType = types.lookupSymbol(name))
            expectedCallable =
                mlir::dyn_cast_if_present<py::CallableType>(*expectedType);
          if (!expectedCallable)
            expectedCallable = lambdaCallSeedContract(name, *rhs);
          if (expectedCallable) {
            value = emitLambda(*rhs, expectedCallable);
            emittedWithContext = true;
          }
        }
      }
    }
    // ⭐ A store into a declared field knows what type it wants, so the value
    // is emitted against it.
    //
    //     class Acc:
    //         xs: list[int]
    //         def __init__(self) -> None:
    //             self.xs = []
    //
    //     attribute value 'list[object]' is not assignable to field
    //     'list[int]'
    //
    // An empty literal has nothing to infer an element type from and comes out
    // as `list[object]`. Writing the annotation inline (`self.xs: list[int] =
    // []`) already works, because AnnAssign passes its annotation down as the
    // expected type -- this is the same expectation, read from the field the
    // target names instead of from an annotation beside it. Same for `{}` into
    // a declared `dict[K, V]`.
    if (!emittedWithContext && rhs) {
      if (const auto *targets = ast::nodeList(statement, "targets"))
        if (targets->size() == 1 && targets->front() &&
            targets->front()->kind == "Attribute")
          if (const parser::Node *object =
                  ast::node(*targets->front(), "value"))
            if (auto attr = ast::string(*targets->front(), "attr")) {
              mlir::Type objectType = types.inferExpr(object);
              if (std::optional<mlir::Type> fieldType =
                      lookupClassField(objectType, *attr)) {
                value = emitExprExpected(rhs, *fieldType);
                emittedWithContext = true;
              }
            }
    }
    // ⭐ An EMPTY container literal into a name that already has a declared
    // type takes that type, the same way the field store above does. It has
    // nothing of its own to infer an element type from, so it came out as
    // `list[object]`, and in the canonical Optional idiom
    //
    //     def f(xs: list[int] | None = None) -> int:
    //         if xs is None:
    //             xs = []
    //
    // the branch join was `list[int] | list[object] | None`, which nothing
    // accepts.
    //
    // ⛔ Empty only: a literal WITH elements is what the rebinding says it
    // is. `xs = [1, 2]` where xs was `list[str]` must become `list[int]`, not
    // be pushed at the old element type.
    if (!emittedWithContext && rhs) {
      bool emptyLiteral =
          (rhs->kind == "List" || rhs->kind == "Tuple" || rhs->kind == "Set") &&
          [&] {
            const auto *elts = ast::nodeList(*rhs, "elts");
            return !elts || elts->empty();
          }();
      if (!emptyLiteral && rhs->kind == "Dict") {
        const auto *keys = ast::nodeList(*rhs, "keys");
        emptyLiteral = !keys || keys->empty();
      }
      // ⭐ AND THE CONSTRUCTOR SPELLING OF THE SAME THING. `set()` is a Call,
      // not a Set literal, so it never reached the scan and kept `object`:
      //
      //     s = set()
      //     s.add(1)
      //     # a type-erased `object` value cannot be stored in a runtime
      //     # container
      //
      // while `s = {1}` and `xs = []` were both fine. `set()` is the ONLY way
      // to write an empty set, so this is not an alternative spelling for that
      // one -- it is the only one.
      llvm::StringRef constructedKind;
      if (!emptyLiteral && rhs->kind == "Call") {
        const parser::Node *callee = ast::node(*rhs, "func");
        const auto *callArguments = ast::nodeList(*rhs, "args");
        const auto *callKeywords = ast::nodeList(*rhs, "keywords");
        if (callee && callee->kind == "Name" &&
            (!callArguments || callArguments->empty()) &&
            (!callKeywords || callKeywords->empty())) {
          llvm::StringRef calleeName = ast::nameSpelling(*callee);
          if (!programBindsName(calleeName)) {
            if (calleeName == "set")
              constructedKind = "Set";
            else if (calleeName == "list")
              constructedKind = "List";
            else if (calleeName == "dict")
              constructedKind = "Dict";
          }
        }
        emptyLiteral = !constructedKind.empty();
      }
      // The None seed asks the same question of the same suite, and answers
      // for the accumulator idiom rather than for an empty container.
      if (!emptyLiteral && isNoneConstantNode(rhs))
        if (const auto *targets = ast::nodeList(statement, "targets"))
          if (targets->size() == 1 && targets->front() &&
              targets->front()->kind == "Name") {
            llvm::StringRef target = ast::nameSpelling(*targets->front());
            mlir::Type declared = narrowedFromTypes.lookup(target);
            // ⛔ The BINDING's type, not just the symbol table's: a plain
            // assignment binds the value and leaves the symbol alone, so a
            // local that took its union from a seed has it here and nowhere
            // else.
            if (!declared)
              if (auto bound = values.find(target); bound != values.end())
                declared = bound->second.type;
            if (!declared)
              if (auto flow = types.lookupSymbol(target))
                declared = *flow;
            // ⭐ RESETTING AN OPTIONAL TO None KEEPS THE OPTIONAL. `cur =
            // None` inside the loop that also builds `cur` is how every
            // "start a new group" scan is written, and rebinding the name to
            // `literal<None>` there put the next `cur + ch` back at None -- for
            // a local the annotated spelling has always kept as its union.
            auto optionalDeclaration = [&](mlir::Type type) {
              auto unionType = mlir::dyn_cast_if_present<py::UnionType>(type);
              return unionType && unionType.isOptional();
            };
            if (declared && optionalDeclaration(types.widenLiteral(declared))) {
              mlir::Type kept = types.widenLiteral(declared);
              value = coerceValue(emitExprExpected(rhs, kept), kept, statement);
              emittedWithContext = true;
            } else if (!declared || isNoneTypeLike(declared))
              if (mlir::Type seeded = noneSeedUnionType(target)) {
                value = emitExprExpected(rhs, seeded);
                value = coerceValue(value, seeded, statement);
                emittedWithContext = true;
              }
          }
      if (emptyLiteral)
        if (const auto *targets = ast::nodeList(statement, "targets"))
          if (targets->size() == 1 && targets->front() &&
              targets->front()->kind == "Name")
          {
            llvm::StringRef target = ast::nameSpelling(*targets->front());
            // The pre-narrowing type wins: inside `if xs is None:` the flow
            // type of xs IS None, which is the right answer for a read and no
            // constraint at all on a write.
            mlir::Type declared = narrowedFromTypes.lookup(target);
            if (!declared)
              if (auto flow = types.lookupSymbol(target))
                declared = *flow;
            // Nothing declared it: ask what the rest of the suite seeds it
            // with (`emptyLiteralSeedType`).
            //
            // ⛔ "Nothing" includes a container whose element is the erased
            // top. `s = set()` at module scope arrives with `set[object]`
            // already bound -- the module-global collection inferred it from
            // the RHS -- and that is not a declaration, it is the same absence
            // the seed exists to fill. A literal never hit this because an
            // empty one infers nothing to bind.
            auto onlyErasedArguments = [&](mlir::Type type) {
              auto contract =
                  mlir::dyn_cast_if_present<py::ContractType>(type);
              if (!contract || contract.getArguments().empty())
                return false;
              return llvm::all_of(contract.getArguments(),
                                  [&](mlir::Type argument) {
                                    return types.widenLiteral(argument) ==
                                           types.object();
                                  });
            };
            if (!declared || types.widenLiteral(declared) == types.object() ||
                onlyErasedArguments(types.widenLiteral(declared)))
              if (mlir::Type seeded = emptyLiteralSeedType(
                      target,
                      constructedKind.empty() ? llvm::StringRef(rhs->kind)
                                              : constructedKind))
                declared = seeded;
            if (declared) {
              value = emitExprExpected(rhs, declared);
              // ⛔ The CONSTRUCTOR spelling needs the coercion too, and the
              // literal does not. `set()` is a call whose result type is
              // decided by the callee contract, so the expectation reaches the
              // emission and the VALUE still comes back `set[object]`; an
              // empty literal is built at whatever it is handed. AnnAssign has
              // always coerced for exactly this reason, which is why
              // `s: set[int] = set()` worked while `s = set()` did not.
              if (!constructedKind.empty())
                value = coerceValue(value, declared, statement);
              emittedWithContext = true;
            }
          }
    }
    if (!emittedWithContext)
      value = emitExpr(rhs);
    if (const auto *targets = ast::nodeList(statement, "targets"))
      for (const parser::NodePtr &target : *targets)
        emitAssignTarget(*target, value);
  } else if (statement.kind == "AnnAssign") {
    mlir::Type annotated =
        types.annotationType(ast::node(statement, "annotation"));
    if (const parser::Node *rhs = ast::node(statement, "value")) {
      Value raw = emitExprExpected(rhs, annotated);
      Value value = coerceValue(raw, annotated, statement);
      emitAssignTarget(*ast::node(statement, "target"), value);
      return;
    }
    const parser::Node *target = ast::node(statement, "target");
    if (target && target->kind == "Name")
      types.bindSymbol(ast::nameSpelling(*target), annotated);
  } else if (statement.kind == "AugAssign") {
    // Desugar to the equivalent BinOp over shared subtrees so the operator
    // dispatch (and the primitive path) is exactly the `x = x <op> v` one.
    // The previous inline emission hardcoded __add__ for every operator.
    auto sharedSubtree = [&](llvm::StringRef name) -> parser::NodePtr {
      const parser::Field *field = parser::findField(statement, name);
      if (!field || !std::holds_alternative<parser::NodePtr>(field->value))
        return nullptr;
      return std::get<parser::NodePtr>(field->value);
    };
    parser::NodePtr target = sharedSubtree("target");
    parser::NodePtr op = sharedSubtree("op");
    parser::NodePtr rhs = sharedSubtree("value");
    // ⭐ The target subtree is used TWICE below -- once as the left operand of
    // the rewritten binary op and once as the store target -- so anything in
    // it with a side effect ran twice, and the store landed wherever the
    // SECOND evaluation pointed:
    //
    //     a: list[int] = [0, 0, 0]
    //     a[f()] += 1        # f() called twice; stored [0, 0, 1]
    //                        # CPython calls it once and stores [0, 1, 0]
    //
    // The receiver, the index and a slice's bounds are each emitted once here
    // and referenced from both places through a `LyValueRef`. A Name target
    // needs none of this -- re-reading a name has no effect -- which is why
    // the defect only ever showed through a subscript or an attribute.
    std::size_t valueRefStart = pendingValueRefs.size();
    auto shareSubexpression = [&](parser::NodePtr &slotNode) {
      if (!slotNode)
        return;
      Value evaluated = emitExpr(slotNode.get());
      parser::NodePtr ref = parser::makeNode("LyValueRef", statement.range);
      parser::addField(*ref, "slot",
                       static_cast<std::int64_t>(pendingValueRefs.size()));
      pendingValueRefs.push_back(evaluated);
      slotNode = std::move(ref);
    };
    auto shareField = [&](parser::Node &node, llvm::StringRef name) {
      parser::Field *field = parser::findField(node, name);
      if (!field || !std::holds_alternative<parser::NodePtr>(field->value))
        return;
      parser::NodePtr child = std::get<parser::NodePtr>(field->value);
      shareSubexpression(child);
      field->value = std::move(child);
    };
    if (target && (target->kind == "Subscript" || target->kind == "Attribute")) {
      parser::NodePtr shared = parser::makeNode(target->kind, target->range);
      for (const parser::Field &field : target->fields)
        parser::addField(*shared, field.name, field.value);
      const parser::Node *targetSlice =
          shared->kind == "Subscript" ? ast::node(*shared, "slice") : nullptr;
      bool sliceTarget = targetSlice && targetSlice->kind == "Slice";
      // ⛔ A SLICE target keeps its receiver as written. The slice-assignment
      // path requires a named local -- it rebinds the local to the resized
      // list -- and handing it a `LyValueRef` refuses the program outright.
      // A slice receiver is re-read, which is a second evaluation this does
      // not remove; the BOUNDS, which are where an index expression usually
      // sits, are shared.
      if (!sliceTarget)
        shareField(*shared, "value");
      if (shared->kind == "Subscript") {
        const parser::Node *slice = ast::node(*shared, "slice");
        if (slice && slice->kind == "Slice") {
          parser::NodePtr sliceCopy =
              parser::makeNode("Slice", slice->range);
          for (const parser::Field &field : slice->fields)
            parser::addField(*sliceCopy, field.name, field.value);
          shareField(*sliceCopy, "lower");
          shareField(*sliceCopy, "upper");
          shareField(*sliceCopy, "step");
          parser::Field *sliceField = parser::findField(*shared, "slice");
          if (sliceField)
            sliceField->value = std::move(sliceCopy);
        } else {
          shareField(*shared, "slice");
        }
      }
      target = std::move(shared);
    }
    auto releaseValueRefs = llvm::make_scope_exit(
        [&] { pendingValueRefs.resize(valueRefStart); });
    if (!target || !op || !rhs) {
      diagnostics.push_back(parser::Diagnostic{parser::Severity::Error,
                                               statement.range.start,
                                               "malformed augmented assignment"});
      return;
    }
    // ⭐ An augmented assignment whose operator has an in-place dunder must
    // MUTATE the object, not rebind the name: every other alias observes it.
    //
    //     a: list[int] = [1, 2]
    //     b: list[int] = a
    //     b += [3]
    //     print(a)      # printed [1, 2]; CPython prints [1, 2, 3]
    //
    // Desugaring to `b = b + [3]` builds a fresh list, so the mutation was
    // invisible through `a` -- and through the caller when the target was a
    // parameter. CPython's `list.__iadd__` is `extend`, and `dict.__ior__` is
    // `update`; the dict case was already rewritten this way, so this is the
    // same rule stated once for both rather than a second special case.
    struct InPlaceRewrite {
      llvm::StringRef opKind;
      llvm::StringRef contract;
      llvm::StringRef method;
      // Whether the call ANSWERS with the new value. A container's in-place
      // dunder mutates and the name keeps naming the same object; a str's
      // cannot -- it appends into the block it has when it holds the only
      // reference and allocates when it does not, so which object the name ends
      // up on is the call's answer rather than a foregone conclusion.
      bool rebinds;
    };
    static constexpr InPlaceRewrite kInPlaceRewrites[] = {
        {"BitOr", "builtins.dict", "update", false},
        {"Add", "builtins.list", "extend", false},
        // set's four, which were missing and silently rebound a fresh set:
        // `a |= {9}` left every alias of `a` holding the old one.
        {"BitOr", "builtins.set", "update", false},
        {"Sub", "builtins.set", "difference_update", false},
        {"BitAnd", "builtins.set", "intersection_update", false},
        {"BitXor", "builtins.set", "symmetric_difference_update", false},
        // ⭐ `s += x` IS THE ONE PLACE A STR CAN GROW WITHOUT COPYING, and it is
        // a rewrite rather than a dunder because CPython has no `str.__iadd__`
        // either -- it specializes BINARY_OP in the interpreter, on exactly the
        // condition this rewrite encodes: the name is about to be rebound, so
        // the frame is giving up its reference.
        {"Add", "builtins.str", "__ly_iadd__", true},
    };
    // ⭐ A SLICE TARGET TAKES NO IN-PLACE ROUTE. `a[i:j] += [99]` is
    // `a[i:j] = a[i:j] + [99]` in CPython -- a slice ASSIGNMENT -- and reading
    // `a[i:j]` produces a new list, so `a[i:j].extend([99])` extends a copy and
    // the splice disappears. The old lenient inference hid this by answering
    // "int" for the slice, which matched no contract in the table; correcting
    // that answer (TypeSystem.cpp) made `a[bump():3] += [99]` print
    // `[1, 2, 3, 4, 5]` where CPython gives `[1, 2, 3, 99, 4, 5]`, caught by
    // `augmented_assignment_evaluates_once`. This is the condition that note
    // deferred: the route does not want a different TYPE, it wants not to run.
    bool sliceTargetSubscript = false;
    if (target->kind == "Subscript")
      if (const parser::Node *targetSliceNode = ast::node(*target, "slice"))
        sliceTargetSubscript = targetSliceNode->kind == "Slice";
    llvm::StringRef inPlaceMethod;
    bool inPlaceRebinds = false;
    if (!sliceTargetSubscript)
      for (const InPlaceRewrite &rewrite : kInPlaceRewrites)
        if (op->kind == rewrite.opKind &&
            exprHasContract(target.get(), rewrite.contract)) {
          inPlaceMethod = rewrite.method;
          inPlaceRebinds = rewrite.rebinds;
        }
    if (!inPlaceMethod.empty()) {
      parser::NodePtr updateAttr = synth::attribute(target, std::string(inPlaceMethod), statement.range);
      parser::NodePtr updateCall = synth::call(std::move(updateAttr), std::vector<parser::NodePtr>{rhs}, statement.range);
      if (inPlaceRebinds) {
        Value updated = emitExpr(updateCall.get());
        emitAssignTarget(*target, updated);
        return;
      }
      parser::NodePtr updateStatement = synth::exprStmt(std::move(updateCall), statement.range);
      emitStatement(*updateStatement);
      return;
    }
    // ⭐ A user class that defines the IN-PLACE dunder gets it. CPython tries
    // `__iadd__` before `__add__` and REBINDS the result, so a class defining
    // both silently ran the wrong one:
    //
    //     class M:
    //         def __add__(self, o): return M(1)
    //         def __iadd__(self, o): return M(2)
    //     x = M(0); x += M(0)
    //     print(x.v)      # printed 1; CPython prints 2
    //
    // The table above is the same rule for the manifest containers, whose
    // in-place dunder is spelled as a named method; this is the source-class
    // half, and both end in "call it, then bind the target".
    static constexpr struct {
        llvm::StringRef opKind;
        llvm::StringRef method;
    } kInPlaceDunders[] = {
        {"Add", "__iadd__"},       {"Sub", "__isub__"},
        {"Mult", "__imul__"},      {"Div", "__itruediv__"},
        {"FloorDiv", "__ifloordiv__"}, {"Mod", "__imod__"},
        {"Pow", "__ipow__"},       {"LShift", "__ilshift__"},
        {"RShift", "__irshift__"}, {"BitAnd", "__iand__"},
        {"BitOr", "__ior__"},      {"BitXor", "__ixor__"},
        {"MatMult", "__imatmul__"},
    };
    for (const auto &entry : kInPlaceDunders) {
      if (op->kind != entry.opKind || sliceTargetSubscript)
        continue;
      mlir::Type targetType = types.inferExpr(target.get());
      std::optional<MethodBinding> inPlace =
          lookupClassMethod(targetType, entry.method);
      if (!inPlace || !inPlace->method)
        break;
      parser::NodePtr attribute = synth::attribute(target, std::string(entry.method), statement.range);
      parser::NodePtr call = synth::call(std::move(attribute), std::vector<parser::NodePtr>{rhs}, statement.range);
      Value updated = emitExpr(call.get());
      emitAssignTarget(*target, updated);
      return;
    }
    parser::NodePtr binop = parser::makeNode("BinOp", statement.range);
    parser::addField(*binop, "left", target);
    parser::addField(*binop, "op", op);
    parser::addField(*binop, "right", rhs);
    Value value = emitBinary(*binop);
    emitAssignTarget(*target, value);
  } else if (statement.kind == "If") {
    emitIf(statement);
  } else if (statement.kind == "For") {
    emitFor(statement);
  } else if (statement.kind == "While") {
    emitWhile(statement);
  } else if (statement.kind == "AsyncFor") {
    emitAsyncFor(statement);
  } else if (statement.kind == "LyWithEnter") {
    // Synthesized by emitWith: evaluate one manager and call its __enter__,
    // inside the try that guards the managers entered before it.
    const parser::Field *slot = parser::findField(statement, "slot");
    if (slot && std::holds_alternative<std::int64_t>(slot->value)) {
      auto index = static_cast<std::size_t>(std::get<std::int64_t>(slot->value));
      if (index < pendingWithItems.size() && pendingWithItems[index].item)
        emitWithEnter(*pendingWithItems[index].item,
                      pendingWithItems[index].async);
    }
  } else if (statement.kind == "LyWithCleanup") {
    // Synthesized by emitWith: the `finally` body of the implicit try that
    // wraps a `with` block. It carries an index into `activeWithCleanups`
    // rather than an expression, because the manager is an already-emitted
    // SSA value and there is no spelling for one in the AST.
    const parser::Field *slot = parser::findField(statement, "slot");
    if (slot && std::holds_alternative<std::int64_t>(slot->value)) {
      auto index = static_cast<std::size_t>(std::get<std::int64_t>(slot->value));
      if (index < activeWithCleanups.size())
        emitWithCleanup(statement, activeWithCleanups[index]);
    }
  } else if (statement.kind == "With") {
    emitWith(statement, false);
  } else if (statement.kind == "AsyncWith") {
    emitWith(statement, true);
  } else if (statement.kind == "Raise") {
    if (const parser::Node *exception = ast::node(statement, "exc")) {
      // ⭐ `raise E` IS `raise E()`. CPython instantiates a raised CLASS with
      // no arguments, and the walk handed the type object straight to
      // `py.raise`, which asks the runtime for a `.raise` primitive on it:
      //
      //     raise ValueError
      //     # runtime manifest has no .raise primitive
      //
      // `raise ValueError("x")` was always fine, so the refusal was the
      // no-argument spelling only -- which is the one `raise StopIteration`
      // inside a hand-written `__next__` is written in.
      const parser::Node *raised = exception;
      parser::NodePtr constructed;
      if (mlir::isa_and_nonnull<py::TypeType>(
              types.widenLiteral(types.inferExpr(exception))))
        if (const parser::Field *field = parser::findField(statement, "exc"))
          if (const auto *node =
                  std::get_if<parser::NodePtr>(&field->value);
              node && *node) {
            constructed = synth::call(*node, std::vector<parser::NodePtr>{},
                                      statement.range);
            raised = constructed.get();
          }
      Value value = emitExpr(raised);
      mlir::Value cause;
      bool fromNone = false;
      if (const parser::Node *causeNode = ast::node(statement, "cause")) {
        Value causeValue = emitExpr(causeNode);
        auto literal =
            mlir::dyn_cast_if_present<py::LiteralType>(causeValue.type);
        if (literal && literal.getSpelling() == "None") {
          // `raise X from None` suppresses implicit __context__ display;
          // there is no cause object to carry.
          fromNone = true;
        } else if (literal || !causeValue.value) {
          diagnostics.push_back(parser::Diagnostic{
              parser::Severity::Error, causeNode->range.start,
              "raise ... from cause must be an exception instance or None"});
          return;
        } else {
          cause = causeValue.value;
        }
      }
      py::RaiseOp::create(builder, loc(statement), value.value, cause,
                          fromNone);
    } else if (exceptHandlerDepth == 0) {
      // ⭐ A bare `raise` outside every handler has nothing to re-raise.
      //
      // CPython raises `RuntimeError: No active exception to reraise`; this
      // compiler emitted `py.raise.current`, whose runtime arm for an empty
      // slot is a trap -- `raise` on its own aborted with no output. Deciding
      // it HERE and not in the lowering is forced: `py.try`'s regions are
      // flattened before the lowering sees the op, so by then the parent chain
      // is just `func.func`, and a handler re-raise looks identical to this.
      //
      // Whether a handler is running is a question about where the statement
      // sits, which is exactly what this walk knows. A handler that already
      // completed does not count, matching CPython: `try/except/pass` then
      // `raise` raises there too.
      parser::NodePtr message = synth::strConstant(std::string("No active exception to reraise"), statement.range);
      parser::NodePtr callee = synth::name(std::string("RuntimeError"), statement.range);
      parser::NodePtr call = synth::call(std::move(callee), std::vector<parser::NodePtr>{std::move(message)}, statement.range);
      parser::NodePtr synthesized = synth::raiseStmt(std::move(call), statement.range);
      emitStatement(*synthesized);
    } else {
      py::RaiseCurrentOp::create(builder, loc(statement));
    }
  } else if (statement.kind == "FunctionDef" ||
             statement.kind == "AsyncFunctionDef") {
    Value function = emitNestedFunctionDecl(statement);
    if (auto name = ast::string(statement, "name")) {
      llvm::StringRef spelling(name->data(), name->size());
      // ⭐ A `def` INSIDE A REGION WRITES THROUGH ITS SLOT, the same way an
      // assignment to a conditionally bound local does. The slot is made
      // before the region so the scope around it can see the name, and
      // rebinding `values` here instead left it unwritten -- the read after
      // the region then raised NameError for a definition that ran.
      auto bound = values.find(spelling);
      if (bound != values.end() && isCellContract(bound->second.type)) {
        emitCellStore(statement, bound->second, function);
        types.bindSymbol(spelling, cellContentType(bound->second.type));
      } else {
        values[spelling] = function;
        types.bindSymbol(spelling, function.type);
      }
    }
    applyFunctionDecorators(statement);
  } else if (statement.kind == "Return") {
    if (parser::NodePtr fallback = notImplementedFallbackStatement(statement)) {
      emitStatement(*fallback);
      return;
    }
    const parser::Node *returnValue = ast::node(statement, "value");
    Value value = returnValue
                      ? emitExprExpected(returnValue, currentReturnType)
                      : emitExpr(returnValue);
    if (!inlineReturnContexts.empty()) {
      InlineReturnContext &ctx = inlineReturnContexts.back();
      if (ctx.carryResult) {
        Value result = ctx.resultType
                           ? coerceValue(value, ctx.resultType, statement)
                           : value;
        mlir::cf::BranchOp::create(builder, loc(statement), ctx.target,
                                   result.value);
      } else {
        mlir::cf::BranchOp::create(builder, loc(statement), ctx.target);
      }
      return;
    }
    if (currentReturnType) {
      Value result = coerceValue(value, currentReturnType, statement);
      mlir::func::ReturnOp::create(builder, loc(statement), result.value);
    }
  } else if (statement.kind == "Break") {
    if (loopControlContexts.empty() ||
        !loopControlContexts.back().breakTarget) {
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, statement.range.start,
          "break outside a supported loop is not implemented yet"});
      return;
    }
    const LoopControlContext &loop = loopControlContexts.back();
    mlir::cf::BranchOp::create(
        builder, loc(statement), loop.breakTarget,
        loopCarriedBranchOperands(statement, loop, loop.breakTarget));
  } else if (statement.kind == "Continue") {
    if (loopControlContexts.empty() ||
        !loopControlContexts.back().continueTarget) {
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, statement.range.start,
          "continue outside a supported loop is not implemented yet"});
      return;
    }
    const LoopControlContext &loop = loopControlContexts.back();
    mlir::cf::BranchOp::create(
        builder, loc(statement), loop.continueTarget,
        loopCarriedBranchOperands(statement, loop, loop.continueTarget));
  } else if (statement.kind == "Global") {
    // `global NAME, ...`: writes to these names in the current function target
    // the module global. Only module globals we track (int-annotated) are
    // storable; others are accepted silently (no local storage change).
    if (const auto *names = ast::stringList(statement, "names"))
      for (const std::string &name : *names)
        currentGlobalDecls.insert(name);
    return;
  } else if (statement.kind == "Delete") {
    emitDelete(statement);
  } else if (statement.kind == "Nonlocal") {
    // The enclosing function already promoted these locals to shared cells
    // (nonlocalBoxedNames) and this function captured the cell instances;
    // reads/writes dispatch on the cell binding, so the declaration itself
    // only validates that each target resolved to a cell.
    if (const auto *names = ast::stringList(statement, "names"))
      for (const std::string &name : *names) {
        auto found = values.find(name);
        if (found == values.end() ||
            !isCellContract(found->second.type)) {
          diagnostics.push_back(parser::Diagnostic{
              parser::Severity::Error, statement.range.start,
              "no binding for nonlocal '" + name +
                  "' found (the target must be assigned in an enclosing "
                  "function before this definition)"});
        }
      }
    return;
  } else if (statement.kind == "Assert") {
    // ⭐ `assert t, m` IS `if not t: raise AssertionError(m)`, so it is written
    // as that and nothing new reaches the dialect. Both halves already work:
    // `raise AssertionError` instantiates a raised class with no arguments, and
    // `raise AssertionError(m)` is an ordinary raise.
    //
    // ⛔ Why NOT elide them like CPython's -O: -O is a flag Lython does not
    // have, and the default in CPython is that asserts run. An assert that
    // silently did not check would be the opposite of "never silently
    // mis-execute".
    const parser::Field *testField = parser::findField(statement, "test");
    const auto *testNode =
        testField ? std::get_if<parser::NodePtr>(&testField->value) : nullptr;
    if (!testNode || !*testNode) {
      diagnostics.push_back(parser::Diagnostic{parser::Severity::Error,
                                               statement.range.start,
                                               "assert has no test expression"});
      return;
    }
    std::vector<parser::NodePtr> messageArguments;
    if (const parser::Field *messageField = parser::findField(statement, "msg"))
      if (const auto *message =
              std::get_if<parser::NodePtr>(&messageField->value);
          message && *message)
        messageArguments.push_back(*message);
    parser::NodePtr guard = synth::ifStmt(
        synth::notOp(*testNode, statement.range),
        std::vector<parser::NodePtr>{synth::raiseStmt(
            synth::call(synth::name("AssertionError", statement.range),
                        std::move(messageArguments), statement.range),
            statement.range)},
        {}, statement.range);
    emitStatement(*guard);
  } else if (statement.kind == "Pass") {
    return;
  } else if (statement.kind == "Match") {
    emitMatch(statement);
  } else if (statement.kind == "Try") {
    emitTry(statement);
  } else if (statement.kind == "TryStar") {
    emitTryStar(statement);
  } else if (statement.kind == "ClassDef") {
    // ⭐ SAY WHICH STATEMENT AND WHY. A class declared inside a function or
    // another class got the generic "unsupported statement kind 'ClassDef'"
    // and then a second, misleading diagnostic at every use of the name
    // ("unresolved name 'D'") -- which reads as a scoping bug rather than as
    // the one limitation it is.
    //
    // ⛔ Not repaired here: a class is a module-level CONTRACT plus its
    // methods as module-level functions, and hoisting one out of a function
    // means deciding what a method that reads an enclosing local closes over.
    // The nested `def` has an answer for that (its captures ride the function
    // object); a nested class would need the same for every method at once.
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, statement.range.start,
        "a class defined inside a function or another class is not supported: "
        "a class is a module-level contract here, so declare '" +
            std::string(ast::nameSpelling(statement)) + "' at module scope"});
  } else {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, statement.range.start,
        "unsupported statement kind '" + statement.kind + "'"});
  }
}

// `del` (R6): only subscript deletion is representable — variables live in
// static SSA scopes (released at scope exit), and instance attributes are
// fixed storage slots, so both are rejected with an explanation instead of a
// generic unsupported-statement error.
void ModuleEmitter::emitDelete(const parser::Node &statement) {
  const auto *targets = ast::nodeList(statement, "targets");
  if (!targets)
    return;
  for (const parser::NodePtr &target : *targets) {
    if (!target)
      continue;
    if (target->kind == "Subscript") {
      if (const parser::Node *sliceNode = ast::node(*target, "slice");
          sliceNode && sliceNode->kind == "Slice") {
        emitSliceMutation(*target, ast::node(*target, "value"), *sliceNode,
                          "__delslice__", std::nullopt);
        continue;
      }
      Value container = emitExpr(ast::node(*target, "value"));
      Value index = emitExpr(ast::node(*target, "slice"));
      if (tryEmitClassDunder(*target, container, "__delitem__", {index}))
        continue;
      CallInferenceResult inference = types.inferMethodCallWithEvidence(
          container.type, "__delitem__", {index.type});
      if (!requireStaticEvidence(*target, inference))
        continue;
      py::DelItemOp::create(
          builder, loc(*target),
          mlir::FlatSymbolRefAttr::get(&context, "__delitem__"),
          callProtocolFor(inference), container.value, index.value);
      continue;
    }
    if (target->kind == "Name") {
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, target->range.start,
          "`del " + std::string(ast::nameSpelling(*target)) +
              "` is rejected (Lython deviation from CPython): locals are "
              "released when their scope ends, so deleting a variable is "
              "unnecessary"});
      continue;
    }
    if (target->kind == "Attribute") {
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, target->range.start,
          "`del` on an attribute is rejected (Lython deviation from CPython): "
          "instance attributes are fixed storage slots in the static object "
          "layout"});
      continue;
    }
    diagnostics.push_back(
        parser::Diagnostic{parser::Severity::Error, target->range.start,
                           "unsupported del target '" + target->kind + "'"});
  }
}

// Slice assignment / deletion on a list local: both are structural
// mutations (the splice may reallocate the items storage), so they lower
// through the same rebinding bound-method call shape as list.append —
// `__setslice__`/`__delslice__` carry the (start, stop, step, mask) pack the
// slice READ path already uses, plus the replacement list for assignment.
void ModuleEmitter::emitSliceMutation(const parser::Node &target,
                                      const parser::Node *containerNode,
                                      const parser::Node &sliceNode,
                                      llvm::StringRef methodName,
                                      std::optional<Value> payload) {
  bool isAssignment = payload.has_value();
  auto unsupported = [&](llvm::StringRef reason) {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, target.range.start,
        (isAssignment ? llvm::Twine("slice assignment ")
                      : llvm::Twine("slice deletion "))
                .concat(reason)
                .str()});
  };
  if (!containerNode) {
    unsupported("has no target expression");
    return;
  }
  Value container = emitExpr(containerNode);
  // ⭐ A NAME IS NOT REQUIRED. `self.rows[:n] = [9]` and `del self.rows[:n]`
  // were "requires a named local list target (field containers are not
  // supported yet)", and `g[0][:1] = [9]` with them. A list is handle-fronted:
  // the splice writes the new items address THROUGH the handle, so a holder
  // that is a field slot, a class-attribute cell or a container element
  // observes it with nothing to rename -- which is exactly why the LOWERING
  // stopped needing the rebind (Runtime/Ops/CallableOps.cpp). What the rebind
  // still buys where it applies is the demotion of the local's element
  // evidence, so the two shapes stay distinct here rather than collapsing:
  // a name gets the two-result rebinding call, anything else the plain one.
  llvm::StringRef containerName;
  if (containerNode->kind == "Name")
    containerName = ast::nameSpelling(*containerNode);
  bool rebindable =
      !containerName.empty() &&
      isStructuralMutationRebindable(containerName, container.value);

  const parser::Node *lower = ast::node(sliceNode, "lower");
  const parser::Node *upper = ast::node(sliceNode, "upper");
  const parser::Node *step = ast::node(sliceNode, "step");
  auto intConstant = [&](long long value) -> Value {
    std::string text = std::to_string(value);
    mlir::Type type = types.literal(text);
    auto op = py::IntConstantOp::create(builder, loc(target), type,
                                        builder.getStringAttr(text));
    return {op.getResult(), type};
  };
  Value startValue = lower ? emitExpr(lower) : intConstant(0);
  Value stopValue = upper ? emitExpr(upper) : intConstant(0);
  Value stepValue = step ? emitExpr(step) : intConstant(1);
  long long maskBits = (lower ? 1 : 0) | (upper ? 2 : 0);
  Value maskValue = intConstant(maskBits);

  llvm::SmallVector<mlir::Type, 5> argumentTypes{
      startValue.type, stopValue.type, stepValue.type, maskValue.type};
  llvm::SmallVector<Value, 5> arguments{startValue, stopValue, stepValue,
                                        maskValue};
  if (payload) {
    argumentTypes.push_back(payload->type);
    arguments.push_back(*payload);
  }
  CallInferenceResult inference = types.inferMethodCallWithEvidence(
      container.type, methodName, argumentTypes);
  if (!inference) {
    unsupported(isAssignment
                    ? "is only supported for a list target with a list value"
                    : "is only supported for list targets");
    return;
  }
  if (!requireStaticEvidence(target, inference))
    return;
  if (!types.isStructuralMutatorMethod(container.type, methodName)) {
    unsupported("target's manifest does not declare the slice mutator");
    return;
  }
  Value posPack = emitPack(arguments);
  Value namePack = emitPack({});
  Value valuePack = emitPack({});
  llvm::SmallVector<mlir::Type, 2> resultTypes{inference.resultType};
  if (rebindable)
    resultTypes.push_back(container.value.getType());
  auto op = py::CallOp::create(builder, loc(target), resultTypes,
                               callProtocolFor(inference), container.value,
                               posPack.value, namePack.value, valuePack.value);
  op->setAttr("ly.bound_method", builder.getStringAttr(methodName));
  if (!rebindable)
    return;
  op->setAttr("ly.structural_mutation", builder.getUnitAttr());
  rebindStructuralMutation(target, containerName,
                           Value{op.getResult(1), container.type});
}

void ModuleEmitter::emitAssignTarget(const parser::Node &target, Value value) {
  if (target.kind == "Name") {
    llvm::StringRef name = ast::nameSpelling(target);
    // ⭐ `global X` where X is not a global this walk can WRITE. A name with
    // no cell falls through to the local binding below, which makes the write
    // a silent no-op -- the module global keeps its old value:
    //
    //     X: list[int] | None = [1]
    //     def f() -> None:
    //         global X
    //         X = [2]
    //     f(); print(X)      # printed [1]; CPython prints [2]
    //
    // Refused rather than made to work: the declaration is an explicit
    // statement that this assignment is not a local one, so binding a local is
    // the one answer it cannot have. The population it names shrank when every
    // contract got a cell -- the example above was `list[int]` until then --
    // and what is left is the annotations with no runtime value group to store
    // (a union, a protocol, a callable, `type[X]`), plus every UNannotated
    // module name, which is value-bound by the opt-in rule.
    if (!atModuleScope && currentGlobalDecls.count(name) &&
        !moduleGlobals.count(name)) {
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, target.range.start,
          "'global " + name.str() +
              "' names a module global this compiler does not give storage "
              "to, so the assignment cannot reach it; a module global is "
              "writable from a function when it is annotated with a concrete "
              "type, which a union, a protocol, a callable and 'type[X]' are "
              "not"});
      return;
    }
    if (isModuleGlobalWrite(name)) {
      mlir::Type type = moduleGlobals.lookup(name);
      // ⭐ A module global's cell has ONE runtime representation, fixed by the
      // declaration, and `coerceValue` no longer retypes between the numeric
      // contracts because that retyping was a lie. So the mismatch has to be
      // reported here, at the write.
      //
      // ⛔ Why NOT convert the value to the cell's type: `x: float = 3` would
      // then print 3.0 where CPython prints 3 -- the annotation does not
      // convert there either (tests/probe/wb_argument_boundary_numeric_tower.py
      // measures the same rejection at a parameter boundary).
      //
      // ⛔ And why NOT let the store through, which is what the retyping used
      // to do: the read still comes back at the cell's declared type, so the
      // int's lanes were reinterpreted as a double and `x: float = 3`
      // printed 5e-324. It reported "assignment value group has 3 values,
      // expected 1" before that, which named the count and not the cause.
      if (mlir::Type widened = types.widenLiteral(value.type);
          widened != type && isNumericPrimitiveContract(widened) &&
          isNumericPrimitiveContract(type)) {
        auto spell = [&](mlir::Type numeric) -> llvm::StringRef {
          if (numeric == types.boolType())
            return "bool";
          return numeric == types.intType() ? "int" : "float";
        };
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, target.range.start,
            "module global '" + name.str() + "' holds " +
                spell(type).str() + " and this assignment gives it " +
                spell(widened).str() +
                "; a module global has one runtime representation and these "
                "two do not share one, so write the value in the declared "
                "type"});
        return;
      }
      Value coerced = coerceValue(value, type, target);
      auto op = py::GlobalSetOp::create(builder, loc(target),
                                        builder.getStringAttr(name),
                                        coerced.value);
      markBoxedModuleGlobal(op);
      return;
    }
    // A name bound to a nonlocal cell (owner scope or a `nonlocal`-declaring
    // closure) writes THROUGH the cell: the binding itself never changes.
    auto bound = values.find(name);
    if (bound != values.end() && isCellContract(bound->second.type)) {
      emitCellStore(target, bound->second, value);
      return;
    }
    if (bound == values.end() && currentBoxedLocals.contains(name)) {
      // First binding of a boxed local: its storage is a fresh shared cell.
      Value cell = emitCellAlloc(target, value);
      values[name] = cell;
      types.bindSymbol(name, cellContentType(cell.type));
      return;
    }
    value = pinLoopCarriedTensor(name, value, target);
    values[name] = value;
    types.bindSymbol(name, value.type);
    return;
  }
  if (target.kind == "Attribute") {
    const parser::Node *objectNode = ast::node(target, "value");
    Value object = emitExpr(objectNode);
    if (auto attr = ast::string(target, "attr")) {
      // ⭐ A FROZEN DATACLASS REFUSES THE STORE, which is the other half of what
      // `frozen=True` means -- CPython raises FrozenInstanceError, and the whole
      // reason to accept the argument is that it is the only spelling for a
      // hashable record. Accepting the keyword and then letting the field be
      // rewritten would make the hash of a live dict key change under it.
      //
      // ⛔ The class's own __init__ is exempt, because that is where CPython
      // fills the fields (through object.__setattr__), and the synthesized one
      // goes through this same path.
      if (auto contract =
              mlir::dyn_cast_if_present<py::ContractType>(object.type);
          contract &&
          frozenDataclassContracts.count(contract.getContractName())) {
        bool inOwnConstructor =
            frozenInitContract &&
            *frozenInitContract == contract.getContractName().str();
        if (!inOwnConstructor) {
          diagnostics.push_back(parser::Diagnostic{
              parser::Severity::Error, target.range.start,
              "cannot assign to field '" + std::string(*attr) + "' of frozen "
              "dataclass '" +
                  py::contracts::displayClassNameForContract(
                      contract.getContractName()) +
                  "' (CPython raises FrozenInstanceError)"});
          return;
        }
      }
      // Property writes inline the setter; a getter without a setter is the
      // CPython AttributeError, surfaced statically.
      if (std::optional<MethodBinding> setter = lookupClassMethod(
              object.type, (llvm::Twine(*attr) + ".setter").str())) {
        if (setter->kind == "property_setter") {
          emitInlineMethodBody(target, object, /*bindDescriptorReceiver=*/true,
                               *setter, {value}, {});
          return;
        }
      }
      if (std::optional<MethodBinding> getter =
              lookupClassMethod(object.type, *attr);
          getter && getter->kind == "property") {
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, target.range.start,
            "property '" + std::string(*attr) + "' has no setter"});
        return;
      }
      // Mutable class attribute writes target the defining class's cell.
      if (auto typeObject =
              mlir::dyn_cast_if_present<py::TypeType>(object.type)) {
        if (auto contract = mlir::dyn_cast_if_present<py::ContractType>(
                typeObject.getInstanceType())) {
          llvm::StringRef className = contract.getContractName();
          if (std::optional<std::pair<llvm::StringRef, mlir::Type>> slot =
                  resolveClassAttrSlot(className, *attr)) {
            if (slot->first != className) {
              // CPython would create a NEW attribute on the subclass,
              // shadowing the base's; static storage has no per-subclass
              // presence, so writing through the subclass name is loud.
              diagnostics.push_back(parser::Diagnostic{
                  parser::Severity::Error, target.range.start,
                  "assigning class attribute '" + std::string(*attr) +
                      "' through subclass '" +
                      py::contracts::displayClassNameForContract(className) +
                      "' would shadow '" +
                      py::contracts::displayClassNameForContract(
                          slot->first) +
                      "'; assign through the defining class"});
              return;
            }
            std::string cellName =
                (llvm::Twine(slot->first) + "." + *attr).str();
            Value coerced = coerceValue(value, slot->second, target);
            py::GlobalSetOp::create(builder, loc(target),
                                    builder.getStringAttr(cellName),
                                    coerced.value);
            return;
          }
        }
      } else if (!lookupClassField(object.type, *attr)) {
        if (auto contract =
                mlir::dyn_cast_if_present<py::ContractType>(object.type)) {
          if (resolveClassAttrSlot(contract.getContractName(), *attr)) {
            // CPython creates an instance attribute shadowing the class
            // attribute; the static layout only has fields declared in
            // __init__, so this write cannot be represented.
            diagnostics.push_back(parser::Diagnostic{
                parser::Severity::Error, target.range.start,
                "assigning '" + std::string(*attr) +
                    "' through an instance would shadow the class "
                    "attribute; initialize it in __init__ to make it an "
                    "instance field"});
            return;
          }
        }
      }
      auto op = py::AttrSetOp::create(builder, loc(target), object.value, *attr,
                                      value.value);
      if (lookupClassField(object.type, *attr))
        op->setAttr("ly.attr.kind", builder.getStringAttr("field"));
      if (auto contract =
              mlir::dyn_cast_if_present<py::ContractType>(object.type))
        op->setAttr("ly.attr.owner",
                    builder.getStringAttr(contract.getContractName()));
      // Manifest-declared field assignments may refine the receiver's
      // contract parameters (ly.typing.field_param_bindings -- e.g. ctypes'
      // `fn.restype = c_int` binds CFuncPtr's T so `__call__` types as int):
      // rebind the local to the refined type. The attr.set op above stays --
      // lowering reads the same assignment as evidence.
      if (objectNode && objectNode->kind == "Name") {
        if (std::optional<mlir::Type> refined =
                types.fieldAssignmentRefinement(object.type, *attr,
                                                value.type)) {
          llvm::StringRef name = ast::nameSpelling(*objectNode);
          auto bound = values.find(name);
          if (bound != values.end() && bound->second.value == object.value) {
            bound->second.type = *refined;
            types.bindSymbol(name, *refined);
          }
        }
      }
    }
    return;
  }
  if (target.kind == "Subscript") {
    const parser::Node *containerNode = ast::node(target, "value");
    Value container = emitExpr(containerNode);
    if (container.value &&
        mlir::isa<mlir::RankedTensorType>(container.value.getType())) {
      // Shaped primitives are values, so the element write produces a new
      // one. Only a named local can observe it; anything else would drop the
      // result on the floor.
      if (!containerNode || containerNode->kind != "Name") {
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, target.range.start,
            "shaped primitive element assignment requires a named local "
            "target"});
        return;
      }
      llvm::StringRef containerName = ast::nameSpelling(*containerNode);
      auto bound = values.find(containerName);
      if (bound == values.end() || bound->second.value != container.value) {
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, target.range.start,
            "shaped primitive element assignment requires a named local "
            "target"});
        return;
      }
      if (std::optional<Value> updated = emitPrimitiveTensorSetItem(
              target, container, ast::node(target, "slice"), value)) {
        if (updated->value)
          values[containerName] =
              pinLoopCarriedTensor(containerName, *updated, target);
        return;
      }
      return;
    }
    if (const parser::Node *sliceNode = ast::node(target, "slice");
        sliceNode && sliceNode->kind == "Slice") {
      emitSliceMutation(target, containerNode, *sliceNode, "__setslice__",
                        value);
      return;
    }
    Value index = emitExpr(ast::node(target, "slice"));
    if (tryEmitClassDunder(target, container, "__setitem__", {index, value}))
      return;
    CallInferenceResult inference = types.inferMethodCallWithEvidence(
        container.type, "__setitem__", {index.type, value.type});
    if (!requireStaticEvidence(target, inference))
      return;
    // Manifest-declared structural mutators may reallocate the container's
    // storage: the op carries an extra container-typed result that rebinds
    // the local (same channel as mutating bound-method calls).
    if (containerNode && containerNode->kind == "Name" &&
        types.isStructuralMutatorMethod(container.type, "__setitem__")) {
      llvm::StringRef containerName = ast::nameSpelling(*containerNode);
      if (isStructuralMutationRebindable(containerName, container.value)) {
        auto op = py::SetItemOp::create(
            builder, loc(target),
            mlir::TypeRange{container.value.getType()},
            mlir::FlatSymbolRefAttr::get(&context, "__setitem__"),
            callProtocolFor(inference), container.value, index.value,
            value.value);
        op->setAttr("ly.structural_mutation", builder.getUnitAttr());
        rebindStructuralMutation(target, containerName,
                                 Value{op.getResult(0), container.type});
        return;
      }
    }
    py::SetItemOp::create(builder, loc(target), mlir::TypeRange{},
                          mlir::FlatSymbolRefAttr::get(&context, "__setitem__"),
                          callProtocolFor(inference), container.value,
                          index.value, value.value);
    return;
  }
  if (target.kind == "Tuple" || target.kind == "List") {
    if (const auto *elts = ast::nodeList(target, "elts")) {
      // ⭐ An unpacking target is answered element by element, so a target
      // this walk cannot answer has to stop it. Both of these used to be
      // SILENT:
      //
      //     a, *rest = [1, 2, 3, 4]     # rest kept its previous value
      //     a, b = (1, 2, 3)            # printed 1 2; CPython raises
      //
      // A starred target has no index to read -- it takes however many are
      // left -- so the loop below simply skipped it and whatever `rest` had
      // been bound to before stayed. Where the name did not exist yet the
      // program was refused, which is why this only ever showed up when the
      // target was pre-declared.
      // ⭐ A STARRED TARGET IS TWO INDEXED READS AND A SLICE. It takes however
      // many elements are left, which is not a count this walk can index --
      // but it IS a slice, and the bound at each end is known: the targets
      // before the star are read from the front and the ones after it from the
      // back, so the star's own share is `src[before:-after]` with the ends
      // counted rather than the middle. Refusing it left `first, *rest = xs`
      // -- the spelling for "the head and everything else" -- unwritable.
      //
      // ⛔ The star's value is a LIST whatever the source was, which is what
      // CPython builds: `a, *b = (1, 2, 3)` leaves `b == [2, 3]`, a list.
      std::size_t starIndex = elts->size();
      unsigned starCount = 0;
      for (auto [index, elt] : llvm::enumerate(*elts))
        if (elt && elt->kind == "Starred") {
          if (starCount++ == 0)
            starIndex = index;
        }
      if (starCount > 1) {
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, target.range.start,
            "multiple starred expressions in assignment"});
        return;
      }
      if (starCount == 1) {
        emitStarredUnpack(target, *elts, starIndex, value);
        return;
      }
      // ⭐ A NAMED TUPLE UNPACKS BY FIELD, because that is what its elements
      // ARE. The indexed walk below asks the method table for `__getitem__`,
      // and a source class does not declare one -- `p[0]` compiles only
      // because the subscript path folds a literal index to the field. So
      // `a, b = p` was refused with "'Pt' does not provide manifest method
      // '__getitem__'", and with it `for a, b in pts`, which is how a list of
      // pairs is read.
      if (auto contract = mlir::dyn_cast_if_present<py::ContractType>(
              types.widenLiteral(value.type)))
        if (namedTupleContracts.count(contract.getContractName())) {
          llvm::ArrayRef<std::string> order =
              classFieldOrders[contract.getContractName()];
          if (order.size() == elts->size()) {
            std::string scratch =
                "__ntunpack" + std::to_string(++syntheticFunctionCounter);
            values[scratch] = value;
            types.bindSymbol(scratch, value.type);
            for (auto [index, elt] : llvm::enumerate(*elts)) {
              parser::NodePtr read = synth::attribute(
                  synth::name(scratch, target.range), order[index],
                  target.range);
              Value field = emitExpr(read.get());
              synthesizedIteratorDefs.push_back(std::move(read));
              emitAssignTarget(*elt, field);
            }
            values.erase(scratch);
            return;
          }
        }
      // The count IS checked, against the object: a check on the type alone
      // catches only heterogeneous tuples (a tuple whose members share a type
      // collapses to `tuple[T]`, and a list carries its length in the object),
      // so it fired wrongly on `a, b = (1, 2)` and took out two working
      // programs before it caught anything.
      emitUnpackArityCheck(target, value, elts->size());
      for (auto [index, elt] : llvm::enumerate(*elts)) {
        Value indexValue{py::IntConstantOp::create(
                             builder, loc(*elt),
                             types.literal(std::to_string(index)),
                             builder.getStringAttr(std::to_string(index)))
                             .getResult(),
                         types.literal(std::to_string(index))};
        CallInferenceResult inference = types.inferMethodCallWithEvidence(
            value.type, "__getitem__", {indexValue.type});
        if (!requireStaticEvidence(*elt, inference))
          return;
        mlir::Type itemType = inference.resultType;
        auto getItem = py::GetItemOp::create(
            builder, loc(*elt), itemType,
            mlir::FlatSymbolRefAttr::get(&context, "__getitem__"),
            callProtocolFor(inference), value.value, indexValue.value);
        Value item{getItem.getResult(), itemType};
        emitAssignTarget(*elt, item);
      }
    }
  }
}

// The runtime half of `a, b = xs`: CPython's UNPACK_SEQUENCE raises when the
// source length is not the target arity, and this walk unpacks by index, so
// a longer source silently dropped its tail and a shorter one read past the
// end. The arity is not in the type for the common case -- a tuple whose
// members share a type collapses to `tuple[T]` (TypeSystem.cpp, `uniform`)
// and a list carries its length in the object -- so the check has to be a
// length comparison against the object.
//
// ⛔ Why NOT a static refusal when the arity IS in the type (a heterogeneous
// tuple): `try: a, b = t; except ValueError:` is legal Python, and the
// runtime raise it expects is what this emits. The static case only skips
// the comparison it can already answer.
//
// CPython's two messages differ in one detail that is decided statically:
// the tuple/list fast paths in ceval report what they got, the generic
// iterator path does not know it ("too many values to unpack (expected 2)"
// for a str source, "(expected 2, got 3)" for a list).
// The starred unpack, written as the Python it means. `a, *b, c = src` is
//
//     if len(src) < 2:
//         raise ValueError("not enough values to unpack (expected at least 2, "
//                          "got " + str(len(src)) + ")")
//     a = src[0]
//     b = list(src[1:-1])
//     c = src[-1]
//
// ⛔ The VALUES are synthesized and the targets are not: each target is handed
// to `emitAssignTarget`, the same walk the unstarred case uses, so a nested
// target keeps working and nothing has to copy an AST node. The values go
// through synthesis so the slice, the `list()` and the raise come from the
// paths that already answer for them -- a slice is `__getslice__` and not
// `__getitem__`, and getting that wrong here is the defect the comment in
// TypeSystem.cpp records for the inference channel.
void ModuleEmitter::emitStarredUnpack(
    const parser::Node &target,
    const std::vector<parser::NodePtr> &elements, std::size_t starIndex,
    Value source) {
  const parser::SourceRange range = target.range;
  const std::size_t before = starIndex;
  const std::size_t after = elements.size() - starIndex - 1;
  const std::size_t least = before + after;

  const parser::Node *starred = elements[starIndex].get();
  const parser::Node *starTarget = ast::node(*starred, "value");
  if (!starTarget) {
    diagnostics.push_back(parser::Diagnostic{parser::Severity::Error,
                                             starred->range.start,
                                             "starred target has no name"});
    return;
  }

  std::string scratch = "__lystar" + std::to_string(++syntheticFunctionCounter);
  values[scratch] = source;
  types.bindSymbol(scratch, source.type);
  auto src = [&] { return synth::name(scratch, range); };
  auto num = [&](std::int64_t value) {
    return synth::intConstant(value, range);
  };
  auto emitSynthesized = [&](parser::NodePtr node) {
    Value emitted = emitExpr(node.get());
    synthesizedIteratorDefs.push_back(std::move(node));
    return emitted;
  };

  // CPython's message, and its two halves: the arity is static, the length is
  // not.
  parser::NodePtr check = synth::ifStmt(
      synth::compare(synth::lenCall(src(), range), "Lt",
                     num(static_cast<std::int64_t>(least)), range),
      {synth::raiseStmt(
          synth::call(
              synth::name("ValueError", range),
              {synth::binOp(
                  synth::binOp(
                      synth::strConstant(
                          "not enough values to unpack (expected at least " +
                              std::to_string(least) + ", got ",
                          range),
                      "Add",
                      synth::call(synth::name("str", range),
                                  {synth::lenCall(src(), range)}, range),
                      range),
                  "Add", synth::strConstant(")", range), range)},
              range),
          range)},
      {}, range);
  emitStatement(*check);
  synthesizedIteratorDefs.push_back(std::move(check));

  for (std::size_t index = 0; index < before; ++index)
    emitAssignTarget(*elements[index],
                     emitSynthesized(synth::subscript(
                         src(), num(static_cast<std::int64_t>(index)), range)));

  // ⛔ The star's value is a LIST whatever the source was, which is what
  // CPython builds: `a, *b = (1, 2, 3)` leaves `b == [2, 3]`, a list.
  //
  // ⛔ A HETEROGENEOUS TUPLE IS SPELLED OUT INSTEAD OF SLICED. Its arity is in
  // its type, and `list(t[1:])` types the elements as their union -- which the
  // list constructor refuses ("iteration over a runtime-mode list of
  // !py.union<...>"), so `for n, *names in pairs` over `list[tuple[int, str]]`
  // did not compile. Naming the indices keeps each element's own type.
  std::optional<std::size_t> staticArity;
  if (auto contract = mlir::dyn_cast_if_present<py::ContractType>(
          types.widenLiteral(source.type)))
    if (contract.getContractName() == "builtins.tuple" &&
        contract.getArguments().size() > 1 &&
        contract.getArguments().size() >= least)
      staticArity = contract.getArguments().size();
  parser::NodePtr starValue;
  if (staticArity) {
    std::vector<parser::NodePtr> members;
    for (std::size_t index = before; index < *staticArity - after; ++index)
      members.push_back(
          synth::subscript(src(), num(static_cast<std::int64_t>(index)),
                           range));
    starValue = parser::makeNode("List", range);
    parser::addField(*starValue, "elts", std::move(members));
  } else {
    parser::NodePtr sliceNode = parser::makeNode("Slice", range);
    parser::addField(*sliceNode, "lower",
                     num(static_cast<std::int64_t>(before)));
    if (after != 0)
      parser::addField(*sliceNode, "upper",
                       num(-static_cast<std::int64_t>(after)));
    starValue = synth::call(
        synth::name("list", range),
        {synth::subscript(src(), std::move(sliceNode), range)}, range);
  }
  emitAssignTarget(*starTarget, emitSynthesized(std::move(starValue)));

  for (std::size_t index = 0; index < after; ++index)
    emitAssignTarget(
        *elements[starIndex + 1 + index],
        emitSynthesized(synth::subscript(
            src(), num(-static_cast<std::int64_t>(after - index)), range)));

  values.erase(scratch);
}

void ModuleEmitter::emitUnpackArityCheck(const parser::Node &target,
                                         Value source, std::size_t expected) {
  mlir::Type sourceType = types.widenLiteral(source.type);
  if (auto contract = mlir::dyn_cast_if_present<py::ContractType>(sourceType))
    if (contract.getContractName() == "builtins.tuple" &&
        contract.getArguments().size() > 1 &&
        contract.getArguments().size() == expected)
      return;
  if (!types.inferMethodCallWithEvidence(sourceType, "__len__", {}))
    return;

  bool reportsCount = false;
  if (auto contract = mlir::dyn_cast_if_present<py::ContractType>(sourceType))
    reportsCount = contract.getContractName() == "builtins.tuple" ||
                   contract.getContractName() == "builtins.list";

  parser::SourceRange range = target.range;
  unsigned serial = ++listCompCounter;
  std::string sourceName = "__lyunpack" + std::to_string(serial) + "_s";
  std::string lengthName = "__lyunpack" + std::to_string(serial) + "_n";
  std::string expectedText = std::to_string(expected);

  auto message = [&](const char *prefix) {
    parser::NodePtr text = synth::strConstant(std::string(prefix) +
                                      " values to unpack (expected " +
                                      expectedText + ", got ",
                                  range);
    parser::NodePtr count = synth::call(synth::name("str", range),
                             {synth::name(lengthName, range)}, range);
    return synth::binOp(synth::binOp(std::move(text), "Add", std::move(count), range),
                     "Add", synth::strConstant(")", range), range);
  };
  auto raiseNode = [&](parser::NodePtr text) {
    parser::NodePtr node = synth::raiseStmt(synth::call(synth::name("ValueError", range),
                              {std::move(text)}, range), range);
    return node;
  };

  parser::NodePtr tooMany =
      reportsCount ? message("too many")
                   : synth::strConstant("too many values to unpack (expected " +
                                        expectedText + ")",
                                    range);
  parser::NodePtr overLong = synth::ifStmt(
      synth::compare(synth::name(lengthName, range), "Gt",
                  synth::intConstant(static_cast<std::int64_t>(expected), range),
                  range),
      {raiseNode(std::move(tooMany))}, {}, range);
  parser::NodePtr tooFew = synth::ifStmt(
      synth::compare(synth::name(lengthName, range), "Lt",
                  synth::intConstant(static_cast<std::int64_t>(expected), range),
                  range),
      {raiseNode(message("not enough"))}, {}, range);

  std::vector<parser::NodePtr> guarded;
  guarded.push_back(std::move(overLong));
  guarded.push_back(std::move(tooFew));
  parser::NodePtr check = synth::ifStmt(
      synth::compare(synth::name(lengthName, range), "NotEq",
                  synth::intConstant(static_cast<std::int64_t>(expected), range),
                  range),
      std::move(guarded), {}, range);

  runWithScratchNames({sourceName, lengthName}, [&] {
    values[sourceName] = source;
    emitStatement(*synth::assign(synth::name(lengthName, range),
                              synth::lenCall(synth::name(sourceName, range), range),
                              range));
    emitStatement(*check);
  });
}

} // namespace lython::emitter
