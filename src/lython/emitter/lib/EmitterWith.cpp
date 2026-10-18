#include "AstSynth.h"
#include "EmitterCore.h"
#include "Parser.h"
#include "EmitterPyOps.h"
#include "EmitterSupport.h"

#include "AstAccess.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "llvm/ADT/STLExtras.h"

#include <cstddef>

namespace lython::emitter {

void ModuleEmitter::emitWithEnter(const parser::Node &item, bool async) {
  {
      Value contextValue = emitExpr(ast::node(item, "context_expr"));
      Value entered;
      if (async) {
        // ⭐ THE ASYNC TWINS OF THE RULE BELOW. `__enter__`/`__exit__` on a
        // source class are inlined here; `__aenter__`/`__aexit__` were not, so
        // `async with R():` over a Python context manager reached the lowering
        // and was refused there -- "runtime manifest has no R.__aexit__
        // method" -- while the synchronous spelling of the same class worked.
        // The method is a coroutine function like any `async def`: calling it
        // makes the coroutine, and the `async with` awaits it.
        if (std::optional<Value> opened =
                tryEmitClassDunder(item, contextValue, "__aenter__")) {
          entered = emitAwaitValue(item, *opened);
          if (const parser::Node *optional = ast::node(item, "optional_vars"))
            emitAssignTarget(*optional, entered);
          if (std::optional<MethodBinding> exit =
                  lookupClassMethod(contextValue.type, "__aexit__"))
            refuseUnrepresentableExitArguments(item, *exit);
          activeWithCleanups.push_back(WithCleanup{contextValue, async});
          return;
        }
        // ⛔ No manifest arm: an `__aenter__` is a coroutine function, and
        // no native class has one -- asyncio is Python (runtime/lib).
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, item.range.start,
            "async with needs a class that defines __aenter__ and __aexit__ "
            "in Python; " +
                typeText(contextValue.type) + " defines no __aenter__"});
        return;
      } else if (std::optional<Value> opened =
                     tryEmitClassDunder(item, contextValue, "__enter__")) {
        // ⭐ A context manager written in Python. `py.enter` is answered from
        // the runtime manifest, so a user class reached the lowering and was
        // refused there -- "runtime manifest has no Ctx.__enter__ method" --
        // for the plainest `with` in the language. Every other dunder on a
        // user class is inlined at this layer (`__len__`, `__getitem__`,
        // `__eq__`); `__enter__` and `__exit__` were the two that were not,
        // and the manifest op is right only for a manager the manifest knows.
        entered = *opened;
      } else {
        CallInferenceResult enterInference =
            types.inferMethodCallWithEvidence(contextValue.type, "__enter__",
                                              {});
        if (!requireStaticEvidence(item, enterInference))
          return;
        mlir::Type enterType = enterInference.resultType;
        auto enter =
            py::EnterOp::create(builder, loc(item), enterType, "__enter__",
                                callProtocolFor(enterInference),
                                contextValue.value, mlir::UnitAttr());
        entered = Value{enter.getResult(), enterType};
      }
      if (const parser::Node *optional = ast::node(item, "optional_vars"))
        emitAssignTarget(*optional, entered);
      if (std::optional<MethodBinding> exit = lookupClassMethod(
              contextValue.type, async ? "__aexit__" : "__exit__"))
        refuseUnrepresentableExitArguments(item, *exit);
      activeWithCleanups.push_back(WithCleanup{contextValue, async});
  }
}

// ⭐ The block runs inside CPython's OWN desugaring of `with`, one
// try/except/finally PER ITEM, with a sentinel that says which way the body
// left:
//
//     enter A
//     raised_A = False
//     try:
//         BODY
//     except as e:
//         raised_A = True
//         if not A.__exit__(None, e, None):
//             raise
//     finally:
//         if not raised_A:
//             A.__exit__(None, None, None)
//
// Emitting the cleanup only where the body falls through left the exception
// path with no __exit__ at all, and entering every manager before opening any
// try left `with A(), B()` skipping A's __exit__ when B's __enter__ raised.
// Routing through `emitTry` rather than adding an unwind edge here is what
// makes return, break, continue and the raise all one path: that machinery
// already answers every one of them.
//
// ⛔ Why NOT the plain try/FINALLY this used to be: a finally body cannot tell
// the two paths apart. The exceptional/normal bit exists only as the `mode`
// block argument `lowerTry` adds to the finally entry AFTER the emitter has
// run (Passes/Runtime/Ops/TryOps.cpp), no py op surfaces it, and
// `py.except.current_match` is not a substitute -- it answers "some exception
// is pending", which is also true on the NORMAL path of a `with` nested in an
// `except` handler. Nor could the exception be READ there:
// LyEH_BorrowCurrentException aborts when nothing is pending, and the finally
// runs on both paths. So __exit__ got (None, None, None) either way and its
// result was dropped -- a truthy return did not suppress.
//
// ⛔ Why NOT desugar into `mgr.__exit__(...)` as an AST call: the manager is
// an already-emitted SSA value, and a manifest manager (a file) does not
// answer __exit__ through attribute access at all. The synthesized statements
// carry an INDEX instead -- into the item list for the enter, into the cleanup
// stack for the exit -- which is the one thing an AST node can hold about a
// value.
//
// ⛔ Why the handler is BARE rather than `except BaseException`: emitTry
// answers a missing `type` field from the contract directly, so the spelling
// cannot be shadowed by a program that binds the name BaseException.
void ModuleEmitter::emitWith(const parser::Node &statement, bool async) {
  // Before the desugar: the body is rewritten into a synthesized try/finally
  // whose suite is not the one the later read lives in, so the slot for a name
  // the body binds has to be taken here, against the real suite.
  //
  // The `as` targets are hints for the same reason a loop target is: a body
  // that assigns FROM one (`text = f.read()`) cannot be typed until the
  // manager has been entered, and the slot has to exist before that. Only the
  // synchronous spelling -- an async `__aenter__` result is behind an await.
  {
    const auto *withItems = ast::nodeList(statement, "items");
    llvm::StringMap<mlir::Type> enterHints;
    if (!async && withItems)
      for (const parser::NodePtr &item : *withItems) {
        if (!item)
          continue;
        const parser::Node *bound = ast::node(*item, "optional_vars");
        if (!bound || bound->kind != "Name")
          continue;
        mlir::Type contextType =
            types.widenLiteral(types.inferExpr(ast::node(*item,
                                                         "context_expr")));
        CallInferenceResult entered =
            types.inferMethodCallWithEvidence(contextType, "__enter__", {});
        if (entered)
          enterHints[ast::nameSpelling(*bound)] = entered.resultType;
      }
    bindConditionallyAssignedLocals(statement, {ast::nodeList(statement, "body")},
                                    &enterHints);
  }
  std::size_t cleanupStart = activeWithCleanups.size();
  std::size_t itemStart = pendingWithItems.size();
  const auto *items = ast::nodeList(statement, "items");
  if (items)
    for (const parser::NodePtr &item : *items)
      pendingWithItems.push_back(PendingWithItem{item.get(), async});

  std::vector<parser::NodePtr> nested(
      ast::nodeList(statement, "body") ? *ast::nodeList(statement, "body")
                                       : std::vector<parser::NodePtr>{});
  std::size_t count = items ? items->size() : 0;
  for (std::size_t offset = count; offset > 0; --offset) {
    std::size_t index = offset - 1;
    unsigned serial = ++listCompCounter;
    std::string flagName = "__lywithraised" + std::to_string(serial);
    std::string excName = "__lywithexc" + std::to_string(serial);
    auto nameNode = [&](const std::string &id) {
      parser::NodePtr node = synth::name(id, statement.range);
      return node;
    };
    auto assignFlag = [&](bool value) {
      parser::NodePtr constant = parser::makeNode("Constant", statement.range);
      parser::addField(*constant, "value", value);
      parser::NodePtr node = synth::assign(nameNode(flagName), std::move(constant), statement.range);
      return node;
    };
    auto cleanupNode = [&](parser::NodePtr exception) {
      parser::NodePtr node =
          parser::makeNode("LyWithCleanup", statement.range);
      parser::addField(*node, "slot",
                       static_cast<std::int64_t>(cleanupStart + index));
      if (exception)
        parser::addField(*node, "exc", std::move(exception));
      return node;
    };

    parser::NodePtr handler =
        parser::makeNode("ExceptHandler", statement.range);
    parser::addField(*handler, "name", excName);
    parser::addField(*handler, "body",
                     std::vector<parser::NodePtr>{
                         assignFlag(true), cleanupNode(nameNode(excName))});

    parser::NodePtr notRaised = parser::makeNode("UnaryOp", statement.range);
    parser::addField(*notRaised, "op",
                     parser::makeNode("Not", statement.range));
    parser::addField(*notRaised, "operand", nameNode(flagName));
    parser::NodePtr normalExit = parser::makeNode("If", statement.range);
    parser::addField(*normalExit, "test", std::move(notRaised));
    parser::addField(*normalExit, "body",
                     std::vector<parser::NodePtr>{cleanupNode(nullptr)});
    parser::addField(*normalExit, "orelse", std::vector<parser::NodePtr>{});

    parser::NodePtr guarded = parser::makeNode("Try", statement.range);
    parser::addField(*guarded, "body", std::move(nested));
    parser::addField(*guarded, "handlers",
                     std::vector<parser::NodePtr>{std::move(handler)});
    parser::addField(*guarded, "orelse", std::vector<parser::NodePtr>{});
    parser::addField(*guarded, "finalbody",
                     std::vector<parser::NodePtr>{std::move(normalExit)});
    parser::NodePtr enter = parser::makeNode("LyWithEnter", statement.range);
    parser::addField(*enter, "slot",
                     static_cast<std::int64_t>(itemStart + index));
    nested = std::vector<parser::NodePtr>{std::move(enter), assignFlag(false),
                                          std::move(guarded)};
  }
  emitStatements(&nested);
  activeWithCleanups.resize(cleanupStart);
  pendingWithItems.resize(itemStart);
}

void ModuleEmitter::emitWithCleanup(const parser::Node &anchor,
                                    const WithCleanup &cleanup) {
  // Present on the cleanup the synthesized handler runs -- the name it bound
  // the live exception to -- and absent on the normal-path cleanup in the
  // finally body, which has no exception and owes no suppression decision.
  const parser::Node *exceptionNode = ast::node(anchor, "exc");
  auto noneOp = py::NoneOp::create(builder, loc(anchor), types.none());
  Value none{noneOp.getResult(), types.none()};
  Value exception = exceptionNode ? emitExpr(exceptionNode) : none;
  // The exception's class, as `type(e)` answers it: the class the raised
  // object HAS, read from its header.
  Value exceptionClass = none;
  if (exceptionNode) {
    mlir::Type bound = types.widenLiteral(exception.type);
    if (!mlir::isa<py::ContractType>(bound))
      bound = types.contract("builtins.BaseException");
    mlir::Type classType = types.typeObject(bound);
    exceptionClass = Value{py::TypeOfOp::create(builder, loc(anchor),
                                                classType, exception.value)
                               .getResult(),
                           classType};
  }
  if (cleanup.async) {
    // The source arm, which CAN carry the exception: a Python `__aexit__`
    // declares the three parameters, so the manifest's shape below is not the
    // constraint here.
    if (std::optional<Value> exited =
            tryEmitClassDunder(anchor, cleanup.manager, "__aexit__",
                               {exceptionClass, exception, none})) {
      Value suppress = emitAwaitValue(anchor, *exited);
      emitWithExitDecision(anchor, exceptionNode ? &suppress : nullptr);
      return;
    }
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, anchor.range.start,
        "async with needs a class that defines __aenter__ and __aexit__ in "
        "Python; " +
            typeText(cleanup.manager.type) + " defines no __aexit__"});
    return;
  }

  // The `__enter__` instance of the same rule; see emitWith.
  if (std::optional<Value> exited =
          tryEmitClassDunder(anchor, cleanup.manager, "__exit__",
                             {exceptionClass, exception, none})) {
    Value suppress = *exited;
    emitWithExitDecision(anchor, exceptionNode ? &suppress : nullptr);
    return;
  }

  // ⛔ Why the MANIFEST arm still hands over three Nones: a manifest __exit__
  // is a native function whose signature has no slots for the triple
  // (`LyNullContext_Exit(header) -> i1`), and a real exception operand is
  // refused there -- measured as "too many runtime arguments for
  // contextlib.nullcontext.__exit__". nullcontext is the only manifest context
  // manager in the tree and it ignores its arguments, so only the RESULT is
  // newly honoured here.
  CallInferenceResult exitInference = types.inferMethodCallWithEvidence(
      cleanup.manager.type, "__exit__", {none.type, none.type, none.type});
  if (!requireStaticEvidence(anchor, exitInference))
    return;
  auto exit = py::ExitOp::create(
      builder, loc(anchor), types.boolType(), "__exit__",
      callProtocolFor(exitInference), cleanup.manager.value, none.value,
      none.value, none.value, mlir::UnitAttr());
  Value suppress{exit.getResult(), types.boolType()};
  emitWithExitDecision(anchor, exceptionNode ? &suppress : nullptr);
}

void ModuleEmitter::emitWithExitDecision(const parser::Node &anchor,
                                         const Value *suppress) {
  if (!suppress || !suppress->value)
    return;
  // ⛔ Not a synthesized `if not <call>: raise`: the tested value is the result
  // of an ALREADY EMITTED inline body, and no AST node names an SSA value.
  // Both arms are ones the try machinery already answers -- falling through
  // the handler runs LyEH_DiscardCurrentException, which IS the suppression,
  // and py.raise.current in an except region routes through the finally and
  // rethrows.
  mlir::Value truth = emitBoolValue(*suppress, anchor);
  mlir::Block *entry = builder.getInsertionBlock();
  mlir::Region *region = entry->getParent();
  mlir::Block *continuation = entry->splitBlock(builder.getInsertionPoint());
  mlir::Block *rethrow =
      builder.createBlock(region, continuation->getIterator());
  builder.setInsertionPointToEnd(entry);
  mlir::cf::CondBranchOp::create(builder, loc(anchor), truth, continuation,
                                 mlir::ValueRange{}, rethrow,
                                 mlir::ValueRange{});
  builder.setInsertionPointToStart(rethrow);
  py::RaiseCurrentOp::create(builder, loc(anchor));
  builder.setInsertionPointToStart(continuation);
}

namespace {

// Does this subtree READ the name? A Name-node walk and not a reuse of
// collectAssignedNames' complement: the question is only whether the body can
// OBSERVE the argument, and the parameter's own `arg` node is not a Name, so
// the declaration does not answer yes to itself.
bool readsName(const parser::Node *node, llvm::StringRef name) {
  return ast::walk(node, [&](const parser::Node &current) {
    return current.kind == "Name" &&
                   llvm::StringRef(ast::nameSpelling(current)) == name
               ? ast::Walk::Stop
               : ast::Walk::Continue;
  });
}

// The `arg` spellings of a def, self included, in declaration order.
llvm::SmallVector<std::string, 4> parameterNames(const parser::Node &def) {
  llvm::SmallVector<std::string, 4> names;
  const parser::Node *arguments = ast::node(def, "args");
  if (!arguments)
    return names;
  for (const char *group : {"posonlyargs", "args"})
    if (const auto *list = ast::nodeList(*arguments, group))
      for (const parser::NodePtr &argument : *list)
        if (argument)
          if (auto spelling = ast::string(*argument, "arg"))
            names.push_back(std::string(*spelling));
  return names;
}

} // namespace

namespace {

parser::NodePtr cloneTree(const parser::Node &node) {
  parser::NodePtr clone = parser::makeNode(node.kind, node.range);
  for (const parser::Field &field : node.fields) {
    if (const auto *child = std::get_if<parser::NodePtr>(&field.value)) {
      parser::addField(*clone, field.name,
                       *child ? cloneTree(**child) : parser::NodePtr());
    } else if (const auto *children =
                   std::get_if<std::vector<parser::NodePtr>>(&field.value)) {
      std::vector<parser::NodePtr> copies;
      copies.reserve(children->size());
      for (const parser::NodePtr &element : *children)
        copies.push_back(element ? cloneTree(*element) : element);
      parser::addField(*clone, field.name, std::move(copies));
    } else {
      parser::addField(*clone, field.name, field.value);
    }
  }
  return clone;
}

// CPython's contextlib._GeneratorContextManager, which every desugared
// @contextmanager function returns: enter runs the generator to its yield,
// exit resumes it, or throws the exception into it and reads the answer off
// how the generator ends.
// ⛔ The generator is held as `types.GeneratorType`, not `typing.Generator`:
// a value typed by the protocol is not reference counted, so it could not be
// resumed.
constexpr const char *kGeneratorContextManagerSource = R"PY(
from types import GeneratorType as __ly_GeneratorType


class __ly_GeneratorContextManager[T]:
    def __init__(self, gen: __ly_GeneratorType[T, None, None]) -> None:
        self.gen = gen

    def __enter__(self) -> T:
        try:
            return next(self.gen)
        except StopIteration:
            raise RuntimeError("generator didn't yield")

    def __exit__(self, typ: type[BaseException] | None,
                 value: BaseException | None, tb: object) -> bool:
        if value is None:
            try:
                next(self.gen)
            except StopIteration:
                return False
            raise RuntimeError("generator didn't stop")
        try:
            self.gen.throw(value)
        except StopIteration as stopped:
            return stopped is not value
        except BaseException as raised:
            if raised is value:
                return False
            raise
        raise RuntimeError("generator didn't stop after throw()")
)PY";

} // namespace

// ⭐ `@contextmanager` IS TWO DEFS. The decorated body stays a generator,
// renamed `f$cmgen`, and `f` becomes a def of the same parameters that hands
// them on and wraps the generator in `__ly_GeneratorContextManager` -- the
// call CPython's decorator makes, written where the program can be typed:
//
//     @contextmanager
//     def opened(name: str) -> Iterator[int]: ...
//
// is `def opened(name: str) -> __ly_GeneratorContextManager[int]:
// return __ly_GeneratorContextManager(opened$cmgen(name))`. A method takes the
// same rewrite with `self.f$cmgen(...)`.
//
// ⛔ Not a decorator applied at run time: the decorator returns a function
// whose parameters are the decorated one's, which only a ParamSpec could type,
// and a ParamSpec cannot be specialized here.
void ModuleEmitter::desugarContextManagers(const parser::Node &moduleNode) {
  auto *body = const_cast<std::vector<parser::NodePtr> *>(
      ast::nodeList(moduleNode, "body"));
  if (!body)
    return;
  // The spellings this module gives contextlib.contextmanager.
  llvm::StringSet<> bareNames;
  llvm::StringSet<> moduleAliases;
  for (const parser::NodePtr &statement : *body) {
    if (!statement)
      continue;
    if (statement->kind == "ImportFrom" &&
        ast::string(*statement, "module").value_or("") == "contextlib") {
      if (const auto *names = ast::nodeList(*statement, "names"))
        for (const parser::NodePtr &alias : *names)
          if (alias && ast::string(*alias, "name").value_or("") ==
                           "contextmanager")
            bareNames.insert(ast::string(*alias, "asname")
                                 .value_or(ast::string(*alias, "name")
                                               .value_or("")));
    } else if (statement->kind == "Import") {
      if (const auto *names = ast::nodeList(*statement, "names"))
        for (const parser::NodePtr &alias : *names)
          if (alias && ast::string(*alias, "name").value_or("") == "contextlib")
            moduleAliases.insert(ast::string(*alias, "asname")
                                     .value_or("contextlib"));
    }
  }
  if (bareNames.empty() && moduleAliases.empty())
    return;
  // The decorator is gone once the defs are rewritten, and contextlib's
  // manifest has no `contextmanager` to import: the name leaves the import,
  // and an import left with no names leaves the module.
  for (std::size_t index = 0; index < body->size(); ++index) {
    parser::NodePtr statement = (*body)[index];
    if (!statement || statement->kind != "ImportFrom" ||
        ast::string(*statement, "module").value_or("") != "contextlib")
      continue;
    auto *names = const_cast<std::vector<parser::NodePtr> *>(
        ast::nodeList(*statement, "names"));
    if (!names)
      continue;
    llvm::erase_if(*names, [](const parser::NodePtr &alias) {
      return alias &&
             ast::string(*alias, "name").value_or("") == "contextmanager";
    });
    if (names->empty()) {
      body->erase(body->begin() + static_cast<std::ptrdiff_t>(index));
      --index;
    }
  }
  auto isContextManager = [&](const parser::Node *decorator) {
    if (!decorator)
      return false;
    if (decorator->kind == "Name")
      return bareNames.contains(ast::nameSpelling(*decorator));
    if (decorator->kind == "Attribute")
      if (const parser::Node *owner = ast::node(*decorator, "value");
          owner && owner->kind == "Name" &&
          moduleAliases.contains(ast::nameSpelling(*owner)))
        return ast::string(*decorator, "attr").value_or("") == "contextmanager";
    return false;
  };
  bool rewrote = false;
  // Rewrites the decorated defs of one statement list; `inClass` makes the
  // forwarding call go through the receiver.
  std::function<void(std::vector<parser::NodePtr> &, bool)> rewrite =
      [&](std::vector<parser::NodePtr> &statements, bool inClass) {
        for (std::size_t index = 0; index < statements.size(); ++index) {
          parser::NodePtr statement = statements[index];
          if (!statement)
            continue;
          if (statement->kind == "ClassDef" && !inClass) {
            if (auto *classBody = const_cast<std::vector<parser::NodePtr> *>(
                    ast::nodeList(*statement, "body")))
              rewrite(*classBody, /*inClass=*/true);
            continue;
          }
          if (statement->kind != "FunctionDef")
            continue;
          auto *decorators = const_cast<std::vector<parser::NodePtr> *>(
              ast::nodeList(*statement, "decorator_list"));
          if (!decorators || decorators->size() != 1 ||
              !isContextManager(decorators->front().get()))
            continue;
          std::string name(ast::string(*statement, "name").value_or(""));
          std::string generatorName = name + "$cmgen";
          parser::SourceRange range = statement->range;
          // The wrapper hands its parameters on by name, and a starred one
          // would be handed on as an unpacking whose size is not static.
          if (const parser::Node *arguments = ast::node(*statement, "args");
              arguments && (ast::node(*arguments, "vararg") ||
                            ast::node(*arguments, "kwarg"))) {
            diagnostics.push_back(parser::Diagnostic{
                parser::Severity::Error, range.start,
                "@contextmanager on '" + name +
                    "' is not supported with *args or **kwargs: the function "
                    "that wraps the generator would have to hand them on "
                    "unpacked; spell the parameters out"});
            continue;
          }
          // The generator keeps the body and the parameters.
          decorators->clear();
          if (parser::Field *field = parser::findField(*statement, "name"))
            field->value = generatorName;

          const parser::Node *arguments = ast::node(*statement, "args");
          std::vector<parser::NodePtr> callArgs;
          std::vector<parser::NodePtr> callKeywords;
          parser::NodePtr callee;
          bool receiverTaken = !inClass;
          auto forwardPositional = [&](const parser::NodePtr &arg) {
            if (!arg)
              return;
            std::string spelling(ast::string(*arg, "arg").value_or(""));
            if (!receiverTaken) {
              receiverTaken = true;
              callee = synth::attribute(synth::name(spelling, range),
                                        generatorName, range);
              return;
            }
            callArgs.push_back(synth::name(spelling, range));
          };
          if (arguments) {
            for (llvm::StringRef group : {"posonlyargs", "args"})
              if (const auto *list = ast::nodeList(*arguments, group))
                for (const parser::NodePtr &arg : *list)
                  forwardPositional(arg);
            if (const parser::Node *vararg = ast::node(*arguments, "vararg")) {
              parser::NodePtr starred = parser::makeNode("Starred", range);
              parser::addField(
                  *starred, "value",
                  synth::name(ast::string(*vararg, "arg").value_or(""), range));
              callArgs.push_back(std::move(starred));
            }
            if (const auto *list = ast::nodeList(*arguments, "kwonlyargs"))
              for (const parser::NodePtr &arg : *list)
                if (arg) {
                  std::string spelling(ast::string(*arg, "arg").value_or(""));
                  callKeywords.push_back(synth::keyword(
                      spelling, synth::name(spelling, range), range));
                }
            if (const parser::Node *kwarg = ast::node(*arguments, "kwarg")) {
              parser::NodePtr keyword = parser::makeNode("keyword", range);
              parser::addField(
                  *keyword, "value",
                  synth::name(ast::string(*kwarg, "arg").value_or(""), range));
              callKeywords.push_back(std::move(keyword));
            }
          }
          if (!callee)
            callee = synth::name(generatorName, range);
          parser::NodePtr generator = synth::callWithKeywords(
              std::move(callee), std::move(callArgs), std::move(callKeywords),
              range);
          parser::NodePtr wrapped = synth::call(
              synth::name("__ly_GeneratorContextManager", range),
              std::vector<parser::NodePtr>{std::move(generator)}, range);

          parser::NodePtr wrapper = parser::makeNode("FunctionDef", range);
          parser::addField(*wrapper, "name", name);
          parser::addField(*wrapper, "args",
                           arguments ? cloneTree(*arguments)
                                     : parser::makeNode("arguments", range));
          parser::addField(*wrapper, "body",
                           std::vector<parser::NodePtr>{
                               synth::returnStmt(std::move(wrapped), range)});
          parser::addField(*wrapper, "decorator_list",
                           std::vector<parser::NodePtr>{});
          // `Iterator[T]` (or Generator[T, ...]) returns T from __enter__.
          if (const parser::Node *returns = ast::node(*statement, "returns");
              returns && returns->kind == "Subscript")
            if (const parser::Node *slice = ast::node(*returns, "slice")) {
              const parser::Node *element = slice;
              if (slice->kind == "Tuple")
                if (const auto *elts = ast::nodeList(*slice, "elts");
                    elts && !elts->empty())
                  element = elts->front().get();
              if (element)
                parser::addField(
                    *wrapper, "returns",
                    synth::subscript(
                        synth::name("__ly_GeneratorContextManager", range),
                        cloneTree(*element), range));
            }
          if (const auto *params = ast::nodeList(*statement, "type_params"))
            if (!params->empty()) {
              std::vector<parser::NodePtr> copies;
              for (const parser::NodePtr &param : *params)
                copies.push_back(param ? cloneTree(*param) : param);
              parser::addField(*wrapper, "type_params", std::move(copies));
            }
          statements.insert(statements.begin() +
                                static_cast<std::ptrdiff_t>(index) + 1,
                            std::move(wrapper));
          ++index;
          rewrote = true;
        }
      };
  rewrite(*body, /*inClass=*/false);
  if (!rewrote)
    return;
  parser::ParseResult helper =
      parser::parse(kGeneratorContextManagerSource, "<contextmanager>");
  const auto *helperBody =
      helper.tree ? ast::nodeList(*helper.tree, "body") : nullptr;
  if (!helper.ok() || !helperBody) {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, moduleNode.range.start,
        "internal: the @contextmanager helper did not parse"});
    return;
  }
  // The helper's own tree owns its nodes; it is kept for as long as the
  // module it is spliced into.
  synthesizedHelperTrees.push_back(helper.tree);
  body->insert(body->begin(), helperBody->begin(), helperBody->end());
}

void ModuleEmitter::refuseUnrepresentableExitArguments(
    const parser::Node &anchor, const MethodBinding &exit) {
  if (!exit.method)
    return;
  llvm::SmallVector<std::string, 4> names = parameterNames(*exit.method);
  const auto *body = ast::nodeList(*exit.method, "body");
  // 3 = the traceback, counting self. The exception's class (1) is its class
  // value and the instance (2) is the caught object, so neither is named
  // here.
  for (std::size_t position : {std::size_t(3)}) {
    if (position >= names.size())
      continue;
    bool observed = false;
    if (body)
      for (const parser::NodePtr &statement : *body)
        observed = observed || readsName(statement.get(), names[position]);
    if (!observed)
      continue;
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, anchor.range.start,
        "__exit__ reads its '" + names[position] +
            "' parameter, and this compiler cannot produce that argument: a "
            "traceback has no value representation. Read the exception "
            "instance (the second parameter) or its class (the first) "
            "instead"});
    return;
  }
}

} // namespace lython::emitter
