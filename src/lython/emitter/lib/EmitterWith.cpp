#include "AstSynth.h"
#include "EmitterCore.h"
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
        AsyncContextMethodInferenceResult enterInference =
            types.inferAsyncContextEnterWithEvidence(contextValue.type);
        if (!requireStaticEvidence(item, enterInference))
          return;
        auto enter = py::AEnterOp::create(
            builder, loc(item), enterInference.awaitableType, "__aenter__",
            callProtocolFor(enterInference.method), contextValue.value,
            mlir::UnitAttr());
        entered =
            emitAwaitValue(item,
                           Value{enter.getResult(),
                                 enterInference.awaitableType},
                           enterInference.awaitResult);
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
  if (cleanup.async) {
    // ⛔ Three Nones even on the exception path: emitWithEnter has no inlined
    // source arm for __aenter__, so this is always the MANIFEST op, and a
    // manifest __aexit__ is a native function with no slots for the triple.
    AsyncContextMethodInferenceResult exitInference =
        types.inferAsyncContextExitWithEvidence(
            cleanup.manager.type, {none.type, none.type, none.type});
    if (!requireStaticEvidence(anchor, exitInference))
      return;
    auto exit = py::AExitOp::create(
        builder, loc(anchor), exitInference.awaitableType, "__aexit__",
        callProtocolFor(exitInference.method), cleanup.manager.value,
        none.value, none.value, none.value, mlir::UnitAttr());
    Value suppress =
        emitAwaitValue(anchor,
                       Value{exit.getResult(), exitInference.awaitableType},
                       exitInference.awaitResult);
    emitWithExitDecision(anchor, exceptionNode ? &suppress : nullptr);
    return;
  }

  // The `__enter__` instance of the same rule; see emitWith.
  if (std::optional<Value> exited = tryEmitClassDunder(
          anchor, cleanup.manager, "__exit__", {none, exception, none})) {
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
  if (!node)
    return false;
  if (node->kind == "Name" && llvm::StringRef(ast::nameSpelling(*node)) == name)
    return true;
  for (const parser::Field &field : node->fields) {
    if (const auto *child = std::get_if<parser::NodePtr>(&field.value)) {
      if (readsName(child->get(), name))
        return true;
    } else if (const auto *children =
                   std::get_if<std::vector<parser::NodePtr>>(&field.value)) {
      for (const parser::NodePtr &each : *children)
        if (readsName(each.get(), name))
          return true;
    }
  }
  return false;
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

void ModuleEmitter::refuseUnrepresentableExitArguments(
    const parser::Node &anchor, const MethodBinding &exit) {
  if (!exit.method)
    return;
  llvm::SmallVector<std::string, 4> names = parameterNames(*exit.method);
  const auto *body = ast::nodeList(*exit.method, "body");
  // 1 = exc_type, 3 = traceback, counting self. 2 = the exception instance,
  // which py.except.current_value DOES produce and which is therefore not
  // named here.
  for (std::size_t position : {std::size_t(1), std::size_t(3)}) {
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
            "' parameter, and this compiler cannot produce that argument: an "
            "exception's class object and its traceback have no value "
            "representation. Read the exception instance (the second "
            "parameter) instead"});
    return;
  }
}

} // namespace lython::emitter
