#include "EmitterCore.h"
#include "EmitterPyOps.h"
#include "EmitterSupport.h"

#include "AstAccess.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <map>

#include <optional>
#include <string>
#include <vector>

// Lazy-iterator loop fusion: `for TGT in enumerate/zip/map/filter/reversed/
// iter(...)` and `for TGT in d.keys()/d.values()/d.items()` rewrite into
// equivalent AST loops before emission, the same technique the genexpr
// fusion and the reducer desugars use. The rewrite preserves CPython's
// per-element evaluation order (the transform/predicate runs once per
// consumed element, interleaved with the body), so laziness is observable
// through side effects even though no iterator object exists.

namespace lython::emitter {
namespace {

using parser::NodePtr;

NodePtr nameNode(const std::string &id, parser::SourceRange range) {
  NodePtr node = parser::makeNode("Name", range);
  parser::addField(*node, "id", id);
  return node;
}

NodePtr intConstant(std::int64_t value, parser::SourceRange range) {
  NodePtr node = parser::makeNode("Constant", range);
  parser::addField(*node, "value", value);
  return node;
}

NodePtr assignNode(NodePtr target, NodePtr value, parser::SourceRange range) {
  NodePtr node = parser::makeNode("Assign", range);
  parser::addField(*node, "targets", std::vector<NodePtr>{std::move(target)});
  parser::addField(*node, "value", std::move(value));
  return node;
}

NodePtr tupleNode(std::vector<NodePtr> elts, parser::SourceRange range) {
  NodePtr node = parser::makeNode("Tuple", range);
  parser::addField(*node, "elts", std::move(elts));
  return node;
}

NodePtr subscriptNode(NodePtr value, NodePtr slice,
                      parser::SourceRange range) {
  NodePtr node = parser::makeNode("Subscript", range);
  parser::addField(*node, "value", std::move(value));
  parser::addField(*node, "slice", std::move(slice));
  return node;
}

NodePtr binOpNode(NodePtr left, const char *opKind, NodePtr right,
                  parser::SourceRange range) {
  NodePtr op = parser::makeNode(opKind, range);
  NodePtr node = parser::makeNode("BinOp", range);
  parser::addField(*node, "left", std::move(left));
  parser::addField(*node, "op", std::move(op));
  parser::addField(*node, "right", std::move(right));
  return node;
}

NodePtr compareNode(NodePtr left, const char *opKind, NodePtr right,
                    parser::SourceRange range) {
  NodePtr op = parser::makeNode(opKind, range);
  NodePtr node = parser::makeNode("Compare", range);
  parser::addField(*node, "left", std::move(left));
  parser::addField(*node, "ops", std::vector<NodePtr>{std::move(op)});
  parser::addField(*node, "comparators", std::vector<NodePtr>{std::move(right)});
  return node;
}

NodePtr callNode(NodePtr func, std::vector<NodePtr> args,
                 parser::SourceRange range) {
  NodePtr node = parser::makeNode("Call", range);
  parser::addField(*node, "func", std::move(func));
  parser::addField(*node, "args", std::move(args));
  parser::addField(*node, "keywords", std::vector<NodePtr>{});
  return node;
}

NodePtr lenCall(NodePtr value, parser::SourceRange range) {
  return callNode(nameNode("len", range), {std::move(value)}, range);
}

NodePtr forNode(NodePtr target, NodePtr iter, std::vector<NodePtr> body,
                std::vector<NodePtr> orelse, parser::SourceRange range) {
  NodePtr node = parser::makeNode("For", range);
  parser::addField(*node, "target", std::move(target));
  parser::addField(*node, "iter", std::move(iter));
  parser::addField(*node, "body", std::move(body));
  parser::addField(*node, "orelse", std::move(orelse));
  return node;
}

NodePtr whileNode(NodePtr test, std::vector<NodePtr> body,
                  std::vector<NodePtr> orelse, parser::SourceRange range) {
  NodePtr node = parser::makeNode("While", range);
  parser::addField(*node, "test", std::move(test));
  parser::addField(*node, "body", std::move(body));
  parser::addField(*node, "orelse", std::move(orelse));
  return node;
}

NodePtr ifNode(NodePtr test, std::vector<NodePtr> body,
               parser::SourceRange range) {
  NodePtr node = parser::makeNode("If", range);
  parser::addField(*node, "test", std::move(test));
  parser::addField(*node, "body", std::move(body));
  parser::addField(*node, "orelse", std::vector<NodePtr>{});
  return node;
}

// Bind TARGET to per-iteration components. A tuple pattern of matching
// arity decomposes into component assignments — no tuple object exists, so
// no component's ownership is transferred twice. Otherwise a real tuple is
// built; `snapshotIndices` marks components that are loop-carried counter
// names, which must be copied through `+ 0` first (a tuple capturing the
// carried header value directly would pair the tuple's ownership transfer
// with the back-edge's release of the same value).
void appendTargetBinding(const NodePtr &target,
                         std::vector<NodePtr> components,
                         llvm::ArrayRef<unsigned> snapshotIndices,
                         const std::string &snapshotName,
                         std::vector<NodePtr> &body,
                         parser::SourceRange range) {
  if (target && target->kind == "Tuple") {
    if (const auto *elts = ast::nodeList(*target, "elts")) {
      bool simple = elts->size() == components.size();
      for (const NodePtr &elt : *elts)
        if (!elt || elt->kind == "Starred")
          simple = false;
      if (simple) {
        for (auto [index, elt] : llvm::enumerate(*elts))
          body.push_back(assignNode(elt, components[index], range));
        return;
      }
    }
  }
  for (unsigned index : snapshotIndices) {
    NodePtr snapshot = nameNode(snapshotName, range);
    body.push_back(assignNode(
        snapshot,
        binOpNode(components[index], "Add", intConstant(0, range), range),
        range));
    components[index] = snapshot;
  }
  body.push_back(
      assignNode(target, tupleNode(std::move(components), range), range));
}

// The original loop pieces every rewrite reuses.
struct ForParts {
  NodePtr target;
  std::vector<NodePtr> body;
  std::vector<NodePtr> orelse;
  parser::SourceRange range;
};

std::optional<ForParts> forParts(const parser::Node &statement) {
  const parser::Field *targetField = parser::findField(statement, "target");
  const parser::Field *bodyField = parser::findField(statement, "body");
  if (!targetField ||
      !std::holds_alternative<NodePtr>(targetField->value) || !bodyField ||
      !std::holds_alternative<std::vector<NodePtr>>(bodyField->value))
    return std::nullopt;
  ForParts parts;
  parts.target = std::get<NodePtr>(targetField->value);
  parts.body = std::get<std::vector<NodePtr>>(bodyField->value);
  parts.range = statement.range;
  if (const parser::Field *orelseField = parser::findField(statement, "orelse"))
    if (const auto *list =
            std::get_if<std::vector<NodePtr>>(&orelseField->value))
      parts.orelse = *list;
  if (!parts.target)
    return std::nullopt;
  return parts;
}

} // namespace

// Prior-binding bookkeeping for synthesized loop locals: the synthetic names
// are counter-unique, but erasing them after emission keeps `values` from
// accumulating dead entries (and restores any pathological collision).
void ModuleEmitter::runWithScratchNames(
    llvm::ArrayRef<std::string> names, llvm::function_ref<void()> emit) {
  llvm::SmallVector<std::pair<std::string, std::optional<Value>>, 4> priors;
  for (const std::string &name : names) {
    std::optional<Value> prior;
    if (auto found = values.find(name); found != values.end())
      prior = found->second;
    priors.push_back({name, prior});
  }
  emit();
  for (auto &[name, prior] : priors) {
    if (prior)
      values[name] = *prior;
    else
      values.erase(name);
  }
}

// A callable expression usable inside the fused loop body. Lambdas
// beta-reduce (parameter assignment + inlined body expression) so their
// parameter types come from the element flow instead of an annotation;
// names and attribute paths re-spell per element (their lookup is
// side-effect free under static dispatch). Anything else is rejected: a
// general callable-producing expression would need a first-class function
// temporary, and re-evaluating it per element could duplicate side effects.
bool ModuleEmitter::lazyCallableParts(const parser::Node &statement,
                                      const NodePtr &callee,
                                      LazyCallable &result) {
  if (!callee)
    return false;
  if (callee->kind == "Name" || callee->kind == "Attribute") {
    result.callee = callee;
    return true;
  }
  if (callee->kind == "Lambda") {
    const parser::Node *arguments = ast::node(*callee, "args");
    const parser::Field *bodyField = parser::findField(*callee, "body");
    if (!arguments || !bodyField ||
        !std::holds_alternative<NodePtr>(bodyField->value))
      return false;
    const auto *posonly = ast::nodeList(*arguments, "posonlyargs");
    const auto *args = ast::nodeList(*arguments, "args");
    const auto *defaults = ast::nodeList(*arguments, "defaults");
    const auto *kwonly = ast::nodeList(*arguments, "kwonlyargs");
    if ((posonly && !posonly->empty()) || (kwonly && !kwonly->empty()) ||
        (defaults && !defaults->empty()))
      return false;
    if (!args)
      return false;
    for (const NodePtr &arg : *args) {
      if (!arg)
        return false;
      auto name = ast::string(*arg, "arg");
      if (!name)
        return false;
      result.lambdaParams.push_back(std::string(*name));
    }
    result.lambdaBody = std::get<NodePtr>(bodyField->value);
    return true;
  }
  diagnostics.push_back(parser::Diagnostic{
      parser::Severity::Error, statement.range.start,
      "map()/filter() in a for loop requires a function name, attribute path, "
      "or lambda (a computed callable expression would re-evaluate per "
      "element)"});
  return false;
}

// Statements applying the callable to `arguments`, leaving the result
// expression in `out`. For a lambda this beta-reduces: each parameter is
// assigned its argument, and the body expression is used directly.
bool ModuleEmitter::buildLazyCall(const parser::Node &statement,
                                  const LazyCallable &callable,
                                  std::vector<NodePtr> arguments,
                                  std::vector<NodePtr> &prologue,
                                  NodePtr &out) {
  if (callable.callee) {
    out = callNode(callable.callee, std::move(arguments), statement.range);
    return true;
  }
  if (callable.lambdaParams.size() != arguments.size()) {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, statement.range.start,
        "lambda argument count does not match the fused iterator arity"});
    return false;
  }
  for (auto [index, param] : llvm::enumerate(callable.lambdaParams))
    prologue.push_back(assignNode(nameNode(param, statement.range),
                                  arguments[index], statement.range));
  out = callable.lambdaBody;
  return true;
}

bool ModuleEmitter::isBuiltinIteratorName(llvm::StringRef name) const {
  if (values.count(name) || genericFunctions.count(name))
    return false;
  if (types.lookupSymbol(name) || types.lookupClass(name))
    return false;
  return true;
}

// Sequence evidence for index-driven rewrites: __len__() plus
// __getitem__(int) must statically resolve.
bool ModuleEmitter::hasIndexableEvidence(const parser::Node *expr) {
  if (!expr)
    return false;
  mlir::Type type = types.inferExpr(expr);
  if (!type)
    return false;
  if (!types.inferMethodCallWithEvidence(type, "__len__", {}))
    return false;
  return static_cast<bool>(
      types.inferMethodCallWithEvidence(type, "__getitem__", {types.intType()}));
}

bool ModuleEmitter::tryEmitLazyIteratorFor(const parser::Node &statement,
                                           const parser::Node &iterCall) {
  auto reject = [&](llvm::StringRef reason) {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, statement.range.start, std::string(reason)});
    return true;
  };

  const parser::Node *calleeNode = ast::node(iterCall, "func");
  if (!calleeNode)
    return false;
  const auto *args = ast::nodeList(iterCall, "args");
  const auto *keywords = ast::nodeList(iterCall, "keywords");
  auto argAt = [&](std::size_t index) -> NodePtr {
    if (!args || index >= args->size())
      return nullptr;
    return (*args)[index];
  };
  if (args)
    for (const NodePtr &arg : *args)
      if (arg && arg->kind == "Starred")
        return false;

  std::optional<ForParts> parts = forParts(statement);
  if (!parts)
    return false;
  parser::SourceRange range = parts->range;
  unsigned serial = ++listCompCounter;
  auto scratch = [&](const char *stem) {
    return "__lyfuse" + std::to_string(serial) + "_" + std::string(stem);
  };

  // ---- dict view methods -------------------------------------------------
  if (calleeNode->kind == "Attribute") {
    auto attr = ast::string(*calleeNode, "attr");
    const parser::Field *receiverField = parser::findField(*calleeNode, "value");
    if (!attr || !receiverField ||
        !std::holds_alternative<NodePtr>(receiverField->value))
      return false;
    if (*attr != "keys" && *attr != "values" && *attr != "items")
      return false;
    NodePtr receiver = std::get<NodePtr>(receiverField->value);
    if (!receiver || (args && !args->empty()) ||
        (keywords && !keywords->empty()))
      return false;
    mlir::Type receiverType = types.inferExpr(receiver.get());
    auto contract = mlir::dyn_cast_if_present<py::ContractType>(receiverType);
    if (!contract || contract.getContractName() != "builtins.dict")
      return false;

    if (*attr == "keys") {
      emitFor(*forNode(parts->target, receiver, parts->body, parts->orelse,
                       range));
      return true;
    }
    // values()/items(): iterate the keys, subscript per element. The dict
    // expression must be a name (or become one) so the subscript re-reads
    // the same object the iteration guards.
    std::string dictName = scratch("d");
    std::string keyName = scratch("k");
    bool needsTemp = receiver->kind != "Name";
    NodePtr dictRef =
        needsTemp ? nameNode(dictName, range) : receiver;
    NodePtr keyRef = nameNode(keyName, range);
    NodePtr element = subscriptNode(dictRef, keyRef, range);
    std::vector<NodePtr> body;
    if (*attr == "items")
      appendTargetBinding(parts->target, {keyRef, element}, {},
                          std::string(), body, range);
    else
      body.push_back(assignNode(parts->target, element, range));
    body.insert(body.end(), parts->body.begin(), parts->body.end());
    NodePtr loop = forNode(keyRef, dictRef, std::move(body), parts->orelse,
                           range);
    runWithScratchNames({dictName, keyName}, [&] {
      if (needsTemp)
        emitStatement(*assignNode(nameNode(dictName, range), receiver, range));
      emitFor(*loop);
    });
    return true;
  }

  if (calleeNode->kind != "Name")
    return false;
  llvm::StringRef name = ast::nameSpelling(*calleeNode);
  if (name != "enumerate" && name != "zip" && name != "map" &&
      name != "filter" && name != "reversed" && name != "iter")
    return false;
  if (!isBuiltinIteratorName(name))
    return false;

  // ---- iter(X) -----------------------------------------------------------
  if (name == "iter") {
    if (!args || args->size() != 1 || (keywords && !keywords->empty()))
      return reject("iter() in a for loop requires exactly one argument");
    emitFor(*forNode(parts->target, argAt(0), parts->body, parts->orelse,
                     range));
    return true;
  }

  // ---- enumerate(X[, start]) ----------------------------------------------
  if (name == "enumerate") {
    NodePtr source = argAt(0);
    NodePtr start;
    if (args && args->size() == 2)
      start = argAt(1);
    else if (args && args->size() != 1)
      return reject("enumerate() takes an iterable and an optional start");
    if (keywords && keywords->size() == 1 && (*keywords)[0] && !start) {
      auto kwName = ast::string(*(*keywords)[0], "arg");
      if (!kwName || *kwName != "start")
        return reject("enumerate() got an unexpected keyword argument");
      const parser::Field *valueField =
          parser::findField(*(*keywords)[0], "value");
      if (valueField && std::holds_alternative<NodePtr>(valueField->value))
        start = std::get<NodePtr>(valueField->value);
    } else if (keywords && !keywords->empty()) {
      return reject("enumerate() got unexpected keyword arguments");
    }
    if (!source)
      return reject("enumerate() requires an iterable argument");
    if (!start)
      start = intConstant(0, range);

    std::string counterName = scratch("i");
    std::string elementName = scratch("v");
    std::string snapshotName = scratch("ic");
    NodePtr counter = nameNode(counterName, range);
    NodePtr element = nameNode(elementName, range);
    std::vector<NodePtr> body;
    appendTargetBinding(parts->target, {counter, element}, {0u}, snapshotName,
                        body, range);
    body.push_back(assignNode(
        counter, binOpNode(counter, "Add", intConstant(1, range), range),
        range));
    body.insert(body.end(), parts->body.begin(), parts->body.end());
    NodePtr loop =
        forNode(element, source, std::move(body), parts->orelse, range);
    runWithScratchNames({counterName, elementName, snapshotName}, [&] {
      emitStatement(*assignNode(nameNode(counterName, range), start, range));
      emitFor(*loop);
    });
    return true;
  }

  // ---- zip(A, B, ...) ------------------------------------------------------
  if (name == "zip") {
    if (keywords && !keywords->empty())
      return reject("zip() takes no keyword arguments");
    if (!args || args->size() < 2)
      return reject("zip() in a for loop requires at least two iterables");
    // Drive with the first iterable (CPython pulls arguments left to right,
    // so only argument 0 may be a non-indexable iterable — driving with a
    // later iterator would consume one extra element from it whenever an
    // earlier sequence stops the zip).
    for (std::size_t index = 1; index < args->size(); ++index)
      if (!hasIndexableEvidence(argAt(index).get()))
        return reject(
            "zip() supports one leading iterator; the remaining iterables "
            "must be indexable sequences (list/str/tuple/bytes) — convert "
            "the others with list(...) first");

    llvm::SmallVector<std::string, 4> scratchNames;
    std::string indexName = scratch("j");
    std::string driverName = scratch("v");
    scratchNames.push_back(indexName);
    scratchNames.push_back(driverName);
    NodePtr indexRef = nameNode(indexName, range);
    NodePtr driverRef = nameNode(driverName, range);

    std::vector<NodePtr> prologue;
    std::vector<NodePtr> body;
    std::vector<NodePtr> elements{driverRef};
    for (std::size_t index = 1; index < args->size(); ++index) {
      NodePtr source = argAt(index);
      NodePtr sourceRef = source;
      if (source->kind != "Name") {
        std::string sourceName = scratch(("s" + std::to_string(index)).c_str());
        scratchNames.push_back(sourceName);
        sourceRef = nameNode(sourceName, range);
        prologue.push_back(assignNode(sourceRef, source, range));
      }
      // if __j >= len(S): break  — the shortest input stops the loop.
      body.push_back(ifNode(
          compareNode(indexRef, "GtE", lenCall(sourceRef, range), range),
          {parser::makeNode("Break", range)}, range));
      elements.push_back(subscriptNode(sourceRef, indexRef, range));
    }
    appendTargetBinding(parts->target, std::move(elements), {},
                        std::string(), body, range);
    body.push_back(assignNode(
        indexRef, binOpNode(indexRef, "Add", intConstant(1, range), range),
        range));
    body.insert(body.end(), parts->body.begin(), parts->body.end());
    NodePtr loop =
        forNode(driverRef, argAt(0), std::move(body), parts->orelse, range);
    runWithScratchNames(scratchNames, [&] {
      for (const NodePtr &statementNode : prologue)
        emitStatement(*statementNode);
      emitStatement(*assignNode(nameNode(indexName, range),
                                intConstant(0, range), range));
      emitFor(*loop);
    });
    return true;
  }

  // ---- map(F, X, ...) ------------------------------------------------------
  if (name == "map") {
    if (keywords && !keywords->empty())
      return reject("map() takes no keyword arguments");
    if (!args || args->size() < 2)
      return reject("map() requires a callable and at least one iterable");
    LazyCallable callable;
    if (!lazyCallableParts(statement, argAt(0), callable))
      return true;
    if (args->size() == 2) {
      std::string elementName = scratch("v");
      NodePtr element = nameNode(elementName, range);
      std::vector<NodePtr> body;
      NodePtr applied;
      if (!buildLazyCall(statement, callable, {element}, body, applied))
        return true;
      body.push_back(assignNode(parts->target, applied, range));
      body.insert(body.end(), parts->body.begin(), parts->body.end());
      NodePtr loop =
          forNode(element, argAt(1), std::move(body), parts->orelse, range);
      llvm::SmallVector<std::string, 4> names{elementName};
      names.append(callable.lambdaParams.begin(), callable.lambdaParams.end());
      runWithScratchNames(names, [&] { emitFor(*loop); });
      return true;
    }
    // Multi-iterable map re-uses the zip rewrite through a synthesized
    // `for <tuple> in zip(...)` whose body applies the callable.
    std::vector<NodePtr> zipArgs(args->begin() + 1, args->end());
    llvm::SmallVector<std::string, 4> names;
    std::vector<NodePtr> elementRefs;
    std::vector<NodePtr> elementNames;
    for (std::size_t index = 0; index < zipArgs.size(); ++index) {
      std::string elementName =
          scratch(("v" + std::to_string(index)).c_str());
      names.push_back(elementName);
      elementRefs.push_back(nameNode(elementName, range));
      elementNames.push_back(nameNode(elementName, range));
    }
    std::vector<NodePtr> body;
    NodePtr applied;
    if (!buildLazyCall(statement, callable, elementRefs, body, applied))
      return true;
    body.push_back(assignNode(parts->target, applied, range));
    body.insert(body.end(), parts->body.begin(), parts->body.end());
    NodePtr zipCall = callNode(nameNode("zip", range), std::move(zipArgs),
                               range);
    NodePtr loop = forNode(tupleNode(std::move(elementNames), range), zipCall,
                           std::move(body), parts->orelse, range);
    names.append(callable.lambdaParams.begin(), callable.lambdaParams.end());
    runWithScratchNames(names, [&] { emitFor(*loop); });
    return true;
  }

  // ---- filter(F, X) ---------------------------------------------------------
  if (name == "filter") {
    if (keywords && !keywords->empty())
      return reject("filter() takes no keyword arguments");
    if (!args || args->size() != 2)
      return reject("filter() requires a predicate (or None) and an iterable");
    NodePtr predicate = argAt(0);
    bool identityPredicate =
        predicate && predicate->kind == "Constant" &&
        ast::isNoneField(*predicate, "value");
    LazyCallable callable;
    if (!identityPredicate &&
        !lazyCallableParts(statement, predicate, callable))
      return true;

    std::string elementName = scratch("v");
    NodePtr element = nameNode(elementName, range);
    std::vector<NodePtr> inner{assignNode(parts->target, element, range)};
    inner.insert(inner.end(), parts->body.begin(), parts->body.end());
    std::vector<NodePtr> body;
    NodePtr test = element;
    if (!identityPredicate &&
        !buildLazyCall(statement, callable, {element}, body, test))
      return true;
    body.push_back(ifNode(test, std::move(inner), range));
    NodePtr loop =
        forNode(element, argAt(1), std::move(body), parts->orelse, range);
    llvm::SmallVector<std::string, 4> names{elementName};
    names.append(callable.lambdaParams.begin(), callable.lambdaParams.end());
    runWithScratchNames(names, [&] { emitFor(*loop); });
    return true;
  }

  // ---- reversed(X) in for position ------------------------------------------
  if (name == "reversed") {
    if (!args || args->size() != 1 || (keywords && !keywords->empty()))
      return reject("reversed() requires exactly one sequence argument");
    if (!hasIndexableEvidence(argAt(0).get()))
      return reject("reversed() requires an indexable sequence "
                    "(list/str/tuple/bytes)");
    NodePtr source = argAt(0);
    llvm::SmallVector<std::string, 2> names;
    NodePtr sourceRef = source;
    std::vector<NodePtr> prologue;
    if (source->kind != "Name") {
      std::string sourceName = scratch("s");
      names.push_back(sourceName);
      sourceRef = nameNode(sourceName, range);
      prologue.push_back(assignNode(sourceRef, source, range));
    }
    std::string indexName = scratch("j");
    names.push_back(indexName);
    NodePtr indexRef = nameNode(indexName, range);
    std::vector<NodePtr> body{
        assignNode(indexRef,
                   binOpNode(indexRef, "Sub", intConstant(1, range), range),
                   range),
        assignNode(parts->target, subscriptNode(sourceRef, indexRef, range),
                   range)};
    body.insert(body.end(), parts->body.begin(), parts->body.end());
    NodePtr loop = whileNode(
        compareNode(indexRef, "Gt", intConstant(0, range), range),
        std::move(body), parts->orelse, range);
    runWithScratchNames(names, [&] {
      for (const NodePtr &statementNode : prologue)
        emitStatement(*statementNode);
      emitStatement(*assignNode(nameNode(indexName, range),
                                lenCall(sourceRef, range), range));
      emitWhile(*loop);
    });
    return true;
  }

  return false;
}

// ---------------------------------------------------------------------------
// Value form: enumerate/zip/map/filter/reversed/iter used as first-class
// values compile to per-call-site synthesized generator FUNCTIONS over
// indexable sequences (len + int __getitem__): the generator object gives
// CPython's observable laziness (next(), partial consumption, interleaved
// side effects). Bodies use index-based while loops on purpose — a for loop
// over a runtime sequence keeps its position in a function-local cell,
// which cannot survive a suspension, while an int index rides an int frame
// lane. Sources that are not indexable (dict/set/another generator) are
// rejected loudly: their iteration state cannot cross the suspension
// boundary yet, and the for-statement fusion above already covers them in
// loop position.
// ---------------------------------------------------------------------------

namespace {

// def <symbol>(<params>): <body> — parameters carry no annotations; the
// caller pins their types through TypeSystem::overrideParameterType.
NodePtr makeSyntheticGeneratorDef(
    const std::string &name, llvm::ArrayRef<std::string> params,
    std::vector<NodePtr> body, parser::SourceRange range,
    llvm::SmallVectorImpl<const parser::Node *> &paramNodes) {
  NodePtr def = parser::makeNode("FunctionDef", range);
  parser::addField(*def, "name", name);
  NodePtr arguments = parser::makeNode("arguments", range);
  std::vector<NodePtr> argNodes;
  for (const std::string &param : params) {
    NodePtr arg = parser::makeNode("arg", range);
    parser::addField(*arg, "arg", param);
    paramNodes.push_back(arg.get());
    argNodes.push_back(std::move(arg));
  }
  parser::addField(*arguments, "posonlyargs", std::vector<NodePtr>{});
  parser::addField(*arguments, "args", std::move(argNodes));
  parser::addField(*arguments, "kwonlyargs", std::vector<NodePtr>{});
  parser::addField(*arguments, "kw_defaults", std::vector<NodePtr>{});
  parser::addField(*arguments, "defaults", std::vector<NodePtr>{});
  parser::addField(*def, "args", std::move(arguments));
  parser::addField(*def, "body", std::move(body));
  return def;
}

NodePtr yieldNode(NodePtr value, parser::SourceRange range) {
  NodePtr yield = parser::makeNode("Yield", range);
  parser::addField(*yield, "value", std::move(value));
  NodePtr statement = parser::makeNode("Expr", range);
  parser::addField(*statement, "value", std::move(yield));
  return statement;
}

NodePtr trueConstant(parser::SourceRange range) {
  NodePtr node = parser::makeNode("Constant", range);
  parser::addField(*node, "value", true);
  return node;
}

std::string typeKey(mlir::Type type) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << type;
  return text;
}

} // namespace

std::optional<Value>
ModuleEmitter::tryEmitLazyIteratorValueCall(const parser::Node &expr,
                                            const parser::Node *calleeNode) {
  if (!calleeNode || calleeNode->kind != "Name")
    return std::nullopt;
  llvm::StringRef name = ast::nameSpelling(*calleeNode);
  if (name != "enumerate" && name != "zip" && name != "map" &&
      name != "filter" && name != "reversed" && name != "iter")
    return std::nullopt;
  if (!isBuiltinIteratorName(name))
    return std::nullopt;

  auto rejectValue = [&](llvm::StringRef reason) -> std::optional<Value> {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, expr.range.start, std::string(reason)});
    return emitNone(expr);
  };

  const auto *args = ast::nodeList(expr, "args");
  const auto *keywords = ast::nodeList(expr, "keywords");
  if (args)
    for (const NodePtr &arg : *args)
      if (arg && arg->kind == "Starred")
        return std::nullopt;
  parser::SourceRange range = expr.range;

  // Split off the callable argument for map/filter (spelled into the body,
  // not passed as a value: a callable has no generator argument lane).
  LazyCallable callable;
  bool identityPredicate = false;
  unsigned firstIterable = 0;
  if (name == "map" || name == "filter") {
    if (keywords && !keywords->empty())
      return rejectValue("map()/filter() take no keyword arguments");
    if (!args || args->size() < 2)
      return rejectValue("map()/filter() require a callable and iterable(s)");
    NodePtr predicate = (*args)[0];
    identityPredicate = name == "filter" && predicate &&
                        predicate->kind == "Constant" &&
                        ast::isNoneField(*predicate, "value");
    if (!identityPredicate && !lazyCallableParts(expr, predicate, callable))
      return emitNone(expr);
    firstIterable = 1;
    if (name == "filter" && args->size() != 2)
      return rejectValue("filter() takes exactly one iterable");
  }

  // enumerate start argument (second positional or start= keyword).
  const parser::Node *startNode = nullptr;
  if (name == "enumerate") {
    if (!args || args->empty() || args->size() > 2)
      return rejectValue("enumerate() takes an iterable and an optional start");
    if (args->size() == 2)
      startNode = (*args)[1].get();
    if (keywords && !keywords->empty()) {
      if (keywords->size() != 1 || startNode ||
          ast::string(*(*keywords)[0], "arg").value_or("") != "start")
        return rejectValue("enumerate() got unexpected keyword arguments");
      startNode = ast::node(*(*keywords)[0], "value");
    }
  } else if (name != "map" && keywords && !keywords->empty()) {
    return rejectValue(std::string(name) + "() takes no keyword arguments");
  }

  unsigned iterableCount =
      args ? static_cast<unsigned>(args->size()) - firstIterable : 0;
  if (name == "enumerate" && startNode)
    iterableCount = 1;
  if ((name == "iter" || name == "reversed" || name == "enumerate" ||
       name == "filter") &&
      iterableCount != 1)
    return rejectValue(std::string(name) + "() takes exactly one iterable");
  if (name == "zip" && iterableCount < 2)
    return rejectValue("zip() requires at least two iterables");
  if (name == "map" && iterableCount < 1)
    return rejectValue("map() requires at least one iterable");

  // Emit the iterable arguments (CPython argument evaluation order: the
  // callable spelling is not a value here, so iterables evaluate first —
  // observably identical because Name/Attribute/Lambda evaluation has no
  // effects).
  llvm::SmallVector<Value, 4> iterableValues;
  llvm::SmallVector<const parser::Node *, 4> iterableNodes;
  for (unsigned index = 0; index < iterableCount; ++index)
    iterableNodes.push_back((*args)[firstIterable + index].get());
  for (const parser::Node *node : iterableNodes)
    iterableValues.push_back(emitExpr(node));

  // iter(x) over something that is already an iterator returns it as-is
  // (CPython: iter(gen) is gen).
  if (name == "iter" &&
      types.inferMethodCallWithEvidence(iterableValues.front().type,
                                        "__next__", {}))
    return iterableValues.front();

  for (const Value &value : iterableValues) {
    mlir::Type widened = types.widenLiteral(value.type);
    if (!types.inferMethodCallWithEvidence(widened, "__len__", {}) ||
        !types.inferMethodCallWithEvidence(widened, "__getitem__",
                                           {types.intType()}))
      return rejectValue(
          std::string(name) +
          "() as a value requires indexable sequences (list/str/tuple/"
          "bytes); iterate non-indexable sources directly in a for loop, or "
          "convert with list(...) first");
  }

  // enumerate start value.
  std::optional<Value> startValue;
  if (name == "enumerate") {
    if (startNode)
      startValue = emitExpr(startNode);
    else {
      mlir::Type zeroType = types.literal("0");
      auto zero = py::IntConstantOp::create(builder, loc(expr), zeroType,
                                            builder.getStringAttr("0"));
      startValue = Value{zero.getResult(), zeroType};
    }
  }

  // Memoized synthesis, keyed by builtin + argument types + the callable's
  // spelling (map/filter bodies inline it syntactically).
  std::string memoKey = name.str();
  for (const Value &value : iterableValues)
    memoKey += "|" + typeKey(types.widenLiteral(value.type));
  if (callable.callee)
    memoKey += "|f:" + ast::qualifiedName(callable.callee.get());
  else if (callable.lambdaBody) {
    llvm::raw_string_ostream stream(memoKey);
    stream << "|lambda:" << callable.lambdaBody.get();
  } else if (identityPredicate) {
    memoKey += "|pred:None";
  }

  auto memoized = lazyIteratorMemo.find(memoKey);
  if (memoized == lazyIteratorMemo.end()) {
    unsigned serial = ++syntheticFunctionCounter;
    std::string symbol =
        ("__lyiter$" + name + "$" + llvm::Twine(serial)).str();

    llvm::SmallVector<std::string, 4> params;
    for (unsigned index = 0; index < iterableCount; ++index)
      params.push_back("__lyit" + std::to_string(index));
    if (name == "enumerate")
      params.push_back("__lyn");

    parser::SourceRange bodyRange = range;
    auto param = [&](unsigned index) {
      return nameNode(params[index], bodyRange);
    };
    NodePtr indexRef = nameNode("__lyi", bodyRange);

    std::vector<NodePtr> loopBody;
    for (unsigned index = 0; index < iterableCount; ++index)
      loopBody.push_back(
          ifNode(compareNode(indexRef, "GtE", lenCall(param(index), bodyRange),
                             bodyRange),
                 {parser::makeNode("Break", bodyRange)}, bodyRange));
    auto elementAt = [&](unsigned index) {
      return subscriptNode(param(index), indexRef, bodyRange);
    };

    if (name == "iter") {
      loopBody.push_back(yieldNode(elementAt(0), bodyRange));
    } else if (name == "enumerate") {
      loopBody.push_back(yieldNode(
          tupleNode({nameNode(params.back(), bodyRange), elementAt(0)},
                    bodyRange),
          bodyRange));
      loopBody.push_back(assignNode(
          nameNode(params.back(), bodyRange),
          binOpNode(nameNode(params.back(), bodyRange), "Add",
                    intConstant(1, bodyRange), bodyRange),
          bodyRange));
    } else if (name == "zip") {
      std::vector<NodePtr> elements;
      for (unsigned index = 0; index < iterableCount; ++index)
        elements.push_back(elementAt(index));
      loopBody.push_back(
          yieldNode(tupleNode(std::move(elements), bodyRange), bodyRange));
    } else if (name == "map") {
      std::vector<NodePtr> arguments;
      for (unsigned index = 0; index < iterableCount; ++index)
        arguments.push_back(elementAt(index));
      NodePtr applied;
      if (!buildLazyCall(expr, callable, std::move(arguments), loopBody,
                         applied))
        return emitNone(expr);
      loopBody.push_back(yieldNode(std::move(applied), bodyRange));
    } else if (name == "filter") {
      NodePtr test = elementAt(0);
      if (!identityPredicate &&
          !buildLazyCall(expr, callable, {elementAt(0)}, loopBody, test))
        return emitNone(expr);
      loopBody.push_back(
          ifNode(std::move(test), {yieldNode(elementAt(0), bodyRange)},
                 bodyRange));
    }
    loopBody.push_back(assignNode(
        indexRef, binOpNode(indexRef, "Add", intConstant(1, bodyRange),
                            bodyRange),
        bodyRange));

    std::vector<NodePtr> body;
    if (name == "reversed") {
      // __lyi = len(src); while __lyi > 0: __lyi -= 1; yield src[__lyi]
      body.push_back(assignNode(indexRef, lenCall(param(0), bodyRange),
                                bodyRange));
      std::vector<NodePtr> reversedBody{
          assignNode(indexRef,
                     binOpNode(indexRef, "Sub", intConstant(1, bodyRange),
                               bodyRange),
                     bodyRange),
          yieldNode(elementAt(0), bodyRange)};
      body.push_back(whileNode(
          compareNode(indexRef, "Gt", intConstant(0, bodyRange), bodyRange),
          std::move(reversedBody), {}, bodyRange));
    } else {
      body.push_back(assignNode(indexRef, intConstant(0, bodyRange),
                                bodyRange));
      body.push_back(whileNode(trueConstant(bodyRange), std::move(loopBody),
                               {}, bodyRange));
    }

    llvm::SmallVector<const parser::Node *, 4> paramNodes;
    NodePtr def = makeSyntheticGeneratorDef(symbol, params, std::move(body),
                                            bodyRange, paramNodes);
    synthesizedIteratorDefs.push_back(def);
    for (auto [index, value] : llvm::enumerate(iterableValues))
      types.overrideParameterType(paramNodes[index],
                                  types.widenLiteral(value.type));
    if (name == "enumerate")
      types.overrideParameterType(paramNodes.back(), types.intType());

    FunctionSignature sig = types.functionSignature(*def);
    emitCallableFunction(*def, symbol, sig, {}, /*isLambda=*/false);
    memoized = lazyIteratorMemo
                   .insert({memoKey, LazyIteratorSynthesis{
                                         symbol, sig.publicCallable}})
                   .first;
  }

  llvm::SmallVector<Value, 4> callArguments(iterableValues.begin(),
                                            iterableValues.end());
  if (startValue)
    callArguments.push_back(*startValue);
  Value callee = emitBindingRef(expr, memoized->second.symbol,
                                memoized->second.callableType);
  return emitCallableDispatch(
      expr, callee,
      emitCallOperands(expr, callArguments, /*includeAstArguments=*/false));
}

} // namespace lython::emitter
