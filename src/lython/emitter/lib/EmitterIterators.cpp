#include "EmitterCore.h"
#include "EmitterPyOps.h"
#include "EmitterSupport.h"
#include "PyProtocols.h"

#include "AstSynth.h"

#include "AstAccess.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <map>

#include <initializer_list>
#include <optional>
#include <string>
#include <vector>

// The loop-fusion desugaring layer: library calls that CPython implements in
// C are rewritten here into equivalent AST loops before emission. In file
// order:
//
//   1. lazy iterators in a for statement -- `for TGT in enumerate/zip/map/
//      filter/reversed/iter(...)` and `for TGT in d.keys()/values()/items()`
//   2. the same names in VALUE position, which synthesize a per-call-site
//      generator function instead
//   3. dict method sugar (get / setdefault / popitem / fromkeys)
//   4. sorted() and list.sort() -- decorate-sort-undecorate
//   5. str.maketrans / str.translate
//   6. the list()/set()/tuple()/dict() constructors
//   7. itertools, the largest consumer of 1 and 2
//
// They share the fusion machinery (forParts, appendTargetBinding,
// lazyCallableParts, buildLazyCall), which is why they share a file: the
// itertools desugars alone are a third of it and are built entirely out of
// the first two sections.
//
// The rewrites preserve CPython's per-element evaluation order (the
// transform/predicate runs once per consumed element, interleaved with the
// body), so laziness is observable through side effects even though no
// iterator object exists.

namespace lython::emitter {
namespace {

using parser::NodePtr;













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
          body.push_back(synth::assign(elt, components[index], range));
        return;
      }
    }
  }
  for (unsigned index : snapshotIndices) {
    NodePtr snapshot = synth::name(snapshotName, range);
    body.push_back(synth::assign(
        snapshot,
        synth::binOp(components[index], "Add", synth::intConstant(0, range), range),
        range));
    components[index] = snapshot;
  }
  body.push_back(
      synth::assign(target, synth::tuple(std::move(components), range), range));
}

// The original loop pieces every rewrite reuses.
// An `int` the source wrote out, `-3` included: a negative literal parses as a
// unary minus over a positive one, and the sign is what decides which way a
// range counts.
std::optional<std::int64_t> constantIntValue(const parser::Node *node) {
  if (!node)
    return std::nullopt;
  if (node->kind == "Constant")
    return ast::integer(*node, "value");
  if (node->kind == "UnaryOp") {
    auto op = ast::string(*node, "op");
    const parser::Node *operand = ast::node(*node, "operand");
    if (op && operand && operand->kind == "Constant") {
      if (std::optional<std::int64_t> value = ast::integer(*operand, "value")) {
        if (*op == "USub")
          return -*value;
        if (*op == "UAdd")
          return *value;
      }
    }
  }
  return std::nullopt;
}

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
    out = synth::call(callable.callee, std::move(arguments), statement.range);
    return true;
  }
  if (callable.lambdaParams.size() != arguments.size()) {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, statement.range.start,
        "lambda argument count does not match the fused iterator arity"});
    return false;
  }
  for (auto [index, param] : llvm::enumerate(callable.lambdaParams))
    prologue.push_back(synth::assign(synth::name(param, statement.range),
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
bool ModuleEmitter::hasIndexWalkableEvidence(mlir::Type type) {
  if (!type)
    return false;
  if (!types.inferMethodCallWithEvidence(type, "__len__", {}))
    return false;
  if (!types.inferMethodCallWithEvidence(type, "__getitem__",
                                         {types.intType()}))
    return false;
  // Why NOT stop at len + int __getitem__: a mapping answers both. `dict`
  // declares `__getitem__: [$K] -> [$V]` while `__iter__` yields `$K`, so for
  // `dict[int, V]` the int subscript type-checks and walks KEY 0, 1, ... --
  // which is a lookup, not a position. The walk then either raises KeyError
  // for absent small ints or, when they happen to be present, silently yields
  // VALUES where CPython yields keys.
  //
  // Why NOT discriminate on the types instead (element type of __iter__ vs
  // result of __getitem__(int)): those coincide for `dict[int, int]`, the
  // exact shape that reported this defect, so the type comparison admits it.
  //
  // Why `keys` and `items`: this is CPython's own mapping duck-test (the one
  // `dict(m)` and `dict.update` use), so it also covers OrderedDict /
  // defaultdict / Counter / user Mappings, none of which are position-
  // addressable. No builtin sequence (list/str/tuple/bytes/bytearray/range)
  // or ctypes array declares either name.
  bool mappingLike =
      static_cast<bool>(types.inferMethodCallWithEvidence(type, "keys", {})) &&
      static_cast<bool>(types.inferMethodCallWithEvidence(type, "items", {}));
  return !mappingLike;
}

bool ModuleEmitter::hasIndexableEvidence(const parser::Node *expr) {
  if (!expr)
    return false;
  return hasIndexWalkableEvidence(types.inferExpr(expr));
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
      emitFor(*synth::forStmt(parts->target, receiver, parts->body, parts->orelse,
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
        needsTemp ? synth::name(dictName, range) : receiver;
    NodePtr keyRef = synth::name(keyName, range);
    NodePtr element = synth::subscript(dictRef, keyRef, range);
    std::vector<NodePtr> body;
    if (*attr == "items")
      appendTargetBinding(parts->target, {keyRef, element}, {},
                          std::string(), body, range);
    else
      body.push_back(synth::assign(parts->target, element, range));
    body.insert(body.end(), parts->body.begin(), parts->body.end());
    NodePtr loop = synth::forStmt(keyRef, dictRef, std::move(body), parts->orelse,
                           range);
    runWithScratchNames({dictName, keyName}, [&] {
      if (needsTemp)
        emitStatement(*synth::assign(synth::name(dictName, range), receiver, range));
      emitFor(*loop);
    });
    return true;
  }

  if (calleeNode->kind != "Name")
    return false;
  llvm::StringRef name = ast::nameSpelling(*calleeNode);

  // ---- range(...) in for position ----------------------------------------
  //
  // ⭐ RANGE IS AN ORDINARY ITERATOR, so the loop is written out as the
  // iterator's own semantics rather than taught to anything downstream. A
  // `range` object in loop position is not observable -- nothing can name it,
  // and its only behaviour is to hand back `start`, `start + step`, ... while
  // the stop has not been passed -- so an int counter and a comparison IS the
  // iterator, and the ordinary lowering then sees a loop over an `int` local
  // with nothing builtin about it.
  //
  // What that buys is not the iterator call it removes. `range_iterator` is a
  // manifest object, so `LyRangeIterator_Next` in the body is a call that
  // ALLOCATES, and a function whose body allocates cannot be given the unboxed
  // i64 clone (a clone is speculated on before its answer is believed, so it
  // has to be a rehearsal). One `for i in range(n)` therefore put every int in
  // the whole function back in a box.
  // ⛔ `programBindsName`, NOT `isBuiltinIteratorName`. The latter also refuses
  // a name the TYPE SYSTEM knows as a class, and `range` is bound there as
  // `builtins.range` -- so that guard is false for `range` always, and the
  // rewrite below never fired at all until this was the test. What has to be
  // excluded is a program that binds `range` itself, which is this.
  if (name == "range" && !programBindsName(name)) {
    if (!args || args->empty() || args->size() > 3 ||
        (keywords && !keywords->empty()))
      return false;
    // ⛔ A NAME TARGET ONLY. `for a, b in range(3)` is a TypeError in CPython
    // ("cannot unpack non-iterable int"), which the range object raises and
    // this rewrite would silently turn into something else.
    if (!parts->target || parts->target->kind != "Name")
      return false;
    // ⛔ AND STATICALLY int ARGUMENTS ONLY. `range(2.0)` is a TypeError, and a
    // counter loop over a float would run instead of raising -- the one
    // direction this may never move in. The object path still refuses it.
    auto isStaticInt = [&](const NodePtr &node) {
      auto contract = mlir::dyn_cast_if_present<py::ContractType>(
          types.widenLiteral(types.inferExpr(node.get())));
      return contract && contract.getContractName() == "builtins.int";
    };
    for (const NodePtr &arg : *args)
      if (!arg || !isStaticInt(arg))
        return false;

    NodePtr startExpr = args->size() == 1 ? synth::intConstant(0, range)
                                          : argAt(0);
    NodePtr stopExpr = args->size() == 1 ? argAt(0) : argAt(1);
    NodePtr stepExpr = args->size() == 3 ? argAt(2)
                                         : synth::intConstant(1, range);
    // A literal step is the overwhelmingly common one and decides the
    // comparison at compile time; anything else carries the sign test and the
    // zero check the range constructor would have run.
    // ⛔ A LITERAL ZERO IS NOT A LITERAL DIRECTION. `range(0, 5, 0)` is a
    // ValueError, so it takes the general path, whose first act is the check
    // that raises it -- reading a zero as "known step" left the counter with
    // no direction and the step temp unassigned.
    std::optional<std::int64_t> literal = constantIntValue(stepExpr.get());
    std::optional<std::int64_t> literalStep =
        literal && *literal != 0 ? literal : std::nullopt;

    std::string counterName = scratch("i");
    std::string stopName = scratch("n");
    std::string stepName = scratch("k");
    NodePtr counter = synth::name(counterName, range);
    NodePtr stop = synth::name(stopName, range);
    NodePtr step = synth::name(stepName, range);
    llvm::SmallVector<std::string, 3> names{counterName, stopName};

    NodePtr test;
    if (literalStep && *literalStep > 0) {
      test = synth::compare(counter, "Lt", stop, range);
    } else if (literalStep && *literalStep < 0) {
      test = synth::compare(counter, "Gt", stop, range);
    } else {
      names.push_back(stepName);
      test = synth::ifExp(
          synth::compare(step, "Gt", synth::intConstant(0, range), range),
          synth::compare(counter, "Lt", stop, range),
          synth::compare(counter, "Gt", stop, range), range);
    }

    // ⛔ THE COUNTER ADVANCES AND THE TARGET IS BOUND BEFORE THE BODY, not
    // after it: `continue` in a `while` jumps to the test, and an advance
    // written after the body would be skipped by it and never terminate. The
    // body may still rebind the target, which changes nothing about the
    // iteration -- as it does not in CPython, where the target and the
    // iterator's position are two different things.
    NodePtr stepValue = literalStep ? synth::intConstant(*literalStep, range)
                                    : synth::name(stepName, range);
    std::vector<NodePtr> body{
        synth::assign(parts->target, counter, range),
        synth::assign(counter,
                      synth::binOp(counter, "Add", stepValue, range), range)};
    body.insert(body.end(), parts->body.begin(), parts->body.end());
    NodePtr loop =
        synth::whileStmt(test, std::move(body), parts->orelse, range);

    runWithScratchNames(names, [&] {
      if (!literalStep) {
        emitStatement(*synth::assign(synth::name(stepName, range), stepExpr,
                                     range));
        emitStatement(*synth::ifStmt(
            synth::compare(synth::name(stepName, range), "Eq",
                           synth::intConstant(0, range), range),
            {synth::raiseValueError("range() arg 3 must not be zero", range)},
            {}, range));
      }
      emitStatement(*synth::assign(synth::name(stopName, range), stopExpr,
                                   range));
      emitStatement(*synth::assign(synth::name(counterName, range), startExpr,
                                   range));
      emitWhile(*loop);
    });
    return true;
  }

  if (name != "enumerate" && name != "zip" && name != "map" &&
      name != "filter" && name != "reversed" && name != "iter")
    return false;
  if (!isBuiltinIteratorName(name))
    return false;

  // ---- iter(X) -----------------------------------------------------------
  if (name == "iter") {
    if (!args || args->size() != 1 || (keywords && !keywords->empty()))
      return reject("iter() in a for loop requires exactly one argument");
    emitFor(*synth::forStmt(parts->target, argAt(0), parts->body, parts->orelse,
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
      start = synth::intConstant(0, range);

    std::string counterName = scratch("i");
    std::string elementName = scratch("v");
    std::string snapshotName = scratch("ic");
    NodePtr counter = synth::name(counterName, range);
    NodePtr element = synth::name(elementName, range);
    std::vector<NodePtr> body;
    appendTargetBinding(parts->target, {counter, element}, {0u}, snapshotName,
                        body, range);
    body.push_back(synth::assign(
        counter, synth::binOp(counter, "Add", synth::intConstant(1, range), range),
        range));
    body.insert(body.end(), parts->body.begin(), parts->body.end());
    NodePtr loop =
        synth::forStmt(element, source, std::move(body), parts->orelse, range);
    runWithScratchNames({counterName, elementName, snapshotName}, [&] {
      emitStatement(*synth::assign(synth::name(counterName, range), start, range));
      emitFor(*loop);
    });
    return true;
  }

  // ---- zip(A, B, ...) ------------------------------------------------------
  if (name == "zip") {
    // ⭐ `strict=` IS THE MODERN SPELLING (3.10+), and refusing every keyword
    // refused it too. False is what zip already does; True adds the length
    // check, and both have to be a LITERAL here because the arms differ in the
    // code emitted, not in a value.
    bool strictZip = false;
    if (keywords && !keywords->empty()) {
      if (keywords->size() != 1 || !keywords->front())
        return reject("zip() takes only the `strict` keyword");
      std::optional<std::string_view> keyword =
          ast::string(*keywords->front(), "arg");
      const parser::Node *value = ast::node(*keywords->front(), "value");
      if (!keyword || *keyword != "strict" || !value)
        return reject("zip() takes only the `strict` keyword");
      mlir::Type valueType = types.inferExpr(value);
      auto literal = mlir::dyn_cast_if_present<py::LiteralType>(valueType);
      llvm::StringRef spelling = literal ? literal.getSpelling()
                                         : llvm::StringRef();
      if (spelling != "True" && spelling != "False")
        return reject("zip(strict=...) needs a literal True or False: the two "
                      "answers are different code, not a different value");
      strictZip = spelling == "True";
    }
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

    // `strict=True` means every input has the same length, and CPython says
    // which one differs and in which direction. The zip below drives with
    // argument 1 and indexes the rest, so the check is `len(arg_i) vs
    // len(arg_1)` in argument order -- the first mismatch is the one CPython
    // reports.
    //
    // ⛔ The driver must be indexable for this, which the indexable-evidence
    // check above already requires of arguments 2..n but NOT of argument 1: a
    // leading iterator has no length to compare, so strict is refused there
    // rather than answered from a length nobody can ask for.
    llvm::SmallVector<NodePtr, 4> strictChecks;
    if (strictZip) {
      if (!hasIndexableEvidence(argAt(0).get()))
        return reject("zip(strict=True) needs every argument to be an "
                      "indexable sequence (list/str/tuple/bytes); the leading "
                      "iterator has no length to compare");
      for (std::size_t index = 1; index < args->size(); ++index) {
        std::string longer = "zip() argument " + std::to_string(index + 1) +
                             " is longer than argument 1";
        std::string shorter = "zip() argument " + std::to_string(index + 1) +
                              " is shorter than argument 1";
        NodePtr driverLength = synth::lenCall(argAt(0), range);
        NodePtr otherLength = synth::lenCall(argAt(index), range);
        strictChecks.push_back(synth::ifStmt(
            synth::compare(synth::lenCall(argAt(index), range), "Gt",
                           synth::lenCall(argAt(0), range), range),
            {synth::raiseCall("ValueError", longer, range)}, {}, range));
        strictChecks.push_back(synth::ifStmt(
            synth::compare(std::move(otherLength), "Lt",
                           std::move(driverLength), range),
            {synth::raiseCall("ValueError", shorter, range)}, {}, range));
      }
    }

    llvm::SmallVector<std::string, 4> scratchNames;
    std::string indexName = scratch("j");
    std::string driverName = scratch("v");
    scratchNames.push_back(indexName);
    scratchNames.push_back(driverName);
    NodePtr indexRef = synth::name(indexName, range);
    NodePtr driverRef = synth::name(driverName, range);

    std::vector<NodePtr> prologue;
    std::vector<NodePtr> body;
    std::vector<NodePtr> elements{driverRef};
    for (std::size_t index = 1; index < args->size(); ++index) {
      NodePtr source = argAt(index);
      NodePtr sourceRef = source;
      if (source->kind != "Name") {
        std::string sourceName = scratch(("s" + std::to_string(index)).c_str());
        scratchNames.push_back(sourceName);
        sourceRef = synth::name(sourceName, range);
        prologue.push_back(synth::assign(sourceRef, source, range));
      }
      // if __j >= len(S): break  — the shortest input stops the loop.
      body.push_back(synth::ifStmt(
          synth::compare(indexRef, "GtE", synth::lenCall(sourceRef, range), range),
          {parser::makeNode("Break", range)}, {}, range));
      elements.push_back(synth::subscript(sourceRef, indexRef, range));
    }
    appendTargetBinding(parts->target, std::move(elements), {},
                        std::string(), body, range);
    body.push_back(synth::assign(
        indexRef, synth::binOp(indexRef, "Add", synth::intConstant(1, range), range),
        range));
    body.insert(body.end(), parts->body.begin(), parts->body.end());
    NodePtr loop =
        synth::forStmt(driverRef, argAt(0), std::move(body), parts->orelse, range);
    runWithScratchNames(scratchNames, [&] {
      for (const NodePtr &check : strictChecks)
        emitStatement(*check);
      for (const NodePtr &statementNode : prologue)
        emitStatement(*statementNode);
      emitStatement(*synth::assign(synth::name(indexName, range),
                                synth::intConstant(0, range), range));
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
      NodePtr element = synth::name(elementName, range);
      std::vector<NodePtr> body;
      NodePtr applied;
      if (!buildLazyCall(statement, callable, {element}, body, applied))
        return true;
      body.push_back(synth::assign(parts->target, applied, range));
      body.insert(body.end(), parts->body.begin(), parts->body.end());
      NodePtr loop =
          synth::forStmt(element, argAt(1), std::move(body), parts->orelse, range);
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
      elementRefs.push_back(synth::name(elementName, range));
      elementNames.push_back(synth::name(elementName, range));
    }
    std::vector<NodePtr> body;
    NodePtr applied;
    if (!buildLazyCall(statement, callable, elementRefs, body, applied))
      return true;
    body.push_back(synth::assign(parts->target, applied, range));
    body.insert(body.end(), parts->body.begin(), parts->body.end());
    NodePtr zipCall = synth::call(synth::name("zip", range), std::move(zipArgs),
                               range);
    NodePtr loop = synth::forStmt(synth::tuple(std::move(elementNames), range), zipCall,
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
    NodePtr element = synth::name(elementName, range);
    std::vector<NodePtr> inner{synth::assign(parts->target, element, range)};
    inner.insert(inner.end(), parts->body.begin(), parts->body.end());
    std::vector<NodePtr> body;
    NodePtr test = element;
    if (!identityPredicate &&
        !buildLazyCall(statement, callable, {element}, body, test))
      return true;
    body.push_back(synth::ifStmt(test, std::move(inner), {}, range));
    NodePtr loop =
        synth::forStmt(element, argAt(1), std::move(body), parts->orelse, range);
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
      sourceRef = synth::name(sourceName, range);
      prologue.push_back(synth::assign(sourceRef, source, range));
    }
    std::string indexName = scratch("j");
    names.push_back(indexName);
    NodePtr indexRef = synth::name(indexName, range);
    std::vector<NodePtr> body{
        synth::assign(indexRef,
                   synth::binOp(indexRef, "Sub", synth::intConstant(1, range), range),
                   range),
        synth::assign(parts->target, synth::subscript(sourceRef, indexRef, range),
                   range)};
    body.insert(body.end(), parts->body.begin(), parts->body.end());
    NodePtr loop = synth::whileStmt(
        synth::compare(indexRef, "Gt", synth::intConstant(0, range), range),
        std::move(body), parts->orelse, range);
    runWithScratchNames(names, [&] {
      for (const NodePtr &statementNode : prologue)
        emitStatement(*statementNode);
      emitStatement(*synth::assign(synth::name(indexName, range),
                                synth::lenCall(sourceRef, range), range));
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
    if (!hasIndexWalkableEvidence(types.widenLiteral(value.type)))
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
    memoKey += "|" + typeText(types.widenLiteral(value.type));
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
      return synth::name(params[index], bodyRange);
    };
    NodePtr indexRef = synth::name("__lyi", bodyRange);

    std::vector<NodePtr> loopBody;
    for (unsigned index = 0; index < iterableCount; ++index)
      loopBody.push_back(
          synth::ifStmt(synth::compare(indexRef, "GtE", synth::lenCall(param(index), bodyRange),
                             bodyRange),
                 {parser::makeNode("Break", bodyRange)}, {}, bodyRange));
    auto elementAt = [&](unsigned index) {
      return synth::subscript(param(index), indexRef, bodyRange);
    };

    if (name == "iter") {
      loopBody.push_back(synth::yieldStmt(elementAt(0), bodyRange));
    } else if (name == "enumerate") {
      loopBody.push_back(synth::yieldStmt(
          synth::tuple({synth::name(params.back(), bodyRange), elementAt(0)},
                    bodyRange),
          bodyRange));
      loopBody.push_back(synth::assign(
          synth::name(params.back(), bodyRange),
          synth::binOp(synth::name(params.back(), bodyRange), "Add",
                    synth::intConstant(1, bodyRange), bodyRange),
          bodyRange));
    } else if (name == "zip") {
      std::vector<NodePtr> elements;
      for (unsigned index = 0; index < iterableCount; ++index)
        elements.push_back(elementAt(index));
      loopBody.push_back(
          synth::yieldStmt(synth::tuple(std::move(elements), bodyRange), bodyRange));
    } else if (name == "map") {
      std::vector<NodePtr> arguments;
      for (unsigned index = 0; index < iterableCount; ++index)
        arguments.push_back(elementAt(index));
      NodePtr applied;
      if (!buildLazyCall(expr, callable, std::move(arguments), loopBody,
                         applied))
        return emitNone(expr);
      loopBody.push_back(synth::yieldStmt(std::move(applied), bodyRange));
    } else if (name == "filter") {
      NodePtr test = elementAt(0);
      if (!identityPredicate &&
          !buildLazyCall(expr, callable, {elementAt(0)}, loopBody, test))
        return emitNone(expr);
      loopBody.push_back(
          synth::ifStmt(std::move(test), {synth::yieldStmt(elementAt(0), bodyRange)}, {},
                 bodyRange));
    }
    loopBody.push_back(synth::assign(
        indexRef, synth::binOp(indexRef, "Add", synth::intConstant(1, bodyRange),
                            bodyRange),
        bodyRange));

    std::vector<NodePtr> body;
    if (name == "reversed") {
      // __lyi = len(src); while __lyi > 0: __lyi -= 1; yield src[__lyi]
      body.push_back(synth::assign(indexRef, synth::lenCall(param(0), bodyRange),
                                bodyRange));
      std::vector<NodePtr> reversedBody{
          synth::assign(indexRef,
                     synth::binOp(indexRef, "Sub", synth::intConstant(1, bodyRange),
                               bodyRange),
                     bodyRange),
          synth::yieldStmt(elementAt(0), bodyRange)};
      body.push_back(synth::whileStmt(
          synth::compare(indexRef, "Gt", synth::intConstant(0, bodyRange), bodyRange),
          std::move(reversedBody), {}, bodyRange));
    } else {
      body.push_back(synth::assign(indexRef, synth::intConstant(0, bodyRange),
                                bodyRange));
      body.push_back(synth::whileStmt(synth::constantBool(true, bodyRange), std::move(loopBody),
                               {}, bodyRange));
    }

    llvm::SmallVector<const parser::Node *, 4> paramNodes;
    llvm::SmallVector<synth::Param, 4> synthParams;
    for (const std::string &param : params)
      synthParams.push_back(synth::Param{param, nullptr});
    NodePtr def = synth::functionDef(symbol, synthParams, {}, std::move(body),
                                     nullptr, {}, bodyRange, &paramNodes);
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

// ---------------------------------------------------------------------------
// dict method sugar: get(k) / setdefault / popitem / dict.fromkeys rewrite
// into statement sequences over the existing dict primitives (membership,
// getitem, setitem, pop). Why AST rewrites and not manifest natives: each is
// pure composition of already-verified operations, and the native tier has
// no way to build the result tuple / fresh dict without duplicating that
// machinery. Deviation: the compositions probe twice where CPython probes
// once — observable only through side-effecting user __hash__/__eq__.
// ---------------------------------------------------------------------------

namespace {





} // namespace

bool ModuleEmitter::isDictTypedExpr(const parser::Node *expr) {
  return exprHasContract(expr, "builtins.dict");
}

bool ModuleEmitter::exprHasContract(const parser::Node *expr,
                                    llvm::StringRef contractName) {
  if (!expr)
    return false;
  auto contract = mlir::dyn_cast_if_present<py::ContractType>(
      types.widenLiteral(types.inferExpr(expr)));
  return contract && contract.getContractName() == contractName;
}

std::optional<Value>
ModuleEmitter::tryEmitDictMethodSugar(const parser::Node &expr,
                                      const parser::Node *calleeNode) {
  if (!calleeNode || calleeNode->kind != "Attribute")
    return std::nullopt;
  auto attr = ast::string(*calleeNode, "attr");
  const parser::Field *receiverField = parser::findField(*calleeNode, "value");
  if (!attr || !receiverField ||
      !std::holds_alternative<NodePtr>(receiverField->value))
    return std::nullopt;
  NodePtr receiver = std::get<NodePtr>(receiverField->value);
  if (!receiver)
    return std::nullopt;
  const auto *args = ast::nodeList(expr, "args");
  const auto *keywords = ast::nodeList(expr, "keywords");
  if (keywords && !keywords->empty())
    return std::nullopt;
  std::size_t argCount = args ? args->size() : 0;
  parser::SourceRange range = expr.range;
  unsigned serial = ++listCompCounter;
  auto scratch = [&](const char *stem) {
    return "__lydict" + std::to_string(serial) + "_" + std::string(stem);
  };
  auto rejectSugar = [&](llvm::StringRef reason) -> std::optional<Value> {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, expr.range.start, std::string(reason)});
    return emitNone(expr);
  };

  // dict.fromkeys(iterable[, value]) — the receiver is the dict CLASS.
  if (*attr == "fromkeys" && receiver->kind == "Name" &&
      ast::nameSpelling(*receiver) == "dict" && !values.count("dict") &&
      !types.lookupSymbol("dict")) {
    if (argCount < 1 || argCount > 2)
      return rejectSugar("dict.fromkeys() takes an iterable and an optional "
                         "value");
    std::string valueName = scratch("fv");
    std::string keyName = scratch("fk");
    NodePtr valueInit = argCount == 2 ? (*args)[1] : synth::noneConstant(range);
    NodePtr comprehension = parser::makeNode("comprehension", range);
    parser::addField(*comprehension, "target", synth::name(keyName, range));
    parser::addField(*comprehension, "iter", (*args)[0]);
    parser::addField(*comprehension, "ifs", std::vector<NodePtr>{});
    parser::addField(*comprehension, "is_async", std::int64_t{0});
    NodePtr comp = parser::makeNode("DictComp", range);
    parser::addField(*comp, "key", synth::name(keyName, range));
    parser::addField(*comp, "value", synth::name(valueName, range));
    parser::addField(*comp, "generators",
                     std::vector<NodePtr>{std::move(comprehension)});
    std::optional<Value> result;
    runWithScratchNames({valueName, keyName}, [&] {
      emitStatement(*synth::assign(synth::name(valueName, range),
                                std::move(valueInit), range));
      result = emitExpr(comp.get());
    });
    return result;
  }

  if (*attr != "get" && *attr != "setdefault" && *attr != "popitem")
    return std::nullopt;
  if (!isDictTypedExpr(receiver.get()))
    return std::nullopt;
  // get with an explicit default has a native lowering; only the one-argument
  // (None-default) form desugars here.
  if (*attr == "get" && argCount != 1)
    return std::nullopt;

  std::string dictName = scratch("d");
  bool needsTemp = receiver->kind != "Name";
  NodePtr dictRef = needsTemp ? synth::name(dictName, range) : receiver;
  auto withPrologue = [&](llvm::ArrayRef<std::string> names,
                          llvm::function_ref<std::optional<Value>()> emit)
      -> std::optional<Value> {
    std::optional<Value> result;
    llvm::SmallVector<std::string, 4> scratchNames(names.begin(), names.end());
    if (needsTemp)
      scratchNames.push_back(dictName);
    runWithScratchNames(scratchNames, [&] {
      if (needsTemp)
        emitStatement(*synth::assign(synth::name(dictName, range), receiver, range));
      result = emit();
    });
    return result;
  };

  if (*attr == "get") {
    // __r = None; if __k in d: __r = d[__k]  →  Optional[V]
    std::string keyName = scratch("gk");
    std::string resultName = scratch("gr");
    return withPrologue({keyName, resultName}, [&]() -> std::optional<Value> {
      emitStatement(*synth::assign(synth::name(keyName, range), (*args)[0], range));
      emitStatement(*synth::assign(synth::name(resultName, range),
                                synth::noneConstant(range), range));
      NodePtr hit = synth::ifStmt(
          synth::compareIn(synth::name(keyName, range), dictRef, range),
          {synth::assign(synth::name(resultName, range),
                      synth::subscript(dictRef, synth::name(keyName, range), range),
                      range)}, {},
          range);
      emitStatement(*hit);
      auto bound = values.find(resultName);
      if (bound == values.end() || !bound->second.value)
        return rejectSugar("cannot lower dict.get(key) over this dict");
      return bound->second;
    });
  }

  if (*attr == "setdefault") {
    if (argCount < 1 || argCount > 2)
      return rejectSugar("dict.setdefault() takes a key and an optional "
                         "default");
    std::string keyName = scratch("sk");
    std::string valueName = scratch("sv");
    std::string resultName = scratch("sr");
    return withPrologue(
        {keyName, valueName, resultName}, [&]() -> std::optional<Value> {
          emitStatement(
              *synth::assign(synth::name(keyName, range), (*args)[0], range));
          emitStatement(*synth::assign(
              synth::name(valueName, range),
              argCount == 2 ? (*args)[1] : synth::noneConstant(range), range));
          // if __k not in d: d[__k] = __v
          // __r = d[__k]
          // Single-arm shape on purpose (a two-arm branch mixing a getitem
          // arm with a setitem arm trips the If-join's structural-mutation
          // threading), and reading back d[__k] returns the STORED value —
          // CPython's setdefault does the same.
          emitStatement(*synth::ifStmt(
              synth::compare(synth::name(keyName, range), "NotIn", dictRef, range),
              {synth::assign(
                  synth::subscript(dictRef, synth::name(keyName, range), range),
                  synth::name(valueName, range), range)}, {},
              range));
          emitStatement(*synth::assign(
              synth::name(resultName, range),
              synth::subscript(dictRef, synth::name(keyName, range), range),
              range));
          auto bound = values.find(resultName);
          if (bound == values.end() || !bound->second.value)
            return rejectSugar(
                "cannot lower dict.setdefault() over this dict");
          return bound->second;
        });
  }

  // popitem(): LIFO removal per the compact-dict insertion order — the keys
  // snapshot gives the last key, pop removes it. O(len) against CPython's
  // O(1) (a documented performance-only deviation).
  if (argCount != 0)
    return rejectSugar("dict.popitem() takes no arguments");
  std::string keysName = scratch("pks");
  std::string elementName = scratch("px");
  std::string keyName = scratch("pk");
  std::string valueName = scratch("pv");
  return withPrologue(
      {keysName, elementName, keyName, valueName},
      [&]() -> std::optional<Value> {
        NodePtr emptyTest =
            synth::compare(synth::lenCall(dictRef, range), "Eq", synth::intConstant(0, range),
                        range);
        NodePtr keyError =
            synth::call(synth::name("KeyError", range),
                     {synth::strConstant("popitem(): dictionary is empty", range)},
                     range);
        parser::NodePtr raiseNode = synth::raiseStmt(std::move(keyError), range);
        emitStatement(*synth::ifStmt(std::move(emptyTest), {std::move(raiseNode)}, {},
                              range));
        NodePtr comprehension = parser::makeNode("comprehension", range);
        parser::addField(*comprehension, "target",
                         synth::name(elementName, range));
        parser::addField(*comprehension, "iter", dictRef);
        parser::addField(*comprehension, "ifs", std::vector<NodePtr>{});
        parser::addField(*comprehension, "is_async", std::int64_t{0});
        NodePtr keysComp = parser::makeNode("ListComp", range);
        parser::addField(*keysComp, "elt", synth::name(elementName, range));
        parser::addField(*keysComp, "generators",
                         std::vector<NodePtr>{std::move(comprehension)});
        emitStatement(*synth::assign(synth::name(keysName, range),
                                  std::move(keysComp), range));
        emitStatement(*synth::assign(
            synth::name(keyName, range),
            synth::subscript(synth::name(keysName, range),
                          synth::binOp(synth::lenCall(synth::name(keysName, range), range),
                                    "Sub", synth::intConstant(1, range), range),
                          range),
            range));
        emitStatement(*synth::assign(
            synth::name(valueName, range),
            synth::methodCall(dictRef, "pop", {synth::name(keyName, range)}, range),
            range));
        NodePtr pair = synth::tuple(
            {synth::name(keyName, range), synth::name(valueName, range)}, range);
        return emitExpr(pair.get());
      });
}

// ---------------------------------------------------------------------------
// sorted(xs, key=, reverse=) / list.sort(key=, reverse=): decorate-sort-
// undecorate over the native stable sort. Pairs are (key, index) — the index
// makes ties positionally unique, so the element type itself never needs an
// ordering. reverse=True keeps CPython's stability contract (equal keys stay
// in original order) by decorating with the REVERSED index, sorting
// ascending, and reversing the output.
// ---------------------------------------------------------------------------

namespace {

// Deep AST clone substituting NAME references. Lambdas beta-reduce by
// SUBSTITUTION here (not parameter assignment): a comprehension element is
// a single expression, and a sort key lambda's body is a pure expression of
// its parameter. Nested binders re-binding the same name stop the walk.
NodePtr cloneSubstituting(const parser::Node &node, llvm::StringRef name,
                          const NodePtr &replacement) {
  if (node.kind == "Name" &&
      llvm::StringRef(ast::nameSpelling(node).data(),
                      ast::nameSpelling(node).size()) == name)
    return replacement;
  NodePtr clone = parser::makeNode(node.kind, node.range);
  bool rebinds = node.kind == "Lambda" || node.kind == "FunctionDef" ||
                 node.kind == "AsyncFunctionDef";
  for (const parser::Field &field : node.fields) {
    if (const auto *child = std::get_if<NodePtr>(&field.value)) {
      if (*child && !rebinds)
        parser::addField(*clone, field.name,
                         cloneSubstituting(**child, name, replacement));
      else
        parser::addField(*clone, field.name, field.value);
      continue;
    }
    if (const auto *children =
            std::get_if<std::vector<NodePtr>>(&field.value)) {
      if (rebinds) {
        parser::addField(*clone, field.name, field.value);
        continue;
      }
      std::vector<NodePtr> mapped;
      mapped.reserve(children->size());
      for (const NodePtr &child : *children)
        mapped.push_back(child ? cloneSubstituting(*child, name, replacement)
                               : child);
      parser::addField(*clone, field.name, std::move(mapped));
      continue;
    }
    parser::addField(*clone, field.name, field.value);
  }
  return clone;
}

} // namespace

// Emits the DSU statements; returns the name holding the sorted result list.
std::optional<std::string> ModuleEmitter::emitDsuSortStatements(
    const parser::Node &anchor, NodePtr source, const LazyCallable *key,
    bool reverse, unsigned serial,
    llvm::SmallVectorImpl<std::string> &scratchNames) {
  parser::SourceRange range = anchor.range;
  auto scratch = [&](const char *stem) {
    std::string name =
        "__lysort" + std::to_string(serial) + "_" + std::string(stem);
    scratchNames.push_back(name);
    return name;
  };
  std::string listName = scratch("lst");
  std::string copyElement = scratch("c");
  // __lst = SRC.copy() for lists (the comprehension-built copy's bundle
  // trips the unwind release placement when a rebind loop follows), the
  // [__c for __c in SRC] materialization for every other iterable.
  {
    auto contract = mlir::dyn_cast_if_present<py::ContractType>(
        types.widenLiteral(types.inferExpr(source.get())));
    if (contract && contract.getContractName() == "builtins.list") {
      emitStatement(*synth::assign(
          synth::name(listName, range),
          synth::methodCall(std::move(source), "copy", {}, range), range));
    } else {
      NodePtr comprehension = parser::makeNode("comprehension", range);
      parser::addField(*comprehension, "target",
                       synth::name(copyElement, range));
      parser::addField(*comprehension, "iter", std::move(source));
      parser::addField(*comprehension, "ifs", std::vector<NodePtr>{});
      parser::addField(*comprehension, "is_async", std::int64_t{0});
      NodePtr copyComp = parser::makeNode("ListComp", range);
      parser::addField(*copyComp, "elt", synth::name(copyElement, range));
      parser::addField(*copyComp, "generators",
                       std::vector<NodePtr>{std::move(comprehension)});
      emitStatement(*synth::assign(synth::name(listName, range),
                                std::move(copyComp), range));
    }
  }
  if (!key) {
    // ⭐ CPython's own shape for a keyless reverse: `L.reverse(); L.sort();
    // L.reverse()`. The decorate-sort-undecorate below breaks ties with the
    // index, and that only helps when the two keys compare EQUAL -- for a
    // class whose `__lt__` answers False both ways the elements are neither
    // less nor equal, so the sort leaves them in input order and the backward
    // undecorate then swapped them:
    //
    //     class T:
    //         def __lt__(self, other: "T") -> bool: return False
    //     sorted([T("a"), T("b")], reverse=True)[0]   # was b; CPython gives a
    //
    // Reversing around a stable ascending sort needs no tie-break at all,
    // which is why CPython does it that way.
    auto call = [&](const char *method) {
      NodePtr statement = parser::makeNode("Expr", range);
      parser::addField(
          *statement, "value",
          synth::methodCall(synth::name(listName, range), method, {}, range));
      emitStatement(*statement);
    };
    if (reverse)
      call("reverse");
    call("sort");
    if (reverse)
      call("reverse");
    return listName;
  }

  // Key application as a pure EXPRESSION: named callables re-spell, lambdas
  // beta-reduce by substitution — a comprehension element cannot carry
  // statement prologues, and a fully-typed pairs comprehension is what lets
  // the undecorate step see tuple[K, int] elements.
  auto keyExprFor = [&](NodePtr argument) -> std::optional<NodePtr> {
    if (!key)
      return argument;
    if (key->callee)
      return synth::call(key->callee, {std::move(argument)}, range);
    if (key->lambdaParams.size() != 1 || !key->lambdaBody) {
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, anchor.range.start,
          "sort key lambda must take exactly one parameter"});
      return std::nullopt;
    }
    return cloneSubstituting(*key->lambdaBody, key->lambdaParams.front(),
                             argument);
  };

  std::string pairsName = scratch("ps");
  std::string indexName = scratch("i");
  std::string lengthName = scratch("n");
  emitStatement(*synth::assign(synth::name(lengthName, range),
                            synth::lenCall(synth::name(listName, range), range),
                            range));

  // The scratch lists start empty, so their element types must be SPELLED —
  // an append-refined empty literal types at the bundle level only, and the
  // undecorate step statically indexes the pair tuples. The concrete key and
  // element types are inferred from probe expressions and bound to scratch
  // type names the annotations reference (the generic-specialization trick).
  std::string keyTypeName = scratch("K");
  std::string elementTypeName = scratch("T");
  {
    NodePtr elementProbe = synth::subscript(synth::name(listName, range),
                                         synth::intConstant(0, range), range);
    std::optional<NodePtr> keyProbe = keyExprFor(elementProbe);
    if (!keyProbe)
      return std::nullopt;
    mlir::Type elementType =
        types.widenLiteral(types.inferExpr(elementProbe.get()));
    mlir::Type keyType =
        types.widenLiteral(types.inferExpr(keyProbe->get()));
    if (!elementType || !keyType) {
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, anchor.range.start,
          "cannot infer the sort key type for this iterable"});
      return std::nullopt;
    }
    // bindSymbol, not bindLocalSymbol: at module scope there is no local
    // scope stack, and the serial-unique names cannot collide.
    types.bindSymbol(keyTypeName, keyType);
    types.bindSymbol(elementTypeName, elementType);
  }
  auto emitTypedEmptyList = [&](const std::string &name,
                                NodePtr annotation) {
    NodePtr emptyList = parser::makeNode("List", range);
    parser::addField(*emptyList, "elts", std::vector<NodePtr>{});
    NodePtr annAssign = parser::makeNode("AnnAssign", range);
    parser::addField(*annAssign, "target", synth::name(name, range));
    parser::addField(*annAssign, "annotation", std::move(annotation));
    parser::addField(*annAssign, "value", std::move(emptyList));
    parser::addField(*annAssign, "simple", std::int64_t{1});
    emitStatement(*annAssign);
  };

  // Statement-level decoration (comprehensions would need the strict
  // inference to type the pairs, and a may-raise key call inside a
  // comprehension trips the range-iterator unwind bookkeeping):
  //   __ps: list[tuple[K, int]] = []; __i = 0
  //   while __i < __n: __ps.append((KEY(__lst[__i]), IDX)); __i += 1
  {
    emitTypedEmptyList(
        pairsName,
        synth::subscript(synth::name("list", range),
                      synth::subscript(synth::name("tuple", range),
                                    synth::tuple({synth::name(keyTypeName, range),
                                               synth::name("int", range)},
                                              range),
                                    range),
                      range));
    emitStatement(*synth::assign(synth::name(indexName, range),
                              synth::intConstant(0, range), range));
    NodePtr decoratedIndex =
        reverse ? synth::binOp(synth::binOp(synth::name(lengthName, range), "Sub",
                                      synth::intConstant(1, range), range),
                            "Sub", synth::name(indexName, range), range)
                : synth::binOp(synth::name(indexName, range), "Add",
                            synth::intConstant(0, range), range);
    std::optional<NodePtr> keyExpr = keyExprFor(synth::subscript(
        synth::name(listName, range), synth::name(indexName, range), range));
    if (!keyExpr)
      return std::nullopt;
    NodePtr appendStatement = parser::makeNode("Expr", range);
    parser::addField(
        *appendStatement, "value",
        synth::methodCall(synth::name(pairsName, range), "append",
                       {synth::tuple({std::move(*keyExpr),
                                   std::move(decoratedIndex)},
                                  range)},
                       range));
    std::vector<NodePtr> loopBody;
    loopBody.push_back(std::move(appendStatement));
    loopBody.push_back(synth::assign(
        synth::name(indexName, range),
        synth::binOp(synth::name(indexName, range), "Add", synth::intConstant(1, range),
                  range),
        range));
    emitWhile(*synth::whileStmt(synth::compare(synth::name(indexName, range), "Lt",
                                     synth::name(lengthName, range), range),
                         std::move(loopBody), {}, range));
  }
  {
    parser::NodePtr sortStatement = synth::exprStmt(synth::methodCall(synth::name(pairsName, range), "sort", {},
                                    range), range);
    emitStatement(*sortStatement);
  }

  // Undecorate by index. reverse=True walks the sorted pairs BACKWARD (the
  // decorated indices were reversed, so backward emission both restores the
  // descending key order and lands equal keys in original order — no
  // list.reverse() call, no iterator over the pairs).
  //   __out = []; __q = 0|__n
  //   while ...: __out.append(__lst[<undecorated __ps[__q][1]>])
  std::string outName = scratch("out");
  std::string cursorName = scratch("q");
  {
    emitTypedEmptyList(outName,
                       synth::subscript(synth::name("list", range),
                                     synth::name(elementTypeName, range),
                                     range));
    emitStatement(*synth::assign(synth::name(cursorName, range),
                              reverse ? synth::name(lengthName, range)
                                      : synth::intConstant(0, range),
                              range));
    NodePtr storedIndex = synth::subscript(
        synth::subscript(synth::name(pairsName, range), synth::name(cursorName, range),
                      range),
        synth::intConstant(1, range), range);
    NodePtr sourceIndex =
        reverse ? synth::binOp(synth::binOp(synth::name(lengthName, range), "Sub",
                                      synth::intConstant(1, range), range),
                            "Sub", std::move(storedIndex), range)
                : std::move(storedIndex);
    NodePtr appendStatement = parser::makeNode("Expr", range);
    parser::addField(
        *appendStatement, "value",
        synth::methodCall(synth::name(outName, range), "append",
                       {synth::subscript(synth::name(listName, range),
                                      std::move(sourceIndex), range)},
                       range));
    std::vector<NodePtr> body;
    if (reverse) {
      body.push_back(synth::assign(
          synth::name(cursorName, range),
          synth::binOp(synth::name(cursorName, range), "Sub",
                    synth::intConstant(1, range), range),
          range));
      body.push_back(std::move(appendStatement));
      emitWhile(*synth::whileStmt(synth::compare(synth::name(cursorName, range), "Gt",
                                       synth::intConstant(0, range), range),
                           std::move(body), {}, range));
    } else {
      body.push_back(std::move(appendStatement));
      body.push_back(synth::assign(
          synth::name(cursorName, range),
          synth::binOp(synth::name(cursorName, range), "Add",
                    synth::intConstant(1, range), range),
          range));
      emitWhile(*synth::whileStmt(synth::compare(synth::name(cursorName, range), "Lt",
                                       synth::name(lengthName, range), range),
                           std::move(body), {}, range));
    }
  }
  return outName;
}

std::optional<Value>
ModuleEmitter::tryEmitSortSugar(const parser::Node &expr,
                                const parser::Node *calleeNode) {
  if (!calleeNode)
    return std::nullopt;
  const auto *keywords = ast::nodeList(expr, "keywords");
  if (!keywords || keywords->empty()) {
    // Keyword-less sorted(<non-list iterable>): the manifest builtin takes
    // a list, so materialize through the list constructor first —
    // sorted(d), sorted(d.keys()), sorted(gen) become sorted(list(...)).
    // NOTE: sorted IS a manifest free function (types.lookupSymbol resolves
    // it), so only user shadowing disables the rewrite — same guard as the
    // keyword sugar below.
    if (calleeNode->kind == "Name" &&
        ast::nameSpelling(*calleeNode) == "sorted" &&
        !programBindsName("sorted") && !genericFunctions.count("sorted") &&
        !types.lookupClass("sorted")) {
      const auto *args = ast::nodeList(expr, "args");
      // The rewrite must not re-wrap its own product: inference has no case
      // for the list() call form, so `sorted(list(X))` would recurse.
      bool alreadyListCall = false;
      if (args && args->size() == 1 && (*args)[0] &&
          (*args)[0]->kind == "Call")
        if (const parser::Node *inner = ast::node(*(*args)[0], "func"))
          alreadyListCall = inner->kind == "Name" &&
                            ast::nameSpelling(*inner) == "list";
      if (args && args->size() == 1 && (*args)[0] &&
          (*args)[0]->kind != "Starred" && !alreadyListCall) {
        auto argContract = mlir::dyn_cast_if_present<py::ContractType>(
            types.widenLiteral(types.inferExpr((*args)[0].get())));
        if (!argContract ||
            argContract.getContractName() != "builtins.list") {
          parser::SourceRange range = expr.range;
          NodePtr wrapped = synth::call(
              synth::name("sorted", range),
              {synth::call(synth::name("list", range), {(*args)[0]}, range)},
              range);
          return emitExpr(wrapped.get());
        }
      }
    }
    return std::nullopt; // keyword-less forms keep their native paths
  }
  const auto *args = ast::nodeList(expr, "args");

  // NOTE: sorted IS a manifest free function (types.lookupSymbol resolves
  // it), so only user shadowing disables the sugar.
  bool isSorted = calleeNode->kind == "Name" &&
                  ast::nameSpelling(*calleeNode) == "sorted" &&
                  !programBindsName("sorted") &&
                  !genericFunctions.count("sorted") &&
                  !types.lookupClass("sorted");
  bool isListSort = false;
  NodePtr receiver;
  if (!isSorted && calleeNode->kind == "Attribute" &&
      ast::string(*calleeNode, "attr").value_or("") == "sort") {
    const parser::Field *receiverField =
        parser::findField(*calleeNode, "value");
    if (receiverField &&
        std::holds_alternative<NodePtr>(receiverField->value)) {
      receiver = std::get<NodePtr>(receiverField->value);
      if (receiver) {
        auto contract = mlir::dyn_cast_if_present<py::ContractType>(
            types.widenLiteral(types.inferExpr(receiver.get())));
        isListSort =
            contract && contract.getContractName() == "builtins.list";
      }
    }
  }
  if (!isSorted && !isListSort)
    return std::nullopt;

  auto rejectSort = [&](llvm::StringRef reason) -> std::optional<Value> {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, expr.range.start, std::string(reason)});
    return emitNone(expr);
  };
  if (isSorted && (!args || args->size() != 1))
    return rejectSort("sorted() takes exactly one iterable plus keyword "
                      "arguments");
  if (isListSort && args && !args->empty())
    return rejectSort("list.sort() takes keyword arguments only");

  LazyCallable key;
  bool hasKey = false;
  bool reverse = false;
  for (const NodePtr &keyword : *keywords) {
    if (!keyword)
      return std::nullopt;
    auto name = ast::string(*keyword, "arg");
    const parser::Field *valueField = parser::findField(*keyword, "value");
    if (!name || !valueField ||
        !std::holds_alternative<NodePtr>(valueField->value))
      return rejectSort("sort()/sorted() keyword arguments must be key= and "
                        "reverse=");
    NodePtr value = std::get<NodePtr>(valueField->value);
    if (*name == "key") {
      if (value && value->kind == "Constant" &&
          ast::isNoneField(*value, "value"))
        continue; // key=None is the default
      if (!lazyCallableParts(expr, value, key))
        return emitNone(expr);
      hasKey = true;
    } else if (*name == "reverse") {
      if (!value || value->kind != "Constant" ||
          !ast::boolean(*value, "value").has_value())
        return rejectSort("sort()/sorted() reverse= must be a literal bool "
                          "(a runtime flag would need both orderings "
                          "compiled)");
      reverse = *ast::boolean(*value, "value");
    } else {
      return rejectSort("sort()/sorted() got an unexpected keyword argument");
    }
  }

  unsigned serial = ++listCompCounter;
  NodePtr source = isSorted ? (*args)[0] : receiver;
  llvm::SmallVector<std::string, 8> scratchNames;
  for (const std::string &param : key.lambdaParams)
    scratchNames.push_back(param);
  std::optional<Value> result;
  runWithScratchNames(scratchNames, [&] {
    // scratchNames grows inside emitDsuSortStatements; capture the rest via
    // a second scope.
    llvm::SmallVector<std::string, 8> innerNames;
    std::optional<std::string> outName = emitDsuSortStatements(
        expr, source, hasKey ? &key : nullptr, reverse, serial, innerNames);
    runWithScratchNames(innerNames, [&] {
      if (!outName) {
        result = emitNone(expr);
        return;
      }
      if (isSorted) {
        auto bound = values.find(*outName);
        result = bound != values.end() && bound->second.value
                     ? std::optional<Value>(bound->second)
                     : std::nullopt;
        if (!result)
          result = emitNone(expr);
        return;
      }
      // list.sort(): write the permutation back in place through
      // clear+extend (loop-free: a per-element setitem loop followed by a
      // later may-raise use trips the unwind release placement).
      parser::SourceRange range = expr.range;
      parser::NodePtr clearStatement = synth::exprStmt(synth::methodCall(receiver, "clear", {}, range), range);
      emitStatement(*clearStatement);
      NodePtr extendStatement = parser::makeNode("Expr", range);
      parser::addField(
          *extendStatement, "value",
          synth::methodCall(receiver, "extend", {synth::name(*outName, range)},
                         range));
      emitStatement(*extendStatement);
      result = emitNone(expr);
    });
  });
  // The DSU scratch locals leak out of the inner scope restore only if the
  // outer lambda captured stale entries; nothing else to do here.
  return result;
}

// ---------------------------------------------------------------------------
// str.maketrans(x, y) / str.translate(table): pure compositions over ord/
// chr/dict lookups. maketrans builds {ord(x[i]): ord(y[i])}; translate maps
// each code point through the table (int values re-encode through chr, str
// values substitute directly, missing keys pass through) and joins.
// ---------------------------------------------------------------------------
std::optional<Value>
ModuleEmitter::tryEmitStrTranslateSugar(const parser::Node &expr,
                                        const parser::Node *calleeNode) {
  if (!calleeNode || calleeNode->kind != "Attribute")
    return std::nullopt;
  auto attr = ast::string(*calleeNode, "attr");
  const parser::Field *receiverField = parser::findField(*calleeNode, "value");
  if (!attr || !receiverField ||
      !std::holds_alternative<NodePtr>(receiverField->value))
    return std::nullopt;
  NodePtr receiver = std::get<NodePtr>(receiverField->value);
  if (!receiver)
    return std::nullopt;
  const auto *args = ast::nodeList(expr, "args");
  const auto *keywords = ast::nodeList(expr, "keywords");
  if (keywords && !keywords->empty())
    return std::nullopt;
  parser::SourceRange range = expr.range;
  unsigned serial = ++listCompCounter;
  auto scratch = [&](const char *stem) {
    return "__lytr" + std::to_string(serial) + "_" + std::string(stem);
  };
  auto rejectTranslate = [&](llvm::StringRef reason) -> std::optional<Value> {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, expr.range.start, std::string(reason)});
    return emitNone(expr);
  };

  if (*attr == "maketrans" && receiver->kind == "Name" &&
      ast::nameSpelling(*receiver) == "str" && !values.count("str")) {
    if (!args || args->size() != 2)
      return rejectTranslate(
          "str.maketrans currently supports the two-string form "
          "(equal-length from/to)");
    for (const NodePtr &argument : *args)
      if (!argument ||
          types.widenLiteral(types.inferExpr(argument.get())) !=
              types.strType())
        return rejectTranslate("str.maketrans arguments must be strings");
    std::string fromName = scratch("x");
    std::string toName = scratch("y");
    std::string indexName = scratch("i");
    std::optional<Value> result;
    runWithScratchNames({fromName, toName, indexName}, [&] {
      emitStatement(*synth::assign(synth::name(fromName, range), (*args)[0],
                                range));
      emitStatement(*synth::assign(synth::name(toName, range), (*args)[1], range));
      // if len(x) != len(y): raise ValueError(...)
      NodePtr lengthTest =
          synth::compare(synth::lenCall(synth::name(fromName, range), range), "NotEq",
                      synth::lenCall(synth::name(toName, range), range), range);
      NodePtr valueError = synth::call(
          synth::name("ValueError", range),
          {synth::strConstant(
              "the first two maketrans arguments must have equal length",
              range)},
          range);
      parser::NodePtr raiseNode = synth::raiseStmt(std::move(valueError), range);
      emitStatement(*synth::ifStmt(std::move(lengthTest), {std::move(raiseNode)}, {},
                            range));
      // {ord(x[i]): ord(y[i]) for i in range(len(x))}
      auto ordAt = [&](const std::string &sourceName) {
        return synth::call(synth::name("ord", range),
                        {synth::subscript(synth::name(sourceName, range),
                                       synth::name(indexName, range), range)},
                        range);
      };
      NodePtr comprehension = parser::makeNode("comprehension", range);
      parser::addField(*comprehension, "target", synth::name(indexName, range));
      parser::addField(*comprehension, "iter",
                       synth::call(synth::name("range", range),
                                {synth::lenCall(synth::name(fromName, range), range)},
                                range));
      parser::addField(*comprehension, "ifs", std::vector<NodePtr>{});
      parser::addField(*comprehension, "is_async", std::int64_t{0});
      NodePtr tableComp = parser::makeNode("DictComp", range);
      parser::addField(*tableComp, "key", ordAt(fromName));
      parser::addField(*tableComp, "value", ordAt(toName));
      parser::addField(*tableComp, "generators",
                       std::vector<NodePtr>{std::move(comprehension)});
      result = emitExpr(tableComp.get());
    });
    return result;
  }

  if (*attr != "translate")
    return std::nullopt;
  if (types.widenLiteral(types.inferExpr(receiver.get())) != types.strType())
    return std::nullopt;
  if (!args || args->size() != 1 || !args->front())
    return rejectTranslate("str.translate takes exactly one table argument");
  // A statically empty table maps nothing: the (immutable) receiver IS the
  // result, and an empty literal's key/value types would stay unbound.
  if ((*args)[0]->kind == "Dict") {
    const auto *tableKeys = ast::nodeList(*(*args)[0], "keys");
    if (!tableKeys || tableKeys->empty())
      return coerceValue(emitExpr(receiver.get()), types.strType(), expr);
  }
  auto table = mlir::dyn_cast_if_present<py::ContractType>(
      types.widenLiteral(types.inferExpr(args->front().get())));
  if (!table || table.getContractName() != "builtins.dict" ||
      table.getArguments().size() != 2)
    return rejectTranslate("str.translate requires a dict table");
  mlir::Type keyType = types.widenLiteral(table.getArguments()[0]);
  mlir::Type valueType = types.widenLiteral(table.getArguments()[1]);
  if (keyType != types.intType())
    return rejectTranslate("str.translate table keys must be int "
                           "(code points)");
  bool intValues = valueType == types.intType();
  if (!intValues && valueType != types.strType())
    return rejectTranslate("str.translate table values must be int or str "
                           "(None deletion is not supported yet)");

  std::string sourceName = scratch("s");
  std::string tableName = scratch("t");
  std::string partsName = scratch("p");
  std::string charName = scratch("c");
  std::string ordName = scratch("o");
  bool sourceIsName = receiver->kind == "Name";
  bool tableIsName = (*args)[0]->kind == "Name";
  NodePtr sourceRef = sourceIsName ? receiver : synth::name(sourceName, range);
  NodePtr tableRef = tableIsName ? (*args)[0] : synth::name(tableName, range);
  std::optional<Value> result;
  runWithScratchNames({sourceName, tableName, partsName, charName, ordName},
                      [&] {
    if (!sourceIsName)
      emitStatement(*synth::assign(synth::name(sourceName, range), receiver,
                                range));
    if (!tableIsName)
      emitStatement(*synth::assign(synth::name(tableName, range), (*args)[0],
                                range));
    // __p: list[str] = []
    NodePtr emptyList = parser::makeNode("List", range);
    parser::addField(*emptyList, "elts", std::vector<NodePtr>{});
    NodePtr annAssign = parser::makeNode("AnnAssign", range);
    parser::addField(*annAssign, "target", synth::name(partsName, range));
    parser::addField(*annAssign, "annotation",
                     synth::subscript(synth::name("list", range),
                                   synth::name("str", range), range));
    parser::addField(*annAssign, "value", std::move(emptyList));
    parser::addField(*annAssign, "simple", std::int64_t{1});
    emitStatement(*annAssign);
    // for __c in s: __o = ord(__c); mapped/pass-through appends
    NodePtr mapped = synth::subscript(tableRef, synth::name(ordName, range), range);
    if (intValues)
      mapped = synth::call(synth::name("chr", range), {std::move(mapped)}, range);
    parser::NodePtr appendMapped = synth::exprStmt(synth::methodCall(synth::name(partsName, range), "append",
                                    {std::move(mapped)}, range), range);
    parser::NodePtr appendPlain = synth::exprStmt(synth::methodCall(synth::name(partsName, range), "append",
                                    {synth::name(charName, range)}, range), range);
    NodePtr branch = synth::ifStmt(
        synth::compareIn(synth::name(ordName, range), tableRef, range),
        {std::move(appendMapped)}, {}, range);
    parser::addField(*branch, "orelse",
                     std::vector<NodePtr>{std::move(appendPlain)});
    std::vector<NodePtr> body{
        synth::assign(synth::name(ordName, range),
                   synth::call(synth::name("ord", range),
                            {synth::name(charName, range)}, range),
                   range),
        std::move(branch)};
    emitFor(*synth::forStmt(synth::name(charName, range), sourceRef, std::move(body),
                     {}, range));
    // "".join(__p)
    NodePtr joinCall =
        synth::methodCall(synth::strConstant("", range), "join",
                       {synth::name(partsName, range)}, range);
    result = emitExpr(joinCall.get());
  });
  return result;
}

// `x in d.keys()` is key membership; `v in d.values()` scans the values;
// `(k, v) in d.items()` is key membership plus value equality. The views
// have no runtime object, so the comparison rewrites against the dict
// before any operand is emitted.
std::optional<Value>
ModuleEmitter::tryEmitDictViewMembership(const parser::Node &expr) {
  const auto *comparators = ast::nodeList(expr, "comparators");
  const auto *ops = ast::nodeList(expr, "ops");
  if (!comparators || comparators->size() != 1 || !ops || ops->size() != 1 ||
      !(*comparators)[0] || !(*ops)[0])
    return std::nullopt;
  bool negated = (*ops)[0]->kind == "NotIn";
  if ((*ops)[0]->kind != "In" && !negated)
    return std::nullopt;
  const parser::Node &comparator = *(*comparators)[0];
  if (comparator.kind != "Call")
    return std::nullopt;
  const parser::Node *viewCallee = ast::node(comparator, "func");
  const auto *viewArgs = ast::nodeList(comparator, "args");
  const auto *viewKeywords = ast::nodeList(comparator, "keywords");
  if (!viewCallee || viewCallee->kind != "Attribute" ||
      (viewArgs && !viewArgs->empty()) ||
      (viewKeywords && !viewKeywords->empty()))
    return std::nullopt;
  auto viewName = ast::string(*viewCallee, "attr");
  if (!viewName || (*viewName != "keys" && *viewName != "values" &&
                    *viewName != "items"))
    return std::nullopt;
  const parser::Field *receiverField = parser::findField(*viewCallee, "value");
  if (!receiverField ||
      !std::holds_alternative<NodePtr>(receiverField->value))
    return std::nullopt;
  NodePtr receiver = std::get<NodePtr>(receiverField->value);
  if (!receiver || !isDictTypedExpr(receiver.get()))
    return std::nullopt;
  const parser::Field *leftField = parser::findField(expr, "left");
  if (!leftField || !std::holds_alternative<NodePtr>(leftField->value))
    return std::nullopt;
  NodePtr left = std::get<NodePtr>(leftField->value);
  if (!left)
    return std::nullopt;
  parser::SourceRange range = expr.range;

  if (*viewName == "keys") {
    NodePtr rewritten = synth::compare(left, negated ? "NotIn" : "In", receiver,
                                    range);
    return emitCompare(*rewritten);
  }

  unsigned serial = ++listCompCounter;
  auto scratch = [&](const char *stem) {
    return "__lyview" + std::to_string(serial) + "_" + std::string(stem);
  };
  std::string dictName = scratch("d");
  bool needsTemp = receiver->kind != "Name";
  NodePtr dictRef = needsTemp ? synth::name(dictName, range) : receiver;
  std::string probeName = scratch("x");
  std::string resultName = scratch("r");
  std::string keyName = scratch("k");
  llvm::SmallVector<std::string, 4> names{probeName, resultName, keyName};
  if (needsTemp)
    names.push_back(dictName);

  std::optional<Value> result;
  runWithScratchNames(names, [&] {
    // Order: CPython evaluates the left operand, then the view expression.
    emitStatement(*synth::assign(synth::name(probeName, range), left, range));
    if (needsTemp)
      emitStatement(*synth::assign(synth::name(dictName, range), receiver, range));
    parser::NodePtr falseInit = synth::constantBool(false, range);
    emitStatement(*synth::assign(synth::name(resultName, range),
                              std::move(falseInit), range));
    parser::NodePtr trueValue = synth::constantBool(true, range);
    if (*viewName == "values") {
      // for __k in d: if d[__k] == __x: __r = True; break
      NodePtr hit = synth::ifStmt(
          synth::compare(synth::subscript(dictRef, synth::name(keyName, range), range),
                      "Eq", synth::name(probeName, range), range),
          {synth::assign(synth::name(resultName, range), std::move(trueValue),
                      range),
           parser::makeNode("Break", range)}, {},
          range);
      emitFor(*synth::forStmt(synth::name(keyName, range), dictRef, {std::move(hit)},
                       {}, range));
    } else {
      // items: __k = __x[0]; if __k in d and d[__k] == __x[1]: __r = True
      emitStatement(*synth::assign(
          synth::name(keyName, range),
          synth::subscript(synth::name(probeName, range), synth::intConstant(0, range),
                        range),
          range));
      NodePtr valueMatches = synth::ifStmt(
          synth::compare(synth::subscript(dictRef, synth::name(keyName, range), range),
                      "Eq",
                      synth::subscript(synth::name(probeName, range),
                                    synth::intConstant(1, range), range),
                      range),
          {synth::assign(synth::name(resultName, range), std::move(trueValue),
                      range)}, {},
          range);
      emitStatement(*synth::ifStmt(
          synth::compareIn(synth::name(keyName, range), dictRef, range),
          {std::move(valueMatches)}, {}, range));
    }
    NodePtr resultExpr = synth::name(resultName, range);
    if (negated) {
      NodePtr notOp = parser::makeNode("Not", range);
      NodePtr flipped = parser::makeNode("UnaryOp", range);
      parser::addField(*flipped, "op", std::move(notOp));
      parser::addField(*flipped, "operand", std::move(resultExpr));
      resultExpr = std::move(flipped);
    }
    result = emitExpr(resultExpr.get());
  });
  return result;
}

// ---------------------------------------------------------------------------
// list()/set()/tuple()/dict() constructors. Iterable construction desugars
// to the comprehension loops the literal forms already use (that keeps
// CPython's per-element evaluation order and set/dict dedup semantics and
// inherits every iterable the loop fusion supports); list-typed sources take
// the O(n) sequence-copy manifest builtins instead, and tuple(t) is the
// identity because CPython returns the immutable argument object itself.
// ---------------------------------------------------------------------------
Value ModuleEmitter::emitConstructorComprehension(const parser::Node &expr,
                                                  parser::NodePtr argNode,
                                                  llvm::StringRef which) {
  parser::SourceRange range = expr.range;
  unsigned serial = ++listCompCounter;
  NodePtr generator = parser::makeNode("comprehension", range);
  parser::addField(*generator, "iter", argNode);
  parser::addField(*generator, "ifs", std::vector<NodePtr>{});
  NodePtr comp;
  if (which == "dict") {
    std::string key = "__ctork" + std::to_string(serial);
    std::string value = "__ctorv" + std::to_string(serial);
    parser::addField(*generator, "target",
                     synth::tuple({synth::name(key, range), synth::name(value, range)},
                               range));
    comp = parser::makeNode("DictComp", range);
    parser::addField(*comp, "key", synth::name(key, range));
    parser::addField(*comp, "value", synth::name(value, range));
    parser::addField(*comp, "generators", std::vector<NodePtr>{generator});
    return emitComprehension(*comp, /*isDict=*/true);
  }
  std::string element = "__ctor" + std::to_string(serial);
  parser::addField(*generator, "target", synth::name(element, range));
  comp = parser::makeNode(which == "set" ? "SetComp" : "ListComp", range);
  parser::addField(*comp, "elt", synth::name(element, range));
  parser::addField(*comp, "generators", std::vector<NodePtr>{generator});
  return emitComprehension(*comp, /*isDict=*/false, /*isSet=*/which == "set");
}

std::optional<Value>
ModuleEmitter::emitListToTupleFreeze(const parser::Node &expr,
                                     const parser::Node &calleeNode,
                                     Value listValue) {
  const py::protocols::Table &table = py::protocols::Table::get(context);
  std::optional<mlir::Type> contract =
      table.freeFunctionContract("builtins.tuple");
  if (!contract) {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, expr.range.start,
        "the builtins manifest does not declare the tuple constructor "
        "contract"});
    return emitNone(expr);
  }
  Value callee = emitBindingRef(calleeNode, "tuple", *contract);
  return emitCallableDispatch(
      expr, callee,
      emitCallOperands(expr, {listValue}, /*includeAstArguments=*/false));
}

std::optional<Value> ModuleEmitter::tryEmitContainerConstructorCall(
    const parser::Node &expr, const parser::Node *calleeNode) {
  if (!calleeNode || calleeNode->kind != "Name")
    return std::nullopt;
  llvm::StringRef ctor = ast::nameSpelling(*calleeNode);
  if (ctor != "list" && ctor != "set" && ctor != "tuple" && ctor != "dict")
    return std::nullopt;
  if (!isBuiltinIteratorName(ctor)) // reuses the user-shadowing guard
    return std::nullopt;

  auto reject = [&](const std::string &message) -> std::optional<Value> {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, expr.range.start, message});
    return emitNone(expr);
  };

  const auto *args = ast::nodeList(expr, "args");
  const auto *keywords = ast::nodeList(expr, "keywords");
  parser::SourceRange range = expr.range;

  if (args)
    for (const NodePtr &arg : *args)
      if (!arg || arg->kind == "Starred")
        return reject(std::string(ctor) +
                      "() does not support starred arguments yet");

  // ⭐ `tuple(p)` / `list(p)` OVER A NamedTuple IS ITS FIELDS, in order. A
  // NamedTuple has no __iter__ here -- it is a source class with named fields,
  // not a runtime tuple -- and the display it spells is what CPython's
  // iteration would produce anyway. `tuple(p)` was "static type
  // !py.contract<"P"> does not provide manifest method '__iter__'".
  //
  // ⛔ Only the two SEQUENCE constructors: a set of the fields would drop
  // duplicates and a dict of them has no keys, so neither is the same answer.
  if ((ctor == "tuple" || ctor == "list") && args && args->size() == 1 &&
      args->front() && (!keywords || keywords->empty()))
    if (auto contract = mlir::dyn_cast_if_present<py::ContractType>(
            types.widenLiteral(types.inferExpr(args->front().get()))))
      if (namedTupleContracts.count(contract.getContractName())) {
        llvm::ArrayRef<std::string> order =
            classFieldOrders[contract.getContractName()];
        if (!order.empty()) {
          std::string tmp =
              "__ntseq" + std::to_string(++syntheticFunctionCounter);
          Value receiver = emitExpr(args->front().get());
          values[tmp] = receiver;
          types.bindSymbol(tmp, receiver.type);
          std::vector<NodePtr> members;
          for (const std::string &field : order)
            members.push_back(
                synth::attribute(synth::name(tmp, range), field, range));
          NodePtr display =
              parser::makeNode(ctor == "tuple" ? "Tuple" : "List", range);
          parser::addField(*display, "elts", std::move(members));
          synthesizedIteratorDefs.push_back(display);
          Value built = emitExpr(display.get());
          values.erase(tmp);
          return built;
        }
      }

  // dict(a=1, b=2): the keyword form builds the literal (values evaluate
  // left to right, CPython's order).
  if (keywords && !keywords->empty()) {
    if (ctor != "dict")
      return reject(std::string(ctor) + "() takes no keyword arguments");
    if (args && !args->empty())
      return reject("dict() with both a positional argument and keyword "
                    "arguments is not supported yet");
    std::vector<NodePtr> keys;
    std::vector<NodePtr> vals;
    for (const NodePtr &keyword : *keywords) {
      auto name = keyword ? ast::string(*keyword, "arg") : std::nullopt;
      const parser::Field *valueField =
          keyword ? parser::findField(*keyword, "value") : nullptr;
      if (!name || !valueField ||
          !std::holds_alternative<NodePtr>(valueField->value) ||
          !std::get<NodePtr>(valueField->value))
        return reject("dict() keyword arguments must be named "
                      "(** splats are not supported yet)");
      keys.push_back(synth::strConstant(std::string(*name), range));
      vals.push_back(std::get<NodePtr>(valueField->value));
    }
    NodePtr literal = parser::makeNode("Dict", range);
    parser::addField(*literal, "keys", std::move(keys));
    parser::addField(*literal, "values", std::move(vals));
    return emitExpr(literal.get());
  }

  if (!args || args->empty()) {
    // Empty construction is the empty literal. set() has no literal
    // spelling: its empty pack mirrors the one the set-comprehension build
    // starts from, typed by the expectation when the inference sees one.
    if (ctor == "set") {
      mlir::Type resultType = types.inferExpr(&expr);
      auto contract =
          mlir::dyn_cast_if_present<py::ContractType>(resultType);
      if (!contract || contract.getContractName() != "builtins.set")
        resultType = py::ContractType::get(builder.getContext(),
                                           "builtins.set", {types.object()});
      auto pack = py::PackOp::create(builder, loc(expr), resultType,
                                     mlir::ValueRange{});
      return Value{pack.getResult(), resultType};
    }
    NodePtr literal;
    if (ctor == "dict") {
      literal = parser::makeNode("Dict", range);
      parser::addField(*literal, "keys", std::vector<NodePtr>{});
      parser::addField(*literal, "values", std::vector<NodePtr>{});
    } else {
      literal = parser::makeNode(ctor == "list" ? "List" : "Tuple", range);
      parser::addField(*literal, "elts", std::vector<NodePtr>{});
    }
    return emitExpr(literal.get());
  }

  if (args->size() != 1)
    return reject(std::string(ctor) + " expected at most 1 argument, got " +
                  std::to_string(args->size()));

  NodePtr argNode = (*args)[0];

  // A genexpr argument reuses the genexpr's own element/generator chain
  // (the existing list(<genexpr>) comprehension route, widened to the other
  // constructors).
  if (argNode->kind == "GeneratorExp") {
    const parser::Field *eltField = parser::findField(*argNode, "elt");
    const auto *generators = ast::nodeList(*argNode, "generators");
    if (!eltField || !std::holds_alternative<NodePtr>(eltField->value) ||
        !std::get<NodePtr>(eltField->value) || !generators)
      return reject("malformed generator expression argument");
    NodePtr elt = std::get<NodePtr>(eltField->value);
    if (ctor == "dict") {
      const auto *pair =
          elt->kind == "Tuple" ? ast::nodeList(*elt, "elts") : nullptr;
      if (!pair || pair->size() != 2 || !(*pair)[0] || !(*pair)[1])
        return reject("dict() over a generator expression requires "
                      "(key, value) tuple elements");
      NodePtr comp = parser::makeNode("DictComp", argNode->range);
      parser::addField(*comp, "key", (*pair)[0]);
      parser::addField(*comp, "value", (*pair)[1]);
      parser::addField(*comp, "generators", *generators);
      return emitComprehension(*comp, /*isDict=*/true);
    }
    NodePtr comp = parser::makeNode(ctor == "set" ? "SetComp" : "ListComp",
                                    argNode->range);
    parser::addField(*comp, "elt", elt);
    parser::addField(*comp, "generators", *generators);
    Value collected =
        emitComprehension(*comp, /*isDict=*/false, /*isSet=*/ctor == "set");
    if (ctor != "tuple")
      return collected;
    return emitListToTupleFreeze(expr, *calleeNode, collected);
  }

  mlir::Type argumentType =
      types.widenLiteral(types.inferExpr(argNode.get()));
  auto argContract =
      mlir::dyn_cast_if_present<py::ContractType>(argumentType);
  llvm::StringRef argClass =
      argContract ? argContract.getContractName() : llvm::StringRef();

  // ⭐ A call whose type the INFERENCE cannot see, but the emission can: the
  // lazy builtin iterators (zip/enumerate/map/filter/reversed) synthesize a
  // generator function when they are emitted, so `zip(...)` is
  // `builtins.object` to inferExpr and a typed generator to emitExpr. The
  // comprehension below reads the static type, so `list(zip(a, b))` was
  // "builtins.object does not provide __iter__" while the SAME program with
  // the call bound to a name first -- `z = zip(a, b); list(z)` -- worked.
  // Binding it here is that spelling, written once.
  //
  // ⛔ Why NOT teach inferExpr the five return types instead: it would have
  // to reproduce what the synthesis decides (zip's tuple arity, map's
  // callable result, enumerate's index pair) from the argument nodes, and
  // the two would drift. The value the synthesis produced already carries it.
  if (argNode->kind == "Call" &&
      !types.inferMethodCallWithEvidence(argumentType, "__iter__", {})) {
    std::string sourceName =
        "__lyctorsrc" + std::to_string(++listCompCounter);
    std::optional<Value> built;
    runWithScratchNames({sourceName}, [&] {
      emitStatement(*synth::assign(synth::name(sourceName, range), argNode, range));
      if (values.find(sourceName) == values.end())
        return;
      NodePtr bound = synth::name(sourceName, range);
      if (ctor == "dict")
        built = emitConstructorComprehension(expr, bound, "dict");
      else if (ctor == "set")
        built = emitConstructorComprehension(expr, bound, "set");
      else if (ctor == "list")
        built = emitConstructorComprehension(expr, bound, "list");
      else
        built = emitListToTupleFreeze(
            expr, *calleeNode,
            emitConstructorComprehension(expr, bound, "list"));
    });
    if (built)
      return *built;
  }

  // CPython's tuple(t) returns t itself: tuples are immutable.
  if (ctor == "tuple" && argClass == "builtins.tuple")
    return emitExpr(argNode.get());

  if (ctor == "dict") {
    if (argClass == "builtins.dict") {
      NodePtr copy = synth::methodCall(argNode, "copy", {}, range);
      return emitExpr(copy.get());
    }
    return emitConstructorComprehension(expr, argNode, "dict");
  }

  if ((ctor == "list" || ctor == "tuple") && argClass == "builtins.list") {
    const py::protocols::Table &table = py::protocols::Table::get(context);
    std::optional<mlir::Type> contract = table.freeFunctionContract(
        ctor == "list" ? "builtins.list" : "builtins.tuple");
    if (!contract)
      return reject("the builtins manifest does not declare the " +
                    std::string(ctor) + " constructor contract");
    Value callee = emitBindingRef(*calleeNode, ctor, *contract);
    return emitCallableDispatch(
        expr, callee,
        emitCallOperands(expr, {}, /*includeAstArguments=*/true,
                         mlir::dyn_cast_if_present<py::CallableType>(
                             callee.type)));
  }

  if (ctor == "set")
    return emitConstructorComprehension(expr, argNode, "set");
  Value collected = emitConstructorComprehension(expr, argNode, "list");
  if (ctor == "list")
    return collected;
  return emitListToTupleFreeze(expr, *calleeNode, collected);
}

// ---------------------------------------------------------------------------
// itertools desugars. The itertools manifest (runtime/modules/itertools.mlir)
// declares the module contract only; every call compiles here. For-position
// consumption fuses into rewritten loops (the count/islice/takewhile family,
// including infinite and generator-backed sources); value position
// synthesizes per-call-site generator functions over indexable sequences
// exactly like the enumerate/zip machinery above. Names neither layer can
// express are rejected with a diagnostic — an itertools call must never fall
// through to generic dispatch, because no native implementation exists.
// ---------------------------------------------------------------------------

namespace {









std::optional<std::int64_t> constantInt(const parser::Node *expr) {
  if (!expr || expr->kind != "Constant")
    return std::nullopt;
  return ast::integer(*expr, "value");
}

bool isNoneConstant(const parser::Node *expr) {
  return expr && expr->kind == "Constant" && ast::isNoneField(*expr, "value");
}

// break/continue that would bind to the fused loop. Nested loops and
// callables keep their own escapes; If/Try/With bodies forward them.
bool containsLoopEscape(const std::vector<NodePtr> &body) {
  for (const NodePtr &stmt : body) {
    if (!stmt)
      continue;
    if (stmt->kind == "Break" || stmt->kind == "Continue")
      return true;
    if (stmt->kind == "For" || stmt->kind == "While" ||
        stmt->kind == "FunctionDef" || stmt->kind == "AsyncFunctionDef" ||
        stmt->kind == "ClassDef")
      continue;
    for (const parser::Field &field : stmt->fields)
      if (const auto *list = std::get_if<std::vector<NodePtr>>(&field.value))
        if (containsLoopEscape(*list))
          return true;
  }
  return false;
}

const char *kIsliceRangeMessage =
    "Indices for islice() must be None or an integer: 0 <= x <= sys.maxsize.";

} // namespace

std::optional<std::string>
ModuleEmitter::itertoolsCalleeName(const parser::Node *calleeNode) {
  if (!calleeNode)
    return std::nullopt;
  auto canonicalToName =
      [](const std::string &canonical) -> std::optional<std::string> {
    llvm::StringRef ref(canonical);
    if (ref.consume_front("itertools."))
      return ref.str();
    return std::nullopt;
  };
  if (calleeNode->kind == "Name") {
    llvm::StringRef name = ast::nameSpelling(*calleeNode);
    if (values.count(name))
      return std::nullopt;
    if (std::optional<std::string> canonical =
            types.lookupCanonicalBinding(name))
      return canonicalToName(*canonical);
    return std::nullopt;
  }
  if (calleeNode->kind == "Attribute") {
    auto attr = ast::string(*calleeNode, "attr");
    if (!attr)
      return std::nullopt;
    if (*attr == "from_iterable")
      if (const parser::Node *base = ast::node(*calleeNode, "value"))
        if (std::optional<std::string> baseName = itertoolsCalleeName(base);
            baseName && *baseName == "chain")
          return std::string("chain.from_iterable");
    std::string qualified = ast::qualifiedName(calleeNode);
    if (!qualified.empty())
      if (std::optional<std::string> canonical =
              types.lookupCanonicalBinding(qualified))
        return canonicalToName(*canonical);
    return std::nullopt;
  }
  return std::nullopt;
}

bool ModuleEmitter::tryEmitItertoolsFor(const parser::Node &statement,
                                        const parser::Node &iterCall) {
  const parser::Node *calleeNode = ast::node(iterCall, "func");
  std::optional<std::string> nameOpt = itertoolsCalleeName(calleeNode);
  if (!nameOpt)
    return false;
  llvm::StringRef name = *nameOpt;
  bool fusedName =
      name == "count" || name == "repeat" || name == "cycle" ||
      name == "islice" || name == "takewhile" || name == "dropwhile" ||
      name == "filterfalse" || name == "accumulate" || name == "pairwise" ||
      name == "zip_longest" || name == "chain.from_iterable";
  if (!fusedName)
    return false; // value synthesis handles chain/product/combinations/...

  auto reject = [&](llvm::StringRef reason) {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, statement.range.start, std::string(reason)});
    return true;
  };

  const auto *args = ast::nodeList(iterCall, "args");
  const auto *keywords = ast::nodeList(iterCall, "keywords");
  if (args)
    for (const NodePtr &arg : *args)
      if (arg && arg->kind == "Starred")
        return reject("itertools." + name.str() +
                      "() does not support * argument unpacking");
  auto argAt = [&](std::size_t index) -> NodePtr {
    if (!args || index >= args->size())
      return nullptr;
    return (*args)[index];
  };
  std::size_t argCount = args ? args->size() : 0;
  auto keywordValue = [&](llvm::StringRef keyword) -> NodePtr {
    if (!keywords)
      return nullptr;
    for (const NodePtr &node : *keywords) {
      if (!node)
        continue;
      auto kwName = ast::string(*node, "arg");
      if (!kwName || llvm::StringRef(*kwName) != keyword)
        continue;
      const parser::Field *valueField = parser::findField(*node, "value");
      if (valueField && std::holds_alternative<NodePtr>(valueField->value))
        return std::get<NodePtr>(valueField->value);
    }
    return nullptr;
  };
  auto keywordsOnly = [&](std::initializer_list<llvm::StringRef> allowed) {
    if (!keywords)
      return true;
    for (const NodePtr &node : *keywords) {
      if (!node)
        continue;
      auto kwName = ast::string(*node, "arg");
      if (!kwName)
        return false;
      if (!llvm::is_contained(allowed, llvm::StringRef(*kwName)))
        return false;
    }
    return true;
  };

  std::optional<ForParts> parts = forParts(statement);
  if (!parts)
    return false;
  parser::SourceRange range = parts->range;
  unsigned serial = ++listCompCounter;
  auto scratch = [&](const char *stem) {
    return "__lyitt" + std::to_string(serial) + "_" + std::string(stem);
  };
  // An expression re-read per iteration (or after other setup) must be a
  // name; anything else evaluates once into a scratch temp during setup.
  llvm::SmallVector<std::string, 6> scratchNames;
  std::vector<NodePtr> setup;
  auto pinned = [&](NodePtr expr, const char *stem) -> NodePtr {
    if (!expr || expr->kind == "Name" || expr->kind == "Constant")
      return expr;
    std::string temp = scratch(stem);
    scratchNames.push_back(temp);
    setup.push_back(synth::assign(synth::name(temp, range), expr, range));
    return synth::name(temp, range);
  };
  auto emitFused = [&](llvm::function_ref<void()> emitLoop) {
    runWithScratchNames(scratchNames, [&] {
      for (const NodePtr &node : setup)
        emitStatement(*node);
      emitLoop();
    });
    return true;
  };

  // ---- count([start[, step]]) --------------------------------------------
  if (name == "count") {
    if (argCount > 2 || !keywordsOnly({"start", "step"}))
      return reject("count() takes a start and a step");
    NodePtr start = argAt(0);
    if (!start)
      start = keywordValue("start");
    NodePtr step = argAt(1);
    if (!step)
      step = keywordValue("step");
    if (!start)
      start = synth::intConstant(0, range);
    if (!step)
      step = synth::intConstant(1, range);
    std::string counterName = scratch("c");
    scratchNames.push_back(counterName);
    NodePtr counter = synth::name(counterName, range);
    setup.push_back(synth::assign(counter, start, range));
    NodePtr stepRef = pinned(step, "st");
    std::vector<NodePtr> body{
        synth::assign(parts->target, counter, range),
        synth::assign(counter, synth::binOp(counter, "Add", stepRef, range), range)};
    body.insert(body.end(), parts->body.begin(), parts->body.end());
    NodePtr loop = synth::whileStmt(synth::constantBool(true, range), std::move(body),
                             parts->orelse, range);
    return emitFused([&] { emitWhile(*loop); });
  }

  // ---- repeat(object[, times]) --------------------------------------------
  if (name == "repeat") {
    NodePtr object = argAt(0);
    NodePtr times = argAt(1);
    if (!times)
      times = keywordValue("times");
    if (!object || argCount > 2 || !keywordsOnly({"times"}))
      return reject("repeat() takes an object and an optional times");
    std::string objectName = scratch("o");
    scratchNames.push_back(objectName);
    NodePtr objectRef = synth::name(objectName, range);
    setup.push_back(synth::assign(objectRef, object, range));
    if (!times) {
      std::vector<NodePtr> body{synth::assign(parts->target, objectRef, range)};
      body.insert(body.end(), parts->body.begin(), parts->body.end());
      NodePtr loop = synth::whileStmt(synth::constantBool(true, range), std::move(body),
                               parts->orelse, range);
      return emitFused([&] { emitWhile(*loop); });
    }
    NodePtr timesRef = pinned(times, "n");
    std::string counterName = scratch("k");
    scratchNames.push_back(counterName);
    NodePtr counter = synth::name(counterName, range);
    setup.push_back(synth::assign(counter, synth::intConstant(0, range), range));
    std::vector<NodePtr> body{
        synth::assign(counter, synth::binOp(counter, "Add", synth::intConstant(1, range),
                                      range),
                   range),
        synth::assign(parts->target, objectRef, range)};
    body.insert(body.end(), parts->body.begin(), parts->body.end());
    NodePtr loop = synth::whileStmt(synth::compare(counter, "Lt", timesRef, range),
                             std::move(body), parts->orelse, range);
    return emitFused([&] { emitWhile(*loop); });
  }

  // ---- cycle(sequence) -----------------------------------------------------
  if (name == "cycle") {
    if (argCount != 1 || (keywords && !keywords->empty()))
      return reject("cycle() takes exactly one iterable");
    if (!hasIndexableEvidence(argAt(0).get()))
      return reject("cycle() requires an indexable sequence "
                    "(list/str/tuple/bytes)");
    NodePtr source = pinned(argAt(0), "s");
    std::string indexName = scratch("i");
    scratchNames.push_back(indexName);
    NodePtr indexRef = synth::name(indexName, range);
    setup.push_back(synth::assign(indexRef, synth::intConstant(0, range), range));
    std::vector<NodePtr> body{
        synth::assign(parts->target, synth::subscript(source, indexRef, range),
                   range),
        synth::assign(indexRef,
                   synth::binOp(synth::binOp(indexRef, "Add", synth::intConstant(1, range),
                                       range),
                             "Mod", synth::lenCall(source, range), range),
                   range)};
    body.insert(body.end(), parts->body.begin(), parts->body.end());
    NodePtr loop = synth::whileStmt(
        synth::compare(synth::lenCall(source, range), "Gt", synth::intConstant(0, range),
                    range),
        std::move(body), parts->orelse, range);
    return emitFused([&] { emitWhile(*loop); });
  }

  // ---- islice(source, [start,] stop [, step]) ------------------------------
  if (name == "islice") {
    if (keywords && !keywords->empty())
      return reject("islice() takes no keyword arguments");
    if (argCount < 2 || argCount > 4)
      return reject("islice() takes a source and stop, or start/stop/step");
    NodePtr source = argAt(0);
    NodePtr start = argCount >= 3 ? argAt(1) : nullptr;
    NodePtr stop = argCount >= 3 ? argAt(2) : argAt(1);
    NodePtr step = argCount == 4 ? argAt(3) : nullptr;
    if (isNoneConstant(start.get()))
      start = nullptr;
    if (isNoneConstant(stop.get()))
      stop = nullptr;
    if (isNoneConstant(step.get()))
      step = nullptr;
    std::optional<std::int64_t> startConst = constantInt(start.get());
    std::optional<std::int64_t> stopConst = constantInt(stop.get());
    std::optional<std::int64_t> stepConst = constantInt(step.get());
    if ((start && !startConst) || (step && !stepConst))
      return reject("islice() start/step must be non-negative compile-time "
                    "constants for now; bind the slice to a variable-free "
                    "form or use a manual loop");
    if ((startConst && *startConst < 0) || (stopConst && *stopConst < 0) ||
        (stepConst && *stepConst < 1))
      return reject(kIsliceRangeMessage);
    // CPython evaluates the source expression before the bounds; the fused
    // setup runs bound temps first, so a bounds expression with effects
    // next to a source call would swap observable order.
    if (source && source->kind == "Call" && stop && !stopConst &&
        stop->kind != "Name")
      return reject("islice() over a source call requires the stop bound to "
                    "be a name or constant (evaluation order would swap); "
                    "bind the stop to a variable first");
    // A named iterator would be advanced one element past the slice by the
    // driving loop; a fresh call's residual position is unobservable.
    if (source && source->kind == "Name") {
      mlir::Type sourceType =
          types.widenLiteral(types.inferExpr(source.get()));
      if (types.inferMethodCallWithEvidence(sourceType, "__next__", {}))
        return reject("islice() over a named iterator would over-consume it; "
                      "call islice on the iterator-producing expression "
                      "directly");
    }
    if (parts->orelse.size() && stop)
      return reject("for/else over islice() with a stop is not supported "
                    "yet");
    if (stopConst && startConst.value_or(0) >= *stopConst) {
      // Statically empty: evaluate the source for its effects only.
      parser::NodePtr effect = synth::exprStmt(source, range);
      return emitFused([&] { emitStatement(*effect); });
    }
    std::int64_t startValue = startConst.value_or(0);
    std::int64_t stepValue = stepConst.value_or(1);
    NodePtr stopRef = stop ? pinned(stop, "e") : nullptr;
    if (stop && !stopConst)
      setup.push_back(synth::ifStmt(
          synth::compare(stopRef, "Lt", synth::intConstant(0, range), range),
          {synth::raiseValueError(kIsliceRangeMessage, range)}, {}, range));
    std::string pulledName = scratch("k");
    std::string indexName = scratch("j");
    scratchNames.push_back(pulledName);
    scratchNames.push_back(indexName);
    NodePtr pulled = synth::name(pulledName, range);
    NodePtr index = synth::name(indexName, range);
    setup.push_back(synth::assign(pulled, synth::intConstant(0, range), range));
    std::string elementName = scratch("v");
    scratchNames.push_back(elementName);
    NodePtr element = synth::name(elementName, range);
    std::vector<NodePtr> body;
    body.push_back(synth::assign(index, pulled, range));
    body.push_back(synth::assign(
        pulled, synth::binOp(index, "Add", synth::intConstant(1, range), range),
        range));
    if (stop)
      body.push_back(synth::ifStmt(synth::compare(index, "GtE", stopRef, range),
                            {synth::breakStmt(range)}, {}, range));
    if (startValue > 0)
      body.push_back(synth::ifStmt(
          synth::compare(index, "Lt", synth::intConstant(startValue, range), range),
          {synth::continueStmt(range)}, {}, range));
    if (stepValue != 1)
      body.push_back(synth::ifStmt(
          synth::compare(
              synth::binOp(synth::binOp(index, "Sub",
                                  synth::intConstant(startValue, range), range),
                        "Mod", synth::intConstant(stepValue, range), range),
              "NotEq", synth::intConstant(0, range), range),
          {synth::continueStmt(range)}, {}, range));
    body.push_back(synth::assign(parts->target, element, range));
    body.insert(body.end(), parts->body.begin(), parts->body.end());
    if (stop)
      body.push_back(synth::ifStmt(
          synth::compare(synth::binOp(index, "Add", synth::intConstant(stepValue, range),
                                range),
                      "GtE", stopRef, range),
          {synth::breakStmt(range)}, {}, range));
    NodePtr loop =
        synth::forStmt(element, source, std::move(body), parts->orelse, range);
    return emitFused([&] { emitFor(*loop); });
  }

  // ---- takewhile/dropwhile/filterfalse(predicate, source) ------------------
  if (name == "takewhile" || name == "dropwhile" || name == "filterfalse") {
    if (keywords && !keywords->empty())
      return reject("itertools." + name.str() +
                    "() takes no keyword arguments");
    if (argCount != 2)
      return reject("itertools." + name.str() +
                    "() requires a predicate and an iterable");
    NodePtr predicate = argAt(0);
    bool identityPredicate = name == "filterfalse" &&
                             isNoneConstant(predicate.get());
    LazyCallable callable;
    if (!identityPredicate &&
        !lazyCallableParts(statement, predicate, callable))
      return true;
    if (name == "takewhile" && !parts->orelse.empty())
      return reject("for/else over takewhile() is not supported yet");
    std::string elementName = scratch("v");
    scratchNames.push_back(elementName);
    for (const std::string &param : callable.lambdaParams)
      scratchNames.push_back(param);
    NodePtr element = synth::name(elementName, range);
    std::vector<NodePtr> body;
    if (name == "takewhile") {
      NodePtr test;
      if (!buildLazyCall(statement, callable, {element}, body, test))
        return true;
      body.push_back(synth::ifStmt(synth::notOp(test, range), {synth::breakStmt(range)}, {},
                            range));
      body.push_back(synth::assign(parts->target, element, range));
      body.insert(body.end(), parts->body.begin(), parts->body.end());
    } else if (name == "dropwhile") {
      std::string flagName = scratch("d");
      scratchNames.push_back(flagName);
      NodePtr flag = synth::name(flagName, range);
      setup.push_back(synth::assign(flag, synth::constantBool(true, range), range));
      std::vector<NodePtr> dropping;
      NodePtr test;
      if (!buildLazyCall(statement, callable, {element}, dropping, test))
        return true;
      dropping.push_back(synth::ifStmt(test, {synth::continueStmt(range)}, {}, range));
      dropping.push_back(synth::assign(flag, synth::constantBool(false, range), range));
      body.push_back(synth::ifStmt(flag, std::move(dropping), {}, range));
      body.push_back(synth::assign(parts->target, element, range));
      body.insert(body.end(), parts->body.begin(), parts->body.end());
    } else { // filterfalse
      NodePtr test = element;
      if (!identityPredicate &&
          !buildLazyCall(statement, callable, {element}, body, test))
        return true;
      std::vector<NodePtr> inner{synth::assign(parts->target, element, range)};
      inner.insert(inner.end(), parts->body.begin(), parts->body.end());
      body.push_back(synth::ifStmt(synth::notOp(test, range), std::move(inner), {}, range));
    }
    NodePtr loop =
        synth::forStmt(element, argAt(1), std::move(body), parts->orelse, range);
    return emitFused([&] { emitFor(*loop); });
  }

  // ---- accumulate(source[, func]) ------------------------------------------
  if (name == "accumulate") {
    if (!keywordsOnly({"func"}))
      return reject("accumulate() with initial= is not supported yet");
    if (argCount < 1 || argCount > 2)
      return reject("accumulate() takes an iterable and an optional "
                    "function");
    NodePtr func = argAt(1);
    if (!func)
      func = keywordValue("func");
    bool defaultAdd = !func || isNoneConstant(func.get());
    LazyCallable callable;
    if (!defaultAdd && !lazyCallableParts(statement, func, callable))
      return true;
    std::string elementName = scratch("v");
    std::string accName = scratch("a");
    std::string flagName = scratch("h");
    scratchNames.push_back(elementName);
    scratchNames.push_back(accName);
    scratchNames.push_back(flagName);
    for (const std::string &param : callable.lambdaParams)
      scratchNames.push_back(param);
    NodePtr element = synth::name(elementName, range);
    NodePtr acc = synth::name(accName, range);
    NodePtr flag = synth::name(flagName, range);
    setup.push_back(synth::assign(flag, synth::constantBool(false, range), range));
    // The accumulator is loop-carried, so it needs a pre-loop definition (a
    // branch-local first assignment is invisible to the sibling branch).
    // The int seed restricts fused accumulate to int elements; other element
    // types fail the merge with a diagnostic instead of mis-executing.
    setup.push_back(synth::assign(acc, synth::intConstant(0, range), range));
    std::vector<NodePtr> accumulateStep;
    NodePtr applied;
    if (defaultAdd)
      applied = synth::binOp(acc, "Add", element, range);
    else if (!buildLazyCall(statement, callable, {acc, element},
                            accumulateStep, applied))
      return true;
    accumulateStep.push_back(synth::assign(acc, applied, range));
    std::vector<NodePtr> firstStep{
        synth::assign(acc, element, range),
        synth::assign(flag, synth::constantBool(true, range), range)};
    std::vector<NodePtr> body{
        synth::ifStmt(flag, std::move(accumulateStep), std::move(firstStep),
                   range),
        synth::assign(parts->target, acc, range)};
    body.insert(body.end(), parts->body.begin(), parts->body.end());
    NodePtr loop =
        synth::forStmt(element, argAt(0), std::move(body), parts->orelse, range);
    return emitFused([&] { emitFor(*loop); });
  }

  // ---- pairwise(source) ----------------------------------------------------
  if (name == "pairwise") {
    if (argCount != 1 || (keywords && !keywords->empty()))
      return reject("pairwise() takes exactly one iterable");
    NodePtr source = argAt(0);
    if (hasIndexableEvidence(source.get())) {
      NodePtr sourceRef = pinned(source, "s");
      std::string indexName = scratch("i");
      std::string cursorName = scratch("j");
      scratchNames.push_back(indexName);
      scratchNames.push_back(cursorName);
      NodePtr index = synth::name(indexName, range);
      NodePtr cursor = synth::name(cursorName, range);
      setup.push_back(synth::assign(index, synth::intConstant(0, range), range));
      std::vector<NodePtr> body{
          synth::assign(cursor, index, range),
          synth::assign(index, synth::binOp(cursor, "Add", synth::intConstant(1, range),
                                      range),
                     range)};
      appendTargetBinding(
          parts->target,
          {synth::subscript(sourceRef, cursor, range),
           synth::subscript(sourceRef,
                         synth::binOp(cursor, "Add", synth::intConstant(1, range),
                                   range),
                         range)},
          {}, std::string(), body, range);
      body.insert(body.end(), parts->body.begin(), parts->body.end());
      NodePtr loop = synth::whileStmt(
          synth::compare(synth::binOp(index, "Add", synth::intConstant(1, range), range),
                      "Lt", synth::lenCall(sourceRef, range), range),
          std::move(body), parts->orelse, range);
      return emitFused([&] { emitWhile(*loop); });
    }
    // General (generator-backed) source: components bind separately, so the
    // target must be a two-name tuple pattern — materializing a tuple from
    // the carried previous element trips generator resume-token accounting.
    const auto *elts = parts->target && parts->target->kind == "Tuple"
                           ? ast::nodeList(*parts->target, "elts")
                           : nullptr;
    if (!elts || elts->size() != 2)
      return reject("pairwise() over a non-indexable source requires a "
                    "two-name tuple target (for a, b in pairwise(...))");
    std::string elementName = scratch("v");
    std::string prevName = scratch("p");
    std::string flagName = scratch("f");
    scratchNames.push_back(elementName);
    scratchNames.push_back(prevName);
    scratchNames.push_back(flagName);
    NodePtr element = synth::name(elementName, range);
    NodePtr prev = synth::name(prevName, range);
    NodePtr flag = synth::name(flagName, range);
    setup.push_back(synth::assign(flag, synth::constantBool(true, range), range));
    // Loop-carried previous element; same int-seed restriction as the
    // fused accumulate (see the comment there).
    setup.push_back(synth::assign(prev, synth::intConstant(0, range), range));
    std::vector<NodePtr> firstStep{
        synth::assign(prev, element, range),
        synth::assign(flag, synth::constantBool(false, range), range),
        synth::continueStmt(range)};
    std::vector<NodePtr> body{synth::ifStmt(flag, std::move(firstStep), {}, range),
                              synth::assign((*elts)[0], prev, range),
                              synth::assign((*elts)[1], element, range),
                              synth::assign(prev, element, range)};
    body.insert(body.end(), parts->body.begin(), parts->body.end());
    NodePtr loop =
        synth::forStmt(element, source, std::move(body), parts->orelse, range);
    return emitFused([&] { emitFor(*loop); });
  }

  // ---- zip_longest(a, b, ..., fillvalue=None) ------------------------------
  if (name == "zip_longest") {
    if (!keywordsOnly({"fillvalue"}))
      return reject("zip_longest() accepts only the fillvalue keyword");
    if (argCount < 2)
      return reject("zip_longest() requires at least two sequences");
    for (std::size_t i = 0; i < argCount; ++i)
      if (!hasIndexableEvidence(argAt(i).get()))
        return reject("zip_longest() requires indexable sequences "
                      "(list/str/tuple/bytes)");
    llvm::SmallVector<NodePtr, 4> sources;
    for (std::size_t i = 0; i < argCount; ++i)
      sources.push_back(
          pinned(argAt(i), ("s" + std::to_string(i)).c_str()));
    NodePtr fill = keywordValue("fillvalue");
    if (!fill)
      fill = synth::noneConstant(range);
    NodePtr fillRef = pinned(fill, "fv");
    std::string indexName = scratch("i");
    std::string cursorName = scratch("j");
    scratchNames.push_back(indexName);
    scratchNames.push_back(cursorName);
    NodePtr index = synth::name(indexName, range);
    NodePtr cursor = synth::name(cursorName, range);
    setup.push_back(synth::assign(index, synth::intConstant(0, range), range));
    std::vector<NodePtr> condition;
    for (const NodePtr &source : sources)
      condition.push_back(
          synth::compare(index, "Lt", synth::lenCall(source, range), range));
    std::vector<NodePtr> body{
        synth::assign(cursor, index, range),
        synth::assign(index, synth::binOp(cursor, "Add", synth::intConstant(1, range),
                                    range),
                   range)};
    std::vector<NodePtr> components;
    for (const NodePtr &source : sources)
      components.push_back(synth::ifExp(
          synth::compare(cursor, "Lt", synth::lenCall(source, range), range),
          synth::subscript(source, cursor, range), fillRef, range));
    appendTargetBinding(parts->target, std::move(components), {},
                        std::string(), body, range);
    body.insert(body.end(), parts->body.begin(), parts->body.end());
    NodePtr loop = synth::whileStmt(synth::orChain(std::move(condition), range),
                             std::move(body), parts->orelse, range);
    return emitFused([&] { emitWhile(*loop); });
  }

  // ---- chain.from_iterable(rows) -------------------------------------------
  if (name == "chain.from_iterable") {
    if (argCount != 1 || (keywords && !keywords->empty()))
      return reject("chain.from_iterable() takes exactly one iterable");
    if (containsLoopEscape(parts->body))
      return reject("break/continue inside a fused chain.from_iterable() "
                    "loop is not supported yet (it would bind to the inner "
                    "loop); restructure with a flag");
    NodePtr rows = pinned(argAt(0), "r");
    std::string rowName = scratch("row");
    scratchNames.push_back(rowName);
    NodePtr rowRef = synth::name(rowName, range);
    NodePtr inner =
        synth::forStmt(parts->target, rowRef, parts->body, {}, range);
    NodePtr outer = synth::forStmt(rowRef, rows, {std::move(inner)}, parts->orelse,
                            range);
    return emitFused([&] { emitFor(*outer); });
  }

  return false;
}

std::optional<Value>
ModuleEmitter::tryEmitItertoolsValueCall(const parser::Node &expr,
                                         const parser::Node *calleeNode) {
  std::optional<std::string> nameOpt = itertoolsCalleeName(calleeNode);
  if (!nameOpt)
    return std::nullopt;
  llvm::StringRef name = *nameOpt;

  auto rejectValue = [&](llvm::StringRef reason) -> std::optional<Value> {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, expr.range.start, std::string(reason)});
    return emitNone(expr);
  };

  if (name == "starmap" || name == "groupby" || name == "tee" ||
      name == "batched" || name == "permutations")
    return rejectValue("itertools." + name.str() + "() is not supported yet");
  if (name == "takewhile" || name == "filterfalse" || name == "accumulate" ||
      name == "zip_longest" || name == "chain.from_iterable")
    return rejectValue("itertools." + name.str() +
                       "() as a first-class value is not supported yet; "
                       "consume it directly in a for loop");

  const auto *args = ast::nodeList(expr, "args");
  const auto *keywords = ast::nodeList(expr, "keywords");
  if (args)
    for (const NodePtr &arg : *args)
      if (arg && arg->kind == "Starred")
        return rejectValue("itertools." + name.str() +
                           "() does not support * argument unpacking");
  std::size_t argCount = args ? args->size() : 0;
  auto argAt = [&](std::size_t index) -> const parser::Node * {
    if (!args || index >= args->size())
      return nullptr;
    return (*args)[index].get();
  };
  auto keywordValue = [&](llvm::StringRef keyword) -> const parser::Node * {
    if (!keywords)
      return nullptr;
    for (const NodePtr &node : *keywords) {
      if (!node)
        continue;
      auto kwName = ast::string(*node, "arg");
      if (!kwName || llvm::StringRef(*kwName) != keyword)
        continue;
      return ast::node(*node, "value");
    }
    return nullptr;
  };
  auto keywordsOnly = [&](std::initializer_list<llvm::StringRef> allowed) {
    if (!keywords)
      return true;
    for (const NodePtr &node : *keywords) {
      if (!node)
        continue;
      auto kwName = ast::string(*node, "arg");
      if (!kwName || !llvm::is_contained(allowed, llvm::StringRef(*kwName)))
        return false;
    }
    return true;
  };
  parser::SourceRange range = expr.range;

  auto intDefault = [&](std::int64_t value) -> Value {
    std::string text = std::to_string(value);
    mlir::Type type = types.literal(text);
    auto op = py::IntConstantOp::create(builder, loc(expr), type,
                                        builder.getStringAttr(text));
    return Value{op.getResult(), type};
  };
  auto indexableOrNull = [&](const Value &value) {
    mlir::Type widened = types.widenLiteral(value.type);
    return types.inferMethodCallWithEvidence(widened, "__len__", {}) &&
           types.inferMethodCallWithEvidence(widened, "__getitem__",
                                             {types.intType()});
  };

  // Collected synthesis inputs.
  llvm::SmallVector<std::string, 4> params;
  llvm::SmallVector<Value, 4> argValues;
  llvm::SmallVector<mlir::Type, 4> paramTypes;
  std::vector<NodePtr> body;
  std::string memoKey = ("itertools." + name).str();
  auto addParam = [&](const char *stem, Value value) -> NodePtr {
    std::string param =
        "__ly" + std::string(stem) + std::to_string(params.size());
    params.push_back(param);
    argValues.push_back(value);
    paramTypes.push_back(types.widenLiteral(value.type));
    memoKey += "|" + typeText(paramTypes.back());
    return synth::name(param, range);
  };
  auto increment = [&](NodePtr counter, std::int64_t by) {
    return synth::assign(counter,
                      synth::binOp(counter, "Add", synth::intConstant(by, range),
                                range),
                      range);
  };

  // ---- count([start[, step]]) ---------------------------------------------
  if (name == "count") {
    if (argCount > 2 || !keywordsOnly({"start", "step"}))
      return rejectValue("count() takes a start and a step");
    const parser::Node *startNode = argAt(0);
    if (!startNode)
      startNode = keywordValue("start");
    const parser::Node *stepNode = argAt(1);
    if (!stepNode)
      stepNode = keywordValue("step");
    Value start = startNode ? emitExpr(startNode) : intDefault(0);
    Value step = stepNode ? emitExpr(stepNode) : intDefault(1);
    NodePtr cursor = addParam("c", start);
    NodePtr stride = addParam("st", step);
    std::vector<NodePtr> loop{
        synth::yieldStmt(cursor, range),
        synth::assign(cursor, synth::binOp(cursor, "Add", stride, range), range)};
    body.push_back(synth::whileStmt(synth::constantBool(true, range), std::move(loop), {},
                             range));
  }

  // ---- repeat(object[, times]) ---------------------------------------------
  else if (name == "repeat") {
    const parser::Node *objectNode = argAt(0);
    const parser::Node *timesNode = argAt(1);
    if (!timesNode)
      timesNode = keywordValue("times");
    if (!objectNode || argCount > 2 || !keywordsOnly({"times"}))
      return rejectValue("repeat() takes an object and an optional times");
    Value object = emitExpr(objectNode);
    NodePtr objectRef = addParam("o", object);
    if (!timesNode) {
      memoKey += "|inf";
      body.push_back(synth::whileStmt(synth::constantBool(true, range),
                               {synth::yieldStmt(objectRef, range)}, {}, range));
    } else {
      memoKey += "|times";
      NodePtr bound = addParam("n", emitExpr(timesNode));
      NodePtr counter = synth::name("__lyi", range);
      body.push_back(synth::assign(counter, synth::intConstant(0, range), range));
      std::vector<NodePtr> loop{synth::yieldStmt(objectRef, range),
                                increment(counter, 1)};
      body.push_back(synth::whileStmt(synth::compare(counter, "Lt", bound, range),
                               std::move(loop), {}, range));
    }
  }

  // ---- indexable-sequence combinators ---------------------------------------
  else {
    // Everything below takes leading iterable argument(s) that must be
    // indexable sequences in value position (iterator/generator sources are
    // covered by the for-position fusions).
    auto sequenceParam = [&](const parser::Node *node,
                             const char *what) -> std::optional<NodePtr> {
      if (!node)
        return std::nullopt;
      Value value = emitExpr(node);
      if (!indexableOrNull(value)) {
        rejectValue(std::string(what) +
                    " as a value requires indexable sequences "
                    "(list/str/tuple/bytes); iterate non-indexable sources "
                    "directly in a for loop, or convert with a comprehension "
                    "first");
        return std::nullopt;
      }
      return addParam("it", value);
    };
    NodePtr index = synth::name("__lyi", range);

    if (name == "cycle") {
      if (argCount != 1 || (keywords && !keywords->empty()))
        return rejectValue("cycle() takes exactly one iterable");
      std::optional<NodePtr> source = sequenceParam(argAt(0), "cycle()");
      if (!source)
        return emitNone(expr);
      body.push_back(synth::assign(index, synth::intConstant(0, range), range));
      std::vector<NodePtr> loop{
          synth::yieldStmt(synth::subscript(*source, index, range), range),
          synth::assign(index,
                     synth::binOp(synth::binOp(index, "Add", synth::intConstant(1, range),
                                         range),
                               "Mod", synth::lenCall(*source, range), range),
                     range)};
      body.push_back(synth::whileStmt(
          synth::compare(synth::lenCall(*source, range), "Gt", synth::intConstant(0, range),
                      range),
          std::move(loop), {}, range));
    } else if (name == "chain") {
      if (argCount < 1 || (keywords && !keywords->empty()))
        return rejectValue("chain() requires at least one sequence");
      llvm::SmallVector<NodePtr, 4> sources;
      for (std::size_t i = 0; i < argCount; ++i) {
        std::optional<NodePtr> source = sequenceParam(argAt(i), "chain()");
        if (!source)
          return emitNone(expr);
        sources.push_back(*source);
      }
      for (const NodePtr &source : sources) {
        body.push_back(synth::assign(index, synth::intConstant(0, range), range));
        std::vector<NodePtr> loop{
            synth::yieldStmt(synth::subscript(source, index, range), range),
            increment(index, 1)};
        body.push_back(
            synth::whileStmt(synth::compare(index, "Lt", synth::lenCall(source, range), range),
                      std::move(loop), {}, range));
      }
    } else if (name == "islice") {
      if (keywords && !keywords->empty())
        return rejectValue("islice() takes no keyword arguments");
      if (argCount < 2 || argCount > 4)
        return rejectValue("islice() takes a source and stop, or "
                           "start/stop/step");
      std::optional<NodePtr> source = sequenceParam(argAt(0), "islice()");
      if (!source)
        return emitNone(expr);
      const parser::Node *startNode = argCount >= 3 ? argAt(1) : nullptr;
      const parser::Node *stopNode = argCount >= 3 ? argAt(2) : argAt(1);
      const parser::Node *stepNode = argCount == 4 ? argAt(3) : nullptr;
      if (isNoneConstant(startNode))
        startNode = nullptr;
      if (isNoneConstant(stopNode))
        stopNode = nullptr;
      if (isNoneConstant(stepNode))
        stepNode = nullptr;
      std::optional<std::int64_t> startConst = constantInt(startNode);
      std::optional<std::int64_t> stopConst = constantInt(stopNode);
      std::optional<std::int64_t> stepConst = constantInt(stepNode);
      if ((startNode && !startConst) || (stepNode && !stepConst))
        return rejectValue("islice() start/step must be compile-time "
                           "constants for now");
      if ((startConst && *startConst < 0) || (stopConst && *stopConst < 0) ||
          (stepConst && *stepConst < 1))
        return rejectValue(kIsliceRangeMessage);
      std::int64_t startValue = startConst.value_or(0);
      std::int64_t stepValue = stepConst.value_or(1);
      memoKey += "|s" + std::to_string(startValue) + "x" +
                 std::to_string(stepValue);
      NodePtr stopRef;
      if (stopNode && !stopConst) {
        stopRef = addParam("e", emitExpr(stopNode));
        memoKey += "|estop";
        body.push_back(synth::ifStmt(
            synth::compare(stopRef, "Lt", synth::intConstant(0, range), range),
            {synth::raiseValueError(kIsliceRangeMessage, range)}, {}, range));
      } else if (stopConst) {
        stopRef = synth::intConstant(*stopConst, range);
        memoKey += "|e" + std::to_string(*stopConst);
      } else {
        memoKey += "|enone";
      }
      body.push_back(synth::assign(index, synth::intConstant(startValue, range),
                                range));
      std::vector<NodePtr> loop{
          synth::ifStmt(synth::compare(index, "GtE", synth::lenCall(*source, range), range),
                 {synth::breakStmt(range)}, {}, range),
          synth::yieldStmt(synth::subscript(*source, index, range), range),
          increment(index, stepValue)};
      NodePtr condition = stopRef
                              ? synth::compare(index, "Lt", stopRef, range)
                              : synth::constantBool(true, range);
      body.push_back(synth::whileStmt(std::move(condition), std::move(loop), {},
                               range));
    } else if (name == "dropwhile") {
      if (argCount != 2 || (keywords && !keywords->empty()))
        return rejectValue("dropwhile() requires a predicate and an "
                           "iterable");
      LazyCallable callable;
      if (!lazyCallableParts(expr, (*args)[0], callable))
        return emitNone(expr);
      std::optional<NodePtr> source = sequenceParam(argAt(1), "dropwhile()");
      if (!source)
        return emitNone(expr);
      if (callable.callee)
        memoKey += "|f:" + ast::qualifiedName(callable.callee.get());
      else {
        llvm::raw_string_ostream stream(memoKey);
        stream << "|lambda:" << callable.lambdaBody.get();
      }
      body.push_back(synth::assign(index, synth::intConstant(0, range), range));
      std::vector<NodePtr> scanLoop;
      NodePtr test;
      if (!buildLazyCall(expr, callable,
                         {synth::subscript(*source, index, range)}, scanLoop,
                         test))
        return emitNone(expr);
      scanLoop.push_back(synth::ifStmt(synth::notOp(test, range),
                                {synth::breakStmt(range)}, {}, range));
      scanLoop.push_back(increment(index, 1));
      body.push_back(
          synth::whileStmt(synth::compare(index, "Lt", synth::lenCall(*source, range), range),
                    std::move(scanLoop), {}, range));
      std::vector<NodePtr> yieldLoop{
          synth::yieldStmt(synth::subscript(*source, index, range), range),
          increment(index, 1)};
      body.push_back(
          synth::whileStmt(synth::compare(index, "Lt", synth::lenCall(*source, range), range),
                    std::move(yieldLoop), {}, range));
    } else if (name == "pairwise") {
      if (argCount != 1 || (keywords && !keywords->empty()))
        return rejectValue("pairwise() takes exactly one iterable");
      std::optional<NodePtr> source = sequenceParam(argAt(0), "pairwise()");
      if (!source)
        return emitNone(expr);
      body.push_back(synth::assign(index, synth::intConstant(0, range), range));
      std::vector<NodePtr> loop{
          synth::yieldStmt(
              synth::tuple({synth::subscript(*source, index, range),
                         synth::subscript(*source,
                                       synth::binOp(index, "Add",
                                                 synth::intConstant(1, range),
                                                 range),
                                       range)},
                        range),
              range),
          increment(index, 1)};
      body.push_back(synth::whileStmt(
          synth::compare(synth::binOp(index, "Add", synth::intConstant(1, range), range),
                      "Lt", synth::lenCall(*source, range), range),
          std::move(loop), {}, range));
    } else if (name == "product") {
      if (keywords && !keywords->empty())
        return rejectValue("product() with repeat= is not supported yet");
      if (argCount < 2 || argCount > 4)
        return rejectValue("product() supports two to four sequences");
      llvm::SmallVector<NodePtr, 4> sources;
      for (std::size_t i = 0; i < argCount; ++i) {
        std::optional<NodePtr> source = sequenceParam(argAt(i), "product()");
        if (!source)
          return emitNone(expr);
        sources.push_back(*source);
      }
      std::vector<NodePtr> elements;
      llvm::SmallVector<NodePtr, 4> indices;
      for (std::size_t i = 0; i < sources.size(); ++i) {
        indices.push_back(synth::name("__lyi" + std::to_string(i), range));
        elements.push_back(synth::subscript(sources[i], indices[i], range));
      }
      // Build inside-out: the innermost body yields, each wrapper resets its
      // index, runs the nested while, then advances the enclosing index.
      std::vector<NodePtr> inner{
          synth::yieldStmt(synth::tuple(std::move(elements), range), range),
          increment(indices.back(), 1)};
      for (std::size_t i = sources.size(); i-- > 1;) {
        std::vector<NodePtr> wrapped{
            synth::assign(indices[i], synth::intConstant(0, range), range)};
        wrapped.push_back(synth::whileStmt(
            synth::compare(indices[i], "Lt", synth::lenCall(sources[i], range), range),
            std::move(inner), {}, range));
        wrapped.push_back(increment(indices[i - 1], 1));
        inner = std::move(wrapped);
      }
      body.push_back(synth::assign(indices[0], synth::intConstant(0, range), range));
      body.push_back(synth::whileStmt(
          synth::compare(indices[0], "Lt", synth::lenCall(sources[0], range), range),
          std::move(inner), {}, range));
    } else if (name == "combinations" ||
               name == "combinations_with_replacement") {
      if (argCount != 2 || (keywords && !keywords->empty()))
        return rejectValue("itertools." + name.str() +
                           "() requires a sequence and r");
      std::optional<std::int64_t> r = constantInt(argAt(1));
      if (!r || *r < 1 || *r > 4)
        return rejectValue("itertools." + name.str() +
                           "() requires a compile-time constant r between "
                           "1 and 4 for now");
      std::optional<NodePtr> source =
          sequenceParam(argAt(0), name == "combinations"
                                      ? "combinations()"
                                      : "combinations_with_replacement()");
      if (!source)
        return emitNone(expr);
      bool withReplacement = name == "combinations_with_replacement";
      memoKey += "|r" + std::to_string(*r);
      std::size_t depth = static_cast<std::size_t>(*r);
      llvm::SmallVector<NodePtr, 4> indices;
      std::vector<NodePtr> elements;
      for (std::size_t i = 0; i < depth; ++i) {
        indices.push_back(synth::name("__lyi" + std::to_string(i), range));
        elements.push_back(synth::subscript(*source, indices[i], range));
      }
      std::vector<NodePtr> inner{
          synth::yieldStmt(synth::tuple(std::move(elements), range), range),
          increment(indices.back(), 1)};
      for (std::size_t i = depth; i-- > 1;) {
        NodePtr firstValue =
            withReplacement
                ? synth::name("__lyi" + std::to_string(i - 1), range)
                : synth::binOp(synth::name("__lyi" + std::to_string(i - 1), range),
                            "Add", synth::intConstant(1, range), range);
        std::vector<NodePtr> wrapped{synth::assign(indices[i],
                                                std::move(firstValue),
                                                range)};
        wrapped.push_back(synth::whileStmt(
            synth::compare(indices[i], "Lt", synth::lenCall(*source, range), range),
            std::move(inner), {}, range));
        wrapped.push_back(increment(indices[i - 1], 1));
        inner = std::move(wrapped);
      }
      body.push_back(synth::assign(indices[0], synth::intConstant(0, range), range));
      body.push_back(synth::whileStmt(
          synth::compare(indices[0], "Lt", synth::lenCall(*source, range), range),
          std::move(inner), {}, range));
    } else {
      return rejectValue("itertools." + name.str() +
                         "() is not supported yet");
    }
  }

  // Memoized synthesis + call, mirroring the builtin lazy-value machinery.
  auto memoized = lazyIteratorMemo.find(memoKey);
  if (memoized == lazyIteratorMemo.end()) {
    unsigned serial = ++syntheticFunctionCounter;
    std::string stem = name.str();
    for (char &c : stem)
      if (c == '.')
        c = '_';
    std::string symbol =
        ("__lyiter$itertools_" + stem + "$" + llvm::Twine(serial)).str();
    llvm::SmallVector<const parser::Node *, 4> paramNodes;
    llvm::SmallVector<synth::Param, 4> synthParams;
    for (llvm::StringRef param : params)
      synthParams.push_back(synth::Param{param.str(), nullptr});
    NodePtr def = synth::functionDef(symbol, synthParams, {}, std::move(body),
                                     nullptr, {}, range, &paramNodes);
    synthesizedIteratorDefs.push_back(def);
    for (auto [i, type] : llvm::enumerate(paramTypes))
      types.overrideParameterType(paramNodes[i], type);
    FunctionSignature sig = types.functionSignature(*def);
    emitCallableFunction(*def, symbol, sig, {}, /*isLambda=*/false);
    memoized =
        lazyIteratorMemo
            .insert({memoKey,
                     LazyIteratorSynthesis{symbol, sig.publicCallable}})
            .first;
  }

  llvm::SmallVector<Value, 4> callArguments(argValues.begin(),
                                            argValues.end());
  Value callee = emitBindingRef(expr, memoized->second.symbol,
                                memoized->second.callableType);
  return emitCallableDispatch(
      expr, callee,
      emitCallOperands(expr, callArguments, /*includeAstArguments=*/false));
}


} // namespace lython::emitter
