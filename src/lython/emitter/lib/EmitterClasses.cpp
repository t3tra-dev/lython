#include "AstSynth.h"
#include "EmitterCore.h"
#include "EmptyLiteralSeed.h"
#include "EmitterPyOps.h"
#include "EmitterSupport.h"

#include "AstAccess.h"
#include "ClosureAnalysis.h"
#include "Contracts.h"
#include "C3Linearization.h"
#include "ExceptionTaxonomy.h"
#include "PyCallableShape.h"
#include "PyProtocols.h"
#include "TypeSystemSolver.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/SaveAndRestore.h"

#include <iterator>
#include <optional>
#include <utility>

namespace lython::emitter {
namespace {

// A method's contract, as the protocol table stores it. One spelling for both
// registrations of a class: the progressive one that makes a method visible
// to the next method's body walk, and the complete one at the end.
void addProtocolMethod(py::protocols::ProtocolInfo &info,
                       llvm::StringRef methodName, mlir::Type methodContract) {
  auto pushSignature = [&](py::CallableType signature) {
    if (!signature)
      return;
    py::protocols::ProtocolMethod method;
    method.signature = signature;
    method.mayThrow = true;
    info.methods[methodName.str()].push_back(method);
  };
  if (auto signature =
          mlir::dyn_cast_if_present<py::CallableType>(methodContract)) {
    pushSignature(signature);
  } else if (auto overload =
                 mlir::dyn_cast_if_present<py::OverloadType>(methodContract)) {
    for (mlir::Type candidate : overload.getCandidateTypes())
      pushSignature(mlir::dyn_cast_if_present<py::CallableType>(candidate));
  }
}

// Positional parameters with no annotation: the synthesized dataclass and
// record constructors take their types from the field declarations, not from
// the AST.
llvm::SmallVector<synth::Param, 8> toSynthParams(
    llvm::ArrayRef<std::string> names) {
  llvm::SmallVector<synth::Param, 8> params;
  for (const std::string &name : names)
    params.push_back(synth::Param{name, nullptr});
  return params;
}


mlir::Attribute sourceExprAttr(mlir::Builder &builder,
                               const parser::Node *node) {
  auto dict = [&](llvm::StringRef kind,
                  llvm::ArrayRef<mlir::NamedAttribute> extra = {}) {
    llvm::SmallVector<mlir::NamedAttribute, 4> attrs;
    attrs.push_back(builder.getNamedAttr("kind", builder.getStringAttr(kind)));
    attrs.append(extra.begin(), extra.end());
    return builder.getDictionaryAttr(attrs);
  };

  if (!node)
    return dict("none");
  if (node->kind == "Constant") {
    if (ast::isNoneField(*node, "value"))
      return dict("constant.none");
    if (auto value = ast::boolean(*node, "value"))
      return dict("constant.bool",
                  {builder.getNamedAttr("value", builder.getBoolAttr(*value))});
    if (auto value = ast::integer(*node, "value"))
      return dict("constant.int",
                  {builder.getNamedAttr(
                      "value", builder.getStringAttr(std::to_string(*value)))});
    if (auto value = ast::floating(*node, "value"))
      return dict(
          "constant.float",
          {builder.getNamedAttr("value", builder.getF64FloatAttr(*value))});
    if (auto value = ast::string(*node, "value"))
      return dict("constant.str", {builder.getNamedAttr(
                                      "value", builder.getStringAttr(*value))});
    if (const auto *fieldValue = ast::field(*node, "value"))
      if (const auto *big = std::get_if<parser::BigInteger>(fieldValue))
        return dict("constant.int",
                    {builder.getNamedAttr(
                        "value", builder.getStringAttr(big->decimal))});
    return dict("unsupported", {builder.getNamedAttr(
                                   "node", builder.getStringAttr("Constant"))});
  }
  if (node->kind == "Name" || node->kind == "Attribute") {
    std::string qualified = ast::qualifiedName(node);
    if (qualified.empty())
      qualified = std::string(ast::nameSpelling(*node));
    return dict("ref", {builder.getNamedAttr(
                           "name", builder.getStringAttr(qualified))});
  }
  if (node->kind == "List" || node->kind == "Tuple") {
    llvm::SmallVector<mlir::Attribute, 8> values;
    if (const auto *elts = ast::nodeList(*node, "elts"))
      for (const parser::NodePtr &element : *elts)
        values.push_back(sourceExprAttr(builder, element.get()));
    return dict(node->kind == "List" ? "list" : "tuple",
                {builder.getNamedAttr("elts", builder.getArrayAttr(values))});
  }
  if (node->kind == "Call") {
    llvm::SmallVector<mlir::Attribute, 8> args;
    if (const auto *argNodes = ast::nodeList(*node, "args"))
      for (const parser::NodePtr &arg : *argNodes)
        args.push_back(sourceExprAttr(builder, arg.get()));
    llvm::SmallVector<mlir::NamedAttribute, 3> attrs;
    attrs.push_back(builder.getNamedAttr(
        "callee", sourceExprAttr(builder, ast::node(*node, "func"))));
    attrs.push_back(builder.getNamedAttr("args", builder.getArrayAttr(args)));
    return dict("call", attrs);
  }
  if (node->kind == "BinOp") {
    llvm::SmallVector<mlir::NamedAttribute, 4> attrs;
    attrs.push_back(builder.getNamedAttr(
        "op", builder.getStringAttr(ast::node(*node, "op")
                                        ? ast::node(*node, "op")->kind
                                        : std::string())));
    attrs.push_back(builder.getNamedAttr(
        "left", sourceExprAttr(builder, ast::node(*node, "left"))));
    attrs.push_back(builder.getNamedAttr(
        "right", sourceExprAttr(builder, ast::node(*node, "right"))));
    return dict("binop", attrs);
  }

  return dict("unsupported", {builder.getNamedAttr(
                                 "node", builder.getStringAttr(node->kind))});
}

// ⭐ A LITERAL TYPE IS A FOLDED VALUE, and the constant channel's currency is
// exactly that. An IMPORTED class attribute is recorded from its AST shape, so
// `capacity = model.LIMIT` becomes `kind = "ref"` and nothing materializes a
// ref -- the attribute could not be read at all. Where the inference has
// already folded the expression to a literal, the fold is what gets recorded.
//
// ⛔ A str only when the spelling round-trips: the literal's text is the source
// spelling with quotes, and one containing a quote or a backslash of its own
// cannot be unwrapped by looking at the ends.
mlir::Attribute literalConstantAttr(mlir::Builder &builder, mlir::Type type) {
  auto literal = mlir::dyn_cast_if_present<py::LiteralType>(type);
  if (!literal)
    return {};
  llvm::StringRef spelling = literal.getSpelling();
  auto dict = [&](llvm::StringRef kind,
                  llvm::ArrayRef<mlir::NamedAttribute> extra = {}) {
    llvm::SmallVector<mlir::NamedAttribute, 4> attrs;
    attrs.push_back(builder.getNamedAttr("kind", builder.getStringAttr(kind)));
    attrs.append(extra.begin(), extra.end());
    return mlir::Attribute(builder.getDictionaryAttr(attrs));
  };
  if (spelling == "None")
    return dict("constant.none");
  if (spelling == "True" || spelling == "False")
    return dict("constant.bool",
                {builder.getNamedAttr(
                    "value", builder.getBoolAttr(spelling == "True"))});
  if (spelling.size() >= 2 && spelling.front() == '"' &&
      spelling.back() == '"') {
    llvm::StringRef text = spelling.drop_front().drop_back();
    if (text.contains('"') || text.contains('\\'))
      return {};
    return dict("constant.str",
                {builder.getNamedAttr("value", builder.getStringAttr(text))});
  }
  llvm::StringRef digits = spelling;
  digits.consume_front("-");
  if (!digits.empty() &&
      llvm::all_of(digits, [](char c) { return c >= '0' && c <= '9'; }))
    return dict("constant.int", {builder.getNamedAttr(
                                    "value", builder.getStringAttr(spelling))});
  double floating = 0.0;
  if (!spelling.getAsDouble(floating))
    return dict("constant.float",
                {builder.getNamedAttr("value",
                                      builder.getF64FloatAttr(floating))});
  return {};
}

std::string sourceMethodSymbolName(llvm::StringRef className,
                                   llvm::StringRef methodName,
                                   const parser::Node &method) {
  return (llvm::Twine("__ly_method$") + sanitizedSymbolPart(className) + "$" +
          sanitizedSymbolPart(methodName) + "$" +
          llvm::Twine(method.range.start.line) + "_" +
          llvm::Twine(method.range.start.column))
      .str();
}

// The builtin exception taxonomy entry for a contract name, matched by the
// manifest contract ("builtins.ValueError") or its leaf name.
const py::exceptions::BuiltinExceptionInfo *
taxonomyEntryForContract(llvm::StringRef contractName) {
  for (const py::exceptions::BuiltinExceptionInfo &entry :
       py::exceptions::kBuiltinExceptions)
    if (entry.contract == contractName)
      return &entry;
  return py::exceptions::findByName(
      py::contracts::manifestClassNameForContract(contractName));
}

// Linearization of a manifest (non-source) class: builtin exceptions chain
// through the shared taxonomy so mixed-base C3 merges see their common
// ancestors; every chain terminates at builtins.object.
llvm::SmallVector<std::string, 8>
manifestLinearization(llvm::StringRef contractName) {
  llvm::SmallVector<std::string, 8> chain;
  chain.push_back(contractName.str());
  const py::exceptions::BuiltinExceptionInfo *entry =
      taxonomyEntryForContract(contractName);
  while (entry && !entry->baseContract.empty()) {
    entry = py::exceptions::findByContract(entry->baseContract);
    if (!entry)
      break;
    chain.push_back(std::string(entry->contract));
  }
  if (chain.back() != "builtins.object")
    chain.push_back("builtins.object");
  return chain;
}

// The callee expression of a decorator (a Call decorator's func, otherwise
// the decorator itself) and its dotted spelling.
std::pair<const parser::Node *, std::string>
decoratorCallee(const parser::Node &decorator) {
  const parser::Node *callee = &decorator;
  if (decorator.kind == "Call")
    callee = ast::node(decorator, "func");
  std::string spelling;
  if (callee) {
    spelling = ast::qualifiedName(callee);
    if (spelling.empty())
      spelling = std::string(ast::nameSpelling(*callee));
  }
  return {callee, std::move(spelling)};
}

// Does this subtree read an attribute of this name anywhere -- `self.walk`,
// `k.walk`, any receiver? Used to decide whether a generator names itself.
bool bodyMentionsAttribute(const parser::Node &node, llvm::StringRef attr) {
  if (node.kind == "Attribute")
    if (auto name = ast::string(node, "attr");
        name && llvm::StringRef(*name) == attr)
      return true;
  for (const parser::Field &field : node.fields) {
    if (const auto *child = std::get_if<parser::NodePtr>(&field.value)) {
      if (*child && bodyMentionsAttribute(**child, attr))
        return true;
    } else if (const auto *children =
                   std::get_if<std::vector<parser::NodePtr>>(&field.value)) {
      for (const parser::NodePtr &child : *children)
        if (child && bodyMentionsAttribute(*child, attr))
          return true;
    }
  }
  return false;
}

llvm::StringRef decoratorLeafName(llvm::StringRef spelling) {
  std::size_t dot = spelling.rfind('.');
  return dot == llvm::StringRef::npos ? spelling : spelling.drop_front(dot + 1);
}

// A `<prop>.setter` accessor decorator: Attribute whose attr is `setter`
// over a plain name. Returns the property name, empty otherwise.
llvm::StringRef propertySetterTarget(const parser::Node &decorator) {
  if (decorator.kind != "Attribute")
    return {};
  auto attr = ast::string(decorator, "attr");
  if (!attr || *attr != "setter")
    return {};
  const parser::Node *base = ast::node(decorator, "value");
  if (!base || base->kind != "Name")
    return {};
  return ast::nameSpelling(*base);
}

// Property accessor names declared by a class body (getter or setter):
// collectClassFields must not turn `self.<prop> = ...` into a field, and
// setter recognition is scoped to these names.
llvm::StringSet<> classPropertyNames(const parser::Node &classDef) {
  llvm::StringSet<> names;
  const auto *body = ast::nodeList(classDef, "body");
  if (!body)
    return names;
  for (const parser::NodePtr &statement : *body) {
    if (!statement || (statement->kind != "FunctionDef" &&
                       statement->kind != "AsyncFunctionDef"))
      continue;
    auto methodName = ast::string(*statement, "name");
    if (!methodName)
      continue;
    const auto *decorators = ast::nodeList(*statement, "decorator_list");
    if (!decorators)
      continue;
    for (const parser::NodePtr &decorator : *decorators) {
      if (!decorator)
        continue;
      if (decorator->kind == "Name" &&
          ast::nameSpelling(*decorator) == "property") {
        names.insert(*methodName);
        continue;
      }
      llvm::StringRef target = propertySetterTarget(*decorator);
      if (!target.empty())
        names.insert(target);
    }
  }
  return names;
}

// --- synthesized-AST builders (dataclass methods) ---








} // namespace

// The decorators that constrain the checker or the emitted shape rather than
// rebinding the name.
bool ModuleEmitter::isRecognizedNonBindingDecorator(llvm::StringRef leaf) {
  return leaf == "native" || leaf == "overload" || leaf == "override" ||
         leaf == "final" || leaf == "runtime_checkable" ||
         leaf == "staticmethod" || leaf == "classmethod" ||
         leaf == "property" || leaf == "abstractmethod" || leaf == "dataclass";
}

// The expression a decorator CALLS when it rebinds the name it decorates:
// `@d` is `f = d(f)` and `@d(arg)` is `f = d(arg)(f)`. Whatever function the
// program can name answers -- a def of this module, an imported one, `mod.d`
// -- and a factory call over one; null for a marker that rebinds nothing and
// for anything else.
parser::NodePtr
ModuleEmitter::bindingDecoratorCallee(const parser::NodePtr &decorator) {
  if (!decorator)
    return nullptr;
  auto [function, spelling] = decoratorCallee(*decorator);
  if (!function || isRecognizedNonBindingDecorator(decoratorLeafName(spelling)))
    return nullptr;
  if (function->kind == "Name" &&
      moduleFunctionNames.count(ast::nameSpelling(*function)))
    return decorator;
  if (function->kind != "Name" && function->kind != "Attribute")
    return nullptr;
  if (!mlir::isa_and_nonnull<py::CallableType>(
          types.widenLiteral(types.inferExpr(function))))
    return nullptr;
  return decorator;
}

// ⭐ A DECORATED METHOD IS `m = d(m)` IN THE CLASS BODY: the class attribute
// holds what the decorator returned, and a call through an instance binds
// the receiver to it like any function. Here every call of a method is
// inlined, so the rewrite keeps that shape in two methods and a cell --
//
//     class C:
//         def __ly_undecorated$m(self, x): <the body>
//         def m(self, x): return __ly_decorated$C$m(self, x)
//     __ly_decorated$C$m = d(C.__ly_undecorated$m)    # after the class
//
// -- so the body keeps its class (`super()`, the fields it reads), the name
// keeps the signature it was written with, and the decorator runs once.
// ⛔ Only what the forwarder can restate: positional parameters, defaults
// included. A pack, a keyword-only parameter, a generator or a coroutine
// keeps its decorators and their refusal, as does a decorator beside
// staticmethod, classmethod or property, which bind differently.
// ⛔ The application runs after the class statement and not in its body:
// CPython runs it before the class name is bound, which only a decorator
// with a side effect interleaved with the rest of the body could tell.
void ModuleEmitter::desugarMethodDecorators(
    const parser::Node &moduleNode, const EmitOptions::SourceModule *source) {
  auto *body = const_cast<std::vector<parser::NodePtr> *>(
      ast::nodeList(moduleNode, "body"));
  if (!body)
    return;
  auto isTypingMarker = [](llvm::StringRef leaf) {
    return leaf == "overload" || leaf == "override" || leaf == "final";
  };
  auto definesItsOwnGenerator = [](const parser::Node &def) {
    bool found = false;
    std::function<void(const parser::Node &)> walk =
        [&](const parser::Node &node) {
          if (found)
            return;
          if (&node != &def &&
              (node.kind == "FunctionDef" || node.kind == "AsyncFunctionDef" ||
               node.kind == "Lambda" || node.kind == "ClassDef"))
            return;
          if (node.kind == "Yield" || node.kind == "YieldFrom") {
            found = true;
            return;
          }
          for (const parser::Field &field : node.fields) {
            if (const auto *child = std::get_if<parser::NodePtr>(&field.value)) {
              if (*child)
                walk(**child);
            } else if (const auto *children =
                           std::get_if<std::vector<parser::NodePtr>>(
                               &field.value)) {
              for (const parser::NodePtr &child : *children)
                if (child)
                  walk(*child);
            }
          }
        };
    walk(def);
    return found;
  };
  for (const parser::NodePtr &statement : *body) {
    if (!statement || statement->kind != "ClassDef")
      continue;
    if (const auto *typeParams = ast::nodeList(*statement, "type_params");
        typeParams && !typeParams->empty())
      continue;
    std::string className(ast::string(*statement, "name").value_or(""));
    auto *members = const_cast<std::vector<parser::NodePtr> *>(
        ast::nodeList(*statement, "body"));
    if (className.empty() || !members)
      continue;
    for (std::size_t index = 0; index < members->size(); ++index) {
      parser::NodePtr method = (*members)[index];
      if (!method || method->kind != "FunctionDef")
        continue;
      auto *decorators = const_cast<std::vector<parser::NodePtr> *>(
          ast::nodeList(*method, "decorator_list"));
      if (!decorators || decorators->empty())
        continue;
      std::vector<parser::NodePtr> markers;
      std::vector<parser::NodePtr> binding;
      bool bindsOtherwise = false;
      for (const parser::NodePtr &decorator : *decorators) {
        if (!decorator)
          continue;
        auto [function, spelling] = decoratorCallee(*decorator);
        llvm::StringRef leaf = decoratorLeafName(spelling);
        if (isTypingMarker(leaf))
          markers.push_back(decorator);
        else if (isRecognizedNonBindingDecorator(leaf) ||
                 !propertySetterTarget(*decorator).empty() || !function ||
                 (function->kind != "Name" && function->kind != "Attribute"))
          bindsOtherwise = true;
        else
          binding.push_back(decorator);
      }
      const parser::Node *arguments = ast::node(*method, "args");
      const auto *kwonly =
          arguments ? ast::nodeList(*arguments, "kwonlyargs") : nullptr;
      if (bindsOtherwise || binding.empty() || !arguments ||
          ast::node(*arguments, "vararg") || ast::node(*arguments, "kwarg") ||
          (kwonly && !kwonly->empty()) || definesItsOwnGenerator(*method))
        continue;
      std::vector<std::string> positional;
      for (llvm::StringRef group : {"posonlyargs", "args"})
        if (const auto *list = ast::nodeList(*arguments, group))
          for (const parser::NodePtr &arg : *list)
            if (arg)
              positional.push_back(
                  std::string(ast::string(*arg, "arg").value_or("")));
      if (positional.empty())
        continue;
      std::string name(ast::string(*method, "name").value_or(""));
      parser::SourceRange range = method->range;
      DecoratedMethod split;
      split.classDef = statement.get();
      split.hiddenDef = method.get();
      split.className = className;
      split.hiddenName = "__ly_undecorated$" + name;
      split.cellName = "__ly_decorated$" + className + "$" + name;
      split.decorators = binding;
      split.range = range;
      split.source = source;
      split.globalName =
          source ? source->moduleName + "." + split.cellName : split.cellName;
      split.contractName =
          source ? source->moduleName + "." + className : className;
      // The method of the written name: its parameters, defaults and
      // markers, forwarding every argument to what the decorators returned.
      parser::NodePtr forwarder = synth::cloneTree(*method);
      if (parser::Field *field = parser::findField(*forwarder, "decorator_list"))
        field->value = std::move(markers);
      std::vector<parser::NodePtr> forwarded;
      for (const std::string &parameter : positional)
        forwarded.push_back(synth::name(parameter, range));
      std::vector<parser::NodePtr> forwardingBody;
      forwardingBody.push_back(synth::returnStmt(
          synth::call(synth::name(split.cellName, range), std::move(forwarded),
                      range),
          range));
      if (parser::Field *field = parser::findField(*forwarder, "body"))
        field->value = std::move(forwardingBody);
      // The body, under the hidden name and undecorated -- the def CPython
      // runs, called what it was written as.
      decorators->clear();
      if (parser::Field *field = parser::findField(*method, "name"))
        field->value = split.hiddenName;
      pythonNameOverrides[method.get()] = name;
      decoratedMethodBodies.insert(method.get());
      members->insert(members->begin() + static_cast<std::ptrdiff_t>(index) + 1,
                      std::move(forwarder));
      ++index;
      decoratedMethods.push_back(std::move(split));
    }
  }
}

// `d1(d2(subject))` -- `subject` the method's function, `C.__ly_undecorated$m`
// -- with the decorator closest to the def applied first, as it is to a
// function.
parser::NodePtr
ModuleEmitter::decoratedMethodApplication(const DecoratedMethod &method,
                                          parser::NodePtr subject) {
  parser::NodePtr applied = std::move(subject);
  for (const parser::NodePtr &decorator : llvm::reverse(method.decorators)) {
    std::vector<parser::NodePtr> arguments;
    arguments.push_back(std::move(applied));
    applied = synth::call(decorator, std::move(arguments), method.range);
  }
  return applied;
}

// ⭐ A CLASS DECORATOR IS `C = d(C)`, run where the class statement runs,
// innermost first. A class is a declared contract here, so the name stays the
// class it names and the decorator has to give that class back: `d(C)` must
// be `type[C]`, which is what `register[T](cls: type[T]) -> type[T]` and every
// decorator that records a class and returns it are.
//
// A decorator VALUE whose class defines `__ly_decorate_class__` is called
// through that method instead (a Lython protocol, not CPython's): one value
// can then decorate a method -- `__call__`, which returns a NEW function --
// and a class, which it cannot replace. unittest's `skip` is such a value.
void ModuleEmitter::applyClassDecorators(const parser::Node &classDef) {
  const auto *decorators = ast::nodeList(classDef, "decorator_list");
  if (!decorators || decorators->empty() ||
      hoistedClassPrefixes.contains(&classDef))
    return;
  std::string className(ast::nameSpelling(classDef));
  std::optional<mlir::Type> instance = types.lookupClass(className);
  for (auto it = decorators->rbegin(); it != decorators->rend(); ++it) {
    const parser::NodePtr &decorator = *it;
    if (!decorator)
      continue;
    auto [callee, spelling] = decoratorCallee(*decorator);
    llvm::StringRef leaf = decoratorLeafName(spelling);
    if (leaf == "dataclass" || leaf == "overload" || leaf == "override" ||
        leaf == "final" || leaf == "runtime_checkable")
      continue;
    parser::SourceRange range = decorator->range;
    parser::NodePtr applied;
    auto decoratorType = mlir::dyn_cast_if_present<py::ContractType>(
        types.inferExpr(decorator.get()));
    if (decoratorType &&
        resolveMroMethod(decoratorType.getContractName(),
                         "__ly_decorate_class__"))
      applied = synth::methodCall(decorator, "__ly_decorate_class__",
                                  {synth::name(className, range)}, range);
    else
      applied = synth::call(decorator, {synth::name(className, range)}, range);
    synthesizedIteratorDefs.push_back(applied);
    Value result = emitExpr(applied.get());
    auto resultType = mlir::dyn_cast_if_present<py::TypeType>(result.type);
    if (!instance || !resultType ||
        types.widenLiteral(resultType.getInstanceType()) !=
            types.widenLiteral(*instance))
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, range.start,
          "class decorator '" + (spelling.empty() ? "<expression>" : spelling) +
              "' does not return the class it decorates: a class here is a "
              "declared contract, so its name cannot be rebound to what a "
              "decorator returns"});
  }
}

void ModuleEmitter::applyMethodDecorators(const parser::Node &classDef) {
  for (const DecoratedMethod &method : decoratedMethods) {
    if (method.classDef != &classDef || !method.typed || method.source)
      continue;
    parser::NodePtr applied = synth::assign(
        synth::name(method.cellName, method.range),
        decoratedMethodApplication(
            method, synth::attribute(synth::name(method.className, method.range),
                                     method.hiddenName, method.range)),
        method.range);
    synthesizedIteratorDefs.push_back(applied);
    emitStatement(*applied);
  }
}

void ModuleEmitter::checkDecorators(const parser::Node &node,
                                    DecoratorRole role,
                                    const llvm::StringSet<> *propertyNames) {
  const auto *decorators = ast::nodeList(node, "decorator_list");
  if (!decorators)
    return;
  // Recognized-and-ignored typing markers: they constrain the checker, not
  // the emitted code.
  auto isTypingMarker = [](llvm::StringRef leaf) {
    return leaf == "overload" || leaf == "override" || leaf == "final" ||
           leaf == "runtime_checkable";
  };
  for (const parser::NodePtr &decorator : *decorators) {
    if (!decorator)
      continue;
    auto [callee, spelling] = decoratorCallee(*decorator);
    llvm::StringRef leaf = decoratorLeafName(spelling);
    bool recognized = false;
    switch (role) {
    case DecoratorRole::Method:
      recognized = leaf == "staticmethod" || leaf == "classmethod" ||
                   leaf == "property" || leaf == "abstractmethod" ||
                   isTypingMarker(leaf);
      if (!recognized) {
        llvm::StringRef target = propertySetterTarget(*decorator);
        recognized = !target.empty() && propertyNames &&
                     propertyNames->contains(target);
        if (recognized)
          spelling = (target + ".setter").str();
      }
      break;
    case DecoratorRole::Function:
      // ⭐ A DECORATOR THAT NAMES A FUNCTION IS `f = d(f)`, which this compiler
      // has been able to run since a function value started carrying its
      // captures -- the hand-written spelling of exactly this program works.
      // Only the SYNTAX was refused, so `@logged` was rejected while
      // `double = logged(double)` beside it compiled.
      //
      // ⭐ AND A DECORATOR FACTORY, `@deco(arg)`, which is `f = deco(arg)(f)`.
      // The refusal here said the intermediate value is "a function the
      // compiler would have to see through" -- and the hand-written spelling,
      // `shown = tagged("v")(show)`, has compiled all along. What was missing
      // was the desugaring in `declareModuleGlobals`, not the seeing.
      // ⭐ AND ANY FUNCTION THE PROGRAM CAN NAME, not only a def of this
      // module: `from decos import twice` then `@twice`, `@decos.twice`, a
      // factory imported the same way. Each is the `f = d(f)` an imported
      // function has always been callable in.
      // ⛔ A module-level def whose application is not a function stays
      // refused: its name is a cell typed by that application, and with no
      // callable to hold, the undecorated body would answer under the name.
      recognized = leaf == "native" || isTypingMarker(leaf) ||
                   (bindingDecoratorCallee(decorator) &&
                    (!attemptedDecoratedDefs.contains(&node) ||
                     rebindingDecoratedDefs.contains(&node)));
      break;
    case DecoratorRole::Class:
      // Any other decorator of a module-level class is applied at the class
      // statement (applyClassDecorators). ⛔ Not of a class written in a
      // function: that one is declared at the top level and has no
      // statement of its own at which to call it.
      recognized = leaf == "dataclass" || isTypingMarker(leaf) ||
                   !hoistedClassPrefixes.contains(&node);
      break;
    }
    if (recognized)
      continue;
    if (spelling.empty())
      spelling = "<expression>";
    if (role == DecoratorRole::Function &&
        attemptedDecoratedDefs.contains(&node) &&
        bindingDecoratorCallee(decorator)) {
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, decorator->range.start,
          "decorator '" + spelling +
              "' does not return a function here: a decorated module-level "
              "function is rebound to what its decorators return, and only a "
              "callable can be held under its name"});
      continue;
    }
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, decorator->range.start,
        "decorator '" + spelling + "' is not supported (unrecognized "
        "decorators are rejected instead of silently ignored)"});
  }
}

bool ModuleEmitter::isSourceClassContract(mlir::Type type) const {
  auto contract = mlir::dyn_cast_if_present<py::ContractType>(type);
  if (!contract)
    return false;
  llvm::StringRef name = contract.getContractName();
  if (name.starts_with("$"))
    return false;
  // A main-module class is bare; an imported one is `mod.Base` and is in the
  // same declaration map, which a manifest contract never is.
  return !name.contains('.') || declaredClassMethods.count(name) != 0;
}

std::string ModuleEmitter::canonicalClassName(llvm::StringRef spelling) const {
  if (std::optional<mlir::Type> bound = types.lookupClass(spelling))
    if (auto contract = mlir::dyn_cast<py::ContractType>(*bound))
      return contract.getContractName().str();
  return spelling.str();
}

llvm::ArrayRef<std::string>
ModuleEmitter::classMro(llvm::StringRef className) const {
  auto found = classMros.find(className);
  if (found == classMros.end())
    return {};
  return found->second;
}

std::optional<MethodBinding>
ModuleEmitter::resolveMroMethod(llvm::StringRef receiverClass,
                                llvm::StringRef methodName,
                                llvm::StringRef startAfter) const {
  llvm::ArrayRef<std::string> mro = classMro(receiverClass);
  if (mro.empty()) {
    // No linearization record (manifest receiver): direct lookup only.
    auto classMethods = classMethodBindings.find(receiverClass);
    if (classMethods == classMethodBindings.end() || !startAfter.empty())
      return std::nullopt;
    auto method = classMethods->second.find(methodName);
    if (method == classMethods->second.end())
      return std::nullopt;
    return method->second;
  }
  bool active = startAfter.empty();
  for (const std::string &cls : mro) {
    if (!active) {
      if (cls == startAfter)
        active = true;
      continue;
    }
    auto classMethods = classMethodBindings.find(cls);
    if (classMethods == classMethodBindings.end())
      continue;
    auto method = classMethods->second.find(methodName);
    if (method != classMethods->second.end())
      return method->second;
  }
  return std::nullopt;
}

// A subclass of `receiverClass` declares `methodName` of its own.
//
// ⭐ Inlining answers a method call from the receiver's STATIC class, and that
// is only the right answer when no subclass can be behind the reference:
//
//     class A:
//         def v(self) -> int: return 1
//     class B(A):
//         def v(self) -> int: return 2
//     xs: list[A] = [A(), B()]
//     print([a.v() for a in xs])      # printed [1, 1]; CPython prints [1, 2]
//
// `a: A = B()` and a parameter typed `A` did the same. A concrete receiver
// (`B().v()`) was always right, and so was a subclass that does not override.
// This project resolves statically or refuses; it has no dynamic dispatch to
// fall back to, so the call is refused at the earliest boundary that can see
// the hierarchy rather than silently running the base's body.
// ⭐ ONE gate for every dispatch, because there are a dozen of them.
//
// The guard started at the `x.m()` call site alone, and every other way to
// reach a method walked past it -- `len(a)`, `a == b`, `a + b`, `a[i]`,
// `if a:`, `repr(a)`, `with a:`, `for x in a:` all bound the base's body on a
// base-typed receiver while `a.__len__()` on the next line was refused. Eleven
// dunders were measured silently wrong. A predicate with one caller and a
// dozen bypasses is not a guard, so this is the question and the refusal
// together, and the sites ask it rather than re-deriving it.
//
// `receiverNode` may be null: an operator has operands, not a receiver
// expression. Only the `super(...)` exemption needs the syntax; the other two
// are properties of the value.
bool ModuleEmitter::refuseUnresolvableDispatch(const parser::Node &anchor,
                                               Value receiver,
                                               llvm::StringRef methodName,
                                               const parser::Node *receiverNode,
                                               bool throughSuper) {
  if (!dispatchIsUnresolvable(receiver, methodName, receiverNode, throughSuper))
    return false;
  auto contract = mlir::cast<py::ContractType>(receiver.type);
  // ⛔ A CLASSMETHOD IS UNRESOLVABLE WITHOUT AN OVERRIDE, so it does not get
  // the override sentence. `class Sub(Base): pass` redeclares nothing and
  // still binds a different `cls`, and telling the author their method is
  // overridden sends them looking for a subclass that does not exist.
  if (!subclassOverridesMethod(contract.getContractName(), methodName) &&
      !subclassShadowsAttribute(contract.getContractName(), methodName))
    if (std::optional<MethodBinding> bound =
            lookupClassMethod(receiver.type, methodName);
        bound && (bound->kind == "class" || bound->kind == "classmethod")) {
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, anchor.range.start,
          "'" + std::string(methodName) + "' is a classmethod and '" +
              py::contracts::displayClassNameForContract(
                  contract.getContractName()) +
              "' has subclasses, so the `cls` its body reads cannot be "
              "resolved from the static type of the receiver"});
      return true;
    }
  diagnostics.push_back(parser::Diagnostic{
      parser::Severity::Error, anchor.range.start,
      "'" + std::string(methodName) + "' is overridden by a subclass of '" +
          contract.getContractName().str() +
          "', so this call cannot be resolved from the static type of the "
          "receiver"});
  return true;
}

// The question alone. Split out so a site that can ANSWER the call -- the
// synthesized runtime-class dispatcher in EmitterCalls -- can ask it without
// filing the refusal first; the refusal above is the same predicate plus the
// diagnostic, so the two can never disagree about which calls are unresolvable.
bool ModuleEmitter::dispatchIsUnresolvable(Value receiver,
                                           llvm::StringRef methodName,
                                           const parser::Node *receiverNode,
                                           bool throughSuper) const {
  auto contract = mlir::dyn_cast_if_present<py::ContractType>(receiver.type);
  if (!contract)
    return false;
  if (throughSuper)
    return false;
  // Inside a property dispatcher's own body: the arms are narrowed to exact
  // classes and the last one is the base, which is what the dispatcher exists
  // to resolve.
  if (virtualPropertyBodyDepth != 0)
    return false;
  if (receiverNode && receiverNode->kind == "Call") {
    const parser::Node *callee = ast::node(*receiverNode, "func");
    if (callee && callee->kind == "Name" &&
        llvm::StringRef(ast::nameSpelling(*callee)) == "super")
      return false;
  }
  // A CONSTRUCTED receiver names its class exactly, so no subclass can be
  // behind it. Asked of the SSA VALUE, not the expression: the value is the
  // construction for both `Left().tag()` and `x = Left(); x.tag()`, and
  // everything that is not exact answers correctly with no bookkeeping -- an
  // upcast is a different op, a joined binding is a block argument, a field or
  // element read is a load.
  if (mlir::Operation *definition = receiver.value.getDefiningOp())
    if (mlir::isa<py::NewOp, py::InitOp>(definition))
      return false;
  // `self` in the STANDALONE copy of a method body. Every real call inlines
  // the body with `self` bound to the actual receiver, so this guard never
  // fires there; the standalone copy is the one place `self` carries the
  // defining class, and refusing it takes out a program whose every call site
  // is exact.
  //
  // ⛔ AND ONLY THERE. While a body is being INLINED, `self` is bound to the
  // caller's value -- and a plain function's first parameter is an entry
  // argument #0 too, so the test above could not tell the standalone copy from
  // a base-typed parameter the body was inlined for:
  //
  //     class Shape:
  //         def name(self) -> str: return "shape"
  //         def describe(self) -> str: return self.name() + str(self.size)
  //     class Local(Shape):
  //         def name(self) -> str: return "local"
  //     def describe(s: "Shape") -> str: return s.describe()
  //     print(describe(Local(6)))   # printed shape6; CPython prints local6
  //
  // A silent wrong answer, and `Local(6).describe()` beside it is right --
  // that receiver names its class, so the inlined copy resolves `name`
  // exactly. The note above says the distinguishing fact is whether the
  // receiver has a known dynamic class at this site; `methodsBeingInlined` is
  // where that is recorded, because an inlining is precisely the case where
  // `self` is somebody else's value.
  // ⛔ Not a decorated method's body: what the decorator returned calls it
  // with an instance of any subclass, and `self.helper()` in it ran the
  // defining class's `helper` over a subclass's override.
  if (!superContexts.empty() && methodsBeingInlined.empty() &&
      superContexts.back().selfIsExact)
    if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(receiver.value))
      if (argument.getArgNumber() == 0 && argument.getOwner() &&
          argument.getOwner()->isEntryBlock())
        return false;
  // Either kind of redeclaration: a method the subclass overrides, or a
  // class-level binding it shadows. Both are read from the defining class
  // through a base-typed receiver.
  bool redeclared =
      subclassOverridesMethod(contract.getContractName(), methodName) ||
      subclassShadowsAttribute(contract.getContractName(), methodName);
  // ⛔ NOT gated on the name resolving ON THE RECEIVER. A subclass may be the
  // only class that declares it -- `class A: pass` / `class B(A): __repr__` --
  // and `repr(a)` on a base-typed `a` then ran object's repr and printed
  // `<__main__.A object at 0x...>` where CPython prints B's. That a subclass
  // declares it is the whole evidence the dispatch is real; requiring the base
  // to declare it too made the gate blind to exactly the case where the
  // subclass introduces the method.
  if (redeclared)
    return true;
  // ⭐ AND A `@classmethod` WHEREVER A SUBCLASS EXISTS, redeclaration or not.
  // `cls` binds the RUNTIME class, so
  //
  //     class Base:
  //         @classmethod
  //         def tag(cls) -> str: return cls.__name__
  //     class Sub(Base): pass
  //     xs: "list[Base]" = [Base(), Sub()]
  //     print([x.tag() for x in xs])   # CPython ['Base', 'Sub']
  //
  // printed ['Base', 'Base']: the static resolution inlined Base's body with
  // `cls` bound to Base, and nothing in the override test could see it because
  // Sub declares nothing. A silent wrong value, which is what this gate exists
  // to turn into a dispatch.
  if (std::optional<MethodBinding> bound =
          lookupClassMethod(receiver.type, methodName))
    if (bound->kind == "class" || bound->kind == "classmethod")
      // ⛔ AND ONLY WHERE THE BODY READS `cls`. Which class is bound is
      // unobservable in a body that never mentions it, and refusing there
      // costs programs the dispatcher cannot express anyway:
      //
      //     @classmethod
      //     def go(cls, *args: int) -> int: return len(args)
      //
      // -- a vararg the arms have no way to restate -- ran correctly before
      // this gate existed and would have become "'go' is overridden by a
      // subclass of 'Base'" for a fact about `cls` its body does not use.
      if (bound->method && methodBodyReadsFirstParameter(*bound->method))
        return classHasDeclaredSubclass(contract.getContractName());
  return false;
}

// Does the body mention the method's first parameter (`cls`, `self`)? Walked
// whole, nested defs included: a closure that captures it reads it too.
bool ModuleEmitter::methodBodyReadsFirstParameter(
    const parser::Node &method) const {
  const parser::Node *arguments = ast::node(method, "args");
  if (!arguments)
    return false;
  llvm::StringRef parameter;
  for (llvm::StringRef group : {"posonlyargs", "args"}) {
    if (!parameter.empty())
      break;
    if (const auto *list = ast::nodeList(*arguments, group))
      if (!list->empty() && list->front())
        if (std::optional<std::string_view> spelled =
                ast::string(*list->front(), "arg"))
          parameter = llvm::StringRef(spelled->data(), spelled->size());
  }
  if (parameter.empty())
    return false;
  std::function<bool(const parser::Node &)> mentions =
      [&](const parser::Node &node) -> bool {
    if (node.kind == "Name" &&
        llvm::StringRef(ast::nameSpelling(node)) == parameter)
      return true;
    for (const parser::Field &field : node.fields) {
      if (const auto *child = std::get_if<parser::NodePtr>(&field.value)) {
        if (*child && mentions(**child))
          return true;
        continue;
      }
      if (const auto *children =
              std::get_if<std::vector<parser::NodePtr>>(&field.value))
        for (const parser::NodePtr &child : *children)
          if (child && mentions(*child))
            return true;
    }
    return false;
  };
  if (const auto *body = ast::nodeList(method, "body"))
    for (const parser::NodePtr &statement : *body)
      if (statement && mentions(*statement))
        return true;
  return false;
}

bool ModuleEmitter::classHasDeclaredSubclass(
    llvm::StringRef receiverClass) const {
  for (const auto &entry : declaredClassBases) {
    if (entry.getKey() == receiverClass)
      continue;
    llvm::SmallVector<llvm::StringRef, 8> worklist{entry.getKey()};
    llvm::StringSet<> seen;
    while (!worklist.empty()) {
      llvm::StringRef current = worklist.pop_back_val();
      auto bases = declaredClassBases.find(current);
      if (bases == declaredClassBases.end())
        continue;
      for (const std::string &base : bases->second) {
        if (base == receiverClass)
          return true;
        if (seen.insert(base).second)
          worklist.push_back(llvm::StringRef(base));
      }
    }
  }
  return false;
}

bool ModuleEmitter::subclassOverridesMethod(llvm::StringRef receiverClass,
                                            llvm::StringRef methodName) const {
  return subclassRedeclares(declaredClassMethods, receiverClass, methodName);
}

bool ModuleEmitter::subclassShadowsAttribute(
    llvm::StringRef receiverClass, llvm::StringRef attributeName) const {
  return subclassRedeclares(declaredClassAttributes, receiverClass,
                            attributeName);
}

bool ModuleEmitter::subclassReadsFieldOtherwise(
    llvm::StringRef receiverClass, llvm::StringRef fieldName) const {
  return subclassRedeclares(declaredClassReadableNames, receiverClass,
                            fieldName);
}

namespace {
bool mentionsName(const parser::Node &node, llvm::StringRef name) {
  if (node.kind == "Name" && llvm::StringRef(ast::nameSpelling(node)) == name)
    return true;
  for (const parser::Field &field : node.fields) {
    if (const auto *child = std::get_if<parser::NodePtr>(&field.value)) {
      if (*child && mentionsName(**child, name))
        return true;
    } else if (const auto *children =
                   std::get_if<std::vector<parser::NodePtr>>(&field.value)) {
      for (const parser::NodePtr &item : *children)
        if (item && mentionsName(*item, name))
          return true;
    }
  }
  return false;
}
// Whether `self` can reach code other than this statement: any mention of it
// except a read of a field already stored (`self.a` once `self.a = ...` ran),
// which reads the instance and hands it to nothing. A method call through it
// (`self.f()`), a property, or `self` itself as a value all escape.
bool selfEscapes(const parser::Node &node, llvm::StringRef self,
                 const llvm::StringSet<> &stored) {
  if (node.kind == "Attribute")
    if (const parser::Node *owner = ast::node(node, "value");
        owner && owner->kind == "Name" &&
        llvm::StringRef(ast::nameSpelling(*owner)) == self)
      return !stored.contains(ast::string(node, "attr").value_or(""));
  if (node.kind == "Call")
    if (const parser::Node *callee = ast::node(node, "func");
        callee && callee->kind == "Attribute")
      if (const parser::Node *owner = ast::node(*callee, "value");
          owner && owner->kind == "Name" &&
          llvm::StringRef(ast::nameSpelling(*owner)) == self)
        return true;
  if (node.kind == "Name" && llvm::StringRef(ast::nameSpelling(node)) == self)
    return true;
  for (const parser::Field &field : node.fields) {
    if (const auto *child = std::get_if<parser::NodePtr>(&field.value)) {
      if (*child && selfEscapes(**child, self, stored))
        return true;
    } else if (const auto *children =
                   std::get_if<std::vector<parser::NodePtr>>(&field.value)) {
      for (const parser::NodePtr &item : *children)
        if (item && selfEscapes(*item, self, stored))
          return true;
    }
  }
  return false;
}
} // namespace

// The fields a construction of `cls` has stored by the time `self` can be
// seen by anything but `__init__`'s own straight-line prefix: the walk takes
// `self.x = <no self>` and a `super().__init__(...)` / `Base.__init__(self,
// ...)` that passes nothing else of self, and stops at the first statement
// that mentions self any other way -- a method call, a self-reading value, a
// branch. A field it reaches is stored before any code that could read it
// runs, so its reads need no test.
//
// ⛔ A branch stops the walk even when both arms store: the cheap answer is
// the conservative one, and the cost of being conservative is a compare.
std::optional<llvm::StringSet<>>
ModuleEmitter::fieldsStoredByConstruction(llvm::StringRef cls) {
  if (auto memo = constructionStoreMemo.find(cls);
      memo != constructionStoreMemo.end())
    return memo->second;
  // A cycle (a class that names itself) answers "nothing" while it is open.
  constructionStoreMemo[cls] = llvm::StringSet<>{};
  std::optional<llvm::StringSet<>> answer = llvm::StringSet<>{};
  auto node = declaredClassNodes.find(cls);
  if (node == declaredClassNodes.end() || !node->second) {
    constructionStoreMemo[cls] = answer;
    return answer;
  }
  const parser::Node &classDef = *node->second;
  // Constructions this walk does not read: every field is a parameter of a
  // synthesized __init__.
  if (const auto *decorators = ast::nodeList(classDef, "decorator_list"))
    for (const parser::NodePtr &decorator : *decorators) {
      if (!decorator)
        continue;
      const parser::Node *callee = decorator->kind == "Call"
                                       ? ast::node(*decorator, "func")
                                       : decorator.get();
      std::string spelled = ast::qualifiedName(callee);
      if (llvm::StringRef(spelled).ends_with("dataclass")) {
        constructionStoreMemo[cls] = std::nullopt;
        return std::nullopt;
      }
    }
  if (const auto *bases = ast::nodeList(classDef, "bases"))
    for (const parser::NodePtr &base : *bases)
      if (base &&
          llvm::StringRef(ast::qualifiedName(base.get())).ends_with("NamedTuple")) {
        constructionStoreMemo[cls] = std::nullopt;
        return std::nullopt;
      }
  auto firstSourceBase = [&]() -> std::optional<std::string> {
    auto bases = declaredClassBases.find(cls);
    if (bases == declaredClassBases.end())
      return std::nullopt;
    for (const std::string &base : bases->second)
      if (declaredClassNodes.count(base))
        return base;
    return std::nullopt;
  };
  const parser::Node *init = nullptr;
  if (const auto *body = ast::nodeList(classDef, "body"))
    for (const parser::NodePtr &member : *body)
      if (member && member->kind == "FunctionDef" &&
          ast::string(*member, "name").value_or("") == "__init__")
        init = member.get();
  if (!init) {
    if (std::optional<std::string> base = firstSourceBase())
      answer = fieldsStoredByConstruction(*base);
    constructionStoreMemo[cls] = answer;
    return answer;
  }
  std::string self = "self";
  if (const parser::Node *args = ast::node(*init, "args"))
    if (const auto *positional = ast::nodeList(*args, "args");
        positional && !positional->empty() && positional->front())
      self = std::string(ast::nameSpelling(*positional->front()));
  auto storedName = [&](const parser::Node *target) -> std::optional<std::string> {
    if (!target || target->kind != "Attribute")
      return std::nullopt;
    const parser::Node *owner = ast::node(*target, "value");
    if (!owner || owner->kind != "Name" || ast::nameSpelling(*owner) != self)
      return std::nullopt;
    if (auto attr = ast::string(*target, "attr"))
      return std::string(*attr);
    return std::nullopt;
  };
  auto merge = [&](const std::optional<llvm::StringSet<>> &more) {
    if (!more) {
      answer = std::nullopt;
      return;
    }
    if (answer)
      for (const auto &entry : *more)
        answer->insert(entry.getKey());
  };
  if (const auto *statements = ast::nodeList(*init, "body"))
    for (const parser::NodePtr &statement : *statements) {
      if (!statement)
        continue;
      if (statement->kind == "Pass")
        continue;
      if (statement->kind == "Assign" || statement->kind == "AnnAssign") {
        const parser::Node *value = ast::node(*statement, "value");
        llvm::SmallVector<std::string, 2> names;
        bool allFields = true;
        if (statement->kind == "Assign") {
          if (const auto *targets = ast::nodeList(*statement, "targets"))
            for (const parser::NodePtr &target : *targets) {
              std::optional<std::string> name = storedName(target.get());
              if (!name)
                allFields = false;
              else
                names.push_back(*name);
            }
        } else if (std::optional<std::string> name =
                       storedName(ast::node(*statement, "target"))) {
          names.push_back(*name);
        } else {
          allFields = false;
        }
        static const llvm::StringSet<> kNothing;
        if (allFields && !names.empty() && value &&
            !selfEscapes(*value, self, answer ? *answer : kNothing)) {
          if (answer)
            for (const std::string &name : names)
              answer->insert(name);
          continue;
        }
      }
      if (statement->kind == "Expr")
        if (const parser::Node *call = ast::node(*statement, "value");
            call && call->kind == "Call") {
          const parser::Node *callee = ast::node(*call, "func");
          const auto *args = ast::nodeList(*call, "args");
          bool superInit = false;
          std::optional<std::string> named;
          if (callee && callee->kind == "Attribute" &&
              ast::string(*callee, "attr").value_or("") == "__init__") {
            const parser::Node *owner = ast::node(*callee, "value");
            if (owner && owner->kind == "Call")
              if (const parser::Node *superName = ast::node(*owner, "func");
                  superName && superName->kind == "Name" &&
                  ast::nameSpelling(*superName) == "super")
                superInit = true;
            if (owner && owner->kind == "Name")
              named = canonicalClassName(ast::nameSpelling(*owner));
          }
          bool othersClean = true;
          if (args)
            for (auto [index, argument] : llvm::enumerate(*args)) {
              if (!argument)
                continue;
              bool receiver = named && index == 0 &&
                              argument->kind == "Name" &&
                              ast::nameSpelling(*argument) == self;
              if (!receiver && mentionsName(*argument, self))
                othersClean = false;
            }
          if (othersClean && superInit) {
            if (std::optional<std::string> base = firstSourceBase())
              merge(fieldsStoredByConstruction(*base));
            continue;
          }
          if (othersClean && named && declaredClassNodes.count(*named) &&
              args && !args->empty()) {
            merge(fieldsStoredByConstruction(*named));
            continue;
          }
        }
      static const llvm::StringSet<> kNone;
      if (statement->kind == "Return" ||
          selfEscapes(*statement, self, answer ? *answer : kNone))
        break;
    }
  constructionStoreMemo[cls] = answer;
  return answer;
}

bool ModuleEmitter::fieldAlwaysStoredBeforeRead(llvm::StringRef cls,
                                                llvm::StringRef field) {
  // The receiver may be an instance of `cls` or of any subclass, and each
  // constructs itself.
  llvm::SmallVector<std::string, 8> classes;
  if (!protocolClassNames.contains(cls))
    classes.push_back(cls.str());
  for (const auto &entry : declaredClassBases) {
    llvm::StringRef candidate = entry.getKey();
    if (candidate == cls)
      continue;
    llvm::SmallVector<llvm::StringRef, 8> worklist{candidate};
    llvm::StringSet<> seen;
    bool derived = false;
    while (!worklist.empty() && !derived) {
      auto bases = declaredClassBases.find(worklist.pop_back_val());
      if (bases == declaredClassBases.end())
        continue;
      for (const std::string &base : bases->second) {
        if (base == cls)
          derived = true;
        if (seen.insert(base).second)
          worklist.push_back(base);
      }
    }
    if (derived)
      classes.push_back(candidate.str());
  }
  for (const std::string &each : classes) {
    if (!declaredClassNodes.count(each))
      continue;
    std::optional<llvm::StringSet<>> stored = fieldsStoredByConstruction(each);
    if (stored && !stored->contains(field))
      return false;
  }
  return true;
}

void ModuleEmitter::markFieldReadMaybeUnset(mlir::Operation *read,
                                            mlir::Type objectType,
                                            llvm::StringRef field) {
  auto contract = mlir::dyn_cast_if_present<py::ContractType>(objectType);
  if (!read || !contract || !declaredClassNodes.count(contract.getContractName()))
    return;
  if (!fieldAlwaysStoredBeforeRead(contract.getContractName(), field))
    read->setAttr("ly.field.maybe_unset", builder.getUnitAttr());
}

bool ModuleEmitter::classLineBindsReadable(llvm::StringRef cls,
                                           llvm::StringRef name) const {
  llvm::SmallVector<llvm::StringRef, 8> line{cls};
  llvm::StringSet<> seen;
  while (!line.empty()) {
    llvm::StringRef current = line.pop_back_val();
    auto readable = declaredClassReadableNames.find(current);
    if (readable != declaredClassReadableNames.end() &&
        readable->second.contains(name))
      return true;
    auto bases = declaredClassBases.find(current);
    if (bases != declaredClassBases.end())
      for (const std::string &base : bases->second)
        if (seen.insert(base).second)
          line.push_back(base);
  }
  return false;
}

bool ModuleEmitter::fieldEverStored(llvm::StringRef cls,
                                    llvm::StringRef fieldName) const {
  // A store through another name may fill it too (`q.longMessage = True`),
  // and which class that receiver has is not known before the types are.
  if (programAttributeStoreNames().contains(fieldName))
    return true;
  auto stores = [&](llvm::StringRef current) {
    auto found = declaredClassStoredFields.find(current);
    return found != declaredClassStoredFields.end() &&
           found->second.contains(fieldName);
  };
  llvm::SmallVector<llvm::StringRef, 8> line{cls};
  llvm::StringSet<> seen;
  while (!line.empty()) {
    llvm::StringRef current = line.pop_back_val();
    if (stores(current))
      return true;
    auto bases = declaredClassBases.find(current);
    if (bases != declaredClassBases.end())
      for (const std::string &base : bases->second)
        if (seen.insert(base).second)
          line.push_back(base);
  }
  return subclassRedeclares(declaredClassStoredFields, cls, fieldName);
}

bool ModuleEmitter::subclassRedeclares(
    const llvm::StringMap<llvm::StringSet<>> &declarations,
    llvm::StringRef receiverClass, llvm::StringRef name) const {
  for (const auto &entry : declaredClassBases)
    if (candidateRedeclares(declarations, receiverClass, entry.getKey(), name))
      return true;
  return false;
}

// ⭐ A declaration the subclass reaches WITHOUT going through the receiver
// class is an override, whether the subclass wrote it or inherited it from a
// mixin. Testing only what the subclass declares itself missed
// `class B(M, A): pass` -- B declares nothing, resolves `v` to M's, and the
// base's body was inlined for it.
//
// The receiver's own ancestors are excluded because a declaration there is
// what the receiver already resolves to, not a competitor. Conservative in a
// diamond, which is the direction that refuses rather than the direction that
// mis-executes.
//
// ⛔ NOT inlined back into the gate. `virtualDispatcherFor` scans for the same
// candidates and used to ask only what each one DECLARES ITSELF, so
// `class Both(Base, Mixin)` was refused by the gate (Mixin declares it) and
// then found no candidate to dispatch to -- one question, two predicates, and
// a valid program in the gap between them.
// Ancestors as WRITTEN, from the pre-pass rather than from the MRO map the
// class emission fills: the answer must not depend on where in the file the
// question is asked (`collectTopLevelBindings`).
// ⭐ AND PAST A RUNTIME BASE, through its own: `class E(Exception)` is a
// BaseException too, and a gate asked about a `BaseException` receiver
// walked from E to Exception and stopped -- `except (E, F) as e: str(e)`
// then printed the args where CPython calls E's `__str__`.
// ⛔ Not to `object`, which no program class names as written either: an
// `object` value is answered at run time by its class's slot, and a gate
// that saw E below it refused `repr(o)` for every `o: object`.
void ModuleEmitter::collectDeclaredAncestors(llvm::StringRef cls,
                                             llvm::StringSet<> &out) const {
  llvm::SmallVector<llvm::StringRef, 8> worklist{cls};
  while (!worklist.empty()) {
    llvm::StringRef current = worklist.pop_back_val();
    auto bases = declaredClassBases.find(current);
    if (bases == declaredClassBases.end()) {
      // ⛔ Not `drop_begin(manifestLinearization(current))` in the loop
      // header: C++17 does not extend the returned vector's life through
      // the adaptor, and the loop read freed strings: a name past the
      // string's inline buffer (libstdc++: every `builtins.*`) was missed,
      // and `str(e)` printed the builtin's text.
      auto linearization = manifestLinearization(current);
      for (const std::string &ancestor : llvm::drop_begin(linearization))
        if (ancestor != "builtins.object")
          out.insert(ancestor);
      continue;
    }
    for (const std::string &base : bases->second)
      if (out.insert(base).second)
        worklist.push_back(base);
  }
}

bool ModuleEmitter::candidateRedeclares(
    const llvm::StringMap<llvm::StringSet<>> &declarations,
    llvm::StringRef receiverClass, llvm::StringRef candidate,
    llvm::StringRef name) const {
  if (candidate == receiverClass)
    return false;
  llvm::StringSet<> candidateAncestors;
  collectDeclaredAncestors(candidate, candidateAncestors);
  if (!candidateAncestors.contains(receiverClass))
    return false;
  llvm::StringSet<> receiverAncestors;
  collectDeclaredAncestors(receiverClass, receiverAncestors);
  llvm::SmallVector<llvm::StringRef, 8> reachable{candidate};
  for (const auto &ancestor : candidateAncestors)
    reachable.push_back(ancestor.getKey());
  for (llvm::StringRef cls : reachable) {
    if (cls == receiverClass || receiverAncestors.contains(cls))
      continue;
    auto declared = declarations.find(cls);
    if (declared != declarations.end() && declared->second.contains(name))
      return true;
  }
  return false;
}

bool ModuleEmitter::isExceptionBackedClass(llvm::StringRef className) const {
  for (const std::string &cls : classMro(className))
    if (!classMros.count(cls) && taxonomyEntryForContract(cls))
      return true;
  return false;
}

bool ModuleEmitter::isImplementedObjectDefault(llvm::StringRef methodName) {
  return methodName == "__eq__" || methodName == "__ne__" ||
         methodName == "__hash__" || methodName == "__bool__" ||
         methodName == "__repr__" || methodName == "__str__";
}

bool ModuleEmitter::inheritsObjectDefaultDunder(
    mlir::Type type, llvm::StringRef methodName) const {
  auto contract =
      mlir::dyn_cast_if_present<py::ContractType>(types.widenLiteral(type));
  if (!contract)
    return false;
  llvm::StringRef className = contract.getContractName();
  if (!classMros.count(className))
    return false;
  const py::protocols::Table &table = py::protocols::Table::get(context);
  for (const std::string &cls : classMro(className)) {
    if (cls == "builtins.object")
      continue;
    auto methods = classMethodBindings.find(cls);
    if (methods != classMethodBindings.end() && methods->second.count(methodName))
      return false;
    if (classMros.count(cls))
      continue;
    // A manifest base's OWN declarations only: asking the table to resolve the
    // name would walk that base up to object too and answer "provided" for
    // every class.
    const py::protocols::ProtocolInfo *info =
        table.lookup(py::contracts::manifestClassNameForContract(cls));
    if (info && info->methods.count(methodName.str()))
      return false;
  }
  return true;
}

std::optional<std::pair<llvm::StringRef, mlir::Type>>
ModuleEmitter::resolveClassAttrSlot(llvm::StringRef className,
                                    llvm::StringRef attrName) const {
  llvm::ArrayRef<std::string> mro = classMro(className);
  if (mro.empty()) {
    auto slots = classAttrSlots.find(className);
    if (slots == classAttrSlots.end())
      return std::nullopt;
    auto slot = slots->second.find(attrName);
    if (slot == slots->second.end())
      return std::nullopt;
    return std::make_pair(slots->first(), slot->second);
  }
  for (const std::string &cls : mro) {
    auto slots = classAttrSlots.find(cls);
    if (slots != classAttrSlots.end()) {
      auto slot = slots->second.find(attrName);
      if (slot != slots->second.end())
        return std::make_pair(slots->first(), slot->second);
    }
    // ⛔ STOP AT THE FIRST CLASS IN THE MRO THAT DECLARES IT, slot or no slot.
    // Only a main-module class gets slots -- an imported one keeps its
    // attributes on the constant channel -- so walking past it handed back a
    // LATER class's storage:
    //
    //     # lib.py: class A: tag = "a"
    //     class M:            tag = "m"
    //     class BaseFirst(A, M): pass
    //     print(BaseFirst().tag)   # printed m; Python reads A's, "a"
    //
    // Silently, and correct with A written in the same file. A main-module
    // class whose own attribute is not storable (a container) is the same
    // question: its value is what Python reads, not its base's slot.
    auto declared = declaredClassAttributes.find(cls);
    if (declared != declaredClassAttributes.end() &&
        declared->second.contains(attrName))
      return std::nullopt;
  }
  return std::nullopt;
}

// ⭐ `__init_subclass__` RUNS WHEN THE SUBCLASS IS DEFINED, and nothing ran it:
// `class Sub(Meta): pass` printed nothing where CPython prints what the hook
// prints, with no diagnostic -- the one shape on this sweep that answered
// differently rather than refusing.
//
// It is an implicit classmethod, so the call is spelled through the SUBCLASS
// (`Sub.__init_subclass__()`): looking an inherited classmethod up through the
// subclass binds `cls` to it, which is the argument CPython passes.
//
// ⛔ NOT through the attribute spelling when the class declares its own: that
// finds the class's own method, which CPython never runs on itself, and
// `Base.__init_subclass__()` binds `cls` to Base, a wrong answer rather than a
// missing one. What reaches the ANCESTOR's body with `cls` bound to the NEW
// class is a specialization -- the same `preboundTypeObject` the bound-method
// wrapper uses, which drops `cls` from the signature and materializes the type
// object inside the copy. Measured in
// tests/probe/wb_init_subclass_through_a_middle_class.py: the class in the
// middle printed only its own hook, never its parent's.
void ModuleEmitter::emitInitSubclassHook(const parser::Node &classDef,
                                         llvm::StringRef contractName) {
  auto name = ast::string(classDef, "name");
  if (!name)
    return;
  // ⭐ AN IMPORTED CLASS'S HOOK RUNS TOO, and it did not: this is emitted at
  // the class statement's position in MODULE flow, and an imported module's
  // body does not run -- so a library whose base declares
  // `__init_subclass__` registered nothing at all, silently, which is the one
  // thing that pattern exists to do. The contract is the QUALIFIED one there;
  // the SPELLING stays the module's own, which is what the synthesized call
  // resolves through inside its defining scope.
  mlir::Type ownContract =
      contractName.empty() ? types.contract(*name) : types.contract(contractName);
  // ⭐ THE CLASS HEADER'S KEYWORDS ARE THE HOOK'S ARGUMENTS, and they were
  // dropped on the floor: `class Sub(Base, tag="s")` compiled, ran, and
  // printed nothing where CPython prints what the hook prints -- and with a
  // hook that takes no keywords at all it compiled where CPython raises
  // TypeError. Both silent.
  //
  // ⛔ `metaclass=` is NOT one of them and is refused rather than passed on:
  // it selects the type that BUILDS the class, which this compiler does not
  // model, so accepting it silently would run none of what it names.
  std::vector<const parser::Node *> classKeywords;
  bool refusedKeyword = false;
  if (const auto *keywords = ast::nodeList(classDef, "keywords"))
    for (const parser::NodePtr &keyword : *keywords) {
      if (!keyword)
        continue;
      auto keywordName = ast::string(*keyword, "arg");
      if (!keywordName) {
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, keyword->range.start,
            "'**' in a class header is not supported: the keywords a class "
            "passes to __init_subclass__ have to be named here"});
        refusedKeyword = true;
        continue;
      }
      if (*keywordName == "metaclass") {
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, keyword->range.start,
            "'metaclass=' is not supported: a class is built by this compiler, "
            "so a metaclass would not run"});
        refusedKeyword = true;
        continue;
      }
      classKeywords.push_back(keyword.get());
    }
  if (refusedKeyword)
    return;
  auto refuseKeywords = [&](llvm::StringRef reason) {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, classDef.range.start,
        "class '" + std::string(*name) +
            "' passes keywords that its __init_subclass__ " + reason.str()});
  };

  std::optional<MethodBinding> hook =
      lookupClassMethod(ownContract, "__init_subclass__");
  if (!hook || !hook->method) {
    if (!classKeywords.empty())
      refuseKeywords("does not declare (no hook takes them)");
    return;
  }
  std::string ownName =
      contractName.empty() ? std::string(*name) : contractName.str();
  bool declaresItsOwn = hook->definingClass == ownName;
  if (declaresItsOwn) {
    // The class declares its own, so the hook that runs for it is the nearest
    // one ABOVE it. `startAfter` is what super() positions itself with.
    std::string canonical =
        contractName.empty() ? canonicalClassName(*name) : ownName;
    hook = resolveMroMethod(canonical, "__init_subclass__", canonical);
    if (!hook || !hook->method) {
      if (!classKeywords.empty())
        refuseKeywords("does not declare (no hook takes them)");
      return;
    }
  }
  const FunctionSignature &hookSig = hook->bodySignature;
  if (hookSig.positionalNames.empty())
    return; // no `cls`: not a shape this models
  // Every class keyword has to name a parameter past `cls`, and every such
  // parameter the header does not name has to have a default -- the same two
  // conditions CPython's own call raises TypeError for.
  llvm::StringSet<> supplied;
  for (const parser::Node *keyword : classKeywords)
    supplied.insert(*ast::string(*keyword, "arg"));
  for (const parser::Node *keyword : classKeywords) {
    llvm::StringRef spelling = *ast::string(*keyword, "arg");
    bool declared = llvm::is_contained(hookSig.kwOnlyNames, spelling);
    for (std::size_t index = 1; index < hookSig.positionalNames.size(); ++index)
      declared = declared || hookSig.positionalNames[index] == spelling;
    if (!declared) {
      refuseKeywords("does not declare");
      return;
    }
  }
  for (std::size_t index = 1; index < hookSig.positionalNames.size(); ++index)
    if (!supplied.contains(hookSig.positionalNames[index]) &&
        (index >= hookSig.positionalDefaults.size() ||
         !hookSig.positionalDefaults[index])) {
      refuseKeywords("requires and the header does not pass");
      return;
    }
  for (auto [index, kwOnly] : llvm::enumerate(hookSig.kwOnlyNames))
    if (!supplied.contains(kwOnly) &&
        (index >= hookSig.kwOnlyDefaults.size() ||
         !hookSig.kwOnlyDefaults[index])) {
      refuseKeywords("requires and the header does not pass");
      return;
    }
  parser::NodePtr callee;
  if (!declaresItsOwn) {
    callee = synth::attribute(synth::name(*name, classDef.range),
                              "__init_subclass__", classDef.range);
  } else {
    // The lookup through the subclass would find the class's own body, so
    // emit a copy of the ancestor's with `cls` prebound to this class.
    FunctionSignature bound = hook->bodySignature;
    bound.positionalTypes.erase(bound.positionalTypes.begin());
    bound.positionalNames.erase(bound.positionalNames.begin());
    if (!bound.positionalDefaults.empty())
      bound.positionalDefaults.erase(bound.positionalDefaults.begin());
    if (bound.positionalOnlyCount > 0)
      --bound.positionalOnlyCount;
    types.refreshCallable(bound);
    std::string symbolName =
        (llvm::Twine(hook->symbolName) + "$initsubclass$" + *name).str();
    emitCallableFunction(*hook->method, symbolName, bound, {},
                         /*isLambda=*/false, /*positionalNodeOffset=*/1,
                         ownContract);
    // ⛔ Not left to the binding `emitCallableFunction` makes: that one is
    // inside the copy's own emission scope and is gone by the time the call is
    // emitted. The spelling carries `$`, so no source name can collide.
    types.bindSymbol(symbolName, bound.callable);
    callee = synth::name(symbolName, classDef.range);
  }
  std::vector<parser::NodePtr> keywordArguments;
  for (const parser::Node *keyword : classKeywords) {
    const parser::Field *value = parser::findField(*keyword, "value");
    if (!value || !std::holds_alternative<parser::NodePtr>(value->value))
      return;
    keywordArguments.push_back(synth::keyword(
        *ast::string(*keyword, "arg"),
        std::get<parser::NodePtr>(value->value), keyword->range));
  }
  parser::NodePtr call = synth::callWithKeywords(
      std::move(callee), std::vector<parser::NodePtr>{},
      std::move(keywordArguments), classDef.range);
  emitStatement(*synth::exprStmt(std::move(call), classDef.range));
}

// The queue `emitClassContract` fills while the top level is declaring classes.
// Everything the body loop there sets up per method is reconstructed from the
// contract name it was queued with: the `super()` context is the class and the
// body's own first positional name, and the frozen-dataclass exemption is the
// class again.
void ModuleEmitter::emitDeferredMethodBodies() {
  // ⛔ Index, not a range-for: emitting a body can demand a generic
  // specialization, whose class contract queues bodies of its own onto this
  // very vector.
  for (std::size_t index = 0; index < deferredMethodBodies.size(); ++index) {
    DeferredMethodBody entry = deferredMethodBodies[index];
    if (!entry.statement)
      continue;
    bool instanceBody =
        entry.kind == "instance" && !entry.signature.positionalNames.empty();
    std::optional<llvm::SaveAndRestore<const std::string *>> frozenInit;
    if (ast::string(*entry.statement, "name").value_or("") == "__init__" &&
        frozenDataclassContracts.count(entry.contractName))
      frozenInit.emplace(frozenInitContract, &entry.contractName);
    // ⭐ IN THE MODULE THE CLASS WAS DECLARED IN. A generic class from an
    // imported module is specialized while the importer is being emitted,
    // and its bodies were queued here and then emitted in the importer's
    // scope: `get_event_loop()` in asyncio's `Task.__init__` was an
    // unresolved name in the program that awaited `asyncio.gather`.
    // ⛔ The super() context is pushed inside: the module scope switch
    // starts from an empty one.
    auto emitBody = [&] {
      // ⭐ A GENERIC CLASS'S BODY SPELLS ITS TYPE PARAMETERS (`self.x: E |
      // None`), and the specialization bound them only for its own
      // emission, which queued this body and returned. Drained here with
      // nothing bound, `E` read as a class of that name, and the store was
      // refused in the lowering against the field the contract had typed
      // right: `Ctx[E](...)` built in a generic method called from any
      // function body.
      auto typeParameterScope = types.pushScope();
      bindClassTypeArguments(entry.definer);
      if (instanceBody)
        superContexts.push_back(SuperContext{
            entry.contractName, entry.signature.positionalNames.front(),
            !decoratedMethodBodies.contains(entry.statement)});
      emitCallableFunction(*entry.statement, entry.symbolName,
                           entry.signature, {}, /*isLambda=*/false);
      if (instanceBody)
        superContexts.pop_back();
    };
    if (entry.source)
      emitInDefiningModuleScope(*entry.source, emitBody);
    else
      emitBody();
  }
  deferredMethodBodies.clear();
}

void ModuleEmitter::emitClassAttrInitializers(const parser::Node &classDef,
                                              llvm::StringRef contractName) {
  auto name = ast::string(classDef, "name");
  if (!name)
    return;
  std::string slotKey =
      contractName.empty() ? std::string(*name) : contractName.str();
  // The class body's own names are bound as each initializer is emitted (see
  // the store below) and must not outlive it: `width` is an attribute of the
  // class, not a module global.
  ScopedEmitterScope classBodyScope(values, types);
  auto slots = classAttrSlots.find(slotKey);
  if (slots == classAttrSlots.end() || slots->second.empty())
    return;
  const auto *body = ast::nodeList(classDef, "body");
  if (!body)
    return;
  for (const parser::NodePtr &statement : *body) {
    if (!statement)
      continue;
    const parser::Node *target = nullptr;
    const parser::Node *value = nullptr;
    if (statement->kind == "Assign") {
      const auto *targets = ast::nodeList(*statement, "targets");
      if (!targets || targets->size() != 1 || !targets->front() ||
          targets->front()->kind != "Name")
        continue;
      target = targets->front().get();
      value = ast::node(*statement, "value");
    } else if (statement->kind == "AnnAssign") {
      target = ast::node(*statement, "target");
      value = ast::node(*statement, "value");
      if (!target || target->kind != "Name")
        continue;
    } else {
      continue;
    }
    if (!value)
      continue;
    // ⭐ `__set_name__` IS CALLED WHEN THE INSTANCE BECOMES A CLASS ATTRIBUTE,
    // and nothing calls it here -- `a = Field()` in a class body printed
    // nothing where CPython prints what the hook prints, with no diagnostic.
    // Refused at the assignment rather than at the method's declaration: the
    // hook is defined on the DESCRIPTOR class, which a program may define and
    // call by hand, and this is the one place the call is implicit.
    //
    // ⛔ NOT implemented, because the owner argument is the CLASS and a class
    // has no object handle: `f(C)` for `def f(o: object)` already dies in the
    // lowering ("cannot pass ... as builtins.object"), which is the standard
    // signature's parameter type. It waits on a type object as a runtime
    // value, recorded in [[lython-remaining-mechanisms]].
    if (mlir::Type valueType = types.widenLiteral(types.inferExpr(value));
        valueType && lookupClassMethod(valueType, "__set_name__"))
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, statement->range.start,
          "a class attribute whose class defines __set_name__ is not "
          "supported: the hook CPython calls here is never run"});
    if (value->kind == "Call") {
      auto [callee, spelling] = decoratorCallee(*value);
      if (decoratorLeafName(spelling) == "field") {
        // dataclasses.field(...) never evaluates as a value; the cell
        // takes the default= expression, and a factory field has no
        // class-level attribute at all (as in CPython).
        const parser::Node *defaultValue = nullptr;
        if (const auto *keywords = ast::nodeList(*value, "keywords"))
          for (const parser::NodePtr &keyword : *keywords)
            if (keyword && ast::string(*keyword, "arg") == "default")
              defaultValue = ast::node(*keyword, "value");
        if (!defaultValue)
          continue;
        value = defaultValue;
      }
    }
    llvm::StringRef attrName = ast::nameSpelling(*target);
    auto slot = slots->second.find(attrName);
    if (slot == slots->second.end())
      continue;
    Value initial = emitExprExpected(value, slot->second);
    // ⭐ THE SAME REFUSAL THE MODULE-GLOBAL WRITE MAKES, and this channel was
    // missing it:
    //
    //     class P:
    //         v: float = 1
    //     print(P.v)
    //     # RuntimeError: module global 'P.v' referenced before assignment
    //
    // `x: float = 1` at module scope says so at EMIT ("a module global has one
    // runtime representation and these two do not share one"), because
    // `coerceValue` deliberately declines to retype between the numeric
    // contracts. The class-attribute cell is the same storage with the same
    // rule, and without the check the store of an int into a float cell was
    // dropped further down -- leaving the cell unassigned and the failure to
    // the reader at RUNTIME, naming an internal cell name.
    //
    // ⭐ AND AT ANY DEPTH, since a container's element type is its storage:
    // `v: list[float] = [1]` printed 5e-324 and `v: list[int] = [True]`
    // printed 0, both silently, because the outer contract names match and the
    // scalar test stopped there.
    mlir::Type declared, supplied;
    // ⛔ And the initializer's OWN type, not only what came back: an expected
    // union wraps the value first, so `v: int | None = True` came back as the
    // union and the bool reached the lowering.
    if (numericRepresentationMismatch(slot->second, initial.type, declared,
                                      supplied) ||
        numericRepresentationMismatch(
            slot->second, types.widenLiteral(types.inferExpr(value)), declared,
            supplied)) {
      auto spell = [&](mlir::Type numeric) -> llvm::StringRef {
        if (numeric == types.boolType())
          return "bool";
        return numeric == types.intType() ? "int" : "float";
      };
      bool insideUnion =
          mlir::isa<py::UnionType>(types.widenLiteral(slot->second));
      bool insideContainer =
          !insideUnion && declared != types.widenLiteral(slot->second);
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, statement->range.start,
          "class attribute '" + std::string(*name) + "." + attrName.str() +
              "' holds " + (insideContainer ? "a container of " : "") +
              (insideUnion ? cellTypeSpelling(slot->second)
                           : spell(declared).str()) +
              " and this initializer gives it " +
              (insideContainer ? "a container of " : "") +
              spell(supplied).str() +
              "; the attribute's cell has one runtime representation and these "
              "two do not share one, so write the value in the declared type"});
      continue;
    }
    Value coerced = coerceValue(initial, slot->second, *statement);
    // The DEFINING class's contract name, not the source spelling: an
    // imported class's cell is `mod.Registry.items` on every reader's side,
    // and a store under the bare name left the cell the readers use unwritten.
    std::string cellName = (llvm::Twine(slotKey) + "." + attrName).str();
    py::GlobalSetOp::create(builder, loc(*statement),
                            builder.getStringAttr(cellName), coerced.value);
    // ⭐ THE CLASS BODY IS A SCOPE, and this is where its names live. CPython
    // executes the body with each assignment visible to the ones after it:
    //
    //     class Layout:
    //         width = 80
    //         half = width // 2
    //
    // and without the binding `width` resolved to nothing, so the attribute
    // fell off the slot channel onto the constant channel -- which has no arm
    // for an expression and says so from the LOWERING. The scope guard the
    // caller holds is what keeps these names out of the module below.
    values[attrName] = coerced;
    types.bindSymbol(attrName, coerced.type);
  }
}

// ⛔ KNOWN DEFECT, now DIAGNOSED rather than silent: which body a method call
// reaches is decided from the STATIC receiver type, so an override behind a
// base-typed reference cannot be resolved.
//
//     class A:
//         def name(self) -> str: return "A"
//     class B(A):
//         def name(self) -> str: return "B"
//     def show(a: A) -> None: print(a.name())
//     show(B())
//
// Also `a: A = B()`, and `list[A]` holding a `B`. A call on `B` itself, or on
// a base no subclass overrides, is correct.
//
// ⭐ It used to PRINT "A" with nothing diagnosed. It now refuses with "'name'
// is overridden by a subclass of 'A', so this call cannot be resolved from
// the static type of the receiver" (re-measured 2026-08-14). That is the
// project's rule applied -- a shape that cannot be resolved statically is
// rejected at the earliest static boundary -- so what is left here is missing
// SURFACE (dynamic dispatch), not a wrong answer.
//
// Why the refusal took three attempts to land: refusing whenever the
// receiver's class has an overriding subclass, which is statically decidable
// from `classMros` and `classMethodBindings`, also refused `class_mro` and
// `class_super` every time.
//
// A base method's own body calls `self.who()` with `self` typed as the base,
// and that call IS resolved correctly -- `emitInlineMethodBody` specialises
// the body into each concrete call site, so the inlined copy binds `who` to
// the receiver's real class. A specialisable `self` and an unresolvable
// base-typed parameter are the same type at the binding.
//
// Two ways to tell them apart were measured and neither is enough:
//
//   - gate on `methodsBeingInlined` being empty, so only calls outside an
//     inlining are refused. Still refuses both goldens.
//   - additionally exempt a receiver spelled `super()`, since `class_super`
//     chains `Left.tag` -> `Right.tag` through the MRO rather than through a
//     type. Still refuses both: by the time the call is emitted the `super()`
//     receiver no longer looks like a `Call` node to the AST test.
//
// So the distinguishing fact is not in the receiver's syntax or in a depth
// counter. It is whether the receiver VALUE has a known dynamic class at this
// site, which is what the inliner knows and does not record.
std::optional<MethodBinding>
ModuleEmitter::lookupClassMethod(mlir::Type receiverType,
                                 llvm::StringRef methodName) const {
  if (auto typeObject = mlir::dyn_cast_if_present<py::TypeType>(receiverType))
    receiverType = typeObject.getInstanceType();
  auto contract = mlir::dyn_cast_if_present<py::ContractType>(receiverType);
  if (!contract)
    return std::nullopt;
  return resolveMroMethod(contract.getContractName(), methodName);
}

std::optional<mlir::Type>
ModuleEmitter::lookupClassField(mlir::Type receiverType,
                                llvm::StringRef fieldName) const {
  if (auto unionType = mlir::dyn_cast_if_present<py::UnionType>(receiverType)) {
    mlir::Type common;
    for (mlir::Type member : unionType.getMemberTypes()) {
      std::optional<mlir::Type> field = lookupClassField(member, fieldName);
      if (!field)
        return std::nullopt;
      if (!common) {
        common = *field;
        continue;
      }
      if (common != *field)
        return std::nullopt;
    }
    return common ? std::optional<mlir::Type>(common) : std::nullopt;
  }
  auto contract = mlir::dyn_cast_if_present<py::ContractType>(receiverType);
  if (!contract)
    return std::nullopt;
  auto classFields = classFieldBindings.find(contract.getContractName());
  if (classFields == classFieldBindings.end())
    return std::nullopt;
  auto field = classFields->second.find(fieldName);
  if (field == classFields->second.end())
    return std::nullopt;
  return field->second;
}

std::optional<mlir::Type>
ModuleEmitter::lookupClassStaticAttr(mlir::Type receiverType,
                                     llvm::StringRef attrName) const {
  if (auto typeObject = mlir::dyn_cast_if_present<py::TypeType>(receiverType))
    receiverType = typeObject.getInstanceType();
  auto contract = mlir::dyn_cast_if_present<py::ContractType>(receiverType);
  if (!contract)
    return std::nullopt;
  auto classAttrs = classStaticAttrBindings.find(contract.getContractName());
  if (classAttrs == classStaticAttrBindings.end())
    return std::nullopt;
  auto attr = classAttrs->second.find(attrName);
  if (attr == classAttrs->second.end())
    return std::nullopt;
  return attr->second;
}

bool ModuleEmitter::methodBindingBindsReceiver(
    const MethodBinding &method) const {
  return method.kind == "instance" || method.kind == "class" ||
         method.kind == "classmethod";
}

Value ModuleEmitter::emitDescriptorReceiver(const parser::Node &anchor,
                                            Value receiver,
                                            const MethodBinding &method) {
  if (method.kind != "class" && method.kind != "classmethod")
    return receiver;
  if (mlir::isa<py::TypeType>(receiver.type))
    return receiver;
  mlir::Type classType = types.typeObject(receiver.type);
  auto classObject =
      py::TypeObjectOp::create(builder, loc(anchor), classType, receiver.type);
  return {classObject.getResult(), classType};
}

bool ModuleEmitter::registerGenericClass(
    const parser::Node &classDef, llvm::StringRef symbolBase,
    const EmitOptions::SourceModule *source) {
  const auto *typeParams = ast::nodeList(classDef, "type_params");
  if (!typeParams || typeParams->empty())
    return false;
  GenericClassInfo &info = genericClasses[symbolBase];
  info.node = &classDef;
  info.symbolBase = symbolBase.str();
  info.source = source;
  info.params.clear();
  info.hasPackParameter = false;
  for (const parser::NodePtr &param : *typeParams) {
    if (!param)
      continue;
    auto name = ast::string(*param, "name");
    if (!name)
      continue;
    info.params.push_back(std::string(*name));
    // A ParamSpec or TypeVarTuple parameter is a parameter-LIST unknown: the
    // instantiation would have to mangle a shape, not a type, and the method
    // bodies would need pack-aware emission. Rejected at the instantiation
    // site rather than here, so a module may ship such a class unused.
    if (param->kind == "ParamSpec" || param->kind == "TypeVarTuple")
      info.hasPackParameter = true;
  }
  const parser::Node *initNode = nullptr;
  llvm::SmallVector<TypeSystem::GenericClassField, 8> fields;
  if (const auto *body = ast::nodeList(classDef, "body"))
    for (const parser::NodePtr &statement : *body) {
      if (!statement)
        continue;
      if (statement->kind == "FunctionDef" &&
          ast::string(*statement, "name") == "__init__")
        initNode = statement.get();
      // Annotated class-body names are the positional parameters of the
      // constructor a dataclass/NamedTuple synthesizes, and the only place
      // such a class's type arguments appear.
      if (statement->kind == "AnnAssign") {
        const parser::Node *target = ast::node(*statement, "target");
        const parser::Node *annotation = ast::node(*statement, "annotation");
        if (target && target->kind == "Name" && annotation)
          fields.emplace_back(std::string(ast::nameSpelling(*target)),
                              annotation);
      }
    }
  types.registerGenericClass(symbolBase, info.params, initNode, fields);
  return true;
}

ModuleEmitter::GenericClassInfo *
ModuleEmitter::lookupGenericClass(llvm::StringRef name) {
  auto found = genericClasses.find(name);
  return found == genericClasses.end() ? nullptr : &found->second;
}

void ModuleEmitter::diagnoseUngroundedGenericClass(const parser::Node &anchor,
                                                   llvm::StringRef name) {
  GenericClassInfo *generic = lookupGenericClass(name);
  std::string spelling = name.str();
  if (generic) {
    spelling = generic->symbolBase;
    std::string arguments;
    for (const std::string &param : generic->params) {
      if (!arguments.empty())
        arguments += ", ";
      arguments += param;
    }
    spelling += "[" + arguments + "]";
  }
  if (generic && generic->hasPackParameter) {
    // A pack parameter is a parameter-LIST unknown, so no instantiation
    // determines a fixed field layout or method arity to specialize.
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, anchor.range.start,
        "generic class '" + spelling +
            "' uses ParamSpec or TypeVarTuple parameters, which cannot be "
            "specialized yet"});
    return;
  }
  diagnostics.push_back(parser::Diagnostic{
      parser::Severity::Error, anchor.range.start,
      "generic class '" + spelling +
          "' requires explicit type arguments here, or an annotated context "
          "that determines them"});
}

std::optional<Value>
ModuleEmitter::rejectGenericClassObject(const parser::Node &anchor,
                                        mlir::Type classType) {
  auto contract = mlir::dyn_cast_if_present<py::ContractType>(classType);
  if (!contract || !contract.getArguments().empty())
    return std::nullopt;
  GenericClassInfo *generic = lookupGenericClass(contract.getContractName());
  if (!generic)
    return std::nullopt;
  std::string arguments;
  for (const std::string &param : generic->params) {
    if (!arguments.empty())
      arguments += ", ";
    arguments += param;
  }
  diagnostics.push_back(parser::Diagnostic{
      parser::Severity::Error, anchor.range.start,
      "generic class '" + generic->symbolBase + "[" + arguments +
          "]' has no class object of its own: each instantiation is a separate "
          "class, so name one here (" +
          generic->symbolBase + "[" + arguments + "])"});
  return emitNone(anchor);
}

mlir::Type
ModuleEmitter::inferredGenericClassInstantiation(const parser::Node &call) {
  // inferExpr's own class-instantiation path already solves the type
  // arguments from __init__; validating the answer here keeps the emitter
  // from constructing a contract the inference did not actually derive from
  // this callee.
  return expectedGenericClassInstantiation(call, types.inferExpr(&call));
}

mlir::Type ModuleEmitter::expectedGenericClassInstantiation(
    const parser::Node &call, mlir::Type expected) {
  auto expectedContract = mlir::dyn_cast_if_present<py::ContractType>(expected);
  if (!expectedContract || !expectedContract.getArguments().empty())
    return {};
  const parser::Node *callee = ast::node(call, "func");
  if (!callee || (callee->kind != "Name" && callee->kind != "Attribute"))
    return {};
  std::string qualified = ast::qualifiedName(callee);
  std::string_view spelling = ast::nameSpelling(*callee);
  std::optional<mlir::Type> cls = types.lookupClass(
      qualified.empty() ? std::string(spelling) : qualified);
  auto baseContract =
      cls ? mlir::dyn_cast_if_present<py::ContractType>(*cls) : py::ContractType();
  if (!baseContract)
    return {};
  GenericClassInfo *generic =
      lookupGenericClass(baseContract.getContractName());
  if (!generic)
    return {};
  // Accepted only when the expectation IS one of this class's
  // specializations. A different class that happens to be assignable would
  // silently construct the wrong type.
  for (const auto &specialization : generic->specializations)
    if (specialization.second == expectedContract.getContractName())
      return expected;
  return {};
}

mlir::Type ModuleEmitter::ensureGenericClassSpecialization(
    llvm::StringRef baseName, mlir::ArrayRef<mlir::Type> arguments) {
  GenericClassInfo *generic = lookupGenericClass(baseName);
  if (!generic || !generic->node)
    return {};
  // Wrong arity and pack parameters resolve to nothing rather than a
  // diagnostic: this runs from const type queries (including speculative
  // ones), so the caller keeps its parameterized reading and the use site
  // reports the failure with its own location.
  if (generic->hasPackParameter ||
      arguments.size() != generic->params.size())
    return {};
  // ⛔ Nor at an argument that still names a type parameter: `Ctx[E]` in a
  // generic method's body is typed before the method is specialized, and a
  // class emitted at the unsolved E was refused ("class value has no type
  // object for builtins.E") by programs that never called the method.
  if (llvm::any_of(arguments, [](mlir::Type argument) {
        return unboundStaticParameterCount(argument) != 0;
      }))
    return {};

  mlir::Type key = types.contract(baseName, arguments);
  auto memoized = generic->specializations.find(key);
  if (memoized != generic->specializations.end())
    return types.contract(memoized->second);
  // Divergence backstop, the same one the function specializer uses: a
  // method body that instantiates its own class at a new ground type
  // (`Box[list[T]]`) re-enters before the previous body finished.
  if (generic->specializations.size() >= 32)
    return {};

  std::string symbol =
      (llvm::Twine(generic->symbolBase) + "$spec" +
       llvm::Twine(static_cast<unsigned>(generic->specializations.size())))
          .str();
  // Register BEFORE emitting: a self-reference inside the class body
  // (`def copy(self) -> Box[T]`, whose T is already this instantiation's
  // argument) must resolve to this same contract instead of allocating a
  // second specialization and recursing forever.
  generic->specializations[key] = symbol;
  // ⛔ NOT while a module's names are being bound for an emission in its
  // scope: binding computes each function's signature, a coroutine's walks
  // its body, and a class specialized there emitted its methods before the
  // module's own functions were bound -- `helper()` in `Box.__init__` was an
  // unresolved name when `async def co[T]` in the same module made a `Box[T]`.
  if (!genericClassEmissionReady || definingScopeSetupDepth != 0) {
    pendingClassSpecializations.push_back(PendingClassSpecialization{
        std::string(baseName), symbol,
        llvm::SmallVector<mlir::Type, 4>(arguments.begin(), arguments.end())});
    return types.contract(symbol);
  }
  emitGenericClassSpecialization(*generic, symbol, arguments);
  return types.contract(symbol);
}

void ModuleEmitter::bindClassTypeArguments(llvm::StringRef className) {
  auto found = classTypeArguments.find(className);
  if (found == classTypeArguments.end())
    return;
  for (const auto &[param, argument] : found->second) {
    types.bindLocalSymbol(param, argument);
    types.bindLocalTypeParameter(param, argument);
  }
}

void ModuleEmitter::emitGenericClassSpecialization(
    GenericClassInfo &generic, llvm::StringRef symbol,
    mlir::ArrayRef<mlir::Type> arguments) {
  llvm::SmallVector<std::pair<std::string, mlir::Type>, 4> &solved =
      classTypeArguments[symbol];
  solved.clear();
  for (auto [param, argument] : llvm::zip_equal(generic.params, arguments))
    solved.emplace_back(param, argument);
  auto emitSpecializedClass = [&] {
    // Field annotations, method signatures and method bodies all spell the
    // type parameters by name (`value: T`); binding each to its ground type
    // for the whole class emission is what makes the specialized contract
    // ground. bindLocalTypeParameter is the channel annotations read —
    // annotationTypeForName deliberately ignores value symbols so a local
    // cannot shadow a class annotation.
    auto scope = types.pushScope();
    bindClassTypeArguments(symbol);
    emitClassContract(*generic.node, symbol);
  };
  if (generic.source)
    emitInDefiningModuleScope(*generic.source, emitSpecializedClass);
  else
    emitSpecializedClass();
}

void ModuleEmitter::drainGenericClassSpecializations(llvm::StringRef onlyBase) {
  // Emitting one specialization may spell further ones, so this rescans
  // instead of iterating a snapshot.
  for (bool progressed = true; progressed;) {
    progressed = false;
    for (auto it = pendingClassSpecializations.begin(),
              e = pendingClassSpecializations.end();
         it != e; ++it) {
      if (!onlyBase.empty() && it->base != onlyBase)
        continue;
      PendingClassSpecialization pending = std::move(*it);
      pendingClassSpecializations.erase(it);
      if (GenericClassInfo *generic = lookupGenericClass(pending.base))
        emitGenericClassSpecialization(*generic, pending.symbol,
                                       pending.arguments);
      progressed = true;
      break;
    }
  }
}

void ModuleEmitter::emitClassContract(const parser::Node &classDef,
                                      llvm::StringRef symbolName,
                                      const EmitOptions::SourceModule *source) {
  auto name = ast::string(classDef, "name");
  if (!name)
    return;
  std::string classSymbol =
      symbolName.empty() ? std::string(*name) : symbolName.str();
  llvm::StringRef contractName(classSymbol);
  checkDecorators(classDef, DecoratorRole::Class);
  for (DecoratedMethod &method : decoratedMethods) {
    if (method.classDef != &classDef)
      continue;
    // An imported module's class: typed here, in its module's scope, and
    // applied with that module's other globals, before the importer's
    // first statement -- where CPython runs the class statement.
    bool typedBefore = method.typed;
    if (typeDecoratedMethod(method, /*reportFailure=*/true) && method.source &&
        !typedBefore) {
      parser::NodePtr applied = decoratedMethodApplication(
          method, synth::attribute(synth::name(method.className, method.range),
                                   method.hiddenName, method.range));
      importedModuleGlobalInits.push_back(PendingModuleGlobalInit{
          applied.get(), method.globalName, method.source});
      synthesizedIteratorDefs.push_back(std::move(applied));
    }
  }

  // @dataclass: init/repr/eq synthesize below (default True); frozen/order
  // are accepted only as explicit False.
  bool isDataclass = false;
  // A NamedTuple is hashable -- it inherits tuple's __hash__ -- while a plain
  // dataclass is not (CPython sets __hash__ to None when eq is synthesized and
  // frozen is not). Only the first gets one below.
  bool isNamedTuple = false;
  bool dataclassOrder = false;
  bool dataclassFrozen = false;
  bool dataclassInit = true;
  bool dataclassRepr = true;
  bool dataclassEq = true;
  if (const auto *decorators = ast::nodeList(classDef, "decorator_list")) {
    for (const parser::NodePtr &decorator : *decorators) {
      if (!decorator)
        continue;
      auto [callee, spelling] = decoratorCallee(*decorator);
      if (decoratorLeafName(spelling) != "dataclass")
        continue;
      isDataclass = true;
      if (decorator->kind != "Call")
        continue;
      if (const auto *args = ast::nodeList(*decorator, "args"))
        if (!args->empty())
          diagnostics.push_back(parser::Diagnostic{
              parser::Severity::Error, decorator->range.start,
              "dataclass() takes keyword arguments only"});
      if (const auto *keywords = ast::nodeList(*decorator, "keywords")) {
        for (const parser::NodePtr &keyword : *keywords) {
          if (!keyword)
            continue;
          auto keywordName = ast::string(*keyword, "arg");
          const parser::Node *value = ast::node(*keyword, "value");
          std::optional<bool> flag =
              value ? ast::boolean(*value, "value") : std::nullopt;
          if (!keywordName || !flag) {
            diagnostics.push_back(parser::Diagnostic{
                parser::Severity::Error, keyword->range.start,
                "dataclass() arguments must be literal True/False keywords"});
            continue;
          }
          if (*keywordName == "init")
            dataclassInit = *flag;
          else if (*keywordName == "repr")
            dataclassRepr = *flag;
          else if (*keywordName == "eq")
            dataclassEq = *flag;
          else if (*keywordName == "order")
            dataclassOrder = *flag;
          else if (*keywordName == "frozen")
            dataclassFrozen = *flag;
          else
            diagnostics.push_back(parser::Diagnostic{
                parser::Severity::Error, keyword->range.start,
                "dataclass argument '" + std::string(*keywordName) +
                    "' is not supported"});
        }
      }
    }
  }

  llvm::SmallVector<llvm::StringRef, 4> bases;
  if (const auto *baseNodes = ast::nodeList(classDef, "bases")) {
    for (const parser::NodePtr &base : *baseNodes) {
      if (!base)
        continue;
      // `class C(Box[int])` inherits from the INSTANTIATION, which is a
      // specialized class contract of its own — so the base list carries that
      // contract name and the C3 merge below linearizes ground classes only.
      // Inside `class Labeled[T](Box[T])` the parameter is already bound to
      // this specialization's argument, so `Box[T]` resolves the same way.
      if (mlir::Type instantiated = types.genericClassSubscript(base.get())) {
        bases.push_back(
            builder
                .getStringAttr(
                    mlir::cast<py::ContractType>(instantiated).getContractName())
                .getValue());
        continue;
      }
      std::string qualified = ast::qualifiedName(base.get());
      // typing.NamedTuple is a class-construction marker, not a base: the
      // annotated body it requires is exactly the dataclass field form, and
      // CPython's namedtuple __init__/__repr__/__eq__ agree with the
      // dataclass synthesis field for field (including the repr's
      // `Name(f=v, ...)` spelling). It is consumed here rather than
      // linearized.
      llvm::StringRef spelling =
          qualified.empty() ? llvm::StringRef(ast::nameSpelling(*base))
                            : llvm::StringRef(qualified);
      if (decoratorLeafName(spelling) == "NamedTuple") {
        isDataclass = true;
        isNamedTuple = true;
        continue;
      }
      if (!qualified.empty()) {
        bases.push_back(builder.getStringAttr(qualified).getValue());
        continue;
      }
      bases.push_back(ast::nameSpelling(*base));
    }
  }

  // C3 linearization over canonical contract names. Bases must already be
  // linearized (Python requires bases defined before the class statement);
  // manifest bases contribute their builtin-exception chains so mixed-base
  // merges see the shared ancestors.
  llvm::SmallVector<std::string, 4> canonicalBases;
  for (llvm::StringRef base : bases) {
    std::string canonical = canonicalClassName(base);
    if (llvm::is_contained(canonicalBases, canonical)) {
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, classDef.range.start,
          "duplicate base class '" + base.str() + "'"});
      return;
    }
    canonicalBases.push_back(std::move(canonical));
  }
  llvm::SmallVector<std::string, 8> mro;
  mro.push_back(contractName.str());
  {
    llvm::SmallVector<llvm::SmallVector<std::string, 8>, 8> sequences;
    for (const std::string &base : canonicalBases) {
      auto baseMro = classMros.find(base);
      if (baseMro != classMros.end()) {
        sequences.emplace_back(baseMro->second.begin(), baseMro->second.end());
        continue;
      }
      sequences.push_back(manifestLinearization(base));
    }
    sequences.emplace_back(canonicalBases.begin(), canonicalBases.end());
    std::optional<llvm::SmallVector<std::string, 8>> merged =
        lython::common::c3Merge<std::string>(std::move(sequences));
    if (!merged) {
      std::string baseList;
      for (const std::string &base : canonicalBases) {
        if (!baseList.empty())
          baseList += ", ";
        baseList += py::contracts::displayClassNameForContract(base);
      }
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, classDef.range.start,
          "Cannot create a consistent method resolution order (MRO) for "
          "bases " +
              baseList});
      return;
    }
    mro.append(merged->begin(), merged->end());
    if (mro.back() != "builtins.object")
      mro.push_back("builtins.object");
  }
  classBaseNames[contractName] = canonicalBases;
  classMros[contractName] = mro;

  // ⭐ THE CLASS ATTRIBUTES ARE TYPED BEFORE THE FIELDS, because a field's
  // initializer may read one:
  //
  //     class Config:
  //         DEFAULTS = {"a": 1}
  //         def __init__(self) -> None:
  //             self.values = dict(Config.DEFAULTS)
  //         def get(self, k: str) -> int:
  //             return self.values[k]
  //     # builtins.object does not provide manifest method '__getitem__'
  //
  // The field walk types `dict(Config.DEFAULTS)` with `inferExpr`, and with the
  // attributes registered only further down, `Config.DEFAULTS` answered
  // `object` -- so the FIELD became object and its reads carried nothing. The
  // same line reading ANOTHER class's attribute compiled, which is what said
  // the SELF reference was the defect.
  //
  // ⛔ Collected ONCE and carried. This walk emits diagnostics of its own
  // ("requires a statically inferred type"), so calling it a second time
  // further down would report each of them twice.
  llvm::SmallVector<std::string, 8> staticAttrNames;
  llvm::SmallVector<mlir::Attribute, 8> staticAttrValues;
  llvm::SmallVector<mlir::Type, 8> staticAttrTypes;
  collectStaticClassAssignments(classDef, staticAttrNames, staticAttrValues,
                                &staticAttrTypes);
  for (auto [attrName, attrType] :
       llvm::zip_equal(staticAttrNames, staticAttrTypes))
    if (attrType)
      types.bindClassStaticAttr(contractName, attrName,
                                types.widenLiteral(attrType));

  llvm::SmallVector<std::string, 8> fieldNames;
  llvm::SmallVector<mlir::Type, 8> fieldTypes;
  // ⭐ EACH FIELD IS VISIBLE TO THE ONES AFTER IT. A field's initializer may
  // read an earlier field:
  //
  //     self.parts = list(parts)
  //     self.frozen = tuple(self.parts)
  //
  // and `self.parts` resolves through the PROTOCOL TABLE, which learned this
  // class's fields only after the whole walk -- so `frozen` typed as `object`
  // and `"".join(self.frozen)` was "cannot adapt builtins.object". The same
  // reason the note below gives for publishing before the method signatures,
  // one step earlier.
  auto publishFieldsSoFar = [&](llvm::ArrayRef<std::string> names,
                                llvm::ArrayRef<mlir::Type> fieldTypesSoFar) {
    py::protocols::ProtocolInfo partial;
    for (const std::string &base : canonicalBases)
      partial.bases.push_back(py::protocols::ProtocolBase{
          py::contracts::manifestClassNameForContract(base), {}});
    partial.bases.push_back(py::protocols::ProtocolBase{
        py::contracts::manifestClassNameForContract("builtins.object"), {}});
    for (auto [fieldName, fieldType] : llvm::zip_equal(names, fieldTypesSoFar))
      if (fieldType)
        partial.fields[fieldName] = fieldType;
    py::protocols::Table::getMutable(context).registerClass(contractName,
                                                            partial);
  };
  classFieldsDeclaredOutsideInit.erase(contractName);
  // A NamedTuple's instances are immutable: a store in a method raises in
  // CPython, and must not become a slot of the tuple.
  collectClassFields(classDef, fieldNames, fieldTypes,
                     /*includeAnnAssignDefaults=*/isDataclass, contractName,
                     publishFieldsSoFar,
                     /*methodsDeclareFields=*/!isNamedTuple);
  // ⭐ A CLASS ATTRIBUTE AN INSTANCE MAY SHADOW IS ALSO A FIELD, declared here
  // on the class that binds it so that every instance of its line has the
  // slot. `self.maxDiff = None` in a test's setUp -- the commonest line in a
  // unittest fixture -- was refused ("would shadow the class attribute"),
  // because a method other than `__init__` may not declare a field a class
  // attribute already answers: that would be a second storage for one name.
  // It is one, and CPython has both: the instance's when stored, the class's
  // otherwise. The field starts unset, and a read that cannot prove a store
  // asks which (emitAttribute).
  // ⛔ Only a name some class of the line stores on `self` (the declaration
  // pre-pass, which has seen every module's classes by now), or that any
  // statement of the program stores through another name (`b.label = "b"`),
  // whose receiver's class is not known before the types are: a class nobody
  // shadows keeps its layout, and one that only looks shadowed gets a slot
  // that stays unset.
  // ⛔ Not a dataclass or NamedTuple, whose annotated names are fields
  // already; not an exception, whose fields are kept in its header; not a
  // callable attribute, which `self.f = ...` would make a bound-method
  // question.
  if (!isDataclass && !isNamedTuple && !isExceptionBackedClass(contractName))
    for (auto [attrName, attrType] :
         llvm::zip_equal(staticAttrNames, staticAttrTypes)) {
      mlir::Type widened = attrType ? types.widenLiteral(attrType) : attrType;
      if (!widened || mlir::isa<py::CallableType>(widened) ||
          widened == types.none() || llvm::is_contained(fieldNames, attrName) ||
          llvm::StringRef(attrName).starts_with("__"))
        continue;
      // ⛔ Not again under a subclass that redeclares the attribute: the
      // base's field is the slot, with the type the base declared -- a
      // subclass's `maxDiff = 10` retyped it `int` under the base's own
      // `self.maxDiff = None`.
      bool inherited = false;
      for (const std::string &ancestor :
           llvm::ArrayRef<std::string>(mro).drop_front())
        if (auto fields = classFieldBindings.find(ancestor);
            fields != classFieldBindings.end() &&
            fields->second.count(attrName))
          inherited = true;
      if (inherited)
        continue;
      bool shadowed = programAttributeStoreNames().contains(attrName);
      for (const auto &entry : declaredClassStoredFields)
        if (entry.second.contains(attrName) &&
            (entry.getKey() == contractName ||
             programClassDerivesFrom(entry.getKey(), contractName)))
          shadowed = true;
      if (!shadowed)
        continue;
      fieldNames.push_back(attrName);
      fieldTypes.push_back(widened);
    }

  // Dataclass field defaults (AnnAssign initializers). dataclasses.field()
  // models default= and default_factory= only: the factory desugars to a
  // synthesized zero-argument call node so the __init__ default machinery's
  // provider path re-evaluates it per omitted argument (the factory
  // contract); the other field() knobs steer runtime introspection we do
  // not model, so they are rejected loudly rather than ignored.
  llvm::StringMap<parser::NodePtr> &fieldDefaults =
      classFieldDefaultNodes[contractName];
  fieldDefaults.clear();
  if (isDataclass) {
    if (const auto *body = ast::nodeList(classDef, "body")) {
      for (const parser::NodePtr &statement : *body) {
        if (!statement || statement->kind != "AnnAssign")
          continue;
        const parser::Node *target = ast::node(*statement, "target");
        const parser::Field *valueField =
            target ? parser::findField(*statement, "value") : nullptr;
        if (!target || target->kind != "Name" || !valueField ||
            !std::holds_alternative<parser::NodePtr>(valueField->value))
          continue;
        parser::NodePtr value = std::get<parser::NodePtr>(valueField->value);
        if (!value)
          continue;
        if (value->kind == "Call") {
          auto [callee, spelling] = decoratorCallee(*value);
          if (decoratorLeafName(spelling) == "field") {
            parser::NodePtr defaultNode;
            parser::NodePtr factoryNode;
            bool unsupported = false;
            if (const auto *args = ast::nodeList(*value, "args");
                args && !args->empty()) {
              diagnostics.push_back(parser::Diagnostic{
                  parser::Severity::Error, value->range.start,
                  "dataclasses.field() takes keyword arguments only"});
              unsupported = true;
            }
            if (const auto *keywords = ast::nodeList(*value, "keywords")) {
              for (const parser::NodePtr &keyword : *keywords) {
                if (!keyword)
                  continue;
                auto keywordName = ast::string(*keyword, "arg");
                const parser::Field *kwField =
                    parser::findField(*keyword, "value");
                parser::NodePtr kwNode;
                if (kwField &&
                    std::holds_alternative<parser::NodePtr>(kwField->value))
                  kwNode = std::get<parser::NodePtr>(kwField->value);
                if (!keywordName || !kwNode) {
                  diagnostics.push_back(parser::Diagnostic{
                      parser::Severity::Error, keyword->range.start,
                      "unsupported dataclasses.field() argument form"});
                  unsupported = true;
                  continue;
                }
                if (*keywordName == "default") {
                  defaultNode = kwNode;
                } else if (*keywordName == "default_factory") {
                  factoryNode = kwNode;
                } else {
                  diagnostics.push_back(parser::Diagnostic{
                      parser::Severity::Error, keyword->range.start,
                      "dataclasses.field() argument '" +
                          std::string(*keywordName) +
                          "' is not supported (only default and "
                          "default_factory are modeled)"});
                  unsupported = true;
                }
              }
            }
            if (unsupported)
              continue;
            if (defaultNode && factoryNode) {
              // CPython raises ValueError at class creation time; the
              // static boundary is class emission, so it is a diagnostic
              // here with the same message.
              diagnostics.push_back(parser::Diagnostic{
                  parser::Severity::Error, value->range.start,
                  "cannot specify both default and default_factory"});
              continue;
            }
            if (factoryNode) {
              parser::NodePtr factoryCall = synth::call(factoryNode, std::vector<parser::NodePtr>{}, value->range);
              fieldDefaults[ast::nameSpelling(*target)] =
                  std::move(factoryCall);
            } else if (defaultNode) {
              fieldDefaults[ast::nameSpelling(*target)] = defaultNode;
            }
            // field() with neither keyword leaves the field required,
            // matching CPython.
            continue;
          }
        }
        fieldDefaults[ast::nameSpelling(*target)] = value;
      }
    }
  }

  // Manifest fields_spec (ly.typing.fields_spec, e.g. ctypes
  // Structure/Union): a class assignment named by the spec declares the
  // aggregate's fields; each field reads/writes as its declared class's
  // via_base value type (falling back to the declared class itself for
  // nested aggregates). Registering them as ordinary class fields also
  // gives the subclass its positional field constructor.
  {
    const py::protocols::Table &table = py::protocols::Table::get(context);
    std::optional<std::pair<std::string, std::string>> spec;
    for (llvm::StringRef base : bases)
      if ((spec = table.aggregateFieldsSpec(base)))
        break;
    const auto *body = spec ? ast::nodeList(classDef, "body") : nullptr;
    if (body)
      for (const parser::NodePtr &statement : *body) {
        if (!statement || statement->kind != "Assign")
          continue;
        const auto *targets = ast::nodeList(*statement, "targets");
        if (!targets || targets->size() != 1 || !targets->front() ||
            targets->front()->kind != "Name" ||
            ast::nameSpelling(*targets->front()) != spec->first)
          continue;
        const parser::Node *value = ast::node(*statement, "value");
        const auto *entries = value ? ast::nodeList(*value, "elts") : nullptr;
        if (!entries)
          continue;
        for (const parser::NodePtr &entry : *entries) {
          if (!entry)
            continue;
          const auto *pair = ast::nodeList(*entry, "elts");
          if (!pair || pair->size() != 2 || !(*pair)[0] || !(*pair)[1])
            continue;
          auto fieldName = ast::string(*(*pair)[0], "value");
          if (!fieldName)
            continue;
          mlir::Type declared = types.inferExpr((*pair)[1].get());
          if (auto typeObject =
                  mlir::dyn_cast_if_present<py::TypeType>(declared)) {
            fieldNames.push_back(std::string(*fieldName));
            fieldTypes.push_back(
                table
                    .conversionTypeViaBase(typeObject.getInstanceType(),
                                           spec->second)
                    .value_or(typeObject.getInstanceType()));
          }
        }
      }
  }

  // Instance layout composes the MRO's per-class field declarations with the
  // base chain first (a derived object's value list extends its bases'
  // prefix), deduplicated by name -- a subclass redeclaration refines the
  // type but keeps the base slot position.
  classOwnFieldOrders[contractName].assign(fieldNames.begin(),
                                           fieldNames.end());
  {
    llvm::SmallVector<std::string, 8> mergedNames;
    llvm::SmallVector<mlir::Type, 8> mergedTypes;
    auto appendField = [&](llvm::StringRef name, mlir::Type type) {
      for (auto [index, existing] : llvm::enumerate(mergedNames)) {
        if (existing == name) {
          mergedTypes[index] = type;
          return;
        }
      }
      mergedNames.push_back(name.str());
      mergedTypes.push_back(type);
    };
    llvm::ArrayRef<std::string> linearization = classMros[contractName];
    for (const std::string &cls : llvm::reverse(linearization)) {
      if (cls == contractName)
        continue;
      auto ownOrder = classOwnFieldOrders.find(cls);
      auto ownTypes = classFieldBindings.find(cls);
      if (ownOrder == classOwnFieldOrders.end() ||
          ownTypes == classFieldBindings.end())
        continue;
      for (const std::string &name : ownOrder->second) {
        auto type = ownTypes->second.find(name);
        if (type != ownTypes->second.end())
          appendField(name, type->second);
      }
    }
    for (auto [fieldName, fieldType] : llvm::zip_equal(fieldNames, fieldTypes))
      appendField(fieldName, fieldType);
    fieldNames = std::move(mergedNames);
    fieldTypes = std::move(mergedTypes);
  }

  // Exception-backed classes use the runtime exception object (header with
  // this class's id + message), so their fields get NO object lanes: the
  // lowering stores them in the extended header's field block, indexed by the
  // same field order registered here.
  llvm::StringMap<mlir::Type> &registeredFields =
      classFieldBindings[contractName];
  registeredFields.clear();
  llvm::SmallVector<std::string, 8> &registeredOrder =
      classFieldOrders[contractName];
  registeredOrder.assign(fieldNames.begin(), fieldNames.end());
  for (auto [fieldName, fieldType] : llvm::zip_equal(fieldNames, fieldTypes))
    registeredFields[fieldName] = fieldType;

  // ⭐ The protocol table learns this class's FIELDS here, not at the bottom
  // of this function with its methods: a method's signature is inferred
  // below, and that inference walks the body, where `self.n` has to resolve
  // against these fields. It could not, so every unannotated method that
  // read a field inferred builtins.object -- `def peek(self): return self.n`
  // then failed at the call site with "builtins.object does not provide
  // manifest method '__add__'". The full registration below overwrites this
  // entry; only the fields are needed this early, and they are already
  // merged with the bases' at this point.
  py::protocols::ProtocolInfo progressiveInfo;
  for (const std::string &base : canonicalBases)
    progressiveInfo.bases.push_back(py::protocols::ProtocolBase{
        py::contracts::manifestClassNameForContract(base), {}});
  progressiveInfo.bases.push_back(py::protocols::ProtocolBase{
      py::contracts::manifestClassNameForContract("builtins.object"), {}});
  for (auto [fieldName, fieldType] : llvm::zip_equal(fieldNames, fieldTypes))
    progressiveInfo.fields[fieldName] = fieldType;
  auto publishProgress = [&] {
    py::protocols::Table::getMutable(context).registerClass(contractName,
                                                            progressiveInfo);
  };
  publishProgress();

  // Class attributes register BEFORE any method body is emitted: method
  // bodies read them (Counter.count += 1) through the very lookups being
  // registered here.
  // Mutable class attributes: attributes of main-module classes whose
  // widened type has module-global cell storage become slot-backed (reads
  // and writes go through the cells; the initializer expression is no
  // longer restricted to constants). Container-typed attributes stay on the
  // constant channel: their storage cells would go stale against
  // reallocation, the same reason collectModuleGlobals excludes them.
  // ⭐ AN IMPORTED CLASS GETS SLOTS TOO. The gate used to be "main-module
  // only", so an imported class's attributes stayed on the constant channel --
  // which re-materializes a value per read and has no arm for a container or
  // for a WRITE. `class Registry: seen: list[str] = []` in a library was
  // "unsupported static class attribute expression for 'seen'" out of the
  // LOWERING, and `Counter.count += 1` the same, both for shapes that work
  // written in one file. A cell is what CPython's class dict is: one storage,
  // shared, mutable.
  //
  // ⛔ A slot only where the INITIALIZER will run. A generic specialization
  // carries a symbolName too and has no position in module flow at all, so it
  // keeps the constant channel -- giving it a cell nothing writes left
  // `Tagged$spec0.label` "referenced before assignment" at run time.
  if (symbolName.empty() || source) {
    llvm::StringMap<mlir::Type> &slots = classAttrSlots[contractName];
    slots.clear();
    for (auto [attrName, attrType] :
         llvm::zip_equal(staticAttrNames, staticAttrTypes)) {
      mlir::Type widened = types.widenLiteral(attrType);
      // ⛔ NOT A CALLABLE, and the reason is measured rather than assumed.
      // Adding it here moves nothing: `C.V(1)` for
      // `V: Callable[[int], int] = lambda n: n + 1` in a class body is still
      // "static type !py.type<...> does not provide manifest method 'V'",
      // at module scope as much as inside a method, so the attribute is not
      // reaching this walk at all -- the class-body collection is where to
      // look, not the storability rule. It is also CPython's unbound-method
      // corner (`C().V(1)` passes the instance, which is what @staticmethod
      // exists to opt out of), so the read is a shape question before it is a
      // storage one.
      auto storableAs = [&, attrName = attrName](mlir::Type widened) -> bool {
        // ⭐ A CLASS VALUE TOO, which is a reference to an immortal type
        // object -- a module global holds one already. Left on the constant
        // channel, `failureException: type[BaseException] = AssertionError`
        // could not be read at all, through the class or an instance
        // ("unsupported static class attribute expression"), and that is the
        // attribute every unittest assertion raises through.
        bool storable =
            widened == types.intType() || widened == types.strType() ||
            widened == types.floatType() || widened == types.boolType() ||
            mlir::isa_and_nonnull<py::CallableType>(widened) ||
            mlir::isa_and_nonnull<py::TypeType>(widened);
        if (!storable) {
          if (auto attrContract =
                  mlir::dyn_cast_if_present<py::ContractType>(widened)) {
            llvm::StringRef attrContractName = attrContract.getContractName();
            // ⭐ A CONTAINER IS SLOT-BACKED TOO, and the reason it was not is a
            // paragraph that stopped being true. The note above said container
            // cells "would go stale against reallocation, the same reason
            // collectModuleGlobals excludes them" -- and that exclusion is
            // gone: the runtime's growth (objects/list.mlir, objects/set.mlir)
            // writes THROUGH the handle, so the cell holds what stays valid.
            // Without this a container class attribute could not even be READ:
            //
            //     class R:
            //         items: list[str] = []
            //     print(R.items)
            //     # unsupported static class attribute expression for 'items'
            //
            // because the constant channel re-materializes the value per read
            // and has no arm for a container -- and could not have one, since
            // every read of a mutable attribute must be the SAME object.
            //
            // ⛔ EXCEPT a container whose ELEMENT type is a union, which
            // `collectModuleGlobals` still excludes for its own measured
            // reason: a cell hands back the handle, and a union-typed element
            // read needs the literal's per-element evidence.
            //
            // ⛔ And EXCEPT a `_dunder_` name. `ctypes.Structure._fields_` is a
            // list the COMPILER consumes, not a runtime value; slotting it
            // emits a module-level store, and a runtime-internal lib module may
            // not run module-level code (`stackguard_support.py` stopped
            // building).
            bool erasedElement = false;
            for (mlir::Type argument : attrContract.getArguments())
              if (mlir::isa<py::UnionType>(argument))
                erasedElement = true;
            llvm::StringRef attrSpelling(attrName);
            bool compilerConsumed = attrSpelling.size() > 2 &&
                                    attrSpelling.starts_with("_") &&
                                    attrSpelling.ends_with("_");
            storable = !erasedElement && !compilerConsumed &&
                       (attrContractName == "builtins.bytes" ||
                        attrContractName == "builtins.list" ||
                        attrContractName == "builtins.dict" ||
                        attrContractName == "builtins.set" ||
                        attrContractName == "builtins.tuple" ||
                        attrContractName == "builtins.frozenset" ||
                        // ⛔ The declaration map, not the name's shape: an
                        // IMPORTED class is `mod.Kind`, so a class attribute
                        // typed by one got no slot and the read reached the
                        // constant channel, which has no arm for an instance --
                        // "Failed to run lowering pipeline", where the same
                        // class written in one file is read fine.
                        isSourceClassContract(attrContract));
          }
        }
        return storable;
      };
      bool storable = storableAs(widened);
      // ⭐ A UNION IS SLOT-BACKED WHEN EACH MEMBER WOULD BE, None aside. The
      // constant channel has nothing to say about it: `debug: bool | None =
      // None` was read as the None CONSTANT with no union tag ("union.test
      // input has no runtime tag"), and `Config.debug = True` was "class
      // static attribute mutation is not supported" -- a settings class, as
      // CPython code writes one. The cell holds the tag and every member's
      // lanes, as a union module global's does.
      if (!storable)
        if (auto unionType = mlir::dyn_cast_if_present<py::UnionType>(widened))
          storable =
              llvm::all_of(unionType.getMemberTypes(), [&](mlir::Type member) {
                member = types.widenLiteral(member);
                return member == types.none() || storableAs(member);
              });
      if (storable)
        slots[attrName] = widened;
    }
    if (source)
      importedClassAttrInits.push_back(
          PendingClassAttrInit{&classDef, classSymbol, source});
  }
  // Inherit base class attributes MRO-forward (own declarations win): a
  // subclass reads its bases' class attributes through its own type object.
  for (const std::string &cls : llvm::ArrayRef<std::string>(mro).drop_front()) {
    auto baseOrder = classStaticAttrOrders.find(cls);
    auto baseValues = classStaticAttrValues.find(cls);
    auto baseTypes = classStaticAttrBindings.find(cls);
    if (baseOrder == classStaticAttrOrders.end() ||
        baseValues == classStaticAttrValues.end() ||
        baseTypes == classStaticAttrBindings.end())
      continue;
    for (const std::string &name : baseOrder->second) {
      if (llvm::is_contained(staticAttrNames, name))
        continue;
      auto value = baseValues->second.find(name);
      auto type = baseTypes->second.find(name);
      if (value == baseValues->second.end() ||
          type == baseTypes->second.end())
        continue;
      staticAttrNames.push_back(name);
      staticAttrValues.push_back(value->second);
      staticAttrTypes.push_back(type->second);
    }
  }
  llvm::StringMap<mlir::Type> &registeredStaticAttrs =
      classStaticAttrBindings[contractName];
  registeredStaticAttrs.clear();
  llvm::StringMap<mlir::Attribute> &registeredStaticValues =
      classStaticAttrValues[contractName];
  registeredStaticValues.clear();
  classStaticAttrOrders[contractName].assign(staticAttrNames.begin(),
                                             staticAttrNames.end());
  for (auto [attrName, attrValue, attrType] :
       llvm::zip_equal(staticAttrNames, staticAttrValues, staticAttrTypes)) {
    // ⛔ A FLOAT LITERAL TYPE IS REGISTERED WIDENED. The constant channel
    // materializes the read from the attribute's own type, and the lowering's
    // default-value path knows the int/str/bool/None literal spellings but not
    // a float one -- "default argument has no concrete runtime contract", for
    // an imported `ratio = 1.5` whose int twin beside it worked. The metadata
    // carries the value; the type only has to say what it is.
    if (mlir::isa_and_nonnull<py::LiteralType>(attrType) &&
        types.widenLiteral(attrType) == types.floatType())
      attrType = types.floatType();
    registeredStaticAttrs[attrName] = attrType;
    registeredStaticValues[attrName] = attrValue;
    types.bindClassStaticAttr(contractName, attrName,
                              types.widenLiteral(attrType));
  }

  llvm::SmallVector<std::string, 8> methodNames;
  llvm::SmallVector<std::string, 8> methodKinds;
  llvm::SmallVector<std::string, 8> methodSymbols;
  llvm::SmallVector<mlir::Type, 8> methodContracts;
  // Pass 1 registers every method binding before pass 2 emits any body:
  // a method body may call a sibling declared later in the class (and MRO
  // lookups during emission must already see the full method set).
  llvm::SmallVector<const parser::Node *, 8> pendingBodies;
  llvm::SmallVector<FunctionSignature, 8> pendingBodySigs;
  llvm::SmallVector<std::string, 8> pendingBodySymbols;
  llvm::SmallVector<std::string, 8> pendingBodyKinds;
  // The module whose names each body reads: this class's, except for a copy
  // of an inherited body (below), which reads the module that wrote it.
  llvm::SmallVector<const EmitOptions::SourceModule *, 8> pendingBodySources;
  // And the class whose type parameters it spells: this class, except for
  // such a copy, which spells the parameters of the class that wrote it.
  llvm::SmallVector<std::string, 8> pendingBodyDefiners;
  llvm::StringSet<> propertyNames = classPropertyNames(classDef);
  // Entries published ahead of their method's turn (below), replaced rather
  // than added to when the turn comes: the table holds a method's overloads
  // as a list, and the early entry would stay as a second overload.
  llvm::StringSet<> provisionalMethods;
  if (const auto *body = ast::nodeList(classDef, "body")) {
    for (const parser::NodePtr &statement : *body) {
      // ⭐ A NESTED CLASS IS DROPPED, and only its USE said so. The refusals
      // for a class statement elsewhere live in `emitStatement`, which never
      // sees a class BODY -- so `class Outer: class Inner: ...` was accepted,
      // the inner class was never emitted, and `Outer.Inner()` reported
      // "static type !py.type<Outer> does not provide manifest method
      // 'Inner'": a sentence about a method, for a class the source declares.
      if (statement && statement->kind == "ClassDef") {
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, statement->range.start,
            "a class defined inside another class is not supported: a class "
            "is a module-level contract here, so declare '" +
                std::string(ast::nameSpelling(*statement)) +
                "' at module scope or in a function"});
        continue;
      }
      // ⭐ AND A STATEMENT THAT WOULD HAVE RUN. A class body executes in
      // CPython, and here it DECLARES -- so everything this walk does not
      // recognise was dropped in silence:
      //
      //     log: "list[str]" = []
      //     class C:
      //         log.append("body")     # printed nothing; CPython appends
      //
      // and a `for` in a class body lost its side effects the same way. The
      // recognised set is the declarations plus the two shapes with no effect
      // at all: a docstring and `pass`. A `type` alias declares as well.
      //
      // ⛔ Including a decidable `if`, which the platform-switch idiom writes
      // (`if sys.platform == "win32": tag = "win"`). Flattening it the way the
      // module walk does would mean threading a rewritten body through the
      // nine walks of this one, and the shape is not silent today -- it
      // reports "'C' object has no attribute 'tag'" at the use, which this
      // refusal replaces with a sentence about the declaration.
      if (statement && statement->kind != "FunctionDef" &&
          statement->kind != "AsyncFunctionDef" &&
          statement->kind != "AnnAssign" && statement->kind != "Assign" &&
          statement->kind != "Pass" && statement->kind != "TypeAlias") {
        bool docstring = false;
        if (statement->kind == "Expr")
          if (const parser::Node *value = ast::node(*statement, "value"))
            docstring = value->kind == "Constant";
        if (!docstring) {
          diagnostics.push_back(parser::Diagnostic{
              parser::Severity::Error, statement->range.start,
              "a '" + statement->kind +
                  "' statement in a class body is not supported: a class body "
                  "here declares members and does not run, so this statement "
                  "would be dropped in silence"});
          continue;
        }
      }
      if (!statement || (statement->kind != "FunctionDef" &&
                         statement->kind != "AsyncFunctionDef"))
        continue;
      classBodyDefs.insert(statement.get());
      auto methodName = ast::string(*statement, "name");
      if (!methodName)
        continue;
      checkDecorators(*statement, DecoratorRole::Method, &propertyNames);
      std::string kind = methodKind(*statement);
      std::string bindingName(*methodName);
      if (const auto *decorators = ast::nodeList(*statement, "decorator_list"))
        for (const parser::NodePtr &decorator : *decorators) {
          if (!decorator)
            continue;
          if (decorator->kind == "Name" &&
              ast::nameSpelling(*decorator) == "property") {
            kind = "property";
            break;
          }
          llvm::StringRef setterTarget = propertySetterTarget(*decorator);
          if (!setterTarget.empty() && propertyNames.contains(setterTarget)) {
            kind = "property_setter";
            bindingName = (setterTarget + ".setter").str();
            break;
          }
        }
      if (*methodName == "__new__" && kind == "instance")
        kind = "class";
      // ⭐ THE TWO IMPLICIT CLASSMETHODS, which CPython wraps for you: written
      // without `@classmethod` (the way everyone writes them) their `cls`
      // reached the ordinary parameter rule and the class was refused with
      // "function parameter 'cls' requires an annotation". `__new__` above is
      // the same rule for the same reason.
      if ((*methodName == "__init_subclass__" ||
           *methodName == "__class_getitem__") &&
          kind == "instance")
        kind = "classmethod";
      bool propertyAccessor = kind == "property" || kind == "property_setter";
      std::optional<llvm::StringRef> receiverName;
      if (kind == "instance" || propertyAccessor)
        receiverName = "self";
      else if (kind == "class" || kind == "classmethod")
        receiverName = "cls";
      mlir::Type receiverType;
      if (kind == "instance" || propertyAccessor)
        receiverType = types.contract(contractName);
      else if (kind == "class" || kind == "classmethod")
        receiverType = types.typeObject(types.contract(contractName));
      std::size_t firstPassDiagnostics = diagnostics.size();
      auto computeBodySignature = [&] {
        FunctionSignature computed = types.functionSignature(
            *statement,
            kind == "static" ? std::optional<llvm::StringRef>() : receiverName,
            py::CallableType(), receiverType);
        if (kind == "instance" || propertyAccessor)
          replaceSelfInSignature(computed, types.contract(contractName), types);
        else if (kind == "class" || kind == "classmethod") {
          replaceSelfInSignature(
              computed, types.typeObject(types.contract(contractName)), types);
          if (!computed.positionalTypes.empty()) {
            computed.positionalTypes.front() =
                types.typeObject(types.contract(contractName));
            types.refreshCallable(computed);
          }
        }
        return computed;
      };
      FunctionSignature bodySig = computeBodySignature();
      // ⭐ A GENERATOR THAT CALLS ITSELF NEEDS TO BE PUBLISHED FIRST. Computing
      // a generator's signature WALKS its body to infer the yield type, and a
      // recursive `k.walk()` in there resolves against a table that does not
      // have `walk` yet -- the publication below happens after this. The tree
      // walk every iterable tree is written as
      //
      //     def walk(self):
      //         yield self.v
      //         for k in self.kids:
      //             for x in k.walk():
      //                 yield x
      //
      // was refused with "static type 'T' does not provide manifest method
      // 'walk'" pointing at the `def`, while the same method called from a
      // SIBLING method compiled: the sibling runs after the whole walk.
      //
      // So a generator that names itself is published from the first pass and
      // its signature recomputed. The second pass is what the annotation makes
      // exact; an UNANNOTATED self-recursive generator still ends at `object`,
      // which is the boundary the progressive publication already documents
      // for two unannotated methods that call each other.
      //
      // ⭐ AND SO DOES ONE THAT CALLS A METHOD WRITTEN BELOW IT. Every other
      // method reaches a later sibling, because its body is emitted after the
      // whole class is published; a generator's walk runs here, in textual
      // order, so difflib's
      //
      //     def _fancy_replace(self, ...) -> Iterator[str]:
      //         yield from self._fancy_helper(...)   # defined below
      //
      // was refused with "does not provide manifest method '_fancy_helper'".
      // The later method is published from its own signature first, and
      // republished in its own turn.
      // ⛔ Only a plain instance method: a property, a static or a class
      // method has its receiver and binding decided further down this loop,
      // and publishing one under the instance rule would be a wrong entry
      // rather than a missing one.
      // A sibling whose own walk went through is published as inferred: the
      // annotation `Iterator[str]` and the inferred `GeneratorType[str, ...]`
      // join to `object`, so `g = self._dump(...)` in one arm and
      // `g = self._fancy_replace(...)` in the other lost the yield type.
      auto annotatedGeneratorEntry = [&](const parser::Node &def,
                                         FunctionSignature signature,
                                         bool spoiled) {
        // ⭐ THE ANNOTATION IS WHAT THE FIRST PASS PUBLISHES. A generator's
        // public result is the INFERRED generator type, and on the first pass
        // that inference is exactly what the missing entry spoiled -- so
        // publishing it would hand the second pass the same `object` back.
        // With `-> Iterator[int]` written down the answer is already there.
        if (spoiled && (signature.isGeneratorFunction ||
                        signature.isAsyncGeneratorFunction))
          if (const parser::Node *returns = ast::node(def, "returns"))
            if (mlir::Type annotated = types.annotationType(returns)) {
              signature.inferredGeneratorType = annotated;
              types.refreshCallable(signature);
            }
        return signature.publicCallable;
      };
      llvm::SmallVector<const parser::Node *, 4> laterSiblings;
      if (bodySig.isGeneratorFunction || bodySig.isAsyncGeneratorFunction) {
        bool below = false;
        for (const parser::NodePtr &other : *body) {
          if (other.get() == statement.get()) {
            below = true;
            continue;
          }
          if (!below || !other || other->kind != "FunctionDef")
            continue;
          const auto *decorators = ast::nodeList(*other, "decorator_list");
          if (decorators && !decorators->empty())
            continue;
          auto otherName = ast::string(*other, "name");
          if (otherName && *otherName != *methodName &&
              bodyMentionsAttribute(*statement, *otherName))
            laterSiblings.push_back(other.get());
        }
      }
      if ((bodySig.isGeneratorFunction || bodySig.isAsyncGeneratorFunction) &&
          (bodyMentionsAttribute(*statement, *methodName) ||
           !laterSiblings.empty())) {
        auto publishProvisionally = [&](llvm::StringRef name,
                                        mlir::Type entry) {
          if (provisionalMethods.insert(name).second)
            addProtocolMethod(progressiveInfo, name, entry);
        };
        publishProvisionally(*methodName,
                             annotatedGeneratorEntry(*statement, bodySig,
                                                     /*spoiled=*/true));
        for (const parser::Node *sibling : laterSiblings) {
          FunctionSignature siblingSig = types.functionSignature(
              *sibling, llvm::StringRef("self"), py::CallableType(),
              types.contract(contractName));
          replaceSelfInSignature(siblingSig, types.contract(contractName),
                                 types);
          publishProvisionally(
              *ast::string(*sibling, "name"),
              annotatedGeneratorEntry(
                  *sibling, siblingSig,
                  !siblingSig.generatorAnalysisFailures.empty() ||
                      !siblingSig.generatorAnnotationMismatch.empty()));
        }
        publishProgress();
        std::size_t before = diagnostics.size();
        bodySig = computeBodySignature();
        // The first pass reported what the missing entry caused; the second is
        // the one whose report is about the program.
        diagnostics.erase(diagnostics.begin() + firstPassDiagnostics,
                          diagnostics.begin() + before);
      }
      if (propertyAccessor) {
        // Accessors inline at attribute-access sites only: no standalone
        // symbol, no method-table entry.
        unsigned expectedArity = kind == "property" ? 1 : 2;
        if (bodySig.positionalNames.size() != expectedArity)
          diagnostics.push_back(parser::Diagnostic{
              parser::Severity::Error, statement->range.start,
              "property " +
                  std::string(kind == "property" ? "getter" : "setter") +
                  " takes " + std::to_string(expectedArity) +
                  " parameters (including self)"});
        classMethodBindings[contractName][bindingName] =
            MethodBinding{statement.get(),
                          bodySig,
                          bodySig,
                          kind,
                          std::string(),
                          statement->kind == "AsyncFunctionDef",
                          std::string(contractName)};
        // The getter's RETURN type is what `instance.name` reads as. Without
        // it the attribute inference answers `builtins.object`.
        if (kind == "property" && methodName)
          types.bindClassPropertyType(contractName, *methodName,
                                      bodySig.resultType);
        continue;
      }
      methodNames.push_back(std::string(*methodName));
      methodKinds.push_back(kind);
      methodContracts.push_back(bodySig.publicCallable);
      // Visible to the NEXT method's body walk. `def twice(self): return
      // self.peek() * 2` inferred against a table that learned this class's
      // methods only after every signature was computed, so a sibling call
      // read as "contract 'Chain' does not provide manifest method 'peek'"
      // -- while the same call compiled once the walk was over. Textual
      // order is what this buys; two unannotated methods that call each
      // other still need the annotation.
      if (provisionalMethods.erase(*methodName))
        progressiveInfo.methods.erase(std::string(*methodName));
      addProtocolMethod(progressiveInfo, *methodName, bodySig.publicCallable);
      publishProgress();

      std::string symbolName =
          sourceMethodSymbolName(contractName, *methodName, *statement);
      methodSymbols.push_back(symbolName);
      // Exception-backed classes get no standalone method symbols: their
      // bodies inline at call sites, and a standalone copy would transfer
      // the borrowed receiver's header through the runtime exception
      // __init__ (an ownership violation the verifier rejects). Nor does a
      // generic method (`def m[T](self, x: T)`): its body has no type until
      // a call site solves T, which the inlining does.
      // ⭐ Except an exception's text, `__str__` and `__repr__`: a boxed
      // exception -- an element printed with its list -- is rendered through
      // its class's slot, which only a standalone body can fill, and without
      // one it printed BaseException's text for a class that declares its
      // own.
      bool exceptionText =
          *methodName == "__str__" || *methodName == "__repr__";
      if (kind != "class" && kind != "classmethod" &&
          (exceptionText || !isExceptionBackedClass(contractName)) &&
          !specializedAtEachCall(*statement)) {
        pendingBodies.push_back(statement.get());
        pendingBodySigs.push_back(bodySig);
        pendingBodySymbols.push_back(symbolName);
        pendingBodyKinds.push_back(kind);
        pendingBodySources.push_back(activeSourceModule);
        pendingBodyDefiners.push_back(contractName.str());
      }
      if (kind == "static")
        types.bindClassStaticMethod(contractName, *methodName,
                                    bodySig.publicCallable ? bodySig.publicCallable
                                                           : bodySig.callable);
      classMethodBindings[contractName][*methodName] =
          MethodBinding{statement.get(),
                        bodySig,
                        bodySig,
                        kind,
                        symbolName,
                        statement->kind == "AsyncFunctionDef",
                        std::string(contractName)};
    }
  }
  // ⭐ AN INHERITED BODY THAT CALLS AN OVERRIDDEN METHOD NEEDS ITS OWN COPY.
  // A class inherits its base's method by pointing at the BASE's symbol, and
  // that symbol's body typed `self` as the base -- so a call inside it bound
  // statically to the base's version:
  //
  //     class Shape:
  //         def area(self) -> int: return 0
  //         def __repr__(self) -> str: return "S(" + str(self.area()) + ")"
  //     class Sq(Shape):
  //         def area(self) -> int: return self.n * self.n
  //     print([Sq(3)])        # printed [S(0)]; CPython prints [S(9)]
  //
  // A SILENT wrong answer, and `print(Sq(3))` beside it is right -- that path
  // INLINES the body at the call site, where `self` is the receiver's own
  // class. The container's repr reaches the body through its symbol instead,
  // and a symbol has one `self` type forever.
  //
  // So a subclass that redeclares something an inherited body calls gets its
  // own emission of that body, with `self` at ITS class. Everything the body
  // resolves is then resolved against the class that will run it.
  //
  // ⛔ Only where it would ANSWER DIFFERENTLY -- the body has to name a member
  // this class redeclares -- because every copy is code the program did not
  // write, and the inherited symbol is correct for all the rest.
  //
  // ⛔ And not a body containing `super()`: the emission below pushes THIS
  // class as the super context, while an inherited body means the DEFINING
  // class's base. Those keep the inherited symbol and its old answer.
  {
    llvm::SmallVector<std::string, 4> ownMethodNames(methodNames.begin(),
                                                     methodNames.end());
    auto bodyNamesRedeclaredMember = [&](const parser::Node &method,
                                         llvm::StringRef selfName) {
      bool found = false;
      bool sawSuper = false;
      ast::walk(&method, [&](const parser::Node &current) {
        if (current.kind == "Name" && ast::nameSpelling(current) == "super")
          sawSuper = true;
        if (current.kind != "Attribute")
          return ast::Walk::Continue;
        const parser::Node *owner = ast::node(current, "value");
        if (!owner || owner->kind != "Name" ||
            llvm::StringRef(ast::nameSpelling(*owner)) != selfName)
          return ast::Walk::Continue;
        if (auto member = ast::string(current, "attr"))
          if (llvm::is_contained(ownMethodNames, llvm::StringRef(*member)))
            found = true;
        return ast::Walk::Continue;
      });
      return found && !sawSuper;
    };
    for (const std::string &ancestor :
         llvm::ArrayRef<std::string>(mro).drop_front()) {
      auto bindings = classMethodBindings.find(ancestor);
      if (bindings == classMethodBindings.end())
        continue;
      for (const auto &entry : bindings->second) {
        llvm::StringRef inheritedName = entry.getKey();
        if (llvm::is_contained(methodNames, inheritedName))
          continue;
        const MethodBinding &inherited = entry.second;
        if (inherited.kind != "instance" || !inherited.method ||
            inherited.symbolName.empty() || inherited.async)
          continue;
        if (inherited.bodySignature.positionalNames.empty())
          continue;
        if (isExceptionBackedClass(contractName))
          continue;
        if (!bodyNamesRedeclaredMember(
                *inherited.method, inherited.bodySignature.positionalNames.front()))
          continue;
        FunctionSignature specialized = inherited.bodySignature;
        specialized.positionalTypes.front() = types.contract(contractName);
        types.refreshCallable(specialized);
        std::string symbolName = sourceMethodSymbolName(
            contractName, inheritedName, *inherited.method);
        methodNames.push_back(inheritedName.str());
        methodKinds.push_back("instance");
        methodContracts.push_back(specialized.publicCallable
                                      ? specialized.publicCallable
                                      : specialized.callable);
        methodSymbols.push_back(symbolName);
        if (specializedAtEachCall(*inherited.method))
          continue;
        pendingBodies.push_back(inherited.method);
        pendingBodySigs.push_back(specialized);
        pendingBodySymbols.push_back(symbolName);
        pendingBodyKinds.push_back("instance");
        // ⭐ THE COPY STILL READS THE MODULE THAT WROTE IT. Only `self`
        // changes class; every other name in the body is the defining
        // module's global, and emitted in this class's module a helper
        // function private to the base's module was an "unresolved name" --
        // a main-module TestCase subclass overriding setUp lost unittest's
        // own functions inside the run() it inherits.
        pendingBodySources.push_back(
            sourceModuleForClass(inherited.definingClass));
        pendingBodyDefiners.push_back(inherited.definingClass);
      }
    }
  }
  if (isDataclass) {
    // Synthesize __init__/__repr__/__eq__ from the MRO-merged field list
    // (CPython composes base dataclass fields the same way); an explicit
    // user definition wins, as in dataclasses._set_new_attribute.
    // A field a method created (`self.total = ...` in __post_init__) is an
    // instance attribute, not a dataclass field.
    llvm::SmallVector<std::string, 8> dataclassFields;
    for (const std::string &field : classFieldOrders[contractName])
      if (llvm::none_of(classMros[contractName], [&](const std::string &cls) {
            auto created = classFieldsDeclaredOutsideInit.find(cls);
            return created != classFieldsDeclaredOutsideInit.end() &&
                   created->second.contains(field);
          }))
        dataclassFields.push_back(field);
    llvm::ArrayRef<std::string> order = dataclassFields;
    const llvm::StringMap<mlir::Type> &fieldTypeMap =
        classFieldBindings[contractName];
    auto defaultNodeFor = [&](llvm::StringRef field) -> parser::NodePtr {
      for (const std::string &cls : classMros[contractName]) {
        auto perClass = classFieldDefaultNodes.find(cls);
        if (perClass == classFieldDefaultNodes.end())
          continue;
        auto found = perClass->second.find(field);
        if (found != perClass->second.end())
          return found->second;
      }
      return nullptr;
    };
    bool sawDefault = false;
    for (const std::string &field : order) {
      parser::NodePtr defaultNode = defaultNodeFor(field);
      if (defaultNode) {
        sawDefault = true;
      } else if (sawDefault) {
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, classDef.range.start,
            "non-default argument '" + field +
                "' follows default argument in dataclass field order"});
      }
    }
    // ⭐ Through the MRO, not the class's own dict. `@dataclass class C(B)`
    // that declares nothing INHERITS B's `__post_init__`, and CPython's
    // generated __init__ calls it (it tests hasattr on the instance). Asking
    // only what this class declares dropped the call silently -- `C().a`
    // printed the default instead of what __post_init__ set.
    //
    // The synthesis decisions above it read the same way: a subclass that
    // inherits `__init__`/`__repr__`/`__eq__` should not have one synthesized
    // over it either, which is what `dataclasses` does by checking the class
    // dict of the class it is decorating... so those three stay OWN-DICT and
    // only the hasattr-shaped question moves to the MRO.
    auto userDefines = [&](llvm::StringRef method) {
      auto ownMethods = classMethodBindings.find(contractName);
      return ownMethods != classMethodBindings.end() &&
             ownMethods->second.count(method);
    };
    auto inheritsOrDefines = [&](llvm::StringRef method) {
      return resolveMroMethod(contractName, method).has_value();
    };
    parser::SourceRange range = classDef.range;
    auto registerSynthesized = [&](parser::NodePtr fn,
                                   FunctionSignature bodySig) {
      auto fnName = ast::string(*fn, "name");
      types.refreshCallable(bodySig);
      std::string symbolName =
          sourceMethodSymbolName(contractName, *fnName, *fn);
      methodNames.push_back(std::string(*fnName));
      methodKinds.push_back("instance");
      methodContracts.push_back(bodySig.publicCallable
                                    ? bodySig.publicCallable
                                    : bodySig.callable);
      methodSymbols.push_back(symbolName);
      pendingBodies.push_back(fn.get());
      pendingBodySigs.push_back(bodySig);
      pendingBodySymbols.push_back(symbolName);
      pendingBodyKinds.push_back("instance");
      pendingBodySources.push_back(activeSourceModule);
      pendingBodyDefiners.push_back(contractName.str());
      classMethodBindings[contractName][*fnName] =
          MethodBinding{fn.get(),   bodySig, bodySig,
                        "instance", symbolName, /*async=*/false,
                        std::string(contractName)};
      synthesizedClassMethods.push_back(std::move(fn));
    };
    auto fieldType = [&](llvm::StringRef field) {
      auto found = fieldTypeMap.find(field);
      return found != fieldTypeMap.end() ? found->second : types.object();
    };

    if (dataclassInit && !userDefines("__init__")) {
      llvm::SmallVector<std::string, 8> paramNames{"self"};
      std::vector<parser::NodePtr> defaults;
      std::vector<parser::NodePtr> body;
      FunctionSignature sig;
      sig.positionalNames.push_back("self");
      sig.positionalTypes.push_back(types.contract(contractName));
      sig.positionalDefaults.push_back(false);
      for (const std::string &field : order) {
        paramNames.push_back(field);
        sig.positionalNames.push_back(field);
        sig.positionalTypes.push_back(fieldType(field));
        parser::NodePtr defaultNode = defaultNodeFor(field);
        sig.positionalDefaults.push_back(defaultNode != nullptr);
        if (defaultNode)
          defaults.push_back(defaultNode);
        parser::NodePtr assign = synth::assign(
                             synth::selfAttribute("self", field, range), synth::name(field, range), range);
        body.push_back(std::move(assign));
      }
      // ⭐ CPython's dataclass `__init__` ends by calling `__post_init__` when
      // the class defines one (`dataclasses._process_class` appends exactly
      // this call). Without it the hook was never reached and nothing said so:
      //
      //     @dataclass
      //     class P:
      //         x: int
      //         def __post_init__(self) -> None: print("ran")
      //     P(1)      # printed nothing; CPython prints "ran"
      //
      // Emitted as a call statement on the synthesized body so it goes through
      // the ordinary method dispatch, the way the field assignments above go
      // through ordinary assignment.
      if (inheritsOrDefines("__post_init__")) {
        parser::NodePtr hook = synth::call(synth::selfAttribute("self", "__post_init__", range), std::vector<parser::NodePtr>{}, range);
        parser::NodePtr statement = synth::exprStmt(std::move(hook), range);
        body.push_back(std::move(statement));
      }
      if (body.empty())
        body.push_back(parser::makeNode("Pass", range));
      sig.resultType = types.none();
      registerSynthesized(
          synth::functionDef("__init__", toSynthParams(paramNames), std::move(defaults),
                           std::move(body), nullptr,
                       llvm::ArrayRef<llvm::StringRef>{}, range),
          std::move(sig));
    }
    if (dataclassRepr && !userDefines("__repr__")) {
      // ⭐ `__qualname__`, WHICH HAS NO MODULE IN IT. CPython's dataclass
      // repr prints the class's qualified NAME, not its module path, so an
      // imported `Point(3, 4)` renders as `Point(x=3, y=4)` there and
      // rendered as `data.Point(x=3, y=4)` here -- a silent wrong answer for
      // every imported dataclass printed.
      //
      // ⛔ NOT the same as the default object repr beside it, which CPython
      // spells `<data.Point object at 0x...>` WITH the module and which this
      // compiler was taught to qualify for exactly that reason. The two reprs
      // read different attributes and disagree on purpose.
      // ⛔ And not a NamedTuple's, which prints `__name__`: the two differ for
      // a class written in a function (`f.<locals>.P(x=1)`, `P(x=1)`).
      std::string className =
          isNamedTuple
              ? py::contracts::pythonClassNameForContract(contractName)
              : py::contracts::pythonQualnameForContract(contractName);
      parser::NodePtr expr;
      if (order.empty()) {
        expr = synth::strConstant(className + "()", range);
      } else {
        expr = synth::strConstant(className + "(", range);
        for (auto [index, field] : llvm::enumerate(order)) {
          std::string label = (index ? ", " : "") + field + "=";
          expr = synth::binOp(std::move(expr), "Add", synth::strConstant(label, range),
                          range);
          expr = synth::binOp(
              std::move(expr), "Add",
              synth::reprCall(synth::selfAttribute("self", field, range), range),
              range);
        }
        expr = synth::binOp(std::move(expr), "Add", synth::strConstant(")", range), range);
      }
      parser::NodePtr returnNode = synth::returnStmt(std::move(expr), range);
      FunctionSignature sig;
      sig.positionalNames.push_back("self");
      sig.positionalTypes.push_back(types.contract(contractName));
      sig.positionalDefaults.push_back(false);
      sig.resultType = types.strType();
      registerSynthesized(
          synth::functionDef("__repr__", toSynthParams({"self"}), {},
                           {std::move(returnNode)}, nullptr,
                       llvm::ArrayRef<llvm::StringRef>{}, range),
          std::move(sig));
    }
    // ⭐ A NamedTuple is hashable, so it can be a dict key.
    //
    //     class Key(NamedTuple):
    //         row: int
    //     d: dict[Key, str] = {}
    //     d[Key(0)] = "origin"
    //     print(d[Key(0)])      # KeyError; CPython prints origin
    //
    // Equality was already right -- `Key(0) == Key(0)` is True -- and the
    // runtime dict probes by hash first, so two equal keys landed in different
    // buckets. CPython's namedtuple inherits tuple.__hash__, which combines
    // the fields; a plain dataclass gets `__hash__ = None` instead, which is
    // why this is gated on the NamedTuple marker rather than on `isDataclass`.
    //
    // ⭐ `hash((self.f0, self.f1, ...))` -- tuple's own hash, which is what
    // CPython's namedtuple inherits. It used to be an XOR fold of the fields'
    // hashes, on the reasoning that Python only requires equal objects to hash
    // equal. They ARE equal: a NamedTuple compares equal to a plain tuple with
    // the same contents, so `hash(P(3)) == hash((3,))` must hold and did not.
    // A dict keyed by tuples then missed a NamedTuple key that compares equal
    // to one already in it.
    //
    // Why NOT a new primitive: the tuple this builds hashes through the same
    // manifest `tuple.__hash__` any other tuple does, so the two answers are
    // the same by construction rather than by agreement.
    // ⭐ A NamedTuple IS a tuple: len() is its field count, known here, and a
    // literal subscript is the field at that position (folded at the
    // subscript, where the index is). `len(p)` and `p[0]` were "contract 'P'
    // does not provide manifest method '__len__' / '__getitem__'" while
    // `p.x` and `print(p)` worked.
    if (isNamedTuple)
      namedTupleContracts.insert(contractName);
    if (isNamedTuple && !userDefines("__len__")) {
      parser::NodePtr count = synth::intConstant(static_cast<std::int64_t>(order.size()), range);
      parser::NodePtr returnCount = synth::returnStmt(std::move(count), range);
      FunctionSignature lenSig;
      lenSig.positionalNames.push_back("self");
      lenSig.positionalTypes.push_back(types.contract(contractName));
      lenSig.positionalDefaults.push_back(false);
      lenSig.resultType = types.intType();
      registerSynthesized(synth::functionDef("__len__", toSynthParams({"self"}), {},
                                           {std::move(returnCount)}, nullptr,
                       llvm::ArrayRef<llvm::StringRef>{}, range),
                          std::move(lenSig));
    }
    // ⭐ A NamedTuple ORDERS like the tuple it is, field by field. CPython's
    // namedtuple inherits tuple's comparisons, so `sorted(recs)` over a list
    // of them is ordinary Python -- and it failed at RUNTIME with "'<' not
    // supported between operand types" while `Rec(...) < Rec(...)` was refused
    // at emit. Built as the comparison of two tuples, for the reason the hash
    // above is: the answer then comes from the same manifest comparison any
    // other tuple gets, rather than from a second implementation that has to
    // agree with it.
    if (isNamedTuple && !order.empty()) {
      struct Ordering {
        const char *method;
        const char *op;
      };
      static constexpr Ordering kOrderings[] = {{"__lt__", "Lt"},
                                                {"__le__", "LtE"},
                                                {"__gt__", "Gt"},
                                                {"__ge__", "GtE"}};
      for (const Ordering &ordering : kOrderings) {
        if (userDefines(ordering.method))
          continue;
        auto fieldTuple = [&](llvm::StringRef receiver) {
          std::vector<parser::NodePtr> elements;
          elements.reserve(order.size());
          for (const std::string &field : order)
            elements.push_back(synth::selfAttribute(receiver, field, range));
          parser::NodePtr tuple = parser::makeNode("Tuple", range);
          parser::addField(*tuple, "elts", std::move(elements));
          return tuple;
        };
        parser::NodePtr comparison =
            synth::compare(fieldTuple("self"), ordering.op,
                           fieldTuple("other"), range);
        FunctionSignature orderSig;
        orderSig.positionalNames.append({"self", "other"});
        orderSig.positionalTypes.push_back(types.contract(contractName));
        orderSig.positionalTypes.push_back(types.contract(contractName));
        orderSig.positionalDefaults.append({false, false});
        orderSig.resultType = types.boolType();
        registerSynthesized(
            synth::functionDef(ordering.method, toSynthParams({"self", "other"}),
                               {},
                               {synth::returnStmt(std::move(comparison), range)},
                               nullptr, llvm::ArrayRef<llvm::StringRef>{},
                               range),
            std::move(orderSig));
      }
    }
    if (isNamedTuple && !userDefines("__hash__") && !order.empty()) {
      std::vector<parser::NodePtr> elements;
      elements.reserve(order.size());
      for (const std::string &field : order)
        elements.push_back(synth::selfAttribute("self", field, range));
      parser::NodePtr tuple = parser::makeNode("Tuple", range);
      parser::addField(*tuple, "elts", std::move(elements));
      parser::NodePtr call = synth::call(synth::name("hash", range), std::vector<parser::NodePtr>{std::move(tuple)}, range);
      parser::NodePtr returnNode = synth::returnStmt(std::move(call), range);
      FunctionSignature sig;
      sig.positionalNames.push_back("self");
      sig.positionalTypes.push_back(types.contract(contractName));
      sig.positionalDefaults.push_back(false);
      sig.resultType = types.intType();
      registerSynthesized(synth::functionDef("__hash__", toSynthParams({"self"}), {},
                                           {std::move(returnNode)}, nullptr,
                       llvm::ArrayRef<llvm::StringRef>{}, range),
                          std::move(sig));
    }
    // ⭐ `_asdict` IS THE DICT LITERAL ITS FIELDS SPELL, and the field names and
    // types are both known right here. CPython's namedtuple builds it from
    // _fields and zip; there is nothing dynamic in it, so a synthesized method
    // is the whole implementation. `p._asdict()` was "'P' inherits
    // builtins.object._asdict, which Lython does not implement".
    //
    // ⛔ The value type is the JOIN of the field types, which is what makes a
    // mixed NamedTuple's dict a dict[str, int | str] rather than a refusal --
    // the same type the equivalent literal gets.
    if (isNamedTuple && !userDefines("_asdict") && !order.empty()) {
      std::vector<parser::NodePtr> keys;
      std::vector<parser::NodePtr> valueNodes;
      llvm::SmallVector<mlir::Type, 8> valueTypes;
      for (const std::string &field : order) {
        keys.push_back(synth::strConstant(field, range));
        valueNodes.push_back(synth::selfAttribute("self", field, range));
        auto found = fieldTypeMap.find(field);
        valueTypes.push_back(found != fieldTypeMap.end() ? found->second
                                                         : types.object());
      }
      mlir::Type valueType = types.join(valueTypes);
      if (valueType) {
        parser::NodePtr mapping = parser::makeNode("Dict", range);
        parser::addField(*mapping, "keys", std::move(keys));
        parser::addField(*mapping, "values", std::move(valueNodes));
        parser::NodePtr returnNode =
            synth::returnStmt(std::move(mapping), range);
        FunctionSignature sig;
        sig.positionalNames.push_back("self");
        sig.positionalTypes.push_back(types.contract(contractName));
        sig.positionalDefaults.push_back(false);
        sig.resultType = types.dictOf(types.strType(), valueType);
        registerSynthesized(
            synth::functionDef("_asdict", toSynthParams({"self"}), {},
                               {std::move(returnNode)}, nullptr,
                               llvm::ArrayRef<llvm::StringRef>{}, range),
            std::move(sig));
      }
    }
    // ⭐ `@dataclass(frozen=True)` IS WHAT MAKES A RECORD HASHABLE, and that is
    // the whole reason it cannot stay refused: an unfrozen dataclass is
    // unhashable in CPython (and now here, at the key as well as at `hash()`),
    // so without frozen there is no spelling for a record used as a dict key.
    // The hash is `hash((self.f0, ...))` -- tuple's own, which is what CPython's
    // dataclass builds -- so it agrees with the synthesized __eq__ by
    // construction rather than by agreement, the same argument the NamedTuple
    // __hash__ above makes.
    if (dataclassFrozen)
      frozenDataclassContracts.insert(contractName);
    if (dataclassFrozen && dataclassEq && !userDefines("__hash__") &&
        !order.empty()) {
      std::vector<parser::NodePtr> members;
      for (const std::string &field : order)
        members.push_back(synth::selfAttribute("self", field, range));
      parser::NodePtr tuple = parser::makeNode("Tuple", range);
      parser::addField(*tuple, "elts", std::move(members));
      parser::NodePtr call = synth::call(
          synth::name("hash", range),
          std::vector<parser::NodePtr>{std::move(tuple)}, range);
      parser::NodePtr returnNode = synth::returnStmt(std::move(call), range);
      FunctionSignature sig;
      sig.positionalNames.push_back("self");
      sig.positionalTypes.push_back(types.contract(contractName));
      sig.positionalDefaults.push_back(false);
      sig.resultType = types.intType();
      registerSynthesized(
          synth::functionDef("__hash__", toSynthParams({"self"}), {},
                             {std::move(returnNode)}, nullptr,
                             llvm::ArrayRef<llvm::StringRef>{}, range),
          std::move(sig));
    }
    // ⭐ `@dataclass(order=True)` COMPARES THE FIELD TUPLES, in declaration
    // order, which is exactly what CPython's dataclasses does -- it builds
    // `(self.f0, ...) < (other.f0, ...)` and lets tuple's own ordering decide.
    // Written here as that same expression, so the answer comes from the
    // manifest tuple comparison rather than from a second implementation of
    // lexicographic order. It was "dataclass argument 'order' is not
    // supported", which is a refusal of `sorted(rows)` on a record type.
    //
    // ⛔ CPython returns NotImplemented for a different class and the operand
    // falls back to the reflected method; here the parameter is typed as this
    // class, so a foreign operand is refused at the call instead -- earlier,
    // and with the class named.
    if (dataclassOrder && !order.empty()) {
      static constexpr std::pair<const char *, const char *> kOrderings[] = {
          {"__lt__", "Lt"},
          {"__le__", "LtE"},
          {"__gt__", "Gt"},
          {"__ge__", "GtE"}};
      for (auto [methodName, opKind] : kOrderings) {
        if (userDefines(methodName))
          continue;
        auto fieldTuple = [&](llvm::StringRef receiver) {
          std::vector<parser::NodePtr> members;
          for (const std::string &field : order)
            members.push_back(synth::selfAttribute(receiver, field, range));
          parser::NodePtr literal = parser::makeNode("Tuple", range);
          parser::addField(*literal, "elts", std::move(members));
          return literal;
        };
        parser::NodePtr comparison = synth::compare(
            fieldTuple("self"), opKind, fieldTuple("other"), range);
        parser::NodePtr returnNode =
            synth::returnStmt(std::move(comparison), range);
        FunctionSignature sig;
        sig.positionalNames.push_back("self");
        sig.positionalTypes.push_back(types.contract(contractName));
        sig.positionalDefaults.push_back(false);
        sig.positionalNames.push_back("other");
        sig.positionalTypes.push_back(types.contract(contractName));
        sig.positionalDefaults.push_back(false);
        sig.resultType = types.boolType();
        registerSynthesized(
            synth::functionDef(methodName, toSynthParams({"self", "other"}), {},
                               {std::move(returnNode)}, nullptr,
                               llvm::ArrayRef<llvm::StringRef>{}, range),
            std::move(sig));
      }
    }
    // A SYNTHESIZED dataclass `__eq__` compares only against its own class and
    // answers False for any other -- which is what lets the comparison of two
    // distinct classes fold to a constant. A NamedTuple's does not: it is
    // tuple's, which compares by contents across classes. Recording which
    // classes got the guarded one is what keeps that fold honest; a
    // hand-written `__eq__` is not in the set and gets called.
    if (dataclassEq && !userDefines("__eq__") && !isNamedTuple)
      classesWithClassGuardedEq.insert(contractName);
    if (dataclassEq && !userDefines("__eq__")) {
      parser::NodePtr expr;
      llvm::SmallVector<parser::NodePtr, 8> comparisons;
      for (const std::string &field : order) {
        parser::NodePtr compare = parser::makeNode("Compare", range);
        parser::addField(*compare, "left", synth::selfAttribute("self", field, range));
        parser::addField(*compare, "ops",
                         std::vector<parser::NodePtr>{
                             parser::makeNode("Eq", range)});
        parser::addField(*compare, "comparators",
                         std::vector<parser::NodePtr>{
                             synth::selfAttribute("other", field, range)});
        comparisons.push_back(std::move(compare));
      }
      if (comparisons.empty()) {
        expr = parser::makeNode("Constant", range);
        parser::addField(*expr, "value", true);
      } else if (comparisons.size() == 1) {
        expr = std::move(comparisons.front());
      } else {
        expr = parser::makeNode("BoolOp", range);
        parser::addField(*expr, "op", parser::makeNode("And", range));
        parser::addField(
            *expr, "values",
            std::vector<parser::NodePtr>(comparisons.begin(),
                                         comparisons.end()));
      }
      parser::NodePtr returnNode = synth::returnStmt(std::move(expr), range);
      FunctionSignature sig;
      sig.positionalNames.append({"self", "other"});
      sig.positionalTypes.push_back(types.contract(contractName));
      sig.positionalTypes.push_back(types.contract(contractName));
      sig.positionalDefaults.append({false, false});
      sig.resultType = types.boolType();
      registerSynthesized(
          synth::functionDef("__eq__", toSynthParams({"self", "other"}), {},
                           {std::move(returnNode)}, nullptr,
                       llvm::ArrayRef<llvm::StringRef>{}, range),
          std::move(sig));
    }
  }

  // The class's protocol-table entry registers BEFORE its method bodies
  // emit: a body may resolve the class's own manifest evidence (its own
  // __init__ through super(), factory methods instantiating the class).
  py::protocols::ProtocolInfo protocolInfo;
  if (const auto *body = ast::nodeList(classDef, "body")) {
    for (const parser::NodePtr &statement : *body) {
      if (!statement || statement->kind != "Assign")
        continue;
      const auto *targets = ast::nodeList(*statement, "targets");
      if (!targets || targets->size() != 1 || !targets->front() ||
          targets->front()->kind != "Name" ||
          ast::nameSpelling(*targets->front()) != "__match_args__")
        continue;
      const parser::Node *value = ast::node(*statement, "value");
      const auto *elts =
          value && value->kind == "Tuple" ? ast::nodeList(*value, "elts")
                                          : nullptr;
      std::vector<std::string> names;
      bool wellFormed = elts != nullptr;
      if (elts)
        for (const parser::NodePtr &element : *elts) {
          std::optional<std::string_view> text =
              element && element->kind == "Constant"
                  ? ast::string(*element, "value")
                  : std::nullopt;
          if (!text) {
            wellFormed = false;
            break;
          }
          names.emplace_back(*text);
        }
      if (wellFormed)
        protocolInfo.matchArgs = std::move(names);
      break;
    }
  }
  for (const std::string &base : canonicalBases)
    protocolInfo.bases.push_back(py::protocols::ProtocolBase{
        py::contracts::manifestClassNameForContract(base), {}});
  // `builtins.object` closes every linearization (the C3 merge above already
  // appends it to `mro`), so the protocol table has to see it as a base too.
  // Appended LAST, not first, because base lookup returns the FIRST provider:
  // a declared base — or a dataclass-synthesized definition on this class —
  // still outranks the default, which is what makes the fall-through match
  // CPython's MRO instead of shadowing user code.
  protocolInfo.bases.push_back(py::protocols::ProtocolBase{
      py::contracts::manifestClassNameForContract("builtins.object"), {}});
  for (auto [fieldName, fieldType] : llvm::zip_equal(fieldNames, fieldTypes))
    protocolInfo.fields[fieldName] = fieldType;
  for (auto [methodName, methodContract] :
       llvm::zip_equal(methodNames, methodContracts))
    addProtocolMethod(protocolInfo, methodName, methodContract);
  py::protocols::Table::getMutable(context).registerClass(
      contractName, std::move(protocolInfo));

  // ⭐ QUEUED, NOT EMITTED, WHILE THE TOP LEVEL IS STILL DECLARING CLASSES. A
  // body is the only thing in a class that reads ANOTHER class's method
  // bindings, and this loop is the only thing in `emitClassContract` that runs
  // after those bindings are registered -- so moving it out is what lets two
  // sibling subclasses call the same base-typed method. `emitTopLevelDeclarations`
  // drains the queue once every class has been declared.
  if (deferClassMethodBodies) {
    for (auto [statement, bodySig, symbolName, kind, source, definer] :
         llvm::zip_equal(pendingBodies, pendingBodySigs, pendingBodySymbols,
                         pendingBodyKinds, pendingBodySources,
                         pendingBodyDefiners))
      deferredMethodBodies.push_back(DeferredMethodBody{
          statement, bodySig, symbolName, std::string(kind),
          std::string(contractName), source, definer});
    pendingBodies.clear();
    pendingBodySigs.clear();
    pendingBodySymbols.clear();
    pendingBodyKinds.clear();
    pendingBodySources.clear();
    pendingBodyDefiners.clear();
  }
  for (std::size_t index = 0; index < pendingBodies.size(); ++index) {
    const parser::Node *statement = pendingBodies[index];
    const FunctionSignature &bodySig = pendingBodySigs[index];
    const std::string &symbolName = pendingBodySymbols[index];
    const EmitOptions::SourceModule *source = pendingBodySources[index];
    const std::string &definer = pendingBodyDefiners[index];
    bool instanceBody = pendingBodyKinds[index] == "instance" &&
                        !bodySig.positionalNames.empty();
    // The frozen-field guard is off inside this class's own __init__, which is
    // where the fields are filled -- CPython goes around its own block there
    // with object.__setattr__. The body is emitted BOTH as a symbol (here) and
    // inlined at each construction, so the exemption is set in both places.
    std::string frozenOwner{contractName};
    std::optional<llvm::SaveAndRestore<const std::string *>> frozenInit;
    if (ast::string(*statement, "name").value_or("") == "__init__" &&
        frozenDataclassContracts.count(frozenOwner))
      frozenInit.emplace(frozenInitContract, &frozenOwner);
    // ⛔ The super() context is pushed inside: the module scope switch
    // starts from an empty one.
    auto emitBody = [&] {
      auto typeParameterScope = types.pushScope();
      bindClassTypeArguments(definer);
      if (instanceBody)
        superContexts.push_back(SuperContext{
            std::string(contractName), bodySig.positionalNames.front(),
            !decoratedMethodBodies.contains(statement)});
      emitCallableFunction(*statement, symbolName, bodySig, {},
                           /*isLambda=*/false);
      if (instanceBody)
        superContexts.pop_back();
    };
    if (source && source != activeSourceModule)
      emitInDefiningModuleScope(*source, emitBody);
    else
      emitBody();
  }

  // A generic class's specialization is demanded from a use site — possibly
  // deep inside a function body being emitted — so the contract op cannot go
  // wherever the builder happens to point. Every class contract is a
  // module-level declaration, which is where the non-generic callers already
  // stood.
  mlir::OpBuilder::InsertionGuard classGuard(builder);
  builder.setInsertionPointToEnd(module.getBody());

  mlir::OperationState state(loc(classDef), py::ClassOp::getOperationName());
  state.addAttribute(mlir::SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(contractName));
  // ⭐ THIS CLASS IS THE PROGRAM'S, and the class-word assignment has to know.
  // A manifest `py.class` carries its contract in its SYMBOL's leaf
  // (`py.class @Task` for `_asyncio.Task`), so the lowering finds a manifest
  // id by trying `builtins.`/`types.`/`_asyncio.`/`asyncio.`/`contextlib.` in
  // front of a bare name -- and `class Task` in a program then took asyncio's
  // id 15 instead of a fresh source id. Its instances were tagged as asyncio
  // Tasks and the program SEGFAULTED. Nothing else distinguishes the two: a
  // manifest class op carries no contract attribute of its own.
  state.addAttribute("ly.class.source", builder.getUnitAttr());
  // ⭐ A BASE IN AN IMPORTED MODULE IS SPELLED BY ITS CONTRACT. `base_names`
  // carried what the source WROTE -- a bare `Base` -- while the class op it
  // names is `@sub.Base`, so importing any module that declares a subclass
  // failed the dialect's own verifier before anything used it:
  //
  //     # sub.py: class Base: ...  /  class Derived(Base): pass
  //     import sub
  //     print("ok")
  //     # 'py.class' op unknown base class 'Base'
  //
  // `mro_names` beside it was already canonical, which is what made the two
  // disagree. The C3 merge computed the canonical name a few lines up; this
  // uses the same answer.
  //
  // ⛔ Only for a base this emitter DECLARED under that canonical name. A
  // manifest base keeps the spelling the verifier and the taxonomy look it up
  // by (`isManifestDeclaredClass`, and the loop lowering's note about reaching
  // a manifest base through `base_names`), and a main-module class is its own
  // canonical name already.
  llvm::SmallVector<std::string, 4> emittedBases;
  emittedBases.reserve(bases.size());
  for (auto [index, base] : llvm::enumerate(bases)) {
    llvm::StringRef canonical = index < canonicalBases.size()
                                    ? llvm::StringRef(canonicalBases[index])
                                    : base;
    if (canonical != base && canonical.contains('.')) {
      if (classBaseNames.count(canonical)) {
        emittedBases.push_back(canonical.str());
      } else {
        // ⭐ AND AN ALIAS OF A MANIFEST BASE RESOLVES TO THE SPELLING THE
        // TAXONOMY KNOWS. `from abc import ABC` binds ABC to `builtins.object`,
        // and keeping the source spelling handed the dialect a base it has
        // never heard of: "'py.class' op unknown base class 'ABC'". Every
        // other manifest base is written under its own name, where the raw
        // spelling and this one agree.
        emittedBases.push_back(
            py::contracts::manifestClassNameForContract(canonical));
      }
    } else {
      emittedBases.push_back(base.str());
    }
  }
  state.addAttribute("base_names", stringArray(builder, emittedBases));
  state.addAttribute("field_names", stringArray(builder, fieldNames));
  state.addAttribute("field_types", typeArray(builder, fieldTypes));
  state.addAttribute("field_contract_types", typeArray(builder, fieldTypes));
  state.addAttribute("method_names", stringArray(builder, methodNames));
  state.addAttribute("method_contracts", typeArray(builder, methodContracts));
  state.addAttribute("method_kinds", stringArray(builder, methodKinds));
  state.addAttribute("method_symbols", stringArray(builder, methodSymbols));

  if (!staticAttrNames.empty()) {
    state.addAttribute("class_static_attr_names",
                       stringArray(builder, staticAttrNames));
    state.addAttribute("class_static_attr_values",
                       builder.getArrayAttr(staticAttrValues));
  }
  state.addAttribute("mro_names",
                     stringArray(builder, llvm::ArrayRef<std::string>(mro)));

  state.addRegion();
  mlir::Operation *op = builder.create(state);
  op->getRegion(0).push_back(new mlir::Block);

}

void ModuleEmitter::collectStaticClassAssignments(
    const parser::Node &classDef, llvm::SmallVectorImpl<std::string> &names,
    llvm::SmallVectorImpl<mlir::Attribute> &values,
    llvm::SmallVectorImpl<mlir::Type> *typesOut) {
  mlir::Builder attrBuilder(&context);
  // The class body's names, in order, so a later attribute can be typed from an
  // earlier one (`half = width // 2`). The scope is this walk's alone; the
  // emitter binds the same names again, to VALUES, as it runs the initializers.
  TypeSystem::Scope classBodyScope = types.pushScope();
  auto appendStaticAttr = [&](llvm::StringRef name, const parser::Node *value,
                              mlir::Type annotatedType = {}) {
    mlir::Type valueType = annotatedType;
    if (!valueType)
      valueType = types.inferExpr(value);
    if (typesOut && !valueType) {
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error,
          value ? value->range.start : classDef.range.start,
          "class static attribute '" + name.str() +
              "' requires a statically inferred type"});
      return;
    }
    names.push_back(std::string(name));
    mlir::Attribute recorded = literalConstantAttr(attrBuilder, valueType);
    if (!recorded)
      recorded = sourceExprAttr(attrBuilder, value);
    values.push_back(recorded);
    if (typesOut)
      typesOut->push_back(valueType);
    if (valueType)
      types.bindLocalSymbol(name, valueType);
  };
  if (const auto *body = ast::nodeList(classDef, "body")) {
    for (const parser::NodePtr &statement : *body) {
      if (!statement)
        continue;
      if (statement->kind == "Assign") {
        const auto *targets = ast::nodeList(*statement, "targets");
        if (!targets || targets->size() != 1 || !targets->front() ||
            targets->front()->kind != "Name")
          continue;
        appendStaticAttr(ast::nameSpelling(*targets->front()),
                         ast::node(*statement, "value"));
        continue;
      }
      if (statement->kind == "AnnAssign") {
        const parser::Node *target = ast::node(*statement, "target");
        const parser::Node *value = ast::node(*statement, "value");
        if (!target || target->kind != "Name" || !value)
          continue;
        if (value->kind == "Call") {
          auto [callee, spelling] = decoratorCallee(*value);
          if (decoratorLeafName(spelling) == "field") {
            // dataclasses.field(...) never evaluates as an attribute
            // value. CPython rewrites the class attribute to the default
            // (and removes it for default_factory); mirror that here so
            // the field() call is not emitted as an initializer.
            const parser::Node *defaultValue = nullptr;
            if (const auto *keywords = ast::nodeList(*value, "keywords"))
              for (const parser::NodePtr &keyword : *keywords)
                if (keyword && ast::string(*keyword, "arg") == "default")
                  defaultValue = ast::node(*keyword, "value");
            if (defaultValue)
              appendStaticAttr(
                  ast::nameSpelling(*target), defaultValue,
                  types.annotationType(ast::node(*statement, "annotation")));
            continue;
          }
        }
        appendStaticAttr(
            ast::nameSpelling(*target), value,
            types.annotationType(ast::node(*statement, "annotation")));
      }
    }
  }
}

void ModuleEmitter::collectStaticModuleAssignments(
    const parser::Node &moduleNode, llvm::SmallVectorImpl<std::string> &names,
    llvm::SmallVectorImpl<mlir::Attribute> &values) const {
  mlir::Builder attrBuilder(&context);
  if (const auto *body = ast::nodeList(moduleNode, "body")) {
    for (const parser::NodePtr &statement : *body) {
      if (!statement || statement->kind != "Assign")
        continue;
      const auto *targets = ast::nodeList(*statement, "targets");
      if (!targets || targets->size() != 1 || !targets->front() ||
          targets->front()->kind != "Name")
        continue;
      names.push_back(std::string(ast::nameSpelling(*targets->front())));
      values.push_back(
          sourceExprAttr(attrBuilder, ast::node(*statement, "value")));
    }
  }
}

// The name an `if` test compares with None, and whether the BODY is where it
// is not None: `x is not None` and `x` say the body, `x is None` the else.
static std::optional<std::pair<std::string, bool>>
noneTestedName(const parser::Node *test) {
  if (!test)
    return std::nullopt;
  if (test->kind == "Name")
    return std::make_pair(std::string(ast::nameSpelling(*test)), true);
  if (test->kind != "Compare")
    return std::nullopt;
  const parser::Node *left = ast::node(*test, "left");
  const auto *ops = ast::nodeList(*test, "ops");
  const auto *comparators = ast::nodeList(*test, "comparators");
  if (!left || left->kind != "Name" || !ops || ops->size() != 1 ||
      !comparators || comparators->size() != 1 || !comparators->front() ||
      comparators->front()->kind != "Constant" ||
      !ast::isNoneField(*comparators->front(), "value"))
    return std::nullopt;
  const parser::NodePtr &op = ops->front();
  if (!op || (op->kind != "IsNot" && op->kind != "Is"))
    return std::nullopt;
  return std::make_pair(std::string(ast::nameSpelling(*left)),
                        op->kind == "IsNot");
}

void ModuleEmitter::collectClassFields(
    const parser::Node &classDef,
    llvm::SmallVectorImpl<std::string> &fieldNames,
    llvm::SmallVectorImpl<mlir::Type> &fieldTypes,
    bool includeAnnAssignDefaults, llvm::StringRef contractName,
    llvm::function_ref<void(llvm::ArrayRef<std::string>,
                            llvm::ArrayRef<mlir::Type>)>
        publishSoFar,
    bool methodsDeclareFields) {
  // The class as the TYPE SYSTEM knows it. An imported class is `lib.R` there
  // and `R` in its own ClassDef, and every question this walk asks -- what
  // `self` is, whether a target's owner is this class -- was asked by the
  // source spelling and answered nothing for the imported ones.
  std::string ownContractName =
      contractName.empty()
          ? std::string(ast::string(classDef, "name").value_or(""))
          : contractName.str();
  // Field names whose only assignment so far was an EMPTY container literal,
  // which has no element type of its own: the next assignment replaces it.
  llvm::StringSet<> provisionalFields;
  auto answeredOutsideTheFields = [&](llvm::StringRef name) {
    if (const auto *classBody = ast::nodeList(classDef, "body"))
      for (const parser::NodePtr &statement : *classBody) {
        if (!statement)
          continue;
        if ((statement->kind == "FunctionDef" ||
             statement->kind == "AsyncFunctionDef") &&
            llvm::StringRef(ast::nameSpelling(*statement)) == name)
          return true;
        llvm::SmallVector<const parser::Node *, 2> targets;
        if (statement->kind == "Assign") {
          if (const auto *list = ast::nodeList(*statement, "targets"))
            for (const parser::NodePtr &target : *list)
              targets.push_back(target.get());
        } else if (statement->kind == "AnnAssign") {
          targets.push_back(ast::node(*statement, "target"));
        }
        for (const parser::Node *target : targets)
          if (target && target->kind == "Name" &&
              llvm::StringRef(ast::nameSpelling(*target)) == name)
            return true;
      }
    auto mro = classMros.find(ownContractName);
    if (mro == classMros.end())
      return false;
    const py::protocols::Table &table = py::protocols::Table::get(context);
    for (const std::string &cls : mro->second) {
      if (cls == ownContractName)
        continue;
      mlir::Type base = types.contract(cls);
      if (lookupClassField(base, name) || resolveMroMethod(cls, name) ||
          lookupClassStaticAttr(base, name) ||
          table.resolveFieldContractWithEvidence(base, name) ||
          !table.methodContractCandidatesWithEvidence(base, name).empty())
        return true;
    }
    return false;
  };
  auto setField = [&](llvm::StringRef name, mlir::Type type,
                      bool overwriteExisting, const parser::Node &anchor,
                      bool provisional = false) {
    if (name.empty())
      return;
    if (!type) {
      // A method other than `__init__` may only refine what the constructor
      // declared, so an expression it cannot type there says nothing about a
      // field and must not be reported as one.
      if (collectFieldsRefineOnly)
        return;
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, anchor.range.start,
          "class field '" + name.str() +
              "' requires a statically inferred type"});
      return;
    }
    // ⭐ A LAMBDA FIELD'S RESULT IS AN INFERENCE, NOT A PROMISE. widenLiteral
    // is shallow, so `self.fn = lambda: 6` typed the field `Callable[[], 6]`
    // while the lambda's own callable is literal-widened before it is checked
    // against that field -- "lambda body is not compatible with its Callable
    // annotation", for a field that has no annotation at all. The named-def
    // spelling worked because a def's result comes from its return annotation.
    mlir::Type storedType = mlir::isa_and_nonnull<py::CallableType>(type)
                                ? widenInferredLiterals(type, types)
                                : types.widenLiteral(type);
    // ⭐ AN EMPTY LITERAL DOES NOT GET TO DECIDE THE ELEMENT TYPE. The first
    // assignment seen wins, and with regions walked that first one is often
    // the empty arm:
    //
    //     def __init__(self, xs: "list[int] | None" = None) -> None:
    //         if xs is None:
    //             self.xs = []          # list[object]
    //         else:
    //             self.xs = xs          # list[int] -- "not assignable"
    //
    // A `[]` has no element type of its own, which is the same rule the
    // generator body walk states for a rebinding; here it means an erased
    // argument yields to a real one on the same contract.
    auto refinesErasedArguments = [](mlir::Type existing, mlir::Type refined) {
      auto before = mlir::dyn_cast_if_present<py::ContractType>(existing);
      auto after = mlir::dyn_cast_if_present<py::ContractType>(refined);
      if (!before || !after ||
          before.getContractName() != after.getContractName() ||
          before.getArguments().size() != after.getArguments().size() ||
          before.getArguments().empty())
        return false;
      bool improves = false;
      for (auto [old, fresh] :
           llvm::zip(before.getArguments(), after.getArguments())) {
        if (old == fresh)
          continue;
        if (!py::isPyObjectType(old))
          return false;
        improves = true;
      }
      return improves;
    };
    // ⭐ `None` AND A VALUE ARE ONE FIELD, AND IT IS OPTIONAL. `self.n = None`
    // in `__init__` and `self.n = v` in a setter is how Python declares a slot
    // that starts empty, and "the first assignment wins" made the field
    // NoneType forever -- so the setter's store died in the lowering as
    // "attribute value 'builtins.int' is not assignable to field
    // '!py.literal<None>'", a sentence about the compiler for a program
    // CPython runs. The ANNOTATED spelling (`self.n: int | None = None`) has
    // always worked, and this makes the two agree.
    //
    // ⛔ The JOIN and not the later type: the slot really does hold None until
    // the setter runs, so typing it `int` would make a read before that a
    // silent wrong answer. `int | None` is what the annotation says and what
    // the reader then has to narrow.
    //
    // ⛔ Only against the None LITERAL. Two unrelated concrete types are a
    // program this compiler still refuses; widening those to a union here
    // would hide a real mistake behind a value nothing can use.
    auto optionalPair = [&](mlir::Type existing, mlir::Type fresh) {
      bool existingIsNone = existing == types.none();
      bool freshIsNone = fresh == types.none();
      if (existingIsNone == freshIsNone)
        return mlir::Type();
      mlir::Type payload = existingIsNone ? fresh : existing;
      if (!mlir::isa_and_nonnull<py::ContractType>(payload))
        return mlir::Type();
      // ⛔ `object` IS THE PRE-PASS SAYING IT DOES NOT KNOW, not a type to
      // widen a field to. A store this walk cannot type -- a local it did not
      // bind, a call it cannot resolve -- would otherwise turn a NoneType field
      // into `object | None`, which is a field whose reads carry nothing and
      // whose diagnostic names the wrong thing.
      if (py::isPyObjectType(payload))
        return mlir::Type();
      return types.join({payload, types.none()});
    };
    for (auto [index, existing] : llvm::enumerate(fieldNames)) {
      if (existing != name)
        continue;
      if (overwriteExisting ||
          refinesErasedArguments(fieldTypes[index], storedType) ||
          (provisionalFields.contains(name) && !provisional)) {
        fieldTypes[index] = storedType;
        if (!provisional)
          provisionalFields.erase(name);
        return;
      }
      if (mlir::Type optional = optionalPair(fieldTypes[index], storedType))
        fieldTypes[index] = optional;
      return;
    }
    // ⭐ A METHOD OTHER THAN `__init__` CREATES THE ATTRIBUTE TOO, as CPython's
    // does when it runs:
    //
    //     class MyTest(TestCase):
    //         def setUp(self) -> None:
    //             self.data = [1, 2, 3]
    //         def test_len(self) -> None:
    //             print(len(self.data))   # 'MyTest' object has no attribute 'data'
    //
    // -- every fixture of a unittest is written this way. The field's reads ask
    // whether construction stored it (`fieldAlwaysStoredBeforeRead`), and
    // construction does not, so one before the store raises AttributeError as
    // it does in CPython.
    //
    // ⛔ Not a name a base, a method or a class attribute already answers: an
    // instance attribute that SHADOWS one of those is a second storage for one
    // name, which this layout cannot hold, and redeclaring a base's field here
    // would retype its slot under the base's own methods.
    if (collectFieldsRefineOnly) {
      if (!methodsDeclareFields || answeredOutsideTheFields(name))
        return;
      classFieldsDeclaredOutsideInit[ownContractName].insert(name);
    }
    fieldNames.push_back(name.str());
    fieldTypes.push_back(storedType);
    if (provisional)
      provisionalFields.insert(name);
    if (publishSoFar)
      publishSoFar(fieldNames, fieldTypes);
  };

  auto collectInitArgTypes = [&](const parser::Node &method,
                                 llvm::StringMap<mlir::Type> &argTypes) {
    const parser::Node *arguments = ast::node(method, "args");
    if (!arguments)
      return;
    auto collectArgs = [&](llvm::StringRef fieldName) {
      if (const auto *args = ast::nodeList(*arguments, fieldName)) {
        for (const parser::NodePtr &arg : *args) {
          if (!arg)
            continue;
          llvm::StringRef name = ast::nameSpelling(*arg);
          if (name == "self")
            continue;
          if (const parser::Node *annotation = ast::node(*arg, "annotation"))
            argTypes[name] = types.annotationType(annotation);
        }
      }
    };
    collectArgs("posonlyargs");
    collectArgs("args");
    // ⭐ AND THE KEYWORD-ONLY AND PACKED ONES. A constructor's parameters are
    // where a field's type comes from, and three of the five kinds were not
    // read -- so `self.tag = tag` after `*, tag: str = "t"` and
    // `self.rest = rest` after `*rest: int` both declared `builtins.object`,
    // which is a field nothing can be read out of:
    //
    //     class A:
    //         def __init__(self, *rest: int) -> None:
    //             self.rest = rest
    //         def size(self) -> int:
    //             return len(self.rest)
    //     # builtins.object does not provide manifest method '__len__'
    //
    // ⛔ The packed ones carry their annotation's CONTAINER, not the
    // annotation: `*rest: int` binds `rest` to `tuple[int]` and `**kw: int` to
    // `dict[str, int]`, which is what the parameter itself is worth inside the
    // body.
    collectArgs("kwonlyargs");
    if (const parser::Node *vararg = ast::node(*arguments, "vararg"))
      if (const parser::Node *annotation = ast::node(*vararg, "annotation"))
        if (mlir::Type element = types.annotationType(annotation))
          argTypes[ast::nameSpelling(*vararg)] =
              types.contract("builtins.tuple", {element});
    if (const parser::Node *kwarg = ast::node(*arguments, "kwarg"))
      if (const parser::Node *annotation = ast::node(*kwarg, "annotation"))
        if (mlir::Type value = types.annotationType(annotation))
          argTypes[ast::nameSpelling(*kwarg)] =
              types.contract("builtins.dict", {types.strType(), value});
  };

  llvm::StringSet<> propertyNames = classPropertyNames(classDef);
  auto collectTarget = [&](const parser::Node &target, mlir::Type type,
                           bool provisional = false) {
    if (target.kind != "Attribute")
      return;
    const parser::Node *object = ast::node(target, "value");
    if (!object)
      return;
    // ⭐ AND THROUGH A LOCAL OF THIS CLASS, not only through `self`. A method
    // that walks its own structure assigns through the node it is standing on:
    //
    //     def add(self, w: str) -> None:
    //         node = self
    //         ...
    //         node.word = w        # the same field of the same class
    //
    // and reading only `self.<name>` left that field with whatever `__init__`
    // said, which for `self.word = None` is NoneType.
    //
    // ⛔ Only when the local's inferred type IS this class: through anything
    // else the target names another class's field, which this walk does not
    // own and must not decide.
    if (!ast::isName(*object, "self")) {
      mlir::Type ownerType = types.widenLiteral(types.inferExpr(object));
      // ⭐ A STORE INTO ANOTHER CLASS'S FIELD IS THAT CLASS'S EVIDENCE, and
      // this walk is the only one that sees it: `self.tail.nxt = node` inside
      // `Queue` is how a linked structure links, and `Cell.nxt` -- seeded
      // `None` in its own constructor -- stayed NoneType ("attribute value
      // 'Cell' is not assignable to field '!py.literal<None>'"). Recorded
      // here, applied once every class is declared
      // (`applyForeignFieldWrites`): the other class may be declared before
      // this one, as Cell is before Queue.
      // ⛔ Through an Optional owner too: a store through it only happens when
      // it is not None.
      mlir::Type ownerClass =
          mlir::isa_and_nonnull<py::UnionType>(ownerType)
              ? removeNoneFromType(ownerType, types)
              : ownerType;

      auto ownerContract =
          mlir::dyn_cast_if_present<py::ContractType>(ownerClass);
      bool ownClass =
          ownerContract && !ownContractName.empty() &&
          ownerContract.getContractName() == llvm::StringRef(ownContractName);
      if (!ownClass) {
        // ⭐ THE OWNER IS RESOLVED LATER, BY ITS PATH. While this walk runs,
        // `self.tail` is still the `None` its constructor seeded -- this
        // class's own stores widen it at the end of the walk, and a branch
        // that tested it against None is not narrowed here -- so the owner
        // is recorded as a root (a name, typed now) and the attributes from
        // it, and followed once every field is known.
        std::vector<std::string> path;
        const parser::Node *root = object;
        while (root && root->kind == "Attribute") {
          path.insert(path.begin(),
                      std::string(ast::string(*root, "attr").value_or("")));
          root = ast::node(*root, "value");
        }
        auto attr = ast::string(target, "attr");
        if (type && attr && root && root->kind == "Name")
          if (mlir::Type rootType =
                  types.widenLiteral(types.inferExpr(root)))
            foreignFieldWrites.push_back(ForeignFieldWrite{
                rootType, std::move(path), std::string(*attr),
                types.widenLiteral(type)});
        return;
      }
      if (object->kind != "Name" || collectFieldsRefineOnly == false)
        return;
    }
    if (auto attr = ast::string(target, "attr")) {
      // `self.<prop> = ...` runs the property setter; it declares no field.
      if (propertyNames.contains(*attr))
        return;
      setField(*attr, type, /*overwriteExisting=*/false, target, provisional);
    }
  };

  if (const auto *body = ast::nodeList(classDef, "body")) {
    for (const parser::NodePtr &statement : *body) {
      if (!statement || statement->kind != "AnnAssign")
        continue;
      const parser::Node *target = ast::node(*statement, "target");
      if (!target || target->kind != "Name" ||
          (ast::node(*statement, "value") && !includeAnnAssignDefaults))
        continue;
      setField(ast::nameSpelling(*target),
               types.annotationType(ast::node(*statement, "annotation")),
               /*overwriteExisting=*/true, *statement);
    }

    // ⭐ AND THE OTHER METHODS, for the fields `__init__` DECLARED. A field
    // seeded `self.n = None` and given its value by a setter is one slot
    // whose type is `int | None`, and walking only `__init__` left it
    // NoneType -- the setter's store then died in the lowering. The second
    // pass adds NO names (a field this class never declares keeps its own
    // diagnostic, "'C' object has no attribute 'n'") and refines only the
    // None-seeded ones, which is exactly what `optionalPair` above allows.
    // ⭐ A FIELD WHOSE ONLY ASSIGNMENT WAS AN EMPTY LITERAL takes its element
    // from the operations that FILL it, and those are usually in another
    // method:
    //
    //     class Bag:
    //         def __init__(self) -> None:
    //             self.xs = []
    //         def put(self, n: int) -> None:
    //             self.xs.append(n)
    //     # operand type 'builtins.object' does not match selected evidence
    //
    // -- which is how a container held by a class is written when nobody
    // annotates it. The scan that answers this for a LOCAL takes the container
    // as an attribute of a receiver now, so the same one answers here; it is
    // asked once per method, with that method's parameters in scope, and two
    // methods that disagree leave the field erased where it started.
    llvm::StringMap<mlir::Type> seededFields;
    llvm::StringSet<> seedDisagreed;
    auto erasedContainerKind = [&](mlir::Type type) -> llvm::StringRef {
      auto contract = mlir::dyn_cast_if_present<py::ContractType>(type);
      if (!contract)
        return {};
      llvm::StringRef contractName = contract.getContractName();
      if (contractName == "builtins.list")
        return "List";
      if (contractName == "builtins.dict")
        return "Dict";
      if (contractName == "builtins.set")
        return "Set";
      return {};
    };
    llvm::SmallVector<const parser::Node *, 4> fieldMethods;
    for (const parser::NodePtr &method : *body)
      if (method && ast::nameSpelling(*method) == "__init__")
        fieldMethods.push_back(method.get());
    unsigned initMethodCount = fieldMethods.size();
    for (const parser::NodePtr &method : *body)
      if (method &&
          (method->kind == "FunctionDef" || method->kind == "AsyncFunctionDef") &&
          ast::nameSpelling(*method) != "__init__")
        fieldMethods.push_back(method.get());
    for (auto [methodIndex, methodPtr] : llvm::enumerate(fieldMethods)) {
      const parser::Node *methodNode = methodPtr;
      llvm::SaveAndRestore<bool> refineOnly(collectFieldsRefineOnly,
                                            methodIndex >= initMethodCount);
      const parser::Node &method = *methodNode;
      // Names in scope for a field's declared type: the parameters, plus
      // locals bound earlier in the body. Why not a private name->type map
      // consulted only where the initializer is a bare `Name`: an initializer
      // that is an EXPRESSION over those names (`self.lineno = pos + 1`) is
      // handed to inferExpr with the names unbound, so it types as object --
      // and an object-typed field is not a diagnostic, it is a field whose
      // reads silently fail to carry wherever a known contract is required.
      // Binding the names into a real scope makes one path serve both.
      llvm::StringMap<mlir::Type> initArgTypes;
      collectInitArgTypes(method, initArgTypes);
      TypeSystem::Scope initScope = types.pushScope();
      // ⭐ A GENERIC METHOD'S TYPE PARAMETERS ARE TYPE VARIABLES HERE TOO, as
      // its signature reads them. Unbound, `E` in `Ctx[E](expected)` read as a
      // class named E, and the walk specialized Ctx at that fabrication --
      // a class the lowering then refused ("class value has no type object for
      // builtins.E") in a program that never called the method.
      if (const auto *typeParams = ast::nodeList(method, "type_params"))
        for (const parser::NodePtr &param : *typeParams) {
          std::optional<std::string_view> paramName =
              param ? ast::string(*param, "name") : std::nullopt;
          if (!paramName)
            continue;
          llvm::StringRef spelling(*paramName);
          if (param->kind == "ParamSpec")
            types.bindLocalSymbol(spelling,
                                  py::ParamSpecType::get(&context, spelling));
          else if (param->kind == "TypeVarTuple")
            types.bindLocalSymbol(
                spelling, py::TypeVarTupleType::get(&context, spelling));
          else
            types.bindLocalSymbol(spelling,
                                  py::TypeVarType::get(&context, spelling));
        }
      // ⭐ `self` IS THE CLASS, and this scope did not say so: it is the one
      // parameter `collectInitArgTypes` skips (it has no annotation to read),
      // so `other.nxt = self` -- how a linked structure links -- typed as
      // nothing and the field kept whatever `__init__` gave it.
      if (!ownContractName.empty())
        if (std::optional<mlir::Type> ownContract =
                types.lookupClass(ownContractName))
          if (*ownContract)
            types.bindLocalSymbol("self", *ownContract);
      // A name the pre-pass cannot type must SHADOW any outer binding of the
      // same name; resolving `self.f = x` to a module global that happens to
      // share the local's spelling would type the field off the wrong value.
      llvm::StringSet<> untypedLocals;
      for (const auto &arg : initArgTypes) {
        if (arg.second)
          types.bindLocalSymbol(arg.getKey(), arg.second);
        else
          untypedLocals.insert(arg.getKey());
      }
      // ⭐ A FIELD BUILT BY ONE OF THIS CLASS'S OWN METHODS. The methods are
      // registered AFTER the fields -- a method's signature is inferred against
      // them -- so `self.cache = self.compute()` typed as `object` and every
      // read of the field then carried nothing. An ANNOTATED method needs
      // nothing from the fields to say what it returns, so its annotation is
      // read straight off the ClassDef here.
      //
      // ⛔ Annotated only. An unannotated method's result is inferred FROM the
      // fields, which is the circle this deliberately does not enter.
      //
      // ⛔ A decorator other than staticmethod/classmethod means the name may
      // not be the function the annotation belongs to.
      auto annotatedSelfCallResult =
          [&](const parser::Node *value) -> mlir::Type {
        if (!value || value->kind != "Call")
          return {};
        const parser::Node *callee = ast::node(*value, "func");
        if (!callee || callee->kind != "Attribute")
          return {};
        const parser::Node *owner = ast::node(*callee, "value");
        if (!owner || !ast::isName(*owner, "self"))
          return {};
        auto called = ast::string(*callee, "attr");
        if (!called)
          return {};
        const auto *classBody = ast::nodeList(classDef, "body");
        if (!classBody)
          return {};
        for (const parser::NodePtr &member : *classBody) {
          if (!member || member->kind != "FunctionDef")
            continue;
          if (ast::nameSpelling(*member) != *called)
            continue;
          if (const auto *decorators = ast::nodeList(*member, "decorator_list"))
            for (const parser::NodePtr &decorator : *decorators) {
              std::string spelling = ast::qualifiedName(decorator.get());
              llvm::StringRef leaf = decoratorLeafName(spelling);
              if (leaf != "staticmethod" && leaf != "classmethod")
                return {};
            }
          if (const parser::Node *returns = ast::node(*member, "returns"))
            return types.annotationType(returns);
          return {};
        }
        // ⭐ AND A BASE'S METHOD, which is registered already: the base class
        // was emitted before this one, so its binding carries the result its
        // annotation promised. Without this a subclass whose field is built by
        // an inherited factory typed that field `object`.
        for (const std::string &ancestor : classMros[contractName])
          if (ancestor != contractName)
            if (auto bindings = classMethodBindings.find(ancestor);
                bindings != classMethodBindings.end())
              if (auto found = bindings->second.find(*called);
                  found != bindings->second.end())
                return found->second.bodySignature.resultType;
        return {};
      };
      auto valueTypeOf = [&](const parser::Node *value) -> mlir::Type {
        if (value && value->kind == "Name" &&
            untypedLocals.contains(ast::nameSpelling(*value)))
          return {};
        if (mlir::Type annotated = annotatedSelfCallResult(value))
          return annotated;
        return types.inferExpr(value);
      };
      // ⭐ AN EMPTY LITERAL TAKES THE ELEMENT TYPE THE NAME ALREADY HAS. The
      // walk binds locals as it goes so a field can be typed from one, and it
      // bound them from the region it found them in -- so the branch that
      // supplies the default overwrote the parameter's type with an erased
      // one, and the field then did not fit the value the emitter builds:
      //
      //     def __init__(self, xs: "list[int] | None" = None) -> None:
      //         if xs is None:
      //             xs = []            # bound list[object] over list[int]
      //         self.xs = xs
      //     # attribute value 'list[int]' is not assignable to field
      //     # 'list[object]'
      //
      // The emitter's own join answers `list[int]` here (`return xs` from the
      // same body compiles), so this is the walk disagreeing with what is
      // built rather than a question with two answers. It is the same rule
      // `setField` states one screen up, asked at the rebinding instead of at
      // the field.
      //
      // ⛔ Only a SAME-CONTRACT member with no erased argument of its own: an
      // `xs: "int | None"` that a branch rebinds to `[]` has nothing to take
      // an element type from and keeps the erased one it had.
      auto refineErasedRebinding = [&](llvm::StringRef name,
                                       mlir::Type erased) -> mlir::Type {
        auto erasedContract =
            mlir::dyn_cast_if_present<py::ContractType>(erased);
        if (!erasedContract || erasedContract.getArguments().empty())
          return erased;
        std::optional<mlir::Type> existing = types.lookupSymbol(name);
        if (!existing || !*existing)
          return erased;
        llvm::SmallVector<mlir::Type, 4> candidates;
        if (auto unionType = mlir::dyn_cast<py::UnionType>(*existing))
          candidates.assign(unionType.getMemberTypes().begin(),
                            unionType.getMemberTypes().end());
        else
          candidates.push_back(*existing);
        for (mlir::Type candidate : candidates) {
          auto contract = mlir::dyn_cast_if_present<py::ContractType>(
              types.widenLiteral(candidate));
          if (!contract ||
              contract.getContractName() !=
                  erasedContract.getContractName() ||
              contract.getArguments().size() !=
                  erasedContract.getArguments().size())
            continue;
          if (llvm::any_of(contract.getArguments(), [](mlir::Type argument) {
                return py::isPyObjectType(argument);
              }))
            continue;
          return contract;
        }
        return erased;
      };
      auto bindInitLocal = [&](const parser::Node *target, mlir::Type type) {
        if (!target || target->kind != "Name")
          return;
        llvm::StringRef name = ast::nameSpelling(*target);
        if (!type) {
          untypedLocals.insert(name);
          return;
        }
        untypedLocals.erase(name);
        types.bindLocalSymbol(name, type);
      };
      // ⭐ A FIELD ASSIGNED INSIDE A REGION IS STILL A FIELD. The walk read
      // only `__init__`'s top-level statements, so the two-branch constructor
      // every optional field is written as declared nothing:
      //
      //     class C:
      //         def __init__(self, flag: bool) -> None:
      //             if flag:
      //                 self.n = 1
      //             else:
      //                 self.n = 2
      //     C(True).n      # 'C' object has no attribute 'n'
      //
      // The `for`, `while`, `try` and `with` spellings were the same. Only an
      // UNCONDITIONAL assignment before the region made the field exist, which
      // is why `self.n = 0` then `if flag: self.n = 1` worked.
      //
      // ⛔ Nested defs and classes are not entered: a `self` inside one is a
      // different scope's, and the first type seen still wins
      // (`overwriteExisting=false`), so two branches that disagree take the
      // first and the other store is a type error where it is written.
      // ⛔ A TUPLE target takes the element type APART: `for i, w in
      // enumerate(xs)` binds two names from one tuple element, and binding the
      // tuple to each of them would type both fields wrong rather than leave
      // them erased.
      auto bindLoopTarget = [&](const parser::Node *target,
                                const parser::Node *iter) {
        if (!target || !iter)
          return;
        mlir::Type element = types.iterationElementType(iter);
        if (!element)
          return;
        if (target->kind == "Name") {
          bindInitLocal(target, types.widenLiteral(element));
          return;
        }
        if (target->kind != "Tuple" && target->kind != "List")
          return;
        const auto *parts = ast::nodeList(*target, "elts");
        if (!parts)
          return;
        auto contract =
            mlir::dyn_cast_if_present<py::ContractType>(types.widenLiteral(element));
        if (!contract)
          return;
        llvm::ArrayRef<mlir::Type> arguments = contract.getArguments();
        if (contract.getContractName() == "builtins.tuple" &&
            arguments.size() == parts->size()) {
          for (auto [part, argument] : llvm::zip(*parts, arguments))
            if (part && part->kind == "Name")
              bindInitLocal(part.get(), types.widenLiteral(argument));
          return;
        }
        if (arguments.size() == 1)
          for (const parser::NodePtr &part : *parts)
            if (part && part->kind == "Name")
              bindInitLocal(part.get(), types.widenLiteral(arguments.front()));
      };
      std::function<void(const std::vector<parser::NodePtr> *)> walkInitBody =
          [&](const std::vector<parser::NodePtr> *stmts) {
            if (!stmts)
              return;
            for (const parser::NodePtr &stmt : *stmts) {
              if (!stmt)
                continue;
              if (stmt->kind == "AnnAssign") {
                const parser::Node *target = ast::node(*stmt, "target");
                if (!target)
                  continue;
                mlir::Type declared =
                    types.annotationType(ast::node(*stmt, "annotation"));
                bindInitLocal(target, declared);
                collectTarget(*target, declared);
                continue;
              }
              if (stmt->kind == "Assign") {
                const parser::Node *value = ast::node(*stmt, "value");
                mlir::Type valueType = valueTypeOf(value);
                // ⭐ AN EMPTY LITERAL DOES NOT GET TO DECIDE THE FIELD. `[]`
                // has no element type of its own, and with regions walked it
                // is often the FIRST assignment seen -- the branch that
                // supplies a real one then does not fit the field it named:
                //
                //     if xs is None:
                //         self.xs = []       # list[object]
                //     else:
                //         self.xs = xs       # "not assignable to field"
                //
                // The same rule the generator body walk states for a
                // rebinding, applied where the field is declared.
                //
                // ⛔ Asked through the canonical predicate and not a local
                // node-kind test, which is what stood here and what left
                // `xs = list()`, `d = dict()` and `s = set()` outside the rule
                // that `[]` and `{}` were inside. The zero-argument
                // constructor is the same empty container and the same absence
                // of an element type -- and `set()` is the only spelling there
                // is for one.
                bool provisional = isEmptyContainerExpression(value);
                if (const auto *targets = ast::nodeList(*stmt, "targets"))
                  for (const parser::NodePtr &target : *targets) {
                    if (!target)
                      continue;
                    mlir::Type boundType =
                        provisional && target->kind == "Name"
                            ? refineErasedRebinding(
                                  ast::nameSpelling(*target), valueType)
                            : valueType;
                    bindInitLocal(&*target, boundType);
                    collectTarget(*target, boundType, provisional);
                  }
                continue;
              }
              if (stmt->kind == "FunctionDef" ||
                  stmt->kind == "AsyncFunctionDef" || stmt->kind == "ClassDef")
                continue;
              // ⭐ AND THE LOOP TARGET IS IN SCOPE FOR THE BODY. Fields
              // assigned inside a region became fields when this walk learned
              // to enter one, and the commonest thing such an assignment
              // mentions is the loop's own target -- which nothing had bound,
              // so the field took the erased top:
              //
              //     class C:
              //         def __init__(self) -> None:
              //             for i in range(3):
              //                 self.n = i
              //     print(C().n + 1)
              //     # static type builtins.object does not provide '__add__'
              //
              // Its type is the iterable's element type, which is known here;
              // the seed scan binds it the same way for the same reason.
              if (stmt->kind == "For" || stmt->kind == "AsyncFor")
                bindLoopTarget(ast::node(*stmt, "target"),
                               ast::node(*stmt, "iter"));
              // ⭐ AND A TEST AGAINST None NARROWS THE BRANCH IT GUARDS, as
              // the emitter's own flow typing does, and the name after the
              // `if` is the JOIN of what each way through leaves it. Read as
              // the parameter's Optional type, a guarded store made the field
              // `int | None` and every read was refused for a None it could
              // never hold:
              //
              //     limit = 5
              //     def __init__(self, limit: int | None = None) -> None:
              //         if limit is not None:
              //             self.limit = limit      # typed int | None
              //
              // and read as the last assignment the walk saw, the default
              // idiom typed the field by the default alone, and a StringIO
              // was then stored as the TextIOWrapper the default is:
              //
              //     if stream is None:
              //         stream = sys.stderr
              //     self.stream = stream            # TextIOWrapper only
              if (stmt->kind == "If") {
                std::optional<std::pair<std::string, bool>> tested =
                    noneTestedName(ast::node(*stmt, "test"));
                std::optional<mlir::Type> before;
                if (tested)
                  before = types.lookupSymbol(tested->first);
                bool narrows = before && *before &&
                               mlir::isa<py::UnionType>(*before);
                llvm::SmallVector<mlir::Type, 2> leaves;
                for (llvm::StringRef region : {"body", "orelse"}) {
                  // ⛔ The None side is not entered AS None: the rule that
                  // gives `xs = []` its element type reads the name's type
                  // before the rebinding, and None has none to give. It
                  // counts as None only if it leaves the name unbound.
                  bool notNoneSide =
                      narrows && tested->second == (region == "body");
                  mlir::Type entering =
                      !narrows ? mlir::Type()
                      : notNoneSide ? removeNoneFromType(*before, types)
                                    : *before;
                  if (narrows)
                    types.bindLocalSymbol(tested->first, entering);
                  if (const auto *nested = ast::nodeList(*stmt, region))
                    walkInitBody(nested);
                  if (narrows) {
                    std::optional<mlir::Type> leaving =
                        types.lookupSymbol(tested->first);
                    mlir::Type left = leaving && *leaving ? *leaving : entering;
                    if (!notNoneSide && left == *before)
                      left = types.none();
                    leaves.push_back(left);
                    types.bindLocalSymbol(tested->first, *before);
                  }
                }
                if (narrows)
                  if (mlir::Type joined = types.join(leaves))
                    types.bindLocalSymbol(tested->first, joined);
                continue;
              }
              for (llvm::StringRef region :
                   {"body", "orelse", "finalbody", "handlers"})
                if (const auto *nested = ast::nodeList(*stmt, region))
                  walkInitBody(nested);
            }
          };
      walkInitBody(ast::nodeList(method, "body"));

      // Asked with this method's scope still pushed, so `self.xs.append(n)`
      // can read `n`.
      for (auto [fieldIndex, fieldName] : llvm::enumerate(fieldNames)) {
        if (!provisionalFields.contains(fieldName))
          continue;
        llvm::StringRef literalKind = erasedContainerKind(fieldTypes[fieldIndex]);
        if (literalKind.empty())
          continue;
        SuiteCursor cursor{ast::nodeList(method, "body"), 0};
        mlir::Type seeded =
            emptyLiteralSeedTypeIn(types, fieldName, literalKind, cursor,
                                   /*localSymbols=*/nullptr, /*depth=*/0,
                                   /*subscriptDepth=*/0, /*receiver=*/"self");
        if (!seeded)
          continue;
        auto found = seededFields.find(fieldName);
        if (found == seededFields.end())
          seededFields[fieldName] = seeded;
        else if (found->second != seeded)
          seedDisagreed.insert(fieldName);
      }
    }

    for (auto [fieldIndex, fieldName] : llvm::enumerate(fieldNames)) {
      if (!provisionalFields.contains(fieldName) ||
          seedDisagreed.contains(fieldName))
        continue;
      auto found = seededFields.find(fieldName);
      if (found != seededFields.end())
        fieldTypes[fieldIndex] = found->second;
    }
    if (publishSoFar)
      publishSoFar(fieldNames, fieldTypes);
  }
}

Value ModuleEmitter::emitSuperExceptionInit(const parser::Node &expr,
                                            Value receiver,
                                            llvm::StringRef baseContract) {
  llvm::SmallVector<Value, 2> positional;
  llvm::SmallVector<mlir::Type, 2> positionalTypes;
  if (const auto *args = ast::nodeList(expr, "args")) {
    for (const parser::NodePtr &arg : *args) {
      if (arg && arg->kind == "Starred") {
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, arg->range.start,
            "starred arguments are not supported for exception __init__"});
        return emitNone(expr);
      }
      positional.push_back(emitExpr(arg.get()));
      positionalTypes.push_back(positional.back().type);
    }
  }
  if (const auto *keywords = ast::nodeList(expr, "keywords");
      keywords && !keywords->empty()) {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, expr.range.start,
        "keyword arguments are not supported for exception __init__"});
    return emitNone(expr);
  }
  // Zero arguments: __new__ already made the empty message (the same
  // shortcut instantiation takes for no-arg builtin exceptions).
  if (positional.empty())
    return emitNone(expr);
  if (positional.size() > 1) {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, expr.range.start,
        "exception __init__ supports at most one message argument yet"});
    return emitNone(expr);
  }
  // Inference runs against the TAXONOMY ANCESTOR, not the receiver: the
  // receiver's own class table would match the class's OWN __init__ (the
  // method this super() call sits inside) when the argument shapes happen to
  // coincide, and misses entirely when they don't (a user exception whose
  // __init__ takes non-str parameters resolved no manifest __init__ at all).
  mlir::Type ancestorType = types.contract(baseContract);
  CallInferenceResult inference = types.inferMethodCallWithEvidence(
      ancestorType, "__init__", positionalTypes);
  if (!requireStaticEvidence(expr, inference))
    return emitNone(expr);
  Value posPack = emitPack(positional);
  Value namePack = emitPack({});
  Value valuePack = emitPack({});
  auto initOp = py::InitOp::create(
      builder, loc(expr), types.none(),
      mlir::FlatSymbolRefAttr::get(&context, "__init__"),
      callProtocolFor(inference), receiver.value, posPack.value,
      namePack.value, valuePack.value);
  initOp->setAttr("ly.constructor.init_kind", builder.getStringAttr("instance"));
  if (auto contract = mlir::dyn_cast_if_present<py::ContractType>(receiver.type))
    initOp->setAttr("ly.constructor.owner",
                    builder.getStringAttr(contract.getContractName()));
  return emitNone(expr);
}

std::optional<Value>
ModuleEmitter::tryEmitSuperCall(const parser::Node &expr,
                                const parser::Node *calleeNode) {
  auto isSuperName = [&](const parser::Node *node) {
    return node && node->kind == "Name" && ast::nameSpelling(*node) == "super";
  };
  auto reject = [&](const std::string &message) -> std::optional<Value> {
    diagnostics.push_back(
        parser::Diagnostic{parser::Severity::Error, expr.range.start, message});
    return emitNone(expr);
  };

  if (isSuperName(calleeNode))
    return reject("super() is only supported as a method-call receiver "
                  "(super().method(...))");
  if (!calleeNode || calleeNode->kind != "Attribute")
    return std::nullopt;
  const parser::Node *superCall = ast::node(*calleeNode, "value");
  if (!superCall || superCall->kind != "Call" ||
      !isSuperName(ast::node(*superCall, "func")))
    return std::nullopt;

  std::optional<std::string_view> methodName = ast::string(*calleeNode, "attr");
  if (!methodName)
    return reject("super() attribute must be a plain method name");

  // Resolve the start-class and receiver: the zero-argument form reads the
  // enclosing method's defining class and receiver parameter; the two-argument
  // form must name a statically known class and receiver expression.
  std::string startClass;
  Value receiver;
  const auto *superArgs = ast::nodeList(*superCall, "args");
  std::size_t superArgCount = superArgs ? superArgs->size() : 0;
  if (superArgCount == 0) {
    if (superContexts.empty())
      return reject("zero-argument super() requires an enclosing class "
                    "method body");
    const SuperContext &context = superContexts.back();
    startClass = context.definingClass;
    auto self = values.find(context.selfName);
    if (self == values.end() || !self->second.value)
      return reject("zero-argument super() requires the enclosing method's "
                    "receiver parameter to be in scope");
    receiver = self->second;
  } else if (superArgCount == 2) {
    const parser::Node *classArg = (*superArgs)[0].get();
    std::string classSpelling = ast::qualifiedName(classArg);
    if (classSpelling.empty() && classArg && classArg->kind == "Name")
      classSpelling = std::string(ast::nameSpelling(*classArg));
    startClass = canonicalClassName(classSpelling);
    if (startClass.empty() || !classMros.count(startClass))
      return reject("super(C, obj) requires C to name a statically known "
                    "source class");
    receiver = emitExpr((*superArgs)[1].get());
  } else {
    return reject("super() takes zero or two arguments");
  }

  auto receiverContract =
      mlir::dyn_cast_if_present<py::ContractType>(receiver.type);
  if (!receiverContract)
    return reject("super() receiver must be a class instance (classmethod "
                  "super() is not supported yet)");
  llvm::StringRef receiverClass = receiverContract.getContractName();
  llvm::ArrayRef<std::string> mro = classMro(receiverClass);
  if (mro.empty())
    return reject("super() receiver class has no static MRO");
  if (!llvm::is_contained(mro, startClass))
    return reject("super(): class '" +
                  py::contracts::displayClassNameForContract(startClass) +
                  "' is not in the receiver's MRO");

  if (std::optional<MethodBinding> method =
          resolveMroMethod(receiverClass, *methodName, startClass)) {
    if (method->kind != "instance")
      return reject("super() only resolves instance methods yet");
    return emitInlineMethodCall(expr, receiver, *method);
  }

  // No source-class provider after startClass: the next provider is a
  // manifest class. object.__init__ is a no-op; anything else is loud.
  bool active = false;
  for (const std::string &cls : mro) {
    if (!active) {
      active = cls == startClass;
      continue;
    }
    if (classMros.count(cls))
      continue;
    if (cls == "builtins.object" && *methodName == "__init__") {
      const auto *callArgs = ast::nodeList(expr, "args");
      if (callArgs && !callArgs->empty())
        return reject("object.__init__() takes no arguments");
      return emitNone(expr);
    }
    if (taxonomyEntryForContract(cls) && *methodName == "__init__")
      return emitSuperExceptionInit(expr, receiver, cls);
    return reject("super(): '" + std::string(*methodName) +
                  "' resolves to builtin base '" +
                  py::contracts::displayClassNameForContract(cls) +
                  "', which super() cannot call yet");
  }
  return reject("'super' object has no attribute '" +
                std::string(*methodName) + "'");
}

std::optional<MethodBinding>
ModuleEmitter::resolveClassDunder(const parser::Node &anchor, Value receiver,
                                  llvm::StringRef dunder, bool &refused) {
  refused = refuseUnresolvableDispatch(anchor, receiver, dunder);
  if (refused)
    return std::nullopt;
  return lookupClassMethod(types.widenLiteral(receiver.type), dunder);
}

std::optional<Value>
ModuleEmitter::tryEmitClassDunder(const parser::Node &anchor, Value receiver,
                                  llvm::StringRef dunder,
                                  llvm::ArrayRef<Value> positional,
                                  bool *refusedOut) {
  // An overridden dunder on a base-typed receiver goes through the same
  // synthesized dispatcher a named call does: `len(bag)` and `bag.__len__()`
  // are one method, and answering only the second would leave the operator
  // spelling refused for exactly the programs the dispatch exists for.
  if (dispatchIsUnresolvable(receiver, dunder, /*receiverNode=*/nullptr,
                             /*throughSuper=*/false))
    if (std::optional<Value> dispatched =
            tryEmitVirtualDispatchWithValues(anchor, receiver, dunder,
                                             positional)) {
      if (refusedOut)
        *refusedOut = false;
      return dispatched;
    }
  bool refused = false;
  std::optional<MethodBinding> method =
      resolveClassDunder(anchor, receiver, dunder, refused);
  if (refusedOut)
    *refusedOut = refused;
  if (refused)
    return emitNone(anchor);
  if (!method)
    return std::nullopt;
  // ⭐ A SUSPENDABLE DUNDER IS CALLED, NOT INLINED. `__aenter__`,
  // `__aexit__` and `__await__` are coroutine or generator functions, and
  // calling one makes the coroutine or generator: its body runs when it is
  // awaited or iterated. Inlining substituted the body's own result -- an
  // `async with` ran `__aenter__` to completion at the call -- which is the
  // named method call's reason for taking the symbol too.
  if ((method->async || method->bodySignature.isGeneratorFunction) &&
      method->kind == "instance" && !method->symbolName.empty() &&
      methodBindingBindsReceiver(*method) &&
      !mlir::isa<py::TypeType>(receiver.type))
    if (auto callable = mlir::dyn_cast_if_present<py::CallableType>(
            method->signature.publicCallable)) {
      Value callee = emitBindingRef(anchor, method->symbolName, callable);
      CallOperands operands;
      operands.positional.push_back(receiver);
      operands.positionalTypes.push_back(receiver.type);
      operands.positionalUnpacked.push_back(0);
      for (const Value &argument : positional) {
        operands.positional.push_back(argument);
        operands.positionalTypes.push_back(argument.type);
        operands.positionalUnpacked.push_back(0);
      }
      return emitCallableDispatch(anchor, callee, operands);
    }
  return emitInlineOperatorCall(anchor, receiver, *method, positional);
}

std::optional<Value>
ModuleEmitter::tryEmitClassDunderCall(const parser::Node &call, Value receiver,
                                      llvm::StringRef dunder) {
  bool refused = false;
  std::optional<MethodBinding> method =
      resolveClassDunder(call, receiver, dunder, refused);
  if (refused)
    return emitNone(call);
  if (!method)
    return std::nullopt;
  return emitInlineMethodCall(call, receiver, *method);
}

Value ModuleEmitter::emitInlineOperatorCall(const parser::Node &anchor,
                                            Value receiver,
                                            const MethodBinding &method,
                                            llvm::ArrayRef<Value> positional) {
  if (!method.method)
    return emitNone(anchor);
  Value descriptorReceiver = emitDescriptorReceiver(anchor, receiver, method);
  bool bindReceiver = methodBindingBindsReceiver(method);
  if (method.kind == "instance" && mlir::isa<py::TypeType>(receiver.type))
    bindReceiver = false;
  KeywordArguments keywords;
  return emitInlineMethodBody(anchor, descriptorReceiver, bindReceiver, method,
                              positional, keywords);
}

// ⛔ The CONSTRUCTOR spelling of an empty container needs the coercion as well
// as the expectation, and a literal does not. `set()` is a call whose result
// type comes from the callee contract, so the expectation reaches the emission
// and the value still comes back `set[object]` -- the same reason AnnAssign has
// always coerced. Only an erased container of the SAME contract is adopted: a
// genuinely different type is the declared-parameter check's business.
Value ModuleEmitter::adoptDeclaredContainer(Value value, mlir::Type declared,
                                            const parser::Node &anchor) {
  auto actual = mlir::dyn_cast_if_present<py::ContractType>(
      types.widenLiteral(value.type));
  auto expected = mlir::dyn_cast_if_present<py::ContractType>(
      types.widenLiteral(declared));
  // ⛔ Not to a type that still names a type parameter: `[]` passed to a
  // generic method's `out: list[T]` became a `list[T]` value, which no
  // solution's `list[int]` accepts. It adopts the solution's type once T is
  // solved.
  if (!actual || !expected ||
      actual.getContractName() != expected.getContractName() ||
      actual.getArguments().empty() ||
      actual.getArguments().size() != expected.getArguments().size() ||
      unboundStaticParameterCount(expected) != 0)
    return value;
  if (!llvm::all_of(actual.getArguments(), [&](mlir::Type argument) {
        return types.widenLiteral(argument) == types.object();
      }))
    return value;
  return coerceValue(value, declared, anchor);
}

// ⭐ A DISPLAY LITERAL PASSED TO A GENERIC PARAMETER IS BUILT AS WHAT T WAS
// SOLVED TO. `assertEqual(m.groups(), ('a', None))` solves T from both
// arguments as `tuple[str | None, ...]`, but the literal was emitted before
// that, against T itself, so it stayed `tuple[str, None]` and was refused
// against its own formal -- while `x: tuple[str | None, ...] = ('a', None)`
// one line up retypes the same literal at its annotation
// (emitContainerLiteral's coercion). The solver accepting the call is the
// check that its elements fit.
// ⛔ Only a FRESH display (`[...]`, `(...)`, `{...}` with no starred element):
// any other container already has a type of its own that other names see.
Value ModuleEmitter::adoptSolvedDisplay(Value value, mlir::Type declared,
                                        const parser::Node *node) {
  if (!node || !declared || value.type == declared ||
      unboundStaticParameterCount(declared) != 0)
    return value;
  llvm::StringRef wanted;
  if (node->kind == "List")
    wanted = "builtins.list";
  else if (node->kind == "Tuple")
    wanted = "builtins.tuple";
  else if (node->kind == "Dict")
    wanted = "builtins.dict";
  else if (node->kind == "Set")
    wanted = "builtins.set";
  if (wanted.empty())
    return value;
  if (const auto *elements = ast::nodeList(*node, "elts"))
    for (const parser::NodePtr &element : *elements)
      if (!element || element->kind == "Starred")
        return value;
  if (const auto *keys = ast::nodeList(*node, "keys"))
    for (const parser::NodePtr &key : *keys)
      if (!key)
        return value;
  auto actual = mlir::dyn_cast_if_present<py::ContractType>(
      types.widenLiteral(value.type));
  auto expected = mlir::dyn_cast_if_present<py::ContractType>(declared);
  if (!actual || !expected || actual.getContractName() != wanted ||
      expected.getContractName() != wanted)
    return value;
  return coerceValue(value, declared, *node);
}

// The parameters an inlined method's call fills, its receiver's slot excluded,
// for a call that collects a starred argument or a `**` mapping of unknown size.
ModuleEmitter::DynamicUnpackTarget
ModuleEmitter::dynamicUnpackTargetForMethod(const MethodBinding &method) {
  const FunctionSignature &signature =
      method.bodySignature.callable ? method.bodySignature : method.signature;
  DynamicUnpackTarget target;
  std::size_t firstSlot = methodBindingBindsReceiver(method) ? 1 : 0;
  for (std::size_t slot = firstSlot; slot < signature.positionalNames.size();
       ++slot)
    target.positionalNames.push_back(signature.positionalNames[slot]);
  target.positionalOnlyCount =
      signature.positionalOnlyCount > firstSlot
          ? signature.positionalOnlyCount - static_cast<unsigned>(firstSlot)
          : 0;
  target.kwOnlyNames.assign(signature.kwOnlyNames.begin(),
                            signature.kwOnlyNames.end());
  if (signature.varargName)
    target.varargType = signature.varargType;
  if (signature.kwargName)
    target.kwargType = signature.kwargType;
  if (auto definition = pythonDefinitionNames.find(method.method);
      definition != pythonDefinitionNames.end()) {
    target.module = definition->second.module;
    target.qualname = definition->second.qualname;
  }
  return target;
}

Value ModuleEmitter::emitInlineMethodCall(const parser::Node &expr,
                                          Value receiver,
                                          const MethodBinding &method) {
  if (!method.method)
    return emitNone(expr);

  // ⭐ AN ARGUMENT IS EMITTED AGAINST THE PARAMETER IT FILLS, the same way a
  // free function's and a constructor's are. Without the expectation an empty
  // literal came out `list[object]` and the declared-parameter check refused
  // the call it was written for:
  //
  //     class C:
  //         def take(self, xs: list[int]) -> int: ...
  //     C().take([])
  //     # argument 'xs' of 'take' is declared list[int] and this call gives it
  //     # list[object]
  //
  // `def f(xs: list[int])` called as `f([])` was always fine, because
  // `emitCallOperands` distributes the callee's positional types there. Slot 0
  // is the receiver, so the written arguments start one past it.
  const FunctionSignature &argumentSig =
      method.bodySignature.callable ? method.bodySignature : method.signature;
  bool receiverTakesSlotZero = methodBindingBindsReceiver(method);
  auto declaredPositional = [&](std::size_t index) -> mlir::Type {
    std::size_t slot = receiverTakesSlotZero ? index + 1 : index;
    if (slot >= argumentSig.positionalTypes.size())
      return {};
    mlir::Type declared = argumentSig.positionalTypes[slot];
    return declared && py::isStaticTypeParameter(declared) ? mlir::Type()
                                                           : declared;
  };
  auto declaredKeyword = [&](llvm::StringRef name) -> mlir::Type {
    for (auto [index, parameter] :
         llvm::enumerate(argumentSig.positionalNames))
      if (parameter == name && index < argumentSig.positionalTypes.size()) {
        mlir::Type declared = argumentSig.positionalTypes[index];
        return declared && py::isStaticTypeParameter(declared) ? mlir::Type()
                                                               : declared;
      }
    for (auto [index, parameter] : llvm::enumerate(argumentSig.kwOnlyNames))
      if (parameter == name && index < argumentSig.kwOnlyTypes.size()) {
        mlir::Type declared = argumentSig.kwOnlyTypes[index];
        return declared && py::isStaticTypeParameter(declared) ? mlir::Type()
                                                               : declared;
      }
    return {};
  };

  llvm::SmallVector<Value, 8> positional;
  // A starred sequence, and a `**` mapping, whose size only the run time
  // knows: collected into the method's *args and **kwargs below
  // (emitDynamicUnpack), as a call of a function collects them.
  std::optional<Value> dynamicTail;
  bool dynamicMapping = false;
  if (const auto *args = ast::nodeList(expr, "args")) {
    for (const parser::NodePtr &arg : *args) {
      if (dynamicTail) {
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, arg->range.start,
            "a starred argument whose length is known only at run time must "
            "be the last positional argument"});
        continue;
      }
      if (arg && arg->kind == "Starred") {
        // ⭐ A STARRED ARGUMENT WHOSE VALUES ARE KNOWN IS THOSE VALUES: an
        // inlined method's own `*args` (the values its call gave), or a tuple
        // whose type says how many it holds, read by position. One whose
        // arity only the run time knows is the method's *args.
        Value starred = emitExpr(ast::node(*arg, "value"));
        if (auto pack = staticPositionalPacks.find(starred.value);
            pack != staticPositionalPacks.end()) {
          positional.append(pack->second.begin(), pack->second.end());
          continue;
        }
        auto tuple = mlir::dyn_cast_if_present<py::ContractType>(
            types.widenLiteral(starred.type));
        if (tuple && tuple.getContractName() == "builtins.tuple" &&
            tuple.getArguments().size() >= 2) {
          for (unsigned index = 0; index < tuple.getArguments().size();
               ++index) {
            mlir::Type indexType = types.literal(std::to_string(index));
            Value indexValue{
                py::IntConstantOp::create(builder, loc(*arg), indexType,
                                          builder.getStringAttr(
                                              std::to_string(index)))
                    .getResult(),
                indexType};
            CallInferenceResult inference = types.inferMethodCallWithEvidence(
                starred.type, "__getitem__", {indexValue.type});
            if (!requireStaticEvidence(*arg, inference))
              break;
            auto item = py::GetItemOp::create(
                builder, loc(*arg), inference.resultType,
                mlir::FlatSymbolRefAttr::get(&context, "__getitem__"),
                callProtocolFor(inference), starred.value, indexValue.value);
            positional.push_back(Value{item.getResult(), inference.resultType});
          }
          continue;
        }
        if (auto sequence = mlir::dyn_cast_if_present<py::ContractType>(
                types.widenLiteral(starred.type));
            sequence && sequence.getArguments().size() == 1 &&
            (sequence.getContractName() == "builtins.tuple" ||
             sequence.getContractName() == "builtins.list")) {
          dynamicTail = starred;
          continue;
        }
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, arg->range.start,
            "starred arguments are not supported for inlined class methods "
            "unless they are a tuple or a list"});
        continue;
      }
      mlir::Type expected = declaredPositional(positional.size());
      positional.push_back(expected
                               ? adoptDeclaredContainer(
                                     emitExprExpected(arg.get(), expected),
                                     expected, *arg)
                               : emitExpr(arg.get()));
    }
  }

  KeywordArguments keywords;
  if (const auto *keywordNodes = ast::nodeList(expr, "keywords")) {
    for (const parser::NodePtr &keyword : *keywordNodes) {
      if (auto name = ast::string(*keyword, "arg")) {
        mlir::Type expected = declaredKeyword(*name);
        keywords[*name] =
            expected ? adoptDeclaredContainer(
                           emitExprExpected(ast::node(*keyword, "value"),
                                            expected),
                           expected, *keyword)
                     : emitExpr(ast::node(*keyword, "value"));
        continue;
      }
      // The inlined caller's own `**kwargs`, forwarded by name.
      Value mapping = emitExpr(ast::node(*keyword, "value"));
      if (auto pack = staticKeywordPacks.find(mapping.value);
          pack != staticKeywordPacks.end()) {
        for (const auto &[packedName, packedValue] : pack->second)
          keywords[packedName] = packedValue;
        continue;
      }
      if (py::callableUnpackedKeywordValueType(
              types.widenLiteral(mapping.type)) &&
          !dynamicMapping) {
        // Under the name "", in its place among the keywords.
        keywords[""] = mapping;
        dynamicMapping = true;
        continue;
      }
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, keyword->range.start,
          dynamicMapping
              ? "a call takes one `**` mapping whose keys are known only at "
                "run time here; merge the mappings into one first"
              : "variadic keyword arguments are not supported for inlined "
                "class methods unless they are a dict with str keys"});
    }
  }

  Value descriptorReceiver = emitDescriptorReceiver(expr, receiver, method);
  bool bindReceiver = methodBindingBindsReceiver(method);
  if (method.kind == "instance" && mlir::isa<py::TypeType>(receiver.type))
    bindReceiver = false;
  std::optional<Value> collectedVararg;
  std::optional<Value> collectedKwargs;
  if (dynamicTail || dynamicMapping) {
    DynamicUnpackTarget target = dynamicUnpackTargetForMethod(method);
    DynamicUnpackCall call;
    call.explicitPositional = static_cast<unsigned>(positional.size());
    if (dynamicTail) {
      for (std::size_t index = target.positionalNames.size();
           index < positional.size(); ++index)
        call.extras.push_back(positional[index]);
      call.tail = *dynamicTail;
    }
    for (const auto &entry : keywords)
      call.keywords.push_back({entry.name, entry.value});
    std::optional<DynamicUnpackResult> collected =
        emitDynamicUnpack(expr, target, call);
    if (!collected)
      return emitNone(expr);
    if (dynamicTail && positional.size() > target.positionalNames.size())
      positional.resize(target.positionalNames.size());
    KeywordArguments named;
    for (const auto &[name, value] : collected->named)
      named[name] = value;
    keywords = std::move(named);
    collectedVararg = collected->vararg;
    collectedKwargs = collected->kwargs;
  }
  if (bindReceiver && !collectedVararg && !collectedKwargs)
    if (std::optional<Value> called = tryCallLargeMethod(
            expr, descriptorReceiver, method, positional, keywords))
      return *called;
  return emitInlineMethodBody(expr, descriptorReceiver, bindReceiver, method,
                              positional, keywords, collectedVararg,
                              collectedKwargs);
}

namespace {

// LYTHON_ABLATE_METHOD_CALLS=1 copies every method body into its call site
// again, for comparing the two with one binary.
bool methodCallsAblated() {
  static const bool ablated =
      std::getenv("LYTHON_ABLATE_METHOD_CALLS") != nullptr;
  return ablated;
}

// The AST size of a def from which its call sites call it instead of copying
// it. A copy is what lets a call site specialize the body, and a small body
// is cheap to copy; a large one nested a few calls deep was most of what a
// program that runs unittest compiled.
// LYTHON_CALLED_METHOD_NODES=N moves it: 0 calls every method that can be
// called, which is how the whole corpus exercises the call path.
unsigned calledMethodNodes() {
  static const unsigned nodes = [] {
    if (const char *value = std::getenv("LYTHON_CALLED_METHOD_NODES"))
      return static_cast<unsigned>(std::max(0, std::atoi(value)));
    return 100u;
  }();
  return nodes;
}

} // namespace

// ⭐ A LARGE METHOD IS CALLED, NOT COPIED, WHERE THE CALL CANNOT TELL THE
// DIFFERENCE. Every call site used to copy the body in (`emitInlineMethodBody`),
// so a method that calls a method that calls a method was copied into each
// level: `TextTestRunner.run` into `main`, `TestSuite.run` into it, `TestCase.run`
// into that -- a third of the IR of a one-test unittest program.
//
// What the copy gives a call site, and why each is the same through a call:
//
// - `self` bound to the receiver, so `self.other()` in a base's body finds the
//   receiver's override. A constructed receiver, or one of a class nothing
//   derives from, has exactly its static class -- the case the method's own
//   symbol is written for (`selfIsExact`). Any other receiver gets a body
//   emitted for its class with `self` NOT exact (`openMethodBody`), which asks
//   every override question the copy asks of a receiver whose class it does
//   not know.
// - The arguments as the call typed them. Taken only when that IS the declared
//   type (a union parameter takes the union, as the copy does), so the body
//   types each parameter as the copy would.
// - An omitted argument's default, read from the cell the class body filled,
//   as the copy reads it, or a constant.
//
// ⛔ Not a dunder, a generic or pack-forwarding method, a decorated one, a
// generator or coroutine, an exception class's, or one with `*args`,
// `**kwargs` or keyword-only parameters: each has its own call-site machinery,
// and the copy is what it is built on.
// ⛔ Not an open body for a class of the program's own module: it would be
// emitted wherever the call is, and inside an imported module's code that is
// that module's scope. An imported class's body is emitted in its own module
// (`emitInDefiningModuleScope`), as its standalone symbol is.
std::optional<Value> ModuleEmitter::tryCallLargeMethod(
    const parser::Node &anchor, Value receiver, const MethodBinding &method,
    llvm::ArrayRef<Value> positional, const KeywordArguments &keywords,
    const MethodSpecialization *specialization) {
  if (methodCallsAblated() || !method.method || method.kind != "instance" ||
      method.symbolName.empty() || method.async)
    return std::nullopt;
  const FunctionSignature &sig =
      specialization ? specialization->signature
      : method.bodySignature.callable ? method.bodySignature
                                      : method.signature;
  if (sig.isGeneratorFunction || sig.isAsyncGeneratorFunction ||
      sig.isCoroutineFunction || sig.varargName || sig.kwargName ||
      !sig.kwOnlyNames.empty() || sig.positionalNames.empty() ||
      sig.positionalNames.size() != sig.positionalTypes.size())
    return std::nullopt;
  const parser::Node &def = *method.method;
  if ((specialization ? forwardsItsPacks(def) : specializedAtEachCall(def)) ||
      decoratedMethodBodies.contains(&def))
    return std::nullopt;
  std::string name(ast::string(def, "name").value_or(""));
  if (name.size() > 4 && llvm::StringRef(name).starts_with("__") &&
      llvm::StringRef(name).ends_with("__"))
    return std::nullopt;
  auto [size, measured] = methodDefSizes.try_emplace(&def, 0);
  if (measured)
    ast::walk(&def, [&](const parser::Node &) {
      ++size->second;
      return ast::Walk::Continue;
    });
  // ⭐ A DISPATCHER'S ARM CALLS WHATEVER ITS SIZE. `getattr(self, name)()`
  // has an arm per name every TestCase subclass answers, and copying each
  // test method -- with the assertion bodies it copies in turn -- into the
  // one dispatcher was a third of a unittest file's IR, growing with every
  // test. The arm passes only the receiver, so nothing is specialized by the
  // copy that the method's own body does not already have.
  // ⭐ AND A GENERIC METHOD IS CALLED WHATEVER ITS SIZE, through one body per
  // solution. Copied, `assertEqual[T]` brought its message formatting and
  // `assertMultiLineEqual` into every assertion -- about 1,100 lines of IR
  // each, half of a unittest file -- for the same few solutions.
  if (size->second < calledMethodNodes() && !specialization &&
      !dispatcherArmCalls.contains(&anchor))
    return std::nullopt;
  auto receiverClass = mlir::dyn_cast_if_present<py::ContractType>(receiver.type);
  if (!receiverClass || !receiverClass.getArguments().empty())
    return std::nullopt;
  llvm::StringRef receiverName = receiverClass.getContractName();
  if (isExceptionBackedClass(receiverName) ||
      isExceptionBackedClass(method.definingClass) ||
      classTypeArguments.count(receiverName) ||
      classTypeArguments.count(method.definingClass))
    return std::nullopt;

  // Every parameter filled by the call or by its default.
  const std::size_t slots = sig.positionalNames.size();
  if (positional.size() + 1 > slots)
    return std::nullopt;
  llvm::SmallVector<Value, 8> arguments(slots);
  arguments[0] = receiver;
  for (auto [index, argument] : llvm::enumerate(positional))
    arguments[index + 1] = argument;
  for (const auto &entry : keywords) {
    auto found = llvm::find(sig.positionalNames, entry.name);
    if (found == sig.positionalNames.end() ||
        found == sig.positionalNames.begin())
      return std::nullopt;
    std::size_t slot =
        static_cast<std::size_t>(std::distance(sig.positionalNames.begin(), found));
    if (arguments[slot].value)
      return std::nullopt;
    arguments[slot] = entry.value;
  }
  const parser::Node *argumentsNode = ast::node(def, "args");
  if (!argumentsNode || positionalArgumentNodes(*argumentsNode).size() != slots)
    return std::nullopt;
  const auto *defaults = ast::nodeList(*argumentsNode, "defaults");
  std::size_t firstDefault = defaults && defaults->size() <= slots
                                 ? slots - defaults->size()
                                 : slots;
  auto defaultCell = [&](std::size_t slot) -> std::optional<std::string> {
    return methodDefaultCell(method, static_cast<unsigned>(slot),
                             slot < sig.positionalTypes.size()
                                 ? sig.positionalTypes[slot]
                                 : mlir::Type());
  };
  for (std::size_t slot = 1; slot < slots; ++slot) {
    if (arguments[slot].value)
      continue;
    if (slot < firstDefault)
      return std::nullopt;
    const parser::Node *defaultNode = (*defaults)[slot - firstDefault].get();
    if (!defaultCell(slot) && (!defaultNode || defaultNode->kind != "Constant"))
      return std::nullopt;
  }
  // Each of the type its parameter declares.
  for (std::size_t slot = 1; slot < slots; ++slot) {
    if (!arguments[slot].value)
      continue;
    mlir::Type declared = sig.positionalTypes[slot];
    mlir::Type actual = arguments[slot].type;
    if (!declared || !actual)
      return std::nullopt;
    // A literal of the declared type is that type to the body: the method's
    // own symbol is typed so, and compiled whether or not anything calls it.
    if (types.widenLiteral(actual) == declared)
      continue;
    auto unionType = mlir::dyn_cast<py::UnionType>(declared);
    if (unionType && !mlir::isa<py::UnionType>(actual) &&
        unionType.hasMember(types.widenLiteral(actual)))
      continue;
    return std::nullopt;
  }

  // Which body runs it.
  mlir::Operation *definition = receiver.value.getDefiningOp();
  bool constructed = definition && mlir::isa<py::NewOp, py::InitOp>(definition);
  bool leaf = !classHasDeclaredSubclass(receiverName);
  FunctionSignature callSig = sig;
  bool exactBody = (constructed || leaf) &&
                   sig.positionalTypes.front() == receiver.type;
  if (!exactBody) {
    if (constructed && !leaf)
      return std::nullopt;
    const EmitOptions::SourceModule *source =
        sourceModuleForClass(method.definingClass);
    if (!source || source->isStub)
      return std::nullopt;
    callSig.positionalTypes.front() = receiver.type;
    types.refreshCallable(callSig);
  }
  auto callable = mlir::dyn_cast_if_present<py::CallableType>(
      callSig.publicCallable ? callSig.publicCallable : callSig.callable);
  if (!callable ||
      !types.inferCallWithEvidence(callable, callSig.positionalTypes, {}))
    return std::nullopt;
  std::string symbol =
      specialization
          ? specializedMethodBody(method, callSig, specialization->typeArguments,
                                  exactBody)
      : exactBody ? method.symbolName
                  : openMethodBody(method, receiver.type, callSig);
  // LYTHON_TRACE_METHOD_CALLS=1: each call that takes a body by symbol, and
  // which body -- the class's own (`exact`) or one for a receiver whose class
  // is not known (`open`).
  static const bool traceCalls =
      std::getenv("LYTHON_TRACE_METHOD_CALLS") != nullptr;
  if (traceCalls)
    llvm::errs() << "[method-call] " << (exactBody ? "exact " : "open ")
                 << method.definingClass << "." << name << " on "
                 << receiverName << " (" << size->second << " nodes)\n";

  for (std::size_t slot = 1; slot < slots; ++slot) {
    mlir::Type declared = sig.positionalTypes[slot];
    if (arguments[slot].value) {
      if (arguments[slot].type != declared)
        arguments[slot] = coerceValue(arguments[slot], declared, anchor);
      continue;
    }
    Value value;
    if (std::optional<std::string> cell = defaultCell(slot)) {
      auto get = py::GlobalGetOp::create(builder, loc(anchor), declared,
                                         builder.getStringAttr(*cell));
      value = Value{get.getResult(), declared};
    } else {
      value = emitExpr((*defaults)[slot - firstDefault].get());
    }
    arguments[slot] = coerceValue(value, declared, anchor);
  }
  Value callee = emitBindingRef(anchor, symbol, callable);
  CallOperands operands;
  for (const Value &argument : arguments) {
    operands.positional.push_back(argument);
    operands.positionalTypes.push_back(argument.type);
    operands.positionalUnpacked.push_back(0);
  }
  return emitCallableDispatch(anchor, callee, operands,
                              sig.resultType ? sig.resultType : types.none());
}

std::string ModuleEmitter::specializedMethodBody(
    const MethodBinding &method, const FunctionSignature &signature,
    const TypeBindingMap &typeArguments, bool selfIsExact) {
  mlir::Type callable =
      signature.publicCallable ? signature.publicCallable : signature.callable;
  auto &memo = specializedMethodBodies[selfIsExact ? 1 : 0];
  auto key = std::make_pair(static_cast<const parser::Node *>(method.method),
                            callable);
  if (auto found = memo.find(key); found != memo.end())
    return found->second;
  std::string symbol = method.symbolName + "$spec" +
                       std::to_string(memo.size()) +
                       (selfIsExact ? "" : "$open");
  // Named before the body, as an open body is: a recursion is a call.
  memo[key] = symbol;
  const EmitOptions::SourceModule *source =
      sourceModuleForClass(method.definingClass);
  const parser::Node *def = method.method;
  std::string definer = method.definingClass;
  auto emitBody = [this, def, symbol, signature, typeArguments, source,
                   definer, selfIsExact] {
    auto savedInlining = std::move(methodsBeingInlined);
    methodsBeingInlined.clear();
    auto savedInlineFrames = std::move(inlineFrames);
    inlineFrames.clear();
    llvm::scope_exit restore([&] {
      methodsBeingInlined = std::move(savedInlining);
      inlineFrames = std::move(savedInlineFrames);
    });
    auto emitSpecialized = [&] {
      auto typeParameterScope = types.pushScope();
      bindClassTypeArguments(definer);
      // Body annotations spell the type parameters by name (`x: T`).
      for (const auto &binding : typeArguments) {
        types.bindLocalSymbol(binding.first, binding.second);
        types.bindLocalTypeParameter(binding.first, binding.second);
      }
      superContexts.push_back(
          SuperContext{definer, signature.positionalNames.front(), selfIsExact});
      emitCallableFunction(*def, symbol, signature, {}, /*isLambda=*/false);
      superContexts.pop_back();
    };
    if (source)
      emitInDefiningModuleScope(*source, emitSpecialized);
    else
      emitSpecialized();
  };
  if (classDeclarationsEmitted)
    emitBody();
  else
    deferredDispatcherBodies.push_back(std::move(emitBody));
  return symbol;
}

std::string ModuleEmitter::openMethodBody(const MethodBinding &method,
                                          mlir::Type receiverType,
                                          const FunctionSignature &signature) {
  auto key = std::make_pair(static_cast<const parser::Node *>(method.method),
                            receiverType);
  if (auto found = openMethodBodies.find(key); found != openMethodBodies.end())
    return found->second;
  std::string receiverName =
      mlir::cast<py::ContractType>(receiverType).getContractName().str();
  std::string symbol =
      method.symbolName + "$open$" + sanitizedSymbolPart(receiverName);
  // Named before the body: a call in it that reaches this body again names
  // this symbol, and a recursion is a call.
  openMethodBodies[key] = symbol;
  const EmitOptions::SourceModule *source =
      sourceModuleForClass(method.definingClass);
  const parser::Node *def = method.method;
  std::string definer = method.definingClass;
  auto emitBody = [this, def, symbol, signature, source, definer] {
    // A function of its own, emitted from wherever the call that asked for it
    // was: nothing of that body's inlining belongs to this one.
    auto savedInlining = std::move(methodsBeingInlined);
    methodsBeingInlined.clear();
    auto savedInlineFrames = std::move(inlineFrames);
    inlineFrames.clear();
    llvm::scope_exit restore([&] {
      methodsBeingInlined = std::move(savedInlining);
      inlineFrames = std::move(savedInlineFrames);
    });
    auto emitOpen = [&] {
      auto typeParameterScope = types.pushScope();
      bindClassTypeArguments(definer);
      superContexts.push_back(SuperContext{definer,
                                           signature.positionalNames.front(),
                                           /*selfIsExact=*/false});
      emitCallableFunction(*def, symbol, signature, {}, /*isLambda=*/false);
      superContexts.pop_back();
    };
    // A class of the main module is emitted in the main module's scope.
    if (source)
      emitInDefiningModuleScope(*source, emitOpen);
    else
      emitOpen();
  };
  if (classDeclarationsEmitted)
    emitBody();
  else
    deferredDispatcherBodies.push_back(std::move(emitBody));
  return symbol;
}

// ⛔ THE MESSAGE NAMES THE DUNDER'S OWN OPERAND ORDER. CPython builds it from
// the operator as WRITTEN, and a reflected dispatch (`a < b` reaching
// `b.__gt__`) would name the two the other way round. The class names and the
// exception are right either way; only a reflected ordering whose BOTH sides
// decline can print the pair reversed, and a class that defines `__lt__` alone
// -- the idiom -- is dispatched directly.
// `@d1 @d2 def f(...)` is `f = d1(d2(f))`: the decorator closest to the def
// applies first, so the list is walked backwards.
void ModuleEmitter::applyFunctionDecorators(const parser::Node &statement) {
  const auto *decorators = ast::nodeList(statement, "decorator_list");
  if (!decorators || decorators->empty())
    return;
  auto name = ast::string(statement, "name");
  if (!name)
    return;
  // ⛔ ONLY THE INNERMOST APPLICATION READS THE SYMBOL. `@a @b def f` is
  // `f = a(b(f))`, emitted as two assignments; the second must read what the
  // first stored, and forcing it to the symbol too wrapped the UNDECORATED
  // function -- `@times_ten @plus_one def triple` answered 60 for 70.
  bool innermost = true;
  for (const parser::NodePtr &decorator : llvm::reverse(*decorators)) {
    if (!decorator)
      continue;
    // ⭐ AND `@deco(arg)` IS `f = deco(arg)(f)`. The factory's own call is the
    // callee of the application, evaluated here rather than named: it is the
    // hand-written `f = deco(arg)(f)` this compiler already runs, one AST
    // shape away.
    parser::NodePtr callee = bindingDecoratorCallee(decorator);
    if (!callee)
      continue;
    std::vector<parser::NodePtr> arguments;
    arguments.push_back(synth::name(*name, statement.range));
    parser::NodePtr applied = synth::assign(
        synth::name(*name, statement.range),
        synth::call(std::move(callee), std::move(arguments), statement.range),
        statement.range);
    {
      llvm::SaveAndRestore<std::string> subject(
          decoratorSubjectName, innermost ? std::string(*name)
                                          : decoratorSubjectName);
      emitStatement(*applied);
    }
    innermost = false;
  }
}

std::optional<ModuleEmitter::NotImplementedFallback>
ModuleEmitter::comparisonDunderFallback(
    llvm::StringRef methodName, llvm::ArrayRef<std::string> positionalNames) {
  static constexpr llvm::StringLiteral kComparisons[] = {
      "__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__"};
  if (!llvm::is_contained(kComparisons, methodName) ||
      positionalNames.size() != 2)
    return std::nullopt;
  return NotImplementedFallback{methodName.str(), positionalNames[0],
                                positionalNames[1]};
}

parser::NodePtr
ModuleEmitter::notImplementedFallbackStatement(const parser::Node &statement) {
  if (!notImplementedFallback)
    return nullptr;
  const parser::Node *returned = ast::node(statement, "value");
  if (!returned || returned->kind != "Name" ||
      llvm::StringRef(ast::nameSpelling(*returned)) != "NotImplemented")
    return nullptr;
  const NotImplementedFallback &fallback = *notImplementedFallback;
  parser::SourceRange range = statement.range;
  auto receiver = [&] { return synth::name(fallback.receiver, range); };
  auto other = [&] { return synth::name(fallback.other, range); };
  if (fallback.method == "__eq__" || fallback.method == "__ne__") {
    parser::NodePtr identical =
        synth::compare(receiver(), "Is", other(), range);
    if (fallback.method == "__ne__")
      identical = synth::notOp(std::move(identical), range);
    return synth::returnStmt(std::move(identical), range);
  }
  llvm::StringRef spelling =
      llvm::StringSwitch<llvm::StringRef>(fallback.method)
          .Case("__lt__", "<")
          .Case("__le__", "<=")
          .Case("__gt__", ">")
          .Case("__ge__", ">=")
          .Default("");
  if (spelling.empty())
    return nullptr;
  auto typeName = [&](parser::NodePtr value) {
    return synth::attribute(
        synth::call(synth::name("type", range), {std::move(value)}, range),
        "__name__", range);
  };
  parser::NodePtr message = synth::strConstant(
      ("'" + spelling + "' not supported between instances of '").str(), range);
  message = synth::binOp(std::move(message), "Add", typeName(receiver()), range);
  message = synth::binOp(std::move(message),
                         "Add", synth::strConstant("' and '", range), range);
  message = synth::binOp(std::move(message), "Add", typeName(other()), range);
  message = synth::binOp(std::move(message), "Add",
                         synth::strConstant("'", range), range);
  std::vector<parser::NodePtr> arguments;
  arguments.push_back(std::move(message));
  return synth::raiseStmt(
      synth::call(synth::name("TypeError", range), std::move(arguments), range),
      range);
}

bool ModuleEmitter::hasTypeParameters(const parser::Node &def) {
  const auto *params = ast::nodeList(def, "type_params");
  return params && !params->empty();
}

// The dict a `**kwargs` parameter receives: string keys, and values named
// through `LyValueRef` because they are already emitted. An EMPTY one takes
// the declared type from the expectation, the same way an empty vararg tuple
// does.
Value ModuleEmitter::emitKeywordPackDict(
    const parser::Node &anchor,
    llvm::ArrayRef<std::pair<std::string, Value>> members,
    mlir::Type declared) {
  std::size_t refStart = pendingValueRefs.size();
  auto releaseRefs =
      llvm::scope_exit([&] { pendingValueRefs.resize(refStart); });
  std::vector<parser::NodePtr> keyNodes;
  std::vector<parser::NodePtr> valueNodes;
  for (const auto &[name, value] : members) {
    keyNodes.push_back(synth::strConstant(name, anchor.range));
    parser::NodePtr ref = parser::makeNode("LyValueRef", anchor.range);
    parser::addField(*ref, "slot",
                     static_cast<std::int64_t>(pendingValueRefs.size()));
    pendingValueRefs.push_back(value);
    valueNodes.push_back(std::move(ref));
  }
  parser::NodePtr dictNode = parser::makeNode("Dict", anchor.range);
  parser::addField(*dictNode, "keys", std::move(keyNodes));
  parser::addField(*dictNode, "values", std::move(valueNodes));
  Value packed = declared ? emitExprExpected(dictNode.get(), declared)
                          : emitExpr(dictNode.get());
  if (declared && packed.type != declared)
    packed = coerceValue(packed, declared, anchor);
  return packed;
}

bool ModuleEmitter::keywordPackIsOnlyRead(const parser::Node &def,
                                          llvm::StringRef name) {
  auto isPack = [&](const parser::Node *node) {
    return node && node->kind == "Name" &&
           llvm::StringRef(ast::nameSpelling(*node)) == name;
  };
  static const llvm::StringSet<> kReaders = {"get", "items", "keys", "values",
                                             "copy"};
  static const llvm::StringSet<> kReadingBuiltins = {
      "len", "sorted", "list", "tuple", "set", "dict", "repr", "str", "bool"};
  unsigned seen = 0;
  unsigned read = 0;
  std::function<void(const parser::Node &)> visit =
      [&](const parser::Node &node) {
        if (isPack(&node))
          ++seen;
        if (node.kind == "Call") {
          const parser::Node *func = ast::node(node, "func");
          if (const auto *keywords = ast::nodeList(node, "keywords"))
            for (const parser::NodePtr &keyword : *keywords)
              if (keyword && !ast::string(*keyword, "arg") &&
                  isPack(ast::node(*keyword, "value")))
                ++read;
          if (func && func->kind == "Attribute" &&
              isPack(ast::node(*func, "value")))
            if (auto attr = ast::string(*func, "attr");
                attr && kReaders.contains(*attr))
              ++read;
          const auto *args = ast::nodeList(node, "args");
          if (func && func->kind == "Name" &&
              kReadingBuiltins.contains(ast::nameSpelling(*func)) && args &&
              args->size() == 1 && isPack(args->front().get()))
            ++read;
        } else if (node.kind == "Subscript") {
          if (isPack(ast::node(node, "value")))
            ++read;
        } else if (node.kind == "Compare") {
          if (const auto *comparators = ast::nodeList(node, "comparators"))
            for (const parser::NodePtr &comparator : *comparators)
              if (isPack(comparator.get()))
                ++read;
        } else if (node.kind == "For" || node.kind == "comprehension") {
          if (isPack(ast::node(node, "iter")))
            ++read;
        } else if (node.kind == "Assign" || node.kind == "AugAssign" ||
                   node.kind == "AnnAssign" || node.kind == "Delete") {
          // A subscript STORE through the pack reads as a Subscript below,
          // so it is taken back here: the pack is then not only read.
          auto storesThrough = [&](const parser::Node *target) {
            return target && target->kind == "Subscript" &&
                   isPack(ast::node(*target, "value"));
          };
          if (const auto *targets = ast::nodeList(node, "targets"))
            for (const parser::NodePtr &target : *targets)
              if (storesThrough(target.get()))
                seen += 2;
          if (storesThrough(ast::node(node, "target")))
            seen += 2;
        }
        for (const parser::Field &field : node.fields) {
          if (const auto *child = std::get_if<parser::NodePtr>(&field.value)) {
            if (*child)
              visit(**child);
          } else if (const auto *list =
                         std::get_if<std::vector<parser::NodePtr>>(
                             &field.value)) {
            for (const parser::NodePtr &item : *list)
              if (item)
                visit(*item);
          }
        }
      };
  if (const auto *body = ast::nodeList(def, "body"))
    for (const parser::NodePtr &statement : *body)
      if (statement)
        visit(*statement);
  return seen == read;
}

bool ModuleEmitter::forwardsItsPacks(const parser::Node &def) {
  const parser::Node *arguments = ast::node(def, "args");
  if (!arguments)
    return false;
  auto packName = [&](llvm::StringRef field) -> std::string {
    const parser::Node *pack = ast::node(*arguments, field);
    return pack ? std::string(ast::nameSpelling(*pack)) : std::string();
  };
  std::string varargName = packName("vararg");
  std::string kwargName = packName("kwarg");
  if (varargName.empty() && kwargName.empty())
    return false;
  bool forwards = false;
  std::function<void(const parser::Node &)> visit =
      [&](const parser::Node &node) {
        if (forwards)
          return;
        if (node.kind == "Call") {
          if (const auto *args = ast::nodeList(node, "args"))
            for (const parser::NodePtr &arg : *args)
              if (arg && arg->kind == "Starred")
                if (const parser::Node *value = ast::node(*arg, "value");
                    value && value->kind == "Name" && !varargName.empty() &&
                    ast::nameSpelling(*value) == varargName)
                  forwards = true;
          if (const auto *keywords = ast::nodeList(node, "keywords"))
            for (const parser::NodePtr &keyword : *keywords)
              if (keyword && !ast::string(*keyword, "arg"))
                if (const parser::Node *value = ast::node(*keyword, "value");
                    value && value->kind == "Name" && !kwargName.empty() &&
                    ast::nameSpelling(*value) == kwargName)
                  forwards = true;
        }
        for (const parser::Field &field : node.fields) {
          if (const auto *child = std::get_if<parser::NodePtr>(&field.value)) {
            if (*child)
              visit(**child);
          } else if (const auto *list =
                         std::get_if<std::vector<parser::NodePtr>>(
                             &field.value)) {
            for (const parser::NodePtr &item : *list)
              if (item)
                visit(*item);
          }
        }
      };
  if (const auto *body = ast::nodeList(def, "body"))
    for (const parser::NodePtr &statement : *body)
      if (statement)
        visit(*statement);
  return forwards;
}

// Whether an argument of type `actual` may bind a parameter declared `expected`
// at an inlined method's call -- the rule `checkArgument` reports, and the
// one a dispatch arm asks before it is written.
bool ModuleEmitter::inlineArgumentFits(mlir::Type actual,
                                       mlir::Type expected) {
  if (!expected || !actual || unboundStaticParameterCount(expected) != 0 ||
      unboundStaticParameterCount(actual) != 0)
    return true;
  if (isAssignableWithStaticEvidence(actual, expected, module))
    return true;
  // ⭐ A VALUE FITS A UNION WHEN IT FITS A MEMBER, by the rules below as much
  // as by assignability: `run(result: TestResult | None)` reached by a
  // TextTestResult was refused, though a TestResult parameter takes the
  // subclass and a free function's `TestResult | None` did too.
  if (auto expectedUnion = mlir::dyn_cast<py::UnionType>(expected))
    if (!mlir::isa<py::UnionType>(actual))
      for (mlir::Type member : expectedUnion.getMemberTypes())
        if (member != expected && inlineArgumentFits(actual, member))
          return true;
  // ⭐ THE EMITTER'S OWN MRO, because the module's may not exist yet. An
  // unbound base-method call inside a subclass method passes `self`:
  //
  //     class Child(Base):
  //         def __init__(self, n: int) -> None:
  //             Base.__init__(self, "c")
  //     # argument 'self' of '__init__' is declared Base and this call gives
  //     # it Child
  //
  // `Base.__init__(c, "z")` at module scope works, and so does
  // `Base.greet(c, "x")`, because by then `py.class @Child` carries its
  // `mro_names` and `isAssignableTo` can walk it. Inside Child's own method
  // the class op is not in the module yet, so the walk finds no bases and
  // answers "not a subtype" for a subtype. `classMros` is populated before any
  // body is emitted -- it is what `resolveMroMethod` already reads.
  //
  // ⛔ Only for source-defined contracts on both sides. A manifest contract's
  // assignability is the manifest's business, and `isAssignableTo` is the one
  // that knows the protocol and subtype relations there.
  // ⛔ NO DOT TEST. It read "a name with a dot is a manifest contract", and
  // an IMPORTED class's name is dotted too (`shapes2.Right`), so passing an
  // imported subclass where the imported base is declared was refused --
  // "argument 'inner' of '__init__' is declared shapes2.Base and this call
  // gives it shapes2.Right" -- while the same hierarchy in ONE file was
  // accepted. The MRO map is the answer on its own: it holds exactly the
  // classes this compiler declared, and a manifest contract is never in it.
  if (auto expectedContract = mlir::dyn_cast<py::ContractType>(expected))
    if (auto actualContract = mlir::dyn_cast<py::ContractType>(actual))
      if (llvm::is_contained(classMro(actualContract.getContractName()),
                             expectedContract.getContractName()))
        return true;
  // ⭐ A SUBCLASS IS A CLASS VALUE OF ITS BASE: `type[Sq]` passed where
  // `type[Shape]` is declared is the class value naming Sq, which is what
  // the parameter holds at run time.
  // (Not along the numeric tower, which `isAssignableWithStaticEvidence`
  // states for class values.)
  if (auto expectedClass = mlir::dyn_cast<py::TypeType>(expected))
    if (auto actualClass = mlir::dyn_cast<py::TypeType>(actual)) {
      auto numeric = [](mlir::Type type) {
        auto contract = mlir::dyn_cast_if_present<py::ContractType>(type);
        return contract && (contract.getContractName() == "builtins.bool" ||
                            contract.getContractName() == "builtins.int" ||
                            contract.getContractName() == "builtins.float" ||
                            contract.getContractName() == "builtins.complex");
      };
      mlir::Type actualInstance = actualClass.getInstanceType();
      mlir::Type expectedInstance = expectedClass.getInstanceType();
      if (!(numeric(actualInstance) && numeric(expectedInstance)) &&
          (isAssignableWithStaticEvidence(actual, expected, module) ||
           pythonSubclassOf(actualInstance, expectedInstance, types,
                            module)))
        return true;
    }
  // ⭐ A READ-ONLY PROTOCOL TAKES ITS ELEMENT BY THESE SAME RULES: a
  // `list[M]` fits `Iterable[Case | Group]` when it is an `Iterable[M]` and M
  // fits the union. `isAssignableTo` asked the element question with the
  // class ops' walk and no union member rule, so unittest's
  // `TestSuite([Mine("test_a")])` was refused while the same call into a
  // free function was not.
  // ⛔ Only the protocols that never write an element back: a list passed
  // where `MutableSequence[Base]` is declared could be given a Base.
  if (auto expectedProtocol = mlir::dyn_cast<py::ProtocolType>(expected))
    if (auto actualContract = mlir::dyn_cast<py::ContractType>(actual))
      if (expectedProtocol.getArguments().size() == 1 &&
          actualContract.getArguments().size() == 1 &&
          llvm::is_contained({llvm::StringRef("Iterable"),
                              llvm::StringRef("Collection"),
                              llvm::StringRef("Sequence"),
                              llvm::StringRef("Container"),
                              llvm::StringRef("Reversible")},
                             expectedProtocol.getProtocolName())) {
        mlir::Type element = actualContract.getArguments().front();
        mlir::Type ownElement = py::ProtocolType::get(
            &context, expectedProtocol.getProtocolName(), {element});
        if (isAssignableWithStaticEvidence(actual, ownElement, module) &&
            inlineArgumentFits(types.widenLiteral(element),
                               expectedProtocol.getArguments().front()))
          return true;
      }
  // ⭐ A BARE generic contract accepts any instantiation of itself. A generic
  // class's own methods spell the receiver without its arguments --
  // `def __add__(self, other: "Counter") -> "Counter"` -- and the argument
  // arrives as `Counter[str]`, which `isAssignableTo` answers false for
  // because the argument lists differ. `c + c` on a `Counter[str]` was
  // therefore refused by this very check: "argument 'other' of '__add__' is
  // declared Counter and this call gives it Counter[str]".
  //
  // ⛔ One direction only. Declared WITH arguments and supplied without is a
  // real mismatch -- the parameter promises an instantiation the argument
  // does not name -- and stays refused.
  if (auto expectedContract = mlir::dyn_cast<py::ContractType>(expected))
    if (auto actualContract = mlir::dyn_cast<py::ContractType>(actual))
      if (expectedContract.getArguments().empty() &&
          !actualContract.getArguments().empty() &&
          expectedContract.getContractName() ==
              actualContract.getContractName())
        return true;
  // ⭐ The numeric tower is admitted HERE and not by `isAssignableTo`, which
  // answers false for int against float. A free function refuses the same
  // call and `emitArgumentSpecializedCall` then emits a SECOND BODY at the
  // argument's rung; an inlined method re-emits its body at every call site
  // already, so the specialization it would need is the emission that is
  // about to happen. Refusing here would take away a shape that works.
  auto rung = [&](mlir::Type type) {
    if (type == types.boolType())
      return 0;
    if (type == types.intType())
      return 1;
    if (type == types.floatType())
      return 2;
    auto contract = mlir::dyn_cast_if_present<py::ContractType>(type);
    if (contract && contract.getArguments().empty() &&
        contract.getContractName() == "builtins.complex")
      return 3;
    return -1;
  };
  int actualRung = rung(actual);
  int expectedRung = rung(expected);
  if (actualRung >= 0 && expectedRung >= 0 && actualRung <= expectedRung)
    return true;
  return false;
}

Value ModuleEmitter::emitInlineMethodBody(
    const parser::Node &anchor, Value receiver, bool bindDescriptorReceiver,
    const MethodBinding &method, llvm::ArrayRef<Value> positional,
    const KeywordArguments &keywords, std::optional<Value> collectedVararg,
    std::optional<Value> collectedKwargs) {
  if (!method.method)
    return emitNone(anchor);
  const std::string callerSourceName = sourceName;
  // A frozen dataclass refuses field stores everywhere EXCEPT the constructor
  // that fills them -- CPython's own __init__ goes around the block with
  // object.__setattr__, and the synthesized one here is inlined through this
  // path, so the exemption is a context rather than a symbol name (there is no
  // symbol: the body is inlined into its caller).
  std::string frozenInitOwner;
  std::optional<llvm::SaveAndRestore<const std::string *>> frozenInit;
  if (ast::string(*method.method, "name").value_or("") == "__init__") {
    if (auto receiverContract =
            mlir::dyn_cast_if_present<py::ContractType>(receiver.type))
      frozenInitOwner = receiverContract.getContractName().str();
    else
      frozenInitOwner = method.definingClass;
    if (frozenDataclassContracts.count(frozenInitOwner))
      frozenInit.emplace(frozenInitContract, &frozenInitOwner);
  }
  // A method body reached from inside its own inlining (directly, or around a
  // cycle like a() -> self.b() -> self.a()) would expand without end: reject
  // at this boundary instead of recursing until the emitter's own stack
  // overflows. The check is on the body node, so the same method inlined
  // twice side by side is unaffected.
  // ⭐ A RECURSIVE METHOD CALLS THE EMITTED SYMBOL instead of expanding. Every
  // class-method call is inlined at its call site, which has no base case
  // around a cycle, so `Node.total()` summing over `self.kids` -- the tree
  // traversal every recursive data structure is walked with -- was refused:
  // "recursive class method call is not supported (total -> total)".
  //
  // The method is ALREADY emitted as a `func.func` under `symbolName` (the
  // class's `method_symbols`); inlining is what call sites do, not the only
  // thing they can do. So the cycle is broken by taking the other path: a
  // `py.binding.ref` to that symbol, called with the receiver as the leading
  // positional, which is exactly how a free function recurses today.
  //
  // ⛔ Only inside the cycle, and that is deliberate rather than conservative.
  // Inlining is what lets a base method's `self.who()` bind to the RECEIVER's
  // real class at each site (see `lookupClassMethod`), and a symbol call fixes
  // the callee at the defining class. Taking it everywhere would turn every
  // overridden method into a base-class call. Here the callee is the method
  // already being inlined, so its class is the one the recursion is in.
  if (llvm::is_contained(methodsBeingInlined, method.method) &&
      specializedAtEachCall(*method.method)) {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, anchor.range.start,
        std::string(hasTypeParameters(*method.method)
                        ? "the generic method '"
                        : "the method '") +
            std::string(ast::string(*method.method, "name").value_or("")) +
            "' calls itself: its body is specialized at each call site, and a "
            "call cycle has no base case to stop the expansion"});
    return emitNone(anchor);
  }
  if (llvm::is_contained(methodsBeingInlined, method.method) &&
      (collectedVararg || collectedKwargs)) {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, anchor.range.start,
        "the method '" +
            std::string(ast::string(*method.method, "name").value_or("")) +
            "' calls itself with a starred argument or a `**` mapping whose "
            "size is known only at run time, which its recursive call does "
            "not pass"});
    return emitNone(anchor);
  }
  if (llvm::is_contained(methodsBeingInlined, method.method)) {
    py::CallableType recursive =
        method.signature.publicCallable
            ? mlir::dyn_cast<py::CallableType>(method.signature.publicCallable)
            : mlir::dyn_cast_if_present<py::CallableType>(
                  method.signature.callable);
    // ⭐ KEYWORDS ARE PLACED, not refused. The symbol call packs positionals,
    // and a recursive call names a signature this walk already has -- so a
    // keyword that names a positional parameter is that parameter's slot, and
    // `self.down(n - step, step=step)` is the ordinary way to write a
    // recursion that carries a setting. Refusing it sent the whole method back
    // to "recursive class method call is not supported".
    //
    // ⛔ Every parameter must end up filled, and by exactly one thing. A gap
    // would have to be closed from the callee's defaults, and this is a real
    // call rather than an inlining, so nothing here evaluates them; a
    // duplicate is the program's error and belongs to the ordinary path's
    // diagnostic. Either way the refusal below still answers.
    const FunctionSignature &recursiveSig =
        method.bodySignature.callable ? method.bodySignature : method.signature;
    llvm::SmallVector<Value, 8> placed;
    bool keywordsPlaced = true;
    if (!keywords.empty()) {
      // Slot 0 is `self`, which the receiver fills.
      llvm::ArrayRef<std::string> names = recursiveSig.positionalNames;
      if (names.empty() || positional.size() + keywords.size() + 1 !=
                               names.size()) {
        keywordsPlaced = false;
      } else {
        placed.assign(names.size() - 1, Value{});
        for (auto [index, argument] : llvm::enumerate(positional))
          placed[index] = argument;
        for (const auto &entry : keywords) {
          auto found = llvm::find(names, entry.getKey().str());
          if (found == names.end() || found == names.begin()) {
            keywordsPlaced = false;
            break;
          }
          std::size_t slot =
              static_cast<std::size_t>(std::distance(names.begin(), found)) - 1;
          if (slot < positional.size() || placed[slot].value) {
            keywordsPlaced = false;
            break;
          }
          placed[slot] = entry.getValue();
        }
        if (keywordsPlaced)
          for (Value slot : placed)
            if (!slot.value)
              keywordsPlaced = false;
      }
    }
    if (!recursive || method.symbolName.empty() || !bindDescriptorReceiver ||
        !keywordsPlaced) {
      std::string cycle;
      bool inCycle = false;
      for (const parser::Node *active : methodsBeingInlined) {
        if (active == method.method)
          inCycle = true;
        if (!inCycle)
          continue;
        if (auto activeName = ast::string(*active, "name"))
          cycle += std::string(*activeName) + " -> ";
      }
      if (auto selfName = ast::string(*method.method, "name"))
        cycle += std::string(*selfName);
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, anchor.range.start,
          "recursive class method call is not supported (" + cycle +
              "): class method bodies are inlined at their call sites, so a "
              "call cycle has no base case to stop the expansion"});
      return emitNone(anchor);
    }
    // ⭐ THE CLASS'S OWN BODY ONLY FOR A RECEIVER OF EXACTLY THAT CLASS. A
    // recursion can come back through another receiver: unittest's
    // `TestSuite.run` calls `test(result)` on the suites it holds, which
    // reaches `BaseTestSuite.__call__` again with a member typed by the base,
    // and the standalone body -- whose `self` is the defining class -- ran
    // `BaseTestSuite.run` for a nested TestSuite, skipping the class fixtures
    // in silence. The open body asks the receiver's class, as the large-method
    // calls do (`exactBody` there).
    std::string recursiveSymbol = method.symbolName;
    if (auto receiverContract =
            mlir::dyn_cast_if_present<py::ContractType>(receiver.type);
        receiverContract && method.kind == "instance") {
      mlir::Operation *definition = receiver.value.getDefiningOp();
      bool constructed =
          definition && mlir::isa<py::NewOp, py::InitOp>(definition);
      bool leaf = !classHasDeclaredSubclass(receiverContract.getContractName());
      // `self` of the standalone body names its class.
      bool standaloneSelf = false;
      if (!superContexts.empty() && superContexts.back().selfIsExact)
        if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(receiver.value))
          standaloneSelf = argument.getArgNumber() == 0 &&
                           argument.getOwner() &&
                           argument.getOwner()->isEntryBlock();
      bool exact = (constructed || leaf || standaloneSelf) &&
                   !recursiveSig.positionalTypes.empty() &&
                   recursiveSig.positionalTypes.front() == receiver.type;
      if (!exact && !recursiveSig.positionalTypes.empty()) {
        FunctionSignature openSig = recursiveSig;
        openSig.positionalTypes.front() = receiver.type;
        types.refreshCallable(openSig);
        if (auto openCallable = mlir::dyn_cast_if_present<py::CallableType>(
                openSig.publicCallable ? openSig.publicCallable
                                       : openSig.callable)) {
          recursiveSymbol = openMethodBody(method, receiver.type, openSig);
          recursive = openCallable;
        }
      }
    }
    Value callee = emitBindingRef(anchor, recursiveSymbol, recursive);
    llvm::SmallVector<Value, 8> arguments{receiver};
    if (keywords.empty())
      arguments.append(positional.begin(), positional.end());
    else
      arguments.append(placed.begin(), placed.end());
    CallOperands operands;
    operands.positional.assign(arguments.begin(), arguments.end());
    for (Value argument : arguments)
      operands.positionalTypes.push_back(argument.type);
    operands.positionalUnpacked.assign(arguments.size(), false);
    return emitCallableDispatch(anchor, callee, operands);
  }
  const FunctionSignature &declaredSig =
      method.bodySignature.callable ? method.bodySignature : method.signature;
  // ⭐ A GENERIC METHOD IS SPECIALIZED HERE, at the one place its arguments
  // are known: T is solved from them the way a generic function's call
  // solves it, the signature is read at the solution, and the body below
  // sees T as that type. Without it the body was emitted with T itself,
  // which has no methods: `def assertEqual[T](self, a: T, b: T)` refused
  // `a == b` while the same def at module level compiled.
  std::optional<FunctionSignature> specializedSig;
  TypeBindingMap typeArguments;
  llvm::SmallVector<Value, 8> adoptedPositional;
  if (hasTypeParameters(*method.method)) {
    py::CallableType declared =
        mlir::dyn_cast_if_present<py::CallableType>(declaredSig.callable);
    // The arguments were emitted before T is solved: an empty container
    // literal is solved as nothing (emptyLiteralSolvingType) and adopts what
    // its parameter became below, as at a generic function's call.
    std::optional<EmptyLiteralArguments> emptyLiterals;
    if (!collectedVararg && !collectedKwargs)
      emptyLiterals = emptyLiteralArgumentsOf(anchor);
    if (emptyLiterals && emptyLiterals->positional.size() != positional.size())
      emptyLiterals.reset();
    llvm::SmallVector<mlir::Type, 8> argumentTypes{receiver.type};
    for (const Value &argument : positional)
      argumentTypes.push_back(argument.type);
    if (collectedVararg) {
      lython::callable::VarargShape<mlir::Type> shape =
          py::callableVarargShape(collectedVararg->type);
      if (shape.valid() && shape.repeated)
        argumentTypes.push_back(py::variadicTailType(*shape.repeated));
    }
    llvm::SmallVector<CallKeywordType, 4> keywordTypes;
    for (const auto &entry : keywords)
      keywordTypes.push_back(
          CallKeywordType{entry.getKey().str(), entry.getValue().type});
    if (collectedKwargs)
      keywordTypes.push_back(CallKeywordType{"", collectedKwargs->type});
    CallInferenceResult solved;
    if (declared && emptyLiterals) {
      llvm::SmallVector<mlir::Type, 8> solvingTypes(argumentTypes);
      llvm::SmallVector<CallKeywordType, 4> solvingKeywords(keywordTypes);
      bool marked = false;
      for (auto [index, isEmpty] : llvm::enumerate(emptyLiterals->positional))
        if (isEmpty)
          if (mlir::Type marker =
                  emptyLiteralSolvingType(solvingTypes[index + 1], types)) {
            solvingTypes[index + 1] = marker;
            marked = true;
          }
      for (CallKeywordType &keyword : solvingKeywords)
        if (emptyLiterals->keywords.contains(keyword.name))
          if (mlir::Type marker = emptyLiteralSolvingType(keyword.type, types)) {
            keyword.type = marker;
            marked = true;
          }
      if (marked) {
        solved = types.inferCallWithEvidence(declared, solvingTypes,
                                             solvingKeywords);
        if (solved &&
            (mentionsEmptyLiteralMarker(solved.resultType) ||
             mentionsEmptyLiteralMarker(solved.evidence.callableContract)))
          solved = CallInferenceResult();
      }
      if (!solved)
        emptyLiterals.reset();
    }
    if (declared && !solved)
      solved = types.inferCallWithEvidence(declared, argumentTypes,
                                           keywordTypes);
    auto resolved = solved ? mlir::dyn_cast_if_present<py::CallableType>(
                                 solved.evidence.callableContract)
                           : py::CallableType();
    // What a call that leaves a default out passes (see
    // `omittedDefaultArgumentTypes`).
    if (declared && !collectedVararg &&
        !bindSolvedInstantiation(types, declared, resolved, typeArguments)) {
      typeArguments.clear();
      llvm::SmallVector<CallKeywordType, 2> defaults =
          omittedDefaultArgumentTypes(
              *method.method, declaredSig,
              static_cast<unsigned>(argumentTypes.size()), keywordTypes);
      if (!defaults.empty()) {
        keywordTypes.append(defaults.begin(), defaults.end());
        solved = types.inferCallWithEvidence(declared, argumentTypes,
                                             keywordTypes);
        resolved = solved ? mlir::dyn_cast_if_present<py::CallableType>(
                                solved.evidence.callableContract)
                          : py::CallableType();
      }
    }
    if (!bindSolvedInstantiation(types, declared, resolved, typeArguments)) {
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, anchor.range.start,
          "this call does not determine the type arguments of the generic "
          "method '" +
              std::string(ast::string(*method.method, "name").value_or("")) +
              "'"});
      return emitNone(anchor);
    }
    specializedSig = declaredSig;
    auto substitute = [&](mlir::Type type) {
      return type ? substituteType(types, type, typeArguments) : type;
    };
    for (mlir::Type &type : specializedSig->positionalTypes)
      type = substitute(type);
    for (mlir::Type &type : specializedSig->kwOnlyTypes)
      type = substitute(type);
    specializedSig->varargType = substitute(specializedSig->varargType);
    specializedSig->callableVarargType =
        substitute(specializedSig->callableVarargType);
    specializedSig->kwargType = substitute(specializedSig->kwargType);
    specializedSig->resultType = substitute(specializedSig->resultType);
    std::size_t firstSlot = methodBindingBindsReceiver(method) ? 1 : 0;
    adoptedPositional.assign(positional.begin(), positional.end());
    // `addCleanup(int, "x")`: the `Callable[P, R]` parameter takes the shape
    // of what this call passes on (concretizeParamSpecCallables), and a
    // builtin or a class given to it becomes a forwarder of that shape.
    if (specializedSig->paramSpecArgs && !collectedVararg && !collectedKwargs &&
        specializedSig->positionalTypes.size() >= firstSlot) {
      std::size_t fixed = specializedSig->positionalTypes.size() - firstSlot;
      llvm::SmallVector<mlir::Type, 4> packPositional;
      for (std::size_t index = fixed; index < positional.size(); ++index)
        packPositional.push_back(types.widenLiteral(positional[index].type));
      llvm::SmallVector<CallKeywordType, 2> packKeywords;
      for (const auto &entry : keywords)
        if (!llvm::is_contained(specializedSig->positionalNames,
                                entry.getKey().str()) &&
            !llvm::is_contained(specializedSig->kwOnlyNames,
                                entry.getKey().str()))
          packKeywords.push_back(
              CallKeywordType{entry.getKey().str(),
                              types.widenLiteral(entry.getValue().type)});
      llvm::SmallVector<bool, 4> forwarded(firstSlot, false);
      for (const Value &argument : positional)
        forwarded.push_back(needsCallableForwarder(argument));
      for (unsigned index : concretizeParamSpecCallables(
               *method.method, *specializedSig,
               static_cast<unsigned>(specializedSig->positionalTypes.size()),
               packPositional, packKeywords, forwarded)) {
        if (index < firstSlot || index - firstSlot >= adoptedPositional.size())
          continue;
        auto target = mlir::cast<py::CallableType>(
            specializedSig->positionalTypes[index]);
        Value &argument = adoptedPositional[index - firstSlot];
        std::optional<Value> wrapped =
            tryWrapBuiltinFunctionValue(argument, target, anchor);
        if (!wrapped)
          wrapped = tryWrapClassValue(argument, target, anchor);
        if (wrapped)
          argument = *wrapped;
      }
    }
    types.refreshCallable(*specializedSig);
    const auto *argumentNodes =
        anchor.kind == "Call" ? ast::nodeList(anchor, "args") : nullptr;
    if (argumentNodes && argumentNodes->size() != adoptedPositional.size())
      argumentNodes = nullptr;
    for (auto [index, argument] : llvm::enumerate(adoptedPositional)) {
      if (emptyLiterals && emptyLiterals->positional[index] &&
          firstSlot + index < specializedSig->positionalTypes.size())
        argument = adoptDeclaredContainer(
            argument, specializedSig->positionalTypes[firstSlot + index],
            anchor);
      else if (argumentNodes &&
               firstSlot + index < specializedSig->positionalTypes.size())
        argument = adoptSolvedDisplay(
            argument, specializedSig->positionalTypes[firstSlot + index],
            (*argumentNodes)[index].get());
    }
    positional = adoptedPositional;
    if (bindDescriptorReceiver && !collectedVararg && !collectedKwargs) {
      MethodSpecialization specialization{*specializedSig, typeArguments};
      if (std::optional<Value> called =
              tryCallLargeMethod(anchor, receiver, method, positional,
                                 keywords, &specialization))
        return *called;
    }
  }
  const FunctionSignature &sig = specializedSig ? *specializedSig : declaredSig;
  const auto *body = ast::nodeList(*method.method, "body");
  mlir::Type resultType = sig.resultType ? sig.resultType : types.none();

  ScopedEmitterScope scope(values, types);
  // Body annotations spell the type parameters by name (`x: T`).
  for (const auto &binding : typeArguments) {
    types.bindLocalSymbol(binding.first, binding.second);
    types.bindLocalTypeParameter(binding.first, binding.second);
  }
  // A method of a source-module class executes under ITS module's globals
  // (Python scoping), not the use site's: re-establish the defining
  // module's environment around the inlined body. Clearing `values` and
  // isolating the scope stack (instead of only pushing) keeps an unbound
  // name in the method body a diagnostic rather than a silent capture of a
  // use-site local.
  const EmitOptions::SourceModule *methodSource =
      sourceModuleForClass(method.definingClass);
  std::size_t crossModuleDiagnosticStart = diagnostics.size();
  std::optional<TypeSystem::ScopeIsolation> isolation;
  // ⛔ `values.clear()` and the scope isolation below are NOT the whole
  // environment. The module-scope value bindings live in three maps of their
  // own, and leaving the use site's in place is how an inlined stdlib method
  // read the PROGRAM's globals: `iterdir` has a local `names`, and any
  // annotated `names` in the program made line 264 of pathlib.py report
  // "'builtins.int' does not provide manifest method 'sort'". The read
  // resolved to the global because `isModuleGlobalRead` consults a map this
  // walk never swapped -- and at module scope the ASSIGNMENT above it went to
  // the global's cell too, so the local was never bound at all.
  std::optional<ImporterModuleScope> importerScope;
  std::optional<llvm::SaveAndRestore<std::string>> savedSourceName;
  std::optional<llvm::SaveAndRestore<std::string>> savedPackageName;
  llvm::SmallVector<LoopControlContext, 4> savedLoopContexts;
  auto crossModuleCleanup = llvm::scope_exit([&] {
    if (!methodSource)
      return;
    // Attribute before SaveAndRestore rolls sourceName back (destruction
    // order): these diagnostics point into the defining module's source.
    for (std::size_t index = crossModuleDiagnosticStart;
         index < diagnostics.size(); ++index)
      if (diagnostics[index].filename.empty())
        diagnostics[index].filename = sourceName;
    loopControlContexts = std::move(savedLoopContexts);
  });
  std::optional<TypeSystem::Scope> moduleScope;
  if (methodSource) {
    values.clear();
    importerScope.emplace(*this);
    isolation.emplace(types.isolateScopes());
    savedSourceName.emplace(sourceName, methodSource->sourceName.empty()
                                            ? methodSource->moduleName
                                            : methodSource->sourceName);
    savedPackageName.emplace(activePackageName, methodSource->packageName);
    savedLoopContexts = std::move(loopControlContexts);
    loopControlContexts.clear();
    moduleScope.emplace(types.pushScope());
    bindModuleImportScope(*methodSource->moduleNode,
                          /*diagnoseUnsupported=*/false);
    bindSourceModuleLocals(methodSource->moduleName, *methodSource->moduleNode,
                           methodSource->isStub);
  } else if (sourceName != mainSourceName) {
    // ⭐ A MAIN-MODULE METHOD WRITTEN INTO AN IMPORTED MODULE'S CODE is still
    // main.py's. `getattr(self, name)()` in an imported runner inlines the
    // program's test method there, and its frames carried the runner's file
    // against the test's lines -- which a traceback filter by file then
    // dropped as the runner's own.
    savedSourceName.emplace(sourceName, mainSourceName);
  }
  // A generic class's method body still spells its type parameters (`item:
  // T`), and this inlining happens far from the specialization's emission
  // scope, so the solved arguments are re-established from the defining
  // class's contract name. Inside the cross-module isolation above, so an
  // imported generic keeps them too.
  bindClassTypeArguments(method.definingClass);
  // And the method's own, solved above: the isolation hid the scope they
  // were bound in, so a def nested in an imported generic method spelled
  // `T` as a class of that name (`builtins.T`) -- a closure the call then
  // found no target for, `@skip("...")` through an imported `skip` with a
  // generic `__call__`.
  if (methodSource)
    for (const auto &binding : typeArguments) {
      types.bindLocalSymbol(binding.first, binding.second);
      types.bindLocalTypeParameter(binding.first, binding.second);
    }
  // ⭐ AN INLINED BODY IS STILL A FUNCTION BODY, and a nested `def` inside it
  // needs the same promotion a free function's gets. `emitCallableFunction`
  // computes `nonlocalBoxedNames` so that a local a nested function declares
  // `nonlocal` becomes a shared cell; this path never did, so
  //
  //     class C:
  //         def go(self) -> None:
  //             total = 0
  //             def bump(n: int) -> None:
  //                 nonlocal total
  //                 total += n
  //             bump(2)
  //
  // was refused with "no binding for nonlocal 'total' found (the target must
  // be assigned in an enclosing function before this definition)" -- a
  // sentence whose own condition the program plainly satisfies. The identical
  // code in a free function has always worked, and CAPTURE BY READ inside a
  // method works too, which is what says the gap is the promotion and not the
  // nesting.
  llvm::StringSet<> savedInlineBoxedLocals = std::move(currentBoxedLocals);
  currentBoxedLocals = nonlocalBoxedNames(*method.method);
  llvm::scope_exit restoreInlineBoxedLocals(
      [&] { currentBoxedLocals = std::move(savedInlineBoxedLocals); });
  // ⭐ THE BODY'S OWN LOCALS SHADOW THE USE SITE'S NAMES. An inlined body is
  // emitted in the caller's `values`, so a method local whose name also stands
  // at the use site inherited that binding: with `a = "hello"` at module scope,
  // `for a in xs:` inside a method made the loop target a carried local of the
  // GLOBAL's type, joined to the element's, and the program was refused with
  // "cannot adapt runtime bundle builtins.int ... to (memref<16xi64>, ...)" --
  // an erased union nothing in the source asked for. The same method as a free
  // function has always been correct, because a real function body never sees
  // the caller's frame.
  //
  // ⛔ NOT `collectAssignedNames`, which is the loop walker's over-
  // approximation: it reports `xs.append(...)` receivers and `x[k] = v`
  // containers, and erasing those would unbind a module-level list a method
  // only MUTATES -- which Python keeps global.
  //
  // ⛔ Names the body declares `global` are excluded: those writes go to the
  // module cell by declaration, so the use-site binding is the right one.
  // Cross-module methods take the wholesale `values.clear()` above instead;
  // this is the same rule for a class defined in the module being compiled.
  if (!methodSource) {
    llvm::StringSet<> globalDecls = moduleGlobalDeclarations(*method.method);
    for (const auto &local : functionLocalNames(*method.method))
      if (!globalDecls.contains(local.getKey()))
        values.erase(local.getKey());
  }
  llvm::StringSet<> bound;
  auto bind = [&](llvm::StringRef name, Value value) {
    values[name] = value;
    types.bindSymbol(name, value.type);
    bound.insert(name);
  };

  // ⭐ THE DECLARED PARAMETER TYPE IS CHECKED HERE, and nowhere else can. An
  // inlined body binds the argument VALUE, so an argument of the wrong type is
  // substituted into the body and the call succeeds or fails on whatever the
  // body happens to do with it. `def take(self, xs: list[str])` reached by
  // `c.take({"a": 1})` compiled and answered `len(dict)`; `collections.Counter`
  // is the same shape and it is the reason this was found --
  // `Counter({"x": 2, "y": 1})` has `list[str] | None` declared, iterated the
  // dict's KEYS, and every count came out 1 (CPython: 2 and 1). Silent, and a
  // free function of the same signature has always refused it
  // ("call arguments do not match the Callable contract").
  //
  // ⛔ Why CHECK and not COERCE: `x: float` reached by `3` must keep the int.
  // CPython leaves an annotation inert at a parameter, and the numeric tower
  // makes int assignable to float, so this passes it through unchanged --
  // which is what `tests/probe/wb_argument_boundary_numeric_tower.py` measures
  // and what the free-function specializer preserves at its own boundary.
  //
  // ⛔ Why an UNGROUND declared type is skipped rather than unified: a generic
  // class's method still spells `T` here even after `bindClassTypeArguments`
  // when the receiver's own arguments did not reach it, and refusing on
  // `list[int]` vs `list[T]` would reject the specializations this walk exists
  // to emit. A type parameter that IS bound has already been substituted, so
  // the skip costs only the cases the class table could not ground.
  //
  // ⛔ Why a SYNTHESIZED method is exempt: its signature is this compiler's
  // spelling, not the program's. `@dataclass`/`NamedTuple` give `__eq__` the
  // parameter type `Self`, and Python's data model gives it `object` --
  // `TupleA(1) == TupleB(1)` is True in CPython and
  // `inherited_post_init_and_cross_class_equality` pins it. Checking a
  // signature nobody wrote against a rule nobody stated is how a correct
  // program gets refused.
  bool methodIsSynthesized = llvm::any_of(
      synthesizedClassMethods, [&](const parser::NodePtr &synthesized) {
        return synthesized.get() == method.method;
      });
  // Forgotten when the body is done: a value of this emission means nothing
  // to the next one, and an erased op's address can be reused.
  llvm::SmallVector<mlir::Value, 2> heldAsObject;
  llvm::SmallVector<mlir::Value, 2> heldPacks;
  llvm::scope_exit forgetHeldAsObject([&] {
    for (mlir::Value held : heldAsObject)
      numbersHeldAsObject.erase(held);
    for (mlir::Value held : heldPacks) {
      staticPositionalPacks.erase(held);
      staticKeywordPacks.erase(held);
    }
  });
  auto checkArgument = [&](llvm::StringRef name, mlir::Type declared,
                           Value argument) {
    if (declared && argument.value &&
        types.widenLiteral(declared) == types.object() &&
        (types.widenLiteral(argument.type) == types.intType() ||
         types.widenLiteral(argument.type) == types.floatType()) &&
        numbersHeldAsObject.insert(argument.value).second)
      heldAsObject.push_back(argument.value);
    if (methodIsSynthesized || !declared || !argument.type)
      return;
    mlir::Type expected = types.widenLiteral(declared);
    mlir::Type actual = types.widenLiteral(argument.type);
    if (inlineArgumentFits(actual, expected))
      return;
    auto spell = [](mlir::Type type) {
      std::string text;
      llvm::raw_string_ostream stream(text);
      stream << type;
      return text;
    };
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, anchor.range.start,
        "argument '" + name.str() + "' of '" +
            std::string(ast::string(*method.method, "name")
                            .value_or(std::string_view("method"))) +
            "' is declared " + spell(expected) + " and this call gives it " +
            spell(actual)});
  };

  unsigned parameterIndex = 0;
  if (bindDescriptorReceiver && !sig.positionalNames.empty()) {
    bind(sig.positionalNames.front(), receiver);
    parameterIndex = 1;
  }

  // ⭐ `*args` ON A METHOD. The extras used to be "too many positional arguments
  // for inlined class method":
  //
  //     class Registry:
  //         def many(self, *items: str) -> int:
  //             return len(items)
  //     Registry().many("p", "q")
  //
  // while the free-function spelling of the same body had always worked -- a
  // real function binds its vararg parameter to the tuple the call packed, and
  // the inlined path had no such step. It has one now: what the declared
  // positionals do not take is packed into a tuple and bound to the vararg name,
  // which is exactly what the callee would have received.
  //
  // ⛔ The name is bound even when nothing is left over, because CPython binds
  // an EMPTY tuple there and `len(items)` must answer 0 rather than "unresolved
  // name". An empty `emitPack` is the empty tuple.
  llvm::SmallVector<Value, 4> variadic;
  for (Value argument : positional) {
    if (parameterIndex >= sig.positionalNames.size()) {
      if (sig.varargName) {
        variadic.push_back(argument);
        continue;
      }
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, anchor.range.start,
          "too many positional arguments for inlined class method"});
      break;
    }
    if (parameterIndex < sig.positionalTypes.size())
      checkArgument(sig.positionalNames[parameterIndex],
                    sig.positionalTypes[parameterIndex], argument);
    // ⭐ A UNION PARAMETER TAKES THE UNION, not the member the call happened to
    // pass. An inlined body binds the argument VALUE, so `b.take(None)` against
    // `def take(self, n: int | None)` bound a `literal<None>` and the body's
    // `n is None` narrowing then had nothing to narrow:
    //
    //     cannot adapt runtime bundle types.NoneType with physical values ...
    //
    // The free-function spelling works because its call emits operands against
    // the declared callable, which wraps; a DEFAULT of None works for the same
    // reason. `Counter.most_common(None)` is this defect reached through the
    // shipped stdlib.
    if (parameterIndex < sig.positionalTypes.size())
      if (mlir::isa_and_nonnull<py::UnionType>(
              sig.positionalTypes[parameterIndex]) &&
          !mlir::isa_and_nonnull<py::UnionType>(argument.type))
        argument = coerceValue(argument, sig.positionalTypes[parameterIndex],
                               anchor);
    bind(sig.positionalNames[parameterIndex++], argument);
  }
  if (sig.varargName && collectedVararg) {
    // The tuple the call collected from a starred argument of unknown length
    // (emitDynamicUnpack): the whole of *args, with no static pack to forward.
    bind(*sig.varargName, *collectedVararg);
  } else if (sig.varargName) {
    // ⛔ An EMPTY pack has no element type to infer, so `emitPack({})` gives
    // `tuple[object]` and the body's `for n in rest` then reports "list
    // iteration evidence match/value count mismatch". An empty tuple LITERAL
    // under the declared vararg type takes that type instead -- the same
    // expectation machinery an annotated `xs: list[int] = []` uses.
    Value packed;
    if (variadic.empty() && sig.varargType)
      packed = emitExprExpected(
          synth::tuple(std::vector<parser::NodePtr>{}, anchor.range).get(),
          sig.varargType);
    else
      packed = emitPack(variadic);
    // ⭐ A PACK EVERY VALUE FITS KEEPS WHAT IT HOLDS. `*args: object` (and a
    // ParamSpec's `P.args`, which arrives as Any) takes anything, and coercing
    // the pack to the declared `tuple[object]` erased its arity, so the one
    // thing such a parameter is for -- `f(*args)` -- was "starred call
    // arguments require a statically sized tuple". The arguments are bound as
    // they came, as an inlined method binds every other argument.
    auto acceptsEverything = [&](mlir::Type vararg) {
      auto tuple = mlir::dyn_cast_if_present<py::ContractType>(vararg);
      if (!tuple || tuple.getContractName() != "builtins.tuple" ||
          tuple.getArguments().size() != 1)
        return false;
      auto element = mlir::dyn_cast_if_present<py::ContractType>(
          tuple.getArguments().front());
      return element && (element.getContractName() == "builtins.object" ||
                         element.getContractName() == "typing.Any");
    };
    if (sig.varargType && packed.type != sig.varargType &&
        !(!variadic.empty() && acceptsEverything(sig.varargType)))
      packed = coerceValue(packed, sig.varargType, anchor);
    else if (packed.value &&
             staticPositionalPacks.try_emplace(packed.value, variadic).second)
      heldPacks.push_back(packed.value);
    bind(*sig.varargName, packed);
  }

  llvm::SmallVector<std::pair<std::string, Value>, 4> variadicKeywords;
  auto bindKeyword = [&](llvm::StringRef name, Value value) {
    for (auto [index, positionalName] : llvm::enumerate(sig.positionalNames)) {
      if (positionalName != name)
        continue;
      if (bound.contains(name)) {
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, anchor.range.start,
            "multiple values for inlined class method argument '" + name.str() +
                "'"});
        return;
      }
      if (index < sig.positionalTypes.size()) {
        checkArgument(name, sig.positionalTypes[index], value);
        // Keywords take the same union wrap as the positionals above.
        if (mlir::isa_and_nonnull<py::UnionType>(sig.positionalTypes[index]) &&
            !mlir::isa_and_nonnull<py::UnionType>(value.type))
          value = coerceValue(value, sig.positionalTypes[index], anchor);
      }
      bind(name, value);
      return;
    }
    for (auto [index, kwOnlyName] : llvm::enumerate(sig.kwOnlyNames)) {
      if (kwOnlyName != name)
        continue;
      if (index < sig.kwOnlyTypes.size()) {
        checkArgument(name, sig.kwOnlyTypes[index], value);
        if (mlir::isa_and_nonnull<py::UnionType>(sig.kwOnlyTypes[index]) &&
            !mlir::isa_and_nonnull<py::UnionType>(value.type))
          value = coerceValue(value, sig.kwOnlyTypes[index], anchor);
      }
      bind(name, value);
      return;
    }
    // ⭐ `**kwargs` ON A METHOD collects what no parameter claimed. Without this
    // the extras were "unexpected keyword argument 'b'", while the same body as a
    // free function bound them -- the mirror of the `*args` gap above.
    //
    // ⛔ Built through `LyValueRef`, because the values are already EMITTED and a
    // dict literal is written in AST: each value goes into `pendingValueRefs` and
    // the synthesized Dict names it by slot, which is the same machinery the
    // augmented-assignment rewrite uses to avoid evaluating a subexpression twice.
    if (sig.kwargName) {
      variadicKeywords.push_back({name.str(), value});
      return;
    }
    diagnostics.push_back(
        parser::Diagnostic{parser::Severity::Error, anchor.range.start,
                           "unexpected keyword argument '" + name.str() +
                               "' for inlined class method"});
  };
  for (const auto &entry : keywords)
    bindKeyword(entry.getKey(), entry.getValue());
  if (sig.kwargName && collectedKwargs) {
    // The fresh dict the call collected from a mapping of unknown keys
    // (emitDynamicUnpack); every other keyword bound a named parameter.
    bind(*sig.kwargName, *collectedKwargs);
  } else if (sig.kwargName) {
    Value packedKeywords =
        emitKeywordPackDict(anchor, variadicKeywords, sig.kwargType);
    if (packedKeywords.value &&
        keywordPackIsOnlyRead(*method.method, *sig.kwargName) &&
        staticKeywordPacks.try_emplace(packedKeywords.value, variadicKeywords)
            .second)
      heldPacks.push_back(packedKeywords.value);
    bind(*sig.kwargName, packedKeywords);
  }

  const parser::Node *arguments = ast::node(*method.method, "args");
  llvm::SmallVector<const parser::Node *, 8> positionalNodes;
  if (arguments)
    positionalNodes = positionalArgumentNodes(*arguments);
  const auto *defaults =
      arguments ? ast::nodeList(*arguments, "defaults") : nullptr;
  const auto *kwDefaults =
      arguments ? ast::nodeList(*arguments, "kw_defaults") : nullptr;
  unsigned firstPositionalDefault =
      defaults && defaults->size() <= positionalNodes.size()
          ? positionalNodes.size() - defaults->size()
          : positionalNodes.size();
  auto positionalDefault = [&](unsigned index) -> const parser::Node * {
    if (!defaults || index < firstPositionalDefault)
      return nullptr;
    unsigned defaultIndex = index - firstPositionalDefault;
    if (defaultIndex >= defaults->size())
      return nullptr;
    return (*defaults)[defaultIndex].get();
  };
  auto reportMissing = [&](llvm::StringRef name) {
    diagnostics.push_back(
        parser::Diagnostic{parser::Severity::Error, anchor.range.start,
                           "missing required argument '" + name.str() +
                               "' for inlined class method"});
  };
  // ⭐ THE DEFAULT'S CELL, not the default's EXPRESSION. An omitted argument
  // used to re-emit the expression here, at the call site:
  //
  //     class Bag:
  //         def add(self, into: list[int] = []) -> int:
  //             into.append(1)
  //             return len(into)
  //     b = Bag(); print(b.add(), b.add())   # printed 1 1; CPython prints 1 2
  //
  // -- a fresh list per call, and a side-effecting default (`n: int = make()`)
  // firing once per call as well. CPython evaluates a def's defaults ONCE, when
  // the def statement executes, and for a method that is class-body time. The
  // free-function spelling was already right because its call sites read the
  // callable's default-value attributes, where the cell lives; an inlined method
  // has no such call, so it reads the cell by name.
  auto slotType = [&](unsigned slot) -> mlir::Type {
    if (slot < sig.positionalTypes.size())
      return sig.positionalTypes[slot];
    unsigned kwIndex = slot - static_cast<unsigned>(sig.positionalTypes.size());
    if (kwIndex < sig.kwOnlyTypes.size())
      return sig.kwOnlyTypes[kwIndex];
    return types.object();
  };
  auto emitDefaultFor = [&](unsigned slot,
                            const parser::Node *defaultNode) -> Value {
    if (std::optional<std::string> cellName =
            methodDefaultCell(method, slot, slotType(slot))) {
      // ⛔ NOT `markBoxedModuleGlobal`: a default cell is not a module global
      // even though both are py.global.get/set, and the lowering says so at the
      // read ("this population is never marked `ly.global.boxed`, so an int
      // default stays in the native word cell"). Marking it made an int default
      // fail with "module global ... referenced before assignment", because the
      // store the class statement emitted went to the other cell.
      auto get = py::GlobalGetOp::create(builder, loc(anchor), slotType(slot),
                                         builder.getStringAttr(*cellName));
      return Value{get.getResult(), get.getResult().getType()};
    }
    return emitExpr(defaultNode);
  };
  // ⭐ A DEFAULT BINDS AS AN ARGUMENT WOULD. Into a protocol parameter, an
  // argument is bound as it came -- `SequenceMatcher(None, '', '')` binds two
  // strs -- while the default was viewed as the protocol, and a protocol
  // view's evidence did not survive into a generator body's state machine:
  // `SequenceMatcher(isjunk)` inside one reached "protocol-typed receiver
  // ... has no concrete runtime method evidence for __next__".
  auto coerceDefault = [&](Value value, mlir::Type declared) -> Value {
    mlir::Type own = types.widenLiteral(value.type);
    if (mlir::isa_and_nonnull<py::ProtocolType>(declared) &&
        mlir::isa_and_nonnull<py::ContractType>(own) &&
        py::isAssignableTo(own, declared))
      return coerceValue(value, own, anchor);
    return coerceValue(value, declared, anchor);
  };
  for (auto [index, name] : llvm::enumerate(sig.positionalNames)) {
    if (bound.contains(name))
      continue;
    if (const parser::Node *defaultNode =
            positionalDefault(static_cast<unsigned>(index))) {
      Value defaultValue =
          emitDefaultFor(static_cast<unsigned>(index), defaultNode);
      bind(name, coerceDefault(defaultValue, sig.positionalTypes[index]));
      continue;
    }
    reportMissing(name);
    bind(name, emitNone(anchor));
  }
  for (auto [index, name] : llvm::enumerate(sig.kwOnlyNames)) {
    if (bound.contains(name))
      continue;
    const parser::Node *defaultNode = nullptr;
    if (kwDefaults && index < kwDefaults->size())
      defaultNode = (*kwDefaults)[index].get();
    if (defaultNode) {
      // Keyword-only slots are numbered after the positionals, the same way
      // `emitCallableDefaultValues` numbered them when it parked the cell.
      Value defaultValue =
          emitDefaultFor(static_cast<unsigned>(sig.positionalNames.size() +
                                               index),
                         defaultNode);
      bind(name, coerceDefault(defaultValue, sig.kwOnlyTypes[index]));
      continue;
    }
    reportMissing(name);
    bind(name, emitNone(anchor));
  }

  mlir::Block *entryBlock = builder.getInsertionBlock();
  mlir::Region *region = entryBlock ? entryBlock->getParent() : nullptr;
  if (!region) {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, anchor.range.start,
        "inlined class method call requires an active insertion region"});
    return emitNone(anchor);
  }
  mlir::Block *continuation =
      entryBlock->splitBlock(builder.getInsertionPoint());
  continuation->addArgument(resultType, loc(anchor));
  mlir::Block *bodyBlock =
      builder.createBlock(region, continuation->getIterator());

  builder.setInsertionPointToEnd(entryBlock);
  mlir::cf::BranchOp::create(builder, loc(anchor), bodyBlock);
  builder.setInsertionPointToStart(bodyBlock);
  inlineReturnContexts.push_back(InlineReturnContext{
      continuation, resultType, /*carryResult=*/true, /*endsFrame=*/true});
  std::vector<std::string> savedFrameLocals =
      std::exchange(currentFrameLocals, frameLocalOrder(*method.method));
  llvm::scope_exit restoreFrameLocals(
      [&] { currentFrameLocals = std::move(savedFrameLocals); });
  llvm::SaveAndRestore<std::optional<NotImplementedFallback>> savedFallback(
      notImplementedFallback,
      comparisonDunderFallback(ast::string(*method.method, "name").value_or(""),
                               sig.positionalNames));
  // ⭐ The inlined body's return type, for the statements that ask for it
  // rather than for the context stack.
  //
  //     class C:
  //         def m(self) -> int:
  //             try:
  //                 return 1
  //             finally:
  //                 print("f")
  //     print(C().m())        # printed None; CPython prints 1
  //
  // `emitTry` decides whether a value can be returned through its completion
  // machinery by reading `currentReturnType`, which the inliner never set --
  // so inside every inlined method it read the ENCLOSING function's type, and
  // for a `-> None` caller that disabled the value path and left the return
  // yielding nothing. The same method as a free function was always correct,
  // because there the ordinary emission sets it.
  mlir::Type enclosingReturnType = currentReturnType;
  currentReturnType = resultType;
  // ⛔ THE UNBOUND SPELLING PUSHES ONE TOO. `bindDescriptorReceiver` says
  // whether the receiver takes slot 0 implicitly, which is not what `super()`
  // asks: the body's zero-argument super() names the method's DEFINING class
  // and its first parameter, and both are the same whether the call was
  // `b.who()` or `B.who(b)`. Requiring the bound form made the virtual
  // dispatcher's fallback arm -- which is written `B.who(__ly_recv)` -- inline
  // a body whose `super()` then had no context:
  //
  //     class A: ...          # who()
  //     class B(A): ...       # who() calls super().who()
  //     class C(B): ...       # any override at all
  //     for x in [A(), B(), C()]: x.who()
  //
  // Two levels worked, because the arm for B resolves B's body directly; three
  // needs a dispatcher for `B.who` as well, and its fallback is the unbound
  // call. The report named B's own `super()` line, in a program whose two-level
  // form compiles.
  bool pushedSuperContext = method.kind == "instance" &&
                            !method.definingClass.empty() &&
                            !sig.positionalNames.empty();
  if (pushedSuperContext)
    superContexts.push_back(
        SuperContext{method.definingClass, sig.positionalNames.front()});
  // The inlined body is another object's frame: its `self` is not the caller's,
  // so a field path proved out here says nothing inside.
  llvm::StringMap<mlir::Type> savedInlineMembers = narrowedMemberTypes;
  narrowedMemberTypes.clear();
  auto restoreInlineMembers = llvm::scope_exit(
      [&] { narrowedMemberTypes = savedInlineMembers; });
  methodsBeingInlined.push_back(method.method);
  // The frame this body would have had if it were a function: everything it
  // emits is located inside `bad`, at a call written in whatever function the
  // inliner is currently writing into.
  InlineFrame frame;
  frame.calleeName = ast::string(*method.method, "name").value_or("<lambda>");
  if (!inlineFrames.empty())
    frame.callerName = inlineFrames.back().calleeName;
  frame.callerFile = callerSourceName;
  frame.line = anchor.range.start.line;
  frame.column = anchor.range.start.column;
  frame.endLine = anchor.range.end.line;
  frame.endColumn = anchor.range.end.column;
  frame.noAnchor = anchorlessCall == &anchor;
  inlineFrames.push_back(std::move(frame));
  // The stub body the function path substitutes, substituted here too: an
  // inlined `def area(self) -> int: ...` has no value to fall through with.
  if (isEllipsisStubBody(body) && resultType != types.none()) {
    parser::NodePtr stub = synth::raiseStmt(
        synth::call(synth::name("NotImplementedError", anchor.range),
                    {synth::strConstant(
                        ast::string(*method.method, "name")
                                .value_or("this function")
                                .data() +
                            std::string(" is a stub"),
                        anchor.range)},
                    anchor.range),
        anchor.range);
    emitStatement(*stub);
    synthesizedIteratorDefs.push_back(std::move(stub));
  } else {
    // ⛔ Here and not beside the `currentBoxedLocals` promotion above: the
    // cells are ops, and the parameter binding between the two is what puts
    // the builder inside the region they belong to.
    //
    // ⛔ AND THE CLOSED SIBLING GROUP TOO, for the same reason the cells are
    // here: a method reached by INLINING does not go through
    // emitCallableFunction, so a mutually recursive pair declared in one kept
    // its cell -- and its reference cycle -- while the same pair in a free
    // function stopped leaking. The two call sites have to agree.
    SiblingGroupState enclosingSiblings = std::move(closedSiblingGroup);
    closedSiblingGroup = SiblingGroupState{};
    auto restoreSiblings = llvm::scope_exit(
        [&] { closedSiblingGroup = std::move(enclosingSiblings); });
    MutualNestedDefGroup group = mutualNestedDefGroup(*method.method);
    if (!group.members.empty())
      if (const auto *groupBody = ast::nodeList(*method.method, "body")) {
        closedSiblingGroup.captures = std::move(group.captures);
        for (const parser::NodePtr &statement : *groupBody) {
          if (!statement || statement->kind != "FunctionDef")
            continue;
          auto memberName = ast::string(*statement, "name");
          if (!memberName || !group.members.contains(*memberName))
            continue;
          closedSiblingGroup.members[*memberName] = ClosedSibling{
              nestedFunctionSymbolName(*memberName, *statement),
              statement.get()};
        }
      }
    emitForwardBoundCells(*method.method);
    emitStatements(body);
  }
  inlineFrames.pop_back();
  methodsBeingInlined.pop_back();
  currentReturnType = enclosingReturnType;
  if (pushedSuperContext)
    superContexts.pop_back();
  inlineReturnContexts.pop_back();
  if (!insertionBlockTerminated(builder)) {
    // ⛔ AND NOT WHERE NOTHING REACHES THE END. A body whose only exit is a
    // return leaves an unreachable tail behind -- `while True:` with no break
    // builds one, and it is how a recursive-descent parser's loops are
    // written -- so reporting a fallthrough there refused a program CPython
    // runs. The same syntactic question the function epilogue asks
    // (`bodyCanComplete`) answers it here, and the tail gets the raise that
    // path gets rather than a None the declared result cannot hold.
    if (resultType != types.none() &&
        bodyCanComplete(ast::nodeList(*method.method, "body"))) {
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, method.method->range.start,
          "inlined class method can fall through without returning a value"});
    }
    Value none = emitNone(anchor);
    emitFrameExitKeepAlives(anchor);
    Value result = coerceValue(none, resultType, anchor);
    mlir::cf::BranchOp::create(builder, loc(anchor), continuation,
                               result.value);
  }
  builder.setInsertionPointToStart(continuation);
  return {continuation->getArgument(0), resultType};
}

Value ModuleEmitter::emitClassInstantiation(const parser::Node &expr,
                                            llvm::StringRef name,
                                            mlir::Type instanceType) {
  // ⭐ A CONSTRUCTOR ARGUMENT IS EMITTED AGAINST THE PARAMETER IT FILLS, the
  // same way a free function's is. Without the expectation an empty literal
  // came out `list[object]` and the declared-parameter check refused the call
  // it was written for:
  //
  //     class C:
  //         def __init__(self, xs: list[int]) -> None: ...
  //     C([])
  //     # argument 'xs' of '__init__' is declared list[int] and this call
  //     # gives it list[object]
  //
  // `def f(xs: list[int])` called as `f([])` was always fine, because
  // `emitCallOperands` distributes the callee's positional types there. The
  // constructor asked for the operands with no contract at all.
  //
  // `self` is dropped from the front: the call site writes the arguments after
  // it.
  py::CallableType initExpectation;
  if (std::optional<MethodBinding> declaredInit =
          lookupClassMethod(instanceType, "__init__"))
    if (auto declared = mlir::dyn_cast_if_present<py::CallableType>(
            declaredInit->bodySignature.callable
                ? declaredInit->bodySignature.callable
                : declaredInit->signature.callable)) {
      llvm::ArrayRef<mlir::Type> declaredPositional =
          declared.getPositionalTypes();
      if (!declaredPositional.empty() &&
          methodBindingBindsReceiver(*declaredInit))
        initExpectation = py::CallableType::get(
            &context, declaredPositional.drop_front(), declared.getKwOnlyTypes(),
            {}, {}, declared.getResultTypes());
    }
  CallOperands operands = emitCallOperands(expr, {}, /*includeAstArguments=*/true,
                                           initExpectation);
  // ⭐ A STARRED ARGUMENT OR A `**` MAPPING OF UNKNOWN SIZE reaches a program
  // class's `__init__` the way it reaches a method's: collected into its *args
  // and **kwargs (emitDynamicUnpack), which the inlined body binds.
  std::optional<Value> collectedVararg;
  std::optional<Value> collectedKwargs;
  llvm::SmallVector<mlir::Type, 8> solvePositional;
  llvm::SmallVector<CallKeywordType, 4> solveKeywords;
  if (!operands.valid && !operands.invalidBeyondDynamic &&
      (operands.dynamicTail || operands.dynamicMapping)) {
    std::optional<MethodBinding> declaredInit =
        lookupClassMethod(instanceType, "__init__");
    unsigned plain = operands.dynamicTail
                         ? *operands.dynamicTail
                         : static_cast<unsigned>(operands.positional.size());
    bool staticStar = false;
    for (unsigned index = 0; index < plain; ++index)
      staticStar = staticStar || operands.positionalUnpacked[index] != 0;
    if (declaredInit && declaredInit->method && !staticStar) {
      DynamicUnpackTarget target = dynamicUnpackTargetForMethod(*declaredInit);
      DynamicUnpackCall call;
      call.explicitPositional = plain;
      for (unsigned index = static_cast<unsigned>(
               target.positionalNames.size());
           index < plain; ++index)
        call.extras.push_back(operands.positional[index]);
      if (operands.dynamicTail)
        call.tail = operands.positional[*operands.dynamicTail];
      for (auto [index, value] : llvm::enumerate(operands.keywordValues))
        call.keywords.push_back({operands.keywordTypes[index].name, value});
      std::optional<DynamicUnpackResult> collected =
          emitDynamicUnpack(expr, target, call);
      if (!collected)
        return emitNone(expr);
      unsigned kept = std::min<unsigned>(
          plain, static_cast<unsigned>(target.positionalNames.size()));
      CallOperands rebuilt;
      for (unsigned index = 0; index < kept; ++index) {
        rebuilt.positional.push_back(operands.positional[index]);
        rebuilt.positionalUnpacked.push_back(0);
        rebuilt.positionalTypes.push_back(operands.positional[index].type);
      }
      for (const auto &[name, value] : collected->named) {
        mlir::Type literal = types.literal("\"" + name + "\"");
        auto stringOp = py::StrConstantOp::create(
            builder, loc(expr), literal, builder.getStringAttr(name));
        rebuilt.keywordNames.push_back({stringOp.getResult(), literal});
        rebuilt.keywordValues.push_back(value);
        rebuilt.keywordTypes.push_back(CallKeywordType{name, value.type});
      }
      solvePositional.assign(rebuilt.positionalTypes.begin(),
                             rebuilt.positionalTypes.end());
      solveKeywords.assign(rebuilt.keywordTypes.begin(),
                           rebuilt.keywordTypes.end());
      if (collected->vararg) {
        lython::callable::VarargShape<mlir::Type> shape =
            py::callableVarargShape(collected->vararg->type);
        if (shape.valid() && shape.repeated)
          solvePositional.push_back(py::variadicTailType(*shape.repeated));
      }
      if (collected->kwargs)
        solveKeywords.push_back(CallKeywordType{"", collected->kwargs->type});
      operands = std::move(rebuilt);
      collectedVararg = collected->vararg;
      collectedKwargs = collected->kwargs;
    }
  }
  if (!operands.valid) {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, expr.range.start, operands.failureReason});
    return emitNone(expr);
  }
  if (!collectedVararg && !collectedKwargs) {
    solvePositional.assign(operands.positionalTypes.begin(),
                           operands.positionalTypes.end());
    solveKeywords.assign(operands.keywordTypes.begin(),
                         operands.keywordTypes.end());
  }

  KeywordArguments keywords;
  for (auto [index, keyword] : llvm::enumerate(operands.keywordTypes)) {
    if (index < operands.keywordValues.size())
      keywords[keyword.name] = operands.keywordValues[index];
  }
  if (operands.keywordValues.size() != operands.keywordTypes.size()) {
    if (const auto *keywordNodes = ast::nodeList(expr, "keywords")) {
      for (const parser::NodePtr &keyword : *keywordNodes) {
        if (keyword && ast::string(*keyword, "arg"))
          continue;
        if (!keyword)
          continue;
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, keyword->range.start,
            "variadic keyword arguments are not supported for class "
            "instantiation"});
      }
    }
  }
  // ⛔ A STARRED ARGUMENT KEEPS THE CONSTRUCTOR OFF THE INLINE PATH. Inlining
  // `__init__` needs one Value per parameter and a starred operand is one
  // Value for several, so the runtime call below takes it -- where the pack's
  // own expansion (`starredSequenceElements`) does the splitting. It used to
  // be refused outright here instead.
  bool hasUnpackedPositional = llvm::any_of(
      operands.positionalUnpacked, [](char value) { return value != 0; });

  mlir::Type inferredInstanceType =
      types.inferClassInstantiation(instanceType, solvePositional, solveKeywords);
  if (!inferredInstanceType) {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, expr.range.start,
        "class instantiation leaves unbound static type parameters for '" +
            name.str() + "'"});
    return emitNone(expr);
  }
  mlir::Type classType = types.typeObject(inferredInstanceType);
  mlir::Value classObject;
  if (pendingConstructionClass && pendingConstructionClass->type == classType)
    classObject = pendingConstructionClass->value;
  else
    classObject = py::TypeObjectOp::create(builder, loc(expr), classType,
                                           inferredInstanceType)
                      .getResult();
  pendingConstructionClass.reset();
  Value posPack = emitPack(operands.positional, operands.positionalUnpacked);
  Value namePack = emitPack(operands.keywordNames);
  Value valuePack = emitPack(operands.keywordValues);

  auto newOp = py::NewOp::create(
      builder, loc(expr), inferredInstanceType,
      mlir::FlatSymbolRefAttr::get(&context, "__new__"), callableProtocol(),
      classObject, posPack.value, namePack.value, valuePack.value);
  newOp->setAttr("ly.constructor.owner", builder.getStringAttr(name));
  if (std::optional<MethodBinding> newBinding =
          lookupClassMethod(inferredInstanceType, "__new__")) {
    if (newBinding->method)
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, newBinding->method->range.start,
          "source class __new__ bodies are not supported yet; use declared "
          "fields and __init__ for user class construction"});
    newOp->setAttr("ly.constructor.new_kind",
                   builder.getStringAttr(newBinding->kind));
  } else {
    newOp->setAttr("ly.constructor.new_kind", builder.getStringAttr("class"));
  }
  std::optional<MethodBinding> init =
      lookupClassMethod(inferredInstanceType, "__init__");
  if (init && !hasUnpackedPositional) {
    Value receiver{newOp.getInstance(), inferredInstanceType};
    Value descriptorReceiver = emitDescriptorReceiver(expr, receiver, *init);
    emitInlineMethodBody(expr, descriptorReceiver,
                         methodBindingBindsReceiver(*init), *init,
                         operands.positional, keywords, collectedVararg,
                         collectedKwargs);
  } else {
    bool noRuntimeInitArgs = operands.positional.empty() &&
                             operands.keywordValues.empty() &&
                             !hasUnpackedPositional;
    // ⛔ NOT PAST A DECLARED __init__ THAT TAKES ARGUMENTS. A call with none
    // skipped the check below, so `range()` reached the lowering ("runtime
    // call is missing input 0 for builtins.range.__new__"), and a constructor
    // whose inputs all have defaults would have built what CPython refuses.
    bool initTakesNoArguments =
        !types.declaresManifestMethod(inferredInstanceType, "__init__") ||
        static_cast<bool>(types.inferMethodCallWithEvidence(
            inferredInstanceType, "__init__", {}, {}));
    if (!init && noRuntimeInitArgs && initTakesNoArguments) {
      (void)namePack;
      (void)valuePack;
      return {newOp.getInstance(), inferredInstanceType};
    }
    bool noArgExceptionInit =
        noRuntimeInitArgs &&
        py::protocols::Table::get(context).isManifestSubclassOf(
            inferredInstanceType, "builtins.BaseException");
    if (noArgExceptionInit) {
      (void)namePack;
      (void)valuePack;
      return {newOp.getInstance(), inferredInstanceType};
    }
    CallInferenceResult initInference = types.inferMethodCallWithEvidence(
        inferredInstanceType, "__init__", operands.positionalTypes,
        operands.keywordTypes);
    mlir::Type initContract =
        initInference ? callProtocolFor(initInference) : mlir::Type();
    if (!initContract) {
      // Field-record construction: a class without a source or manifest
      // __init__ takes its declared fields positionally, every field
      // optional (ctypes Structure/Union subclasses; plain field records).
      auto order = classFieldOrders.find(name);
      auto fields = classFieldBindings.find(name);
      if (order != classFieldOrders.end() &&
          fields != classFieldBindings.end() && !order->second.empty() &&
          operands.positionalTypes.size() <= order->second.size() &&
          operands.keywordTypes.empty()) {
        llvm::SmallVector<mlir::Type, 8> positional{inferredInstanceType};
        llvm::SmallVector<mlir::StringAttr, 8> positionalNames{
            builder.getStringAttr("self")};
        llvm::SmallVector<mlir::BoolAttr, 8> positionalDefaults{
            builder.getBoolAttr(false)};
        for (const std::string &fieldName : order->second) {
          auto field = fields->second.find(fieldName);
          positional.push_back(field == fields->second.end()
                                   ? types.object()
                                   : field->second);
          positionalNames.push_back(builder.getStringAttr(fieldName));
          positionalDefaults.push_back(builder.getBoolAttr(true));
        }
        llvm::SmallVector<mlir::Type, 1> results{types.none()};
        initContract = py::CallableType::get(
            &context, positional, {}, {}, {}, results, positionalNames, {},
            positionalDefaults, {});
      }
    }
    if (!initContract) {
      if (!requireStaticEvidence(expr, initInference))
        return emitNone(expr);
      initContract = callProtocolFor(initInference);
    }
    if (std::string mismatch = keptArgumentRepresentationMismatch(
            name, inferredInstanceType, initInference,
            operands.positionalTypes);
        !mismatch.empty()) {
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, expr.range.start, std::move(mismatch)});
      return emitNone(expr);
    }
    auto initOp =
        py::InitOp::create(builder, loc(expr), types.none(),
                           mlir::FlatSymbolRefAttr::get(&context, "__init__"),
                           initContract, newOp.getInstance(),
                           posPack.value, namePack.value, valuePack.value);
    initOp->setAttr("ly.constructor.owner", builder.getStringAttr(name));
    initOp->setAttr("ly.constructor.init_kind",
                    builder.getStringAttr(init ? init->kind : "instance"));
  }
  (void)name;
  return {newOp.getInstance(), inferredInstanceType};
}

// ---------------------------------------------------------------------------
// R6 nonlocal cells. A cell is an ordinary user class with one field "v":
// instances are refcounted owned values (QTT-tracked like any object), and
// the shared mutation of the content is the field's interior mutability, so
// the whole existing class machinery (field boxes, aggregate retain/release,
// deallocator hooks, affine verification) applies without new proof rules.
// ---------------------------------------------------------------------------

static constexpr llvm::StringLiteral kCellClassPrefix{"__ly_cell$"};
// ⭐ THE SAME STORAGE, PLUS THE QUESTION CPython ANSWERS WITH A NULL SLOT. A
// name that only some paths bind needs a place to live that outlives the
// region binding it AND a record of whether it was bound, because reading it
// unbound is `UnboundLocalError` and not a value. The cell already provides
// the first; the second is one more field.
static constexpr llvm::StringLiteral kBindingFieldName{"d"};
// ⛔ A DIFFERENT PREFIX, because the LOWERING keys on the cell one. A cell is
// lowered by `lowerCellAttrGet`, which reads the content out of a box slot
// because a cell's content can be replaced through any frame holding it; that
// path requires every field to be box-fronted, and the binding flag is a bool,
// which is an inline word. A maybe-unbound slot needs none of that -- it lives
// in one frame -- so it is an ORDINARY class and takes the ordinary field
// path. The emitter still routes reads and writes through it, because
// `isCellContract` answers for both spellings.
static constexpr llvm::StringLiteral kSlotClassPrefix{"__ly_slot$"};

bool ModuleEmitter::isCellContract(mlir::Type type) {
  auto contract = mlir::dyn_cast_if_present<py::ContractType>(type);
  return contract &&
         (contract.getContractName().starts_with(kCellClassPrefix) ||
          contract.getContractName().starts_with(kSlotClassPrefix));
}

// ⭐ A CARRIED LOCAL REBOUND AFTER A LOOP OR A TRY IS BOUND BY ITS CONTENT.
// The region's results carry the STORAGE, which for a name bound on only some
// paths is its slot, and binding that made the inference read the slot class
// where the program reads the value: `out.append((i, v))` with `v` bound in a
// try was "no signature that accepts tuple[int, __ly_slot$1]", while
// `print((i, v))` on the line above ran.
void ModuleEmitter::bindStoredLocalType(llvm::StringRef name,
                                        mlir::Type storage) {
  if (isCellContract(storage))
    if (mlir::Type content = cellContentType(storage)) {
      types.bindSymbol(name, content);
      return;
    }
  types.bindSymbol(name, storage);
}

mlir::Type ModuleEmitter::cellContentType(mlir::Type cellType) {
  auto contract = mlir::dyn_cast_if_present<py::ContractType>(cellType);
  if (!contract)
    return {};
  auto fields = classFieldBindings.find(contract.getContractName());
  if (fields == classFieldBindings.end())
    return {};
  auto field = fields->second.find("v");
  return field == fields->second.end() ? mlir::Type() : field->second;
}

// The stores `collectClassFields` saw into ANOTHER class's field, applied now
// that every class is declared: a field still `None` from its own constructor
// becomes `T | None` -- the rule `optionalPair` gives a same-class store --
// in every place the declaration recorded it (the field bindings, the
// protocol table, the class op) and in the subclasses that inherited it.
// ⛔ Only a field that is exactly None: any other disagreement is a store this
// compiler still refuses where it is written.
const llvm::StringSet<> &ModuleEmitter::programAttributeStoreNames() const {
  if (programAttributeStoresScanned)
    return programAttributeStores;
  programAttributeStoresScanned = true;
  std::function<void(const parser::Node &)> walk = [&](const parser::Node &node) {
    auto record = [&](const parser::Node *target) {
      if (target && target->kind == "Attribute")
        if (auto attr = ast::string(*target, "attr"))
          programAttributeStores.insert(*attr);
    };
    if (node.kind == "Assign") {
      if (const auto *targets = ast::nodeList(node, "targets"))
        for (const parser::NodePtr &target : *targets)
          record(target.get());
    } else if (node.kind == "AnnAssign" || node.kind == "AugAssign") {
      record(ast::node(node, "target"));
    }
    for (const parser::Field &field : node.fields) {
      if (const auto *child = std::get_if<parser::NodePtr>(&field.value)) {
        if (*child)
          walk(**child);
      } else if (const auto *children =
                     std::get_if<std::vector<parser::NodePtr>>(&field.value)) {
        for (const parser::NodePtr &each : *children)
          if (each)
            walk(*each);
      }
    }
  };
  walk(moduleNode);
  for (const EmitOptions::SourceModule &source : options.sourceModules)
    if (source.moduleNode)
      walk(*source.moduleNode);
  return programAttributeStores;
}

void ModuleEmitter::applyForeignFieldWrites() {
  if (foreignFieldWrites.empty())
    return;
  std::vector<ForeignFieldWrite> writes = std::move(foreignFieldWrites);
  foreignFieldWrites.clear();
  // A field's type, through the class's MRO.
  auto fieldOf = [&](llvm::StringRef cls,
                     llvm::StringRef name) -> mlir::Type {
    llvm::SmallVector<std::string, 4> line{cls.str()};
    if (auto mro = classMros.find(cls); mro != classMros.end())
      line.assign(mro->second.begin(), mro->second.end());
    for (const std::string &each : line)
      if (auto fields = classFieldBindings.find(each);
          fields != classFieldBindings.end())
        if (auto found = fields->second.find(name);
            found != fields->second.end())
          return found->second;
    return {};
  };
  auto ownerClassOf = [&](mlir::Type type) -> std::string {
    auto contract = mlir::dyn_cast_if_present<py::ContractType>(
        mlir::isa_and_nonnull<py::UnionType>(type)
            ? removeNoneFromType(type, types)
            : type);
    if (!contract || !isSourceClassContract(contract))
      return {};
    return contract.getContractName().str();
  };
  for (const ForeignFieldWrite &write : writes) {
    if (!write.type || write.type == types.none() ||
        py::isPyObjectType(write.type) ||
        !mlir::isa<py::ContractType>(write.type))
      continue;
    std::string owner = ownerClassOf(write.root);
    for (const std::string &step : write.path) {
      if (owner.empty())
        break;
      owner = ownerClassOf(fieldOf(owner, step));
    }
    if (owner.empty())
      continue;
    mlir::Type widened = types.join({write.type, types.none()});
    for (auto &entry : classFieldBindings) {
      llvm::StringRef cls = entry.getKey();
      if (cls != owner) {
        auto mro = classMros.find(cls);
        if (mro == classMros.end() || !llvm::is_contained(mro->second, owner))
          continue;
      }
      auto field = entry.getValue().find(write.field);
      if (field == entry.getValue().end() || field->second != types.none())
        continue;
      field->second = widened;
      if (const py::protocols::ProtocolInfo *info =
              py::protocols::Table::get(context).lookup(cls)) {
        py::protocols::ProtocolInfo updated = *info;
        updated.fields[write.field] = widened;
        py::protocols::Table::getMutable(context).registerClass(
            cls, std::move(updated));
      }
      module.walk([&](py::ClassOp classOp) {
        if (classOp.getSymName() != cls)
          return;
        auto names = classOp->getAttrOfType<mlir::ArrayAttr>("field_names");
        if (!names)
          return;
        for (llvm::StringRef attrName : {"field_types", "field_contract_types"}) {
          auto typesAttr = classOp->getAttrOfType<mlir::ArrayAttr>(attrName);
          if (!typesAttr || typesAttr.size() != names.size())
            continue;
          llvm::SmallVector<mlir::Attribute, 8> updated(typesAttr.begin(),
                                                        typesAttr.end());
          for (auto [index, name] : llvm::enumerate(names))
            if (mlir::cast<mlir::StringAttr>(name).getValue() == write.field)
              updated[index] = mlir::TypeAttr::get(widened);
          classOp->setAttr(attrName,
                           mlir::ArrayAttr::get(&context, updated));
        }
      });
    }
  }
}

// A `type[X]` whose X is one of the program's classes that nothing derives
// from: the type alone says which class object the value is.
bool ModuleEmitter::typeValueIsStatic(mlir::Type type) const {
  auto typeType = mlir::dyn_cast_if_present<py::TypeType>(type);
  if (!typeType)
    return false;
  auto instance =
      mlir::dyn_cast_if_present<py::ContractType>(typeType.getInstanceType());
  if (!instance || !instance.getArguments().empty())
    return false;
  llvm::StringRef name = instance.getContractName();
  if (!declaredClassBases.count(name))
    return false;
  for (const auto &entry : declaredClassBases)
    if (llvm::is_contained(entry.getValue(), name))
      return false;
  return true;
}

// A slot whose content has no word: None, or a class object its type names.
// It holds the binding flag alone.
bool ModuleEmitter::zeroWidthSlotContent(mlir::Type content) const {
  return isNoneTypeLike(content) || typeValueIsStatic(content);
}

bool ModuleEmitter::cellTracksBinding(mlir::Type cellType) const {
  auto contract = mlir::dyn_cast_if_present<py::ContractType>(cellType);
  return contract &&
         bindingCellContractNames.contains(contract.getContractName());
}

mlir::Type ModuleEmitter::ensureCellClass(mlir::Type contentType,
                                          const parser::Node &anchor,
                                          bool tracksBinding) {
  llvm::DenseMap<mlir::Type, mlir::Type> &memo =
      tracksBinding ? bindingCellClassContracts : cellClassContracts;
  auto memoized = memo.find(contentType);
  if (memoized != memo.end())
    return memoized->second;

  // ⛔ AN OPTIONAL CONTENT TAKES THE SLOT SPELLING TOO. The cell-specific
  // lowering reads its content out of the box with a rank-1 shape check, which
  // an optional's tag fails -- "nonlocal over `T | None` is not supported yet".
  // The ORDINARY field path has read an optional since the box-fronted field
  // work, so a cell holding one is an ordinary class; nothing that used the
  // cell path before reaches this, because that path refused it.
  auto optionalContent = [&](mlir::Type type) {
    auto unionType = mlir::dyn_cast_if_present<py::UnionType>(type);
    return unionType && unionType.isOptional();
  };
  // A class value is a body word of the instance, the slot an ordinary field
  // of that type has; the boxed cell would write the word where its reader
  // looks for a box.
  bool ordinaryStorage = tracksBinding || optionalContent(contentType) ||
                         mlir::isa_and_nonnull<py::TypeType>(contentType);
  std::string cellName =
      (llvm::Twine(ordinaryStorage ? kSlotClassPrefix : kCellClassPrefix) +
       llvm::Twine(++cellClassCounter))
          .str();
  mlir::Type contract = types.contract(cellName);

  // ⭐ A None CONTENT HAS NO WORD, so its slot is the binding flag alone; the
  // content type is still recorded for `cellContentType`, and the load and
  // store below never touch a `v` that a None slot does not have.
  const bool zeroWidth = tracksBinding && zeroWidthSlotContent(contentType);
  classFieldBindings[cellName]["v"] = contentType;
  llvm::SmallVector<std::string, 8> fieldNames;
  llvm::SmallVector<mlir::Type, 2> fieldTypes;
  if (!zeroWidth) {
    fieldNames.push_back("v");
    fieldTypes.push_back(contentType);
  }
  // ⛔ THE ERASED STORAGE CONTRACT IS THE CELL'S, not every slot's. A cell is
  // read back through `lowerCellAttrGet`, which rebuilds the content from the
  // box and knows the declared type; an ORDINARY field read compares the
  // storage contract against the result and reported "attribute evidence
  // 'builtins.object' is not assignable to result 'builtins.int'".
  llvm::SmallVector<mlir::Type, 8> fieldStorage;
  if (!zeroWidth)
    fieldStorage.push_back(ordinaryStorage ? contentType
                                           : types.contract("builtins.object"));
  if (tracksBinding) {
    fieldNames.push_back(kBindingFieldName.str());
    fieldTypes.push_back(types.contract("builtins.bool"));
    fieldStorage.push_back(types.contract("builtins.bool"));
  }

  classBaseNames[cellName] = {};
  classMros[cellName] = {cellName, "builtins.object"};
  classOwnFieldOrders[cellName] = fieldNames;
  classFieldOrders[cellName] = fieldNames;
  for (auto [index, field] : llvm::enumerate(fieldNames))
    classFieldBindings[cellName][field] = fieldTypes[index];

  py::protocols::ProtocolInfo protocolInfo;
  for (auto [index, field] : llvm::enumerate(fieldNames))
    protocolInfo.fields[field] = fieldTypes[index];
  py::protocols::Table::getMutable(context).registerClass(
      cellName, std::move(protocolInfo));

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(module.getBody());
  mlir::OperationState state(loc(anchor), py::ClassOp::getOperationName());
  state.addAttribute(mlir::SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(cellName));
  state.addAttribute("base_names",
                     stringArray(builder, llvm::ArrayRef<std::string>{}));
  state.addAttribute("field_names", stringArray(builder, fieldNames));
  state.addAttribute("field_types", typeArray(builder, fieldTypes));
  // The STORAGE contract is the erased object: the existing box-fronted
  // field rule then gives the cell one stable box16 slot (allocation,
  // deallocator and init paths all follow that rule), which is what lets a
  // closure's store reach every other frame holding the cell.
  state.addAttribute("field_contract_types", typeArray(builder, fieldStorage));
  state.addAttribute("method_names",
                     stringArray(builder, llvm::ArrayRef<std::string>{}));
  state.addAttribute("method_contracts",
                     typeArray(builder, llvm::ArrayRef<mlir::Type>{}));
  state.addAttribute("method_kinds",
                     stringArray(builder, llvm::ArrayRef<std::string>{}));
  state.addAttribute("method_symbols",
                     stringArray(builder, llvm::ArrayRef<std::string>{}));
  state.addAttribute("mro_names",
                     stringArray(builder, llvm::ArrayRef<std::string>{
                                              cellName, "builtins.object"}));
  state.addRegion();
  mlir::Operation *op = builder.create(state);
  op->getRegion(0).push_back(new mlir::Block);

  memo[contentType] = contract;
  if (tracksBinding)
    bindingCellContractNames.insert(cellName);
  return contract;
}

Value ModuleEmitter::emitCellAlloc(const parser::Node &anchor, Value initial,
                                   bool tracksBinding) {
  mlir::Type content = types.widenLiteral(initial.type);
  mlir::Type cellType = ensureCellClass(content, anchor, tracksBinding);
  const bool zeroWidth = tracksBinding && zeroWidthSlotContent(content);
  mlir::Type classType = types.typeObject(cellType);
  auto classObject =
      py::TypeObjectOp::create(builder, loc(anchor), classType, cellType);
  llvm::SmallVector<Value, 2> initialFields;
  if (!zeroWidth)
    initialFields.push_back(coerceValue(initial, content, anchor));
  if (tracksBinding) {
    auto falseOp = py::BoolConstantOp::create(
        builder, loc(anchor), types.literal("False"), builder.getBoolAttr(false));
    initialFields.push_back(coerceValue({falseOp.getResult(),
                                         types.literal("False")},
                                        types.contract("builtins.bool"),
                                        anchor));
  }
  Value posPack = emitPack(initialFields);
  Value namePack = emitPack({});
  Value valuePack = emitPack({});
  auto newOp = py::NewOp::create(
      builder, loc(anchor), cellType,
      mlir::FlatSymbolRefAttr::get(&context, "__new__"), callableProtocol(),
      classObject.getResult(), posPack.value, namePack.value, valuePack.value);
  auto contract = mlir::cast<py::ContractType>(cellType);
  newOp->setAttr("ly.constructor.owner",
                 builder.getStringAttr(contract.getContractName()));
  newOp->setAttr("ly.constructor.new_kind", builder.getStringAttr("class"));
  // Field-record initialization (the no-__init__ construction rule): the
  // initial content boxes into the cell's slot during lowerInit.
  llvm::SmallVector<mlir::Type, 3> positional{cellType};
  llvm::SmallVector<mlir::StringAttr, 3> positionalNames{
      builder.getStringAttr("self")};
  llvm::SmallVector<mlir::BoolAttr, 3> positionalDefaults{
      builder.getBoolAttr(false)};
  if (!zeroWidth) {
    positional.push_back(content);
    positionalNames.push_back(builder.getStringAttr("v"));
    positionalDefaults.push_back(builder.getBoolAttr(true));
  }
  if (tracksBinding) {
    positional.push_back(types.contract("builtins.bool"));
    positionalNames.push_back(builder.getStringAttr(kBindingFieldName));
    positionalDefaults.push_back(builder.getBoolAttr(true));
  }
  llvm::SmallVector<mlir::Type, 1> results{types.none()};
  mlir::Type initContract = py::CallableType::get(
      &context, positional, {}, {}, {}, results, positionalNames, {},
      positionalDefaults, {});
  auto initOp =
      py::InitOp::create(builder, loc(anchor), types.none(),
                         mlir::FlatSymbolRefAttr::get(&context, "__init__"),
                         initContract, newOp.getInstance(), posPack.value,
                         namePack.value, valuePack.value);
  initOp->setAttr("ly.constructor.owner",
                  builder.getStringAttr(contract.getContractName()));
  initOp->setAttr("ly.constructor.init_kind", builder.getStringAttr("instance"));
  return {newOp.getInstance(), cellType};
}

// ⭐ THE GUARD IS AT THE READ, which is where CPython puts it: a name bound on
// one path and read on another is an error only if the read runs. Raising at
// the join instead would refuse `if c: x = 1` in a program that never reads
// `x`, which CPython accepts.
void ModuleEmitter::emitUnboundLocalGuard(const parser::Node &anchor,
                                          const Value &cell,
                                          llvm::StringRef name) {
  mlir::Type flagType = types.contract("builtins.bool");
  auto flag = py::AttrGetOp::create(builder, loc(anchor), flagType, cell.value,
                                    kBindingFieldName);
  flag->setAttr("ly.attr.kind", builder.getStringAttr("field"));
  auto contract = mlir::cast<py::ContractType>(cell.type);
  flag->setAttr("ly.attr.owner",
                builder.getStringAttr(contract.getContractName()));
  mlir::Value bound = emitBoolValue({flag.getResult(), flagType}, anchor);

  mlir::Block *entry = builder.getInsertionBlock();
  mlir::Region *region = entry->getParent();
  mlir::Block *continuation = entry->splitBlock(builder.getInsertionPoint());
  mlir::Block *unboundBlock =
      builder.createBlock(region, continuation->getIterator());
  builder.setInsertionPointToEnd(entry);
  mlir::cf::CondBranchOp::create(builder, loc(anchor), bound, continuation,
                                 unboundBlock);

  builder.setInsertionPointToStart(unboundBlock);
  // Module scope says NameError and a function body says UnboundLocalError,
  // which is what CPython reports for the same source in the two places.
  bool atModuleScope = currentFunctionPrefix.empty();
  llvm::StringRef className =
      atModuleScope ? "NameError" : "UnboundLocalError";
  std::string message =
      atModuleScope
          ? ("name '" + name + "' is not defined").str()
          : ("cannot access local variable '" + name +
             "' where it is not associated with a value")
                .str();
  std::vector<parser::NodePtr> arguments;
  arguments.push_back(synth::strConstant(message, anchor.range));
  Value raised = emitExpr(
      synth::call(synth::name(className, anchor.range), std::move(arguments),
                  anchor.range)
          .get());
  if (raised.value)
    py::RaiseOp::create(builder, loc(anchor), raised.value, mlir::Value{},
                        false);
  builder.setInsertionPointToEnd(continuation);
}

Value ModuleEmitter::emitCellLoad(const parser::Node &anchor,
                                  const Value &cell) {
  mlir::Type content = cellContentType(cell.type);
  if (!content) {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, anchor.range.start,
        "internal: nonlocal cell has no registered content type"});
    return emitNone(anchor);
  }
  if (cellTracksBinding(cell.type) && isNoneTypeLike(content))
    return emitNone(anchor);
  if (cellTracksBinding(cell.type) && typeValueIsStatic(content)) {
    auto typeType = mlir::cast<py::TypeType>(content);
    auto op = py::TypeObjectOp::create(builder, loc(anchor), content,
                                       typeType.getInstanceType());
    return {op.getResult(), content};
  }
  auto op =
      py::AttrGetOp::create(builder, loc(anchor), content, cell.value, "v");
  op->setAttr("ly.attr.kind", builder.getStringAttr("field"));
  auto contract = mlir::cast<py::ContractType>(cell.type);
  op->setAttr("ly.attr.owner",
              builder.getStringAttr(contract.getContractName()));
  return {op.getResult(), content};
}

void ModuleEmitter::emitCellStore(const parser::Node &anchor, const Value &cell,
                                  Value value) {
  mlir::Type content = cellContentType(cell.type);
  if (!content) {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, anchor.range.start,
        "internal: nonlocal cell has no registered content type"});
    return;
  }
  Value coerced = coerceValue(value, content, anchor);
  if (coerced.type != content) {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, anchor.range.start,
        "nonlocal assignment does not match the variable's established "
        "type"});
    return;
  }
  auto contract = mlir::cast<py::ContractType>(cell.type);
  mlir::Operation *op = nullptr;
  if (!(cellTracksBinding(cell.type) && zeroWidthSlotContent(content))) {
    op = py::AttrSetOp::create(builder, loc(anchor), cell.value, "v",
                               coerced.value);
    op->setAttr("ly.attr.kind", builder.getStringAttr("field"));
  }
  if (cellTracksBinding(cell.type)) {
    auto trueOp = py::BoolConstantOp::create(
        builder, loc(anchor), types.literal("True"), builder.getBoolAttr(true));
    Value flag = coerceValue({trueOp.getResult(), types.literal("True")},
                             types.contract("builtins.bool"), anchor);
    auto mark = py::AttrSetOp::create(builder, loc(anchor), cell.value,
                                      kBindingFieldName, flag.value);
    mark->setAttr("ly.attr.kind", builder.getStringAttr("field"));
    mark->setAttr("ly.attr.owner",
                  builder.getStringAttr(contract.getContractName()));
  }
  if (op)
    op->setAttr("ly.attr.owner",
                builder.getStringAttr(contract.getContractName()));
}

} // namespace lython::emitter
