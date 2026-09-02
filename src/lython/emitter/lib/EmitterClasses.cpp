#include "AstSynth.h"
#include "EmitterCore.h"
#include "EmitterPyOps.h"
#include "EmitterSupport.h"

#include "AstAccess.h"
#include "ClosureAnalysis.h"
#include "Contracts.h"
#include "C3Linearization.h"
#include "ExceptionTaxonomy.h"
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
  while (entry && entry->baseClassId != py::exceptions::kRootClassId) {
    entry = py::exceptions::findByClassId(entry->baseClassId);
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
      // ⛔ Only a bare NAME. A decorator FACTORY (`@deco(arg)`) is
      // `f = deco(arg)(f)`, one more call whose intermediate value is a
      // function the compiler would have to see through; it keeps the refusal
      // rather than getting a silent partial answer.
      recognized = leaf == "native" || isTypingMarker(leaf) ||
                   (decorator->kind == "Name" && moduleFunctionNames.count(leaf));
      break;
    case DecoratorRole::Class:
      recognized = leaf == "dataclass" || isTypingMarker(leaf);
      break;
    }
    if (recognized)
      continue;
    if (spelling.empty())
      spelling = "<expression>";
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, decorator->range.start,
        "decorator '" + spelling + "' is not supported (unrecognized "
        "decorators are rejected instead of silently ignored)"});
  }
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
  if (!superContexts.empty())
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
  return redeclared;
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

bool ModuleEmitter::subclassRedeclares(
    const llvm::StringMap<llvm::StringSet<>> &declarations,
    llvm::StringRef receiverClass, llvm::StringRef name) const {
  // Ancestors as WRITTEN, from the pre-pass rather than from the MRO map the
  // class emission fills: the answer must not depend on where in the file the
  // question is asked (`collectTopLevelBindings`).
  auto ancestors = [&](llvm::StringRef cls, llvm::StringSet<> &out) {
    llvm::SmallVector<llvm::StringRef, 8> worklist{cls};
    while (!worklist.empty()) {
      llvm::StringRef current = worklist.pop_back_val();
      auto bases = declaredClassBases.find(current);
      if (bases == declaredClassBases.end())
        continue;
      for (const std::string &base : bases->second)
        if (out.insert(base).second)
          worklist.push_back(base);
    }
  };

  llvm::StringSet<> receiverAncestors;
  ancestors(receiverClass, receiverAncestors);

  for (const auto &entry : declaredClassBases) {
    llvm::StringRef candidate = entry.getKey();
    if (candidate == receiverClass)
      continue;
    llvm::StringSet<> candidateAncestors;
    ancestors(candidate, candidateAncestors);
    if (!candidateAncestors.contains(receiverClass))
      continue;
    // ⭐ A declaration the subclass reaches WITHOUT going through the receiver
    // class is an override, whether the subclass wrote it or inherited it from
    // a mixin. Testing only what the subclass declares itself missed
    // `class B(M, A): pass` -- B declares nothing, resolves `v` to M's, and
    // the base's body was inlined for it.
    //
    // The receiver's own ancestors are excluded because a declaration there is
    // what the receiver already resolves to, not a competitor. Conservative in
    // a diamond, which is the direction that refuses rather than the direction
    // that mis-executes.
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
    if (slots == classAttrSlots.end())
      continue;
    auto slot = slots->second.find(attrName);
    if (slot != slots->second.end())
      return std::make_pair(slots->first(), slot->second);
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
// ⛔ NOT when the class declares its own, and that shape is still missing its
// parent's hook. CPython runs the PARENT's for the new class, never the
// class's own on itself, and the spelling above would run the class's own.
// Reaching the parent's with `cls` bound to the NEW class is what `super()`
// does inside a method body and there is no expression for it out here:
// `Base.__init_subclass__()` binds `cls` to Base, which is a wrong answer
// rather than a missing one. Measured in
// tests/probe/wb_init_subclass_through_a_middle_class.py.
void ModuleEmitter::emitInitSubclassHook(const parser::Node &classDef) {
  auto name = ast::string(classDef, "name");
  if (!name)
    return;
  std::optional<MethodBinding> hook =
      lookupClassMethod(types.contract(*name), "__init_subclass__");
  if (!hook || !hook->method || hook->definingClass == *name)
    return;
  // The hook takes no arguments beyond `cls`; anything else is a shape this
  // does not model, and calling it would pass the wrong count.
  if (hook->bodySignature.positionalNames.size() != 1)
    return;
  parser::NodePtr call = synth::call(
      synth::attribute(synth::name(*name, classDef.range),
                       "__init_subclass__", classDef.range),
      std::vector<parser::NodePtr>{}, classDef.range);
  emitStatement(*synth::exprStmt(std::move(call), classDef.range));
}

void ModuleEmitter::emitClassAttrInitializers(const parser::Node &classDef) {
  auto name = ast::string(classDef, "name");
  if (!name)
    return;
  auto slots = classAttrSlots.find(*name);
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
    if (mlir::Type declared = types.widenLiteral(slot->second),
        supplied = types.widenLiteral(initial.type);
        declared && supplied && declared != supplied &&
        isNumericPrimitiveContract(declared) &&
        isNumericPrimitiveContract(supplied)) {
      auto spell = [&](mlir::Type numeric) -> llvm::StringRef {
        if (numeric == types.boolType())
          return "bool";
        return numeric == types.intType() ? "int" : "float";
      };
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, statement->range.start,
          "class attribute '" + std::string(*name) + "." + attrName.str() +
              "' holds " + spell(declared).str() +
              " and this initializer gives it " + spell(supplied).str() +
              "; the attribute's cell has one runtime representation and these "
              "two do not share one, so write the value in the declared type"});
      continue;
    }
    Value coerced = coerceValue(initial, slot->second, *statement);
    std::string cellName = (llvm::Twine(*name) + "." + attrName).str();
    py::GlobalSetOp::create(builder, loc(*statement),
                            builder.getStringAttr(cellName), coerced.value);
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
  if (!genericClassEmissionReady) {
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
                                      llvm::StringRef symbolName) {
  auto name = ast::string(classDef, "name");
  if (!name)
    return;
  std::string classSymbol =
      symbolName.empty() ? std::string(*name) : symbolName.str();
  llvm::StringRef contractName(classSymbol);
  checkDecorators(classDef, DecoratorRole::Class);

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

  llvm::SmallVector<std::string, 8> fieldNames;
  llvm::SmallVector<mlir::Type, 8> fieldTypes;
  collectClassFields(classDef, fieldNames, fieldTypes,
                     /*includeAnnAssignDefaults=*/isDataclass);

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
  llvm::SmallVector<std::string, 8> staticAttrNames;
  llvm::SmallVector<mlir::Attribute, 8> staticAttrValues;
  llvm::SmallVector<mlir::Type, 8> staticAttrTypes;
  collectStaticClassAssignments(classDef, staticAttrNames, staticAttrValues,
                                &staticAttrTypes);
  // Mutable class attributes: attributes of main-module classes whose
  // widened type has module-global cell storage become slot-backed (reads
  // and writes go through the cells; the initializer expression is no
  // longer restricted to constants). Container-typed attributes stay on the
  // constant channel: their storage cells would go stale against
  // reallocation, the same reason collectModuleGlobals excludes them.
  if (symbolName.empty()) {
    llvm::StringMap<mlir::Type> &slots = classAttrSlots[contractName];
    slots.clear();
    for (auto [attrName, attrType] :
         llvm::zip_equal(staticAttrNames, staticAttrTypes)) {
      mlir::Type widened = types.widenLiteral(attrType);
      bool storable =
          widened == types.intType() || widened == types.strType() ||
          widened == types.floatType() || widened == types.boolType();
      if (!storable) {
        if (auto attrContract =
                mlir::dyn_cast_if_present<py::ContractType>(widened)) {
          llvm::StringRef attrContractName = attrContract.getContractName();
          // ⭐ A CONTAINER IS SLOT-BACKED TOO, and the reason it was not is a
          // paragraph that stopped being true. The note above said container
          // cells "would go stale against reallocation, the same reason
          // collectModuleGlobals excludes them" -- and that exclusion is gone:
          // `builtins.mlir`'s growth writes THROUGH the handle, so the cell
          // holds what stays valid. Without this a container class attribute
          // could not even be READ:
          //
          //     class R:
          //         items: list[str] = []
          //     print(R.items)
          //     # unsupported static class attribute expression for 'items'
          //
          // because the constant channel re-materializes the value per read and
          // has no arm for a container -- and could not have one, since every
          // read of a mutable attribute must be the SAME object.
          //
          // ⛔ EXCEPT a container whose ELEMENT type is a union, which
          // `collectModuleGlobals` still excludes for its own measured reason: a
          // cell hands back the handle, and a union-typed element read needs the
          // literal's per-element evidence.
          //
          // ⛔ And EXCEPT a `_dunder_` name. `ctypes.Structure._fields_` is a
          // list the COMPILER consumes, not a runtime value; slotting it emits a
          // module-level store, and a runtime-internal lib module may not run
          // module-level code (`stackguard_support.py` stopped building).
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
                      !attrContractName.contains('.'));
        }
      }
      if (storable)
        slots[attrName] = widened;
    }
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
  llvm::StringSet<> propertyNames = classPropertyNames(classDef);
  if (const auto *body = ast::nodeList(classDef, "body")) {
    for (const parser::NodePtr &statement : *body) {
      if (!statement || (statement->kind != "FunctionDef" &&
                         statement->kind != "AsyncFunctionDef"))
        continue;
      auto methodName = ast::string(*statement, "name");
      if (!methodName)
        continue;
      // ⭐ A FINALIZER THAT NEVER RUNS IS THE SILENT KIND. Nothing calls
      // `__del__` here -- not at scope exit, not when a container drops its
      // last reference, not at module teardown -- and the program simply
      // printed one line fewer than CPython's. Running it means the
      // deallocator calling back into a source method, which is a mechanism
      // this compiler does not have; until it does, the class is refused at
      // the earliest boundary that can see the declaration.
      if (*methodName == "__del__") {
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, statement->range.start,
            "__del__ is not supported: object finalizers are never run, and a "
            "class that defines one would be finalized silently differently "
            "from CPython"});
        continue;
      }
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
      if ((bodySig.isGeneratorFunction || bodySig.isAsyncGeneratorFunction) &&
          bodyMentionsAttribute(*statement, *methodName)) {
        // ⭐ THE ANNOTATION IS WHAT THE FIRST PASS PUBLISHES. A generator's
        // public result is the INFERRED generator type, and on the first pass
        // that inference is exactly what the missing entry spoiled -- so
        // publishing it would hand the second pass the same `object` back.
        // With `-> Iterator[int]` written down the answer is already there.
        FunctionSignature published = bodySig;
        if (const parser::Node *returns = ast::node(*statement, "returns"))
          if (mlir::Type annotated = types.annotationType(returns)) {
            published.inferredGeneratorType = annotated;
            types.refreshCallable(published);
          }
        addProtocolMethod(progressiveInfo, *methodName,
                          published.publicCallable);
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
      addProtocolMethod(progressiveInfo, *methodName, bodySig.publicCallable);
      publishProgress();

      std::string symbolName =
          sourceMethodSymbolName(contractName, *methodName, *statement);
      methodSymbols.push_back(symbolName);
      // Exception-backed classes get no standalone method symbols: their
      // bodies inline at call sites, and a standalone copy would transfer
      // the borrowed receiver's header through the runtime exception
      // __init__ (an ownership violation the verifier rejects).
      if (kind != "class" && kind != "classmethod" &&
          !isExceptionBackedClass(contractName)) {
        pendingBodies.push_back(statement.get());
        pendingBodySigs.push_back(bodySig);
        pendingBodySymbols.push_back(symbolName);
        pendingBodyKinds.push_back(kind);
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
  if (isDataclass) {
    // Synthesize __init__/__repr__/__eq__ from the MRO-merged field list
    // (CPython composes base dataclass fields the same way); an explicit
    // user definition wins, as in dataclasses._set_new_attribute.
    llvm::ArrayRef<std::string> order = classFieldOrders[contractName];
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
      std::string className =
          py::contracts::displayClassNameForContract(contractName);
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

  for (auto [statement, bodySig, symbolName, kind] :
       llvm::zip_equal(pendingBodies, pendingBodySigs, pendingBodySymbols,
                       pendingBodyKinds)) {
    bool instanceBody = kind == "instance" && !bodySig.positionalNames.empty();
    if (instanceBody)
      superContexts.push_back(SuperContext{std::string(contractName),
                                           bodySig.positionalNames.front()});
    // The frozen-field guard is off inside this class's own __init__, which is
    // where the fields are filled -- CPython goes around its own block there
    // with object.__setattr__. The body is emitted BOTH as a symbol (here) and
    // inlined at each construction, so the exemption is set in both places.
    std::string frozenOwner{contractName};
    std::optional<llvm::SaveAndRestore<const std::string *>> frozenInit;
    if (ast::string(*statement, "name").value_or("") == "__init__" &&
        frozenDataclassContracts.count(frozenOwner))
      frozenInit.emplace(frozenInitContract, &frozenOwner);
    emitCallableFunction(*statement, symbolName, bodySig, {},
                         /*isLambda=*/false);
    if (instanceBody)
      superContexts.pop_back();
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
  // ⭐ THIS CLASS IS THE PROGRAM'S, and the class-id assignment has to know.
  // A manifest `py.class` carries its contract in its SYMBOL's leaf
  // (`py.class @Task` for `_asyncio.Task`), so the lowering finds a manifest
  // id by trying `builtins.`/`types.`/`_asyncio.`/`asyncio.`/`contextlib.` in
  // front of a bare name -- and `class Task` in a program then took asyncio's
  // id 15 instead of a fresh source id. Its instances were tagged as asyncio
  // Tasks and the program SEGFAULTED. Nothing else distinguishes the two: a
  // manifest class op carries no contract attribute of its own.
  state.addAttribute("ly.class.source", builder.getUnitAttr());
  state.addAttribute("base_names", stringArray(builder, bases));
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
    values.push_back(sourceExprAttr(attrBuilder, value));
    if (typesOut)
      typesOut->push_back(valueType);
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

void ModuleEmitter::collectClassFields(
    const parser::Node &classDef,
    llvm::SmallVectorImpl<std::string> &fieldNames,
    llvm::SmallVectorImpl<mlir::Type> &fieldTypes,
    bool includeAnnAssignDefaults) {
  // Field names whose only assignment so far was an EMPTY container literal,
  // which has no element type of its own: the next assignment replaces it.
  llvm::StringSet<> provisionalFields;
  auto setField = [&](llvm::StringRef name, mlir::Type type,
                      bool overwriteExisting, const parser::Node &anchor,
                      bool provisional = false) {
    if (name.empty())
      return;
    if (!type) {
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
    for (auto [index, existing] : llvm::enumerate(fieldNames)) {
      if (existing != name)
        continue;
      if (overwriteExisting ||
          refinesErasedArguments(fieldTypes[index], storedType) ||
          (provisionalFields.contains(name) && !provisional)) {
        fieldTypes[index] = storedType;
        if (!provisional)
          provisionalFields.erase(name);
      }
      return;
    }
    fieldNames.push_back(name.str());
    fieldTypes.push_back(storedType);
    if (provisional)
      provisionalFields.insert(name);
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
  };

  llvm::StringSet<> propertyNames = classPropertyNames(classDef);
  auto collectTarget = [&](const parser::Node &target, mlir::Type type,
                           bool provisional = false) {
    if (target.kind != "Attribute")
      return;
    const parser::Node *object = ast::node(target, "value");
    if (!object || !ast::isName(*object, "self"))
      return;
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

    for (const parser::NodePtr &method : *body) {
      if (!method || ast::nameSpelling(*method) != "__init__")
        continue;
      // Names in scope for a field's declared type: the parameters, plus
      // locals bound earlier in the body. Why not a private name->type map
      // consulted only where the initializer is a bare `Name`: an initializer
      // that is an EXPRESSION over those names (`self.lineno = pos + 1`) is
      // handed to inferExpr with the names unbound, so it types as object --
      // and an object-typed field is not a diagnostic, it is a field whose
      // reads silently fail to carry wherever a known contract is required.
      // Binding the names into a real scope makes one path serve both.
      llvm::StringMap<mlir::Type> initArgTypes;
      collectInitArgTypes(*method, initArgTypes);
      TypeSystem::Scope initScope = types.pushScope();
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
      auto valueTypeOf = [&](const parser::Node *value) -> mlir::Type {
        if (value && value->kind == "Name" &&
            untypedLocals.contains(ast::nameSpelling(*value)))
          return {};
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
              for (llvm::StringRef region :
                   {"body", "orelse", "finalbody", "handlers"})
                if (const auto *nested = ast::nodeList(*stmt, region))
                  walkInitBody(nested);
            }
          };
      walkInitBody(ast::nodeList(*method, "body"));
    }
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
  llvm::StringMap<Value> keywords;
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
  if (!actual || !expected ||
      actual.getContractName() != expected.getContractName() ||
      actual.getArguments().empty() ||
      actual.getArguments().size() != expected.getArguments().size())
    return value;
  if (!llvm::all_of(actual.getArguments(), [&](mlir::Type argument) {
        return types.widenLiteral(argument) == types.object();
      }))
    return value;
  return coerceValue(value, declared, anchor);
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
  if (const auto *args = ast::nodeList(expr, "args")) {
    for (const parser::NodePtr &arg : *args) {
      if (arg && arg->kind == "Starred") {
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, arg->range.start,
            "starred arguments are not supported for inlined class methods"});
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

  llvm::StringMap<Value> keywords;
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
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, keyword->range.start,
          "variadic keyword arguments are not supported for inlined class "
          "methods"});
    }
  }

  Value descriptorReceiver = emitDescriptorReceiver(expr, receiver, method);
  bool bindReceiver = methodBindingBindsReceiver(method);
  if (method.kind == "instance" && mlir::isa<py::TypeType>(receiver.type))
    bindReceiver = false;
  return emitInlineMethodBody(expr, descriptorReceiver, bindReceiver, method,
                              positional, keywords);
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
    if (!decorator || decorator->kind != "Name")
      continue;
    llvm::StringRef spelling = ast::nameSpelling(*decorator);
    if (!moduleFunctionNames.count(spelling))
      continue;
    std::vector<parser::NodePtr> arguments;
    arguments.push_back(synth::name(*name, statement.range));
    parser::NodePtr applied = synth::assign(
        synth::name(*name, statement.range),
        synth::call(synth::name(spelling, statement.range),
                    std::move(arguments), statement.range),
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

Value ModuleEmitter::emitInlineMethodBody(
    const parser::Node &anchor, Value receiver, bool bindDescriptorReceiver,
    const MethodBinding &method, llvm::ArrayRef<Value> positional,
    const llvm::StringMap<Value> &keywords) {
  if (!method.method)
    return emitNone(anchor);
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
    Value callee = emitBindingRef(anchor, method.symbolName, recursive);
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
  const FunctionSignature &sig =
      method.bodySignature.callable ? method.bodySignature : method.signature;
  const auto *body = ast::nodeList(*method.method, "body");
  mlir::Type resultType = sig.resultType ? sig.resultType : types.none();

  ScopedEmitterScope scope(values, types);
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
  auto crossModuleCleanup = llvm::make_scope_exit([&] {
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
  }
  // A generic class's method body still spells its type parameters (`item:
  // T`), and this inlining happens far from the specialization's emission
  // scope, so the solved arguments are re-established from the defining
  // class's contract name. Inside the cross-module isolation above, so an
  // imported generic keeps them too.
  bindClassTypeArguments(method.definingClass);
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
  auto checkArgument = [&](llvm::StringRef name, mlir::Type declared,
                           Value argument) {
    if (methodIsSynthesized || !declared || !argument.type)
      return;
    mlir::Type expected = types.widenLiteral(declared);
    mlir::Type actual = types.widenLiteral(argument.type);
    if (!expected || !actual || unboundStaticParameterCount(expected) != 0 ||
        unboundStaticParameterCount(actual) != 0)
      return;
    if (isAssignableWithStaticEvidence(actual, expected, module))
      return;
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
    if (auto expectedContract = mlir::dyn_cast<py::ContractType>(expected))
      if (auto actualContract = mlir::dyn_cast<py::ContractType>(actual))
        if (!expectedContract.getContractName().contains('.') &&
            !actualContract.getContractName().contains('.') &&
            llvm::is_contained(classMro(actualContract.getContractName()),
                               expectedContract.getContractName()))
          return;
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
          return;
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
  if (sig.varargName) {
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
    if (sig.varargType && packed.type != sig.varargType)
      packed = coerceValue(packed, sig.varargType, anchor);
    bind(*sig.varargName, packed);
  }

  llvm::SmallVector<std::string, 4> variadicKeywordNames;
  llvm::SmallVector<Value, 4> variadicKeywordValues;
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
      variadicKeywordNames.push_back(name.str());
      variadicKeywordValues.push_back(value);
      return;
    }
    diagnostics.push_back(
        parser::Diagnostic{parser::Severity::Error, anchor.range.start,
                           "unexpected keyword argument '" + name.str() +
                               "' for inlined class method"});
  };
  for (auto &entry : keywords)
    bindKeyword(entry.getKey(), entry.getValue());
  if (sig.kwargName) {
    // The dict the callee would have received: string keys, and values named
    // through `LyValueRef` because they are already emitted. An EMPTY one takes
    // the declared type from the expectation, the same way the empty vararg
    // tuple above does.
    std::size_t refStart = pendingValueRefs.size();
    auto releaseRefs = llvm::make_scope_exit(
        [&] { pendingValueRefs.resize(refStart); });
    std::vector<parser::NodePtr> keyNodes;
    std::vector<parser::NodePtr> valueNodes;
    for (auto [index, name] : llvm::enumerate(variadicKeywordNames)) {
      keyNodes.push_back(synth::strConstant(name, anchor.range));
      parser::NodePtr ref = parser::makeNode("LyValueRef", anchor.range);
      parser::addField(*ref, "slot",
                       static_cast<std::int64_t>(pendingValueRefs.size()));
      pendingValueRefs.push_back(variadicKeywordValues[index]);
      valueNodes.push_back(std::move(ref));
    }
    parser::NodePtr dictNode = parser::makeNode("Dict", anchor.range);
    parser::addField(*dictNode, "keys", std::move(keyNodes));
    parser::addField(*dictNode, "values", std::move(valueNodes));
    Value packedKeywords = sig.kwargType
                               ? emitExprExpected(dictNode.get(), sig.kwargType)
                               : emitExpr(dictNode.get());
    if (sig.kwargType && packedKeywords.type != sig.kwargType)
      packedKeywords = coerceValue(packedKeywords, sig.kwargType, anchor);
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
  auto defaultCellFor = [&](unsigned slot) -> const std::string * {
    auto found = methodDefaultCells.find(method.method);
    if (found == methodDefaultCells.end())
      return nullptr;
    for (const auto &entry : found->second)
      if (entry.first == slot)
        return &entry.second;
    return nullptr;
  };
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
    if (const std::string *cellName = defaultCellFor(slot)) {
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
  for (auto [index, name] : llvm::enumerate(sig.positionalNames)) {
    if (bound.contains(name))
      continue;
    if (const parser::Node *defaultNode =
            positionalDefault(static_cast<unsigned>(index))) {
      Value defaultValue =
          emitDefaultFor(static_cast<unsigned>(index), defaultNode);
      bind(name, coerceValue(defaultValue, sig.positionalTypes[index], anchor));
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
      bind(name, coerceValue(defaultValue, sig.kwOnlyTypes[index], anchor));
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
  inlineReturnContexts.push_back(InlineReturnContext{continuation, resultType});
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
  methodsBeingInlined.push_back(method.method);
  // The frame this body would have had if it were a function: everything it
  // emits is located inside `bad`, at a call written in whatever function the
  // inliner is currently writing into.
  InlineFrame frame;
  frame.calleeName = ast::string(*method.method, "name").value_or("<lambda>");
  if (!inlineFrames.empty())
    frame.callerName = inlineFrames.back().calleeName;
  frame.line = anchor.range.start.line;
  frame.column = anchor.range.start.column;
  frame.endLine = anchor.range.end.line;
  frame.endColumn = anchor.range.end.column;
  frame.noAnchor = anchorlessCall == &anchor;
  inlineFrames.push_back(std::move(frame));
  emitStatements(body);
  inlineFrames.pop_back();
  methodsBeingInlined.pop_back();
  currentReturnType = enclosingReturnType;
  if (pushedSuperContext)
    superContexts.pop_back();
  inlineReturnContexts.pop_back();
  if (!insertionBlockTerminated(builder)) {
    if (resultType != types.none()) {
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, method.method->range.start,
          "inlined class method can fall through without returning a value"});
    }
    Value none = emitNone(anchor);
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
  // `object()` is refused at the earliest static boundary rather than lowered.
  // Why NOT allocate a bare handle: the runtime object header reserves class
  // id 0 for builtins.object, and that is also the None singleton's id — the
  // boxed dispatchers (__ly_box_hash, __ly_box_equal) read id 0 as None, so a
  // plain instance would hash and compare as None. It is a representation
  // conflict, not a missing implementation, which is why the fix is a class
  // rather than a manifest __new__.
  if (auto contract =
          mlir::dyn_cast_if_present<py::ContractType>(instanceType);
      contract && contract.getContractName() == "builtins.object") {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, expr.range.start,
        "`object()` cannot be constructed: a bare object shares the runtime "
        "class id of the None singleton, so its identity, hash and equality "
        "would be None's. Declare a class (`class Sentinel: pass`) and "
        "instantiate that instead"});
    return emitNone(expr);
  }
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
  if (!operands.valid) {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, expr.range.start, operands.failureReason});
    return emitNone(expr);
  }

  llvm::StringMap<Value> keywords;
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

  mlir::Type inferredInstanceType = types.inferClassInstantiation(
      instanceType, operands.positionalTypes, operands.keywordTypes);
  if (!inferredInstanceType) {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, expr.range.start,
        "class instantiation leaves unbound static type parameters for '" +
            name.str() + "'"});
    return emitNone(expr);
  }
  mlir::Type classType = types.typeObject(inferredInstanceType);
  auto classObject = py::TypeObjectOp::create(builder, loc(expr), classType,
                                              inferredInstanceType);
  Value posPack = emitPack(operands.positional, operands.positionalUnpacked);
  Value namePack = emitPack(operands.keywordNames);
  Value valuePack = emitPack(operands.keywordValues);

  auto newOp = py::NewOp::create(
      builder, loc(expr), inferredInstanceType,
      mlir::FlatSymbolRefAttr::get(&context, "__new__"), callableProtocol(),
      classObject.getResult(), posPack.value, namePack.value, valuePack.value);
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
                         operands.positional, keywords);
  } else {
    bool noRuntimeInitArgs = operands.positional.empty() &&
                             operands.keywordValues.empty() &&
                             !hasUnpackedPositional;
    if (!init && noRuntimeInitArgs) {
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
  bool ordinaryStorage = tracksBinding || optionalContent(contentType);
  std::string cellName =
      (llvm::Twine(ordinaryStorage ? kSlotClassPrefix : kCellClassPrefix) +
       llvm::Twine(++cellClassCounter))
          .str();
  mlir::Type contract = types.contract(cellName);

  llvm::SmallVector<std::string, 8> fieldNames{"v"};
  llvm::SmallVector<mlir::Type, 2> fieldTypes{contentType};
  // ⛔ THE ERASED STORAGE CONTRACT IS THE CELL'S, not every slot's. A cell is
  // read back through `lowerCellAttrGet`, which rebuilds the content from the
  // box and knows the declared type; an ORDINARY field read compares the
  // storage contract against the result and reported "attribute evidence
  // 'builtins.object' is not assignable to result 'builtins.int'".
  llvm::SmallVector<mlir::Type, 8> fieldStorage{
      ordinaryStorage ? contentType : types.contract("builtins.object")};
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
  Value coerced = coerceValue(initial, content, anchor);
  mlir::Type classType = types.typeObject(cellType);
  auto classObject =
      py::TypeObjectOp::create(builder, loc(anchor), classType, cellType);
  llvm::SmallVector<Value, 2> initialFields{coerced};
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
  llvm::SmallVector<mlir::Type, 3> positional{cellType, content};
  llvm::SmallVector<mlir::StringAttr, 3> positionalNames{
      builder.getStringAttr("self"), builder.getStringAttr("v")};
  llvm::SmallVector<mlir::BoolAttr, 3> positionalDefaults{
      builder.getBoolAttr(false), builder.getBoolAttr(true)};
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
  auto op = py::AttrSetOp::create(builder, loc(anchor), cell.value, "v",
                                  coerced.value);
  op->setAttr("ly.attr.kind", builder.getStringAttr("field"));
  auto contract = mlir::cast<py::ContractType>(cell.type);
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
  op->setAttr("ly.attr.owner",
              builder.getStringAttr(contract.getContractName()));
}

} // namespace lython::emitter
