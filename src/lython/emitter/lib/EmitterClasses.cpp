#include "AstSynth.h"
#include "EmitterCore.h"
#include "EmitterPyOps.h"
#include "EmitterSupport.h"

#include "AstAccess.h"
#include "ClosureAnalysis.h"
#include "Contracts.h"
#include "ExceptionTaxonomy.h"
#include "PyProtocols.h"
#include "TypeSystemSolver.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/SaveAndRestore.h"

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

// C3 merge over contract-name sequences. Nullopt when no linearization
// exists (the CPython TypeError case); the caller owns the diagnostic.
std::optional<llvm::SmallVector<std::string, 8>>
c3MergeNames(llvm::SmallVector<llvm::SmallVector<std::string, 8>, 8> sequences) {
  llvm::SmallVector<std::string, 8> result;
  auto compact = [&]() {
    llvm::SmallVector<llvm::SmallVector<std::string, 8>, 8> next;
    for (auto &sequence : sequences)
      if (!sequence.empty())
        next.push_back(std::move(sequence));
    sequences = std::move(next);
  };
  compact();
  while (!sequences.empty()) {
    std::optional<std::string> candidate;
    for (const auto &sequence : sequences) {
      const std::string &head = sequence.front();
      bool appearsInTail = false;
      for (const auto &other : sequences) {
        if (llvm::is_contained(
                llvm::ArrayRef<std::string>(other).drop_front(), head)) {
          appearsInTail = true;
          break;
        }
      }
      if (!appearsInTail) {
        candidate = head;
        break;
      }
    }
    if (!candidate)
      return std::nullopt;
    result.push_back(*candidate);
    for (auto &sequence : sequences)
      if (!sequence.empty() && sequence.front() == *candidate)
        sequence.erase(sequence.begin());
    compact();
  }
  return result;
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
      recognized = leaf == "native" || isTypingMarker(leaf);
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
  auto contract = mlir::dyn_cast_if_present<py::ContractType>(receiver.type);
  if (!contract)
    return false;
  if (throughSuper)
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
  if (!redeclared)
    return false;
  // ⛔ NOT gated on the name resolving ON THE RECEIVER. A subclass may be the
  // only class that declares it -- `class A: pass` / `class B(A): __repr__` --
  // and `repr(a)` on a base-typed `a` then ran object's repr and printed
  // `<__main__.A object at 0x...>` where CPython prints B's. That a subclass
  // declares it is the whole evidence the dispatch is real; requiring the base
  // to declare it too made the gate blind to exactly the case where the
  // subclass introduces the method.
  diagnostics.push_back(parser::Diagnostic{
      parser::Severity::Error, anchor.range.start,
      "'" + std::string(methodName) + "' is overridden by a subclass of '" +
          contract.getContractName().str() +
          "', so this call cannot be resolved from the static type of the "
          "receiver"});
  return true;
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
          else if ((*keywordName == "frozen" || *keywordName == "order") &&
                   !*flag)
            ; // explicit False matches the synthesized behavior
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
        c3MergeNames(std::move(sequences));
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
          storable = attrContractName == "builtins.bytes" ||
                     !attrContractName.contains('.');
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
      FunctionSignature bodySig = types.functionSignature(
          *statement,
          kind == "static" ? std::optional<llvm::StringRef>() : receiverName,
          py::CallableType(), receiverType);
      if (kind == "instance" || propertyAccessor)
        replaceSelfInSignature(bodySig, types.contract(contractName), types);
      else if (kind == "class" || kind == "classmethod") {
        replaceSelfInSignature(
            bodySig, types.typeObject(types.contract(contractName)), types);
        if (!bodySig.positionalTypes.empty()) {
          bodySig.positionalTypes.front() =
              types.typeObject(types.contract(contractName));
          types.refreshCallable(bodySig);
        }
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
  auto setField = [&](llvm::StringRef name, mlir::Type type,
                      bool overwriteExisting, const parser::Node &anchor) {
    if (name.empty())
      return;
    if (!type) {
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, anchor.range.start,
          "class field '" + name.str() +
              "' requires a statically inferred type"});
      return;
    }
    mlir::Type storedType = types.widenLiteral(type);
    for (auto [index, existing] : llvm::enumerate(fieldNames)) {
      if (existing != name)
        continue;
      if (overwriteExisting)
        fieldTypes[index] = storedType;
      return;
    }
    fieldNames.push_back(name.str());
    fieldTypes.push_back(storedType);
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
  auto collectTarget = [&](const parser::Node &target, mlir::Type type) {
    if (target.kind != "Attribute")
      return;
    const parser::Node *object = ast::node(target, "value");
    if (!object || !ast::isName(*object, "self"))
      return;
    if (auto attr = ast::string(target, "attr")) {
      // `self.<prop> = ...` runs the property setter; it declares no field.
      if (propertyNames.contains(*attr))
        return;
      setField(*attr, type, /*overwriteExisting=*/false, target);
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
      if (const auto *stmts = ast::nodeList(*method, "body")) {
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
          } else if (stmt->kind == "Assign") {
            mlir::Type valueType = valueTypeOf(ast::node(*stmt, "value"));
            if (const auto *targets = ast::nodeList(*stmt, "targets"))
              for (const parser::NodePtr &target : *targets) {
                if (!target)
                  continue;
                bindInitLocal(&*target, valueType);
                collectTarget(*target, valueType);
              }
          }
        }
      }
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

Value ModuleEmitter::emitInlineMethodCall(const parser::Node &expr,
                                          Value receiver,
                                          const MethodBinding &method) {
  if (!method.method)
    return emitNone(expr);

  llvm::SmallVector<Value, 8> positional;
  if (const auto *args = ast::nodeList(expr, "args")) {
    for (const parser::NodePtr &arg : *args) {
      if (arg && arg->kind == "Starred") {
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, arg->range.start,
            "starred arguments are not supported for inlined class methods"});
        continue;
      }
      positional.push_back(emitExpr(arg.get()));
    }
  }

  llvm::StringMap<Value> keywords;
  if (const auto *keywordNodes = ast::nodeList(expr, "keywords")) {
    for (const parser::NodePtr &keyword : *keywordNodes) {
      if (auto name = ast::string(*keyword, "arg")) {
        keywords[*name] = emitExpr(ast::node(*keyword, "value"));
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

Value ModuleEmitter::emitInlineMethodBody(
    const parser::Node &anchor, Value receiver, bool bindDescriptorReceiver,
    const MethodBinding &method, llvm::ArrayRef<Value> positional,
    const llvm::StringMap<Value> &keywords) {
  if (!method.method)
    return emitNone(anchor);
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
    if (!recursive || method.symbolName.empty() || !bindDescriptorReceiver ||
        !keywords.empty()) {
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
    arguments.append(positional.begin(), positional.end());
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

  for (Value argument : positional) {
    if (parameterIndex >= sig.positionalNames.size()) {
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, anchor.range.start,
          "too many positional arguments for inlined class method"});
      break;
    }
    if (parameterIndex < sig.positionalTypes.size())
      checkArgument(sig.positionalNames[parameterIndex],
                    sig.positionalTypes[parameterIndex], argument);
    bind(sig.positionalNames[parameterIndex++], argument);
  }

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
      if (index < sig.positionalTypes.size())
        checkArgument(name, sig.positionalTypes[index], value);
      bind(name, value);
      return;
    }
    for (auto [index, kwOnlyName] : llvm::enumerate(sig.kwOnlyNames)) {
      if (kwOnlyName != name)
        continue;
      if (index < sig.kwOnlyTypes.size())
        checkArgument(name, sig.kwOnlyTypes[index], value);
      bind(name, value);
      return;
    }
    diagnostics.push_back(
        parser::Diagnostic{parser::Severity::Error, anchor.range.start,
                           "unexpected keyword argument '" + name.str() +
                               "' for inlined class method"});
  };
  for (auto &entry : keywords)
    bindKeyword(entry.getKey(), entry.getValue());

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
  for (auto [index, name] : llvm::enumerate(sig.positionalNames)) {
    if (bound.contains(name))
      continue;
    if (const parser::Node *defaultNode =
            positionalDefault(static_cast<unsigned>(index))) {
      Value defaultValue = emitExpr(defaultNode);
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
      Value defaultValue = emitExpr(defaultNode);
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
  bool pushedSuperContext = bindDescriptorReceiver &&
                            method.kind == "instance" &&
                            !method.definingClass.empty() &&
                            !sig.positionalNames.empty();
  if (pushedSuperContext)
    superContexts.push_back(
        SuperContext{method.definingClass, sig.positionalNames.front()});
  methodsBeingInlined.push_back(method.method);
  emitStatements(body);
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
  CallOperands operands = emitCallOperands(expr);
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
  bool hasUnpackedPositional = llvm::any_of(
      operands.positionalUnpacked, [](char value) { return value != 0; });
  if (hasUnpackedPositional) {
    if (const auto *args = ast::nodeList(expr, "args")) {
      for (const parser::NodePtr &arg : *args) {
        if (!arg || arg->kind != "Starred")
          continue;
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, arg->range.start,
            "starred arguments are not supported for source class "
            "instantiation"});
      }
    }
  }

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

bool ModuleEmitter::isCellContract(mlir::Type type) {
  auto contract = mlir::dyn_cast_if_present<py::ContractType>(type);
  return contract && contract.getContractName().starts_with(kCellClassPrefix);
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

mlir::Type ModuleEmitter::ensureCellClass(mlir::Type contentType,
                                          const parser::Node &anchor) {
  auto memoized = cellClassContracts.find(contentType);
  if (memoized != cellClassContracts.end())
    return memoized->second;

  std::string cellName =
      (llvm::Twine(kCellClassPrefix) + llvm::Twine(++cellClassCounter)).str();
  mlir::Type contract = types.contract(cellName);

  classBaseNames[cellName] = {};
  classMros[cellName] = {cellName, "builtins.object"};
  classOwnFieldOrders[cellName] = {"v"};
  classFieldOrders[cellName] = {"v"};
  classFieldBindings[cellName]["v"] = contentType;

  py::protocols::ProtocolInfo protocolInfo;
  protocolInfo.fields["v"] = contentType;
  py::protocols::Table::getMutable(context).registerClass(
      cellName, std::move(protocolInfo));

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(module.getBody());
  mlir::OperationState state(loc(anchor), py::ClassOp::getOperationName());
  state.addAttribute(mlir::SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(cellName));
  state.addAttribute("base_names",
                     stringArray(builder, llvm::ArrayRef<std::string>{}));
  state.addAttribute("field_names",
                     stringArray(builder, llvm::ArrayRef<std::string>{"v"}));
  state.addAttribute("field_types",
                     typeArray(builder, llvm::ArrayRef<mlir::Type>{contentType}));
  // The STORAGE contract is the erased object: the existing box-fronted
  // field rule then gives the cell one stable box16 slot (allocation,
  // deallocator and init paths all follow that rule), which is what lets a
  // closure's store reach every other frame holding the cell.
  state.addAttribute(
      "field_contract_types",
      typeArray(builder, llvm::ArrayRef<mlir::Type>{
                             types.contract("builtins.object")}));
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

  cellClassContracts[contentType] = contract;
  return contract;
}

Value ModuleEmitter::emitCellAlloc(const parser::Node &anchor, Value initial) {
  mlir::Type content = types.widenLiteral(initial.type);
  mlir::Type cellType = ensureCellClass(content, anchor);
  Value coerced = coerceValue(initial, content, anchor);
  mlir::Type classType = types.typeObject(cellType);
  auto classObject =
      py::TypeObjectOp::create(builder, loc(anchor), classType, cellType);
  Value posPack = emitPack({coerced});
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
  llvm::SmallVector<mlir::Type, 2> positional{cellType, content};
  llvm::SmallVector<mlir::StringAttr, 2> positionalNames{
      builder.getStringAttr("self"), builder.getStringAttr("v")};
  llvm::SmallVector<mlir::BoolAttr, 2> positionalDefaults{
      builder.getBoolAttr(false), builder.getBoolAttr(true)};
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
  op->setAttr("ly.attr.owner",
              builder.getStringAttr(contract.getContractName()));
}

} // namespace lython::emitter
