#include "AstAccess.h"
#include "AstSynth.h"
#include "Contracts.h"
#include "EmitterCore.h"
#include "EmitterPyOps.h"
#include "PyProtocols.h"
#include "TypeInference.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"

#include <limits>
#include <optional>
#include <string>

namespace lython::emitter {
namespace {

// CPython 3.14's `_PyCompile_CleanDoc`, which every docstring goes through
// before it is stored: tabs expanded to columns of eight, the first line's
// leading spaces dropped, and the smallest indentation of the later lines
// that have text taken off each later line. Blank and trailing lines stay.
std::string cleanDocstring(llvm::StringRef raw) {
  std::string doc;
  doc.reserve(raw.size());
  unsigned column = 0;
  for (char c : raw) {
    if (c == '\t') {
      unsigned spaces = 8 - column % 8;
      doc.append(spaces, ' ');
      column += spaces;
      continue;
    }
    doc.push_back(c);
    if (c == '\n' || c == '\r')
      column = 0;
    else if ((static_cast<unsigned char>(c) & 0xC0) != 0x80)
      ++column;
  }
  // The reads past `end` stop on the terminating NUL, as CPython's do.
  const char *begin = doc.c_str();
  const char *end = begin + doc.size();
  const char *p = begin;
  while (p < end && *p++ != '\n') {
  }
  std::size_t margin = std::numeric_limits<std::size_t>::max();
  while (p < end) {
    const char *start = p;
    while (*p == ' ')
      ++p;
    if (p < end && *p != '\n')
      margin = std::min<std::size_t>(margin, static_cast<std::size_t>(p - start));
    while (p < end && *p++ != '\n') {
    }
  }
  if (margin == std::numeric_limits<std::size_t>::max())
    margin = 0;
  p = begin;
  while (*p == ' ')
    ++p;
  if (p == begin && margin == 0)
    return doc;
  std::string out;
  out.reserve(doc.size());
  while (p < end) {
    char c = *p++;
    out.push_back(c);
    if (c == '\n')
      break;
  }
  while (p < end) {
    for (std::size_t index = 0; index < margin; ++index, ++p)
      if (*p != ' ')
        break;
    while (p < end) {
      char c = *p++;
      out.push_back(c);
      if (c == '\n')
        break;
    }
  }
  return out;
}

// The string a body opens with, which is the docstring of the def, class or
// module that has the body. An f-string or a bytes literal is not one.
std::optional<std::string> docstringOf(const parser::Node &owner) {
  const auto *body = ast::nodeList(owner, "body");
  if (!body || body->empty() || !body->front() ||
      body->front()->kind != "Expr")
    return std::nullopt;
  const parser::Node *value = ast::node(*body->front(), "value");
  if (!value || value->kind != "Constant")
    return std::nullopt;
  std::optional<std::string_view> text = ast::string(*value, "value");
  if (!text)
    return std::nullopt;
  return cleanDocstring(llvm::StringRef(text->data(), text->size()));
}

bool isPropertyOrBindingMarker(const parser::Node &decorator) {
  std::string spelled = ast::qualifiedName(&decorator);
  if (spelled.empty() && decorator.kind == "Attribute")
    if (auto attr = ast::string(decorator, "attr"))
      // `@p.setter` and `@p.deleter` keep the getter's docstring.
      return *attr == "setter" || *attr == "deleter" || *attr == "getter";
  llvm::StringRef leaf = llvm::StringRef(spelled).rsplit('.').second;
  if (leaf.empty())
    leaf = spelled;
  return leaf == "staticmethod" || leaf == "classmethod" ||
         leaf == "property" || leaf == "overload" || leaf == "override" ||
         leaf == "final" || leaf == "setter" || leaf == "deleter" ||
         leaf == "getter";
}

} // namespace

// ⭐ A DOCSTRING IS KEPT WHERE IT IS READ AND NOWHERE ELSE. Nothing stores
// one on a function or a class: a read whose subject is known answers the
// constant, and a read whose subject the run time chooses answers from a
// function listing the docstrings of the subjects it may be. A docstring no
// read can reach is never emitted, which is the dead-code elimination the
// decision asks for, at no cost to the programs that never read one.
//
// Recorded here, before the first desugar: the desugars make classes and
// functions of their own -- a decorated method's forwarder, a context
// manager's class -- whose docstring is what CPython keeps for the original,
// so a definition this walk did not see is answered as not known rather than
// as having none.
void ModuleEmitter::recordDocstrings(const parser::Node &moduleNode,
                                     llvm::StringRef moduleName) {
  moduleDocstrings[moduleName] = docstringOf(moduleNode);
  ast::walk(&moduleNode, [&](const parser::Node &node) {
    if (node.kind == "FunctionDef" || node.kind == "AsyncFunctionDef" ||
        node.kind == "ClassDef")
      definitionDocstrings[&node] = docstringOf(node);
    else if (node.kind == "Lambda")
      definitionDocstrings[&node] = std::nullopt;
    return ast::Walk::Continue;
  });
}

ModuleEmitter::DocAnswer
ModuleEmitter::docOfDefinition(const parser::Node *definition,
                               llvm::StringRef shown) const {
  DocAnswer answer;
  auto found =
      definition ? definitionDocstrings.find(definition)
                 : definitionDocstrings.end();
  if (found == definitionDocstrings.end()) {
    answer.text = "'" + shown.str() + "' was written by the compiler";
    return answer;
  }
  // ⛔ A decorator other than the ones that bind a def as a method replaces
  // what the name holds, and the docstring CPython answers is the one of
  // what it returned.
  if (const auto *decorators = ast::nodeList(*definition, "decorator_list"))
    for (const parser::NodePtr &decorator : *decorators)
      if (decorator && !isPropertyOrBindingMarker(*decorator)) {
        answer.text = "'" + shown.str() + "' is decorated";
        return answer;
      }
  if (found->second) {
    answer.kind = DocAnswer::Kind::Text;
    answer.text = *found->second;
  } else {
    answer.kind = DocAnswer::Kind::None;
  }
  return answer;
}

// A class's own docstring: CPython does not inherit it, and a class
// statement without one stores None.
// ⛔ Not a class whose docstring something writes: a decorated class (a
// dataclass's is generated, `P(x: int)`) or a NamedTuple (`P(x, y)`).
ModuleEmitter::DocAnswer ModuleEmitter::docOfClass(llvm::StringRef cls) const {
  std::string shown = py::contracts::pythonQualnameForContract(cls);
  DocAnswer answer;
  const parser::Node *classDef = declaredClassNodes.lookup(cls);
  if (!classDef) {
    answer.text = "'" + shown + "' is not a class this program declares";
    return answer;
  }
  if (const auto *decorators = ast::nodeList(*classDef, "decorator_list");
      decorators && !decorators->empty()) {
    answer.text = "'" + shown + "' is decorated, and a decorator may write "
                                "its docstring";
    return answer;
  }
  if (namedTupleContracts.count(cls)) {
    answer.text = "the docstring of NamedTuple '" + shown + "' is generated";
    return answer;
  }
  return docOfDefinition(classDef, shown);
}

// The docstring of the method `method` resolves to on class `cls`.
ModuleEmitter::DocAnswer ModuleEmitter::docOfMethod(llvm::StringRef cls,
                                                    llvm::StringRef method) {
  std::string shown =
      py::contracts::pythonQualnameForContract(cls) + "." + method.str();
  DocAnswer answer;
  mlir::Type receiver = types.lookupClass(cls).value_or(mlir::Type());
  std::optional<MethodBinding> binding =
      receiver ? lookupClassMethod(receiver, method) : std::nullopt;
  if (!binding || !binding->method) {
    answer.text = "'" + shown + "' is not a method this program declares";
    return answer;
  }
  // A decorated method is the cell its decorators filled
  // (desugarMethodDecorators), and its docstring is the one of what they
  // returned.
  const parser::Node *definer = declaredClassNodes.lookup(binding->definingClass);
  for (const DecoratedMethod &decorated : decoratedMethods)
    if (decorated.classDef == definer &&
        decorated.hiddenName == "__ly_undecorated$" + method.str()) {
      if (decorated.source) {
        answer.text = "'" + shown +
                      "' is decorated in an imported module, whose cell is "
                      "not read from here";
        return answer;
      }
      answer.kind = DocAnswer::Kind::Cell;
      answer.text = decorated.cellName;
      return answer;
    }
  return docOfDefinition(binding->method,
                         py::contracts::pythonQualnameForContract(
                             binding->definingClass) +
                             "." + method.str());
}

std::optional<ModuleEmitter::DocAnswer>
ModuleEmitter::decidedMethodDoc(llvm::StringRef cls, llvm::StringRef method) {
  DocAnswer own = docOfMethod(cls, method);
  for (const std::string &sub : programSubclassesOf(cls))
    if (!(docOfMethod(sub, method) == own))
      return std::nullopt;
  return own;
}

// Whether `x.__doc__` has an answer for a value of `type` without refusing:
// the program's functions and classes, and the instances of its classes,
// except a class whose one possible answer is not known.
bool ModuleEmitter::docOfValueIsAnswered(mlir::Type type) const {
  type = types.widenLiteral(type);
  if (mlir::isa_and_nonnull<py::CallableType>(type))
    return true;
  mlir::Type instance = type;
  if (auto typeObject = mlir::dyn_cast_if_present<py::TypeType>(type))
    instance = typeObject.getInstanceType();
  auto contract = mlir::dyn_cast_if_present<py::ContractType>(instance);
  if (!contract || !declaredClassNodes.count(contract.getContractName()) ||
      lookupClassField(contract, "__doc__"))
    return false;
  return !programSubclassesOf(contract.getContractName()).empty() ||
         docOfClass(contract.getContractName()).kind !=
             DocAnswer::Kind::Unknown;
}

parser::NodePtr
ModuleEmitter::docAnswerStatement(const DocAnswer &answer,
                                  parser::SourceRange range) {
  switch (answer.kind) {
  case DocAnswer::Kind::Text:
    return synth::returnStmt(synth::strConstant(answer.text, range), range);
  case DocAnswer::Kind::None:
    return synth::returnStmt(synth::noneConstant(range), range);
  case DocAnswer::Kind::Cell:
    return synth::returnStmt(
        synth::attribute(synth::name(answer.text, range), "__doc__", range),
        range);
  case DocAnswer::Kind::Unknown:
    break;
  }
  return synth::raiseCall("TypeError",
                          "'__doc__' is not supported here: " + answer.text,
                          range);
}

Value ModuleEmitter::emitDocConstant(const parser::Node &anchor,
                                     const DocAnswer &answer) {
  mlir::Type docType = types.join({types.strType(), types.none()});
  Value raw;
  if (answer.kind == DocAnswer::Kind::Text) {
    mlir::Type literal = types.literal("\"" + answer.text + "\"");
    raw = Value{py::StrConstantOp::create(builder, loc(anchor), literal,
                                          builder.getStringAttr(answer.text))
                    .getResult(),
                literal};
  } else {
    raw = Value{
        py::NoneOp::create(builder, loc(anchor), types.none()).getResult(),
        types.none()};
  }
  return coerceValue(raw, docType, anchor);
}

// One function per question, `def <prefix>N(<params>) -> str | None`, whose
// body is written after the main module's last statement, in its scope: a
// body that lists functions has to come after the last one, and a body that
// reads a decorated method's cell reads a global of the main module, which a
// body written while an imported module's code is emitted could not see.
std::string ModuleEmitter::docDispatcher(
    llvm::StringRef key, llvm::StringRef prefix,
    llvm::ArrayRef<std::pair<std::string, mlir::Type>> params,
    parser::SourceRange range,
    std::function<std::vector<parser::NodePtr>()> body) {
  if (auto found = computedLookupDispatchers.find(key);
      found != computedLookupDispatchers.end())
    return found->second;
  std::string symbol =
      prefix.str() + std::to_string(++syntheticFunctionCounter);
  computedLookupDispatchers[key] = symbol;
  llvm::SmallVector<synth::Param, 2> synthParams;
  for (const auto &param : params)
    synthParams.push_back(synth::Param{param.first, nullptr});
  llvm::SmallVector<const parser::Node *, 2> paramNodes;
  parser::NodePtr def = synth::functionDef(
      symbol, synthParams, {}, {},
      synth::binOp(synth::name("str", range), "BitOr",
                   synth::noneConstant(range), range),
      {}, range, &paramNodes);
  synthesizedIteratorDefs.push_back(def);
  for (auto [node, param] : llvm::zip(paramNodes, params))
    types.overrideParameterType(node, param.second);
  FunctionSignature sig = types.functionSignature(*def);
  types.bindRootSymbol(symbol, sig.publicCallable);
  auto emitBody = [this, def, symbol, sig, body = std::move(body)] {
    if (parser::Field *field = parser::findField(*def, "body"))
      field->value = body();
    emitDispatcherBody(*def, symbol, sig, /*suppressGate=*/false);
  };
  lateDispatcherBodies.push_back(std::move(emitBody));
  return symbol;
}

// `__doc__` of a class the value only names at run time: the docstring of
// whichever class of `cls`'s line it holds.
std::string ModuleEmitter::classDocDispatcher(llvm::StringRef cls,
                                              parser::SourceRange range) {
  mlir::Type classType =
      types.typeObject(types.lookupClass(cls).value_or(mlir::Type()));
  return docDispatcher(
      "classdoc|" + cls.str(), "__lyclassdoc$", {{"__ly_cls", classType}},
      range,
      [this, cls = cls.str(), range] {
        std::vector<parser::NodePtr> body;
        DocAnswer own = docOfClass(cls);
        for (const std::string &sub : programSubclassesOf(cls)) {
          DocAnswer answer = docOfClass(sub);
          if (answer == own)
            continue;
          body.push_back(synth::ifStmt(
              synth::compare(synth::name("__ly_cls", range), "Is",
                             synth::name(sub, range), range),
              {docAnswerStatement(answer, range)}, {}, range));
        }
        body.push_back(docAnswerStatement(own, range));
        return body;
      });
}

// `__doc__` of the method `method` of a class the value only names at run
// time: a subclass that redefines it answers its own.
std::string ModuleEmitter::methodDocDispatcher(llvm::StringRef cls,
                                               llvm::StringRef method,
                                               parser::SourceRange range) {
  mlir::Type classType =
      types.typeObject(types.lookupClass(cls).value_or(mlir::Type()));
  return docDispatcher(
      "methoddoc|" + cls.str() + "|" + method.str(), "__lymethdoc$",
      {{"__ly_cls", classType}}, range,
      [this, cls = cls.str(), method = method.str(), range] {
        std::vector<parser::NodePtr> body;
        DocAnswer own = docOfMethod(cls, method);
        for (const std::string &sub : programSubclassesOf(cls)) {
          DocAnswer answer = docOfMethod(sub, method);
          if (answer == own)
            continue;
          body.push_back(synth::ifStmt(
              synth::compare(synth::name("__ly_cls", range), "Is",
                             synth::name(sub, range), range),
              {docAnswerStatement(answer, range)}, {}, range));
        }
        body.push_back(docAnswerStatement(own, range));
        return body;
      });
}

// `__doc__` of a function VALUE: the docstring of the function it was made
// for, which only the object knows -- its target, the one an indirect call
// through it reaches. The candidates are the functions an indirect call
// through the same type considers.
std::string ModuleEmitter::functionDocDispatcher(mlir::Type callable,
                                                 parser::SourceRange range) {
  if (!types.lookupSymbol("__ly$target_is"))
    types.bindRootSymbol(
        "__ly$target_is",
        py::CallableType::get(&context, {types.object(), types.strType()}, {},
                              mlir::Type(), mlir::Type(), {types.boolType()}));
  return docDispatcher(
      "fndoc|" + typeText(callable), "__lyfndoc$", {{"__ly_f", callable}},
      range,
      [this, callable, range] { return functionDocBody(callable, range); });
}

std::vector<parser::NodePtr>
ModuleEmitter::functionDocBody(mlir::Type callable,
                               parser::SourceRange range) {
  std::vector<parser::NodePtr> body;
  for (mlir::func::FuncOp function : module.getOps<mlir::func::FuncOp>()) {
    auto typeAttr = function->getAttrOfType<mlir::TypeAttr>("callable_type");
    llvm::StringRef symbol = function.getSymName();
    // The dispatchers this file and the computed lookups write are nobody's
    // value.
    if (!typeAttr || function.isDeclaration() || symbol.starts_with("__lyfndoc$") ||
        symbol.starts_with("__lyclassdoc$") ||
        symbol.starts_with("__lymethdoc$") ||
        symbol.starts_with("__lygetattr") || symbol.starts_with("__lyhasattr$"))
      continue;
    if (!py::isAssignableTo(typeAttr.getValue(), callable, module.getOperation()))
      continue;
    DocAnswer answer;
    if (auto text = function->getAttrOfType<mlir::StringAttr>("ly.function.doc")) {
      answer.kind = DocAnswer::Kind::Text;
      answer.text = text.getValue().str();
    } else if (function->hasAttr("ly.function.undocumented")) {
      answer.kind = DocAnswer::Kind::None;
    } else if (auto builtin =
                   function->getAttrOfType<mlir::StringAttr>("ly.function.doc_of")) {
      answer.text = "the docstring of builtin '" + builtin.getValue().str() +
                    "' is not retained";
    } else {
      auto qualname =
          function->getAttrOfType<mlir::StringAttr>("ly.function.qualname");
      answer.text = "'" +
                    (qualname ? qualname.getValue() : symbol).str() +
                    "' was written by the compiler";
    }
    body.push_back(synth::ifStmt(
        synth::call(synth::name("__ly$target_is", range),
                    {synth::name("__ly_f", range),
                     synth::strConstant(symbol, range)},
                    range),
        {docAnswerStatement(answer, range)}, {}, range));
  }
  body.push_back(synth::raiseCall(
      "TypeError",
      "'__doc__' is not supported here: the function value was made for a "
      "function this program does not declare",
      range));
  return body;
}

std::optional<Value> ModuleEmitter::tryEmitDocRead(const parser::Node &expr) {
  if (ast::string(expr, "attr").value_or("") != "__doc__")
    return std::nullopt;
  const parser::Node *subjectNode = ast::node(expr, "value");
  if (!subjectNode)
    return std::nullopt;
  parser::SourceRange range = expr.range;
  auto refuse = [&](const std::string &why) -> std::optional<Value> {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, range.start,
        "'__doc__' is not available here: " + why});
    return emitNone(expr);
  };
  // A class the way a reader spells it: `int`, not its contract.
  auto shownType = [&](mlir::Type type) -> std::string {
    if (auto contract = mlir::dyn_cast_if_present<py::ContractType>(type))
      return "'" +
             py::contracts::displayClassNameForContract(
                 contract.getContractName()) +
             "'";
    return typeText(type);
  };
  auto callDispatcher = [&](const std::string &symbol,
                            std::vector<parser::NodePtr> args) -> Value {
    parser::NodePtr call =
        synth::call(synth::name(symbol, range), std::move(args), range);
    synthesizedIteratorDefs.push_back(call);
    return emitExpr(call.get());
  };
  // The subject node itself, not a copy: what the emitter knows about a node
  // (a narrowing, a comprehension's scope) is keyed on it.
  auto shared = [](const parser::Node &parent) -> parser::NodePtr {
    if (const parser::Field *field = parser::findField(parent, "value"))
      if (const auto *child = std::get_if<parser::NodePtr>(&field->value))
        return *child;
    return nullptr;
  };
  auto typeOf = [&](parser::NodePtr subject) {
    return synth::call(synth::name("type", range), {std::move(subject)},
                       range);
  };

  // `getattr(x, name).__doc__`: with a literal name it is `x.name.__doc__`;
  // a computed one is answered per name (computedDocBody).
  if (subjectNode->kind == "Call")
    if (const parser::Node *callee = ast::node(*subjectNode, "func");
        callsUnshadowedBuiltin(callee, "getattr")) {
      const auto *args = ast::nodeList(*subjectNode, "args");
      const auto *keywords = ast::nodeList(*subjectNode, "keywords");
      if (args && args->size() == 2 && (!keywords || keywords->empty()) &&
          (*args)[0] && (*args)[1] && (*args)[0]->kind != "Starred" &&
          (*args)[1]->kind != "Starred") {
        // ⛔ Only a name the class has: a missing one raises CPython's
        // AttributeError from the per-name function, where the attribute
        // spelling would be refused.
        if (std::optional<llvm::StringRef> literal =
                literalStringArgument((*args)[1].get())) {
          mlir::Type receiverType =
              types.widenLiteral(types.inferExpr((*args)[0].get()));
          if (lookupClassMethod(receiverType, *literal) ||
              lookupClassField(receiverType, *literal) ||
              lookupClassStaticAttr(receiverType, *literal)) {
            parser::NodePtr read = synth::attribute(
                synth::attribute((*args)[0], *literal, range), "__doc__",
                range);
            synthesizedIteratorDefs.push_back(read);
            return emitExpr(read.get());
          }
        }
        return emitComputedDocRead(expr, (*args)[0], (*args)[1]);
      }
    }

  // A module's: `lib.__doc__`, through any name the import bound -- unless a
  // local of that spelling shadows the module, as in CPython.
  if (std::string spelled = ast::qualifiedName(subjectNode);
      !spelled.empty() && types.isImportedModuleName(spelled) &&
      !values.count(spelled)) {
    std::string moduleName =
        types.lookupCanonicalBinding(spelled).value_or(spelled);
    auto module = moduleDocstrings.find(moduleName);
    if (module == moduleDocstrings.end())
      return refuse("module '" + moduleName +
                    "' is not compiled from source here, and a runtime "
                    "module keeps no docstring");
    DocAnswer answer;
    answer.kind =
        module->second ? DocAnswer::Kind::Text : DocAnswer::Kind::None;
    answer.text = module->second.value_or("");
    return emitDocConstant(expr, answer);
  }

  // A method read off a class or an instance of the program: `C.m`, `x.m`.
  if (subjectNode->kind == "Attribute") {
    const parser::Node *owner = ast::node(*subjectNode, "value");
    std::string member(ast::string(*subjectNode, "attr").value_or(""));
    mlir::Type ownerType = types.widenLiteral(types.inferExpr(owner));
    bool ownerIsClass = mlir::isa_and_nonnull<py::TypeType>(ownerType);
    mlir::Type instance = ownerIsClass
                              ? mlir::cast<py::TypeType>(ownerType)
                                    .getInstanceType()
                              : ownerType;
    auto contract = mlir::dyn_cast_if_present<py::ContractType>(instance);
    std::optional<MethodBinding> binding;
    if (contract && declaredClassNodes.count(contract.getContractName()) &&
        !member.empty())
      binding = lookupClassMethod(contract, member);
    // A property read off an instance is its value, whose docstring is the
    // value's class's.
    if (binding && (ownerIsClass || binding->kind != "property")) {
      llvm::StringRef cls = contract.getContractName();
      DocAnswer own = docOfMethod(cls, member);
      bool decided = decidedMethodDoc(cls, member).has_value();
      bool exact = ownerIsClass && classValueIsExact(owner, ownerType);
      // A decorated method's answer is its cell's value's, which the
      // dispatcher reads where the cell is in scope.
      if ((exact || decided) && own.kind != DocAnswer::Kind::Cell) {
        (void)emitExpr(owner);
        if (own.kind == DocAnswer::Kind::Unknown)
          return refuse(own.text);
        return emitDocConstant(expr, own);
      }
      std::string symbol = methodDocDispatcher(cls, member, range);
      parser::NodePtr ownerPtr = shared(*subjectNode);
      return callDispatcher(symbol, {ownerIsClass ? ownerPtr : typeOf(ownerPtr)});
    }
  }

  mlir::Type subjectType = types.widenLiteral(types.inferExpr(subjectNode));
  // A function: the def's own when the subject names it, else the
  // function value's, chosen by its target.
  if (mlir::isa_and_nonnull<py::CallableType>(subjectType) ||
      mlir::isa_and_nonnull<py::OverloadType>(subjectType)) {
    Value function = emitExpr(subjectNode);
    // ⛔ A builtin is a manifest entry, and no builtin keeps its docstring.
    if (auto reference = function.value.getDefiningOp<py::BindingRefOp>();
        reference && reference.getCaptures().empty()) {
      llvm::StringRef binding = reference.getBinding();
      std::string qualified = binding.contains('.')
                                  ? binding.str()
                                  : ("builtins." + binding).str();
      if (py::protocols::Table::get(context).freeFunctionContract(qualified))
        return refuse("the docstring of builtin '" + binding.str() +
                      "' is not retained: only this program's functions, "
                      "classes and modules keep theirs");
    }
    // The def the subject names, when it is a function of this program
    // referred to by its name -- `f`, or `lib.f` and what `from lib import
    // f` bound -- and not a value holding one.
    if (auto reference = function.value.getDefiningOp<py::BindingRefOp>();
        reference && reference.getCaptures().empty()) {
      std::string spelling = ast::qualifiedName(subjectNode);
      const parser::Node *def = nullptr;
      if (subjectNode->kind == "Name" && moduleFunctionNames.count(spelling))
        def = moduleFunctionDef(spelling);
      if (!def && !spelling.empty())
        if (std::optional<std::string> canonical =
                types.lookupCanonicalBinding(spelling)) {
          auto [moduleName, leaf] = llvm::StringRef(*canonical).rsplit('.');
          for (const EmitOptions::SourceModule &source : options.sourceModules)
            if (source.moduleName == moduleName && source.moduleNode)
              if (const auto *body = ast::nodeList(*source.moduleNode, "body"))
                for (const parser::NodePtr &statement : *body)
                  if (statement &&
                      (statement->kind == "FunctionDef" ||
                       statement->kind == "AsyncFunctionDef") &&
                      ast::string(*statement, "name").value_or("") ==
                          std::string_view(leaf.data(), leaf.size()))
                    def = statement.get();
        }
      if (def) {
        DocAnswer answer = docOfDefinition(def, spelling);
        if (answer.kind == DocAnswer::Kind::Unknown)
          return refuse(answer.text);
        return emitDocConstant(expr, answer);
      }
    }
    if (!mlir::isa<py::CallableType>(subjectType))
      return refuse("the docstring of an overloaded function value is chosen "
                    "by the overload its target is, which is not decided here");
    // The value is emitted already, so it is handed over as one: the subject
    // runs once, as CPython runs it.
    std::string symbol = functionDocDispatcher(subjectType, range);
    std::optional<mlir::Type> dispatcherType = types.lookupSymbol(symbol);
    Value callee = emitBindingRef(expr, symbol,
                                  dispatcherType.value_or(mlir::Type()));
    CallOperands operands;
    operands.positional.push_back(function);
    operands.positionalTypes.push_back(function.type);
    operands.positionalUnpacked.push_back(0);
    return emitCallableDispatch(expr, callee, operands);
  }
  // A class: its own docstring, or the one of the class the value holds.
  if (auto typeObject = mlir::dyn_cast_if_present<py::TypeType>(subjectType)) {
    auto contract =
        mlir::dyn_cast_if_present<py::ContractType>(typeObject.getInstanceType());
    if (!contract || !declaredClassNodes.count(contract.getContractName()))
      return refuse("the docstring of " + shownType(typeObject.getInstanceType()) +
                    " is not retained: only this program's functions, classes "
                    "and modules keep theirs");
    llvm::StringRef cls = contract.getContractName();
    if (classValueIsExact(subjectNode, subjectType) ||
        programSubclassesOf(cls).empty()) {
      (void)emitExpr(subjectNode);
      DocAnswer answer = docOfClass(cls);
      if (answer.kind == DocAnswer::Kind::Unknown)
        return refuse(answer.text);
      return emitDocConstant(expr, answer);
    }
    return callDispatcher(classDocDispatcher(cls, range), {shared(expr)});
  }
  // An instance: its class's.
  if (auto contract = mlir::dyn_cast_if_present<py::ContractType>(subjectType);
      contract && declaredClassNodes.count(contract.getContractName())) {
    if (lookupClassField(contract, "__doc__"))
      return std::nullopt;
    llvm::StringRef cls = contract.getContractName();
    if (programSubclassesOf(cls).empty()) {
      (void)emitExpr(subjectNode);
      DocAnswer answer = docOfClass(cls);
      if (answer.kind == DocAnswer::Kind::Unknown)
        return refuse(answer.text);
      return emitDocConstant(expr, answer);
    }
    return callDispatcher(classDocDispatcher(cls, range),
                          {typeOf(shared(expr))});
  }
  return refuse("the docstring of " + shownType(subjectType) +
                " is not retained: only this program's functions, classes and "
                "modules keep theirs");
}

} // namespace lython::emitter
