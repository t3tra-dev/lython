#include "EmitterCore.h"
#include "EmitterPyOps.h"
#include "EmitterSupport.h"

#include "AstAccess.h"
#include "ClosureAnalysis.h"
#include "Contracts.h"
#include "ExceptionTaxonomy.h"
#include "PyProtocols.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/StringSet.h"

#include <utility>

namespace lython::emitter {
namespace {

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

parser::NodePtr synthName(llvm::StringRef id, parser::SourceRange range) {
  parser::NodePtr node = parser::makeNode("Name", range);
  parser::addField(*node, "id", std::string(id));
  return node;
}

parser::NodePtr synthSelfAttr(llvm::StringRef self, llvm::StringRef attr,
                              parser::SourceRange range) {
  parser::NodePtr node = parser::makeNode("Attribute", range);
  parser::addField(*node, "value", synthName(self, range));
  parser::addField(*node, "attr", std::string(attr));
  return node;
}

parser::NodePtr synthStrConstant(llvm::StringRef text,
                                 parser::SourceRange range) {
  parser::NodePtr node = parser::makeNode("Constant", range);
  parser::addField(*node, "value", std::string(text));
  return node;
}

parser::NodePtr synthAdd(parser::NodePtr lhs, parser::NodePtr rhs,
                         parser::SourceRange range) {
  parser::NodePtr node = parser::makeNode("BinOp", range);
  parser::NodePtr op = parser::makeNode("Add", range);
  parser::addField(*node, "left", std::move(lhs));
  parser::addField(*node, "op", std::move(op));
  parser::addField(*node, "right", std::move(rhs));
  return node;
}

parser::NodePtr synthReprCall(parser::NodePtr argument,
                              parser::SourceRange range) {
  parser::NodePtr node = parser::makeNode("Call", range);
  parser::addField(*node, "func", synthName("repr", range));
  parser::addField(*node, "args", std::vector<parser::NodePtr>{argument});
  parser::addField(*node, "keywords", std::vector<parser::NodePtr>{});
  return node;
}

parser::NodePtr synthArg(llvm::StringRef name, parser::SourceRange range) {
  parser::NodePtr node = parser::makeNode("arg", range);
  parser::addField(*node, "arg", std::string(name));
  return node;
}

parser::NodePtr
synthFunctionDef(llvm::StringRef name, llvm::ArrayRef<std::string> paramNames,
                 std::vector<parser::NodePtr> defaults,
                 std::vector<parser::NodePtr> body,
                 parser::SourceRange range) {
  parser::NodePtr arguments = parser::makeNode("arguments", range);
  std::vector<parser::NodePtr> args;
  for (const std::string &param : paramNames)
    args.push_back(synthArg(param, range));
  parser::addField(*arguments, "posonlyargs", std::vector<parser::NodePtr>{});
  parser::addField(*arguments, "args", std::move(args));
  parser::addField(*arguments, "kwonlyargs", std::vector<parser::NodePtr>{});
  parser::addField(*arguments, "kw_defaults", std::vector<parser::NodePtr>{});
  parser::addField(*arguments, "defaults", std::move(defaults));

  parser::NodePtr node = parser::makeNode("FunctionDef", range);
  parser::addField(*node, "name", std::string(name));
  parser::addField(*node, "args", std::move(arguments));
  parser::addField(*node, "body", std::move(body));
  parser::addField(*node, "decorator_list", std::vector<parser::NodePtr>{});
  return node;
}

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

bool ModuleEmitter::isExceptionBackedClass(llvm::StringRef className) const {
  for (const std::string &cls : classMro(className))
    if (!classMros.count(cls) && taxonomyEntryForContract(cls))
      return true;
  return false;
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
      std::string qualified = ast::qualifiedName(base.get());
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
        baseList += py::contracts::manifestClassNameForContract(base);
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
  // machinery (default_factory etc.) is not modeled.
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
            diagnostics.push_back(parser::Diagnostic{
                parser::Severity::Error, value->range.start,
                "dataclasses.field(...) defaults are not supported yet"});
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
  // this class's id + message); it has no field storage, so declared
  // instance fields are rejected rather than silently dropped.
  if (isExceptionBackedClass(contractName) && !fieldNames.empty()) {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, classDef.range.start,
        "user exception class '" + classSymbol +
            "' cannot declare instance fields yet (field '" +
            fieldNames.front() + "'); use the exception message instead"});
    return;
  }

  llvm::StringMap<mlir::Type> &registeredFields =
      classFieldBindings[contractName];
  registeredFields.clear();
  llvm::SmallVector<std::string, 8> &registeredOrder =
      classFieldOrders[contractName];
  registeredOrder.assign(fieldNames.begin(), fieldNames.end());
  for (auto [fieldName, fieldType] : llvm::zip_equal(fieldNames, fieldTypes))
    registeredFields[fieldName] = fieldType;

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
      FunctionSignature bodySig = types.functionSignature(
          *statement,
          kind == "static" ? std::optional<llvm::StringRef>() : receiverName);
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
    auto userDefines = [&](llvm::StringRef method) {
      auto ownMethods = classMethodBindings.find(contractName);
      return ownMethods != classMethodBindings.end() &&
             ownMethods->second.count(method);
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
        parser::NodePtr assign = parser::makeNode("Assign", range);
        parser::addField(*assign, "targets",
                         std::vector<parser::NodePtr>{
                             synthSelfAttr("self", field, range)});
        parser::addField(*assign, "value", synthName(field, range));
        body.push_back(std::move(assign));
      }
      if (body.empty())
        body.push_back(parser::makeNode("Pass", range));
      sig.resultType = types.none();
      registerSynthesized(
          synthFunctionDef("__init__", paramNames, std::move(defaults),
                           std::move(body), range),
          std::move(sig));
    }
    if (dataclassRepr && !userDefines("__repr__")) {
      std::string className =
          py::contracts::manifestClassNameForContract(contractName);
      parser::NodePtr expr;
      if (order.empty()) {
        expr = synthStrConstant(className + "()", range);
      } else {
        expr = synthStrConstant(className + "(", range);
        for (auto [index, field] : llvm::enumerate(order)) {
          std::string label = (index ? ", " : "") + field + "=";
          expr = synthAdd(std::move(expr), synthStrConstant(label, range),
                          range);
          expr = synthAdd(
              std::move(expr),
              synthReprCall(synthSelfAttr("self", field, range), range),
              range);
        }
        expr = synthAdd(std::move(expr), synthStrConstant(")", range), range);
      }
      parser::NodePtr returnNode = parser::makeNode("Return", range);
      parser::addField(*returnNode, "value", std::move(expr));
      FunctionSignature sig;
      sig.positionalNames.push_back("self");
      sig.positionalTypes.push_back(types.contract(contractName));
      sig.positionalDefaults.push_back(false);
      sig.resultType = types.strType();
      registerSynthesized(
          synthFunctionDef("__repr__", {"self"}, {},
                           {std::move(returnNode)}, range),
          std::move(sig));
    }
    if (dataclassEq && !userDefines("__eq__")) {
      parser::NodePtr expr;
      llvm::SmallVector<parser::NodePtr, 8> comparisons;
      for (const std::string &field : order) {
        parser::NodePtr compare = parser::makeNode("Compare", range);
        parser::addField(*compare, "left", synthSelfAttr("self", field, range));
        parser::addField(*compare, "ops",
                         std::vector<parser::NodePtr>{
                             parser::makeNode("Eq", range)});
        parser::addField(*compare, "comparators",
                         std::vector<parser::NodePtr>{
                             synthSelfAttr("other", field, range)});
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
      parser::NodePtr returnNode = parser::makeNode("Return", range);
      parser::addField(*returnNode, "value", std::move(expr));
      FunctionSignature sig;
      sig.positionalNames.append({"self", "other"});
      sig.positionalTypes.push_back(types.contract(contractName));
      sig.positionalTypes.push_back(types.contract(contractName));
      sig.positionalDefaults.append({false, false});
      sig.resultType = types.boolType();
      registerSynthesized(
          synthFunctionDef("__eq__", {"self", "other"}, {},
                           {std::move(returnNode)}, range),
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
  for (auto [fieldName, fieldType] : llvm::zip_equal(fieldNames, fieldTypes))
    protocolInfo.fields[fieldName] = fieldType;
  for (auto [methodName, methodContract] :
       llvm::zip_equal(methodNames, methodContracts)) {
    std::string registeredMethodName = methodName;
    auto pushSignature = [&](py::CallableType signature) {
      if (!signature)
        return;
      py::protocols::ProtocolMethod method;
      method.signature = signature;
      method.mayThrow = true;
      protocolInfo.methods[registeredMethodName].push_back(method);
    };
    if (auto signature =
            mlir::dyn_cast_if_present<py::CallableType>(methodContract)) {
      pushSignature(signature);
    } else if (auto overload = mlir::dyn_cast_if_present<py::OverloadType>(
                   methodContract)) {
      for (mlir::Type candidate : overload.getCandidateTypes())
        pushSignature(mlir::dyn_cast_if_present<py::CallableType>(candidate));
    }
  }
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

void ModuleEmitter::collectModuleGlobals(const parser::Node &moduleNode) {
  // Opt-in: an annotated module-level assignment (`NAME: T = ...`) becomes
  // a storage-backed mutable global (int keeps its unboxed i64 cell for the
  // signal-safe channel; other contracts store their physical value words).
  // Plain `NAME = expr` at module scope keeps its value-binding behavior
  // (module-scope constants).
  const auto *body = ast::nodeList(moduleNode, "body");
  if (!body)
    return;
  for (const parser::NodePtr &statement : *body) {
    if (!statement || statement->kind != "AnnAssign")
      continue;
    const parser::Node *target = ast::node(*statement, "target");
    if (!target || target->kind != "Name")
      continue;
    mlir::Type annotated = types.widenLiteral(
        types.annotationType(ast::node(*statement, "annotation")));
    if (!annotated)
      continue;
    // Storage-backed globals cover the immutable scalars plus user classes
    // (whose mutation happens in place on the heap). Containers stay
    // value-bound: their structural mutations reallocate the interior
    // arrays through SSA rebinding, which a storage cell would go stale
    // against; unions stay value-bound so isinstance narrowing keeps
    // working on the module flow.
    bool storageBacked =
        annotated == types.intType() || annotated == types.strType() ||
        annotated == types.floatType() || annotated == types.boolType();
    if (!storageBacked) {
      if (auto contract = mlir::dyn_cast<py::ContractType>(annotated)) {
        llvm::StringRef contractName = contract.getContractName();
        storageBacked = contractName == "builtins.bytes" ||
                        !contractName.contains('.');
      }
    }
    if (!storageBacked)
      continue;
    llvm::StringRef name = ast::nameSpelling(*target);
    moduleGlobals[name] = annotated;
    types.bindSymbol(name, annotated);
  }
}

bool ModuleEmitter::isModuleGlobalRead(llvm::StringRef name) const {
  // A read resolves to the module global unless a local (function-scope)
  // binding shadows it.
  return moduleGlobals.count(name) && values.find(name) == values.end();
}

bool ModuleEmitter::isModuleGlobalWrite(llvm::StringRef name) const {
  if (!moduleGlobals.count(name))
    return false;
  // Module scope always writes the global; a function writes it only when it
  // declared `global NAME` (otherwise the assignment makes a local).
  return atModuleScope || currentGlobalDecls.count(name);
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
      llvm::StringMap<mlir::Type> initArgTypes;
      collectInitArgTypes(*method, initArgTypes);
      if (const auto *stmts = ast::nodeList(*method, "body")) {
        for (const parser::NodePtr &stmt : *stmts) {
          if (!stmt)
            continue;
          if (stmt->kind == "AnnAssign") {
            collectTarget(*ast::node(*stmt, "target"),
                          types.annotationType(ast::node(*stmt, "annotation")));
          } else if (stmt->kind == "Assign") {
            const parser::Node *value = ast::node(*stmt, "value");
            mlir::Type valueType;
            if (value && value->kind == "Name") {
              auto found = initArgTypes.find(ast::nameSpelling(*value));
              if (found != initArgTypes.end())
                valueType = found->second;
            } else {
              valueType = types.inferExpr(value);
            }
            if (const auto *targets = ast::nodeList(*stmt, "targets"))
              for (const parser::NodePtr &target : *targets)
                collectTarget(*target, valueType);
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
  // Inference runs against the receiver type: the protocol table resolves
  // __init__ through the class's bases anyway, and the evidence verifier
  // checks the selected contract against the receiver.
  (void)baseContract;
  CallInferenceResult inference = types.inferMethodCallWithEvidence(
      receiver.type, "__init__", positionalTypes);
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
                  py::contracts::manifestClassNameForContract(startClass) +
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
                  py::contracts::manifestClassNameForContract(cls) +
                  "', which super() cannot call yet");
  }
  return reject("'super' object has no attribute '" +
                std::string(*methodName) + "'");
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
  const FunctionSignature &sig =
      method.bodySignature.callable ? method.bodySignature : method.signature;
  const auto *body = ast::nodeList(*method.method, "body");
  mlir::Type resultType = sig.resultType ? sig.resultType : types.none();

  ScopedEmitterScope scope(values, types);
  llvm::StringSet<> bound;
  auto bind = [&](llvm::StringRef name, Value value) {
    values[name] = value;
    types.bindSymbol(name, value.type);
    bound.insert(name);
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
    bind(sig.positionalNames[parameterIndex++], argument);
  }

  auto bindKeyword = [&](llvm::StringRef name, Value value) {
    for (llvm::StringRef positionalName : sig.positionalNames) {
      if (positionalName != name)
        continue;
      if (bound.contains(name)) {
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, anchor.range.start,
            "multiple values for inlined class method argument '" + name.str() +
                "'"});
        return;
      }
      bind(name, value);
      return;
    }
    for (llvm::StringRef kwOnlyName : sig.kwOnlyNames) {
      if (kwOnlyName != name)
        continue;
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
  bool pushedSuperContext = bindDescriptorReceiver &&
                            method.kind == "instance" &&
                            !method.definingClass.empty() &&
                            !sig.positionalNames.empty();
  if (pushedSuperContext)
    superContexts.push_back(
        SuperContext{method.definingClass, sig.positionalNames.front()});
  emitStatements(body);
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

} // namespace lython::emitter
