#include "AstSynth.h"
#include "EmitterCore.h"

#include <cctype>

#include "AstAccess.h"

#include "llvm/ADT/StringSet.h"

namespace lython::emitter {

namespace {

// --- AST construction / mutation helpers -----------------------------------
//
// The desugar rewrites the parsed tree in place: an Enum subclass becomes a
// plain class whose members are class attributes instantiated at the ClassDef
// statement position, so every downstream layer (inference, class emission,
// the class-attribute slot channel) sees ordinary Python and needs no enum
// knowledge of its own.

void setField(parser::Node &node, std::string name, parser::FieldValue value) {
  if (parser::Field *existing = parser::findField(node, name)) {
    existing->value = std::move(value);
    return;
  }
  parser::addField(node, std::move(name), std::move(value));
}











// f"{value}<suffix>": an interpolated expression followed by a literal tail.
// Plain interpolation (no !r) goes through __format__, whose int rendering is
// repr's digits without pulling in the repr builtin's ownership shape.
parser::NodePtr synthFormattedMessage(parser::NodePtr value,
                                      llvm::StringRef suffix,
                                      parser::SourceRange range) {
  parser::NodePtr formatted = parser::makeNode("FormattedValue", range);
  parser::addField(*formatted, "value", std::move(value));
  parser::addField(*formatted, "conversion", static_cast<std::int64_t>(-1));
  parser::NodePtr joined = parser::makeNode("JoinedStr", range);
  parser::addField(*joined, "values",
                   std::vector<parser::NodePtr>{std::move(formatted),
                                                synth::strConstant(suffix, range)});
  return joined;
}



llvm::StringRef leafName(llvm::StringRef spelling) {
  auto [head, tail] = spelling.rsplit('.');
  return tail.empty() ? spelling : tail;
}

} // namespace

// The reverse-lookup entry points live on the class rather than at module
// level: a class body is the only place the desugar can inject callables
// without rewriting the module statement list. They are staticmethods, not
// classmethods, because a classmethod body inlines into its caller — which
// puts the caller's argument box on an unwind path the ownership verifier
// rejects when the not-found branch formats the value into its message.
static constexpr llvm::StringLiteral kFromValueMethod = "_lyenum_from_value_";
static constexpr llvm::StringLiteral kFromNameMethod = "_lyenum_from_name_";

std::optional<ModuleEmitter::EnumKind>
ModuleEmitter::enumBaseKind(const parser::Node &classDef) const {
  const auto *baseNodes = ast::nodeList(classDef, "bases");
  if (!baseNodes || baseNodes->size() != 1 || !baseNodes->front())
    return std::nullopt;
  // Bound to a std::string: qualifiedName returns by value, so a StringRef
  // into it dangles past the end of the initializing expression.
  const std::string qualified = ast::qualifiedName(baseNodes->front().get());
  llvm::StringRef base = leafName(qualified);
  if (base == "Enum")
    return EnumKind::Plain;
  if (base == "IntEnum")
    return EnumKind::Int;
  if (base == "StrEnum")
    return EnumKind::Str;
  return std::nullopt;
}

namespace {

// The identifiers a quoted annotation mentions. A string annotation is the
// spelling `list[T]` arrives in, and the parameter it names is inside it.
void collectAnnotationIdentifiers(llvm::StringRef text,
                                  llvm::SmallVectorImpl<std::string> &out) {
  std::size_t index = 0;
  while (index < text.size()) {
    if (!std::isalpha(static_cast<unsigned char>(text[index])) &&
        text[index] != '_') {
      ++index;
      continue;
    }
    std::size_t start = index;
    while (index < text.size() &&
           (std::isalnum(static_cast<unsigned char>(text[index])) ||
            text[index] == '_'))
      ++index;
    out.push_back(text.substr(start, index - start).str());
  }
}

// Every name an annotation mentions, in source order: a bare `T`, a `list[T]`
// subscript, and the quoted spelling of either.
void collectAnnotationNames(const parser::Node *annotation,
                            llvm::SmallVectorImpl<std::string> &out) {
  if (!annotation)
    return;
  if (annotation->kind == "Name") {
    out.push_back(std::string(ast::nameSpelling(*annotation)));
    return;
  }
  if (annotation->kind == "Constant") {
    if (std::optional<std::string_view> text = ast::string(*annotation, "value"))
      collectAnnotationIdentifiers(llvm::StringRef(*text), out);
    return;
  }
  for (const parser::Field &field : annotation->fields) {
    if (const auto *child = std::get_if<parser::NodePtr>(&field.value)) {
      if (*child)
        collectAnnotationNames(child->get(), out);
      continue;
    }
    if (const auto *children =
            std::get_if<std::vector<parser::NodePtr>>(&field.value))
      for (const parser::NodePtr &child : *children)
        collectAnnotationNames(child.get(), out);
  }
}

// The `TypeVar(name=...)` node PEP 695 puts in `type_params`.
parser::NodePtr typeParamNode(llvm::StringRef name, parser::SourceRange range) {
  parser::NodePtr node = parser::makeNode("TypeVar", range);
  parser::addField(*node, "name", std::string(name));
  parser::addField(*node, "bound", parser::NodePtr{});
  parser::addField(*node, "default_value", parser::NodePtr{});
  return node;
}

llvm::StringRef calleeLeafName(const parser::Node *callee) {
  if (!callee)
    return {};
  if (callee->kind == "Name")
    return ast::nameSpelling(*callee);
  if (callee->kind == "Attribute")
    if (std::optional<std::string_view> attr = ast::string(*callee, "attr"))
      return llvm::StringRef(*attr);
  return {};
}

} // namespace

// ⭐ THE CLASSIC GENERIC SPELLING IS THE PEP 695 ONE, one declaration back.
// `class Stack[T]` compiles; `T = TypeVar("T")` with `class Stack(Generic[T])`
// did not, and it failed by FABRICATING a contract -- `T` became
// `builtins.T`, so the error arrived as "print() cannot render argument of
// type builtins.T" for a program whose only unusual feature is that it is
// written the way every generic before Python 3.12 is written.
//
// The rewrite is the whole repair: the parameters move into `type_params`,
// the `Generic[...]` base is consumed, and the `TypeVar(...)` assignments go
// away. Everything downstream then sees the tree the new syntax parses to.
//
// ⛔ A FUNCTION GETS ITS OWN PARAMETERS ONLY WHERE NO ENCLOSING CLASS DECLARES
// THEM. Inside `class Stack(Generic[T])`, the `T` in `def push(self, v: T)` is
// the CLASS's parameter, and re-declaring it on the method would make a
// second, unrelated one.
void ModuleEmitter::desugarClassicGenerics(const parser::Node &moduleNode) {
  const auto *body = ast::nodeList(moduleNode, "body");
  if (!body)
    return;
  llvm::StringSet<> typeVars;
  for (const parser::NodePtr &statement : *body) {
    if (!statement || statement->kind != "Assign")
      continue;
    const auto *targets = ast::nodeList(*statement, "targets");
    if (!targets || targets->size() != 1 || !targets->front() ||
        targets->front()->kind != "Name")
      continue;
    const parser::Node *value = ast::node(*statement, "value");
    if (!value || value->kind != "Call")
      continue;
    llvm::StringRef callee = calleeLeafName(ast::node(*value, "func"));
    if (callee != "TypeVar")
      continue;
    typeVars.insert(ast::nameSpelling(*targets->front()));
  }
  if (typeVars.empty())
    return;

  // The parameters a def needs, in the order its annotations name them.
  auto declareFunctionParams = [&](const parser::Node &function,
                                   const llvm::StringSet<> &declared) {
    const auto *existing = ast::nodeList(function, "type_params");
    if (existing && !existing->empty())
      return;
    llvm::SmallVector<std::string, 4> mentioned;
    if (const parser::Node *args = ast::node(function, "args")) {
      for (llvm::StringRef group :
           {"posonlyargs", "args", "kwonlyargs"})
        if (const auto *list = ast::nodeList(*args, group))
          for (const parser::NodePtr &arg : *list)
            if (arg)
              collectAnnotationNames(ast::node(*arg, "annotation"), mentioned);
      for (llvm::StringRef group : {"vararg", "kwarg"})
        if (const parser::Node *arg = ast::node(*args, group))
          collectAnnotationNames(ast::node(*arg, "annotation"), mentioned);
    }
    collectAnnotationNames(ast::node(function, "returns"), mentioned);
    llvm::SmallVector<parser::NodePtr, 2> params;
    llvm::StringSet<> seen;
    for (const std::string &name : mentioned) {
      if (!typeVars.count(name) || declared.count(name) || !seen.insert(name).second)
        continue;
      params.push_back(typeParamNode(name, function.range));
    }
    if (params.empty())
      return;
    setField(const_cast<parser::Node &>(function), "type_params",
             std::vector<parser::NodePtr>(params.begin(), params.end()));
  };

  std::function<void(const parser::Node &, const llvm::StringSet<> &)> walk =
      [&](const parser::Node &statement, const llvm::StringSet<> &declared) {
        if (statement.kind == "FunctionDef" ||
            statement.kind == "AsyncFunctionDef") {
          declareFunctionParams(statement, declared);
          llvm::StringSet<> inner(declared);
          if (const auto *params = ast::nodeList(statement, "type_params"))
            for (const parser::NodePtr &param : *params)
              if (param)
                if (std::optional<std::string_view> name =
                        ast::string(*param, "name"))
                  inner.insert(*name);
          if (const auto *nested = ast::nodeList(statement, "body"))
            for (const parser::NodePtr &child : *nested)
              if (child)
                walk(*child, inner);
          return;
        }
        if (statement.kind != "ClassDef") {
          if (const auto *nested = ast::nodeList(statement, "body"))
            for (const parser::NodePtr &child : *nested)
              if (child)
                walk(*child, declared);
          return;
        }
        llvm::SmallVector<std::string, 2> classParams;
        std::vector<parser::NodePtr> keptBases;
        if (const auto *bases = ast::nodeList(statement, "bases"))
          for (const parser::NodePtr &base : *bases) {
            if (!base) {
              continue;
            }
            const parser::Node *owner =
                base->kind == "Subscript" ? ast::node(*base, "value") : nullptr;
            if (!owner || owner->kind != "Name" ||
                llvm::StringRef(ast::nameSpelling(*owner)) != "Generic") {
              keptBases.push_back(base);
              continue;
            }
            llvm::SmallVector<std::string, 2> named;
            collectAnnotationNames(ast::node(*base, "slice"), named);
            for (const std::string &name : named)
              if (typeVars.count(name) &&
                  !llvm::is_contained(classParams, name))
                classParams.push_back(name);
          }
        llvm::StringSet<> inner(declared);
        if (const auto *params = ast::nodeList(statement, "type_params"))
          for (const parser::NodePtr &param : *params)
            if (param)
              if (std::optional<std::string_view> name =
                      ast::string(*param, "name"))
                inner.insert(*name);
        if (!classParams.empty()) {
          llvm::SmallVector<parser::NodePtr, 2> params;
          for (const std::string &name : classParams) {
            params.push_back(typeParamNode(name, statement.range));
            inner.insert(name);
          }
          setField(const_cast<parser::Node &>(statement), "type_params",
                   std::vector<parser::NodePtr>(params.begin(), params.end()));
          setField(const_cast<parser::Node &>(statement), "bases",
                   std::move(keptBases));
        }
        if (const auto *nested = ast::nodeList(statement, "body"))
          for (const parser::NodePtr &child : *nested)
            if (child)
              walk(*child, inner);
      };

  llvm::StringSet<> empty;
  for (const parser::NodePtr &statement : *body)
    if (statement)
      walk(*statement, empty);

  // ⛔ AND THE `TypeVar(...)` CALLS GO. They are declarations, not values:
  // left in place they emit as a call to a name bound to `object`, which is
  // "static type builtins.object is not callable" -- the message the whole
  // program used to fail with.
  std::vector<parser::NodePtr> kept;
  for (const parser::NodePtr &statement : *body) {
    if (statement && statement->kind == "Assign") {
      const auto *targets = ast::nodeList(*statement, "targets");
      const parser::Node *value = ast::node(*statement, "value");
      if (targets && targets->size() == 1 && targets->front() &&
          targets->front()->kind == "Name" && value && value->kind == "Call" &&
          calleeLeafName(ast::node(*value, "func")) == "TypeVar" &&
          typeVars.count(ast::nameSpelling(*targets->front())))
        continue;
    }
    kept.push_back(statement);
  }
  setField(const_cast<parser::Node &>(moduleNode), "body", std::move(kept));
}

void ModuleEmitter::desugarEnumClasses(const parser::Node &moduleNode) {
  const auto *body = ast::nodeList(moduleNode, "body");
  if (!body)
    return;
  // Two passes over the module: collect the enum classes first so the use-site
  // rewrite (which walks every statement, including function bodies defined
  // before the class) sees the complete set.
  for (const parser::NodePtr &statement : *body) {
    if (!statement || statement->kind != "ClassDef")
      continue;
    if (std::optional<EnumKind> kind = enumBaseKind(*statement))
      collectEnumMembers(*statement, *kind);
  }
  if (enumClasses.empty())
    return;
  for (const parser::NodePtr &statement : *body)
    if (statement)
      rewriteEnumUses(*statement);
  for (const parser::NodePtr &statement : *body) {
    if (!statement || statement->kind != "ClassDef")
      continue;
    auto name = ast::string(*statement, "name");
    if (name && enumClasses.count(*name))
      rewriteEnumClassDef(*statement);
  }
}

void ModuleEmitter::collectEnumMembers(const parser::Node &classDef,
                                       EnumKind kind) {
  auto className = ast::string(classDef, "name");
  if (!className)
    return;
  bool requireUnique = false;
  if (const auto *decorators = ast::nodeList(classDef, "decorator_list"))
    for (const parser::NodePtr &decorator : *decorators)
      if (decorator && leafName(ast::qualifiedName(decorator.get())) == "unique")
        requireUnique = true;

  EnumInfo info;
  info.kind = kind;
  info.name = std::string(*className);
  std::int64_t autoCounter = 0;
  const auto *classBody = ast::nodeList(classDef, "body");
  if (!classBody)
    return;
  for (const parser::NodePtr &statement : *classBody) {
    if (!statement || statement->kind != "Assign")
      continue;
    const auto *targets = ast::nodeList(*statement, "targets");
    if (!targets || targets->size() != 1 || !targets->front() ||
        targets->front()->kind != "Name")
      continue;
    llvm::StringRef memberName = ast::nameSpelling(*targets->front());
    // CPython excludes dunder and sunder names from the member map.
    if (memberName.starts_with("__") || memberName.starts_with("_"))
      continue;
    const parser::Node *value = ast::node(*statement, "value");
    if (!value)
      continue;

    EnumMember member;
    member.name = std::string(memberName);
    bool valueResolved = false;
    if (value->kind == "Call" &&
        leafName(ast::qualifiedName(ast::node(*value, "func"))) == "auto") {
      // auto(): _generate_next_value_ — the next integer for Enum/IntEnum,
      // the lowercased member name for StrEnum (CPython 3.14).
      if (kind == EnumKind::Str) {
        member.strValue = llvm::StringRef(member.name).lower();
        member.isStr = true;
      } else {
        member.intValue = ++autoCounter;
      }
      valueResolved = true;
    } else if (value->kind == "Constant") {
      if (std::optional<std::int64_t> literal = ast::integer(*value, "value")) {
        member.intValue = *literal;
        autoCounter = *literal;
        valueResolved = true;
      } else if (auto text = ast::string(*value, "value")) {
        member.strValue = std::string(*text);
        member.isStr = true;
        valueResolved = true;
      }
    }
    if (!valueResolved) {
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, value->range.start,
          "enum member '" + member.name +
              "' needs an int literal, a str literal, or auto(): the members "
              "are instantiated at compile time"});
      continue;
    }
    if (kind == EnumKind::Int && member.isStr) {
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, value->range.start,
          "IntEnum member '" + member.name + "' needs an int value"});
      continue;
    }
    if (kind == EnumKind::Str && !member.isStr) {
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, value->range.start,
          "StrEnum member '" + member.name + "' needs a str value"});
      continue;
    }
    if (!info.members.empty() && info.members.front().isStr != member.isStr) {
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, value->range.start,
          "enum '" + info.name +
              "' mixes int and str member values; a single value type is "
              "required for the static member layout"});
      continue;
    }
    // An equal value makes this an alias of the earlier member (CPython
    // canonicalizes to the first definition); @unique rejects it.
    for (const EnumMember &earlier : info.members) {
      if (earlier.isAlias)
        continue;
      bool same = member.isStr ? earlier.strValue == member.strValue
                               : earlier.intValue == member.intValue;
      if (!same)
        continue;
      member.isAlias = true;
      member.aliasOf = earlier.name;
      break;
    }
    if (member.isAlias && requireUnique) {
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, statement->range.start,
          "duplicate values found in <enum '" + info.name + "'>: " +
              member.name + " -> " + member.aliasOf});
      continue;
    }
    info.members.push_back(std::move(member));
  }
  if (info.members.empty()) {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, classDef.range.start,
        "enum '" + info.name + "' has no members (an empty enum has no "
        "statically instantiable members)"});
    return;
  }
  enumClasses[info.name] = std::move(info);
}

void ModuleEmitter::rewriteEnumClassDef(const parser::Node &classDef) {
  auto className = ast::string(classDef, "name");
  if (!className)
    return;
  const EnumInfo &info = enumClasses[*className];
  parser::Node &mutableClassDef = const_cast<parser::Node &>(classDef);
  parser::SourceRange range = classDef.range;
  bool isStr = info.members.front().isStr;

  // Names the user defined explicitly win over the synthesized versions, as
  // in CPython's Enum (a member's __str__ override is honored).
  llvm::StringSet<> userMethods;
  std::vector<parser::NodePtr> keptStatements;
  if (const auto *classBody = ast::nodeList(classDef, "body"))
    for (const parser::NodePtr &statement : *classBody) {
      if (!statement)
        continue;
      if (statement->kind == "FunctionDef" ||
          statement->kind == "AsyncFunctionDef") {
        if (auto methodName = ast::string(*statement, "name"))
          userMethods.insert(*methodName);
        keptStatements.push_back(statement);
        continue;
      }
      // Member assignments are replaced by the annotated instantiations
      // below; everything else in an enum body (docstrings, nested classes)
      // survives untouched.
      if (statement->kind == "Assign") {
        const auto *targets = ast::nodeList(*statement, "targets");
        if (targets && targets->size() == 1 && targets->front() &&
            targets->front()->kind == "Name") {
          llvm::StringRef target = ast::nameSpelling(*targets->front());
          bool isMember = llvm::any_of(info.members,
                                       [&](const EnumMember &member) {
                                         return member.name == target;
                                       });
          if (isMember)
            continue;
        }
      }
      keptStatements.push_back(statement);
    }

  auto classAnnotation = [&] { return synth::name(info.name, range); };
  auto valueAnnotation = [&] {
    return synth::name(isStr ? "str" : "int", range);
  };
  auto memberLiteral = [&](const EnumMember &member) {
    return member.isStr ? synth::strConstant(member.strValue, range)
                        : synth::intConstant(member.intValue, range);
  };
  auto canonicalMemberOf = [&](const EnumMember &member) -> const EnumMember & {
    if (!member.isAlias)
      return member;
    for (const EnumMember &candidate : info.members)
      if (candidate.name == member.aliasOf)
        return candidate;
    return member;
  };

  std::vector<parser::NodePtr> synthesized;
  if (!userMethods.contains("__init__")) {
    std::vector<parser::NodePtr> body;
    body.push_back(
        synth::assign(synth::selfAttribute("self", "name", range), synth::name("name", range), range));
    body.push_back(
        synth::assign(synth::selfAttribute("self", "value", range), synth::name("value", range), range));
    synthesized.push_back(synth::functionDef(
        "__init__",
        {synth::Param{"self", nullptr},
         synth::Param{"name", synth::name("str", range)},
         synth::Param{"value", valueAnnotation()}}, {}, std::move(body),
        synth::name("None", range), llvm::ArrayRef<llvm::StringRef>{},
        range));
  }
  // Both display methods dispatch on the member name and return a literal:
  // every member's rendered text is known at compile time, so nothing has to
  // stringify the value at runtime. The name comparison is the only runtime
  // work, and the last member's text is the unconditional tail.
  // Aliases share the canonical member's singleton, so their name never
  // reaches these dispatches.
  llvm::SmallVector<const EnumMember *, 8> distinctMembers;
  for (const EnumMember &member : info.members)
    if (!member.isAlias)
      distinctMembers.push_back(&member);
  auto synthTextDispatch = [&](llvm::StringRef methodName,
                              llvm::function_ref<std::string(const EnumMember &)>
                                  textFor) {
    std::vector<parser::NodePtr> body;
    for (auto [index, member] : llvm::enumerate(distinctMembers)) {
      parser::NodePtr text = synth::strConstant(textFor(*member), range);
      if (index + 1 == distinctMembers.size()) {
        body.push_back(synth::returnStmt(std::move(text), range));
        break;
      }
      body.push_back(synth::ifStmt(
          synth::compare(synth::attribute(synth::name("self", range), "name", range), "Eq",
                  synth::strConstant(member->name, range), range), {synth::returnStmt(std::move(text), range)}, {}, range));
    }
    synthesized.push_back(synth::functionDef(methodName, {synth::Param{"self", nullptr}}, {}, std::move(body),
                                      synth::name("str", range), llvm::ArrayRef<llvm::StringRef>{}, range));
  };
  auto memberValueText = [&](const EnumMember &member) {
    return member.isStr ? "'" + member.strValue + "'"
                        : std::to_string(member.intValue);
  };
  if (!userMethods.contains("__str__")) {
    // Enum.__str__ is "Class.MEMBER"; IntEnum/StrEnum inherit the mixin's str
    // (the value's own text), which is what print and f-string interpolation
    // produce in CPython 3.14.
    synthTextDispatch("__str__", [&](const EnumMember &member) -> std::string {
      switch (info.kind) {
      case EnumKind::Plain:
        return info.name + "." + canonicalMemberOf(member).name;
      case EnumKind::Int:
        return std::to_string(member.intValue);
      case EnumKind::Str:
        return member.strValue;
      }
      return {};
    });
  }
  if (!userMethods.contains("__repr__"))
    synthTextDispatch("__repr__", [&](const EnumMember &member) {
      const EnumMember &canonical = canonicalMemberOf(member);
      return "<" + info.name + "." + canonical.name + ": " +
             memberValueText(canonical) + ">";
    });
  if (!userMethods.contains("__eq__")) {
    // Members are singletons, so value equality and identity coincide; the
    // typed `other` makes a cross-type comparison a diagnostic instead of
    // CPython's silent False.
    synthesized.push_back(synth::functionDef(
        "__eq__",
        {synth::Param{"self", nullptr}, synth::Param{"other", classAnnotation()}}, {},
        {synth::returnStmt(
             synth::compare(synth::attribute(synth::name("self", range), "value", range), "Eq",
                     synth::attribute(synth::name("other", range), "value", range),
                     range),
             range)},
        synth::name("bool", range), llvm::ArrayRef<llvm::StringRef>{}, range));
  }

  {
    std::vector<parser::NodePtr> body;
    for (const EnumMember *member : distinctMembers)
      body.push_back(synth::ifStmt(
          synth::compare(synth::name("value", range), "Eq", memberLiteral(*member), range), {synth::returnStmt(synth::attribute(synth::name(info.name, range), member->name, range),
          range)}, {},
          range));
    // DEVIATION (documented): CPython's message names the offending value
    // ("9 is not a valid Color"). Interpolating it here — in any spelling:
    // f-string, str(), repr(), % — leaves the value's box owned across the
    // message's may-unwind string construction, which the ownership verifier
    // rejects once the class carries its other synthesized methods. The
    // exception type and the enum name are preserved; the value is not.
    body.push_back(synth::raiseStmt(
        synth::call(synth::name("ValueError", range),
                  {synth::strConstant("not a valid " + info.name, range)}, range),
        range));
    synthesized.push_back(synth::functionDef(
        kFromValueMethod, {synth::Param{"value", valueAnnotation()}}, {}, std::move(body),
        classAnnotation(), llvm::ArrayRef<llvm::StringRef>{}, range));
  }
  {
    std::vector<parser::NodePtr> body;
    // Aliases resolve by name too (CPython's `E["ALIAS"]` yields the
    // canonical member), so every declared name gets a branch.
    for (const EnumMember &member : info.members)
      body.push_back(synth::ifStmt(
          synth::compare(synth::name("name", range), "Eq", synth::strConstant(member.name, range),
                  range), {synth::returnStmt(synth::attribute(synth::name(info.name, range),
                         canonicalMemberOf(member).name, range),
          range)}, {},
          range));
    // The interpolation is what makes the raised key an owned string: handing
    // the borrowed parameter straight to KeyError would transfer a borrow.
    body.push_back(synth::raiseStmt(
        synth::call(synth::name("KeyError", range),
                  {synthFormattedMessage(synth::name("name", range), "", range)},
                  range),
        range));
    synthesized.push_back(synth::functionDef(
        kFromNameMethod, {synth::Param{"name", synth::name("str", range)}}, {}, std::move(body),
        classAnnotation(), llvm::ArrayRef<llvm::StringRef>{}, range));
  }

  // The member attributes come last: their initializers run at the ClassDef
  // statement position (after the class contract exists), and an alias reads
  // the canonical member's already-initialized slot.
  std::vector<parser::NodePtr> memberAttrs;
  for (const EnumMember &member : info.members) {
    parser::NodePtr value;
    if (member.isAlias) {
      value = synth::attribute(synth::name(info.name, range),
                             canonicalMemberOf(member).name, range);
    } else {
      value = synth::call(synth::name(info.name, range),
                        {synth::strConstant(member.name, range), memberLiteral(member)},
                        range);
    }
    memberAttrs.push_back(
        synth::annAssign(synth::name(member.name,
                       range), classAnnotation(), std::move(value),
                       range));
  }

  std::vector<parser::NodePtr> newBody;
  newBody.insert(newBody.end(), keptStatements.begin(), keptStatements.end());
  newBody.insert(newBody.end(), synthesized.begin(), synthesized.end());
  newBody.insert(newBody.end(), memberAttrs.begin(), memberAttrs.end());
  setField(mutableClassDef, "body", std::move(newBody));
  // The Enum base and the @unique marker are consumed by the desugar: what
  // remains is a plain class.
  setField(mutableClassDef, "bases", std::vector<parser::NodePtr>{});
  setField(mutableClassDef, "decorator_list", std::vector<parser::NodePtr>{});
}

parser::NodePtr
ModuleEmitter::enumMemberListNode(const EnumInfo &info,
                                  parser::SourceRange range) const {
  std::vector<parser::NodePtr> elements;
  for (const EnumMember &member : info.members) {
    if (member.isAlias)
      continue;
    elements.push_back(
        synth::attribute(synth::name(info.name, range), member.name, range));
  }
  parser::NodePtr list = parser::makeNode("List", range);
  parser::addField(*list, "elts", std::move(elements));
  return list;
}

const ModuleEmitter::EnumInfo *
ModuleEmitter::enumInfoForNameNode(const parser::Node *node) const {
  if (!node || node->kind != "Name")
    return nullptr;
  auto found = enumClasses.find(ast::nameSpelling(*node));
  return found == enumClasses.end() ? nullptr : &found->second;
}

void ModuleEmitter::rewriteEnumUses(const parser::Node &node) {
  parser::Node &mutableNode = const_cast<parser::Node &>(node);

  // `E(value)` is CPython's by-value lookup, not construction; `E["NAME"]` is
  // the by-name lookup. Both become calls to the synthesized classmethods.
  if (node.kind == "Call") {
    const parser::Node *callee = ast::node(node, "func");
    if (const EnumInfo *info = enumInfoForNameNode(callee)) {
      const auto *args = ast::nodeList(node, "args");
      const auto *keywords = ast::nodeList(node, "keywords");
      if (args && args->size() == 1 && (!keywords || keywords->empty())) {
        setField(mutableNode, "func",
                 synth::attribute(synth::name(info->name, node.range),
                                kFromValueMethod, node.range));
      } else {
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, node.range.start,
            "enum '" + info->name +
                "' takes exactly one positional argument (the member value); "
                "members are declared in the class body"});
      }
    }
    // A bare enum class in a consuming builtin's argument position iterates
    // its members (CPython's EnumType.__iter__).
    if (const auto *args = ast::nodeList(node, "args");
        args && args->size() == 1 && args->front()) {
      // Re-fetched, not reused: the by-value rewrite above replaced the "func"
      // field, which dropped the last reference to the node `callee` named.
      const std::string qualified = ast::qualifiedName(ast::node(node, "func"));
      llvm::StringRef consumer = leafName(qualified);
      bool iterates = consumer == "list" || consumer == "tuple" ||
                      consumer == "set" || consumer == "sorted" ||
                      consumer == "len" || consumer == "reversed" ||
                      consumer == "iter";
      if (iterates)
        if (const EnumInfo *info = enumInfoForNameNode(args->front().get()))
          setField(mutableNode, "args",
                   std::vector<parser::NodePtr>{
                       enumMemberListNode(*info, node.range)});
    }
  } else if (node.kind == "Subscript") {
    if (const EnumInfo *info = enumInfoForNameNode(ast::node(node, "value"))) {
      const parser::Node *index = ast::node(node, "slice");
      if (index && index->kind != "Slice") {
        parser::NodePtr indexNode;
        if (const parser::Field *field = parser::findField(node, "slice"))
          if (const auto *ptr = std::get_if<parser::NodePtr>(&field->value))
            indexNode = *ptr;
        if (indexNode) {
          parser::NodePtr call = synth::call(
              synth::attribute(synth::name(info->name, node.range),
                             kFromNameMethod, node.range),
              {indexNode}, node.range);
          // The Subscript node is referenced by its parent, so it is rewritten
          // into the call in place rather than replaced.
          mutableNode.kind = "Call";
          mutableNode.fields.clear();
          mutableNode.fieldIndicesBySlot.clear();
          for (parser::Field &field : call->fields)
            parser::addField(mutableNode, field.name, std::move(field.value));
        }
      }
    }
  } else if (node.kind == "For" || node.kind == "AsyncFor" ||
             node.kind == "comprehension") {
    if (const EnumInfo *info = enumInfoForNameNode(ast::node(node, "iter")))
      setField(mutableNode, "iter", enumMemberListNode(*info, node.range));
  }

  for (parser::Field &field : mutableNode.fields) {
    if (auto *child = std::get_if<parser::NodePtr>(&field.value)) {
      if (*child)
        rewriteEnumUses(**child);
      continue;
    }
    if (auto *children = std::get_if<std::vector<parser::NodePtr>>(&field.value))
      for (const parser::NodePtr &child : *children)
        if (child)
          rewriteEnumUses(*child);
  }
}

} // namespace lython::emitter
