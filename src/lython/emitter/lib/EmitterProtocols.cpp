#include "AstSynth.h"
#include "EmitterCore.h"

#include "AstAccess.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"

#include <functional>

namespace lython::emitter {

// `class Shape(Protocol)` is a structural type: every class with the members
// it names IS a Shape, whether or not it says so. Here it becomes an ordinary
// class, and each class that has its members gets it as one more base -- the
// relation PEP 544 states, written where this compiler reads relations.
//
// What that buys is everything a base already has: a Shape-typed value is an
// instance of a class (counted, dispatched through `__lyvdisp`, stored in a
// slot as itself), so nothing downstream learns the word "protocol".
//
// ⛔ Not a `!py.protocol` type. A value of protocol type has no runtime
// contract, so ownership calls it NonObject and takes no references
// (lython-protocol-values-uncounted): a Shape read out of a list would be a
// use-after-free. A base class has none of that.
//
// ⛔ Not decided at the conversion site (`one(Square(2))`). A dispatcher is
// memoized at its first use and enumerates the classes that implement the
// member THEN; a conformance learned later would leave its class out, and the
// call would run the protocol's stub. The relation is fixed before anything
// is declared, so the class set every dispatcher sees is the whole program's.

namespace {

void setNodeField(parser::Node &node, std::string name,
                  parser::FieldValue value) {
  if (parser::Field *existing = parser::findField(node, name)) {
    existing->value = std::move(value);
    return;
  }
  parser::addField(node, std::move(name), std::move(value));
}

struct MemberShape {
  bool isMethod = false;
  // Positional parameters after `self`, and how many of them have no default.
  unsigned positional = 0;
  unsigned required = 0;
  bool variadic = false;
};

MemberShape methodShape(const parser::Node &def) {
  MemberShape shape;
  shape.isMethod = true;
  const parser::Node *args = ast::node(def, "args");
  if (!args)
    return shape;
  unsigned total = 0;
  for (llvm::StringRef group : {"posonlyargs", "args"})
    if (const auto *list = ast::nodeList(*args, group))
      total += list->size();
  unsigned defaults = 0;
  if (const auto *list = ast::nodeList(*args, "defaults"))
    defaults = list->size();
  bool bound = true;
  if (const auto *decorators = ast::nodeList(def, "decorator_list"))
    for (const parser::NodePtr &decorator : *decorators)
      if (decorator && decorator->kind == "Name" &&
          ast::nameSpelling(*decorator) == "staticmethod")
        bound = false;
  unsigned self = bound && total > 0 ? 1 : 0;
  shape.positional = total - self;
  shape.required = total > defaults ? total - defaults - self : 0;
  shape.variadic = ast::node(*args, "vararg") != nullptr;
  return shape;
}

bool isPropertyDef(const parser::Node &def) {
  if (const auto *decorators = ast::nodeList(def, "decorator_list"))
    for (const parser::NodePtr &decorator : *decorators)
      if (decorator && decorator->kind == "Name" &&
          ast::nameSpelling(*decorator) == "property")
        return true;
  return false;
}

// `self.<name> = ...` anywhere in a method body: an instance field.
void collectSelfFields(const parser::Node &node, llvm::StringSet<> &out) {
  auto record = [&](const parser::Node *target) {
    if (target && target->kind == "Attribute")
      if (const parser::Node *owner = ast::node(*target, "value"))
        if (owner->kind == "Name" && ast::nameSpelling(*owner) == "self")
          if (auto attr = ast::string(*target, "attr"))
            out.insert(*attr);
  };
  if (node.kind == "Assign") {
    if (const auto *targets = ast::nodeList(node, "targets"))
      for (const parser::NodePtr &target : *targets)
        record(target.get());
  } else if (node.kind == "AnnAssign" || node.kind == "AugAssign") {
    record(ast::node(node, "target"));
  }
  for (llvm::StringRef group : {"body", "orelse", "finalbody", "handlers"})
    if (const auto *children = ast::nodeList(node, group))
      for (const parser::NodePtr &child : *children)
        if (child && child->kind != "FunctionDef" &&
            child->kind != "AsyncFunctionDef" && child->kind != "ClassDef")
          collectSelfFields(*child, out);
}

} // namespace

void ModuleEmitter::desugarProtocols() {
  struct ModuleView {
    std::string name;
    const parser::Node *node = nullptr;
  };
  llvm::SmallVector<ModuleView, 8> modules;
  for (const EmitOptions::SourceModule &source : options.sourceModules)
    if (source.moduleNode && !source.isStub)
      modules.push_back({source.moduleName, source.moduleNode});
  modules.push_back({moduleName, &moduleNode});

  // The local names a module binds `typing.Protocol` to, and the module
  // aliases that reach `typing`.
  auto protocolSpellings = [](const parser::Node &module,
                              llvm::StringSet<> &names,
                              llvm::StringSet<> &typingModules) {
    if (const auto *body = ast::nodeList(module, "body"))
      for (const parser::NodePtr &statement : *body) {
        if (!statement)
          continue;
        if (statement->kind == "ImportFrom") {
          auto from = ast::string(*statement, "module");
          if (!from || (*from != "typing" && *from != "typing_extensions"))
            continue;
          if (const auto *aliases = ast::nodeList(*statement, "names"))
            for (const parser::NodePtr &alias : *aliases)
              if (alias && ast::string(*alias, "name").value_or("") ==
                               "Protocol")
                names.insert(ast::string(*alias, "asname").value_or(
                    "Protocol"));
        } else if (statement->kind == "Import") {
          if (const auto *aliases = ast::nodeList(*statement, "names"))
            for (const parser::NodePtr &alias : *aliases) {
              if (!alias)
                continue;
              auto imported = ast::string(*alias, "name").value_or("");
              if (imported == "typing" || imported == "typing_extensions")
                typingModules.insert(
                    ast::string(*alias, "asname").value_or(imported));
            }
        }
      }
  };

  struct ClassView {
    unsigned module = 0;
    std::string name;
    parser::Node *node = nullptr;
    bool isProtocol = false;
    // Bases that are classes of this program, as (module, name).
    llvm::SmallVector<std::pair<unsigned, std::string>, 2> sourceBases;
    // A base this program does not define (a builtin, an exception, Enum):
    // such a class keeps the bases it wrote.
    bool foreignBase = false;
    llvm::StringMap<MemberShape> own;
  };
  std::vector<ClassView> classes;
  llvm::StringMap<unsigned> classIndex; // "module\0name" -> index
  auto key = [](unsigned module, llvm::StringRef name) {
    return (llvm::Twine(module) + ":" + name).str();
  };
  auto moduleIndexOf = [&](llvm::StringRef name) -> std::optional<unsigned> {
    for (auto [index, view] : llvm::enumerate(modules))
      if (view.name == name)
        return static_cast<unsigned>(index);
    return std::nullopt;
  };

  for (auto [moduleIndex, view] : llvm::enumerate(modules)) {
    const auto *body = ast::nodeList(*view.node, "body");
    if (!body)
      continue;
    for (const parser::NodePtr &statement : *body) {
      if (!statement || statement->kind != "ClassDef")
        continue;
      auto name = ast::string(*statement, "name");
      if (!name)
        continue;
      ClassView entry;
      entry.module = static_cast<unsigned>(moduleIndex);
      entry.name = std::string(*name);
      entry.node = const_cast<parser::Node *>(statement.get());
      classIndex[key(entry.module, entry.name)] = classes.size();
      classes.push_back(std::move(entry));
    }
  }

  // A base expression, resolved to a class of this program when it is one:
  // a bare name of the same module, a name imported from another, or
  // `module.Name`.
  auto resolveBase = [&](unsigned module, const parser::Node &base)
      -> std::optional<unsigned> {
    auto lookupIn = [&](unsigned owner,
                        llvm::StringRef name) -> std::optional<unsigned> {
      auto found = classIndex.find(key(owner, name));
      if (found == classIndex.end())
        return std::nullopt;
      return found->second;
    };
    const auto *body = ast::nodeList(*modules[module].node, "body");
    if (base.kind == "Name") {
      llvm::StringRef spelled = ast::nameSpelling(base);
      if (std::optional<unsigned> local = lookupIn(module, spelled))
        return local;
      if (body)
        for (const parser::NodePtr &statement : *body) {
          if (!statement || statement->kind != "ImportFrom")
            continue;
          auto from = ast::string(*statement, "module");
          std::optional<unsigned> source =
              from ? moduleIndexOf(*from) : std::nullopt;
          if (!source)
            continue;
          if (const auto *aliases = ast::nodeList(*statement, "names"))
            for (const parser::NodePtr &alias : *aliases) {
              if (!alias)
                continue;
              auto imported = ast::string(*alias, "name").value_or("");
              if (llvm::StringRef(ast::string(*alias, "asname").value_or(imported)) == spelled)
                return lookupIn(*source, imported);
            }
        }
      return std::nullopt;
    }
    if (base.kind == "Attribute") {
      const parser::Node *owner = ast::node(base, "value");
      auto attr = ast::string(base, "attr");
      if (!owner || !attr)
        return std::nullopt;
      std::string spelledModule = ast::qualifiedName(owner);
      if (body)
        for (const parser::NodePtr &statement : *body) {
          if (!statement || statement->kind != "Import")
            continue;
          if (const auto *aliases = ast::nodeList(*statement, "names"))
            for (const parser::NodePtr &alias : *aliases) {
              if (!alias)
                continue;
              auto imported = ast::string(*alias, "name").value_or("");
              if (ast::string(*alias, "asname").value_or(imported) ==
                  spelledModule)
                if (std::optional<unsigned> source = moduleIndexOf(imported))
                  return lookupIn(*source, *attr);
            }
        }
    }
    return std::nullopt;
  };

  // Which classes are protocols, and the bases of every class.
  for (ClassView &entry : classes) {
    llvm::StringSet<> protocolNames, typingModules;
    protocolSpellings(*modules[entry.module].node, protocolNames,
                      typingModules);
    auto isProtocolMarker = [&](const parser::Node *base) {
      if (!base)
        return false;
      if (base->kind == "Name")
        return protocolNames.count(ast::nameSpelling(*base)) > 0;
      if (base->kind == "Attribute")
        if (const parser::Node *owner = ast::node(*base, "value"))
          return owner->kind == "Name" &&
                 typingModules.count(ast::nameSpelling(*owner)) &&
                 ast::string(*base, "attr").value_or("") == "Protocol";
      return false;
    };
    std::vector<parser::NodePtr> kept;
    if (const auto *bases = ast::nodeList(*entry.node, "bases"))
      for (const parser::NodePtr &base : *bases) {
        if (!base)
          continue;
        if (base->kind == "Subscript" &&
            isProtocolMarker(ast::node(*base, "value"))) {
          diagnostics.push_back(parser::Diagnostic{
              parser::Severity::Error, base->range.start,
              "a generic Protocol (`Protocol[T]`) is not supported yet: a "
              "class that satisfies it would need its type arguments "
              "inferred from its members"});
          entry.isProtocol = true;
          continue;
        }
        if (isProtocolMarker(base.get())) {
          entry.isProtocol = true;
          continue;
        }
        kept.push_back(base);
      }
    if (entry.isProtocol) {
      if (const auto *params = ast::nodeList(*entry.node, "type_params");
          params && !params->empty())
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, entry.node->range.start,
            "a generic Protocol (`class " + entry.name +
                "[T](Protocol)`) is not supported yet: a class that satisfies "
                "it would need its type arguments inferred from its members"});
      setNodeField(*entry.node, "bases", std::move(kept));
      protocolClassNames.insert(entry.name);
      // ⭐ A PROPERTY MEMBER IS DATA TO EVERY CLASS THAT SATISFIES IT, and a
      // class can satisfy it with a field, a class attribute or a property.
      // Left a property, it was a base property with no setter, so a
      // satisfying class's own `self.kind = ...` was refused ("property 'kind'
      // has no setter"). As `kind: str` it is the field a field satisfies,
      // and a class attribute or property is read through the field dispatch.
      if (const auto *body = ast::nodeList(*entry.node, "body")) {
        std::vector<parser::NodePtr> rewritten;
        llvm::StringSet<> declared;
        for (const parser::NodePtr &member : *body) {
          if (member && member->kind == "FunctionDef") {
            bool getter = false, setter = false;
            if (const auto *decorators =
                    ast::nodeList(*member, "decorator_list"))
              for (const parser::NodePtr &decorator : *decorators) {
                if (decorator && decorator->kind == "Name" &&
                    ast::nameSpelling(*decorator) == "property")
                  getter = true;
                if (decorator && decorator->kind == "Attribute" &&
                    ast::string(*decorator, "attr").value_or("") == "setter")
                  setter = true;
              }
            auto name = ast::string(*member, "name");
            const parser::Field *returns = parser::findField(*member, "returns");
            const parser::NodePtr *annotation =
                returns ? std::get_if<parser::NodePtr>(&returns->value)
                        : nullptr;
            if (setter && name && declared.count(*name))
              continue;
            if (getter && name && annotation && *annotation) {
              declared.insert(*name);
              rewritten.push_back(synth::annAssign(
                  synth::name(*name, member->range), *annotation, nullptr,
                  member->range));
              continue;
            }
          }
          rewritten.push_back(member);
        }
        setNodeField(*entry.node, "body", std::move(rewritten));
      }
    }
    if (const auto *bases = ast::nodeList(*entry.node, "bases"))
      for (const parser::NodePtr &base : *bases) {
        if (!base)
          continue;
        if (std::optional<unsigned> resolved =
                resolveBase(entry.module, *base))
          entry.sourceBases.push_back(
              {classes[*resolved].module, classes[*resolved].name});
        else if (!(base->kind == "Name" &&
                   ast::nameSpelling(*base) == "object"))
          entry.foreignBase = true;
      }
    if (const auto *keywords = ast::nodeList(*entry.node, "keywords");
        keywords && !keywords->empty())
      entry.foreignBase = true;
    // Its own members: methods, class-level names, and `self.x` fields.
    llvm::StringSet<> fields;
    if (const auto *body = ast::nodeList(*entry.node, "body"))
      for (const parser::NodePtr &statement : *body) {
        if (!statement)
          continue;
        if (statement->kind == "FunctionDef" ||
            statement->kind == "AsyncFunctionDef") {
          auto name = ast::string(*statement, "name");
          if (!name)
            continue;
          MemberShape shape = isPropertyDef(*statement) ? MemberShape{}
                                                        : methodShape(*statement);
          // A setter re-declares the property's name; the getter decides.
          if (!entry.own.count(*name))
            entry.own[*name] = shape;
          collectSelfFields(*statement, fields);
        } else if (statement->kind == "AnnAssign") {
          if (const parser::Node *target = ast::node(*statement, "target");
              target && target->kind == "Name")
            entry.own[ast::nameSpelling(*target)] = MemberShape{};
        } else if (statement->kind == "Assign") {
          if (const auto *targets = ast::nodeList(*statement, "targets"))
            for (const parser::NodePtr &target : *targets)
              if (target && target->kind == "Name")
                entry.own[ast::nameSpelling(*target)] = MemberShape{};
        }
      }
    for (const auto &field : fields)
      if (!entry.own.count(field.getKey()))
        entry.own[field.getKey()] = MemberShape{};
  }
  if (protocolClassNames.empty())
    return;

  // Members with what each class inherits from the program's own bases.
  std::vector<std::optional<llvm::StringMap<MemberShape>>> allMembers(
      classes.size());
  std::function<const llvm::StringMap<MemberShape> &(unsigned)> membersOf =
      [&](unsigned index) -> const llvm::StringMap<MemberShape> & {
    if (allMembers[index])
      return *allMembers[index];
    allMembers[index].emplace();
    llvm::StringMap<MemberShape> merged;
    // Bases first, so the class's own definition wins.
    for (const auto &[baseModule, baseName] : classes[index].sourceBases) {
      auto found = classIndex.find(key(baseModule, baseName));
      if (found == classIndex.end())
        continue;
      for (const auto &member : membersOf(found->second))
        merged[member.getKey()] = member.getValue();
    }
    for (const auto &member : classes[index].own)
      merged[member.getKey()] = member.getValue();
    allMembers[index] = std::move(merged);
    return *allMembers[index];
  };
  std::function<bool(unsigned, unsigned)> derivesFrom =
      [&](unsigned index, unsigned ancestor) -> bool {
    for (const auto &[baseModule, baseName] : classes[index].sourceBases) {
      auto found = classIndex.find(key(baseModule, baseName));
      if (found == classIndex.end())
        continue;
      if (found->second == ancestor || derivesFrom(found->second, ancestor))
        return true;
    }
    return false;
  };

  // A class's contract name: bare in the main module, qualified in another.
  auto contractOf = [&](const ClassView &entry) {
    if (entry.module + 1 == modules.size())
      return entry.name;
    return modules[entry.module].name + "." + entry.name;
  };
  auto recordMiss = [&](unsigned candidate, unsigned protocol,
                        std::string why) {
    protocolMisses[contractOf(classes[candidate]) + '\0' +
                   contractOf(classes[protocol])] = std::move(why);
  };
  auto conforms = [&](unsigned candidate, unsigned protocol) {
    const llvm::StringMap<MemberShape> &have = membersOf(candidate);
    const llvm::StringMap<MemberShape> &want = membersOf(protocol);
    if (want.empty())
      return false;
    for (const auto &member : want) {
      std::string name = member.getKey().str();
      auto found = have.find(name);
      if (found == have.end()) {
        recordMiss(candidate, protocol, "it has no member '" + name + "'");
        return false;
      }
      const MemberShape &needed = member.getValue();
      const MemberShape &given = found->getValue();
      if (!needed.isMethod)
        continue;
      if (!given.isMethod) {
        recordMiss(candidate, protocol,
                   "its '" + name + "' is not a method");
        return false;
      }
      // It must take every call the protocol's signature allows.
      if (given.required > needed.positional ||
          (!given.variadic && given.positional < needed.positional)) {
        recordMiss(candidate, protocol,
                   "its '" + name +
                       "' does not take the parameters the protocol's does");
        return false;
      }
    }
    return true;
  };

  // How a class's module spells a protocol, if it can: the same module, a
  // `from m import P`, or an `import m`.
  auto spellingIn = [&](unsigned module,
                        const ClassView &protocol) -> parser::NodePtr {
    parser::SourceRange range = protocol.node->range;
    if (module == protocol.module)
      return synth::name(protocol.name, range);
    const auto *body = ast::nodeList(*modules[module].node, "body");
    if (!body)
      return nullptr;
    llvm::StringRef protocolModule = modules[protocol.module].name;
    for (const parser::NodePtr &statement : *body) {
      if (!statement)
        continue;
      if (statement->kind == "ImportFrom" &&
          llvm::StringRef(ast::string(*statement, "module").value_or("")) == protocolModule) {
        if (const auto *aliases = ast::nodeList(*statement, "names"))
          for (const parser::NodePtr &alias : *aliases)
            if (alias &&
                ast::string(*alias, "name").value_or("") == protocol.name)
              return synth::name(
                  ast::string(*alias, "asname").value_or(protocol.name), range);
      } else if (statement->kind == "Import") {
        if (const auto *aliases = ast::nodeList(*statement, "names"))
          for (const parser::NodePtr &alias : *aliases)
            if (alias &&
                llvm::StringRef(ast::string(*alias, "name").value_or("")) == protocolModule &&
                !protocolModule.contains('.'))
              return synth::attribute(
                  synth::name(ast::string(*alias, "asname")
                                  .value_or(std::string(protocolModule)),
                              range),
                  protocol.name, range);
      }
    }
    return nullptr;
  };

  // Every (class, protocol) the class satisfies, keeping only what is not
  // already said: by its own bases (written, or given here to a base), or by
  // a more derived protocol it also gets.
  llvm::SmallVector<unsigned, 8> protocols;
  for (auto [index, entry] : llvm::enumerate(classes))
    if (entry.isProtocol)
      protocols.push_back(static_cast<unsigned>(index));
  std::vector<llvm::SmallVector<unsigned, 2>> satisfied(classes.size());
  for (auto [index, entry] : llvm::enumerate(classes)) {
    if (entry.isProtocol || entry.foreignBase ||
        enumClasses.count(entry.name)) {
      for (unsigned protocol : protocols)
        if (!entry.isProtocol)
          recordMiss(static_cast<unsigned>(index), protocol,
                     "a class with a base this program does not define (a "
                     "builtin, an exception, an Enum) is not matched against "
                     "protocols yet");
      continue;
    }
    if (const auto *params = ast::nodeList(*entry.node, "type_params");
        params && !params->empty())
      continue;
    for (unsigned protocol : protocols)
      if (!derivesFrom(index, protocol) && conforms(index, protocol))
        satisfied[index].push_back(protocol);
  }
  for (auto [index, entry] : llvm::enumerate(classes)) {
    llvm::SmallVector<unsigned, 2> &mine = satisfied[index];
    if (mine.empty())
      continue;
    std::function<bool(unsigned, unsigned)> baseSatisfies =
        [&](unsigned cls, unsigned protocol) -> bool {
      for (const auto &[baseModule, baseName] : classes[cls].sourceBases) {
        auto found = classIndex.find(key(baseModule, baseName));
        if (found == classIndex.end())
          continue;
        if (llvm::is_contained(satisfied[found->second], protocol) ||
            baseSatisfies(found->second, protocol))
          return true;
      }
      return false;
    };
    std::vector<parser::NodePtr> bases;
    if (const auto *existing = ast::nodeList(*entry.node, "bases"))
      bases = *existing;
    bool changed = false;
    for (unsigned protocol : mine) {
      if (baseSatisfies(static_cast<unsigned>(index), protocol))
        continue;
      bool subsumed = false;
      for (unsigned other : mine)
        if (other != protocol && derivesFrom(other, protocol))
          subsumed = true;
      if (subsumed)
        continue;
      parser::NodePtr spelled = spellingIn(entry.module, classes[protocol]);
      if (!spelled) {
        recordMiss(static_cast<unsigned>(index), protocol,
                   "it has the members, but its module does not import '" +
                       classes[protocol].name +
                       "', and a class satisfies a protocol here only where "
                       "it can name it");
        continue;
      }
      bases.push_back(std::move(spelled));
      changed = true;
    }
    if (changed)
      setNodeField(*entry.node, "bases", std::move(bases));
  }

  // What isinstance() may answer for each protocol: CPython's check needs
  // @runtime_checkable, looks at METHOD names only, and is True for a class
  // with those names whether or not its parameters fit.
  for (unsigned protocol : protocols) {
    const ClassView &view = classes[protocol];
    std::string why;
    bool checkable = false;
    if (const auto *decorators = ast::nodeList(*view.node, "decorator_list"))
      for (const parser::NodePtr &decorator : *decorators)
        if (decorator &&
            ((decorator->kind == "Name" &&
              ast::nameSpelling(*decorator) == "runtime_checkable") ||
             (decorator->kind == "Attribute" &&
              ast::string(*decorator, "attr").value_or("") ==
                  "runtime_checkable")))
          checkable = true;
    const llvm::StringMap<MemberShape> &want = membersOf(protocol);
    if (!checkable)
      why = "it is not @runtime_checkable (CPython raises TypeError)";
    for (const auto &member : want)
      if (why.empty() && !member.getValue().isMethod)
        why = "its member '" + member.getKey().str() +
              "' is data, which CPython looks for on the instance at run time";
    if (why.empty())
      for (auto [index, entry] : llvm::enumerate(classes)) {
        if (entry.isProtocol || index == protocol)
          continue;
        bool namesMatch = true;
        const llvm::StringMap<MemberShape> &have =
            membersOf(static_cast<unsigned>(index));
        for (const auto &member : want)
          if (!have.count(member.getKey()))
            namesMatch = false;
        if (!namesMatch)
          continue;
        bool based = derivesFrom(static_cast<unsigned>(index), protocol);
        for (unsigned other : protocols)
          if (llvm::is_contained(satisfied[index], other) &&
              (other == protocol || derivesFrom(other, protocol)))
            based = true;
        if (!based) {
          why = "'" + entry.name +
                "' has its method names, which CPython's check accepts, but "
                "not its signatures (or cannot name it), so this answer "
                "would differ";
          break;
        }
      }
    protocolIsinstance[contractOf(view)] = why;
  }
}

} // namespace lython::emitter
