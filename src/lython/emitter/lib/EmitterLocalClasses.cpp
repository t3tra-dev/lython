#include "EmitterCore.h"

#include "AstAccess.h"
#include "ClosureAnalysis.h"
#include "Contracts.h"
#include "Parser.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"

#include <functional>

namespace lython::emitter {
namespace {

bool isFunctionScope(const parser::Node &node) {
  return node.kind == "FunctionDef" || node.kind == "AsyncFunctionDef";
}

bool isComprehension(const parser::Node &node) {
  return node.kind == "ListComp" || node.kind == "SetComp" ||
         node.kind == "DictComp" || node.kind == "GeneratorExp";
}

// The Name spellings a binding target writes: `a`, `a, (b, *c)`.
void collectTargetNames(const parser::Node *target,
                        llvm::SmallVectorImpl<std::string> &names) {
  if (!target)
    return;
  if (target->kind == "Name") {
    names.push_back(std::string(ast::nameSpelling(*target)));
    return;
  }
  if (target->kind == "Starred") {
    collectTargetNames(ast::node(*target, "value"), names);
    return;
  }
  if (target->kind == "Tuple" || target->kind == "List")
    if (const auto *elements = ast::nodeList(*target, "elts"))
      for (const parser::NodePtr &element : *elements)
        collectTargetNames(element.get(), names);
}

// The names a def or a lambda's own body declares `global` / `nonlocal`.
void collectDeclarations(const parser::Node *node, llvm::StringRef kind,
                         llvm::StringSet<> &names) {
  if (!node)
    return;
  if (node->kind == kind) {
    if (const auto *declared = ast::stringList(*node, "names"))
      for (const std::string &name : *declared)
        names.insert(name);
    return;
  }
  if (isFunctionScope(*node) || node->kind == "ClassDef" ||
      node->kind == "Lambda")
    return;
  for (const parser::Field &field : node->fields) {
    if (const auto *child = std::get_if<parser::NodePtr>(&field.value))
      collectDeclarations(child->get(), kind, names);
    else if (const auto *children =
                 std::get_if<std::vector<parser::NodePtr>>(&field.value))
      for (const parser::NodePtr &element : *children)
        collectDeclarations(element.get(), kind, names);
  }
}

// The names a def nested in `function` -- at any depth -- declares
// `nonlocal`: each may rebind a name of `function`'s.
void collectNestedNonlocals(const parser::Node *node, bool nested,
                            llvm::StringSet<> &names) {
  if (!node)
    return;
  if (nested && node->kind == "Nonlocal")
    if (const auto *declared = ast::stringList(*node, "names"))
      for (const std::string &name : *declared)
        names.insert(name);
  bool scope = isFunctionScope(*node) || node->kind == "Lambda";
  for (const parser::Field &field : node->fields) {
    if (const auto *child = std::get_if<parser::NodePtr>(&field.value))
      collectNestedNonlocals(child->get(), nested || scope, names);
    else if (const auto *children =
                 std::get_if<std::vector<parser::NodePtr>>(&field.value))
      for (const parser::NodePtr &element : *children)
        collectNestedNonlocals(element.get(), nested || scope, names);
  }
}

// Does `scope`'s body name something of its own by `name`, so that it does
// not see the binding around it? A def that declares `global name` reads the
// module's; one that declares `nonlocal name` reads the enclosing one.
bool scopeBindsOwn(const parser::Node &scope, llvm::StringRef name) {
  llvm::StringSet<> globals;
  llvm::StringSet<> nonlocals;
  if (const auto *body = ast::nodeList(scope, "body"))
    for (const parser::NodePtr &statement : *body) {
      collectDeclarations(statement.get(), "Global", globals);
      collectDeclarations(statement.get(), "Nonlocal", nonlocals);
    }
  if (globals.contains(name))
    return true;
  if (nonlocals.contains(name))
    return false;
  return functionLocalNames(scope).contains(name);
}

bool lambdaTakes(const parser::Node &lambda, llvm::StringRef name) {
  const parser::Node *arguments = ast::node(lambda, "args");
  if (!arguments)
    return false;
  for (const parser::Field &field : arguments->fields) {
    auto takes = [&](const parser::Node *arg) {
      return arg && arg->kind == "arg" &&
             llvm::StringRef(ast::string(*arg, "arg").value_or("")) == name;
    };
    if (const auto *one = std::get_if<parser::NodePtr>(&field.value)) {
      if (takes(one->get()))
        return true;
    } else if (const auto *many =
                   std::get_if<std::vector<parser::NodePtr>>(&field.value)) {
      for (const parser::NodePtr &arg : *many)
        if (takes(arg.get()))
          return true;
    }
  }
  return false;
}

bool comprehensionTargets(const parser::Node &comprehension,
                          llvm::StringRef name) {
  if (const auto *generators = ast::nodeList(comprehension, "generators"))
    for (const parser::NodePtr &generator : *generators) {
      llvm::SmallVector<std::string, 2> targets;
      if (generator)
        collectTargetNames(ast::node(*generator, "target"), targets);
      if (llvm::is_contained(targets, name))
        return true;
    }
  return false;
}

// Does a class body (not its methods) bind `name`?
bool classBodyBinds(const parser::Node &classDef, llvm::StringRef name) {
  const auto *body = ast::nodeList(classDef, "body");
  if (!body)
    return false;
  for (const parser::NodePtr &statement : *body) {
    if (!statement)
      continue;
    if ((isFunctionScope(*statement) || statement->kind == "ClassDef") &&
        llvm::StringRef(ast::string(*statement, "name").value_or("")) == name)
      return true;
    llvm::SmallVector<std::string, 4> targets;
    if (statement->kind == "Assign") {
      if (const auto *list = ast::nodeList(*statement, "targets"))
        for (const parser::NodePtr &target : *list)
          collectTargetNames(target.get(), targets);
    } else if (statement->kind == "AnnAssign" ||
               statement->kind == "AugAssign") {
      collectTargetNames(ast::node(*statement, "target"), targets);
    }
    if (llvm::is_contained(targets, name))
      return true;
  }
  return false;
}

// Every node of `node`'s tree takes `range`: a tree parsed from an
// annotation's TEXT carries positions inside that text, which no diagnostic
// should name.
void setRanges(parser::Node &node, const parser::SourceRange &range) {
  node.range = range;
  for (parser::Field &field : node.fields) {
    if (auto *child = std::get_if<parser::NodePtr>(&field.value)) {
      if (*child)
        setRanges(**child, range);
    } else if (auto *children =
                   std::get_if<std::vector<parser::NodePtr>>(&field.value)) {
      for (parser::NodePtr &element : *children)
        if (element)
          setRanges(*element, range);
    }
  }
}

using NameVisit = llvm::function_ref<void(parser::Node &)>;
using TextVisit = llvm::function_ref<void(parser::Node &)>;

// Every reference to the binding `name` of the scope being walked: a Name
// node, or a string annotation (`onText`). It stops where a nested scope binds
// `name` itself -- a def or lambda with its own `name`, a comprehension whose
// target is `name` past its first iterable, a class body that assigns it (not
// that class's methods, which see past the class scope).
void forEachReference(parser::Node *node, llvm::StringRef name,
                      NameVisit onName, TextVisit onText,
                      bool annotation = false) {
  if (!node)
    return;
  if (node->kind == "Name") {
    if (llvm::StringRef(ast::nameSpelling(*node)) == name)
      onName(*node);
    return;
  }
  if (annotation && node->kind == "Constant") {
    if (ast::string(*node, "value"))
      onText(*node);
    return;
  }
  auto visitField = [&](parser::Field &field, bool inAnnotation) {
    if (auto *child = std::get_if<parser::NodePtr>(&field.value))
      forEachReference(child->get(), name, onName, onText, inAnnotation);
    else if (auto *children =
                 std::get_if<std::vector<parser::NodePtr>>(&field.value))
      for (parser::NodePtr &element : *children)
        forEachReference(element.get(), name, onName, onText, inAnnotation);
  };
  if (isFunctionScope(*node)) {
    // Its decorators, defaults and annotations are evaluated around it.
    for (parser::Field &field : node->fields)
      if (field.name != "body")
        visitField(field, field.name == "returns");
    if (!scopeBindsOwn(*node, name))
      if (parser::Field *body = parser::findField(*node, "body"))
        visitField(*body, false);
    return;
  }
  if (node->kind == "arg") {
    if (parser::Field *field = parser::findField(*node, "annotation"))
      visitField(*field, true);
    return;
  }
  if (node->kind == "AnnAssign") {
    for (parser::Field &field : node->fields)
      visitField(field, field.name == "annotation");
    return;
  }
  if (node->kind == "Lambda") {
    if (parser::Field *field = parser::findField(*node, "args"))
      visitField(*field, false);
    if (!lambdaTakes(*node, name))
      if (parser::Field *body = parser::findField(*node, "body"))
        visitField(*body, false);
    return;
  }
  if (node->kind == "ClassDef") {
    bool shadowed = classBodyBinds(*node, name);
    for (parser::Field &field : node->fields) {
      if (field.name != "body") {
        visitField(field, false);
        continue;
      }
      if (auto *body = std::get_if<std::vector<parser::NodePtr>>(&field.value))
        for (parser::NodePtr &statement : *body)
          if (statement && (isFunctionScope(*statement) || !shadowed))
            forEachReference(statement.get(), name, onName, onText);
    }
    return;
  }
  if (isComprehension(*node) && comprehensionTargets(*node, name)) {
    if (const auto *generators = ast::nodeList(*node, "generators"))
      if (!generators->empty() && generators->front())
        forEachReference(const_cast<parser::Node *>(
                             ast::node(*generators->front(), "iter")),
                         name, onName, onText);
    return;
  }
  for (parser::Field &field : node->fields)
    visitField(field, false);
}

// How often each name is bound in a scope's own body -- a def's parameters
// and every statement form that binds -- not inside the scopes nested in it.
// A binding in a loop counts twice: it runs again.
llvm::StringMap<unsigned> ownBindingCounts(const parser::Node &scope) {
  llvm::StringMap<unsigned> counts;
  if (const parser::Node *arguments = ast::node(scope, "args"))
    for (const parser::Field &field : arguments->fields) {
      auto take = [&](const parser::Node *arg) {
        if (arg && arg->kind == "arg")
          ++counts[ast::string(*arg, "arg").value_or("")];
      };
      if (const auto *one = std::get_if<parser::NodePtr>(&field.value))
        take(one->get());
      else if (const auto *many =
                   std::get_if<std::vector<parser::NodePtr>>(&field.value))
        for (const parser::NodePtr &arg : *many)
          take(arg.get());
    }
  std::function<void(const parser::Node *, unsigned)> walk =
      [&](const parser::Node *node, unsigned weight) {
        if (!node)
          return;
        auto bind = [&](llvm::StringRef name) {
          if (!name.empty())
            counts[name] += weight;
        };
        auto targets = [&](const parser::Node *target) {
          llvm::SmallVector<std::string, 4> names;
          collectTargetNames(target, names);
          for (const std::string &name : names)
            bind(name);
        };
        llvm::StringRef kind = node->kind;
        if (isFunctionScope(*node) || kind == "ClassDef") {
          bind(ast::string(*node, "name").value_or(""));
          // Decorators and defaults are this scope's; nothing in them binds.
          return;
        }
        if (kind == "Lambda")
          return;
        if (kind == "Assign" || kind == "Delete") {
          if (const auto *list = ast::nodeList(*node, "targets"))
            for (const parser::NodePtr &target : *list)
              targets(target.get());
        } else if (kind == "AnnAssign") {
          if (ast::node(*node, "value"))
            targets(ast::node(*node, "target"));
        } else if (kind == "AugAssign" || kind == "NamedExpr") {
          targets(ast::node(*node, "target"));
        } else if (kind == "For" || kind == "AsyncFor") {
          // Its target is bound once per item.
          const parser::Node *target = ast::node(*node, "target");
          targets(target);
          targets(target);
        } else if (kind == "withitem") {
          targets(ast::node(*node, "optional_vars"));
        } else if (kind == "ExceptHandler" || kind == "MatchAs" ||
                   kind == "MatchStar") {
          bind(ast::string(*node, "name").value_or(""));
        } else if (kind == "alias") {
          llvm::StringRef bound = ast::string(*node, "asname").value_or("");
          if (bound.empty())
            bound = llvm::StringRef(ast::string(*node, "name").value_or(""))
                        .split('.')
                        .first;
          bind(bound);
        } else if (kind == "Global" || kind == "Nonlocal") {
          if (const auto *declared = ast::stringList(*node, "names"))
            for (const std::string &name : *declared)
              bind(name);
        }
        unsigned inner =
            kind == "For" || kind == "AsyncFor" || kind == "While" ? 2 * weight
                                                                    : weight;
        for (const parser::Field &field : node->fields) {
          if (const auto *child = std::get_if<parser::NodePtr>(&field.value))
            walk(child->get(), inner);
          else if (const auto *children =
                       std::get_if<std::vector<parser::NodePtr>>(&field.value))
            for (const parser::NodePtr &element : *children)
              walk(element.get(), inner);
        }
      };
  if (const auto *body = ast::nodeList(scope, "body"))
    for (const parser::NodePtr &statement : *body)
      walk(statement.get(), 1);
  return counts;
}

// The module names whose binding a call may find changed: what the module
// binds more than once, and what any def declares `global`.
llvm::StringSet<> rebindableModuleNames(const parser::Node &moduleNode) {
  llvm::StringSet<> names;
  for (const auto &entry : ownBindingCounts(moduleNode))
    if (entry.getValue() > 1)
      names.insert(entry.getKey());
  std::function<void(const parser::Node *)> walk = [&](const parser::Node *node) {
    if (!node)
      return;
    if (node->kind == "Global")
      if (const auto *declared = ast::stringList(*node, "names"))
        for (const std::string &name : *declared)
          names.insert(name);
    for (const parser::Field &field : node->fields) {
      if (const auto *child = std::get_if<parser::NodePtr>(&field.value))
        walk(child->get());
      else if (const auto *children =
                   std::get_if<std::vector<parser::NodePtr>>(&field.value))
        for (const parser::NodePtr &element : *children)
          walk(element.get());
    }
  };
  walk(&moduleNode);
  return names;
}

// The names the module's `import` statements bind to a module.
llvm::StringSet<> importedModuleNames(const parser::Node &moduleNode) {
  llvm::StringSet<> names;
  if (const auto *body = ast::nodeList(moduleNode, "body"))
    for (const parser::NodePtr &statement : *body)
      if (statement && statement->kind == "Import")
        if (const auto *aliases = ast::nodeList(*statement, "names"))
          for (const parser::NodePtr &alias : *aliases) {
            if (!alias)
              continue;
            llvm::StringRef bound = ast::string(*alias, "asname").value_or("");
            if (bound.empty())
              bound = llvm::StringRef(ast::string(*alias, "name").value_or(""))
                          .split('.')
                          .first;
            names.insert(bound);
          }
  return names;
}

// What a class statement evaluates is evaluated once here, where CPython
// evaluates it for every call of the function: only a value no call could
// see differ may be written there.
struct OnceEvaluation {
  const llvm::StringSet<> &rebindable;
  const llvm::StringSet<> &modules;
  // The class attributes the body has bound so far.
  llvm::StringSet<> attributes;

  // Literals and what arithmetic makes of them: the same immutable value
  // every time.
  static bool constantOnly(const parser::Node *node) {
    if (!node)
      return true;
    llvm::StringRef kind = node->kind;
    auto all = [](const std::vector<parser::NodePtr> *nodes) {
      return !nodes || llvm::all_of(*nodes, [](const parser::NodePtr &each) {
               return constantOnly(each.get());
             });
    };
    if (kind == "Constant")
      return true;
    if (kind == "UnaryOp")
      return constantOnly(ast::node(*node, "operand"));
    if (kind == "BinOp")
      return constantOnly(ast::node(*node, "left")) &&
             constantOnly(ast::node(*node, "right"));
    if (kind == "BoolOp")
      return all(ast::nodeList(*node, "values"));
    if (kind == "Compare")
      return constantOnly(ast::node(*node, "left")) &&
             all(ast::nodeList(*node, "comparators"));
    if (kind == "Tuple")
      return all(ast::nodeList(*node, "elts"));
    return false;
  }

  static std::string spelling(const parser::Node *node) {
    if (!node)
      return "";
    if (node->kind == "Name")
      return std::string(ast::nameSpelling(*node));
    if (node->kind == "Attribute")
      return spelling(ast::node(*node, "value")) + "." +
             std::string(ast::string(*node, "attr").value_or(""));
    return "";
  }

  // Why evaluating `node` again could give a call another value -- a new
  // object, something run, a name rebound -- or nothing when it cannot.
  std::string differs(const parser::Node *node) const {
    if (!node)
      return {};
    llvm::StringRef kind = node->kind;
    if (kind == "Constant" || constantOnly(node))
      return {};
    if (kind == "Name") {
      llvm::StringRef id = ast::nameSpelling(*node);
      if (!attributes.contains(id) && rebindable.contains(id))
        return "reads '" + id.str() +
               "', which the module binds more than once";
      return {};
    }
    if (kind == "Attribute") {
      const parser::Node *owner = ast::node(*node, "value");
      if (owner && owner->kind == "Name" &&
          modules.contains(ast::nameSpelling(*owner)) &&
          !rebindable.contains(ast::nameSpelling(*owner)))
        return {};
      return "reads '" + spelling(node) +
             "', an attribute a call may find changed";
    }
    if (kind == "Tuple") {
      if (const auto *elements = ast::nodeList(*node, "elts"))
        for (const parser::NodePtr &element : *elements)
          if (std::string reason = differs(element.get()); !reason.empty())
            return reason;
      return {};
    }
    if (kind == "Call") {
      std::string callee = spelling(ast::node(*node, "func"));
      const auto *arguments = ast::nodeList(*node, "args");
      const auto *keywords = ast::nodeList(*node, "keywords");
      bool noArguments = (!arguments || arguments->empty()) &&
                         (!keywords || keywords->empty());
      // enum's numbering, and a dataclass field's spec: each call's class
      // would get the same ones.
      if ((callee == "auto" || callee == "enum.auto") && noArguments)
        return {};
      if (callee == "field" || callee == "dataclasses.field") {
        if (arguments && !arguments->empty())
          return "calls '" + callee + "' with positional arguments";
        if (keywords)
          for (const parser::NodePtr &keyword : *keywords)
            if (keyword)
              if (std::string reason = differs(ast::node(*keyword, "value"));
                  !reason.empty())
                return reason;
        return {};
      }
      return "calls " + (callee.empty() ? std::string("a function")
                                        : "'" + callee + "'") +
             ", which runs again for every call";
    }
    if (kind == "List" || kind == "ListComp")
      return "makes a new list for every call";
    if (kind == "Dict" || kind == "DictComp")
      return "makes a new dict for every call";
    if (kind == "Set" || kind == "SetComp")
      return "makes a new set for every call";
    if (kind == "Lambda")
      return "makes a new function for every call";
    return "is evaluated again for every call";
  }

  // A method's decorator: the descriptors the class statement wraps a def
  // in. A function of the program's own would run again for every call.
  std::string decoratorDiffers(const parser::Node *node) const {
    std::string callee = spelling(node);
    llvm::StringRef name = callee;
    if (name.consume_front("abc.") || name.consume_front("typing.") ||
        name.consume_front("functools.") || name.consume_front("builtins."))
      ;
    static constexpr llvm::StringLiteral kDescriptors[] = {
        "staticmethod", "classmethod",     "property", "abstractmethod",
        "override",     "cached_property", "final",    "overload"};
    if (llvm::is_contained(kDescriptors, name))
      return {};
    // `@x.setter` on a property the body defined.
    static constexpr llvm::StringLiteral kAccessors[] = {"setter", "getter",
                                                         "deleter"};
    if (node && node->kind == "Attribute")
      if (const parser::Node *owner = ast::node(*node, "value");
          owner && owner->kind == "Name" &&
          attributes.contains(ast::nameSpelling(*owner)) &&
          llvm::is_contained(kAccessors, llvm::StringRef(ast::string(
                                             *node, "attr")
                                             .value_or(""))))
        return {};
    return "applies " +
           (callee.empty() ? std::string("a decorator") : "'" + callee + "'") +
           ", which runs again for every call";
  }

  // The first thing a class statement evaluates that `differs`, and what
  // it is: "the value of 'items' makes a new list for every call".
  std::pair<const parser::Node *, std::string>
  body(const parser::Node &classDef) {
    for (llvm::StringRef field : {"decorator_list", "keywords"})
      if (const auto *list = ast::nodeList(classDef, field))
        for (const parser::NodePtr &element : *list) {
          if (!element)
            continue;
          const parser::Node *value = element->kind == "keyword"
                                          ? ast::node(*element, "value")
                                          : element.get();
          // A decorator call's callee is the decorator, and only its
          // arguments are evaluated as values: `@dataclass(order=True)`.
          if (value && value->kind == "Call") {
            for (llvm::StringRef part : {"args", "keywords"})
              if (const auto *items = ast::nodeList(*value, part))
                for (const parser::NodePtr &item : *items)
                  if (item)
                    if (std::string reason = differs(
                            item->kind == "keyword" ? ast::node(*item, "value")
                                                    : item.get());
                        !reason.empty())
                      return {item.get(), "an argument of its decorator " +
                                              reason};
          } else if (std::string reason = differs(value); !reason.empty()) {
            return {element.get(), (field == "keywords"
                                        ? "a keyword of its statement "
                                        : "its decorator ") +
                                       reason};
          }
        }
    const auto *members = ast::nodeList(classDef, "body");
    if (!members)
      return {};
    for (const parser::NodePtr &member : *members) {
      if (!member)
        continue;
      llvm::StringRef kind = member->kind;
      if (kind == "Pass")
        continue;
      if (kind == "Expr") {
        const parser::Node *value = ast::node(*member, "value");
        if (value && value->kind == "Constant")
          continue;
        return {member.get(), "its body runs a statement"};
      }
      if (kind == "Assign" || kind == "AnnAssign") {
        llvm::SmallVector<std::string, 2> names;
        if (kind == "Assign") {
          if (const auto *targets = ast::nodeList(*member, "targets"))
            for (const parser::NodePtr &target : *targets)
              collectTargetNames(target.get(), names);
        } else {
          collectTargetNames(ast::node(*member, "target"), names);
        }
        if (std::string reason = differs(ast::node(*member, "value"));
            !reason.empty())
          return {member.get(),
                  (names.empty() ? std::string("an assignment")
                                 : "the value of '" + names.front() + "'") +
                      " " + reason};
        for (const std::string &name : names)
          attributes.insert(name);
        continue;
      }
      if (isFunctionScope(*member)) {
        std::string method(ast::string(*member, "name").value_or(""));
        if (const auto *decorators = ast::nodeList(*member, "decorator_list"))
          for (const parser::NodePtr &decorator : *decorators)
            if (std::string reason = decoratorDiffers(decorator.get());
                !reason.empty())
              return {decorator.get(), "'" + method + "' " + reason};
        // Its defaults are evaluated as the def statement runs.
        if (const parser::Node *arguments = ast::node(*member, "args"))
          for (llvm::StringRef field : {"defaults", "kw_defaults"})
            if (const auto *defaults = ast::nodeList(*arguments, field))
              for (const parser::NodePtr &value : *defaults)
                if (std::string reason = differs(value.get());
                    !reason.empty())
                  return {value.get(),
                          "a default of '" + method + "' " + reason};
        attributes.insert(ast::string(*member, "name").value_or(""));
        continue;
      }
      // A class nested in this one is refused where classes are emitted.
      if (kind == "ClassDef") {
        attributes.insert(ast::string(*member, "name").value_or(""));
        continue;
      }
      return {member.get(), "its body runs a statement"};
    }
    return {};
  }
};

// A class statement of a def's own scope, where it stands.
struct OwnClass {
  parser::NodePtr classDef;
  std::vector<parser::NodePtr> *list = nullptr;
  std::size_t index = 0;
};

void collectOwnClasses(std::vector<parser::NodePtr> &statements,
                       std::vector<OwnClass> &found) {
  for (std::size_t index = 0; index < statements.size(); ++index) {
    parser::NodePtr &statement = statements[index];
    if (!statement)
      continue;
    if (statement->kind == "ClassDef") {
      found.push_back(OwnClass{statement, &statements, index});
      continue;
    }
    if (isFunctionScope(*statement))
      continue;
    for (parser::Field &field : statement->fields) {
      if (auto *children =
              std::get_if<std::vector<parser::NodePtr>>(&field.value)) {
        if (field.name == "body" || field.name == "orelse" ||
            field.name == "finalbody") {
          collectOwnClasses(*children, found);
        } else if (field.name == "handlers" || field.name == "cases") {
          for (parser::NodePtr &clause : *children)
            if (clause)
              if (auto *body = const_cast<std::vector<parser::NodePtr> *>(
                      ast::nodeList(*clause, "body")))
                collectOwnClasses(*body, found);
        }
      }
    }
  }
}

// The names a class statement reads from the scopes around it: its bases,
// keywords and decorators, its body's statements, and what its methods
// capture.
llvm::StringSet<> classStatementReads(const parser::Node &classDef) {
  llvm::StringSet<> reads;
  std::function<void(const parser::Node *)> collect =
      [&](const parser::Node *node) {
        if (!node)
          return;
        if (node->kind == "Name")
          reads.insert(ast::nameSpelling(*node));
        if (node->kind == "Lambda" || isFunctionScope(*node))
          return;
        for (const parser::Field &field : node->fields) {
          if (const auto *child = std::get_if<parser::NodePtr>(&field.value))
            collect(child->get());
          else if (const auto *children =
                       std::get_if<std::vector<parser::NodePtr>>(&field.value))
            for (const parser::NodePtr &element : *children)
              collect(element.get());
        }
      };
  for (llvm::StringRef field : {"bases", "keywords", "decorator_list"})
    if (const auto *list = ast::nodeList(classDef, field))
      for (const parser::NodePtr &element : *list)
        collect(element.get());
  if (const auto *members = ast::nodeList(classDef, "body"))
    for (const parser::NodePtr &member : *members) {
      if (!member)
        continue;
      if (isFunctionScope(*member)) {
        for (const std::string &capture : lexicalCaptureNames(*member))
          reads.insert(capture);
        // Its decorators and defaults are read by the class body.
        if (const auto *decorators = ast::nodeList(*member, "decorator_list"))
          for (const parser::NodePtr &decorator : *decorators)
            collect(decorator.get());
        if (const parser::Node *arguments = ast::node(*member, "args"))
          for (llvm::StringRef field : {"defaults", "kw_defaults"})
            if (const auto *defaults = ast::nodeList(*arguments, field))
              for (const parser::NodePtr &value : *defaults)
                collect(value.get());
        continue;
      }
      collect(member.get());
    }
  return reads;
}

} // namespace

// ⭐ A CLASS WRITTEN IN A FUNCTION IS ONE CLASS, declared where the module's
// classes are, and its NAME is the function's: only the references that name
// the function's binding -- inside the function, its nested defs, the class's
// own methods and annotations -- are respelled as the class's contract,
// `f.<locals>.A` as `f$<locals>$A` (Contracts.h). Every other part of this
// compiler then sees a module-level class it already handles, and reads its
// qualified name back for what CPython prints.
//
// ⛔ What a class statement in a function means that one class does not, each
// refused here:
//   - a name used where the statement may not have run (before it, or past
//     the block it is written in) -- CPython's NameError;
//   - the function binding the name again, or two classes of one name;
//   - a class that reads the function's locals: each call would make a class
//     with its own closure, and one class has one;
//   - a class statement that evaluates something a call could see differ --
//     a new list as a class attribute, a call, a program decorator: CPython
//     evaluates it for every call, and here it is evaluated once.
// And one refused where it is emitted: a class attribute assigned outside the
// body, which each call's class would have its own of (EmitterStatements).
// Identity is the one difference left: two calls' classes are one class here
// (`type(make()) is type(make())`), as a statically built class is.
void ModuleEmitter::hoistFunctionLocalClasses(const parser::Node &moduleNode,
                                              llvm::StringRef moduleName) {
  auto *moduleBody = const_cast<std::vector<parser::NodePtr> *>(
      ast::nodeList(moduleNode, "body"));
  if (!moduleBody)
    return;
  auto refuse = [&](const parser::Node &anchor, std::string message) {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, anchor.range.start, std::move(message)});
  };
  llvm::StringSet<> rebindable = rebindableModuleNames(moduleNode);
  llvm::StringSet<> modules = importedModuleNames(moduleNode);
  // Hoisted classes, each to go in front of the module statement holding it.
  std::vector<std::pair<const parser::Node *, parser::NodePtr>> hoisted;

  // `function` is a def scope; `prefix` its qualified name and `.<locals>.`;
  // `enclosingLocals` what the defs around it bind.
  std::function<void(parser::Node &, const std::string &,
                     const parser::Node *, const llvm::StringSet<> &)>
      processScope;
  processScope = [&](parser::Node &function, const std::string &prefix,
                     const parser::Node *topLevel,
                     const llvm::StringSet<> &enclosingLocals) {
    auto *body = const_cast<std::vector<parser::NodePtr> *>(
        ast::nodeList(function, "body"));
    if (!body)
      return;
    llvm::StringSet<> locals = enclosingLocals;
    for (const auto &entry : functionLocalNames(function))
      locals.insert(entry.getKey());
    llvm::StringMap<unsigned> bindings = ownBindingCounts(function);
    std::vector<OwnClass> classes;
    collectOwnClasses(*body, classes);
    auto refuseClass = [&](const OwnClass &own, std::string message) {
      refuse(*own.classDef, std::move(message));
      refusedLocalClasses.insert(own.classDef.get());
    };
    // What the statement alone decides.
    llvm::StringMap<unsigned> statements;
    for (const OwnClass &own : classes)
      ++statements[ast::string(*own.classDef, "name").value_or("")];
    llvm::StringSet<> declared;
    llvm::StringSet<> rebound;
    if (const auto *statementsOfBody = ast::nodeList(function, "body"))
      for (const parser::NodePtr &statement : *statementsOfBody) {
        collectDeclarations(statement.get(), "Global", declared);
        collectDeclarations(statement.get(), "Nonlocal", declared);
        collectNestedNonlocals(statement.get(), /*nested=*/false, rebound);
      }
    std::vector<OwnClass> candidates;
    for (const OwnClass &own : classes) {
      std::string name(ast::string(*own.classDef, "name").value_or(""));
      if (statements.lookup(name) > 1) {
        refuseClass(own, "two classes named '" + name +
                             "' in one function are not supported: each "
                             "would be '" + prefix + name + "'");
        continue;
      }
      if (declared.contains(name)) {
        refuseClass(own, "'" + name + "' is declared global or nonlocal in "
                         "the function that defines the class: the statement "
                         "would bind a name outside the function at every "
                         "call, which one class cannot");
        continue;
      }
      if (bindings.lookup(name) != 1 || rebound.contains(name)) {
        refuseClass(own, "'" + name + "' is bound again in the function that "
                         "defines the class" +
                             (rebound.contains(name)
                                  ? std::string(", by a nested def's nonlocal")
                                  : std::string()) +
                             ": a class written in a function is its one "
                             "binding of that name here");
        continue;
      }
      OnceEvaluation once{rebindable, modules, {}};
      if (auto [at, reason] = once.body(*own.classDef); at) {
        refuse(*at, "class '" + name + "' is written in a function, and " +
                        reason + ": the class statement runs once here, "
                        "where CPython runs it for every call");
        refusedLocalClasses.insert(own.classDef.get());
        continue;
      }
      candidates.push_back(own);
    }
    // A class that reads a local is refused, and so, in turn, is one that
    // reads a class refused: until nothing changes.
    llvm::StringSet<> classNames;
    for (const OwnClass &own : candidates)
      classNames.insert(ast::string(*own.classDef, "name").value_or(""));
    for (bool changed = true; changed;) {
      changed = false;
      for (auto it = candidates.begin(); it != candidates.end(); ++it) {
        std::string name(ast::string(*it->classDef, "name").value_or(""));
        std::string captured;
        for (const auto &entry : classStatementReads(*it->classDef)) {
          llvm::StringRef read = entry.getKey();
          if (read != name && locals.contains(read) &&
              !classNames.contains(read)) {
            captured = read.str();
            break;
          }
        }
        if (captured.empty())
          continue;
        refuseClass(*it, "class '" + name + "' reads '" + captured +
                             "', a local of the function that defines it: "
                             "a class written in a function cannot read the "
                             "function's locals yet");
        classNames.erase(name);
        candidates.erase(it);
        changed = true;
        break;
      }
    }
    // Respelled where the statement has run: what follows it in its own
    // block, the class's methods, and the annotations of its body.
    for (const OwnClass &own : candidates) {
      parser::Node &classDef = *own.classDef;
      std::string name(ast::string(classDef, "name").value_or(""));
      std::string contract =
          py::contracts::localClassContractName(prefix + name);
      auto rename = [&](parser::Node &reference) {
        for (parser::Field &field : reference.fields)
          if (field.name == "id")
            field.value = contract;
      };
      // A quoted annotation is the expression it spells: parsed, respelled,
      // and standing where the string stood -- the contract has a `$` no
      // string of Python source can hold.
      auto renameText = [&](parser::Node &constant) {
        std::optional<std::string_view> text = ast::string(constant, "value");
        if (!text)
          return;
        parser::ParseResult parsed = parser::parse(
            *text, "<annotation>",
            parser::ParseOptions{parser::ParseMode::Expression,
                                 /*typeComments=*/false});
        if (!parsed.tree || !parsed.diagnostics.empty())
          return;
        parser::NodePtr expression;
        if (const parser::Field *field = parser::findField(*parsed.tree, "body"))
          if (const auto *node = std::get_if<parser::NodePtr>(&field->value))
            expression = *node;
        if (!expression || expression->kind == "Constant")
          return;
        bool named = false;
        forEachReference(
            expression.get(), name,
            [&](parser::Node &reference) {
              rename(reference);
              named = true;
            },
            [](parser::Node &) {});
        if (!named)
          return;
        setRanges(*expression, constant.range);
        constant.kind = expression->kind;
        constant.fields = expression->fields;
        constant.fieldIndicesBySlot = expression->fieldIndicesBySlot;
      };
      for (std::size_t at = own.index + 1; at < own.list->size(); ++at)
        forEachReference((*own.list)[at].get(), name, rename, renameText);
      if (auto *members = const_cast<std::vector<parser::NodePtr> *>(
              ast::nodeList(classDef, "body")))
        for (parser::NodePtr &member : *members) {
          if (!member)
            continue;
          if (isFunctionScope(*member))
            forEachReference(member.get(), name, rename, renameText);
          else if (member->kind == "AnnAssign")
            if (parser::Field *annotation =
                    parser::findField(*member, "annotation"))
              if (auto *node = std::get_if<parser::NodePtr>(&annotation->value))
                forEachReference(node->get(), name, rename, renameText,
                                 /*annotation=*/true);
        }
    }
    // What still names an accepted class of this function is a use the
    // statement may not have run before.
    for (const OwnClass &own : candidates) {
      std::string name(ast::string(*own.classDef, "name").value_or(""));
      for (parser::NodePtr &statement : *body)
        forEachReference(
            statement.get(), name,
            [&](parser::Node &reference) {
              refuse(reference,
                     "'" + name + "' is used where its class statement may "
                     "not have run: a class written in a function is bound "
                     "by that statement, and only what follows it in its "
                     "block can name it");
            },
            [](parser::Node &) {});
    }
    for (const OwnClass &own : candidates) {
      parser::Node &classDef = *own.classDef;
      std::string name(ast::string(classDef, "name").value_or(""));
      // The class leaves the function; the statement's place stays a
      // statement, so a block of one class is still a block.
      (*own.list)[own.index] = parser::makeNode("Pass", classDef.range);
      if (parser::Field *field = parser::findField(classDef, "name"))
        field->value = py::contracts::localClassContractName(prefix + name);
      pythonNameOverrides[&classDef] = name;
      hoistedClassPrefixes[&classDef] = prefix;
      hoisted.push_back({topLevel, own.classDef});
      // Its methods are scopes of their own, with classes of their own.
      if (auto *members = const_cast<std::vector<parser::NodePtr> *>(
              ast::nodeList(classDef, "body")))
        for (parser::NodePtr &member : *members)
          if (member && isFunctionScope(*member))
            processScope(*member,
                         prefix + name + "." +
                             std::string(
                                 ast::string(*member, "name").value_or("")) +
                             ".<locals>.",
                         topLevel, locals);
    }
    // The defs nested in this one.
    std::function<void(std::vector<parser::NodePtr> &)> nested =
        [&](std::vector<parser::NodePtr> &statements) {
          for (parser::NodePtr &statement : statements) {
            if (!statement)
              continue;
            if (isFunctionScope(*statement)) {
              processScope(
                  *statement,
                  prefix +
                      std::string(
                          ast::string(*statement, "name").value_or("")) +
                      ".<locals>.",
                  topLevel, locals);
              continue;
            }
            if (statement->kind == "ClassDef")
              continue;
            for (parser::Field &field : statement->fields)
              if (auto *children =
                      std::get_if<std::vector<parser::NodePtr>>(&field.value)) {
                if (field.name == "body" || field.name == "orelse" ||
                    field.name == "finalbody") {
                  nested(*children);
                } else if (field.name == "handlers" || field.name == "cases") {
                  for (parser::NodePtr &clause : *children)
                    if (clause)
                      if (auto *clauseBody =
                              const_cast<std::vector<parser::NodePtr> *>(
                                  ast::nodeList(*clause, "body")))
                        nested(*clauseBody);
                }
              }
          }
        };
    nested(*body);
  };

  // The module's defs and classes, at its top level or in a block there;
  // a hoisted class goes in front of the top-level statement holding it.
  llvm::StringSet<> none;
  std::function<void(std::vector<parser::NodePtr> &, const parser::Node *)>
      moduleScope = [&](std::vector<parser::NodePtr> &statements,
                        const parser::Node *topLevel) {
        for (parser::NodePtr &statement : statements) {
          if (!statement)
            continue;
          const parser::Node *holder = topLevel ? topLevel : statement.get();
          std::string name(ast::string(*statement, "name").value_or(""));
          if (isFunctionScope(*statement)) {
            processScope(*statement, name + ".<locals>.", holder, none);
            continue;
          }
          if (statement->kind == "ClassDef") {
            if (auto *members = const_cast<std::vector<parser::NodePtr> *>(
                    ast::nodeList(*statement, "body")))
              for (parser::NodePtr &member : *members)
                if (member && isFunctionScope(*member))
                  processScope(
                      *member,
                      name + "." +
                          std::string(
                              ast::string(*member, "name").value_or("")) +
                          ".<locals>.",
                      holder, none);
            continue;
          }
          for (parser::Field &field : statement->fields)
            if (auto *children =
                    std::get_if<std::vector<parser::NodePtr>>(&field.value)) {
              if (field.name == "body" || field.name == "orelse" ||
                  field.name == "finalbody") {
                moduleScope(*children, holder);
              } else if (field.name == "handlers" || field.name == "cases") {
                for (parser::NodePtr &clause : *children)
                  if (clause)
                    if (auto *clauseBody =
                            const_cast<std::vector<parser::NodePtr> *>(
                                ast::nodeList(*clause, "body")))
                      moduleScope(*clauseBody, holder);
              }
            }
        }
      };
  moduleScope(*moduleBody, nullptr);
  if (hoisted.empty())
    return;
  // In front of the module statement that held each, in source order.
  std::vector<parser::NodePtr> rebuilt;
  rebuilt.reserve(moduleBody->size() + hoisted.size());
  for (parser::NodePtr &statement : *moduleBody) {
    for (const auto &[topLevel, classDef] : hoisted)
      if (topLevel == statement.get())
        rebuilt.push_back(classDef);
    rebuilt.push_back(statement);
  }
  *moduleBody = std::move(rebuilt);
  (void)moduleName;
}

} // namespace lython::emitter
