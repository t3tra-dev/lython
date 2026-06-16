#include "BuilderImpl.h"

#include <set>

namespace lython::emitter {
namespace {

const std::set<std::string> &asyncioBuiltins() {
  static const std::set<std::string> values = {
      "run", "create_task", "ensure_future", "gather", "sleep"};
  return values;
}

const std::set<std::string> &lyrtBuiltins() {
  static const std::set<std::string> values = {"native", "to_prim", "from_prim",
                                               "alloc", "dealloc"};
  return values;
}

const std::set<std::string> &primitiveTypes() {
  static const std::set<std::string> values = {"Int",    "UInt",   "Float",
                                               "Vector", "Matrix", "Tensor"};
  return values;
}

std::optional<std::pair<std::string, std::string>>
lythonFacadeSymbol(llvm::StringRef name) {
  if (lyrtBuiltins().count(name.str()))
    return std::make_pair(std::string("lyrt"), name.str());
  if (primitiveTypes().count(name.str()))
    return std::make_pair(std::string("lyrt.prim"), name.str());
  return std::nullopt;
}

const std::set<std::string> &futureFeatures() {
  static const std::set<std::string> values = {"annotations", "barry_as_FLUFL"};
  return values;
}

const std::set<std::string> &typingModules() {
  static const std::set<std::string> values = {"typing", "collections.abc"};
  return values;
}

const std::set<std::string> &typesAnnotationNames() {
  static const std::set<std::string> values = {"NoneType", "TracebackType"};
  return values;
}

const std::set<std::string> &typingAnnotationNames() {
  static const std::set<std::string> values = {"Any",
                                               "Callable",
                                               "Coroutine",
                                               "Task",
                                               "Future",
                                               "Awaitable",
                                               "Generator",
                                               "AsyncIterable",
                                               "AsyncIterator",
                                               "AsyncGenerator",
                                               "ContextManager",
                                               "AsyncContextManager",
                                               "TracebackType",
                                               "Type",
                                               "SupportsInt",
                                               "SupportsFloat",
                                               "SupportsComplex",
                                               "SupportsBytes",
                                               "SupportsIndex",
                                               "SupportsAbs",
                                               "SupportsRound",
                                               "Iterable",
                                               "Iterator",
                                               "Container",
                                               "Sized",
                                               "Reversible",
                                               "Collection",
                                               "Sequence",
                                               "MutableSequence",
                                               "Mapping",
                                               "MutableMapping",
                                               "AbstractSet",
                                               "MutableSet",
                                               "Hashable",
                                               "List",
                                               "Tuple",
                                               "Dict",
                                               "list",
                                               "tuple",
                                               "dict",
                                               "type",
                                               "TypeAlias",
                                               "Optional",
                                               "Union",
                                               "TypeVar",
                                               "TypeVarTuple",
                                               "ParamSpec",
                                               "Concatenate",
                                               "Generic"};
  return values;
}

} // namespace

void Builder::Impl::scanStaticImports(const parser::Node &moduleNode) {
  const std::vector<parser::NodePtr> *body = nodeListField(moduleNode, "body");
  if (!body)
    return;
  for (const parser::NodePtr &stmt : *body) {
    if (!stmt)
      continue;
    if (stmt->kind == "Import") {
      emitImport(*stmt, /*reportErrors=*/false);
      continue;
    }
    if (stmt->kind == "ImportFrom")
      emitImportFrom(*stmt, /*reportErrors=*/false);
  }
}

void Builder::Impl::emitImport(const parser::Node &stmt, bool reportErrors) {
  auto report = [&](const parser::Node &node, std::string message) {
    if (reportErrors)
      error(node, std::move(message));
  };
  const std::vector<parser::NodePtr> *names = nodeListField(stmt, "names");
  if (!names) {
    report(stmt, "Import.names is missing");
    return;
  }
  for (const parser::NodePtr &alias : *names) {
    if (!alias)
      continue;
    const std::string *name = stringField(*alias, "name");
    const std::string *asname = stringField(*alias, "asname");
    if (!name) {
      report(*alias, "import alias name is missing");
      continue;
    }
    if (*name == "asyncio") {
      staticModules[(asname && !asname->empty()) ? *asname : *name] = "asyncio";
      continue;
    }
    if (typingModules().count(*name)) {
      staticModules[(asname && !asname->empty()) ? *asname : *name] = *name;
      continue;
    }
    if (*name == "types") {
      staticModules[(asname && !asname->empty()) ? *asname : *name] = "types";
      continue;
    }
    if (*name == "lyrt") {
      staticModules[(asname && !asname->empty()) ? *asname : *name] = "lyrt";
      continue;
    }
    if (*name == "lyrt.prim") {
      staticModules[(asname && !asname->empty()) ? *asname : *name] =
          "lyrt.prim";
      continue;
    }
    if (*name == "lython") {
      staticModules[(asname && !asname->empty()) ? *asname : *name] = "lython";
      continue;
    }
    report(*alias,
           "Import '" + *name + "' is not supported by the C++ emitter");
  }
}

void Builder::Impl::emitImportFrom(const parser::Node &stmt,
                                   bool reportErrors) {
  auto report = [&](const parser::Node &node, std::string message) {
    if (reportErrors)
      error(node, std::move(message));
  };
  const std::string *moduleName = stringField(stmt, "module");
  const std::vector<parser::NodePtr> *names = nodeListField(stmt, "names");
  const parser::FieldValue *levelValue = valueField(stmt, "level");
  const auto *level =
      levelValue ? std::get_if<std::int64_t>(levelValue) : nullptr;
  if (!names) {
    report(stmt, "ImportFrom.names is missing");
    return;
  }
  if (!moduleName) {
    if (level && *level > 0)
      report(stmt, "Relative imports are not supported by the C++ emitter");
    else
      report(stmt, "ImportFrom.module is missing");
    return;
  }

  if (*moduleName == "__future__" && (!level || *level == 0)) {
    for (const parser::NodePtr &alias : *names) {
      if (!alias)
        continue;
      const std::string *name = stringField(*alias, "name");
      if (!name || !futureFeatures().count(*name)) {
        report(alias ? *alias : stmt, "Unsupported __future__ import");
        continue;
      }
    }
    return;
  }

  if (typingModules().count(*moduleName) && (!level || *level == 0)) {
    for (const parser::NodePtr &alias : *names) {
      if (!alias)
        continue;
      const std::string *name = stringField(*alias, "name");
      const std::string *asname = stringField(*alias, "asname");
      if (!name) {
        report(*alias, "import alias name is missing");
        continue;
      }
      if (!typingAnnotationNames().count(*name)) {
        report(*alias, "Unsupported type-only import from " + *moduleName +
                           ": " + *name);
        continue;
      }
      staticAnnotationAliases[(asname && !asname->empty()) ? *asname : *name] =
          *name;
    }
    return;
  }

  if (*moduleName == "types" && (!level || *level == 0)) {
    for (const parser::NodePtr &alias : *names) {
      if (!alias)
        continue;
      const std::string *name = stringField(*alias, "name");
      const std::string *asname = stringField(*alias, "asname");
      if (!name) {
        report(*alias, "import alias name is missing");
        continue;
      }
      if (!typesAnnotationNames().count(*name)) {
        report(*alias, "Unsupported type-only import from types: " + *name);
        continue;
      }
      staticAnnotationAliases[(asname && !asname->empty()) ? *asname : *name] =
          *name;
    }
    return;
  }

  if (*moduleName == "asyncio") {
    for (const parser::NodePtr &alias : *names) {
      if (!alias)
        continue;
      const std::string *name = stringField(*alias, "name");
      const std::string *asname = stringField(*alias, "asname");
      if (!name) {
        report(*alias, "import alias name is missing");
        continue;
      }
      if (!asyncioBuiltins().count(*name)) {
        report(*alias, "Unsupported asyncio import: " + *name);
        continue;
      }
      staticModuleSymbols[(asname && !asname->empty()) ? *asname : *name] =
          std::make_pair("asyncio", *name);
    }
    return;
  }

  if (*moduleName == "lyrt") {
    for (const parser::NodePtr &alias : *names) {
      if (!alias)
        continue;
      const std::string *name = stringField(*alias, "name");
      const std::string *asname = stringField(*alias, "asname");
      if (!name || !lyrtBuiltins().count(*name)) {
        report(alias ? *alias : stmt, "Unsupported lyrt import");
        continue;
      }
      staticModuleSymbols[(asname && !asname->empty()) ? *asname : *name] =
          std::make_pair("lyrt", *name);
    }
    return;
  }

  if (*moduleName == "lyrt.prim") {
    for (const parser::NodePtr &alias : *names) {
      if (!alias)
        continue;
      const std::string *name = stringField(*alias, "name");
      const std::string *asname = stringField(*alias, "asname");
      if (!name || !primitiveTypes().count(*name)) {
        report(alias ? *alias : stmt, "Unsupported lyrt.prim import");
        continue;
      }
      staticModuleSymbols[(asname && !asname->empty()) ? *asname : *name] =
          std::make_pair("lyrt.prim", *name);
    }
    return;
  }

  if (*moduleName == "lython") {
    for (const parser::NodePtr &alias : *names) {
      if (!alias)
        continue;
      const std::string *name = stringField(*alias, "name");
      const std::string *asname = stringField(*alias, "asname");
      std::optional<std::pair<std::string, std::string>> symbol =
          name ? lythonFacadeSymbol(*name) : std::nullopt;
      if (!symbol) {
        report(alias ? *alias : stmt, "Unsupported lython import");
        continue;
      }
      staticModuleSymbols[(asname && !asname->empty()) ? *asname : *name] =
          *symbol;
    }
    return;
  }

  report(stmt, "Import from '" + *moduleName +
                   "' is not supported by the C++ emitter");
}

std::optional<std::string>
Builder::Impl::staticSymbol(const parser::Node &node,
                            llvm::StringRef moduleName) const {
  if (node.kind == "Name") {
    const std::string *name = stringField(node, "id");
    if (!name)
      return std::nullopt;
    auto found = staticModuleSymbols.find(*name);
    if (found != staticModuleSymbols.end() && found->second.first == moduleName)
      return found->second.second;
    return std::nullopt;
  }
  if (node.kind != "Attribute")
    return std::nullopt;
  const parser::NodePtr *value = nodeField(node, "value");
  const std::string *attr = stringField(node, "attr");
  if (!value || !*value || (*value)->kind != "Name" || !attr)
    return std::nullopt;
  const std::string *moduleAlias = stringField(**value, "id");
  if (!moduleAlias)
    return std::nullopt;
  auto found = staticModules.find(*moduleAlias);
  if (found == staticModules.end())
    return std::nullopt;
  if (found->second == moduleName)
    return *attr;
  if (found->second == "lython") {
    std::optional<std::pair<std::string, std::string>> symbol =
        lythonFacadeSymbol(*attr);
    if (symbol && symbol->first == moduleName)
      return symbol->second;
  }
  return std::nullopt;
}

std::optional<std::string>
Builder::Impl::primitiveTypeName(const parser::Node &node) const {
  if (std::optional<std::string> imported = staticSymbol(node, "lyrt.prim")) {
    if (primitiveTypes().count(*imported))
      return imported;
    return std::nullopt;
  }
  if (node.kind != "Name")
    return std::nullopt;
  const std::string *name = stringField(node, "id");
  if (name && primitiveTypes().count(*name))
    return *name;
  return std::nullopt;
}

std::optional<std::string>
Builder::Impl::lyrtBuiltinName(const parser::Node &node) const {
  if (std::optional<std::string> imported = staticSymbol(node, "lyrt")) {
    if (lyrtBuiltins().count(*imported))
      return imported;
    return std::nullopt;
  }
  if (node.kind != "Name")
    return std::nullopt;
  const std::string *name = stringField(node, "id");
  if (name && lyrtBuiltins().count(*name))
    return *name;
  return std::nullopt;
}

} // namespace lython::emitter
