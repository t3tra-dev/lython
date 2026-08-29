#include "EmitterCore.h"
#include "EmitterSupport.h"
#include "TypeSystemSolver.h"

#include "AstAccess.h"
#include "PyProtocols.h"

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace lython::emitter {
namespace {

bool isTopLevelFunction(const parser::Node &statement) {
  return statement.kind == "FunctionDef" ||
         statement.kind == "AsyncFunctionDef";
}

bool isTopLevelClass(const parser::Node &statement) {
  return statement.kind == "ClassDef";
}

std::string sourceModuleFunctionSymbol(llvm::StringRef module,
                                       llvm::StringRef function) {
  return (llvm::Twine(module) + "." + function).str();
}

std::string sourceModuleClassSymbol(llvm::StringRef module,
                                    llvm::StringRef className) {
  return (llvm::Twine(module) + "." + className).str();
}

void bindSourceClassLocals(
    TypeSystem &types, llvm::StringRef moduleName,
    const std::vector<parser::NodePtr> &body) {
  for (const parser::NodePtr &statement : body) {
    if (!statement || !isTopLevelClass(*statement))
      continue;
    std::optional<std::string_view> name = ast::string(*statement, "name");
    if (!name)
      continue;
    types.bindClass(*name, types.contract(sourceModuleClassSymbol(moduleName,
                                                                  *name)));
  }
}

FunctionSignature sourceModuleFunctionSignature(
    TypeSystem &types, llvm::StringRef moduleName,
    const std::vector<parser::NodePtr> &body, const parser::Node &function,
    bool isStub) {
  (void)isStub;
  auto classScope = types.pushScope();
  bindSourceClassLocals(types, moduleName, body);
  return types.functionSignature(function);
}

// Module-level `alias = other_name` (single Name target, Name value):
// CPython Lib modules publish aliases this way (bisect = bisect_right).
std::optional<std::string_view>
moduleAliasTarget(const std::vector<parser::NodePtr> &body,
                  llvm::StringRef name) {
  for (const parser::NodePtr &statement : body) {
    if (!statement || statement->kind != "Assign")
      continue;
    const auto *targets = ast::nodeList(*statement, "targets");
    if (!targets || targets->size() != 1 || !targets->front() ||
        targets->front()->kind != "Name" ||
        llvm::StringRef(ast::nameSpelling(*targets->front())) != name)
      continue;
    const parser::Node *value = ast::node(*statement, "value");
    if (value && value->kind == "Name" &&
        llvm::StringRef(ast::nameSpelling(*value)) != name)
      return ast::nameSpelling(*value);
  }
  return std::nullopt;
}

// Alias chains are finite in real modules; the bound only breaks
// pathological `a = b; b = a` cycles.
constexpr unsigned kMaxAliasDepth = 8;

std::optional<llvm::SmallVector<std::string, 8>>
staticAllExportNames(const parser::Node &moduleNode) {
  const auto *body = ast::nodeList(moduleNode, "body");
  if (!body)
    return std::nullopt;
  for (const parser::NodePtr &statement : *body) {
    if (!statement || statement->kind != "Assign")
      continue;
    const auto *targets = ast::nodeList(*statement, "targets");
    if (!targets || targets->size() != 1 || !targets->front() ||
        targets->front()->kind != "Name" ||
        ast::nameSpelling(*targets->front()) != "__all__")
      continue;

    const parser::Node *value = ast::node(*statement, "value");
    if (!value || (value->kind != "List" && value->kind != "Tuple"))
      return std::nullopt;
    const auto *elts = ast::nodeList(*value, "elts");
    if (!elts)
      return std::nullopt;

    llvm::SmallVector<std::string, 8> names;
    for (const parser::NodePtr &element : *elts) {
      if (!element || element->kind != "Constant")
        return std::nullopt;
      std::optional<std::string_view> name = ast::string(*element, "value");
      if (!name || name->empty())
        return std::nullopt;
      names.push_back(std::string(*name));
    }
    return names;
  }
  return std::nullopt;
}

std::optional<std::string_view> importAliasLocalName(const parser::Node &alias) {
  std::optional<std::string_view> name = ast::string(alias, "name");
  if (!name || *name == "*")
    return std::nullopt;
  std::optional<std::string_view> asname = ast::string(alias, "asname");
  return asname ? asname : name;
}

// Module-level `import X as member` (or `import X`): the member IS a module,
// which is how os.py publishes `path` (`import posixpath as path`). Returns X.
std::optional<std::string_view>
moduleMemberModule(const std::vector<parser::NodePtr> &body,
                   llvm::StringRef member) {
  for (const parser::NodePtr &statement : body) {
    if (!statement || statement->kind != "Import")
      continue;
    const auto *names = ast::nodeList(*statement, "names");
    if (!names)
      continue;
    for (const parser::NodePtr &alias : *names) {
      if (!alias)
        continue;
      std::optional<std::string_view> imported = ast::string(*alias, "name");
      std::optional<std::string_view> local = importAliasLocalName(*alias);
      if (imported && local && llvm::StringRef(*local) == member)
        return imported;
    }
  }
  return std::nullopt;
}


std::string joinModuleName(llvm::StringRef prefix, llvm::StringRef suffix) {
  if (prefix.empty())
    return suffix.str();
  if (suffix.empty())
    return prefix.str();
  return (llvm::Twine(prefix) + "." + suffix).str();
}

std::optional<std::string>
resolveRelativeModule(llvm::StringRef packageName, std::int64_t level,
                      std::optional<std::string_view> module) {
  if (level <= 0)
    return module ? std::optional<std::string>{std::string(*module)}
                  : std::nullopt;
  if (packageName.empty())
    return std::nullopt;

  llvm::SmallVector<llvm::StringRef, 8> parts;
  packageName.split(parts, '.');
  if (level > static_cast<std::int64_t>(parts.size()))
    return std::nullopt;

  std::string resolved;
  std::size_t keep = parts.size() - static_cast<std::size_t>(level - 1);
  for (std::size_t index = 0; index < keep; ++index) {
    if (!resolved.empty())
      resolved += ".";
    resolved += parts[index].str();
  }
  if (module && !module->empty())
    resolved = joinModuleName(resolved, llvm::StringRef(*module));
  return resolved;
}

// Module bodies seen through the static import machinery: top-level
// statements plus the statements of the statically TAKEN branch of any
// module-level `if` whose test folds (the platform-switch idiom CPython's
// Lib modules use, e.g. `if name == 'posix': from posix import *`).
// Unfoldable module-level ifs contribute no static bindings.
std::vector<parser::NodePtr>
staticModuleStatements(TypeSystem &types,
                       const std::vector<parser::NodePtr> &body) {
  std::vector<parser::NodePtr> out;
  out.reserve(body.size());
  for (const parser::NodePtr &statement : body) {
    if (!statement)
      continue;
    if (statement->kind == "If") {
      const parser::Node *test = ast::node(*statement, "test");
      std::optional<bool> truth =
          test ? optionalStaticBranchTruth(*test, types, /*from=*/nullptr)
               : std::nullopt;
      if (!truth)
        continue;
      const auto *branch =
          ast::nodeList(*statement, *truth ? "body" : "orelse");
      if (!branch)
        continue;
      std::vector<parser::NodePtr> nested =
          staticModuleStatements(types, *branch);
      out.insert(out.end(), nested.begin(), nested.end());
      continue;
    }
    out.push_back(statement);
  }
  return out;
}

} // namespace

const EmitOptions::SourceModule *
ModuleEmitter::lookupSourceModule(llvm::StringRef module) const {
  for (const EmitOptions::SourceModule &source : options.sourceModules)
    if (source.moduleName == module && source.moduleNode)
      return &source;
  return nullptr;
}

const EmitOptions::SourceModule *
ModuleEmitter::sourceModuleForClass(llvm::StringRef className) const {
  std::pair<llvm::StringRef, llvm::StringRef> split = className.rsplit('.');
  if (split.first.empty() || split.second.empty())
    return nullptr;
  return lookupSourceModule(split.first);
}

bool ModuleEmitter::isStubSourceModuleSymbol(llvm::StringRef symbol) const {
  std::pair<llvm::StringRef, llvm::StringRef> split = symbol.rsplit('.');
  if (split.first.empty() || split.second.empty())
    return false;
  const EmitOptions::SourceModule *source = lookupSourceModule(split.first);
  return source && source->isStub;
}

static std::optional<mlir::Type>
sourceModuleLiteralConstant(TypeSystem &types,
                            const std::vector<parser::NodePtr> &body,
                            llvm::StringRef exportedName);

// A module member that is itself a module (`import posixpath as path` inside
// os.py) nests one namespace inside another. Real module graphs nest a step or
// two; the bound only breaks a mutual-import cycle (a.py `import b as x`,
// b.py `import a as y`), which would otherwise recurse forever.
static constexpr unsigned kMaxNamespaceDepth = 4;

bool ModuleEmitter::bindSourceModuleNamespace(llvm::StringRef module,
                                              llvm::StringRef localName,
                                              unsigned namespaceDepth) {
  const EmitOptions::SourceModule *source = lookupSourceModule(module);
  if (!source)
    return false;
  // The module namespace symbol itself is a pure lookup root, not a runtime
  // receiver: qualified members are bound below through canonical
  // `localName.attr` symbols carrying their real callable/class contracts.
  // The `object` top here is an AGENTS.md namespace placeholder; a bare module
  // value carries no protocol contract, so any attempt to dispatch on it (call,
  // len, iteration) is rejected for lack of evidence rather than erased.
  types.bindCanonicalSymbol(localName, module, types.object());
  const auto *rawBody = ast::nodeList(*source->moduleNode, "body");
  if (!rawBody)
    return true;
  const std::vector<parser::NodePtr> flattened =
      staticModuleStatements(types, *rawBody);
  const std::vector<parser::NodePtr> *body = &flattened;
  for (const parser::NodePtr &statement : *body) {
    if (!statement || !isTopLevelFunction(*statement))
      continue;
    std::optional<std::string_view> name = ast::string(*statement, "name");
    if (!name)
      continue;
    FunctionSignature sig = sourceModuleFunctionSignature(
        types, module, *body, *statement, source->isStub);
    std::string local =
        (llvm::Twine(localName) + "." + llvm::StringRef(*name)).str();
    std::string canonical = sourceModuleFunctionSymbol(module, *name);
    types.bindCanonicalSymbol(local, canonical, sig.publicCallable);
    continue;
  }
  for (const parser::NodePtr &statement : *body) {
    if (!statement || statement->kind != "Assign")
      continue;
    const auto *targets = ast::nodeList(*statement, "targets");
    const parser::Node *value = ast::node(*statement, "value");
    if (!targets || targets->size() != 1 || !targets->front() ||
        targets->front()->kind != "Name" || !value || value->kind != "Name")
      continue;
    llvm::StringRef aliasName = ast::nameSpelling(*targets->front());
    std::string local = (llvm::Twine(localName) + "." + aliasName).str();
    bindSourceModuleName(module, aliasName, local);
  }
  for (const parser::NodePtr &statement : *body) {
    if (!statement || statement->kind != "ImportFrom")
      continue;
    const auto *names = ast::nodeList(*statement, "names");
    if (!names)
      continue;
    for (const parser::NodePtr &alias : *names) {
      if (!alias)
        continue;
      std::optional<std::string_view> aliasName = ast::string(*alias, "name");
      if (aliasName && *aliasName == "*") {
        // Star reexport: every static __all__ name of the source module the
        // star pulls from becomes a member of this namespace.
        std::int64_t level = ast::integer(*statement, "level").value_or(0);
        std::optional<std::string_view> fromModule =
            ast::string(*statement, "module");
        std::optional<std::string> resolved =
            resolveRelativeModule(source->packageName, level, fromModule);
        if (!resolved)
          continue;
        const EmitOptions::SourceModule *fromSource =
            lookupSourceModule(*resolved);
        if (!fromSource) {
          // The star pulls from a native manifest (os.py's `from posix import
          // *`), so the export list is the manifest's public names rather than
          // an __all__ list.
          bindNativeModuleNamespaceStar(*resolved, localName);
          continue;
        }
        std::optional<llvm::SmallVector<std::string, 8>> exports =
            staticAllExportNames(*fromSource->moduleNode);
        if (!exports)
          continue;
        for (const std::string &starName : *exports) {
          std::string local =
              (llvm::Twine(localName) + "." + starName).str();
          bindSourceModuleName(*resolved, starName, local);
        }
        continue;
      }
      std::optional<std::string_view> exported =
          importAliasLocalName(*alias);
      if (!exported)
        continue;
      std::string local =
          (llvm::Twine(localName) + "." + llvm::StringRef(*exported)).str();
      bindSourceModuleReexport(*source, llvm::StringRef(*exported),
                               llvm::StringRef(local));
    }
  }
  // `import M as A` inside a source module publishes M's whole namespace as
  // the member `A`: this is how CPython's os.py exposes os.path (`import
  // posixpath as path`), and the recursion is what makes `os.path.join`
  // resolve to the canonical `posixpath.join` symbol. Not a runtime module
  // object — the flat `localName.attr` symbol table carries every member.
  if (namespaceDepth < kMaxNamespaceDepth) {
    for (const parser::NodePtr &statement : *body) {
      if (!statement || statement->kind != "Import")
        continue;
      const auto *names = ast::nodeList(*statement, "names");
      if (!names)
        continue;
      for (const parser::NodePtr &alias : *names) {
        if (!alias)
          continue;
        std::optional<std::string_view> imported = ast::string(*alias, "name");
        std::optional<std::string_view> member = importAliasLocalName(*alias);
        if (!imported || !member || imported->find('.') != std::string::npos)
          continue;
        std::string nested =
            (llvm::Twine(localName) + "." + llvm::StringRef(*member)).str();
        bindSourceModuleNamespace(llvm::StringRef(*imported), nested,
                                  namespaceDepth + 1);
      }
    }
  }
  for (const parser::NodePtr &statement : *body) {
    if (!statement || !isTopLevelClass(*statement))
      continue;
    std::optional<std::string_view> name = ast::string(*statement, "name");
    if (!name)
      continue;
    std::string local =
        (llvm::Twine(localName) + "." + llvm::StringRef(*name)).str();
    types.bindClass(local, types.contract(sourceModuleClassSymbol(module, *name)));
  }
  for (const parser::NodePtr &statement : *body) {
    if (!statement ||
        (statement->kind != "AnnAssign" && statement->kind != "Assign"))
      continue;
    const parser::Node *target =
        statement->kind == "AnnAssign"
            ? ast::node(*statement, "target")
            : (ast::nodeList(*statement, "targets") &&
                       ast::nodeList(*statement, "targets")->size() == 1
                   ? ast::nodeList(*statement, "targets")->front().get()
                   : nullptr);
    if (!target || target->kind != "Name")
      continue;
    llvm::StringRef name = ast::nameSpelling(*target);
    if (std::optional<mlir::Type> literal =
            sourceModuleLiteralConstant(types, *body, name)) {
      std::string local = (llvm::Twine(localName) + "." + name).str();
      types.bindSymbol(local, *literal);
    }
  }
  return true;
}

// A top-level `NAME: T = <literal>` / `NAME = <literal>` assigned exactly once
// in a source module is a static literal constant: its literal type fully
// determines the value, so importers materialize it without module state.
static std::optional<mlir::Type>
sourceModuleLiteralConstant(TypeSystem &types,
                            const std::vector<parser::NodePtr> &body,
                            llvm::StringRef exportedName) {
  const parser::Node *constantNode = nullptr;
  unsigned assignments = 0;
  for (const parser::NodePtr &statement : body) {
    if (!statement)
      continue;
    const parser::Node *target = nullptr;
    const parser::Node *value = nullptr;
    if (statement->kind == "AnnAssign" || statement->kind == "AugAssign") {
      target = ast::node(*statement, "target");
      value = ast::node(*statement, "value");
    } else if (statement->kind == "Assign") {
      const auto *targets = ast::nodeList(*statement, "targets");
      if (targets && targets->size() == 1)
        target = targets->front().get();
      value = ast::node(*statement, "value");
    } else {
      continue;
    }
    if (!target || target->kind != "Name" ||
        llvm::StringRef(ast::nameSpelling(*target)) != exportedName)
      continue;
    ++assignments;
    constantNode = statement->kind == "AugAssign" ? nullptr : value;
  }
  if (assignments != 1 || !constantNode)
    return std::nullopt;
  // Platform-switch ternaries (`"nt" if sys.platform == "win32" else
  // "posix"`) fold to the taken arm: the test compares target string
  // literals, the same compile-time switch idiom function bodies use.
  while (constantNode->kind == "IfExp") {
    const parser::Node *test = ast::node(*constantNode, "test");
    std::optional<bool> truth =
        test ? optionalStaticBranchTruth(*test, types, /*from=*/nullptr)
             : std::nullopt;
    if (!truth)
      return std::nullopt;
    constantNode = ast::node(*constantNode, *truth ? "body" : "orelse");
    if (!constantNode)
      return std::nullopt;
  }
  if (constantNode->kind != "Constant")
    return std::nullopt;
  if (auto text = ast::string(*constantNode, "value"))
    return types.literal("\"" + std::string(*text) + "\"");
  if (auto flag = ast::boolean(*constantNode, "value"))
    return types.literal(*flag ? "True" : "False");
  if (auto number = ast::integer(*constantNode, "value"))
    return types.literal(std::to_string(*number));
  return std::nullopt;
}

bool ModuleEmitter::bindSourceModuleName(llvm::StringRef module,
                                         llvm::StringRef exportedName,
                                         llvm::StringRef localName,
                                         unsigned aliasDepth) {
  const EmitOptions::SourceModule *source = lookupSourceModule(module);
  if (!source)
    return false;
  if (exportedName == "*")
    return false;
  const auto *rawBody = ast::nodeList(*source->moduleNode, "body");
  if (!rawBody)
    return false;
  const std::vector<parser::NodePtr> flattened =
      staticModuleStatements(types, *rawBody);
  const std::vector<parser::NodePtr> *body = &flattened;
  for (const parser::NodePtr &statement : *body) {
    if (!statement || !isTopLevelFunction(*statement))
      continue;
    std::optional<std::string_view> name = ast::string(*statement, "name");
    if (!name || llvm::StringRef(*name) != exportedName)
      continue;
    FunctionSignature sig = sourceModuleFunctionSignature(
        types, module, *body, *statement, source->isStub);
    std::string canonical = sourceModuleFunctionSymbol(module, exportedName);
    types.bindCanonicalSymbol(localName, canonical, sig.publicCallable);
    return true;
  }
  for (const parser::NodePtr &statement : *body) {
    if (!statement || !isTopLevelClass(*statement))
      continue;
    std::optional<std::string_view> name = ast::string(*statement, "name");
    if (!name || llvm::StringRef(*name) != exportedName)
      continue;
    types.bindClass(localName,
                    types.contract(sourceModuleClassSymbol(module, *name)));
    return true;
  }
  if (std::optional<mlir::Type> literal =
          sourceModuleLiteralConstant(types, *body, exportedName)) {
    types.bindSymbol(localName, *literal);
    return true;
  }
  if (aliasDepth < kMaxAliasDepth)
    if (std::optional<std::string_view> aliased =
            moduleAliasTarget(*body, exportedName))
      if (bindSourceModuleName(module, llvm::StringRef(*aliased), localName,
                               aliasDepth + 1))
        return true;
  // `import posixpath as path` publishes a module as the member `path`, so
  // `from os import *` (whose __all__ lists "path") binds a whole namespace
  // here, not a single value.
  for (const parser::NodePtr &statement : *body) {
    if (!statement || statement->kind != "Import")
      continue;
    const auto *names = ast::nodeList(*statement, "names");
    if (!names)
      continue;
    for (const parser::NodePtr &alias : *names) {
      if (!alias)
        continue;
      std::optional<std::string_view> imported = ast::string(*alias, "name");
      std::optional<std::string_view> member = importAliasLocalName(*alias);
      if (!imported || !member || llvm::StringRef(*member) != exportedName)
        continue;
      if (bindSourceModuleNamespace(llvm::StringRef(*imported), localName))
        return true;
    }
  }
  if (bindSourceModuleReexport(*source, exportedName, localName))
    return true;
  return false;
}

bool ModuleEmitter::bindSourceModuleReexport(
    const EmitOptions::SourceModule &source, llvm::StringRef exportedName,
    llvm::StringRef localName) {
  if (!source.moduleNode)
    return false;
  const auto *rawBody = ast::nodeList(*source.moduleNode, "body");
  if (!rawBody)
    return false;
  const std::vector<parser::NodePtr> flattened =
      staticModuleStatements(types, *rawBody);
  const std::vector<parser::NodePtr> *body = &flattened;
  for (const parser::NodePtr &statement : *body) {
    if (!statement || statement->kind != "ImportFrom")
      continue;
    std::int64_t level = ast::integer(*statement, "level").value_or(0);
    std::optional<std::string_view> module = ast::string(*statement, "module");
    std::optional<std::string> resolvedModule =
        resolveRelativeModule(source.packageName, level, module);
    if (!resolvedModule)
      continue;
    const auto *names = ast::nodeList(*statement, "names");
    if (!names)
      continue;
    for (const parser::NodePtr &alias : *names) {
      if (!alias)
        continue;
      std::optional<std::string_view> importName = ast::string(*alias, "name");
      if (importName && *importName == "*") {
        // `from M import *`: the name reexports when it is in M's __all__.
        const EmitOptions::SourceModule *fromSource =
            lookupSourceModule(*resolvedModule);
        if (!fromSource) {
          // M is a native manifest (os.py's `from posix import *`), which has
          // no __all__: the public-name convention is the export list, and the
          // manifest export itself is what the name binds to.
          if (!exportedName.empty() && exportedName.front() != '_' &&
              types.bindImportedName(*resolvedModule, exportedName, localName))
            return true;
          continue;
        }
        std::optional<llvm::SmallVector<std::string, 8>> exports =
            staticAllExportNames(*fromSource->moduleNode);
        if (!exports || !llvm::is_contained(*exports, exportedName.str()))
          continue;
        if (bindSourceModuleName(*resolvedModule, exportedName, localName))
          return true;
        continue;
      }
      std::optional<std::string_view> localExport =
          importAliasLocalName(*alias);
      if (!importName || !localExport ||
          llvm::StringRef(*localExport) != exportedName)
        continue;
      if (level != 0 && !module) {
        std::string submodule = joinModuleName(*resolvedModule, *importName);
        if (bindSourceModuleNamespace(submodule, localName))
          return true;
      }
      if (bindSourceModuleName(*resolvedModule, llvm::StringRef(*importName),
                               localName))
        return true;
      if (types.bindImportedName(*resolvedModule, llvm::StringRef(*importName),
                                 localName))
        return true;
    }
  }
  return false;
}

bool ModuleEmitter::bindSourceModuleStar(llvm::StringRef module,
                                         const parser::Node &anchor,
                                         bool diagnoseUnsupported) {
  const EmitOptions::SourceModule *source = lookupSourceModule(module);
  if (!source)
    return false;
  std::optional<llvm::SmallVector<std::string, 8>> exports =
      staticAllExportNames(*source->moduleNode);
  if (!exports) {
    if (diagnoseUnsupported)
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, anchor.range.start,
          "star import from '" + module.str() +
              "' requires a static __all__"});
    return true;
  }

  bool ok = true;
  for (const std::string &exported : *exports) {
    if (bindSourceModuleName(module, exported, exported))
      continue;
    ok = false;
    if (diagnoseUnsupported)
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, anchor.range.start,
          "star import from '" + module.str() +
              "' references unsupported export '" +
              sourceModuleFunctionSymbol(module, exported) + "'"});
  }
  return ok || diagnoseUnsupported;
}

// The public export names of a native manifest module: it declares no
// __all__, so the convention (no leading underscore) is the export list.
static llvm::SmallVector<std::string, 32>
nativeModuleStarNames(mlir::MLIRContext &context, llvm::StringRef module) {
  const py::protocols::Table &table = py::protocols::Table::get(context);
  llvm::SmallVector<std::string, 32> names;
  for (const std::string &name : table.moduleCallableExports(module))
    names.push_back(name);
  for (const auto &[exported, qualified] : table.moduleClassExports(module))
    names.push_back(exported);
  for (const std::string &name : table.moduleFloatConstantExports(module))
    names.push_back(name);
  for (const std::string &name : table.moduleIntConstantExports(module))
    names.push_back(name);
  for (const std::string &name : table.moduleStrConstantExports(module))
    names.push_back(name);
  llvm::sort(names);
  names.erase(llvm::unique(names), names.end());
  return names;
}

// `from <manifest> import *` inside a SOURCE module, seen from the module's
// importer: each re-exported name becomes a `localName.<name>` member, so
// `os.getcwd()` reaches posix.getcwd through os.py's star re-export the way
// CPython's os.py re-exports posix.
void ModuleEmitter::bindNativeModuleNamespaceStar(llvm::StringRef module,
                                                  llvm::StringRef localName) {
  for (const std::string &name : nativeModuleStarNames(context, module)) {
    if (name.empty() || name.front() == '_')
      continue;
    std::string local = (llvm::Twine(localName) + "." + name).str();
    types.bindImportedName(module, name, local);
  }
}

bool ModuleEmitter::bindNativeModuleStar(llvm::StringRef module,
                                         const parser::Node &anchor,
                                         bool diagnoseUnsupported) {
  const py::protocols::Table &table = py::protocols::Table::get(context);
  llvm::SmallVector<std::string, 32> names;
  for (const std::string &name : table.moduleCallableExports(module))
    names.push_back(name);
  for (const auto &[exported, qualified] : table.moduleClassExports(module))
    names.push_back(exported);
  for (const std::string &name : table.moduleFloatConstantExports(module))
    names.push_back(name);
  for (const std::string &name : table.moduleIntConstantExports(module))
    names.push_back(name);
  for (const std::string &name : table.moduleStrConstantExports(module))
    names.push_back(name);
  if (names.empty())
    return false;
  llvm::sort(names);
  names.erase(llvm::unique(names), names.end());
  bool ok = true;
  for (const std::string &name : names) {
    if (name.empty() || name.front() == '_')
      continue;
    if (types.bindImportedName(module, name, name))
      continue;
    ok = false;
    if (diagnoseUnsupported)
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, anchor.range.start,
          "star import from '" + module.str() +
              "' references unsupported export '" + module.str() + "." + name +
              "'"});
  }
  return ok || diagnoseUnsupported;
}

void ModuleEmitter::bindSourceModuleLocals(llvm::StringRef moduleName,
                                           const parser::Node &sourceModule,
                                           bool isStub) {
  const auto *rawBody = ast::nodeList(sourceModule, "body");
  if (!rawBody)
    return;
  const std::vector<parser::NodePtr> flattened =
      staticModuleStatements(types, *rawBody);
  const std::vector<parser::NodePtr> *body = &flattened;
  bindSourceClassLocals(types, moduleName, *body);
  for (const parser::NodePtr &statement : *body) {
    if (!statement)
      continue;
    if (isTopLevelFunction(*statement)) {
      std::optional<std::string_view> name = ast::string(*statement, "name");
      if (!name)
        continue;
      FunctionSignature sig = types.functionSignature(*statement);
      types.bindCanonicalSymbol(*name,
                                sourceModuleFunctionSymbol(moduleName, *name),
                                sig.publicCallable);
      continue;
    }
  }
  // Module-level literal constants and `alias = name` bindings are part of
  // the module's own scope too: function and method bodies read them
  // (imported modules have no executed module body to bind them at runtime,
  // so uses materialize the literal / resolve the alias statically).
  for (const parser::NodePtr &statement : *body) {
    if (!statement ||
        (statement->kind != "AnnAssign" && statement->kind != "Assign"))
      continue;
    const parser::Node *target =
        statement->kind == "AnnAssign"
            ? ast::node(*statement, "target")
            : (ast::nodeList(*statement, "targets") &&
                       ast::nodeList(*statement, "targets")->size() == 1
                   ? ast::nodeList(*statement, "targets")->front().get()
                   : nullptr);
    if (!target || target->kind != "Name")
      continue;
    llvm::StringRef name = ast::nameSpelling(*target);
    if (std::optional<mlir::Type> literal =
            sourceModuleLiteralConstant(types, *body, name)) {
      types.bindSymbol(name, *literal);
      continue;
    }
    if (moduleAliasTarget(*body, name))
      bindSourceModuleName(moduleName, name, name);
  }
}

void ModuleEmitter::bindModuleImportScope(const parser::Node &sourceModule,
                                          bool diagnoseUnsupported) {
  const auto *rawBody = ast::nodeList(sourceModule, "body");
  if (!rawBody)
    return;
  const std::vector<parser::NodePtr> flattened =
      staticModuleStatements(types, *rawBody);
  const std::vector<parser::NodePtr> *body = &flattened;
  for (const parser::NodePtr &statement : *body) {
    if (!statement)
      continue;
    if (statement->kind == "Import" || statement->kind == "ImportFrom")
      bindImportStatement(*statement, diagnoseUnsupported);
  }
}

void ModuleEmitter::predeclareSourceModules() {
  for (const EmitOptions::SourceModule &source : options.sourceModules) {
    if (!source.moduleNode)
      continue;
    bindSourceModuleNamespace(source.moduleName, source.moduleName);
    // Generic classes must be registered before ANY signature is resolved:
    // a `deque[int]` annotation is what allocates the specialization, and a
    // signature memoized against the unspecialized reading would never be
    // recomputed.
    if (source.isStub)
      continue;
    if (const auto *body = ast::nodeList(*source.moduleNode, "body"))
      for (const parser::NodePtr &statement : *body)
        if (statement && isTopLevelClass(*statement))
          if (auto name = ast::string(*statement, "name"))
            registerGenericClass(
                *statement, sourceModuleClassSymbol(source.moduleName, *name),
                &source);
  }
}

void ModuleEmitter::emitSourceModuleDeclarations() {
  for (const EmitOptions::SourceModule &source : options.sourceModules) {
    if (!source.moduleNode)
      continue;
    const auto *body = ast::nodeList(*source.moduleNode, "body");
    if (!body)
      continue;
    std::string savedSourceName = sourceName;
    std::string savedPackageName = activePackageName;
    sourceName =
        source.sourceName.empty() ? source.moduleName : source.sourceName;
    activePackageName = source.packageName;
    // The pair `emitInDefiningModuleScope` already uses for a specialization
    // emitted in its own module. This walk pushed a scope but left the
    // importer's below it, which is a scope that shadows rather than one that
    // isolates.
    ImporterModuleScope importerScope(*this);
    TypeSystem::ScopeIsolation isolation = types.isolateScopes();
    auto moduleScope = types.pushScope();
    std::size_t importDiagnosticStart = diagnostics.size();
    bindModuleImportScope(*source.moduleNode, /*diagnoseUnsupported=*/true);
    for (std::size_t index = importDiagnosticStart; index < diagnostics.size();
         ++index)
      if (diagnostics[index].filename.empty())
        diagnostics[index].filename = sourceName;
    bindSourceModuleLocals(source.moduleName, *source.moduleNode,
                           source.isStub);
    if (source.isStub) {
      activePackageName = std::move(savedPackageName);
      sourceName = std::move(savedSourceName);
      continue;
    }
    for (const parser::NodePtr &statement : *body) {
      if (!statement)
        continue;
      std::size_t diagnosticStart = diagnostics.size();
      if (isTopLevelFunction(*statement)) {
        std::optional<std::string_view> name = ast::string(*statement, "name");
        if (!name)
          continue;
        FunctionSignature sig = types.functionSignature(*statement);
        std::string canonical =
            sourceModuleFunctionSymbol(source.moduleName, *name);
        if (unboundStaticParameterCount(sig.publicCallable) != 0) {
          // Same monomorphization strategy as main-module generics: no
          // direct emission (the py ABI cannot carry a type parameter), one
          // specialization per ground instantiation demanded by a use site.
          // Registration is canonical-keyed so call sites reach it through
          // the import binding regardless of local spelling.
          GenericFunctionInfo &info = genericFunctions[canonical];
          info.node = statement.get();
          info.signature = sig;
          info.symbolBase = canonical;
          info.source = &source;
        } else {
          emitCallableFunction(*statement, canonical, sig, {},
                               /*isLambda=*/false);
          recordMonomorphicFunction(canonical, *statement, sig, canonical,
                                    &source);
        }
      } else if (isTopLevelClass(*statement)) {
        std::optional<std::string_view> name = ast::string(*statement, "name");
        if (!name)
          continue;
        // Generic classes were registered during predeclaration; the generic
        // itself is never emitted, and its specializations take its place at
        // this position so a later class can inherit from one.
        std::string classSymbol =
            sourceModuleClassSymbol(source.moduleName, *name);
        if (genericClasses.count(classSymbol))
          drainGenericClassSpecializations(classSymbol);
        else
          emitClassContract(*statement, classSymbol);
      } else {
        continue;
      }
      for (std::size_t index = diagnosticStart; index < diagnostics.size();
           ++index)
        if (diagnostics[index].filename.empty())
          diagnostics[index].filename = sourceName;
    }
    activePackageName = std::move(savedPackageName);
    sourceName = std::move(savedSourceName);
  }
}

void ModuleEmitter::predeclareTopLevel() {
  if (const auto *body = ast::nodeList(moduleNode, "body")) {
    for (const parser::NodePtr &statement : *body) {
      if (!statement)
        continue;
      if (statement->kind == "Import" || statement->kind == "ImportFrom") {
        bindImportStatement(*statement, /*diagnoseUnsupported=*/false);
        continue;
      }
      if (statement->kind == "ClassDef")
        if (auto name = ast::string(*statement, "name")) {
          types.bindClass(*name, types.contract(*name));
          registerGenericClass(*statement, *name, /*source=*/nullptr);
        }
      if (statement->kind == "Assign") {
        const auto *targets = ast::nodeList(*statement, "targets");
        if (!targets || targets->size() != 1 || !targets->front() ||
            targets->front()->kind != "Name")
          continue;
        std::optional<std::pair<mlir::IntegerType, std::int64_t>> primitive =
            primitiveIntegerConstantConstructor(ast::node(*statement, "value"),
                                                types);
        if (!primitive)
          continue;
        llvm::StringRef name = ast::nameSpelling(*targets->front());
        primitiveConstants[name] =
            PrimitiveConstant{primitive->first, primitive->second};
        types.bindSymbol(name, primitive->first);
      }
    }
  }
}

bool ModuleEmitter::bindImportStatement(const parser::Node &statement,
                                        bool diagnoseUnsupported) {
  if (statement.kind == "Import") {
    const auto *names = ast::nodeList(statement, "names");
    if (!names)
      return true;
    for (const parser::NodePtr &alias : *names) {
      if (!alias)
        continue;
      std::optional<std::string_view> name = ast::string(*alias, "name");
      if (!name)
        continue;
      std::optional<std::string_view> asname = ast::string(*alias, "asname");
      std::string local = importBindingName(*name, asname);
      // ⭐ FIXED 2026-08-19, and the cause was not here at all. `os` IS NOT A
      // SOURCE MODULE at this point was the true observation the old note
      // recorded; what it did not ask is WHY. The driver decides which stdlib
      // sources to compile from the import statements, and for a dotted name it
      // requested only prefixes that are PACKAGE DIRECTORIES -- so `import
      // os.path` requested nothing for `os`, os.py was never compiled, and
      // every repair attempted in this function was binding a module that did
      // not exist yet. Requesting each prefix that resolves to a source at all
      // (Frontend.cpp, appendDottedImportSourceRequests) is the whole fix, and
      // then the source-module branch below binds `os` the way `import os`
      // does.
      //
      // ⛔ Still unsupported, and a different mechanism: `import os.path as p`
      // and `from os.path import join` bind the SUBMODULE itself, which needs a
      // module value -- `path` is a name inside os.py's scope, not a module the
      // resolver knows.
      if (!asname && llvm::StringRef(*name).contains('.')) {
        if (bindSourceModuleNamespace(llvm::StringRef(*name),
                                      llvm::StringRef(*name))) {
          std::pair<llvm::StringRef, llvm::StringRef> split =
              llvm::StringRef(*name).split('.');
          bindSourceModuleNamespace(split.first, split.first);
          continue;
        }
        // `import os.path` binds ONLY `os` in CPython -- the submodule is
        // reached as an attribute of it -- so when the dotted name is not a
        // source module but its root is importable, importing the root IS the
        // statement, and nothing here binds the dotted name to anything.
        llvm::StringRef root = llvm::StringRef(*name).split('.').first;
        if (bindSourceModuleNamespace(root, root))
          continue;
        if (types.bindImportedModule(root, root))
          continue;
      }
      // ⭐ `import os.path as p` binds the SUBMODULE, not the root, and the
      // submodule is a name inside the root's own body (`import posixpath as
      // path`). Asking the root what it publishes under that name is what turns
      // the dotted spelling into one this emitter already has: a namespace.
      if (asname && llvm::StringRef(*name).contains('.')) {
        std::pair<llvm::StringRef, llvm::StringRef> split =
            llvm::StringRef(*name).rsplit('.');
        if (const EmitOptions::SourceModule *rootModule =
                lookupSourceModule(split.first))
          if (rootModule->moduleNode)
            if (const auto *rootBody = ast::nodeList(*rootModule->moduleNode,
                                                     "body"))
              if (std::optional<std::string_view> published =
                      moduleMemberModule(*rootBody, split.second))
                if (bindSourceModuleNamespace(llvm::StringRef(*published),
                                              llvm::StringRef(local)))
                  continue;
      }
      if (bindSourceModuleNamespace(llvm::StringRef(*name),
                                    llvm::StringRef(local))) {
        continue;
      }
      if (!types.bindImportedModule(llvm::StringRef(*name),
                                    llvm::StringRef(local)) &&
          diagnoseUnsupported) {
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, alias->range.start,
            "unsupported import '" + std::string(*name) + "'"});
      }
    }
    return true;
  }

  if (statement.kind != "ImportFrom")
    return false;

  std::int64_t level = ast::integer(statement, "level").value_or(0);
  std::optional<std::string_view> module = ast::string(statement, "module");
  // ⭐ `from __future__ import annotations` and its siblings are NO-OPS here,
  // and refusing them took the whole file with them. Every future feature
  // CPython still names is mandatory behaviour in 3.14 except `annotations`,
  // and that one only asks that annotations be treated as strings -- which is
  // what this compiler does with them anyway, now that a quoted annotation is
  // parsed as the annotation it spells. Binding nothing is the whole
  // implementation.
  //
  // ⛔ Named one by one rather than accepting the module: a future feature
  // this compiler has NOT implemented must still be refused, and there is no
  // way to tell the two apart from the module name.
  if (level == 0 && module && *module == "__future__") {
    static constexpr llvm::StringLiteral kInertFutures[] = {
        llvm::StringLiteral("annotations"),
        llvm::StringLiteral("absolute_import"),
        llvm::StringLiteral("division"),
        llvm::StringLiteral("generators"),
        llvm::StringLiteral("generator_stop"),
        llvm::StringLiteral("nested_scopes"),
        llvm::StringLiteral("print_function"),
        llvm::StringLiteral("unicode_literals"),
        llvm::StringLiteral("with_statement")};
    if (const auto *futureNames = ast::nodeList(statement, "names")) {
      bool everyOneInert = true;
      for (const parser::NodePtr &alias : *futureNames) {
        std::optional<std::string_view> name =
            alias ? ast::string(*alias, "name") : std::nullopt;
        everyOneInert =
            everyOneInert && name &&
            llvm::is_contained(kInertFutures, llvm::StringRef(*name));
      }
      if (everyOneInert)
        return true;
    }
  }
  std::optional<std::string> resolvedModule =
      resolveRelativeModule(activePackageName, level, module);
  if (!resolvedModule) {
    if (diagnoseUnsupported) {
      std::string message =
          level == 0 ? "from import requires a static module name"
                     : "relative import requires a static package context";
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, statement.range.start, std::move(message)});
    }
    return true;
  }
  const auto *names = ast::nodeList(statement, "names");
  if (!names)
    return true;
  for (const parser::NodePtr &alias : *names) {
    if (!alias)
      continue;
    std::optional<std::string_view> name = ast::string(*alias, "name");
    if (!name || *name == "*") {
      if (name && bindSourceModuleStar(*resolvedModule, *alias,
                                       diagnoseUnsupported))
        continue;
      if (name && bindNativeModuleStar(*resolvedModule, *alias,
                                       diagnoseUnsupported))
        continue;
      if (diagnoseUnsupported)
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, alias->range.start,
            "star import from '" + *resolvedModule +
                "' is not statically resolvable"});
      continue;
    }
    std::optional<std::string_view> asname = ast::string(*alias, "asname");
    llvm::StringRef local =
        asname ? llvm::StringRef(*asname) : llvm::StringRef(*name);
    if (bindSourceModuleName(*resolvedModule, llvm::StringRef(*name), local))
      continue;
    std::string submodule = joinModuleName(*resolvedModule, *name);
    if (bindSourceModuleNamespace(submodule, local))
      continue;
    // ⭐ `from os.path import join`: the module being imported FROM is itself a
    // member of another module (os publishes `path` as posixpath), so the name
    // resolves against what that member names. Same question as the `import
    // os.path as p` branch above, asked of the tail instead of the root.
    if (llvm::StringRef(*resolvedModule).contains('.')) {
      std::pair<llvm::StringRef, llvm::StringRef> split =
          llvm::StringRef(*resolvedModule).rsplit('.');
      if (const EmitOptions::SourceModule *rootModule =
              lookupSourceModule(split.first))
        if (rootModule->moduleNode)
          if (const auto *rootBody =
                  ast::nodeList(*rootModule->moduleNode, "body"))
            if (std::optional<std::string_view> published =
                    moduleMemberModule(*rootBody, split.second))
              if (bindSourceModuleName(llvm::StringRef(*published),
                                       llvm::StringRef(*name), local))
                continue;
    }
    if (!types.bindImportedName(*resolvedModule, llvm::StringRef(*name),
                                local) &&
        diagnoseUnsupported) {
      std::string importName = joinModuleName(*resolvedModule, *name);
      diagnostics.push_back(
          parser::Diagnostic{parser::Severity::Error, alias->range.start,
                             "unsupported import '" + importName + "'"});
    }
  }
  return true;
}

namespace {

// A `yield` anywhere in this function's own body -- nested defs, lambdas and
// classes have their own. Same rule `containsYieldExpression` applies inside
// EmitterExceptions.cpp; kept local because that one is file-static there.
bool functionBodyContainsYield(const parser::Node &node) {
  return ast::walk(&node, [&](const parser::Node &current) {
    if (current.kind == "Yield" || current.kind == "YieldFrom")
      return ast::Walk::Stop;
    // ⛔ The boundary applies to the CHILDREN only: the caller hands in the
    // def whose body this is, and skipping it would answer no every time.
    if (&current != &node &&
        (current.kind == "FunctionDef" || current.kind == "AsyncFunctionDef" ||
         current.kind == "Lambda" || current.kind == "ClassDef"))
      return ast::Walk::SkipChildren;
    return ast::Walk::Continue;
  });
}

} // namespace

void ModuleEmitter::emitTopLevelDeclarations() {
  // ⭐ A TOP-LEVEL GENERATOR'S YIELD TYPE IS RECOMPUTED HERE. `registerModule`
  // memoized every top-level signature before any class contract existed --
  // it has to run first, because a signature may name a class and the class's
  // bodies are typed against the function symbols. For an ordinary function
  // that order is fine: only its annotations matter. A generator's signature
  // also depends on its BODY, and a body reading a source class inferred
  // `builtins.object`:
  //
  //     class C:
  //         def __init__(self) -> None: self.n = 5
  //     def gen(c: C):
  //         yield c.n
  //     print(list(gen(C())))
  //     # runtime bundle for 'builtins.object' has 5 values, but ABI expects 1
  //
  // The same generator NESTED inside a function worked, because its signature
  // is computed during body emission, after the classes. Dropping the memo
  // makes the walk below recompute each one at the point it is declared, with
  // every class ABOVE it in the file published.
  //
  // ⛔ A generator textually BEFORE the class it reads (only reachable with a
  // string annotation) still gets the early answer: this respects source
  // order rather than emitting all classes first, because a class body may
  // reference a module-level function and reordering the two would trade this
  // defect for that one.
  if (const auto *declarations = ast::nodeList(moduleNode, "body"))
    for (const parser::NodePtr &statement : *declarations)
      if (statement && (statement->kind == "FunctionDef" ||
                        statement->kind == "AsyncFunctionDef") &&
          functionBodyContainsYield(*statement))
        types.forgetSignature(statement.get());
  // A class is emitted when it is first NEEDED rather than where it is
  // written. `class A` whose method returns `B(1)` used to be refused with
  // "static type B does not provide manifest method '__init__'": B's contract
  // registers inside its own emitClassContract, which had not run yet, so a
  // class -- or a function -- textually above the class it constructs could
  // not construct it. The iterable/iterator pair is the everyday shape of it:
  //
  //     class Range:
  //         def __iter__(self) -> "RangeIter": return RangeIter(self.stop)
  //     class RangeIter: ...
  //
  // The annotation resolves (predeclareTopLevel binds every class NAME up
  // front); only the members were missing.
  //
  // ⛔ NOT "emit all classes before all functions", which the ⛔ above rejects
  // for the right reason -- a class body may reference a module-level
  // function. Pulling forward only what a statement NAMES leaves every other
  // pair in source order.
  //
  // Erasing from `deferred` BEFORE emitting is the cycle guard: two classes
  // that construct each other resolve the first one's reference to nothing,
  // which is exactly the old behaviour for that pair and no worse.
  llvm::StringMap<const parser::Node *> deferred;
  if (const auto *body = ast::nodeList(moduleNode, "body"))
    for (const parser::NodePtr &statement : *body)
      if (statement && statement->kind == "ClassDef")
        if (auto name = ast::string(*statement, "name"))
          deferred[*name] = statement.get();

  std::function<void(llvm::StringRef)> emitClassNow;
  auto emitNamedClassesFirst = [&](const parser::Node &statement) {
    ast::walk(&statement, [&](const parser::Node &node) {
      if (node.kind == "Name")
        emitClassNow(ast::nameSpelling(node));
      return ast::Walk::Continue;
    });
  };
  emitClassNow = [&](llvm::StringRef name) {
    auto found = deferred.find(name);
    if (found == deferred.end())
      return;
    const parser::Node *statement = found->second;
    deferred.erase(found);
    emitNamedClassesFirst(*statement);
    if (genericClasses.count(name))
      drainGenericClassSpecializations(name);
    else
      emitClassContract(*statement);
  };

  if (const auto *body = ast::nodeList(moduleNode, "body")) {
    for (const parser::NodePtr &statement : *body) {
      if (!statement)
        continue;
      if (statement->kind == "FunctionDef" ||
          statement->kind == "AsyncFunctionDef") {
        emitNamedClassesFirst(*statement);
        emitFunctionDecl(*statement);
      } else if (statement->kind == "ClassDef") {
        auto name = ast::string(*statement, "name");
        if (!name)
          continue;
        emitClassNow(*name);
      }
    }
  }
  // Stub-declared and never-walked generics still owe their specializations.
  drainGenericClassSpecializations();
}

} // namespace lython::emitter
