#include "EmitterCore.h"
#include "EmitterSupport.h"

#include "AstAccess.h"

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
    AlgorithmM &types, llvm::StringRef moduleName,
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
    AlgorithmM &types, llvm::StringRef moduleName,
    const std::vector<parser::NodePtr> &body, const parser::Node &function,
    bool isStub) {
  (void)isStub;
  auto classScope = types.pushScope();
  bindSourceClassLocals(types, moduleName, body);
  return types.functionSignature(function);
}

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
staticModuleStatements(AlgorithmM &types,
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

bool ModuleEmitter::isStubSourceModuleSymbol(llvm::StringRef symbol) const {
  std::pair<llvm::StringRef, llvm::StringRef> split = symbol.rsplit('.');
  if (split.first.empty() || split.second.empty())
    return false;
  const EmitOptions::SourceModule *source = lookupSourceModule(split.first);
  return source && source->isStub;
}

static std::optional<mlir::Type>
sourceModuleLiteralConstant(AlgorithmM &types,
                            const std::vector<parser::NodePtr> &body,
                            llvm::StringRef exportedName);

bool ModuleEmitter::bindSourceModuleNamespace(llvm::StringRef module,
                                              llvm::StringRef localName) {
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
        const EmitOptions::SourceModule *fromSource =
            resolved ? lookupSourceModule(*resolved) : nullptr;
        if (!fromSource)
          continue;
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
sourceModuleLiteralConstant(AlgorithmM &types,
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
                                         llvm::StringRef localName) {
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
        if (!fromSource)
          continue;
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
        emitCallableFunction(
            *statement, sourceModuleFunctionSymbol(source.moduleName, *name),
            sig, {}, /*isLambda=*/false);
      } else if (isTopLevelClass(*statement)) {
        std::optional<std::string_view> name = ast::string(*statement, "name");
        if (!name)
          continue;
        emitClassContract(*statement,
                          sourceModuleClassSymbol(source.moduleName, *name));
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
        if (auto name = ast::string(*statement, "name"))
          types.bindClass(*name, types.contract(*name));
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
      if (!asname && llvm::StringRef(*name).contains('.')) {
        if (bindSourceModuleNamespace(llvm::StringRef(*name),
                                      llvm::StringRef(*name))) {
          std::pair<llvm::StringRef, llvm::StringRef> split =
              llvm::StringRef(*name).split('.');
          bindSourceModuleNamespace(split.first, split.first);
          continue;
        }
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
      if (diagnoseUnsupported)
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, alias->range.start,
            "star import is not supported by the static emitter"});
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

void ModuleEmitter::emitTopLevelDeclarations() {
  if (const auto *body = ast::nodeList(moduleNode, "body")) {
    for (const parser::NodePtr &statement : *body) {
      if (!statement)
        continue;
      if (statement->kind == "FunctionDef" ||
          statement->kind == "AsyncFunctionDef")
        emitFunctionDecl(*statement);
      else if (statement->kind == "ClassDef")
        emitClassContract(*statement);
    }
  }
}

} // namespace lython::emitter
