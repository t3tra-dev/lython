#include "EmitterCore.h"
#include "EmitterSupport.h"
#include "TypeSystemSolver.h"

#include "AstAccess.h"
#include "AstSynth.h"
#include "ClosureAnalysis.h"
#include "Contracts.h"
#include "EmitterPyOps.h"
#include "JsHost.h"
#include "PyProtocols.h"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/SaveAndRestore.h"

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

// ⭐ A STATEMENT IN AN IMPORTED MODULE THAT WOULD HAVE RUN AT IMPORT. The walk
// below recognises DECLARATIONS -- defs, classes, imports and the constant
// bindings `bindSourceModuleLocals` reads -- and everything else fell through
// a bare `continue`, so a module-level `print(...)`, an `ITEMS.append(1)` or a
// loop that fills a table simply did not happen:
//
//     # o.py
//     print("side effect")
//     # main
//     import o        # printed nothing; CPython prints "side effect"
//
// A declaration is not a statement that runs, which is why the list is exactly
// the declarations plus the two shapes with no effect at all: a docstring (a
// bare string expression, which every ported stdlib module opens with) and
// `pass`.
//
// ⛔ REFUSED rather than executed. Running an imported module's body needs a
// module-init function and an import order to call it in, which is a mechanism
// this compiler does not have; the rule for what it cannot do is to say so at
// the earliest static boundary rather than answer as if the body were empty.
bool isImportTimeDeclaration(const parser::Node &statement) {
  llvm::StringRef kind(statement.kind);
  if (kind == "FunctionDef" || kind == "AsyncFunctionDef" ||
      kind == "ClassDef" || kind == "Import" || kind == "ImportFrom" ||
      kind == "Assign" || kind == "AnnAssign" || kind == "Pass")
    return true;
  if (kind != "Expr")
    return false;
  const parser::Node *value = ast::node(statement, "value");
  return value && value->kind == "Constant";
}

// The leaf spelling of a decorator, whatever shape it is written in. Kept
// local rather than shared with EmitterClasses.cpp's richer version: the only
// question here is whether the decorator is one of the markers that emit no
// code, and a marker is always a bare name or a dotted one.
llvm::StringRef importedDecoratorLeaf(const parser::Node &decorator) {
  const parser::Node *node = &decorator;
  if (node->kind == "Call")
    if (const parser::Node *callee = ast::node(*node, "func"))
      node = callee;
  if (node->kind == "Name")
    return ast::nameSpelling(*node);
  if (node->kind == "Attribute")
    if (std::optional<std::string_view> attr = ast::string(*node, "attr"))
      return llvm::StringRef(attr->data(), attr->size());
  return llvm::StringRef();
}

// ⭐ A DECORATOR ON AN IMPORTED DEF WAS DROPPED IN SILENCE, and this is the
// one shape in the family that ANSWERS rather than refusing:
//
//     # m.py
//     def twice(f): ...            # returns a wrapper that doubles
//     @twice
//     def scaled(n: int) -> int:
//         return n + 1
//     # main
//     import m
//     print(m.scaled(1))           # printed 2; CPython prints 4
//
// The same program in ONE file is right. `f = d(f)` is a module-level
// rebinding evaluated at the def's position in module flow, and an imported
// module has no flow to evaluate it in -- the emission walk called
// `emitCallableFunction` for the body and neither `checkDecorators` nor
// `applyFunctionDecorators` ran, so the decorator vanished and the
// UNDECORATED function answered under the decorated name.
//
// ⛔ Except the markers that emit no code at all (`@overload`, `@override`,
// `@final`, `@runtime_checkable`, `@native`): those constrain the checker, and
// dropping them drops nothing.
bool importedDecoratorEmitsNoCode(const parser::Node &decorator) {
  llvm::StringRef leaf = importedDecoratorLeaf(decorator);
  return leaf == "overload" || leaf == "override" || leaf == "final" ||
         leaf == "runtime_checkable" || leaf == "native";
}

void refuseImportTimeStatements(TypeSystem &types,
                                const std::vector<parser::NodePtr> &body,
                                llvm::StringRef sourceName,
                                std::vector<parser::Diagnostic> &diagnostics) {
  for (const parser::NodePtr &statement : body) {
    if (!statement)
      continue;
    // A module-level `if` is not itself a statement that runs when its test is
    // decidable here: `staticModuleStatements` replaces it with the branch
    // taken, and this walk asks the same question so the two agree about what
    // the module contains. One it cannot decide is a branch nobody chooses,
    // which is the silent drop again.
    if (statement->kind == "If") {
      const parser::Node *test = ast::node(*statement, "test");
      std::optional<bool> truth =
          test ? optionalStaticBranchTruth(*test, types, /*from=*/nullptr)
               : std::nullopt;
      if (!truth) {
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, statement->range.start,
            "a module-level 'if' in an imported module needs a test this "
            "compiler can decide: an imported module's body does not run, so "
            "neither branch would be taken",
            sourceName.str()});
        continue;
      }
      if (const auto *branch =
              ast::nodeList(*statement, *truth ? "body" : "orelse"))
        refuseImportTimeStatements(types, *branch, sourceName, diagnostics);
      continue;
    }
    if (isImportTimeDeclaration(*statement)) {
      if (statement->kind == "FunctionDef" ||
          statement->kind == "AsyncFunctionDef")
        if (const auto *decorators =
                ast::nodeList(*statement, "decorator_list")) {
          bool everyDecoratorIsALocalDef = true;
          for (const parser::NodePtr &decorator : *decorators)
            if (!decorator || decorator->kind != "Name")
              everyDecoratorIsALocalDef = false;
          for (const parser::NodePtr &decorator : *decorators) {
            if (!decorator || importedDecoratorEmitsNoCode(*decorator))
              continue;
            // ⭐ A plain NAME decorator is `f = d(f)`, which the module-global
            // initializer queue now runs at the start of `__main__`. Only the
            // spellings that are not one keep the refusal.
            if (everyDecoratorIsALocalDef)
              continue;
            diagnostics.push_back(parser::Diagnostic{
                parser::Severity::Error, statement->range.start,
                // ⛔ THE MECHANISM NOW EXISTS AND THIS STILL REFUSES, which
                // is a scope decision rather than a missing one: an imported
                // module's class bodies and module-global initializers run at
                // the start of `__main__` in import order (2026-09-06), and a
                // decorator application is exactly such an initializer --
                // `f = d(f)` into the cell the name reads. What it needs
                // before it can be queued is the DECORATED type, and that is
                // `inferExpr` over `d(f)`, which cannot answer until the
                // module's own defs are declared -- one pass later than where
                // the importer's binders hand out the name.
                "a decorator on a function in an imported module is not "
                "supported: an imported module's body does not run, so the "
                "decorator would never be applied and the undecorated "
                "function would answer under its name",
                sourceName.str()});
            break;
          }
        }
      continue;
    }
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, statement->range.start,
        "a module-level statement in an imported module is not supported: an "
        "imported module's body does not run, so this statement would be "
        "dropped in silence",
        sourceName.str()});
  }
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

// A stub's classes, as contracts the protocol table can answer member
// questions about: fields from `name: T`, methods from each `def` with
// `@overload` siblings as alternative signatures, bases by name. Nothing is
// emitted; a stub has no runtime of its own.
//
// ⛔ Static methods are not published unless the policy gives them the
// value as a receiver: the table keys a method by the receiver's class, and a
// static one has none of its own.
void declareStubClassContracts(TypeSystem &types, mlir::MLIRContext &context,
                               llvm::StringRef moduleName,
                               const std::vector<parser::NodePtr> &body,
                               const StubContractPolicy &policy) {
  mlir::Type any = types.any();
  auto read = [&](mlir::Type type) {
    return policy.anyResult && type == any ? policy.anyResult : type;
  };
  auto written = [&](mlir::Type type) {
    return policy.anyParameter && type == any ? policy.anyParameter : type;
  };
  for (const parser::NodePtr &statement : body) {
    if (!statement || !isTopLevelClass(*statement))
      continue;
    std::optional<std::string_view> name = ast::string(*statement, "name");
    const auto *classBody = ast::nodeList(*statement, "body");
    if (!name || !classBody)
      continue;
    std::string contractName = sourceModuleClassSymbol(moduleName, *name);
    py::protocols::ProtocolInfo info;
    if (const auto *bases = ast::nodeList(*statement, "bases"))
      for (const parser::NodePtr &base : *bases)
        if (base)
          if (auto contract = mlir::dyn_cast_if_present<py::ContractType>(
                  types.annotationType(base.get())))
            info.bases.push_back(py::protocols::ProtocolBase{
                py::contracts::manifestClassNameForContract(
                    contract.getContractName()),
                {}});
    if (!policy.commonBase.empty())
      info.bases.push_back(py::protocols::ProtocolBase{
          py::contracts::manifestClassNameForContract(policy.commonBase), {}});
    info.bases.push_back(py::protocols::ProtocolBase{
        py::contracts::manifestClassNameForContract("builtins.object"), {}});
    mlir::Type receiverType = types.contract(contractName);
    for (const parser::NodePtr &member : *classBody) {
      if (!member)
        continue;
      if (member->kind == "AnnAssign") {
        const parser::Node *target = ast::node(*member, "target");
        if (target && target->kind == "Name")
          if (mlir::Type type =
                  types.annotationType(ast::node(*member, "annotation")))
            info.fields[std::string(ast::nameSpelling(*target))] = read(type);
        continue;
      }
      if (member->kind != "FunctionDef")
        continue;
      std::optional<std::string_view> methodName = ast::string(*member, "name");
      if (!methodName)
        continue;
      bool isStatic = false;
      if (const auto *decorators = ast::nodeList(*member, "decorator_list"))
        for (const parser::NodePtr &decorator : *decorators)
          if (decorator && importedDecoratorLeaf(*decorator) == "staticmethod")
            isStatic = true;
      if (isStatic && !policy.staticMethodsTakeTheValue)
        continue;
      FunctionSignature signature =
          isStatic ? types.functionSignature(*member)
                   : types.functionSignature(*member, llvm::StringRef("self"),
                                             py::CallableType(), receiverType);
      if (!signature.publicCallable)
        continue;
      if (isStatic) {
        signature.positionalNames.insert(signature.positionalNames.begin(),
                                         "self");
        signature.positionalTypes.insert(signature.positionalTypes.begin(),
                                         receiverType);
        signature.positionalDefaults.insert(
            signature.positionalDefaults.begin(), false);
        ++signature.positionalOnlyCount;
      }
      for (mlir::Type &type : signature.positionalTypes)
        type = written(type);
      for (mlir::Type &type : signature.kwOnlyTypes)
        type = written(type);
      signature.callableVarargType = written(signature.callableVarargType);
      signature.varargType = written(signature.varargType);
      signature.resultType = read(signature.resultType);
      signature.publicResultType = read(signature.publicResultType);
      types.refreshCallable(signature);
      py::protocols::ProtocolMethod method;
      method.signature = signature.publicCallable;
      method.mayThrow = true;
      method.firstApplicable = true;
      info.methods[std::string(*methodName)].push_back(method);
    }
    py::protocols::Table::getMutable(context).registerClass(contractName,
                                                            std::move(info));
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
  // A source stdlib module is a module too, and the attribute check needs to
  // know it: without this, `os.nonexistent` and `time.nonexistent` fell
  // through to the dynamic read the manifest modules no longer take.
  types.noteImportedModuleName(localName);
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
    std::string local =
        (llvm::Twine(localName) + "." + llvm::StringRef(*name)).str();
    std::string canonical = sourceModuleFunctionSymbol(module, *name);
    // A DECORATED def is the wrapper, not the symbol: the cell holds what the
    // decorator returned, and binding the plain symbol here would answer the
    // undecorated body under the name.
    if (moduleGlobals.count(canonical)) {
      types.bindCanonicalSymbol(local, canonical, moduleGlobals[canonical]);
      continue;
    }
    FunctionSignature sig =
        importedFunctionSignature(*source, *body, *statement);
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
    // A host global of the same name is what `js.Object` READS, and the class
    // is what the annotation `js.Object` means.
    //
    // ⛔ The class as an annotation alias and not a class binding: a value
    // read prefers a class of its spelling, and `isinstance(v, js.Object)`
    // then handed the host a type object, which has no value to test with.
    if (isJsHostModule(*source) &&
        moduleGlobals.count(sourceModuleClassSymbol(module, *name))) {
      std::string global = sourceModuleClassSymbol(module, *name);
      types.bindCanonicalSymbol(local, global, moduleGlobals[global]);
      types.bindAnnotationTypeAlias(
          local, types.contract(sourceModuleClassSymbol(module, *name)));
      continue;
    }
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
      continue;
    }
    std::string globalName = (llvm::Twine(module) + "." + name).str();
    if (moduleGlobals.count(globalName)) {
      std::string local = (llvm::Twine(localName) + "." + name).str();
      types.bindCanonicalSymbol(local, globalName, moduleGlobals[globalName]);
    }
  }
  return true;
}

// The literal type for a double, spelled so that `widenLiteral` reads it back
// as a float: a decimal point is appended where `%.17g` produced none.
static mlir::Type floatLiteralType(TypeSystem &types, double value) {
  char buffer[40];
  std::snprintf(buffer, sizeof(buffer), "%.17g", value);
  std::string spelling(buffer);
  if (spelling.find('.') == std::string::npos &&
      spelling.find('e') == std::string::npos &&
      spelling.find('E') == std::string::npos &&
      spelling.find("inf") == std::string::npos &&
      spelling.find("nan") == std::string::npos)
    spelling += ".0";
  return types.literal(spelling);
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
  // ⭐ A NEGATIVE LITERAL IS A UnaryOp OVER ONE, not a Constant, so `LIMIT = -1`
  // in an imported module did not resolve while `LIMIT = 1` beside it did --
  // for an int as much as for a float. The sign is folded here; the spellings
  // below already carry one.
  bool negated = false;
  while (constantNode->kind == "UnaryOp") {
    const parser::Node *op = ast::node(*constantNode, "op");
    bool minus = ast::isOperator(op, "USub");
    if (!minus && !ast::isOperator(op, "UAdd"))
      return std::nullopt;
    negated = negated != minus;
    constantNode = ast::node(*constantNode, "operand");
    if (!constantNode)
      return std::nullopt;
  }
  if (constantNode->kind != "Constant")
    return std::nullopt;
  if (negated) {
    if (auto number = ast::integer(*constantNode, "value"))
      return types.literal(std::to_string(-*number));
    if (auto number = ast::floating(*constantNode, "value"))
      return floatLiteralType(types, -*number);
    return std::nullopt;
  }
  if (auto text = ast::string(*constantNode, "value"))
    return types.literal("\"" + std::string(*text) + "\"");
  if (auto flag = ast::boolean(*constantNode, "value"))
    return types.literal(*flag ? "True" : "False");
  if (auto number = ast::integer(*constantNode, "value"))
    return types.literal(std::to_string(*number));
  // ⭐ AND `None`, which the three arms above left out. An imported module's
  // `NOTHING = None` did not resolve -- "module 'm' has no attribute
  // 'NOTHING' that resolves statically" -- while `FLAG = True` beside it did,
  // and the literal channel has carried the None spelling all along:
  // `widenLiteral` maps it to the none type and `emitLiteralTypeConstant`
  // materializes it.
  //
  if (ast::isNoneField(*constantNode, "value"))
    return types.literal("None");
  // ⭐ AND A FLOAT, which the note here used to say was not this function's to
  // make: "a 1.5 spelling would widen to INT, because every reader of a literal
  // spelling that is not True/False/None/quoted takes it for one". That was
  // true of `widenLiteral`, and it is the reader that has been taught the
  // difference -- the spelling carries a decimal point, an exponent, or a
  // non-finite name, and nothing else in the tree writes such a spelling.
  if (auto number = ast::floating(*constantNode, "value"))
    return floatLiteralType(types, *number);
  return std::nullopt;
}

// ⭐ AN IMPORTED MODULE'S CONSTANT TRAVELS AS A LITERAL, and a container has
// no literal spelling. A module that merely DEFINES one still imports; a
// function in it that READS one did not, and the sentence it got was
// "unresolved name 'ITEMS'" pointing at a line where the name is plainly in
// scope:
//
//     # lib.py
//     ITEMS: list[int] = [1, 2, 3]
//     def total() -> int:
//         return len(ITEMS)      # unresolved name 'ITEMS'
//
// An imported module has no executed body (a module-level STATEMENT in one is
// refused outright), so there is no cell for the name to read; the same three
// lines in the MAIN module compile, because there the annotated assignment IS
// a cell.
std::string
ModuleEmitter::importedModuleBindingReason(llvm::StringRef name) const {
  if (!activeSourceModuleNode)
    return {};
  const auto *body = ast::nodeList(*activeSourceModuleNode, "body");
  if (!body)
    return {};
  for (const parser::NodePtr &statement : *body) {
    if (!statement)
      continue;
    const parser::Node *target = nullptr;
    if (statement->kind == "AnnAssign") {
      target = ast::node(*statement, "target");
    } else if (statement->kind == "Assign") {
      if (const auto *targets = ast::nodeList(*statement, "targets"))
        if (targets->size() == 1)
          target = targets->front().get();
    }
    if (!target || target->kind != "Name" ||
        llvm::StringRef(ast::nameSpelling(*target)) != name)
      continue;
    return "'" + name.str() +
           "' is assigned at the top level of this imported module, but its "
           "type is not one a module global can hold: a scalar travels as a "
           "literal and a container gets a cell only when its element type is "
           "resolved. Annotate it, or define it in the importing module";
  }
  return {};
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
  // `from js import Math` is `globalThis.Math`, Pyodide's reading: the object,
  // where the stub's class of the same name only says what the object is.
  if (isJsHostModule(*source))
    if (std::string global = (llvm::Twine(module) + "." + exportedName).str();
        moduleGlobals.count(global)) {
      types.bindCanonicalSymbol(localName, global, moduleGlobals[global]);
      // `def f(p: URLSearchParams)` names the class the global constructs.
      if (py::protocols::Table::get(context).lookup(global))
        types.bindAnnotationTypeAlias(localName, types.contract(global));
      return true;
    }
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
    std::string canonical = sourceModuleFunctionSymbol(module, exportedName);
    if (moduleGlobals.count(canonical)) {
      types.bindCanonicalSymbol(localName, canonical, moduleGlobals[canonical]);
      return true;
    }
    FunctionSignature sig =
        importedFunctionSignature(*source, *body, *statement);
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
  if (std::string globalName =
          (llvm::Twine(module) + "." + exportedName).str();
      moduleGlobals.count(globalName)) {
    types.bindCanonicalSymbol(localName, globalName, moduleGlobals[globalName]);
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

namespace {
// A type a module global can hold: the same set `collectModuleGlobals` slots
// for the main module, with every argument resolved so the cell has one
// runtime representation.
bool isStorableContainerType(mlir::Type type) {
  // A CALLABLE is a function OBJECT, which is what makes a module-level lambda
  // in a library reachable at all: `DOUBLE: Callable[[int], int] = lambda ...`
  // had no symbol to bind and no literal spelling, so it resolved from nowhere
  // on either side of the boundary.
  if (mlir::isa<py::CallableType>(type))
    return true;
  auto contract = mlir::dyn_cast_if_present<py::ContractType>(type);
  if (!contract)
    return false;
  llvm::StringRef name = contract.getContractName();
  if (name == "builtins.list" || name == "builtins.dict" ||
      name == "builtins.set" || name == "builtins.tuple" ||
      name == "builtins.frozenset") {
    if (contract.getArguments().empty())
      return false;
    for (mlir::Type argument : contract.getArguments()) {
      if (mlir::isa<py::UnionType>(argument))
        return false;
      auto element = mlir::dyn_cast_if_present<py::ContractType>(argument);
      if (!element || element.getContractName() == "builtins.object")
        return false;
    }
    return true;
  }
  // ⭐ AND A SCALAR WHOSE VALUE IS AN EXPRESSION. The literal channel carries
  // `LIMIT = 3600` because its TYPE is the value; `LIMIT = 60 * 60` has no
  // literal spelling, so it resolved from nowhere -- "module 'lib' has no
  // attribute 'RATIO' that resolves statically" for `RATIO = 1.0 / 4.0`, and
  // the same three lines in the MAIN module compile. The caller reaches this
  // only after the literal channel has declined, so nothing that used to fold
  // starts taking a cell.
  //
  // ⛔ NOT the erased top, and not a bare generic: a cell whose contract has
  // no arguments to describe it accepts every write and refuses every read,
  // which is the rule `collectModuleGlobals` lives under for the main module.
  if (name == "builtins.object")
    return false;
  return contract.getArguments().empty();
}
} // namespace

// ⭐ THE CONSTANTS AN IMPORTED MODULE CANNOT SPELL AS A LITERAL. Everything
// scalar rides the literal channel -- its TYPE carries the value, so the
// importer materializes it with no module state at all -- and a container has
// no such spelling, so `NAMES = ["a"]` resolved from nowhere:
//
//     # lib.py
//     NAMES = ["a", "b"]
//     def count() -> int: return len(NAMES)   # unresolved name 'NAMES'
//     # main.py
//     import lib
//     print(lib.NAMES)   # module 'lib' has no attribute 'NAMES' ...
//
// and the same three lines in the MAIN module compile, because there the
// assignment IS a module global cell. This gives it the same cell, named
// `mod.NAME`, filled at the start of `__main__` in import order -- the place
// an imported module's class bodies already run.
//
// ⛔ Only a CONSTANT whose type is fully resolved. An erased element (`[]`) or
// an `object` top has no storage the readers can agree on, which is the rule
// `collectModuleGlobals` lives under for the main module; those keep the
// refusal rather than a cell nothing can read.
void ModuleEmitter::collectImportedModuleGlobals() {
  for (const EmitOptions::SourceModule &source : options.sourceModules) {
    if (!source.moduleNode || source.isStub)
      continue;
    const auto *rawBody = ast::nodeList(*source.moduleNode, "body");
    if (!rawBody)
      continue;
    const std::vector<parser::NodePtr> body =
        staticModuleStatements(types, *rawBody);
    for (const parser::NodePtr &statement : body) {
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
      const parser::Node *value = ast::node(*statement, "value");
      if (!target || target->kind != "Name" || !value)
        continue;
      llvm::StringRef name = ast::nameSpelling(*target);
      if (sourceModuleLiteralConstant(types, body, name))
        continue; // the literal channel already carries it
      mlir::Type declared;
      if (statement->kind == "AnnAssign")
        declared = types.annotationType(ast::node(*statement, "annotation"));
      if (!declared)
        declared = types.widenLiteral(types.inferExpr(value));
      if (!isStorableContainerType(declared))
        continue;
      std::string globalName =
          (llvm::Twine(source.moduleName) + "." + name).str();
      if (moduleGlobals.count(globalName))
        continue;
      moduleGlobals[globalName] = declared;
      importedModuleGlobalInits.push_back(
          PendingModuleGlobalInit{value, globalName, &source});
    }
    // ⭐ A DECORATED def IS A MODULE GLOBAL TOO -- `f = d(f)` is exactly the
    // initializer this queue exists for. It was refused outright ("an imported
    // module's body does not run, so the decorator would never be applied"),
    // which was true until the queue existed.
    //
    // ⛔ The decorated TYPE is folded from the SIGNATURES, not inferred from
    // the call: a signature is computable with nothing bound, and inferring
    // `d(f)` would need the module's own scope, which is one pass later than
    // where the importer's binders hand out the name.
    llvm::StringSet<> moduleDefs;
    for (const parser::NodePtr &statement : body)
      if (statement && isTopLevelFunction(*statement))
        if (auto defName = ast::string(*statement, "name"))
          moduleDefs.insert(*defName);
    for (const parser::NodePtr &statement : body) {
      if (!statement || !isTopLevelFunction(*statement))
        continue;
      const auto *decorators = ast::nodeList(*statement, "decorator_list");
      auto defName = ast::string(*statement, "name");
      if (!decorators || decorators->empty() || !defName)
        continue;
      bool plain = true;
      for (const parser::NodePtr &decorator : *decorators)
        if (!decorator || decorator->kind != "Name" ||
            !moduleDefs.contains(ast::nameSpelling(*decorator)))
          plain = false;
      if (!plain)
        continue;
      mlir::Type applied = types.functionSignature(*statement).publicCallable;
      parser::NodePtr expression =
          synth::name(*defName, statement->range);
      for (const parser::NodePtr &decorator : llvm::reverse(*decorators)) {
        llvm::StringRef spelling = ast::nameSpelling(*decorator);
        const parser::Node *decoratorDef = nullptr;
        for (const parser::NodePtr &candidate : body)
          if (candidate && isTopLevelFunction(*candidate))
            if (auto candidateName = ast::string(*candidate, "name");
                candidateName && llvm::StringRef(*candidateName) == spelling)
              decoratorDef = candidate.get();
        if (!decoratorDef) {
          applied = mlir::Type();
          break;
        }
        applied = types.functionSignature(*decoratorDef).resultType;
        std::vector<parser::NodePtr> arguments;
        arguments.push_back(expression);
        expression = synth::call(synth::name(spelling, statement->range),
                                 std::move(arguments), statement->range);
      }
      if (!mlir::isa_and_nonnull<py::CallableType>(applied))
        continue;
      std::string globalName =
          (llvm::Twine(source.moduleName) + "." + *defName).str();
      if (moduleGlobals.count(globalName))
        continue;
      moduleGlobals[globalName] = applied;
      importedModuleGlobalInits.push_back(PendingModuleGlobalInit{
          expression.get(), globalName, &source, std::string(*defName)});
      synthesizedIteratorDefs.push_back(std::move(expression));
    }
  }
}

std::string
ModuleEmitter::importedModuleGlobalFor(llvm::StringRef binding) const {
  if (binding.empty() || !binding.contains('.'))
    return {};
  return moduleGlobals.count(binding) ? binding.str() : std::string();
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
    // Its own bodies read it too, and for an imported module there is no
    // executed module body to bind it at run time -- the cell is what stands
    // in for one.
    std::string globalName = (llvm::Twine(moduleName) + "." + name).str();
    if (moduleGlobals.count(globalName)) {
      types.bindCanonicalSymbol(name, globalName, moduleGlobals[globalName]);
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
  declareJsHostModule();
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
  // ⭐ THE IMPORTED MODULES' HIERARCHIES, RECORDED BEFORE ANYTHING IS EMITTED.
  // `collectTopLevelBindings` does this for the main module and the override
  // guard reads it -- but it walks only the main module, so a base-typed
  // reference to an IMPORTED class was told the hierarchy had no override and
  // the base's body was inlined:
  //
  //     # shapes.py: class Base: show() -> "B"  /  class Derived(Base): show() -> "D"
  //     xs: "list[shapes.Base]" = [shapes.Base(1), shapes.Derived(2)]
  //     print([x.show() for x in xs])
  //     # printed ['B1', 'B2']; CPython prints ['B1', 'D2']
  //
  // A silent wrong answer, and the same program written in ONE file is right.
  // The names are the CONTRACTS (`shapes.Base`), which is what a receiver's
  // type is spelled as by the time the guard asks.
  //
  // ⛔ Before the emission loop and not inside it: a module emitted later can
  // subclass one emitted earlier, and the answer must not depend on the order
  // the modules are walked -- which is the same reason the main module's pass
  // runs up front.
  for (const EmitOptions::SourceModule &source : options.sourceModules) {
    if (!source.moduleNode || source.isStub)
      continue;
    const auto *declarations = ast::nodeList(*source.moduleNode, "body");
    if (!declarations)
      continue;
    // ⭐ AND THE MODULE'S OWN IMPORT SPELLINGS, read from its AST. A base that
    // lives in ANOTHER module was recorded wrong in all three spellings:
    //
    //     # base.py: class Root:  show() -> "R"
    //     # mid.py:  import base
    //     #          class Middle(base.Root):  show() -> "M"
    //     xs: "list[base.Root]" = [base.Root(1), mid.Middle(2)]
    //     print([x.show() for x in xs])
    //     # printed ['R1', 'R2']; CPython prints ['R1', 'M2']
    //
    // A dotted base is an Attribute, which the walk below skipped outright,
    // and `from base import Root` gives a bare Name that the walk qualified
    // with the WRONG module -- `mid.Root`, a class nothing declares. Either
    // way `mid.Middle` had no recorded base, so the override guard was told
    // the hierarchy is flat and `isinstance(x, mid.Middle)` answered False for
    // a Middle.
    //
    // ⛔ Read from the AST rather than from the type system: this pass runs
    // before any module scope is pushed, precisely so that the answer does not
    // depend on the order the modules are walked, and nothing else knows yet
    // what `base` or `Root` mean inside this file.
    llvm::StringMap<std::string> moduleAliases;
    llvm::StringMap<std::string> importedClasses;
    for (const parser::NodePtr &statement : *declarations) {
      if (!statement)
        continue;
      const bool isFrom = statement->kind == "ImportFrom";
      if (statement->kind != "Import" && !isFrom)
        continue;
      std::optional<std::string_view> fromModule =
          isFrom ? ast::string(*statement, "module") : std::nullopt;
      if (isFrom && !fromModule)
        continue;
      const auto *names = ast::nodeList(*statement, "names");
      if (!names)
        continue;
      for (const parser::NodePtr &alias : *names) {
        if (!alias)
          continue;
        std::optional<std::string_view> name = ast::string(*alias, "name");
        if (!name || *name == "*")
          continue;
        std::optional<std::string_view> asname = ast::string(*alias, "asname");
        std::string local(asname ? *asname : *name);
        if (isFrom)
          importedClasses[local] =
              (llvm::Twine(*fromModule) + "." + llvm::StringRef(*name)).str();
        else
          moduleAliases[local] = std::string(*name);
      }
    }
    // The classes this module declares itself, so a base that is neither one
    // of them nor an import can be recognised as MANIFEST.
    llvm::StringSet<> ownClassNames;
    for (const parser::NodePtr &statement : *declarations)
      if (statement && statement->kind == "ClassDef")
        if (auto declared = ast::string(*statement, "name"))
          ownClassNames.insert(*declared);
    auto qualifyBase = [&](const parser::Node &base) -> std::string {
      if (base.kind == "Name") {
        llvm::StringRef spelling = ast::nameSpelling(base);
        auto imported = importedClasses.find(spelling);
        if (imported != importedClasses.end())
          return imported->second;
        // ⭐ A MANIFEST BASE IS NOT THIS MODULE'S CLASS. Qualifying every bare
        // name with the module recorded `class MyErr(Exception)` as deriving
        // from `lib_err.Exception`, a class nothing declares -- so every
        // hierarchy question about an imported exception walked into a dead
        // end and `isinstance(e, MyErr)` on an `Exception`-typed value folded
        // to False, compiling the handler away. The same class written in ONE
        // file was right, because there the base is recorded as written.
        if (!ownClassNames.contains(spelling))
          return spelling.str();
        return sourceModuleClassSymbol(source.moduleName, spelling);
      }
      std::string dotted = ast::qualifiedName(&base);
      if (dotted.empty())
        return {};
      auto [head, rest] = llvm::StringRef(dotted).split('.');
      if (rest.empty())
        return {};
      if (auto alias = moduleAliases.find(head); alias != moduleAliases.end())
        return (llvm::Twine(alias->second) + "." + rest).str();
      // `from pkg import mod` binds a MODULE under a leaf name too, and a base
      // written through it (`mod.X`) is the same question one level down.
      if (auto nested = importedClasses.find(head);
          nested != importedClasses.end())
        return (llvm::Twine(nested->second) + "." + rest).str();
      return {};
    };
    for (const parser::NodePtr &statement : *declarations) {
      if (!statement || statement->kind != "ClassDef")
        continue;
      auto name = ast::string(*statement, "name");
      if (!name)
        continue;
      std::string qualified =
          sourceModuleClassSymbol(source.moduleName, *name);
      auto &bases = declaredClassBases[qualified];
      if (const auto *baseNodes = ast::nodeList(*statement, "bases"))
        for (const parser::NodePtr &base : *baseNodes) {
          if (!base)
            continue;
          std::string baseName = qualifyBase(*base);
          if (!baseName.empty())
            bases.push_back(std::move(baseName));
        }
      // ⛔ AND THE TYPE SYSTEM'S COPY, which is a different map with the same
      // content: `declaredSubclassOfType` reads it, and that is what decides
      // whether the dispatcher's `isinstance(recv, Derived)` arm survives.
      // With only the emitter's map filled, the guard built a dispatcher whose
      // every arm then folded to AlwaysFalse -- the same wrong answer, now
      // reached through a helper.
      types.bindDeclaredBases(qualified, bases);
      if (getenv("LYTHON_PROBE_BASES")) {
        llvm::errs() << "[probe] class " << qualified << " bases:";
        for (const std::string &b : bases)
          llvm::errs() << " " << b;
        llvm::errs() << "\n";
      }
      auto &methods = declaredClassMethods[qualified];
      auto &attributes = declaredClassAttributes[qualified];
      if (const auto *classBody = ast::nodeList(*statement, "body"))
        for (const parser::NodePtr &member : *classBody) {
          if (!member)
            continue;
          if (member->kind == "FunctionDef" ||
              member->kind == "AsyncFunctionDef") {
            if (auto methodName = ast::string(*member, "name"))
              methods.insert(*methodName);
            continue;
          }
          if (member->kind == "AnnAssign") {
            if (const parser::Node *target = ast::node(*member, "target"))
              if (target->kind == "Name")
                attributes.insert(ast::nameSpelling(*target));
            continue;
          }
          if (member->kind == "Assign")
            if (const auto *targets = ast::nodeList(*member, "targets"))
              for (const parser::NodePtr &target : *targets)
                if (target && target->kind == "Name")
                  attributes.insert(ast::nameSpelling(*target));
        }
    }
  }
  for (const EmitOptions::SourceModule &source : options.sourceModules) {
    if (!source.moduleNode)
      continue;
    const auto *rawBody = ast::nodeList(*source.moduleNode, "body");
    if (!rawBody)
      continue;
    std::string savedSourceName = sourceName;
    std::string savedPackageName = activePackageName;
    sourceName =
        source.sourceName.empty() ? source.moduleName : source.sourceName;
    activePackageName = source.packageName;
    llvm::SaveAndRestore<const parser::Node *> savedSourceModuleNode(
        activeSourceModuleNode, source.moduleNode);
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
      if (!isJsHostModule(source))
        declareStubClassContracts(types, context, source.moduleName,
                                  staticModuleStatements(types, *rawBody),
                                  StubContractPolicy{});
      activePackageName = std::move(savedPackageName);
      sourceName = std::move(savedSourceName);
      continue;
    }
    // ⭐ THE SAME FLATTENING EVERY BINDER IN THIS FILE DOES. A module-level
    // `if` whose test is statically decidable IS one branch's statements, and
    // `bindSourceModuleLocals` reads the module that way -- but this EMISSION
    // loop read the raw body, so a def inside such a branch had its name bound
    // and its body never emitted:
    //
    //     # q.py: if sys.platform == "win32": def sep(): ... else: def sep(): ...
    //     import q
    //     print(q.sep())      # error: unresolved runtime binding 'q.sep'
    //
    // A lowering sentence for a program CPython runs, and the disagreement is
    // between two walks of one module -- so the fix is to walk it once the
    // same way, not to teach this loop a second recogniser for `If`.
    const std::vector<parser::NodePtr> body =
        staticModuleStatements(types, *rawBody);
    // ⛔ Over the RAW body, because the flattener DROPS what it cannot decide
    // and the refusal is precisely for what gets dropped.
    refuseImportTimeStatements(types, *rawBody, sourceName, diagnostics);
    // ⭐ THE SAME DECLARE-THEN-DEFINE ORDER THE MAIN MODULE TAKES. Two sibling
    // subclasses in an IMPORTED module that both recurse through the base read
    // as "static type shapes.Right does not provide manifest method 'show'":
    // the first one's body needs the second's method bindings, which its own
    // `emitClassContract` fills, and this walk emitted bodies as it went.
    //
    // ⛔ The queue is drained INSIDE this module's scope, not with the main
    // module's: the bodies are typed against `sourceName`, the module's own
    // import scope and its isolated type scopes, all of which are restored
    // below. A hierarchy split across two imported modules therefore keeps the
    // old answer -- one module at a time is what this walk can promise.
    llvm::SaveAndRestore<bool> deferBodies(deferClassMethodBodies, true);
    std::vector<DeferredMethodBody> outerDeferred;
    outerDeferred.swap(deferredMethodBodies);
    for (const parser::NodePtr &statement : body) {
      if (!statement || !isTopLevelClass(*statement))
        continue;
      std::size_t classDiagnosticStart = diagnostics.size();
      // ⛔ AN ENUM CLASS IN AN IMPORTED MODULE IS REFUSED HERE, where the
      // program can read the reason. `desugarEnumClasses` runs on the MAIN
      // module only, so an imported one kept its `Enum` base and reached the
      // dialect verifier as "'py.class' op unknown base class 'Enum'" -- the
      // compiler's own sentence for a class CPython has no trouble with.
      //
      // ⛔ Not desugared here instead: a desugared member is an INSTANCE held
      // as a class attribute, and an imported class attribute is carried by
      // the literal channel, which has no form for one. The refusal is the
      // honest half until that channel can hold an object.
      if (enumBaseKind(*statement)) {
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, statement->range.start,
            "an Enum class in an imported module is not supported: its members "
            "are instances, and an imported class attribute is carried as a "
            "compile-time constant; define the enum in the main module"});
        continue;
      }
      if (std::optional<std::string_view> name =
              ast::string(*statement, "name")) {
        std::string classSymbol =
            sourceModuleClassSymbol(source.moduleName, *name);
        if (genericClasses.count(classSymbol)) {
          // A specialization's bodies are typed inside the scope that binds
          // its type parameters; see the same guard in the main module's
          // puller.
          llvm::SaveAndRestore<bool> emitNow(deferClassMethodBodies, false);
          drainGenericClassSpecializations(classSymbol);
        } else {
          // `&source` is what says its attribute initializers will be run:
          // an imported module's body does not, so they are queued for the
          // start of `__main__`, in import order.
          emitClassContract(*statement, classSymbol, &source);
        }
      }
      for (std::size_t index = classDiagnosticStart; index < diagnostics.size();
           ++index)
        if (diagnostics[index].filename.empty())
          diagnostics[index].filename = sourceName;
    }
    for (const parser::NodePtr &statement : body) {
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
      } else {
        continue;
      }
      for (std::size_t index = diagnosticStart; index < diagnostics.size();
           ++index)
        if (diagnostics[index].filename.empty())
          diagnostics[index].filename = sourceName;
    }
    {
      std::size_t bodyDiagnosticStart = diagnostics.size();
      emitDeferredMethodBodies();
      for (std::size_t index = bodyDiagnosticStart; index < diagnostics.size();
           ++index)
        if (diagnostics[index].filename.empty())
          diagnostics[index].filename = sourceName;
    }
    deferredMethodBodies.swap(outerDeferred);
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
    // An alias is only an alias while the module does not rebind it.
    llvm::StringSet<> boundOnce = singleAssignmentNames(moduleNode);
    // ⭐ A SECOND PASS FOR TYPE ALIASES, because an alias may name a class the
    // first pass has not reached yet -- `W = Widget` above `class Widget` is
    // legal Python only in an annotation, and that is exactly what an alias is
    // for. Aliases themselves are bound in SOURCE order, so `A = str; B = A`
    // resolves through the first.
    for (const parser::NodePtr &statement : *body) {
      if (!statement)
        continue;
      // PEP 695: `type Name = str`.
      if (statement->kind == "TypeAlias") {
        const parser::Node *target = ast::node(*statement, "name");
        const parser::Node *value = ast::node(*statement, "value");
        if (target && target->kind == "Name" && types.namesAType(value))
          types.bindAnnotationTypeAlias(ast::nameSpelling(*target),
                                        types.annotationType(value));
        continue;
      }
      // ⭐ AND AN ANNOTATED ONE. `CLS = Widget` binds the alias and
      // `CLS: type[Widget] = Widget` did not, so writing the annotation that
      // says what the name IS broke the program: "unresolved name 'CLS'" at
      // every read inside a function, where the unannotated line works. The
      // annotation agrees with the walk rather than replacing it.
      const parser::Node *aliasTarget = nullptr;
      if (statement->kind == "AnnAssign") {
        aliasTarget = ast::node(*statement, "target");
      } else if (statement->kind == "Assign") {
        const auto *targets = ast::nodeList(*statement, "targets");
        if (targets && targets->size() == 1)
          aliasTarget = targets->front().get();
      }
      if (!aliasTarget || aliasTarget->kind != "Name")
        continue;
      const parser::Node *value = ast::node(*statement, "value");
      if (!types.namesAType(value))
        continue;
      types.bindAnnotationTypeAlias(ast::nameSpelling(*aliasTarget),
                                    types.annotationType(value));
      // ⭐ AN ALIAS OF A CLASS IS ALSO THE CLASS AS A VALUE. `W = Widget` then
      // `cls = W` inside a function was "unresolved name 'W'", while
      // `cls = Widget` on the same line works: a class NAME is predeclared and
      // emits its type object, and a module global holding a class is a plain
      // global this compiler gives no storage to. The same is what
      // `Err = ValueError` needs for `raise Err(...)` and `except Err`.
      //
      // ⛔ This was tried once and dropped: `bindClass` here is UNSCOPED, so
      // the alias beat a PARAMETER of the same spelling and `t = A` at module
      // scope made `def build_b(t: type[B]) -> B: return t(n)` construct an A.
      // What made it safe is the shadowing rule that came after -- a name bound
      // to a type object or a callable now outranks a class of its spelling in
      // both the emitter's call path and the inference -- so the parameter wins
      // where it should. cases/type_object_representation is the case that
      // caught it and the one that pins it now.
      if (value && value->kind == "Name") {
        llvm::StringRef aliasedName = ast::nameSpelling(*value);
        llvm::StringRef aliasName = ast::nameSpelling(*aliasTarget);
        if (std::optional<mlir::Type> aliased = types.lookupClass(aliasedName))
          types.bindClass(aliasName, *aliased);
        // ⛔ AND THE SPELLING TOO, because the class binding is not enough for
        // every class. `str`, `int` and `bool` are intercepted by name BEFORE
        // the class-instantiation path -- `str(x)` is `__str__` dispatch, not
        // construction -- so an alias of one reached that path and said
        // "builtins.str has manifest method '__init__' but no signature that
        // accepts ...". A source class and an exception class have no such
        // interception and are served by the class binding above; recording
        // both costs nothing and lets whichever path claims the call first be
        // right.
        if (boundOnce.contains(aliasName))
          builtinValueAliases[aliasName] = aliasedName.str();
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
    if (genericClasses.count(name)) {
      // ⛔ A SPECIALIZATION'S BODIES ARE NEVER DEFERRED. They are typed inside
      // the scope that binds the class's type parameters to this
      // instantiation's arguments, so emitting them later types `T` as itself
      // -- "static type list[builtins.T] does not provide manifest method
      // 'append'", which is what two generic goldens said when the queue took
      // them.
      llvm::SaveAndRestore<bool> emitNow(deferClassMethodBodies, false);
      drainGenericClassSpecializations(name);
    } else {
      emitClassContract(*statement);
    }
  };

  // ⭐ EVERY CLASS IS DECLARED BEFORE ANY METHOD BODY IS EMITTED. Two sibling
  // subclasses that both call a method through the base could not be compiled
  // in any order:
  //
  //     class Expr:  def show(self) -> str: ...
  //     class Add(Expr):  def show(self): return self.a.show() + ...
  //     class Mul(Expr):  def show(self): return self.a.show() + ...
  //     # 'Mul.show' is used before 'Mul' is defined
  //
  // and swapping Add and Mul only swaps which of the two is refused, because
  // each one's body needs the dispatcher over the base, and the dispatcher
  // needs every subclass's method bindings. Those bindings are registered by
  // `emitClassContract` BEFORE it emits any body, so declaring every class
  // first is enough -- the bodies are queued and drained below.
  //
  // ⛔ This is NOT "emit all classes before all functions", which the ⛔ above
  // rejects because a class body may reference a module-level function. What
  // runs early here is the DECLARATION; the bodies still run last, after the
  // function declarations, so both orders hold at once.
  {
    llvm::SaveAndRestore<bool> deferBodies(deferClassMethodBodies, true);
    if (const auto *body = ast::nodeList(moduleNode, "body")) {
      for (const parser::NodePtr &statement : *body) {
        if (!statement || statement->kind != "ClassDef")
          continue;
        if (auto name = ast::string(*statement, "name"))
          emitClassNow(*name);
      }
      for (const parser::NodePtr &statement : *body) {
        if (!statement)
          continue;
        if (statement->kind == "FunctionDef" ||
            statement->kind == "AsyncFunctionDef") {
          emitNamedClassesFirst(*statement);
          emitFunctionDecl(*statement);
        }
      }
    }
  }
  // Stub-declared and never-walked generics still owe their specializations.
  drainGenericClassSpecializations();
  emitDeferredMethodBodies();
}

bool isJsHostModule(const EmitOptions::SourceModule &source) {
  return source.isEmbedded && source.isStub && source.moduleName == "js";
}

// The host's module is declared before any import is bound, because every
// name in it -- `from js import console` -- is a host global whose type has to
// exist when the import names it.
//
// Its globals are what Pyodide's are: `globalThis[name]`. The stub declares
// some at module level (`console: Console`) and the rest as Window's
// properties (`Math`, `JSON`), which is the object globalThis is.
void ModuleEmitter::declareJsHostModule() {
  const EmitOptions::SourceModule *host = nullptr;
  for (const EmitOptions::SourceModule &source : options.sourceModules)
    if (isJsHostModule(source))
      host = &source;
  if (!host || !host->moduleNode)
    return;
  module->setAttr(py::kJsHostModuleAttr, mlir::UnitAttr::get(&context));
  const auto *rawBody = ast::nodeList(*host->moduleNode, "body");
  if (!rawBody)
    return;
  // ⛔ The stub's own diagnostics are not the program's: it is generated from
  // TypeScript, and what it spells that this compiler cannot read types as
  // `object` at the stub's use sites, which is where it is the program's
  // business.
  parser::Diagnostics programAnnotationDiagnostics =
      types.takeAnnotationDiagnostics();
  {
    TypeSystem::ScopeIsolation isolation = types.isolateScopes();
    auto moduleScope = types.pushScope();
    bindModuleImportScope(*host->moduleNode, /*diagnoseUnsupported=*/false);
    const std::vector<parser::NodePtr> body =
        staticModuleStatements(types, *rawBody);
    bindSourceClassLocals(types, host->moduleName, body);
    // Aliases in source order, twice: the stub spells some before the types
    // they name. Bound only while the stub is read, then taken back.
    llvm::SmallVector<std::string, 64> aliases;
    for (int pass = 0; pass < 2; ++pass)
      for (const parser::NodePtr &statement : body) {
        if (!statement || statement->kind != "TypeAlias")
          continue;
        const parser::Node *target = ast::node(*statement, "name");
        const parser::Node *value = ast::node(*statement, "value");
        if (!target || target->kind != "Name" || !types.namesAType(value))
          continue;
        llvm::StringRef alias = ast::nameSpelling(*target);
        types.bindAnnotationTypeAlias(alias, types.annotationType(value));
        if (pass == 0)
          aliases.push_back(alias.str());
      }
    StubContractPolicy policy{
        types.contract(py::kJsProxyContract), types.object(),
        /*staticMethodsTakeTheValue=*/true, py::kJsProxyContract.str()};
    declareStubClassContracts(types, context, host->moduleName, body, policy);
    auto declareGlobal = [&](llvm::StringRef name, mlir::Type type) {
      if (!type)
        return;
      if (type == types.any())
        type = policy.anyResult;
      std::string global = (llvm::Twine(host->moduleName) + "." + name).str();
      if (!moduleGlobals.count(global))
        moduleGlobals[global] = type;
    };
    for (const parser::NodePtr &statement : body) {
      if (!statement || statement->kind != "AnnAssign")
        continue;
      const parser::Node *target = ast::node(*statement, "target");
      if (target && target->kind == "Name")
        declareGlobal(
            ast::nameSpelling(*target),
            types.annotationType(ast::node(*statement, "annotation")));
    }
    const py::protocols::Table &table = py::protocols::Table::get(context);
    llvm::SmallVector<std::string, 16> pending{
        sourceModuleClassSymbol(host->moduleName, "Window")};
    llvm::StringSet<> visited;
    while (!pending.empty()) {
      std::string className = pending.pop_back_val();
      if (!visited.insert(className).second)
        continue;
      const py::protocols::ProtocolInfo *info = table.lookup(className);
      if (!info)
        continue;
      for (const auto &[field, type] : info->fields)
        declareGlobal(field, type);
      for (const py::protocols::ProtocolBase &base : info->bases)
        pending.push_back(base.name);
    }
    for (const std::string &alias : aliases)
      types.unbindAnnotationTypeAlias(alias);
  }
  types.takeAnnotationDiagnostics();
  types.restoreAnnotationDiagnostics(std::move(programAnnotationDiagnostics));
}

bool ModuleEmitter::isJsHostValueType(mlir::Type type) const {
  return isJsHostType(type, types);
}

// `el.offsetParent` is `Element | None`: the host hands back whichever it
// has, and the program gets a value of the union it was typed with. The read
// is retyped to the host value as it comes; each member is then tested for
// in turn and the value converted to the first it is -- None, then the
// scalars, then a host object, which takes whatever is left.
//
// ⛔ Here and not in the lowering: the union is built by the same branch
// merge every conditional value takes, which is the shape the ownership
// verifier follows; a union assembled from runtime tests inside one lowered
// op would be one it has never seen.
Value ModuleEmitter::adaptJsHostResult(const parser::Node &anchor,
                                       mlir::Operation *op, Value declared) {
  auto unionType = mlir::dyn_cast_if_present<py::UnionType>(declared.type);
  if (!unionType)
    return declared;
  struct Arm {
    mlir::Type member;
    llvm::StringRef test;
    llvm::StringRef conversion;
  };
  llvm::SmallVector<Arm, 4> arms;
  std::optional<Arm> object;
  for (mlir::Type member : unionType.getMemberTypes()) {
    if (member == types.none())
      arms.insert(arms.begin(), Arm{member, "__ly_js_is_none__", ""});
    else if (member == types.boolType())
      arms.push_back(Arm{member, "__ly_js_is_bool__", "__ly_js_as_bool__"});
    else if (member == types.intType())
      arms.push_back(Arm{member, "__ly_js_is_int__", "__ly_js_as_int__"});
    else if (member == types.floatType())
      arms.push_back(Arm{member, "__ly_js_is_float__", "__ly_js_as_float__"});
    else if (member == types.strType())
      arms.push_back(Arm{member, "__ly_js_is_str__", "__ly_js_as_str__"});
    else if (isJsHostValueType(member) && !object)
      object = Arm{member, "", "__ly_js_as_proxy__"};
    else {
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, anchor.range.start,
          "a JavaScript value typed " + typeText(declared.type) +
              " cannot be read: its members must be None, bool, int, float, "
              "str or one JavaScript class"});
      return declared;
    }
  }
  if (object)
    arms.push_back(*object);
  mlir::Type raw = types.contract(py::kJsProxyContract);
  op->getResult(0).setType(raw);
  // The call's selected signature says what it returns, and is checked
  // against the result's type.
  if (auto call = mlir::dyn_cast<py::CallOp>(op))
    if (auto callable =
            mlir::dyn_cast<py::CallableType>(call.getCallContract())) {
      auto retyped = py::CallableType::get(
          &context, callable.getPositionalTypes(), callable.getKwOnlyTypes(),
          callable.getVarargType(), callable.getKwargType(), {raw},
          callable.getPositionalNames(), callable.getKwOnlyNames(),
          callable.getPositionalDefaults(), callable.getKwOnlyDefaults(),
          callable.getVarargName(), callable.getKwargName(),
          callable.getPositionalOnlyCount());
      call.setCallContractAttr(mlir::TypeAttr::get(callProtocolFor(retyped)));
    }
  Value host{op->getResult(0), raw};
  auto internal = [&](llvm::StringRef method, mlir::Type result) -> Value {
    auto callable =
        py::CallableType::get(&context, {raw}, {}, {}, {}, {result});
    Value positional = emitPack({});
    Value names = emitPack({});
    Value values = emitPack({});
    auto call =
        py::CallOp::create(builder, loc(anchor), mlir::TypeRange{result},
                           callProtocolFor(callable), host.value,
                           positional.value, names.value, values.value);
    call->setAttr("ly.bound_method", builder.getStringAttr(method));
    return Value{call.getResults().front(), result};
  };
  std::function<mlir::Value(std::size_t)> emitArm = [&](std::size_t index) {
    const Arm &arm = arms[index];
    auto convert = [&]() -> mlir::Value {
      Value converted = arm.conversion.empty()
                            ? emitNone(anchor)
                            : internal(arm.conversion, arm.member);
      return coerceValue(converted, declared.type, anchor).value;
    };
    if (index + 1 == arms.size())
      return convert();
    mlir::Value condition =
        emitBoolValue(internal(arm.test, types.boolType()), anchor);
    return emitValueDiamond(loc(anchor), condition, declared.type, convert,
                            [&] { return emitArm(index + 1); });
  };
  return Value{emitArm(0), declared.type};
}

// An imported function's signature as its importer sees it: read in the
// module's OWN scope -- its classes and the names it imports -- which is the
// scope its definition is emitted in.
//
// ⛔ Not the classes alone, which is what it was: `def make() -> Thing` in a
// module that imports Thing typed the call as returning `builtins.Thing`, a
// contract nothing declares, while the definition said `b.Thing` -- so
// `a.make(3).n` read a field of nothing ("attr.get object type has no class
// schema").
//
// ⛔ A module already being read here falls back to the classes alone: two
// modules that import each other would otherwise read each other forever.
FunctionSignature ModuleEmitter::importedFunctionSignature(
    const EmitOptions::SourceModule &source,
    const std::vector<parser::NodePtr> &body, const parser::Node &function) {
  if (!importedSignatureScopes.insert(source.moduleName).second)
    return sourceModuleFunctionSignature(types, source.moduleName, body,
                                         function, source.isStub);
  auto done = llvm::make_scope_exit(
      [&] { importedSignatureScopes.erase(source.moduleName); });
  TypeSystem::ScopeIsolation isolation = types.isolateScopes();
  auto moduleScope = types.pushScope();
  bindModuleImportScope(*source.moduleNode, /*diagnoseUnsupported=*/false);
  return sourceModuleFunctionSignature(types, source.moduleName, body, function,
                                       source.isStub);
}

} // namespace lython::emitter
