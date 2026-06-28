#include "EmitterCore.h"
#include "EmitterSupport.h"

#include "AstAccess.h"

#include <optional>
#include <string>
#include <string_view>

namespace lython::emitter {

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
  if (level != 0 || !module) {
    if (diagnoseUnsupported)
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, statement.range.start,
          "relative import is not supported by the static emitter"});
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
      if (diagnoseUnsupported)
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, alias->range.start,
            "star import is not supported by the static emitter"});
      continue;
    }
    std::optional<std::string_view> asname = ast::string(*alias, "asname");
    llvm::StringRef local =
        asname ? llvm::StringRef(*asname) : llvm::StringRef(*name);
    if (!types.bindImportedName(llvm::StringRef(*module),
                                llvm::StringRef(*name), local) &&
        diagnoseUnsupported) {
      diagnostics.push_back(
          parser::Diagnostic{parser::Severity::Error, alias->range.start,
                             "unsupported import '" + std::string(*module) +
                                 "." + std::string(*name) + "'"});
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
