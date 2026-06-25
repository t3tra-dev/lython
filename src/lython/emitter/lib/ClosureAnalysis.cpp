#include "ClosureAnalysis.h"

#include "AstAccess.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"

namespace lython::emitter {
namespace {

bool hasContext(const parser::Node &node, llvm::StringRef expected) {
  const parser::Node *ctx = ast::node(node, "ctx");
  return ctx && ctx->kind == expected;
}

void collectTargetNames(const parser::Node *node, llvm::StringSet<> &names) {
  if (!node)
    return;
  if (node->kind == "Name") {
    names.insert(ast::nameSpelling(*node));
    return;
  }
  if (node->kind == "Tuple" || node->kind == "List") {
    if (const auto *elts = ast::nodeList(*node, "elts"))
      for (const parser::NodePtr &elt : *elts)
        collectTargetNames(elt.get(), names);
  }
}

void collectParameterNames(const parser::Node *arguments,
                           llvm::StringSet<> &names) {
  if (!arguments)
    return;
  if (const auto *posOnly = ast::nodeList(*arguments, "posonlyargs"))
    for (const parser::NodePtr &arg : *posOnly)
      names.insert(ast::nameSpelling(*arg));
  if (const auto *args = ast::nodeList(*arguments, "args"))
    for (const parser::NodePtr &arg : *args)
      names.insert(ast::nameSpelling(*arg));
  if (const auto *kwonly = ast::nodeList(*arguments, "kwonlyargs"))
    for (const parser::NodePtr &arg : *kwonly)
      names.insert(ast::nameSpelling(*arg));
  if (const parser::Node *vararg = ast::node(*arguments, "vararg"))
    names.insert(ast::nameSpelling(*vararg));
  if (const parser::Node *kwarg = ast::node(*arguments, "kwarg"))
    names.insert(ast::nameSpelling(*kwarg));
}

void collectLocalNames(const parser::Node *node, llvm::StringSet<> &names) {
  if (!node)
    return;
  if (node->kind == "FunctionDef" || node->kind == "AsyncFunctionDef" ||
      node->kind == "ClassDef") {
    if (auto name = ast::string(*node, "name"))
      names.insert(*name);
    return;
  }
  if (node->kind == "Lambda")
    return;
  if (node->kind == "Assign") {
    if (const auto *targets = ast::nodeList(*node, "targets"))
      for (const parser::NodePtr &target : *targets)
        collectTargetNames(target.get(), names);
  } else if (node->kind == "AnnAssign" || node->kind == "AugAssign" ||
             node->kind == "NamedExpr") {
    collectTargetNames(ast::node(*node, "target"), names);
  } else if (node->kind == "For" || node->kind == "AsyncFor") {
    collectTargetNames(ast::node(*node, "target"), names);
  } else if (node->kind == "With" || node->kind == "AsyncWith") {
    if (const auto *items = ast::nodeList(*node, "items"))
      for (const parser::NodePtr &item : *items)
        collectTargetNames(ast::node(*item, "optional_vars"), names);
  }
  for (const parser::Field &field : node->fields) {
    if (const auto *child = std::get_if<parser::NodePtr>(&field.value)) {
      if (*child)
        collectLocalNames(child->get(), names);
    } else if (const auto *children =
                   std::get_if<std::vector<parser::NodePtr>>(&field.value)) {
      for (const parser::NodePtr &child : *children)
        if (child)
          collectLocalNames(child.get(), names);
    }
  }
}

void collectReadNames(const parser::Node *node, llvm::StringSet<> &names) {
  if (!node)
    return;
  if (node->kind == "Name") {
    if (!hasContext(*node, "Store") && !hasContext(*node, "Del"))
      names.insert(ast::nameSpelling(*node));
    return;
  }
  if (node->kind == "FunctionDef" || node->kind == "AsyncFunctionDef") {
    collectReadNames(ast::node(*node, "args"), names);
    collectReadNames(ast::node(*node, "returns"), names);
    if (const auto *decorators = ast::nodeList(*node, "decorator_list"))
      for (const parser::NodePtr &decorator : *decorators)
        collectReadNames(decorator.get(), names);
    if (const auto *body = ast::nodeList(*node, "body"))
      for (const parser::NodePtr &statement : *body)
        collectReadNames(statement.get(), names);
    return;
  }
  for (const parser::Field &field : node->fields) {
    if (field.name == "ctx")
      continue;
    if (const auto *child = std::get_if<parser::NodePtr>(&field.value)) {
      if (*child)
        collectReadNames(child->get(), names);
    } else if (const auto *children =
                   std::get_if<std::vector<parser::NodePtr>>(&field.value)) {
      for (const parser::NodePtr &child : *children)
        if (child)
          collectReadNames(child.get(), names);
    }
  }
}

} // namespace

llvm::SmallVector<std::string, 4>
lexicalCaptureNames(const parser::Node &callable) {
  llvm::StringSet<> locals;
  collectParameterNames(ast::node(callable, "args"), locals);
  if (const auto *body = ast::nodeList(callable, "body"))
    for (const parser::NodePtr &statement : *body)
      collectLocalNames(statement.get(), locals);
  if (const parser::Node *body = ast::node(callable, "body"))
    collectLocalNames(body, locals);

  llvm::StringSet<> reads;
  if (const auto *body = ast::nodeList(callable, "body"))
    for (const parser::NodePtr &statement : *body)
      collectReadNames(statement.get(), reads);
  if (const parser::Node *body = ast::node(callable, "body"))
    collectReadNames(body, reads);

  llvm::SmallVector<std::string, 4> captures;
  for (const auto &entry : reads)
    if (!locals.contains(entry.getKey()))
      captures.push_back(entry.getKey().str());
  llvm::sort(captures);
  return captures;
}

std::string sanitizedSymbolPart(llvm::StringRef text) {
  std::string result;
  result.reserve(text.size());
  for (char ch : text)
    result.push_back(llvm::isAlnum(ch) ? ch : '_');
  return result.empty() ? "callable" : result;
}

} // namespace lython::emitter
