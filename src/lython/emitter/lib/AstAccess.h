#pragma once

#include "Ast.h"

#include "llvm/ADT/STLFunctionalExtras.h"

#include <optional>
#include <string>
#include <string_view>

namespace lython::emitter::ast {

// ⭐ ONE AST WALK. Every question asked of a subtree -- does it yield, does it
// read this name, which names does it bind, where are its calls -- is a
// depth-first pass over `fields`, and each was written out again. What differs
// is what a node answers.
//
// `SkipChildren` is what makes the scope boundaries expressible: which of
// def/class/lambda ends a walk differs by question (a yield search stops at
// all four; a name-binding walk records the def's own name first), so the walk
// cannot decide it and the visitor must.
enum class Walk { Continue, SkipChildren, Stop };

// Depth-first over the subtree, `node` included. True when a visit answered
// `Stop` -- which is how a "does this subtree contain ..." question is asked.
inline bool walk(const parser::Node *node,
                 llvm::function_ref<Walk(const parser::Node &)> visit) {
  if (!node)
    return false;
  switch (visit(*node)) {
  case Walk::Stop:
    return true;
  case Walk::SkipChildren:
    return false;
  case Walk::Continue:
    break;
  }
  for (const parser::Field &field : node->fields) {
    if (const auto *child = std::get_if<parser::NodePtr>(&field.value)) {
      if (walk(child->get(), visit))
        return true;
      continue;
    }
    if (const auto *children =
            std::get_if<std::vector<parser::NodePtr>>(&field.value))
      for (const parser::NodePtr &nested : *children)
        if (walk(nested.get(), visit))
          return true;
  }
  return false;
}

inline const parser::FieldValue *field(const parser::Node &node,
                                       std::string_view name) {
  if (const parser::Field *found = parser::findField(node, name))
    return &found->value;
  return nullptr;
}

template <typename T> const T *as(const parser::FieldValue *value) {
  if (!value)
    return nullptr;
  return std::get_if<T>(value);
}

inline const parser::Node *node(const parser::Node &owner,
                                std::string_view name) {
  const auto *ptr = as<parser::NodePtr>(field(owner, name));
  return ptr && *ptr ? ptr->get() : nullptr;
}

inline const std::vector<parser::NodePtr> *nodeList(const parser::Node &owner,
                                                    std::string_view name) {
  return as<std::vector<parser::NodePtr>>(field(owner, name));
}

inline const std::vector<std::string> *stringList(const parser::Node &owner,
                                                  std::string_view name) {
  return as<std::vector<std::string>>(field(owner, name));
}

inline std::optional<std::string_view> string(const parser::Node &owner,
                                              std::string_view name) {
  if (const auto *value = as<std::string>(field(owner, name)))
    return std::string_view(*value);
  return std::nullopt;
}

inline const std::vector<std::uint8_t> *bytes(const parser::Node &owner,
                                              std::string_view name) {
  return as<std::vector<std::uint8_t>>(field(owner, name));
}

inline std::optional<std::int64_t> integer(const parser::Node &owner,
                                           std::string_view name) {
  if (const auto *value = as<std::int64_t>(field(owner, name)))
    return *value;
  return std::nullopt;
}

inline std::optional<double> floating(const parser::Node &owner,
                                      std::string_view name) {
  if (const auto *value = as<double>(field(owner, name)))
    return *value;
  return std::nullopt;
}

inline std::optional<bool> boolean(const parser::Node &owner,
                                   std::string_view name) {
  if (const auto *value = as<bool>(field(owner, name)))
    return *value;
  return std::nullopt;
}

inline bool isNoneField(const parser::Node &owner, std::string_view name) {
  const parser::FieldValue *value = field(owner, name);
  return value && std::holds_alternative<std::monostate>(*value);
}

inline bool isEllipsisField(const parser::Node &owner, std::string_view name) {
  const parser::FieldValue *value = field(owner, name);
  return value && std::holds_alternative<parser::Ellipsis>(*value);
}

inline bool isName(const parser::Node &node, std::string_view spelling) {
  auto id = string(node, "id");
  return node.kind == "Name" && id && *id == spelling;
}

inline bool isOperator(const parser::Node *node, std::string_view kind) {
  return node && node->kind == kind;
}

inline std::string_view nameSpelling(const parser::Node &node) {
  if (auto id = string(node, "id"))
    return *id;
  if (auto attr = string(node, "attr"))
    return *attr;
  if (auto name = string(node, "name"))
    return *name;
  if (auto arg = string(node, "arg"))
    return *arg;
  return {};
}

inline std::string qualifiedName(const parser::Node *node) {
  if (!node)
    return {};
  if (node->kind == "Name")
    return std::string(nameSpelling(*node));
  if (node->kind != "Attribute")
    return {};

  std::string base = qualifiedName(ast::node(*node, "value"));
  auto attr = string(*node, "attr");
  if (base.empty() || !attr)
    return {};
  base.push_back('.');
  base.append(attr->begin(), attr->end());
  return base;
}

} // namespace lython::emitter::ast
