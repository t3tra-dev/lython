#include "BuilderImpl.h"

namespace lython::emitter {

const parser::Field *field(const parser::Node &node, std::string_view name) {
  return parser::findField(node, name);
}

const std::string *stringField(const parser::Node &node,
                               std::string_view name) {
  const parser::Field *selected = field(node, name);
  if (!selected)
    return nullptr;
  return std::get_if<std::string>(&selected->value);
}

std::optional<std::string> astSymbol(const parser::Node &node) {
  if (node.kind == "Load" || node.kind == "Store" || node.kind == "Del")
    return node.kind;
  if (node.kind == "And")
    return std::string("and");
  if (node.kind == "Or")
    return std::string("or");
  if (node.kind == "Add")
    return std::string("+");
  if (node.kind == "Sub")
    return std::string("-");
  if (node.kind == "Mult")
    return std::string("*");
  if (node.kind == "MatMult")
    return std::string("@");
  if (node.kind == "Div")
    return std::string("/");
  if (node.kind == "Mod")
    return std::string("%");
  if (node.kind == "Pow")
    return std::string("**");
  if (node.kind == "LShift")
    return std::string("<<");
  if (node.kind == "RShift")
    return std::string(">>");
  if (node.kind == "BitOr")
    return std::string("|");
  if (node.kind == "BitXor")
    return std::string("^");
  if (node.kind == "BitAnd")
    return std::string("&");
  if (node.kind == "FloorDiv")
    return std::string("//");
  if (node.kind == "Invert")
    return std::string("~");
  if (node.kind == "Not")
    return std::string("not");
  if (node.kind == "UAdd")
    return std::string("+");
  if (node.kind == "USub")
    return std::string("-");
  if (node.kind == "Eq")
    return std::string("==");
  if (node.kind == "NotEq")
    return std::string("!=");
  if (node.kind == "Lt")
    return std::string("<");
  if (node.kind == "LtE")
    return std::string("<=");
  if (node.kind == "Gt")
    return std::string(">");
  if (node.kind == "GtE")
    return std::string(">=");
  if (node.kind == "Is")
    return std::string("is");
  if (node.kind == "IsNot")
    return std::string("is not");
  if (node.kind == "In")
    return std::string("in");
  if (node.kind == "NotIn")
    return std::string("not in");
  return std::nullopt;
}

std::optional<std::string> symbolField(const parser::Node &node,
                                       std::string_view name) {
  const parser::Field *selected = field(node, name);
  if (!selected)
    return std::nullopt;
  if (const auto *value = std::get_if<std::string>(&selected->value))
    return *value;
  if (const auto *child = std::get_if<parser::NodePtr>(&selected->value))
    return *child ? astSymbol(**child) : std::nullopt;
  return std::nullopt;
}

std::optional<std::vector<std::string>>
symbolListField(const parser::Node &node, std::string_view name) {
  const parser::Field *selected = field(node, name);
  if (!selected)
    return std::nullopt;
  if (const auto *values =
          std::get_if<std::vector<std::string>>(&selected->value))
    return *values;
  const auto *nodes =
      std::get_if<std::vector<parser::NodePtr>>(&selected->value);
  if (!nodes)
    return std::nullopt;
  std::vector<std::string> result;
  result.reserve(nodes->size());
  for (const parser::NodePtr &node : *nodes) {
    if (!node)
      return std::nullopt;
    std::optional<std::string> value = astSymbol(*node);
    if (!value)
      return std::nullopt;
    result.push_back(std::move(*value));
  }
  return result;
}

const parser::NodePtr *nodeField(const parser::Node &node,
                                 std::string_view name) {
  const parser::Field *selected = field(node, name);
  if (!selected)
    return nullptr;
  return std::get_if<parser::NodePtr>(&selected->value);
}

const std::vector<parser::NodePtr> *nodeListField(const parser::Node &node,
                                                  std::string_view name) {
  const parser::Field *selected = field(node, name);
  if (!selected)
    return nullptr;
  return std::get_if<std::vector<parser::NodePtr>>(&selected->value);
}

const parser::FieldValue *valueField(const parser::Node &node,
                                     std::string_view name) {
  const parser::Field *selected = field(node, name);
  if (!selected)
    return nullptr;
  return &selected->value;
}

bool hasNodeListEntries(const parser::Node &node, std::string_view name) {
  const std::vector<parser::NodePtr> *items = nodeListField(node, name);
  return items && !items->empty();
}

bool hasTypeParams(const parser::Node &node) {
  return hasNodeListEntries(node, "type_params");
}

std::optional<int> singletonKey(const parser::Node &node) {
  if (node.kind != "Constant")
    return std::nullopt;
  const parser::FieldValue *value = valueField(node, "value");
  if (!value)
    return std::nullopt;
  if (std::holds_alternative<std::monostate>(*value))
    return 0;
  if (const auto *boolean = std::get_if<bool>(value))
    return *boolean ? 1 : 2;
  return std::nullopt;
}

} // namespace lython::emitter
