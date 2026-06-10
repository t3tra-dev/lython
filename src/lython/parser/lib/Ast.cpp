#include "lython/parser/Ast.h"

#include "lython/parser/CpythonSpec.h"

#include <optional>
#include <sstream>
#include <type_traits>

namespace lython::parser {
namespace {

void dumpByteEscape(std::ostream &os, std::uint8_t byte) {
  const char *hex = "0123456789abcdef";
  os << "\\x" << hex[(byte >> 4) & 0xf] << hex[byte & 0xf];
}

void dumpEscaped(std::ostream &os, std::string_view value) {
  os << '"';
  for (unsigned char byte : value) {
    const char ch = static_cast<char>(byte);
    switch (ch) {
    case '\\':
      os << "\\\\";
      break;
    case '"':
      os << "\\\"";
      break;
    case '\n':
      os << "\\n";
      break;
    case '\r':
      os << "\\r";
      break;
    case '\t':
      os << "\\t";
      break;
    default:
      if (byte < 0x20 || byte == 0x7f)
        dumpByteEscape(os, static_cast<std::uint8_t>(byte));
      else
        os << ch;
      break;
    }
  }
  os << '"';
}

void dumpBytes(std::ostream &os, const std::vector<std::uint8_t> &value) {
  os << "b\"";
  for (std::uint8_t byte : value) {
    const char ch = static_cast<char>(byte);
    switch (ch) {
    case '\\':
      os << "\\\\";
      break;
    case '"':
      os << "\\\"";
      break;
    case '\n':
      os << "\\n";
      break;
    case '\r':
      os << "\\r";
      break;
    case '\t':
      os << "\\t";
      break;
    default:
      if (byte < 0x20 || byte >= 0x7f) {
        dumpByteEscape(os, byte);
      } else {
        os << ch;
      }
      break;
    }
  }
  os << '"';
}

void dumpComplex(std::ostream &os, const std::complex<double> &value) {
  if (value.real() != 0.0)
    os << value.real();
  if (value.imag() >= 0.0 && value.real() != 0.0)
    os << '+';
  os << value.imag() << 'j';
}

void dumpValue(std::ostream &os, const FieldValue &value,
               bool includeAttributes);

bool hasSourceAttributes(const Node &node) {
  return isCpythonAstKindOfType(node.kind, "stmt") ||
         isCpythonAstKindOfType(node.kind, "expr") ||
         isCpythonAstKindOfType(node.kind, "excepthandler") ||
         isCpythonAstKindOfType(node.kind, "arg") ||
         isCpythonAstKindOfType(node.kind, "keyword") ||
         isCpythonAstKindOfType(node.kind, "alias") ||
         isCpythonAstKindOfType(node.kind, "pattern") ||
         isCpythonAstKindOfType(node.kind, "type_param");
}

void dumpAttributes(std::ostream &os, const Node &node, bool needsComma) {
  if (!hasSourceAttributes(node))
    return;
  if (needsComma)
    os << ", ";
  os << "lineno=" << node.range.start.line
     << ", col_offset=" << node.range.start.column
     << ", end_lineno=" << node.range.end.line
     << ", end_col_offset=" << node.range.end.column;
}

void dumpNode(std::ostream &os, const Node &node, bool includeAttributes) {
  os << node.kind << '(';
  for (std::size_t i = 0; i < node.fields.size(); ++i) {
    if (i != 0)
      os << ", ";
    os << node.fields[i].name << '=';
    dumpValue(os, node.fields[i].value, includeAttributes);
  }
  if (includeAttributes)
    dumpAttributes(os, node, !node.fields.empty());
  os << ')';
}

void dumpNodeList(std::ostream &os, const std::vector<NodePtr> &nodes,
                  bool includeAttributes) {
  os << '[';
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    if (i != 0)
      os << ", ";
    if (nodes[i])
      dumpNode(os, *nodes[i], includeAttributes);
    else
      os << "None";
  }
  os << ']';
}

void dumpStringList(std::ostream &os, const std::vector<std::string> &values) {
  os << '[';
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0)
      os << ", ";
    dumpEscaped(os, values[i]);
  }
  os << ']';
}

void dumpValue(std::ostream &os, const FieldValue &value,
               bool includeAttributes) {
  std::visit(
      [&](const auto &typedValue) {
        using T = std::decay_t<decltype(typedValue)>;
        if constexpr (std::is_same_v<T, std::monostate>) {
          os << "None";
        } else if constexpr (std::is_same_v<T, bool>) {
          os << (typedValue ? "True" : "False");
        } else if constexpr (std::is_same_v<T, std::int64_t> ||
                             std::is_same_v<T, double>) {
          os << typedValue;
        } else if constexpr (std::is_same_v<T, Ellipsis>) {
          os << "Ellipsis";
        } else if constexpr (std::is_same_v<T, BigInteger>) {
          os << typedValue.decimal;
        } else if constexpr (std::is_same_v<T, std::complex<double>>) {
          dumpComplex(os, typedValue);
        } else if constexpr (std::is_same_v<T, std::string>) {
          dumpEscaped(os, typedValue);
        } else if constexpr (std::is_same_v<T, std::vector<std::uint8_t>>) {
          dumpBytes(os, typedValue);
        } else if constexpr (std::is_same_v<T, NodePtr>) {
          if (typedValue)
            dumpNode(os, *typedValue, includeAttributes);
          else
            os << "None";
        } else if constexpr (std::is_same_v<T, std::vector<NodePtr>>) {
          dumpNodeList(os, typedValue, includeAttributes);
        } else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
          dumpStringList(os, typedValue);
        }
      },
      value);
}

} // namespace

NodePtr makeNode(std::string kind, SourceRange range) {
  NodePtr node = std::make_shared<Node>(std::move(kind));
  node->range = range;
  return node;
}

void addField(Node &node, std::string name, FieldValue value) {
  std::size_t slot = invalidFieldSlot;
  if (std::optional<std::size_t> fieldSlot =
          cpythonAstFieldIndex(node.kind, name)) {
    slot = *fieldSlot;
  }

  const std::size_t fieldIndex = node.fields.size();
  node.fields.push_back(Field{std::move(name), std::move(value), slot});
  if (slot == invalidFieldSlot)
    return;

  if (node.fieldIndicesBySlot.size() <= slot)
    node.fieldIndicesBySlot.resize(slot + 1, invalidFieldSlot);
  node.fieldIndicesBySlot[slot] = fieldIndex;
}

const Field *findField(const Node &node, std::string_view name) {
  if (std::optional<std::size_t> slot = cpythonAstFieldIndex(node.kind, name)) {
    if (*slot < node.fieldIndicesBySlot.size()) {
      std::size_t fieldIndex = node.fieldIndicesBySlot[*slot];
      if (fieldIndex < node.fields.size() &&
          node.fields[fieldIndex].name == name)
        return &node.fields[fieldIndex];
    }
  }

  for (const Field &field : node.fields)
    if (field.name == name)
      return &field;
  return nullptr;
}

Field *findField(Node &node, std::string_view name) {
  return const_cast<Field *>(findField(static_cast<const Node &>(node), name));
}

std::string dumpAst(const Node &node) {
  return dumpAst(node, /*includeAttributes=*/false);
}

std::string dumpAst(const Node &node, bool includeAttributes) {
  std::ostringstream os;
  dumpNode(os, node, includeAttributes);
  return os.str();
}

} // namespace lython::parser
