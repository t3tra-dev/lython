#pragma once

#include <complex>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace lython::parser {

inline constexpr std::size_t invalidFieldSlot =
    std::numeric_limits<std::size_t>::max();

struct SourceLocation {
  int line = 1;
  int column = 0;
  int offset = 0;
};

struct SourceRange {
  SourceLocation start;
  SourceLocation end;
};

struct Node;
using NodePtr = std::shared_ptr<Node>;

struct Ellipsis {};
struct BigInteger {
  std::string decimal;
};

using FieldValue = std::variant<std::monostate, bool, std::int64_t, double,
                                Ellipsis, BigInteger, std::complex<double>,
                                std::string, std::vector<std::uint8_t>, NodePtr,
                                std::vector<NodePtr>, std::vector<std::string>>;

struct Field {
  std::string name;
  FieldValue value;
  std::size_t slot = invalidFieldSlot;
};

struct Node {
  std::string kind;
  SourceRange range;
  std::vector<Field> fields;
  std::vector<std::size_t> fieldIndicesBySlot;

  explicit Node(std::string kindName) : kind(std::move(kindName)) {}
};

NodePtr makeNode(std::string kind, SourceRange range = {});
void addField(Node &node, std::string name, FieldValue value);
const Field *findField(const Node &node, std::string_view name);
Field *findField(Node &node, std::string_view name);
std::string dumpAst(const Node &node);
std::string dumpAst(const Node &node, bool includeAttributes);

} // namespace lython::parser
