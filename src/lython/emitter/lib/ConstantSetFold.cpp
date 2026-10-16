#include "ConstantSetFold.h"

#include "AstAccess.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringRef.h"

#include <cmath>
#include <complex>
#include <optional>

namespace lython::emitter {
namespace {

// CPython's flowgraph.c limits: a folded int may have at most 128 bits, a
// folded str or bytes at most 4096 items. An operation past them, or one that
// raises, is left to run -- and the display holding it is not folded.
constexpr unsigned kMaxIntBits = 128;
constexpr std::uint64_t kMaxStrSize = 4096;
// Wide enough for any product of two values within the limit.
constexpr unsigned kWorkBits = 2 * kMaxIntBits + 64;

struct Folded {
  enum class Kind { Int, Float, Complex, Str, Bytes, Other } kind;
  llvm::APInt integer{kWorkBits, 0};
  double real = 0;
  std::uint64_t length = 0;
  bool isBool = false;
};

unsigned bitLength(const llvm::APInt &value) {
  return value.isNegative() ? (-value).getActiveBits() : value.getActiveBits();
}

std::optional<Folded> intValue(llvm::APInt value) {
  if (bitLength(value) > kWorkBits - 2)
    return std::nullopt;
  Folded folded{Folded::Kind::Int};
  folded.integer = std::move(value);
  return folded;
}

std::optional<Folded> constantValue(const parser::Node &node) {
  if (ast::isNoneField(node, "value"))
    return Folded{Folded::Kind::Other};
  if (std::optional<bool> value = ast::boolean(node, "value")) {
    Folded folded{Folded::Kind::Int};
    folded.integer = llvm::APInt(kWorkBits, *value ? 1 : 0);
    folded.isBool = true;
    return folded;
  }
  if (std::optional<std::int64_t> value = ast::integer(node, "value"))
    return intValue(llvm::APInt(kWorkBits, static_cast<std::uint64_t>(*value),
                                /*isSigned=*/true));
  if (std::optional<double> value = ast::floating(node, "value")) {
    Folded folded{Folded::Kind::Float};
    folded.real = *value;
    return folded;
  }
  if (std::optional<std::string_view> value = ast::string(node, "value")) {
    Folded folded{Folded::Kind::Str};
    // Code points, not bytes: the limit counts items.
    for (unsigned char byte : *value)
      folded.length += (byte & 0xC0) != 0x80;
    return folded;
  }
  if (const auto *value = ast::bytes(node, "value")) {
    Folded folded{Folded::Kind::Bytes};
    folded.length = value->size();
    return folded;
  }
  if (const auto *field = ast::field(node, "value")) {
    if (const auto *big = std::get_if<parser::BigInteger>(field)) {
      llvm::StringRef digits = big->decimal;
      bool negative = digits.consume_front("-");
      // A literal wider than the work width is still a constant; it only
      // cannot take part in an operation this evaluator checks.
      if (digits.size() * 4 + 8 > kWorkBits) {
        Folded folded{Folded::Kind::Other};
        return folded;
      }
      llvm::APInt value(kWorkBits, digits, 10);
      return intValue(negative ? -value : value);
    }
    if (std::holds_alternative<std::complex<double>>(*field))
      return Folded{Folded::Kind::Complex};
    if (std::holds_alternative<parser::Ellipsis>(*field))
      return Folded{Folded::Kind::Other};
  }
  return std::nullopt;
}

bool numeric(const Folded &value) {
  return value.kind == Folded::Kind::Int || value.kind == Folded::Kind::Float ||
         value.kind == Folded::Kind::Complex;
}

double asReal(const Folded &value) {
  return value.kind == Folded::Kind::Int
             ? value.integer.roundToDouble(/*isSigned=*/true)
             : value.real;
}

// Python's floor division and modulo of two ints.
std::pair<llvm::APInt, llvm::APInt> floorDivMod(const llvm::APInt &a,
                                                const llvm::APInt &b) {
  llvm::APInt quotient = a.sdiv(b);
  llvm::APInt remainder = a.srem(b);
  if (!remainder.isZero() && (remainder.isNegative() != b.isNegative())) {
    quotient -= 1;
    remainder += b;
  }
  return {quotient, remainder};
}

std::optional<Folded> fold(const parser::Node *node);

std::optional<Folded> foldUnary(const parser::Node &node) {
  const parser::Node *op = ast::node(node, "op");
  std::optional<Folded> operand = fold(ast::node(node, "operand"));
  if (!op || !operand)
    return std::nullopt;
  if (op->kind == "Not")
    return Folded{Folded::Kind::Int, llvm::APInt(kWorkBits, 0), 0, 0, true};
  if (!numeric(*operand))
    return std::nullopt;
  if (op->kind == "UAdd" || op->kind == "USub") {
    Folded result = *operand;
    result.isBool = false;
    if (op->kind == "USub") {
      result.integer = -result.integer;
      result.real = -result.real;
    }
    return result;
  }
  // ⛔ `~` on a bool is deprecated, and the compiler leaves it to warn.
  if (op->kind == "Invert" && operand->kind == Folded::Kind::Int &&
      !operand->isBool)
    return intValue(~operand->integer);
  return std::nullopt;
}

std::optional<Folded> foldBinary(const parser::Node &node) {
  const parser::Node *op = ast::node(node, "op");
  std::optional<Folded> left = fold(ast::node(node, "left"));
  std::optional<Folded> right = fold(ast::node(node, "right"));
  if (!op || !left || !right)
    return std::nullopt;
  llvm::StringRef kind = op->kind;
  using K = Folded::Kind;
  bool ints = left->kind == K::Int && right->kind == K::Int;
  auto sequence = [](const Folded &value) {
    return value.kind == K::Str || value.kind == K::Bytes;
  };
  if (kind == "Add" && sequence(*left) && left->kind == right->kind) {
    Folded result = *left;
    result.length += right->length;
    return result;
  }
  if (kind == "Mult" && (sequence(*left) || sequence(*right))) {
    const Folded &items = sequence(*left) ? *left : *right;
    const Folded &count = sequence(*left) ? *right : *left;
    if (count.kind != K::Int)
      return std::nullopt;
    // safe_multiply: a negative count, or one past 4096 / length, is left
    // to run.
    if (count.integer.isNegative())
      return std::nullopt;
    if (items.length != 0 &&
        (bitLength(count.integer) > 32 ||
         count.integer.getZExtValue() > kMaxStrSize / items.length))
      return std::nullopt;
    Folded result = items;
    result.length =
        items.length == 0 ? 0 : items.length * count.integer.getZExtValue();
    return result;
  }
  if (!numeric(*left) || !numeric(*right))
    return std::nullopt;
  if (ints) {
    const llvm::APInt &a = left->integer;
    const llvm::APInt &b = right->integer;
    if (kind == "Add")
      return intValue(a + b);
    if (kind == "Sub")
      return intValue(a - b);
    if (kind == "Mult") {
      if (!a.isZero() && !b.isZero() &&
          bitLength(a) + bitLength(b) > kMaxIntBits)
        return std::nullopt;
      return intValue(a * b);
    }
    if (kind == "BitAnd")
      return intValue(a & b);
    if (kind == "BitOr")
      return intValue(a | b);
    if (kind == "BitXor")
      return intValue(a ^ b);
    if (kind == "FloorDiv" || kind == "Mod") {
      if (b.isZero())
        return std::nullopt;
      auto [quotient, remainder] = floorDivMod(a, b);
      return intValue(kind == "FloorDiv" ? quotient : remainder);
    }
    if (kind == "LShift") {
      if (b.isNegative())
        return std::nullopt;
      if (a.isZero() || b.isZero())
        return intValue(a);
      unsigned bits = bitLength(a);
      if (bits > kMaxIntBits || bitLength(b) > 16 ||
          b.getZExtValue() > kMaxIntBits ||
          bits > kMaxIntBits - b.getZExtValue())
        return std::nullopt;
      return intValue(a.shl(static_cast<unsigned>(b.getZExtValue())));
    }
    if (kind == "RShift") {
      if (b.isNegative())
        return std::nullopt;
      unsigned shift = bitLength(b) > 16
                           ? kWorkBits - 1
                           : std::min<unsigned>(b.getZExtValue(), kWorkBits - 1);
      return intValue(a.ashr(shift));
    }
    // safe_power: a nonzero base to a positive power folds only while
    // bits(base) <= 128 / power.
    if (kind == "Pow") {
      if (b.isNegative()) {
        if (a.isZero())
          return std::nullopt;
        return Folded{K::Float};
      }
      if (a.isZero())
        return intValue(llvm::APInt(kWorkBits, b.isZero() ? 1 : 0));
      if (b.isZero())
        return intValue(llvm::APInt(kWorkBits, 1));
      if (b.getActiveBits() > 32 ||
          bitLength(a) > kMaxIntBits / b.getZExtValue())
        return std::nullopt;
      llvm::APInt result(kWorkBits, 1);
      for (std::uint64_t i = 0; i < b.getZExtValue(); ++i)
        result *= a;
      return intValue(result);
    }
    if (kind == "Div") {
      if (b.isZero())
        return std::nullopt;
      Folded result{K::Float};
      result.real = asReal(*left) / asReal(*right);
      return result;
    }
    return std::nullopt;
  }
  // A float or complex operand: the result is that kind; only a zero
  // divisor (or a complex in floor division or modulo) raises.
  K resultKind =
      left->kind == K::Complex || right->kind == K::Complex ? K::Complex
                                                            : K::Float;
  bool divides = kind == "Div" || kind == "FloorDiv" || kind == "Mod";
  if (divides && resultKind == K::Complex && kind != "Div")
    return std::nullopt;
  if (divides && right->kind != K::Complex && asReal(*right) == 0)
    return std::nullopt;
  if (resultKind == K::Complex && divides)
    return std::nullopt;
  if (kind != "Add" && kind != "Sub" && kind != "Mult" && kind != "Div" &&
      kind != "FloorDiv" && kind != "Mod" && kind != "Pow")
    return std::nullopt;
  if (kind == "Pow") {
    // A negative float to a fractional power is complex, zero to a negative
    // power raises: leave both to run.
    if (resultKind == K::Complex)
      return std::nullopt;
    double base = asReal(*left);
    double exponent = asReal(*right);
    if ((base == 0 && exponent < 0) ||
        (base < 0 && exponent != std::floor(exponent)))
      return std::nullopt;
  }
  Folded result{resultKind};
  return result;
}

std::optional<Folded> fold(const parser::Node *node) {
  if (!node)
    return std::nullopt;
  if (node->kind == "Constant")
    return constantValue(*node);
  if (node->kind == "UnaryOp")
    return foldUnary(*node);
  if (node->kind == "BinOp")
    return foldBinary(*node);
  if (node->kind == "Tuple") {
    const auto *elts = ast::nodeList(*node, "elts");
    if (!elts)
      return std::nullopt;
    for (const parser::NodePtr &element : *elts)
      if (!element || element->kind == "Starred" || !fold(element.get()))
        return std::nullopt;
    return Folded{Folded::Kind::Other};
  }
  return std::nullopt;
}

void collect(const parser::Node &node,
             llvm::DenseSet<const parser::Node *> &out) {
  auto mark = [&](const parser::Node *candidate) {
    if (candidate && candidate->kind == "Set")
      out.insert(candidate);
  };
  if (node.kind == "For" || node.kind == "AsyncFor" ||
      node.kind == "comprehension")
    mark(ast::node(node, "iter"));
  if (node.kind == "Compare") {
    const auto *ops = ast::nodeList(node, "ops");
    const auto *comparators = ast::nodeList(node, "comparators");
    if (ops && comparators)
      for (std::size_t index = 0;
           index < ops->size() && index < comparators->size(); ++index)
        if ((*ops)[index] && ((*ops)[index]->kind == "In" ||
                              (*ops)[index]->kind == "NotIn"))
          mark((*comparators)[index].get());
  }
  for (const parser::Field &field : node.fields) {
    if (const auto *child = std::get_if<parser::NodePtr>(&field.value)) {
      if (*child)
        collect(**child, out);
      continue;
    }
    if (const auto *children =
            std::get_if<std::vector<parser::NodePtr>>(&field.value))
      for (const parser::NodePtr &child : *children)
        if (child)
          collect(*child, out);
  }
}

} // namespace

bool cpythonFoldsSetDisplay(const parser::Node &set) {
  if (set.kind != "Set")
    return false;
  const auto *elts = ast::nodeList(set, "elts");
  if (!elts || elts->size() < 3)
    return false;
  for (const parser::NodePtr &element : *elts)
    if (!element || element->kind == "Starred" || !fold(element.get()))
      return false;
  return true;
}

void collectFrozensetSetDisplays(const parser::Node &tree,
                                 llvm::DenseSet<const parser::Node *> &out) {
  collect(tree, out);
}

} // namespace lython::emitter
