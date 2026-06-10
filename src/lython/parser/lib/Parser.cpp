#include "lython/parser/Parser.h"

#include "lython/parser/CpythonSpec.h"

#include "../pegen.h"
#include "CpythonPegAdapter.h"
#include "GeneratedTokenStream.h"
#include "Lexer.h"
#include "SourceEncoding.h"
#include "Token.h"
#include "UnicodeNames.h"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <iomanip>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <string_view>
#include <system_error>
#include <utility>

namespace lython::parser {
namespace {

int cpythonGeneratedStartRule(ParseMode mode) {
  switch (mode) {
  case ParseMode::Module:
    return Py_file_input;
  case ParseMode::Interactive:
    return Py_single_input;
  case ParseMode::Expression:
    return Py_eval_input;
  case ParseMode::FunctionType:
    return Py_func_type_input;
  }
  return Py_file_input;
}

std::string_view cpythonGeneratedRootKind(ParseMode mode) {
  switch (mode) {
  case ParseMode::Module:
    return "Module";
  case ParseMode::Interactive:
    return "Interactive";
  case ParseMode::Expression:
    return "Expression";
  case ParseMode::FunctionType:
    return "FunctionType";
  }
  return "Module";
}

NodePtr astEnum(std::string kind, SourceRange range = {}) {
  return makeNode(std::move(kind), range);
}

NodePtr exprContext(std::string_view context, SourceRange range = {}) {
  if (context == "Load")
    return astEnum("Load", range);
  if (context == "Store")
    return astEnum("Store", range);
  if (context == "Del")
    return astEnum("Del", range);
  return astEnum("InvalidExprContext", range);
}

NodePtr boolOperator(std::string_view op, SourceRange range = {}) {
  if (op == "and")
    return astEnum("And", range);
  if (op == "or")
    return astEnum("Or", range);
  return astEnum("InvalidBoolOp", range);
}

NodePtr binaryOperator(std::string_view op, SourceRange range = {}) {
  static const std::map<std::string, std::string> operators = {
      {"+", "Add"},      {"-", "Sub"},   {"*", "Mult"},   {"@", "MatMult"},
      {"/", "Div"},      {"%", "Mod"},   {"**", "Pow"},   {"<<", "LShift"},
      {">>", "RShift"},  {"|", "BitOr"}, {"^", "BitXor"}, {"&", "BitAnd"},
      {"//", "FloorDiv"}};
  auto found = operators.find(std::string(op));
  return astEnum(found == operators.end() ? "InvalidBinOp" : found->second,
                 range);
}

NodePtr unaryOperator(std::string_view op, SourceRange range = {}) {
  if (op == "~")
    return astEnum("Invert", range);
  if (op == "not")
    return astEnum("Not", range);
  if (op == "+")
    return astEnum("UAdd", range);
  if (op == "-")
    return astEnum("USub", range);
  return astEnum("InvalidUnaryOp", range);
}

NodePtr comparisonOperator(std::string_view op, SourceRange range = {}) {
  static const std::map<std::string, std::string> operators = {
      {"==", "Eq"}, {"!=", "NotEq"},    {"<", "Lt"},  {"<=", "LtE"},
      {">", "Gt"},  {">=", "GtE"},      {"is", "Is"}, {"is not", "IsNot"},
      {"in", "In"}, {"not in", "NotIn"}};
  auto found = operators.find(std::string(op));
  return astEnum(found == operators.end() ? "InvalidCmpOp" : found->second,
                 range);
}

std::string withoutUnderscores(std::string_view text) {
  std::string result;
  result.reserve(text.size());
  for (char ch : text)
    if (ch != '_')
      result.push_back(ch);
  return result;
}

bool hasImaginarySuffix(std::string_view text) {
  return !text.empty() && (text.back() == 'j' || text.back() == 'J');
}

bool hasIntegerBasePrefix(std::string_view text) {
  if (hasImaginarySuffix(text))
    text.remove_suffix(1);
  return text.size() > 2 && text[0] == '0' &&
         (text[1] == 'b' || text[1] == 'B' || text[1] == 'o' ||
          text[1] == 'O' || text[1] == 'x' || text[1] == 'X');
}

bool isFloatLiteralText(std::string_view text) {
  if (hasImaginarySuffix(text))
    text.remove_suffix(1);
  if (hasIntegerBasePrefix(text))
    return false;
  return text.find('.') != std::string_view::npos ||
         text.find('e') != std::string_view::npos ||
         text.find('E') != std::string_view::npos;
}

bool isDigitForBase(char ch, int base) {
  if (ch >= '0' && ch <= '9')
    return ch - '0' < base;
  if (ch >= 'a' && ch <= 'f')
    return base == 16;
  if (ch >= 'A' && ch <= 'F')
    return base == 16;
  return false;
}

int digitValue(char ch) {
  if (ch >= '0' && ch <= '9')
    return ch - '0';
  if (ch >= 'a' && ch <= 'f')
    return 10 + ch - 'a';
  if (ch >= 'A' && ch <= 'F')
    return 10 + ch - 'A';
  return -1;
}

bool validIntegerUnderscores(std::string_view text, int base,
                             std::size_t firstDigit) {
  for (std::size_t i = 0; i < text.size(); ++i) {
    if (text[i] != '_')
      continue;
    if (i + 1 >= text.size())
      return false;
    bool afterBasePrefix = i == firstDigit && firstDigit == 2;
    bool previousOk =
        afterBasePrefix || (i > 0 && isDigitForBase(text[i - 1], base));
    bool nextOk = isDigitForBase(text[i + 1], base);
    if (!previousOk || !nextOk)
      return false;
  }
  return true;
}

bool validDecimalIntegerDigits(std::string_view text) {
  if (text.empty())
    return false;
  if (text.front() != '0')
    return true;
  for (char ch : text) {
    if (ch == '_')
      continue;
    if (ch != '0')
      return false;
  }
  return true;
}

bool validFloatUnderscores(std::string_view text) {
  for (std::size_t i = 0; i < text.size(); ++i) {
    if (text[i] != '_')
      continue;
    if (i == 0 || i + 1 >= text.size())
      return false;
    if (!std::isdigit(static_cast<unsigned char>(text[i - 1])) ||
        !std::isdigit(static_cast<unsigned char>(text[i + 1])))
      return false;
  }
  return true;
}

void appendBaseDigitToDecimal(std::string &decimal, int base, int digit) {
  int carry = digit;
  for (auto it = decimal.rbegin(); it != decimal.rend(); ++it) {
    int value = (*it - '0') * base + carry;
    *it = static_cast<char>('0' + (value % 10));
    carry = value / 10;
  }
  while (carry > 0) {
    decimal.insert(decimal.begin(), static_cast<char>('0' + (carry % 10)));
    carry /= 10;
  }
}

std::optional<std::string> parseIntegerLiteralDecimal(std::string_view text) {
  int base = 10;
  std::size_t firstDigit = 0;
  if (hasIntegerBasePrefix(text)) {
    char prefix = text[1];
    if (prefix == 'b' || prefix == 'B') {
      base = 2;
      firstDigit = 2;
    } else if (prefix == 'o' || prefix == 'O') {
      base = 8;
      firstDigit = 2;
    } else if (prefix == 'x' || prefix == 'X') {
      base = 16;
      firstDigit = 2;
    }
  }
  if (firstDigit >= text.size() ||
      !validIntegerUnderscores(text, base, firstDigit))
    return std::nullopt;
  if (base == 10 && !validDecimalIntegerDigits(text))
    return std::nullopt;
  std::string decimal = "0";
  for (std::size_t i = firstDigit; i < text.size(); ++i) {
    if (text[i] == '_')
      continue;
    int digit = digitValue(text[i]);
    if (digit < 0 || digit >= base)
      return std::nullopt;
    appendBaseDigitToDecimal(decimal, base, digit);
  }
  const std::size_t firstNonZero = decimal.find_first_not_of('0');
  if (firstNonZero == std::string::npos)
    return std::string("0");
  decimal.erase(0, firstNonZero);
  return decimal;
}

std::optional<std::int64_t> parseIntegerLiteral(std::string_view text) {
  std::optional<std::string> decimal = parseIntegerLiteralDecimal(text);
  if (!decimal)
    return std::nullopt;
  std::int64_t value = 0;
  const char *begin = decimal->data();
  const char *end = decimal->data() + decimal->size();
  std::from_chars_result parsed = std::from_chars(begin, end, value, 10);
  if (parsed.ec != std::errc() || parsed.ptr != end)
    return std::nullopt;
  return value;
}

std::optional<double> parseFloatLiteral(std::string_view text) {
  if (!validFloatUnderscores(text))
    return std::nullopt;
  std::string cleaned = withoutUnderscores(text);
  char *end = nullptr;
  double value = std::strtod(cleaned.c_str(), &end);
  if (end != cleaned.c_str() + cleaned.size())
    return std::nullopt;
  return value;
}

std::optional<std::complex<double>>
parseImaginaryLiteral(std::string_view text) {
  if (!hasImaginarySuffix(text))
    return std::nullopt;
  text.remove_suffix(1);
  if (hasIntegerBasePrefix(text))
    return std::nullopt;
  std::optional<double> imag;
  if (isFloatLiteralText(text)) {
    imag = parseFloatLiteral(text);
  } else if (std::optional<std::int64_t> integer = parseIntegerLiteral(text)) {
    imag = static_cast<double>(*integer);
  } else if (std::optional<std::string> decimal =
                 parseIntegerLiteralDecimal(text)) {
    char *end = nullptr;
    double value = std::strtod(decimal->c_str(), &end);
    if (end == decimal->c_str() + decimal->size())
      imag = value;
  }
  if (!imag)
    return std::nullopt;
  return std::complex<double>{0.0, *imag};
}

bool validateWithCpythonGeneratedParser(const std::vector<Token> &tokens,
                                        ParseMode mode,
                                        std::size_t typeIgnoreCount,
                                        Diagnostics &diagnostics) {
  GeneratedTokenStream stream;
  if (!buildGeneratedTokenStream(tokens, mode == ParseMode::Interactive, stream,
                                 diagnostics))
    return false;

  if (lython_cpython_generated_parse_tokens(
          stream.tokens.data(), stream.tokens.size(),
          cpythonGeneratedStartRule(mode), typeIgnoreCount)) {
    std::string_view rootKind = lython_cpython_generated_last_root_kind();
    if (rootKind != "none" && rootKind != "unknown" &&
        rootKind != cpythonGeneratedRootKind(mode)) {
      diagnostics.push_back(Diagnostic{
          Severity::Error,
          tokens.empty() ? SourceLocation{} : tokens.front().range.start,
          "CPython 3.14 generated PEG parser returned unexpected root kind '" +
              std::string(rootKind) + "'"});
      return false;
    }
    return true;
  }

  std::ostringstream tokenSummary;
  tokenSummary << "CPython 3.14 generated PEG parser rejected the token stream";
  tokenSummary << " (mark=" << lython_cpython_generated_last_mark()
               << ", error=" << lython_cpython_generated_last_error_indicator()
               << ", source=" << lython_cpython_generated_last_error_source()
               << ')';
  if (!tokens.empty()) {
    tokenSummary << " near " << tokens.front().cpythonName << '('
                 << tokens.front().cpythonId << ')';
    if (!tokens.front().text.empty())
      tokenSummary << " '" << tokens.front().text << "'";
  }
  diagnostics.push_back(
      Diagnostic{Severity::Error,
                 tokens.empty() ? SourceLocation{} : tokens.front().range.start,
                 tokenSummary.str()});
  return false;
}

class ParserImpl {
public:
  ParserImpl(std::vector<Token> tokens, Diagnostics &diagnostics,
             std::vector<TypeIgnoreInfo> typeIgnores = {},
             TypeCommentMap typeComments = {})
      : tokens(std::move(tokens)), diagnostics(diagnostics),
        typeIgnores(std::move(typeIgnores)),
        typeComments(std::move(typeComments)) {}

  NodePtr parseModule() {
    std::vector<NodePtr> body;
    skipNewlines();
    while (!at(TokenKind::End)) {
      std::vector<NodePtr> statements = parseStatementList();
      body.insert(body.end(), std::make_move_iterator(statements.begin()),
                  std::make_move_iterator(statements.end()));
      skipNewlines();
      if (!diagnostics.empty())
        synchronize();
    }
    NodePtr module = makeNode(actionHelperAstKind(
        "file", "_PyPegen_make_module", "Module", "Module"));
    addField(*module, "body", std::move(body));
    addField(*module, "type_ignores", makeTypeIgnoreNodes());
    return module;
  }

  SourceLocation lastEnd(const std::vector<NodePtr> &nodes,
                         SourceLocation fallback) const {
    for (auto it = nodes.rbegin(); it != nodes.rend(); ++it) {
      if (*it)
        return (*it)->range.end;
    }
    return fallback;
  }

  void rememberGroupBounds(const NodePtr &node, SourceLocation start,
                           SourceLocation end) {
    if (node) {
      groupStarts[node.get()] = start;
      groupEnds[node.get()] = end;
    }
  }

  SourceLocation extendedStart(const NodePtr &node) const {
    if (!node)
      return SourceLocation{};
    auto found = groupStarts.find(node.get());
    if (found != groupStarts.end())
      return found->second;
    return node->range.start;
  }

  SourceLocation extendedEnd(const NodePtr &node) const {
    if (!node)
      return SourceLocation{};
    auto found = groupEnds.find(node.get());
    if (found != groupEnds.end())
      return found->second;
    return node->range.end;
  }

  NodePtr parseInteractiveMode() {
    std::vector<NodePtr> body;
    if (at(TokenKind::End)) {
      error(current().range.start, "invalid syntax");
    } else if (match(TokenKind::Newline)) {
      body.push_back(
          makeNode(actionAstKindIfPresent("statement_newline", "Pass", "Pass"),
                   previous().range));
      skipNewlines();
    } else {
      std::vector<NodePtr> statements = parseStatementList();
      body.insert(body.end(), std::make_move_iterator(statements.begin()),
                  std::make_move_iterator(statements.end()));
      skipNewlines();
    }
    if (!at(TokenKind::End))
      error(current().range.start,
            "unexpected token after interactive statement");

    NodePtr root = makeNode(actionAstKind("interactive", "Interactive"));
    addField(*root, "body", std::move(body));
    return root;
  }

  NodePtr parseExpressionMode() {
    skipNewlines();
    NodePtr expr = parseEvalExpressions();
    skipNewlines();
    if (diagnostics.empty() && !at(TokenKind::End))
      error(current().range.start, "unexpected token after expression");
    NodePtr root = makeNode(actionAstKind("eval", "Expression"));
    addField(*root, "body", expr);
    return root;
  }

  NodePtr parseAnnotatedRhsMode() {
    skipNewlines();
    NodePtr expr = parseAnnotatedRhs();
    skipNewlines();
    if (diagnostics.empty() && !at(TokenKind::End))
      error(current().range.start, "unexpected token after expression");
    NodePtr root = makeNode("Expression");
    addField(*root, "body", expr);
    return root;
  }

  bool atInvalidEvalExpressionStart() const {
    return peg.matchesLiteral("yield_expr", pegToken(index), "yield") ||
           (at(TokenKind::Name) && index + 1 < tokens.size() &&
            tokens[index + 1].rawText == ":=");
  }

  NodePtr parseEvalExpressions() {
    if (atInvalidEvalExpressionStart()) {
      error(current().range.start, "invalid syntax");
      return makeNode("Error", current().range);
    }

    NodePtr first = parseExpression();
    if (!matchText(","))
      return first;

    std::vector<NodePtr> elements{first};
    while (!at(TokenKind::Newline) && !at(TokenKind::End)) {
      if (atText("*")) {
        error(current().range.start, "invalid syntax");
        break;
      }
      if (atInvalidEvalExpressionStart()) {
        error(current().range.start, "invalid syntax");
        break;
      }
      elements.push_back(parseExpression());
      if (!matchText(","))
        break;
    }

    NodePtr node =
        makeNode(actionAstKindIfPresent("expressions", "Tuple", "Tuple"),
                 SourceRange{first->range.start, previous().range.end});
    addField(*node, "elts", std::move(elements));
    addField(*node, "ctx", exprContext("Load"));
    return node;
  }

  NodePtr parseFunctionTypeMode() {
    skipNewlines();
    SourceLocation start = current().range.start;
    consumeText("(", "expected '(' before function type arguments");
    std::vector<NodePtr> argTypes;
    if (!atText(")"))
      argTypes = parseFunctionTypeArgs();
    consumeText(")", "expected ')' after function type arguments");
    consumeText("->", "expected '->' in function type");
    NodePtr returns = parseExpression();
    skipNewlines();
    if (!at(TokenKind::End))
      error(current().range.start, "unexpected token after function type");

    NodePtr root = makeNode(actionAstKind("func_type", "FunctionType"),
                            SourceRange{start, previous().range.end});
    addField(*root, "argtypes", std::move(argTypes));
    addField(*root, "returns", returns);
    return root;
  }

  std::vector<NodePtr> parseFunctionTypeArgs() {
    std::vector<NodePtr> argTypes;
    auto failAndRecover = [&] {
      error(current().range.start, "invalid syntax");
      while (!atText(")") && !at(TokenKind::End))
        ++index;
    };

    if (matchText("*")) {
      argTypes.push_back(parseExpression());
      if (matchText(",")) {
        if (!matchText("**")) {
          failAndRecover();
          return argTypes;
        }
        argTypes.push_back(parseExpression());
      }
      return argTypes;
    }

    if (matchText("**")) {
      argTypes.push_back(parseExpression());
      if (matchText(","))
        failAndRecover();
      return argTypes;
    }

    argTypes.push_back(parseExpression());
    while (matchText(",")) {
      if (atText(")")) {
        failAndRecover();
        return argTypes;
      }
      if (matchText("*")) {
        argTypes.push_back(parseExpression());
        if (matchText(",")) {
          if (!matchText("**")) {
            failAndRecover();
            return argTypes;
          }
          argTypes.push_back(parseExpression());
        }
        return argTypes;
      }
      if (matchText("**")) {
        argTypes.push_back(parseExpression());
        return argTypes;
      }
      argTypes.push_back(parseExpression());
    }
    return argTypes;
  }

private:
  std::vector<Token> tokens;
  Diagnostics &diagnostics;
  std::vector<TypeIgnoreInfo> typeIgnores;
  TypeCommentMap typeComments;
  CpythonPegAdapter peg;
  std::size_t index = 0;
  std::map<const Node *, SourceLocation> groupStarts;
  std::map<const Node *, SourceLocation> groupEnds;

  const Token &current() const { return tokens[index]; }
  const Token &previous() const { return tokens[index - 1]; }
  bool at(TokenKind kind) const { return current().kind == kind; }
  bool atText(std::string_view text) const { return current().rawText == text; }

  CpythonPegToken pegToken(std::size_t position) const {
    if (tokens.empty())
      return CpythonPegToken{};
    if (position >= tokens.size())
      position = tokens.size() - 1;
    const Token &token = tokens[position];
    return CpythonPegToken{token.rawText, token.cpythonName,
                           token.kind == TokenKind::Name,
                           token.kind == TokenKind::Keyword};
  }

  std::string actionAstKind(std::string_view rule,
                            std::string_view fallback) const {
    if (!peg.hasRule(rule)) {
      contractError("vendored CPython 3.14 PEG grammar is missing rule '" +
                    std::string(rule) + "' required to emit AST kind " +
                    std::string(fallback));
      return std::string(fallback);
    }
    std::optional<std::string> constructor = peg.singleAstConstructor(rule);
    if (!constructor) {
      contractError("vendored CPython 3.14 PEG grammar rule '" +
                    std::string(rule) +
                    "' no longer has a single AST constructor required to "
                    "emit " +
                    std::string(fallback));
      return std::string(fallback);
    }
    if (*constructor != fallback) {
      contractError("vendored CPython 3.14 PEG grammar rule '" +
                    std::string(rule) +
                    "' changed AST constructor from expected " +
                    std::string(fallback) + " to " + *constructor);
      return std::string(fallback);
    }
    return *constructor;
  }

  std::string actionAstKindIfPresent(std::string_view rule,
                                     std::string_view expected,
                                     std::string_view fallback) const {
    if (!peg.hasRule(rule)) {
      contractError("vendored CPython 3.14 PEG grammar is missing rule '" +
                    std::string(rule) + "' required to emit AST kind " +
                    std::string(expected));
      return std::string(fallback);
    }
    if (!peg.ruleHasAstConstructor(rule, expected)) {
      contractError(
          "vendored CPython 3.14 PEG grammar rule '" + std::string(rule) +
          "' no longer has expected AST constructor " + std::string(expected));
      return std::string(fallback);
    }
    return std::string(expected);
  }

  std::string actionHelperAstKind(std::string_view rule,
                                  std::string_view helper,
                                  std::string_view expected,
                                  std::string_view fallback) const {
    if (!peg.hasRule(rule)) {
      contractError("vendored CPython 3.14 PEG grammar is missing rule '" +
                    std::string(rule) + "' required to emit AST kind " +
                    std::string(expected));
      return std::string(fallback);
    }
    if (!peg.ruleCallsAction(rule, helper)) {
      contractError(
          "vendored CPython 3.14 PEG grammar rule '" + std::string(rule) +
          "' no longer calls expected helper action " + std::string(helper));
      return std::string(fallback);
    }
    if (!peg.hasAstNodeKind(expected)) {
      contractError("vendored CPython 3.14 ASDL has no AST node kind " +
                    std::string(expected));
      return std::string(fallback);
    }
    return std::string(expected);
  }

  void contractError(std::string message) const {
    diagnostics.push_back(
        Diagnostic{Severity::Error, current().range.start, std::move(message)});
  }

  bool match(TokenKind kind) {
    if (!at(kind))
      return false;
    ++index;
    return true;
  }

  bool matchText(std::string_view text) {
    if (!atText(text))
      return false;
    ++index;
    return true;
  }

  const Token &consume(TokenKind kind, std::string message) {
    if (at(kind))
      return tokens[index++];
    error(current().range.start, std::move(message));
    return current();
  }

  const Token &consumeText(std::string_view text, std::string message) {
    if (atText(text))
      return tokens[index++];
    error(current().range.start, std::move(message));
    return current();
  }

  bool matchPegLiteral(std::string_view rule, std::string_view literal) {
    if (!peg.matchesLiteral(rule, pegToken(index), literal))
      return false;
    return matchText(literal);
  }

  bool matchesAnyPegLiteral(std::initializer_list<std::string_view> rules,
                            std::string_view literal) const {
    CpythonPegToken token = pegToken(index);
    return std::any_of(rules.begin(), rules.end(), [&](std::string_view rule) {
      return peg.matchesLiteral(rule, token, literal);
    });
  }

  bool matchAnyPegLiteral(std::initializer_list<std::string_view> rules,
                          std::string_view literal) {
    if (!matchesAnyPegLiteral(rules, literal))
      return false;
    return matchText(literal);
  }

  const Token &consumePegLiteral(std::string_view rule,
                                 std::string_view literal,
                                 std::string message) {
    if (peg.matchesLiteral(rule, pegToken(index), literal))
      return consumeText(literal, std::move(message));
    error(current().range.start, std::move(message));
    return current();
  }

  void error(SourceLocation location, std::string message) {
    diagnostics.push_back(
        Diagnostic{Severity::Error, location, std::move(message)});
  }

  void consumeRejectedDefault(SourceLocation location, std::string message,
                              std::string_view terminator) {
    error(location, std::move(message));
    if (atText(",") || atText(terminator) || at(TokenKind::Newline) ||
        at(TokenKind::Dedent) || at(TokenKind::End))
      return;
    (void)parseExpression();
  }

  NodePtr parseParameterDefault(SourceLocation location,
                                std::string_view terminator) {
    if (atText(",") || atText(terminator) || at(TokenKind::Newline) ||
        at(TokenKind::Dedent) || at(TokenKind::End)) {
      error(location, "expected default value expression");
      return {};
    }
    return parseExpression();
  }

  void skipInvalidParameterTail(std::string_view terminator) {
    while (!atText(terminator) && !at(TokenKind::Newline) &&
           !at(TokenKind::Dedent) && !at(TokenKind::End))
      ++index;
  }

  void skipBalancedParentheses() {
    if (!matchText("("))
      return;
    int depth = 1;
    while (depth > 0 && !at(TokenKind::Newline) && !at(TokenKind::Dedent) &&
           !at(TokenKind::End)) {
      if (atText("(")) {
        ++depth;
      } else if (atText(")")) {
        --depth;
      }
      ++index;
    }
  }

  void skipNewlines() {
    while (match(TokenKind::Newline))
      ;
  }

  std::vector<NodePtr> makeTypeIgnoreNodes() const {
    std::vector<NodePtr> nodes;
    nodes.reserve(typeIgnores.size());
    for (const TypeIgnoreInfo &typeIgnore : typeIgnores) {
      NodePtr node = makeNode("TypeIgnore", typeIgnore.range);
      addField(*node, "lineno", static_cast<std::int64_t>(typeIgnore.line));
      addField(*node, "tag", typeIgnore.tag);
      nodes.push_back(std::move(node));
    }
    return nodes;
  }

  std::string takeTypeCommentOnLine(int line) {
    std::optional<TypeCommentInfo> typeComment =
        takeTypeCommentInfoOnLine(line);
    if (!typeComment)
      return std::string();
    return std::move(typeComment->text);
  }

  std::optional<TypeCommentInfo> takeTypeCommentInfoOnLine(int line) {
    auto found = typeComments.find(line);
    if (found == typeComments.end())
      return std::nullopt;
    TypeCommentInfo value = std::move(found->second);
    typeComments.erase(found);
    return value;
  }

  std::optional<TypeCommentInfo>
  takeTypeCommentInfoBeforeLineOffset(int line, int offset) {
    auto found = typeComments.find(line);
    if (found == typeComments.end() ||
        found->second.range.start.offset >= offset)
      return std::nullopt;
    TypeCommentInfo value = std::move(found->second);
    typeComments.erase(found);
    return value;
  }

  std::string takeTypeCommentBeforeLineOffset(int line, int offset) {
    std::optional<TypeCommentInfo> typeComment =
        takeTypeCommentInfoBeforeLineOffset(line, offset);
    if (!typeComment)
      return std::string();
    std::string value = std::move(typeComment->text);
    return value;
  }

  std::string takeFunctionTypeComment(int colonLine, int colonEndOffset,
                                      const std::vector<NodePtr> &body) {
    if (body.empty()) {
      auto inlineComment = typeComments.find(colonLine);
      if (inlineComment == typeComments.end() ||
          inlineComment->second.range.start.offset < colonEndOffset)
        return std::string();
      return takeTypeCommentBeforeLineOffset(colonLine,
                                             std::numeric_limits<int>::max());
    }

    int firstBodyLine =
        body.front() ? body.front()->range.start.line : colonLine + 1;

    std::vector<int> leadingLines;
    for (auto found = typeComments.lower_bound(colonLine + 1);
         found != typeComments.end() && found->first < firstBodyLine; ++found)
      leadingLines.push_back(found->first);

    auto inlineComment = typeComments.find(colonLine);
    const bool hasInline =
        inlineComment != typeComments.end() &&
        inlineComment->second.range.start.offset >= colonEndOffset;
    if (hasInline && !leadingLines.empty()) {
      if (auto found = typeComments.find(leadingLines.front());
          found != typeComments.end())
        error(found->second.range.start,
              "Cannot have two type comments on def");
    } else if (leadingLines.size() > 1) {
      if (auto found = typeComments.find(leadingLines[1]);
          found != typeComments.end())
        error(found->second.range.start,
              "Cannot have two type comments on def");
    }

    if (hasInline)
      return takeTypeCommentBeforeLineOffset(colonLine,
                                             std::numeric_limits<int>::max());
    if (leadingLines.empty())
      return std::string();

    std::string value;
    for (int line : leadingLines) {
      std::string current = takeTypeCommentOnLine(line);
      if (value.empty())
        value = std::move(current);
    }
    return value;
  }

  void synchronize() {
    while (!at(TokenKind::End) && !at(TokenKind::Newline) &&
           !at(TokenKind::Dedent))
      ++index;
    skipNewlines();
  }

  std::string compoundStatementHeadRule() const {
    CpythonPegToken nextToken = pegToken(index + 1);
    const CpythonPegToken *next =
        index + 1 < tokens.size() ? &nextToken : nullptr;
    return peg.compoundStatementHead(pegToken(index), next,
                                     looksLikeMatchStatement());
  }

  CpythonStatementForm currentStatementForm() const {
    CpythonPegToken nextToken = pegToken(index + 1);
    const CpythonPegToken *next =
        index + 1 < tokens.size() ? &nextToken : nullptr;
    return peg.statementForm(pegToken(index), next, looksLikeMatchStatement());
  }

  bool looksLikeMatchStatement() const {
    if (!atText("match"))
      return false;
    std::size_t cursor = index + 1;
    bool sawSubject = false;
    int depth = 0;
    while (cursor < tokens.size()) {
      const Token &token = tokens[cursor];
      if (token.kind == TokenKind::End || token.kind == TokenKind::Dedent)
        return false;
      if (depth == 0 &&
          (token.kind == TokenKind::Newline || token.rawText == ";"))
        return false;
      if (token.rawText == "(" || token.rawText == "[" ||
          token.rawText == "{") {
        ++depth;
      } else if (token.rawText == ")" || token.rawText == "]" ||
                 token.rawText == "}") {
        --depth;
      } else if (depth == 0 && token.rawText == ":") {
        if (!sawSubject)
          return false;
        if (cursor + 2 >= tokens.size() ||
            tokens[cursor + 1].kind != TokenKind::Newline ||
            tokens[cursor + 2].kind != TokenKind::Indent)
          return true;
        std::size_t body = cursor + 3;
        while (body < tokens.size() && tokens[body].kind == TokenKind::Newline)
          ++body;
        return body < tokens.size() && tokens[body].rawText == "case";
      }
      if (depth == 0 || token.kind != TokenKind::Newline)
        sawSubject = true;
      ++cursor;
    }
    return false;
  }

  std::string simpleStatementHeadRule() const {
    return peg.simpleStatementHead(pegToken(index), looksLikeTypeAlias());
  }

  bool isStandaloneClauseStart() const {
    CpythonPegToken token = pegToken(index);
    return peg.matchesLiteral("elif_stmt", token, "elif") ||
           peg.matchesLiteral("else_block", token, "else") ||
           peg.matchesLiteral("except_block", token, "except") ||
           peg.matchesLiteral("except_star_block", token, "except") ||
           peg.matchesLiteral("finally_block", token, "finally");
  }

  std::vector<NodePtr> parseStatementList() {
    if (isStandaloneClauseStart() ||
        currentStatementForm() == CpythonStatementForm::Compound)
      return std::vector<NodePtr>{parseStatement()};
    return parseSimpleStatementList();
  }

  NodePtr parseStatement() {
    const std::string compoundHead = compoundStatementHeadRule();
    if (compoundHead == "decorated")
      return parseDecorated();
    if (isStandaloneClauseStart())
      return parseInvalidStandaloneClause();
    if (compoundHead == "function_def" &&
        peg.matchesLiteral("function_def_raw", pegToken(index), "async"))
      return parseFunction(/*isAsync=*/true);
    if (compoundHead == "with_stmt" &&
        peg.matchesLiteral("with_stmt", pegToken(index), "async"))
      return parseWith(/*isAsync=*/true);
    if (compoundHead == "for_stmt" &&
        peg.matchesLiteral("for_stmt", pegToken(index), "async"))
      return parseFor(/*isAsync=*/true);
    if (compoundHead == "function_def")
      return parseFunction(/*isAsync=*/false);
    if (compoundHead == "class_def")
      return parseClass();
    if (compoundHead == "if_stmt")
      return parseIf();
    if (compoundHead == "while_stmt")
      return parseWhile();
    if (compoundHead == "for_stmt")
      return parseFor(/*isAsync=*/false);
    if (compoundHead == "with_stmt")
      return parseWith(/*isAsync=*/false);
    if (compoundHead == "match_stmt")
      return parseMatch();
    if (compoundHead == "try_stmt")
      return parseTry();
    return parseSimpleStatement();
  }

  NodePtr parseInvalidStandaloneClause() {
    SourceLocation start = current().range.start;
    error(start, "invalid syntax");
    skipInvalidClause();
    return makeNode("Error", SourceRange{start, previous().range.end});
  }

  void skipInvalidClause() {
    while (!at(TokenKind::Newline) && !at(TokenKind::Dedent) &&
           !at(TokenKind::End))
      ++index;
    if (!match(TokenKind::Newline))
      return;
    if (!match(TokenKind::Indent))
      return;
    int depth = 1;
    while (!at(TokenKind::End) && depth > 0) {
      if (match(TokenKind::Indent)) {
        ++depth;
        continue;
      }
      if (match(TokenKind::Dedent)) {
        --depth;
        continue;
      }
      ++index;
    }
  }

  std::vector<NodePtr> parseSimpleStatementList() {
    std::vector<NodePtr> statements;
    statements.push_back(parseSimpleStatement(/*consumeEnd=*/false));
    while (matchText(";")) {
      if (at(TokenKind::Newline) || at(TokenKind::Dedent) || at(TokenKind::End))
        break;
      statements.push_back(parseSimpleStatement(/*consumeEnd=*/false));
    }
    consumeStatementEnd();
    return statements;
  }

  NodePtr parseDecorated() {
    SourceLocation start = current().range.start;
    std::vector<NodePtr> decorators;
    while (matchText("@")) {
      decorators.push_back(parseExpression(/*allowNamedExpression=*/true));
      consume(TokenKind::Newline, "expected end of decorator");
    }
    const std::string compoundHead = compoundStatementHeadRule();
    NodePtr decorated;
    if (compoundHead == "function_def" &&
        peg.matchesLiteral("function_def_raw", pegToken(index), "async")) {
      decorated = parseFunction(/*isAsync=*/true);
    } else if (compoundHead == "function_def") {
      decorated = parseFunction(/*isAsync=*/false);
    } else if (compoundHead == "class_def") {
      decorated = parseClass();
    } else {
      error(current().range.start, "invalid syntax");
      skipInvalidClause();
      return makeNode("Error", SourceRange{start, previous().range.end});
    }
    setField(*decorated, "decorator_list", std::move(decorators));
    return decorated;
  }

  NodePtr parseFunction(bool isAsync) {
    SourceLocation start = current().range.start;
    if (isAsync)
      consumePegLiteral("function_def_raw", "async", "expected 'async'");
    consumePegLiteral("function_def_raw", "def", "expected 'def'");
    std::string name = consume(TokenKind::Name, "expected function name").text;
    std::vector<NodePtr> typeParams;
    if (peg.matchesLiteral("type_params", pegToken(index), "["))
      typeParams = parseTypeParams();
    consumeText("(", "expected '(' after function name");
    NodePtr arguments = parseParameters();
    consumeText(")", "expected ')' after parameter list");
    NodePtr returns;
    if (matchText("->"))
      returns = parseExpression();
    SourceRange colonRange = consumeText(":", "expected ':'").range;
    std::vector<NodePtr> body = parseSuite("function", start.line);
    std::string typeComment = takeFunctionTypeComment(
        colonRange.start.line, colonRange.end.offset, body);

    std::string kind =
        isAsync ? actionAstKindIfPresent("function_def_raw", "AsyncFunctionDef",
                                         "AsyncFunctionDef")
                : actionAstKindIfPresent("function_def_raw", "FunctionDef",
                                         "FunctionDef");
    NodePtr node =
        makeNode(std::move(kind),
                 SourceRange{start, lastEnd(body, previous().range.end)});
    addField(*node, "name", name);
    addField(*node, "args", arguments);
    addField(*node, "body", std::move(body));
    addField(*node, "decorator_list", std::vector<NodePtr>{});
    addField(*node, "returns", returns);
    addField(*node, "type_comment",
             typeComment.empty() ? FieldValue{} : FieldValue{typeComment});
    addField(*node, "type_params", std::move(typeParams));
    return node;
  }

  NodePtr parseParameters() {
    NodePtr arguments = makeNode("arguments");
    std::vector<NodePtr> posonlyargs;
    std::vector<NodePtr> args;
    NodePtr vararg;
    std::vector<NodePtr> kwonlyargs;
    std::vector<NodePtr> kwDefaults;
    NodePtr kwarg;
    std::vector<NodePtr> defaults;

    bool keywordOnly = false;
    bool seenPositionalDefault = false;
    bool seenSlash = false;
    bool bareStar = false;

    while (!atText(")") && !at(TokenKind::End)) {
      if (matchText("/")) {
        if (seenSlash)
          error(previous().range.start, "/ may appear only once");
        if (keywordOnly)
          error(previous().range.start, "/ must be ahead of *");
        if (args.empty() && !keywordOnly)
          error(previous().range.start, "at least one argument must precede /");
        posonlyargs.insert(posonlyargs.end(), args.begin(), args.end());
        args.clear();
        seenSlash = true;
        if (matchText(","))
          continue;
        break;
      }

      if (matchText("**")) {
        if (bareStar && kwonlyargs.empty())
          error(previous().range.start, "named arguments must follow bare *");
        kwarg = parseParameterArg();
        if (matchText("="))
          consumeRejectedDefault(
              previous().range.start,
              "var-keyword argument cannot have default value", ")");
        if (matchText(",") && !atText(")")) {
          error(current().range.start,
                "arguments cannot follow var-keyword argument");
          skipInvalidParameterTail(")");
        }
        break;
      }

      if (matchText("*")) {
        const bool duplicateStar = static_cast<bool>(vararg) || bareStar;
        if (duplicateStar)
          error(previous().range.start, "* argument may appear only once");
        keywordOnly = true;
        if (at(TokenKind::Name)) {
          NodePtr parsedVararg =
              parseParameterArg(/*allowStarAnnotation=*/true);
          if (matchText("="))
            consumeRejectedDefault(
                previous().range.start,
                "var-positional argument cannot have default value", ")");
          if (!duplicateStar)
            vararg = std::move(parsedVararg);
          if (matchText(","))
            continue;
          break;
        }
        bareStar = true;
        if (matchText(",")) {
          if (std::optional<TypeCommentInfo> typeComment =
                  takeTypeCommentInfoBeforeLineOffset(
                      previous().range.start.line,
                      parameterListEndOffsetFromCurrent())) {
            error(typeComment->range.start,
                  "bare * has associated type comment");
          }
          if (atText(")") || atText("**"))
            error(previous().range.start, "named arguments must follow bare *");
          continue;
        }
        if (atText(")"))
          error(previous().range.start, "named arguments must follow bare *");
        if (!atText(")"))
          error(current().range.start, "expected keyword-only parameter");
        break;
      }

      if (atText("(")) {
        error(current().range.start,
              "Function parameters cannot be parenthesized");
        skipBalancedParentheses();
        skipInvalidParameterTail(")");
        break;
      }

      NodePtr arg = parseParameterArg();
      NodePtr defaultValue;
      if (matchText("="))
        defaultValue = parseParameterDefault(previous().range.start, ")");
      if (atText("/") && index + 1 < tokens.size() &&
          tokens[index + 1].rawText == "*") {
        error(tokens[index + 1].range.start, "expected comma between / and *");
        skipInvalidParameterTail(")");
        break;
      }

      if (keywordOnly) {
        kwonlyargs.push_back(std::move(arg));
        kwDefaults.push_back(std::move(defaultValue));
      } else {
        if (defaultValue)
          seenPositionalDefault = true;
        else if (seenPositionalDefault)
          error(arg->range.start,
                "parameter without a default follows parameter with a "
                "default");
        args.push_back(std::move(arg));
        if (defaultValue)
          defaults.push_back(std::move(defaultValue));
      }

      if (matchText(","))
        continue;
      break;
    }

    addField(*arguments, "posonlyargs", std::move(posonlyargs));
    addField(*arguments, "args", std::move(args));
    addField(*arguments, "vararg", vararg);
    addField(*arguments, "kwonlyargs", std::move(kwonlyargs));
    addField(*arguments, "kw_defaults", std::move(kwDefaults));
    addField(*arguments, "kwarg", kwarg);
    addField(*arguments, "defaults", std::move(defaults));
    return arguments;
  }

  NodePtr parseParameterArg(bool allowStarAnnotation = false) {
    SourceLocation start = current().range.start;
    std::string name = consume(TokenKind::Name, "expected parameter name").text;
    NodePtr annotation;
    bool usedStarAnnotation = false;
    if (matchText(":")) {
      if (allowStarAnnotation) {
        usedStarAnnotation = atText("*");
        annotation = parseStarredExpression();
      } else {
        annotation = parseExpression();
      }
    }
    NodePtr arg = makeNode(usedStarAnnotation
                               ? actionAstKind("param_star_annotation", "arg")
                               : actionAstKind("param", "arg"),
                           SourceRange{start, previous().range.end});
    addField(*arg, "arg", name);
    addField(*arg, "annotation", annotation);
    std::string typeComment = takeTypeCommentBeforeLineOffset(
        start.line, parameterListEndOffsetFromCurrent());
    addField(*arg, "type_comment",
             typeComment.empty() ? FieldValue{} : FieldValue{typeComment});
    return arg;
  }

  int parameterListEndOffsetFromCurrent() const {
    int depth = 0;
    for (std::size_t cursor = index; cursor < tokens.size(); ++cursor) {
      std::string_view text = tokens[cursor].rawText;
      if (text == "(" || text == "[" || text == "{") {
        ++depth;
        continue;
      }
      if (text == ")" || text == "]" || text == "}") {
        if (depth == 0 && text == ")")
          return tokens[cursor].range.start.offset;
        if (depth > 0)
          --depth;
      }
    }
    return std::numeric_limits<int>::max();
  }

  NodePtr parseUnannotatedArg(std::string message) {
    SourceLocation start = current().range.start;
    std::string name = consume(TokenKind::Name, std::move(message)).text;
    NodePtr arg = makeNode(actionAstKind("lambda_param", "arg"),
                           SourceRange{start, previous().range.end});
    addField(*arg, "arg", name);
    addField(*arg, "annotation", NodePtr{});
    addField(*arg, "type_comment", FieldValue{});
    return arg;
  }

  struct CallArguments {
    std::vector<NodePtr> args;
    std::vector<NodePtr> keywords;
    SourceLocation end;
  };

  enum class StarredOperand { BitwiseOr, Expression };

  std::optional<std::string>
  constantKeywordAssignmentTarget(const NodePtr &argument) const {
    if (!argument || argument->kind != "Constant")
      return std::nullopt;
    for (const Field &field : argument->fields) {
      if (field.name != "value")
        continue;
      if (const auto *boolean = std::get_if<bool>(&field.value))
        return *boolean ? std::string("True") : std::string("False");
      if (std::holds_alternative<std::monostate>(field.value))
        return std::string("None");
      return std::nullopt;
    }
    return std::nullopt;
  }

  void consumeInvalidCallKeywordAssignment(const NodePtr &argument,
                                           std::string_view closeText) {
    if (std::optional<std::string> name =
            constantKeywordAssignmentTarget(argument)) {
      error(argument ? argument->range.start : current().range.start,
            "cannot assign to " + *name);
    } else {
      error(argument ? argument->range.start : current().range.start,
            "expression cannot contain assignment, perhaps you meant \"==\"?");
    }
    if (!atText(closeText) && !atText(",") && !at(TokenKind::Newline) &&
        !at(TokenKind::Dedent) && !at(TokenKind::End))
      (void)parseExpression();
  }

  CallArguments parseCallArguments(std::string_view closeText) {
    CallArguments result;
    bool seenKeywordArgument = false;
    bool seenKeywordUnpacking = false;
    if (!atText(closeText)) {
      do {
        if (matchText("**")) {
          SourceLocation keywordStart = previous().range.start;
          NodePtr value = parseExpression();
          if (matchText("=")) {
            error(keywordStart, "cannot assign to keyword argument unpacking");
            if (!atText(closeText) && !atText(",") && !at(TokenKind::Newline) &&
                !at(TokenKind::Dedent) && !at(TokenKind::End))
              (void)parseExpression();
            continue;
          }
          NodePtr keyword =
              makeNode(actionAstKindIfPresent("kwarg_or_double_starred",
                                              "keyword", "keyword"),
                       SourceRange{keywordStart, previous().range.end});
          addField(*keyword, "arg", FieldValue{});
          addField(*keyword, "value", value);
          result.keywords.push_back(std::move(keyword));
          seenKeywordUnpacking = true;
          continue;
        }
        if (at(TokenKind::Name) && index + 1 < tokens.size() &&
            tokens[index + 1].rawText == "=") {
          SourceLocation keywordStart = current().range.start;
          std::string name = current().text;
          ++index;
          consumeText("=", "expected '=' in keyword argument");
          NodePtr value = parseExpression();
          if (startsComprehensionClause()) {
            error(keywordStart,
                  "invalid syntax. Maybe you meant '==' or ':=' instead of "
                  "'='?");
            (void)parseComprehensionClauses();
            continue;
          }
          NodePtr keyword = makeNode(
              actionAstKindIfPresent("kwarg_or_starred", "keyword", "keyword"),
              SourceRange{keywordStart, previous().range.end});
          addField(*keyword, "arg", name);
          addField(*keyword, "value", value);
          result.keywords.push_back(std::move(keyword));
          seenKeywordArgument = true;
          continue;
        }
        NodePtr argument = parseStarredExpression(/*allowNamedExpression=*/true,
                                                  StarredOperand::Expression);
        const bool isStarred = argument && argument->kind == "Starred";
        if (isStarred && matchText("=")) {
          error(argument->range.start,
                "cannot assign to iterable argument unpacking");
          if (!atText(closeText) && !atText(",") && !at(TokenKind::Newline) &&
              !at(TokenKind::Dedent) && !at(TokenKind::End))
            (void)parseExpression();
          continue;
        }
        if (!isStarred && matchText("=")) {
          consumeInvalidCallKeywordAssignment(argument, closeText);
          continue;
        }
        if (seenKeywordUnpacking) {
          error(argument ? argument->range.start : current().range.start,
                isStarred ? "iterable argument unpacking follows keyword "
                            "argument unpacking"
                          : "positional argument follows keyword argument "
                            "unpacking");
        } else if (!isStarred && seenKeywordArgument) {
          error(argument ? argument->range.start : current().range.start,
                "positional argument follows keyword argument");
        }
        if (startsComprehensionClause()) {
          if (!result.args.empty() || seenKeywordArgument ||
              seenKeywordUnpacking) {
            error(argument ? argument->range.start : current().range.start,
                  "Generator expression must be parenthesized");
          }
          NodePtr gen = makeNode(
              actionAstKind("genexp", "GeneratorExp"),
              SourceRange{argument->range.start, previous().range.end});
          addField(*gen, "elt", argument);
          addField(*gen, "generators", parseComprehensionClauses());
          result.args.push_back(std::move(gen));
          break;
        }
        result.args.push_back(std::move(argument));
      } while (matchText(",") && !atText(closeText));
    }
    result.end =
        consumeText(closeText, "expected closing delimiter after arguments")
            .range.end;
    return result;
  }

  NodePtr parseClass() {
    SourceLocation start = current().range.start;
    consumePegLiteral("class_def_raw", "class", "expected 'class'");
    std::string name = consume(TokenKind::Name, "expected class name").text;
    std::vector<NodePtr> typeParams;
    if (peg.matchesLiteral("type_params", pegToken(index), "["))
      typeParams = parseTypeParams();
    std::vector<NodePtr> bases;
    std::vector<NodePtr> keywords;
    if (matchText("(")) {
      CallArguments arguments = parseCallArguments(")");
      bases = std::move(arguments.args);
      keywords = std::move(arguments.keywords);
    }
    consumeText(":", "expected ':'");
    std::vector<NodePtr> body = parseSuite("class", start.line);
    NodePtr node =
        makeNode(actionAstKind("class_def_raw", "ClassDef"),
                 SourceRange{start, lastEnd(body, previous().range.end)});
    addField(*node, "name", name);
    addField(*node, "bases", std::move(bases));
    addField(*node, "keywords", std::move(keywords));
    addField(*node, "body", std::move(body));
    addField(*node, "decorator_list", std::vector<NodePtr>{});
    addField(*node, "type_params", std::move(typeParams));
    return node;
  }

  NodePtr parseIf() {
    SourceLocation start = current().range.start;
    consumePegLiteral("if_stmt", "if", "expected 'if'");
    return parseIfRest(start, "if");
  }

  NodePtr parseIfRest(SourceLocation start, std::string_view header) {
    NodePtr test = parseExpression(/*allowNamedExpression=*/true);
    consumeText(":", "expected ':'");
    std::vector<NodePtr> body = parseSuite(header, start.line);
    std::vector<NodePtr> orelse;
    if (matchPegLiteral("elif_stmt", "elif")) {
      SourceLocation elifStart = previous().range.start;
      orelse.push_back(parseIfRest(elifStart, "elif"));
    } else if (matchPegLiteral("else_block", "else")) {
      SourceLocation elseStart = previous().range.start;
      consumeText(":", "expected ':'");
      orelse = parseSuite("else", elseStart.line);
      while (peg.matchesLiteral("elif_stmt", pegToken(index), "elif")) {
        error(current().range.start, "'elif' block follows an 'else' block");
        skipInvalidClause();
      }
    }
    SourceLocation end = orelse.empty() ? lastEnd(body, previous().range.end)
                                        : lastEnd(orelse, previous().range.end);
    NodePtr node = makeNode(
        actionAstKind(header == "elif" ? "elif_stmt" : "if_stmt", "If"),
        SourceRange{start, end});
    addField(*node, "test", test);
    addField(*node, "body", std::move(body));
    addField(*node, "orelse", std::move(orelse));
    return node;
  }

  NodePtr parseWhile() {
    SourceLocation start = current().range.start;
    consumePegLiteral("while_stmt", "while", "expected 'while'");
    NodePtr test = parseExpression(/*allowNamedExpression=*/true);
    consumeText(":", "expected ':'");
    std::vector<NodePtr> body = parseSuite("while", start.line);
    std::vector<NodePtr> orelse;
    if (matchPegLiteral("else_block", "else")) {
      SourceLocation elseStart = previous().range.start;
      consumeText(":", "expected ':'");
      orelse = parseSuite("else", elseStart.line);
    }
    SourceLocation end = orelse.empty() ? lastEnd(body, previous().range.end)
                                        : lastEnd(orelse, previous().range.end);
    NodePtr node =
        makeNode(actionAstKind("while_stmt", "While"), SourceRange{start, end});
    addField(*node, "test", test);
    addField(*node, "body", std::move(body));
    addField(*node, "orelse", std::move(orelse));
    return node;
  }

  NodePtr parseFor(bool isAsync) {
    SourceLocation start = current().range.start;
    if (isAsync)
      consumePegLiteral("for_stmt", "async", "expected 'async'");
    consumePegLiteral("for_stmt", "for", "expected 'for'");
    NodePtr target = parseForTarget();
    if (!peg.matchesLiteral("for_stmt", pegToken(index), "in")) {
      if (atText(":") || at(TokenKind::Newline) || at(TokenKind::Dedent) ||
          at(TokenKind::End)) {
        error(current().range.start, "'in' expected after for-loop variables");
      } else {
        error(target ? target->range.start : current().range.start,
              "cannot assign to expression");
        skipForTargetRemainder();
      }
    }
    NodePtr iter;
    if (matchPegLiteral("for_stmt", "in"))
      iter = parseForIterable();
    else
      iter = makeNode("Error",
                      SourceRange{current().range.start, current().range.end});
    consumeText(":", "expected ':'");
    std::vector<NodePtr> body = parseSuite("for", start.line);
    std::vector<NodePtr> orelse;
    if (matchPegLiteral("else_block", "else")) {
      SourceLocation elseStart = previous().range.start;
      consumeText(":", "expected ':'");
      orelse = parseSuite("else", elseStart.line);
    }
    std::string kind =
        isAsync ? actionAstKindIfPresent("for_stmt", "AsyncFor", "AsyncFor")
                : actionAstKindIfPresent("for_stmt", "For", "For");
    SourceLocation end = orelse.empty() ? lastEnd(body, previous().range.end)
                                        : lastEnd(orelse, previous().range.end);
    NodePtr node = makeNode(std::move(kind), SourceRange{start, end});
    addField(*node, "target", target);
    addField(*node, "iter", iter);
    addField(*node, "body", std::move(body));
    addField(*node, "orelse", std::move(orelse));
    std::string typeComment = takeTypeCommentOnLine(start.line);
    addField(*node, "type_comment",
             typeComment.empty() ? FieldValue{} : FieldValue{typeComment});
    return node;
  }

  void skipForTargetRemainder() {
    while (!peg.matchesLiteral("for_stmt", pegToken(index), "in") &&
           !atText(":") && !at(TokenKind::Newline) && !at(TokenKind::Dedent) &&
           !at(TokenKind::End))
      ++index;
  }

  NodePtr parseForIterable() {
    NodePtr first = parseStarredExpression();
    if (!matchText(","))
      return first;

    std::vector<NodePtr> elements{first};
    while (!atText(":") && !at(TokenKind::Newline) && !at(TokenKind::Dedent) &&
           !at(TokenKind::End)) {
      elements.push_back(parseStarredExpression());
      if (!matchText(","))
        break;
    }

    NodePtr node =
        makeNode(actionAstKindIfPresent("star_expressions", "Tuple", "Tuple"),
                 SourceRange{first->range.start, previous().range.end});
    addField(*node, "elts", std::move(elements));
    addField(*node, "ctx", exprContext("Load"));
    return node;
  }

  NodePtr parseWith(bool isAsync) {
    SourceLocation start = current().range.start;
    if (isAsync)
      consumePegLiteral("with_stmt", "async", "expected 'async'");
    consumePegLiteral("with_stmt", "with", "expected 'with'");
    const bool parenthesized = matchText("(");
    bool parseAsParenthesizedItemList = parenthesized;
    std::vector<NodePtr> items;
    if (parenthesized && hasTopLevelNamedExpressionBeforeClosingParen()) {
      SourceLocation groupStart = previous().range.start;
      NodePtr contextExpr = parseTupleOrGrouped(groupStart);
      items.push_back(finishWithItem(contextExpr->range.start, contextExpr));
      parseAsParenthesizedItemList = false;
      while (matchText(",")) {
        if (atText(":")) {
          error(previous().range.start,
                "trailing comma in with statement requires parentheses");
          break;
        }
        items.push_back(parseWithItem());
      }
    } else if (parenthesized && atText(")")) {
      error(current().range.start, "expected with item");
      consumeText(")", "expected ')' after with items");
      parseAsParenthesizedItemList = false;
    } else {
      do {
        items.push_back(parseWithItem());
      } while (matchText(",") && [&] {
        if (parenthesized)
          return !atText(")");
        if (atText(":")) {
          error(previous().range.start,
                "trailing comma in with statement requires parentheses");
          return false;
        }
        return true;
      }());
    }
    if (parseAsParenthesizedItemList)
      consumeText(")", "expected ')' after with item list");
    consumeText(":", "expected ':'");
    std::vector<NodePtr> body = parseSuite("with", start.line);
    std::string kind =
        isAsync ? actionAstKindIfPresent("with_stmt", "AsyncWith", "AsyncWith")
                : actionAstKindIfPresent("with_stmt", "With", "With");
    NodePtr node =
        makeNode(std::move(kind),
                 SourceRange{start, lastEnd(body, previous().range.end)});
    addField(*node, "items", std::move(items));
    addField(*node, "body", std::move(body));
    const bool typeCommentAllowed =
        !isAsync || !parenthesized || !parseAsParenthesizedItemList;
    std::string typeComment =
        typeCommentAllowed ? takeTypeCommentOnLine(start.line) : std::string{};
    addField(*node, "type_comment",
             typeComment.empty() ? FieldValue{} : FieldValue{typeComment});
    return node;
  }

  bool hasTopLevelNamedExpressionBeforeClosingParen() const {
    int depth = 0;
    for (std::size_t cursor = index; cursor < tokens.size(); ++cursor) {
      const Token &token = tokens[cursor];
      if (depth == 0 && token.rawText == ")")
        return false;
      if (depth == 0 && token.rawText == ":=")
        return true;
      if (token.rawText == "(" || token.rawText == "[" ||
          token.rawText == "{") {
        ++depth;
        continue;
      }
      if (token.rawText == ")" || token.rawText == "]" ||
          token.rawText == "}") {
        if (depth > 0)
          --depth;
        continue;
      }
      if (token.kind == TokenKind::End)
        return false;
    }
    return false;
  }

  NodePtr finishWithItem(SourceLocation start, NodePtr contextExpr) {
    NodePtr optionalVars;
    if (matchPegLiteral("with_item", "as")) {
      optionalVars = parseTargetAtom();
      validateTarget(optionalVars, TargetContext::Assign);
      setContext(optionalVars, "Store");
    }
    NodePtr node = makeNode(actionAstKind("with_item", "withitem"),
                            SourceRange{start, previous().range.end});
    addField(*node, "context_expr", contextExpr);
    addField(*node, "optional_vars", optionalVars);
    return node;
  }

  NodePtr parseWithItem() {
    SourceLocation start = current().range.start;
    NodePtr contextExpr = parseExpression();
    return finishWithItem(start, contextExpr);
  }

  NodePtr parseMatch() {
    SourceLocation start = current().range.start;
    consumePegLiteral("match_stmt", "match", "expected 'match'");
    NodePtr subject = parseMatchSubject();
    consumeText(":", "expected ':'");
    consume(TokenKind::Newline, "expected newline after match subject");
    if (!match(TokenKind::Indent)) {
      error(current().range.start,
            "expected an indented block after 'match' statement on line " +
                std::to_string(start.line));
      NodePtr node = makeNode(actionAstKind("match_stmt", "Match"),
                              SourceRange{start, previous().range.end});
      addField(*node, "subject", subject);
      addField(*node, "cases", std::vector<NodePtr>{});
      return node;
    }
    std::vector<NodePtr> cases;
    skipNewlines();
    while (!at(TokenKind::Dedent) && !at(TokenKind::End)) {
      cases.push_back(parseCaseBlock());
      skipNewlines();
    }
    consume(TokenKind::Dedent, "expected end of match statement");
    NodePtr node =
        makeNode(actionAstKind("match_stmt", "Match"),
                 SourceRange{start, lastEnd(cases, previous().range.end)});
    addField(*node, "subject", subject);
    addField(*node, "cases", std::move(cases));
    return node;
  }

  NodePtr parseMatchSubject() {
    SourceLocation start = current().range.start;
    NodePtr first = parseStarredExpression(/*allowNamedExpression=*/true);
    if (!matchText(","))
      return first;

    std::vector<NodePtr> elements{first};
    while (!atText(":") && !at(TokenKind::Newline) && !at(TokenKind::Dedent) &&
           !at(TokenKind::End)) {
      elements.push_back(parseStarredExpression(/*allowNamedExpression=*/true));
      if (!matchText(","))
        break;
    }
    NodePtr tuple = makeNode(actionAstKind("subject_expr", "Tuple"),
                             SourceRange{start, previous().range.end});
    addField(*tuple, "elts", std::move(elements));
    addField(*tuple, "ctx", exprContext("Load"));
    return tuple;
  }

  NodePtr parseCaseBlock() {
    SourceLocation start = current().range.start;
    consumePegLiteral("case_block", "case",
                      "expected 'case' in match statement");
    NodePtr pattern = parsePatterns();
    NodePtr guard;
    if (matchPegLiteral("guard", "if"))
      guard = parseExpression(/*allowNamedExpression=*/true);
    consumeText(":", "expected ':'");
    std::vector<NodePtr> body = parseSuite("case", start.line);
    NodePtr node =
        makeNode(actionAstKind("case_block", "match_case"),
                 SourceRange{start, lastEnd(body, previous().range.end)});
    addField(*node, "pattern", pattern);
    addField(*node, "guard", guard);
    addField(*node, "body", std::move(body));
    return node;
  }

  const std::string *stringField(const NodePtr &node,
                                 std::string_view name) const {
    const Field *field = findField(node, name);
    if (!field)
      return nullptr;
    return std::get_if<std::string>(&field->value);
  }

  NodePtr parsePatterns() {
    SourceLocation start = current().range.start;
    NodePtr first = parseMaybeStarPattern();
    if (!matchText(",")) {
      if (first && first->kind == "MatchStar") {
        error(first->range.start,
              "can't use starred name here; use sequence pattern instead");
        return makeNode("Error", SourceRange{start, previous().range.end});
      }
      return first;
    }

    std::vector<NodePtr> patterns{first};
    while (!atPatternTerminator()) {
      patterns.push_back(parseMaybeStarPattern());
      if (!matchText(","))
        break;
    }
    NodePtr node = makeNode(actionAstKind("sequence_pattern", "MatchSequence"),
                            SourceRange{start, previous().range.end});
    addField(*node, "patterns", std::move(patterns));
    return node;
  }

  NodePtr parseMaybeStarPattern() {
    SourceLocation start = current().range.start;
    if (!matchText("*"))
      return parsePattern();
    std::optional<std::string> name;
    if (atText("_")) {
      ++index;
    } else {
      name = consume(TokenKind::Name, "expected capture name after '*'").text;
      if (name == "_")
        name = std::nullopt;
    }
    NodePtr node = makeNode(actionAstKind("star_pattern", "MatchStar"),
                            SourceRange{start, previous().range.end});
    addField(*node, "name", name ? FieldValue{*name} : FieldValue{});
    return node;
  }

  NodePtr parsePattern() {
    SourceLocation start = current().range.start;
    NodePtr pattern = parseOrPattern();
    if (!matchPegLiteral("as_pattern", "as"))
      return pattern;
    if (atText("_")) {
      error(current().range.start, "cannot use '_' as a target");
      ++index;
      return makeNode("Error", SourceRange{start, previous().range.end});
    }
    if (!at(TokenKind::Name) ||
        (index + 1 < tokens.size() && tokens[index + 1].rawText == ".")) {
      SourceLocation targetStart = current().range.start;
      std::string targetName = patternTargetExpressionName();
      if (!atPatternTerminator())
        (void)parseExpression();
      error(targetStart, "cannot use " + targetName + " as pattern target");
      return makeNode("Error", SourceRange{start, previous().range.end});
    }
    std::string name =
        consume(TokenKind::Name, "expected pattern capture name").text;
    NodePtr node = makeNode(actionAstKind("capture_pattern", "MatchAs"),
                            SourceRange{start, previous().range.end});
    addField(*node, "pattern", pattern);
    addField(*node, "name", name);
    return node;
  }

  NodePtr parseOrPattern() {
    SourceLocation start = current().range.start;
    std::vector<NodePtr> patterns;
    patterns.push_back(parseClosedPattern());
    while (matchText("|"))
      patterns.push_back(parseClosedPattern());
    if (patterns.size() == 1)
      return patterns.front();
    NodePtr node =
        makeNode(actionAstKindIfPresent("or_pattern", "MatchOr", "MatchOr"),
                 SourceRange{start, previous().range.end});
    addField(*node, "patterns", std::move(patterns));
    return node;
  }

  bool atNumber(std::size_t cursor) const {
    return cursor < tokens.size() && tokens[cursor].kind == TokenKind::Number;
  }

  bool atImaginaryNumber(std::size_t cursor) const {
    return atNumber(cursor) && hasImaginarySuffix(tokens[cursor].text);
  }

  bool atRealNumber(std::size_t cursor) const {
    return atNumber(cursor) && !hasImaginarySuffix(tokens[cursor].text);
  }

  bool startsSignedRealNumber() const {
    if (atRealNumber(index))
      return true;
    return atText("-") && index + 1 < tokens.size() && atRealNumber(index + 1);
  }

  NodePtr parseLiteralPatternExpression() {
    SourceLocation start = current().range.start;
    if (startsSignedRealNumber()) {
      NodePtr left = atText("-") ? parseUnary() : parseAtom();
      if ((atText("+") || atText("-")) && index + 1 < tokens.size() &&
          atImaginaryNumber(index + 1)) {
        std::string op = current().rawText;
        ++index;
        NodePtr right = parseAtom();
        NodePtr node = makeNode(actionAstKind("sum", "BinOp"),
                                SourceRange{start, previous().range.end});
        addField(*node, "left", left);
        addField(*node, "op", binaryOperator(op));
        addField(*node, "right", right);
        return node;
      }
      return left;
    }
    if (atImaginaryNumber(index) || at(TokenKind::String))
      return parseAtom();
    return {};
  }

  NodePtr parseClosedPattern() {
    SourceLocation start = current().range.start;
    if (matchText("[")) {
      std::vector<NodePtr> patterns;
      if (!atText("]")) {
        do {
          patterns.push_back(parseMaybeStarPattern());
        } while (matchText(",") && !atText("]"));
      }
      consumeText("]", "expected ']' after sequence pattern");
      NodePtr node =
          makeNode(actionAstKind("sequence_pattern", "MatchSequence"),
                   SourceRange{start, previous().range.end});
      addField(*node, "patterns", std::move(patterns));
      return node;
    }

    if (matchText("("))
      return parseParenthesizedPattern(start);

    if (matchText("{"))
      return parseMappingPattern(start);

    if (NodePtr value = parseLiteralPatternExpression()) {
      NodePtr node = makeNode(
          actionAstKindIfPresent("literal_pattern", "MatchValue", "MatchValue"),
          SourceRange{start, previous().range.end});
      addField(*node, "value", value);
      return node;
    }

    if (at(TokenKind::Keyword) &&
        (atText("None") || atText("True") || atText("False"))) {
      FieldValue value;
      if (atText("True"))
        value = true;
      else if (atText("False"))
        value = false;
      SourceLocation end = current().range.end;
      ++index;
      NodePtr node =
          makeNode(actionAstKindIfPresent("literal_pattern", "MatchSingleton",
                                          "MatchSingleton"),
                   SourceRange{start, end});
      addField(*node, "value", std::move(value));
      return node;
    }

    if (at(TokenKind::Name))
      return parseNamePattern();

    error(start, "expected pattern");
    ++index;
    return makeNode("Error", SourceRange{start, previous().range.end});
  }

  NodePtr parseParenthesizedPattern(SourceLocation start) {
    if (matchText(")")) {
      NodePtr node =
          makeNode(actionAstKind("sequence_pattern", "MatchSequence"),
                   SourceRange{start, previous().range.end});
      addField(*node, "patterns", std::vector<NodePtr>{});
      return node;
    }
    NodePtr first = parseMaybeStarPattern();
    if (!matchText(",")) {
      if (first && first->kind == "MatchStar") {
        error(first->range.start,
              "can't use starred name here; use sequence pattern instead");
        consumeText(")", "expected ')' after group pattern");
        return makeNode("Error", SourceRange{start, previous().range.end});
      }
      consumeText(")", "expected ')' after group pattern");
      return first;
    }
    std::vector<NodePtr> patterns{first};
    while (!atText(")") && !at(TokenKind::End)) {
      patterns.push_back(parseMaybeStarPattern());
      if (!matchText(","))
        break;
    }
    consumeText(")", "expected ')' after sequence pattern");
    NodePtr node = makeNode(actionAstKind("sequence_pattern", "MatchSequence"),
                            SourceRange{start, previous().range.end});
    addField(*node, "patterns", std::move(patterns));
    return node;
  }

  NodePtr parseMappingPattern(SourceLocation start) {
    std::vector<NodePtr> keys;
    std::vector<NodePtr> patterns;
    std::optional<std::string> rest;
    if (!atText("}")) {
      do {
        if (matchText("**")) {
          rest = parsePatternCaptureName();
          matchText(",");
          break;
        }
        NodePtr key = parsePatternKey();
        keys.push_back(std::move(key));
        consumeText(":", "expected ':' in mapping pattern");
        patterns.push_back(parsePattern());
      } while (matchText(",") && !atText("}"));
      if (matchText(",")) {
        if (matchText("**"))
          rest = parsePatternCaptureName();
      }
    }
    consumeText("}", "expected '}' after mapping pattern");
    NodePtr node = makeNode(actionAstKind("mapping_pattern", "MatchMapping"),
                            SourceRange{start, previous().range.end});
    addField(*node, "keys", std::move(keys));
    addField(*node, "patterns", std::move(patterns));
    addField(*node, "rest", rest ? FieldValue{*rest} : FieldValue{});
    return node;
  }

  NodePtr parsePatternKey() {
    SourceLocation start = current().range.start;
    if (at(TokenKind::Name) && index + 1 < tokens.size() &&
        tokens[index + 1].rawText == ".")
      return parseDottedNameExpression();
    if (at(TokenKind::Name)) {
      error(start, "invalid syntax");
      ++index;
      return makeNode("Error", SourceRange{start, previous().range.end});
    }
    if (at(TokenKind::Keyword) &&
        (atText("None") || atText("True") || atText("False")))
      return parseAtom();
    if (NodePtr value = parseLiteralPatternExpression())
      return value;
    error(start, "expected literal or dotted name as mapping pattern key");
    ++index;
    return makeNode("Error", SourceRange{start, previous().range.end});
  }

  NodePtr parseNamePattern() {
    SourceLocation start = current().range.start;
    if (atText("_")) {
      SourceLocation end = current().range.end;
      ++index;
      NodePtr node = makeNode(actionAstKind("wildcard_pattern", "MatchAs"),
                              SourceRange{start, end});
      addField(*node, "pattern", NodePtr{});
      addField(*node, "name", FieldValue{});
      return node;
    }

    NodePtr nameOrAttr = parseDottedNameExpression();
    if (matchText("("))
      return parseClassPattern(std::move(nameOrAttr), start);

    if (nameOrAttr->kind == "Attribute") {
      NodePtr node = makeNode(actionAstKind("value_pattern", "MatchValue"),
                              SourceRange{start, previous().range.end});
      addField(*node, "value", nameOrAttr);
      return node;
    }

    const std::string *name = nodeName(*nameOrAttr);
    NodePtr node = makeNode(actionAstKind("capture_pattern", "MatchAs"),
                            SourceRange{start, previous().range.end});
    addField(*node, "pattern", NodePtr{});
    addField(*node, "name", name ? FieldValue{*name} : FieldValue{});
    return node;
  }

  NodePtr parseClassPattern(NodePtr cls, SourceLocation start) {
    std::vector<NodePtr> patterns;
    std::vector<std::string> kwdAttrs;
    std::vector<NodePtr> kwdPatterns;
    bool seenKeyword = false;
    if (!atText(")")) {
      do {
        if (at(TokenKind::Name) && index + 1 < tokens.size() &&
            tokens[index + 1].rawText == "=") {
          seenKeyword = true;
          std::string attr = current().text;
          kwdAttrs.push_back(std::move(attr));
          ++index;
          consumeText("=", "expected '=' in class pattern keyword");
          kwdPatterns.push_back(parsePattern());
          continue;
        }
        if (seenKeyword)
          error(current().range.start,
                "positional pattern follows keyword pattern");
        patterns.push_back(parsePattern());
      } while (matchText(",") && !atText(")"));
    }
    consumeText(")", "expected ')' after class pattern");
    NodePtr node = makeNode(actionAstKind("class_pattern", "MatchClass"),
                            SourceRange{start, previous().range.end});
    addField(*node, "cls", cls);
    addField(*node, "patterns", std::move(patterns));
    addField(*node, "kwd_attrs", std::move(kwdAttrs));
    addField(*node, "kwd_patterns", std::move(kwdPatterns));
    return node;
  }

  NodePtr parseDottedNameExpression() {
    Token name = consume(TokenKind::Name, "expected name");
    NodePtr expr = makeNode("Name", name.range);
    addField(*expr, "id", name.text);
    addField(*expr, "ctx", exprContext("Load"));
    while (matchText(".")) {
      Token attr = consume(TokenKind::Name, "expected attribute name");
      NodePtr node =
          makeNode(actionAstKindIfPresent("primary", "Attribute", "Attribute"),
                   SourceRange{expr->range.start, attr.range.end});
      addField(*node, "value", expr);
      addField(*node, "attr", attr.text);
      addField(*node, "ctx", exprContext("Load"));
      expr = std::move(node);
    }
    return expr;
  }

  const std::string *nodeName(const Node &node) const {
    for (const Field &field : node.fields) {
      if (field.name != "id")
        continue;
      return std::get_if<std::string>(&field.value);
    }
    return nullptr;
  }

  std::string parsePatternCaptureName() {
    if (atText("_"))
      error(current().range.start, "'_' cannot be used as a capture name");
    return consume(TokenKind::Name, "expected pattern capture name").text;
  }

  std::string patternTargetExpressionName() const {
    if (at(TokenKind::Number) || at(TokenKind::String) || atText("None") ||
        atText("True") || atText("False"))
      return "literal";
    return "expression";
  }

  bool atPatternTerminator() const {
    return atText(":") || atText("if") || atText(")") || atText("]") ||
           atText("}") || at(TokenKind::Newline) || at(TokenKind::Dedent) ||
           at(TokenKind::End);
  }

  NodePtr parseForTarget() {
    SourceLocation start = current().range.start;
    std::vector<NodePtr> elements;
    elements.push_back(parseTargetAtom());
    bool sawComma = false;
    while (matchText(",")) {
      sawComma = true;
      if (matchesAnyPegLiteral({"for_stmt", "for_if_clause"}, "in"))
        break;
      elements.push_back(parseTargetAtom());
    }
    NodePtr target;
    if (!sawComma) {
      target = std::move(elements.front());
    } else {
      target =
          makeNode(actionAstKindIfPresent("star_targets", "Tuple", "Tuple"),
                   SourceRange{start, previous().range.end});
      addField(*target, "elts", std::move(elements));
      addField(*target, "ctx", exprContext("Load"));
    }
    validateTarget(target, TargetContext::Assign);
    setContext(target, "Store");
    return target;
  }

  NodePtr parseTargetAtom(bool allowStar = true) {
    if (matchText("*")) {
      SourceLocation start = previous().range.start;
      if (atText("*"))
        error(current().range.start, "invalid syntax");
      NodePtr value = parseTargetAtom(allowStar);
      NodePtr node =
          makeNode(actionAstKindIfPresent("star_target", "Starred", "Starred"),
                   SourceRange{start, previous().range.end});
      addField(*node, "value", value);
      addField(*node, "ctx", exprContext("Load"));
      return node;
    }

    NodePtr target;
    if (matchText("(")) {
      SourceLocation start = previous().range.start;
      if (matchText(")")) {
        target = makeNode(actionAstKindIfPresent("star_atom", "Tuple", "Tuple"),
                          SourceRange{start, previous().range.end});
        addField(*target, "elts", std::vector<NodePtr>{});
        addField(*target, "ctx", exprContext("Load"));
        return parseTargetTrailers(std::move(target));
      }
      target = parseParenthesizedTarget(start, allowStar);
    } else if (matchText("[")) {
      SourceLocation start = previous().range.start;
      std::vector<NodePtr> elements;
      if (!atText("]")) {
        do {
          elements.push_back(parseTargetAtom(allowStar));
        } while (matchText(",") && !atText("]"));
      }
      consumeText("]", "expected ']' after target list");
      target = makeNode(actionAstKindIfPresent("star_atom", "List", "List"),
                        SourceRange{start, previous().range.end});
      addField(*target, "elts", std::move(elements));
      addField(*target, "ctx", exprContext("Load"));
    } else {
      Token name = consume(TokenKind::Name, "expected target");
      target = makeNode("Name", name.range);
      addField(*target, "id", name.text);
      addField(*target, "ctx", exprContext("Load"));
    }

    return parseTargetTrailers(std::move(target));
  }

  NodePtr parseParenthesizedTarget(SourceLocation start,
                                   bool allowStar = true) {
    std::vector<NodePtr> elements;
    elements.push_back(parseTargetAtom(allowStar));
    bool sawComma = false;
    while (matchText(",")) {
      sawComma = true;
      if (atText(")"))
        break;
      elements.push_back(parseTargetAtom(allowStar));
    }
    consumeText(")", "expected ')' after target");
    if (!sawComma && elements.size() == 1) {
      rememberGroupBounds(elements.front(), start, previous().range.end);
      return std::move(elements.front());
    }
    NodePtr tuple =
        makeNode(actionAstKindIfPresent("star_atom", "Tuple", "Tuple"),
                 SourceRange{start, previous().range.end});
    addField(*tuple, "elts", std::move(elements));
    addField(*tuple, "ctx", exprContext("Load"));
    return tuple;
  }

  NodePtr parseTargetTrailers(NodePtr target) {
    while (true) {
      if (matchText(".")) {
        std::string attr = consume(TokenKind::Name, "expected attribute").text;
        NodePtr node = makeNode(
            actionAstKindIfPresent("t_primary", "Attribute", "Attribute"),
            SourceRange{extendedStart(target), previous().range.end});
        addField(*node, "value", target);
        addField(*node, "attr", attr);
        addField(*node, "ctx", exprContext("Load"));
        target = std::move(node);
        continue;
      }
      if (matchText("(")) {
        CallArguments arguments = parseCallArguments(")");
        NodePtr node =
            makeNode(actionAstKindIfPresent("t_primary", "Call", "Call"),
                     SourceRange{extendedStart(target), arguments.end});
        addField(*node, "func", target);
        addField(*node, "args", std::move(arguments.args));
        addField(*node, "keywords", std::move(arguments.keywords));
        target = std::move(node);
        continue;
      }
      if (matchText("[")) {
        NodePtr slice = parseSubscriptSlice();
        NodePtr node = makeNode(
            actionAstKindIfPresent("t_primary", "Subscript", "Subscript"),
            SourceRange{extendedStart(target), previous().range.end});
        addField(*node, "value", target);
        addField(*node, "slice", slice);
        addField(*node, "ctx", exprContext("Load"));
        target = std::move(node);
        continue;
      }
      break;
    }
    return target;
  }

  NodePtr parseTry() {
    SourceLocation start = current().range.start;
    consumePegLiteral("try_stmt", "try", "expected 'try'");
    consumeText(":", "expected ':'");
    std::vector<NodePtr> body = parseSuite("try", start.line);
    std::vector<NodePtr> handlers;
    std::optional<bool> exceptStar;
    bool seenBareExcept = false;
    while (matchPegLiteral("except_block", "except")) {
      SourceLocation handlerStart = previous().range.start;
      if (seenBareExcept)
        error(handlerStart, "default 'except:' must be last");
      const bool isStarHandler = matchText("*");
      if (exceptStar && *exceptStar != isStarHandler)
        error(previous().range.start,
              "cannot have both 'except' and 'except*' on the same 'try'");
      exceptStar = isStarHandler;
      NodePtr type;
      std::optional<std::string> name;
      if (at(TokenKind::Newline) || at(TokenKind::Dedent) ||
          at(TokenKind::End)) {
        error(current().range.start,
              isStarHandler ? "expected one or more exception types"
                            : "expected ':'");
        skipInvalidClause();
        return makeNode("Error", SourceRange{start, previous().range.end});
      } else if (!atText(":")) {
        type = parseExceptTypeList(isStarHandler);
        if (matchPegLiteral("except_block", "as")) {
          name = parseExceptAlias(isStarHandler);
        }
      } else if (isStarHandler) {
        error(current().range.start, "expected one or more exception types");
      }
      consumeText(":", "expected ':'");
      std::vector<NodePtr> handlerBody = parseSuite(
          isStarHandler ? "except*" : "except", previous().range.start.line);
      NodePtr handler = makeNode(
          actionAstKind(isStarHandler ? "except_star_block" : "except_block",
                        "ExceptHandler"),
          SourceRange{handlerStart,
                      lastEnd(handlerBody, previous().range.end)});
      addField(*handler, "type", type);
      addField(*handler, "name", name ? FieldValue{*name} : FieldValue{});
      addField(*handler, "body", std::move(handlerBody));
      handlers.push_back(std::move(handler));
      if (!type && !isStarHandler)
        seenBareExcept = true;
    }
    std::vector<NodePtr> orelse;
    if (matchPegLiteral("else_block", "else")) {
      if (handlers.empty())
        error(previous().range.start,
              "try/else requires at least one except handler");
      SourceLocation elseStart = previous().range.start;
      consumeText(":", "expected ':'");
      orelse = parseSuite("else", elseStart.line);
    }
    std::vector<NodePtr> finalbody;
    if (matchPegLiteral("finally_block", "finally")) {
      SourceLocation finallyStart = previous().range.start;
      consumeText(":", "expected ':'");
      finalbody = parseSuite("finally", finallyStart.line);
    }
    if (handlers.empty() && finalbody.empty())
      error(current().range.start, "expected 'except' or 'finally' block");
    std::string kind =
        (exceptStar && *exceptStar)
            ? actionAstKindIfPresent("try_stmt", "TryStar", "TryStar")
            : actionAstKindIfPresent("try_stmt", "Try", "Try");
    SourceLocation end = lastEnd(finalbody, previous().range.end);
    if (finalbody.empty())
      end = lastEnd(orelse, previous().range.end);
    if (finalbody.empty() && orelse.empty())
      end = lastEnd(handlers, previous().range.end);
    if (finalbody.empty() && orelse.empty() && handlers.empty())
      end = lastEnd(body, previous().range.end);
    NodePtr node = makeNode(std::move(kind), SourceRange{start, end});
    addField(*node, "body", std::move(body));
    addField(*node, "handlers", std::move(handlers));
    addField(*node, "orelse", std::move(orelse));
    addField(*node, "finalbody", std::move(finalbody));
    return node;
  }

  NodePtr parseExceptTypeList(bool isStarHandler) {
    SourceLocation start = current().range.start;
    NodePtr first = parseExpression();
    if (!matchText(","))
      return first;

    std::vector<NodePtr> elements{first};
    while (!atText(":") &&
           !matchesAnyPegLiteral({"except_block", "except_star_block"}, "as") &&
           !at(TokenKind::Newline) && !at(TokenKind::Dedent) &&
           !at(TokenKind::End)) {
      elements.push_back(parseExpression());
      if (!matchText(","))
        break;
    }

    NodePtr tuple =
        makeNode(actionAstKindIfPresent("expressions", "Tuple", "Tuple"),
                 SourceRange{start, previous().range.end});
    addField(*tuple, "elts", std::move(elements));
    addField(*tuple, "ctx", exprContext("Load"));

    if (matchesAnyPegLiteral({"except_block", "except_star_block"}, "as")) {
      error(current().range.start,
            "cannot use " + std::string(isStarHandler ? "except*" : "except") +
                " statement with multiple exception types and 'as' alias");
    }
    return tuple;
  }

  std::optional<std::string> parseExceptAlias(bool isStarHandler) {
    if (at(TokenKind::Name) && index + 1 < tokens.size() &&
        tokens[index + 1].rawText == ":")
      return consume(TokenKind::Name, "expected exception alias").text;
    SourceLocation start = current().range.start;
    if (atText(":") || at(TokenKind::Newline) || at(TokenKind::Dedent) ||
        at(TokenKind::End)) {
      error(start, "expected exception alias");
      return std::nullopt;
    }
    NodePtr target = parseExpression();
    error(target ? target->range.start : start,
          "cannot use " + std::string(isStarHandler ? "except*" : "except") +
              " statement with " + expressionName(target));
    return std::nullopt;
  }

  std::string expressionName(const NodePtr &node) const {
    if (!node)
      return "expression";
    if (node->kind == "Attribute")
      return "attribute";
    if (node->kind == "Constant")
      return "literal";
    if (node->kind == "List")
      return "list";
    if (node->kind == "Tuple")
      return "tuple";
    if (node->kind == "Name")
      return "name";
    return "expression";
  }

  NodePtr parseImport(bool consumeEnd = true) {
    SourceLocation start = current().range.start;
    consumePegLiteral("import_name", "import", "expected 'import'");
    std::vector<NodePtr> names;
    if (at(TokenKind::Newline) || at(TokenKind::Dedent) || at(TokenKind::End)) {
      error(previous().range.end, "Expected one or more names after 'import'");
    } else {
      names = parseAliasList(/*allowTrailingComma=*/false);
    }
    if (atText("from")) {
      error(start, "Did you mean to use 'from ... import ...' instead?");
      skipToStatementEnd();
    }
    if (consumeEnd)
      consumeStatementEnd();
    NodePtr node = makeNode(actionAstKind("import_name", "Import"),
                            SourceRange{start, previous().range.end});
    addField(*node, "names", std::move(names));
    return node;
  }

  NodePtr parseImportFrom(bool consumeEnd = true) {
    SourceLocation start = current().range.start;
    consumePegLiteral("import_from", "from", "expected 'from'");
    int level = 0;
    while (atText(".") || atText("...")) {
      level += atText("...") ? 3 : 1;
      ++index;
    }
    std::optional<std::string> module;
    if (at(TokenKind::Name)) {
      module = parseDottedName();
    } else if (level == 0) {
      error(current().range.start, "expected module name in from import");
    }
    consumePegLiteral("import_from", "import",
                      "expected 'import' in from import");
    std::vector<NodePtr> names;
    if (matchText("*")) {
      NodePtr alias = makeNode(actionHelperAstKind(
          "import_from_targets", "_PyPegen_alias_for_star", "alias", "alias"));
      addField(*alias, "name", std::string("*"));
      addField(*alias, "asname", FieldValue{});
      names.push_back(std::move(alias));
      if (matchText(",")) {
        error(previous().range.start, "invalid syntax");
        skipToStatementEnd();
      }
    } else if (matchText("(")) {
      if (atText(")"))
        error(current().range.start, "expected one or more import targets");
      else
        names = parseAliasList(/*allowTrailingComma=*/true);
      consumeText(")", "expected ')' after import targets");
    } else if (at(TokenKind::Newline) || at(TokenKind::Dedent) ||
               at(TokenKind::End)) {
      error(previous().range.end, "Expected one or more names after 'import'");
    } else {
      names = parseAliasList(/*allowTrailingComma=*/false);
    }
    if (consumeEnd)
      consumeStatementEnd();
    NodePtr node = makeNode(
        actionAstKindIfPresent("import_from", "ImportFrom", "ImportFrom"),
        SourceRange{start, previous().range.end});
    addField(*node, "module", module ? FieldValue{*module} : FieldValue{});
    addField(*node, "names", std::move(names));
    addField(*node, "level", static_cast<std::int64_t>(level));
    return node;
  }

  NodePtr parseGlobal(bool consumeEnd = true) {
    SourceLocation start = current().range.start;
    consumePegLiteral("global_stmt", "global", "expected 'global'");
    std::vector<std::string> names = parseNameList("global");
    if (consumeEnd)
      consumeStatementEnd();
    NodePtr node = makeNode(actionAstKind("global_stmt", "Global"),
                            SourceRange{start, previous().range.end});
    addField(*node, "names", std::move(names));
    return node;
  }

  NodePtr parseNonlocal(bool consumeEnd = true) {
    SourceLocation start = current().range.start;
    consumePegLiteral("nonlocal_stmt", "nonlocal", "expected 'nonlocal'");
    std::vector<std::string> names = parseNameList("nonlocal");
    if (consumeEnd)
      consumeStatementEnd();
    NodePtr node = makeNode(actionAstKind("nonlocal_stmt", "Nonlocal"),
                            SourceRange{start, previous().range.end});
    addField(*node, "names", std::move(names));
    return node;
  }

  NodePtr parseDelete(bool consumeEnd = true) {
    SourceLocation start = current().range.start;
    consumePegLiteral("del_stmt", "del", "expected 'del'");
    std::vector<NodePtr> targets;
    do {
      NodePtr target = parseTargetAtom(/*allowStar=*/false);
      validateTarget(target, TargetContext::Delete);
      setContext(target, "Del");
      targets.push_back(std::move(target));
    } while (matchText(",") && !atText(";") && !at(TokenKind::Newline) &&
             !at(TokenKind::Dedent) && !at(TokenKind::End));
    if (consumeEnd)
      consumeStatementEnd();
    NodePtr node = makeNode(actionAstKind("del_stmt", "Delete"),
                            SourceRange{start, previous().range.end});
    addField(*node, "targets", std::move(targets));
    return node;
  }

  std::vector<NodePtr> parseAliasList(bool allowTrailingComma) {
    std::vector<NodePtr> aliases;
    do {
      SourceLocation start = current().range.start;
      std::string name = parseDottedName();
      std::optional<std::string> asname;
      if (matchAnyPegLiteral({"dotted_as_name", "import_from_as_name"}, "as")) {
        if (at(TokenKind::Name)) {
          Token aliasName = consume(TokenKind::Name, "expected alias name");
          if (!atImportAliasTerminator()) {
            error(aliasName.range.start, "cannot use " +
                                             importTargetExpressionName() +
                                             " as import target");
            skipInvalidImportTarget();
          } else {
            asname = aliasName.text;
          }
        } else {
          error(current().range.start, "cannot use " +
                                           importTargetExpressionName() +
                                           " as import target");
          skipInvalidImportTarget();
        }
      }
      NodePtr alias =
          makeNode(actionAstKindIfPresent("dotted_as_name", "alias", "alias"),
                   SourceRange{start, previous().range.end});
      addField(*alias, "name", name);
      addField(*alias, "asname", asname ? FieldValue{*asname} : FieldValue{});
      aliases.push_back(std::move(alias));
      if (!matchText(","))
        break;
      if (at(TokenKind::Newline) || at(TokenKind::Dedent) ||
          at(TokenKind::End) || atText(")")) {
        if (!allowTrailingComma)
          error(previous().range.start,
                "trailing comma not allowed without surrounding parentheses");
        break;
      }
    } while (true);
    return aliases;
  }

  bool atImportAliasTerminator() const {
    return at(TokenKind::Newline) || at(TokenKind::Dedent) ||
           at(TokenKind::End) || atText(",") || atText(")") || atText(";");
  }

  std::string importTargetExpressionName() const {
    if (at(TokenKind::Number) || at(TokenKind::String) || atText("None") ||
        atText("True") || atText("False"))
      return "literal";
    if (atText("."))
      return "attribute";
    return "expression";
  }

  void skipInvalidImportTarget() {
    if (atText(".")) {
      while (matchText(".")) {
        if (at(TokenKind::Name))
          ++index;
        else
          break;
      }
      return;
    }
    if (!atImportAliasTerminator())
      (void)parseExpression();
  }

  std::vector<std::string> parseNameList(std::string_view statementName) {
    std::vector<std::string> names;
    do {
      names.push_back(consume(TokenKind::Name, "expected name in " +
                                                   std::string(statementName) +
                                                   " statement")
                          .text);
    } while (matchText(","));
    return names;
  }

  std::string parseDottedName() {
    std::string name = consume(TokenKind::Name, "expected imported name").text;
    while (matchText(".")) {
      name += ".";
      name += consume(TokenKind::Name, "expected name after '.'").text;
    }
    return name;
  }

  bool looksLikeTypeAlias() const {
    if (!peg.matchesLiteral("type_alias", pegToken(index), "type") ||
        index + 2 >= tokens.size() || tokens[index + 1].kind != TokenKind::Name)
      return false;
    std::size_t cursor = index + 2;
    if (cursor < tokens.size() && tokens[cursor].rawText == "[") {
      int depth = 1;
      ++cursor;
      while (cursor < tokens.size() && depth > 0) {
        if (tokens[cursor].rawText == "[")
          ++depth;
        else if (tokens[cursor].rawText == "]")
          --depth;
        ++cursor;
      }
    }
    return cursor < tokens.size() && tokens[cursor].rawText == "=";
  }

  NodePtr parseTypeAlias(bool consumeEnd = true) {
    SourceLocation start = current().range.start;
    consumePegLiteral("type_alias", "type", "expected 'type'");
    Token nameToken = consume(TokenKind::Name, "expected type alias name");
    NodePtr name = makeNode("Name", nameToken.range);
    addField(*name, "id", nameToken.text);
    addField(*name, "ctx", exprContext("Store"));
    std::vector<NodePtr> typeParams;
    if (peg.matchesLiteral("type_params", pegToken(index), "["))
      typeParams = parseTypeParams();
    consumePegLiteral("type_alias", "=", "expected '=' in type alias");
    NodePtr value = parseExpression();
    if (consumeEnd)
      consumeStatementEnd();
    NodePtr node = makeNode(actionAstKind("type_alias", "TypeAlias"),
                            SourceRange{start, previous().range.end});
    addField(*node, "name", name);
    addField(*node, "type_params", std::move(typeParams));
    addField(*node, "value", value);
    return node;
  }

  std::vector<NodePtr> parseTypeParams() {
    consumePegLiteral("type_params", "[",
                      "expected '[' before type parameters");
    std::vector<NodePtr> params;
    if (atText("]"))
      error(current().range.start, "Type parameter list cannot be empty");
    if (!atText("]")) {
      do {
        params.push_back(parseTypeParam());
      } while (matchText(",") && !atText("]"));
    }
    consumePegLiteral("type_params", "]", "expected ']' after type parameters");
    return params;
  }

  NodePtr parseTypeParam() {
    SourceLocation start = current().range.start;
    std::string kind = "TypeVar";
    if (matchPegLiteral("type_param", "*"))
      kind = "TypeVarTuple";
    else if (matchPegLiteral("type_param", "**"))
      kind = "ParamSpec";
    Token name = consume(TokenKind::Name, "expected type parameter name");
    NodePtr bound;
    if (kind == "TypeVar" && matchPegLiteral("type_param_bound", ":"))
      bound = parseExpression();
    if (kind != "TypeVar" && matchPegLiteral("type_param_bound", ":")) {
      SourceLocation colon = previous().range.start;
      NodePtr invalidBound = parseExpression();
      bool constraints = invalidBound && invalidBound->kind == "Tuple";
      error(colon, constraints ? "cannot use constraints with " + kind
                               : "cannot use bound with " + kind);
    }
    NodePtr defaultValue;
    if (matchPegLiteral(kind == "TypeVarTuple" ? "type_param_starred_default"
                                               : "type_param_default",
                        "="))
      defaultValue =
          kind == "TypeVarTuple" ? parseStarredExpression() : parseExpression();
    std::string nodeKind =
        kind == "TypeVarTuple"
            ? actionAstKindIfPresent("type_param", "TypeVarTuple",
                                     "TypeVarTuple")
        : kind == "ParamSpec"
            ? actionAstKindIfPresent("type_param", "ParamSpec", "ParamSpec")
            : actionAstKindIfPresent("type_param", "TypeVar", "TypeVar");
    NodePtr node =
        makeNode(std::move(nodeKind), SourceRange{start, previous().range.end});
    addField(*node, "name", name.text);
    if (kind == "TypeVar")
      addField(*node, "bound", bound);
    addField(*node, "default_value", defaultValue);
    return node;
  }

  NodePtr parseSimpleStatement(bool consumeEnd = true) {
    SourceLocation start = current().range.start;
    const std::string statementHead = simpleStatementHeadRule();
    if (statementHead == "type_alias")
      return parseTypeAlias(consumeEnd);
    if (statementHead == "import_stmt")
      return peg.matchesLiteral("import_from", pegToken(index), "from")
                 ? parseImportFrom(consumeEnd)
                 : parseImport(consumeEnd);
    if (statementHead == "global_stmt")
      return parseGlobal(consumeEnd);
    if (statementHead == "nonlocal_stmt")
      return parseNonlocal(consumeEnd);
    if (statementHead == "del_stmt")
      return parseDelete(consumeEnd);
    if (statementHead == "yield_stmt") {
      NodePtr value = parseYield();
      if (matchText("=")) {
        error(value ? value->range.start : start,
              "assignment to yield expression not possible");
        if (!at(TokenKind::Newline) && !at(TokenKind::Dedent) &&
            !at(TokenKind::End))
          (void)parseAnnotatedRhs();
        if (consumeEnd)
          consumeStatementEnd();
        return makeNode("Error", SourceRange{start, previous().range.end});
      }
      if (consumeEnd)
        consumeStatementEnd();
      NodePtr node =
          makeNode(actionAstKindIfPresent("simple_stmt", "Expr", "Expr"),
                   SourceRange{start, previous().range.end});
      addField(*node, "value", value);
      return node;
    }
    if (statementHead == "pass_stmt" && matchPegLiteral("pass_stmt", "pass")) {
      if (consumeEnd)
        consumeStatementEnd();
      return makeNode(actionAstKind("pass_stmt", "Pass"),
                      SourceRange{start, previous().range.end});
    }
    if (statementHead == "break_stmt" &&
        matchPegLiteral("break_stmt", "break")) {
      if (consumeEnd)
        consumeStatementEnd();
      return makeNode(actionAstKind("break_stmt", "Break"),
                      SourceRange{start, previous().range.end});
    }
    if (statementHead == "continue_stmt" &&
        matchPegLiteral("continue_stmt", "continue")) {
      if (consumeEnd)
        consumeStatementEnd();
      return makeNode(actionAstKind("continue_stmt", "Continue"),
                      SourceRange{start, previous().range.end});
    }
    if (statementHead == "return_stmt" &&
        matchPegLiteral("return_stmt", "return")) {
      NodePtr value;
      if (!at(TokenKind::Newline) && !at(TokenKind::Dedent) &&
          !at(TokenKind::End))
        value = parseExpressionList();
      if (consumeEnd)
        consumeStatementEnd();
      NodePtr node = makeNode(actionAstKind("return_stmt", "Return"),
                              SourceRange{start, previous().range.end});
      addField(*node, "value", value);
      return node;
    }
    if (statementHead == "raise_stmt" &&
        matchPegLiteral("raise_stmt", "raise")) {
      NodePtr exc;
      NodePtr cause;
      if (!at(TokenKind::Newline) && !at(TokenKind::Dedent) &&
          !at(TokenKind::End))
        exc = parseExpression();
      if (exc && matchPegLiteral("raise_stmt", "from"))
        cause = parseExpression();
      if (consumeEnd)
        consumeStatementEnd();
      NodePtr node = makeNode(actionAstKind("raise_stmt", "Raise"),
                              SourceRange{start, previous().range.end});
      addField(*node, "exc", exc);
      addField(*node, "cause", cause);
      return node;
    }
    if (statementHead == "assert_stmt" &&
        matchPegLiteral("assert_stmt", "assert")) {
      NodePtr test = parseExpression();
      NodePtr msg;
      if (matchPegLiteral("assert_stmt", ","))
        msg = parseExpression();
      if (consumeEnd)
        consumeStatementEnd();
      NodePtr node = makeNode(actionAstKind("assert_stmt", "Assert"),
                              SourceRange{start, previous().range.end});
      addField(*node, "test", test);
      addField(*node, "msg", msg);
      return node;
    }

    NodePtr first = parseStarredExpression();
    if (matchText(",")) {
      std::vector<NodePtr> elements{first};
      while (!atText("=") && !atExpressionTerminator()) {
        elements.push_back(parseStarredExpression());
        if (!matchText(","))
          break;
      }
      NodePtr target = makeNode(
          "Tuple", SourceRange{first->range.start, previous().range.end});
      addField(*target, "elts", std::move(elements));
      addField(*target, "ctx", exprContext("Load"));
      if (matchText(":")) {
        error(first->range.start,
              "only single target (not tuple) can be annotated");
        (void)parseExpression();
        if (matchText("="))
          (void)parseExpressionList();
        if (consumeEnd)
          consumeStatementEnd();
        return makeNode("Error", SourceRange{start, previous().range.end});
      }
      if (!matchText("=")) {
        if (NodePtr op = parseAugAssignOperator()) {
          (void)op;
          setContext(target, "Store");
          invalidTarget(target, TargetContext::AugAssign);
          (void)parseAnnotatedRhs();
          if (consumeEnd)
            consumeStatementEnd();
          return makeNode("Error", SourceRange{start, previous().range.end});
        }
        if (consumeEnd)
          consumeStatementEnd();
        NodePtr node =
            makeNode(actionAstKindIfPresent("simple_stmt", "Expr", "Expr"),
                     SourceRange{start, previous().range.end});
        addField(*node, "value", target);
        return node;
      }
      validateTarget(target, TargetContext::Assign);
      setContext(target, "Store");
      NodePtr value = parseAnnotatedRhs();
      if (consumeEnd)
        consumeStatementEnd();
      std::optional<TypeCommentInfo> typeComment =
          takeTypeCommentInfoOnLine(start.line);
      SourceLocation end =
          typeComment ? typeComment->range.end : previous().range.end;
      NodePtr node =
          makeNode(actionAstKindIfPresent("assignment", "Assign", "Assign"),
                   SourceRange{start, end});
      addField(*node, "targets", std::vector<NodePtr>{target});
      addField(*node, "value", value);
      addField(*node, "type_comment",
               !typeComment || typeComment->text.empty()
                   ? FieldValue{}
                   : FieldValue{typeComment->text});
      return node;
    }
    if (matchText(":")) {
      validateTarget(first, TargetContext::AnnAssign);
      setContext(first, "Store");
      NodePtr annotation = parseExpression();
      NodePtr value;
      if (matchText("="))
        value = parseAnnotatedRhs();
      if (consumeEnd)
        consumeStatementEnd();
      NodePtr node = makeNode(
          actionAstKindIfPresent("assignment", "AnnAssign", "AnnAssign"),
          SourceRange{start, previous().range.end});
      addField(*node, "target", first);
      addField(*node, "annotation", annotation);
      addField(*node, "value", value);
      addField(*node, "simple",
               std::int64_t{first->kind == "Name" &&
                                    first->range.start.offset == start.offset
                                ? 1
                                : 0});
      return node;
    }
    if (matchText("=")) {
      validateTarget(first, TargetContext::Assign);
      setContext(first, "Store");
      std::vector<NodePtr> targets{first};
      NodePtr value;
      while (true) {
        value = parseAnnotatedRhs();
        if (!matchText("="))
          break;
        validateTarget(value, TargetContext::Assign);
        setContext(value, "Store");
        targets.push_back(std::move(value));
      }
      if (consumeEnd)
        consumeStatementEnd();
      std::optional<TypeCommentInfo> typeComment =
          takeTypeCommentInfoOnLine(start.line);
      SourceLocation end =
          typeComment ? typeComment->range.end : previous().range.end;
      NodePtr node =
          makeNode(actionAstKindIfPresent("assignment", "Assign", "Assign"),
                   SourceRange{start, end});
      addField(*node, "targets", std::move(targets));
      addField(*node, "value", value);
      addField(*node, "type_comment",
               !typeComment || typeComment->text.empty()
                   ? FieldValue{}
                   : FieldValue{typeComment->text});
      return node;
    }
    if (NodePtr op = parseAugAssignOperator()) {
      validateTarget(first, TargetContext::AugAssign);
      setContext(first, "Store");
      NodePtr value = parseAnnotatedRhs();
      if (consumeEnd)
        consumeStatementEnd();
      NodePtr node = makeNode(
          actionAstKindIfPresent("assignment", "AugAssign", "AugAssign"),
          SourceRange{start, previous().range.end});
      addField(*node, "target", first);
      addField(*node, "op", op);
      addField(*node, "value", value);
      return node;
    }

    if (isLegacyStatementExpression(first) && !atExpressionTerminator()) {
      const std::string *name = stringField(first, "id");
      error(first->range.start, "Missing parentheses in call to '" + *name +
                                    "'. Did you mean " + *name + "(...)?");
      skipToStatementEnd();
      if (consumeEnd)
        consumeStatementEnd();
      return makeNode("Error", SourceRange{start, previous().range.end});
    }

    if (consumeEnd)
      consumeStatementEnd();
    NodePtr node =
        makeNode(actionAstKindIfPresent("simple_stmt", "Expr", "Expr"),
                 SourceRange{start, previous().range.end});
    addField(*node, "value", first);
    return node;
  }

  NodePtr parseAugAssignOperator() {
    static const std::map<std::string, std::string> operators = {
        {"+=", "+"},   {"-=", "-"},   {"*=", "*"},   {"/=", "/"}, {"//=", "//"},
        {"%=", "%"},   {"@=", "@"},   {"&=", "&"},   {"|=", "|"}, {"^=", "^"},
        {"<<=", "<<"}, {">>=", ">>"}, {"**=", "**"},
    };
    auto found = operators.find(current().rawText);
    if (found == operators.end())
      return {};
    SourceRange range = current().range;
    ++index;
    return binaryOperator(found->second, range);
  }

  bool isLegacyStatementExpression(const NodePtr &node) const {
    if (!node || node->kind != "Name")
      return false;
    const std::string *name = stringField(node, "id");
    return name && (*name == "print" || *name == "exec");
  }

  NodePtr parseExpressionList() {
    NodePtr first = parseStarredExpression();
    if (!matchText(","))
      return first;

    std::vector<NodePtr> elements{first};
    while (!atText(";") && !atText(")") && !atText("]") && !atText("}") &&
           !at(TokenKind::Newline) && !at(TokenKind::Dedent) &&
           !at(TokenKind::End)) {
      elements.push_back(parseStarredExpression());
      if (!matchText(","))
        break;
    }
    NodePtr node =
        makeNode(actionAstKindIfPresent("star_expressions", "Tuple", "Tuple"),
                 SourceRange{first->range.start, previous().range.end});
    addField(*node, "elts", std::move(elements));
    addField(*node, "ctx", exprContext("Load"));
    return node;
  }

  NodePtr parseAnnotatedRhs() {
    if (peg.matchesLiteral("yield_expr", pegToken(index), "yield"))
      return parseYield();
    return parseExpressionList();
  }

  std::vector<NodePtr> parseSuite(std::string_view header = {},
                                  int headerLine = 0) {
    if (match(TokenKind::Newline)) {
      if (!match(TokenKind::Indent)) {
        if (header.empty()) {
          error(current().range.start, "expected indented block");
        } else if (header == "function" || header == "class") {
          error(current().range.start,
                "expected an indented block after " + std::string(header) +
                    " definition on line " + std::to_string(headerLine));
        } else {
          error(current().range.start,
                "expected an indented block after '" + std::string(header) +
                    "' statement on line " + std::to_string(headerLine));
        }
        return {};
      }
      std::vector<NodePtr> body;
      skipNewlines();
      while (!at(TokenKind::Dedent) && !at(TokenKind::End)) {
        std::vector<NodePtr> statements = parseStatementList();
        body.insert(body.end(), std::make_move_iterator(statements.begin()),
                    std::make_move_iterator(statements.end()));
        skipNewlines();
      }
      consume(TokenKind::Dedent, "expected end of block");
      return body;
    }
    return parseSimpleStatementList();
  }

  void consumeStatementEnd() {
    if (at(TokenKind::End) || at(TokenKind::Dedent))
      return;
    consume(TokenKind::Newline, "expected end of statement");
  }

  void skipToStatementEnd() {
    while (!at(TokenKind::Newline) && !at(TokenKind::Dedent) &&
           !at(TokenKind::End))
      ++index;
  }

  void skipToContainerEnd(std::string_view closingDelimiter) {
    while (!atText(closingDelimiter) && !at(TokenKind::Newline) &&
           !at(TokenKind::Dedent) && !at(TokenKind::End))
      ++index;
  }

  bool recoverMissingComma(std::string_view closingDelimiter) {
    if (atText(closingDelimiter) || at(TokenKind::Newline) ||
        at(TokenKind::Dedent) || at(TokenKind::End))
      return false;
    error(current().range.start, "invalid syntax. Perhaps you forgot a comma?");
    skipToContainerEnd(closingDelimiter);
    return true;
  }

  bool atExpressionTerminator() const {
    return at(TokenKind::Newline) || at(TokenKind::Dedent) ||
           at(TokenKind::End) || atText(";") || atText(")") || atText("]") ||
           atText("}") || atText(":");
  }

  NodePtr parseExpression(bool allowNamedExpression = false) {
    if (peg.startsLambda(pegToken(index)))
      return parseLambda();
    if (!allowNamedExpression)
      return parseIfExpression();
    CpythonPegToken nextToken = pegToken(index + 1);
    const CpythonPegToken *next =
        index + 1 < tokens.size() ? &nextToken : nullptr;
    if (peg.startsAssignmentExpression(pegToken(index), next))
      return parseNamedExpression();
    NodePtr expr = parseIfExpression();
    if (matchText(":=")) {
      error(expr ? expr->range.start : previous().range.start,
            "cannot use assignment expressions with " +
                invalidTargetName(expr));
      (void)parseExpression();
      return expr;
    }
    if (atText("=") && index + 1 < tokens.size() &&
        tokens[index + 1].rawText != "=" && tokens[index + 1].rawText != ":=") {
      if (expr && expr->kind == "Name") {
        error(expr->range.start,
              "invalid syntax. Maybe you meant '==' or ':=' instead of '='?");
      } else {
        error(expr ? expr->range.start : current().range.start,
              "cannot assign to " + invalidTargetName(expr) +
                  " here. Maybe you meant '==' instead of '='?");
      }
      ++index;
      (void)parseBitwiseOr();
    }
    return expr;
  }

  NodePtr parseNamedExpression() {
    SourceLocation start = current().range.start;
    Token name = consume(TokenKind::Name, "expected named expression target");
    consumeText(":=", "expected ':=' in named expression");
    NodePtr target = makeNode("Name", name.range);
    addField(*target, "id", name.text);
    addField(*target, "ctx", exprContext("Store"));
    NodePtr value = parseExpression();
    NodePtr node = makeNode(actionAstKind("assignment_expression", "NamedExpr"),
                            SourceRange{start, previous().range.end});
    addField(*node, "target", target);
    addField(*node, "value", value);
    return node;
  }

  NodePtr parseLambda() {
    SourceLocation start = current().range.start;
    consumePegLiteral("lambdef", "lambda", "expected 'lambda'");
    NodePtr args = atText(":") ? makeEmptyArguments() : parseLambdaParameters();
    consumeText(":", "expected ':' after lambda parameters");
    NodePtr body = parseExpression();
    NodePtr node = makeNode(actionAstKind("lambdef", "Lambda"),
                            SourceRange{start, previous().range.end});
    addField(*node, "args", args);
    addField(*node, "body", body);
    return node;
  }

  NodePtr makeEmptyArguments() {
    NodePtr arguments = makeNode("arguments");
    addField(*arguments, "posonlyargs", std::vector<NodePtr>{});
    addField(*arguments, "args", std::vector<NodePtr>{});
    addField(*arguments, "vararg", NodePtr{});
    addField(*arguments, "kwonlyargs", std::vector<NodePtr>{});
    addField(*arguments, "kw_defaults", std::vector<NodePtr>{});
    addField(*arguments, "kwarg", NodePtr{});
    addField(*arguments, "defaults", std::vector<NodePtr>{});
    return arguments;
  }

  NodePtr parseLambdaParameters() {
    NodePtr arguments = makeNode("arguments");
    std::vector<NodePtr> posonlyargs;
    std::vector<NodePtr> args;
    NodePtr vararg;
    std::vector<NodePtr> kwonlyargs;
    std::vector<NodePtr> kwDefaults;
    NodePtr kwarg;
    std::vector<NodePtr> defaults;
    bool keywordOnly = false;
    bool seenPositionalDefault = false;
    bool seenSlash = false;
    bool bareStar = false;

    while (!atText(":") && !at(TokenKind::End)) {
      if (matchText("/")) {
        if (seenSlash)
          error(previous().range.start, "/ may appear only once");
        if (keywordOnly)
          error(previous().range.start, "/ must be ahead of *");
        if (args.empty() && !keywordOnly)
          error(previous().range.start, "at least one argument must precede /");
        posonlyargs.insert(posonlyargs.end(), args.begin(), args.end());
        args.clear();
        seenSlash = true;
        if (matchText(","))
          continue;
        break;
      }

      if (matchText("**")) {
        if (bareStar && kwonlyargs.empty())
          error(previous().range.start, "named arguments must follow bare *");
        kwarg = parseUnannotatedArg("expected lambda ** parameter name");
        if (matchText("="))
          consumeRejectedDefault(
              previous().range.start,
              "var-keyword argument cannot have default value", ":");
        if (matchText(",") && !atText(":")) {
          error(current().range.start,
                "arguments cannot follow var-keyword argument");
          skipInvalidParameterTail(":");
        }
        break;
      }

      if (matchText("*")) {
        const bool duplicateStar = static_cast<bool>(vararg) || bareStar;
        if (duplicateStar)
          error(previous().range.start, "* argument may appear only once");
        keywordOnly = true;
        if (at(TokenKind::Name)) {
          NodePtr parsedVararg =
              parseUnannotatedArg("expected lambda * parameter name");
          if (matchText("="))
            consumeRejectedDefault(
                previous().range.start,
                "var-positional argument cannot have default value", ":");
          if (!duplicateStar)
            vararg = std::move(parsedVararg);
          if (matchText(","))
            continue;
          break;
        }
        bareStar = true;
        if (matchText(",")) {
          if (atText(":") || atText("**"))
            error(previous().range.start, "named arguments must follow bare *");
          continue;
        }
        if (atText(":"))
          error(previous().range.start, "named arguments must follow bare *");
        if (!atText(":"))
          error(current().range.start, "expected keyword-only lambda "
                                       "parameter");
        break;
      }

      if (atText("(")) {
        error(current().range.start,
              "Lambda expression parameters cannot be parenthesized");
        skipInvalidParameterTail(":");
        break;
      }

      NodePtr arg = parseUnannotatedArg("expected lambda parameter name");
      NodePtr defaultValue;
      if (matchText("="))
        defaultValue = parseParameterDefault(previous().range.start, ":");
      if (atText("/") && index + 1 < tokens.size() &&
          tokens[index + 1].rawText == "*") {
        error(tokens[index + 1].range.start, "expected comma between / and *");
        skipInvalidParameterTail(":");
        break;
      }

      if (keywordOnly) {
        kwonlyargs.push_back(std::move(arg));
        kwDefaults.push_back(std::move(defaultValue));
      } else {
        if (defaultValue)
          seenPositionalDefault = true;
        else if (seenPositionalDefault)
          error(arg->range.start,
                "parameter without a default follows parameter with a "
                "default");
        args.push_back(std::move(arg));
        if (defaultValue)
          defaults.push_back(std::move(defaultValue));
      }

      if (matchText(","))
        continue;
      break;
    }

    addField(*arguments, "posonlyargs", std::move(posonlyargs));
    addField(*arguments, "args", std::move(args));
    addField(*arguments, "vararg", vararg);
    addField(*arguments, "kwonlyargs", std::move(kwonlyargs));
    addField(*arguments, "kw_defaults", std::move(kwDefaults));
    addField(*arguments, "kwarg", kwarg);
    addField(*arguments, "defaults", std::move(defaults));
    return arguments;
  }

  NodePtr parseYield() {
    SourceLocation start = current().range.start;
    consumePegLiteral("yield_expr", "yield", "expected 'yield'");
    if (matchPegLiteral("yield_expr", "from")) {
      NodePtr value = parseExpression();
      NodePtr node = makeNode(
          actionAstKindIfPresent("yield_expr", "YieldFrom", "YieldFrom"),
          SourceRange{start, previous().range.end});
      addField(*node, "value", value);
      return node;
    }
    NodePtr value;
    if (!atExpressionTerminator())
      value = parseExpressionList();
    NodePtr node =
        makeNode(actionAstKindIfPresent("yield_expr", "Yield", "Yield"),
                 SourceRange{start, previous().range.end});
    addField(*node, "value", value);
    return node;
  }

  NodePtr
  parseStarredExpression(bool allowNamedExpression = false,
                         StarredOperand operand = StarredOperand::BitwiseOr) {
    SourceLocation start = current().range.start;
    if (!matchText("*"))
      return parseExpression(allowNamedExpression);
    NodePtr value = operand == StarredOperand::Expression ? parseExpression()
                                                          : parseBitwiseOr();
    NodePtr node = makeNode(
        actionAstKindIfPresent("star_expression", "Starred", "Starred"),
        SourceRange{start, previous().range.end});
    addField(*node, "value", value);
    addField(*node, "ctx", exprContext("Load"));
    return node;
  }

  CpythonComprehensionClauseForm currentComprehensionClauseForm() const {
    CpythonPegToken nextToken = pegToken(index + 1);
    const CpythonPegToken *next =
        index + 1 < tokens.size() ? &nextToken : nullptr;
    return peg.comprehensionClauseForm(pegToken(index), next);
  }

  bool startsComprehensionClause() const {
    return currentComprehensionClauseForm() !=
           CpythonComprehensionClauseForm::None;
  }

  void rejectStarredComprehensionElement(const NodePtr &element) {
    if (element && element->kind == "Starred")
      error(element->range.start,
            "iterable unpacking cannot be used in comprehension");
  }

  void rejectUnparenthesizedComprehensionTarget(const NodePtr &first,
                                                const NodePtr &last) {
    SourceLocation location =
        last ? last->range.end
             : (first ? first->range.start : current().range.start);
    error(location, "did you forget parentheses around the comprehension "
                    "target?");
  }

  std::vector<NodePtr> parseComprehensionClauses() {
    std::vector<NodePtr> generators;
    do {
      generators.push_back(parseComprehensionClause());
    } while (startsComprehensionClause());
    return generators;
  }

  NodePtr parseComprehensionClause() {
    SourceLocation start = current().range.start;
    bool isAsync = false;
    CpythonComprehensionClauseForm form = currentComprehensionClauseForm();
    if (form == CpythonComprehensionClauseForm::Async) {
      consumePegLiteral("for_if_clause", "async", "expected 'async'");
      isAsync = true;
    }
    consumePegLiteral("for_if_clause", "for",
                      "expected 'for' in comprehension");
    NodePtr target = parseForTarget();
    NodePtr iter;
    if (!matchPegLiteral("for_if_clause", "in")) {
      error(current().range.start, "'in' expected after for-loop variables");
      iter = makeNode("Error",
                      SourceRange{current().range.start, current().range.end});
    } else {
      iter = parseOr();
    }
    std::vector<NodePtr> ifs;
    while (peg.startsComprehensionIf(pegToken(index))) {
      consumePegLiteral("for_if_clause", "if",
                        "expected 'if' in comprehension");
      ifs.push_back(parseOr());
    }
    NodePtr node = makeNode(actionAstKind("for_if_clause", "comprehension"),
                            SourceRange{start, previous().range.end});
    addField(*node, "target", target);
    addField(*node, "iter", iter);
    addField(*node, "ifs", std::move(ifs));
    addField(*node, "is_async", static_cast<std::int64_t>(isAsync ? 1 : 0));
    return node;
  }

  NodePtr parseIfExpression() {
    NodePtr body = parseOr();
    if (!matchPegLiteral("expression", "if"))
      return body;
    NodePtr test = parseOr();
    NodePtr orelse;
    if (!matchPegLiteral("expression", "else")) {
      error(test ? test->range.start : current().range.start,
            "expected 'else' after 'if' expression");
      orelse = makeNode(
          "Error", SourceRange{current().range.start, current().range.start});
    } else if (atExpressionTerminator()) {
      error(current().range.start,
            "expected expression after 'else', but statement is given");
      orelse = makeNode(
          "Error", SourceRange{current().range.start, current().range.start});
    } else {
      orelse = parseExpression();
    }
    NodePtr node =
        makeNode(actionAstKindIfPresent("expression", "IfExp", "IfExp"),
                 SourceRange{extendedStart(body), previous().range.end});
    addField(*node, "test", test);
    addField(*node, "body", body);
    addField(*node, "orelse", orelse);
    return node;
  }

  NodePtr parseOr() {
    NodePtr expr = parseAnd();
    if (!peg.matchesLiteral("disjunction", pegToken(index), "or"))
      return expr;
    std::vector<NodePtr> values{expr};
    while (matchPegLiteral("disjunction", "or")) {
      values.push_back(parseAnd());
    }
    NodePtr node =
        makeNode(actionAstKind("disjunction", "BoolOp"),
                 SourceRange{extendedStart(expr), extendedEnd(values.back())});
    addField(*node, "op", boolOperator("or"));
    addField(*node, "values", std::move(values));
    return node;
  }

  NodePtr parseAnd() {
    NodePtr expr = parseInversion();
    if (!peg.matchesLiteral("conjunction", pegToken(index), "and"))
      return expr;
    std::vector<NodePtr> values{expr};
    while (matchPegLiteral("conjunction", "and")) {
      values.push_back(parseInversion());
    }
    NodePtr node =
        makeNode(actionAstKind("conjunction", "BoolOp"),
                 SourceRange{extendedStart(expr), extendedEnd(values.back())});
    addField(*node, "op", boolOperator("and"));
    addField(*node, "values", std::move(values));
    return node;
  }

  NodePtr parseInversion() {
    if (!peg.startsInversion(pegToken(index)))
      return parseComparison();
    SourceLocation start = current().range.start;
    ++index;
    NodePtr operand = parseInversion();
    NodePtr node =
        makeNode(actionAstKindIfPresent("inversion", "UnaryOp", "UnaryOp"),
                 SourceRange{start, previous().range.end});
    addField(*node, "op", unaryOperator("not"));
    addField(*node, "operand", operand);
    return node;
  }

  NodePtr parseComparison() {
    NodePtr left = parseBitwiseOr();
    NodePtr firstOp = parseComparisonOperator();
    if (!firstOp)
      return left;
    std::vector<NodePtr> ops;
    std::vector<NodePtr> comparators;
    do {
      ops.push_back(firstOp);
      comparators.push_back(parseBitwiseOr());
      firstOp = parseComparisonOperator();
    } while (firstOp);
    NodePtr node = makeNode(
        actionAstKind("comparison", "Compare"),
        SourceRange{extendedStart(left), extendedEnd(comparators.back())});
    addField(*node, "left", left);
    addField(*node, "ops", std::move(ops));
    addField(*node, "comparators", std::move(comparators));
    return node;
  }

  NodePtr parseComparisonOperator() {
    CpythonPegToken nextToken = pegToken(index + 1);
    const CpythonPegToken *next =
        index + 1 < tokens.size() ? &nextToken : nullptr;
    std::optional<CpythonComparisonOperator> op =
        peg.comparisonOperator(pegToken(index), next);
    if (!op)
      return {};

    SourceRange range = current().range;
    if (op->tokenCount == 2 && index + 1 < tokens.size())
      range.end = tokens[index + 1].range.end;
    index += op->tokenCount;
    return comparisonOperator(op->spelling, range);
  }

  NodePtr parseBitwiseOr() {
    NodePtr expr = parseBitwiseXor();
    while (std::optional<std::string_view> op = peg.infixOperator(
               CpythonInfixOperatorFamily::BitwiseOr, pegToken(index))) {
      std::string opText(*op);
      ++index;
      NodePtr right = parseBitwiseXor();
      NodePtr node =
          makeNode(actionAstKind("bitwise_or", "BinOp"),
                   SourceRange{extendedStart(expr), extendedEnd(right)});
      addField(*node, "left", expr);
      addField(*node, "op", binaryOperator(opText));
      addField(*node, "right", right);
      expr = std::move(node);
    }
    return expr;
  }

  NodePtr parseBitwiseXor() {
    NodePtr expr = parseBitwiseAnd();
    while (std::optional<std::string_view> op = peg.infixOperator(
               CpythonInfixOperatorFamily::BitwiseXor, pegToken(index))) {
      std::string opText(*op);
      ++index;
      NodePtr right = parseBitwiseAnd();
      NodePtr node =
          makeNode(actionAstKind("bitwise_xor", "BinOp"),
                   SourceRange{extendedStart(expr), extendedEnd(right)});
      addField(*node, "left", expr);
      addField(*node, "op", binaryOperator(opText));
      addField(*node, "right", right);
      expr = std::move(node);
    }
    return expr;
  }

  NodePtr parseBitwiseAnd() {
    NodePtr expr = parseShift();
    while (std::optional<std::string_view> op = peg.infixOperator(
               CpythonInfixOperatorFamily::BitwiseAnd, pegToken(index))) {
      std::string opText(*op);
      ++index;
      NodePtr right = parseShift();
      NodePtr node =
          makeNode(actionAstKind("bitwise_and", "BinOp"),
                   SourceRange{extendedStart(expr), extendedEnd(right)});
      addField(*node, "left", expr);
      addField(*node, "op", binaryOperator(opText));
      addField(*node, "right", right);
      expr = std::move(node);
    }
    return expr;
  }

  NodePtr parseShift() {
    NodePtr expr = parseAdditive();
    while (std::optional<std::string_view> op = peg.infixOperator(
               CpythonInfixOperatorFamily::Shift, pegToken(index))) {
      std::string opText(*op);
      ++index;
      NodePtr right = parseAdditive();
      NodePtr node =
          makeNode(actionAstKind("shift_expr", "BinOp"),
                   SourceRange{extendedStart(expr), extendedEnd(right)});
      addField(*node, "left", expr);
      addField(*node, "op", binaryOperator(opText));
      addField(*node, "right", right);
      expr = std::move(node);
    }
    return expr;
  }

  NodePtr parseAdditive() {
    NodePtr expr = parseMultiplicative();
    while (std::optional<std::string_view> op = peg.infixOperator(
               CpythonInfixOperatorFamily::Sum, pegToken(index))) {
      std::string opText(*op);
      ++index;
      NodePtr right = parseAdditiveRightHandSide();
      NodePtr node =
          makeNode(actionAstKind("sum", "BinOp"),
                   SourceRange{extendedStart(expr), extendedEnd(right)});
      addField(*node, "left", expr);
      addField(*node, "op", binaryOperator(opText));
      addField(*node, "right", right);
      expr = std::move(node);
    }
    return expr;
  }

  NodePtr parseMultiplicative() {
    NodePtr expr = parseUnary();
    while (std::optional<std::string_view> op = peg.infixOperator(
               CpythonInfixOperatorFamily::Term, pegToken(index))) {
      std::string opText(*op);
      ++index;
      NodePtr right = parseMultiplicativeRightHandSide();
      NodePtr node =
          makeNode(actionAstKind("term", "BinOp"),
                   SourceRange{extendedStart(expr), extendedEnd(right)});
      addField(*node, "left", expr);
      addField(*node, "op", binaryOperator(opText));
      addField(*node, "right", right);
      expr = std::move(node);
    }
    return expr;
  }

  NodePtr parseAdditiveRightHandSide() {
    if (peg.startsInversion(pegToken(index))) {
      error(current().range.start,
            "'not' after an operator must be parenthesized");
      return parseInversion();
    }
    return parseMultiplicative();
  }

  NodePtr parseMultiplicativeRightHandSide() {
    if (peg.startsInversion(pegToken(index))) {
      error(current().range.start,
            "'not' after an operator must be parenthesized");
      return parseInversion();
    }
    return parseUnary();
  }

  NodePtr parseUnary() {
    if (std::optional<std::string_view> op =
            peg.unaryOperator(pegToken(index))) {
      SourceLocation start = current().range.start;
      std::string opText(*op);
      ++index;
      NodePtr operand;
      if (peg.startsInversion(pegToken(index))) {
        error(current().range.start,
              "'not' after an operator must be parenthesized");
        operand = parseInversion();
      } else {
        operand = parseUnary();
      }
      NodePtr node =
          makeNode(actionAstKindIfPresent("factor", "UnaryOp", "UnaryOp"),
                   SourceRange{start, previous().range.end});
      addField(*node, "op", unaryOperator(opText));
      addField(*node, "operand", operand);
      return node;
    }
    return parsePower();
  }

  NodePtr parsePower() {
    NodePtr expr = parseAwait();
    std::optional<std::string_view> op =
        peg.infixOperator(CpythonInfixOperatorFamily::Power, pegToken(index));
    if (!op)
      return expr;
    std::string opText(*op);
    ++index;
    NodePtr right = parseUnary();
    NodePtr node =
        makeNode(actionAstKindIfPresent("power", "BinOp", "BinOp"),
                 SourceRange{extendedStart(expr), extendedEnd(right)});
    addField(*node, "left", expr);
    addField(*node, "op", binaryOperator(opText));
    addField(*node, "right", right);
    return node;
  }

  NodePtr parseAwait() {
    if (!peg.startsAwait(pegToken(index)))
      return parsePrimary();
    SourceLocation start = current().range.start;
    ++index;
    NodePtr value = parsePrimary();
    NodePtr node =
        makeNode(actionAstKindIfPresent("await_primary", "Await", "Await"),
                 SourceRange{start, previous().range.end});
    addField(*node, "value", value);
    return node;
  }

  NodePtr parsePrimary() {
    NodePtr expr = parseAtom();
    while (true) {
      switch (peg.primarySuffixForm(pegToken(index))) {
      case CpythonPrimarySuffixForm::None:
        return expr;
      case CpythonPrimarySuffixForm::Attribute: {
        consumeText(".", "expected '.'");
        std::string attr = consume(TokenKind::Name, "expected attribute").text;
        NodePtr node = makeNode(
            actionAstKindIfPresent("primary", "Attribute", "Attribute"),
            SourceRange{extendedStart(expr), previous().range.end});
        addField(*node, "value", expr);
        addField(*node, "attr", attr);
        addField(*node, "ctx", exprContext("Load"));
        expr = std::move(node);
        continue;
      }
      case CpythonPrimarySuffixForm::Call: {
        consumeText("(", "expected '('");
        CallArguments arguments = parseCallArguments(")");
        NodePtr node =
            makeNode(actionAstKindIfPresent("primary", "Call", "Call"),
                     SourceRange{extendedStart(expr), arguments.end});
        addField(*node, "func", expr);
        addField(*node, "args", std::move(arguments.args));
        addField(*node, "keywords", std::move(arguments.keywords));
        expr = std::move(node);
        continue;
      }
      case CpythonPrimarySuffixForm::Subscript: {
        consumeText("[", "expected '['");
        NodePtr slice = parseSubscriptSlice();
        NodePtr node = makeNode(
            actionAstKindIfPresent("primary", "Subscript", "Subscript"),
            SourceRange{extendedStart(expr), previous().range.end});
        addField(*node, "value", expr);
        addField(*node, "slice", slice);
        addField(*node, "ctx", exprContext("Load"));
        expr = std::move(node);
        continue;
      }
      }
    }
  }

  NodePtr parseSubscriptSlice() {
    NodePtr slice = parseSliceItem();
    if (matchText(",")) {
      std::vector<NodePtr> elements{slice};
      while (!atText("]") && !at(TokenKind::End)) {
        elements.push_back(parseSliceItem());
        if (!matchText(","))
          break;
      }
      SourceLocation sliceStart = elements.front()
                                      ? elements.front()->range.start
                                      : current().range.start;
      NodePtr tuple =
          makeNode(actionAstKindIfPresent("slices", "Tuple", "Tuple"),
                   SourceRange{sliceStart, previous().range.end});
      addField(*tuple, "elts", std::move(elements));
      addField(*tuple, "ctx", exprContext("Load"));
      slice = std::move(tuple);
    }
    consumeText("]", "expected ']' after subscript");
    return slice;
  }

  bool atSliceTerminator() const {
    return atText("]") || atText(",") || at(TokenKind::End);
  }

  NodePtr parseSliceItem() {
    SourceLocation start = current().range.start;
    NodePtr lower;
    if (!atText(":") && !atSliceTerminator())
      lower = parseStarredExpression(/*allowNamedExpression=*/true,
                                     StarredOperand::Expression);
    if (!matchText(":")) {
      if (lower)
        return lower;
      error(start, "expected subscript expression");
      return makeNode("Error", SourceRange{start, previous().range.end});
    }

    NodePtr upper;
    if (!atText(":") && !atSliceTerminator())
      upper = parseExpression();
    NodePtr step;
    if (matchText(":") && !atSliceTerminator())
      step = parseExpression();
    NodePtr node = makeNode(actionAstKind("slice", "Slice"),
                            SourceRange{start, previous().range.end});
    addField(*node, "lower", lower);
    addField(*node, "upper", upper);
    addField(*node, "step", step);
    return node;
  }

  NodePtr parseAtom() {
    SourceLocation start = current().range.start;
    switch (peg.atomForm(pegToken(index))) {
    case CpythonAtomForm::None:
      error(start, "expected expression");
      ++index;
      return makeNode("Error", SourceRange{start, previous().range.end});
    case CpythonAtomForm::Name: {
      consume(TokenKind::Name, "expected name");
      NodePtr node = makeNode("Name", SourceRange{start, previous().range.end});
      addField(*node, "id", previous().text);
      addField(*node, "ctx", exprContext("Load"));
      return node;
    }
    case CpythonAtomForm::Singleton: {
      ++index;
      const std::string text = previous().text;
      NodePtr node =
          makeNode(actionAstKindIfPresent("atom", "Constant", "Constant"),
                   SourceRange{start, previous().range.end});
      if (text == "True")
        addField(*node, "value", true);
      else if (text == "False")
        addField(*node, "value", false);
      else
        addField(*node, "value", FieldValue{});
      addField(*node, "kind", FieldValue{});
      return node;
    }
    case CpythonAtomForm::Number:
      consume(TokenKind::Number, "expected number");
      return parseNumberConstant(start, previous());
    case CpythonAtomForm::Strings:
      return parseStringLiteralSequence(start);
    case CpythonAtomForm::Ellipsis: {
      consumeText("...", "expected ellipsis");
      NodePtr node =
          makeNode(actionAstKindIfPresent("atom", "Constant", "Constant"),
                   SourceRange{start, previous().range.end});
      addField(*node, "value", Ellipsis{});
      addField(*node, "kind", FieldValue{});
      return node;
    }
    case CpythonAtomForm::Parenthesized:
      consumeText("(", "expected '('");
      return parseTupleOrGrouped(start);
    case CpythonAtomForm::List:
      consumeText("[", "expected '['");
      return parseList(start);
    case CpythonAtomForm::DictOrSet:
      consumeText("{", "expected '{'");
      return parseDict(start);
    }
    error(start, "expected expression");
    ++index;
    return makeNode("Error", SourceRange{start, previous().range.end});
  }

  NodePtr parseNumberConstant(SourceLocation start, const Token &token) {
    NodePtr node =
        makeNode(actionAstKindIfPresent("atom", "Constant", "Constant"),
                 SourceRange{start, token.range.end});
    if (hasImaginarySuffix(token.text)) {
      std::optional<std::complex<double>> value =
          parseImaginaryLiteral(token.text);
      if (!value) {
        error(start, "invalid imaginary literal: " + token.text);
        return makeNode("Error", SourceRange{start, token.range.end});
      }
      addField(*node, "value", *value);
      addField(*node, "kind", FieldValue{});
      return node;
    }
    if (isFloatLiteralText(token.text)) {
      std::optional<double> value = parseFloatLiteral(token.text);
      if (!value) {
        error(start, "invalid float literal: " + token.text);
        return makeNode("Error", SourceRange{start, token.range.end});
      }
      addField(*node, "value", *value);
      addField(*node, "kind", FieldValue{});
      return node;
    }
    std::optional<std::int64_t> value = parseIntegerLiteral(token.text);
    if (value) {
      addField(*node, "value", *value);
      addField(*node, "kind", FieldValue{});
      return node;
    }

    std::optional<std::string> decimal = parseIntegerLiteralDecimal(token.text);
    if (!decimal) {
      error(start, "invalid integer literal: " + token.text);
      return makeNode("Error", SourceRange{start, token.range.end});
    }
    addField(*node, "value", BigInteger{*decimal});
    addField(*node, "kind", FieldValue{});
    return node;
  }

  NodePtr parseTupleOrGrouped(SourceLocation start) {
    if (matchText(")")) {
      NodePtr node = makeNode(actionAstKind("tuple", "Tuple"),
                              SourceRange{start, previous().range.end});
      addField(*node, "elts", std::vector<NodePtr>{});
      addField(*node, "ctx", exprContext("Load"));
      return node;
    }
    if (matchText("**")) {
      SourceLocation dstar = previous().range.start;
      error(dstar, "cannot use double starred expression here");
      (void)parseExpression();
      consumeText(")", "expected ')' after grouped expression");
      return makeNode("Error", SourceRange{start, previous().range.end});
    }
    if (peg.matchesLiteral("yield_expr", pegToken(index), "yield")) {
      NodePtr value = parseYield();
      consumeText(")", "expected ')' after grouped expression");
      return value;
    }
    NodePtr first = parseStarredExpression(/*allowNamedExpression=*/true);
    if (startsComprehensionClause()) {
      rejectStarredComprehensionElement(first);
      std::vector<NodePtr> generators = parseComprehensionClauses();
      consumeText(")", "expected ')' after generator expression");
      NodePtr node = makeNode(actionAstKind("genexp", "GeneratorExp"),
                              SourceRange{start, previous().range.end});
      addField(*node, "elt", first);
      addField(*node, "generators", std::move(generators));
      return node;
    }
    if (!matchText(",")) {
      if (first && first->kind == "Starred") {
        error(first->range.start, "cannot use starred expression here");
        consumeText(")", "expected ')' after grouped expression");
        return makeNode("Error", SourceRange{start, previous().range.end});
      }
      if (recoverMissingComma(")")) {
        consumeText(")", "expected ')' after grouped expression");
        return makeNode("Error", SourceRange{start, previous().range.end});
      }
      consumeText(")", "expected ')' after grouped expression");
      rememberGroupBounds(first, start, previous().range.end);
      return first;
    }
    std::vector<NodePtr> elements{first};
    while (!atText(")") && !at(TokenKind::End)) {
      elements.push_back(parseStarredExpression(/*allowNamedExpression=*/true));
      if (!matchText(","))
        break;
    }
    if (recoverMissingComma(")")) {
      consumeText(")", "expected ')' after tuple");
      return makeNode("Error", SourceRange{start, previous().range.end});
    }
    consumeText(")", "expected ')' after tuple");
    NodePtr node = makeNode(actionAstKind("tuple", "Tuple"),
                            SourceRange{start, previous().range.end});
    addField(*node, "elts", std::move(elements));
    addField(*node, "ctx", exprContext("Load"));
    return node;
  }

  NodePtr parseList(SourceLocation start) {
    std::vector<NodePtr> elements;
    if (!atText("]")) {
      NodePtr first = parseStarredExpression(/*allowNamedExpression=*/true);
      if (startsComprehensionClause()) {
        rejectStarredComprehensionElement(first);
        std::vector<NodePtr> generators = parseComprehensionClauses();
        consumeText("]", "expected ']' after list comprehension");
        NodePtr node = makeNode(actionAstKind("listcomp", "ListComp"),
                                SourceRange{start, previous().range.end});
        addField(*node, "elt", first);
        addField(*node, "generators", std::move(generators));
        return node;
      }
      elements.push_back(std::move(first));
      while (matchText(",") && !atText("]")) {
        NodePtr element = parseStarredExpression(/*allowNamedExpression=*/true);
        if (startsComprehensionClause()) {
          rejectUnparenthesizedComprehensionTarget(elements.front(), element);
          (void)parseComprehensionClauses();
          break;
        }
        elements.push_back(std::move(element));
      }
    }
    if (recoverMissingComma("]")) {
      consumeText("]", "expected ']' after list");
      return makeNode("Error", SourceRange{start, previous().range.end});
    }
    consumeText("]", "expected ']' after list");
    NodePtr node = makeNode(actionAstKind("list", "List"),
                            SourceRange{start, previous().range.end});
    addField(*node, "elts", std::move(elements));
    addField(*node, "ctx", exprContext("Load"));
    return node;
  }

  NodePtr parseDictValue(SourceLocation colon) {
    if (atText("}") || atText(",") || at(TokenKind::Newline) ||
        at(TokenKind::Dedent) || at(TokenKind::End)) {
      error(colon, "expression expected after dictionary key and ':'");
      return makeNode("Error", SourceRange{colon, colon});
    }
    if (matchText("*")) {
      error(previous().range.start,
            "cannot use a starred expression in a dictionary value");
      (void)parseExpression();
      return makeNode("Error", SourceRange{colon, previous().range.end});
    }
    return parseExpression();
  }

  void rejectDictUnpackAssignment(SourceLocation unpackStart) {
    if (!matchText(":"))
      return;
    error(unpackStart, "invalid syntax");
    if (!atText("}") && !atText(",") && !at(TokenKind::Newline) &&
        !at(TokenKind::Dedent) && !at(TokenKind::End))
      (void)parseExpression();
  }

  NodePtr parseDict(SourceLocation start) {
    if (matchText("}")) {
      NodePtr node = makeNode(actionAstKind("dict", "Dict"),
                              SourceRange{start, previous().range.end});
      addField(*node, "keys", std::vector<NodePtr>{});
      addField(*node, "values", std::vector<NodePtr>{});
      return node;
    }

    std::vector<NodePtr> keys;
    std::vector<NodePtr> values;
    if (matchText("**")) {
      SourceLocation unpackStart = previous().range.start;
      keys.push_back(NodePtr{});
      values.push_back(parseExpression());
      rejectDictUnpackAssignment(unpackStart);
      if (startsComprehensionClause())
        error(start, "dict unpacking cannot be used in dict comprehension");
      while (matchText(",") && !atText("}")) {
        if (matchText("**")) {
          SourceLocation nestedUnpackStart = previous().range.start;
          keys.push_back(NodePtr{});
          values.push_back(parseExpression());
          rejectDictUnpackAssignment(nestedUnpackStart);
          continue;
        }
        keys.push_back(parseExpression());
        SourceLocation colon =
            consumeText(":", "expected ':' in dict item").range.start;
        values.push_back(parseDictValue(colon));
      }
      consumeText("}", "expected '}' after dict");
      NodePtr node = makeNode(actionAstKind("dict", "Dict"),
                              SourceRange{start, previous().range.end});
      addField(*node, "keys", std::move(keys));
      addField(*node, "values", std::move(values));
      return node;
    }

    NodePtr first = parseStarredExpression(/*allowNamedExpression=*/true);
    if (!matchText(":")) {
      if (startsComprehensionClause()) {
        rejectStarredComprehensionElement(first);
        std::vector<NodePtr> generators = parseComprehensionClauses();
        consumeText("}", "expected '}' after set comprehension");
        NodePtr node = makeNode(actionAstKind("setcomp", "SetComp"),
                                SourceRange{start, previous().range.end});
        addField(*node, "elt", first);
        addField(*node, "generators", std::move(generators));
        return node;
      }
      std::vector<NodePtr> elements{first};
      while (matchText(",") && !atText("}")) {
        NodePtr element = parseStarredExpression(/*allowNamedExpression=*/true);
        if (startsComprehensionClause()) {
          rejectUnparenthesizedComprehensionTarget(elements.front(), element);
          (void)parseComprehensionClauses();
          break;
        }
        elements.push_back(std::move(element));
      }
      if (!atText("}") && !at(TokenKind::Newline) && !at(TokenKind::Dedent) &&
          !at(TokenKind::End)) {
        error(current().range.start,
              "invalid syntax. Perhaps you forgot a comma?");
        skipToContainerEnd("}");
        consumeText("}", "expected '}' after set");
        return makeNode("Error", SourceRange{start, previous().range.end});
      }
      consumeText("}", "expected '}' after set");
      NodePtr node = makeNode(actionAstKind("set", "Set"),
                              SourceRange{start, previous().range.end});
      addField(*node, "elts", std::move(elements));
      return node;
    }

    keys.push_back(std::move(first));
    values.push_back(parseDictValue(previous().range.start));
    if (startsComprehensionClause()) {
      std::vector<NodePtr> generators = parseComprehensionClauses();
      consumeText("}", "expected '}' after dict comprehension");
      NodePtr node = makeNode(actionAstKind("dictcomp", "DictComp"),
                              SourceRange{start, previous().range.end});
      addField(*node, "key", keys.front());
      addField(*node, "value", values.front());
      addField(*node, "generators", std::move(generators));
      return node;
    }
    while (matchText(",") && !atText("}")) {
      if (matchText("**")) {
        SourceLocation unpackStart = previous().range.start;
        keys.push_back(NodePtr{});
        values.push_back(parseExpression());
        rejectDictUnpackAssignment(unpackStart);
        continue;
      }
      keys.push_back(parseExpression());
      SourceLocation colon =
          consumeText(":", "expected ':' in dict item").range.start;
      values.push_back(parseDictValue(colon));
    }
    consumeText("}", "expected '}' after dict");
    NodePtr node = makeNode(actionAstKind("dict", "Dict"),
                            SourceRange{start, previous().range.end});
    addField(*node, "keys", std::move(keys));
    addField(*node, "values", std::move(values));
    return node;
  }

  enum class TargetContext { Assign, AnnAssign, AugAssign, Delete };

  const Field *findField(const NodePtr &node, std::string_view name) const {
    if (!node)
      return nullptr;
    for (const Field &field : node->fields)
      if (field.name == name)
        return &field;
    return nullptr;
  }

  const std::vector<NodePtr> *nodeListField(const NodePtr &node,
                                            std::string_view name) const {
    const Field *field = findField(node, name);
    if (!field)
      return nullptr;
    return std::get_if<std::vector<NodePtr>>(&field->value);
  }

  const NodePtr *nodeField(const NodePtr &node, std::string_view name) const {
    const Field *field = findField(node, name);
    if (!field)
      return nullptr;
    return std::get_if<NodePtr>(&field->value);
  }

  std::string invalidTargetName(const NodePtr &node) const {
    if (!node)
      return "expression";
    if (node->kind == "Call")
      return "function call";
    if (node->kind == "Constant")
      return "literal";
    if (node->kind == "Dict")
      return "dict literal";
    if (node->kind == "Set")
      return "set display";
    if (node->kind == "ListComp")
      return "list comprehension";
    if (node->kind == "DictComp")
      return "dict comprehension";
    if (node->kind == "SetComp")
      return "set comprehension";
    if (node->kind == "GeneratorExp")
      return "generator expression";
    if (node->kind == "Await")
      return "await expression";
    if (node->kind == "Attribute")
      return "attribute";
    if (node->kind == "Subscript")
      return "subscript";
    if (node->kind == "JoinedStr")
      return "f-string expression";
    if (node->kind == "TemplateStr")
      return "template string expression";
    if (node->kind == "Lambda")
      return "lambda";
    if (node->kind == "Tuple")
      return "tuple";
    if (node->kind == "List")
      return "list";
    if (node->kind == "Starred")
      return "starred";
    return "expression";
  }

  void invalidTarget(const NodePtr &node, TargetContext context) {
    SourceLocation location = node ? node->range.start : current().range.start;
    switch (context) {
    case TargetContext::Assign:
      error(location, "cannot assign to " + invalidTargetName(node) +
                          " here. Maybe you meant '==' instead of '='?");
      return;
    case TargetContext::AnnAssign:
      if (node && (node->kind == "List" || node->kind == "Tuple")) {
        error(location,
              "only single target (not " +
                  std::string(node->kind == "List" ? "list" : "tuple") +
                  ") can be annotated");
        return;
      }
      error(location, "illegal target for annotation");
      return;
    case TargetContext::AugAssign:
      error(location,
            "'" + invalidTargetName(node) +
                "' is an illegal expression for augmented assignment");
      return;
    case TargetContext::Delete:
      error(location, "cannot delete " + invalidTargetName(node));
      return;
    }
  }

  bool validateTarget(const NodePtr &node, TargetContext context,
                      bool nested = false) {
    if (!node) {
      invalidTarget(node, context);
      return false;
    }

    if (node->kind == "Name" || node->kind == "Attribute" ||
        node->kind == "Subscript")
      return true;

    if (node->kind == "Starred") {
      if (context == TargetContext::Delete) {
        error(node->range.start, "cannot delete starred");
        return false;
      }
      if (context == TargetContext::AugAssign ||
          context == TargetContext::AnnAssign) {
        invalidTarget(node, context);
        return false;
      }
      const NodePtr *value = nodeField(node, "value");
      return value && validateTarget(*value, context, /*nested=*/true);
    }

    if (node->kind == "Tuple" || node->kind == "List") {
      if (context == TargetContext::AugAssign ||
          context == TargetContext::AnnAssign) {
        invalidTarget(node, context);
        return false;
      }
      const std::vector<NodePtr> *elements = nodeListField(node, "elts");
      bool valid = elements != nullptr;
      if (elements)
        for (const NodePtr &element : *elements)
          valid = validateTarget(element, context, /*nested=*/true) && valid;
      return valid;
    }

    invalidTarget(node, context);
    return false;
  }

  void setContext(const NodePtr &node, const std::string &context) {
    if (!node)
      return;
    if (node->kind != "Name" && node->kind != "Attribute" &&
        node->kind != "Subscript" && node->kind != "Tuple" &&
        node->kind != "List" && node->kind != "Starred")
      return;

    for (Field &field : node->fields) {
      if (field.name == "ctx") {
        field.value = exprContext(context);
        continue;
      }
      if ((node->kind == "Tuple" || node->kind == "List") &&
          field.name == "elts") {
        auto *elements = std::get_if<std::vector<NodePtr>>(&field.value);
        if (!elements)
          continue;
        for (const NodePtr &element : *elements)
          setContext(element, context);
        continue;
      }
      if (node->kind == "Starred" && field.name == "value") {
        auto *value = std::get_if<NodePtr>(&field.value);
        if (value)
          setContext(*value, context);
      }
    }
  }

  void setField(Node &node, const std::string &name, FieldValue value) {
    if (Field *field = parser::findField(node, name)) {
      field->value = std::move(value);
      return;
    }
    addField(node, name, std::move(value));
  }

  static int hexValue(char ch) {
    if (ch >= '0' && ch <= '9')
      return ch - '0';
    if (ch >= 'a' && ch <= 'f')
      return 10 + ch - 'a';
    if (ch >= 'A' && ch <= 'F')
      return 10 + ch - 'A';
    return -1;
  }

  static bool isOctalDigit(char ch) { return ch >= '0' && ch <= '7'; }

  static bool appendUtf8(std::string &out, std::uint32_t codepoint) {
    if (codepoint <= 0x7f) {
      out.push_back(static_cast<char>(codepoint));
      return true;
    }
    if (codepoint >= 0xd800 && codepoint <= 0xdfff)
      return false;
    if (codepoint <= 0x7ff) {
      out.push_back(static_cast<char>(0xc0 | (codepoint >> 6)));
      out.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
      return true;
    }
    if (codepoint <= 0xffff) {
      out.push_back(static_cast<char>(0xe0 | (codepoint >> 12)));
      out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
      out.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
      return true;
    }
    if (codepoint <= 0x10ffff) {
      out.push_back(static_cast<char>(0xf0 | (codepoint >> 18)));
      out.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3f)));
      out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
      out.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
      return true;
    }
    return false;
  }

  static bool appendHexEscape(std::string &out, const std::string &literal,
                              std::size_t &index, std::size_t contentEnd) {
    if (index + 2 >= contentEnd)
      return false;
    int high = hexValue(literal[index + 1]);
    int low = hexValue(literal[index + 2]);
    if (high < 0 || low < 0)
      return false;
    out.push_back(static_cast<char>((high << 4) | low));
    index += 2;
    return true;
  }

  static bool appendUnicodeEscape(std::string &out, const std::string &literal,
                                  std::size_t &index, std::size_t contentEnd,
                                  int digits) {
    if (index + static_cast<std::size_t>(digits) >= contentEnd)
      return false;
    std::uint32_t codepoint = 0;
    for (int digit = 1; digit <= digits; ++digit) {
      int value = hexValue(literal[index + static_cast<std::size_t>(digit)]);
      if (value < 0)
        return false;
      codepoint = (codepoint << 4) | static_cast<std::uint32_t>(value);
    }
    if (!appendUtf8(out, codepoint))
      return false;
    index += static_cast<std::size_t>(digits);
    return true;
  }

  static std::optional<std::string>
  parseNamedUnicodeEscape(const std::string &literal, std::size_t nameStart,
                          std::size_t contentEnd, std::size_t &closingBrace) {
    if (nameStart >= contentEnd || literal[nameStart] != '{')
      return std::nullopt;
    closingBrace = literal.find('}', nameStart + 1);
    if (closingBrace == std::string::npos || closingBrace >= contentEnd ||
        closingBrace == nameStart + 1)
      return std::nullopt;
    return cpythonUnicodeNameString(
        std::string_view(literal.data(), literal.size())
            .substr(nameStart + 1, closingBrace - nameStart - 1));
  }

  static bool appendNamedUnicodeEscape(std::string &out,
                                       const std::string &literal,
                                       std::size_t &index,
                                       std::size_t contentEnd) {
    std::size_t closingBrace = 0;
    std::optional<std::string> value =
        parseNamedUnicodeEscape(literal, index + 1, contentEnd, closingBrace);
    if (!value)
      return false;
    out += *value;
    index = closingBrace;
    return true;
  }

  static void appendUnknownEscape(std::string &out, char escaped) {
    out.push_back('\\');
    out.push_back(escaped);
  }

  struct StringContentRange {
    std::size_t start = 0;
    std::size_t end = 0;
  };

  StringContentRange stringContentRange(const std::string &literal) {
    const std::size_t firstQuote = literal.find_first_of("'\"");
    if (firstQuote == std::string::npos)
      return StringContentRange{0, literal.size()};

    const char quote = literal[firstQuote];
    const bool triple = firstQuote + 2 < literal.size() &&
                        literal[firstQuote + 1] == quote &&
                        literal[firstQuote + 2] == quote;
    const std::size_t contentStart = firstQuote + (triple ? 3 : 1);
    std::size_t contentEnd = literal.size();
    if (triple) {
      if (contentEnd >= 3)
        contentEnd -= 3;
    } else if (contentEnd >= 1) {
      contentEnd -= 1;
    }

    return StringContentRange{contentStart, contentEnd};
  }

  std::string stringContent(const std::string &literal) {
    StringContentRange range = stringContentRange(literal);
    return literal.substr(range.start, range.end - range.start);
  }

  SourceLocation advanceLocation(SourceLocation location, char ch) const {
    ++location.offset;
    if (ch == '\n') {
      ++location.line;
      location.column = 0;
    } else {
      ++location.column;
    }
    return location;
  }

  SourceLocation locationAt(SourceLocation start, std::string_view text,
                            std::size_t offset) const {
    SourceLocation location = start;
    std::size_t limit = std::min(offset, text.size());
    for (std::size_t i = 0; i < limit; ++i)
      location = advanceLocation(location, text[i]);
    return location;
  }

  SourceRange rangeAt(SourceLocation start, std::string_view text,
                      std::size_t begin, std::size_t end) const {
    return SourceRange{locationAt(start, text, begin),
                       locationAt(start, text, end)};
  }

  void translateRanges(const NodePtr &node, SourceLocation sourceStart,
                       std::string_view source) const {
    if (!node)
      return;
    node->range = rangeAt(sourceStart, source, node->range.start.offset,
                          node->range.end.offset);
    for (Field &field : node->fields) {
      if (auto *child = std::get_if<NodePtr>(&field.value)) {
        translateRanges(*child, sourceStart, source);
      } else if (auto *children =
                     std::get_if<std::vector<NodePtr>>(&field.value)) {
        for (const NodePtr &nested : *children)
          translateRanges(nested, sourceStart, source);
      }
    }
  }

  void appendDecodedEscape(std::string &decoded, const std::string &literal,
                           std::size_t &i, std::size_t contentEnd,
                           bool bytesLiteral) {
    const char escaped = literal[++i];
    switch (escaped) {
    case '\r':
      if (i + 1 < contentEnd && literal[i + 1] == '\n')
        ++i;
      break;
    case '\n':
      break;
    case 'a':
      decoded.push_back('\a');
      break;
    case 'b':
      decoded.push_back('\b');
      break;
    case 'f':
      decoded.push_back('\f');
      break;
    case 'n':
      decoded.push_back('\n');
      break;
    case 'r':
      decoded.push_back('\r');
      break;
    case 't':
      decoded.push_back('\t');
      break;
    case 'v':
      decoded.push_back('\v');
      break;
    case '\\':
      decoded.push_back('\\');
      break;
    case '\'':
      decoded.push_back('\'');
      break;
    case '"':
      decoded.push_back('"');
      break;
    case 'x':
      if (!appendHexEscape(decoded, literal, i, contentEnd))
        appendUnknownEscape(decoded, escaped);
      break;
    case 'u':
      if (bytesLiteral ||
          !appendUnicodeEscape(decoded, literal, i, contentEnd, 4))
        appendUnknownEscape(decoded, escaped);
      break;
    case 'U':
      if (bytesLiteral ||
          !appendUnicodeEscape(decoded, literal, i, contentEnd, 8))
        appendUnknownEscape(decoded, escaped);
      break;
    case 'N':
      if (bytesLiteral ||
          !appendNamedUnicodeEscape(decoded, literal, i, contentEnd))
        appendUnknownEscape(decoded, escaped);
      break;
    default:
      if (isOctalDigit(escaped)) {
        int value = escaped - '0';
        int consumed = 0;
        while (consumed < 2 && i + 1 < contentEnd &&
               isOctalDigit(literal[i + 1])) {
          value = (value << 3) | (literal[i + 1] - '0');
          ++i;
          ++consumed;
        }
        decoded.push_back(static_cast<char>(value & 0xff));
        break;
      }
      appendUnknownEscape(decoded, escaped);
      break;
    }
  }

  std::string decodeStringLiteral(const std::string &literal) {
    StringContentRange range = stringContentRange(literal);
    if (literalPrefixContains(literal, 'r'))
      return literal.substr(range.start, range.end - range.start);

    const bool bytesLiteral = literalPrefixContains(literal, 'b');
    std::string decoded;
    for (std::size_t i = range.start; i < range.end; ++i) {
      const char ch = literal[i];
      if (ch != '\\' || i + 1 >= range.end) {
        decoded.push_back(ch);
        continue;
      }
      appendDecodedEscape(decoded, literal, i, range.end, bytesLiteral);
    }
    return decoded;
  }

  bool isFStringLiteral(const std::string &literal) {
    return literalPrefixContains(literal, 'f');
  }

  bool isTStringLiteral(const std::string &literal) {
    return literalPrefixContains(literal, 't');
  }

  bool isBytesStringLiteral(const std::string &literal) {
    return literalPrefixContains(literal, 'b');
  }

  bool isAsciiBytesLiteralContent(const std::string &literal) {
    StringContentRange range = stringContentRange(literal);
    for (std::size_t i = range.start; i < range.end; ++i) {
      unsigned char ch = static_cast<unsigned char>(literal[i]);
      if (ch == '\\' && i + 1 < range.end) {
        ++i;
        continue;
      }
      if (ch >= 0x80)
        return false;
    }
    return true;
  }

  bool hasHexDigits(const std::string &literal, std::size_t first,
                    std::size_t contentEnd, int count) {
    if (first + static_cast<std::size_t>(count) > contentEnd)
      return false;
    for (int index = 0; index < count; ++index)
      if (hexValue(literal[first + static_cast<std::size_t>(index)]) < 0)
        return false;
    return true;
  }

  bool validUnicodeScalarEscape(const std::string &literal, std::size_t first,
                                std::size_t contentEnd, int digits) {
    if (!hasHexDigits(literal, first, contentEnd, digits))
      return false;
    std::uint32_t codepoint = 0;
    for (int index = 0; index < digits; ++index)
      codepoint = (codepoint << 4) |
                  static_cast<std::uint32_t>(hexValue(
                      literal[first + static_cast<std::size_t>(index)]));
    std::string ignored;
    return appendUtf8(ignored, codepoint);
  }

  bool validateStringEscapes(const Token &literal) {
    if (literalPrefixContains(literal.text, 'r'))
      return true;

    const bool bytesLiteral = isBytesStringLiteral(literal.text);
    StringContentRange range = stringContentRange(literal.text);
    for (std::size_t i = range.start; i < range.end; ++i) {
      if (literal.text[i] != '\\' || i + 1 >= range.end)
        continue;
      const char escaped = literal.text[++i];
      if (escaped == 'x') {
        if (!hasHexDigits(literal.text, i + 1, range.end, 2)) {
          error(literal.range.start, "invalid \\x escape");
          return false;
        }
        i += 2;
        continue;
      }
      if (!bytesLiteral && escaped == 'u') {
        if (!validUnicodeScalarEscape(literal.text, i + 1, range.end, 4)) {
          error(literal.range.start, "invalid \\u escape");
          return false;
        }
        i += 4;
        continue;
      }
      if (!bytesLiteral && escaped == 'U') {
        if (!validUnicodeScalarEscape(literal.text, i + 1, range.end, 8)) {
          error(literal.range.start, "invalid \\U escape");
          return false;
        }
        i += 8;
        continue;
      }
      if (!bytesLiteral && escaped == 'N') {
        std::size_t closingBrace = 0;
        if (!parseNamedUnicodeEscape(literal.text, i + 1, range.end,
                                     closingBrace)) {
          error(literal.range.start, "unknown Unicode character name");
          return false;
        }
        i = closingBrace;
      }
    }
    return true;
  }

  std::optional<std::string> unicodeLiteralKind(const std::string &literal) {
    if (literalPrefixContains(literal, 'u'))
      return std::string("u");
    return std::nullopt;
  }

  NodePtr makeStringConstant(SourceRange range, std::string value,
                             std::optional<std::string> kind = std::nullopt) {
    NodePtr node =
        makeNode(actionHelperAstKind("string", "_PyPegen_constant_from_string",
                                     "Constant", "Constant"),
                 range);
    addField(*node, "value", std::move(value));
    addField(*node, "kind", kind ? FieldValue{*kind} : FieldValue{});
    return node;
  }

  NodePtr makeBytesConstant(SourceRange range, std::string value) {
    NodePtr node =
        makeNode(actionHelperAstKind("string", "_PyPegen_constant_from_string",
                                     "Constant", "Constant"),
                 range);
    std::vector<std::uint8_t> bytes;
    bytes.reserve(value.size());
    for (unsigned char ch : value)
      bytes.push_back(static_cast<std::uint8_t>(ch));
    addField(*node, "value", std::move(bytes));
    addField(*node, "kind", FieldValue{});
    return node;
  }

  NodePtr parseStringLiteralSequence(SourceLocation start) {
    std::vector<Token> literals;
    bool hasFString = false;
    bool hasTemplateString = false;
    bool hasBytesString = false;
    bool hasPlainString = false;
    do {
      Token token = current();
      ++index;
      hasFString = hasFString || isFStringLiteral(token.text);
      hasTemplateString = hasTemplateString || isTStringLiteral(token.text);
      hasBytesString = hasBytesString || isBytesStringLiteral(token.text);
      hasPlainString = hasPlainString || (!isBytesStringLiteral(token.text) &&
                                          !isFStringLiteral(token.text) &&
                                          !isTStringLiteral(token.text));
      literals.push_back(std::move(token));
    } while (at(TokenKind::String));

    SourceRange range{start, literals.back().range.end};
    if (hasTemplateString && (hasBytesString || hasFString || hasPlainString)) {
      error(start,
            "cannot mix t-string literals with string or bytes literals");
      return makeNode("Error", range);
    }
    if (hasBytesString && (hasPlainString || hasFString)) {
      error(start, "cannot mix bytes and nonbytes literals");
      return makeNode("Error", range);
    }
    for (const Token &literal : literals)
      if (!validateStringEscapes(literal))
        return makeNode("Error", range);
    if (hasBytesString) {
      std::string value;
      for (const Token &literal : literals) {
        if (!isAsciiBytesLiteralContent(literal.text)) {
          error(literal.range.start,
                "bytes can only contain ASCII literal characters");
          return makeNode("Error", range);
        }
        value += decodeStringLiteral(literal.text);
      }
      return makeBytesConstant(range, std::move(value));
    }
    if (hasTemplateString) {
      NodePtr node =
          makeNode(actionHelperAstKind("tstring", "_PyPegen_template_str",
                                       "TemplateStr", "TemplateStr"),
                   range);
      std::vector<NodePtr> values;
      for (const Token &literal : literals) {
        StringContentRange contentRange = stringContentRange(literal.text);
        std::string content = literal.text.substr(
            contentRange.start, contentRange.end - contentRange.start);
        SourceLocation contentStart =
            locationAt(literal.range.start, literal.text, contentRange.start);
        std::vector<NodePtr> parts =
            parseTStringValues(literal.range, content, contentStart,
                               literalPrefixContains(literal.text, 'r'));
        appendStringNodes(values, std::move(parts));
      }
      addField(*node, "values", std::move(values));
      return node;
    }

    if (!hasFString) {
      std::string value;
      std::optional<std::string> kind =
          unicodeLiteralKind(literals.front().text);
      for (const Token &literal : literals)
        value += decodeStringLiteral(literal.text);
      return makeStringConstant(range, std::move(value), std::move(kind));
    }

    NodePtr node =
        makeNode(actionHelperAstKind("fstring", "_PyPegen_joined_str",
                                     "JoinedStr", "JoinedStr"),
                 range);
    std::vector<NodePtr> values;
    std::string plain;
    std::optional<SourceRange> plainRange;
    auto flushPlain = [&]() {
      if (plain.empty())
        return;
      appendStringNode(values, makeStringConstant(plainRange.value_or(range),
                                                  std::move(plain)));
      plain.clear();
      plainRange.reset();
    };
    for (const Token &literal : literals) {
      std::string decoded = decodeStringLiteral(literal.text);
      if (!isFStringLiteral(literal.text)) {
        if (!plainRange)
          plainRange = literal.range;
        else
          plainRange->end = literal.range.end;
        plain += std::move(decoded);
        continue;
      }
      flushPlain();
      StringContentRange contentRange = stringContentRange(literal.text);
      std::string content = literal.text.substr(
          contentRange.start, contentRange.end - contentRange.start);
      SourceLocation contentStart =
          locationAt(literal.range.start, literal.text, contentRange.start);
      std::vector<NodePtr> fValues =
          parseFStringValues(literal.range, content, contentStart,
                             literalPrefixContains(literal.text, 'r'));
      appendStringNodes(values, std::move(fValues));
    }
    flushPlain();
    addField(*node, "values", std::move(values));
    return node;
  }

  NodePtr parseInlineExpression(std::string_view source) {
    Diagnostics nestedDiagnostics;
    LexResult lexed =
        lexSource(source, nestedDiagnostics, /*typeComments=*/false);
    std::vector<Token> nestedTokens = std::move(lexed.tokens);
    diagnostics.insert(diagnostics.end(), nestedDiagnostics.begin(),
                       nestedDiagnostics.end());
    if (!nestedDiagnostics.empty())
      return makeNode("Error");

    ParserImpl parser(std::move(nestedTokens), diagnostics);
    NodePtr expression = parser.parseExpressionMode();
    for (const Field &field : expression->fields) {
      if (field.name != "body")
        continue;
      if (const auto *body = std::get_if<NodePtr>(&field.value))
        return *body;
    }
    return makeNode("Error");
  }

  NodePtr parseInlineAnnotatedRhs(
      std::string_view source, SourceRange outerRange,
      std::string_view stringKind,
      std::optional<SourceLocation> sourceStart = std::nullopt) {
    Diagnostics nestedDiagnostics;
    LexResult lexed =
        lexSource(source, nestedDiagnostics, /*typeComments=*/false);
    std::vector<Token> nestedTokens = std::move(lexed.tokens);
    if (!nestedDiagnostics.empty()) {
      diagnostics.insert(diagnostics.end(), nestedDiagnostics.begin(),
                         nestedDiagnostics.end());
      return makeNode("Error");
    }

    ParserImpl parser(std::move(nestedTokens), nestedDiagnostics);
    NodePtr expression = parser.parseAnnotatedRhsMode();
    if (!nestedDiagnostics.empty()) {
      bool hasTrailingExpressionToken = std::any_of(
          nestedDiagnostics.begin(), nestedDiagnostics.end(),
          [](const Diagnostic &diagnostic) {
            return diagnostic.message == "unexpected token after expression";
          });
      if (hasTrailingExpressionToken) {
        error(outerRange.start, std::string(stringKind) +
                                    ": expecting '=', or '!', or ':', or '}'");
      } else {
        diagnostics.insert(diagnostics.end(), nestedDiagnostics.begin(),
                           nestedDiagnostics.end());
      }
      return makeNode("Error", outerRange);
    }
    for (const Field &field : expression->fields) {
      if (field.name != "body")
        continue;
      if (const auto *body = std::get_if<NodePtr>(&field.value)) {
        if (sourceStart)
          translateRanges(*body, *sourceStart, source);
        return *body;
      }
    }
    return makeNode("Error");
  }

  void appendStringChunk(std::vector<NodePtr> &values, SourceRange range,
                         std::string &chunk) {
    if (chunk.empty())
      return;
    appendStringNode(values, makeStringConstant(range, std::move(chunk)));
    chunk.clear();
  }

  std::string *mutableStringConstantValue(const NodePtr &node) {
    if (!node || node->kind != "Constant")
      return nullptr;
    for (Field &field : node->fields)
      if (field.name == "value")
        return std::get_if<std::string>(&field.value);
    return nullptr;
  }

  void appendStringNode(std::vector<NodePtr> &values, NodePtr node) {
    if (!node)
      return;
    std::string *text = mutableStringConstantValue(node);
    if (text && text->empty())
      return;
    if (text && !values.empty()) {
      if (std::string *previous = mutableStringConstantValue(values.back())) {
        previous->append(*text);
        values.back()->range.end = node->range.end;
        return;
      }
    }
    values.push_back(std::move(node));
  }

  void appendStringNodes(std::vector<NodePtr> &values,
                         std::vector<NodePtr> nodes) {
    for (NodePtr &node : nodes)
      appendStringNode(values, std::move(node));
  }

  std::size_t skipQuotedText(std::string_view text, std::size_t quoteIndex) {
    char quote = text[quoteIndex];
    bool triple = quoteIndex + 2 < text.size() &&
                  text[quoteIndex + 1] == quote &&
                  text[quoteIndex + 2] == quote;
    std::size_t i = quoteIndex + (triple ? 3 : 1);
    while (i < text.size()) {
      char ch = text[i++];
      if (ch == '\\') {
        if (i < text.size())
          ++i;
        continue;
      }
      if (ch != quote)
        continue;
      if (!triple)
        return i;
      if (i + 1 < text.size() && text[i] == quote && text[i + 1] == quote)
        return i + 2;
    }
    return text.size();
  }

  std::size_t skipFStringComment(std::string_view text, std::size_t index,
                                 std::size_t limit) {
    while (index < limit && text[index] != '\n' && text[index] != '\r')
      ++index;
    if (index < limit && text[index] == '\r')
      ++index;
    if (index < limit && text[index] == '\n')
      ++index;
    return index;
  }

  std::size_t findFStringFieldEnd(std::string_view text, std::size_t start) {
    int depth = 0;
    for (std::size_t i = start; i < text.size();) {
      char ch = text[i];
      if (ch == '#') {
        i = skipFStringComment(text, i, text.size());
        continue;
      }
      if (ch == '\'' || ch == '"') {
        i = skipQuotedText(text, i);
        continue;
      }
      if (ch == '{') {
        ++depth;
        ++i;
        continue;
      }
      if (ch == '}') {
        if (depth == 0)
          return i;
        --depth;
        ++i;
        continue;
      }
      ++i;
    }
    return std::string_view::npos;
  }

  std::size_t findFStringFieldDelimiter(std::string_view text, char delimiter) {
    int depth = 0;
    for (std::size_t i = 0; i < text.size();) {
      char ch = text[i];
      if (ch == '#') {
        i = skipFStringComment(text, i, text.size());
        continue;
      }
      if (ch == '\'' || ch == '"') {
        i = skipQuotedText(text, i);
        continue;
      }
      if (ch == '(' || ch == '[' || ch == '{') {
        ++depth;
        ++i;
        continue;
      }
      if (ch == ')' || ch == ']' || ch == '}') {
        --depth;
        ++i;
        continue;
      }
      if (depth == 0 && ch == delimiter) {
        if (delimiter == '!' && i + 1 < text.size() && text[i + 1] == '=') {
          i += 2;
          continue;
        }
        return i;
      }
      ++i;
    }
    return std::string_view::npos;
  }

  std::size_t findFStringDebugEqual(std::string_view text, std::size_t limit) {
    int depth = 0;
    for (std::size_t i = 0; i < limit;) {
      char ch = text[i];
      if (ch == '#') {
        i = skipFStringComment(text, i, limit);
        continue;
      }
      if (ch == '\'' || ch == '"') {
        i = skipQuotedText(text, i);
        continue;
      }
      if (ch == '(' || ch == '[' || ch == '{') {
        ++depth;
        ++i;
        continue;
      }
      if (ch == ')' || ch == ']' || ch == '}') {
        --depth;
        ++i;
        continue;
      }
      if (depth == 0 && ch == '=') {
        char previous = i > 0 ? text[i - 1] : '\0';
        char next = i + 1 < text.size() ? text[i + 1] : '\0';
        if (previous == '!' || previous == '<' || previous == '>' ||
            previous == '=' || previous == ':' || next == '=')
          ++i;
        else
          return i;
        continue;
      }
      ++i;
    }
    return std::string_view::npos;
  }

  struct FStringField {
    std::string expression;
    std::string interpolationText;
    std::int64_t conversion = -1;
    std::optional<std::string> debugText;
    std::optional<std::string> formatSpec;
    std::size_t expressionOffset = 0;
    std::size_t formatOffset = std::string_view::npos;
    std::size_t formatDelimiterOffset = std::string_view::npos;
    bool invalid = false;
  };

  std::string stripInterpolationSource(std::string_view text) {
    std::size_t end = text.size();
    while (end > 0) {
      unsigned char ch = static_cast<unsigned char>(text[end - 1]);
      if (!std::isspace(ch) && text[end - 1] != '=')
        break;
      --end;
    }
    return std::string(text.substr(0, end));
  }

  std::string trimInterpolationExpression(std::string_view text,
                                          std::size_t &offset) {
    const std::size_t first = text.find_first_not_of(" \t\r\n");
    if (first == std::string_view::npos) {
      offset = 0;
      return std::string();
    }
    const std::size_t last = text.find_last_not_of(" \t\r\n");
    offset = first;
    return std::string(text.substr(first, last - first + 1));
  }

  bool startsWithLambdaKeyword(std::string_view expression) {
    constexpr std::string_view lambda = "lambda";
    if (expression.size() < lambda.size() ||
        expression.substr(0, lambda.size()) != lambda)
      return false;
    return expression.size() == lambda.size() ||
           std::isspace(static_cast<unsigned char>(expression[lambda.size()]));
  }

  FStringField parseFStringField(SourceRange range, std::string field,
                                 std::string_view stringKind) {
    FStringField parts;
    std::string_view view(field);
    std::size_t conversion = findFStringFieldDelimiter(view, '!');
    std::size_t format = findFStringFieldDelimiter(view, ':');
    std::size_t expressionEnd = view.size();
    if (conversion != std::string_view::npos)
      expressionEnd = std::min(expressionEnd, conversion);
    if (format != std::string_view::npos)
      expressionEnd = std::min(expressionEnd, format);
    parts.interpolationText =
        stripInterpolationSource(view.substr(0, expressionEnd));

    std::size_t debugEqual = findFStringDebugEqual(view, expressionEnd);
    if (debugEqual != std::string_view::npos) {
      parts.debugText = std::string(view.substr(0, expressionEnd));
      expressionEnd = debugEqual;
    }

    std::size_t expressionOffset = 0;
    parts.expression = trimInterpolationExpression(
        view.substr(0, expressionEnd), expressionOffset);
    parts.expressionOffset = expressionOffset;
    if (parts.expression.empty()) {
      parts.invalid = true;
      if (debugEqual != std::string_view::npos) {
        error(range.start, std::string(stringKind) +
                               ": valid expression required before '='");
      } else if (conversion == 0) {
        error(range.start, std::string(stringKind) +
                               ": valid expression required before '!'");
      } else if (format == 0) {
        error(range.start, std::string(stringKind) +
                               ": valid expression required before ':'");
      } else {
        error(range.start, std::string(stringKind) +
                               ": valid expression required before '}'");
      }
      return parts;
    }
    if (debugEqual != std::string_view::npos && debugEqual + 1 < view.size()) {
      std::size_t afterDebugEqual = debugEqual + 1;
      while (afterDebugEqual < view.size() &&
             std::isspace(static_cast<unsigned char>(view[afterDebugEqual])))
        ++afterDebugEqual;
      char marker = afterDebugEqual < view.size() ? view[afterDebugEqual] : '}';
      if (marker != '!' && marker != ':' && marker != '}') {
        parts.invalid = true;
        error(range.start,
              std::string(stringKind) + ": expecting '!', or ':', or '}'");
        return parts;
      }
    }
    if (format != std::string_view::npos &&
        startsWithLambdaKeyword(parts.expression)) {
      parts.invalid = true;
      error(range.start, std::string(stringKind) +
                             ": lambda expressions are not allowed without "
                             "parentheses");
      return parts;
    }

    if (conversion != std::string_view::npos &&
        (format == std::string_view::npos || conversion < format)) {
      if (conversion + 1 >= view.size()) {
        parts.invalid = true;
        error(range.start,
              std::string(stringKind) + ": missing conversion character");
        return parts;
      }
      char marker = view[conversion + 1];
      if (marker == ':') {
        parts.invalid = true;
        error(range.start,
              std::string(stringKind) + ": missing conversion character");
        return parts;
      }
      if (marker != 'r' && marker != 's' && marker != 'a') {
        parts.invalid = true;
        if (stringKind == "f-string") {
          error(range.start, "f-string: invalid conversion character '" +
                                 std::string(1, marker) +
                                 "': expected 's', 'r', or 'a'");
        } else {
          error(range.start,
                std::string(stringKind) + ": invalid conversion character");
        }
        return parts;
      }
      parts.conversion = static_cast<std::int64_t>(marker);
      std::size_t afterConversion = conversion + 2;
      if (afterConversion < view.size()) {
        if (view[afterConversion] != ':') {
          parts.invalid = true;
          error(range.start,
                std::string(stringKind) + ": expecting ':' or '}'");
          return parts;
        }
        parts.formatDelimiterOffset = afterConversion;
        parts.formatOffset = afterConversion + 1;
        parts.formatSpec = std::string(view.substr(parts.formatOffset));
      }
      return parts;
    }

    if (format != std::string_view::npos) {
      parts.formatDelimiterOffset = format;
      parts.formatOffset = format + 1;
      parts.formatSpec = std::string(view.substr(parts.formatOffset));
    }
    if (parts.debugText && parts.conversion == -1 && !parts.formatSpec)
      parts.conversion = static_cast<std::int64_t>('r');
    return parts;
  }

  bool appendFStringEscapedChunk(std::string &chunk, const std::string &content,
                                 std::size_t &i, bool rawString) {
    if (rawString || content[i] != '\\' || i + 1 >= content.size())
      return false;
    if (content[i + 1] == '{' || content[i + 1] == '}') {
      chunk.push_back('\\');
      return true;
    }
    appendDecodedEscape(chunk, content, i, content.size(),
                        /*bytesLiteral=*/false);
    return true;
  }

  std::vector<NodePtr> parseFStringValues(
      SourceRange range, const std::string &content,
      SourceLocation contentStart, bool rawString = false,
      std::string_view replacementRule = "fstring_replacement_field",
      std::string_view fullFormatSpecRule = "fstring_full_format_spec") {
    std::vector<NodePtr> values;
    std::string chunk;
    std::optional<std::size_t> chunkStart;
    auto pushChunkChar = [&](char value, std::size_t index) {
      if (chunk.empty())
        chunkStart = index;
      chunk.push_back(value);
    };
    auto flushChunk = [&](std::size_t end) {
      if (chunk.empty())
        return;
      appendStringNode(values, makeStringConstant(rangeAt(contentStart, content,
                                                          *chunkStart, end),
                                                  std::move(chunk)));
      chunk.clear();
      chunkStart.reset();
    };
    for (std::size_t i = 0; i < content.size(); ++i) {
      char ch = content[i];
      if (!rawString && ch == '\\' && i + 1 < content.size()) {
        if (chunk.empty())
          chunkStart = i;
      }
      if (appendFStringEscapedChunk(chunk, content, i, rawString))
        continue;
      if (ch == '{' && i + 1 < content.size() && content[i + 1] == '{') {
        pushChunkChar('{', i);
        ++i;
        continue;
      }
      if (ch == '}' && i + 1 < content.size() && content[i + 1] == '}') {
        pushChunkChar('}', i);
        ++i;
        continue;
      }
      if (ch == '}') {
        error(range.start, "single '}' is not allowed in f-string");
        break;
      }
      if (ch != '{') {
        pushChunkChar(ch, i);
        continue;
      }

      flushChunk(i);
      std::size_t end = findFStringFieldEnd(content, i + 1);
      if (end == std::string::npos) {
        error(range.start, "unterminated f-string replacement field");
        break;
      }
      SourceRange formattedRange = rangeAt(contentStart, content, i, end + 1);
      FStringField field = parseFStringField(
          formattedRange, content.substr(i + 1, end - i - 1), "f-string");
      if (field.debugText)
        appendStringNode(
            values, makeStringConstant(rangeAt(contentStart, content, i + 1,
                                               i + 1 + field.debugText->size()),
                                       *field.debugText));
      NodePtr formatted = makeNode(
          actionHelperAstKind(replacementRule, "_PyPegen_formatted_value",
                              "FormattedValue", "FormattedValue"),
          formattedRange);
      SourceLocation expressionStart =
          locationAt(contentStart, content, i + 1 + field.expressionOffset);
      addField(*formatted, "value",
               field.invalid
                   ? makeNode("Error", formattedRange)
                   : parseInlineAnnotatedRhs(field.expression, formattedRange,
                                             "f-string", expressionStart));
      addField(*formatted, "conversion", field.conversion);
      NodePtr formatSpecNode;
      if (field.formatSpec) {
        SourceRange formatSpecRange = rangeAt(
            contentStart, content, i + 1 + field.formatDelimiterOffset, end);
        SourceLocation formatStart =
            locationAt(contentStart, content, i + 1 + field.formatOffset);
        formatSpecNode =
            makeNode(actionHelperAstKind(fullFormatSpecRule,
                                         "_PyPegen_setup_full_format_spec",
                                         "JoinedStr", "JoinedStr"),
                     formatSpecRange);
        addField(*formatSpecNode, "values",
                 parseFStringValues(formatSpecRange, *field.formatSpec,
                                    formatStart, rawString, replacementRule,
                                    fullFormatSpecRule));
      }
      addField(*formatted, "format_spec", formatSpecNode);
      values.push_back(std::move(formatted));
      i = end;
    }
    flushChunk(content.size());
    return values;
  }

  std::vector<NodePtr> parseTStringValues(SourceRange range,
                                          const std::string &content,
                                          SourceLocation contentStart,
                                          bool rawString = false) {
    std::vector<NodePtr> values;
    std::string chunk;
    std::optional<std::size_t> chunkStart;
    auto pushChunkChar = [&](char value, std::size_t index) {
      if (chunk.empty())
        chunkStart = index;
      chunk.push_back(value);
    };
    auto flushChunk = [&](std::size_t end) {
      if (chunk.empty())
        return;
      appendStringNode(values, makeStringConstant(rangeAt(contentStart, content,
                                                          *chunkStart, end),
                                                  std::move(chunk)));
      chunk.clear();
      chunkStart.reset();
    };
    for (std::size_t i = 0; i < content.size(); ++i) {
      char ch = content[i];
      if (!rawString && ch == '\\' && i + 1 < content.size()) {
        if (chunk.empty())
          chunkStart = i;
      }
      if (appendFStringEscapedChunk(chunk, content, i, rawString))
        continue;
      if (ch == '{' && i + 1 < content.size() && content[i + 1] == '{') {
        pushChunkChar('{', i);
        ++i;
        continue;
      }
      if (ch == '}' && i + 1 < content.size() && content[i + 1] == '}') {
        pushChunkChar('}', i);
        ++i;
        continue;
      }
      if (ch == '}') {
        error(range.start, "single '}' is not allowed in t-string");
        break;
      }
      if (ch != '{') {
        pushChunkChar(ch, i);
        continue;
      }

      flushChunk(i);
      std::size_t end = findFStringFieldEnd(content, i + 1);
      if (end == std::string::npos) {
        error(range.start, "unterminated t-string replacement field");
        break;
      }

      SourceRange interpolationRange =
          rangeAt(contentStart, content, i, end + 1);
      FStringField field = parseFStringField(
          interpolationRange, content.substr(i + 1, end - i - 1), "t-string");
      if (field.debugText)
        appendStringNode(
            values, makeStringConstant(rangeAt(contentStart, content, i + 1,
                                               i + 1 + field.debugText->size()),
                                       *field.debugText));

      NodePtr interpolation =
          makeNode(actionHelperAstKind("tstring_replacement_field",
                                       "_PyPegen_interpolation",
                                       "Interpolation", "Interpolation"),
                   interpolationRange);
      SourceLocation expressionStart =
          locationAt(contentStart, content, i + 1 + field.expressionOffset);
      addField(*interpolation, "value",
               field.invalid ? makeNode("Error", interpolationRange)
                             : parseInlineAnnotatedRhs(
                                   field.expression, interpolationRange,
                                   "t-string", expressionStart));
      addField(*interpolation, "str", field.interpolationText);
      addField(*interpolation, "conversion", field.conversion);
      NodePtr formatSpecNode;
      if (field.formatSpec) {
        SourceRange formatSpecRange = rangeAt(
            contentStart, content, i + 1 + field.formatDelimiterOffset, end);
        SourceLocation formatStart =
            locationAt(contentStart, content, i + 1 + field.formatOffset);
        formatSpecNode =
            makeNode(actionHelperAstKind("tstring_full_format_spec",
                                         "_PyPegen_setup_full_format_spec",
                                         "JoinedStr", "JoinedStr"),
                     formatSpecRange);
        addField(*formatSpecNode, "values",
                 parseFStringValues(formatSpecRange, *field.formatSpec,
                                    formatStart, rawString,
                                    "tstring_format_spec_replacement_field",
                                    "tstring_full_format_spec"));
      }
      addField(*interpolation, "format_spec", formatSpecNode);
      values.push_back(std::move(interpolation));
      i = end;
    }
    flushChunk(content.size());
    return values;
  }

  NodePtr parseJoinedStringLiteral(SourceLocation start, const Token &token) {
    NodePtr node =
        makeNode(actionHelperAstKind("fstring", "_PyPegen_joined_str",
                                     "JoinedStr", "JoinedStr"),
                 SourceRange{start, token.range.end});
    StringContentRange contentRange = stringContentRange(token.text);
    std::string content = token.text.substr(
        contentRange.start, contentRange.end - contentRange.start);
    SourceLocation contentStart =
        locationAt(token.range.start, token.text, contentRange.start);
    addField(*node, "values",
             parseFStringValues(node->range, content, contentStart,
                                literalPrefixContains(token.text, 'r')));
    return node;
  }

  NodePtr parseTemplateStringLiteral(SourceLocation start, const Token &token) {
    NodePtr node =
        makeNode(actionHelperAstKind("tstring", "_PyPegen_template_str",
                                     "TemplateStr", "TemplateStr"),
                 SourceRange{start, token.range.end});
    StringContentRange contentRange = stringContentRange(token.text);
    std::string content = token.text.substr(
        contentRange.start, contentRange.end - contentRange.start);
    SourceLocation contentStart =
        locationAt(token.range.start, token.text, contentRange.start);
    addField(*node, "values",
             parseTStringValues(node->range, content, contentStart,
                                literalPrefixContains(token.text, 'r')));
    return node;
  }
};

} // namespace

ParseResult parse(std::string_view source, std::string filename,
                  ParseOptions options) {
  (void)filename;
  ParseResult result;
  if (!lython_cpython_generated_parser_is_linked()) {
    result.diagnostics.push_back(
        Diagnostic{Severity::Error, SourceLocation{},
                   "patched CPython generated parser backend is not linked"});
    return result;
  }
  DecodedSource decoded = decodeSource(source, result.diagnostics);
  if (!result.diagnostics.empty())
    return result;
  LexResult lexed =
      lexSource(decoded.view(source), result.diagnostics, options.typeComments,
                options.mode == ParseMode::Interactive);
  std::vector<Token> tokens = std::move(lexed.tokens);
  if (!result.diagnostics.empty())
    return result;
  if (!validateWithCpythonGeneratedParser(
          tokens, options.mode, lexed.typeIgnoreCount, result.diagnostics))
    return result;
  ParserImpl parser(std::move(tokens), result.diagnostics,
                    std::move(lexed.typeIgnores),
                    std::move(lexed.typeComments));
  if (options.mode == ParseMode::Interactive)
    result.tree = parser.parseInteractiveMode();
  else if (options.mode == ParseMode::Expression)
    result.tree = parser.parseExpressionMode();
  else if (options.mode == ParseMode::FunctionType)
    result.tree = parser.parseFunctionTypeMode();
  else
    result.tree = parser.parseModule();
  if (!result.diagnostics.empty())
    result.tree.reset();
  return result;
}

} // namespace lython::parser
