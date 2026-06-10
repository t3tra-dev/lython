#include "lython/parser/Grammar.h"

#include <algorithm>
#include <cctype>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <utility>

namespace lython::parser {
namespace {

bool startsWith(std::string_view text, std::string_view prefix) {
  return text.size() >= prefix.size() &&
         text.compare(0, prefix.size(), prefix) == 0;
}

std::string trim(std::string text) {
  const std::size_t first = text.find_first_not_of(" \t\r\n");
  if (first == std::string::npos)
    return std::string();
  const std::size_t last = text.find_last_not_of(" \t\r\n");
  return text.substr(first, last - first + 1);
}

bool isIdentStart(char ch) {
  unsigned char typed = static_cast<unsigned char>(ch);
  return std::isalpha(typed) || ch == '_';
}

bool isIdentContinue(char ch) {
  unsigned char typed = static_cast<unsigned char>(ch);
  return std::isalnum(typed) || ch == '_';
}

bool isTokenName(std::string_view name) {
  return std::all_of(name.begin(), name.end(), [](char ch) {
    unsigned char typed = static_cast<unsigned char>(ch);
    return std::isupper(typed) || std::isdigit(typed) || ch == '_';
  });
}

std::size_t matchingBracket(std::string_view text, std::size_t open,
                            char close) {
  int depth = 0;
  char openChar = text[open];
  for (std::size_t cursor = open; cursor < text.size(); ++cursor) {
    if (text[cursor] == openChar) {
      ++depth;
      continue;
    }
    if (text[cursor] != close)
      continue;
    --depth;
    if (depth == 0)
      return cursor;
  }
  return std::string_view::npos;
}

struct RuleHeader {
  std::string name;
  std::string returnType;
  bool memo = false;
  std::string body;
};

bool parseRuleHeader(std::string_view line, RuleHeader &header) {
  if (line.empty() || std::isspace(static_cast<unsigned char>(line.front())) ||
      !isIdentStart(line.front()))
    return false;

  std::size_t cursor = 1;
  while (cursor < line.size() && isIdentContinue(line[cursor]))
    ++cursor;
  header.name = std::string(line.substr(0, cursor));

  while (cursor < line.size() &&
         std::isspace(static_cast<unsigned char>(line[cursor])))
    ++cursor;

  if (cursor < line.size() && line[cursor] == '[') {
    std::size_t close = matchingBracket(line, cursor, ']');
    if (close == std::string_view::npos)
      return false;
    header.returnType =
        trim(std::string(line.substr(cursor + 1, close - cursor - 1)));
    cursor = close + 1;
  }

  while (cursor < line.size() &&
         std::isspace(static_cast<unsigned char>(line[cursor])))
    ++cursor;

  if (cursor < line.size() && line[cursor] == '(') {
    std::size_t close = matchingBracket(line, cursor, ')');
    if (close == std::string_view::npos)
      return false;
    std::string attrs =
        trim(std::string(line.substr(cursor + 1, close - cursor - 1)));
    header.memo = attrs.find("memo") != std::string::npos;
    cursor = close + 1;
  }

  while (cursor < line.size() &&
         std::isspace(static_cast<unsigned char>(line[cursor])))
    ++cursor;

  if (cursor >= line.size() || line[cursor] != ':')
    return false;
  header.body = std::string(line.substr(cursor + 1));
  return true;
}

std::string stripActionsAndComments(std::string_view body) {
  std::string output;
  output.reserve(body.size());
  for (std::size_t i = 0; i < body.size(); ++i) {
    char ch = body[i];
    if (ch == '\'' || ch == '"') {
      char quote = ch;
      output.push_back(ch);
      bool escaped = false;
      for (++i; i < body.size(); ++i) {
        output.push_back(body[i]);
        if (escaped) {
          escaped = false;
          continue;
        }
        if (body[i] == '\\') {
          escaped = true;
          continue;
        }
        if (body[i] == quote)
          break;
      }
      continue;
    }
    if (ch == '#') {
      while (i < body.size() && body[i] != '\n')
        ++i;
      if (i < body.size())
        output.push_back('\n');
      continue;
    }
    if (ch == '{') {
      int depth = 1;
      bool escaped = false;
      char quote = '\0';
      for (++i; i < body.size(); ++i) {
        char actionCh = body[i];
        if (quote != '\0') {
          if (escaped) {
            escaped = false;
            continue;
          }
          if (actionCh == '\\') {
            escaped = true;
            continue;
          }
          if (actionCh == quote)
            quote = '\0';
          continue;
        }
        if (actionCh == '\'' || actionCh == '"') {
          quote = actionCh;
          continue;
        }
        if (actionCh == '{') {
          ++depth;
          continue;
        }
        if (actionCh == '}') {
          --depth;
          if (depth == 0)
            break;
        }
      }
      output.push_back(' ');
      continue;
    }
    output.push_back(ch);
  }
  return output;
}

enum class ExprKind {
  Empty,
  Name,
  Literal,
  Sequence,
  Choice,
  Optional,
  Repeat0,
  Repeat1,
  SepRepeat1,
  And,
  Not,
  Eager,
  Cut,
};

struct Expr {
  ExprKind kind = ExprKind::Empty;
  std::string value;
  char quote = '\0';
  std::string label;
  std::vector<std::unique_ptr<Expr>> children;
};

std::unique_ptr<Expr> makeExpr(ExprKind kind) {
  auto expr = std::make_unique<Expr>();
  expr->kind = kind;
  return expr;
}

std::unique_ptr<Expr> makeLeaf(ExprKind kind, std::string value,
                               char quote = '\0') {
  auto expr = makeExpr(kind);
  expr->value = std::move(value);
  expr->quote = quote;
  return expr;
}

enum class GrammarTokenKind { Identifier, Literal, Symbol, End };

struct GrammarToken {
  GrammarTokenKind kind = GrammarTokenKind::End;
  std::string text;
  char quote = '\0';
};

class GrammarLexer {
public:
  explicit GrammarLexer(std::string_view input) : input(input) {}

  std::vector<GrammarToken> lex() {
    std::vector<GrammarToken> tokens;
    while (true) {
      GrammarToken token = next();
      const bool end = token.kind == GrammarTokenKind::End;
      tokens.push_back(std::move(token));
      if (end)
        break;
    }
    return tokens;
  }

private:
  std::string_view input;
  std::size_t cursor = 0;

  GrammarToken next() {
    while (cursor < input.size() &&
           std::isspace(static_cast<unsigned char>(input[cursor])))
      ++cursor;
    if (cursor >= input.size())
      return {};

    char ch = input[cursor];
    if (isIdentStart(ch)) {
      std::size_t start = cursor++;
      while (cursor < input.size() && isIdentContinue(input[cursor]))
        ++cursor;
      return {GrammarTokenKind::Identifier,
              std::string(input.substr(start, cursor - start)), '\0'};
    }

    if (ch == '\'' || ch == '"') {
      char quote = ch;
      std::string literal;
      bool escaped = false;
      for (++cursor; cursor < input.size(); ++cursor) {
        char literalCh = input[cursor];
        if (escaped) {
          literal.push_back(literalCh);
          escaped = false;
          continue;
        }
        if (literalCh == '\\') {
          escaped = true;
          continue;
        }
        if (literalCh == quote) {
          ++cursor;
          break;
        }
        literal.push_back(literalCh);
      }
      return {GrammarTokenKind::Literal, std::move(literal), quote};
    }

    if (startsWith(input.substr(cursor), "&&")) {
      cursor += 2;
      return {GrammarTokenKind::Symbol, "&&", '\0'};
    }

    ++cursor;
    return {GrammarTokenKind::Symbol, std::string(1, ch), '\0'};
  }
};

class GrammarParser {
public:
  explicit GrammarParser(std::string_view input)
      : tokens(GrammarLexer(input).lex()) {}

  std::unique_ptr<Expr> parse() { return parseChoice(); }

private:
  std::vector<GrammarToken> tokens;
  std::size_t cursor = 0;

  const GrammarToken &current() const { return tokens[cursor]; }
  bool at(GrammarTokenKind kind) const { return current().kind == kind; }
  bool atSymbol(std::string_view text) const {
    return at(GrammarTokenKind::Symbol) && current().text == text;
  }
  bool matchSymbol(std::string_view text) {
    if (!atSymbol(text))
      return false;
    ++cursor;
    return true;
  }

  std::unique_ptr<Expr> parseChoice() {
    auto choice = makeExpr(ExprKind::Choice);
    if (matchSymbol("|")) {
      choice->children.push_back(parseSequence());
      while (matchSymbol("|"))
        choice->children.push_back(parseSequence());
    } else {
      choice->children.push_back(parseSequence());
      while (matchSymbol("|"))
        choice->children.push_back(parseSequence());
    }
    if (choice->children.size() == 1)
      return std::move(choice->children.front());
    return choice;
  }

  std::unique_ptr<Expr> parseSequence() {
    auto sequence = makeExpr(ExprKind::Sequence);
    while (!at(GrammarTokenKind::End) && !atSymbol("|") && !atSymbol(")") &&
           !atSymbol("]")) {
      sequence->children.push_back(parseItem());
    }
    if (sequence->children.empty())
      return makeExpr(ExprKind::Empty);
    if (sequence->children.size() == 1)
      return std::move(sequence->children.front());
    return sequence;
  }

  std::unique_ptr<Expr> parseItem() {
    std::string label = parseLabel();
    auto attachLabel = [&](std::unique_ptr<Expr> expr) {
      expr->label = std::move(label);
      return expr;
    };
    if (matchSymbol("~"))
      return attachLabel(makeExpr(ExprKind::Cut));
    if (matchSymbol("&&")) {
      auto expr = makeExpr(ExprKind::Eager);
      expr->children.push_back(parseItem());
      return attachLabel(std::move(expr));
    }
    if (matchSymbol("&")) {
      auto expr = makeExpr(ExprKind::And);
      expr->children.push_back(parseItem());
      return attachLabel(std::move(expr));
    }
    if (matchSymbol("!")) {
      auto expr = makeExpr(ExprKind::Not);
      expr->children.push_back(parseItem());
      return attachLabel(std::move(expr));
    }
    return attachLabel(parseSuffix());
  }

  std::string parseLabel() {
    if (!at(GrammarTokenKind::Identifier))
      return std::string();
    std::size_t saved = cursor;
    std::string label = current().text;
    ++cursor;
    if (matchSymbol("[")) {
      int depth = 1;
      while (!at(GrammarTokenKind::End) && depth > 0) {
        if (matchSymbol("[")) {
          ++depth;
          continue;
        }
        if (matchSymbol("]")) {
          --depth;
          continue;
        }
        ++cursor;
      }
    }
    if (matchSymbol("="))
      return label;
    cursor = saved;
    return std::string();
  }

  std::unique_ptr<Expr> parseSuffix() {
    std::unique_ptr<Expr> primary = parsePrimary();
    if (matchSymbol(".")) {
      auto separated = makeExpr(ExprKind::SepRepeat1);
      separated->children.push_back(std::move(primary));
      separated->children.push_back(parsePrimary());
      matchSymbol("+");
      return separated;
    }
    if (matchSymbol("*")) {
      auto repeated = makeExpr(ExprKind::Repeat0);
      repeated->children.push_back(std::move(primary));
      return repeated;
    }
    if (matchSymbol("+")) {
      auto repeated = makeExpr(ExprKind::Repeat1);
      repeated->children.push_back(std::move(primary));
      return repeated;
    }
    if (matchSymbol("?")) {
      auto optional = makeExpr(ExprKind::Optional);
      optional->children.push_back(std::move(primary));
      return optional;
    }
    return primary;
  }

  std::unique_ptr<Expr> parsePrimary() {
    if (at(GrammarTokenKind::Identifier))
      return makeLeaf(ExprKind::Name, tokens[cursor++].text);
    if (at(GrammarTokenKind::Literal)) {
      GrammarToken token = tokens[cursor++];
      return makeLeaf(ExprKind::Literal, std::move(token.text), token.quote);
    }
    if (matchSymbol("(")) {
      std::unique_ptr<Expr> expr = parseChoice();
      matchSymbol(")");
      return expr;
    }
    if (matchSymbol("[")) {
      auto optional = makeExpr(ExprKind::Optional);
      optional->children.push_back(parseChoice());
      matchSymbol("]");
      return optional;
    }

    if (!at(GrammarTokenKind::End))
      ++cursor;
    return makeExpr(ExprKind::Empty);
  }
};

struct RuleImpl {
  CpythonPegRule rule;
  std::unique_ptr<Expr> expr;
  std::vector<std::string> actions;
};

struct FirstAccumulator {
  bool nullable = false;
  bool unknown = false;
  std::set<std::string> literals;
  std::set<std::string> tokens;
};

void mergeInto(FirstAccumulator &target, const FirstAccumulator &source) {
  target.unknown = target.unknown || source.unknown;
  target.literals.insert(source.literals.begin(), source.literals.end());
  target.tokens.insert(source.tokens.begin(), source.tokens.end());
}

FirstAccumulator evalFirst(const Expr *expr,
                           const std::map<std::string, FirstAccumulator> &rules,
                           bool includeInvalidRules) {
  FirstAccumulator result;
  if (!expr) {
    result.nullable = true;
    return result;
  }

  switch (expr->kind) {
  case ExprKind::Empty:
  case ExprKind::Cut:
    result.nullable = true;
    return result;
  case ExprKind::Literal:
    result.literals.insert(expr->value);
    return result;
  case ExprKind::Name: {
    if (isTokenName(expr->value)) {
      result.tokens.insert(expr->value);
      return result;
    }
    auto found = rules.find(expr->value);
    if (found == rules.end()) {
      result.unknown = true;
      return result;
    }
    if (!includeInvalidRules && startsWith(expr->value, "invalid_"))
      return result;
    return found->second;
  }
  case ExprKind::Choice:
    for (const auto &child : expr->children) {
      FirstAccumulator childFirst =
          evalFirst(child.get(), rules, includeInvalidRules);
      mergeInto(result, childFirst);
      result.nullable = result.nullable || childFirst.nullable;
    }
    return result;
  case ExprKind::Sequence:
    result.nullable = true;
    for (const auto &child : expr->children) {
      FirstAccumulator childFirst =
          evalFirst(child.get(), rules, includeInvalidRules);
      mergeInto(result, childFirst);
      if (!childFirst.nullable) {
        result.nullable = false;
        return result;
      }
    }
    return result;
  case ExprKind::Optional:
  case ExprKind::Repeat0:
    if (!expr->children.empty())
      mergeInto(result, evalFirst(expr->children.front().get(), rules,
                                  includeInvalidRules));
    result.nullable = true;
    return result;
  case ExprKind::Repeat1:
    if (!expr->children.empty())
      result =
          evalFirst(expr->children.front().get(), rules, includeInvalidRules);
    return result;
  case ExprKind::SepRepeat1:
    if (expr->children.size() >= 2)
      return evalFirst(expr->children[1].get(), rules, includeInvalidRules);
    result.unknown = true;
    return result;
  case ExprKind::And:
    if (!expr->children.empty())
      mergeInto(result, evalFirst(expr->children.front().get(), rules,
                                  includeInvalidRules));
    result.nullable = true;
    return result;
  case ExprKind::Not:
    result.nullable = true;
    return result;
  case ExprKind::Eager:
    if (!expr->children.empty())
      return evalFirst(expr->children.front().get(), rules,
                       includeInvalidRules);
    result.unknown = true;
    return result;
  }
  result.unknown = true;
  return result;
}

bool sameFirst(const FirstAccumulator &lhs, const FirstAccumulator &rhs) {
  return lhs.nullable == rhs.nullable && lhs.unknown == rhs.unknown &&
         lhs.literals == rhs.literals && lhs.tokens == rhs.tokens;
}

std::vector<std::string> sortedVector(const std::set<std::string> &values) {
  return std::vector<std::string>(values.begin(), values.end());
}

void collectExprRefs(const Expr *expr, std::set<std::string> &ruleRefs,
                     std::set<std::string> &tokenRefs,
                     std::set<std::string> &literalRefs) {
  if (!expr)
    return;

  if (expr->kind == ExprKind::Name) {
    if (isTokenName(expr->value))
      tokenRefs.insert(expr->value);
    else
      ruleRefs.insert(expr->value);
    return;
  }
  if (expr->kind == ExprKind::Literal) {
    literalRefs.insert(expr->value);
    return;
  }

  for (const auto &child : expr->children)
    collectExprRefs(child.get(), ruleRefs, tokenRefs, literalRefs);
}

std::map<std::string, FirstAccumulator>
computeFirstSets(const std::vector<RuleImpl> &rules, bool includeInvalidRules) {
  std::map<std::string, FirstAccumulator> firstSets;
  for (const RuleImpl &impl : rules)
    firstSets.emplace(impl.rule.name, FirstAccumulator{});

  for (int iteration = 0; iteration < 256; ++iteration) {
    bool changed = false;
    for (const RuleImpl &impl : rules) {
      FirstAccumulator next;
      if (includeInvalidRules || !impl.rule.invalid)
        next = evalFirst(impl.expr.get(), firstSets, includeInvalidRules);
      FirstAccumulator &current = firstSets[impl.rule.name];
      if (!sameFirst(current, next)) {
        current = std::move(next);
        changed = true;
      }
    }
    if (!changed)
      break;
  }
  return firstSets;
}

CpythonPegFirstSet toPublicFirstSet(const FirstAccumulator &first) {
  CpythonPegFirstSet result;
  result.nullable = first.nullable;
  result.unknown = first.unknown;
  result.literals = sortedVector(first.literals);
  result.tokens = sortedVector(first.tokens);
  return result;
}

std::vector<const Expr *> topLevelAlternatives(const Expr *expr) {
  if (!expr)
    return {};
  if (expr->kind != ExprKind::Choice)
    return {expr};

  std::vector<const Expr *> alternatives;
  alternatives.reserve(expr->children.size());
  for (const auto &child : expr->children)
    alternatives.push_back(child.get());
  return alternatives;
}

CpythonPegExprKind toPublicKind(ExprKind kind) {
  switch (kind) {
  case ExprKind::Empty:
    return CpythonPegExprKind::Empty;
  case ExprKind::Name:
    return CpythonPegExprKind::Name;
  case ExprKind::Literal:
    return CpythonPegExprKind::Literal;
  case ExprKind::Sequence:
    return CpythonPegExprKind::Sequence;
  case ExprKind::Choice:
    return CpythonPegExprKind::Choice;
  case ExprKind::Optional:
    return CpythonPegExprKind::Optional;
  case ExprKind::Repeat0:
    return CpythonPegExprKind::Repeat0;
  case ExprKind::Repeat1:
    return CpythonPegExprKind::Repeat1;
  case ExprKind::SepRepeat1:
    return CpythonPegExprKind::SepRepeat1;
  case ExprKind::And:
    return CpythonPegExprKind::And;
  case ExprKind::Not:
    return CpythonPegExprKind::Not;
  case ExprKind::Eager:
    return CpythonPegExprKind::Eager;
  case ExprKind::Cut:
    return CpythonPegExprKind::Cut;
  }
  return CpythonPegExprKind::Empty;
}

CpythonPegExpr toPublicExpr(const Expr *expr) {
  CpythonPegExpr result;
  if (!expr)
    return result;
  result.kind = toPublicKind(expr->kind);
  result.value = expr->value;
  result.quote = expr->quote;
  result.label = expr->label;
  result.children.reserve(expr->children.size());
  for (const auto &child : expr->children)
    result.children.push_back(toPublicExpr(child.get()));
  return result;
}

void skipQuoted(std::string_view text, std::size_t &cursor) {
  char quote = text[cursor];
  bool escaped = false;
  for (++cursor; cursor < text.size(); ++cursor) {
    if (escaped) {
      escaped = false;
      continue;
    }
    if (text[cursor] == '\\') {
      escaped = true;
      continue;
    }
    if (text[cursor] == quote) {
      ++cursor;
      return;
    }
  }
}

void skipComment(std::string_view text, std::size_t &cursor) {
  while (cursor < text.size() && text[cursor] != '\n')
    ++cursor;
}

std::size_t matchingBrace(std::string_view text, std::size_t open) {
  int depth = 0;
  for (std::size_t cursor = open; cursor < text.size();) {
    char ch = text[cursor];
    if (ch == '\'' || ch == '"') {
      skipQuoted(text, cursor);
      continue;
    }
    if (ch == '#') {
      skipComment(text, cursor);
      continue;
    }
    if (ch == '{') {
      ++depth;
      ++cursor;
      continue;
    }
    if (ch == '}') {
      --depth;
      if (depth == 0)
        return cursor;
      ++cursor;
      continue;
    }
    ++cursor;
  }
  return std::string_view::npos;
}

std::vector<std::string> splitTopLevelAlternatives(std::string_view body) {
  std::vector<std::string> alternatives;
  std::size_t start = 0;
  int groupingDepth = 0;
  for (std::size_t cursor = 0; cursor < body.size();) {
    char ch = body[cursor];
    if (ch == '\'' || ch == '"') {
      skipQuoted(body, cursor);
      continue;
    }
    if (ch == '#') {
      skipComment(body, cursor);
      continue;
    }
    if (ch == '{') {
      std::size_t close = matchingBrace(body, cursor);
      cursor = close == std::string_view::npos ? body.size() : close + 1;
      continue;
    }
    if (ch == '(' || ch == '[') {
      ++groupingDepth;
      ++cursor;
      continue;
    }
    if (ch == ')' || ch == ']') {
      groupingDepth = std::max(0, groupingDepth - 1);
      ++cursor;
      continue;
    }
    if (ch == '|' && groupingDepth == 0) {
      std::string alternative =
          trim(std::string(body.substr(start, cursor - start)));
      if (!alternative.empty())
        alternatives.push_back(std::move(alternative));
      start = cursor + 1;
    }
    ++cursor;
  }

  std::string alternative = trim(std::string(body.substr(start)));
  if (!alternative.empty())
    alternatives.push_back(std::move(alternative));
  return alternatives;
}

std::string extractTopLevelAction(std::string_view alternative) {
  int groupingDepth = 0;
  for (std::size_t cursor = 0; cursor < alternative.size();) {
    char ch = alternative[cursor];
    if (ch == '\'' || ch == '"') {
      skipQuoted(alternative, cursor);
      continue;
    }
    if (ch == '#') {
      skipComment(alternative, cursor);
      continue;
    }
    if (ch == '(' || ch == '[') {
      ++groupingDepth;
      ++cursor;
      continue;
    }
    if (ch == ')' || ch == ']') {
      groupingDepth = std::max(0, groupingDepth - 1);
      ++cursor;
      continue;
    }
    if (ch == '{' && groupingDepth == 0) {
      std::size_t close = matchingBrace(alternative, cursor);
      if (close == std::string_view::npos)
        return std::string();
      return trim(
          std::string(alternative.substr(cursor + 1, close - cursor - 1)));
    }
    ++cursor;
  }
  return std::string();
}

std::vector<std::string> extractAlternativeActions(std::string_view body) {
  std::vector<std::string> actions;
  for (const std::string &alternative : splitTopLevelAlternatives(body))
    actions.push_back(extractTopLevelAction(alternative));
  return actions;
}

bool isCIdentifierStart(char ch) {
  unsigned char typed = static_cast<unsigned char>(ch);
  return std::isalpha(typed) || ch == '_';
}

bool isCIdentifierContinue(char ch) {
  unsigned char typed = static_cast<unsigned char>(ch);
  return std::isalnum(typed) || ch == '_';
}

std::vector<std::string> collectActionCallees(std::string_view action) {
  std::set<std::string> callees;
  for (std::size_t cursor = 0; cursor < action.size();) {
    char ch = action[cursor];
    if (ch == '\'' || ch == '"') {
      skipQuoted(action, cursor);
      continue;
    }
    if (ch == '#') {
      skipComment(action, cursor);
      continue;
    }
    if (!isCIdentifierStart(ch)) {
      ++cursor;
      continue;
    }

    std::size_t start = cursor++;
    while (cursor < action.size() && isCIdentifierContinue(action[cursor]))
      ++cursor;
    std::string name(action.substr(start, cursor - start));

    std::size_t lookahead = cursor;
    while (lookahead < action.size() &&
           std::isspace(static_cast<unsigned char>(action[lookahead])))
      ++lookahead;
    if (lookahead < action.size() && action[lookahead] == '(')
      callees.insert(std::move(name));
  }
  return sortedVector(callees);
}

std::vector<std::string>
collectAstConstructors(const std::vector<std::string> &callees) {
  std::vector<std::string> constructors;
  constexpr std::string_view prefix = "_PyAST_";
  for (const std::string &callee : callees) {
    if (!startsWith(callee, prefix))
      continue;
    constructors.push_back(callee.substr(prefix.size()));
  }
  std::sort(constructors.begin(), constructors.end());
  constructors.erase(std::unique(constructors.begin(), constructors.end()),
                     constructors.end());
  return constructors;
}

std::string headRule(const Expr *expr) {
  if (!expr)
    return std::string();

  switch (expr->kind) {
  case ExprKind::Name:
    return isTokenName(expr->value) ? std::string() : expr->value;
  case ExprKind::Sequence:
  case ExprKind::Choice:
    for (const auto &child : expr->children) {
      std::string found = headRule(child.get());
      if (!found.empty())
        return found;
    }
    return std::string();
  case ExprKind::Optional:
  case ExprKind::Repeat0:
  case ExprKind::Repeat1:
  case ExprKind::SepRepeat1:
  case ExprKind::And:
  case ExprKind::Not:
  case ExprKind::Eager:
    for (const auto &child : expr->children) {
      std::string found = headRule(child.get());
      if (!found.empty())
        return found;
    }
    return std::string();
  case ExprKind::Empty:
  case ExprKind::Literal:
  case ExprKind::Cut:
    return std::string();
  }
  return std::string();
}

std::vector<CpythonPegAlternative> computeAlternatives(
    const Expr *expr, const std::map<std::string, FirstAccumulator> &firstSets,
    const std::map<std::string, FirstAccumulator> &recoveryFirstSets,
    const std::vector<std::string> &actions) {
  std::vector<CpythonPegAlternative> alternatives;
  std::vector<const Expr *> expressions = topLevelAlternatives(expr);
  for (std::size_t index = 0; index < expressions.size(); ++index) {
    const Expr *alternativeExpr = expressions[index];
    CpythonPegAlternative alternative;
    alternative.expr = toPublicExpr(alternativeExpr);
    alternative.headRule = headRule(alternativeExpr);
    if (index < actions.size()) {
      alternative.action = actions[index];
      alternative.actionCallees = collectActionCallees(alternative.action);
      alternative.astConstructors =
          collectAstConstructors(alternative.actionCallees);
    }
    alternative.first = toPublicFirstSet(
        evalFirst(alternativeExpr, firstSets, /*includeInvalidRules=*/false));
    alternative.recoveryFirst = toPublicFirstSet(evalFirst(
        alternativeExpr, recoveryFirstSets, /*includeInvalidRules=*/true));
    std::set<std::string> ruleRefs;
    std::set<std::string> tokenRefs;
    std::set<std::string> literalRefs;
    collectExprRefs(alternativeExpr, ruleRefs, tokenRefs, literalRefs);
    alternative.ruleRefs = sortedVector(ruleRefs);
    alternative.tokenRefs = sortedVector(tokenRefs);
    alternative.literalRefs = sortedVector(literalRefs);
    alternatives.push_back(std::move(alternative));
  }
  return alternatives;
}

std::vector<RuleImpl> extractRules(std::string_view grammar,
                                   std::vector<std::string> &diagnostics) {
  std::vector<RuleImpl> rules;
  std::istringstream lines{std::string(grammar)};
  std::string line;
  std::optional<RuleHeader> currentHeader;
  std::ostringstream currentBody;
  bool skippingTrailer = false;

  auto flush = [&] {
    if (!currentHeader)
      return;
    std::string body = currentHeader->body + "\n" + currentBody.str();
    std::string cleaned = stripActionsAndComments(body);
    RuleImpl impl;
    impl.rule.name = currentHeader->name;
    impl.rule.returnType = currentHeader->returnType;
    impl.rule.memo = currentHeader->memo;
    impl.rule.invalid = startsWith(impl.rule.name, "invalid_");
    impl.actions = extractAlternativeActions(body);
    impl.expr = GrammarParser(cleaned).parse();
    rules.push_back(std::move(impl));
    currentHeader.reset();
    currentBody.str(std::string());
    currentBody.clear();
  };

  while (std::getline(lines, line)) {
    if (skippingTrailer) {
      if (line.find("'''") != std::string::npos)
        skippingTrailer = false;
      continue;
    }
    if (startsWith(trim(line), "@trailer")) {
      if (line.find("'''") == std::string::npos ||
          line.find("'''") == line.rfind("'''"))
        skippingTrailer = true;
      continue;
    }

    RuleHeader parsed;
    if (parseRuleHeader(line, parsed)) {
      flush();
      currentHeader = std::move(parsed);
      continue;
    }
    if (currentHeader)
      currentBody << line << '\n';
  }
  flush();

  if (rules.empty())
    diagnostics.push_back("CPython python.gram did not contain PEG rules");
  return rules;
}

} // namespace

CpythonPegFirstSet::CpythonPegFirstSet(bool nullable, bool unknown,
                                       std::vector<std::string> literals,
                                       std::vector<std::string> tokens)
    : nullable(nullable), unknown(unknown), literals(std::move(literals)),
      tokens(std::move(tokens)) {}

CpythonPegRule::CpythonPegRule(std::string name, std::string returnType,
                               bool memo, bool invalid, CpythonPegExpr expr,
                               CpythonPegFirstSet first,
                               CpythonPegFirstSet recoveryFirst,
                               std::vector<CpythonPegAlternative> alternatives)
    : name(std::move(name)), returnType(std::move(returnType)), memo(memo),
      invalid(invalid), expr(std::move(expr)), first(std::move(first)),
      recoveryFirst(std::move(recoveryFirst)),
      alternatives(std::move(alternatives)) {}

void CpythonPegFirstSet::buildLookupTables() {
  literalLookup.clear();
  literalLookup.reserve(literals.size());
  for (const std::string &literal : literals)
    literalLookup.emplace(literal);

  tokenLookup.clear();
  tokenLookup.reserve(tokens.size());
  for (const std::string &token : tokens)
    tokenLookup.emplace(token);
}

bool CpythonPegFirstSet::hasLiteral(std::string_view literal) const {
  if (!literalLookup.empty())
    return literalLookup.find(literal) != literalLookup.end();
  return std::find(literals.begin(), literals.end(), literal) != literals.end();
}

bool CpythonPegFirstSet::hasToken(std::string_view token) const {
  if (!tokenLookup.empty())
    return tokenLookup.find(token) != tokenLookup.end();
  return std::find(tokens.begin(), tokens.end(), token) != tokens.end();
}

void CpythonPegRule::buildLookupTables() {
  first.buildLookupTables();
  recoveryFirst.buildLookupTables();

  alternativeIndices.clear();
  actionCalleeLookup.clear();
  astConstructorLookup.clear();
  literalRefLookup.clear();
  cachedUniqueAstConstructors.clear();
  cachedSingleAstConstructor.reset();

  alternativeIndices.reserve(alternatives.size());
  std::set<std::string> constructors;
  for (std::size_t i = 0; i < alternatives.size(); ++i) {
    CpythonPegAlternative &alternative = alternatives[i];
    alternative.first.buildLookupTables();
    alternative.recoveryFirst.buildLookupTables();
    if (!alternative.headRule.empty())
      alternativeIndices.emplace(alternative.headRule, i);
    for (const std::string &callee : alternative.actionCallees)
      actionCalleeLookup.emplace(callee);
    for (const std::string &constructor : alternative.astConstructors) {
      astConstructorLookup.emplace(constructor);
      constructors.insert(constructor);
    }
    for (const std::string &literal : alternative.literalRefs)
      literalRefLookup.emplace(literal);
  }

  cachedUniqueAstConstructors.assign(
      std::make_move_iterator(constructors.begin()),
      std::make_move_iterator(constructors.end()));
  if (cachedUniqueAstConstructors.size() == 1)
    cachedSingleAstConstructor = cachedUniqueAstConstructors.front();
}

void CpythonPegGrammar::buildLookupTables() {
  ruleLookup.clear();
  ruleLookup.reserve(rules.size());
  for (std::size_t i = 0; i < rules.size(); ++i) {
    rules[i].buildLookupTables();
    ruleLookup.emplace(rules[i].name, i);
  }
}

const CpythonPegRule *CpythonPegGrammar::findRule(std::string_view name) const {
  if (!ruleLookup.empty()) {
    auto found = ruleLookup.find(name);
    if (found == ruleLookup.end())
      return nullptr;
    return &rules[found->second];
  }
  auto found =
      std::find_if(rules.begin(), rules.end(), [&](const CpythonPegRule &rule) {
        return rule.name == name;
      });
  return found == rules.end() ? nullptr : &*found;
}

const CpythonPegAlternative *
CpythonPegRule::findAlternativeByHead(std::string_view head) const {
  if (!alternativeIndices.empty()) {
    auto found = alternativeIndices.find(head);
    if (found == alternativeIndices.end())
      return nullptr;
    return &alternatives[found->second];
  }
  for (const CpythonPegAlternative &alternative : alternatives)
    if (alternative.headRule == head)
      return &alternative;
  return nullptr;
}

bool CpythonPegRule::callsAction(std::string_view callee) const {
  if (!actionCalleeLookup.empty())
    return actionCalleeLookup.find(callee) != actionCalleeLookup.end();
  for (const CpythonPegAlternative &alternative : alternatives) {
    if (std::find(alternative.actionCallees.begin(),
                  alternative.actionCallees.end(),
                  std::string(callee)) != alternative.actionCallees.end())
      return true;
  }
  return false;
}

bool CpythonPegRule::hasAstConstructor(std::string_view constructor) const {
  if (!astConstructorLookup.empty())
    return astConstructorLookup.find(constructor) != astConstructorLookup.end();
  for (const CpythonPegAlternative &alternative : alternatives) {
    if (std::find(alternative.astConstructors.begin(),
                  alternative.astConstructors.end(),
                  std::string(constructor)) !=
        alternative.astConstructors.end())
      return true;
  }
  return false;
}

bool CpythonPegRule::referencesLiteral(std::string_view literal) const {
  if (!literalRefLookup.empty())
    return literalRefLookup.find(literal) != literalRefLookup.end();
  for (const CpythonPegAlternative &alternative : alternatives)
    if (std::find(alternative.literalRefs.begin(),
                  alternative.literalRefs.end(),
                  literal) != alternative.literalRefs.end())
      return true;
  return false;
}

std::vector<std::string> CpythonPegRule::uniqueAstConstructors() const {
  if (!cachedUniqueAstConstructors.empty())
    return cachedUniqueAstConstructors;
  std::set<std::string> constructors;
  for (const CpythonPegAlternative &alternative : alternatives)
    constructors.insert(alternative.astConstructors.begin(),
                        alternative.astConstructors.end());
  return sortedVector(constructors);
}

std::optional<std::string> CpythonPegRule::singleAstConstructor() const {
  if (cachedSingleAstConstructor)
    return cachedSingleAstConstructor;
  std::vector<std::string> constructors = uniqueAstConstructors();
  if (constructors.size() != 1)
    return std::nullopt;
  return constructors.front();
}

CpythonPegGrammar parseCpythonPegGrammar(std::string_view grammarText) {
  CpythonPegGrammar grammar;
  std::vector<RuleImpl> rules = extractRules(grammarText, grammar.diagnostics);

  std::map<std::string, FirstAccumulator> firstSets =
      computeFirstSets(rules, /*includeInvalidRules=*/false);
  std::map<std::string, FirstAccumulator> recoveryFirstSets =
      computeFirstSets(rules, /*includeInvalidRules=*/true);

  grammar.rules.reserve(rules.size());
  for (RuleImpl &impl : rules) {
    impl.rule.expr = toPublicExpr(impl.expr.get());
    impl.rule.first = toPublicFirstSet(firstSets[impl.rule.name]);
    impl.rule.recoveryFirst =
        toPublicFirstSet(recoveryFirstSets[impl.rule.name]);
    impl.rule.alternatives = computeAlternatives(
        impl.expr.get(), firstSets, recoveryFirstSets, impl.actions);
    grammar.ruleIndices.emplace(impl.rule.name, grammar.rules.size());
    grammar.rules.push_back(std::move(impl.rule));
  }

  grammar.buildLookupTables();
  return grammar;
}

} // namespace lython::parser
