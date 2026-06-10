#include "lython/parser/CpythonSpec.h"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <fstream>
#include <iterator>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#ifndef LYTHON_CPYTHON_314_PARSER_ROOT
#define LYTHON_CPYTHON_314_PARSER_ROOT ""
#endif

namespace lython::parser {
namespace {

#if defined(LYTHON_CPYTHON_SPEC_RUNTIME_FILE_IO)

std::string trim(std::string text);

std::string readTextFile(const std::string &path) {
  std::ifstream input(path);
  if (!input)
    throw std::runtime_error("failed to open CPython parser resource: " + path);
  std::ostringstream buffer;
  buffer << input.rdbuf();
  return buffer.str();
}

std::vector<TokenSpec> parseTokens(const std::string &tokens) {
  std::vector<TokenSpec> result;
  std::istringstream lines(tokens);
  std::string line;
  int tokenId = 0;
  while (std::getline(lines, line)) {
    const std::size_t comment = line.find('#');
    if (comment != std::string::npos)
      line.erase(comment);
    const std::size_t first = line.find_first_not_of(" \t\r\n");
    if (first == std::string::npos)
      continue;
    const std::size_t nameEnd = line.find_first_of(" \t\r\n", first);
    TokenSpec token;
    token.id = tokenId++;
    token.name = line.substr(first, nameEnd - first);
    const std::size_t quote = line.find('\'', nameEnd);
    if (quote != std::string::npos) {
      const std::size_t endQuote = line.find('\'', quote + 1);
      if (endQuote != std::string::npos)
        token.spelling = line.substr(quote + 1, endQuote - quote - 1);
    }
    result.push_back(std::move(token));
  }
  return result;
}

std::map<std::string, int>
parseGeneratedRuleTypeIds(const std::string &parserSource) {
  std::map<std::string, int> result;
  std::istringstream lines(parserSource);
  std::string line;
  while (std::getline(lines, line)) {
    line = trim(std::move(line));
    constexpr std::string_view prefix = "#define ";
    constexpr std::string_view suffix = "_type";
    if (line.compare(0, prefix.size(), prefix) != 0)
      continue;

    std::istringstream fields(line.substr(prefix.size()));
    std::string name;
    int typeId = -1;
    if (!(fields >> name >> typeId))
      continue;
    if (name.size() <= suffix.size() ||
        name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0)
      continue;

    name.erase(name.size() - suffix.size());
    result.emplace(std::move(name), typeId);
  }
  return result;
}

std::vector<GeneratedRuleSpec>
parseGeneratedRules(const std::string &parserSource,
                    const std::map<std::string, int> &typeIds) {
  std::vector<GeneratedRuleSpec> result;
  std::istringstream lines(parserSource);
  std::string line;
  while (std::getline(lines, line)) {
    line = trim(std::move(line));
    constexpr std::string_view prefix = "static ";
    constexpr std::string_view suffix = "(Parser *p);";
    if (line.compare(0, prefix.size(), prefix) != 0)
      continue;
    if (line.size() < suffix.size() ||
        line.compare(line.size() - suffix.size(), suffix.size(), suffix) != 0)
      continue;

    std::string declaration =
        line.substr(prefix.size(), line.size() - prefix.size() - suffix.size());
    const std::size_t space = declaration.find_last_of(" \t");
    if (space == std::string::npos)
      continue;
    GeneratedRuleSpec rule;
    rule.returnType = trim(declaration.substr(0, space));
    rule.functionName = trim(declaration.substr(space + 1));
    while (!rule.functionName.empty() && rule.functionName.front() == '*') {
      rule.returnType += '*';
      rule.functionName.erase(rule.functionName.begin());
    }
    constexpr std::string_view ruleSuffix = "_rule";
    if (rule.functionName.size() <= ruleSuffix.size() ||
        rule.functionName.compare(rule.functionName.size() - ruleSuffix.size(),
                                  ruleSuffix.size(), ruleSuffix) != 0)
      continue;
    rule.name = rule.functionName.substr(0, rule.functionName.size() -
                                                ruleSuffix.size());
    if (auto found = typeIds.find(rule.name); found != typeIds.end())
      rule.typeId = found->second;
    if (!rule.name.empty() && !rule.returnType.empty())
      result.push_back(std::move(rule));
  }
  return result;
}

std::map<std::string, std::size_t>
collectGeneratedRuleIndices(const std::vector<GeneratedRuleSpec> &rules) {
  std::map<std::string, std::size_t> result;
  for (std::size_t i = 0; i < rules.size(); ++i)
    result.emplace(rules[i].name, i);
  return result;
}

std::vector<std::string>
collectTokenNames(const std::vector<TokenSpec> &tokens) {
  std::vector<std::string> result;
  result.reserve(tokens.size());
  for (const TokenSpec &token : tokens)
    result.push_back(token.name);
  return result;
}

std::map<std::string, std::string>
collectTokenNameBySpelling(const std::vector<TokenSpec> &tokens) {
  std::map<std::string, std::string> result;
  for (const TokenSpec &token : tokens) {
    if (!token.spelling.empty())
      result.emplace(token.spelling, token.name);
  }
  return result;
}

std::map<std::string, int>
collectTokenIdByName(const std::vector<TokenSpec> &tokens) {
  std::map<std::string, int> result;
  for (const TokenSpec &token : tokens)
    result.emplace(token.name, token.id);
  return result;
}

std::map<std::string, int>
collectTokenIdBySpelling(const std::vector<TokenSpec> &tokens) {
  std::map<std::string, int> result;
  for (const TokenSpec &token : tokens) {
    if (!token.spelling.empty())
      result.emplace(token.spelling, token.id);
  }
  return result;
}

std::map<std::string, int>
collectKeywordTokenIdByText(const std::vector<GeneratedKeywordSpec> &keywords) {
  std::map<std::string, int> result;
  for (const GeneratedKeywordSpec &keyword : keywords)
    result.emplace(keyword.text, keyword.tokenId);
  return result;
}

bool isIdentifierLiteral(std::string_view text) {
  if (text.empty())
    return false;
  unsigned char first = static_cast<unsigned char>(text.front());
  if (!std::isalpha(first) && text.front() != '_')
    return false;
  for (char ch : text) {
    unsigned char typed = static_cast<unsigned char>(ch);
    if (!std::isalnum(typed) && ch != '_')
      return false;
  }
  return true;
}

std::vector<std::string> sorted(std::set<std::string> values) {
  return std::vector<std::string>(std::make_move_iterator(values.begin()),
                                  std::make_move_iterator(values.end()));
}

std::string decodeCStringLiteral(std::string_view literal) {
  std::string result;
  result.reserve(literal.size());
  bool escaped = false;
  for (char ch : literal) {
    if (!escaped) {
      if (ch == '\\') {
        escaped = true;
        continue;
      }
      result.push_back(ch);
      continue;
    }

    switch (ch) {
    case 'n':
      result.push_back('\n');
      break;
    case 'r':
      result.push_back('\r');
      break;
    case 't':
      result.push_back('\t');
      break;
    case '\\':
    case '"':
      result.push_back(ch);
      break;
    default:
      result.push_back(ch);
      break;
    }
    escaped = false;
  }
  return result;
}

std::vector<std::string> collectCStringLiterals(std::string_view text) {
  std::set<std::string> values;
  for (std::size_t cursor = 0; cursor < text.size(); ++cursor) {
    if (text[cursor] != '"')
      continue;

    const std::size_t start = cursor + 1;
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
      if (text[cursor] != '"')
        continue;

      values.insert(decodeCStringLiteral(text.substr(start, cursor - start)));
      break;
    }
  }
  return sorted(std::move(values));
}

std::string_view generatedParserSlice(std::string_view source,
                                      std::string_view beginMarker,
                                      std::string_view endMarker) {
  const std::size_t begin = source.find(beginMarker);
  if (begin == std::string_view::npos)
    return {};
  const std::size_t contentBegin = begin + beginMarker.size();
  const std::size_t end = source.find(endMarker, contentBegin);
  if (end == std::string_view::npos)
    return source.substr(contentBegin);
  return source.substr(contentBegin, end - contentBegin);
}

std::vector<std::string>
collectGeneratedHardKeywords(const std::string &parserSource) {
  std::string_view source(parserSource);
  std::string_view reserved = generatedParserSlice(
      source, "static KeywordToken *reserved_keywords[] = {",
      "static char *soft_keywords[] = {");
  return collectCStringLiterals(reserved);
}

std::vector<GeneratedKeywordSpec>
collectGeneratedHardKeywordSpecs(const std::string &parserSource) {
  std::vector<GeneratedKeywordSpec> result;
  std::string_view source(parserSource);
  std::string_view reserved = generatedParserSlice(
      source, "static KeywordToken *reserved_keywords[] = {",
      "static char *soft_keywords[] = {");

  for (std::size_t cursor = 0; cursor < reserved.size(); ++cursor) {
    if (reserved[cursor] != '{')
      continue;
    std::size_t quote = cursor + 1;
    while (quote < reserved.size() &&
           std::isspace(static_cast<unsigned char>(reserved[quote])))
      ++quote;
    if (quote >= reserved.size() || reserved[quote] != '"')
      continue;

    cursor = quote;
    const std::size_t stringStart = cursor + 1;
    bool escaped = false;
    for (++cursor; cursor < reserved.size(); ++cursor) {
      if (escaped) {
        escaped = false;
        continue;
      }
      if (reserved[cursor] == '\\') {
        escaped = true;
        continue;
      }
      if (reserved[cursor] != '"')
        continue;

      GeneratedKeywordSpec spec;
      spec.text = decodeCStringLiteral(
          reserved.substr(stringStart, cursor - stringStart));
      ++cursor;
      while (cursor < reserved.size() &&
             std::isspace(static_cast<unsigned char>(reserved[cursor])))
        ++cursor;
      if (cursor >= reserved.size() || reserved[cursor] != ',')
        break;
      ++cursor;
      while (cursor < reserved.size() &&
             std::isspace(static_cast<unsigned char>(reserved[cursor])))
        ++cursor;

      const char *begin = reserved.data() + cursor;
      const char *end = reserved.data() + reserved.size();
      int tokenId = -1;
      auto parsed = std::from_chars(begin, end, tokenId);
      if (parsed.ec == std::errc()) {
        spec.tokenId = tokenId;
        result.push_back(std::move(spec));
        cursor = static_cast<std::size_t>(parsed.ptr - reserved.data());
      }
      break;
    }
  }

  std::sort(
      result.begin(), result.end(),
      [](const GeneratedKeywordSpec &lhs, const GeneratedKeywordSpec &rhs) {
        return lhs.text < rhs.text;
      });
  result.erase(std::unique(result.begin(), result.end(),
                           [](const GeneratedKeywordSpec &lhs,
                              const GeneratedKeywordSpec &rhs) {
                             return lhs.text == rhs.text;
                           }),
               result.end());
  return result;
}

std::vector<std::string>
collectGeneratedSoftKeywords(const std::string &parserSource) {
  std::string_view source(parserSource);
  std::string_view soft = generatedParserSlice(
      source, "static char *soft_keywords[] = {", "#define ");
  std::vector<std::string> values = collectCStringLiterals(soft);
  values.erase(std::remove(values.begin(), values.end(), std::string()),
               values.end());
  return values;
}

std::optional<std::string> trailingGeneratedTokenText(std::string_view line) {
  constexpr std::string_view marker = "// token='";
  const std::size_t markerPos = line.find(marker);
  if (markerPos == std::string_view::npos)
    return std::nullopt;

  const std::size_t start = markerPos + marker.size();
  std::string text;
  bool escaped = false;
  for (std::size_t cursor = start; cursor < line.size(); ++cursor) {
    char ch = line[cursor];
    if (escaped) {
      text.push_back(ch);
      escaped = false;
      continue;
    }
    if (ch == '\\') {
      escaped = true;
      continue;
    }
    if (ch == '\'')
      return text;
    text.push_back(ch);
  }
  return std::nullopt;
}

std::vector<GeneratedTokenRefSpec>
collectGeneratedTokenRefs(const std::string &parserSource) {
  std::vector<GeneratedTokenRefSpec> result;
  std::istringstream lines(parserSource);
  std::string line;
  constexpr std::string_view marker = "_PyPegen_expect_token(p,";
  while (std::getline(lines, line)) {
    const std::size_t markerPos = line.find(marker);
    if (markerPos == std::string::npos)
      continue;

    std::size_t cursor = markerPos + marker.size();
    while (cursor < line.size() &&
           std::isspace(static_cast<unsigned char>(line[cursor])))
      ++cursor;
    int tokenId = -1;
    const char *begin = line.data() + cursor;
    const char *end = line.data() + line.size();
    auto parsed = std::from_chars(begin, end, tokenId);
    if (parsed.ec != std::errc())
      continue;

    std::optional<std::string> text = trailingGeneratedTokenText(line);
    if (!text || text->empty())
      continue;
    result.push_back(GeneratedTokenRefSpec{std::move(*text), tokenId});
  }
  return result;
}

void collectGrammarKeywords(const std::string &grammar,
                            std::vector<std::string> &hardKeywords,
                            std::vector<std::string> &softKeywords) {
  std::set<std::string> hard;
  std::set<std::string> soft;

  for (std::size_t i = 0; i < grammar.size(); ++i) {
    char quote = grammar[i];
    if (quote == '#') {
      while (i < grammar.size() && grammar[i] != '\n')
        ++i;
      continue;
    }
    if (quote != '\'' && quote != '"')
      continue;

    std::string literal;
    bool escaped = false;
    for (++i; i < grammar.size(); ++i) {
      char ch = grammar[i];
      if (escaped) {
        literal.push_back(ch);
        escaped = false;
        continue;
      }
      if (ch == '\\') {
        escaped = true;
        continue;
      }
      if (ch == quote)
        break;
      literal.push_back(ch);
    }
    if (!isIdentifierLiteral(literal))
      continue;
    if (quote == '\'')
      hard.insert(std::move(literal));
    else
      soft.insert(std::move(literal));
  }

  hardKeywords = sorted(std::move(hard));
  softKeywords = sorted(std::move(soft));
}

std::vector<std::string>
collectOperatorSpellings(const std::vector<TokenSpec> &tokens) {
  std::vector<std::string> result;
  for (const TokenSpec &token : tokens) {
    if (token.spelling.empty())
      continue;
    if (isIdentifierLiteral(token.spelling))
      continue;
    result.push_back(token.spelling);
  }
  std::sort(result.begin(), result.end(),
            [](const std::string &lhs, const std::string &rhs) {
              if (lhs.size() != rhs.size())
                return lhs.size() > rhs.size();
              return lhs < rhs;
            });
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

std::string trim(std::string text) {
  const std::size_t first = text.find_first_not_of(" \t\r\n");
  if (first == std::string::npos)
    return std::string();
  const std::size_t last = text.find_last_not_of(" \t\r\n");
  return text.substr(first, last - first + 1);
}

std::string stripAsdlComments(const std::string &asdl) {
  std::ostringstream output;
  std::istringstream lines(asdl);
  std::string line;
  while (std::getline(lines, line)) {
    const std::size_t comment = line.find("--");
    if (comment != std::string::npos)
      line.erase(comment);
    output << line << '\n';
  }
  return output.str();
}

bool startsWithUppercaseIdentifier(std::string_view text) {
  if (text.empty())
    return false;
  return std::isupper(static_cast<unsigned char>(text.front())) ||
         text.front() == '_';
}

std::string leadingIdentifier(std::string_view text) {
  std::string result;
  for (char ch : text) {
    if (!std::isalnum(static_cast<unsigned char>(ch)) && ch != '_')
      break;
    result.push_back(ch);
  }
  return result;
}

void collectAsdlConstructors(std::set<std::string> &kinds,
                             const std::string &text) {
  for (std::size_t i = 0; i < text.size(); ++i) {
    if (!std::isalpha(static_cast<unsigned char>(text[i])) && text[i] != '_')
      continue;
    const std::size_t start = i;
    while (
        i < text.size() &&
        (std::isalnum(static_cast<unsigned char>(text[i])) || text[i] == '_'))
      ++i;
    std::size_t cursor = i;
    while (cursor < text.size() &&
           std::isspace(static_cast<unsigned char>(text[cursor])))
      ++cursor;
    if (cursor < text.size() && text[cursor] == '(') {
      std::string name = text.substr(start, i - start);
      if (startsWithUppercaseIdentifier(name))
        kinds.insert(std::move(name));
    }
  }
}

void collectAsdlAlternatives(std::set<std::string> &kinds,
                             const std::string &text) {
  std::size_t cursor = 0;
  while (cursor < text.size()) {
    const std::size_t pipe = text.find('|', cursor);
    std::string alternative = trim(text.substr(cursor, pipe - cursor));
    std::string name = leadingIdentifier(alternative);
    if (startsWithUppercaseIdentifier(name))
      kinds.insert(std::move(name));
    if (pipe == std::string::npos)
      break;
    cursor = pipe + 1;
  }
}

std::vector<std::string> collectAstNodeKinds(const std::string &asdl) {
  std::set<std::string> kinds;
  std::istringstream lines(asdl);
  std::string line;
  while (std::getline(lines, line)) {
    const std::size_t comment = line.find("--");
    if (comment != std::string::npos)
      line.erase(comment);
    line = trim(std::move(line));
    if (line.empty() || line == "{" || line == "}")
      continue;

    const std::size_t equal = line.find('=');
    if (equal != std::string::npos) {
      std::string lhs = trim(line.substr(0, equal));
      std::string rhs = trim(line.substr(equal + 1));
      if (!lhs.empty() && !rhs.empty() && rhs.front() == '(')
        kinds.insert(std::move(lhs));
      collectAsdlConstructors(kinds, rhs);
      collectAsdlAlternatives(kinds, rhs);
      continue;
    }

    collectAsdlConstructors(kinds, line);
    collectAsdlAlternatives(kinds, line);
  }
  return sorted(std::move(kinds));
}

std::size_t skipSpaces(std::string_view text, std::size_t cursor) {
  while (cursor < text.size() &&
         std::isspace(static_cast<unsigned char>(text[cursor])))
    ++cursor;
  return cursor;
}

std::size_t findMatchingParen(std::string_view text, std::size_t open) {
  int depth = 0;
  for (std::size_t cursor = open; cursor < text.size(); ++cursor) {
    if (text[cursor] == '(') {
      ++depth;
      continue;
    }
    if (text[cursor] != ')')
      continue;
    --depth;
    if (depth == 0)
      return cursor;
  }
  return std::string_view::npos;
}

std::string trailingIdentifier(std::string text) {
  text = trim(std::move(text));
  if (text.empty())
    return std::string();

  std::size_t end = text.size();
  while (end > 0 && !std::isalnum(static_cast<unsigned char>(text[end - 1])) &&
         text[end - 1] != '_')
    --end;
  std::size_t begin = end;
  while (begin > 0 &&
         (std::isalnum(static_cast<unsigned char>(text[begin - 1])) ||
          text[begin - 1] == '_'))
    --begin;
  return text.substr(begin, end - begin);
}

std::vector<std::string> parseAsdlFieldNames(std::string_view fields) {
  std::vector<std::string> result;
  std::size_t cursor = 0;
  while (cursor < fields.size()) {
    const std::size_t comma = fields.find(',', cursor);
    std::string field = std::string(fields.substr(cursor, comma - cursor));
    std::string name = trailingIdentifier(std::move(field));
    if (!name.empty())
      result.push_back(std::move(name));
    if (comma == std::string_view::npos)
      break;
    cursor = comma + 1;
  }
  return result;
}

std::vector<std::string> splitTopLevelAlternatives(std::string_view text) {
  std::vector<std::string> result;
  std::size_t cursor = 0;
  int depth = 0;
  for (std::size_t i = 0; i <= text.size(); ++i) {
    if (i == text.size() || (text[i] == '|' && depth == 0)) {
      result.push_back(trim(std::string(text.substr(cursor, i - cursor))));
      cursor = i + 1;
      continue;
    }
    if (text[i] == '(')
      ++depth;
    else if (text[i] == ')' && depth > 0)
      --depth;
  }
  return result;
}

std::vector<std::string> splitTopLevelFields(std::string_view text) {
  std::vector<std::string> result;
  std::size_t cursor = 0;
  int depth = 0;
  for (std::size_t i = 0; i <= text.size(); ++i) {
    if (i == text.size() || (text[i] == ',' && depth == 0)) {
      result.push_back(trim(std::string(text.substr(cursor, i - cursor))));
      cursor = i + 1;
      continue;
    }
    if (text[i] == '(')
      ++depth;
    else if (text[i] == ')' && depth > 0)
      --depth;
  }
  return result;
}

AstFieldSpec parseAsdlFieldSpec(std::string field) {
  field = trim(std::move(field));
  AstFieldSpec spec;
  if (field.empty())
    return spec;

  const std::size_t space = field.find_first_of(" \t\r\n");
  std::string typeToken =
      space == std::string::npos ? field : field.substr(0, space);
  std::string nameToken =
      space == std::string::npos ? std::string() : trim(field.substr(space));

  spec.name = trailingIdentifier(std::move(nameToken));
  spec.sequence = !typeToken.empty() && typeToken.back() == '*';
  if (spec.sequence)
    typeToken.pop_back();
  spec.optional = !typeToken.empty() && typeToken.back() == '?';
  if (spec.optional)
    typeToken.pop_back();
  spec.nullableElement = spec.sequence && spec.optional;
  if (spec.sequence)
    spec.optional = false;
  spec.type = std::move(typeToken);
  return spec;
}

std::vector<AstFieldSpec> parseAsdlFieldSpecs(std::string_view fields) {
  std::vector<AstFieldSpec> result;
  for (std::string field : splitTopLevelFields(fields)) {
    AstFieldSpec spec = parseAsdlFieldSpec(std::move(field));
    if (!spec.name.empty() && !spec.type.empty())
      result.push_back(std::move(spec));
  }
  return result;
}

std::map<std::string, std::vector<std::string>>
collectAstFieldSchemas(const std::string &asdl,
                       const std::vector<std::string> &nodeKinds) {
  std::map<std::string, std::vector<std::string>> schemas;
  for (const std::string &kind : nodeKinds)
    schemas.emplace(kind, std::vector<std::string>{});

  const std::string text = stripAsdlComments(asdl);
  for (std::size_t cursor = 0; cursor < text.size();) {
    if (!std::isalpha(static_cast<unsigned char>(text[cursor])) &&
        text[cursor] != '_') {
      ++cursor;
      continue;
    }

    const std::size_t start = cursor;
    while (cursor < text.size() &&
           (std::isalnum(static_cast<unsigned char>(text[cursor])) ||
            text[cursor] == '_'))
      ++cursor;
    std::string name = text.substr(start, cursor - start);
    std::size_t afterName = skipSpaces(text, cursor);

    if (afterName < text.size() && text[afterName] == '=') {
      std::size_t rhs = skipSpaces(text, afterName + 1);
      if (rhs < text.size() && text[rhs] == '(') {
        const std::size_t close = findMatchingParen(text, rhs);
        if (close != std::string_view::npos)
          schemas[name] = parseAsdlFieldNames(
              std::string_view(text).substr(rhs + 1, close - rhs - 1));
      }
      cursor = afterName + 1;
      continue;
    }

    if (startsWithUppercaseIdentifier(name) && afterName < text.size() &&
        text[afterName] == '(') {
      const std::size_t close = findMatchingParen(text, afterName);
      if (close != std::string_view::npos) {
        schemas[name] = parseAsdlFieldNames(std::string_view(text).substr(
            afterName + 1, close - afterName - 1));
        cursor = close + 1;
        continue;
      }
    }
  }
  return schemas;
}

struct AstSchema {
  std::map<std::string, std::string> kindTypes;
  std::map<std::string, std::vector<AstFieldSpec>> fields;
};

std::size_t findNextAsdlDefinition(const std::string &text,
                                   std::size_t cursor) {
  while (cursor < text.size()) {
    if (cursor == 0 || text[cursor - 1] == '\n') {
      std::size_t line = cursor;
      while (line < text.size() &&
             std::isspace(static_cast<unsigned char>(text[line])) &&
             text[line] != '\n')
        ++line;
      if (line < text.size() &&
          (std::isalpha(static_cast<unsigned char>(text[line])) ||
           text[line] == '_')) {
        std::size_t nameEnd = line + 1;
        while (nameEnd < text.size() &&
               (std::isalnum(static_cast<unsigned char>(text[nameEnd])) ||
                text[nameEnd] == '_'))
          ++nameEnd;
        std::size_t afterName = skipSpaces(text, nameEnd);
        if (afterName < text.size() && text[afterName] == '=')
          return line;
      }
    }
    ++cursor;
  }
  return std::string::npos;
}

AstSchema collectAstSchema(const std::string &asdl,
                           const std::vector<std::string> &nodeKinds) {
  AstSchema schema;
  for (const std::string &kind : nodeKinds) {
    schema.kindTypes.emplace(kind, std::string());
    schema.fields.emplace(kind, std::vector<AstFieldSpec>{});
  }

  const std::string text = stripAsdlComments(asdl);
  std::size_t cursor = findNextAsdlDefinition(text, 0);
  while (cursor != std::string::npos) {
    std::size_t nameEnd = cursor + 1;
    while (nameEnd < text.size() &&
           (std::isalnum(static_cast<unsigned char>(text[nameEnd])) ||
            text[nameEnd] == '_'))
      ++nameEnd;
    std::string lhs = text.substr(cursor, nameEnd - cursor);
    std::size_t equal = skipSpaces(text, nameEnd);
    std::size_t rhsStart = skipSpaces(text, equal + 1);
    std::size_t next = findNextAsdlDefinition(text, rhsStart);
    std::string rhs = trim(text.substr(rhsStart, next - rhsStart));

    if (!rhs.empty() && rhs.front() == '(') {
      const std::size_t close = findMatchingParen(rhs, 0);
      if (close != std::string::npos) {
        schema.kindTypes[lhs] = lhs;
        schema.fields[lhs] =
            parseAsdlFieldSpecs(std::string_view(rhs).substr(1, close - 1));
      }
      cursor = next;
      continue;
    }

    for (const std::string &alternative : splitTopLevelAlternatives(rhs)) {
      std::string name = leadingIdentifier(alternative);
      if (!startsWithUppercaseIdentifier(name))
        continue;
      schema.kindTypes[name] = lhs;
      std::size_t afterName = skipSpaces(alternative, name.size());
      if (afterName < alternative.size() && alternative[afterName] == '(') {
        const std::size_t close = findMatchingParen(alternative, afterName);
        if (close != std::string::npos)
          schema.fields[name] = parseAsdlFieldSpecs(
              std::string_view(alternative)
                  .substr(afterName + 1, close - afterName - 1));
      }
    }
    cursor = next;
  }
  return schema;
}

[[maybe_unused]] CpythonSpec loadSpecFromVendoredFiles() {
  CpythonSpec spec;
  spec.root = LYTHON_CPYTHON_314_PARSER_ROOT;
  const std::string prefix =
      spec.root.empty() ? std::string() : spec.root + "/";
  const std::string tokens = readTextFile(prefix + "Tokens");
  spec.tokens = parseTokens(tokens);
  spec.tokenNames = collectTokenNames(spec.tokens);
  spec.grammar = readTextFile(prefix + "python.gram");
  spec.asdl = readTextFile(prefix + "Python.asdl");
  spec.generatedParser = readTextFile(prefix + "parser.c");
  spec.generatedRuleTypeIds = parseGeneratedRuleTypeIds(spec.generatedParser);
  spec.generatedRules =
      parseGeneratedRules(spec.generatedParser, spec.generatedRuleTypeIds);
  spec.generatedRuleIndices = collectGeneratedRuleIndices(spec.generatedRules);
  spec.peg = parseCpythonPegGrammar(spec.grammar);
  collectGrammarKeywords(spec.grammar, spec.grammarHardKeywords,
                         spec.grammarSoftKeywords);
  spec.generatedHardKeywords =
      collectGeneratedHardKeywords(spec.generatedParser);
  spec.generatedHardKeywordSpecs =
      collectGeneratedHardKeywordSpecs(spec.generatedParser);
  spec.generatedSoftKeywords =
      collectGeneratedSoftKeywords(spec.generatedParser);
  spec.generatedTokenRefs = collectGeneratedTokenRefs(spec.generatedParser);
  spec.generatedHardKeywordTokenIdByText =
      collectKeywordTokenIdByText(spec.generatedHardKeywordSpecs);
  spec.hardKeywords = spec.generatedHardKeywords;
  spec.softKeywords = spec.generatedSoftKeywords;
  spec.operatorSpellings = collectOperatorSpellings(spec.tokens);
  spec.tokenNameBySpelling = collectTokenNameBySpelling(spec.tokens);
  spec.tokenIdByName = collectTokenIdByName(spec.tokens);
  spec.tokenIdBySpelling = collectTokenIdBySpelling(spec.tokens);
  spec.astNodeKinds = collectAstNodeKinds(spec.asdl);
  spec.astFields = collectAstFieldSchemas(spec.asdl, spec.astNodeKinds);
  AstSchema astSchema = collectAstSchema(spec.asdl, spec.astNodeKinds);
  spec.astKindTypes = std::move(astSchema.kindTypes);
  spec.astFieldSpecs = std::move(astSchema.fields);
  return spec;
}

#endif

#if !defined(LYTHON_CPYTHON_SPEC_RUNTIME_FILE_IO)
#include "CpythonSpecSnapshot.inc"
#endif

CpythonSpec loadSpec() {
#if defined(LYTHON_CPYTHON_SPEC_RUNTIME_FILE_IO)
  return loadSpecFromVendoredFiles();
#else
  return loadGeneratedSpec();
#endif
}

struct StringViewPair {
  std::string_view first;
  std::string_view second;

  bool operator==(const StringViewPair &other) const {
    return first == other.first && second == other.second;
  }
};

struct StringViewPairHash {
  std::size_t operator()(const StringViewPair &pair) const {
    std::hash<std::string_view> hash;
    std::size_t seed = hash(pair.first);
    seed ^=
        hash(pair.second) + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
    return seed;
  }
};

struct CpythonSpecIndex {
  explicit CpythonSpecIndex(CpythonSpec loaded) : spec(std::move(loaded)) {
    spec.peg.buildLookupTables();
    indexStrings(spec.hardKeywords, hardKeywords);
    indexStrings(spec.softKeywords, softKeywords);
    indexStrings(spec.operatorSpellings, operatorSpellings);
    indexStrings(spec.tokenNames, tokenNames);
    indexStrings(spec.astNodeKinds, astNodeKinds);

    for (const auto &entry : spec.generatedHardKeywordTokenIdByText)
      generatedHardKeywordTokenIdByText.emplace(entry.first, entry.second);
    for (const auto &entry : spec.tokenNameBySpelling)
      tokenNameBySpelling.emplace(entry.first, entry.second);
    for (const auto &entry : spec.tokenIdByName)
      tokenIdByName.emplace(entry.first, entry.second);
    for (const auto &entry : spec.tokenIdBySpelling)
      tokenIdBySpelling.emplace(entry.first, entry.second);
    for (const auto &entry : spec.astKindTypes)
      astKindTypes.emplace(entry.first, entry.second);
    for (const auto &entry : spec.generatedRuleIndices)
      generatedRuleIndices.emplace(entry.first, entry.second);
    for (const auto &entry : spec.astFieldSpecs) {
      astFieldSpecsByKind.emplace(entry.first, &entry.second);
      for (std::size_t i = 0; i < entry.second.size(); ++i) {
        const AstFieldSpec &field = entry.second[i];
        astFieldSpecs.emplace(StringViewPair{entry.first, field.name}, &field);
        astFieldIndices.emplace(StringViewPair{entry.first, field.name}, i);
      }
    }
  }

  CpythonSpec spec;
  std::unordered_set<std::string_view> hardKeywords;
  std::unordered_set<std::string_view> softKeywords;
  std::unordered_set<std::string_view> operatorSpellings;
  std::unordered_set<std::string_view> tokenNames;
  std::unordered_set<std::string_view> astNodeKinds;
  std::unordered_map<std::string_view, int> generatedHardKeywordTokenIdByText;
  std::unordered_map<std::string_view, std::string_view> tokenNameBySpelling;
  std::unordered_map<std::string_view, int> tokenIdByName;
  std::unordered_map<std::string_view, int> tokenIdBySpelling;
  std::unordered_map<std::string_view, std::string_view> astKindTypes;
  std::unordered_map<std::string_view, std::size_t> generatedRuleIndices;
  std::unordered_map<std::string_view, const std::vector<AstFieldSpec> *>
      astFieldSpecsByKind;
  std::unordered_map<StringViewPair, const AstFieldSpec *, StringViewPairHash>
      astFieldSpecs;
  std::unordered_map<StringViewPair, std::size_t, StringViewPairHash>
      astFieldIndices;

private:
  static void indexStrings(const std::vector<std::string> &source,
                           std::unordered_set<std::string_view> &target) {
    target.reserve(source.size());
    for (const std::string &value : source)
      target.emplace(value);
  }
};

const CpythonSpecIndex &cpython314Index() {
  static const CpythonSpecIndex *index = [] {
    return new CpythonSpecIndex(loadSpec());
  }();
  return *index;
}

} // namespace

const CpythonSpec &cpython314Spec() { return cpython314Index().spec; }

bool isCpythonHardKeyword(std::string_view text) {
  const CpythonSpecIndex &index = cpython314Index();
  return index.hardKeywords.find(text) != index.hardKeywords.end();
}

bool isCpythonSoftKeyword(std::string_view text) {
  const CpythonSpecIndex &index = cpython314Index();
  return index.softKeywords.find(text) != index.softKeywords.end();
}

bool isCpythonOperator(std::string_view text) {
  const CpythonSpecIndex &index = cpython314Index();
  return index.operatorSpellings.find(text) != index.operatorSpellings.end();
}

std::optional<std::string_view>
cpythonLongestOperatorPrefix(std::string_view text) {
  const CpythonSpec &spec = cpython314Spec();
  for (const std::string &op : spec.operatorSpellings)
    if (text.size() >= op.size() && text.compare(0, op.size(), op) == 0)
      return std::string_view(op);
  return std::nullopt;
}

bool isCpythonTokenName(std::string_view text) {
  const CpythonSpecIndex &index = cpython314Index();
  return index.tokenNames.find(text) != index.tokenNames.end();
}

std::optional<std::string> cpythonTokenNameForSpelling(std::string_view text) {
  const CpythonSpecIndex &index = cpython314Index();
  auto found = index.tokenNameBySpelling.find(text);
  if (found == index.tokenNameBySpelling.end())
    return std::nullopt;
  return std::string(found->second);
}

std::optional<int> cpythonTokenIdForName(std::string_view text) {
  const CpythonSpecIndex &index = cpython314Index();
  auto found = index.tokenIdByName.find(text);
  if (found == index.tokenIdByName.end())
    return std::nullopt;
  return found->second;
}

std::optional<int> cpythonTokenIdForSpelling(std::string_view text) {
  const CpythonSpecIndex &index = cpython314Index();
  auto found = index.tokenIdBySpelling.find(text);
  if (found == index.tokenIdBySpelling.end())
    return std::nullopt;
  return found->second;
}

std::optional<int> cpythonHardKeywordTokenId(std::string_view text) {
  const CpythonSpecIndex &index = cpython314Index();
  auto found = index.generatedHardKeywordTokenIdByText.find(text);
  if (found == index.generatedHardKeywordTokenIdByText.end())
    return std::nullopt;
  return found->second;
}

bool isCpythonAstNodeKind(std::string_view text) {
  const CpythonSpecIndex &index = cpython314Index();
  return index.astNodeKinds.find(text) != index.astNodeKinds.end();
}

bool isCpythonAstField(std::string_view kind, std::string_view field) {
  const CpythonSpecIndex &index = cpython314Index();
  return index.astFieldSpecs.find(StringViewPair{kind, field}) !=
         index.astFieldSpecs.end();
}

const AstFieldSpec *cpythonAstFieldSpec(std::string_view kind,
                                        std::string_view field) {
  const CpythonSpecIndex &index = cpython314Index();
  auto found = index.astFieldSpecs.find(StringViewPair{kind, field});
  return found == index.astFieldSpecs.end() ? nullptr : found->second;
}

const std::vector<AstFieldSpec> *cpythonAstFieldSpecs(std::string_view kind) {
  const CpythonSpecIndex &index = cpython314Index();
  auto found = index.astFieldSpecsByKind.find(kind);
  if (found == index.astFieldSpecsByKind.end())
    return nullptr;
  return found->second;
}

std::optional<std::size_t> cpythonAstFieldIndex(std::string_view kind,
                                                std::string_view field) {
  const CpythonSpecIndex &index = cpython314Index();
  auto found = index.astFieldIndices.find(StringViewPair{kind, field});
  if (found == index.astFieldIndices.end())
    return std::nullopt;
  return found->second;
}

bool isCpythonAstKindOfType(std::string_view kind, std::string_view type) {
  const CpythonSpecIndex &index = cpython314Index();
  auto found = index.astKindTypes.find(kind);
  if (found == index.astKindTypes.end())
    return false;
  return found->second == type;
}

const GeneratedRuleSpec *cpythonGeneratedRule(std::string_view name) {
  const CpythonSpecIndex &index = cpython314Index();
  auto found = index.generatedRuleIndices.find(name);
  if (found == index.generatedRuleIndices.end())
    return nullptr;
  return &index.spec.generatedRules[found->second];
}

} // namespace lython::parser
