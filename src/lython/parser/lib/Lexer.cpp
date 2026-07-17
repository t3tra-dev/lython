#include "Lexer.h"

#include "CpythonSpec.h"
#include "UnicodeNormalize.h"
#include "UnicodeXid.h"

#include <algorithm>
#include <cctype>
#include <optional>
#include <string>
#include <utility>

namespace lython::parser {
namespace {

class Lexer {
public:
  Lexer(std::string_view source, Diagnostics &diagnostics, bool typeComments,
        bool preserveBlankNewlines = false)
      : source(source), diagnostics(diagnostics),
        collectTypeComments(typeComments),
        preserveBlankNewlines(preserveBlankNewlines) {}

  std::vector<Token> lex() {
    consumeInitialUtf8Bom();
    while (!eof() && !stopLexing)
      lexPhysicalLine();
    while (indentStack.size() > 1)
      emitDedent(currentLocation());
    emit(TokenKind::End, "", currentLocation(), currentLocation());
    return std::move(tokens);
  }

  std::vector<TypeIgnoreInfo> takeTypeIgnores() {
    return std::move(typeIgnores);
  }

  std::size_t typeIgnoreCount() const { return typeIgnores.size(); }

  TypeCommentMap takeTypeComments() { return std::move(typeCommentByLine); }

private:
  std::string_view source;
  Diagnostics &diagnostics;
  bool collectTypeComments = false;
  bool preserveBlankNewlines = false;
  std::vector<Token> tokens;
  std::vector<TypeIgnoreInfo> typeIgnores;
  TypeCommentMap typeCommentByLine;
  std::vector<int> indentStack{0};
  std::vector<int> altIndentStack{0};
  std::size_t offset = 0;
  int line = 1;
  int column = 0;
  int parenDepth = 0;
  bool stopLexing = false;

  bool eof() const { return offset >= source.size(); }

  void consumeInitialUtf8Bom() {
    if (offset != 0 || source.size() < 3)
      return;
    if (static_cast<unsigned char>(source[0]) != 0xef ||
        static_cast<unsigned char>(source[1]) != 0xbb ||
        static_cast<unsigned char>(source[2]) != 0xbf)
      return;
    offset = 3;
    line = 1;
    column = 0;
  }

  char peek(std::size_t lookahead = 0) const {
    if (offset + lookahead >= source.size())
      return '\0';
    return source[offset + lookahead];
  }

  static bool isAsciiNameStart(char ch) {
    unsigned char typed = static_cast<unsigned char>(ch);
    return std::isalpha(typed) || ch == '_';
  }

  static bool isAsciiNameContinue(char ch) {
    unsigned char typed = static_cast<unsigned char>(ch);
    return std::isalnum(typed) || ch == '_';
  }

  static bool isUtf8Continuation(unsigned char byte) {
    return (byte & 0xc0) == 0x80;
  }

  std::size_t utf8CodepointLengthAt(std::size_t cursor) const {
    if (cursor >= source.size())
      return 0;
    unsigned char first = static_cast<unsigned char>(source[cursor]);
    if (first < 0x80)
      return 1;

    std::size_t length = 0;
    if ((first & 0xe0) == 0xc0)
      length = 2;
    else if ((first & 0xf0) == 0xe0)
      length = 3;
    else if ((first & 0xf8) == 0xf0)
      length = 4;
    else
      return 0;

    if (cursor + length > source.size())
      return 0;
    for (std::size_t i = 1; i < length; ++i)
      if (!isUtf8Continuation(static_cast<unsigned char>(source[cursor + i])))
        return 0;
    return length;
  }

  std::optional<std::uint32_t> decodeUtf8CodepointAt(std::size_t cursor) const {
    std::size_t length = utf8CodepointLengthAt(cursor);
    if (length == 0 || cursor + length > source.size())
      return std::nullopt;
    unsigned char first = static_cast<unsigned char>(source[cursor]);
    if (length == 1)
      return first;

    std::uint32_t value = 0;
    if (length == 2)
      value = first & 0x1f;
    else if (length == 3)
      value = first & 0x0f;
    else
      value = first & 0x07;
    for (std::size_t i = 1; i < length; ++i) {
      value <<= 6;
      value |= static_cast<unsigned char>(source[cursor + i]) & 0x3f;
    }
    return value;
  }

  static bool inRange(std::uint32_t value, std::uint32_t first,
                      std::uint32_t last) {
    return value >= first && value <= last;
  }

  bool isNameStartAt(std::size_t cursor) const {
    if (cursor >= source.size())
      return false;
    char ch = source[cursor];
    if (static_cast<unsigned char>(ch) < 0x80)
      return isAsciiNameStart(ch);
    std::optional<std::uint32_t> codepoint = decodeUtf8CodepointAt(cursor);
    return codepoint && unicode::isXidStart(*codepoint);
  }

  bool isNameContinueAt(std::size_t cursor) const {
    if (cursor >= source.size())
      return false;
    char ch = source[cursor];
    if (static_cast<unsigned char>(ch) < 0x80)
      return isAsciiNameContinue(ch);
    std::optional<std::uint32_t> codepoint = decodeUtf8CodepointAt(cursor);
    return codepoint && unicode::isXidContinue(*codepoint);
  }

  static void appendUtf8(std::string &text, std::uint32_t value) {
    if (value <= 0x7f) {
      text.push_back(static_cast<char>(value));
      return;
    }
    if (value <= 0x7ff) {
      text.push_back(static_cast<char>(0xc0 | (value >> 6)));
      text.push_back(static_cast<char>(0x80 | (value & 0x3f)));
      return;
    }
    if (value <= 0xffff) {
      text.push_back(static_cast<char>(0xe0 | (value >> 12)));
      text.push_back(static_cast<char>(0x80 | ((value >> 6) & 0x3f)));
      text.push_back(static_cast<char>(0x80 | (value & 0x3f)));
      return;
    }
    text.push_back(static_cast<char>(0xf0 | (value >> 18)));
    text.push_back(static_cast<char>(0x80 | ((value >> 12) & 0x3f)));
    text.push_back(static_cast<char>(0x80 | ((value >> 6) & 0x3f)));
    text.push_back(static_cast<char>(0x80 | (value & 0x3f)));
  }

  void appendNameCodepoint(std::string &rawText) {
    std::size_t length = utf8CodepointLengthAt(offset);
    if (length == 0)
      length = 1;
    for (std::size_t i = 0; i < length; ++i)
      rawText.push_back(advance());
  }

  SourceLocation currentLocation() const {
    return SourceLocation{line, column, static_cast<int>(offset)};
  }

  char advance() {
    char ch = peek();
    if (ch == '\0')
      return ch;
    ++offset;
    if (ch == '\n') {
      ++line;
      column = 0;
    } else {
      ++column;
    }
    return ch;
  }

  void emit(TokenKind kind, std::string text, SourceLocation start,
            SourceLocation end) {
    std::string rawText = text;
    emit(kind, std::move(text), std::move(rawText), start, end);
  }

  void emit(TokenKind kind, std::string text, std::string rawText,
            SourceLocation start, SourceLocation end) {
    std::string cpythonName = cpythonTokenName(kind, rawText);
    int cpythonId = cpythonTokenId(kind, rawText, cpythonName);
    tokens.push_back(Token{kind, std::move(text), std::move(rawText),
                           std::move(cpythonName), cpythonId,
                           SourceRange{start, end}});
  }

  std::string cpythonTokenName(TokenKind kind, std::string_view text) const {
    switch (kind) {
    case TokenKind::Name:
    case TokenKind::Keyword:
      return "NAME";
    case TokenKind::Number:
      return "NUMBER";
    case TokenKind::String:
      if (literalPrefixContains(text, 'f'))
        return "FSTRING_START";
      if (literalPrefixContains(text, 't'))
        return "TSTRING_START";
      return "STRING";
    case TokenKind::Op:
      if (std::optional<std::string> name = cpythonTokenNameForSpelling(text))
        return *name;
      return "OP";
    case TokenKind::Newline:
      return "NEWLINE";
    case TokenKind::Indent:
      return "INDENT";
    case TokenKind::Dedent:
      return "DEDENT";
    case TokenKind::End:
      return "ENDMARKER";
    case TokenKind::Invalid:
      return "ERRORTOKEN";
    }
    return "ERRORTOKEN";
  }

  int cpythonTokenId(TokenKind kind, std::string_view text,
                     std::string_view cpythonName) const {
    if (kind == TokenKind::Keyword)
      if (std::optional<int> id = cpythonHardKeywordTokenId(text))
        return *id;
    if (kind == TokenKind::Op)
      if (std::optional<int> id = cpythonTokenIdForSpelling(text))
        return *id;
    if (std::optional<int> id = cpythonTokenIdForName(cpythonName))
      return *id;
    return -1;
  }

  void error(SourceLocation location, std::string message) {
    diagnostics.push_back(
        Diagnostic{Severity::Error, location, std::move(message)});
  }

  void lexPhysicalLine() {
    int indent = 0;
    int altIndent = 0;
    int continuationIndent = 0;
    SourceLocation lineStart = currentLocation();
    for (;;) {
      char leading = peek();
      if (leading == '\f') {
        indent = 0;
        altIndent = 0;
        advance();
        continue;
      }
      if (leading == '\t') {
        indent += 8 - (indent % 8);
        ++altIndent;
        advance();
        continue;
      }
      if (leading == ' ') {
        ++indent;
        ++altIndent;
        advance();
        continue;
      }
      if (leading == '\\') {
        if (continuationIndent == 0)
          continuationIndent = indent;
        if (!consumeLineContinuation()) {
          error(currentLocation(), "unexpected character after line "
                                   "continuation character");
          break;
        }
        continue;
      }
      break;
    }
    if (continuationIndent != 0) {
      indent = continuationIndent;
      altIndent = continuationIndent;
    }

    if (peek() == '\n' || peek() == '\r' || peek() == '#' || eof()) {
      if (peek() == '#')
        collectTypeComment(/*standaloneLine=*/true);
      const bool blankPhysicalLine =
          preserveBlankNewlines && (peek() == '\n' || peek() == '\r');
      skipUntilLineEnd();
      if (blankPhysicalLine)
        emit(TokenKind::Newline, "\n", currentLocation(), currentLocation());
      return;
    }

    if (parenDepth == 0)
      handleIndent(indent, altIndent, lineStart);

    while (!eof()) {
      char ch = peek();
      if (ch == '\n' || ch == '\r')
        break;
      if (consumeLineContinuation())
        continue;
      if (ch == '#') {
        collectTypeComment(/*standaloneLine=*/false);
        while (!eof() && peek() != '\n' && peek() != '\r')
          advance();
        break;
      }
      if (std::isspace(static_cast<unsigned char>(ch))) {
        advance();
        continue;
      }
      if (isNameStartAt(offset)) {
        lexNameOrStringPrefix();
        continue;
      }
      if (std::isdigit(static_cast<unsigned char>(ch))) {
        lexNumber();
        continue;
      }
      if (ch == '.' && std::isdigit(static_cast<unsigned char>(peek(1)))) {
        lexNumber();
        continue;
      }
      if (ch == '\'' || ch == '"') {
        lexString(/*prefix=*/"");
        continue;
      }
      if (static_cast<unsigned char>(ch) >= 0x80) {
        lexInvalidCodepoint();
        continue;
      }
      lexOperator();
    }

    if (!eof())
      consumeLineEnd();
    if (parenDepth == 0)
      emit(TokenKind::Newline, "\n", currentLocation(), currentLocation());
  }

  void handleIndent(int indent, int altIndent, SourceLocation lineStart) {
    if (indent > indentStack.back()) {
      if (altIndent <= altIndentStack.back())
        error(lineStart, "inconsistent use of tabs and spaces in indentation");
      indentStack.push_back(indent);
      altIndentStack.push_back(altIndent);
      emit(TokenKind::Indent, "", lineStart, lineStart);
      return;
    }
    while (indent < indentStack.back())
      emitDedent(lineStart);
    if (indent != indentStack.back()) {
      error(lineStart, "inconsistent indentation");
      return;
    }
    if (altIndent != altIndentStack.back())
      error(lineStart, "inconsistent use of tabs and spaces in indentation");
  }

  void emitDedent(SourceLocation location) {
    if (indentStack.size() > 1) {
      indentStack.pop_back();
      altIndentStack.pop_back();
    }
    emit(TokenKind::Dedent, "", location, location);
  }

  void skipUntilLineEnd() {
    while (!eof() && peek() != '\n' && peek() != '\r')
      advance();
    if (!eof())
      consumeLineEnd();
  }

  void consumeLineEnd() {
    if (peek() == '\r')
      advance();
    if (peek() == '\n')
      advance();
  }

  std::size_t lineEndOffset(std::size_t cursor) const {
    while (cursor < source.size() && source[cursor] != '\n' &&
           source[cursor] != '\r')
      ++cursor;
    return cursor;
  }

  static bool asciiAlphaNum(char ch) {
    return std::isalnum(static_cast<unsigned char>(ch)) != 0;
  }

  bool consumeFlexibleSpace(std::size_t &cursor) const {
    bool consumed = false;
    while (cursor < source.size() &&
           (source[cursor] == ' ' || source[cursor] == '\t')) {
      consumed = true;
      ++cursor;
    }
    return consumed;
  }

  bool consumeLiteral(std::size_t &cursor, std::string_view literal) const {
    if (source.substr(cursor, literal.size()) != literal)
      return false;
    cursor += literal.size();
    return true;
  }

  void collectTypeComment(bool standaloneLine) {
    if (!collectTypeComments || peek() != '#')
      return;

    const SourceLocation start = currentLocation();
    std::size_t cursor = offset;
    if (!consumeLiteral(cursor, "#"))
      return;
    consumeFlexibleSpace(cursor);
    if (!consumeLiteral(cursor, "type:"))
      return;
    consumeFlexibleSpace(cursor);

    const std::size_t payloadStart = cursor;
    const std::size_t payloadEnd = lineEndOffset(cursor);
    const std::string payload(
        source.substr(payloadStart, payloadEnd - payloadStart));

    const bool isTypeIgnore =
        payload.size() >= 6 && payload.compare(0, 6, "ignore") == 0 &&
        (payload.size() == 6 || (static_cast<unsigned char>(payload[6]) < 128 &&
                                 !asciiAlphaNum(payload[6])));

    if (isTypeIgnore) {
      const std::size_t tagStart = payloadStart + 6;
      const std::size_t tagEnd = payloadEnd;
      std::string tag(source.substr(tagStart, tagEnd - tagStart));
      if (standaloneLine)
        tag.push_back('\n');
      typeIgnores.push_back(TypeIgnoreInfo{
          start.line, std::move(tag),
          SourceRange{start,
                      SourceLocation{line, column, static_cast<int>(tagEnd)}}});
      return;
    }

    typeCommentByLine[start.line] = TypeCommentInfo{
        start.line, payload,
        SourceRange{
            start,
            SourceLocation{line, column + static_cast<int>(payloadEnd - offset),
                           static_cast<int>(payloadEnd)}}};
  }

  bool consumeLineContinuation() {
    if (peek() != '\\')
      return false;
    if (peek(1) == '\r' && peek(2) == '\n') {
      advance();
      advance();
      advance();
      return true;
    }
    if (peek(1) == '\n') {
      advance();
      advance();
      return true;
    }
    return false;
  }

  static bool isAcceptedStringPrefix(std::string_view lowered) {
    return lowered == "r" || lowered == "u" || lowered == "b" ||
           lowered == "f" || lowered == "t" || lowered == "br" ||
           lowered == "rb" || lowered == "fr" || lowered == "rf" ||
           lowered == "tr" || lowered == "rt";
  }

  static bool looksLikeInvalidStringPrefix(std::string_view lowered) {
    if (lowered.empty())
      return false;
    for (char ch : lowered)
      if (ch != 'r' && ch != 'u' && ch != 'b' && ch != 'f' && ch != 't')
        return false;
    return !isAcceptedStringPrefix(lowered);
  }

  void lexNameOrStringPrefix() {
    SourceLocation start = currentLocation();
    std::string rawText;
    while (isNameContinueAt(offset))
      appendNameCodepoint(rawText);

    std::string lowered = rawText;
    std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                   [](unsigned char ch) { return std::tolower(ch); });
    if (isAcceptedStringPrefix(lowered) && (peek() == '\'' || peek() == '"')) {
      lexString(rawText, start);
      return;
    }
    if ((peek() == '\'' || peek() == '"') &&
        looksLikeInvalidStringPrefix(lowered))
      error(start, "invalid string literal prefix '" + rawText + "'");

    TokenKind kind =
        isCpythonHardKeyword(rawText) ? TokenKind::Keyword : TokenKind::Name;
    std::string normalizedText = kind == TokenKind::Name
                                     ? unicode::normalizeIdentifierNfkc(rawText)
                                     : rawText;
    emit(kind, std::move(normalizedText), std::move(rawText), start,
         currentLocation());
  }

  void lexNumber() {
    SourceLocation start = currentLocation();
    std::string text;

    auto consumeDigits = [&](auto validDigit) {
      while (validDigit(peek()) || peek() == '_')
        text.push_back(advance());
    };

    if (peek() == '0' && (peek(1) == 'b' || peek(1) == 'B' || peek(1) == 'o' ||
                          peek(1) == 'O' || peek(1) == 'x' || peek(1) == 'X')) {
      text.push_back(advance());
      text.push_back(advance());
      while (std::isalnum(static_cast<unsigned char>(peek())) || peek() == '_')
        text.push_back(advance());
      if (peek() == 'j' || peek() == 'J')
        text.push_back(advance());
      emit(TokenKind::Number, std::move(text), start, currentLocation());
      return;
    }

    consumeDigits(
        [](char ch) { return std::isdigit(static_cast<unsigned char>(ch)); });
    if (peek() == '.') {
      text.push_back(advance());
      consumeDigits(
          [](char ch) { return std::isdigit(static_cast<unsigned char>(ch)); });
    }
    if (peek() == 'e' || peek() == 'E') {
      text.push_back(advance());
      if (peek() == '+' || peek() == '-')
        text.push_back(advance());
      consumeDigits(
          [](char ch) { return std::isdigit(static_cast<unsigned char>(ch)); });
    }
    if (peek() == 'j' || peek() == 'J')
      text.push_back(advance());
    if (isNameStartAt(offset)) {
      error(start, "invalid decimal literal");
      return;
    }
    emit(TokenKind::Number, std::move(text), start, currentLocation());
  }

  void appendEscapedStringChar(std::string &text) {
    if (eof())
      return;
    char escaped = advance();
    text.push_back(escaped);
    if (escaped == '\r' && peek() == '\n')
      text.push_back(advance());
  }

  void consumeQuotedTextInFormattedField(std::string &text, char quote) {
    bool triple = peek() == quote && peek(1) == quote;
    if (triple) {
      text.push_back(advance());
      text.push_back(advance());
    }
    while (!eof()) {
      char ch = advance();
      text.push_back(ch);
      if (ch == '\\') {
        appendEscapedStringChar(text);
        continue;
      }
      if (ch != quote)
        continue;
      if (triple) {
        if (peek() == quote && peek(1) == quote) {
          text.push_back(advance());
          text.push_back(advance());
          return;
        }
        continue;
      }
      return;
    }
  }

  void consumeFormattedFieldComment(std::string &text) {
    while (!eof() && peek() != '\n' && peek() != '\r')
      text.push_back(advance());
    if (peek() == '\r') {
      text.push_back(advance());
      if (peek() == '\n')
        text.push_back(advance());
      return;
    }
    if (peek() == '\n')
      text.push_back(advance());
  }

  void lexString(std::string prefix,
                 std::optional<SourceLocation> tokenStart = std::nullopt) {
    SourceLocation start = tokenStart.value_or(currentLocation());
    char quote = advance();
    bool triple = peek() == quote && peek(1) == quote;
    const bool formattedString = literalPrefixContains(prefix, 'f') ||
                                 literalPrefixContains(prefix, 't');
    std::string text = std::move(prefix);
    text.push_back(quote);
    if (triple) {
      text.push_back(advance());
      text.push_back(advance());
    }

    bool closed = false;
    bool rejectedNewline = false;
    int replacementDepth = 0;
    // Format-spec state of the outermost replacement field: after its
    // top-level ':' the remainder is literal spec text, where '#' and
    // quotes are ordinary characters ('{x:#x}', '{x:"^7}'). Bracket depth
    // keeps slice colons ('{a[1:2]}') from starting the spec early.
    bool fieldSawColon = false;
    int fieldGroupDepth = 0;
    while (!eof()) {
      if (!triple && replacementDepth == 0 &&
          (peek() == '\n' || peek() == '\r')) {
        rejectedNewline = true;
        stopLexing = true;
        error(start, "unterminated string literal");
        break;
      }
      char ch = advance();
      text.push_back(ch);

      if (formattedString && replacementDepth > 0) {
        bool inSpec = fieldSawColon && replacementDepth == 1;
        if (ch == ':' && replacementDepth == 1 && fieldGroupDepth == 0)
          fieldSawColon = true;
        if ((ch == '\'' || ch == '"') && !inSpec) {
          consumeQuotedTextInFormattedField(text, ch);
          continue;
        }
        // Spec text treats the other quote kind as literal, but the outer
        // quote still terminates the literal exactly like CPython (only a
        // full triple closes a triple-quoted f-string).
        if (inSpec && ch == quote) {
          if (!triple) {
            closed = true;
            break;
          }
          if (peek() == quote && peek(1) == quote) {
            text.push_back(advance());
            text.push_back(advance());
            closed = true;
            break;
          }
          continue;
        }
        if (ch == '\\') {
          appendEscapedStringChar(text);
          continue;
        }
        if (ch == '#' && !inSpec) {
          consumeFormattedFieldComment(text);
          continue;
        }
        if ((ch == '(' || ch == '[') && !inSpec) {
          ++fieldGroupDepth;
          continue;
        }
        if ((ch == ')' || ch == ']') && !inSpec) {
          --fieldGroupDepth;
          continue;
        }
        if (ch == '{') {
          ++replacementDepth;
          continue;
        }
        if (ch == '}') {
          --replacementDepth;
          if (replacementDepth == 0) {
            fieldSawColon = false;
            fieldGroupDepth = 0;
          }
          continue;
        }
        continue;
      }

      if (ch == '\\') {
        appendEscapedStringChar(text);
        continue;
      }
      if (formattedString && ch == '{') {
        if (peek() == '{') {
          text.push_back(advance());
          continue;
        }
        replacementDepth = 1;
        continue;
      }
      if (formattedString && ch == '}' && peek() == '}') {
        text.push_back(advance());
        continue;
      }
      if (ch != quote)
        continue;
      if (triple) {
        if (peek() == quote && peek(1) == quote) {
          text.push_back(advance());
          text.push_back(advance());
          closed = true;
          break;
        }
      } else {
        closed = true;
        break;
      }
    }
    if (!closed && !rejectedNewline) {
      stopLexing = true;
      error(start, "unterminated string literal");
    }
    emit(TokenKind::String, std::move(text), start, currentLocation());
  }

  void lexOperator() {
    SourceLocation start = currentLocation();
    if (std::optional<std::string_view> op =
            cpythonLongestOperatorPrefix(source.substr(offset))) {
      for (std::size_t i = 0; i < op->size(); ++i)
        advance();
      updateParenDepth(*op);
      emit(TokenKind::Op, std::string(*op), start, currentLocation());
      return;
    }

    char ch = advance();
    error(start, "invalid character '" + std::string(1, ch) + "'");
    emit(TokenKind::Invalid, std::string(1, ch), start, currentLocation());
  }

  void lexInvalidCodepoint() {
    SourceLocation start = currentLocation();
    std::size_t length = utf8CodepointLengthAt(offset);
    if (length == 0)
      length = 1;
    std::string text;
    text.reserve(length);
    for (std::size_t i = 0; i < length; ++i)
      text.push_back(advance());
    error(start, "invalid character '" + text + "'");
    emit(TokenKind::Invalid, text, start, currentLocation());
  }

  void updateParenDepth(std::string_view op) {
    if (op == "(" || op == "[" || op == "{")
      ++parenDepth;
    if ((op == ")" || op == "]" || op == "}") && parenDepth > 0)
      --parenDepth;
  }
};

} // namespace

LexResult lexSource(std::string_view source, Diagnostics &diagnostics,
                    bool typeComments, bool preserveBlankNewlines) {
  Lexer lexer(source, diagnostics, typeComments, preserveBlankNewlines);
  LexResult result;
  result.tokens = lexer.lex();
  result.typeIgnoreCount = lexer.typeIgnoreCount();
  result.typeIgnores = lexer.takeTypeIgnores();
  result.typeComments = lexer.takeTypeComments();
  return result;
}

} // namespace lython::parser
