#include "GeneratedTokenStream.h"

#include "Lexer.h"
#include "lython/parser/CpythonSpec.h"

#include <algorithm>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace lython::parser {

void GeneratedTokenStream::reserve(std::size_t size) { tokens.reserve(size); }

void GeneratedTokenStream::push(int type, std::string text, SourceRange range) {
  storage.push_back(std::move(text));
  tokens.push_back(LythonCpythonToken{type, storage.back().c_str(),
                                      range.start.line, range.start.column,
                                      range.end.line, range.end.column});
}

void GeneratedTokenStream::pushView(int type, std::string_view text,
                                    SourceRange range) {
  push(type, std::string(text), range);
}

void GeneratedTokenStream::pushToken(const Token &token) {
  push(token.cpythonId, token.text, token.range);
}

namespace {

struct GeneratedStringContentRange {
  std::size_t start = 0;
  std::size_t end = 0;
};

GeneratedStringContentRange
generatedStringContentRange(std::string_view literal) {
  const std::size_t firstQuote = literal.find_first_of("'\"");
  if (firstQuote == std::string_view::npos)
    return GeneratedStringContentRange{0, literal.size()};

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
  return GeneratedStringContentRange{contentStart, contentEnd};
}

SourceLocation generatedAdvanceLocation(SourceLocation location, char ch) {
  ++location.offset;
  if (ch == '\n') {
    ++location.line;
    location.column = 0;
  } else {
    ++location.column;
  }
  return location;
}

SourceLocation generatedLocationAt(SourceLocation start, std::string_view text,
                                   std::size_t offset) {
  SourceLocation location = start;
  const std::size_t limit = std::min(offset, text.size());
  for (std::size_t i = 0; i < limit; ++i)
    location = generatedAdvanceLocation(location, text[i]);
  return location;
}

SourceRange generatedRangeAt(SourceLocation start, std::string_view text,
                             std::size_t begin, std::size_t end) {
  return SourceRange{generatedLocationAt(start, text, begin),
                     generatedLocationAt(start, text, end)};
}

std::size_t generatedSkipQuotedText(std::string_view text,
                                    std::size_t quoteIndex) {
  char quote = text[quoteIndex];
  bool triple = quoteIndex + 2 < text.size() && text[quoteIndex + 1] == quote &&
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

std::size_t generatedSkipFStringComment(std::string_view text,
                                        std::size_t index, std::size_t limit) {
  while (index < limit && text[index] != '\n' && text[index] != '\r')
    ++index;
  if (index < limit && text[index] == '\r')
    ++index;
  if (index < limit && text[index] == '\n')
    ++index;
  return index;
}

std::size_t generatedFindFStringFieldEnd(std::string_view text,
                                         std::size_t start) {
  int depth = 0;
  for (std::size_t i = start; i < text.size();) {
    char ch = text[i];
    if (ch == '#') {
      i = generatedSkipFStringComment(text, i, text.size());
      continue;
    }
    if (ch == '\'' || ch == '"') {
      i = generatedSkipQuotedText(text, i);
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

std::size_t generatedFindFStringFieldDelimiter(std::string_view text,
                                               char delimiter) {
  int depth = 0;
  for (std::size_t i = 0; i < text.size();) {
    char ch = text[i];
    if (ch == '#') {
      i = generatedSkipFStringComment(text, i, text.size());
      continue;
    }
    if (ch == '\'' || ch == '"') {
      i = generatedSkipQuotedText(text, i);
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

std::size_t generatedFindFStringDebugEqual(std::string_view text,
                                           std::size_t limit) {
  int depth = 0;
  for (std::size_t i = 0; i < limit;) {
    char ch = text[i];
    if (ch == '#') {
      i = generatedSkipFStringComment(text, i, limit);
      continue;
    }
    if (ch == '\'' || ch == '"') {
      i = generatedSkipQuotedText(text, i);
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
          previous == '=' || previous == ':' || next == '=') {
        ++i;
      } else {
        return i;
      }
      continue;
    }
    ++i;
  }
  return std::string_view::npos;
}

std::string generatedTrimExpression(std::string_view text,
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

struct GeneratedFStringField {
  std::string expression;
  std::optional<char> conversion;
  std::optional<std::string> formatSpec;
  std::size_t expressionOffset = 0;
  std::size_t debugOffset = std::string_view::npos;
  std::size_t conversionOffset = std::string_view::npos;
  std::size_t formatDelimiterOffset = std::string_view::npos;
  std::size_t formatOffset = std::string_view::npos;
};

GeneratedFStringField generatedParseFStringField(std::string_view field) {
  GeneratedFStringField result;
  const std::size_t conversion = generatedFindFStringFieldDelimiter(field, '!');
  const std::size_t format = generatedFindFStringFieldDelimiter(field, ':');
  std::size_t expressionEnd = field.size();
  if (conversion != std::string_view::npos)
    expressionEnd = std::min(expressionEnd, conversion);
  if (format != std::string_view::npos)
    expressionEnd = std::min(expressionEnd, format);

  const std::size_t debugEqual =
      generatedFindFStringDebugEqual(field, expressionEnd);
  if (debugEqual != std::string_view::npos) {
    result.debugOffset = debugEqual;
    expressionEnd = debugEqual;
  }

  result.expression = generatedTrimExpression(field.substr(0, expressionEnd),
                                              result.expressionOffset);

  if (conversion != std::string_view::npos &&
      (format == std::string_view::npos || conversion < format)) {
    result.conversionOffset = conversion;
    if (conversion + 1 < field.size())
      result.conversion = field[conversion + 1];
    const std::size_t afterConversion = conversion + 2;
    if (afterConversion < field.size() && field[afterConversion] == ':') {
      result.formatDelimiterOffset = afterConversion;
      result.formatOffset = afterConversion + 1;
      result.formatSpec = std::string(field.substr(result.formatOffset));
    }
    return result;
  }

  if (format != std::string_view::npos) {
    result.formatDelimiterOffset = format;
    result.formatOffset = format + 1;
    result.formatSpec = std::string(field.substr(result.formatOffset));
  }
  return result;
}

bool appendGeneratedToken(const Token &token, GeneratedTokenStream &sink,
                          Diagnostics &diagnostics);

bool appendGeneratedSpelling(GeneratedTokenStream &sink,
                             std::string_view spelling, SourceRange range,
                             Diagnostics &diagnostics) {
  std::optional<int> id = cpythonTokenIdForSpelling(spelling);
  if (!id) {
    diagnostics.push_back(Diagnostic{
        Severity::Error, range.start,
        "internal parser bridge error: unknown CPython token spelling '" +
            std::string(spelling) + "'"});
    return false;
  }
  sink.pushView(*id, spelling, range);
  return true;
}

bool appendGeneratedInlineTokens(std::string_view source,
                                 GeneratedTokenStream &sink,
                                 Diagnostics &diagnostics) {
  Diagnostics nestedDiagnostics;
  LexResult lexed =
      lexSource(source, nestedDiagnostics, /*typeComments=*/false);
  std::vector<Token> nestedTokens = std::move(lexed.tokens);
  diagnostics.insert(diagnostics.end(), nestedDiagnostics.begin(),
                     nestedDiagnostics.end());
  if (!nestedDiagnostics.empty())
    return false;

  for (std::size_t i = 0; i < nestedTokens.size(); ++i) {
    const Token &token = nestedTokens[i];
    if (token.kind == TokenKind::End)
      continue;
    if (token.kind == TokenKind::Newline) {
      bool trailing = true;
      for (std::size_t lookahead = i + 1; lookahead < nestedTokens.size();
           ++lookahead) {
        if (nestedTokens[lookahead].kind == TokenKind::End)
          continue;
        trailing = false;
        break;
      }
      if (trailing)
        continue;
    }
    if (!appendGeneratedToken(token, sink, diagnostics))
      return false;
  }
  return true;
}

bool appendGeneratedFStringField(std::string_view field, int middleType,
                                 SourceLocation fieldStart,
                                 GeneratedTokenStream &sink,
                                 Diagnostics &diagnostics);

bool appendGeneratedFormatSpecTokens(std::string_view formatSpec,
                                     int middleType, SourceLocation formatStart,
                                     GeneratedTokenStream &sink,
                                     Diagnostics &diagnostics) {
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
    sink.push(middleType, std::move(chunk),
              generatedRangeAt(formatStart, formatSpec, *chunkStart, end));
    chunk.clear();
    chunkStart.reset();
  };

  for (std::size_t i = 0; i < formatSpec.size(); ++i) {
    char ch = formatSpec[i];
    if (ch == '{' && i + 1 < formatSpec.size() && formatSpec[i + 1] == '{') {
      pushChunkChar('{', i);
      ++i;
      continue;
    }
    if (ch == '}' && i + 1 < formatSpec.size() && formatSpec[i + 1] == '}') {
      pushChunkChar('}', i);
      ++i;
      continue;
    }
    if (ch == '}') {
      diagnostics.push_back(Diagnostic{
          Severity::Error, generatedLocationAt(formatStart, formatSpec, i),
          "single '}' is not allowed in formatted string"});
      return false;
    }
    if (ch != '{') {
      pushChunkChar(ch, i);
      continue;
    }

    flushChunk(i);
    const std::size_t end = generatedFindFStringFieldEnd(formatSpec, i + 1);
    if (end == std::string_view::npos) {
      diagnostics.push_back(Diagnostic{
          Severity::Error, generatedLocationAt(formatStart, formatSpec, i),
          "unterminated formatted string replacement field"});
      return false;
    }
    if (!appendGeneratedFStringField(
            formatSpec.substr(i + 1, end - i - 1), middleType,
            generatedLocationAt(formatStart, formatSpec, i + 1), sink,
            diagnostics))
      return false;
    i = end;
  }

  flushChunk(formatSpec.size());
  return true;
}

bool appendGeneratedFStringField(std::string_view field, int middleType,
                                 SourceLocation fieldStart,
                                 GeneratedTokenStream &sink,
                                 Diagnostics &diagnostics) {
  const SourceRange openRange{generatedLocationAt(fieldStart, field, 0),
                              generatedLocationAt(fieldStart, field, 0)};
  if (!appendGeneratedSpelling(sink, "{", openRange, diagnostics))
    return false;

  GeneratedFStringField parts = generatedParseFStringField(field);
  if (parts.expression.empty()) {
    diagnostics.push_back(Diagnostic{
        Severity::Error, fieldStart,
        "formatted string replacement field requires an expression"});
    return false;
  }
  if (!appendGeneratedInlineTokens(parts.expression, sink, diagnostics))
    return false;

  if (parts.debugOffset != std::string_view::npos) {
    SourceRange equalRange = generatedRangeAt(
        fieldStart, field, parts.debugOffset, parts.debugOffset + 1);
    if (!appendGeneratedSpelling(sink, "=", equalRange, diagnostics))
      return false;
  }

  if (parts.conversionOffset != std::string_view::npos) {
    SourceRange bangRange = generatedRangeAt(
        fieldStart, field, parts.conversionOffset, parts.conversionOffset + 1);
    if (!appendGeneratedSpelling(sink, "!", bangRange, diagnostics))
      return false;
    if (parts.conversion) {
      SourceRange conversionRange =
          generatedRangeAt(fieldStart, field, parts.conversionOffset + 1,
                           parts.conversionOffset + 2);
      sink.push(NAME, std::string(1, *parts.conversion), conversionRange);
    }
  }

  if (parts.formatSpec) {
    SourceRange colonRange =
        generatedRangeAt(fieldStart, field, parts.formatDelimiterOffset,
                         parts.formatDelimiterOffset + 1);
    if (!appendGeneratedSpelling(sink, ":", colonRange, diagnostics))
      return false;
    if (!appendGeneratedFormatSpecTokens(
            *parts.formatSpec, middleType,
            generatedLocationAt(fieldStart, field, parts.formatOffset), sink,
            diagnostics))
      return false;
  }

  const SourceRange closeRange{
      generatedLocationAt(fieldStart, field, field.size()),
      generatedLocationAt(fieldStart, field, field.size())};
  return appendGeneratedSpelling(sink, "}", closeRange, diagnostics);
}

bool appendGeneratedInterpolatedString(const Token &token,
                                       GeneratedTokenStream &sink,
                                       Diagnostics &diagnostics) {
  const bool templateString = literalPrefixContains(token.rawText, 't');
  const int startType = templateString ? TSTRING_START : FSTRING_START;
  const int middleType = templateString ? TSTRING_MIDDLE : FSTRING_MIDDLE;
  const int endType = templateString ? TSTRING_END : FSTRING_END;
  const GeneratedStringContentRange contentRange =
      generatedStringContentRange(token.rawText);

  sink.pushView(startType,
                std::string_view(token.rawText).substr(0, contentRange.start),
                generatedRangeAt(token.range.start, token.rawText, 0,
                                 contentRange.start));

  std::string_view content(token.rawText.data() + contentRange.start,
                           contentRange.end - contentRange.start);
  const SourceLocation contentStart =
      generatedLocationAt(token.range.start, token.rawText, contentRange.start);
  if (!appendGeneratedFormatSpecTokens(content, middleType, contentStart, sink,
                                       diagnostics))
    return false;

  sink.pushView(endType,
                std::string_view(token.rawText).substr(contentRange.end),
                generatedRangeAt(token.range.start, token.rawText,
                                 contentRange.end, token.rawText.size()));
  return true;
}

bool appendGeneratedToken(const Token &token, GeneratedTokenStream &sink,
                          Diagnostics &diagnostics) {
  if (token.kind == TokenKind::String &&
      (literalPrefixContains(token.rawText, 'f') ||
       literalPrefixContains(token.rawText, 't'))) {
    return appendGeneratedInterpolatedString(token, sink, diagnostics);
  }
  sink.pushToken(token);
  return true;
}

} // namespace

bool buildGeneratedTokenStream(const std::vector<Token> &tokens,
                               bool interactive, GeneratedTokenStream &stream,
                               Diagnostics &diagnostics) {
  stream.reserve(tokens.size() + 1);
  for (std::size_t index = 0; index < tokens.size(); ++index) {
    const Token &token = tokens[index];
    if (interactive && token.kind == TokenKind::End && index > 0 &&
        tokens[index - 1].kind == TokenKind::Dedent) {
      SourceRange range{token.range.start, token.range.start};
      stream.push(NEWLINE, "\n", range);
    }
    if (!appendGeneratedToken(token, stream, diagnostics))
      return false;
  }
  return true;
}

} // namespace lython::parser
