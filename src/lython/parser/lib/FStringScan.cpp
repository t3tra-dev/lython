#include "FStringScan.h"

#include <algorithm>

namespace lython::parser {

StringContentRange
stringContentRange(std::string_view literal) {
  const std::size_t firstQuote = literal.find_first_of("'\"");
  if (firstQuote == std::string_view::npos)
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

SourceLocation advanceLocation(SourceLocation location, char ch) {
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
                                   std::size_t offset) {
  SourceLocation location = start;
  const std::size_t limit = std::min(offset, text.size());
  for (std::size_t i = 0; i < limit; ++i)
    location = advanceLocation(location, text[i]);
  return location;
}

SourceRange rangeAt(SourceLocation start, std::string_view text,
                             std::size_t begin, std::size_t end) {
  return SourceRange{locationAt(start, text, begin),
                     locationAt(start, text, end)};
}

std::size_t skipQuotedText(std::string_view text,
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

std::size_t skipFStringComment(std::string_view text,
                                        std::size_t index, std::size_t limit) {
  while (index < limit && text[index] != '\n' && text[index] != '\r')
    ++index;
  if (index < limit && text[index] == '\r')
    ++index;
  if (index < limit && text[index] == '\n')
    ++index;
  return index;
}

std::size_t findFStringFieldEnd(std::string_view text,
                                         std::size_t start) {
  int depth = 0;
  // After the top-level ':' the remainder is the format spec: literal text
  // where '#' and quotes are ordinary characters ('{x:#x}', '{x:"^7}').
  // Bracket depth keeps slice colons ('{a[1:2]}') from starting it early.
  int groupDepth = 0;
  bool inSpec = false;
  for (std::size_t i = start; i < text.size();) {
    char ch = text[i];
    if (ch == '#' && !inSpec) {
      i = skipFStringComment(text, i, text.size());
      continue;
    }
    if ((ch == '\'' || ch == '"') && !inSpec) {
      i = skipQuotedText(text, i);
      continue;
    }
    if ((ch == '(' || ch == '[') && !inSpec) {
      ++groupDepth;
      ++i;
      continue;
    }
    if ((ch == ')' || ch == ']') && !inSpec) {
      --groupDepth;
      ++i;
      continue;
    }
    if (ch == ':' && depth == 0 && groupDepth == 0) {
      inSpec = true;
      ++i;
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

std::size_t findFStringFieldDelimiter(std::string_view text,
                                               char delimiter) {
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
    // A top-level ':' starts the format spec; a '!' beyond it would be
    // spec text, never a conversion marker.
    if (depth == 0 && ch == ':' && delimiter == '!')
      return std::string_view::npos;
    ++i;
  }
  return std::string_view::npos;
}

std::size_t findFStringDebugEqual(std::string_view text,
                                           std::size_t limit) {
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

} // namespace lython::parser
