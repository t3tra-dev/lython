#include "SourceEncoding.h"

#include <algorithm>
#include <cctype>
#include <optional>
#include <string>
#include <utility>

namespace lython::parser {
namespace {

struct SourceEncoding {
  std::string name;
  int line = 1;
  bool found = false;
};

struct Utf8ValidationError {
  SourceLocation location;
  std::string message;
};

bool hasUtf8Bom(std::string_view source) {
  return source.size() >= 3 && static_cast<unsigned char>(source[0]) == 0xef &&
         static_cast<unsigned char>(source[1]) == 0xbb &&
         static_cast<unsigned char>(source[2]) == 0xbf;
}

std::string normalizeEncodingName(std::string name) {
  std::string prefix;
  const std::size_t limit = std::min<std::size_t>(name.size(), 12);
  prefix.reserve(limit);
  for (std::size_t i = 0; i < limit; ++i) {
    unsigned char ch = static_cast<unsigned char>(name[i]);
    prefix.push_back(ch == '_' ? '-' : static_cast<char>(std::tolower(ch)));
  }

  if (prefix == "utf-8" || prefix.rfind("utf-8-", 0) == 0)
    return "utf-8";
  if (prefix == "ascii" || prefix == "us-ascii" || prefix == "iso646-us" ||
      prefix.rfind("ascii-", 0) == 0 || prefix.rfind("us-ascii-", 0) == 0 ||
      prefix.rfind("ansi-x3.4", 0) == 0)
    return "ascii";
  if (prefix == "latin-1" || prefix == "iso-8859-1" ||
      prefix == "iso-latin-1" || prefix.rfind("latin-1-", 0) == 0 ||
      prefix.rfind("iso-8859-1-", 0) == 0 ||
      prefix.rfind("iso-latin-1-", 0) == 0)
    return "iso-8859-1";
  return name;
}

std::optional<std::string> codingSpecOnLine(std::string_view line) {
  std::size_t cursor = 0;
  for (; cursor + 6 <= line.size(); ++cursor) {
    char ch = line[cursor];
    if (ch == '#')
      break;
    if (ch != ' ' && ch != '\t' && ch != '\f')
      return std::nullopt;
  }
  if (cursor >= line.size() || line[cursor] != '#')
    return std::nullopt;

  for (; cursor + 6 <= line.size(); ++cursor) {
    if (line.substr(cursor, 6) != "coding")
      continue;
    cursor += 6;
    if (cursor >= line.size() || (line[cursor] != ':' && line[cursor] != '='))
      continue;
    do {
      ++cursor;
    } while (cursor < line.size() &&
             (line[cursor] == ' ' || line[cursor] == '\t'));

    const std::size_t begin = cursor;
    while (cursor < line.size()) {
      unsigned char ch = static_cast<unsigned char>(line[cursor]);
      if (!std::isalnum(ch) && ch != '-' && ch != '_' && ch != '.')
        break;
      ++cursor;
    }
    if (begin < cursor)
      return normalizeEncodingName(
          std::string(line.substr(begin, cursor - begin)));
  }
  return std::nullopt;
}

bool lineKeepsCodingSearchAlive(std::string_view line) {
  for (char ch : line) {
    if (ch == '#' || ch == '\n' || ch == '\r')
      return true;
    if (ch != ' ' && ch != '\t' && ch != '\f')
      return false;
  }
  return true;
}

std::string_view physicalLineAt(std::string_view source, std::size_t start) {
  std::size_t end = start;
  while (end < source.size() && source[end] != '\n' && source[end] != '\r')
    ++end;
  return source.substr(start, end - start);
}

std::size_t nextPhysicalLineOffset(std::string_view source, std::size_t start) {
  std::size_t cursor = start;
  while (cursor < source.size() && source[cursor] != '\n' &&
         source[cursor] != '\r')
    ++cursor;
  if (cursor < source.size() && source[cursor] == '\r')
    ++cursor;
  if (cursor < source.size() && source[cursor] == '\n')
    ++cursor;
  return cursor;
}

SourceEncoding findSourceEncoding(std::string_view source) {
  std::size_t offset = hasUtf8Bom(source) ? 3 : 0;
  for (int line = 1; line <= 2 && offset <= source.size(); ++line) {
    std::string_view text = physicalLineAt(source, offset);
    if (std::optional<std::string> spec = codingSpecOnLine(text))
      return SourceEncoding{*spec, line, true};
    if (!lineKeepsCodingSearchAlive(text))
      break;
    offset = nextPhysicalLineOffset(source, offset);
  }
  return {};
}

std::string latin1ToUtf8(std::string_view source) {
  std::string decoded;
  decoded.reserve(source.size());
  for (unsigned char ch : source) {
    if (ch < 0x80) {
      decoded.push_back(static_cast<char>(ch));
      continue;
    }
    decoded.push_back(static_cast<char>(0xc0 | (ch >> 6)));
    decoded.push_back(static_cast<char>(0x80 | (ch & 0x3f)));
  }
  return decoded;
}

std::optional<Utf8ValidationError> validateUtf8Source(std::string_view source) {
  auto invalid = [](int line, int column, std::size_t offset,
                    std::string message) {
    return Utf8ValidationError{
        SourceLocation{line, column, static_cast<int>(offset)},
        std::move(message)};
  };

  int line = 1;
  int column = 0;
  std::size_t cursor = 0;
  auto advanceAscii = [&] {
    if (source[cursor] == '\n') {
      ++line;
      column = 0;
    } else {
      ++column;
    }
    ++cursor;
  };
  auto validContinuation = [&](std::size_t index) {
    return index < source.size() &&
           (static_cast<unsigned char>(source[index]) & 0xc0) == 0x80;
  };
  auto advanceCodepoint = [&](std::size_t width) {
    cursor += width;
    ++column;
  };

  while (cursor < source.size()) {
    const unsigned char first = static_cast<unsigned char>(source[cursor]);
    if (first == 0)
      return invalid(line, column, cursor,
                     "source code string cannot contain null bytes");
    if (first < 0x80) {
      advanceAscii();
      continue;
    }

    const std::size_t start = cursor;
    if (first >= 0xc2 && first <= 0xdf) {
      if (!validContinuation(cursor + 1))
        return invalid(line, column, start, "invalid UTF-8 source");
      advanceCodepoint(2);
      continue;
    }
    if (first == 0xe0) {
      if (cursor + 2 >= source.size() ||
          static_cast<unsigned char>(source[cursor + 1]) < 0xa0 ||
          static_cast<unsigned char>(source[cursor + 1]) > 0xbf ||
          !validContinuation(cursor + 2))
        return invalid(line, column, start, "invalid UTF-8 source");
      advanceCodepoint(3);
      continue;
    }
    if ((first >= 0xe1 && first <= 0xec) || (first >= 0xee && first <= 0xef)) {
      if (!validContinuation(cursor + 1) || !validContinuation(cursor + 2))
        return invalid(line, column, start, "invalid UTF-8 source");
      advanceCodepoint(3);
      continue;
    }
    if (first == 0xed) {
      if (cursor + 2 >= source.size() ||
          static_cast<unsigned char>(source[cursor + 1]) < 0x80 ||
          static_cast<unsigned char>(source[cursor + 1]) > 0x9f ||
          !validContinuation(cursor + 2))
        return invalid(line, column, start, "invalid UTF-8 source");
      advanceCodepoint(3);
      continue;
    }
    if (first == 0xf0) {
      if (cursor + 3 >= source.size() ||
          static_cast<unsigned char>(source[cursor + 1]) < 0x90 ||
          static_cast<unsigned char>(source[cursor + 1]) > 0xbf ||
          !validContinuation(cursor + 2) || !validContinuation(cursor + 3))
        return invalid(line, column, start, "invalid UTF-8 source");
      advanceCodepoint(4);
      continue;
    }
    if (first >= 0xf1 && first <= 0xf3) {
      if (!validContinuation(cursor + 1) || !validContinuation(cursor + 2) ||
          !validContinuation(cursor + 3))
        return invalid(line, column, start, "invalid UTF-8 source");
      advanceCodepoint(4);
      continue;
    }
    if (first == 0xf4) {
      if (cursor + 3 >= source.size() ||
          static_cast<unsigned char>(source[cursor + 1]) < 0x80 ||
          static_cast<unsigned char>(source[cursor + 1]) > 0x8f ||
          !validContinuation(cursor + 2) || !validContinuation(cursor + 3))
        return invalid(line, column, start, "invalid UTF-8 source");
      advanceCodepoint(4);
      continue;
    }

    return invalid(line, column, start, "invalid UTF-8 source");
  }
  return std::nullopt;
}

std::optional<Utf8ValidationError>
validateAsciiSource(std::string_view source) {
  auto invalid = [](int line, int column, std::size_t offset,
                    std::string message) {
    return Utf8ValidationError{
        SourceLocation{line, column, static_cast<int>(offset)},
        std::move(message)};
  };

  int line = 1;
  int column = 0;
  for (std::size_t cursor = 0; cursor < source.size(); ++cursor) {
    const unsigned char ch = static_cast<unsigned char>(source[cursor]);
    if (ch == 0)
      return invalid(line, column, cursor,
                     "source code string cannot contain null bytes");
    if (ch >= 0x80)
      return invalid(line, column, cursor, "encoding problem: ascii");
    if (ch == '\n') {
      ++line;
      column = 0;
    } else {
      ++column;
    }
  }
  return std::nullopt;
}

} // namespace

DecodedSource decodeSource(std::string_view source, Diagnostics &diagnostics) {
  SourceEncoding encoding = findSourceEncoding(source);
  if (hasUtf8Bom(source) && encoding.found && encoding.name != "utf-8") {
    diagnostics.push_back(
        Diagnostic{Severity::Error, SourceLocation{encoding.line, 0, 0},
                   "encoding problem: " + encoding.name + " with BOM"});
    return DecodedSource{};
  }
  if (!encoding.found || encoding.name == "utf-8") {
    if (std::optional<Utf8ValidationError> utf8Error =
            validateUtf8Source(source)) {
      diagnostics.push_back(Diagnostic{Severity::Error, utf8Error->location,
                                       std::move(utf8Error->message)});
    }
    return DecodedSource{};
  }
  if (encoding.name == "ascii") {
    if (std::optional<Utf8ValidationError> asciiError =
            validateAsciiSource(source)) {
      diagnostics.push_back(Diagnostic{Severity::Error, asciiError->location,
                                       std::move(asciiError->message)});
    }
    return DecodedSource{};
  }
  if (encoding.name == "iso-8859-1") {
    DecodedSource decoded;
    decoded.storage = latin1ToUtf8(source);
    decoded.ownsStorage = true;
    return decoded;
  }

  diagnostics.push_back(Diagnostic{Severity::Error,
                                   SourceLocation{encoding.line, 0, 0},
                                   "encoding problem: " + encoding.name});
  return DecodedSource{};
}

} // namespace lython::parser
