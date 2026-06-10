#include "UnicodeNames.h"

#include <cctype>
#include <cstddef>
#include <string>

namespace lython::parser {
namespace {

using Py_UCS2 = std::uint16_t;
using Py_UCS4 = std::uint32_t;

#include "../unicodename_db.h"

std::string canonicalUnicodeName(std::string_view name) {
  std::string canonical;
  canonical.reserve(name.size());
  for (char ch : name) {
    unsigned char typed = static_cast<unsigned char>(ch);
    canonical.push_back(static_cast<char>(std::toupper(typed)));
  }
  return canonical;
}

bool startsWith(std::string_view text, std::string_view prefix) {
  return text.size() >= prefix.size() &&
         text.substr(0, prefix.size()) == prefix;
}

std::optional<std::uint32_t> parseHexCode(std::string_view suffix) {
  if (suffix.size() < 4 || suffix.size() > 6)
    return std::nullopt;
  if (suffix.front() == '0')
    return std::nullopt;

  std::uint32_t value = 0;
  for (char ch : suffix) {
    value *= 16;
    if (ch >= '0' && ch <= '9') {
      value += static_cast<std::uint32_t>(ch - '0');
    } else if (ch >= 'A' && ch <= 'F') {
      value += static_cast<std::uint32_t>(ch - 'A' + 10);
    } else {
      return std::nullopt;
    }
  }
  if (value > 0x10ffff)
    return std::nullopt;
  return value;
}

int derivedPrefixId(std::uint32_t codepoint) {
  struct Range {
    std::uint32_t first;
    std::uint32_t last;
    int prefixId;
  };

  constexpr Range ranges[] = {
      {0x3400, 0x4dbf, 1},   {0x4e00, 0x9fff, 1},   {0xac00, 0xd7a3, 0},
      {0x17000, 0x187f7, 2}, {0x18d00, 0x18d08, 2}, {0x20000, 0x2a6df, 1},
      {0x2a700, 0x2b739, 1}, {0x2b740, 0x2b81d, 1}, {0x2b820, 0x2cea1, 1},
      {0x2ceb0, 0x2ebe0, 1}, {0x2ebf0, 0x2ee5d, 1}, {0x30000, 0x3134a, 1},
      {0x31350, 0x323af, 1},
  };

  for (const Range &range : ranges) {
    if (codepoint < range.first)
      return -1;
    if (codepoint <= range.last)
      return range.prefixId;
  }
  return -1;
}

std::optional<std::uint32_t>
generatedIdeographNameCodepoint(std::string_view name, std::string_view prefix,
                                int prefixId) {
  if (!startsWith(name, prefix))
    return std::nullopt;
  std::optional<std::uint32_t> codepoint =
      parseHexCode(name.substr(prefix.size()));
  if (!codepoint || derivedPrefixId(*codepoint) != prefixId)
    return std::nullopt;
  return codepoint;
}

template <std::size_t Size>
bool matchHangulPart(std::string_view name, std::size_t offset,
                     const std::string_view (&parts)[Size], int &index,
                     std::size_t &length) {
  bool found = false;
  index = -1;
  length = 0;
  for (std::size_t i = 0; i < Size; ++i) {
    std::string_view candidate = parts[i];
    if (found && candidate.size() <= length)
      continue;
    if (offset + candidate.size() > name.size())
      continue;
    if (name.substr(offset, candidate.size()) != candidate)
      continue;
    found = true;
    index = static_cast<int>(i);
    length = candidate.size();
  }
  return found;
}

std::optional<std::uint32_t> hangulNameCodepoint(std::string_view name) {
  constexpr std::string_view prefix = "HANGUL SYLLABLE ";
  if (!startsWith(name, prefix))
    return std::nullopt;

  constexpr std::string_view leading[] = {"G", "GG", "N", "D",  "DD", "R", "M",
                                          "B", "BB", "S", "SS", "",   "J", "JJ",
                                          "C", "K",  "T", "P",  "H"};
  constexpr std::string_view vowel[] = {
      "A",  "AE", "YA", "YAE", "EO", "E",  "YEO", "YE", "O",  "WA", "WAE",
      "OE", "YO", "U",  "WEO", "WE", "WI", "YU",  "EU", "YI", "I"};
  constexpr std::string_view trailing[] = {
      "",   "G",  "GG", "GS", "N",  "NJ", "NH", "D", "L",  "LG",
      "LM", "LB", "LS", "LT", "LP", "LH", "M",  "B", "BS", "S",
      "SS", "NG", "J",  "C",  "K",  "T",  "P",  "H"};

  int leadingIndex = -1;
  int vowelIndex = -1;
  int trailingIndex = -1;
  std::size_t length = 0;
  std::size_t offset = prefix.size();
  if (!matchHangulPart(name, offset, leading, leadingIndex, length))
    return std::nullopt;
  offset += length;
  if (!matchHangulPart(name, offset, vowel, vowelIndex, length))
    return std::nullopt;
  offset += length;
  if (!matchHangulPart(name, offset, trailing, trailingIndex, length))
    return std::nullopt;
  offset += length;
  if (offset != name.size())
    return std::nullopt;

  constexpr std::uint32_t sBase = 0xac00;
  constexpr int vowelCount = 21;
  constexpr int trailingCount = 28;
  return sBase + static_cast<std::uint32_t>(
                     (leadingIndex * vowelCount + vowelIndex) * trailingCount +
                     trailingIndex);
}

std::optional<std::uint32_t> derivedNameCodepoint(std::string_view name) {
  if (auto codepoint = hangulNameCodepoint(name))
    return codepoint;
  if (auto codepoint =
          generatedIdeographNameCodepoint(name, "CJK UNIFIED IDEOGRAPH-", 1))
    return codepoint;
  if (auto codepoint =
          generatedIdeographNameCodepoint(name, "TANGUT IDEOGRAPH-", 2))
    return codepoint;
  return std::nullopt;
}

bool appendUtf8(std::string &out, std::uint32_t codepoint) {
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

std::size_t dawgDecodeVarint(std::size_t index, unsigned int &result) {
  result = 0;
  unsigned int shift = 0;
  for (;;) {
    unsigned char byte = packed_name_dawg[index++];
    result |= static_cast<unsigned int>(byte & 0x7f) << shift;
    shift += 7;
    if ((byte & 0x80) == 0)
      return index;
  }
}

bool dawgMatchEdge(std::string_view name, unsigned int size,
                   unsigned int labelOffset, unsigned int namePos, int &match) {
  match = 0;
  if (size > 1 && namePos + size > name.size())
    return true;
  for (unsigned int i = 0; i < size; ++i) {
    unsigned char typed = static_cast<unsigned char>(name[namePos + i]);
    if (packed_name_dawg[labelOffset + i] != std::toupper(typed)) {
      match = i > 0 ? -1 : 0;
      return true;
    }
  }
  match = 1;
  return true;
}

std::size_t dawgDecodeNode(std::size_t nodeOffset, bool &final) {
  unsigned int node = 0;
  nodeOffset = dawgDecodeVarint(nodeOffset, node);
  final = (node & 1) != 0;
  return nodeOffset;
}

bool dawgNodeIsFinal(std::size_t nodeOffset) {
  unsigned int node = 0;
  dawgDecodeVarint(nodeOffset, node);
  return (node & 1) != 0;
}

unsigned int dawgNodeDescendantCount(std::size_t nodeOffset) {
  unsigned int node = 0;
  dawgDecodeVarint(nodeOffset, node);
  return node >> 1;
}

int dawgDecodeEdge(bool isFirstEdge, unsigned int previousTargetNodeOffset,
                   unsigned int edgeOffset, unsigned int &size,
                   unsigned int &labelOffset, unsigned int &targetNodeOffset) {
  unsigned int edge = 0;
  edgeOffset = static_cast<unsigned int>(dawgDecodeVarint(edgeOffset, edge));
  if (edge == 0 && isFirstEdge)
    return -1;

  bool lastEdge = (edge & 1) != 0;
  edge >>= 1;
  bool lengthIsOne = (edge & 1) != 0;
  edge >>= 1;
  targetNodeOffset = previousTargetNodeOffset + edge;
  if (lengthIsOne) {
    size = 1;
  } else {
    size = packed_name_dawg[edgeOffset++];
  }
  labelOffset = edgeOffset;
  return lastEdge ? 1 : 0;
}

std::optional<unsigned int> dawgLookupPosition(std::string_view name) {
  unsigned int stringPos = 0;
  unsigned int nodeOffset = 0;
  unsigned int result = 0;
  while (stringPos < name.size()) {
    bool final = false;
    unsigned int edgeOffset =
        static_cast<unsigned int>(dawgDecodeNode(nodeOffset, final));
    unsigned int previousTargetNodeOffset = edgeOffset;
    bool isFirstEdge = true;
    for (;;) {
      unsigned int size = 0;
      unsigned int labelOffset = 0;
      unsigned int targetNodeOffset = 0;
      int lastEdge =
          dawgDecodeEdge(isFirstEdge, previousTargetNodeOffset, edgeOffset,
                         size, labelOffset, targetNodeOffset);
      if (lastEdge == -1)
        return std::nullopt;

      isFirstEdge = false;
      previousTargetNodeOffset = targetNodeOffset;
      int matched = 0;
      dawgMatchEdge(name, size, labelOffset, stringPos, matched);
      if (matched == -1)
        return std::nullopt;
      if (matched == 1) {
        if (final)
          result += 1;
        stringPos += size;
        nodeOffset = targetNodeOffset;
        break;
      }
      if (lastEdge)
        return std::nullopt;
      result += dawgNodeDescendantCount(targetNodeOffset);
      edgeOffset = labelOffset + size;
    }
  }
  if (dawgNodeIsFinal(nodeOffset))
    return result;
  return std::nullopt;
}

std::optional<std::uint32_t> dawgNameCodepoint(std::string_view name) {
  std::optional<unsigned int> position = dawgLookupPosition(name);
  if (!position)
    return std::nullopt;
  std::uint32_t codepoint = dawg_pos_to_codepoint[*position];
  if (codepoint >= aliases_start && codepoint < aliases_end)
    codepoint = name_aliases[codepoint - aliases_start];
  if (codepoint >= named_sequences_start && codepoint < named_sequences_end)
    return std::nullopt;
  if (codepoint > 0x10ffff)
    return std::nullopt;
  return codepoint;
}

std::optional<std::string> dawgNameString(std::string_view name) {
  std::optional<unsigned int> position = dawgLookupPosition(name);
  if (!position)
    return std::nullopt;
  std::uint32_t codepoint = dawg_pos_to_codepoint[*position];
  std::string result;
  if (codepoint >= named_sequences_start && codepoint < named_sequences_end) {
    const named_sequence &sequence =
        named_sequences[codepoint - named_sequences_start];
    for (int index = 0; index < sequence.seqlen; ++index) {
      if (!appendUtf8(result, sequence.seq[index]))
        return std::nullopt;
    }
    return result;
  }
  if (codepoint >= aliases_start && codepoint < aliases_end)
    codepoint = name_aliases[codepoint - aliases_start];
  if (codepoint > 0x10ffff)
    return std::nullopt;
  if (!appendUtf8(result, codepoint))
    return std::nullopt;
  return result;
}

} // namespace

std::optional<std::uint32_t>
cpythonUnicodeNameCodepoint(std::string_view rawName) {
  std::string name = canonicalUnicodeName(rawName);
  if (auto codepoint = derivedNameCodepoint(name))
    return codepoint;
  if (auto codepoint = dawgNameCodepoint(name))
    return codepoint;
  return std::nullopt;
}

std::optional<std::string> cpythonUnicodeNameString(std::string_view rawName) {
  std::string name = canonicalUnicodeName(rawName);
  if (auto codepoint = derivedNameCodepoint(name)) {
    std::string result;
    if (!appendUtf8(result, *codepoint))
      return std::nullopt;
    return result;
  }
  if (auto value = dawgNameString(name))
    return value;
  return std::nullopt;
}

} // namespace lython::parser
