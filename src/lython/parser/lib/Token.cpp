#include "Token.h"

#include <cctype>

namespace lython::parser {

bool literalPrefixContains(std::string_view literal, char marker) {
  for (char ch : literal) {
    if (ch == '\'' || ch == '"')
      return false;
    if (std::tolower(static_cast<unsigned char>(ch)) == marker)
      return true;
  }
  return false;
}

} // namespace lython::parser
