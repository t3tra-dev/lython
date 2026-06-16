#pragma once

#include "Diagnostics.h"
#include "Token.h"

#include <cstddef>
#include <string_view>
#include <vector>

namespace lython::parser {

struct LexResult {
  std::vector<Token> tokens;
  std::vector<TypeIgnoreInfo> typeIgnores;
  TypeCommentMap typeComments;
  std::size_t typeIgnoreCount = 0;
};

LexResult lexSource(std::string_view source, Diagnostics &diagnostics,
                    bool typeComments, bool preserveBlankNewlines = false);

} // namespace lython::parser
