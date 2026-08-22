#pragma once

#include "Diagnostics.h"

#include <cstdint>
#include <string>
#include <string_view>

namespace lython::parser {

struct DecodedSource {
  std::string storage;
  bool ownsStorage = false;

  std::string_view view(std::string_view original) const {
    if (ownsStorage)
      return storage;
    return original;
  }
};

DecodedSource decodeSource(std::string_view source, Diagnostics &diagnostics);

// The UTF-8 encoding of `codepoint`, appended to `out`. False, with nothing
// appended, for a surrogate: it has no UTF-8 form, and encoding one anyway
// produces bytes no decoder in this parser accepts.
//
// Here because this module is what says the rest of the parser is looking at
// UTF-8. Four copies had grown -- the lexer's, the string-escape decoder's,
// the \N{...} name lookup's and the NFKC pass's -- of which two skipped the
// surrogate check.
bool appendUtf8(std::string &out, std::uint32_t codepoint);

} // namespace lython::parser
