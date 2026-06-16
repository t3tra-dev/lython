#pragma once

#include "Diagnostics.h"

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

} // namespace lython::parser
