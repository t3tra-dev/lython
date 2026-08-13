#pragma once

#include <optional>
#include <string>
#include <string_view>

namespace lython::parser {

std::optional<std::string> cpythonUnicodeNameString(std::string_view rawName);

} // namespace lython::parser
