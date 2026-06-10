#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace lython::parser {

std::optional<std::uint32_t>
cpythonUnicodeNameCodepoint(std::string_view rawName);
std::optional<std::string> cpythonUnicodeNameString(std::string_view rawName);

} // namespace lython::parser
