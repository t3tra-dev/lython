#pragma once

#include <string>
#include <string_view>

namespace lython::parser::unicode {

std::string normalizeIdentifierNfkc(std::string_view utf8);

} // namespace lython::parser::unicode
