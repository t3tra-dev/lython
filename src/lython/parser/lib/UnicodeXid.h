#pragma once

#include <cstdint>

namespace lython::parser::unicode {

bool isXidStart(std::uint32_t codepoint);
bool isXidContinue(std::uint32_t codepoint);

} // namespace lython::parser::unicode
