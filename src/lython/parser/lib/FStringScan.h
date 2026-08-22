#pragma once

// Locating the pieces of a string or f-string INSIDE its source text: where
// the quotes end, where a field ends, where the conversion and format-spec
// delimiters are, and what source location each offset corresponds to.
//
// ⭐ ONE JOB, TWO CALLERS, AND THEY MUST NOT DRIFT. The hand-written parser
// (Parser.cpp) builds AST nodes from these offsets; the generated-token stream
// (GeneratedTokenStream.cpp) re-derives them to hand CPython's own parser the
// same slices. Both carried a full copy of this family -- nine functions, the
// second set spelled `generated*` -- identical to the character. A divergence
// between the two copies is a divergence between what this parser accepts and
// what the generated one is told, which is the single thing the two-parser
// arrangement exists to keep aligned, so it cannot be left to two edits
// staying in step.

#include "Ast.h"

#include <cstddef>
#include <string>
#include <string_view>

namespace lython::parser {

// The content between the quotes of a string literal, as offsets into it.
struct StringContentRange {
  std::size_t start = 0;
  std::size_t end = 0;
};

StringContentRange stringContentRange(std::string_view literal);

// Source locations over the literal's text: one character, an offset, a span.
SourceLocation advanceLocation(SourceLocation location, char ch);
SourceLocation locationAt(SourceLocation start, std::string_view text,
                          std::size_t offset);
SourceRange rangeAt(SourceLocation start, std::string_view text,
                    std::size_t begin, std::size_t end);

// Past the string that starts at `quoteIndex` (single or triple quoted), or
// the end of `text` when it is unterminated.
std::size_t skipQuotedText(std::string_view text, std::size_t quoteIndex);

// Past the `#` comment starting at `index`, including its newline.
std::size_t skipFStringComment(std::string_view text, std::size_t index,
                               std::size_t limit);

// The `}` closing the replacement field that starts at `start`, npos when the
// field is unterminated. Nested braces, bracket groups, quoted text and
// comments are skipped; after the top-level `:` the rest is a format spec,
// where `#` and quotes are ordinary characters.
std::size_t findFStringFieldEnd(std::string_view text, std::size_t start);

// The top-level `!` or `:` inside a field, npos when there is none.
std::size_t findFStringFieldDelimiter(std::string_view text, char delimiter);

// The `=` of a debug field (`f"{x=}"`) before `limit`, npos when there is
// none.
std::size_t findFStringDebugEqual(std::string_view text, std::size_t limit);

// The field's expression with surrounding whitespace removed; `offset`
// receives how far in it started.
std::string trimInterpolationExpression(std::string_view text,
                                        std::size_t &offset);

} // namespace lython::parser
