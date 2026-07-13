#include "Parser.h"

#include <cstddef>
#include <cstdint>
#include <string_view>

// Exercises the vendored CPython PEG parser and the C++ AST builder. The
// contract under test: arbitrary bytes never crash, and every accepted tree
// survives dumpAst.
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  if (size > (64u << 10))
    return 0;
  std::string_view source(reinterpret_cast<const char *>(data), size);
  lython::parser::ParseResult result =
      lython::parser::parse(source, "<fuzz>.py");
  if (result.ok())
    (void)lython::parser::dumpAst(*result.tree, /*includeAttributes=*/true);
  return 0;
}
