#include "Parser.h"

#include <gtest/gtest.h>

namespace {

using lython::parser::ParseMode;
using lython::parser::ParseOptions;
using lython::parser::ParseResult;

TEST(ParserTest, AcceptsSimpleModule) {
  ParseResult result = lython::parser::parse("print(\"hi\")\n", "<test>.py");
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result.tree->kind, "Module");
  EXPECT_NE(lython::parser::dumpAst(*result.tree, /*includeAttributes=*/false)
                .find("Call"),
            std::string::npos);
}

TEST(ParserTest, RejectsSyntaxErrorWithDiagnostics) {
  ParseResult result = lython::parser::parse("def broken(:\n", "<test>.py");
  EXPECT_FALSE(result.ok());
  ASSERT_FALSE(result.diagnostics.empty());
  EXPECT_FALSE(result.diagnostics.front().message.empty());
}

TEST(ParserTest, ExpressionMode) {
  ParseOptions options;
  options.mode = ParseMode::Expression;
  ParseResult result = lython::parser::parse("1 + 2", "<test>.py", options);
  EXPECT_TRUE(result.ok());
}

TEST(ParserTest, EmptySourceDoesNotCrash) {
  ParseResult result = lython::parser::parse("", "<test>.py");
  if (result.ok())
    EXPECT_EQ(result.tree->kind, "Module");
}

// The vendored PEG parser keeps per-parse arena and last-result globals;
// interleaved accept/reject cycles must not leak state across calls.
TEST(ParserTest, RepeatedParseIsStable) {
  for (int round = 0; round < 50; ++round) {
    ParseResult good = lython::parser::parse("x = 1\ny = x + 2\n", "<a>.py");
    EXPECT_TRUE(good.ok()) << "round " << round;
    ParseResult bad = lython::parser::parse("def broken(:\n", "<b>.py");
    EXPECT_FALSE(bad.ok()) << "round " << round;
  }
}

// `with (a) as b:` is one item whose expression is parenthesized, and
// `with (a as b):` is a parenthesized item list: what follows the `)` tells
// them apart.
TEST(ParserTest, AParenthesizedWithItemTakesItsTarget) {
  auto items = [](const char *source) {
    ParseResult result = lython::parser::parse(source, "<test>.py");
    EXPECT_TRUE(result.ok()) << source;
    return result.ok() ? lython::parser::dumpAst(*result.tree,
                                                 /*includeAttributes=*/false)
                       : std::string();
  };
  const std::string one = items("with (a) as cur:\n    pass\n");
  EXPECT_NE(one.find("context_expr=Name(id=\"a\""), std::string::npos) << one;
  EXPECT_NE(one.find("optional_vars=Name(id=\"cur\""), std::string::npos)
      << one;
  const std::string tuple = items("with (a, b) as c:\n    pass\n");
  EXPECT_NE(tuple.find("context_expr=Tuple("), std::string::npos) << tuple;
  const std::string subscript = items("with (a)[0] as c:\n    pass\n");
  EXPECT_NE(subscript.find("context_expr=Subscript("), std::string::npos)
      << subscript;
  const std::string listed = items("with (a as b, c as d):\n    pass\n");
  EXPECT_NE(listed.find("optional_vars=Name(id=\"d\""), std::string::npos)
      << listed;
}

TEST(ParserTest, FStringTokens) {
  ParseResult result =
      lython::parser::parse("name = \"w\"\nprint(f\"hello {name}!\")\n",
                            "<test>.py");
  EXPECT_TRUE(result.ok());
}

} // namespace
