#include "CpythonContract.h"
#include "CpythonSpec.h"

#include <algorithm>
#include <initializer_list>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <string_view>

namespace lython::parser {
namespace {

void report(Diagnostics &diagnostics, std::string message) {
  diagnostics.push_back(
      Diagnostic{Severity::Error, SourceLocation{}, std::move(message)});
}

bool contains(const std::vector<std::string> &values, std::string_view value) {
  return std::find(values.begin(), values.end(), value) != values.end();
}

std::string join(const std::vector<std::string> &values) {
  std::string result;
  for (const std::string &value : values) {
    if (!result.empty())
      result += ", ";
    result += value;
  }
  return result;
}

std::vector<std::string> setDifference(const std::vector<std::string> &lhs,
                                       const std::vector<std::string> &rhs) {
  std::vector<std::string> result;
  for (const std::string &value : lhs) {
    if (!contains(rhs, value))
      result.push_back(value);
  }
  return result;
}

bool alternativeHasRuleRef(const CpythonPegAlternative &alternative,
                           std::string_view ruleRef) {
  return contains(alternative.ruleRefs, ruleRef);
}

bool alternativeHasLiteralRef(const CpythonPegAlternative &alternative,
                              std::string_view literal) {
  return contains(alternative.literalRefs, literal);
}

bool alternativeHasTokenRef(const CpythonPegAlternative &alternative,
                            std::string_view token) {
  return contains(alternative.tokenRefs, token);
}

bool ruleHasHead(const CpythonPegRule &rule, std::string_view head) {
  return std::any_of(rule.alternatives.begin(), rule.alternatives.end(),
                     [&](const CpythonPegAlternative &alternative) {
                       return alternative.headRule == head;
                     });
}

bool ruleReferences(const CpythonPegRule &rule, std::string_view ruleRef) {
  return std::any_of(rule.alternatives.begin(), rule.alternatives.end(),
                     [&](const CpythonPegAlternative &alternative) {
                       return alternativeHasRuleRef(alternative, ruleRef);
                     });
}

bool ruleReferencesLiteral(const CpythonPegRule &rule,
                           std::string_view literal) {
  return std::any_of(rule.alternatives.begin(), rule.alternatives.end(),
                     [&](const CpythonPegAlternative &alternative) {
                       return alternativeHasLiteralRef(alternative, literal);
                     });
}

bool ruleReferencesToken(const CpythonPegRule &rule, std::string_view token) {
  return std::any_of(rule.alternatives.begin(), rule.alternatives.end(),
                     [&](const CpythonPegAlternative &alternative) {
                       return alternativeHasTokenRef(alternative, token);
                     });
}

void collectLabels(const CpythonPegExpr &expr, std::set<std::string> &labels) {
  if (!expr.label.empty())
    labels.insert(expr.label);
  for (const CpythonPegExpr &child : expr.children)
    collectLabels(child, labels);
}

bool alternativeCalls(const CpythonPegAlternative &alternative,
                      std::string_view callee) {
  return contains(alternative.actionCallees, callee);
}

bool hasLabels(const CpythonPegAlternative &alternative,
               std::initializer_list<std::string_view> labels) {
  std::set<std::string> actual;
  collectLabels(alternative.expr, actual);
  for (std::string_view label : labels) {
    if (actual.find(std::string(label)) == actual.end())
      return false;
  }
  return true;
}

const GeneratedRuleSpec *findGeneratedRule(const CpythonSpec &spec,
                                           std::string_view ruleName) {
  auto found = spec.generatedRuleIndices.find(std::string(ruleName));
  if (found == spec.generatedRuleIndices.end())
    return nullptr;
  return &spec.generatedRules[found->second];
}

const CpythonPegRule *requireRule(Diagnostics &diagnostics,
                                  const CpythonSpec &spec,
                                  std::string_view ruleName) {
  const CpythonPegRule *rule = spec.peg.findRule(ruleName);
  if (!rule) {
    report(diagnostics,
           "vendored CPython 3.14 PEG grammar is missing required rule: " +
               std::string(ruleName));
    return nullptr;
  }
  if (!findGeneratedRule(spec, ruleName)) {
    report(diagnostics,
           "vendored CPython 3.14 generated parser.c is missing required "
           "rule function: " +
               std::string(ruleName) + "_rule");
    return nullptr;
  }
  return rule;
}

void requireActionLabels(Diagnostics &diagnostics, const CpythonSpec &spec,
                         std::string_view ruleName, std::string_view callee,
                         std::initializer_list<std::string_view> labels);

void validateGeneratedParserContract(Diagnostics &diagnostics,
                                     const CpythonSpec &spec) {
  if (spec.generatedRules.empty()) {
    report(diagnostics,
           "vendored CPython 3.14 generated parser.c exposed no rule "
           "declarations");
    return;
  }

  for (std::string_view ruleName :
       {"file",          "interactive", "eval",
        "func_type",     "statement",   "simple_stmt",
        "compound_stmt", "assignment",  "function_def_raw",
        "class_def_raw", "if_stmt",     "while_stmt",
        "for_stmt",      "with_stmt",   "try_stmt",
        "match_stmt",    "expression",  "disjunction",
        "comparison",    "primary",     "atom",
        "arguments",     "patterns"}) {
    (void)requireRule(diagnostics, spec, ruleName);
  }

  std::set<int> seenTypeIds;
  for (const CpythonPegRule &rule : spec.peg.rules) {
    const GeneratedRuleSpec *generated = findGeneratedRule(spec, rule.name);
    if (!generated) {
      report(diagnostics,
             "vendored CPython 3.14 generated parser.c is missing rule "
             "function for grammar rule: " +
                 rule.name);
      continue;
    }
    if (generated->typeId < 0) {
      report(diagnostics,
             "vendored CPython 3.14 generated parser.c is missing rule type "
             "id for grammar rule: " +
                 rule.name);
      continue;
    }
    if (!seenTypeIds.insert(generated->typeId).second) {
      report(diagnostics,
             "vendored CPython 3.14 generated parser.c has duplicate rule "
             "type id: " +
                 std::to_string(generated->typeId));
    }
  }
}

void validateActionLabelContract(Diagnostics &diagnostics,
                                 const CpythonSpec &spec) {
  requireActionLabels(diagnostics, spec, "return_stmt", "_PyAST_Return", {"a"});
  requireActionLabels(diagnostics, spec, "raise_stmt", "_PyAST_Raise",
                      {"a", "b"});
  requireActionLabels(diagnostics, spec, "assignment", "_PyAST_AnnAssign",
                      {"a", "b", "c"});
  requireActionLabels(diagnostics, spec, "function_def_raw",
                      "_PyAST_FunctionDef",
                      {"n", "t", "params", "a", "tc", "b"});
  requireActionLabels(diagnostics, spec, "function_def_raw",
                      "_PyAST_AsyncFunctionDef",
                      {"n", "t", "params", "a", "tc", "b"});
  requireActionLabels(diagnostics, spec, "class_def_raw", "_PyAST_ClassDef",
                      {"a", "t", "b", "c"});
  requireActionLabels(diagnostics, spec, "if_stmt", "_PyAST_If",
                      {"a", "b", "c"});
  requireActionLabels(diagnostics, spec, "elif_stmt", "_PyAST_If",
                      {"a", "b", "c"});
  requireActionLabels(diagnostics, spec, "while_stmt", "_PyAST_While",
                      {"a", "b", "c"});
  requireActionLabels(diagnostics, spec, "for_stmt", "_PyAST_For",
                      {"t", "ex", "tc", "b", "el"});
  requireActionLabels(diagnostics, spec, "for_stmt", "_PyAST_AsyncFor",
                      {"t", "ex", "tc", "b", "el"});
  requireActionLabels(diagnostics, spec, "with_stmt", "_PyAST_With",
                      {"a", "b"});
  requireActionLabels(diagnostics, spec, "with_stmt", "_PyAST_AsyncWith",
                      {"a", "b"});
  requireActionLabels(diagnostics, spec, "lambdef", "_PyAST_Lambda",
                      {"a", "b"});
  requireActionLabels(diagnostics, spec, "primary", "_PyAST_Attribute",
                      {"a", "b"});
  requireActionLabels(diagnostics, spec, "primary", "_PyAST_Call", {"a", "b"});
  requireActionLabels(diagnostics, spec, "primary", "_PyAST_Subscript",
                      {"a", "b"});
  requireActionLabels(diagnostics, spec, "star_expression", "_PyAST_Starred",
                      {"a"});
  requireActionLabels(diagnostics, spec, "for_if_clause",
                      "_PyAST_comprehension", {"a", "b", "c"});
}

void requireActionLabels(Diagnostics &diagnostics, const CpythonSpec &spec,
                         std::string_view ruleName, std::string_view callee,
                         std::initializer_list<std::string_view> labels) {
  const CpythonPegRule *rule = requireRule(diagnostics, spec, ruleName);
  if (!rule)
    return;
  for (const CpythonPegAlternative &alternative : rule->alternatives) {
    if (!alternativeCalls(alternative, callee))
      continue;
    if (hasLabels(alternative, labels))
      return;
  }
  report(diagnostics, "vendored CPython 3.14 PEG grammar rule '" +
                          std::string(ruleName) +
                          "' no longer exposes labels required by action " +
                          std::string(callee));
}

void requireActionReferences(Diagnostics &diagnostics, const CpythonSpec &spec,
                             std::string_view ruleName,
                             std::initializer_list<std::string_view> refs) {
  const CpythonPegRule *rule = requireRule(diagnostics, spec, ruleName);
  if (!rule)
    return;
  for (std::string_view ref : refs) {
    if (rule->callsAction(ref))
      continue;
    report(diagnostics,
           "vendored CPython 3.14 PEG grammar rule '" + std::string(ruleName) +
               "' no longer calls expected AST action: " + std::string(ref));
  }
}

void requireSingleAstConstructor(Diagnostics &diagnostics,
                                 const CpythonSpec &spec,
                                 std::string_view ruleName,
                                 std::string_view constructor) {
  const CpythonPegRule *rule = requireRule(diagnostics, spec, ruleName);
  if (!rule)
    return;
  std::optional<std::string> actual = rule->singleAstConstructor();
  if (actual && *actual == constructor)
    return;

  std::string found = actual ? *actual : "<none-or-multiple>";
  report(diagnostics, "vendored CPython 3.14 PEG grammar rule '" +
                          std::string(ruleName) +
                          "' no longer has expected single AST constructor " +
                          std::string(constructor) + "; found " + found);
}

void requireAstConstructors(Diagnostics &diagnostics, const CpythonSpec &spec,
                            std::string_view ruleName,
                            std::initializer_list<std::string_view> names) {
  const CpythonPegRule *rule = requireRule(diagnostics, spec, ruleName);
  if (!rule)
    return;
  for (std::string_view name : names) {
    if (rule->hasAstConstructor(name))
      continue;
    report(
        diagnostics,
        "vendored CPython 3.14 PEG grammar rule '" + std::string(ruleName) +
            "' no longer has expected AST constructor: " + std::string(name));
  }
}

void requireNonEmptyFirstSet(Diagnostics &diagnostics,
                             const CpythonPegRule &rule) {
  if (!rule.first.literals.empty() || !rule.first.tokens.empty())
    return;
  report(diagnostics,
         "vendored CPython 3.14 PEG grammar produced an empty FIRST set for "
         "required rule: " +
             rule.name);
}

void requireAlternativeHeads(Diagnostics &diagnostics, const CpythonSpec &spec,
                             std::string_view ruleName,
                             std::initializer_list<std::string_view> heads) {
  const CpythonPegRule *rule = requireRule(diagnostics, spec, ruleName);
  if (!rule)
    return;
  for (std::string_view head : heads) {
    if (ruleHasHead(*rule, head))
      continue;
    report(diagnostics,
           "vendored CPython 3.14 PEG grammar rule '" + std::string(ruleName) +
               "' is missing expected alternative head: " + std::string(head));
  }
}

void requireRuleReferences(Diagnostics &diagnostics, const CpythonSpec &spec,
                           std::string_view ruleName,
                           std::initializer_list<std::string_view> refs) {
  const CpythonPegRule *rule = requireRule(diagnostics, spec, ruleName);
  if (!rule)
    return;
  for (std::string_view ref : refs) {
    if (ruleReferences(*rule, ref))
      continue;
    report(diagnostics,
           "vendored CPython 3.14 PEG grammar rule '" + std::string(ruleName) +
               "' no longer references expected rule: " + std::string(ref));
  }
}

void requireLiteralReferences(Diagnostics &diagnostics, const CpythonSpec &spec,
                              std::string_view ruleName,
                              std::initializer_list<std::string_view> refs) {
  const CpythonPegRule *rule = requireRule(diagnostics, spec, ruleName);
  if (!rule)
    return;
  for (std::string_view ref : refs) {
    if (ruleReferencesLiteral(*rule, ref))
      continue;
    report(diagnostics,
           "vendored CPython 3.14 PEG grammar rule '" + std::string(ruleName) +
               "' no longer references expected literal: " + std::string(ref));
  }
}

void requireTokenReferences(Diagnostics &diagnostics, const CpythonSpec &spec,
                            std::string_view ruleName,
                            std::initializer_list<std::string_view> refs) {
  const CpythonPegRule *rule = requireRule(diagnostics, spec, ruleName);
  if (!rule)
    return;
  for (std::string_view ref : refs) {
    if (ruleReferencesToken(*rule, ref))
      continue;
    report(diagnostics,
           "vendored CPython 3.14 PEG grammar rule '" + std::string(ruleName) +
               "' no longer references expected token: " + std::string(ref));
  }
}

void requireWithTypeCommentShape(Diagnostics &diagnostics,
                                 const CpythonSpec &spec) {
  const CpythonPegRule *rule = requireRule(diagnostics, spec, "with_stmt");
  if (!rule)
    return;

  bool sawAsyncParenthesized = false;
  bool sawAsyncUnparenthesized = false;
  for (const CpythonPegAlternative &alternative : rule->alternatives) {
    if (!alternativeCalls(alternative, "_PyAST_AsyncWith"))
      continue;

    const bool parenthesized = alternativeHasLiteralRef(alternative, "(");
    const bool hasTypeComment =
        alternativeHasTokenRef(alternative, "TYPE_COMMENT");
    if (parenthesized) {
      sawAsyncParenthesized = true;
      if (hasTypeComment) {
        report(diagnostics,
               "vendored CPython 3.14 async parenthesized with_stmt "
               "unexpectedly accepts TYPE_COMMENT");
      }
      continue;
    }

    sawAsyncUnparenthesized = true;
    if (!hasTypeComment) {
      report(diagnostics,
             "vendored CPython 3.14 async unparenthesized with_stmt no longer "
             "accepts TYPE_COMMENT");
    }
  }

  if (!sawAsyncParenthesized)
    report(diagnostics,
           "vendored CPython 3.14 with_stmt is missing async parenthesized "
           "alternative");
  if (!sawAsyncUnparenthesized)
    report(diagnostics, "vendored CPython 3.14 with_stmt is missing async "
                        "unparenthesized alternative");
}

void validateTokenContract(Diagnostics &diagnostics, const CpythonSpec &spec) {
  for (std::string_view requiredToken :
       {"ENDMARKER", "NAME", "NUMBER", "STRING", "NEWLINE", "INDENT", "DEDENT",
        "FSTRING_START", "TSTRING_START", "ERRORTOKEN"}) {
    if (isCpythonTokenName(requiredToken))
      continue;
    report(diagnostics,
           "vendored CPython 3.14 token spec is missing required token: " +
               std::string(requiredToken));
  }

  for (const std::string &op : spec.operatorSpellings) {
    if (cpythonTokenNameForSpelling(op))
      continue;
    report(diagnostics,
           "vendored CPython 3.14 token spec has operator spelling without a "
           "token name: " +
               op);
  }

  if (spec.generatedTokenRefs.empty()) {
    report(diagnostics,
           "vendored CPython 3.14 generated parser.c exposed no literal token "
           "references");
    return;
  }

  std::map<std::string, int> generatedIds;
  for (const GeneratedTokenRefSpec &ref : spec.generatedTokenRefs) {
    auto seen = generatedIds.emplace(ref.text, ref.tokenId);
    if (!seen.second && seen.first->second != ref.tokenId) {
      report(diagnostics,
             "vendored CPython 3.14 generated parser.c references literal "
             "token with inconsistent ids: " +
                 ref.text);
      continue;
    }

    auto expected = spec.tokenIdBySpelling.find(ref.text);
    if (expected == spec.tokenIdBySpelling.end())
      continue;
    if (expected->second == ref.tokenId)
      continue;
    report(diagnostics,
           "vendored CPython 3.14 generated parser.c token id disagrees with "
           "Tokens for literal '" +
               ref.text + "': parser.c=" + std::to_string(ref.tokenId) +
               ", Tokens=" + std::to_string(expected->second));
  }
}

void validateKeywordContract(Diagnostics &diagnostics,
                             const CpythonSpec &spec) {
  if (spec.generatedHardKeywords.empty()) {
    report(diagnostics,
           "vendored CPython 3.14 generated parser.c exposed no hard keyword "
           "table entries");
  }
  if (spec.generatedSoftKeywords.empty()) {
    report(diagnostics,
           "vendored CPython 3.14 generated parser.c exposed no soft keyword "
           "table entries");
  }

  auto reportDifference = [&](std::string_view label,
                              const std::vector<std::string> &lhs,
                              const std::vector<std::string> &rhs) {
    std::vector<std::string> missing = setDifference(lhs, rhs);
    if (!missing.empty()) {
      report(diagnostics,
             "vendored CPython 3.14 " + std::string(label) +
                 " keyword tables disagree; missing from generated parser.c: " +
                 join(missing));
    }
    std::vector<std::string> extra = setDifference(rhs, lhs);
    if (!extra.empty()) {
      report(diagnostics,
             "vendored CPython 3.14 " + std::string(label) +
                 " keyword tables disagree; missing from python.gram: " +
                 join(extra));
    }
  };

  reportDifference("hard", spec.grammarHardKeywords,
                   spec.generatedHardKeywords);
  reportDifference("soft", spec.grammarSoftKeywords,
                   spec.generatedSoftKeywords);

  std::vector<std::string> generatedSpecTexts;
  std::map<std::string, int> hardKeywordIds;
  std::set<int> tokenIds;
  for (const GeneratedKeywordSpec &keyword : spec.generatedHardKeywordSpecs) {
    generatedSpecTexts.push_back(keyword.text);
    if (keyword.tokenId < 0) {
      report(diagnostics,
             "vendored CPython 3.14 generated parser.c has hard keyword "
             "without token id: " +
                 keyword.text);
      continue;
    }
    if (!tokenIds.insert(keyword.tokenId).second) {
      report(diagnostics,
             "vendored CPython 3.14 generated parser.c has duplicate keyword "
             "token id: " +
                 std::to_string(keyword.tokenId));
    }
    hardKeywordIds.emplace(keyword.text, keyword.tokenId);
  }
  std::sort(generatedSpecTexts.begin(), generatedSpecTexts.end());
  reportDifference("hard generated", spec.generatedHardKeywords,
                   generatedSpecTexts);

  for (const GeneratedTokenRefSpec &ref : spec.generatedTokenRefs) {
    auto expected = hardKeywordIds.find(ref.text);
    if (expected == hardKeywordIds.end())
      continue;
    if (expected->second == ref.tokenId)
      continue;
    report(diagnostics,
           "vendored CPython 3.14 generated parser.c keyword token id "
           "disagrees between keyword table and expect_token for '" +
               ref.text +
               "': keyword table=" + std::to_string(expected->second) +
               ", expect_token=" + std::to_string(ref.tokenId));
  }
}

void validateFirstSetContract(Diagnostics &diagnostics,
                              const CpythonSpec &spec) {
  auto validateFirstSetTokens = [&](std::string_view ruleName,
                                    const CpythonPegFirstSet &first) {
    for (const std::string &token : first.tokens) {
      if (isCpythonTokenName(token))
        continue;
      report(diagnostics, "vendored CPython 3.14 PEG grammar rule '" +
                              std::string(ruleName) +
                              "' references unknown token: " + token);
    }
  };

  for (const CpythonPegRule &rule : spec.peg.rules) {
    validateFirstSetTokens(rule.name, rule.first);
    validateFirstSetTokens(rule.name, rule.recoveryFirst);
    for (const CpythonPegAlternative &alternative : rule.alternatives) {
      validateFirstSetTokens(rule.name, alternative.first);
      validateFirstSetTokens(rule.name, alternative.recoveryFirst);
    }
  }
}

void validateActionContract(Diagnostics &diagnostics, const CpythonSpec &spec) {
  for (const CpythonPegRule &rule : spec.peg.rules) {
    for (const CpythonPegAlternative &alternative : rule.alternatives) {
      for (const std::string &constructor : alternative.astConstructors) {
        if (isCpythonAstNodeKind(constructor))
          continue;
        report(diagnostics,
               "vendored CPython 3.14 PEG grammar rule '" + rule.name +
                   "' calls _PyAST_" + constructor +
                   ", but that constructor is not present in Python.asdl");
      }
    }
  }
}

void validateRequiredEntryRules(Diagnostics &diagnostics,
                                const CpythonSpec &spec) {
  for (std::string_view ruleName :
       {"file", "interactive", "eval", "func_type", "statement",
        "statement_newline", "simple_stmt", "compound_stmt", "expressions"}) {
    const CpythonPegRule *rule = requireRule(diagnostics, spec, ruleName);
    if (!rule)
      continue;
    if (ruleName == "statement" || ruleName == "simple_stmt" ||
        ruleName == "compound_stmt" || ruleName == "expressions")
      requireNonEmptyFirstSet(diagnostics, *rule);
  }

  requireAlternativeHeads(diagnostics, spec, "statement",
                          {"compound_stmt", "simple_stmts"});
  requireAlternativeHeads(diagnostics, spec, "statement_newline",
                          {"single_compound_stmt", "simple_stmts"});
  requireAlternativeHeads(diagnostics, spec, "simple_stmt",
                          {"assignment", "type_alias", "star_expressions",
                           "return_stmt", "import_stmt", "raise_stmt",
                           "pass_stmt", "del_stmt", "yield_stmt", "assert_stmt",
                           "break_stmt", "continue_stmt", "global_stmt",
                           "nonlocal_stmt"});
  requireAstConstructors(diagnostics, spec, "simple_stmt", {"Expr"});
  requireAstConstructors(diagnostics, spec, "assignment",
                         {"AnnAssign", "Assign", "AugAssign"});
  requireAlternativeHeads(diagnostics, spec, "compound_stmt",
                          {"function_def", "if_stmt", "class_def", "with_stmt",
                           "for_stmt", "try_stmt", "while_stmt", "match_stmt"});

  requireActionReferences(diagnostics, spec, "file", {"_PyPegen_make_module"});
  requireActionReferences(diagnostics, spec, "interactive",
                          {"_PyAST_Interactive"});
  requireActionReferences(diagnostics, spec, "statement_newline",
                          {"_PyAST_Pass"});
  requireAstConstructors(diagnostics, spec, "statement_newline", {"Pass"});
  requireActionReferences(diagnostics, spec, "eval", {"_PyAST_Expression"});
  requireActionReferences(diagnostics, spec, "func_type",
                          {"_PyAST_FunctionType"});
  requireSingleAstConstructor(diagnostics, spec, "interactive", "Interactive");
  requireSingleAstConstructor(diagnostics, spec, "eval", "Expression");
  requireSingleAstConstructor(diagnostics, spec, "func_type", "FunctionType");
  requireRuleReferences(diagnostics, spec, "func_type",
                        {"type_expressions", "expression"});
  requireLiteralReferences(diagnostics, spec, "func_type", {"(", ")", "->"});
  requireRuleReferences(diagnostics, spec, "type_expressions", {"expression"});
  requireLiteralReferences(diagnostics, spec, "type_expressions", {"*", "**"});
  requireActionReferences(diagnostics, spec, "return_stmt", {"_PyAST_Return"});
  requireActionReferences(diagnostics, spec, "raise_stmt", {"_PyAST_Raise"});
  requireActionReferences(diagnostics, spec, "pass_stmt", {"_PyAST_Pass"});
  requireActionReferences(diagnostics, spec, "break_stmt", {"_PyAST_Break"});
  requireActionReferences(diagnostics, spec, "continue_stmt",
                          {"_PyAST_Continue"});
  requireActionReferences(diagnostics, spec, "global_stmt", {"_PyAST_Global"});
  requireActionReferences(diagnostics, spec, "nonlocal_stmt",
                          {"_PyAST_Nonlocal"});
  requireActionReferences(diagnostics, spec, "del_stmt", {"_PyAST_Delete"});
  requireActionReferences(diagnostics, spec, "yield_stmt", {"_PyAST_Expr"});
  requireActionReferences(diagnostics, spec, "assert_stmt", {"_PyAST_Assert"});
  requireSingleAstConstructor(diagnostics, spec, "return_stmt", "Return");
  requireSingleAstConstructor(diagnostics, spec, "raise_stmt", "Raise");
  requireSingleAstConstructor(diagnostics, spec, "pass_stmt", "Pass");
  requireSingleAstConstructor(diagnostics, spec, "break_stmt", "Break");
  requireSingleAstConstructor(diagnostics, spec, "continue_stmt", "Continue");
  requireSingleAstConstructor(diagnostics, spec, "global_stmt", "Global");
  requireSingleAstConstructor(diagnostics, spec, "nonlocal_stmt", "Nonlocal");
  requireSingleAstConstructor(diagnostics, spec, "del_stmt", "Delete");
  requireSingleAstConstructor(diagnostics, spec, "yield_stmt", "Expr");
  requireSingleAstConstructor(diagnostics, spec, "assert_stmt", "Assert");
  requireSingleAstConstructor(diagnostics, spec, "class_def_raw", "ClassDef");
  requireSingleAstConstructor(diagnostics, spec, "if_stmt", "If");
  requireSingleAstConstructor(diagnostics, spec, "elif_stmt", "If");
  requireSingleAstConstructor(diagnostics, spec, "while_stmt", "While");
  requireSingleAstConstructor(diagnostics, spec, "match_stmt", "Match");
  requireAstConstructors(diagnostics, spec, "function_def_raw",
                         {"FunctionDef", "AsyncFunctionDef"});
  requireAstConstructors(diagnostics, spec, "for_stmt", {"For", "AsyncFor"});
  requireAstConstructors(diagnostics, spec, "with_stmt", {"With", "AsyncWith"});
  requireAstConstructors(diagnostics, spec, "try_stmt", {"Try", "TryStar"});
}

void validateExpressionGrammarContract(Diagnostics &diagnostics,
                                       const CpythonSpec &spec) {
  for (std::string_view ruleName : {"expression",
                                    "yield_expr",
                                    "star_expressions",
                                    "star_expression",
                                    "assignment_expression",
                                    "named_expression",
                                    "disjunction",
                                    "conjunction",
                                    "inversion",
                                    "comparison",
                                    "compare_op_bitwise_or_pair",
                                    "bitwise_or",
                                    "bitwise_xor",
                                    "bitwise_and",
                                    "shift_expr",
                                    "sum",
                                    "term",
                                    "factor",
                                    "power",
                                    "await_primary",
                                    "primary",
                                    "slices",
                                    "slice",
                                    "atom",
                                    "fstring",
                                    "fstring_middle",
                                    "fstring_replacement_field",
                                    "fstring_conversion",
                                    "fstring_full_format_spec",
                                    "fstring_format_spec",
                                    "tstring",
                                    "tstring_middle",
                                    "tstring_replacement_field",
                                    "tstring_format_spec_replacement_field",
                                    "tstring_full_format_spec",
                                    "tstring_format_spec",
                                    "lambdef"}) {
    requireRule(diagnostics, spec, ruleName);
  }

  requireAlternativeHeads(diagnostics, spec, "expression",
                          {"invalid_expression", "invalid_legacy_expression",
                           "disjunction", "lambdef"});
  requireRuleReferences(diagnostics, spec, "expressions", {"expression"});
  requireLiteralReferences(diagnostics, spec, "expressions", {","});
  requireAstConstructors(diagnostics, spec, "expressions", {"Tuple"});
  requireRuleReferences(diagnostics, spec, "expression",
                        {"disjunction", "lambdef"});
  requireLiteralReferences(diagnostics, spec, "expression", {"if", "else"});
  requireRuleReferences(diagnostics, spec, "invalid_legacy_expression",
                        {"star_expressions"});
  requireTokenReferences(diagnostics, spec, "invalid_legacy_expression",
                         {"NAME"});
  requireRuleReferences(diagnostics, spec, "invalid_expression",
                        {"expression_without_invalid", "disjunction",
                         "simple_stmt", "pass_stmt", "break_stmt",
                         "continue_stmt"});
  requireTokenReferences(diagnostics, spec, "invalid_expression",
                         {"FSTRING_MIDDLE", "TSTRING_MIDDLE"});
  requireLiteralReferences(diagnostics, spec, "invalid_expression",
                           {"if", "else", "lambda", ":"});

  requireRuleReferences(diagnostics, spec, "yield_expr",
                        {"expression", "star_expressions"});
  requireLiteralReferences(diagnostics, spec, "yield_expr", {"yield", "from"});
  requireRuleReferences(diagnostics, spec, "star_expressions",
                        {"star_expression"});
  requireLiteralReferences(diagnostics, spec, "star_expressions", {","});
  requireRuleReferences(diagnostics, spec, "star_expression",
                        {"bitwise_or", "expression"});
  requireLiteralReferences(diagnostics, spec, "star_expression", {"*"});

  requireRuleReferences(diagnostics, spec, "assignment_expression",
                        {"expression"});
  requireLiteralReferences(diagnostics, spec, "assignment_expression", {":="});
  requireRuleReferences(
      diagnostics, spec, "named_expression",
      {"assignment_expression", "invalid_named_expression", "expression"});
  requireRuleReferences(diagnostics, spec, "invalid_named_expression",
                        {"expression", "bitwise_or"});
  requireLiteralReferences(diagnostics, spec, "invalid_named_expression",
                           {":=", "="});

  requireRuleReferences(diagnostics, spec, "disjunction", {"conjunction"});
  requireLiteralReferences(diagnostics, spec, "disjunction", {"or"});
  requireRuleReferences(diagnostics, spec, "conjunction", {"inversion"});
  requireLiteralReferences(diagnostics, spec, "conjunction", {"and"});
  requireRuleReferences(diagnostics, spec, "inversion",
                        {"inversion", "comparison"});
  requireLiteralReferences(diagnostics, spec, "inversion", {"not"});

  requireRuleReferences(diagnostics, spec, "comparison",
                        {"bitwise_or", "compare_op_bitwise_or_pair"});
  requireAlternativeHeads(diagnostics, spec, "compare_op_bitwise_or_pair",
                          {"eq_bitwise_or", "noteq_bitwise_or",
                           "lte_bitwise_or", "lt_bitwise_or", "gte_bitwise_or",
                           "gt_bitwise_or", "notin_bitwise_or", "in_bitwise_or",
                           "isnot_bitwise_or", "is_bitwise_or"});

  requireRuleReferences(diagnostics, spec, "bitwise_or",
                        {"bitwise_or", "bitwise_xor"});
  requireLiteralReferences(diagnostics, spec, "bitwise_or", {"|"});
  requireRuleReferences(diagnostics, spec, "bitwise_xor",
                        {"bitwise_xor", "bitwise_and"});
  requireLiteralReferences(diagnostics, spec, "bitwise_xor", {"^"});
  requireRuleReferences(diagnostics, spec, "bitwise_and",
                        {"bitwise_and", "shift_expr"});
  requireLiteralReferences(diagnostics, spec, "bitwise_and", {"&"});

  requireRuleReferences(diagnostics, spec, "shift_expr",
                        {"shift_expr", "invalid_arithmetic", "sum"});
  requireLiteralReferences(diagnostics, spec, "shift_expr", {"<<", ">>"});
  requireRuleReferences(diagnostics, spec, "invalid_arithmetic",
                        {"sum", "inversion"});
  requireLiteralReferences(diagnostics, spec, "invalid_arithmetic",
                           {"+", "-", "*", "/", "%", "//", "@", "not"});
  requireRuleReferences(diagnostics, spec, "sum", {"sum", "term"});
  requireLiteralReferences(diagnostics, spec, "sum", {"+", "-"});
  requireRuleReferences(diagnostics, spec, "term",
                        {"term", "invalid_factor", "factor"});
  requireLiteralReferences(diagnostics, spec, "term",
                           {"*", "/", "//", "%", "@"});
  requireRuleReferences(diagnostics, spec, "invalid_factor", {"factor"});
  requireLiteralReferences(diagnostics, spec, "invalid_factor",
                           {"+", "-", "~", "not"});
  requireRuleReferences(diagnostics, spec, "factor", {"factor", "power"});
  requireLiteralReferences(diagnostics, spec, "factor", {"+", "-", "~"});
  requireRuleReferences(diagnostics, spec, "power",
                        {"await_primary", "factor"});
  requireLiteralReferences(diagnostics, spec, "power", {"**"});

  requireRuleReferences(diagnostics, spec, "await_primary", {"primary"});
  requireLiteralReferences(diagnostics, spec, "await_primary", {"await"});
  requireRuleReferences(diagnostics, spec, "primary",
                        {"primary", "atom", "arguments", "slices"});
  requireLiteralReferences(diagnostics, spec, "primary", {".", "(", "[", "]"});
  requireRuleReferences(diagnostics, spec, "atom",
                        {"strings", "tuple", "group", "genexp", "list",
                         "listcomp", "dict", "set", "dictcomp", "setcomp"});
  requireLiteralReferences(diagnostics, spec, "atom",
                           {"True", "False", "None", "(", "[", "{", "..."});
  requireRuleReferences(
      diagnostics, spec, "strings",
      {"invalid_string_tstring_concat", "fstring", "string", "tstring"});
  requireRuleReferences(diagnostics, spec, "invalid_string_tstring_concat",
                        {"fstring", "string", "tstring"});
  requireTokenReferences(diagnostics, spec, "string", {"STRING"});
  requireTokenReferences(diagnostics, spec, "fstring",
                         {"FSTRING_START", "FSTRING_END"});
  requireRuleReferences(diagnostics, spec, "fstring", {"fstring_middle"});
  requireRuleReferences(diagnostics, spec, "fstring_middle",
                        {"fstring_replacement_field"});
  requireTokenReferences(diagnostics, spec, "fstring_middle",
                         {"FSTRING_MIDDLE"});
  requireRuleReferences(diagnostics, spec, "fstring_replacement_field",
                        {"annotated_rhs", "fstring_conversion",
                         "fstring_full_format_spec",
                         "invalid_fstring_replacement_field"});
  requireLiteralReferences(diagnostics, spec, "fstring_replacement_field",
                           {"{", "}", "="});
  requireTokenReferences(diagnostics, spec, "fstring_conversion", {"NAME"});
  requireLiteralReferences(diagnostics, spec, "fstring_conversion", {"!"});
  requireRuleReferences(diagnostics, spec, "fstring_full_format_spec",
                        {"fstring_format_spec"});
  requireLiteralReferences(diagnostics, spec, "fstring_full_format_spec",
                           {":"});
  requireRuleReferences(diagnostics, spec, "fstring_format_spec",
                        {"fstring_replacement_field"});
  requireTokenReferences(diagnostics, spec, "fstring_format_spec",
                         {"FSTRING_MIDDLE"});
  requireRuleReferences(diagnostics, spec, "invalid_fstring_replacement_field",
                        {"annotated_rhs",
                         "invalid_fstring_conversion_character",
                         "fstring_format_spec"});
  requireLiteralReferences(diagnostics, spec,
                           "invalid_fstring_replacement_field",
                           {"{", "}", "=", "!", ":"});
  requireTokenReferences(diagnostics, spec,
                         "invalid_fstring_conversion_character", {"NAME"});
  requireLiteralReferences(diagnostics, spec,
                           "invalid_fstring_conversion_character", {"!"});
  requireTokenReferences(diagnostics, spec, "tstring",
                         {"TSTRING_START", "TSTRING_END"});
  requireRuleReferences(diagnostics, spec, "tstring", {"tstring_middle"});
  requireRuleReferences(diagnostics, spec, "tstring_middle",
                        {"tstring_replacement_field"});
  requireTokenReferences(diagnostics, spec, "tstring_middle",
                         {"TSTRING_MIDDLE"});
  requireRuleReferences(diagnostics, spec, "tstring_replacement_field",
                        {"annotated_rhs", "fstring_conversion",
                         "tstring_full_format_spec",
                         "invalid_tstring_replacement_field"});
  requireLiteralReferences(diagnostics, spec, "tstring_replacement_field",
                           {"{", "}", "="});
  requireRuleReferences(
      diagnostics, spec, "tstring_format_spec_replacement_field",
      {"annotated_rhs", "fstring_conversion", "tstring_full_format_spec",
       "invalid_tstring_replacement_field"});
  requireLiteralReferences(diagnostics, spec,
                           "tstring_format_spec_replacement_field",
                           {"{", "}", "="});
  requireRuleReferences(diagnostics, spec, "tstring_full_format_spec",
                        {"tstring_format_spec"});
  requireLiteralReferences(diagnostics, spec, "tstring_full_format_spec",
                           {":"});
  requireRuleReferences(diagnostics, spec, "tstring_format_spec",
                        {"tstring_format_spec_replacement_field"});
  requireTokenReferences(diagnostics, spec, "tstring_format_spec",
                         {"TSTRING_MIDDLE"});
  requireRuleReferences(diagnostics, spec, "invalid_tstring_replacement_field",
                        {"annotated_rhs",
                         "invalid_tstring_conversion_character",
                         "fstring_format_spec"});
  requireLiteralReferences(diagnostics, spec,
                           "invalid_tstring_replacement_field",
                           {"{", "}", "=", "!", ":"});
  requireTokenReferences(diagnostics, spec,
                         "invalid_tstring_conversion_character", {"NAME"});
  requireLiteralReferences(diagnostics, spec,
                           "invalid_tstring_conversion_character", {"!"});
  requireRuleReferences(diagnostics, spec, "group",
                        {"yield_expr", "named_expression"});
  requireLiteralReferences(diagnostics, spec, "group", {"(", ")"});
  requireRuleReferences(diagnostics, spec, "lambdef",
                        {"lambda_params", "expression"});
  requireLiteralReferences(diagnostics, spec, "lambdef", {"lambda", ":"});

  requireSingleAstConstructor(diagnostics, spec, "assignment_expression",
                              "NamedExpr");
  requireSingleAstConstructor(diagnostics, spec, "lambdef", "Lambda");
  requireAstConstructors(diagnostics, spec, "yield_expr",
                         {"Yield", "YieldFrom"});
  requireAstConstructors(diagnostics, spec, "star_expression", {"Starred"});
  requireAstConstructors(diagnostics, spec, "star_named_expression",
                         {"Starred"});
  requireAstConstructors(diagnostics, spec, "expression", {"IfExp"});
  requireSingleAstConstructor(diagnostics, spec, "disjunction", "BoolOp");
  requireSingleAstConstructor(diagnostics, spec, "conjunction", "BoolOp");
  requireAstConstructors(diagnostics, spec, "inversion", {"UnaryOp"});
  requireSingleAstConstructor(diagnostics, spec, "comparison", "Compare");
  requireSingleAstConstructor(diagnostics, spec, "bitwise_or", "BinOp");
  requireSingleAstConstructor(diagnostics, spec, "bitwise_xor", "BinOp");
  requireSingleAstConstructor(diagnostics, spec, "bitwise_and", "BinOp");
  requireSingleAstConstructor(diagnostics, spec, "shift_expr", "BinOp");
  requireSingleAstConstructor(diagnostics, spec, "sum", "BinOp");
  requireSingleAstConstructor(diagnostics, spec, "term", "BinOp");
  requireAstConstructors(diagnostics, spec, "factor", {"UnaryOp"});
  requireAstConstructors(diagnostics, spec, "power", {"BinOp"});
  requireAstConstructors(diagnostics, spec, "await_primary", {"Await"});
  requireAstConstructors(diagnostics, spec, "primary",
                         {"Attribute", "Call", "Subscript"});
  requireActionReferences(diagnostics, spec, "fstring",
                          {"_PyPegen_joined_str"});
  requireActionReferences(diagnostics, spec, "fstring_middle",
                          {"_PyPegen_constant_from_token"});
  requireActionReferences(diagnostics, spec, "fstring_replacement_field",
                          {"_PyPegen_formatted_value"});
  requireActionReferences(diagnostics, spec, "fstring_full_format_spec",
                          {"_PyPegen_setup_full_format_spec"});
  requireActionReferences(diagnostics, spec, "fstring_format_spec",
                          {"_PyPegen_decoded_constant_from_token"});
  requireActionReferences(diagnostics, spec, "tstring",
                          {"_PyPegen_template_str"});
  requireActionReferences(diagnostics, spec, "tstring_middle",
                          {"_PyPegen_constant_from_token"});
  requireActionReferences(diagnostics, spec, "tstring_replacement_field",
                          {"_PyPegen_interpolation"});
  requireActionReferences(diagnostics, spec, "tstring_full_format_spec",
                          {"_PyPegen_setup_full_format_spec"});
  requireActionReferences(diagnostics, spec, "tstring_format_spec",
                          {"_PyPegen_decoded_constant_from_token"});
  requireActionReferences(diagnostics, spec,
                          "tstring_format_spec_replacement_field",
                          {"_PyPegen_formatted_value"});
  requireActionReferences(diagnostics, spec, "string",
                          {"_PyPegen_constant_from_string"});
  requireActionReferences(
      diagnostics, spec, "strings",
      {"_PyPegen_concatenate_strings", "_PyPegen_concatenate_tstrings"});
  requireAstConstructors(diagnostics, spec, "expressions", {"Tuple"});
  requireAstConstructors(diagnostics, spec, "star_expressions", {"Tuple"});
  requireAstConstructors(diagnostics, spec, "slices", {"Tuple"});
  requireSingleAstConstructor(diagnostics, spec, "slice", "Slice");
  requireAstConstructors(diagnostics, spec, "atom", {"Constant"});
  requireSingleAstConstructor(diagnostics, spec, "list", "List");
  requireSingleAstConstructor(diagnostics, spec, "tuple", "Tuple");
  requireSingleAstConstructor(diagnostics, spec, "dict", "Dict");
  requireRuleReferences(
      diagnostics, spec, "invalid_double_starred_kvpairs",
      {"double_starred_kvpair", "invalid_kvpair", "expression", "bitwise_or"});
  requireLiteralReferences(diagnostics, spec, "invalid_double_starred_kvpairs",
                           {",", ":", "*"});
  requireRuleReferences(diagnostics, spec, "invalid_kvpair",
                        {"expression", "bitwise_or"});
  requireLiteralReferences(diagnostics, spec, "invalid_kvpair", {":", "*"});
  requireSingleAstConstructor(diagnostics, spec, "set", "Set");
  requireSingleAstConstructor(diagnostics, spec, "listcomp", "ListComp");
  requireSingleAstConstructor(diagnostics, spec, "dictcomp", "DictComp");
  requireSingleAstConstructor(diagnostics, spec, "setcomp", "SetComp");
  requireSingleAstConstructor(diagnostics, spec, "genexp", "GeneratorExp");
}

void validateStatementGrammarContract(Diagnostics &diagnostics,
                                      const CpythonSpec &spec) {
  requireRuleReferences(diagnostics, spec, "assignment",
                        {"expression", "annotated_rhs", "star_targets",
                         "single_target", "augassign", "invalid_assignment"});
  requireLiteralReferences(diagnostics, spec, "assignment",
                           {":", "=", "(", ")"});
  requireRuleReferences(diagnostics, spec, "invalid_assignment",
                        {"invalid_ann_assign_target", "star_named_expression",
                         "star_named_expressions", "expression",
                         "star_expressions", "star_targets", "yield_expr",
                         "augassign", "annotated_rhs"});
  requireLiteralReferences(diagnostics, spec, "invalid_assignment",
                           {":", "=", ","});
  requireRuleReferences(diagnostics, spec, "annotated_rhs",
                        {"yield_expr", "star_expressions"});
  requireRuleReferences(diagnostics, spec, "star_targets", {"star_target"});
  requireLiteralReferences(diagnostics, spec, "star_targets", {","});
  requireRuleReferences(diagnostics, spec, "star_targets_list_seq",
                        {"star_target"});
  requireLiteralReferences(diagnostics, spec, "star_targets_list_seq", {","});
  requireRuleReferences(diagnostics, spec, "star_targets_tuple_seq",
                        {"star_target"});
  requireLiteralReferences(diagnostics, spec, "star_targets_tuple_seq", {","});
  requireRuleReferences(diagnostics, spec, "star_target",
                        {"star_target", "target_with_star_atom"});
  requireLiteralReferences(diagnostics, spec, "star_target", {"*"});
  requireRuleReferences(diagnostics, spec, "target_with_star_atom",
                        {"t_primary", "slices", "star_atom"});
  requireTokenReferences(diagnostics, spec, "target_with_star_atom", {"NAME"});
  requireLiteralReferences(diagnostics, spec, "target_with_star_atom",
                           {".", "[", "]"});
  requireRuleReferences(diagnostics, spec, "star_atom",
                        {"target_with_star_atom", "star_targets_tuple_seq",
                         "star_targets_list_seq"});
  requireTokenReferences(diagnostics, spec, "star_atom", {"NAME"});
  requireLiteralReferences(diagnostics, spec, "star_atom",
                           {"(", ")", "[", "]"});
  requireRuleReferences(diagnostics, spec, "single_target",
                        {"single_subscript_attribute_target"});
  requireTokenReferences(diagnostics, spec, "single_target", {"NAME"});
  requireLiteralReferences(diagnostics, spec, "single_target", {"(", ")"});
  requireRuleReferences(diagnostics, spec, "single_subscript_attribute_target",
                        {"t_primary", "slices"});
  requireTokenReferences(diagnostics, spec, "single_subscript_attribute_target",
                         {"NAME"});
  requireLiteralReferences(
      diagnostics, spec, "single_subscript_attribute_target", {".", "[", "]"});
  requireRuleReferences(
      diagnostics, spec, "t_primary",
      {"t_primary", "genexp", "arguments", "slices", "atom", "t_lookahead"});
  requireLiteralReferences(diagnostics, spec, "t_primary",
                           {".", "(", ")", "[", "]"});
  requireLiteralReferences(diagnostics, spec, "t_lookahead", {"(", "[", "."});
  requireAstConstructors(diagnostics, spec, "star_targets", {"Tuple"});
  requireAstConstructors(diagnostics, spec, "star_target", {"Starred"});
  requireAstConstructors(diagnostics, spec, "target_with_star_atom",
                         {"Attribute", "Subscript"});
  requireAstConstructors(diagnostics, spec, "star_atom", {"Tuple", "List"});
  requireAstConstructors(diagnostics, spec, "single_subscript_attribute_target",
                         {"Attribute", "Subscript"});
  requireAstConstructors(diagnostics, spec, "t_primary",
                         {"Attribute", "Subscript", "Call"});
  requireRuleReferences(diagnostics, spec, "return_stmt", {"star_expressions"});
  requireLiteralReferences(diagnostics, spec, "return_stmt", {"return"});
  requireRuleReferences(diagnostics, spec, "raise_stmt", {"expression"});
  requireLiteralReferences(diagnostics, spec, "raise_stmt", {"raise", "from"});
  requireLiteralReferences(diagnostics, spec, "global_stmt", {"global", ","});
  requireTokenReferences(diagnostics, spec, "global_stmt", {"NAME"});
  requireLiteralReferences(diagnostics, spec, "nonlocal_stmt",
                           {"nonlocal", ","});
  requireTokenReferences(diagnostics, spec, "nonlocal_stmt", {"NAME"});
  requireRuleReferences(diagnostics, spec, "del_stmt",
                        {"del_targets", "invalid_del_stmt"});
  requireLiteralReferences(diagnostics, spec, "del_stmt", {"del"});
  requireRuleReferences(diagnostics, spec, "del_targets", {"del_target"});
  requireLiteralReferences(diagnostics, spec, "del_targets", {","});
  requireRuleReferences(diagnostics, spec, "del_target",
                        {"t_primary", "slices", "del_t_atom"});
  requireTokenReferences(diagnostics, spec, "del_target", {"NAME"});
  requireLiteralReferences(diagnostics, spec, "del_target", {".", "[", "]"});
  requireRuleReferences(diagnostics, spec, "del_t_atom", {"del_target"});
  requireTokenReferences(diagnostics, spec, "del_t_atom", {"NAME"});
  requireLiteralReferences(diagnostics, spec, "del_t_atom",
                           {"(", ")", "[", "]"});
  requireAstConstructors(diagnostics, spec, "del_target",
                         {"Attribute", "Subscript"});
  requireAstConstructors(diagnostics, spec, "del_t_atom", {"Tuple", "List"});
  requireRuleReferences(diagnostics, spec, "invalid_del_stmt",
                        {"star_expressions"});
  requireLiteralReferences(diagnostics, spec, "invalid_del_stmt", {"del"});
  requireRuleReferences(diagnostics, spec, "yield_stmt", {"yield_expr"});
  requireRuleReferences(diagnostics, spec, "assert_stmt", {"expression"});
  requireLiteralReferences(diagnostics, spec, "assert_stmt", {"assert", ","});
  requireSingleAstConstructor(diagnostics, spec, "global_stmt", "Global");
  requireSingleAstConstructor(diagnostics, spec, "nonlocal_stmt", "Nonlocal");
  requireSingleAstConstructor(diagnostics, spec, "del_stmt", "Delete");
  requireAlternativeHeads(diagnostics, spec, "import_stmt",
                          {"invalid_import", "import_name", "import_from"});
  requireRuleReferences(diagnostics, spec, "invalid_import", {"dotted_name"});
  requireLiteralReferences(diagnostics, spec, "invalid_import",
                           {"import", "from"});
  requireRuleReferences(diagnostics, spec, "invalid_dotted_as_name",
                        {"dotted_name", "expression"});
  requireLiteralReferences(diagnostics, spec, "invalid_dotted_as_name", {"as"});
  requireRuleReferences(diagnostics, spec, "invalid_import_from_as_name",
                        {"expression"});
  requireTokenReferences(diagnostics, spec, "invalid_import_from_as_name",
                         {"NAME"});
  requireLiteralReferences(diagnostics, spec, "invalid_import_from_as_name",
                           {"as"});
  requireRuleReferences(diagnostics, spec, "invalid_import_from_targets",
                        {"import_from_as_names"});
  requireLiteralReferences(diagnostics, spec, "invalid_import_from_targets",
                           {","});
  requireSingleAstConstructor(diagnostics, spec, "import_name", "Import");
  requireRuleReferences(diagnostics, spec, "import_from",
                        {"dotted_name", "import_from_targets"});
  requireLiteralReferences(diagnostics, spec, "import_from",
                           {"from", ".", "...", "import"});
  requireAstConstructors(diagnostics, spec, "import_from", {"ImportFrom"});
  requireRuleReferences(
      diagnostics, spec, "import_from_targets",
      {"import_from_as_names", "invalid_import_from_targets"});
  requireLiteralReferences(diagnostics, spec, "import_from_targets",
                           {"(", ")", ",", "*"});
  requireActionReferences(diagnostics, spec, "import_from_targets",
                          {"_PyPegen_alias_for_star"});
  requireRuleReferences(diagnostics, spec, "import_from_as_names",
                        {"import_from_as_name"});
  requireTokenReferences(diagnostics, spec, "import_from_as_name", {"NAME"});
  requireLiteralReferences(diagnostics, spec, "import_from_as_name", {"as"});
  requireSingleAstConstructor(diagnostics, spec, "dotted_as_name", "alias");
  requireSingleAstConstructor(diagnostics, spec, "import_from_as_name",
                              "alias");

  requireAlternativeHeads(diagnostics, spec, "function_def",
                          {"decorators", "function_def_raw"});
  requireRuleReferences(
      diagnostics, spec, "function_def_raw",
      {"type_params", "params", "expression", "func_type_comment", "block"});
  requireTokenReferences(diagnostics, spec, "function_def_raw", {"NAME"});
  requireLiteralReferences(diagnostics, spec, "function_def_raw",
                           {"def", "async", "(", ")", "->", ":"});
  requireAlternativeHeads(diagnostics, spec, "class_def",
                          {"decorators", "class_def_raw"});
  requireRuleReferences(diagnostics, spec, "class_def_raw",
                        {"type_params", "arguments", "block"});
  requireTokenReferences(diagnostics, spec, "class_def_raw", {"NAME"});
  requireLiteralReferences(diagnostics, spec, "class_def_raw",
                           {"class", "(", ")", ":"});

  requireRuleReferences(
      diagnostics, spec, "for_stmt",
      {"star_targets", "star_expressions", "block", "else_block"});
  requireLiteralReferences(diagnostics, spec, "for_stmt",
                           {"for", "async", "in", ":"});
  requireRuleReferences(diagnostics, spec, "with_stmt", {"with_item", "block"});
  requireLiteralReferences(diagnostics, spec, "with_stmt",
                           {"with", "async", "(", ")", ",", ":"});
  requireRuleReferences(diagnostics, spec, "with_item",
                        {"expression", "star_target"});
  requireLiteralReferences(diagnostics, spec, "with_item", {"as"});
  requireSingleAstConstructor(diagnostics, spec, "with_item", "withitem");
  requireWithTypeCommentShape(diagnostics, spec);
  requireRuleReferences(diagnostics, spec, "try_stmt",
                        {"block", "finally_block", "except_block",
                         "except_star_block", "else_block"});
  requireLiteralReferences(diagnostics, spec, "try_stmt", {"try", ":"});
  requireRuleReferences(diagnostics, spec, "except_block",
                        {"expression", "expressions", "block"});
  requireTokenReferences(diagnostics, spec, "except_block", {"NAME"});
  requireLiteralReferences(diagnostics, spec, "except_block",
                           {"except", "as", ":"});
  requireRuleReferences(diagnostics, spec, "except_star_block",
                        {"expression", "expressions", "block"});
  requireTokenReferences(diagnostics, spec, "except_star_block", {"NAME"});
  requireLiteralReferences(diagnostics, spec, "except_star_block",
                           {"except", "*", "as", ":"});
  requireSingleAstConstructor(diagnostics, spec, "except_block",
                              "ExceptHandler");
  requireSingleAstConstructor(diagnostics, spec, "except_star_block",
                              "ExceptHandler");
  requireRuleReferences(diagnostics, spec, "match_stmt",
                        {"subject_expr", "case_block"});
  requireLiteralReferences(diagnostics, spec, "match_stmt", {"match", ":"});
  requireRuleReferences(diagnostics, spec, "case_block",
                        {"patterns", "guard", "block"});
  requireLiteralReferences(diagnostics, spec, "case_block", {"case", ":"});
  requireSingleAstConstructor(diagnostics, spec, "subject_expr", "Tuple");
  requireSingleAstConstructor(diagnostics, spec, "case_block", "match_case");
}

void validateParameterGrammarContract(Diagnostics &diagnostics,
                                      const CpythonSpec &spec) {
  requireAlternativeHeads(diagnostics, spec, "params",
                          {"invalid_parameters", "parameters"});
  requireAlternativeHeads(diagnostics, spec, "parameters",
                          {"slash_no_default", "slash_with_default",
                           "param_no_default", "param_with_default",
                           "star_etc"});
  requireRuleReferences(diagnostics, spec, "parameters",
                        {"slash_no_default", "slash_with_default",
                         "param_no_default", "param_with_default", "star_etc"});
  requireRuleReferences(diagnostics, spec, "star_etc",
                        {"param_no_default", "param_no_default_star_annotation",
                         "param_maybe_default", "kwds"});
  requireLiteralReferences(diagnostics, spec, "star_etc", {"*", ","});
  requireRuleReferences(diagnostics, spec, "kwds", {"param_no_default"});
  requireLiteralReferences(diagnostics, spec, "kwds", {"**"});
  requireRuleReferences(diagnostics, spec, "param_no_default", {"param"});
  requireTokenReferences(diagnostics, spec, "param_no_default",
                         {"TYPE_COMMENT"});
  requireRuleReferences(diagnostics, spec, "param_no_default_star_annotation",
                        {"param_star_annotation"});
  requireTokenReferences(diagnostics, spec, "param_no_default_star_annotation",
                         {"TYPE_COMMENT"});
  requireRuleReferences(diagnostics, spec, "param_with_default",
                        {"param", "default"});
  requireTokenReferences(diagnostics, spec, "param_with_default",
                         {"TYPE_COMMENT"});
  requireRuleReferences(diagnostics, spec, "param_maybe_default",
                        {"param", "default"});
  requireTokenReferences(diagnostics, spec, "param_maybe_default",
                         {"TYPE_COMMENT"});
  requireTokenReferences(diagnostics, spec, "param", {"NAME"});
  requireRuleReferences(diagnostics, spec, "param", {"annotation"});
  requireTokenReferences(diagnostics, spec, "param_star_annotation", {"NAME"});
  requireRuleReferences(diagnostics, spec, "param_star_annotation",
                        {"star_annotation"});
  requireSingleAstConstructor(diagnostics, spec, "param", "arg");
  requireSingleAstConstructor(diagnostics, spec, "param_star_annotation",
                              "arg");
  requireRuleReferences(diagnostics, spec, "annotation", {"expression"});
  requireRuleReferences(diagnostics, spec, "star_annotation",
                        {"star_expression"});
  requireRuleReferences(diagnostics, spec, "default", {"expression"});
  requireLiteralReferences(diagnostics, spec, "annotation", {":"});
  requireLiteralReferences(diagnostics, spec, "star_annotation", {":"});
  requireLiteralReferences(diagnostics, spec, "default", {"="});

  requireAlternativeHeads(diagnostics, spec, "lambda_params",
                          {"invalid_lambda_parameters", "lambda_parameters"});
  requireAlternativeHeads(diagnostics, spec, "lambda_parameters",
                          {"lambda_slash_no_default",
                           "lambda_slash_with_default",
                           "lambda_param_no_default",
                           "lambda_param_with_default", "lambda_star_etc"});
  requireRuleReferences(diagnostics, spec, "lambda_parameters",
                        {"lambda_slash_no_default", "lambda_slash_with_default",
                         "lambda_param_no_default", "lambda_param_with_default",
                         "lambda_star_etc"});
  requireRuleReferences(diagnostics, spec, "lambda_slash_no_default",
                        {"lambda_param_no_default"});
  requireLiteralReferences(diagnostics, spec, "lambda_slash_no_default",
                           {"/", ",", ":"});
  requireRuleReferences(
      diagnostics, spec, "lambda_slash_with_default",
      {"lambda_param_no_default", "lambda_param_with_default"});
  requireLiteralReferences(diagnostics, spec, "lambda_slash_with_default",
                           {"/", ",", ":"});
  requireRuleReferences(
      diagnostics, spec, "lambda_star_etc",
      {"lambda_param_no_default", "lambda_param_maybe_default", "lambda_kwds"});
  requireLiteralReferences(diagnostics, spec, "lambda_star_etc", {"*", ","});
  requireRuleReferences(diagnostics, spec, "lambda_kwds",
                        {"lambda_param_no_default"});
  requireLiteralReferences(diagnostics, spec, "lambda_kwds", {"**"});
  requireRuleReferences(diagnostics, spec, "lambda_param_no_default",
                        {"lambda_param"});
  requireLiteralReferences(diagnostics, spec, "lambda_param_no_default",
                           {",", ":"});
  requireRuleReferences(diagnostics, spec, "lambda_param_with_default",
                        {"lambda_param", "default"});
  requireLiteralReferences(diagnostics, spec, "lambda_param_with_default",
                           {",", ":"});
  requireRuleReferences(diagnostics, spec, "lambda_param_maybe_default",
                        {"lambda_param", "default"});
  requireLiteralReferences(diagnostics, spec, "lambda_param_maybe_default",
                           {",", ":"});
  requireTokenReferences(diagnostics, spec, "lambda_param", {"NAME"});
  requireSingleAstConstructor(diagnostics, spec, "lambda_param", "arg");

  requireRuleReferences(diagnostics, spec, "type_params", {"type_param_seq"});
  requireLiteralReferences(diagnostics, spec, "type_params", {"[", "]"});
  requireRuleReferences(diagnostics, spec, "type_param_seq", {"type_param"});
  requireRuleReferences(
      diagnostics, spec, "type_param",
      {"type_param_bound", "type_param_default", "type_param_starred_default"});
  requireTokenReferences(diagnostics, spec, "type_param", {"NAME"});
  requireLiteralReferences(diagnostics, spec, "type_param", {"*", "**"});
  requireSingleAstConstructor(diagnostics, spec, "type_alias", "TypeAlias");
  requireAstConstructors(diagnostics, spec, "type_param",
                         {"TypeVar", "TypeVarTuple", "ParamSpec"});
}

void validateCallArgumentGrammarContract(Diagnostics &diagnostics,
                                         const CpythonSpec &spec) {
  requireRuleReferences(diagnostics, spec, "arguments",
                        {"args", "invalid_arguments"});
  requireLiteralReferences(diagnostics, spec, "arguments", {","});
  requireRuleReferences(
      diagnostics, spec, "args",
      {"starred_expression", "assignment_expression", "expression", "kwargs"});
  requireRuleReferences(diagnostics, spec, "kwargs",
                        {"kwarg_or_starred", "kwarg_or_double_starred"});
  requireRuleReferences(diagnostics, spec, "kwarg_or_starred",
                        {"expression", "starred_expression"});
  requireTokenReferences(diagnostics, spec, "kwarg_or_starred", {"NAME"});
  requireLiteralReferences(diagnostics, spec, "kwarg_or_starred", {"="});
  requireAstConstructors(diagnostics, spec, "kwarg_or_starred", {"keyword"});
  requireRuleReferences(diagnostics, spec, "kwarg_or_double_starred",
                        {"expression"});
  requireTokenReferences(diagnostics, spec, "kwarg_or_double_starred",
                         {"NAME"});
  requireLiteralReferences(diagnostics, spec, "kwarg_or_double_starred",
                           {"=", "**"});
  requireAstConstructors(diagnostics, spec, "kwarg_or_double_starred",
                         {"keyword"});
  requireRuleReferences(diagnostics, spec, "starred_expression",
                        {"invalid_starred_expression_unpacking",
                         "invalid_starred_expression", "expression"});
  requireLiteralReferences(diagnostics, spec, "starred_expression", {"*"});
  requireSingleAstConstructor(diagnostics, spec, "starred_expression",
                              "Starred");
  requireRuleReferences(diagnostics, spec,
                        "invalid_starred_expression_unpacking", {"expression"});
  requireLiteralReferences(diagnostics, spec,
                           "invalid_starred_expression_unpacking", {"*", "="});
  requireLiteralReferences(diagnostics, spec, "invalid_starred_expression",
                           {"*"});

  requireRuleReferences(diagnostics, spec, "invalid_arguments",
                        {"args", "assignment_expression", "expression",
                         "for_if_clauses", "kwargs", "starred_expression"});
  requireLiteralReferences(diagnostics, spec, "invalid_arguments", {",", "="});
  requireRuleReferences(diagnostics, spec, "invalid_kwarg", {"expression"});
  requireTokenReferences(diagnostics, spec, "invalid_kwarg", {"NAME"});
  requireLiteralReferences(diagnostics, spec, "invalid_kwarg",
                           {"True", "False", "None", "=", "**"});

  requireRuleReferences(diagnostics, spec, "for_if_clauses", {"for_if_clause"});
  requireRuleReferences(diagnostics, spec, "for_if_clause",
                        {"star_targets", "disjunction"});
  requireLiteralReferences(diagnostics, spec, "for_if_clause",
                           {"for", "async", "in", "if"});
  requireSingleAstConstructor(diagnostics, spec, "for_if_clause",
                              "comprehension");
}

void validatePatternGrammarContract(Diagnostics &diagnostics,
                                    const CpythonSpec &spec) {
  requireRuleReferences(diagnostics, spec, "patterns",
                        {"open_sequence_pattern", "pattern"});
  requireAlternativeHeads(diagnostics, spec, "pattern",
                          {"as_pattern", "or_pattern"});
  requireRuleReferences(diagnostics, spec, "as_pattern",
                        {"or_pattern", "pattern_capture_target"});
  requireLiteralReferences(diagnostics, spec, "as_pattern", {"as"});
  requireRuleReferences(diagnostics, spec, "or_pattern", {"closed_pattern"});
  requireLiteralReferences(diagnostics, spec, "or_pattern", {"|"});
  requireAlternativeHeads(diagnostics, spec, "closed_pattern",
                          {"literal_pattern", "capture_pattern",
                           "wildcard_pattern", "value_pattern", "group_pattern",
                           "sequence_pattern", "mapping_pattern",
                           "class_pattern"});
  requireRuleReferences(diagnostics, spec, "literal_pattern",
                        {"signed_number", "complex_number", "strings"});
  requireLiteralReferences(diagnostics, spec, "literal_pattern",
                           {"None", "True", "False"});
  requireRuleReferences(diagnostics, spec, "literal_expr",
                        {"signed_number", "complex_number", "strings"});
  requireLiteralReferences(diagnostics, spec, "literal_expr",
                           {"None", "True", "False"});
  requireTokenReferences(diagnostics, spec, "literal_expr",
                         {"STRING", "FSTRING_START", "TSTRING_START"});
  requireRuleReferences(diagnostics, spec, "complex_number",
                        {"signed_real_number", "imaginary_number"});
  requireLiteralReferences(diagnostics, spec, "complex_number", {"+", "-"});
  requireTokenReferences(diagnostics, spec, "signed_number", {"NUMBER"});
  requireLiteralReferences(diagnostics, spec, "signed_number", {"-"});
  requireRuleReferences(diagnostics, spec, "signed_real_number",
                        {"real_number"});
  requireLiteralReferences(diagnostics, spec, "signed_real_number", {"-"});
  requireTokenReferences(diagnostics, spec, "real_number", {"NUMBER"});
  requireTokenReferences(diagnostics, spec, "imaginary_number", {"NUMBER"});
  requireTokenReferences(diagnostics, spec, "pattern_capture_target", {"NAME"});
  requireLiteralReferences(diagnostics, spec, "pattern_capture_target",
                           {"_", ".", "(", "="});
  requireRuleReferences(diagnostics, spec, "value_pattern", {"attr"});
  requireRuleReferences(diagnostics, spec, "group_pattern", {"pattern"});
  requireLiteralReferences(diagnostics, spec, "group_pattern", {"(", ")"});
  requireRuleReferences(diagnostics, spec, "sequence_pattern",
                        {"maybe_sequence_pattern", "open_sequence_pattern"});
  requireLiteralReferences(diagnostics, spec, "sequence_pattern",
                           {"[", "]", "(", ")"});
  requireRuleReferences(diagnostics, spec, "maybe_sequence_pattern",
                        {"maybe_star_pattern"});
  requireRuleReferences(diagnostics, spec, "maybe_star_pattern",
                        {"star_pattern", "pattern"});
  requireRuleReferences(diagnostics, spec, "star_pattern",
                        {"pattern_capture_target", "wildcard_pattern"});
  requireLiteralReferences(diagnostics, spec, "star_pattern", {"*"});
  requireRuleReferences(diagnostics, spec, "mapping_pattern",
                        {"double_star_pattern", "items_pattern"});
  requireLiteralReferences(diagnostics, spec, "mapping_pattern",
                           {"{", "}", ","});
  requireRuleReferences(diagnostics, spec, "key_value_pattern",
                        {"literal_expr", "attr", "pattern"});
  requireLiteralReferences(diagnostics, spec, "key_value_pattern", {":"});
  requireRuleReferences(diagnostics, spec, "double_star_pattern",
                        {"pattern_capture_target"});
  requireLiteralReferences(diagnostics, spec, "double_star_pattern", {"**"});
  requireRuleReferences(
      diagnostics, spec, "class_pattern",
      {"name_or_attr", "positional_patterns", "keyword_patterns"});
  requireLiteralReferences(diagnostics, spec, "class_pattern", {"(", ")", ","});

  requireAstConstructors(diagnostics, spec, "literal_pattern",
                         {"MatchValue", "MatchSingleton"});
  requireSingleAstConstructor(diagnostics, spec, "capture_pattern", "MatchAs");
  requireSingleAstConstructor(diagnostics, spec, "wildcard_pattern", "MatchAs");
  requireSingleAstConstructor(diagnostics, spec, "value_pattern", "MatchValue");
  requireSingleAstConstructor(diagnostics, spec, "sequence_pattern",
                              "MatchSequence");
  requireSingleAstConstructor(diagnostics, spec, "star_pattern", "MatchStar");
  requireSingleAstConstructor(diagnostics, spec, "mapping_pattern",
                              "MatchMapping");
  requireSingleAstConstructor(diagnostics, spec, "class_pattern", "MatchClass");
  requireAstConstructors(diagnostics, spec, "or_pattern", {"MatchOr"});
}

} // namespace

void validateCpythonParserContract(Diagnostics &diagnostics) {
  const CpythonSpec &spec = cpython314Spec();
  for (const std::string &message : spec.peg.diagnostics) {
    report(diagnostics,
           "failed to load vendored CPython 3.14 PEG grammar: " + message);
  }

  validateTokenContract(diagnostics, spec);
  validateKeywordContract(diagnostics, spec);
  validateFirstSetContract(diagnostics, spec);
  validateActionContract(diagnostics, spec);
  validateGeneratedParserContract(diagnostics, spec);
  validateActionLabelContract(diagnostics, spec);
  validateRequiredEntryRules(diagnostics, spec);
  validateExpressionGrammarContract(diagnostics, spec);
  validateStatementGrammarContract(diagnostics, spec);
  validateParameterGrammarContract(diagnostics, spec);
  validateCallArgumentGrammarContract(diagnostics, spec);
  validatePatternGrammarContract(diagnostics, spec);
}

} // namespace lython::parser
