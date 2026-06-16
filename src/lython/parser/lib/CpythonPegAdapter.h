#pragma once

#include "Grammar.h"

#include <optional>
#include <string>
#include <string_view>

namespace lython::parser {

struct CpythonPegToken {
  std::string_view rawText;
  std::string_view cpythonName;
  bool isName = false;
  bool isHardKeyword = false;
};

enum class CpythonStatementForm { None, Compound, Simple };
enum class CpythonAtomForm {
  None,
  Name,
  Singleton,
  Strings,
  Number,
  Parenthesized,
  List,
  DictOrSet,
  Ellipsis,
};
enum class CpythonPrimarySuffixForm { None, Attribute, Call, Subscript };
enum class CpythonInfixOperatorFamily {
  BitwiseOr,
  BitwiseXor,
  BitwiseAnd,
  Shift,
  Sum,
  Term,
  Power,
};
enum class CpythonComprehensionClauseForm { None, Sync, Async };

struct CpythonComparisonOperator {
  std::string_view spelling;
  unsigned tokenCount = 0;
};

class CpythonPegAdapter {
public:
  bool firstSetMatches(const CpythonPegFirstSet &first,
                       const CpythonPegToken &token) const;
  bool hasRule(std::string_view rule) const;
  bool ruleCanStart(std::string_view rule, const CpythonPegToken &token) const;
  bool alternativeCanStart(std::string_view rule, std::string_view headRule,
                           const CpythonPegToken &token) const;
  bool matchesLiteral(std::string_view rule, const CpythonPegToken &current,
                      std::string_view literal) const;
  bool ruleCallsAction(std::string_view rule, std::string_view callee) const;
  bool ruleHasAstConstructor(std::string_view rule,
                             std::string_view constructor) const;
  std::optional<std::string> singleAstConstructor(std::string_view rule) const;
  bool hasAstNodeKind(std::string_view kind) const;

  std::string compoundStatementHead(const CpythonPegToken &current,
                                    const CpythonPegToken *next,
                                    bool looksLikeMatchStatement) const;
  std::string simpleStatementHead(const CpythonPegToken &current,
                                  bool looksLikeTypeAlias) const;
  CpythonStatementForm statementForm(const CpythonPegToken &current,
                                     const CpythonPegToken *next,
                                     bool looksLikeMatchStatement) const;
  CpythonAtomForm atomForm(const CpythonPegToken &current) const;
  CpythonPrimarySuffixForm
  primarySuffixForm(const CpythonPegToken &current) const;
  std::optional<CpythonComparisonOperator>
  comparisonOperator(const CpythonPegToken &current,
                     const CpythonPegToken *next) const;
  std::optional<std::string_view>
  infixOperator(CpythonInfixOperatorFamily family,
                const CpythonPegToken &current) const;
  std::optional<std::string_view>
  unaryOperator(const CpythonPegToken &current) const;
  bool startsInversion(const CpythonPegToken &current) const;
  bool startsLambda(const CpythonPegToken &current) const;
  bool startsAssignmentExpression(const CpythonPegToken &current,
                                  const CpythonPegToken *next) const;
  bool startsAwait(const CpythonPegToken &current) const;
  CpythonComprehensionClauseForm
  comprehensionClauseForm(const CpythonPegToken &current,
                          const CpythonPegToken *next) const;
  bool startsComprehensionIf(const CpythonPegToken &current) const;
};

} // namespace lython::parser
