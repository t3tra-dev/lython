#include "CpythonPegAdapter.h"

#include "CpythonSpec.h"

#include <initializer_list>

namespace lython::parser {
namespace {

bool ruleReferencesLiteral(std::string_view ruleName,
                           std::string_view literal) {
  const CpythonPegRule *rule = cpython314Spec().peg.findRule(ruleName);
  return rule && rule->referencesLiteral(literal);
}

std::string chooseAlternative(const CpythonPegAdapter &adapter,
                              const CpythonPegToken &token,
                              std::string_view rule,
                              std::initializer_list<std::string_view> heads) {
  for (std::string_view head : heads)
    if (adapter.alternativeCanStart(rule, head, token))
      return std::string(head);
  return std::string();
}

std::string_view infixOperatorRule(CpythonInfixOperatorFamily family) {
  switch (family) {
  case CpythonInfixOperatorFamily::BitwiseOr:
    return "bitwise_or";
  case CpythonInfixOperatorFamily::BitwiseXor:
    return "bitwise_xor";
  case CpythonInfixOperatorFamily::BitwiseAnd:
    return "bitwise_and";
  case CpythonInfixOperatorFamily::Shift:
    return "shift_expr";
  case CpythonInfixOperatorFamily::Sum:
    return "sum";
  case CpythonInfixOperatorFamily::Term:
    return "term";
  case CpythonInfixOperatorFamily::Power:
    return "power";
  }
  return std::string_view();
}

bool infixFamilyContains(CpythonInfixOperatorFamily family,
                         std::string_view spelling) {
  switch (family) {
  case CpythonInfixOperatorFamily::BitwiseOr:
    return spelling == "|";
  case CpythonInfixOperatorFamily::BitwiseXor:
    return spelling == "^";
  case CpythonInfixOperatorFamily::BitwiseAnd:
    return spelling == "&";
  case CpythonInfixOperatorFamily::Shift:
    return spelling == "<<" || spelling == ">>";
  case CpythonInfixOperatorFamily::Sum:
    return spelling == "+" || spelling == "-";
  case CpythonInfixOperatorFamily::Term:
    return spelling == "*" || spelling == "/" || spelling == "//" ||
           spelling == "%" || spelling == "@";
  case CpythonInfixOperatorFamily::Power:
    return spelling == "**";
  }
  return false;
}

} // namespace

bool CpythonPegAdapter::firstSetMatches(const CpythonPegFirstSet &first,
                                        const CpythonPegToken &token) const {
  if (first.hasLiteral(token.rawText))
    return true;
  if (first.hasToken("SOFT_KEYWORD") && token.isName &&
      isCpythonSoftKeyword(token.rawText))
    return true;
  if (token.isHardKeyword && token.cpythonName == "NAME")
    return false;
  return first.hasToken(token.cpythonName);
}

bool CpythonPegAdapter::hasRule(std::string_view rule) const {
  return cpython314Spec().peg.findRule(rule) != nullptr;
}

bool CpythonPegAdapter::ruleCanStart(std::string_view rule,
                                     const CpythonPegToken &token) const {
  const CpythonPegRule *pegRule = cpython314Spec().peg.findRule(rule);
  return pegRule && firstSetMatches(pegRule->first, token);
}

bool CpythonPegAdapter::alternativeCanStart(
    std::string_view rule, std::string_view headRule,
    const CpythonPegToken &token) const {
  const CpythonPegRule *pegRule = cpython314Spec().peg.findRule(rule);
  if (!pegRule)
    return false;
  const CpythonPegAlternative *alternative =
      pegRule->findAlternativeByHead(headRule);
  return alternative && firstSetMatches(alternative->first, token);
}

bool CpythonPegAdapter::matchesLiteral(std::string_view rule,
                                       const CpythonPegToken &current,
                                       std::string_view literal) const {
  return current.rawText == literal && ruleReferencesLiteral(rule, literal);
}

bool CpythonPegAdapter::ruleCallsAction(std::string_view rule,
                                        std::string_view callee) const {
  const CpythonPegRule *pegRule = cpython314Spec().peg.findRule(rule);
  return pegRule && pegRule->callsAction(callee);
}

bool CpythonPegAdapter::ruleHasAstConstructor(
    std::string_view rule, std::string_view constructor) const {
  const CpythonPegRule *pegRule = cpython314Spec().peg.findRule(rule);
  return pegRule && pegRule->hasAstConstructor(constructor);
}

std::optional<std::string>
CpythonPegAdapter::singleAstConstructor(std::string_view rule) const {
  const CpythonPegRule *pegRule = cpython314Spec().peg.findRule(rule);
  if (!pegRule)
    return std::nullopt;
  return pegRule->singleAstConstructor();
}

bool CpythonPegAdapter::hasAstNodeKind(std::string_view kind) const {
  return isCpythonAstNodeKind(kind);
}

std::string
CpythonPegAdapter::compoundStatementHead(const CpythonPegToken &current,
                                         const CpythonPegToken *next,
                                         bool looksLikeMatchStatement) const {
  if (!ruleCanStart("compound_stmt", current))
    return std::string();

  if (current.rawText == "@" &&
      (alternativeCanStart("compound_stmt", "function_def", current) ||
       alternativeCanStart("compound_stmt", "class_def", current)))
    return "decorated";

  if (current.rawText == "async") {
    if (!next)
      return std::string();
    if (next->rawText == "def" &&
        alternativeCanStart("compound_stmt", "function_def", current))
      return "function_def";
    if (next->rawText == "with" &&
        alternativeCanStart("compound_stmt", "with_stmt", current))
      return "with_stmt";
    if (next->rawText == "for" &&
        alternativeCanStart("compound_stmt", "for_stmt", current))
      return "for_stmt";
    return std::string();
  }

  if (std::string head = chooseAlternative(*this, current, "compound_stmt",
                                           {"function_def", "class_def",
                                            "if_stmt", "while_stmt", "for_stmt",
                                            "with_stmt", "try_stmt"});
      !head.empty())
    return head;

  if (looksLikeMatchStatement &&
      alternativeCanStart("compound_stmt", "match_stmt", current))
    return "match_stmt";
  return std::string();
}

std::string
CpythonPegAdapter::simpleStatementHead(const CpythonPegToken &current,
                                       bool looksLikeTypeAlias) const {
  if (!ruleCanStart("simple_stmt", current))
    return std::string();
  if (looksLikeTypeAlias &&
      alternativeCanStart("simple_stmt", "type_alias", current))
    return "type_alias";
  return chooseAlternative(*this, current, "simple_stmt",
                           {"import_stmt", "global_stmt", "nonlocal_stmt",
                            "del_stmt", "yield_stmt", "pass_stmt", "break_stmt",
                            "continue_stmt", "return_stmt", "raise_stmt",
                            "assert_stmt"});
}

CpythonStatementForm
CpythonPegAdapter::statementForm(const CpythonPegToken &current,
                                 const CpythonPegToken *next,
                                 bool looksLikeMatchStatement) const {
  if (!ruleCanStart("statement", current))
    return CpythonStatementForm::None;

  if (alternativeCanStart("statement", "compound_stmt", current) &&
      !compoundStatementHead(current, next, looksLikeMatchStatement).empty())
    return CpythonStatementForm::Compound;

  if (alternativeCanStart("statement", "simple_stmts", current) &&
      ruleCanStart("simple_stmt", current))
    return CpythonStatementForm::Simple;

  return CpythonStatementForm::None;
}

CpythonAtomForm
CpythonPegAdapter::atomForm(const CpythonPegToken &current) const {
  if (!ruleCanStart("atom", current))
    return CpythonAtomForm::None;

  if (!current.isHardKeyword && current.cpythonName == "NAME")
    return CpythonAtomForm::Name;
  if (current.rawText == "True" || current.rawText == "False" ||
      current.rawText == "None")
    return CpythonAtomForm::Singleton;
  if (alternativeCanStart("atom", "strings", current) &&
      (current.cpythonName == "STRING" ||
       current.cpythonName == "FSTRING_START" ||
       current.cpythonName == "TSTRING_START"))
    return CpythonAtomForm::Strings;
  if (current.cpythonName == "NUMBER")
    return CpythonAtomForm::Number;
  if (alternativeCanStart("atom", "tuple", current) && current.rawText == "(")
    return CpythonAtomForm::Parenthesized;
  if (alternativeCanStart("atom", "list", current) && current.rawText == "[")
    return CpythonAtomForm::List;
  if (alternativeCanStart("atom", "dict", current) && current.rawText == "{")
    return CpythonAtomForm::DictOrSet;
  if (current.rawText == "...")
    return CpythonAtomForm::Ellipsis;
  return CpythonAtomForm::None;
}

CpythonPrimarySuffixForm
CpythonPegAdapter::primarySuffixForm(const CpythonPegToken &current) const {
  if (current.rawText == "." && ruleReferencesLiteral("primary", "."))
    return CpythonPrimarySuffixForm::Attribute;
  if (current.rawText == "(" && ruleReferencesLiteral("primary", "("))
    return CpythonPrimarySuffixForm::Call;
  if (current.rawText == "[" && ruleReferencesLiteral("primary", "["))
    return CpythonPrimarySuffixForm::Subscript;
  return CpythonPrimarySuffixForm::None;
}

std::optional<CpythonComparisonOperator>
CpythonPegAdapter::comparisonOperator(const CpythonPegToken &current,
                                      const CpythonPegToken *next) const {
  if (!ruleCanStart("compare_op_bitwise_or_pair", current))
    return std::nullopt;

  if (current.rawText == "is" && next && next->rawText == "not" &&
      ruleCanStart("isnot_bitwise_or", current))
    return CpythonComparisonOperator{"is not", 2};
  if (current.rawText == "not" && next && next->rawText == "in" &&
      ruleCanStart("notin_bitwise_or", current))
    return CpythonComparisonOperator{"not in", 2};

  struct SingleTokenComparison {
    std::string_view spelling;
    std::string_view rule;
  };
  for (SingleTokenComparison op :
       {SingleTokenComparison{"==", "eq_bitwise_or"},
        SingleTokenComparison{"!=", "noteq_bitwise_or"},
        SingleTokenComparison{"<=", "lte_bitwise_or"},
        SingleTokenComparison{"<", "lt_bitwise_or"},
        SingleTokenComparison{">=", "gte_bitwise_or"},
        SingleTokenComparison{">", "gt_bitwise_or"},
        SingleTokenComparison{"in", "in_bitwise_or"},
        SingleTokenComparison{"is", "is_bitwise_or"}}) {
    if (current.rawText == op.spelling && ruleCanStart(op.rule, current))
      return CpythonComparisonOperator{op.spelling, 1};
  }
  return std::nullopt;
}

std::optional<std::string_view>
CpythonPegAdapter::infixOperator(CpythonInfixOperatorFamily family,
                                 const CpythonPegToken &current) const {
  std::string_view rule = infixOperatorRule(family);
  if (rule.empty() || !infixFamilyContains(family, current.rawText))
    return std::nullopt;
  if (!ruleReferencesLiteral(rule, current.rawText))
    return std::nullopt;
  return current.rawText;
}

std::optional<std::string_view>
CpythonPegAdapter::unaryOperator(const CpythonPegToken &current) const {
  if (!ruleCanStart("factor", current))
    return std::nullopt;
  for (std::string_view spelling : {"+", "-", "~"}) {
    if (current.rawText == spelling &&
        ruleReferencesLiteral("factor", spelling))
      return spelling;
  }
  return std::nullopt;
}

bool CpythonPegAdapter::startsInversion(const CpythonPegToken &current) const {
  return current.rawText == "not" && ruleCanStart("inversion", current) &&
         ruleReferencesLiteral("inversion", "not");
}

bool CpythonPegAdapter::startsLambda(const CpythonPegToken &current) const {
  return current.rawText == "lambda" && ruleCanStart("lambdef", current) &&
         ruleReferencesLiteral("lambdef", "lambda");
}

bool CpythonPegAdapter::startsAssignmentExpression(
    const CpythonPegToken &current, const CpythonPegToken *next) const {
  return !current.isHardKeyword && current.cpythonName == "NAME" && next &&
         next->rawText == ":=" &&
         ruleCanStart("assignment_expression", current) &&
         ruleReferencesLiteral("assignment_expression", ":=");
}

bool CpythonPegAdapter::startsAwait(const CpythonPegToken &current) const {
  return current.rawText == "await" && ruleCanStart("await_primary", current) &&
         ruleReferencesLiteral("await_primary", "await");
}

CpythonComprehensionClauseForm
CpythonPegAdapter::comprehensionClauseForm(const CpythonPegToken &current,
                                           const CpythonPegToken *next) const {
  if (!ruleCanStart("for_if_clause", current))
    return CpythonComprehensionClauseForm::None;
  if (current.rawText == "for" && ruleReferencesLiteral("for_if_clause", "for"))
    return CpythonComprehensionClauseForm::Sync;
  if (current.rawText == "async" && next && next->rawText == "for" &&
      ruleReferencesLiteral("for_if_clause", "async"))
    return CpythonComprehensionClauseForm::Async;
  return CpythonComprehensionClauseForm::None;
}

bool CpythonPegAdapter::startsComprehensionIf(
    const CpythonPegToken &current) const {
  return current.rawText == "if" &&
         ruleReferencesLiteral("for_if_clause", "if");
}

} // namespace lython::parser
