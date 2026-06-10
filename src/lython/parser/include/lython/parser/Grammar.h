#pragma once

#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace lython::parser {

enum class CpythonPegExprKind {
  Empty,
  Name,
  Literal,
  Sequence,
  Choice,
  Optional,
  Repeat0,
  Repeat1,
  SepRepeat1,
  And,
  Not,
  Eager,
  Cut,
};

struct CpythonPegExpr {
  CpythonPegExprKind kind = CpythonPegExprKind::Empty;
  std::string value;
  char quote = '\0';
  std::string label;
  std::vector<CpythonPegExpr> children;
};

struct CpythonPegFirstSet {
  CpythonPegFirstSet() = default;
  CpythonPegFirstSet(bool nullable, bool unknown,
                     std::vector<std::string> literals,
                     std::vector<std::string> tokens);

  bool nullable = false;
  bool unknown = false;
  std::vector<std::string> literals;
  std::vector<std::string> tokens;
  std::unordered_set<std::string_view> literalLookup;
  std::unordered_set<std::string_view> tokenLookup;

  void buildLookupTables();
  bool hasLiteral(std::string_view literal) const;
  bool hasToken(std::string_view token) const;
};

struct CpythonPegAlternative {
  CpythonPegExpr expr;
  std::string headRule;
  std::string action;
  std::vector<std::string> actionCallees;
  std::vector<std::string> astConstructors;
  CpythonPegFirstSet first;
  CpythonPegFirstSet recoveryFirst;
  std::vector<std::string> ruleRefs;
  std::vector<std::string> tokenRefs;
  std::vector<std::string> literalRefs;
};

struct CpythonPegRule {
  CpythonPegRule() = default;
  CpythonPegRule(std::string name, std::string returnType, bool memo,
                 bool invalid, CpythonPegExpr expr, CpythonPegFirstSet first,
                 CpythonPegFirstSet recoveryFirst,
                 std::vector<CpythonPegAlternative> alternatives);

  std::string name;
  std::string returnType;
  bool memo = false;
  bool invalid = false;
  CpythonPegExpr expr;
  CpythonPegFirstSet first;
  CpythonPegFirstSet recoveryFirst;
  std::vector<CpythonPegAlternative> alternatives;
  std::unordered_map<std::string_view, std::size_t> alternativeIndices;
  std::unordered_set<std::string_view> actionCalleeLookup;
  std::unordered_set<std::string_view> astConstructorLookup;
  std::unordered_set<std::string_view> literalRefLookup;
  std::vector<std::string> cachedUniqueAstConstructors;
  std::optional<std::string> cachedSingleAstConstructor;

  void buildLookupTables();
  const CpythonPegAlternative *
  findAlternativeByHead(std::string_view head) const;
  bool callsAction(std::string_view callee) const;
  bool hasAstConstructor(std::string_view constructor) const;
  bool referencesLiteral(std::string_view literal) const;
  std::vector<std::string> uniqueAstConstructors() const;
  std::optional<std::string> singleAstConstructor() const;
};

struct CpythonPegGrammar {
  std::vector<CpythonPegRule> rules;
  std::map<std::string, std::size_t> ruleIndices;
  std::vector<std::string> diagnostics;
  std::unordered_map<std::string_view, std::size_t> ruleLookup;

  void buildLookupTables();
  const CpythonPegRule *findRule(std::string_view name) const;
};

CpythonPegGrammar parseCpythonPegGrammar(std::string_view grammar);

} // namespace lython::parser
