#pragma once

#include "Grammar.h"

#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace lython::parser {

struct TokenSpec {
  int id = -1;
  std::string name;
  std::string spelling;
};

struct GeneratedKeywordSpec {
  std::string text;
  int tokenId = -1;
};

struct GeneratedTokenRefSpec {
  std::string text;
  int tokenId = -1;
};

struct GeneratedRuleSpec {
  std::string name;
  std::string functionName;
  std::string returnType;
  int typeId = -1;
};

struct AstFieldSpec {
  std::string name;
  std::string type;
  bool optional = false;
  bool sequence = false;
  bool nullableElement = false;
};

struct CpythonSpec {
  std::string root;
  std::vector<TokenSpec> tokens;
  std::vector<std::string> tokenNames;
  std::vector<std::string> hardKeywords;
  std::vector<std::string> softKeywords;
  std::vector<std::string> grammarHardKeywords;
  std::vector<std::string> grammarSoftKeywords;
  std::vector<std::string> generatedHardKeywords;
  std::vector<std::string> generatedSoftKeywords;
  std::vector<GeneratedKeywordSpec> generatedHardKeywordSpecs;
  std::vector<GeneratedTokenRefSpec> generatedTokenRefs;
  std::map<std::string, int> generatedHardKeywordTokenIdByText;
  std::vector<std::string> operatorSpellings;
  std::map<std::string, std::string> tokenNameBySpelling;
  std::map<std::string, int> tokenIdByName;
  std::map<std::string, int> tokenIdBySpelling;
  std::vector<std::string> astNodeKinds;
  std::map<std::string, std::string> astKindTypes;
  std::map<std::string, std::vector<std::string>> astFields;
  std::map<std::string, std::vector<AstFieldSpec>> astFieldSpecs;
  std::string grammar;
  std::string asdl;
  std::string generatedParser;
  std::vector<GeneratedRuleSpec> generatedRules;
  std::map<std::string, std::size_t> generatedRuleIndices;
  std::map<std::string, int> generatedRuleTypeIds;
  CpythonPegGrammar peg;
};

const CpythonSpec &cpython314Spec();
bool isCpythonHardKeyword(std::string_view text);
bool isCpythonSoftKeyword(std::string_view text);
bool isCpythonOperator(std::string_view text);
std::optional<std::string_view>
cpythonLongestOperatorPrefix(std::string_view text);
bool isCpythonTokenName(std::string_view text);
std::optional<std::string> cpythonTokenNameForSpelling(std::string_view text);
std::optional<int> cpythonTokenIdForName(std::string_view text);
std::optional<int> cpythonTokenIdForSpelling(std::string_view text);
std::optional<int> cpythonHardKeywordTokenId(std::string_view text);
bool isCpythonAstNodeKind(std::string_view text);
bool isCpythonAstField(std::string_view kind, std::string_view field);
const AstFieldSpec *cpythonAstFieldSpec(std::string_view kind,
                                        std::string_view field);
const std::vector<AstFieldSpec> *cpythonAstFieldSpecs(std::string_view kind);
std::optional<std::size_t> cpythonAstFieldIndex(std::string_view kind,
                                                std::string_view field);
bool isCpythonAstKindOfType(std::string_view kind, std::string_view type);
const GeneratedRuleSpec *cpythonGeneratedRule(std::string_view name);

} // namespace lython::parser
