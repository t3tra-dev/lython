#include "EmitterCore.h"
#include "EmitterPyOps.h"
#include "EmitterSupport.h"

#include "AstAccess.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"

namespace lython::emitter {

void ModuleEmitter::emitMatch(const parser::Node &statement) {
  const parser::Node *subjectNode = ast::node(statement, "subject");
  const auto *cases = ast::nodeList(statement, "cases");
  if (!subjectNode || !cases || cases->empty()) {
    diagnostics.push_back(parser::Diagnostic{parser::Severity::Error,
                                             statement.range.start,
                                             "empty match is not supported"});
    return;
  }
  Value subject = emitExpr(subjectNode);

  mlir::Block *entry = builder.getInsertionBlock();
  mlir::Region *region = entry->getParent();
  mlir::Block *continuation = entry->splitBlock(builder.getInsertionPoint());
  mlir::Block *check = builder.createBlock(region, continuation->getIterator());
  builder.setInsertionPointToEnd(entry);
  mlir::cf::BranchOp::create(builder, loc(statement), check);

  // Equality test `subject == <constant node>` yielding an i1 condition.
  auto equalsConstant = [&](const parser::Node &anchor,
                            const parser::Node *valueNode) -> mlir::Value {
    Value patternValue = emitExpr(valueNode);
    Value compared = emitBinarySpecial<py::EqOp>(anchor, "__eq__", subject,
                                                 patternValue, types.boolType());
    return emitBoolValue(compared, anchor);
  };

  bool matchedAll = false;
  // Flow-sensitive subject narrowing across the case chain: after a failed
  // union-member class test, the remaining cases see the union minus that
  // member, so the final member's class pattern becomes irrefutable and the
  // chain provably terminates (no fall-through path).
  mlir::Type matchSubjectType = subject.type;
  for (const parser::NodePtr &caseNodePtr : *cases) {
    if (!caseNodePtr)
      continue;
    const parser::Node &caseNode = *caseNodePtr;
    const parser::Node *pattern = ast::node(caseNode, "pattern");
    const parser::Node *guard = ast::node(caseNode, "guard");
    const auto *body = ast::nodeList(caseNode, "body");
    if (!pattern) {
      diagnostics.push_back(parser::Diagnostic{parser::Severity::Error,
                                               statement.range.start,
                                               "match case has no pattern"});
      return;
    }

    ScopedEmitterScope scope(values, types);
    builder.setInsertionPointToStart(check);

    // A nullopt condition means the pattern is irrefutable; unsupported
    // pattern kinds are rejected below with a diagnostic instead of silently
    // falling through.
    std::optional<mlir::Value> condition;
    bool unsupported = false;
    bool staticallyFalse = false;
    if (pattern->kind == "MatchAs" && !ast::node(*pattern, "pattern")) {
      if (std::optional<std::string_view> name =
              ast::string(*pattern, "name")) {
        values[std::string(*name)] = subject;
        types.bindSymbol(*name, subject.type);
      }
    } else if (pattern->kind == "MatchValue") {
      condition = equalsConstant(*pattern, ast::node(*pattern, "value"));
    } else if (pattern->kind == "MatchSingleton" &&
               ast::isNoneField(*pattern, "value")) {
      // `case None:` — identity test against the None singleton.
      if (auto unionType =
              mlir::dyn_cast_if_present<py::UnionType>(subject.type)) {
        if (unionType.hasMember(types.none()))
          condition = py::UnionTestOp::create(
                          builder, loc(statement), builder.getI1Type(),
                          subject.value, mlir::TypeAttr::get(types.none()))
                          .getResult();
        else
          unsupported = true;
      } else if (subject.type == types.none()) {
        // Subject is always None: irrefutable (condition stays nullopt).
      } else {
        unsupported = true;
      }
    } else if (pattern->kind == "MatchSingleton") {
      // `case True:` / `case False:` — use the subject's truthiness (its
      // runtime `__eq__` is not available). Only sound for a bool subject,
      // where the truth value distinguishes the two singletons; for other
      // subjects `case True` means `== 1`, which truthiness does not capture.
      std::optional<bool> flag = ast::boolean(*pattern, "value");
      if (flag && subject.type == types.boolType()) {
        mlir::Value truth = emitBoolValue(subject, *pattern);
        if (*flag) {
          condition = truth;
        } else {
          mlir::Value one =
              mlir::arith::ConstantIntOp::create(builder, loc(statement), 1, 1);
          condition =
              mlir::arith::XOrIOp::create(builder, loc(statement), truth, one)
                  .getResult();
        }
      } else {
        unsupported = true;
      }
    } else if (pattern->kind == "MatchOr") {
      const auto *alts = ast::nodeList(*pattern, "patterns");
      if (!alts || alts->empty()) {
        unsupported = true;
      } else {
        for (const parser::NodePtr &alt : *alts) {
          if (!alt || alt->kind != "MatchValue") {
            unsupported = true;
            break;
          }
          mlir::Value altCond = equalsConstant(*alt, ast::node(*alt, "value"));
          condition = condition ? mlir::arith::OrIOp::create(
                                      builder, loc(statement), *condition,
                                      altCond)
                                      .getResult()
                                : altCond;
        }
      }
    } else if (pattern->kind == "MatchSequence") {
      // Sequence destructuring over a tuple/list subject. A sequence pattern
      // is a runtime length test (`len(subject) == N`) guarding per-element
      // extraction; element getitems are emitted only behind the length gate
      // so a shorter subject never reaches an out-of-range access.
      const auto *subPatterns = ast::nodeList(*pattern, "patterns");
      auto contract =
          mlir::dyn_cast_if_present<py::ContractType>(subject.type);
      bool sequenceSubject =
          contract && (contract.getContractName() == "builtins.tuple" ||
                       contract.getContractName() == "builtins.list");
      bool shapeSupported = sequenceSubject && subPatterns && !guard;
      constexpr unsigned kNoStar = ~0u;
      unsigned starIndex = kNoStar;
      if (shapeSupported)
        for (auto [subIndex, subPattern] : llvm::enumerate(*subPatterns)) {
          bool captureLike = subPattern && subPattern->kind == "MatchAs" &&
                             !ast::node(*subPattern, "pattern");
          bool literalLike = subPattern && subPattern->kind == "MatchValue";
          bool starLike = subPattern && subPattern->kind == "MatchStar";
          if (starLike) {
            // One star, and only in the trailing position for now.
            if (starIndex != kNoStar || subIndex + 1 != subPatterns->size()) {
              shapeSupported = false;
              break;
            }
            starIndex = static_cast<unsigned>(subIndex);
            continue;
          }
          if (!captureLike && !literalLike) {
            shapeSupported = false;
            break;
          }
        }
      if (!shapeSupported) {
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, statement.range.start,
            "match sequence pattern requires a tuple/list subject with "
            "capture or literal elements (one trailing *rest allowed; guards "
            "not supported here yet)"});
        return;
      }
      unsigned prefixCount = starIndex == kNoStar
                                 ? static_cast<unsigned>(subPatterns->size())
                                 : starIndex;

      CallInferenceResult lenInference =
          types.inferMethodCallWithEvidence(subject.type, "__len__", {});
      if (!lenInference) {
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, statement.range.start,
            "match sequence pattern subject has no __len__ evidence"});
        return;
      }
      Value length{py::LenOp::create(
                       builder, loc(statement), lenInference.resultType,
                       mlir::FlatSymbolRefAttr::get(&context, "__len__"),
                       callProtocolFor(lenInference), subject.value)
                       .getResult(),
                   lenInference.resultType};
      std::string arityText = std::to_string(prefixCount);
      mlir::Type arityType = types.literal(arityText);
      Value arity{py::IntConstantOp::create(builder, loc(statement), arityType,
                                            builder.getStringAttr(arityText))
                      .getResult(),
                  arityType};
      Value lengthCompared =
          starIndex == kNoStar
              ? emitBinarySpecial<py::EqOp>(*pattern, "__eq__", length, arity,
                                            types.boolType())
              : emitBinarySpecial<py::GeOp>(*pattern, "__ge__", length, arity,
                                            types.boolType());
      mlir::Value lengthMatches = emitBoolValue(lengthCompared, *pattern);

      mlir::Block *elementBlock =
          builder.createBlock(region, continuation->getIterator());
      mlir::Block *nextCheck =
          builder.createBlock(region, continuation->getIterator());
      builder.setInsertionPointToEnd(check);
      mlir::cf::CondBranchOp::create(builder, loc(statement), lengthMatches,
                                     elementBlock, mlir::ValueRange{},
                                     nextCheck, mlir::ValueRange{});

      builder.setInsertionPointToStart(elementBlock);
      auto sequenceElement = [&](unsigned index) -> std::optional<Value> {
        std::string text = std::to_string(index);
        mlir::Type literalType = types.literal(text);
        Value indexValue{
            py::IntConstantOp::create(builder, loc(statement), literalType,
                                      builder.getStringAttr(text))
                .getResult(),
            literalType};
        CallInferenceResult inference = types.inferMethodCallWithEvidence(
            subject.type, "__getitem__", {indexValue.type});
        if (!inference)
          return std::nullopt;
        auto op = py::GetItemOp::create(
            builder, loc(statement), inference.resultType,
            mlir::FlatSymbolRefAttr::get(&context, "__getitem__"),
            callProtocolFor(inference), subject.value, indexValue.value);
        return Value{op.getResult(), inference.resultType};
      };
      std::optional<mlir::Value> elementCondition;
      bool elementsSupported = true;
      for (auto [index, subPattern] : llvm::enumerate(*subPatterns)) {
        if (subPattern->kind == "MatchStar")
          continue; // handled below
        if (subPattern->kind == "MatchAs") {
          std::optional<std::string_view> name =
              ast::string(*subPattern, "name");
          if (!name)
            continue; // wildcard element
          std::optional<Value> element =
              sequenceElement(static_cast<unsigned>(index));
          if (!element) {
            elementsSupported = false;
            break;
          }
          values[std::string(*name)] = *element;
          types.bindSymbol(*name, element->type);
          continue;
        }
        std::optional<Value> element =
            sequenceElement(static_cast<unsigned>(index));
        if (!element) {
          elementsSupported = false;
          break;
        }
        Value patternValue = emitExpr(ast::node(*subPattern, "value"));
        Value compared = emitBinarySpecial<py::EqOp>(
            *subPattern, "__eq__", *element, patternValue, types.boolType());
        mlir::Value elementCond = emitBoolValue(compared, *subPattern);
        elementCondition =
            elementCondition
                ? mlir::arith::AndIOp::create(builder, loc(statement),
                                              *elementCondition, elementCond)
                      .getResult()
                : elementCond;
      }
      if (!elementsSupported) {
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, statement.range.start,
            "match sequence pattern element has no __getitem__ evidence"});
        return;
      }
      if (starIndex != kNoStar) {
        // `*rest` materializes the remaining elements as a fresh list via a
        // synthetic build loop; `*_` needs no materialization (the >= length
        // gate is the whole check).
        std::optional<std::string_view> starName =
            ast::string(*(*subPatterns)[starIndex], "name");
        if (starName) {
          CallInferenceResult getInference = types.inferMethodCallWithEvidence(
              subject.type, "__getitem__", {types.contract("builtins.int")});
          if (!getInference) {
            diagnostics.push_back(parser::Diagnostic{
                parser::Severity::Error, statement.range.start,
                "match sequence *rest requires runtime-index __getitem__ "
                "evidence"});
            return;
          }
          mlir::Type elementType = types.widenLiteral(getInference.resultType);
          mlir::Type restType = py::ContractType::get(
              builder.getContext(), "builtins.list", {elementType});
          std::string subjLocal =
              "__matchseq" + std::to_string(++listCompCounter);
          std::string restLocal =
              "__matchrest" + std::to_string(listCompCounter);
          std::string idxLocal = "__matchidx" + std::to_string(listCompCounter);
          values[subjLocal] = subject;
          types.bindSymbol(subjLocal, subject.type);
          auto packRest = py::PackOp::create(builder, loc(statement), restType,
                                             mlir::ValueRange{});
          values[restLocal] = Value{packRest.getResult(), restType};
          types.bindSymbol(restLocal, restType);
          auto nameNode = [&](const std::string &id) {
            parser::NodePtr node = parser::makeNode("Name", statement.range);
            parser::addField(*node, "id", id);
            return node;
          };
          // for __idx in range(<prefix>, len(__subj)):
          //   __rest.append(__subj[__idx])
          parser::NodePtr prefixNode =
              parser::makeNode("Constant", statement.range);
          parser::addField(*prefixNode, "value",
                           static_cast<std::int64_t>(prefixCount));
          parser::NodePtr lenCall = parser::makeNode("Call", statement.range);
          parser::addField(*lenCall, "func", nameNode("len"));
          parser::addField(*lenCall, "args",
                           std::vector<parser::NodePtr>{nameNode(subjLocal)});
          parser::addField(*lenCall, "keywords",
                           std::vector<parser::NodePtr>{});
          parser::NodePtr rangeCall = parser::makeNode("Call", statement.range);
          parser::addField(*rangeCall, "func", nameNode("range"));
          parser::addField(*rangeCall, "args",
                           std::vector<parser::NodePtr>{prefixNode, lenCall});
          parser::addField(*rangeCall, "keywords",
                           std::vector<parser::NodePtr>{});
          parser::NodePtr subscript =
              parser::makeNode("Subscript", statement.range);
          parser::addField(*subscript, "value", nameNode(subjLocal));
          parser::addField(*subscript, "slice", nameNode(idxLocal));
          parser::NodePtr appendAttr =
              parser::makeNode("Attribute", statement.range);
          parser::addField(*appendAttr, "value", nameNode(restLocal));
          parser::addField(*appendAttr, "attr", std::string("append"));
          parser::NodePtr appendCall =
              parser::makeNode("Call", statement.range);
          parser::addField(*appendCall, "func", appendAttr);
          parser::addField(*appendCall, "args",
                           std::vector<parser::NodePtr>{subscript});
          parser::addField(*appendCall, "keywords",
                           std::vector<parser::NodePtr>{});
          parser::NodePtr appendStmt =
              parser::makeNode("Expr", statement.range);
          parser::addField(*appendStmt, "value", appendCall);
          parser::NodePtr buildLoop =
              parser::makeNode("For", statement.range);
          parser::addField(*buildLoop, "target", nameNode(idxLocal));
          parser::addField(*buildLoop, "iter", rangeCall);
          parser::addField(*buildLoop, "body",
                           std::vector<parser::NodePtr>{appendStmt});
          parser::addField(*buildLoop, "orelse",
                           std::vector<parser::NodePtr>{});
          emitFor(*buildLoop);
          auto builtRest = values.find(restLocal);
          if (builtRest != values.end() && builtRest->second.value) {
            values[std::string(*starName)] = builtRest->second;
            types.bindSymbol(*starName, builtRest->second.type);
          }
          values.erase(restLocal);
          values.erase(subjLocal);
          values.erase(idxLocal);
        }
      }
      if (elementCondition) {
        mlir::Block *conditionBlock = builder.getInsertionBlock();
        mlir::Block *bodyBlock =
            builder.createBlock(region, continuation->getIterator());
        builder.setInsertionPointToEnd(conditionBlock);
        mlir::cf::CondBranchOp::create(builder, loc(statement),
                                       *elementCondition, bodyBlock,
                                       mlir::ValueRange{}, nextCheck,
                                       mlir::ValueRange{});
        builder.setInsertionPointToStart(bodyBlock);
      }
      emitStatements(body);
      if (!insertionBlockTerminated(builder))
        mlir::cf::BranchOp::create(builder, loc(statement), continuation);
      check = nextCheck;
      continue;
    } else if (pattern->kind == "MatchClass") {
      // Class pattern over a statically resolved class: reuses the isinstance
      // evidence analysis (union member test / subclass test), then binds
      // attribute captures and evaluates literal sub-pattern equalities from
      // the narrowed value inside the gated block. Positional sub-patterns
      // resolve their attribute names through the class's __match_args__.
      const parser::Node *clsNode = ast::node(*pattern, "cls");
      const auto *positionalSubs = ast::nodeList(*pattern, "patterns");
      const auto *kwdAttrs = ast::stringList(*pattern, "kwd_attrs");
      const auto *kwdPatterns = ast::nodeList(*pattern, "kwd_patterns");
      auto supportedSubPattern = [](const parser::NodePtr &sub) {
        if (!sub)
          return false;
        if (sub->kind == "MatchAs")
          return ast::node(*sub, "pattern") == nullptr;
        return sub->kind == "MatchValue";
      };
      bool shapeSupported =
          !guard && ((kwdAttrs == nullptr) == (kwdPatterns == nullptr));
      std::size_t keywordCount = kwdAttrs ? kwdAttrs->size() : 0;
      if (shapeSupported && kwdPatterns) {
        if (kwdPatterns->size() != keywordCount)
          shapeSupported = false;
        else
          for (const parser::NodePtr &sub : *kwdPatterns)
            if (!supportedSubPattern(sub)) {
              shapeSupported = false;
              break;
            }
      }
      if (shapeSupported && positionalSubs)
        for (const parser::NodePtr &sub : *positionalSubs)
          if (!supportedSubPattern(sub)) {
            shapeSupported = false;
            break;
          }
      std::optional<mlir::Type> target =
          shapeSupported ? isinstanceTargetType(clsNode, types) : std::nullopt;
      if (!shapeSupported || !target) {
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, statement.range.start,
            "match class pattern requires a statically resolved class with "
            "capture or literal sub-patterns (no guards)"});
        return;
      }
      // (attribute name, sub-pattern) pairs: positional names resolve through
      // the class's __match_args__ tuple, keyword names are explicit.
      llvm::SmallVector<std::pair<std::string, const parser::Node *>, 4>
          attrPatterns;
      if (positionalSubs && !positionalSubs->empty()) {
        std::optional<std::vector<std::string>> matchArgs =
            types.classMatchArgs(*target);
        if (!matchArgs || positionalSubs->size() > matchArgs->size()) {
          diagnostics.push_back(parser::Diagnostic{
              parser::Severity::Error, statement.range.start,
              "match class positional patterns require a __match_args__ "
              "string-literal tuple with at least as many names"});
          return;
        }
        for (auto [index, sub] : llvm::enumerate(*positionalSubs))
          attrPatterns.push_back({(*matchArgs)[index], sub.get()});
      }
      for (std::size_t index = 0; index < keywordCount; ++index)
        attrPatterns.push_back(
            {std::string((*kwdAttrs)[index]), (*kwdPatterns)[index].get()});
      IsInstanceAnalysis analysis =
          analyzeIsInstance(matchSubjectType, *target, types, module);
      if (analysis.kind == IsInstanceAnalysis::Kind::Unsupported ||
          analysis.kind == IsInstanceAnalysis::Kind::UnionClassTest ||
          (analysis.kind == IsInstanceAnalysis::Kind::UnionTest &&
           !analysis.trueType)) {
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, statement.range.start,
            analysis.failureReason.empty()
                ? "match class pattern has unsupported isinstance evidence"
                : analysis.failureReason});
        return;
      }
      if (analysis.kind == IsInstanceAnalysis::Kind::AlwaysFalse)
        continue; // statically impossible case

      mlir::Block *matchBlock = nullptr;
      mlir::Block *nextCheck = nullptr;
      if (analysis.kind != IsInstanceAnalysis::Kind::AlwaysTrue) {
        mlir::Value bit;
        if (analysis.kind == IsInstanceAnalysis::Kind::UnionTest) {
          bit = py::UnionTestOp::create(
                    builder, loc(statement), builder.getI1Type(), subject.value,
                    mlir::TypeAttr::get(analysis.trueType))
                    .getResult();
        } else { // ClassTest
          bit = py::ClassTestOp::create(
                    builder, loc(statement), builder.getI1Type(), subject.value,
                    mlir::TypeAttr::get(analysis.targetType))
                    .getResult();
        }
        matchBlock = builder.createBlock(region, continuation->getIterator());
        nextCheck = builder.createBlock(region, continuation->getIterator());
        builder.setInsertionPointToEnd(check);
        mlir::cf::CondBranchOp::create(builder, loc(statement), bit, matchBlock,
                                       mlir::ValueRange{}, nextCheck,
                                       mlir::ValueRange{});
        builder.setInsertionPointToStart(matchBlock);
      }

      Value narrowed = subject;
      if (analysis.kind == IsInstanceAnalysis::Kind::UnionTest) {
        auto unwrap = py::UnionUnwrapOp::create(builder, loc(statement),
                                                analysis.trueType,
                                                subject.value);
        narrowed = Value{unwrap.getResult(), analysis.trueType};
        if (analysis.trueType != analysis.targetType &&
            mlir::isa<py::ContractType>(analysis.trueType) &&
            mlir::isa<py::ContractType>(analysis.targetType) &&
            py::isAssignableTo(analysis.targetType, analysis.trueType,
                               module)) {
          auto refine = py::ClassRefineOp::create(
              builder, loc(statement), analysis.targetType, narrowed.value);
          narrowed = Value{refine.getResult(), analysis.targetType};
        }
      } else if (analysis.kind == IsInstanceAnalysis::Kind::ClassTest) {
        auto refine = py::ClassRefineOp::create(
            builder, loc(statement), analysis.targetType, subject.value);
        narrowed = Value{refine.getResult(), analysis.targetType};
      } else if (analysis.kind == IsInstanceAnalysis::Kind::AlwaysTrue &&
                 mlir::isa<py::UnionType>(subject.value.getType()) &&
                 mlir::isa<py::ContractType>(matchSubjectType)) {
        // The chain narrowed the subject to a single union member; the SSA
        // value is still union-shaped, so extract the member payload.
        auto unwrap = py::UnionUnwrapOp::create(builder, loc(statement),
                                                matchSubjectType,
                                                subject.value);
        narrowed = Value{unwrap.getResult(), matchSubjectType};
      }

      bool capturesSupported = true;
      std::optional<mlir::Value> valueCondition;
      for (auto &[attrName, sub] : attrPatterns) {
        std::optional<mlir::Type> field =
            lookupClassField(narrowed.type, attrName);
        if (!field) {
          capturesSupported = false;
          break;
        }
        bool isCapture = sub->kind == "MatchAs";
        std::optional<std::string_view> captureName =
            isCapture ? ast::string(*sub, "name") : std::nullopt;
        if (isCapture && !captureName)
          continue; // wildcard positional: field existence is the only check
        auto attrGet = py::AttrGetOp::create(
            builder, loc(statement), *field, narrowed.value, attrName);
        attrGet->setAttr("ly.attr.kind", builder.getStringAttr("field"));
        if (auto contract =
                mlir::dyn_cast_if_present<py::ContractType>(narrowed.type))
          attrGet->setAttr("ly.attr.owner",
                           builder.getStringAttr(contract.getContractName()));
        if (isCapture) {
          values[std::string(*captureName)] = Value{attrGet.getResult(), *field};
          types.bindSymbol(*captureName, *field);
          continue;
        }
        // MatchValue: gate the case body on attribute equality.
        Value element{attrGet.getResult(), *field};
        Value patternValue = emitExpr(ast::node(*sub, "value"));
        Value compared = emitBinarySpecial<py::EqOp>(
            *sub, "__eq__", element, patternValue, types.boolType());
        mlir::Value condition = emitBoolValue(compared, *sub);
        valueCondition =
            valueCondition
                ? mlir::arith::AndIOp::create(builder, loc(statement),
                                              *valueCondition, condition)
                      .getResult()
                : condition;
      }
      if (!capturesSupported) {
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, statement.range.start,
            "match class pattern sub-pattern must name a declared field"});
        return;
      }

      bool refutableByValue = valueCondition.has_value();
      if (valueCondition) {
        mlir::Block *conditionBlock = builder.getInsertionBlock();
        if (!nextCheck)
          nextCheck = builder.createBlock(region, continuation->getIterator());
        mlir::Block *bodyBlock =
            builder.createBlock(region, continuation->getIterator());
        builder.setInsertionPointToEnd(conditionBlock);
        mlir::cf::CondBranchOp::create(builder, loc(statement),
                                       *valueCondition, bodyBlock,
                                       mlir::ValueRange{}, nextCheck,
                                       mlir::ValueRange{});
        builder.setInsertionPointToStart(bodyBlock);
      }
      emitStatements(body);
      if (!insertionBlockTerminated(builder))
        mlir::cf::BranchOp::create(builder, loc(statement), continuation);
      if (!nextCheck) {
        // Irrefutable class pattern: terminates the chain.
        matchedAll = true;
        break;
      }
      // On the fall-through edge the tested member is excluded — but only
      // when falling through can only mean the class test failed (a value
      // inequality also falls through without excluding the member).
      if (analysis.kind == IsInstanceAnalysis::Kind::UnionTest &&
          analysis.falseType && !refutableByValue)
        matchSubjectType = analysis.falseType;
      check = nextCheck;
      continue;
    } else if (pattern->kind == "MatchMapping") {
      // Mapping pattern over a dict subject: a `key in subject` test per
      // pattern key guards the value extraction, so a missing key is a
      // non-match (never a KeyError).
      const auto *keys = ast::nodeList(*pattern, "keys");
      const auto *valuePatterns = ast::nodeList(*pattern, "patterns");
      auto contract =
          mlir::dyn_cast_if_present<py::ContractType>(subject.type);
      bool shapeSupported = contract &&
                            contract.getContractName() == "builtins.dict" &&
                            keys && valuePatterns &&
                            keys->size() == valuePatterns->size() && !guard &&
                            !ast::node(*pattern, "rest");
      if (shapeSupported)
        for (const parser::NodePtr &sub : *valuePatterns) {
          bool captureLike = sub && sub->kind == "MatchAs" &&
                             !ast::node(*sub, "pattern");
          bool literalLike = sub && sub->kind == "MatchValue";
          if (!captureLike && !literalLike) {
            shapeSupported = false;
            break;
          }
        }
      if (!shapeSupported) {
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, statement.range.start,
            "match mapping pattern requires a dict subject with capture or "
            "literal values (no rest/guard)"});
        return;
      }

      // Stage 1: presence conditions in the check block.
      llvm::SmallVector<Value, 4> keyValues;
      std::optional<mlir::Value> present;
      for (const parser::NodePtr &keyNode : *keys) {
        Value key = emitExpr(keyNode.get());
        keyValues.push_back(key);
        CallInferenceResult inference = types.inferMethodCallWithEvidence(
            subject.type, "__contains__", {key.type});
        if (!inference) {
          diagnostics.push_back(parser::Diagnostic{
              parser::Severity::Error, statement.range.start,
              "match mapping subject has no __contains__ evidence"});
          return;
        }
        auto contains = py::ContainsOp::create(
            builder, loc(statement), builder.getI1Type(),
            mlir::FlatSymbolRefAttr::get(&context, "__contains__"),
            callProtocolFor(inference), subject.value, key.value);
        present = present ? mlir::arith::AndIOp::create(
                                builder, loc(statement), *present,
                                contains.getResult())
                                .getResult()
                          : contains.getResult();
      }

      mlir::Block *valueBlock =
          builder.createBlock(region, continuation->getIterator());
      mlir::Block *nextCheck =
          builder.createBlock(region, continuation->getIterator());
      builder.setInsertionPointToEnd(check);
      if (present) {
        mlir::cf::CondBranchOp::create(builder, loc(statement), *present,
                                       valueBlock, mlir::ValueRange{},
                                       nextCheck, mlir::ValueRange{});
      } else {
        mlir::cf::BranchOp::create(builder, loc(statement), valueBlock);
      }

      // Stage 2: gated value extraction, capture binds, literal compares.
      builder.setInsertionPointToStart(valueBlock);
      std::optional<mlir::Value> valueCondition;
      for (auto [index, sub] : llvm::enumerate(*valuePatterns)) {
        Value key = keyValues[index];
        CallInferenceResult inference = types.inferMethodCallWithEvidence(
            subject.type, "__getitem__", {key.type});
        if (!inference) {
          diagnostics.push_back(parser::Diagnostic{
              parser::Severity::Error, statement.range.start,
              "match mapping subject has no __getitem__ evidence"});
          return;
        }
        auto item = py::GetItemOp::create(
            builder, loc(statement), inference.resultType,
            mlir::FlatSymbolRefAttr::get(&context, "__getitem__"),
            callProtocolFor(inference), subject.value, key.value);
        Value element{item.getResult(), inference.resultType};
        if (sub->kind == "MatchAs") {
          if (std::optional<std::string_view> name = ast::string(*sub, "name")) {
            values[std::string(*name)] = element;
            types.bindSymbol(*name, element.type);
          }
          continue;
        }
        Value patternValue = emitExpr(ast::node(*sub, "value"));
        Value compared = emitBinarySpecial<py::EqOp>(
            *sub, "__eq__", element, patternValue, types.boolType());
        mlir::Value bit = emitBoolValue(compared, *sub);
        valueCondition = valueCondition
                             ? mlir::arith::AndIOp::create(
                                   builder, loc(statement), *valueCondition, bit)
                                   .getResult()
                             : bit;
      }
      if (valueCondition) {
        mlir::Block *conditionBlock = builder.getInsertionBlock();
        mlir::Block *bodyBlock =
            builder.createBlock(region, continuation->getIterator());
        builder.setInsertionPointToEnd(conditionBlock);
        mlir::cf::CondBranchOp::create(builder, loc(statement),
                                       *valueCondition, bodyBlock,
                                       mlir::ValueRange{}, nextCheck,
                                       mlir::ValueRange{});
        builder.setInsertionPointToStart(bodyBlock);
      }
      emitStatements(body);
      if (!insertionBlockTerminated(builder))
        mlir::cf::BranchOp::create(builder, loc(statement), continuation);
      check = nextCheck;
      continue;
    } else {
      unsupported = true;
    }
    if (staticallyFalse)
      continue;
    if (unsupported) {
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, statement.range.start,
          "match pattern '" + pattern->kind + "' is not implemented yet"});
      return;
    }

    // A guard makes even an irrefutable pattern refutable.
    if (guard) {
      mlir::Value guardCond = emitBoolValue(emitExpr(guard), *guard);
      condition = condition ? mlir::arith::AndIOp::create(
                                  builder, loc(statement), *condition, guardCond)
                                  .getResult()
                            : guardCond;
    }

    if (!condition) {
      // Irrefutable: emit the body and terminate the chain.
      emitStatements(body);
      if (!insertionBlockTerminated(builder))
        mlir::cf::BranchOp::create(builder, loc(statement), continuation);
      matchedAll = true;
      break;
    }

    mlir::Block *bodyBlock =
        builder.createBlock(region, continuation->getIterator());
    mlir::Block *nextCheck =
        builder.createBlock(region, continuation->getIterator());
    builder.setInsertionPointToEnd(check);
    mlir::cf::CondBranchOp::create(builder, loc(statement), *condition,
                                   bodyBlock, mlir::ValueRange{}, nextCheck,
                                   mlir::ValueRange{});
    builder.setInsertionPointToStart(bodyBlock);
    emitStatements(body);
    if (!insertionBlockTerminated(builder))
      mlir::cf::BranchOp::create(builder, loc(statement), continuation);
    check = nextCheck;
  }

  // No irrefutable case matched: fall through to the continuation.
  if (!matchedAll) {
    builder.setInsertionPointToEnd(check);
    if (!insertionBlockTerminated(builder))
      mlir::cf::BranchOp::create(builder, loc(statement), continuation);
  }
  builder.setInsertionPointToStart(continuation);
}


} // namespace lython::emitter
