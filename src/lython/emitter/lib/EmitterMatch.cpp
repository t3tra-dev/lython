#include "AstSynth.h"
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

#include <cstddef>

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
  // Every case body is an arm of one statement, and a name any of them binds
  // is a local of the scope around the match -- the same rule an `if` chain
  // follows.
  llvm::SmallVector<std::pair<std::string, Value>, 4> promotedCells;
  {
    llvm::SmallVector<const std::vector<parser::NodePtr> *, 4> bodies;
    for (const parser::NodePtr &matchCase : *cases)
      if (matchCase)
        bodies.push_back(ast::nodeList(*matchCase, "body"));
    bindConditionallyAssignedLocals(statement, bodies);

    // ⭐ A LOCAL BOUND BEFORE THE MATCH AND REASSIGNED INSIDE A CASE KEPT ITS
    // OLD VALUE:
    //
    //     def f(n: int) -> int:
    //         total = 100
    //         match n:
    //             case 0:
    //                 total = 1
    //         return total
    //     print(f(0))       # printed 100; CPython prints 1
    //
    // A silent wrong answer for every assignment in a case body -- and the
    // same program written with `if` is right. Each case is emitted in its own
    // scope, because a value defined in one case's block does not dominate the
    // next case's, and the assignment therefore landed in a scope that is
    // popped when the case ends.
    //
    // A name bound ONLY inside the match already worked, because
    // `bindConditionallyAssignedLocals` gives that one a CELL -- storage the
    // arms write through. The name that was already bound is skipped there (it
    // has a binding), so it kept plain SSA and had nowhere to put the write.
    // It gets the same storage here.
    //
    // ⛔ NOT continuation block arguments, which is what `if` threads. That
    // machinery balances ownership tokens across edges that assign and edges
    // that do not, and a match has one continuation with an arm per case plus
    // a fall-through; the cell is the mechanism this statement already uses
    // for the name one line over, and it carries the tokens itself.
    //
    // ⛔ ONLY A NAME SOMETHING LATER READS, the same rule (and the same
    // same-suite limitation) `bindConditionallyAssignedLocals` documents: a
    // write nobody reads needs no storage, and giving one to every name a case
    // assigns changes the representation of code that already works.
    // ⛔ NOT WHEN A CASE JUMPS OUT OF AN ENCLOSING LOOP. A `continue` leaves
    // the match without passing the continuation that loads the cell back,
    // so the loop's carried local would be handed the CELL where its own
    // block argument is the content -- "cannot adapt runtime bundle
    // __ly_cell$2 ... to expected ABI". The shape keeps the old answer and is
    // recorded in tests/probe/wb_a_match_case_that_jumps_out_of_a_loop.py.
    llvm::StringRef jumpKinds[] = {"Break", "Continue"};
    bool jumpsOut = false;
    for (const std::vector<parser::NodePtr> *body : bodies)
      jumpsOut = jumpsOut ||
                 containsStatementKind(body, jumpKinds, /*stopAtLoops=*/true);
    llvm::StringSet<> assigned;
    for (const std::vector<parser::NodePtr> *body : bodies)
      collectAssignedNames(body, assigned);
    llvm::SmallVector<std::string, 4> promoted;
    for (const auto &entry : assigned) {
      llvm::StringRef name = entry.getKey();
      auto bound = values.find(name);
      if (bound == values.end() || isCellContract(bound->second.type))
        continue;
      // ⛔ The CONSERVATIVE reader, not the same-suite one
      // `bindConditionallyAssignedLocals` uses. That one cannot see a read in
      // an enclosing suite, which is exactly where the commonest shape puts it
      // -- a match inside a loop whose accumulator is read after the loop --
      // and here the cheap direction is safe: the cell is loaded back into SSA
      // at the continuation, so promoting a name nobody reads costs a heap
      // slot and changes nothing.
      if (!nameMayBeReadAfterCurrentStatement(name))
        continue;
      promoted.push_back(name.str());
    }
    llvm::sort(promoted);
    // ⛔ AND REFUSED, not silently dropped, when a sibling case leaves the
    // loop. A `continue` exits the match without passing the continuation
    // that loads the cell back, so the loop's carried local would be handed
    // the CELL where its block argument is the content. The name keeps the
    // value it had before the match otherwise -- the same wrong answer this
    // whole block exists to remove -- so the shape says so instead.
    //
    // ⛔ A name the enclosing `try` already promoted is not a candidate (it is
    // cell-backed before this runs), which is why
    // tests/golden/cases/a_jump_out_of_a_match_inside_a_try.py keeps working:
    // that mechanism carries the write across the jump on its own.
    if (jumpsOut && !promoted.empty()) {
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, statement.range.start,
          "local '" + promoted.front() +
              "' is reassigned in a match case while another case leaves the "
              "loop, and the write cannot reach that jump: bind the result to "
              "a new name inside the case, or put the match in a `try`"});
      return;
    }
    for (const std::string &name : promoted) {
      Value outer = values.find(name)->second;
      mlir::Type inner = inferConditionalLocalType(bodies, name);
      mlir::Type content = types.join(
          {types.widenLiteral(outer.type),
           inner ? types.widenLiteral(inner) : types.widenLiteral(outer.type)});
      // ⛔ The same storability rule the conditional-slot path ends with: a
      // cell whose content is the erased top accepts every write and refuses
      // every read, which trades a wrong answer for a lowering sentence.
      if (!mlir::isa_and_nonnull<py::ContractType>(content) ||
          py::isPyObjectType(content))
        continue;
      Value initial = coerceValue(outer, content, statement);
      if (initial.type != content)
        continue;
      Value cell = emitCellAlloc(statement, initial, /*tracksBinding=*/false);
      values[name] = cell;
      types.bindSymbol(name, content);
      promotedCells.emplace_back(name, cell);
    }
  }

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

    // The subject as the chain now knows it: a `case None:` above this one
    // (or a class test) has already ruled members out, and a capture that
    // bound the whole union would be refused for operators the remaining
    // member has. Emitted lazily so a case that captures nothing adds no op.
    std::optional<Value> narrowedSubject;
    auto capturedSubject = [&]() -> Value {
      if (narrowedSubject)
        return *narrowedSubject;
      narrowedSubject = subject;
      if (matchSubjectType != subject.type &&
          mlir::isa<py::UnionType>(subject.value.getType()) &&
          mlir::isa<py::ContractType>(matchSubjectType)) {
        auto unwrap = py::UnionUnwrapOp::create(builder, loc(statement),
                                                matchSubjectType,
                                                subject.value);
        narrowedSubject = Value{unwrap.getResult(), matchSubjectType};
      }
      return *narrowedSubject;
    };

    // ⭐ `<pattern> as <name>` BINDS THE WHOLE SUBJECT and then matches the
    // pattern -- two independent things the chain below can do separately, and
    // it refused the pair outright ("match pattern 'MatchAs' is not
    // implemented") while `case name:` and `case [x]:` each worked on their
    // own. The name is bound here, where the capture-only form binds it, and
    // the chain sees the inner pattern; the case's own scope keeps the binding
    // from outliving a case whose condition then fails.
    llvm::SmallVector<std::string, 2> capturedNames;
    while (pattern->kind == "MatchAs" && ast::node(*pattern, "pattern")) {
      if (std::optional<std::string_view> name =
              ast::string(*pattern, "name")) {
        Value bound = capturedSubject();
        values[std::string(*name)] = bound;
        types.bindSymbol(*name, bound.type);
        capturedNames.push_back(std::string(*name));
      }
      pattern = ast::node(*pattern, "pattern");
    }

    // A nullopt condition means the pattern is irrefutable; unsupported
    // pattern kinds are rejected below with a diagnostic instead of silently
    // falling through.
    std::optional<mlir::Value> condition;
    bool unsupported = false;
    bool staticallyFalse = false;
    if (pattern->kind == "MatchAs" && !ast::node(*pattern, "pattern")) {
      if (std::optional<std::string_view> name =
              ast::string(*pattern, "name")) {
        // ⭐ THE CAPTURE SEES WHAT THE CHAIN ALREADY RULED OUT. `case None:`
        // above this one excludes None from what can still arrive, and the
        // capture bound the WHOLE union anyway -- so `case n: return str(n +
        // 1)` on an `int | None` was refused for an operator the remaining
        // member has. The class arm already keeps this bookkeeping in
        // `matchSubjectType`; the captures are the other place that read it.
        Value captured = capturedSubject();
        values[std::string(*name)] = captured;
        types.bindSymbol(*name, captured.type);
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
      // ⭐ FALLING THROUGH THIS CASE MEANS THE SUBJECT IS NOT None, which is
      // the same bookkeeping the class arm keeps: without it every later case
      // still saw the None member and `case n: return str(n + 1)` was refused.
      if (!unsupported)
        if (auto unionType =
                mlir::dyn_cast_if_present<py::UnionType>(matchSubjectType)) {
          llvm::SmallVector<mlir::Type, 4> remaining;
          for (mlir::Type member : unionType.getMemberTypes())
            if (types.widenLiteral(member) != types.none())
              remaining.push_back(member);
          if (!remaining.empty() &&
              remaining.size() < unionType.getMemberTypes().size())
            matchSubjectType = types.join(remaining);
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
      bool shapeSupported = sequenceSubject && subPatterns;
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
            "capture or literal elements (one trailing *rest allowed)"});
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
            parser::NodePtr node = synth::name(id, statement.range);
            return node;
          };
          // for __idx in range(<prefix>, len(__subj)):
          //   __rest.append(__subj[__idx])
          parser::NodePtr prefixNode = synth::intConstant(static_cast<std::int64_t>(prefixCount), statement.range);
          parser::NodePtr lenCall = synth::call(nameNode("len"), std::vector<parser::NodePtr>{nameNode(subjLocal)}, statement.range);
          parser::NodePtr rangeCall = synth::call(nameNode("range"), std::vector<parser::NodePtr>{prefixNode, lenCall}, statement.range);
          parser::NodePtr subscript =
              parser::makeNode("Subscript", statement.range);
          parser::addField(*subscript, "value", nameNode(subjLocal));
          parser::addField(*subscript, "slice", nameNode(idxLocal));
          parser::NodePtr appendAttr = synth::attribute(nameNode(restLocal), std::string("append"), statement.range);
          parser::NodePtr appendCall = synth::call(appendAttr, std::vector<parser::NodePtr>{subscript}, statement.range);
          parser::NodePtr appendStmt = synth::exprStmt(appendCall, statement.range);
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
      // The guard runs after the length test and the element captures, which
      // is where CPython runs it: `case [a, b] if a < b` needs both names, and
      // a subject of the wrong length must not reach the comparison at all.
      if (guard) {
        mlir::Value guardCond = emitBoolValue(emitExpr(guard), *guard);
        mlir::Block *guardBlock = builder.getInsertionBlock();
        mlir::Block *guardBody =
            builder.createBlock(region, continuation->getIterator());
        builder.setInsertionPointToEnd(guardBlock);
        mlir::cf::CondBranchOp::create(builder, loc(statement), guardCond,
                                       guardBody, mlir::ValueRange{},
                                       nextCheck, mlir::ValueRange{});
        builder.setInsertionPointToStart(guardBody);
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
      bool shapeSupported = (kwdAttrs == nullptr) == (kwdPatterns == nullptr);
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
            "capture or literal sub-patterns"});
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

      // ⭐ THE CASE'S BODY SEES THE CLASS THE PATTERN PROVED, and it did not:
      //
      //     match v:                 # v: int | str | None
      //         case int():
      //             return str(v + 1)
      //     # union<int, str, None> does not provide manifest method '__add__'
      //
      //     match x:                 # x: A, B(A) declares `only`
      //         case B() as b:
      //             return b.only()  # 'only' is overridden by a subclass
      //
      // `narrowed` -- the unwrapped member, or the refined class -- was built
      // and then spent only on the SUB-PATTERNS. The subject's own name and
      // the `as` capture kept the type they had before the test, so the one
      // spelling of narrowing that has no `if` to hang on had none. The `as`
      // capture is bound before the chain reaches this pattern, which is why
      // it needs rebinding rather than binding.
      //
      // ⛔ Not the subject name when the match promoted it to a CELL: a case
      // body that assigns to it reads and writes through that cell, and
      // rebinding the name to a refined view would take the write away from
      // it. The captures are per-case and have no such second life.
      llvm::SmallVector<std::pair<std::string, std::optional<Value>>, 3>
          restoreAfterBody;
      if (narrowed.type != subject.type) {
        auto narrowName = [&](llvm::StringRef name) {
          auto found = values.find(name);
          restoreAfterBody.push_back(
              {name.str(), found == values.end()
                               ? std::nullopt
                               : std::optional<Value>(found->second)});
          values[name.str()] = narrowed;
          types.bindSymbol(name, narrowed.type);
        };
        for (const std::string &name : capturedNames)
          narrowName(name);
        if (subjectNode->kind == "Name") {
          llvm::StringRef subjectName = ast::nameSpelling(*subjectNode);
          bool promoted = llvm::any_of(
              promotedCells, [&](const std::pair<std::string, Value> &cell) {
                return cell.first == subjectName;
              });
          if (!promoted && !llvm::is_contained(capturedNames, subjectName))
            narrowName(subjectName);
        }
      }
      auto restoreNarrowedNames = [&] {
        for (auto &[name, saved] : restoreAfterBody) {
          if (saved) {
            values[name] = *saved;
            types.bindSymbol(name, saved->type);
          } else {
            values.erase(name);
          }
        }
        restoreAfterBody.clear();
      };

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
        restoreNarrowedNames();
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, statement.range.start,
            "match class pattern sub-pattern must name a declared field"});
        return;
      }

      bool refutableByValue = valueCondition.has_value() || guard;
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
      // ⭐ THE GUARD IS EMITTED HERE, after the captures are bound and after
      // the literal sub-patterns have branched, because CPython evaluates it
      // in exactly that position: `case Point(x=0) if log(p):` does not call
      // log for a point whose x is 1, and `case P(x=n) if n > 3` needs n. The
      // arm used to refuse a guard outright rather than place it.
      if (guard) {
        // The guard is EMITTED before the blocks are made: createBlock moves
        // the builder into the block it makes, so making them first puts the
        // guard's own ops in the wrong one.
        mlir::Value guardCond = emitBoolValue(emitExpr(guard), *guard);
        mlir::Block *guardBlock = builder.getInsertionBlock();
        if (!nextCheck)
          nextCheck = builder.createBlock(region, continuation->getIterator());
        mlir::Block *guardBody =
            builder.createBlock(region, continuation->getIterator());
        builder.setInsertionPointToEnd(guardBlock);
        mlir::cf::CondBranchOp::create(builder, loc(statement), guardCond,
                                       guardBody, mlir::ValueRange{},
                                       nextCheck, mlir::ValueRange{});
        builder.setInsertionPointToStart(guardBody);
      }
      emitStatements(body);
      restoreNarrowedNames();
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
                            keys->size() == valuePatterns->size() &&
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
      // As in the sequence arm: after the key tests and the captures, so a
      // subject missing the key never reaches the guard.
      if (guard) {
        mlir::Value guardCond = emitBoolValue(emitExpr(guard), *guard);
        mlir::Block *guardBlock = builder.getInsertionBlock();
        mlir::Block *guardBody =
            builder.createBlock(region, continuation->getIterator());
        builder.setInsertionPointToEnd(guardBlock);
        mlir::cf::CondBranchOp::create(builder, loc(statement), guardCond,
                                       guardBody, mlir::ValueRange{},
                                       nextCheck, mlir::ValueRange{});
        builder.setInsertionPointToStart(guardBody);
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
    //
    // ⭐ AND IT ONLY RUNS WHEN THE PATTERN MATCHED. The two used to be ANDed
    // together in one block, which evaluates the guard either way -- so
    // `case 1 if note():` called note() for a subject that is not 1, where
    // CPython does not. A branch rather than an `and` is what sequences them;
    // the failing-guard edge and the failing-pattern edge are the same block,
    // because both go on to the next case.
    mlir::Block *guardFail = nullptr;
    if (guard) {
      if (condition) {
        guardFail = builder.createBlock(region, continuation->getIterator());
        mlir::Block *guardBlock =
            builder.createBlock(region, guardFail->getIterator());
        builder.setInsertionPointToEnd(check);
        mlir::cf::CondBranchOp::create(builder, loc(statement), *condition,
                                       guardBlock, mlir::ValueRange{},
                                       guardFail, mlir::ValueRange{});
        builder.setInsertionPointToStart(guardBlock);
      }
      condition = emitBoolValue(emitExpr(guard), *guard);
      // The guard's own control flow (a short-circuit `and`) may have moved on
      // from the block it started in; the branch below has to leave from where
      // its value actually is.
      check = builder.getInsertionBlock();
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
        guardFail ? guardFail
                  : builder.createBlock(region, continuation->getIterator());
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
  // ⛔ AND BACK TO SSA on the far side. The cell exists for the arms to write
  // through, not for the rest of the scope: leaving the name cell-backed makes
  // it a heap slot everything after the match reads through, and a match
  // inside a LOOP would then hand the loop a cell where its carried local is
  // an int. The load here is the value the executed arm left.
  for (const auto &[promotedName, cell] : promotedCells) {
    Value loaded = emitCellLoad(statement, cell);
    values[promotedName] = loaded;
    types.bindSymbol(promotedName, loaded.type);
  }
}


} // namespace lython::emitter
