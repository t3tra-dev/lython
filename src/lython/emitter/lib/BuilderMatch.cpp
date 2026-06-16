#include "BuilderImpl.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

#include "llvm/ADT/STLExtras.h"

#include <set>

namespace lython::emitter {
namespace {

mlir::Value constantI1(mlir::OpBuilder &builder, mlir::Location loc,
                       bool value) {
  return builder.create<mlir::arith::ConstantIntOp>(loc, value ? 1 : 0, 1);
}

llvm::SmallVector<mlir::Value>
carriedValues(const std::map<std::string, Value> &symbols,
              llvm::ArrayRef<std::string> names) {
  llvm::SmallVector<mlir::Value> values;
  values.reserve(names.size());
  for (const std::string &name : names)
    values.push_back(symbols.at(name).value);
  return values;
}

bool hasStarPattern(const std::vector<parser::NodePtr> &patterns) {
  return llvm::any_of(patterns, [](const parser::NodePtr &pattern) {
    return pattern && pattern->kind == "MatchStar";
  });
}

bool isExhaustiveMatchAs(const parser::Node &pattern) {
  if (pattern.kind != "MatchAs")
    return false;
  const parser::NodePtr *nested = nodeField(pattern, "pattern");
  return !nested || !*nested;
}

} // namespace

mlir::Value Builder::Impl::emitMatchEquality(const parser::Node &anchor,
                                             const Value &subject,
                                             const Value &candidate) {
  if (!subject.value || !candidate.value)
    return {};
  if (subject.type != candidate.type)
    return constantI1(builder, loc(anchor), false);

  if (mlir::isa<mlir::IntegerType>(subject.type)) {
    return builder.create<mlir::arith::CmpIOp>(loc(anchor),
                                               mlir::arith::CmpIPredicate::eq,
                                               subject.value, candidate.value);
  }
  if (mlir::isa<mlir::FloatType>(subject.type)) {
    return builder.create<mlir::arith::CmpFOp>(loc(anchor),
                                               mlir::arith::CmpFPredicate::OEQ,
                                               subject.value, candidate.value);
  }
  if (subject.type == noneType())
    return constantI1(builder, loc(anchor), true);

  mlir::Value result = builder.create<py::EqOp>(loc(anchor), boolType(),
                                                subject.value, candidate.value);
  return builder.create<py::CastToPrimOp>(loc(anchor), i1Type(), result,
                                          "exact");
}

mlir::Value Builder::Impl::emitMatchPatternCondition(
    const parser::Node &pattern, const Value &subject,
    std::vector<std::pair<std::string, Value>> &captures) {
  if (!subject.value)
    return {};

  if (pattern.kind == "MatchValue") {
    const parser::NodePtr *valueNode = nodeField(pattern, "value");
    if (!valueNode || !*valueNode) {
      error(pattern, "MatchValue.value is missing");
      return {};
    }
    if ((*valueNode)->kind == "Constant") {
      const parser::FieldValue *literal = valueField(**valueNode, "value");
      if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(subject.type)) {
        const auto *integer =
            literal ? std::get_if<std::int64_t>(literal) : nullptr;
        if (integer) {
          Value candidate{builder.create<mlir::arith::ConstantIntOp>(
                              loc(**valueNode), *integer, intTy.getWidth()),
                          subject.type};
          return emitMatchEquality(pattern, subject, candidate);
        }
      }
      if (auto floatTy = mlir::dyn_cast<mlir::FloatType>(subject.type)) {
        const auto *number = literal ? std::get_if<double>(literal) : nullptr;
        if (number) {
          Value candidate{builder.create<mlir::arith::ConstantOp>(
                              loc(**valueNode), subject.type,
                              builder.getFloatAttr(floatTy, *number)),
                          subject.type};
          return emitMatchEquality(pattern, subject, candidate);
        }
      }
    }
    Value candidate = emitExpression(**valueNode);
    return emitMatchEquality(pattern, subject, candidate);
  }

  if (pattern.kind == "MatchSingleton") {
    const parser::FieldValue *value = valueField(pattern, "value");
    if (!value) {
      error(pattern, "MatchSingleton.value is missing");
      return {};
    }
    if (std::holds_alternative<std::monostate>(*value)) {
      Value none{builder.create<py::NoneOp>(loc(pattern), noneType()),
                 noneType()};
      return emitMatchEquality(pattern, subject, none);
    }
    if (const auto *boolean = std::get_if<bool>(value)) {
      mlir::Value prim = builder.create<mlir::arith::ConstantIntOp>(
          loc(pattern), *boolean ? 1 : 0, 1);
      Value candidate{
          builder.create<py::CastFromPrimOp>(loc(pattern), boolType(), prim),
          boolType()};
      return emitMatchEquality(pattern, subject, candidate);
    }
    error(pattern, "unsupported singleton pattern value");
    return {};
  }

  if (pattern.kind == "MatchAs") {
    const parser::NodePtr *nested = nodeField(pattern, "pattern");
    const std::string *name = stringField(pattern, "name");
    mlir::Value condition = constantI1(builder, loc(pattern), true);
    if (nested && *nested)
      condition = emitMatchPatternCondition(**nested, subject, captures);
    if (!condition)
      return {};
    if (name && !name->empty())
      captures.emplace_back(*name, subject);
    return condition;
  }

  if (pattern.kind == "MatchOr") {
    const std::vector<parser::NodePtr> *patterns =
        nodeListField(pattern, "patterns");
    if (!patterns || patterns->empty()) {
      error(pattern, "MatchOr.patterns is missing");
      return {};
    }
    mlir::Value combined = constantI1(builder, loc(pattern), false);
    for (const parser::NodePtr &alternative : *patterns) {
      if (!alternative) {
        error(pattern, "MatchOr contains an empty alternative");
        return {};
      }
      std::vector<std::pair<std::string, Value>> alternativeCaptures;
      mlir::Value condition =
          emitMatchPatternCondition(*alternative, subject, alternativeCaptures);
      if (!condition)
        return {};
      if (!alternativeCaptures.empty()) {
        error(*alternative,
              "capture names in OR patterns are not supported by the C++ "
              "emitter yet");
        return {};
      }
      combined =
          builder.create<mlir::arith::OrIOp>(loc(pattern), combined, condition);
    }
    return combined;
  }

  if (pattern.kind == "MatchSequence") {
    const std::vector<parser::NodePtr> *patterns =
        nodeListField(pattern, "patterns");
    if (!patterns) {
      error(pattern, "MatchSequence.patterns is missing");
      return {};
    }
    if (hasStarPattern(*patterns)) {
      error(pattern, "starred sequence patterns are parsed but tuple slicing "
                     "lowering is not implemented in the C++ emitter yet");
      return {};
    }
    auto tupleType = mlir::dyn_cast<py::TupleType>(subject.type);
    if (!tupleType)
      return constantI1(builder, loc(pattern), false);
    auto elementTypes = tupleType.getElementTypes();
    if (elementTypes.size() != patterns->size())
      return constantI1(builder, loc(pattern), false);

    mlir::Value combined = constantI1(builder, loc(pattern), true);
    for (auto [index, child] : llvm::enumerate(*patterns)) {
      if (!child) {
        error(pattern, "MatchSequence contains an empty pattern");
        return {};
      }
      mlir::Value indexValue = builder.create<mlir::arith::ConstantIndexOp>(
          loc(*child), static_cast<int64_t>(index));
      py::CallableType contract = unaryMethodContract(
          subject.type, indexValue.getType(), elementTypes[index]);
      mlir::Value component =
          builder.create<py::GetItemOp>(loc(*child), elementTypes[index],
                                        contract, subject.value, indexValue);
      mlir::Value condition = emitMatchPatternCondition(
          *child, Value{component, elementTypes[index]}, captures);
      if (!condition)
        return {};
      combined = builder.create<mlir::arith::AndIOp>(loc(pattern), combined,
                                                     condition);
    }
    return combined;
  }

  error(pattern, "pattern kind '" + pattern.kind +
                     "' is parsed but match lowering is not implemented in "
                     "the C++ emitter yet");
  return {};
}

void Builder::Impl::emitMatch(const parser::Node &stmt) {
  const parser::NodePtr *subjectNode = nodeField(stmt, "subject");
  const std::vector<parser::NodePtr> *cases = nodeListField(stmt, "cases");
  if (!subjectNode || !*subjectNode || !cases) {
    error(stmt, "Match.subject or Match.cases is missing");
    return;
  }

  Value subject = emitExpression(**subjectNode);
  if (!subject.value)
    return;

  std::set<std::string> assignedNames;
  for (const parser::NodePtr &matchCase : *cases) {
    if (!matchCase)
      continue;
    const std::vector<parser::NodePtr> *body =
        nodeListField(*matchCase, "body");
    if (!body)
      continue;
    for (const parser::NodePtr &child : *body)
      if (child)
        collectAssignedNames(*child, assignedNames);
  }

  auto finalCaseIsExhaustive = [&]() {
    if (cases->empty() || !cases->back())
      return false;
    const parser::NodePtr *pattern = nodeField(*cases->back(), "pattern");
    const parser::NodePtr *guard = nodeField(*cases->back(), "guard");
    return pattern && *pattern && (!guard || !*guard) &&
           isExhaustiveMatchAs(**pattern);
  };

  struct BranchBinding {
    mlir::Type type;
    bool assigned = false;
    bool unsupported = false;
  };

  auto directNameTarget = [](const parser::Node &target, llvm::StringRef name) {
    if (target.kind != "Name")
      return false;
    const std::string *targetName = stringField(target, "id");
    return targetName && *targetName == name;
  };

  auto inferDirectBinding = [&](const parser::Node &branchStmt,
                                llvm::StringRef name) -> BranchBinding {
    if (branchStmt.kind == "Assign") {
      const std::vector<parser::NodePtr> *targets =
          nodeListField(branchStmt, "targets");
      const parser::NodePtr *valueNode = nodeField(branchStmt, "value");
      if (!targets || !valueNode || !*valueNode)
        return BranchBinding{mlir::Type{}, false, true};
      bool direct = false;
      for (const parser::NodePtr &targetNode : *targets) {
        if (targetNode && directNameTarget(*targetNode, name)) {
          direct = true;
          break;
        }
      }
      if (!direct)
        return BranchBinding{};
      std::optional<mlir::Type> valueType = inferExpressionType(**valueNode);
      if (!valueType)
        return BranchBinding{mlir::Type{}, false, true};
      return BranchBinding{*valueType, true, false};
    }

    if (branchStmt.kind == "AnnAssign") {
      const parser::NodePtr *target = nodeField(branchStmt, "target");
      if (!target || !*target || !directNameTarget(**target, name))
        return BranchBinding{};
      const parser::NodePtr *annotation = nodeField(branchStmt, "annotation");
      const parser::NodePtr *valueNode = nodeField(branchStmt, "value");
      std::optional<mlir::Type> annotated =
          annotation ? typeFromAnnotation(*annotation) : std::nullopt;
      if (!annotated || !valueNode || !*valueNode)
        return BranchBinding{mlir::Type{}, false, true};
      std::optional<mlir::Type> valueType = inferExpressionType(**valueNode);
      if (!valueType || !typeAssignable(*annotated, *valueType))
        return BranchBinding{mlir::Type{}, false, true};
      return BranchBinding{*annotated, true, false};
    }

    std::set<std::string> nestedAssignments;
    collectAssignedNames(branchStmt, nestedAssignments);
    if (nestedAssignments.count(name.str()))
      return BranchBinding{mlir::Type{}, false, true};
    return BranchBinding{};
  };

  auto inferCaseBinding = [&](const std::vector<parser::NodePtr> &statements,
                              llvm::StringRef name) -> BranchBinding {
    BranchBinding current;
    for (const parser::NodePtr &child : statements) {
      if (!child)
        continue;
      BranchBinding next = inferDirectBinding(*child, name);
      if (next.unsupported)
        return next;
      if (next.assigned)
        current = next;
    }
    return current;
  };

  std::vector<std::string> carriedNames;
  llvm::SmallVector<mlir::Type> carriedTypes;
  for (const std::string &name : assignedNames) {
    auto found = symbols.find(name);
    if (found == symbols.end()) {
      if (!finalCaseIsExhaustive()) {
        error(stmt, "match case assignment to new local variable '" + name +
                        "' requires an exhaustive final case");
        return;
      }
      std::optional<mlir::Type> inferredType;
      for (const parser::NodePtr &matchCase : *cases) {
        if (!matchCase)
          continue;
        const std::vector<parser::NodePtr> *body =
            nodeListField(*matchCase, "body");
        if (!body) {
          error(*matchCase, "match_case.body is missing");
          return;
        }
        BranchBinding binding = inferCaseBinding(*body, name);
        if (binding.unsupported || !binding.assigned) {
          error(stmt, "match case assignment to new local variable '" + name +
                          "' requires direct assignments in every case");
          return;
        }
        if (!inferredType) {
          inferredType = binding.type;
          continue;
        }
        if (*inferredType != binding.type) {
          error(stmt, "match case assignment to new local variable '" + name +
                          "' has mismatched case types: " +
                          typeString(*inferredType) + " vs " +
                          typeString(binding.type));
          return;
        }
      }
      if (!inferredType) {
        error(stmt, "match case assignment to new local variable '" + name +
                        "' could not infer a type");
        return;
      }
      if (py::isPyType(*inferredType)) {
        error(stmt, "match case assignment to new Python object variable '" +
                        name + "' needs ownership-aware phi lowering");
        return;
      }
      carriedNames.push_back(name);
      carriedTypes.push_back(*inferredType);
      continue;
    }
    if (py::isPyType(found->second.type)) {
      error(stmt, "match case assignment to Python object variable '" + name +
                      "' needs ownership-aware phi lowering");
      return;
    }
    carriedNames.push_back(name);
    carriedTypes.push_back(found->second.type);
  }

  mlir::Region *region = builder.getBlock()->getParent();
  mlir::Block *afterBlock = new mlir::Block();
  for (mlir::Type type : carriedTypes)
    afterBlock->addArgument(type, loc(stmt));

  std::map<std::string, Value> outerSymbols = symbols;
  std::map<std::string, FunctionInfo> outerCallableAliases = callableAliases;
  for (auto [caseIndex, matchCase] : llvm::enumerate(*cases)) {
    if (!matchCase) {
      error(stmt, "Match.cases contains an empty case");
      return;
    }
    const parser::NodePtr *pattern = nodeField(*matchCase, "pattern");
    const parser::NodePtr *guard = nodeField(*matchCase, "guard");
    const std::vector<parser::NodePtr> *body =
        nodeListField(*matchCase, "body");
    if (!pattern || !*pattern || !body) {
      error(*matchCase, "match_case.pattern or match_case.body is missing");
      return;
    }

    std::vector<std::pair<std::string, Value>> captures;
    symbols = outerSymbols;
    callableAliases = outerCallableAliases;
    bool exhaustiveFinalCase = caseIndex + 1 == cases->size() &&
                               (!guard || !*guard) &&
                               isExhaustiveMatchAs(**pattern);
    mlir::Value condition;
    if (exhaustiveFinalCase) {
      if (const std::string *name = stringField(**pattern, "name"))
        if (!name->empty())
          captures.emplace_back(*name, subject);
    } else {
      condition = emitMatchPatternCondition(**pattern, subject, captures);
      if (!condition)
        return;
    }
    for (const auto &[name, value] : captures)
      symbols[name] = value;
    mlir::Block *caseBlock = new mlir::Block();
    mlir::Block *nextBlock = exhaustiveFinalCase ? nullptr : new mlir::Block();
    mlir::Block *guardBlock = guard && *guard ? new mlir::Block() : nullptr;
    if (guardBlock)
      region->push_back(guardBlock);
    region->push_back(caseBlock);
    if (nextBlock)
      region->push_back(nextBlock);
    if (exhaustiveFinalCase)
      builder.create<mlir::cf::BranchOp>(loc(*matchCase), caseBlock);
    else
      builder.create<mlir::cf::CondBranchOp>(
          loc(*matchCase), condition, guardBlock ? guardBlock : caseBlock,
          nextBlock);
    blockTerminated = true;

    if (guardBlock) {
      builder.setInsertionPointToStart(guardBlock);
      symbols = outerSymbols;
      callableAliases = outerCallableAliases;
      for (const auto &[name, value] : captures)
        symbols[name] = value;
      Value guardValue = emitCondition(**guard);
      if (!guardValue.value)
        return;
      builder.create<mlir::cf::CondBranchOp>(loc(**guard), guardValue.value,
                                             caseBlock, nextBlock);
    }

    builder.setInsertionPointToStart(caseBlock);
    symbols = outerSymbols;
    callableAliases = outerCallableAliases;
    for (const auto &[name, value] : captures)
      symbols[name] = value;
    blockTerminated = false;
    for (const parser::NodePtr &child : *body)
      if (child && !blockTerminated)
        emitStatement(*child);
    if (!blockTerminated) {
      llvm::SmallVector<mlir::Value> values =
          carriedValues(symbols, carriedNames);
      builder.create<mlir::cf::BranchOp>(loc(*matchCase), afterBlock, values);
    }

    if (nextBlock) {
      builder.setInsertionPointToStart(nextBlock);
      symbols = outerSymbols;
      callableAliases = outerCallableAliases;
      blockTerminated = false;
    }
  }

  if (!finalCaseIsExhaustive()) {
    llvm::SmallVector<mlir::Value> values =
        carriedValues(outerSymbols, carriedNames);
    builder.create<mlir::cf::BranchOp>(loc(stmt), afterBlock, values);
  }
  region->push_back(afterBlock);
  symbols = std::move(outerSymbols);
  callableAliases = std::move(outerCallableAliases);
  builder.setInsertionPointToStart(afterBlock);
  for (auto [index, name] : llvm::enumerate(carriedNames))
    symbols[name] = Value{afterBlock->getArgument(index), carriedTypes[index]};
  blockTerminated = false;
}

} // namespace lython::emitter
