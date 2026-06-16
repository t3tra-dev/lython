#include "BuilderImpl.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include <set>
#include <utility>

namespace lython::emitter {
namespace {

void ensureScfYield(mlir::OpBuilder &builder, mlir::Location loc,
                    mlir::ValueRange values = {}) {
  mlir::Block *block = builder.getBlock();
  if (block &&
      mlir::isa_and_nonnull<mlir::scf::YieldOp>(block->getTerminator()))
    return;
  builder.create<mlir::scf::YieldOp>(loc, values);
}

bool hasUnstructuredControl(const parser::Node &stmt) {
  if (stmt.kind == "Break" || stmt.kind == "Continue" ||
      stmt.kind == "Return" || stmt.kind == "Raise" || stmt.kind == "Assert")
    return true;
  for (llvm::StringRef fieldName :
       {"body", "orelse", "finalbody", "handlers", "cases"}) {
    const std::vector<parser::NodePtr> *items = nodeListField(stmt, fieldName);
    if (!items)
      continue;
    for (const parser::NodePtr &item : *items) {
      if (!item)
        continue;
      if (item->kind == "ExceptHandler") {
        const std::vector<parser::NodePtr> *handlerBody =
            nodeListField(*item, "body");
        if (!handlerBody)
          continue;
        for (const parser::NodePtr &handlerStmt : *handlerBody)
          if (handlerStmt && hasUnstructuredControl(*handlerStmt))
            return true;
        continue;
      }
      if (item->kind == "match_case") {
        const std::vector<parser::NodePtr> *caseBody =
            nodeListField(*item, "body");
        if (!caseBody)
          continue;
        for (const parser::NodePtr &caseStmt : *caseBody)
          if (caseStmt && hasUnstructuredControl(*caseStmt))
            return true;
        continue;
      }
      if (hasUnstructuredControl(*item))
        return true;
    }
  }
  return false;
}

bool statementListHasUnstructuredControl(
    const std::vector<parser::NodePtr> &statements) {
  for (const parser::NodePtr &stmt : statements)
    if (stmt && hasUnstructuredControl(*stmt))
      return true;
  return false;
}

std::optional<std::size_t> finiteTupleArity(py::TupleType type,
                                            mlir::Value value) {
  llvm::ArrayRef<mlir::Type> elementTypes = type.getElementTypes();
  if (elementTypes.empty())
    return 0;
  if (elementTypes.size() > 1)
    return elementTypes.size();
  if (auto create = value.getDefiningOp<py::TupleCreateOp>())
    return create.getElements().size();
  return std::nullopt;
}

mlir::Type finiteTupleElementType(py::TupleType type, std::size_t index) {
  llvm::ArrayRef<mlir::Type> elementTypes = type.getElementTypes();
  if (elementTypes.size() == 1)
    return elementTypes.front();
  return elementTypes[index];
}

} // namespace

void Builder::Impl::emitStatement(const parser::Node &stmt) {
  if (stmt.kind == "Pass" || stmt.kind == "AsyncFunctionDef" ||
      stmt.kind == "ClassDef")
    return;
  if (isTypeVarDefinition(stmt))
    return;
  if (stmt.kind == "FunctionDef") {
    if (!inModuleMain) {
      emitNestedFunctionDef(stmt);
      return;
    }
    emitFunctionBinding(stmt);
    return;
  }
  if (stmt.kind == "Import") {
    emitImport(stmt);
    return;
  }
  if (stmt.kind == "ImportFrom") {
    emitImportFrom(stmt);
    return;
  }
  if (stmt.kind == "Global") {
    if (inModuleMain)
      return;
    std::optional<std::vector<std::string>> names =
        symbolListField(stmt, "names");
    if (!names || names->empty()) {
      error(stmt, "global statement is missing names");
      return;
    }
    for (const std::string &name : *names) {
      if (!primitiveConstants.count(name)) {
        error(stmt, "global name '" + name +
                        "' is not a static primitive module constant; "
                        "global rebinding is not implemented yet");
        return;
      }
    }
    activeGlobalNames.insert(names->begin(), names->end());
    return;
  }
  if (stmt.kind == "Nonlocal") {
    if (inModuleMain) {
      error(stmt, "nonlocal statement is invalid at module scope");
      return;
    }
    std::optional<std::vector<std::string>> names =
        symbolListField(stmt, "names");
    if (!names || names->empty()) {
      error(stmt, "nonlocal statement is missing names");
      return;
    }
    for (const std::string &name : *names) {
      if (!symbols.count(name)) {
        error(stmt, "nonlocal name '" + name +
                        "' has no captured static enclosing binding");
        return;
      }
    }
    activeNonlocalNames.insert(names->begin(), names->end());
    return;
  }
  if (stmt.kind == "Delete") {
    emitDelete(stmt);
    return;
  }
  if (stmt.kind == "Assign") {
    emitAssign(stmt);
    return;
  }
  if (stmt.kind == "AnnAssign") {
    emitAnnAssign(stmt);
    return;
  }
  if (stmt.kind == "TypeAlias") {
    return;
  }
  if (stmt.kind == "AugAssign") {
    emitAugAssign(stmt);
    return;
  }
  if (stmt.kind == "If") {
    emitIf(stmt);
    return;
  }
  if (stmt.kind == "For") {
    emitFor(stmt);
    return;
  }
  if (stmt.kind == "AsyncFor") {
    emitAsyncFor(stmt);
    return;
  }
  if (stmt.kind == "With") {
    emitWith(stmt);
    return;
  }
  if (stmt.kind == "AsyncWith") {
    emitAsyncWith(stmt);
    return;
  }
  if (stmt.kind == "While") {
    emitWhile(stmt);
    return;
  }
  if (stmt.kind == "Break") {
    emitBreak(stmt);
    return;
  }
  if (stmt.kind == "Continue") {
    emitContinue(stmt);
    return;
  }
  if (stmt.kind == "Try") {
    emitTry(stmt);
    return;
  }
  if (stmt.kind == "TryStar") {
    error(stmt, "try/except* statement is parsed but exception-group lowering "
                "is not implemented in the C++ emitter yet");
    return;
  }
  if (stmt.kind == "Match") {
    emitMatch(stmt);
    return;
  }
  if (stmt.kind == "Return") {
    emitReturn(stmt);
    return;
  }
  if (stmt.kind == "Raise") {
    emitRaise(stmt);
    return;
  }
  if (stmt.kind == "Assert") {
    emitAssert(stmt);
    return;
  }
  if (stmt.kind == "Expr") {
    const parser::NodePtr *value = nodeField(stmt, "value");
    if (!value || !*value) {
      error(stmt, "Expr.value is missing");
      return;
    }
    emitExpression(**value);
    return;
  }
  error(stmt,
        "C++ emitter does not support statement kind '" + stmt.kind + "' yet");
}

void Builder::Impl::emitDelete(const parser::Node &stmt) {
  const std::vector<parser::NodePtr> *targets = nodeListField(stmt, "targets");
  if (!targets || targets->empty()) {
    error(stmt, "Delete.targets is missing");
    return;
  }

  for (const parser::NodePtr &target : *targets) {
    if (!target) {
      error(stmt, "delete target is missing");
      return;
    }
    if (target->kind != "Name") {
      error(*target, "delete currently supports only statically scoped names");
      return;
    }
    const std::string *name = stringField(*target, "id");
    if (!name) {
      error(*target, "delete target name is missing");
      return;
    }
    if (activeGlobalNames.count(*name)) {
      error(*target, "delete of global name '" + *name +
                         "' is not implemented in the C++ emitter yet");
      return;
    }
    if (activeNonlocalNames.count(*name)) {
      error(*target, "delete of nonlocal name '" + *name +
                         "' is not implemented in the C++ emitter yet");
      return;
    }
    auto symbol = symbols.find(*name);
    auto moduleBinding = staticModules.find(*name);
    auto moduleSymbolBinding = staticModuleSymbols.find(*name);
    if (symbol == symbols.end() && moduleBinding == staticModules.end() &&
        moduleSymbolBinding == staticModuleSymbols.end() &&
        !staticAnnotationAliases.count(*name)) {
      error(*target, "cannot delete unbound name '" + *name + "'");
      return;
    }
    if (symbol != symbols.end() && py::isPyType(symbol->second.type))
      builder.create<py::DecRefOp>(loc(*target), symbol->second.value);
    if (symbol != symbols.end())
      symbols.erase(symbol);
    primitiveConstants.erase(*name);
    callableAliases.erase(*name);
    staticModules.erase(*name);
    staticModuleSymbols.erase(*name);
    staticAnnotationAliases.erase(*name);
  }
}

void Builder::Impl::emitAssign(const parser::Node &stmt) {
  const std::vector<parser::NodePtr> *targets = nodeListField(stmt, "targets");
  const parser::NodePtr *valueNode = nodeField(stmt, "value");
  if (!targets || targets->empty() || !valueNode || !*valueNode) {
    error(stmt, "Assign.targets or Assign.value is missing");
    return;
  }
  if (targets->size() == 1 && targets->front() &&
      assignLiteralElementsToTarget(stmt, *targets->front(), **valueNode))
    return;

  std::optional<mlir::Type> expectedType;
  if (targets->size() == 1 && targets->front())
    expectedType = inferExpressionType(*targets->front());

  Value value = expectedType
                    ? emitExpressionWithExpectedType(**valueNode, *expectedType)
                    : emitExpression(**valueNode);
  if (!value.value)
    return;
  if (targets->size() > 1 && py::isPyType(value.type)) {
    error(stmt, "chained assignment of owning Python values requires explicit "
                "ownership splitting and is not implemented yet");
    return;
  }
  for (const parser::NodePtr &target : *targets) {
    if (!target) {
      error(stmt, "assignment target is missing");
      return;
    }
    assignValueToTarget(stmt, *target, value, valueNode->get());
  }
}

bool Builder::Impl::assignLiteralElementsToTarget(const parser::Node &stmt,
                                                  const parser::Node &target,
                                                  const parser::Node &source) {
  if (target.kind == "Starred") {
    if (source.kind != "Tuple" && source.kind != "List")
      return false;

    const parser::NodePtr *starredValue = nodeField(target, "value");
    const std::vector<parser::NodePtr> *values = nodeListField(source, "elts");
    if (!starredValue || !*starredValue || !values) {
      error(target, "starred literal assignment is missing target or values");
      return true;
    }

    std::vector<Value> restValues;
    restValues.reserve(values->size());
    for (const parser::NodePtr &valueElement : *values) {
      if (!valueElement || valueElement->kind == "Starred") {
        error(valueElement ? *valueElement : source,
              "starred expressions inside literal unpack values are not "
              "implemented yet");
        return true;
      }
      Value value = emitExpression(*valueElement);
      if (!value.value)
        return true;
      restValues.push_back(value);
    }

    mlir::Type restElementType;
    if (!restValues.empty()) {
      restElementType = restValues.front().type;
      for (const Value &value : restValues) {
        if (value.type == restElementType)
          continue;
        error(target, "starred assignment values must have the same static "
                      "type");
        return true;
      }
    } else if ((*starredValue)->kind == "Name") {
      const std::string *name = stringField(**starredValue, "id");
      auto found = name ? symbols.find(*name) : symbols.end();
      if (found != symbols.end()) {
        auto listTy = mlir::dyn_cast<py::ListType>(found->second.type);
        if (listTy)
          restElementType = listTy.getElementType();
      }
    }
    if (!restElementType) {
      error(target, "empty starred assignment requires an existing typed list "
                    "target");
      return true;
    }

    mlir::Type restListType = listType(restElementType);
    mlir::Value restList =
        builder.create<py::ListNewOp>(loc(target), restListType);
    for (const Value &value : restValues)
      builder.create<py::ListAppendOp>(loc(target), restList, value.value);
    assignValueToTarget(stmt, **starredValue, Value{restList, restListType});
    return true;
  }

  if (target.kind != "Tuple" && target.kind != "List")
    return false;
  if (source.kind != "Tuple" && source.kind != "List")
    return false;

  const std::vector<parser::NodePtr> *targets = nodeListField(target, "elts");
  const std::vector<parser::NodePtr> *values = nodeListField(source, "elts");
  if (!targets || !values) {
    error(target, "literal unpack assignment is missing elements");
    return true;
  }
  int declaredStarredIndex = -1;
  for (auto [index, targetElement] : llvm::enumerate(*targets)) {
    if (!targetElement)
      continue;
    if (targetElement->kind != "Starred")
      continue;
    if (declaredStarredIndex >= 0) {
      error(*targetElement,
            "multiple starred targets in unpack assignment are invalid");
      return true;
    }
    declaredStarredIndex = static_cast<int>(index);
  }

  if (targets->size() != values->size()) {
    if (declaredStarredIndex < 0) {
      error(target, "literal unpack assignment arity mismatch: target has " +
                        std::to_string(targets->size()) +
                        " elements but value has " +
                        std::to_string(values->size()));
      return true;
    }
  }

  auto assignPair = [&](const parser::NodePtr &targetElement,
                        const parser::NodePtr &valueElement) {
    if (!targetElement || !valueElement) {
      error(target, "literal unpack assignment contains an empty element");
      return false;
    }
    if (valueElement->kind == "Starred") {
      error(*valueElement,
            "starred expressions inside literal unpack values are not "
            "implemented yet");
      return false;
    }
    if ((targetElement->kind == "Tuple" || targetElement->kind == "List") &&
        (valueElement->kind == "Tuple" || valueElement->kind == "List")) {
      assignLiteralElementsToTarget(stmt, *targetElement, *valueElement);
      return true;
    }
    Value value = emitExpression(*valueElement);
    if (!value.value)
      return false;
    assignValueToTarget(stmt, *targetElement, value, valueElement.get());
    return true;
  };

  int starredIndex = declaredStarredIndex;

  if (starredIndex < 0) {
    if (targets->size() != values->size()) {
      error(target, "literal unpack assignment arity mismatch: target has " +
                        std::to_string(targets->size()) +
                        " elements but value has " +
                        std::to_string(values->size()));
      return true;
    }
    for (auto [index, targetElement] : llvm::enumerate(*targets))
      if (!assignPair(targetElement, (*values)[index]))
        return true;
    return true;
  }

  std::size_t prefixCount = static_cast<std::size_t>(starredIndex);
  std::size_t suffixCount = targets->size() - prefixCount - 1;
  if (values->size() < prefixCount + suffixCount) {
    error(target, "literal starred unpack assignment arity mismatch");
    return true;
  }

  for (std::size_t index = 0; index < prefixCount; ++index)
    if (!assignPair((*targets)[index], (*values)[index]))
      return true;

  const parser::NodePtr &starredTarget = (*targets)[prefixCount];
  const parser::NodePtr *starredValue = nodeField(*starredTarget, "value");
  if (!starredValue || !*starredValue) {
    error(*starredTarget, "starred assignment target is missing");
    return true;
  }

  std::size_t restBegin = prefixCount;
  std::size_t restEnd = values->size() - suffixCount;
  std::vector<Value> restValues;
  restValues.reserve(restEnd - restBegin);
  for (std::size_t index = restBegin; index < restEnd; ++index) {
    const parser::NodePtr &valueElement = (*values)[index];
    if (!valueElement || valueElement->kind == "Starred") {
      error(valueElement ? *valueElement : source,
            "starred expressions inside literal unpack values are not "
            "implemented yet");
      return true;
    }
    Value value = emitExpression(*valueElement);
    if (!value.value)
      return true;
    restValues.push_back(value);
  }

  mlir::Type restElementType;
  if (!restValues.empty()) {
    restElementType = restValues.front().type;
    for (const Value &value : restValues) {
      if (value.type == restElementType)
        continue;
      error(*starredTarget,
            "starred unpack rest values must have the same static type");
      return true;
    }
  } else if ((*starredValue)->kind == "Name") {
    const std::string *name = stringField(**starredValue, "id");
    auto found = name ? symbols.find(*name) : symbols.end();
    if (found != symbols.end()) {
      auto listTy = mlir::dyn_cast<py::ListType>(found->second.type);
      if (listTy)
        restElementType = listTy.getElementType();
    }
  }
  if (!restElementType) {
    error(*starredTarget, "empty starred unpack rest requires an existing "
                          "typed list target");
    return true;
  }

  mlir::Type restListType = listType(restElementType);
  mlir::Value restList =
      builder.create<py::ListNewOp>(loc(*starredTarget), restListType);
  for (const Value &value : restValues)
    builder.create<py::ListAppendOp>(loc(*starredTarget), restList,
                                     value.value);
  assignValueToTarget(stmt, **starredValue, Value{restList, restListType});

  for (std::size_t suffix = 0; suffix < suffixCount; ++suffix) {
    std::size_t targetIndex = prefixCount + 1 + suffix;
    std::size_t valueIndex = restEnd + suffix;
    if (!assignPair((*targets)[targetIndex], (*values)[valueIndex]))
      return true;
  }
  return true;
}

void Builder::Impl::updateCallableAliasForBinding(
    llvm::StringRef name, const Value &value, const parser::Node *sourceNode) {
  std::string key = name.str();
  if (!py::isCallableType(value.type)) {
    callableAliases.erase(key);
    return;
  }

  if (value.callableInfo) {
    callableAliases[key] = *value.callableInfo;
    return;
  }

  std::optional<FunctionInfo> info;
  if (sourceNode)
    info = resolveCallableInfo(*sourceNode);
  if (!info)
    info = resolveCallableInfo(value.value);
  if (info)
    callableAliases[key] = *info;
  else
    callableAliases.erase(key);
}

void Builder::Impl::assignValueToTarget(const parser::Node &stmt,
                                        const parser::Node &target,
                                        const Value &value,
                                        const parser::Node *sourceNode) {
  if (!value.value)
    return;
  auto tupleType = mlir::dyn_cast<py::TupleType>(value.type);
  std::optional<std::size_t> tupleArity =
      tupleType ? finiteTupleArity(tupleType, value.value) : std::nullopt;
  auto emitTupleComponent = [&](const parser::Node &anchor,
                                std::size_t index) -> Value {
    mlir::Type elementType = finiteTupleElementType(tupleType, index);
    mlir::Value indexValue = builder.create<mlir::arith::ConstantIndexOp>(
        loc(anchor), static_cast<std::int64_t>(index));
    py::CallableType contract =
        unaryMethodContract(value.type, indexValue.getType(), elementType);
    mlir::Value component = builder.create<py::GetItemOp>(
        loc(anchor), elementType, contract, value.value, indexValue);
    return Value{component, elementType};
  };
  auto assignTupleRestList = [&](const parser::Node &starredTarget,
                                 std::size_t begin, std::size_t end) -> bool {
    const parser::NodePtr *starredValue = nodeField(starredTarget, "value");
    if (!starredValue || !*starredValue) {
      error(starredTarget, "starred assignment target is missing");
      return false;
    }

    std::vector<Value> restValues;
    restValues.reserve(end - begin);
    for (std::size_t index = begin; index < end; ++index)
      restValues.push_back(emitTupleComponent(starredTarget, index));

    mlir::Type restElementType;
    if (!restValues.empty()) {
      restElementType = restValues.front().type;
      for (const Value &rest : restValues) {
        if (rest.type == restElementType)
          continue;
        error(starredTarget, "starred unpack rest values must have the same "
                             "static type");
        return false;
      }
    } else if ((*starredValue)->kind == "Name") {
      const std::string *name = stringField(**starredValue, "id");
      auto found = name ? symbols.find(*name) : symbols.end();
      if (found != symbols.end()) {
        auto listTy = mlir::dyn_cast<py::ListType>(found->second.type);
        if (listTy)
          restElementType = listTy.getElementType();
      }
    }
    if (!restElementType) {
      error(starredTarget, "empty starred unpack rest requires an existing "
                           "typed list target");
      return false;
    }

    mlir::Type restListType = listType(restElementType);
    mlir::Value restList =
        builder.create<py::ListNewOp>(loc(starredTarget), restListType);
    for (const Value &rest : restValues)
      builder.create<py::ListAppendOp>(loc(starredTarget), restList,
                                       rest.value);
    assignValueToTarget(stmt, **starredValue, Value{restList, restListType});
    return true;
  };

  if (target.kind == "Starred") {
    if (!tupleType) {
      error(target, "starred assignment requires a statically typed tuple "
                    "value, got " +
                        typeString(value.type));
      return;
    }
    if (!tupleArity) {
      error(target, "starred assignment requires a statically finite tuple "
                    "value");
      return;
    }
    assignTupleRestList(target, 0, *tupleArity);
    return;
  }

  if (target.kind == "Name") {
    const std::string *name = stringField(target, "id");
    if (!name) {
      error(target, "assignment target name is missing");
      return;
    }
    if (activeGlobalNames.count(*name)) {
      error(target, "rebinding global name '" + *name +
                        "' is not implemented in the C++ emitter yet");
      return;
    }
    if (activeNonlocalNames.count(*name)) {
      error(target, "rebinding nonlocal name '" + *name +
                        "' is not implemented in the C++ emitter yet");
      return;
    }
    symbols[*name] = value;
    updateCallableAliasForBinding(*name, value, sourceNode);
    if (inModuleMain) {
      if (sourceNode) {
        if (std::optional<PrimitiveConstant> constant =
                primitiveScalarConstructorConstant(*sourceNode))
          primitiveConstants[*name] = *constant;
        else
          primitiveConstants.erase(*name);
      } else {
        primitiveConstants.erase(*name);
      }
    }
    return;
  }

  if (target.kind == "Attribute") {
    assignAttributeValue(stmt, target, value);
    return;
  }
  if (target.kind == "Subscript") {
    assignSubscriptValue(stmt, target, value);
    return;
  }
  if (target.kind == "Tuple" || target.kind == "List") {
    const std::vector<parser::NodePtr> *elements =
        nodeListField(target, "elts");
    if (!elements) {
      error(target, target.kind + ".elts is missing");
      return;
    }
    if (!tupleType) {
      error(target, "unpack assignment requires a statically typed tuple "
                    "value, got " +
                        typeString(value.type));
      return;
    }
    if (!tupleArity) {
      error(target,
            "unpack assignment requires a statically finite tuple value");
      return;
    }
    int starredIndex = -1;
    for (auto [index, element] : llvm::enumerate(*elements)) {
      if (!element || element->kind != "Starred")
        continue;
      if (starredIndex >= 0) {
        error(*element,
              "multiple starred targets in unpack assignment are invalid");
        return;
      }
      starredIndex = static_cast<int>(index);
    }

    if (starredIndex < 0 && *tupleArity != elements->size()) {
      error(target, "unpack assignment arity mismatch: target has " +
                        std::to_string(elements->size()) +
                        " elements but value has " +
                        std::to_string(*tupleArity));
      return;
    }
    if (starredIndex >= 0) {
      std::size_t prefixCount = static_cast<std::size_t>(starredIndex);
      std::size_t suffixCount = elements->size() - prefixCount - 1;
      if (*tupleArity < prefixCount + suffixCount) {
        error(target, "starred unpack assignment arity mismatch");
        return;
      }

      for (std::size_t index = 0; index < prefixCount; ++index) {
        if (!(*elements)[index]) {
          error(target, "unpack assignment contains an empty target");
          return;
        }
        Value component = emitTupleComponent(*(*elements)[index], index);
        assignValueToTarget(stmt, *(*elements)[index], component);
      }

      const parser::NodePtr &starredTarget = (*elements)[prefixCount];
      std::size_t restBegin = prefixCount;
      std::size_t restEnd = *tupleArity - suffixCount;
      if (!starredTarget ||
          !assignTupleRestList(*starredTarget, restBegin, restEnd))
        return;

      for (std::size_t suffix = 0; suffix < suffixCount; ++suffix) {
        std::size_t targetIndex = prefixCount + 1 + suffix;
        std::size_t valueIndex = restEnd + suffix;
        if (!(*elements)[targetIndex]) {
          error(target, "unpack assignment contains an empty target");
          return;
        }
        Value component =
            emitTupleComponent(*(*elements)[targetIndex], valueIndex);
        assignValueToTarget(stmt, *(*elements)[targetIndex], component);
      }
      return;
    }

    for (auto [index, element] : llvm::enumerate(*elements)) {
      if (!element) {
        error(target, "unpack assignment contains an empty target");
        return;
      }
      Value component = emitTupleComponent(*element, index);
      assignValueToTarget(stmt, *element, component);
    }
    return;
  }

  error(target, "assignment target type '" + target.kind +
                    "' is not supported by the C++ emitter");
}

void Builder::Impl::assignAttributeValue(const parser::Node &stmt,
                                         const parser::Node &target,
                                         const Value &inputValue) {
  Value value = inputValue;
  if (!value.value)
    return;
  const parser::NodePtr *objectNode = nodeField(target, "value");
  const std::string *name = stringField(target, "attr");
  if (!objectNode || !*objectNode || !name) {
    error(target, "Attribute.value or Attribute.attr is missing");
    return;
  }
  Value object = emitExpression(**objectNode);
  if (!object.value)
    return;
  std::optional<std::string> staticClassName = classNameFromType(object.type);
  if (!staticClassName) {
    error(target, "attribute assignment requires a class receiver");
    return;
  }

  auto findField = [&](llvm::StringRef candidate)
      -> std::optional<std::pair<std::string, mlir::Type>> {
    auto classFound = classes.find(candidate.str());
    if (classFound == classes.end())
      return std::nullopt;
    auto fieldFound = classFound->second.fields.find(*name);
    if (fieldFound == classFound->second.fields.end())
      return std::nullopt;
    return std::make_pair(candidate.str(), fieldFound->second);
  };

  std::optional<std::pair<std::string, mlir::Type>> resolved;
  if (std::optional<std::string> fact = classFactForView(object))
    resolved = findField(*fact);
  if (!resolved)
    resolved = findField(*staticClassName);
  if (!resolved) {
    error(target,
          "class '" + *staticClassName + "' has no field '" + *name + "'");
    return;
  }
  if (resolved->first != *staticClassName) {
    object = viewClassAs(target, std::move(object), resolved->first);
    if (!object.value)
      return;
  }
  value = coerceToExpectedType(stmt, std::move(value), resolved->second);
  if (!typeAssignable(resolved->second, value.type)) {
    error(stmt, "attribute assignment type mismatch: expected " +
                    typeString(resolved->second) + ", got " +
                    typeString(value.type));
    return;
  }
  builder.create<py::AttrSetOp>(loc(stmt), object.value, *name, value.value);
}

std::optional<DictSubscriptTarget>
Builder::Impl::emitDictSubscriptTarget(const parser::Node &target) {
  const parser::NodePtr *containerNode = nodeField(target, "value");
  const parser::NodePtr *keyNode = nodeField(target, "slice");
  if (!containerNode || !*containerNode || !keyNode || !*keyNode) {
    error(target, "Subscript.value or Subscript.slice is missing");
    return std::nullopt;
  }

  Value container = emitExpression(**containerNode);
  if (!container.value)
    return std::nullopt;
  if ((*keyNode)->kind == "Slice") {
    error(**keyNode,
          "slice subscript assignment is parsed but slice lowering is not "
          "implemented in the C++ emitter yet");
    return std::nullopt;
  }
  Value key = emitExpression(**keyNode);
  if (!key.value)
    return std::nullopt;
  std::optional<std::pair<mlir::Type, mlir::Type>> dictTypes =
      dictKeyValueTypes(container.type);
  if (!dictTypes) {
    error(target, "subscript assignment supports only typed dict values");
    return std::nullopt;
  }
  if (!dictStorageSupported(dictTypes->first, dictTypes->second)) {
    error(target, "dict assignment key/value types are not supported by typed "
                  "memref lowering yet: " +
                      typeString(dictTypes->first) + ", " +
                      typeString(dictTypes->second));
    return std::nullopt;
  }
  if (key.type != dictTypes->first) {
    error(target, "dict assignment key type mismatch: expected " +
                      typeString(dictTypes->first) + ", got " +
                      typeString(key.type));
    return std::nullopt;
  }
  return DictSubscriptTarget{container, key, dictTypes->second};
}

void Builder::Impl::assignDictSubscriptValue(
    const parser::Node &stmt, const DictSubscriptTarget &dictTarget,
    const Value &value) {
  if (!value.value)
    return;
  if (value.type != dictTarget.valueType) {
    error(stmt, "dict assignment value type mismatch: expected " +
                    typeString(dictTarget.valueType) + ", got " +
                    typeString(value.type));
    return;
  }
  builder.create<py::DictInsertOp>(loc(stmt), dictTarget.container.value,
                                   dictTarget.key.value, value.value);
}

void Builder::Impl::assignSubscriptValue(const parser::Node &stmt,
                                         const parser::Node &target,
                                         const Value &value) {
  if (!value.value)
    return;
  std::optional<DictSubscriptTarget> dictTarget =
      emitDictSubscriptTarget(target);
  if (!dictTarget)
    return;
  assignDictSubscriptValue(stmt, *dictTarget, value);
}

void Builder::Impl::emitAttributeAssign(const parser::Node &stmt,
                                        const parser::Node &target,
                                        const parser::Node &valueNode) {
  Value value = emitExpression(valueNode);
  if (!value.value)
    return;
  assignAttributeValue(stmt, target, value);
}

void Builder::Impl::emitSubscriptAssign(const parser::Node &stmt,
                                        const parser::Node &target,
                                        const parser::Node &valueNode) {
  Value value = emitExpression(valueNode);
  if (!value.value)
    return;
  assignSubscriptValue(stmt, target, value);
}

void Builder::Impl::emitAnnAssign(const parser::Node &stmt) {
  const parser::NodePtr *target = nodeField(stmt, "target");
  const parser::NodePtr *annotation = nodeField(stmt, "annotation");
  const parser::NodePtr *valueNode = nodeField(stmt, "value");
  if (!target || !*target) {
    error(stmt, "AnnAssign.target is missing");
    return;
  }
  if (annotation && isTypeAliasMarker(*annotation))
    return;
  std::optional<mlir::Type> annotatedType =
      annotation ? typeFromAnnotation(*annotation) : std::nullopt;
  if (!annotatedType) {
    error(stmt, "AnnAssign.annotation is missing or unsupported");
    return;
  }
  if (!valueNode || !*valueNode)
    return;

  Value value = emitExpressionWithExpectedType(**valueNode, *annotatedType);
  if (!value.value)
    return;
  value = coerceToExpectedType(**valueNode, std::move(value), *annotatedType);
  if (!typeAssignable(*annotatedType, value.type)) {
    error(stmt, "annotated assignment type mismatch: expected " +
                    typeString(*annotatedType) + ", got " +
                    typeString(value.type));
    return;
  }

  if ((*target)->kind == "Name") {
    const std::string *name = stringField(**target, "id");
    if (!name) {
      error(**target, "annotated assignment target name is missing");
      return;
    }
    if (activeGlobalNames.count(*name)) {
      error(**target, "rebinding global name '" + *name +
                          "' is not implemented in the C++ emitter yet");
      return;
    }
    if (activeNonlocalNames.count(*name)) {
      error(**target, "rebinding nonlocal name '" + *name +
                          "' is not implemented in the C++ emitter yet");
      return;
    }
    symbols[*name] = value;
    updateCallableAliasForBinding(*name, value, valueNode->get());
    if (inModuleMain) {
      if (std::optional<PrimitiveConstant> constant =
              primitiveScalarConstructorConstant(**valueNode))
        primitiveConstants[*name] = *constant;
      else
        primitiveConstants.erase(*name);
    }
    return;
  }

  if ((*target)->kind == "Attribute") {
    std::optional<mlir::Type> targetType = inferExpressionType(**target);
    if (!targetType) {
      error(**target, "annotated attribute assignment target has no static "
                      "field type");
      return;
    }
    if (*targetType != *annotatedType) {
      error(stmt, "annotated attribute assignment type mismatch: field has " +
                      typeString(*targetType) + " but annotation is " +
                      typeString(*annotatedType));
      return;
    }
    assignAttributeValue(stmt, **target, value);
    return;
  }

  if ((*target)->kind == "Subscript") {
    std::optional<DictSubscriptTarget> dictTarget =
        emitDictSubscriptTarget(**target);
    if (!dictTarget)
      return;
    if (dictTarget->valueType != *annotatedType) {
      error(stmt, "annotated dict assignment type mismatch: dict value has " +
                      typeString(dictTarget->valueType) +
                      " but annotation is " + typeString(*annotatedType));
      return;
    }
    assignDictSubscriptValue(stmt, *dictTarget, value);
    return;
  }

  error(stmt, "C++ emitter supports only name, attribute, and typed dict "
              "subscript annotated assignment targets for now");
}

void Builder::Impl::emitAugAssign(const parser::Node &stmt) {
  const parser::NodePtr *target = nodeField(stmt, "target");
  const parser::NodePtr *valueNode = nodeField(stmt, "value");
  std::optional<std::string> op = symbolField(stmt, "op");
  if (!target || !*target || !valueNode || !*valueNode || !op) {
    error(stmt,
          "AugAssign.target, AugAssign.value, or AugAssign.op is missing");
    return;
  }

  if ((*target)->kind == "Name") {
    const std::string *name = stringField(**target, "id");
    if (!name) {
      error(**target, "augmented assignment target name is missing");
      return;
    }
    if (activeGlobalNames.count(*name)) {
      error(**target, "rebinding global name '" + *name +
                          "' is not implemented in the C++ emitter yet");
      return;
    }
    if (activeNonlocalNames.count(*name)) {
      error(**target, "rebinding nonlocal name '" + *name +
                          "' is not implemented in the C++ emitter yet");
      return;
    }
    auto found = symbols.find(*name);
    if (found == symbols.end()) {
      error(**target, "unknown name '" + *name + "'");
      return;
    }
    Value rhs = emitExpression(**valueNode);
    if (!rhs.value)
      return;
    Value result = emitBinaryOperation(stmt, *op, found->second, rhs);
    if (!result.value)
      return;
    symbols[*name] = result;
    updateCallableAliasForBinding(*name, result, valueNode->get());
    if (inModuleMain)
      primitiveConstants.erase(*name);
    return;
  }

  if ((*target)->kind == "Attribute") {
    const parser::NodePtr *objectNode = nodeField(**target, "value");
    const std::string *name = stringField(**target, "attr");
    if (!objectNode || !*objectNode || !name) {
      error(**target, "Attribute.value or Attribute.attr is missing");
      return;
    }
    Value object = emitExpression(**objectNode);
    if (!object.value)
      return;
    std::optional<std::string> className = classNameFromType(object.type);
    if (!className) {
      error(**target,
            "attribute augmented assignment requires a class receiver");
      return;
    }
    auto classFound = classes.find(*className);
    if (classFound == classes.end()) {
      error(**target, "unknown class '" + *className + "'");
      return;
    }
    auto fieldFound = classFound->second.fields.find(*name);
    if (fieldFound == classFound->second.fields.end()) {
      error(**target,
            "class '" + *className + "' has no field '" + *name + "'");
      return;
    }
    mlir::Value current = builder.create<py::AttrGetOp>(
        loc(**target), fieldFound->second, object.value, *name);
    Value rhs = emitExpression(**valueNode);
    if (!rhs.value)
      return;
    Value result =
        emitBinaryOperation(stmt, *op, Value{current, fieldFound->second}, rhs);
    if (!result.value)
      return;
    if (result.type != fieldFound->second) {
      error(stmt, "attribute augmented assignment type mismatch: expected " +
                      typeString(fieldFound->second) + ", got " +
                      typeString(result.type));
      return;
    }
    builder.create<py::AttrSetOp>(loc(stmt), object.value, *name, result.value);
    return;
  }

  if ((*target)->kind == "Subscript") {
    std::optional<DictSubscriptTarget> dictTarget =
        emitDictSubscriptTarget(**target);
    if (!dictTarget)
      return;
    mlir::Value current = builder.create<py::GetItemOp>(
        loc(**target), dictTarget->valueType,
        unaryMethodContract(dictTarget->container.type, dictTarget->key.type,
                            dictTarget->valueType),
        dictTarget->container.value, dictTarget->key.value);
    Value rhs = emitExpression(**valueNode);
    if (!rhs.value)
      return;
    Value result = emitBinaryOperation(
        stmt, *op, Value{current, dictTarget->valueType}, rhs);
    if (!result.value)
      return;
    assignDictSubscriptValue(stmt, *dictTarget, result);
    return;
  }

  error(stmt, "C++ emitter supports only name, attribute, and typed dict "
              "subscript augmented assignment targets for now");
}

std::optional<std::pair<const parser::Node *, mlir::Type>>
Builder::Impl::matchIsinstanceCall(const parser::Node &call) {
  if (call.kind != "Call")
    return std::nullopt;
  const parser::NodePtr *func = nodeField(call, "func");
  const std::vector<parser::NodePtr> *args = nodeListField(call, "args");
  const std::vector<parser::NodePtr> *keywords =
      nodeListField(call, "keywords");
  if (!func || !*func || (*func)->kind != "Name" || !args ||
      args->size() != 2 || !(*args)[0] || !(*args)[1] ||
      (keywords && !keywords->empty()))
    return std::nullopt;
  const std::string *callee = stringField(**func, "id");
  if (!callee || *callee != "isinstance")
    return std::nullopt;
  std::optional<mlir::Type> memberType = typeFromAnnotation((*args)[1]);
  if (!memberType)
    return std::nullopt;
  return std::make_pair((*args)[0].get(), *memberType);
}

std::optional<Builder::Impl::NarrowingTest>
Builder::Impl::matchUnionNarrowingTest(const parser::Node &test) {
  const parser::Node *nameNode = nullptr;
  mlir::Type testedType;
  bool negated = false;

  if (test.kind == "Compare") {
    const parser::NodePtr *lhsNode = nodeField(test, "left");
    std::optional<std::vector<std::string>> ops = symbolListField(test, "ops");
    const std::vector<parser::NodePtr> *comparators =
        nodeListField(test, "comparators");
    if (!lhsNode || !*lhsNode || !ops || !comparators || ops->size() != 1 ||
        comparators->size() != 1 || !(*comparators)[0])
      return std::nullopt;
    llvm::StringRef op = (*ops)[0];
    if (op != "is" && op != "is not")
      return std::nullopt;

    std::optional<int> lhsKey = singletonKey(**lhsNode);
    std::optional<int> rhsKey = singletonKey(*(*comparators)[0]);
    if (!lhsKey && rhsKey && *rhsKey == 0)
      nameNode = lhsNode->get();
    else if (!rhsKey && lhsKey && *lhsKey == 0)
      nameNode = (*comparators)[0].get();
    testedType = noneType();
    negated = op == "is not";
  } else if (test.kind == "Call") {
    auto isinstanceMatch = matchIsinstanceCall(test);
    if (!isinstanceMatch)
      return std::nullopt;
    nameNode = isinstanceMatch->first;
    testedType = isinstanceMatch->second;
  }

  if (!nameNode || nameNode->kind != "Name" || !testedType)
    return std::nullopt;
  const std::string *name = stringField(*nameNode, "id");
  if (!name)
    return std::nullopt;
  auto found = symbols.find(*name);
  if (found == symbols.end())
    return std::nullopt;
  Value source = found->second;

  auto baseStaticTruth = [&]() -> std::optional<bool> {
    if (mlir::isa<py::NoneType>(testedType)) {
      if (mlir::isa<py::NoneType>(source.type))
        return true;
      if (source.exactClass || mlir::isa<py::ClassType>(source.type))
        return false;
      return std::nullopt;
    }

    auto testedClass = mlir::dyn_cast<py::ClassType>(testedType);
    if (!testedClass)
      return std::nullopt;
    if (source.exactClass)
      return classSubtypeOf(*source.exactClass, testedClass.getClassName());
    if (source.provenClass) {
      if (classSubtypeOf(*source.provenClass, testedClass.getClassName()))
        return true;
      if (!classSubtypeOf(testedClass.getClassName(), *source.provenClass))
        return false;
    }
    if (typeSubtypeOf(source.type, testedType))
      return true;
    auto sourceClass = mlir::dyn_cast<py::ClassType>(source.type);
    if (sourceClass &&
        !classSubtypeOf(testedClass.getClassName(),
                        sourceClass.getClassName()) &&
        !classSubtypeOf(sourceClass.getClassName(), testedClass.getClassName()))
      return false;
    return std::nullopt;
  };

  auto makeStaticTruth = [&](bool baseTruth) {
    return negated ? !baseTruth : baseTruth;
  };

  std::optional<bool> knownBaseTruth = baseStaticTruth();
  auto unionType = mlir::dyn_cast<py::UnionType>(source.type);
  if (!unionType) {
    auto sourceClass = mlir::dyn_cast<py::ClassType>(source.type);
    auto testedClass = mlir::dyn_cast<py::ClassType>(testedType);
    if (!sourceClass || !testedClass) {
      if (!knownBaseTruth)
        return std::nullopt;
      return NarrowingTest{*name,
                           source.type,
                           {},
                           {},
                           negated,
                           /*staticTruthKnown=*/true,
                           makeStaticTruth(*knownBaseTruth)};
    }

    mlir::Type matchType;
    if (typeSubtypeOf(source.type, testedType)) {
      matchType = source.type;
    } else if (classSubtypeOf(testedClass.getClassName(),
                              sourceClass.getClassName())) {
      matchType = testedType;
    } else if (!knownBaseTruth || *knownBaseTruth) {
      return std::nullopt;
    }

    bool staticTruthKnown = knownBaseTruth.has_value();
    bool staticTruth =
        staticTruthKnown ? makeStaticTruth(*knownBaseTruth) : false;
    return NarrowingTest{*name,   source.type,      matchType,  {},
                         negated, staticTruthKnown, staticTruth};
  }

  std::vector<UnionMemberMatch> matches =
      unionMembersMatchingType(unionType, testedType,
                               /*requireLayoutCompatibleDowncast=*/false);
  if (matches.empty())
    return knownBaseTruth ? std::optional<NarrowingTest>(
                                NarrowingTest{*name,
                                              source.type,
                                              {},
                                              {},
                                              negated,
                                              /*staticTruthKnown=*/true,
                                              makeStaticTruth(*knownBaseTruth)})
                          : std::nullopt;

  bool viewAsTestedType =
      mlir::isa<py::ClassType>(testedType) &&
      llvm::all_of(matches, [&](const UnionMemberMatch &match) {
        return match.sourceMember == match.narrowedType &&
               typeSubtypeOf(match.sourceMember, testedType);
      });

  llvm::SmallVector<mlir::Type> matchTypes;
  llvm::SmallVector<mlir::Type> excludedFromComplement;
  if (viewAsTestedType) {
    matchTypes.push_back(testedType);
  }
  for (const UnionMemberMatch &match : matches) {
    if (!viewAsTestedType)
      matchTypes.push_back(match.narrowedType);
    if (match.sourceMember == match.narrowedType)
      excludedFromComplement.push_back(match.sourceMember);
  }

  llvm::SmallVector<mlir::Type> complement;
  for (mlir::Type member : unionType.getMemberTypes())
    if (!llvm::is_contained(excludedFromComplement, member))
      complement.push_back(member);

  mlir::Type matchType = py::UnionType::getNormalized(&context, matchTypes);
  mlir::Type complementType =
      complement.empty() ? mlir::Type()
                         : py::UnionType::getNormalized(&context, complement);
  if (!matchType)
    return std::nullopt;
  bool staticTruthKnown = knownBaseTruth.has_value() || complement.empty();
  bool staticTruth =
      knownBaseTruth ? makeStaticTruth(*knownBaseTruth) : !negated;
  return NarrowingTest{*name,   source.type,      matchType,  complementType,
                       negated, staticTruthKnown, staticTruth};
}

void Builder::Impl::emitIf(const parser::Node &stmt) {
  const parser::NodePtr *test = nodeField(stmt, "test");
  const std::vector<parser::NodePtr> *body = nodeListField(stmt, "body");
  const std::vector<parser::NodePtr> *orelse = nodeListField(stmt, "orelse");
  if (!test || !*test || !body || !orelse) {
    error(stmt, "If.test, If.body, or If.orelse is missing");
    return;
  }

  std::optional<NarrowingTest> narrowing = matchUnionNarrowingTest(**test);

  Value condition =
      narrowing && narrowing->staticTruthKnown
          ? Value{builder.create<mlir::arith::ConstantIntOp>(
                      loc(**test), narrowing->staticTruth ? 1 : 0, 1),
                  i1Type()}
          : emitCondition(**test);
  if (!condition.value)
    return;

  // Branch-local narrowing: rebind the tested union local to its proven
  // member (or to the complement subset union) via py.union.unwrap. The
  // unwrap must be emitted at the current insertion point, which the caller
  // positions inside the branch. Branches that never read the name skip the
  // rebind: a dead unwrap would keep the union root live into early-exit
  // branches and block the divergent-successor ownership drop. The proven
  // None binding is skipped for the same reason.
  auto narrowedSymbols = [&](const std::map<std::string, Value> &base,
                             bool branchTruth,
                             const std::vector<parser::NodePtr> *branchBody) {
    std::map<std::string, Value> branchSymbols = base;
    if (!narrowing)
      return branchSymbols;
    auto found = branchSymbols.find(narrowing->name);
    if (found == branchSymbols.end() ||
        found->second.type != narrowing->sourceType)
      return branchSymbols;
    const bool provesMatch = narrowing->negated ? !branchTruth : branchTruth;
    mlir::Type targetType =
        provesMatch ? narrowing->matchType : narrowing->complementType;
    if (!targetType || mlir::isa<py::NoneType>(targetType))
      return branchSymbols;
    if (targetType == found->second.type)
      return branchSymbols;
    if (branchBody && !referencesName(*branchBody, narrowing->name))
      return branchSymbols;
    if (mlir::isa<py::UnionType>(found->second.type)) {
      mlir::Value unwrapped = builder.create<py::UnionUnwrapOp>(
          loc(stmt), targetType, found->second.value);
      Value narrowed{unwrapped, targetType, found->second.exactClass,
                     found->second.provenClass};
      if (auto targetClass = mlir::dyn_cast<py::ClassType>(targetType)) {
        if (narrowed.exactClass &&
            !classSubtypeOf(*narrowed.exactClass, targetClass.getClassName()))
          narrowed.exactClass.reset();
        narrowed =
            markProvenClass(std::move(narrowed), targetClass.getClassName());
      }
      found->second = std::move(narrowed);
      return branchSymbols;
    }
    if (mlir::isa<py::ClassType>(found->second.type)) {
      auto targetClass = mlir::dyn_cast<py::ClassType>(targetType);
      if (!targetClass)
        return branchSymbols;
      found->second = viewClassAs(stmt, std::move(found->second),
                                  targetClass.getClassName());
    }
    return branchSymbols;
  };

  std::set<std::string> assignedNames;
  for (const parser::NodePtr &child : *body)
    if (child)
      collectAssignedNames(*child, assignedNames);
  for (const parser::NodePtr &child : *orelse)
    if (child)
      collectAssignedNames(*child, assignedNames);

  const bool hasUnstructuredBranches =
      statementListHasUnstructuredControl(*body) ||
      statementListHasUnstructuredControl(*orelse);

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

  auto inferFinalBinding = [&](const std::vector<parser::NodePtr> &statements,
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
      if (orelse->empty() || hasUnstructuredBranches) {
        error(stmt, "if branch assignment to new local variable '" + name +
                        "' requires both branches to assign it directly");
        return;
      }
      BranchBinding trueBinding = inferFinalBinding(*body, name);
      BranchBinding falseBinding = inferFinalBinding(*orelse, name);
      if (trueBinding.unsupported || falseBinding.unsupported ||
          !trueBinding.assigned || !falseBinding.assigned) {
        error(stmt, "if branch assignment to new local variable '" + name +
                        "' requires direct assignments in both branches");
        return;
      }
      if (trueBinding.type != falseBinding.type) {
        error(stmt, "if branch assignment to new local variable '" + name +
                        "' has mismatched branch types: " +
                        typeString(trueBinding.type) + " vs " +
                        typeString(falseBinding.type));
        return;
      }
      if (py::isPyType(trueBinding.type)) {
        error(stmt, "if branch assignment to new Python object variable '" +
                        name + "' needs ownership-aware phi lowering");
        return;
      }
      carriedNames.push_back(name);
      carriedTypes.push_back(trueBinding.type);
      continue;
    }
    if (py::isPyType(found->second.type)) {
      error(stmt, "if branch assignment to Python object variable '" + name +
                      "' needs ownership-aware phi lowering");
      return;
    }
    carriedNames.push_back(name);
    carriedTypes.push_back(found->second.type);
  }

  auto currentCarriedValues = [&](const std::map<std::string, Value> &scope) {
    llvm::SmallVector<mlir::Value> values;
    values.reserve(carriedNames.size());
    for (const std::string &name : carriedNames)
      values.push_back(scope.at(name).value);
    return values;
  };

  bool needsImplicitElseCarriedValues =
      orelse->empty() && !carriedNames.empty();
  if (!needsImplicitElseCarriedValues && !hasUnstructuredBranches) {
    auto ifOp = builder.create<mlir::scf::IfOp>(
        loc(stmt), carriedTypes, condition.value, /*withElseRegion=*/true);
    std::map<std::string, Value> outerSymbols = symbols;
    std::map<std::string, FunctionInfo> outerCallableAliases = callableAliases;
    bool savedTerminated = blockTerminated;

    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
    symbols = narrowedSymbols(outerSymbols, /*branchTruth=*/true, body);
    callableAliases = outerCallableAliases;
    blockTerminated = false;
    for (const parser::NodePtr &child : *body) {
      if (child && !blockTerminated)
        emitStatement(*child);
    }
    if (!blockTerminated)
      ensureScfYield(builder, loc(stmt), currentCarriedValues(symbols));

    builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
    symbols = narrowedSymbols(outerSymbols, /*branchTruth=*/false, orelse);
    callableAliases = outerCallableAliases;
    blockTerminated = false;
    for (const parser::NodePtr &child : *orelse) {
      if (child && !blockTerminated)
        emitStatement(*child);
    }
    if (!blockTerminated)
      ensureScfYield(builder, loc(stmt), currentCarriedValues(symbols));

    symbols = std::move(outerSymbols);
    callableAliases = std::move(outerCallableAliases);
    for (auto [index, name] : llvm::enumerate(carriedNames))
      symbols[name] = Value{ifOp.getResult(index), carriedTypes[index]};
    blockTerminated = savedTerminated;
    builder.setInsertionPointAfter(ifOp);
    return;
  }

  mlir::Region *region = builder.getBlock()->getParent();
  mlir::Block *trueBlock = new mlir::Block();
  mlir::Block *falseBlock = new mlir::Block();
  mlir::Block *continueBlock = new mlir::Block();
  region->push_back(trueBlock);
  region->push_back(falseBlock);
  region->push_back(continueBlock);
  for (mlir::Type type : carriedTypes)
    continueBlock->addArgument(type, loc(stmt));
  builder.create<mlir::cf::CondBranchOp>(loc(), condition.value, trueBlock,
                                         falseBlock);
  blockTerminated = true;

  std::map<std::string, Value> outerSymbols = symbols;
  std::map<std::string, FunctionInfo> outerCallableAliases = callableAliases;
  builder.setInsertionPointToStart(trueBlock);
  bool trueTerminated = emitStatementList(
      *body, narrowedSymbols(outerSymbols, true, body), outerCallableAliases);
  if (!trueTerminated)
    builder.create<mlir::cf::BranchOp>(loc(), continueBlock,
                                       currentCarriedValues(symbols));

  builder.setInsertionPointToStart(falseBlock);
  bool falseTerminated =
      emitStatementList(*orelse, narrowedSymbols(outerSymbols, false, orelse),
                        outerCallableAliases);
  if (!falseTerminated)
    builder.create<mlir::cf::BranchOp>(loc(), continueBlock,
                                       currentCarriedValues(symbols));

  symbols = std::move(outerSymbols);
  callableAliases = std::move(outerCallableAliases);
  if (trueTerminated && falseTerminated) {
    continueBlock->erase();
    blockTerminated = true;
    return;
  }
  builder.setInsertionPointToStart(continueBlock);
  for (auto [index, name] : llvm::enumerate(carriedNames))
    symbols[name] =
        Value{continueBlock->getArgument(index), carriedTypes[index]};
  blockTerminated = false;
  // Early-exit narrowing: when exactly one branch reaches the continuation,
  // its branch-local facts hold for the rest of the block
  // (e.g. `if x is None: return` proves x is not None below the if).
  if (trueTerminated != falseTerminated)
    symbols = narrowedSymbols(symbols, /*branchTruth=*/falseTerminated,
                              /*branchBody=*/nullptr);
}

void Builder::Impl::emitWith(const parser::Node &stmt) {
  if (inNativeFunction) {
    error(stmt, "with statements are not supported inside @native functions");
    return;
  }

  const std::vector<parser::NodePtr> *items = nodeListField(stmt, "items");
  const std::vector<parser::NodePtr> *body = nodeListField(stmt, "body");
  if (!items || items->empty() || !body) {
    error(stmt, "With.items or With.body is missing");
    return;
  }
  if (statementListHasUnstructuredControl(*body)) {
    error(stmt, "with body containing return, raise, break, continue, or "
                "assert requires guaranteed __exit__ control-flow lowering "
                "and is not implemented yet");
    return;
  }
  if (statementListMayThrow(*body)) {
    error(stmt, "with body may throw; __exit__ exception suppression lowering "
                "is not implemented yet");
    return;
  }

  struct ActiveContext {
    Value manager;
    FunctionInfo exit;
  };
  std::vector<ActiveContext> activeContexts;
  activeContexts.reserve(items->size());

  for (const parser::NodePtr &item : *items) {
    if (!item || item->kind != "withitem") {
      error(stmt, "With.items must contain withitem nodes");
      return;
    }
    const parser::NodePtr *contextExpr = nodeField(*item, "context_expr");
    const parser::NodePtr *optionalVars = nodeField(*item, "optional_vars");
    if (!contextExpr || !*contextExpr) {
      error(*item, "withitem.context_expr is missing");
      return;
    }
    if ((*contextExpr)->kind == "Call") {
      error(**contextExpr, "with context expression calls require maythrow "
                           "constructor/invoke analysis and are not "
                           "implemented yet; bind the manager before with");
      return;
    }
    if (expressionMayThrow(**contextExpr)) {
      error(**contextExpr, "with context expression may throw; __exit__ "
                           "exception-path lowering is not implemented yet");
      return;
    }

    Value manager = emitExpression(**contextExpr);
    if (!manager.value)
      return;
    if (std::optional<Value> concrete = concreteProtocolValue(manager))
      manager = *concrete;
    std::optional<FunctionInfo> enter =
        resolveClassMethod(**contextExpr, manager, "__enter__");
    std::optional<FunctionInfo> exit =
        resolveClassMethod(**contextExpr, manager, "__exit__");
    if (!enter || !exit)
      return;
    llvm::SmallVector<mlir::Type, 3> exitArgTypes{noneType(), noneType(),
                                                  noneType()};
    std::optional<py::ProtocolType> contextManager =
        protocolType("ContextManager", {enter->resultType});
    if (!contextManager) {
      error(**contextExpr, "failed to instantiate ContextManager protocol");
      return;
    }
    std::optional<mlir::Type> expectedEnter =
        resolveProtocolMethodResult(**contextExpr, *contextManager, "__enter__",
                                    {}, "ContextManager.__enter__");
    if (!expectedEnter)
      return;
    if (!typeAssignable(*expectedEnter, enter->resultType)) {
      error(**contextExpr, "ContextManager.__enter__ contract for " +
                               typeString(*contextManager) +
                               " does not match method result " +
                               typeString(enter->resultType));
      return;
    }
    std::optional<mlir::Type> expectedExit =
        resolveProtocolMethodResult(**contextExpr, *contextManager, "__exit__",
                                    exitArgTypes, "ContextManager.__exit__");
    if (!expectedExit)
      return;
    if (!typeAssignable(*expectedExit, exit->resultType)) {
      error(**contextExpr, "__exit__ must satisfy " +
                               typeString(*contextManager) +
                               " for __exit__(None, None, None), got " +
                               typeString(exit->resultType));
      return;
    }
    if (auto managerClass = mlir::dyn_cast<py::ClassType>(manager.type)) {
      if (!classConformsToProtocol(managerClass, *contextManager)) {
        error(**contextExpr,
              "with context manager " + typeString(manager.type) +
                  " does not satisfy " + typeString(*contextManager));
        return;
      }
    }
    if (enter->mayThrow || exit->mayThrow) {
      error(**contextExpr, "with requires nothrow __enter__ and __exit__ "
                           "methods until exception-path lowering is "
                           "implemented");
      return;
    }
    if (enter->isAsync || exit->isAsync) {
      error(**contextExpr, "with requires synchronous __enter__ and __exit__ "
                           "methods; use async with for async context "
                           "managers");
      return;
    }

    std::optional<std::vector<Value>> enterArgs =
        prepareResolvedMethodCallArguments(**contextExpr, manager, *enter, {});
    if (!enterArgs)
      return;
    auto enterOp = builder.create<py::EnterOp>(
        loc(**contextExpr), enter->resultType, enter->symbolName,
        enter->functionType, (*enterArgs)[0].value, mlir::UnitAttr{});
    Value entered = applyReturnedClassSummary(
        Value{enterOp.getResult(), enter->resultType}, *enter);
    if (!entered.value)
      return;
    if (optionalVars && *optionalVars)
      assignValueToTarget(stmt, **optionalVars, entered);
    activeContexts.push_back(ActiveContext{manager, *exit});
  }

  for (const parser::NodePtr &child : *body) {
    if (child && !blockTerminated)
      emitStatement(*child);
  }
  if (blockTerminated)
    return;

  for (auto it = activeContexts.rbegin(); it != activeContexts.rend(); ++it) {
    mlir::Value excType = builder.create<py::NoneOp>(loc(stmt), noneType());
    mlir::Value excValue = builder.create<py::NoneOp>(loc(stmt), noneType());
    mlir::Value traceback = builder.create<py::NoneOp>(loc(stmt), noneType());
    llvm::SmallVector<Value, 3> exitArgs{Value{excType, noneType()},
                                         Value{excValue, noneType()},
                                         Value{traceback, noneType()}};
    std::optional<std::vector<Value>> prepared =
        prepareResolvedMethodCallArguments(stmt, it->manager, it->exit,
                                           exitArgs);
    if (!prepared)
      return;
    builder.create<py::ExitOp>(
        loc(stmt), it->exit.resultType, it->exit.symbolName,
        it->exit.functionType, (*prepared)[0].value, (*prepared)[1].value,
        (*prepared)[2].value, (*prepared)[3].value, mlir::UnitAttr{});
  }
}

void Builder::Impl::emitAsyncWith(const parser::Node &stmt) {
  if (!inAsyncFunction) {
    error(stmt, "async with statements are valid only inside async functions");
    return;
  }
  if (inNativeFunction) {
    error(stmt,
          "async with statements are not supported inside @native functions");
    return;
  }

  const std::vector<parser::NodePtr> *items = nodeListField(stmt, "items");
  const std::vector<parser::NodePtr> *body = nodeListField(stmt, "body");
  if (!items || items->empty() || !body) {
    error(stmt, "AsyncWith.items or AsyncWith.body is missing");
    return;
  }
  if (statementListHasUnstructuredControl(*body)) {
    error(stmt, "async with body containing return, raise, break, continue, or "
                "assert requires guaranteed __aexit__ control-flow lowering "
                "and is not implemented yet");
    return;
  }
  if (statementListMayThrow(*body)) {
    error(stmt, "async with body may throw; __aexit__ exception suppression "
                "lowering is not implemented yet");
    return;
  }

  struct ActiveAsyncContext {
    Value manager;
    FunctionInfo exit;
  };
  std::vector<ActiveAsyncContext> activeContexts;
  activeContexts.reserve(items->size());

  for (const parser::NodePtr &item : *items) {
    if (!item || item->kind != "withitem") {
      error(stmt, "AsyncWith.items must contain withitem nodes");
      return;
    }
    const parser::NodePtr *contextExpr = nodeField(*item, "context_expr");
    const parser::NodePtr *optionalVars = nodeField(*item, "optional_vars");
    if (!contextExpr || !*contextExpr) {
      error(*item, "withitem.context_expr is missing");
      return;
    }
    if ((*contextExpr)->kind == "Call") {
      error(**contextExpr,
            "async with context expression calls require "
            "maythrow constructor/invoke analysis and are not "
            "implemented yet; bind the manager before async with");
      return;
    }
    if (expressionMayThrow(**contextExpr)) {
      error(**contextExpr, "async with context expression may throw; __aexit__ "
                           "exception-path lowering is not implemented yet");
      return;
    }

    Value manager = emitExpression(**contextExpr);
    if (!manager.value)
      return;
    if (std::optional<Value> concrete = concreteProtocolValue(manager))
      manager = *concrete;
    std::optional<FunctionInfo> enter =
        resolveClassMethod(**contextExpr, manager, "__aenter__");
    std::optional<FunctionInfo> exit =
        resolveClassMethod(**contextExpr, manager, "__aexit__");
    if (!enter || !exit)
      return;

    mlir::Type enterAwaitableType = methodAwaitableType(*enter);
    mlir::Type exitAwaitableType = methodAwaitableType(*exit);
    mlir::Type enterPayload = awaitablePayloadType(enterAwaitableType);
    mlir::Type exitPayload = awaitablePayloadType(exitAwaitableType);
    if (!enterPayload) {
      error(**contextExpr, "__aenter__ must return an awaitable in async with, "
                           "got " +
                               typeString(enterAwaitableType));
      return;
    }
    if (!exitPayload) {
      error(**contextExpr, "__aexit__ must return an awaitable in async with, "
                           "got " +
                               typeString(exitAwaitableType));
      return;
    }
    llvm::SmallVector<mlir::Type, 3> exitArgTypes{noneType(), noneType(),
                                                  noneType()};
    std::optional<py::ProtocolType> asyncContextManager =
        protocolType("AsyncContextManager", {enterPayload});
    if (!asyncContextManager) {
      error(**contextExpr,
            "failed to instantiate AsyncContextManager protocol");
      return;
    }
    std::optional<mlir::Type> expectedEnterAwaitable =
        resolveProtocolMethodResult(**contextExpr, *asyncContextManager,
                                    "__aenter__", {},
                                    "AsyncContextManager.__aenter__");
    if (!expectedEnterAwaitable)
      return;
    if (!typeAssignable(*expectedEnterAwaitable, enterAwaitableType)) {
      error(**contextExpr, "AsyncContextManager.__aenter__ contract for " +
                               typeString(*asyncContextManager) +
                               " does not match method result " +
                               typeString(enterAwaitableType));
      return;
    }
    std::optional<mlir::Type> expectedExitAwaitable =
        resolveProtocolMethodResult(**contextExpr, *asyncContextManager,
                                    "__aexit__", exitArgTypes,
                                    "AsyncContextManager.__aexit__");
    if (!expectedExitAwaitable)
      return;
    if (!typeAssignable(*expectedExitAwaitable, exitAwaitableType)) {
      error(**contextExpr, "__aexit__ must satisfy " +
                               typeString(*asyncContextManager) +
                               " for __aexit__(None, None, None), got " +
                               typeString(exitAwaitableType));
      return;
    }
    if (!lowerableAwaitableType(enterAwaitableType) ||
        !lowerableAwaitableType(exitAwaitableType)) {
      error(**contextExpr,
            "async with currently requires native Coroutine protocol "
            "descriptors or async.value results from __aenter__ and __aexit__");
      return;
    }
    if (auto managerClass = mlir::dyn_cast<py::ClassType>(manager.type)) {
      if (!classConformsToProtocol(managerClass, *asyncContextManager)) {
        error(**contextExpr,
              "async with context manager " + typeString(manager.type) +
                  " does not satisfy " + typeString(*asyncContextManager));
        return;
      }
    }
    if (enter->mayThrow || exit->mayThrow) {
      error(**contextExpr, "async with requires nothrow __aenter__ and "
                           "__aexit__ methods until exception-path lowering is "
                           "implemented");
      return;
    }

    std::optional<std::vector<Value>> enterArgs =
        prepareResolvedMethodCallArguments(**contextExpr, manager, *enter, {});
    if (!enterArgs)
      return;
    mlir::UnitAttr enterAsync =
        enter->isAsync ? builder.getUnitAttr() : mlir::UnitAttr{};
    auto enterOp = builder.create<py::AEnterOp>(
        loc(**contextExpr), enterAwaitableType, enter->symbolName,
        enter->functionType, (*enterArgs)[0].value, enterAsync);
    Value enteredAwaitable{enterOp.getResult(), enterAwaitableType};
    Value entered = awaitConcreteValue(**contextExpr, enteredAwaitable,
                                       "__aenter__ result");
    if (!entered.value)
      return;
    if (optionalVars && *optionalVars)
      assignValueToTarget(stmt, **optionalVars, entered);
    activeContexts.push_back(ActiveAsyncContext{manager, *exit});
  }

  for (const parser::NodePtr &child : *body) {
    if (child && !blockTerminated)
      emitStatement(*child);
  }
  if (blockTerminated)
    return;

  for (auto it = activeContexts.rbegin(); it != activeContexts.rend(); ++it) {
    mlir::Value excType = builder.create<py::NoneOp>(loc(stmt), noneType());
    mlir::Value excValue = builder.create<py::NoneOp>(loc(stmt), noneType());
    mlir::Value traceback = builder.create<py::NoneOp>(loc(stmt), noneType());
    llvm::SmallVector<Value, 3> exitArgs{Value{excType, noneType()},
                                         Value{excValue, noneType()},
                                         Value{traceback, noneType()}};
    std::optional<std::vector<Value>> prepared =
        prepareResolvedMethodCallArguments(stmt, it->manager, it->exit,
                                           exitArgs);
    if (!prepared)
      return;
    mlir::Type exitAwaitableType = methodAwaitableType(it->exit);
    mlir::UnitAttr exitAsync =
        it->exit.isAsync ? builder.getUnitAttr() : mlir::UnitAttr{};
    auto exitOp = builder.create<py::AExitOp>(
        loc(stmt), exitAwaitableType, it->exit.symbolName,
        it->exit.functionType, (*prepared)[0].value, (*prepared)[1].value,
        (*prepared)[2].value, (*prepared)[3].value, exitAsync);
    Value exitAwaitable{exitOp.getResult(), exitAwaitableType};
    Value ignored = awaitConcreteValue(stmt, exitAwaitable, "__aexit__ result");
    if (!ignored.value)
      return;
  }
}

bool Builder::Impl::emitStatementList(
    const std::vector<parser::NodePtr> &statements,
    const std::map<std::string, Value> &baseSymbols,
    const std::map<std::string, FunctionInfo> &baseCallableAliases) {
  symbols = baseSymbols;
  callableAliases = baseCallableAliases;
  blockTerminated = false;
  for (const parser::NodePtr &stmt : statements) {
    if (stmt && !blockTerminated)
      emitStatement(*stmt);
  }
  return blockTerminated;
}

void Builder::Impl::emitReturn(const parser::Node &stmt) {
  const parser::NodePtr *valueNode = nodeField(stmt, "value");
  if (inNativeFunction && (!valueNode || !*valueNode) &&
      currentReturnType == noneType()) {
    builder.create<mlir::func::ReturnOp>(loc(stmt));
    blockTerminated = true;
    return;
  }

  Value value;
  if (valueNode && *valueNode) {
    value = emitExpressionWithExpectedType(**valueNode, currentReturnType);
  } else {
    mlir::Value none = builder.create<py::NoneOp>(loc(), noneType());
    value = Value{none, noneType()};
  }
  if (!value.value)
    return;
  value = coerceToExpectedType(stmt, std::move(value), currentReturnType);
  if (!typeAssignable(currentReturnType, value.type)) {
    error(stmt, "return type mismatch: expected " +
                    typeString(currentReturnType) + ", got " +
                    typeString(value.type));
    return;
  }
  if (value.value.getType() != currentReturnType &&
      value.type == currentReturnType) {
    if (mlir::isa<py::ProtocolType>(currentReturnType) &&
        mlir::isa<py::ClassType>(value.value.getType())) {
      value.value = builder.create<py::ProtocolViewOp>(
          loc(stmt), currentReturnType, value.value);
    } else if (py::isCallableType(currentReturnType) &&
               py::isCallableType(value.value.getType()) &&
               typeAssignable(currentReturnType, value.value.getType())) {
      value.type = value.value.getType();
    } else {
      error(stmt, "return value type " + typeString(value.value.getType()) +
                      " cannot be materialized as " +
                      typeString(currentReturnType));
      return;
    }
  }
  if (inNativeFunction) {
    if (currentReturnType == noneType())
      builder.create<mlir::func::ReturnOp>(loc());
    else
      builder.create<mlir::func::ReturnOp>(loc(),
                                           mlir::ValueRange{value.value});
    blockTerminated = true;
    return;
  }
  if (inAsyncFunction) {
    builder.create<mlir::async::ReturnOp>(loc(stmt),
                                          mlir::ValueRange{value.value});
    blockTerminated = true;
    return;
  }
  builder.create<py::ReturnOp>(loc(), mlir::ValueRange{value.value});
  blockTerminated = true;
}

} // namespace lython::emitter
