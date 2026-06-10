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
    error(stmt, "async for statement is parsed but async iterator lowering is "
                "not implemented in the C++ emitter yet");
    return;
  }
  if (stmt.kind == "With") {
    emitWith(stmt);
    return;
  }
  if (stmt.kind == "AsyncWith") {
    error(stmt, "async with statement is parsed but async context-manager "
                "lowering is not implemented in the C++ emitter yet");
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

  Value value = emitExpression(**valueNode);
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
    mlir::Value component = builder.create<py::TupleGetOp>(
        loc(anchor), elementType, value.value, indexValue);
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
    if (mlir::isa<py::FuncType>(value.type)) {
      std::optional<FunctionInfo> info;
      if (sourceNode)
        info = resolveCallableInfo(*sourceNode);
      if (!info)
        info = resolveCallableInfo(value.value);
      if (info)
        callableAliases[*name] = *info;
      else
        callableAliases.erase(*name);
    } else {
      callableAliases.erase(*name);
    }
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
                                         const Value &value) {
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
  std::optional<std::string> className = classNameFromType(object.type);
  if (!className) {
    error(target, "attribute assignment requires a class receiver");
    return;
  }
  auto classFound = classes.find(*className);
  if (classFound == classes.end()) {
    error(target, "unknown class '" + *className + "'");
    return;
  }
  auto fieldFound = classFound->second.fields.find(*name);
  if (fieldFound == classFound->second.fields.end()) {
    error(target, "class '" + *className + "' has no field '" + *name + "'");
    return;
  }
  if (value.type != fieldFound->second) {
    error(stmt, "attribute assignment type mismatch: expected " +
                    typeString(fieldFound->second) + ", got " +
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
    mlir::Value current = builder.create<py::DictGetOp>(
        loc(**target), dictTarget->valueType, dictTarget->container.value,
        dictTarget->key.value);
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

void Builder::Impl::emitIf(const parser::Node &stmt) {
  const parser::NodePtr *test = nodeField(stmt, "test");
  const std::vector<parser::NodePtr> *body = nodeListField(stmt, "body");
  const std::vector<parser::NodePtr> *orelse = nodeListField(stmt, "orelse");
  if (!test || !*test || !body || !orelse) {
    error(stmt, "If.test, If.body, or If.orelse is missing");
    return;
  }

  Value condition = emitCondition(**test);
  if (!condition.value)
    return;

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
    symbols = outerSymbols;
    callableAliases = outerCallableAliases;
    blockTerminated = false;
    for (const parser::NodePtr &child : *body) {
      if (child && !blockTerminated)
        emitStatement(*child);
    }
    if (!blockTerminated)
      ensureScfYield(builder, loc(stmt), currentCarriedValues(symbols));

    builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
    symbols = outerSymbols;
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
  bool trueTerminated =
      emitStatementList(*body, outerSymbols, outerCallableAliases);
  if (!trueTerminated)
    builder.create<mlir::cf::BranchOp>(loc(), continueBlock,
                                       currentCarriedValues(symbols));

  builder.setInsertionPointToStart(falseBlock);
  bool falseTerminated =
      emitStatementList(*orelse, outerSymbols, outerCallableAliases);
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
    std::optional<FunctionInfo> enter =
        resolveClassMethod(**contextExpr, manager, "__enter__");
    std::optional<FunctionInfo> exit =
        resolveClassMethod(**contextExpr, manager, "__exit__");
    if (!enter || !exit)
      return;
    if (enter->mayThrow || exit->mayThrow) {
      error(**contextExpr, "with requires nothrow __enter__ and __exit__ "
                           "methods until exception-path lowering is "
                           "implemented");
      return;
    }
    if (exit->resultType != boolType() && exit->resultType != noneType()) {
      error(**contextExpr, "__exit__ must return bool or None in the current "
                           "C++ emitter subset");
      return;
    }

    Value entered = emitResolvedMethodCall(**contextExpr, manager, *enter, {});
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
    mlir::Value none = builder.create<py::NoneOp>(loc(stmt), noneType());
    Value noneValue{none, noneType()};
    llvm::SmallVector<Value, 3> exitArgs{noneValue, noneValue, noneValue};
    Value ignored =
        emitResolvedMethodCall(stmt, it->manager, it->exit, exitArgs);
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
    value = emitExpression(**valueNode);
  } else {
    mlir::Value none = builder.create<py::NoneOp>(loc(), noneType());
    value = Value{none, noneType()};
  }
  if (!value.value)
    return;
  if (!typeAssignable(currentReturnType, value.type)) {
    error(stmt, "return type mismatch: expected " +
                    typeString(currentReturnType) + ", got " +
                    typeString(value.type));
    return;
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
