#include "BuilderImpl.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include <functional>

namespace lython::emitter {
namespace {

bool isTupleLiteralElementSupported(mlir::Type type) {
  return mlir::isa<mlir::IntegerType, mlir::FloatType, py::IntType,
                   py::BoolType, py::FloatType, py::NoneType, py::StrType,
                   py::ExceptionType, py::ClassType>(type);
}

bool exactScalarEqualityAlwaysFalse(mlir::Type lhs, mlir::Type rhs) {
  if (lhs == rhs)
    return false;
  auto isNumeric = [](mlir::Type type) {
    return mlir::isa<py::IntType, py::FloatType, py::BoolType>(type);
  };
  auto isExactScalar = [&](mlir::Type type) {
    return isNumeric(type) || mlir::isa<py::StrType, py::NoneType>(type);
  };
  if (!isExactScalar(lhs) || !isExactScalar(rhs))
    return false;
  return !(isNumeric(lhs) && isNumeric(rhs));
}

std::int64_t clampSliceIndex(std::int64_t value, std::int64_t length,
                             std::int64_t lower, std::int64_t upper) {
  if (value < 0)
    value += length;
  if (value < lower)
    return lower;
  if (value > upper)
    return upper;
  return value;
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

bool isNameRef(const parser::Node &node, llvm::StringRef name) {
  if (node.kind != "Name")
    return false;
  const std::string *id = stringField(node, "id");
  return id && *id == name;
}

bool containsNameRef(const parser::Node &node, llvm::StringRef name) {
  if (isNameRef(node, name))
    return true;
  for (const parser::Field &field : node.fields) {
    if (const auto *child = std::get_if<parser::NodePtr>(&field.value)) {
      if (*child && containsNameRef(**child, name))
        return true;
      continue;
    }
    if (const auto *children =
            std::get_if<std::vector<parser::NodePtr>>(&field.value)) {
      for (const parser::NodePtr &child : *children)
        if (child && containsNameRef(*child, name))
          return true;
    }
  }
  return false;
}

} // namespace

Value Builder::Impl::emitAttribute(const parser::Node &expr) {
  const parser::NodePtr *objectNode = nodeField(expr, "value");
  const std::string *name = stringField(expr, "attr");
  if (!objectNode || !*objectNode || !name) {
    error(expr, "Attribute.value or Attribute.attr is missing");
    return Value{{}, noneType()};
  }

  Value object = emitExpression(**objectNode);
  if (!object.value)
    return Value{{}, noneType()};
  std::optional<std::string> staticClassName = classNameFromType(object.type);
  if (!staticClassName) {
    error(expr, "attribute access requires a statically known class receiver");
    return Value{{}, noneType()};
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
    error(expr,
          "class '" + *staticClassName + "' has no field '" + *name + "'");
    return Value{{}, noneType()};
  }
  if (resolved->first != *staticClassName) {
    object = viewClassAs(expr, std::move(object), resolved->first);
    if (!object.value)
      return Value{{}, resolved->second};
  }
  mlir::Value result = builder.create<py::AttrGetOp>(
      loc(expr), resolved->second, object.value, *name);
  return Value{result, resolved->second};
}

std::optional<std::vector<Value>>
Builder::Impl::finiteSequenceElements(const parser::Node &anchor,
                                      const Value &sequence,
                                      llvm::StringRef context) {
  if (auto tupleType = mlir::dyn_cast<py::TupleType>(sequence.type)) {
    std::optional<std::size_t> arity =
        finiteTupleArity(tupleType, sequence.value);
    if (!arity) {
      error(anchor, context.str() + " requires a statically finite tuple "
                                    "source");
      return std::nullopt;
    }

    std::vector<Value> elements;
    elements.reserve(*arity);
    for (std::size_t index = 0; index < *arity; ++index) {
      mlir::Value indexValue = builder.create<mlir::arith::ConstantIndexOp>(
          loc(anchor), static_cast<std::int64_t>(index));
      mlir::Type elementType = finiteTupleElementType(tupleType, index);
      py::CallableType contract =
          unaryMethodContract(sequence.type, indexValue.getType(), elementType);
      mlir::Value component = builder.create<py::GetItemOp>(
          loc(anchor), elementType, contract, sequence.value, indexValue);
      elements.push_back(Value{component, elementType});
    }
    return elements;
  }

  if (listElementType(sequence.type)) {
    auto found = finiteListElements.find(sequence.value);
    if (found == finiteListElements.end()) {
      error(anchor, context.str() +
                        " requires a list source with statically known "
                        "elements");
      return std::nullopt;
    }
    return found->second;
  }

  error(anchor, context.str() +
                    " currently supports only statically finite tuple or "
                    "list sources");
  return std::nullopt;
}

Value Builder::Impl::emitListFromValues(const parser::Node &anchor,
                                        llvm::ArrayRef<Value> values,
                                        mlir::Type elementType) {
  if (values.empty() && !elementType) {
    error(anchor, "empty list literal requires an explicit element type");
    return Value{{}, noneType()};
  }

  if (!elementType)
    elementType = values.front().type;
  for (const Value &value : values) {
    if (value.type != elementType) {
      error(anchor, "list literal elements must have the same static type");
      return Value{{}, noneType()};
    }
  }

  mlir::Type resultType = listType(elementType);
  mlir::Value result = builder.create<py::ListNewOp>(loc(anchor), resultType);
  for (const Value &value : values)
    builder.create<py::ListAppendOp>(loc(anchor), result, value.value);
  finiteListElements[result] = std::vector<Value>(values.begin(), values.end());
  return Value{result, resultType};
}

Value Builder::Impl::emitList(const parser::Node &expr,
                              mlir::Type preferredElementType) {
  const std::vector<parser::NodePtr> *elements = nodeListField(expr, "elts");
  if (!elements) {
    error(expr, "List.elts is missing");
    return Value{{}, noneType()};
  }
  if (elements->empty()) {
    return emitListFromValues(expr, {}, preferredElementType);
  }

  std::vector<Value> values;
  values.reserve(elements->size());
  for (const parser::NodePtr &element : *elements) {
    if (!element)
      continue;
    if (element->kind == "Starred") {
      const parser::NodePtr *starredValue = nodeField(*element, "value");
      if (!starredValue || !*starredValue) {
        error(*element, "starred list element is missing a value");
        return Value{{}, noneType()};
      }
      Value sequence = emitExpression(**starredValue);
      if (!sequence.value)
        return Value{{}, noneType()};
      std::optional<std::vector<Value>> expanded =
          finiteSequenceElements(*element, sequence, "starred list display");
      if (!expanded)
        return Value{{}, noneType()};
      if (preferredElementType) {
        for (const Value &expandedValue : *expanded) {
          if (expandedValue.type != preferredElementType) {
            error(*element, "starred list element type mismatch: expected " +
                                typeString(preferredElementType) + ", got " +
                                typeString(expandedValue.type));
            return Value{{}, noneType()};
          }
        }
      }
      values.insert(values.end(), expanded->begin(), expanded->end());
      continue;
    }
    Value value =
        preferredElementType
            ? emitExpressionWithExpectedType(*element, preferredElementType)
            : emitExpression(*element);
    if (!value.value)
      return Value{{}, noneType()};
    values.push_back(std::move(value));
  }
  if (values.empty())
    return Value{{}, noneType()};

  return emitListFromValues(expr, values, preferredElementType);
}

Value Builder::Impl::emitListComprehension(const parser::Node &expr,
                                           mlir::Type preferredElementType) {
  const parser::NodePtr *elementNode = nodeField(expr, "elt");
  const std::vector<parser::NodePtr> *generators =
      nodeListField(expr, "generators");
  if (!elementNode || !*elementNode || !generators || generators->size() != 1 ||
      !generators->front()) {
    error(expr, "ListComp.elt or ListComp.generators is missing");
    return Value{{}, noneType()};
  }

  const parser::Node &generator = *generators->front();
  const parser::NodePtr *targetNode = nodeField(generator, "target");
  const parser::NodePtr *iterNode = nodeField(generator, "iter");
  const std::vector<parser::NodePtr> *ifs = nodeListField(generator, "ifs");
  const parser::FieldValue *isAsyncValue = valueField(generator, "is_async");
  const auto *isAsync =
      isAsyncValue ? std::get_if<std::int64_t>(isAsyncValue) : nullptr;
  if (isAsync && *isAsync != 0) {
    error(generator, "async list comprehensions are parsed but async iterator "
                     "lowering is not implemented in the C++ emitter yet");
    return Value{{}, noneType()};
  }
  if (!targetNode || !*targetNode || (*targetNode)->kind != "Name" ||
      !iterNode || !*iterNode) {
    error(generator, "list comprehension currently requires a name target and "
                     "range(...) iterator");
    return Value{{}, noneType()};
  }
  const std::string *targetName = stringField(**targetNode, "id");
  if (!targetName) {
    error(**targetNode, "list comprehension target name is missing");
    return Value{{}, noneType()};
  }

  if ((!ifs || ifs->empty()) && (*elementNode)->kind == "Subscript" &&
      (*iterNode)->kind == "Call") {
    const parser::NodePtr *elementValue = nodeField(**elementNode, "value");
    const parser::NodePtr *elementIndex = nodeField(**elementNode, "slice");
    const parser::NodePtr *rangeFunc = nodeField(**iterNode, "func");
    const std::vector<parser::NodePtr> *rangeArgs =
        nodeListField(**iterNode, "args");
    const std::vector<parser::NodePtr> *rangeKeywords =
        nodeListField(**iterNode, "keywords");
    if (elementValue && *elementValue && elementIndex && *elementIndex &&
        isNameRef(**elementIndex, *targetName) && rangeFunc && *rangeFunc &&
        isNameRef(**rangeFunc, "range") && rangeArgs &&
        rangeArgs->size() == 1 && rangeArgs->front() &&
        (!rangeKeywords || rangeKeywords->empty()) &&
        rangeArgs->front()->kind == "Call") {
      const parser::NodePtr *lenFunc = nodeField(*rangeArgs->front(), "func");
      const std::vector<parser::NodePtr> *lenArgs =
          nodeListField(*rangeArgs->front(), "args");
      const std::vector<parser::NodePtr> *lenKeywords =
          nodeListField(*rangeArgs->front(), "keywords");
      if (lenFunc && *lenFunc && isNameRef(**lenFunc, "len") && lenArgs &&
          lenArgs->size() == 1 && lenArgs->front() &&
          (!lenKeywords || lenKeywords->empty()) &&
          lenArgs->front()->kind == "Name" && (*elementValue)->kind == "Name") {
        const std::string *lenSource = stringField(*lenArgs->front(), "id");
        const std::string *elementSource = stringField(**elementValue, "id");
        if (lenSource && elementSource && *lenSource == *elementSource) {
          Value source = emitExpression(**elementValue);
          if (!source.value)
            return Value{{}, noneType()};
          mlir::Type elementType = listElementType(source.type);
          if (!elementType) {
            error(**elementValue, "range(len(...)) list copy requires a typed "
                                  "list source");
            return Value{{}, noneType()};
          }
          auto found = finiteListElements.find(source.value);
          if (found == finiteListElements.end()) {
            error(**elementValue,
                  "range(len(...)) list copy requires a list source with "
                  "statically known elements");
            return Value{{}, noneType()};
          }
          return emitListFromValues(expr, found->second, elementType);
        }
      }
    }
  }

  std::optional<mlir::Type> targetType = inferRangeTargetType(**iterNode);
  if (!targetType) {
    error(**iterNode, "list comprehension currently supports only range(...) "
                      "over primitive integer values");
    return Value{{}, noneType()};
  }
  if (preferredElementType) {
    if (!mlir::isa<mlir::IntegerType>(preferredElementType)) {
      error(**iterNode, "list comprehension expected element type must be a "
                        "primitive integer when iterating over range(...)");
      return Value{{}, noneType()};
    }
    targetType = preferredElementType;
  }
  auto intTy = mlir::dyn_cast<mlir::IntegerType>(*targetType);
  if (!intTy) {
    error(**iterNode, "list comprehension range target must be a primitive "
                      "integer type");
    return Value{{}, noneType()};
  }

  NameBindingSnapshot saved = snapshotNameBinding(*targetName);
  bindTemporaryName(*targetName, Value{{}, *targetType});
  std::optional<mlir::Type> elementType;
  if (preferredElementType) {
    elementType = preferredElementType;
  } else {
    elementType = inferExpressionType(**elementNode);
  }
  restoreNameBinding(*targetName, std::move(saved));
  if (!elementType) {
    error(**elementNode, "list comprehension element type must be statically "
                         "inferrable");
    return Value{{}, noneType()};
  }

  const parser::NodePtr *func = nodeField(**iterNode, "func");
  const std::vector<parser::NodePtr> *args = nodeListField(**iterNode, "args");
  const std::vector<parser::NodePtr> *keywords =
      nodeListField(**iterNode, "keywords");
  if (!func || !*func || (*func)->kind != "Name" || !args || args->empty() ||
      args->size() > 3 || (keywords && !keywords->empty())) {
    error(**iterNode, "malformed range(...) call in list comprehension");
    return Value{{}, noneType()};
  }

  Value start;
  Value stop;
  Value step;
  auto emitRangeBound = [&](const parser::Node &node) -> Value {
    if (std::optional<PrimitiveConstant> constant =
            primitiveIntConstructorConstant(node))
      return emitPrimitiveIntConstructor(node, *constant);
    if (std::optional<std::int64_t> staticValue = staticIndexValue(node)) {
      mlir::Value value = builder.create<mlir::arith::ConstantIntOp>(
          loc(node), *staticValue, intTy.getWidth());
      return Value{value, *targetType};
    }
    Value value = emitExpression(node);
    if (!value.value)
      return value;
    if (value.type == intType()) {
      if (std::optional<std::int64_t> staticValue =
              staticPyIntValue(value.value)) {
        mlir::Value primitive = builder.create<mlir::arith::ConstantIntOp>(
            loc(node), *staticValue, intTy.getWidth());
        return Value{primitive, *targetType};
      }
    }
    if (value.type != *targetType) {
      error(node, "range(...) bound must match the inferred primitive integer "
                  "target type");
      return Value{{}, *targetType};
    }
    return value;
  };
  if (args->size() == 1) {
    stop = emitRangeBound(*args->front());
    if (!stop.value)
      return Value{{}, noneType()};
    mlir::Value zero = builder.create<mlir::arith::ConstantIntOp>(
        loc(expr), 0, intTy.getWidth());
    mlir::Value one = builder.create<mlir::arith::ConstantIntOp>(
        loc(expr), 1, intTy.getWidth());
    start = Value{zero, *targetType};
    step = Value{one, *targetType};
  } else {
    start = emitRangeBound(*(*args)[0]);
    stop = emitRangeBound(*(*args)[1]);
    if (!start.value || !stop.value)
      return Value{{}, noneType()};
    if (args->size() == 2) {
      mlir::Value one = builder.create<mlir::arith::ConstantIntOp>(
          loc(expr), 1, intTy.getWidth());
      step = Value{one, *targetType};
    } else {
      std::optional<std::int64_t> staticStep;
      if (std::optional<PrimitiveConstant> constant =
              primitiveIntConstructorConstant(*(*args)[2]))
        staticStep = constant->integerValue;
      else
        staticStep = staticIndexValue(*(*args)[2]);
      if (!staticStep || *staticStep <= 0) {
        error(*(*args)[2], "range(...) step in a list comprehension must be a "
                           "positive static primitive integer");
        return Value{{}, noneType()};
      }
      step = emitRangeBound(*(*args)[2]);
      if (!step.value)
        return Value{{}, noneType()};
    }
  }

  auto toIndex = [&](mlir::Value value) {
    return builder.create<mlir::arith::IndexCastOp>(
        loc(expr), builder.getIndexType(), value);
  };
  mlir::Value lower = toIndex(start.value);
  mlir::Value upper = toIndex(stop.value);
  mlir::Value stride = toIndex(step.value);

  mlir::Type resultType = listType(*elementType);
  mlir::Value result = builder.create<py::ListNewOp>(loc(expr), resultType);
  auto loop = builder.create<mlir::scf::ForOp>(loc(expr), lower, upper, stride);
  bool bodyOk = true;
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(loop.getBody());
    mlir::Value induction = builder.create<mlir::arith::IndexCastOp>(
        loc(expr), *targetType, loop.getInductionVar());
    std::map<std::string, Value> outerSymbols = symbols;
    std::map<std::string, FunctionInfo> outerCallableAliases = callableAliases;
    std::map<std::string, PrimitiveConstant> outerConstants =
        primitiveConstants;
    symbols[*targetName] = Value{induction, *targetType};
    callableAliases.erase(*targetName);
    primitiveConstants.erase(*targetName);

    std::function<bool(std::size_t)> emitFilteredAppend =
        [&](std::size_t filterIndex) -> bool {
      if (ifs && filterIndex < ifs->size()) {
        const parser::NodePtr &filter = (*ifs)[filterIndex];
        if (!filter) {
          error(generator, "list comprehension filter is missing");
          return false;
        }
        Value condition = emitCondition(*filter);
        if (!condition.value)
          return false;
        auto ifOp = builder.create<mlir::scf::IfOp>(
            loc(*filter), mlir::TypeRange{}, condition.value,
            /*withElseRegion=*/false);
        {
          mlir::OpBuilder::InsertionGuard ifGuard(builder);
          mlir::Operation *terminator = ifOp.thenBlock()->getTerminator();
          if (terminator)
            builder.setInsertionPoint(terminator);
          else
            builder.setInsertionPointToEnd(ifOp.thenBlock());
          if (!emitFilteredAppend(filterIndex + 1))
            return false;
        }
        builder.setInsertionPointAfter(ifOp);
        return true;
      }

      Value element =
          emitExpressionWithExpectedType(**elementNode, *elementType);
      if (!element.value)
        return false;
      if (element.type != *elementType) {
        error(**elementNode, "list comprehension element type changed from " +
                                 typeString(*elementType) + " to " +
                                 typeString(element.type));
        return false;
      }
      builder.create<py::ListAppendOp>(loc(expr), result, element.value);
      return true;
    };
    bodyOk = emitFilteredAppend(0);
    symbols = std::move(outerSymbols);
    callableAliases = std::move(outerCallableAliases);
    primitiveConstants = std::move(outerConstants);
  }
  builder.setInsertionPointAfter(loop);
  if (!bodyOk)
    return Value{{}, noneType()};
  return Value{result, resultType};
}

std::optional<std::vector<Value>>
Builder::Impl::emitStaticGeneratorElements(const parser::Node &expr,
                                           llvm::StringRef context,
                                           mlir::Type preferredElementType) {
  if (expr.kind != "GeneratorExp") {
    error(expr, context.str() + " requires a generator expression");
    return std::nullopt;
  }

  const parser::NodePtr *elementNode = nodeField(expr, "elt");
  const std::vector<parser::NodePtr> *generators =
      nodeListField(expr, "generators");
  if (!elementNode || !*elementNode || !generators || generators->size() != 1 ||
      !generators->front()) {
    error(expr, context.str() + " generator element or generator clause is "
                                "missing");
    return std::nullopt;
  }

  const parser::Node &generator = *generators->front();
  const parser::NodePtr *targetNode = nodeField(generator, "target");
  const parser::NodePtr *iterNode = nodeField(generator, "iter");
  const std::vector<parser::NodePtr> *ifs = nodeListField(generator, "ifs");
  const parser::FieldValue *isAsyncValue = valueField(generator, "is_async");
  const auto *isAsync =
      isAsyncValue ? std::get_if<std::int64_t>(isAsyncValue) : nullptr;
  if (isAsync && *isAsync != 0) {
    error(generator,
          context.str() + " does not support async generator expressions yet");
    return std::nullopt;
  }
  if (ifs && !ifs->empty()) {
    error(generator, context.str() +
                         " requires generator filters to be statically absent "
                         "so tuple arity is known");
    return std::nullopt;
  }
  if (!targetNode || !*targetNode || (*targetNode)->kind != "Name" ||
      !iterNode || !*iterNode) {
    error(generator,
          context.str() + " currently requires a name target over range(...)");
    return std::nullopt;
  }

  const std::string *targetName = stringField(**targetNode, "id");
  if (!targetName) {
    error(**targetNode, context.str() + " generator target name is missing");
    return std::nullopt;
  }

  std::optional<StaticRangeElements> range =
      emitStaticRangeElements(**iterNode, context, preferredElementType);
  if (!range)
    return std::nullopt;

  NameBindingSnapshot saved = snapshotNameBinding(*targetName);

  std::vector<Value> elements;
  elements.reserve(range->values.size());
  for (const Value &index : range->values) {
    bindTemporaryName(*targetName, index);

    Value element = emitExpression(**elementNode);
    if (!element.value) {
      restoreNameBinding(*targetName, std::move(saved));
      return std::nullopt;
    }
    if (!elements.empty() && element.type != elements.front().type) {
      error(**elementNode, context.str() +
                               " generator element type changed from " +
                               typeString(elements.front().type) + " to " +
                               typeString(element.type));
      restoreNameBinding(*targetName, std::move(saved));
      return std::nullopt;
    }
    elements.push_back(std::move(element));
  }

  restoreNameBinding(*targetName, std::move(saved));
  return elements;
}

Value Builder::Impl::emitTupleLiteral(const parser::Node &expr,
                                      mlir::Type expectedTupleType) {
  const std::vector<parser::NodePtr> *elements = nodeListField(expr, "elts");
  if (!elements) {
    error(expr, "Tuple.elts is missing");
    return Value{{}, noneType()};
  }

  auto expectedTuple = mlir::dyn_cast_if_present<py::TupleType>(
      expectedTupleType ? expectedTupleType : mlir::Type{});
  if (expectedTupleType && !expectedTuple) {
    error(expr, "tuple literal requires a tuple[...] expected type");
    return Value{{}, noneType()};
  }
  llvm::ArrayRef<mlir::Type> expectedElements =
      expectedTuple ? expectedTuple.getElementTypes()
                    : llvm::ArrayRef<mlir::Type>{};

  auto expectedElementType = [&](std::size_t index) -> mlir::Type {
    if (!expectedTuple)
      return {};
    if (expectedElements.empty())
      return {};
    if (expectedElements.size() == 1)
      return expectedElements.front();
    if (index >= expectedElements.size())
      return {};
    return expectedElements[index];
  };

  std::vector<Value> values;
  values.reserve(elements->size());
  for (const parser::NodePtr &element : *elements) {
    if (!element) {
      error(expr, "Tuple contains an empty element");
      return Value{{}, noneType()};
    }
    if (element->kind == "Starred") {
      const parser::NodePtr *starredValue = nodeField(*element, "value");
      if (!starredValue || !*starredValue) {
        error(*element, "starred tuple element is missing a value");
        return Value{{}, noneType()};
      }
      Value sequence = emitExpression(**starredValue);
      if (!sequence.value)
        return Value{{}, noneType()};
      std::optional<std::vector<Value>> expanded =
          finiteSequenceElements(*element, sequence, "starred tuple display");
      if (!expanded)
        return Value{{}, noneType()};
      if (expectedTuple) {
        if (expectedElements.empty() && !expanded->empty()) {
          error(*element, "empty tuple annotation cannot accept starred "
                          "elements");
          return Value{{}, noneType()};
        }
        if (expectedElements.size() > 1 &&
            values.size() + expanded->size() > expectedElements.size()) {
          error(*element, "starred tuple expansion exceeds fixed tuple arity");
          return Value{{}, noneType()};
        }
        for (auto indexed : llvm::enumerate(*expanded)) {
          mlir::Type expected =
              expectedElementType(values.size() + indexed.index());
          if (!expected || indexed.value().type != expected) {
            error(*element, "starred tuple element type mismatch");
            return Value{{}, noneType()};
          }
        }
      }
      values.insert(values.end(), expanded->begin(), expanded->end());
      continue;
    }
    mlir::Type preferred = expectedElementType(values.size());
    if (expectedTuple && !preferred) {
      error(*element, "tuple literal has more elements than its fixed "
                      "expected type");
      return Value{{}, noneType()};
    }
    Value value = preferred
                      ? emitExpressionWithExpectedType(*element, preferred)
                      : emitExpression(*element);
    if (!value.value)
      return Value{{}, noneType()};
    values.push_back(std::move(value));
  }
  if (expectedTuple && expectedElements.size() > 1 &&
      values.size() != expectedElements.size()) {
    error(expr, "tuple literal fixed expected type requires " +
                    std::to_string(expectedElements.size()) +
                    " elements, got " + std::to_string(values.size()));
    return Value{{}, noneType()};
  }
  if (!values.empty()) {
    mlir::Type elementType = values.front().type;
    if (!isTupleLiteralElementSupported(elementType)) {
      error(expr, "tuple literal element type is not yet supported by the "
                  "memref tuple lowering: " +
                      typeString(elementType));
      return Value{{}, noneType()};
    }
    for (const Value &value : values) {
      if (value.type == elementType)
        continue;
      error(expr, "heterogeneous tuple literals are not yet supported by the "
                  "memref tuple lowering");
      return Value{{}, noneType()};
    }
  }
  return emitTuple(values);
}

Value Builder::Impl::emitDict(const parser::Node &expr,
                              mlir::Type preferredKeyType,
                              mlir::Type preferredValueType) {
  const std::vector<parser::NodePtr> *keys = nodeListField(expr, "keys");
  const std::vector<parser::NodePtr> *values = nodeListField(expr, "values");
  if (!keys || !values || keys->size() != values->size()) {
    error(expr, "Dict.keys or Dict.values is missing");
    return Value{{}, noneType()};
  }
  if (keys->empty()) {
    if (!preferredKeyType || !preferredValueType) {
      error(expr, "empty dict literal requires explicit key and value types");
      return Value{{}, noneType()};
    }
    if (!dictStorageSupported(preferredKeyType, preferredValueType)) {
      error(expr, "empty dict annotation key/value types are not supported by "
                  "typed memref lowering yet: " +
                      typeString(preferredKeyType) + ", " +
                      typeString(preferredValueType));
      return Value{{}, noneType()};
    }
    mlir::Type resultType = dictType(preferredKeyType, preferredValueType);
    mlir::Value result = builder.create<py::DictEmptyOp>(loc(expr), resultType);
    return Value{result, resultType};
  }

  std::vector<std::pair<Value, Value>> entries;
  entries.reserve(keys->size());
  for (std::size_t i = 0; i < keys->size(); ++i) {
    if (!(*keys)[i]) {
      error(expr, "dict unpacking is not supported by the C++ emitter");
      return Value{{}, noneType()};
    }
    if (!(*values)[i]) {
      error(expr, "Dict value is missing");
      return Value{{}, noneType()};
    }
    Value key =
        preferredKeyType
            ? emitExpressionWithExpectedType(*(*keys)[i], preferredKeyType)
            : emitExpression(*(*keys)[i]);
    Value value =
        preferredValueType
            ? emitExpressionWithExpectedType(*(*values)[i], preferredValueType)
            : emitExpression(*(*values)[i]);
    if (!key.value || !value.value)
      return Value{{}, noneType()};
    entries.emplace_back(std::move(key), std::move(value));
  }

  mlir::Type keyType =
      preferredKeyType ? preferredKeyType : entries.front().first.type;
  mlir::Type valueType =
      preferredValueType ? preferredValueType : entries.front().second.type;
  for (const auto &[key, value] : entries) {
    if (key.type != keyType) {
      error(expr, "dict literal keys must have the same static type");
      return Value{{}, noneType()};
    }
    if (value.type != valueType) {
      error(expr, "dict literal values must have the same static type");
      return Value{{}, noneType()};
    }
  }
  if (!dictStorageSupported(keyType, valueType)) {
    error(expr, "dict literal key/value types are not supported by typed "
                "memref lowering yet: " +
                    typeString(keyType) + ", " + typeString(valueType));
    return Value{{}, noneType()};
  }

  mlir::Type resultType = dictType(keyType, valueType);
  mlir::Value result = builder.create<py::DictEmptyOp>(loc(expr), resultType);
  for (const auto &[key, value] : entries)
    builder.create<py::DictInsertOp>(loc(expr), result, key.value, value.value);
  return Value{result, resultType};
}

Value Builder::Impl::emitDictComprehension(const parser::Node &expr,
                                           mlir::Type preferredKeyType,
                                           mlir::Type preferredValueType) {
  const parser::NodePtr *keyNode = nodeField(expr, "key");
  const parser::NodePtr *valueNode = nodeField(expr, "value");
  const std::vector<parser::NodePtr> *generators =
      nodeListField(expr, "generators");
  if (!keyNode || !*keyNode || !valueNode || !*valueNode || !generators ||
      generators->size() != 1 || !generators->front()) {
    error(expr, "DictComp.key, DictComp.value, or DictComp.generators is "
                "missing");
    return Value{{}, noneType()};
  }

  const parser::Node &generator = *generators->front();
  const parser::NodePtr *targetNode = nodeField(generator, "target");
  const parser::NodePtr *iterNode = nodeField(generator, "iter");
  const std::vector<parser::NodePtr> *ifs = nodeListField(generator, "ifs");
  const parser::FieldValue *isAsyncValue = valueField(generator, "is_async");
  const auto *isAsync =
      isAsyncValue ? std::get_if<std::int64_t>(isAsyncValue) : nullptr;
  if (isAsync && *isAsync != 0) {
    error(generator, "async dict comprehensions are parsed but async iterator "
                     "lowering is not implemented in the C++ emitter yet");
    return Value{{}, noneType()};
  }
  if (!targetNode || !*targetNode || (*targetNode)->kind != "Name" ||
      !iterNode || !*iterNode) {
    error(generator, "dict comprehension currently requires a name target and "
                     "range(...) iterator");
    return Value{{}, noneType()};
  }
  const std::string *targetName = stringField(**targetNode, "id");
  if (!targetName) {
    error(**targetNode, "dict comprehension target name is missing");
    return Value{{}, noneType()};
  }
  std::optional<mlir::Type> targetType = inferRangeTargetType(**iterNode);
  if (!targetType) {
    error(**iterNode, "dict comprehension currently supports only range(...) "
                      "over primitive integer values");
    return Value{{}, noneType()};
  }
  std::optional<mlir::Type> preferredTargetType;
  auto applyPreferredTarget = [&](const parser::Node &node, mlir::Type type,
                                  llvm::StringRef role) -> bool {
    if (!type || !containsNameRef(node, *targetName))
      return true;
    if (!mlir::isa<mlir::IntegerType>(type)) {
      error(node, "dict comprehension " + role.str() +
                      " expected type cannot drive range(...) because it is "
                      "not a primitive integer");
      return false;
    }
    if (preferredTargetType && *preferredTargetType != type) {
      error(node, "dict comprehension key/value expected types require "
                  "different range target types");
      return false;
    }
    preferredTargetType = type;
    return true;
  };
  if (!applyPreferredTarget(**keyNode, preferredKeyType, "key") ||
      !applyPreferredTarget(**valueNode, preferredValueType, "value"))
    return Value{{}, noneType()};
  if (preferredTargetType)
    targetType = *preferredTargetType;
  auto intTy = mlir::dyn_cast<mlir::IntegerType>(*targetType);
  if (!intTy) {
    error(**iterNode, "dict comprehension range target must be a primitive "
                      "integer type");
    return Value{{}, noneType()};
  }

  NameBindingSnapshot saved = snapshotNameBinding(*targetName);
  bindTemporaryName(*targetName, Value{{}, *targetType});
  std::optional<mlir::Type> keyType =
      preferredKeyType ? std::optional<mlir::Type>{preferredKeyType}
                       : inferExpressionType(**keyNode);
  std::optional<mlir::Type> valueType =
      preferredValueType ? std::optional<mlir::Type>{preferredValueType}
                         : inferExpressionType(**valueNode);
  restoreNameBinding(*targetName, std::move(saved));
  if (!keyType || !valueType) {
    error(expr, "dict comprehension key/value types must be statically "
                "inferrable");
    return Value{{}, noneType()};
  }
  if (!dictStorageSupported(*keyType, *valueType)) {
    error(expr, "dict comprehension key/value types are not supported by typed "
                "memref lowering yet: " +
                    typeString(*keyType) + ", " + typeString(*valueType));
    return Value{{}, noneType()};
  }

  const parser::NodePtr *func = nodeField(**iterNode, "func");
  const std::vector<parser::NodePtr> *args = nodeListField(**iterNode, "args");
  const std::vector<parser::NodePtr> *keywords =
      nodeListField(**iterNode, "keywords");
  if (!func || !*func || (*func)->kind != "Name" || !args || args->empty() ||
      args->size() > 3 || (keywords && !keywords->empty())) {
    error(**iterNode, "malformed range(...) call in dict comprehension");
    return Value{{}, noneType()};
  }

  Value start;
  Value stop;
  Value step;
  auto emitRangeBound = [&](const parser::Node &node) -> Value {
    if (std::optional<PrimitiveConstant> constant =
            primitiveIntConstructorConstant(node))
      return emitPrimitiveIntConstructor(node, *constant);
    if (std::optional<std::int64_t> staticValue = staticIndexValue(node)) {
      mlir::Value value = builder.create<mlir::arith::ConstantIntOp>(
          loc(node), *staticValue, intTy.getWidth());
      return Value{value, *targetType};
    }
    Value value = emitExpression(node);
    if (!value.value)
      return value;
    if (value.type == intType()) {
      if (std::optional<std::int64_t> staticValue =
              staticPyIntValue(value.value)) {
        mlir::Value primitive = builder.create<mlir::arith::ConstantIntOp>(
            loc(node), *staticValue, intTy.getWidth());
        return Value{primitive, *targetType};
      }
    }
    if (value.type != *targetType) {
      error(node, "range(...) bound must match the inferred primitive integer "
                  "target type");
      return Value{{}, *targetType};
    }
    return value;
  };
  if (args->size() == 1) {
    stop = emitRangeBound(*args->front());
    if (!stop.value)
      return Value{{}, noneType()};
    mlir::Value zero = builder.create<mlir::arith::ConstantIntOp>(
        loc(expr), 0, intTy.getWidth());
    mlir::Value one = builder.create<mlir::arith::ConstantIntOp>(
        loc(expr), 1, intTy.getWidth());
    start = Value{zero, *targetType};
    step = Value{one, *targetType};
  } else {
    start = emitRangeBound(*(*args)[0]);
    stop = emitRangeBound(*(*args)[1]);
    if (!start.value || !stop.value)
      return Value{{}, noneType()};
    if (args->size() == 2) {
      mlir::Value one = builder.create<mlir::arith::ConstantIntOp>(
          loc(expr), 1, intTy.getWidth());
      step = Value{one, *targetType};
    } else {
      std::optional<std::int64_t> staticStep;
      if (std::optional<PrimitiveConstant> constant =
              primitiveIntConstructorConstant(*(*args)[2]))
        staticStep = constant->integerValue;
      else
        staticStep = staticIndexValue(*(*args)[2]);
      if (!staticStep || *staticStep <= 0) {
        error(*(*args)[2], "range(...) step in a dict comprehension must be a "
                           "positive static primitive integer");
        return Value{{}, noneType()};
      }
      step = emitRangeBound(*(*args)[2]);
      if (!step.value)
        return Value{{}, noneType()};
    }
  }

  auto toIndex = [&](mlir::Value value) {
    return builder.create<mlir::arith::IndexCastOp>(
        loc(expr), builder.getIndexType(), value);
  };
  mlir::Value lower = toIndex(start.value);
  mlir::Value upper = toIndex(stop.value);
  mlir::Value stride = toIndex(step.value);

  mlir::Type resultType = dictType(*keyType, *valueType);
  mlir::Value result = builder.create<py::DictEmptyOp>(loc(expr), resultType);
  auto loop = builder.create<mlir::scf::ForOp>(loc(expr), lower, upper, stride);
  bool bodyOk = true;
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(loop.getBody());
    mlir::Value induction = builder.create<mlir::arith::IndexCastOp>(
        loc(expr), *targetType, loop.getInductionVar());
    std::map<std::string, Value> outerSymbols = symbols;
    std::map<std::string, FunctionInfo> outerCallableAliases = callableAliases;
    std::map<std::string, PrimitiveConstant> outerConstants =
        primitiveConstants;
    symbols[*targetName] = Value{induction, *targetType};
    callableAliases.erase(*targetName);
    primitiveConstants.erase(*targetName);

    std::function<bool(std::size_t)> emitFilteredInsert =
        [&](std::size_t filterIndex) -> bool {
      if (ifs && filterIndex < ifs->size()) {
        const parser::NodePtr &filter = (*ifs)[filterIndex];
        if (!filter) {
          error(generator, "dict comprehension filter is missing");
          return false;
        }
        Value condition = emitCondition(*filter);
        if (!condition.value)
          return false;
        auto ifOp = builder.create<mlir::scf::IfOp>(
            loc(*filter), mlir::TypeRange{}, condition.value,
            /*withElseRegion=*/false);
        {
          mlir::OpBuilder::InsertionGuard ifGuard(builder);
          mlir::Operation *terminator = ifOp.thenBlock()->getTerminator();
          if (terminator)
            builder.setInsertionPoint(terminator);
          else
            builder.setInsertionPointToEnd(ifOp.thenBlock());
          if (!emitFilteredInsert(filterIndex + 1))
            return false;
        }
        builder.setInsertionPointAfter(ifOp);
        return true;
      }

      Value key = emitExpressionWithExpectedType(**keyNode, *keyType);
      Value value = emitExpressionWithExpectedType(**valueNode, *valueType);
      if (!key.value || !value.value)
        return false;
      if (key.type != *keyType) {
        error(**keyNode, "dict comprehension key type changed from " +
                             typeString(*keyType) + " to " +
                             typeString(key.type));
        return false;
      }
      if (value.type != *valueType) {
        error(**valueNode, "dict comprehension value type changed from " +
                               typeString(*valueType) + " to " +
                               typeString(value.type));
        return false;
      }
      builder.create<py::DictInsertOp>(loc(expr), result, key.value,
                                       value.value);
      return true;
    };
    bodyOk = emitFilteredInsert(0);
    symbols = std::move(outerSymbols);
    callableAliases = std::move(outerCallableAliases);
    primitiveConstants = std::move(outerConstants);
  }
  builder.setInsertionPointAfter(loop);
  if (!bodyOk)
    return Value{{}, noneType()};
  return Value{result, resultType};
}

Value Builder::Impl::emitGetItemAccess(const parser::Node &expr,
                                       Value container,
                                       const parser::Node &indexNode,
                                       llvm::StringRef contextLabel) {
  if (std::optional<Value> concrete = concreteProtocolValue(container))
    container = *concrete;

  if (auto tupleType = mlir::dyn_cast<py::TupleType>(container.type)) {
    if (indexNode.kind == "Slice") {
      error(indexNode, contextLabel.str() +
                           " slice lowering is handled before __getitem__");
      return Value{{}, noneType()};
    }
    llvm::ArrayRef<mlir::Type> elementTypes = tupleType.getElementTypes();
    std::optional<std::int64_t> index = staticIndexValue(indexNode);
    if (!index || *index < 0) {
      error(indexNode, contextLabel.str() +
                           " for tuple requires a non-negative static "
                           "integer index");
      return Value{{}, noneType()};
    }
    if (elementTypes.empty()) {
      error(expr, "cannot subscript an empty tuple");
      return Value{{}, noneType()};
    }
    mlir::Type elementType = elementTypes.front();
    if (elementTypes.size() > 1) {
      if (*index >= static_cast<std::int64_t>(elementTypes.size())) {
        error(indexNode, "tuple subscript index is out of range");
        return Value{{}, noneType()};
      }
      elementType = elementTypes[*index];
    }
    std::optional<protocols::ProtocolMethod> getitemContract =
        resolveProtocolMethodContract(expr, container.type, "__getitem__",
                                      {intType()}, contextLabel);
    if (!getitemContract)
      return Value{{}, elementType};
    llvm::ArrayRef<mlir::Type> results =
        getitemContract->signature.getResultTypes();
    if (results.size() != 1 || results.front() != elementType) {
      error(expr, "Sequence.__getitem__ result mismatch: expected " +
                      typeString(elementType));
      return Value{{}, elementType};
    }
    mlir::Value indexValue =
        builder.create<mlir::arith::ConstantIndexOp>(loc(indexNode), *index);
    mlir::Value result = builder.create<py::GetItemOp>(
        loc(expr), elementType, getitemContract->signature, container.value,
        indexValue);
    return Value{result, elementType};
  }

  mlir::Type elementType = listElementType(container.type);
  if (elementType) {
    mlir::Type indexType = builder.getI64Type();
    Value index = emitExpressionWithExpectedType(indexNode, indexType);
    if (!index.value)
      return Value{{}, elementType};
    if (!mlir::isa<py::IntType, mlir::IntegerType>(index.type)) {
      if (std::optional<Value> converted = emitProtocolUnaryConversionCall(
              indexNode, index, "SupportsIndex", "__index__", intType())) {
        index = *converted;
        if (!index.value)
          return Value{{}, elementType};
      }
    }
    std::optional<protocols::ProtocolMethod> getitemContract =
        resolveProtocolMethodContract(expr, container.type, "__getitem__",
                                      {index.type}, contextLabel);
    if (!getitemContract)
      return Value{{}, elementType};
    llvm::ArrayRef<mlir::Type> results =
        getitemContract->signature.getResultTypes();
    if (results.size() != 1 || results.front() != elementType) {
      error(expr, "Sequence.__getitem__ result mismatch: expected " +
                      typeString(elementType));
      return Value{{}, elementType};
    }
    mlir::Value result = builder.create<py::GetItemOp>(
        loc(expr), elementType, getitemContract->signature, container.value,
        index.value);
    return Value{result, elementType};
  }

  if (std::optional<std::pair<mlir::Type, mlir::Type>> dictTypes =
          dictKeyValueTypes(container.type)) {
    Value index = emitExpressionWithExpectedType(indexNode, dictTypes->first);
    if (!index.value)
      return Value{{}, dictTypes->second};
    if (!dictStorageSupported(dictTypes->first, dictTypes->second)) {
      error(expr, "dict subscript key/value types are not supported by typed "
                  "memref lowering yet: " +
                      typeString(dictTypes->first) + ", " +
                      typeString(dictTypes->second));
      return Value{{}, dictTypes->second};
    }
    if (index.type != dictTypes->first) {
      error(expr, "dict subscript key type mismatch: expected " +
                      typeString(dictTypes->first) + ", got " +
                      typeString(index.type));
      return Value{{}, dictTypes->second};
    }
    std::optional<protocols::ProtocolMethod> getitemContract =
        resolveProtocolMethodContract(expr, container.type, "__getitem__",
                                      {index.type}, contextLabel);
    if (!getitemContract)
      return Value{{}, dictTypes->second};
    llvm::ArrayRef<mlir::Type> results =
        getitemContract->signature.getResultTypes();
    if (results.size() != 1 || results.front() != dictTypes->second) {
      error(expr, "Mapping.__getitem__ result mismatch: expected " +
                      typeString(dictTypes->second));
      return Value{{}, dictTypes->second};
    }
    mlir::Value result = builder.create<py::GetItemOp>(
        loc(expr), dictTypes->second, getitemContract->signature,
        container.value, index.value);
    return Value{result, dictTypes->second};
  }

  if (mlir::isa<py::ClassType>(container.type)) {
    if (indexNode.kind == "Slice") {
      error(indexNode, contextLabel.str() +
                           " class receiver slice lowering is not "
                           "implemented yet");
      return Value{{}, noneType()};
    }
    std::optional<FunctionInfo> method =
        resolveClassMethod(expr, container, "__getitem__");
    if (!method)
      return Value{{}, noneType()};
    if (method->argTypes.size() != 2) {
      error(expr, "class __getitem__ on " + typeString(container.type) +
                      " must take exactly one index argument");
      return Value{{}, method->resultType};
    }
    mlir::Type expectedIndexType = method->argTypes[1];
    Value index = emitExpressionWithExpectedType(indexNode, expectedIndexType);
    if (!index.value)
      return Value{{}, method->resultType};
    return emitResolvedMethodCall(expr, container, *method, {index});
  }

  if (mlir::isa<py::ProtocolType>(container.type)) {
    std::optional<mlir::Type> indexType = inferExpressionType(indexNode);
    if (!indexType) {
      error(indexNode, "protocol __getitem__ requires a statically known "
                       "index type");
      return Value{{}, noneType()};
    }
    std::optional<mlir::Type> resultType = resolveProtocolMethodResult(
        expr, container.type, "__getitem__", {*indexType}, contextLabel);
    if (!resultType)
      return Value{{}, noneType()};
    error(expr, "protocol __getitem__ on " + typeString(container.type) +
                    " resolves statically to " + typeString(*resultType) +
                    ", but lowering for protocol-typed receivers is not "
                    "implemented yet");
    return Value{{}, *resultType};
  }

  error(expr,
        contextLabel.str() + " supports only typed tuples, lists and dicts");
  return Value{{}, noneType()};
}

Value Builder::Impl::emitSubscript(const parser::Node &expr) {
  const parser::NodePtr *containerNode = nodeField(expr, "value");
  const parser::NodePtr *indexNode = nodeField(expr, "slice");
  if (!containerNode || !*containerNode || !indexNode || !*indexNode) {
    error(expr, "Subscript.value or Subscript.slice is missing");
    return Value{{}, noneType()};
  }

  Value container = emitExpression(**containerNode);
  if (!container.value)
    return Value{{}, noneType()};
  if (std::optional<Value> concrete = concreteProtocolValue(container))
    container = *concrete;
  if (auto tupleType = mlir::dyn_cast<py::TupleType>(container.type)) {
    auto elementTypes = tupleType.getElementTypes();
    if ((*indexNode)->kind == "Slice") {
      if (elementTypes.size() == 1) {
        error(**indexNode,
              "tuple slice requires a statically finite tuple length");
        return Value{{}, noneType()};
      }

      const parser::NodePtr *lowerNode = nodeField(**indexNode, "lower");
      const parser::NodePtr *upperNode = nodeField(**indexNode, "upper");
      const parser::NodePtr *stepNode = nodeField(**indexNode, "step");
      std::optional<std::int64_t> lower = lowerNode && *lowerNode
                                              ? staticIndexValue(**lowerNode)
                                              : std::optional<std::int64_t>{};
      std::optional<std::int64_t> upper = upperNode && *upperNode
                                              ? staticIndexValue(**upperNode)
                                              : std::optional<std::int64_t>{};
      std::optional<std::int64_t> step = stepNode && *stepNode
                                             ? staticIndexValue(**stepNode)
                                             : std::optional<std::int64_t>{1};
      if ((lowerNode && *lowerNode && !lower) ||
          (upperNode && *upperNode && !upper) ||
          (stepNode && *stepNode && !step)) {
        error(**indexNode,
              "tuple slice bounds must be statically known integers");
        return Value{{}, noneType()};
      }
      if (*step == 0) {
        error(**indexNode, "tuple slice step cannot be zero");
        return Value{{}, noneType()};
      }

      const std::int64_t length =
          static_cast<std::int64_t>(elementTypes.size());
      std::int64_t start = 0;
      std::int64_t stop = length;
      if (*step < 0) {
        start = length - 1;
        stop = -1;
      }
      if (lower)
        start = *step > 0 ? clampSliceIndex(*lower, length, 0, length)
                          : clampSliceIndex(*lower, length, -1, length - 1);
      if (upper)
        stop = *step > 0 ? clampSliceIndex(*upper, length, 0, length)
                         : clampSliceIndex(*upper, length, -1, length - 1);

      std::vector<Value> elements;
      for (std::int64_t i = start; *step > 0 ? i < stop : i > stop;
           i += *step) {
        mlir::Type elementType = elementTypes[static_cast<std::size_t>(i)];
        mlir::Value indexValue =
            builder.create<mlir::arith::ConstantIndexOp>(loc(**indexNode), i);
        py::CallableType contract = unaryMethodContract(
            container.type, indexValue.getType(), elementType);
        mlir::Value result = builder.create<py::GetItemOp>(
            loc(expr), elementType, contract, container.value, indexValue);
        elements.push_back(Value{result, elementType});
      }
      return emitTuple(elements);
    }
  }

  if ((*indexNode)->kind == "Slice") {
    mlir::Type elementType = listElementType(container.type);
    if (!elementType) {
      error(**indexNode, "slice subscript currently supports only statically "
                         "finite typed tuples and lists");
      return Value{{}, noneType()};
    }

    auto found = finiteListElements.find(container.value);
    if (found == finiteListElements.end()) {
      error(**indexNode,
            "list slice requires a list source with statically known elements");
      return Value{{}, noneType()};
    }

    const parser::NodePtr *lowerNode = nodeField(**indexNode, "lower");
    const parser::NodePtr *upperNode = nodeField(**indexNode, "upper");
    const parser::NodePtr *stepNode = nodeField(**indexNode, "step");
    std::optional<std::int64_t> lower = lowerNode && *lowerNode
                                            ? staticIndexValue(**lowerNode)
                                            : std::optional<std::int64_t>{};
    std::optional<std::int64_t> upper = upperNode && *upperNode
                                            ? staticIndexValue(**upperNode)
                                            : std::optional<std::int64_t>{};
    std::optional<std::int64_t> step = stepNode && *stepNode
                                           ? staticIndexValue(**stepNode)
                                           : std::optional<std::int64_t>{1};
    if ((lowerNode && *lowerNode && !lower) ||
        (upperNode && *upperNode && !upper) ||
        (stepNode && *stepNode && !step)) {
      error(**indexNode, "list slice bounds must be statically known integers");
      return Value{{}, noneType()};
    }
    if (*step == 0) {
      error(**indexNode, "list slice step cannot be zero");
      return Value{{}, noneType()};
    }

    const std::int64_t length = static_cast<std::int64_t>(found->second.size());
    std::int64_t start = 0;
    std::int64_t stop = length;
    if (*step < 0) {
      start = length - 1;
      stop = -1;
    }
    if (lower)
      start = *step > 0 ? clampSliceIndex(*lower, length, 0, length)
                        : clampSliceIndex(*lower, length, -1, length - 1);
    if (upper)
      stop = *step > 0 ? clampSliceIndex(*upper, length, 0, length)
                       : clampSliceIndex(*upper, length, -1, length - 1);

    std::vector<Value> elements;
    for (std::int64_t i = start; *step > 0 ? i < stop : i > stop; i += *step)
      elements.push_back(found->second[static_cast<std::size_t>(i)]);
    return emitListFromValues(expr, elements, elementType);
  }

  if ((*indexNode)->kind == "Slice") {
    error(**indexNode, "slice subscript currently supports only statically "
                       "finite typed tuples and lists");
    return Value{{}, noneType()};
  }

  return emitGetItemAccess(expr, container, **indexNode, "subscript access");
}

Value Builder::Impl::emitListMethodCall(
    const parser::Node &expr, const Value &receiver, llvm::StringRef methodName,
    const std::vector<parser::NodePtr> &args) {
  mlir::Type elementType = listElementType(receiver.type);
  if (!elementType) {
    error(expr, "list method receiver is not a typed list");
    return Value{{}, noneType()};
  }
  if (args.size() != 1 || !args.front()) {
    error(expr, "list." + methodName.str() + " expects exactly one argument");
    return Value{{}, noneType()};
  }

  Value value = emitExpressionWithExpectedType(*args.front(), elementType);
  if (!value.value)
    return Value{{}, noneType()};
  std::optional<mlir::Type> resultType =
      resolveProtocolMethodResult(expr, receiver.type, methodName, {value.type},
                                  "list." + methodName.str());
  if (!resultType)
    return Value{{}, noneType()};
  if (value.type != elementType) {
    error(*args.front(),
          "list." + methodName.str() + " argument type mismatch: expected " +
              typeString(elementType) + ", got " + typeString(value.type));
    return Value{{}, noneType()};
  }

  if (methodName == "count") {
    if (*resultType != intType()) {
      error(expr,
            "list.count contract result mismatch: expected !py.int, got " +
                typeString(*resultType));
      return Value{{}, intType()};
    }
    if (!mlir::isa<py::IntType, py::FloatType, py::BoolType, py::StrType>(
            elementType)) {
      error(expr, "list.count currently supports only int, float, bool, or "
                  "str element equality lowering");
      return Value{{}, intType()};
    }

    std::optional<protocols::ProtocolMethod> lenContract =
        resolveProtocolMethodContract(expr, receiver.type, "__len__", {},
                                      "list.count length");
    if (!lenContract)
      return Value{{}, intType()};
    llvm::ArrayRef<mlir::Type> lenResults =
        lenContract->signature.getResultTypes();
    if (lenResults.size() != 1 || lenResults.front() != intType()) {
      error(expr, "list.count length contract on " + typeString(receiver.type) +
                      " must return !py.int");
      return Value{{}, intType()};
    }

    mlir::Value lengthInt = builder.create<py::LenOp>(
        loc(expr), intType(), lenContract->signature, receiver.value);
    mlir::Value stop = builder.create<py::CastToPrimOp>(
        loc(expr), builder.getI64Type(), lengthInt, "exact");
    mlir::Value start =
        builder.create<mlir::arith::ConstantIntOp>(loc(expr), 0, 64);
    mlir::Value step =
        builder.create<mlir::arith::ConstantIntOp>(loc(expr), 1, 64);
    mlir::Value initialCount =
        builder.create<mlir::arith::ConstantIntOp>(loc(expr), 0, 64);

    mlir::Region *region = builder.getBlock()->getParent();
    mlir::Block *condBlock = new mlir::Block();
    mlir::Block *bodyBlock = new mlir::Block();
    mlir::Block *stepBlock = new mlir::Block();
    mlir::Block *afterBlock = new mlir::Block();
    region->push_back(condBlock);
    region->push_back(bodyBlock);
    region->push_back(stepBlock);
    region->push_back(afterBlock);

    condBlock->addArgument(builder.getI64Type(), loc(expr));
    condBlock->addArgument(builder.getI64Type(), loc(expr));
    bodyBlock->addArgument(builder.getI64Type(), loc(expr));
    bodyBlock->addArgument(builder.getI64Type(), loc(expr));
    stepBlock->addArgument(builder.getI64Type(), loc(expr));
    stepBlock->addArgument(builder.getI64Type(), loc(expr));
    afterBlock->addArgument(builder.getI64Type(), loc(expr));

    builder.create<mlir::cf::BranchOp>(loc(expr), condBlock,
                                       mlir::ValueRange{start, initialCount});

    builder.setInsertionPointToStart(condBlock);
    mlir::Value condIv = condBlock->getArgument(0);
    mlir::Value condCount = condBlock->getArgument(1);
    mlir::Value keepGoing = builder.create<mlir::arith::CmpIOp>(
        loc(expr), mlir::arith::CmpIPredicate::slt, condIv, stop);
    builder.create<mlir::cf::CondBranchOp>(
        loc(expr), keepGoing, bodyBlock, mlir::ValueRange{condIv, condCount},
        afterBlock, mlir::ValueRange{condCount});

    builder.setInsertionPointToStart(bodyBlock);
    mlir::Value bodyIv = bodyBlock->getArgument(0);
    mlir::Value bodyCount = bodyBlock->getArgument(1);
    py::CallableType getitemContract =
        unaryMethodContract(receiver.type, bodyIv.getType(), elementType);
    mlir::Value element = builder.create<py::GetItemOp>(
        loc(expr), elementType, getitemContract, receiver.value, bodyIv);
    mlir::Value equals =
        builder.create<py::EqOp>(loc(expr), boolType(), value.value, element);
    mlir::Value equalsBit =
        builder.create<py::CastToPrimOp>(loc(expr), i1Type(), equals, "exact");
    mlir::Value one =
        builder.create<mlir::arith::ConstantIntOp>(loc(expr), 1, 64);
    mlir::Value incremented =
        builder.create<mlir::arith::AddIOp>(loc(expr), bodyCount, one);
    mlir::Value nextCount = builder.create<mlir::arith::SelectOp>(
        loc(expr), equalsBit, incremented, bodyCount);
    builder.create<mlir::cf::BranchOp>(loc(expr), stepBlock,
                                       mlir::ValueRange{bodyIv, nextCount});

    builder.setInsertionPointToStart(stepBlock);
    mlir::Value nextIv = builder.create<mlir::arith::AddIOp>(
        loc(expr), stepBlock->getArgument(0), step);
    builder.create<mlir::cf::BranchOp>(
        loc(expr), condBlock,
        mlir::ValueRange{nextIv, stepBlock->getArgument(1)});

    builder.setInsertionPointToStart(afterBlock);
    mlir::Value result = builder.create<py::CastFromPrimOp>(
        loc(expr), intType(), afterBlock->getArgument(0));
    return Value{result, intType()};
  }

  if (methodName == "append") {
    finiteListElements.erase(receiver.value);
    builder.create<py::ListAppendOp>(loc(expr), receiver.value, value.value);
  } else if (methodName == "remove") {
    finiteListElements.erase(receiver.value);
    builder.create<py::ListRemoveOp>(loc(expr), receiver.value, value.value);
  } else {
    error(expr, "unsupported typed list method '" + methodName.str() + "'");
  }

  mlir::Value none = builder.create<py::NoneOp>(loc(expr), noneType());
  return Value{none, noneType()};
}

Value Builder::Impl::emitTupleCountMethodCall(
    const parser::Node &expr, const Value &receiver,
    const std::vector<parser::NodePtr> &args) {
  auto tupleType = mlir::dyn_cast<py::TupleType>(receiver.type);
  if (!tupleType) {
    error(expr, "tuple.count receiver is not a typed tuple");
    return Value{{}, intType()};
  }
  if (args.size() != 1 || !args.front()) {
    error(expr, "tuple.count expects exactly one argument");
    return Value{{}, intType()};
  }

  llvm::ArrayRef<mlir::Type> elementTypes = tupleType.getElementTypes();
  if (elementTypes.empty()) {
    mlir::Value zero =
        builder.create<mlir::arith::ConstantIntOp>(loc(expr), 0, 64);
    mlir::Value result =
        builder.create<py::CastFromPrimOp>(loc(expr), intType(), zero);
    return Value{result, intType()};
  }
  mlir::Type elementType = elementTypes.front();
  for (mlir::Type current : elementTypes) {
    if (current != elementType) {
      error(expr, "tuple.count currently requires a homogeneous tuple element "
                  "type");
      return Value{{}, intType()};
    }
  }
  if (!mlir::isa<py::IntType, py::FloatType, py::BoolType, py::StrType>(
          elementType)) {
    error(expr, "tuple.count currently supports only int, float, bool, or "
                "str element equality lowering");
    return Value{{}, intType()};
  }

  Value value = emitExpressionWithExpectedType(*args.front(), elementType);
  if (!value.value)
    return Value{{}, intType()};
  if (value.type != elementType) {
    error(*args.front(), "tuple.count argument type mismatch: expected " +
                             typeString(elementType) + ", got " +
                             typeString(value.type));
    return Value{{}, intType()};
  }

  std::optional<mlir::Type> resultType = resolveProtocolMethodResult(
      expr, receiver.type, "count", {value.type}, "tuple.count");
  if (!resultType)
    return Value{{}, intType()};
  if (*resultType != intType()) {
    error(expr, "tuple.count does not satisfy Sequence.count");
    return Value{{}, intType()};
  }

  mlir::Value count =
      builder.create<mlir::arith::ConstantIntOp>(loc(expr), 0, 64);
  mlir::Value one =
      builder.create<mlir::arith::ConstantIntOp>(loc(expr), 1, 64);
  for (auto indexed : llvm::enumerate(elementTypes)) {
    mlir::Value index = builder.create<mlir::arith::ConstantIndexOp>(
        loc(expr), static_cast<std::int64_t>(indexed.index()));
    py::CallableType getitemContract =
        unaryMethodContract(receiver.type, index.getType(), elementType);
    mlir::Value element = builder.create<py::GetItemOp>(
        loc(expr), elementType, getitemContract, receiver.value, index);
    mlir::Value equals =
        builder.create<py::EqOp>(loc(expr), boolType(), value.value, element);
    mlir::Value equalsBit =
        builder.create<py::CastToPrimOp>(loc(expr), i1Type(), equals, "exact");
    mlir::Value incremented =
        builder.create<mlir::arith::AddIOp>(loc(expr), count, one);
    count = builder.create<mlir::arith::SelectOp>(loc(expr), equalsBit,
                                                  incremented, count);
  }
  mlir::Value result =
      builder.create<py::CastFromPrimOp>(loc(expr), intType(), count);
  return Value{result, intType()};
}

Value Builder::Impl::emitDictGetMethodCall(
    const parser::Node &expr, const Value &receiver,
    const std::vector<parser::NodePtr> &args) {
  std::optional<std::pair<mlir::Type, mlir::Type>> dictTypes =
      dictKeyValueTypes(receiver.type);
  if (!dictTypes) {
    error(expr, "dict.get receiver is not a typed dict");
    return Value{{}, noneType()};
  }
  if (args.size() != 1 && args.size() != 2) {
    error(expr, "dict.get expects one key argument and an optional default");
    return Value{{}, dictTypes->second};
  }
  if (!args.front()) {
    error(expr, "dict.get key argument is missing");
    return Value{{}, dictTypes->second};
  }

  Value key = emitExpressionWithExpectedType(*args.front(), dictTypes->first);
  if (!key.value)
    return Value{{}, dictTypes->second};
  if (key.type != dictTypes->first) {
    error(*args.front(), "dict.get key type mismatch: expected " +
                             typeString(dictTypes->first) + ", got " +
                             typeString(key.type));
    return Value{{}, dictTypes->second};
  }

  std::vector<mlir::Type> argumentTypes{key.type};
  Value defaultValue{{}, dictTypes->second};
  if (args.size() == 2) {
    if (!args[1]) {
      error(expr, "dict.get default argument is missing");
      return Value{{}, dictTypes->second};
    }
    defaultValue = emitExpression(*args[1]);
    if (!defaultValue.value)
      return Value{{}, dictTypes->second};
    argumentTypes.push_back(defaultValue.type);
  }

  std::optional<mlir::Type> resultType = resolveProtocolMethodResult(
      expr, receiver.type, "get", argumentTypes, "dict.get");
  if (!resultType)
    return Value{{}, dictTypes->second};
  std::optional<protocols::ProtocolMethod> containsContract =
      resolveProtocolMethodContract(expr, receiver.type, "__contains__",
                                    {key.type}, "dict.get contains");
  if (!containsContract)
    return Value{{}, dictTypes->second};
  std::optional<protocols::ProtocolMethod> getitemContract =
      resolveProtocolMethodContract(expr, receiver.type, "__getitem__",
                                    {key.type}, "dict.get item");
  if (!getitemContract)
    return Value{{}, dictTypes->second};
  llvm::ArrayRef<mlir::Type> itemResults =
      getitemContract->signature.getResultTypes();
  if (itemResults.size() != 1 || itemResults.front() != dictTypes->second) {
    error(expr, "dict.get item contract on " + typeString(receiver.type) +
                    " must return " + typeString(dictTypes->second));
    return Value{{}, dictTypes->second};
  }
  mlir::Value contains = builder.create<py::ContainsOp>(
      loc(expr), i1Type(), containsContract->signature, receiver.value,
      key.value);

  auto ifOp = builder.create<mlir::scf::IfOp>(
      loc(expr), mlir::TypeRange{*resultType}, contains,
      /*withElseRegion=*/true);
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(ifOp.thenBlock());
    mlir::Value value = builder.create<py::GetItemOp>(
        loc(expr), dictTypes->second, getitemContract->signature,
        receiver.value, key.value);
    if (value.getType() != *resultType) {
      if (!mlir::isa<py::UnionType>(*resultType)) {
        error(expr, "dict.get value result mismatch: expected " +
                        typeString(*resultType) + ", got " +
                        typeString(value.getType()));
        value = {};
      } else {
        value = builder.create<py::UnionWrapOp>(loc(expr), *resultType, value);
      }
    }
    if (!value)
      value = builder.create<py::NoneOp>(loc(expr), noneType());
    builder.create<mlir::scf::YieldOp>(loc(expr), value);
  }
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(ifOp.elseBlock());
    mlir::Value value;
    if (defaultValue.value) {
      value = defaultValue.value;
      if (value.getType() != *resultType) {
        if (!mlir::isa<py::UnionType>(*resultType)) {
          error(expr, "dict.get default result mismatch: expected " +
                          typeString(*resultType) + ", got " +
                          typeString(value.getType()));
          value = {};
        } else {
          value =
              builder.create<py::UnionWrapOp>(loc(expr), *resultType, value);
        }
      }
    } else {
      value = builder.create<py::NoneOp>(loc(expr), noneType());
      if (value.getType() != *resultType)
        value = builder.create<py::UnionWrapOp>(loc(expr), *resultType, value);
    }
    if (!value)
      value = builder.create<py::NoneOp>(loc(expr), noneType());
    builder.create<mlir::scf::YieldOp>(loc(expr), value);
  }
  builder.setInsertionPointAfter(ifOp);
  return Value{ifOp.getResult(0), *resultType};
}

std::optional<FunctionInfo>
Builder::Impl::resolveClassMethod(const parser::Node &anchor,
                                  const Value &receiver,
                                  llvm::StringRef methodName) {
  std::optional<std::string> staticClassName = classNameFromType(receiver.type);
  if (!staticClassName) {
    error(anchor, "method call requires a statically known receiver type");
    return std::nullopt;
  }
  std::string className =
      receiver.exactClass
          ? *receiver.exactClass
          : (receiver.provenClass ? *receiver.provenClass : *staticClassName);

  auto hasKnownSubclassOverride = [&](llvm::StringRef baseName) {
    for (const auto &[candidateName, candidateInfo] : classes) {
      if (candidateName == baseName)
        continue;
      if (!classSubtypeOf(candidateName, baseName))
        continue;
      if (candidateInfo.ownMethodNodes.count(methodName.str()))
        return true;
    }
    return false;
  };

  if (!receiver.exactClass && hasKnownSubclassOverride(className)) {
    error(anchor, "dynamic dispatch for method '" + methodName.str() +
                      "' on non-exact class '" + className +
                      "' is not implemented; exact class facts are required "
                      "for devirtualization when subclasses override it");
    return std::nullopt;
  }

  auto classFound = classes.find(className);
  if (classFound == classes.end()) {
    error(anchor, "unknown class '" + className + "'");
    return std::nullopt;
  }
  auto methodFound = classFound->second.methods.find(methodName.str());
  if (methodFound == classFound->second.methods.end()) {
    error(anchor,
          "class '" + className + "' has no method '" + methodName.str() + "'");
    return std::nullopt;
  }
  auto functionFound = functions.find(methodFound->second);
  if (functionFound == functions.end()) {
    error(anchor,
          "method function '" + methodFound->second + "' is not defined");
    return std::nullopt;
  }
  return functionFound->second;
}

Value Builder::Impl::emitResolvedMethodCall(const parser::Node &anchor,
                                            const Value &receiver,
                                            const FunctionInfo &method,
                                            llvm::ArrayRef<Value> userArgs) {
  std::optional<std::vector<Value>> prepared =
      prepareResolvedMethodCallArguments(anchor, receiver, method, userArgs);
  if (!prepared)
    return Value{{}, method.resultType};
  std::vector<Value> callArgs = std::move(*prepared);

  if (!method.mayThrow && method.returnedCallableArgIndex) {
    std::size_t index = *method.returnedCallableArgIndex;
    if (index >= callArgs.size()) {
      error(anchor, "returned callable argument summary for method '" +
                        method.name + "' has invalid argument index");
      return Value{{}, method.resultType};
    }
    Value returned = callArgs[index];
    if (!py::isCallableType(returned.type)) {
      error(anchor, "returned callable argument for method '" + method.name +
                        "' must be Callable, got " + typeString(returned.type));
      return Value{{}, method.resultType};
    }
    return returned;
  }

  if (!method.mayThrow && method.returnedCallable) {
    FunctionInfo returned = method.returnedCallable->info;
    if (returned.closureCaptures.size() !=
        method.returnedCallable->closureCaptureArgIndices.size()) {
      error(anchor, "returned callable summary for method '" + method.name +
                        "' has inconsistent closure metadata");
      return Value{{}, method.resultType};
    }
    for (std::size_t index = 0; index < returned.closureCaptures.size();
         ++index) {
      std::optional<std::size_t> argIndex =
          method.returnedCallable->closureCaptureArgIndices[index];
      if (!argIndex || *argIndex >= callArgs.size()) {
        error(anchor, "returned callable summary for method '" + method.name +
                          "' cannot map closure capture '" +
                          returned.closureCaptures[index].name + "'");
        return Value{{}, method.resultType};
      }
      returned.closureCaptures[index].value = callArgs[*argIndex];
    }
    Value callable = emitFunctionObject(anchor, returned);
    if (!callable.value)
      return Value{{}, method.resultType};
    return callable;
  }

  if (!method.mayThrow && method.returnedClassArgIndex) {
    std::size_t index = *method.returnedClassArgIndex;
    if (index >= callArgs.size()) {
      error(anchor, "returned class argument summary for method '" +
                        method.name + "' has invalid argument index");
      return Value{{}, method.resultType};
    }
    Value returned =
        coerceToExpectedType(anchor, callArgs[index], method.resultType);
    if (!returned.value)
      return Value{{}, method.resultType};
    return returned;
  }

  if (!method.mayThrow && method.returnedValueArgIndex) {
    std::size_t index = *method.returnedValueArgIndex;
    if (index >= callArgs.size()) {
      error(anchor, "returned value argument summary for method '" +
                        method.name + "' has invalid argument index");
      return Value{{}, method.resultType};
    }
    Value returned =
        coerceToExpectedType(anchor, callArgs[index], method.resultType);
    if (!returned.value)
      return Value{{}, method.resultType};
    if (!typeAssignable(method.resultType, returned.type)) {
      error(anchor, "returned value argument for method '" + method.name +
                        "' must be " + typeString(method.resultType) +
                        ", got " + typeString(returned.type));
      return Value{{}, method.resultType};
    }
    return returned;
  }

  if (!method.mayThrow && method.returnedCallableSymbolName) {
    std::optional<FunctionInfo> returned =
        findCallableInfoBySymbol(*method.returnedCallableSymbolName);
    if (returned) {
      Value callable = emitFunctionObject(anchor, *returned);
      if (!callable.value)
        return Value{{}, method.resultType};
      return callable;
    }
  }

  if (method.isAsync) {
    mlir::Type resultType = coroutineType(method.resultType);
    mlir::Value callee = builder.create<py::CallableObjectOp>(
        loc(anchor), method.functionType, method.symbolName);
    CallArgumentTuples tuples = emitCallArgumentTuples(method, callArgs);
    auto call = builder.create<py::CallOp>(
        loc(anchor), mlir::TypeRange{resultType}, callee, tuples.posargs.value,
        tuples.kwnames.value, tuples.kwvalues.value);
    return Value{call.getResults().front(), resultType};
  }

  mlir::Value callee = builder.create<py::CallableObjectOp>(
      loc(anchor), method.functionType, method.symbolName);
  CallArgumentTuples tuples = emitCallArgumentTuples(method, callArgs);
  if (method.mayThrow)
    return emitMayThrowFunctionCall(anchor, method, callee, tuples.posargs,
                                    tuples.kwnames, tuples.kwvalues);
  auto call = builder.create<py::CallOp>(
      loc(anchor), mlir::TypeRange{method.resultType}, callee,
      tuples.posargs.value, tuples.kwnames.value, tuples.kwvalues.value);
  return applyReturnedClassSummary(
      Value{call.getResults().front(), method.resultType}, method);
}

std::optional<std::vector<Value>>
Builder::Impl::prepareResolvedMethodCallArguments(
    const parser::Node &anchor, const Value &receiver,
    const FunctionInfo &method, llvm::ArrayRef<Value> userArgs) {
  Value actualReceiver = receiver;
  if (!method.argTypes.empty()) {
    if (auto selfType =
            mlir::dyn_cast<py::ClassType>(method.argTypes.front())) {
      if (actualReceiver.type != method.argTypes.front()) {
        actualReceiver = viewClassAs(anchor, std::move(actualReceiver),
                                     selfType.getClassName());
        if (!actualReceiver.value)
          return std::nullopt;
      }
    }
  }

  std::vector<Value> callArgs;
  callArgs.reserve(userArgs.size() + 1);
  callArgs.push_back(actualReceiver);
  callArgs.insert(callArgs.end(), userArgs.begin(), userArgs.end());
  if (callArgs.size() != method.argTypes.size()) {
    const std::size_t expectedUserArgs =
        method.argTypes.empty() ? 0 : method.argTypes.size() - 1;
    error(anchor, "method '" + method.name + "' expects " +
                      std::to_string(expectedUserArgs) + " arguments, got " +
                      std::to_string(userArgs.size()));
    return std::nullopt;
  }
  for (std::size_t index = 0; index < callArgs.size(); ++index) {
    if (!callArgs[index].value)
      return std::nullopt;
    if (!typeAssignable(method.argTypes[index], callArgs[index].type)) {
      error(anchor, "method '" + method.name + "' argument " +
                        std::to_string(index) + " type mismatch: expected " +
                        typeString(method.argTypes[index]) + ", got " +
                        typeString(callArgs[index].type));
      return std::nullopt;
    }
    Value coerced =
        coerceToExpectedType(anchor, callArgs[index], method.argTypes[index]);
    if (!coerced.value)
      return std::nullopt;
    callArgs[index] = coerced;
  }
  return callArgs;
}

Value Builder::Impl::emitMethodCall(const parser::Node &expr,
                                    const parser::Node &func,
                                    const std::vector<parser::NodePtr> &args) {
  const parser::NodePtr *receiverNode = nodeField(func, "value");
  const std::string *methodName = stringField(func, "attr");
  if (!receiverNode || !*receiverNode || !methodName) {
    error(expr, "method call receiver or method name is missing");
    return Value{{}, noneType()};
  }

  Value receiver = emitExpression(**receiverNode);
  if (!receiver.value)
    return Value{{}, noneType()};
  if (std::optional<Value> concrete = concreteProtocolValue(receiver))
    receiver = *concrete;

  const std::vector<parser::NodePtr> *keywords =
      nodeListField(expr, "keywords");
  auto protocolMethodArgumentTypes = [&](llvm::StringRef contextLabel)
      -> std::optional<llvm::SmallVector<mlir::Type>> {
    if (keywords && !keywords->empty()) {
      error(expr, contextLabel.str() + "s do not accept keyword arguments yet");
      return std::nullopt;
    }
    llvm::SmallVector<mlir::Type> argumentTypes;
    argumentTypes.reserve(args.size());
    for (const parser::NodePtr &arg : args) {
      if (!arg) {
        error(expr, contextLabel.str() + " argument is missing");
        return std::nullopt;
      }
      std::optional<mlir::Type> argType = inferExpressionType(*arg);
      if (!argType) {
        error(*arg, contextLabel.str() +
                        " requires statically known argument types before "
                        "lowering");
        return std::nullopt;
      }
      argumentTypes.push_back(*argType);
    }
    return argumentTypes;
  };

  auto requireNoKeywords = [&](llvm::StringRef contextLabel) -> bool {
    if (keywords && !keywords->empty()) {
      error(expr, contextLabel.str() + " does not accept keyword arguments");
      return false;
    }
    return true;
  };

  auto resolveInstanceSpecialMethod =
      [&](llvm::StringRef specialName) -> std::optional<FunctionInfo> {
    std::optional<FunctionInfo> method =
        resolveClassMethod(expr, receiver, specialName);
    if (!method)
      return std::nullopt;
    if (!method->methodKind.empty() && method->methodKind != "instance") {
      error(expr, "special method '" + specialName.str() +
                      "' must be an instance method");
      return std::nullopt;
    }
    return method;
  };

  if (*methodName == "__enter__") {
    if (!requireNoKeywords("__enter__"))
      return Value{{}, noneType()};
    std::optional<FunctionInfo> method =
        resolveInstanceSpecialMethod("__enter__");
    if (!method)
      return Value{{}, noneType()};
    if (method->isAsync || method->mayThrow) {
      error(expr, "__enter__ dialect lowering requires a synchronous nothrow "
                  "method");
      return Value{{}, method->resultType};
    }
    std::optional<std::vector<Value>> userArgs =
        emitStaticArguments(expr, *method, args, /*firstFormal=*/1);
    if (!userArgs)
      return Value{{}, method->resultType};
    std::optional<std::vector<Value>> prepared =
        prepareResolvedMethodCallArguments(expr, receiver, *method, *userArgs);
    if (!prepared)
      return Value{{}, method->resultType};
    std::optional<py::ProtocolType> contextManager =
        protocolType("ContextManager", {method->resultType});
    if (!contextManager) {
      error(expr, "failed to instantiate ContextManager protocol");
      return Value{{}, method->resultType};
    }
    std::optional<mlir::Type> expected = resolveProtocolMethodResult(
        expr, *contextManager, "__enter__", {}, "__enter__ method call");
    if (!expected)
      return Value{{}, method->resultType};
    if (!typeAssignable(*expected, method->resultType)) {
      error(expr, "ContextManager.__enter__ contract for " +
                      typeString(*contextManager) +
                      " does not match method result " +
                      typeString(method->resultType));
      return Value{{}, method->resultType};
    }
    if (auto managerClass = mlir::dyn_cast<py::ClassType>(receiver.type)) {
      if (!classConformsToProtocol(managerClass, *contextManager)) {
        error(expr, "receiver " + typeString(receiver.type) +
                        " does not satisfy " + typeString(*contextManager));
        return Value{{}, method->resultType};
      }
    }
    auto enterOp = builder.create<py::EnterOp>(
        loc(expr), method->resultType, method->symbolName, method->functionType,
        (*prepared)[0].value, mlir::UnitAttr{});
    return applyReturnedClassSummary(
        Value{enterOp.getResult(), method->resultType}, *method);
  }

  if (*methodName == "__exit__") {
    if (!requireNoKeywords("__exit__"))
      return Value{{}, noneType()};
    std::optional<FunctionInfo> method =
        resolveInstanceSpecialMethod("__exit__");
    if (!method)
      return Value{{}, noneType()};
    if (method->isAsync || method->mayThrow) {
      error(expr, "__exit__ dialect lowering requires a synchronous nothrow "
                  "method");
      return Value{{}, method->resultType};
    }
    std::optional<std::vector<Value>> userArgs =
        emitStaticArguments(expr, *method, args, /*firstFormal=*/1);
    if (!userArgs)
      return Value{{}, method->resultType};
    std::optional<std::vector<Value>> prepared =
        prepareResolvedMethodCallArguments(expr, receiver, *method, *userArgs);
    if (!prepared)
      return Value{{}, method->resultType};
    llvm::SmallVector<mlir::Type, 3> exitArgTypes;
    for (const Value &arg : *userArgs)
      exitArgTypes.push_back(arg.type);
    std::optional<py::ProtocolType> contextManager = protocolType(
        "ContextManager", {py::ObjectType::get(&context), method->resultType});
    if (!contextManager) {
      error(expr, "failed to instantiate ContextManager protocol");
      return Value{{}, method->resultType};
    }
    std::optional<mlir::Type> expected =
        resolveProtocolMethodResult(expr, *contextManager, "__exit__",
                                    exitArgTypes, "__exit__ method call");
    if (!expected)
      return Value{{}, method->resultType};
    if (!typeAssignable(*expected, method->resultType)) {
      error(expr, "ContextManager.__exit__ contract for " +
                      typeString(*contextManager) +
                      " does not match method result " +
                      typeString(method->resultType));
      return Value{{}, method->resultType};
    }
    if (auto managerClass = mlir::dyn_cast<py::ClassType>(receiver.type)) {
      if (!classConformsToProtocol(managerClass, *contextManager)) {
        error(expr, "receiver " + typeString(receiver.type) +
                        " does not satisfy " + typeString(*contextManager));
        return Value{{}, method->resultType};
      }
    }
    auto exitOp = builder.create<py::ExitOp>(
        loc(expr), method->resultType, method->symbolName, method->functionType,
        (*prepared)[0].value, (*prepared)[1].value, (*prepared)[2].value,
        (*prepared)[3].value, mlir::UnitAttr{});
    return applyReturnedClassSummary(
        Value{exitOp.getResult(), method->resultType}, *method);
  }

  if (*methodName == "__aenter__") {
    if (!requireNoKeywords("__aenter__"))
      return Value{{}, noneType()};
    std::optional<FunctionInfo> method =
        resolveInstanceSpecialMethod("__aenter__");
    if (!method)
      return Value{{}, noneType()};
    if (method->mayThrow) {
      error(expr, "__aenter__ dialect lowering requires a nothrow method");
      return Value{{}, methodAwaitableType(*method)};
    }
    std::optional<std::vector<Value>> userArgs =
        emitStaticArguments(expr, *method, args, /*firstFormal=*/1);
    if (!userArgs)
      return Value{{}, methodAwaitableType(*method)};
    std::optional<std::vector<Value>> prepared =
        prepareResolvedMethodCallArguments(expr, receiver, *method, *userArgs);
    if (!prepared)
      return Value{{}, methodAwaitableType(*method)};
    mlir::Type awaitableType = methodAwaitableType(*method);
    mlir::Type payloadType = awaitablePayloadType(awaitableType);
    if (!payloadType) {
      error(expr, "__aenter__ must return an awaitable value, got " +
                      typeString(awaitableType));
      return Value{{}, awaitableType};
    }
    std::optional<py::ProtocolType> asyncContextManager =
        protocolType("AsyncContextManager", {payloadType});
    if (!asyncContextManager) {
      error(expr, "failed to instantiate AsyncContextManager protocol");
      return Value{{}, awaitableType};
    }
    std::optional<mlir::Type> expected = resolveProtocolMethodResult(
        expr, *asyncContextManager, "__aenter__", {}, "__aenter__ method call");
    if (!expected)
      return Value{{}, awaitableType};
    if (!typeAssignable(*expected, awaitableType)) {
      error(expr, "AsyncContextManager.__aenter__ contract for " +
                      typeString(*asyncContextManager) +
                      " does not match method result " +
                      typeString(awaitableType));
      return Value{{}, awaitableType};
    }
    if (auto managerClass = mlir::dyn_cast<py::ClassType>(receiver.type)) {
      if (!classConformsToProtocol(managerClass, *asyncContextManager)) {
        error(expr, "receiver " + typeString(receiver.type) +
                        " does not satisfy " +
                        typeString(*asyncContextManager));
        return Value{{}, awaitableType};
      }
    }
    mlir::UnitAttr asyncMethod =
        method->isAsync ? builder.getUnitAttr() : mlir::UnitAttr{};
    auto enterOp = builder.create<py::AEnterOp>(
        loc(expr), awaitableType, method->symbolName, method->functionType,
        (*prepared)[0].value, asyncMethod);
    return Value{enterOp.getResult(), awaitableType};
  }

  if (*methodName == "__aexit__") {
    if (!requireNoKeywords("__aexit__"))
      return Value{{}, noneType()};
    std::optional<FunctionInfo> method =
        resolveInstanceSpecialMethod("__aexit__");
    if (!method)
      return Value{{}, noneType()};
    if (method->mayThrow) {
      error(expr, "__aexit__ dialect lowering requires a nothrow method");
      return Value{{}, methodAwaitableType(*method)};
    }
    std::optional<std::vector<Value>> userArgs =
        emitStaticArguments(expr, *method, args, /*firstFormal=*/1);
    if (!userArgs)
      return Value{{}, methodAwaitableType(*method)};
    std::optional<std::vector<Value>> prepared =
        prepareResolvedMethodCallArguments(expr, receiver, *method, *userArgs);
    if (!prepared)
      return Value{{}, methodAwaitableType(*method)};
    mlir::Type awaitableType = methodAwaitableType(*method);
    mlir::Type payloadType = awaitablePayloadType(awaitableType);
    if (!payloadType) {
      error(expr, "__aexit__ must return an awaitable value, got " +
                      typeString(awaitableType));
      return Value{{}, awaitableType};
    }
    llvm::SmallVector<mlir::Type, 3> exitArgTypes;
    for (const Value &arg : *userArgs)
      exitArgTypes.push_back(arg.type);
    std::optional<py::ProtocolType> asyncContextManager = protocolType(
        "AsyncContextManager", {py::ObjectType::get(&context), payloadType});
    if (!asyncContextManager) {
      error(expr, "failed to instantiate AsyncContextManager protocol");
      return Value{{}, awaitableType};
    }
    std::optional<mlir::Type> expected =
        resolveProtocolMethodResult(expr, *asyncContextManager, "__aexit__",
                                    exitArgTypes, "__aexit__ method call");
    if (!expected)
      return Value{{}, awaitableType};
    if (!typeAssignable(*expected, awaitableType)) {
      error(expr, "AsyncContextManager.__aexit__ contract for " +
                      typeString(*asyncContextManager) +
                      " does not match method result " +
                      typeString(awaitableType));
      return Value{{}, awaitableType};
    }
    if (auto managerClass = mlir::dyn_cast<py::ClassType>(receiver.type)) {
      if (!classConformsToProtocol(managerClass, *asyncContextManager)) {
        error(expr, "receiver " + typeString(receiver.type) +
                        " does not satisfy " +
                        typeString(*asyncContextManager));
        return Value{{}, awaitableType};
      }
    }
    mlir::UnitAttr asyncMethod =
        method->isAsync ? builder.getUnitAttr() : mlir::UnitAttr{};
    auto exitOp = builder.create<py::AExitOp>(
        loc(expr), awaitableType, method->symbolName, method->functionType,
        (*prepared)[0].value, (*prepared)[1].value, (*prepared)[2].value,
        (*prepared)[3].value, asyncMethod);
    return Value{exitOp.getResult(), awaitableType};
  }

  if (*methodName == "__iter__") {
    if (!requireNoKeywords("__iter__"))
      return Value{{}, noneType()};
    if (!args.empty()) {
      error(expr, "__iter__ expects no arguments");
      return Value{{}, noneType()};
    }
    std::optional<SyncIteratorResolution> resolution =
        resolveSyncIterator(expr, receiver, /*requireConcreteIterator=*/false);
    if (!resolution)
      return Value{{}, noneType()};
    std::optional<std::vector<Value>> prepared =
        prepareResolvedMethodCallArguments(expr, resolution->iterable,
                                           resolution->iterMethod, {});
    if (!prepared)
      return Value{{}, resolution->iteratorType};
    bool returnedSelf =
        resolution->iterMethod.returnedValueArgIndex &&
        *resolution->iterMethod.returnedValueArgIndex == 0 &&
        typeAssignable(resolution->iteratorType, resolution->iterable.type);
    mlir::Type resultType =
        returnedSelf ? resolution->iterable.type : resolution->iteratorType;
    mlir::UnitAttr returnedSelfAttr =
        returnedSelf ? builder.getUnitAttr() : mlir::UnitAttr{};
    auto iterOp = builder.create<py::IterOp>(
        loc(expr), resultType, resolution->iterMethod.symbolName,
        resolution->iterMethod.functionType, (*prepared)[0].value,
        returnedSelfAttr);
    if (returnedSelf)
      return Value{iterOp.getResult(), resultType,
                   resolution->iterable.exactClass,
                   resolution->iterable.provenClass};
    return Value{iterOp.getResult(), resultType};
  }
  if (*methodName == "__next__") {
    if (!requireNoKeywords("__next__"))
      return Value{{}, noneType()};
    if (!args.empty()) {
      error(expr, "__next__ expects no arguments");
      return Value{{}, noneType()};
    }
    return emitNextValue(expr, receiver, "__next__ method call");
  }
  if (*methodName == "__len__" && !classNameFromType(receiver.type)) {
    if (keywords && !keywords->empty()) {
      error(expr, "__len__ does not accept keyword arguments");
      return Value{{}, intType()};
    }
    if (!args.empty()) {
      error(expr, "__len__ expects no arguments");
      return Value{{}, intType()};
    }
    std::optional<protocols::ProtocolMethod> lenContract =
        resolveProtocolMethodContract(expr, receiver.type, "__len__", {},
                                      "__len__ method call");
    if (!lenContract)
      return Value{{}, intType()};
    llvm::ArrayRef<mlir::Type> lenResults =
        lenContract->signature.getResultTypes();
    if (lenResults.size() != 1 || lenResults.front() != intType()) {
      error(expr, "__len__ contract on " + typeString(receiver.type) +
                      " must return !py.int");
      return Value{{}, intType()};
    }
    if (mlir::isa<py::ProtocolType>(receiver.type)) {
      error(expr, "protocol __len__ on " + typeString(receiver.type) +
                      " resolves statically to !py.int, but lowering for "
                      "protocol-typed receivers is not implemented yet");
      return Value{{}, intType()};
    }
    mlir::Value result = builder.create<py::LenOp>(
        loc(expr), intType(), lenContract->signature, receiver.value);
    return Value{result, intType()};
  }
  if (*methodName == "__getitem__" && !classNameFromType(receiver.type)) {
    if (keywords && !keywords->empty()) {
      error(expr, "__getitem__ does not accept keyword arguments yet");
      return Value{{}, noneType()};
    }
    if (args.size() != 1 || !args.front()) {
      error(expr, "__getitem__ expects exactly one argument");
      return Value{{}, noneType()};
    }
    return emitGetItemAccess(expr, receiver, *args.front(),
                             "__getitem__ method call");
  }
  if (*methodName == "__contains__" && !classNameFromType(receiver.type)) {
    if (keywords && !keywords->empty()) {
      error(expr, "__contains__ does not accept keyword arguments yet");
      return Value{{}, boolType()};
    }
    if (args.size() != 1 || !args.front()) {
      error(expr, "__contains__ expects exactly one argument");
      return Value{{}, boolType()};
    }
    Value item = emitExpression(*args.front());
    if (!item.value)
      return Value{{}, boolType()};
    std::optional<protocols::ProtocolMethod> containsContract =
        resolveProtocolMethodContract(expr, receiver.type, "__contains__",
                                      {item.type}, "__contains__ method call");
    if (!containsContract)
      return Value{{}, boolType()};
    llvm::ArrayRef<mlir::Type> results =
        containsContract->signature.getResultTypes();
    if (results.size() != 1 || results.front() != boolType()) {
      error(expr, "__contains__ contract on " + typeString(receiver.type) +
                      " must return !py.bool");
      return Value{{}, boolType()};
    }
    if (mlir::isa<py::ProtocolType>(receiver.type)) {
      error(expr, "protocol __contains__ on " + typeString(receiver.type) +
                      " resolves statically to !py.bool, but lowering for "
                      "protocol-typed receivers is not implemented yet");
      return Value{{}, boolType()};
    }
    if (std::optional<std::pair<mlir::Type, mlir::Type>> dictTypes =
            dictKeyValueTypes(receiver.type)) {
      if (item.type != dictTypes->first) {
        if (exactScalarEqualityAlwaysFalse(item.type, dictTypes->first)) {
          mlir::Value falseBit =
              builder.create<mlir::arith::ConstantIntOp>(loc(expr), 0, 1);
          mlir::Value result = builder.create<py::CastFromPrimOp>(
              loc(expr), boolType(), falseBit);
          return Value{result, boolType()};
        }
        error(expr, "__contains__ on " + typeString(receiver.type) +
                        " resolves statically to !py.bool, but dictionary "
                        "lookup lowering for item type " +
                        typeString(item.type) + " and key type " +
                        typeString(dictTypes->first) +
                        " is not implemented yet");
        return Value{{}, boolType()};
      }
    }
    mlir::Value bit = builder.create<py::ContainsOp>(
        loc(expr), i1Type(), containsContract->signature, receiver.value,
        item.value);
    mlir::Value result =
        builder.create<py::CastFromPrimOp>(loc(expr), boolType(), bit);
    return Value{result, boolType()};
  }

  auto findReceiverClassMethod =
      [&](llvm::StringRef name) -> std::optional<FunctionInfo> {
    std::optional<std::string> staticClassName =
        classNameFromType(receiver.type);
    if (!staticClassName)
      return std::nullopt;
    std::string className =
        receiver.exactClass
            ? *receiver.exactClass
            : (receiver.provenClass ? *receiver.provenClass : *staticClassName);
    auto classFound = classes.find(className);
    if (classFound == classes.end())
      return std::nullopt;
    auto methodFound = classFound->second.methods.find(name.str());
    if (methodFound == classFound->second.methods.end())
      return std::nullopt;
    auto functionFound = functions.find(methodFound->second);
    if (functionFound == functions.end())
      return std::nullopt;
    if (!functionFound->second.methodKind.empty() &&
        functionFound->second.methodKind != "instance")
      return std::nullopt;
    return functionFound->second;
  };

  auto inferredSyncGeneratorArgs =
      [&](const FunctionInfo &method,
          llvm::ArrayRef<Value> userArgs) -> llvm::SmallVector<mlir::Type, 3> {
    mlir::Type yieldType = method.resultType;
    if (*methodName == "close") {
      if (std::optional<FunctionInfo> next =
              findReceiverClassMethod("__next__"))
        yieldType = next->resultType;
      else if (std::optional<FunctionInfo> send =
                   findReceiverClassMethod("send"))
        yieldType = send->resultType;
      else
        yieldType = py::ObjectType::get(&context);
    }
    mlir::Type sendType = py::ObjectType::get(&context);
    if (*methodName == "send" && !userArgs.empty()) {
      sendType = userArgs.front().type;
    } else if (std::optional<FunctionInfo> send =
                   findReceiverClassMethod("send")) {
      if (send->argTypes.size() >= 2)
        sendType = send->argTypes[1];
    }
    return {yieldType, sendType, py::ObjectType::get(&context)};
  };

  auto inferredAsyncGeneratorArgs =
      [&](const FunctionInfo &method, llvm::ArrayRef<Value> userArgs,
          mlir::Type awaitableType) -> llvm::SmallVector<mlir::Type, 2> {
    mlir::Type yieldType = awaitablePayloadType(awaitableType);
    if (!yieldType || *methodName == "aclose") {
      if (std::optional<FunctionInfo> asend =
              findReceiverClassMethod("asend")) {
        mlir::Type asendAwaitable = methodAwaitableType(*asend);
        yieldType = awaitablePayloadType(asendAwaitable);
      }
      if (!yieldType)
        yieldType = py::ObjectType::get(&context);
    }
    mlir::Type sendType = py::ObjectType::get(&context);
    if (*methodName == "asend" && !userArgs.empty()) {
      sendType = userArgs.front().type;
    } else if (std::optional<FunctionInfo> asend =
                   findReceiverClassMethod("asend")) {
      if (asend->argTypes.size() >= 2)
        sendType = asend->argTypes[1];
    }
    return {yieldType, sendType};
  };

  auto emitResolvedGeneratorFamilyOp = [&]() -> std::optional<Value> {
    bool syncMethod = *methodName == "send" || *methodName == "throw" ||
                      *methodName == "close";
    bool asyncMethod = *methodName == "asend" || *methodName == "athrow" ||
                       *methodName == "aclose";
    if (!syncMethod && !asyncMethod)
      return std::nullopt;
    if (!requireNoKeywords(*methodName))
      return Value{{}, noneType()};
    if (!classNameFromType(receiver.type))
      return std::nullopt;
    std::optional<FunctionInfo> method =
        resolveClassMethod(expr, receiver, *methodName);
    if (!method)
      return Value{{}, noneType()};
    if (!method->methodKind.empty() && method->methodKind != "instance") {
      error(expr, "special protocol method '" + *methodName +
                      "' must be an instance method");
      return Value{{}, method->resultType};
    }
    if (method->mayThrow)
      return std::nullopt;

    std::optional<std::vector<Value>> userArgs =
        emitStaticArguments(expr, *method, args, /*firstFormal=*/1);
    if (!userArgs)
      return Value{
          {}, syncMethod ? method->resultType : methodAwaitableType(*method)};
    std::optional<std::vector<Value>> prepared =
        prepareResolvedMethodCallArguments(expr, receiver, *method, *userArgs);
    if (!prepared)
      return Value{
          {}, syncMethod ? method->resultType : methodAwaitableType(*method)};

    llvm::SmallVector<mlir::Type> argumentTypes;
    argumentTypes.reserve(userArgs->size());
    for (const Value &arg : *userArgs)
      argumentTypes.push_back(arg.type);

    if (syncMethod) {
      llvm::SmallVector<mlir::Type, 3> protocolArgs =
          inferredSyncGeneratorArgs(*method, *userArgs);
      std::optional<py::ProtocolType> generator =
          protocolType("Generator", protocolArgs);
      if (!generator) {
        error(expr, "failed to instantiate Generator protocol");
        return Value{{}, method->resultType};
      }
      std::optional<mlir::Type> expected = resolveProtocolMethodResult(
          expr, *generator, *methodName, argumentTypes,
          *methodName + " method call");
      if (!expected)
        return Value{{}, method->resultType};
      if (!typeAssignable(*expected, method->resultType)) {
        error(expr, "Generator." + *methodName + " contract for " +
                        typeString(*generator) +
                        " does not match method result " +
                        typeString(method->resultType));
        return Value{{}, method->resultType};
      }
      if (*methodName == "send") {
        auto op = builder.create<py::SendOp>(
            loc(expr), method->resultType, method->symbolName,
            method->functionType, (*prepared)[0].value, (*prepared)[1].value);
        return applyReturnedClassSummary(
            Value{op.getResult(), method->resultType}, *method);
      }
      if (*methodName == "throw") {
        std::vector<Value> throwValues;
        for (std::size_t index = 1; index < prepared->size(); ++index)
          throwValues.push_back((*prepared)[index]);
        Value throwArgs = emitTuple(throwValues);
        auto op = builder.create<py::ThrowOp>(
            loc(expr), method->resultType, method->symbolName,
            method->functionType, (*prepared)[0].value, throwArgs.value);
        return applyReturnedClassSummary(
            Value{op.getResult(), method->resultType}, *method);
      }
      auto op = builder.create<py::CloseOp>(
          loc(expr), method->resultType, method->symbolName,
          method->functionType, (*prepared)[0].value);
      return applyReturnedClassSummary(
          Value{op.getResult(), method->resultType}, *method);
    }

    mlir::Type awaitableType = methodAwaitableType(*method);
    llvm::SmallVector<mlir::Type, 2> protocolArgs =
        inferredAsyncGeneratorArgs(*method, *userArgs, awaitableType);
    std::optional<py::ProtocolType> asyncGenerator =
        protocolType("AsyncGenerator", protocolArgs);
    if (!asyncGenerator) {
      error(expr, "failed to instantiate AsyncGenerator protocol");
      return Value{{}, awaitableType};
    }
    std::optional<mlir::Type> expected = resolveProtocolMethodResult(
        expr, *asyncGenerator, *methodName, argumentTypes,
        *methodName + " method call");
    if (!expected)
      return Value{{}, awaitableType};
    if (!typeAssignable(*expected, awaitableType)) {
      error(expr, "AsyncGenerator." + *methodName + " contract for " +
                      typeString(*asyncGenerator) +
                      " does not match method result " +
                      typeString(awaitableType));
      return Value{{}, awaitableType};
    }
    mlir::UnitAttr asyncAttr =
        method->isAsync ? builder.getUnitAttr() : mlir::UnitAttr{};
    if (*methodName == "asend") {
      auto op = builder.create<py::ASendOp>(
          loc(expr), awaitableType, method->symbolName, method->functionType,
          (*prepared)[0].value, (*prepared)[1].value, asyncAttr);
      return Value{op.getResult(), awaitableType};
    }
    if (*methodName == "athrow") {
      std::vector<Value> throwValues;
      for (std::size_t index = 1; index < prepared->size(); ++index)
        throwValues.push_back((*prepared)[index]);
      Value throwArgs = emitTuple(throwValues);
      auto op = builder.create<py::AThrowOp>(
          loc(expr), awaitableType, method->symbolName, method->functionType,
          (*prepared)[0].value, throwArgs.value, asyncAttr);
      return Value{op.getResult(), awaitableType};
    }
    auto op = builder.create<py::ACloseOp>(
        loc(expr), awaitableType, method->symbolName, method->functionType,
        (*prepared)[0].value, asyncAttr);
    return Value{op.getResult(), awaitableType};
  };

  if (std::optional<Value> special = emitResolvedGeneratorFamilyOp())
    return *special;

  if (listElementType(receiver.type)) {
    if (keywords && !keywords->empty()) {
      error(expr, "typed list methods do not accept keyword arguments");
      return Value{{}, noneType()};
    }
    return emitListMethodCall(expr, receiver, *methodName, args);
  }
  if (mlir::isa<py::TupleType>(receiver.type) && *methodName == "count") {
    if (keywords && !keywords->empty()) {
      error(expr, "typed tuple.count does not accept keyword arguments yet");
      return Value{{}, intType()};
    }
    return emitTupleCountMethodCall(expr, receiver, args);
  }
  if (dictKeyValueTypes(receiver.type) && *methodName == "get") {
    if (keywords && !keywords->empty()) {
      error(expr, "typed dict.get does not accept keyword arguments yet");
      return Value{{}, noneType()};
    }
    return emitDictGetMethodCall(expr, receiver, args);
  }
  if (!mlir::isa<py::ProtocolType>(receiver.type) &&
      !classNameFromType(receiver.type)) {
    const protocols::Table &table = protocols::Table::get(context);
    if (!table.methodOverloadsOn(receiver.type, *methodName).empty()) {
      std::optional<llvm::SmallVector<mlir::Type>> argumentTypes =
          protocolMethodArgumentTypes("primitive protocol method call");
      if (!argumentTypes)
        return Value{{}, noneType()};
      std::optional<mlir::Type> resultType = resolveProtocolMethodResult(
          expr, receiver.type, *methodName, *argumentTypes,
          "primitive protocol method call");
      if (!resultType)
        return Value{{}, noneType()};
      error(expr,
            "primitive protocol method '" + *methodName + "' on " +
                typeString(receiver.type) + " resolves statically to " +
                typeString(*resultType) +
                ", but this dialect primitive has no method lowering yet");
      return Value{{}, *resultType};
    }
  }
  if (mlir::isa<py::ProtocolType>(receiver.type)) {
    std::optional<llvm::SmallVector<mlir::Type>> argumentTypes =
        protocolMethodArgumentTypes("protocol method call");
    if (!argumentTypes)
      return Value{{}, noneType()};
    std::optional<mlir::Type> resultType =
        resolveProtocolMethodResult(expr, receiver.type, *methodName,
                                    *argumentTypes, "protocol method call");
    if (!resultType)
      return Value{{}, noneType()};
    error(expr, "protocol method '" + *methodName + "' on " +
                    typeString(receiver.type) + " resolves statically to " +
                    typeString(*resultType) +
                    ", but lowering for protocol-typed receivers is not "
                    "implemented yet");
    return Value{{}, *resultType};
  }

  std::optional<FunctionInfo> method =
      resolveClassMethod(expr, receiver, *methodName);
  if (!method)
    return Value{{}, noneType()};
  if (!method->methodKind.empty() && method->methodKind != "instance") {
    error(expr, "only instance method calls are implemented for class method "
                "syntax; method '" +
                    method->name + "' is " + method->methodKind);
    return Value{{}, method->resultType};
  }
  if (hasCallableMetadata(*method)) {
    if (!method->argTypes.empty()) {
      if (auto selfType =
              mlir::dyn_cast<py::ClassType>(method->argTypes.front())) {
        if (receiver.type != method->argTypes.front()) {
          receiver =
              viewClassAs(expr, std::move(receiver), selfType.getClassName());
          if (!receiver.value)
            return Value{{}, method->resultType};
        }
      }
    }
    std::optional<CallArgumentTuples> tuples =
        emitExplicitCallArgumentTuples(expr, *method, args, /*firstFormal=*/1,
                                       llvm::ArrayRef<Value>{receiver});
    if (!tuples)
      return Value{{}, method->resultType};
    Value callee = emitFunctionObject(expr, *method);
    if (!callee.value)
      return Value{{}, method->resultType};
    if (method->mayThrow)
      return emitMayThrowFunctionCall(expr, *method, callee.value,
                                      tuples->posargs, tuples->kwnames,
                                      tuples->kwvalues);
    auto call = builder.create<py::CallOp>(
        loc(expr), mlir::TypeRange{method->resultType}, callee.value,
        tuples->posargs.value, tuples->kwnames.value, tuples->kwvalues.value);
    return applyReturnedClassSummary(
        Value{call.getResults().front(), method->resultType}, *method);
  }
  std::optional<std::vector<Value>> userArgs =
      emitStaticArguments(expr, *method, args, /*firstFormal=*/1);
  if (!userArgs)
    return Value{{}, method->resultType};
  return emitResolvedMethodCall(expr, receiver, *method, *userArgs);
}

} // namespace lython::emitter
