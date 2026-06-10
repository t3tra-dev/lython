#include "BuilderImpl.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include <functional>

namespace lython::emitter {
namespace {

bool isTupleLiteralElementSupported(mlir::Type type) {
  return mlir::isa<mlir::IntegerType, mlir::FloatType, py::IntType,
                   py::BoolType, py::FloatType, py::NoneType, py::StrType,
                   py::ExceptionType, py::ClassType>(type);
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
  std::optional<std::string> className = classNameFromType(object.type);
  if (!className) {
    error(expr, "attribute access requires a statically known class receiver");
    return Value{{}, noneType()};
  }
  auto classFound = classes.find(*className);
  if (classFound == classes.end()) {
    error(expr, "unknown class '" + *className + "'");
    return Value{{}, noneType()};
  }
  auto fieldFound = classFound->second.fields.find(*name);
  if (fieldFound == classFound->second.fields.end()) {
    error(expr, "class '" + *className + "' has no field '" + *name + "'");
    return Value{{}, noneType()};
  }
  mlir::Value result = builder.create<py::AttrGetOp>(
      loc(expr), fieldFound->second, object.value, *name);
  return Value{result, fieldFound->second};
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
      mlir::Value component = builder.create<py::TupleGetOp>(
          loc(anchor), elementType, sequence.value, indexValue);
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
        mlir::Value result = builder.create<py::TupleGetOp>(
            loc(expr), elementType, container.value, indexValue);
        elements.push_back(Value{result, elementType});
      }
      return emitTuple(elements);
    }

    std::optional<std::int64_t> index = staticIndexValue(**indexNode);
    if (!index || *index < 0) {
      error(**indexNode,
            "tuple subscript requires a non-negative static integer index");
      return Value{{}, noneType()};
    }
    if (elementTypes.empty()) {
      error(expr, "cannot subscript an empty tuple");
      return Value{{}, noneType()};
    }
    mlir::Type elementType = elementTypes.front();
    if (elementTypes.size() > 1) {
      if (*index >= static_cast<std::int64_t>(elementTypes.size())) {
        error(**indexNode, "tuple subscript index is out of range");
        return Value{{}, noneType()};
      }
      elementType = elementTypes[*index];
    }
    mlir::Value indexValue =
        builder.create<mlir::arith::ConstantIndexOp>(loc(**indexNode), *index);
    mlir::Value result = builder.create<py::TupleGetOp>(
        loc(expr), elementType, container.value, indexValue);
    return Value{result, elementType};
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

  mlir::Type elementType = listElementType(container.type);
  if (elementType) {
    mlir::Type indexType = builder.getI64Type();
    Value index = emitExpressionWithExpectedType(**indexNode, indexType);
    if (!index.value)
      return Value{{}, elementType};
    mlir::Value result = builder.create<py::ListGetOp>(
        loc(expr), elementType, container.value, index.value);
    return Value{result, elementType};
  }

  if ((*indexNode)->kind == "Slice") {
    error(**indexNode, "slice subscript currently supports only statically "
                       "finite typed tuples and lists");
    return Value{{}, noneType()};
  }

  if (std::optional<std::pair<mlir::Type, mlir::Type>> dictTypes =
          dictKeyValueTypes(container.type)) {
    Value index = emitExpressionWithExpectedType(**indexNode, dictTypes->first);
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
    mlir::Value result = builder.create<py::DictGetOp>(
        loc(expr), dictTypes->second, container.value, index.value);
    return Value{result, dictTypes->second};
  }

  error(expr, "subscript access supports only typed tuples, lists and dicts");
  return Value{{}, noneType()};
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

  Value value = emitExpression(*args.front());
  if (!value.value)
    return Value{{}, noneType()};
  if (value.type != elementType) {
    error(*args.front(),
          "list." + methodName.str() + " argument type mismatch: expected " +
              typeString(elementType) + ", got " + typeString(value.type));
    return Value{{}, noneType()};
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

std::optional<FunctionInfo>
Builder::Impl::resolveClassMethod(const parser::Node &anchor,
                                  const Value &receiver,
                                  llvm::StringRef methodName) {
  std::optional<std::string> className = classNameFromType(receiver.type);
  if (!className) {
    error(anchor, "method call requires a statically known receiver type");
    return std::nullopt;
  }
  auto classFound = classes.find(*className);
  if (classFound == classes.end()) {
    error(anchor, "unknown class '" + *className + "'");
    return std::nullopt;
  }
  auto methodFound = classFound->second.methods.find(methodName.str());
  if (methodFound == classFound->second.methods.end()) {
    error(anchor, "class '" + *className + "' has no method '" +
                      methodName.str() + "'");
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
  std::vector<Value> callArgs;
  callArgs.reserve(userArgs.size() + 1);
  callArgs.push_back(receiver);
  callArgs.insert(callArgs.end(), userArgs.begin(), userArgs.end());
  if (callArgs.size() != method.argTypes.size()) {
    const std::size_t expectedUserArgs =
        method.argTypes.empty() ? 0 : method.argTypes.size() - 1;
    error(anchor, "method '" + method.name + "' expects " +
                      std::to_string(expectedUserArgs) + " arguments, got " +
                      std::to_string(userArgs.size()));
    return Value{{}, method.resultType};
  }
  for (std::size_t index = 0; index < callArgs.size(); ++index) {
    if (!callArgs[index].value)
      return Value{{}, method.resultType};
    if (!typeAssignable(method.argTypes[index], callArgs[index].type)) {
      error(anchor, "method '" + method.name + "' argument " +
                        std::to_string(index) + " type mismatch: expected " +
                        typeString(method.argTypes[index]) + ", got " +
                        typeString(callArgs[index].type));
      return Value{{}, method.resultType};
    }
  }

  if (!method.mayThrow && method.returnedCallableArgIndex) {
    std::size_t index = *method.returnedCallableArgIndex;
    if (index >= callArgs.size()) {
      error(anchor, "returned callable argument summary for method '" +
                        method.name + "' has invalid argument index");
      return Value{{}, method.resultType};
    }
    Value returned = callArgs[index];
    if (!mlir::isa<py::FuncType>(returned.type)) {
      error(anchor, "returned callable argument for method '" + method.name +
                        "' must be !py.func, got " + typeString(returned.type));
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

  mlir::Value callee = builder.create<py::FuncObjectOp>(
      loc(anchor), method.functionType, method.symbolName);
  CallArgumentTuples tuples = emitCallArgumentTuples(method, callArgs);
  if (method.mayThrow)
    return emitMayThrowFunctionCall(anchor, method, callee, tuples.posargs,
                                    tuples.kwnames, tuples.kwvalues);
  auto call = builder.create<py::CallVectorOp>(
      loc(anchor), mlir::TypeRange{method.resultType}, callee,
      tuples.posargs.value, tuples.kwnames.value, tuples.kwvalues.value,
      mlir::UnitAttr{});
  return Value{call.getResults().front(), method.resultType};
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

  const std::vector<parser::NodePtr> *keywords =
      nodeListField(expr, "keywords");
  if (listElementType(receiver.type)) {
    if (keywords && !keywords->empty()) {
      error(expr, "typed list methods do not accept keyword arguments");
      return Value{{}, noneType()};
    }
    return emitListMethodCall(expr, receiver, *methodName, args);
  }

  std::optional<FunctionInfo> method =
      resolveClassMethod(expr, receiver, *methodName);
  if (!method)
    return Value{{}, noneType()};
  if (hasCallableMetadata(*method)) {
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
    auto call = builder.create<py::CallVectorOp>(
        loc(expr), mlir::TypeRange{method->resultType}, callee.value,
        tuples->posargs.value, tuples->kwnames.value, tuples->kwvalues.value,
        mlir::UnitAttr{});
    return Value{call.getResults().front(), method->resultType};
  }
  std::optional<std::vector<Value>> userArgs =
      emitStaticArguments(expr, *method, args, /*firstFormal=*/1);
  if (!userArgs)
    return Value{{}, method->resultType};
  return emitResolvedMethodCall(expr, receiver, *method, *userArgs);
}

} // namespace lython::emitter
