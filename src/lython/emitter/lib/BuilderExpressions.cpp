#include "BuilderImpl.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include <functional>

namespace lython::emitter {
namespace {

bool isStructuredIfExpType(mlir::Type type) {
  return mlir::isa<mlir::IntegerType, mlir::FloatType, mlir::RankedTensorType>(
      type);
}

std::optional<mlir::arith::CmpIPredicate> intCmpPredicate(llvm::StringRef op) {
  if (op == "==")
    return mlir::arith::CmpIPredicate::eq;
  if (op == "!=")
    return mlir::arith::CmpIPredicate::ne;
  if (op == "<")
    return mlir::arith::CmpIPredicate::slt;
  if (op == "<=")
    return mlir::arith::CmpIPredicate::sle;
  if (op == ">")
    return mlir::arith::CmpIPredicate::sgt;
  if (op == ">=")
    return mlir::arith::CmpIPredicate::sge;
  return std::nullopt;
}

std::optional<mlir::arith::CmpFPredicate>
floatCmpPredicate(llvm::StringRef op) {
  if (op == "==")
    return mlir::arith::CmpFPredicate::OEQ;
  if (op == "!=")
    return mlir::arith::CmpFPredicate::ONE;
  if (op == "<")
    return mlir::arith::CmpFPredicate::OLT;
  if (op == "<=")
    return mlir::arith::CmpFPredicate::OLE;
  if (op == ">")
    return mlir::arith::CmpFPredicate::OGT;
  if (op == ">=")
    return mlir::arith::CmpFPredicate::OGE;
  return std::nullopt;
}

bool isEmptyFStringFormatSpec(const parser::NodePtr *formatSpecNode) {
  if (!formatSpecNode || !*formatSpecNode)
    return true;
  if ((*formatSpecNode)->kind != "JoinedStr")
    return false;
  const std::vector<parser::NodePtr> *values =
      nodeListField(**formatSpecNode, "values");
  if (!values)
    return false;
  for (const parser::NodePtr &part : *values) {
    if (!part || part->kind != "Constant")
      return false;
    const parser::FieldValue *value = valueField(*part, "value");
    const auto *text = value ? std::get_if<std::string>(value) : nullptr;
    if (!text || !text->empty())
      return false;
  }
  return true;
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

std::optional<std::size_t> staticTupleArity(mlir::Type type,
                                            mlir::Value value) {
  auto tupleType = mlir::dyn_cast<py::TupleType>(type);
  if (!tupleType)
    return std::nullopt;

  llvm::ArrayRef<mlir::Type> elementTypes = tupleType.getElementTypes();
  if (elementTypes.empty())
    return 0;
  if (elementTypes.size() > 1)
    return elementTypes.size();
  if (auto create = value.getDefiningOp<py::TupleCreateOp>())
    return create.getElements().size();
  return std::nullopt;
}

mlir::Value emitBoolConstant(mlir::OpBuilder &builder, mlir::Location loc,
                             bool value) {
  return builder.create<mlir::arith::ConstantIntOp>(loc, value ? 1 : 0, 1);
}

} // namespace

Value Builder::Impl::emitExpression(const parser::Node &expr) {
  if (expr.kind == "Constant")
    return emitConstant(expr);
  if (expr.kind == "Name")
    return emitName(expr);
  if (expr.kind == "Await")
    return emitAwait(expr);
  if (expr.kind == "Yield" || expr.kind == "YieldFrom") {
    error(expr, "generator yield expression is parsed but generator lowering "
                "is not implemented in the C++ emitter yet");
    return Value{{}, noneType()};
  }
  if (expr.kind == "BoolOp")
    return emitBoolOp(expr);
  if (expr.kind == "IfExp")
    return emitIfExp(expr);
  if (expr.kind == "Lambda") {
    error(expr, "lambda expression requires an expected Callable[...] type in "
                "the C++ emitter");
    return Value{{}, noneType()};
  }
  if (expr.kind == "NamedExpr")
    return emitNamedExpr(expr);
  if (expr.kind == "BinOp")
    return emitBinOp(expr);
  if (expr.kind == "Compare")
    return emitCompare(expr);
  if (expr.kind == "UnaryOp")
    return emitUnaryOp(expr);
  if (expr.kind == "Call")
    return emitCall(expr);
  if (expr.kind == "Attribute")
    return emitAttribute(expr);
  if (expr.kind == "Dict")
    return emitDict(expr);
  if (expr.kind == "DictComp")
    return emitDictComprehension(expr);
  if (expr.kind == "Set") {
    error(expr, "set literal is parsed but set lowering is not implemented in "
                "the C++ emitter yet");
    return Value{{}, noneType()};
  }
  if (expr.kind == "JoinedStr")
    return emitJoinedStr(expr);
  if (expr.kind == "FormattedValue")
    return emitFormattedValue(expr);
  if (expr.kind == "TemplateStr")
    return emitTemplateStr(expr);
  if (expr.kind == "Interpolation")
    return emitInterpolation(expr);
  if (expr.kind == "List")
    return emitList(expr);
  if (expr.kind == "ListComp")
    return emitListComprehension(expr);
  if (expr.kind == "SetComp" || expr.kind == "GeneratorExp") {
    error(expr, "comprehension expression is parsed but comprehension "
                "lowering is not implemented in the C++ emitter yet");
    return Value{{}, noneType()};
  }
  if (expr.kind == "Tuple")
    return emitTupleLiteral(expr);
  if (expr.kind == "Subscript")
    return emitSubscript(expr);
  if (expr.kind == "Slice") {
    error(expr, "slice expression is valid only in subscript lowering and is "
                "not implemented as a standalone value");
    return Value{{}, noneType()};
  }
  if (expr.kind == "Starred") {
    error(expr, "starred expression is parsed but unpack lowering is not "
                "implemented in the C++ emitter yet");
    return Value{{}, noneType()};
  }
  error(expr,
        "C++ emitter does not support expression kind '" + expr.kind + "' yet");
  return Value{{}, noneType()};
}

Value Builder::Impl::emitExpressionWithExpectedType(const parser::Node &expr,
                                                    mlir::Type expectedType) {
  if (!expectedType)
    return emitExpression(expr);

  if (expr.kind == "Lambda")
    return emitLambda(expr, expectedType);

  if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(expectedType)) {
    if (std::optional<std::int64_t> value = staticIndexValue(expr)) {
      mlir::Value result = builder.create<mlir::arith::ConstantIntOp>(
          loc(expr), *value, intTy.getWidth());
      return Value{result, expectedType};
    }
  }

  if (expr.kind == "Call") {
    const parser::NodePtr *func = nodeField(expr, "func");
    const std::vector<parser::NodePtr> *args = nodeListField(expr, "args");
    const std::vector<parser::NodePtr> *keywords =
        nodeListField(expr, "keywords");
    const std::string *name = func && *func && (*func)->kind == "Name"
                                  ? stringField(**func, "id")
                                  : nullptr;
    if (name && *name == "list" && args && (!keywords || keywords->empty())) {
      mlir::Type elementType = listLiteralElementTypeForExpected(expectedType);
      if (!elementType) {
        error(expr, "list() requires a list-like expected type");
        return Value{{}, expectedType};
      }
      if (args->empty())
        return emitListFromValues(expr, {}, elementType);
      if (args->size() == 1 && args->front()) {
        if (args->front()->kind == "GeneratorExp") {
          return emitListComprehension(*args->front(), elementType);
        }
        if (args->front()->kind == "Call") {
          const parser::NodePtr *rangeFunc = nodeField(*args->front(), "func");
          if (rangeFunc && *rangeFunc && (*rangeFunc)->kind == "Name") {
            const std::string *rangeName = stringField(**rangeFunc, "id");
            if (rangeName && *rangeName == "range") {
              std::optional<StaticRangeElements> range =
                  emitStaticRangeElements(*args->front(), "list()",
                                          elementType);
              if (!range)
                return Value{{}, expectedType};
              return emitListFromValues(expr, range->values, elementType);
            }
          }
        }
        if (args->front()->kind == "List") {
          const std::vector<parser::NodePtr> *elements =
              nodeListField(*args->front(), "elts");
          if (elements && elements->empty())
            return emitListFromValues(expr, {}, elementType);
        }
        Value source = emitExpression(*args->front());
        if (!source.value)
          return Value{{}, expectedType};
        std::optional<std::vector<Value>> elements =
            finiteSequenceElements(*args->front(), source, "list()");
        if (!elements)
          return Value{{}, expectedType};
        return emitListFromValues(expr, *elements, elementType);
      }
      error(expr, "list() expects at most one argument");
      return Value{{}, expectedType};
    }
    if (name && *name == "tuple" && args && (!keywords || keywords->empty())) {
      auto tupleTy = mlir::dyn_cast<py::TupleType>(expectedType);
      if (!tupleTy) {
        error(expr, "tuple() requires a tuple[...] expected type");
        return Value{{}, expectedType};
      }
      llvm::ArrayRef<mlir::Type> elementTypes = tupleTy.getElementTypes();
      auto homogeneousElementType = [&]() -> mlir::Type {
        if (elementTypes.empty())
          return {};
        mlir::Type first = elementTypes.front();
        if (llvm::all_of(elementTypes,
                         [&](mlir::Type type) { return type == first; }))
          return first;
        return {};
      };
      auto emitExpectedTuple = [&](llvm::ArrayRef<Value> elements) -> Value {
        if (elementTypes.empty()) {
          if (!elements.empty()) {
            error(expr, "empty tuple annotation cannot accept elements");
            return Value{{}, expectedType};
          }
          mlir::Value result =
              builder.create<py::TupleEmptyOp>(loc(expr), expectedType);
          return Value{result, expectedType};
        }
        if (elementTypes.size() > 1 && elements.size() != elementTypes.size()) {
          error(expr, "tuple() fixed-arity expected type requires " +
                          std::to_string(elementTypes.size()) +
                          " elements, got " + std::to_string(elements.size()));
          return Value{{}, expectedType};
        }
        mlir::Type homogeneous =
            elementTypes.size() == 1 ? elementTypes.front() : mlir::Type{};
        for (auto indexed : llvm::enumerate(elements)) {
          mlir::Type expectedElement =
              homogeneous ? homogeneous : elementTypes[indexed.index()];
          if (!typeAssignable(expectedElement, indexed.value().type)) {
            error(expr, "tuple() element type mismatch: expected " +
                            typeString(expectedElement) + ", got " +
                            typeString(indexed.value().type));
            return Value{{}, expectedType};
          }
        }
        llvm::SmallVector<mlir::Value> values;
        values.reserve(elements.size());
        for (const Value &element : elements)
          values.push_back(element.value);
        mlir::Value result =
            builder.create<py::TupleCreateOp>(loc(expr), expectedType, values);
        return Value{result, expectedType};
      };

      if (args->empty())
        return emitExpectedTuple({});
      if (args->size() != 1 || !args->front()) {
        error(expr, "tuple() expects at most one argument");
        return Value{{}, expectedType};
      }

      mlir::Type preferredElementType = homogeneousElementType();
      if (!preferredElementType && elementTypes.size() > 1) {
        error(expr, "tuple() generator/range expected type must be "
                    "homogeneous so range element type is known");
        return Value{{}, expectedType};
      }
      if (args->front()->kind == "GeneratorExp") {
        std::optional<std::vector<Value>> elements =
            emitStaticGeneratorElements(*args->front(), "tuple()",
                                        preferredElementType);
        if (!elements)
          return Value{{}, expectedType};
        return emitExpectedTuple(*elements);
      }
      if (args->front()->kind == "Call") {
        const parser::NodePtr *rangeFunc = nodeField(*args->front(), "func");
        if (rangeFunc && *rangeFunc && (*rangeFunc)->kind == "Name") {
          const std::string *rangeName = stringField(**rangeFunc, "id");
          if (rangeName && *rangeName == "range") {
            std::optional<StaticRangeElements> range = emitStaticRangeElements(
                *args->front(), "tuple()", preferredElementType);
            if (!range)
              return Value{{}, expectedType};
            return emitExpectedTuple(range->values);
          }
        }
      }
    }
    if (name && args && args->empty() && (!keywords || keywords->empty())) {
      if (*name == "dict") {
        std::optional<std::pair<mlir::Type, mlir::Type>> dictTypes =
            dictLiteralKeyValueTypesForExpected(expectedType);
        if (!dictTypes) {
          error(expr, "dict() requires a mapping-like expected type");
          return Value{{}, expectedType};
        }
        if (!dictStorageSupported(dictTypes->first, dictTypes->second)) {
          error(expr, "dict() expected key/value types are not supported by "
                      "typed memref lowering yet: " +
                          typeString(dictTypes->first) + ", " +
                          typeString(dictTypes->second));
          return Value{{}, expectedType};
        }
        mlir::Type concreteType = dictType(dictTypes->first, dictTypes->second);
        mlir::Value result =
            builder.create<py::DictEmptyOp>(loc(expr), concreteType);
        return Value{result, concreteType};
      }
    }
  }

  if (expr.kind == "ListComp") {
    mlir::Type elementType = listLiteralElementTypeForExpected(expectedType);
    if (!elementType) {
      error(expr, "list comprehension requires a list-like expected type");
      return Value{{}, expectedType};
    }
    return emitListComprehension(expr, elementType);
  }

  if (expr.kind == "DictComp") {
    std::optional<std::pair<mlir::Type, mlir::Type>> dictTypes =
        dictLiteralKeyValueTypesForExpected(expectedType);
    if (!dictTypes) {
      error(expr, "dict comprehension requires a mapping-like expected type");
      return Value{{}, expectedType};
    }
    return emitDictComprehension(expr, dictTypes->first, dictTypes->second);
  }

  if (expr.kind == "Tuple") {
    if (!mlir::isa<py::TupleType>(expectedType)) {
      error(expr, "tuple literal requires a tuple[...] expected type");
      return Value{{}, expectedType};
    }
    return emitTupleLiteral(expr, expectedType);
  }

  if (expr.kind == "List") {
    mlir::Type elementType = listLiteralElementTypeForExpected(expectedType);
    if (!elementType) {
      error(expr, "list literal requires a list-like expected type");
      return Value{{}, expectedType};
    }
    return emitList(expr, elementType);
  }

  if (expr.kind == "Dict") {
    std::optional<std::pair<mlir::Type, mlir::Type>> dictTypes =
        dictLiteralKeyValueTypesForExpected(expectedType);
    if (!dictTypes) {
      error(expr, "dict literal requires a mapping-like expected type");
      return Value{{}, expectedType};
    }
    return emitDict(expr, dictTypes->first, dictTypes->second);
  }

  return emitExpression(expr);
}

Value Builder::Impl::emitCondition(const parser::Node &expr) {
  Value value = emitExpression(expr);
  if (!value.value)
    return Value{{}, i1Type()};
  if (value.type == i1Type())
    return value;
  if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(value.type)) {
    mlir::Value zero = builder.create<mlir::arith::ConstantIntOp>(
        loc(expr), 0, intTy.getWidth());
    mlir::Value bit = builder.create<mlir::arith::CmpIOp>(
        loc(expr), mlir::arith::CmpIPredicate::ne, value.value, zero);
    return Value{bit, i1Type()};
  }
  if (auto floatTy = mlir::dyn_cast<mlir::FloatType>(value.type)) {
    mlir::Value zero = builder.create<mlir::arith::ConstantOp>(
        loc(expr), value.type, builder.getFloatAttr(floatTy, 0.0));
    mlir::Value bit = builder.create<mlir::arith::CmpFOp>(
        loc(expr), mlir::arith::CmpFPredicate::UNE, value.value, zero);
    return Value{bit, i1Type()};
  }
  if (value.type == intType()) {
    mlir::Value zero =
        builder.create<py::IntConstantOp>(loc(expr), intType(), "0");
    mlir::Value truth =
        builder.create<py::NeOp>(loc(expr), boolType(), value.value, zero);
    mlir::Value bit =
        builder.create<py::CastToPrimOp>(loc(expr), i1Type(), truth, "exact");
    return Value{bit, i1Type()};
  }
  if (value.type == floatType()) {
    mlir::Value zero = builder.create<py::FloatConstantOp>(
        loc(expr), floatType(), builder.getF64FloatAttr(0.0));
    mlir::Value truth =
        builder.create<py::NeOp>(loc(expr), boolType(), value.value, zero);
    mlir::Value bit =
        builder.create<py::CastToPrimOp>(loc(expr), i1Type(), truth, "exact");
    return Value{bit, i1Type()};
  }
  if (value.type == strType()) {
    mlir::Value empty =
        builder.create<py::StrConstantOp>(loc(expr), strType(), "");
    mlir::Value truth =
        builder.create<py::NeOp>(loc(expr), boolType(), value.value, empty);
    mlir::Value bit =
        builder.create<py::CastToPrimOp>(loc(expr), i1Type(), truth, "exact");
    return Value{bit, i1Type()};
  }
  if (std::optional<std::size_t> arity =
          staticTupleArity(value.type, value.value)) {
    mlir::Value bit = emitBoolConstant(builder, loc(expr), *arity != 0);
    return Value{bit, i1Type()};
  }
  if (listElementType(value.type)) {
    auto found = finiteListElements.find(value.value);
    if (found == finiteListElements.end()) {
      error(expr,
            "list truthiness requires a list with statically known elements");
      return Value{{}, i1Type()};
    }
    mlir::Value bit =
        emitBoolConstant(builder, loc(expr), !found->second.empty());
    return Value{bit, i1Type()};
  }
  if (value.type == noneType()) {
    mlir::Value bit = emitBoolConstant(builder, loc(expr), false);
    return Value{bit, i1Type()};
  }
  if (value.type != boolType()) {
    error(expr,
          "if condition must currently lower to numeric, !py.str, !py.bool, "
          "statically finite tuple/list, None, or i1");
    return Value{{}, i1Type()};
  }
  mlir::Value bit = builder.create<py::CastToPrimOp>(loc(expr), i1Type(),
                                                     value.value, "exact");
  return Value{bit, i1Type()};
}

Value Builder::Impl::emitIfExp(const parser::Node &expr) {
  const parser::NodePtr *testNode = nodeField(expr, "test");
  const parser::NodePtr *bodyNode = nodeField(expr, "body");
  const parser::NodePtr *elseNode = nodeField(expr, "orelse");
  if (!testNode || !*testNode || !bodyNode || !*bodyNode || !elseNode ||
      !*elseNode) {
    error(expr, "IfExp.test, IfExp.body, or IfExp.orelse is missing");
    return Value{{}, noneType()};
  }

  std::optional<mlir::Type> resultType = inferExpressionType(expr);
  if (!resultType) {
    error(expr, "conditional expression requires statically identical branch "
                "types");
    return Value{{}, noneType()};
  }
  if (!isStructuredIfExpType(*resultType)) {
    error(expr, "conditional expression currently supports only "
                "ownership-neutral primitive or tensor result types, got " +
                    typeString(*resultType));
    return Value{{}, *resultType};
  }

  Value condition = emitCondition(**testNode);
  if (!condition.value)
    return Value{{}, *resultType};

  auto ifOp = builder.create<mlir::scf::IfOp>(
      loc(expr), mlir::TypeRange{*resultType}, condition.value,
      /*withElseRegion=*/true);
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(ifOp.thenBlock());
    Value body = emitExpression(**bodyNode);
    if (!body.value)
      return Value{{}, *resultType};
    if (body.type != *resultType) {
      error(**bodyNode, "conditional expression then-branch has type " +
                            typeString(body.type) + ", expected " +
                            typeString(*resultType));
      return Value{{}, *resultType};
    }
    builder.create<mlir::scf::YieldOp>(loc(**bodyNode), body.value);
  }
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(ifOp.elseBlock());
    Value elseValue = emitExpression(**elseNode);
    if (!elseValue.value)
      return Value{{}, *resultType};
    if (elseValue.type != *resultType) {
      error(**elseNode, "conditional expression else-branch has type " +
                            typeString(elseValue.type) + ", expected " +
                            typeString(*resultType));
      return Value{{}, *resultType};
    }
    builder.create<mlir::scf::YieldOp>(loc(**elseNode), elseValue.value);
  }
  builder.setInsertionPointAfter(ifOp);
  return Value{ifOp.getResult(0), *resultType};
}

Value Builder::Impl::emitNamedExpr(const parser::Node &expr) {
  const parser::NodePtr *targetNode = nodeField(expr, "target");
  const parser::NodePtr *valueNode = nodeField(expr, "value");
  if (!targetNode || !*targetNode || !valueNode || !*valueNode) {
    error(expr, "NamedExpr.target or NamedExpr.value is missing");
    return Value{{}, noneType()};
  }
  if ((*targetNode)->kind != "Name") {
    error(**targetNode, "named expression currently supports only name "
                        "targets in the C++ emitter");
    return Value{{}, noneType()};
  }
  const std::string *name = stringField(**targetNode, "id");
  if (!name) {
    error(**targetNode, "named expression target name is missing");
    return Value{{}, noneType()};
  }

  Value value = emitExpression(**valueNode);
  if (!value.value)
    return Value{{}, noneType()};
  assignValueToTarget(expr, **targetNode, value, valueNode->get());
  return value;
}

Value Builder::Impl::emitName(const parser::Node &expr) {
  const std::string *name = stringField(expr, "id");
  if (!name) {
    error(expr, "Name.id is missing");
    return Value{{}, noneType()};
  }
  auto found = symbols.find(*name);
  if (found != symbols.end())
    return found->second;

  auto constant = primitiveConstants.find(*name);
  if (constant != primitiveConstants.end()) {
    mlir::Value materialized;
    if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(constant->second.type)) {
      materialized = builder.create<mlir::arith::ConstantIntOp>(
          loc(expr), constant->second.integerValue, intTy.getWidth());
    } else if (auto floatTy =
                   mlir::dyn_cast<mlir::FloatType>(constant->second.type)) {
      materialized = builder.create<mlir::arith::ConstantOp>(
          loc(expr), constant->second.type,
          builder.getFloatAttr(floatTy, constant->second.floatValue));
    }
    if (!materialized) {
      error(expr, "unsupported static primitive constant type " +
                      typeString(constant->second.type));
      return Value{{}, constant->second.type};
    }
    return Value{materialized, constant->second.type};
  }

  auto localFunction = localFunctions.find(*name);
  if (localFunction != localFunctions.end())
    return emitFunctionObject(expr, localFunction->second);

  auto function = functions.find(*name);
  if (function != functions.end()) {
    return emitFunctionObject(expr, function->second);
  }

  if (isBuiltinExceptionClass(*name)) {
    mlir::Type resultType = py::TypeType::get(&context, exceptionType());
    mlir::Value object =
        builder.create<py::ClassObjectOp>(loc(expr), resultType, *name);
    return Value{object, resultType};
  }
  if (classes.count(*name)) {
    mlir::Type resultType = py::TypeType::get(&context, classType(*name));
    mlir::Value object =
        builder.create<py::ClassObjectOp>(loc(expr), resultType, *name);
    return Value{object, resultType};
  }

  error(expr, "unknown name '" + *name + "'");
  return Value{{}, noneType()};
}

Value Builder::Impl::emitBinaryOperation(const parser::Node &expr,
                                         llvm::StringRef op, const Value &lhs,
                                         const Value &rhs) {
  if (op == "+" && mlir::isa<py::TupleType>(lhs.type) &&
      mlir::isa<py::TupleType>(rhs.type)) {
    std::optional<std::vector<Value>> lhsElements =
        finiteSequenceElements(expr, lhs, "tuple concatenation");
    if (!lhsElements)
      return Value{{}, noneType()};
    std::optional<std::vector<Value>> rhsElements =
        finiteSequenceElements(expr, rhs, "tuple concatenation");
    if (!rhsElements)
      return Value{{}, noneType()};
    std::vector<Value> elements;
    elements.reserve(lhsElements->size() + rhsElements->size());
    elements.insert(elements.end(), lhsElements->begin(), lhsElements->end());
    elements.insert(elements.end(), rhsElements->begin(), rhsElements->end());
    return emitTuple(elements);
  }
  if (op == "+" && listElementType(lhs.type) && listElementType(rhs.type)) {
    if (listElementType(lhs.type) != listElementType(rhs.type)) {
      error(expr, "list concatenation element type mismatch: lhs has " +
                      typeString(listElementType(lhs.type)) + ", rhs has " +
                      typeString(listElementType(rhs.type)));
      return Value{{}, noneType()};
    }
    std::optional<std::vector<Value>> lhsElements =
        finiteSequenceElements(expr, lhs, "list concatenation");
    if (!lhsElements)
      return Value{{}, noneType()};
    std::optional<std::vector<Value>> rhsElements =
        finiteSequenceElements(expr, rhs, "list concatenation");
    if (!rhsElements)
      return Value{{}, noneType()};
    std::vector<Value> elements;
    elements.reserve(lhsElements->size() + rhsElements->size());
    elements.insert(elements.end(), lhsElements->begin(), lhsElements->end());
    elements.insert(elements.end(), rhsElements->begin(), rhsElements->end());
    return emitListFromValues(expr, elements, listElementType(lhs.type));
  }
  if (lhs.type != rhs.type) {
    if (op == "@" && mlir::isa<mlir::RankedTensorType>(lhs.type) &&
        mlir::isa<mlir::RankedTensorType>(rhs.type))
      return emitTensorMatmul(expr, lhs, rhs);
    error(expr, "C++ emitter cannot infer mixed-type binary operation result");
    return Value{{}, noneType()};
  }
  if (mlir::isa<mlir::RankedTensorType>(lhs.type)) {
    if (op == "@")
      return emitTensorMatmul(expr, lhs, rhs);
    auto tensorType = mlir::cast<mlir::RankedTensorType>(lhs.type);
    if (!mlir::isa<mlir::FloatType>(tensorType.getElementType())) {
      error(expr, "tensor arithmetic currently supports only float tensors");
      return Value{{}, lhs.type};
    }
    if (op == "+") {
      mlir::Value result =
          builder.create<mlir::arith::AddFOp>(loc(expr), lhs.value, rhs.value);
      return Value{result, lhs.type};
    }
    if (op == "-") {
      mlir::Value result =
          builder.create<mlir::arith::SubFOp>(loc(expr), lhs.value, rhs.value);
      return Value{result, lhs.type};
    }
    if (op == "*") {
      mlir::Value result =
          builder.create<mlir::arith::MulFOp>(loc(expr), lhs.value, rhs.value);
      return Value{result, lhs.type};
    }
  }
  if (mlir::isa<mlir::IntegerType>(lhs.type)) {
    if (op == "+") {
      mlir::Value result =
          builder.create<mlir::arith::AddIOp>(loc(expr), lhs.value, rhs.value);
      return Value{result, lhs.type};
    }
    if (op == "-") {
      mlir::Value result =
          builder.create<mlir::arith::SubIOp>(loc(expr), lhs.value, rhs.value);
      return Value{result, lhs.type};
    }
    if (op == "*") {
      mlir::Value result =
          builder.create<mlir::arith::MulIOp>(loc(expr), lhs.value, rhs.value);
      return Value{result, lhs.type};
    }
    if (op == "/") {
      mlir::Value result =
          builder.create<mlir::arith::DivSIOp>(loc(expr), lhs.value, rhs.value);
      return Value{result, lhs.type};
    }
    if (op == "//") {
      mlir::Value result = builder.create<mlir::arith::FloorDivSIOp>(
          loc(expr), lhs.value, rhs.value);
      return Value{result, lhs.type};
    }
    if (op == "%") {
      mlir::Value quotient = builder.create<mlir::arith::FloorDivSIOp>(
          loc(expr), lhs.value, rhs.value);
      mlir::Value product =
          builder.create<mlir::arith::MulIOp>(loc(expr), quotient, rhs.value);
      mlir::Value result =
          builder.create<mlir::arith::SubIOp>(loc(expr), lhs.value, product);
      return Value{result, lhs.type};
    }
    if (op == "<<") {
      mlir::Value result =
          builder.create<mlir::arith::ShLIOp>(loc(expr), lhs.value, rhs.value);
      return Value{result, lhs.type};
    }
    if (op == ">>") {
      mlir::Value result =
          builder.create<mlir::arith::ShRSIOp>(loc(expr), lhs.value, rhs.value);
      return Value{result, lhs.type};
    }
    if (op == "&") {
      mlir::Value result =
          builder.create<mlir::arith::AndIOp>(loc(expr), lhs.value, rhs.value);
      return Value{result, lhs.type};
    }
    if (op == "|") {
      mlir::Value result =
          builder.create<mlir::arith::OrIOp>(loc(expr), lhs.value, rhs.value);
      return Value{result, lhs.type};
    }
    if (op == "^") {
      mlir::Value result =
          builder.create<mlir::arith::XOrIOp>(loc(expr), lhs.value, rhs.value);
      return Value{result, lhs.type};
    }
  }
  if (mlir::isa<mlir::FloatType>(lhs.type)) {
    if (op == "+") {
      mlir::Value result =
          builder.create<mlir::arith::AddFOp>(loc(expr), lhs.value, rhs.value);
      return Value{result, lhs.type};
    }
    if (op == "-") {
      mlir::Value result =
          builder.create<mlir::arith::SubFOp>(loc(expr), lhs.value, rhs.value);
      return Value{result, lhs.type};
    }
    if (op == "*") {
      mlir::Value result =
          builder.create<mlir::arith::MulFOp>(loc(expr), lhs.value, rhs.value);
      return Value{result, lhs.type};
    }
    if (op == "/") {
      mlir::Value result =
          builder.create<mlir::arith::DivFOp>(loc(expr), lhs.value, rhs.value);
      return Value{result, lhs.type};
    }
    if (op == "%") {
      mlir::Value result =
          builder.create<mlir::arith::RemFOp>(loc(expr), lhs.value, rhs.value);
      return Value{result, lhs.type};
    }
  }
  if (op == "+") {
    mlir::Value result =
        builder.create<py::AddOp>(loc(), lhs.type, lhs.value, rhs.value);
    return Value{result, lhs.type};
  }
  if (op == "-") {
    if (lhs.type == strType()) {
      error(expr, "string subtraction is unsupported");
      return Value{{}, noneType()};
    }
    mlir::Value result =
        builder.create<py::SubOp>(loc(), lhs.type, lhs.value, rhs.value);
    return Value{result, lhs.type};
  }
  if (op == "*") {
    mlir::Value result =
        builder.create<py::MulOp>(loc(), lhs.type, lhs.value, rhs.value);
    return Value{result, lhs.type};
  }
  if (op == "/") {
    mlir::Value result =
        builder.create<py::DivOp>(loc(), lhs.type, lhs.value, rhs.value);
    return Value{result, lhs.type};
  }
  if (op == "//") {
    mlir::Value result =
        builder.create<py::FloorDivOp>(loc(), lhs.type, lhs.value, rhs.value);
    return Value{result, lhs.type};
  }
  if (op == "%") {
    mlir::Value result =
        builder.create<py::ModOp>(loc(), lhs.type, lhs.value, rhs.value);
    return Value{result, lhs.type};
  }
  if (op == "<<") {
    mlir::Value result =
        builder.create<py::LShiftOp>(loc(), lhs.type, lhs.value, rhs.value);
    return Value{result, lhs.type};
  }
  if (op == ">>") {
    mlir::Value result =
        builder.create<py::RShiftOp>(loc(), lhs.type, lhs.value, rhs.value);
    return Value{result, lhs.type};
  }
  if (op == "&") {
    mlir::Value result =
        builder.create<py::BitAndOp>(loc(), lhs.type, lhs.value, rhs.value);
    return Value{result, lhs.type};
  }
  if (op == "|") {
    mlir::Value result =
        builder.create<py::BitOrOp>(loc(), lhs.type, lhs.value, rhs.value);
    return Value{result, lhs.type};
  }
  if (op == "^") {
    mlir::Value result =
        builder.create<py::BitXorOp>(loc(), lhs.type, lhs.value, rhs.value);
    return Value{result, lhs.type};
  }
  error(expr, "C++ emitter does not support binary operator '" + op.str() +
                  "' for type " + typeString(lhs.type));
  return Value{{}, noneType()};
}

Value Builder::Impl::emitBinOp(const parser::Node &expr) {
  const parser::NodePtr *lhsNode = nodeField(expr, "left");
  const parser::NodePtr *rhsNode = nodeField(expr, "right");
  std::optional<std::string> op = symbolField(expr, "op");
  if (!lhsNode || !*lhsNode || !rhsNode || !*rhsNode || !op) {
    error(expr, "BinOp.left, BinOp.right, or BinOp.op is missing");
    return Value{{}, noneType()};
  }

  Value lhs = emitExpression(**lhsNode);
  Value rhs = lhs.type ? emitExpressionWithExpectedType(**rhsNode, lhs.type)
                       : emitExpression(**rhsNode);
  if (!lhs.value || !rhs.value)
    return Value{{}, noneType()};
  if (*op == "*") {
    auto repeatSequence = [&](const Value &sequence,
                              std::int64_t count) -> Value {
      if (!mlir::isa<py::TupleType>(sequence.type) &&
          !listElementType(sequence.type)) {
        error(expr, "sequence repetition requires a tuple or list operand");
        return Value{{}, noneType()};
      }
      std::optional<std::vector<Value>> source =
          finiteSequenceElements(expr, sequence, "sequence repetition");
      if (!source)
        return Value{{}, noneType()};

      std::vector<Value> elements;
      if (count > 0) {
        elements.reserve(source->size() * static_cast<std::size_t>(count));
        for (std::int64_t iteration = 0; iteration < count; ++iteration)
          elements.insert(elements.end(), source->begin(), source->end());
      }
      if (mlir::isa<py::TupleType>(sequence.type))
        return emitTuple(elements);
      return emitListFromValues(expr, elements, listElementType(sequence.type));
    };

    if (std::optional<std::int64_t> count = staticIndexValue(**rhsNode)) {
      if (mlir::isa<py::TupleType>(lhs.type) || listElementType(lhs.type))
        return repeatSequence(lhs, *count);
    }
    if (std::optional<std::int64_t> count = staticIndexValue(**lhsNode)) {
      if (mlir::isa<py::TupleType>(rhs.type) || listElementType(rhs.type))
        return repeatSequence(rhs, *count);
    }
  }
  if (lhs.type != rhs.type) {
    Value coercedLhs = emitExpressionWithExpectedType(**lhsNode, rhs.type);
    if (coercedLhs.value && coercedLhs.type == rhs.type)
      lhs = coercedLhs;
  }
  if (*op == "**") {
    auto intTy = mlir::dyn_cast<mlir::IntegerType>(lhs.type);
    std::optional<std::int64_t> exponent = staticIndexValue(**rhsNode);
    if (!intTy || !exponent) {
      error(expr, "primitive integer exponentiation requires a statically "
                  "known integer exponent");
      return Value{{}, noneType()};
    }
    if (*exponent < 0) {
      error(expr, "negative primitive integer exponent would produce a Python "
                  "float and is not supported for fixed-width Int lowering");
      return Value{{}, noneType()};
    }
    mlir::Value result = builder.create<mlir::arith::ConstantIntOp>(
        loc(expr), 1, intTy.getWidth());
    mlir::Value base = lhs.value;
    std::int64_t remaining = *exponent;
    while (remaining > 0) {
      if ((remaining & 1) != 0)
        result = builder.create<mlir::arith::MulIOp>(loc(expr), result, base);
      remaining >>= 1;
      if (remaining > 0)
        base = builder.create<mlir::arith::MulIOp>(loc(expr), base, base);
    }
    return Value{result, lhs.type};
  }
  return emitBinaryOperation(expr, *op, lhs, rhs);
}

Value Builder::Impl::emitFormattedValue(const parser::Node &expr) {
  const parser::NodePtr *valueNode = nodeField(expr, "value");
  const parser::FieldValue *conversionValue = valueField(expr, "conversion");
  const parser::NodePtr *formatSpecNode = nodeField(expr, "format_spec");
  if (!valueNode || !*valueNode) {
    error(expr, "FormattedValue.value is missing");
    return Value{{}, strType()};
  }
  const auto *conversion =
      conversionValue ? std::get_if<std::int64_t>(conversionValue) : nullptr;
  if (!conversion) {
    error(expr, "FormattedValue.conversion is missing");
    return Value{{}, strType()};
  }
  if (*conversion != -1 && *conversion != static_cast<std::int64_t>('r') &&
      *conversion != static_cast<std::int64_t>('s')) {
    if (*conversion == static_cast<std::int64_t>('a'))
      error(expr, "f-string !a conversion requires ascii(repr(...)) escaping "
                  "and is not implemented in the C++ emitter yet");
    else
      error(expr, "unsupported f-string conversion in the C++ emitter");
    return Value{{}, strType()};
  }
  if (!isEmptyFStringFormatSpec(formatSpecNode)) {
    error(expr, "non-empty f-string format specs require __format__ lowering "
                "and are not implemented in the C++ emitter yet");
    return Value{{}, strType()};
  }

  Value value = emitExpression(**valueNode);
  if (!value.value)
    return Value{{}, strType()};
  if (*conversion == static_cast<std::int64_t>('r'))
    return emitRepr(value);
  if (*conversion == static_cast<std::int64_t>('s')) {
    if (value.type == strType())
      return value;
    return emitRepr(value);
  }
  if (value.type == strType())
    return value;
  return emitRepr(value);
}

Value Builder::Impl::emitJoinedStr(const parser::Node &expr) {
  const std::vector<parser::NodePtr> *values = nodeListField(expr, "values");
  if (!values) {
    error(expr, "JoinedStr.values is missing");
    return Value{{}, strType()};
  }

  Value result;
  for (const parser::NodePtr &partNode : *values) {
    if (!partNode) {
      error(expr, "JoinedStr contains an empty part");
      return Value{{}, strType()};
    }

    Value part;
    if (partNode->kind == "Constant") {
      part = emitConstant(*partNode);
      if (part.type != strType()) {
        error(*partNode, "JoinedStr constant parts must be strings");
        return Value{{}, strType()};
      }
    } else if (partNode->kind == "FormattedValue") {
      part = emitFormattedValue(*partNode);
    } else if (partNode->kind == "JoinedStr") {
      part = emitJoinedStr(*partNode);
    } else {
      error(*partNode, "JoinedStr part kind '" + partNode->kind +
                           "' is not supported by the C++ emitter");
      return Value{{}, strType()};
    }
    if (!part.value)
      return Value{{}, strType()};
    if (part.type != strType()) {
      error(*partNode, "JoinedStr part lowered to " + typeString(part.type) +
                           ", expected !py.str");
      return Value{{}, strType()};
    }
    if (!result.value) {
      result = part;
      continue;
    }
    mlir::Value concatenated = builder.create<py::AddOp>(
        loc(expr), strType(), result.value, part.value);
    result = Value{concatenated, strType()};
  }

  if (result.value)
    return result;
  mlir::Value empty =
      builder.create<py::StrConstantOp>(loc(expr), strType(), "");
  return Value{empty, strType()};
}

Value Builder::Impl::emitInterpolation(const parser::Node &expr) {
  const parser::NodePtr *valueNode = nodeField(expr, "value");
  const parser::FieldValue *conversionValue = valueField(expr, "conversion");
  const parser::NodePtr *formatSpecNode = nodeField(expr, "format_spec");
  if (!valueNode || !*valueNode) {
    error(expr, "Interpolation.value is missing");
    return Value{{}, strType()};
  }

  const auto *conversion =
      conversionValue ? std::get_if<std::int64_t>(conversionValue) : nullptr;
  if (!conversion) {
    error(expr, "Interpolation.conversion is missing");
    return Value{{}, strType()};
  }
  if (*conversion != -1 && *conversion != static_cast<std::int64_t>('r') &&
      *conversion != static_cast<std::int64_t>('s')) {
    if (*conversion == static_cast<std::int64_t>('a'))
      error(expr, "t-string !a conversion requires ascii(repr(...)) escaping "
                  "and is not implemented in the C++ emitter yet");
    else
      error(expr, "unsupported t-string conversion in the C++ emitter");
    return Value{{}, strType()};
  }
  if (!isEmptyFStringFormatSpec(formatSpecNode)) {
    error(expr, "non-empty t-string format specs require __format__ lowering "
                "and are not implemented in the C++ emitter yet");
    return Value{{}, strType()};
  }

  Value value = emitExpression(**valueNode);
  if (!value.value)
    return Value{{}, strType()};
  if (*conversion == static_cast<std::int64_t>('r'))
    return emitRepr(value);
  if (*conversion == static_cast<std::int64_t>('s')) {
    if (value.type == strType())
      return value;
    return emitRepr(value);
  }
  if (value.type == strType())
    return value;
  return emitRepr(value);
}

Value Builder::Impl::emitTemplateStr(const parser::Node &expr) {
  const std::vector<parser::NodePtr> *values = nodeListField(expr, "values");
  if (!values) {
    error(expr, "TemplateStr.values is missing");
    return Value{{}, strType()};
  }

  Value result;
  for (const parser::NodePtr &partNode : *values) {
    if (!partNode) {
      error(expr, "TemplateStr contains an empty part");
      return Value{{}, strType()};
    }

    Value part;
    if (partNode->kind == "Constant") {
      part = emitConstant(*partNode);
      if (part.type != strType()) {
        error(*partNode, "TemplateStr constant parts must be strings");
        return Value{{}, strType()};
      }
    } else if (partNode->kind == "Interpolation") {
      part = emitInterpolation(*partNode);
    } else if (partNode->kind == "JoinedStr") {
      part = emitJoinedStr(*partNode);
    } else {
      error(*partNode, "TemplateStr part kind '" + partNode->kind +
                           "' is not supported by the C++ emitter");
      return Value{{}, strType()};
    }
    if (!part.value)
      return Value{{}, strType()};
    if (part.type != strType()) {
      error(*partNode, "TemplateStr part lowered to " + typeString(part.type) +
                           ", expected !py.str");
      return Value{{}, strType()};
    }
    if (!result.value) {
      result = part;
      continue;
    }
    mlir::Value concatenated = builder.create<py::AddOp>(
        loc(expr), strType(), result.value, part.value);
    result = Value{concatenated, strType()};
  }

  if (result.value)
    return result;
  mlir::Value empty =
      builder.create<py::StrConstantOp>(loc(expr), strType(), "");
  return Value{empty, strType()};
}

Value Builder::Impl::emitTupleLiteralMembership(
    const parser::Node &expr, const parser::Node &candidateNode,
    const parser::Node &tupleNode, bool negate) {
  const std::vector<parser::NodePtr> *elements =
      nodeListField(tupleNode, "elts");
  if (!elements) {
    error(tupleNode, "Tuple.elts is missing");
    return Value{{}, i1Type()};
  }

  Value candidate = emitExpression(candidateNode);
  if (!candidate.value)
    return Value{{}, i1Type()};
  if (!candidate.type) {
    error(candidateNode, "tuple membership candidate type is unknown");
    return Value{{}, i1Type()};
  }
  bool integerMembership = mlir::isa<mlir::IntegerType>(candidate.type);
  bool floatMembership = mlir::isa<mlir::FloatType>(candidate.type);
  bool pyIntMembership = candidate.type == intType();
  bool pyFloatMembership = candidate.type == floatType();
  if (!integerMembership && !floatMembership && !pyIntMembership &&
      !pyFloatMembership) {
    error(expr, "tuple literal membership currently supports only primitive "
                "or object integer/float elements");
    return Value{{}, i1Type()};
  }

  llvm::SmallVector<Value> values;
  values.reserve(elements->size());
  for (const parser::NodePtr &elementNode : *elements) {
    if (!elementNode) {
      error(tupleNode, "Tuple contains an empty element");
      return Value{{}, i1Type()};
    }
    Value element = emitExpression(*elementNode);
    if (!element.value)
      return Value{{}, i1Type()};
    if (!element.type) {
      error(*elementNode, "tuple membership element type is unknown");
      return Value{{}, i1Type()};
    }
    if (element.type != candidate.type) {
      error(*elementNode, "tuple membership requires the candidate and every "
                          "element to have the same numeric static type");
      return Value{{}, i1Type()};
    }
    values.push_back(element);
  }

  mlir::Value result =
      builder.create<mlir::arith::ConstantIntOp>(loc(expr), 0, 1);
  for (const Value &element : values) {
    mlir::Value equals;
    if (integerMembership) {
      equals = builder.create<mlir::arith::CmpIOp>(
          loc(expr), mlir::arith::CmpIPredicate::eq, candidate.value,
          element.value);
    } else if (floatMembership) {
      equals = builder.create<mlir::arith::CmpFOp>(
          loc(expr), mlir::arith::CmpFPredicate::OEQ, candidate.value,
          element.value);
    } else {
      mlir::Value pyEquals = builder.create<py::EqOp>(
          loc(expr), boolType(), candidate.value, element.value);
      equals = builder.create<py::CastToPrimOp>(loc(expr), i1Type(), pyEquals,
                                                "exact");
    }
    result = builder.create<mlir::arith::OrIOp>(loc(expr), result, equals);
  }
  if (negate) {
    mlir::Value trueBit =
        builder.create<mlir::arith::ConstantIntOp>(loc(expr), 1, 1);
    result = builder.create<mlir::arith::XOrIOp>(loc(expr), result, trueBit);
  }
  return Value{result, i1Type()};
}

Value Builder::Impl::emitCompare(const parser::Node &expr) {
  const parser::NodePtr *lhsNode = nodeField(expr, "left");
  std::optional<std::vector<std::string>> ops = symbolListField(expr, "ops");
  const std::vector<parser::NodePtr> *comparators =
      nodeListField(expr, "comparators");
  if (!lhsNode || !*lhsNode || !ops || !comparators || ops->empty() ||
      ops->size() != comparators->size()) {
    error(expr, "Compare.left, Compare.ops, or Compare.comparators is missing");
    return Value{{}, boolType()};
  }

  auto falseBit = [&](mlir::Location at) -> mlir::Value {
    return builder.create<mlir::arith::ConstantIntOp>(at, 0, 1);
  };
  auto boolBit = [&](mlir::Location at, bool value) -> mlir::Value {
    return builder.create<mlir::arith::ConstantIntOp>(at, value ? 1 : 0, 1);
  };
  auto invertBit = [&](mlir::Location at, mlir::Value bit) -> mlir::Value {
    mlir::Value trueBit = builder.create<mlir::arith::ConstantIntOp>(at, 1, 1);
    return builder.create<mlir::arith::XOrIOp>(at, bit, trueBit);
  };
  auto boolObject = [&](mlir::Location at, mlir::Value bit) -> Value {
    mlir::Value result =
        builder.create<py::CastFromPrimOp>(at, boolType(), bit);
    return Value{result, boolType()};
  };
  auto boolConstant = [&](mlir::Location at, bool value) -> Value {
    return boolObject(at, boolBit(at, value));
  };

  auto emitSingletonIdentity = [&](const parser::Node &lhsExpr,
                                   const parser::Node &rhsExpr,
                                   llvm::StringRef op) -> Value {
    std::optional<int> lhsKey = singletonKey(lhsExpr);
    std::optional<int> rhsKey = singletonKey(rhsExpr);
    bool negate = op == "is not";
    if (lhsKey && rhsKey)
      return boolConstant(loc(expr), (*lhsKey == *rhsKey) != negate);

    if (!lhsKey && !rhsKey) {
      error(expr, "identity comparison requires a statically known singleton "
                  "operand");
      return Value{{}, boolType()};
    }

    int key = lhsKey ? *lhsKey : *rhsKey;
    const parser::Node &valueExpr = lhsKey ? rhsExpr : lhsExpr;
    std::optional<mlir::Type> valueType = inferExpressionType(valueExpr);
    if (!valueType) {
      error(valueExpr,
            "identity comparison requires a statically known operand type");
      return Value{{}, boolType()};
    }

    mlir::Value resultBit;
    if (*valueType == boolType()) {
      Value value = emitExpression(valueExpr);
      if (!value.value)
        return Value{{}, boolType()};
      resultBit = builder.create<py::CastToPrimOp>(loc(valueExpr), i1Type(),
                                                   value.value, "exact");
      if (key == 2)
        resultBit = invertBit(loc(valueExpr), resultBit);
      else if (key != 1)
        resultBit = boolBit(loc(valueExpr), false);
    } else if (*valueType == noneType()) {
      Value value = emitExpression(valueExpr);
      if (!value.value)
        return Value{{}, boolType()};
      resultBit = boolBit(loc(valueExpr), key == 0);
    } else if (auto unionTy = mlir::dyn_cast<py::UnionType>(*valueType);
               unionTy && key == 0 && unionTy.hasMember(noneType())) {
      Value value = emitExpression(valueExpr);
      if (!value.value)
        return Value{{}, boolType()};
      resultBit =
          builder.create<py::UnionTestOp>(loc(valueExpr), i1Type(), value.value,
                                          mlir::TypeAttr::get(noneType()));
    } else {
      if (valueExpr.kind != "Name" && valueExpr.kind != "Constant") {
        error(valueExpr,
              "identity comparison against a singleton for this expression "
              "type is not side-effect-safe yet");
        return Value{{}, boolType()};
      }
      resultBit = boolBit(loc(valueExpr), false);
    }

    if (negate)
      resultBit = invertBit(loc(expr), resultBit);
    return boolObject(loc(expr), resultBit);
  };

  auto emitPair = [&](const parser::Node &anchor, llvm::StringRef op,
                      const Value &lhs, const Value &rhs) -> Value {
    if (!lhs.value || !rhs.value)
      return Value{{}, i1Type()};
    if (lhs.type != rhs.type) {
      error(anchor, "C++ emitter cannot compare mixed static types yet");
      return Value{{}, i1Type()};
    }

    if (mlir::isa<mlir::IntegerType>(lhs.type)) {
      std::optional<mlir::arith::CmpIPredicate> predicate = intCmpPredicate(op);
      if (!predicate) {
        error(anchor,
              "unsupported integer comparison operator '" + op.str() + "'");
        return Value{{}, i1Type()};
      }
      mlir::Value result = builder.create<mlir::arith::CmpIOp>(
          loc(anchor), *predicate, lhs.value, rhs.value);
      return Value{result, i1Type()};
    }

    if (mlir::isa<mlir::FloatType>(lhs.type)) {
      std::optional<mlir::arith::CmpFPredicate> predicate =
          floatCmpPredicate(op);
      if (!predicate) {
        error(anchor,
              "unsupported float comparison operator '" + op.str() + "'");
        return Value{{}, i1Type()};
      }
      mlir::Value result = builder.create<mlir::arith::CmpFOp>(
          loc(anchor), *predicate, lhs.value, rhs.value);
      return Value{result, i1Type()};
    }

    mlir::Value result;
    if (op == "==")
      result = builder.create<py::EqOp>(loc(anchor), boolType(), lhs.value,
                                        rhs.value);
    else if (op == "!=")
      result = builder.create<py::NeOp>(loc(anchor), boolType(), lhs.value,
                                        rhs.value);
    else if (op == "<")
      result = builder.create<py::LtOp>(loc(anchor), boolType(), lhs.value,
                                        rhs.value);
    else if (op == "<=")
      result = builder.create<py::LeOp>(loc(anchor), boolType(), lhs.value,
                                        rhs.value);
    else if (op == ">")
      result = builder.create<py::GtOp>(loc(anchor), boolType(), lhs.value,
                                        rhs.value);
    else if (op == ">=")
      result = builder.create<py::GeOp>(loc(anchor), boolType(), lhs.value,
                                        rhs.value);
    else {
      error(anchor, "unsupported comparison operator '" + op.str() + "'");
      return Value{{}, boolType()};
    }
    return Value{result, boolType()};
  };

  if (ops->size() == 1) {
    const parser::NodePtr &rhsNode = comparators->front();
    if (!rhsNode) {
      error(expr, "Compare.comparators contains an empty comparator");
      return Value{{}, boolType()};
    }
    if (ops->front() == "is" || ops->front() == "is not")
      return emitSingletonIdentity(**lhsNode, *rhsNode, ops->front());
    if ((ops->front() == "in" || ops->front() == "not in") &&
        rhsNode->kind == "Tuple")
      return emitTupleLiteralMembership(expr, **lhsNode, *rhsNode,
                                        ops->front() == "not in");
    if (ops->front() == "in" || ops->front() == "not in") {
      std::optional<mlir::Type> rhsStaticType = inferExpressionType(*rhsNode);
      std::optional<Value> concreteProtocolRhs;
      std::optional<Value> emittedProtocolRhs;
      if (rhsStaticType && mlir::isa<py::ProtocolType>(*rhsStaticType)) {
        Value rhs = emitExpression(*rhsNode);
        if (!rhs.value)
          return Value{{}, i1Type()};
        emittedProtocolRhs = rhs;
        if (std::optional<Value> concrete = concreteProtocolValue(rhs)) {
          concreteProtocolRhs = *concrete;
          rhsStaticType = concrete->type;
        }
      }
      auto emitRhs = [&]() -> Value {
        if (concreteProtocolRhs)
          return *concreteProtocolRhs;
        if (emittedProtocolRhs)
          return *emittedProtocolRhs;
        return emitExpression(*rhsNode);
      };
      if (mlir::Type elementType =
              rhsStaticType ? listElementType(*rhsStaticType) : mlir::Type{}) {
        Value rhs = emitRhs();
        if (!rhs.value)
          return Value{{}, i1Type()};
        Value lhs = emitExpression(**lhsNode);
        if (!lhs.value)
          return Value{{}, i1Type()};
        std::optional<protocols::ProtocolMethod> containsContract =
            resolveProtocolMethodContract(expr, rhs.type, "__contains__",
                                          {lhs.type}, "list membership");
        if (!containsContract)
          return Value{{}, i1Type()};
        llvm::ArrayRef<mlir::Type> containsResults =
            containsContract->signature.getResultTypes();
        if (containsResults.size() != 1 ||
            containsResults.front() != boolType()) {
          error(expr, "list membership does not satisfy Sequence.__contains__");
          return Value{{}, i1Type()};
        }
        mlir::Value bit = builder.create<py::ContainsOp>(
            loc(expr), i1Type(), containsContract->signature, rhs.value,
            lhs.value);
        if (ops->front() == "not in")
          bit = invertBit(loc(expr), bit);
        return Value{bit, i1Type()};
      }
      if (auto tupleType = rhsStaticType
                               ? mlir::dyn_cast<py::TupleType>(*rhsStaticType)
                               : py::TupleType{}) {
        llvm::ArrayRef<mlir::Type> elementTypes = tupleType.getElementTypes();
        Value rhs = emitRhs();
        if (!rhs.value)
          return Value{{}, i1Type()};
        if (elementTypes.empty()) {
          mlir::Value bit = boolBit(loc(expr), ops->front() == "not in");
          return Value{bit, i1Type()};
        }
        Value lhs = emitExpression(**lhsNode);
        if (!lhs.value)
          return Value{{}, i1Type()};
        std::optional<protocols::ProtocolMethod> containsContract =
            resolveProtocolMethodContract(expr, rhs.type, "__contains__",
                                          {lhs.type}, "tuple membership");
        if (!containsContract)
          return Value{{}, i1Type()};
        llvm::ArrayRef<mlir::Type> containsResults =
            containsContract->signature.getResultTypes();
        if (containsResults.size() != 1 ||
            containsResults.front() != boolType()) {
          error(expr,
                "tuple membership does not satisfy Sequence.__contains__");
          return Value{{}, i1Type()};
        }

        mlir::Value bit = builder.create<py::ContainsOp>(
            loc(expr), i1Type(), containsContract->signature, rhs.value,
            lhs.value);
        if (ops->front() == "not in")
          bit = invertBit(loc(expr), bit);
        return Value{bit, i1Type()};
      }
      if (std::optional<std::pair<mlir::Type, mlir::Type>> dictTypes =
              rhsStaticType ? dictKeyValueTypes(*rhsStaticType)
                            : std::nullopt) {
        Value rhs = emitRhs();
        if (!rhs.value)
          return Value{{}, i1Type()};
        Value lhs = emitExpression(**lhsNode);
        if (!lhs.value)
          return Value{{}, i1Type()};
        std::optional<protocols::ProtocolMethod> containsContract =
            resolveProtocolMethodContract(expr, rhs.type, "__contains__",
                                          {lhs.type}, "dict membership");
        if (!containsContract)
          return Value{{}, i1Type()};
        llvm::ArrayRef<mlir::Type> containsResults =
            containsContract->signature.getResultTypes();
        if (containsResults.size() != 1 ||
            containsResults.front() != boolType()) {
          error(expr, "dict membership does not satisfy Mapping.__contains__");
          return Value{{}, i1Type()};
        }
        if (lhs.type != dictTypes->first) {
          if (exactScalarEqualityAlwaysFalse(lhs.type, dictTypes->first)) {
            mlir::Value bit = boolBit(loc(expr), ops->front() == "not in");
            return Value{bit, i1Type()};
          }
          error(expr, "dict membership resolves " + typeString(rhs.type) +
                          ".__contains__(" + typeString(lhs.type) +
                          ") -> bool, but dictionary lookup lowering for item "
                          "type " +
                          typeString(lhs.type) + " and key type " +
                          typeString(dictTypes->first) +
                          " is not implemented yet");
          return Value{{}, i1Type()};
        }
        mlir::Value bit = builder.create<py::ContainsOp>(
            loc(expr), i1Type(), containsContract->signature, rhs.value,
            lhs.value);
        if (ops->front() == "not in")
          bit = invertBit(loc(expr), bit);
        return Value{bit, i1Type()};
      }
      if (auto classType = rhsStaticType
                               ? mlir::dyn_cast<py::ClassType>(*rhsStaticType)
                               : py::ClassType{}) {
        Value rhs = emitRhs();
        if (!rhs.value)
          return Value{{}, i1Type()};
        Value lhs = emitExpression(**lhsNode);
        if (!lhs.value)
          return Value{{}, i1Type()};

        std::optional<py::ProtocolType> container =
            protocolType("Container", {lhs.type});
        if (!container) {
          error(expr, "unknown Container protocol");
          return Value{{}, i1Type()};
        }
        if (!classConformsToProtocol(classType, *container)) {
          error(expr, "class receiver " + typeString(rhs.type) +
                          " does not satisfy Container.__contains__(" +
                          typeString(lhs.type) + ") -> bool");
          return Value{{}, i1Type()};
        }

        std::optional<FunctionInfo> method =
            resolveClassMethod(expr, rhs, "__contains__");
        if (!method)
          return Value{{}, i1Type()};
        if (method->resultType != boolType()) {
          error(expr, "class __contains__ on " + typeString(rhs.type) +
                          " must return !py.bool, got " +
                          typeString(method->resultType));
          return Value{{}, i1Type()};
        }

        Value result = emitResolvedMethodCall(expr, rhs, *method, {lhs});
        if (!result.value)
          return Value{{}, i1Type()};
        mlir::Value bit = builder.create<py::CastToPrimOp>(
            loc(expr), i1Type(), result.value, "exact");
        if (ops->front() == "not in")
          bit = invertBit(loc(expr), bit);
        return Value{bit, i1Type()};
      }
      if (rhsStaticType && mlir::isa<py::ProtocolType>(*rhsStaticType)) {
        std::optional<mlir::Type> lhsStaticType =
            inferExpressionType(**lhsNode);
        if (!lhsStaticType) {
          error(**lhsNode,
                "protocol membership requires a statically known item type");
          return Value{{}, i1Type()};
        }
        std::optional<mlir::Type> containsResult = resolveProtocolMethodResult(
            expr, *rhsStaticType, "__contains__", {*lhsStaticType},
            "protocol membership");
        if (!containsResult)
          return Value{{}, i1Type()};
        if (*containsResult != boolType()) {
          error(expr, "protocol membership __contains__ result must be bool, "
                      "got " +
                          typeString(*containsResult));
          return Value{{}, i1Type()};
        }
        error(expr, "protocol membership on " + typeString(*rhsStaticType) +
                        " resolves statically to " + typeString(boolType()) +
                        ", but lowering for protocol-typed receivers is not "
                        "implemented yet");
        return Value{{}, i1Type()};
      }
    }
  }

  Value lhs = emitExpression(**lhsNode);
  if (!lhs.value)
    return Value{{}, boolType()};

  if (ops->size() == 1) {
    const parser::NodePtr &rhsNode = comparators->front();
    Value rhs = emitExpressionWithExpectedType(*rhsNode, lhs.type);
    if (rhs.value && lhs.type != rhs.type) {
      Value coercedLhs = emitExpressionWithExpectedType(**lhsNode, rhs.type);
      if (coercedLhs.value && coercedLhs.type == rhs.type)
        lhs = coercedLhs;
    }
    return emitPair(expr, ops->front(), lhs, rhs);
  }

  if (!mlir::isa<mlir::IntegerType, mlir::FloatType>(lhs.type)) {
    error(expr, "chained comparisons currently support only "
                "ownership-neutral primitive operands");
    return Value{{}, i1Type()};
  }

  std::function<mlir::Value(std::size_t, Value)> emitChain =
      [&](std::size_t index, Value currentLhs) -> mlir::Value {
    const parser::NodePtr &rhsNode = (*comparators)[index];
    if (!rhsNode) {
      error(expr, "Compare.comparators contains an empty comparator");
      return falseBit(loc(expr));
    }
    Value rhs = emitExpressionWithExpectedType(*rhsNode, currentLhs.type);
    Value compared = emitPair(*rhsNode, (*ops)[index], currentLhs, rhs);
    if (!compared.value || compared.type != i1Type())
      return falseBit(loc(*rhsNode));
    if (index + 1 == ops->size())
      return compared.value;

    auto ifOp = builder.create<mlir::scf::IfOp>(
        loc(*rhsNode), mlir::TypeRange{i1Type()}, compared.value,
        /*withElseRegion=*/true);
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(ifOp.thenBlock());
      mlir::Value rest =
          rhs.value ? emitChain(index + 1, rhs) : falseBit(loc(*rhsNode));
      builder.create<mlir::scf::YieldOp>(loc(*rhsNode), rest);
    }
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(ifOp.elseBlock());
      builder.create<mlir::scf::YieldOp>(loc(*rhsNode),
                                         falseBit(loc(*rhsNode)));
    }
    builder.setInsertionPointAfter(ifOp);
    return ifOp.getResult(0);
  };

  return Value{emitChain(0, lhs), i1Type()};
}

Value Builder::Impl::emitUnaryOp(const parser::Node &expr) {
  std::optional<std::string> op = symbolField(expr, "op");
  const parser::NodePtr *operandNode = nodeField(expr, "operand");
  if (!op || !operandNode || !*operandNode) {
    error(expr, "UnaryOp.op or UnaryOp.operand is missing");
    return Value{{}, noneType()};
  }

  Value operand = emitExpression(**operandNode);
  if (!operand.value)
    return Value{{}, noneType()};

  if (*op == "+")
    return operand;

  if (*op == "-") {
    if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(operand.type)) {
      mlir::Value zero = builder.create<mlir::arith::ConstantIntOp>(
          loc(expr), 0, intTy.getWidth());
      mlir::Value result =
          builder.create<mlir::arith::SubIOp>(loc(expr), zero, operand.value);
      return Value{result, operand.type};
    }
    if (auto floatTy = mlir::dyn_cast<mlir::FloatType>(operand.type)) {
      mlir::Value zero = builder.create<mlir::arith::ConstantOp>(
          loc(expr), operand.type, builder.getFloatAttr(floatTy, 0.0));
      mlir::Value result =
          builder.create<mlir::arith::SubFOp>(loc(expr), zero, operand.value);
      return Value{result, operand.type};
    }
    if (operand.type == intType()) {
      mlir::Value zero =
          builder.create<py::IntConstantOp>(loc(expr), intType(), "0");
      mlir::Value result =
          builder.create<py::SubOp>(loc(expr), intType(), zero, operand.value);
      return Value{result, intType()};
    }
    if (operand.type == floatType()) {
      mlir::Value zero = builder.create<py::FloatConstantOp>(
          loc(expr), floatType(), builder.getF64FloatAttr(0.0));
      mlir::Value result = builder.create<py::SubOp>(loc(expr), floatType(),
                                                     zero, operand.value);
      return Value{result, floatType()};
    }
    error(expr, "unary minus is unsupported for " + typeString(operand.type));
    return Value{{}, operand.type};
  }

  if (*op == "not") {
    if (operand.type == boolType()) {
      mlir::Value prim = builder.create<py::CastToPrimOp>(
          loc(expr), i1Type(), operand.value, "exact");
      mlir::Value zero =
          builder.create<mlir::arith::ConstantIntOp>(loc(expr), 0, 1);
      mlir::Value inverted = builder.create<mlir::arith::CmpIOp>(
          loc(expr), mlir::arith::CmpIPredicate::eq, prim, zero);
      mlir::Value result =
          builder.create<py::CastFromPrimOp>(loc(expr), boolType(), inverted);
      return Value{result, boolType()};
    }
    if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(operand.type)) {
      mlir::Value zero = builder.create<mlir::arith::ConstantIntOp>(
          loc(expr), 0, intTy.getWidth());
      mlir::Value result = builder.create<mlir::arith::CmpIOp>(
          loc(expr), mlir::arith::CmpIPredicate::eq, operand.value, zero);
      return Value{result, i1Type()};
    }
    if (auto floatTy = mlir::dyn_cast<mlir::FloatType>(operand.type)) {
      mlir::Value zero = builder.create<mlir::arith::ConstantOp>(
          loc(expr), operand.type, builder.getFloatAttr(floatTy, 0.0));
      mlir::Value result = builder.create<mlir::arith::CmpFOp>(
          loc(expr), mlir::arith::CmpFPredicate::OEQ, operand.value, zero);
      return Value{result, i1Type()};
    }
    if (operand.type == intType()) {
      mlir::Value zero =
          builder.create<py::IntConstantOp>(loc(expr), intType(), "0");
      mlir::Value result =
          builder.create<py::EqOp>(loc(expr), boolType(), operand.value, zero);
      return Value{result, boolType()};
    }
    if (operand.type == floatType()) {
      mlir::Value zero = builder.create<py::FloatConstantOp>(
          loc(expr), floatType(), builder.getF64FloatAttr(0.0));
      mlir::Value result =
          builder.create<py::EqOp>(loc(expr), boolType(), operand.value, zero);
      return Value{result, boolType()};
    }
    if (operand.type == strType()) {
      mlir::Value empty =
          builder.create<py::StrConstantOp>(loc(expr), strType(), "");
      mlir::Value result =
          builder.create<py::EqOp>(loc(expr), boolType(), operand.value, empty);
      return Value{result, boolType()};
    }
    if (std::optional<std::size_t> arity =
            staticTupleArity(operand.type, operand.value)) {
      mlir::Value prim = emitBoolConstant(builder, loc(expr), *arity == 0);
      mlir::Value result =
          builder.create<py::CastFromPrimOp>(loc(expr), boolType(), prim);
      return Value{result, boolType()};
    }
    if (listElementType(operand.type)) {
      auto found = finiteListElements.find(operand.value);
      if (found == finiteListElements.end()) {
        error(expr, "logical not on list requires statically known elements");
        return Value{{}, boolType()};
      }
      mlir::Value prim =
          emitBoolConstant(builder, loc(expr), found->second.empty());
      mlir::Value result =
          builder.create<py::CastFromPrimOp>(loc(expr), boolType(), prim);
      return Value{result, boolType()};
    }
    if (operand.type == noneType()) {
      mlir::Value prim = emitBoolConstant(builder, loc(expr), true);
      mlir::Value result =
          builder.create<py::CastFromPrimOp>(loc(expr), boolType(), prim);
      return Value{result, boolType()};
    }
    error(expr, "logical not is unsupported for " + typeString(operand.type));
    return Value{{}, boolType()};
  }

  if (*op == "~") {
    if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(operand.type)) {
      mlir::Value allOnes = builder.create<mlir::arith::ConstantIntOp>(
          loc(expr), -1, intTy.getWidth());
      mlir::Value result = builder.create<mlir::arith::XOrIOp>(
          loc(expr), operand.value, allOnes);
      return Value{result, operand.type};
    }
    if (operand.type == intType()) {
      mlir::Value minusOne =
          builder.create<py::IntConstantOp>(loc(expr), intType(), "-1");
      mlir::Value result = builder.create<py::SubOp>(loc(expr), intType(),
                                                     minusOne, operand.value);
      return Value{result, intType()};
    }
    error(expr,
          "bitwise invert is unsupported for " + typeString(operand.type));
    return Value{{}, operand.type};
  }

  error(expr, "unsupported unary operator '" + *op + "'");
  return Value{{}, operand.type};
}

} // namespace lython::emitter
