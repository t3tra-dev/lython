#include "EmitterCore.h"
#include "EmitterOps.h" // IWYU pragma: keep
#include "EmitterPyOps.h"
#include "EmitterSupport.h"

#include "AstAccess.h"
#include "PyProtocols.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <string>

namespace lython::emitter {

Value ModuleEmitter::emitExpr(const parser::Node *expr) {
  if (!expr)
    return {py::NoneOp::create(builder, builder.getUnknownLoc(), types.none())
                .getResult(),
            types.none()};
  if (expr->kind == "Constant")
    return emitConstant(*expr);
  if (expr->kind == "Name") {
    llvm::StringRef name = ast::nameSpelling(*expr);
    auto found = values.find(name);
    if (found != values.end())
      return found->second;
    auto primitiveConstant = primitiveConstants.find(name);
    if (primitiveConstant != primitiveConstants.end())
      return emitPrimitiveConstant(*expr, primitiveConstant->second);
    mlir::Type type = types.lookupSymbol(name).value_or(types.object());
    if (auto cls = types.lookupClass(name)) {
      mlir::Type typeType = types.typeObject(*cls);
      auto op = py::TypeObjectOp::create(builder, loc(*expr), typeType, *cls);
      return {op.getResult(), typeType};
    }
    std::string binding = std::string(name);
    if (std::optional<std::string> canonical =
            types.lookupCanonicalBinding(name))
      binding = *canonical;
    return emitBindingRef(*expr, binding, type);
  }
  if (expr->kind == "Call")
    return emitCall(*expr);
  if (expr->kind == "UnaryOp")
    return emitUnary(*expr);
  if (expr->kind == "BinOp")
    return emitBinary(*expr);
  if (expr->kind == "Compare")
    return emitCompare(*expr);
  if (expr->kind == "Subscript")
    return emitSubscript(*expr);
  if (expr->kind == "Attribute") {
    std::string qualified = ast::qualifiedName(expr);
    if (!qualified.empty())
      if (auto cls = types.lookupClass(qualified)) {
        mlir::Type typeType = types.typeObject(*cls);
        auto op = py::TypeObjectOp::create(builder, loc(*expr), typeType, *cls);
        return {op.getResult(), typeType};
      }
    if (!qualified.empty())
      if (auto symbol = types.lookupSymbol(qualified)) {
        std::string binding = qualified;
        if (std::optional<std::string> canonical =
                types.lookupCanonicalBinding(qualified))
          binding = *canonical;
        return emitBindingRef(*expr, binding, *symbol);
      }
    return emitAttribute(*expr);
  }
  if (expr->kind == "Await")
    return emitAwait(*expr);
  if (expr->kind == "List" || expr->kind == "Tuple" || expr->kind == "Dict")
    return emitContainerLiteral(*expr);
  if (expr->kind == "IfExp") {
    const parser::Node *bodyNode = ast::node(*expr, "body");
    const parser::Node *elseNode = ast::node(*expr, "orelse");
    mlir::Type resultType =
        types.join({types.inferExpr(bodyNode), types.inferExpr(elseNode)});
    mlir::Value condition =
        emitBoolValue(emitExpr(ast::node(*expr, "test")), *expr);
    auto ifOp = mlir::scf::IfOp::create(
        builder, loc(*expr), mlir::TypeRange{resultType}, condition, true);

    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
      Value body = coerceValue(emitExpr(bodyNode), resultType, *expr);
      mlir::scf::YieldOp::create(builder, loc(*expr), body.value);
    }
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
      Value other = coerceValue(emitExpr(elseNode), resultType, *expr);
      mlir::scf::YieldOp::create(builder, loc(*expr), other.value);
    }

    builder.setInsertionPointAfter(ifOp);
    return {ifOp.getResult(0), resultType};
  }
  if (expr->kind == "Lambda")
    return emitLambda(*expr);
  diagnostics.push_back(
      parser::Diagnostic{parser::Severity::Error, expr->range.start,
                         "unsupported expression kind '" + expr->kind + "'"});
  return emitNone(*expr);
}

Value ModuleEmitter::emitConstant(const parser::Node &expr) {
  if (ast::isNoneField(expr, "value")) {
    auto op = py::NoneOp::create(builder, loc(expr), types.none());
    return {op.getResult(), types.none()};
  }
  if (auto value = ast::boolean(expr, "value")) {
    mlir::Type type = types.literal(*value ? "True" : "False");
    auto op = py::BoolConstantOp::create(builder, loc(expr), type,
                                         builder.getBoolAttr(*value));
    return {op.getResult(), type};
  }
  if (auto value = ast::integer(expr, "value")) {
    std::string text = std::to_string(*value);
    mlir::Type type = types.literal(text);
    auto op = py::IntConstantOp::create(builder, loc(expr), type,
                                        builder.getStringAttr(text));
    return {op.getResult(), type};
  }
  if (auto value = ast::floating(expr, "value")) {
    auto op = py::FloatConstantOp::create(builder, loc(expr), types.floatType(),
                                          builder.getF64FloatAttr(*value));
    return {op.getResult(), types.floatType()};
  }
  if (auto value = ast::string(expr, "value")) {
    mlir::Type type = types.literal("\"" + std::string(*value) + "\"");
    auto op = py::StrConstantOp::create(builder, loc(expr), type,
                                        builder.getStringAttr(*value));
    return {op.getResult(), type};
  }
  if (const auto *fieldValue = ast::field(expr, "value")) {
    if (const auto *big = std::get_if<parser::BigInteger>(fieldValue)) {
      mlir::Type type = types.literal(big->decimal);
      auto op = py::IntConstantOp::create(builder, loc(expr), type,
                                          builder.getStringAttr(big->decimal));
      return {op.getResult(), type};
    }
  }
  diagnostics.push_back(parser::Diagnostic{parser::Severity::Error,
                                           expr.range.start,
                                           "unsupported constant literal"});
  return emitNone(expr);
}

Value ModuleEmitter::emitUnary(const parser::Node &expr) {
  const parser::Node *op = ast::node(expr, "op");
  const parser::Node *operandNode = ast::node(expr, "operand");

  if (ast::isOperator(op, "USub") && operandNode &&
      operandNode->kind == "Constant") {
    if (auto value = ast::integer(*operandNode, "value")) {
      std::string text = "-" + std::to_string(*value);
      mlir::Type type = types.literal(text);
      auto constOp = py::IntConstantOp::create(builder, loc(expr), type,
                                               builder.getStringAttr(text));
      return {constOp.getResult(), type};
    }
    if (auto value = ast::floating(*operandNode, "value")) {
      auto constOp =
          py::FloatConstantOp::create(builder, loc(expr), types.floatType(),
                                      builder.getF64FloatAttr(-*value));
      return {constOp.getResult(), types.floatType()};
    }
    if (const auto *fieldValue = ast::field(*operandNode, "value")) {
      if (const auto *big = std::get_if<parser::BigInteger>(fieldValue)) {
        std::string text = "-" + big->decimal;
        mlir::Type type = types.literal(text);
        auto constOp = py::IntConstantOp::create(builder, loc(expr), type,
                                                 builder.getStringAttr(text));
        return {constOp.getResult(), type};
      }
    }
  }

  Value operand = emitExpr(operandNode);
  mlir::Type result = types.widenLiteral(types.inferExpr(&expr));
  if (ast::isOperator(op, "USub"))
    return emitUnarySpecial<py::NegOp>(expr, "__neg__", operand, result);
  if (ast::isOperator(op, "UAdd"))
    return emitUnarySpecial<py::PosOp>(expr, "__pos__", operand, result);
  if (ast::isOperator(op, "Invert"))
    return emitUnarySpecial<py::InvertOp>(expr, "__invert__", operand, result);
  if (ast::isOperator(op, "Not")) {
    mlir::Value truth = emitBoolValue(operand, expr);
    auto one = mlir::arith::ConstantIntOp::create(builder, loc(expr), 1, 1);
    auto inverted = mlir::arith::XOrIOp::create(builder, loc(expr), truth, one);
    auto pyBool = py::CastFromPrimOp::create(
        builder, loc(expr), types.boolType(), inverted.getResult());
    return {pyBool.getResult(), types.boolType()};
  }
  diagnostics.push_back(parser::Diagnostic{
      parser::Severity::Error, expr.range.start, "unsupported unary operator"});
  return emitNone(expr);
}

Value ModuleEmitter::emitBinary(const parser::Node &expr) {
  Value lhs = emitExpr(ast::node(expr, "left"));
  Value rhs = emitExpr(ast::node(expr, "right"));
  const parser::Node *op = ast::node(expr, "op");
  if (std::optional<Value> primitive = emitPrimitiveBinary(expr, lhs, rhs, op))
    return *primitive;
  mlir::Type left = types.widenLiteral(lhs.type);
  mlir::Type right = types.widenLiteral(rhs.type);
  mlir::Type result = types.join({left, right});
  if (left == types.strType() && right == types.strType()) {
    result = types.strType();
  } else if (ast::isOperator(op, "Div") &&
             (left == types.intType() || left == types.floatType()) &&
             (right == types.intType() || right == types.floatType())) {
    result = types.floatType();
  } else if (left == types.floatType() || right == types.floatType()) {
    result = types.floatType();
  } else if (left == types.intType() && right == types.intType()) {
    result = types.intType();
  }
  if (ast::isOperator(op, "Sub"))
    return emitBinarySpecial<py::SubOp>(expr, "__sub__", lhs, rhs, result);
  if (ast::isOperator(op, "Mult"))
    return emitBinarySpecial<py::MulOp>(expr, "__mul__", lhs, rhs, result);
  if (ast::isOperator(op, "Div"))
    return emitBinarySpecial<py::DivOp>(expr, "__truediv__", lhs, rhs, result);
  if (ast::isOperator(op, "FloorDiv"))
    return emitBinarySpecial<py::FloorDivOp>(expr, "__floordiv__", lhs, rhs,
                                             result);
  if (ast::isOperator(op, "Mod"))
    return emitBinarySpecial<py::ModOp>(expr, "__mod__", lhs, rhs, result);
  if (ast::isOperator(op, "LShift"))
    return emitBinarySpecial<py::LShiftOp>(expr, "__lshift__", lhs, rhs,
                                           result);
  if (ast::isOperator(op, "RShift"))
    return emitBinarySpecial<py::RShiftOp>(expr, "__rshift__", lhs, rhs,
                                           result);
  if (ast::isOperator(op, "BitAnd"))
    return emitBinarySpecial<py::BitAndOp>(expr, "__and__", lhs, rhs, result);
  if (ast::isOperator(op, "BitOr"))
    return emitBinarySpecial<py::BitOrOp>(expr, "__or__", lhs, rhs, result);
  if (ast::isOperator(op, "BitXor"))
    return emitBinarySpecial<py::BitXorOp>(expr, "__xor__", lhs, rhs, result);
  return emitBinarySpecial<py::AddOp>(expr, "__add__", lhs, rhs, result);
}

Value ModuleEmitter::emitCompare(const parser::Node &expr) {
  Value lhs = emitExpr(ast::node(expr, "left"));
  const auto *comparators = ast::nodeList(expr, "comparators");
  const auto *ops = ast::nodeList(expr, "ops");
  if (!comparators || comparators->empty()) {
    auto op = py::BoolConstantOp::create(
        builder, loc(expr), types.literal("False"), builder.getBoolAttr(false));
    return {op.getResult(), types.literal("False")};
  }
  Value rhs = emitExpr(comparators->front().get());
  const parser::Node *op = ops && !ops->empty() ? ops->front().get() : nullptr;
  if (std::optional<Value> primitive = emitPrimitiveCompare(expr, lhs, rhs, op))
    return *primitive;
  auto emitNoneIdentityTest = [&](Value candidate,
                                  Value other) -> std::optional<Value> {
    auto unionType = mlir::dyn_cast_if_present<py::UnionType>(candidate.type);
    if (!unionType || !unionType.hasMember(types.none()) ||
        !isNoneTypeLike(other.type))
      return std::nullopt;

    auto test = py::UnionTestOp::create(builder, loc(expr), builder.getI1Type(),
                                        candidate.value,
                                        mlir::TypeAttr::get(types.none()));
    mlir::Value bit = test.getResult();
    if (ast::isOperator(op, "IsNot")) {
      auto one = mlir::arith::ConstantIntOp::create(builder, loc(expr), 1, 1);
      bit = mlir::arith::XOrIOp::create(builder, loc(expr), bit, one);
    }
    auto pyBool =
        py::CastFromPrimOp::create(builder, loc(expr), types.boolType(), bit);
    return Value{pyBool.getResult(), types.boolType()};
  };
  if (ast::isOperator(op, "Is") || ast::isOperator(op, "IsNot")) {
    if (auto narrowed = emitNoneIdentityTest(lhs, rhs))
      return *narrowed;
    if (auto narrowed = emitNoneIdentityTest(rhs, lhs))
      return *narrowed;
  }
  if (ast::isOperator(op, "In") || ast::isOperator(op, "NotIn")) {
    CallInferenceResult inference =
        types.inferMethodCallWithEvidence(rhs.type, "__contains__", {lhs.type});
    auto contains = py::ContainsOp::create(
        builder, loc(expr), builder.getI1Type(),
        mlir::FlatSymbolRefAttr::get(&context, "__contains__"),
        callProtocolFor(inference), rhs.value, lhs.value);
    mlir::Value bit = contains.getResult();
    if (ast::isOperator(op, "NotIn")) {
      auto one = mlir::arith::ConstantIntOp::create(builder, loc(expr), 1, 1);
      bit = mlir::arith::XOrIOp::create(builder, loc(expr), bit, one);
    }
    auto pyBool =
        py::CastFromPrimOp::create(builder, loc(expr), types.boolType(), bit);
    return Value{pyBool.getResult(), types.boolType()};
  }
  if (ast::isOperator(op, "NotEq") || ast::isOperator(op, "IsNot"))
    return emitBinarySpecial<py::NeOp>(expr, "__ne__", lhs, rhs,
                                       types.boolType());
  if (ast::isOperator(op, "Lt"))
    return emitBinarySpecial<py::LtOp>(expr, "__lt__", lhs, rhs,
                                       types.boolType());
  if (ast::isOperator(op, "LtE"))
    return emitBinarySpecial<py::LeOp>(expr, "__le__", lhs, rhs,
                                       types.boolType());
  if (ast::isOperator(op, "Gt"))
    return emitBinarySpecial<py::GtOp>(expr, "__gt__", lhs, rhs,
                                       types.boolType());
  if (ast::isOperator(op, "GtE"))
    return emitBinarySpecial<py::GeOp>(expr, "__ge__", lhs, rhs,
                                       types.boolType());
  return emitBinarySpecial<py::EqOp>(expr, "__eq__", lhs, rhs,
                                     types.boolType());
}

Value ModuleEmitter::emitSubscript(const parser::Node &expr) {
  Value container = emitExpr(ast::node(expr, "value"));
  Value index = emitExpr(ast::node(expr, "slice"));
  CallInferenceResult inference = types.inferMethodCallWithEvidence(
      container.type, "__getitem__", {index.type});
  mlir::Type result = inference ? inference.resultType : types.inferExpr(&expr);
  auto op = py::GetItemOp::create(
      builder, loc(expr), result,
      mlir::FlatSymbolRefAttr::get(&context, "__getitem__"),
      callProtocolFor(inference), container.value, index.value);
  return {op.getResult(), result};
}

Value ModuleEmitter::emitMethodObject(const parser::Node &anchor, Value object,
                                      const MethodBinding &methodBinding) {
  if (methodBinding.symbolName.empty())
    return emitNone(anchor);

  bool bindReceiver = methodBindingBindsReceiver(methodBinding);
  if (methodBinding.kind == "instance" && mlir::isa<py::TypeType>(object.type))
    bindReceiver = false;
  if (!bindReceiver)
    return emitFunctionObject(anchor, methodBinding.symbolName,
                              methodBinding.signature.callable, {});

  if (!methodBinding.method ||
      methodBinding.signature.positionalTypes.empty() ||
      methodBinding.signature.positionalNames.empty() ||
      methodBinding.bodySignature.positionalTypes.empty() ||
      methodBinding.bodySignature.positionalNames.empty())
    return emitNone(anchor);

  auto bindSignature = [&](FunctionSignature sig) {
    sig.positionalTypes.erase(sig.positionalTypes.begin());
    sig.positionalNames.erase(sig.positionalNames.begin());
    if (!sig.positionalDefaults.empty())
      sig.positionalDefaults.erase(sig.positionalDefaults.begin());
    if (sig.positionalOnlyCount > 0)
      --sig.positionalOnlyCount;
    types.refreshCallable(sig);
    return sig;
  };

  FunctionSignature boundPublicSig = bindSignature(methodBinding.signature);
  FunctionSignature boundBodySig = bindSignature(methodBinding.bodySignature);

  llvm::SmallVector<Capture, 1> captures;
  mlir::Type preboundTypeObject;
  if (methodBinding.kind == "class" || methodBinding.kind == "classmethod") {
    preboundTypeObject = object.type;
    if (auto typeObject = mlir::dyn_cast<py::TypeType>(object.type))
      preboundTypeObject = typeObject.getInstanceType();
  } else {
    Value descriptorReceiver =
        emitDescriptorReceiver(anchor, object, methodBinding);
    captures.push_back(
        Capture{methodBinding.bodySignature.positionalNames.front(),
                descriptorReceiver});
  }
  std::string symbolName = (llvm::Twine(methodBinding.symbolName) + "$bound$" +
                            llvm::Twine(++syntheticFunctionCounter) + "$" +
                            llvm::Twine(anchor.range.start.line) + "_" +
                            llvm::Twine(anchor.range.start.column))
                               .str();
  emitCallableFunction(*methodBinding.method, symbolName, boundBodySig,
                       captures, /*isLambda=*/false,
                       /*positionalNodeOffset=*/1, preboundTypeObject);
  return emitFunctionObject(anchor, symbolName, boundPublicSig.callable,
                            captures);
}

Value ModuleEmitter::emitAttribute(const parser::Node &expr) {
  Value object = emitExpr(ast::node(expr, "value"));
  mlir::Type result = types.inferExpr(&expr);
  auto attr = ast::string(expr, "attr");
  if (!attr)
    return emitNone(expr);
  std::optional<mlir::Type> field = lookupClassField(object.type, *attr);
  if (field)
    result = *field;

  std::optional<mlir::Type> staticAttr =
      lookupClassStaticAttr(object.type, *attr);
  if (staticAttr)
    result = *staticAttr;

  std::optional<MethodBinding> methodBinding =
      lookupClassMethod(object.type, *attr);
  if (methodBinding && !methodBinding->symbolName.empty())
    return emitMethodObject(expr, object, *methodBinding);

  auto op =
      py::AttrGetOp::create(builder, loc(expr), result, object.value, *attr);
  if (field)
    op->setAttr("ly.attr.kind", builder.getStringAttr("field"));
  else if (staticAttr)
    op->setAttr("ly.attr.kind", builder.getStringAttr("static"));
  else if (methodBinding)
    op->setAttr("ly.attr.kind", builder.getStringAttr(methodBinding->kind));
  if (auto typeObject = mlir::dyn_cast_if_present<py::TypeType>(object.type)) {
    if (auto contract = mlir::dyn_cast_if_present<py::ContractType>(
            typeObject.getInstanceType()))
      op->setAttr("ly.attr.owner",
                  builder.getStringAttr(contract.getContractName()));
  } else if (auto contract =
                 mlir::dyn_cast_if_present<py::ContractType>(object.type)) {
    op->setAttr("ly.attr.owner",
                builder.getStringAttr(contract.getContractName()));
  }
  Value value{op.getResult(), result};
  if (methodBinding)
    value.boundMethod = std::make_shared<BoundMethodValue>(
        BoundMethodValue{object, *methodBinding});
  return value;
}

Value ModuleEmitter::emitAwait(const parser::Node &expr) {
  Value awaitable = emitExpr(ast::node(expr, "value"));
  return emitAwaitValue(expr, awaitable);
}

Value ModuleEmitter::emitAwaitValue(const parser::Node &anchor,
                                    Value awaitable) {
  const py::protocols::Table &table = py::protocols::Table::get(context);
  std::optional<py::protocols::AwaitableResolution> resolution =
      table.resolveAwaitableWithEvidence(types.widenLiteral(awaitable.type));
  mlir::Type result = resolution ? resolution->payloadType : types.object();
  mlir::Type awaitContract = callableProtocol();
  if (resolution && resolution->awaitContract)
    awaitContract =
        callProtocolFor(resolution->awaitContract->method.signature);
  if (!resolution) {
    std::string typeText;
    llvm::raw_string_ostream typeStream(typeText);
    typeStream << awaitable.type;
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, anchor.range.start,
        "await expression requires an Awaitable value, got " +
            typeStream.str()});
  }

  auto op = py::AwaitOp::create(builder, loc(anchor), result, awaitContract,
                                awaitable.value);
  return {op.getResult(), result};
}

Value ModuleEmitter::emitContainerLiteral(const parser::Node &expr) {
  llvm::SmallVector<Value, 8> valuesToPack;
  if (const auto *elts = ast::nodeList(expr, "elts"))
    for (const parser::NodePtr &elt : *elts)
      valuesToPack.push_back(emitExpr(elt.get()));
  if (const auto *keys = ast::nodeList(expr, "keys")) {
    const auto *vals = ast::nodeList(expr, "values");
    for (auto [index, key] : llvm::enumerate(*keys)) {
      if (key)
        valuesToPack.push_back(emitExpr(key.get()));
      if (vals && index < vals->size())
        valuesToPack.push_back(emitExpr((*vals)[index].get()));
    }
  }
  mlir::Type resultType = types.inferExpr(&expr);
  llvm::SmallVector<mlir::Value, 8> operands;
  for (Value value : valuesToPack)
    operands.push_back(value.value);
  auto op = py::PackOp::create(builder, loc(expr), resultType, operands);
  return {op.getResult(), resultType};
}

Value ModuleEmitter::emitBindingRef(const parser::Node &anchor,
                                    llvm::StringRef binding, mlir::Type type,
                                    llvm::ArrayRef<Value> captures) {
  mlir::Type resultType = type ? type : types.object();
  llvm::SmallVector<mlir::Value, 4> captureValues;
  for (Value capture : captures)
    captureValues.push_back(capture.value);
  auto op =
      py::BindingRefOp::create(builder, loc(anchor), resultType,
                               builder.getStringAttr(binding), captureValues);
  return {op.getResult(), resultType};
}

Value ModuleEmitter::emitFunctionObject(const parser::Node &anchor,
                                        llvm::StringRef symbolName,
                                        mlir::Type type,
                                        llvm::ArrayRef<Capture> captures) {
  llvm::SmallVector<Value, 4> captureValues;
  for (const Capture &capture : captures)
    captureValues.push_back(capture.value);
  return emitBindingRef(anchor, symbolName, type, captureValues);
}

Value ModuleEmitter::emitNone(const parser::Node &anchor) {
  auto op = py::NoneOp::create(builder, loc(anchor), types.none());
  return {op.getResult(), types.none()};
}

Value ModuleEmitter::emitPack(mlir::ArrayRef<Value> valuesIn,
                              llvm::ArrayRef<char> unpacked) {
  llvm::SmallVector<mlir::Value, 8> operands;
  llvm::SmallVector<mlir::Type, 8> elementTypes;
  for (Value value : valuesIn) {
    operands.push_back(value.value);
    elementTypes.push_back(value.type);
  }
  mlir::Type element =
      elementTypes.empty() ? types.object() : types.join(elementTypes);
  mlir::Type resultType = types.tupleOf(element);
  auto op = py::PackOp::create(builder, builder.getUnknownLoc(), resultType,
                               operands);
  if (!unpacked.empty() && anyTrue(unpacked)) {
    if (unpacked.size() != valuesIn.size()) {
      diagnostics.push_back(
          parser::Diagnostic{parser::Severity::Error,
                             {},
                             "internal pack unpack metadata mismatch"});
    } else {
      op->setAttr(kPackUnpackedOperandsAttr, boolArray(builder, unpacked));
    }
  }
  return {op.getResult(), resultType};
}

Value ModuleEmitter::coerceValue(Value value, mlir::Type targetType,
                                 const parser::Node &anchor) {
  if (!targetType || value.type == targetType)
    return value;
  if (auto unionType = mlir::dyn_cast<py::UnionType>(targetType)) {
    if (unionType.hasMember(value.type)) {
      auto op = py::UnionWrapOp::create(builder, loc(anchor), targetType,
                                        value.value);
      return {op.getResult(), targetType};
    }
  }
  if (mlir::isa<py::ProtocolType>(targetType)) {
    auto op = py::ProtocolViewOp::create(builder, loc(anchor), targetType,
                                         value.value);
    return {op.getResult(), targetType};
  }
  if (mlir::isa<py::ContractType, py::LiteralType, py::CallableType,
                py::TypeType, py::SelfType, py::TypeVarType, py::ParamSpecType>(
          targetType)) {
    auto op = py::ClassUpcastOp::create(builder, loc(anchor), targetType,
                                        value.value);
    return {op.getResult(), targetType};
  }
  return value;
}

mlir::Value ModuleEmitter::emitBoolValue(Value value,
                                         const parser::Node &anchor) {
  if (value.value && value.value.getType().isInteger(1))
    return value.value;
  CallInferenceResult inference =
      types.inferMethodCallWithEvidence(value.type, "__bool__", {});
  auto op =
      py::BoolOp::create(builder, loc(anchor), builder.getI1Type(),
                         mlir::FlatSymbolRefAttr::get(&context, "__bool__"),
                         callProtocolFor(inference), value.value);
  return op.getResult();
}

} // namespace lython::emitter
