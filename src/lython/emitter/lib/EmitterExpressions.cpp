#include "EmitterCore.h"
#include "EmitterOps.h" // IWYU pragma: keep
#include "EmitterPyOps.h"
#include "EmitterSupport.h"
#include "PlatformConstants.h"
#include "TypeSystemSolver.h"

#include "AstAccess.h"
#include "PyProtocols.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

#include <optional>
#include <string>

namespace lython::emitter {

// origin -> then/else -> merge(block argument) with cf branches: the one
// shape every two-armed value merge in the emitter uses (a conditional whose
// arms may themselves open new blocks, so each arm branches from wherever
// its value became available).
mlir::Value ModuleEmitter::emitValueDiamond(
    mlir::Location location, mlir::Value condition, mlir::Type resultType,
    llvm::function_ref<mlir::Value()> emitThen,
    llvm::function_ref<mlir::Value()> emitElse) {
  mlir::Block *origin = builder.getInsertionBlock();
  mlir::Region *region = origin->getParent();
  mlir::Block *thenBlock =
      builder.createBlock(region, std::next(origin->getIterator()));
  mlir::Block *elseBlock =
      builder.createBlock(region, std::next(thenBlock->getIterator()));
  mlir::Block *merge =
      builder.createBlock(region, std::next(elseBlock->getIterator()));
  mlir::BlockArgument result = merge->addArgument(resultType, location);

  builder.setInsertionPointToEnd(origin);
  mlir::cf::CondBranchOp::create(builder, location, condition, thenBlock,
                                 mlir::ValueRange{}, elseBlock,
                                 mlir::ValueRange{});
  builder.setInsertionPointToStart(thenBlock);
  mlir::Value thenValue = emitThen();
  mlir::cf::BranchOp::create(builder, location, merge,
                             mlir::ValueRange{thenValue});
  builder.setInsertionPointToStart(elseBlock);
  mlir::Value elseValue = emitElse();
  mlir::cf::BranchOp::create(builder, location, merge,
                             mlir::ValueRange{elseValue});

  builder.setInsertionPointToStart(merge);
  return result;
}

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
    if (isModuleGlobalRead(name)) {
      mlir::Type type = moduleGlobals.lookup(name);
      auto op = py::GlobalGetOp::create(builder, loc(*expr), type,
                                        builder.getStringAttr(name));
      return {op.getResult(), type};
    }
    auto primitiveConstant = primitiveConstants.find(name);
    if (primitiveConstant != primitiveConstants.end())
      return emitPrimitiveConstant(*expr, primitiveConstant->second);
    std::optional<mlir::Type> symbolType = types.lookupSymbol(name);
    if (auto cls = types.lookupClass(name)) {
      mlir::Type typeType = types.typeObject(*cls);
      auto op = py::TypeObjectOp::create(builder, loc(*expr), typeType, *cls);
      return {op.getResult(), typeType};
    }
    std::string binding = std::string(name);
    if (std::optional<std::string> canonical =
            types.lookupCanonicalBinding(name))
      binding = *canonical;
    if (!symbolType) {
      diagnostics.push_back(
          parser::Diagnostic{parser::Severity::Error, expr->range.start,
                             "unresolved name '" + std::string(name) + "'"});
      return emitNone(*expr);
    }
    if (std::optional<Value> constant =
            emitStaticStringConstant(*expr, binding))
      return *constant;
    if (std::optional<Value> constant = emitStaticIntConstant(*expr, binding))
      return *constant;
    if (std::optional<Value> constant =
            emitManifestFloatConstant(*expr, binding))
      return *constant;
    if (std::optional<Value> constant =
            emitManifestIntConstant(*expr, binding))
      return *constant;
    if (std::optional<Value> constant =
            emitManifestStrConstant(*expr, binding))
      return *constant;
    if (std::optional<Value> literal =
            emitLiteralTypeConstant(*expr, *symbolType))
      return *literal;
    if (genericFunctions.count(name)) {
      // No ground context reached this reference (calls and expected-typed
      // uses are intercepted earlier), so there is no instantiation to
      // materialize; emitting the type-parameterized contract would only
      // defer the failure to the ABI check.
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, expr->range.start,
          "reference to generic function '" + std::string(name) +
              "' requires a call or an annotated Callable context to "
              "determine its type arguments"});
      return emitNone(*expr);
    }
    return emitBindingRef(*expr, binding, *symbolType);
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
    if (!qualified.empty()) {
      std::string binding = qualified;
      if (std::optional<std::string> canonical =
              types.lookupCanonicalBinding(qualified))
        binding = *canonical;
      // Platform constants fold whether or not the module attribute is a
      // known symbol (inference already typed them as target literals).
      if (std::optional<Value> constant =
              emitStaticStringConstant(*expr, binding))
        return *constant;
      if (auto symbol = types.lookupSymbol(qualified)) {
        if (std::optional<Value> constant =
                emitStaticIntConstant(*expr, binding))
          return *constant;
        if (std::optional<Value> constant =
                emitManifestFloatConstant(*expr, binding))
          return *constant;
        if (std::optional<Value> constant =
                emitManifestIntConstant(*expr, binding))
          return *constant;
        if (std::optional<Value> constant =
                emitManifestStrConstant(*expr, binding))
          return *constant;
        if (std::optional<Value> literal =
                emitLiteralTypeConstant(*expr, *symbol))
          return *literal;
        return emitBindingRef(*expr, binding, *symbol);
      }
    }
    return emitAttribute(*expr);
  }
  if (expr->kind == "Await")
    return emitAwait(*expr);
  if (expr->kind == "Yield") {
    const parser::Node *valueNode = ast::node(*expr, "value");
    Value yielded = valueNode ? emitExpr(valueNode) : emitNone(*expr);
    mlir::Type sentType =
        currentGeneratorSendType ? currentGeneratorSendType : types.none();
    auto op =
        py::YieldValueOp::create(builder, loc(*expr), sentType, yielded.value);
    return {op.getSent(), sentType};
  }
  if (expr->kind == "YieldFrom") {
    const parser::Node *source = ast::node(*expr, "value");
    if (source && (source->kind == "List" || source->kind == "Tuple")) {
      if (const auto *elts = ast::nodeList(*source, "elts"))
        for (const parser::NodePtr &element : *elts) {
          Value yielded = emitExpr(element.get());
          mlir::Type sentType = currentGeneratorSendType
                                    ? currentGeneratorSendType
                                    : types.none();
          py::YieldValueOp::create(builder, loc(*expr), sentType,
                                   yielded.value);
        }
      return emitNone(*expr);
    }
    Value delegated = emitExpr(source);
    YieldFromInferenceResult yieldFromInference =
        types.inferYieldFromWithEvidence(delegated.type);
    if (!requireStaticEvidence(*expr, yieldFromInference))
      return emitNone(*expr);
    auto op = py::YieldFromOp::create(
        builder, loc(*expr), yieldFromInference.completionType,
        yieldFromInference.protocolContract, delegated.value);
    return {op.getResult(), yieldFromInference.completionType};
  }
  if (expr->kind == "List" || expr->kind == "Tuple" || expr->kind == "Dict")
    return emitContainerLiteral(*expr);
  if (expr->kind == "ListComp")
    return emitListComp(*expr);
  if (expr->kind == "SetComp")
    return emitComprehension(*expr, /*isDict=*/false, /*isSet=*/true);
  if (expr->kind == "DictComp")
    return emitDictComp(*expr);
  if (expr->kind == "IfExp") {
    // Conditional expression via the same cf-block merge the if STATEMENT
    // uses (one merge mechanism; region-based scf.if results are invisible
    // to the runtime bundle machinery and the affine ownership verifier).
    const parser::Node *bodyNode = ast::node(*expr, "body");
    const parser::Node *elseNode = ast::node(*expr, "orelse");
    // Widen literal arms to their contracts before joining: CPython types
    // `"a" if c else "b"` as `str`, and a contract-typed merge argument
    // avoids literal-union runtime representations.
    mlir::Type resultType =
        types.join({types.widenLiteral(types.inferExpr(bodyNode)),
                    types.widenLiteral(types.inferExpr(elseNode))});
    mlir::Value condition =
        emitBoolValue(emitExpr(ast::node(*expr, "test")), *expr);

    mlir::Value result = emitValueDiamond(
        loc(*expr), condition, resultType,
        [&] { return coerceValue(emitExpr(bodyNode), resultType, *expr).value; },
        [&] {
          return coerceValue(emitExpr(elseNode), resultType, *expr).value;
        });
    return {result, resultType};
  }
  if (expr->kind == "BoolOp") {
    // Short-circuit `and`/`or` over BOOL-typed operands via the same
    // cf-block merge as IfExp: later operands are only evaluated when the
    // accumulated truth requires them. Restricted to bool operands because
    // CPython's BoolOp returns the deciding OPERAND VALUE — for bools the
    // truth bit IS the value, so typing the result `bool` is exact; for
    // other operand types (`x or default`) it would not be.
    const parser::Node *operatorNode = ast::node(*expr, "op");
    const auto *operandNodes = ast::nodeList(*expr, "values");
    bool isAnd = operatorNode && operatorNode->kind == "And";
    bool isOr = operatorNode && operatorNode->kind == "Or";
    if ((isAnd || isOr) && operandNodes && operandNodes->size() >= 2) {
      auto boolTyped = [&](const parser::Node *operand) {
        mlir::Type type = types.inferExpr(operand);
        if (type == types.boolType())
          return true;
        auto literal = mlir::dyn_cast_if_present<py::LiteralType>(type);
        return literal && (literal.getSpelling() == "True" ||
                           literal.getSpelling() == "False");
      };
      bool allBool = true;
      for (const parser::NodePtr &operand : *operandNodes)
        if (!operand || !boolTyped(operand.get()))
          allBool = false;
      if (allBool) {
        mlir::Value accumulated =
            emitBoolValue(emitExpr(operandNodes->front().get()), *expr);
        for (unsigned index = 1; index < operandNodes->size(); ++index) {
          mlir::Block *origin = builder.getInsertionBlock();
          mlir::Region *region = origin->getParent();
          mlir::Block *evalBlock =
              builder.createBlock(region, std::next(origin->getIterator()));
          mlir::Block *merge =
              builder.createBlock(region, std::next(evalBlock->getIterator()));
          mlir::BlockArgument joined =
              merge->addArgument(builder.getI1Type(), loc(*expr));

          builder.setInsertionPointToEnd(origin);
          mlir::Value decided = mlir::arith::ConstantIntOp::create(
              builder, loc(*expr), isOr ? 1 : 0, 1);
          // and: true -> evaluate next, false -> decided(false)
          // or:  true -> decided(true), false -> evaluate next
          if (isAnd)
            mlir::cf::CondBranchOp::create(builder, loc(*expr), accumulated,
                                           evalBlock, mlir::ValueRange{},
                                           merge, mlir::ValueRange{decided});
          else
            mlir::cf::CondBranchOp::create(builder, loc(*expr), accumulated,
                                           merge, mlir::ValueRange{decided},
                                           evalBlock, mlir::ValueRange{});
          builder.setInsertionPointToStart(evalBlock);
          mlir::Value next =
              emitBoolValue(emitExpr((*operandNodes)[index].get()), *expr);
          mlir::cf::BranchOp::create(builder, loc(*expr), merge,
                                     mlir::ValueRange{next});
          builder.setInsertionPointToStart(merge);
          accumulated = joined;
        }
        auto boxed = py::CastFromPrimOp::create(builder, loc(*expr),
                                                types.boolType(), accumulated);
        return {boxed.getResult(), types.boolType()};
      }
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, expr->range.start,
          "`and`/`or` requires bool-typed operands (CPython's operand-value "
          "result is only representable for bools); wrap non-bool operands "
          "in explicit comparisons"});
      return emitNone(*expr);
    }
  }
  if (expr->kind == "Lambda")
    return emitLambda(*expr);
  if (expr->kind == "JoinedStr")
    return emitJoinedStr(*expr);
  if (expr->kind == "FormattedValue")
    return emitFormattedValue(*expr);
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
  if (const auto *value = ast::bytes(expr, "value")) {
    mlir::Type type = types.contract("builtins.bytes");
    auto op = py::BytesConstantOp::create(
        builder, loc(expr), type,
        builder.getStringAttr(llvm::StringRef(
            reinterpret_cast<const char *>(value->data()), value->size())));
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
  // str % args is printf-style formatting, not a manifest __mod__; it needs
  // the unevaluated right-hand AST (tuple literals supply the arguments), so
  // intercept before the operands are emitted.
  if (ast::isOperator(ast::node(expr, "op"), "Mod")) {
    const parser::Node *leftNode = ast::node(expr, "left");
    if (leftNode && types.widenLiteral(types.inferExpr(leftNode)) ==
                        types.contract("builtins.str"))
      return emitPercentFormat(expr);
  }
  Value lhs = emitExpr(ast::node(expr, "left"));
  Value rhs = emitExpr(ast::node(expr, "right"));
  const parser::Node *op = ast::node(expr, "op");
  if (std::optional<Value> primitive = emitPrimitiveBinary(expr, lhs, rhs, op))
    return *primitive;
  // int ** compile-time negative int is a float in CPython (decision log):
  // desugar to float(base) ** float(exponent) so the manifest float.__pow__
  // carries the runtime semantics (0 ** -n raises ZeroDivisionError there).
  if (ast::isOperator(op, "Pow")) {
    auto literalType = mlir::dyn_cast_if_present<py::LiteralType>(rhs.type);
    llvm::StringRef spelling =
        literalType ? literalType.getSpelling() : llvm::StringRef();
    bool negativeIntLiteral =
        spelling.size() > 1 && spelling.front() == '-' &&
        llvm::all_of(spelling.drop_front(),
                     [](char c) { return c >= '0' && c <= '9'; });
    if (negativeIntLiteral &&
        types.widenLiteral(lhs.type) == types.intType()) {
      llvm::APFloat exponent(llvm::APFloat::IEEEdouble());
      llvm::Expected<llvm::APFloat::opStatus> status =
          exponent.convertFromString(spelling,
                                     llvm::APFloat::rmNearestTiesToEven);
      if (!status) {
        llvm::consumeError(status.takeError());
      } else if (!exponent.isInfinity()) {
        // Exponents beyond the double range stay on the int path; its
        // runtime rejection mirrors CPython's OverflowError there.
        Value baseFloat = emitFloatFromInt(expr, lhs);
        auto exponentConst = py::FloatConstantOp::create(
            builder, loc(expr), types.floatType(),
            builder.getF64FloatAttr(exponent.convertToDouble()));
        Value exponentValue{exponentConst.getResult(), types.floatType()};
        return emitBinarySpecial<py::PowOp>(expr, "__pow__", baseFloat,
                                            exponentValue, types.floatType());
      }
    }
  }
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
  if (ast::isOperator(op, "Pow"))
    return emitBinarySpecial<py::PowOp>(expr, "__pow__", lhs, rhs, result);
  if (ast::isOperator(op, "Add"))
    return emitBinarySpecial<py::AddOp>(expr, "__add__", lhs, rhs, result);
  // A fall-through to __add__ here would silently mis-execute unhandled
  // operators (`a @ b` on non-tensors used to become an addition).
  std::string spelling = op ? op->kind : std::string("<missing>");
  diagnostics.push_back(parser::Diagnostic{
      parser::Severity::Error, expr.range.start,
      "binary operator '" + spelling + "' is not supported for these operand "
      "types yet"});
  return emitNone(expr);
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
  if (std::optional<Value> optional = emitOptionalCompare(expr, lhs, rhs, op))
    return *optional;
  return emitScalarCompare(expr, lhs, rhs, op);
}

Value ModuleEmitter::emitScalarCompare(const parser::Node &expr, Value lhs,
                                       Value rhs, const parser::Node *op) {
  if (std::optional<Value> primitive = emitPrimitiveCompare(expr, lhs, rhs, op))
    return *primitive;
  // bool-vs-bool equality lowers through the values' truth bits: the runtime
  // has no `bool.__eq__` manifest method, and two bools are equal exactly when
  // their truth values agree.
  auto isBoolLike = [&](mlir::Type type) {
    if (type == types.boolType())
      return true;
    auto literal = mlir::dyn_cast_if_present<py::LiteralType>(type);
    return literal && (literal.getSpelling() == "True" ||
                       literal.getSpelling() == "False");
  };
  if ((ast::isOperator(op, "Eq") || ast::isOperator(op, "NotEq") ||
       ast::isOperator(op, "Is") || ast::isOperator(op, "IsNot")) &&
      isBoolLike(lhs.type) && isBoolLike(rhs.type)) {
    mlir::Value lhsBit = emitBoolValue(lhs, expr);
    mlir::Value rhsBit = emitBoolValue(rhs, expr);
    bool negated =
        ast::isOperator(op, "NotEq") || ast::isOperator(op, "IsNot");
    auto compared = mlir::arith::CmpIOp::create(
        builder, loc(expr),
        negated ? mlir::arith::CmpIPredicate::ne : mlir::arith::CmpIPredicate::eq,
        lhsBit, rhsBit);
    auto pyBool = py::CastFromPrimOp::create(builder, loc(expr),
                                             types.boolType(),
                                             compared.getResult());
    return Value{pyBool.getResult(), types.boolType()};
  }
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
    if (std::optional<MethodBinding> method =
            lookupClassMethod(rhs.type, "__contains__")) {
      Value membership = emitInlineOperatorCall(expr, rhs, *method, {lhs});
      if (ast::isOperator(op, "In"))
        return membership;
      mlir::Value bit = emitBoolValue(membership, expr);
      auto one = mlir::arith::ConstantIntOp::create(builder, loc(expr), 1, 1);
      mlir::Value flipped =
          mlir::arith::XOrIOp::create(builder, loc(expr), bit, one);
      auto pyBool = py::CastFromPrimOp::create(builder, loc(expr),
                                               types.boolType(), flipped);
      return Value{pyBool.getResult(), types.boolType()};
    }
    CallInferenceResult inference =
        types.inferMethodCallWithEvidence(rhs.type, "__contains__", {lhs.type});
    if (!requireStaticEvidence(expr, inference))
      return emitNone(expr);
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

std::optional<Value> ModuleEmitter::emitOptionalCompare(
    const parser::Node &expr, Value lhs, Value rhs, const parser::Node *op) {
  // Only `==` / `!=` have well-defined semantics against `None`; ordering
  // comparisons on an Optional are a TypeError in CPython, so we let those
  // fall through to the manifest path (which rejects them).
  bool negated = ast::isOperator(op, "NotEq");
  if (!ast::isOperator(op, "Eq") && !negated)
    return std::nullopt;

  // An `Optional[T]` operand is a union carrying `None` plus exactly one
  // concrete member T. The single-member restriction keeps the active-member
  // dispatch two-way (None vs T); richer unions stay unsupported.
  auto singleMemberOptional = [&](mlir::Type type) -> mlir::Type {
    auto unionType = mlir::dyn_cast_if_present<py::UnionType>(type);
    if (!unionType || !unionType.hasMember(types.none()))
      return nullptr;
    mlir::Type member;
    for (mlir::Type candidate : unionType.getMemberTypes()) {
      if (isNoneTypeLike(candidate))
        continue;
      if (member)
        return nullptr;
      member = candidate;
    }
    return member;
  };

  auto concrete = [&](Value value) {
    return !mlir::isa<py::UnionType>(value.type) && !isNoneTypeLike(value.type);
  };
  mlir::Type lhsMember = singleMemberOptional(lhs.type);
  mlir::Type rhsMember = singleMemberOptional(rhs.type);
  bool bothOptional = lhsMember && rhsMember;
  bool lhsOptional = lhsMember && concrete(rhs);
  bool rhsOptional = rhsMember && concrete(lhs);
  if (!bothOptional && !lhsOptional && !rhsOptional)
    return std::nullopt;

  mlir::Type resultType = types.boolType();
  mlir::Location location = loc(expr);
  auto testNone = [&](Value value) {
    return py::UnionTestOp::create(builder, location, builder.getI1Type(),
                                   value.value,
                                   mlir::TypeAttr::get(types.none()))
        .getResult();
  };
  auto negate = [&](mlir::Value bit) {
    mlir::Value one =
        mlir::arith::ConstantIntOp::create(builder, location, 1, 1);
    return mlir::arith::XOrIOp::create(builder, location, bit, one).getResult();
  };

  // `present` (computed in the origin block) selects the member-compare branch;
  // its complement is the "absent" branch whose value `absentBit` supplies. For
  // one Optional operand the member is always present iff that union is not
  // None; for two, both must be present, and when they are not, equality holds
  // exactly when both are None.
  mlir::Value present;
  mlir::Value absentBit;
  if (bothOptional) {
    mlir::Value lhsNone = testNone(lhs);
    mlir::Value rhsNone = testNone(rhs);
    present = mlir::arith::AndIOp::create(builder, location, negate(lhsNone),
                                          negate(rhsNone));
    absentBit =
        mlir::arith::AndIOp::create(builder, location, lhsNone, rhsNone);
  } else {
    Value optional = lhsOptional ? lhs : rhs;
    present = negate(testNone(optional));
    // `None == concrete` is False (True under `!=`); the negation below flips it.
    absentBit = mlir::arith::ConstantIntOp::create(builder, location, 0, 1);
  }
  if (negated)
    absentBit = negate(absentBit);

  // cf-block merge, mirroring the IfExp lowering (region-based scf.if results
  // are invisible to the runtime bundle machinery and the ownership verifier).
  mlir::Value result = emitValueDiamond(
      location, present, resultType,
      // Present branch: project the concrete member(s) and re-enter scalar
      // dispatch.
      [&] {
        auto unwrap = [&](Value value, mlir::Type member) -> Value {
          auto op =
              py::UnionUnwrapOp::create(builder, location, member, value.value);
          return Value{op.getResult(), member};
        };
        Value presentLhs = lhsMember ? unwrap(lhs, lhsMember) : lhs;
        Value presentRhs = rhsMember ? unwrap(rhs, rhsMember) : rhs;
        Value compared = emitScalarCompare(expr, presentLhs, presentRhs, op);
        return coerceValue(compared, resultType, expr).value;
      },
      // Absent branch: the statically known equality bit for the None case(s).
      [&] {
        return py::CastFromPrimOp::create(builder, location, resultType,
                                          absentBit)
            .getResult();
      });
  return Value{result, resultType};
}

Value ModuleEmitter::emitSubscript(const parser::Node &expr) {
  Value container = emitExpr(ast::node(expr, "value"));
  // Shaped primitives are indexed before the slice is emitted: their indices
  // are static shape coordinates, not values a manifest __getitem__ receives.
  if (container.value &&
      mlir::isa<mlir::RankedTensorType>(container.value.getType())) {
    if (std::optional<Value> element = emitPrimitiveTensorGetItem(
            expr, container, ast::node(expr, "slice")))
      return element->value ? *element : emitNone(expr);
  }
  Value index = emitExpr(ast::node(expr, "slice"));
  if (std::optional<MethodBinding> method =
          lookupClassMethod(container.type, "__getitem__"))
    return emitInlineOperatorCall(expr, container, *method, {index});
  CallInferenceResult inference = types.inferMethodCallWithEvidence(
      container.type, "__getitem__", {index.type});
  if (!requireStaticEvidence(expr, inference))
    return emitNone(expr);
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
                              methodBinding.signature.publicCallable, {});

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
  return emitFunctionObject(anchor, symbolName, boundPublicSig.publicCallable,
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

// `asyncio.run(coro)` drives the coroutine to completion. The accepted subset
// executes awaited chains eagerly (see the top-level-await dispatch), so it
// desugars to awaiting the argument; await inference then types the result
// from the coroutine's evidence instead of the manifest contract's Any.
Value ModuleEmitter::emitAsyncioRunCall(const parser::Node &expr) {
  const auto *args = ast::nodeList(expr, "args");
  const auto *keywords = ast::nodeList(expr, "keywords");
  if (!args || args->size() != 1 || !args->front() ||
      (keywords && !keywords->empty())) {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, expr.range.start,
        "asyncio.run supports exactly one coroutine argument (the debug "
        "keyword is not supported)"});
    return emitNone(expr);
  }
  Value awaitable = emitExpr(args->front().get());
  return emitAwaitValue(expr, awaitable);
}

Value ModuleEmitter::emitAwaitValue(const parser::Node &anchor,
                                    Value awaitable) {
  AwaitInferenceResult inference = types.inferAwaitWithEvidence(awaitable.type);
  return emitAwaitValue(anchor, awaitable, inference);
}

Value ModuleEmitter::emitAwaitValue(const parser::Node &anchor, Value awaitable,
                                    const AwaitInferenceResult &inference) {
  if (!requireStaticEvidence(anchor, inference))
    return emitNone(anchor);

  auto op = py::AwaitOp::create(builder, loc(anchor), inference.resultType,
                                inference.awaitContract, awaitable.value);
  return {op.getResult(), inference.resultType};
}

// `[expr for x in it]` desugars to a runtime-list build loop over the
// structural-mutation machinery:
//   __listcomp<N>: list[T] = []   (T = element expression type)
//   for x in it: __listcomp<N>.append(expr)
// The synthetic For statement shares the original target/iter/element
// subtrees, so iteration typing, scoping diagnostics, and loop-carried
// threading match the handwritten loop exactly. Neither the temp list nor
// the loop target leaks into the enclosing scope.
Value ModuleEmitter::emitListComp(const parser::Node &expr) {
  return emitComprehension(expr, /*isDict=*/false);
}

Value ModuleEmitter::emitDictComp(const parser::Node &expr) {
  return emitComprehension(expr, /*isDict=*/true);
}

Value ModuleEmitter::emitComprehension(const parser::Node &expr,
                                       bool isDict, bool isSet) {
  auto reject = [&](const std::string &message) {
    diagnostics.push_back(
        parser::Diagnostic{parser::Severity::Error, expr.range.start, message});
    return emitNone(expr);
  };
  auto sharedSubtree = [&](llvm::StringRef name) -> parser::NodePtr {
    const parser::Field *field = parser::findField(expr, name);
    if (!field || !std::holds_alternative<parser::NodePtr>(field->value))
      return nullptr;
    return std::get<parser::NodePtr>(field->value);
  };
  const auto *generators = ast::nodeList(expr, "generators");
  if (!generators || generators->empty())
    return reject("malformed comprehension");
  parser::NodePtr elt, keyExpr, valueExpr;
  if (isDict) {
    keyExpr = sharedSubtree("key");
    valueExpr = sharedSubtree("value");
    if (!keyExpr || !valueExpr)
      return reject("malformed dict comprehension");
  } else {
    elt = sharedSubtree("elt");
    if (!elt)
      return reject("malformed list comprehension");
  }

  struct CompGenerator {
    parser::NodePtr target;
    parser::NodePtr iter;
    llvm::SmallVector<parser::NodePtr, 2> filters;
    llvm::StringRef targetName;
  };
  llvm::SmallVector<CompGenerator, 2> chain;
  for (const parser::NodePtr &generator : *generators) {
    if (!generator)
      return reject("malformed list comprehension");
    if (ast::integer(*generator, "is_async").value_or(0))
      return reject("async comprehensions are not supported");
    const parser::Field *targetField = parser::findField(*generator, "target");
    const parser::Field *iterField = parser::findField(*generator, "iter");
    if (!targetField ||
        !std::holds_alternative<parser::NodePtr>(targetField->value) ||
        !iterField ||
        !std::holds_alternative<parser::NodePtr>(iterField->value))
      return reject("malformed list comprehension generator");
    CompGenerator entry;
    entry.target = std::get<parser::NodePtr>(targetField->value);
    entry.iter = std::get<parser::NodePtr>(iterField->value);
    if (!entry.target || entry.target->kind != "Name" || !entry.iter)
      return reject("list comprehension target must be a simple name");
    entry.targetName = ast::nameSpelling(*entry.target);
    if (const auto *ifs = ast::nodeList(*generator, "ifs"))
      entry.filters.append(ifs->begin(), ifs->end());
    chain.push_back(std::move(entry));
  }

  // Element type: bind each generator's target to its iteration element type
  // (later iterables may reference earlier targets), then infer the element
  // expression under those bindings.
  mlir::Type elementType, keyType, valueType;
  {
    auto scope = types.pushScope();
    for (const CompGenerator &entry : chain) {
      mlir::Type iterableType = types.inferExpr(entry.iter.get());
      CallInferenceResult iterInference =
          types.inferMethodCallWithEvidence(iterableType, "__iter__", {});
      if (!requireStaticEvidence(expr, iterInference))
        return emitNone(expr);
      CallInferenceResult nextInference = types.inferMethodCallWithEvidence(
          iterInference.resultType, "__next__", {});
      if (!requireStaticEvidence(expr, nextInference))
        return emitNone(expr);
      types.bindLocalSymbol(entry.targetName,
                            types.widenLiteral(nextInference.resultType));
    }
    if (isDict) {
      keyType = types.widenLiteral(types.inferExpr(keyExpr.get()));
      valueType = types.widenLiteral(types.inferExpr(valueExpr.get()));
    } else {
      elementType = types.widenLiteral(types.inferExpr(elt.get()));
    }
  }
  if (isDict ? (!keyType || !valueType) : !elementType)
    return reject("cannot infer the comprehension element type");

  // Temp result container, bound as a local so the build loop threads it.
  std::string tmp =
      (isDict ? "__dictcomp" : (isSet ? "__setcomp" : "__listcomp")) +
      std::to_string(++listCompCounter);
  mlir::Type containerType =
      isDict ? py::ContractType::get(builder.getContext(), "builtins.dict",
                                     {keyType, valueType})
             : py::ContractType::get(builder.getContext(),
                                     isSet ? "builtins.set" : "builtins.list",
                                     {elementType});
  auto pack =
      py::PackOp::create(builder, loc(expr), containerType, mlir::ValueRange{});
  values[tmp] = Value{pack.getResult(), containerType};

  // list: for <target> in <iter>: <tmp>.append(<elt>)
  // dict: for <target> in <iter>: <tmp>[<key>] = <value>
  parser::NodePtr tmpName = parser::makeNode("Name", expr.range);
  parser::addField(*tmpName, "id", tmp);
  parser::NodePtr statement;
  if (isDict) {
    parser::NodePtr subscript = parser::makeNode("Subscript", expr.range);
    parser::addField(*subscript, "value", tmpName);
    parser::addField(*subscript, "slice", keyExpr);
    statement = parser::makeNode("Assign", expr.range);
    parser::addField(*statement, "targets",
                     std::vector<parser::NodePtr>{subscript});
    parser::addField(*statement, "value", valueExpr);
  } else {
    parser::NodePtr appendAttr = parser::makeNode("Attribute", expr.range);
    parser::addField(*appendAttr, "value", tmpName);
    parser::addField(*appendAttr, "attr",
                     std::string(isSet ? "add" : "append"));
    parser::NodePtr appendCall = parser::makeNode("Call", expr.range);
    parser::addField(*appendCall, "func", appendAttr);
    parser::addField(*appendCall, "args", std::vector<parser::NodePtr>{elt});
    parser::addField(*appendCall, "keywords", std::vector<parser::NodePtr>{});
    statement = parser::makeNode("Expr", expr.range);
    parser::addField(*statement, "value", appendCall);
  }
  // Build inside-out: each generator's filters wrap the current statement
  // (`if c1: if c2: ...`), then its For wraps that.
  for (const CompGenerator &entry : llvm::reverse(chain)) {
    for (const parser::NodePtr &filter : llvm::reverse(entry.filters)) {
      parser::NodePtr guard = parser::makeNode("If", expr.range);
      parser::addField(*guard, "test", filter);
      parser::addField(*guard, "body",
                       std::vector<parser::NodePtr>{statement});
      parser::addField(*guard, "orelse", std::vector<parser::NodePtr>{});
      statement = guard;
    }
    parser::NodePtr loop = parser::makeNode("For", expr.range);
    parser::addField(*loop, "target", entry.target);
    parser::addField(*loop, "iter", entry.iter);
    parser::addField(*loop, "body", std::vector<parser::NodePtr>{statement});
    parser::addField(*loop, "orelse", std::vector<parser::NodePtr>{});
    statement = loop;
  }

  llvm::SmallVector<std::pair<llvm::StringRef, std::optional<Value>>, 2>
      priorTargets;
  for (const CompGenerator &entry : chain) {
    std::optional<Value> prior;
    if (auto found = values.find(entry.targetName); found != values.end())
      prior = found->second;
    priorTargets.push_back({entry.targetName, prior});
  }
  emitFor(*statement);

  auto built = values.find(tmp);
  Value result = built != values.end()
                     ? built->second
                     : Value{pack.getResult(), containerType};
  values.erase(tmp);
  for (auto &[name, prior] : priorTargets) {
    if (prior)
      values[name] = *prior;
    else
      values.erase(name);
  }
  return result;
}

Value ModuleEmitter::emitExprExpected(const parser::Node *expr,
                                      mlir::Type expected) {
  if (!expr || !expected)
    return emitExpr(expr);
  if (expr->kind == "Lambda")
    if (auto expectedCallable =
            mlir::dyn_cast_if_present<py::CallableType>(expected))
      return emitLambda(*expr, expectedCallable);
  if (expr->kind == "List" || expr->kind == "Tuple" || expr->kind == "Dict")
    return emitContainerLiteral(*expr, expected);
  if (expr->kind == "Name") {
    llvm::StringRef name = ast::nameSpelling(*expr);
    if (values.find(name) == values.end()) {
      auto generic = genericFunctions.find(name);
      if (generic != genericFunctions.end()) {
        // A ground expected callable determines the instantiation, so a
        // first-class reference to a generic function materializes as a
        // reference to the matching specialization.
        auto expectedCallable =
            mlir::dyn_cast_if_present<py::CallableType>(expected);
        if (!expectedCallable ||
            unboundStaticParameterCount(expectedCallable) != 0)
          return emitExpr(expr);
        std::optional<std::pair<std::string, py::CallableType>>
            specialization = ensureGenericSpecialization(
                *expr, generic->second, expectedCallable);
        if (!specialization)
          return emitNone(*expr);
        return emitBindingRef(*expr, specialization->first,
                              specialization->second);
      }
    }
  }
  return emitExpr(expr);
}

Value ModuleEmitter::emitContainerLiteral(const parser::Node &expr,
                                          mlir::Type expected) {
  // The expectation only distributes when its container class matches the
  // literal's node kind; a mismatched expectation falls back to synthesis so
  // the caller's contract check reports it at the right place.
  auto expectedContract = mlir::dyn_cast_if_present<py::ContractType>(expected);
  llvm::StringRef expectedName =
      expectedContract ? expectedContract.getContractName() : llvm::StringRef();
  llvm::ArrayRef<mlir::Type> expectedArgs =
      expectedContract ? expectedContract.getArguments()
                       : llvm::ArrayRef<mlir::Type>();
  bool container = false;

  llvm::SmallVector<Value, 8> valuesToPack;
  bool empty = true;
  if (const auto *elts = ast::nodeList(expr, "elts")) {
    mlir::Type elementExpected;
    llvm::ArrayRef<mlir::Type> positionalExpected;
    if (expr.kind == "List" && expectedName == "builtins.list" &&
        expectedArgs.size() == 1) {
      container = true;
      elementExpected = expectedArgs.front();
    } else if (expr.kind == "Tuple" && expectedName == "builtins.tuple") {
      if (expectedArgs.size() == elts->size()) {
        container = true;
        positionalExpected = expectedArgs;
      } else if (expectedArgs.size() == 1) {
        container = true;
        elementExpected = expectedArgs.front();
      }
    }
    for (auto [index, elt] : llvm::enumerate(*elts)) {
      empty = false;
      mlir::Type eltExpected = index < positionalExpected.size()
                                   ? positionalExpected[index]
                                   : elementExpected;
      valuesToPack.push_back(emitExprExpected(elt.get(), eltExpected));
    }
  }
  if (const auto *keys = ast::nodeList(expr, "keys")) {
    mlir::Type keyExpected;
    mlir::Type valueExpected;
    if (expr.kind == "Dict" && expectedName == "builtins.dict" &&
        expectedArgs.size() == 2) {
      container = true;
      keyExpected = expectedArgs[0];
      valueExpected = expectedArgs[1];
    }
    const auto *vals = ast::nodeList(expr, "values");
    for (auto [index, key] : llvm::enumerate(*keys)) {
      empty = false;
      if (key)
        valuesToPack.push_back(emitExprExpected(key.get(), keyExpected));
      if (vals && index < vals->size())
        valuesToPack.push_back(
            emitExprExpected((*vals)[index].get(), valueExpected));
    }
  }
  // An empty literal synthesizes its top-erased element type, which the
  // stricter lowering contract match later rejects against a concrete
  // formal; adopting the expectation types the pack correctly from the
  // start. Non-empty literals keep the synthesized type: their elements
  // determine it, and the caller's coercion validates the expectation.
  mlir::Type resultType =
      (empty && container) ? expected : types.inferExpr(&expr);
  llvm::SmallVector<mlir::Value, 8> operands;
  for (Value value : valuesToPack)
    operands.push_back(value.value);
  auto op = py::PackOp::create(builder, loc(expr), resultType, operands);
  return {op.getResult(), resultType};
}

// A literal-typed symbol's value is fully determined by its type, so a
// reference materializes the constant directly instead of requiring a runtime
// binding (used for imported module-level literal constants).
std::optional<Value>
ModuleEmitter::emitLiteralTypeConstant(const parser::Node &anchor,
                                       mlir::Type type) {
  auto literal = mlir::dyn_cast_if_present<py::LiteralType>(type);
  if (!literal)
    return std::nullopt;
  llvm::StringRef spelling = literal.getSpelling();
  if (spelling == "None")
    return emitNone(anchor);
  if (spelling == "True" || spelling == "False") {
    auto op = py::BoolConstantOp::create(builder, loc(anchor), type,
                                         builder.getBoolAttr(spelling == "True"));
    return Value{op.getResult(), type};
  }
  llvm::StringRef digits = spelling;
  if (digits.consume_front("-") ? !digits.empty() : !digits.empty()) {
    bool allDigits = llvm::all_of(digits, [](char c) {
      return c >= '0' && c <= '9';
    });
    if (allDigits) {
      auto op = py::IntConstantOp::create(builder, loc(anchor), type,
                                          builder.getStringAttr(spelling));
      return Value{op.getResult(), type};
    }
  }
  if (spelling.size() >= 2 && spelling.front() == '"' &&
      spelling.back() == '"') {
    auto op = py::StrConstantOp::create(
        builder, loc(anchor), type,
        builder.getStringAttr(spelling.drop_front().drop_back()));
    return Value{op.getResult(), type};
  }
  return std::nullopt;
}

Value ModuleEmitter::emitBindingRef(const parser::Node &anchor,
                                    llvm::StringRef binding, mlir::Type type,
                                    llvm::ArrayRef<Value> captures) {
  // A binding reference materializes a solver-resolved Python object whose
  // later dispatch (call, attribute access) observes its type. Fabricating an
  // `object` top here would let unresolved evidence reach lowering as a
  // dynamic receiver, which the static contract model forbids. A missing type
  // is an internal solver invariant violation: fail explicitly at this static
  // boundary instead of erasing the receiver.
  if (!type) {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, anchor.range.start,
        "internal: binding reference '" + binding.str() +
            "' has no resolved type; refusing to erase to object"});
    return emitNone(anchor);
  }
  mlir::Type resultType = type;
  llvm::SmallVector<mlir::Value, 4> captureValues;
  for (Value capture : captures)
    captureValues.push_back(capture.value);
  auto op =
      py::BindingRefOp::create(builder, loc(anchor), resultType,
                               builder.getStringAttr(binding), captureValues);
  return {op.getResult(), resultType};
}

std::optional<Value> ModuleEmitter::emitManifestFloatConstant(
    const parser::Node &anchor, llvm::StringRef binding) {
  const py::protocols::Table &table =
      py::protocols::Table::get(context);
  std::optional<double> value = table.moduleFloatConstant(binding);
  if (!value)
    return std::nullopt;
  mlir::Type type = types.floatType();
  auto op = py::FloatConstantOp::create(builder, loc(anchor), type,
                                        builder.getF64FloatAttr(*value));
  return Value{op.getResult(), type};
}

std::optional<Value> ModuleEmitter::emitManifestIntConstant(
    const parser::Node &anchor, llvm::StringRef binding) {
  const py::protocols::Table &table =
      py::protocols::Table::get(context);
  std::optional<long long> value = table.moduleIntConstant(binding);
  if (!value)
    return std::nullopt;
  std::string spelling = std::to_string(*value);
  mlir::Type type = types.literal(spelling);
  auto op = py::IntConstantOp::create(builder, loc(anchor), type,
                                      builder.getStringAttr(spelling));
  return Value{op.getResult(), type};
}

std::optional<Value> ModuleEmitter::emitManifestStrConstant(
    const parser::Node &anchor, llvm::StringRef binding) {
  const py::protocols::Table &table =
      py::protocols::Table::get(context);
  std::optional<std::string> value = table.moduleStrConstant(binding);
  if (!value)
    return std::nullopt;
  mlir::Type type = types.literal("\"" + *value + "\"");
  auto op = py::StrConstantOp::create(builder, loc(anchor), type,
                                      builder.getStringAttr(*value));
  return Value{op.getResult(), type};
}

std::optional<Value> ModuleEmitter::emitStaticIntConstant(
    const parser::Node &anchor, llvm::StringRef binding) {
  if (!py::platform_constants::isStaticIntBinding(binding))
    return std::nullopt;
  std::optional<long long> value =
      py::platform_constants::staticIntValue(binding, options.targetTriple);
  if (!value) {
    diagnostics.push_back(
        parser::Diagnostic{parser::Severity::Error, anchor.range.start,
                           "unsupported target platform for static constant '" +
                               std::string(binding) + "'"});
    return emitNone(anchor);
  }
  std::string spelling = std::to_string(*value);
  mlir::Type type = types.literal(spelling);
  auto op = py::IntConstantOp::create(builder, loc(anchor), type,
                                      builder.getStringAttr(spelling));
  return Value{op.getResult(), type};
}

std::optional<Value> ModuleEmitter::emitStaticStringConstant(
    const parser::Node &anchor, llvm::StringRef binding, bool allowCallable) {
  bool isStaticBinding = py::platform_constants::isStaticStringBinding(binding);
  bool isStaticCallable =
      allowCallable && py::platform_constants::isStaticStringCallable(binding);
  if (!isStaticBinding && !isStaticCallable)
    return std::nullopt;
  std::optional<std::string> value =
      py::platform_constants::staticStringValue(binding, options.targetTriple);
  if (!value) {
    diagnostics.push_back(
        parser::Diagnostic{parser::Severity::Error, anchor.range.start,
                           "unsupported target platform for static constant '" +
                               std::string(binding) + "'"});
    return emitNone(anchor);
  }
  mlir::Type type = types.literal("\"" + *value + "\"");
  auto op = py::StrConstantOp::create(builder, loc(anchor), type,
                                      builder.getStringAttr(*value));
  return Value{op.getResult(), type};
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
  // An empty pack has no elements, so no element is ever retrieved and the
  // placeholder element type can never be observed by later dispatch. This is
  // the AGENTS.md-sanctioned empty-container placeholder, not an erased
  // dynamic receiver: a non-empty pack always joins concrete element evidence.
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
    mlir::Type actual = types.widenLiteral(value.type);
    for (mlir::Type member : unionType.getMemberTypes()) {
      if (!isAssignableWithStaticEvidence(actual, member, module))
        continue;
      Value memberValue = coerceValue(value, member, anchor);
      if (memberValue.type != member)
        continue;
      auto op = py::UnionWrapOp::create(builder, loc(anchor), targetType,
                                        memberValue.value);
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
  if (!requireStaticEvidence(anchor, inference)) {
    auto fallback =
        mlir::arith::ConstantIntOp::create(builder, loc(anchor), 1, 1);
    return fallback.getResult();
  }
  auto op =
      py::BoolOp::create(builder, loc(anchor), builder.getI1Type(),
                         mlir::FlatSymbolRefAttr::get(&context, "__bool__"),
                         callProtocolFor(inference), value.value);
  return op.getResult();
}

} // namespace lython::emitter
