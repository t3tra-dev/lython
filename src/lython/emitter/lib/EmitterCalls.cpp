#include "EmitterCore.h"
#include "EmitterPyOps.h"
#include "EmitterSupport.h"
#include "PlatformConstants.h"
#include "PyProtocols.h"

#include "AstAccess.h"
#include "EmitterOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <string>

namespace lython::emitter {
namespace {

mlir::Value constantI1(mlir::OpBuilder &builder, mlir::Location loc,
                       bool value) {
  return mlir::arith::ConstantIntOp::create(builder, loc, value ? 1 : 0, 1)
      .getResult();
}

Value boxedBool(mlir::OpBuilder &builder, mlir::Location loc, TypeSystem &types,
                mlir::Value bit) {
  auto pyBool = py::CastFromPrimOp::create(builder, loc, types.boolType(), bit);
  return {pyBool.getResult(), types.boolType()};
}

std::string typeText(mlir::Type type) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << type;
  return text;
}

bool callHasNoArguments(const parser::Node &expr) {
  const auto *args = ast::nodeList(expr, "args");
  const auto *keywords = ast::nodeList(expr, "keywords");
  return (!args || args->empty()) && (!keywords || keywords->empty());
}

std::optional<llvm::StringRef> contractName(mlir::Type type) {
  auto contract = mlir::dyn_cast_if_present<py::ContractType>(type);
  if (!contract)
    return std::nullopt;
  return contract.getContractName();
}

} // namespace

CallOperands
ModuleEmitter::emitCallOperands(const parser::Node &expr,
                                llvm::ArrayRef<Value> leadingPositional,
                                bool includeAstArguments,
                                py::CallableType expectedContract) {
  CallOperands operands;
  for (Value value : leadingPositional) {
    operands.positional.push_back(value);
    operands.positionalUnpacked.push_back(0);
    operands.positionalTypes.push_back(value.type);
  }
  if (!includeAstArguments)
    return operands;

  llvm::ArrayRef<mlir::Type> expectedPositional =
      expectedContract ? expectedContract.getPositionalTypes()
                       : llvm::ArrayRef<mlir::Type>();
  // A static type parameter as formal means the expectation is the CALL's
  // output (the argument determines it), not an input to distribute; a
  // starred argument breaks positional alignment for everything after it.
  auto expectedFor = [&](std::size_t index) -> mlir::Type {
    if (index >= expectedPositional.size())
      return {};
    mlir::Type formal = expectedPositional[index];
    if (formal && py::isStaticTypeParameter(formal))
      return {};
    return formal;
  };

  if (const auto *args = ast::nodeList(expr, "args")) {
    bool positionalAligned = true;
    for (const parser::NodePtr &arg : *args) {
      bool unpacked = arg && arg->kind == "Starred";
      if (unpacked)
        positionalAligned = false;
      const parser::Node *valueNode =
          unpacked ? ast::node(*arg, "value") : arg.get();
      Value value = positionalAligned
                        ? emitExprExpected(
                              valueNode,
                              expectedFor(operands.positionalTypes.size()))
                        : emitExpr(valueNode);
      operands.positional.push_back(value);
      operands.positionalUnpacked.push_back(unpacked ? 1 : 0);
      if (unpacked) {
        if (!appendStarredArgumentTypes(value.type, types,
                                        operands.positionalTypes)) {
          operands.valid = false;
          operands.failureReason =
              "starred call arguments require a statically sized tuple";
        }
      } else {
        operands.positionalTypes.push_back(value.type);
      }
    }
  }

  if (const auto *keywords = ast::nodeList(expr, "keywords")) {
    for (const parser::NodePtr &keyword : *keywords) {
      if (auto name = ast::string(*keyword, "arg")) {
        mlir::Type literal = types.literal("\"" + std::string(*name) + "\"");
        auto stringOp = py::StrConstantOp::create(
            builder, loc(*keyword), literal, builder.getStringAttr(*name));
        operands.keywordNames.push_back({stringOp.getResult(), literal});
        Value keywordValue = emitExpr(ast::node(*keyword, "value"));
        operands.keywordValues.push_back(keywordValue);
        operands.keywordTypes.push_back(
            CallKeywordType{std::string(*name), keywordValue.type});
        continue;
      }
      operands.keywordValues.push_back(emitExpr(ast::node(*keyword, "value")));
    }
  }

  return operands;
}

Value ModuleEmitter::emitCallableDispatch(const parser::Node &anchor,
                                          Value callee,
                                          const CallOperands &operands,
                                          mlir::Type resultOverride) {
  if (!operands.valid) {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, anchor.range.start, operands.failureReason});
    return emitNone(anchor);
  }
  Value posPack = emitPack(operands.positional, operands.positionalUnpacked);
  Value namePack = emitPack(operands.keywordNames);
  Value valuePack = emitPack(operands.keywordValues);
  CallInferenceResult inference = types.inferCallWithEvidence(
      callee.type, operands.positionalTypes, operands.keywordTypes);
  if (!requireStaticEvidence(anchor, inference))
    return emitNone(anchor);
  mlir::Type resultType = resultOverride ? resultOverride : inference.resultType;
  auto op =
      py::CallOp::create(builder, loc(anchor), mlir::TypeRange{resultType},
                         callProtocolFor(inference, callee.type), callee.value,
                         posPack.value, namePack.value, valuePack.value);
  return {op.getResults().front(), resultType};
}

Value ModuleEmitter::emitCall(const parser::Node &expr) {
  const parser::Node *calleeNode = ast::node(expr, "func");
  std::string calleeQualified = ast::qualifiedName(calleeNode);

  if (std::optional<Value> primitive =
          emitPrimitiveConstructorCall(expr, calleeNode))
    return *primitive;
  if (std::optional<Value> primitive = emitPrimitiveFactoryCall(expr, calleeNode))
    return *primitive;
  if (std::optional<Value> primitive =
          emitPrimitiveRuntimeCall(expr, calleeNode))
    return *primitive;

  if (std::optional<Value> v = tryEmitIsInstanceCall(expr, calleeNode))
    return *v;
  if (std::optional<Value> v = tryEmitIntCall(expr, calleeNode))
    return *v;
  if (std::optional<Value> v = tryEmitFloatCall(expr, calleeNode))
    return *v;
  if (std::optional<Value> v = tryEmitStrCall(expr, calleeNode))
    return *v;
  if (std::optional<Value> v = tryEmitListCall(expr, calleeNode))
    return *v;
  if (std::optional<Value> v = tryEmitPrintCall(expr, calleeNode))
    return *v;
  if (std::optional<Value> v = tryEmitReducerCall(expr, calleeNode))
    return *v;

  if (!calleeQualified.empty())
    if (auto cls = types.lookupClass(calleeQualified)) {
      if (std::optional<llvm::StringRef> symbol = contractName(*cls))
        if (std::optional<Value> v =
                rejectStubSourceCall(expr, *symbol, /*instantiation=*/true))
          return *v;
      return emitClassInstantiation(expr, llvm::StringRef(calleeQualified),
                                    *cls);
    }

  if (calleeNode && calleeNode->kind == "Name") {
    llvm::StringRef name = ast::nameSpelling(*calleeNode);
    if (auto cls = types.lookupClass(name)) {
      if (std::optional<llvm::StringRef> symbol = contractName(*cls))
        if (std::optional<Value> v =
                rejectStubSourceCall(expr, *symbol, /*instantiation=*/true))
          return *v;
      return emitClassInstantiation(expr, name, *cls);
    }
  }

  if (std::optional<Value> v = tryEmitLenCall(expr, calleeNode))
    return *v;
  if (std::optional<Value> v = tryEmitNextCall(expr, calleeNode))
    return *v;
  if (std::optional<Value> v = tryEmitRoundCall(expr, calleeNode))
    return *v;

  if (std::optional<Value> primitiveCall =
          emitDirectPrimitiveFunctionCall(expr, calleeNode))
    return *primitiveCall;

  if (std::optional<Value> v = tryEmitReprCall(expr, calleeNode))
    return *v;

  if (calleeNode && calleeNode->kind == "Name") {
    llvm::StringRef name = ast::nameSpelling(*calleeNode);
    auto local = values.find(name);
    if (local != values.end() && local->second.boundMethod)
      return emitInlineMethodCall(expr, local->second.boundMethod->receiver,
                                  local->second.boundMethod->method);
    if (values.find(name) == values.end())
      if (std::optional<std::string> canonical =
              types.lookupCanonicalBinding(name)) {
        if (std::optional<Value> v =
                rejectStubSourceCall(expr, *canonical, /*instantiation=*/false))
          return *v;
        if (py::platform_constants::isStaticStringCallable(*canonical) &&
            callHasNoArguments(expr))
          if (std::optional<Value> constant =
                  emitStaticStringConstant(expr, *canonical,
                                           /*allowCallable=*/true))
            return *constant;
        // open() with a literal 'b' mode dispatches to the binary arm
        // (FileIO result); see the matching special case in inferExpr.
        if (*canonical == "_io.open") {
          const auto *openArgs = ast::nodeList(expr, "args");
          if (openArgs && openArgs->size() >= 2 && (*openArgs)[1]) {
            auto mode = ast::string(*(*openArgs)[1], "value");
            if (mode && mode->find('b') != std::string_view::npos) {
              const py::protocols::Table &table =
                  py::protocols::Table::get(context);
              mlir::Type calleeType =
                  table.freeFunctionContract("_io.open_binary")
                      .value_or(types.contract("builtins.function"));
              Value binaryCallee = emitBindingRef(*calleeNode,
                                                  "_io.open_binary",
                                                  calleeType);
              return emitCallableDispatch(expr, binaryCallee,
                                          emitCallOperands(expr));
            }
          }
        }
        if (*canonical == "asyncio.sleep")
          if (auto symbol = types.lookupSymbol(name))
            return emitCallableDispatch(
                expr, emitBindingRef(*calleeNode, *canonical, *symbol),
                emitCallOperands(expr), types.inferExpr(&expr));
        if (*canonical == "asyncio.run")
          return emitAsyncioRunCall(expr);
      }
  }

  if (calleeNode && calleeNode->kind == "Attribute" &&
      !calleeQualified.empty()) {
    if (auto symbol = types.lookupSymbol(calleeQualified)) {
      std::string binding = calleeQualified;
      if (std::optional<std::string> canonical =
              types.lookupCanonicalBinding(calleeQualified))
        binding = *canonical;
      if (std::optional<Value> v =
              rejectStubSourceCall(expr, binding, /*instantiation=*/false))
        return *v;
      if (py::platform_constants::isStaticStringCallable(binding) &&
          callHasNoArguments(expr))
        if (std::optional<Value> constant =
                emitStaticStringConstant(expr, binding,
                                         /*allowCallable=*/true))
          return *constant;
      if (binding == "asyncio.run")
        return emitAsyncioRunCall(expr);
      mlir::Type resultOverride =
          binding == "asyncio.sleep" ? types.inferExpr(&expr) : mlir::Type();
      Value callee = emitBindingRef(*calleeNode, binding, *symbol);
      return emitCallableDispatch(
          expr, callee,
          emitCallOperands(expr, {}, /*includeAstArguments=*/true,
                           mlir::dyn_cast_if_present<py::CallableType>(
                               callee.type)),
          resultOverride);
    }
  }

  if (calleeNode && calleeNode->kind == "Attribute") {
    if (const parser::Node *receiverNode = ast::node(*calleeNode, "value")) {
      if (auto methodName = ast::string(*calleeNode, "attr")) {
        Value receiver = emitExpr(receiverNode);
        if (std::optional<MethodBinding> method =
                lookupClassMethod(receiver.type, *methodName)) {
          if (method->async && !method->symbolName.empty())
            return emitCallableDispatch(
                expr, emitMethodObject(*calleeNode, receiver, *method),
                emitCallOperands(expr));
          return emitInlineMethodCall(expr, receiver, *method);
        }

        CallOperands operands = emitCallOperands(expr);
        if (!operands.valid) {
          diagnostics.push_back(parser::Diagnostic{
              parser::Severity::Error, expr.range.start,
              operands.failureReason});
          return emitNone(expr);
        }
        Value posPack =
            emitPack(operands.positional, operands.positionalUnpacked);
        Value namePack = emitPack(operands.keywordNames);
        Value valuePack = emitPack(operands.keywordValues);
        CallInferenceResult inference = types.inferMethodCallWithEvidence(
            receiver.type, *methodName, operands.positionalTypes,
            operands.keywordTypes);
        if (!requireStaticEvidence(expr, inference))
          return emitNone(expr);
        mlir::Type resultType =
            inference ? inference.resultType : types.inferExpr(&expr);
        // Manifest-declared structural mutators (`ly.typing.structural_mutators`)
        // may reallocate the receiver's storage, so the call carries an extra
        // receiver-typed result that rebinds the local — the mutation becomes
        // an ordinary SSA reassignment and loop-carried threading forwards the
        // (possibly grown) representation across back-edges.
        if (receiverNode->kind == "Name" &&
            types.isStructuralMutatorMethod(receiver.type, *methodName)) {
          llvm::StringRef receiverName = ast::nameSpelling(*receiverNode);
          auto bound = values.find(receiverName);
          if (bound != values.end() && bound->second.value == receiver.value) {
            auto op = py::CallOp::create(
                builder, loc(expr),
                mlir::TypeRange{resultType, receiver.value.getType()},
                callProtocolFor(inference), receiver.value, posPack.value,
                namePack.value, valuePack.value);
            op->setAttr("ly.bound_method", builder.getStringAttr(*methodName));
            op->setAttr("ly.structural_mutation", builder.getUnitAttr());
            values[receiverName] = Value{op.getResult(1), receiver.type};
            return {op.getResults().front(), resultType};
          }
        }
        auto op =
            py::CallOp::create(builder, loc(expr), mlir::TypeRange{resultType},
                               callProtocolFor(inference), receiver.value,
                               posPack.value, namePack.value, valuePack.value);
        op->setAttr("ly.bound_method", builder.getStringAttr(*methodName));
        return {op.getResults().front(), resultType};
      }
      }
  }

  if (calleeNode && calleeNode->kind == "Name") {
    llvm::StringRef name = ast::nameSpelling(*calleeNode);
    if (values.find(name) == values.end()) {
      auto generic = genericFunctions.find(name);
      if (generic != genericFunctions.end())
        return emitGenericCall(expr, *calleeNode, generic->second);
    }
    if (!types.lookupSymbol(name) && !types.lookupClass(name)) {
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, calleeNode->range.start,
          "unresolved name '" + name.str() + "'"});
      return emitNone(expr);
    }
  }

  // The callee is emitted before the operands on purpose: Python evaluates
  // the callee first, and its Callable contract is the expectation the
  // argument emission distributes (lambda parameters, empty literals).
  Value callee = emitExpr(calleeNode);
  return emitCallableDispatch(
      expr, callee,
      emitCallOperands(expr, {}, /*includeAstArguments=*/true,
                       mlir::dyn_cast_if_present<py::CallableType>(
                           callee.type)));
}

Value ModuleEmitter::emitGenericCall(const parser::Node &expr,
                                     const parser::Node &calleeNode,
                                     GenericFunctionInfo &generic) {
  // The generic contract still distributes as the argument expectation: its
  // ground formals propagate, and expectedFor skips the type-parameter ones.
  CallOperands operands =
      emitCallOperands(expr, {}, /*includeAstArguments=*/true,
                       generic.signature.publicCallable);
  if (!operands.valid) {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, expr.range.start, operands.failureReason});
    return emitNone(expr);
  }
  CallInferenceResult inference = types.inferCallWithEvidence(
      generic.signature.publicCallable, operands.positionalTypes,
      operands.keywordTypes);
  if (!requireStaticEvidence(expr, inference))
    return emitNone(expr);
  auto resolved = mlir::dyn_cast_if_present<py::CallableType>(
      inference.evidence.callableContract);
  if (!resolved) {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, expr.range.start,
        "generic call did not resolve to a Callable contract"});
    return emitNone(expr);
  }
  std::optional<std::pair<std::string, py::CallableType>> specialization =
      ensureGenericSpecialization(expr, generic, resolved);
  if (!specialization)
    return emitNone(expr);
  Value callee = emitBindingRef(calleeNode, specialization->first,
                                specialization->second);
  return emitCallableDispatch(expr, callee, operands);
}

std::optional<Value>
ModuleEmitter::tryEmitIsInstanceCall(const parser::Node &expr,
                                     const parser::Node *calleeNode) {
  if (!calleeNode || calleeNode->kind != "Name" ||
      ast::nameSpelling(*calleeNode) != "isinstance" ||
      values.find("isinstance") != values.end())
    return std::nullopt;
  const auto *keywords = ast::nodeList(expr, "keywords");
  const auto *args = ast::nodeList(expr, "args");
  if ((keywords && !keywords->empty()) || !args || args->size() != 2 ||
      !args->front() || args->front()->kind == "Starred" || !(*args)[1] ||
      (*args)[1]->kind == "Starred") {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, expr.range.start,
        "isinstance requires exactly two positional arguments"});
    return emitNone(expr);
  }

  std::optional<mlir::Type> target =
      isinstanceTargetType((*args)[1].get(), types);
  if (!target) {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, expr.range.start,
        "second argument to isinstance must be a statically resolved class "
        "type"});
    return emitNone(expr);
  }

  Value input = emitExpr(args->front().get());
  IsInstanceAnalysis analysis =
      analyzeIsInstance(input.type, *target, types, module);
  if (analysis.kind == IsInstanceAnalysis::Kind::Unsupported) {
    std::string reason = analysis.failureReason.empty()
                             ? "unsupported isinstance evidence"
                             : analysis.failureReason;
    diagnostics.push_back(parser::Diagnostic{parser::Severity::Error,
                                             expr.range.start, reason});
    return emitNone(expr);
  }

  mlir::Value bit;
  if (analysis.kind == IsInstanceAnalysis::Kind::AlwaysTrue) {
    bit = constantI1(builder, loc(expr), true);
  } else if (analysis.kind == IsInstanceAnalysis::Kind::AlwaysFalse) {
    bit = constantI1(builder, loc(expr), false);
  } else if (analysis.kind == IsInstanceAnalysis::Kind::UnionTest) {
    if (!mlir::isa<py::UnionType>(input.value.getType())) {
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, expr.range.start,
          "isinstance union evidence expected a union-typed value, got " +
              typeText(input.value.getType())});
      return emitNone(expr);
    }
    for (mlir::Type member : analysis.unionMembers) {
      auto test =
          py::UnionTestOp::create(builder, loc(expr), builder.getI1Type(),
                                  input.value, mlir::TypeAttr::get(member));
      bit = bit ? mlir::arith::OrIOp::create(builder, loc(expr), bit,
                                             test.getResult())
                      .getResult()
                : test.getResult();
    }
    if (!bit)
      bit = constantI1(builder, loc(expr), false);
  } else if (analysis.kind == IsInstanceAnalysis::Kind::UnionClassTest) {
    if (!mlir::isa<py::UnionType>(input.value.getType()) ||
        analysis.unionMembers.size() != 1) {
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, expr.range.start,
          "isinstance union class evidence expected one union member"});
      return emitNone(expr);
    }
    mlir::Type member = analysis.unionMembers.front();
    auto unionTest =
        py::UnionTestOp::create(builder, loc(expr), builder.getI1Type(),
                                input.value, mlir::TypeAttr::get(member));
    auto unwrap =
        py::UnionUnwrapOp::create(builder, loc(expr), member, input.value);
    auto classTest = py::ClassTestOp::create(
        builder, loc(expr), builder.getI1Type(), unwrap.getResult(),
        mlir::TypeAttr::get(analysis.targetType));
    bit =
        mlir::arith::AndIOp::create(builder, loc(expr), unionTest.getResult(),
                                    classTest.getResult())
            .getResult();
  } else if (analysis.kind == IsInstanceAnalysis::Kind::ClassTest) {
    auto test = py::ClassTestOp::create(
        builder, loc(expr), builder.getI1Type(), input.value,
        mlir::TypeAttr::get(analysis.targetType));
    bit = test.getResult();
  }
  return boxedBool(builder, loc(expr), types, bit);
}

std::optional<Value>
ModuleEmitter::tryEmitIntCall(const parser::Node &expr,
                              const parser::Node *calleeNode) {
  // int(x) is __int__ dispatch / literal parsing (CPython semantics), not
  // construction — intercept before the class-instantiation paths claim
  // builtins.int. Zero-argument int() stays on the instantiation path.
  if (!calleeNode || calleeNode->kind != "Name" ||
      ast::nameSpelling(*calleeNode) != "int" ||
      values.find("int") != values.end())
    return std::nullopt;
  const auto *intArgs = ast::nodeList(expr, "args");
  const auto *intKeywords = ast::nodeList(expr, "keywords");
  auto intClass = types.lookupClass("int");
  std::optional<llvm::StringRef> intSymbol =
      intClass ? contractName(*intClass) : std::nullopt;
  if (!intSymbol || *intSymbol != "builtins.int" || !intArgs ||
      intArgs->size() != 1 || (intKeywords && !intKeywords->empty()) ||
      !intArgs->front() || intArgs->front()->kind == "Starred")
    return std::nullopt;
  mlir::Type argumentType =
      types.widenLiteral(types.inferExpr(intArgs->front().get()));
  if (argumentType == types.intType()) {
    // int is immutable, so int(n) is the identity (CPython returns n).
    Value argument = emitExpr(intArgs->front().get());
    return coerceValue(argument, types.intType(), expr);
  }
  if (argumentType == types.strType() || argumentType == types.floatType()) {
    // The runtime-level __int__ methods of str (base-10 parse) and float
    // (truncation) are deliberately not part of the typed manifest surface —
    // CPython has no str.__int__ — so the contract is built here instead of
    // going through method inference.
    Value argument =
        coerceValue(emitExpr(intArgs->front().get()), argumentType, expr);
    mlir::Type resultType = types.intType();
    mlir::Type contract = py::CallableType::get(&context, {argumentType}, {},
                                                {}, {}, {resultType});
    auto op = py::IntOp::create(
        builder, loc(expr), resultType,
        mlir::FlatSymbolRefAttr::get(&context, "__int__"),
        mlir::TypeAttr::get(contract), argument.value);
    return Value{op.getResult(), resultType};
  }
  return std::nullopt;
}

Value ModuleEmitter::emitFloatFromInt(const parser::Node &anchor,
                                      Value argument) {
  // The correctly rounded conversion is the runtime-level __float__ of
  // builtins.int (not on the typed manifest surface, like str.__int__), so
  // the contract is built here instead of going through method inference.
  argument = coerceValue(argument, types.intType(), anchor);
  mlir::Type resultType = types.floatType();
  mlir::Type contract = py::CallableType::get(&context, {types.intType()}, {},
                                              {}, {}, {resultType});
  auto op = py::FloatOp::create(
      builder, loc(anchor), resultType,
      mlir::FlatSymbolRefAttr::get(&context, "__float__"),
      mlir::TypeAttr::get(contract), argument.value);
  return Value{op.getResult(), resultType};
}

std::optional<Value>
ModuleEmitter::tryEmitFloatCall(const parser::Node &expr,
                                const parser::Node *calleeNode) {
  // float(x) is __float__ dispatch (CPython semantics), not construction —
  // intercept before the class-instantiation paths claim builtins.float.
  if (!calleeNode || calleeNode->kind != "Name" ||
      ast::nameSpelling(*calleeNode) != "float" ||
      values.find("float") != values.end())
    return std::nullopt;
  const auto *floatArgs = ast::nodeList(expr, "args");
  const auto *floatKeywords = ast::nodeList(expr, "keywords");
  auto floatClass = types.lookupClass("float");
  std::optional<llvm::StringRef> floatSymbol =
      floatClass ? contractName(*floatClass) : std::nullopt;
  if (!floatSymbol || *floatSymbol != "builtins.float" || !floatArgs ||
      floatArgs->size() != 1 || (floatKeywords && !floatKeywords->empty()) ||
      !floatArgs->front() || floatArgs->front()->kind == "Starred")
    return std::nullopt;
  mlir::Type argumentType =
      types.widenLiteral(types.inferExpr(floatArgs->front().get()));
  if (argumentType == types.floatType()) {
    // float is immutable, so float(x) is the identity (CPython returns x).
    Value argument = emitExpr(floatArgs->front().get());
    return coerceValue(argument, types.floatType(), expr);
  }
  if (argumentType == types.intType()) {
    Value argument = emitExpr(floatArgs->front().get());
    return emitFloatFromInt(expr, argument);
  }
  return std::nullopt;
}

std::optional<Value>
ModuleEmitter::tryEmitStrCall(const parser::Node &expr,
                              const parser::Node *calleeNode) {
  // str(x) is __str__ dispatch (CPython semantics), not construction —
  // intercept before the class-instantiation paths claim builtins.str.
  if (!calleeNode || calleeNode->kind != "Name" ||
      ast::nameSpelling(*calleeNode) != "str" ||
      values.find("str") != values.end())
    return std::nullopt;
  const auto *strArgs = ast::nodeList(expr, "args");
  const auto *strKeywords = ast::nodeList(expr, "keywords");
  auto strClass = types.lookupClass("str");
  std::optional<llvm::StringRef> strSymbol =
      strClass ? contractName(*strClass) : std::nullopt;
  if (strSymbol && *strSymbol == "builtins.str" && strArgs &&
      strArgs->size() == 1 && (!strKeywords || strKeywords->empty())) {
    mlir::Type argumentType =
        types.widenLiteral(types.inferExpr(strArgs->front().get()));
    if (std::optional<MethodBinding> method =
            lookupClassMethod(argumentType, "__str__")) {
      Value argument = emitExpr(strArgs->front().get());
      llvm::StringMap<Value> emptyKeywords;
      Value descriptorReceiver =
          emitDescriptorReceiver(expr, argument, *method);
      return emitInlineMethodBody(expr, descriptorReceiver,
                                  methodBindingBindsReceiver(*method),
                                  *method, {}, emptyKeywords);
    }
    if (CallInferenceResult inference = types.inferMethodCallWithEvidence(
            argumentType, "__str__", {})) {
      Value argument =
          coerceValue(emitExpr(strArgs->front().get()), argumentType, expr);
      mlir::Type resultType = types.contract("builtins.str");
      auto op = py::StrOp::create(
          builder, loc(expr), resultType,
          mlir::FlatSymbolRefAttr::get(&context, "__str__"),
          mlir::TypeAttr::get(callProtocolFor(inference)), argument.value);
      return Value{op.getResult(), resultType};
    }
    // No __str__ evidence: fall through to the instantiation path's
    // explicit rejection.
  }
  return std::nullopt;
}

std::optional<Value>
ModuleEmitter::tryEmitListCall(const parser::Node &expr,
                               const parser::Node *calleeNode) {
  // list(<genexpr>) is the list comprehension over the same element/generator
  // chain — route to the comprehension emitter before the class-instantiation
  // paths claim builtins.list.
  if (!calleeNode || calleeNode->kind != "Name" ||
      ast::nameSpelling(*calleeNode) != "list" ||
      values.find("list") != values.end())
    return std::nullopt;
  const auto *listArgs = ast::nodeList(expr, "args");
  const auto *listKeywords = ast::nodeList(expr, "keywords");
  if (listArgs && listArgs->size() == 1 && listArgs->front() &&
      listArgs->front()->kind == "GeneratorExp" &&
      (!listKeywords || listKeywords->empty()))
    return emitComprehension(*listArgs->front(), /*isDict=*/false);
  return std::nullopt;
}

std::optional<Value>
ModuleEmitter::tryEmitPrintCall(const parser::Node &expr,
                                const parser::Node *calleeNode) {
  // Multi-argument print desugars to one write of the space-joined
  // stringified arguments (CPython's sep=" " default): the unified print
  // resolver stays single-argument. Zero-argument print desugars to one
  // empty-string write (builtin_print_impl with objects_length == 0 emits
  // only the end="\n" terminator).
  if (!calleeNode || calleeNode->kind != "Name" ||
      ast::nameSpelling(*calleeNode) != "print" ||
      values.find("print") != values.end())
    return std::nullopt;
  const auto *printArgs = ast::nodeList(expr, "args");
  const auto *printKeywords = ast::nodeList(expr, "keywords");
  bool noPrintKeywords = !printKeywords || printKeywords->empty();
  if (noPrintKeywords && (!printArgs || printArgs->empty())) {
    mlir::Type emptyType = types.literal("\"\"");
    auto empty = py::StrConstantOp::create(builder, loc(expr), emptyType,
                                           builder.getStringAttr(""));
    Value piece = coerceValue(Value{empty.getResult(), emptyType},
                              types.contract("builtins.str"), expr);
    Value printCallee = emitExpr(calleeNode);
    CallOperands operands =
        emitCallOperands(expr, {piece}, /*includeAstArguments=*/false);
    return emitCallableDispatch(expr, printCallee, operands);
  }
  bool plainArguments = printArgs && printArgs->size() >= 2 &&
                        noPrintKeywords;
  if (plainArguments)
    for (const parser::NodePtr &argument : *printArgs)
      if (!argument || argument->kind == "Starred")
        plainArguments = false;
  if (plainArguments) {
    mlir::Type strType = types.contract("builtins.str");
    auto stringify = [&](const parser::Node *argNode) -> std::optional<Value> {
      mlir::Type argumentType = types.widenLiteral(types.inferExpr(argNode));
      if (argumentType == strType)
        return coerceValue(emitExpr(argNode), strType, expr);
      if (std::optional<MethodBinding> method =
              lookupClassMethod(argumentType, "__str__")) {
        Value argument = emitExpr(argNode);
        llvm::StringMap<Value> emptyKeywords;
        Value receiver = emitDescriptorReceiver(expr, argument, *method);
        return emitInlineMethodBody(expr, receiver,
                                    methodBindingBindsReceiver(*method),
                                    *method, {}, emptyKeywords);
      }
      // Non-str builtins render via __repr__ (CPython's str(x) equals
      // repr(x) for every non-str builtin, containers included; the
      // runtime manifest has no container __str__).
      if (CallInferenceResult inference =
              types.inferMethodCallWithEvidence(argumentType, "__repr__",
                                                {})) {
        Value argument = coerceValue(emitExpr(argNode), argumentType, expr);
        auto op = py::ReprOp::create(
            builder, loc(expr), strType,
            mlir::FlatSymbolRefAttr::get(&context, "__repr__"),
            mlir::TypeAttr::get(callProtocolFor(inference)), argument.value);
        return Value{op.getResult(), strType};
      }
      return std::nullopt;
    };
    bool allConverted = true;
    Value joined;
    for (auto [index, argument] : llvm::enumerate(*printArgs)) {
      std::optional<Value> piece = stringify(argument.get());
      if (!piece) {
        allConverted = false;
        break;
      }
      if (index == 0) {
        joined = *piece;
        continue;
      }
      mlir::Type separatorType = types.literal("\" \"");
      auto separator = py::StrConstantOp::create(
          builder, loc(expr), separatorType, builder.getStringAttr(" "));
      joined = emitBinarySpecial<py::AddOp>(
          expr, "__add__", joined,
          Value{separator.getResult(), separatorType}, strType);
      joined = emitBinarySpecial<py::AddOp>(expr, "__add__", joined, *piece,
                                            strType);
    }
    if (allConverted) {
      Value printCallee = emitExpr(calleeNode);
      CallOperands operands =
          emitCallOperands(expr, {joined}, /*includeAstArguments=*/false);
      return emitCallableDispatch(expr, printCallee, operands);
    }
    // Fall through: the lowering reports the single-argument restriction
    // for arguments without __str__ evidence.
  }
  return std::nullopt;
}

std::optional<Value>
ModuleEmitter::tryEmitReducerCall(const parser::Node &expr,
                                  const parser::Node *calleeNode) {
  // sum/any/all/max/min over an iterable desugar to accumulator loops
  // (any/all with an early-exit break, preserving CPython short-circuiting;
  // max/min carry a seen-flag and raise ValueError when the iterable is
  // empty); generator expression arguments fuse through the emitFor path.
  if (!calleeNode || calleeNode->kind != "Name")
    return std::nullopt;
  llvm::StringRef reducer = ast::nameSpelling(*calleeNode);
  if (!(reducer == "sum" || reducer == "any" || reducer == "all" ||
        reducer == "max" || reducer == "min") ||
      values.find(reducer) != values.end())
    return std::nullopt;
  const auto *reducerArgs = ast::nodeList(expr, "args");
  const auto *reducerKeywords = ast::nodeList(expr, "keywords");
  // The element type of the reducer's iterable: genexpr arguments infer
  // their element expression under progressively bound chain targets
  // (like emitComprehension); plain iterables go through __iter__/__next__.
  auto reducerElementType = [&]() -> mlir::Type {
    const parser::Node *arg = reducerArgs->front().get();
    auto iterationElement = [&](const parser::Node *iterable) -> mlir::Type {
      mlir::Type iterableType = types.inferExpr(iterable);
      if (!iterableType)
        return {};
      CallInferenceResult iterInference = types.inferMethodCallWithEvidence(
          types.widenLiteral(iterableType), "__iter__", {});
      if (!iterInference)
        return {};
      CallInferenceResult nextInference = types.inferMethodCallWithEvidence(
          iterInference.resultType, "__next__", {});
      if (!nextInference)
        return {};
      return types.widenLiteral(nextInference.resultType);
    };
    if (arg->kind != "GeneratorExp")
      return iterationElement(arg);
    const parser::Field *eltField = parser::findField(*arg, "elt");
    const auto *gens = ast::nodeList(*arg, "generators");
    if (!eltField ||
        !std::holds_alternative<parser::NodePtr>(eltField->value) || !gens)
      return {};
    auto scope = types.pushScope();
    for (const parser::NodePtr &gen : *gens) {
      if (!gen)
        return {};
      const parser::Node *target = ast::node(*gen, "target");
      const parser::Node *iter = ast::node(*gen, "iter");
      if (!target || target->kind != "Name" || !iter)
        return {};
      mlir::Type elementType = iterationElement(iter);
      if (!elementType)
        return {};
      types.bindLocalSymbol(ast::nameSpelling(*target), elementType);
    }
    return types.widenLiteral(types.inferExpr(
        std::get<parser::NodePtr>(eltField->value).get()));
  };
  // Two-scalar form `min(a, b)` / `max(a, b)`: evaluate both operands
  // once, compare, and merge through the same cf-block pattern as IfExp
  // (`min(a, b)` keeps `a` on ties, matching CPython's first-minimal
  // rule). The non-selected operand's edge gets its release from the
  // partial-forward placement.
  if (reducerArgs && reducerArgs->size() == 2 && reducerArgs->front() &&
      (*reducerArgs)[1] && (!reducerKeywords || reducerKeywords->empty()) &&
      (reducer == "max" || reducer == "min")) {
    Value lhs = emitExpr(reducerArgs->front().get());
    Value rhs = emitExpr((*reducerArgs)[1].get());
    mlir::Type resultType = types.join(
        {types.widenLiteral(lhs.type), types.widenLiteral(rhs.type)});
    if (resultType) {
      // Literal-vs-literal selects statically (a constant-condition
      // merge would strand the unselected literal's materialized object
      // without a release).
      auto lhsLiteral = mlir::dyn_cast<py::LiteralType>(lhs.type);
      auto rhsLiteral = mlir::dyn_cast<py::LiteralType>(rhs.type);
      if (lhsLiteral && rhsLiteral) {
        llvm::StringRef lhsSpelling = lhsLiteral.getSpelling();
        llvm::StringRef rhsSpelling = rhsLiteral.getSpelling();
        long long lhsInt = 0, rhsInt = 0;
        std::optional<bool> pickRhs;
        if (!lhsSpelling.getAsInteger(10, lhsInt) &&
            !rhsSpelling.getAsInteger(10, rhsInt))
          pickRhs = reducer == "min" ? rhsInt < lhsInt : rhsInt > lhsInt;
        else if (lhsSpelling.size() >= 2 && rhsSpelling.size() >= 2 &&
                 lhsSpelling.front() == '"' && rhsSpelling.front() == '"') {
          llvm::StringRef lhsText = lhsSpelling.drop_front().drop_back();
          llvm::StringRef rhsText = rhsSpelling.drop_front().drop_back();
          pickRhs = reducer == "min" ? rhsText < lhsText : rhsText > lhsText;
        }
        if (pickRhs)
          return coerceValue(*pickRhs ? rhs : lhs, resultType, expr);
      }
      parser::Node comparisonOp(reducer == "min" ? "Lt" : "Gt");
      comparisonOp.range = expr.range;
      Value comparison =
          emitScalarCompare(expr, rhs, lhs, &comparisonOp);
      mlir::Value condition = emitBoolValue(comparison, expr);
      // Literal-vs-literal comparisons fold at emit time: select the
      // operand statically instead of emitting a constant-condition
      // merge (whose dead arm would strand the unselected literal's
      // materialized object without a release).
      if (auto constantCondition =
              condition.getDefiningOp<mlir::arith::ConstantIntOp>())
        return coerceValue(constantCondition.value() != 0 ? rhs : lhs,
                           resultType, expr);
      if (auto constantBool =
              comparison.value.getDefiningOp<py::BoolConstantOp>())
        return coerceValue(constantBool.getValue() ? rhs : lhs, resultType,
                           expr);

      mlir::Value result = emitValueDiamond(
          loc(expr), condition, resultType,
          [&] { return coerceValue(rhs, resultType, expr).value; },
          [&] { return coerceValue(lhs, resultType, expr).value; });
      return Value{result, resultType};
    }
  }
  if (reducerArgs && reducerArgs->size() == 1 && reducerArgs->front() &&
      (!reducerKeywords || reducerKeywords->empty()) &&
      (reducer == "max" || reducer == "min")) {
    mlir::Type elementType = reducerElementType();
    auto contract =
        mlir::dyn_cast_if_present<py::ContractType>(elementType);
    llvm::StringRef contractName =
        contract ? contract.getContractName() : llvm::StringRef();
    mlir::Value placeholder;
    if (contractName == "builtins.int") {
      placeholder = py::IntConstantOp::create(builder, loc(expr),
                                              types.literal("0"),
                                              builder.getStringAttr("0"))
                        .getResult();
    } else if (contractName == "builtins.str") {
      placeholder =
          py::StrConstantOp::create(builder, loc(expr),
                                    types.literal("\"\""),
                                    builder.getStringAttr(""))
              .getResult();
    } else if (contractName == "builtins.float") {
      placeholder = py::FloatConstantOp::create(
                        builder, loc(expr), elementType,
                        builder.getF64FloatAttr(0.0))
                        .getResult();
    }
    if (!placeholder) {
      // max()/min() over an EMPTY literal always raises: emit the
      // ValueError directly (there is no element type to desugar with).
      const parser::Node *arg = reducerArgs->front().get();
      bool emptyLiteral =
          (arg->kind == "List" || arg->kind == "Tuple") &&
          [&] {
            const auto *elts = ast::nodeList(*arg, "elts");
            return !elts || elts->empty();
          }();
      if (emptyLiteral) {
        parser::NodePtr errorName = parser::makeNode("Name", expr.range);
        parser::addField(*errorName, "id", std::string("ValueError"));
        parser::NodePtr message = parser::makeNode("Constant", expr.range);
        parser::addField(*message, "value",
                         reducer.str() + "() iterable argument is empty");
        parser::NodePtr errorCall = parser::makeNode("Call", expr.range);
        parser::addField(*errorCall, "func", errorName);
        parser::addField(*errorCall, "args",
                         std::vector<parser::NodePtr>{message});
        parser::addField(*errorCall, "keywords",
                         std::vector<parser::NodePtr>{});
        parser::NodePtr raiseNode = parser::makeNode("Raise", expr.range);
        parser::addField(*raiseNode, "exc", errorCall);
        emitStatement(*raiseNode);
        // py.raise terminates the block; park the (unreachable) rest of
        // the enclosing expression in a fresh block.
        mlir::Block *dead = builder.createBlock(
            builder.getInsertionBlock()->getParent(),
            std::next(builder.getInsertionBlock()->getIterator()));
        builder.setInsertionPointToStart(dead);
        return emitNone(expr);
      }
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, expr.range.start,
          reducer.str() +
              "() requires int/str/float iterable element evidence"});
      return emitNone(expr);
    }
    std::string tmp =
        "__" + reducer.str() + std::to_string(++listCompCounter);
    std::string flag =
        "__" + reducer.str() + "seen" + std::to_string(listCompCounter);
    std::string element =
        "__" + reducer.str() + "el" + std::to_string(listCompCounter);
    values[tmp] = Value{placeholder,
                        placeholder.getType()};
    types.bindSymbol(tmp, placeholder.getType());
    // The seen-flag is an INT (0/1): loop-carried bool contract block
    // arguments have no boxed physical form yet.
    mlir::Type flagType = types.literal("0");
    auto flagInit = py::IntConstantOp::create(builder, loc(expr), flagType,
                                              builder.getStringAttr("0"));
    values[flag] = Value{flagInit.getResult(), flagType};
    types.bindSymbol(flag, flagType);
    auto nameNode = [&](const std::string &id) {
      parser::NodePtr node = parser::makeNode("Name", expr.range);
      parser::addField(*node, "id", id);
      return node;
    };
    parser::NodePtr tmpName = nameNode(tmp);
    parser::NodePtr flagName = nameNode(flag);
    parser::NodePtr elementName = nameNode(element);
    // if __seen: (if el >/< __acc: __acc = el) else: __acc = el; __seen = True
    parser::NodePtr assignAcc = parser::makeNode("Assign", expr.range);
    parser::addField(*assignAcc, "targets",
                     std::vector<parser::NodePtr>{tmpName});
    parser::addField(*assignAcc, "value", elementName);
    parser::NodePtr cmpOp = parser::makeNode(
        reducer == "max" ? "Gt" : "Lt", expr.range);
    parser::NodePtr compare = parser::makeNode("Compare", expr.range);
    parser::addField(*compare, "left", elementName);
    parser::addField(*compare, "ops", std::vector<parser::NodePtr>{cmpOp});
    parser::addField(*compare, "comparators",
                     std::vector<parser::NodePtr>{tmpName});
    parser::NodePtr better = parser::makeNode("If", expr.range);
    parser::addField(*better, "test", compare);
    parser::addField(*better, "body",
                     std::vector<parser::NodePtr>{assignAcc});
    parser::addField(*better, "orelse", std::vector<parser::NodePtr>{});
    parser::NodePtr trueValue = parser::makeNode("Constant", expr.range);
    parser::addField(*trueValue, "value", std::int64_t{1});
    parser::NodePtr markSeen = parser::makeNode("Assign", expr.range);
    parser::addField(*markSeen, "targets",
                     std::vector<parser::NodePtr>{flagName});
    parser::addField(*markSeen, "value", trueValue);
    parser::NodePtr seenSwitch = parser::makeNode("If", expr.range);
    parser::addField(*seenSwitch, "test", flagName);
    parser::addField(*seenSwitch, "body",
                     std::vector<parser::NodePtr>{better});
    parser::addField(*seenSwitch, "orelse",
                     std::vector<parser::NodePtr>{assignAcc, markSeen});
    parser::NodePtr loop = parser::makeNode("For", expr.range);
    parser::addField(*loop, "target", elementName);
    parser::addField(*loop, "iter", reducerArgs->front());
    parser::addField(*loop, "body",
                     std::vector<parser::NodePtr>{seenSwitch});
    parser::addField(*loop, "orelse", std::vector<parser::NodePtr>{});
    // if not __seen: raise ValueError("max()/min() iterable argument is empty")
    parser::NodePtr notSeenOp = parser::makeNode("Not", expr.range);
    parser::NodePtr notSeen = parser::makeNode("UnaryOp", expr.range);
    parser::addField(*notSeen, "op", notSeenOp);
    parser::addField(*notSeen, "operand", flagName);
    parser::NodePtr message = parser::makeNode("Constant", expr.range);
    parser::addField(*message, "value",
                     reducer.str() + "() iterable argument is empty");
    parser::NodePtr errorCall = parser::makeNode("Call", expr.range);
    parser::addField(*errorCall, "func", nameNode("ValueError"));
    parser::addField(*errorCall, "args",
                     std::vector<parser::NodePtr>{message});
    parser::addField(*errorCall, "keywords",
                     std::vector<parser::NodePtr>{});
    parser::NodePtr raiseNode = parser::makeNode("Raise", expr.range);
    parser::addField(*raiseNode, "exc", errorCall);
    parser::NodePtr emptyGuard = parser::makeNode("If", expr.range);
    parser::addField(*emptyGuard, "test", notSeen);
    parser::addField(*emptyGuard, "body",
                     std::vector<parser::NodePtr>{raiseNode});
    parser::addField(*emptyGuard, "orelse",
                     std::vector<parser::NodePtr>{});
    std::optional<Value> priorElement;
    if (auto found = values.find(element); found != values.end())
      priorElement = found->second;
    emitFor(*loop);
    emitStatement(*emptyGuard);
    auto built = values.find(tmp);
    if (built == values.end() || !built->second.value) {
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, expr.range.start,
          "cannot lower " + reducer.str() + "() over this iterable"});
      return emitNone(expr);
    }
    Value result = built->second;
    values.erase(tmp);
    values.erase(flag);
    if (priorElement)
      values[element] = *priorElement;
    else
      values.erase(element);
    return result;
  }
  if (reducerArgs && reducerArgs->size() == 1 && reducerArgs->front() &&
      (!reducerKeywords || reducerKeywords->empty()) &&
      (reducer == "sum" || reducer == "any" || reducer == "all")) {
    std::string tmp =
        "__" + reducer.str() + std::to_string(++listCompCounter);
    std::string element = "__" + reducer.str() + "el" +
                          std::to_string(listCompCounter);
    if (reducer == "sum") {
      mlir::Type zeroType = types.literal("0");
      auto zero = py::IntConstantOp::create(builder, loc(expr), zeroType,
                                            builder.getStringAttr("0"));
      values[tmp] = Value{zero.getResult(), zeroType};
      types.bindSymbol(tmp, zeroType);
    } else {
      bool initial = reducer == "all";
      mlir::Type initType = types.literal(initial ? "True" : "False");
      auto init = py::BoolConstantOp::create(
          builder, loc(expr), initType, builder.getBoolAttr(initial));
      values[tmp] = Value{init.getResult(), initType};
      types.bindSymbol(tmp, initType);
    }
    parser::NodePtr tmpName = parser::makeNode("Name", expr.range);
    parser::addField(*tmpName, "id", tmp);
    parser::NodePtr elementName = parser::makeNode("Name", expr.range);
    parser::addField(*elementName, "id", element);
    std::vector<parser::NodePtr> body;
    if (reducer == "sum") {
      // <tmp> = <tmp> + <element>
      parser::NodePtr addOp = parser::makeNode("Add", expr.range);
      parser::NodePtr add = parser::makeNode("BinOp", expr.range);
      parser::addField(*add, "left", tmpName);
      parser::addField(*add, "op", addOp);
      parser::addField(*add, "right", elementName);
      parser::NodePtr assign = parser::makeNode("Assign", expr.range);
      parser::addField(*assign, "targets",
                       std::vector<parser::NodePtr>{tmpName});
      parser::addField(*assign, "value", add);
      body.push_back(assign);
    } else {
      // any: if <element>: <tmp> = True; break
      // all: if not <element>: <tmp> = False; break
      bool flipped = reducer == "any";
      parser::NodePtr flippedValue =
          parser::makeNode("Constant", expr.range);
      parser::addField(*flippedValue, "value", flipped);
      parser::NodePtr assign = parser::makeNode("Assign", expr.range);
      parser::addField(*assign, "targets",
                       std::vector<parser::NodePtr>{tmpName});
      parser::addField(*assign, "value", flippedValue);
      parser::NodePtr breakNode = parser::makeNode("Break", expr.range);
      parser::NodePtr test = elementName;
      if (reducer == "all") {
        parser::NodePtr notOp = parser::makeNode("Not", expr.range);
        parser::NodePtr negated = parser::makeNode("UnaryOp", expr.range);
        parser::addField(*negated, "op", notOp);
        parser::addField(*negated, "operand", elementName);
        test = negated;
      }
      parser::NodePtr guard = parser::makeNode("If", expr.range);
      parser::addField(*guard, "test", test);
      parser::addField(*guard, "body",
                       std::vector<parser::NodePtr>{assign, breakNode});
      parser::addField(*guard, "orelse", std::vector<parser::NodePtr>{});
      body.push_back(guard);
    }
    parser::NodePtr loop = parser::makeNode("For", expr.range);
    parser::addField(*loop, "target", elementName);
    parser::addField(*loop, "iter", reducerArgs->front());
    parser::addField(*loop, "body", body);
    parser::addField(*loop, "orelse", std::vector<parser::NodePtr>{});
    std::optional<Value> priorElement;
    if (auto found = values.find(element); found != values.end())
      priorElement = found->second;
    emitFor(*loop);
    auto built = values.find(tmp);
    if (built == values.end() || !built->second.value) {
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, expr.range.start,
          "cannot lower " + reducer.str() + "() over this iterable"});
      return emitNone(expr);
    }
    Value result = built->second;
    values.erase(tmp);
    if (priorElement)
      values[element] = *priorElement;
    else
      values.erase(element);
    return result;
  }
  return std::nullopt;
}

std::optional<Value>
ModuleEmitter::tryEmitLenCall(const parser::Node &expr,
                              const parser::Node *calleeNode) {
  if (!calleeNode || calleeNode->kind != "Name" ||
      ast::nameSpelling(*calleeNode) != "len")
    return std::nullopt;
  const auto *args = ast::nodeList(expr, "args");
  if (args && args->size() == 1) {
    Value input = emitExpr(args->front().get());
    if (std::optional<MethodBinding> method =
            lookupClassMethod(input.type, "__len__"))
      return emitInlineOperatorCall(expr, input, *method, {});
    CallInferenceResult inference =
        types.inferMethodCallWithEvidence(input.type, "__len__", {});
    if (!requireStaticEvidence(expr, inference))
      return emitNone(expr);
    mlir::Type resultType =
        inference ? inference.resultType : types.intType();
    auto op =
        py::LenOp::create(builder, loc(expr), resultType,
                          mlir::FlatSymbolRefAttr::get(&context, "__len__"),
                          callProtocolFor(inference), input.value);
    return Value{op.getResult(), resultType};
  }
  return std::nullopt;
}

std::optional<Value>
ModuleEmitter::tryEmitNextCall(const parser::Node &expr,
                               const parser::Node *calleeNode) {
  if (!calleeNode || calleeNode->kind != "Name" ||
      ast::nameSpelling(*calleeNode) != "next")
    return std::nullopt;
  // A local binding named `next` shadows the builtin.
  if (values.find("next") != values.end())
    return std::nullopt;
  const auto *args = ast::nodeList(expr, "args");
  const auto *keywords = ast::nodeList(expr, "keywords");
  if (keywords && !keywords->empty())
    return std::nullopt;
  // Only the one-argument form; `next(it, default)` needs a union result
  // and StopIteration interception, which the static surface does not
  // provide yet.
  if (!args || args->size() != 1) {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, expr.range.start,
        "next() currently supports exactly one iterator argument"});
    return emitNone(expr);
  }
  Value receiver = emitExpr(args->front().get());
  CallInferenceResult inference =
      types.inferMethodCallWithEvidence(receiver.type, "__next__", {});
  if (!requireStaticEvidence(expr, inference))
    return emitNone(expr);
  mlir::Type resultType = inference ? inference.resultType : types.object();
  Value posPack = emitPack({});
  Value namePack = emitPack({});
  Value valuePack = emitPack({});
  auto op =
      py::CallOp::create(builder, loc(expr), mlir::TypeRange{resultType},
                         callProtocolFor(inference), receiver.value,
                         posPack.value, namePack.value, valuePack.value);
  op->setAttr("ly.bound_method", builder.getStringAttr("__next__"));
  return Value{op.getResults().front(), resultType};
}

std::optional<Value>
ModuleEmitter::tryEmitRoundCall(const parser::Node &expr,
                                const parser::Node *calleeNode) {
  if (!calleeNode || calleeNode->kind != "Name" ||
      ast::nameSpelling(*calleeNode) != "round")
    return std::nullopt;
  const auto *args = ast::nodeList(expr, "args");
  if (args && (args->size() == 1 || args->size() == 2)) {
    llvm::SmallVector<mlir::Value, 2> inputs;
    llvm::SmallVector<mlir::Type, 1> extraTypes;
    Value receiver = emitExpr(args->front().get());
    // round(int) is the identity (CPython); skipping the runtime call also
    // keeps the manifest __round__ contract at a fixed two-argument arity.
    if (args->size() == 1 &&
        types.widenLiteral(receiver.type) == types.intType())
      return coerceValue(receiver, types.intType(), expr);
    inputs.push_back(receiver.value);
    if (args->size() == 2) {
      Value ndigits = emitExpr((*args)[1].get());
      inputs.push_back(ndigits.value);
      extraTypes.push_back(ndigits.type);
    }
    CallInferenceResult inference = types.inferMethodCallWithEvidence(
        receiver.type, "__round__", extraTypes);
    if (!requireStaticEvidence(expr, inference))
      return emitNone(expr);
    mlir::Type resultType =
        inference ? inference.resultType : types.inferExpr(&expr);
    auto op =
        py::RoundOp::create(builder, loc(expr), resultType, "__round__",
                            callProtocolFor(inference), inputs);
    return Value{op.getResult(), resultType};
  }
  return std::nullopt;
}

std::optional<Value>
ModuleEmitter::tryEmitReprCall(const parser::Node &expr,
                               const parser::Node *calleeNode) {
  if (!calleeNode || calleeNode->kind != "Name")
    return std::nullopt;
  auto emitBuiltinBinding = [&](llvm::StringRef name) -> std::optional<Value> {
    std::optional<mlir::Type> type = types.lookupSymbol(name);
    if (!type) {
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, expr.range.start,
          "unresolved builtin '" + name.str() + "'"});
      return std::nullopt;
    }
    std::string binding = name.str();
    if (std::optional<std::string> canonical =
            types.lookupCanonicalBinding(name))
      binding = *canonical;
    return emitBindingRef(*calleeNode, binding, *type);
  };
  llvm::StringRef name = ast::nameSpelling(*calleeNode);
  const auto *args = ast::nodeList(expr, "args");
  const auto *keywords = ast::nodeList(expr, "keywords");
  bool builtinVisible = values.find(name) == values.end();
  bool hasKeywords = keywords && !keywords->empty();
  if (builtinVisible && args && args->size() == 1 && !hasKeywords &&
      (name == "repr" || name == "print")) {
    // Widen literals to their contract (`repr(5)` sees `builtins.int`, not
    // `literal<5>`) so the manifest `__repr__` resolves.
    mlir::Type argumentType =
        types.widenLiteral(types.inferExpr(args->front().get()));
    std::optional<Value> repr;
    if (std::optional<MethodBinding> method =
            lookupClassMethod(argumentType, "__repr__")) {
      // Source-class __repr__: inline the method body.
      Value argument = emitExpr(args->front().get());
      llvm::StringMap<Value> emptyKeywords;
      Value descriptorReceiver =
          emitDescriptorReceiver(expr, argument, *method);
      repr = emitInlineMethodBody(expr, descriptorReceiver,
                                  methodBindingBindsReceiver(*method), *method,
                                  {}, emptyKeywords);
    } else if (name == "repr") {
      // Manifest-typed receiver (int/str/...): emit py.repr dispatch, the
      // same manifest path `str()` uses (avoid altering `print`'s existing
      // function-binding lowering, which this special case only optimizes).
      if (CallInferenceResult inference = types.inferMethodCallWithEvidence(
              argumentType, "__repr__", {})) {
        Value argument = coerceValue(emitExpr(args->front().get()),
                                     argumentType, expr);
        mlir::Type resultType = types.contract("builtins.str");
        auto op = py::ReprOp::create(
            builder, loc(expr), resultType,
            mlir::FlatSymbolRefAttr::get(&context, "__repr__"),
            mlir::TypeAttr::get(callProtocolFor(inference)), argument.value);
        repr = Value{op.getResult(), resultType};
      }
    }
    if (repr) {
      if (name == "repr")
        return *repr;
      CallOperands operands =
          emitCallOperands(expr, {*repr}, /*includeAstArguments=*/false);
      std::optional<Value> builtin = emitBuiltinBinding(name);
      if (!builtin)
        return emitNone(expr);
      return emitCallableDispatch(expr, *builtin, operands, types.none());
    }
  }
  return std::nullopt;
}

std::optional<Value>
ModuleEmitter::rejectStubSourceCall(const parser::Node &expr,
                                    llvm::StringRef symbol,
                                    bool instantiation) {
  if (!isStubSourceModuleSymbol(symbol))
    return std::nullopt;
  diagnostics.push_back(parser::Diagnostic{
      parser::Severity::Error, expr.range.start,
      "cannot " + std::string(instantiation ? "instantiate" : "call") +
          " stub-only import '" + symbol.str() + "' at runtime"});
  return emitNone(expr);
}

} // namespace lython::emitter
