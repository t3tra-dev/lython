#include "EmitterCore.h"
#include "EmitterPyOps.h"
#include "EmitterSupport.h"

#include "AstAccess.h"

#include "mlir/IR/BuiltinAttributes.h"

#include <optional>
#include <string>

namespace lython::emitter {

CallOperands
ModuleEmitter::emitCallOperands(const parser::Node &expr,
                                llvm::ArrayRef<Value> leadingPositional,
                                bool includeAstArguments) {
  CallOperands operands;
  for (Value value : leadingPositional) {
    operands.positional.push_back(value);
    operands.positionalUnpacked.push_back(0);
    operands.positionalTypes.push_back(value.type);
  }
  if (!includeAstArguments)
    return operands;

  if (const auto *args = ast::nodeList(expr, "args")) {
    for (const parser::NodePtr &arg : *args) {
      bool unpacked = arg && arg->kind == "Starred";
      const parser::Node *valueNode =
          unpacked ? ast::node(*arg, "value") : arg.get();
      Value value = emitExpr(valueNode);
      operands.positional.push_back(value);
      operands.positionalUnpacked.push_back(unpacked ? 1 : 0);
      if (unpacked)
        appendStarredArgumentTypes(value.type, types, operands.positionalTypes);
      else
        operands.positionalTypes.push_back(value.type);
    }
  }

  if (const auto *keywords = ast::nodeList(expr, "keywords")) {
    for (const parser::NodePtr &keyword : *keywords) {
      if (auto name = ast::string(*keyword, "arg")) {
        mlir::Type literal = types.literal("\"" + std::string(*name) + "\"");
        auto stringOp = builder.create<py::StrConstantOp>(
            loc(*keyword), literal, builder.getStringAttr(*name));
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
  Value posPack = emitPack(operands.positional, operands.positionalUnpacked);
  Value namePack = emitPack(operands.keywordNames);
  Value valuePack = emitPack(operands.keywordValues);
  CallInferenceResult inference = types.inferCallWithEvidence(
      callee.type, operands.positionalTypes, operands.keywordTypes);
  mlir::Type resultType =
      resultOverride ? resultOverride
                     : (inference ? inference.resultType : types.object());
  auto op = builder.create<py::CallOp>(loc(anchor), mlir::TypeRange{resultType},
                                       callProtocolFor(inference, callee.type),
                                       callee.value, posPack.value,
                                       namePack.value, valuePack.value);
  return {op.getResults().front(), resultType};
}

Value ModuleEmitter::emitCall(const parser::Node &expr) {
  const parser::Node *calleeNode = ast::node(expr, "func");
  std::string calleeQualified = ast::qualifiedName(calleeNode);

  if (std::optional<Value> primitive =
          emitPrimitiveConstructorCall(expr, calleeNode))
    return *primitive;
  if (std::optional<Value> primitive =
          emitPrimitiveRuntimeCall(expr, calleeNode))
    return *primitive;

  if (!calleeQualified.empty())
    if (auto cls = types.lookupClass(calleeQualified))
      return emitClassInstantiation(expr, llvm::StringRef(calleeQualified),
                                    *cls);

  if (calleeNode && calleeNode->kind == "Name") {
    llvm::StringRef name = ast::nameSpelling(*calleeNode);
    if (auto cls = types.lookupClass(name))
      return emitClassInstantiation(expr, name, *cls);
    if (name == "len") {
      const auto *args = ast::nodeList(expr, "args");
      if (args && args->size() == 1) {
        Value input = emitExpr(args->front().get());
        CallInferenceResult inference =
            types.inferMethodCallWithEvidence(input.type, "__len__", {});
        mlir::Type resultType =
            inference ? inference.resultType : types.intType();
        auto op = builder.create<py::LenOp>(
            loc(expr), resultType,
            mlir::FlatSymbolRefAttr::get(&context, "__len__"),
            callProtocolFor(inference), input.value);
        return {op.getResult(), resultType};
      }
    }
    if (name == "round") {
      const auto *args = ast::nodeList(expr, "args");
      if (args && (args->size() == 1 || args->size() == 2)) {
        llvm::SmallVector<mlir::Value, 2> inputs;
        llvm::SmallVector<mlir::Type, 1> extraTypes;
        Value receiver = emitExpr(args->front().get());
        inputs.push_back(receiver.value);
        if (args->size() == 2) {
          Value ndigits = emitExpr((*args)[1].get());
          inputs.push_back(ndigits.value);
          extraTypes.push_back(ndigits.type);
        }
        CallInferenceResult inference = types.inferMethodCallWithEvidence(
            receiver.type, "__round__", extraTypes);
        mlir::Type resultType =
            inference ? inference.resultType : types.inferExpr(&expr);
        auto op =
            builder.create<py::RoundOp>(loc(expr), resultType, "__round__",
                                        callProtocolFor(inference), inputs);
        return {op.getResult(), resultType};
      }
    }
  }

  auto emitBuiltinBinding = [&](llvm::StringRef name) -> Value {
    mlir::Type type = types.lookupSymbol(name).value_or(types.object());
    std::string binding = name.str();
    if (std::optional<std::string> canonical =
            types.lookupCanonicalBinding(name))
      binding = *canonical;
    return emitBindingRef(*calleeNode, binding, type);
  };

  if (std::optional<Value> primitiveCall =
          emitDirectPrimitiveFunctionCall(expr, calleeNode))
    return *primitiveCall;

  if (calleeNode && calleeNode->kind == "Name") {
    llvm::StringRef name = ast::nameSpelling(*calleeNode);
    const auto *args = ast::nodeList(expr, "args");
    const auto *keywords = ast::nodeList(expr, "keywords");
    bool builtinVisible = values.find(name) == values.end();
    bool hasKeywords = keywords && !keywords->empty();
    if (builtinVisible && args && args->size() == 1 && !hasKeywords &&
        (name == "repr" || name == "print")) {
      mlir::Type argumentType = types.inferExpr(args->front().get());
      if (std::optional<MethodBinding> method =
              lookupClassMethod(argumentType, "__repr__")) {
        Value argument = emitExpr(args->front().get());
        llvm::StringMap<Value> emptyKeywords;
        Value descriptorReceiver =
            emitDescriptorReceiver(expr, argument, *method);
        Value repr = emitInlineMethodBody(
            expr, descriptorReceiver, methodBindingBindsReceiver(*method),
            *method, {}, emptyKeywords);
        if (name == "repr")
          return repr;
        CallOperands operands =
            emitCallOperands(expr, {repr}, /*includeAstArguments=*/false);
        return emitCallableDispatch(expr, emitBuiltinBinding(name), operands,
                                    types.none());
      }
    }
  }

  if (calleeNode && calleeNode->kind == "Name") {
    llvm::StringRef name = ast::nameSpelling(*calleeNode);
    auto local = values.find(name);
    if (local != values.end() && local->second.boundMethod)
      return emitInlineMethodCall(expr, local->second.boundMethod->receiver,
                                  local->second.boundMethod->method);
    if (values.find(name) == values.end())
      if (std::optional<std::string> canonical =
              types.lookupCanonicalBinding(name))
        if (*canonical == "asyncio.sleep")
          if (auto symbol = types.lookupSymbol(name))
            return emitCallableDispatch(
                expr, emitBindingRef(*calleeNode, *canonical, *symbol),
                emitCallOperands(expr), types.inferExpr(&expr));
  }

  if (calleeNode && calleeNode->kind == "Attribute" &&
      !calleeQualified.empty()) {
    if (auto symbol = types.lookupSymbol(calleeQualified)) {
      std::string binding = calleeQualified;
      if (std::optional<std::string> canonical =
              types.lookupCanonicalBinding(calleeQualified))
        binding = *canonical;
      mlir::Type resultOverride =
          binding == "asyncio.sleep" ? types.inferExpr(&expr) : mlir::Type();
      return emitCallableDispatch(expr,
                                  emitBindingRef(*calleeNode, binding, *symbol),
                                  emitCallOperands(expr), resultOverride);
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
        Value posPack =
            emitPack(operands.positional, operands.positionalUnpacked);
        Value namePack = emitPack(operands.keywordNames);
        Value valuePack = emitPack(operands.keywordValues);
        CallInferenceResult inference = types.inferMethodCallWithEvidence(
            receiver.type, *methodName, operands.positionalTypes,
            operands.keywordTypes);
        mlir::Type resultType =
            inference ? inference.resultType : types.inferExpr(&expr);
        auto op = builder.create<py::CallOp>(
            loc(expr), mlir::TypeRange{resultType}, callProtocolFor(inference),
            receiver.value, posPack.value, namePack.value, valuePack.value);
        op->setAttr("ly.bound_method", builder.getStringAttr(*methodName));
        return {op.getResults().front(), resultType};
      }
    }
  }

  return emitCallableDispatch(expr, emitExpr(calleeNode),
                              emitCallOperands(expr));
}

} // namespace lython::emitter
