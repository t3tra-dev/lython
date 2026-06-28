#include "EmitterCore.h"
#include "EmitterPyOps.h"
#include "EmitterSupport.h"

#include "AstAccess.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/StringSet.h"

#include <utility>

namespace lython::emitter {

std::optional<MethodBinding>
ModuleEmitter::lookupClassMethod(mlir::Type receiverType,
                                 llvm::StringRef methodName) const {
  auto contract = mlir::dyn_cast_if_present<py::ContractType>(receiverType);
  if (!contract)
    return std::nullopt;
  auto classMethods = classMethodBindings.find(contract.getContractName());
  if (classMethods == classMethodBindings.end())
    return std::nullopt;
  auto method = classMethods->second.find(methodName);
  if (method == classMethods->second.end())
    return std::nullopt;
  return method->second;
}

std::optional<mlir::Type>
ModuleEmitter::lookupClassField(mlir::Type receiverType,
                                llvm::StringRef fieldName) const {
  auto contract = mlir::dyn_cast_if_present<py::ContractType>(receiverType);
  if (!contract)
    return std::nullopt;
  auto classFields = classFieldBindings.find(contract.getContractName());
  if (classFields == classFieldBindings.end())
    return std::nullopt;
  auto field = classFields->second.find(fieldName);
  if (field == classFields->second.end())
    return std::nullopt;
  return field->second;
}

void ModuleEmitter::emitClassContract(const parser::Node &classDef) {
  auto name = ast::string(classDef, "name");
  if (!name)
    return;

  llvm::SmallVector<llvm::StringRef, 4> bases;
  if (const auto *baseNodes = ast::nodeList(classDef, "bases")) {
    for (const parser::NodePtr &base : *baseNodes)
      bases.push_back(ast::nameSpelling(*base));
  }

  llvm::StringMap<mlir::Type> fieldMap;
  collectClassFields(classDef, fieldMap);
  llvm::SmallVector<std::string, 8> fieldNames;
  llvm::SmallVector<mlir::Type, 8> fieldTypes;
  llvm::StringMap<mlir::Type> &registeredFields = classFieldBindings[*name];
  registeredFields.clear();
  for (const auto &entry : fieldMap) {
    fieldNames.push_back(entry.getKey().str());
    fieldTypes.push_back(entry.getValue());
    registeredFields[entry.getKey()] = entry.getValue();
  }

  llvm::SmallVector<std::string, 8> methodNames;
  llvm::SmallVector<std::string, 8> methodKinds;
  llvm::SmallVector<mlir::Type, 8> methodContracts;
  if (const auto *body = ast::nodeList(classDef, "body")) {
    for (const parser::NodePtr &statement : *body) {
      if (!statement || (statement->kind != "FunctionDef" &&
                         statement->kind != "AsyncFunctionDef"))
        continue;
      auto methodName = ast::string(*statement, "name");
      if (!methodName)
        continue;
      std::string kind = methodKind(*statement);
      FunctionSignature sig = types.functionSignature(
          *statement, kind == "static"
                          ? std::optional<llvm::StringRef>()
                          : std::optional<llvm::StringRef>("self"));
      if (kind == "instance")
        replaceSelfInSignature(sig, types.contract(*name), types);
      else if (kind == "class")
        replaceSelfInSignature(sig, types.typeObject(types.contract(*name)),
                               types);
      if (statement->kind == "AsyncFunctionDef")
        sig = asyncPublicSignature(sig);
      methodNames.push_back(std::string(*methodName));
      methodKinds.push_back(std::move(kind));
      methodContracts.push_back(sig.callable);

      classMethodBindings[*name][*methodName] =
          MethodBinding{statement.get(), sig};
    }
  }

  mlir::OperationState state(loc(classDef), py::ClassOp::getOperationName());
  state.addAttribute(mlir::SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(*name));
  state.addAttribute("base_names", stringArray(builder, bases));
  state.addAttribute("field_names", stringArray(builder, fieldNames));
  state.addAttribute("field_types", typeArray(builder, fieldTypes));
  state.addAttribute("field_contract_types", typeArray(builder, fieldTypes));
  state.addAttribute("method_names", stringArray(builder, methodNames));
  state.addAttribute("method_contracts", typeArray(builder, methodContracts));
  state.addAttribute("method_kinds", stringArray(builder, methodKinds));
  state.addRegion();
  mlir::Operation *op = builder.create(state);
  op->getRegion(0).push_back(new mlir::Block);
}

void ModuleEmitter::collectClassFields(
    const parser::Node &classDef, llvm::StringMap<mlir::Type> &fields) const {
  auto collectInitArgTypes = [&](const parser::Node &method,
                                 llvm::StringMap<mlir::Type> &argTypes) {
    const parser::Node *arguments = ast::node(method, "args");
    if (!arguments)
      return;
    auto collectArgs = [&](llvm::StringRef fieldName) {
      if (const auto *args = ast::nodeList(*arguments, fieldName)) {
        for (const parser::NodePtr &arg : *args) {
          if (!arg)
            continue;
          llvm::StringRef name = ast::nameSpelling(*arg);
          if (name == "self")
            continue;
          argTypes[name] = types.annotationType(ast::node(*arg, "annotation"));
        }
      }
    };
    collectArgs("posonlyargs");
    collectArgs("args");
  };

  auto collectTarget = [&](const parser::Node &target, mlir::Type type) {
    if (target.kind != "Attribute")
      return;
    const parser::Node *object = ast::node(target, "value");
    if (!object || !ast::isName(*object, "self"))
      return;
    if (auto attr = ast::string(target, "attr"))
      fields[*attr] = type ? type : types.object();
  };

  if (const auto *body = ast::nodeList(classDef, "body")) {
    for (const parser::NodePtr &method : *body) {
      if (!method || ast::nameSpelling(*method) != "__init__")
        continue;
      llvm::StringMap<mlir::Type> initArgTypes;
      collectInitArgTypes(*method, initArgTypes);
      if (const auto *stmts = ast::nodeList(*method, "body")) {
        for (const parser::NodePtr &stmt : *stmts) {
          if (!stmt)
            continue;
          if (stmt->kind == "AnnAssign") {
            collectTarget(*ast::node(*stmt, "target"),
                          types.annotationType(ast::node(*stmt, "annotation")));
          } else if (stmt->kind == "Assign") {
            const parser::Node *value = ast::node(*stmt, "value");
            mlir::Type valueType = types.inferExpr(value);
            if (value && value->kind == "Name") {
              auto found = initArgTypes.find(ast::nameSpelling(*value));
              if (found != initArgTypes.end())
                valueType = found->second;
            }
            if (const auto *targets = ast::nodeList(*stmt, "targets"))
              for (const parser::NodePtr &target : *targets)
                collectTarget(*target, valueType);
          }
        }
      }
    }
  }
}

Value ModuleEmitter::emitInlineMethodCall(const parser::Node &expr,
                                          Value receiver,
                                          const MethodBinding &method) {
  if (!method.method)
    return emitNone(expr);

  llvm::SmallVector<Value, 8> positional;
  if (const auto *args = ast::nodeList(expr, "args")) {
    for (const parser::NodePtr &arg : *args) {
      if (arg && arg->kind == "Starred") {
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, arg->range.start,
            "starred arguments are not supported for inlined class methods"});
        continue;
      }
      positional.push_back(emitExpr(arg.get()));
    }
  }

  llvm::StringMap<Value> keywords;
  if (const auto *keywordNodes = ast::nodeList(expr, "keywords")) {
    for (const parser::NodePtr &keyword : *keywordNodes) {
      if (auto name = ast::string(*keyword, "arg")) {
        keywords[*name] = emitExpr(ast::node(*keyword, "value"));
        continue;
      }
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, keyword->range.start,
          "variadic keyword arguments are not supported for inlined class "
          "methods"});
    }
  }

  return emitInlineMethodBody(expr, receiver, method, positional, keywords);
}

Value ModuleEmitter::emitInlineMethodBody(
    const parser::Node &anchor, Value receiver, const MethodBinding &method,
    llvm::ArrayRef<Value> positional, const llvm::StringMap<Value> &keywords) {
  if (!method.method)
    return emitNone(anchor);
  const FunctionSignature &sig = method.signature;
  const auto *body = ast::nodeList(*method.method, "body");
  mlir::Type resultType = sig.resultType ? sig.resultType : types.none();

  ScopedEmitterScope scope(values, types);
  llvm::StringSet<> bound;
  auto bind = [&](llvm::StringRef name, Value value) {
    values[name] = value;
    types.bindSymbol(name, value.type);
    bound.insert(name);
  };

  unsigned parameterIndex = 0;
  if (!sig.positionalNames.empty()) {
    bind(sig.positionalNames.front(), receiver);
    parameterIndex = 1;
  }

  for (Value argument : positional) {
    if (parameterIndex >= sig.positionalNames.size()) {
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, anchor.range.start,
          "too many positional arguments for inlined class method"});
      break;
    }
    bind(sig.positionalNames[parameterIndex++], argument);
  }

  auto bindKeyword = [&](llvm::StringRef name, Value value) {
    for (llvm::StringRef positionalName : sig.positionalNames) {
      if (positionalName != name)
        continue;
      if (bound.contains(name)) {
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, anchor.range.start,
            "multiple values for inlined class method argument '" + name.str() +
                "'"});
        return;
      }
      bind(name, value);
      return;
    }
    for (llvm::StringRef kwOnlyName : sig.kwOnlyNames) {
      if (kwOnlyName != name)
        continue;
      bind(name, value);
      return;
    }
    diagnostics.push_back(
        parser::Diagnostic{parser::Severity::Error, anchor.range.start,
                           "unexpected keyword argument '" + name.str() +
                               "' for inlined class method"});
  };
  for (auto &entry : keywords)
    bindKeyword(entry.getKey(), entry.getValue());

  mlir::Block *entryBlock = builder.getInsertionBlock();
  mlir::Region *region = entryBlock ? entryBlock->getParent() : nullptr;
  if (!region) {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, anchor.range.start,
        "inlined class method call requires an active insertion region"});
    return emitNone(anchor);
  }
  mlir::Block *continuation =
      entryBlock->splitBlock(builder.getInsertionPoint());
  continuation->addArgument(resultType, loc(anchor));
  mlir::Block *bodyBlock =
      builder.createBlock(region, continuation->getIterator());

  builder.setInsertionPointToEnd(entryBlock);
  builder.create<mlir::cf::BranchOp>(loc(anchor), bodyBlock);
  builder.setInsertionPointToStart(bodyBlock);
  inlineReturnContexts.push_back(InlineReturnContext{continuation, resultType});
  emitStatements(body);
  inlineReturnContexts.pop_back();
  if (!insertionBlockTerminated(builder)) {
    if (resultType != types.none()) {
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, method.method->range.start,
          "inlined class method can fall through without returning a value"});
    }
    Value none = emitNone(anchor);
    Value result = coerceValue(none, resultType, anchor);
    builder.create<mlir::cf::BranchOp>(loc(anchor), continuation, result.value);
  }
  builder.setInsertionPointToStart(continuation);
  return {continuation->getArgument(0), resultType};
}

Value ModuleEmitter::emitClassInstantiation(const parser::Node &expr,
                                            llvm::StringRef name,
                                            mlir::Type instanceType) {
  llvm::SmallVector<Value, 8> positional;
  if (const auto *args = ast::nodeList(expr, "args"))
    for (const parser::NodePtr &arg : *args)
      positional.push_back(emitExpr(arg.get()));
  llvm::SmallVector<mlir::Type, 8> positionalTypes;
  positionalTypes.reserve(positional.size());
  for (const Value &arg : positional)
    positionalTypes.push_back(arg.type);

  mlir::Type inferredInstanceType =
      types.inferClassInstantiation(instanceType, positionalTypes, {});
  mlir::Type classType = types.typeObject(inferredInstanceType);
  auto classObject = builder.create<py::TypeObjectOp>(loc(expr), classType,
                                                      inferredInstanceType);
  Value posPack = emitPack(positional);
  Value emptyNames = emitPack({});
  Value emptyValues = emitPack({});

  auto newOp = builder.create<py::NewOp>(
      loc(expr), inferredInstanceType,
      mlir::FlatSymbolRefAttr::get(&context, "__new__"), callableProtocol(),
      classObject.getResult(), posPack.value, emptyNames.value,
      emptyValues.value);
  builder.create<py::InitOp>(
      loc(expr), types.none(),
      mlir::FlatSymbolRefAttr::get(&context, "__init__"), callableProtocol(),
      newOp.getInstance(), posPack.value, emptyNames.value, emptyValues.value);
  (void)name;
  return {newOp.getInstance(), inferredInstanceType};
}

} // namespace lython::emitter
