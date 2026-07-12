#include "EmitterCore.h"
#include "EmitterPyOps.h"
#include "EmitterSupport.h"

#include "AstAccess.h"
#include "ClosureAnalysis.h"
#include "PyProtocols.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/StringSet.h"

#include <utility>

namespace lython::emitter {
namespace {

mlir::Attribute sourceExprAttr(mlir::Builder &builder,
                               const parser::Node *node) {
  auto dict = [&](llvm::StringRef kind,
                  llvm::ArrayRef<mlir::NamedAttribute> extra = {}) {
    llvm::SmallVector<mlir::NamedAttribute, 4> attrs;
    attrs.push_back(builder.getNamedAttr("kind", builder.getStringAttr(kind)));
    attrs.append(extra.begin(), extra.end());
    return builder.getDictionaryAttr(attrs);
  };

  if (!node)
    return dict("none");
  if (node->kind == "Constant") {
    if (ast::isNoneField(*node, "value"))
      return dict("constant.none");
    if (auto value = ast::boolean(*node, "value"))
      return dict("constant.bool",
                  {builder.getNamedAttr("value", builder.getBoolAttr(*value))});
    if (auto value = ast::integer(*node, "value"))
      return dict("constant.int",
                  {builder.getNamedAttr(
                      "value", builder.getStringAttr(std::to_string(*value)))});
    if (auto value = ast::floating(*node, "value"))
      return dict(
          "constant.float",
          {builder.getNamedAttr("value", builder.getF64FloatAttr(*value))});
    if (auto value = ast::string(*node, "value"))
      return dict("constant.str", {builder.getNamedAttr(
                                      "value", builder.getStringAttr(*value))});
    if (const auto *fieldValue = ast::field(*node, "value"))
      if (const auto *big = std::get_if<parser::BigInteger>(fieldValue))
        return dict("constant.int",
                    {builder.getNamedAttr(
                        "value", builder.getStringAttr(big->decimal))});
    return dict("unsupported", {builder.getNamedAttr(
                                   "node", builder.getStringAttr("Constant"))});
  }
  if (node->kind == "Name" || node->kind == "Attribute") {
    std::string qualified = ast::qualifiedName(node);
    if (qualified.empty())
      qualified = std::string(ast::nameSpelling(*node));
    return dict("ref", {builder.getNamedAttr(
                           "name", builder.getStringAttr(qualified))});
  }
  if (node->kind == "List" || node->kind == "Tuple") {
    llvm::SmallVector<mlir::Attribute, 8> values;
    if (const auto *elts = ast::nodeList(*node, "elts"))
      for (const parser::NodePtr &element : *elts)
        values.push_back(sourceExprAttr(builder, element.get()));
    return dict(node->kind == "List" ? "list" : "tuple",
                {builder.getNamedAttr("elts", builder.getArrayAttr(values))});
  }
  if (node->kind == "Call") {
    llvm::SmallVector<mlir::Attribute, 8> args;
    if (const auto *argNodes = ast::nodeList(*node, "args"))
      for (const parser::NodePtr &arg : *argNodes)
        args.push_back(sourceExprAttr(builder, arg.get()));
    llvm::SmallVector<mlir::NamedAttribute, 3> attrs;
    attrs.push_back(builder.getNamedAttr(
        "callee", sourceExprAttr(builder, ast::node(*node, "func"))));
    attrs.push_back(builder.getNamedAttr("args", builder.getArrayAttr(args)));
    return dict("call", attrs);
  }
  if (node->kind == "BinOp") {
    llvm::SmallVector<mlir::NamedAttribute, 4> attrs;
    attrs.push_back(builder.getNamedAttr(
        "op", builder.getStringAttr(ast::node(*node, "op")
                                        ? ast::node(*node, "op")->kind
                                        : std::string())));
    attrs.push_back(builder.getNamedAttr(
        "left", sourceExprAttr(builder, ast::node(*node, "left"))));
    attrs.push_back(builder.getNamedAttr(
        "right", sourceExprAttr(builder, ast::node(*node, "right"))));
    return dict("binop", attrs);
  }

  return dict("unsupported", {builder.getNamedAttr(
                                 "node", builder.getStringAttr(node->kind))});
}

std::string sourceMethodSymbolName(llvm::StringRef className,
                                   llvm::StringRef methodName,
                                   const parser::Node &method) {
  return (llvm::Twine("__ly_method$") + sanitizedSymbolPart(className) + "$" +
          sanitizedSymbolPart(methodName) + "$" +
          llvm::Twine(method.range.start.line) + "_" +
          llvm::Twine(method.range.start.column))
      .str();
}

} // namespace

std::optional<MethodBinding>
ModuleEmitter::lookupClassMethod(mlir::Type receiverType,
                                 llvm::StringRef methodName) const {
  if (auto typeObject = mlir::dyn_cast_if_present<py::TypeType>(receiverType))
    receiverType = typeObject.getInstanceType();
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
  if (auto unionType = mlir::dyn_cast_if_present<py::UnionType>(receiverType)) {
    mlir::Type common;
    for (mlir::Type member : unionType.getMemberTypes()) {
      std::optional<mlir::Type> field = lookupClassField(member, fieldName);
      if (!field)
        return std::nullopt;
      if (!common) {
        common = *field;
        continue;
      }
      if (common != *field)
        return std::nullopt;
    }
    return common ? std::optional<mlir::Type>(common) : std::nullopt;
  }
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

std::optional<mlir::Type>
ModuleEmitter::lookupClassStaticAttr(mlir::Type receiverType,
                                     llvm::StringRef attrName) const {
  if (auto typeObject = mlir::dyn_cast_if_present<py::TypeType>(receiverType))
    receiverType = typeObject.getInstanceType();
  auto contract = mlir::dyn_cast_if_present<py::ContractType>(receiverType);
  if (!contract)
    return std::nullopt;
  auto classAttrs = classStaticAttrBindings.find(contract.getContractName());
  if (classAttrs == classStaticAttrBindings.end())
    return std::nullopt;
  auto attr = classAttrs->second.find(attrName);
  if (attr == classAttrs->second.end())
    return std::nullopt;
  return attr->second;
}

bool ModuleEmitter::methodBindingBindsReceiver(
    const MethodBinding &method) const {
  return method.kind == "instance" || method.kind == "class" ||
         method.kind == "classmethod";
}

Value ModuleEmitter::emitDescriptorReceiver(const parser::Node &anchor,
                                            Value receiver,
                                            const MethodBinding &method) {
  if (method.kind != "class" && method.kind != "classmethod")
    return receiver;
  if (mlir::isa<py::TypeType>(receiver.type))
    return receiver;
  mlir::Type classType = types.typeObject(receiver.type);
  auto classObject =
      py::TypeObjectOp::create(builder, loc(anchor), classType, receiver.type);
  return {classObject.getResult(), classType};
}

void ModuleEmitter::emitClassContract(const parser::Node &classDef,
                                      llvm::StringRef symbolName) {
  auto name = ast::string(classDef, "name");
  if (!name)
    return;
  std::string classSymbol =
      symbolName.empty() ? std::string(*name) : symbolName.str();
  llvm::StringRef contractName(classSymbol);

  llvm::SmallVector<llvm::StringRef, 4> bases;
  if (const auto *baseNodes = ast::nodeList(classDef, "bases")) {
    for (const parser::NodePtr &base : *baseNodes) {
      if (!base)
        continue;
      std::string qualified = ast::qualifiedName(base.get());
      if (!qualified.empty()) {
        bases.push_back(builder.getStringAttr(qualified).getValue());
        continue;
      }
      bases.push_back(ast::nameSpelling(*base));
    }
  }

  llvm::SmallVector<std::string, 8> fieldNames;
  llvm::SmallVector<mlir::Type, 8> fieldTypes;
  collectClassFields(classDef, fieldNames, fieldTypes);

  // Manifest fields_spec (ly.typing.fields_spec, e.g. ctypes
  // Structure/Union): a class assignment named by the spec declares the
  // aggregate's fields; each field reads/writes as its declared class's
  // via_base value type (falling back to the declared class itself for
  // nested aggregates). Registering them as ordinary class fields also
  // gives the subclass its positional field constructor.
  {
    const py::protocols::Table &table = py::protocols::Table::get(context);
    std::optional<std::pair<std::string, std::string>> spec;
    for (llvm::StringRef base : bases)
      if ((spec = table.aggregateFieldsSpec(base)))
        break;
    const auto *body = spec ? ast::nodeList(classDef, "body") : nullptr;
    if (body)
      for (const parser::NodePtr &statement : *body) {
        if (!statement || statement->kind != "Assign")
          continue;
        const auto *targets = ast::nodeList(*statement, "targets");
        if (!targets || targets->size() != 1 || !targets->front() ||
            targets->front()->kind != "Name" ||
            ast::nameSpelling(*targets->front()) != spec->first)
          continue;
        const parser::Node *value = ast::node(*statement, "value");
        const auto *entries = value ? ast::nodeList(*value, "elts") : nullptr;
        if (!entries)
          continue;
        for (const parser::NodePtr &entry : *entries) {
          if (!entry)
            continue;
          const auto *pair = ast::nodeList(*entry, "elts");
          if (!pair || pair->size() != 2 || !(*pair)[0] || !(*pair)[1])
            continue;
          auto fieldName = ast::string(*(*pair)[0], "value");
          if (!fieldName)
            continue;
          mlir::Type declared = types.inferExpr((*pair)[1].get());
          if (auto typeObject =
                  mlir::dyn_cast_if_present<py::TypeType>(declared)) {
            fieldNames.push_back(std::string(*fieldName));
            fieldTypes.push_back(
                table
                    .conversionTypeViaBase(typeObject.getInstanceType(),
                                           spec->second)
                    .value_or(typeObject.getInstanceType()));
          }
        }
      }
  }

  llvm::StringMap<mlir::Type> &registeredFields =
      classFieldBindings[contractName];
  registeredFields.clear();
  llvm::SmallVector<std::string, 8> &registeredOrder =
      classFieldOrders[contractName];
  registeredOrder.assign(fieldNames.begin(), fieldNames.end());
  for (auto [fieldName, fieldType] : llvm::zip_equal(fieldNames, fieldTypes))
    registeredFields[fieldName] = fieldType;

  llvm::SmallVector<std::string, 8> methodNames;
  llvm::SmallVector<std::string, 8> methodKinds;
  llvm::SmallVector<std::string, 8> methodSymbols;
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
      if (*methodName == "__new__" && kind == "instance")
        kind = "class";
      std::optional<llvm::StringRef> receiverName;
      if (kind == "instance")
        receiverName = "self";
      else if (kind == "class" || kind == "classmethod")
        receiverName = "cls";
      FunctionSignature bodySig = types.functionSignature(
          *statement,
          kind == "static" ? std::optional<llvm::StringRef>() : receiverName);
      if (kind == "instance")
        replaceSelfInSignature(bodySig, types.contract(contractName), types);
      else if (kind == "class" || kind == "classmethod") {
        replaceSelfInSignature(
            bodySig, types.typeObject(types.contract(contractName)), types);
        if (!bodySig.positionalTypes.empty()) {
          bodySig.positionalTypes.front() =
              types.typeObject(types.contract(contractName));
          types.refreshCallable(bodySig);
        }
      }
      methodNames.push_back(std::string(*methodName));
      methodKinds.push_back(kind);
      methodContracts.push_back(bodySig.publicCallable);

      std::string symbolName =
          sourceMethodSymbolName(contractName, *methodName, *statement);
      methodSymbols.push_back(symbolName);
      if (kind != "class" && kind != "classmethod")
        emitCallableFunction(*statement, symbolName, bodySig, {},
                             /*isLambda=*/false);
      classMethodBindings[contractName][*methodName] =
          MethodBinding{statement.get(), bodySig,
                        bodySig,         kind,
                        symbolName,      statement->kind == "AsyncFunctionDef"};
    }
  }

  mlir::OperationState state(loc(classDef), py::ClassOp::getOperationName());
  state.addAttribute(mlir::SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(contractName));
  state.addAttribute("base_names", stringArray(builder, bases));
  state.addAttribute("field_names", stringArray(builder, fieldNames));
  state.addAttribute("field_types", typeArray(builder, fieldTypes));
  state.addAttribute("field_contract_types", typeArray(builder, fieldTypes));
  state.addAttribute("method_names", stringArray(builder, methodNames));
  state.addAttribute("method_contracts", typeArray(builder, methodContracts));
  state.addAttribute("method_kinds", stringArray(builder, methodKinds));
  state.addAttribute("method_symbols", stringArray(builder, methodSymbols));

  llvm::SmallVector<std::string, 8> staticAttrNames;
  llvm::SmallVector<mlir::Attribute, 8> staticAttrValues;
  llvm::SmallVector<mlir::Type, 8> staticAttrTypes;
  collectStaticClassAssignments(classDef, staticAttrNames, staticAttrValues,
                                &staticAttrTypes);
  llvm::StringMap<mlir::Type> &registeredStaticAttrs =
      classStaticAttrBindings[contractName];
  registeredStaticAttrs.clear();
  for (auto [attrName, attrType] :
       llvm::zip_equal(staticAttrNames, staticAttrTypes))
    registeredStaticAttrs[attrName] = attrType;
  if (!staticAttrNames.empty()) {
    state.addAttribute("class_static_attr_names",
                       stringArray(builder, staticAttrNames));
    state.addAttribute("class_static_attr_values",
                       builder.getArrayAttr(staticAttrValues));
  }

  state.addRegion();
  mlir::Operation *op = builder.create(state);
  op->getRegion(0).push_back(new mlir::Block);

  py::protocols::ProtocolInfo protocolInfo;
  if (const auto *body = ast::nodeList(classDef, "body")) {
    for (const parser::NodePtr &statement : *body) {
      if (!statement || statement->kind != "Assign")
        continue;
      const auto *targets = ast::nodeList(*statement, "targets");
      if (!targets || targets->size() != 1 || !targets->front() ||
          targets->front()->kind != "Name" ||
          ast::nameSpelling(*targets->front()) != "__match_args__")
        continue;
      const parser::Node *value = ast::node(*statement, "value");
      const auto *elts =
          value && value->kind == "Tuple" ? ast::nodeList(*value, "elts")
                                          : nullptr;
      std::vector<std::string> names;
      bool wellFormed = elts != nullptr;
      if (elts)
        for (const parser::NodePtr &element : *elts) {
          std::optional<std::string_view> text =
              element && element->kind == "Constant"
                  ? ast::string(*element, "value")
                  : std::nullopt;
          if (!text) {
            wellFormed = false;
            break;
          }
          names.emplace_back(*text);
        }
      if (wellFormed)
        protocolInfo.matchArgs = std::move(names);
      break;
    }
  }
  for (llvm::StringRef base : bases)
    protocolInfo.bases.push_back(py::protocols::ProtocolBase{base.str(), {}});
  for (auto [fieldName, fieldType] : llvm::zip_equal(fieldNames, fieldTypes))
    protocolInfo.fields[fieldName] = fieldType;
  for (auto [methodName, methodContract] :
       llvm::zip_equal(methodNames, methodContracts)) {
    std::string registeredMethodName = methodName;
    auto pushSignature = [&](py::CallableType signature) {
      if (!signature)
        return;
      py::protocols::ProtocolMethod method;
      method.signature = signature;
      method.mayThrow = true;
      protocolInfo.methods[registeredMethodName].push_back(method);
    };
    if (auto signature =
            mlir::dyn_cast_if_present<py::CallableType>(methodContract)) {
      pushSignature(signature);
    } else if (auto overload = mlir::dyn_cast_if_present<py::OverloadType>(
                   methodContract)) {
      for (mlir::Type candidate : overload.getCandidateTypes())
        pushSignature(mlir::dyn_cast_if_present<py::CallableType>(candidate));
    }
  }
  py::protocols::Table::getMutable(context).registerClass(
      contractName, std::move(protocolInfo));
}

void ModuleEmitter::collectStaticClassAssignments(
    const parser::Node &classDef, llvm::SmallVectorImpl<std::string> &names,
    llvm::SmallVectorImpl<mlir::Attribute> &values,
    llvm::SmallVectorImpl<mlir::Type> *typesOut) {
  mlir::Builder attrBuilder(&context);
  auto appendStaticAttr = [&](llvm::StringRef name, const parser::Node *value,
                              mlir::Type annotatedType = {}) {
    mlir::Type valueType = annotatedType;
    if (!valueType)
      valueType = types.inferExpr(value);
    if (typesOut && !valueType) {
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error,
          value ? value->range.start : classDef.range.start,
          "class static attribute '" + name.str() +
              "' requires a statically inferred type"});
      return;
    }
    names.push_back(std::string(name));
    values.push_back(sourceExprAttr(attrBuilder, value));
    if (typesOut)
      typesOut->push_back(valueType);
  };
  if (const auto *body = ast::nodeList(classDef, "body")) {
    for (const parser::NodePtr &statement : *body) {
      if (!statement)
        continue;
      if (statement->kind == "Assign") {
        const auto *targets = ast::nodeList(*statement, "targets");
        if (!targets || targets->size() != 1 || !targets->front() ||
            targets->front()->kind != "Name")
          continue;
        appendStaticAttr(ast::nameSpelling(*targets->front()),
                         ast::node(*statement, "value"));
        continue;
      }
      if (statement->kind == "AnnAssign") {
        const parser::Node *target = ast::node(*statement, "target");
        const parser::Node *value = ast::node(*statement, "value");
        if (!target || target->kind != "Name" || !value)
          continue;
        appendStaticAttr(
            ast::nameSpelling(*target), value,
            types.annotationType(ast::node(*statement, "annotation")));
      }
    }
  }
}

void ModuleEmitter::collectStaticModuleAssignments(
    const parser::Node &moduleNode, llvm::SmallVectorImpl<std::string> &names,
    llvm::SmallVectorImpl<mlir::Attribute> &values) const {
  mlir::Builder attrBuilder(&context);
  if (const auto *body = ast::nodeList(moduleNode, "body")) {
    for (const parser::NodePtr &statement : *body) {
      if (!statement || statement->kind != "Assign")
        continue;
      const auto *targets = ast::nodeList(*statement, "targets");
      if (!targets || targets->size() != 1 || !targets->front() ||
          targets->front()->kind != "Name")
        continue;
      names.push_back(std::string(ast::nameSpelling(*targets->front())));
      values.push_back(
          sourceExprAttr(attrBuilder, ast::node(*statement, "value")));
    }
  }
}

void ModuleEmitter::collectModuleGlobals(const parser::Node &moduleNode) {
  // Opt-in: an int-annotated module-level assignment (`NAME: int = ...`)
  // becomes a storage-backed mutable global. Plain `NAME = expr` at module
  // scope keeps its value-binding behavior (module-scope constants).
  const auto *body = ast::nodeList(moduleNode, "body");
  if (!body)
    return;
  for (const parser::NodePtr &statement : *body) {
    if (!statement || statement->kind != "AnnAssign")
      continue;
    const parser::Node *target = ast::node(*statement, "target");
    if (!target || target->kind != "Name")
      continue;
    mlir::Type annotated =
        types.annotationType(ast::node(*statement, "annotation"));
    if (types.widenLiteral(annotated) != types.intType())
      continue;
    llvm::StringRef name = ast::nameSpelling(*target);
    moduleGlobals[name] = types.intType();
    types.bindSymbol(name, types.intType());
  }
}

bool ModuleEmitter::isModuleGlobalRead(llvm::StringRef name) const {
  // A read resolves to the module global unless a local (function-scope)
  // binding shadows it.
  return moduleGlobals.count(name) && values.find(name) == values.end();
}

bool ModuleEmitter::isModuleGlobalWrite(llvm::StringRef name) const {
  if (!moduleGlobals.count(name))
    return false;
  // Module scope always writes the global; a function writes it only when it
  // declared `global NAME` (otherwise the assignment makes a local).
  return atModuleScope || currentGlobalDecls.count(name);
}

void ModuleEmitter::collectClassFields(
    const parser::Node &classDef,
    llvm::SmallVectorImpl<std::string> &fieldNames,
    llvm::SmallVectorImpl<mlir::Type> &fieldTypes) {
  auto setField = [&](llvm::StringRef name, mlir::Type type,
                      bool overwriteExisting, const parser::Node &anchor) {
    if (name.empty())
      return;
    if (!type) {
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, anchor.range.start,
          "class field '" + name.str() +
              "' requires a statically inferred type"});
      return;
    }
    mlir::Type storedType = types.widenLiteral(type);
    for (auto [index, existing] : llvm::enumerate(fieldNames)) {
      if (existing != name)
        continue;
      if (overwriteExisting)
        fieldTypes[index] = storedType;
      return;
    }
    fieldNames.push_back(name.str());
    fieldTypes.push_back(storedType);
  };

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
          if (const parser::Node *annotation = ast::node(*arg, "annotation"))
            argTypes[name] = types.annotationType(annotation);
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
      setField(*attr, type, /*overwriteExisting=*/false, target);
  };

  if (const auto *body = ast::nodeList(classDef, "body")) {
    for (const parser::NodePtr &statement : *body) {
      if (!statement || statement->kind != "AnnAssign")
        continue;
      const parser::Node *target = ast::node(*statement, "target");
      if (!target || target->kind != "Name" || ast::node(*statement, "value"))
        continue;
      setField(ast::nameSpelling(*target),
               types.annotationType(ast::node(*statement, "annotation")),
               /*overwriteExisting=*/true, *statement);
    }

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
            mlir::Type valueType;
            if (value && value->kind == "Name") {
              auto found = initArgTypes.find(ast::nameSpelling(*value));
              if (found != initArgTypes.end())
                valueType = found->second;
            } else {
              valueType = types.inferExpr(value);
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

Value ModuleEmitter::emitInlineOperatorCall(const parser::Node &anchor,
                                            Value receiver,
                                            const MethodBinding &method,
                                            llvm::ArrayRef<Value> positional) {
  if (!method.method)
    return emitNone(anchor);
  Value descriptorReceiver = emitDescriptorReceiver(anchor, receiver, method);
  bool bindReceiver = methodBindingBindsReceiver(method);
  if (method.kind == "instance" && mlir::isa<py::TypeType>(receiver.type))
    bindReceiver = false;
  llvm::StringMap<Value> keywords;
  return emitInlineMethodBody(anchor, descriptorReceiver, bindReceiver, method,
                              positional, keywords);
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

  Value descriptorReceiver = emitDescriptorReceiver(expr, receiver, method);
  bool bindReceiver = methodBindingBindsReceiver(method);
  if (method.kind == "instance" && mlir::isa<py::TypeType>(receiver.type))
    bindReceiver = false;
  return emitInlineMethodBody(expr, descriptorReceiver, bindReceiver, method,
                              positional, keywords);
}

Value ModuleEmitter::emitInlineMethodBody(
    const parser::Node &anchor, Value receiver, bool bindDescriptorReceiver,
    const MethodBinding &method, llvm::ArrayRef<Value> positional,
    const llvm::StringMap<Value> &keywords) {
  if (!method.method)
    return emitNone(anchor);
  const FunctionSignature &sig =
      method.bodySignature.callable ? method.bodySignature : method.signature;
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
  if (bindDescriptorReceiver && !sig.positionalNames.empty()) {
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

  const parser::Node *arguments = ast::node(*method.method, "args");
  llvm::SmallVector<const parser::Node *, 8> positionalNodes;
  if (arguments)
    positionalNodes = positionalArgumentNodes(*arguments);
  const auto *defaults =
      arguments ? ast::nodeList(*arguments, "defaults") : nullptr;
  const auto *kwDefaults =
      arguments ? ast::nodeList(*arguments, "kw_defaults") : nullptr;
  unsigned firstPositionalDefault =
      defaults && defaults->size() <= positionalNodes.size()
          ? positionalNodes.size() - defaults->size()
          : positionalNodes.size();
  auto positionalDefault = [&](unsigned index) -> const parser::Node * {
    if (!defaults || index < firstPositionalDefault)
      return nullptr;
    unsigned defaultIndex = index - firstPositionalDefault;
    if (defaultIndex >= defaults->size())
      return nullptr;
    return (*defaults)[defaultIndex].get();
  };
  auto reportMissing = [&](llvm::StringRef name) {
    diagnostics.push_back(
        parser::Diagnostic{parser::Severity::Error, anchor.range.start,
                           "missing required argument '" + name.str() +
                               "' for inlined class method"});
  };
  for (auto [index, name] : llvm::enumerate(sig.positionalNames)) {
    if (bound.contains(name))
      continue;
    if (const parser::Node *defaultNode =
            positionalDefault(static_cast<unsigned>(index))) {
      Value defaultValue = emitExpr(defaultNode);
      bind(name, coerceValue(defaultValue, sig.positionalTypes[index], anchor));
      continue;
    }
    reportMissing(name);
    bind(name, emitNone(anchor));
  }
  for (auto [index, name] : llvm::enumerate(sig.kwOnlyNames)) {
    if (bound.contains(name))
      continue;
    const parser::Node *defaultNode = nullptr;
    if (kwDefaults && index < kwDefaults->size())
      defaultNode = (*kwDefaults)[index].get();
    if (defaultNode) {
      Value defaultValue = emitExpr(defaultNode);
      bind(name, coerceValue(defaultValue, sig.kwOnlyTypes[index], anchor));
      continue;
    }
    reportMissing(name);
    bind(name, emitNone(anchor));
  }

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
  mlir::cf::BranchOp::create(builder, loc(anchor), bodyBlock);
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
    mlir::cf::BranchOp::create(builder, loc(anchor), continuation,
                               result.value);
  }
  builder.setInsertionPointToStart(continuation);
  return {continuation->getArgument(0), resultType};
}

Value ModuleEmitter::emitClassInstantiation(const parser::Node &expr,
                                            llvm::StringRef name,
                                            mlir::Type instanceType) {
  CallOperands operands = emitCallOperands(expr);
  if (!operands.valid) {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, expr.range.start, operands.failureReason});
    return emitNone(expr);
  }

  llvm::StringMap<Value> keywords;
  for (auto [index, keyword] : llvm::enumerate(operands.keywordTypes)) {
    if (index < operands.keywordValues.size())
      keywords[keyword.name] = operands.keywordValues[index];
  }
  if (operands.keywordValues.size() != operands.keywordTypes.size()) {
    if (const auto *keywordNodes = ast::nodeList(expr, "keywords")) {
      for (const parser::NodePtr &keyword : *keywordNodes) {
        if (keyword && ast::string(*keyword, "arg"))
          continue;
        if (!keyword)
          continue;
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, keyword->range.start,
            "variadic keyword arguments are not supported for class "
            "instantiation"});
      }
    }
  }
  bool hasUnpackedPositional = llvm::any_of(
      operands.positionalUnpacked, [](char value) { return value != 0; });
  if (hasUnpackedPositional) {
    if (const auto *args = ast::nodeList(expr, "args")) {
      for (const parser::NodePtr &arg : *args) {
        if (!arg || arg->kind != "Starred")
          continue;
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, arg->range.start,
            "starred arguments are not supported for source class "
            "instantiation"});
      }
    }
  }

  mlir::Type inferredInstanceType = types.inferClassInstantiation(
      instanceType, operands.positionalTypes, operands.keywordTypes);
  if (!inferredInstanceType) {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, expr.range.start,
        "class instantiation leaves unbound static type parameters for '" +
            name.str() + "'"});
    return emitNone(expr);
  }
  mlir::Type classType = types.typeObject(inferredInstanceType);
  auto classObject = py::TypeObjectOp::create(builder, loc(expr), classType,
                                              inferredInstanceType);
  Value posPack = emitPack(operands.positional, operands.positionalUnpacked);
  Value namePack = emitPack(operands.keywordNames);
  Value valuePack = emitPack(operands.keywordValues);

  auto newOp = py::NewOp::create(
      builder, loc(expr), inferredInstanceType,
      mlir::FlatSymbolRefAttr::get(&context, "__new__"), callableProtocol(),
      classObject.getResult(), posPack.value, namePack.value, valuePack.value);
  newOp->setAttr("ly.constructor.owner", builder.getStringAttr(name));
  if (std::optional<MethodBinding> newBinding =
          lookupClassMethod(inferredInstanceType, "__new__")) {
    if (newBinding->method)
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, newBinding->method->range.start,
          "source class __new__ bodies are not supported yet; use declared "
          "fields and __init__ for user class construction"});
    newOp->setAttr("ly.constructor.new_kind",
                   builder.getStringAttr(newBinding->kind));
  } else {
    newOp->setAttr("ly.constructor.new_kind", builder.getStringAttr("class"));
  }
  std::optional<MethodBinding> init =
      lookupClassMethod(inferredInstanceType, "__init__");
  if (init && !hasUnpackedPositional) {
    Value receiver{newOp.getInstance(), inferredInstanceType};
    Value descriptorReceiver = emitDescriptorReceiver(expr, receiver, *init);
    emitInlineMethodBody(expr, descriptorReceiver,
                         methodBindingBindsReceiver(*init), *init,
                         operands.positional, keywords);
  } else {
    bool noRuntimeInitArgs = operands.positional.empty() &&
                             operands.keywordValues.empty() &&
                             !hasUnpackedPositional;
    if (!init && noRuntimeInitArgs) {
      (void)namePack;
      (void)valuePack;
      return {newOp.getInstance(), inferredInstanceType};
    }
    bool noArgExceptionInit =
        noRuntimeInitArgs &&
        py::protocols::Table::get(context).isManifestSubclassOf(
            inferredInstanceType, "builtins.BaseException");
    if (noArgExceptionInit) {
      (void)namePack;
      (void)valuePack;
      return {newOp.getInstance(), inferredInstanceType};
    }
    CallInferenceResult initInference = types.inferMethodCallWithEvidence(
        inferredInstanceType, "__init__", operands.positionalTypes,
        operands.keywordTypes);
    mlir::Type initContract =
        initInference ? callProtocolFor(initInference) : mlir::Type();
    if (!initContract) {
      // Field-record construction: a class without a source or manifest
      // __init__ takes its declared fields positionally, every field
      // optional (ctypes Structure/Union subclasses; plain field records).
      auto order = classFieldOrders.find(name);
      auto fields = classFieldBindings.find(name);
      if (order != classFieldOrders.end() &&
          fields != classFieldBindings.end() && !order->second.empty() &&
          operands.positionalTypes.size() <= order->second.size() &&
          operands.keywordTypes.empty()) {
        llvm::SmallVector<mlir::Type, 8> positional{inferredInstanceType};
        llvm::SmallVector<mlir::StringAttr, 8> positionalNames{
            builder.getStringAttr("self")};
        llvm::SmallVector<mlir::BoolAttr, 8> positionalDefaults{
            builder.getBoolAttr(false)};
        for (const std::string &fieldName : order->second) {
          auto field = fields->second.find(fieldName);
          positional.push_back(field == fields->second.end()
                                   ? types.object()
                                   : field->second);
          positionalNames.push_back(builder.getStringAttr(fieldName));
          positionalDefaults.push_back(builder.getBoolAttr(true));
        }
        llvm::SmallVector<mlir::Type, 1> results{types.none()};
        initContract = py::CallableType::get(
            &context, positional, {}, {}, {}, results, positionalNames, {},
            positionalDefaults, {});
      }
    }
    if (!initContract) {
      if (!requireStaticEvidence(expr, initInference))
        return emitNone(expr);
      initContract = callProtocolFor(initInference);
    }
    auto initOp =
        py::InitOp::create(builder, loc(expr), types.none(),
                           mlir::FlatSymbolRefAttr::get(&context, "__init__"),
                           initContract, newOp.getInstance(),
                           posPack.value, namePack.value, valuePack.value);
    initOp->setAttr("ly.constructor.owner", builder.getStringAttr(name));
    initOp->setAttr("ly.constructor.init_kind",
                    builder.getStringAttr(init ? init->kind : "instance"));
  }
  (void)name;
  return {newOp.getInstance(), inferredInstanceType};
}

} // namespace lython::emitter
