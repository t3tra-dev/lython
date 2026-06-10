#include "BuilderImpl.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <set>
#include <utility>

namespace lython::emitter {

namespace {

bool isSelfAttribute(const parser::Node &node) {
  if (node.kind != "Attribute")
    return false;
  const parser::NodePtr *value = nodeField(node, "value");
  if (!value || !*value || (*value)->kind != "Name")
    return false;
  const std::string *name = stringField(**value, "id");
  return name && *name == "self";
}

bool isSelfFieldMutationCall(const parser::Node &node) {
  if (node.kind != "Call")
    return false;
  const parser::NodePtr *func = nodeField(node, "func");
  if (!func || !*func || (*func)->kind != "Attribute")
    return false;
  const std::string *methodName = stringField(**func, "attr");
  if (!methodName || (*methodName != "append" && *methodName != "remove"))
    return false;
  const parser::NodePtr *receiver = nodeField(**func, "value");
  return receiver && *receiver && isSelfAttribute(**receiver);
}

bool statementMutatesSelf(const parser::Node &stmt) {
  if (stmt.kind == "Assign") {
    const std::vector<parser::NodePtr> *targets =
        nodeListField(stmt, "targets");
    if (!targets)
      return false;
    return llvm::any_of(*targets, [](const parser::NodePtr &target) {
      return target && isSelfAttribute(*target);
    });
  }
  if (stmt.kind == "AnnAssign") {
    const parser::NodePtr *target = nodeField(stmt, "target");
    return target && *target && isSelfAttribute(**target);
  }
  if (stmt.kind == "AugAssign") {
    const parser::NodePtr *target = nodeField(stmt, "target");
    return target && *target && isSelfAttribute(**target);
  }
  if (stmt.kind == "Expr") {
    const parser::NodePtr *value = nodeField(stmt, "value");
    return value && *value && isSelfFieldMutationCall(**value);
  }
  if (stmt.kind == "If") {
    const std::vector<parser::NodePtr> *body = nodeListField(stmt, "body");
    const std::vector<parser::NodePtr> *orelse = nodeListField(stmt, "orelse");
    auto anyMutates = [](const std::vector<parser::NodePtr> *statements) {
      return statements &&
             llvm::any_of(*statements, [](const parser::NodePtr &child) {
               return child && statementMutatesSelf(*child);
             });
    };
    return anyMutates(body) || anyMutates(orelse);
  }
  return false;
}

std::vector<std::vector<std::string>>
nonEmptySequences(std::vector<std::vector<std::string>> sequences) {
  std::vector<std::vector<std::string>> result;
  for (std::vector<std::string> &sequence : sequences)
    if (!sequence.empty())
      result.push_back(std::move(sequence));
  return result;
}

} // namespace

std::optional<FunctionInfo>
Builder::Impl::parseMethodInfo(const parser::Node &function,
                               llvm::StringRef className) {
  const std::string *methodName = stringField(function, "name");
  const parser::NodePtr *argsNode = nodeField(function, "args");
  if (!methodName || !argsNode || !*argsNode) {
    error(function, "method name or args is missing");
    return std::nullopt;
  }
  if (hasTypeParams(function)) {
    error(function, "generic method type parameters are parsed from CPython "
                    "3.14 syntax but static specialization is not implemented "
                    "in the C++ emitter yet");
    return std::nullopt;
  }
  if (hasNodeListEntries(function, "decorator_list")) {
    error(function, "method decorators are parsed but not implemented in the "
                    "C++ emitter yet");
    return std::nullopt;
  }

  const std::vector<parser::NodePtr> *posonlyargs =
      nodeListField(**argsNode, "posonlyargs");
  const std::vector<parser::NodePtr> *args = nodeListField(**argsNode, "args");
  std::vector<const parser::Node *> positionalParams;
  if (posonlyargs)
    for (const parser::NodePtr &arg : *posonlyargs)
      if (arg)
        positionalParams.push_back(arg.get());
  if (args)
    for (const parser::NodePtr &arg : *args)
      if (arg)
        positionalParams.push_back(arg.get());
  if (positionalParams.empty()) {
    error(function, "method must have self parameter");
    return std::nullopt;
  }
  const parser::Node &selfArg = *positionalParams.front();
  const std::string *selfName = stringField(selfArg, "arg");
  if (!selfName || *selfName != "self") {
    error(selfArg, "first method parameter must be 'self'");
    return std::nullopt;
  }

  FunctionInfo info;
  info.name = (className + "." + *methodName).str();
  info.symbolName = info.name;
  info.isInitMethod = *methodName == "__init__";
  info.argNames.push_back("self");
  info.argTypes.push_back(classType(className));

  std::optional<FunctionTypeComment> typeComment =
      parseFunctionTypeComment(function);
  if (stringField(function, "type_comment") && !typeComment)
    return std::nullopt;
  std::size_t typeCommentArgIndex = 0;
  auto nextCommentArgType = [&]() -> mlir::Type {
    if (!typeComment || typeCommentArgIndex >= typeComment->argTypes.size())
      return {};
    return typeComment->argTypes[typeCommentArgIndex++];
  };

  std::size_t posonlyCount = posonlyargs ? posonlyargs->size() : 0;
  std::size_t firstRegularArg = posonlyCount == 0 ? 1 : 0;
  std::size_t methodParameterCount =
      (posonlyCount > 0 ? posonlyCount - 1 : 0) +
      (args && args->size() > firstRegularArg ? args->size() - firstRegularArg
                                              : 0);
  if (typeComment && typeComment->argTypes.size() != methodParameterCount) {
    error(function, "method type_comment argument count does not match method "
                    "parameters excluding self");
    return std::nullopt;
  }

  for (std::size_t index = 1; index < posonlyCount; ++index) {
    const parser::NodePtr &arg = (*posonlyargs)[index];
    if (arg &&
        !appendAnnotatedParameter(*arg, info, "positional-only method argument",
                                  nextCommentArgType()))
      return std::nullopt;
  }
  info.positionalOnlyCount = info.argNames.size();

  if (args) {
    for (std::size_t index = firstRegularArg; index < args->size(); ++index) {
      const parser::NodePtr &arg = (*args)[index];
      if (arg && !appendAnnotatedParameter(*arg, info, "method argument",
                                           nextCommentArgType()))
        return std::nullopt;
    }
  }
  info.positionalCount = info.argNames.size();
  if (!collectCallableDefaults(function, **argsNode, info))
    return std::nullopt;

  const parser::NodePtr *returns = nodeField(function, "returns");
  std::optional<mlir::Type> resultType;
  if (returns && *returns)
    resultType = typeFromAnnotation(*returns);
  else if (typeComment)
    resultType = typeComment->resultType;
  else
    resultType = noneType();
  if (!resultType) {
    error(function, "method return annotation is unsupported");
    return std::nullopt;
  }
  info.resultType = *resultType;
  llvm::ArrayRef<mlir::Type> allArgTypes(info.argTypes);
  info.signatureType = functionSignatureType(
      allArgTypes.take_front(info.positionalCount), info.resultType,
      info.varargType, allArgTypes.drop_front(info.positionalCount),
      info.kwargType);
  info.functionType = py::FuncType::get(&context, info.signatureType);
  const std::vector<parser::NodePtr> *body = nodeListField(function, "body");
  info.mutatesSelf =
      info.isInitMethod ||
      (body && llvm::any_of(*body, [](const parser::NodePtr &stmt) {
         return stmt && statementMutatesSelf(*stmt);
       }));

  if (mlir::isa<py::FuncType>(info.resultType)) {
    info.returnedCallable = directReturnedNestedCallable(info, function);
    if (info.returnedCallable) {
      info.returnedCallableSymbolName = info.returnedCallable->info.symbolName;
      callableFunctionsBySymbol[info.returnedCallable->info.symbolName] =
          info.returnedCallable->info;
    } else {
      info.returnedCallableArgIndex =
          directReturnedCallableArgIndex(info, function);
      if (!info.returnedCallableArgIndex)
        info.returnedCallableSymbolName =
            directReturnedCallableSymbol(function);
    }
  }
  callableFunctionsBySymbol[info.symbolName] = info;
  return info;
}

void Builder::Impl::scanTypeAliases(const parser::Node &moduleNode) {
  const std::vector<parser::NodePtr> *body = nodeListField(moduleNode, "body");
  if (!body)
    return;

  for (const parser::NodePtr &stmt : *body) {
    if (!stmt)
      continue;
    const parser::NodePtr *nameNode = nullptr;
    const parser::NodePtr *valueNode = nullptr;
    if (stmt->kind == "TypeAlias") {
      nameNode = nodeField(*stmt, "name");
      valueNode = nodeField(*stmt, "value");
      const std::vector<parser::NodePtr> *typeParams =
          nodeListField(*stmt, "type_params");
      if (typeParams && !typeParams->empty()) {
        if (!nameNode || !*nameNode || !valueNode || !*valueNode ||
            (*nameNode)->kind != "Name") {
          error(*stmt, "type alias name or value is missing");
          continue;
        }
        const std::string *name = stringField(**nameNode, "id");
        if (!name) {
          error(*stmt, "type alias name is missing");
          continue;
        }
        TypeAliasInfo info;
        bool supported = true;
        bool seenPackParameter = false;
        for (std::size_t paramIndex = 0; paramIndex < typeParams->size();
             ++paramIndex) {
          const parser::NodePtr &param = (*typeParams)[paramIndex];
          if (!param) {
            supported = false;
            continue;
          }
          if (param->kind != "TypeVar" && param->kind != "TypeVarTuple" &&
              param->kind != "ParamSpec") {
            error(*param, "generic type alias parameter kind '" + param->kind +
                              "' is not supported");
            supported = false;
            continue;
          }
          if (param->kind == "TypeVarTuple" || param->kind == "ParamSpec") {
            if (seenPackParameter) {
              error(*param, "generic type aliases currently support only one "
                            "pack parameter");
              supported = false;
              continue;
            }
            if (paramIndex + 1 != typeParams->size()) {
              error(*param,
                    "generic type aliases currently require pack parameters "
                    "to be final");
              supported = false;
              continue;
            }
            seenPackParameter = true;
          }
          const std::string *paramName = stringField(*param, "name");
          if (!paramName) {
            error(*param, "type parameter name is missing");
            supported = false;
            continue;
          }
          TypeAliasParameter aliasParam;
          if (param->kind == "TypeVarTuple")
            aliasParam.kind = TypeAliasParameterKind::TypeVarTuple;
          else if (param->kind == "ParamSpec")
            aliasParam.kind = TypeAliasParameterKind::ParamSpec;
          else
            aliasParam.kind = TypeAliasParameterKind::TypeVar;
          aliasParam.name = *paramName;
          const parser::NodePtr *defaultValue =
              nodeField(*param, "default_value");
          if (defaultValue && *defaultValue)
            aliasParam.defaultValue = *defaultValue;
          const parser::NodePtr *bound = nodeField(*param, "bound");
          if (bound && *bound) {
            std::optional<mlir::Type> boundType = typeFromAnnotation(*bound);
            if (!boundType) {
              error(**bound, "type parameter bound must resolve to a static "
                             "type");
              supported = false;
              continue;
            }
            aliasParam.bound = *boundType;
          }
          info.parameters.push_back(std::move(aliasParam));
        }
        if (!supported)
          continue;
        if (typeAliases.count(*name) || genericTypeAliases.count(*name)) {
          error(*stmt, "duplicate type alias '" + *name + "'");
          continue;
        }
        info.value = *valueNode;
        genericTypeAliases.emplace(*name, std::move(info));
        continue;
      }
    } else if (stmt->kind == "AnnAssign") {
      const parser::NodePtr *annotation = nodeField(*stmt, "annotation");
      if (!annotation || !isTypeAliasMarker(*annotation))
        continue;
      nameNode = nodeField(*stmt, "target");
      valueNode = nodeField(*stmt, "value");
    } else {
      continue;
    }
    if (!nameNode || !*nameNode || !valueNode || !*valueNode ||
        (*nameNode)->kind != "Name") {
      error(*stmt, "type alias name or value is missing");
      continue;
    }
    const std::string *name = stringField(**nameNode, "id");
    std::optional<mlir::Type> value = typeFromAnnotation(*valueNode);
    if (!name || !value) {
      error(*stmt, "type alias value must resolve to a static type");
      continue;
    }
    if (typeAliases.count(*name) || genericTypeAliases.count(*name)) {
      error(*stmt, "duplicate type alias '" + *name + "'");
      continue;
    }
    typeAliases.emplace(*name, *value);
  }
}

std::optional<std::vector<std::string>>
Builder::Impl::computeClassMro(llvm::StringRef className,
                               const parser::Node &anchor,
                               std::set<std::string> &visiting) {
  auto found = classes.find(className.str());
  if (found == classes.end()) {
    error(anchor, "unknown base class '" + className.str() + "'");
    return std::nullopt;
  }
  ClassInfo &info = found->second;
  if (!info.mro.empty())
    return info.mro;
  if (visiting.count(info.name)) {
    error(anchor, "class inheritance cycle through '" + info.name + "'");
    return std::nullopt;
  }

  std::set<std::string> directBases;
  for (const std::string &baseName : info.baseNames) {
    if (!directBases.insert(baseName).second) {
      error(anchor, "duplicate base class '" + baseName + "'");
      return std::nullopt;
    }
  }

  visiting.insert(info.name);
  std::vector<std::vector<std::string>> sequences;
  for (const std::string &baseName : info.baseNames) {
    std::optional<std::vector<std::string>> baseMro =
        computeClassMro(baseName, anchor, visiting);
    if (!baseMro) {
      visiting.erase(info.name);
      return std::nullopt;
    }
    sequences.push_back(*baseMro);
  }
  sequences.push_back(info.baseNames);
  visiting.erase(info.name);

  std::vector<std::string> result{info.name};
  sequences = nonEmptySequences(std::move(sequences));
  while (!sequences.empty()) {
    std::optional<std::string> candidate;
    for (const std::vector<std::string> &sequence : sequences) {
      const std::string &head = sequence.front();
      bool inTail = llvm::any_of(sequences, [&](const auto &other) {
        return llvm::is_contained(llvm::ArrayRef(other).drop_front(), head);
      });
      if (!inTail) {
        candidate = head;
        break;
      }
    }
    if (!candidate) {
      error(anchor, "inconsistent C3 MRO for class '" + info.name + "'");
      return std::nullopt;
    }
    result.push_back(*candidate);
    for (std::vector<std::string> &sequence : sequences) {
      if (!sequence.empty() && sequence.front() == *candidate)
        sequence.erase(sequence.begin());
    }
    sequences = nonEmptySequences(std::move(sequences));
  }

  info.mro = result;
  return result;
}

bool Builder::Impl::resolveClassInheritance(llvm::StringRef className,
                                            const parser::Node &anchor,
                                            std::set<std::string> &resolving) {
  auto found = classes.find(className.str());
  if (found == classes.end()) {
    error(anchor, "unknown class '" + className.str() + "'");
    return false;
  }
  ClassInfo &info = found->second;
  if (info.inheritanceResolved)
    return true;
  if (resolving.count(info.name)) {
    error(anchor, "class inheritance cycle through '" + info.name + "'");
    return false;
  }

  std::set<std::string> visiting;
  std::optional<std::vector<std::string>> mro =
      computeClassMro(info.name, anchor, visiting);
  if (!mro)
    return false;

  resolving.insert(info.name);
  for (llvm::StringRef ancestorName : llvm::ArrayRef(*mro).drop_front()) {
    if (!resolveClassInheritance(ancestorName, anchor, resolving)) {
      resolving.erase(info.name);
      return false;
    }
  }

  std::map<std::string, mlir::Type> ownFields = info.fields;
  std::map<std::string, std::string> ownMethods = info.methods;
  std::map<std::string, parser::NodePtr> ownMethodNodes = info.methodNodes;

  std::map<std::string, mlir::Type> resolvedFields;
  for (llvm::StringRef ancestorName :
       llvm::reverse(llvm::ArrayRef(*mro).drop_front())) {
    const ClassInfo &ancestor = classes[ancestorName.str()];
    for (const auto &[fieldName, fieldType] : ancestor.fields) {
      auto inserted = resolvedFields.emplace(fieldName, fieldType);
      if (!inserted.second && inserted.first->second != fieldType) {
        error(anchor, "inherited field '" + fieldName + "' in class '" +
                          info.name + "' has incompatible static types: " +
                          typeString(inserted.first->second) + " vs " +
                          typeString(fieldType));
        resolving.erase(info.name);
        return false;
      }
    }
  }
  for (const auto &[fieldName, fieldType] : ownFields) {
    auto existing = resolvedFields.find(fieldName);
    if (existing != resolvedFields.end() && existing->second != fieldType) {
      error(anchor, "field '" + fieldName + "' in class '" + info.name +
                        "' overrides an inherited field with incompatible "
                        "static type: " +
                        typeString(existing->second) + " vs " +
                        typeString(fieldType));
      resolving.erase(info.name);
      return false;
    }
    resolvedFields[fieldName] = fieldType;
  }
  info.fields = std::move(resolvedFields);

  info.methods = std::move(ownMethods);
  info.methodNodes = std::move(ownMethodNodes);
  for (llvm::StringRef ancestorName : llvm::ArrayRef(*mro).drop_front()) {
    const ClassInfo &ancestor = classes[ancestorName.str()];
    for (const auto &[methodName, methodNode] : ancestor.ownMethodNodes) {
      if (!methodNode || info.methods.count(methodName))
        continue;
      std::optional<FunctionInfo> method =
          parseMethodInfo(*methodNode, info.name);
      if (!method)
        continue;
      info.methods[methodName] = method->name;
      info.methodNodes[methodName] = methodNode;
      functions[method->name] = std::move(*method);
    }
  }

  info.inheritanceResolved = true;
  resolving.erase(info.name);
  return true;
}

void Builder::Impl::scanClasses(const parser::Node &moduleNode) {
  const std::vector<parser::NodePtr> *body = nodeListField(moduleNode, "body");
  if (!body)
    return;

  for (const parser::NodePtr &stmt : *body) {
    if (!stmt || stmt->kind != "ClassDef")
      continue;
    const std::string *name = stringField(*stmt, "name");
    if (!name) {
      error(*stmt, "ClassDef.name is missing");
      continue;
    }
    if (hasTypeParams(*stmt)) {
      error(*stmt, "generic class type parameters are parsed from CPython 3.14 "
                   "syntax but static specialization is not implemented in "
                   "the C++ emitter yet");
      continue;
    }
    if (hasNodeListEntries(*stmt, "decorator_list")) {
      error(*stmt, "class decorators are parsed but not implemented in the "
                   "C++ emitter yet");
      continue;
    }
    if (hasNodeListEntries(*stmt, "keywords")) {
      error(*stmt, "class keyword arguments are parsed but not implemented in "
                   "the C++ emitter yet");
      continue;
    }
    ClassInfo info;
    info.name = *name;
    const std::vector<parser::NodePtr> *bases = nodeListField(*stmt, "bases");
    if (bases) {
      for (const parser::NodePtr &base : *bases) {
        if (!base || base->kind != "Name") {
          error(*stmt, "class bases must be statically named");
          continue;
        }
        const std::string *baseName = stringField(*base, "id");
        if (baseName)
          info.baseNames.push_back(*baseName);
      }
    }
    classes.emplace(*name, std::move(info));
  }

  scanTypeAliases(moduleNode);

  for (const parser::NodePtr &stmt : *body) {
    if (!stmt || stmt->kind != "ClassDef")
      continue;
    if (hasTypeParams(*stmt))
      continue;
    if (hasNodeListEntries(*stmt, "decorator_list") ||
        hasNodeListEntries(*stmt, "keywords"))
      continue;
    const std::string *className = stringField(*stmt, "name");
    const std::vector<parser::NodePtr> *classBody =
        nodeListField(*stmt, "body");
    if (!className || !classBody)
      continue;
    ClassInfo &classInfo = classes[*className];

    for (const parser::NodePtr &member : *classBody) {
      if (member && member->kind == "AnnAssign") {
        const parser::NodePtr *target = nodeField(*member, "target");
        const parser::NodePtr *annotation = nodeField(*member, "annotation");
        if (!target || !*target || (*target)->kind != "Name" || !annotation ||
            !*annotation)
          continue;
        const std::string *fieldName = stringField(**target, "id");
        std::optional<mlir::Type> fieldType = typeFromAnnotation(*annotation);
        if (fieldName && fieldType)
          classInfo.fields[*fieldName] = *fieldType;
        continue;
      }
      if (!member || member->kind != "FunctionDef")
        continue;
      std::optional<FunctionInfo> method = parseMethodInfo(*member, *className);
      if (!method)
        continue;
      const std::string *methodName = stringField(*member, "name");
      if (methodName) {
        classInfo.methods[*methodName] = method->name;
        classInfo.methodNodes[*methodName] = member;
        classInfo.ownMethodNodes[*methodName] = member;
      }
      functions.emplace(method->name, std::move(*method));
    }

    auto init = llvm::find_if(*classBody, [](const parser::NodePtr &member) {
      if (!member || member->kind != "FunctionDef")
        return false;
      const std::string *methodName = stringField(*member, "name");
      return methodName && *methodName == "__init__";
    });
    if (init == classBody->end() || !*init)
      continue;

    const parser::NodePtr *argsNode = nodeField(**init, "args");
    const std::vector<parser::NodePtr> *posonlyargs =
        argsNode && *argsNode ? nodeListField(**argsNode, "posonlyargs")
                              : nullptr;
    const std::vector<parser::NodePtr> *args =
        argsNode && *argsNode ? nodeListField(**argsNode, "args") : nullptr;
    const std::vector<parser::NodePtr> *initBody =
        nodeListField(**init, "body");
    if (!args || !initBody)
      continue;

    std::map<std::string, mlir::Type> argTypes;
    std::vector<const parser::Node *> initParams;
    if (posonlyargs)
      for (const parser::NodePtr &arg : *posonlyargs)
        if (arg)
          initParams.push_back(arg.get());
    for (const parser::NodePtr &arg : *args)
      if (arg)
        initParams.push_back(arg.get());
    for (std::size_t i = 1; i < initParams.size(); ++i) {
      const parser::Node &arg = *initParams[i];
      const std::string *argName = stringField(arg, "arg");
      const parser::NodePtr *annotation = nodeField(arg, "annotation");
      std::optional<mlir::Type> argType =
          annotation ? typeFromAnnotation(*annotation) : std::nullopt;
      if (argName && argType)
        argTypes[*argName] = *argType;
    }

    for (const parser::NodePtr &statement : *initBody) {
      if (!statement)
        continue;
      if (statement->kind == "AnnAssign") {
        const parser::NodePtr *target = nodeField(*statement, "target");
        const parser::NodePtr *annotation = nodeField(*statement, "annotation");
        if (!target || !*target || !isSelfAttribute(**target) || !annotation ||
            !*annotation)
          continue;
        const std::string *fieldName = stringField(**target, "attr");
        std::optional<mlir::Type> fieldType = typeFromAnnotation(*annotation);
        if (fieldName && fieldType)
          classInfo.fields[*fieldName] = *fieldType;
        continue;
      }
      if (statement->kind == "Assign") {
        const std::vector<parser::NodePtr> *targets =
            nodeListField(*statement, "targets");
        const parser::NodePtr *value = nodeField(*statement, "value");
        if (!targets || targets->size() != 1 || !targets->front() || !value ||
            !*value || !isSelfAttribute(*targets->front()) ||
            (*value)->kind != "Name")
          continue;
        const std::string *fieldName = stringField(*targets->front(), "attr");
        const std::string *valueName = stringField(**value, "id");
        if (!fieldName || !valueName)
          continue;
        auto found = argTypes.find(*valueName);
        if (found != argTypes.end())
          classInfo.fields[*fieldName] = found->second;
      }
    }
  }

  for (const parser::NodePtr &stmt : *body) {
    if (!stmt || stmt->kind != "ClassDef")
      continue;
    if (hasTypeParams(*stmt))
      continue;
    if (hasNodeListEntries(*stmt, "decorator_list") ||
        hasNodeListEntries(*stmt, "keywords"))
      continue;
    const std::string *className = stringField(*stmt, "name");
    if (!className)
      continue;
    std::set<std::string> resolving;
    resolveClassInheritance(*className, *stmt, resolving);
  }
}

void Builder::Impl::emitClassDefs(const parser::Node &moduleNode) {
  const std::vector<parser::NodePtr> *body = nodeListField(moduleNode, "body");
  if (!body)
    return;
  for (const parser::NodePtr &stmt : *body) {
    if (stmt && stmt->kind == "ClassDef" && !hasTypeParams(*stmt) &&
        !hasNodeListEntries(*stmt, "decorator_list") &&
        !hasNodeListEntries(*stmt, "keywords"))
      emitClassDef(*stmt);
  }
}

void Builder::Impl::emitClassDef(const parser::Node &classNode) {
  const std::string *name = stringField(classNode, "name");
  const std::vector<parser::NodePtr> *body = nodeListField(classNode, "body");
  if (!name || !body)
    return;
  auto found = classes.find(*name);
  if (found == classes.end())
    return;
  const ClassInfo &info = found->second;

  llvm::SmallVector<mlir::Attribute> fieldNames;
  llvm::SmallVector<mlir::Attribute> fieldTypes;
  for (const auto &[fieldName, fieldType] : info.fields) {
    fieldNames.push_back(builder.getStringAttr(fieldName));
    fieldTypes.push_back(mlir::TypeAttr::get(fieldType));
  }

  py::ClassOp classOp = builder.create<py::ClassOp>(
      loc(classNode), *name, stringArrayAttr(info.baseNames),
      fieldNames.empty() ? mlir::ArrayAttr{} : builder.getArrayAttr(fieldNames),
      fieldTypes.empty() ? mlir::ArrayAttr{}
                         : builder.getArrayAttr(fieldTypes));
  classOp.getBody().emplaceBlock();

  llvm::SmallVector<std::string> methodNames;
  if (info.methodNodes.count("__init__"))
    methodNames.push_back("__init__");
  for (const auto &[methodName, ignored] : info.methodNodes) {
    (void)ignored;
    if (methodName != "__init__")
      methodNames.push_back(methodName);
  }

  for (const std::string &methodName : methodNames) {
    auto nodeFound = info.methodNodes.find(methodName);
    if (nodeFound == info.methodNodes.end() || !nodeFound->second)
      continue;
    auto methodFound = info.methods.find(methodName);
    if (methodFound == info.methods.end())
      continue;
    auto functionFound = functions.find(methodFound->second);
    if (functionFound != functions.end())
      emitMethodDef(*nodeFound->second, functionFound->second);
  }
  builder.setInsertionPointToEnd(module->getBody());
}

void Builder::Impl::emitMethodDef(const parser::Node &function,
                                  const FunctionInfo &info) {
  const std::vector<parser::NodePtr> *body = nodeListField(function, "body");
  if (!body) {
    error(function, "method body is missing");
    return;
  }

  llvm::ArrayRef<std::string> allArgNames(info.argNames);
  py::FuncOp func =
      createFunc(info.symbolName, info.signatureType,
                 stringArrayAttr(allArgNames.take_front(info.positionalCount)),
                 /*hasVararg=*/static_cast<bool>(info.varargType),
                 /*hasKwarg=*/static_cast<bool>(info.kwargType), info.mayThrow,
                 stringArrayAttr(info.kwonlyNames));
  if (info.isInitMethod)
    func->setAttr("init_method", builder.getUnitAttr());
  if (info.mutatesSelf)
    func->setAttr("mutates_self", builder.getUnitAttr());
  llvm::SmallVector<mlir::Type, 8> entryTypes(info.argTypes.begin(),
                                              info.argTypes.end());
  if (info.varargType)
    entryTypes.push_back(info.varargType);
  if (info.kwargType)
    entryTypes.push_back(info.kwargType);
  addEntryBlock(func, entryTypes);

  std::map<std::string, Value> savedSymbols = std::move(symbols);
  std::map<std::string, FunctionInfo> savedCallableAliases =
      std::move(callableAliases);
  mlir::Type savedReturnType = currentReturnType;
  bool savedTerminated = blockTerminated;
  bool savedInNativeFunction = inNativeFunction;
  unsigned savedExceptionContextDepth = exceptionContextDepth;
  functionSymbolStack.push_back(info.symbolName);
  symbols.clear();
  callableAliases.clear();
  currentReturnType = info.resultType;
  blockTerminated = false;
  inNativeFunction = false;
  exceptionContextDepth = 0;
  for (auto indexed : llvm::enumerate(info.argNames)) {
    mlir::Value arg = func.getBody().front().getArgument(indexed.index());
    symbols.emplace(indexed.value(),
                    Value{arg, info.argTypes[indexed.index()]});
  }
  if (info.varargName && info.varargType) {
    mlir::Value arg = func.getBody().front().getArgument(info.argTypes.size());
    symbols.emplace(*info.varargName, Value{arg, info.varargType});
  }
  if (info.kwargName && info.kwargType) {
    mlir::Value arg = func.getBody().front().getArgument(
        info.argTypes.size() + (info.varargType ? 1 : 0));
    symbols.emplace(*info.kwargName, Value{arg, info.kwargType});
  }
  for (const parser::NodePtr &stmt : *body) {
    if (stmt && !blockTerminated)
      emitStatement(*stmt);
  }
  if (!blockTerminated) {
    if (info.resultType == noneType()) {
      mlir::Value none = builder.create<py::NoneOp>(loc(function), noneType());
      builder.create<py::ReturnOp>(loc(function), mlir::ValueRange{none});
    } else {
      error(function,
            "method may exit without returning " + typeString(info.resultType));
    }
  }

  symbols = std::move(savedSymbols);
  callableAliases = std::move(savedCallableAliases);
  currentReturnType = savedReturnType;
  blockTerminated = savedTerminated;
  inNativeFunction = savedInNativeFunction;
  exceptionContextDepth = savedExceptionContextDepth;
  functionSymbolStack.pop_back();
  builder.setInsertionPointToEnd(module->getBody());
}

Value Builder::Impl::emitClassConstructorCall(
    const parser::Node &expr, llvm::StringRef name,
    const std::vector<parser::NodePtr> &args) {
  mlir::Type resultType = classType(name);
  mlir::Value instance =
      builder.create<py::ClassNewOp>(loc(expr), resultType, name);
  auto classFound = classes.find(name.str());
  if (classFound == classes.end())
    return Value{instance, resultType};
  auto initFound = classFound->second.methods.find("__init__");
  if (initFound == classFound->second.methods.end())
    return Value{instance, resultType};
  auto functionFound = functions.find(initFound->second);
  if (functionFound == functions.end())
    return Value{instance, resultType};

  const FunctionInfo &init = functionFound->second;
  Value receiver{instance, resultType};
  if (hasCallableMetadata(init)) {
    std::optional<CallArgumentTuples> tuples = emitExplicitCallArgumentTuples(
        expr, init, args, /*firstFormal=*/1, llvm::ArrayRef<Value>{receiver});
    if (!tuples)
      return Value{{}, resultType};
    Value callee = emitFunctionObject(expr, init);
    if (!callee.value)
      return Value{{}, resultType};
    builder.create<py::CallVectorOp>(loc(expr), mlir::TypeRange{noneType()},
                                     callee.value, tuples->posargs.value,
                                     tuples->kwnames.value,
                                     tuples->kwvalues.value, mlir::UnitAttr{});
    return Value{instance, resultType};
  }

  std::vector<Value> initArgs;
  initArgs.push_back(receiver);
  std::optional<std::vector<Value>> userArgs =
      emitStaticArguments(expr, init, args, /*firstFormal=*/1);
  if (!userArgs)
    return Value{{}, resultType};
  initArgs.insert(initArgs.end(), userArgs->begin(), userArgs->end());

  mlir::Value callee =
      builder.create<py::FuncObjectOp>(loc(expr), init.functionType, init.name);
  CallArgumentTuples tuples = emitCallArgumentTuples(init, initArgs);
  builder.create<py::CallVectorOp>(
      loc(expr), mlir::TypeRange{noneType()}, callee, tuples.posargs.value,
      tuples.kwnames.value, tuples.kwvalues.value, mlir::UnitAttr{});
  return Value{instance, resultType};
}

} // namespace lython::emitter
