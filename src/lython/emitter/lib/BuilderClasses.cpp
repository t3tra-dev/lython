#include "BuilderImpl.h"

#include "mlir/Dialect/Async/IR/Async.h"
#include "llvm/ADT/STLExtras.h"

#include <cctype>
#include <functional>
#include <set>
#include <utility>

namespace lython::emitter {

namespace {

static constexpr llvm::StringLiteral kInstanceMethodKind("instance");
static constexpr llvm::StringLiteral kStaticMethodKind("static");
static constexpr llvm::StringLiteral kClassMethodKind("class");

bool isSelfAttribute(const parser::Node &node) {
  if (node.kind != "Attribute")
    return false;
  const parser::NodePtr *value = nodeField(node, "value");
  if (!value || !*value || (*value)->kind != "Name")
    return false;
  const std::string *name = stringField(**value, "id");
  return name && *name == "self";
}

std::optional<std::string> selfAttributeName(const parser::Node &node) {
  if (!isSelfAttribute(node))
    return std::nullopt;
  const std::string *name = stringField(node, "attr");
  if (!name)
    return std::nullopt;
  return *name;
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

bool isNoArgCallNamed(const parser::Node &node, llvm::StringRef name) {
  if (node.kind != "Call")
    return false;
  const parser::NodePtr *func = nodeField(node, "func");
  const std::vector<parser::NodePtr> *args = nodeListField(node, "args");
  const std::vector<parser::NodePtr> *keywords =
      nodeListField(node, "keywords");
  if (!func || !*func || (*func)->kind != "Name" || !args || !args->empty() ||
      (keywords && !keywords->empty()))
    return false;
  const std::string *callee = stringField(**func, "id");
  return callee && *callee == name;
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

std::optional<std::pair<std::string, std::string>>
selfFieldAssignedFromName(const parser::Node &stmt) {
  const parser::Node *target = nullptr;
  const parser::NodePtr *value = nullptr;

  if (stmt.kind == "Assign") {
    const std::vector<parser::NodePtr> *targets =
        nodeListField(stmt, "targets");
    if (!targets || targets->size() != 1 || !targets->front())
      return std::nullopt;
    target = targets->front().get();
    value = nodeField(stmt, "value");
  } else if (stmt.kind == "AnnAssign") {
    const parser::NodePtr *targetNode = nodeField(stmt, "target");
    if (!targetNode || !*targetNode)
      return std::nullopt;
    target = targetNode->get();
    value = nodeField(stmt, "value");
  } else {
    return std::nullopt;
  }

  if (!target || !isSelfAttribute(*target) || !value || !*value ||
      (*value)->kind != "Name")
    return std::nullopt;

  std::optional<std::string> fieldName = selfAttributeName(*target);
  const std::string *valueName = stringField(**value, "id");
  if (!fieldName || !valueName)
    return std::nullopt;
  return std::make_pair(*fieldName, *valueName);
}

std::vector<std::vector<std::string>>
nonEmptySequences(std::vector<std::vector<std::string>> sequences) {
  std::vector<std::vector<std::string>> result;
  for (std::vector<std::string> &sequence : sequences)
    if (!sequence.empty())
      result.push_back(std::move(sequence));
  return result;
}

std::string joinFieldNames(const std::set<std::string> &fields) {
  std::string result;
  for (const std::string &field : fields) {
    if (!result.empty())
      result += ", ";
    result += field;
  }
  return result;
}

bool sameMethodOverrideSignature(const FunctionInfo &derived,
                                 const FunctionInfo &base) {
  if (derived.methodKind != base.methodKind)
    return false;
  const bool skipReceiver = derived.methodKind == kInstanceMethodKind ||
                            derived.methodKind == kClassMethodKind;
  const std::size_t skip = skipReceiver ? 1 : 0;
  if (derived.argTypes.size() < skip || base.argTypes.size() < skip)
    return false;
  if (derived.argTypes.size() - skip != base.argTypes.size() - skip)
    return false;
  for (std::size_t index = skip; index < derived.argTypes.size(); ++index)
    if (derived.argTypes[index] != base.argTypes[index])
      return false;
  auto adjustedCount = [&](std::size_t count) {
    return count >= skip ? count - skip : count;
  };
  return adjustedCount(derived.positionalOnlyCount) ==
             adjustedCount(base.positionalOnlyCount) &&
         adjustedCount(derived.positionalCount) ==
             adjustedCount(base.positionalCount) &&
         derived.kwonlyNames == base.kwonlyNames &&
         derived.varargType == base.varargType &&
         derived.kwargType == base.kwargType &&
         derived.resultType == base.resultType;
}

} // namespace

std::optional<FunctionInfo> Builder::Impl::parseMethodInfo(
    const parser::Node &function, llvm::StringRef className,
    const std::map<std::string, mlir::Type> *typeVariables) {
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
  std::string methodKind = kInstanceMethodKind.str();
  const std::vector<parser::NodePtr> *decorators =
      nodeListField(function, "decorator_list");
  if (decorators) {
    for (const parser::NodePtr &decorator : *decorators) {
      if (!decorator || decorator->kind != "Name") {
        error(function,
              "method decorators must be statically named @staticmethod or "
              "@classmethod");
        return std::nullopt;
      }
      const std::string *decoratorName = stringField(*decorator, "id");
      if (!decoratorName) {
        error(function, "method decorator name is missing");
        return std::nullopt;
      }
      if (*decoratorName == "staticmethod") {
        if (methodKind != kInstanceMethodKind) {
          error(function, "method has multiple incompatible decorators");
          return std::nullopt;
        }
        methodKind = kStaticMethodKind.str();
      } else if (*decoratorName == "classmethod") {
        if (methodKind != kInstanceMethodKind) {
          error(function, "method has multiple incompatible decorators");
          return std::nullopt;
        }
        methodKind = kClassMethodKind.str();
      } else {
        error(function,
              "unsupported method decorator '@" + *decoratorName + "'");
        return std::nullopt;
      }
    }
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
  if (methodKind != kStaticMethodKind && positionalParams.empty()) {
    error(function, "method must have self parameter");
    return std::nullopt;
  }
  if (methodKind == kInstanceMethodKind) {
    const parser::Node &selfArg = *positionalParams.front();
    const std::string *selfName = stringField(selfArg, "arg");
    if (!selfName || *selfName != "self") {
      error(selfArg, "first method parameter must be 'self'");
      return std::nullopt;
    }
  } else if (methodKind == kClassMethodKind) {
    const parser::Node &clsArg = *positionalParams.front();
    const std::string *clsName = stringField(clsArg, "arg");
    if (!clsName || *clsName != "cls") {
      error(clsArg, "first classmethod parameter must be 'cls'");
      return std::nullopt;
    }
  }

  FunctionInfo info;
  info.name = (className + "." + *methodName).str();
  info.definition = &function;
  info.isAsync = function.kind == "AsyncFunctionDef";
  info.symbolName = info.name;
  if (typeVariables)
    info.typeSubstitutions = *typeVariables;
  info.isInitMethod = *methodName == "__init__";
  info.methodKind = methodKind;
  if (methodKind == kInstanceMethodKind) {
    info.argNames.push_back("self");
    info.argTypes.push_back(classType(className));
  } else if (methodKind == kClassMethodKind) {
    info.argNames.push_back("cls");
    info.argTypes.push_back(classType(className));
  }

  std::optional<FunctionTypeComment> typeComment =
      parseFunctionTypeComment(function, typeVariables);
  if (stringField(function, "type_comment") && !typeComment)
    return std::nullopt;
  std::size_t typeCommentArgIndex = 0;
  auto nextCommentArgType = [&]() -> mlir::Type {
    if (!typeComment || typeCommentArgIndex >= typeComment->argTypes.size())
      return {};
    return typeComment->argTypes[typeCommentArgIndex++];
  };

  std::size_t posonlyCount = posonlyargs ? posonlyargs->size() : 0;
  std::size_t firstRegularArg =
      posonlyCount == 0 && methodKind != kStaticMethodKind ? 1 : 0;
  std::size_t implicitReceiverCount = methodKind == kStaticMethodKind ? 0 : 1;
  std::size_t methodParameterCount =
      (posonlyCount > implicitReceiverCount
           ? posonlyCount - implicitReceiverCount
           : 0) +
      (args && args->size() > firstRegularArg ? args->size() - firstRegularArg
                                              : 0);
  if (typeComment && typeComment->argTypes.size() != methodParameterCount) {
    error(function, "method type_comment argument count does not match method "
                    "parameters excluding self");
    return std::nullopt;
  }

  for (std::size_t index = implicitReceiverCount; index < posonlyCount;
       ++index) {
    const parser::NodePtr &arg = (*posonlyargs)[index];
    if (arg &&
        !appendAnnotatedParameter(*arg, info, "positional-only method argument",
                                  typeVariables, nextCommentArgType()))
      return std::nullopt;
  }
  info.positionalOnlyCount = info.argNames.size();

  if (args) {
    for (std::size_t index = firstRegularArg; index < args->size(); ++index) {
      const parser::NodePtr &arg = (*args)[index];
      if (arg && !appendAnnotatedParameter(*arg, info, "method argument",
                                           typeVariables, nextCommentArgType()))
        return std::nullopt;
    }
  }
  info.positionalCount = info.argNames.size();
  if (!collectCallableDefaults(function, **argsNode, info, typeVariables))
    return std::nullopt;
  if (info.isAsync && (info.varargType || info.kwargType)) {
    error(function, "*args/**kwargs are not supported on async methods yet");
    return std::nullopt;
  }

  const parser::NodePtr *returns = nodeField(function, "returns");
  std::optional<mlir::Type> resultType;
  if (returns && *returns)
    resultType = typeVariables ? typeFromAnnotation(*returns, *typeVariables)
                               : typeFromAnnotation(*returns);
  else if (typeComment)
    resultType = typeComment->resultType;
  else
    resultType = noneType();
  if (!resultType) {
    error(function, "method return annotation is unsupported");
    return std::nullopt;
  }
  if (!info.isAsync)
    if (std::optional<mlir::Type> concreteResult =
            directReturnedConcreteType(*resultType, function))
      resultType = *concreteResult;
  info.resultType = *resultType;
  refreshFunctionTypes(info);
  const std::vector<parser::NodePtr> *body = nodeListField(function, "body");
  info.mutatesSelf =
      info.isInitMethod ||
      (body && llvm::any_of(*body, [](const parser::NodePtr &stmt) {
         return stmt && statementMutatesSelf(*stmt);
       }));

  if (!info.isAsync && info.resultType != noneType()) {
    info.returnedExactClass = directReturnedExactClass(info, function);
    info.returnedValueArgIndex = directReturnedValueArgIndex(info, function);
    if (!info.returnedExactClass)
      info.returnedClassArgIndex = directReturnedClassArgIndex(info, function);
  }

  if (py::isCallableType(info.resultType)) {
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

bool Builder::Impl::verifyInitFieldInitialization(const ClassInfo &info,
                                                  const parser::Node &anchor) {
  std::set<std::string> requiredFields;
  for (const auto &[fieldName, ignored] : info.fields) {
    (void)ignored;
    requiredFields.insert(fieldName);
  }
  if (requiredFields.empty())
    return true;

  auto initNode = info.methodNodes.find("__init__");
  if (initNode == info.methodNodes.end() || !initNode->second) {
    error(anchor, "class '" + info.name +
                      "' has fields but no __init__ to initialize them");
    return false;
  }
  const std::vector<parser::NodePtr> *body =
      nodeListField(*initNode->second, "body");
  if (!body) {
    error(*initNode->second, "__init__ body is missing");
    return false;
  }

  auto missingFrom = [&](const std::set<std::string> &initialized) {
    std::set<std::string> missing;
    for (const std::string &field : requiredFields)
      if (!initialized.count(field))
        missing.insert(field);
    return missing;
  };

  auto addInitialized = [&](const parser::Node &assignmentTarget,
                            std::set<std::string> &initialized) -> bool {
    std::optional<std::string> fieldName = selfAttributeName(assignmentTarget);
    if (!fieldName)
      return true;
    if (!requiredFields.count(*fieldName)) {
      error(assignmentTarget, "assignment to undeclared field '" + *fieldName +
                                  "' in class '" + info.name + "'");
      return false;
    }
    initialized.insert(*fieldName);
    return true;
  };

  struct Flow {
    std::set<std::string> initialized;
    bool continues = true;
    bool ok = true;
  };

  std::function<Flow(const std::vector<parser::NodePtr> &,
                     std::set<std::string>)>
      visitStatements = [&](const std::vector<parser::NodePtr> &statements,
                            std::set<std::string> initialized) -> Flow {
    for (const parser::NodePtr &stmt : statements) {
      if (!stmt)
        continue;

      if (stmt->kind == "Assign") {
        const std::vector<parser::NodePtr> *targets =
            nodeListField(*stmt, "targets");
        if (targets) {
          for (const parser::NodePtr &target : *targets)
            if (target && !addInitialized(*target, initialized))
              return Flow{initialized, true, false};
        }
        continue;
      }

      if (stmt->kind == "AnnAssign" || stmt->kind == "AugAssign") {
        const parser::NodePtr *target = nodeField(*stmt, "target");
        if (target && *target) {
          if (stmt->kind == "AugAssign") {
            std::optional<std::string> fieldName = selfAttributeName(**target);
            if (fieldName && !initialized.count(*fieldName)) {
              error(**target, "augmented assignment reads field '" +
                                  *fieldName +
                                  "' before it is definitely initialized");
              return Flow{initialized, true, false};
            }
          }
          if (!addInitialized(**target, initialized))
            return Flow{initialized, true, false};
        }
        continue;
      }

      if (stmt->kind == "If") {
        const std::vector<parser::NodePtr> *thenBody =
            nodeListField(*stmt, "body");
        const std::vector<parser::NodePtr> *elseBody =
            nodeListField(*stmt, "orelse");
        Flow thenFlow = thenBody ? visitStatements(*thenBody, initialized)
                                 : Flow{initialized, true, true};
        if (!thenFlow.ok)
          return thenFlow;
        Flow elseFlow = elseBody ? visitStatements(*elseBody, initialized)
                                 : Flow{initialized, true, true};
        if (!elseFlow.ok)
          return elseFlow;
        if (thenFlow.continues && elseFlow.continues) {
          std::set<std::string> both;
          for (const std::string &field : thenFlow.initialized)
            if (elseFlow.initialized.count(field))
              both.insert(field);
          initialized = std::move(both);
          continue;
        }
        if (thenFlow.continues) {
          initialized = std::move(thenFlow.initialized);
          continue;
        }
        if (elseFlow.continues) {
          initialized = std::move(elseFlow.initialized);
          continue;
        }
        return Flow{initialized, false, true};
      }

      if (stmt->kind == "Return") {
        std::set<std::string> missing = missingFrom(initialized);
        if (!missing.empty()) {
          error(*stmt, "__init__ may return before initializing fields: " +
                           joinFieldNames(missing));
          return Flow{initialized, false, false};
        }
        return Flow{initialized, false, true};
      }

      if (stmt->kind == "Raise")
        return Flow{initialized, false, true};
    }
    return Flow{initialized, true, true};
  };

  Flow finalFlow = visitStatements(*body, {});
  if (!finalFlow.ok)
    return false;
  if (!finalFlow.continues)
    return true;
  std::set<std::string> missing = missingFrom(finalFlow.initialized);
  if (!missing.empty()) {
    error(*initNode->second, "__init__ may exit without initializing fields: " +
                                 joinFieldNames(missing));
    return false;
  }
  return true;
}

bool Builder::Impl::isTypingName(const parser::Node &node,
                                 llvm::StringRef name) const {
  if (node.kind == "Name") {
    const std::string *id = stringField(node, "id");
    if (!id)
      return false;
    auto alias = staticAnnotationAliases.find(*id);
    if (alias != staticAnnotationAliases.end())
      return alias->second == name;
    return *id == name;
  }
  if (node.kind != "Attribute")
    return false;
  const parser::NodePtr *value = nodeField(node, "value");
  const std::string *attr = stringField(node, "attr");
  if (!value || !*value || (*value)->kind != "Name" || !attr || *attr != name)
    return false;
  const std::string *module = stringField(**value, "id");
  if (!module)
    return false;
  auto found = staticModules.find(*module);
  return found != staticModules.end() &&
         (found->second == "typing" || found->second == "collections.abc");
}

bool Builder::Impl::isTypeVarDefinition(const parser::Node &stmt) const {
  if (stmt.kind != "Assign")
    return false;
  const std::vector<parser::NodePtr> *targets = nodeListField(stmt, "targets");
  const parser::NodePtr *value = nodeField(stmt, "value");
  if (!targets || targets->size() != 1 || !targets->front() || !value ||
      !*value || targets->front()->kind != "Name" || (*value)->kind != "Call")
    return false;
  const parser::NodePtr *func = nodeField(**value, "func");
  return func && *func &&
         (isTypingName(**func, "TypeVar") ||
          isTypingName(**func, "TypeVarTuple") ||
          isTypingName(**func, "ParamSpec"));
}

void Builder::Impl::scanTypeVariables(const parser::Node &moduleNode) {
  const std::vector<parser::NodePtr> *body = nodeListField(moduleNode, "body");
  if (!body)
    return;
  for (const parser::NodePtr &stmt : *body) {
    if (!stmt || !isTypeVarDefinition(*stmt))
      continue;
    const std::vector<parser::NodePtr> *targets =
        nodeListField(*stmt, "targets");
    const parser::NodePtr *value = nodeField(*stmt, "value");
    const std::string *targetName = stringField(*targets->front(), "id");
    const parser::NodePtr *func = nodeField(**value, "func");
    if (!targetName || !func || !*func)
      continue;

    TypeAliasParameter parameter;
    parameter.name = *targetName;
    if (isTypingName(**func, "TypeVarTuple"))
      parameter.kind = TypeAliasParameterKind::TypeVarTuple;
    else if (isTypingName(**func, "ParamSpec"))
      parameter.kind = TypeAliasParameterKind::ParamSpec;

    const std::vector<parser::NodePtr> *args = nodeListField(**value, "args");
    if (args && !args->empty() && args->front() &&
        args->front()->kind == "Constant") {
      const parser::FieldValue *literal = valueField(*args->front(), "value");
      const auto *spelled =
          literal ? std::get_if<std::string>(literal) : nullptr;
      if (spelled && *spelled != *targetName)
        error(*args->front(), "TypeVar name '" + *spelled +
                                  "' does not match assignment target '" +
                                  *targetName + "'");
    }

    const std::vector<parser::NodePtr> *keywords =
        nodeListField(**value, "keywords");
    if (keywords) {
      for (const parser::NodePtr &keyword : *keywords) {
        if (!keyword)
          continue;
        const std::string *argName = stringField(*keyword, "arg");
        const parser::NodePtr *keywordValue = nodeField(*keyword, "value");
        if (!argName || !keywordValue || !*keywordValue)
          continue;
        if (*argName == "bound") {
          std::optional<mlir::Type> bound = typeFromAnnotation(*keywordValue);
          if (!bound) {
            error(**keywordValue,
                  "TypeVar bound must resolve to a static type");
            continue;
          }
          parameter.bound = *bound;
        } else if (*argName == "default") {
          parameter.defaultValue = *keywordValue;
        }
      }
    }
    globalTypeVariables[*targetName] = std::move(parameter);
  }
}

std::optional<TypeAliasParameter>
Builder::Impl::parseTypeParameter(const parser::Node &param,
                                  bool allowParamSpec) {
  if (param.kind != "TypeVar" && param.kind != "TypeVarTuple" &&
      param.kind != "ParamSpec") {
    error(param, "type parameter kind '" + param.kind + "' is not supported");
    return std::nullopt;
  }
  if (param.kind == "TypeVarTuple" ||
      (param.kind == "ParamSpec" && !allowParamSpec)) {
    error(param,
          allowParamSpec
              ? "generic functions currently support TypeVar and ParamSpec "
                "parameters only"
              : "generic classes currently support TypeVar parameters only");
    return std::nullopt;
  }
  const std::string *name = stringField(param, "name");
  if (!name) {
    error(param, "type parameter name is missing");
    return std::nullopt;
  }
  TypeAliasParameter result;
  result.name = *name;
  if (param.kind == "TypeVarTuple")
    result.kind = TypeAliasParameterKind::TypeVarTuple;
  else if (param.kind == "ParamSpec")
    result.kind = TypeAliasParameterKind::ParamSpec;
  const parser::NodePtr *bound = nodeField(param, "bound");
  if (bound && *bound) {
    std::optional<mlir::Type> boundType = typeFromAnnotation(*bound);
    if (!boundType) {
      error(**bound, "type parameter bound must resolve to a static type");
      return std::nullopt;
    }
    result.bound = *boundType;
  }
  const parser::NodePtr *defaultValue = nodeField(param, "default_value");
  if (defaultValue && *defaultValue)
    result.defaultValue = *defaultValue;
  return result;
}

std::vector<TypeAliasParameter>
Builder::Impl::parseTypeParameters(const parser::Node &owner,
                                   bool allowParamSpec) {
  std::vector<TypeAliasParameter> result;
  const std::vector<parser::NodePtr> *typeParams =
      nodeListField(owner, "type_params");
  if (!typeParams)
    return result;
  std::set<std::string> seen;
  for (const parser::NodePtr &param : *typeParams) {
    if (!param)
      continue;
    std::optional<TypeAliasParameter> parsed =
        parseTypeParameter(*param, allowParamSpec);
    if (!parsed)
      continue;
    if (!seen.insert(parsed->name).second) {
      error(*param, "duplicate type parameter '" + parsed->name + "'");
      continue;
    }
    result.push_back(std::move(*parsed));
  }
  return result;
}

std::vector<TypeAliasParameter> Builder::Impl::referencedFunctionTypeParameters(
    const parser::Node &function) const {
  std::vector<TypeAliasParameter> result;
  const parser::NodePtr *argsNode = nodeField(function, "args");
  const parser::NodePtr *returns = nodeField(function, "returns");
  std::vector<const parser::Node *> annotations;
  if (argsNode && *argsNode) {
    for (llvm::StringRef fieldName : {"posonlyargs", "args", "kwonlyargs"}) {
      const std::vector<parser::NodePtr> *args =
          nodeListField(**argsNode, fieldName);
      if (!args)
        continue;
      for (const parser::NodePtr &arg : *args) {
        const parser::NodePtr *annotation =
            arg ? nodeField(*arg, "annotation") : nullptr;
        if (annotation && *annotation)
          annotations.push_back(annotation->get());
      }
    }
    for (llvm::StringRef fieldName : {"vararg", "kwarg"}) {
      const parser::NodePtr *arg = nodeField(**argsNode, fieldName);
      const parser::NodePtr *annotation =
          arg && *arg ? nodeField(**arg, "annotation") : nullptr;
      if (annotation && *annotation)
        annotations.push_back(annotation->get());
    }
  }
  if (returns && *returns)
    annotations.push_back(returns->get());

  for (const auto &entry : globalTypeVariables) {
    const std::string &name = entry.first;
    const TypeAliasParameter &parameter = entry.second;
    bool used = llvm::any_of(annotations, [&](const parser::Node *node) {
      return node && referencesName(*node, name);
    });
    if (used)
      result.push_back(parameter);
  }
  return result;
}

std::vector<TypeAliasParameter>
Builder::Impl::genericBaseTypeParameters(const parser::Node &classNode) {
  std::vector<TypeAliasParameter> result;
  const std::vector<parser::NodePtr> *bases = nodeListField(classNode, "bases");
  if (!bases)
    return result;
  std::set<std::string> seen;
  for (const parser::NodePtr &base : *bases) {
    if (!base || base->kind != "Subscript")
      continue;
    const parser::NodePtr *value = nodeField(*base, "value");
    const parser::NodePtr *slice = nodeField(*base, "slice");
    if (!value || !*value || !slice || !*slice ||
        !isTypingName(**value, "Generic"))
      continue;
    llvm::SmallVector<parser::NodePtr> items;
    if ((*slice)->kind == "Tuple") {
      const std::vector<parser::NodePtr> *elts = nodeListField(**slice, "elts");
      if (elts)
        items.append(elts->begin(), elts->end());
    } else {
      items.push_back(*slice);
    }
    for (const parser::NodePtr &item : items) {
      if (!item || item->kind != "Name")
        continue;
      const std::string *name = stringField(*item, "id");
      auto found =
          name ? globalTypeVariables.find(*name) : globalTypeVariables.end();
      if (found == globalTypeVariables.end())
        continue;
      if (!seen.insert(*name).second)
        continue;
      if (found->second.kind != TypeAliasParameterKind::TypeVar) {
        error(*item, "generic classes currently support TypeVar parameters "
                     "only");
        continue;
      }
      result.push_back(found->second);
    }
  }
  return result;
}

std::string Builder::Impl::specializationTypeKey(mlir::Type type) const {
  std::string raw;
  if (auto classType = mlir::dyn_cast<py::ClassType>(type))
    raw = classType.getClassName().str();
  else
    raw = typeString(type);
  std::string result;
  result.reserve(raw.size());
  for (unsigned char ch : raw) {
    if (std::isalnum(ch))
      result.push_back(static_cast<char>(ch));
    else
      result.push_back('_');
  }
  while (!result.empty() && result.back() == '_')
    result.pop_back();
  return result.empty() ? "type" : result;
}

std::optional<std::string>
Builder::Impl::instantiateGenericClass(const parser::Node &anchor,
                                       llvm::StringRef name,
                                       llvm::ArrayRef<mlir::Type> arguments) {
  auto found = classes.find(name.str());
  if (found == classes.end() || !found->second.isGenericTemplate ||
      !found->second.definition) {
    error(anchor, "class '" + name.str() + "' is not a generic template");
    return std::nullopt;
  }
  const ClassInfo &templ = found->second;
  if (arguments.size() != templ.typeParameters.size()) {
    error(anchor, "generic class '" + name.str() + "' expects " +
                      std::to_string(templ.typeParameters.size()) +
                      " type arguments, got " +
                      std::to_string(arguments.size()));
    return std::nullopt;
  }

  std::string key = name.str();
  std::map<std::string, mlir::Type> substitutions;
  for (auto indexed : llvm::enumerate(arguments)) {
    const TypeAliasParameter &parameter = templ.typeParameters[indexed.index()];
    mlir::Type argument = indexed.value();
    if (parameter.bound && !typeAssignable(parameter.bound, argument)) {
      error(anchor, "type argument " + typeString(argument) +
                        " does not satisfy bound " +
                        typeString(parameter.bound) + " for '" +
                        parameter.name + "'");
      return std::nullopt;
    }
    substitutions[parameter.name] = argument;
    key += "|" + specializationTypeKey(argument);
  }
  auto existing = genericClassSpecializations.find(key);
  if (existing != genericClassSpecializations.end())
    return existing->second;

  std::string specializedName = name.str();
  for (mlir::Type argument : arguments)
    specializedName += "__" + specializationTypeKey(argument);
  if (classes.count(specializedName)) {
    error(anchor, "generic class specialization name collision for '" +
                      specializedName + "'");
    return std::nullopt;
  }

  const parser::Node &classNode = *templ.definition;
  const std::vector<parser::NodePtr> *classBody =
      nodeListField(classNode, "body");
  if (!classBody) {
    error(classNode, "ClassDef.body is missing");
    return std::nullopt;
  }

  ClassInfo info;
  info.name = specializedName;
  info.definition = &classNode;
  info.templateName = name.str();
  info.typeParameters = templ.typeParameters;
  info.typeSubstitutions = substitutions;
  info.isGenericSpecialization = true;
  classes.emplace(specializedName, std::move(info));
  ClassInfo &classInfo = classes[specializedName];

  std::set<std::string> activeAliases;
  const std::vector<parser::NodePtr> *bases = nodeListField(classNode, "bases");
  if (bases) {
    for (const parser::NodePtr &base : *bases) {
      if (!base)
        continue;
      if (base->kind == "Subscript") {
        const parser::NodePtr *value = nodeField(*base, "value");
        if (value && *value && isTypingName(**value, "Generic"))
          continue;
        std::optional<mlir::Type> baseType =
            typeFromAnnotation(base, substitutions, activeAliases);
        std::optional<std::string> baseName =
            baseType ? classNameFromType(*baseType) : std::nullopt;
        if (baseName)
          classInfo.baseNames.push_back(*baseName);
        else
          error(*base, "generic class base must resolve to a class type");
        continue;
      }
      if (base->kind == "Name") {
        const std::string *baseName = stringField(*base, "id");
        if (baseName && *baseName != "Generic")
          classInfo.baseNames.push_back(*baseName);
        continue;
      }
      error(*base, "class bases must be statically named");
    }
  }

  for (const parser::NodePtr &member : *classBody) {
    if (member && member->kind == "AnnAssign") {
      const parser::NodePtr *target = nodeField(*member, "target");
      const parser::NodePtr *annotation = nodeField(*member, "annotation");
      const parser::NodePtr *value = nodeField(*member, "value");
      if (!target || !*target || (*target)->kind != "Name" || !annotation ||
          !*annotation)
        continue;
      const std::string *fieldName = stringField(**target, "id");
      std::optional<mlir::Type> fieldType =
          typeFromAnnotation(*annotation, substitutions, activeAliases);
      if (!fieldName)
        continue;
      if (!fieldType) {
        error(**annotation, "class field '" + *fieldName +
                                "' annotation must resolve to a static type");
        continue;
      }
      classInfo.fields[*fieldName] = *fieldType;
      if (value && *value)
        refineProtocolFieldFromValue(classInfo, *fieldName, **value);
      continue;
    }
    if (!member ||
        (member->kind != "FunctionDef" && member->kind != "AsyncFunctionDef"))
      continue;
    std::optional<FunctionInfo> method =
        parseMethodInfo(*member, specializedName, &substitutions);
    if (!method)
      continue;
    const std::string *methodName = stringField(*member, "name");
    if (methodName) {
      classInfo.methods[*methodName] = method->name;
      classInfo.methodNodes[*methodName] = member;
      classInfo.ownMethodNodes[*methodName] = member;
    }
    functions[method->name] = std::move(*method);
  }

  auto init = llvm::find_if(*classBody, [](const parser::NodePtr &member) {
    if (!member || member->kind != "FunctionDef")
      return false;
    const std::string *methodName = stringField(*member, "name");
    return methodName && *methodName == "__init__";
  });
  if (init != classBody->end() && *init) {
    const parser::NodePtr *argsNode = nodeField(**init, "args");
    const std::vector<parser::NodePtr> *posonlyargs =
        argsNode && *argsNode ? nodeListField(**argsNode, "posonlyargs")
                              : nullptr;
    const std::vector<parser::NodePtr> *args =
        argsNode && *argsNode ? nodeListField(**argsNode, "args") : nullptr;
    const std::vector<parser::NodePtr> *initBody =
        nodeListField(**init, "body");
    if (args && initBody) {
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
            annotation
                ? typeFromAnnotation(*annotation, substitutions, activeAliases)
                : std::nullopt;
        if (argName && argType)
          argTypes[*argName] = *argType;
      }

      for (const parser::NodePtr &statement : *initBody) {
        if (!statement)
          continue;
        if (statement->kind == "AnnAssign") {
          const parser::NodePtr *target = nodeField(*statement, "target");
          const parser::NodePtr *annotation =
              nodeField(*statement, "annotation");
          if (!target || !*target || !isSelfAttribute(**target) ||
              !annotation || !*annotation)
            continue;
          const std::string *fieldName = stringField(**target, "attr");
          std::optional<mlir::Type> fieldType =
              typeFromAnnotation(*annotation, substitutions, activeAliases);
          if (!fieldName)
            continue;
          if (!fieldType) {
            error(**annotation,
                  "instance field '" + *fieldName +
                      "' annotation must resolve to a static type");
            continue;
          }
          classInfo.fields[*fieldName] = *fieldType;
          const parser::NodePtr *value = nodeField(*statement, "value");
          if (value && *value)
            refineProtocolFieldFromValue(classInfo, *fieldName, **value,
                                         argTypes);
          continue;
        }
        if (statement->kind != "Assign")
          continue;
        const std::vector<parser::NodePtr> *targets =
            nodeListField(*statement, "targets");
        const parser::NodePtr *value = nodeField(*statement, "value");
        if (!targets || targets->size() != 1 || !targets->front() || !value ||
            !*value || !isSelfAttribute(*targets->front()))
          continue;
        const std::string *fieldName = stringField(*targets->front(), "attr");
        if (!fieldName)
          continue;
        auto field = classInfo.fields.find(*fieldName);
        if (field != classInfo.fields.end() &&
            mlir::isa<py::ProtocolType>(field->second)) {
          refineProtocolFieldFromValue(classInfo, *fieldName, **value,
                                       argTypes);
          continue;
        }
        if ((*value)->kind != "Name")
          continue;
        const std::string *valueName = stringField(**value, "id");
        if (!valueName)
          continue;
        auto argType = argTypes.find(*valueName);
        if (argType != argTypes.end())
          classInfo.fields[*fieldName] = argType->second;
      }
    }
  }

  std::set<std::string> resolving;
  resolveClassInheritance(specializedName, classNode, resolving);
  genericClassSpecializations[key] = specializedName;
  return specializedName;
}

void Builder::Impl::refineProtocolFieldFromValue(
    ClassInfo &info, llvm::StringRef fieldName, const parser::Node &value,
    const std::map<std::string, mlir::Type> &knownNames) {
  auto field = info.fields.find(fieldName.str());
  if (field == info.fields.end() || !mlir::isa<py::ProtocolType>(field->second))
    return;

  std::optional<mlir::Type> concreteType;
  if (value.kind == "Name") {
    const std::string *name = stringField(value, "id");
    auto found = name ? knownNames.find(*name) : knownNames.end();
    if (found != knownNames.end())
      concreteType = found->second;
  }
  if (!concreteType)
    concreteType = inferExpressionType(value);
  if (!concreteType) {
    if (value.kind == "List" || isNoArgCallNamed(value, "list")) {
      if (std::optional<mlir::Type> element =
              protocolIterableElement(field->second))
        concreteType = listType(*element);
    } else if (value.kind == "Dict" || isNoArgCallNamed(value, "dict")) {
      std::optional<std::pair<mlir::Type, mlir::Type>> dictTypes =
          protocolMappingKeyValueTypes(field->second);
      if (dictTypes)
        concreteType = dictType(dictTypes->first, dictTypes->second);
    }
  }
  if (!concreteType || mlir::isa<py::ProtocolType>(*concreteType) ||
      *concreteType == field->second)
    return;
  if (auto classType = mlir::dyn_cast<py::ClassType>(*concreteType)) {
    auto classFound = classes.find(classType.getClassName().str());
    if (classFound != classes.end() && classFound->second.definition &&
        classFound->second.name != info.name) {
      std::set<std::string> resolving;
      if (!resolveClassInheritance(classFound->second.name,
                                   *classFound->second.definition, resolving))
        return;
    }
  }
  if (typeAssignable(field->second, *concreteType))
    field->second = *concreteType;
}

std::optional<std::string> Builder::Impl::specializeClassForConstructorFields(
    const parser::Node &anchor, llvm::StringRef name, const FunctionInfo &init,
    llvm::ArrayRef<Value> userArgs) {
  llvm::SmallVector<mlir::Type> userArgTypes;
  userArgTypes.reserve(userArgs.size());
  for (const Value &arg : userArgs)
    userArgTypes.push_back(arg.protocolConcreteType ? *arg.protocolConcreteType
                                                    : arg.type);
  return specializeClassForConstructorFieldTypes(anchor, name, init,
                                                 userArgTypes);
}

std::optional<std::string>
Builder::Impl::specializeClassForConstructorFieldTypes(
    const parser::Node &anchor, llvm::StringRef name, const FunctionInfo &init,
    llvm::ArrayRef<mlir::Type> userArgTypes) {
  auto classFound = classes.find(name.str());
  if (classFound == classes.end() || !classFound->second.definition)
    return std::nullopt;

  ClassInfo &base = classFound->second;
  if (base.isGenericTemplate)
    return std::nullopt;
  if (!base.inheritanceResolved) {
    std::set<std::string> resolving;
    if (!resolveClassInheritance(base.name, *base.definition, resolving))
      return std::nullopt;
  }

  if (!init.definition)
    return std::nullopt;
  const std::vector<parser::NodePtr> *body =
      nodeListField(*init.definition, "body");
  if (!body)
    return std::nullopt;

  std::map<std::string, mlir::Type> concreteArgs;
  for (std::size_t formalIndex = 1; formalIndex < init.argTypes.size();
       ++formalIndex) {
    std::size_t actualIndex = formalIndex - 1;
    if (actualIndex >= userArgTypes.size() ||
        formalIndex >= init.argNames.size())
      break;
    mlir::Type formalType = init.argTypes[formalIndex];
    mlir::Type actualType = userArgTypes[actualIndex];
    if (!formalType || !actualType || !mlir::isa<py::ProtocolType>(formalType))
      continue;
    if (mlir::isa<py::ProtocolType>(actualType) || actualType == formalType)
      continue;
    if (!typeAssignable(formalType, actualType))
      continue;
    concreteArgs[init.argNames[formalIndex]] = actualType;
  }
  if (concreteArgs.empty())
    return std::nullopt;

  std::map<std::string, mlir::Type> fieldOverrides;
  for (const parser::NodePtr &stmt : *body) {
    if (!stmt)
      continue;
    std::optional<std::pair<std::string, std::string>> assignment =
        selfFieldAssignedFromName(*stmt);
    if (!assignment)
      continue;
    auto argType = concreteArgs.find(assignment->second);
    if (argType == concreteArgs.end())
      continue;
    auto field = base.fields.find(assignment->first);
    if (field == base.fields.end() ||
        !mlir::isa<py::ProtocolType>(field->second))
      continue;
    if (!typeAssignable(field->second, argType->second))
      continue;
    fieldOverrides[assignment->first] = argType->second;
  }
  if (fieldOverrides.empty())
    return std::nullopt;

  std::string key = "ctor-fields|" + base.name;
  std::string specializedName = base.name + "__ctor";
  for (const auto &[fieldName, fieldType] : fieldOverrides) {
    std::string typeKey = specializationTypeKey(fieldType);
    key += "|" + fieldName + "=" + typeKey;
    specializedName += "__" + fieldName + "__" + typeKey;
  }
  auto existing = genericClassSpecializations.find(key);
  if (existing != genericClassSpecializations.end())
    return existing->second;
  if (classes.count(specializedName)) {
    error(anchor, "constructor-specialized class name collision for '" +
                      specializedName + "'");
    return std::nullopt;
  }

  ClassInfo info;
  info.name = specializedName;
  info.definition = base.definition;
  info.templateName = base.name;
  info.typeParameters = base.typeParameters;
  info.typeSubstitutions = base.typeSubstitutions;
  info.baseNames.push_back(base.name);
  info.mro.push_back(specializedName);
  if (!base.mro.empty())
    info.mro.insert(info.mro.end(), base.mro.begin(), base.mro.end());
  else
    info.mro.push_back(base.name);
  info.fields = base.fields;
  for (const auto &[fieldName, fieldType] : fieldOverrides)
    info.fields[fieldName] = fieldType;
  info.methodNodes = base.methodNodes;
  info.ownMethodNodes = base.methodNodes;
  info.isGenericSpecialization = true;
  info.inheritanceResolved = true;

  auto inserted = classes.emplace(specializedName, std::move(info));
  ClassInfo &specialized = inserted.first->second;
  const std::map<std::string, mlir::Type> *typeVariables =
      specialized.typeSubstitutions.empty() ? nullptr
                                            : &specialized.typeSubstitutions;

  for (const auto &[methodName, methodNode] : specialized.methodNodes) {
    if (!methodNode)
      continue;
    std::optional<FunctionInfo> method =
        parseMethodInfo(*methodNode, specialized.name, typeVariables);
    if (!method)
      continue;
    if (methodName == "__init__") {
      for (std::size_t index = 1; index < method->argTypes.size(); ++index) {
        if (index >= method->argNames.size())
          break;
        auto concrete = concreteArgs.find(method->argNames[index]);
        if (concrete != concreteArgs.end())
          method->argTypes[index] = concrete->second;
      }
      refreshFunctionTypes(*method);
      callableFunctionsBySymbol[method->symbolName] = *method;
    }
    specialized.methods[methodName] = method->name;
    functions[method->name] = std::move(*method);
  }

  genericClassSpecializations[key] = specializedName;
  return specializedName;
}

std::optional<std::string> Builder::Impl::instantiateGenericClassFromAnnotation(
    const parser::Node &anchor, llvm::StringRef name,
    llvm::ArrayRef<parser::NodePtr> arguments,
    const std::map<std::string, mlir::Type> &typeVariables,
    std::set<std::string> &activeAliases) {
  llvm::SmallVector<mlir::Type> types;
  types.reserve(arguments.size());
  for (const parser::NodePtr &argument : arguments) {
    std::optional<mlir::Type> type =
        typeFromAnnotation(argument, typeVariables, activeAliases);
    if (!type)
      return std::nullopt;
    types.push_back(*type);
  }
  return instantiateGenericClass(anchor, name, types);
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

  for (const auto &[methodName, symbolName] : ownMethods) {
    if (methodName == "__init__")
      continue;
    auto derivedMethod = functions.find(symbolName);
    if (derivedMethod == functions.end())
      continue;
    for (llvm::StringRef ancestorName : llvm::ArrayRef(*mro).drop_front()) {
      const ClassInfo &ancestor = classes[ancestorName.str()];
      auto inherited = ancestor.methods.find(methodName);
      if (inherited == ancestor.methods.end())
        continue;
      auto baseMethod = functions.find(inherited->second);
      if (baseMethod == functions.end())
        continue;
      if (!sameMethodOverrideSignature(derivedMethod->second,
                                       baseMethod->second)) {
        error(anchor, "method '" + methodName + "' in class '" + info.name +
                          "' has incompatible override signature for base '" +
                          ancestor.name + "'");
        resolving.erase(info.name);
        return false;
      }
      break;
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

  if (!verifyInitFieldInitialization(info, anchor)) {
    resolving.erase(info.name);
    return false;
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
    info.definition = stmt.get();
    info.typeParameters = parseTypeParameters(*stmt, /*allowParamSpec=*/false);
    if (info.typeParameters.empty())
      info.typeParameters = genericBaseTypeParameters(*stmt);
    info.isGenericTemplate = !info.typeParameters.empty();
    const std::vector<parser::NodePtr> *bases = nodeListField(*stmt, "bases");
    if (bases) {
      for (const parser::NodePtr &base : *bases) {
        if (!base)
          continue;
        if (base->kind == "Subscript") {
          const parser::NodePtr *value = nodeField(*base, "value");
          if (value && *value && isTypingName(**value, "Generic"))
            continue;
          error(*base, "generic class bases must be specialized before use");
          continue;
        }
        if (base->kind != "Name") {
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
    if (hasNodeListEntries(*stmt, "decorator_list") ||
        hasNodeListEntries(*stmt, "keywords"))
      continue;
    const std::string *className = stringField(*stmt, "name");
    const std::vector<parser::NodePtr> *classBody =
        nodeListField(*stmt, "body");
    if (!className || !classBody)
      continue;
    ClassInfo &classInfo = classes[*className];
    if (classInfo.isGenericTemplate)
      continue;

    for (const parser::NodePtr &member : *classBody) {
      if (member && member->kind == "AnnAssign") {
        const parser::NodePtr *target = nodeField(*member, "target");
        const parser::NodePtr *annotation = nodeField(*member, "annotation");
        const parser::NodePtr *value = nodeField(*member, "value");
        if (!target || !*target || (*target)->kind != "Name" || !annotation ||
            !*annotation)
          continue;
        const std::string *fieldName = stringField(**target, "id");
        std::optional<mlir::Type> fieldType = typeFromAnnotation(*annotation);
        if (!fieldName)
          continue;
        if (!fieldType) {
          error(**annotation, "class field '" + *fieldName +
                                  "' annotation must resolve to a static type");
          continue;
        }
        classInfo.fields[*fieldName] = *fieldType;
        if (value && *value)
          refineProtocolFieldFromValue(classInfo, *fieldName, **value);
        continue;
      }
      if (!member ||
          (member->kind != "FunctionDef" && member->kind != "AsyncFunctionDef"))
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
        if (!fieldName)
          continue;
        if (!fieldType) {
          error(**annotation, "instance field '" + *fieldName +
                                  "' annotation must resolve to a static type");
          continue;
        }
        classInfo.fields[*fieldName] = *fieldType;
        const parser::NodePtr *value = nodeField(*statement, "value");
        if (value && *value)
          refineProtocolFieldFromValue(classInfo, *fieldName, **value,
                                       argTypes);
        continue;
      }
      if (statement->kind == "Assign") {
        const std::vector<parser::NodePtr> *targets =
            nodeListField(*statement, "targets");
        const parser::NodePtr *value = nodeField(*statement, "value");
        if (!targets || targets->size() != 1 || !targets->front() || !value ||
            !*value || !isSelfAttribute(*targets->front()))
          continue;
        const std::string *fieldName = stringField(*targets->front(), "attr");
        if (!fieldName)
          continue;
        auto field = classInfo.fields.find(*fieldName);
        if (field != classInfo.fields.end() &&
            mlir::isa<py::ProtocolType>(field->second)) {
          refineProtocolFieldFromValue(classInfo, *fieldName, **value,
                                       argTypes);
          continue;
        }
        if ((*value)->kind != "Name")
          continue;
        const std::string *valueName = stringField(**value, "id");
        if (!valueName)
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
    if (hasNodeListEntries(*stmt, "decorator_list") ||
        hasNodeListEntries(*stmt, "keywords"))
      continue;
    const std::string *className = stringField(*stmt, "name");
    if (!className)
      continue;
    auto classFound = classes.find(*className);
    if (classFound != classes.end() && classFound->second.isGenericTemplate)
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
    if (stmt && stmt->kind == "ClassDef" &&
        !hasNodeListEntries(*stmt, "decorator_list") &&
        !hasNodeListEntries(*stmt, "keywords")) {
      const std::string *name = stringField(*stmt, "name");
      auto found = name ? classes.find(*name) : classes.end();
      if (found != classes.end() && found->second.isGenericTemplate)
        continue;
      emitClassDef(*stmt);
    }
  }
  emitPendingGenericClassDefs();
}

void Builder::Impl::emitClassDef(const parser::Node &classNode) {
  const std::string *name = stringField(classNode, "name");
  if (!name)
    return;
  auto found = classes.find(*name);
  if (found == classes.end())
    return;
  emitClassInfo(classNode, found->second);
}

void Builder::Impl::emitPendingGenericClassDefs() {
  bool emittedAny = true;
  while (emittedAny) {
    emittedAny = false;
    std::vector<std::string> pending;
    for (const auto &[name, info] : classes) {
      if (!info.isGenericSpecialization || emittedClasses.count(name) ||
          !info.definition)
        continue;
      pending.push_back(name);
    }
    for (const std::string &name : pending) {
      auto found = classes.find(name);
      if (found == classes.end() || !found->second.definition)
        continue;
      emitClassInfo(*found->second.definition, found->second);
      emittedAny = true;
    }
  }
}

void Builder::Impl::emitClassInfo(const parser::Node &classNode,
                                  const ClassInfo &info) {
  if (!emittedClasses.insert(info.name).second)
    return;

  llvm::SmallVector<mlir::Attribute> fieldNames;
  llvm::SmallVector<mlir::Attribute> fieldTypes;
  for (const auto &[fieldName, fieldType] : info.fields) {
    fieldNames.push_back(builder.getStringAttr(fieldName));
    fieldTypes.push_back(mlir::TypeAttr::get(fieldType));
  }

  llvm::SmallVector<mlir::Attribute> methodNameAttrs;
  llvm::SmallVector<mlir::Attribute> methodTypes;
  llvm::SmallVector<mlir::Attribute> methodKinds;
  for (const auto &[methodName, symbolName] : info.methods) {
    auto functionFound = functions.find(symbolName);
    if (functionFound == functions.end())
      continue;
    const FunctionInfo &method = functionFound->second;
    methodNameAttrs.push_back(builder.getStringAttr(methodName));
    methodTypes.push_back(mlir::TypeAttr::get(method.signatureType));
    llvm::StringRef methodKind = method.methodKind.empty()
                                     ? llvm::StringRef(kInstanceMethodKind)
                                     : llvm::StringRef(method.methodKind);
    methodKinds.push_back(builder.getStringAttr(methodKind));
  }

  py::ClassOp classOp = builder.create<py::ClassOp>(
      loc(classNode), info.name, stringArrayAttr(info.baseNames),
      fieldNames.empty() ? mlir::ArrayAttr{} : builder.getArrayAttr(fieldNames),
      fieldTypes.empty() ? mlir::ArrayAttr{} : builder.getArrayAttr(fieldTypes),
      methodNameAttrs.empty() ? mlir::ArrayAttr{}
                              : builder.getArrayAttr(methodNameAttrs),
      methodTypes.empty() ? mlir::ArrayAttr{}
                          : builder.getArrayAttr(methodTypes),
      methodKinds.empty() ? mlir::ArrayAttr{}
                          : builder.getArrayAttr(methodKinds));
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
  if (info.isAsync) {
    emitAsyncMethodDef(function, info);
    return;
  }

  const std::vector<parser::NodePtr> *body = nodeListField(function, "body");
  if (!body) {
    error(function, "method body is missing");
    return;
  }

  llvm::ArrayRef<std::string> allArgNames(info.argNames);
  py::CallableFuncOp func =
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
    Value value{arg, info.varargType};
    value.paramSpecArgs = info.varargParameterPack;
    symbols.emplace(*info.varargName, value);
  }
  if (info.kwargName && info.kwargType) {
    mlir::Value arg = func.getBody().front().getArgument(
        info.argTypes.size() + (info.varargType ? 1 : 0));
    Value value{arg, info.kwargType};
    value.paramSpecKwargs = info.kwargParameterPack;
    symbols.emplace(*info.kwargName, value);
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

void Builder::Impl::emitAsyncMethodDef(const parser::Node &function,
                                       const FunctionInfo &info) {
  const std::vector<parser::NodePtr> *body = nodeListField(function, "body");
  if (!body) {
    error(function, "async method body is missing");
    return;
  }

  auto func = builder.create<mlir::async::FuncOp>(
      loc(function), info.symbolName, info.asyncFunctionType);
  func->setAttr(info.mayThrow ? "maythrow" : "nothrow", builder.getUnitAttr());
  if (info.mutatesSelf)
    func->setAttr("mutates_self", builder.getUnitAttr());
  addAsyncEntryBlock(func.getOperation(), info.argTypes);

  std::map<std::string, Value> savedSymbols = std::move(symbols);
  std::map<std::string, FunctionInfo> savedCallableAliases =
      std::move(callableAliases);
  mlir::Type savedReturnType = currentReturnType;
  bool savedTerminated = blockTerminated;
  bool savedInNativeFunction = inNativeFunction;
  bool savedInAsyncFunction = inAsyncFunction;
  unsigned savedExceptionContextDepth = exceptionContextDepth;
  functionSymbolStack.push_back(info.symbolName);
  symbols.clear();
  callableAliases.clear();
  currentReturnType = info.resultType;
  blockTerminated = false;
  inNativeFunction = false;
  inAsyncFunction = true;
  exceptionContextDepth = 0;
  for (auto indexed : llvm::enumerate(info.argNames)) {
    mlir::Value arg = func.getBody().front().getArgument(indexed.index());
    symbols.emplace(indexed.value(),
                    Value{arg, info.argTypes[indexed.index()]});
  }

  for (const parser::NodePtr &stmt : *body) {
    if (stmt && !blockTerminated)
      emitStatement(*stmt);
  }
  if (!blockTerminated) {
    if (info.resultType == noneType()) {
      mlir::Value none = builder.create<py::NoneOp>(loc(function), noneType());
      builder.create<mlir::async::ReturnOp>(loc(function),
                                            mlir::ValueRange{none});
    } else {
      error(function, "async method may exit without returning " +
                          typeString(info.resultType));
    }
  }

  symbols = std::move(savedSymbols);
  callableAliases = std::move(savedCallableAliases);
  currentReturnType = savedReturnType;
  blockTerminated = savedTerminated;
  inNativeFunction = savedInNativeFunction;
  inAsyncFunction = savedInAsyncFunction;
  exceptionContextDepth = savedExceptionContextDepth;
  functionSymbolStack.pop_back();
  builder.setInsertionPointToEnd(module->getBody());
}

Value Builder::Impl::emitClassConstructorCall(
    const parser::Node &expr, llvm::StringRef name,
    const std::vector<parser::NodePtr> &args) {
  auto classFound = classes.find(name.str());
  auto createInstance = [&](llvm::StringRef className) {
    mlir::Type resultType = classType(className);
    mlir::Value instance =
        builder.create<py::ClassNewOp>(loc(expr), resultType, className);
    return markExactClass(Value{instance, resultType}, className);
  };

  if (classFound == classes.end())
    return createInstance(name);
  auto initFound = classFound->second.methods.find("__init__");
  if (initFound == classFound->second.methods.end())
    return createInstance(name);
  auto functionFound = functions.find(initFound->second);
  if (functionFound == functions.end())
    return createInstance(name);

  const FunctionInfo *init = &functionFound->second;
  if (hasCallableMetadata(*init)) {
    Value instanceValue = createInstance(name);
    Value receiver = instanceValue;
    std::optional<CallArgumentTuples> tuples = emitExplicitCallArgumentTuples(
        expr, *init, args, /*firstFormal=*/1, llvm::ArrayRef<Value>{receiver});
    if (!tuples)
      return Value{{}, classType(name)};
    Value callee = emitFunctionObject(expr, *init);
    if (!callee.value)
      return Value{{}, classType(name)};
    builder.create<py::CallOp>(loc(expr), mlir::TypeRange{noneType()},
                               callee.value, tuples->posargs.value,
                               tuples->kwnames.value, tuples->kwvalues.value);
    return instanceValue;
  }

  std::optional<std::vector<Value>> userArgs =
      emitStaticArguments(expr, *init, args, /*firstFormal=*/1);
  if (!userArgs)
    return Value{{}, classType(name)};

  std::string actualClassName = name.str();
  if (std::optional<std::string> specialized =
          specializeClassForConstructorFields(expr, name, *init, *userArgs)) {
    actualClassName = *specialized;
    auto specializedClass = classes.find(actualClassName);
    if (specializedClass != classes.end() &&
        !emittedClasses.count(actualClassName) &&
        specializedClass->second.definition) {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToEnd(module->getBody());
      emitClassInfo(*specializedClass->second.definition,
                    specializedClass->second);
    }

    classFound = classes.find(actualClassName);
    if (classFound != classes.end()) {
      initFound = classFound->second.methods.find("__init__");
      if (initFound != classFound->second.methods.end()) {
        functionFound = functions.find(initFound->second);
        if (functionFound != functions.end())
          init = &functionFound->second;
      }
    }
  }

  Value instanceValue = createInstance(actualClassName);
  Value receiver = instanceValue;
  std::vector<Value> initArgs;
  initArgs.push_back(receiver);
  initArgs.insert(initArgs.end(), userArgs->begin(), userArgs->end());
  if (initArgs.size() != init->argTypes.size()) {
    error(expr, "constructor '" + actualClassName + "' expects " +
                    std::to_string(init->argTypes.size() - 1) +
                    " arguments, got " + std::to_string(userArgs->size()));
    return Value{{}, classType(actualClassName)};
  }
  for (std::size_t index = 0; index < initArgs.size(); ++index) {
    initArgs[index] = coerceToExpectedType(expr, std::move(initArgs[index]),
                                           init->argTypes[index]);
    if (!typeAssignable(init->argTypes[index], initArgs[index].type)) {
      error(expr, "constructor '" + actualClassName + "' argument " +
                      std::to_string(index) + " type mismatch: expected " +
                      typeString(init->argTypes[index]) + ", got " +
                      typeString(initArgs[index].type));
      return Value{{}, classType(actualClassName)};
    }
  }

  mlir::Value callee = builder.create<py::CallableObjectOp>(
      loc(expr), init->functionType, init->name);
  CallArgumentTuples tuples = emitCallArgumentTuples(*init, initArgs);
  builder.create<py::CallOp>(loc(expr), mlir::TypeRange{noneType()}, callee,
                             tuples.posargs.value, tuples.kwnames.value,
                             tuples.kwvalues.value);
  return instanceValue;
}

} // namespace lython::emitter
