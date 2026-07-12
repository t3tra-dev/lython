#include "TypeSystem.h"

#include "AstAccess.h"
#include "CandidateSelection.h"
#include "PlatformConstants.h"
#include "PrimitiveTypes.h"
#include "PyCallableShape.h"
#include "PyProtocols.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <map>
#include <string>

namespace lython::emitter {
namespace {

bool isNoneConstant(const parser::Node *node) {
  return node && node->kind == "Constant" && ast::isNoneField(*node, "value");
}

std::string literalSpelling(const parser::Node &constant) {
  if (ast::isNoneField(constant, "value"))
    return "None";
  if (auto value = ast::boolean(constant, "value"))
    return *value ? "True" : "False";
  if (auto value = ast::integer(constant, "value"))
    return std::to_string(*value);
  if (auto value = ast::string(constant, "value"))
    return "\"" + std::string(*value) + "\"";
  if (const auto *fieldValue = ast::field(constant, "value")) {
    if (const auto *big = std::get_if<parser::BigInteger>(fieldValue))
      return big->decimal;
  }
  return "object";
}

std::string typeText(mlir::Type type) {
  if (!type)
    return "<unknown>";
  std::string result;
  llvm::raw_string_ostream stream(result);
  stream << type;
  return stream.str();
}

CallInferenceResult unresolvedMethodCall(const AlgorithmM &types,
                                         mlir::Type receiverType,
                                         llvm::StringRef methodName) {
  return CallInferenceResult{
      {},
      {},
      false,
      ("static type " + typeText(types.widenLiteral(receiverType)) +
       " does not provide manifest method '" + methodName.str() + "'")};
}

CallInferenceResult unresolvedCallable(mlir::Type calleeType,
                                       llvm::StringRef detail) {
  std::string message =
      "static type " + typeText(calleeType) + " is not callable";
  if (!detail.empty()) {
    message += ": ";
    message += detail;
  }
  return CallInferenceResult{{}, {}, false, std::move(message)};
}

llvm::SmallVector<parser::NodePtr, 8>
concatArgs(const parser::Node &arguments, unsigned &positionalOnlyCount) {
  llvm::SmallVector<parser::NodePtr, 8> result;
  if (const auto *posOnly = ast::nodeList(arguments, "posonlyargs")) {
    positionalOnlyCount = static_cast<unsigned>(posOnly->size());
    result.append(posOnly->begin(), posOnly->end());
  }
  if (const auto *args = ast::nodeList(arguments, "args"))
    result.append(args->begin(), args->end());
  return result;
}

bool hasDefault(std::size_t index, std::size_t total, std::size_t defaults) {
  return defaults != 0 && index + defaults >= total;
}

py::CallableType makeZeroArgStrCallable(const AlgorithmM &types) {
  mlir::MLIRContext *context = &types.getContext();
  llvm::SmallVector<mlir::Type, 1> results{types.strType()};
  return py::CallableType::get(context, {}, {}, {}, {}, results);
}

mlir::Type inferAsyncioSleepResult(const AlgorithmM &types,
                                   mlir::ArrayRef<mlir::Type> positional,
                                   mlir::ArrayRef<CallKeywordType> keywords) {
  mlir::Type payload = types.none();
  if (positional.size() > 1)
    payload = positional[1];
  for (const CallKeywordType &keyword : keywords)
    if (keyword.name == "result")
      payload = keyword.type;
  return types.contract("types.CoroutineType", {types.any(), types.any(),
                                                types.widenLiteral(payload)});
}

bool appendStarredCallArgumentTypes(const AlgorithmM &types, mlir::Type type,
                                    llvm::SmallVectorImpl<mlir::Type> &out);

void recordInferenceFailure(
    llvm::SmallVectorImpl<std::string> *failureReasons, std::string reason) {
  if (failureReasons && !reason.empty())
    failureReasons->push_back(std::move(reason));
}

mlir::Type inferExprWithLocalCallables(
    const AlgorithmM &types, const parser::Node *node,
    const llvm::StringMap<mlir::Type> &localCallables,
    llvm::SmallVectorImpl<std::string> *failureReasons = nullptr) {
  if (!node)
    return types.inferExpr(node);
  auto fail = [&](std::string reason) -> mlir::Type {
    recordInferenceFailure(failureReasons, std::move(reason));
    return {};
  };
  auto inferCallArguments =
      [&](llvm::SmallVectorImpl<mlir::Type> &positional,
          llvm::SmallVectorImpl<CallKeywordType> &keywords) -> bool {
    if (const auto *args = ast::nodeList(*node, "args")) {
      for (const parser::NodePtr &arg : *args) {
        if (arg && arg->kind == "Starred") {
          mlir::Type starredType = inferExprWithLocalCallables(
              types, ast::node(*arg, "value"), localCallables, failureReasons);
          if (!starredType)
            return false;
          if (!appendStarredCallArgumentTypes(types, starredType, positional)) {
            recordInferenceFailure(
                failureReasons,
                "starred call arguments require a statically sized tuple");
            return false;
          }
          continue;
        }
        mlir::Type argType = inferExprWithLocalCallables(
            types, arg.get(), localCallables, failureReasons);
        if (!argType)
          return false;
        positional.push_back(argType);
      }
    }

    if (const auto *keywordNodes = ast::nodeList(*node, "keywords")) {
      for (const parser::NodePtr &keyword : *keywordNodes) {
        auto name = ast::string(*keyword, "arg");
        if (!name) {
          recordInferenceFailure(
              failureReasons,
              "keyword splat call arguments require static keyword names");
          return false;
        }
        mlir::Type keywordType = inferExprWithLocalCallables(
            types, ast::node(*keyword, "value"), localCallables, failureReasons);
        if (!keywordType)
          return false;
        keywords.push_back(CallKeywordType{std::string(*name), keywordType});
      }
    }
    return true;
  };
  if (node && node->kind == "Name") {
    auto found = localCallables.find(ast::nameSpelling(*node));
    if (found != localCallables.end())
      return found->second;
  }
  if (node->kind == "List" || node->kind == "Tuple") {
    llvm::SmallVector<mlir::Type, 8> elementTypes;
    if (const auto *elements = ast::nodeList(*node, "elts")) {
      elementTypes.reserve(elements->size());
      for (const parser::NodePtr &element : *elements) {
        mlir::Type elementType = inferExprWithLocalCallables(
            types, element.get(), localCallables, failureReasons);
        if (!elementType)
          return {};
        elementTypes.push_back(types.widenLiteral(elementType));
      }
    }
    mlir::Type joined = types.join(elementTypes);
    return node->kind == "List" ? types.listOf(joined) : types.tupleOf(joined);
  }
  if (node->kind == "Dict") {
    const auto *keys = ast::nodeList(*node, "keys");
    const auto *values = ast::nodeList(*node, "values");
    if (!keys || !values)
      return types.inferExpr(node);
    llvm::SmallVector<mlir::Type, 8> keyTypes;
    llvm::SmallVector<mlir::Type, 8> valueTypes;
    for (auto [index, key] : llvm::enumerate(*keys)) {
      if (!key)
        return types.inferExpr(node);
      mlir::Type keyType = inferExprWithLocalCallables(
          types, key.get(), localCallables, failureReasons);
      if (!keyType)
        return {};
      keyTypes.push_back(types.widenLiteral(keyType));
      if (index < values->size()) {
        mlir::Type valueType = inferExprWithLocalCallables(
            types, (*values)[index].get(), localCallables, failureReasons);
        if (!valueType)
          return {};
        valueTypes.push_back(types.widenLiteral(valueType));
      }
    }
    return types.dictOf(types.join(keyTypes), types.join(valueTypes));
  }
  if (node->kind == "Subscript") {
    if (std::optional<PrimitiveTypeSpec> primitive =
            primitiveTypeSpecFromSubscript(node, types))
      return types.typeObject(primitive->type);
    mlir::Type container = inferExprWithLocalCallables(
        types, ast::node(*node, "value"), localCallables, failureReasons);
    mlir::Type index = inferExprWithLocalCallables(
        types, ast::node(*node, "slice"), localCallables, failureReasons);
    if (!container || !index)
      return {};
    CallInferenceResult inference = types.inferMethodCallWithEvidence(
        types.widenLiteral(container), "__getitem__", {index});
    if (inference)
      return inference.resultType;
    return fail(inference.failureReason);
  }
  if (node->kind == "Await") {
    mlir::Type awaitable = inferExprWithLocalCallables(
        types, ast::node(*node, "value"), localCallables, failureReasons);
    if (!awaitable)
      return {};
    AwaitInferenceResult inference =
        types.inferAwaitWithEvidence(types.widenLiteral(awaitable));
    if (inference)
      return inference.resultType;
    return fail(inference.failureReason);
  }
  if (node->kind == "Call") {
    const parser::Node *callee = ast::node(*node, "func");
    if (std::optional<PrimitiveTypeSpec> primitive =
            primitiveTypeSpecFromSubscript(callee, types))
      return primitive->type;

    llvm::SmallVector<mlir::Type, 8> positional;
    llvm::SmallVector<CallKeywordType, 4> keywords;
    if (!inferCallArguments(positional, keywords))
      return {};

    if (callee && callee->kind == "Name") {
      auto found = localCallables.find(ast::nameSpelling(*callee));
      if (found != localCallables.end()) {
        CallInferenceResult inference =
            types.inferCallWithEvidence(found->second, positional, keywords);
        if (inference)
          return inference.resultType;
        return fail(inference.failureReason);
      }
      llvm::StringRef name = ast::nameSpelling(*callee);
      if (name == "isinstance")
        return types.boolType();
      if (name == "len") {
        if (positional.empty())
          return fail("len expects one positional argument");
        CallInferenceResult inference = types.inferMethodCallWithEvidence(
            types.widenLiteral(positional.front()), "__len__", {});
        if (inference)
          return inference.resultType;
        return fail(inference.failureReason);
      }
      if (name == "round") {
        if (positional.empty())
          return fail("round expects at least one positional argument");
        llvm::SmallVector<mlir::Type, 1> extra;
        if (positional.size() > 1)
          extra.push_back(positional[1]);
        mlir::Type input = types.widenLiteral(positional.front());
        CallInferenceResult inference =
            types.inferMethodCallWithEvidence(input, "__round__", extra);
        if (inference)
          return inference.resultType;
        if (input == types.intType())
          return types.intType();
        return fail(inference.failureReason);
      }
      if (name == "range")
        return types.contract("builtins.range");
      if (auto cls = types.lookupClass(name)) {
        mlir::Type instance =
            types.inferClassInstantiation(*cls, positional, keywords);
        if (instance)
          return instance;
        return fail("class instantiation leaves unbound static type "
                    "parameters for '" +
                    name.str() + "'");
      }
      if (std::optional<std::string> canonical =
              types.lookupCanonicalBinding(name)) {
        if (*canonical == "lyrt.from_prim" && positional.size() == 1 &&
            keywords.empty()) {
          mlir::Type result = primitivePythonResultType(positional.front(), types);
          if (result)
            return result;
          return fail(
              "lyrt.from_prim expects a primitive scalar or shaped primitive value");
        }
        if (*canonical == "asyncio.sleep")
          return inferAsyncioSleepResult(types, positional, keywords);
      }
      if (auto symbol = types.lookupSymbol(name)) {
        CallInferenceResult inference =
            types.inferCallWithEvidence(*symbol, positional, keywords);
        if (inference)
          return inference.resultType;
        return fail(inference.failureReason);
      }
    }
    if (callee && callee->kind == "Attribute") {
      std::string qualified = ast::qualifiedName(callee);
      if (std::optional<std::string> canonical =
              types.lookupCanonicalBinding(qualified)) {
        if (*canonical == "lyrt.from_prim" && positional.size() == 1 &&
            keywords.empty()) {
          mlir::Type result = primitivePythonResultType(positional.front(), types);
          if (result)
            return result;
          return fail(
              "lyrt.from_prim expects a primitive scalar or shaped primitive value");
        }
        if (*canonical == "asyncio.sleep")
          return inferAsyncioSleepResult(types, positional, keywords);
      }
      if (auto symbol = types.lookupSymbol(qualified)) {
        CallInferenceResult inference =
            types.inferCallWithEvidence(*symbol, positional, keywords);
        if (inference)
          return inference.resultType;
        return fail(inference.failureReason);
      }
      if (const parser::Node *receiverNode = ast::node(*callee, "value")) {
        if (auto methodName = ast::string(*callee, "attr")) {
          mlir::Type receiver = inferExprWithLocalCallables(
              types, receiverNode, localCallables, failureReasons);
          if (!receiver)
            return {};
          CallInferenceResult inference = types.inferMethodCallWithEvidence(
              types.widenLiteral(receiver), *methodName, positional, keywords);
          if (inference)
            return inference.resultType;
          return fail(inference.failureReason);
        }
      }
    }
    if (callee) {
      mlir::Type calleeType = inferExprWithLocalCallables(
          types, callee, localCallables, failureReasons);
      if (!calleeType)
        return {};
      CallInferenceResult inference =
          types.inferCallWithEvidence(calleeType, positional, keywords);
      if (inference)
        return inference.resultType;
      return fail(inference.failureReason);
    }
    return fail("call expression is missing a callee");
  }
  if (node->kind == "Lambda") {
    return types.functionSignature(*node).callable;
  }
  if (node->kind == "UnaryOp") {
    const parser::Node *operandNode = ast::node(*node, "operand");
    mlir::Type operand =
        inferExprWithLocalCallables(types, operandNode, localCallables,
                                    failureReasons);
    if (!operand)
      return {};
    return types.inferExpr(node);
  }
  if (node->kind == "BinOp") {
    const parser::Node *leftNode = ast::node(*node, "left");
    const parser::Node *rightNode = ast::node(*node, "right");
    if (!inferExprWithLocalCallables(types, leftNode, localCallables,
                                     failureReasons))
      return {};
    if (!inferExprWithLocalCallables(types, rightNode, localCallables,
                                     failureReasons))
      return {};
  }
  return types.inferExpr(node);
}

mlir::Type inferReturnExpr(const AlgorithmM &types, const parser::Node *node,
                           const llvm::StringMap<mlir::Type> &localCallables,
                           llvm::SmallVectorImpl<std::string> *failureReasons) {
  return inferExprWithLocalCallables(types, node, localCallables,
                                     failureReasons);
}

llvm::StringMap<mlir::Type>
localCallableTypesInFunction(const AlgorithmM &types,
                             const parser::Node &function) {
  llvm::StringMap<mlir::Type> localCallables;
  if (const auto *body = ast::nodeList(function, "body")) {
    for (const parser::NodePtr &statement : *body) {
      if (!statement || (statement->kind != "FunctionDef" &&
                         statement->kind != "AsyncFunctionDef"))
        continue;
      if (auto name = ast::string(*statement, "name"))
        localCallables[*name] =
            types.functionSignature(*statement).publicCallable;
    }
  }
  return localCallables;
}

void collectReturnTypes(const AlgorithmM &types, const parser::Node *node,
                        const llvm::StringMap<mlir::Type> &localCallables,
                        llvm::SmallVectorImpl<mlir::Type> &results,
                        llvm::SmallVectorImpl<std::string> *failureReasons) {
  if (!node)
    return;
  if (node->kind == "FunctionDef" || node->kind == "AsyncFunctionDef" ||
      node->kind == "Lambda" || node->kind == "ClassDef")
    return;
  if (node->kind == "Return") {
    mlir::Type type =
        inferReturnExpr(types, ast::node(*node, "value"), localCallables,
                        failureReasons);
    if (type)
      results.push_back(type);
    return;
  }
  for (const parser::Field &field : node->fields) {
    if (const auto *child = std::get_if<parser::NodePtr>(&field.value)) {
      if (*child)
        collectReturnTypes(types, child->get(), localCallables, results,
                           failureReasons);
    } else if (const auto *children =
                   std::get_if<std::vector<parser::NodePtr>>(&field.value)) {
      for (const parser::NodePtr &child : *children)
        collectReturnTypes(types, child.get(), localCallables, results,
                           failureReasons);
    }
  }
}

mlir::Type inferredFunctionResult(const AlgorithmM &types,
                                  const parser::Node &function,
                                  llvm::SmallVectorImpl<std::string>
                                      *failureReasons = nullptr) {
  llvm::StringMap<mlir::Type> localCallables =
      localCallableTypesInFunction(types, function);

  llvm::SmallVector<mlir::Type, 4> results;
  if (const auto *body = ast::nodeList(function, "body"))
    for (const parser::NodePtr &statement : *body)
      collectReturnTypes(types, statement.get(), localCallables, results,
                         failureReasons);
  return results.empty() ? types.none() : types.join(results);
}

struct GeneratorFunctionAnalysis {
  bool hasYield = false;
  bool sawYieldFrom = false;
  bool hasReturnValue = false;
  llvm::SmallVector<mlir::Type, 4> yieldTypes;
  llvm::SmallVector<mlir::Type, 4> returnTypes;
  llvm::SmallVector<std::string, 4> failureReasons;
};

std::optional<mlir::Type> generatorYieldFromElementType(
    const AlgorithmM &types, const parser::Node *value,
    const llvm::StringMap<mlir::Type> &localCallables,
    llvm::SmallVectorImpl<std::string> &failureReasons) {
  mlir::Type rawSource =
      inferExprWithLocalCallables(types, value, localCallables, &failureReasons);
  if (!rawSource)
    return std::nullopt;
  mlir::Type source = types.widenLiteral(rawSource);
  YieldFromInferenceResult inference = types.inferYieldFromWithEvidence(source);
  if (inference)
    return inference.elementType;
  failureReasons.push_back(
      inference.failureReason.empty()
          ? std::string("yield from requires manifest-backed iterable evidence")
          : inference.failureReason);
  return std::nullopt;
}

void collectGeneratorFunctionAnalysis(
    const AlgorithmM &types, const parser::Node *node,
    const llvm::StringMap<mlir::Type> &localCallables,
    mlir::Type generatorSendHint, GeneratorFunctionAnalysis &analysis) {
  if (!node)
    return;
  if (node->kind == "FunctionDef" || node->kind == "AsyncFunctionDef" ||
      node->kind == "Lambda" || node->kind == "ClassDef")
    return;
  if (node->kind == "Yield") {
    analysis.hasYield = true;
    const parser::Node *value = ast::node(*node, "value");
    if (!value) {
      analysis.yieldTypes.push_back(types.none());
      return;
    }
    mlir::Type valueType = inferExprWithLocalCallables(
        types, value, localCallables, &analysis.failureReasons);
    if (valueType)
      analysis.yieldTypes.push_back(types.widenLiteral(valueType));
    return;
  }
  if (node->kind == "YieldFrom") {
    analysis.hasYield = true;
    analysis.sawYieldFrom = true;
    if (std::optional<mlir::Type> element = generatorYieldFromElementType(
            types, ast::node(*node, "value"), localCallables,
            analysis.failureReasons))
      analysis.yieldTypes.push_back(*element);
    return;
  }
  if (node->kind == "Return") {
    const parser::Node *value = ast::node(*node, "value");
    if (value)
      analysis.hasReturnValue = true;
    analysis.returnTypes.push_back(
        value ? inferReturnExpr(types, value, localCallables,
                                &analysis.failureReasons)
              : types.none());
    return;
  }
  if (node->kind == "Assign") {
    const parser::Node *value = ast::node(*node, "value");
    collectGeneratorFunctionAnalysis(types, value, localCallables,
                                     generatorSendHint, analysis);
    mlir::Type valueType = value && value->kind == "Yield" && generatorSendHint
                               ? generatorSendHint
                               : types.inferExpr(value);
    if (const auto *targets = ast::nodeList(*node, "targets")) {
      for (const parser::NodePtr &target : *targets) {
        if (target && target->kind == "Name")
          types.bindLocalSymbol(ast::nameSpelling(*target), valueType);
      }
    }
    return;
  }
  if (node->kind == "AnnAssign") {
    const parser::Node *value = ast::node(*node, "value");
    collectGeneratorFunctionAnalysis(types, value, localCallables,
                                     generatorSendHint, analysis);
    const parser::Node *target = ast::node(*node, "target");
    if (target && target->kind == "Name") {
      mlir::Type type = types.annotationType(ast::node(*node, "annotation"));
      if (!type && value)
        type = types.inferExpr(value);
      if (type)
        types.bindLocalSymbol(ast::nameSpelling(*target), type);
    }
    return;
  }
  if (node->kind == "AugAssign") {
    const parser::Node *target = ast::node(*node, "target");
    const parser::Node *value = ast::node(*node, "value");
    collectGeneratorFunctionAnalysis(types, value, localCallables,
                                     generatorSendHint, analysis);
    if (target && target->kind == "Name") {
      mlir::Type lhs = types.widenLiteral(types.inferExpr(target));
      mlir::Type rhs = types.widenLiteral(types.inferExpr(value));
      types.bindLocalSymbol(ast::nameSpelling(*target),
                            types.widenLiteral(types.join({lhs, rhs})));
    }
    return;
  }
  if (node->kind == "For" || node->kind == "AsyncFor") {
    // Bind the loop target to the iteration element type before the generic
    // child walk reaches the body, so yields over the target infer correctly.
    const parser::Node *target = ast::node(*node, "target");
    const parser::Node *iter = ast::node(*node, "iter");
    if (target && target->kind == "Name" && iter) {
      mlir::Type iterableType = inferExprWithLocalCallables(
          types, iter, localCallables, &analysis.failureReasons);
      if (iterableType) {
        CallInferenceResult iterInference = types.inferMethodCallWithEvidence(
            types.widenLiteral(iterableType), "__iter__", {});
        CallInferenceResult nextInference =
            iterInference ? types.inferMethodCallWithEvidence(
                                iterInference.resultType, "__next__", {})
                          : CallInferenceResult{};
        if (nextInference)
          types.bindLocalSymbol(ast::nameSpelling(*target),
                                types.widenLiteral(nextInference.resultType));
      }
    }
  }
  for (const parser::Field &field : node->fields) {
    if (const auto *child = std::get_if<parser::NodePtr>(&field.value)) {
      if (*child)
        collectGeneratorFunctionAnalysis(types, child->get(), localCallables,
                                         generatorSendHint, analysis);
    } else if (const auto *children =
                   std::get_if<std::vector<parser::NodePtr>>(&field.value)) {
      for (const parser::NodePtr &child : *children)
        collectGeneratorFunctionAnalysis(types, child.get(), localCallables,
                                         generatorSendHint, analysis);
    }
  }
}

GeneratorFunctionAnalysis
analyzeGeneratorFunction(const AlgorithmM &types, const parser::Node &function,
                         mlir::Type generatorSendHint = {}) {
  GeneratorFunctionAnalysis analysis;
  llvm::StringMap<mlir::Type> localCallables =
      localCallableTypesInFunction(types, function);
  if (const auto *body = ast::nodeList(function, "body"))
    for (const parser::NodePtr &statement : *body)
      collectGeneratorFunctionAnalysis(types, statement.get(), localCallables,
                                       generatorSendHint, analysis);
  return analysis;
}

std::optional<mlir::Type>
generatorSendTypeFromAnnotation(const AlgorithmM &types, mlir::Type annotation,
                                llvm::StringRef protocolName) {
  if (!annotation)
    return std::nullopt;
  const py::protocols::Table &table =
      py::protocols::Table::get(types.getContext());
  std::optional<std::vector<mlir::Type>> args =
      table.protocolArgumentsFor(annotation, protocolName);
  if (!args || args->size() < 2 || !(*args)[1])
    return std::nullopt;
  return (*args)[1];
}

// Tuple member typing: uniform members keep the arity-erased homogeneous
// spelling `tuple[T]`; differing members are kept POSITIONALLY
// (`tuple[A, B]`, one contract argument per position — the same shape the
// manifest uses for dict.items()'s `tuple[$K, $V]` and starred call
// arguments). Literal-index `__getitem__` resolves positional tuples to the
// indexed member's type, so heterogeneous elements never need a union.
mlir::Type tupleOfMembers(const AlgorithmM &types,
                          llvm::ArrayRef<mlir::Type> members) {
  if (members.empty())
    return types.tupleOf(types.object());
  bool uniform = llvm::all_of(
      members, [&](mlir::Type type) { return type == members.front(); });
  if (uniform)
    return types.tupleOf(members.front());
  return types.contract("builtins.tuple", members);
}

std::optional<std::int64_t> literalIntegerFromType(mlir::Type type) {
  auto literal = mlir::dyn_cast_if_present<py::LiteralType>(type);
  if (!literal)
    return std::nullopt;
  std::int64_t value = 0;
  if (literal.getSpelling().getAsInteger(10, value))
    return std::nullopt;
  return value;
}

mlir::Type joinedLiteralElementType(const AlgorithmM &types,
                                    const parser::Node &node,
                                    llvm::StringRef fieldName) {
  llvm::SmallVector<mlir::Type, 8> elementTypes;
  if (const auto *elements = ast::nodeList(node, fieldName)) {
    elementTypes.reserve(elements->size());
    for (const parser::NodePtr &element : *elements)
      elementTypes.push_back(
          types.widenLiteral(types.inferExpr(element.get())));
  }
  return types.join(elementTypes);
}

std::optional<std::pair<mlir::Type, mlir::Type>>
joinedDictLiteralTypes(const AlgorithmM &types, const parser::Node &node) {
  const auto *keys = ast::nodeList(node, "keys");
  const auto *values = ast::nodeList(node, "values");
  if (!keys || !values)
    return std::nullopt;

  llvm::SmallVector<mlir::Type, 8> keyTypes;
  llvm::SmallVector<mlir::Type, 8> valueTypes;
  keyTypes.reserve(keys->size());
  valueTypes.reserve(values->size());
  for (auto [index, key] : llvm::enumerate(*keys)) {
    if (!key)
      return std::nullopt;
    keyTypes.push_back(types.widenLiteral(types.inferExpr(key.get())));
    if (index < values->size())
      valueTypes.push_back(
          types.widenLiteral(types.inferExpr((*values)[index].get())));
  }
  return std::make_pair(types.join(keyTypes), types.join(valueTypes));
}

bool isObjectTop(const AlgorithmM &types, mlir::Type type) {
  if (!type)
    return false;
  if (type == types.object())
    return true;
  if (auto contract = mlir::dyn_cast_if_present<py::ContractType>(type))
    return contract.getContractName() == "typing.Any";
  return false;
}

bool appendStarredCallArgumentTypes(const AlgorithmM &types, mlir::Type type,
                                    llvm::SmallVectorImpl<mlir::Type> &out) {
  type = types.widenLiteral(type);
  if (auto contract = mlir::dyn_cast_if_present<py::ContractType>(type)) {
    if (contract.getContractName() == "builtins.tuple") {
      llvm::ArrayRef<mlir::Type> arguments = contract.getArguments();
      if (arguments.size() <= 1)
        return false;
      out.append(arguments.begin(), arguments.end());
      return true;
    }
  }
  return false;
}

std::string manifestNameForContract(llvm::StringRef name) {
  for (llvm::StringRef prefix :
       {"builtins.", "typing.", "types.", "contextlib.", "_asyncio.",
        "asyncio.", "contextvars."}) {
    if (name.consume_front(prefix))
      return name.str();
  }
  return name.str();
}

bool bindManifestClassImport(AlgorithmM &types, llvm::StringRef localName,
                             llvm::StringRef contractName) {
  const py::protocols::Table &table =
      py::protocols::Table::get(types.getContext());
  if (!table.lookup(manifestNameForContract(contractName)))
    return false;
  types.bindClass(localName, types.contract(contractName));
  return true;
}

mlir::Type genericClassTemplate(const AlgorithmM &types,
                                mlir::Type instanceType) {
  auto contract = mlir::dyn_cast_if_present<py::ContractType>(instanceType);
  if (!contract || !contract.getArguments().empty())
    return instanceType;

  const py::protocols::Table &table =
      py::protocols::Table::get(types.getContext());
  const py::protocols::ProtocolInfo *info =
      table.lookup(manifestNameForContract(contract.getContractName()));
  if (!info || info->params.empty())
    return instanceType;

  llvm::SmallVector<mlir::Type, 4> arguments;
  arguments.reserve(info->params.size());
  for (const std::string &param : info->params)
    arguments.push_back(types.contract((llvm::Twine("$") + param).str()));
  return types.contract(contract.getContractName(), arguments);
}

llvm::StringRef annotationNamespaceTail(llvm::StringRef name) {
  for (llvm::StringRef prefix :
       {"typing.", "typing_extensions.", "collections.abc.", "builtins."})
    if (name.consume_front(prefix))
      return name;
  return name;
}

bool annotationNameIs(llvm::StringRef name, llvm::StringRef bareName) {
  return annotationNamespaceTail(name) == bareName;
}

std::optional<std::string> protocolAnnotationName(llvm::StringRef name) {
  name = annotationNamespaceTail(name);
  for (llvm::StringRef protocol :
       {"Awaitable", "Coroutine", "AsyncIterable", "AsyncIterator",
        "AsyncGenerator", "Sized", "Iterable", "Iterator", "Generator",
        "Collection", "Sequence", "Mapping", "MutableMapping", "ContextManager",
        "AsyncContextManager"})
    if (name == protocol)
      return protocol.str();
  return std::nullopt;
}

std::optional<std::string> contractAnnotationName(llvm::StringRef name) {
  if (name == "_asyncio.Future" || name == "asyncio.Future" ||
      annotationNamespaceTail(name) == "Future")
    return std::string("_asyncio.Future");
  if (name == "_asyncio.Task" || name == "asyncio.Task" ||
      annotationNamespaceTail(name) == "Task")
    return std::string("_asyncio.Task");
  if (name == "asyncio.AbstractEventLoop" ||
      name == "asyncio.events.AbstractEventLoop" ||
      annotationNamespaceTail(name) == "AbstractEventLoop")
    return std::string("asyncio.AbstractEventLoop");
  if (name == "asyncio.CancelledError" ||
      name == "asyncio.exceptions.CancelledError" ||
      annotationNamespaceTail(name) == "CancelledError")
    return std::string("asyncio.CancelledError");
  if (name == "contextvars.Context" ||
      annotationNamespaceTail(name) == "Context")
    return std::string("contextvars.Context");
  return std::nullopt;
}

bool isImportedAnnotationName(llvm::StringRef name) {
  if (name == "Any" || name == "Self" || name == "Optional" ||
      name == "Union" || name == "Literal" || name == "Type" ||
      name == "type" || name == "Callable" || name == "List" ||
      name == "Dict" || name == "Tuple" || name == "Set" ||
      name == "FrozenSet" || name == "ParamSpec" || name == "TypeVar" ||
      name == "TypeVarTuple" || name == "Unpack")
    return true;
  return protocolAnnotationName(name) || contractAnnotationName(name);
}

std::optional<std::string> bareGenericAnnotationName(llvm::StringRef name) {
  llvm::StringRef tail = annotationNamespaceTail(name);
  for (llvm::StringRef generic :
       {"Callable",       "Collection",   "Sequence",      "Mapping",
        "MutableMapping", "Iterable",     "Iterator",      "Generator",
        "AsyncIterable",  "AsyncIterator", "AsyncGenerator", "list",
        "List",           "dict",         "Dict",          "tuple",
        "Tuple",          "set",          "Set",           "frozenset",
        "FrozenSet",      "Optional",     "Union"})
    if (tail == generic)
      return tail.str();
  return std::nullopt;
}

void bindAnnotationModuleAliases(
    llvm::function_ref<void(llvm::StringRef)> bind) {
  for (llvm::StringRef name : {"Any",
                               "Self",
                               "Optional",
                               "Union",
                               "Literal",
                               "Type",
                               "type",
                               "Callable",
                               "List",
                               "Dict",
                               "Tuple",
                               "Set",
                               "FrozenSet",
                               "ParamSpec",
                               "TypeVar",
                               "TypeVarTuple",
                               "Unpack",
                               "Awaitable",
                               "Coroutine",
                               "AsyncIterable",
                               "AsyncIterator",
                               "AsyncGenerator",
                               "Sized",
                               "Iterable",
                               "Iterator",
                               "Generator",
                               "Collection",
                               "Sequence",
                               "Mapping",
                               "MutableMapping",
                               "ContextManager",
                               "AsyncContextManager"})
    bind(name);
}

enum class ImportCallableFactory {
  BuiltinsFunction,
  StaticZeroArgStr,
};

mlir::Type importCallableType(const AlgorithmM &types,
                              ImportCallableFactory factory) {
  switch (factory) {
  case ImportCallableFactory::BuiltinsFunction:
    return types.contract("builtins.function");
  case ImportCallableFactory::StaticZeroArgStr:
    return makeZeroArgStrCallable(types);
  }
  llvm_unreachable("unknown import callable factory");
}

struct ModuleCallableImport {
  const char *module;
  const char *localAttr;
  const char *canonicalName;
  ImportCallableFactory factory;
};

struct ModuleAliasImport {
  const char *module;
  const char *localAttr;
  const char *canonicalName;
  bool annotationAlias = false;
};

struct NameCallableImport {
  const char *module;
  const char *exportedName;
  const char *canonicalName;
  ImportCallableFactory factory;
};

struct NameAliasImport {
  const char *module;
  const char *exportedName;
  const char *canonicalName;
  bool annotationAlias = false;
};

struct ModuleStringConstantImport {
  const char *module;
  const char *localAttr;
  const char *canonicalName;
};

struct NameStringConstantImport {
  const char *module;
  const char *exportedName;
  const char *canonicalName;
};

constexpr ModuleCallableImport kModuleCallableImports[] = {
    // Manifest-declared callables (ly.typing.callable_exports +
    // ly.typing.function_contracts, e.g. asyncio.*, os.getpid, ctypes.*)
    // bind through bindManifestModuleCallableExports -- ONLY names without a
    // manifest contract belong here (C++ factory-typed).
    {"platform", "system", "platform.system",
     ImportCallableFactory::StaticZeroArgStr},
    {"lyrt", "from_prim", "lyrt.from_prim",
     ImportCallableFactory::BuiltinsFunction},
    {"lyrt", "native", "lyrt.native", ImportCallableFactory::BuiltinsFunction},
};

constexpr ModuleAliasImport kModuleAliasImports[] = {
    {"lyrt", "prim.Int", "lyrt.prim.Int", true},
    {"lyrt", "prim.Float", "lyrt.prim.Float", true},
    {"lyrt", "prim.Vector", "lyrt.prim.Vector", true},
    {"lyrt", "prim.Matrix", "lyrt.prim.Matrix", true},
    {"lyrt", "prim.Tensor", "lyrt.prim.Tensor", true},
    {"lyrt.prim", "Int", "lyrt.prim.Int", true},
    {"lyrt.prim", "Float", "lyrt.prim.Float", true},
    {"lyrt.prim", "Vector", "lyrt.prim.Vector", true},
    {"lyrt.prim", "Matrix", "lyrt.prim.Matrix", true},
    {"lyrt.prim", "Tensor", "lyrt.prim.Tensor", true},
};

constexpr NameCallableImport kNameCallableImports[] = {
    // Manifest-declared callables bind through the ly.typing.callable_exports
    // channel in bindImportedName -- ONLY factory-typed names belong here.
    {"platform", "system", "platform.system",
     ImportCallableFactory::StaticZeroArgStr},
    {"lyrt", "from_prim", "lyrt.from_prim",
     ImportCallableFactory::BuiltinsFunction},
    {"lyrt", "native", "lyrt.native", ImportCallableFactory::BuiltinsFunction},
};

constexpr NameAliasImport kNameAliasImports[] = {
    {"lyrt.prim", "Int", "lyrt.prim.Int", true},
    {"lyrt.prim", "Float", "lyrt.prim.Float", true},
    {"lyrt.prim", "Vector", "lyrt.prim.Vector", true},
    {"lyrt.prim", "Matrix", "lyrt.prim.Matrix", true},
    {"lyrt.prim", "Tensor", "lyrt.prim.Tensor", true},
};

constexpr ModuleStringConstantImport kModuleStringConstantImports[] = {
    {"sys", "platform", "sys.platform"},
    {"os", "name", "os.name"},
};

constexpr NameStringConstantImport kNameStringConstantImports[] = {
    {"sys", "platform", "sys.platform"},
    {"os", "name", "os.name"},
};

std::string importedAttribute(llvm::StringRef localName, llvm::StringRef attr) {
  return (llvm::Twine(localName) + "." + attr).str();
}

enum class AnnotationModuleStyle {
  Direct,
  CollectionsAbc,
  Collections,
};

std::optional<AnnotationModuleStyle>
annotationModuleStyle(llvm::StringRef module) {
  if (module == "typing" || module == "typing_extensions")
    return AnnotationModuleStyle::Direct;
  if (module == "collections.abc")
    return AnnotationModuleStyle::CollectionsAbc;
  if (module == "collections")
    return AnnotationModuleStyle::Collections;
  return std::nullopt;
}

std::string importedAnnotationAlias(llvm::StringRef module,
                                    llvm::StringRef localName,
                                    AnnotationModuleStyle style,
                                    llvm::StringRef name) {
  switch (style) {
  case AnnotationModuleStyle::Direct:
    return importedAttribute(localName, name);
  case AnnotationModuleStyle::CollectionsAbc: {
    std::string prefix = localName == module.split('.').first
                             ? importedAttribute(localName, "abc")
                             : std::string(localName);
    return (llvm::Twine(prefix) + "." + name).str();
  }
  case AnnotationModuleStyle::Collections:
    return importedAttribute(localName, (llvm::Twine("abc.") + name).str());
  }
  return name.str();
}

bool moduleExportsAnnotationNames(llvm::StringRef module) {
  return module == "typing" || module == "typing_extensions" ||
         module == "collections.abc";
}

std::string importedManifestModuleAttribute(llvm::StringRef module,
                                            llvm::StringRef localName,
                                            llvm::StringRef attr) {
  std::pair<llvm::StringRef, llvm::StringRef> root = module.split('.');
  if (root.second.empty() || localName != root.first)
    return importedAttribute(localName, attr);
  return importedAttribute(
      localName, (llvm::Twine(root.second) + "." + attr).str());
}

bool bindManifestModuleObject(AlgorithmM &types, llvm::StringRef module,
                              llvm::StringRef localName) {
  // Module and submodule namespace symbols are lookup roots, not runtime
  // receivers. Their `object` top is an AGENTS.md namespace placeholder:
  // member access resolves through separately bound `name.attr` canonical
  // symbols with real contracts, and a bare module value carries no protocol
  // contract, so dispatching on it fails for lack of evidence.
  std::pair<llvm::StringRef, llvm::StringRef> root = module.split('.');
  if (!root.second.empty() && localName == root.first) {
    types.bindCanonicalSymbol(localName, root.first, types.object());
    types.bindCanonicalSymbol(importedAttribute(localName, root.second), module,
                              types.object());
    return true;
  }
  types.bindCanonicalSymbol(localName, module, types.object());
  return true;
}

bool bindManifestModuleClassExports(AlgorithmM &types, llvm::StringRef module,
                                    llvm::StringRef localName) {
  const py::protocols::Table &table =
      py::protocols::Table::get(types.getContext());
  bool handled = false;
  for (const auto &[exportedName, contract] : table.moduleClassExports(module)) {
    if (!handled)
      bindManifestModuleObject(types, module, localName);
    handled = true;
    if (!bindManifestClassImport(
            types,
            importedManifestModuleAttribute(module, localName, exportedName),
            contract))
      return false;
  }
  return handled;
}

bool bindManifestModuleFloatConstants(AlgorithmM &types,
                                      llvm::StringRef module,
                                      llvm::StringRef localName) {
  const py::protocols::Table &table =
      py::protocols::Table::get(types.getContext());
  bool handled = false;
  for (const std::string &exportedName :
       table.moduleFloatConstantExports(module)) {
    if (!handled)
      bindManifestModuleObject(types, module, localName);
    handled = true;
    std::string canonical = (llvm::Twine(module) + "." + exportedName).str();
    types.bindCanonicalSymbol(
        importedManifestModuleAttribute(module, localName, exportedName),
        canonical, types.floatType());
  }
  return handled;
}

bool bindManifestModuleCallableExports(AlgorithmM &types,
                                       llvm::StringRef module,
                                       llvm::StringRef localName) {
  const py::protocols::Table &table =
      py::protocols::Table::get(types.getContext());
  bool handled = false;
  for (const std::string &exportedName : table.moduleCallableExports(module)) {
    std::string canonical = (llvm::Twine(module) + "." + exportedName).str();
    // Prefer the manifest-declared Callable contract; a declared callable
    // export WITHOUT a contract (e.g. ctypes byref/pointer/POINTER/cast)
    // binds as a generic function object -- its calls resolve from lowering
    // evidence instead of a static signature.
    mlir::Type contract = table.freeFunctionContract(canonical)
                              .value_or(types.contract("builtins.function"));
    if (!handled)
      bindManifestModuleObject(types, module, localName);
    handled = true;
    types.bindCanonicalSymbol(
        importedManifestModuleAttribute(module, localName, exportedName),
        canonical, contract);
  }
  return handled;
}

std::optional<std::string> staticParameterName(mlir::Type type) {
  if (mlir::isa<py::SelfType>(type))
    return std::string("Self");
  if (auto typeVar = mlir::dyn_cast_if_present<py::TypeVarType>(type))
    return typeVar.getName().str();
  if (auto paramSpec = mlir::dyn_cast_if_present<py::ParamSpecType>(type))
    return paramSpec.getName().str();
  if (auto typeVarTuple = mlir::dyn_cast_if_present<py::TypeVarTupleType>(type))
    return typeVarTuple.getName().str();
  if (auto contract = mlir::dyn_cast_if_present<py::ContractType>(type)) {
    llvm::StringRef name = contract.getContractName();
    if (name.starts_with("$"))
      return name.drop_front().str();
  }
  return std::nullopt;
}

bool structuralProtocolAccepts(const AlgorithmM &types,
                               py::ContractType expected, mlir::Type actual) {
  const py::protocols::Table &table =
      py::protocols::Table::get(types.getContext());
  std::string protocolName =
      manifestNameForContract(expected.getContractName());
  const py::protocols::ProtocolInfo *info = table.lookup(protocolName);
  if (!info || !info->isProtocol)
    return false;

  if (std::optional<std::vector<mlir::Type>> args =
          table.protocolArgumentsFor(actual, protocolName)) {
    llvm::ArrayRef<mlir::Type> expectedArgs = expected.getArguments();
    if (expectedArgs.empty() || expectedArgs.size() == args->size())
      return true;
  }

  return llvm::all_of(info->methods, [&](const auto &method) {
    return !table.methodContractCandidatesWithEvidence(actual, method.first)
                .empty();
  });
}

bool typeAccepts(const AlgorithmM &types, mlir::Type expected,
                 mlir::Type actual) {
  expected = types.widenLiteral(expected);
  actual = types.widenLiteral(actual);
  if (!expected || !actual)
    return true;
  if (expected == actual)
    return true;
  if (isObjectTop(types, expected) || isObjectTop(types, actual))
    return true;
  if (auto expectedContract =
          mlir::dyn_cast_if_present<py::ContractType>(expected)) {
    const py::protocols::Table &table =
        py::protocols::Table::get(types.getContext());
    if (table.isManifestSubclassOf(actual, expectedContract.getContractName()))
      return true;
    if (structuralProtocolAccepts(types, expectedContract, actual))
      return true;
  }
  return py::isAssignableTo(actual, expected);
}

using TypeBindingMap = std::map<std::string, mlir::Type>;

struct CallSolution {
  mlir::Type result;
  TypeBindingMap bindings;
  py::CallableType callableContract;
  std::string methodName;
  std::optional<std::string> receiverManifestClass;
  int score = 0;
};

mlir::Type substituteType(const AlgorithmM &types, mlir::Type type,
                          const TypeBindingMap &bindings,
                          bool eraseUnbound = false);

bool bindExpectedType(const AlgorithmM &types, mlir::Type expected,
                      mlir::Type actual, TypeBindingMap &bindings);

std::optional<py::CallableType>
expandParamSpecForInvocation(const AlgorithmM &types, py::CallableType callable,
                             mlir::ArrayRef<mlir::Type> positional,
                             mlir::ArrayRef<CallKeywordType> keywords,
                             TypeBindingMap &bindings,
                             std::size_t firstParameter = 0);

std::optional<std::string> paramSpecName(mlir::Type type) {
  if (auto paramSpec = mlir::dyn_cast_if_present<py::ParamSpecType>(type))
    return paramSpec.getName().str();
  return std::nullopt;
}

std::optional<std::string> typeVarTupleName(mlir::Type type) {
  if (auto typeVarTuple = mlir::dyn_cast_if_present<py::TypeVarTupleType>(type))
    return typeVarTuple.getName().str();
  return std::nullopt;
}

std::optional<std::string> unpackedTypeVarTupleName(mlir::Type type) {
  auto unpack = mlir::dyn_cast_if_present<py::UnpackType>(type);
  if (!unpack)
    return std::nullopt;
  return typeVarTupleName(unpack.getPackedType());
}

llvm::ArrayRef<mlir::Type> typeVarTupleElements(mlir::Type type) {
  if (auto pack = mlir::dyn_cast_if_present<py::CallableType>(type))
    return pack.getPositionalTypes();
  if (auto tuple = mlir::dyn_cast_if_present<py::ContractType>(type))
    if (tuple.getContractName() == "builtins.tuple")
      return tuple.getArguments();
  return {};
}

bool sameKeywords(mlir::ArrayRef<CallKeywordType> lhs,
                  mlir::ArrayRef<CallKeywordType> rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (auto [left, right] : llvm::zip(lhs, rhs))
    if (left.name != right.name || left.type != right.type)
      return false;
  return true;
}

void collectExplicitKeywordFormalNames(
    py::CallableType callable, std::size_t firstParameter,
    std::optional<std::size_t> paramSpecIndex, llvm::StringSet<> &names) {
  llvm::ArrayRef<mlir::StringAttr> positionalNames =
      callable.getPositionalNames();
  for (auto [index, name] : llvm::enumerate(positionalNames)) {
    if (!name || index < firstParameter ||
        (paramSpecIndex && index == *paramSpecIndex) ||
        index >= callable.getPositionalTypes().size() ||
        index < callable.getPositionalOnlyCount())
      continue;
    names.insert(name.getValue());
  }

  for (mlir::StringAttr name : callable.getKwOnlyNames())
    if (name)
      names.insert(name.getValue());
}

llvm::SmallVector<CallKeywordType, 4>
captureParamSpecKeywords(py::CallableType callable, std::size_t firstParameter,
                         std::optional<std::size_t> paramSpecIndex,
                         mlir::ArrayRef<CallKeywordType> keywords) {
  llvm::StringSet<> explicitNames;
  collectExplicitKeywordFormalNames(callable, firstParameter, paramSpecIndex,
                                    explicitNames);

  llvm::SmallVector<CallKeywordType, 4> captured;
  captured.reserve(keywords.size());
  for (const CallKeywordType &keyword : keywords) {
    if (!keyword.name.empty() && explicitNames.contains(keyword.name))
      continue;
    captured.push_back(keyword);
  }
  return captured;
}

py::CallableType makeParamSpecPack(mlir::MLIRContext *context,
                                   mlir::ArrayRef<mlir::Type> positional,
                                   mlir::ArrayRef<CallKeywordType> keywords) {
  llvm::SmallVector<mlir::Type, 4> kwonly;
  llvm::SmallVector<mlir::StringAttr, 4> kwonlyNames;
  kwonly.reserve(keywords.size());
  kwonlyNames.reserve(keywords.size());
  for (const CallKeywordType &keyword : keywords) {
    kwonly.push_back(keyword.type);
    kwonlyNames.push_back(mlir::StringAttr::get(context, keyword.name));
  }
  return py::CallableType::get(context, positional, kwonly, {}, {}, {}, {},
                               kwonlyNames, {}, {});
}

std::optional<llvm::SmallVector<CallKeywordType, 4>>
paramSpecPackKeywords(py::CallableType pack) {
  llvm::SmallVector<CallKeywordType, 4> keywords;
  llvm::ArrayRef<mlir::StringAttr> names = pack.getKwOnlyNames();
  for (auto [index, kwType] : llvm::enumerate(pack.getKwOnlyTypes())) {
    if (index >= names.size())
      return std::nullopt;
    keywords.push_back(CallKeywordType{names[index].getValue().str(), kwType});
  }
  return keywords;
}

bool bindParamSpecPack(const AlgorithmM &types, llvm::StringRef name,
                       mlir::ArrayRef<mlir::Type> positional,
                       mlir::ArrayRef<CallKeywordType> keywords,
                       TypeBindingMap &bindings, bool merge = false) {
  py::CallableType pack =
      makeParamSpecPack(&types.getContext(), positional, keywords);
  auto found = bindings.find(name.str());
  if (found == bindings.end()) {
    bindings[name.str()] = pack;
    return true;
  }
  auto existing = mlir::dyn_cast_if_present<py::CallableType>(found->second);
  if (!existing)
    return false;
  std::optional<llvm::SmallVector<CallKeywordType, 4>> existingKeywords =
      paramSpecPackKeywords(existing);
  if (!existingKeywords)
    return false;
  if (!merge) {
    return existing.getPositionalTypes() == positional &&
           sameKeywords(*existingKeywords, keywords);
  }

  llvm::ArrayRef<mlir::Type> existingPositionals =
      existing.getPositionalTypes();
  llvm::SmallVector<mlir::Type, 4> mergedPositionals;
  if (!existingPositionals.empty() && !positional.empty() &&
      existingPositionals != positional)
    return false;
  llvm::ArrayRef<mlir::Type> selectedPositionals =
      !positional.empty() ? positional : existingPositionals;
  mergedPositionals.append(selectedPositionals.begin(),
                           selectedPositionals.end());

  llvm::SmallVector<CallKeywordType, 4> mergedKeywords;
  if (!existingKeywords->empty() && !keywords.empty() &&
      !sameKeywords(*existingKeywords, keywords))
    return false;
  llvm::ArrayRef<CallKeywordType> selectedKeywords =
      !keywords.empty() ? keywords : *existingKeywords;
  mergedKeywords.append(selectedKeywords.begin(), selectedKeywords.end());

  found->second =
      makeParamSpecPack(&types.getContext(), mergedPositionals, mergedKeywords);
  return true;
}

bool bindTypeVarTuplePack(const AlgorithmM &types, llvm::StringRef name,
                          mlir::ArrayRef<mlir::Type> positional,
                          TypeBindingMap &bindings) {
  if (positional.size() == 1)
    if (std::optional<std::string> forwarded =
            unpackedTypeVarTupleName(positional.front()))
      if (*forwarded == name)
        return true;
  mlir::Type pack =
      py::CallableType::get(&types.getContext(), positional, {}, {}, {}, {});
  auto found = bindings.find(name.str());
  if (found == bindings.end()) {
    bindings[name.str()] = pack;
    return true;
  }
  return found->second == pack;
}

bool bindTypeParameter(const AlgorithmM &types, llvm::StringRef name,
                       mlir::Type actual, TypeBindingMap &bindings) {
  actual = types.widenLiteral(actual);
  if (!actual)
    return false;
  auto found = bindings.find(name.str());
  if (found == bindings.end()) {
    bindings[name.str()] = actual;
    return true;
  }
  if (std::optional<std::string> existing = staticParameterName(found->second))
    if (*existing == name) {
      found->second = actual;
      return true;
    }
  if (typeAccepts(types, found->second, actual))
    return true;
  if (typeAccepts(types, actual, found->second)) {
    found->second = actual;
    return true;
  }
  return false;
}

bool bindTypeList(const AlgorithmM &types, mlir::ArrayRef<mlir::Type> expected,
                  mlir::ArrayRef<mlir::Type> actual, TypeBindingMap &bindings) {
  if (expected.size() != actual.size())
    return false;
  for (auto [expectedType, actualType] : llvm::zip(expected, actual))
    if (!bindExpectedType(types, expectedType, actualType, bindings))
      return false;
  return true;
}

bool bindUnionMember(const AlgorithmM &types, py::UnionType expected,
                     mlir::Type actual, TypeBindingMap &bindings) {
  for (mlir::Type member : expected.getMemberTypes()) {
    TypeBindingMap candidate = bindings;
    if (bindExpectedType(types, member, actual, candidate)) {
      bindings = std::move(candidate);
      return true;
    }
  }
  return false;
}

bool bindProtocolView(const AlgorithmM &types, py::ProtocolType expected,
                      mlir::Type actual, TypeBindingMap &bindings) {
  const py::protocols::Table &table =
      py::protocols::Table::get(types.getContext());
  llvm::ArrayRef<mlir::Type> expectedArgs = expected.getArguments();
  if (auto actualProtocol = mlir::dyn_cast_if_present<py::ProtocolType>(actual))
    if (actualProtocol.getProtocolName() == expected.getProtocolName())
      return expectedArgs.empty() ||
             bindTypeList(types, expectedArgs, actualProtocol.getArguments(),
                          bindings);

  std::optional<std::vector<mlir::Type>> actualArgs =
      table.protocolArgumentsFor(actual, expected.getProtocolName());
  if (!actualArgs)
    return typeAccepts(types, expected, actual);
  if (expectedArgs.empty())
    return true;
  return bindTypeList(types, expectedArgs, *actualArgs, bindings);
}

bool bindContractView(const AlgorithmM &types, py::ContractType expected,
                      mlir::Type actual, TypeBindingMap &bindings) {
  if (auto actualContract = mlir::dyn_cast_if_present<py::ContractType>(actual))
    if (actualContract.getContractName() == expected.getContractName())
      return expected.getArguments().empty() ||
             bindTypeList(types, expected.getArguments(),
                          actualContract.getArguments(), bindings);

  if (structuralProtocolAccepts(types, expected, actual))
    return true;
  return typeAccepts(types, expected, actual);
}

bool bindCallableView(const AlgorithmM &types, py::CallableType expected,
                      py::CallableType actual, TypeBindingMap &bindings) {
  llvm::SmallVector<CallKeywordType, 4> actualKeywords;
  llvm::ArrayRef<mlir::StringAttr> actualKwNames = actual.getKwOnlyNames();
  for (auto [index, type] : llvm::enumerate(actual.getKwOnlyTypes())) {
    std::string name;
    if (index < actualKwNames.size() && actualKwNames[index])
      name = actualKwNames[index].getValue().str();
    actualKeywords.push_back(CallKeywordType{std::move(name), type});
  }

  TypeBindingMap candidateBindings = bindings;
  std::optional<py::CallableType> expandedExpected =
      expandParamSpecForInvocation(types, expected, actual.getPositionalTypes(),
                                   actualKeywords, candidateBindings);
  if (!expandedExpected)
    return typeAccepts(types, expected, actual);
  expected = *expandedExpected;

  if (expected.getPositionalTypes().size() !=
          actual.getPositionalTypes().size() ||
      expected.getKwOnlyTypes().size() != actual.getKwOnlyTypes().size() ||
      expected.getResultTypes().size() != actual.getResultTypes().size())
    return typeAccepts(types, expected, actual);

  for (auto [expectedArg, actualArg] :
       llvm::zip(expected.getPositionalTypes(), actual.getPositionalTypes()))
    if (!bindExpectedType(types, expectedArg, actualArg, candidateBindings))
      return false;
  for (auto [expectedArg, actualArg] :
       llvm::zip(expected.getKwOnlyTypes(), actual.getKwOnlyTypes()))
    if (!bindExpectedType(types, expectedArg, actualArg, candidateBindings))
      return false;
  for (auto [expectedResult, actualResult] :
       llvm::zip(expected.getResultTypes(), actual.getResultTypes()))
    if (!bindExpectedType(types, expectedResult, actualResult,
                          candidateBindings))
      return false;
  bindings = std::move(candidateBindings);
  return true;
}

bool bindExpectedType(const AlgorithmM &types, mlir::Type expected,
                      mlir::Type actual, TypeBindingMap &bindings) {
  expected = substituteType(types, expected, bindings);
  actual = types.widenLiteral(actual);

  if (!expected || !actual)
    return true;
  if (std::optional<std::string> name = staticParameterName(expected))
    return bindTypeParameter(types, *name, actual, bindings);
  if (expected == actual)
    return true;
  if (isObjectTop(types, expected) || isObjectTop(types, actual))
    return true;

  if (auto unionType = mlir::dyn_cast_if_present<py::UnionType>(expected))
    return bindUnionMember(types, unionType, actual, bindings);
  if (auto expectedType = mlir::dyn_cast_if_present<py::TypeType>(expected)) {
    if (auto actualType = mlir::dyn_cast_if_present<py::TypeType>(actual))
      return bindExpectedType(types, expectedType.getInstanceType(),
                              actualType.getInstanceType(), bindings);
  }
  if (auto expectedProtocol =
          mlir::dyn_cast_if_present<py::ProtocolType>(expected))
    return bindProtocolView(types, expectedProtocol, actual, bindings);
  if (auto expectedContract =
          mlir::dyn_cast_if_present<py::ContractType>(expected))
    return bindContractView(types, expectedContract, actual, bindings);
  if (auto expectedCallable =
          mlir::dyn_cast_if_present<py::CallableType>(expected))
    if (auto actualCallable =
            mlir::dyn_cast_if_present<py::CallableType>(actual))
      return bindCallableView(types, expectedCallable, actualCallable,
                              bindings);

  return typeAccepts(types, expected, actual);
}

py::CallableType substituteCallable(const AlgorithmM &types,
                                    py::CallableType callable,
                                    const TypeBindingMap &bindings,
                                    bool eraseUnbound) {
  if (!callable)
    return {};

  auto boundParamSpecPack = [&](llvm::StringRef name) -> py::CallableType {
    auto found = bindings.find(name.str());
    if (found == bindings.end())
      return {};
    return mlir::dyn_cast_if_present<py::CallableType>(found->second);
  };
  auto boundTypeVarTuplePack =
      [&](llvm::StringRef name) -> std::optional<llvm::ArrayRef<mlir::Type>> {
    auto found = bindings.find(name.str());
    if (found == bindings.end())
      return std::nullopt;
    return typeVarTupleElements(found->second);
  };

  llvm::SmallVector<mlir::Type, 8> positional;
  llvm::SmallVector<mlir::StringAttr, 8> positionalNames;
  llvm::SmallVector<mlir::BoolAttr, 8> positionalDefaults;
  bool hasPositionalNames = !callable.getPositionalNames().empty();
  bool hasPositionalDefaults = !callable.getPositionalDefaults().empty();
  for (auto [index, arg] : llvm::enumerate(callable.getPositionalTypes())) {
    if (std::optional<std::string> name = paramSpecName(arg)) {
      py::CallableType pack = boundParamSpecPack(*name);
      if (pack) {
        for (mlir::Type packArg : pack.getPositionalTypes()) {
          positional.push_back(
              substituteType(types, packArg, bindings, eraseUnbound));
          if (hasPositionalNames)
            positionalNames.push_back(mlir::StringAttr());
          if (hasPositionalDefaults)
            positionalDefaults.push_back(
                mlir::BoolAttr::get(callable.getContext(), false));
        }
        continue;
      }
    }
    if (std::optional<std::string> name = unpackedTypeVarTupleName(arg)) {
      std::optional<llvm::ArrayRef<mlir::Type>> pack =
          boundTypeVarTuplePack(*name);
      if (pack) {
        for (mlir::Type packArg : *pack) {
          positional.push_back(
              substituteType(types, packArg, bindings, eraseUnbound));
          if (hasPositionalNames)
            positionalNames.push_back(mlir::StringAttr());
          if (hasPositionalDefaults)
            positionalDefaults.push_back(
                mlir::BoolAttr::get(callable.getContext(), false));
        }
        continue;
      }
    }
    positional.push_back(substituteType(types, arg, bindings, eraseUnbound));
    if (hasPositionalNames) {
      llvm::ArrayRef<mlir::StringAttr> names = callable.getPositionalNames();
      positionalNames.push_back(index < names.size() ? names[index]
                                                     : mlir::StringAttr());
    }
    if (hasPositionalDefaults) {
      llvm::ArrayRef<mlir::BoolAttr> defaults =
          callable.getPositionalDefaults();
      positionalDefaults.push_back(
          index < defaults.size()
              ? defaults[index]
              : mlir::BoolAttr::get(callable.getContext(), false));
    }
  }

  mlir::Type vararg;
  if (callable.hasVararg()) {
    mlir::Type varargType = callable.getVarargType();
    std::optional<std::string> name = paramSpecName(varargType);
    py::CallableType pack =
        name ? boundParamSpecPack(*name) : py::CallableType();
    if (pack) {
      for (mlir::Type packArg : pack.getPositionalTypes()) {
        positional.push_back(
            substituteType(types, packArg, bindings, eraseUnbound));
        if (hasPositionalNames)
          positionalNames.push_back(mlir::StringAttr());
        if (hasPositionalDefaults)
          positionalDefaults.push_back(
              mlir::BoolAttr::get(callable.getContext(), false));
      }
    } else if (std::optional<std::string> tupleName =
                   unpackedTypeVarTupleName(varargType)) {
      std::optional<llvm::ArrayRef<mlir::Type>> tuplePack =
          boundTypeVarTuplePack(*tupleName);
      if (tuplePack) {
        for (mlir::Type packArg : *tuplePack) {
          positional.push_back(
              substituteType(types, packArg, bindings, eraseUnbound));
          if (hasPositionalNames)
            positionalNames.push_back(mlir::StringAttr());
          if (hasPositionalDefaults)
            positionalDefaults.push_back(
                mlir::BoolAttr::get(callable.getContext(), false));
        }
      } else {
        vararg = substituteType(types, varargType, bindings, eraseUnbound);
      }
    } else {
      vararg = substituteType(types, varargType, bindings, eraseUnbound);
    }
  }

  llvm::SmallVector<mlir::Type, 4> kwonly;
  llvm::SmallVector<mlir::StringAttr, 4> kwonlyNames;
  llvm::SmallVector<mlir::BoolAttr, 4> kwonlyDefaults;
  for (mlir::Type arg : callable.getKwOnlyTypes())
    kwonly.push_back(substituteType(types, arg, bindings, eraseUnbound));
  kwonlyNames.append(callable.getKwOnlyNames().begin(),
                     callable.getKwOnlyNames().end());
  kwonlyDefaults.append(callable.getKwOnlyDefaults().begin(),
                        callable.getKwOnlyDefaults().end());

  llvm::StringSet<> expandedKeywordParamSpecs;
  auto appendParamSpecKeywords = [&](llvm::StringRef name) {
    if (!expandedKeywordParamSpecs.insert(name).second)
      return;
    py::CallableType pack = boundParamSpecPack(name);
    if (!pack)
      return;
    if (!pack.getKwOnlyTypes().empty() && kwonlyNames.empty() &&
        !kwonly.empty()) {
      for (std::size_t index = 0, end = kwonly.size(); index < end; ++index)
        kwonlyNames.push_back(mlir::StringAttr());
    }
    if (!pack.getKwOnlyTypes().empty() && kwonlyDefaults.empty() &&
        !kwonly.empty()) {
      for (std::size_t index = 0, end = kwonly.size(); index < end; ++index)
        kwonlyDefaults.push_back(
            mlir::BoolAttr::get(callable.getContext(), false));
    }
    for (auto [index, kwType] : llvm::enumerate(pack.getKwOnlyTypes())) {
      kwonly.push_back(substituteType(types, kwType, bindings, eraseUnbound));
      llvm::ArrayRef<mlir::StringAttr> names = pack.getKwOnlyNames();
      kwonlyNames.push_back(index < names.size() ? names[index]
                                                 : mlir::StringAttr());
      kwonlyDefaults.push_back(
          mlir::BoolAttr::get(callable.getContext(), false));
    }
  };

  for (mlir::Type arg : callable.getPositionalTypes())
    if (std::optional<std::string> name = paramSpecName(arg))
      appendParamSpecKeywords(*name);
  if (callable.hasVararg())
    if (std::optional<std::string> name =
            paramSpecName(callable.getVarargType()))
      appendParamSpecKeywords(*name);
  if (callable.hasKwarg())
    if (std::optional<std::string> name =
            paramSpecName(callable.getKwargType()))
      appendParamSpecKeywords(*name);

  llvm::SmallVector<mlir::Type, 1> results;
  for (mlir::Type result : callable.getResultTypes())
    results.push_back(substituteType(types, result, bindings, eraseUnbound));

  mlir::Type kwarg;
  if (callable.hasKwarg()) {
    mlir::Type kwargType = callable.getKwargType();
    std::optional<std::string> name = paramSpecName(kwargType);
    if (!name || !boundParamSpecPack(*name))
      kwarg = substituteType(types, kwargType, bindings, eraseUnbound);
  }

  return py::CallableType::get(
      callable.getContext(), positional, kwonly, vararg, kwarg, results,
      positionalNames, kwonlyNames, positionalDefaults, kwonlyDefaults,
      vararg ? callable.getVarargName() : mlir::StringAttr(),
      kwarg ? callable.getKwargName() : mlir::StringAttr(),
      callable.getPositionalOnlyCount());
}

mlir::Type substituteType(const AlgorithmM &types, mlir::Type type,
                          const TypeBindingMap &bindings, bool eraseUnbound) {
  if (!type)
    return type;
  if (std::optional<std::string> name = staticParameterName(type)) {
    auto found = bindings.find(*name);
    if (found != bindings.end())
      return found->second;
    // An unbound type parameter with eraseUnbound set becomes an `object` top
    // (e.g. an unspecialized generic call result or storage crossing). This is
    // not an invalid-operation fallback: the erased top carries no protocol
    // contract, so any later observation of the value requires fresh evidence
    // it cannot obtain and is rejected at the static boundary rather than
    // silently dispatched.
    return eraseUnbound ? types.object() : type;
  }
  if (auto contract = mlir::dyn_cast_if_present<py::ContractType>(type)) {
    llvm::SmallVector<mlir::Type, 4> args;
    for (mlir::Type arg : contract.getArguments())
      args.push_back(substituteType(types, arg, bindings, eraseUnbound));
    return py::ContractType::get(type.getContext(), contract.getContractName(),
                                 args);
  }
  if (auto protocol = mlir::dyn_cast_if_present<py::ProtocolType>(type)) {
    llvm::SmallVector<mlir::Type, 4> args;
    for (mlir::Type arg : protocol.getArguments())
      args.push_back(substituteType(types, arg, bindings, eraseUnbound));
    return py::ProtocolType::get(type.getContext(), protocol.getProtocolName(),
                                 args);
  }
  if (auto typeType = mlir::dyn_cast_if_present<py::TypeType>(type)) {
    mlir::Type instance = substituteType(types, typeType.getInstanceType(),
                                         bindings, eraseUnbound);
    return instance ? py::TypeType::get(type.getContext(), instance)
                    : mlir::Type();
  }
  if (auto unpack = mlir::dyn_cast_if_present<py::UnpackType>(type)) {
    mlir::Type packed =
        substituteType(types, unpack.getPackedType(), bindings, eraseUnbound);
    return packed ? py::UnpackType::get(type.getContext(), packed)
                  : mlir::Type();
  }
  if (auto unionType = mlir::dyn_cast_if_present<py::UnionType>(type)) {
    llvm::SmallVector<mlir::Type, 4> members;
    for (mlir::Type member : unionType.getMemberTypes())
      members.push_back(substituteType(types, member, bindings, eraseUnbound));
    return py::UnionType::getNormalized(type.getContext(), members);
  }
  if (auto callable = mlir::dyn_cast_if_present<py::CallableType>(type))
    return substituteCallable(types, callable, bindings, eraseUnbound);
  if (auto overload = mlir::dyn_cast_if_present<py::OverloadType>(type)) {
    llvm::SmallVector<mlir::Type, 4> candidates;
    for (mlir::Type candidate : overload.getCandidateTypes())
      candidates.push_back(
          substituteType(types, candidate, bindings, eraseUnbound));
    return py::OverloadType::get(type.getContext(), candidates);
  }
  return type;
}

mlir::Type callableResultType(const AlgorithmM &types,
                              py::CallableType callable,
                              const TypeBindingMap &bindings) {
  llvm::ArrayRef<mlir::Type> results = callable.getResultTypes();
  if (results.size() == 1)
    return substituteType(types, results.front(), bindings,
                          /*eraseUnbound=*/true);
  if (!results.empty()) {
    llvm::SmallVector<mlir::Type, 4> substituted;
    for (mlir::Type result : results)
      substituted.push_back(
          substituteType(types, result, bindings, /*eraseUnbound=*/true));
    return types.tupleOf(types.join(substituted));
  }
  return mlir::Type();
}

std::optional<py::CallableType>
expandParamSpecForInvocation(const AlgorithmM &types, py::CallableType callable,
                             mlir::ArrayRef<mlir::Type> positional,
                             mlir::ArrayRef<CallKeywordType> keywords,
                             TypeBindingMap &bindings,
                             std::size_t firstParameter) {
  llvm::ArrayRef<mlir::Type> formals = callable.getPositionalTypes();
  std::optional<std::size_t> paramSpecIndex;
  std::optional<std::size_t> typeVarTupleIndex;
  for (std::size_t index = firstParameter, end = formals.size(); index < end;
       ++index) {
    bool isParamSpec = static_cast<bool>(paramSpecName(formals[index]));
    bool isTypeVarTuple =
        static_cast<bool>(unpackedTypeVarTupleName(formals[index]));
    if (!isParamSpec && !isTypeVarTuple)
      continue;
    if (isParamSpec && (paramSpecIndex || typeVarTupleIndex))
      return std::nullopt;
    if (isTypeVarTuple && (paramSpecIndex || typeVarTupleIndex))
      return std::nullopt;
    if (isParamSpec)
      paramSpecIndex = index;
    else
      typeVarTupleIndex = index;
  }

  if (paramSpecIndex) {
    std::size_t visibleBefore = *paramSpecIndex - firstParameter;
    std::size_t trailing = formals.size() - *paramSpecIndex - 1;
    if (positional.size() < visibleBefore + trailing)
      return std::nullopt;

    std::size_t capturedCount = positional.size() - visibleBefore - trailing;
    llvm::ArrayRef<mlir::Type> captured =
        positional.slice(visibleBefore, capturedCount);
    std::optional<std::string> name = paramSpecName(formals[*paramSpecIndex]);
    llvm::SmallVector<CallKeywordType, 4> capturedKeywords =
        captureParamSpecKeywords(callable, firstParameter, paramSpecIndex,
                                 keywords);
    if (!name ||
        !bindParamSpecPack(types, *name, captured, capturedKeywords, bindings))
      return std::nullopt;
  } else if (typeVarTupleIndex) {
    std::size_t visibleBefore = *typeVarTupleIndex - firstParameter;
    std::size_t trailing = formals.size() - *typeVarTupleIndex - 1;
    if (positional.size() < visibleBefore + trailing)
      return std::nullopt;

    std::size_t capturedCount = positional.size() - visibleBefore - trailing;
    llvm::ArrayRef<mlir::Type> captured =
        positional.slice(visibleBefore, capturedCount);
    std::optional<std::string> name =
        unpackedTypeVarTupleName(formals[*typeVarTupleIndex]);
    if (!name || !bindTypeVarTuplePack(types, *name, captured, bindings))
      return std::nullopt;
  } else if (callable.hasVararg()) {
    if (std::optional<std::string> name =
            paramSpecName(callable.getVarargType())) {
      std::size_t fixedVisible = formals.size() - firstParameter;
      llvm::ArrayRef<mlir::Type> captured;
      if (positional.size() > fixedVisible)
        captured = positional.drop_front(fixedVisible);
      if (!bindParamSpecPack(types, *name, captured, {}, bindings,
                             /*merge=*/true))
        return std::nullopt;
    } else if (std::optional<std::string> tupleName =
                   unpackedTypeVarTupleName(callable.getVarargType())) {
      std::size_t fixedVisible = formals.size() - firstParameter;
      llvm::ArrayRef<mlir::Type> captured;
      if (positional.size() > fixedVisible)
        captured = positional.drop_front(fixedVisible);
      if (!bindTypeVarTuplePack(types, *tupleName, captured, bindings))
        return std::nullopt;
    }
  }

  if (callable.hasKwarg()) {
    if (std::optional<std::string> name =
            paramSpecName(callable.getKwargType())) {
      llvm::SmallVector<CallKeywordType, 4> capturedKeywords =
          captureParamSpecKeywords(callable, firstParameter, paramSpecIndex,
                                   keywords);
      if (!bindParamSpecPack(types, *name, {}, capturedKeywords, bindings,
                             /*merge=*/true))
        return std::nullopt;
    }
  }

  return substituteCallable(types, callable, bindings,
                            /*eraseUnbound=*/false);
}

int unboundStaticParameterCount(mlir::Type type) {
  if (!type)
    return 0;
  if (staticParameterName(type))
    return 1;
  if (auto contract = mlir::dyn_cast_if_present<py::ContractType>(type)) {
    int count = 0;
    for (mlir::Type arg : contract.getArguments())
      count += unboundStaticParameterCount(arg);
    return count;
  }
  if (auto protocol = mlir::dyn_cast_if_present<py::ProtocolType>(type)) {
    int count = 0;
    for (mlir::Type arg : protocol.getArguments())
      count += unboundStaticParameterCount(arg);
    return count;
  }
  if (auto unionType = mlir::dyn_cast_if_present<py::UnionType>(type)) {
    int count = 0;
    for (mlir::Type member : unionType.getMemberTypes())
      count += unboundStaticParameterCount(member);
    return count;
  }
  if (auto typeType = mlir::dyn_cast_if_present<py::TypeType>(type))
    return unboundStaticParameterCount(typeType.getInstanceType());
  if (auto unpack = mlir::dyn_cast_if_present<py::UnpackType>(type))
    return unboundStaticParameterCount(unpack.getPackedType());
  if (auto callable = mlir::dyn_cast_if_present<py::CallableType>(type)) {
    int count = 0;
    for (mlir::Type arg : callable.getPositionalTypes())
      count += unboundStaticParameterCount(arg);
    for (mlir::Type arg : callable.getKwOnlyTypes())
      count += unboundStaticParameterCount(arg);
    for (mlir::Type result : callable.getResultTypes())
      count += unboundStaticParameterCount(result);
    if (callable.hasVararg())
      count += unboundStaticParameterCount(callable.getVarargType());
    if (callable.hasKwarg())
      count += unboundStaticParameterCount(callable.getKwargType());
    return count;
  }
  return 0;
}

int matchSpecificity(const AlgorithmM &types, mlir::Type expected,
                     mlir::Type actual) {
  expected = types.widenLiteral(expected);
  actual = types.widenLiteral(actual);
  if (!expected || !actual)
    return 0;
  if (expected == actual)
    return 12;
  if (isObjectTop(types, expected))
    return 0;
  if (auto unionType = mlir::dyn_cast_if_present<py::UnionType>(expected)) {
    int best = 0;
    for (mlir::Type member : unionType.getMemberTypes())
      best = std::max(best, matchSpecificity(types, member, actual));
    return best > 0 ? best - 1 : 0;
  }
  if (typeAccepts(types, expected, actual))
    return 4;
  return 0;
}

int callableSpecificity(const AlgorithmM &types, py::CallableType callable,
                        std::size_t firstParameter) {
  if (!callable)
    return 0;
  int score = 0;
  llvm::ArrayRef<mlir::Type> positional = callable.getPositionalTypes();
  for (std::size_t index = firstParameter, end = positional.size(); index < end;
       ++index) {
    if (!isObjectTop(types, positional[index]))
      score += 2;
    score -= 4 * unboundStaticParameterCount(positional[index]);
  }
  for (mlir::Type arg : callable.getKwOnlyTypes()) {
    if (!isObjectTop(types, arg))
      score += 2;
    score -= 4 * unboundStaticParameterCount(arg);
  }
  for (mlir::Type result : callable.getResultTypes()) {
    if (!isObjectTop(types, result))
      score += 3;
    score -= 8 * unboundStaticParameterCount(result);
  }
  if (!callable.hasVararg())
    score += 2;
  if (!callable.hasKwarg())
    score += 2;
  return score;
}

std::optional<CallSolution>
tryCallableApplication(const AlgorithmM &types, py::CallableType callable,
                       mlir::ArrayRef<mlir::Type> positional,
                       mlir::ArrayRef<CallKeywordType> keywords,
                       TypeBindingMap bindings = {},
                       std::size_t firstParameter = 0) {
  std::optional<py::CallableType> expanded = expandParamSpecForInvocation(
      types, callable, positional, keywords, bindings, firstParameter);
  if (!expanded)
    return std::nullopt;

  py::CallableApplicationShapeOptions opts;
  opts.firstParameter = firstParameter;
  int specificity = callableSpecificity(types, *expanded, firstParameter);
  auto resolved = py::resolveCallableApplicationShape(
      *expanded, positional, keywords, opts,
      [&](mlir::Type expected, mlir::Type actual) {
        if (!bindExpectedType(types, expected, actual, bindings))
          return false;
        specificity += matchSpecificity(
            types,
            substituteType(types, expected, bindings, /*eraseUnbound=*/true),
            actual);
        return true;
      },
      [](const CallKeywordType &keyword) -> llvm::StringRef {
        return keyword.name;
      },
      [](const CallKeywordType &keyword) { return keyword.type; });
  if (!resolved)
    return std::nullopt;
  py::CallableType resolvedEvidence =
      substituteCallable(types, *expanded, bindings, /*eraseUnbound=*/false);
  if (!resolvedEvidence || unboundStaticParameterCount(resolvedEvidence) != 0)
    return std::nullopt;
  mlir::Type result = callableResultType(types, *expanded, bindings);
  if (!result || unboundStaticParameterCount(result) != 0)
    return std::nullopt;
  return CallSolution{result,
                      std::move(bindings),
                      resolvedEvidence,
                      {},
                      std::nullopt,
                      resolved->score + specificity};
}

bool sameCallSolution(const CallSolution &lhs, const CallSolution &rhs) {
  return lhs.result == rhs.result &&
         lhs.callableContract == rhs.callableContract &&
         lhs.methodName == rhs.methodName &&
         lhs.receiverManifestClass == rhs.receiverManifestClass;
}

std::optional<CallSolution> selectCallableApplication(
    const AlgorithmM &types, llvm::ArrayRef<py::CallableType> candidates,
    mlir::ArrayRef<mlir::Type> positional,
    mlir::ArrayRef<CallKeywordType> keywords, TypeBindingMap bindings = {},
    std::size_t firstParameter = 0) {
  auto selection = lython::selection::bestCandidate<CallSolution>(
      [](const CallSolution &solution) { return solution.score; },
      sameCallSolution);
  for (py::CallableType candidate : candidates) {
    if (std::optional<CallSolution> solution = tryCallableApplication(
            types, candidate, positional, keywords, bindings, firstParameter))
      selection.consider(std::move(*solution));
  }
  return std::move(selection).finish();
}

std::optional<CallSolution>
tryManifestMethod(const AlgorithmM &types, mlir::Type receiverType,
                  llvm::StringRef methodName,
                  mlir::ArrayRef<mlir::Type> positional,
                  mlir::ArrayRef<CallKeywordType> keywords = {}) {
  // Positional tuple typing: a tuple contract carrying one argument PER
  // POSITION (heterogeneous annotations/literals, dict.items()'s
  // tuple[$K,$V]) types a literal-index __getitem__ as that position's
  // member. The homogeneous view (joined members) instantiates the manifest
  // contract; only the RESULT narrows to the indexed member.
  if (methodName == "__getitem__" && positional.size() == 1 &&
      keywords.empty()) {
    auto tuple = mlir::dyn_cast_if_present<py::ContractType>(
        types.widenLiteral(receiverType));
    if (tuple && tuple.getContractName() == "builtins.tuple" &&
        tuple.getArguments().size() > 1) {
      if (std::optional<std::int64_t> index =
              literalIntegerFromType(positional.front())) {
        llvm::ArrayRef<mlir::Type> members = tuple.getArguments();
        std::int64_t position =
            *index < 0 ? *index + static_cast<std::int64_t>(members.size())
                       : *index;
        if (position >= 0 &&
            position < static_cast<std::int64_t>(members.size())) {
          mlir::Type joinedView = types.tupleOf(types.join(members));
          if (std::optional<CallSolution> solution = tryManifestMethod(
                  types, joinedView, methodName, positional, keywords)) {
            solution->result = members[position];
            return solution;
          }
        }
      }
    }
  }

  const py::protocols::Table &table =
      py::protocols::Table::get(types.getContext());
  auto selection = lython::selection::bestCandidate<CallSolution>(
      [](const CallSolution &solution) { return solution.score; },
      sameCallSolution);

  for (py::protocols::ContractResolution candidate :
       table.methodContractCandidatesWithEvidence(receiverType, methodName)) {
    py::CallableType signature = candidate.method.signature;
    TypeBindingMap bindings = std::move(candidate.typeBindings);
    if (signature.getPositionalTypes().empty())
      continue;
    if (!bindExpectedType(types, signature.getPositionalTypes().front(),
                          receiverType, bindings))
      continue;
    std::optional<CallSolution> solution = tryCallableApplication(
        types, signature, positional, keywords, std::move(bindings),
        /*firstParameter=*/1);
    if (!solution)
      continue;
    solution->score += candidate.score;
    solution->methodName = methodName.str();
    if (candidate.receiverEvidence)
      solution->receiverManifestClass =
          candidate.receiverEvidence->manifestClass;
    selection.consider(std::move(*solution));
  }
  return std::move(selection).finish();
}

std::optional<mlir::Type> primitiveBinaryResultType(mlir::Type left,
                                                    mlir::Type right,
                                                    const parser::Node *op) {
  if (!left || !right)
    return std::nullopt;
  if (ast::isOperator(op, "Add") || ast::isOperator(op, "Sub") ||
      ast::isOperator(op, "Mult")) {
    if (left == right && (mlir::isa<mlir::IntegerType, mlir::FloatType>(left) ||
                          mlir::isa<mlir::RankedTensorType>(left)))
      return left;
  }
  if (!ast::isOperator(op, "MatMult"))
    return std::nullopt;

  auto lhsTensor = mlir::dyn_cast<mlir::RankedTensorType>(left);
  auto rhsTensor = mlir::dyn_cast<mlir::RankedTensorType>(right);
  if (!lhsTensor || !rhsTensor || lhsTensor.getRank() != 2 ||
      rhsTensor.getRank() != 2 ||
      lhsTensor.getElementType() != rhsTensor.getElementType() ||
      lhsTensor.getDimSize(1) != rhsTensor.getDimSize(0))
    return std::nullopt;
  llvm::SmallVector<std::int64_t, 2> shape{lhsTensor.getDimSize(0),
                                           rhsTensor.getDimSize(1)};
  return mlir::RankedTensorType::get(shape, lhsTensor.getElementType());
}

} // namespace

AlgorithmM::AlgorithmM(mlir::MLIRContext &context) : context(context) {}

AlgorithmM::Scope::Scope(Scope &&other) noexcept : owner(other.owner) {
  other.owner = nullptr;
}

AlgorithmM::Scope &AlgorithmM::Scope::operator=(Scope &&other) noexcept {
  if (this == &other)
    return *this;
  reset();
  owner = other.owner;
  other.owner = nullptr;
  return *this;
}

AlgorithmM::Scope::~Scope() { reset(); }

void AlgorithmM::Scope::reset() {
  if (!owner)
    return;
  owner->popScope();
  owner = nullptr;
}

void AlgorithmM::seedBuiltins() {
  bindSymbol("None", none());
  bindSymbol("True", literal("True"));
  bindSymbol("False", literal("False"));
  // Builtin free-function signatures come from the module manifests
  // (ly.typing.function_contracts) so manifests stay the single trusted
  // source for Python-visible contracts. The C++ fallbacks remain only if no
  // manifest declares the contract.
  const py::protocols::Table &table = py::protocols::Table::get(context);
  bindSymbol("print", table.freeFunctionContract("builtins.print")
                          .value_or(py::CallableType::get(
                              &context, {}, {}, tupleOf(object()), {},
                              {none()})));
  bindSymbol("len", table.freeFunctionContract("builtins.len")
                        .value_or(py::CallableType::get(&context, {object()}, {},
                                                        {}, {}, {intType()})));
  bindClass("object", object());
  bindClass("bool", boolType());
  bindClass("int", intType());
  bindClass("float", floatType());
  bindClass("str", strType());
  bindClass("BaseException", contract("builtins.BaseException"));
  bindClass("Exception", contract("builtins.Exception"));
  bindClass("RuntimeError", contract("builtins.RuntimeError"));
  bindClass("TypeError", contract("builtins.TypeError"));
  bindClass("ValueError", contract("builtins.ValueError"));
  bindClass("ArithmeticError", contract("builtins.ArithmeticError"));
  bindClass("LookupError", contract("builtins.LookupError"));
  bindClass("ZeroDivisionError", contract("builtins.ZeroDivisionError"));
  bindClass("KeyError", contract("builtins.KeyError"));
  bindClass("IndexError", contract("builtins.IndexError"));
  bindClass("AssertionError", contract("builtins.AssertionError"));
  bindClass("StopIteration", contract("builtins.StopIteration"));
  bindClass("StopAsyncIteration", contract("builtins.StopAsyncIteration"));
  bindClass("nullcontext", contract("contextlib.nullcontext"));
  bindClass("range", contract("builtins.range"));
}

mlir::Type AlgorithmM::object() const { return contract("builtins.object"); }
mlir::Type AlgorithmM::any() const { return contract("typing.Any"); }
mlir::Type AlgorithmM::none() const { return literal("None"); }
mlir::Type AlgorithmM::boolType() const { return contract("builtins.bool"); }
mlir::Type AlgorithmM::intType() const { return contract("builtins.int"); }
mlir::Type AlgorithmM::strType() const { return contract("builtins.str"); }
mlir::Type AlgorithmM::floatType() const { return contract("builtins.float"); }

mlir::Type AlgorithmM::contract(llvm::StringRef name,
                                mlir::ArrayRef<mlir::Type> arguments) const {
  return py::ContractType::get(&context, name, arguments);
}

mlir::Type AlgorithmM::protocol(llvm::StringRef name,
                                mlir::ArrayRef<mlir::Type> arguments) const {
  return py::ProtocolType::get(&context, name, arguments);
}

mlir::Type AlgorithmM::literal(llvm::StringRef spelling) const {
  return py::LiteralType::get(&context, spelling);
}

mlir::Type AlgorithmM::typeObject(mlir::Type instanceType) const {
  return py::TypeType::get(&context, instanceType);
}

mlir::Type AlgorithmM::tupleOf(mlir::Type elementType) const {
  llvm::SmallVector<mlir::Type, 1> args;
  if (elementType)
    args.push_back(elementType);
  return contract("builtins.tuple", args);
}

mlir::Type AlgorithmM::listOf(mlir::Type elementType) const {
  llvm::SmallVector<mlir::Type, 1> args;
  if (elementType)
    args.push_back(elementType);
  return contract("builtins.list", args);
}

mlir::Type AlgorithmM::dictOf(mlir::Type keyType, mlir::Type valueType) const {
  llvm::SmallVector<mlir::Type, 2> args;
  if (keyType)
    args.push_back(keyType);
  if (valueType)
    args.push_back(valueType);
  return contract("builtins.dict", args);
}

mlir::Type AlgorithmM::iteratorOf(mlir::Type elementType) const {
  llvm::SmallVector<mlir::Type, 1> args;
  if (elementType)
    args.push_back(elementType);
  return protocol("Iterator", args);
}

mlir::Type AlgorithmM::coroutineOf(mlir::Type resultType) const {
  return contract("types.CoroutineType",
                  {any(), any(), resultType ? resultType : any()});
}

AlgorithmM::Scope AlgorithmM::pushScope() const {
  scopes.emplace_back();
  scopedCanonicalBindings.emplace_back();
  scopedClasses.emplace_back();
  return Scope(*this);
}

void AlgorithmM::popScope() const {
  if (!scopes.empty()) {
    scopes.pop_back();
    scopedCanonicalBindings.pop_back();
    scopedClasses.pop_back();
  }
}

// The `type ? type : object()` guards below defend an internal solver
// invariant: every symbol binding produced by AlgorithmM carries a resolved
// type. A null type would be a solver gap, not a language feature. The `object`
// top used as the guard value carries no protocol contract, so if such a gap
// ever reached lowering the erased binding would be rejected for lack of
// evidence rather than dispatched dynamically. The guard is never exercised by
// the accepted example suite (verified via null-binding instrumentation).
void AlgorithmM::bindLocalSymbol(llvm::StringRef name, mlir::Type type) const {
  if (scopes.empty())
    return;
  scopes.back()[name] = type ? type : object();
}

void AlgorithmM::bindSymbol(llvm::StringRef name, mlir::Type type) {
  if (!scopes.empty()) {
    scopes.back()[name] = type ? type : object();
    scopedCanonicalBindings.back().erase(name);
    return;
  }
  symbols[name] = type ? type : object();
  canonicalBindings.erase(name);
}

void AlgorithmM::bindCanonicalSymbol(llvm::StringRef name,
                                     llvm::StringRef canonical,
                                     mlir::Type type) {
  bindSymbol(name, type);
  if (!scopedCanonicalBindings.empty())
    scopedCanonicalBindings.back()[name] = canonical.str();
  else
    canonicalBindings[name] = canonical.str();
}

void AlgorithmM::bindAnnotationAlias(llvm::StringRef name,
                                     llvm::StringRef target) {
  annotationAliases[name] = target.str();
}

std::string AlgorithmM::resolveAnnotationName(llvm::StringRef name) const {
  auto found = annotationAliases.find(name);
  if (found != annotationAliases.end())
    return found->second;
  return name.str();
}

std::optional<mlir::Type> AlgorithmM::lookupSymbol(llvm::StringRef name) const {
  for (auto it = scopes.rbegin(), e = scopes.rend(); it != e; ++it) {
    auto found = it->find(name);
    if (found != it->end())
      return found->second;
  }
  auto found = symbols.find(name);
  if (found == symbols.end())
    return std::nullopt;
  return found->second;
}

std::optional<std::string>
AlgorithmM::lookupCanonicalBinding(llvm::StringRef name) const {
  llvm::StringRef root = name.split('.').first;
  auto canonicalIt = scopedCanonicalBindings.rbegin();
  for (auto scopeIt = scopes.rbegin(), scopeEnd = scopes.rend();
       scopeIt != scopeEnd; ++scopeIt, ++canonicalIt) {
    auto scopedCanonical = canonicalIt->find(name);
    if (scopedCanonical != canonicalIt->end())
      return scopedCanonical->second;
    if (scopeIt->find(name) != scopeIt->end() ||
        scopeIt->find(root) != scopeIt->end())
      return std::nullopt;
  }
  auto found = canonicalBindings.find(name);
  if (found == canonicalBindings.end())
    return std::nullopt;
  return found->second;
}

void AlgorithmM::bindClass(llvm::StringRef name, mlir::Type instanceType) {
  mlir::Type resolved = instanceType ? instanceType : contract(name);
  if (!scopes.empty()) {
    scopedClasses.back()[name] = resolved;
    scopes.back()[name] = typeObject(resolved);
    scopedCanonicalBindings.back().erase(name);
    return;
  }
  classes[name] = resolved;
  symbols[name] = typeObject(resolved);
  canonicalBindings.erase(name);
}

std::optional<mlir::Type> AlgorithmM::lookupClass(llvm::StringRef name) const {
  for (auto it = scopedClasses.rbegin(), e = scopedClasses.rend(); it != e;
       ++it) {
    auto found = it->find(name);
    if (found != it->end())
      return found->second;
  }
  auto found = classes.find(name);
  if (found == classes.end())
    return std::nullopt;
  return found->second;
}

bool AlgorithmM::bindImportedModule(llvm::StringRef module,
                                    llvm::StringRef localName) {
  std::string localStorage;
  if (localName.empty()) {
    localStorage = module.split('.').first.str();
    localName = localStorage;
  }

  bool handled = false;
  auto bindModuleObject = [&] {
    if (!handled)
      bindSymbol(localName, object());
    handled = true;
  };

  const py::protocols::Table &manifestTable =
      py::protocols::Table::get(context);
  auto importCallableContract = [&](const char *canonicalName,
                                    ImportCallableFactory factory) -> mlir::Type {
    // Prefer the manifest-declared contract; fall back to the C++ factory only
    // for names not yet declared in a runtime manifest.
    if (std::optional<mlir::Type> contract =
            manifestTable.freeFunctionContract(canonicalName))
      return *contract;
    return importCallableType(*this, factory);
  };

  for (const ModuleCallableImport &entry : kModuleCallableImports) {
    if (module != entry.module)
      continue;
    bindModuleObject();
    bindCanonicalSymbol(importedAttribute(localName, entry.localAttr),
                        entry.canonicalName,
                        importCallableContract(entry.canonicalName,
                                               entry.factory));
  }

  for (const ModuleAliasImport &entry : kModuleAliasImports) {
    if (module != entry.module)
      continue;
    bindModuleObject();
    std::string local = importedAttribute(localName, entry.localAttr);
    bindCanonicalSymbol(local, entry.canonicalName, typeObject(object()));
    if (entry.annotationAlias)
      bindAnnotationAlias(local, entry.canonicalName);
  }

  for (const ModuleStringConstantImport &entry : kModuleStringConstantImports) {
    if (module != entry.module)
      continue;
    bindModuleObject();
    bindCanonicalSymbol(importedAttribute(localName, entry.localAttr),
                        entry.canonicalName, strType());
  }

  if (std::optional<AnnotationModuleStyle> style =
          annotationModuleStyle(module)) {
    bindModuleObject();
    bindAnnotationModuleAliases([&](llvm::StringRef name) {
      bindAnnotationAlias(
          importedAnnotationAlias(module, localName, *style, name), name);
    });
  }

  if (bindManifestModuleClassExports(*this, module, localName))
    handled = true;
  if (bindManifestModuleCallableExports(*this, module, localName))
    handled = true;
  if (bindManifestModuleFloatConstants(*this, module, localName))
    handled = true;

  return handled;
}

bool AlgorithmM::bindImportedName(llvm::StringRef module,
                                  llvm::StringRef exportedName,
                                  llvm::StringRef localName) {
  if (localName.empty())
    localName = exportedName;

  const py::protocols::Table &manifestTable =
      py::protocols::Table::get(context);
  for (const NameCallableImport &entry : kNameCallableImports) {
    if (module != entry.module || exportedName != entry.exportedName)
      continue;
    // Prefer the manifest-declared contract; fall back to the C++ factory only
    // for names not yet declared in a runtime manifest.
    mlir::Type contract =
        manifestTable.freeFunctionContract(entry.canonicalName)
            .value_or(importCallableType(*this, entry.factory));
    bindCanonicalSymbol(localName, entry.canonicalName, contract);
    return true;
  }

  for (const NameAliasImport &entry : kNameAliasImports) {
    if (module != entry.module || exportedName != entry.exportedName)
      continue;
    bindCanonicalSymbol(localName, entry.canonicalName, typeObject(object()));
    if (entry.annotationAlias)
      bindAnnotationAlias(localName, entry.canonicalName);
    return true;
  }

  for (const NameStringConstantImport &entry : kNameStringConstantImports) {
    if (module != entry.module || exportedName != entry.exportedName)
      continue;
    bindCanonicalSymbol(localName, entry.canonicalName, strType());
    return true;
  }

  const py::protocols::Table &table =
      py::protocols::Table::get(getContext());
  if (std::optional<std::string> contract =
          table.moduleClassExport(module, exportedName))
    return bindManifestClassImport(*this, localName, *contract);

  if (table.isModuleCallableExport(module, exportedName)) {
    std::string canonical = (llvm::Twine(module) + "." + exportedName).str();
    // Prefer the manifest-declared Callable contract; a declared callable
    // export without a contract binds as a generic function object (calls
    // resolve from lowering evidence).
    bindCanonicalSymbol(localName, canonical,
                        table.freeFunctionContract(canonical)
                            .value_or(contract("builtins.function")));
    return true;
  }

  {
    std::string canonical = (llvm::Twine(module) + "." + exportedName).str();
    if (table.moduleFloatConstant(canonical)) {
      bindCanonicalSymbol(localName, canonical, floatType());
      return true;
    }
  }

  std::string submodule = (llvm::Twine(module) + "." + exportedName).str();
  bool boundSubmodule = bindManifestModuleClassExports(*this, submodule,
                                                       localName);
  boundSubmodule =
      bindManifestModuleCallableExports(*this, submodule, localName) ||
      boundSubmodule;
  if (boundSubmodule)
    return true;

  if (moduleExportsAnnotationNames(module)) {
    if (isImportedAnnotationName(exportedName)) {
      // annotationType interprets these names directly. Binding the local
      // spelling acknowledges import aliases without creating module objects.
      bindSymbol(localName, object());
      bindAnnotationAlias(localName, exportedName);
      return true;
    }
  }

  return false;
}

mlir::Type AlgorithmM::annotationType(const parser::Node *node) const {
  if (!node)
    return object();
  if (node->kind == "Name") {
    std::string resolved = resolveAnnotationName(ast::nameSpelling(*node));
    llvm::StringRef name(resolved);
    if (auto symbol = lookupSymbol(name)) {
      if (mlir::isa<py::TypeVarType, py::ParamSpecType, py::TypeVarTupleType>(
              *symbol))
        return *symbol;
    }
    if (annotationNameIs(name, "int"))
      return intType();
    if (annotationNameIs(name, "str"))
      return strType();
    if (annotationNameIs(name, "bool"))
      return boolType();
    if (annotationNameIs(name, "float"))
      return floatType();
    if (annotationNameIs(name, "object"))
      return object();
    if (annotationNameIs(name, "Any"))
      return any();
    if (annotationNameIs(name, "None"))
      return none();
    if (annotationNameIs(name, "Self"))
      return py::SelfType::get(&context);
    if (auto protocolName = protocolAnnotationName(name))
      return protocol(*protocolName);
    if (auto contractName = contractAnnotationName(name))
      return contract(*contractName);
    if (auto knownClass = lookupClass(name))
      return *knownClass;
    return contract((llvm::Twine("builtins.") + name).str());
  }
  if (node->kind == "Constant")
    return isNoneConstant(node) ? none() : literal(literalSpelling(*node));
  if (node->kind == "Attribute") {
    std::string qualified = ast::qualifiedName(node);
    std::string_view spelling = ast::nameSpelling(*node);
    std::string resolved = resolveAnnotationName(
        qualified.empty() ? llvm::StringRef(spelling.data(), spelling.size())
                          : llvm::StringRef(qualified));
    llvm::StringRef name(resolved);
    if (annotationNameIs(name, "Any"))
      return any();
    if (annotationNameIs(name, "Self"))
      return py::SelfType::get(&context);
    if (auto protocolName = protocolAnnotationName(name))
      return protocol(*protocolName);
    if (auto contractName = contractAnnotationName(name))
      return contract(*contractName);
    if (auto knownClass = lookupClass(name))
      return *knownClass;
    return contract(name);
  }
  if (node->kind == "BinOp" && ast::isOperator(ast::node(*node, "op"), "BitOr"))
    return py::UnionType::getNormalized(
        &context, {annotationType(ast::node(*node, "left")),
                   annotationType(ast::node(*node, "right"))});
  if (node->kind == "Subscript") {
    if (std::optional<PrimitiveTypeSpec> primitive =
            primitiveTypeSpecFromSubscript(node, *this))
      return primitive->type;

    const parser::Node *base = ast::node(*node, "value");
    const parser::Node *slice = ast::node(*node, "slice");
    std::string qualifiedBase = ast::qualifiedName(base);
    std::string_view baseSpelling =
        base ? ast::nameSpelling(*base) : std::string_view();
    std::string resolvedBase = resolveAnnotationName(
        qualifiedBase.empty()
            ? llvm::StringRef(baseSpelling.data(), baseSpelling.size())
            : llvm::StringRef(qualifiedBase));
    llvm::StringRef baseName(resolvedBase);
    if (annotationNameIs(baseName, "Optional"))
      return py::UnionType::getNormalized(&context,
                                          {annotationType(slice), none()});
    if (annotationNameIs(baseName, "Union")) {
      llvm::SmallVector<mlir::Type, 4> members;
      if (slice && slice->kind == "Tuple") {
        if (const auto *elts = ast::nodeList(*slice, "elts"))
          for (const parser::NodePtr &elt : *elts)
            members.push_back(annotationType(elt.get()));
      } else {
        members.push_back(annotationType(slice));
      }
      return py::UnionType::getNormalized(&context, members);
    }
    if (annotationNameIs(baseName, "list") ||
        annotationNameIs(baseName, "List"))
      return listOf(annotationType(slice));
    if (annotationNameIs(baseName, "tuple") ||
        annotationNameIs(baseName, "Tuple")) {
      if (slice && slice->kind == "Tuple") {
        // tuple[A, B, ...]: uniform members collapse to the homogeneous
        // spelling, differing members stay positional (see tupleOfMembers).
        // An Ellipsis member (tuple[T, ...]) only marks arbitrary arity and
        // contributes no type.
        if (const auto *elts = ast::nodeList(*slice, "elts")) {
          llvm::SmallVector<mlir::Type, 4> members;
          for (const parser::NodePtr &elt : *elts) {
            if (elt && elt->kind == "Constant" &&
                ast::isEllipsisField(*elt, "value"))
              continue;
            members.push_back(annotationType(elt.get()));
          }
          return tupleOfMembers(*this, members);
        }
      }
      return tupleOf(annotationType(slice));
    }
    if (annotationNameIs(baseName, "dict") ||
        annotationNameIs(baseName, "Dict") ||
        annotationNameIs(baseName, "Mapping")) {
      mlir::Type key = object();
      mlir::Type value = object();
      if (slice && slice->kind == "Tuple") {
        if (const auto *elts = ast::nodeList(*slice, "elts")) {
          if (!elts->empty())
            key = annotationType(elts->front().get());
          if (elts->size() > 1)
            value = annotationType((*elts)[1].get());
        }
      }
      return dictOf(key, value);
    }
    if (annotationNameIs(baseName, "type") ||
        annotationNameIs(baseName, "Type"))
      return typeObject(annotationType(slice));
    if (annotationNameIs(baseName, "Unpack"))
      return py::UnpackType::get(&context, annotationType(slice));
    if (annotationNameIs(baseName, "Callable")) {
      llvm::SmallVector<mlir::Type, 4> positional;
      mlir::Type vararg;
      mlir::Type kwarg;
      mlir::Type result = object();
      if (slice && slice->kind == "Tuple") {
        if (const auto *elts = ast::nodeList(*slice, "elts")) {
          if (!elts->empty()) {
            const parser::Node *params = elts->front().get();
            if (params && params->kind == "List") {
              if (const auto *paramElts = ast::nodeList(*params, "elts"))
                for (const parser::NodePtr &param : *paramElts)
                  positional.push_back(annotationType(param.get()));
            } else if (params && params->kind == "Tuple") {
              if (const auto *paramElts = ast::nodeList(*params, "elts"))
                for (const parser::NodePtr &param : *paramElts)
                  positional.push_back(annotationType(param.get()));
            } else {
              vararg = tupleOf(any());
              kwarg = dictOf(strType(), any());
            }
          }
          if (elts->size() > 1)
            result = annotationType((*elts)[1].get());
        }
      }
      return py::CallableType::get(&context, positional, {}, vararg, kwarg,
                                   {result});
    }
    if (annotationNameIs(baseName, "Literal"))
      return literal(slice && slice->kind == "Constant"
                         ? literalSpelling(*slice)
                         : "object");
    llvm::SmallVector<mlir::Type, 4> arguments;
    if (slice && slice->kind == "Tuple") {
      if (const auto *elts = ast::nodeList(*slice, "elts"))
        for (const parser::NodePtr &elt : *elts)
          arguments.push_back(annotationType(elt.get()));
    } else if (slice) {
      arguments.push_back(annotationType(slice));
    }
    if (auto protocolName = protocolAnnotationName(baseName))
      return protocol(*protocolName, arguments);
    if (auto contractName = contractAnnotationName(baseName))
      return contract(*contractName, arguments);
    if (auto knownClass = lookupClass(baseName)) {
      if (auto contractType =
              mlir::dyn_cast_if_present<py::ContractType>(*knownClass))
        return contract(contractType.getContractName(), arguments);
      return *knownClass;
    }
  }
  return object();
}

mlir::Type AlgorithmM::inferExpr(const parser::Node *node) const {
  if (!node)
    return object();
  if (node->kind == "Constant") {
    if (ast::isNoneField(*node, "value"))
      return none();
    if (auto value = ast::boolean(*node, "value"))
      return literal(*value ? "True" : "False");
    if (auto value = ast::integer(*node, "value"))
      return literal(std::to_string(*value));
    if (ast::floating(*node, "value"))
      return floatType();
    if (auto value = ast::string(*node, "value"))
      return literal("\"" + std::string(*value) + "\"");
    if (const auto *fieldValue = ast::field(*node, "value"))
      if (const auto *big = std::get_if<parser::BigInteger>(fieldValue))
        return literal(big->decimal);
    return object();
  }
  // Platform constants type as string literals of the CURRENT TARGET, so
  // `sys.platform == "win32"` branches fold statically (the platform switch
  // idiom runtime lib modules rely on).
  auto staticStringLiteral =
      [&](llvm::StringRef binding) -> std::optional<mlir::Type> {
    std::string canonical =
        lookupCanonicalBinding(binding).value_or(binding.str());
    if (!py::platform_constants::isStaticStringBinding(canonical))
      return std::nullopt;
    if (std::optional<std::string> value =
            py::platform_constants::staticStringValue(canonical, targetTriple))
      return literal("\"" + *value + "\"");
    return std::nullopt;
  };
  if (node->kind == "Name") {
    llvm::StringRef name = ast::nameSpelling(*node);
    if (auto found = lookupSymbol(name))
      return *found;
    if (std::optional<mlir::Type> constant = staticStringLiteral(name))
      return *constant;
    return object();
  }
  if (node->kind == "Attribute") {
    std::string qualified = ast::qualifiedName(node);
    if (!qualified.empty()) {
      if (std::optional<mlir::Type> constant = staticStringLiteral(qualified))
        return *constant;
      if (auto cls = lookupClass(qualified))
        return typeObject(*cls);
      if (auto found = lookupSymbol(qualified))
        return *found;
    }
    if (mlir::Type objectType = inferExpr(ast::node(*node, "value"))) {
      const py::protocols::Table &table = py::protocols::Table::get(context);
      if (auto attr = ast::string(*node, "attr")) {
        if (std::optional<py::protocols::FieldResolution> field =
                table.resolveFieldContractWithEvidence(widenLiteral(objectType),
                                                       *attr))
          return field->contractType;
        if (std::optional<CallSolution> method =
                tryManifestMethod(*this, widenLiteral(objectType), *attr, {}))
          return method->result;
      }
      return object();
    }
  }
  if (node->kind == "List")
    return listOf(joinedLiteralElementType(*this, *node, "elts"));
  if (node->kind == "Tuple") {
    llvm::SmallVector<mlir::Type, 8> memberTypes;
    if (const auto *elements = ast::nodeList(*node, "elts")) {
      memberTypes.reserve(elements->size());
      for (const parser::NodePtr &element : *elements)
        memberTypes.push_back(widenLiteral(inferExpr(element.get())));
    }
    return tupleOfMembers(*this, memberTypes);
  }
  if (node->kind == "Dict") {
    std::optional<std::pair<mlir::Type, mlir::Type>> keyValueTypes =
        joinedDictLiteralTypes(*this, *node);
    if (!keyValueTypes)
      return dictOf(object(), object());
    return dictOf(keyValueTypes->first, keyValueTypes->second);
  }
  if (node->kind == "Subscript") {
    if (std::optional<PrimitiveTypeSpec> primitive =
            primitiveTypeSpecFromSubscript(node, *this))
      return typeObject(primitive->type);
    mlir::Type container = widenLiteral(inferExpr(ast::node(*node, "value")));
    mlir::Type index = inferExpr(ast::node(*node, "slice"));
    if (std::optional<CallSolution> result =
            tryManifestMethod(*this, container, "__getitem__", {index}))
      return result->result;
    return object();
  }
  if (node->kind == "Compare")
    return boolType();
  if (node->kind == "BoolOp")
    return boolType();
  if (node->kind == "IfExp")
    // Mirrors the emitter: literal arms widen to their contracts (CPython
    // types a ternary of two literals by the common class).
    return join({widenLiteral(inferExpr(ast::node(*node, "body"))),
                 widenLiteral(inferExpr(ast::node(*node, "orelse")))});
  if (node->kind == "UnaryOp") {
    const parser::Node *op = ast::node(*node, "op");
    mlir::Type operand = inferExpr(ast::node(*node, "operand"));
    if (ast::isOperator(op, "Not"))
      return boolType();
    if (ast::isOperator(op, "USub")) {
      const parser::Node *operandNode = ast::node(*node, "operand");
      if (operandNode && operandNode->kind == "Constant") {
        if (auto value = ast::integer(*operandNode, "value"))
          return literal("-" + std::to_string(*value));
        if (const auto *fieldValue = ast::field(*operandNode, "value"))
          if (const auto *big = std::get_if<parser::BigInteger>(fieldValue))
            return literal("-" + big->decimal);
      }
      if (std::optional<CallSolution> result =
              tryManifestMethod(*this, widenLiteral(operand), "__neg__", {}))
        return result->result;
    }
    if (ast::isOperator(op, "UAdd"))
      if (std::optional<CallSolution> result =
              tryManifestMethod(*this, widenLiteral(operand), "__pos__", {}))
        return result->result;
    if (ast::isOperator(op, "Invert"))
      if (std::optional<CallSolution> result =
              tryManifestMethod(*this, widenLiteral(operand), "__invert__", {}))
        return result->result;
    return widenLiteral(operand);
  }
  if (node->kind == "BinOp") {
    const parser::Node *op = ast::node(*node, "op");
    mlir::Type rawLeft = inferExpr(ast::node(*node, "left"));
    mlir::Type rawRight = inferExpr(ast::node(*node, "right"));
    mlir::Type left = widenLiteral(rawLeft);
    mlir::Type right = widenLiteral(rawRight);
    llvm::StringRef method = "__add__";
    if (ast::isOperator(op, "Sub"))
      method = "__sub__";
    else if (ast::isOperator(op, "Mult"))
      method = "__mul__";
    else if (ast::isOperator(op, "Div"))
      method = "__truediv__";
    else if (ast::isOperator(op, "FloorDiv"))
      method = "__floordiv__";
    else if (ast::isOperator(op, "Mod"))
      method = "__mod__";
    else if (ast::isOperator(op, "LShift"))
      method = "__lshift__";
    else if (ast::isOperator(op, "RShift"))
      method = "__rshift__";
    else if (ast::isOperator(op, "BitAnd"))
      method = "__and__";
    else if (ast::isOperator(op, "BitOr"))
      method = "__or__";
    else if (ast::isOperator(op, "BitXor"))
      method = "__xor__";
    if (std::optional<mlir::Type> primitive =
            primitiveBinaryResultType(left, right, op))
      return *primitive;
    if (std::optional<CallSolution> result =
            tryManifestMethod(*this, left, method, {right}))
      return result->result;
    if (left == strType() && right == strType())
      return strType();
    if (ast::isOperator(op, "Div") &&
        (left == intType() || left == floatType()) &&
        (right == intType() || right == floatType()))
      return floatType();
    if (left == floatType() || right == floatType())
      return floatType();
    if (left == intType() && right == intType())
      return intType();
    return join({left, right});
  }
  if (node->kind == "Await") {
    mlir::Type awaitable = widenLiteral(inferExpr(ast::node(*node, "value")));
    if (AwaitInferenceResult inference = inferAwaitWithEvidence(awaitable))
      return inference.resultType;
    return object();
  }
  if (node->kind == "Call") {
    const parser::Node *callee = ast::node(*node, "func");
    if (std::optional<PrimitiveTypeSpec> primitive =
            primitiveTypeSpecFromSubscript(callee, *this))
      return primitive->type;

    llvm::SmallVector<mlir::Type, 8> positional;
    if (const auto *args = ast::nodeList(*node, "args")) {
      for (const parser::NodePtr &arg : *args) {
        if (arg && arg->kind == "Starred") {
          if (!appendStarredCallArgumentTypes(
                  *this, inferExpr(ast::node(*arg, "value")), positional))
            return object();
          continue;
        }
        positional.push_back(inferExpr(arg.get()));
      }
    }

    llvm::SmallVector<CallKeywordType, 4> keywords;
    if (const auto *keywordNodes = ast::nodeList(*node, "keywords")) {
      for (const parser::NodePtr &keyword : *keywordNodes) {
        auto name = ast::string(*keyword, "arg");
        if (!name)
          continue;
        keywords.push_back(CallKeywordType{
            std::string(*name), inferExpr(ast::node(*keyword, "value"))});
      }
    }

    if (callee && callee->kind == "Name") {
      llvm::StringRef name = ast::nameSpelling(*callee);
      if (name == "isinstance")
        return boolType();
      if (name == "len") {
        if (!positional.empty())
          if (std::optional<CallSolution> result = tryManifestMethod(
                  *this, widenLiteral(positional.front()), "__len__", {}))
            return result->result;
        return object();
      }
      if (name == "round") {
        const auto *args = ast::nodeList(*node, "args");
        if (args && !args->empty()) {
          mlir::Type input = widenLiteral(inferExpr(args->front().get()));
          llvm::SmallVector<mlir::Type, 1> extra;
          if (args->size() > 1)
            extra.push_back(inferExpr((*args)[1].get()));
          if (std::optional<CallSolution> result =
                  tryManifestMethod(*this, input, "__round__", extra))
            return result->result;
          if (input == intType())
            return intType();
        }
        return object();
      }
      if (name == "range")
        return contract("builtins.range");
      if (auto cls = lookupClass(name))
        return inferClassInstantiation(*cls, positional, keywords);
      if (std::optional<std::string> canonical = lookupCanonicalBinding(name)) {
        if (*canonical == "lyrt.from_prim" && positional.size() == 1 &&
            keywords.empty())
          return primitivePythonResultType(positional.front(), *this);
        if (*canonical == "asyncio.sleep")
          return inferAsyncioSleepResult(*this, positional, keywords);
      }
      if (auto symbol = lookupSymbol(name))
        return inferCall(*symbol, positional, keywords);
    }
    if (callee && callee->kind == "Attribute") {
      std::string qualified = ast::qualifiedName(callee);
      if (std::optional<std::string> canonical =
              lookupCanonicalBinding(qualified)) {
        if (*canonical == "lyrt.from_prim" && positional.size() == 1 &&
            keywords.empty())
          return primitivePythonResultType(positional.front(), *this);
        if (*canonical == "asyncio.sleep")
          return inferAsyncioSleepResult(*this, positional, keywords);
      }
      if (auto symbol = lookupSymbol(qualified))
        return inferCall(*symbol, positional, keywords);
      if (const parser::Node *receiverNode = ast::node(*callee, "value")) {
        if (auto methodName = ast::string(*callee, "attr")) {
          mlir::Type receiver = widenLiteral(inferExpr(receiverNode));
          CallInferenceResult inference = inferMethodCallWithEvidence(
              receiver, *methodName, positional, keywords);
          return inference ? inference.resultType : object();
        }
      }
    }
    if (callee)
      return inferCall(inferExpr(callee), positional, keywords);
    return object();
  }
  if (node->kind == "Lambda") {
    return functionSignature(*node).callable;
  }
  return object();
}

mlir::Type
AlgorithmM::inferCall(mlir::Type calleeType,
                      mlir::ArrayRef<mlir::Type> positional,
                      mlir::ArrayRef<CallKeywordType> keywords) const {
  CallInferenceResult inference =
      inferCallWithEvidence(calleeType, positional, keywords);
  return inference ? inference.resultType : object();
}

mlir::Type AlgorithmM::inferClassInstantiation(
    mlir::Type instanceType, mlir::ArrayRef<mlir::Type> positional,
    mlir::ArrayRef<CallKeywordType> keywords) const {
  // Type parameters the constructor leaves unbound fall back to their
  // manifest defaults (`ly.typing.param_defaults`, PEP 696 semantics) --
  // e.g. instantiating a bare CFuncPtr leaves its result parameter at Any.
  auto applyParamDefaults = [&](mlir::Type type) -> mlir::Type {
    auto contract = mlir::dyn_cast_if_present<py::ContractType>(type);
    if (!contract || unboundStaticParameterCount(type) == 0)
      return type;
    const py::protocols::Table &table = py::protocols::Table::get(context);
    const py::protocols::ProtocolInfo *info =
        table.lookup(manifestNameForContract(contract.getContractName()));
    if (!info)
      return type;
    llvm::SmallVector<mlir::Type, 4> arguments(
        contract.getArguments().begin(), contract.getArguments().end());
    for (auto [index, argument] : llvm::enumerate(arguments)) {
      if (unboundStaticParameterCount(argument) == 0)
        continue;
      if (index < info->paramDefaults.size() && info->paramDefaults[index])
        argument = info->paramDefaults[index];
    }
    return py::ContractType::get(&context, contract.getContractName(),
                                 arguments);
  };
  auto complete = [&](mlir::Type type) -> mlir::Type {
    type = applyParamDefaults(type);
    if (!type || unboundStaticParameterCount(type) != 0)
      return {};
    return type;
  };
  mlir::Type templated = genericClassTemplate(*this, instanceType);
  if (std::optional<CallSolution> init =
          tryManifestMethod(*this, templated, "__init__", positional, keywords))
    return complete(substituteType(*this, templated, init->bindings,
                                   /*eraseUnbound=*/false));
  if (templated != instanceType)
    return complete(substituteType(*this, templated, TypeBindingMap{},
                                   /*eraseUnbound=*/false));
  return complete(instanceType);
}

CallInferenceResult AlgorithmM::inferMethodCallWithEvidence(
    mlir::Type receiverType, llvm::StringRef methodName,
    mlir::ArrayRef<mlir::Type> positional,
    mlir::ArrayRef<CallKeywordType> keywords) const {
  if (std::optional<CallSolution> result = tryManifestMethod(
          *this, receiverType, methodName, positional, keywords)) {
    return CallInferenceResult{
        result->result,
        CallInferenceEvidence{result->callableContract, result->methodName,
                              result->receiverManifestClass},
        true,
        {}};
  }
  return unresolvedMethodCall(*this, receiverType, methodName);
}

bool AlgorithmM::isStructuralMutatorMethod(mlir::Type receiverType,
                                           llvm::StringRef methodName) const {
  if (!receiverType)
    return false;
  const py::protocols::Table &table = py::protocols::Table::get(context);
  return table.isStructuralMutator(widenLiteral(receiverType), methodName);
}

std::optional<std::vector<std::string>>
AlgorithmM::classMatchArgs(mlir::Type receiverType) const {
  if (!receiverType)
    return std::nullopt;
  const py::protocols::Table &table = py::protocols::Table::get(context);
  return table.matchArgsFor(widenLiteral(receiverType));
}

AwaitInferenceResult
AlgorithmM::inferAwaitWithEvidence(mlir::Type awaitableType) const {
  mlir::Type awaitable = widenLiteral(awaitableType);
  const py::protocols::Table &table = py::protocols::Table::get(context);
  std::optional<py::protocols::AwaitableResolution> resolution =
      table.resolveAwaitableWithEvidence(awaitable);
  if (!resolution) {
    return AwaitInferenceResult{
        {},
        {},
        false,
        "await expression requires an Awaitable value, got " +
            typeText(awaitable)};
  }

  mlir::Type awaitContract = protocol("Callable");
  if (resolution->awaitContract)
    awaitContract = resolution->awaitContract->method.signature;
  return AwaitInferenceResult{resolution->payloadType, awaitContract, true, {}};
}

YieldFromInferenceResult
AlgorithmM::inferYieldFromWithEvidence(mlir::Type sourceType) const {
  mlir::Type source = widenLiteral(sourceType);
  const py::protocols::Table &table = py::protocols::Table::get(context);

  auto protocolResult = [&](llvm::StringRef name,
                            std::vector<mlir::Type> arguments)
      -> YieldFromInferenceResult {
    if (arguments.empty())
      return YieldFromInferenceResult{
          {}, {}, {}, false,
          std::string("yield from ") + typeText(source) + " has no " +
              name.str() +
              " element type evidence"};
    return YieldFromInferenceResult{
        arguments.front(), none(), protocol(name, arguments), true, {}};
  };

  if (std::optional<std::vector<mlir::Type>> generator =
          table.protocolArgumentsFor(source, "Generator"))
    return protocolResult("Generator", *generator);
  if (std::optional<std::vector<mlir::Type>> iterator =
          table.protocolArgumentsFor(source, "Iterator"))
    return protocolResult("Iterator", *iterator);
  if (std::optional<std::vector<mlir::Type>> iterable =
          table.protocolArgumentsFor(source, "Iterable"))
    return protocolResult("Iterable", *iterable);

  return YieldFromInferenceResult{
      {}, {}, {}, false,
      std::string(
          "yield from requires a Generator, Iterator, or Iterable value, got ") +
          typeText(source)};
}

AsyncIterationInferenceResult
AlgorithmM::inferAsyncIterationWithEvidence(mlir::Type iterableType) const {
  mlir::Type iterable = widenLiteral(iterableType);
  AsyncIterationInferenceResult result;
  result.aiter = inferMethodCallWithEvidence(iterable, "__aiter__", {});
  if (!result.aiter) {
    result.failureReason = result.aiter.failureReason;
    return result;
  }

  result.iteratorType = widenLiteral(result.aiter.resultType);
  const py::protocols::Table &table = py::protocols::Table::get(context);
  std::optional<std::vector<mlir::Type>> iteratorArgs =
      table.protocolArgumentsFor(result.iteratorType, "AsyncIterator");
  if (!iteratorArgs || iteratorArgs->size() != 1) {
    result.failureReason =
        "__aiter__ must return an AsyncIterator value, got " +
        typeText(result.iteratorType);
    return result;
  }

  result.anext =
      inferMethodCallWithEvidence(result.iteratorType, "__anext__", {});
  if (!result.anext) {
    result.failureReason = result.anext.failureReason;
    return result;
  }

  result.nextAwaitableType = widenLiteral(result.anext.resultType);
  result.awaitNext = inferAwaitWithEvidence(result.nextAwaitableType);
  if (!result.awaitNext) {
    result.failureReason = "__anext__ must return an Awaitable value: " +
                           result.awaitNext.failureReason;
    return result;
  }

  result.itemType = result.awaitNext.resultType;
  result.resolved = true;
  return result;
}

static AsyncContextMethodInferenceResult
inferAsyncContextMethod(const AlgorithmM &types, mlir::Type managerType,
                        llvm::StringRef methodName,
                        mlir::ArrayRef<mlir::Type> positional) {
  AsyncContextMethodInferenceResult result;
  result.method =
      types.inferMethodCallWithEvidence(managerType, methodName, positional);
  if (!result.method) {
    result.failureReason = result.method.failureReason;
    return result;
  }

  result.awaitableType = types.widenLiteral(result.method.resultType);
  result.awaitResult = types.inferAwaitWithEvidence(result.awaitableType);
  if (!result.awaitResult) {
    result.failureReason =
        methodName.str() +
        " must return an Awaitable value: " + result.awaitResult.failureReason;
    return result;
  }

  result.resultType = result.awaitResult.resultType;
  result.resolved = true;
  return result;
}

AsyncContextMethodInferenceResult
AlgorithmM::inferAsyncContextEnterWithEvidence(mlir::Type managerType) const {
  return inferAsyncContextMethod(*this, managerType, "__aenter__", {});
}

AsyncContextMethodInferenceResult AlgorithmM::inferAsyncContextExitWithEvidence(
    mlir::Type managerType, mlir::ArrayRef<mlir::Type> exceptionTypes) const {
  return inferAsyncContextMethod(*this, managerType, "__aexit__",
                                 exceptionTypes);
}

CallInferenceResult AlgorithmM::inferCallWithEvidence(
    mlir::Type calleeType, mlir::ArrayRef<mlir::Type> positional,
    mlir::ArrayRef<CallKeywordType> keywords) const {
  if (!calleeType)
    return unresolvedCallable(calleeType, "missing callee type");
  if (auto typeType = mlir::dyn_cast_if_present<py::TypeType>(calleeType)) {
    mlir::Type instance =
        inferClassInstantiation(typeType.getInstanceType(), positional, keywords);
    if (instance) {
      // Synthesize the applied Callable evidence (supplied argument types ->
      // instance): a bare Callable protocol is not stable call evidence.
      llvm::SmallVector<mlir::Type, 4> suppliedTypes(positional.begin(),
                                                     positional.end());
      llvm::SmallVector<mlir::Type, 1> resultTypes{instance};
      mlir::Type applied = py::CallableType::get(&context, suppliedTypes, {},
                                                 {}, {}, resultTypes);
      return CallInferenceResult{
          instance,
          CallInferenceEvidence{applied, "__call__", std::nullopt},
          true,
          {}};
    }
    return unresolvedCallable(
        calleeType,
        "class instantiation leaves unbound static type parameters");
  }
  if (auto callable = mlir::dyn_cast_if_present<py::CallableType>(calleeType)) {
    if (std::optional<CallSolution> result =
            tryCallableApplication(*this, callable, positional, keywords))
      return CallInferenceResult{result->result,
                                 CallInferenceEvidence{result->callableContract,
                                                       "__call__",
                                                       std::nullopt},
                                 true,
                                 {}};
    return unresolvedCallable(
        calleeType, "call arguments do not match the Callable contract");
  }
  if (auto overload = mlir::dyn_cast_if_present<py::OverloadType>(calleeType)) {
    llvm::SmallVector<py::CallableType, 4> callables;
    for (mlir::Type candidate : overload.getCandidateTypes()) {
      if (auto callable =
              mlir::dyn_cast_if_present<py::CallableType>(candidate)) {
        callables.push_back(callable);
        continue;
      }
      if (auto typeType = mlir::dyn_cast_if_present<py::TypeType>(candidate))
        if (positional.empty() && keywords.empty())
          return CallInferenceResult{typeType.getInstanceType(),
                                     CallInferenceEvidence{protocol("Callable"),
                                                           "__call__",
                                                           std::nullopt},
                                     true,
                                     {}};
    }
    if (std::optional<CallSolution> selected =
            selectCallableApplication(*this, callables, positional, keywords))
      return CallInferenceResult{
          selected->result,
          CallInferenceEvidence{selected->callableContract, "__call__",
                                std::nullopt},
          true,
          {}};
  }
  if (std::optional<CallSolution> result = tryManifestMethod(
          *this, calleeType, "__call__", positional, keywords))
    return CallInferenceResult{
        result->result,
        CallInferenceEvidence{result->callableContract, result->methodName,
                              result->receiverManifestClass},
        true,
        {}};
  return unresolvedCallable(calleeType, "no manifest __call__ contract");
}

std::optional<mlir::Type>
AlgorithmM::fieldAssignmentRefinement(mlir::Type receiverType,
                                      llvm::StringRef fieldName,
                                      mlir::Type valueType) const {
  const py::protocols::Table &table = py::protocols::Table::get(context);
  return table.refineContractByFieldAssignment(widenLiteral(receiverType),
                                               fieldName, valueType);
}

mlir::Type AlgorithmM::join(mlir::ArrayRef<mlir::Type> types) const {
  llvm::SmallVector<mlir::Type, 4> present;
  llvm::SmallVector<mlir::Type, 8> worklist(types.begin(), types.end());
  while (!worklist.empty()) {
    mlir::Type type = worklist.pop_back_val();
    if (!type)
      continue;
    if (isObjectTop(*this, type)) {
      present.clear();
      present.push_back(object());
      continue;
    }
    if (isObjectTop(*this, present.empty() ? mlir::Type{} : present.front()))
      continue;
    if (auto unionType = mlir::dyn_cast_if_present<py::UnionType>(type)) {
      for (mlir::Type member : unionType.getMemberTypes())
        worklist.push_back(member);
      continue;
    }
    if (!llvm::is_contained(present, type))
      present.push_back(type);
  }
  if (present.empty())
    return object();
  if (present.size() == 1)
    return present.front();
  return py::UnionType::getNormalized(&context, present);
}

mlir::Type AlgorithmM::widenLiteral(mlir::Type type) const {
  auto literalType = mlir::dyn_cast_or_null<py::LiteralType>(type);
  if (!literalType)
    return type ? type : object();
  llvm::StringRef spelling = literalType.getSpelling();
  if (spelling == "True" || spelling == "False")
    return boolType();
  if (spelling == "None")
    return none();
  if (!spelling.empty() && spelling.front() == '"')
    return strType();
  return intType();
}

FunctionSignature
AlgorithmM::functionSignature(const parser::Node &function,
                              std::optional<llvm::StringRef> selfName,
                              py::CallableType expectedCallable) const {
  FunctionSignature sig;
  sig.isAsyncFunction = function.kind == "AsyncFunctionDef";
  auto typeParamScope = pushScope();
  if (const auto *typeParams = ast::nodeList(function, "type_params")) {
    for (const parser::NodePtr &param : *typeParams) {
      auto name = ast::string(*param, "name");
      if (!name)
        continue;
      llvm::StringRef paramName(*name);
      if (param->kind == "ParamSpec") {
        bindLocalSymbol(paramName, py::ParamSpecType::get(&context, paramName));
      } else if (param->kind == "TypeVarTuple") {
        bindLocalSymbol(paramName,
                        py::TypeVarTupleType::get(&context, paramName));
      } else {
        bindLocalSymbol(paramName, py::TypeVarType::get(&context, paramName));
      }
    }
  }
  const parser::Node *arguments = ast::node(function, "args");
  if (arguments) {
    auto recordAnnotationIssue = [&](const parser::Node *annotation,
                                     llvm::StringRef parameterName) {
      if (!annotation) {
        sig.missingParameterAnnotations.push_back(parameterName.str());
        return;
      }
      if (annotation->kind != "Name" && annotation->kind != "Attribute")
        return;
      std::string qualified = ast::qualifiedName(annotation);
      std::string_view spelling = ast::nameSpelling(*annotation);
      std::string resolved = resolveAnnotationName(
          qualified.empty() ? llvm::StringRef(spelling.data(), spelling.size())
                            : llvm::StringRef(qualified));
      if (std::optional<std::string> generic =
              bareGenericAnnotationName(resolved))
        sig.invalidParameterAnnotations.push_back(
            "generic annotation '" + *generic +
            "' requires explicit type arguments for parameter '" +
            parameterName.str() + "'");
    };

    unsigned positionalOnlyCount = 0;
    llvm::SmallVector<parser::NodePtr, 8> positional =
        concatArgs(*arguments, positionalOnlyCount);
    sig.positionalOnlyCount = positionalOnlyCount;
    std::size_t defaults = ast::nodeList(*arguments, "defaults")
                               ? ast::nodeList(*arguments, "defaults")->size()
                               : 0;
    llvm::ArrayRef<mlir::Type> expectedPositional =
        expectedCallable ? expectedCallable.getPositionalTypes()
                         : llvm::ArrayRef<mlir::Type>();
    llvm::ArrayRef<mlir::Type> expectedKwOnly =
        expectedCallable ? expectedCallable.getKwOnlyTypes()
                         : llvm::ArrayRef<mlir::Type>();
    for (auto [index, arg] : llvm::enumerate(positional)) {
      std::string name(ast::nameSpelling(*arg));
      const parser::Node *annotation = ast::node(*arg, "annotation");
      mlir::Type type = annotationType(annotation);
      bool isSelfParameter = selfName && index == 0 && name == *selfName;
      bool fromExpectedCallable =
          function.kind == "Lambda" && index < expectedPositional.size();
      if (isSelfParameter)
        type = py::SelfType::get(&context);
      if (fromExpectedCallable)
        type = expectedPositional[index];
      if (!isSelfParameter && !fromExpectedCallable)
        recordAnnotationIssue(annotation, name);
      sig.positionalNames.push_back(std::move(name));
      sig.positionalTypes.push_back(type);
      sig.positionalDefaults.push_back(
          hasDefault(index, positional.size(), defaults));
    }

    if (const auto *kwonly = ast::nodeList(*arguments, "kwonlyargs")) {
      std::size_t index = 0;
      for (const parser::NodePtr &arg : *kwonly) {
        std::string name(ast::nameSpelling(*arg));
        const parser::Node *annotation = ast::node(*arg, "annotation");
        sig.kwOnlyNames.push_back(name);
        mlir::Type type = annotationType(annotation);
        bool fromExpectedCallable =
            function.kind == "Lambda" && index < expectedKwOnly.size();
        if (fromExpectedCallable)
          type = expectedKwOnly[index];
        if (!fromExpectedCallable)
          recordAnnotationIssue(annotation, name);
        sig.kwOnlyTypes.push_back(type);
        bool hasKwDefault = false;
        if (const auto *kwDefaults = ast::nodeList(*arguments, "kw_defaults"))
          hasKwDefault = index < kwDefaults->size() && (*kwDefaults)[index];
        sig.kwOnlyDefaults.push_back(hasKwDefault);
        ++index;
      }
    }
    if (const parser::Node *vararg = ast::node(*arguments, "vararg")) {
      sig.varargName = std::string(ast::nameSpelling(*vararg));
      const parser::Node *annotationNode = ast::node(*vararg, "annotation");
      recordAnnotationIssue(annotationNode, "*" + *sig.varargName);
      mlir::Type annotation = annotationType(annotationNode);
      sig.varargType = tupleOf(annotation);
      sig.callableVarargType =
          mlir::isa<py::UnpackType>(annotation) ? annotation : sig.varargType;
    }
    if (const parser::Node *kwarg = ast::node(*arguments, "kwarg")) {
      sig.kwargName = std::string(ast::nameSpelling(*kwarg));
      const parser::Node *annotationNode = ast::node(*kwarg, "annotation");
      recordAnnotationIssue(annotationNode, "**" + *sig.kwargName);
      sig.kwargType =
          dictOf(strType(), annotationType(annotationNode));
    }
  }

  auto scope = pushScope();
  for (auto [index, name] : llvm::enumerate(sig.positionalNames))
    bindLocalSymbol(name, sig.positionalTypes[index]);
  for (auto [index, name] : llvm::enumerate(sig.kwOnlyNames))
    bindLocalSymbol(name, sig.kwOnlyTypes[index]);
  if (sig.varargName)
    bindLocalSymbol(*sig.varargName, sig.varargType);
  if (sig.kwargName)
    bindLocalSymbol(*sig.kwargName, sig.kwargType);

  const parser::Node *returns = ast::node(function, "returns");
  mlir::Type annotatedReturn = returns ? annotationType(returns) : mlir::Type();
  std::optional<mlir::Type> annotatedGeneratorSendType;
  if (annotatedReturn)
    annotatedGeneratorSendType = generatorSendTypeFromAnnotation(
        *this, annotatedReturn,
        function.kind == "AsyncFunctionDef" ? "AsyncGenerator" : "Generator");

  GeneratorFunctionAnalysis generator = analyzeGeneratorFunction(
      *this, function, annotatedGeneratorSendType.value_or(mlir::Type()));
  if (generator.hasYield && function.kind != "Lambda") {
    sig.generatorAnalysisFailures.append(generator.failureReasons.begin(),
                                         generator.failureReasons.end());
    sig.generatorYieldType =
        generator.yieldTypes.empty() ? none() : join(generator.yieldTypes);
    sig.generatorSendType = annotatedGeneratorSendType.value_or(none());
    sig.generatorReturnType =
        generator.returnTypes.empty() ? none() : join(generator.returnTypes);
    if (function.kind == "AsyncFunctionDef") {
      sig.isAsyncGeneratorFunction = true;
      sig.asyncGeneratorReturnsValue = generator.hasReturnValue;
      sig.inferredGeneratorType = protocol(
          "AsyncGenerator", {sig.generatorYieldType, sig.generatorSendType});
    } else {
      sig.isGeneratorFunction = true;
      sig.inferredGeneratorType = contract(
          "types.GeneratorType", {sig.generatorYieldType, sig.generatorSendType,
                                  sig.generatorReturnType});
    }

    if (returns) {
      sig.generatorAnnotationIncompatible =
          !py::isAssignableTo(sig.inferredGeneratorType, annotatedReturn);
    }
    sig.resultType = sig.generatorReturnType;
  } else if (function.kind == "Lambda") {
    sig.resultType = inferExpr(ast::node(function, "body"));
  } else if (returns) {
    sig.resultType = annotationType(returns);
  } else {
    sig.resultType = ast::nameSpelling(function) == "__init__"
                         ? none()
                         : inferredFunctionResult(*this, function,
                                                  &sig.bodyInferenceFailures);
  }
  refreshCallable(sig);
  return sig;
}

void AlgorithmM::refreshCallable(FunctionSignature &sig) const {
  llvm::SmallVector<mlir::StringAttr, 8> posNames;
  llvm::SmallVector<mlir::StringAttr, 4> kwNames;
  llvm::SmallVector<mlir::BoolAttr, 8> posDefaults;
  llvm::SmallVector<mlir::BoolAttr, 4> kwDefaults;
  for (const std::string &name : sig.positionalNames)
    posNames.push_back(mlir::StringAttr::get(&context, name));
  for (const std::string &name : sig.kwOnlyNames)
    kwNames.push_back(mlir::StringAttr::get(&context, name));
  for (bool value : sig.positionalDefaults)
    posDefaults.push_back(mlir::BoolAttr::get(&context, value));
  for (bool value : sig.kwOnlyDefaults)
    kwDefaults.push_back(mlir::BoolAttr::get(&context, value));

  mlir::Type callableVararg =
      sig.callableVarargType ? sig.callableVarargType : sig.varargType;
  auto makeCallable = [&](mlir::Type resultType) {
    llvm::SmallVector<mlir::Type, 1> results{resultType};
    return py::CallableType::get(
        &context, sig.positionalTypes, sig.kwOnlyTypes, callableVararg,
        sig.kwargType, results, posNames, kwNames, posDefaults, kwDefaults,
        sig.varargName ? mlir::StringAttr::get(&context, *sig.varargName)
                       : mlir::StringAttr(),
        sig.kwargName ? mlir::StringAttr::get(&context, *sig.kwargName)
                      : mlir::StringAttr(),
        sig.positionalOnlyCount);
  };

  sig.callable = makeCallable(sig.resultType);
  if (sig.isAsyncFunction && !sig.isAsyncGeneratorFunction)
    sig.publicResultType = coroutineOf(sig.resultType);
  else if (sig.isGeneratorFunction || sig.isAsyncGeneratorFunction)
    sig.publicResultType = sig.inferredGeneratorType;
  else
    sig.publicResultType = sig.resultType;
  sig.publicCallable = makeCallable(sig.publicResultType);
}

} // namespace lython::emitter
