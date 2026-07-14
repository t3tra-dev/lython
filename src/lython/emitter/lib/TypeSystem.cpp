#include "TypeSystem.h"

#include "TypeSystemSolver.h"

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

CallInferenceResult unresolvedMethodCall(const TypeSystem &types,
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

py::CallableType makeZeroArgStrCallable(const TypeSystem &types) {
  mlir::MLIRContext *context = &types.getContext();
  llvm::SmallVector<mlir::Type, 1> results{types.strType()};
  return py::CallableType::get(context, {}, {}, {}, {}, results);
}

mlir::Type inferAsyncioSleepResult(const TypeSystem &types,
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

bool appendStarredCallArgumentTypes(const TypeSystem &types, mlir::Type type,
                                    llvm::SmallVectorImpl<mlir::Type> &out);

void recordInferenceFailure(
    llvm::SmallVectorImpl<std::string> *failureReasons, std::string reason) {
  if (failureReasons && !reason.empty())
    failureReasons->push_back(std::move(reason));
}

mlir::Type inferExprWithLocalCallables(
    const TypeSystem &types, const parser::Node *node,
    const llvm::StringMap<mlir::Type> &localCallables,
    llvm::SmallVectorImpl<std::string> *failureReasons = nullptr,
    const llvm::StringMap<mlir::Type> *localSymbols = nullptr) {
  return types.inferExpr(node, ExprInferenceContext{localCallables,
                                                    failureReasons,
                                                    localSymbols});
}

mlir::Type inferReturnExpr(const TypeSystem &types, const parser::Node *node,
                           const llvm::StringMap<mlir::Type> &localCallables,
                           llvm::SmallVectorImpl<std::string> *failureReasons,
                           const llvm::StringMap<mlir::Type> *localSymbols =
                               nullptr) {
  return inferExprWithLocalCallables(types, node, localCallables,
                                     failureReasons, localSymbols);
}

llvm::StringMap<mlir::Type>
localCallableTypesInFunction(const TypeSystem &types,
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

void collectReturnTypes(const TypeSystem &types, const parser::Node *node,
                        const llvm::StringMap<mlir::Type> &localCallables,
                        llvm::SmallVectorImpl<mlir::Type> &results,
                        llvm::SmallVectorImpl<std::string> *failureReasons,
                        const llvm::StringMap<mlir::Type> *localSymbols) {
  if (!node)
    return;
  if (node->kind == "FunctionDef" || node->kind == "AsyncFunctionDef" ||
      node->kind == "Lambda" || node->kind == "ClassDef")
    return;
  if (node->kind == "Return") {
    mlir::Type type =
        inferReturnExpr(types, ast::node(*node, "value"), localCallables,
                        failureReasons, localSymbols);
    if (type)
      results.push_back(type);
    return;
  }
  for (const parser::Field &field : node->fields) {
    if (const auto *child = std::get_if<parser::NodePtr>(&field.value)) {
      if (*child)
        collectReturnTypes(types, child->get(), localCallables, results,
                           failureReasons, localSymbols);
    } else if (const auto *children =
                   std::get_if<std::vector<parser::NodePtr>>(&field.value)) {
      for (const parser::NodePtr &child : *children)
        collectReturnTypes(types, child.get(), localCallables, results,
                           failureReasons, localSymbols);
    }
  }
}

mlir::Type inferredFunctionResult(const TypeSystem &types,
                                  const parser::Node &function,
                                  llvm::SmallVectorImpl<std::string>
                                      *failureReasons = nullptr,
                                  const llvm::StringMap<mlir::Type>
                                      *localSymbols = nullptr) {
  llvm::StringMap<mlir::Type> localCallables =
      localCallableTypesInFunction(types, function);

  llvm::SmallVector<mlir::Type, 4> results;
  if (const auto *body = ast::nodeList(function, "body"))
    for (const parser::NodePtr &statement : *body)
      collectReturnTypes(types, statement.get(), localCallables, results,
                         failureReasons, localSymbols);
  return results.empty() ? types.none() : types.join(results);
}

struct GeneratorFunctionAnalysis {
  bool hasYield = false;
  // Locals bound by the walk in statement order; replaces the former
  // bindLocalSymbol side effect on the shared symbol-table scope, and is
  // reused by the return-type inference that runs after the walk.
  llvm::StringMap<mlir::Type> localSymbols;
  bool sawYieldFrom = false;
  bool hasReturnValue = false;
  llvm::SmallVector<mlir::Type, 4> yieldTypes;
  llvm::SmallVector<mlir::Type, 4> returnTypes;
  llvm::SmallVector<std::string, 4> failureReasons;
};

std::optional<mlir::Type> generatorYieldFromElementType(
    const TypeSystem &types, const parser::Node *value,
    const llvm::StringMap<mlir::Type> &localCallables,
    llvm::SmallVectorImpl<std::string> &failureReasons,
    const llvm::StringMap<mlir::Type> *localSymbols) {
  mlir::Type rawSource = inferExprWithLocalCallables(
      types, value, localCallables, &failureReasons, localSymbols);
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

const llvm::StringMap<mlir::Type> &noLocalCallables() {
  static const llvm::StringMap<mlir::Type> empty;
  return empty;
}

// Lenient inference during the body walk: object() fallbacks, but reads the
// locals bound so far.
mlir::Type lenientWalkInfer(const TypeSystem &types, const parser::Node *node,
                            const GeneratorFunctionAnalysis &analysis) {
  return types.inferExpr(
      node, ExprInferenceContext{noLocalCallables(), nullptr,
                                 &analysis.localSymbols, /*strict=*/false});
}

void collectGeneratorFunctionAnalysis(
    const TypeSystem &types, const parser::Node *node,
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
    mlir::Type valueType =
        inferExprWithLocalCallables(types, value, localCallables,
                                    &analysis.failureReasons,
                                    &analysis.localSymbols);
    if (valueType)
      analysis.yieldTypes.push_back(types.widenLiteral(valueType));
    return;
  }
  if (node->kind == "YieldFrom") {
    analysis.hasYield = true;
    analysis.sawYieldFrom = true;
    if (std::optional<mlir::Type> element = generatorYieldFromElementType(
            types, ast::node(*node, "value"), localCallables,
            analysis.failureReasons, &analysis.localSymbols))
      analysis.yieldTypes.push_back(*element);
    return;
  }
  if (node->kind == "Return") {
    const parser::Node *value = ast::node(*node, "value");
    if (value)
      analysis.hasReturnValue = true;
    analysis.returnTypes.push_back(
        value ? inferReturnExpr(types, value, localCallables,
                                &analysis.failureReasons,
                                &analysis.localSymbols)
              : types.none());
    return;
  }
  if (node->kind == "Assign") {
    const parser::Node *value = ast::node(*node, "value");
    collectGeneratorFunctionAnalysis(types, value, localCallables,
                                     generatorSendHint, analysis);
    mlir::Type valueType = value && value->kind == "Yield" && generatorSendHint
                               ? generatorSendHint
                               : lenientWalkInfer(types, value, analysis);
    if (const auto *targets = ast::nodeList(*node, "targets")) {
      for (const parser::NodePtr &target : *targets) {
        if (target && target->kind == "Name")
          analysis.localSymbols[ast::nameSpelling(*target)] =
              valueType ? valueType : types.object();
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
        type = lenientWalkInfer(types, value, analysis);
      if (type)
        analysis.localSymbols[ast::nameSpelling(*target)] = type;
    }
    return;
  }
  if (node->kind == "AugAssign") {
    const parser::Node *target = ast::node(*node, "target");
    const parser::Node *value = ast::node(*node, "value");
    collectGeneratorFunctionAnalysis(types, value, localCallables,
                                     generatorSendHint, analysis);
    if (target && target->kind == "Name") {
      mlir::Type lhs = types.widenLiteral(lenientWalkInfer(types, target, analysis));
      mlir::Type rhs = types.widenLiteral(lenientWalkInfer(types, value, analysis));
      mlir::Type joined = types.widenLiteral(types.join({lhs, rhs}));
      analysis.localSymbols[ast::nameSpelling(*target)] =
          joined ? joined : types.object();
    }
    return;
  }
  if (node->kind == "For" || node->kind == "AsyncFor") {
    // Bind the loop target to the iteration element type before the generic
    // child walk reaches the body, so yields over the target infer correctly.
    const parser::Node *target = ast::node(*node, "target");
    const parser::Node *iter = ast::node(*node, "iter");
    if (target && target->kind == "Name" && iter) {
      mlir::Type iterableType =
          inferExprWithLocalCallables(types, iter, localCallables,
                                      &analysis.failureReasons,
                                      &analysis.localSymbols);
      if (iterableType) {
        CallInferenceResult iterInference = types.inferMethodCallWithEvidence(
            types.widenLiteral(iterableType), "__iter__", {});
        CallInferenceResult nextInference =
            iterInference ? types.inferMethodCallWithEvidence(
                                iterInference.resultType, "__next__", {})
                          : CallInferenceResult{};
        if (nextInference) {
          mlir::Type element = types.widenLiteral(nextInference.resultType);
          analysis.localSymbols[ast::nameSpelling(*target)] =
              element ? element : types.object();
        }
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
analyzeGeneratorFunction(const TypeSystem &types, const parser::Node &function,
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
generatorSendTypeFromAnnotation(const TypeSystem &types, mlir::Type annotation,
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
mlir::Type tupleOfMembers(const TypeSystem &types,
                          llvm::ArrayRef<mlir::Type> members) {
  if (members.empty())
    return types.tupleOf(types.object());
  bool uniform = llvm::all_of(
      members, [&](mlir::Type type) { return type == members.front(); });
  if (uniform)
    return types.tupleOf(members.front());
  return types.contract("builtins.tuple", members);
}

bool appendStarredCallArgumentTypes(const TypeSystem &types, mlir::Type type,
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

bool bindManifestClassImport(TypeSystem &types, llvm::StringRef localName,
                             llvm::StringRef contractName) {
  const py::protocols::Table &table =
      py::protocols::Table::get(types.getContext());
  if (!table.lookup(manifestNameForContract(contractName)))
    return false;
  types.bindClass(localName, types.contract(contractName));
  return true;
}

mlir::Type genericClassTemplate(const TypeSystem &types,
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

mlir::Type importCallableType(const TypeSystem &types,
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

struct ModuleIntConstantImport {
  const char *module;
  const char *localAttr;
  const char *canonicalName;
};

struct NameIntConstantImport {
  const char *module;
  const char *exportedName;
  const char *canonicalName;
};

// Module attributes that bind to a RUNTIME value (materialized by a lowering
// hook on the canonical binding), not a folded constant: sys.argv.
struct ModuleStrListImport {
  const char *module;
  const char *localAttr;
  const char *canonicalName;
};

struct NameStrListImport {
  const char *module;
  const char *exportedName;
  const char *canonicalName;
};

// Runtime module attributes typed by a manifest contract: sys.stdout/stderr.
struct ModuleContractValueImport {
  const char *module;
  const char *localAttr;
  const char *canonicalName;
  const char *contract;
};

struct NameContractValueImport {
  const char *module;
  const char *exportedName;
  const char *canonicalName;
  const char *contract;
};

constexpr ModuleCallableImport kModuleCallableImports[] = {
    // Manifest-declared callables (ly.typing.callable_exports +
    // ly.typing.function_contracts, e.g. asyncio.*, os.getpid, ctypes.*)
    // bind through bindManifestModuleCallableExports -- ONLY names without a
    // manifest contract belong here (C++ factory-typed).
    {"platform", "system", "platform.system",
     ImportCallableFactory::StaticZeroArgStr},
    {"sys", "getdefaultencoding", "sys.getdefaultencoding",
     ImportCallableFactory::StaticZeroArgStr},
    {"sys", "getfilesystemencoding", "sys.getfilesystemencoding",
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
    {"sys", "getdefaultencoding", "sys.getdefaultencoding",
     ImportCallableFactory::StaticZeroArgStr},
    {"sys", "getfilesystemencoding", "sys.getfilesystemencoding",
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
    {"sys", "byteorder", "sys.byteorder"},
    {"os", "name", "os.name"},
};

constexpr NameStringConstantImport kNameStringConstantImports[] = {
    {"sys", "platform", "sys.platform"},
    {"sys", "byteorder", "sys.byteorder"},
    {"os", "name", "os.name"},
};

constexpr ModuleIntConstantImport kModuleIntConstantImports[] = {
    {"sys", "maxsize", "sys.maxsize"},
};

constexpr NameIntConstantImport kNameIntConstantImports[] = {
    {"sys", "maxsize", "sys.maxsize"},
};

constexpr ModuleStrListImport kModuleStrListImports[] = {
    {"sys", "argv", "sys.argv"},
};

constexpr NameStrListImport kNameStrListImports[] = {
    {"sys", "argv", "sys.argv"},
};

constexpr ModuleContractValueImport kModuleContractValueImports[] = {
    {"sys", "stdout", "sys.stdout", "_io.TextIOWrapper"},
    {"sys", "stderr", "sys.stderr", "_io.TextIOWrapper"},
};

constexpr NameContractValueImport kNameContractValueImports[] = {
    {"sys", "stdout", "sys.stdout", "_io.TextIOWrapper"},
    {"sys", "stderr", "sys.stderr", "_io.TextIOWrapper"},
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

bool bindManifestModuleObject(TypeSystem &types, llvm::StringRef module,
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

bool bindManifestModuleClassExports(TypeSystem &types, llvm::StringRef module,
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

bool bindManifestModuleFloatConstants(TypeSystem &types,
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

bool bindManifestModuleIntConstants(TypeSystem &types, llvm::StringRef module,
                                    llvm::StringRef localName) {
  const py::protocols::Table &table =
      py::protocols::Table::get(types.getContext());
  bool handled = false;
  for (const std::string &exportedName :
       table.moduleIntConstantExports(module)) {
    if (!handled)
      bindManifestModuleObject(types, module, localName);
    handled = true;
    std::string canonical = (llvm::Twine(module) + "." + exportedName).str();
    types.bindCanonicalSymbol(
        importedManifestModuleAttribute(module, localName, exportedName),
        canonical, types.intType());
  }
  return handled;
}

bool bindManifestModuleStrConstants(TypeSystem &types, llvm::StringRef module,
                                    llvm::StringRef localName) {
  const py::protocols::Table &table =
      py::protocols::Table::get(types.getContext());
  bool handled = false;
  for (const std::string &exportedName :
       table.moduleStrConstantExports(module)) {
    if (!handled)
      bindManifestModuleObject(types, module, localName);
    handled = true;
    std::string canonical = (llvm::Twine(module) + "." + exportedName).str();
    types.bindCanonicalSymbol(
        importedManifestModuleAttribute(module, localName, exportedName),
        canonical, types.strType());
  }
  return handled;
}

bool bindManifestModuleCallableExports(TypeSystem &types,
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

} // namespace

TypeSystem::TypeSystem(mlir::MLIRContext &context)
    : context(context), inferenceState(context) {}

TypeSystem::Scope::Scope(Scope &&other) noexcept : owner(other.owner) {
  other.owner = nullptr;
}

TypeSystem::Scope &TypeSystem::Scope::operator=(Scope &&other) noexcept {
  if (this == &other)
    return *this;
  reset();
  owner = other.owner;
  other.owner = nullptr;
  return *this;
}

TypeSystem::Scope::~Scope() { reset(); }

void TypeSystem::Scope::reset() {
  if (!owner)
    return;
  owner->popScope();
  owner = nullptr;
}

void TypeSystem::seedBuiltins() {
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
  bindClass("bytes", contract("builtins.bytes"));
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
  bindClass("SystemExit", contract("builtins.SystemExit"));
  bindClass("OSError", contract("builtins.OSError"));
  bindClass("FileNotFoundError", contract("builtins.FileNotFoundError"));
  // open is io.open (CPython aliases the builtin to the io module's opener);
  // the contract and the runtime implementation live in the _io manifest.
  bindCanonicalSymbol(
      "open", "_io.open",
      table.freeFunctionContract("_io.open")
          .value_or(contract("builtins.function")));
  bindClass("nullcontext", contract("contextlib.nullcontext"));
  bindClass("range", contract("builtins.range"));
}

mlir::Type TypeSystem::object() const { return contract("builtins.object"); }
mlir::Type TypeSystem::any() const { return contract("typing.Any"); }
mlir::Type TypeSystem::none() const { return literal("None"); }
mlir::Type TypeSystem::boolType() const { return contract("builtins.bool"); }
mlir::Type TypeSystem::intType() const { return contract("builtins.int"); }
mlir::Type TypeSystem::strType() const { return contract("builtins.str"); }
mlir::Type TypeSystem::floatType() const { return contract("builtins.float"); }

mlir::Type TypeSystem::contract(llvm::StringRef name,
                                mlir::ArrayRef<mlir::Type> arguments) const {
  return py::ContractType::get(&context, name, arguments);
}

mlir::Type TypeSystem::protocol(llvm::StringRef name,
                                mlir::ArrayRef<mlir::Type> arguments) const {
  return py::ProtocolType::get(&context, name, arguments);
}

mlir::Type TypeSystem::literal(llvm::StringRef spelling) const {
  return py::LiteralType::get(&context, spelling);
}

mlir::Type TypeSystem::typeObject(mlir::Type instanceType) const {
  return py::TypeType::get(&context, instanceType);
}

mlir::Type TypeSystem::tupleOf(mlir::Type elementType) const {
  llvm::SmallVector<mlir::Type, 1> args;
  if (elementType)
    args.push_back(elementType);
  return contract("builtins.tuple", args);
}

mlir::Type TypeSystem::listOf(mlir::Type elementType) const {
  llvm::SmallVector<mlir::Type, 1> args;
  if (elementType)
    args.push_back(elementType);
  return contract("builtins.list", args);
}

mlir::Type TypeSystem::dictOf(mlir::Type keyType, mlir::Type valueType) const {
  llvm::SmallVector<mlir::Type, 2> args;
  if (keyType)
    args.push_back(keyType);
  if (valueType)
    args.push_back(valueType);
  return contract("builtins.dict", args);
}

mlir::Type TypeSystem::iteratorOf(mlir::Type elementType) const {
  llvm::SmallVector<mlir::Type, 1> args;
  if (elementType)
    args.push_back(elementType);
  return protocol("Iterator", args);
}

mlir::Type TypeSystem::coroutineOf(mlir::Type resultType) const {
  return contract("types.CoroutineType",
                  {any(), any(), resultType ? resultType : any()});
}

namespace {

void collectNameReferences(const parser::Node *node, llvm::StringSet<> &out) {
  if (!node)
    return;
  if (node->kind == "Name") {
    if (auto id = ast::string(*node, "id"))
      out.insert(*id);
  }
  for (const parser::Field &field : node->fields) {
    if (const auto *child = std::get_if<parser::NodePtr>(&field.value)) {
      collectNameReferences(child->get(), out);
      continue;
    }
    if (const auto *children =
            std::get_if<std::vector<parser::NodePtr>>(&field.value))
      for (const parser::NodePtr &nested : *children)
        collectNameReferences(nested.get(), out);
  }
}

// Call sites in module-level statement position, used by the inference
// fixpoint to constrain unannotated parameters. Function, lambda, and class
// subtrees are excluded: their expressions type under scopes this walk does
// not know, and a lenient mis-typing here would pollute the union-find store
// with wrong parameter bindings.
void collectModuleCallNodes(const parser::Node *node,
                            std::vector<const parser::Node *> &out) {
  if (!node)
    return;
  if (node->kind == "FunctionDef" || node->kind == "AsyncFunctionDef" ||
      node->kind == "Lambda" || node->kind == "ClassDef")
    return;
  if (node->kind == "Call")
    out.push_back(node);
  for (const parser::Field &field : node->fields) {
    if (const auto *child = std::get_if<parser::NodePtr>(&field.value)) {
      collectModuleCallNodes(child->get(), out);
      continue;
    }
    if (const auto *children =
            std::get_if<std::vector<parser::NodePtr>>(&field.value))
      for (const parser::NodePtr &nested : *children)
        collectModuleCallNodes(nested.get(), out);
  }
}

} // namespace

void TypeSystem::registerModule(const parser::Node &moduleNode) {
  const auto *body = ast::nodeList(moduleNode, "body");
  if (!body)
    return;

  struct TopLevelFunction {
    const parser::Node *node;
    std::string name;
    llvm::SmallVector<unsigned, 4> callees;
  };
  std::vector<TopLevelFunction> functions;
  llvm::StringMap<unsigned> indexByName;
  for (const parser::NodePtr &statement : *body) {
    if (!statement || (statement->kind != "FunctionDef" &&
                       statement->kind != "AsyncFunctionDef"))
      continue;
    auto name = ast::string(*statement, "name");
    if (!name)
      continue;
    indexByName[*name] = static_cast<unsigned>(functions.size());
    functions.push_back(
        TopLevelFunction{statement.get(), std::string(*name), {}});
  }
  if (functions.empty())
    return;

  // Reference edges are name-based over the whole function subtree. Local
  // shadowing can produce a false edge, but an edge only influences the
  // processing order, never the inferred types, so precision is not worth a
  // scope-aware walk here.
  for (TopLevelFunction &function : functions) {
    llvm::StringSet<> referenced;
    collectNameReferences(function.node, referenced);
    for (const auto &entry : referenced) {
      auto found = indexByName.find(entry.getKey());
      if (found != indexByName.end() &&
          functions[found->second].node != function.node)
        function.callees.push_back(found->second);
    }
  }

  // Tarjan emits each SCC after all SCCs it points to, so processing SCCs in
  // emission order binds callees before the callers whose unannotated
  // returns need them. Members inside one SCC keep source order: annotated
  // signatures are order-independent, and unannotated mutual recursion is
  // diagnosed (monomorphic-recursion inference lands with the body walk).
  struct TarjanState {
    llvm::SmallVector<int> index, lowlink;
    llvm::SmallVector<bool> onStack;
    llvm::SmallVector<unsigned> stack;
    llvm::SmallVector<llvm::SmallVector<unsigned, 2>> components;
    int counter = 0;
  } state;
  state.index.assign(functions.size(), -1);
  state.lowlink.assign(functions.size(), 0);
  state.onStack.assign(functions.size(), false);

  auto strongConnect = [&](auto &&self, unsigned v) -> void {
    state.index[v] = state.lowlink[v] = state.counter++;
    state.stack.push_back(v);
    state.onStack[v] = true;
    for (unsigned w : functions[v].callees) {
      if (state.index[w] < 0) {
        self(self, w);
        state.lowlink[v] = std::min(state.lowlink[v], state.lowlink[w]);
      } else if (state.onStack[w]) {
        state.lowlink[v] = std::min(state.lowlink[v], state.index[w]);
      }
    }
    if (state.lowlink[v] == state.index[v]) {
      llvm::SmallVector<unsigned, 2> component;
      unsigned w;
      do {
        w = state.stack.pop_back_val();
        state.onStack[w] = false;
        component.push_back(w);
      } while (w != v);
      llvm::sort(component);
      state.components.push_back(std::move(component));
    }
  };
  for (unsigned v = 0; v < functions.size(); ++v)
    if (state.index[v] < 0)
      strongConnect(strongConnect, v);

  llvm::SmallVector<unsigned, 8> ordered;
  for (const llvm::SmallVector<unsigned, 2> &component : state.components)
    ordered.append(component.begin(), component.end());

  auto sweep = [&](bool memoize) {
    for (unsigned index : ordered) {
      const TopLevelFunction &function = functions[index];
      FunctionSignature sig = functionSignature(*function.node);
      if (memoize)
        signatureMemo[function.node] = sig;
      bindSymbol(function.name, sig.publicCallable);
    }
  };

  // Assign inference variables to every unannotated parameter, plus a result
  // variable so callers typed during the fixpoint below can consume a
  // function's result before its own body walk succeeds.
  bool anyInference = false;
  for (unsigned index : ordered) {
    const TopLevelFunction &function = functions[index];
    const parser::Node *arguments = ast::node(*function.node, "args");
    if (!arguments)
      continue;
    bool assigned = false;
    auto assignParameters = [&](const std::vector<parser::NodePtr> *args) {
      if (!args)
        return;
      for (const parser::NodePtr &arg : *args) {
        if (!arg || ast::node(*arg, "annotation"))
          continue;
        std::string role = ("parameter '" + llvm::Twine(ast::nameSpelling(*arg)) +
                            "' of '" + function.name + "'")
                               .str();
        parameterTypeOverrides[arg.get()] = inferenceState.freshVar(
            InferenceContext::VarKind::Inference, arg.get(), role);
        assigned = true;
      }
    };
    assignParameters(ast::nodeList(*arguments, "posonlyargs"));
    assignParameters(ast::nodeList(*arguments, "args"));
    assignParameters(ast::nodeList(*arguments, "kwonlyargs"));
    if (!assigned)
      continue;
    anyInference = true;
    if (!ast::node(*function.node, "returns") && function.name != "__init__")
      resultTypeOverrides[function.node] = inferenceState.freshVar(
          InferenceContext::VarKind::Inference, function.node,
          "return type of '" + function.name + "'");
  }

  if (!anyInference) {
    sweep(/*memoize=*/true);
    return;
  }

  std::vector<const parser::Node *> moduleCalls;
  if (const auto *statements = ast::nodeList(moduleNode, "body"))
    for (const parser::NodePtr &statement : *statements)
      collectModuleCallNodes(statement.get(), moduleCalls);

  // Module-wide fixpoint: signature sweeps propagate constraints through
  // function bodies (returns), module-level call sites constrain parameters
  // through the P2 unify bridge in bindExpectedType. Progress is compared on
  // the resolved override types, not the store's generation counter, because
  // speculative candidate exploration bumps the counter even when nothing
  // committed. The iteration cap is a divergence backstop; resolution is
  // monotonic, so a genuine fixpoint is reached in a handful of rounds.
  auto resolvedOverrides = [&]() {
    std::vector<mlir::Type> snapshot;
    snapshot.reserve(parameterTypeOverrides.size() +
                     resultTypeOverrides.size());
    for (const auto &entry : parameterTypeOverrides)
      snapshot.push_back(inferenceState.zonk(entry.second));
    for (const auto &entry : resultTypeOverrides)
      snapshot.push_back(inferenceState.zonk(entry.second));
    return snapshot;
  };
  static const llvm::StringMap<mlir::Type> kNoLocalCallables;
  for (unsigned iteration = 0; iteration < 8; ++iteration) {
    std::vector<mlir::Type> before = resolvedOverrides();
    sweep(/*memoize=*/false);
    ExprInferenceContext moduleCallContext{kNoLocalCallables, nullptr, nullptr,
                                           /*strict=*/true};
    for (const parser::Node *call : moduleCalls)
      (void)inferExpr(call, moduleCallContext);
    if (resolvedOverrides() == before)
      break;
  }

  sweep(/*memoize=*/true);
}

TypeSystem::Scope TypeSystem::pushScope() const {
  scopes.emplace_back();
  scopedCanonicalBindings.emplace_back();
  scopedClasses.emplace_back();
  return Scope(*this);
}

void TypeSystem::popScope() const {
  if (!scopes.empty()) {
    scopes.pop_back();
    scopedCanonicalBindings.pop_back();
    scopedClasses.pop_back();
  }
}

// The `type ? type : object()` guards below defend an internal solver
// invariant: every symbol binding produced by TypeSystem carries a resolved
// type. A null type would be a solver gap, not a language feature. The `object`
// top used as the guard value carries no protocol contract, so if such a gap
// ever reached lowering the erased binding would be rejected for lack of
// evidence rather than dispatched dynamically. The guard is never exercised by
// the accepted example suite (verified via null-binding instrumentation).
void TypeSystem::bindLocalSymbol(llvm::StringRef name, mlir::Type type) const {
  if (scopes.empty())
    return;
  scopes.back()[name] = type ? type : object();
}

void TypeSystem::bindSymbol(llvm::StringRef name, mlir::Type type) {
  if (!scopes.empty()) {
    scopes.back()[name] = type ? type : object();
    scopedCanonicalBindings.back().erase(name);
    return;
  }
  symbols[name] = type ? type : object();
  canonicalBindings.erase(name);
}

void TypeSystem::bindCanonicalSymbol(llvm::StringRef name,
                                     llvm::StringRef canonical,
                                     mlir::Type type) {
  bindSymbol(name, type);
  if (!scopedCanonicalBindings.empty())
    scopedCanonicalBindings.back()[name] = canonical.str();
  else
    canonicalBindings[name] = canonical.str();
}

void TypeSystem::bindAnnotationAlias(llvm::StringRef name,
                                     llvm::StringRef target) {
  annotationAliases[name] = target.str();
}

std::string TypeSystem::resolveAnnotationName(llvm::StringRef name) const {
  auto found = annotationAliases.find(name);
  if (found != annotationAliases.end())
    return found->second;
  return name.str();
}

std::optional<mlir::Type> TypeSystem::lookupSymbol(llvm::StringRef name) const {
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
TypeSystem::lookupCanonicalBinding(llvm::StringRef name) const {
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

void TypeSystem::bindClass(llvm::StringRef name, mlir::Type instanceType) {
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

std::optional<mlir::Type> TypeSystem::lookupClass(llvm::StringRef name) const {
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

bool TypeSystem::bindImportedModule(llvm::StringRef module,
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

  for (const ModuleIntConstantImport &entry : kModuleIntConstantImports) {
    if (module != entry.module)
      continue;
    bindModuleObject();
    bindCanonicalSymbol(importedAttribute(localName, entry.localAttr),
                        entry.canonicalName, intType());
  }

  for (const ModuleStrListImport &entry : kModuleStrListImports) {
    if (module != entry.module)
      continue;
    bindModuleObject();
    bindCanonicalSymbol(importedAttribute(localName, entry.localAttr),
                        entry.canonicalName, listOf(strType()));
  }

  for (const ModuleContractValueImport &entry : kModuleContractValueImports) {
    if (module != entry.module)
      continue;
    bindModuleObject();
    bindCanonicalSymbol(importedAttribute(localName, entry.localAttr),
                        entry.canonicalName, contract(entry.contract));
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
  if (bindManifestModuleIntConstants(*this, module, localName))
    handled = true;
  if (bindManifestModuleStrConstants(*this, module, localName))
    handled = true;

  return handled;
}

bool TypeSystem::bindImportedName(llvm::StringRef module,
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

  for (const NameIntConstantImport &entry : kNameIntConstantImports) {
    if (module != entry.module || exportedName != entry.exportedName)
      continue;
    bindCanonicalSymbol(localName, entry.canonicalName, intType());
    return true;
  }

  for (const NameStrListImport &entry : kNameStrListImports) {
    if (module != entry.module || exportedName != entry.exportedName)
      continue;
    bindCanonicalSymbol(localName, entry.canonicalName, listOf(strType()));
    return true;
  }

  for (const NameContractValueImport &entry : kNameContractValueImports) {
    if (module != entry.module || exportedName != entry.exportedName)
      continue;
    bindCanonicalSymbol(localName, entry.canonicalName,
                        contract(entry.contract));
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
    if (table.moduleIntConstant(canonical)) {
      bindCanonicalSymbol(localName, canonical, intType());
      return true;
    }
    if (table.moduleStrConstant(canonical)) {
      bindCanonicalSymbol(localName, canonical, strType());
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

mlir::Type TypeSystem::annotationType(const parser::Node *node) const {
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

mlir::Type TypeSystem::inferExpr(const parser::Node *node) const {
  return inferenceState.zonk(inferExprImpl(node, nullptr));
}

mlir::Type TypeSystem::inferExpr(const parser::Node *node,
                                 const ExprInferenceContext &ctx) const {
  return inferenceState.zonk(inferExprImpl(node, &ctx));
}

// Lenient and strict inference share one walk. Without a context every
// unresolved construct falls back to object(); with one (unannotated-body
// return/generator inference) local callables shadow the symbol table and
// failures propagate as a null type with a recorded reason.
mlir::Type TypeSystem::inferExprImpl(const parser::Node *node,
                                     const ExprInferenceContext *ctx) const {
  if (!node)
    return object();
  const bool strict = ctx && ctx->strict;
  auto fail = [&](std::string reason) -> mlir::Type {
    if (ctx)
      recordInferenceFailure(ctx->failureReasons, std::move(reason));
    return {};
  };
  auto recurse = [&](const parser::Node *child) {
    return inferExprImpl(child, ctx);
  };
  // Lenient re-inference of a subexpression. The walk-bound locals stay
  // visible (they used to live on the scope), but local callables and strict
  // failure propagation do not — the historical lenient view.
  auto lenientRecurse = [&](const parser::Node *child) -> mlir::Type {
    if (!ctx)
      return inferExprImpl(child, nullptr);
    static const llvm::StringMap<mlir::Type> kNoCallables;
    ExprInferenceContext lenient{kNoCallables, nullptr, ctx->localSymbols,
                                 /*strict=*/false};
    return inferExprImpl(child, &lenient);
  };
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
    if (ast::bytes(*node, "value"))
      return contract("builtins.bytes");
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
    if (ctx) {
      auto found = ctx->localCallables.find(name);
      if (found != ctx->localCallables.end())
        return found->second;
      if (ctx->localSymbols) {
        auto local = ctx->localSymbols->find(name);
        if (local != ctx->localSymbols->end())
          return local->second;
      }
    }
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
    if (mlir::Type objectType = lenientRecurse(ast::node(*node, "value"))) {
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
  if (node->kind == "List" || node->kind == "Tuple") {
    llvm::SmallVector<mlir::Type, 8> elementTypes;
    if (const auto *elements = ast::nodeList(*node, "elts")) {
      elementTypes.reserve(elements->size());
      for (const parser::NodePtr &element : *elements) {
        mlir::Type elementType = recurse(element.get());
        if (strict && !elementType)
          return {};
        elementTypes.push_back(widenLiteral(elementType));
      }
    }
    if (node->kind == "List")
      return listOf(join(elementTypes));
    // The strict path keeps the historical joined (homogeneous) tuple view;
    // the lenient path types heterogeneous tuples positionally.
    return strict ? tupleOf(join(elementTypes))
                  : tupleOfMembers(*this, elementTypes);
  }
  if (node->kind == "Dict") {
    const auto *keys = ast::nodeList(*node, "keys");
    const auto *values = ast::nodeList(*node, "values");
    if (!keys || !values)
      return dictOf(object(), object());
    llvm::SmallVector<mlir::Type, 8> keyTypes;
    llvm::SmallVector<mlir::Type, 8> valueTypes;
    keyTypes.reserve(keys->size());
    valueTypes.reserve(values->size());
    for (auto [index, key] : llvm::enumerate(*keys)) {
      if (!key)
        return dictOf(object(), object());
      mlir::Type keyType = recurse(key.get());
      if (strict && !keyType)
        return {};
      keyTypes.push_back(widenLiteral(keyType));
      if (index < values->size()) {
        mlir::Type valueType = recurse((*values)[index].get());
        if (strict && !valueType)
          return {};
        valueTypes.push_back(widenLiteral(valueType));
      }
    }
    return dictOf(join(keyTypes), join(valueTypes));
  }
  if (node->kind == "Subscript") {
    if (std::optional<PrimitiveTypeSpec> primitive =
            primitiveTypeSpecFromSubscript(node, *this))
      return typeObject(primitive->type);
    mlir::Type container = recurse(ast::node(*node, "value"));
    mlir::Type index = recurse(ast::node(*node, "slice"));
    if (strict) {
      if (!container || !index)
        return {};
      CallInferenceResult inference = inferMethodCallWithEvidence(
          widenLiteral(container), "__getitem__", {index});
      if (inference)
        return inference.resultType;
      return fail(inference.failureReason);
    }
    if (std::optional<CallSolution> result = tryManifestMethod(
            *this, widenLiteral(container), "__getitem__", {index}))
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
    return join({widenLiteral(lenientRecurse(ast::node(*node, "body"))),
                 widenLiteral(lenientRecurse(ast::node(*node, "orelse")))});
  if (node->kind == "UnaryOp") {
    if (strict && !recurse(ast::node(*node, "operand")))
      return {};
    const parser::Node *op = ast::node(*node, "op");
    mlir::Type operand = lenientRecurse(ast::node(*node, "operand"));
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
    if (strict) {
      if (!recurse(ast::node(*node, "left")))
        return {};
      if (!recurse(ast::node(*node, "right")))
        return {};
    }
    const parser::Node *op = ast::node(*node, "op");
    mlir::Type rawLeft = lenientRecurse(ast::node(*node, "left"));
    mlir::Type rawRight = lenientRecurse(ast::node(*node, "right"));
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
    mlir::Type awaitable = recurse(ast::node(*node, "value"));
    if (strict && !awaitable)
      return {};
    AwaitInferenceResult inference =
        inferAwaitWithEvidence(widenLiteral(awaitable));
    if (inference)
      return inference.resultType;
    return strict ? fail(inference.failureReason) : object();
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
          mlir::Type starredType = recurse(ast::node(*arg, "value"));
          if (strict && !starredType)
            return {};
          if (!appendStarredCallArgumentTypes(*this, starredType, positional))
            return strict ? fail("starred call arguments require a "
                                 "statically sized tuple")
                          : object();
          continue;
        }
        mlir::Type argType = recurse(arg.get());
        if (strict && !argType)
          return {};
        positional.push_back(argType);
      }
    }

    llvm::SmallVector<CallKeywordType, 4> keywords;
    if (const auto *keywordNodes = ast::nodeList(*node, "keywords")) {
      for (const parser::NodePtr &keyword : *keywordNodes) {
        auto name = ast::string(*keyword, "arg");
        if (!name) {
          if (strict)
            return fail(
                "keyword splat call arguments require static keyword names");
          continue;
        }
        mlir::Type keywordType = recurse(ast::node(*keyword, "value"));
        if (strict && !keywordType)
          return {};
        keywords.push_back(CallKeywordType{std::string(*name), keywordType});
      }
    }

    if (callee && callee->kind == "Name") {
      llvm::StringRef name = ast::nameSpelling(*callee);
      if (ctx) {
        auto found = ctx->localCallables.find(name);
        if (found != ctx->localCallables.end()) {
          CallInferenceResult inference =
              inferCallWithEvidence(found->second, positional, keywords);
          if (inference)
            return inference.resultType;
          return fail(inference.failureReason);
        }
      }
      if (name == "isinstance")
        return boolType();
      // open()'s return type depends on the MODE: a str literal containing
      // 'b' selects the binary arm statically (FileIO); everything else is
      // the text wrapper. A non-literal binary mode cannot type as FileIO
      // and is rejected at runtime by the text arm's mode parser.
      if (!strict && lookupCanonicalBinding(name) ==
                         std::optional<std::string>("_io.open")) {
        if (const auto *args = ast::nodeList(*node, "args"))
          if (args->size() >= 2 && (*args)[1]) {
            auto mode = ast::string(*(*args)[1], "value");
            if (mode && mode->find('b') != std::string_view::npos)
              return contract("_io.FileIO");
          }
        return contract("_io.TextIOWrapper");
      }
      if (name == "len") {
        if (strict) {
          if (positional.empty())
            return fail("len expects one positional argument");
          CallInferenceResult inference = inferMethodCallWithEvidence(
              widenLiteral(positional.front()), "__len__", {});
          if (inference)
            return inference.resultType;
          return fail(inference.failureReason);
        }
        if (!positional.empty())
          if (std::optional<CallSolution> result = tryManifestMethod(
                  *this, widenLiteral(positional.front()), "__len__", {}))
            return result->result;
        return object();
      }
      if (name == "round") {
        if (strict) {
          if (positional.empty())
            return fail("round expects at least one positional argument");
          llvm::SmallVector<mlir::Type, 1> extra;
          if (positional.size() > 1)
            extra.push_back(positional[1]);
          mlir::Type input = widenLiteral(positional.front());
          CallInferenceResult inference =
              inferMethodCallWithEvidence(input, "__round__", extra);
          if (inference)
            return inference.resultType;
          if (input == intType())
            return intType();
          return fail(inference.failureReason);
        }
        const auto *args = ast::nodeList(*node, "args");
        if (args && !args->empty()) {
          mlir::Type input = widenLiteral(lenientRecurse(args->front().get()));
          llvm::SmallVector<mlir::Type, 1> extra;
          if (args->size() > 1)
            extra.push_back(lenientRecurse((*args)[1].get()));
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
      if (auto cls = lookupClass(name)) {
        mlir::Type instance = inferClassInstantiation(*cls, positional, keywords);
        if (!strict || instance)
          return instance;
        return fail("class instantiation leaves unbound static type "
                    "parameters for '" +
                    name.str() + "'");
      }
      if (std::optional<std::string> canonical = lookupCanonicalBinding(name)) {
        if (*canonical == "lyrt.from_prim" && positional.size() == 1 &&
            keywords.empty()) {
          mlir::Type result = primitivePythonResultType(positional.front(), *this);
          if (!strict || result)
            return result;
          return fail("lyrt.from_prim expects a primitive scalar or shaped "
                      "primitive value");
        }
        if (*canonical == "asyncio.sleep")
          return inferAsyncioSleepResult(*this, positional, keywords);
      }
      std::optional<mlir::Type> symbol;
      if (ctx && ctx->localSymbols) {
        auto local = ctx->localSymbols->find(name);
        if (local != ctx->localSymbols->end())
          symbol = local->second;
      }
      if (!symbol)
        symbol = lookupSymbol(name);
      if (symbol) {
        if (strict) {
          CallInferenceResult inference =
              inferCallWithEvidence(*symbol, positional, keywords);
          if (inference)
            return inference.resultType;
          return fail(inference.failureReason);
        }
        return inferCall(*symbol, positional, keywords);
      }
    }
    if (callee && callee->kind == "Attribute") {
      std::string qualified = ast::qualifiedName(callee);
      if (std::optional<std::string> canonical =
              lookupCanonicalBinding(qualified)) {
        if (*canonical == "lyrt.from_prim" && positional.size() == 1 &&
            keywords.empty()) {
          mlir::Type result = primitivePythonResultType(positional.front(), *this);
          if (!strict || result)
            return result;
          return fail("lyrt.from_prim expects a primitive scalar or shaped "
                      "primitive value");
        }
        if (*canonical == "asyncio.sleep")
          return inferAsyncioSleepResult(*this, positional, keywords);
      }
      if (auto symbol = lookupSymbol(qualified)) {
        if (strict) {
          CallInferenceResult inference =
              inferCallWithEvidence(*symbol, positional, keywords);
          if (inference)
            return inference.resultType;
          return fail(inference.failureReason);
        }
        return inferCall(*symbol, positional, keywords);
      }
      if (const parser::Node *receiverNode = ast::node(*callee, "value")) {
        if (auto methodName = ast::string(*callee, "attr")) {
          mlir::Type receiver = recurse(receiverNode);
          if (strict && !receiver)
            return {};
          CallInferenceResult inference = inferMethodCallWithEvidence(
              widenLiteral(receiver), *methodName, positional, keywords);
          if (inference)
            return inference.resultType;
          return strict ? fail(inference.failureReason) : object();
        }
      }
    }
    if (callee) {
      mlir::Type calleeType = recurse(callee);
      if (strict) {
        if (!calleeType)
          return {};
        CallInferenceResult inference =
            inferCallWithEvidence(calleeType, positional, keywords);
        if (inference)
          return inference.resultType;
        return fail(inference.failureReason);
      }
      return inferCall(calleeType, positional, keywords);
    }
    return strict ? fail("call expression is missing a callee") : object();
  }
  if (node->kind == "Lambda") {
    return functionSignature(*node).callable;
  }
  return object();
}

mlir::Type
TypeSystem::inferCall(mlir::Type calleeType,
                      mlir::ArrayRef<mlir::Type> positional,
                      mlir::ArrayRef<CallKeywordType> keywords) const {
  CallInferenceResult inference =
      inferCallWithEvidence(calleeType, positional, keywords);
  return inference ? inference.resultType : object();
}

mlir::Type TypeSystem::inferClassInstantiation(
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

CallInferenceResult TypeSystem::inferMethodCallWithEvidence(
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

bool TypeSystem::isStructuralMutatorMethod(mlir::Type receiverType,
                                           llvm::StringRef methodName) const {
  if (!receiverType)
    return false;
  const py::protocols::Table &table = py::protocols::Table::get(context);
  return table.isStructuralMutator(widenLiteral(receiverType), methodName);
}

std::optional<std::vector<std::string>>
TypeSystem::classMatchArgs(mlir::Type receiverType) const {
  if (!receiverType)
    return std::nullopt;
  const py::protocols::Table &table = py::protocols::Table::get(context);
  return table.matchArgsFor(widenLiteral(receiverType));
}

AwaitInferenceResult
TypeSystem::inferAwaitWithEvidence(mlir::Type awaitableType) const {
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
TypeSystem::inferYieldFromWithEvidence(mlir::Type sourceType) const {
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
TypeSystem::inferAsyncIterationWithEvidence(mlir::Type iterableType) const {
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
inferAsyncContextMethod(const TypeSystem &types, mlir::Type managerType,
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
TypeSystem::inferAsyncContextEnterWithEvidence(mlir::Type managerType) const {
  return inferAsyncContextMethod(*this, managerType, "__aenter__", {});
}

AsyncContextMethodInferenceResult TypeSystem::inferAsyncContextExitWithEvidence(
    mlir::Type managerType, mlir::ArrayRef<mlir::Type> exceptionTypes) const {
  return inferAsyncContextMethod(*this, managerType, "__aexit__",
                                 exceptionTypes);
}

CallInferenceResult TypeSystem::inferCallWithEvidence(
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
TypeSystem::fieldAssignmentRefinement(mlir::Type receiverType,
                                      llvm::StringRef fieldName,
                                      mlir::Type valueType) const {
  const py::protocols::Table &table = py::protocols::Table::get(context);
  return table.refineContractByFieldAssignment(widenLiteral(receiverType),
                                               fieldName, valueType);
}

mlir::Type TypeSystem::join(mlir::ArrayRef<mlir::Type> types) const {
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

mlir::Type TypeSystem::widenLiteral(mlir::Type type) const {
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
TypeSystem::functionSignature(const parser::Node &function,
                              std::optional<llvm::StringRef> selfName,
                              py::CallableType expectedCallable) const {
  if (!selfName && !expectedCallable) {
    auto memoized = signatureMemo.find(&function);
    if (memoized != signatureMemo.end())
      return memoized->second;
  }
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
    // An unannotated parameter with a registered inference variable takes
    // its (partially) resolved type instead of a missing-annotation record;
    // if the module fixpoint left it unresolved, the record comes back so
    // the emit boundary still refuses the function explicitly.
    auto overriddenParameterType =
        [&](const parser::Node &arg) -> std::optional<mlir::Type> {
      auto found = parameterTypeOverrides.find(&arg);
      if (found == parameterTypeOverrides.end())
        return std::nullopt;
      return inferenceState.zonk(found->second);
    };
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
      std::optional<mlir::Type> overridden =
          !annotation && !isSelfParameter && !fromExpectedCallable
              ? overriddenParameterType(*arg)
              : std::nullopt;
      if (isSelfParameter)
        type = py::SelfType::get(&context);
      if (fromExpectedCallable)
        type = expectedPositional[index];
      if (overridden) {
        type = *overridden;
        if (py::containsPyInferVar(type))
          sig.missingParameterAnnotations.push_back(name);
      } else if (!isSelfParameter && !fromExpectedCallable) {
        recordAnnotationIssue(annotation, name);
      }
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
        std::optional<mlir::Type> overridden =
            !annotation && !fromExpectedCallable ? overriddenParameterType(*arg)
                                                 : std::nullopt;
        if (fromExpectedCallable)
          type = expectedKwOnly[index];
        if (overridden) {
          type = *overridden;
          if (py::containsPyInferVar(type))
            sig.missingParameterAnnotations.push_back(name);
        } else if (!fromExpectedCallable) {
          recordAnnotationIssue(annotation, name);
        }
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
  } else if (ast::nameSpelling(function) == "__init__") {
    sig.resultType = none();
  } else {
    mlir::Type walked = inferredFunctionResult(*this, function,
                                               &sig.bodyInferenceFailures,
                                               &generator.localSymbols);
    auto overridden = resultTypeOverrides.find(&function);
    if (overridden == resultTypeOverrides.end()) {
      sig.resultType = walked;
    } else {
      // The result variable exists so callers typed during the module
      // fixpoint can consume this function's result before its body walk
      // succeeds. Only a fully resolved walk binds it: partially resolved
      // results would freeze a stale type into the union-find store. The
      // walk result is literal-widened (member-wise through unions) before
      // binding — successive fixpoint rounds join a recursive function's
      // literal base case with its widened recursive case, and an equational
      // variable cannot hold both spellings of the same contract.
      if (walked) {
        mlir::Type resolved = inferenceState.zonk(walked);
        if (!py::containsPyInferVar(resolved)) {
          mlir::Type widened;
          if (auto unionType =
                  mlir::dyn_cast_if_present<py::UnionType>(resolved)) {
            llvm::SmallVector<mlir::Type, 4> members;
            for (mlir::Type member : unionType.getMemberTypes())
              members.push_back(widenLiteral(member));
            widened = join(members);
          } else {
            widened = widenLiteral(resolved);
          }
          if (InferenceContext::UnifyResult bound =
                  inferenceState.unify(overridden->second, widened);
              !bound)
            sig.bodyInferenceFailures.push_back(bound.reason);
        }
      }
      sig.resultType = inferenceState.zonk(overridden->second);
    }
  }
  refreshCallable(sig);
  return sig;
}

void TypeSystem::refreshCallable(FunctionSignature &sig) const {
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
