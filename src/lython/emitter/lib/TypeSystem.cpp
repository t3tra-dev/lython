#include "TypeSystem.h"

#include "TypeSystemSolver.h"

#include "llvm/ADT/StringExtras.h"

#include "AstAccess.h"
#include "AstSynth.h"
#include "CandidateSelection.h"
#include "ExceptionTaxonomy.h"
#include "PlatformConstants.h"
#include "PrimitiveTypes.h"
#include "PyCallableShape.h"
#include "Parser.h"
#include "PyProtocols.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/SaveAndRestore.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <map>
#include <string>

namespace lython::emitter {

// ⭐ An expression that builds an EMPTY container and so carries no element
// type of its own. Five joins ask this -- the container literal's siblings, a
// conditional expression's arms, a boolean operator's operands, a class
// field's assignments and a parameter's default -- and each absence of it
// surfaced as a different message.
//
// ⛔ The zero-argument CONSTRUCTOR counts, because `set()` is the only
// spelling an empty set has: `set() if flag else {1}` reached the same erased
// join a `[]` arm did and was refused for it.
bool isEmptyContainerExpression(const parser::Node *node) {
  if (!node)
    return false;
  if (node->kind == "List" || node->kind == "Tuple" || node->kind == "Set") {
    const auto *elements = ast::nodeList(*node, "elts");
    return !elements || elements->empty();
  }
  if (node->kind == "Dict") {
    const auto *keys = ast::nodeList(*node, "keys");
    return !keys || keys->empty();
  }
  if (node->kind == "Call") {
    const parser::Node *callee = ast::node(*node, "func");
    if (!callee || callee->kind != "Name")
      return false;
    llvm::StringRef name = ast::nameSpelling(*callee);
    if (name != "set" && name != "list" && name != "dict" &&
        name != "tuple" && name != "frozenset")
      return false;
    const auto *args = ast::nodeList(*node, "args");
    const auto *keywords = ast::nodeList(*node, "keywords");
    return (!args || args->empty()) && (!keywords || keywords->empty());
  }
  return false;
}
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

// ⭐ "DOES NOT PROVIDE" AND "DOES NOT ACCEPT THESE" ARE DIFFERENT SENTENCES,
// and the first was said for both:
//
//     xs: list[float] = [1.0]
//     xs.append(1)
//     # static type list[float] does not provide manifest method 'append'
//
// `list[float]` provides `append`; what it does not have is an overload taking
// an int. A reader given the first sentence looks for a missing method, which
// is not there to find. The protocol table already enumerates candidates BY
// NAME, so the two cases are one lookup apart.
CallInferenceResult unresolvedMethodCall(const TypeSystem &types,
                                         mlir::Type receiverType,
                                         llvm::StringRef methodName,
                                         mlir::ArrayRef<mlir::Type> positional,
                                         mlir::ArrayRef<CallKeywordType>
                                             keywords) {
  std::string subject =
      "static type " + typeText(types.widenLiteral(receiverType));
  if (!types.declaresManifestMethod(receiverType, methodName))
    return CallInferenceResult{
        {},
        {},
        false,
        subject + " does not provide manifest method '" + methodName.str() +
            "'"};
  std::string arguments;
  for (mlir::Type argument : positional) {
    if (!arguments.empty())
      arguments += ", ";
    arguments += typeText(types.widenLiteral(argument));
  }
  for (const CallKeywordType &keyword : keywords) {
    if (!arguments.empty())
      arguments += ", ";
    arguments += keyword.name + "=" + typeText(types.widenLiteral(keyword.type));
  }
  return CallInferenceResult{
      {},
      {},
      false,
      subject + " has manifest method '" + methodName.str() +
          "' but no signature that accepts (" + arguments + ")"};
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

// Where a type stands in the numeric tower: bool below int below float below
// complex, and -1 for anything that is not one of them (a union included).
int numericTowerRung(const TypeSystem &types, mlir::Type type) {
  if (!type)
    return -1;
  if (type == types.boolType())
    return 0;
  if (type == types.intType())
    return 1;
  if (type == types.floatType())
    return 2;
  auto contract = mlir::dyn_cast_if_present<py::ContractType>(type);
  if (contract && contract.getArguments().empty() &&
      contract.getContractName() == "builtins.complex")
    return 3;
  return -1;
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
  // The node each yield type came from, so an EMPTY container yield can be
  // dropped from the join the way an empty sibling is: `yield []` in one arm
  // and `yield [1]` in the other joined to a union of two lists, and the
  // frame lane for that is what refused the generator.
  llvm::SmallVector<const parser::Node *, 4> yieldNodes;
  llvm::SmallVector<mlir::Type, 4> returnTypes;
  // Send types of `yield from` delegates: PEP 380 forwards send() into the
  // active delegate, so an unannotated outer generator's send channel is
  // the join of its delegates'.
  llvm::SmallVector<mlir::Type, 2> delegatedSendTypes;
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


// ⭐ An EMPTY container literal contributes NOTHING to a sibling join. It has
// no element type of its own, so it types as `list[object]`, and joining that
// with a sibling's `list[int]` gives `list[int] | list[object]` -- a union of
// two list types that nothing accepts:
//
//     g = {0: [1, 2], 1: []}
//     len(g[0])
//     # '!py.union<list[int], list[object]>' does not provide '__len__'
//
// which is every adjacency map, bucket table and grouping with an empty entry
// in its literal. The sibling IS the element type here, the same way an
// annotation is when there is one; an empty literal beside a typed one is not
// evidence of heterogeneity, it is an absence of evidence.
//
// ⛔ Only when a non-empty sibling exists. `[[], []]` and `{0: []}` have
// nothing to take a type from and keep the erased element they always had.
bool isEmptyContainerLiteral(const parser::Node *node) {
  return isEmptyContainerExpression(node);
}

// The join over `types`, with the entries whose node is an empty container
// literal dropped when anything else contributed.
mlir::Type joinIgnoringEmptyLiterals(
    const TypeSystem &types, llvm::ArrayRef<mlir::Type> collected,
    llvm::ArrayRef<const parser::Node *> nodes) {
  llvm::SmallVector<mlir::Type, 8> kept;
  for (auto [index, type] : llvm::enumerate(collected))
    if (index >= nodes.size() || !isEmptyContainerLiteral(nodes[index]))
      kept.push_back(type);
  return types.join(kept.empty() ? collected : kept);
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

// ⭐ A GENERATOR'S LOCAL MAY BE BOUND BY AN UNPACK. This walk binds the names
// a `yield` will read, and it only ever looked at a bare `Name` target -- so
// the most common generator there is went out with `object` as its yield type:
//
//     def fib():
//         a, b = 0, 1
//         while True:
//             yield a
//             a, b = b, a + b
//     # static type builtins.object does not provide manifest method '__gt__'
//
// `a = 0` followed by `b = 1` worked, which is the whole difference.
//
// The RHS is read positionally when it is a literal of the same arity, because
// that is exact and does not depend on how a heterogeneous tuple happens to be
// spelled. Otherwise the value's TYPE is distributed: a positional
// `tuple[A, B]` by position, and a one-argument container (`tuple[T]`,
// `list[T]`) to every name. A shape this cannot read binds nothing, which is
// the answer it gave before.
void bindGeneratorAnalysisTarget(const TypeSystem &types,
                                 const parser::Node *target,
                                 mlir::Type valueType,
                                 const parser::Node *valueNode,
                                 const llvm::StringMap<mlir::Type> &localCallables,
                                 GeneratorFunctionAnalysis &analysis) {
  if (!target)
    return;
  if (target->kind == "Name") {
    llvm::StringRef name = ast::nameSpelling(*target);
    // ⭐ AN EMPTY CONTAINER LITERAL KEEPS THE TYPE THE NAME ALREADY HAS, which
    // is the rule the emitter applies to the same rebind outside a generator
    // (an empty literal has no element type of its own). Overwriting with
    // `list[object]` here made the frame's slot a union of the two readings:
    //
    //     buf: list[int] = []
    //     ...
    //     yield buf
    //     buf = []          # list[object] -> union<list[int], list[object]>
    //
    //     # runtime bundle for '!py.union<list[int], list[object]>' has 1 values
    //
    // which is the chunking idiom and every accumulate-and-flush generator with
    // it.
    if (isEmptyContainerLiteral(valueNode)) {
      auto existing = analysis.localSymbols.find(name);
      if (existing != analysis.localSymbols.end() && existing->second &&
          types.widenLiteral(existing->second) != types.object())
        return;
    }
    analysis.localSymbols[name] = valueType ? valueType : types.object();
    return;
  }
  if (target->kind != "Tuple" && target->kind != "List")
    return;
  const auto *elements = ast::nodeList(*target, "elts");
  if (!elements || elements->empty())
    return;
  if (valueNode && (valueNode->kind == "Tuple" || valueNode->kind == "List"))
    if (const auto *valueElements = ast::nodeList(*valueNode, "elts");
        valueElements && valueElements->size() == elements->size()) {
      for (auto [index, element] : llvm::enumerate(*elements)) {
        const parser::Node *source = (*valueElements)[index].get();
        mlir::Type elementType = inferExprWithLocalCallables(
            types, source, localCallables, nullptr, &analysis.localSymbols);
        bindGeneratorAnalysisTarget(
            types, element.get(),
            elementType ? types.widenLiteral(elementType) : mlir::Type(),
            source, localCallables, analysis);
      }
      return;
    }
  auto contract =
      mlir::dyn_cast_if_present<py::ContractType>(types.widenLiteral(valueType));
  if (!contract)
    return;
  llvm::ArrayRef<mlir::Type> arguments = contract.getArguments();
  if (contract.getContractName() == "builtins.tuple" &&
      arguments.size() == elements->size()) {
    for (auto [index, element] : llvm::enumerate(*elements))
      bindGeneratorAnalysisTarget(types, element.get(), arguments[index],
                                  nullptr, localCallables, analysis);
    return;
  }
  if (arguments.size() == 1)
    for (const parser::NodePtr &element : *elements)
      bindGeneratorAnalysisTarget(types, element.get(), arguments.front(),
                                  nullptr, localCallables, analysis);
}

void collectGeneratorFunctionAnalysis(
    const TypeSystem &types, const parser::Node *node,
    const llvm::StringMap<mlir::Type> &localCallables,
    mlir::Type generatorSendHint, GeneratorFunctionAnalysis &analysis) {
  if (!node)
    return;
  if (node->kind == "FunctionDef" || node->kind == "AsyncFunctionDef") {
    // ⭐ The walk declines to look INSIDE a nested def, which is about its own
    // binding order; what the NAME it leaves behind is worth is a different
    // question, and nothing answered it. `yield f` for a def declared in the
    // loop body yielded `Callable[[], object]`, and the generator's frame lane
    // for that erased result then failed in the lowering.
    if (auto nestedName = ast::string(*node, "name")) {
      FunctionSignature nested = types.functionSignature(*node);
      if (nested.publicCallable)
        analysis.localSymbols[*nestedName] = nested.publicCallable;
    }
    return;
  }
  if (node->kind == "Lambda" || node->kind == "ClassDef")
    return;
  if (node->kind == "Yield") {
    analysis.hasYield = true;
    const parser::Node *value = ast::node(*node, "value");
    if (!value) {
      analysis.yieldTypes.push_back(types.none());
      analysis.yieldNodes.push_back(nullptr);
      return;
    }
    mlir::Type valueType =
        inferExprWithLocalCallables(types, value, localCallables,
                                    &analysis.failureReasons,
                                    &analysis.localSymbols);
    if (valueType) {
      analysis.yieldTypes.push_back(types.widenLiteral(valueType));
      analysis.yieldNodes.push_back(value);
    }
    return;
  }
  if (node->kind == "YieldFrom") {
    analysis.hasYield = true;
    analysis.sawYieldFrom = true;
    if (std::optional<mlir::Type> element = generatorYieldFromElementType(
            types, ast::node(*node, "value"), localCallables,
            analysis.failureReasons, &analysis.localSymbols))
      analysis.yieldTypes.push_back(*element);
      analysis.yieldNodes.push_back(nullptr);
    if (mlir::Type rawSource = inferExprWithLocalCallables(
            types, ast::node(*node, "value"), localCallables, nullptr,
            &analysis.localSymbols)) {
      const py::protocols::Table &table = py::protocols::Table::get(
          types.getContext());
      if (std::optional<std::vector<mlir::Type>> generator =
              table.protocolArgumentsFor(types.widenLiteral(rawSource),
                                         "Generator"))
        if (generator->size() >= 2)
          analysis.delegatedSendTypes.push_back((*generator)[1]);
    }
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
    mlir::Type valueType;
    if (value && value->kind == "Yield" && generatorSendHint) {
      valueType = generatorSendHint;
    } else if (value && value->kind == "YieldFrom") {
      // `x = yield from g()` binds the delegate's COMPLETION type (its
      // return value), not the yielded element type the generic walk sees.
      mlir::Type rawSource = inferExprWithLocalCallables(
          types, ast::node(*value, "value"), localCallables, nullptr,
          &analysis.localSymbols);
      if (rawSource) {
        YieldFromInferenceResult inference =
            types.inferYieldFromWithEvidence(types.widenLiteral(rawSource));
        if (inference)
          valueType = inference.completionType;
      }
    }
    if (!valueType)
      valueType = lenientWalkInfer(types, value, analysis);
    if (const auto *targets = ast::nodeList(*node, "targets"))
      for (const parser::NodePtr &target : *targets)
        bindGeneratorAnalysisTarget(types, target.get(), valueType, value,
                                    localCallables, analysis);
    return;
  }
  if (node->kind == "AnnAssign") {
    const parser::Node *value = ast::node(*node, "value");
    collectGeneratorFunctionAnalysis(types, value, localCallables,
                                     generatorSendHint, analysis);
    const parser::Node *target = ast::node(*node, "target");
    if (target && target->kind == "Name") {
      mlir::Type type = types.annotationType(ast::node(*node, "annotation"));
      // ⭐ A NUMERIC ANNOTATION DOES NOT RETYPE THE VALUE, and this walk has to
      // say what the EMITTER does rather than what the declaration asks for.
      // `coerceValue` declines to retype between int, float and bool -- they
      // share no representation -- so `v: float = 1` binds the int, and a walk
      // that recorded `float` here made the function disagree with its own
      // body: `def outer() -> float: v: float = 1; return v` was refused with
      // "type of return operand 0 ('!py.literal<1>') doesn't match function
      // result type".
      //
      // ⛔ Not for a `complex` annotation: `inferExpr` answers
      // `builtins.float` for `1.0 + 0.0j`, so re-reading one would bind float
      // over a complex value. Same measurement as the return walk's.
      if (value)
        if (int declaredRung = numericTowerRung(types, type);
            declaredRung > 0 && declaredRung <= 2) {
          mlir::Type supplied =
              types.widenLiteral(lenientWalkInfer(types, value, analysis));
          if (int suppliedRung = numericTowerRung(types, supplied);
              suppliedRung >= 0 && suppliedRung < declaredRung)
            type = supplied;
        }
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
  if (node->kind == "With") {
    // ⭐ AND A `with ... as X` TARGET, for the same reason the loop target is
    // bound: the walk descends into the body, so a yield over the target has
    // to know what the target is. Without it `X` was `object` and every
    // generator holding a context manager was refused --
    //
    //     def go() -> Iterator[int]:
    //         with Ctx() as base:
    //             yield base
    //
    // "generator function is annotated Iterator[int] but yields
    // builtins.object" -- while the same `with` around a yield of a LITERAL
    // reaches the lowering, which is what says the target is the gap.
    //
    // ⛔ `With` and not `AsyncWith`: `__aenter__` answers with an awaitable
    // and the target binds what awaiting it produces, which is a second
    // question this walk has no answer for; binding the awaitable would be
    // worse than binding nothing.
    if (const auto *items = ast::nodeList(*node, "items"))
      for (const parser::NodePtr &item : *items) {
        if (!item)
          continue;
        const parser::Node *target = ast::node(*item, "optional_vars");
        const parser::Node *contextExpr = ast::node(*item, "context_expr");
        if (!target || !contextExpr)
          continue;
        mlir::Type contextType = inferExprWithLocalCallables(
            types, contextExpr, localCallables, &analysis.failureReasons,
            &analysis.localSymbols);
        if (!contextType)
          continue;
        CallInferenceResult entered = types.inferMethodCallWithEvidence(
            types.widenLiteral(contextType), "__enter__", {});
        if (!entered)
          continue;
        bindGeneratorAnalysisTarget(types, target,
                                    types.widenLiteral(entered.resultType),
                                    nullptr, localCallables, analysis);
      }
  }
  if (node->kind == "For" || node->kind == "AsyncFor") {
    // Bind the loop target to the iteration element type before the generic
    // child walk reaches the body, so yields over the target infer correctly.
    const parser::Node *target = ast::node(*node, "target");
    const parser::Node *iter = ast::node(*node, "iter");
    if (target && iter) {
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
          bindGeneratorAnalysisTarget(types, target, element, nullptr,
                                      localCallables, analysis);
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

// ⭐ A RECURSIVE GENERATOR CALLS ITSELF, and the walk that decides what it
// yields has to be able to type that call:
//
//     def walk(n: Node) -> Iterator[int]:
//         yield n.v
//         for k in n.kids:
//             for v in walk(k):      # walk: literal<None> here
//                 yield v
//
// which is every tree traversal. Unannotated, the self-call is unknowable and
// stays refused; ANNOTATED, the answer is written right there, so the name is
// bound to a callable built from the annotations alone.
//
// ⛔ Built from the annotations rather than from `functionSignature`: that call
// is what is running, and its memo is not filled yet, so asking it here
// recurses forever. Any unannotated parameter skips the binding -- a partial
// callable would answer the call with the wrong arity rather than not at all.
llvm::StringMap<mlir::Type>
selfCallableFromAnnotations(const TypeSystem &types,
                            const parser::Node &function) {
  llvm::StringMap<mlir::Type> self;
  std::optional<std::string_view> name = ast::string(function, "name");
  const parser::Node *returns = ast::node(function, "returns");
  if (!name || !returns)
    return self;
  mlir::Type resultType = types.annotationType(returns);
  if (!resultType)
    return self;
  llvm::SmallVector<mlir::Type, 4> parameters;
  const parser::Node *arguments = ast::node(function, "args");
  if (!arguments)
    return self;
  for (llvm::StringRef field : {"posonlyargs", "args"}) {
    const auto *args = ast::nodeList(*arguments, field);
    if (!args)
      continue;
    for (const parser::NodePtr &arg : *args) {
      if (!arg)
        return {};
      const parser::Node *annotation = ast::node(*arg, "annotation");
      if (!annotation)
        return {};
      mlir::Type parameterType = types.annotationType(annotation);
      if (!parameterType)
        return {};
      parameters.push_back(parameterType);
    }
  }
  if (ast::nodeList(*arguments, "kwonlyargs") &&
      !ast::nodeList(*arguments, "kwonlyargs")->empty())
    return {};
  self[*name] = py::CallableType::get(&types.getContext(), parameters, {},
                                      mlir::Type(), mlir::Type(), {resultType});
  return self;
}

GeneratorFunctionAnalysis
analyzeGeneratorFunction(const TypeSystem &types, const parser::Node &function,
                         mlir::Type generatorSendHint = {}) {
  GeneratorFunctionAnalysis analysis;
  llvm::StringMap<mlir::Type> localCallables =
      localCallableTypesInFunction(types, function);
  for (const auto &entry : selfCallableFromAnnotations(types, function))
    if (!localCallables.count(entry.getKey()))
      localCallables[entry.getKey()] = entry.getValue();
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
// iter/reversed/filter/enumerate/zip/map as VALUES: the emitter turns each
// into a synthesized generator, and this is that generator's contract.
// nullptr means "not one of these, or not a form this can type" -- the
// caller falls through to the ordinary callee lookup.
mlir::Type lazyIteratorCallType(const TypeSystem &types, llvm::StringRef name,
                                const std::vector<parser::NodePtr> *args);

mlir::Type tupleOfMembers(const TypeSystem &types,
                          llvm::ArrayRef<mlir::Type> members) {
  if (members.empty())
    return types.tupleOf(types.object());
  // ⭐ ONLY A ONE-MEMBER SPELLING IS HOMOGENEOUS. `tuple[T]` means "any number
  // of T" -- it is what `tuple[T, ...]` becomes once the Ellipsis is dropped --
  // so collapsing uniform members into it threw the ARITY away, and the arity
  // is the whole of what a starred call needs: `add(*ys)` for
  // `ys: tuple[int, int]` was refused with "starred call arguments require a
  // statically sized tuple" about a tuple whose size is written in its own
  // annotation. Differing members were already kept positionally; uniform ones
  // now are too, and nothing else distinguishes the two cases.
  if (members.size() == 1)
    return types.tupleOf(members.front());
  return types.contract("builtins.tuple", members);
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
    {"lyrt", "to_prim", "lyrt.to_prim",
     ImportCallableFactory::BuiltinsFunction},
    {"lyrt", "native", "lyrt.native", ImportCallableFactory::BuiltinsFunction},
};

constexpr ModuleAliasImport kModuleAliasImports[] = {
    {"enum", "Enum", "enum.Enum", true},
    {"enum", "IntEnum", "enum.IntEnum", true},
    {"enum", "StrEnum", "enum.StrEnum", true},
    {"enum", "auto", "enum.auto", false},
    {"enum", "unique", "enum.unique", false},
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
    {"lyrt", "to_prim", "lyrt.to_prim",
     ImportCallableFactory::BuiltinsFunction},
    {"lyrt", "native", "lyrt.native", ImportCallableFactory::BuiltinsFunction},
};

constexpr NameAliasImport kNameAliasImports[] = {
    {"lyrt.prim", "Int", "lyrt.prim.Int", true},
    {"lyrt.prim", "Float", "lyrt.prim.Float", true},
    {"lyrt.prim", "Vector", "lyrt.prim.Vector", true},
    {"lyrt.prim", "Matrix", "lyrt.prim.Matrix", true},
    {"lyrt.prim", "Tensor", "lyrt.prim.Tensor", true},
    // Decorator names the emitter recognizes syntactically; the bindings
    // exist so the imports resolve (the decorators never evaluate as values).
    {"dataclasses", "dataclass", "dataclasses.dataclass", false},
    {"dataclasses", "field", "dataclasses.field", false},
    // enum bases/markers the emitter desugars syntactically (same channel as
    // dataclass): an Enum subclass is rewritten into a plain class with
    // compile-time-instantiated members, so these names never evaluate.
    {"enum", "Enum", "enum.Enum", true},
    {"enum", "IntEnum", "enum.IntEnum", true},
    {"enum", "StrEnum", "enum.StrEnum", true},
    {"enum", "auto", "enum.auto", false},
    {"enum", "unique", "enum.unique", false},
    // typing.NamedTuple is a class-construction marker the emitter consumes
    // syntactically (the annotated body desugars like a dataclass), so the
    // binding exists only so the import resolves.
    {"typing", "NamedTuple", "typing.NamedTuple", false},
    {"typing_extensions", "NamedTuple", "typing.NamedTuple", false},
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

// The manifest constants a module exports, bound under the importing name.
// One walk over the three channels: they differ in which export list the
// protocol table hands back and what type the name gets, and in nothing else.
bool bindManifestModuleConstants(TypeSystem &types, llvm::StringRef module,
                                 llvm::StringRef localName) {
  const py::protocols::Table &table =
      py::protocols::Table::get(types.getContext());
  const std::pair<std::vector<std::string>, mlir::Type> channels[] = {
      {table.moduleFloatConstantExports(module), types.floatType()},
      {table.moduleIntConstantExports(module), types.intType()},
      {table.moduleStrConstantExports(module), types.strType()}};

  bool handled = false;
  for (const auto &[exports, type] : channels)
    for (const std::string &exportedName : exports) {
      if (!handled)
        bindManifestModuleObject(types, module, localName);
      handled = true;
      std::string canonical = (llvm::Twine(module) + "." + exportedName).str();
      types.bindCanonicalSymbol(
          importedManifestModuleAttribute(module, localName, exportedName),
          canonical, type);
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

mlir::Type lazyIteratorCallType(const TypeSystem &types, llvm::StringRef name,
                                const std::vector<parser::NodePtr> *args) {
  // Iterator, not types.GeneratorType: the synthesized generator IS a
  // GeneratorType, but a generator value that crosses a function return
  // loses the frame target its manifest __next__ needs ("runtime manifest
  // has no types.GeneratorType.__next__ method"), while the protocol
  // spelling is the one a caller can consume -- and it is what the working
  // annotated form of this same code says.
  auto generatorOf = [&](mlir::Type yielded) -> mlir::Type {
    if (!yielded)
      return {};
    return types.protocol("Iterator", {yielded});
  };
  auto positional = [&](unsigned index) -> const parser::Node * {
    if (!args || index >= args->size() || !(*args)[index] ||
        (*args)[index]->kind == "Starred")
      return nullptr;
    return (*args)[index].get();
  };
  unsigned count = args ? static_cast<unsigned>(args->size()) : 0;

  if (name == "iter" && count == 1) {
    const parser::Node *source = positional(0);
    if (!source)
      return {};
    // iter(gen) is gen: an object that already answers __next__ comes back
    // unchanged, which is what the emitter does too.
    mlir::Type sourceType = types.widenLiteral(types.inferExpr(source));
    if (types.inferMethodCallWithEvidence(sourceType, "__next__", {}))
      return sourceType;
    return generatorOf(types.iterationElementType(source));
  }
  if ((name == "reversed" || name == "filter") &&
      count == (name == "filter" ? 2u : 1u)) {
    const parser::Node *source = positional(name == "filter" ? 1 : 0);
    return source ? generatorOf(types.iterationElementType(source))
                  : mlir::Type();
  }
  if (name == "enumerate" && (count == 1 || count == 2)) {
    const parser::Node *source = positional(0);
    if (!source)
      return {};
    mlir::Type element = types.iterationElementType(source);
    if (!element)
      return {};
    return generatorOf(
        tupleOfMembers(types, {types.contract("builtins.int"), element}));
  }
  if (name == "zip" && count >= 2) {
    llvm::SmallVector<mlir::Type, 4> elements;
    for (unsigned index = 0; index < count; ++index) {
      const parser::Node *source = positional(index);
      if (!source)
        return {};
      mlir::Type element = types.iterationElementType(source);
      if (!element)
        return {};
      elements.push_back(element);
    }
    return generatorOf(tupleOfMembers(types, elements));
  }
  if (name == "map" && count == 2) {
    const parser::Node *callee = positional(0);
    const parser::Node *source = positional(1);
    if (!callee || !source)
      return {};
    mlir::Type element = types.iterationElementType(source);
    if (!element)
      return {};
    // A lambda's own parameter is unannotated, so its type comes from the
    // sequence being mapped -- the same expectation the emitter distributes
    // when it inlines the body. Without it `map(lambda v: v * 2, xs)` inside
    // an unannotated method typed as object, and the object-typed result met
    // the real generator at the ABI as "bundle has 5 values, expects 1".
    if (callee->kind == "Lambda") {
      py::CallableType expected = py::CallableType::get(
          &types.getContext(), {element}, {}, {}, {}, {});
      FunctionSignature lambdaSig =
          types.functionSignature(*callee, std::nullopt, expected);
      return generatorOf(types.widenLiteral(lambdaSig.resultType));
    }
    auto callable = mlir::dyn_cast_if_present<py::CallableType>(
        types.widenLiteral(types.inferExpr(callee)));
    if (!callable || callable.getResultTypes().size() != 1)
      return {};
    return generatorOf(callable.getResultTypes().front());
  }
  return {};
}

} // namespace

bool appendStarredArgumentTypes(const TypeSystem &types, mlir::Type type,
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

TypeSystem::ScopeIsolation::ScopeIsolation(ScopeIsolation &&other) noexcept
    : owner(other.owner), savedScopes(std::move(other.savedScopes)),
      savedCanonicalBindings(std::move(other.savedCanonicalBindings)),
      savedClasses(std::move(other.savedClasses)),
      savedTypeParameters(std::move(other.savedTypeParameters)) {
  other.owner = nullptr;
}

TypeSystem::ScopeIsolation &
TypeSystem::ScopeIsolation::operator=(ScopeIsolation &&other) noexcept {
  if (this == &other)
    return *this;
  reset();
  owner = other.owner;
  savedScopes = std::move(other.savedScopes);
  savedCanonicalBindings = std::move(other.savedCanonicalBindings);
  savedClasses = std::move(other.savedClasses);
  savedTypeParameters = std::move(other.savedTypeParameters);
  other.owner = nullptr;
  return *this;
}

TypeSystem::ScopeIsolation::~ScopeIsolation() { reset(); }

void TypeSystem::ScopeIsolation::reset() {
  if (!owner)
    return;
  owner->scopes = std::move(savedScopes);
  owner->scopedCanonicalBindings = std::move(savedCanonicalBindings);
  owner->scopedClasses = std::move(savedClasses);
  owner->scopedTypeParameters = std::move(savedTypeParameters);
  owner = nullptr;
}

void TypeSystem::bindDeclaredBases(llvm::StringRef name,
                                   llvm::ArrayRef<std::string> bases) {
  declaredBases[name].assign(bases.begin(), bases.end());
}

bool TypeSystem::declaredSubclassOf(llvm::StringRef sub,
                                    llvm::StringRef super) const {
  if (sub == super)
    return true;
  llvm::SmallVector<llvm::StringRef, 8> worklist{sub};
  llvm::StringSet<> seen;
  while (!worklist.empty()) {
    llvm::StringRef current = worklist.pop_back_val();
    auto entry = declaredBases.find(current);
    if (entry == declaredBases.end())
      continue;
    for (const std::string &base : entry->second) {
      if (base == super)
        return true;
      if (seen.insert(base).second)
        worklist.push_back(base);
    }
  }
  return false;
}

TypeSystem::ScopeIsolation TypeSystem::isolateScopes() const {
  ScopeIsolation isolation(*this);
  isolation.savedScopes = std::move(scopes);
  isolation.savedCanonicalBindings = std::move(scopedCanonicalBindings);
  isolation.savedClasses = std::move(scopedClasses);
  isolation.savedTypeParameters = std::move(scopedTypeParameters);
  scopes.clear();
  scopedCanonicalBindings.clear();
  scopedClasses.clear();
  scopedTypeParameters.clear();
  return isolation;
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
  bindSymbol("hash", table.freeFunctionContract("builtins.hash")
                         .value_or(py::CallableType::get(
                             &context, {object()}, {}, {}, {}, {intType()})));
  for (llvm::StringRef manifestBuiltin :
       {"sorted", "abs", "divmod", "pow", "ord", "chr", "hex", "oct", "bin",
        "input"})
    if (std::optional<mlir::Type> manifestContract = table.freeFunctionContract(
            (llvm::Twine("builtins.") + manifestBuiltin).str()))
      bindSymbol(manifestBuiltin, *manifestContract);
  bindClass("object", object());
  bindClass("bool", boolType());
  bindClass("int", intType());
  bindClass("float", floatType());
  bindClass("str", strType());
  bindClass("bytes", contract("builtins.bytes"));
  // ⭐ complex was reachable only as a LITERAL: `1 + 2j` runs, and the manifest
  // has the whole arithmetic surface plus a __new__ that takes two f64 with
  // defaults -- but the NAME was never bound, so `complex(1, 2)` was
  // "unresolved name 'complex'" while the same value one line up was fine.
  bindClass("complex", contract("builtins.complex"));
  bindClass("frozenset", contract("builtins.frozenset"));
  // The whole builtin exception taxonomy binds from the shared table so the
  // emitter's name surface cannot drift from the class-id hierarchy the
  // runtime matches against. Non-builtins members (asyncio.CancelledError,
  // _io.UnsupportedOperation) bind through their module imports instead.
  for (const py::exceptions::BuiltinExceptionInfo &info :
       py::exceptions::kBuiltinExceptions) {
    llvm::StringRef contractName(info.contract);
    if (contractName.starts_with("builtins."))
      bindClass(info.name, contract(contractName));
  }
  // CPython-compatible aliases of OSError.
  bindClass("IOError", contract("builtins.OSError"));
  bindClass("EnvironmentError", contract("builtins.OSError"));
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

mlir::Type
TypeSystem::manifestMethodReceiverContract(mlir::Type typeObject,
                                           llvm::StringRef methodName) const {
  auto typeObjectType = mlir::dyn_cast_if_present<py::TypeType>(typeObject);
  if (!typeObjectType)
    return {};
  auto contract = mlir::dyn_cast_if_present<py::ContractType>(
      typeObjectType.getInstanceType());
  if (!contract || lookupClass(contract.getContractName()))
    return {};
  const py::protocols::Table &table = py::protocols::Table::get(context);
  std::vector<py::protocols::ContractResolution> methods =
      table.methodContractCandidatesWithEvidence(contract, methodName);
  if (methods.empty())
    return {};
  // ⛔ THE FIRST PARAMETER HAS TO BE THE CLASS ITSELF, in EVERY overload. A
  // manifest classmethod or staticmethod is reached through the class too --
  // `int.from_bytes(b, "big")`, `dict.fromkeys(ks)` -- and shifting its first
  // argument into a receiver would call something else entirely. Those already
  // resolve as plain calls and must keep doing so. A method with overloads
  // (`str.strip` declares two) is one method either way, so all of them have
  // to agree before any of them is shifted.
  for (const py::protocols::ContractResolution &candidate : methods) {
    py::CallableType signature = candidate.method.signature;
    if (!signature || signature.getPositionalTypes().empty() ||
        signature.getPositionalTypes().front() != contract)
      return {};
  }
  return contract;
}

std::optional<py::CallableType>
TypeSystem::unboundManifestMethodCallable(mlir::Type typeObject,
                                          llvm::StringRef methodName) const {
  auto typeObjectType = mlir::dyn_cast_if_present<py::TypeType>(typeObject);
  if (!typeObjectType)
    return std::nullopt;
  auto contract = mlir::dyn_cast_if_present<py::ContractType>(
      typeObjectType.getInstanceType());
  if (!manifestMethodReceiverContract(typeObject, methodName))
    return std::nullopt;
  // A leaf `builtins.` contract with no arguments, because the forwarder writes
  // its parameter and result as ANNOTATIONS and only such a type has a spelling
  // to write. `list.copy` returns `list[T]`; it keeps the refusal.
  auto spellable = [](mlir::Type type) {
    auto leaf = mlir::dyn_cast_if_present<py::ContractType>(type);
    if (!leaf || !leaf.getArguments().empty())
      return false;
    llvm::StringRef name = leaf.getContractName();
    return name.consume_front("builtins.") && !name.contains('.');
  };
  if (!spellable(contract))
    return std::nullopt;
  const py::protocols::Table &table = py::protocols::Table::get(context);
  std::vector<py::protocols::ContractResolution> methods =
      table.methodContractCandidatesWithEvidence(contract, methodName);
  // The overload a zero-argument call reaches is the SHORTEST one -- that is
  // the resolution `__ly_recv.strip()` will get, so it is the one whose result
  // this callable has to name.
  py::CallableType signature;
  for (const py::protocols::ContractResolution &candidate : methods) {
    py::CallableType option = candidate.method.signature;
    if (!option)
      continue;
    if (!signature || option.getPositionalTypes().size() <
                          signature.getPositionalTypes().size())
      signature = option;
  }
  if (!signature || signature.getPositionalTypes().empty() ||
      signature.hasVararg() || signature.hasKwarg() ||
      !signature.getKwOnlyTypes().empty() ||
      signature.getResultTypes().size() != 1)
    return std::nullopt;
  // Every parameter past the receiver has to be optional: the forwarder calls
  // the method with no arguments and lets the running body fill the rest, which
  // is what `key=str.strip` means and all a `key=` can pass.
  //
  // ⛔ A signature with no parameter metadata at all -- most of them, and
  // `str.lower` among them -- says every parameter is REQUIRED, so it passes
  // only when the receiver is the whole list. Reading an absent `arg_defaults`
  // as "all optional" would let a two-argument method through and the
  // forwarder would then call it wrong.
  llvm::ArrayRef<mlir::BoolAttr> defaults = signature.getPositionalDefaults();
  llvm::ArrayRef<mlir::Type> positional = signature.getPositionalTypes();
  for (unsigned index = 1; index < positional.size(); ++index)
    if (index >= defaults.size() || !defaults[index].getValue())
      return std::nullopt;
  mlir::Type result = signature.getResultTypes().front();
  if (!spellable(result))
    return std::nullopt;
  mlir::Type receiver = contract;
  return py::CallableType::get(&context, {receiver}, {}, {}, {}, {result});
}
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

mlir::Type TypeSystem::coroutineOf(mlir::Type resultType) const {
  return contract("types.CoroutineType",
                  {any(), any(), resultType ? resultType : any()});
}

namespace {

void collectNameReferences(const parser::Node *node, llvm::StringSet<> &out) {
  ast::walk(node, [&](const parser::Node &current) {
    if (current.kind == "Name")
      if (auto id = ast::string(current, "id"))
        out.insert(*id);
    return ast::Walk::Continue;
  });
}

// Call sites in module-level statement position, used by the inference
// fixpoint to constrain unannotated parameters. Function, lambda, and class
// subtrees are excluded: their expressions type under scopes this walk does
// not know, and a lenient mis-typing here would pollute the union-find store
// with wrong parameter bindings.
void collectModuleCallNodes(const parser::Node *node,
                            std::vector<const parser::Node *> &out) {
  ast::walk(node, [&](const parser::Node &current) {
    if (current.kind == "FunctionDef" || current.kind == "AsyncFunctionDef" ||
        current.kind == "Lambda" || current.kind == "ClassDef")
      return ast::Walk::SkipChildren;
    if (current.kind == "Call")
      out.push_back(&current);
    return ast::Walk::Continue;
  });
}

} // namespace

void TypeSystem::forgetSignature(const parser::Node *function) {
  if (function)
    signatureMemo.erase(function);
}

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
    llvm::SaveAndRestore<bool> duringFixpoint(defaultsDescribeParameters,
                                              memoize);
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
  if (const auto *statements = ast::nodeList(moduleNode, "body"))
    for (const parser::NodePtr &statement : *statements) {
      if (!statement || (statement->kind != "FunctionDef" &&
                         statement->kind != "AsyncFunctionDef"))
        continue;
      auto decorated = ast::string(*statement, "name");
      const auto *decorators = ast::nodeList(*statement, "decorator_list");
      if (!decorated || !decorators)
        continue;
      for (const parser::NodePtr &decorator : *decorators) {
        if (!decorator || decorator->kind != "Name")
          continue;
        std::vector<parser::NodePtr> arguments;
        arguments.push_back(synth::name(*decorated, statement->range));
        decoratorCallNodes.push_back(synth::call(
            synth::name(ast::nameSpelling(*decorator), statement->range),
            std::move(arguments), statement->range));
        moduleCalls.push_back(decoratorCallNodes.back().get());
      }
    }

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
  scopedTypeParameters.emplace_back();
  return Scope(*this);
}

void TypeSystem::popScope() const {
  if (!scopes.empty()) {
    scopes.pop_back();
    scopedCanonicalBindings.pop_back();
    scopedClasses.pop_back();
    scopedTypeParameters.pop_back();
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

void TypeSystem::bindLocalTypeParameter(llvm::StringRef name,
                                        mlir::Type type) const {
  if (scopedTypeParameters.empty() || !type)
    return;
  scopedTypeParameters.back()[name] = type;
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

void TypeSystem::bindRootSymbol(llvm::StringRef name, mlir::Type type) {
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

void TypeSystem::setGenericClassResolver(GenericClassResolver resolver) {
  genericClassResolver = std::move(resolver);
}

mlir::Type
TypeSystem::resolveGenericClass(llvm::StringRef baseName,
                                mlir::ArrayRef<mlir::Type> arguments) const {
  if (!genericClassResolver || arguments.empty())
    return {};
  // A non-ground argument is not an instantiation: it is the generic body
  // still talking about its own parameters, so the parameterized reading
  // stands and no specialization is allocated.
  for (mlir::Type argument : arguments) {
    if (!argument || unboundStaticParameterCount(argument) != 0)
      return {};
    bool unresolved = false;
    py::mapPyTypeStructure(argument,
                           [&](mlir::Type node) -> std::optional<mlir::Type> {
                             if (py::isPyInferVarType(node))
                               unresolved = true;
                             return std::nullopt;
                           });
    if (unresolved)
      return {};
  }
  return genericClassResolver(baseName, arguments);
}

void TypeSystem::registerGenericClass(
    llvm::StringRef contractName, llvm::ArrayRef<std::string> params,
    const parser::Node *initNode, llvm::ArrayRef<GenericClassField> fields) {
  GenericClassTemplate &tmpl = genericClassTemplates[contractName];
  tmpl.params.assign(params.begin(), params.end());
  tmpl.initNode = initNode;
  tmpl.fields.assign(fields.begin(), fields.end());
}

mlir::Type TypeSystem::solveGenericClassInstantiation(
    llvm::StringRef contractName, mlir::ArrayRef<mlir::Type> positional,
    mlir::ArrayRef<CallKeywordType> keywords) const {
  auto tmpl = genericClassTemplates.find(contractName);
  if (tmpl == genericClassTemplates.end())
    return {};
  // The parameters have to stand as TypeVars for the match to have anything
  // to solve: without the binding, annotationTypeForName would read a bare
  // `T` as a class named T and fabricate a `builtins.T` contract.
  auto scope = pushScope();
  for (const std::string &param : tmpl->second.params)
    bindLocalSymbol(param, py::TypeVarType::get(&context, param));

  // The constructor's parameters, by name, in call order.
  llvm::SmallVector<std::pair<llvm::StringRef, mlir::Type>, 8> formals;
  FunctionSignature init;
  if (tmpl->second.initNode) {
    init = functionSignature(*tmpl->second.initNode, llvm::StringRef("self"));
    // positionalTypes[0] is the receiver, which is not an argument.
    for (auto [index, name] : llvm::enumerate(init.positionalNames)) {
      if (index == 0 || index >= init.positionalTypes.size())
        continue;
      formals.emplace_back(name, init.positionalTypes[index]);
    }
    for (auto [index, name] : llvm::enumerate(init.kwOnlyNames))
      if (index < init.kwOnlyTypes.size())
        formals.emplace_back(name, init.kwOnlyTypes[index]);
  } else {
    for (const GenericClassField &field : tmpl->second.fields)
      formals.emplace_back(field.first, annotationType(field.second));
  }

  TypeBindingMap bindings;
  for (auto [index, argument] : llvm::enumerate(positional)) {
    if (index >= formals.size())
      break;
    bindExpectedType(*this, formals[index].second, widenLiteral(argument),
                     bindings);
  }
  for (const CallKeywordType &keyword : keywords)
    for (const auto &formal : formals)
      if (formal.first == keyword.name)
        bindExpectedType(*this, formal.second, widenLiteral(keyword.type),
                         bindings);

  llvm::SmallVector<mlir::Type, 4> arguments;
  for (const std::string &param : tmpl->second.params) {
    auto solved = bindings.find(param);
    if (solved == bindings.end())
      return {};
    arguments.push_back(solved->second);
  }
  return resolveGenericClass(contractName, arguments);
}

mlir::Type
TypeSystem::genericClassSubscript(const parser::Node *node) const {
  if (!genericClassResolver || !node || node->kind != "Subscript")
    return {};
  const parser::Node *base = ast::node(*node, "value");
  if (!base || (base->kind != "Name" && base->kind != "Attribute"))
    return {};
  std::string qualified = ast::qualifiedName(base);
  std::string_view spelling = ast::nameSpelling(*base);
  std::string resolved = resolveAnnotationName(
      qualified.empty() ? llvm::StringRef(spelling.data(), spelling.size())
                        : llvm::StringRef(qualified));
  std::optional<mlir::Type> knownClass = lookupClass(resolved);
  auto contractType = knownClass
                          ? mlir::dyn_cast_if_present<py::ContractType>(*knownClass)
                          : py::ContractType();
  if (!contractType)
    return {};
  const parser::Node *slice = ast::node(*node, "slice");
  llvm::SmallVector<mlir::Type, 4> arguments;
  if (slice && slice->kind == "Tuple") {
    if (const auto *elts = ast::nodeList(*slice, "elts"))
      for (const parser::NodePtr &elt : *elts)
        arguments.push_back(annotationType(elt.get()));
  } else if (slice) {
    arguments.push_back(annotationType(slice));
  }
  return resolveGenericClass(contractType.getContractName(), arguments);
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

void TypeSystem::bindClassStaticAttr(llvm::StringRef className,
                                     llvm::StringRef attrName,
                                     mlir::Type type) {
  if (!type)
    return;
  classStaticAttrTypes[(llvm::Twine(className) + "." + attrName).str()] = type;
}

std::optional<mlir::Type>
TypeSystem::lookupClassStaticAttrType(llvm::StringRef className,
                                      llvm::StringRef attrName) const {
  auto found =
      classStaticAttrTypes.find((llvm::Twine(className) + "." + attrName).str());
  if (found == classStaticAttrTypes.end())
    return std::nullopt;
  return found->second;
}

void TypeSystem::bindClassPropertyType(llvm::StringRef className,
                                       llvm::StringRef propertyName,
                                       mlir::Type type) {
  if (className.empty() || propertyName.empty() || !type)
    return;
  classPropertyTypes[(className + "." + propertyName).str()] = type;
}

std::optional<mlir::Type>
TypeSystem::lookupClassPropertyType(llvm::StringRef className,
                                    llvm::StringRef propertyName) const {
  auto found = classPropertyTypes.find((className + "." + propertyName).str());
  if (found == classPropertyTypes.end())
    return std::nullopt;
  return found->second;
}

void TypeSystem::bindClassStaticMethod(llvm::StringRef className,
                                       llvm::StringRef methodName,
                                       mlir::Type callable) {
  if (!callable)
    return;
  classStaticMethodTypes[(llvm::Twine(className) + "." + methodName).str()] =
      callable;
}

std::optional<mlir::Type>
TypeSystem::lookupClassStaticMethod(llvm::StringRef className,
                                    llvm::StringRef methodName) const {
  auto found = classStaticMethodTypes.find(
      (llvm::Twine(className) + "." + methodName).str());
  if (found == classStaticMethodTypes.end())
    return std::nullopt;
  return found->second;
}

bool TypeSystem::isImportedModuleName(llvm::StringRef name) const {
  return importedModuleLocalNames.contains(name);
}

void TypeSystem::noteImportedModuleName(llvm::StringRef name) {
  importedModuleLocalNames.insert(name);
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
    if (!handled) {
      bindSymbol(localName, object());
      importedModuleLocalNames.insert(localName);
    }
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
  if (bindManifestModuleConstants(*this, module, localName))
    handled = true;

  // A manifest module (math, os, time, json ...) binds its exports without
  // going through `bindModuleObject`, so the local name has to be recorded
  // here as well -- it is what tells an unresolvable attribute on it apart
  // from one on an ordinary object.
  if (handled)
    importedModuleLocalNames.insert(localName);
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

mlir::Type TypeSystem::annotationTypeForName(llvm::StringRef rawName) const {
  std::string resolved = resolveAnnotationName(rawName);
  llvm::StringRef name(resolved);
  // A specialization's solved type parameters win over every other reading of
  // the name: inside `def f[T](...)`'s specialized body, `T` in an annotation
  // denotes THIS instantiation's ground type, and nothing else may claim the
  // spelling for the duration.
  for (auto it = scopedTypeParameters.rbegin(), e = scopedTypeParameters.rend();
       it != e; ++it) {
    auto found = it->find(name);
    if (found != it->end())
      return found->second;
  }
  if (auto symbol = lookupSymbol(name)) {
    if (mlir::isa<py::TypeVarType, py::ParamSpecType, py::TypeVarTupleType>(
            *symbol))
      return *symbol;
    // Compiler-synthesized type aliases (emitter rewrites bind a concrete
    // inferred type under a reserved "__ly*" name and spell it in
    // synthesized annotations). User names never take this path: a plain
    // local would otherwise shadow a class annotation.
    if (name.starts_with("__ly"))
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
  // ⭐ A CLASS THE PROGRAM DECLARES WINS OVER A MANIFEST NAME. Five bare
  // spellings are claimed by manifest contracts whether or not anything
  // imported them -- `Task`, `Future`, `AbstractEventLoop`, `CancelledError`,
  // `Context` -- so `class Task` followed by `def top(ts: list[Task])` typed
  // the parameter as asyncio's Task and the call was refused with "arguments
  // do not match Callable contract for function target top", naming neither
  // the class nor the collision. `Task` is an ordinary name for an ordinary
  // class, and in Python the module-level binding shadows anything a name
  // could otherwise mean.
  //
  // ⛔ The PROTOCOL spellings are claimed the same way -- `Sequence`,
  // `Iterator`, `Generator` and eleven more -- and letting a declared class
  // win over those was tried and REVERTED: the emitter's own iteration typing
  // asks this function for `Iterator`, so `class Iterator` in a program broke
  // every `for` loop in it with "static type !py.protocol<"Iterator", [...]>
  // does not provide manifest method '__next__'". A user class of that name
  // keeps the old refusal until the compiler's internal spellings are ones a
  // program cannot shadow.
  //
  // ⛔ Bare names only: `asyncio.Task` spelled with its module still means the
  // manifest contract, `collections.abc.Sequence` still means the protocol,
  // and so does a name inside a runtime module that declares no class of that
  // name.
  if (auto protocolName = protocolAnnotationName(name))
    return protocol(*protocolName);
  bool bare = !name.contains('.');
  if (bare)
    if (auto declared = lookupClass(name))
      return *declared;
  if (auto contractName = contractAnnotationName(name))
    return contract(*contractName);
  if (auto knownClass = lookupClass(name))
    return *knownClass;
  return contract((llvm::Twine("builtins.") + name).str());
}

parser::Diagnostics TypeSystem::takeAnnotationDiagnostics() {
  parser::Diagnostics drained = std::move(annotationDiagnostics);
  annotationDiagnostics.clear();
  return drained;
}

// The element an iteration over `node` yields: a generator expression infers
// its element expression under progressively bound chain targets (like a
// comprehension), a plain iterable goes through __iter__/__next__.
//
// ⭐ Shared by the reducer folds in the emitter and by the INFERENCE of those
// same builtins, so the two cannot answer differently. They did: the emitter
// folded `max(sum(r) for r in rows)` and the inference had nothing to say
// about the inner `sum(r)`, so the outer fold was refused for an element type
// it could not see.
mlir::Type TypeSystem::iterationElementType(const parser::Node *arg) const {
  if (!arg)
    return {};
  auto iterationElement = [&](const parser::Node *iterable) -> mlir::Type {
    mlir::Type iterableType = inferExpr(iterable);
    if (!iterableType)
      return {};
    CallInferenceResult iterInference = inferMethodCallWithEvidence(
        widenLiteral(iterableType), "__iter__", {});
    if (!iterInference) {
      // ⭐ THE SEQUENCE PROTOCOL. A class with `__len__` and `__getitem__` and
      // no `__iter__` is iterable -- CPython's `iter()` falls back to indexing
      // from 0 -- and the element is what the subscript answers. The loop
      // rewrite that runs it lives in EmitterLoops.cpp; this is the same rule
      // for the walks that only need the TYPE (a comprehension's target, a
      // reducer's accumulator).
      if (inferMethodCallWithEvidence(widenLiteral(iterableType), "__len__",
                                      {}))
        if (CallInferenceResult indexed = inferMethodCallWithEvidence(
                widenLiteral(iterableType), "__getitem__", {intType()}))
          return widenLiteral(indexed.resultType);
      return {};
    }
    CallInferenceResult nextInference = inferMethodCallWithEvidence(
        iterInference.resultType, "__next__", {});
    if (!nextInference)
      return {};
    return widenLiteral(nextInference.resultType);
  };
  if (arg->kind != "GeneratorExp")
    return iterationElement(arg);
  const parser::Field *eltField = parser::findField(*arg, "elt");
  const auto *gens = ast::nodeList(*arg, "generators");
  if (!eltField ||
      !std::holds_alternative<parser::NodePtr>(eltField->value) || !gens)
    return {};
  auto scope = pushScope();
  for (const parser::NodePtr &gen : *gens) {
    if (!gen)
      return {};
    const parser::Node *target = ast::node(*gen, "target");
    const parser::Node *iter = ast::node(*gen, "iter");
    if (!target || !iter)
      return {};
    mlir::Type elementType = iterationElement(iter);
    if (!elementType)
      return {};
    if (target->kind == "Name") {
      bindLocalSymbol(ast::nameSpelling(*target), elementType);
      continue;
    }
    // ⭐ A TUPLE target binds member-wise, the way the loop does. Without
    // it the element type came back empty and `max(a + b for a, b in
    // pairs)` was refused for an element type it could not see -- while
    // the loop and the list comprehension over the same pairs both bind.
    if (target->kind != "Tuple" && target->kind != "List")
      return {};
    const auto *names = ast::nodeList(*target, "elts");
    auto elementContract =
        mlir::dyn_cast_if_present<py::ContractType>(elementType);
    if (!names || names->empty() || !elementContract ||
        elementContract.getContractName() != "builtins.tuple")
      return {};
    llvm::ArrayRef<mlir::Type> members = elementContract.getArguments();
    for (auto [position, name] : llvm::enumerate(*names)) {
      if (!name || name->kind != "Name" || members.empty())
        return {};
      // A uniform `tuple[T]` gives every position the same member.
      mlir::Type memberType =
          members.size() == 1 ? members.front()
                              : (position < members.size()
                                     ? members[position]
                                     : mlir::Type());
      if (!memberType)
        return {};
      bindLocalSymbol(ast::nameSpelling(*name), memberType);
    }
  }
  return widenLiteral(inferExpr(
      std::get<parser::NodePtr>(eltField->value).get()));
}

mlir::Type TypeSystem::annotationType(const parser::Node *node) const {
  if (!node)
    return object();
  if (node->kind == "Name")
    return annotationTypeForName(ast::nameSpelling(*node));
  if (node->kind == "Constant") {
    if (isNoneConstant(node))
      return none();
    // PEP 484 string annotation: the text is resolved lazily as a type
    // reference. Only a simple (optionally dotted) name is accepted — the
    // classes it can name are all predeclared before bodies are typed, so
    // "lazy" needs no second pass; a complex expression inside the string
    // would need real deferred evaluation and is rejected loudly instead of
    // silently typing as a str literal.
    if (std::optional<std::string_view> text = ast::string(*node, "value")) {
      llvm::StringRef name = llvm::StringRef(text->data(), text->size()).trim();
      auto isSimpleName = [](llvm::StringRef candidate) {
        if (candidate.empty())
          return false;
        llvm::SmallVector<llvm::StringRef, 4> parts;
        candidate.split(parts, '.');
        for (llvm::StringRef part : parts) {
          if (part.empty())
            return false;
          if (!llvm::isAlpha(part.front()) && part.front() != '_')
            return false;
          for (char ch : part)
            if (!llvm::isAlnum(ch) && ch != '_')
              return false;
        }
        return true;
      };
      if (isSimpleName(name))
        return annotationTypeForName(name);
      // ⭐ A union of simple names resolves on the same terms. Every class it
      // can name is predeclared before bodies are typed, which is what makes
      // the simple-name case need no second pass -- splitting on `|` does not
      // change that. `self.next: "Node | None" = None` is how a
      // self-referential node type is spelled (the class is not bound yet at
      // its own body, so the annotation MUST be a string), and it was refused
      // while the unquoted `int | None` resolved one line away.
      if (name.contains('|')) {
        llvm::SmallVector<llvm::StringRef, 4> members;
        name.split(members, '|');
        llvm::SmallVector<mlir::Type, 4> resolved;
        bool everyMemberSimple = !members.empty();
        for (llvm::StringRef member : members) {
          member = member.trim();
          if (member == "None") {
            resolved.push_back(none());
            continue;
          }
          if (!isSimpleName(member)) {
            everyMemberSimple = false;
            break;
          }
          resolved.push_back(annotationTypeForName(member));
        }
        if (everyMemberSimple)
          return py::UnionType::getNormalized(&context, resolved);
      }
      // ⭐ ANYTHING ELSE IS PARSED. `"list[int]"`, `"tuple[int, str]"` and
      // `"Callable[[int], int]"` are ordinary annotations that happen to be
      // quoted -- a method returning its own class writes the second one, and
      // an annotation under `from __future__ import annotations` is a string
      // for every type there is. They resolved to `object`, and the program
      // then failed as "static type builtins.object does not provide ...",
      // which names neither the annotation nor the quoting.
      //
      // ⛔ The TREE is cached, not the type: the walk below reads the nodes,
      // so they must outlive this call. And the diagnostics the walk files
      // carry positions inside the annotation TEXT, which is a different file
      // from the program -- they are re-pointed at the string itself, or a
      // reader gets a line number from a source that does not exist.
      auto cached = parsedAnnotations.find(name);
      if (cached == parsedAnnotations.end()) {
        parser::ParseResult parsed =
            parser::parse(name, "<annotation>",
                          parser::ParseOptions{parser::ParseMode::Expression,
                                               /*typeComments=*/false});
        parsedAnnotations[name] =
            parsed.diagnostics.empty() ? parsed.tree : parser::NodePtr();
        cached = parsedAnnotations.find(name);
      }
      if (cached->second) {
        const parser::Node *body = ast::node(*cached->second, "body");
        if (body && body->kind != "Constant") {
          std::size_t before = annotationDiagnostics.size();
          mlir::Type resolved = annotationType(body);
          for (std::size_t index = before; index < annotationDiagnostics.size();
               ++index)
            annotationDiagnostics[index].location = node->range.start;
          if (resolved && resolved != object())
            return resolved;
        }
      }
      parser::Diagnostic diagnostic{
          parser::Severity::Error, node->range.start,
          "string annotation \"" + name.str() +
              "\" does not resolve to a type; a quoted annotation is parsed "
              "as the annotation it spells, so it must be one"};
      bool duplicate = false;
      for (const parser::Diagnostic &existing : annotationDiagnostics)
        if (existing.location.line == diagnostic.location.line &&
            existing.location.column == diagnostic.location.column &&
            existing.message == diagnostic.message)
          duplicate = true;
      if (!duplicate)
        annotationDiagnostics.push_back(std::move(diagnostic));
      return object();
    }
    return literal(literalSpelling(*node));
  }
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
    if (annotationNameIs(baseName, "set") || annotationNameIs(baseName, "Set"))
      return contract("builtins.set", {annotationType(slice)});
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
    if (annotationNameIs(baseName, "Literal")) {
      if (slice && slice->kind == "Constant")
        return literal(literalSpelling(*slice));
      // ⭐ `Literal[a, b]` IS THE JOIN OF ITS MEMBERS, and the fallback below
      // made it `!py.literal<object>` -- a literal whose spelling is the word
      // "object", which no operation and no call accepts:
      //
      //     def go(mode: Literal["a", "b"]) -> int: ...
      //     go("a")   # is not callable: call arguments do not match the
      //               # Callable contract
      //
      // A one-member `Literal["a"]` has always worked, which is what says the
      // tuple slice is the gap and not the annotation.
      //
      // ⛔ The members are WIDENED before the join, so `Literal[1, 2]` is
      // `int` rather than a union of two int literals: a union would have to
      // answer `__mul__` to be usable, and every value the annotation admits
      // is an int. A mixed `Literal[1, "a"]` still joins to `int | str`, which
      // is what it means.
      if (slice && slice->kind == "Tuple")
        if (const auto *elts = ast::nodeList(*slice, "elts")) {
          llvm::SmallVector<mlir::Type, 4> members;
          for (const parser::NodePtr &elt : *elts)
            members.push_back(elt && elt->kind == "Constant"
                                  ? widenLiteral(literal(literalSpelling(*elt)))
                                  : object());
          if (!members.empty())
            return join(members);
        }
      return object();
    }
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
              mlir::dyn_cast_if_present<py::ContractType>(*knownClass)) {
        if (mlir::Type specialized = resolveGenericClass(
                contractType.getContractName(), arguments))
          return specialized;
        return contract(contractType.getContractName(), arguments);
      }
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
  // f-strings always evaluate to str; the emitter enforces that each
  // interpolation is statically formattable.
  if (node->kind == "JoinedStr" || node->kind == "FormattedValue")
    return contract("builtins.str");
  // ⭐ A COMPREHENSION HAS A TYPE. This walk had no arm for one, so every
  // question about a comprehension USED DIRECTLY got `object`, and the sugar
  // that answers `keys()`/`values()`/`items()` asks exactly that question:
  //
  //     print(sorted({x: x * x for x in xs}.items()))
  //     # runtime manifest has no builtins.dict.items method
  //
  // Binding it to a name first worked, and so did the same call on a dict
  // LITERAL temporary -- the receiver's TYPE is the whole difference. The
  // emitter builds the value correctly either way; nothing could say what it
  // was.
  //
  // ⛔ Only Name targets, and only when every part infers to something
  // concrete: a chained comprehension's second `iter` may mention the first
  // target, which this does not bind, so it falls back to `object` rather
  // than guessing. That is the answer it gave before.
  if (node->kind == "ListComp" || node->kind == "SetComp" ||
      node->kind == "DictComp") {
    const auto *generators = ast::nodeList(*node, "generators");
    if (!generators || generators->empty())
      return object();
    llvm::StringMap<mlir::Type> bound;
    if (ctx && ctx->localSymbols)
      bound = *ctx->localSymbols;
    // A TUPLE target is the `for i, n in rows` shape, which is most of the
    // comprehensions written over a list of pairs. Distributed the same way
    // the generator walk distributes one: positionally from a positional
    // tuple, uniformly from a one-argument container.
    auto bindTarget = [&](const parser::Node *target, mlir::Type element,
                          auto &&recurse) -> bool {
      if (!target || !element || widenLiteral(element) == object())
        return false;
      if (target->kind == "Name") {
        bound[ast::nameSpelling(*target)] = element;
        return true;
      }
      if (target->kind != "Tuple" && target->kind != "List")
        return false;
      const auto *elements = ast::nodeList(*target, "elts");
      if (!elements || elements->empty())
        return false;
      auto contract =
          mlir::dyn_cast_if_present<py::ContractType>(widenLiteral(element));
      if (!contract)
        return false;
      llvm::ArrayRef<mlir::Type> arguments = contract.getArguments();
      if (arguments.size() == elements->size()) {
        for (auto [index, part] : llvm::enumerate(*elements))
          if (!recurse(part.get(), arguments[index], recurse))
            return false;
        return true;
      }
      if (arguments.size() != 1)
        return false;
      for (const parser::NodePtr &part : *elements)
        if (!recurse(part.get(), arguments.front(), recurse))
          return false;
      return true;
    };
    // ⭐ THE SECOND GENERATOR ITERATES THE FIRST ONE'S TARGET. `{y for x in
    // rows for y in x}` asks the element type of `x`, which is bound by the
    // generator before it and nowhere else -- so the walk answered `object`
    // for the whole comprehension and `sorted(...)` over it was refused,
    // while the same comprehension with ONE generator typed fine and `len()`
    // of the nested one worked (it needs no element type). The generator
    // EXPRESSION branch of `iterationElementType` already binds them into a
    // pushed scope as it goes; this is the same thing for the comprehension
    // forms.
    auto generatorScope = pushScope();
    for (const parser::NodePtr &generator : *generators) {
      if (!generator)
        return object();
      mlir::Type element = iterationElementType(ast::node(*generator, "iter"));
      if (!bindTarget(ast::node(*generator, "target"), element, bindTarget))
        return object();
      for (const auto &entry : bound)
        bindLocalSymbol(entry.getKey(), entry.getValue());
    }
    static const llvm::StringMap<mlir::Type> kNoCallables;
    ExprInferenceContext inner{ctx ? ctx->localCallables : kNoCallables,
                               nullptr, &bound, /*strict=*/false};
    auto part = [&](const parser::Node *child) -> mlir::Type {
      mlir::Type inferred = widenLiteral(inferExprImpl(child, &inner));
      return inferred == object() ? mlir::Type() : inferred;
    };
    if (node->kind == "DictComp") {
      mlir::Type key = part(ast::node(*node, "key"));
      mlir::Type value = part(ast::node(*node, "value"));
      if (!key || !value)
        return object();
      return contract("builtins.dict", {key, value});
    }
    mlir::Type element = part(ast::node(*node, "elt"));
    if (!element)
      return object();
    return node->kind == "ListComp" ? listOf(element)
                                    : contract("builtins.set", {element});
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
        // ⭐ `C.__name__` is the string constant the emitter folds it to, and
        // this channel has to say so too. Without it the emitter's fold was
        // invisible to every consumer that asks the TYPE first: a list of two of
        // them joined to `list[object]` ("a type-erased `object` value cannot be
        // stored in a runtime container slot"), because the join is computed from
        // the inferred element types, not from the emitted values.
        // ⭐ A FUNCTION'S `__name__` IS A str TO THIS CHANNEL TOO. The emitter
        // folds it to the def's name; without an arm here the read answered
        // `object`, and a container of two of them joined to `list[object]` --
        // "a type-erased `object` value cannot be stored in a runtime container
        // slot" -- for values the emitter had already made strings.
        if (*attr == "__name__") {
          if (mlir::isa_and_nonnull<py::CallableType>(widenLiteral(objectType)))
            return strType();
          // `C.m.__name__`: the method read does not always come back as a
          // callable through this channel, and the emitter folds it either way.
          if (const parser::Node *owner = ast::node(*node, "value"))
            if (owner->kind == "Attribute")
              if (const parser::Node *root = ast::node(*owner, "value"))
                if (root->kind == "Name" &&
                    lookupClass(ast::nameSpelling(*root)) != mlir::Type())
                  return strType();
        }
        if (*attr == "__name__")
          if (auto typeObjectType = mlir::dyn_cast_if_present<py::TypeType>(
                  widenLiteral(objectType)))
            if (auto contractType = mlir::dyn_cast_if_present<py::ContractType>(
                    typeObjectType.getInstanceType())) {
              llvm::StringRef qualifiedName = contractType.getContractName();
              llvm::StringRef simple = qualifiedName;
              if (auto dot = qualifiedName.rfind('.');
                  dot != llvm::StringRef::npos)
                simple = qualifiedName.drop_front(dot + 1);
              return literal("\"" + simple.str() + "\"");
            }
        if (std::optional<py::protocols::FieldResolution> field =
                table.resolveFieldContractWithEvidence(widenLiteral(objectType),
                                                       *attr))
          return field->contractType;
        // Class static attributes read through the class object or through an
        // instance (an instance field of the same name shadows them, and the
        // field channel above already claimed that case).
        mlir::Type receiverInstance = widenLiteral(objectType);
        if (auto typeObjectType =
                mlir::dyn_cast_if_present<py::TypeType>(receiverInstance))
          receiverInstance = typeObjectType.getInstanceType();
        if (auto contractType =
                mlir::dyn_cast_if_present<py::ContractType>(receiverInstance)) {
          if (std::optional<mlir::Type> staticAttr = lookupClassStaticAttrType(
                  contractType.getContractName(), *attr))
            return *staticAttr;
          // ⭐ A @property is neither a field nor a manifest method, so this
          // walk used to fall past it to `object()` below. That answer is what
          // `str(x)` reads to choose its dispatch, and an erased object routes
          // to the manifest `object.__str__`, which reads a payload class id a
          // source instance's header does not carry -- `str(Path("/x").parent)`
          // SEGFAULTED. The emitter already resolves the read itself, which is
          // why binding it to a name first worked and using it directly did
          // not.
          if (std::optional<mlir::Type> property = lookupClassPropertyType(
                  contractType.getContractName(), *attr))
            return *property;
          // ⭐ A METHOD READ WITHOUT A CALL IS THE BOUND METHOD, not the value
          // that calling it would give. The arm below answers with the RESULT
          // of a zero-argument call, which is right for the manifest reads
          // this compiler folds to a value and wrong for a method used as one:
          // the emitter builds the bound object (emitMethodObject), so
          // `m = c.go` then `m()` worked, while `[c.go]` -- whose element type
          // comes from this channel -- was `list[int]` and then "builtins.int
          // is not callable". A method that takes an argument did not even
          // reach the arm, because a zero-argument call does not resolve it.
          //
          // ⛔ Source classes and an INSTANCE receiver only: `C.go` read off
          // the class is the plain function, which binds nothing.
          if (!mlir::isa<py::TypeType>(widenLiteral(objectType)) &&
              lookupClass(contractType.getContractName())) {
            std::vector<py::protocols::ContractResolution> methods =
                table.methodContractCandidatesWithEvidence(receiverInstance,
                                                           *attr);
            if (methods.size() == 1 &&
                !methods.front().method.signature.getPositionalTypes().empty())
              return py::protocols::bindReceiverCallable(
                  methods.front().method.signature);
          }
        }
        if (std::optional<CallSolution> method =
                tryManifestMethod(*this, widenLiteral(objectType), *attr, {}))
          return method->result;
        // ⭐ AN UNBOUND METHOD READ OFF A MANIFEST CLASS IS A CALLABLE. The
        // emitter synthesizes a forwarder for it (`str.lower` as a `key=`), and
        // this is the same question so the two cannot disagree.
        if (std::optional<py::CallableType> unbound =
                unboundManifestMethodCallable(widenLiteral(objectType), *attr))
          return *unbound;
      }
      return object();
    }
  }
  if (node->kind == "List" || node->kind == "Tuple") {
    llvm::SmallVector<mlir::Type, 8> elementTypes;
    llvm::SmallVector<const parser::Node *, 8> elementNodes;
    if (const auto *elements = ast::nodeList(*node, "elts")) {
      elementTypes.reserve(elements->size());
      for (const parser::NodePtr &element : *elements) {
        mlir::Type elementType = recurse(element.get());
        if (strict && !elementType)
          return {};
        elementTypes.push_back(widenLiteral(elementType));
        elementNodes.push_back(element.get());
      }
    }
    if (node->kind == "List")
      return listOf(joinIgnoringEmptyLiterals(*this, elementTypes,
                                              elementNodes));
    // Both paths type heterogeneous tuples positionally: the joined
    // (homogeneous-union) view made `yield (i, "x")` infer as
    // tuple[int | str], whose literal-index __getitem__ result is a union
    // the runtime element rebuild cannot shape.
    return tupleOfMembers(*this, elementTypes);
  }
  if (node->kind == "Set") {
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
    return py::ContractType::get(&context, "builtins.set",
                                 {join(elementTypes)});
  }
  if (node->kind == "Dict") {
    const auto *keys = ast::nodeList(*node, "keys");
    const auto *values = ast::nodeList(*node, "values");
    if (!keys || !values)
      return dictOf(object(), object());
    llvm::SmallVector<mlir::Type, 8> keyTypes;
    llvm::SmallVector<mlir::Type, 8> valueTypes;
    llvm::SmallVector<const parser::Node *, 8> valueNodes;
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
        valueNodes.push_back((*values)[index].get());
      }
    }
    return dictOf(join(keyTypes),
                  joinIgnoringEmptyLiterals(*this, valueTypes, valueNodes));
  }
  if (node->kind == "Subscript") {
    if (std::optional<PrimitiveTypeSpec> primitive =
            primitiveTypeSpecFromSubscript(node, *this))
      return typeObject(primitive->type);
    // `C[int]` over a generic class is the instantiation's class object, not
    // a __getitem__ on a value.
    if (mlir::Type instantiated = genericClassSubscript(node))
      return typeObject(instantiated);
    mlir::Type container = recurse(ast::node(*node, "value"));
    // A shaped primitive indexes down to its element type; there is no
    // manifest __getitem__ behind it to infer through.
    if (auto tensor = mlir::dyn_cast_or_null<mlir::RankedTensorType>(container))
      return tensor.getElementType();
    // ⭐ A SLICE IS `__getslice__`, NOT `__getitem__`. The emitter has always
    // known this (EmitterExpressions.cpp), and inference did not: it recursed
    // into the `Slice` node for an index type and resolved `__getitem__`,
    // which for `list[int]` answers the ELEMENT. Anything that types an
    // expression without emitting it therefore saw `xs[0:2]` as an int, and
    // the generator's yield-type walk is exactly that:
    //
    //     def chunks(xs: list[int], n: int) -> Iterator[list[int]]:
    //         yield xs[i:i + n]
    //     # annotated Iterator[list[int]] but yields builtins.int
    //
    // The same slice in a plain `return` compiled, because that goes through
    // the emitter.
    //
    // ⛔ STRICT ONLY, and the lenient answer is left exactly as it was. It is
    // load-bearing somewhere this walk does not own: with the correction
    // applied to both, `a[bump():3] += [99]` stopped splicing and printed
    // `[1, 2, 3, 4, 5]` where CPython gives `[1, 2, 3, 99, 4, 5]` -- a SILENT
    // wrong answer, caught by `augmented_assignment_evaluates_once`. The
    // lenient reading feeds the augmented-assignment slice route, and what it
    // wants there is not the slice's own type. Correcting that too means
    // finding what the route actually needs, which is a separate question
    // from the one this fixes.
    if (const parser::Node *sliceNode = ast::node(*node, "slice");
        sliceNode && sliceNode->kind == "Slice") {
      mlir::Type intType = this->intType();
      mlir::Type widened = widenLiteral(container);
      CallInferenceResult sliced = inferMethodCallWithEvidence(
          widened, "__getslice__", {intType, intType, intType, intType});
      if (sliced)
        return sliced.resultType;
      if (strict)
        return fail(sliced.failureReason);
    }
    mlir::Type index = recurse(ast::node(*node, "slice"));
    if (strict) {
      if (!container || !index)
        return {};
      CallInferenceResult inference = inferMethodCallWithEvidence(
          widenLiteral(container), "__getitem__", {index});
      if (inference)
        return inference.resultType;
      if (mlir::Type unionResult =
              unionOperatorResult(container, "__getitem__", {index}))
        return unionResult;
      return fail(inference.failureReason);
    }
    if (std::optional<CallSolution> result = tryManifestMethod(
            *this, widenLiteral(container), "__getitem__", {index}))
      return result->result;
    if (mlir::Type unionResult =
            unionOperatorResult(container, "__getitem__", {index}))
      return unionResult;
    return object();
  }
  if (node->kind == "Compare")
    return boolType();
  if (node->kind == "BoolOp") {
    // ⭐ `a or b` IS AN OPERAND, NOT A BOOL. CPython yields the value that
    // decided the expression, and the EMITTER builds exactly that (the join
    // of what each position can contribute) -- this channel answered `bool`,
    // so every reader that asks the type first disagreed with it. The
    // class-field walk is one: `self.v = xs or []` declared a bool field and
    // then refused the list the emitter stored into it.
    //
    // ⛔ Falls back to bool when the join is not representable, which is the
    // shape the emitter rejects anyway -- a condition asks this channel too,
    // and answering `object` there would be worse than answering `bool`.
    const auto *operands = ast::nodeList(*node, "values");
    const parser::Node *op = ast::node(*node, "op");
    if (!operands || operands->empty())
      return boolType();
    const bool isOr = op && op->kind == "Or";
    llvm::SmallVector<mlir::Type, 4> parts;
    llvm::SmallVector<const parser::Node *, 4> partNodes;
    for (auto [index, operand] : llvm::enumerate(*operands)) {
      if (!operand)
        return boolType();
      mlir::Type operandType = widenLiteral(lenientRecurse(operand.get()));
      if (isOr && index + 1 != operands->size())
        if (auto unionType =
                mlir::dyn_cast_if_present<py::UnionType>(operandType)) {
          // `or` keeps a TRUTHY non-final operand: an Optional's kept value is
          // its present member.
          llvm::SmallVector<mlir::Type, 4> present;
          for (mlir::Type member : unionType.getMemberTypes())
            if (member != none() && widenLiteral(member) != none())
              present.push_back(member);
          if (present.size() == 1)
            operandType = widenLiteral(present.front());
        }
      parts.push_back(operandType);
      partNodes.push_back(operand.get());
    }
    mlir::Type joined = joinIgnoringEmptyLiterals(*this, parts, partNodes);
    if (!joined || isObjectTop(*this, joined))
      return boolType();
    return joined;
  }
  // ⭐ `(y := e)` IS `e`. Without this the walk answered `object` for the
  // assignment expression, and a set comprehension whose element is one --
  // `sorted({(k := x) for x in xs})` -- typed its result as a set of object
  // and `sorted` refused it. The name it binds is the emitter's business; the
  // VALUE is the expression's own.
  if (node->kind == "NamedExpr")
    return recurse(ast::node(*node, "value"));
  if (node->kind == "IfExp") {
    // Mirrors the emitter: literal arms widen to their contracts (CPython
    // types a ternary of two literals by the common class).
    //
    // ⭐ AND EACH ARM SEES THE NARROWING ITS SIDE OF THE TEST PROVES, which
    // the EMITTER already applies. The two channels disagreeing is what a
    // pre-pass reads: the class-field walk types `self.xs = [] if xs is None
    // else xs` from here, got `list[object] | list[int] | None` where the
    // emitter stores a `list[int]`, and declared a field nothing could be
    // read out of ("does not provide manifest method '__len__'").
    //
    // ⛔ The None comparison only. The emitter's narrowing analysis lives a
    // layer up (it needs the emitted values to unwrap through), and the
    // isinstance and truthiness forms it also handles have no reader down
    // here that has been measured to need them.
    const parser::Node *bodyNode = ast::node(*node, "body");
    const parser::Node *elseNode = ast::node(*node, "orelse");
    const parser::Node *testNode = ast::node(*node, "test");
    llvm::StringRef narrowedName;
    mlir::Type narrowedPayload;
    bool trueBranchIsNone = false;
    if (testNode && testNode->kind == "Compare") {
      const auto *comparators = ast::nodeList(*testNode, "comparators");
      const auto *ops = ast::nodeList(*testNode, "ops");
      if (comparators && comparators->size() == 1 && ops && ops->size() == 1) {
        const parser::Node *op = ops->front().get();
        bool isIs = ast::isOperator(op, "Is");
        bool isIsNot = ast::isOperator(op, "IsNot");
        const parser::Node *left = ast::node(*testNode, "left");
        const parser::Node *right = comparators->front().get();
        const parser::Node *named = nullptr;
        if (left && right && left->kind == "Name" &&
            right->kind == "Constant" && isNoneConstant(right))
          named = left;
        else if (left && right && right->kind == "Name" &&
                 left->kind == "Constant" && isNoneConstant(left))
          named = right;
        if (named && (isIs || isIsNot)) {
          llvm::StringRef spelling = ast::nameSpelling(*named);
          if (std::optional<mlir::Type> current = lookupSymbol(spelling))
            if (auto unionType =
                    mlir::dyn_cast_if_present<py::UnionType>(*current)) {
              llvm::SmallVector<mlir::Type, 4> payload;
              for (mlir::Type member : unionType.getMemberTypes())
                if (member != none() && widenLiteral(member) != none())
                  payload.push_back(member);
              if (!payload.empty() &&
                  payload.size() != unionType.getMemberTypes().size()) {
                narrowedName = spelling;
                narrowedPayload = join(payload);
                trueBranchIsNone = isIs;
              }
            }
        }
      }
    }
    auto armType = [&](const parser::Node *arm,
                       bool conditionIsTrue) -> mlir::Type {
      if (narrowedName.empty())
        return widenLiteral(lenientRecurse(arm));
      mlir::Type narrowed = conditionIsTrue == trueBranchIsNone
                                ? none()
                                : narrowedPayload;
      auto scope = pushScope();
      bindLocalSymbol(narrowedName, narrowed);
      return widenLiteral(lenientRecurse(arm));
    };
    llvm::SmallVector<mlir::Type, 2> collected{
        armType(bodyNode, /*conditionIsTrue=*/true),
        armType(elseNode, /*conditionIsTrue=*/false)};
    llvm::SmallVector<const parser::Node *, 2> armNodes{bodyNode, elseNode};
    return joinIgnoringEmptyLiterals(*this, collected, armNodes);
  }
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
      if (mlir::Type unionResult =
              unionOperatorResult(operand, "__neg__", {}))
        return unionResult;
    }
    if (ast::isOperator(op, "UAdd"))
      if (std::optional<CallSolution> result =
              tryManifestMethod(*this, widenLiteral(operand), "__pos__", {}))
        return result->result;
      if (mlir::Type unionResult =
              unionOperatorResult(operand, "__pos__", {}))
        return unionResult;
    if (ast::isOperator(op, "Invert"))
      if (std::optional<CallSolution> result =
              tryManifestMethod(*this, widenLiteral(operand), "__invert__", {}))
        return result->result;
      if (mlir::Type unionResult =
              unionOperatorResult(operand, "__invert__", {}))
        return unionResult;
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
    // str % args is printf-style formatting (the emitter expands it); the
    // result is always str regardless of the right operand.
    if (ast::isOperator(op, "Mod") && left == strType())
      return strType();
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
    else if (ast::isOperator(op, "Pow"))
      method = "__pow__";
    // ⭐ A set operator answers the LEFT operand's type: CPython runs the
    // left's __or__, which builds its own kind, so `frozenset | set` is a
    // frozenset and `set | frozenset` is a set. The manifest declares the
    // parameter as the other kind too, and the solution joined the two into
    // the union `frozenset | set` -- which nothing downstream accepts, so
    // `sorted(f | {3})` was refused for a value that is an ordinary
    // frozenset at run time.
    {
      auto setKind = [&](mlir::Type type) {
        auto contract = mlir::dyn_cast_if_present<py::ContractType>(type);
        if (!contract)
          return false;
        llvm::StringRef name = contract.getContractName();
        return name == "builtins.set" || name == "builtins.frozenset";
      };
      if (setKind(left) && setKind(right) && left != right &&
          (method == "__or__" || method == "__and__" || method == "__xor__" ||
           method == "__sub__"))
        return left;
    }
    // int ** compile-time negative int types as float (CPython; the emitter
    // desugars it to float(base) ** float(exponent)). Exponents beyond the
    // double range keep the int path, matching the emitter's fallback.
    if (ast::isOperator(op, "Pow") && left == intType()) {
      auto rightLiteral = mlir::dyn_cast_if_present<py::LiteralType>(rawRight);
      llvm::StringRef spelling =
          rightLiteral ? rightLiteral.getSpelling() : llvm::StringRef();
      if (spelling.size() > 1 && spelling.front() == '-' &&
          llvm::all_of(spelling.drop_front(),
                       [](char c) { return c >= '0' && c <= '9'; })) {
        llvm::APFloat exponent(llvm::APFloat::IEEEdouble());
        llvm::Expected<llvm::APFloat::opStatus> status =
            exponent.convertFromString(spelling,
                                       llvm::APFloat::rmNearestTiesToEven);
        if (!status)
          llvm::consumeError(status.takeError());
        else if (!exponent.isInfinity())
          return floatType();
      }
    }
    // Tensor-scalar broadcast types as the tensor operand (the emitter splats
    // the scalar). Not for MatMult -- promoting the scalar there would type
    // `m @ 2.0` as a contraction of equal shapes -- and a float scalar never
    // broadcasts into an integer tensor (no silent element dtype change).
    mlir::Type primLeft = left;
    mlir::Type primRight = right;
    if (!ast::isOperator(op, "MatMult")) {
      auto broadcastable = [&](mlir::RankedTensorType tensor,
                               mlir::Type scalar) {
        if (mlir::isa_and_present<mlir::IntegerType>(scalar) ||
            scalar == intType())
          return true;
        if (mlir::isa_and_present<mlir::FloatType>(scalar) ||
            scalar == floatType())
          return mlir::isa<mlir::FloatType>(tensor.getElementType());
        return false;
      };
      if (auto tensor =
              mlir::dyn_cast_if_present<mlir::RankedTensorType>(left)) {
        if (broadcastable(tensor, right))
          primRight = left;
      } else if (auto tensor =
                     mlir::dyn_cast_if_present<mlir::RankedTensorType>(
                         right)) {
        if (broadcastable(tensor, left))
          primLeft = right;
      }
    }
    if (std::optional<mlir::Type> primitive =
            primitiveBinaryResultType(primLeft, primRight, op))
      return *primitive;
    if (std::optional<CallSolution> result =
            tryManifestMethod(*this, left, method, {right}))
      return result->result;
    if (mlir::Type unionResult = unionOperatorResult(left, method, {right}))
      return unionResult;
    if (mlir::Type unionResult =
            unionArgumentOperatorResult(left, method, right))
      return unionResult;
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

    // ⭐ The reducers the EMITTER folds are typed here, by the same walk the
    // fold uses to find its element type. Without it the inference had
    // nothing to say about them, so a reducer over a reducer --
    // `max(sum(r) for r in rows)` -- was refused for an element type it could
    // not see, while `max(len(r) for r in rows)` (a manifest function) was
    // fine.
    if (callee && callee->kind == "Name") {
      llvm::StringRef reducer = ast::nameSpelling(*callee);
      const auto *reducerArgs = ast::nodeList(*node, "args");
      bool oneArgument = reducerArgs && reducerArgs->size() == 1 &&
                         reducerArgs->front() &&
                         reducerArgs->front()->kind != "Starred";
      if ((reducer == "any" || reducer == "all") && oneArgument)
        return boolType();
      if ((reducer == "sum" || reducer == "max" || reducer == "min") &&
          oneArgument)
        if (mlir::Type element =
                iterationElementType(reducerArgs->front().get()))
          return element;
      // The lazy-iterator builtins are the same story one step further:
      // the emitter synthesizes a generator function for each of them, so
      // the type exists only once that function is emitted. A walk that ran
      // first -- an unannotated `def __iter__(self): return iter(self.items)`
      // -- read the callee as builtins.object and reported "is not callable",
      // naming the compiler's position rather than the program's. The yield
      // channel comes from iterationElementType, the same walk the fold
      // itself uses; send and return are None because a synthesized
      // generator has neither.
      if (mlir::Type lazy = lazyIteratorCallType(*this, reducer, reducerArgs))
        return lazy;
      if ((reducer == "max" || reducer == "min") && reducerArgs &&
          reducerArgs->size() > 1) {
        llvm::SmallVector<mlir::Type, 4> operands;
        for (const parser::NodePtr &argument : *reducerArgs) {
          if (!argument || argument->kind == "Starred")
            return object();
          mlir::Type operandType = widenLiteral(lenientRecurse(argument.get()));
          if (!operandType)
            return object();
          operands.push_back(operandType);
        }
        if (mlir::Type merged = join(operands))
          return merged;
      }
    }

    llvm::SmallVector<mlir::Type, 8> positional;
    if (const auto *args = ast::nodeList(*node, "args")) {
      for (const parser::NodePtr &arg : *args) {
        if (arg && arg->kind == "Starred") {
          mlir::Type starredType = recurse(ast::node(*arg, "value"));
          if (strict && !starredType)
            return {};
          if (!appendStarredArgumentTypes(*this, starredType, positional))
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

    // `C[int](...)` names its instantiation directly; the specialized class
    // is an ordinary non-generic contract from here on.
    if (mlir::Type instantiated = genericClassSubscript(callee))
      return instantiated;

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
      if (name == "next") {
        // ⭐ `next(it, default)` IS THE JOIN of the element and the default,
        // and this channel only knew the one-argument form -- so it answered
        // `object` for the two-argument one while the emitter built the union.
        // The disagreement surfaced wherever the value was USED without being
        // bound to a name first:
        //
        //     xs = iter([1])
        //     print(next(xs, None))
        //     # unnarrowed !py.union<int, None> cannot be used where a
        //     # concrete object is required
        //
        // `v = next(xs, None); print(v)` worked, because the binding carries
        // the emitter's answer and nothing re-asks this one.
        if (positional.size() == 2) {
          CallInferenceResult inference = inferMethodCallWithEvidence(
              widenLiteral(positional.front()), "__next__", {});
          if (!inference)
            return strict ? fail(inference.failureReason) : object();
          return join({widenLiteral(inference.resultType),
                       widenLiteral(positional[1])});
        }
        if (strict) {
          if (positional.size() != 1)
            return fail("next expects one positional argument");
          CallInferenceResult inference = inferMethodCallWithEvidence(
              widenLiteral(positional.front()), "__next__", {});
          if (inference)
            return inference.resultType;
          return fail(inference.failureReason);
        }
        if (positional.size() == 1)
          if (std::optional<CallSolution> result = tryManifestMethod(
                  *this, widenLiteral(positional.front()), "__next__", {}))
            return result->result;
        return object();
      }
      if (name == "repr") {
        // Not evidence-gated like next/len: repr's contract fixes the result
        // to str for every receiver (manifest __repr__, source-class
        // __repr__, default object repr), and the emitter's repr paths reject
        // unreachable receivers themselves. Falling through to object() here
        // made multi-argument print re-repr the already-rendered string.
        if (positional.size() == 1)
          return strType();
      }
      if (name == "format") {
        // Same shape as repr: every __format__ contract returns str, and the
        // emitter's dispatch rejects receivers without a resolvable
        // __format__ itself.
        if (positional.size() == 1 || positional.size() == 2)
          return strType();
      }
      if (name == "len") {
        if (strict) {
          if (positional.empty())
            return fail("len expects one positional argument");
          CallInferenceResult inference = inferMethodCallWithEvidence(
              widenLiteral(positional.front()), "__len__", {});
          if (inference)
            return inference.resultType;
          if (mlir::Type unionResult =
                  unionOperatorResult(positional.front(), "__len__", {}))
            return unionResult;
          return fail(inference.failureReason);
        }
        if (!positional.empty())
          if (mlir::Type unionResult =
                  unionOperatorResult(positional.front(), "__len__", {}))
            return unionResult;
        if (!positional.empty())
          if (std::optional<CallSolution> result = tryManifestMethod(
                  *this, widenLiteral(positional.front()), "__len__", {}))
            return result->result;
        return object();
      }
      if (name == "int") {
        // int(x) is conversion (emitter: tryEmitIntCall), not construction,
        // for the argument types the runtime __int__ methods cover. Other
        // shapes (zero args, unsupported types) stay on the instantiation
        // path and its diagnostics.
        if (positional.size() == 1) {
          mlir::Type argument = widenLiteral(positional.front());
          if (argument == intType() || argument == floatType() ||
              argument == strType())
            return intType();
        }
      }
      if (name == "float") {
        // float(x) is conversion (emitter: tryEmitFloatCall) for the types
        // the runtime __float__ methods cover; other shapes stay on the
        // instantiation path and its diagnostics.
        if (positional.size() == 1) {
          mlir::Type argument = widenLiteral(positional.front());
          if (argument == intType() || argument == floatType())
            return floatType();
        }
      }
      if ((name == "list" || name == "set" || name == "tuple" ||
           name == "dict") &&
          !lookupSymbol(name) && !lookupClass(name)) {
        // Container constructors are emitter desugars
        // (tryEmitContainerConstructorCall); the inference mirrors their
        // result shapes so a constructor call composes as an argument
        // (sorted(set(...)), dup(list(...))).
        auto elementOf = [&](mlir::Type iterableType) -> mlir::Type {
          std::optional<CallSolution> iter = tryManifestMethod(
              *this, widenLiteral(iterableType), "__iter__", {});
          if (!iter)
            return {};
          std::optional<CallSolution> next =
              tryManifestMethod(*this, iter->result, "__next__", {});
          if (!next)
            return {};
          return widenLiteral(next->result);
        };
        auto contractOf = [&](mlir::Type type) -> py::ContractType {
          return mlir::dyn_cast_if_present<py::ContractType>(
              widenLiteral(type));
        };
        if (positional.empty()) {
          if (name == "list")
            return listOf(join({}));
          if (name == "tuple")
            return tupleOf(join({}));
          if (name == "dict")
            return dictOf(join({}), join({}));
          return py::ContractType::get(&context, "builtins.set",
                                       {join({})});
        }
        if (positional.size() == 1) {
          py::ContractType argument = contractOf(positional.front());
          llvm::StringRef argumentClass =
              argument ? argument.getContractName() : llvm::StringRef();
          if (name == "tuple" && argumentClass == "builtins.tuple")
            return argument;
          if (name == "dict" && argumentClass == "builtins.dict")
            return argument;
          if (mlir::Type element = elementOf(positional.front())) {
            if (name == "list")
              return listOf(element);
            if (name == "tuple")
              return tupleOf(element);
            if (name == "set")
              return py::ContractType::get(&context, "builtins.set",
                                           {element});
            auto pair = contractOf(element);
            if (pair && pair.getContractName() == "builtins.tuple" &&
                pair.getArguments().size() == 2)
              return dictOf(pair.getArguments()[0], pair.getArguments()[1]);
          }
        }
        // Unsupported shapes fall to the emitter interceptor's diagnostics.
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
        if (*canonical == "lyrt.to_prim" && positional.size() == 2 &&
            keywords.empty()) {
          if (auto typeObject = mlir::dyn_cast_if_present<py::TypeType>(
                  positional.back()))
            if (mlir::isa<mlir::IntegerType, mlir::FloatType,
                          mlir::RankedTensorType>(
                    typeObject.getInstanceType()))
              return typeObject.getInstanceType();
          if (strict)
            return fail(
                "lyrt.to_prim expects a lyrt.prim type as its second "
                "argument");
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
        if (*canonical == "lyrt.to_prim" && positional.size() == 2 &&
            keywords.empty()) {
          if (auto typeObject = mlir::dyn_cast_if_present<py::TypeType>(
                  positional.back()))
            if (mlir::isa<mlir::IntegerType, mlir::FloatType,
                          mlir::RankedTensorType>(
                    typeObject.getInstanceType()))
              return typeObject.getInstanceType();
          if (strict)
            return fail(
                "lyrt.to_prim expects a lyrt.prim type as its second "
                "argument");
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
          // Static methods carry no receiver parameter, so they resolve from
          // their own channel rather than the receiver-bound method contracts.
          mlir::Type staticOwner = widenLiteral(receiver);
          if (auto typeObjectType =
                  mlir::dyn_cast_if_present<py::TypeType>(staticOwner))
            staticOwner = typeObjectType.getInstanceType();
          if (auto ownerContract =
                  mlir::dyn_cast_if_present<py::ContractType>(staticOwner))
            if (std::optional<mlir::Type> staticMethod = lookupClassStaticMethod(
                    ownerContract.getContractName(), *methodName)) {
              if (strict) {
                CallInferenceResult inference = inferCallWithEvidence(
                    *staticMethod, positional, keywords);
                if (inference)
                  return inference.resultType;
                return fail(inference.failureReason);
              }
              return inferCall(*staticMethod, positional, keywords);
            }
          // str.format is expanded by the emitter (no manifest method); its
          // result is always str.
          if (*methodName == "format" &&
              widenLiteral(receiver) == strType())
            return strType();
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
    // ⭐ A LAMBDA'S BODY IS INFERRED IN THE SCOPE THAT CONTAINS IT. The
    // signature walk sees only the symbol table, and a body walk's locals
    // (assignments, loop targets) live beside it -- so a lambda that reads one
    // answered `object`:
    //
    //     def gen(n: int):
    //         for i in range(n):
    //             yield lambda: i     # yields Callable[[], object]
    //
    // and the generator's frame lane for that erased result then failed in the
    // lowering. The parameter-reading spelling (`yield lambda: n`) worked all
    // along, because a parameter IS in the symbol table.
    auto lambdaScope = pushScope();
    if (ctx) {
      if (ctx->localSymbols)
        for (const auto &entry : *ctx->localSymbols)
          bindLocalSymbol(entry.getKey(), entry.getValue());
      for (const auto &entry : ctx->localCallables)
        bindLocalSymbol(entry.getKey(), entry.getValue());
    }
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
  // A monomorphized user generic resolves to its specialization, which is an
  // ordinary ground class; the manifest paths below only know parameterized
  // MANIFEST classes.
  if (auto contract = mlir::dyn_cast_if_present<py::ContractType>(instanceType))
    if (contract.getArguments().empty())
      if (mlir::Type solved = solveGenericClassInstantiation(
              contract.getContractName(), positional, keywords))
        return solved;
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
  // ⭐ `str.upper(s)` IS `s.upper()`. An unbound method called through its
  // class is ordinary Python, and it is also what a `map(str.upper, xs)` fast
  // path re-spells the callable as -- so both were refused with a sentence
  // about the TYPE object not providing the method.
  if (!positional.empty())
    if (manifestMethodReceiverContract(receiverType, methodName))
      return inferMethodCallWithEvidence(positional.front(), methodName,
                                         positional.drop_front(), keywords);
  return unresolvedMethodCall(*this, receiverType, methodName, positional,
                              keywords);
}

bool TypeSystem::declaresManifestMethod(mlir::Type receiverType,
                                        llvm::StringRef methodName) const {
  const py::protocols::Table &table = py::protocols::Table::get(getContext());
  for (const py::protocols::ContractResolution &candidate :
       table.methodContractCandidatesWithEvidence(receiverType, methodName)) {
    (void)candidate;
    return true;
  }
  return false;
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
    // A generator delegate completes with its return type R; plain
    // iterators/iterables complete with None (PEP 380).
    mlir::Type completion = name == "Generator" && arguments.size() >= 3
                                ? arguments[2]
                                : none();
    return YieldFromInferenceResult{
        arguments.front(), completion, protocol(name, arguments), true, {}};
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
    // ⛔ This refusal is CORRECT and must stay: the declared signature does not
    // admit these arguments, and nothing here may widen it. What used to be a
    // defect behind it -- an int argument where a float parameter is declared
    // -- is handled a layer up, by emitting a SECOND BODY for the argument's
    // own rung (`emitArgumentSpecializedCall`), never by converting: CPython
    // leaves the annotation inert at a parameter, so `def p(x: float)` reached
    // by `p(3)` prints 3 and not 3.0.
    //
    // ⛔ Why the specializer does not simply call back in here to ask: it
    // would have to, to cover a specialized body that calls another
    // specializable function with the narrowed parameter, and that call needs
    // a way from a callable TYPE back to the function NODE -- which this class
    // does not have and which two same-signature functions would make
    // ambiguous. The limitation and its shape are in
    // tests/probe/wb_argument_boundary_numeric_tower.py.
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

// ⭐ EVERY MEMBER OR NONE. `1 if c else 1.5` is a union, and CPython adds to it
// without knowing which member is live because BOTH answer `__add__`. This
// compiler refused the whole expression -- "static type !py.union<int, float>
// does not provide manifest method '__add__'" -- for a program whose every
// execution is well typed.
//
// The answer is the JOIN of what the members answer, which is what the tag
// dispatch the emitter builds actually produces.
//
// ⛔ Every member, not a majority: a union with one member that cannot do it
// has executions that would fail, and CPython fails those at RUN time. Refusing
// the whole expression is this compiler's rule for that
// (`never silently mis-execute`), and it is the answer `int | str` keeps.
//
// ⛔ And OPERATORS only. A general union method call would need the same
// dispatch built at every call site that can reach one; the operator paths are
// where the emitter has it.
mlir::Type
TypeSystem::unionOperatorResult(mlir::Type receiver, llvm::StringRef method,
                                mlir::ArrayRef<mlir::Type> arguments) const {
  auto unionType =
      mlir::dyn_cast_if_present<py::UnionType>(widenLiteral(receiver));
  if (!unionType)
    return {};
  llvm::SmallVector<mlir::Type, 4> results;
  for (mlir::Type member : unionType.getMemberTypes()) {
    mlir::Type receiverMember = widenLiteral(member);
    llvm::SmallVector<mlir::Type, 2> memberArguments(arguments.begin(),
                                                     arguments.end());
    // ⭐ CPython's numeric tower reaches INTO the union. `int` has no
    // `__add__` taking a float, so an `int | float` member pair declined here
    // while the plain `1 + 2.0` beside it compiles -- that one is promoted at
    // the OPERANDS before the dispatch, and the arm the emitter builds for
    // this member promotes the same way.
    if (memberArguments.size() == 1) {
      mlir::Type argument = widenLiteral(memberArguments.front());
      if (receiverMember == intType() && argument == floatType())
        receiverMember = floatType();
      else if (receiverMember == floatType() && argument == intType())
        memberArguments[0] = floatType();
    }
    // ⭐ AND A UNION ON BOTH SIDES. `mk(-1) + mk(3)` is two of these values
    // added together, and each member of the left has to ask the same question
    // of the whole right -- which is the mirror rule, applied per member. The
    // emitter's arms nest the same way: the left's tag chooses a member, and
    // the recursive call meets the right's tag with a concrete receiver.
    if (memberArguments.size() == 1 &&
        mlir::isa_and_nonnull<py::UnionType>(
            widenLiteral(memberArguments.front()))) {
      mlir::Type nested = unionArgumentOperatorResult(
          receiverMember, method, memberArguments.front());
      if (!nested)
        return {};
      results.push_back(nested);
      continue;
    }
    std::optional<CallSolution> solved =
        tryManifestMethod(*this, receiverMember, method, memberArguments);
    if (!solved || !solved->result)
      return {};
    results.push_back(widenLiteral(solved->result));
  }
  if (results.empty())
    return {};
  return join(results);
}

// ⭐ AND THE UNION ON THE RIGHT. `total += scaled(i)` is the accumulator every
// running sum is written as, and the value being added is the union -- the
// receiver is an ordinary float. The left-hand rule above does not see it, so
// this asks the same question of the argument's members.
mlir::Type TypeSystem::unionArgumentOperatorResult(mlir::Type receiver,
                                                   llvm::StringRef method,
                                                   mlir::Type argument) const {
  auto unionType =
      mlir::dyn_cast_if_present<py::UnionType>(widenLiteral(argument));
  if (!unionType)
    return {};
  mlir::Type receiverType = widenLiteral(receiver);
  llvm::SmallVector<mlir::Type, 4> results;
  for (mlir::Type member : unionType.getMemberTypes()) {
    mlir::Type memberType = widenLiteral(member);
    mlir::Type armReceiver = receiverType;
    // The same operand promotion the left-hand rule makes; see the note there.
    if (armReceiver == intType() && memberType == floatType())
      armReceiver = floatType();
    else if (armReceiver == floatType() && memberType == intType())
      memberType = floatType();
    std::optional<CallSolution> solved =
        tryManifestMethod(*this, armReceiver, method, {memberType});
    if (!solved || !solved->result)
      return {};
    results.push_back(widenLiteral(solved->result));
  }
  if (results.empty())
    return {};
  return join(results);
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
  // ⭐ SOURCE CLASSES JOIN AT THEIR BASE, not into a union of themselves.
  // `[Dog(), Cat()]` and `Dog() if c else Cat()` both inferred
  // `Cat | Dog`, and a union provides no method, so `a.speak()` was refused --
  // for the two spellings of the polymorphism this compiler already supports
  // through a base-typed receiver. `list[Animal]` and a function returning
  // `Animal` both work, so what was missing is only the inference: the
  // dispatch, the coercion of a subclass into a base-typed value, and the
  // boxed list element are all in place.
  //
  // ⛔ ONLY when the base is a SOURCE class. Collapsing at `object` is the
  // erased top and loses everything; a builtin base (a user exception under
  // `ValueError`) is not a receiver this can dispatch through either, so the
  // walk asks `lookupClass` and stops where the source classes stop.
  //
  // Why NOT resolve it at the method lookup instead, leaving the union: the
  // value is physically a union there -- lanes and a tag -- so calling
  // through it needs a tag switch with a call per member. This is the same
  // answer one step earlier, where the coercion already exists.
  if (mlir::Type base = nearestCommonSourceBase(present))
    return base;
  if (mlir::Type callable = commonCallableJoin(present))
    return callable;
  return py::UnionType::getNormalized(&context, present);
}

// ⭐ TWO FUNCTIONS OF THE SAME SHAPE JOIN AT ONE FUNCTION, not into a union of
// themselves. `[lambda: 1, lambda: 2]` inferred
// `Callable[[], 1] | Callable[[], 2]`, and a union is not callable, so
// `fs[0]()` and `[f() for f in fs]` were both refused -- for a list of
// same-signature functions, which is what a jump table IS in Python. The same
// list written with two `def`s already worked, because their annotated
// results were already the one type.
//
// ⛔ ONLY when the parameter shapes are identical, which is checked by
// rebuilding each member with the joined result and requiring the rebuilds to
// agree: a real join over differing parameters is a MEET on each one
// (contravariance), and a callable that claims to accept more than a member
// does would be unsound at the call the union member cannot serve.
mlir::Type
TypeSystem::commonCallableJoin(mlir::ArrayRef<mlir::Type> members) const {
  llvm::SmallVector<py::CallableType, 4> callables;
  llvm::SmallVector<mlir::Type, 4> results;
  for (mlir::Type member : members) {
    auto callable = mlir::dyn_cast_if_present<py::CallableType>(member);
    if (!callable || callable.getResultTypes().size() != 1)
      return {};
    callables.push_back(callable);
    // The literal widening is what makes the two spellings the same shape at
    // all: `lambda: 1` and `lambda: 2` return `literal<1>` and `literal<2>`.
    results.push_back(widenLiteral(callable.getResultTypes().front()));
  }
  if (callables.size() < 2)
    return {};
  // ⛔ And only when the RESULTS agree too. `[lambda: 1, lambda: 2.0]` joins
  // to `Callable[[], int | float]`, and an indirect call cannot return a union
  // -- the program left the emitter and died in the lowering ("runtime bundle
  // value 0 for builtins.bool"). A union of callables keeps the emit-boundary
  // refusal it already had, which is the honest answer for it.
  mlir::Type joinedResult = join(results);
  if (!joinedResult || mlir::isa<py::UnionType>(joinedResult) ||
      isObjectTop(*this, joinedResult))
    return {};
  auto withResult = [&](py::CallableType callable) {
    llvm::SmallVector<mlir::Type, 1> single{joinedResult};
    return py::CallableType::get(
        &context, callable.getPositionalTypes(), callable.getKwOnlyTypes(),
        callable.getVarargType(), callable.getKwargType(), single,
        callable.getPositionalNames(), callable.getKwOnlyNames(),
        callable.getPositionalDefaults(), callable.getKwOnlyDefaults(),
        callable.getVarargName(), callable.getKwargName(),
        callable.getPositionalOnlyCount());
  };
  py::CallableType joined = withResult(callables.front());
  for (py::CallableType callable : llvm::ArrayRef(callables).drop_front())
    if (withResult(callable) != joined)
      return {};
  return joined;
}

mlir::Type
TypeSystem::nearestCommonSourceBase(mlir::ArrayRef<mlir::Type> members) const {
  llvm::SmallVector<llvm::StringRef, 4> names;
  for (mlir::Type member : members) {
    auto contract = mlir::dyn_cast_if_present<py::ContractType>(member);
    if (!contract || !contract.getArguments().empty() ||
        !lookupClass(contract.getContractName()))
      return {};
    names.push_back(contract.getContractName());
  }
  // Candidates in nearest-first order: the first member, then its bases
  // breadth-first, which is the order a common base should be found in.
  llvm::SmallVector<llvm::StringRef, 8> candidates{names.front()};
  llvm::StringSet<> seen;
  for (unsigned index = 0; index < candidates.size(); ++index) {
    auto entry = declaredBases.find(candidates[index]);
    if (entry == declaredBases.end())
      continue;
    for (const std::string &base : entry->second)
      if (seen.insert(base).second)
        candidates.push_back(base);
  }
  for (llvm::StringRef candidate : candidates) {
    std::optional<mlir::Type> candidateType = lookupClass(candidate);
    if (!candidateType)
      continue;
    if (llvm::all_of(names, [&](llvm::StringRef name) {
          return declaredSubclassOf(name, candidate);
        }))
      return *candidateType;
  }
  return {};
}

// The spellings `types.literal` may hold for a float: a decimal point, an
// exponent, or one of the two non-finite names, with an optional sign.
static bool isFloatLiteralSpelling(llvm::StringRef spelling) {
  llvm::StringRef body = spelling;
  body.consume_front("-");
  body.consume_front("+");
  if (body == "inf" || body == "nan")
    return true;
  if (body.empty() || body.front() == '"')
    return false;
  if (!body.contains('.') && !body.contains('e') && !body.contains('E'))
    return false;
  double parsed = 0.0;
  return !spelling.getAsDouble(parsed);
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
  // ⭐ A FLOAT SPELLING IS A FLOAT. Everything unquoted that is not
  // True/False/None used to widen to INT, which is why an imported module's
  // `RATIO = 1.5` could not be a literal constant at all -- the import channel
  // had to decline it rather than hand back a spelling every reader would take
  // for an integer.
  //
  // ⛔ The mark, not `getAsDouble`: "3" parses as a double too, and an int
  // literal must stay an int. A decimal point, an exponent, or the two
  // non-finite spellings are what a float has and an int cannot.
  if (isFloatLiteralSpelling(spelling))
    return floatType();
  return intType();
}

FunctionSignature
TypeSystem::functionSignature(const parser::Node &function,
                              std::optional<llvm::StringRef> selfName,
                              py::CallableType expectedCallable,
                              mlir::Type selfType, bool monomorphize) const {
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
    // A default is a complete static description of its parameter when
    // nothing else supplies one. Not folded into the parameter's inference
    // variable: a caller that passes something else has to WIDEN the
    // parameter rather than conflict with the default, so this is read back
    // where the variable is resolved, not unified into it.
    auto defaultParameterType =
        [&](const parser::Node *defaultNode) -> mlir::Type {
      if (!defaultNode)
        return {};
      mlir::Type inferred = inferenceState.zonk(inferExpr(defaultNode));
      if (!inferred || py::containsPyInferVar(inferred))
        return {};
      return widenLiteral(inferred);
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
          (function.kind == "Lambda" || monomorphize) &&
          index < expectedPositional.size();
      std::optional<mlir::Type> overridden =
          !annotation && !isSelfParameter && !fromExpectedCallable
              ? overriddenParameterType(*arg)
              : std::nullopt;
      mlir::Type fromDefault;
      // ⛔ AN EMPTY CONTAINER DEFAULT DESCRIBES THE PARAMETER ONLY UNTIL A
      // CALL SITE DOES. `def f(xs=[])` called `f([1, 2])` joined `list[object]`
      // with `list[int]` into a union nothing accepts; `[]` has no element
      // type of its own, which is the rule four other joins already state.
      bool defaultIsEmptyContainer = false;
      if (!annotation && !isSelfParameter && !fromExpectedCallable &&
          hasDefault(index, positional.size(), defaults)) {
        const parser::Node *defaultNode =
            (*ast::nodeList(*arguments, "defaults"))[index + defaults -
                                                     positional.size()]
                .get();
        fromDefault = defaultParameterType(defaultNode);
        defaultIsEmptyContainer = isEmptyContainerExpression(defaultNode);
      }
      if (isSelfParameter)
        type = selfType ? selfType : py::SelfType::get(&context);
      if (fromExpectedCallable)
        type = expectedPositional[index];
      if (overridden) {
        type = *overridden;
        if (py::containsPyInferVar(type)) {
          // Only once the module fixpoint has stopped: substituting the
          // default earlier would replace the variable the CALL SITES bind
          // through, and every argument would then be checked against the
          // default's type instead of widening the parameter.
          if (fromDefault && defaultsDescribeParameters)
            type = fromDefault;
          else if (!fromDefault || !defaultsDescribeParameters)
            sig.missingParameterAnnotations.push_back(name);
        } else if (fromDefault && !py::isAssignableTo(fromDefault, type) &&
                   !defaultIsEmptyContainer) {
          type = join({type, fromDefault});
        }
      } else if (!isSelfParameter && !fromExpectedCallable) {
        if (fromDefault)
          type = fromDefault;
        else
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
            (function.kind == "Lambda" || monomorphize) &&
            index < expectedKwOnly.size();
        std::optional<mlir::Type> overridden =
            !annotation && !fromExpectedCallable ? overriddenParameterType(*arg)
                                                 : std::nullopt;
        bool hasKwDefault = false;
        if (const auto *kwDefaults = ast::nodeList(*arguments, "kw_defaults"))
          hasKwDefault = index < kwDefaults->size() && (*kwDefaults)[index];
        mlir::Type fromDefault;
        if (!annotation && !fromExpectedCallable && hasKwDefault)
          fromDefault = defaultParameterType(
              (*ast::nodeList(*arguments, "kw_defaults"))[index].get());
        if (fromExpectedCallable)
          type = expectedKwOnly[index];
        if (overridden) {
          type = *overridden;
          if (py::containsPyInferVar(type)) {
            if (fromDefault && defaultsDescribeParameters)
              type = fromDefault;
            else if (!fromDefault || !defaultsDescribeParameters)
              sig.missingParameterAnnotations.push_back(name);
          } else if (fromDefault && !py::isAssignableTo(fromDefault, type)) {
            type = join({type, fromDefault});
          }
        } else if (!fromExpectedCallable) {
          if (fromDefault)
            type = fromDefault;
          else
            recordAnnotationIssue(annotation, name);
        }
        sig.kwOnlyTypes.push_back(type);
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
        generator.yieldTypes.empty()
            ? none()
            : (generator.yieldNodes.size() == generator.yieldTypes.size()
                   ? joinIgnoringEmptyLiterals(*this, generator.yieldTypes,
                                               generator.yieldNodes)
                   : join(generator.yieldTypes));
    // Without an annotation, the send channel is what the delegates accept:
    // PEP 380 forwards send() into the active `yield from` delegate.
    sig.generatorSendType = annotatedGeneratorSendType.value_or(
        generator.delegatedSendTypes.empty()
            ? none()
            : join(generator.delegatedSendTypes));
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

    if (returns && !py::isAssignableTo(sig.inferredGeneratorType,
                                       annotatedReturn)) {
      // ⭐ A YIELD INSIDE A GUARD YIELDS WHAT THE GUARD PROVED, and this walk
      // cannot see guards: it types each `yield` expression on its own, so
      //
      //     def gen(xs: list[int | None]) -> Iterator[int]:
      //         for v in xs:
      //             if v is not None:
      //                 yield v * 2
      //
      // came out as `int | None` and the annotation was reported as a
      // mismatch, for a program whose every yield is an int. The annotation is
      // the CONTRACT here, as it is everywhere else: when every inferred yield
      // either satisfies it or is a union that CONTAINS it, the annotation is
      // taken and each yield is checked at its own site, where the narrowing
      // is available. A yield that really cannot produce the annotated type is
      // then refused there, naming the yield instead of the function.
      //
      // ⛔ Only when the union CONTAINS it: a yield of an unrelated type is
      // still the function-level mismatch, because no guard could make it
      // right and the message that names the whole function is the better one.
      std::optional<mlir::Type> annotatedYield;
      if (const py::protocols::Table &table =
              py::protocols::Table::get(context);
          true)
        for (llvm::StringRef protocolName :
             {"Generator", "AsyncGenerator", "Iterator", "AsyncIterator",
              "Iterable"})
          if (!annotatedYield)
            if (std::optional<std::vector<mlir::Type>> args =
                    table.protocolArgumentsFor(annotatedReturn, protocolName))
              if (!args->empty() && (*args)[0])
                annotatedYield = (*args)[0];
      bool everyYieldNarrows =
          annotatedYield && *annotatedYield && !generator.yieldTypes.empty() &&
          llvm::all_of(generator.yieldTypes, [&](mlir::Type yielded) {
            mlir::Type widened = widenLiteral(yielded);
            if (py::isAssignableTo(widened, *annotatedYield))
              return true;
            auto unionType = mlir::dyn_cast_if_present<py::UnionType>(widened);
            return unionType && unionType.hasMember(*annotatedYield);
          });
      if (everyYieldNarrows) {
        sig.generatorYieldType = *annotatedYield;
        sig.generatorYieldTypeIsAnnotated = true;
        sig.inferredGeneratorType =
            function.kind == "AsyncFunctionDef"
                ? protocol("AsyncGenerator",
                           {sig.generatorYieldType, sig.generatorSendType})
                : contract("types.GeneratorType",
                           {sig.generatorYieldType, sig.generatorSendType,
                            sig.generatorReturnType});
      }
      if (!everyYieldNarrows ||
          !py::isAssignableTo(sig.inferredGeneratorType, annotatedReturn))
        // ⛔ MEASURED AND DROPPED: taking the annotation whatever the walk
        // inferred. It also accepts the shapes whose yield EXPRESSION needs
        // the narrowing (`yield v.upper()` types as None because the lookup on
        // the union fails), but it moved a genuinely wrong generator --
        // `-> Iterator[str]` with `yield v` for an int v -- from this
        // sentence to "Failed to run lowering pipeline", because the site
        // coercion does not refuse int where str is declared. A worse
        // diagnostic for a wrong program is not a trade worth the extra right
        // ones; those need a narrowing-aware yield walk.
        sig.generatorAnnotationMismatch =
            "generator function is annotated " + typeText(annotatedReturn) +
            " but yields " + typeText(sig.generatorYieldType) + " (inferred " +
            typeText(sig.inferredGeneratorType) + ")";
    }
    sig.resultType = sig.generatorReturnType;
  } else if (function.kind == "Lambda") {
    sig.resultType = inferExpr(ast::node(function, "body"));
  } else if (returns && !monomorphize) {
    mlir::Type annotated = annotationType(returns);
    sig.resultType = annotated;
    // ⭐ A NUMERIC RETURN ANNOTATION IS A CONSTRAINT A LOWER RUNG ALREADY
    // SATISFIES, and taking it literally is what refused
    //
    //     def half(n: int) -> float: return n // 2
    //     def positive(n: int) -> int: return n > 0
    //
    // with "type of return operand 0 ... doesn't match function result type"
    // from the MLIR verifier -- a message about the compiler, over ordinary
    // Python. CPython's `half(7)` is the int 3 and `positive(3)` is True, and
    // every other boundary in this compiler already answers that way: a local
    // (`x: float = 3` prints 3), a parameter and a parameter default (both by
    // specialization). The return was the last one reading the annotation as
    // the answer rather than as the constraint.
    //
    // ⛔ NOT by converting: `print(half(7))` would answer 3.0. The annotation
    // does not convert in CPython either.
    //
    // ⛔ And only a LOWER rung of the same tower. A body whose arms return
    // different rungs joins to a union, whose rung is -1, so it keeps the
    // annotation and keeps its refusal -- the py ABI cannot return a union and
    // collapsing it along the tower would print 0 for False.
    //
    // ⛔ NOT for a `complex` annotation, and this is a measurement, not a
    // caution: `inferExpr` answers `builtins.float` for `1.0 + 0.0j`, so
    // `def rotate(z: complex, n: int) -> complex` re-read at float emitted a
    // body that returns a complex against a float ABI -- "cannot adapt
    // builtins.complex return value to callable return ABI 0 of rotate"
    // (golden scalar_loop_carried_mutate). It is the same inference the
    // argument specializer refuses to trust, for the same reason and on the
    // same program.
    if (int declaredRung = numericTowerRung(*this, annotated);
        declaredRung > 0 && declaredRung <= 2 &&
        returnRungWalks.insert(&function).second) {
      mlir::Type walked = inferredFunctionResult(*this, function,
                                                 /*failureReasons=*/nullptr,
                                                 &generator.localSymbols);
      returnRungWalks.erase(&function);
      mlir::Type widened = walked ? widenLiteral(walked) : mlir::Type();
      if (int walkedRung = numericTowerRung(*this, widened);
          walkedRung >= 0 && walkedRung < declaredRung)
        sig.resultType = widened;
    }
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
