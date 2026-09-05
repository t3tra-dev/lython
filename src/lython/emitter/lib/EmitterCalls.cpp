#include "Contracts.h"
#include "AstSynth.h"
#include "EmitterCore.h"
#include "EmitterPyOps.h"
#include "EmitterSupport.h"
#include "ArithBuilders.h"
#include "ExceptionTaxonomy.h"
#include "PlatformConstants.h"
#include "PyProtocols.h"
#include "TypeSystemSolver.h"

#include "AstAccess.h"

#include <functional>

#include "llvm/ADT/ScopeExit.h"
#include "EmitterOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Support/raw_ostream.h"

#include <iterator>
#include <optional>
#include <string>

namespace lython::emitter {
namespace {

using common::constantBool;

// CPython's numeric tower as a total order on the four rungs that are
// implicitly acceptable where a higher one is declared (PEP 484 §"The numeric
// tower"). -1 is "not on the tower", which never compares.
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


Value boxedBool(mlir::OpBuilder &builder, mlir::Location loc, TypeSystem &types,
                mlir::Value bit) {
  auto pyBool = py::CastFromPrimOp::create(builder, loc, types.boolType(), bit);
  return {pyBool.getResult(), types.boolType()};
}

bool callHasNoArguments(const parser::Node &expr) {
  const auto *args = ast::nodeList(expr, "args");
  const auto *keywords = ast::nodeList(expr, "keywords");
  return (!args || args->empty()) && (!keywords || keywords->empty());
}

// A builtin fast-path's arguments in parameter order, keywords bound to
// `parameters` -- or nothing, which sends the call down the generic path that
// already refuses what CPython raises TypeError for.
//
// ⭐ Every fast path above declines when a keyword appears; two forgot, and
// both silently DROPPED it. `round(2.567, ndigits=1)` printed 3 where CPython
// prints 2.6, and `len([1], bogus=2)` printed 1 where CPython raises. A
// dropped argument is the one failure a `!keywords->empty()` guard cannot
// have, so the two that need to look at names come here instead of each
// growing its own reading of the keyword list.
//
// `positionalOnly` is CPython's `/`: `len(obj, /)` accepts no keyword at all,
// while `round(number, ndigits=None)` accepts both by name.
std::optional<llvm::SmallVector<const parser::Node *, 4>>
bindBuiltinArguments(const parser::Node &expr,
                     llvm::ArrayRef<llvm::StringRef> parameters,
                     unsigned positionalOnly) {
  llvm::SmallVector<const parser::Node *, 4> bound(parameters.size(), nullptr);
  const auto *args = ast::nodeList(expr, "args");
  unsigned positional = args ? static_cast<unsigned>(args->size()) : 0;
  if (positional > parameters.size())
    return std::nullopt;
  for (unsigned index = 0; index < positional; ++index)
    bound[index] = (*args)[index].get();
  if (const auto *keywords = ast::nodeList(expr, "keywords"))
    for (const parser::NodePtr &keyword : *keywords) {
      std::optional<std::string_view> name = ast::string(*keyword, "arg");
      // `**kwargs` has no name; nothing static can be said about what it binds.
      if (!name)
        return std::nullopt;
      const auto *found = llvm::find(parameters, llvm::StringRef(*name));
      if (found == parameters.end())
        return std::nullopt;
      auto index = static_cast<unsigned>(found - parameters.begin());
      if (index < positionalOnly || bound[index])
        return std::nullopt;
      bound[index] = ast::node(*keyword, "value");
      if (!bound[index])
        return std::nullopt;
    }
  // A gap means an unfilled parameter before a filled one (`round(ndigits=1)`),
  // which is a missing argument rather than a shorter call.
  while (!bound.empty() && !bound.back())
    bound.pop_back();
  if (llvm::is_contained(bound, nullptr))
    return std::nullopt;
  return bound;
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
  // ⭐ A LAMBDA'S EXPECTED CALLABLE MAY MENTION A TYPEVAR A LATER ARGUMENT
  // DECIDES. Arguments are emitted left to right against the callee's
  // contract as written, and a generic one still has its parameters in it:
  //
  //     functools.reduce(lambda a, b: a + b, [1, 2, 3])
  //     # static type !py.typevar<"T"> does not provide manifest method
  //     # '__add__'
  //
  // `reduce[T](function: Callable[[T, T], T], sequence: list[T])` binds T from
  // the SECOND argument, and the lambda is the first -- so its body was
  // compiled against T. The call itself then specialized correctly, which is
  // why the second diagnostic already names `Callable[[int, int], int]`: the
  // types were known, just not yet when the body needed them.
  //
  // So the parameters are bound from the arguments that can be typed WITHOUT
  // being emitted, and the formals are substituted before the walk starts. A
  // lambda is skipped on that pass for the same reason it needs this: it has
  // no type until an expectation gives it one.
  //
  // ⛔ Only when a lambda is actually present. `types.inferExpr` over every
  // argument of every call is a second full walk of each one, and nothing else
  // here needs it -- an ordinary argument reaches the same bindings through
  // `tryCallableApplication` after it is emitted.
  TypeBindingMap lambdaBindings;
  const auto *positionalArgs = ast::nodeList(expr, "args");
  if (expectedContract && positionalArgs &&
      llvm::any_of(*positionalArgs,
                   [](const parser::NodePtr &arg) {
                     return arg && arg->kind == "Lambda";
                   }) &&
      llvm::any_of(expectedPositional, [](mlir::Type formal) {
        return formal && unboundStaticParameterCount(formal) > 0;
      })) {
    for (auto [index, arg] : llvm::enumerate(*positionalArgs)) {
      if (index >= expectedPositional.size())
        break;
      if (!arg || arg->kind == "Lambda" || arg->kind == "Starred")
        continue;
      if (mlir::Type actual = types.inferExpr(arg.get()))
        bindExpectedType(types, expectedPositional[index],
                         types.widenLiteral(actual), lambdaBindings);
    }
  }

  // A static type parameter as formal means the expectation is the CALL's
  // output (the argument determines it), not an input to distribute; a
  // starred argument breaks positional alignment for everything after it.
  auto expectedFor = [&](std::size_t index) -> mlir::Type {
    if (index >= expectedPositional.size())
      return {};
    mlir::Type formal = expectedPositional[index];
    if (formal && py::isStaticTypeParameter(formal))
      return {};
    if (formal && !lambdaBindings.empty())
      formal = substituteType(types, formal, lambdaBindings);
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
        if (!appendStarredArgumentTypes(types, value.type,
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
      // A keyword with no `arg` is `**mapping`, and its names are the
      // mapping's keys -- which nothing static reads. Pushing the value alone
      // left `keywordNames` one short of `keywordValues`, and the size was not
      // checked until callable planning eight phases later, where it reported
      // "kw names and kw values must have the same size" against a fused
      // location with no source line in it.
      //
      // ⛔ NOT a rewrite of the two shapes that ARE static -- `f(**{"a": 1})`,
      // whose keys are literals, and `g(**kwargs)` forwarding a parameter that
      // is itself a dict -- because each needs the callee's keyword ABI, and
      // this is the boundary that has to refuse first either way.
      emitExpr(ast::node(*keyword, "value"));
      operands.valid = false;
      operands.failureReason =
          "`**` call arguments require statically known keyword names; spell "
          "the keywords out";
    }
  }

  return operands;
}

// A rewritten call whose positional arguments are widened along the numeric
// tower to the declared parameter, or null when nothing needs widening. Only
// for a MANIFEST export -- see the ⭐ at the caller for why that is the whole
// question. The returned node owns its rewritten arguments and must outlive
// the operand emission, which is why it is handed back rather than emitted.
parser::NodePtr ModuleEmitter::widenNumericArgumentsForManifestCall(
    const parser::Node &expr, llvm::StringRef binding,
    py::CallableType declared) {
  if (!declared || binding.empty())
    return nullptr;
  const py::protocols::Table &table = py::protocols::Table::get(context);
  if (!table.freeFunctionContract(binding))
    return nullptr;
  const auto *args = ast::nodeList(expr, "args");
  if (!args || args->empty())
    return nullptr;
  // Keywords and starred arguments do not line up positionally with the
  // declared parameters here, and a partial rewrite would widen the wrong one.
  if (const auto *keywords = ast::nodeList(expr, "keywords"))
    if (!keywords->empty())
      return nullptr;
  llvm::ArrayRef<mlir::Type> parameters = declared.getPositionalTypes();
  if (args->size() != parameters.size())
    return nullptr;

  auto spelling = [&](mlir::Type type) -> llvm::StringRef {
    if (type == types.floatType())
      return "float";
    if (type == types.intType())
      return "int";
    return {};
  };
  std::vector<parser::NodePtr> rewritten;
  rewritten.reserve(args->size());
  bool changed = false;
  for (auto [index, argument] : llvm::enumerate(*args)) {
    if (!argument || argument->kind == "Starred")
      return nullptr;
    mlir::Type supplied = types.widenLiteral(types.inferExpr(argument.get()));
    int suppliedRung = numericTowerRung(types, supplied);
    int declaredRung = numericTowerRung(types, parameters[index]);
    llvm::StringRef constructor = spelling(parameters[index]);
    if (suppliedRung < 0 || declaredRung < 0 || suppliedRung >= declaredRung ||
        constructor.empty()) {
      rewritten.push_back(argument);
      continue;
    }
    rewritten.push_back(synth::call(synth::name(constructor, argument->range),
                                    std::vector<parser::NodePtr>{argument},
                                    argument->range));
    changed = true;
  }
  if (!changed)
    return nullptr;
  parser::NodePtr call = parser::makeNode("Call", expr.range);
  parser::addField(*call, "func",
                   std::get<parser::NodePtr>(
                       parser::findField(expr, "func")->value));
  parser::addField(*call, "args", std::move(rewritten));
  parser::addField(*call, "keywords", std::vector<parser::NodePtr>{});
  return call;
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
  // ⭐ A CALLEE REFERENCE THAT THE ARGUMENTS LEFT BEHIND IS RE-EMITTED HERE.
  // The callee is emitted before its arguments, and an argument that needs a
  // fused REGION (a comprehension, a reducer) ends the block it was emitted
  // in -- so the reference had to cross a block boundary. Inside a generator
  // the state machine then threads it as a frame lane, and a builtin callable
  // has no runtime object to put in one:
  //
  //     def g():
  //         print(sum([1, 2]))     # cannot adapt runtime bundle
  //         yield 0                # builtins.function with physical values ()
  //                                # to expected ABI (memref<8xi64>)
  //
  // A binding reference with no captures is a pure name lookup with no
  // operands, so re-emitting it where the call is built is exact -- and
  // binding the same call's argument to a local first has always worked,
  // which is the same value reached without the crossing.
  if (auto ref = callee.value.getDefiningOp<py::BindingRefOp>();
      ref && ref.getCaptures().empty() &&
      ref->getBlock() != builder.getInsertionBlock()) {
    auto reemitted = py::BindingRefOp::create(
        builder, ref.getLoc(), ref.getResult().getType(),
        ref.getBindingAttr(), mlir::ValueRange{});
    callee.value = reemitted.getResult();
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
  // ⭐ A LOCAL WINS OVER AN IMPORTED NAMESPACE. `from os import path` binds the
  // canonical symbol `path`, and that binding is visible while the STDLIB module
  // is compiled -- where posixpath's own `def normpath(path: str)` has a
  // parameter of that name:
  //
  //     from os import path
  //     print(path.basename("a/b.py"))
  //     # <stdlib>/posixpath.py:221:12: unresolved runtime binding 'path.split'
  //
  // Line 221 is `comps = path.split("/")`, a str method on that parameter, read
  // as a member of the importer's module alias. `import os` never collides
  // because nothing in the stdlib is called `os`; the collision is what the alias
  // brings.
  //
  // ⛔ The ROOT only. `a.b.c` where `a` is a local is a local's attribute chain
  // whatever `b` is, and a qualified symbol table cannot answer it.
  if (!calleeQualified.empty()) {
    llvm::StringRef root =
        llvm::StringRef(calleeQualified).split('.').first;
    if (!root.empty() && values.find(root) != values.end())
      calleeQualified.clear();
  }

  if (std::optional<Value> v = tryEmitSuperCall(expr, calleeNode))
    return *v;

  if (std::optional<Value> primitive =
          emitPrimitiveConstructorCall(expr, calleeNode))
    return *primitive;
  if (std::optional<Value> primitive = emitPrimitiveFactoryCall(expr, calleeNode))
    return *primitive;
  if (std::optional<Value> primitive =
          emitPrimitiveRuntimeCall(expr, calleeNode))
    return *primitive;

  // ⭐ A generator expression handed to a METHOD is materialized. The lazy
  // spellings all belong to name callees -- the reducers fuse it into an
  // accumulator loop, the container constructors into their build loop, and
  // a for-loop iterable into nested loops -- and those folds run below on
  // the unrewritten node. A method is a manifest native: it consumes the
  // whole iterable at the call, so a list is what it would have made anyway.
  // `"-".join(str(v) for v in xs)` was "unsupported expression kind
  // 'GeneratorExp'" followed by "builtins.str does not provide manifest
  // method 'join'" -- the argument had no type for the overload to match.
  //
  // A call to one of the lazy builtin iterators is materialized for the same
  // reason and by the same rule: `" ".join(map(str, xs))` handed the manifest
  // join a synthesized generator and died in the lowering. `list(...)` around
  // it is the spelling that already works.
  auto lazyArgument = [&](const parser::NodePtr &arg) {
    if (!arg || arg->kind != "Call")
      return false;
    const parser::Node *func = ast::node(*arg, "func");
    if (!func || func->kind != "Name")
      return false;
    llvm::StringRef name = ast::nameSpelling(*func);
    return (name == "zip" || name == "enumerate" || name == "map" ||
            name == "filter" || name == "reversed") &&
           isBuiltinIteratorName(name);
  };
  if (calleeNode && calleeNode->kind == "Attribute")
    if (const auto *args = ast::nodeList(expr, "args");
        args && llvm::any_of(*args, [&](const parser::NodePtr &arg) {
          return (arg && arg->kind == "GeneratorExp") || lazyArgument(arg);
        })) {
      parser::NodePtr rewritten = parser::makeNode("Call", expr.range);
      if (const parser::Field *func = parser::findField(expr, "func"))
        rewritten->fields.push_back(*func);
      std::vector<parser::NodePtr> rewrittenArgs;
      for (const parser::NodePtr &arg : *args) {
        if (arg && arg->kind == "GeneratorExp") {
          parser::NodePtr materialized =
              parser::makeNode("ListComp", arg->range);
          for (const parser::Field &field : arg->fields)
            materialized->fields.push_back(field);
          rewrittenArgs.push_back(std::move(materialized));
          continue;
        }
        if (lazyArgument(arg)) {
          parser::NodePtr listName = synth::name(std::string("list"), arg->range);
          parser::NodePtr materialized = synth::call(std::move(listName), std::vector<parser::NodePtr>{arg}, arg->range);
          rewrittenArgs.push_back(std::move(materialized));
          continue;
        }
        rewrittenArgs.push_back(arg);
      }
      parser::addField(*rewritten, "args", std::move(rewrittenArgs));
      if (const parser::Field *keywords = parser::findField(expr, "keywords"))
        rewritten->fields.push_back(*keywords);
      return emitCall(*rewritten);
    }

  // ⭐ A keyword spelling of a positional method argument is rewritten to the
  // position CPython's own signature gives it. `"a,b".split(sep=",")` and
  // `"aa".replace("a", "b", count=1)` are both accepted there and were
  // "builtins.str does not provide manifest method 'split' / 'replace'" here
  // -- the manifest names a str parameter across TWO physical values
  // (sep_header, sep_bytes), so a keyword can never match one by name.
  //
  // The table is CPython's argument clinic for the handful of methods that
  // accept keywords at all: find/index/center/startswith and the rest are
  // positional-only THERE too, and their refusal here is the same answer.
  // Binding runs through the same bindBuiltinArguments the builtin fast paths
  // use, so a gap (`split(maxsplit=1)`, which names no separator) stays
  // refused rather than being invented.
  if (calleeNode && calleeNode->kind == "Attribute")
    if (const auto *methodKeywords = ast::nodeList(expr, "keywords");
        methodKeywords && !methodKeywords->empty()) {
      struct MethodParameters {
        llvm::StringLiteral method;
        llvm::ArrayRef<llvm::StringRef> names;
        unsigned positionalOnly;
      };
      static const llvm::StringRef kSplitNames[] = {"sep", "maxsplit"};
      static const llvm::StringRef kReplaceNames[] = {"old", "new", "count"};
      static const llvm::StringRef kSplitlinesNames[] = {"keepends"};
      static const llvm::StringRef kCodecNames[] = {"encoding", "errors"};
      static const MethodParameters kMethodParameters[] = {
          {llvm::StringLiteral("split"), kSplitNames, 0},
          {llvm::StringLiteral("rsplit"), kSplitNames, 0},
          {llvm::StringLiteral("replace"), kReplaceNames, 2},
          {llvm::StringLiteral("splitlines"), kSplitlinesNames, 0},
          {llvm::StringLiteral("encode"), kCodecNames, 0},
          {llvm::StringLiteral("decode"), kCodecNames, 0},
      };
      llvm::StringRef methodName =
          ast::string(*calleeNode, "attr").value_or("");
      for (const MethodParameters &entry : kMethodParameters) {
        if (methodName != entry.method)
          continue;
        std::optional<llvm::SmallVector<const parser::Node *, 4>> bound =
            bindBuiltinArguments(expr, entry.names, entry.positionalOnly);
        if (!bound)
          break;
        std::vector<parser::NodePtr> positional;
        const auto *originalArgs = ast::nodeList(expr, "args");
        bool recovered = true;
        for (const parser::Node *argument : *bound) {
          parser::NodePtr shared;
          if (originalArgs)
            for (const parser::NodePtr &candidate : *originalArgs)
              if (candidate.get() == argument)
                shared = candidate;
          if (!shared)
            for (const parser::NodePtr &keyword : *methodKeywords)
              if (const parser::Field *value =
                      parser::findField(*keyword, "value");
                  value && std::holds_alternative<parser::NodePtr>(value->value) &&
                  std::get<parser::NodePtr>(value->value).get() == argument)
                shared = std::get<parser::NodePtr>(value->value);
          if (!shared) {
            recovered = false;
            break;
          }
          positional.push_back(std::move(shared));
        }
        if (!recovered)
          break;
        parser::NodePtr rewritten = parser::makeNode("Call", expr.range);
        if (const parser::Field *func = parser::findField(expr, "func"))
          rewritten->fields.push_back(*func);
        parser::addField(*rewritten, "args", std::move(positional));
        parser::addField(*rewritten, "keywords",
                         std::vector<parser::NodePtr>{});
        return emitCall(*rewritten);
      }
    }

  // ⭐ A method argument that is a CALL the inference cannot type is bound
  // first. `"x".translate(str.maketrans("l", "L"))` was "str.translate
  // requires a dict table" while the same call through a temporary --
  // `t = str.maketrans("l", "L"); "x".translate(t)` -- worked: inferExpr sees
  // `builtins.object` for the inner call and the emission sees the dict. The
  // binding is that temporary, written once. (Same shape as the lazy-iterator
  // and generator-expression materializations below; those two need a
  // container around them, this one only needs a name.)
  if (calleeNode && calleeNode->kind == "Attribute") {
    const auto *bindArgs = ast::nodeList(expr, "args");
    llvm::SmallVector<unsigned, 2> needBinding;
    if (bindArgs)
      for (auto [index, arg] : llvm::enumerate(*bindArgs)) {
        if (!arg || arg->kind != "Call")
          continue;
        mlir::Type inferred = types.widenLiteral(types.inferExpr(arg.get()));
        auto contract = mlir::dyn_cast_if_present<py::ContractType>(inferred);
        if (!inferred ||
            (contract && contract.getContractName() == "builtins.object"))
          needBinding.push_back(static_cast<unsigned>(index));
      }
    if (!needBinding.empty()) {
      llvm::SmallVector<std::string, 2> scratch;
      std::vector<parser::NodePtr> boundArgs(bindArgs->begin(),
                                             bindArgs->end());
      std::vector<parser::NodePtr> binds;
      for (unsigned index : needBinding) {
        std::string name = "__lyargbind" + std::to_string(++listCompCounter);
        scratch.push_back(name);
        parser::NodePtr target = synth::name(name, expr.range);
        parser::NodePtr bind = synth::assign(target, boundArgs[index], expr.range);
        binds.push_back(std::move(bind));
        parser::NodePtr named = synth::name(name, expr.range);
        boundArgs[index] = std::move(named);
      }
      parser::NodePtr rewritten = parser::makeNode("Call", expr.range);
      if (const parser::Field *func = parser::findField(expr, "func"))
        rewritten->fields.push_back(*func);
      parser::addField(*rewritten, "args", std::move(boundArgs));
      if (const parser::Field *keywords = parser::findField(expr, "keywords"))
        rewritten->fields.push_back(*keywords);
      Value bound = emitNone(expr);
      runWithScratchNames(scratch, [&] {
        for (const parser::NodePtr &bind : binds)
          emitStatement(*bind);
        bound = emitCall(*rewritten);
      });
      return bound;
    }
  }

  // ⭐ A module attribute that the runtime does not provide says so as a
  // MODULE attribute. `math.gcd(12, 8)` reported "static type
  // builtins.object does not provide manifest method 'gcd'" -- the module
  // namespace root is typed `object` (it is a lookup root, not a receiver),
  // so the report described that placeholder instead of the module, and the
  // reader had no way to tell a missing function from a broken call.
  if (calleeNode && calleeNode->kind == "Attribute") {
    const parser::Node *receiverNode = ast::node(*calleeNode, "value");
    llvm::StringRef attribute =
        ast::string(*calleeNode, "attr").value_or("");
    if (receiverNode && receiverNode->kind == "Name" && !attribute.empty()) {
      llvm::StringRef space = ast::nameSpelling(*receiverNode);
      std::string qualified = (space + "." + attribute).str();
      const py::protocols::Table &table = py::protocols::Table::get(context);
      std::optional<mlir::Type> spaceType = types.lookupSymbol(space);
      // The namespace root is the object placeholder and nothing else: a
      // local of the same name (or a real value) keeps its own dispatch.
      // ⛔ AND THE ALIAS SPELLING. `lookupSourceModule` takes the module's
      // own name, so `import m as x` left `x` unrecognised here and the call
      // fell through to a receiver the module namespace has no value for --
      // reported against `x` itself rather than against the attribute it does
      // not have. `isImportedModuleName` is the binding, under whichever
      // spelling the import chose.
      bool knownModule = !table.moduleCallableExports(space).empty() ||
                         lookupSourceModule(space) != nullptr ||
                         types.isImportedModuleName(space);
      if (knownModule &&
          !types.lookupSymbol(qualified) && !types.lookupClass(qualified) &&
          !values.count(space) && spaceType && *spaceType == types.object()) {
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, expr.range.start,
            "module '" + space.str() + "' has no attribute '" +
                attribute.str() + "' in this runtime"});
        return emitNone(expr);
      }
    }
  }

  // ⭐ `s.startswith((a, b))` is `s.startswith(a) or s.startswith(b)`, which
  // is what CPython's C loop over the tuple computes. The tuple form was
  // "builtins.str does not provide manifest method 'startswith'" -- the
  // manifest declares the str parameter, and there is no second
  // implementation to declare: the answer is a disjunction of the one that
  // exists.
  //
  // The receiver is emitted ONCE (captured in the first disjunct, named by
  // the rest), and the disjunction is RIGHT-nested for the same reason the
  // comparison chain is: a capture is defined in the arm that made it, so a
  // left-nested tree puts later arms where it does not dominate them.
  if (calleeNode && calleeNode->kind == "Attribute") {
    llvm::StringRef affix = ast::string(*calleeNode, "attr").value_or("");
    const auto *affixArgs = ast::nodeList(expr, "args");
    const auto *affixKeywords = ast::nodeList(expr, "keywords");
    const parser::Node *tupleArg =
        affixArgs && !affixArgs->empty() ? affixArgs->front().get() : nullptr;
    if ((affix == "startswith" || affix == "endswith") && tupleArg &&
        tupleArg->kind == "Tuple" && affixArgs->size() <= 3 &&
        (!affixKeywords || affixKeywords->empty())) {
      const auto *elts = ast::nodeList(*tupleArg, "elts");
      const parser::Field *receiverField =
          parser::findField(*calleeNode, "value");
      if (elts && receiverField &&
          std::holds_alternative<parser::NodePtr>(receiverField->value)) {
        if (elts->empty()) {
          auto constant = py::BoolConstantOp::create(
              builder, loc(expr), types.literal("False"),
              builder.getBoolAttr(false));
          return Value{constant.getResult(), types.literal("False")};
        }
        parser::NodePtr receiver =
            std::get<parser::NodePtr>(receiverField->value);
        std::string subject =
            "__lyaffix" + std::to_string(++listCompCounter);
        auto affixCall = [&](std::size_t index) {
          parser::NodePtr self = synth::name(subject, expr.range);
          parser::NodePtr attribute = synth::attribute(std::move(self), affix.str(), calleeNode->range);
          std::vector<parser::NodePtr> callArgs{(*elts)[index]};
          for (std::size_t rest = 1; rest < affixArgs->size(); ++rest)
            callArgs.push_back((*affixArgs)[rest]);
          parser::NodePtr call = synth::call(std::move(attribute), std::move(callArgs), expr.range);
          return call;
        };
        parser::NodePtr folded = affixCall(elts->size() - 1);
        for (std::size_t index = elts->size() - 1; index > 0; --index) {
          parser::NodePtr disjunction = parser::makeNode("BoolOp", expr.range);
          parser::addField(*disjunction, "op",
                           parser::makeNode("Or", expr.range));
          parser::addField(*disjunction, "values",
                           std::vector<parser::NodePtr>{affixCall(index - 1),
                                                        std::move(folded)});
          folded = std::move(disjunction);
        }
        parser::NodePtr target = synth::name(subject, expr.range);
        parser::NodePtr bind = synth::assign(std::move(target), receiver, expr.range);
        Value result = emitNone(expr);
        runWithScratchNames({subject}, [&] {
          emitStatement(*bind);
          result = emitExpr(folded.get());
        });
        return result;
      }
    }
  }

  // ⭐ Two argument spellings that ARE the no-argument one, folded where
  // CPython's own C dispatch folds them: `s.split(None)` is the whitespace
  // split (sep=None IS the default), and `s.encode("utf-8")` is the default
  // encoding. Both were "does not provide manifest method" for a method the
  // no-argument spelling right next to them resolves. A non-literal argument
  // is left alone -- the fold is only for what is decidable here.
  if (calleeNode && calleeNode->kind == "Attribute") {
    llvm::StringRef method = ast::string(*calleeNode, "attr").value_or("");
    const auto *args = ast::nodeList(expr, "args");
    const auto *keywords = ast::nodeList(expr, "keywords");
    bool single = args && args->size() == 1 && args->front() &&
                  (!keywords || keywords->empty());
    bool defaulted = false;
    if (single && (method == "split" || method == "rsplit"))
      defaulted = args->front()->kind == "Constant" &&
                  ast::isNoneField(*args->front(), "value");
    // ⭐ `s.split(None, k)` and `s.split(maxsplit=k)` are the whitespace split
    // WITH a cap -- CPython's own two spellings of it -- and both were "does
    // not provide manifest method 'split'". The cap is dropped onto the
    // whitespace overload, which takes it as its only argument; rsplit is
    // left alone because its cap withholds splits from the right, which the
    // manifest's left-to-right walk cannot produce.
    if (method == "split") {
      const parser::Node *cap = nullptr;
      if (args && args->size() == 2 && args->front() && (*args)[1] &&
          (!keywords || keywords->empty()) &&
          args->front()->kind == "Constant" &&
          ast::isNoneField(*args->front(), "value"))
        cap = (*args)[1].get();
      if ((!args || args->empty()) && keywords && keywords->size() == 1)
        if (const parser::NodePtr &keyword = keywords->front();
            keyword && ast::string(*keyword, "arg").value_or("") == "maxsplit")
          cap = ast::node(*keyword, "value");
      if (cap) {
        parser::NodePtr shared;
        if (args)
          for (const parser::NodePtr &candidate : *args)
            if (candidate.get() == cap)
              shared = candidate;
        if (keywords && !shared)
          for (const parser::NodePtr &keyword : *keywords)
            if (const parser::Field *value =
                    parser::findField(*keyword, "value");
                value && std::holds_alternative<parser::NodePtr>(value->value) &&
                std::get<parser::NodePtr>(value->value).get() == cap)
              shared = std::get<parser::NodePtr>(value->value);
        if (shared) {
          parser::NodePtr rewritten = parser::makeNode("Call", expr.range);
          if (const parser::Field *func = parser::findField(expr, "func"))
            rewritten->fields.push_back(*func);
          parser::addField(*rewritten, "args",
                           std::vector<parser::NodePtr>{shared});
          parser::addField(*rewritten, "keywords",
                           std::vector<parser::NodePtr>{});
          return emitCall(*rewritten);
        }
      }
    }
    if (single && (method == "encode" || method == "decode"))
      if (args->front()->kind == "Constant")
        if (std::optional<llvm::StringRef> encoding =
                ast::string(*args->front(), "value"))
          defaulted = encoding->equals_insensitive("utf-8") ||
                      encoding->equals_insensitive("utf8");
    if (defaulted) {
      parser::NodePtr rewritten = parser::makeNode("Call", expr.range);
      if (const parser::Field *func = parser::findField(expr, "func"))
        rewritten->fields.push_back(*func);
      parser::addField(*rewritten, "args", std::vector<parser::NodePtr>{});
      parser::addField(*rewritten, "keywords", std::vector<parser::NodePtr>{});
      return emitCall(*rewritten);
    }
  }

  // ⭐ A one-argument builtin whose whole job is to call a dunder calls the
  // SOURCE class's when there is one. Each of these is a manifest free
  // function with builtin overloads, so a user class reached them as
  // "!py.overload<...> is not callable with these arguments" or as a missing
  // manifest method -- while the operators next to them dispatch the class's
  // own dunder. One table rather than one special case per builtin: they
  // differ only in the name.
  if (calleeNode && calleeNode->kind == "Name") {
    static constexpr std::pair<llvm::StringLiteral, llvm::StringLiteral>
        kDunderBuiltins[] = {
            {llvm::StringLiteral("abs"), llvm::StringLiteral("__abs__")},
            {llvm::StringLiteral("int"), llvm::StringLiteral("__int__")},
            {llvm::StringLiteral("float"), llvm::StringLiteral("__float__")},
            {llvm::StringLiteral("round"), llvm::StringLiteral("__round__")},
            {llvm::StringLiteral("reversed"),
             llvm::StringLiteral("__reversed__")},
            {llvm::StringLiteral("bytes"), llvm::StringLiteral("__bytes__")},
            {llvm::StringLiteral("complex"),
             llvm::StringLiteral("__complex__")},
        };
    // ⛔ Not len() and not hash(): those two already reach a source class
    // through their own folds, and hash's fold is not just the call -- it
    // remaps CPython's -1 error sentinel to -2. Routing it here bypassed the
    // remap and hash_sentinel_and_math_domain_text printed -1.
    llvm::StringRef builtinName = ast::nameSpelling(*calleeNode);
    const auto *dunderArgs = ast::nodeList(expr, "args");
    const auto *dunderKeywords = ast::nodeList(expr, "keywords");
    // ⛔ `round` IS THE ONE WITH A SECOND ARGUMENT, and `round(c, 2)` reached
    // the lowering as "runtime manifest has no C.__round__ method" while
    // `round(c)` dispatched here. Only round: `int(x, base)` is a different
    // manifest overload and must NOT be routed to `__int__`.
    std::size_t maximumArguments = builtinName == "round" ? 2u : 1u;
    if (dunderArgs && !dunderArgs->empty() &&
        dunderArgs->size() <= maximumArguments &&
        llvm::none_of(*dunderArgs,
                      [](const parser::NodePtr &argument) {
                        return !argument || argument->kind == "Starred";
                      }) &&
        (!dunderKeywords || dunderKeywords->empty()) &&
        !programBindsName(builtinName))
      for (const auto &[name, dunder] : kDunderBuiltins) {
        if (builtinName != name)
          continue;
        mlir::Type argumentType =
            types.widenLiteral(types.inferExpr(dunderArgs->front().get()));
        // The presence check comes first because the argument must not be
        // emitted twice: if no source class provides the dunder, the manifest
        // path below emits it.
        if (lookupClassMethod(argumentType, dunder)) {
          Value receiver = emitExpr(dunderArgs->front().get());
          llvm::SmallVector<Value, 1> extra;
          for (const parser::NodePtr &argument : llvm::drop_begin(*dunderArgs))
            extra.push_back(emitExpr(argument.get()));
          if (std::optional<Value> dispatched =
                  tryEmitClassDunder(expr, receiver, dunder, extra))
            return *dispatched;
        }
        break;
      }
  }

  // ⭐ AND THE MODULE FUNCTIONS WHOSE JOB IS THE SAME ONE. `math.floor(x)`
  // calls `type(x).__floor__(x)` in CPython, so a class that provides one is
  // the argument these take -- and it reached the manifest signature instead
  // ("!py.callable<[builtins.float], returns = [builtins.int]> is not
  // callable: call arguments do not match the Callable contract"), while
  // `abs()` and `round()` next to it dispatched the class's own dunder. Same
  // table shape as the builtins above, keyed by the qualified name.
  if (calleeNode && calleeNode->kind == "Attribute") {
    static constexpr std::pair<llvm::StringLiteral, llvm::StringLiteral>
        kDunderModuleFunctions[] = {
            {llvm::StringLiteral("math.floor"),
             llvm::StringLiteral("__floor__")},
            {llvm::StringLiteral("math.ceil"),
             llvm::StringLiteral("__ceil__")},
            {llvm::StringLiteral("math.trunc"),
             llvm::StringLiteral("__trunc__")},
        };
    const parser::Node *spaceNode = ast::node(*calleeNode, "value");
    llvm::StringRef attribute = ast::string(*calleeNode, "attr").value_or("");
    const auto *moduleArgs = ast::nodeList(expr, "args");
    const auto *moduleKeywords = ast::nodeList(expr, "keywords");
    if (spaceNode && spaceNode->kind == "Name" && !attribute.empty() &&
        moduleArgs && moduleArgs->size() == 1 && moduleArgs->front() &&
        moduleArgs->front()->kind != "Starred" &&
        (!moduleKeywords || moduleKeywords->empty())) {
      llvm::StringRef space = ast::nameSpelling(*spaceNode);
      // A local of the module's name keeps its own dispatch, exactly as the
      // module-attribute diagnostic above requires.
      if (!values.count(space)) {
        std::string qualified = (space + "." + attribute).str();
        for (const auto &[name, dunder] : kDunderModuleFunctions) {
          if (qualified != name)
            continue;
          mlir::Type argumentType =
              types.widenLiteral(types.inferExpr(moduleArgs->front().get()));
          // Presence first, so the argument is not emitted twice when the
          // manifest path below is the one that runs.
          if (lookupClassMethod(argumentType, dunder)) {
            Value receiver = emitExpr(moduleArgs->front().get());
            if (std::optional<Value> dispatched =
                    tryEmitClassDunder(expr, receiver, dunder))
              return *dispatched;
          }
          break;
        }
      }
    }
  }

  // ⭐ `divmod(x, y)` on a SOURCE class calls its __divmod__, for the same
  // reason the one-argument builtins above call theirs: the builtin is a
  // manifest free function with numeric overloads, and a user class reached
  // it as "!py.callable<[int, int], ...> is not callable with these
  // arguments". Two arguments, so the receiver is the first and the rest are
  // the method's.
  if (calleeNode && calleeNode->kind == "Name" &&
      ast::nameSpelling(*calleeNode) == "divmod" &&
      !programBindsName("divmod")) {
    const auto *divArgs = ast::nodeList(expr, "args");
    const auto *divKeywords = ast::nodeList(expr, "keywords");
    if (divArgs && divArgs->size() == 2 && divArgs->front() &&
        (*divArgs)[1] && divArgs->front()->kind != "Starred" &&
        (*divArgs)[1]->kind != "Starred" &&
        (!divKeywords || divKeywords->empty())) {
      mlir::Type receiverType =
          types.widenLiteral(types.inferExpr(divArgs->front().get()));
      if (lookupClassMethod(receiverType, "__divmod__")) {
        Value receiver = emitExpr(divArgs->front().get());
        llvm::SmallVector<Value, 1> rest{emitExpr((*divArgs)[1].get())};
        if (std::optional<Value> dispatched =
                tryEmitClassDunder(expr, receiver, "__divmod__", rest))
          return *dispatched;
      }
      // ⭐ `divmod` OVER FLOATS IS THE PAIR ITS OPERATORS ALREADY ANSWER.
      // CPython defines divmod(x, y) and (x // y, x % y) together -- the same
      // quotient and the same remainder -- and the manifest's divmod is typed
      // [int, int], so `divmod(7.5, 2)` was "!py.callable<[int, int], ...> is
      // not callable with these arguments" for a pair this compiler can
      // already compute.
      //
      // ⛔ The operands are bound to temporaries first: `divmod(f(), g())`
      // names each of them twice below, and CPython calls each once.
      mlir::Type otherType =
          types.widenLiteral(types.inferExpr((*divArgs)[1].get()));
      if (receiverType == types.floatType() || otherType == types.floatType()) {
        unsigned serial = ++syntheticFunctionCounter;
        std::string left = "__divmodl" + std::to_string(serial);
        std::string right = "__divmodr" + std::to_string(serial);
        Value leftValue = emitExpr(divArgs->front().get());
        Value rightValue = emitExpr((*divArgs)[1].get());
        values[left] = leftValue;
        types.bindSymbol(left, leftValue.type);
        values[right] = rightValue;
        types.bindSymbol(right, rightValue.type);
        auto operand = [&](const std::string &name) {
          return synth::name(name, expr.range);
        };
        parser::NodePtr pair = parser::makeNode("Tuple", expr.range);
        parser::addField(
            *pair, "elts",
            std::vector<parser::NodePtr>{
                synth::binOp(operand(left), "FloorDiv", operand(right),
                             expr.range),
                synth::binOp(operand(left), "Mod", operand(right),
                             expr.range)});
        synthesizedIteratorDefs.push_back(pair);
        Value built = emitExpr(pair.get());
        values.erase(left);
        values.erase(right);
        return built;
      }
    }
  }

  if (std::optional<Value> v = tryEmitTypeCall(expr, calleeNode))
    return *v;
  if (std::optional<Value> v = tryEmitHasattrCall(expr, calleeNode))
    return *v;
  if (std::optional<Value> v = tryEmitCallableCall(expr, calleeNode))
    return *v;
  if (std::optional<Value> v = tryEmitGetattrCall(expr, calleeNode))
    return *v;
  if (std::optional<Value> v = tryEmitSetattrCall(expr, calleeNode))
    return *v;
  if (std::optional<Value> v = tryEmitNamedTupleReplace(expr, calleeNode))
    return *v;
  if (std::optional<Value> v = tryEmitIsInstanceCall(expr, calleeNode))
    return *v;
  if (std::optional<Value> v = tryEmitIntBaseCall(expr, calleeNode))
    return *v;
  if (std::optional<Value> v = tryEmitIntCall(expr, calleeNode))
    return *v;
  if (std::optional<Value> v = tryEmitFloatCall(expr, calleeNode))
    return *v;
  if (std::optional<Value> v = tryEmitPowCall(expr, calleeNode))
    return *v;
  if (std::optional<Value> v = tryEmitBoolCall(expr, calleeNode))
    return *v;
  if (std::optional<Value> v = tryEmitAsciiCall(expr, calleeNode))
    return *v;
  if (std::optional<Value> v = tryEmitIssubclassCall(expr, calleeNode))
    return *v;
  // frozenset() with no argument forwards an empty list to the single
  // manifest __new__ (one native per initializer name; a memref span cannot
  // carry a default).
  if (calleeNode && calleeNode->kind == "Name" &&
      ast::nameSpelling(*calleeNode) == "frozenset" &&
      !programBindsName("frozenset") && callHasNoArguments(expr)) {
    parser::NodePtr emptyList = parser::makeNode("List", expr.range);
    parser::addField(*emptyList, "elts", std::vector<parser::NodePtr>{});
    parser::NodePtr rewritten = parser::makeNode("Call", expr.range);
    parser::NodePtr frozensetName = synth::name(std::string("frozenset"), expr.range);
    parser::addField(*rewritten, "func", std::move(frozensetName));
    parser::addField(*rewritten, "args",
                     std::vector<parser::NodePtr>{std::move(emptyList)});
    parser::addField(*rewritten, "keywords", std::vector<parser::NodePtr>{});
    synthesizedIteratorDefs.push_back(rewritten);
    return emitCall(*rewritten);
  }
  // input() with no prompt forwards an empty prompt to the single manifest
  // contract (input(prompt: str) -> str).
  if (calleeNode && calleeNode->kind == "Name" &&
      ast::nameSpelling(*calleeNode) == "input" &&
      !programBindsName("input") && callHasNoArguments(expr)) {
    parser::NodePtr empty = synth::strConstant(std::string(), expr.range);
    parser::NodePtr promptCall = parser::makeNode("Call", expr.range);
    parser::NodePtr inputName = synth::name(std::string("input"), expr.range);
    parser::addField(*promptCall, "func", std::move(inputName));
    parser::addField(*promptCall, "args",
                     std::vector<parser::NodePtr>{std::move(empty)});
    parser::addField(*promptCall, "keywords", std::vector<parser::NodePtr>{});
    synthesizedIteratorDefs.push_back(promptCall);
    return emitCall(*promptCall);
  }
  if (std::optional<Value> v = tryEmitStrCall(expr, calleeNode))
    return *v;
  if (std::optional<Value> v = tryEmitListCall(expr, calleeNode))
    return *v;
  if (std::optional<Value> v = tryEmitContainerConstructorCall(expr, calleeNode))
    return *v;
  if (std::optional<Value> v = tryEmitPrintCall(expr, calleeNode))
    return *v;
  if (std::optional<Value> v = tryEmitReducerCall(expr, calleeNode))
    return *v;
  if (std::optional<Value> v = tryEmitLazyIteratorValueCall(expr, calleeNode))
    return *v;
  if (std::optional<Value> v = tryEmitItertoolsValueCall(expr, calleeNode))
    return *v;
  if (std::optional<Value> v = tryEmitDictMethodSugar(expr, calleeNode))
    return *v;
  if (std::optional<Value> v = tryEmitSortSugar(expr, calleeNode))
    return *v;
  if (std::optional<Value> v = tryEmitStrTranslateSugar(expr, calleeNode))
    return *v;

  // `C[int](...)` spells its instantiation at the call site, so it needs no
  // annotated context. The subscript is a class object, never a __getitem__.
  if (mlir::Type instantiated = types.genericClassSubscript(calleeNode)) {
    llvm::StringRef instantiatedName =
        contractName(instantiated).value_or(llvm::StringRef());
    return emitClassInstantiation(expr, instantiatedName, instantiated);
  }

  // A bare `C(...)` on a generic class: the arguments themselves may determine
  // the instantiation through __init__'s parameter types (`Box(5)`). Failing
  // that, emitExprExpected already had its chance at an annotated context, and
  // the py ABI cannot carry the type parameter — so this is the class-side twin
  // of the generic function's "requires a call or an annotated Callable
  // context".
  auto emitBareGenericInstantiation =
      [&](llvm::StringRef base) -> std::optional<Value> {
    if (!lookupGenericClass(base))
      return std::nullopt;
    if (mlir::Type solved = inferredGenericClassInstantiation(expr))
      return emitClassInstantiation(
          expr, mlir::cast<py::ContractType>(solved).getContractName(), solved);
    diagnoseUngroundedGenericClass(expr, base);
    return emitNone(expr);
  };

  // Same rule as the bare-Name constructor path below: a top-level `def int`
  // outranks the builtin class contract of that spelling. A bare Name has a
  // qualified spelling equal to itself, so this branch sees `int` first.
  if (!calleeQualified.empty() && !moduleFunctionNames.count(calleeQualified))
    if (auto cls = types.lookupClass(calleeQualified)) {
      if (std::optional<llvm::StringRef> symbol = contractName(*cls)) {
        if (std::optional<Value> v =
                rejectStubSourceCall(expr, *symbol, /*instantiation=*/true))
          return *v;
        if (std::optional<Value> v = emitBareGenericInstantiation(*symbol))
          return *v;
      }
      return emitClassInstantiation(expr, llvm::StringRef(calleeQualified),
                                    *cls);
    }

  if (calleeNode && calleeNode->kind == "Name") {
    llvm::StringRef name = ast::nameSpelling(*calleeNode);
    // A top-level `def str` outranks the builtin `str` class contract. The
    // guard is deliberately narrower than programBindsName: a VALUE binding
    // that happens to share a class's spelling keeps reaching the constructor
    // as it does today, because only a `def` introduces a competing callable
    // under the same top-level name.
    if (auto cls = moduleFunctionNames.count(name) ? std::optional<mlir::Type>()
                                                   : types.lookupClass(name)) {
      if (std::optional<llvm::StringRef> symbol = contractName(*cls)) {
        if (std::optional<Value> v =
                rejectStubSourceCall(expr, *symbol, /*instantiation=*/true))
          return *v;
        if (std::optional<Value> v = emitBareGenericInstantiation(*symbol))
          return *v;
      }
      return emitClassInstantiation(expr, name, *cls);
    }
  }

  if (std::optional<Value> v = tryEmitLenCall(expr, calleeNode))
    return *v;
  if (std::optional<Value> v = tryEmitNextCall(expr, calleeNode))
    return *v;
  if (std::optional<Value> v = tryEmitRoundCall(expr, calleeNode))
    return *v;
  if (std::optional<Value> v = tryEmitHashCall(expr, calleeNode))
    return *v;

  if (std::optional<Value> primitiveCall =
          emitDirectPrimitiveFunctionCall(expr, calleeNode))
    return *primitiveCall;

  if (std::optional<Value> v = tryEmitReprCall(expr, calleeNode))
    return *v;
  if (std::optional<Value> v = tryEmitFormatCall(expr, calleeNode))
    return *v;

  if (calleeNode && calleeNode->kind == "Name") {
    llvm::StringRef name = ast::nameSpelling(*calleeNode);
    auto local = values.find(name);
    if (local != values.end() && local->second.boundMethod) {
      const BoundMethodValue &bound = *local->second.boundMethod;
      // ⭐ A BOUND METHOD DISPATCHES TOO. `x.f()` on a base-typed receiver goes
      // through the synthesized dispatcher; `m = x.f` then `m()` inlined the
      // STATIC method instead, so the same question asked in two spellings
      // answered the base's body for a subclass instance:
      //
      //     x: Base = Sub()
      //     m = x.f
      //     print(m())      # printed "B"; CPython prints "S"
      //
      // silently, and `x.f()` one line over was right. The receiver and the
      // method are both in hand here, which is everything the dispatcher takes.
      //
      // ⛔ The helper is asked BEFORE the arguments are emitted, the same
      // order `tryEmitVirtualMethodCall` keeps: a bail after emitting them
      // would evaluate every argument twice.
      std::optional<std::string_view> boundName =
          bound.method.method ? ast::string(*bound.method.method, "name")
                              : std::nullopt;
      const auto *boundKeywords = ast::nodeList(expr, "keywords");
      const auto *boundArgs = ast::nodeList(expr, "args");
      bool plainCall = boundName && (!boundKeywords || boundKeywords->empty());
      if (plainCall && boundArgs)
        for (const parser::NodePtr &argument : *boundArgs)
          if (argument && argument->kind == "Starred")
            plainCall = false;
      if (plainCall &&
          virtualDispatcherFor(
              expr, bound.receiver, *boundName,
              boundArgs ? static_cast<unsigned>(boundArgs->size()) : 0)) {
        llvm::SmallVector<Value, 4> positional;
        if (boundArgs)
          for (const parser::NodePtr &argument : *boundArgs)
            positional.push_back(emitExpr(argument.get()));
        if (std::optional<Value> dispatched = tryEmitVirtualDispatchWithValues(
                expr, bound.receiver, *boundName, positional))
          return *dispatched;
      }
      return emitInlineMethodCall(expr, bound.receiver, bound.method);
    }
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
      // Qualified references to imported generics (module.fn(...)) resolve
      // through the canonical binding to the same registration the bare
      // import-name path uses.
      auto generic = genericFunctions.find(binding);
      if (generic != genericFunctions.end())
        return emitGenericCall(expr, *calleeNode, generic->second);
      if (auto mono = monomorphicFunctions.find(binding);
          mono != monomorphicFunctions.end() &&
          mayArgumentSpecialize(expr, mono->second))
        return emitArgumentSpecializedCall(
            expr, *calleeNode, mono->second,
            emitBindingRef(*calleeNode, binding, *symbol));
      mlir::Type resultOverride =
          binding == "asyncio.sleep" ? types.inferExpr(&expr) : mlir::Type();
      Value callee = emitBindingRef(*calleeNode, binding, *symbol);
      auto declaredCallable =
          mlir::dyn_cast_if_present<py::CallableType>(callee.type);
      // ⭐ `math.sqrt(16)` CONVERTS, and `def p(x: float)` reached by `p(3)`
      // does not. Both were refused with "call arguments do not match the
      // Callable contract" and only one of them should have been.
      //
      // The difference is what is on the other side of the parameter. A Python
      // body keeps whatever it was handed -- CPython leaves the annotation
      // inert, so `p(3)` sees an int, which is why THAT boundary is answered
      // by emitting a second body at the argument's rung
      // (`emitArgumentSpecializedCall`) and never by converting
      // (tests/probe/wb_argument_boundary_numeric_tower.py). A manifest export
      // is C against a double: there is no Python-visible parameter to keep an
      // int in, and CPython converts through `__float__` at the boundary. So
      // the two rules agree rather than compete, and this is the arm for the
      // second one.
      //
      // ⛔ Why `freeFunctionContract` is the discriminator and not "the
      // binding has a dot in it": a source module's function is reached
      // through the same qualified path and must keep the Python rule. That
      // table holds exactly the manifest's `ly.typing.function_contracts`, so
      // asking it IS asking whether the callee is one.
      parser::NodePtr widened =
          widenNumericArgumentsForManifestCall(expr, binding, declaredCallable);
      return emitCallableDispatch(
          expr, callee,
          emitCallOperands(widened ? *widened : expr, {},
                           /*includeAstArguments=*/true, declaredCallable),
          resultOverride);
    }
  }

  if (std::optional<Value> v = tryEmitStrFormatCall(expr, calleeNode))
    return *v;

  if (calleeNode && calleeNode->kind == "Attribute") {
    if (const parser::Node *receiverNode = ast::node(*calleeNode, "value")) {
      if (auto methodName = ast::string(*calleeNode, "attr")) {
        // ⭐ A FIELD holding a type object CONSTRUCTS when called, and the
        // class is in the field's type. The receiver is emitted first because
        // it still has to happen -- `Box(Other).t(5)` builds a Box -- and then
        // the call is re-spelled as the class name, which is the path a class
        // NAME already takes. Reaching the lowering instead reports "calling a
        // type object held in a value is not supported", which describes the
        // compiler rather than the program.
        if (auto typeObject = mlir::dyn_cast_if_present<py::TypeType>(
                types.widenLiteral(types.inferExpr(calleeNode)))) {
          std::string spelling;
          if (auto contract = mlir::dyn_cast_if_present<py::ContractType>(
                  typeObject.getInstanceType())) {
            llvm::StringRef name = contract.getContractName();
            auto tail = name.rsplit('.').second;
            spelling = tail.empty() ? name.str() : tail.str();
          }
          if (!spelling.empty() &&
              types.lookupClass(spelling) == typeObject.getInstanceType()) {
            emitExpr(receiverNode);
            parser::NodePtr named = synth::name(spelling, calleeNode->range);
            parser::NodePtr rewritten = parser::makeNode("Call", expr.range);
            parser::addField(*rewritten, "func", named);
            for (llvm::StringRef field : {"args", "keywords"})
              if (const parser::Field *original = parser::findField(expr, field))
                rewritten->fields.push_back(*original);
            return emitCall(*rewritten);
          }
        }
        // ⭐ `str.upper(s)` IS `s.upper()`, and the receiver is the first
        // argument. Written by hand it is ordinary Python; reached through a
        // `map(str.upper, xs)` whose fast path re-spells the callable as a
        // call, it is how the idiom gets here at all. Both were refused with
        // "!py.type<builtins.str> does not provide manifest method 'upper'",
        // which is the compiler saying it looked on the class object.
        if (types.manifestMethodReceiverContract(
                types.widenLiteral(types.inferExpr(receiverNode)),
                *methodName)) {
          const auto *unboundArgs = ast::nodeList(expr, "args");
          if (unboundArgs && !unboundArgs->empty() && unboundArgs->front() &&
              unboundArgs->front()->kind != "Starred") {
            std::vector<parser::NodePtr> rest(std::next(unboundArgs->begin()),
                                              unboundArgs->end());
            parser::NodePtr rewritten = parser::makeNode("Call", expr.range);
            parser::addField(*rewritten, "func",
                             synth::attribute(unboundArgs->front(), *methodName,
                                              calleeNode->range));
            parser::addField(*rewritten, "args", std::move(rest));
            if (const parser::Field *keywords =
                    parser::findField(expr, "keywords"))
              rewritten->fields.push_back(*keywords);
            else
              parser::addField(*rewritten, "keywords",
                               std::vector<parser::NodePtr>{});
            synthesizedIteratorDefs.push_back(rewritten);
            return emitCall(*rewritten);
          }
        }
        Value receiver = emitExpr(receiverNode);
        if (dispatchIsUnresolvable(receiver, *methodName, receiverNode,
                                   /*throughSuper=*/false)) {
          if (std::optional<Value> dispatched = tryEmitVirtualDispatch(
                  expr, *calleeNode, receiverNode, receiver, *methodName))
            return *dispatched;
        }
        if (refuseUnresolvableDispatch(*calleeNode, receiver, *methodName,
                                       receiverNode))
          return emitNone(expr);
        if (std::optional<MethodBinding> method =
                lookupClassMethod(receiver.type, *methodName)) {
          // A generator method is routed through the bound function object for
          // the same reason an async one is: inlining substitutes the body's
          // OWN result (the StopIteration value, `None` for a bare `yield`),
          // and a suspendable body has no straight-line expansion to
          // substitute in the first place. The bound clone carries
          // publicCallable, whose result is the generator object.
          bool suspendable = method->async ||
                             method->bodySignature.isGeneratorFunction ||
                             method->bodySignature.isAsyncGeneratorFunction;
          if (suspendable && !method->symbolName.empty()) {
            // ⭐ A DIRECT CALL TAKES THE SYMBOL, with the receiver as the
            // leading positional -- the same route a recursive method takes.
            // The bound object captures the receiver in a CLOSURE, and a
            // generator's resume clone has no lane for a capture: its argument
            // lanes are built from the callable's POSITIONALS, so
            //
            //     class Bag:
            //         def each(self):
            //             for x in self.xs:
            //                 yield x
            //     # a generator cannot carry a value of contract 'Bag' across
            //     # a suspension yet ... and a user class has neither
            //
            // -- and an EMPTY class refused too, which is what says the layout
            // was never the problem. As a positional the receiver rides the
            // argument lane that already exists for a source class.
            //
            // ⛔ Only a direct call with no keywords, and only an instance
            // method. `m = b.each` still builds the bound object (there is no
            // call to attach the receiver to), and a keyword would need the
            // slot placement the recursive path does; both keep the old route,
            // which is correct wherever the frame does not have to carry the
            // receiver.
            const auto *callKeywords = ast::nodeList(expr, "keywords");
            // The PUBLIC callable: the symbol's own `callable_type` returns
            // the body result (None for a generator), and the value a call
            // produces is the generator object.
            py::CallableType directCallable =
                mlir::dyn_cast_if_present<py::CallableType>(
                    method->signature.publicCallable);
            if (directCallable && method->kind == "instance" &&
                methodBindingBindsReceiver(*method) &&
                (!callKeywords || callKeywords->empty()) &&
                !mlir::isa<py::TypeType>(receiver.type)) {
              Value callee =
                  emitBindingRef(*calleeNode, method->symbolName, directCallable);
              CallOperands operands = emitCallOperands(expr);
              operands.positional.insert(operands.positional.begin(), receiver);
              operands.positionalTypes.insert(operands.positionalTypes.begin(),
                                              receiver.type);
              operands.positionalUnpacked.insert(
                  operands.positionalUnpacked.begin(), 0);
              return emitCallableDispatch(expr, callee, operands);
            }
            return emitCallableDispatch(
                expr, emitMethodObject(*calleeNode, receiver, *method),
                emitCallOperands(expr));
          }
          return emitInlineMethodCall(expr, receiver, *method);
        }
        if (*methodName == "__str__") {
          const auto *strArgs = ast::nodeList(expr, "args");
          const auto *strKeywords = ast::nodeList(expr, "keywords");
          if ((!strArgs || strArgs->empty()) &&
              (!strKeywords || strKeywords->empty()))
            if (std::optional<Value> stringified =
                    emitInheritedObjectStr(expr, receiver))
              return *stringified;
        }
        // ⭐ A CALLABLE-VALUED FIELD IS CALLED, NOT DISPATCHED. `self._f()`
        // where `_f` is declared `Callable[[], int]` calls the value the field
        // holds; there is no method `_f` on the class and none inherited
        // either, so the refusal below claimed "'Holder' inherits
        // builtins.object._f" -- naming a member of object that does not
        // exist. That predicate answers for ANY name once the class
        // linearizes onto object, so a field name reaches it looking like a
        // missing dunder.
        //
        // ⛔ Why NOT leave it to the generic method path further down: that
        // path infers a METHOD call, which passes the receiver as the first
        // argument, and the field's callable does not take one. Binding
        // through a local (`g = self._f; g()`) already resolved the value,
        // which is what says the callable is fine and only the syntax was
        // routed wrongly.
        std::optional<mlir::Type> calleeFieldType =
            lookupClassField(receiver.type, *methodName);
        if (calleeFieldType) {
          mlir::Type widenedField = types.widenLiteral(*calleeFieldType);
          // A type object in a field is called to CONSTRUCT, so it takes the
          // same route: the callee is the value, not a method of the receiver.
          if (mlir::isa<py::CallableType, py::TypeType>(widenedField)) {
            if (std::optional<Value> fieldValue =
                    emitValueAttribute(*calleeNode, receiver, *methodName))
              return emitCallableDispatch(
                  expr, *fieldValue,
                  emitCallOperands(
                      expr, {}, /*includeAstArguments=*/true,
                      mlir::dyn_cast<py::CallableType>(widenedField)));
          }
        }

        // ⭐ A CLASS DECLARED FURTHER DOWN THE MODULE has no method table yet:
        // its ClassDef has not been emitted, so `lookupClassMethod` finds
        // nothing and the generic path reports "static type !py.contract<"B">
        // does not provide manifest method 'v'" -- which is false, B declares
        // it. Reached through a function body above the class, either by an
        // `isinstance` guard that narrowed to it or by the dispatch synthesis.
        // Named here rather than left to that wording, because the fix is to
        // move the class, and nothing in the other message says so.
        if (auto contract = mlir::dyn_cast_if_present<py::ContractType>(
                types.widenLiteral(receiver.type))) {
          llvm::StringRef className = contract.getContractName();
          auto declares = declaredClassMethods.find(className);
          if (moduleClassNames.contains(className) &&
              !classMethodBindings.count(className) &&
              declares != declaredClassMethods.end() &&
              declares->second.contains(*methodName)) {
            diagnostics.push_back(parser::Diagnostic{
                parser::Severity::Error, expr.range.start,
                "'" + className.str() + "." + std::string(*methodName) +
                    "' is used before '" + className.str() +
                    "' is defined; a method of a class declared later in the "
                    "module cannot be resolved here, so move the class above "
                    "this use"});
            return emitNone(expr);
          }
        }

        // A source class inherits ALL of builtins.object's declared methods
        // through its protocol-table base, but only the defaults above have
        // something behind them. The rest are refused here, located and naming
        // the class, rather than left to report "runtime manifest has no
        // C.__setattr__ method" from the lowering — which is the same
        // points-away-from-the-defect wording the contract audit was written
        // about.
        // ⛔ Never for a name the class declares as a FIELD. That predicate
        // answers for ANY name once the class linearizes onto object, so a
        // field whose value this path could not call reached it and was
        // reported as "'Box' inherits builtins.object.t" -- a member of object
        // that does not exist. Whatever is wrong with calling the field, the
        // message for it belongs to the paths below, which know what the field
        // holds.
        if (!calleeFieldType &&
            inheritsObjectDefaultDunder(receiver.type, *methodName) &&
            !isImplementedObjectDefault(*methodName)) {
          auto contract =
              mlir::cast<py::ContractType>(types.widenLiteral(receiver.type));
          diagnostics.push_back(parser::Diagnostic{
              parser::Severity::Error, expr.range.start,
              "'" +
                  py::contracts::displayClassNameForContract(
                      contract.getContractName()) +
                  "' inherits builtins.object." + std::string(*methodName) +
                  ", which Lython does not implement; the inherited object "
                  "defaults are __eq__, __ne__, __hash__, __bool__, __repr__ "
                  "and __str__"});
          return emitNone(expr);
        }

        // ⭐ A GENERATOR ARGUMENT TO A MANIFEST METHOD IS MATERIALIZED. Every
        // manifest method that takes an iterable consumes the whole of it --
        // `join`, `extend`, `update` -- and none of them takes a generator's
        // physical shape:
        //
        //     "-".join(g())
        //     # cannot adapt types.GeneratorType to runtime input 2 of
        //     # builtins.str.join
        //
        // `"-".join(list(g()))` and `"-".join(x for x in xs)` (the genexpr
        // fuses) both worked, so the gap was the generator OBJECT. `list(...)`
        // is surface that already compiles, and because the callee consumes
        // everything the rewrite is exact rather than a change of semantics.
        //
        // ⛔ Manifest receivers only. A source class's method may hold the
        // generator and consume it lazily, which is its own business, and
        // nothing in the manifest does.
        const parser::Node *methodCallNode = &expr;
        parser::NodePtr rewrittenCall;
        if (mlir::isa_and_nonnull<py::ContractType>(
                types.widenLiteral(receiver.type)))
          if (const auto *methodArgs = ast::nodeList(expr, "args");
              methodArgs && !methodArgs->empty()) {
            // ⭐ AND ANY OTHER ITERABLE THE METHOD WILL NOT TAKE. Same
            // rewrite, asked by TYPE rather than by name:
            //
            //     xs: list[int] = []
            //     xs.extend((1, 2))
            //     # cannot adapt builtins.tuple to runtime input 1 of
            //     # builtins.list.extend
            //
            // and the same for `range(3)`, for a str, and for every other
            // iterable that is not a list. `xs.extend(list(t))` compiles, which
            // is the whole repair: the callee consumes the argument entirely, so
            // materializing it is exact.
            //
            // ⛔ Only where the call is OTHERWISE REFUSED, and only if
            // materializing makes it resolve -- both asked on TYPES, with no
            // emission, so nothing is evaluated twice. A method that genuinely
            // wants a tuple still gets one: its inference succeeds first and this
            // never runs. That is what replaces an enumeration of the consuming
            // methods, which would have been wrong for some receiver nobody
            // tested.
            llvm::SmallVector<mlir::Type, 4> argumentTypes;
            bool haveAllTypes = true;
            for (const parser::NodePtr &argument : *methodArgs) {
              mlir::Type actual =
                  argument ? types.widenLiteral(types.inferExpr(argument.get()))
                           : mlir::Type();
              if (!actual)
                haveAllTypes = false;
              argumentTypes.push_back(actual);
            }
            // The DECLARED parameter, from the resolved callable contract. The
            // type check accepts a tuple for `extend` -- the manifest spells the
            // parameter as a list and `isAssignableTo` lets an iterable through
            // -- and the refusal comes later, from the runtime ABI ("cannot adapt
            // builtins.tuple to runtime input 1"). So "the call does not resolve"
            // is the wrong trigger; "the parameter says list and this is not one"
            // is the right one.
            llvm::SmallVector<mlir::Type, 4> declaredTypes;
            if (haveAllTypes)
              if (CallInferenceResult resolved =
                      types.inferMethodCallWithEvidence(receiver.type,
                                                        *methodName,
                                                        argumentTypes))
                if (auto callable = mlir::dyn_cast_if_present<py::CallableType>(
                        resolved.evidence.callableContract)) {
                  llvm::ArrayRef<mlir::Type> positional =
                      callable.getPositionalTypes();
                  // The manifest's receiver occupies slot 0 of the contract.
                  unsigned skip = positional.size() == methodArgs->size() + 1
                                      ? 1u
                                      : 0u;
                  for (unsigned index = 0; index < methodArgs->size(); ++index)
                    declaredTypes.push_back(index + skip < positional.size()
                                                ? positional[index + skip]
                                                : mlir::Type());
                }

            std::vector<parser::NodePtr> materialized;
            bool anyMaterialized = false;
            for (auto [index, argument] : llvm::enumerate(*methodArgs)) {
              if (argument && argument->kind != "Starred")
                if (auto actual = mlir::dyn_cast_if_present<py::ContractType>(
                        types.widenLiteral(types.inferExpr(argument.get())));
                    actual) {
                  bool generator =
                      actual.getContractName() == "types.GeneratorType";
                  // The declared parameter is the PROTOCOL `Iterable`, not a
                  // list: the manifest promises to consume any iterable and the
                  // runtime implements the list case, which is why the refusal
                  // arrives from the ABI ("cannot adapt builtins.tuple to runtime
                  // input 1") and not from the type check.
                  //
                  // ⛔ Except an argument of the RECEIVER's own contract, which
                  // is the shape the runtime does implement directly
                  // (`s.update(other_set)`, `xs.extend(other_list)`).
                  // Materializing those would take a working call and break it.
                  bool wantsList = false;
                  if (!generator && index < declaredTypes.size() &&
                      actual.getContractName() != "builtins.list")
                    if (auto declared = mlir::dyn_cast_if_present<py::ProtocolType>(
                            declaredTypes[index])) {
                      auto receiverContract =
                          mlir::dyn_cast_if_present<py::ContractType>(
                              types.widenLiteral(receiver.type));
                      wantsList =
                          declared.getProtocolName() == "Iterable" &&
                          types.iterationElementType(argument.get()) &&
                          (!receiverContract ||
                           receiverContract.getContractName() !=
                               actual.getContractName());
                    }
                  if (generator || wantsList) {
                    materialized.push_back(synth::call(
                        synth::name(std::string("list"), argument->range),
                        std::vector<parser::NodePtr>{argument},
                        argument->range));
                    anyMaterialized = true;
                    continue;
                  }
                }
              materialized.push_back(argument);
            }
            if (anyMaterialized) {
              rewrittenCall = parser::makeNode("Call", expr.range);
              for (const parser::Field &field : expr.fields) {
                if (field.name == "args") {
                  parser::addField(*rewrittenCall, "args",
                                   std::move(materialized));
                  continue;
                }
                rewrittenCall->fields.push_back(field);
              }
              methodCallNode = rewrittenCall.get();
            }
          }
        // ⭐ A DICT VIEW OUTSIDE A CONSUMER, named where the program can be
        // fixed. `ks = d.keys()` reached the manifest lookup and reported
        // "runtime manifest has no builtins.dict.keys method" -- a sentence
        // about the manifest for a program that did nothing to it, while
        // `len(d.keys())`, `sorted(d.keys())`, `list(d.keys())` and
        // `for k in d.keys()` all work, because each of those unwraps the view
        // before emitting it.
        //
        // ⛔ Refused rather than snapshotted: CPython's view TRACKS later
        // mutations of the dict, and `ks = list(d.keys())` does not. Binding a
        // list where the program asked for a view is a silent wrong answer the
        // moment anything inserts, so the message says what to write and lets
        // the author decide whether a snapshot is what they meant.
        if (mlir::isa_and_nonnull<py::ContractType>(
                types.widenLiteral(receiver.type)) &&
            mlir::cast<py::ContractType>(types.widenLiteral(receiver.type))
                    .getContractName() == "builtins.dict" &&
            (*methodName == "keys" || *methodName == "values" ||
             *methodName == "items")) {
          const auto *viewArgs = ast::nodeList(expr, "args");
          const auto *viewKeywords = ast::nodeList(expr, "keywords");
          if ((!viewArgs || viewArgs->empty()) &&
              (!viewKeywords || viewKeywords->empty())) {
            diagnostics.push_back(parser::Diagnostic{
                parser::Severity::Error, expr.range.start,
                "a dict view ('" + std::string(*methodName) +
                    "()') has no value here: it is supported where it is "
                    "CONSUMED (len, sorted, list, tuple, set, a for statement, "
                    "'in'), because those read it without keeping it. Binding it "
                    "to a name would need a live view -- write list(d." +
                    std::string(*methodName) + "()) if a snapshot is what you "
                    "mean, and note that it does not track later mutations"});
            return emitNone(expr);
          }
        }
        CallOperands operands = emitCallOperands(*methodCallNode);
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
        if (std::string mismatch = cellElementRepresentationMismatch(
                receiverNode, receiver.type, inference,
                operands.positionalTypes);
            !mismatch.empty()) {
          diagnostics.push_back(parser::Diagnostic{
              parser::Severity::Error, expr.range.start, mismatch});
          return emitNone(expr);
        }
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
          if (isStructuralMutationRebindable(receiverName, receiver.value)) {
            auto op = py::CallOp::create(
                builder, loc(expr),
                mlir::TypeRange{resultType, receiver.value.getType()},
                callProtocolFor(inference), receiver.value, posPack.value,
                namePack.value, valuePack.value);
            op->setAttr("ly.bound_method", builder.getStringAttr(*methodName));
            op->setAttr("ly.structural_mutation", builder.getUnitAttr());
            rebindStructuralMutation(expr, receiverName,
                                     Value{op.getResult(1), receiver.type});
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
      if (GenericFunctionInfo *generic = lookupGenericFunction(name))
        return emitGenericCall(expr, *calleeNode, *generic);
      if (GenericFunctionInfo *mono = lookupMonomorphicFunction(name);
          mono && mayArgumentSpecialize(expr, *mono))
        return emitArgumentSpecializedCall(expr, *calleeNode, *mono,
                                           emitExpr(calleeNode));
    }
    if (!types.lookupSymbol(name) && !types.lookupClass(name)) {
      std::string reason = importedModuleBindingReason(name);
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, calleeNode->range.start,
          reason.empty() ? "unresolved name '" + name.str() + "'" : reason});
      return emitNone(expr);
    }
  }

  // ⭐ Calling a VALUE whose static type is a type object constructs that
  // class: `cls(0)` in a classmethod (emitDescriptorReceiver binds cls to a
  // py.type of the receiver's class), and `t = int; t()`. The instance
  // contract is right there in the type, and the construction path is the one
  // the class NAME already takes -- so re-spell the callee as that name and
  // go down it. Reaching the lowering instead could only report "calling a
  // type object held in a value is not supported", which describes the
  // compiler's position rather than the program's.
  // An ATTRIBUTE callee reaches the same rewrite, with one difference that is
  // not optional: the receiver still has to be EVALUATED. `Box(Other).t(5)`
  // constructs a Box, and re-spelling the call as `Other(5)` would drop that
  // construction along with anything it did. So the receiver is emitted for
  // its effects and the class name replaces only the callee.
  if (auto typeObject =
          calleeNode ? mlir::dyn_cast_if_present<py::TypeType>(
                           types.widenLiteral(types.inferExpr(calleeNode)))
                     : py::TypeType()) {
    mlir::Type instance = typeObject.getInstanceType();
    std::string spelling;
    if (auto contract =
            mlir::dyn_cast_if_present<py::ContractType>(instance)) {
      llvm::StringRef name = contract.getContractName();
      spelling = name.rsplit('.').second.empty() ? name.str()
                                                 : name.rsplit('.').second.str();
    }
    bool isName = calleeNode->kind == "Name";
    if (!spelling.empty() &&
        (!isName || spelling != ast::nameSpelling(*calleeNode)) &&
        types.lookupClass(spelling) == instance) {
      // Anything but a bare name still has to be EVALUATED -- `pick(A)(9)`
      // calls pick -- so the callee expression is emitted for its effects and
      // only the callee spelling is replaced.
      if (!isName)
        emitExpr(calleeNode);
      parser::NodePtr named = synth::name(spelling, calleeNode->range);
      parser::NodePtr rewritten = parser::makeNode("Call", expr.range);
      parser::addField(*rewritten, "func", named);
      for (llvm::StringRef field : {"args", "keywords"})
        if (const parser::Field *original = parser::findField(expr, field))
          rewritten->fields.push_back(*original);
      return emitCall(*rewritten);
    }
  }

  // ⭐ Calling an INSTANCE calls its class's __call__. py.call resolves its
  // target against the runtime manifest, so `v(2)` over a class that defines
  // __call__ died in the lowering as "runtime manifest has no V.__call__
  // method" -- the same repair __iter__ and the unary dunders needed. The
  // inline form reads the arguments from this call node, which is what a
  // __call__ takes.
  if (calleeNode)
    if (auto calleeContract = mlir::dyn_cast_if_present<py::ContractType>(
            types.widenLiteral(types.inferExpr(calleeNode))))
      if (lookupClassMethod(calleeContract, "__call__")) {
        Value receiver = emitExpr(calleeNode);
        if (std::optional<Value> called =
                tryEmitClassDunderCall(expr, receiver, "__call__"))
          return *called;
      }

  // The callee is emitted before the operands on purpose: Python evaluates
  // the callee first, and its Callable contract is the expectation the
  // argument emission distributes (lambda parameters, empty literals).
  //
  // ⭐ Which leaves a LAMBDA callee with nothing to be expected against, and
  // an unannotated lambda has no type of its own: `(lambda v: v * 2)(5)` was
  // "lambda requires a Callable annotation because its type contains
  // unresolved Unknown". The arguments are what say the parameter types here,
  // exactly as the sequence does for `map`, so they are INFERRED (not emitted)
  // first and handed back as the expectation. Order is preserved because
  // inference emits nothing; the lambda expression itself has no effects to
  // sequence against.
  //
  // ⛔ Why only when every argument infers and there are no starred ones: a
  // partial expectation would bind some parameters and leave the rest
  // Unknown, which is the diagnostic above with fewer names in it. Falling
  // through to the unexpected emission keeps that case exactly as it was.
  Value callee;
  if (calleeNode && calleeNode->kind == "Lambda") {
    llvm::SmallVector<mlir::Type, 4> argumentTypes;
    bool complete = true;
    if (const auto *args = ast::nodeList(expr, "args"))
      for (const parser::NodePtr &argument : *args) {
        if (!argument || argument->kind == "Starred") {
          complete = false;
          break;
        }
        mlir::Type argumentType =
            types.widenLiteral(types.inferExpr(argument.get()));
        if (!argumentType) {
          complete = false;
          break;
        }
        argumentTypes.push_back(argumentType);
      }
    else
      complete = false;
    if (complete && !argumentTypes.empty()) {
      // Two steps, because `emitLambda` checks the body against the
      // expectation's RESULT: a params-only callable makes every lambda
      // "not compatible with its Callable annotation". The signature pass
      // binds the parameters and reads the body's own type back out, and that
      // is what completes the contract the emission is then checked against.
      py::CallableType parameters =
          py::CallableType::get(&context, argumentTypes, {}, {}, {}, {});
      mlir::Type resultType =
          types.functionSignature(*calleeNode, std::nullopt, parameters)
              .resultType;
      if (resultType)
        callee = emitExprExpected(
            calleeNode, py::CallableType::get(&context, argumentTypes, {}, {},
                                              {}, {resultType}));
    }
  }
  if (!callee.value)
    callee = emitExpr(calleeNode);
  return emitCallableDispatch(
      expr, callee,
      emitCallOperands(expr, {}, /*includeAstArguments=*/true,
                       mlir::dyn_cast_if_present<py::CallableType>(
                           callee.type)));
}

// `def f(x: float)` reached by `f(3)`. The declared parameter is a promise
// about what the function ACCEPTS, and CPython honours it by accepting the int
// unchanged -- so the body has to be emitted a second time against the int,
// which is monomorphization and not coercion. Returns nullopt when the call is
// not that shape, leaving the ordinary dispatch to run (and, when the call is
// simply wrong, to report it).
// ⭐ AND AN OMITTED PARAMETER STANDS FOR ITS DEFAULT. `def go(v: float = 0)`
// called as `go()` never reached the decision below, because the decision is
// made on the operands the call supplies and there are none -- so the declared
// float ABI was used and the default, carried as an int attribute, was
// materialised against it: "runtime bundle value 0 for
// '!py.contract<"builtins.float">' has type 'i64', but ABI expects
// 'memref<3xi64>'", from the lowering, over one of Python's commonest
// spellings.
//
// ⛔ Literal defaults only. The rung has to be known without emitting the
// expression, and a literal is the only default whose inferred type cannot
// disagree with what emission would produce -- the disagreement the operand
// rule below exists to avoid (`1.0 + 0.0j` infers float).
//
// ⛔ And NOT by converting the default to the declared rung: `print(go())`
// would answer 0.0 where CPython answers 0, the same measurement that rejected
// converting at the argument boundary.
bool ModuleEmitter::specializationArgumentNodes(
    const parser::Node &expr, const GenericFunctionInfo &info,
    llvm::SmallVectorImpl<const parser::Node *> &out) const {
  const auto *keywords = ast::nodeList(expr, "keywords");
  if (keywords && !keywords->empty())
    return false;
  const auto *args = ast::nodeList(expr, "args");
  llvm::ArrayRef<mlir::Type> declared = info.signature.positionalTypes;
  std::size_t supplied = args ? args->size() : 0;
  if (supplied > declared.size())
    return false;
  out.clear();
  if (args)
    for (const parser::NodePtr &arg : *args) {
      if (!arg || arg->kind == "Starred")
        return false;
      out.push_back(arg.get());
    }
  if (supplied == declared.size())
    return true;
  if (!info.node)
    return false;
  const parser::Node *arguments = ast::node(*info.node, "args");
  const auto *defaults =
      arguments ? ast::nodeList(*arguments, "defaults") : nullptr;
  if (!defaults || defaults->empty())
    return false;
  std::size_t firstDefault = declared.size() - defaults->size();
  for (std::size_t index = supplied; index < declared.size(); ++index) {
    if (index < firstDefault)
      return false;
    const parser::Node *value = (*defaults)[index - firstDefault].get();
    if (!value || value->kind != "Constant")
      return false;
    if (numericTowerRung(types, types.widenLiteral(types.inferExpr(value))) < 0)
      return false;
    out.push_back(value);
  }
  return true;
}

bool ModuleEmitter::mayArgumentSpecialize(const parser::Node &expr,
                                          const GenericFunctionInfo &info) {
  llvm::SmallVector<const parser::Node *, 4> nodes;
  if (!specializationArgumentNodes(expr, info, nodes))
    return false;
  llvm::ArrayRef<mlir::Type> declared = info.signature.positionalTypes;
  if (nodes.size() != declared.size())
    return false;
  for (auto [index, node] : llvm::enumerate(nodes)) {
    mlir::Type supplied = types.widenLiteral(types.inferExpr(node));
    int suppliedRung = numericTowerRung(types, supplied);
    int declaredRung = numericTowerRung(types, declared[index]);
    if (suppliedRung >= 0 && declaredRung >= 0 && suppliedRung < declaredRung)
      return true;
  }
  return false;
}

Value ModuleEmitter::emitArgumentSpecializedCall(const parser::Node &expr,
                                                 const parser::Node &calleeNode,
                                                 GenericFunctionInfo &info,
                                                 Value declaredCallee) {
  // ⭐ The decision is made on the types the operands ACTUALLY came out as,
  // not on what inference predicted, and the two do disagree: `types.inferExpr`
  // answers `builtins.float` for `1.0 + 0.0j`, so a pre-inference decision
  // specialized `def rotate(z: complex, n: int)` at float and then emitted a
  // body that could not compile (golden scalar_loop_carried_mutate). Deciding
  // after emission also means the fallback costs nothing: the operands are
  // already the ones the ordinary dispatch wanted.
  auto declaredCallable =
      mlir::dyn_cast_if_present<py::CallableType>(declaredCallee.type);
  CallOperands operands = emitCallOperands(expr, {},
                                           /*includeAstArguments=*/true,
                                           declaredCallable);
  auto ordinary = [&] {
    return emitCallableDispatch(expr, declaredCallee, operands);
  };
  if (!operands.valid)
    return ordinary();
  llvm::ArrayRef<mlir::Type> declared = info.signature.positionalTypes;
  // The positions the call did not supply stand for their literal defaults,
  // whose inferred type is exact.
  llvm::SmallVector<mlir::Type, 4> suppliedTypes(
      operands.positionalTypes.begin(), operands.positionalTypes.end());
  if (suppliedTypes.size() < declared.size()) {
    llvm::SmallVector<const parser::Node *, 4> nodes;
    if (!specializationArgumentNodes(expr, info, nodes) ||
        nodes.size() != declared.size())
      return ordinary();
    for (std::size_t index = suppliedTypes.size(); index < nodes.size();
         ++index)
      suppliedTypes.push_back(types.widenLiteral(types.inferExpr(nodes[index])));
  }
  if (suppliedTypes.size() != declared.size())
    return ordinary();
  llvm::SmallVector<mlir::Type, 4> actual;
  bool anyLowerRung = false;
  for (auto [index, supplied] : llvm::enumerate(suppliedTypes)) {
    mlir::Type widened = types.widenLiteral(supplied);
    int suppliedRung = numericTowerRung(types, widened);
    int declaredRung = numericTowerRung(types, declared[index]);
    // Only a tower rung is re-read. Every other proper subtype already
    // reaches the declared body correctly (a subclass instance travels the
    // object ABI, `take(Dog(4))` against `def take(a: Animal)` runs today),
    // and specializing those would emit a second body for no difference in
    // behaviour.
    if (suppliedRung >= 0 && declaredRung >= 0 && suppliedRung < declaredRung) {
      actual.push_back(widened);
      anyLowerRung = true;
      continue;
    }
    if (widened != declared[index])
      return ordinary();
    actual.push_back(declared[index]);
  }
  if (!anyLowerRung)
    return ordinary();

  py::CallableType target = py::CallableType::get(
      builder.getContext(), actual, {}, {}, {},
      llvm::ArrayRef<mlir::Type>{info.signature.resultType});
  FunctionSignature specialized = types.functionSignature(
      *info.node, /*selfName=*/std::nullopt, target, /*selfType=*/{},
      /*monomorphize=*/true);
  if (specialized.positionalTypes.size() != actual.size())
    return ordinary();
  for (auto [index, type] : llvm::enumerate(specialized.positionalTypes))
    if (type != actual[index])
      return ordinary();
  // ⭐ A signature the emit boundary would reject means the specialization is
  // not available, NOT that the program is wrong -- so bail silently and let
  // the ordinary dispatch report the call site. Emitting it instead would
  // attribute the failure to the DECLARATION, which the reader did not write
  // wrong: it is the specialized reading of it that does not type.
  //
  // The failure that actually occurs here is a chain. `def outer(x: float):
  // return r(x)` specialized at int has to type `r(x)` with an int, and
  // inference answers on `r`'s DECLARED signature, which is the refusal this
  // whole path exists to lift. Covering it means teaching inference to
  // monomorphize too -- a hook from `inferCallWithEvidence` back to this
  // registry, keyed on the declared callable, with a cycle guard -- and the
  // registry is name-keyed today, so two functions of identical signature
  // would have to be detected and dropped. Scoped, not built; recorded in
  // tests/probe/wb_argument_boundary_numeric_tower.py.
  if (!specialized.bodyInferenceFailures.empty() ||
      !specialized.missingParameterAnnotations.empty() ||
      !specialized.invalidParameterAnnotations.empty() ||
      !specialized.generatorAnalysisFailures.empty() ||
      !specialized.generatorAnnotationMismatch.empty())
    return ordinary();
  // ⭐ A result the annotation was WIDENING must not be specialized, and this
  // is the guard that keeps the whole path from turning a refusal into a wrong
  // answer. Re-reading `def pick(n: int) -> int` at bool gives its two
  // branches `builtins.bool` and the literal 5, whose join is a union -- and
  // the annotation is exactly what used to collapse that to int.
  //
  // ⛔ Why NOT collapse it here instead, along the tower, which is what the
  // annotation did: because CPython does not. `def pick(n: int) -> int` with
  // `return n` reached by `pick(False)` prints False, and an int-collapsed
  // result prints 0. The union is the honest type and the py ABI cannot carry
  // one as a return value, so the call goes back to being refused -- which is
  // what it already was, and a refusal is the outcome this project prefers to
  // a plausible number.
  if (mlir::isa_and_present<py::UnionType>(specialized.resultType) &&
      !mlir::isa_and_present<py::UnionType>(info.signature.resultType))
    return ordinary();

  auto memoized = info.specializations.find(specialized.publicCallable);
  std::string symbol;
  if (memoized != info.specializations.end()) {
    symbol = memoized->second;
  } else {
    // The same divergence backstop the generic specializer carries: a body
    // that calls itself one rung down would otherwise re-enter forever.
    if (info.specializations.size() >= 32)
      return ordinary();
    symbol = (llvm::Twine(info.symbolBase) + "$arg" +
              llvm::Twine(static_cast<unsigned>(info.specializations.size())))
                 .str();
    // Memoize BEFORE the body, so a recursive call at the same ground
    // signature resolves to this symbol instead of specializing again.
    info.specializations[specialized.publicCallable] = symbol;
    auto emitBody = [&] {
      emitCallableFunction(*info.node, symbol, specialized, {},
                           /*isLambda=*/false);
    };
    if (info.source)
      emitInDefiningModuleScope(*info.source, emitBody);
    else
      emitBody();
  }

  Value callee =
      emitBindingRef(calleeNode, symbol, specialized.publicCallable);
  return emitCallableDispatch(expr, callee, operands);
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

// ⭐ `hasattr(x, "v")` AND `callable(f)` ARE COMPILE-TIME QUESTIONS HERE. Both
// names were simply unbound, and both are decided by the same table the rest of
// the emitter dispatches through -- the attribute either exists on the static
// class or it does not, and a value either has a callable contract or it does
// not.
//
// ⛔ A SUBCLASS CAN ONLY ADD, which is what makes the two answers asymmetric: a
// True stands (the base has it, so every instance does), and a False is refused
// when the class has a subclass, because the subclass may define exactly that
// attribute. Answering False there is the silent wrong answer this project
// exists to avoid; refusing says which class to look at.
// The attribute name a `"..."` literal argument spells, or nothing when the
// argument is not one. The quotes are part of a LiteralType's spelling, so a
// name is only there when both are.
std::optional<llvm::StringRef>
ModuleEmitter::literalStringArgument(const parser::Node *node) {
  auto literal =
      mlir::dyn_cast_if_present<py::LiteralType>(types.inferExpr(node));
  if (!literal)
    return std::nullopt;
  llvm::StringRef spelling = literal.getSpelling();
  if (spelling.size() < 2 || spelling.front() != '"' || spelling.back() != '"')
    return std::nullopt;
  return spelling.drop_front().drop_back();
}

// ⭐ THE ONE GATE EVERY BUILTIN INTERCEPTION OPENS WITH. Folding `len(x)` or
// `getattr(x, "v")` is only legal while the name still means the builtin, and
// ANY binding the program makes for that spelling shadows it -- a local, a
// parameter, a top-level `def next`. Gating on locals alone once made the
// winner depend on the argument count. Nineteen sites spelled this out.
bool ModuleEmitter::callsUnshadowedBuiltin(const parser::Node *calleeNode,
                                           llvm::StringRef name) const {
  return calleeNode && calleeNode->kind == "Name" &&
         llvm::StringRef(ast::nameSpelling(*calleeNode)) == name &&
         !programBindsName(name);
}

std::optional<Value>
ModuleEmitter::tryEmitHasattrCall(const parser::Node &expr,
                                  const parser::Node *calleeNode) {
  if (!callsUnshadowedBuiltin(calleeNode, "hasattr"))
    return std::nullopt;
  const auto *args = ast::nodeList(expr, "args");
  const auto *keywords = ast::nodeList(expr, "keywords");
  auto refuse = [&](std::string reason) -> std::optional<Value> {
    diagnostics.push_back(parser::Diagnostic{parser::Severity::Error,
                                             expr.range.start,
                                             std::move(reason)});
    return emitNone(expr);
  };
  if (!args || args->size() != 2 || (keywords && !keywords->empty()) ||
      !args->front() || !(*args)[1])
    return refuse("hasattr() takes exactly two arguments");
  std::optional<llvm::StringRef> name =
      literalStringArgument((*args)[1].get());
  if (!name)
    return refuse("hasattr() needs a literal attribute name: the answer is "
                  "decided at compile time here");
  llvm::StringRef attribute = *name;
  mlir::Type subject = types.widenLiteral(types.inferExpr(args->front().get()));
  auto contract = mlir::dyn_cast_if_present<py::ContractType>(subject);
  if (!contract)
    return refuse("hasattr() needs a statically resolved receiver, and " +
                  typeText(subject) + " is not one");
  bool present = lookupClassField(subject, attribute).has_value() ||
                 lookupClassMethod(subject, attribute).has_value() ||
                 lookupClassStaticAttr(subject, attribute).has_value();
  if (!present) {
    const py::protocols::Table &table = py::protocols::Table::get(context);
    present = !table.methodContractCandidatesWithEvidence(subject, attribute)
                   .empty() ||
              table.resolveFieldContractWithEvidence(subject, attribute)
                  .has_value();
  }
  if (!present)
    for (const auto &entry : classMros)
      if (entry.getKey() != contract.getContractName() &&
          llvm::is_contained(entry.second, contract.getContractName()))
        return refuse("hasattr(x, '" + attribute.str() + "') on '" +
                      contract.getContractName().str() +
                      "' cannot be answered: '" + entry.getKey().str() +
                      "' derives from it and may define that attribute");
  (void)emitExpr(args->front().get());
  mlir::Type literalType = types.literal(present ? "True" : "False");
  auto constant = py::BoolConstantOp::create(builder, loc(expr), literalType,
                                             builder.getBoolAttr(present));
  return Value{constant.getResult(), literalType};
}

// `setattr(x, "v", value)` with a literal name IS `x.v = value` -- the same
// store written as a call -- so it becomes one, and the field's declared type,
// the ownership traffic and the refusals all come from the assignment path
// unchanged. The call's own value is None, which is what CPython returns.

// ⭐ `p._replace(x=5)` IS THE CONSTRUCTION IT DESCRIBES: every field the call
// does not name comes from p, and the class, the field order and the field
// names are all known here. CPython's namedtuple builds it out of _make and
// _fields; there is nothing dynamic in it, so the rewrite IS the
// implementation. It was "'P' inherits builtins.object._replace, which Lython
// does not implement".
//
// ⛔ The receiver is bound to a temporary first. `make()._replace(x=1)` would
// otherwise call make() once per field the replacement does not name, which is
// a side effect CPython does not have.
std::optional<Value>
ModuleEmitter::tryEmitNamedTupleReplace(const parser::Node &expr,
                                        const parser::Node *calleeNode) {
  if (!calleeNode || calleeNode->kind != "Attribute")
    return std::nullopt;
  std::optional<std::string_view> attr = ast::string(*calleeNode, "attr");
  if (!attr || *attr != "_replace")
    return std::nullopt;
  const parser::Node *receiverNode = ast::node(*calleeNode, "value");
  if (!receiverNode)
    return std::nullopt;
  auto contract = mlir::dyn_cast_if_present<py::ContractType>(
      types.widenLiteral(types.inferExpr(receiverNode)));
  if (!contract || !namedTupleContracts.count(contract.getContractName()))
    return std::nullopt;
  llvm::StringRef className = contract.getContractName();
  // A shadowed class name would build the wrong thing; the rewrite spells the
  // class by name, so it has to be the name that still means this class.
  std::optional<mlir::Type> bound = types.lookupClass(className);
  if (!bound || *bound != mlir::Type(contract))
    return std::nullopt;

  parser::SourceRange range = expr.range;
  auto refuse = [&](const std::string &message) -> std::optional<Value> {
    diagnostics.push_back(
        parser::Diagnostic{parser::Severity::Error, range.start, message});
    return emitNone(expr);
  };
  const auto *args = ast::nodeList(expr, "args");
  const auto *keywords = ast::nodeList(expr, "keywords");
  if (args && !args->empty())
    return refuse("_replace() takes keyword arguments only");
  llvm::ArrayRef<std::string> order = classFieldOrders[className];
  if (order.empty())
    return refuse("_replace() needs a NamedTuple with fields");
  llvm::StringMap<parser::NodePtr> replacements;
  if (keywords)
    for (const parser::NodePtr &entry : *keywords) {
      if (!entry)
        continue;
      std::optional<std::string_view> keyword = ast::string(*entry, "arg");
      const parser::Field *value = parser::findField(*entry, "value");
      if (!keyword || !value ||
          !std::holds_alternative<parser::NodePtr>(value->value))
        return refuse("_replace() takes keyword arguments only");
      if (!llvm::is_contained(order, std::string(*keyword)))
        return refuse("_replace() got an unexpected field name '" +
                      std::string(*keyword) + "'");
      replacements[*keyword] = std::get<parser::NodePtr>(value->value);
    }

  std::string tmp = "__replace" + std::to_string(++syntheticFunctionCounter);
  Value receiver = emitExpr(receiverNode);
  values[tmp] = receiver;
  types.bindSymbol(tmp, receiver.type);
  std::vector<parser::NodePtr> arguments;
  for (const std::string &field : order) {
    auto found = replacements.find(field);
    arguments.push_back(found != replacements.end()
                            ? found->second
                            : synth::attribute(synth::name(tmp, range), field,
                                               range));
  }
  parser::NodePtr construction = synth::call(
      synth::name(className, range), std::move(arguments), range);
  synthesizedIteratorDefs.push_back(construction);
  Value built = emitExpr(construction.get());
  values.erase(tmp);
  return built;
}


// The class whose instances CPython makes unhashable: one that defines __eq__
// and inherits object's __hash__, which is every unfrozen dataclass. Answering
// object's identity hash instead would place two instances the class calls
// EQUAL in different buckets, and every hash container built on them would miss
// without a word -- which is what `{K(1): 2}` did before this was asked at the
// key as well as at `hash()`.
//
// ⛔ Enum members are exempt: CPython's Enum defines no __eq__ of its own, so
// members stay hashable; the __eq__ Lython synthesizes for an enum is a
// lowering artifact rather than the author's value equality.
std::optional<std::string>
ModuleEmitter::unhashableClassName(mlir::Type type) const {
  auto contract =
      mlir::dyn_cast_if_present<py::ContractType>(types.widenLiteral(type));
  if (!contract || enumClasses.count(contract.getContractName()))
    return std::nullopt;
  if (!inheritsObjectDefaultDunder(type, "__hash__") ||
      !lookupClassMethod(type, "__eq__"))
    return std::nullopt;
  return py::contracts::displayClassNameForContract(contract.getContractName());
}

// ⭐ ASKED AT THE KEY, not only at `hash()`. The two are the same question and
// CPython raises for both -- "cannot use 'K' as a dict key (unhashable type:
// 'K')" -- but only `hash()` was asking, so a dict literal accepted the key,
// stored it under an identity hash, and then MISSED on an equal one:
//
//     @dataclass
//     class K:
//         a: int
//     d = {K(1): 2}
//     print(d[K(1)])       # KeyError: K(a=1)
bool ModuleEmitter::refuseUnhashableKey(const parser::Node &site,
                                        mlir::Type type,
                                        llvm::StringRef role) {
  std::optional<std::string> unhashable = unhashableClassName(type);
  if (!unhashable)
    return false;
  diagnostics.push_back(parser::Diagnostic{
      parser::Severity::Error, site.range.start,
      "cannot use '" + *unhashable + "' as a " + role.str() +
          " (unhashable type: '" + *unhashable +
          "'): it defines __eq__ without __hash__, so CPython sets __hash__ to "
          "None for it, as it does for every unfrozen dataclass"});
  return true;
}

std::optional<Value>
ModuleEmitter::tryEmitSetattrCall(const parser::Node &expr,
                                  const parser::Node *calleeNode) {
  if (!callsUnshadowedBuiltin(calleeNode, "setattr"))
    return std::nullopt;
  const auto *args = ast::nodeList(expr, "args");
  const auto *keywords = ast::nodeList(expr, "keywords");
  auto refuse = [&](std::string reason) -> std::optional<Value> {
    diagnostics.push_back(parser::Diagnostic{parser::Severity::Error,
                                             expr.range.start,
                                             std::move(reason)});
    return emitNone(expr);
  };
  if (!args || args->size() != 3 || (keywords && !keywords->empty()) ||
      !args->front() || !(*args)[1] || !(*args)[2])
    return refuse("setattr() takes exactly three arguments");
  std::optional<llvm::StringRef> name =
      literalStringArgument((*args)[1].get());
  if (!name)
    return refuse("setattr() needs a literal attribute name: the store is "
                  "resolved at compile time here");
  parser::NodePtr target =
      synth::attribute(args->front(), name->str(), expr.range);
  synthesizedIteratorDefs.push_back(target);
  emitAssignTarget(*target, emitExpr((*args)[2].get()));
  return emitNone(expr);
}

// `getattr(x, "v")` with a literal name IS `x.v` -- the same lookup written as a
// call -- so it is rewritten to the attribute and every rule about attributes
// applies unchanged. A computed name has no static answer here and says so; the
// three-argument form would need the hasattr fold to pick an arm and is not
// built.
std::optional<Value>
ModuleEmitter::tryEmitGetattrCall(const parser::Node &expr,
                                  const parser::Node *calleeNode) {
  if (!callsUnshadowedBuiltin(calleeNode, "getattr"))
    return std::nullopt;
  const auto *args = ast::nodeList(expr, "args");
  const auto *keywords = ast::nodeList(expr, "keywords");
  auto refuse = [&](std::string reason) -> std::optional<Value> {
    diagnostics.push_back(parser::Diagnostic{parser::Severity::Error,
                                             expr.range.start,
                                             std::move(reason)});
    return emitNone(expr);
  };
  if (!args || args->size() != 2 || (keywords && !keywords->empty()) ||
      !args->front() || !(*args)[1])
    return refuse("getattr() takes exactly two arguments here: the default "
                  "form would need a runtime attribute lookup");
  std::optional<llvm::StringRef> name =
      literalStringArgument((*args)[1].get());
  if (!name)
    return refuse("getattr() needs a literal attribute name: the lookup is "
                  "resolved at compile time here");
  parser::NodePtr attribute =
      synth::attribute(args->front(), name->str(), expr.range);
  synthesizedIteratorDefs.push_back(attribute);
  return emitExpr(attribute.get());
}

std::optional<Value>
ModuleEmitter::tryEmitCallableCall(const parser::Node &expr,
                                   const parser::Node *calleeNode) {
  if (!callsUnshadowedBuiltin(calleeNode, "callable"))
    return std::nullopt;
  const auto *args = ast::nodeList(expr, "args");
  const auto *keywords = ast::nodeList(expr, "keywords");
  if (!args || args->size() != 1 || (keywords && !keywords->empty()) ||
      !args->front() || args->front()->kind == "Starred") {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, expr.range.start,
        "callable() takes exactly one argument"});
    return emitNone(expr);
  }
  mlir::Type subject = types.widenLiteral(types.inferExpr(args->front().get()));
  bool answer = mlir::isa<py::CallableType>(subject) ||
                mlir::isa<py::TypeType>(subject) ||
                lookupClassMethod(subject, "__call__").has_value();
  if (!answer) {
    auto contract = mlir::dyn_cast_if_present<py::ContractType>(subject);
    if (!contract) {
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, expr.range.start,
          "callable() needs a statically resolved value, and " +
              typeText(subject) + " is not one"});
      return emitNone(expr);
    }
    for (const auto &entry : classMros)
      if (entry.getKey() != contract.getContractName() &&
          llvm::is_contained(entry.second, contract.getContractName())) {
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, expr.range.start,
            "callable(x) on '" + contract.getContractName().str() +
                "' cannot be answered: '" + entry.getKey().str() +
                "' derives from it and may define __call__"});
        return emitNone(expr);
      }
  }
  (void)emitExpr(args->front().get());
  mlir::Type literalType = types.literal(answer ? "True" : "False");
  auto constant = py::BoolConstantOp::create(builder, loc(expr), literalType,
                                             builder.getBoolAttr(answer));
  return Value{constant.getResult(), literalType};
}

std::optional<Value>
ModuleEmitter::tryEmitTypeCall(const parser::Node &expr,
                               const parser::Node *calleeNode) {
  // ⭐ `type(x)` is the class of x, and the class of x is a STATIC fact here
  // exactly when nothing can put a subclass instance in x. That is the whole
  // condition: a manifest contract is its own runtime class (a bool is not
  // stored as an int here, it is a truth bit), and a source class is too unless
  // the program declares a subclass of it.
  //
  // Until now the name was simply unbound -- "unresolved name 'type'" -- which
  // took `type(e).__name__` with it, the idiom for reporting what was caught.
  //
  // ⛔ NOT bound as the `type` CLASS: that would make `type(x)` an
  // instantiation, and a type object built from an instance is not what CPython
  // returns. The interception happens before any class binding for the same
  // reason int() and str() are intercepted.
  if (!callsUnshadowedBuiltin(calleeNode, "type"))
    return std::nullopt;
  const auto *typeArgs = ast::nodeList(expr, "args");
  const auto *typeKeywords = ast::nodeList(expr, "keywords");
  if (!typeArgs || typeArgs->size() != 1 || !typeArgs->front() ||
      typeArgs->front()->kind == "Starred" ||
      (typeKeywords && !typeKeywords->empty()))
    return std::nullopt;
  const parser::Node *subjectNode = typeArgs->front().get();
  mlir::Type subject = types.widenLiteral(types.inferExpr(subjectNode));
  auto contract = mlir::dyn_cast_if_present<py::ContractType>(subject);
  auto refuse = [&](std::string reason) -> std::optional<Value> {
    diagnostics.push_back(parser::Diagnostic{parser::Severity::Error,
                                             expr.range.start,
                                             std::move(reason)});
    return emitNone(expr);
  };
  if (!contract)
    return refuse("type(x) needs a statically resolved class, and " +
                  typeText(subject) + " is not one");
  llvm::StringRef contractName = contract.getContractName();
  if (contractName == "builtins.object" || contractName == "typing.Any")
    return refuse(
        "type(x) on a type-erased value would need the runtime class, which is "
        "excluded from the static evidence kernel");
  // ⛔ The SUBCLASS check is the soundness of the whole fold. `x: A = B()`
  // makes the static class A and the runtime class B, and answering A there is
  // a wrong answer with no diagnostic -- which is what the default repr does
  // today and is recorded separately.
  for (const auto &entry : classMros) {
    if (entry.getKey() == contractName)
      continue;
    if (llvm::is_contained(entry.second, contractName))
      return refuse("type(x) is not supported for '" + contractName.str() +
                    "': '" + entry.getKey().str() +
                    "' derives from it, so a value of this type can hold an "
                    "instance of the subclass and the answer would name the "
                    "static class instead");
  }
  // The argument still runs: `type(f())` calls f.
  (void)emitExpr(subjectNode);
  mlir::Type typeType = types.typeObject(contract);
  auto object = py::TypeObjectOp::create(builder, loc(expr), typeType, contract);
  return Value{object.getResult(), typeType};
}

std::optional<Value>
ModuleEmitter::tryEmitIsInstanceCall(const parser::Node &expr,
                                     const parser::Node *calleeNode) {
  if (!callsUnshadowedBuiltin(calleeNode, "isinstance"))
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

  std::optional<llvm::SmallVector<mlir::Type, 4>> targets =
      isinstanceTargetTypes((*args)[1].get(), types);
  if (!targets) {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, expr.range.start,
        "second argument to isinstance must be a statically resolved class "
        "type, or a tuple of them, got " +
            typeText(types.inferExpr((*args)[1].get()))});
    return emitNone(expr);
  }

  Value input = emitExpr(args->front().get());
  IsInstanceAnalysis analysis =
      analyzeIsInstanceAny(input.type, *targets, types, module);
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
    bit = constantBool(builder, loc(expr), true);
  } else if (analysis.kind == IsInstanceAnalysis::Kind::AlwaysFalse) {
    bit = constantBool(builder, loc(expr), false);
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
      bit = constantBool(builder, loc(expr), false);
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
    for (mlir::Type subclass : analysis.classTestTypes) {
      auto also = py::ClassTestOp::create(builder, loc(expr),
                                          builder.getI1Type(), input.value,
                                          mlir::TypeAttr::get(subclass));
      bit = mlir::arith::OrIOp::create(builder, loc(expr), bit,
                                       also.getResult())
                .getResult();
    }
  }
  return boxedBool(builder, loc(expr), types, bit);
}

// ⭐ `int(s, base)` as a SYNTHESIZED PYTHON FUNCTION, not a new native. The
// whole parse -- strip, sign, the 0x/0o/0b prefix, underscores, digit value
// by position in "0123456789abc...z", and the multiply-accumulate that grows
// into a bigint on its own -- is ordinary Python over surface that already
// compiles. A native would have duplicated LyLong_FromStr's limb arithmetic
// for the one thing it does not do, and the emitter is where CPython's own
// int() dispatch decides between the two anyway.
//
// ⛔ base=0 (auto-detect from the prefix) is NOT accepted: it raises here
// rather than guessing, because "0" is also a valid base-10 literal and the
// CPython rule for it (prefix decides, bare leading zeros are an error) is a
// different parse, not a parameter of this one.
// ⭐ THE CALL THE STATIC RECEIVER TYPE CANNOT ANSWER, ANSWERED. When a
// subclass overrides the method, the receiver's static class does not say which
// body runs, and this project has no vtable to fall back on -- so the call was
// refused ("'name' is overridden by a subclass of 'A', ..."), which took every
// base-typed collection, parameter and declared binding with it.
//
// The dispatch is a synthesized module FUNCTION, one per (class, method):
//
//     def __lyvdisp$1(__ly_recv: A, p: int) -> str:
//         if isinstance(__ly_recv, B):
//             return __ly_recv.name(p)     # narrowed: resolves to B's body
//         return A.name(__ly_recv, p)      # the base's body, by class
//
// Every piece of it already existed and is exercised by ordinary programs:
// `isinstance` on a source class is a runtime class test, an `isinstance` guard
// NARROWS the receiver so the call inside resolves statically, and an unbound
// `A.name(recv)` names the base's body without asking the receiver's type. The
// same three lines written by hand compile and print CPython's answers, which
// is what made this a synthesis rather than a new lowering.
//
// ⛔ Per (class, method), not per call site: a body reached through the
// dispatcher may dispatch on the same method again (`self.area()` where self is
// base-typed), and a per-site expansion would not terminate. The memo entry is
// written BEFORE the body is emitted for exactly that reason -- the same order
// the generic specializer uses.
//
// ⛔ Only for a shape whose signature the dispatcher can restate: plain
// positional parameters, all annotated, an annotated return, and a base that
// declares the method. A generator, an async method, a property, a
// classmethod/staticmethod, defaults, *args/**kwargs or keyword arguments at
// the call site fall through to the refusal, which is what shipped for all of
// them before. A dispatcher may not GUESS a signature it will then call
// through: getting that wrong is a silent wrong body, and the refusal is not.
namespace {
// The shared_ptr behind a node field, so a synthesized def can reuse an
// annotation the source already wrote instead of re-parsing its spelling.
parser::NodePtr sharedField(const parser::Node &node, llvm::StringRef field) {
  const parser::Field *found = parser::findField(node, field);
  if (!found)
    return nullptr;
  if (const auto *child = std::get_if<parser::NodePtr>(&found->value))
    return *child;
  return nullptr;
}
} // namespace

const ModuleEmitter::VirtualDispatchHelper *
ModuleEmitter::virtualDispatcherFor(const parser::Node &anchor, Value receiver,
                                    llvm::StringRef methodName,
                                    unsigned argumentCount, bool asProperty,
                                    llvm::ArrayRef<std::string> keywordNames,
                                    bool asAttribute) {
  auto contract = mlir::dyn_cast_if_present<py::ContractType>(receiver.type);
  if (!contract)
    return nullptr;
  llvm::StringRef receiverClass = contract.getContractName();
  const parser::Node &expr = anchor;
  // A property and a class attribute are both READ; only a method is called.
  bool readsWithoutCall = asProperty || asAttribute;

  // ⭐ A CLASS ATTRIBUTE A SUBCLASS REDECLARES DISPATCHES TOO, and it was the
  // one redeclaration this synthesis did not cover:
  //
  //     class Shape:
  //         kind = "shape"
  //         def describe(self) -> str: return self.kind
  //     class Square(Shape):
  //         kind = "square"
  //     # 'kind' is overridden by a subclass of 'Shape', so this call cannot
  //     # be resolved from the static type of the receiver
  //
  // The gate that reports that is right -- the attribute is read from the
  // DEFINING class's cell, so a base-typed receiver got the base's value --
  // and the repair is the same one the method side already has: test the
  // runtime class, most-derived first, and read through the narrowed receiver.
  //
  // ⛔ The fallback arm names the DEFINING class rather than re-reading
  // `__ly_recv`, which is what the property arm does. A property has no
  // unbound spelling; a class attribute does (`Shape.kind`), and using it ends
  // the recursion at the base instead of relying on the suppressed gate.
  std::string fallbackClass;
  parser::NodePtr returns;
  parser::SourceRange range = expr.range;
  std::optional<MethodBinding> base;
  const parser::Node *arguments = nullptr;
  if (asAttribute) {
    if (argumentCount != 0 || !keywordNames.empty())
      return nullptr;
    // ⭐ THE ATTRIBUTE'S TYPE, FROM EITHER CHANNEL. A main-module class keeps
    // its attributes in cells and an IMPORTED one keeps them on the constant
    // channel, and asking only the cells left every imported hierarchy
    // refused -- `self.kind` inside an imported base method, which is where a
    // library puts it.
    std::optional<std::pair<llvm::StringRef, mlir::Type>> slot =
        resolveClassAttrSlot(receiverClass, methodName);
    llvm::StringRef definingClass = receiverClass;
    mlir::Type attributeType;
    if (slot) {
      definingClass = slot->first;
      attributeType = slot->second;
    } else if (std::optional<mlir::Type> constant =
                   lookupClassStaticAttr(receiver.type, methodName)) {
      attributeType = *constant;
    }
    if (!attributeType)
      return nullptr;
    // ⛔ Only a type the dispatcher can WRITE as its return annotation, which
    // is the same restriction every synthesized def here lives under. A
    // container-typed class attribute keeps the refusal rather than a guess.
    llvm::StringRef spelling;
    if (auto attrContract =
            mlir::dyn_cast_if_present<py::ContractType>(
                types.widenLiteral(attributeType)))
      if (attrContract.getArguments().empty()) {
        spelling = attrContract.getContractName();
        if (spelling.consume_front("builtins.") && spelling.contains('.'))
          spelling = {};
      }
    if (spelling.empty())
      return nullptr;
    fallbackClass = definingClass.str();
    returns = synth::name(spelling, range);
  }

  // The body the STATIC type resolves to: its signature is the one the
  // dispatcher restates, and its declaring class is how the fallback names it.
  if (!asAttribute) {
  base = lookupClassMethod(receiver.type, methodName);
  if (!base || !base->method || base->definingClass.empty())
    return nullptr;
  if (base->kind != (asProperty ? "property" : "instance") || base->async ||
      base->bodySignature.isGeneratorFunction ||
      base->bodySignature.isAsyncGeneratorFunction)
    return nullptr;
  if (types.lookupClass(base->definingClass) == mlir::Type())
    return nullptr;

  fallbackClass = base->definingClass;
  arguments = ast::node(*base->method, "args");
  if (!arguments)
    return nullptr;
  if (ast::node(*arguments, "vararg") || ast::node(*arguments, "kwarg"))
    return nullptr;
  // ⭐ A DEFAULT IS NOT A REASON TO REFUSE, once the dispatcher restates only
  // the parameters the CALL passed. `def f(self, a: int = 1)` overridden by a
  // subclass was refused outright -- "'f' is overridden by a subclass of
  // 'Base'" -- for a method shape ordinary Python writes everywhere.
  //
  // ⛔ A KEYWORD-ONLY PARAMETER IS NOT A REASON TO REFUSE EITHER, for the same
  // reason: the arms restate only what the call passed, and a `*`-parameter is
  // passed by name or not at all. The blanket `kwonlyargs` refusal that stood
  // here rejected `x.f(3)` -- a call naming NOTHING keyword-only -- whenever
  // the base merely declared one.
  //
  // ⛔ The arms must not restate the default, only omit the parameter: each
  // arm calls `recv.f(p1..pN)` with exactly what the site passed, so the body
  // that RUNS fills the rest from its OWN default -- which is what CPython
  // does and what a dispatcher restating the base's default would get wrong
  // for a subclass that changed it. That is why the memo is keyed by the
  // argument count as well: one dispatcher per (class, method, arity).
  returns = sharedField(*base->method, "returns");
  if (!returns)
    return nullptr;
  // ⛔ A method that hands back an ITERATOR cannot be reached through a
  // function: the frame a generator resumes into is not carried by the
  // returned value ("a generator returned out of a function cannot be
  // resumed"), so a dispatched `__iter__` would compile to a loop over a
  // generator that cannot run. `for v in b` with an overridden `__iter__`
  // keeps the refusal, which is what tests/golden/errors/
  // for_over_a_subclass_overridden_iter.py pins.
  if (auto protocol = mlir::dyn_cast_if_present<py::ProtocolType>(
          types.widenLiteral(base->signature.resultType))) {
    llvm::StringRef name = protocol.getProtocolName();
    if (name == "Iterator" || name == "Generator" || name == "AsyncIterator" ||
        name == "AsyncGenerator" || name == "Iterable")
      return nullptr;
  }
  }

  llvm::SmallVector<std::string, 4> parameterNames;
  llvm::SmallVector<synth::Param, 4> params;
  llvm::SmallVector<std::string, 2> keywordParameters;
  params.push_back(synth::Param{"__ly_recv", synth::name(receiverClass, range)});
  if (!asAttribute) {
  bool selfSeen = false;
  bool enough = false;
  for (llvm::StringRef field : {"posonlyargs", "args"}) {
    if (enough)
      break;
    if (const auto *list = ast::nodeList(*arguments, field))
      for (const parser::NodePtr &argument : *list) {
        if (!argument)
          return nullptr;
        // The rest come from the running body's own defaults.
        if (selfSeen && parameterNames.size() == argumentCount) {
          enough = true;
          break;
        }
        llvm::StringRef name = ast::nameSpelling(*argument);
        if (!selfSeen) {
          selfSeen = true;
          continue; // the receiver, restated as __ly_recv above
        }
        parser::NodePtr annotation = sharedField(*argument, "annotation");
        if (!annotation)
          return nullptr;
        parameterNames.push_back(name.str());
        params.push_back(synth::Param{name.str(), std::move(annotation)});
      }
  }
  if (!selfSeen || parameterNames.size() != argumentCount)
    return nullptr;

  // ⭐ A KEYWORD AT THE CALL SITE IS A PARAMETER OF THE DISPATCHER TOO. It used
  // to bail the whole synthesis ("the dispatcher forwards positionals only, so
  // a keyword would silently move to another parameter"), which refused
  // `x.f(k=1)` on every overridden method. The keyword rides as an ordinary
  // parameter here and is forwarded BY NAME in each arm, so the body that runs
  // binds it the way CPython does.
  //
  // ⛔ Every keyword must name a parameter the base declares AFTER the
  // positional prefix -- otherwise the call is already wrong -- and its
  // annotation is the base's. A keyword the base does not declare, or one that
  // repeats a positional, falls through to the refusal.
  for (const std::string &keywordName : keywordNames) {
    if (llvm::is_contained(parameterNames, keywordName) ||
        llvm::is_contained(keywordParameters, keywordName))
      return nullptr;
    parser::NodePtr annotation;
    unsigned position = 0;
    unsigned keywordPosition = 0;
    bool keywordOnly = false;
    for (llvm::StringRef field : {"posonlyargs", "args", "kwonlyargs"})
      if (const auto *list = ast::nodeList(*arguments, field))
        for (const parser::NodePtr &argument : *list) {
          if (!argument)
            return nullptr;
          if (field != "kwonlyargs")
            ++position;
          if (position == 1 && field != "kwonlyargs")
            continue; // the receiver
          // A `/` parameter cannot be named at a call at all, so a keyword
          // matching one is not this parameter -- CPython rejects the call.
          if (field == "posonlyargs" ||
              ast::nameSpelling(*argument) != keywordName)
            continue;
          annotation = sharedField(*argument, "annotation");
          keywordPosition = position - 1;
          keywordOnly = field == "kwonlyargs";
        }
    if (!annotation || (!keywordOnly && keywordPosition <= argumentCount))
      return nullptr;
    keywordParameters.push_back(keywordName);
    params.push_back(synth::Param{keywordName, std::move(annotation)});
  }
  }

  std::string key =
      (receiverClass + "." + methodName + "/" + llvm::Twine(argumentCount) +
       (asProperty ? "$get" : "") + (asAttribute ? "$attr" : ""))
          .str();
  for (const std::string &keywordName : keywordParameters)
    key += "," + keywordName;
  auto memo = virtualDispatchHelpers.find(key);
  if (memo == virtualDispatchHelpers.end()) {
    // Every class that declares the method and has the receiver's class among
    // its ancestors, most-derived first so a subclass of a subclass wins.
    llvm::SmallVector<std::pair<unsigned, std::string>, 4> candidates;
    for (const auto &entry : declaredClassBases) {
      llvm::StringRef candidate = entry.getKey();
      if (candidate == receiverClass)
        continue;
      // ⭐ THE SAME PREDICATE THE GATE USES. Asking only what the candidate
      // DECLARES ITSELF left `class Both(Base, Mixin)` refused by the gate
      // (Mixin declares it, so the dispatch is real) and unanswerable here (no
      // candidate declares it), which is one question with two predicates and
      // a valid program in the gap.
      const llvm::StringMap<llvm::StringSet<>> &declarations =
          asAttribute ? declaredClassAttributes : declaredClassMethods;
      if (!candidateRedeclares(declarations, receiverClass, candidate,
                               methodName))
        continue;
      unsigned depth = 0;
      llvm::SmallVector<llvm::StringRef, 8> worklist{candidate};
      llvm::StringSet<> seen;
      bool derived = false;
      while (!worklist.empty()) {
        llvm::StringRef current = worklist.pop_back_val();
        auto bases = declaredClassBases.find(current);
        if (bases == declaredClassBases.end())
          continue;
        for (const std::string &base : bases->second)
          if (seen.insert(base).second) {
            worklist.push_back(base);
            derived = derived || base == receiverClass;
          }
      }
      if (!derived)
        continue;
      depth = seen.size();
      if (types.lookupClass(candidate) == mlir::Type())
        return nullptr;
      candidates.push_back({depth, candidate.str()});
    }
    if (candidates.empty())
      return nullptr;
    llvm::sort(candidates, [](const auto &lhs, const auto &rhs) {
      if (lhs.first != rhs.first)
        return lhs.first > rhs.first;
      return lhs.second < rhs.second;
    });

    // ⛔ EVERY ARM HAS TO BE ABLE TO READ THE ATTRIBUTE, and a candidate in a
    // module emitted LATER cannot yet: `pets.Bird` is a known subclass to the
    // declaration pre-pass while its own attributes are registered only when
    // its module is emitted, and base.py is emitted first because pets.py
    // imports it. Building the dispatcher anyway produced "'pets.Bird' object
    // has no attribute 'sound'" -- a sentence that is false about the program
    // and worse than the refusal it replaced.
    if (asAttribute)
      for (const auto &candidate : candidates) {
        if (resolveClassAttrSlot(candidate.second, methodName))
          continue;
        std::optional<mlir::Type> candidateType =
            types.lookupClass(candidate.second);
        if (!candidateType ||
            !lookupClassStaticAttr(*candidateType, methodName))
          return nullptr;
      }

    std::string symbol = "__lyvdisp$" + std::to_string(++syntheticFunctionCounter);
    virtualDispatchHelpers[key].symbol = symbol;

    auto forwarded = [&] {
      std::vector<parser::NodePtr> out;
      for (const std::string &name : parameterNames)
        out.push_back(synth::name(name, range));
      return out;
    };
    // ⭐ A PROPERTY READS INSTEAD OF CALLING, and it has no unbound spelling
    // to fall back on: `Base.v` through the class is a property OBJECT, which
    // this compiler does not represent. So the last arm reads through the
    // parameter itself, and the whole body is emitted with the unresolvable-
    // dispatch gate suppressed -- which is sound here and nowhere else,
    // because the candidates are enumerated MOST-DERIVED FIRST and every class
    // that declares the property is tested before its own ancestors. That
    // ordering is Python's own resolution, so each arm binds the body an
    // instance of that class would have run.
    auto keywordArguments = [&] {
      std::vector<parser::NodePtr> out;
      for (const std::string &name : keywordParameters)
        out.push_back(synth::keyword(name, synth::name(name, range), range));
      return out;
    };
    auto read = [&](parser::NodePtr receiverNode) {
      if (readsWithoutCall)
        return synth::attribute(std::move(receiverNode), methodName, range);
      return synth::callWithKeywords(
          synth::attribute(std::move(receiverNode), methodName, range),
          forwarded(), keywordArguments(), range);
    };
    std::vector<parser::NodePtr> body;
    for (const auto &candidate : candidates) {
      // ⭐ AN ATTRIBUTE ARM READS THROUGH THE CLASS, not through the narrowed
      // receiver. Inside the arm the runtime class IS this candidate (or a
      // subclass that declares nothing of its own, whose Python answer is this
      // candidate's binding), so `Candidate.attr` is the same value -- and it
      // is available on BOTH channels, where a read through the receiver needs
      // the cell only a main-module class has.
      parser::NodePtr subject =
          asAttribute ? synth::name(candidate.second, range)
                      : synth::name("__ly_recv", range);
      body.push_back(synth::ifStmt(
          synth::call(synth::name("isinstance", range),
                      {synth::name("__ly_recv", range),
                       synth::name(candidate.second, range)},
                      range),
          {synth::returnStmt(read(std::move(subject)), range)}, {}, range));
    }
    if (asAttribute) {
      body.push_back(synth::returnStmt(
          synth::attribute(synth::name(fallbackClass, range), methodName,
                           range),
          range));
    } else if (asProperty) {
      body.push_back(
          synth::returnStmt(read(synth::name("__ly_recv", range)), range));
    } else {
      std::vector<parser::NodePtr> fallbackArguments;
      fallbackArguments.push_back(synth::name("__ly_recv", range));
      for (parser::NodePtr &argument : forwarded())
        fallbackArguments.push_back(std::move(argument));
      body.push_back(synth::returnStmt(
          synth::callWithKeywords(
              synth::attribute(synth::name(fallbackClass, range), methodName,
                               range),
              std::move(fallbackArguments), keywordArguments(), range),
          range));
    }

    parser::NodePtr def = synth::functionDef(symbol, params, {},
                                             std::move(body), returns, {}, range);
    synthesizedIteratorDefs.push_back(def);
    FunctionSignature sig = types.functionSignature(*def);
    // ⭐ THE MEMO IS COMPLETE BEFORE THE BODY IS EMITTED, callable included.
    // A body reached from here may dispatch the same method again -- a
    // recursive class does it on its own field (`Pair.size` calling
    // `self.l.size()` where `l: N` and Pair overrides N.size) -- and an entry
    // that carried only the symbol answered "no dispatcher yet" to that call,
    // which came back as the refusal for a program the dispatcher was already
    // being built for.
    virtualDispatchHelpers[key].callable = sig.publicCallable;
    // ⛔ Bound as a NAME as well as a symbol: a bound method object's wrapper
    // reaches the dispatcher from a synthesized body, where the only spelling
    // available is a name. `$` cannot appear in a source identifier, so the
    // binding cannot shadow one.
    types.bindRootSymbol(symbol, sig.publicCallable);
    {
      // ⛔ The dispatcher is a FUNCTION, even when the call that needed it was
      // inside an inlined method body. Emitting it under the inliner's state
      // made its `return` branch to the INLINER's continuation block --
      // "reference to block defined in another region", from a `self.area()`
      // inside a base method. The loop and super contexts are cleared for the
      // same reason: none of them belong to this body.
      auto savedLoops = std::move(loopControlContexts);
      loopControlContexts.clear();
      auto savedInlineReturns = std::move(inlineReturnContexts);
      inlineReturnContexts.clear();
      auto savedSupers = std::move(superContexts);
      superContexts.clear();
      auto savedInlining = std::move(methodsBeingInlined);
      methodsBeingInlined.clear();
      // A real function gets a real LLVM function, so its traceback frame comes
      // from that; the inline stack belongs to the body being interrupted.
      auto savedInlineFrames = std::move(inlineFrames);
      inlineFrames.clear();
      if (readsWithoutCall)
        ++virtualPropertyBodyDepth;
      llvm::scope_exit restoreSuppression([&, readsWithoutCall] {
        if (readsWithoutCall)
          --virtualPropertyBodyDepth;
      });
      llvm::scope_exit restoreContexts([&] {
        loopControlContexts = std::move(savedLoops);
        inlineReturnContexts = std::move(savedInlineReturns);
        superContexts = std::move(savedSupers);
        methodsBeingInlined = std::move(savedInlining);
        inlineFrames = std::move(savedInlineFrames);
      });
      emitCallableFunction(*def, symbol, sig, {}, /*isLambda=*/false);
    }
    memo = virtualDispatchHelpers.find(key);
  }
  if (!memo->second.callable)
    return nullptr;
  return &memo->second;
}

// ⭐ A METHOD OBJECT DISPATCHES TOO, and it did not: `x.m()` on a base-typed
// receiver goes through the dispatcher above, while `m = x.m` built a wrapper
// around the STATIC body -- so one question asked in two spellings gave two
// answers, silently:
//
//     x: Base = Sub()
//     m = x.f
//     print(m())          # printed "B"; CPython prints "S", and `x.f()` does
//
// and every way of letting the object escape carried the same wrong body:
// passed as a `Callable` argument, stored in a list, rebound through a second
// name. The wrapper's body is the only thing that has to change: it forwards
// to the dispatcher instead of restating the base's method.
//
// ⛔ Not a new dispatcher, and not a new shape rule: `virtualDispatcherFor`
// decides both, so a method object over a shape the dispatch cannot restate
// (a generator, a property, defaults, *args) keeps the static wrapper it had.
const parser::Node *
ModuleEmitter::virtualMethodObjectDef(const parser::Node &anchor, Value receiver,
                                      const MethodBinding &binding) {
  if (binding.kind != "instance" || !binding.method)
    return nullptr;
  std::optional<std::string_view> methodName = ast::string(*binding.method, "name");
  if (!methodName)
    return nullptr;
  const parser::Node *arguments = ast::node(*binding.method, "args");
  if (!arguments)
    return nullptr;
  parser::NodePtr returns = sharedField(*binding.method, "returns");
  if (!returns)
    return nullptr;

  parser::SourceRange range = anchor.range;
  llvm::SmallVector<synth::Param, 4> params;
  llvm::SmallVector<std::string, 4> forwardedNames;
  std::string receiverName;
  auto contract = mlir::dyn_cast_if_present<py::ContractType>(receiver.type);
  if (!contract)
    return nullptr;
  for (llvm::StringRef field : {"posonlyargs", "args"})
    if (const auto *list = ast::nodeList(*arguments, field))
      for (const parser::NodePtr &argument : *list) {
        if (!argument)
          return nullptr;
        llvm::StringRef name = ast::nameSpelling(*argument);
        if (receiverName.empty()) {
          receiverName = name.str();
          params.push_back(synth::Param{
              receiverName, synth::name(contract.getContractName(), range)});
          continue;
        }
        parser::NodePtr annotation = sharedField(*argument, "annotation");
        if (!annotation)
          return nullptr;
        forwardedNames.push_back(name.str());
        params.push_back(synth::Param{name.str(), std::move(annotation)});
      }
  if (receiverName.empty())
    return nullptr;

  const VirtualDispatchHelper *helper = virtualDispatcherFor(
      anchor, receiver, *methodName,
      static_cast<unsigned>(forwardedNames.size()));
  if (!helper)
    return nullptr;

  std::vector<parser::NodePtr> callArguments;
  callArguments.push_back(synth::name(receiverName, range));
  for (const std::string &name : forwardedNames)
    callArguments.push_back(synth::name(name, range));
  std::vector<parser::NodePtr> body;
  body.push_back(synth::returnStmt(
      synth::call(synth::name(helper->symbol, range), std::move(callArguments),
                  range),
      range));
  parser::NodePtr def = synth::functionDef(
      "__lyvbound$" + std::to_string(++syntheticFunctionCounter), params, {},
      std::move(body), std::move(returns), {}, range);
  synthesizedIteratorDefs.push_back(def);
  return def.get();
}

// The AST call site: `x.m(a, b)` with a base-typed `x`.
std::optional<Value> ModuleEmitter::tryEmitVirtualDispatch(
    const parser::Node &expr, const parser::Node &calleeNode,
    const parser::Node *receiverNode, Value receiver,
    llvm::StringRef methodName) {
  // A keyword at the CALL site becomes a parameter of the dispatcher, which
  // forwards it by name. `**mapping` cannot: its names are not known here, so
  // there is nothing to declare.
  llvm::SmallVector<std::string, 2> keywordNames;
  const auto *keywords = ast::nodeList(expr, "keywords");
  if (keywords)
    for (const parser::NodePtr &keyword : *keywords) {
      std::optional<std::string_view> name =
          keyword ? ast::string(*keyword, "arg") : std::nullopt;
      if (!name)
        return std::nullopt;
      keywordNames.push_back(std::string(*name));
    }
  const auto *args = ast::nodeList(expr, "args");
  unsigned argumentCount = args ? static_cast<unsigned>(args->size()) : 0;
  if (args)
    for (const parser::NodePtr &argument : *args)
      if (argument && argument->kind == "Starred")
        return std::nullopt;
  const VirtualDispatchHelper *helper = virtualDispatcherFor(
      expr, receiver, methodName, argumentCount, /*asProperty=*/false,
      keywordNames);
  if (!helper)
    return std::nullopt;
  // Emitted only now: a bail above must not have evaluated the arguments, or
  // the normal path below would evaluate them a second time.
  llvm::SmallVector<Value, 4> callArguments{receiver};
  if (args)
    for (const parser::NodePtr &argument : *args)
      callArguments.push_back(emitExpr(argument.get()));
  if (keywords)
    for (const parser::NodePtr &keyword : *keywords)
      callArguments.push_back(emitExpr(ast::node(*keyword, "value")));
  Value callee = emitBindingRef(expr, helper->symbol, helper->callable);
  return emitCallableDispatch(
      expr, callee,
      emitCallOperands(expr, callArguments, /*includeAstArguments=*/false));
}

// The OPERATOR sites -- `len(x)`, `str(x)`, `x == y`, `x[i]`, `for x in y` --
// which reach a method with their operands already emitted. Same dispatcher,
// same memo: a dunder is a method, and eleven of them were measured silently
// wrong on a base-typed receiver before the refusal existed.
// The class-attribute spelling of the same read. Declines for a TYPE receiver:
// `Shape.kind` names the base's own binding and has never been ambiguous.
std::optional<Value>
ModuleEmitter::tryEmitVirtualAttributeRead(const parser::Node &anchor,
                                           Value receiver,
                                           llvm::StringRef attrName) {
  if (!mlir::isa_and_nonnull<py::ContractType>(receiver.type))
    return std::nullopt;
  const VirtualDispatchHelper *helper = virtualDispatcherFor(
      anchor, receiver, attrName, /*argumentCount=*/0, /*asProperty=*/false,
      /*keywordNames=*/{}, /*asAttribute=*/true);
  if (!helper)
    return std::nullopt;
  Value callee = emitBindingRef(anchor, helper->symbol, helper->callable);
  return emitCallableDispatch(
      anchor, callee,
      emitCallOperands(anchor, {receiver}, /*includeAstArguments=*/false));
}

std::optional<Value>
ModuleEmitter::tryEmitVirtualPropertyRead(const parser::Node &anchor,
                                          Value receiver,
                                          llvm::StringRef propertyName) {
  const VirtualDispatchHelper *helper = virtualDispatcherFor(
      anchor, receiver, propertyName, /*argumentCount=*/0,
      /*asProperty=*/true);
  if (!helper)
    return std::nullopt;
  Value callee = emitBindingRef(anchor, helper->symbol, helper->callable);
  return emitCallableDispatch(
      anchor, callee,
      emitCallOperands(anchor, {receiver}, /*includeAstArguments=*/false));
}

std::optional<Value> ModuleEmitter::tryEmitVirtualDispatchWithValues(
    const parser::Node &anchor, Value receiver, llvm::StringRef methodName,
    llvm::ArrayRef<Value> positional) {
  const VirtualDispatchHelper *helper = virtualDispatcherFor(
      anchor, receiver, methodName, static_cast<unsigned>(positional.size()));
  if (!helper)
    return std::nullopt;
  llvm::SmallVector<Value, 4> callArguments{receiver};
  callArguments.append(positional.begin(), positional.end());
  Value callee = emitBindingRef(anchor, helper->symbol, helper->callable);
  return emitCallableDispatch(
      anchor, callee,
      emitCallOperands(anchor, callArguments, /*includeAstArguments=*/false));
}

std::optional<Value>
ModuleEmitter::tryEmitIntBaseCall(const parser::Node &expr,
                                  const parser::Node *calleeNode) {
  if (!callsUnshadowedBuiltin(calleeNode, "int"))
    return std::nullopt;
  const auto *args = ast::nodeList(expr, "args");
  const auto *keywords = ast::nodeList(expr, "keywords");
  // ⭐ `int(s, base=16)` is the same call as `int(s, 16)`, and CPython names
  // that parameter, so the keyword spelling is the one a reader writes. It
  // reached the class-instantiation path instead ("builtins.int does not
  // provide manifest method '__init__'"), because this interception took the
  // base positionally and declined the moment a keyword appeared.
  const parser::Node *subject = nullptr;
  const parser::Node *base = nullptr;
  if (args && args->size() == 2 && (!keywords || keywords->empty())) {
    subject = args->front().get();
    base = (*args)[1].get();
  } else if (args && args->size() == 1 && keywords && keywords->size() == 1 &&
             keywords->front()) {
    std::optional<std::string_view> keyword =
        ast::string(*keywords->front(), "arg");
    if (keyword && *keyword == "base") {
      subject = args->front().get();
      base = ast::node(*keywords->front(), "value");
    }
  }
  if (!subject || !base || subject->kind == "Starred" ||
      base->kind == "Starred")
    return std::nullopt;
  mlir::Type subjectType = types.widenLiteral(types.inferExpr(subject));
  if (subjectType != types.contract("builtins.str"))
    return std::nullopt;

  parser::SourceRange range = expr.range;
  if (intBaseHelperSymbol.empty()) {
    std::string symbol = "__lyintbase$" + std::to_string(++syntheticFunctionCounter);
    auto name = [&](llvm::StringRef id) { return synth::name(id, range); };
    auto str = [&](llvm::StringRef text) { return synth::strConstant(text, range); };
    auto num = [&](std::int64_t value) { return synth::intConstant(value, range); };
    auto slice = [&](llvm::StringRef target, std::int64_t from) {
      parser::NodePtr sliceNode = parser::makeNode("Slice", range);
      parser::addField(*sliceNode, "lower", num(from));
      return synth::subscript(name(target), std::move(sliceNode), range);
    };

    // The message CPython raises, rebuilt from the ORIGINAL argument:
    //   invalid literal for int() with base 16: '  zz  '
    auto invalidLiteral = [&] {
      parser::NodePtr message = synth::binOp(
          synth::binOp(
              synth::binOp(str("invalid literal for int() with base "), "Add",
                           synth::call(name("str"), {name("base")}, range),
                           range),
              "Add", str(": "), range),
          "Add", synth::reprCall(name("s"), range), range);
      return synth::raiseStmt(
          synth::call(name("ValueError"), {std::move(message)}, range), range);
    };

    std::vector<parser::NodePtr> body;
    body.push_back(synth::ifStmt(
        synth::boolOp(
            "And",
            {synth::compare(name("base"), "NotEq", num(0), range),
             synth::orChain(
                 {synth::compare(name("base"), "Lt", num(2), range),
                  synth::compare(name("base"), "Gt", num(36), range)},
                 range)},
            range),
        {synth::raiseValueError("int() base must be >= 2 and <= 36, or 0",
                                range)},
        {}, range));
    body.push_back(synth::assign(
        name("text"), synth::methodCall(name("s"), "strip", {}, range), range));
    body.push_back(
        synth::assign(name("negative"), synth::constantBool(false, range), range));
    body.push_back(synth::ifStmt(
        synth::methodCall(name("text"), "startswith", {str("-")}, range),
        {synth::assign(name("negative"), synth::constantBool(true, range), range),
         synth::assign(name("text"), slice("text", 1), range)},
        {synth::ifStmt(
            synth::methodCall(name("text"), "startswith", {str("+")}, range),
            {synth::assign(name("text"), slice("text", 1), range)}, {}, range)},
        range));
    body.push_back(synth::assign(
        name("body"), synth::methodCall(name("text"), "lower", {}, range), range));
    // An underscore is legal only BETWEEN digits, so the scan starts as if it
    // had just seen one -- except right after a prefix, where CPython allows
    // `int("0x_1f", 16)`.
    body.push_back(
        synth::assign(name("pending"), synth::constantBool(true, range), range));
    // ⭐ base=0 IS THE SAME PARSE WITH THE RADIX READ OFF THE PREFIX, which is
    // why it is a variable rather than a second function: everything below
    // consumes `radix`, and only the ERROR MESSAGES keep `base`, because
    // CPython reports "with base 0" for a string it auto-detected.
    body.push_back(synth::assign(name("radix"), name("base"), range));
    {
      std::vector<parser::NodePtr> detect;
      detect.push_back(synth::assign(name("radix"), num(10), range));
      for (auto [prefixBase, prefix] :
           {std::pair<std::int64_t, const char *>{16, "0x"},
            {8, "0o"},
            {2, "0b"}})
        detect.push_back(synth::ifStmt(
            synth::methodCall(name("body"), "startswith", {str(prefix)},
                              range),
            {synth::assign(name("radix"), num(prefixBase), range)}, {},
            range));
      body.push_back(synth::ifStmt(
          synth::compare(name("base"), "Eq", num(0), range), std::move(detect),
          {}, range));
    }
    // The prefix is accepted only when it agrees with the radix, as CPython does.
    for (auto [prefixBase, prefix] :
         {std::pair<std::int64_t, const char *>{16, "0x"}, {8, "0o"}, {2, "0b"}})
      body.push_back(synth::ifStmt(
          synth::boolOp("And",
                        {synth::compare(name("radix"), "Eq", num(prefixBase),
                                        range),
                         synth::methodCall(name("body"), "startswith",
                                           {str(prefix)}, range)},
                        range),
          {synth::assign(name("body"), slice("body", 2), range),
           synth::assign(name("pending"), synth::constantBool(false, range),
                         range)},
          {}, range));
    // ⛔ AND A BARE LEADING ZERO IS AN ERROR UNDER base=0, which is the half
    // that makes auto-detect a different parse rather than a default: CPython
    // reads `int("012", 0)` as an ambiguity between the old octal spelling and
    // decimal and refuses it, while `int("00", 0)` and `int("0_0", 0)` are 0.
    {
      std::vector<parser::NodePtr> scan{synth::ifStmt(
          synth::boolOp("And",
                        {synth::compare(name("ch"), "NotEq", str("0"), range),
                         synth::compare(name("ch"), "NotEq", str("_"), range)},
                        range),
          {invalidLiteral()}, {}, range)};
      body.push_back(synth::ifStmt(
          synth::boolOp(
              "And",
              {synth::compare(name("base"), "Eq", num(0), range),
               synth::boolOp(
                   "And",
                   {synth::compare(name("radix"), "Eq", num(10), range),
                    synth::methodCall(name("body"), "startswith", {str("0")},
                                      range)},
                   range)},
              range),
          {synth::forStmt(name("ch"), name("body"), std::move(scan), {},
                          range)},
          {}, range));
    }
    body.push_back(synth::assign(
        name("alphabet"), str("0123456789abcdefghijklmnopqrstuvwxyz"), range));
    body.push_back(synth::assign(name("total"), num(0), range));
    body.push_back(synth::assign(name("seen"), num(0), range));
    std::vector<parser::NodePtr> loop;
    loop.push_back(synth::ifStmt(
        synth::compare(name("ch"), "Eq", str("_"), range),
        {synth::ifStmt(name("pending"), {invalidLiteral()}, {}, range),
         synth::assign(name("pending"), synth::constantBool(true, range), range),
         synth::continueStmt(range)},
        {}, range));
    loop.push_back(synth::assign(
        name("digit"),
        synth::methodCall(name("alphabet"), "find", {name("ch")}, range), range));
    loop.push_back(synth::ifStmt(
        synth::orChain({synth::compare(name("digit"), "Lt", num(0), range),
                        synth::compare(name("digit"), "GtE", name("radix"),
                                       range)},
                       range),
        {invalidLiteral()}, {}, range));
    loop.push_back(synth::assign(
        name("total"),
        synth::binOp(synth::binOp(name("total"), "Mult", name("radix"), range),
                     "Add", name("digit"), range),
        range));
    loop.push_back(synth::assign(
        name("seen"), synth::binOp(name("seen"), "Add", num(1), range), range));
    loop.push_back(
        synth::assign(name("pending"), synth::constantBool(false, range), range));
    body.push_back(
        synth::forStmt(name("ch"), name("body"), std::move(loop), {}, range));
    // A trailing underscore and an empty digit run are the same refusal.
    body.push_back(synth::ifStmt(
        synth::orChain({name("pending"),
                        synth::compare(name("seen"), "Eq", num(0), range)},
                       range),
        {invalidLiteral()}, {}, range));
    {
      parser::NodePtr negate = parser::makeNode("UnaryOp", range);
      parser::addField(*negate, "op", parser::makeNode("USub", range));
      parser::addField(*negate, "operand", name("total"));
      body.push_back(synth::ifStmt(
          name("negative"),
          {synth::returnStmt(std::move(negate), range)}, {}, range));
    }
    body.push_back(synth::returnStmt(name("total"), range));

    llvm::SmallVector<synth::Param, 2> params{
        synth::Param{"s", synth::name("str", range)},
        synth::Param{"base", synth::name("int", range)}};
    parser::NodePtr def =
        synth::functionDef(symbol, params, {}, std::move(body),
                           synth::name("int", range), {}, range);
    synthesizedIteratorDefs.push_back(def);
    FunctionSignature sig = types.functionSignature(*def);
    emitCallableFunction(*def, symbol, sig, {}, /*isLambda=*/false);
    intBaseHelperSymbol = symbol;
    intBaseHelperCallable = sig.publicCallable;
  }

  llvm::SmallVector<Value, 2> arguments{emitExpr(subject), emitExpr(base)};
  Value callee =
      emitBindingRef(expr, intBaseHelperSymbol, intBaseHelperCallable);
  return emitCallableDispatch(
      expr, callee,
      emitCallOperands(expr, arguments, /*includeAstArguments=*/false));
}

std::optional<Value>
ModuleEmitter::tryEmitIntCall(const parser::Node &expr,
                              const parser::Node *calleeNode) {
  // int(x) is __int__ dispatch / literal parsing (CPython semantics), not
  // construction — intercept before the class-instantiation paths claim
  // builtins.int. Zero-argument int() stays on the instantiation path.
  if (!callsUnshadowedBuiltin(calleeNode, "int"))
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
  if (argumentType == types.boolType()) {
    // int(True) == 1 / int(False) == 0: widen the truth bit.
    return emitIntFromBool(expr, emitExpr(intArgs->front().get()));
  }
  if (argumentType == types.strType() || argumentType == types.floatType() ||
      argumentType == types.contract("builtins.bytes")) {
    // The runtime-level __int__ methods of str (base-10 parse), bytes (the
    // same parse over the payload) and float (truncation) are deliberately not
    // part of the typed manifest surface — CPython has no str.__int__ — so the
    // contract is built here instead of going through method inference.
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

std::optional<Value>
ModuleEmitter::tryEmitBoolCall(const parser::Node &expr,
                               const parser::Node *calleeNode) {
  if (!callsUnshadowedBuiltin(calleeNode, "bool"))
    return std::nullopt;
  const auto *args = ast::nodeList(expr, "args");
  const auto *keywords = ast::nodeList(expr, "keywords");
  if (keywords && !keywords->empty())
    return std::nullopt;
  if (!args || args->empty()) {
    mlir::Type literalType = types.literal("False");
    auto constant = py::BoolConstantOp::create(builder, loc(expr), literalType,
                                               builder.getBoolAttr(false));
    return Value{constant.getResult(), literalType};
  }
  if (args->size() != 1 || !args->front() ||
      args->front()->kind == "Starred")
    return std::nullopt;
  const parser::Node *argNode = args->front().get();
  mlir::Type argumentType = types.widenLiteral(types.inferExpr(argNode));
  // bool(n) on numbers is an EXPLICIT conversion — R1 only rejects the
  // implicit truthiness of `if n:` — so it desugars to the comparison the
  // diagnostic would suggest.
  if (argumentType == types.intType() || argumentType == types.floatType()) {
    parser::NodePtr zero = parser::makeNode("Constant", expr.range);
    if (argumentType == types.floatType())
      parser::addField(*zero, "value", 0.0);
    else
      parser::addField(*zero, "value", std::int64_t{0});
    parser::NodePtr op = parser::makeNode("NotEq", expr.range);
    parser::NodePtr compare = parser::makeNode("Compare", expr.range);
    parser::addField(*compare, "left", (*args)[0]);
    parser::addField(*compare, "ops", std::vector<parser::NodePtr>{op});
    parser::addField(*compare, "comparators",
                     std::vector<parser::NodePtr>{zero});
    return coerceValue(emitExpr(compare.get()), types.boolType(), expr);
  }
  // Everything else rides the same truthiness emitBoolValue implements for
  // conditions (containers by emptiness, Optional by None-ness, bool as-is).
  Value argument = emitExpr(argNode);
  mlir::Value bit = emitBoolValue(argument, expr);
  return boxedBool(builder, loc(expr), types, bit);
}

std::optional<Value>
ModuleEmitter::tryEmitAsciiCall(const parser::Node &expr,
                                const parser::Node *calleeNode) {
  if (!callsUnshadowedBuiltin(calleeNode, "ascii"))
    return std::nullopt;
  const auto *args = ast::nodeList(expr, "args");
  const auto *keywords = ast::nodeList(expr, "keywords");
  if (!args || args->size() != 1 || (keywords && !keywords->empty()) ||
      !args->front() || args->front()->kind == "Starred") {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, expr.range.start,
        "ascii() takes exactly one argument"});
    return emitNone(expr);
  }
  Value argument = emitExpr(args->front().get());
  if (std::optional<Value> converted =
          emitConversionValue(expr, argument, 'a'))
    return converted;
  diagnostics.push_back(parser::Diagnostic{
      parser::Severity::Error, expr.range.start,
      "ascii() is not supported for this argument type"});
  return emitNone(expr);
}

std::optional<Value>
ModuleEmitter::tryEmitIssubclassCall(const parser::Node &expr,
                                     const parser::Node *calleeNode) {
  if (!callsUnshadowedBuiltin(calleeNode, "issubclass"))
    return std::nullopt;
  const auto *args = ast::nodeList(expr, "args");
  const auto *keywords = ast::nodeList(expr, "keywords");
  auto rejectIssubclass = [&](llvm::StringRef reason) -> std::optional<Value> {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, expr.range.start, std::string(reason)});
    return emitNone(expr);
  };
  if (!args || args->size() != 2 || (keywords && !keywords->empty()))
    return rejectIssubclass("issubclass() takes exactly two arguments");
  // Static classes only: the hierarchy is compile-time (C3 linearized), so
  // the answer folds to a constant.
  auto classOf = [&](const parser::Node *node) -> std::optional<mlir::Type> {
    if (!node)
      return std::nullopt;
    std::string qualified = ast::qualifiedName(node);
    if (qualified.empty())
      return std::nullopt;
    return types.lookupClass(qualified);
  };
  std::optional<mlir::Type> subClass = classOf((*args)[0].get());
  std::optional<mlir::Type> superClass = classOf((*args)[1].get());
  if (!subClass || !superClass)
    return rejectIssubclass(
        "issubclass() requires statically resolvable class names");
  bool truth = *subClass == *superClass ||
               pythonSubclassOf(*subClass, *superClass, types, module);
  mlir::Type literalType = types.literal(truth ? "True" : "False");
  auto constant = py::BoolConstantOp::create(builder, loc(expr), literalType,
                                             builder.getBoolAttr(truth));
  return Value{constant.getResult(), literalType};
}

// The bottom rung of the numeric tower: bool IS an int, so widening the truth
// bit is the whole conversion. `int(True)` has always spelled it; this is that
// spelling given a name so the OPERAND promotion in emitBinary can reach it.
//
// ⛔ Why NOT `coerceValue` to int, which is what the note here used to call the
// obvious repair: it produces a bundle carrying bool's single value where the
// int ABI expects three, and took out `float_floordiv_mod_round` with "runtime
// bundle for builtins.int has 1 values, but ABI expects 3". A rung of the tower
// is a CONVERSION, not a retyping -- the same reason `emitFloatFromInt` exists
// one rung up.
Value ModuleEmitter::emitIntFromBool(const parser::Node &anchor,
                                     Value argument) {
  mlir::Value bit = emitBoolValue(argument, anchor);
  auto wide = mlir::arith::ExtUIOp::create(
      builder, loc(anchor), mlir::IntegerType::get(&context, 64), bit);
  auto op = py::CastFromPrimOp::create(builder, loc(anchor), types.intType(),
                                       wide.getResult());
  return Value{op.getResult(), types.intType()};
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
  if (!callsUnshadowedBuiltin(calleeNode, "float"))
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
  if (argumentType == types.boolType()) {
    // ⛔ Why NOT let this fall through to the class-instantiation paths, which
    // is what it did: they claim builtins.float and report "does not provide
    // manifest method '__init__'", naming a constructor CPython never calls.
    // float(True) is int's __float__ reached through bool, so both rungs run.
    Value argument = emitExpr(floatArgs->front().get());
    return emitFloatFromInt(expr, emitIntFromBool(expr, argument));
  }
  if (argumentType == types.strType()) {
    // The str parse, the twin of the str.__int__ dispatch above: also a
    // runtime-level __float__ that the typed manifest surface does not carry
    // (CPython has no str.__float__ either), so its contract is built here.
    Value argument =
        coerceValue(emitExpr(floatArgs->front().get()), argumentType, expr);
    mlir::Type resultType = types.floatType();
    mlir::Type contract = py::CallableType::get(&context, {argumentType}, {},
                                                {}, {}, {resultType});
    auto op = py::FloatOp::create(
        builder, loc(expr), resultType,
        mlir::FlatSymbolRefAttr::get(&context, "__float__"),
        mlir::TypeAttr::get(contract), argument.value);
    return Value{op.getResult(), resultType};
  }
  return std::nullopt;
}

// Two-argument pow IS the ** operator: CPython's builtin_pow calls
// PyNumber_Power with Py_None for the modulus, the same slot `**` reaches.
// The manifest declares only the three-argument form (the modular one, which
// has no operator spelling), so `pow(2, 10)` was refused for an arity the
// operator accepts.
//
// ⛔ Why NOT a second manifest contract for the two-argument form: `**` is
// not one call. It picks between the int and float towers, folds a negative
// literal exponent to the float path, and reaches the complex tower -- all
// in emitBinary. A parallel two-argument pow would be that ladder again,
// drifting from the day it is written. Rewriting to the operator's own AST
// node is the ladder itself.
std::optional<Value>
ModuleEmitter::tryEmitPowCall(const parser::Node &expr,
                              const parser::Node *calleeNode) {
  if (!callsUnshadowedBuiltin(calleeNode, "pow"))
    return std::nullopt;
  const auto *args = ast::nodeList(expr, "args");
  const auto *keywords = ast::nodeList(expr, "keywords");
  if (!args || args->size() != 2 || (keywords && !keywords->empty()))
    return std::nullopt;
  const parser::NodePtr &base = (*args)[0];
  const parser::NodePtr &exponent = (*args)[1];
  if (!base || !exponent || base->kind == "Starred" ||
      exponent->kind == "Starred")
    return std::nullopt;
  parser::NodePtr rewritten = parser::makeNode("BinOp", expr.range);
  parser::addField(*rewritten, "left", base);
  parser::addField(*rewritten, "op", parser::makeNode("Pow", expr.range));
  parser::addField(*rewritten, "right", exponent);
  return emitBinary(*rewritten);
}

std::optional<Value>
ModuleEmitter::tryEmitStrCall(const parser::Node &expr,
                              const parser::Node *calleeNode) {
  // str(x) is __str__ dispatch (CPython semantics), not construction —
  // intercept before the class-instantiation paths claim builtins.str.
  if (!callsUnshadowedBuiltin(calleeNode, "str"))
    return std::nullopt;
  const auto *strArgs = ast::nodeList(expr, "args");
  const auto *strKeywords = ast::nodeList(expr, "keywords");
  auto strClass = types.lookupClass("str");
  std::optional<llvm::StringRef> strSymbol =
      strClass ? contractName(*strClass) : std::nullopt;
  // ⭐ `str(b, "utf-8")` IS `b.decode("utf-8")`: CPython's str() takes the
  // bytes-and-encoding form, and the runtime has all three decode arities
  // already. Without this the class-instantiation path claimed the call and
  // reported "builtins.str does not provide manifest method '__init__'" -- a
  // message about str, when it is the ARGUMENT that decides which str() this
  // is. The one-argument spelling stays the __str__ dispatch below, which is
  // why `str(b"ab")` correctly prints the repr and this form does not.
  if (strArgs && strArgs->size() >= 2 && strArgs->size() <= 3 &&
      (!strKeywords || strKeywords->empty()) &&
      llvm::all_of(*strArgs,
                   [](const parser::NodePtr &argument) {
                     return argument && argument->kind != "Starred";
                   }) &&
      types.widenLiteral(types.inferExpr(strArgs->front().get())) ==
          types.contract("builtins.bytes")) {
    std::vector<parser::NodePtr> rest(strArgs->begin() + 1, strArgs->end());
    parser::NodePtr decoded = synth::methodCall(strArgs->front(), "decode",
                                                std::move(rest), expr.range);
    synthesizedIteratorDefs.push_back(decoded);
    return emitExpr(decoded.get());
  }
  if (strSymbol && *strSymbol == "builtins.str" && strArgs &&
      strArgs->size() == 1 && (!strKeywords || strKeywords->empty())) {
    mlir::Type argumentType =
        types.widenLiteral(types.inferExpr(strArgs->front().get()));
    mlir::Type strType = types.contract("builtins.str");
    if (argumentType == strType) {
      // str is immutable, so str(s) is the identity (CPython returns s).
      return coerceValue(emitExpr(strArgs->front().get()), strType, expr);
    }
    // None has no physical header for a dispatch to receive; its text is a
    // compile-time constant anyway (the emitStringifyValue rule).
    if (argumentType == types.none())
      return emitStrLiteralPiece(expr, "None");
    // A union renders by tag, which is `emitStringifyValue`'s union arm and
    // not anything the ladder below can ask: `str(d.get("b"))` fell all the
    // way through to the str CLASS and came out as "unresolved name 'repr'".
    if (mlir::isa<py::UnionType>(argumentType))
      if (std::optional<Value> rendered =
              emitStringifyValue(expr, emitExpr(strArgs->front().get())))
        return *rendered;
    if (lookupClassMethod(argumentType, "__str__")) {
      Value argument = emitExpr(strArgs->front().get());
      if (std::optional<Value> dispatched =
              tryEmitClassDunder(expr, argument, "__str__"))
        return *dispatched;
    }
    // Manifest __str__ evidence over-accepts through the object contract
    // (container manifests only implement __repr__), so gate the __str__
    // dispatch on the exception taxonomy — the one family whose __str__
    // (the message) differs from __repr__ (ClassName(...)). Erased
    // builtins.object receivers keep the __str__ dispatch too: the manifest
    // object __str__ resolves the payload class dynamically and falls back
    // to the repr form exactly where CPython's str(x) does.
    auto isErasedObject = [&](mlir::Type type) {
      auto contractType = mlir::dyn_cast<py::ContractType>(type);
      return contractType &&
             contractType.getContractName() == "builtins.object";
    };
    if (isCellContract(argumentType))
      argumentType = types.widenLiteral(cellContentType(argumentType));
    if (CallInferenceResult inference =
            isExceptionContractType(argumentType) ||
                    isErasedObject(argumentType)
                ? types.inferMethodCallWithEvidence(argumentType, "__str__", {})
                : CallInferenceResult()) {
      Value argument =
          coerceValue(emitExpr(strArgs->front().get()), argumentType, expr);
      auto op = py::StrOp::create(
          builder, loc(expr), strType,
          mlir::FlatSymbolRefAttr::get(&context, "__str__"),
          mlir::TypeAttr::get(callProtocolFor(inference)), argument.value);
      return Value{op.getResult(), strType};
    }
    // No distinct __str__: str(x) is repr(x) (CPython object.__str__
    // delegates to type(x).__repr__). Reroute through the repr call path
    // instead of teaching this path a second dispatch ladder — repr owns
    // the source-__repr__ inline and the default-object-repr fallback.
    if (!programBindsName("repr")) {
      parser::NodePtr reprName = synth::name(std::string("repr"), expr.range);
      parser::NodePtr reprCall = synth::call(std::move(reprName), std::vector<parser::NodePtr>{strArgs->front()}, expr.range);
      synthesizedIteratorDefs.push_back(reprCall);
      return emitCall(*reprCall);
    }
    // Fall through to the instantiation path's explicit rejection.
  }
  return std::nullopt;
}

bool ModuleEmitter::isExceptionContractType(mlir::Type type) const {
  auto contractType = mlir::dyn_cast<py::ContractType>(type);
  if (!contractType)
    return false;
  // ⛔ The builtin exception TABLE is consulted for BUILTIN contracts only, so
  // the leaf of a dotted name and never a bare one. A source class may be
  // written with the same name --
  //
  //     class ConnectionError:
  //         def __repr__(self) -> str: return "CE-repr"
  //     print(str(ConnectionError()))
  //
  // -- and taking it for the builtin gave it the taxonomy's str/repr split,
  // which it does not have: `str(x)` was refused with "runtime manifest has no
  // ConnectionError.__str__ method". A user class that really IS an exception
  // reaches the manifest-subclass walk below, which asks about the hierarchy
  // rather than about the spelling.
  auto [qualifier, leaf] = contractType.getContractName().rsplit('.');
  if (!leaf.empty() && py::exceptions::findByName(leaf) != nullptr)
    return true;
  // User exception classes share the taxonomy's str/repr semantics (str is
  // the message, repr is ClassName(...)): the manifest-subclass walk is the
  // class-side analog of the taxonomy name lookup.
  return py::protocols::Table::get(context).isManifestSubclassOf(
      type, "builtins.BaseException");
}

std::optional<Value>
ModuleEmitter::tryEmitListCall(const parser::Node &expr,
                               const parser::Node *calleeNode) {
  // list(<genexpr>) is the list comprehension over the same element/generator
  // chain — route to the comprehension emitter before the class-instantiation
  // paths claim builtins.list.
  if (!callsUnshadowedBuiltin(calleeNode, "list"))
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
  if (!callsUnshadowedBuiltin(calleeNode, "print"))
    return std::nullopt;
  const auto *printArgs = ast::nodeList(expr, "args");
  const auto *printKeywords = ast::nodeList(expr, "keywords");
  // ⭐ `sep=` IS THIS LADDER'S OWN SEPARATOR. The join below already builds the
  // space-separated string CPython's default produces, so a different one is a
  // different constant and nothing else. Any keyword at all used to make the
  // whole ladder decline, and the call then landed on `builtins.print`'s
  // contract -- which has no keyword parameters, so the report was "call
  // arguments do not match the Callable contract" with the offending keyword
  // named nowhere in it.
  //
  // ⛔ `end=` is NOT here, and the reason is the sink rather than the join.
  // The only builtin write is `LyUnicode_PrintLine`, which appends the
  // newline; `LyUnicode_Print` next to it does not, and nothing names it as a
  // builtin, so the emitter cannot reach it. Saying that by name beats the
  // contract mismatch, which is what the diagnostic below is for.
  const parser::Node *separatorNode = nullptr;
  const parser::Node *endNode = nullptr;
  std::string refusedKeyword;
  if (printKeywords)
    for (const parser::NodePtr &keyword : *printKeywords) {
      if (!keyword) {
        refusedKeyword = "**";
        break;
      }
      std::optional<llvm::StringRef> name = ast::string(*keyword, "arg");
      if (name && *name == "sep" && !separatorNode) {
        separatorNode = ast::node(*keyword, "value");
        if (separatorNode)
          continue;
      }
      if (name && *name == "end" && !endNode) {
        endNode = ast::node(*keyword, "value");
        if (endNode)
          continue;
      }
      refusedKeyword = name ? name->str() : "**";
      break;
    }
  if (!refusedKeyword.empty()) {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, expr.range.start,
        "print() does not take the keyword argument '" + refusedKeyword + "'"});
    return emitNone(expr);
  }
  // ⭐ `end=` IS THE FILE'S OWN `write`, which is what CPython's print does:
  // `builtin_print_impl` writes the joined arguments and then writes `end`,
  // and the only reason this could not be spelled here was the sink. The
  // builtin `print` reaches `LyUnicode_PrintLine`, which appends the newline;
  // `sys.stdout.write` reaches `LyTextIO_Write`, which does not -- and the
  // binding resolves without an import, because the lowering answers
  // `py.binding.ref "sys.stdout"` from the manifest primitive.
  //
  // ⛔ ONE write and not two: `end` is concatenated onto the joined arguments
  // rather than written after them, so an interleaved failure cannot leave
  // half a line. CPython makes two calls, but the difference is only
  // observable through a `file` this fold does not take.
  auto writeWithEnd = [&](Value text) -> std::optional<Value> {
    mlir::Type strContract = types.contract("builtins.str");
    Value end = coerceValue(emitExpr(endNode), strContract, expr);
    text = emitBinarySpecial<py::AddOp>(expr, "__add__", text, end,
                                        strContract);
    mlir::Type wrapper = types.contract("_io.TextIOWrapper");
    Value stream = emitBindingRef(expr, "sys.stdout", wrapper, {});
    std::string streamName =
        "__lyprintout" + std::to_string(++syntheticFunctionCounter);
    std::string textName =
        "__lyprinttext" + std::to_string(++syntheticFunctionCounter);
    values[streamName] = stream;
    types.bindSymbol(streamName, wrapper);
    values[textName] = text;
    types.bindSymbol(textName, text.type);
    parser::NodePtr write = synth::methodCall(
        synth::name(streamName, expr.range), "write",
        {synth::name(textName, expr.range)}, expr.range);
    (void)emitExpr(write.get());
    synthesizedIteratorDefs.push_back(std::move(write));
    values.erase(streamName);
    values.erase(textName);
    return emitNone(expr);
  };
  bool noPrintKeywords = !separatorNode && !endNode;
  if (endNode && (!printArgs || printArgs->empty())) {
    mlir::Type emptyType = types.literal("\"\"");
    auto empty = py::StrConstantOp::create(builder, loc(expr), emptyType,
                                           builder.getStringAttr(""));
    return writeWithEnd(coerceValue(Value{empty.getResult(), emptyType},
                                    types.contract("builtins.str"), expr));
  }
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
  // ⭐ ONE argument normally goes to the manifest print, which renders more
  // than this ladder can and is the reason the threshold is two. A UNION is
  // the case it cannot take: it has no header to dispatch on, so
  // `print(d.get("b"))` came out of the lowering as "unnarrowed
  // !py.union<...> cannot be used where a concrete object is required".
  // Rendering it here -- by tag, member by member -- is the only place that
  // question can be answered, because only the emitter still knows the
  // members.
  // ⛔ A CLASS IS NOT A VALUE HERE, and the lowering could not say so: it
  // reported "runtime method receiver has no concrete contract" for
  // `print(type(1))` and `print(x.__class__)`, which describes the dispatch
  // rather than the program. A type object is compile-time evidence in this
  // compiler and has no object handle to render; `type(x).__name__` and
  // `type(x) is int` are folds and keep working.
  //
  // ⛔ The two SPELLINGS and not only the inferred type: `type(x)` is folded by
  // the emitter and the inference walk does not model the fold, so it answers
  // `object` for the very expression this is about.
  auto rendersAClass = [&](const parser::Node &argument) {
    if (mlir::isa_and_nonnull<py::TypeType>(
            types.widenLiteral(types.inferExpr(&argument))))
      return true;
    if (argument.kind == "Call")
      return callsUnshadowedBuiltin(ast::node(argument, "func"), "type");
    return argument.kind == "Attribute" &&
           ast::string(argument, "attr") == "__class__";
  };
  if (printArgs)
    for (const parser::NodePtr &argument : *printArgs)
      if (argument && argument->kind != "Starred" &&
          rendersAClass(*argument)) {
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, expr.range.start,
            "print() cannot render a class: a type object is compile-time "
            "evidence here and has no runtime value, so write "
            "`type(x).__name__` for the name"});
        return emitNone(expr);
      }
  bool singleUnionArgument =
      printArgs && printArgs->size() == 1 && printArgs->front() &&
      printArgs->front()->kind != "Starred" &&
      mlir::isa<py::UnionType>(
          types.widenLiteral(types.inferExpr(printArgs->front().get())));
  // With an explicit separator the join is the whole point, so one argument
  // takes this path too -- there is nothing to separate, and routing it to the
  // manifest print would silently drop the keyword.
  bool plainArguments =
      printArgs && (printArgs->size() >= 2 || singleUnionArgument ||
                    ((separatorNode || endNode) && !printArgs->empty()));
  if (plainArguments)
    for (const parser::NodePtr &argument : *printArgs)
      if (!argument || argument->kind == "Starred")
        plainArguments = false;
  if (plainArguments) {
    mlir::Type strType = types.contract("builtins.str");
    // ⭐ Evaluate every argument, THEN render the values, because that is the
    // order CPython runs in: builtin_print_impl is handed an already-built
    // argument tuple and only then calls str() on each element.
    //
    //     a: list[int] = [1, 2]
    //     print(a, a.pop())      # printed [1, 2] 2; CPython prints [1] 2
    //
    // Rendering as the walk evaluated showed the first argument a list the
    // second had not shortened yet -- and the same for a source class, whose
    // `__str__` ran before the later argument mutated what that body reads.
    //
    // ⛔ Why NOT keep a `stringify` that picks the renderer from the
    // argument's STATIC type, which is what stood here: the pick has to
    // happen before the value exists, so the one case inference cannot answer
    // (it says `builtins.object` for the max/min fold this file performs and
    // does not model) emitted the argument speculatively and rewound the
    // builder when the guess was wrong. A rewind moves the insertion point;
    // it does not erase what it rewinds past, so the arms behind it emitted
    // the argument a SECOND time -- `print(max(f(), g()), 0)` ran f and g
    // twice, and `print(max(f(), 3), 0)` left the orphaned literal for the
    // ownership verifier to report as "reaches function exit without
    // release". Asking the value deletes the guess, the rewind, and with it
    // the restore into a block that a reducer or a comprehension has since
    // given a terminator ("operation with block successors must terminate its
    // parent block").
    //
    // ⛔ Why NOT a renderer local to print: `emitStringifyValue` is this same
    // ladder over a value, already shared by f-strings, format() and %s, and
    // the two copies had drifted -- the copy here could not render `None`
    // ("types.NoneType runtime object has no physical header value") and
    // asked the subclass-override gate only about `__repr__`, so a base-typed
    // receiver whose subclass overrides `__str__` rendered the base's
    // `__repr__` here while `f"{x!s}"` refused it.
    llvm::SmallVector<Value, 4> evaluated;
    evaluated.reserve(printArgs->size());
    for (const parser::NodePtr &argument : *printArgs)
      evaluated.push_back(emitExpr(argument.get()));
    bool allConverted = true;
    std::size_t unconverted = 0;
    Value joined;
    for (auto [index, value] : llvm::enumerate(evaluated)) {
      std::optional<Value> piece = emitStringifyValue(expr, value);
      if (!piece) {
        allConverted = false;
        unconverted = index;
        break;
      }
      if (index == 0) {
        joined = *piece;
        continue;
      }
      Value separator;
      if (separatorNode) {
        separator = coerceValue(emitExpr(separatorNode), strType, expr);
      } else {
        mlir::Type separatorType = types.literal("\" \"");
        separator = Value{py::StrConstantOp::create(
                              builder, loc(expr), separatorType,
                              builder.getStringAttr(" "))
                              .getResult(),
                          separatorType};
      }
      joined = emitBinarySpecial<py::AddOp>(expr, "__add__", joined, separator,
                                            strType);
      joined = emitBinarySpecial<py::AddOp>(expr, "__add__", joined, *piece,
                                            strType);
    }
    if (allConverted) {
      if (endNode)
        return writeWithEnd(joined);
      Value printCallee = emitExpr(calleeNode);
      CallOperands operands =
          emitCallOperands(expr, {joined}, /*includeAstArguments=*/false);
      return emitCallableDispatch(expr, printCallee, operands);
    }
    // ⭐ With ONE argument, falling through is right: the lowering reports
    // what is wrong with that argument (an unnarrowed union says so by
    // name). With more, the fall-through lands on the manifest print, whose
    // arity is one, and the report became "builtin callable 'print' expects
    // exactly one positional argument" -- the argument count is not the
    // problem, and the count is the only thing that message mentions.
    if (evaluated.size() > 1) {
      std::string described;
      llvm::raw_string_ostream stream(described);
      stream << evaluated[unconverted].type;
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, expr.range.start,
          "print() cannot render argument " + std::to_string(unconverted + 1) +
              " of type " + described +
              ": it has no __str__ or __repr__ this dispatch can resolve"});
      return emitNone(expr);
    }
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
      programBindsName(reducer))
    return std::nullopt;
  const auto *reducerArgs = ast::nodeList(expr, "args");
  const auto *reducerKeywords = ast::nodeList(expr, "keywords");
  // The element type of the reducer's iterable: genexpr arguments infer
  // their element expression under progressively bound chain targets
  // (like emitComprehension); plain iterables go through __iter__/__next__.
  // The element type of the reducer's iterable, from the shared walk the
  // inference uses for the same question.
  auto reducerElementType = [&]() -> mlir::Type {
    return types.iterationElementType(reducerArgs->front().get());
  };
  // Two-scalar form `min(a, b)` / `max(a, b)`: evaluate both operands
  // once, compare, and merge through the same cf-block pattern as IfExp
  // (`min(a, b)` keeps `a` on ties, matching CPython's first-minimal
  // rule). The non-selected operand's edge gets its release from the
  // partial-forward placement.
  // ⭐ More than two operands fold left, the way CPython's builtin_max walks
  // its argument tuple: `max(a, b, c)` is `max(max(a, b), c)`.
  //
  // Without this the call fell out of the reducer path entirely and the name
  // was reported UNRESOLVED -- "unresolved name 'max'" for a builtin that
  // works with two arguments, which points away from the actual limit. Written
  // as a rewrite onto the two-operand fold rather than as a third arm, so
  // there is one comparison lowering however many operands arrive.
  if (reducerArgs && reducerArgs->size() > 2 &&
      (!reducerKeywords || reducerKeywords->empty()) &&
      (reducer == "max" || reducer == "min")) {
    const parser::Field *calleeField = parser::findField(expr, "func");
    parser::NodePtr calleeShared =
        calleeField && std::holds_alternative<parser::NodePtr>(calleeField->value)
            ? std::get<parser::NodePtr>(calleeField->value)
            : nullptr;
    bool allPresent = calleeShared != nullptr;
    for (const parser::NodePtr &argument : *reducerArgs)
      allPresent = allPresent && argument && argument->kind != "Starred";
    if (allPresent) {
      parser::NodePtr folded = reducerArgs->front();
      for (std::size_t index = 1; index < reducerArgs->size(); ++index) {
        parser::NodePtr pair = synth::call(calleeShared, std::vector<parser::NodePtr>{
                             folded, (*reducerArgs)[index]}, expr.range);
        folded = std::move(pair);
      }
      return emitExpr(folded.get());
    }
  }
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
  // ⭐ `key=` rides the SAME loop: one more carried accumulator holding the
  // best key, compared in place of the element. CPython's builtin_max does
  // exactly that (keyfunc applied once per item, the item kept). It was
  // refused for an argument shape the fold could take with one extra slot.
  //
  // ⭐ AND `default=` IS THE EMPTY GUARD'S OTHER ANSWER. The fold already emits
  // `if not seen: raise ValueError(...)`; with a default that branch assigns
  // instead of raising, which is all CPython's builtin_max does with it. It was
  // refused as "max() with the 'default' keyword argument is not supported",
  // for a shape one arm of an `if` away.
  const parser::Node *reducerKeyNode = nullptr;
  parser::NodePtr reducerKeyValue;
  parser::NodePtr reducerDefaultValue;
  if ((reducer == "max" || reducer == "min") && reducerKeywords)
    for (const parser::NodePtr &entry : *reducerKeywords) {
      if (!entry)
        continue;
      llvm::StringRef name = ast::string(*entry, "arg").value_or("");
      const parser::Field *field = parser::findField(*entry, "value");
      if (!field || !std::holds_alternative<parser::NodePtr>(field->value))
        continue;
      const parser::NodePtr &value = std::get<parser::NodePtr>(field->value);
      if (name == "key") {
        reducerKeyValue = value;
        reducerKeyNode = value.get();
      } else if (name == "default") {
        reducerDefaultValue = value;
      }
    }
  bool reducerKeywordsUnderstood =
      !reducerKeywords || reducerKeywords->empty() ||
      (reducerKeywords->size() ==
       static_cast<std::size_t>(reducerKeyNode ? 1 : 0) +
           static_cast<std::size_t>(reducerDefaultValue ? 1 : 0));
  if (reducerArgs && reducerArgs->size() == 1 && reducerArgs->front() &&
      reducerKeywordsUnderstood && (reducer == "max" || reducer == "min")) {
    mlir::Type elementType = reducerElementType();
    // The accumulator needs a value of the element type before the first
    // trip; the seen-flag is what keeps it from ever being READ, so only its
    // TYPE has to be right.
    //
    // ⭐ A tuple gets one member-wise: a uniform `tuple[T]` has lost its
    // arity, but a one-element `(t,)` has exactly that type, and the arity
    // never matters because the value is unread. `max(rows)` over
    // `list[tuple[str, int]]` was refused for a comparison the tuple contract
    // already implements -- sorted() orders the same rows.
    std::function<mlir::Value(mlir::Type)> placeholderFor =
        [&](mlir::Type type) -> mlir::Value {
      auto contractType = mlir::dyn_cast_if_present<py::ContractType>(type);
      if (!contractType)
        return {};
      llvm::StringRef name = contractType.getContractName();
      if (name == "builtins.int")
        return py::IntConstantOp::create(builder, loc(expr),
                                         types.literal("0"),
                                         builder.getStringAttr("0"))
            .getResult();
      if (name == "builtins.str")
        return py::StrConstantOp::create(builder, loc(expr),
                                         types.literal("\"\""),
                                         builder.getStringAttr(""))
            .getResult();
      if (name == "builtins.float")
        return py::FloatConstantOp::create(builder, loc(expr), type,
                                           builder.getF64FloatAttr(0.0))
            .getResult();
      if (name == "builtins.bool")
        return py::BoolConstantOp::create(builder, loc(expr),
                                          types.literal("False"),
                                          builder.getBoolAttr(false))
            .getResult();
      if (name != "builtins.tuple" || contractType.getArguments().empty())
        return {};
      llvm::SmallVector<mlir::Value, 4> members;
      for (mlir::Type member : contractType.getArguments()) {
        mlir::Value part = placeholderFor(member);
        if (!part)
          return {};
        members.push_back(part);
      }
      return py::PackOp::create(builder, loc(expr), type, members).getResult();
    };
    mlir::Value placeholder = placeholderFor(elementType);
    // ⭐ A NON-PRIMITIVE ELEMENT IS FOLDED BY INDEX, NOT BY ACCUMULATOR.
    // `max(rows, key=lambda r: r.score)` over a list of INSTANCES -- what
    // every "pick the best record" line looks like -- was refused, because the
    // accumulator needs a value of the element type before the first trip and
    // `placeholderFor` can only fabricate one for int/str/float/bool and
    // tuples of those. `sorted(rows, key=...)` beside it has always worked.
    //
    // Carrying the best ELEMENT instead was built first and does not work: the
    // frame then owns the element twice, once from the seed read and once from
    // the loop's reassignment ("ly.ownership.owned_local_object marks a value
    // this frame already owns"). So nothing but INTS crosses the loop edge --
    // the best index and the best key -- and the element is read once, after
    // the loop:
    //
    //     __src = <arg>
    //     if len(__src) == 0: raise ValueError("max() iterable argument is empty")
    //     __bi = 0
    //     __bk = key(__src[0])
    //     for __i in range(1, len(__src)):
    //         __ck = key(__src[__i])
    //         if __ck > __bk:          # Lt for min
    //             __bi = __i
    //             __bk = __ck
    //     __src[__bi]
    //
    // Strict `>` keeps CPython's tie rule: the FIRST maximal element wins.
    //
    // ⭐ WITHOUT A KEY THE COMPARISON IS SPELLED IN PLACE and no second name
    // is carried: `__src[__i] > __src[__bi]`. Carrying the best ELEMENT is
    // what the note above rules out, and a key-less fold does not need to --
    // only the index crosses the loop edge either way, and the two elements
    // are read as temporaries inside one trip.
    //
    // `max(versions)` over a list of instances that order themselves was
    // refused for a seed it does not need, and the message offered "or an
    // indexable argument to take the first element from" -- which is what a
    // list is. `sorted(versions)` beside it has always worked. A class with
    // no ordering now gets the diagnostic that says so, from the comparison
    // itself, instead of one about seeding.
    // ⛔ And an INDEXABLE argument only: a generator has no first element to
    // read without consuming it.
    // ⛔ An EMPTY literal is left to the branch below, which raises the
    // ValueError directly: this path would emit `key(__src[0])` against an
    // element type that does not exist, and `max([], key=len)` came back as
    // "builtins.object does not provide manifest method '__len__'".
    bool emptyLiteralArgument =
        (reducerArgs->front()->kind == "List" ||
         reducerArgs->front()->kind == "Tuple") &&
        [&] {
          const auto *elts = ast::nodeList(*reducerArgs->front(), "elts");
          return !elts || elts->empty();
        }();
    if (!placeholder && !reducerDefaultValue && elementType &&
        elementType != types.object() && !emptyLiteralArgument) {
      mlir::Type argType =
          types.widenLiteral(types.inferExpr(reducerArgs->front().get()));
      auto argContract = mlir::dyn_cast_if_present<py::ContractType>(argType);
      llvm::StringRef argName =
          argContract ? argContract.getContractName() : llvm::StringRef();
      // ⛔ A LIST, not a tuple. Reading `t[0]` and then `t[i]` out of a tuple
      // the frame owns is refused as a second retain of one entity ("a re-read
      // of an entity the frame owns is a borrow: reuse the existing token"),
      // which is a rule about tuple elements and not about this fold. A tuple
      // argument keeps the seed refusal.
      if (argName == "builtins.list") {
        ++listCompCounter;
        std::string counter = std::to_string(listCompCounter);
        std::string src = "__" + reducer.str() + "src" + counter;
        std::string bestIndex = "__" + reducer.str() + "bi" + counter;
        std::string bestKey = "__" + reducer.str() + "bk" + counter;
        std::string index = "__" + reducer.str() + "i" + counter;
        std::string currentKey = "__" + reducer.str() + "ck" + counter;
        auto nameOf = [&](const std::string &id) {
          return synth::name(id, expr.range);
        };
        auto elementAt = [&](parser::NodePtr position) {
          return synth::subscript(nameOf(src), std::move(position), expr.range);
        };
        auto keyOf = [&](parser::NodePtr element) {
          return synth::call(reducerKeyValue,
                             std::vector<parser::NodePtr>{std::move(element)},
                             expr.range);
        };
        emitStatement(
            *synth::assign(nameOf(src), reducerArgs->front(), expr.range));
        emitStatement(*synth::ifStmt(
            synth::compare(synth::lenCall(nameOf(src), expr.range), "Eq",
                           synth::intConstant(0, expr.range), expr.range),
            {synth::raiseCall("ValueError",
                              reducer.str() + "() iterable argument is empty",
                              expr.range)},
            {}, expr.range));
        emitStatement(*synth::assign(
            nameOf(bestIndex), synth::intConstant(0, expr.range), expr.range));
        llvm::StringRef comparison = reducer == "max" ? "Gt" : "Lt";
        std::vector<parser::NodePtr> body;
        if (reducerKeyNode) {
          emitStatement(*synth::assign(
              nameOf(bestKey),
              keyOf(elementAt(synth::intConstant(0, expr.range))), expr.range));
          std::vector<parser::NodePtr> better{
              synth::assign(nameOf(bestIndex), nameOf(index), expr.range),
              synth::assign(nameOf(bestKey), nameOf(currentKey), expr.range)};
          body.push_back(synth::assign(
              nameOf(currentKey), keyOf(elementAt(nameOf(index))), expr.range));
          body.push_back(synth::ifStmt(
              synth::compare(nameOf(currentKey), comparison, nameOf(bestKey),
                             expr.range),
              std::move(better), {}, expr.range));
        } else {
          std::vector<parser::NodePtr> better{
              synth::assign(nameOf(bestIndex), nameOf(index), expr.range)};
          body.push_back(synth::ifStmt(
              synth::compare(elementAt(nameOf(index)), comparison,
                             elementAt(nameOf(bestIndex)), expr.range),
              std::move(better), {}, expr.range));
        }
        parser::NodePtr span = synth::call(
            nameOf("range"),
            std::vector<parser::NodePtr>{
                synth::intConstant(1, expr.range),
                synth::lenCall(nameOf(src), expr.range)},
            expr.range);
        emitStatement(*synth::forStmt(nameOf(index), std::move(span),
                                      std::move(body), {}, expr.range));
        Value result = emitExpr(elementAt(nameOf(bestIndex)).get());
        for (const std::string &name :
             {src, bestIndex, bestKey, index, currentKey})
          values.erase(name);
        return result;
      }
    }
    // The key accumulator needs its own placeholder, on the same terms.
    mlir::Value keyPlaceholder;
    if (placeholder && reducerKeyNode) {
      mlir::Type keyType;
      // ⭐ A LAMBDA key is asked directly, against the element type. The probe
      // below infers a synthesized `key(__lykeyprobe)` call, and for a lambda
      // that call carries no expectation into the body: the parameter is
      // unannotated, so it types as `object` and so does the result. Both
      // `max(xs, key=lambda p: p[1])` and `max(xs, key=lambda v: -v)` were
      // refused with "needs a key the fold can seed ... this one produces
      // builtins.object", while the same key spelled as a `def` and
      // `sorted(xs, key=lambda ...)` worked -- so the refusal was about the
      // spelling, not the fold.
      //
      // Same shape as `map`'s lambda arm in TypeSystem.cpp, and the same
      // reason: a lambda's parameter type comes from the sequence, which is
      // the expectation the emitter already distributes when it inlines the
      // body.
      if (reducerKeyNode->kind == "Lambda") {
        py::CallableType expected =
            py::CallableType::get(&context, {elementType}, {}, {}, {}, {});
        keyType = types.widenLiteral(
            types.functionSignature(*reducerKeyNode, std::nullopt, expected)
                .resultType);
      } else {
        auto scope = types.pushScope();
        types.bindLocalSymbol("__lykeyprobe", elementType);
        parser::NodePtr probeArg = synth::name(std::string("__lykeyprobe"), expr.range);
        parser::NodePtr probe = synth::call(
            reducerKeyValue, std::vector<parser::NodePtr>{std::move(probeArg)},
            expr.range);
        keyType = types.widenLiteral(types.inferExpr(probe.get()));
      }
      keyPlaceholder = placeholderFor(keyType);
      if (!keyPlaceholder) {
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, expr.range.start,
            keyType ? reducer.str() +
                          "() needs a key the fold can seed (int, str, float, "
                          "bool, or a tuple of those); this one produces " +
                          [&] {
                            std::string text;
                            llvm::raw_string_ostream stream(text);
                            stream << keyType;
                            return text;
                          }()
                    : reducer.str() +
                          "() cannot see what its key returns; give the key "
                          "function a return annotation"});
        return emitNone(expr);
      }
    }
    if (!placeholder && !reducerDefaultValue) {
      // max()/min() over an EMPTY literal always raises: emit the
      // ValueError directly (there is no element type to desugar with).
      const parser::Node *arg = reducerArgs->front().get();
      bool emptyLiteral =
          (arg->kind == "List" || arg->kind == "Tuple") &&
          [&] {
            const auto *elts = ast::nodeList(*arg, "elts");
            return !elts || elts->empty();
          }();
      if (emptyLiteral && reducerDefaultValue)
        return emitExpr(reducerDefaultValue.get());
      if (emptyLiteral) {
        parser::NodePtr errorName = synth::name(std::string("ValueError"), expr.range);
        parser::NodePtr message = parser::makeNode("Constant", expr.range);
        parser::addField(*message, "value",
                         reducer.str() + "() iterable argument is empty");
        parser::NodePtr errorCall = synth::call(errorName, std::vector<parser::NodePtr>{message}, expr.range);
        parser::NodePtr raiseNode = synth::raiseStmt(errorCall, expr.range);
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
              "() needs an element type the fold can seed (int, str, float, "
              "bool, or a tuple of those), or an indexable argument to take "
              "the first element from"});
      return emitNone(expr);
    }
    std::string tmp =
        "__" + reducer.str() + std::to_string(++listCompCounter);
    std::string flag =
        "__" + reducer.str() + "seen" + std::to_string(listCompCounter);
    std::string element =
        "__" + reducer.str() + "el" + std::to_string(listCompCounter);
    // ⭐ WITH A DEFAULT THE SEED *IS* THE ANSWER FOR AN EMPTY ITERABLE, so the
    // accumulator starts there and the empty guard disappears instead of
    // growing a second arm. Seeding the placeholder and assigning the default
    // afterwards compiled but leaked: the placeholder is a fabricated value the
    // first element overwrites, and it was only ever unread because the empty
    // path RAISED. Give it a path that returns and the fabrication reaches the
    // exit ("owned resource ... reaches function exit without release").
    if (reducerDefaultValue) {
      Value seed = emitExpr(reducerDefaultValue.get());
      values[tmp] = seed;
      types.bindSymbol(tmp, seed.type);
    } else {
      values[tmp] = Value{placeholder, placeholder.getType()};
      types.bindSymbol(tmp, placeholder.getType());
    }
    // The seen-flag is an INT (0/1): loop-carried bool contract block
    // arguments have no boxed physical form yet.
    mlir::Type flagType = types.literal("0");
    auto flagInit = py::IntConstantOp::create(builder, loc(expr), flagType,
                                              builder.getStringAttr("0"));
    values[flag] = Value{flagInit.getResult(), flagType};
    types.bindSymbol(flag, flagType);
    auto nameNode = [&](const std::string &id) {
      parser::NodePtr node = synth::name(id, expr.range);
      return node;
    };
    std::string keyAcc = "__" + reducer.str() + "key" +
                         std::to_string(listCompCounter);
    std::string keyOfElement =
        "__" + reducer.str() + "k" + std::to_string(listCompCounter);
    if (keyPlaceholder) {
      values[keyAcc] = Value{keyPlaceholder, keyPlaceholder.getType()};
      types.bindSymbol(keyAcc, keyPlaceholder.getType());
    }
    parser::NodePtr tmpName = nameNode(tmp);
    parser::NodePtr flagName = nameNode(flag);
    parser::NodePtr elementName = nameNode(element);
    parser::NodePtr keyAccName = nameNode(keyAcc);
    parser::NodePtr keyName = nameNode(keyOfElement);
    // if __seen: (if el >/< __acc: __acc = el) else: __acc = el; __seen = True
    // With a key the compared operands are the KEYS and both accumulators
    // move together; the key is computed once per element, above the switch.
    parser::NodePtr assignAcc = synth::assign(tmpName, elementName, expr.range);
    parser::NodePtr assignKeyAcc;
    if (reducerKeyNode) {
      assignKeyAcc = parser::makeNode("Assign", expr.range);
      parser::addField(*assignKeyAcc, "targets",
                       std::vector<parser::NodePtr>{keyAccName});
      parser::addField(*assignKeyAcc, "value", keyName);
    }
    parser::NodePtr cmpOp = parser::makeNode(
        reducer == "max" ? "Gt" : "Lt", expr.range);
    parser::NodePtr compare = parser::makeNode("Compare", expr.range);
    parser::addField(*compare, "left",
                     reducerKeyNode ? keyName : elementName);
    parser::addField(*compare, "ops", std::vector<parser::NodePtr>{cmpOp});
    parser::addField(*compare, "comparators",
                     std::vector<parser::NodePtr>{
                         reducerKeyNode ? keyAccName : tmpName});
    std::vector<parser::NodePtr> betterBody{assignAcc};
    if (assignKeyAcc)
      betterBody.push_back(assignKeyAcc);
    parser::NodePtr better = parser::makeNode("If", expr.range);
    parser::addField(*better, "test", compare);
    parser::addField(*better, "body", std::move(betterBody));
    parser::addField(*better, "orelse", std::vector<parser::NodePtr>{});
    parser::NodePtr trueValue = synth::intConstant(std::int64_t{1}, expr.range);
    parser::NodePtr markSeen = synth::assign(flagName, trueValue, expr.range);
    // The seen-flag is an int, and int truthiness is rejected (R1): the
    // synthesized tests spell the comparison out.
    auto flagCompare = [&](const char *opKind) {
      parser::NodePtr zero = synth::intConstant(std::int64_t{0}, expr.range);
      parser::NodePtr op = parser::makeNode(opKind, expr.range);
      parser::NodePtr compare = parser::makeNode("Compare", expr.range);
      parser::addField(*compare, "left", flagName);
      parser::addField(*compare, "ops", std::vector<parser::NodePtr>{op});
      parser::addField(*compare, "comparators",
                       std::vector<parser::NodePtr>{zero});
      return compare;
    };
    std::vector<parser::NodePtr> firstBody{assignAcc};
    if (assignKeyAcc)
      firstBody.push_back(assignKeyAcc);
    firstBody.push_back(markSeen);
    parser::NodePtr seenSwitch = parser::makeNode("If", expr.range);
    parser::addField(*seenSwitch, "test", flagCompare("NotEq"));
    parser::addField(*seenSwitch, "body",
                     std::vector<parser::NodePtr>{better});
    parser::addField(*seenSwitch, "orelse", std::move(firstBody));
    std::vector<parser::NodePtr> loopBody;
    if (reducerKeyNode) {
      parser::NodePtr keyCall = synth::call(
          reducerKeyValue, std::vector<parser::NodePtr>{elementName},
          expr.range);
      parser::NodePtr bindKey = synth::assign(keyName, std::move(keyCall), expr.range);
      loopBody.push_back(std::move(bindKey));
    }
    loopBody.push_back(seenSwitch);
    parser::NodePtr loop = parser::makeNode("For", expr.range);
    parser::addField(*loop, "target", elementName);
    parser::addField(*loop, "iter", reducerArgs->front());
    parser::addField(*loop, "body", std::move(loopBody));
    parser::addField(*loop, "orelse", std::vector<parser::NodePtr>{});
    // if __seen == 0: raise ValueError("max()/min() iterable argument is
    // empty")
    parser::NodePtr emptyGuard;
    if (!reducerDefaultValue) {
      parser::NodePtr notSeen = flagCompare("Eq");
      parser::NodePtr message = parser::makeNode("Constant", expr.range);
      parser::addField(*message, "value",
                       reducer.str() + "() iterable argument is empty");
      parser::NodePtr errorCall = synth::call(
          nameNode("ValueError"), std::vector<parser::NodePtr>{message},
          expr.range);
      parser::NodePtr raiseNode = synth::raiseStmt(errorCall, expr.range);
      emptyGuard = parser::makeNode("If", expr.range);
      parser::addField(*emptyGuard, "test", notSeen);
      parser::addField(*emptyGuard, "body",
                       std::vector<parser::NodePtr>{raiseNode});
      parser::addField(*emptyGuard, "orelse",
                       std::vector<parser::NodePtr>{});
    }
    std::optional<Value> priorElement;
    if (auto found = values.find(element); found != values.end())
      priorElement = found->second;
    emitFor(*loop);
    if (emptyGuard)
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
    values.erase(keyAcc);
    values.erase(keyOfElement);
    if (priorElement)
      values[element] = *priorElement;
    else
      values.erase(element);
    return result;
  }
  // ⭐ `sum(xs, start)` seeds the accumulator with `start` instead of 0, which
  // is all CPython's builtin_sum does with its second argument.
  //
  // Without it the two-argument call left the reducer path and the name came
  // back UNRESOLVED -- "unresolved name 'sum'" for a builtin whose
  // one-argument form works, a message about the wrong thing entirely.
  bool sumHasStart = reducer == "sum" && reducerArgs &&
                     reducerArgs->size() == 2 && (*reducerArgs)[1] &&
                     (*reducerArgs)[1]->kind != "Starred" &&
                     (!reducerKeywords || reducerKeywords->empty());
  if (reducerArgs && reducerArgs->front() &&
      (reducerArgs->size() == 1 || sumHasStart) &&
      (!reducerKeywords || reducerKeywords->empty()) &&
      (reducer == "sum" || reducer == "any" || reducer == "all")) {
    std::string tmp =
        "__" + reducer.str() + std::to_string(++listCompCounter);
    std::string element = "__" + reducer.str() + "el" +
                          std::to_string(listCompCounter);
    if (reducer == "sum") {
      if (sumHasStart) {
        Value seed = emitExpr((*reducerArgs)[1].get());
        values[tmp] = seed;
        types.bindSymbol(tmp, seed.type);
      } else if (reducerElementType() == types.floatType()) {
        // ⭐ CPython's implicit start is the int 0 and `0 + 1.5` promotes,
        // but the accumulator here is one SSA value with one type: seeding a
        // float sum with the int zero asked the lowering to store a float
        // into an int lane -- "cannot adapt runtime bundle builtins.float
        // with physical values (memref<3xi64>) to expected ABI". Seeding the
        // promoted zero is the same answer for every non-empty iterable
        // (sum([1.5, 2.5]) is 4.0 either way).
        //
        // ⛔ AND IT IS NOT THE SAME FOR AN EMPTY ONE: `sum(xs)` over an empty
        // list[float] prints 0.0 here and 0 in CPython, because CPython's
        // accumulator is still the int start it never added to. Matching that
        // needs the accumulator to be int-or-float chosen at run time, which
        // is one SSA value with two types -- the union construction this
        // compiler does not build. Measured 2026-08-21; the non-empty answers
        // all agree, compensation included.
        auto zero = py::FloatConstantOp::create(builder, loc(expr),
                                                types.floatType(),
                                                builder.getF64FloatAttr(0.0));
        values[tmp] = Value{zero.getResult(), types.floatType()};
        types.bindSymbol(tmp, types.floatType());
      } else {
      mlir::Type zeroType = types.literal("0");
      auto zero = py::IntConstantOp::create(builder, loc(expr), zeroType,
                                            builder.getStringAttr("0"));
      values[tmp] = Value{zero.getResult(), zeroType};
      types.bindSymbol(tmp, zeroType);
      }
    } else {
      bool initial = reducer == "all";
      mlir::Type initType = types.literal(initial ? "True" : "False");
      auto init = py::BoolConstantOp::create(
          builder, loc(expr), initType, builder.getBoolAttr(initial));
      values[tmp] = Value{init.getResult(), initType};
      types.bindSymbol(tmp, initType);
    }
    parser::NodePtr tmpName = synth::name(tmp, expr.range);
    parser::NodePtr elementName = synth::name(element, expr.range);
    std::vector<parser::NodePtr> body;
    // ⭐ A FLOAT SUM IS COMPENSATED, because CPython's has been since 3.12:
    // builtin_sum carries a Neumaier correction term and adds it back at the
    // end. Without it this answered `sum([0.1, 0.2, 0.3])` as
    // 0.6000000000000001 where CPython says 0.6, and -- the case that shows it
    // is not a rounding quibble -- `sum([1e100, 1.0, -1e100])` as 0.0 where
    // CPython says 1.0. The naive fold loses the small term entirely.
    //
    // ⛔ Written as the same synthesized Python the rest of the fold is, so the
    // arithmetic is the compiler's own float ops rather than a second
    // implementation: t = acc + x; c += (acc - t) + x when |acc| >= |x|, else
    // c += (x - t) + acc; acc = t; and acc + c at the end.
    std::string compensation = "__" + reducer.str() + "comp" +
                               std::to_string(listCompCounter);
    std::string partial =
        "__" + reducer.str() + "part" + std::to_string(listCompCounter);
    bool compensated =
        reducer == "sum" &&
        types.widenLiteral(values[tmp].type) == types.floatType();
    if (compensated) {
      auto zero = py::FloatConstantOp::create(builder, loc(expr),
                                              types.floatType(),
                                              builder.getF64FloatAttr(0.0));
      values[compensation] = Value{zero.getResult(), types.floatType()};
      types.bindSymbol(compensation, types.floatType());
    }
    if (compensated) {
      parser::NodePtr compName = synth::name(compensation, expr.range);
      parser::NodePtr partialName = synth::name(partial, expr.range);
      auto absOf = [&](parser::NodePtr value) {
        return synth::call(synth::name("abs", expr.range),
                           std::vector<parser::NodePtr>{std::move(value)},
                           expr.range);
      };
      // <partial> = <tmp> + <element>
      body.push_back(synth::assign(
          partialName,
          synth::binOp(tmpName, "Add", elementName, expr.range), expr.range));
      auto correction = [&](parser::NodePtr large, parser::NodePtr small) {
        return synth::binOp(
            compName, "Add",
            synth::binOp(synth::binOp(std::move(large), "Sub", partialName,
                                      expr.range),
                         "Add", std::move(small), expr.range),
            expr.range);
      };
      body.push_back(synth::ifStmt(
          synth::compare(absOf(tmpName), "GtE", absOf(elementName), expr.range),
          {synth::assign(compName, correction(tmpName, elementName),
                         expr.range)},
          {synth::assign(compName, correction(elementName, tmpName),
                         expr.range)},
          expr.range));
      body.push_back(synth::assign(tmpName, partialName, expr.range));
    } else if (reducer == "sum") {
      // <tmp> = <tmp> + <element>
      parser::NodePtr addOp = parser::makeNode("Add", expr.range);
      parser::NodePtr add = parser::makeNode("BinOp", expr.range);
      parser::addField(*add, "left", tmpName);
      parser::addField(*add, "op", addOp);
      parser::addField(*add, "right", elementName);
      parser::NodePtr assign = synth::assign(tmpName, add, expr.range);
      body.push_back(assign);
    } else {
      // any: if <element>: <tmp> = True; break
      // all: if not <element>: <tmp> = False; break
      bool flipped = reducer == "any";
      parser::NodePtr flippedValue =
          parser::makeNode("Constant", expr.range);
      parser::addField(*flippedValue, "value", flipped);
      parser::NodePtr assign = synth::assign(tmpName, flippedValue, expr.range);
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
    if (compensated)
      emitStatement(*synth::assign(
          synth::name(tmp, expr.range),
          synth::binOp(synth::name(tmp, expr.range), "Add",
                       synth::name(compensation, expr.range), expr.range),
          expr.range));
    auto built = values.find(tmp);
    if (built == values.end() || !built->second.value) {
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, expr.range.start,
          "cannot lower " + reducer.str() + "() over this iterable"});
      return emitNone(expr);
    }
    Value result = built->second;
    values.erase(tmp);
    values.erase(compensation);
    values.erase(partial);
    if (priorElement)
      values[element] = *priorElement;
    else
      values.erase(element);
    return result;
  }
  // ⭐ Every shape this fold does not take is refused HERE, naming the
  // reducer. Falling through left the generic call path to look the name up,
  // and it is not a value -- `min(xs, key=f)` was "unresolved name 'min'",
  // which points away from the actual limit (the keyword, not the name).
  // The same wrong report is recorded twice above for the argument COUNT;
  // this is that report closed for the argument SHAPE.
  std::string limit;
  if (reducerKeywords && !reducerKeywords->empty()) {
    llvm::StringRef keyword;
    for (const parser::NodePtr &entry : *reducerKeywords)
      if (entry && keyword.empty())
        if (std::optional<llvm::StringRef> name = ast::string(*entry, "arg"))
          keyword = *name;
    limit = keyword.empty() ? "a keyword argument"
                            : ("the '" + keyword.str() + "' keyword argument");
  } else if (!reducerArgs || reducerArgs->empty()) {
    limit = "no arguments";
  } else {
    for (const parser::NodePtr &argument : *reducerArgs)
      if (argument && argument->kind == "Starred")
        limit = "a starred argument";
  }
  if (!limit.empty()) {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, expr.range.start,
        reducer.str() + "() with " + limit +
            " is not supported: it is folded over an iterable or over two or "
            "more operands, and neither form takes it"});
    return emitNone(expr);
  }
  return std::nullopt;
}

std::optional<Value>
ModuleEmitter::tryEmitLenCall(const parser::Node &expr,
                              const parser::Node *calleeNode) {
  if (!callsUnshadowedBuiltin(calleeNode, "len"))
    return std::nullopt;
  static constexpr llvm::StringRef kParameters[] = {"obj"};
  std::optional<llvm::SmallVector<const parser::Node *, 4>> bound =
      bindBuiltinArguments(expr, kParameters, /*positionalOnly=*/1);
  if (bound && bound->size() == 1) {
    // len(d.keys()/values()/items()) measures the dict itself — the views
    // have no runtime object.
    const parser::Node *argNode = bound->front();
    if (argNode && argNode->kind == "Call") {
      const parser::Node *viewCallee = ast::node(*argNode, "func");
      const auto *viewArgs = ast::nodeList(*argNode, "args");
      const auto *viewKeywords = ast::nodeList(*argNode, "keywords");
      if (viewCallee && viewCallee->kind == "Attribute" &&
          (!viewArgs || viewArgs->empty()) &&
          (!viewKeywords || viewKeywords->empty())) {
        auto viewName = ast::string(*viewCallee, "attr");
        const parser::Node *viewReceiver = ast::node(*viewCallee, "value");
        if (viewName &&
            (*viewName == "keys" || *viewName == "values" ||
             *viewName == "items") &&
            isDictTypedExpr(viewReceiver))
          argNode = viewReceiver;
      }
    }
    Value input = emitExpr(argNode);
  if (std::optional<Value> count = tryEmitClassDunder(expr, input, "__len__"))
    return *count;
  // ⭐ A UNION COUNTS BY TAG, like it adds by tag. `def f(x: "list[int] | str")`
  // is an ordinary Python signature and `len(x)` in its body was refused for a
  // question both members answer.
  if (auto inputUnion =
          mlir::dyn_cast_if_present<py::UnionType>(types.widenLiteral(input.type)))
    if (mlir::Type joined = types.unionOperatorResult(input.type, "__len__", {}))
      return emitUnionMemberDispatch(
          expr, input, inputUnion, joined, [&](Value member) {
            CallInferenceResult memberInference =
                types.inferMethodCallWithEvidence(member.type, "__len__", {});
            mlir::Type memberResult =
                memberInference ? memberInference.resultType : joined;
            auto memberOp = py::LenOp::create(
                builder, loc(expr), memberResult,
                mlir::FlatSymbolRefAttr::get(&context, "__len__"),
                callProtocolFor(memberInference), member.value);
            return Value{memberOp.getResult(), memberResult};
          });
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
  if (!callsUnshadowedBuiltin(calleeNode, "next"))
    return std::nullopt;
  const auto *args = ast::nodeList(expr, "args");
  const auto *keywords = ast::nodeList(expr, "keywords");
  if (keywords && !keywords->empty())
    return std::nullopt;
  // `next(it, default)` desugars to the pre-bound try/except form
  //   __nx = default; try: __nx = next(__it) except StopIteration: pass
  // (a binding CREATED inside a try does not escape the handler scope, but
  // rebinding a pre-existing local does; the iterator is snapshot first to
  // keep CPython's left-to-right argument evaluation).
  if (args && args->size() == 2 && args->front() && (*args)[1]) {
    unsigned serial = ++listCompCounter;
    std::string iteratorName = "__lynextit" + std::to_string(serial);
    std::string resultName = "__lynext" + std::to_string(serial);
    parser::SourceRange range = expr.range;
    auto nameNode = [&](const std::string &id) {
      parser::NodePtr node = synth::name(id, range);
      return node;
    };
    auto assign = [&](parser::NodePtr target, parser::NodePtr value) {
      parser::NodePtr node = synth::assign(std::move(target), std::move(value), range);
      return node;
    };
    parser::NodePtr nextCall = synth::call(nameNode("next"), std::vector<parser::NodePtr>{nameNode(iteratorName)}, range);
    parser::NodePtr handler = parser::makeNode("ExceptHandler", range);
    parser::addField(*handler, "type", nameNode("StopIteration"));
    parser::addField(*handler, "body",
                     std::vector<parser::NodePtr>{
                         parser::makeNode("Pass", range)});
    parser::NodePtr tryNode = parser::makeNode("Try", range);
    parser::addField(*tryNode, "body",
                     std::vector<parser::NodePtr>{
                         assign(nameNode(resultName), nextCall)});
    parser::addField(*tryNode, "handlers",
                     std::vector<parser::NodePtr>{handler});
    parser::addField(*tryNode, "orelse", std::vector<parser::NodePtr>{});
    parser::addField(*tryNode, "finalbody", std::vector<parser::NodePtr>{});

    llvm::SmallVector<std::pair<std::string, std::optional<Value>>, 2> priors;
    for (const std::string &scratch : {iteratorName, resultName}) {
      std::optional<Value> prior;
      if (auto found = values.find(scratch); found != values.end())
        prior = found->second;
      priors.push_back({scratch, prior});
    }
    emitStatement(*assign(nameNode(iteratorName), args->front()));
    // ⭐ THE SCRATCH IS PRE-BOUND AT THE JOIN, not at the default's own type.
    // `next(it, None)` bound it `literal<None>`, and the try then reassigned
    // it to the element -- which is the one shape the carry-out rule refuses,
    // reported about a name the program never wrote:
    //
    //     xs = iter([1])
    //     print(next(xs, None))
    //     # local '__lynext1' is reassigned inside this try and its type
    //     # !py.literal<None> cannot be carried out of the statement
    //
    // `next(it, 0)` worked because a widened int is the same contract the
    // element is. The join is what both spellings actually need, and for a
    // None default it is the Optional the rule accepts.
    //
    // ⛔ Only when the element type is known: an iterator whose `__next__` has
    // no static evidence keeps the plain binding, which is the diagnostic that
    // question deserves rather than one about a scratch name.
    mlir::Type joinedType;
    if (auto iterator = values.find(iteratorName); iterator != values.end())
      if (CallInferenceResult elementInference =
              types.inferMethodCallWithEvidence(iterator->second.type,
                                                "__next__", {}))
        if (mlir::Type defaultType =
                types.widenLiteral(types.inferExpr((*args)[1].get())))
          joinedType = types.join(
              {types.widenLiteral(elementInference.resultType), defaultType});
    if (auto joinedUnion =
            mlir::dyn_cast_if_present<py::UnionType>(joinedType)) {
      // ⛔ A WIDER union than an Optional has no slot to be carried out of the
      // try in, which is the same wall every other union-typed local meets.
      // Saying so HERE is the difference between a sentence about the program
      // and one about the compiler: the refusal is reported on `__lynext1`, a
      // name the program never wrote, and its advice ("bind the reassignment
      // to a new name") cannot be followed for a scratch the emitter owns.
      if (!joinedUnion.isOptional() ||
          !mlir::isa_and_nonnull<py::ContractType>(
              joinedUnion.getOptionalPayloadType())) {
        std::string text;
        llvm::raw_string_ostream stream(text);
        stream << "next(iterator, default) needs a default that fits the "
                  "iterator's element type or is None; these join at "
               << joinedType;
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, expr.range.start, text});
        return emitNone(expr);
      }
      Value defaultValue = coerceValue(
          emitExprExpected((*args)[1].get(), joinedType), joinedType, expr);
      values[resultName] = defaultValue;
      types.bindSymbol(resultName, joinedType);
    } else {
      emitStatement(*assign(nameNode(resultName), (*args)[1]));
    }
    emitStatement(*tryNode);
    auto bound = values.find(resultName);
    if (bound == values.end() || !bound->second.value) {
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, expr.range.start,
          "cannot lower next(iterator, default) over this iterator"});
      return emitNone(expr);
    }
    Value result = bound->second;
    for (auto &[scratch, prior] : priors) {
      if (prior)
        values[scratch] = *prior;
      else
        values.erase(scratch);
    }
    return result;
  }
  if (!args || args->size() != 1) {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, expr.range.start,
        "next() takes one iterator and an optional default"});
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
ModuleEmitter::tryEmitHashCall(const parser::Node &expr,
                               const parser::Node *calleeNode) {
  if (!callsUnshadowedBuiltin(calleeNode, "hash"))
    return std::nullopt;
  const auto *args = ast::nodeList(expr, "args");
  const auto *keywords = ast::nodeList(expr, "keywords");
  if (!args || args->size() != 1 || (keywords && !keywords->empty()))
    return std::nullopt;
  // The builtin's manifest __hash__ dispatch only sees manifest classes, so
  // a source class's own __hash__ dispatches inline here. The argument type
  // is probed without emitting: declining must leave no side effects for
  // the generic path to re-emit.
  mlir::Type argType = types.inferExpr(args->front().get());
  // CPython sets __hash__ to None on a class that defines __eq__ and not
  // __hash__ (every unfrozen dataclass included), so its instances are
  // unhashable. Refused rather than answered with object's identity hash:
  // two instances the class calls EQUAL would hash apart, and every hash
  // container built on them would then miss without a word.
  // Enum members are exempt: CPython's Enum defines no __eq__ of its own, so
  // members stay hashable; the __eq__ Lython synthesizes for an enum is a
  // lowering artifact, not the author's value equality.
  if (std::optional<std::string> unhashable = unhashableClassName(argType)) {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, expr.range.start,
        "'" + *unhashable +
            "' defines __eq__ without __hash__, so its instances are "
            "unhashable (CPython sets __hash__ to None for such a class, "
            "including every unfrozen dataclass); define __hash__ consistently "
            "with __eq__ to hash it"});
    return emitNone(expr);
  }
  if (!lookupClassMethod(argType, "__hash__"))
    return std::nullopt;
  Value input = emitExpr(args->front().get());
  std::optional<Value> dispatched = tryEmitClassDunder(expr, input, "__hash__");
  if (!dispatched)
    return std::nullopt;
  Value hashed = *dispatched;
  // ⭐ CPython never lets a hash be -1: that value is its "error" sentinel, so
  // `Py_hash_t` -1 is remapped to -2 on the way out of `__hash__`
  // (Objects/object.c, PyObject_Hash). A class whose `__hash__` returns -1
  // printed -1 here. Only the source-class path needs it -- the manifest
  // hashes already answer through the runtime, which does the remap.
  mlir::Type intType = types.intType();
  Value hashedInt = coerceValue(hashed, intType, expr);
  // ⛔ Done on the PRIMITIVE lane, not with two `builtins.int` constants and
  // an object select: those are owned values, both of them live from before
  // the comparison, and one is dead on either branch -- the affine-ownership
  // verifier refused `hash_abs_overload` for exactly that ("still owned when
  // a call to 'LyComplex_FromParts' may unwind"). An `arith.constant` has no
  // reference to account for.
  mlir::Type i64Type = builder.getI64Type();
  mlir::Value raw =
      py::CastToPrimOp::create(builder, loc(expr), i64Type, hashedInt.value,
                               builder.getStringAttr("exact"))
          .getResult();
  mlir::Value sentinel =
      mlir::arith::ConstantIntOp::create(builder, loc(expr), -1, 64)
          .getResult();
  mlir::Value replacement =
      mlir::arith::ConstantIntOp::create(builder, loc(expr), -2, 64)
          .getResult();
  mlir::Value isSentinel =
      mlir::arith::CmpIOp::create(builder, loc(expr),
                                  mlir::arith::CmpIPredicate::eq, raw, sentinel)
          .getResult();
  mlir::Value picked = mlir::arith::SelectOp::create(builder, loc(expr),
                                                     isSentinel, replacement,
                                                     raw)
                           .getResult();
  auto boxed =
      py::CastFromPrimOp::create(builder, loc(expr), intType, picked);
  return Value{boxed.getResult(), intType};
}

std::optional<Value>
ModuleEmitter::tryEmitRoundCall(const parser::Node &expr,
                                const parser::Node *calleeNode) {
  if (!callsUnshadowedBuiltin(calleeNode, "round"))
    return std::nullopt;
  static constexpr llvm::StringRef kParameters[] = {"number", "ndigits"};
  std::optional<llvm::SmallVector<const parser::Node *, 4>> bound =
      bindBuiltinArguments(expr, kParameters, /*positionalOnly=*/0);
  if (bound && !bound->empty()) {
    llvm::SmallVector<mlir::Value, 2> inputs;
    llvm::SmallVector<mlir::Type, 1> extraTypes;
    // `ndigits=None` IS the default (CPython's signature spells it that way),
    // so it means "no second argument" rather than an argument of type None --
    // which is what the __round__ contract would otherwise be asked to accept.
    bool explicitDigits =
        bound->size() == 2 && !((*bound)[1]->kind == "Constant" &&
                                ast::isNoneField(*(*bound)[1], "value"));
    Value receiver = emitExpr(bound->front());
    // round(int) is the identity (CPython); skipping the runtime call also
    // keeps the manifest __round__ contract at a fixed two-argument arity.
    if (!explicitDigits && types.widenLiteral(receiver.type) == types.intType())
      return coerceValue(receiver, types.intType(), expr);
    inputs.push_back(receiver.value);
    if (explicitDigits) {
      Value ndigits = emitExpr((*bound)[1]);
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
  bool builtinVisible = !programBindsName(name);
  bool hasKeywords = keywords && !keywords->empty();
  if (builtinVisible && args && args->size() == 1 && !hasKeywords &&
      (name == "repr" || name == "print")) {
    // Widen literals to their contract (`repr(5)` sees `builtins.int`, not
    // `literal<5>`) so the manifest `__repr__` resolves.
    //
    // ⛔ AND LOOK THROUGH A CELL. A name that a `with` body rebinds is promoted
    // to storage for the duration of the statement, so inside it the name's
    // static type is the cell and not what the cell holds:
    //
    //     count = 0
    //     with open(p) as f:
    //         for line in f:
    //             count += 1
    //         print(str(count))   # unresolved name 'repr'
    //
    // Every other reader of that name demotes on the way in -- `count + 1`,
    // `len`, `abs`, an f-string and `print` were all measured working in the
    // same position -- so this ladder is the one that asked the cell for a
    // `__repr__`, found none, and fell through to a `repr` BINDING that no
    // program declares.
    mlir::Type argumentType =
        types.widenLiteral(types.inferExpr(args->front().get()));
    if (isCellContract(argumentType))
      argumentType = types.widenLiteral(cellContentType(argumentType));
    std::optional<Value> repr;
    // print renders through str(), not repr(): a source-class __str__
    // outranks __repr__ here, and an exception subclass without its own
    // __str__ must fall to the sink's ancestor __str__ (the message form)
    // rather than inline a source __repr__ (the ClassName(...) form).
    // ⭐ Asked BEFORE the source lookup, because a subclass may be the only
    // class that declares the method: `class A: pass` / `class B(A): __repr__`
    // resolves nothing on A, fell through to the manifest repr, and printed
    // `<__main__.A object at 0x...>` where CPython prints B's.
    if (auto argumentContract =
            mlir::dyn_cast_if_present<py::ContractType>(argumentType)) {
      llvm::StringRef wanted = name == "print" ? "__str__" : "__repr__";
      if (subclassOverridesMethod(argumentContract.getContractName(), wanted)) {
        Value argument = emitExpr(args->front().get());
        if (dispatchIsUnresolvable(argument, wanted, /*receiverNode=*/nullptr,
                                   /*throughSuper=*/false)) {
          // The dispatcher answers the same string this path would have
          // inlined, so it joins the existing `repr` flow below rather than
          // printing on its own.
          if (std::optional<Value> dispatched =
                  tryEmitVirtualDispatchWithValues(expr, argument, wanted,
                                                   {})) {
            if (name == "repr")
              return *dispatched;
            CallOperands dispatchedOperands =
                emitCallOperands(expr, {*dispatched},
                                 /*includeAstArguments=*/false);
            std::optional<Value> printBinding = emitBuiltinBinding(name);
            if (!printBinding)
              return emitNone(expr);
            return emitCallableDispatch(expr, *printBinding,
                                        dispatchedOperands, types.none());
          }
          if (refuseUnresolvableDispatch(expr, argument, wanted))
            return emitNone(expr);
        }
      }
    }
    std::optional<MethodBinding> sourceMethod;
    if (name == "print")
      sourceMethod = lookupClassMethod(argumentType, "__str__");
    if (!sourceMethod &&
        !(name == "print" &&
          py::protocols::Table::get(context).isManifestSubclassOf(
              argumentType, "builtins.BaseException")))
      sourceMethod = lookupClassMethod(argumentType, "__repr__");
    if (sourceMethod) {
      // Source-class method: inline the method body.
      Value argument = emitExpr(args->front().get());
      if (refuseUnresolvableDispatch(
              expr, argument,
              name == "print" && lookupClassMethod(argumentType, "__str__")
                  ? "__str__"
                  : "__repr__"))
        return emitNone(expr);
      llvm::StringMap<Value> emptyKeywords;
      Value descriptorReceiver =
          emitDescriptorReceiver(expr, argument, *sourceMethod);
      repr = emitInlineMethodBody(expr, descriptorReceiver,
                                  methodBindingBindsReceiver(*sourceMethod),
                                  *sourceMethod, {}, emptyKeywords);
    } else if (name == "repr") {
      // A union has no contract of its own to resolve `__repr__` against, so
      // the ladder below answered nothing and the fall-through then tried to
      // resolve the NAME `repr` -- `repr(d.get("b"))` read "unresolved name
      // 'repr'", a builtin the program never mentioned. `str()` has said the
      // same thing about the same value since it grew its union arm; this is
      // that arm on the repr side, and `emitConversionValue` is where it
      // lives because !r and %r reach the union through the same door.
      if (mlir::isa<py::UnionType>(argumentType))
        if (std::optional<Value> rendered = emitConversionValue(
                expr, emitExpr(args->front().get()), 'r'))
          return *rendered;
      // Manifest-typed receiver (int/str/...): emit py.repr dispatch, the
      // same manifest path `str()` uses (avoid altering `print`'s existing
      // function-binding lowering, which this special case only optimizes).
      // User exception classes without their own __repr__ resolve against
      // their taxonomy ancestor instead of falling to object.__repr__: the
      // manifest exception __repr__ renders ClassName(...) by DYNAMIC class
      // id, so the user class's name survives the widening.
      mlir::Type reprReceiverType = argumentType;
      if (auto contractType = mlir::dyn_cast<py::ContractType>(argumentType)) {
        llvm::StringRef leaf =
            contractType.getContractName().rsplit('.').second;
        if (leaf.empty())
          leaf = contractType.getContractName();
        if (!py::exceptions::findByName(leaf)) {
          for (const std::string &cls :
               classMro(contractType.getContractName())) {
            llvm::StringRef clsLeaf = llvm::StringRef(cls).rsplit('.').second;
            if (clsLeaf.empty())
              clsLeaf = cls;
            if (py::exceptions::findByName(clsLeaf)) {
              reprReceiverType = types.contract(cls);
              break;
            }
          }
        }
      }
      if (CallInferenceResult inference = types.inferMethodCallWithEvidence(
              reprReceiverType, "__repr__", {})) {
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
    // ⭐ A FUNCTION VALUE HAS NO REPRESENTATION, and saying so here is the
    // difference between a sentence about the program and one about the
    // compiler. Nothing above resolves a `__repr__` for a callable, and the
    // fall-through then tried to resolve the NAME `repr` -- so `print(f)`,
    // `str(f)` and `repr(f)` all read "unresolved name 'repr'", which is a
    // builtin the program never mentioned. CPython renders the function's
    // identity (`<function g at 0x...>`), which is an address this compiler
    // does not keep.
    if (mlir::isa_and_nonnull<py::CallableType>(argumentType) ||
        (mlir::isa_and_nonnull<py::ProtocolType>(argumentType) &&
         mlir::cast<py::ProtocolType>(argumentType).getProtocolName() ==
             "Callable")) {
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, expr.range.start,
          "a function value has no repr(): CPython renders its identity and "
          "address, which this compiler does not keep"});
      return emitNone(expr);
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
