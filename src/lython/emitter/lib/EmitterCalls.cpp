#include "Contracts.h"
#include "AstSynth.h"
#include "BuiltinMethodKeywords.h"
#include "EmitterCore.h"
#include "EmitterPyOps.h"
#include "EmitterSupport.h"
#include "ArithBuilders.h"
#include "ExceptionTaxonomy.h"
#include "PlatformConstants.h"
#include "PyCallableShape.h"
#include "PyProtocols.h"
#include "TypeSystemSolver.h"

#include "AstAccess.h"

#include <functional>

#include "llvm/ADT/ScopeExit.h"
#include "EmitterOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Support/raw_ostream.h"

#include <iterator>
#include <optional>
#include <string>

namespace lython::emitter {
namespace {

using common::constantBool;


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
      // ⭐ A UNION OF bool AND int WHERE int IS EXPECTED is the int it is:
      // `xs.append(d[k])` over `{1: 5, 4: True}` had no signature for the
      // union, and the element keeps the bool's own object either way.
      if (positionalAligned && !unpacked &&
          mlir::isa<py::UnionType>(value.type) &&
          boolKeptAsInt(expectedFor(operands.positionalTypes.size()),
                        value.type))
        value = rungsAsInt(value, *valueNode);
      // ⭐ A UNION OF CLASSES BELOW THE PARAMETER'S IS THAT CLASS: `describe(
      // pick())` with `pick() -> ValueError | KeyError` and `describe(e:
      // Exception)` handed the call the union's tag and both members' lanes,
      // and the lowering could not adapt them to one Exception. Seen as the
      // base, the tag picks the member's lanes (aliasViewBundle).
      if (positionalAligned && !unpacked)
        if (auto expectedClass = mlir::dyn_cast_if_present<py::ContractType>(
                expectedFor(operands.positionalTypes.size()));
            expectedClass && !py::isPyObjectType(expectedClass))
          if (auto unionType = mlir::dyn_cast<py::UnionType>(value.type);
              unionType &&
              llvm::all_of(unionType.getMemberTypes(), [&](mlir::Type member) {
                auto memberClass = mlir::dyn_cast<py::ContractType>(
                    types.widenLiteral(member));
                return memberClass &&
                       pythonSubclassOf(memberClass, expectedClass, types,
                                        module);
              }))
            value = coerceValue(value, expectedClass, *valueNode);
      // ⭐ AN INLINED METHOD'S OWN `*args` IS THE VALUES ITS CALL GAVE. A pack
      // of one has no arity in its type (`tuple[str]` is also the
      // homogeneous spelling), so it is forwarded as those values, not
      // unpacked as a tuple.
      if (unpacked)
        if (auto pack = staticPositionalPacks.find(value.value);
            pack != staticPositionalPacks.end()) {
          for (const Value &element : pack->second) {
            operands.positional.push_back(element);
            operands.positionalUnpacked.push_back(0);
            operands.positionalTypes.push_back(element.type);
          }
          continue;
        }
      // ⛔ Nothing positional after a starred sequence of unknown length:
      // where its members end is a run-time fact, so a later argument has no
      // static position.
      if (operands.dynamicTail) {
        operands.valid = false;
        operands.invalidBeyondDynamic = true;
        operands.failureReason =
            "a starred argument whose length is known only at run time must "
            "be the last positional argument";
      }
      operands.positional.push_back(value);
      operands.positionalUnpacked.push_back(unpacked ? 1 : 0);
      if (unpacked) {
        if (!appendStarredArgumentTypes(types, value.type,
                                        operands.positionalTypes)) {
          // ⭐ A SEQUENCE OF UNKNOWN LENGTH IS KEPT WHOLE, as the `*tuple[T]`
          // the matcher reads (py::variadicTailElement): a callee whose
          // parameters the emitter knows can take it into its *args
          // (normalizeDynamicUnpacks).
          std::optional<mlir::Type> element =
              py::dynamicStarElementType(value.type);
          if (element && !operands.dynamicTail) {
            operands.dynamicTail =
                static_cast<unsigned>(operands.positional.size() - 1);
            operands.positionalTypes.push_back(py::variadicTailType(*element));
            operands.failureReason =
                "a starred argument whose length is known only at run time "
                "can be passed here only to the *args of a function or "
                "method this program defines";
          } else {
            operands.positionalTypes.push_back(value.type);
            operands.invalidBeyondDynamic = true;
            operands.failureReason =
                "starred call arguments require a statically sized tuple";
          }
          operands.valid = false;
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
      // ⭐ EXCEPT AN INLINED METHOD'S OWN `**kwargs`, whose names are the
      // call site's and whose body only reads it: forwarded as those
      // keywords, one by one.
      Value mapping = emitExpr(ast::node(*keyword, "value"));
      if (auto pack = staticKeywordPacks.find(mapping.value);
          pack != staticKeywordPacks.end()) {
        for (const auto &[packedName, packedValue] : pack->second) {
          mlir::Type literal = types.literal("\"" + packedName + "\"");
          auto stringOp = py::StrConstantOp::create(
              builder, loc(*keyword), literal,
              builder.getStringAttr(packedName));
          operands.keywordNames.push_back({stringOp.getResult(), literal});
          operands.keywordValues.push_back(packedValue);
          operands.keywordTypes.push_back(
              CallKeywordType{packedName, packedValue.type});
        }
        continue;
      }
      // ⭐ AND A MAPPING OF UNKNOWN KEYS, kept whole under the name "" -- the
      // spelling the matcher reads as `**mapping` -- for a callee whose
      // parameters the emitter knows (normalizeDynamicUnpacks).
      operands.valid = false;
      if (py::callableUnpackedKeywordValueType(
              types.widenLiteral(mapping.type)) &&
          !operands.dynamicMapping) {
        mlir::Type literal = types.literal("\"\"");
        auto stringOp = py::StrConstantOp::create(
            builder, loc(*keyword), literal, builder.getStringAttr(""));
        operands.dynamicMapping =
            static_cast<unsigned>(operands.keywordValues.size());
        operands.keywordNames.push_back({stringOp.getResult(), literal});
        operands.keywordValues.push_back(mapping);
        operands.keywordTypes.push_back(CallKeywordType{"", mapping.type});
        operands.failureReason =
            "a `**` mapping whose keys are known only at run time can be "
            "passed here only to the **kwargs of a function or method this "
            "program defines";
        continue;
      }
      operands.invalidBeyondDynamic = true;
      operands.failureReason =
          operands.dynamicMapping
              ? "a call takes one `**` mapping whose keys are known only at "
                "run time here; merge the mappings into one first"
              : "`**` call arguments require statically known keyword "
                "names; spell the keywords out";
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

// ⭐ A STARRED SEQUENCE OR A `**` MAPPING WHOSE SIZE ONLY THE RUN TIME KNOWS,
// collected the way CPython collects them: what goes to *args is one tuple --
// the positional values beyond the parameters, then the sequence's members --
// and what goes to **kwargs is one FRESH dict, the keywords and the mapping's
// items in call order, so a callee that writes to it does not write to the
// caller's mapping. A key the call also gives as a keyword raises CPython's
// TypeError, from the call site where CPython raises it.
//
// ⛔ ONLY WHEN THE SEQUENCE AND THE MAPPING CAN REACH NOTHING BUT *args AND
// **kwargs. Python hands a sequence's first members to the positional
// parameters the call leaves open, and binds a mapping's keys to the named
// parameters before anything reaches **kwargs -- defaulted ones included, so
// sending them all to *args or **kwargs would answer with the default where
// CPython answers with the member. Those calls are refused, naming the
// parameters to give explicitly.
std::optional<ModuleEmitter::DynamicUnpackResult>
ModuleEmitter::emitDynamicUnpack(const parser::Node &anchor,
                                 const DynamicUnpackTarget &target,
                                 const DynamicUnpackCall &call) {
  auto refuse = [&](const std::string &message)
      -> std::optional<DynamicUnpackResult> {
    diagnostics.push_back(parser::Diagnostic{parser::Severity::Error,
                                             anchor.range.start, message});
    return std::nullopt;
  };
  auto quoted = [](llvm::ArrayRef<std::string> names) {
    std::string text;
    for (auto [index, name] : llvm::enumerate(names)) {
      if (index != 0)
        text += index + 1 == names.size() ? " and " : ", ";
      text += "'" + name + "'";
    }
    return text;
  };
  std::string shown =
      target.qualname.empty() ? std::string("the callee")
                              : "'" + target.qualname + "'";
  parser::SourceRange range = anchor.range;
  llvm::SmallVector<std::string, 8> bound;
  auto releaseNames = llvm::make_scope_exit([&] {
    for (const std::string &name : bound)
      values.erase(name);
  });
  auto bind = [&](Value value) {
    std::string name =
        "__lyunpack" + std::to_string(++syntheticFunctionCounter);
    values[name] = value;
    types.bindSymbol(name, value.type);
    bound.push_back(name);
    return name;
  };
  auto emitSynthesized = [&](parser::NodePtr statement) {
    emitStatement(*statement);
    synthesizedIteratorDefs.push_back(std::move(statement));
  };
  DynamicUnpackResult result;

  if (call.tail) {
    unsigned positionalCount =
        static_cast<unsigned>(target.positionalNames.size());
    if (call.explicitPositional < positionalCount) {
      llvm::ArrayRef<std::string> open =
          llvm::ArrayRef<std::string>(target.positionalNames)
              .slice(call.explicitPositional);
      return refuse(
          "a starred argument whose length is known only at run time can fill "
          "only *args here, and its first members would fill " +
          quoted(open) + " of " + shown + ": give " +
          (open.size() == 1 ? "it" : "them") + " explicitly before it");
    }
    if (!target.varargType)
      return refuse(shown + " has no *args to take a starred argument whose "
                            "length is known only at run time");
    if (programBindsName("tuple"))
      return refuse("a starred argument whose length is known only at run "
                    "time is collected with tuple(), which this program "
                    "rebinds");
    std::vector<parser::NodePtr> extras;
    for (Value extra : call.extras)
      extras.push_back(synth::name(bind(extra), range));
    parser::NodePtr collected = synth::call(
        synth::name("tuple", range),
        std::vector<parser::NodePtr>{synth::name(bind(*call.tail), range)},
        range);
    if (!extras.empty())
      collected = synth::binOp(synth::tuple(std::move(extras), range), "Add",
                               std::move(collected), range);
    // ⛔ Not at a declared type that still names a type parameter: a generic
    // method's `*args: T` is solved FROM what is collected here, and a tuple
    // made at `tuple[T]` solved T as itself. It is made at the join of what
    // it holds instead.
    mlir::Type collectedType = target.varargType;
    if (unboundStaticParameterCount(collectedType) != 0) {
      llvm::SmallVector<mlir::Type, 4> members;
      for (Value extra : call.extras)
        members.push_back(types.widenLiteral(extra.type));
      if (std::optional<mlir::Type> element = py::dynamicStarElementType(
              types.widenLiteral(call.tail->type)))
        members.push_back(*element);
      collectedType = types.tupleOf(types.join(members));
    }
    Value merged = emitExprExpected(collected.get(), collectedType);
    synthesizedIteratorDefs.push_back(std::move(collected));
    result.vararg = coerceValue(merged, collectedType, anchor);
  }

  auto mapping = llvm::find_if(call.keywords, [](const auto &entry) {
    return entry.first.empty();
  });
  if (mapping == call.keywords.end()) {
    result.named.assign(call.keywords.begin(), call.keywords.end());
    return result;
  }

  // The parameters a keyword can name, and which of them the call fills by
  // position.
  llvm::SmallVector<std::string, 8> nameable;
  llvm::SmallVector<std::string, 4> byPosition;
  for (unsigned index = target.positionalOnlyCount;
       index < target.positionalNames.size(); ++index) {
    const std::string &name = target.positionalNames[index];
    if (name.empty())
      continue;
    nameable.push_back(name);
    if (index < call.explicitPositional)
      byPosition.push_back(name);
  }
  nameable.append(target.kwOnlyNames.begin(), target.kwOnlyNames.end());
  auto givenByKeyword = [&](llvm::StringRef name) {
    return llvm::any_of(call.keywords, [&](const auto &entry) {
      return entry.first == name;
    });
  };
  llvm::SmallVector<std::string, 4> open;
  for (const std::string &name : nameable)
    if (!llvm::is_contained(byPosition, name) && !givenByKeyword(name))
      open.push_back(name);
  if (!open.empty())
    return refuse(
        "a `**` mapping whose keys are known only at run time can fill only "
        "**kwargs here, and its keys could name " +
        quoted(open) + " of " + shown +
        ", which Python binds first: give " +
        (open.size() == 1 ? "it" : "them") + " explicitly");
  if (!target.kwargType)
    return refuse(shown + " has no **kwargs to take a `**` mapping whose "
                          "keys are known only at run time");
  if (programBindsName("dict"))
    return refuse("a `**` mapping whose keys are known only at run time is "
                  "copied with dict(), which this program rebinds");

  std::string qualified = target.module.empty()
                              ? target.qualname
                              : target.module + "." + target.qualname;
  std::string source = bind(mapping->second);
  auto raiseIfKey = [&](llvm::StringRef key, const std::string &message) {
    emitSynthesized(synth::ifStmt(
        synth::compareIn(synth::strConstant(key, range),
                         synth::name(source, range), range),
        {synth::raiseCall("TypeError", message, range)}, {}, range));
  };
  // CPython merges the keywords before it binds the parameters, so a key the
  // call also spells as a keyword is reported first, and as a keyword.
  for (const auto &[name, value] : call.keywords)
    if (!name.empty())
      raiseIfKey(name, qualified + "() got multiple values for keyword "
                                   "argument '" + name + "'");
  for (const std::string &name : byPosition)
    raiseIfKey(name, target.qualname + "() got multiple values for argument '" +
                         name + "'");

  // The dict: the extra keywords before the mapping, its items, then the extra
  // keywords after it.
  std::vector<parser::NodePtr> keys;
  std::vector<parser::NodePtr> initial;
  llvm::SmallVector<std::pair<std::string, Value>, 4> after;
  bool pastMapping = false;
  for (const auto &entry : call.keywords) {
    if (entry.first.empty()) {
      pastMapping = true;
      continue;
    }
    if (llvm::is_contained(nameable, entry.first)) {
      result.named.push_back(entry);
      continue;
    }
    if (pastMapping) {
      after.push_back(entry);
      continue;
    }
    keys.push_back(synth::strConstant(entry.first, range));
    initial.push_back(synth::name(bind(entry.second), range));
  }
  bool seeded = !keys.empty();
  parser::NodePtr start;
  if (seeded) {
    start = parser::makeNode("Dict", range);
    parser::addField(*start, "keys", std::move(keys));
    parser::addField(*start, "values", std::move(initial));
  } else {
    start = synth::call(
        synth::name("dict", range),
        std::vector<parser::NodePtr>{synth::name(source, range)}, range);
  }
  // ⛔ Not at a declared type that still names a type parameter, as for the
  // tuple above: at the join of the values the dict holds.
  mlir::Type collectedType = target.kwargType;
  if (unboundStaticParameterCount(collectedType) != 0) {
    llvm::SmallVector<mlir::Type, 4> members;
    for (const auto &[name, value] : call.keywords) {
      if (name.empty()) {
        if (std::optional<mlir::Type> member =
                py::callableUnpackedKeywordValueType(
                    types.widenLiteral(value.type)))
          members.push_back(*member);
      } else if (!llvm::is_contained(nameable, name)) {
        members.push_back(types.widenLiteral(value.type));
      }
    }
    collectedType = types.dictOf(types.strType(), types.join(members));
  }
  Value made = emitExprExpected(start.get(), collectedType);
  synthesizedIteratorDefs.push_back(std::move(start));
  std::string collected = bind(coerceValue(made, collectedType, anchor));
  if (seeded)
    emitSynthesized(synth::exprStmt(
        synth::methodCall(synth::name(collected, range), "update",
                          {synth::name(source, range)}, range),
        range));
  for (const auto &[name, value] : after)
    emitSynthesized(synth::assign(
        synth::subscript(synth::name(collected, range),
                         synth::strConstant(name, range), range),
        synth::name(bind(value), range), range));
  result.kwargs = values.lookup(collected);
  return result;
}

bool ModuleEmitter::normalizeDynamicUnpacks(const parser::Node &anchor,
                                            const DynamicUnpackTarget &target,
                                            CallOperands &operands) {
  if (operands.invalidBeyondDynamic ||
      (!operands.dynamicTail && !operands.dynamicMapping))
    return false;
  DynamicUnpackCall call;
  unsigned positionalCount =
      static_cast<unsigned>(target.positionalNames.size());
  // The operands that fill parameters, then the ones *args takes first. A
  // statically sized starred tuple among them stays whole only where it does
  // not straddle the two.
  unsigned firstExtra = static_cast<unsigned>(operands.positional.size());
  if (operands.dynamicTail) {
    unsigned tail = *operands.dynamicTail;
    firstExtra = tail;
    unsigned covered = 0;
    for (unsigned index = 0; index < tail; ++index) {
      llvm::SmallVector<mlir::Type, 4> widths;
      unsigned width = 1;
      if (operands.positionalUnpacked[index] &&
          appendStarredArgumentTypes(types, operands.positional[index].type,
                                     widths))
        width = static_cast<unsigned>(widths.size());
      if (covered >= positionalCount) {
        if (operands.positionalUnpacked[index]) {
          diagnostics.push_back(parser::Diagnostic{
              parser::Severity::Error, anchor.range.start,
              "a starred tuple after the parameters and before a starred "
              "argument whose length is known only at run time is not "
              "collected into *args here; give its members one by one"});
          return false;
        }
        if (firstExtra == tail)
          firstExtra = index;
        call.extras.push_back(operands.positional[index]);
      } else if (covered + width > positionalCount) {
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, anchor.range.start,
            "a starred tuple that fills both parameters and *args is not "
            "collected here; give its members one by one"});
        return false;
      }
      covered += width;
    }
    call.explicitPositional = covered;
    call.tail = operands.positional[tail];
  } else {
    call.explicitPositional =
        static_cast<unsigned>(operands.positionalTypes.size());
  }
  for (auto [index, value] : llvm::enumerate(operands.keywordValues))
    call.keywords.push_back({operands.keywordTypes[index].name, value});

  std::optional<DynamicUnpackResult> result =
      emitDynamicUnpack(anchor, target, call);
  if (!result)
    return false;

  CallOperands rebuilt;
  if (result->vararg) {
    // The parameters' own operands stay; the extras went into the tuple, which
    // is the last operand and the whole of *args (`ly.variadic_tail`).
    rebuilt.positional.assign(operands.positional.begin(),
                              operands.positional.begin() + firstExtra);
    rebuilt.positionalUnpacked.assign(
        operands.positionalUnpacked.begin(),
        operands.positionalUnpacked.begin() + firstExtra);
    rebuilt.positionalTypes.assign(operands.positionalTypes.begin(),
                                   operands.positionalTypes.begin() +
                                       positionalCount);
    rebuilt.positional.push_back(*result->vararg);
    rebuilt.positionalUnpacked.push_back(1);
    lython::callable::VarargShape<mlir::Type> shape =
        py::callableVarargShape(target.varargType);
    rebuilt.positionalTypes.push_back(py::variadicTailType(
        shape.repeated ? *shape.repeated : target.varargType));
    rebuilt.variadicTail = true;
  } else {
    rebuilt.positional = operands.positional;
    rebuilt.positionalUnpacked = operands.positionalUnpacked;
    rebuilt.positionalTypes = operands.positionalTypes;
  }
  auto pushKeyword = [&](llvm::StringRef name, Value value) {
    mlir::Type literal = types.literal("\"" + name.str() + "\"");
    auto stringOp = py::StrConstantOp::create(builder, loc(anchor), literal,
                                              builder.getStringAttr(name));
    rebuilt.keywordNames.push_back({stringOp.getResult(), literal});
    rebuilt.keywordValues.push_back(value);
    rebuilt.keywordTypes.push_back(CallKeywordType{name.str(), value.type});
  };
  for (const auto &[name, value] : result->named)
    pushKeyword(name, value);
  if (result->kwargs)
    pushKeyword("", *result->kwargs);
  operands = std::move(rebuilt);
  return true;
}

// The callee of a direct call the emitter knows the parameters of: a function
// this program defines, called through its symbol. A manifest export lowers to
// its own runtime call, which takes no collected *args or **kwargs.
std::optional<ModuleEmitter::DynamicUnpackTarget>
ModuleEmitter::dynamicUnpackTargetFor(const parser::Node &anchor,
                                      Value callee) {
  auto ref = callee.value ? callee.value.getDefiningOp<py::BindingRefOp>()
                          : py::BindingRefOp();
  auto callable = mlir::dyn_cast_if_present<py::CallableType>(callee.type);
  if (!ref || !callable)
    return std::nullopt;
  // ⛔ ONLY A FUNCTION THIS PROGRAM DEFINES: one already emitted with its
  // callable type, or a module-level def the call reaches before its body is
  // emitted. Asked positively because the reverse question -- is this binding
  // a builtin? -- missed `print`, and `print(*xs)` then printed the collected
  // TUPLE: a builtin lowers to its own runtime call, which never reads the
  // collected *args the way a function's parameter list does.
  llvm::StringRef binding = ref.getBinding();
  bool emitted = false;
  if (auto function = module.lookupSymbol<mlir::func::FuncOp>(binding))
    emitted = !function.isDeclaration() && function->hasAttr("callable_type");
  bool pending = llvm::any_of(moduleFunctionNames, [&](const auto &entry) {
    return topLevelFunctionSymbol(entry.getKey()) == binding;
  });
  if (!emitted && !pending)
    return std::nullopt;
  DynamicUnpackTarget target;
  llvm::ArrayRef<mlir::StringAttr> names = callable.getPositionalNames();
  for (std::size_t index = 0; index < callable.getPositionalTypes().size();
       ++index)
    target.positionalNames.push_back(
        index < names.size() && names[index] ? names[index].getValue().str()
                                             : std::string());
  target.positionalOnlyCount = callable.getPositionalOnlyCount();
  for (mlir::StringAttr name : callable.getKwOnlyNames())
    target.kwOnlyNames.push_back(name ? name.getValue().str() : std::string());
  if (callable.hasVararg())
    target.varargType = callable.getVarargType();
  if (callable.hasKwarg())
    target.kwargType = callable.getKwargType();
  for (const EmittedPythonDefinition &definition : emittedPythonDefinitions)
    if (definition.symbol == ref.getBinding()) {
      target.module = definition.module;
      target.qualname = definition.qualname;
      return target;
    }
  // Not emitted yet (a function written after the call): its name as the
  // call spells it, in the module being emitted.
  if (const parser::Node *function = ast::node(anchor, "func")) {
    if (function->kind == "Name")
      target.qualname = std::string(ast::nameSpelling(*function));
    else if (std::optional<std::string_view> attr =
                 ast::string(*function, "attr"))
      target.qualname = std::string(*attr);
  }
  target.module =
      activeSourceModule ? activeSourceModule->moduleName : moduleName;
  return target;
}

Value ModuleEmitter::emitCallableDispatch(const parser::Node &anchor,
                                          Value callee,
                                          const CallOperands &givenOperands,
                                          mlir::Type resultOverride) {
  CallOperands normalized;
  const CallOperands *chosen = &givenOperands;
  if (!givenOperands.valid && !givenOperands.invalidBeyondDynamic &&
      (givenOperands.dynamicTail || givenOperands.dynamicMapping))
    if (std::optional<DynamicUnpackTarget> target =
            dynamicUnpackTargetFor(anchor, callee)) {
      normalized = givenOperands;
      if (!normalizeDynamicUnpacks(anchor, *target, normalized))
        return emitNone(anchor);
      chosen = &normalized;
    }
  // ⭐ A bool, OR A UNION OF bool AND int, GIVEN AN int PARAMETER is passed
  // as the int it is. A function reached through a value has one body, at
  // its declared type, so it cannot be specialized at the argument's rung as
  // a direct call is: `f(True)` for `f: Callable[[int], int]` failed in the
  // lowering ("arguments do not match Callable contract") and `f(d[k])` was
  // refused at emit. The int keeps the bool's object, so the callee sees it.
  // A builtin's overloads say which one the call takes only once it is
  // resolved: asked with each such union as the int it becomes, the overload
  // that answers says where an int is declared. ⛔ Not a lone bool there: the
  // manifest ABI widens its bit, and its result is the builtin's.
  CallOperands asInts;
  py::CallableType callable = mlir::dyn_cast_if_present<py::CallableType>(
      types.widenLiteral(callee.type));
  bool unionsOnly = false;
  if (!callable && chosen->valid &&
      !mlir::isa<py::TypeType>(types.widenLiteral(callee.type))) {
    llvm::SmallVector<mlir::Type, 4> asked(chosen->positionalTypes.begin(),
                                           chosen->positionalTypes.end());
    bool anyUnion = false;
    for (mlir::Type &type : asked)
      if (mlir::isa<py::UnionType>(type) &&
          boolKeptAsInt(types.intType(), type)) {
        type = types.intType();
        anyUnion = true;
      }
    if (anyUnion) {
      CallInferenceResult chosenOverload = types.inferCallWithEvidence(
          callee.type, asked, chosen->keywordTypes);
      callable = mlir::dyn_cast_if_present<py::CallableType>(
          chosenOverload.evidence.callableContract);
      unionsOnly = true;
    }
  }
  if (callable && chosen->valid)
    for (auto [index, value] : llvm::enumerate(chosen->positional)) {
      if (index >= callable.getPositionalTypes().size() ||
          index >= chosen->positionalUnpacked.size() ||
          chosen->positionalUnpacked[index] ||
          (unionsOnly && !mlir::isa<py::UnionType>(value.type)) ||
          !boolKeptAsInt(callable.getPositionalTypes()[index], value.type))
        continue;
      if (chosen != &asInts) {
        asInts = *chosen;
        chosen = &asInts;
      }
      asInts.positional[index] = rungsAsInt(value, anchor);
      asInts.positionalTypes[index] = asInts.positional[index].type;
    }
  const CallOperands &operands = *chosen;
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
  if (operands.variadicTail)
    posPack.value.getDefiningOp()->setAttr(kPackVariadicTailAttr,
                                           builder.getUnitAttr());
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

// ⭐ A UNION RECEIVER CALLS ITS METHOD BY TAG, the way it adds, counts and
// indexes: every member has the method or the call is refused
// (TypeSystem::unionMethodResult), and the value is the join of the arms.
// `s.write(t)` where `s` is a `StringIO | TextIOWrapper` -- a stream chosen at
// run time -- was "static type ... does not provide manifest method 'write'"
// for a call both members answer.
//
// The arguments are evaluated once, in order, before the tag is asked; each
// arm is then an ordinary call on the member, so a program class's member
// inlines its body or dispatches as a direct call on it would.
//
// ⛔ ASKED ON TYPES BEFORE ANYTHING IS EMITTED: a member that cannot answer
// leaves the call to the refusal below with nothing evaluated twice.
//
// ⛔ NOT A STRUCTURAL MUTATOR, and that one is refused by name. Such a method
// may reallocate its receiver and hands the new one back to rebind the NAME it
// was called through; an arm's receiver is the member it unwrapped, so the
// rebinding would land there and the union would keep the old storage.
std::optional<Value>
ModuleEmitter::tryEmitUnionMethodCall(const parser::Node &expr, Value receiver,
                                      llvm::StringRef methodName) {
  auto unionType =
      mlir::dyn_cast_if_present<py::UnionType>(types.widenLiteral(receiver.type));
  if (!unionType)
    return std::nullopt;
  const auto *args = ast::nodeList(expr, "args");
  const auto *keywords = ast::nodeList(expr, "keywords");
  llvm::SmallVector<mlir::Type, 4> positionalTypes;
  llvm::SmallVector<CallKeywordType, 2> keywordTypes;
  if (args)
    for (const parser::NodePtr &argument : *args) {
      if (!argument || argument->kind == "Starred")
        return std::nullopt;
      positionalTypes.push_back(types.inferExpr(argument.get()));
    }
  if (keywords)
    for (const parser::NodePtr &keyword : *keywords) {
      std::optional<std::string_view> name =
          keyword ? ast::string(*keyword, "arg") : std::nullopt;
      if (!name)
        return std::nullopt;
      keywordTypes.push_back(CallKeywordType{
          std::string(*name), types.inferExpr(ast::node(*keyword, "value"))});
    }
  mlir::Type joined = types.unionMethodResult(receiver.type, methodName,
                                              positionalTypes, keywordTypes);
  if (!joined)
    return std::nullopt;
  for (mlir::Type member : unionType.getMemberTypes())
    if (types.isStructuralMutatorMethod(types.widenLiteral(member),
                                        methodName)) {
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, expr.range.start,
          "'" + std::string(methodName) +
              "' can reallocate the object it is called on, and here that "
              "object is one member of a union: narrow the value to that "
              "member first, so the call rebinds the name that holds it"});
      return emitNone(expr);
    }

  llvm::SmallVector<Value, 4> positional;
  if (args)
    for (const parser::NodePtr &argument : *args)
      positional.push_back(emitExpr(argument.get()));
  llvm::SmallVector<std::pair<std::string, Value>, 2> named;
  if (keywords)
    for (const parser::NodePtr &keyword : *keywords)
      named.emplace_back(std::string(*ast::string(*keyword, "arg")),
                         emitExpr(ast::node(*keyword, "value")));

  return emitUnionMemberDispatch(
      expr, receiver, unionType, joined, [&](Value member) -> Value {
        llvm::SmallVector<std::string, 4> bound;
        auto bind = [&](Value value) {
          std::string name =
              "__lyunioncall" + std::to_string(++syntheticFunctionCounter);
          values[name] = value;
          types.bindSymbol(name, value.type);
          bound.push_back(name);
          return synth::name(name, expr.range);
        };
        parser::NodePtr memberNode = bind(member);
        std::vector<parser::NodePtr> argumentNodes;
        for (Value value : positional)
          argumentNodes.push_back(bind(value));
        std::vector<parser::NodePtr> keywordNodes;
        for (const auto &[name, value] : named)
          keywordNodes.push_back(synth::keyword(name, bind(value), expr.range));
        parser::NodePtr call = synth::callWithKeywords(
            synth::attribute(std::move(memberNode), methodName, expr.range),
            std::move(argumentNodes), std::move(keywordNodes), expr.range);
        Value result = emitExpr(call.get());
        synthesizedIteratorDefs.push_back(std::move(call));
        for (const std::string &name : bound)
          values.erase(name);
        return result;
      });
}

llvm::SmallVector<std::string, 8>
ModuleEmitter::directSubclassesOf(llvm::StringRef cls) {
  llvm::SmallVector<std::pair<unsigned, std::string>, 8> found;
  for (const auto &entry : declaredClassBases) {
    if (!llvm::is_contained(entry.second, cls))
      continue;
    auto rank = declaredClassRank.find(entry.getKey());
    found.push_back({rank == declaredClassRank.end() ? ~0u : rank->second,
                     entry.getKey().str()});
  }
  llvm::sort(found);
  llvm::SmallVector<std::string, 8> names;
  for (auto &[rank, name] : found)
    names.push_back(std::move(name));
  return names;
}

// ⭐ `C.__subclasses__()` IS ANSWERED FROM THE PROGRAM, which holds every
// class there will ever be: the classes that name C as a base, in the order
// they are created -- an imported module's before the main module's, each in
// the order written. A receiver that is the class itself is a list display;
// one that is a class VALUE asks which class it holds, among C and the classes
// below it. unittest's `main()` finds the test classes this way.
// ⛔ Program classes only: `object.__subclasses__()` in CPython also lists the
// interpreter's own, which no program here can match, so it is not answered.
// ⛔ Refused when a list would hold a class written inside a function. Such a
// class is a subclass only from the call that runs its statement until the
// class is collected (a class is in a reference cycle, so that is when the
// cycle collector runs), and every call makes another one -- none of which the
// program text decides. It is hoisted to the module's top level here, so
// answering would list it before the function has run, and listing it last
// would still be one class where CPython has one per call.
std::optional<Value>
ModuleEmitter::tryEmitSubclassesCall(const parser::Node &expr) {
  const parser::Node *callee = ast::node(expr, "func");
  if (!callee || callee->kind != "Attribute" ||
      ast::string(*callee, "attr").value_or("") != "__subclasses__")
    return std::nullopt;
  const auto *args = ast::nodeList(expr, "args");
  const auto *keywords = ast::nodeList(expr, "keywords");
  if ((args && !args->empty()) || (keywords && !keywords->empty()))
    return std::nullopt;
  const parser::Node *receiverNode = ast::node(*callee, "value");
  auto receiverType = mlir::dyn_cast_if_present<py::TypeType>(
      types.widenLiteral(types.inferExpr(receiverNode)));
  auto cls = receiverType ? mlir::dyn_cast_if_present<py::ContractType>(
                                receiverType.getInstanceType())
                          : py::ContractType();
  if (!cls || !types.lookupClass(cls.getContractName()) ||
      !declaredClassBases.count(cls.getContractName()))
    return std::nullopt;
  Value receiver = emitExpr(receiverNode);
  mlir::Type elementType = receiverType;
  mlir::Type listType = types.contract("builtins.list", {elementType});
  auto typeObjectOf = [&](llvm::StringRef name) -> std::optional<Value> {
    std::optional<mlir::Type> instance = types.lookupClass(name);
    if (!instance)
      return std::nullopt;
    mlir::Type classType = types.typeObject(*instance);
    return Value{py::TypeObjectOp::create(builder, loc(expr), classType,
                                          *instance)
                     .getResult(),
                 classType};
  };
  auto listOf = [&](llvm::StringRef name) -> mlir::Value {
    llvm::SmallVector<mlir::Value, 8> operands;
    for (const std::string &sub : directSubclassesOf(name))
      if (std::optional<Value> object = typeObjectOf(sub))
        operands.push_back(coerceValue(*object, elementType, expr).value);
    return py::PackOp::create(builder, loc(expr), listType, operands)
        .getResult();
  };
  // The classes a value of `type[C]` may hold, below C.
  llvm::SmallVector<std::string, 8> below;
  llvm::SmallVector<std::string, 8> pending{cls.getContractName().str()};
  llvm::StringSet<> seen;
  while (!pending.empty()) {
    std::string next = pending.pop_back_val();
    for (const std::string &sub : directSubclassesOf(next))
      if (seen.insert(sub).second) {
        below.push_back(sub);
        pending.push_back(sub);
      }
  }
  bool staticReceiver = receiver.value.getDefiningOp<py::TypeObjectOp>();
  llvm::SmallVector<std::string, 8> answered{cls.getContractName().str()};
  if (!staticReceiver)
    answered.append(below.begin(), below.end());
  for (const std::string &asked : answered)
    for (const std::string &sub : directSubclassesOf(asked))
      if (llvm::StringRef(sub).contains(py::contracts::kLocalClassScopeMarker)) {
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, expr.range.start,
            "'" + py::contracts::pythonQualnameForContract(asked) +
                ".__subclasses__()' would list '" +
                py::contracts::pythonQualnameForContract(sub) +
                "', a class written inside a function: it is a subclass only "
                "while a call has made it and something holds it, which the "
                "program does not decide statically"});
        return emitNone(expr);
      }
  if (staticReceiver || below.empty())
    return Value{listOf(cls.getContractName()), listType};
  std::function<mlir::Value(std::size_t)> chain =
      [&](std::size_t index) -> mlir::Value {
    if (index == below.size())
      return listOf(cls.getContractName());
    std::optional<Value> candidate = typeObjectOf(below[index]);
    if (!candidate)
      return chain(index + 1);
    mlir::Value same =
        py::IsOp::create(builder, loc(expr), builder.getI1Type(),
                         receiver.value, candidate->value)
            .getResult();
    return emitValueDiamond(
        loc(expr), same, listType,
        [&] { return listOf(below[index]); },
        [&] { return chain(index + 1); });
  };
  return Value{chain(0), listType};
}

Value ModuleEmitter::emitCall(const parser::Node &expr) {
  const parser::Node *calleeNode = ast::node(expr, "func");
  // `__ly$target_is(f, "symbol")`, which only a function-docstring dispatcher
  // writes (EmitterDocstrings.cpp): whether the function value `f` was made
  // for that function. No source can spell the name.
  if (calleeNode && calleeNode->kind == "Name" &&
      ast::nameSpelling(*calleeNode) == "__ly$target_is")
    if (const auto *args = ast::nodeList(expr, "args");
        args && args->size() == 2 && (*args)[1])
      if (std::optional<std::string_view> target =
              ast::string(*(*args)[1], "value")) {
        Value function = emitExpr((*args)[0].get());
        auto test = py::TargetIsOp::create(
            builder, loc(expr), builder.getI1Type(), function.value,
            builder.getStringAttr(llvm::StringRef(target->data(),
                                                  target->size())));
        auto boolean = py::CastFromPrimOp::create(builder, loc(expr),
                                                  types.boolType(), test);
        return Value{boolean.getResult(), types.boolType()};
      }
  if (calleeNode && calleeNode->kind == "Attribute" &&
      refuseInternalManifestMethod(*calleeNode))
    return emitNone(expr);
  if (std::optional<Value> subclasses = tryEmitSubclassesCall(expr))
    return *subclasses;
  if (std::optional<Value> exited = tryEmitSysExit(expr))
    return *exited;
  // ⛔ A METHOD OF A CLASS THAT DEFINES `__getattribute__` IS NOT CALLED
  // DIRECTLY: CPython looks `obj.m` up through that method first, and what it
  // answers is what is called. Calling the method the class declares would
  // skip it in silence, and the delegation it would need --
  // `object.__getattribute__(self, name)` with the name a value -- has no
  // static answer here.
  if (calleeNode && calleeNode->kind == "Attribute")
    if (std::optional<std::string_view> method = ast::string(*calleeNode, "attr");
        method && *method != "__getattribute__" && *method != "__getattr__")
      if (auto receiver = mlir::dyn_cast_if_present<py::ContractType>(
              types.widenLiteral(types.inferExpr(ast::node(*calleeNode, "value")))))
        if (std::optional<MethodBinding> hook =
                lookupClassMethod(receiver, "__getattribute__");
            hook && hook->method && lookupClassMethod(receiver, *method)) {
          diagnostics.push_back(parser::Diagnostic{
              parser::Severity::Error, expr.range.start,
              "'" + receiver.getContractName().str() + "' defines "
              "__getattribute__, which CPython asks for '" +
                  std::string(*method) +
                  "' before calling it; a method call cannot be resolved "
                  "around it statically"});
          return emitNone(expr);
        }
  if (std::optional<Value> program = tryEmitTestProgramWithModuleFixtures(expr))
    return *program;
  if (calleeNode && calleeNode->kind == "Call")
    if (std::optional<Value> curried =
            tryEmitCurriedGenericCall(expr, *calleeNode))
      return *curried;
  if (calleeNode &&
      (calleeNode->kind == "Name" || calleeNode->kind == "Attribute"))
    if (std::string spelled = ast::qualifiedName(calleeNode); !spelled.empty())
      if (std::optional<std::string> canonical =
              types.lookupCanonicalBinding(spelled);
          canonical && *canonical == "sys.exc_info")
        return emitExcInfo(expr);
  if (std::optional<Value> dispatched = emitClassValueCall(expr, calleeNode))
    return *dispatched;
  // ⭐ A UNION OF CALLABLE INSTANCES IS CALLED THROUGH ITS LIVE MEMBER'S
  // `__call__`: `test(result)` for `test: TestCase | TestSuite` is how
  // unittest's suite runs what it holds, and the union method path answers
  // `test.__call__(result)` already. Re-spelled as that, the callee is still
  // evaluated once -- it is the call's receiver.
  // ⛔ Only when every member declares `__call__`: a member that does not is
  // a TypeError CPython raises at run time, and the refusal says which.
  if (calleeNode && calleeNode->kind != "Lambda")
    if (auto unionCallee = mlir::dyn_cast_if_present<py::UnionType>(
            types.widenLiteral(types.inferExpr(calleeNode)));
        unionCallee &&
        llvm::all_of(unionCallee.getMemberTypes(), [&](mlir::Type member) {
          return mlir::isa<py::ContractType>(member) &&
                 lookupClassMethod(member, "__call__").has_value();
        })) {
      parser::NodePtr callee;
      if (const parser::Field *func = parser::findField(expr, "func"))
        if (const auto *held = std::get_if<parser::NodePtr>(&func->value))
          callee = *held;
      parser::NodePtr rewritten = parser::makeNode("Call", expr.range);
      parser::addField(*rewritten, "func",
                       synth::attribute(callee, "__call__", calleeNode->range));
      for (llvm::StringRef field : {"args", "keywords"})
        if (const parser::Field *original = parser::findField(expr, field))
          rewritten->fields.push_back(*original);
      synthesizedIteratorDefs.push_back(rewritten);
      return emitCall(*rewritten);
    }
  if (calleeNode && calleeNode->kind == "Attribute")
    if (std::optional<Value> refused = refuseUnboundAttributeRoot(*calleeNode))
      return *refused;
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
  // CPython's `Protocol.__init__` refuses: a protocol describes classes, it
  // is not one to build. Here it is an ordinary class (desugarProtocols), so
  // the call would construct an instance whose methods are the stubs.
  if (calleeNode && (calleeNode->kind == "Name" ||
                     calleeNode->kind == "Attribute")) {
    llvm::StringRef leaf = calleeNode->kind == "Name"
                               ? llvm::StringRef(ast::nameSpelling(*calleeNode))
                               : llvm::StringRef(ast::string(*calleeNode, "attr")
                                                     .value_or(""));
    if (protocolClassNames.count(leaf) &&
        (calleeNode->kind != "Name" || values.find(leaf) == values.end()) &&
        types.lookupClass(canonicalClassName(calleeQualified)) !=
            mlir::Type()) {
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, expr.range.start,
          "'" + leaf.str() + "' is a Protocol, and protocols cannot be "
          "instantiated (CPython raises TypeError); construct a class that "
          "has its members"});
      return emitNone(expr);
    }
  }
  if (!calleeQualified.empty())
    if (std::optional<Value> host =
            tryEmitJsHostFunctionCall(expr, calleeQualified))
      return *host;

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
  // The table is CPython's argument clinic (BuiltinMethodKeywords.h).
  // Binding runs through the same bindBuiltinArguments the builtin fast paths
  // use, so a gap (`split(maxsplit=1)`, which names no separator) stays
  // refused rather than being invented.
  if (calleeNode && calleeNode->kind == "Attribute")
    if (const auto *methodKeywords = ast::nodeList(expr, "keywords");
        methodKeywords && !methodKeywords->empty()) {
      llvm::StringRef methodName =
          ast::string(*calleeNode, "attr").value_or("");
      std::optional<BuiltinMethodParameters> entry =
          builtinMethodParameters(methodName);
      std::optional<llvm::SmallVector<const parser::Node *, 4>> bound;
      if (entry)
        bound = bindBuiltinArguments(expr, entry->names, entry->positionalOnly);
      if (bound) {
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
        if (recovered) {
          parser::NodePtr rewritten = parser::makeNode("Call", expr.range);
          if (const parser::Field *func = parser::findField(expr, "func"))
            rewritten->fields.push_back(*func);
          parser::addField(*rewritten, "args", std::move(positional));
          parser::addField(*rewritten, "keywords",
                           std::vector<parser::NodePtr>{});
          return emitCall(*rewritten);
        }
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
  if (std::optional<Value> v = tryEmitObjectSlotCall(expr, calleeNode))
    return *v;
  if (std::optional<Value> v = tryEmitHasattrCall(expr, calleeNode))
    return *v;
  if (std::optional<Value> v = tryEmitCallableCall(expr, calleeNode))
    return *v;
  if (std::optional<Value> v = tryEmitGetattrCall(expr, calleeNode))
    return *v;
  if (std::optional<Value> v = tryEmitComputedGetattrCall(expr, calleeNode))
    return *v;
  if (std::optional<Value> v = tryEmitDirCall(expr, calleeNode))
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

  // ⭐ A NAME BOUND TO A TYPE OBJECT NAMES THE CLASS IT HOLDS, not the class of
  // its own spelling. The note below says a VALUE binding that shares a class's
  // spelling keeps reaching the constructor, and for an ordinary value that is
  // right -- it is not callable as a class anyway. A `type[X]` value IS, and it
  // may hold a DIFFERENT class:
  //
  //     def build(A: type[B], n: int) -> B:
  //         return A(n)
  //     print(build(B, 1).tag)     # 'B'; this built an A
  //
  // which is `type[X]` under any parameter name that a class in the module
  // happens to share. Every other spelling of the shadowing works -- an int
  // parameter, a local, a loop target -- because none of them reaches this
  // branch.
  auto boundTypeObject = [&](llvm::StringRef name) {
    auto bound = values.find(name);
    if (bound == values.end())
      return false;
    // A nested `def` binds a callable under its own name; a top-level one is
    // already excluded by `moduleFunctionNames`, and this is the same rule one
    // scope in.
    if (mlir::isa_and_nonnull<py::CallableType>(bound->second.type))
      return true;
    auto typeObject = mlir::dyn_cast_if_present<py::TypeType>(bound->second.type);
    if (!typeObject)
      return false;
    // ⛔ ONLY WHEN THE TWO DISAGREE, which the inference already required and
    // this did not: `def build(Widget: type[Widget], n: int)` binds a type
    // object holding the very class its name spells, and steering that away
    // from the constructor sent it to "calling a type object held in a value
    // is not supported" -- a program that worked before the shadowing rule
    // existed. What the rule is for is the case where the value holds a
    // DIFFERENT class.
    std::optional<mlir::Type> sameName = types.lookupClass(name);
    return !sameName || *sameName != typeObject.getInstanceType();
  };

  // Same rule as the bare-Name constructor path below: a top-level `def int`
  // -- or one imported under that name -- outranks the builtin class contract
  // of that spelling. A bare Name has a qualified spelling equal to itself, so
  // this branch sees `int` first.
  if (!calleeQualified.empty() && !moduleFunctionNames.count(calleeQualified) &&
      !boundTypeObject(calleeQualified) &&
      !importBindsCallable(calleeQualified))
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
    // An imported def under a class spelling is the same competing callable
    // (`importBindsCallable`). It was once added only here and measured to
    // change nothing: a bare Name reaches the qualified-spelling gate above
    // first, which is where it has to be asked too.
    if (auto cls = (moduleFunctionNames.count(name) || boundTypeObject(name) ||
                    importBindsCallable(name))
                       ? std::optional<mlir::Type>()
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
        // (a Buffered* result); see the matching special case in inferExpr.
        if (*canonical == "_io.open") {
          const auto *openArgs = ast::nodeList(expr, "args");
          if (openArgs && openArgs->size() >= 2 && (*openArgs)[1]) {
            auto mode = ast::string(*(*openArgs)[1], "value");
            if (mode && mode->find('b') != std::string_view::npos) {
              // The Buffered* opener when io is part of the program (the
              // driver adds it for exactly these calls); the raw FileIO
              // otherwise, which is what this was before io could be asked.
              const char *opener = binaryOpenFunction(*mode);
              if (std::optional<mlir::Type> symbol =
                      types.lookupSymbol(opener))
                return emitCallableDispatch(
                    expr, emitBindingRef(*calleeNode, opener, *symbol),
                    emitCallOperands(expr));
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
      // Qualified references to imported generics (module.fn(...)) resolve
      // through the canonical binding to the same registration the bare
      // import-name path uses.
      // ⭐ AN IMPORTED MODULE'S CALLABLE CONSTANT IS A CELL, and a CALL
      // through it took the binding-ref route: `lib.DOUBLE(2)` for a
      // module-level lambda was "unresolved runtime binding 'lib.DOUBLE'"
      // while the read `lib.DOUBLE` one line over resolved.
      if (std::string global = importedModuleGlobalFor(binding);
          !global.empty()) {
        mlir::Type stored = moduleGlobals.lookup(global);
        auto op = py::GlobalGetOp::create(builder, loc(*calleeNode), stored,
                                          builder.getStringAttr(global));
        markBoxedModuleGlobal(op);
        return emitCallableDispatch(expr, Value{op.getResult(), stored},
                                    emitCallOperands(expr));
      }
      auto generic = genericFunctions.find(binding);
      if (generic != genericFunctions.end())
        return emitGenericCall(expr, *calleeNode, generic->second);
      if (auto mono = monomorphicFunctions.find(binding);
          mono != monomorphicFunctions.end() &&
          mayArgumentSpecialize(expr, mono->second))
        return emitArgumentSpecializedCall(
            expr, *calleeNode, mono->second,
            emitBindingRef(*calleeNode, binding, *symbol));
      mlir::Type resultOverride;
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
        // ⭐ A CLASSMETHOD CALLED THROUGH AN INSTANCE IS THE CLASS'S CALL:
        // `b"x".fromhex("41")` is `bytes.fromhex("41")` and `{}.fromkeys(ks)`
        // is `dict.fromkeys(ks)`, as CPython binds a classmethod to the
        // instance's type. The receiver is emitted first because it still has
        // to happen, and the call is re-spelled as the class name -- the path
        // that already resolves. Without this the instance was offered to a
        // first parameter that is the class object, and the call was reported
        // as "has manifest method 'fromhex' but no signature that accepts
        // (str)".
        //
        // ⭐ And the class-level methods the emitter builds itself instead of
        // taking from the manifest -- `str.maketrans` (a staticmethod) and
        // `dict.fromkeys` -- which no signature describes, so they are named.
        //
        // ⛔ Only when the class name still names the class here: a program
        // that binds `bytes` to something else keeps the refusal rather than
        // calling whatever the name now holds.
        // ⛔ Not a class the program declares: the call below binds `cls` to
        // the receiver's class at run time, and its bare name is not even in
        // scope outside its module -- a computed-name dispatcher written into
        // the importer called `TestCase.setUpClass()` there and found no
        // `TestCase`.
        if (auto instance = mlir::dyn_cast_if_present<py::ContractType>(
                types.widenLiteral(types.inferExpr(receiverNode)));
            instance && !isSourceClassContract(instance) &&
            (types.isManifestClassMethod(instance, *methodName) ||
                         llvm::is_contained(
                             {std::pair<llvm::StringRef, llvm::StringRef>{
                                  "builtins.str", "maketrans"},
                              {"builtins.dict", "fromkeys"}},
                             std::pair<llvm::StringRef, llvm::StringRef>{
                                 instance.getContractName(), *methodName}))) {
          llvm::StringRef name = instance.getContractName();
          auto tail = name.rsplit('.').second;
          std::string spelling = tail.empty() ? name.str() : tail.str();
          // The builtin name either resolves to this class or is not bound at
          // all (`dict` is the emitter's, and only an unbound name is it).
          std::optional<mlir::Type> bound = types.lookupClass(spelling);
          auto named = mlir::dyn_cast_if_present<py::ContractType>(
              bound.value_or(mlir::Type()));
          bool namesTheClass = bound ? named && named.getContractName() == name
                                     : !types.lookupSymbol(spelling);
          if (namesTheClass && !values.count(spelling) &&
              !programBindsName(spelling)) {
            emitExpr(receiverNode);
            parser::NodePtr rewritten = parser::makeNode("Call", expr.range);
            parser::addField(
                *rewritten, "func",
                synth::attribute(synth::name(spelling, receiverNode->range),
                                 *methodName, calleeNode->range));
            for (llvm::StringRef field : {"args", "keywords"})
              if (const parser::Field *original =
                      parser::findField(expr, field))
                rewritten->fields.push_back(*original);
            synthesizedIteratorDefs.push_back(rewritten);
            return emitCall(*rewritten);
          }
        }
        // ⭐ `Exception.__init__(self, msg)` IS `super().__init__(msg)` when
        // the class named is the base the receiver derives from. Both are how
        // an exception subclass forwards its message, both appear in real
        // code, and only the second was answered: the first went looking for
        // `__init__` ON THE CLASS OBJECT and reported "type<Exception> has
        // manifest method '__init__' but no signature that accepts (E, str)",
        // which is the compiler saying it asked the wrong receiver.
        //
        // ⛔ The manifest bases only. A SOURCE base already works through the
        // unbound spelling below (`A.__init__(self, v)` binds no receiver and
        // passes it as the first argument); what has no method to bind is a
        // taxonomy class, whose `__init__` is the runtime's.
        if (*methodName == "__init__" && receiverNode) {
          std::string named = ast::qualifiedName(receiverNode);
          if (named.empty() && receiverNode->kind == "Name")
            named = std::string(ast::nameSpelling(*receiverNode));
          std::string baseContract = canonicalClassName(named);
          const auto *initArgs = ast::nodeList(expr, "args");
          bool taxonomyBase =
              !baseContract.empty() &&
              (py::exceptions::findByName(baseContract) ||
               py::exceptions::findByName(
                   py::contracts::manifestClassNameForContract(baseContract)));
          if (taxonomyBase && !classMros.count(baseContract) &&
              initArgs && !initArgs->empty() && initArgs->front() &&
              initArgs->front()->kind != "Starred") {
            Value self = emitExpr(initArgs->front().get());
            auto selfContract =
                mlir::dyn_cast_if_present<py::ContractType>(self.type);
            if (selfContract &&
                llvm::is_contained(classMro(selfContract.getContractName()),
                                   baseContract)) {
              parser::NodePtr forwarded =
                  parser::makeNode("Call", expr.range);
              parser::addField(*forwarded, "func", synth::name("__init__",
                                                               expr.range));
              parser::addField(
                  *forwarded, "args",
                  std::vector<parser::NodePtr>(std::next(initArgs->begin()),
                                               initArgs->end()));
              if (const parser::Field *keywords =
                      parser::findField(expr, "keywords"))
                forwarded->fields.push_back(*keywords);
              else
                parser::addField(*forwarded, "keywords",
                                 std::vector<parser::NodePtr>{});
              Value result = emitSuperExceptionInit(*forwarded, self,
                                                    baseContract);
              synthesizedIteratorDefs.push_back(std::move(forwarded));
              return result;
            }
          }
        }
        // ⭐ `str.upper(s)` IS `s.upper()`, and the receiver is the first
        // argument. Written by hand it is ordinary Python; reached through a
        // `map(str.upper, xs)` whose fast path re-spells the callable as a
        // call, it is how the idiom gets here at all. Both were refused with
        // "!py.type<builtins.str> does not provide manifest method 'upper'",
        // which is the compiler saying it looked on the class object.
        if (mlir::Type owner = types.manifestMethodReceiverContract(
                types.widenLiteral(types.inferExpr(receiverNode)),
                *methodName)) {
          const auto *unboundArgs = ast::nodeList(expr, "args");
          if (unboundArgs && !unboundArgs->empty() && unboundArgs->front() &&
              unboundArgs->front()->kind != "Starred") {
            if (std::optional<Value> text = tryEmitBaseExceptionText(
                    expr, owner, *methodName, unboundArgs->front()))
              return *text;
            // ⭐ The call names the class, so it is not virtual:
            // `int.__repr__(True)` is int's repr, "1", and `True.__repr__()`
            // printed "True". The first argument becomes the named class
            // first, the way CPython's slot wrapper takes it.
            // ⛔ Not when it already is that class: the plain re-spelling keeps
            // the receiver's NAME, which a structural mutator
            // (`list.append(xs, 1)`) rebinds.
            // ⭐ A class below the named one that declares the method itself
            // has a body the conversion keeps -- a bool converted to int is
            // still the bool object, whose repr is "True". bool's int VALUE is
            // the whole of it, so `int(x)` is int's receiver for the methods
            // CPython's bool overrides; the rest it inherits, and int's
            // `__str__` is object's, which asks the bool's repr again
            // (`int.__str__(False)` is "False"). Any other override has no
            // non-virtual route here, and calling it would be the override's
            // answer.
            parser::NodePtr receiverSpelling = unboundArgs->front();
            mlir::Type given =
                types.widenLiteral(types.inferExpr(receiverSpelling.get()));
            auto givenContract = mlir::dyn_cast_if_present<py::ContractType>(given);
            bool boolUnderInt =
                givenContract &&
                givenContract.getContractName() == "builtins.bool" &&
                mlir::cast<py::ContractType>(owner).getContractName() ==
                    "builtins.int";
            bool givenOverrides =
                given != owner && givenContract &&
                (boolUnderInt
                     ? llvm::is_contained(
                           {"__repr__", "__and__", "__or__", "__xor__",
                            "__rand__", "__ror__", "__rxor__", "__invert__"},
                           *methodName)
                 : isSourceClassContract(givenContract)
                     ? lookupClassMethod(given, *methodName).has_value()
                     : static_cast<bool>(types.manifestMethodReceiverContract(
                           types.typeObject(given), *methodName)));
            bool callThroughGiven = boolUnderInt && !givenOverrides;
            if (givenOverrides) {
              if (boolUnderInt) {
                receiverSpelling = synth::call(
                    synth::name("int", receiverSpelling->range),
                    {receiverSpelling}, receiverSpelling->range);
                synthesizedIteratorDefs.push_back(receiverSpelling);
              } else {
                diagnostics.push_back(parser::Diagnostic{
                    parser::Severity::Error, expr.range.start,
                    typeText(owner) + "." + std::string(*methodName) +
                        "(x) calls that class's own method, and x is " +
                        typeText(given) + ", which declares its own; only x." +
                        std::string(*methodName) +
                        "() (its own method) can be called here"});
                return emitNone(expr);
              }
            }
            std::optional<std::string> coercedName;
            if (!callThroughGiven &&
                types.widenLiteral(types.inferExpr(receiverSpelling.get())) !=
                    owner) {
              Value subject = coerceValue(
                  emitExpr(receiverSpelling.get()), owner, *receiverSpelling);
              coercedName =
                  "__ly_unbound" + std::to_string(++syntheticFunctionCounter);
              values[*coercedName] = subject;
              types.bindSymbol(*coercedName, subject.type);
              receiverSpelling =
                  synth::name(*coercedName, receiverSpelling->range);
            }
            auto forget = llvm::scope_exit([&] {
              if (coercedName)
                values.erase(*coercedName);
            });
            std::vector<parser::NodePtr> rest(std::next(unboundArgs->begin()),
                                              unboundArgs->end());
            parser::NodePtr rewritten = parser::makeNode("Call", expr.range);
            parser::addField(*rewritten, "func",
                             synth::attribute(std::move(receiverSpelling),
                                              *methodName, calleeNode->range));
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
        // ⭐ A CLASS ATTRIBUTE THAT HOLDS A FUNCTION IS CALLED, NOT
        // DISPATCHED. `C.V(1)` for `V: Callable[[int], int] = lambda n: n + 1`
        // in a class body went looking for a METHOD named V on the type object
        // and reported "type<C> does not provide manifest method 'V'" -- the
        // compiler saying it asked the wrong table. The attribute is in the
        // class's static-attribute map, and reading it (`f = C.V`) is what the
        // slot repair made work; this is the call spelling of the same read.
        //
        // ⛔ Before the receiver is emitted, so the attribute read below is the
        // only emission of it. A type object is pure, but two emissions of the
        // same node is how a bound-method path grows a second receiver.
        //
        // ⛔ AND `type(x)` NAMES A CLASS HERE TOO, which the inference alone
        // does not say: that fold happens at EMIT time, so asking `inferExpr`
        // what `type(c)` is answers nothing and `type(c).V(1)` fell through to
        // the method table while `t = type(c); t.V(1)` -- the same two steps
        // written apart -- resolved.
        auto staticAttrClassName = [&]() -> std::string {
          if (auto typeObject = mlir::dyn_cast_if_present<py::TypeType>(
                  types.widenLiteral(types.inferExpr(receiverNode))))
            if (auto instance = mlir::dyn_cast_if_present<py::ContractType>(
                    typeObject.getInstanceType()))
              return instance.getContractName().str();
          if (receiverNode->kind == "Call" &&
              callsUnshadowedBuiltin(ast::node(*receiverNode, "func"), "type"))
            if (const auto *typeArgs = ast::nodeList(*receiverNode, "args");
                typeArgs && typeArgs->size() == 1 && typeArgs->front())
              if (auto subject = mlir::dyn_cast_if_present<py::ContractType>(
                      types.widenLiteral(
                          types.inferExpr(typeArgs->front().get()))))
                return subject.getContractName().str();
          return {};
        };
        if (std::string staticAttrClass = staticAttrClassName();
            !staticAttrClass.empty())
          if (std::optional<mlir::Type> attrType =
                  types.lookupClassStaticAttrType(staticAttrClass, *methodName))
            if (mlir::isa_and_nonnull<py::CallableType>(
                    types.widenLiteral(*attrType))) {
              Value callee = emitExpr(calleeNode);
              CallOperands operands = emitCallOperands(expr);
              return emitCallableDispatch(expr, callee, operands);
            }
        Value receiver = emitExpr(receiverNode);
        if (std::optional<Value> dispatched =
                tryEmitUnionMethodCall(expr, receiver, *methodName))
          return *dispatched;
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
              // The arguments are emitted at the parameters' types, as any
              // call's are: a tuple display passed to `xs: tuple[int, ...]`
              // is built homogeneous, and its fixed-length type matched no
              // signature here.
              CallOperands operands =
                  emitCallOperands(expr, {receiver},
                                   /*includeAstArguments=*/true, directCallable);
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
        // ⭐ AND A CLASS ATTRIBUTE HOLDING A CLASS: unittest's
        // `self.resultclass(stream, ...)`, which an instance may shadow. Read
        // through the instance it is the class, unbound -- a class is not a
        // descriptor -- so it is called the same way.
        if (!calleeFieldType)
          if (auto receiverContract =
                  mlir::dyn_cast_if_present<py::ContractType>(
                      types.widenLiteral(receiver.type)))
            if (std::optional<mlir::Type> attrType =
                    types.lookupClassStaticAttrType(
                        receiverContract.getContractName(), *methodName);
                attrType &&
                mlir::isa<py::TypeType>(types.widenLiteral(*attrType)))
              calleeFieldType = attrType;
        // A type object in a field is called to CONSTRUCT, so it takes the
        // same route: the callee is the value, not a method of the receiver.
        auto callsAValue = [](mlir::Type type) {
          return mlir::isa<py::CallableType, py::TypeType>(type);
        };
        if (calleeFieldType) {
          mlir::Type widenedField = types.widenLiteral(*calleeFieldType);
          // ⭐ AN OPTIONAL ONE TOO, once a test has narrowed it:
          // `if self._setUpFunc is not None: self._setUpFunc()`. The read is
          // the narrowed value, which is what decides below.
          bool optionalValue = false;
          if (auto unionType = mlir::dyn_cast<py::UnionType>(widenedField)) {
            optionalValue = true;
            for (mlir::Type member : unionType.getMemberTypes())
              if (member != types.none() &&
                  !callsAValue(types.widenLiteral(member)))
                optionalValue = false;
          }
          if (callsAValue(widenedField) || optionalValue) {
            // ⛔ Not the raw field read when the receiver is a name: the
            // attribute expression is what knows a narrowing of `self._f` and
            // an instance's shadow of a class attribute, and reading a name
            // again has no effect of its own.
            const parser::Node *receiverNode = ast::node(*calleeNode, "value");
            std::optional<Value> fieldValue =
                receiverNode && receiverNode->kind == "Name"
                    ? std::optional<Value>(emitExpr(calleeNode))
                    : emitValueAttribute(*calleeNode, receiver, *methodName);
            if (fieldValue) {
              mlir::Type held = types.widenLiteral(fieldValue->type);
              if (callsAValue(held))
                return emitCallableDispatch(
                    expr, *fieldValue,
                    emitCallOperands(expr, {}, /*includeAstArguments=*/true,
                                     mlir::dyn_cast<py::CallableType>(held)));
              diagnostics.push_back(parser::Diagnostic{
                  parser::Severity::Error, expr.range.start,
                  "'" + std::string(*methodName) +
                      "' may hold None here, and calling None raises "
                      "TypeError: test it against None before the call"});
              return emitNone(expr);
            }
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
        // ⛔ AND A CLASS ATTRIBUTE HOLDING A FUNCTION, READ THROUGH AN
        // INSTANCE, which is the descriptor corner and not a missing object
        // default. `c.V(1)` binds the receiver in CPython -- that is what
        // `@staticmethod` exists to opt out of -- and the class spelling
        // `C.V(c, 1)` is the same call written out. Saying "'C' inherits
        // builtins.object.V" for it names a member of object that does not
        // exist.
        if (auto receiverContract = mlir::dyn_cast_if_present<py::ContractType>(
                types.widenLiteral(receiver.type)))
          if (std::optional<mlir::Type> attrType =
                  types.lookupClassStaticAttrType(
                      receiverContract.getContractName(), *methodName))
            if (mlir::isa_and_nonnull<py::CallableType>(
                    types.widenLiteral(*attrType))) {
              diagnostics.push_back(parser::Diagnostic{
                  parser::Severity::Error, expr.range.start,
                  "'" + std::string(*methodName) +
                      "' is a class attribute holding a function, and reading "
                      "it through an instance binds the receiver, which is not "
                      "supported; call it through the class (" +
                      py::contracts::displayClassNameForContract(
                          receiverContract.getContractName()) +
                      "." + std::string(*methodName) +
                      "(obj, ...)) or declare it @staticmethod"});
              return emitNone(expr);
            }
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
              // ⭐ AN ITERATOR KNOWN ONLY BY ITS PROTOCOL IS A GENERATOR TOO.
              // A function annotated `-> Iterator[str]` that returns a
              // generator it did not create itself -- difflib's `ndiff` returns
              // `Differ().compare(a, b)` -- reads as the protocol, not as
              // `types.GeneratorType`, so `"".join(ndiff(a, b))` skipped the
              // rewrite below and reached the ABI ("cannot adapt
              // types.GeneratorType to runtime input 2 of builtins.str.join"),
              // while `list(ndiff(a, b))` worked.
              // ⛔ Only into a parameter declared `Iterable`: one that takes
              // the iterator itself keeps it. And only an `Iterator` or a
              // `Generator`: an `Iterable` -- a bound `"-".join`'s own
              // parameter -- is specialized from what is passed, a list
              // more often than not, which the copy would only duplicate.
              if (argument && argument->kind != "Starred" &&
                  index < declaredTypes.size())
                if (auto actualProtocol =
                        mlir::dyn_cast_if_present<py::ProtocolType>(
                            types.widenLiteral(
                                types.inferExpr(argument.get())))) {
                  auto declaredProtocol =
                      mlir::dyn_cast_if_present<py::ProtocolType>(
                          declaredTypes[index]);
                  llvm::StringRef name = actualProtocol.getProtocolName();
                  if (declaredProtocol &&
                      declaredProtocol.getProtocolName() == "Iterable" &&
                      (name == "Iterator" || name == "Generator")) {
                    materialized.push_back(synth::call(
                        synth::name("list", argument->range),
                        std::vector<parser::NodePtr>{argument},
                        argument->range));
                    anyMaterialized = true;
                    continue;
                  }
                }
              if (argument && argument->kind != "Starred")
                if (auto actual = mlir::dyn_cast_if_present<py::ContractType>(
                        types.widenLiteral(types.inferExpr(argument.get())));
                    actual) {
                  // ⛔ Not when the parameter takes the generator ITSELF.
                  // `tasks.append(task)` declares the list's element type, and
                  // materializing there stored a list where the program stored
                  // a generator -- refused only because the element types
                  // disagreed. A consumer declares `Iterable`; a parameter of
                  // no declared type keeps the rewrite.
                  mlir::Type declaredHere = index < declaredTypes.size()
                                                ? declaredTypes[index]
                                                : mlir::Type();
                  auto declaredProtocol =
                      mlir::dyn_cast_if_present<py::ProtocolType>(declaredHere);
                  bool generator =
                      actual.getContractName() == "types.GeneratorType" &&
                      (!declaredHere ||
                       (declaredProtocol &&
                        declaredProtocol.getProtocolName() == "Iterable"));
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
                  //
                  // ⛔ That exception is keyed on the contract NAME, which says
                  // "peer container" for every contract but one: `str` is the
                  // only one that iterates to ITSELF, so `"-".join(s)` read as
                  // a peer and skipped the materialization that `set`,
                  // `frozenset`, a generator, `map`, `filter` and a dict view
                  // all get -- and reached the ABI as "cannot adapt builtins.str
                  // to runtime input 2 of builtins.str.join". A self-iterating
                  // argument is a sequence OF the receiver's contract, not one
                  // of it, so the exception does not apply to it.
                  // ⭐ THE CONTAINER TO BUILD IS THE RECEIVER'S OWN. The
                  // runtime implements the peer-container case, so what has to
                  // be materialized is a peer: `s.update([2, 3])` needs a SET
                  // beside `s`, and a list there only moves the refusal to the
                  // ABI ("cannot adapt builtins.list to runtime input 1 of
                  // builtins.set.update"). Asked before the guard below, which
                  // is "the argument is already what we would build".
                  llvm::StringRef builder = "list";
                  llvm::StringRef builderContract = "builtins.list";
                  if (auto ownContract =
                          mlir::dyn_cast_if_present<py::ContractType>(
                              types.widenLiteral(receiver.type))) {
                    llvm::StringRef ownName = ownContract.getContractName();
                    if (ownName == "builtins.set") {
                      builder = "set";
                      builderContract = "builtins.set";
                    } else if (ownName == "builtins.dict") {
                      builder = "dict";
                      builderContract = "builtins.dict";
                    }
                  }
                  bool wantsList = false;
                  if (!generator && index < declaredTypes.size() &&
                      actual.getContractName() != builderContract)
                    if (auto declared = mlir::dyn_cast_if_present<py::ProtocolType>(
                            declaredTypes[index])) {
                      auto receiverContract =
                          mlir::dyn_cast_if_present<py::ContractType>(
                              types.widenLiteral(receiver.type));
                      mlir::Type element =
                          types.iterationElementType(argument.get());
                      bool selfIterating =
                          element == mlir::Type(actual) ||
                          types.widenLiteral(element) == mlir::Type(actual);
                      wantsList =
                          declared.getProtocolName() == "Iterable" && element &&
                          (!receiverContract || selfIterating ||
                           receiverContract.getContractName() !=
                               actual.getContractName());
                    }
                  if (generator || wantsList) {
                    materialized.push_back(synth::call(
                        synth::name(builder.str(), argument->range),
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
        // ⭐ A LAMBDA ARGUMENT IS TYPED BY THE PARAMETER IT REACHES, as it is
        // in a call to a function: `fs.append(lambda: value())` on a
        // `list[Callable[[], object]]` was refused as "lambda requires a
        // Callable annotation", since a lambda whose body answers `object`
        // has no type of its own to offer. The method's parameters on this
        // receiver -- `append(x: Callable[[], object])` here -- are that
        // expectation.
        // ⭐ And a union of classes, which the parameter's class views
        // (`es.append(pick())` into a `list[Exception]`).
        // ⛔ Only for those two: every other argument already has the type it
        // is passed as, and its emission is not changed by this.
        py::CallableType lambdaExpectations;
        if (const auto *callArgs = ast::nodeList(*methodCallNode, "args");
            callArgs &&
            llvm::any_of(*callArgs, [&](const parser::NodePtr &argument) {
              return argument &&
                     (argument->kind == "Lambda" ||
                      mlir::isa_and_nonnull<py::UnionType>(
                          types.inferExpr(argument.get())));
            }) &&
            llvm::none_of(*callArgs, [](const parser::NodePtr &argument) {
              return !argument || argument->kind == "Starred";
            }))
          if (std::optional<py::CallableType> expectations =
                  types.manifestMethodCallExpectations(
                      receiver.type, *methodName,
                      static_cast<unsigned>(callArgs->size())))
            lambdaExpectations = *expectations;
        CallOperands operands = emitCallOperands(
            *methodCallNode, {}, /*includeAstArguments=*/true,
            lambdaExpectations);
        if (!operands.valid) {
          diagnostics.push_back(parser::Diagnostic{
              parser::Severity::Error, expr.range.start,
              operands.failureReason});
          return emitNone(expr);
        }
        CallInferenceResult inference = types.inferMethodCallWithEvidence(
            receiver.type, *methodName, operands.positionalTypes,
            operands.keywordTypes);
        if (!requireStaticEvidence(expr, inference))
          return emitNone(expr);
        if (isJsHostValueType(receiver.type))
          wrapJsCallbackArguments(expr, inference, operands);
        Value posPack =
            emitPack(operands.positional, operands.positionalUnpacked);
        Value namePack = emitPack(operands.keywordNames);
        Value valuePack = emitPack(operands.keywordValues);
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
        if (isJsHostValueType(receiver.type))
          return adaptJsHostResult(expr, op,
                                   Value{op.getResults().front(), resultType});
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
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, calleeNode->range.start,
          unresolvedNameMessage(name)});
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
    // ⭐ A CLASS OF ANOTHER MODULE has no bare name here: `c(verbosity=2)`
    // for `c: type[lib.Runner]` found no `Runner` and fell through to a
    // call of the type object, which the lowering cannot bind a keyword to
    // ("kw names lowering is not keyword-aware yet"). It is spelled by a
    // reserved name bound to the class, which no source identifier is.
    if (!spelling.empty() && types.lookupClass(spelling) != instance)
      if (auto contract = mlir::dyn_cast_if_present<py::ContractType>(instance);
          contract && classMros.count(contract.getContractName())) {
        spelling = "__lyclass$" + contract.getContractName().str();
        types.bindClass(spelling, instance);
      }
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
        // ⭐ `x(...)` OVER AN OVERRIDDEN `__call__` IS A DISPATCH TOO. This
        // site went to the refusal through `resolveClassDunder`, so a
        // base-typed callable object was rejected while every other dunder on
        // the same hierarchy dispatched.
        //
        // ⛔ Through the AST entry point rather than the values one, because
        // this call node carries the arguments: that path reads the keywords
        // as well (`x(n=4)` is an ordinary spelling of a call) and asks the
        // dispatcher BEFORE emitting anything, which is what stops
        // `tryEmitClassDunderCall` below from evaluating each argument twice.
        if (dispatchIsUnresolvable(receiver, "__call__", calleeNode,
                                   /*throughSuper=*/false))
          if (std::optional<Value> dispatched = tryEmitVirtualDispatch(
                  expr, *calleeNode, calleeNode, receiver, "__call__"))
            return *dispatched;
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
  for (auto [index, node] : llvm::enumerate(nodes))
    if (lowerRungParameter(types, declared[index],
                           types.widenLiteral(types.inferExpr(node))))
      return true;
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
  // The decision inference types the call by, so the call's type is the
  // value's (TypeSystem::argumentRungSpecialization).
  std::optional<FunctionSignature> chosen =
      types.argumentRungSpecialization(*info.node, info.signature,
                                       suppliedTypes);
  if (!chosen)
    return ordinary();
  const FunctionSignature &specialized = *chosen;

  auto memoized = info.specializations.find(specialized.publicCallable);
  std::string symbol;
  if (memoized != info.specializations.end()) {
    symbol = memoized->second;
  } else {
    // The same divergence backstop the generic specializer carries: a body
    // that calls itself one rung down would otherwise re-enter forever.
    if (info.emittingSpecializations >= 32)
      return ordinary();
    symbol = (llvm::Twine(info.symbolBase) + "$arg" +
              llvm::Twine(static_cast<unsigned>(info.specializations.size())))
                 .str();
    // Memoize BEFORE the body, so a recursive call at the same ground
    // signature resolves to this symbol instead of specializing again.
    info.specializations[specialized.publicCallable] = symbol;
    ++info.emittingSpecializations;
    auto leave = llvm::scope_exit([&] { --info.emittingSpecializations; });
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

// `factory(args)(applied)` with a generic factory its own arguments do not
// solve (TypeSystem::solveCurriedGenericCall): the factory is specialized at
// the solution the applied arguments give, called, and its result applied.
std::optional<Value>
ModuleEmitter::tryEmitCurriedGenericCall(const parser::Node &expr,
                                         const parser::Node &factoryCall) {
  const parser::Node *factoryNode = ast::node(factoryCall, "func");
  if (!factoryNode)
    return std::nullopt;
  GenericFunctionInfo *generic = nullptr;
  if (factoryNode->kind == "Name") {
    generic = lookupGenericFunction(ast::nameSpelling(*factoryNode));
  } else if (factoryNode->kind == "Attribute") {
    std::string qualified = ast::qualifiedName(factoryNode);
    if (!qualified.empty())
      generic = lookupGenericFunction(qualified);
  }
  if (!generic)
    return std::nullopt;
  auto positionalOnly = [](const parser::Node &call) {
    const auto *keywords = ast::nodeList(call, "keywords");
    const auto *args = ast::nodeList(call, "args");
    return (!keywords || keywords->empty()) &&
           (!args || llvm::none_of(*args, [](const parser::NodePtr &arg) {
              return !arg || arg->kind == "Starred";
            }));
  };
  if (!positionalOnly(factoryCall) || !positionalOnly(expr))
    return std::nullopt;
  auto argumentTypes = [&](const parser::Node &call) {
    llvm::SmallVector<mlir::Type, 4> out;
    if (const auto *args = ast::nodeList(call, "args"))
      for (const parser::NodePtr &arg : *args)
        out.push_back(types.inferExpr(arg.get()));
    return out;
  };
  std::optional<TypeSystem::CurriedCallSolution> solved =
      types.solveCurriedGenericCall(generic->signature.publicCallable,
                                    argumentTypes(factoryCall),
                                    argumentTypes(expr));
  if (!solved)
    return std::nullopt;
  std::optional<std::pair<std::string, py::CallableType>> specialization =
      ensureGenericSpecialization(factoryCall, *generic, solved->factory);
  if (!specialization)
    return emitNone(expr);
  Value factory = emitBindingRef(*factoryNode, specialization->first,
                                 specialization->second);
  Value made = emitCallableDispatch(
      factoryCall, factory,
      emitCallOperands(factoryCall, {}, /*includeAstArguments=*/true,
                       specialization->second));
  return emitCallableDispatch(
      expr, made,
      emitCallOperands(expr, {}, /*includeAstArguments=*/true,
                       solved->applied));
}

// ⭐ AN OMITTED DEFAULT IS AN ARGUMENT TOO, and the only one that can solve a
// type parameter nothing else names: `def pick[E](expected: type[E] =
// BaseException)` called as `pick()` passes BaseException, so E is
// BaseException -- what a type checker infers, and what the body receives.
// Both the generic function and the generic method were refused ("this call
// does not determine the type arguments"), and a computed-name call writes
// such a zero-argument call for every method of the class.
// ⛔ Asked only when the call alone does not solve, so a call that already
// did is specialized exactly as before; and only for a parameter whose type
// still names a type parameter. A default whose type cannot be read here --
// a name of the defining module, seen from another -- adds nothing.
llvm::SmallVector<CallKeywordType, 2> ModuleEmitter::omittedDefaultArgumentTypes(
    const parser::Node &def, const FunctionSignature &sig,
    unsigned positionalGiven, llvm::ArrayRef<CallKeywordType> keywords) {
  llvm::SmallVector<CallKeywordType, 2> out;
  const parser::Node *arguments = ast::node(def, "args");
  if (!arguments)
    return out;
  auto given = [&](llvm::StringRef name) {
    return llvm::any_of(keywords, [&](const CallKeywordType &keyword) {
      return keyword.name == name;
    });
  };
  auto add = [&](llvm::StringRef name, mlir::Type declared,
                 const parser::Node *defaultNode) {
    if (!defaultNode || !declared || given(name) ||
        unboundStaticParameterCount(declared) == 0)
      return;
    mlir::Type defaultType = types.widenLiteral(types.inferExpr(defaultNode));
    if (!defaultType || defaultType == types.object())
      return;
    out.push_back(CallKeywordType{name.str(), defaultType});
  };
  llvm::SmallVector<const parser::Node *, 8> positional;
  for (llvm::StringRef field : {"posonlyargs", "args"})
    if (const auto *list = ast::nodeList(*arguments, field))
      for (const parser::NodePtr &argument : *list)
        positional.push_back(argument.get());
  const auto *defaults = ast::nodeList(*arguments, "defaults");
  std::size_t firstDefault =
      defaults && defaults->size() <= positional.size()
          ? positional.size() - defaults->size()
          : positional.size();
  for (std::size_t index = std::max<std::size_t>(positionalGiven, firstDefault);
       index < positional.size(); ++index) {
    // A positional-only parameter cannot be named, and the solver takes the
    // default as a keyword.
    if (index < sig.positionalOnlyCount || index >= sig.positionalNames.size() ||
        index >= sig.positionalTypes.size() || !positional[index])
      continue;
    add(sig.positionalNames[index], sig.positionalTypes[index],
        (*defaults)[index - firstDefault].get());
  }
  const auto *kwDefaults = ast::nodeList(*arguments, "kw_defaults");
  for (std::size_t index = 0; index < sig.kwOnlyNames.size(); ++index)
    if (kwDefaults && index < kwDefaults->size() &&
        index < sig.kwOnlyTypes.size())
      add(sig.kwOnlyNames[index], sig.kwOnlyTypes[index],
          (*kwDefaults)[index].get());
  return out;
}

Value ModuleEmitter::emitGenericCall(const parser::Node &expr,
                                     const parser::Node &calleeNode,
                                     GenericFunctionInfo &generic) {
  // The generic contract still distributes as the argument expectation: its
  // ground formals propagate, and expectedFor skips the type-parameter ones.
  CallOperands operands =
      emitCallOperands(expr, {}, /*includeAstArguments=*/true,
                       generic.signature.publicCallable);
  bool paramSpec = forwardsParamSpec(generic.signature);
  // ⭐ A SEQUENCE OF UNKNOWN LENGTH OR A MAPPING OF UNKNOWN KEYS solves the
  // type parameters like any argument -- `*xs` with xs a `list[int]` binds
  // the T of `*args: T` -- and the specialization takes them into its *args
  // and **kwargs the way any function this program defines does
  // (emitCallableDispatch). They were refused here as passed to something
  // that is not such a function.
  bool onlyDynamic = !operands.valid && !operands.invalidBeyondDynamic &&
                     (operands.dynamicTail || operands.dynamicMapping);
  if (!operands.valid && (paramSpec || !onlyDynamic)) {
    // ⛔ A ParamSpec forwarder is specialized at the call's shape, which a
    // sequence or mapping whose members only the run knows does not have.
    std::string reason =
        paramSpec && onlyDynamic
            ? "a starred sequence of unknown length or a `**` mapping of "
              "unknown keys cannot fill `*args: P.args` or `**kwargs: "
              "P.kwargs`, whose members are parameters of their own in the "
              "body each call is compiled to: spell the arguments out"
            : operands.failureReason;
    diagnostics.push_back(
        parser::Diagnostic{parser::Severity::Error, expr.range.start, reason});
    return emitNone(expr);
  }
  // An empty container literal is solved as nothing (emptyLiteralSolvingType)
  // and adopts what its parameter became below.
  // ⛔ Unless that leaves T unsolved or names the marker (`same([], [])`):
  // then it is solved as it was emitted, `list[object]`.
  std::optional<EmptyLiteralArguments> emptyLiterals =
      operands.dynamicTail || operands.dynamicMapping
          ? std::nullopt
          : emptyLiteralArgumentsOf(expr);
  if (emptyLiterals &&
      emptyLiterals->positional.size() != operands.positionalTypes.size())
    emptyLiterals.reset();
  CallInferenceResult inference;
  if (emptyLiterals) {
    llvm::SmallVector<mlir::Type, 8> solvingTypes(operands.positionalTypes);
    llvm::SmallVector<CallKeywordType, 4> solvingKeywords(
        operands.keywordTypes.begin(), operands.keywordTypes.end());
    bool marked = false;
    for (auto [index, type] : llvm::enumerate(solvingTypes))
      if (emptyLiterals->positional[index])
        if (mlir::Type marker = emptyLiteralSolvingType(type, types)) {
          type = marker;
          marked = true;
        }
    for (CallKeywordType &keyword : solvingKeywords)
      if (emptyLiterals->keywords.contains(keyword.name))
        if (mlir::Type marker = emptyLiteralSolvingType(keyword.type, types)) {
          keyword.type = marker;
          marked = true;
        }
    if (marked) {
      inference = types.inferCallWithEvidence(generic.signature.publicCallable,
                                              solvingTypes, solvingKeywords);
      if (inference &&
          (mentionsEmptyLiteralMarker(inference.resultType) ||
           mentionsEmptyLiteralMarker(inference.evidence.callableContract)))
        inference = CallInferenceResult();
    }
    if (!inference)
      emptyLiterals.reset();
  }
  if (!inference)
    inference = types.inferCallWithEvidence(generic.signature.publicCallable,
                                            operands.positionalTypes,
                                            operands.keywordTypes);
  if (!inference && generic.node && !operands.dynamicTail) {
    llvm::SmallVector<CallKeywordType, 2> defaults = omittedDefaultArgumentTypes(
        *generic.node, generic.signature,
        static_cast<unsigned>(operands.positionalTypes.size()),
        operands.keywordTypes);
    if (!defaults.empty()) {
      llvm::SmallVector<CallKeywordType, 4> withDefaults(
          operands.keywordTypes.begin(), operands.keywordTypes.end());
      withDefaults.append(defaults.begin(), defaults.end());
      inference = types.inferCallWithEvidence(
          generic.signature.publicCallable, operands.positionalTypes,
          withDefaults);
    }
  }
  // ⭐ A TYPE PARAMETER ONLY THE RESULT NAMES IS SOLVED BY WHAT THE RESULT IS
  // WANTED AS, where that is known: `return skipIf(not condition, reason)` in
  // a `skipUnless` specialized at a test class wants the decorator of that
  // class, which is what its return annotation says. A factory that hands on
  // another factory's result was refused ("generic in a type parameter no
  // argument names") although its own signature names the type.
  if (!inference && expectedResultCall == &expr && expectedResultType)
    if (auto declared = mlir::dyn_cast_if_present<py::CallableType>(
            generic.signature.publicCallable);
        declared && declared.getResultTypes().size() == 1) {
      TypeBindingMap bindings;
      if (bindExpectedType(types, declared.getResultTypes().front(),
                           expectedResultType, bindings) &&
          !bindings.empty()) {
        mlir::Type solved = substituteType(types, declared, bindings);
        if (solved && unboundStaticParameterCount(solved) == 0)
          inference = types.inferCallWithEvidence(
              solved, operands.positionalTypes, operands.keywordTypes);
      }
    }
  // A type parameter only the result names is solved by what the result is
  // applied to, which a held factory (`d = skip(r)`) has not been yet.
  if (!inference)
    if (auto declared = mlir::dyn_cast_if_present<py::CallableType>(
            generic.signature.publicCallable)) {
      llvm::SmallVector<mlir::Type, 8> parameterTypes(
          declared.getPositionalTypes().begin(),
          declared.getPositionalTypes().end());
      parameterTypes.append(declared.getKwOnlyTypes().begin(),
                            declared.getKwOnlyTypes().end());
      // `*items: T` names T too: its arguments are what solve it.
      if (declared.hasVararg())
        parameterTypes.push_back(declared.getVarargType());
      bool parametersGround = llvm::all_of(parameterTypes, [](mlir::Type t) {
        return unboundStaticParameterCount(t) == 0;
      });
      if (parametersGround && !declared.getResultTypes().empty() &&
          unboundStaticParameterCount(declared.getResultTypes().front()) != 0) {
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, expr.range.start,
            "this call's result is generic in a type parameter no argument "
            "names: apply the result where it is made -- `" +
                ast::qualifiedName(&calleeNode) +
                "(...)(f)`, as a decorator does -- which is what solves it"});
        return emitNone(expr);
      }
    }
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
  std::optional<ParamSpecCallShape> shape;
  if (paramSpec)
    shape = paramSpecCallShape(generic.signature, operands);
  if (generic.node && !operands.dynamicTail && !operands.dynamicMapping)
    if (std::string mismatch = genericOmittedDefaultMismatch(
            *generic.node, resolved,
            static_cast<unsigned>(operands.positionalTypes.size()),
            operands.keywordTypes);
        !mismatch.empty()) {
      diagnostics.push_back(parser::Diagnostic{parser::Severity::Error,
                                               expr.range.start, mismatch});
      return emitNone(expr);
    }
  llvm::SmallVector<bool, 4> forwarded;
  for (const Value &argument : operands.positional)
    forwarded.push_back(needsCallableForwarder(argument));
  llvm::SmallVector<bool, 4> outerForwarded =
      std::exchange(paramSpecForwarderArguments, forwarded);
  auto restoreForwarded = llvm::scope_exit(
      [&] { paramSpecForwarderArguments = std::move(outerForwarded); });
  std::optional<std::pair<std::string, py::CallableType>> specialization =
      ensureGenericSpecialization(expr, generic, resolved,
                                  shape ? &*shape : nullptr);
  if (!specialization)
    return emitNone(expr);
  llvm::ArrayRef<mlir::Type> parameterTypes =
      specialization->second.getPositionalTypes();
  const auto *argumentNodes = ast::nodeList(expr, "args");
  if (argumentNodes && argumentNodes->size() != operands.positional.size())
    argumentNodes = nullptr;
  for (auto [index, argument] : llvm::enumerate(operands.positional))
    if (emptyLiterals && emptyLiterals->positional[index] &&
        index < parameterTypes.size()) {
      argument = adoptDeclaredContainer(argument, parameterTypes[index], expr);
      operands.positionalTypes[index] = argument.type;
    } else if (argumentNodes && index < parameterTypes.size()) {
      argument = adoptSolvedDisplay(argument, parameterTypes[index],
                                    (*argumentNodes)[index].get());
      operands.positionalTypes[index] = argument.type;
    }
  // A builtin or a class handed to a generic callable parameter becomes its
  // forwarder only now: the arguments were emitted against `Callable[[A],
  // R]` -- or the erased `Callable[..., R]` of a `Callable[P, R]` -- and the
  // specialization is what gave it parameters.
  for (auto [index, argument] : llvm::enumerate(operands.positional)) {
    auto target = index < parameterTypes.size()
                      ? mlir::dyn_cast_if_present<py::CallableType>(
                            parameterTypes[index])
                      : py::CallableType();
    if (!target)
      continue;
    std::optional<Value> wrapped =
        tryWrapBuiltinFunctionValue(argument, target, expr);
    if (!wrapped)
      wrapped = tryWrapClassValue(argument, target, expr);
    if (wrapped) {
      argument = *wrapped;
      operands.positionalTypes[index] = argument.type;
    }
  }
  Value callee = emitBindingRef(calleeNode, specialization->first,
                                specialization->second);
  return emitCallableDispatch(expr, callee, operands);
}

// ⭐ `unittest.main()` RUNS ITS MODULE'S FIXTURES, which CPython reads off
// `sys.modules[module]` -- `setUpModule` before the module's first test and
// `tearDownModule` after its last. A module is not a value here, so the
// functions are found where the call is written: the calling module's own
// top-level `setUpModule` / `tearDownModule`, passed to TestProgram as the
// keywords it registers them under (unittest/main.py).
// ⛔ Only the main module's: it is the one `unittest.main()`'s default
// `module='__main__'` names, and an imported module's call would register
// its fixtures under that name too.
std::optional<Value>
ModuleEmitter::tryEmitTestProgramWithModuleFixtures(const parser::Node &expr) {
  const parser::Node *calleeNode = ast::node(expr, "func");
  if (!calleeNode || sourceName != mainSourceName)
    return std::nullopt;
  auto classValue =
      mlir::dyn_cast_if_present<py::TypeType>(types.inferExpr(calleeNode));
  auto program = classValue ? mlir::dyn_cast_if_present<py::ContractType>(
                                  classValue.getInstanceType())
                            : py::ContractType();
  if (!program || program.getContractName() != "unittest.main.TestProgram")
    return std::nullopt;
  std::vector<parser::NodePtr> keywords;
  if (const auto *given = ast::nodeList(expr, "keywords"))
    for (const parser::NodePtr &keyword : *given) {
      if (keyword && ast::string(*keyword, "arg").value_or("") ==
                         std::string_view("_setUpModule"))
        return std::nullopt; // already passed: this is the rewritten call
      keywords.push_back(keyword);
    }
  bool added = false;
  for (llvm::StringRef fixture : {"setUpModule", "tearDownModule"})
    if (moduleFunctionNames.contains(fixture)) {
      keywords.push_back(synth::keyword(
          ("_" + fixture).str(), synth::name(fixture, expr.range),
          expr.range));
      added = true;
    }
  if (!added)
    return std::nullopt;
  if (llvm::none_of(keywords, [](const parser::NodePtr &keyword) {
        return keyword && ast::string(*keyword, "arg").value_or("") ==
                              std::string_view("_setUpModule");
      }))
    keywords.push_back(synth::keyword(
        "_setUpModule", synth::name("None", expr.range), expr.range));
  const parser::Field *func = parser::findField(expr, "func");
  std::vector<parser::NodePtr> arguments;
  if (const auto *given = ast::nodeList(expr, "args"))
    arguments = *given;
  parser::NodePtr call = synth::callWithKeywords(
      std::get<parser::NodePtr>(func->value), std::move(arguments),
      std::move(keywords), expr.range);
  synthesizedIteratorDefs.push_back(call);
  return emitExpr(call.get());
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
  if (!calleeNode || calleeNode->kind != "Name")
    return false;
  llvm::StringRef spelling = ast::nameSpelling(*calleeNode);
  // ⭐ A MODULE-LEVEL ALIAS OF A BUILTIN IS THAT BUILTIN'S CALL. Every builtin
  // handled here is keyed on the SPELLING, so `Text = str` then `Text(v)`
  // reached none of them: it fell to the class-instantiation path and said
  // "builtins.str has manifest method '__init__' but no signature that accepts
  // ...", a sentence about str for a program that never named it. `str(v)` on
  // the same line is the `__str__` dispatch it wanted.
  //
  // ⛔ Only a name the module binds ONCE, and only to a bare builtin spelling:
  // a name it rebinds is a variable that happens to hold a class, and the call
  // through it is whatever it holds at that point.
  if (spelling != name) {
    auto alias = builtinValueAliases.find(spelling);
    if (alias == builtinValueAliases.end() || alias->second != name)
      return false;
  }
  return !programBindsName(name);
}

ModuleEmitter::AttributePresence
ModuleEmitter::attributePresence(mlir::Type subject, llvm::StringRef attribute,
                                 std::string *undecidedBy) {
  // A CLASS answers with what its class line declares -- methods and class
  // attributes; an instance field is not an attribute of the class.
  bool throughClass = false;
  if (auto typeObject = mlir::dyn_cast_if_present<py::TypeType>(subject)) {
    subject = typeObject.getInstanceType();
    throughClass = true;
  }
  auto contract = mlir::dyn_cast_if_present<py::ContractType>(subject);
  if (!contract)
    return AttributePresence::Undecided;
  // ⛔ A name with `$` is a body this compiler split out, which no source
  // identifier spells and CPython's class does not have.
  if (attribute.contains('$') && isSourceClassContract(subject))
    return AttributePresence::Absent;
  bool field = !throughClass && lookupClassField(subject, attribute).has_value();
  bool present = field || lookupClassMethod(subject, attribute).has_value() ||
                 lookupClassStaticAttr(subject, attribute).has_value();
  if (!present) {
    const py::protocols::Table &table = py::protocols::Table::get(context);
    present = !table.methodContractCandidatesWithEvidence(subject, attribute)
                   .empty() ||
              (!throughClass &&
               table.resolveFieldContractWithEvidence(subject, attribute)
                   .has_value());
  }
  if (present) {
    if (field && !fieldAlwaysStoredBeforeRead(contract.getContractName(),
                                              attribute))
      return AttributePresence::FieldMaybeUnset;
    return AttributePresence::Present;
  }
  for (const auto &entry : classMros)
    if (entry.getKey() != contract.getContractName() &&
        llvm::is_contained(entry.second, contract.getContractName())) {
      if (undecidedBy)
        *undecidedBy = entry.getKey().str();
      return AttributePresence::Undecided;
    }
  return AttributePresence::Absent;
}

// ⭐ A FIELD NO STORE MAY HAVE FILLED IS ASKED OF THE INSTANCE. `hasattr` used
// to answer from the class alone, so a field a method creates -- or one the
// class only annotates -- was there before anything stored it:
//
//     class T:
//         def setUp(self) -> None:
//             self.n = 3
//     print(hasattr(T(), "n"))    # printed True; CPython prints False
//
// The probe is a synthesized function over surface that already compiles: the
// field read raises AttributeError when the slot is empty
// (`ly.field.maybe_unset`), and the `try` turns that into the answer.
std::string ModuleEmitter::attributeProbe(llvm::StringRef cls,
                                          llvm::StringRef field,
                                          parser::SourceRange range) {
  std::string key = (cls + "\x1f" + field).str();
  if (auto found = attributeProbes.find(key); found != attributeProbes.end())
    return found->second;
  std::string symbol =
      "__ly_hasattr$" + std::to_string(++syntheticFunctionCounter);
  attributeProbes[key] = symbol;
  std::vector<parser::NodePtr> body;
  body.push_back(synth::tryExcept(
      {synth::exprStmt(synth::attribute(synth::name("__ly_recv", range),
                                        field, range),
                       range)},
      synth::name("AttributeError", range),
      {synth::returnStmt(synth::constantBool(false, range), range)}, range));
  body.push_back(synth::returnStmt(synth::constantBool(true, range), range));
  parser::NodePtr def = synth::functionDef(
      symbol, {synth::Param{"__ly_recv", synth::name(cls, range)}}, {},
      std::move(body), synth::name("bool", range), {}, range);
  synthesizedIteratorDefs.push_back(def);
  FunctionSignature sig = types.functionSignature(*def);
  types.bindRootSymbol(symbol, sig.publicCallable);
  auto emitBody = [this, def, symbol, sig] {
    emitDispatcherBody(*def, symbol, sig, /*suppressGate=*/false);
  };
  if (classDeclarationsEmitted)
    emitBody();
  else
    deferredDispatcherBodies.push_back(std::move(emitBody));
  return symbol;
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
    return emitComputedHasattr(expr, (*args)[0], (*args)[1]);
  llvm::StringRef attribute = *name;
  mlir::Type subject = types.widenLiteral(types.inferExpr(args->front().get()));
  std::string undecidedBy;
  AttributePresence presence =
      attributePresence(subject, attribute, &undecidedBy);
  // ⭐ A RECEIVER WHOSE CLASS LINE LACKS THE NAME, BUT WHOSE SUBCLASSES MAY
  // DECLARE IT, is answered by which class it has at run time: the program's
  // classes are all known, so the answer is whether that class derives from
  // one that declares the name. unittest's loader asks `hasattr(cls,
  // 'runTest')` of every test class, and the question was refused because a
  // test class may define it.
  // ⛔ An instance whose subclasses STORE the name on `self` is not answered
  // this way: whether that instance has stored it is its own history.
  if (presence == AttributePresence::Undecided && !undecidedBy.empty()) {
    bool throughClass = mlir::isa<py::TypeType>(subject);
    auto staticClass = mlir::dyn_cast<py::ContractType>(
        throughClass ? mlir::cast<py::TypeType>(subject).getInstanceType()
                     : subject);
    llvm::SmallVector<std::string, 4> definers;
    bool stored = false;
    if (staticClass)
      for (const std::string &candidate :
           programSubclassesOf(staticClass.getContractName())) {
        auto declares = [&](const llvm::StringMap<llvm::StringSet<>> &map) {
          auto found = map.find(candidate);
          return found != map.end() && found->second.contains(attribute);
        };
        if (declares(declaredClassStoredFields))
          stored = true;
        bool valued = false;
        if (const parser::Node *classDef = declaredClassNodes.lookup(candidate))
          if (const auto *classBody = ast::nodeList(*classDef, "body"))
            for (const parser::NodePtr &member : *classBody) {
              if (!member)
                continue;
              if ((member->kind == "FunctionDef" ||
                   member->kind == "AsyncFunctionDef") &&
                  llvm::StringRef(ast::string(*member, "name").value_or("")) ==
                      attribute)
                valued = true;
              const parser::Node *target = nullptr;
              if (member->kind == "AnnAssign" && ast::node(*member, "value"))
                target = ast::node(*member, "target");
              else if (member->kind == "Assign")
                if (const auto *targets = ast::nodeList(*member, "targets");
                    targets && targets->size() == 1)
                  target = targets->front().get();
              if (target && target->kind == "Name" &&
                  llvm::StringRef(ast::nameSpelling(*target)) == attribute)
                valued = true;
            }
        if (valued)
          definers.push_back(candidate);
      }
    // ⭐ AND WHEN NONE OF THEM DECLARES IT, THE ANSWER IS NO: the walk above
    // read every subclass the program has. A class with a subclass at all was
    // refused here -- `@skip` on a test class with a subclass reached the
    // loader's `hasattr(cls, 'runTest')` and stopped compiling.
    if (staticClass && definers.empty() && (throughClass || !stored)) {
      (void)emitExpr(args->front().get());
      mlir::Type literalType = types.literal("False");
      auto constant = py::BoolConstantOp::create(
          builder, loc(expr), literalType, builder.getBoolAttr(false));
      return Value{constant.getResult(), literalType};
    }
    if (staticClass && !definers.empty() && (throughClass || !stored)) {
      Value receiver = emitExpr(args->front().get());
      mlir::Value classValue =
          throughClass ? receiver.value
                       : emitClassOfValue(expr, receiver,
                                          types.typeObject(staticClass))
                             .value;
      mlir::Value test;
      for (const std::string &definer : definers) {
        mlir::Type definerType = types.contract(definer);
        auto definerClass = py::TypeObjectOp::create(
            builder, loc(expr), types.typeObject(definerType), definerType);
        mlir::Value one =
            py::TypeSubclassOp::create(builder, loc(expr), builder.getI1Type(),
                                       classValue, definerClass.getResult())
                .getResult();
        test = test ? mlir::arith::OrIOp::create(builder, loc(expr), test, one)
                          .getResult()
                    : one;
      }
      auto pyBool = py::CastFromPrimOp::create(builder, loc(expr),
                                               types.boolType(), test);
      return Value{pyBool.getResult(), types.boolType()};
    }
  }
  switch (presence) {
  case AttributePresence::Undecided:
    if (undecidedBy.empty())
      return refuse("hasattr() needs a statically resolved receiver, and " +
                    typeText(subject) + " is not one");
    return refuse("hasattr(x, '" + attribute.str() + "') on '" +
                  mlir::cast<py::ContractType>(
                      mlir::isa<py::TypeType>(subject)
                          ? mlir::cast<py::TypeType>(subject).getInstanceType()
                          : subject)
                      .getContractName()
                      .str() +
                  "' cannot be answered: '" + undecidedBy +
                  "' derives from it and may define that attribute");
  case AttributePresence::FieldMaybeUnset: {
    std::string probe = attributeProbe(
        mlir::cast<py::ContractType>(subject).getContractName(), attribute,
        expr.range);
    parser::NodePtr call = synth::call(synth::name(probe, expr.range),
                                       {(*args)[0]}, expr.range);
    synthesizedIteratorDefs.push_back(call);
    return emitExpr(call.get());
  }
  case AttributePresence::Present:
  case AttributePresence::Absent:
    break;
  }
  bool present = presence == AttributePresence::Present;
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
  if (!args || args->size() < 2 || args->size() > 3 ||
      (keywords && !keywords->empty()) ||
      llvm::any_of(*args, [](const parser::NodePtr &arg) { return !arg; }))
    return refuse("getattr() takes two or three arguments");
  std::optional<llvm::StringRef> name =
      literalStringArgument((*args)[1].get());
  if (!name)
    return refuse("getattr() with a computed name is supported as the call "
                  "`getattr(x, name)(...)`: the attribute itself has no type "
                  "here that every name it may be shares");
  if (args->size() == 2) {
    parser::NodePtr attribute =
        synth::attribute(args->front(), name->str(), expr.range);
    synthesizedIteratorDefs.push_back(attribute);
    return emitExpr(attribute.get());
  }
  // ⭐ THE DEFAULT FORM IS `hasattr` PICKING AN ARM. `getattr(result,
  // "addSkip", None)` is how unittest asks for an optional protocol method,
  // and the class line answers it the way it answers `hasattr`: the attribute,
  // or the default. CPython evaluates all three arguments before the lookup,
  // so the receiver and the default are each emitted exactly once, in order,
  // whichever arm is taken.
  mlir::Type subject = types.widenLiteral(types.inferExpr(args->front().get()));
  std::string undecidedBy;
  AttributePresence presence =
      attributePresence(subject, *name, &undecidedBy);
  if (presence == AttributePresence::Undecided)
    return refuse(
        undecidedBy.empty()
            ? "getattr() with a default needs a statically resolved "
              "receiver, and " +
                  typeText(subject) + " is not one"
            : "getattr(x, '" + name->str() + "', default) cannot be answered: '" +
                  undecidedBy +
                  "' derives from the receiver's class and may define that "
                  "attribute");
  std::string receiverName =
      "__getattr" + std::to_string(++syntheticFunctionCounter);
  std::string defaultName = receiverName + "$default";
  Value receiver = emitExpr(args->front().get());
  values[receiverName] = receiver;
  types.bindSymbol(receiverName, receiver.type);
  Value fallback = emitExpr((*args)[2].get());
  values[defaultName] = fallback;
  types.bindSymbol(defaultName, fallback.type);
  auto cleanup = llvm::scope_exit([&] {
    values.erase(receiverName);
    values.erase(defaultName);
  });
  if (presence == AttributePresence::Absent)
    return fallback;
  parser::NodePtr read = synth::attribute(synth::name(receiverName, expr.range),
                                          name->str(), expr.range);
  if (presence == AttributePresence::Present) {
    synthesizedIteratorDefs.push_back(read);
    return emitExpr(read.get());
  }
  std::string probe = attributeProbe(
      mlir::cast<py::ContractType>(subject).getContractName(), *name,
      expr.range);
  parser::NodePtr choice = synth::ifExp(
      synth::call(synth::name(probe, expr.range),
                  {synth::name(receiverName, expr.range)}, expr.range),
      std::move(read), synth::name(defaultName, expr.range), expr.range);
  synthesizedIteratorDefs.push_back(choice);
  return emitExpr(choice.get());
}

namespace {
// `dir(object)` in CPython 3.14: every instance answers these, whatever its
// class.
constexpr llvm::StringLiteral kObjectAttributeNames[] = {
    "__class__",    "__delattr__",       "__dir__",       "__doc__",
    "__eq__",       "__format__",        "__ge__",        "__getattribute__",
    "__getstate__", "__gt__",            "__hash__",      "__init__",
    "__init_subclass__", "__le__",       "__lt__",        "__ne__",
    "__new__",      "__reduce__",        "__reduce_ex__", "__repr__",
    "__setattr__",  "__sizeof__",        "__str__",       "__subclasshook__"};
// And what a class statement gives every class without `__slots__`.
constexpr llvm::StringLiteral kClassStatementAttributeNames[] = {
    "__dict__", "__firstlineno__", "__module__", "__static_attributes__",
    "__weakref__"};

// CPython's list of parameter names in an arity message: 'a', 'a' and 'b',
// 'a', 'b', and 'c'.
std::string quotedNameList(llvm::ArrayRef<std::string> names) {
  std::string out;
  for (auto [index, name] : llvm::enumerate(names)) {
    if (index > 0)
      out += names.size() == 2 ? " and "
             : index + 1 == names.size() ? ", and "
                                         : ", ";
    out += "'" + name + "'";
  }
  return out;
}

// CPython's `_Py_CalculateSuggestions` (Python/suggestions.c): the candidate
// within a third of the characters involved, by its Levenshtein distance where
// a case flip costs half a move. Empty when none is close enough.
std::string closestName(llvm::ArrayRef<std::string> candidates,
                        llvm::StringRef name) {
  constexpr std::size_t kMaxStringSize = 40;
  constexpr std::size_t kMoveCost = 2;
  constexpr std::size_t kCaseCost = 1;
  if (candidates.size() >= 750)
    return "";
  auto substitutionCost = [&](char a, char b) -> std::size_t {
    if ((a & 31) != (b & 31))
      return kMoveCost;
    if (a == b)
      return 0;
    if ('A' <= a && a <= 'Z')
      a += 'a' - 'A';
    if ('A' <= b && b <= 'Z')
      b += 'a' - 'A';
    return a == b ? kCaseCost : kMoveCost;
  };
  auto distance = [&](llvm::StringRef a, llvm::StringRef b,
                      std::size_t maxCost) -> std::size_t {
    while (!a.empty() && !b.empty() && a.front() == b.front()) {
      a = a.drop_front();
      b = b.drop_front();
    }
    while (!a.empty() && !b.empty() && a.back() == b.back()) {
      a = a.drop_back();
      b = b.drop_back();
    }
    if (a.empty() || b.empty())
      return (a.size() + b.size()) * kMoveCost;
    if (a.size() > kMaxStringSize || b.size() > kMaxStringSize)
      return maxCost + 1;
    if (b.size() < a.size())
      std::swap(a, b);
    if ((b.size() - a.size()) * kMoveCost > maxCost)
      return maxCost + 1;
    llvm::SmallVector<std::size_t, kMaxStringSize> buffer(a.size());
    std::size_t tmp = kMoveCost;
    for (std::size_t &cell : buffer) {
      cell = tmp;
      tmp += kMoveCost;
    }
    std::size_t result = 0;
    for (std::size_t bIndex = 0; bIndex < b.size(); ++bIndex) {
      char code = b[bIndex];
      std::size_t previous = result = bIndex * kMoveCost;
      std::size_t minimum = std::numeric_limits<std::size_t>::max();
      for (std::size_t index = 0; index < a.size(); ++index) {
        std::size_t substitute = previous + substitutionCost(code, a[index]);
        previous = buffer[index];
        std::size_t insertDelete = std::min(result, previous) + kMoveCost;
        result = std::min(insertDelete, substitute);
        buffer[index] = result;
        minimum = std::min(minimum, result);
      }
      if (minimum > maxCost)
        return maxCost + 1;
    }
    return result;
  };
  std::string suggestion;
  std::size_t suggestionDistance = std::numeric_limits<std::size_t>::max();
  for (const std::string &candidate : candidates) {
    if (candidate == name)
      continue;
    std::size_t maxDistance =
        (name.size() + candidate.size() + 3) * kMoveCost / 6;
    maxDistance = std::min(maxDistance, suggestionDistance - 1);
    std::size_t current = distance(name, candidate, maxDistance);
    if (current > maxDistance)
      continue;
    if (suggestion.empty() || current < suggestionDistance) {
      suggestion = candidate;
      suggestionDistance = current;
    }
  }
  return suggestion;
}

// What CPython raises when `qualname` is called with `given` positional
// arguments and the keywords `keywordNames` -- the checks of
// `initialize_locals` (Python/ceval.c) in its order: each keyword, then too
// many positionals, then the missing positionals, then the missing keyword-only
// ones. `bound`: the first parameter is the receiver, which CPython counts.
// Empty when the call binds.
std::string callBindingError(llvm::StringRef qualname, const parser::Node &def,
                             bool bound, unsigned given,
                             llvm::ArrayRef<std::string> keywordNames = {}) {
  const parser::Node *arguments = ast::node(def, "args");
  if (!arguments)
    return "";
  llvm::SmallVector<std::string, 8> names;
  unsigned positionalOnly = 0;
  for (llvm::StringRef field : {"posonlyargs", "args", "kwonlyargs"})
    if (const auto *list = ast::nodeList(*arguments, field))
      for (const parser::NodePtr &argument : *list)
        if (argument) {
          names.push_back(std::string(ast::nameSpelling(*argument)));
          if (field == "posonlyargs")
            ++positionalOnly;
        }
  const auto *kwonly = ast::nodeList(*arguments, "kwonlyargs");
  unsigned keywordOnlyCount = kwonly ? static_cast<unsigned>(kwonly->size()) : 0;
  unsigned argcount = static_cast<unsigned>(names.size()) - keywordOnlyCount;
  unsigned total = static_cast<unsigned>(names.size());
  bool varargs = ast::node(*arguments, "vararg") != nullptr;
  bool varkeywords = ast::node(*arguments, "kwarg") != nullptr;
  const auto *defaultList = ast::nodeList(*arguments, "defaults");
  unsigned defaults = std::min<unsigned>(
      defaultList ? static_cast<unsigned>(defaultList->size()) : 0, argcount);
  const auto *kwDefaults = ast::nodeList(*arguments, "kw_defaults");
  if (bound)
    ++given;
  std::string head = (qualname + "() ").str();
  llvm::SmallVector<bool, 8> filled(total, false);
  for (unsigned index = 0; index < std::min(given, argcount); ++index)
    filled[index] = true;
  for (const std::string &keyword : keywordNames) {
    std::optional<unsigned> slot;
    for (unsigned index = positionalOnly; index < total; ++index)
      if (names[index] == keyword) {
        slot = index;
        break;
      }
    if (!slot) {
      if (varkeywords)
        continue;
      llvm::SmallVector<std::string, 2> conflicts;
      for (unsigned index = 0; index < positionalOnly; ++index)
        for (const std::string &other : keywordNames)
          if (other == names[index])
            conflicts.push_back(other);
      if (!conflicts.empty())
        return head +
               "got some positional-only arguments passed as keyword "
               "arguments: '" +
               llvm::join(conflicts, ", ") + "'";
      std::string message =
          head + "got an unexpected keyword argument '" + keyword + "'";
      std::string suggestion = closestName(
          llvm::ArrayRef<std::string>(names).drop_front(positionalOnly),
          keyword);
      if (!suggestion.empty())
        message += ". Did you mean '" + suggestion + "'?";
      return message;
    }
    if (filled[*slot])
      return head + "got multiple values for argument '" + keyword + "'";
    filled[*slot] = true;
  }
  auto count = [](unsigned n, llvm::StringRef noun) {
    return std::to_string(n) + " " + noun.str() + (n == 1 ? "" : "s");
  };
  if (given > argcount && !varargs) {
    unsigned keywordOnlyGiven = static_cast<unsigned>(
        llvm::count(llvm::ArrayRef<bool>(filled).drop_front(argcount), true));
    std::string takes =
        defaults ? "from " + std::to_string(argcount - defaults) + " to " +
                       std::to_string(argcount) + " positional arguments"
                 : count(argcount, "positional argument");
    std::string givenText = std::to_string(given);
    if (keywordOnlyGiven)
      givenText += " positional argument" + std::string(given != 1 ? "s" : "") +
                   " (and " + count(keywordOnlyGiven, "keyword-only argument") +
                   ")";
    return head + "takes " + takes + " but " + givenText +
           (given == 1 && !keywordOnlyGiven ? " was given" : " were given");
  }
  llvm::SmallVector<std::string, 4> missing;
  for (unsigned index = 0; index < argcount - defaults; ++index)
    if (!filled[index])
      missing.push_back(names[index]);
  if (!missing.empty())
    return head + "missing " +
           count(static_cast<unsigned>(missing.size()),
                 "required positional argument") +
           ": " + quotedNameList(missing);
  for (unsigned index = argcount; index < total; ++index) {
    unsigned keywordIndex = index - argcount;
    bool hasDefault = kwDefaults && keywordIndex < kwDefaults->size() &&
                      (*kwDefaults)[keywordIndex];
    if (!filled[index] && !hasDefault)
      missing.push_back(names[index]);
  }
  if (!missing.empty())
    return head + "missing " +
           count(static_cast<unsigned>(missing.size()),
                 "required keyword-only argument") +
           ": " + quotedNameList(missing);
  return "";
}

std::string arityMismatch(llvm::StringRef qualname, const parser::Node &def,
                          bool bound, unsigned given) {
  return callBindingError(qualname, def, bound, given);
}
} // namespace

bool ModuleEmitter::programClassDerivesFrom(llvm::StringRef candidate,
                                            llvm::StringRef ancestor) const {
  // A builtin base may be recorded as written (`ValueError`).
  llvm::StringRef builtinLeaf = ancestor;
  if (!builtinLeaf.consume_front("builtins."))
    builtinLeaf = {};
  llvm::SmallVector<llvm::StringRef, 8> worklist{candidate};
  llvm::StringSet<> seen;
  while (!worklist.empty()) {
    llvm::StringRef current = worklist.pop_back_val();
    auto bases = declaredClassBases.find(current);
    if (bases == declaredClassBases.end())
      continue;
    for (const std::string &base : bases->second) {
      if (base == ancestor || (!builtinLeaf.empty() && base == builtinLeaf))
        return true;
      if (seen.insert(base).second)
        worklist.push_back(base);
    }
  }
  return false;
}

llvm::SmallVector<std::string, 8>
ModuleEmitter::programSubclassesOf(llvm::StringRef cls) const {
  llvm::SmallVector<std::string, 8> subclasses;
  for (const auto &entry : declaredClassBases)
    if (entry.getKey() != cls && programClassDerivesFrom(entry.getKey(), cls))
      subclasses.push_back(entry.getKey().str());
  llvm::sort(subclasses);
  return subclasses;
}

llvm::StringSet<>
ModuleEmitter::programAttributeNames(llvm::StringRef cls) const {
  llvm::StringSet<> names;
  llvm::SmallVector<llvm::StringRef, 8> worklist{cls};
  llvm::StringSet<> seen;
  seen.insert(cls);
  while (!worklist.empty()) {
    llvm::StringRef current = worklist.pop_back_val();
    for (const llvm::StringMap<llvm::StringSet<>> *declarations :
         {&declaredClassMethods, &declaredClassAttributes,
          &declaredClassStoredFields})
      if (auto found = declarations->find(current);
          found != declarations->end())
        for (const auto &name : found->second)
          // ⛔ Not a name with `$`, which no source identifier has: it is a
          // body this compiler split out (`m$cmgen`, `__ly_undecorated$m`),
          // and CPython's class has no such attribute.
          if (!name.getKey().contains('$'))
            names.insert(name.getKey());
    if (auto bases = declaredClassBases.find(current);
        bases != declaredClassBases.end())
      for (const std::string &base : bases->second)
        if (seen.insert(base).second)
          worklist.push_back(base);
  }
  return names;
}

std::optional<llvm::SmallVector<std::string, 8>>
ModuleEmitter::computedLookupHierarchy(const parser::Node &expr,
                                       llvm::StringRef what,
                                       mlir::Type receiverType) {
  auto refuse = [&](std::string reason) {
    diagnostics.push_back(parser::Diagnostic{parser::Severity::Error,
                                             expr.range.start,
                                             std::move(reason)});
    return std::nullopt;
  };
  auto contract = mlir::dyn_cast_if_present<py::ContractType>(receiverType);
  if (!contract || !declaredClassNodes.count(contract.getContractName()))
    return refuse(what.str() +
                  " with a computed name needs a receiver whose class this "
                  "program declares, and " +
                  typeText(receiverType) + " is not one");
  llvm::StringRef cls = contract.getContractName();
  // ⛔ Not a class whose members the compiler writes: a NamedTuple's or an
  // enum's attributes are not in its class body, so a name lookup over the
  // body would answer AttributeError for a field CPython has.
  if (namedTupleContracts.count(cls) || enumClasses.count(cls))
    return refuse(what.str() + " with a computed name is not supported on '" +
                  cls.str() +
                  "': its attributes are made by the compiler, not declared "
                  "in its body");
  llvm::SmallVector<std::string, 8> hierarchy{cls.str()};
  for (const std::string &subclass : programSubclassesOf(cls))
    hierarchy.push_back(subclass);
  for (const std::string &member : hierarchy)
    if (!types.lookupClass(member).value_or(mlir::Type()))
      return refuse(what.str() + " with a computed name cannot be answered "
                                 "here: '" +
                    member + "' derives from '" + cls.str() +
                    "' and is declared after this lookup");
  return hierarchy;
}

// ⭐ A COMPUTED NAME IS ONE OF A CLOSED SET OF NAMES. `getattr(self, name)()`
// is how unittest runs a test method, and the names a value of class `A` can
// answer are the ones `A` and the program's subclasses of it declare, plus
// what every object answers -- so the lookup is a function comparing the name
// with each of them, one per (class, argument types):
//
//     def __lygetattr$1(__ly_recv: A, __ly_name: str) -> object:
//         if __ly_name == "run":          # A answers it: A's call, which the
//             return __ly_recv.run()      # virtual dispatcher resolves
//         if __ly_name == "test_good":    # only a subclass answers it
//             if isinstance(__ly_recv, MyTest):
//                 return __ly_recv.test_good()
//         raise AttributeError("'" + type(__ly_recv).__name__ + ...)
//
// An attribute the arguments do not fit raises the TypeError CPython raises:
// the arity sentence for a method, "'str' object is not callable" for a
// value, read from the value's own class.
// ⛔ The arms are written when every class is declared, not at the call: a
// framework's method is emitted before the module that subclasses it has
// bound the subclass's methods, and arms read then took them for values.
// The signature does not depend on them, so the call is typed at once.
// ⛔ The CALL form only. `m = getattr(o, name)` has no type to give `m`
// without one per attribute, so it keeps the refusal.
// ⛔ Arguments that are names or constants only: CPython evaluates the call's
// arguments AFTER the lookup, so one with an effect would run before an
// AttributeError it should never have reached.
std::optional<Value>
ModuleEmitter::tryEmitComputedGetattrCall(const parser::Node &expr,
                                          const parser::Node *calleeNode) {
  if (!calleeNode || calleeNode->kind != "Call")
    return std::nullopt;
  const parser::Node *innerCallee = ast::node(*calleeNode, "func");
  if (!callsUnshadowedBuiltin(innerCallee, "getattr"))
    return std::nullopt;
  const auto *lookup = ast::nodeList(*calleeNode, "args");
  const auto *lookupKeywords = ast::nodeList(*calleeNode, "keywords");
  if (!lookup || lookup->size() != 2 ||
      (lookupKeywords && !lookupKeywords->empty()) || !(*lookup)[0] ||
      !(*lookup)[1] || (*lookup)[0]->kind == "Starred" ||
      (*lookup)[1]->kind == "Starred" ||
      literalStringArgument((*lookup)[1].get()))
    return std::nullopt;
  parser::SourceRange range = expr.range;
  auto refuse = [&](std::string reason) -> std::optional<Value> {
    diagnostics.push_back(
        parser::Diagnostic{parser::Severity::Error, range.start,
                           std::move(reason)});
    return emitNone(expr);
  };
  const auto *args = ast::nodeList(expr, "args");
  const auto *keywords = ast::nodeList(expr, "keywords");
  if (keywords && !keywords->empty())
    return refuse("getattr(x, name)(...) with a computed name takes no keyword "
                  "arguments here");
  llvm::SmallVector<mlir::Type, 4> argumentTypes;
  if (args)
    for (const parser::NodePtr &argument : *args) {
      if (!argument || (argument->kind != "Name" &&
                        argument->kind != "Constant"))
        return refuse("getattr(x, name)(...) with a computed name takes only "
                      "names and constants as arguments here: CPython "
                      "evaluates them after the lookup, which may raise");
      argumentTypes.push_back(
          types.widenLiteral(types.inferExpr(argument.get())));
    }
  mlir::Type nameType = types.widenLiteral(types.inferExpr((*lookup)[1].get()));
  if (nameType != types.strType())
    return refuse("getattr() needs a str attribute name, and " +
                  typeText(nameType) + " is not one");
  mlir::Type receiverType =
      types.widenLiteral(types.inferExpr((*lookup)[0].get()));
  std::optional<llvm::SmallVector<std::string, 8>> hierarchy =
      computedLookupHierarchy(expr, "getattr()", receiverType);
  if (!hierarchy)
    return emitNone(expr);
  std::string cls = hierarchy->front();

  std::string key = "getattr|" + cls;
  for (mlir::Type argumentType : argumentTypes)
    key += "|" + typeText(argumentType);
  std::string symbol;
  if (auto found = computedLookupDispatchers.find(key);
      found != computedLookupDispatchers.end()) {
    symbol = found->second;
  } else {
    symbol = "__lygetattr$" + std::to_string(++syntheticFunctionCounter);
    computedLookupDispatchers[key] = symbol;
    llvm::SmallVector<synth::Param, 4> params{
        synth::Param{"__ly_recv", nullptr}, synth::Param{"__ly_name", nullptr}};
    for (unsigned index = 0; index < argumentTypes.size(); ++index)
      params.push_back(
          synth::Param{"__ly_a" + std::to_string(index), nullptr});
    llvm::SmallVector<const parser::Node *, 4> paramNodes;
    parser::NodePtr def =
        synth::functionDef(symbol, params, {}, {}, synth::name("object", range),
                           {}, range, &paramNodes);
    synthesizedIteratorDefs.push_back(def);
    types.overrideParameterType(paramNodes[0],
                                types.lookupClass(cls).value_or(mlir::Type()));
    types.overrideParameterType(paramNodes[1], types.strType());
    for (auto [index, argumentType] : llvm::enumerate(argumentTypes))
      types.overrideParameterType(paramNodes[index + 2], argumentType);
    FunctionSignature sig = types.functionSignature(*def);
    types.bindRootSymbol(symbol, sig.publicCallable);
    auto emitBody = [this, def, symbol, sig, range, cls,
                     members = *hierarchy, argumentTypes] {
      if (parser::Field *body = parser::findField(*def, "body"))
        body->value = computedGetattrBody(range, cls, members, argumentTypes);
      emitDispatcherBody(*def, symbol, sig, /*suppressGate=*/false);
    };
    if (classDeclarationsEmitted)
      emitBody();
    else
      deferredDispatcherBodies.push_back(std::move(emitBody));
  }
  std::vector<parser::NodePtr> callArguments{(*lookup)[0], (*lookup)[1]};
  if (args)
    callArguments.insert(callArguments.end(), args->begin(), args->end());
  parser::NodePtr call =
      synth::call(synth::name(symbol, range), std::move(callArguments), range);
  synthesizedIteratorDefs.push_back(call);
  return emitExpr(call.get());
}

std::vector<parser::NodePtr> ModuleEmitter::computedGetattrBody(
    parser::SourceRange range, llvm::StringRef cls,
    llvm::ArrayRef<std::string> hierarchy,
    llvm::ArrayRef<mlir::Type> argumentTypes) {
  llvm::StringMap<llvm::StringSet<>> namesOf;
  std::set<std::string> universe;
  for (const std::string &member : hierarchy) {
    namesOf[member] = programAttributeNames(member);
    for (const auto &name : namesOf[member])
      if (!name.getKey().contains('.'))
        universe.insert(name.getKey().str());
  }
  auto receiver = [&] { return synth::name("__ly_recv", range); };
  auto forwarded = [&] {
    std::vector<parser::NodePtr> out;
    for (unsigned index = 0; index < argumentTypes.size(); ++index)
      out.push_back(synth::name("__ly_a" + std::to_string(index), range));
    return out;
  };
  auto callArm = [&](llvm::StringRef name) {
    parser::NodePtr call = synth::call(
        synth::attribute(receiver(), name, range), forwarded(), range);
    dispatcherArmCalls.insert(call.get());
    return synth::returnStmt(std::move(call), range);
  };
  auto qualname = [&](llvm::StringRef definingClass, llvm::StringRef name) {
    return py::contracts::pythonQualnameForContract(definingClass) + "." +
           name.str();
  };
  auto notCallable = [&](llvm::StringRef className) {
    return synth::raiseCall(
        "TypeError", "'" + className.str() + "' object is not callable",
        range);
  };
  // The arm for `name` on a receiver of class `member`.
  auto armFor = [&](llvm::StringRef member,
                    llvm::StringRef name) -> parser::NodePtr {
    mlir::Type memberType = types.lookupClass(member).value_or(mlir::Type());
    std::optional<MethodBinding> method = lookupClassMethod(memberType, name);
    // A method is called when its parameters take the count; the arm's own
    // call checks the types, as any call does.
    if (method && method->kind != "property") {
      if (!method->method)
        return types.inferMethodCallWithEvidence(memberType, name,
                                                 argumentTypes)
                   ? callArm(name)
                   : synth::raiseCall("TypeError",
                                      qualname(method->definingClass, name) +
                                          "() cannot take these arguments here",
                                      range);
      std::string message =
          arityMismatch(qualname(method->definingClass, name), *method->method,
                        method->kind != "static",
                        static_cast<unsigned>(argumentTypes.size()));
      if (message.empty())
        return callArm(name);
      return synth::raiseCall("TypeError", message, range);
    }
    mlir::Type valueType;
    if (method)
      valueType = method->signature.resultType;
    else if (std::optional<mlir::Type> field =
                 lookupClassField(memberType, name))
      valueType = *field;
    else if (std::optional<mlir::Type> attribute =
                 lookupClassStaticAttr(memberType, name))
      valueType = *attribute;
    valueType = types.widenLiteral(valueType);
    if (valueType && (mlir::isa<py::CallableType>(valueType) ||
                      mlir::isa<py::TypeType>(valueType)))
      return callArm(name);
    // CPython's sentence, with the class the value has at run time.
    parser::NodePtr className = synth::attribute(
        synth::call(synth::name("type", range),
                    {synth::attribute(receiver(), name, range)}, range),
        "__name__", range);
    return synth::raiseStmt(
        synth::call(
            synth::name("TypeError", range),
            {synth::binOp(
                synth::binOp(synth::strConstant("'", range), "Add",
                             std::move(className), range),
                "Add", synth::strConstant("' object is not callable", range),
                range)},
            range),
        range);
  };
  auto nameIs = [&](llvm::StringRef name) {
    return synth::compare(synth::name("__ly_name", range), "Eq",
                          synth::strConstant(name, range), range);
  };
  auto exactly = [&](llvm::StringRef member) {
    return synth::compare(
        synth::call(synth::name("type", range), {receiver()}, range), "Is",
        synth::name(member, range), range);
  };
  mlir::Type receiverClassType =
      types.lookupClass(cls).value_or(mlir::Type());
  // What every instance answers. The data among it is never callable, and
  // CPython's sentence names its class: a docstring is a str and its absence
  // None, per class, since `__doc__` is not inherited.
  static const llvm::StringMap<llvm::StringRef> kDataAttributeClasses = {
      {"__dict__", "dict"},
      {"__module__", "str"},
      {"__weakref__", "NoneType"},
      {"__firstlineno__", "int"},
      {"__static_attributes__", "tuple"}};
  auto everyInstanceHas = [&](llvm::StringRef name) {
    return llvm::is_contained(kObjectAttributeNames, name) ||
           llvm::is_contained(kClassStatementAttributeNames, name);
  };
  auto unsupported = [&](llvm::StringRef name, llvm::StringRef why) {
    return synth::raiseCall("TypeError",
                            "getattr(x, '" + name.str() +
                                "')(...) is not supported here: " + why.str(),
                            range);
  };
  // The arms of `name` for a receiver whose class declares nothing by it.
  // `belowDefiners`: some subclass of the receiver's static class declares
  // it, so `recv.name(...)` would ask for a dispatch over those overrides --
  // the receiver reaching this arm is none of them, and what answers is the
  // default its class inherits.
  auto objectDefault = [&](llvm::StringRef name, bool belowDefiners) {
    std::vector<parser::NodePtr> block;
    bool objectsOwn = inheritsObjectDefaultDunder(receiverClassType, name);
    if (belowDefiners && isImplementedObjectDefault(name)) {
      // object's own slot, called with the receiver, as CPython's
      // `object.__str__(x)` names it; a default from a runtime base, or one
      // with no value here, is not.
      if (!objectsOwn || !argumentTypes.empty() ||
          (name != "__repr__" && name != "__str__" && name != "__hash__")) {
        block.push_back(unsupported(
            name, "the receiver's class inherits it, and only object's own "
                  "__repr__, __str__ and __hash__ are called by a computed "
                  "name past a subclass's override"));
      } else if (name == "__hash__" &&
                 !inheritsObjectDefaultDunder(receiverClassType, "__eq__")) {
        // A class that declares `__eq__` and not `__hash__` has `__hash__ =
        // None`.
        block.push_back(notCallable("NoneType"));
      } else {
        block.push_back(synth::returnStmt(
            synth::call(
                synth::attribute(synth::name("object", range), name, range),
                {receiver()}, range),
            range));
      }
      return block;
    }
    if (auto data = kDataAttributeClasses.find(name);
        data != kDataAttributeClasses.end()) {
      block.push_back(notCallable(data->second));
    } else if (name == "__doc__") {
      for (const std::string &member : hierarchy) {
        const parser::Node *classDef = declaredClassNodes.lookup(member);
        const auto *classBody =
            classDef ? ast::nodeList(*classDef, "body") : nullptr;
        const parser::Node *first = classBody && !classBody->empty()
                                        ? classBody->front().get()
                                        : nullptr;
        const parser::Node *docstring = first && first->kind == "Expr"
                                            ? ast::node(*first, "value")
                                            : nullptr;
        if (!docstring || !literalStringArgument(docstring))
          continue;
        block.push_back(
            synth::ifStmt(exactly(member), {notCallable("str")}, {}, range));
      }
      block.push_back(notCallable("NoneType"));
    } else if (name == "__class__") {
      // The class called: a new instance of the class the receiver has at
      // run time, refused as CPython refuses it when its `__init__` does
      // not take the arguments.
      for (const std::string &member : hierarchy) {
        mlir::Type memberType =
            types.lookupClass(member).value_or(mlir::Type());
        std::string message;
        std::optional<MethodBinding> init =
            lookupClassMethod(memberType, "__init__");
        if (init && init->method)
          message = arityMismatch(
              qualname(init->definingClass, "__init__"), *init->method,
              /*bound=*/true, static_cast<unsigned>(argumentTypes.size()));
        else if (!argumentTypes.empty())
          message = py::contracts::displayClassNameForContract(member) +
                    "() takes no arguments";
        parser::NodePtr arm =
            message.empty()
                ? synth::returnStmt(synth::call(synth::name(member, range),
                                                forwarded(), range),
                                    range)
                : synth::raiseCall("TypeError", message, range);
        block.push_back(
            synth::ifStmt(exactly(member), {std::move(arm)}, {}, range));
      }
    } else if (name == "__init__") {
      // object's: nothing to do, and CPython's sentence, naming the class
      // the receiver has at run time, when the call passes anything.
      if (argumentTypes.empty()) {
        block.push_back(synth::returnStmt(synth::noneConstant(range), range));
      } else {
        parser::NodePtr className = synth::attribute(
            synth::call(synth::name("type", range), {receiver()}, range),
            "__name__", range);
        block.push_back(synth::raiseStmt(
            synth::call(
                synth::name("TypeError", range),
                {synth::binOp(
                    std::move(className), "Add",
                    synth::strConstant(".__init__() takes exactly one argument "
                                       "(the instance to initialize)",
                                       range),
                    range)},
                range),
            range));
      }
    } else if (objectsOwn && !isImplementedObjectDefault(name)) {
      // ⛔ Raised here, not refused: the program reaches this arm only if
      // its run time computes the name, and refusing every computed call
      // on a class that inherits an unimplemented default refused them all.
      block.push_back(unsupported(
          name, "Lython does not implement builtins.object." + name.str()));
    } else if (types.inferMethodCallWithEvidence(receiverClassType, name,
                                                 argumentTypes)) {
      block.push_back(callArm(name));
    } else {
      block.push_back(unsupported(name, "a computed name calls the attributes "
                                        "the program's classes declare"));
    }
    return block;
  };
  std::vector<parser::NodePtr> body;
  for (const std::string &name : universe) {
    std::vector<parser::NodePtr> block;
    llvm::SmallVector<llvm::StringRef, 4> definers;
    for (const std::string &member : hierarchy)
      if (namesOf[member].contains(name))
        definers.push_back(member);
    if (llvm::is_contained(definers, cls)) {
      block.push_back(armFor(cls, name));
    } else {
      for (llvm::StringRef member : definers) {
        bool inherited = llvm::any_of(definers, [&](llvm::StringRef other) {
          return other != member && programClassDerivesFrom(member, other);
        });
        if (inherited)
          continue;
        block.push_back(synth::ifStmt(
            synth::call(synth::name("isinstance", range),
                        {receiver(), synth::name(member, range)}, range),
            {armFor(member, name)}, {}, range));
      }
      // ⭐ A RECEIVER OF NONE OF THE DEFINERS STILL HAS WHAT EVERY INSTANCE
      // HAS. `__str__` declared only by a subclass is object's for the base:
      // `getattr(Base(), "__str__")()` fell past the arms to "'Base' object
      // has no attribute '__str__'" where CPython answers the default str.
      if (everyInstanceHas(name))
        for (parser::NodePtr &arm : objectDefault(name, /*belowDefiners=*/true))
          block.push_back(std::move(arm));
    }
    body.push_back(synth::ifStmt(nameIs(name), std::move(block), {}, range));
  }
  for (llvm::ArrayRef<llvm::StringLiteral> list :
       {llvm::ArrayRef<llvm::StringLiteral>(kObjectAttributeNames),
        llvm::ArrayRef<llvm::StringLiteral>(kClassStatementAttributeNames)})
    for (llvm::StringLiteral name : list) {
      if (universe.count(name.str()))
        continue;
      body.push_back(synth::ifStmt(nameIs(name),
                                   objectDefault(name, /*belowDefiners=*/false),
                                   {}, range));
    }
  // CPython's sentence, naming the class the receiver has at run time.
  parser::NodePtr message = synth::binOp(
      synth::binOp(
          synth::binOp(synth::strConstant("'", range), "Add",
                       synth::attribute(synth::call(synth::name("type", range),
                                                    {receiver()}, range),
                                        "__name__", range),
                       range),
          "Add", synth::strConstant("' object has no attribute '", range),
          range),
      "Add",
      synth::binOp(synth::name("__ly_name", range), "Add",
                   synth::strConstant("'", range), range),
      range);
  body.push_back(
      synth::raiseStmt(synth::call(synth::name("AttributeError", range),
                                   {std::move(message)}, range),
                       range));
  return body;
}

// ⭐ `getattr(recv, name).__doc__` WITH `name` COMPUTED is the docstring of
// whatever the name finds, which is how unittest describes a test:
// `self._testMethodDoc = getattr(self, methodName).__doc__`. The same closed
// set of names as the call form, each answered as the literal spelling
// `recv.name.__doc__` is -- a method's docstring by the class the receiver
// has at run time, a value's by its class.
// ⛔ A name whose answer is a builtin's docstring -- a str field, object's own
// `__eq__` -- raises a TypeError saying so: no builtin keeps its docstring
// here, and the program reaches the arm only if its run time computes that
// name, so refusing it statically would refuse every computed read.
std::optional<Value>
ModuleEmitter::emitComputedDocRead(const parser::Node &expr,
                                   const parser::NodePtr &receiverNode,
                                   const parser::NodePtr &nameNode) {
  parser::SourceRange range = expr.range;
  mlir::Type nameType = types.widenLiteral(types.inferExpr(nameNode.get()));
  if (nameType != types.strType()) {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, range.start,
        "getattr() needs a str attribute name, and " + typeText(nameType) +
            " is not one"});
    return emitNone(expr);
  }
  mlir::Type receiverType =
      types.widenLiteral(types.inferExpr(receiverNode.get()));
  // ⭐ ONE DISPATCHER FOR A CLASS AND ITS SUBCLASSES: the base's already asks
  // `isinstance` for every name a subclass declares. unittest reads the
  // docstring in `TestCase.__init__`, which is copied into each class's
  // construction, and each test class cost a dispatcher of every name a
  // TestCase answers.
  // ⛔ Not an ancestor with a subclass declared after this read: its
  // dispatcher would be refused where the receiver's own is not.
  if (auto contract = mlir::dyn_cast_if_present<py::ContractType>(receiverType);
      contract && declaredClassNodes.count(contract.getContractName()))
    for (const std::string &ancestor :
         llvm::reverse(classMro(contract.getContractName()))) {
      if (ancestor == contract.getContractName())
        break;
      if (!declaredClassNodes.count(ancestor) ||
          namedTupleContracts.count(ancestor) || enumClasses.count(ancestor))
        continue;
      llvm::SmallVector<std::string, 8> below = programSubclassesOf(ancestor);
      if (!llvm::is_contained(below, contract.getContractName().str()) ||
          !llvm::all_of(below, [&](const std::string &member) {
            return static_cast<bool>(
                types.lookupClass(member).value_or(mlir::Type()));
          }))
        continue;
      if (mlir::Type ancestorType =
              types.lookupClass(ancestor).value_or(mlir::Type())) {
        receiverType = ancestorType;
        break;
      }
    }
  std::optional<llvm::SmallVector<std::string, 8>> hierarchy =
      computedLookupHierarchy(expr, "getattr(...).__doc__", receiverType);
  if (!hierarchy)
    return emitNone(expr);
  std::string cls = hierarchy->front();
  std::string symbol = docDispatcher(
      "getattrdoc|" + cls, "__lygetattrdoc$",
      {{"__ly_recv", types.lookupClass(cls).value_or(mlir::Type())},
       {"__ly_name", types.strType()}},
      range,
      [this, range, cls, members = *hierarchy] {
        return computedDocBody(range, cls, members);
      });
  parser::NodePtr call = synth::call(synth::name(symbol, range),
                                     {receiverNode, nameNode}, range);
  synthesizedIteratorDefs.push_back(call);
  return emitExpr(call.get());
}

std::vector<parser::NodePtr>
ModuleEmitter::computedDocBody(parser::SourceRange range, llvm::StringRef cls,
                               llvm::ArrayRef<std::string> hierarchy) {
  llvm::StringMap<llvm::StringSet<>> namesOf;
  std::set<std::string> universe;
  for (const std::string &member : hierarchy) {
    namesOf[member] = programAttributeNames(member);
    for (const auto &name : namesOf[member])
      if (!name.getKey().contains('.'))
        universe.insert(name.getKey().str());
  }
  auto receiver = [&] { return synth::name("__ly_recv", range); };
  auto unsupported = [&](llvm::StringRef name, llvm::StringRef why) {
    return synth::raiseCall("TypeError",
                            "getattr(x, '" + name.str() +
                                "').__doc__ is not supported here: " +
                                why.str(),
                            range);
  };
  auto docOf = [&](parser::NodePtr subject) {
    return synth::returnStmt(
        synth::attribute(std::move(subject), "__doc__", range), range);
  };
  // The arm for `name` on a receiver of class `member`.
  auto armFor = [&](llvm::StringRef member,
                    llvm::StringRef name) -> parser::NodePtr {
    mlir::Type memberType = types.lookupClass(member).value_or(mlir::Type());
    std::optional<MethodBinding> method = lookupClassMethod(memberType, name);
    if (method && method->kind != "property") {
      // One answer that is not known is refused where the read is written,
      // which here would refuse every computed read.
      if (std::optional<DocAnswer> decided = decidedMethodDoc(member, name);
          decided && decided->kind == DocAnswer::Kind::Unknown)
        return unsupported(name, decided->text);
      return docOf(synth::attribute(receiver(), name, range));
    }
    mlir::Type valueType;
    if (method)
      valueType = method->signature.resultType;
    else if (std::optional<mlir::Type> field =
                 lookupClassField(memberType, name))
      valueType = *field;
    else if (std::optional<mlir::Type> attribute =
                 lookupClassStaticAttr(memberType, name))
      valueType = *attribute;
    if (valueType && docOfValueIsAnswered(valueType))
      return docOf(synth::attribute(receiver(), name, range));
    return unsupported(name, "the docstring of " +
                                 (valueType ? typeText(types.widenLiteral(
                                                  valueType))
                                            : std::string("its value")) +
                                 " is not retained");
  };
  // What every instance has and the program's classes do not declare.
  auto objectDefault = [&](llvm::StringRef name) -> parser::NodePtr {
    if (name == "__class__")
      return docOf(synth::call(synth::name("type", range), {receiver()},
                               range));
    return unsupported(name, "the docstring of builtins.object's '" +
                                 name.str() + "' is not retained");
  };
  auto everyInstanceHas = [&](llvm::StringRef name) {
    return llvm::is_contained(kObjectAttributeNames, name) ||
           llvm::is_contained(kClassStatementAttributeNames, name);
  };
  auto nameIs = [&](llvm::StringRef name) {
    return synth::compare(synth::name("__ly_name", range), "Eq",
                          synth::strConstant(name, range), range);
  };
  std::vector<parser::NodePtr> body;
  for (const std::string &name : universe) {
    std::vector<parser::NodePtr> block;
    llvm::SmallVector<llvm::StringRef, 4> definers;
    for (const std::string &member : hierarchy)
      if (namesOf[member].contains(name))
        definers.push_back(member);
    if (llvm::is_contained(definers, cls)) {
      block.push_back(armFor(cls, name));
    } else {
      for (llvm::StringRef member : definers) {
        bool inherited = llvm::any_of(definers, [&](llvm::StringRef other) {
          return other != member && programClassDerivesFrom(member, other);
        });
        if (inherited)
          continue;
        block.push_back(synth::ifStmt(
            synth::call(synth::name("isinstance", range),
                        {receiver(), synth::name(member, range)}, range),
            {armFor(member, name)}, {}, range));
      }
      if (everyInstanceHas(name))
        block.push_back(objectDefault(name));
    }
    body.push_back(synth::ifStmt(nameIs(name), std::move(block), {}, range));
  }
  for (llvm::ArrayRef<llvm::StringLiteral> list :
       {llvm::ArrayRef<llvm::StringLiteral>(kObjectAttributeNames),
        llvm::ArrayRef<llvm::StringLiteral>(kClassStatementAttributeNames)})
    for (llvm::StringLiteral name : list)
      if (!universe.count(name.str()))
        body.push_back(
            synth::ifStmt(nameIs(name), {objectDefault(name)}, {}, range));
  // CPython's sentence, naming the class the receiver has at run time.
  parser::NodePtr message = synth::binOp(
      synth::binOp(
          synth::binOp(synth::strConstant("'", range), "Add",
                       synth::attribute(synth::call(synth::name("type", range),
                                                    {receiver()}, range),
                                        "__name__", range),
                       range),
          "Add", synth::strConstant("' object has no attribute '", range),
          range),
      "Add",
      synth::binOp(synth::name("__ly_name", range), "Add",
                   synth::strConstant("'", range), range),
      range);
  body.push_back(
      synth::raiseStmt(synth::call(synth::name("AttributeError", range),
                                   {std::move(message)}, range),
                       range));
  return body;
}

// `hasattr(recv, name)` with `name` computed: the same closed set of names,
// each answered the way the literal form answers it -- and written as late as
// the getattr arms, for the same reason.
std::optional<Value>
ModuleEmitter::emitComputedHasattr(const parser::Node &expr,
                                   const parser::NodePtr &receiverNode,
                                   const parser::NodePtr &nameNode) {
  parser::SourceRange range = expr.range;
  mlir::Type nameType = types.widenLiteral(types.inferExpr(nameNode.get()));
  if (nameType != types.strType()) {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, range.start,
        "hasattr() needs a str attribute name, and " + typeText(nameType) +
            " is not one"});
    return emitNone(expr);
  }
  mlir::Type receiverType =
      types.widenLiteral(types.inferExpr(receiverNode.get()));
  std::optional<llvm::SmallVector<std::string, 8>> hierarchy =
      computedLookupHierarchy(expr, "hasattr()", receiverType);
  if (!hierarchy)
    return emitNone(expr);
  std::string cls = hierarchy->front();
  std::string key = "hasattr|" + cls;
  std::string symbol;
  if (auto found = computedLookupDispatchers.find(key);
      found != computedLookupDispatchers.end()) {
    symbol = found->second;
  } else {
    symbol = "__lyhasattr$" + std::to_string(++syntheticFunctionCounter);
    computedLookupDispatchers[key] = symbol;
    llvm::SmallVector<const parser::Node *, 2> paramNodes;
    parser::NodePtr def = synth::functionDef(
        symbol,
        {synth::Param{"__ly_recv", nullptr}, synth::Param{"__ly_name", nullptr}},
        {}, {}, synth::name("bool", range), {}, range, &paramNodes);
    synthesizedIteratorDefs.push_back(def);
    types.overrideParameterType(paramNodes[0],
                                types.lookupClass(cls).value_or(mlir::Type()));
    types.overrideParameterType(paramNodes[1], types.strType());
    FunctionSignature sig = types.functionSignature(*def);
    types.bindRootSymbol(symbol, sig.publicCallable);
    auto emitBody = [this, def, symbol, sig, range, cls, members = *hierarchy] {
      if (parser::Field *body = parser::findField(*def, "body"))
        body->value = computedHasattrBody(range, cls, members);
      emitDispatcherBody(*def, symbol, sig, /*suppressGate=*/false);
    };
    if (classDeclarationsEmitted)
      emitBody();
    else
      deferredDispatcherBodies.push_back(std::move(emitBody));
  }
  parser::NodePtr call = synth::call(synth::name(symbol, range),
                                     {receiverNode, nameNode}, range);
  synthesizedIteratorDefs.push_back(call);
  return emitExpr(call.get());
}

std::vector<parser::NodePtr>
ModuleEmitter::computedHasattrBody(parser::SourceRange range,
                                   llvm::StringRef cls,
                                   llvm::ArrayRef<std::string> hierarchy) {
  llvm::StringMap<llvm::StringSet<>> namesOf;
  std::set<std::string> universe;
  for (const std::string &member : hierarchy) {
    namesOf[member] = programAttributeNames(member);
    for (const auto &name : namesOf[member])
      if (!name.getKey().contains('.'))
        universe.insert(name.getKey().str());
  }
  auto receiver = [&] { return synth::name("__ly_recv", range); };
  auto answer = [&](bool present) {
    return synth::returnStmt(synth::constantBool(present, range), range);
  };
  // A method or a class attribute is there; a field or a property is asked of
  // the instance, as the literal form asks it.
  auto armFor = [&](llvm::StringRef member,
                    llvm::StringRef name) -> parser::NodePtr {
    mlir::Type memberType = types.lookupClass(member).value_or(mlir::Type());
    if (attributePresence(memberType, name) == AttributePresence::Present)
      return answer(true);
    return synth::returnStmt(
        synth::call(synth::name(attributeProbe(member, name, range), range),
                    {receiver()}, range),
        range);
  };
  auto nameIs = [&](llvm::StringRef name) {
    return synth::compare(synth::name("__ly_name", range), "Eq",
                          synth::strConstant(name, range), range);
  };
  std::vector<parser::NodePtr> body;
  for (const std::string &name : universe) {
    std::vector<parser::NodePtr> block;
    llvm::SmallVector<llvm::StringRef, 4> definers;
    for (const std::string &member : hierarchy)
      if (namesOf[member].contains(name))
        definers.push_back(member);
    if (llvm::is_contained(definers, cls)) {
      block.push_back(armFor(cls, name));
    } else {
      for (llvm::StringRef member : definers) {
        bool inherited = llvm::any_of(definers, [&](llvm::StringRef other) {
          return other != member && programClassDerivesFrom(member, other);
        });
        if (inherited)
          continue;
        block.push_back(synth::ifStmt(
            synth::call(synth::name("isinstance", range),
                        {receiver(), synth::name(member, range)}, range),
            {armFor(member, name)}, {}, range));
      }
      block.push_back(answer(false));
    }
    body.push_back(synth::ifStmt(nameIs(name), std::move(block), {}, range));
  }
  for (llvm::ArrayRef<llvm::StringLiteral> list :
       {llvm::ArrayRef<llvm::StringLiteral>(kObjectAttributeNames),
        llvm::ArrayRef<llvm::StringLiteral>(kClassStatementAttributeNames)})
    for (llvm::StringLiteral name : list)
      if (!universe.count(name.str()))
        body.push_back(synth::ifStmt(nameIs(name), {answer(true)}, {}, range));
  body.push_back(answer(false));
  return body;
}

// ⭐ `dir(C)` IS A LIST THE CLASS BODIES SPELL. What CPython 3.14 lists for a
// class written as a plain `class` statement is object's names, the five a
// class statement adds, `__annotate_func__` when a body in its line
// annotates, and the class attributes of its line -- every def, every
// assignment, every annotation with a value -- sorted.
// ⛔ Not a class whose attributes are made elsewhere: a decorator (a
// dataclass), a base outside the program (an exception, NamedTuple, Enum,
// Protocol), a generic class, a metaclass keyword or `__slots__` each add or
// remove names this walk cannot see, and an answer missing one is wrong.
std::optional<std::set<std::string>>
ModuleEmitter::classDirNames(llvm::StringRef cls, std::string &why) const {
  std::set<std::string> names;
  for (llvm::StringLiteral name : kObjectAttributeNames)
    names.insert(name.str());
  for (llvm::StringLiteral name : kClassStatementAttributeNames)
    names.insert(name.str());
  bool annotated = false;
  llvm::SmallVector<std::string, 8> worklist{cls.str()};
  llvm::StringSet<> seen;
  while (!worklist.empty()) {
    std::string current = worklist.pop_back_val();
    if (!seen.insert(current).second)
      continue;
    std::string shown = py::contracts::displayClassNameForContract(current);
    const parser::Node *classDef = declaredClassNodes.lookup(current);
    if (!classDef) {
      why = "'" + shown + "' is not declared in this program";
      return std::nullopt;
    }
    for (llvm::StringRef field : {"decorator_list", "keywords", "type_params"})
      if (const auto *list = ast::nodeList(*classDef, field);
          list && !list->empty()) {
        // ⛔ Only `@dataclass` adds names among the decorators: any other
        // has to return the class it was given (applyClassDecorators), whose
        // names are the ones its body spells.
        if (field == "decorator_list" &&
            llvm::none_of(*list, [](const parser::NodePtr &decorator) {
              const parser::Node *named = decorator.get();
              if (named && named->kind == "Call")
                named = ast::node(*named, "func");
              return named && llvm::StringRef(ast::qualifiedName(named))
                                  .ends_with("dataclass");
            }))
          continue;
        why = "'" + shown + "' has " +
              (field == "decorator_list" ? "a decorator"
               : field == "keywords"     ? "a class keyword"
                                         : "type parameters") +
              ", which add names its body does not spell";
        return std::nullopt;
      }
    if (const auto *body = ast::nodeList(*classDef, "body"))
      for (const parser::NodePtr &member : *body) {
        if (!member)
          continue;
        if (member->kind == "FunctionDef" ||
            member->kind == "AsyncFunctionDef" || member->kind == "ClassDef") {
          if (auto name = ast::string(*member, "name");
              name && !llvm::StringRef(*name).contains('$'))
            names.insert(std::string(*name));
        } else if (member->kind == "AnnAssign") {
          annotated = true;
          const parser::Node *target = ast::node(*member, "target");
          if (target && target->kind == "Name" && ast::node(*member, "value"))
            names.insert(std::string(ast::nameSpelling(*target)));
        } else if (member->kind == "Assign") {
          if (const auto *targets = ast::nodeList(*member, "targets"))
            for (const parser::NodePtr &target : *targets)
              if (target && target->kind == "Name") {
                if (ast::nameSpelling(*target) == "__slots__") {
                  why = "'" + shown + "' declares __slots__";
                  return std::nullopt;
                }
                names.insert(std::string(ast::nameSpelling(*target)));
              }
        }
      }
    if (auto bases = declaredClassBases.find(current);
        bases != declaredClassBases.end())
      for (const std::string &base : bases->second) {
        if (base == "object" || base == "builtins.object")
          continue;
        if (!declaredClassNodes.count(base)) {
          why = "'" + shown + "' derives from '" +
                py::contracts::displayClassNameForContract(base) +
                "', whose attributes are not in a class body here";
          return std::nullopt;
        }
        worklist.push_back(base);
      }
  }
  if (annotated)
    names.insert("__annotate_func__");
  return names;
}

std::optional<Value>
ModuleEmitter::tryEmitDirCall(const parser::Node &expr,
                              const parser::Node *calleeNode) {
  if (!callsUnshadowedBuiltin(calleeNode, "dir"))
    return std::nullopt;
  parser::SourceRange range = expr.range;
  auto refuse = [&](std::string reason) -> std::optional<Value> {
    diagnostics.push_back(
        parser::Diagnostic{parser::Severity::Error, range.start,
                           std::move(reason)});
    return emitNone(expr);
  };
  const auto *args = ast::nodeList(expr, "args");
  const auto *keywords = ast::nodeList(expr, "keywords");
  if ((keywords && !keywords->empty()) || (args && args->size() > 1) ||
      (args && args->size() == 1 &&
       (!args->front() || args->front()->kind == "Starred")))
    return refuse("dir() takes at most one positional argument");
  if (!args || args->empty())
    return refuse("dir() without an argument lists the names of the scope it "
                  "runs in, which is not supported here");
  const parser::NodePtr &argument = args->front();
  mlir::Type subject = types.widenLiteral(types.inferExpr(argument.get()));
  mlir::Type listOfStr = types.contract("builtins.list", {types.strType()});
  auto listOf = [&](const std::set<std::string> &names) {
    std::vector<parser::NodePtr> elements;
    for (const std::string &name : names)
      elements.push_back(synth::strConstant(name, range));
    parser::NodePtr list = parser::makeNode("List", range);
    parser::addField(*list, "elts", std::move(elements));
    return list;
  };
  std::string why;
  if (auto classType = mlir::dyn_cast_if_present<py::TypeType>(subject)) {
    auto bound =
        mlir::dyn_cast_if_present<py::ContractType>(classType.getInstanceType());
    if (!bound)
      return refuse("dir() needs a class this program declares, and " +
                    typeText(subject) + " is not one");
    if (classValueIsExact(argument.get(), subject)) {
      std::optional<std::set<std::string>> names =
          classDirNames(bound.getContractName(), why);
      if (!names)
        return refuse("dir() of this class is not supported: " + why);
      (void)emitExpr(argument.get());
      parser::NodePtr list = listOf(*names);
      synthesizedIteratorDefs.push_back(list);
      return emitExpr(list.get());
    }
    llvm::StringMap<std::set<std::string>> namesBySpelling;
    for (const std::string &candidate :
         classValueCandidates(bound.getContractName())) {
      std::optional<std::set<std::string>> names =
          classDirNames(candidate, why);
      if (!names)
        return refuse("dir() of a class value of '" +
                      py::contracts::displayClassNameForContract(
                          bound.getContractName()) +
                      "' is not supported: " + why);
      llvm::StringRef spelling = candidate;
      spelling.consume_front("builtins.");
      namesBySpelling[spelling] = std::move(*names);
    }
    Value classValue = emitExpr(argument.get());
    return emitClassValueDispatch(
        expr, classValue, bound, listOfStr, [&](llvm::StringRef spelling) {
          return listOf(namesBySpelling[spelling]);
        });
  }
  // ⭐ AN INSTANCE ADDS WHAT IS IN ITS `__dict__`: the fields its methods
  // store, for the class it has at run time -- one an `__init__` always
  // stores is listed, one that may be unset is asked of the instance.
  auto contract = mlir::dyn_cast_if_present<py::ContractType>(subject);
  if (!contract || !declaredClassNodes.count(contract.getContractName()))
    return refuse("dir() of " + typeText(subject) +
                  " is not supported here: it lists a class this program "
                  "declares, or an instance of one");
  std::optional<llvm::SmallVector<std::string, 8>> hierarchy =
      computedLookupHierarchy(expr, "dir()", subject);
  if (!hierarchy)
    return emitNone(expr);
  for (const std::string &member : *hierarchy)
    if (!classDirNames(member, why))
      return refuse("dir() of an instance of '" +
                    py::contracts::displayClassNameForContract(
                        contract.getContractName()) +
                    "' is not supported: " + why);
  std::string cls = hierarchy->front();
  std::string key = "dir|" + cls;
  std::string symbol;
  if (auto found = computedLookupDispatchers.find(key);
      found != computedLookupDispatchers.end()) {
    symbol = found->second;
  } else {
    symbol = "__lydir$" + std::to_string(++syntheticFunctionCounter);
    computedLookupDispatchers[key] = symbol;
    llvm::SmallVector<const parser::Node *, 1> paramNodes;
    parser::NodePtr returns = synth::subscript(
        synth::name("list", range), synth::name("str", range), range);
    parser::NodePtr def =
        synth::functionDef(symbol, {synth::Param{"__ly_recv", nullptr}}, {}, {},
                           std::move(returns), {}, range, &paramNodes);
    synthesizedIteratorDefs.push_back(def);
    types.overrideParameterType(paramNodes[0],
                                types.lookupClass(cls).value_or(mlir::Type()));
    FunctionSignature sig = types.functionSignature(*def);
    types.bindRootSymbol(symbol, sig.publicCallable);
    auto emitBody = [this, def, symbol, sig, range, members = *hierarchy] {
      auto receiver = [&] { return synth::name("__ly_recv", range); };
      auto names = [&] { return synth::name("__ly_names", range); };
      std::vector<parser::NodePtr> body;
      for (const std::string &member : members) {
        std::string unused;
        std::set<std::string> listed = *classDirNames(member, unused);
        mlir::Type memberType =
            types.lookupClass(member).value_or(mlir::Type());
        llvm::SmallVector<std::string, 4> probed;
        llvm::SmallVector<llvm::StringRef, 8> line{member};
        llvm::StringSet<> seen;
        std::set<std::string> stored;
        while (!line.empty()) {
          llvm::StringRef current = line.pop_back_val();
          if (!seen.insert(current).second)
            continue;
          if (auto fields = declaredClassStoredFields.find(current);
              fields != declaredClassStoredFields.end())
            for (const auto &field : fields->second)
              stored.insert(field.getKey().str());
          if (auto bases = declaredClassBases.find(current);
              bases != declaredClassBases.end())
            for (const std::string &base : bases->second)
              line.push_back(base);
        }
        for (const std::string &field : stored) {
          if (listed.count(field))
            continue;
          if (attributePresence(memberType, field) ==
              AttributePresence::Present)
            listed.insert(field);
          else
            probed.push_back(field);
        }
        std::vector<parser::NodePtr> arm;
        std::vector<parser::NodePtr> elements;
        for (const std::string &name : listed)
          elements.push_back(synth::strConstant(name, range));
        parser::NodePtr list = parser::makeNode("List", range);
        parser::addField(*list, "elts", std::move(elements));
        arm.push_back(synth::assign(names(), std::move(list), range));
        for (const std::string &field : probed)
          arm.push_back(synth::ifStmt(
              synth::call(
                  synth::name(attributeProbe(member, field, range), range),
                  {receiver()}, range),
              {synth::exprStmt(synth::methodCall(names(), "append",
                                                 {synth::strConstant(field,
                                                                     range)},
                                                 range),
                               range)},
              {}, range));
        if (!probed.empty())
          arm.push_back(synth::exprStmt(
              synth::methodCall(names(), "sort", {}, range), range));
        arm.push_back(synth::returnStmt(names(), range));
        body.push_back(synth::ifStmt(
            synth::compare(
                synth::call(synth::name("type", range), {receiver()}, range),
                "Is", synth::name(member, range), range),
            std::move(arm), {}, range));
      }
      body.push_back(synth::raiseCall(
          "TypeError", "dir(): the instance's class is none of its line",
          range));
      if (parser::Field *field = parser::findField(*def, "body"))
        field->value = std::move(body);
      emitDispatcherBody(*def, symbol, sig, /*suppressGate=*/false);
    };
    if (classDeclarationsEmitted)
      emitBody();
    else
      deferredDispatcherBodies.push_back(std::move(emitBody));
  }
  parser::NodePtr call =
      synth::call(synth::name(symbol, range), {argument}, range);
  synthesizedIteratorDefs.push_back(call);
  return emitExpr(call.get());
}

// ⭐ `object.__repr__(x)` IS OBJECT'S OWN SLOT, NOT x's. It is how a
// class's `__repr__` reaches the default it overrides, and how unittest's
// `safe_repr` falls back when `repr(x)` raises -- and it reached the manifest
// `object.__repr__`, whose body dispatches through the class's slot: it
// answered "A()" for an A that overrides `__repr__`, and "1" for an int,
// where CPython prints `<__main__.A object at 0x...>` and `<int object at
// 0x...>`. Each spelling here is what CPython's object does:
//   __repr__  `<{module}.{qualname} object at {hex(id(x))}>`, the module left
//             out for a builtin (object_repr);
//   __str__   `repr(x)` -- type(x).__repr__ (object_str);
//   __hash__  the pointer hash of x, past the class's own `__hash__`.
// ⛔ `__eq__` and `__ne__` are refused: for two different objects they answer
// NotImplemented, which this compiler has no value for.
std::optional<Value>
ModuleEmitter::tryEmitObjectSlotCall(const parser::Node &expr,
                                     const parser::Node *calleeNode) {
  if (!calleeNode || calleeNode->kind != "Attribute")
    return std::nullopt;
  const parser::Node *owner = ast::node(*calleeNode, "value");
  if (!owner || owner->kind != "Name" ||
      ast::nameSpelling(*owner) != "object" || programBindsName("object"))
    return std::nullopt;
  std::optional<std::string_view> attr = ast::string(*calleeNode, "attr");
  if (!attr)
    return std::nullopt;
  const auto *args = ast::nodeList(expr, "args");
  const auto *keywords = ast::nodeList(expr, "keywords");
  if ((keywords && !keywords->empty()) || !args ||
      llvm::any_of(*args, [](const parser::NodePtr &arg) {
        return !arg || arg->kind == "Starred";
      }))
    return std::nullopt;
  parser::SourceRange range = expr.range;
  if (*attr == "__eq__" || *attr == "__ne__") {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, range.start,
        "object." + std::string(*attr) +
            "(a, b) answers NotImplemented for two different objects, which "
            "this compiler has no value for; compare with `a is b`"});
    return emitNone(expr);
  }
  if ((*attr != "__repr__" && *attr != "__str__" && *attr != "__hash__") ||
      args->size() != 1)
    return std::nullopt;
  if (*attr == "__str__") {
    parser::NodePtr call = synth::reprCall(args->front(), range);
    synthesizedIteratorDefs.push_back(call);
    return emitExpr(call.get());
  }
  Value subject = emitExpr(args->front().get());
  if (*attr == "__hash__") {
    Value boxed = coerceValue(subject, types.object(), expr);
    CallInferenceResult inference = types.inferMethodCallWithEvidence(
        types.object(), "__ly_identity_hash__", {});
    if (!inference)
      return std::nullopt;
    Value posPack = emitPack({});
    Value namePack = emitPack({});
    Value valuePack = emitPack({});
    auto call = py::CallOp::create(
        builder, loc(expr), mlir::TypeRange{inference.resultType},
        callProtocolFor(inference), boxed.value, posPack.value,
        namePack.value, valuePack.value);
    call->setAttr("ly.bound_method",
                  builder.getStringAttr("__ly_identity_hash__"));
    return Value{call.getResults().front(), inference.resultType};
  }
  std::string temp =
      "__ly_objrepr" + std::to_string(++syntheticFunctionCounter);
  values[temp] = subject;
  types.bindSymbol(temp, subject.type);
  auto forget = llvm::scope_exit([&] { values.erase(temp); });
  auto classOf = [&] {
    return synth::call(synth::name("type", range), {synth::name(temp, range)},
                       range);
  };
  auto classAttribute = [&](llvm::StringRef name) {
    return synth::attribute(classOf(), name, range);
  };
  parser::NodePtr className = synth::ifExp(
      synth::compare(classAttribute("__module__"), "Eq",
                     synth::strConstant("builtins", range), range),
      classAttribute("__qualname__"),
      synth::binOp(synth::binOp(classAttribute("__module__"), "Add",
                                synth::strConstant(".", range), range),
                   "Add", classAttribute("__qualname__"), range),
      range);
  parser::NodePtr address = synth::call(
      synth::name("hex", range),
      {synth::call(synth::name("id", range), {synth::name(temp, range)},
                   range)},
      range);
  parser::NodePtr text = synth::binOp(
      synth::binOp(
          synth::binOp(synth::binOp(synth::strConstant("<", range), "Add",
                                    std::move(className), range),
                       "Add", synth::strConstant(" object at ", range), range),
          "Add", std::move(address), range),
      "Add", synth::strConstant(">", range), range);
  synthesizedIteratorDefs.push_back(text);
  return emitExpr(text.get());
}

// CPython's BaseException_str and BaseException_repr, written over `args`:
//   __str__   "" for no argument, str(args[0]) for one, repr(args) otherwise;
//   __repr__  the class's name, then "(repr(args[0]))" or repr(args).
// ⛔ Not the runtime's BaseException text: a builtin exception keeps the text
// its own class rendered at construction (KeyError stores its key's repr), and
// a program exception's `__str__` is reached by calling through the value.
// `args` is the one record no class rewrites.
std::optional<Value> ModuleEmitter::tryEmitBaseExceptionText(
    const parser::Node &expr, mlir::Type owner, llvm::StringRef methodName,
    const parser::NodePtr &subject) {
  auto contract = mlir::dyn_cast_if_present<py::ContractType>(owner);
  if (!contract || contract.getContractName() != "builtins.BaseException" ||
      (methodName != "__str__" && methodName != "__repr__"))
    return std::nullopt;
  const auto *args = ast::nodeList(expr, "args");
  const auto *keywords = ast::nodeList(expr, "keywords");
  if (!args || args->size() != 1 || (keywords && !keywords->empty()))
    return std::nullopt;
  parser::SourceRange range = expr.range;
  Value receiver = coerceValue(emitExpr(subject.get()), owner, *subject);
  std::string receiverName =
      "__ly_exctext" + std::to_string(++syntheticFunctionCounter);
  values[receiverName] = receiver;
  types.bindSymbol(receiverName, receiver.type);
  parser::NodePtr argsRead =
      synth::attribute(synth::name(receiverName, range), "args", range);
  synthesizedIteratorDefs.push_back(argsRead);
  Value argsValue = emitExpr(argsRead.get());
  std::string argsName = receiverName + "$args";
  values[argsName] = argsValue;
  types.bindSymbol(argsName, argsValue.type);
  auto forget = llvm::scope_exit([&] {
    values.erase(receiverName);
    values.erase(argsName);
  });
  auto argsNamed = [&] { return synth::name(argsName, range); };
  auto lengthIs = [&](std::int64_t n) {
    return synth::compare(synth::lenCall(argsNamed(), range), "Eq",
                          synth::intConstant(n, range), range);
  };
  auto first = [&] {
    return synth::subscript(argsNamed(), synth::intConstant(0, range), range);
  };
  parser::NodePtr text;
  if (methodName == "__str__") {
    text = synth::ifExp(
        lengthIs(0), synth::strConstant("", range),
        synth::ifExp(lengthIs(1),
                     synth::call(synth::name("str", range), {first()}, range),
                     synth::reprCall(argsNamed(), range), range),
        range);
  } else {
    parser::NodePtr className = synth::attribute(
        synth::call(synth::name("type", range),
                    {synth::name(receiverName, range)}, range),
        "__name__", range);
    parser::NodePtr one = synth::binOp(
        synth::binOp(synth::strConstant("(", range), "Add",
                     synth::reprCall(first(), range), range),
        "Add", synth::strConstant(")", range), range);
    text = synth::binOp(std::move(className), "Add",
                        synth::ifExp(lengthIs(1), std::move(one),
                                     synth::reprCall(argsNamed(), range),
                                     range),
                        range);
  }
  synthesizedIteratorDefs.push_back(text);
  return emitExpr(text.get());
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

// ⛔ `int` IS CLOSED although CPython's bool derives from it: a bool is a
// truth bit here and an int is not, so a `type[int]` never names bool (the
// coercion that would put it there is refused), and `t()` through one is
// `int()`.
bool ModuleEmitter::classIsClosed(llvm::StringRef contract) const {
  if (contract == "builtins.object" || contract == "typing.Any" ||
      contract == "builtins.type")
    return false;
  // ⭐ AND `int`: bool is its builtin subclass, and a list or field declared
  // int keeps a bool's object -- `type(xs[0])` for `xs: list[int] = [True]`
  // is bool. ⛔ Not found below, which walks only the program's classes.
  if (contract == "builtins.int")
    return false;
  llvm::StringRef leaf = contract.rsplit('.').second;
  if (leaf.empty())
    leaf = contract;
  // A generic's own instantiations are not its subclasses.
  std::string ownSpecialization = (contract + "$spec").str();
  for (const auto &entry : classMros)
    if (entry.getKey() != contract &&
        !entry.getKey().starts_with(ownSpecialization) &&
        (llvm::is_contained(entry.second, contract) ||
         llvm::is_contained(entry.second, leaf)))
      return false;
  // ⭐ AND EVERY CLASS THE PROGRAM DECLARES, emitted yet or not. `classMros`
  // holds what has been emitted, and an imported module's functions are
  // emitted before the module that imports it declares its classes -- so a
  // `type[Base]` parameter there took Base for closed: `cls.__name__` folded
  // to "Base", `cls(1)` built a Base and `cls is Base` was True, for a Sub,
  // silently.
  if (!programSubclassesOf(contract).empty())
    return false;
  if (py::exceptions::findByName(leaf))
    for (const py::exceptions::BuiltinExceptionInfo &entry :
         py::exceptions::kBuiltinExceptions)
      if (entry.name != leaf &&
          py::exceptions::isBuiltinExceptionSubclassName(entry.name, leaf))
        return false;
  return true;
}

bool ModuleEmitter::classValueIsExact(const parser::Node *expr,
                                      mlir::Type type) const {
  auto typeType =
      mlir::dyn_cast_if_present<py::TypeType>(types.widenLiteral(type));
  if (!typeType)
    return false;
  auto contract =
      mlir::dyn_cast_if_present<py::ContractType>(typeType.getInstanceType());
  if (!contract || classIsClosed(contract.getContractName()))
    return true;
  // `Box[int]` names the instantiation itself.
  if (expr && expr->kind == "Subscript" && types.namesAType(expr))
    return true;
  if (!expr || (expr->kind != "Name" && expr->kind != "Attribute"))
    return false;
  std::string spelling = expr->kind == "Name"
                             ? std::string(ast::nameSpelling(*expr))
                             : ast::qualifiedName(expr);
  if (spelling.empty())
    return false;
  // `type` and `list` spelled bare name their classes, as a declared class
  // name does.
  if (expr->kind == "Name" && values.find(spelling) == values.end() &&
      !types.lookupSymbol(spelling) && types.builtinClassValue(spelling))
    return true;
  std::optional<mlir::Type> bound = types.lookupClass(spelling);
  return bound && (*bound == mlir::Type(contract) || *bound == type);
}

llvm::SmallVector<std::string, 8>
ModuleEmitter::classValueCandidates(llvm::StringRef contract) const {
  llvm::SmallVector<std::string, 8> candidates;
  auto add = [&](llvm::StringRef name) {
    if (!llvm::is_contained(candidates, name))
      candidates.push_back(name.str());
  };
  llvm::StringRef leaf = contract.rsplit('.').second;
  if (leaf.empty())
    leaf = contract;
  for (const auto &entry : classMros)
    if (entry.getKey() != contract &&
        (llvm::is_contained(entry.second, contract) ||
         llvm::is_contained(entry.second, leaf)))
      add(entry.getKey());
  // And the program's subclasses not emitted yet (see classIsClosed).
  for (const std::string &subclass : programSubclassesOf(contract))
    add(subclass);
  // Builtin exceptions are built by the bound's arm with the value's class
  // word, except those with an initializer of their own.
  for (llvm::StringRef own : py::exceptions::kBuiltinExceptionsWithOwnInit)
    if (own != contract &&
        py::exceptions::isBuiltinExceptionSubclassContract(own, contract))
      add(own);
  add(contract);
  return candidates;
}

// ⭐ A CLASS VALUE THE STATIC TYPE DOES NOT DECIDE DISPATCHES ON WHICH CLASS IT
// NAMES. The call, the class attribute and the class method used to be
// re-spelled as the STATIC class, which was right only while such a value
// could not exist -- with class values carrying their type object it would
// construct a Base for a value naming Sub:
//
//     for c in [A, B]:      # list[type[A]]
//         print(c(2).n)     # printed A's n twice; CPython builds a B
//
// The arms are the program's own spelling of each class, chosen by identity
// (`t is D`), so each arm is the static construction or attribute read that
// class already compiles to. The bound itself is last, the fallback.
parser::NodePtr ModuleEmitter::raisingCall(parser::SourceRange range,
                                           llvm::StringRef exception,
                                           const std::string &message,
                                           mlir::Type resultType) {
  std::string symbol =
      "__lyraise$" + std::to_string(++syntheticFunctionCounter);
  parser::NodePtr def = synth::functionDef(
      symbol, {}, {}, {synth::raiseCall(exception, message, range)}, nullptr,
      {}, range);
  synthesizedIteratorDefs.push_back(def);
  types.pinResultType(def.get(), resultType);
  FunctionSignature sig = types.functionSignature(*def);
  types.bindRootSymbol(symbol, sig.publicCallable);
  emitDispatcherBody(*def, symbol, sig, /*suppressGate=*/false);
  return synth::call(synth::name(symbol, range), {}, range);
}

std::optional<Value> ModuleEmitter::emitClassValueDispatch(
    const parser::Node &anchor, Value classValue, py::ContractType bound,
    mlir::Type resultType,
    llvm::function_ref<parser::NodePtr(llvm::StringRef spelling)> armFor,
    llvm::ArrayRef<std::string> capturedNames,
    std::function<std::string(llvm::StringRef spelling)> armMismatch) {
  llvm::SmallVector<std::string, 8> candidates =
      classValueCandidates(bound.getContractName());
  // ⛔ An instantiation of a generic subclass has no spelling of its own to
  // put in an arm (`Labeled$spec0` is the compiler's name for it).
  for (const std::string &candidate : candidates)
    if (llvm::StringRef(candidate).contains('$')) {
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, anchor.range.start,
          "a class value of '" +
              py::contracts::displayClassNameForContract(
                  bound.getContractName()) +
              "' may name an instantiation of a generic subclass, which a "
              "call through the value cannot spell; call the class by name"});
      return emitNone(anchor);
    }
  auto spell = [](llvm::StringRef contract) {
    contract.consume_front("builtins.");
    return contract.str();
  };
  // ⭐ A CANDIDATE NOT DECLARED YET IS DISPATCHED FROM A FUNCTION WRITTEN
  // LATER. An imported module's body is emitted before the importer declares
  // the subclasses that are candidates here, and an arm that constructs or
  // reads one cannot be typed until it is. The arms' syntax is built now and
  // typed in a function emitted once every class is declared, which takes the
  // class value and what the arms read from this scope.
  // ⛔ Not for a builtin exception's bound: its arm reads the value's class
  // word from this scope (`pendingConstructionClass`).
  bool laterCandidate =
      !py::exceptions::findByContract(bound.getContractName()) &&
      llvm::any_of(candidates, [&](const std::string &candidate) {
        return declaredClassNodes.count(candidate) &&
               !classMros.count(candidate);
      });
  if (laterCandidate && !classDeclarationsEmitted) {
    parser::SourceRange range = anchor.range;
    std::string symbol =
        "__lyclassdisp$" + std::to_string(++syntheticFunctionCounter);
    std::string valueName = symbol + "$cls";
    std::vector<std::pair<std::string, parser::NodePtr>> arms;
    for (const std::string &candidate : candidates)
      arms.push_back({spell(candidate), armFor(spell(candidate))});
    llvm::SmallVector<synth::Param, 4> params{
        synth::Param{"__ly_cls", nullptr}};
    for (const std::string &name : capturedNames)
      params.push_back(synth::Param{name, nullptr});
    llvm::SmallVector<const parser::Node *, 4> paramNodes;
    parser::NodePtr def = synth::functionDef(symbol, params, {}, {}, nullptr,
                                             {}, range, &paramNodes);
    synthesizedIteratorDefs.push_back(def);
    types.overrideParameterType(paramNodes[0], classValue.type);
    for (auto [index, name] : llvm::enumerate(capturedNames))
      types.overrideParameterType(paramNodes[index + 1],
                                  values.lookup(name).type);
    types.pinResultType(def.get(), resultType);
    FunctionSignature sig = types.functionSignature(*def);
    types.bindRootSymbol(symbol, sig.publicCallable);
    deferredDispatcherBodies.push_back(
        [this, def, symbol, sig, range, arms = std::move(arms),
         armMismatch = std::move(armMismatch)] {
          std::vector<parser::NodePtr> body;
          for (auto [index, arm] : llvm::enumerate(arms)) {
            std::string mismatch = armMismatch ? armMismatch(arm.first) : "";
            parser::NodePtr answer =
                mismatch.empty()
                    ? synth::returnStmt(arm.second, range)
                    : synth::raiseCall("TypeError", mismatch, range);
            if (index + 1 == arms.size()) {
              body.push_back(std::move(answer));
              continue;
            }
            body.push_back(synth::ifStmt(
                synth::compare(synth::name("__ly_cls", range), "Is",
                               synth::name(arm.first, range), range),
                {std::move(answer)}, {}, range));
          }
          if (parser::Field *field = parser::findField(*def, "body"))
            field->value = std::move(body);
          emitDispatcherBody(*def, symbol, sig, /*suppressGate=*/false);
        });
    values[valueName] = classValue;
    types.bindSymbol(valueName, classValue.type);
    auto forget = llvm::scope_exit([&] { values.erase(valueName); });
    std::vector<parser::NodePtr> callArguments{synth::name(valueName, range)};
    for (const std::string &name : capturedNames)
      callArguments.push_back(synth::name(name, range));
    parser::NodePtr call = synth::call(synth::name(symbol, range),
                                       std::move(callArguments), range);
    synthesizedIteratorDefs.push_back(call);
    return emitExpr(call.get());
  }
  std::function<mlir::Value(std::size_t)> chain =
      [&](std::size_t index) -> mlir::Value {
    auto arm = [&]() -> mlir::Value {
      std::string mismatch =
          armMismatch ? armMismatch(spell(candidates[index])) : "";
      parser::NodePtr node =
          mismatch.empty()
              ? armFor(spell(candidates[index]))
              : raisingCall(anchor.range, "TypeError", mismatch, resultType);
      synthesizedIteratorDefs.push_back(node);
      Value armValue = emitExpr(node.get());
      pendingConstructionClass.reset();
      return coerceValue(armValue, resultType, anchor).value;
    };
    if (index + 1 == candidates.size())
      return arm();
    mlir::Type candidateType = types.contract(candidates[index]);
    auto candidate = py::TypeObjectOp::create(
        builder, loc(anchor), types.typeObject(candidateType), candidateType);
    mlir::Value same =
        py::IsOp::create(builder, loc(anchor), builder.getI1Type(),
                         classValue.value, candidate.getResult())
            .getResult();
    return emitValueDiamond(loc(anchor), same, resultType, arm,
                            [&] { return chain(index + 1); });
  };
  return Value{chain(0), resultType};
}

std::optional<Value>
ModuleEmitter::emitClassValueCall(const parser::Node &expr,
                                  const parser::Node *calleeNode) {
  if (!calleeNode)
    return std::nullopt;
  // `t(args)` constructs -- `self.cls(args)` too, a field holding a class;
  // `t.m(args)` calls a class's method by name.
  const parser::Node *classNode = calleeNode;
  std::optional<std::string> method;
  if (calleeNode->kind == "Attribute" &&
      !mlir::isa_and_nonnull<py::TypeType>(
          types.widenLiteral(types.inferExpr(calleeNode)))) {
    classNode = ast::node(*calleeNode, "value");
    if (auto attr = ast::string(*calleeNode, "attr"))
      method = std::string(*attr);
    if (!classNode || !method)
      return std::nullopt;
  }
  mlir::Type classType = types.widenLiteral(types.inferExpr(classNode));
  auto typeObject = mlir::dyn_cast_if_present<py::TypeType>(classType);
  if (!typeObject || classValueIsExact(classNode, classType))
    return std::nullopt;
  auto bound =
      mlir::dyn_cast_if_present<py::ContractType>(typeObject.getInstanceType());
  if (!bound)
    return std::nullopt;
  auto refuse = [&](std::string reason) -> std::optional<Value> {
    diagnostics.push_back(parser::Diagnostic{parser::Severity::Error,
                                             expr.range.start,
                                             std::move(reason)});
    return emitNone(expr);
  };
  if (bound.getContractName() == "builtins.object")
    return refuse("calling through a class value that may name any class is "
                  "not supported: each class it could name would be an arm");
  const auto *args = ast::nodeList(expr, "args");
  const auto *keywords = ast::nodeList(expr, "keywords");
  if ((args && llvm::any_of(*args, [](const parser::NodePtr &arg) {
         return !arg || arg->kind == "Starred";
       })) ||
      (keywords && llvm::any_of(*keywords, [](const parser::NodePtr &kw) {
         return !kw || !ast::string(*kw, "arg");
       })))
    return refuse("a call through a class value takes its arguments spelled "
                  "out, not unpacked");
  // The class value first, then the arguments, each once and in order.
  Value classValue = emitExpr(classNode);
  std::vector<std::string> temps;
  auto cleanup = llvm::scope_exit([&] {
    for (const std::string &name : temps)
      values.erase(name);
  });
  auto bindTemp = [&](const parser::Node *node) {
    std::string name =
        "__ly_classcall" + std::to_string(++syntheticFunctionCounter);
    Value value = emitExpr(node);
    values[name] = value;
    types.bindSymbol(name, value.type);
    temps.push_back(name);
    return name;
  };
  std::vector<std::string> positional;
  if (args)
    for (const parser::NodePtr &arg : *args)
      positional.push_back(bindTemp(arg.get()));
  std::vector<std::pair<std::string, std::string>> named;
  if (keywords)
    for (const parser::NodePtr &kw : *keywords)
      named.emplace_back(std::string(*ast::string(*kw, "arg")),
                         bindTemp(ast::node(*kw, "value")));
  parser::SourceRange range = expr.range;
  // ⭐ THE BOUND'S ARM BUILDS EVERY BUILTIN EXCEPTION BELOW IT. They share
  // one layout and one initializer here, which takes the class word as an
  // argument, so the arm passes the value's class instead of enumerating
  // sixty classes -- one of which (`_io.UnsupportedOperation`) no program
  // can even spell.
  bool builtinExceptionBound =
      !method && py::exceptions::findByContract(bound.getContractName());
  // ⭐ A CLASS WHOSE `__init__` CANNOT TAKE THE CALL IS AN ARM THAT RAISES.
  // `cls(name)` over `type[TestCase]` names every subclass, and a subclass's
  // `__init__` may take anything -- a constructor is not an override a type
  // checker holds to its base's signature. Each arm was compiled as that
  // class's own construction, so one subclass with another `__init__`
  // refused the call for every class: unittest's loader, and
  // `self.failureException(msg)` over every exception class of the program.
  // CPython raises only if that class is the one called, and so does the
  // arm: its binding error word for word.
  // ⛔ An argument of a TYPE the `__init__` does not declare raises too, with
  // Lython's reason: CPython would build the object, and an object whose
  // field holds a value of another type is not one this compiler can lay
  // out. Refusing the program instead is what stood here.
  // ⛔ Asked as each arm is WRITTEN, with the argument types copied: a
  // candidate declared after this call (a test class, for unittest's loader)
  // has no `__init__` to ask yet.
  std::function<std::string(llvm::StringRef)> constructionMismatch;
  if (!method) {
    llvm::SmallVector<mlir::Type, 4> positionalTypes;
    for (const std::string &temp : positional)
      positionalTypes.push_back(values.lookup(temp).type);
    llvm::SmallVector<std::pair<std::string, mlir::Type>, 2> keywordTypes;
    for (const auto &[keyword, temp] : named)
      keywordTypes.push_back({keyword, values.lookup(temp).type});
    constructionMismatch = [this, positionalTypes, keywordTypes](
                               llvm::StringRef spelling) -> std::string {
      std::optional<mlir::Type> classType = types.lookupClass(spelling);
      if (!classType || !*classType)
        return "";
      std::optional<MethodBinding> init =
          lookupClassMethod(*classType, "__init__");
      if (!init || !init->method || init->definingClass.empty() ||
          init->kind != "instance")
        return "";
      std::string qualname =
          py::contracts::pythonQualnameForContract(init->definingClass) +
          ".__init__";
      llvm::SmallVector<std::string, 2> keywordNames;
      for (const auto &entry : keywordTypes)
        keywordNames.push_back(entry.first);
      std::string binding = callBindingError(
          qualname, *init->method, /*bound=*/true,
          static_cast<unsigned>(positionalTypes.size()), keywordNames);
      if (!binding.empty())
        return binding;
      const FunctionSignature &sig = init->bodySignature;
      auto check = [&](llvm::StringRef parameter, mlir::Type declared,
                       mlir::Type actual) -> std::string {
        if (!declared || !actual ||
            inlineArgumentFits(types.widenLiteral(actual),
                               types.widenLiteral(declared)))
          return "";
        return qualname + "() is not supported here: argument '" +
               parameter.str() + "' is declared " +
               pythonTypeSpelling(types.widenLiteral(declared)) +
               " and this call gives it " +
               pythonTypeSpelling(types.widenLiteral(actual));
      };
      for (auto [index, actual] : llvm::enumerate(positionalTypes)) {
        std::size_t slot = index + 1;
        if (slot >= sig.positionalTypes.size() ||
            slot >= sig.positionalNames.size())
          break;
        std::string message = check(sig.positionalNames[slot],
                                    sig.positionalTypes[slot], actual);
        if (!message.empty())
          return message;
      }
      for (const auto &[keyword, actual] : keywordTypes) {
        mlir::Type declared;
        for (auto [index, name] : llvm::enumerate(sig.positionalNames))
          if (name == keyword && index < sig.positionalTypes.size())
            declared = sig.positionalTypes[index];
        for (auto [index, name] : llvm::enumerate(sig.kwOnlyNames))
          if (name == keyword && index < sig.kwOnlyTypes.size())
            declared = sig.kwOnlyTypes[index];
        std::string message = check(keyword, declared, actual);
        if (!message.empty())
          return message;
      }
      return "";
    };
  }
  auto armFor = [&](llvm::StringRef spelling) {
    parser::NodePtr callee = synth::name(spelling, range);
    if (builtinExceptionBound &&
        types.lookupClass(spelling) == mlir::Type(bound))
      pendingConstructionClass = classValue;
    if (method)
      callee = synth::attribute(std::move(callee), *method, range);
    std::vector<parser::NodePtr> callArgs;
    for (const std::string &name : positional)
      callArgs.push_back(synth::name(name, range));
    std::vector<parser::NodePtr> callKeywords;
    for (const auto &[keyword, name] : named)
      callKeywords.push_back(
          synth::keyword(keyword, synth::name(name, range), range));
    return synth::callWithKeywords(std::move(callee), std::move(callArgs),
                                   std::move(callKeywords), range);
  };
  // The bound's own answer types the merge: its construction is a `bound`,
  // its method returns what the bound's method declares.
  mlir::Type resultType = bound;
  if (method) {
    parser::NodePtr probe = armFor(bound.getContractName().starts_with(
                                       "builtins.")
                                       ? bound.getContractName().drop_front(9)
                                       : bound.getContractName());
    synthesizedIteratorDefs.push_back(probe);
    resultType = types.widenLiteral(types.inferExpr(probe.get()));
  }
  return emitClassValueDispatch(expr, classValue, bound, resultType, armFor,
                                temps, std::move(constructionMismatch));
}

std::optional<Value>
ModuleEmitter::emitClassValueAttribute(const parser::Node &expr,
                                       Value classValue,
                                       llvm::StringRef attribute) {
  auto typeObject = mlir::dyn_cast_if_present<py::TypeType>(classValue.type);
  auto bound = typeObject ? mlir::dyn_cast_if_present<py::ContractType>(
                                typeObject.getInstanceType())
                          : py::ContractType();
  if (!bound)
    return std::nullopt;
  if (bound.getContractName() == "builtins.object") {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, expr.range.start,
        "reading '" + attribute.str() +
            "' through a class value that may name any class is not "
            "supported: each class it could name would be an arm"});
    return emitNone(expr);
  }
  parser::SourceRange range = expr.range;
  auto armFor = [&](llvm::StringRef spelling) {
    return synth::attribute(synth::name(spelling, range), attribute, range);
  };
  parser::NodePtr probe =
      armFor(bound.getContractName().starts_with("builtins.")
                 ? bound.getContractName().drop_front(9)
                 : bound.getContractName());
  synthesizedIteratorDefs.push_back(probe);
  mlir::Type resultType = types.widenLiteral(types.inferExpr(probe.get()));
  return emitClassValueDispatch(expr, classValue, bound, resultType, armFor);
}

// ⭐ THE CLASS OF AN ALREADY-EMITTED VALUE, as a class value of
// `resultType`: a static class where the value's type decides it, its header's
// where it does not, and a union's active member's, chosen by the tag -- each
// arm unwraps its member and answers the way a value of the member's own type
// would.
Value ModuleEmitter::emitClassOfValue(const parser::Node &anchor, Value value,
                                      mlir::Type resultType) {
  auto classOf = [&](Value subject) -> mlir::Value {
    mlir::Type widened = types.widenLiteral(subject.type);
    mlir::Value classValue;
    mlir::Type classType;
    auto contract = mlir::dyn_cast_if_present<py::ContractType>(widened);
    if (isNoneTypeLike(widened) || !contract) {
      mlir::Type instance = isNoneTypeLike(widened)
                                ? types.contract("types.NoneType")
                            : mlir::isa<py::TypeType>(widened)
                                ? types.contract("builtins.type")
                                : types.object();
      classType = types.typeObject(instance);
      classValue =
          py::TypeObjectOp::create(builder, loc(anchor), classType, instance)
              .getResult();
    } else if (!classIsClosed(contract.getContractName())) {
      classType = types.typeObject(contract);
      classValue = py::TypeOfOp::create(builder, loc(anchor), classType,
                                        subject.value)
                       .getResult();
    } else {
      classType = types.typeObject(contract);
      classValue =
          py::TypeObjectOp::create(builder, loc(anchor), classType, contract)
              .getResult();
    }
    return coerceValue(Value{classValue, classType}, resultType, anchor).value;
  };
  auto unionType =
      mlir::dyn_cast_if_present<py::UnionType>(types.widenLiteral(value.type));
  if (!unionType)
    return Value{classOf(value), resultType};
  llvm::SmallVector<mlir::Type, 4> members(unionType.getMemberTypes().begin(),
                                           unionType.getMemberTypes().end());
  auto classOfMember = [&](mlir::Type member) {
    return classOf(Value{py::UnionUnwrapOp::create(builder, loc(anchor), member,
                                                   value.value)
                             .getResult(),
                         member});
  };
  std::function<mlir::Value(std::size_t)> chain =
      [&](std::size_t index) -> mlir::Value {
    if (index + 1 == members.size())
      return classOfMember(members[index]);
    mlir::Value active =
        py::UnionTestOp::create(builder, loc(anchor), builder.getI1Type(),
                                value.value, mlir::TypeAttr::get(members[index]))
            .getResult();
    return emitValueDiamond(
        loc(anchor), active, resultType,
        [&] { return classOfMember(members[index]); },
        [&] { return chain(index + 1); });
  };
  return Value{chain(0), resultType};
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
  // `type(None)` is NoneType, whose only value None is.
  if (!contract && subject == types.none()) {
    (void)emitExpr(subjectNode);
    mlir::Type noneClass = types.contract("types.NoneType");
    mlir::Type typeType = types.typeObject(noneClass);
    auto object =
        py::TypeObjectOp::create(builder, loc(expr), typeType, noneClass);
    return Value{object.getResult(), typeType};
  }
  if (mlir::isa_and_nonnull<py::UnionType>(subject)) {
    Value receiver = emitExpr(subjectNode);
    mlir::Type resultType = types.widenLiteral(types.inferExpr(&expr));
    if (!mlir::isa_and_nonnull<py::TypeType>(resultType))
      resultType = types.typeObject(types.object());
    return emitClassOfValue(expr, receiver, resultType);
  }
  // A class's class is `type`: no class here declares a metaclass.
  if (mlir::isa_and_nonnull<py::TypeType>(subject)) {
    (void)emitExpr(subjectNode);
    mlir::Type metaclass = types.contract("builtins.type");
    mlir::Type typeType = types.typeObject(metaclass);
    auto object =
        py::TypeObjectOp::create(builder, loc(expr), typeType, metaclass);
    return Value{object.getResult(), typeType};
  }
  if (!contract)
    return refuse("type(x) needs a statically resolved class, and " +
                  typeText(subject) + " is not one");
  llvm::StringRef contractName = contract.getContractName();
  // ⭐ A CLASS THE STATIC TYPE DOES NOT DECIDE IS READ AT RUN TIME. `x: A =
  // B()` makes the static class A and the runtime class B; the value's header
  // (or its box) keeps the class word, and the class value is that type
  // object. It was refused, which took `type(self).__qualname__`,
  // `type(a) is type(b)` and `type(self)(name)` with it.
  if (contractName == "builtins.object" || contractName == "typing.Any" ||
      !classIsClosed(contractName)) {
    Value receiver = emitExpr(subjectNode);
    mlir::Type bound = contractName == "typing.Any" ? types.object()
                                                    : mlir::Type(contract);
    mlir::Type typeType = types.typeObject(bound);
    auto dynamic =
        py::TypeOfOp::create(builder, loc(expr), typeType, receiver.value);
    return Value{dynamic.getResult(), typeType};
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

  // ⭐ A CLASS VALUE THE STATIC TYPE DOES NOT DECIDE IS TESTED AT RUN TIME:
  // the target names its bound or any subclass of it, so the static answer
  // for the bound -- what a `t: type[Base]` used to get -- is wrong for a
  // value naming a subclass. The value's own class walks its MRO for it.
  if (targetHasDynamicClassValue((*args)[1].get()))
    return emitClassValueSubclassTest(expr, args->front().get(),
                                      (*args)[1].get(),
                                      /*subjectIsInstance=*/true);
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

  // ⭐ A PROTOCOL TARGET IS CPython's STRUCTURAL CHECK, which this answers
  // with the base the desugar gave every satisfying class -- exact only where
  // the two agree. CPython refuses outright without @runtime_checkable; with
  // it, it asks only whether the METHOD NAMES exist, so a class that has the
  // names but not the parameters (no base here) or a builtin value would be
  // True there. Each is refused rather than answered False.
  for (mlir::Type target : *targets) {
    auto contract = mlir::dyn_cast_if_present<py::ContractType>(target);
    if (!contract)
      continue;
    auto info = protocolIsinstance.find(contract.getContractName());
    if (info == protocolIsinstance.end())
      continue;
    if (!info->second.empty()) {
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, expr.range.start,
          "isinstance() against the protocol '" +
              contract.getContractName().str() + "' is refused: " +
              info->second});
      return emitNone(expr);
    }
    mlir::Type subject = types.widenLiteral(types.inferExpr(args->front().get()));
    llvm::SmallVector<mlir::Type, 4> members;
    if (auto unionType = mlir::dyn_cast_if_present<py::UnionType>(subject))
      members.append(unionType.getMemberTypes().begin(),
                     unionType.getMemberTypes().end());
    else
      members.push_back(subject);
    for (mlir::Type member : members) {
      member = types.widenLiteral(member);
      if (member == types.none())
        continue;
      auto memberContract = mlir::dyn_cast_if_present<py::ContractType>(member);
      if (!memberContract || !declaredClassBases.count(
                                 memberContract.getContractName())) {
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, expr.range.start,
            "isinstance() against the protocol '" +
                contract.getContractName().str() +
                "' is decided here only for values of this program's own "
                "classes; this one may be " + typeText(member) +
                ", whose methods CPython would inspect at run time"});
        return emitNone(expr);
      }
    }
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
  } else if (analysis.kind == IsInstanceAnalysis::Kind::HostTest) {
    // The constructor is the host's value and the test is the host's.
    Value constructor = emitExpr((*args)[1].get());
    mlir::Type boolType = types.boolType();
    auto callable = py::CallableType::get(
        &context, {input.type, constructor.type}, {}, {}, {}, {boolType});
    Value positional = emitPack({constructor});
    Value names = emitPack({});
    Value values = emitPack({});
    auto call =
        py::CallOp::create(builder, loc(expr), mlir::TypeRange{boolType},
                           callProtocolFor(callable), input.value,
                           positional.value, names.value, values.value);
    call->setAttr("ly.bound_method",
                  builder.getStringAttr("__ly_js_instanceof__"));
    return Value{call.getResults().front(), boolType};
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
                                    bool asAttribute, bool asSetter,
                                    bool asField) {
  auto contract = mlir::dyn_cast_if_present<py::ContractType>(receiver.type);
  if (!contract)
    return nullptr;
  llvm::StringRef receiverClass = contract.getContractName();
  const parser::Node &expr = anchor;
  // A property, a class attribute and a field are all READ; only a method is
  // called.
  bool readsWithoutCall = asProperty || asAttribute || asField;

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
  // The answer's type when it has no spelling a return annotation can carry
  // (`type[BaseException]`, `int | None`): pinned on the def instead.
  mlir::Type pinnedResult;
  parser::SourceRange range = expr.range;
  std::optional<MethodBinding> base;
  bool staticKind = false;
  bool classKind = false;
  const parser::Node *arguments = nullptr;
  // ⭐ A FIELD THE BASE DECLARES AND A SUBCLASS ANSWERS ANOTHER WAY. `class
  // Named: name: str` gives every instance a slot; `class Robot(Named): name
  // = "R2"` never fills it, and a Named-typed read loaded the empty slot as a
  // str (a segfault where CPython prints R2). The arms read through the
  // narrowed receiver, so each class answers the way its own instance does --
  // a class attribute, a property, or its own store -- and the fallback is the
  // field read itself. This is how a protocol's data member is satisfied by a
  // class attribute or a property.
  if (asField) {
    if (argumentCount != 0 || !keywordNames.empty())
      return nullptr;
    std::optional<mlir::Type> fieldType =
        lookupClassField(receiver.type, methodName);
    llvm::StringRef spelling;
    if (fieldType)
      if (auto fieldContract = mlir::dyn_cast_if_present<py::ContractType>(
              types.widenLiteral(*fieldType)))
        if (fieldContract.getArguments().empty()) {
          spelling = fieldContract.getContractName();
          if (spelling.consume_front("builtins.") && spelling.contains('.'))
            spelling = {};
        }
    if (!fieldType)
      return nullptr;
    if (spelling.empty())
      pinnedResult = types.widenLiteral(*fieldType);
    else
      returns = synth::name(spelling, range);
  }
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
    // ⭐ ANY TYPE, pinned where it has no spelling. Only a type the
    // dispatcher could WRITE as its return annotation used to qualify, so
    // the attributes a test class overrides -- `failureException` (a class
    // value) and `maxDiff` (`int | None`) -- kept the refusal when read
    // through the base.
    llvm::StringRef spelling;
    if (auto attrContract =
            mlir::dyn_cast_if_present<py::ContractType>(
                types.widenLiteral(attributeType)))
      if (attrContract.getArguments().empty()) {
        spelling = attrContract.getContractName();
        if (spelling.consume_front("builtins.") && spelling.contains('.'))
          spelling = {};
      }
    fallbackClass = definingClass.str();
    if (spelling.empty())
      pinnedResult = types.widenLiteral(attributeType);
    else
      returns = synth::name(spelling, range);
  }

  // ⭐ A DEFAULT THE BASE TAKES FROM OBJECT DISPATCHES TOO. `class Sub(Base)`
  // declaring `__repr__` over a Base that declares none left the base with no
  // body to fall back on, so `repr(b)` and `str(b)` on a Base-typed value
  // were refused ("'__repr__' is overridden by a subclass of 'Base'") while
  // `print(b)` answered right. The fallback is object's own slot, called with
  // the receiver -- `object.__repr__(b)`, as CPython names it.
  // ⛔ Not `__hash__` under a class that declares `__eq__`: its `__hash__` is
  // None, not object's.
  bool objectDefaultFallback =
      !asAttribute && !asField && !asProperty && !asSetter &&
      argumentCount == 0 && keywordNames.empty() &&
      (methodName == "__repr__" || methodName == "__str__" ||
       (methodName == "__hash__" &&
        inheritsObjectDefaultDunder(receiver.type, "__eq__"))) &&
      !lookupClassMethod(receiver.type, methodName) &&
      inheritsObjectDefaultDunder(receiver.type, methodName);
  if (objectDefaultFallback) {
    fallbackClass = "object";
    returns = synth::name(methodName == "__hash__" ? "int" : "str", range);
  }
  // ⭐ AN EXCEPTION'S TEXT DISPATCHES TOO, under a static class whose text is
  // the builtin's. `except (A, B) as e` and `e: Exception` hold a program
  // exception whose `__str__` or `__repr__` the builtin's text knows nothing
  // of, and `str(e)` printed the args where CPython calls the override. The
  // last arm is the builtin's own text, which the runtime class still picks
  // (KeyError's quotes its key).
  // ⛔ Not spelled `BaseException.__str__(__ly_recv)`: that is BaseException's
  // text whatever the class, a KeyError's included.
  bool builtinTextFallback =
      !objectDefaultFallback && !asAttribute && !asField && !asProperty &&
      !asSetter && argumentCount == 0 && keywordNames.empty() &&
      (methodName == "__str__" || methodName == "__repr__") &&
      !lookupClassMethod(receiver.type, methodName) &&
      isExceptionContractType(receiver.type);
  if (builtinTextFallback)
    returns = synth::name("str", range);
  // The body the STATIC type resolves to: its signature is the one the
  // dispatcher restates, and its declaring class is how the fallback names it.
  if (!asAttribute && !asField && !objectDefaultFallback &&
      !builtinTextFallback) {
  // ⭐ A `@property` WRITE IS A DISPATCH TOO, and the assignment path had no
  // gate at all: `x.v = 5` inlined the setter the STATIC class resolves to, so
  // a base-typed receiver ran the BASE's setter while `x.v` one line over
  // dispatched. [5, 50] where CPython prints [5, 60], silently. The BINDING is
  // the only thing that differs -- the arms, the fallback and the suppression
  // are the property READ's, because inside an arm the narrowed receiver
  // resolves its own setter exactly the way it resolves its own getter.
  base = asSetter
             ? lookupClassMethod(receiver.type,
                                 (llvm::Twine(methodName) + ".setter").str())
             : lookupClassMethod(receiver.type, methodName);
  if (!base || !base->method || base->definingClass.empty())
    return nullptr;
  // ⭐ A `@staticmethod` DISPATCHES THROUGH THE SAME ARMS, and it was the one
  // callable kind the synthesis refused outright:
  //
  //     class Base:
  //         @staticmethod
  //         def s() -> str: return "B"
  //     class Sub(Base): ... overrides s ...
  //     b: Base = Sub()
  //     b.s()   # 's' is overridden by a subclass of 'Base', so this call ...
  //
  // The base's own note says a dispatcher may not GUESS a signature, and a
  // staticmethod's is the one it does NOT have to: there is no receiver to
  // restate, so the arms call `Candidate.s(args)` through the CLASS -- exactly
  // what the attribute arms already do -- and the answer is the body the
  // runtime class's MRO picks.
  //
  // ⛔ NOT the classmethod, which looks like the same gap and is not. `cls`
  // must be the RUNTIME class, and the arms enumerate only classes that
  // REDECLARE the method: a subclass that inherits it lands in its parent's
  // arm, so `cls.__name__` would answer the parent -- a silent wrong value
  // where the refusal is not. Covering it means enumerating every subclass,
  // which is a different candidate set.
  staticKind = !asProperty && !asSetter && base->kind == "static";
  // ⭐ AND A `@classmethod` DISPATCHES THROUGH THE SAME ARMS, once the
  // candidate set is the right one. The note above said `cls` must be the
  // RUNTIME class and the arms enumerate only classes that REDECLARE the
  // method, so an inheriting subclass would land in its parent's arm and
  // `cls.__name__` would answer the parent -- a silent wrong value. Both
  // halves are fixed below: the candidates are every SUBCLASS, and the arm
  // calls through the candidate CLASS, which is what binds `cls` to it.
  //
  // ⛔ The cost is the whole subtree rather than the overriding slice of it,
  // and that is the price of the answer: `class Leaf(Sub)` that redeclares
  // nothing still needs its own arm, because `Sub.tag()` and `Leaf.tag()` run
  // the same body and return different strings.
  classKind = !asProperty && !asSetter && !asAttribute &&
              (base->kind == "class" || base->kind == "classmethod");
  llvm::StringRef wantedKind = asSetter     ? "property_setter"
                               : asProperty ? "property"
                                            : "instance";
  if ((base->kind != wantedKind && !staticKind && !classKind) || base->async ||
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
  // A builtin receiver is spelled as the source spells it: `Exception`.
  llvm::StringRef receiverSpelling = receiverClass;
  if (receiverSpelling.starts_with("builtins.") &&
      !receiverSpelling.drop_front(9).contains('.'))
    receiverSpelling = receiverSpelling.drop_front(9);
  params.push_back(
      synth::Param{"__ly_recv", synth::name(receiverSpelling, range)});
  if (!asAttribute && !asField && !objectDefaultFallback &&
      !builtinTextFallback) {
  // A staticmethod declares no receiver, so there is no first parameter to
  // skip -- `__ly_recv` above is the dispatcher's own, not the method's.
  // A classmethod's first parameter is `cls`, which the arm supplies by
  // calling through the class -- the same skip a receiver gets.
  bool selfSeen = staticKind;
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
          if (!staticKind && position == 1 && field != "kwonlyargs")
            continue; // the receiver
          // A `/` parameter cannot be named at a call at all, so a keyword
          // matching one is not this parameter -- CPython rejects the call.
          if (field == "posonlyargs" ||
              ast::nameSpelling(*argument) != keywordName)
            continue;
          annotation = sharedField(*argument, "annotation");
          keywordPosition = position - (staticKind ? 0 : 1);
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
       (asProperty ? "$get" : "") + (asAttribute ? "$attr" : "") +
       (asSetter ? "$set" : "") + (asField ? "$field" : ""))
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
      // ⛔ EVERY SUBCLASS FOR A CLASSMETHOD, not only the redeclaring ones:
      // `cls` is the runtime class, so a subclass that declares nothing still
      // has its own answer.
      if (asField ? !candidateRedeclares(declaredClassReadableNames,
                                         receiverClass, candidate, methodName)
                  : (!classKind &&
                     !candidateRedeclares(declarations, receiverClass,
                                          candidate, methodName)))
        continue;
      llvm::StringSet<> seen;
      collectDeclaredAncestors(candidate, seen);
      if (!seen.contains(receiverClass))
        continue;
      unsigned depth = seen.size();
      // ⛔ Not refused for a class whose contract is still to come when the
      // body waits for it (below): a library's method body is emitted before
      // the program's classes are, and the program's `maxDiff = None` on its
      // TestCase is the commonest override there is.
      if (types.lookupClass(candidate) == mlir::Type() &&
          classDeclarationsEmitted)
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
    if (asAttribute && classDeclarationsEmitted)
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
    // The setter's one parameter, restated BY NAME so the arm binds it the way
    // the body that runs declares it.
    auto write = [&](parser::NodePtr receiverNode) {
      return synth::assign(
          synth::attribute(std::move(receiverNode), methodName, range),
          synth::name(parameterNames.front(), range), range);
    };
    // ⭐ AN OVERRIDE THAT CANNOT TAKE THE CALL RAISES WHAT CPYTHON RAISES.
    // The arms forward the base's parameters, and a subclass may declare
    // others -- every `__init__` does, and a computed-name call reaches
    // `__init__` with whatever the call passed. An arm that cannot bind was
    // refused for the whole program ("missing required argument
    // 'description' for inlined class method") although CPython calls it
    // only if that class arrives, and then raises a TypeError; the arm now
    // raises it, word for word.
    // ⛔ Only the binding. An argument the override declares another TYPE for
    // stays refused: that is an incompatible override, which a type checker
    // reports too, and CPython would run the body with it anyway.
    auto callMismatch = [&](llvm::StringRef candidate) -> std::string {
      if (readsWithoutCall || asSetter)
        return "";
      std::optional<MethodBinding> running =
          resolveMroMethod(candidate, methodName);
      if (!running || !running->method || running->definingClass.empty())
        return "";
      return callBindingError(
          py::contracts::pythonQualnameForContract(running->definingClass) +
              "." + methodName.str(),
          *running->method, /*bound=*/running->kind != "static",
          argumentCount, keywordParameters);
    };
    std::vector<parser::NodePtr> body;
    for (const auto &candidate : candidates) {
      // ⭐ AN ATTRIBUTE ARM READS THROUGH THE CLASS, not through the narrowed
      // receiver. Inside the arm the runtime class IS this candidate (or a
      // subclass that declares nothing of its own, whose Python answer is this
      // candidate's binding), so `Candidate.attr` is the same value -- and it
      // is available on BOTH channels, where a read through the receiver needs
      // the cell only a main-module class has.
      // ⭐ A field arm reads a CLASS attribute through the class when no
      // class in the candidate's line stores the name on an instance: the
      // inherited field slot is then always empty, and the narrowed receiver
      // would resolve to it rather than to the attribute CPython finds.
      bool fieldArmThroughClass = false;
      if (asField) {
        auto readable = declaredClassReadableNames.find(candidate.second);
        bool isProperty = false;
        if (std::optional<MethodBinding> binding = resolveMroMethod(
                candidate.second, methodName))
          isProperty = binding->kind == "property";
        if (!isProperty && readable != declaredClassReadableNames.end()) {
          bool storedSomewhere = false;
          llvm::SmallVector<llvm::StringRef, 8> line{candidate.second};
          llvm::StringSet<> seen;
          while (!line.empty()) {
            llvm::StringRef current = line.pop_back_val();
            auto stores = declaredClassStoredFields.find(current);
            if (stores != declaredClassStoredFields.end() &&
                stores->second.contains(methodName))
              storedSomewhere = true;
            auto bases = declaredClassBases.find(current);
            if (bases != declaredClassBases.end())
              for (const std::string &base : bases->second)
                if (seen.insert(base).second)
                  line.push_back(base);
          }
          // ⛔ A store through any other name counts too: the instance may
          // then hold its own value (`q.longMessage = True`), and only the
          // narrowed receiver's read asks.
          fieldArmThroughClass =
              !storedSomewhere &&
              !programAttributeStoreNames().contains(methodName);
        }
      }
      parser::NodePtr subject =
          asAttribute || staticKind || classKind || fieldArmThroughClass
              ? synth::name(candidate.second, range)
              : synth::name("__ly_recv", range);
      // ⛔ A candidate that redeclares the GETTER and no setter has no setter
      // at all -- redeclaring the property replaces the whole descriptor -- so
      // its arm RAISES the AttributeError CPython raises, rather than falling
      // through to an inherited setter Python cannot reach through it.
      std::vector<parser::NodePtr> armBody;
      if (asSetter && !classPropertyHasSetter(candidate.second, methodName))
        armBody.push_back(synth::raiseCall(
            "AttributeError",
            "property '" + std::string(methodName) + "' of '" +
                py::contracts::displayClassNameForContract(candidate.second) +
                "' object has no setter",
            range));
      else if (std::string mismatch = callMismatch(candidate.second);
               !mismatch.empty())
        armBody.push_back(synth::raiseCall("TypeError", mismatch, range));
      else if (asSetter) {
        armBody.push_back(write(std::move(subject)));
        // ⛔ The arm has to RETURN. A read arm ends in one because it produces
        // a value; a write arm is a statement, so without this control fell
        // through to the fallback and the base's setter overwrote what the
        // subclass's had just stored -- the same wrong answer, now reached
        // through a correct arm.
        armBody.push_back(
            synth::returnStmt(synth::noneConstant(range), range));
      }
      else
        armBody.push_back(synth::returnStmt(read(std::move(subject)), range));
      body.push_back(synth::ifStmt(
          synth::call(synth::name("isinstance", range),
                      {synth::name("__ly_recv", range),
                       synth::name(candidate.second, range)},
                      range),
          std::move(armBody), {}, range));
    }
    if (asAttribute) {
      body.push_back(synth::returnStmt(
          synth::attribute(synth::name(fallbackClass, range), methodName,
                           range),
          range));
    } else if (asField) {
      body.push_back(
          synth::returnStmt(read(synth::name("__ly_recv", range)), range));
    } else if (asSetter) {
      body.push_back(write(synth::name("__ly_recv", range)));
    } else if (asProperty) {
      body.push_back(
          synth::returnStmt(read(synth::name("__ly_recv", range)), range));
    } else if (staticKind || classKind) {
      body.push_back(
          synth::returnStmt(read(synth::name(fallbackClass, range)), range));
    } else if (builtinTextFallback) {
      parser::NodePtr text = synth::call(
          synth::name(methodName == "__str__" ? "str" : "repr", range),
          {synth::name("__ly_recv", range)}, range);
      builtinTextFallbacks.insert(text.get());
      body.push_back(synth::returnStmt(std::move(text), range));
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

    // ⭐ THE BASE'S ANNOTATIONS ARE READ WHERE THE BASE IS WRITTEN. The
    // dispatcher is typed in the CALLER's scope, and an annotation names what
    // the base's module imports -- `def add(self, t: c.TC)` in r.py, called
    // from c.py, which has no `c`, read `c.TC` as a class nobody declares:
    // "'c.TC' object has no attribute 'name'" in the override's body.
    //
    // ⛔ A spelling whose type keeps a type parameter is left as written:
    // the parameter means what the caller's specialization binds it to.
    if (const EmitOptions::SourceModule *definingSource =
            sourceModuleForClass(base ? base->definingClass : std::string());
        definingSource && definingSource != activeSourceModule) {
      llvm::SmallVector<parser::NodePtr *, 8> spelled;
      for (synth::Param &param : llvm::drop_begin(params))
        if (param.annotation)
          spelled.push_back(&param.annotation);
      if (returns)
        spelled.push_back(&returns);
      llvm::SmallVector<mlir::Type, 8> resolved(spelled.size());
      emitInDefiningModuleScope(*definingSource, [&] {
        auto typeParameterScope = types.pushScope();
        bindClassTypeArguments(base->definingClass);
        for (auto [index, node] : llvm::enumerate(spelled))
          resolved[index] = types.annotationType(node->get());
      });
      for (auto [index, node] : llvm::enumerate(spelled)) {
        mlir::Type type = resolved[index];
        if (!type || unboundStaticParameterCount(type) != 0 ||
            mlir::isa<py::SelfType>(type))
          continue;
        std::string alias =
            "__lydispt$" + std::to_string(++syntheticFunctionCounter);
        types.bindAnnotationTypeAlias(alias, type);
        *node = synth::name(alias, range);
      }
    }
    parser::NodePtr def = synth::functionDef(symbol, params, {},
                                             std::move(body), returns, {}, range);
    synthesizedIteratorDefs.push_back(def);
    if (pinnedResult)
      types.pinResultType(def.get(), pinnedResult);
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
    bool suppressGate = readsWithoutCall || asSetter;
    auto emitBody = [this, def, symbol, sig, suppressGate] {
      emitDispatcherBody(*def, symbol, sig, suppressGate);
    };
    // ⭐ THE BODY WAITS FOR EVERY CLASS IT NAMES. Its arms call methods of
    // candidates gathered from every module's declarations, and a candidate
    // in a module emitted later (the program's own class implementing a
    // library's base or protocol) has no method table until its ClassDef is
    // emitted: `total(xs)` in base.py over `class Sq(Shape)` in main was
    // "'Sq.area' is used before 'Sq' is defined" -- an ordering the program
    // cannot change. The symbol and signature above are all a caller needs,
    // so only the body is deferred, to the end of the declarations.
    if (classDeclarationsEmitted)
      emitBody();
    else
      deferredDispatcherBodies.push_back(std::move(emitBody));
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

// ⭐ THE VALUE SPELLING OF THE SAME QUESTION. `x.s()` dispatches through the
// arms above, and `m = x.s; m()` did not: a static binding has no receiver, so
// `emitMethodObject` handed back the base's own function object and the call
// answered the base's body -- silently, and with the call spelling one line
// over answering correctly.
//
// The wrapper INTRODUCES a receiver parameter the static method never had, so
// the result is an ordinary instance-kind binding: `emitMethodObject` binds
// `__ly_recv` as the capture and the dispatcher tests it. The signature is
// computed from the synthesized def rather than adapted from the static one --
// there is no first parameter to erase from a signature that has none.
//
// ⛔ No defaults are carried, and that is deliberate rather than an omission:
// each arm's body fills its own, so a wrapper restating the BASE's default
// would answer the base's value for a subclass that changed it. Without them
// a short call is an arity refusal, which is the honest half of the pair.
std::optional<MethodBinding> ModuleEmitter::virtualStaticMethodObjectBinding(
    const parser::Node &anchor, Value receiver, const MethodBinding &binding) {
  if (binding.kind != "static" || !binding.method)
    return std::nullopt;
  auto contract = mlir::dyn_cast_if_present<py::ContractType>(receiver.type);
  if (!contract)
    return std::nullopt;
  std::optional<std::string_view> methodName =
      ast::string(*binding.method, "name");
  if (!methodName)
    return std::nullopt;
  const parser::Node *arguments = ast::node(*binding.method, "args");
  if (!arguments)
    return std::nullopt;
  parser::NodePtr returns = sharedField(*binding.method, "returns");
  if (!returns)
    return std::nullopt;

  parser::SourceRange range = anchor.range;
  llvm::SmallVector<synth::Param, 4> params;
  llvm::SmallVector<std::string, 4> forwardedNames;
  params.push_back(
      synth::Param{"__ly_recv", synth::name(contract.getContractName(), range)});
  for (llvm::StringRef field : {"posonlyargs", "args"})
    if (const auto *list = ast::nodeList(*arguments, field))
      for (const parser::NodePtr &argument : *list) {
        if (!argument)
          return std::nullopt;
        parser::NodePtr annotation = sharedField(*argument, "annotation");
        if (!annotation)
          return std::nullopt;
        forwardedNames.push_back(
            std::string(llvm::StringRef(ast::nameSpelling(*argument))));
        params.push_back(
            synth::Param{forwardedNames.back(), std::move(annotation)});
      }

  const VirtualDispatchHelper *helper = virtualDispatcherFor(
      anchor, receiver, *methodName,
      static_cast<unsigned>(forwardedNames.size()));
  if (!helper)
    return std::nullopt;

  std::vector<parser::NodePtr> callArguments;
  callArguments.push_back(synth::name("__ly_recv", range));
  for (const std::string &name : forwardedNames)
    callArguments.push_back(synth::name(name, range));
  std::vector<parser::NodePtr> body;
  body.push_back(synth::returnStmt(
      synth::call(synth::name(helper->symbol, range), std::move(callArguments),
                  range),
      range));
  std::string symbol =
      "__lyvsbound$" + std::to_string(++syntheticFunctionCounter);
  parser::NodePtr def =
      synth::functionDef(symbol, params, {}, std::move(body),
                         std::move(returns), {}, range);
  synthesizedIteratorDefs.push_back(def);

  MethodBinding bound;
  bound.method = def.get();
  bound.kind = "instance";
  bound.symbolName = symbol;
  bound.bodySignature = types.functionSignature(*def);
  bound.signature = bound.bodySignature;
  // ⛔ Left empty on purpose: `emitMethodObject` pushes a super() context for a
  // binding that names its defining class, and this body is a forwarder with
  // no class behind it.
  bound.definingClass.clear();
  return bound;
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
ModuleEmitter::tryEmitVirtualFieldRead(const parser::Node &anchor,
                                       Value receiver,
                                       llvm::StringRef fieldName,
                                       mlir::Type fieldType) {
  (void)fieldType;
  if (!mlir::isa_and_nonnull<py::ContractType>(receiver.type))
    return std::nullopt;
  const VirtualDispatchHelper *helper = virtualDispatcherFor(
      anchor, receiver, fieldName, /*argumentCount=*/0, /*asProperty=*/false,
      /*keywordNames=*/{}, /*asAttribute=*/false, /*asSetter=*/false,
      /*asField=*/true);
  if (!helper)
    return std::nullopt;
  Value callee = emitBindingRef(anchor, helper->symbol, helper->callable);
  return emitCallableDispatch(
      anchor, callee,
      emitCallOperands(anchor, {receiver}, /*includeAstArguments=*/false));
}

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

bool ModuleEmitter::classPropertyHasSetter(
    llvm::StringRef className, llvm::StringRef propertyName) const {
  std::optional<MethodBinding> getter =
      resolveMroMethod(className, propertyName);
  if (!getter || getter->kind != "property")
    return false;
  std::optional<MethodBinding> setter = resolveMroMethod(
      className, (llvm::Twine(propertyName) + ".setter").str());
  // The two have to come from the SAME class body: a subclass that redeclares
  // the getter builds a new property object, and an inherited setter belongs
  // to the one it replaced.
  return setter && setter->kind == "property_setter" &&
         setter->definingClass == getter->definingClass;
}

bool ModuleEmitter::tryEmitVirtualPropertyWrite(const parser::Node &anchor,
                                                Value receiver,
                                                llvm::StringRef propertyName,
                                                Value value) {
  const VirtualDispatchHelper *helper = virtualDispatcherFor(
      anchor, receiver, propertyName, /*argumentCount=*/1,
      /*asProperty=*/false, /*keywordNames=*/{}, /*asAttribute=*/false,
      /*asSetter=*/true);
  if (!helper)
    return false;
  Value callee = emitBindingRef(anchor, helper->symbol, helper->callable);
  emitCallableDispatch(
      anchor, callee,
      emitCallOperands(anchor, {receiver, value},
                       /*includeAstArguments=*/false));
  return true;
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

void ModuleEmitter::emitDispatcherBody(const parser::Node &def,
                                       const std::string &symbol,
                                       const FunctionSignature &sig,
                                       bool suppressGate) {
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
  if (suppressGate)
    ++virtualPropertyBodyDepth;
  llvm::scope_exit restoreSuppression([&, suppressGate] {
    if (suppressGate)
      --virtualPropertyBodyDepth;
  });
  llvm::scope_exit restoreContexts([&] {
    loopControlContexts = std::move(savedLoops);
    inlineReturnContexts = std::move(savedInlineReturns);
    superContexts = std::move(savedSupers);
    methodsBeingInlined = std::move(savedInlining);
    inlineFrames = std::move(savedInlineFrames);
  });
  emitCallableFunction(def, symbol, sig, {}, /*isLambda=*/false);
}

// ⭐ THE RIGHT OPERAND GOES FIRST WHEN ITS CLASS STRICTLY DERIVES THE LEFT'S,
// as do_richcompare decides: `b == a` with `a` a Sub and `b` a Base asks
// `a.__eq__(b)` before `b.__eq__(a)`. The one-receiver dispatcher answers
// "which body runs for this receiver" and cannot ask a question about two
// runtime classes, so `b == a` ran Base's body -- silently.
//
// The body pins the LEFT's runtime class by testing every class of its line
// most-derived first (the first `isinstance` that holds names the exact class,
// because every subclass was tested before it), then asks the same of the
// right among that class's strict subclasses. An arm whose right class would
// answer the reflected comparison writes it the other way round; every other
// arm writes the comparison as the program did. Both are emitted with this
// synthesis switched off, so each is the one-receiver dispatch it always was.
//
// ⛔ Only where some pair of the program's classes can reach the reflected
// arm: everywhere else the comparison is emitted exactly as before.
// ⛔ Only a comparison whose every candidate method is annotated `-> bool`:
// the dispatcher has one return annotation to write.
// ⛔ Not the arithmetic operators: CPython reflects those first only when the
// subclass OVERRIDES the reflected method (`method_is_overloaded`), and a
// subclass that does so is already what the probe measured correct.
const ModuleEmitter::VirtualDispatchHelper *
ModuleEmitter::subclassFirstCompareFor(const parser::Node &anchor, Value lhs,
                                       Value rhs, llvm::StringRef opKind) {
  if (subclassFirstSuppressed)
    return nullptr;
  static constexpr std::array<std::array<llvm::StringLiteral, 3>, 6> kOps{{
      {"Eq", "__eq__", "Eq"},
      {"NotEq", "__ne__", "NotEq"},
      {"Lt", "__lt__", "Gt"},
      {"LtE", "__le__", "GtE"},
      {"Gt", "__gt__", "Lt"},
      {"GtE", "__ge__", "LtE"}}};
  llvm::StringRef mirrorKind;
  llvm::StringRef reflected;
  for (const auto &entry : kOps)
    if (entry[0] == opKind) {
      mirrorKind = entry[2];
      for (const auto &mirror : kOps)
        if (mirror[0] == mirrorKind)
          reflected = mirror[1];
    }
  if (reflected.empty())
    return nullptr;
  auto leftContract =
      mlir::dyn_cast_if_present<py::ContractType>(types.widenLiteral(lhs.type));
  auto rightContract =
      mlir::dyn_cast_if_present<py::ContractType>(types.widenLiteral(rhs.type));
  if (!leftContract || !rightContract ||
      !isSourceClassContract(leftContract) ||
      !isSourceClassContract(rightContract) ||
      !leftContract.getArguments().empty() ||
      !rightContract.getArguments().empty())
    return nullptr;
  llvm::StringRef leftClass = leftContract.getContractName();
  llvm::StringRef rightClass = rightContract.getContractName();

  auto ancestorsOf = [&](llvm::StringRef cls) {
    llvm::StringSet<> seen;
    llvm::SmallVector<llvm::StringRef, 8> worklist{cls};
    while (!worklist.empty()) {
      auto bases = declaredClassBases.find(worklist.pop_back_val());
      if (bases == declaredClassBases.end())
        continue;
      for (const std::string &base : bases->second)
        if (seen.insert(base).second)
          worklist.push_back(base);
    }
    return seen;
  };
  // Every declared class with its strict ancestors, most-derived first.
  struct Line {
    std::string name;
    llvm::StringSet<> ancestors;
  };
  std::vector<Line> lines;
  for (const auto &entry : declaredClassBases)
    lines.push_back(Line{entry.getKey().str(), ancestorsOf(entry.getKey())});
  llvm::sort(lines, [](const Line &a, const Line &b) {
    if (a.ancestors.size() != b.ancestors.size())
      return a.ancestors.size() > b.ancestors.size();
    return a.name < b.name;
  });
  auto derives = [](const Line &line, llvm::StringRef cls) {
    return line.name == cls || line.ancestors.contains(cls);
  };
  auto returnsBool = [&](const MethodBinding &binding) {
    const parser::Node *returns =
        binding.method ? ast::node(*binding.method, "returns") : nullptr;
    return returns && returns->kind == "Name" &&
           ast::nameSpelling(*returns) == "bool";
  };
  // Whether an instance of exactly `cls` answers the reflected comparison
  // from a body of the program's: `!=` falls back on `__eq__`, as
  // object.__ne__ does.
  bool unannotated = false;
  auto answers = [&](llvm::StringRef cls) {
    for (llvm::StringRef method :
         {reflected, reflected == "__ne__" ? llvm::StringRef("__eq__")
                                          : llvm::StringRef()}) {
      if (method.empty())
        continue;
      std::optional<MethodBinding> binding = resolveMroMethod(cls, method);
      if (!binding || !binding->method || binding->kind != "instance")
        continue;
      if (!returnsBool(*binding))
        unannotated = true;
      return true;
    }
    return false;
  };

  const parser::SourceRange range = anchor.range;
  auto left = [&] { return synth::name("__ly_l", range); };
  auto right = [&] { return synth::name("__ly_r", range); };
  auto asWritten = [&] {
    return synth::returnStmt(synth::compare(left(), opKind, right(), range),
                             range);
  };
  auto isinstanceOf = [&](parser::NodePtr subject, llvm::StringRef cls) {
    return synth::call(synth::name("isinstance", range),
                       {std::move(subject), synth::name(cls, range)}, range);
  };
  // First the arms, then which left classes need one: a class whose line
  // holds a reflecting pair, and every class below one -- so that an
  // ancestor's arm does not claim its instances.
  struct LeftArm {
    const Line *line;
    std::vector<parser::NodePtr> arms;
    bool reflects = false;
  };
  std::vector<LeftArm> leftArms;
  llvm::StringSet<> reflecting;
  for (const Line &leftLine : lines) {
    if (!derives(leftLine, leftClass))
      continue;
    LeftArm entry{&leftLine, {}, false};
    for (const Line &rightLine : lines) {
      if (!rightLine.ancestors.contains(leftLine.name))
        continue;
      // Only a class an operand of the right's static type can be.
      if (!derives(rightLine, rightClass) &&
          !llvm::any_of(lines, [&](const Line &line) {
            return line.name == rightClass &&
                   line.ancestors.contains(rightLine.name);
          }))
        continue;
      bool first = answers(rightLine.name);
      entry.reflects = entry.reflects || first;
      std::vector<parser::NodePtr> inner;
      inner.push_back(
          first ? synth::returnStmt(
                      synth::compare(right(), mirrorKind, left(), range), range)
                : asWritten());
      entry.arms.push_back(synth::ifStmt(isinstanceOf(right(), rightLine.name),
                                         std::move(inner), {}, range));
    }
    if (entry.reflects)
      reflecting.insert(leftLine.name);
    leftArms.push_back(std::move(entry));
  }
  if (reflecting.empty() || unannotated)
    return nullptr;
  std::vector<parser::NodePtr> body;
  for (LeftArm &entry : leftArms) {
    bool below = llvm::any_of(entry.line->ancestors, [&](const auto &a) {
      return reflecting.contains(a.getKey());
    });
    if (!entry.reflects && !below)
      continue;
    std::vector<parser::NodePtr> arm;
    if (entry.reflects)
      arm = std::move(entry.arms);
    arm.push_back(asWritten());
    body.push_back(synth::ifStmt(isinstanceOf(left(), entry.line->name),
                                 std::move(arm), {}, range));
  }
  body.push_back(asWritten());

  std::string key =
      ("cmp:" + leftClass + "," + rightClass + "," + opKind).str();
  auto memo = virtualDispatchHelpers.find(key);
  if (memo != virtualDispatchHelpers.end())
    return memo->second.callable ? &memo->second : nullptr;

  std::string symbol =
      "__lysubfirst$" + std::to_string(++syntheticFunctionCounter);
  llvm::SmallVector<synth::Param, 2> params{
      synth::Param{"__ly_l", synth::name(leftClass, range)},
      synth::Param{"__ly_r", synth::name(rightClass, range)}};
  parser::NodePtr def =
      synth::functionDef(symbol, params, {}, std::move(body),
                         synth::name("bool", range), {}, range);
  synthesizedIteratorDefs.push_back(def);
  FunctionSignature sig = types.functionSignature(*def);
  virtualDispatchHelpers[key] = VirtualDispatchHelper{symbol, sig.publicCallable};
  types.bindRootSymbol(symbol, sig.publicCallable);
  auto emitBody = [this, def, symbol, sig] {
    ++subclassFirstSuppressed;
    llvm::scope_exit restore([&] { --subclassFirstSuppressed; });
    emitDispatcherBody(*def, symbol, sig, /*suppressGate=*/false);
  };
  if (classDeclarationsEmitted)
    emitBody();
  else
    deferredDispatcherBodies.push_back(std::move(emitBody));
  memo = virtualDispatchHelpers.find(key);
  return memo->second.callable ? &memo->second : nullptr;
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
    // ⭐ int(n) IS AN EXACT INT, and an int-declared slot or field may hold a
    // bool: `int(xs[0])` for `xs: list[int] = [True]` is 1. `n + 0` is the
    // int of n's value, n itself in value for an exact int.
    // ⛔ Not the identity it was, which answered True there.
    parser::NodePtr exact = synth::binOp(
        intArgs->front(), "Add", synth::intConstant(0, expr.range), expr.range);
    return coerceValue(emitExpr(exact.get()), types.intType(), expr);
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

// `isinstance(x, t)` / `issubclass(s, t)` over class values: the class of x
// (or s itself) has t in its MRO.
bool ModuleEmitter::targetHasDynamicClassValue(const parser::Node *node) {
  if (!node)
    return false;
  if (node->kind == "Tuple") {
    if (const auto *elts = ast::nodeList(*node, "elts"))
      for (const parser::NodePtr &elt : *elts)
        if (targetHasDynamicClassValue(elt.get()))
          return true;
    return false;
  }
  mlir::Type type = types.widenLiteral(types.inferExpr(node));
  return mlir::isa<py::TypeType>(type) && !classValueIsExact(node, type);
}

std::optional<Value> ModuleEmitter::emitClassValueSubclassTest(
    const parser::Node &expr, const parser::Node *subjectNode,
    const parser::Node *targetNode, bool subjectIsInstance) {
  Value subject = emitExpr(subjectNode);
  // A tuple of classes is any of them, each evaluated in order.
  llvm::SmallVector<Value, 4> targets;
  if (targetNode->kind == "Tuple") {
    if (const auto *elts = ast::nodeList(*targetNode, "elts"))
      for (const parser::NodePtr &elt : *elts)
        targets.push_back(emitExpr(elt.get()));
  } else {
    targets.push_back(emitExpr(targetNode));
  }
  auto refuse = [&](mlir::Type offending) -> std::optional<Value> {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, expr.range.start,
        std::string(subjectIsInstance ? "isinstance()" : "issubclass()") +
            " needs class values on both sides here, and " +
            typeText(offending) + " is not one"});
    return emitNone(expr);
  };
  for (const Value &target : targets)
    if (!mlir::isa<py::TypeType>(target.type))
      return refuse(target.type);
  if (targets.empty())
    return refuse(types.inferExpr(targetNode));
  mlir::Value subjectClass = subject.value;
  if (subjectIsInstance) {
    mlir::Type instance = types.widenLiteral(subject.type);
    mlir::Type bound = mlir::isa<py::ContractType>(instance) ? instance
                                                             : types.object();
    subjectClass =
        emitClassOfValue(expr, subject, types.typeObject(bound)).value;
  } else if (!mlir::isa<py::TypeType>(subject.type)) {
    return refuse(subject.type);
  }
  mlir::Value test;
  for (const Value &target : targets) {
    mlir::Value one =
        py::TypeSubclassOp::create(builder, loc(expr), builder.getI1Type(),
                                   subjectClass, target.value)
            .getResult();
    test = test ? mlir::arith::OrIOp::create(builder, loc(expr), test, one)
                      .getResult()
                : one;
  }
  auto pyBool =
      py::CastFromPrimOp::create(builder, loc(expr), types.boolType(), test);
  return Value{pyBool.getResult(), types.boolType()};
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
  // A class value on either side that the static type does not decide is
  // answered at run time, from the MRO of the class it names.
  if (targetHasDynamicClassValue((*args)[0].get()) ||
      targetHasDynamicClassValue((*args)[1].get()))
    return emitClassValueSubclassTest(expr, (*args)[0].get(), (*args)[1].get(),
                                      /*subjectIsInstance=*/false);
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
  // ⭐ A CLASS VALUE THAT IS NOT SPELLED AS A CLASS NAME -- a field
  // (`self.expected`), a call's result -- is a value like any other, even
  // when its type alone says which class it is: the run-time test reads it.
  // Only a name could be folded here, so `issubclass(t, self.expected)` in
  // `Ctx[KeyError]` was refused while `Ctx[ValueError]`, whose class has
  // subclasses, ran.
  auto isClassValue = [&](const parser::Node *node) {
    return node && (node->kind == "Tuple" ||
                    mlir::isa_and_nonnull<py::TypeType>(
                        types.widenLiteral(types.inferExpr(node))));
  };
  if ((!subClass || !superClass) && isClassValue((*args)[0].get()) &&
      isClassValue((*args)[1].get()))
    return emitClassValueSubclassTest(expr, (*args)[0].get(), (*args)[1].get(),
                                      /*subjectIsInstance=*/false);
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
    // A class renders through its type object, the same arm.
    if (mlir::isa<py::UnionType, py::TypeType>(argumentType))
      if (std::optional<Value> rendered =
              emitStringifyValue(expr, emitExpr(strArgs->front().get())))
        return *rendered;
    if (lookupClassMethod(argumentType, "__str__")) {
      Value argument = emitExpr(strArgs->front().get());
      if (std::optional<Value> dispatched =
              tryEmitClassDunder(expr, argument, "__str__"))
        return *dispatched;
    }
    // ⛔ Nor is it repr when a SUBCLASS declares `__str__`: the argument may
    // be that subclass. `emitInheritedObjectStr` asks the runtime class.
    if (inheritsObjectDefaultDunder(argumentType, "__str__") &&
        subclassOverridesMethod(
            mlir::cast<py::ContractType>(argumentType).getContractName(),
            "__str__"))
      if (std::optional<Value> rendered = emitInheritedObjectStr(
              expr, emitExpr(strArgs->front().get())))
        return *rendered;
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
      // ⛔ Not the builtin's text when a program exception under it declares
      // `__str__`: the argument may be that one. Every other spelling of
      // the question -- `print(e)`, `repr(e)`, `f"{e}"` -- asked the gate,
      // and this one printed the args.
      if (isExceptionContractType(argumentType) &&
          !builtinTextFallbacks.contains(&expr) &&
          dispatchIsUnresolvable(argument, "__str__", strArgs->front().get(),
                                 /*throughSuper=*/false)) {
        if (std::optional<Value> dispatched =
                tryEmitVirtualDispatchWithValues(expr, argument, "__str__", {}))
          return *dispatched;
        if (refuseUnresolvableDispatch(expr, argument, "__str__",
                                       strArgs->front().get()))
          return emitNone(expr);
      }
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
  // ⭐ `list(filter(f, xs))` AND `list(map(f, xs))` WITH A LOCAL `f` ARE THE
  // COMPREHENSION, in this function. The lazy iterator is a generator of its
  // own at module level that spells `f` by name, and a nested def there names
  // nothing: unittest's `list(filter(shouldIncludeMethod, dir(cls)))` reached
  // the lowering as "unresolved runtime binding 'shouldIncludeMethod'".
  // `list(...)` consumes the iterator where it stands, so the comprehension
  // is the same evaluation: the iterable once, then `f` per element.
  if (listArgs && listArgs->size() == 1 && listArgs->front() &&
      listArgs->front()->kind == "Call" &&
      (!listKeywords || listKeywords->empty())) {
    const parser::Node &inner = *listArgs->front();
    const parser::Node *innerCallee = ast::node(inner, "func");
    const auto *innerArgs = ast::nodeList(inner, "args");
    const auto *innerKeywords = ast::nodeList(inner, "keywords");
    bool isFilter = callsUnshadowedBuiltin(innerCallee, "filter");
    bool isMap = callsUnshadowedBuiltin(innerCallee, "map");
    if ((isFilter || isMap) && innerArgs && innerArgs->size() == 2 &&
        (!innerKeywords || innerKeywords->empty())) {
      const parser::NodePtr &function = (*innerArgs)[0];
      const parser::NodePtr &source = (*innerArgs)[1];
      if (function && source && source->kind != "Starred" &&
          function->kind == "Name" &&
          values.count(ast::nameSpelling(*function))) {
        parser::SourceRange range = expr.range;
        std::string element =
            "__lyfilter" + std::to_string(++listCompCounter) + "_v";
        parser::NodePtr applied = synth::call(
            function, {synth::name(element, range)}, range);
        parser::NodePtr generator = parser::makeNode("comprehension", range);
        parser::addField(*generator, "target", synth::name(element, range));
        parser::addField(*generator, "iter", source);
        std::vector<parser::NodePtr> ifs;
        if (isFilter)
          ifs.push_back(applied);
        parser::addField(*generator, "ifs", std::move(ifs));
        parser::addField(*generator, "is_async", static_cast<std::int64_t>(0));
        parser::NodePtr comprehension = parser::makeNode("ListComp", range);
        parser::addField(*comprehension, "elt",
                         isFilter ? synth::name(element, range) : applied);
        parser::addField(*comprehension, "generators",
                         std::vector<parser::NodePtr>{std::move(generator)});
        synthesizedIteratorDefs.push_back(comprehension);
        return emitComprehension(*comprehension, /*isDict=*/false);
      }
    }
  }
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
  // named nowhere in it. A keyword print does not take is refused by name
  // below, which beats that contract mismatch.
  const parser::Node *separatorNode = nullptr;
  const parser::Node *endNode = nullptr;
  // `file=` and `flush=` choose the sink: the line goes to that object's
  // `write` (and then its `flush`), which is all CPython's print does with
  // them.
  const parser::Node *fileNode = nullptr;
  parser::NodePtr flushNode;
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
      if (name && *name == "file" && !fileNode) {
        fileNode = ast::node(*keyword, "value");
        if (fileNode)
          continue;
      }
      if (name && *name == "flush" && !flushNode) {
        if (const parser::Field *value = parser::findField(*keyword, "value"))
          if (const auto *held = std::get_if<parser::NodePtr>(&value->value))
            flushNode = *held;
        if (flushNode)
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
  // half a line. CPython makes a call per piece, which only a sink of the
  // program's own can observe -- and a `file=` takes the piece-by-piece path
  // below; `sys.stdout` here is always the runtime's stream.
  auto writeWithEnd = [&](Value text) -> std::optional<Value> {
    mlir::Type strContract = types.contract("builtins.str");
    Value end;
    if (endNode) {
      end = coerceValue(emitExpr(endNode), strContract, expr);
    } else {
      mlir::Type newlineType = types.literal("\"\\n\"");
      end = coerceValue(Value{py::StrConstantOp::create(
                                  builder, loc(expr), newlineType,
                                  builder.getStringAttr("\n"))
                                  .getResult(),
                              newlineType},
                        strContract, expr);
    }
    text = emitBinarySpecial<py::AddOp>(expr, "__add__", text, end,
                                        strContract);
    mlir::Type wrapper = types.contract("_io.TextIOWrapper");
    Value stream = emitBindingRef(expr, "sys.stdout", wrapper, {});
    std::string streamName =
        "__lyprintout" + std::to_string(++syntheticFunctionCounter);
    std::string textName =
        "__lyprinttext" + std::to_string(++syntheticFunctionCounter);
    values[streamName] = stream;
    types.bindSymbol(streamName, stream.type);
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
  // ⭐ A SINK OF ITS OWN IS WRITTEN PIECE BY PIECE, as `builtin_print_impl`
  // writes it: every argument's str(), each separator and the end are one
  // `file.write` each, then `file.flush()` when flush is true. A sink that
  // records its writes sees the pieces, so they cannot be joined first the
  // way the stdout line above is.
  // The arguments are evaluated first and the keywords after them in their
  // written order, which is when CPython evaluates a call's arguments; the
  // str() of each argument is taken between the writes, as there.
  // ⭐ AND SO IS A `sys.stdout` THE PROGRAM REPLACES: print writes to what
  // the stream holds when it runs, piece by piece through its `write`, which
  // is CPython's print -- the joined line above is a shortcut only the
  // runtime's own stream can take (EmitterGlobals.cpp, replaced streams).
  bool replacedStdout = llvm::is_contained(replacedStreams, "sys.stdout");
  bool chosenSink = fileNode || flushNode || replacedStdout;
  if (chosenSink) {
    if (printArgs)
      for (const parser::NodePtr &argument : *printArgs)
        if (!argument || argument->kind == "Starred") {
          diagnostics.push_back(parser::Diagnostic{
              parser::Severity::Error, expr.range.start,
              fileNode || flushNode
                  ? "print() to a file takes its arguments spelled out, not "
                    "unpacked"
                  : "print() takes its arguments spelled out, not unpacked, "
                    "in a program that replaces sys.stdout: each one is "
                    "written to the stream sys.stdout holds"});
          return emitNone(expr);
        }
    mlir::Type strType = types.contract("builtins.str");
    llvm::SmallVector<Value, 4> evaluated;
    if (printArgs)
      for (const parser::NodePtr &argument : *printArgs)
        evaluated.push_back(emitExpr(argument.get()));
    std::optional<Value> separator, end, stream, flush;
    for (const parser::NodePtr &keyword : *printKeywords) {
      llvm::StringRef name = ast::string(*keyword, "arg").value_or("");
      const parser::Node *value = ast::node(*keyword, "value");
      if (name == "sep")
        separator = coerceValue(emitExpr(value), strType, expr);
      else if (name == "end")
        end = coerceValue(emitExpr(value), strType, expr);
      else if (name == "file") {
        // `file=None` is sys.stdout, as in CPython.
        Value file = emitExpr(value);
        if (!isNoneTypeLike(types.widenLiteral(file.type)))
          stream = file;
      }
      else if (name == "flush")
        flush = emitExpr(value);
    }
    auto constant = [&](llvm::StringRef text) {
      mlir::Type literal = types.literal(
          (llvm::Twine("\"") + (text == "\n" ? "\\n" : text) + "\"").str());
      return coerceValue(
          Value{py::StrConstantOp::create(builder, loc(expr), literal,
                                          builder.getStringAttr(text))
                    .getResult(),
                literal},
          strType, expr);
    };
    if (!separator)
      separator = constant(" ");
    if (!end)
      end = constant("\n");
    if (!stream)
      stream = emitBindingRef(expr, "sys.stdout",
                              types.contract("_io.TextIOWrapper"), {});
    std::string streamName =
        "__lyprintout" + std::to_string(++syntheticFunctionCounter);
    values[streamName] = *stream;
    types.bindSymbol(streamName, stream->type);
    auto call = [&](llvm::StringRef method, std::optional<Value> argument) {
      std::vector<parser::NodePtr> arguments;
      std::string argumentName;
      if (argument) {
        argumentName =
            "__lyprinttext" + std::to_string(++syntheticFunctionCounter);
        values[argumentName] = *argument;
        types.bindSymbol(argumentName, argument->type);
        arguments.push_back(synth::name(argumentName, expr.range));
      }
      parser::NodePtr node =
          synth::methodCall(synth::name(streamName, expr.range), method,
                            std::move(arguments), expr.range);
      (void)emitExpr(node.get());
      synthesizedIteratorDefs.push_back(std::move(node));
      if (!argumentName.empty())
        values.erase(argumentName);
    };
    for (auto [index, value] : llvm::enumerate(evaluated)) {
      if (index != 0)
        call("write", separator);
      std::optional<Value> piece = emitStringifyValue(expr, value);
      if (!piece) {
        std::string described;
        llvm::raw_string_ostream os(described);
        os << value.type;
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, expr.range.start,
            "print() cannot render argument " + std::to_string(index + 1) +
                " of type " + described +
                ": it has no __str__ or __repr__ this dispatch can resolve"});
        values.erase(streamName);
        return emitNone(expr);
      }
      call("write", piece);
    }
    call("write", end);
    if (flush) {
      mlir::Value truth = emitBoolValue(*flush, expr);
      mlir::Block *entry = builder.getInsertionBlock();
      mlir::Region *region = entry->getParent();
      mlir::Block *continuation =
          entry->splitBlock(builder.getInsertionPoint());
      mlir::Block *flushBlock =
          builder.createBlock(region, continuation->getIterator());
      builder.setInsertionPointToEnd(entry);
      mlir::cf::CondBranchOp::create(builder, loc(expr), truth, flushBlock,
                                     mlir::ValueRange{}, continuation,
                                     mlir::ValueRange{});
      builder.setInsertionPointToStart(flushBlock);
      call("flush", std::nullopt);
      mlir::cf::BranchOp::create(builder, loc(expr), continuation,
                                 mlir::ValueRange{});
      builder.setInsertionPointToStart(continuation);
    }
    values.erase(streamName);
    return emitNone(expr);
  }
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
    // ⭐ A LIST OF INTS, FLOATS, STRS OR BOOLS WITH NO KEY is CPython's
    // min_max loop in the runtime (`list.__best_index__`), and the winner is
    // read once. ⛔ Not the folds below: the accumulator carries the best
    // element across the loop as an object, boxing every immediate int it
    // compares, and the index fold boxes the index at every join -- `max(xs)`
    // over 10,000 ints ran at 0.75x CPython either way. Only these element
    // types, whose `a > b` is `b < a`, which is the one comparison the runtime
    // loop asks.
    if (!reducerKeyNode && !reducerDefaultValue) {
      auto argContract = mlir::dyn_cast_if_present<py::ContractType>(
          types.widenLiteral(types.inferExpr(reducerArgs->front().get())));
      auto elementContract = mlir::dyn_cast_if_present<py::ContractType>(
          types.widenLiteral(elementType));
      llvm::StringRef elementName =
          elementContract ? elementContract.getContractName()
                          : llvm::StringRef();
      if (argContract && argContract.getContractName() == "builtins.list" &&
          (elementName == "builtins.int" || elementName == "builtins.float" ||
           elementName == "builtins.str" || elementName == "builtins.bool")) {
        std::string counter = std::to_string(++listCompCounter);
        std::string src = "__" + reducer.str() + "src" + counter;
        std::string bestIndex = "__" + reducer.str() + "bi" + counter;
        Value list = emitExpr(reducerArgs->front().get());
        values[src] = list;
        types.bindSymbol(src, list.type);
        parser::NodePtr greatestNode =
            synth::constantBool(reducer == "max", expr.range);
        Value greatest = emitExpr(greatestNode.get());
        CallInferenceResult inference = types.inferMethodCallWithEvidence(
            list.type, "__best_index__", {greatest.type});
        if (inference && requireStaticEvidence(expr, inference)) {
          Value posPack = emitPack({greatest});
          Value namePack = emitPack({});
          Value valuePack = emitPack({});
          auto call = py::CallOp::create(
              builder, loc(expr), mlir::TypeRange{inference.resultType},
              callProtocolFor(inference), list.value, posPack.value,
              namePack.value, valuePack.value);
          call->setAttr("ly.bound_method",
                        builder.getStringAttr("__best_index__"));
          values[bestIndex] =
              Value{call.getResults().front(), inference.resultType};
          types.bindSymbol(bestIndex, inference.resultType);
          auto nameOf = [&](const std::string &id) {
            return synth::name(id, expr.range);
          };
          parser::NodePtr emptyCheck = synth::ifStmt(
              synth::compare(nameOf(bestIndex), "Lt",
                             synth::intConstant(0, expr.range), expr.range),
              {synth::raiseCall("ValueError",
                                reducer.str() + "() iterable argument is empty",
                                expr.range)},
              {}, expr.range);
          synthesizedIteratorDefs.push_back(emptyCheck);
          emitStatement(*emptyCheck);
          parser::NodePtr winner = synth::subscript(
              nameOf(src), nameOf(bestIndex), expr.range);
          synthesizedIteratorDefs.push_back(winner);
          Value result = emitExpr(winner.get());
          values.erase(src);
          values.erase(bestIndex);
          return result;
        }
        values.erase(src);
        return emitNone(expr);
      }
    }
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

// ⭐ A CLASS'S `__len__` IS A LENGTH ONLY AFTER slot_sq_length's CHECKS:
// CPython raises ValueError for a negative answer and OverflowError for one
// past the word, in `len` and in a truth test alike. Handed back as it was,
// `len(b)` printed -1 and `bool(b)` answered True for a `__len__` returning
// -1, and `len` printed 2**70.
std::optional<Value>
ModuleEmitter::tryEmitClassLength(const parser::Node &anchor, Value receiver) {
  std::optional<Value> count = tryEmitClassDunder(anchor, receiver, "__len__");
  if (!count)
    return std::nullopt;
  mlir::Type intType = types.intType();
  Value counted = coerceValue(*count, intType, anchor);
  CallInferenceResult inference =
      types.inferMethodCallWithEvidence(intType, "__ly_length__", {});
  if (!inference)
    return counted;
  Value posPack = emitPack({});
  Value namePack = emitPack({});
  Value valuePack = emitPack({});
  auto call = py::CallOp::create(
      builder, loc(anchor), mlir::TypeRange{inference.resultType},
      callProtocolFor(inference), counted.value, posPack.value,
      namePack.value, valuePack.value);
  call->setAttr("ly.bound_method", builder.getStringAttr("__ly_length__"));
  return Value{call.getResults().front(), inference.resultType};
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
  if (std::optional<Value> count = tryEmitClassLength(expr, input))
    return *count;
  // ⭐ A UNION COUNTS BY TAG, like it adds by tag. `def f(x: "list[int] | str")`
  // is an ordinary Python signature and `len(x)` in its body was refused for a
  // question both members answer.
  if (auto inputUnion =
          mlir::dyn_cast_if_present<py::UnionType>(types.widenLiteral(input.type))) {
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
  }
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
  // ⭐ id() answers only where `is` does: a value type's identity depends on
  // whether it was boxed (an int read twice from a lane is two boxes), and a
  // function value is built at each reference -- `id(x) == id(x)` would be
  // False. Same refusal, same reason, as the `is` operator's.
  if (callsUnshadowedBuiltin(calleeNode, "id")) {
    const auto *idArgs = ast::nodeList(expr, "args");
    if (idArgs && idArgs->size() == 1 && idArgs->front() &&
        idArgs->front()->kind != "Starred") {
      mlir::Type argType =
          types.widenLiteral(types.inferExpr(idArgs->front().get()));
      bool valueType =
          argType == types.intType() || argType == types.floatType() ||
          argType == types.strType() ||
          argType == types.contract("builtins.bytes") ||
          argType == types.contract("builtins.complex");
      // None is one entity (its box word is 0) and an empty literal builds a
      // container, so both have the identity `is` gives them.
      bool reference = mlir::isa_and_present<py::ContractType>(argType) ||
                       argType == types.none() ||
                       isEmptyContainerExpression(idArgs->front().get());
      if (valueType) {
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, expr.range.start,
            "id() of an int/str/float/bytes/complex is rejected (identity of "
            "value types is an implementation detail, as `is` on them is); "
            "compare with `==` instead"});
        return emitNone(expr);
      }
      if (!reference) {
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, expr.range.start,
            "id() needs an argument of one reference type; a function value "
            "is built at each reference and a union is not one type, so this "
            "argument has no stable identity here"});
        return emitNone(expr);
      }
      // None's identity key is 0 (LyObject_IdentityKey reads the None
      // object's word as it), which is what id() of an `object` holding None
      // answers, so a None-typed argument is evaluated and answers the same
      // without building a box.
      if (argType == types.none()) {
        emitExpr(idArgs->front().get());
        parser::NodePtr zero = synth::intConstant(0, expr.range);
        return emitExpr(zero.get());
      }
    }
    return std::nullopt;
  }
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
      // A union of bool and int is the int it is, as an argument
      // (`emitCallableDispatch`).
      if (mlir::isa<py::UnionType>(ndigits.type) &&
          boolKeptAsInt(types.intType(), ndigits.type))
        ndigits = rungsAsInt(ndigits, expr);
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
      if (subclassOverridesMethod(argumentContract.getContractName(), wanted) &&
          !builtinTextFallbacks.contains(&expr)) {
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
      KeywordArguments emptyKeywords;
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
      //
      // ⭐ AND `None`, for the same reason and through the same door. A None
      // resolves a manifest `__repr__` here, so the ladder emitted a py.repr
      // on a value that expands to NO physical values -- "types.NoneType
      // runtime object has no physical header value", out of the LOWERING, for
      // `print(repr(None))`. `f"{v!r}"` renders it, `str(v)` renders it and
      // `print(v)` renders it, all three through `emitConversionValue`; this
      // was the one door that had its own copy.
      if (mlir::isa<py::UnionType, py::TypeType>(argumentType) ||
          isNoneTypeLike(argumentType))
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
        // ⭐ THE VALUE MAY BE A LOWER RUNG THAN THE TYPE INFERRED FOR IT. A
        // direct call specialized at its argument's rung returns what that
        // body returns -- `ident(True)` is a bool, inferred int from the
        // declaration -- and int's `__repr__` over a bool reached the lowering
        // as "evidence is not a manifest candidate for receiver type bool".
        if (mlir::Type actual = types.widenLiteral(argument.value.getType());
            actual != argumentType && isNumericPrimitiveContract(actual) &&
            isNumericPrimitiveContract(argumentType))
          if (CallInferenceResult own =
                  types.inferMethodCallWithEvidence(actual, "__repr__", {}))
            inference = own;
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
